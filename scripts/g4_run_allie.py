"""G4 (airtight): does the evolving latent add think-time value OVER Allie's
ACTUAL released think-time prediction? Multi-seed, on an Allie-cached cohort.

Allie (ICLR'25) is a released model with a real think-time head -- unlike
Maia-2 (move-only, whose entropy is only a difficulty PROXY). We cached its
per-decision prediction as context['external_time_pred']
(scripts/g4_cache_allie.py). Three baselines B, each vs B+z:
  maia_combined  -- Elo+clock+branching+Maia-entropy (the previous strongest)
  allie_feature  -- Elo+clock + Allie's predicted think-time as a fitted
                    feature (external_pred): baseline >= Allie + our aggregate
  allie_locked   -- Allie's prediction as a LOCKED offset, latent the only
                    extra predictor (pure_external): "Allie" vs "Allie + z"

A negative, significant (B+z)-B on allie_locked is the strongest claim: the
per-individual evolving latent adds think-time value even over a released model
that already predicts think-time well (Allie per-player Spearman ~0.4-0.6).

Usage: python scripts/g4_run_allie.py ALLIE_CACHED.jsonl.gz [n_seeds] [epochs]
"""

from __future__ import annotations

import statistics as st
import sys

import numpy as np

from gps.data.store import load_dataset
from gps.eval.bootstrap import bootstrap_ci
from gps.experiments.ec import run_timing_vs_aggregate


def _pool(ds, *, seeds, epochs, **kw):
    per_player: dict[int, list[float]] = {}
    spears, b_nll, bz_nll = [], [], []
    for seed in range(seeds):
        r = run_timing_vs_aggregate(
            ds, split_mode="session", epochs=epochs, seed=seed,
            bootstrap_n=2000, **kw,
        )
        for i, d in enumerate(r.add_per_player or []):
            per_player.setdefault(i, []).append(d)
        spears.append(r.b4z_spearman)
        b_nll.append(r.b4_nll)
        bz_nll.append(r.b4z_nll)
    pooled = [st.mean(v) for v in per_player.values()]
    ci = bootstrap_ci(pooled, n_resamples=4000, seed=0)
    return dict(
        add=float(np.mean(pooled)), lo=ci.low, hi=ci.high,
        p=ci.p_below_zero, n=len(pooled), spear=float(np.mean(spears)),
        b=float(np.mean(b_nll)), bz=float(np.mean(bz_nll)),
    )


def main() -> None:
    path = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    ds = load_dataset(path)
    print(f"[g4-allie] {len(ds.trajectories)} players, {seeds} seeds\n")
    for label, kw in [
        ("B=Elo+clock+branching+MAIA (prev strongest)",
         {"position_aware": True, "maia_complexity": True}),
        ("B=Elo+clock+ALLIE-pred (fitted feature)",
         {"external_pred": True}),
        ("B=ALLIE prediction LOCKED (Allie vs Allie+z)",
         {"pure_external": True}),
    ]:
        r = _pool(ds, seeds=seeds, epochs=epochs, **kw)
        verdict = "LATENT ADDS VALUE" if r["p"] >= 0.975 else "ns"
        print(
            f"{label}\n"
            f"  (B+z)-B = {r['add']:+.4f}  CI [{r['lo']:+.4f},{r['hi']:+.4f}]"
            f"  P(<0)={r['p']:.3f}  n={r['n']}  -> {verdict}\n"
            f"  B NLL={r['b']:.4f} -> B+z={r['bz']:.4f} | "
            f"baseline Spearman={r['spear']:.3f} (ChessMimic 0.41)\n"
        )


if __name__ == "__main__":
    main()
