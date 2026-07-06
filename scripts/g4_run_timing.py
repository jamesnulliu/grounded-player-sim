"""G4: does the per-individual evolving latent add think-time value OVER a
released-SOTA-informed baseline? Multi-seed, on a Maia-cached real cohort.

Compares three timing baselines B (all Elo+clock), each vs B+z:
  * position_aware   -- B + branching-factor complexity (the E-C6 proxy)
  * maia_complexity  -- B + Maia-2's move-entropy (a RELEASED SOTA's learned
                        position difficulty) -- the G4 headline baseline
  * (optional) external / pure_external if an Allie think-time prediction is
    cached as ``external_time_pred``.

For each, reports (B+z)-B pooled per-player mean, 95% bootstrap CI, P(<0), and
the baseline's Spearman (vs the ChessMimic r=0.41 yardstick). A negative,
significant (B+z)-B on the maia_complexity baseline is the strong claim: the
latent adds value beyond Elo, clock, AND a released model's difficulty signal.

Usage:
  python scripts/g4_run_timing.py CACHED.jsonl.gz [n_seeds] [epochs]
"""

from __future__ import annotations

import statistics as st
import sys

from gps.data.store import load_dataset
from gps.experiments.ec import run_timing_vs_aggregate


def _pool(ds, *, seeds, epochs, **kw):
    """Pool (B+z)-B per player across seeds, then bootstrap over players."""
    import numpy as np

    from gps.eval.bootstrap import bootstrap_ci

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
    return {
        "add_mean": float(np.mean(pooled)),
        "ci_low": ci.low,
        "ci_high": ci.high,
        "p_below": ci.p_below_zero,
        "n_players": len(pooled),
        "spearman": float(np.mean(spears)),
        "b_nll": float(np.mean(b_nll)),
        "bz_nll": float(np.mean(bz_nll)),
    }


def main() -> None:
    path = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    ds = load_dataset(path)
    print(
        f"[g4-timing] {len(ds.trajectories)} players, {seeds} seeds, "
        f"{epochs} epochs\n",
        flush=True,
    )

    for label, kw in [
        ("B=Elo+clock+branching (E-C6 proxy)", {"position_aware": True}),
        ("B=Elo+clock+MAIA-entropy (released difficulty)",
         {"maia_complexity": True}),
        ("B=Elo+clock+branching+MAIA (STRONGEST combined)",
         {"position_aware": True, "maia_complexity": True}),
    ]:
        res = _pool(ds, seeds=seeds, epochs=epochs, **kw)
        verdict = "LATENT ADDS VALUE" if res["p_below"] >= 0.975 else "ns"
        print(
            f"{label}\n"
            f"  (B+z)-B = {res['add_mean']:+.4f}  95% CI "
            f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]  "
            f"P(<0)={res['p_below']:.3f}  n={res['n_players']}  -> {verdict}\n"
            f"  baseline B NLL={res['b_nll']:.4f} -> B+z={res['bz_nll']:.4f} "
            f"| baseline Spearman={res['spearman']:.3f} (ChessMimic 0.41)\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
