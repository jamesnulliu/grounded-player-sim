"""G4: cache a released SOTA (Maia-2) per-decision prediction onto a dataset.

For every decision we run Maia-2 (CSSLab, released, human-move SOTA) and cache:
  * ``maia_entropy``       -- entropy of Maia's legal-move distribution, a
                              released-model position-difficulty signal used as
                              the TIMING baseline's complexity feature
                              (``run_timing_vs_aggregate(maia_complexity=True)``).
  * ``maia_move_logprob``  -- log P_maia(observed move), the MOVE-channel
                              baseline (does the latent beat Maia's moves?).
  * ``maia_top1``          -- Maia's top-move probability (context/logging).

Usage:
  python scripts/g4_cache_maia.py IN.jsonl.gz OUT.jsonl.gz [blitz|rapid]

The augmented dataset re-saves from committed code (``save_dataset``), so the
G4 runs reproduce from disk. GPU recommended (1 A100 is plenty).
"""

from __future__ import annotations

import math
import sys
import time

from gps.data.store import load_dataset, save_dataset


def main() -> None:
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    maia_type = sys.argv[3] if len(sys.argv) > 3 else "blitz"
    save_root = "/project2/xiangren_1715/liuyanch/g4_data/maia2_models"

    from maia2 import inference as I
    from maia2 import model as M

    print(f"[g4-cache] loading Maia-2 {maia_type} ...", flush=True)
    mdl = M.from_pretrained(type=maia_type, device="gpu", save_root=save_root)
    prep = I.prepare()

    ds = load_dataset(in_path)
    n_dec = sum(len(t.decisions) for t in ds.trajectories)
    print(
        f"[g4-cache] {len(ds.trajectories)} players, {n_dec} decisions",
        flush=True,
    )

    done = 0
    t0 = time.time()
    miss = 0  # observed move not in Maia's legal dict (parity/promotion edge)
    for traj in ds.trajectories:
        for dp, obs in zip(traj.decisions, traj.observations):
            elo_self = int(dp.context.get("player_elo") or 1500)
            elo_oppo = int(dp.context.get("opponent_elo") or elo_self)
            move_probs, win_prob = I.inference_each(
                mdl, prep, dp.state, elo_self, elo_oppo
            )
            ps = [p for p in move_probs.values() if p > 0]
            ent = -sum(p * math.log(p) for p in ps) if ps else 0.0
            pm = move_probs.get(obs.move)
            if pm is None or pm <= 0:
                miss += 1
                pm = 1e-6
            dp.context["maia_entropy"] = float(ent)
            dp.context["maia_move_logprob"] = float(math.log(pm))
            dp.context["maia_top1"] = float(ps[0]) if ps else 0.0
            dp.context["maia_win_prob"] = float(win_prob)
            done += 1
            if done % 5000 == 0:
                rate = done / (time.time() - t0)
                print(
                    f"[g4-cache] {done}/{n_dec} ({rate:.0f}/s, miss={miss})",
                    flush=True,
                )

    save_dataset(ds, out_path)
    print(
        f"[g4-cache] wrote {out_path} ({done} decisions, "
        f"{miss} observed-move misses = {100 * miss / max(done, 1):.1f}%) "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
