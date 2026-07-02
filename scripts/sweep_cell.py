#!/usr/bin/env python
"""Run ONE cell of the E-C scaling grid (a GPU) and dump per-player NLLs.

One cell = (dataset, seed, trunk, split, timing_model). Designed to be fanned
out one-per-GPU by a SLURM array or ``xargs -P`` -- each process pins itself to
the GPU in ``CUDA_VISIBLE_DEVICES`` and writes a self-describing result JSON.
``scripts/sweep_pool.py`` then pools the JSONs across seeds.

Example (single GPU)::

    CUDA_VISIBLE_DEVICES=0 python scripts/sweep_cell.py \
        --dataset data/2017-04-blitz.jsonl.gz --seed 0 --trunk conv \
        --n-players 1000 --epochs 30 --out runs/2017_conv_s0.json
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cohort", default=None, help="label (default: basename)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trunk", default="mlp", choices=["mlp", "conv"])
    ap.add_argument("--split-mode", default="session")
    ap.add_argument("--control", default="memoryless")
    ap.add_argument("--timing-model", default="lognormal")
    ap.add_argument("--n-players", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--max-decisions",
        type=int,
        default=0,
        help="truncate each player to their first N decisions (0 = no cap); "
        "bounds the padded [T,B] tensor so it fits GPU memory",
    )
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")

    from gps.data.store import load_dataset
    from gps.experiments.ec import run_ec
    from gps.train.base import TrajectoryDataset

    ds = load_dataset(args.dataset)
    if args.n_players and args.n_players < len(ds.trajectories):
        ds = TrajectoryDataset(ds.trajectories[: args.n_players])
    if args.max_decisions:
        from gps.train.base import Trajectory

        n = args.max_decisions
        ds = TrajectoryDataset(
            [
                Trajectory(t.player_id, t.decisions[:n], t.observations[:n])
                for t in ds.trajectories
            ]
        )

    t0 = time.time()
    r = run_ec(
        ds,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        split_mode=args.split_mode,
        control=args.control,
        timing_model=args.timing_model,
        trunk=args.trunk,
    )
    out = {
        "cohort": args.cohort or os.path.basename(args.dataset),
        "dataset": args.dataset,
        "seed": args.seed,
        "trunk": args.trunk,
        "control": args.control,
        "split_mode": args.split_mode,
        "timing_model": args.timing_model,
        "n_players": len(ds.trajectories),
        "players": [t.player_id for t in ds.trajectories],
        "move_d": r.d_per_player,
        "move_b": r.b_per_player,
        "timing_d": getattr(r, "d_timing_per_player", []),
        "timing_b": getattr(r, "b_timing_per_player", []),
        "wall_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    move = r.ci.point
    print(
        f"[cell] {out['cohort']} trunk={args.trunk} seed={args.seed} "
        f"n={out['n_players']} move D-B={move:+.4f} "
        f"({out['wall_seconds']}s) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
