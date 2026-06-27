"""``gps`` command-line entrypoint.

Subcommands:

* ``gps phase0``  -- run the CPU Phase-0 synthetic experiments (no GPU).
* ``gps info``    -- print environment + which backends are importable.

Kept tiny and dependency-free so ``gps phase0`` runs on a bare Python.
"""

from __future__ import annotations

import argparse
import sys

from gps import __version__
from gps.latent.base import InjectionKind


def _cmd_phase0(args: argparse.Namespace) -> int:
    from gps.experiments.phase0 import run_phase0

    kinds = (
        ["tilt", "time_pressure", "fatigue", "hysteresis"]
        if args.player == "all"
        else [args.player]
    )
    kind = (
        InjectionKind.VERBAL
        if args.injection == "verbal"
        else InjectionKind.HIDDEN
    )
    ok = True
    for k in kinds:
        res = run_phase0(
            player_kind=k,
            n_games=args.games,
            seed=args.seed,
            injector_kind=kind,
        )
        print(res.summary())
        # The load-bearing Phase-0 claim is that knowing the dynamic state
        # helps (oracle > static). The history vs. heuristic column is the
        # Milestone-A control (E-A1): does *accumulating* the same features
        # into an evolving latent beat consuming them memorylessly? It is
        # reported, not asserted -- the untrained heuristic is not expected
        # to win it; the trained injector must (E-C2, GPU). See
        # documents/milestone_a.md.
        ok = ok and res.oracle_helps
    if not ok:
        print(
            "\nNote: oracle did not beat static on every mechanism; "
            "inspect per-mechanism output above."
        )
    return 0


def _cmd_train_ea1(args: argparse.Namespace) -> int:
    from gps.experiments.ea1 import run_ea1, run_ea1_capacity_sweep

    common = dict(
        n_players=args.players,
        n_games=args.games,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        bootstrap_n=args.bootstrap_n,
    )
    if args.capacity_sweep:
        print("=== E-A1 capacity sweep (B at 1x/2x/4x D width) ===")
        for mult, res in run_ea1_capacity_sweep(
            latent_dim=args.latent_dim,
            n_players=args.players,
            n_games=args.games,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            bootstrap_n=args.bootstrap_n,
        ).items():
            print(f"\n--- B width = {mult}x D ---")
            print(res.summary())
        return 0
    res = run_ea1(b_latent_dim=args.b_latent_dim, **common)
    print(res.summary())
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    import os

    print(f"grounded-player-sim {__version__}")
    print(f"python: {sys.version.split()[0]}")
    for name, mod in [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("sglang", "sglang"),
        ("slime", "slime"),
        ("wandb", "wandb"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("python-chess", "chess"),
    ]:
        try:
            __import__(mod)
            status = "available"
        except ImportError:
            status = "not installed"
        print(f"  {name:14s} {status}")

    # Backend + tracking policy at a glance (see gps.backends / gps.tracking).
    print("policy:")
    print("  LLM inference -> sglang ; LLM training -> slime+sglang")
    key_set = bool(os.environ.get("WANDB_API_KEY", "").strip())
    print(
        f"  WANDB_API_KEY  {'set' if key_set else 'NOT set'} "
        f"(required for training runs)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gps", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"gps {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p0 = sub.add_parser("phase0", help="run CPU Phase-0 experiments")
    p0.add_argument(
        "--player",
        default="all",
        choices=["all", "tilt", "time_pressure", "fatigue", "hysteresis"],
    )
    p0.add_argument("--games", type=int, default=24)
    p0.add_argument("--seed", type=int, default=0)
    p0.add_argument(
        "--injection", default="hidden", choices=["hidden", "verbal"]
    )
    p0.set_defaults(func=_cmd_phase0)

    ea1 = sub.add_parser(
        "train-ea1",
        help="train arm D vs B (Milestone A); needs WANDB_API_KEY",
    )
    ea1.add_argument("--players", type=int, default=32)
    ea1.add_argument("--games", type=int, default=20)
    ea1.add_argument("--latent-dim", type=int, default=16)
    ea1.add_argument("--epochs", type=int, default=400)
    ea1.add_argument("--lr", type=float, default=1e-2)
    ea1.add_argument("--seed", type=int, default=0)
    ea1.add_argument("--bootstrap-n", type=int, default=2000)
    ea1.add_argument(
        "--b-latent-dim",
        type=int,
        default=None,
        help="give arm B a different width (capacity check); default=D",
    )
    ea1.add_argument(
        "--capacity-sweep",
        action="store_true",
        help="run B at 1x/2x/4x D width instead of a single run",
    )
    ea1.set_defaults(func=_cmd_train_ea1)

    info = sub.add_parser("info", help="print environment + backend status")
    info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
