"""``gps`` command-line entrypoint.

Subcommands:

* ``gps phase0``    -- run the CPU Phase-0 synthetic experiments (no GPU).
* ``gps ingest``    -- parse a Lichess archive into a persisted E-C dataset.
* ``gps train-ec``  -- E-C2 (dynamic vs memoryless) on a persisted dataset.
* ``gps kt``        -- RQ5/Milestone-F on knowledge tracing (synth or real).
* ``gps info``      -- print environment + which backends are importable.

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


def _cmd_train_ec(args: argparse.Namespace) -> int:
    from gps.data.store import load_dataset
    from gps.experiments.ec import run_ec

    dataset = load_dataset(args.dataset)
    if len(dataset) < 2:
        print(
            f"E-C needs >=2 players for a bootstrap over players; "
            f"{args.dataset} has {len(dataset)}."
        )
        return 1
    res = run_ec(
        dataset,
        train_frac=args.train_frac,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        b_latent_dim=args.b_latent_dim,
        bootstrap_n=args.bootstrap_n,
        batch_size=args.batch_size,
        split_mode=args.split,
        control=args.control,
    )
    print(res.summary())
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from gps.data.ingest import run_ingest

    run_ingest(
        args.archive,
        args.out,
        speed=None if args.speed == "all" else args.speed,
        min_games=args.min_games,
        min_sessions=args.min_sessions,
        max_players=args.max_players,
        gap_threshold_seconds=args.gap_threshold,
        workers=args.workers,
        batch_size=args.batch_size,
        max_games=args.max_games,
        max_games_per_player=args.max_games_per_player,
    )
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


def _cmd_kt(args: argparse.Namespace) -> int:
    """RQ5 / Milestone-F on knowledge tracing.

    Synthetic cohort by default; pass ``--data <csv>`` to run on a real
    preprocessed KT export (``gps.data.kt_csv.load_kt_csv``). Training run, so
    it needs ``WANDB_API_KEY`` (or ``WANDB_MODE=offline``).
    """
    from gps.experiments.kt import build_kt_dataset, run_kt, run_population

    if args.data:
        from gps.data.kt_csv import load_kt_csv

        ds = load_kt_csv(
            args.data,
            n_students=args.n_students,
            min_responses=args.min_responses,
            train_frac=args.train_frac,
            response_time_col=args.response_time_col,
        )
        print(
            f"real KT ({args.data}): {len(ds.trajectories)} students, "
            f"{sum(len(t.decisions) for t in ds.trajectories)} responses"
        )
    else:
        ds = build_kt_dataset(n_students=args.n_students, seed=args.seed)
        print(f"synthetic KT: {len(ds.trajectories)} students")

    r = run_kt(
        ds,
        train_frac=args.train_frac,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(r.summary())
    if args.population:
        p = run_population(
            ds,
            train_frac=args.train_frac,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(p.summary())
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

    ec = sub.add_parser(
        "train-ec",
        help="E-C2 on a persisted chess dataset (D vs memoryless); needs "
        "WANDB_API_KEY",
    )
    ec.add_argument(
        "dataset", help="path to a persisted dataset.jsonl(.gz) (gps ingest)"
    )
    ec.add_argument("--train-frac", type=float, default=0.7)
    ec.add_argument(
        "--split",
        default="fraction",
        choices=["fraction", "session"],
        help="temporal split: move-fraction (E-C2) or later-sessions (E-C3)",
    )
    ec.add_argument(
        "--control",
        default="memoryless",
        choices=["memoryless", "static"],
        help="arm B: memoryless history-conditioned (E-C2/3) or static "
        "per-player embedding (E-C1)",
    )
    ec.add_argument("--latent-dim", type=int, default=16)
    ec.add_argument("--hidden-dim", type=int, default=64)
    ec.add_argument("--epochs", type=int, default=300)
    ec.add_argument("--lr", type=float, default=1e-2)
    ec.add_argument("--seed", type=int, default=0)
    ec.add_argument("--bootstrap-n", type=int, default=2000)
    ec.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="players per minibatch (scales large cohorts past full-batch)",
    )
    ec.add_argument(
        "--b-latent-dim",
        type=int,
        default=None,
        help="give arm B a wider latent (capacity check); default=D",
    )
    ec.set_defaults(func=_cmd_train_ec)

    ing = sub.add_parser(
        "ingest",
        help="parse a Lichess .pgn(.zst) archive into a persisted dataset",
    )
    ing.add_argument("archive", help="path to a .pgn or .pgn.zst archive")
    ing.add_argument(
        "--out",
        required=True,
        help="output directory (dataset.jsonl.gz + manifest.json)",
    )
    ing.add_argument(
        "--speed",
        default="blitz",
        choices=[
            "ultrabullet",
            "bullet",
            "blitz",
            "rapid",
            "classical",
            "correspondence",
            "all",
        ],
        help="single time-control class to keep (default blitz; 'all' mixes)",
    )
    ing.add_argument("--min-games", type=int, default=50)
    ing.add_argument("--min-sessions", type=int, default=3)
    ing.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="cap the cohort to the top-N players by game count",
    )
    ing.add_argument(
        "--gap-threshold",
        type=float,
        default=1800.0,
        help="session gap threshold in seconds (default 1800 = 30 min)",
    )
    ing.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes for the pass-2 parse (the bottleneck); 1 = serial",
    )
    ing.add_argument("--batch-size", type=int, default=512)
    ing.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="cap games read per pass (smoke runs)",
    )
    ing.add_argument(
        "--max-games-per-player",
        type=int,
        default=None,
        help="cap each player to their earliest N games (bounds + equalizes "
        "trajectory length for the full-batch trainer)",
    )
    ing.set_defaults(func=_cmd_ingest)

    kt = sub.add_parser(
        "kt",
        help="RQ5/Milestone-F on knowledge tracing (synthetic or real --data);"
        " needs WANDB_API_KEY or WANDB_MODE=offline",
    )
    kt.add_argument(
        "--data",
        default=None,
        help="path to a preprocessed KT CSV (real); omit for synthetic",
    )
    kt.add_argument("--n-students", type=int, default=48)
    kt.add_argument("--min-responses", type=int, default=50)
    kt.add_argument("--train-frac", type=float, default=0.7)
    kt.add_argument(
        "--response-time-col",
        type=int,
        default=None,
        help="0-indexed column with a per-row response time in ms (real "
        "timing channel); omit for the correctness-only 5-column format",
    )
    kt.add_argument("--latent-dim", type=int, default=16)
    kt.add_argument("--hidden-dim", type=int, default=32)
    kt.add_argument("--epochs", type=int, default=40)
    kt.add_argument("--seed", type=int, default=0)
    kt.add_argument(
        "--population",
        action="store_true",
        help="also run the Milestone-F population-heterogeneity eval",
    )
    kt.set_defaults(func=_cmd_kt)

    info = sub.add_parser("info", help="print environment + backend status")
    info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
