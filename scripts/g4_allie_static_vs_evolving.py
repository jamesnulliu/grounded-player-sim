#!/usr/bin/env python
"""Compare locked-Allie + static-individual vs locked-Allie + evolving.

Each cohort/seed cell trains both latent controls under the existing G4
protocol, scores the same held-out sessions, and saves paired per-player NLLs.
Aggregation averages each player's evolving-minus-static difference over seeds
before bootstrapping players.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from gps.data.store import load_dataset
from gps.eval.bootstrap import bootstrap_ci
from gps.experiments.ec import run_timing_vs_aggregate

DEFAULT_OUT_DIR = Path("runs/g4-allie-static-vs-evolving")
DEFAULT_SUMMARY = Path("results/g4_allie_static_vs_evolving.json")
DEFAULT_COHORTS = {
    "2017-04": Path(
        "/project2/xiangren_1715/liuyanch/g4_data/"
        "ec2017/dataset_allie.jsonl.gz"
    ),
    "2019-07": Path(
        "/project2/xiangren_1715/liuyanch/g4_data/"
        "ec2019/dataset_allie.jsonl.gz"
    ),
    "2021-06": Path(
        "/project2/xiangren_1715/liuyanch/g4_data/"
        "ec2021/dataset_allie.jsonl.gz"
    ),
}
FROZEN_SEEDS = (0, 1, 2, 3, 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _ci_dict(ci) -> dict:
    return {
        "point": float(ci.point),
        "low": float(ci.low),
        "high": float(ci.high),
        "p_below_zero": float(ci.p_below_zero),
        "n_units": int(ci.n_units),
        "confidence": float(ci.confidence),
    }


def _cell_path(out_dir: Path, cohort: str, seed: int) -> Path:
    return out_dir / cohort / f"seed-{seed}.json"


def _parse_cells(values: list[str] | None) -> list[tuple[str, int]]:
    if not values:
        return [
            (cohort, seed)
            for cohort in DEFAULT_COHORTS
            for seed in FROZEN_SEEDS
        ]
    cells = []
    for value in values:
        try:
            cohort, raw_seed = value.rsplit(":", 1)
            seed = int(raw_seed)
        except ValueError as error:
            raise ValueError(
                f"invalid --cell {value!r}; expected COHORT:SEED"
            ) from error
        if cohort not in DEFAULT_COHORTS:
            raise ValueError(f"unknown cohort {cohort!r}")
        if seed not in FROZEN_SEEDS:
            raise ValueError(f"seed {seed} is outside {FROZEN_SEEDS}")
        cells.append((cohort, seed))
    return cells


def _run_cell(cohort: str, seed: int, path: Path, epochs: int) -> dict:
    dataset = load_dataset(str(path))
    common = {
        "split_mode": "session",
        "epochs": epochs,
        "seed": seed,
        "bootstrap_n": 2000,
        "pure_external": True,
    }
    started = time.monotonic()
    static = run_timing_vs_aggregate(
        dataset, latent_control="static", **common
    )
    evolving = run_timing_vs_aggregate(
        dataset, latent_control="evolving", **common
    )
    if static.player_ids != evolving.player_ids:
        raise ValueError(f"{cohort} seed {seed}: player order changed")
    diffs = [
        float(evolving_nll) - float(static_nll)
        for evolving_nll, static_nll in zip(
            evolving.b4z_per_player or (), static.b4z_per_player or ()
        )
    ]
    if len(diffs) != len(static.player_ids or ()):
        raise ValueError(f"{cohort} seed {seed}: incomplete player scores")
    ci = bootstrap_ci(diffs, n_resamples=2000, seed=seed)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort,
        "seed": seed,
        "dataset_path": str(path),
        "dataset_sha256": _sha256(path),
        "protocol": {
            "split_mode": "session",
            "epochs": epochs,
            "latent_dim": 16,
            "hidden_dim": 64,
            "timing_lambda": 0.5,
            "pure_external": True,
            "allie_offset": "locked_with_global_intercept",
        },
        "players": static.player_ids,
        "static_nll": static.b4z_per_player,
        "evolving_nll": evolving.b4z_per_player,
        "evolving_minus_static": diffs,
        "ci": _ci_dict(ci),
        "means": {
            "allie": float(static.b4_nll),
            "allie_plus_static": float(static.b4z_nll),
            "allie_plus_evolving": float(evolving.b4z_nll),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }


def _read_cells(out_dir: Path) -> list[dict]:
    cells = []
    missing = []
    for cohort in DEFAULT_COHORTS:
        for seed in FROZEN_SEEDS:
            path = _cell_path(out_dir, cohort, seed)
            if not path.exists():
                missing.append(str(path))
                continue
            cell = json.loads(path.read_text())
            if cell.get("schema_version") != 1:
                raise ValueError(f"{path}: unsupported schema")
            if cell.get("cohort") != cohort or cell.get("seed") != seed:
                raise ValueError(f"{path}: cell identity mismatch")
            cells.append(cell)
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Allie comparison cells missing:\n" + listing)
    return cells


def _pool_cells(cells: list[dict]) -> dict:
    summaries = []
    all_player_diffs: dict[str, list[float]] = defaultdict(list)
    n_cohort_players = 0
    for cohort in DEFAULT_COHORTS:
        cohort_cells = [cell for cell in cells if cell["cohort"] == cohort]
        hashes = {cell["dataset_sha256"] for cell in cohort_cells}
        orders = {tuple(cell["players"]) for cell in cohort_cells}
        if len(hashes) != 1 or len(orders) != 1:
            raise ValueError(f"{cohort}: seed cells use different data")
        player_diffs = []
        players = cohort_cells[0]["players"]
        for index, player in enumerate(players):
            diff = statistics.mean(
                cell["evolving_minus_static"][index]
                for cell in cohort_cells
            )
            player_diffs.append(diff)
            all_player_diffs[player].append(diff)
            n_cohort_players += 1
        ci = bootstrap_ci(player_diffs, n_resamples=5000, seed=0)
        summaries.append(
            {
                "cohort": cohort,
                "dataset_sha256": next(iter(hashes)),
                "n_players": len(players),
                "seeds": list(FROZEN_SEEDS),
                "evolving_minus_static": _ci_dict(ci),
                "mean_nll": {
                    key: statistics.mean(
                        cell["means"][key] for cell in cohort_cells
                    )
                    for key in (
                        "allie",
                        "allie_plus_static",
                        "allie_plus_evolving",
                    )
                },
                "seed_results": [
                    {"seed": cell["seed"], "ci": cell["ci"]}
                    for cell in sorted(
                        cohort_cells, key=lambda value: value["seed"]
                    )
                ],
            }
        )
    unique_player_diffs = [
        statistics.mean(values) for values in all_player_diffs.values()
    ]
    overall_ci = bootstrap_ci(
        unique_player_diffs, n_resamples=5000, seed=0
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": "Allie+evolving minus Allie+static-individual",
        "protocol": {
            "cohorts": list(DEFAULT_COHORTS),
            "seeds": list(FROZEN_SEEDS),
            "split_mode": "session",
            "epochs": 15,
            "pure_external": True,
        },
        "cohorts": summaries,
        "overall": {
            "n_unique_players": len(unique_player_diffs),
            "n_cohort_players": n_cohort_players,
            "evolving_minus_static": _ci_dict(overall_ci),
        },
    }


def _print_summary(summary: dict, path: Path) -> None:
    for item in summary["cohorts"]:
        ci = item["evolving_minus_static"]
        print(
            f"[allie-static] {item['cohort']} evolving-static="
            f"{ci['point']:+.4f} [{ci['low']:+.4f},{ci['high']:+.4f}] "
            f"P(<0)={ci['p_below_zero']:.3f}"
        )
    ci = summary["overall"]["evolving_minus_static"]
    print(
        f"[allie-static] overall evolving-static={ci['point']:+.4f} "
        f"[{ci['low']:+.4f},{ci['high']:+.4f}] "
        f"P(<0)={ci['p_below_zero']:.3f} -> {path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        help="COHORT:SEED; repeat to partition work (default: all cells)",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.epochs != 15:
        raise ValueError("the frozen comparison uses exactly 15 epochs")
    if args.aggregate_only:
        summary = _pool_cells(_read_cells(args.out_dir))
        _write_json_atomic(args.summary_out, summary)
        _print_summary(summary, args.summary_out)
        return 0

    os.environ.setdefault("WANDB_PROJECT", "gps-g4-allie-static-vs-evolving")
    for cohort, seed in _parse_cells(args.cell):
        source = DEFAULT_COHORTS[cohort]
        if not source.exists():
            raise FileNotFoundError(source)
        target = _cell_path(args.out_dir, cohort, seed)
        if target.exists() and not args.force:
            cell = json.loads(target.read_text())
            if cell.get("dataset_sha256") != _sha256(source):
                raise ValueError(f"{target}: dataset changed")
            print(f"[skip] {target}")
            continue
        cell = _run_cell(cohort, seed, source, args.epochs)
        _write_json_atomic(target, cell)
        ci = cell["ci"]
        print(
            f"[cell] {cohort} seed={seed} evolving-static="
            f"{ci['point']:+.4f} -> {target}",
            flush=True,
        )
    try:
        summary = _pool_cells(_read_cells(args.out_dir))
    except FileNotFoundError:
        print("[pending] run remaining cells before pooling")
    else:
        _write_json_atomic(args.summary_out, summary)
        _print_summary(summary, args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
