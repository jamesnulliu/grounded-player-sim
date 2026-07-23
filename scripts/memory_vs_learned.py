#!/usr/bin/env python
"""Structured-memory control vs learned latents over locked Allie (G5).

The reviewer objection this answers: "an evolving learned latent is just a
memory -- a structured memory module storing running summary statistics of the
player's history would do the same, so distinguish yourself from memory."

Protocol: identical to the G4 Allie static-vs-evolving comparison
(`scripts/g4_allie_static_vs_evolving.py`) -- same cohorts, same session
split, same locked Allie offset + global intercept, same lstsq readout --
with a third arm, ``latent_control="memory"`` (`gps.experiments.ec.
_memory_latents`): hand-designed causal running statistics of the player's
history (running/EWMA mean of log think-time, residuals vs Allie's own
prediction, premove rate, event memory, current engineered features). The
memory arm is deliberately *stronger-input* than the learned arms (it reads
raw past think-times and Allie's past errors, which the GRU never sees) and
involves no gradient training, so it is exactly "what a memory module could
store, optimally linearly read out".

The memory arm is deterministic given the dataset (no training stochasticity;
the lstsq fit is exact), so it runs once per cohort and pairs per player
against each cached static/evolving seed cell from
``runs/g4-allie-static-vs-evolving/``. Aggregation mirrors G4: average each
player's paired difference over seeds, bootstrap players; the cross-cohort
summary averages repeated usernames first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from gps.data.store import load_dataset
from gps.eval.bootstrap import bootstrap_ci
from gps.experiments.ec import MEMORY_FEATURE_NAMES, run_timing_vs_aggregate

G4_CELL_DIR = Path("runs/g4-allie-static-vs-evolving")
DEFAULT_OUT_DIR = Path("runs/memory-vs-learned")
DEFAULT_SUMMARY = Path("results/memory_baseline.json")
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


def _load_g4_cells(cohort: str, dataset_sha: str) -> list[dict]:
    cells = []
    for seed in FROZEN_SEEDS:
        path = G4_CELL_DIR / cohort / f"seed-{seed}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run scripts/g4_allie_static_vs_evolving.py"
                " first (the memory comparison pairs against its cells)"
            )
        cell = json.loads(path.read_text())
        if cell.get("dataset_sha256") != dataset_sha:
            raise ValueError(f"{path}: dataset hash mismatch")
        cells.append(cell)
    orders = {tuple(cell["players"]) for cell in cells}
    if len(orders) != 1:
        raise ValueError(f"{cohort}: G4 seed cells disagree on player order")
    return cells


def _run_cohort(cohort: str, path: Path) -> dict:
    dataset_sha = _sha256(path)
    g4_cells = _load_g4_cells(cohort, dataset_sha)
    dataset = load_dataset(str(path))
    memory = run_timing_vs_aggregate(
        dataset,
        split_mode="session",
        epochs=15,  # ignored by the memory arm; kept for protocol parity
        seed=0,
        bootstrap_n=2000,
        pure_external=True,
        latent_control="memory",
    )
    players = list(memory.player_ids or ())
    if players != list(g4_cells[0]["players"]):
        raise ValueError(f"{cohort}: player order differs from G4 cells")
    mem_nll = [float(x) for x in memory.b4z_per_player or ()]

    # Per player: seed-averaged learned-arm NLLs from the cached G4 cells.
    static_nll = [
        statistics.mean(cell["static_nll"][i] for cell in g4_cells)
        for i in range(len(players))
    ]
    evolving_nll = [
        statistics.mean(cell["evolving_nll"][i] for cell in g4_cells)
        for i in range(len(players))
    ]
    evolving_minus_memory = [e - m for e, m in zip(evolving_nll, mem_nll)]
    memory_minus_static = [m - s for m, s in zip(mem_nll, static_nll)]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort,
        "dataset_path": str(path),
        "dataset_sha256": dataset_sha,
        "protocol": {
            "split_mode": "session",
            "pure_external": True,
            "allie_offset": "locked_with_global_intercept",
            "memory_features": list(MEMORY_FEATURE_NAMES),
            "learned_arm_seeds": list(FROZEN_SEEDS),
            "memory_arm": "deterministic (lstsq readout only, no training)",
        },
        "players": players,
        "memory_nll": mem_nll,
        "static_nll_seed_mean": static_nll,
        "evolving_nll_seed_mean": evolving_nll,
        "evolving_minus_memory": evolving_minus_memory,
        "memory_minus_static": memory_minus_static,
        "memory_add_over_allie": _ci_dict(memory.add_ci),
        "means": {
            "allie": float(memory.b4_nll),
            "allie_plus_memory": float(memory.b4z_nll),
            "allie_plus_static": statistics.mean(static_nll),
            "allie_plus_evolving": statistics.mean(evolving_nll),
        },
        "cis": {
            "evolving_minus_memory": _ci_dict(
                bootstrap_ci(evolving_minus_memory, n_resamples=5000, seed=0)
            ),
            "memory_minus_static": _ci_dict(
                bootstrap_ci(memory_minus_static, n_resamples=5000, seed=0)
            ),
        },
    }


def _pool(cohort_results: list[dict]) -> dict:
    pooled: dict[str, dict[str, list[float]]] = {
        "evolving_minus_memory": defaultdict(list),
        "memory_minus_static": defaultdict(list),
    }
    for result in cohort_results:
        for key in pooled:
            for player, diff in zip(result["players"], result[key]):
                pooled[key][player].append(diff)
    overall = {}
    for key, per_player in pooled.items():
        unique = [statistics.mean(vals) for vals in per_player.values()]
        overall[key] = {
            "n_unique_players": len(unique),
            "ci": _ci_dict(bootstrap_ci(unique, n_resamples=5000, seed=0)),
        }
    return overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", action="append", choices=list(DEFAULT_COHORTS)
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cohorts = args.cohort or list(DEFAULT_COHORTS)
    cohort_results = []
    for cohort in cohorts:
        target = args.out_dir / f"{cohort}.json"
        source = DEFAULT_COHORTS[cohort]
        if target.exists() and not args.force:
            result = json.loads(target.read_text())
            if result.get("dataset_sha256") != _sha256(source):
                raise ValueError(f"{target}: dataset changed; use --force")
            print(f"[skip] {target}")
        else:
            result = _run_cohort(cohort, source)
            _write_json_atomic(target, result)
        cohort_results.append(result)
        means = result["means"]
        add = result["memory_add_over_allie"]
        em = result["cis"]["evolving_minus_memory"]
        ms = result["cis"]["memory_minus_static"]
        print(
            f"[memory] {cohort} allie={means['allie']:.4f} "
            f"+mem={means['allie_plus_memory']:.4f} "
            f"+static={means['allie_plus_static']:.4f} "
            f"+evolving={means['allie_plus_evolving']:.4f}"
        )
        print(
            f"[memory] {cohort} (allie+mem)-allie={add['point']:+.4f} "
            f"[{add['low']:+.4f},{add['high']:+.4f}] "
            f"P(<0)={add['p_below_zero']:.3f}"
        )
        print(
            f"[memory] {cohort} evolving-memory={em['point']:+.4f} "
            f"[{em['low']:+.4f},{em['high']:+.4f}] "
            f"P(<0)={em['p_below_zero']:.3f}"
            f" | memory-static={ms['point']:+.4f} "
            f"[{ms['low']:+.4f},{ms['high']:+.4f}] "
            f"P(<0)={ms['p_below_zero']:.3f}",
            flush=True,
        )

    if set(cohorts) == set(DEFAULT_COHORTS):
        overall = _pool(cohort_results)
        summary = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "comparison": (
                "structured-memory control vs learned static/evolving latents"
                " over locked Allie"
            ),
            "cohorts": {
                result["cohort"]: {
                    "means": result["means"],
                    "memory_add_over_allie": result["memory_add_over_allie"],
                    "cis": result["cis"],
                }
                for result in cohort_results
            },
            "overall": overall,
        }
        _write_json_atomic(args.summary_out, summary)
        for key, entry in overall.items():
            ci = entry["ci"]
            print(
                f"[memory] overall {key}={ci['point']:+.4f} "
                f"[{ci['low']:+.4f},{ci['high']:+.4f}] "
                f"P(<0)={ci['p_below_zero']:.3f} "
                f"n={entry['n_unique_players']} -> {args.summary_out}"
            )
    else:
        print("[pending] run all cohorts to produce the pooled summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
