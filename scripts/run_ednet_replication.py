#!/usr/bin/env python
"""Run and pool the frozen real-timing EdNet comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from gps.data.kt_csv import load_kt_csv
from gps.eval.bootstrap import bootstrap_ci
from gps.experiments.kt import run_kt
from gps.experiments.kt_replication import (
    cohort_fingerprint,
    inspect_kt_export,
)

DEFAULT_MANIFEST = Path(__file__).with_name("ednet_manifest.json")
DEFAULT_OUT_DIR = Path("runs/ednet-kt1-singleton")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / value).resolve()


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


def _load_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported EdNet manifest schema")
    raw["_path"] = str(path)
    raw["_sha256"] = _sha256(path)
    raw["_prepared_path"] = str(
        _resolve(path, raw["dataset"]["prepared_path"])
    )
    return raw


def _validate_prepared(raw: dict) -> tuple[Path, dict, dict]:
    prepared = Path(raw["_prepared_path"])
    receipt_path = prepared.with_suffix(
        prepared.suffix + ".provenance.json"
    )
    if not prepared.exists() or not receipt_path.exists():
        raise FileNotFoundError(
            "missing prepared EdNet cohort; run scripts/prepare_ednet.py"
        )
    receipt = json.loads(receipt_path.read_text())
    checks = {
        "manifest_sha256": raw["_sha256"],
        "prepared_sha256": _sha256(prepared),
        "source_sha256": raw["dataset"]["source_sha256"],
        "contents_sha256": raw["dataset"]["contents_sha256"],
    }
    for key, expected in checks.items():
        if receipt.get(key) != expected:
            raise ValueError(f"EdNet provenance {key} mismatch")
    prep = raw["preparation"]
    stats = inspect_kt_export(
        prepared,
        min_responses=int(prep["min_responses"]),
        n_students=int(prep["n_students"]),
        max_len=int(prep["max_len"]),
    )
    if stats["n_selected_users"] != int(prep["n_students"]):
        raise ValueError("prepared EdNet cohort has the wrong student count")
    return prepared, receipt, stats


def _cell_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"seed-{seed}.json"


def _make_cell(raw, receipt, stats, dataset, result, seed, wall_seconds):
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": raw["dataset"]["id"],
        "seed": seed,
        "manifest_sha256": raw["_sha256"],
        "prepared_sha256": receipt["prepared_sha256"],
        "cohort_fingerprint": cohort_fingerprint(dataset),
        "input_stats": stats,
        "players": [t.player_id for t in dataset.trajectories],
        "response_d": [float(value) for value in result.d_per_player],
        "response_b": [float(value) for value in result.b_per_player],
        "timing_d": [float(value) for value in result.d_timing],
        "timing_b": [float(value) for value in result.b_timing],
        "response_ci": _ci_dict(result.ci),
        "timing_ci": _ci_dict(result.timing_ci),
        "wall_seconds": round(wall_seconds, 3),
    }


def _complete_cells(raw: dict, out_dir: Path) -> list[dict]:
    cells = []
    missing = []
    for seed in raw["training"]["seeds"]:
        path = _cell_path(out_dir, int(seed))
        if not path.exists():
            missing.append(str(path))
            continue
        cell = json.loads(path.read_text())
        if cell.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported cell schema")
        if cell.get("manifest_sha256") != raw["_sha256"]:
            raise ValueError(f"{path}: manifest mismatch")
        if cell.get("dataset_id") != raw["dataset"]["id"]:
            raise ValueError(f"{path}: dataset identity mismatch")
        if cell.get("seed") != seed:
            raise ValueError(f"{path}: seed mismatch")
        cells.append(cell)
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("EdNet cells still missing:\n" + listing)
    fingerprints = {cell["cohort_fingerprint"] for cell in cells}
    player_orders = {tuple(cell["players"]) for cell in cells}
    prepared_hashes = {cell["prepared_sha256"] for cell in cells}
    if len(fingerprints) != 1 or len(player_orders) != 1:
        raise ValueError("EdNet seed cells use different cohorts")
    if len(prepared_hashes) != 1:
        raise ValueError("EdNet seed cells use different prepared files")
    n_players = len(cells[0]["players"])
    channel_keys = ("response_d", "response_b", "timing_d", "timing_b")
    if any(
        len(cell.get(key, ())) != n_players
        for cell in cells
        for key in channel_keys
    ):
        raise ValueError("EdNet cell has incomplete per-student channels")
    return cells


def _pool_channel(cells: list[dict], d_key: str, b_key: str) -> dict:
    diffs = []
    for index, _player in enumerate(cells[0]["players"]):
        values = [
            float(cell[d_key][index]) - float(cell[b_key][index])
            for cell in cells
        ]
        diffs.append(statistics.mean(values))
    ci = bootstrap_ci(diffs, n_resamples=5000, seed=0)
    return {"d_minus_b": float(ci.point), "ci": _ci_dict(ci)}


def _aggregate(raw: dict, out_dir: Path, summary_out: Path) -> dict:
    cells = _complete_cells(raw, out_dir)
    response = _pool_channel(cells, "response_d", "response_b")
    timing = _pool_channel(cells, "timing_d", "timing_b")
    timing_transfer = (
        timing["ci"]["high"] < 0
        and all(cell["timing_ci"]["point"] < 0 for cell in cells)
    )
    response_ci = response["ci"]
    response_includes_zero = response_ci["low"] <= 0 <= response_ci["high"]
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": raw["dataset"]["id"],
        "manifest_path": raw["_path"],
        "manifest_sha256": raw["_sha256"],
        "protocol": {
            "preparation": raw["preparation"],
            "training": raw["training"],
            "success_criteria": raw["success_criteria"],
        },
        "n_students": len(cells[0]["players"]),
        "seeds": sorted(cell["seed"] for cell in cells),
        "response": response,
        "timing": timing,
        "seed_results": [
            {
                "seed": cell["seed"],
                "response_ci": cell["response_ci"],
                "timing_ci": cell["timing_ci"],
            }
            for cell in sorted(cells, key=lambda value: value["seed"])
        ],
        "criteria": {
            "timing_transfer": timing_transfer,
            "response_ci_includes_zero": response_includes_zero,
            "full_when_not_what": (
                timing_transfer and response_includes_zero
            ),
        },
    }
    _write_json_atomic(summary_out, summary)
    print(
        f"[ednet] response D-B={response['ci']['point']:+.4f} "
        f"[{response['ci']['low']:+.4f},{response['ci']['high']:+.4f}]"
    )
    print(
        f"[ednet] timing D-B={timing['ci']['point']:+.4f} "
        f"[{timing['ci']['low']:+.4f},{timing['ci']['high']:+.4f}]"
    )
    print(f"[ednet] criteria={summary['criteria']} -> {summary_out}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    raw = _load_manifest(manifest_path)
    if args.aggregate_only:
        _aggregate(
            raw,
            args.out_dir,
            args.summary_out or args.out_dir / "summary.json",
        )
        return 0
    prepared, receipt, stats = _validate_prepared(raw)
    print(
        f"[validated] EdNet: {stats['n_rows']} rows, "
        f"{stats['n_selected_users']} selected"
    )
    if args.validate_only:
        return 0

    training = raw["training"]
    allowed_seeds = set(training["seeds"])
    seeds = args.seed or training["seeds"]
    if set(seeds) - allowed_seeds:
        raise ValueError("seed outside frozen EdNet protocol")
    clip = tuple(
        float(value)
        for value in raw["preparation"]["rt_clip_seconds"]
    )
    dataset = load_kt_csv(
        str(prepared),
        n_students=int(raw["preparation"]["n_students"]),
        min_responses=int(raw["preparation"]["min_responses"]),
        max_len=int(raw["preparation"]["max_len"]),
        train_frac=float(training["train_frac"]),
        response_time_col=5,
        rt_clip=clip,
    )
    os.environ.setdefault("WANDB_PROJECT", "gps-ednet-kt1-singleton")
    for seed in seeds:
        path = _cell_path(args.out_dir, int(seed))
        if path.exists() and not args.force:
            cell = json.loads(path.read_text())
            if (
                cell.get("manifest_sha256") != raw["_sha256"]
                or cell.get("prepared_sha256") != receipt["prepared_sha256"]
                or cell.get("dataset_id") != raw["dataset"]["id"]
                or cell.get("seed") != int(seed)
            ):
                raise ValueError(f"{path}: stale cell; rerun with --force")
            print(f"[skip] verified existing cell {path}")
            continue
        started = time.monotonic()
        result = run_kt(
            dataset,
            train_frac=float(training["train_frac"]),
            latent_dim=int(training["latent_dim"]),
            hidden_dim=int(training["hidden_dim"]),
            epochs=int(training["epochs"]),
            lr=float(training["lr"]),
            seed=int(seed),
            batch_size=int(training["batch_size"]),
            bootstrap_n=int(training["bootstrap_n"]),
            timing_lambda=float(training["timing_lambda"]),
        )
        cell = _make_cell(
            raw,
            receipt,
            stats,
            dataset,
            result,
            int(seed),
            time.monotonic() - started,
        )
        _write_json_atomic(path, cell)
        print(
            f"[cell] EdNet seed={seed} response={result.ci.point:+.4f} "
            f"timing={result.timing_ci.point:+.4f} -> {path}"
        )
    if all(
        _cell_path(args.out_dir, int(seed)).exists()
        for seed in allowed_seeds
    ):
        _aggregate(
            raw,
            args.out_dir,
            args.summary_out or args.out_dir / "summary.json",
        )
    else:
        print("[pending] run remaining EdNet cells before pooling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
