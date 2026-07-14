#!/usr/bin/env python
"""Run and aggregate the leakage-fixed eight-dataset KT replication.

Each dataset/seed cell writes a self-describing JSON record under an ignored
runs directory. Aggregation is allowed only when all 24 frozen cells exist and
agree on their dataset checksum and selected-cohort fingerprint.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from gps.data.kt_csv import load_kt_csv
from gps.experiments.kt import KTResult, run_kt
from gps.experiments.kt_replication import (
    CELL_SCHEMA_VERSION,
    cohort_fingerprint,
    fit_scaling_relationship,
    inspect_kt_export,
    load_and_verify_provenance,
    load_replication_manifest,
    observed_accuracy_spread,
)

DEFAULT_MANIFEST = Path(__file__).with_name("kt_replication_manifest.json")
DEFAULT_OUT_DIR = Path("runs/kt-replication-fixed-loader")


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


def _channel_dict(d_values: list[float], b_values: list[float], ci) -> dict:
    return {
        "d_mean": float(statistics.mean(d_values)),
        "b_mean": float(statistics.mean(b_values)),
        "d_minus_b": float(ci.point),
        "ci": _ci_dict(ci),
        "d_wins_fraction": sum(
            d < b for d, b in zip(d_values, b_values)
        )
        / len(d_values),
    }


def _cell_path(out_dir: Path, dataset_id: str, seed: int) -> Path:
    return out_dir / dataset_id / f"seed-{seed}.json"


def _make_cell(
    *,
    manifest,
    spec,
    seed: int,
    receipt: dict,
    stats: dict,
    dataset,
    result: KTResult,
    wall_seconds: float,
) -> dict:
    timing = None
    if result.timing_ci is not None:
        timing = _channel_dict(
            result.d_timing, result.b_timing, result.timing_ci
        )
    return {
        "schema_version": CELL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest.sha256,
        "dataset_id": spec.dataset_id,
        "label": spec.label,
        "prepared_path": str(spec.prepared_path),
        "prepared_sha256": receipt["prepared_sha256"],
        "source_sha256": receipt["source_sha256"],
        "cohort_fingerprint": cohort_fingerprint(dataset),
        "n_students": len(dataset.trajectories),
        "seed": seed,
        "protocol": manifest.protocol.as_dict(),
        "input_stats": stats,
        "observed_spread": observed_accuracy_spread(
            dataset, manifest.protocol.train_frac
        ),
        "response": _channel_dict(
            result.d_per_player, result.b_per_player, result.ci
        ),
        "timing": timing,
        "players": [
            {
                "player_id": trajectory.player_id,
                "d_nll": float(d),
                "b_nll": float(b),
                "d_minus_b": float(d - b),
            }
            for trajectory, d, b in zip(
                dataset.trajectories,
                result.d_per_player,
                result.b_per_player,
            )
        ],
        "wall_seconds": round(wall_seconds, 3),
    }


def _validate_existing_cell(
    path: Path,
    *,
    manifest_sha: str,
    prepared_sha: str,
    dataset_id: str,
    seed: int,
):
    cell = json.loads(path.read_text())
    if cell.get("schema_version") != CELL_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported cell schema")
    if cell.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"{path}: manifest changed; rerun with --force")
    if cell.get("prepared_sha256") != prepared_sha:
        raise ValueError(
            f"{path}: prepared dataset changed; rerun with --force"
        )
    if cell.get("dataset_id") != dataset_id or cell.get("seed") != seed:
        raise ValueError(f"{path}: cell identity mismatch; rerun with --force")
    return cell


def _read_complete_cells(manifest, out_dir: Path) -> list[dict]:
    cells = []
    missing = []
    for spec in manifest.datasets:
        for seed in manifest.protocol.seeds:
            path = _cell_path(out_dir, spec.dataset_id, seed)
            if not path.exists():
                missing.append(str(path))
                continue
            cell = json.loads(path.read_text())
            if cell.get("schema_version") != CELL_SCHEMA_VERSION:
                raise ValueError(f"{path}: unsupported cell schema")
            if cell.get("manifest_sha256") != manifest.sha256:
                raise ValueError(f"{path}: result uses a different manifest")
            if cell.get("dataset_id") != spec.dataset_id:
                raise ValueError(f"{path}: dataset identity mismatch")
            if cell.get("seed") != seed:
                raise ValueError(f"{path}: seed identity mismatch")
            cells.append(cell)
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "cannot aggregate until every frozen cell exists:\n" + listing
        )
    expected_seeds = set(manifest.protocol.seeds)
    for spec in manifest.datasets:
        actual = {
            int(cell["seed"])
            for cell in cells
            if cell["dataset_id"] == spec.dataset_id
        }
        if actual != expected_seeds:
            raise ValueError(
                f"{spec.dataset_id}: expected seeds {sorted(expected_seeds)}, "
                f"got {sorted(actual)}"
            )
    return cells


def _aggregate(manifest, out_dir: Path, summary_out: Path) -> dict:
    cells = _read_complete_cells(manifest, out_dir)
    fit = fit_scaling_relationship(
        cells,
        expected_dataset_ids=[d.dataset_id for d in manifest.datasets],
    )
    summary = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "protocol": manifest.protocol.as_dict(),
        "scaling_fit": fit,
    }
    _write_json_atomic(summary_out, summary)
    signed = fit["signed_advantage_fit"]
    absolute = fit["historical_absolute_effect_fit"]
    print(
        f"[scaling] n={fit['n_datasets']} "
        f"signed Pearson={signed['pearson']:.3f} "
        f"Spearman={signed['spearman']:.3f}; historical-absolute "
        f"Pearson={absolute['pearson']:.3f} "
        f"Spearman={absolute['spearman']:.3f} "
        f"-> {summary_out}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="dataset id; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="seed from the frozen manifest; repeat as needed (default: all)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_replication_manifest(args.manifest)
    if args.aggregate_only:
        target = args.summary_out or args.out_dir / "summary.json"
        _aggregate(manifest, args.out_dir, target)
        return 0

    selected = manifest.select(args.dataset)
    seeds = tuple(args.seed or manifest.protocol.seeds)
    unknown_seeds = set(seeds) - set(manifest.protocol.seeds)
    if unknown_seeds:
        raise SystemExit(
            f"seeds outside frozen protocol: {sorted(unknown_seeds)}"
        )

    missing = [
        spec.prepared_path
        for spec in selected
        if not spec.prepared_path.exists()
    ]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "missing prepared KT exports; run "
            "scripts/prepare_kt_replications.py first:\n" + listing
        )

    preflight = {}
    for spec in selected:
        receipt = load_and_verify_provenance(spec, manifest.sha256)
        stats = inspect_kt_export(
            spec.prepared_path,
            min_responses=manifest.protocol.min_responses,
            n_students=spec.n_students,
            max_len=manifest.protocol.max_len,
        )
        if stats["n_selected_users"] != spec.n_students:
            raise ValueError(
                f"{spec.dataset_id}: expected {spec.n_students} selected "
                f"students, got {stats['n_selected_users']}"
            )
        preflight[spec.dataset_id] = (receipt, stats)
        print(
            f"[validated] {spec.dataset_id}: {stats['n_rows']} rows, "
            f"{stats['n_selected_users']} selected"
        )
    if args.validate_only:
        return 0

    os.environ.setdefault("WANDB_PROJECT", "gps-kt-scaling-fixed-loader")
    for spec in selected:
        receipt, stats = preflight[spec.dataset_id]
        dataset = load_kt_csv(
            str(spec.prepared_path),
            n_students=spec.n_students,
            min_responses=manifest.protocol.min_responses,
            max_len=manifest.protocol.max_len,
            train_frac=manifest.protocol.train_frac,
        )
        for seed in seeds:
            path = _cell_path(args.out_dir, spec.dataset_id, seed)
            if path.exists() and not args.force:
                _validate_existing_cell(
                    path,
                    manifest_sha=manifest.sha256,
                    prepared_sha=receipt["prepared_sha256"],
                    dataset_id=spec.dataset_id,
                    seed=seed,
                )
                print(f"[skip] verified existing cell {path}")
                continue
            started = time.monotonic()
            result = run_kt(
                dataset,
                train_frac=manifest.protocol.train_frac,
                latent_dim=manifest.protocol.latent_dim,
                hidden_dim=manifest.protocol.hidden_dim,
                epochs=manifest.protocol.epochs,
                lr=manifest.protocol.lr,
                seed=seed,
                batch_size=manifest.protocol.batch_size,
                bootstrap_n=manifest.protocol.bootstrap_n,
            )
            cell = _make_cell(
                manifest=manifest,
                spec=spec,
                seed=seed,
                receipt=receipt,
                stats=stats,
                dataset=dataset,
                result=result,
                wall_seconds=time.monotonic() - started,
            )
            _write_json_atomic(path, cell)
            print(
                f"[cell] {spec.dataset_id} seed={seed} "
                f"D-B={result.ci.point:+.4f} -> {path}"
            )

    all_complete = all(
        _cell_path(args.out_dir, spec.dataset_id, seed).exists()
        for spec in manifest.datasets
        for seed in manifest.protocol.seeds
    )
    if all_complete:
        target = args.summary_out or args.out_dir / "summary.json"
        _aggregate(manifest, args.out_dir, target)
    else:
        print("[pending] run remaining cells before fitting the headline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
