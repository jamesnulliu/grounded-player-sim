#!/usr/bin/env python
"""Run and pool the frozen three-cohort stable-speed extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from gps.data.store import load_dataset
from gps.eval.bootstrap import bootstrap_ci
from gps.experiments.ec import run_ec

DEFAULT_MANIFEST = Path(__file__).with_name("stable_speed_manifest.json")
DEFAULT_OUT_DIR = Path("runs/stable-speed-extension")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / value).resolve()


def _ci_dict(ci) -> dict | None:
    if ci is None:
        return None
    return {
        "point": float(ci.point),
        "low": float(ci.low),
        "high": float(ci.high),
        "p_below_zero": float(ci.p_below_zero),
        "n_units": int(ci.n_units),
        "confidence": float(ci.confidence),
    }


def _load_manifest(path: Path) -> tuple[dict, list[dict]]:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported stable-speed manifest schema")
    cohorts = []
    for item in raw["cohorts"]:
        cohorts.append(
            {
                **item,
                "out_dir": _resolve(path, item["out_dir"]),
                "source_path": _resolve(path, item["source_path"]),
            }
        )
    return raw, cohorts


def _cell_path(out_dir: Path, cohort_id: str, seed: int) -> Path:
    return out_dir / cohort_id / f"seed-{seed}.json"


def _validate_ingest(
    manifest_path: Path, manifest_sha: str, cohort: dict
) -> tuple[Path, dict, str]:
    dataset_path = cohort["out_dir"] / "dataset.jsonl.gz"
    ingest_path = cohort["out_dir"] / "manifest.json"
    if not dataset_path.exists() or not ingest_path.exists():
        raise FileNotFoundError(
            f"missing prepared cohort {cohort['id']}; run "
            "scripts/prepare_stable_speed_cohorts.py"
        )
    ingest = json.loads(ingest_path.read_text())
    if ingest.get("protocol_manifest_sha256") != manifest_sha:
        raise ValueError(
            f"{cohort['id']}: ingest does not match {manifest_path}"
        )
    if ingest.get("n_players_selected") != 100:
        raise ValueError(f"{cohort['id']}: expected 100 selected players")
    return dataset_path, ingest, _sha256(dataset_path)


def _make_cell(
    *,
    cohort: dict,
    seed: int,
    manifest_sha: str,
    dataset_sha: str,
    dataset,
    result,
    training: dict,
    wall_seconds: float,
) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort["id"],
        "seed": seed,
        "manifest_sha256": manifest_sha,
        "dataset_sha256": dataset_sha,
        "training": training,
        "players": [
            trajectory.player_id for trajectory in dataset.trajectories
        ],
        "move_d": result.d_per_player,
        "move_b": result.b_per_player,
        "move_ci": _ci_dict(result.ci),
        "timing_d": result.d_timing_per_player,
        "timing_b": result.b_timing_per_player,
        "timing_ci": _ci_dict(result.timing_ci),
        "wall_seconds": round(wall_seconds, 3),
    }


def _read_complete_cells(raw: dict, cohorts: list[dict], out_dir: Path):
    cells = []
    missing = []
    manifest_sha = _sha256(Path(raw["_manifest_path"]))
    for cohort in cohorts:
        for seed in raw["training"]["seeds"]:
            path = _cell_path(out_dir, cohort["id"], int(seed))
            if not path.exists():
                missing.append(str(path))
                continue
            cell = json.loads(path.read_text())
            if cell.get("schema_version") != 1:
                raise ValueError(f"{path}: unsupported cell schema")
            if cell.get("manifest_sha256") != manifest_sha:
                raise ValueError(f"{path}: manifest mismatch")
            if cell.get("cohort") != cohort["id"] or cell.get("seed") != seed:
                raise ValueError(f"{path}: cell identity mismatch")
            cells.append(cell)
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "stable-speed cells still missing:\n" + listing
        )
    return cells


def _pool_channel(cells: list[dict], d_key: str, b_key: str) -> dict:
    per_player: dict[str, list[float]] = {}
    for cell in cells:
        for player, d, b in zip(cell["players"], cell[d_key], cell[b_key]):
            per_player.setdefault(player, []).append(float(d) - float(b))
    diffs = [statistics.mean(values) for values in per_player.values()]
    ci = bootstrap_ci(diffs, n_resamples=5000, seed=0)
    return {
        "d_minus_b": float(ci.point),
        "ci": _ci_dict(ci),
        "n_players": len(diffs),
    }


def _aggregate(
    raw: dict, cohorts: list[dict], out_dir: Path, summary_out: Path
) -> dict:
    cells = _read_complete_cells(raw, cohorts, out_dir)
    summaries = []
    for cohort in cohorts:
        cohort_cells = [c for c in cells if c["cohort"] == cohort["id"]]
        checksums = {c["dataset_sha256"] for c in cohort_cells}
        player_orders = {tuple(c["players"]) for c in cohort_cells}
        if len(checksums) != 1 or len(player_orders) != 1:
            raise ValueError(f"{cohort['id']}: seed cells use different data")
        summaries.append(
            {
                "cohort": cohort["id"],
                "seeds": sorted(c["seed"] for c in cohort_cells),
                "dataset_sha256": next(iter(checksums)),
                "move": _pool_channel(cohort_cells, "move_d", "move_b"),
                "timing": _pool_channel(
                    cohort_cells, "timing_d", "timing_b"
                ),
                "seed_timing": [
                    {"seed": c["seed"], **c["timing_ci"]}
                    for c in sorted(
                        cohort_cells, key=lambda item: item["seed"]
                    )
                ],
            }
        )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": _sha256(Path(raw["_manifest_path"])),
        "training": raw["training"],
        "cohorts": summaries,
    }
    _write_json_atomic(summary_out, summary)
    for item in summaries:
        ci = item["timing"]["ci"]
        print(
            f"[stable-speed] {item['cohort']} timing D-B="
            f"{ci['point']:+.4f} [{ci['low']:+.4f},{ci['high']:+.4f}] "
            f"P(<0)={ci['p_below_zero']:.3f}"
        )
    print(f"[stable-speed] summary -> {summary_out}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cohort", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    raw, cohorts = _load_manifest(manifest_path)
    raw["_manifest_path"] = str(manifest_path)
    if args.aggregate_only:
        _aggregate(
            raw,
            cohorts,
            args.out_dir,
            args.summary_out or args.out_dir / "summary.json",
        )
        return 0

    requested = set(args.cohort or [item["id"] for item in cohorts])
    cohorts = [item for item in cohorts if item["id"] in requested]
    if requested - {item["id"] for item in cohorts}:
        raise ValueError("unknown cohort requested")
    seeds = tuple(args.seed or raw["training"]["seeds"])
    if set(seeds) - set(raw["training"]["seeds"]):
        raise ValueError("seed outside frozen protocol")

    manifest_sha = _sha256(manifest_path)
    prepared = {}
    for cohort in cohorts:
        prepared[cohort["id"]] = _validate_ingest(
            manifest_path, manifest_sha, cohort
        )
        print(f"[validated] {cohort['id']}: {prepared[cohort['id']][0]}")
    if args.validate_only:
        return 0

    os.environ.setdefault("WANDB_PROJECT", "gps-stable-speed-extension")
    training = raw["training"]
    for cohort in cohorts:
        dataset_path, _ingest, dataset_sha = prepared[cohort["id"]]
        dataset = load_dataset(str(dataset_path))
        for seed in seeds:
            path = _cell_path(args.out_dir, cohort["id"], int(seed))
            if path.exists() and not args.force:
                cell = json.loads(path.read_text())
                if (
                    cell.get("manifest_sha256") != manifest_sha
                    or cell.get("dataset_sha256") != dataset_sha
                ):
                    raise ValueError(f"{path}: stale cell; rerun with --force")
                print(f"[skip] verified existing cell {path}")
                continue
            started = time.monotonic()
            result = run_ec(
                dataset,
                train_frac=float(training["train_frac"]),
                latent_dim=int(training["latent_dim"]),
                hidden_dim=int(training["hidden_dim"]),
                epochs=int(training["epochs"]),
                lr=float(training["lr"]),
                seed=int(seed),
                bootstrap_n=int(training["bootstrap_n"]),
                batch_size=int(training["batch_size"]),
                split_mode=training["split_mode"],
                control=training["control"],
                timing_lambda=float(training["timing_lambda"]),
                timing_model=training["timing_model"],
                trunk=training["trunk"],
            )
            cell = _make_cell(
                cohort=cohort,
                seed=int(seed),
                manifest_sha=manifest_sha,
                dataset_sha=dataset_sha,
                dataset=dataset,
                result=result,
                training=training,
                wall_seconds=time.monotonic() - started,
            )
            _write_json_atomic(path, cell)
            print(
                f"[cell] {cohort['id']} seed={seed} timing D-B="
                f"{result.timing_ci.point:+.4f} -> {path}"
            )

    complete = all(
        _cell_path(args.out_dir, cohort["id"], int(seed)).exists()
        for cohort in _load_manifest(manifest_path)[1]
        for seed in raw["training"]["seeds"]
    )
    if complete:
        _aggregate(
            raw,
            _load_manifest(manifest_path)[1],
            args.out_dir,
            args.summary_out or args.out_dir / "summary.json",
        )
    else:
        print("[pending] run remaining stable-speed cells before pooling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
