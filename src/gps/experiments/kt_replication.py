"""Reproducibility helpers for the eight-dataset real-KT replication.

The original cross-dataset runs came from uncommitted scratch files. This
module makes the replacement protocol structured and testable: it loads a
frozen manifest, validates canonical exports, fingerprints inputs/cohorts, and
refits the spread-vs-effect relationship from per-seed result cells.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gps.policy.board_native import BoardNativeBackbone
from gps.train.base import TrajectoryDataset

MANIFEST_SCHEMA_VERSION = 1
CELL_SCHEMA_VERSION = 1
CANONICAL_COLUMNS = (
    "user_id",
    "item_id",
    "timestamp",
    "correct",
    "skill_id",
)


@dataclass(frozen=True)
class KTReplicationProtocol:
    """Hyperparameters shared by all real-KT replication cells."""

    seeds: tuple[int, ...]
    min_responses: int
    max_len: int
    train_frac: float
    latent_dim: int
    hidden_dim: int
    epochs: int
    lr: float
    batch_size: int
    bootstrap_n: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KTReplicationProtocol":
        protocol = cls(
            seeds=tuple(int(x) for x in raw["seeds"]),
            min_responses=int(raw["min_responses"]),
            max_len=int(raw["max_len"]),
            train_frac=float(raw["train_frac"]),
            latent_dim=int(raw["latent_dim"]),
            hidden_dim=int(raw["hidden_dim"]),
            epochs=int(raw["epochs"]),
            lr=float(raw["lr"]),
            batch_size=int(raw["batch_size"]),
            bootstrap_n=int(raw["bootstrap_n"]),
        )
        if not protocol.seeds:
            raise ValueError("KT replication manifest needs at least one seed")
        if not 0.0 < protocol.train_frac < 1.0:
            raise ValueError("train_frac must be strictly between 0 and 1")
        for name in (
            "min_responses",
            "max_len",
            "latent_dim",
            "hidden_dim",
            "epochs",
            "batch_size",
            "bootstrap_n",
        ):
            if getattr(protocol, name) <= 0:
                raise ValueError(f"{name} must be positive")
        return protocol

    def as_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "min_responses": self.min_responses,
            "max_len": self.max_len,
            "train_frac": self.train_frac,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "bootstrap_n": self.bootstrap_n,
        }


@dataclass(frozen=True)
class KTDatasetSpec:
    """One source export and cohort definition in the replication."""

    dataset_id: str
    label: str
    source_path: Path
    prepared_path: Path
    preparer: str
    source_delimiter: str
    n_students: int
    source_reference: str
    provenance_note: str


@dataclass(frozen=True)
class KTReplicationManifest:
    """Resolved replication manifest; all paths are absolute."""

    path: Path
    protocol: KTReplicationProtocol
    datasets: tuple[KTDatasetSpec, ...]

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)

    def select(self, ids: list[str] | None = None) -> list[KTDatasetSpec]:
        if not ids or ids == ["all"]:
            return list(self.datasets)
        requested = set(ids)
        known = {d.dataset_id for d in self.datasets}
        unknown = requested - known
        if unknown:
            raise ValueError(
                "unknown dataset id(s): " + ", ".join(sorted(unknown))
            )
        return [d for d in self.datasets if d.dataset_id in requested]


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_replication_manifest(path: str | Path) -> KTReplicationManifest:
    """Load and validate the frozen JSON protocol manifest."""
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported KT replication manifest schema: "
            f"{raw.get('schema_version')!r}"
        )
    protocol = KTReplicationProtocol.from_dict(raw["protocol"])
    datasets = []
    seen: set[str] = set()
    for item in raw["datasets"]:
        dataset_id = str(item["id"])
        if dataset_id in seen:
            raise ValueError(f"duplicate KT dataset id: {dataset_id}")
        seen.add(dataset_id)
        n_students = int(item["n_students"])
        if n_students <= 0:
            raise ValueError(f"{dataset_id}: n_students must be positive")
        delimiter_name = item.get("source_delimiter", "tab")
        delimiters = {"tab": "\t", "comma": ","}
        if delimiter_name not in delimiters:
            raise ValueError(
                f"{dataset_id}: source_delimiter must be 'tab' or 'comma'"
            )
        datasets.append(
            KTDatasetSpec(
                dataset_id=dataset_id,
                label=str(item["label"]),
                source_path=_resolve_path(
                    manifest_path, item["source_path"]
                ),
                prepared_path=_resolve_path(
                    manifest_path, item["prepared_path"]
                ),
                preparer=str(item["preparer"]),
                source_delimiter=delimiters[delimiter_name],
                n_students=n_students,
                source_reference=str(item["source_reference"]),
                provenance_note=str(item["provenance_note"]),
            )
        )
    if not datasets:
        raise ValueError("KT replication manifest has no datasets")
    return KTReplicationManifest(manifest_path, protocol, tuple(datasets))


def file_sha256(path: str | Path) -> str:
    """Hash a potentially large dataset without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(prepared_path: str | Path) -> Path:
    """Return the preparation receipt path for a canonical export."""
    path = Path(prepared_path)
    return path.with_name(path.name + ".provenance.json")


def inspect_kt_export(
    path: str | Path,
    *,
    delimiter: str = "\t",
    min_responses: int = 50,
    n_students: int = 500,
    max_len: int = 200,
) -> dict[str, Any]:
    """Strictly validate a canonical export and report cohort counts."""
    counts: dict[str, int] = {}
    skills: set[str] = set()
    n_rows = 0
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty KT export: {path}") from exc
        if tuple(header[:5]) != CANONICAL_COLUMNS:
            raise ValueError(
                f"{path}: first five columns must be "
                f"{list(CANONICAL_COLUMNS)}, got {header[:5]}"
            )
        for line_number, parts in enumerate(reader, start=2):
            if len(parts) < 5:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 5 columns"
                )
            user_id, item_id, _timestamp, correct, skill_id = parts[:5]
            if not user_id or not item_id or not skill_id:
                raise ValueError(
                    f"{path}:{line_number}: user/item/skill cannot be blank"
                )
            if correct not in {"0", "1"}:
                raise ValueError(
                    f"{path}:{line_number}: correctness must be 0 or 1"
                )
            counts[user_id] = counts.get(user_id, 0) + 1
            skills.add(skill_id)
            n_rows += 1
    eligible = [u for u, count in counts.items() if count >= min_responses]
    selected = eligible[:n_students]
    return {
        "n_rows": n_rows,
        "n_users": len(counts),
        "n_skills": len(skills),
        "n_eligible_users": len(eligible),
        "n_selected_users": len(selected),
        "n_selected_responses": sum(min(counts[u], max_len) for u in selected),
    }


def load_and_verify_provenance(
    spec: KTDatasetSpec, manifest_sha256: str
) -> dict[str, Any]:
    """Require a matching source/prepared-data receipt before training."""
    receipt_path = provenance_path(spec.prepared_path)
    if not receipt_path.exists():
        raise FileNotFoundError(
            f"missing provenance receipt for {spec.dataset_id}: "
            f"{receipt_path}; "
            "run scripts/prepare_kt_replications.py first"
        )
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "dataset_id": spec.dataset_id,
        "manifest_sha256": manifest_sha256,
        "prepared_sha256": file_sha256(spec.prepared_path),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"{spec.dataset_id}: provenance {key} mismatch; "
                "re-run preparation from the frozen manifest"
            )
    return receipt


def observed_accuracy_spread(
    dataset: TrajectoryDataset, train_frac: float
) -> float:
    """Return held-out per-student accuracy spread for the regression."""
    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=train_frac
    )
    accuracies = []
    for trajectory, boundary in zip(dataset.trajectories, splits):
        outcomes = trajectory.observations[boundary:]
        if outcomes:
            accuracies.append(
                sum(x.move == "correct" for x in outcomes) / len(outcomes)
            )
    if len(accuracies) < 2:
        raise ValueError("need at least two evaluated students for spread")
    return float(statistics.pstdev(accuracies))


def cohort_fingerprint(dataset: TrajectoryDataset) -> str:
    """Hash selected player IDs and sequence lengths in file order."""
    payload = [
        [trajectory.player_id, len(trajectory.decisions)]
        for trajectory in dataset.trajectories
    ]
    blob = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _rank(values: list[float]) -> list[float]:
    """Return average ranks for ties, like scipy.stats.rankdata."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = average_rank
        i = j
    return ranks


def _correlation(x: list[float], y: list[float]) -> float:
    import numpy as np

    if len(x) < 2 or statistics.pstdev(x) == 0 or statistics.pstdev(y) == 0:
        raise ValueError("correlation needs two non-constant vectors")
    return float(np.corrcoef(x, y)[0, 1])


def _fit_correlation(
    points: list[dict[str, Any]],
    *,
    y_key: str,
    bootstrap_n: int,
    seed: int,
) -> dict[str, Any]:
    """Fit one correlation definition with identical sensitivity checks."""
    import numpy as np

    x = [p["observed_spread"] for p in points]
    y = [p[y_key] for p in points]
    pearson = _correlation(x, y)
    spearman = _correlation(_rank(x), _rank(y))
    slope, intercept = np.polyfit(x, y, 1)

    leave_one_out = []
    for omitted in range(len(points)):
        xx = [value for i, value in enumerate(x) if i != omitted]
        yy = [value for i, value in enumerate(y) if i != omitted]
        leave_one_out.append(
            {
                "omitted": points[omitted]["dataset_id"],
                "pearson": _correlation(xx, yy),
                "spearman": _correlation(_rank(xx), _rank(yy)),
            }
        )

    rng = np.random.default_rng(seed)
    pearson_samples = []
    spearman_samples = []
    for _ in range(bootstrap_n):
        indices = rng.integers(0, len(points), size=len(points)).tolist()
        xx = [x[i] for i in indices]
        yy = [y[i] for i in indices]
        try:
            pearson_samples.append(_correlation(xx, yy))
            spearman_samples.append(_correlation(_rank(xx), _rank(yy)))
        except ValueError:
            continue
    if not pearson_samples:
        raise ValueError("all dataset bootstrap samples were degenerate")

    return {
        "y_key": y_key,
        "pearson": pearson,
        "spearman": spearman,
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "pearson_bootstrap_95": [
            float(value)
            for value in np.percentile(pearson_samples, [2.5, 97.5])
        ],
        "spearman_bootstrap_95": [
            float(value)
            for value in np.percentile(spearman_samples, [2.5, 97.5])
        ],
        "leave_one_out": leave_one_out,
        "leave_one_out_pearson_range": [
            min(item["pearson"] for item in leave_one_out),
            max(item["pearson"] for item in leave_one_out),
        ],
        "leave_one_out_spearman_range": [
            min(item["spearman"] for item in leave_one_out),
            max(item["spearman"] for item in leave_one_out),
        ],
        "bootstrap_n": bootstrap_n,
        "bootstrap_seed": seed,
    }


def fit_scaling_relationship(
    cells: list[dict[str, Any]],
    *,
    expected_dataset_ids: list[str] | None = None,
    bootstrap_n: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Pool seeds and fit signed and historical absolute-effect regressions.

    The signed definition, ``-mean(D-B)``, is the interpretable latent
    advantage: positive means D wins. The historical analysis used
    ``abs(mean(D-B))``. Both are retained because they are identical while all
    datasets favor D, but diverge if a fixed-loader cohort reverses sign.
    """
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_dataset.setdefault(cell["dataset_id"], []).append(cell)
    if expected_dataset_ids is not None:
        missing = set(expected_dataset_ids) - set(by_dataset)
        extra = set(by_dataset) - set(expected_dataset_ids)
        if missing or extra:
            raise ValueError(
                f"scaling cells mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

    points = []
    for dataset_id, dataset_cells in sorted(by_dataset.items()):
        checksums = {c["prepared_sha256"] for c in dataset_cells}
        fingerprints = {c["cohort_fingerprint"] for c in dataset_cells}
        spreads = {
            round(float(c["observed_spread"]), 12) for c in dataset_cells
        }
        seeds = [int(c["seed"]) for c in dataset_cells]
        if len(checksums) != 1 or len(fingerprints) != 1 or len(spreads) != 1:
            raise ValueError(f"{dataset_id}: seed cells use different cohorts")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{dataset_id}: duplicate seed cell")
        effects = [
            float(c["response"]["d_minus_b"]) for c in dataset_cells
        ]
        mean_effect = float(statistics.mean(effects))
        points.append(
            {
                "dataset_id": dataset_id,
                "n_students": int(dataset_cells[0]["n_students"]),
                "seeds": sorted(seeds),
                "observed_spread": next(iter(spreads)),
                "mean_d_minus_b": mean_effect,
                "signed_advantage": -mean_effect,
                "effect_magnitude": abs(mean_effect),
                "seed_d_minus_b": effects,
                "seed_results": [
                    {
                        "seed": int(cell["seed"]),
                        "d_minus_b": float(cell["response"]["d_minus_b"]),
                        "ci": cell["response"].get("ci"),
                    }
                    for cell in sorted(
                        dataset_cells, key=lambda item: int(item["seed"])
                    )
                ],
                "prepared_sha256": next(iter(checksums)),
                "cohort_fingerprint": next(iter(fingerprints)),
            }
        )
    if len(points) < 3:
        raise ValueError("scaling fit needs at least three datasets")

    sign_audit = {
        "n_seed_cells": sum(len(p["seed_d_minus_b"]) for p in points),
        "n_seed_cells_d_wins": sum(
            effect < 0 for p in points for effect in p["seed_d_minus_b"]
        ),
        "n_datasets": len(points),
        "n_datasets_mean_d_wins": sum(
            p["mean_d_minus_b"] < 0 for p in points
        ),
        "datasets_mean_d_loses": [
            p["dataset_id"] for p in points if p["mean_d_minus_b"] >= 0
        ],
    }
    seed_cis = [
        result["ci"] for p in points for result in p["seed_results"]
    ]
    if all(ci is not None for ci in seed_cis):
        sign_audit.update(
            {
                "n_seed_cells_significant_d_wins": sum(
                    ci["high"] < 0 for ci in seed_cis
                ),
                "n_seed_cells_significant_b_wins": sum(
                    ci["low"] > 0 for ci in seed_cis
                ),
                "n_seed_cells_null": sum(
                    ci["low"] <= 0 <= ci["high"] for ci in seed_cis
                ),
            }
        )

    return {
        "n_datasets": len(points),
        "points": points,
        "signed_advantage_fit": _fit_correlation(
            points,
            y_key="signed_advantage",
            bootstrap_n=bootstrap_n,
            seed=seed,
        ),
        "historical_absolute_effect_fit": _fit_correlation(
            points,
            y_key="effect_magnitude",
            bootstrap_n=bootstrap_n,
            seed=seed,
        ),
        "sign_audit": sign_audit,
    }
