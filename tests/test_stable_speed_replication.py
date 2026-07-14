"""Frozen-protocol and aggregation tests for the stable-speed extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_stable_speed_replication import (
    _load_manifest,
    _pool_channel,
    _read_complete_cells,
)

MANIFEST = Path("scripts/stable_speed_manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_stable_speed_manifest():
    raw, cohorts = _load_manifest(MANIFEST.resolve())

    assert raw["source_prefix_bytes"] == 370_000_000
    assert [item["id"] for item in cohorts] == [
        "2021-04",
        "2023-04",
        "2021-06",
    ]
    assert raw["ingest"] == {
        "speed": "blitz",
        "min_games": 30,
        "min_sessions": 3,
        "max_players": 100,
        "gap_threshold_seconds": 1800.0,
        "workers": 8,
        "batch_size": 512,
        "max_games_per_player": 20,
    }
    assert raw["training"]["seeds"] == [0, 1, 2]
    assert raw["training"]["control"] == "static"
    assert raw["training"]["timing_lambda"] == 0.5
    assert raw["training"]["split_mode"] == "session"
    assert raw["training"]["timing_model"] == "lognormal"


def test_pool_channel_averages_seeds_before_bootstrap():
    cells = [
        {
            "players": ["alice", "bob"],
            "timing_d": [1.0, 3.0],
            "timing_b": [3.0, 2.0],
        },
        {
            "players": ["alice", "bob"],
            "timing_d": [3.0, 3.0],
            "timing_b": [1.0, 2.0],
        },
    ]

    pooled = _pool_channel(cells, "timing_d", "timing_b")

    # Per-player seed means are alice=0 and bob=1, hence cohort mean=0.5.
    assert pooled["d_minus_b"] == pytest.approx(0.5)
    assert pooled["n_players"] == 2
    assert pooled["ci"]["n_units"] == 2


def test_complete_cell_reader_rejects_stale_and_missing_cells(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    raw = {
        "_manifest_path": str(manifest),
        "training": {"seeds": [0, 1]},
    }
    cohorts = [{"id": "cohort"}]
    out_dir = tmp_path / "runs"
    cell_dir = out_dir / "cohort"
    cell_dir.mkdir(parents=True)
    common = {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest),
        "cohort": "cohort",
    }
    (cell_dir / "seed-0.json").write_text(
        json.dumps({**common, "seed": 0})
    )

    with pytest.raises(FileNotFoundError, match="seed-1.json"):
        _read_complete_cells(raw, cohorts, out_dir)

    (cell_dir / "seed-1.json").write_text(
        json.dumps({**common, "seed": 1, "manifest_sha256": "stale"})
    )
    with pytest.raises(ValueError, match="manifest mismatch"):
        _read_complete_cells(raw, cohorts, out_dir)
