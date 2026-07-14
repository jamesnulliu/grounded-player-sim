"""Pooling tests for the Allie static-vs-evolving comparison."""

import pytest

from scripts.g4_allie_static_vs_evolving import (
    DEFAULT_COHORTS,
    FROZEN_SEEDS,
    _parse_cells,
    _pool_cells,
)


def _cells():
    cells = []
    for cohort_index, cohort in enumerate(DEFAULT_COHORTS):
        players = ["shared-player", f"only-{cohort}"]
        for seed in FROZEN_SEEDS:
            cells.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    "dataset_sha256": f"sha-{cohort}",
                    "players": players,
                    "evolving_minus_static": [
                        -0.01 * (cohort_index + 1),
                        -0.02,
                    ],
                    "means": {
                        "allie": 2.0,
                        "allie_plus_static": 1.9,
                        "allie_plus_evolving": 1.8,
                    },
                    "ci": {
                        "point": -0.01,
                        "low": -0.02,
                        "high": 0.0,
                        "p_below_zero": 0.9,
                        "n_units": 2,
                        "confidence": 0.95,
                    },
                }
            )
    return cells


def test_pooling_bootstraps_unique_players_across_cohorts():
    summary = _pool_cells(_cells())

    assert len(summary["cohorts"]) == 3
    assert summary["overall"]["n_cohort_players"] == 6
    assert summary["overall"]["n_unique_players"] == 4
    assert summary["overall"]["evolving_minus_static"]["point"] < 0


def test_cell_parser_accepts_partition_and_rejects_unknowns():
    assert _parse_cells(["2017-04:0", "2021-06:4"]) == [
        ("2017-04", 0),
        ("2021-06", 4),
    ]
    with pytest.raises(ValueError, match="unknown cohort"):
        _parse_cells(["unknown:0"])
    with pytest.raises(ValueError, match="outside"):
        _parse_cells(["2017-04:5"])
