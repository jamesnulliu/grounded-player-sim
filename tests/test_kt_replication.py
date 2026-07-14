"""Frozen protocol, preparation, and scaling-fit tests for real KT."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gps.data.kt_csv import load_kt_csv
from gps.experiments.kt_replication import (
    KTDatasetSpec,
    file_sha256,
    fit_scaling_relationship,
    inspect_kt_export,
    load_and_verify_provenance,
    load_replication_manifest,
    observed_accuracy_spread,
    provenance_path,
)
from scripts.prepare_kt_data import prepare_standard_kt

MANIFEST = Path("scripts/kt_replication_manifest.json")


def test_frozen_manifest_preserves_original_protocol():
    manifest = load_replication_manifest(MANIFEST)
    assert [d.dataset_id for d in manifest.datasets] == [
        "assist09",
        "assist12",
        "assist15",
        "assist17",
        "algebra05",
        "bridge06",
        "spanish",
        "statics",
    ]
    assert manifest.protocol.seeds == (0, 1, 2)
    assert manifest.protocol.epochs == 60
    assert manifest.protocol.max_len == 200
    assert manifest.protocol.train_frac == 0.7
    sizes = {d.dataset_id: d.n_students for d in manifest.datasets}
    assert sizes["spanish"] == 150
    assert sizes["statics"] == 200
    assert {sizes[k] for k in sizes if k not in {"spanish", "statics"}} == {
        500
    }


def test_standard_preparer_and_inspector_preserve_order(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "skill_id,correct,user_id,timestamp,item_id,ignored\n"
        "s1,1,A,1,i1,x\n"
        "s2,0,A,2,i2,x\n"
        "s1,1,A,3,i3,x\n"
        "s1,0,B,1,i4,x\n"
        "s2,0,B,2,i5,x\n"
        "s2,0,B,3,i6,x\n"
    )
    prepared = tmp_path / "prepared.tsv"
    prepare_standard_kt(str(source), str(prepared), delimiter=",")

    lines = prepared.read_text().splitlines()
    assert lines[0] == "user_id\titem_id\ttimestamp\tcorrect\tskill_id"
    assert lines[1].startswith("A\ti1\t1\t1\ts1")
    stats = inspect_kt_export(
        prepared, min_responses=3, n_students=2, max_len=2
    )
    assert stats == {
        "n_rows": 6,
        "n_users": 2,
        "n_skills": 2,
        "n_eligible_users": 2,
        "n_selected_users": 2,
        "n_selected_responses": 4,
    }

    dataset = load_kt_csv(
        str(prepared),
        min_responses=3,
        n_students=2,
        max_len=3,
        train_frac=2 / 3,
    )
    assert observed_accuracy_spread(dataset, train_frac=2 / 3) == 0.5


def test_standard_preparer_rejects_nonbinary_input(tmp_path):
    source = tmp_path / "source.tsv"
    source.write_text(
        "user_id\titem_id\ttimestamp\tcorrect\tskill_id\n"
        "A\ti1\t1\t0.5\ts1\n"
    )
    with pytest.raises(ValueError, match="correctness must be 0 or 1"):
        prepare_standard_kt(
            str(source), str(tmp_path / "prepared.tsv"), delimiter="\t"
        )


def test_provenance_receipt_must_match_prepared_hash(tmp_path):
    prepared = tmp_path / "prepared.tsv"
    prepared.write_text(
        "user_id\titem_id\ttimestamp\tcorrect\tskill_id\n"
        "A\ti1\t1\t1\ts1\n"
    )
    spec = KTDatasetSpec(
        dataset_id="tiny",
        label="Tiny",
        source_path=tmp_path / "source.tsv",
        prepared_path=prepared,
        preparer="standard_5col",
        source_delimiter="\t",
        n_students=1,
        source_reference="test source",
        provenance_note="test",
    )
    receipt = {
        "dataset_id": "tiny",
        "manifest_sha256": "manifest",
        "prepared_sha256": file_sha256(prepared),
    }
    provenance_path(prepared).write_text(json.dumps(receipt))
    assert load_and_verify_provenance(spec, "manifest") == receipt

    prepared.write_text(prepared.read_text() + "B\ti2\t2\t0\ts2\n")
    with pytest.raises(ValueError, match="prepared_sha256 mismatch"):
        load_and_verify_provenance(spec, "manifest")


def _historical_cells():
    # Values recorded in results/real_kt.txt before the loader fix. This test
    # verifies the fitter, not the validity of those historical inputs.
    values = {
        "bridge06": (0.096, [0.0045, 0.0041, 0.0037]),
        "algebra05": (0.123, [0.0095, 0.0102, 0.0103]),
        "statics": (0.142, [0.0050, 0.0094, 0.0099]),
        "assist17": (0.147, [0.0145, 0.0128, 0.0143]),
        "assist12": (0.154, [0.0087, 0.0110, 0.0068]),
        "assist15": (0.158, [0.0112, 0.0123, 0.0083]),
        "assist09": (0.190, [0.0095, 0.0116, 0.0090]),
        "spanish": (0.258, [0.0313, 0.0329, 0.0327]),
    }
    return [
        {
            "dataset_id": dataset_id,
            "seed": seed,
            "n_students": 500,
            "prepared_sha256": f"sha-{dataset_id}",
            "cohort_fingerprint": f"cohort-{dataset_id}",
            "observed_spread": spread,
            "response": {"d_minus_b": -effects[seed]},
        }
        for dataset_id, (spread, effects) in values.items()
        for seed in (0, 1, 2)
    ]


def test_scaling_fitter_reproduces_historical_correlation():
    cells = _historical_cells()
    fit = fit_scaling_relationship(
        cells,
        expected_dataset_ids=sorted({c["dataset_id"] for c in cells}),
        bootstrap_n=200,
    )
    assert fit["n_datasets"] == 8
    signed = fit["signed_advantage_fit"]
    absolute = fit["historical_absolute_effect_fit"]
    assert signed["pearson"] == pytest.approx(0.888, abs=0.001)
    assert signed["spearman"] == pytest.approx(0.738, abs=0.001)
    assert absolute["pearson"] == signed["pearson"]
    assert len(signed["leave_one_out"]) == 8
    assert signed["pearson_bootstrap_95"][0] < signed["pearson"]
    assert fit["sign_audit"]["n_seed_cells_d_wins"] == 24


def test_scaling_fitter_rejects_seed_cohort_mismatch():
    cells = _historical_cells()
    cells[1]["cohort_fingerprint"] = "different"
    with pytest.raises(ValueError, match="different cohorts"):
        fit_scaling_relationship(cells, bootstrap_n=10)
