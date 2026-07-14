"""Protocol, preparation, and pooling tests for the EdNet timing arm."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_ednet import prepare_ednet
from scripts.run_ednet_replication import _pool_channel

MANIFEST = Path("scripts/ednet_manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_text(fieldnames, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def test_frozen_ednet_protocol_uses_singleton_bundles():
    raw = json.loads(MANIFEST.read_text())
    prep = raw["preparation"]
    training = raw["training"]

    assert prep["singleton_bundle_only"] is True
    assert prep["require_positive_elapsed_time"] is True
    assert prep["difficulty_key"] == "question_id"
    assert prep["n_students"] == 500
    assert prep["min_responses"] == 50
    assert prep["max_len"] == 200
    assert prep["rt_clip_seconds"] == [0.5, 300.0]
    assert training["seeds"] == [0, 1, 2]
    assert training["train_frac"] == 0.7
    assert training["timing_lambda"] == 0.5
    assert re.fullmatch(r"[0-9a-f]{64}", raw["dataset"]["source_sha256"])
    assert "CI includes zero" in raw["success_criteria"][
        "full_when_not_what"
    ]


def test_ednet_preparer_filters_and_orders_users(tmp_path):
    contents = tmp_path / "contents.zip"
    questions = [
        {
            "question_id": "q1",
            "bundle_id": "b1",
            "correct_answer": "a",
        },
        {
            "question_id": "q2",
            "bundle_id": "b2",
            "correct_answer": "b",
        },
        {
            "question_id": "q3",
            "bundle_id": "b2",
            "correct_answer": "c",
        },
    ]
    with zipfile.ZipFile(contents, "w") as archive:
        archive.writestr(
            "contents/questions.csv",
            _csv_text(
                ["question_id", "bundle_id", "correct_answer"],
                questions,
            ),
        )

    fields = [
        "timestamp",
        "solving_id",
        "question_id",
        "user_answer",
        "elapsed_time",
    ]
    source = tmp_path / "kt1.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "KT1/u10.csv",
            _csv_text(
                fields,
                [
                    {
                        "timestamp": 2,
                        "solving_id": 2,
                        "question_id": "q1",
                        "user_answer": "b",
                        "elapsed_time": 2000,
                    },
                    {
                        "timestamp": 1,
                        "solving_id": 1,
                        "question_id": "q1",
                        "user_answer": "a",
                        "elapsed_time": 1000,
                    },
                ],
            ),
        )
        archive.writestr(
            "KT1/u2.csv",
            _csv_text(
                fields,
                [
                    {
                        "timestamp": 1,
                        "solving_id": 1,
                        "question_id": "q2",
                        "user_answer": "b",
                        "elapsed_time": 1000,
                    },
                    {
                        "timestamp": 2,
                        "solving_id": 2,
                        "question_id": "q1",
                        "user_answer": "a",
                        "elapsed_time": 0,
                    },
                    {
                        "timestamp": 4,
                        "solving_id": 4,
                        "question_id": "q1",
                        "user_answer": "a",
                        "elapsed_time": 4000,
                    },
                    {
                        "timestamp": 3,
                        "solving_id": 3,
                        "question_id": "q1",
                        "user_answer": "b",
                        "elapsed_time": 3000,
                    },
                ],
            ),
        )

    prepared = tmp_path / "prepared.tsv"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {
                    "id": "tiny",
                    "source_path": source.name,
                    "source_url": "unused",
                    "source_sha256": _sha256(source),
                    "contents_path": contents.name,
                    "contents_url": "unused",
                    "contents_sha256": _sha256(contents),
                    "questions_member": "contents/questions.csv",
                    "prepared_path": prepared.name,
                },
                "preparation": {
                    "n_students": 2,
                    "min_responses": 2,
                    "max_len": 2,
                    "rt_clip_seconds": [0.5, 300.0],
                },
            }
        )
    )

    receipt = prepare_ednet(manifest)

    lines = prepared.read_text().splitlines()
    assert lines[1].startswith("u2\tq1\t3\t0\tq1\t3000")
    assert lines[2].startswith("u2\tq1\t4\t1\tq1\t4000")
    assert lines[3].startswith("u10\tq1\t1\t1\tq1\t1000")
    assert receipt["n_students"] == 2
    assert receipt["n_rows"] == 4
    assert receipt["dropped"]["nonpositive_time"] == 1
    assert receipt["dropped"]["non_singleton_or_unknown_question"] == 1


def test_ednet_pooling_averages_seeds_per_student():
    cells = [
        {
            "players": ["u1", "u2"],
            "timing_d": [1.0, 3.0],
            "timing_b": [3.0, 2.0],
        },
        {
            "players": ["u1", "u2"],
            "timing_d": [3.0, 3.0],
            "timing_b": [1.0, 2.0],
        },
    ]

    pooled = _pool_channel(cells, "timing_d", "timing_b")

    assert pooled["d_minus_b"] == pytest.approx(0.5)
    assert pooled["ci"]["n_units"] == 2
