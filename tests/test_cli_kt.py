"""Smoke tests for the ``gps kt`` CLI subcommand (synthetic + real --data).

Runs a tiny cohort / few epochs offline so it stays fast; the point is that
the subcommand parses, dispatches, and completes end-to-end (exit 0).
"""

from __future__ import annotations

import pytest

from gps.cli import main

torch = pytest.importorskip("torch")  # kt runs the SFT trainer


def test_gps_kt_synthetic(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    rc = main(["kt", "--n-students", "6", "--epochs", "2", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "synthetic KT: 6 students" in out


def test_gps_kt_real_data(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    # tiny inline real-format CSV (2 eligible students, min-responses=5)
    lines = ["user_id\titem_id\ttimestamp\tcorrect\tskill_id"]
    for i, (sk, c) in enumerate(
        [(1, 1), (1, 0), (2, 0), (1, 1), (2, 1), (2, 0)]
    ):
        lines.append(f"A\t{i}\t{i}\t{c}\t{sk}")
    for i, (sk, c) in enumerate([(1, 0), (2, 1), (1, 1), (2, 0), (1, 1)]):
        lines.append(f"B\t{i}\t{i}\t{c}\t{sk}")
    csv_path = tmp_path / "kt.csv"
    csv_path.write_text("\n".join(lines) + "\n")

    rc = main(
        [
            "kt",
            "--data",
            str(csv_path),
            "--min-responses",
            "5",
            "--epochs",
            "2",
        ]
    )
    assert rc == 0
    assert "real KT" in capsys.readouterr().out
