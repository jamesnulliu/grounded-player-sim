#!/usr/bin/env python
"""Download, verify, and prepare the frozen EdNet-KT1 cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).with_name("ednet_manifest.json")
USER_MEMBER = re.compile(r"(?:^|/)u([0-9]+)\.csv$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / value).resolve()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with (
            urllib.request.urlopen(url) as response,
            temporary.open("wb") as out,
        ):
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                out.write(chunk)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verified_file(
    path: Path, url: str, expected: str, download: bool
) -> None:
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"missing {path}; rerun with --download")
        _download(url, path)
    actual = _sha256(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(
            f"freeze the SHA-256 for {path} in the EdNet manifest: {actual}"
        )
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 mismatch ({actual})")


def _question_metadata(contents: Path, member: str):
    with zipfile.ZipFile(contents) as archive:
        with archive.open(member) as raw:
            rows = list(csv.DictReader(io.TextIOWrapper(raw, "utf-8-sig")))
    bundle_counts = Counter(row["bundle_id"] for row in rows)
    return {
        row["question_id"]: row
        for row in rows
        if bundle_counts[row["bundle_id"]] == 1
        and row["correct_answer"] in {"a", "b", "c", "d"}
    }, bundle_counts


def _member_rows(archive: zipfile.ZipFile, member: str, questions: dict):
    dropped = Counter()
    valid = []
    with archive.open(member) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, "utf-8-sig"))
        required = {
            "timestamp",
            "solving_id",
            "question_id",
            "user_answer",
            "elapsed_time",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{member}: missing columns {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            question_id = row["question_id"]
            if question_id not in questions:
                dropped["non_singleton_or_unknown_question"] += 1
                continue
            answer = row["user_answer"]
            if answer not in {"a", "b", "c", "d"}:
                dropped["invalid_answer"] += 1
                continue
            try:
                timestamp = int(row["timestamp"])
                elapsed_ms = float(row["elapsed_time"])
            except (TypeError, ValueError):
                dropped["invalid_time"] += 1
                continue
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                dropped["nonpositive_time"] += 1
                continue
            if not row["solving_id"].strip():
                dropped["missing_solving_id"] += 1
                continue
            correct = int(answer == questions[question_id]["correct_answer"])
            valid.append(
                (timestamp, row_number, question_id, correct, elapsed_ms)
            )
    valid.sort(key=lambda item: (item[0], item[1]))
    return valid, dropped


def prepare_ednet(manifest_path: Path, *, download: bool = False) -> dict:
    manifest_path = manifest_path.resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported EdNet manifest schema")
    dataset = raw["dataset"]
    prep = raw["preparation"]
    source = _resolve(manifest_path, dataset["source_path"])
    contents = _resolve(manifest_path, dataset["contents_path"])
    prepared = _resolve(manifest_path, dataset["prepared_path"])
    _verified_file(
        source,
        dataset["source_url"],
        dataset["source_sha256"],
        download,
    )
    _verified_file(
        contents,
        dataset["contents_url"],
        dataset["contents_sha256"],
        download,
    )
    questions, bundle_counts = _question_metadata(
        contents, dataset["questions_member"]
    )
    n_students = int(prep["n_students"])
    min_responses = int(prep["min_responses"])
    max_len = int(prep["max_len"])
    selected = []
    dropped = Counter()
    files_scanned = 0
    raw_rows_scanned = 0
    with zipfile.ZipFile(source) as archive:
        members = []
        for member in archive.namelist():
            match = USER_MEMBER.search(member)
            if match:
                members.append((int(match.group(1)), member))
        members.sort()
        for user_number, member in members:
            files_scanned += 1
            rows, member_dropped = _member_rows(archive, member, questions)
            raw_rows_scanned += len(rows) + sum(member_dropped.values())
            dropped.update(member_dropped)
            if len(rows) < min_responses:
                dropped["users_below_min_responses"] += 1
                continue
            selected.append((f"u{user_number}", rows[:max_len]))
            if len(selected) == n_students:
                break
    if len(selected) != n_students:
        raise ValueError(
            f"only {len(selected)} users meet the frozen cohort rule; "
            f"expected {n_students}"
        )

    prepared.parent.mkdir(parents=True, exist_ok=True)
    temporary = prepared.with_name(prepared.name + ".tmp")
    all_times = []
    with temporary.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(
            [
                "user_id",
                "item_id",
                "timestamp",
                "correct",
                "skill_id",
                "elapsed_time_ms",
            ]
        )
        for user_id, rows in selected:
            for timestamp, _line, question_id, correct, elapsed_ms in rows:
                writer.writerow(
                    [
                        user_id,
                        question_id,
                        timestamp,
                        correct,
                        question_id,
                        format(elapsed_ms, ".12g"),
                    ]
                )
                all_times.append(elapsed_ms)
    temporary.replace(prepared)

    lo_ms, hi_ms = [
        float(value) * 1000 for value in prep["rt_clip_seconds"]
    ]
    receipt = {
        "schema_version": 1,
        "dataset_id": dataset["id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "contents_path": str(contents),
        "contents_sha256": _sha256(contents),
        "prepared_path": str(prepared),
        "prepared_sha256": _sha256(prepared),
        "files_scanned": files_scanned,
        "raw_rows_scanned": raw_rows_scanned,
        "n_students": len(selected),
        "n_rows": sum(len(rows) for _, rows in selected),
        "first_user": selected[0][0],
        "last_user": selected[-1][0],
        "n_questions": sum(bundle_counts.values()),
        "n_bundles": len(bundle_counts),
        "n_singleton_questions": len(questions),
        "n_times_clipped_low": sum(value < lo_ms for value in all_times),
        "n_times_clipped_high": sum(value > hi_ms for value in all_times),
        "dropped": dict(sorted(dropped.items())),
    }
    _write_json_atomic(
        prepared.with_suffix(prepared.suffix + ".provenance.json"), receipt
    )
    print(
        f"[ednet] {receipt['n_students']} users, {receipt['n_rows']} rows "
        f"-> {prepared}"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prepare_ednet(Path(args.manifest), download=args.download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
