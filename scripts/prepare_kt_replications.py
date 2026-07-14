#!/usr/bin/env python
"""Prepare and fingerprint the eight canonical real-KT exports.

Raw/licensed datasets stay under the git-ignored data directory. This script
resolves sources and outputs from the replication manifest and writes a
provenance receipt beside each prepared TSV.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from gps.experiments.kt_replication import (
    file_sha256,
    inspect_kt_export,
    load_replication_manifest,
    provenance_path,
)
from prepare_kt_data import prepare_assistments09, prepare_standard_kt

DEFAULT_MANIFEST = Path(__file__).with_name("kt_replication_manifest.json")


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="dataset id; repeat as needed (default: all)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_replication_manifest(args.manifest)
    selected = manifest.select(args.dataset)
    missing = [
        spec.source_path for spec in selected if not spec.source_path.exists()
    ]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"missing KT source exports:\n{paths}")

    for spec in selected:
        receipt_path = provenance_path(spec.prepared_path)
        if (
            spec.prepared_path.exists() or receipt_path.exists()
        ) and not args.force:
            raise SystemExit(
                f"refusing to overwrite {spec.prepared_path}; pass --force"
            )
        spec.prepared_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = spec.prepared_path.with_name(
            spec.prepared_path.name + ".tmp"
        )
        try:
            if spec.preparer == "assistments09_raw":
                prepare_assistments09(str(spec.source_path), str(temporary))
            elif spec.preparer == "standard_5col":
                prepare_standard_kt(
                    str(spec.source_path),
                    str(temporary),
                    delimiter=spec.source_delimiter,
                )
            else:
                raise ValueError(
                    f"{spec.dataset_id}: unsupported preparer "
                    f"{spec.preparer!r}"
                )

            stats = inspect_kt_export(
                temporary,
                min_responses=manifest.protocol.min_responses,
                n_students=spec.n_students,
                max_len=manifest.protocol.max_len,
            )
            if stats["n_selected_users"] != spec.n_students:
                raise ValueError(
                    f"{spec.dataset_id}: requested {spec.n_students} students "
                    f"but only {stats['n_selected_users']} are eligible"
                )
            temporary.replace(spec.prepared_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        receipt = {
            "schema_version": 1,
            "dataset_id": spec.dataset_id,
            "label": spec.label,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest.path),
            "manifest_sha256": manifest.sha256,
            "preparer": spec.preparer,
            "source_path": str(spec.source_path),
            "source_reference": spec.source_reference,
            "source_sha256": file_sha256(spec.source_path),
            "prepared_path": str(spec.prepared_path),
            "prepared_sha256": file_sha256(spec.prepared_path),
            "stats": stats,
            "provenance_note": spec.provenance_note,
        }
        _write_json_atomic(receipt_path, receipt)
        print(
            f"[prepared] {spec.dataset_id}: {stats['n_rows']} rows, "
            f"{stats['n_selected_users']} selected -> {spec.prepared_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
