#!/usr/bin/env python
"""Download/verify and ingest the frozen stable-speed chess cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from gps.data.ingest import run_ingest

DEFAULT_MANIFEST = Path(__file__).with_name("stable_speed_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _download_prefix(url: str, path: Path, n_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    request = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{n_bytes - 1}"}
    )
    try:
        with (
            urllib.request.urlopen(request) as response,
            temporary.open("wb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if temporary.stat().st_size != n_bytes:
            raise ValueError(
                f"{url}: expected {n_bytes} bytes, got "
                f"{temporary.stat().st_size}"
            )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _resolve(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cohort", action="append", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported stable-speed manifest schema")
    manifest_sha = _sha256(manifest_path)
    requested = set(args.cohort or [item["id"] for item in raw["cohorts"]])
    cohorts = [item for item in raw["cohorts"] if item["id"] in requested]
    unknown = requested - {item["id"] for item in cohorts}
    if unknown:
        raise ValueError(f"unknown cohort(s): {sorted(unknown)}")

    n_bytes = int(raw["source_prefix_bytes"])
    for cohort in cohorts:
        source = _resolve(manifest_path, cohort["source_path"])
        out_dir = _resolve(manifest_path, cohort["out_dir"])
        if not source.exists():
            if not args.download:
                raise FileNotFoundError(
                    f"missing {source}; rerun with --download"
                )
            _download_prefix(cohort["url"], source, n_bytes)
        if source.stat().st_size != n_bytes:
            raise ValueError(
                f"{source}: expected exactly {n_bytes} bytes, "
                f"got {source.stat().st_size}"
            )
        source_sha = _sha256(source)
        if source_sha != cohort["source_sha256"]:
            raise ValueError(f"{source}: SHA-256 does not match the manifest")
        dataset_path = out_dir / "dataset.jsonl.gz"
        if dataset_path.exists() and not args.force:
            print(f"[skip] existing cohort {cohort['id']}: {dataset_path}")
            continue

        ingest = raw["ingest"]
        result = run_ingest(
            str(source),
            str(out_dir),
            speed=ingest["speed"],
            min_games=int(ingest["min_games"]),
            min_sessions=int(ingest["min_sessions"]),
            max_players=int(ingest["max_players"]),
            gap_threshold_seconds=float(ingest["gap_threshold_seconds"]),
            workers=int(ingest["workers"]),
            batch_size=int(ingest["batch_size"]),
            max_games_per_player=int(ingest["max_games_per_player"]),
        )
        result.update(
            {
                "protocol_manifest": str(manifest_path),
                "protocol_manifest_sha256": manifest_sha,
                "source_url": cohort["url"],
                "source_prefix_bytes": n_bytes,
                "source_sha256": source_sha,
            }
        )
        _write_json_atomic(out_dir / "manifest.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
