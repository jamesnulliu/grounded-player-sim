"""The result-store rule: one directory layout + JSON format for every run.

Every experiment and training run -- Phase-0 on a laptop, a board-native
chess run on one GPU, the LLM headline on a cluster -- writes its outputs the
*same* way, so results are comparable, scriptable, and never lost in stdout.
This module is the single source of truth for that layout. It is pure stdlib
(no numpy, no torch, no wandb) so it imports and runs anywhere the rest of the
CPU path does, and so it can be unit-tested without a GPU.

Directory layout
----------------
A run lives at ``<root>/<experiment>/<run_id>/`` (``root`` defaults to
``runs/``, which is git-ignored like ``checkpoints/`` / ``data/``)::

    runs/
      <experiment>/                 # logical experiment name, e.g. "E-C2"
        <run_id>/                   # <UTC-timestamp>__<experiment>__<slug>
          run.json                  # run metadata (id, git sha, host, status,
                                    #   wandb url, schema_version)
          config.json               # the *resolved* config dict for this run
          env.json                  # python + package versions, hostname
          metrics.json              # final/summary metrics (the headline #s)
          metrics.jsonl             # append-only per-step / per-epoch stream
          artifacts/                # checkpoints, plots, per-player CSVs, ...

Why this shape
--------------
* ``run.json`` / ``config.json`` / ``env.json`` make a run **reproducible**:
  what was run, on what code (git sha + dirty flag), in what environment.
* ``metrics.jsonl`` is append-only so a long run streams progress without
  rewriting a file; ``metrics.json`` holds the *final* numbers a results
  table reads. Keeping the stream and the summary in separate files means a
  crashed run still leaves a readable partial ``metrics.jsonl``.
* The ``run_id`` sorts chronologically (timestamp first) and is unique, so
  ``ls`` over an experiment dir is already time-ordered.

This store records results locally; :mod:`gps.tracking` mirrors training runs
to Weights & Biases (mandatory for training). The two are wired together by
:meth:`RunHandle.attach_wandb`, which writes the W&B run id/url into
``run.json`` so a local result dir always points back to its W&B run.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Bump when the on-disk record format changes incompatibly.
SCHEMA_VERSION = "1"

#: Default root for all runs. Git-ignored (see .gitignore).
DEFAULT_ROOT = "runs"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(dt: datetime) -> str:
    """Sortable UTC timestamp, e.g. ``20260624-141203-512`` (ms precision)."""
    return dt.strftime("%Y%m%d-%H%M%S-") + f"{dt.microsecond // 1000:03d}"


def _slugify(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in text.lower()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def _git_info() -> dict:
    """Best-effort git sha + dirty flag; never raises."""
    info: dict = {"sha": None, "dirty": None}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if sha.returncode == 0:
            info["sha"] = sha.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _package_versions() -> dict:
    """Versions of relevant packages *without importing* them (cheap, safe)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict = {}
    for pkg in (
        "grounded-player-sim",
        "numpy",
        "torch",
        "sglang",
        "slime",
        "wandb",
        "transformers",
        "python-chess",
        "openai",
        "anthropic",
    ):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


@dataclass
class RunHandle:
    """A handle to one run's directory; the only object callers write through.

    Construct via :meth:`ResultStore.create`. Methods are idempotent-friendly:
    ``set_summary`` merges, ``log_metrics`` appends, ``finalize`` is safe to
    call once at the end (or in a ``finally``).
    """

    run_id: str
    experiment: str
    dir: Path
    created_at: str
    _metadata: dict = field(default_factory=dict)

    # --- paths ----------------------------------------------------------
    @property
    def artifacts_dir(self) -> Path:
        d = self.dir / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def artifact_path(self, name: str) -> Path:
        """Path under ``artifacts/`` for a checkpoint/plot/table to write."""
        return self.artifacts_dir / name

    # --- writing --------------------------------------------------------
    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Append one record to ``metrics.jsonl`` (per-step / per-epoch).

        ``metrics`` must be JSON-serialisable scalars. ``step`` is stamped in
        when given so a stream can be plotted against it.
        """
        record = {"step": step, **metrics}
        with (self.dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def set_summary(self, metrics: dict) -> None:
        """Merge ``metrics`` into ``metrics.json`` (final headline numbers)."""
        path = self.dir / "metrics.json"
        current: dict = {}
        if path.exists():
            current = json.loads(path.read_text())
        current.update(metrics)
        _write_json(path, current)

    def attach_wandb(self, *, run_id: str, url: str | None = None) -> None:
        """Record the W&B run this local dir mirrors, into ``run.json``."""
        self._metadata["wandb"] = {"run_id": run_id, "url": url}
        self._flush_metadata()

    def finalize(self, status: str = "completed", **extra: object) -> None:
        """Stamp terminal status + finish time into ``run.json``.

        ``status`` is typically ``"completed"`` or ``"failed"``. Extra
        key/values (e.g. an error string) are recorded alongside.
        """
        self._metadata["status"] = status
        self._metadata["finished_at"] = _timestamp(_utc_now())
        self._metadata.update(extra)
        self._flush_metadata()

    def _flush_metadata(self) -> None:
        _write_json(self.dir / "run.json", self._metadata)


class ResultStore:
    """Creates run directories under a root following the result-store rule."""

    def __init__(self, root: str | os.PathLike = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def create(
        self,
        experiment: str,
        config: dict,
        *,
        run_id: str | None = None,
        tags: list[str] | None = None,
    ) -> RunHandle:
        """Create ``<root>/<experiment>/<run_id>/`` and write the manifest.

        Writes ``run.json``, ``config.json``, and ``env.json`` immediately so
        even a run that dies on the first step leaves a reproducible record.
        Returns a :class:`RunHandle` to stream metrics + artifacts through.
        """
        now = _utc_now()
        exp = _slugify(experiment)
        if run_id is None:
            run_id = (
                f"{_timestamp(now)}__{exp}__{_slugify(_short_hash(config))}"
            )
        run_dir = self.root / exp / run_id
        if run_dir.exists():
            raise FileExistsError(f"run dir already exists: {run_dir}")
        run_dir.mkdir(parents=True)

        created_at = _timestamp(now)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "experiment": experiment,
            "created_at": created_at,
            "status": "running",
            "tags": tags or [],
            "git": _git_info(),
            "hostname": socket.gethostname(),
            "wandb": None,
        }
        handle = RunHandle(
            run_id=run_id,
            experiment=experiment,
            dir=run_dir,
            created_at=created_at,
            _metadata=metadata,
        )
        handle._flush_metadata()
        _write_json(run_dir / "config.json", config)
        _write_json(
            run_dir / "env.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _package_versions(),
            },
        )
        return handle


def _short_hash(obj: object) -> str:
    """Stable 8-hex digest of a config, for the run_id slug."""
    import hashlib

    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]
