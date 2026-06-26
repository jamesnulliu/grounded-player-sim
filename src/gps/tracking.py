"""Mandatory experiment tracking via Weights & Biases.

**Policy (enforced in code, not prose):** every *training* run logs to W&B.
There is no opt-out flag. The single way in is :func:`require_wandb`, which:

1. reads ``WANDB_API_KEY`` from the environment and **raises**
   :class:`TrackingError` if it is missing or empty -- so a training run
   aborts immediately with a clear message instead of silently running
   untracked;
2. imports ``wandb`` (raising an install hint if the package is absent);
3. logs in non-interactively with that key and starts a run.

Why hard-fail rather than warn: an untracked training run is effectively lost
work -- you cannot compare it, resume reasoning about it, or trust a number
that has no provenance. Making the key mandatory means "it ran" implies "it
is tracked".

This module is import-safe on a laptop: ``wandb`` is imported lazily inside
:func:`require_wandb`, never at module load, so the CPU path
(``gps phase0`` etc.) does not need it installed. The trainers
(:mod:`gps.train`) call :func:`require_wandb` at the top of ``fit`` so the
failure happens before any expensive setup.

The returned :class:`WandbRun` is a thin wrapper that also mirrors metrics
into the local :class:`~gps.results.RunHandle`, so the two stores never drift:
one call logs to both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from gps.results import RunHandle

WANDB_API_KEY_ENV = "WANDB_API_KEY"
#: Optional overrides; project/entity can also be passed explicitly.
WANDB_PROJECT_ENV = "WANDB_PROJECT"
WANDB_ENTITY_ENV = "WANDB_ENTITY"

DEFAULT_PROJECT = "grounded-player-sim"


class TrackingError(RuntimeError):
    """Raised when mandatory tracking cannot be satisfied (e.g. no API key)."""


@dataclass
class WandbRun:
    """Wrapper over a live ``wandb`` run, mirroring to the local result store.

    Use :meth:`log` for streaming metrics (goes to both W&B and the local
    ``metrics.jsonl``) and :meth:`summary` for final headline numbers (both
    W&B run summary and the local ``metrics.json``).
    """

    run: object  # the wandb.sdk.wandb_run.Run
    handle: RunHandle | None = None

    @property
    def id(self) -> str:
        return self.run.id

    @property
    def url(self) -> str | None:
        getter = getattr(self.run, "get_url", None)
        return getter() if callable(getter) else getattr(self.run, "url", None)

    def log(self, metrics: dict, step: int | None = None) -> None:
        self.run.log(metrics, step=step)
        if self.handle is not None:
            self.handle.log_metrics(metrics, step=step)

    def summary(self, metrics: dict) -> None:
        for k, v in metrics.items():
            self.run.summary[k] = v
        if self.handle is not None:
            self.handle.set_summary(metrics)

    def finish(self, status: str = "completed") -> None:
        exit_code = 0 if status == "completed" else 1
        try:
            self.run.finish(exit_code=exit_code)
        except TypeError:  # older wandb signatures
            self.run.finish()


def require_wandb_key() -> str:
    """Return the W&B API key from the env, or raise :class:`TrackingError`.

    Separated from :func:`require_wandb` so callers (and tests) can assert the
    key-presence rule without importing or contacting ``wandb``.
    """
    key = os.environ.get(WANDB_API_KEY_ENV, "").strip()
    if not key:
        raise TrackingError(
            f"{WANDB_API_KEY_ENV} is not set. Training runs must log to "
            "Weights & Biases; there is no opt-out. Export your key first:\n"
            f"    export {WANDB_API_KEY_ENV}=<your-key>\n"
            "(get it from https://wandb.ai/authorize). To override the "
            f"project/entity, set {WANDB_PROJECT_ENV} / {WANDB_ENTITY_ENV}."
        )
    return key


def require_wandb(
    *,
    experiment: str,
    config: dict,
    handle: RunHandle | None = None,
    project: str | None = None,
    entity: str | None = None,
    tags: list[str] | None = None,
) -> WandbRun:
    """Enforce the tracking policy and start a W&B run.

    Raises :class:`TrackingError` if ``WANDB_API_KEY`` is unset, or an
    ``ImportError`` (with an install hint) if ``wandb`` is not installed.
    When a local ``handle`` is supplied, the W&B run id/url is written back
    into the run's ``run.json`` so the local dir points at its W&B run.
    """
    key = require_wandb_key()

    try:
        import wandb
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "wandb is required for training runs but is not installed. "
            "Install it on the training host: pip install wandb"
        ) from e

    # Non-interactive login with the env key (never prompts).
    wandb.login(key=key, relogin=False, verify=False)

    run = wandb.init(
        project=project or os.environ.get(WANDB_PROJECT_ENV, DEFAULT_PROJECT),
        entity=entity or os.environ.get(WANDB_ENTITY_ENV) or None,
        name=handle.run_id if handle is not None else None,
        group=experiment,
        tags=tags or [],
        config=config,
        # Reuse the local run dir so wandb's own files land beside ours.
        dir=str(handle.dir) if handle is not None else None,
    )
    wrapped = WandbRun(run=run, handle=handle)
    if handle is not None:
        handle.attach_wandb(run_id=wrapped.id, url=wrapped.url)
    return wrapped
