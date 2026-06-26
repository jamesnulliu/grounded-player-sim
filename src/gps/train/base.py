"""Trainer interface + dataset abstraction.

A trainer fits the parameters ``phi`` of a
:class:`~gps.latent.base.LatentStateInjector` (and optionally a backbone) so
that the simulator reproduces a player's real move+timing trajectories. The
dataset is a sequence of per-player, chronologically-ordered trajectories;
the trainer is responsible for respecting the temporal split (it must never
train on a player's test sessions).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from gps.interface import DecisionPoint
from gps.latent.base import LatentStateInjector, Observation
from gps.policy.base import PolicyBackbone


@dataclass
class Trajectory:
    """One player's ordered decisions + ground-truth observations."""

    player_id: str
    decisions: list[DecisionPoint]
    observations: list[Observation]


@dataclass
class TrajectoryDataset:
    """A collection of per-player trajectories in time order."""

    trajectories: list[Trajectory] = field(default_factory=list)

    def players(self) -> set[str]:
        return {t.player_id for t in self.trajectories}

    def __len__(self) -> int:
        return len(self.trajectories)


@dataclass
class TrainConfig:
    """Common training hyperparameters."""

    epochs: int = 3
    lr: float = 1e-4
    batch_size: int = 8
    # Whether to also update the backbone (True) or freeze it and train only
    # the latent injector (False -- the cheaper, more common setting).
    train_backbone: bool = False
    grad_clip: float = 1.0
    seed: int = 0
    # Logical experiment name; becomes the result-store sub-directory and the
    # W&B group (e.g. "E-C2", "E-B1"). Defaults to the trainer class name.
    experiment: str | None = None
    # Root of the local result store (git-ignored). See gps.results.
    results_root: str = "runs"
    # Optional W&B project/entity overrides (else read from env / default).
    wandb_project: str | None = None
    wandb_entity: str | None = None
    out_dir: str = "checkpoints"
    extra: dict = field(default_factory=dict)


class Trainer(abc.ABC):
    """Fits injector (and optionally backbone) to trajectories.

    All trainers share three enforced rules, applied by :meth:`begin_run`
    which every concrete ``fit`` must call before doing work:

    * results are written under the result-store layout (:mod:`gps.results`);
    * the run is tracked in Weights & Biases -- mandatory, no opt-out
      (:mod:`gps.tracking`); a missing ``WANDB_API_KEY`` aborts the run;
    * the backend policy is validated -- LLM training must use slime + sglang
      (:mod:`gps.backends`).
    """

    def __init__(
        self,
        injector: LatentStateInjector,
        backbone: PolicyBackbone,
        config: TrainConfig | None = None,
    ) -> None:
        self.injector = injector
        self.backbone = backbone
        self.config = config or TrainConfig()

    @property
    def experiment_name(self) -> str:
        return self.config.experiment or type(self).__name__

    def run_config(self) -> dict:
        """The resolved config dict recorded to the result store + W&B."""
        from dataclasses import asdict

        cfg = asdict(self.config)
        cfg.update(
            {
                "trainer": type(self).__name__,
                "injector": type(self.injector).__name__,
                "backbone": self.backbone.name,
            }
        )
        return cfg

    def begin_run(self, dataset: TrajectoryDataset):
        """Enforce the three rules and open a tracked run.

        Returns ``(run_handle, wandb_run)``. Order matters: the backend rule
        and the mandatory W&B key are checked *before* a local run dir is
        created, so a misconfigured job fails fast and leaves no orphan dir.
        Concrete ``fit`` implementations call this first, then stream metrics
        via ``wandb_run.log(...)`` / ``wandb_run.summary(...)`` and end with
        ``wandb_run.finish(...)`` + ``run_handle.finalize(...)``.
        """
        from gps.backends import assert_llm_training_uses_slime
        from gps.results import ResultStore
        from gps.tracking import require_wandb

        # 1. Backend policy (LLM training -> slime+sglang) -- cheap, fail fast.
        assert_llm_training_uses_slime(
            self.backbone, train_backbone=self.config.train_backbone
        )
        # 2. Mandatory tracking key check happens inside require_wandb, but do
        #    the key-presence check first so we never make a run dir without a
        #    key. (require_wandb re-checks; this keeps ordering explicit.)
        from gps.tracking import require_wandb_key

        require_wandb_key()

        config = self.run_config()
        handle = ResultStore(self.config.results_root).create(
            self.experiment_name, config
        )
        wandb_run = require_wandb(
            experiment=self.experiment_name,
            config=config,
            handle=handle,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
        )
        return handle, wandb_run

    @abc.abstractmethod
    def fit(self, dataset: TrajectoryDataset) -> dict:
        """Train; return a metrics/summary dict. Must not touch test data."""

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist trained parameters."""
