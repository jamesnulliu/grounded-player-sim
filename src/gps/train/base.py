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
    out_dir: str = "checkpoints"
    extra: dict = field(default_factory=dict)


class Trainer(abc.ABC):
    """Fits injector (and optionally backbone) to trajectories."""

    def __init__(
        self,
        injector: LatentStateInjector,
        backbone: PolicyBackbone,
        config: TrainConfig | None = None,
    ) -> None:
        self.injector = injector
        self.backbone = backbone
        self.config = config or TrainConfig()

    @abc.abstractmethod
    def fit(self, dataset: TrajectoryDataset) -> dict:
        """Train; return a metrics/summary dict. Must not touch test data."""

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist trained parameters."""
