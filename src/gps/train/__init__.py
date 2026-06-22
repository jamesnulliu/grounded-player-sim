"""Trainers for the latent injector (and optionally the backbone).

Two paths behind one :class:`~gps.train.base.Trainer` interface:

* :class:`~gps.train.sft.SFTTrainer` -- supervised fine-tuning. "Imitate
  this player's observed moves/timing" is naturally a maximum-likelihood
  problem, so SFT is the default path for fitting both the latent injector
  and (optionally) a backbone to reproduce real trajectories.
* :class:`~gps.train.slime_rl.SlimeRLTrainer` -- RL via slime. Used when the
  objective is a *behaviour-matching reward* rather than per-move likelihood
  (e.g. matching distributional/rollout statistics, or the stretch
  opponent-preparation loop), or to fine-tune the agent to *fit* the latent.

Both lazily import torch / slime so this package imports on a CPU box.
"""

from gps.train.base import TrainConfig, Trainer, TrajectoryDataset

__all__ = ["TrainConfig", "Trainer", "TrajectoryDataset"]
