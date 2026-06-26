"""RL trainer via slime (behaviour-matching / agent-fits-latent path).

Use this when the objective is a *reward* rather than per-move likelihood:

* matching distributional rollout statistics (aggression, error-by-phase,
  time-allocation) to the target player -- a behaviour-matching reward;
* the stretch opponent-preparation loop (proposal section 5, Phase 4.7);
* fine-tuning the open-weight agent to *fit* the injected latent (you said
  you might "train the agent to fit that").

slime is an LLM post-training (RL) framework that pairs a training backend
with an sglang rollout engine. This trainer owns the glue: it defines the
rollout (the simulator producing trajectories), the reward (behaviour-match
to the target player), and hands them to slime. slime + torch are imported
lazily; the GPU host pins the slime install.
"""

from __future__ import annotations

from collections.abc import Callable

from gps.train.base import Trainer, TrajectoryDataset


class SlimeRLTrainer(Trainer):
    """RL fine-tuning of the agent/injector against a behaviour reward."""

    def __init__(
        self,
        injector,
        backbone,
        config=None,
        reward_fn: Callable | None = None,
    ) -> None:
        super().__init__(injector, backbone, config)
        # reward_fn(target_trajectory, rollout) -> float. Defaults to a
        # distribution-match reward defined in this module.
        self.reward_fn = reward_fn or behavior_match_reward

    def fit(self, dataset: TrajectoryDataset) -> dict:
        # Enforce result store + mandatory W&B + backend policy first. For a
        # served-LLM backbone this also asserts slime+sglang are present; the
        # explicit slime import below keeps the message specific to the RL
        # path even when the backbone is non-LLM (e.g. board-native RL).
        handle, wandb_run = self.begin_run(dataset)

        try:
            import slime  # noqa: F401
        except ImportError as e:  # pragma: no cover - env-dependent
            wandb_run.finish(status="failed")
            handle.finalize(status="failed", error="slime not installed")
            raise ImportError(
                "slime is required for SlimeRLTrainer; install it from "
                "source on the GPU host (see documents/training.md)."
            ) from e
        # TODO(gpu): configure slime with:
        #   - rollout engine = sglang serving self.backbone
        #   - rollout fn      = Simulator(injector, backbone).run_trajectory
        #   - reward          = self.reward_fn vs. the target player's games
        #   - PPO/GRPO config from self.config
        raise NotImplementedError(
            "slime RL wiring is finalised on the GPU host; the rollout + "
            "reward contract is defined here."
        )

    def save(self, path: str) -> None:
        raise NotImplementedError


def behavior_match_reward(target_trajectory, rollout) -> float:
    """Reward = negative distance between rollout and target behaviour.

    Placeholder contract: compares aggregate statistics (move-quality
    distribution, time-allocation profile, error-by-phase) between a rollout
    and the target player's real games, returning higher reward for closer
    matches. The concrete distance (KL/JS/Wasserstein over the chosen
    statistics, per proposal Phase 4.4) is implemented alongside the GPU
    rollout. Defined here so the RL contract is explicit.
    """
    raise NotImplementedError(
        "behavior_match_reward is specified; concrete statistic-distance is "
        "implemented with the GPU rollout."
    )
