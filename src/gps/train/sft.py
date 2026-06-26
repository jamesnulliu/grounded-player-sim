"""Supervised fine-tuning trainer (default path).

Maximises the likelihood of each player's observed moves (and think-times)
under the simulator, w.r.t. the latent injector's parameters ``phi`` (and
optionally the backbone). Concretely, teacher-force the latent along each
trajectory and minimise::

    sum_t  -log P(move_t | state_t, player, z_t)
           - lambda * log p(time_t | state_t, player, z_t)

torch-backed and GPU-friendly but the loss is small; lazily imported so the
module loads on CPU. The training *loop* skeleton is written out (it is
plain orchestration); the per-step tensor ops are marked where they bind to
a concrete differentiable injector/backbone.
"""

from __future__ import annotations

from gps.train.base import Trainer, TrajectoryDataset


class SFTTrainer(Trainer):
    """Maximum-likelihood imitation of real trajectories."""

    def fit(self, dataset: TrajectoryDataset) -> dict:
        try:
            import torch
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "torch required for SFTTrainer; install the 'train' extra "
                "on a GPU host: pip install '.[train]'"
            ) from e

        # Enforce the three project rules (result store + mandatory W&B +
        # backend policy) and open a tracked run before any training work.
        # This is what raises if WANDB_API_KEY is unset, or if a served-LLM
        # backbone is used without slime+sglang.
        handle, wandb_run = self.begin_run(dataset)

        # The loop below is the intended orchestration. It runs only once the
        # injector/backbone expose differentiable forward passes (the neural
        # injector variants + an open-weight backbone). The structured
        # reference injector has no parameters, so SFT over it is a no-op and
        # we surface that explicitly rather than silently "succeeding".
        if not list(self._trainable_parameters(torch)):
            summary = {
                "status": "no-op",
                "reason": (
                    "no trainable parameters; the structured reference "
                    "injector is parameter-free. Use a neural injector "
                    "variant to exercise SFT."
                ),
                "n_trajectories": len(dataset),
            }
            wandb_run.summary(summary)
            wandb_run.finish(status="completed")
            handle.finalize(status="completed", note="no-op (parameter-free)")
            return summary

        # TODO(gpu): real loop -- stream per-epoch metrics with
        #   wandb_run.log({"loss": float(loss), "move_nll": ...}, step=epoch)
        # and the final headline numbers with wandb_run.summary({...}); end
        # with wandb_run.finish() + handle.finalize(). Checkpoints go to
        # handle.artifact_path("injector.pt").
        #   opt = torch.optim.AdamW(params, lr=self.config.lr)
        #   for epoch in range(self.config.epochs):
        #     for batch in self._batches(dataset):
        #       z = injector.initial_state(...)
        #       loss = 0
        #       for dp, obs in trajectory:
        #         inj = injector.render(z, dp)       # differentiable
        #         pred = backbone.predict(dp, inj)   # differentiable logits
        #         loss += move_nll(pred, obs) + lam * timing_nll(pred, obs)
        #         z = injector.update(z, dp, obs)
        #       loss.backward(); clip; opt.step(); opt.zero_grad()
        raise NotImplementedError(
            "SFT tensor loop binds to a differentiable injector/backbone on "
            "the GPU host; orchestration + no-op guard are implemented."
        )

    def _trainable_parameters(self, torch):
        """Yield torch parameters from injector/backbone if any exist."""
        for obj in (self.injector, self.backbone):
            params = getattr(obj, "parameters", None)
            if callable(params):
                yield from params()

    def save(self, path: str) -> None:
        import importlib.util

        if importlib.util.find_spec("torch") is None:  # pragma: no cover
            raise ImportError("torch required to save")
        # TODO(gpu): torch.save of injector (+backbone) state_dicts.
        raise NotImplementedError
