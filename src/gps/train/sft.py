"""Supervised fine-tuning trainer (default path).

Maximises the likelihood of each player's observed moves (and think-times)
under the simulator, w.r.t. the latent injector's parameters ``phi`` (and the
differentiable backbone head). Concretely, teacher-force the latent along each
trajectory and minimise::

    sum_t  -log P(move_t | state_t, player, z_t)
           - lambda * log p(time_t | state_t, player, z_t)

The loop is generic over two protocols so it stays game-agnostic:

* the **injector** exposes ``latent_trajectory(feats_seq) -> [T, B, L]`` (the
  differentiable recurrence; see :class:`~gps.latent.neural.NeuralInjector`);
* the **backbone** exposes ``encode_batch(trajectories) -> EncodedBatch``,
  ``trajectory_loss(latent_seq, batch, lam) -> {...}``, and ``parameters()``
  (see :class:`~gps.policy.diff_policy.DiffMovePolicy`).

A future board-native / LLM backbone implements the same three methods for
real chess and this loop is unchanged. torch is imported lazily so the module
loads on CPU; CPU is enough for the tiny Phase-0 models, a GPU only speeds it.
"""

from __future__ import annotations

from dataclasses import dataclass

from gps.train.base import Trainer, TrajectoryDataset


@dataclass
class EvalSpec:
    """Held-out evaluation for the strict temporal split (RQ3 in miniature).

    ``dataset`` holds the *full* per-player sessions; ``window`` is the
    ``(start, stop)`` step range to score -- the player's *later* games. The
    latent is warmed over the whole session (no grad) and the move-NLL is
    reported on the held-out tail, so the number measures prediction of
    *future* behaviour, not memorised history.
    """

    dataset: TrajectoryDataset
    window: tuple[int, int]


class SFTTrainer(Trainer):
    """Maximum-likelihood imitation of real trajectories."""

    def fit(
        self,
        dataset: TrajectoryDataset,
        eval_spec: EvalSpec | None = None,
    ) -> dict:
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

        # The structured reference injector has no parameters, so SFT over it
        # is a no-op; surface that explicitly rather than silently "succeed".
        params = list(self._trainable_parameters(torch))
        if not params:
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

        try:
            summary = self._train(
                torch, dataset, eval_spec, handle, wandb_run, params
            )
            wandb_run.summary(summary)
            wandb_run.finish(status="completed")
            handle.finalize(status="completed")
            return summary
        except Exception as e:  # noqa: BLE001 - re-raised after recording
            wandb_run.finish(status="failed")
            handle.finalize(status="failed", error=repr(e))
            raise

    def _train(self, torch, dataset, eval_spec, handle, wandb_run, params):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Place both modules on the device (params are moved in place, so the
        # optimizer below still references the right tensors).
        self.injector.to(device)
        self.backbone.to(device)

        lam = float(self.config.extra.get("timing_lambda", 0.5))
        opt = torch.optim.AdamW(params, lr=self.config.lr)

        train_batch = self.backbone.encode_batch(dataset.trajectories).to(
            device
        )
        eval_batch = window = None
        if eval_spec is not None:
            eval_batch = self.backbone.encode_batch(
                eval_spec.dataset.trajectories
            ).to(device)
            window = eval_spec.window

        final: dict = {}
        for epoch in range(self.config.epochs):
            self._set_train(True)
            latent = self.injector.latent_trajectory(train_batch.feats)
            out = self.backbone.trajectory_loss(latent, train_batch, lam)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
            opt.step()

            metrics = {
                "loss": out["loss"].item(),
                "move_nll": out["move_nll"].item(),
                "timing_nll": out["timing_nll"].item(),
            }
            if eval_batch is not None:
                metrics["val_move_nll"] = self._eval_move_nll(
                    torch, eval_batch, window
                )
            wandb_run.log(metrics, step=epoch)
            final = metrics

        summary = {
            "status": "completed",
            "epochs": self.config.epochs,
            **final,
        }
        summary.update(self._param_report())
        self._save_checkpoint(torch, handle)
        return summary

    # --- helpers -------------------------------------------------------
    def _eval_move_nll(self, torch, batch, window) -> float:
        a, b = window
        self._set_train(False)
        with torch.no_grad():
            latent = self.injector.latent_trajectory(batch.feats)
            out = self.backbone.trajectory_loss(
                latent[a:b], batch.window(a, b), lam=0.0
            )
        return float(out["move_nll"])

    def _set_train(self, mode: bool) -> None:
        for obj in (self.injector, self.backbone):
            net = getattr(obj, "_net", None)
            if net is not None and hasattr(net, "train"):
                net.train(mode)

    def _param_report(self) -> dict:
        report: dict = {}
        for label, obj in (
            ("injector", self.injector),
            ("backbone", self.backbone),
        ):
            fn = getattr(obj, "parameters", None)
            if callable(fn):
                report[f"{label}_params"] = int(sum(p.numel() for p in fn()))
        report["total_params"] = sum(
            v for k, v in report.items() if k.endswith("_params")
        )
        return report

    def _save_checkpoint(self, torch, handle) -> None:
        state = {}
        for label, obj in (
            ("injector", self.injector),
            ("backbone", self.backbone),
        ):
            net = getattr(obj, "_net", None)
            if net is not None:
                state[label] = net.state_dict()
        torch.save(state, handle.artifact_path("checkpoint.pt"))

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
        import torch

        state = {}
        for label, obj in (
            ("injector", self.injector),
            ("backbone", self.backbone),
        ):
            net = getattr(obj, "_net", None)
            if net is not None:
                state[label] = net.state_dict()
        torch.save(state, path)
