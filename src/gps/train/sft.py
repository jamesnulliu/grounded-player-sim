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

    Two modes, both scoring move-NLL on a player's *later* games (the latent is
    warmed over the whole session, no grad, so the number measures prediction
    of *future* behaviour, not memorised history):

    * **Global window** (``window``): a single ``(start, stop)`` step range,
      valid only when every trajectory has the same length (the Phase-0 / E-A1
      synthetic sessions). The train ``dataset`` passed to ``fit`` is the
      *training prefix*; ``dataset`` here is the *full* sessions.
    * **Per-player split** (``splits``): one boundary index per trajectory,
      for **variable-length real data** (E-C). The ``dataset`` passed to
      ``fit`` is the *full* sessions; the trainer builds train/eval **masks**
      from ``splits`` (``backbone.train_eval_masks``) so training never sees a
      player's held-out tail and eval scores only that tail. Requires a
      mask-aware backbone
      (:class:`~gps.policy.board_native.BoardNativeBackbone`).

    Exactly one of ``window`` / ``splits`` is used; ``splits`` wins if set.
    """

    dataset: TrajectoryDataset
    window: tuple[int, int] = (0, 0)
    splits: list[int] | None = None


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

        # Per-player masked split (E-C, variable-length real data) goes through
        # the minibatched path so it scales past the full-batch wall (the
        # full-batch loop pads to the longest trajectory and runs out of room
        # at ~40 chess players). The global-window path (E-A1 / Phase-0,
        # equal-length synthetic sessions) is small and stays full-batch below,
        # byte-for-byte unchanged.
        if eval_spec is not None and eval_spec.splits is not None:
            return self._train_masked_minibatched(
                torch, dataset, eval_spec, opt, lam, handle, wandb_run, params
            )

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
            latent = self._latent(train_batch)
            out = self._loss(train_batch, latent, lam, None)
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

    def _train_masked_minibatched(
        self, torch, dataset, eval_spec, opt, lam, handle, wandb_run, params
    ):
        """Masked per-player split, minibatched over players (E-C, scalable).

        Players are length-sorted and chunked into ``config.batch_size`` groups
        that are **encoded once** (so each minibatch pads only to *its* longest
        trajectory, not the global longest -- the whole reason this scales).
        Each epoch shuffles the minibatch visit order (seeded) and takes one
        optimizer step per minibatch (minibatch SGD). Both arms D/B run the
        identical schedule, so the comparison stays paired.
        """
        import random

        device = next(self.backbone.parameters()).device
        mbs = self._encode_masked_minibatches(
            dataset.trajectories, eval_spec.splits, device
        )
        rng = random.Random(self.config.seed)

        final: dict = {}
        for epoch in range(self.config.epochs):
            self._set_train(True)
            order = list(range(len(mbs)))
            rng.shuffle(order)
            loss_w = move_w = time_w = tr_steps = 0.0
            for i in order:
                mb = mbs[i]
                latent = self._latent(mb["batch"])
                out = self.backbone.trajectory_loss(
                    latent, mb["batch"], lam, step_mask=mb["train_mask"]
                )
                opt.zero_grad()
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
                opt.step()
                w = mb["n_train_steps"]
                loss_w += out["loss"].item() * w
                move_w += out["move_nll"].item() * w
                time_w += out["timing_nll"].item() * w
                tr_steps += w

            self._set_train(False)
            ev_w = ev_steps = 0.0
            with torch.no_grad():
                for mb in mbs:
                    latent = self._latent(mb["batch"])
                    o = self.backbone.trajectory_loss(
                        latent, mb["batch"], lam=0.0, step_mask=mb["eval_mask"]
                    )
                    ev_w += o["move_nll"].item() * mb["n_eval_steps"]
                    ev_steps += mb["n_eval_steps"]

            tr = max(tr_steps, 1.0)
            metrics = {
                "loss": loss_w / tr,
                "move_nll": move_w / tr,
                "timing_nll": time_w / tr,
                "val_move_nll": ev_w / max(ev_steps, 1.0),
            }
            wandb_run.log(metrics, step=epoch)
            final = metrics

        summary = {
            "status": "completed",
            "epochs": self.config.epochs,
            "n_minibatches": len(mbs),
            **final,
        }
        summary.update(self._param_report())
        self._save_checkpoint(torch, handle)
        return summary

    def _encode_masked_minibatches(self, trajectories, splits, device):
        """Length-sorted, pre-encoded minibatches of (batch, train/eval masks).

        Length-sorting keeps each minibatch's padding tight; encoding once and
        caching avoids re-tensorizing FENs every epoch.
        """
        bs = max(1, self.config.batch_size)
        pairs = sorted(
            zip(trajectories, splits),
            key=lambda ts: len(ts[0].decisions),
            reverse=True,
        )
        mbs = []
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            trajs = [t for t, _ in chunk]
            sp = [s for _, s in chunk]
            batch = self.backbone.encode_batch(trajs).to(device)
            tmask, emask = self.backbone.train_eval_masks(batch, sp)
            mbs.append(
                {
                    "batch": batch,
                    "train_mask": tmask,
                    "eval_mask": emask,
                    "n_train_steps": float(tmask.sum().item()),
                    "n_eval_steps": float(emask.sum().item()),
                }
            )
        return mbs

    # --- helpers -------------------------------------------------------
    def _latent(self, batch):
        """Injector latent for a batch, forwarding identity when present.

        Board batches carry ``player_ids`` (column order); identity-keyed
        injectors (static-individual) use them, the GRU injector ignores them.
        Window-path batches (E-A1) have no such field -> ``None``.
        """
        return self.injector.latent_trajectory(
            batch.feats, player_ids=getattr(batch, "player_ids", None) or None
        )

    def _loss(self, batch, latent, lam, train_mask):
        """Train loss, masking to the train steps when a mask is given.

        ``train_mask is None`` keeps the global-window backbone signature
        (DiffMovePolicy) untouched; a mask routes through the mask-aware
        board-native ``trajectory_loss``.
        """
        if train_mask is None:
            return self.backbone.trajectory_loss(latent, batch, lam)
        return self.backbone.trajectory_loss(
            latent, batch, lam, step_mask=train_mask
        )

    def _eval_move_nll(self, torch, batch, window) -> float:
        a, b = window
        self._set_train(False)
        with torch.no_grad():
            latent = self._latent(batch)
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
