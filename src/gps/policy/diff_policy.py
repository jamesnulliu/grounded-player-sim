"""A small *differentiable* policy backbone for training (Phase-0 / E-A1).

The mock backbone reacts to the latent but is not differentiable, so it cannot
be used to *fit* an injector. This module is the differentiable counterpart
used by :class:`~gps.train.sft.SFTTrainer`: it maps a latent vector plus the
per-move engine value-advantages to a move distribution and a think-time
distribution, with trainable parameters, so SFT can backprop a move-NLL +
timing-NLL loss into the injector (and this head).

The move model deliberately mirrors the synthetic data-generating process:

    move_logits = sharpness(latent) * value_advantage

where ``value_advantage`` is the engine value of each legal move normalised to
``[0, 1]`` within the position, and ``sharpness`` (an inverse temperature, like
the synthetic players' ``beta``) is a learned positive function of the latent.
A model that recovers the player's instantaneous ``beta`` from the injected
state therefore recovers the player's move distribution -- which is exactly the
quantity the dynamic latent is supposed to carry. Think-time is a log-normal
whose ``mu`` is a learned linear function of the latent (a learned scalar
``sigma``), matching :class:`~gps.prediction.TimingPrediction`.

The backbone owns *encoding* (raw trajectories -> padded tensors) and the
*loss* so that :class:`SFTTrainer` stays game-agnostic: a future board-native
or LLM backbone implements the same two methods for real chess and the trainer
is unchanged.

torch-backed, lazily imported (CPU-importable; CPU is enough to train these
tiny models, a GPU only speeds it up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gps.interface import DecisionPoint
from gps.latent.base import Injection, InjectionKind
from gps.latent.structured import DIMENSIONS, history_features
from gps.policy.base import PolicyBackbone
from gps.prediction import MoveDistribution, Prediction, TimingPrediction
from gps.train.base import Trajectory


@dataclass
class EncodedBatch:
    """A batch of trajectories as aligned tensors ``[T, B, ...]``.

    All trajectories in a batch must share length ``T`` (the Phase-0 sessions
    do, by construction). ``feats`` is the per-step history-feature input the
    injector consumes; the rest are the supervision targets / context.
    """

    feats: object  # FloatTensor [T, B, n_features]
    value_adv: object  # FloatTensor [T, B, n_actions]  (normalised to [0,1])
    move_idx: object  # LongTensor  [T, B]  (index into the action axis)
    times: object  # FloatTensor [T, B]  (observed think-time, seconds)
    n_steps: int
    n_traj: int

    def to(self, device) -> EncodedBatch:
        """Move all tensors to a torch device (returns a new batch)."""
        return EncodedBatch(
            feats=self.feats.to(device),
            value_adv=self.value_adv.to(device),
            move_idx=self.move_idx.to(device),
            times=self.times.to(device),
            n_steps=self.n_steps,
            n_traj=self.n_traj,
        )

    def window(self, a: int, b: int) -> EncodedBatch:
        """Slice the time axis to ``[a, b)`` (the held-out later games)."""
        return EncodedBatch(
            feats=self.feats[a:b],
            value_adv=self.value_adv[a:b],
            move_idx=self.move_idx[a:b],
            times=self.times[a:b],
            n_steps=b - a,
            n_traj=self.n_traj,
        )


class DiffMovePolicy(PolicyBackbone):
    """Differentiable move+timing head over engine value-advantages.

    Parameters
    ----------
    latent_dim:
        Width of the latent vector this head consumes (the injector's hidden
        size for the E-A1 arms; the anchored ``DIMENSIONS`` count for the
        memoryless feature control).
    n_actions:
        Size of the (fixed) legal-action axis -- the toy game's branching
        factor. Positions with fewer legal moves would be masked; the toy game
        is constant-branching so no mask is needed.
    """

    accepts = (InjectionKind.HIDDEN,)

    def __init__(self, latent_dim: int, n_actions: int) -> None:
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self._net = None

    # --- network -------------------------------------------------------
    def _build(self):
        if self._net is not None:
            return self._net
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "torch required for DiffMovePolicy; install the 'train' "
                "extra: pip install '.[train]'."
            ) from e

        class _Head(nn.Module):
            def __init__(self, latent_dim):
                super().__init__()
                # Sharpness (inverse temperature) >= 0 via softplus.
                self.sharpness = nn.Linear(latent_dim, 1)
                # Think-time log-mean; log-std is a free scalar.
                self.mu = nn.Linear(latent_dim, 1)
                self.log_sigma = nn.Parameter(torch.zeros(1))

            def forward(self, latent, value_adv):
                # latent [..., L]; value_adv [..., A].
                beta = nn.functional.softplus(self.sharpness(latent))
                logits = beta * value_adv
                logp = torch.log_softmax(logits, dim=-1)
                mu = self.mu(latent).squeeze(-1)
                sigma = nn.functional.softplus(self.log_sigma) + 1e-3
                return logp, mu, sigma

        self._net = _Head(self.latent_dim)
        return self._net

    def parameters(self):
        return self._build().parameters()

    def to(self, device):
        self._build().to(device)
        return self

    # --- encoding (raw trajectories -> tensors) ------------------------
    @staticmethod
    def _value_adv(dp: DecisionPoint) -> list[float]:
        """Engine values of the legal moves, normalised to ``[0, 1]``.

        Matches the synthetic softmax and the mock backbone: ``(v - min) /
        span`` over the position's legal moves, in ``legal_actions`` order.
        """
        ref = dp.engine_reference
        vals = [ref.candidate_values[m] for m in dp.legal_actions]
        vmin, vmax = min(vals), max(vals)
        span = (vmax - vmin) or 1.0
        return [(v - vmin) / span for v in vals]

    def encode_batch(self, trajectories: list[Trajectory]) -> EncodedBatch:
        import torch

        lengths = {len(t.decisions) for t in trajectories}
        if len(lengths) != 1:
            raise ValueError(
                "DiffMovePolicy.encode_batch needs equal-length trajectories; "
                f"got lengths {sorted(lengths)}"
            )
        feats, value_adv, move_idx, times = [], [], [], []
        for t in trajectories:
            f_col, v_col, m_col, ti_col = [], [], [], []
            for dp, obs in zip(t.decisions, t.observations):
                h = history_features(dp)
                f_col.append([h[d] for d in DIMENSIONS])
                v_col.append(self._value_adv(dp))
                m_col.append(dp.legal_actions.index(obs.move))
                ti_col.append(float(obs.time_spent or 1e-3))
            feats.append(f_col)
            value_adv.append(v_col)
            move_idx.append(m_col)
            times.append(ti_col)

        # Stored as [T, B, ...] (time-major) for the recurrence.
        def tb(x, dtype):
            return torch.tensor(x, dtype=dtype).transpose(0, 1).contiguous()

        return EncodedBatch(
            feats=tb(feats, torch.float32),
            value_adv=tb(value_adv, torch.float32),
            move_idx=tb(move_idx, torch.long),
            times=tb(times, torch.float32),
            n_steps=len(next(iter(trajectories)).decisions),
            n_traj=len(trajectories),
        )

    # --- loss + eval ---------------------------------------------------
    def trajectory_loss(self, latent_seq, batch: EncodedBatch, lam: float):
        """Move-NLL + ``lam`` * timing-NLL over a (sub)trajectory window.

        ``latent_seq`` is ``[T, B, L]`` from the injector; the window scored is
        whatever ``T`` the caller passes (the trainer scores the train games,
        the evaluator scores the held-out later games). Returns a dict of
        differentiable scalars.
        """
        import torch

        net = self._build()
        logp, mu, sigma = net(latent_seq, batch.value_adv)
        # Move NLL: gather the log-prob of the played move at each step.
        chosen = logp.gather(-1, batch.move_idx.unsqueeze(-1)).squeeze(-1)
        move_nll = -chosen.mean()
        # Timing NLL: log-normal on seconds (see TimingPrediction.logpdf).
        logt = torch.log(batch.times.clamp_min(1e-3))
        z = (logt - mu) / sigma
        timing_nll = (
            0.5 * z * z + torch.log(sigma) + 0.5 * math.log(2 * math.pi) + logt
        ).mean()
        loss = move_nll + lam * timing_nll
        return {
            "loss": loss,
            "move_nll": move_nll,
            "timing_nll": timing_nll,
        }

    def per_traj_move_nll(self, latent_seq, batch: EncodedBatch):
        """Per-trajectory mean move-NLL over the window: a ``[B]`` tensor.

        One number per player (the batch axis), averaged over the scored time
        window -- the unit the Milestone-A bootstrap resamples
        (:func:`gps.eval.bootstrap.bootstrap_ci`). ``latent_seq`` and ``batch``
        must already be sliced to the eval window.
        """
        net = self._build()
        logp, _, _ = net(latent_seq, batch.value_adv)
        chosen = logp.gather(-1, batch.move_idx.unsqueeze(-1)).squeeze(-1)
        return -chosen.mean(dim=0)

    def predict(
        self, dp: DecisionPoint, injection: Injection | None = None
    ) -> Prediction:
        """Single-step eval prediction (detached), for the Simulator path.

        Mirrors :meth:`trajectory_loss` for one decision so the trained head
        can also be scored through the standard eval harness.
        """
        import torch

        net = self._build()
        if injection is None or injection.vector is None:
            latent = torch.zeros(self.latent_dim)
        else:
            latent = torch.tensor(injection.vector, dtype=torch.float32)
        value_adv = torch.tensor(self._value_adv(dp), dtype=torch.float32)
        with torch.no_grad():
            logp, mu, sigma = net(latent, value_adv)
            probs = torch.exp(logp)
        moves = {m: float(probs[i]) for i, m in enumerate(dp.legal_actions)}
        return Prediction(
            moves=MoveDistribution(probs=moves),
            timing=TimingPrediction(mu=float(mu), sigma=float(sigma)),
            latent=injection.vector if injection else None,
        )

    @property
    def name(self) -> str:
        return f"DiffMovePolicy(latent_dim={self.latent_dim})"
