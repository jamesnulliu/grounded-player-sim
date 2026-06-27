"""Trainable recurrent latent-state injector (the real ``f_phi``).

This is arm **D** of Milestone A in its *trainable* form -- the thing the
parameter-free :class:`~gps.latent.structured.StructuredInjector` only stands
in for. Where the structured injector smooths the engineered history features
with a hand-set EMA ``alpha``, this injector accumulates the *same* features
into an evolving hidden state with a **learned** time constant (a GRU
recurrence over the trajectory), so SFT (:mod:`gps.train.sft`) has real
``parameters()`` to fit and the "accumulate history with the right dynamics"
hypothesis can actually be tested rather than asserted.

Design choices that keep the Milestone-A comparison honest:

* **Equal inputs.** The per-step input is exactly
  :func:`gps.latent.structured.history_features` -- byte-for-byte the feature
  set the memoryless control
  (:class:`~gps.latent.structured.HistoryConditionedInjector` /
  :class:`~gps.policy.history_conditioned.HistoryConditionedBackbone`) reads.
  The *only* thing this injector adds over the control is the recurrence that
  carries those features across steps. Any win is therefore attributable to
  the evolving state, not to a richer input (see ``documents/milestone_a.md``
  section 2).
* **One state, two channels.** A single learned hidden state ``h_t`` is read
  out to the anchored :data:`~gps.latent.structured.DIMENSIONS` and rendered
  as *either* a hidden vector *or* the same verbal note the structured
  injector emits. So the verbal-vs-hidden comparison (RQ6, Milestone E) is a
  channel-only contrast over an identical state, never a state difference.
* **Probeable.** ``probe_vector`` exposes the anchored readout, so the RQ2
  state-recovery probe works unchanged.

torch-backed and lazily imported, mirroring the concrete backbones, so this
module stays importable on a CPU-only laptop; the network is built on first
use. CPU is sufficient to construct, run, and unit-test it -- a GPU only
speeds up the E-A1 smoke-train.
"""

from __future__ import annotations

from gps.interface import DecisionPoint
from gps.latent.base import (
    Injection,
    InjectionKind,
    LatentState,
    LatentStateInjector,
    Observation,
)
from gps.latent.structured import (
    _Z,
    DIMENSIONS,
    StructuredInjector,
    history_features,
)


class NeuralInjector(LatentStateInjector):
    """A GRU recurrence over the shared history features (trainable ``f_phi``).

    Parameters
    ----------
    kind:
        Injection channel to produce (``VERBAL`` or ``HIDDEN``). Advertised via
        :attr:`produces` so the simulator's compatibility check is meaningful.
    latent_dim:
        Width of the recurrent hidden state ``h_t``. Larger than the four
        anchored dimensions on purpose: the readout projects ``h_t`` back down
        to the anchored dims for injection and probing, leaving the extra
        capacity free for un-anchored dynamics.
    seed:
        Seeds parameter initialisation so construction is reproducible without
        perturbing the global torch RNG (uses a forked RNG).
    persist:
        Whether the hidden state carries across steps. ``True`` is the proposed
        evolving latent (arm D). ``False`` zeroes the state before every step,
        giving a **memoryless** twin with *identical parameters and inputs* --
        the airtight equal-capacity Milestone-A control (arm B): the only
        thing that differs between a ``persist=True`` and ``persist=False``
        injector is whether the recurrence accumulates state across steps.
        See :meth:`latent_trajectory`.
    """

    def __init__(
        self,
        kind: InjectionKind = InjectionKind.VERBAL,
        latent_dim: int = 8,
        seed: int = 0,
        persist: bool = True,
    ) -> None:
        self.kind = kind
        self.latent_dim = latent_dim
        self.seed = seed
        self.persist = persist
        self.produces = (kind,)
        self.input_dim = len(DIMENSIONS)
        # Reuse the structured injector purely as the verbal renderer, so a
        # verbal NeuralInjector and a verbal StructuredInjector speak the same
        # vocabulary -- any gap between them is state, never wording.
        self._renderer = StructuredInjector(kind=kind)
        self._net = None  # built lazily on the first call needing torch

    # --- network construction ------------------------------------------
    def _build(self):
        if self._net is not None:
            return self._net
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "torch required for NeuralInjector; install the 'train' "
                "extra: pip install '.[train]' (CPU is fine for construction "
                "and unit tests; a GPU only speeds up training)."
            ) from e

        class _Net(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.cell = nn.GRUCell(input_dim, hidden_dim)
                # Readout to the anchored dimensions (probe + injection view).
                self.readout = nn.Linear(hidden_dim, len(DIMENSIONS))

            def step(self, x, h):
                return self.cell(x, h)

            def anchored(self, h):
                # First three anchored dims are severities in [0, 1]
                # (sigmoid); momentum is signed in [-1, 1] (tanh).
                raw = self.readout(h)
                sev = torch.sigmoid(raw[..., :3])
                mom = torch.tanh(raw[..., 3:4])
                return torch.cat([sev, mom], dim=-1)

        # Build with a forked RNG so seeding is reproducible and side-effect
        # free w.r.t. any surrounding training run.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            net = _Net(self.input_dim, self.latent_dim)
        self._net = net
        return net

    def parameters(self):
        """torch parameters of ``f_phi`` (so :class:`SFTTrainer` can fit them).

        Returns the GRU + readout parameters; a non-empty iterable here is what
        flips the trainer out of its parameter-free no-op guard.
        """
        return self._build().parameters()

    def to(self, device):
        """Move ``f_phi`` to a torch device (returns self, for chaining)."""
        self._build().to(device)
        return self

    def latent_trajectory(self, feats_seq):
        """Differentiable recurrence over a whole trajectory (training path).

        ``feats_seq`` is a ``[T, B, input_dim]`` tensor of per-step
        :func:`history_features`. Returns the hidden-state sequence
        ``[T, B, latent_dim]`` -- the latent the differentiable backbone
        consumes during SFT. This is the *training* counterpart of the
        :meth:`render`/:meth:`update` inference path (which detaches to a list
        for the Simulator); both run the same GRU cell.

        When :attr:`persist` is ``False`` the hidden state is reset to zero
        before each step, so the output depends only on the current features --
        the memoryless arm-B control at identical capacity (the GRU parameters
        are the same; only the carry is removed).
        """
        net = self._build()
        steps, batch, _ = feats_seq.shape
        h = feats_seq.new_zeros(batch, self.latent_dim)
        outs = []
        for t in range(steps):
            h_in = (
                h
                if self.persist
                else feats_seq.new_zeros(batch, self.latent_dim)
            )
            h = net.step(feats_seq[t], h_in)
            outs.append(h)
        import torch

        return torch.stack(outs, dim=0)

    # --- helpers --------------------------------------------------------
    def _features_tensor(self, dp: DecisionPoint):
        import torch

        feats = history_features(dp)
        return torch.tensor(
            [feats[d] for d in DIMENSIONS], dtype=torch.float32
        )

    def _anchored_list(self, h) -> list[float]:
        net = self._build()
        vec = net.anchored(h).detach().reshape(-1).tolist()
        return [float(x) for x in vec]

    # --- lifecycle ------------------------------------------------------
    def initial_state(self, player_id: str) -> LatentState:
        import torch

        self._build()
        h0 = torch.zeros(self.latent_dim, dtype=torch.float32)
        return LatentState(
            payload=h0,
            probe_vector=self._anchored_list(h0),
            meta={"player_id": player_id},
        )

    def render(self, state: LatentState, dp: DecisionPoint) -> Injection:
        # Read the learned state out to the anchored dims, then render through
        # the shared structured renderer -- hidden returns that vector, verbal
        # returns the same templated note -- so the channel is the only thing
        # that varies between hidden and verbal (RQ6).
        anchored = self._anchored_list(state.payload)
        z = _Z(values=dict(zip(DIMENSIONS, anchored)))
        return self._renderer.render(
            LatentState(payload=z, probe_vector=z.as_vector()), dp
        )

    def update(
        self,
        state: LatentState,
        dp: DecisionPoint,
        observed: Observation | None = None,
    ) -> LatentState:
        # ``observed`` is intentionally unused: the recurrence is fed the same
        # engineered history_features the memoryless control sees, so the
        # Milestone-A "equal inputs" contract holds and any win is the
        # recurrence, not a richer signal. (A learned variant could fold
        # ``observed`` in, but that would change the inputs and the test.)
        net = self._build()
        x = self._features_tensor(dp)
        h_new = net.step(x.unsqueeze(0), state.payload.unsqueeze(0)).squeeze(0)
        return LatentState(
            payload=h_new,
            probe_vector=self._anchored_list(h_new),
            meta=state.meta,
        )

    # --- capacity bookkeeping ------------------------------------------
    def param_report(self) -> dict[str, object]:
        """Parameter budget, for the equal-capacity claim (vs. the control).

        The Milestone-A headline (D - B) must be reported with both arms'
        parameter counts side by side, so a latent win cannot be dismissed as
        "more parameters" (``documents/milestone_a.md`` sections 2 and 6). This
        returns the concrete count for arm D; size
        :class:`~gps.policy.history_conditioned.HistoryConditionedBackbone`'s
        fusion MLP to match it.
        """
        net = self._build()
        total = sum(p.numel() for p in net.parameters())
        return {
            "injector": type(self).__name__,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "n_parameters": int(total),
        }

    @property
    def name(self) -> str:
        return (
            f"NeuralInjector(latent_dim={self.latent_dim}, "
            f"kind={self.kind.value})"
        )
