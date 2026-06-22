"""A structured, probeable latent-state injector.

This is the CPU-runnable reference implementation of the contribution. The
latent ``z_t`` is a small vector with *semantically anchored* dimensions
(proposal section 4.2, "structured latent with semantically anchored
dimensions for time-pressure / post-loss / fatigue"). It is updated move to
move and game to game from the player's own trajectory, and it can be
rendered either as natural-language memory (verbal) or as a raw vector
(hidden) -- the same state, two injection channels.

Why structured (not a free recurrent latent) for the reference impl?
Proposal Risk 5: a flexible recurrent latent can fit anything and explain
nothing. A structured latent is trivially probeable, gives Phase-0 a crisp
recovery target, and serves as the interpretable anchor that the trainable
neural variants (in :mod:`gps.train`) are compared against. The neural
variants implement the *same* :class:`LatentStateInjector` interface and
swap in transparently.

The update rule here is a hand-specified exponential-moving-average over
engineered indicators. It has **no trained parameters** -- it exists so the
whole pipeline runs and is testable today, and so RQ2 has a ground-truth-
adjacent baseline. The learned injectors replace :meth:`_indicators` /
:meth:`update` internals with ``f_phi`` while keeping ``render``.
"""

from __future__ import annotations

from dataclasses import dataclass

from gps.interface import DecisionPoint
from gps.latent.base import (
    Injection,
    InjectionKind,
    LatentState,
    LatentStateInjector,
    Observation,
)

# Anchored dimensions, in fixed order. The probe_vector follows this order.
DIMENSIONS = ("time_pressure", "post_loss", "fatigue", "momentum")


@dataclass
class _Z:
    """Payload for the structured latent: one float per anchored dimension."""

    values: dict[str, float]

    def as_vector(self) -> list[float]:
        return [self.values[d] for d in DIMENSIONS]


class StructuredInjector(LatentStateInjector):
    """EMA over engineered indicators; renders verbal or hidden.

    Parameters
    ----------
    kind:
        Which injection channel to produce. ``VERBAL`` splices a short text
        memory into the prompt; ``HIDDEN`` hands the raw vector to the
        backbone as a soft prompt. The injector advertises exactly this one
        kind via :attr:`produces`, so the simulator's compatibility check is
        meaningful.
    alpha:
        EMA smoothing for within/cross-game updates (0..1, higher = faster).
    """

    def __init__(
        self, kind: InjectionKind = InjectionKind.VERBAL, alpha: float = 0.4
    ) -> None:
        self.kind = kind
        self.alpha = alpha
        self.produces = (kind,)

    # --- lifecycle ------------------------------------------------------
    def initial_state(self, player_id: str) -> LatentState:
        z = _Z(values={d: 0.0 for d in DIMENSIONS})
        return LatentState(
            payload=z,
            probe_vector=z.as_vector(),
            meta={"player_id": player_id},
        )

    def render(self, state: LatentState, dp: DecisionPoint) -> Injection:
        z: _Z = state.payload
        if self.kind is InjectionKind.HIDDEN:
            return Injection(kind=self.kind, vector=z.as_vector())
        return Injection(kind=self.kind, text=self._verbalize(z))

    def update(
        self,
        state: LatentState,
        dp: DecisionPoint,
        observed: Observation | None = None,
    ) -> LatentState:
        z: _Z = state.payload
        ind = self._indicators(dp, observed)
        new_vals = {
            d: (1 - self.alpha) * z.values[d] + self.alpha * ind[d]
            for d in DIMENSIONS
        }
        new_z = _Z(values=new_vals)
        return LatentState(
            payload=new_z,
            probe_vector=new_z.as_vector(),
            meta=state.meta,
        )

    # --- internals (these are what f_phi replaces in learned variants) --
    def _indicators(
        self, dp: DecisionPoint, observed: Observation | None
    ) -> dict[str, float]:
        """Engineered indicators in [0,1] (or signed for momentum)."""
        ts = dp.time_signal
        stream = dp.recent_outcomes

        # Time pressure: 1 when out of time, 0 when comfortable.
        tp = 0.0
        if ts.time_remaining is not None:
            tp = max(0.0, min(1.0, 1.0 - ts.time_remaining / 30.0))

        # Post-loss: 1 right after a loss, decaying with games since.
        pl = 0.0
        for i, o in enumerate(reversed(stream.recent)):
            if o.won is False:
                pl = max(0.0, 1.0 - i / 3.0)
                break

        # Fatigue: ramps with session position.
        fat = max(0.0, min(1.0, stream.session_position / 20.0))

        # Momentum: signed recent win rate centred at 0.
        wr = stream.recent_win_rate(k=5)
        mom = 0.0 if wr is None else (wr - 0.5) * 2.0

        return {
            "time_pressure": tp,
            "post_loss": pl,
            "fatigue": fat,
            "momentum": mom,
        }

    def _verbalize(self, z: _Z) -> str:
        """Render the latent as a compact natural-language scouting note.

        Kept terse and templated on purpose: it is a *memory injected into
        the prompt*, not prose for a human. Learned verbal variants may
        instead emit free-form text; both satisfy the interface.
        """
        parts = []
        v = z.values
        if v["time_pressure"] > 0.5:
            parts.append("under time pressure (rushing, more errors likely)")
        if v["post_loss"] > 0.5:
            parts.append("just lost (may be tilted / over-aggressive)")
        if v["fatigue"] > 0.5:
            parts.append("deep into the session (fatigued, slower decline)")
        if v["momentum"] > 0.4:
            parts.append("on a winning streak (confident)")
        elif v["momentum"] < -0.4:
            parts.append("on a losing streak (shaken)")
        if not parts:
            parts.append("composed, near baseline form")
        return "Current player state: " + "; ".join(parts) + "."


class OracleInjector(LatentStateInjector):
    """Phase-0-only injector that reads the *true* degradation.

    Synthetic players stamp the ground-truth degradation driving each move
    into ``dp.context["true_degradation"]``. This injector surfaces it
    directly. It exists to make P0.2 ("a model that *knows* the dynamic
    state beats a static one") non-circular: the gain is true by
    construction, so it upper-bounds what any learned injector could buy and
    confirms the eval can see the effect at all. It cheats by reading ground
    truth and must never touch real data.
    """

    def __init__(self, kind: InjectionKind = InjectionKind.HIDDEN) -> None:
        self.kind = kind
        self.produces = (kind,)

    def initial_state(self, player_id: str) -> LatentState:
        return LatentState(payload=0.0, probe_vector=[0.0])

    def render(self, state: LatentState, dp: DecisionPoint) -> Injection:
        degr = float(dp.context.get("true_degradation", 0.0))
        if self.kind is InjectionKind.HIDDEN:
            # Place degradation in the first 3 (unsigned) dims the mock
            # backbone reads; momentum left at 0.
            return Injection(kind=self.kind, vector=[degr, degr, degr, 0.0])
        tag = "off-form" if degr > 0.3 else "near baseline form"
        return Injection(kind=self.kind, text=f"Current player state: {tag}.")

    def update(self, state, dp, observed=None) -> LatentState:
        degr = float(dp.context.get("true_degradation", 0.0))
        return LatentState(payload=degr, probe_vector=[degr])
