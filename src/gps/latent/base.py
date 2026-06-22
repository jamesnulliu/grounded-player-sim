"""Latent-state injector interface (proposal section 4.2, module 3).

The latent state is updated both *within* a game (move to move) and *across*
games within a session, from the player's own trajectory::

    z_t = f_phi(z_{t-1}, state_{t-1}, action_{t-1}, engine_outcome_{t-1},
                time_signal_{t-1}, result_stream_{t-1})

This module defines the abstraction only; concrete update rules
(state-space / recurrent / structured) live in the verbal and hidden
implementations. Trainable parameters ``phi`` are owned by subclasses; the
trainers in :mod:`gps.train` fit them.

How the latent reaches the policy is deliberately abstracted by
:class:`Injection`: a *verbal* injector returns text to splice into the
prompt; a *hidden* injector returns a vector the backbone consumes as a soft
prompt / prefix. The :class:`~gps.policy.base.PolicyBackbone` decides which
injection kinds it can honour.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum

from gps.interface import DecisionPoint
from gps.prediction import Prediction


class InjectionKind(str, Enum):
    """How a latent state is delivered to a policy backbone."""

    VERBAL = "verbal"  # natural-language memory spliced into the prompt
    HIDDEN = "hidden"  # vector -> soft prompt / prefix tokens


@dataclass
class Injection:
    """A latent state rendered for one specific backbone.

    Exactly one of ``text`` / ``vector`` is set, matching ``kind``. The
    backbone reads only the field for the kind it supports.
    """

    kind: InjectionKind
    text: str | None = None
    vector: list[float] | None = None


@dataclass
class LatentState:
    """Opaque carrier for z_t, threaded through a trajectory.

    Subclasses stash whatever they need (a vector, a rolling text buffer,
    HMM posteriors, ...) in ``payload``. ``probe_vector`` is the
    fixed-width numeric view used by RQ2 state-recovery probes; subclasses
    should populate it so probing does not need to know their internals.
    """

    payload: object = None
    probe_vector: list[float] | None = None
    # Bookkeeping for debugging / interpretability narration.
    meta: dict[str, object] = field(default_factory=dict)


class LatentStateInjector(abc.ABC):
    """Maintains and injects the per-individual dynamic latent state.

    Lifecycle, per player trajectory::

        z = injector.initial_state(player_id)
        for dp in trajectory:
            inj = injector.render(z, dp)          # -> Injection for policy
            pred = policy.predict(dp, inj)        # policy consumes injection
            z = injector.update(z, dp, observed)  # advance z_t -> z_{t+1}

    Implementations must be deterministic given their parameters and inputs
    so that training and the strict temporal eval split are reproducible.
    """

    #: Injection kinds this injector can produce. The simulator checks this
    #: against what the chosen backbone accepts.
    produces: tuple[InjectionKind, ...] = ()

    @abc.abstractmethod
    def initial_state(self, player_id: str) -> LatentState:
        """z_0 for a player, before any moves are observed."""

    @abc.abstractmethod
    def render(self, state: LatentState, dp: DecisionPoint) -> Injection:
        """Render z_t as an :class:`Injection` for the current decision."""

    @abc.abstractmethod
    def update(
        self,
        state: LatentState,
        dp: DecisionPoint,
        observed: Observation | None = None,
    ) -> LatentState:
        """Advance z_t -> z_{t+1} after observing what actually happened.

        ``observed`` is ``None`` at inference time when the true move/timing
        is not yet known; in that case the injector advances on its own
        prediction or on the engine reference, as the subclass sees fit.
        """


@dataclass
class Observation:
    """Ground truth at a decision point, used to advance the latent state.

    Mirrors the prediction targets: which move was actually played and how
    long it took. Carried separately from :class:`DecisionPoint` so the same
    decision point can be used for both prediction (no peeking) and update.
    """

    move: str
    time_spent: float | None = None
    # The policy's own prediction at this step, if we want to advance the
    # latent on prediction error rather than ground truth.
    prediction: Prediction | None = None
