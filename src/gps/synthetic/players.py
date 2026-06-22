"""Synthetic players with known dynamic mechanisms (Phase-0).

Each player is a *stochastic move policy over the toy game* whose move-choice
temperature (and thus suboptimality) is modulated by a known latent
mechanism. Because we know exactly when and why the policy degrades, we can
test whether a learned latent state recovers that mechanism (P0.1), whether
a static model misses it (P0.2), and whether persona prompting mis-shoots it
(P0.3).

The behavioural knob is an *inverse temperature* ``beta`` over engine
values: high ``beta`` -> nearly optimal, low ``beta`` -> noisy/blundery.
Each mechanism perturbs ``beta`` from a player-specific baseline.

These players also emit a think-time per move (log-normal), modulated by the
same mechanisms (e.g. time pressure speeds the player up), so the synthetic
data exercises the timing head too.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass

from gps.interface import (
    DecisionPoint,
    Game,
    Outcome,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import Observation
from gps.synthetic.toy_game import ToyGame
from gps.util.rng import LCG


@dataclass
class GeneratedGame:
    """One synthetic game: aligned decisions + ground-truth observations."""

    decisions: list[DecisionPoint]
    observations: list[Observation]
    won: bool
    # The hidden latent scalar the mechanism used at each ply -- this is the
    # ground truth the RQ2 probe tries to recover. Not visible to models.
    hidden_state: list[float]


class SyntheticPlayer(abc.ABC):
    """Base synthetic player: softmax-over-engine-value move policy.

    Subclasses implement :meth:`beta_at` to express their mechanism: the
    inverse temperature in effect at a given ply, given how the session has
    gone so far.
    """

    def __init__(
        self,
        player_id: str,
        game: ToyGame,
        base_beta: float = 6.0,
        base_log_time: float = 1.5,
        seed: int = 0,
    ) -> None:
        self.player_id = player_id
        self.game = game
        self.base_beta = base_beta
        self.base_log_time = base_log_time
        self._rng = LCG(seed)

    # --- mechanism hook -------------------------------------------------
    @abc.abstractmethod
    def beta_at(
        self,
        ply: int,
        time_remaining: float,
        stream: OutcomeStream,
    ) -> float:
        """Inverse temperature in effect at this decision."""

    def think_time_at(
        self, ply: int, time_remaining: float, beta: float
    ) -> float:
        """Default think-time: log-normal, faster when sharper (high beta).

        Subclasses override to inject timing-specific mechanisms (e.g. a
        speed-up under time pressure).
        """
        mu = self.base_log_time - 0.05 * (beta - self.base_beta)
        return math.exp(self._rng.uniform(mu - 0.3, mu + 0.3))

    # --- generation -----------------------------------------------------
    def _softmax_pick(self, values: dict[str, float], beta: float) -> str:
        moves = list(values.keys())
        # Normalise values to [0,1] before applying beta so beta is on a
        # comparable scale across positions.
        vmax = max(values.values())
        vmin = min(values.values())
        span = (vmax - vmin) or 1.0
        weights = [math.exp(beta * (values[m] - vmin) / span) for m in moves]
        return moves[self._rng.categorical(weights)]

    def play_game(
        self,
        session_stream: OutcomeStream,
        starting_clock: float = 60.0,
        increment: float = 0.0,
    ) -> GeneratedGame:
        """Play one toy game, returning decisions + ground-truth obs."""
        decisions: list[DecisionPoint] = []
        observations: list[Observation] = []
        hidden: list[float] = []
        played_qualities: list[float] = []
        clock = starting_clock

        # Snapshot the session history *as of this game*. ``session_stream``
        # keeps mutating across games, so we must freeze a copy here; binding
        # the live object to every DecisionPoint would make all decisions see
        # the final history and erase the cross-game signal we are modelling.
        history = OutcomeStream(
            recent=list(session_stream.recent),
            session_position=session_stream.session_position,
        )

        for ply in range(self.game.plies):
            pos = self.game.position(ply)
            ref = pos.engine_reference()
            beta = self.beta_at(ply, clock, history)
            hidden.append(beta)

            ts = TimeSignal(
                time_remaining=clock,
                increment=increment,
                move_number=ply,
                phase=_phase(ply, self.game.plies),
            )
            dp = DecisionPoint(
                game=Game.CHESS,  # toy positions reuse the chess tag
                player_id=self.player_id,
                state=f"toy:ply={ply}",
                legal_actions=pos.legal_moves,
                engine_reference=ref,
                time_signal=ts,
                recent_outcomes=history,
                context={"synthetic": True, "starting_clock": starting_clock},
            )

            # Record the *true* degradation driving this move so an oracle
            # model (Phase-0 P0.2) and the recovery probe (P0.1) have ground
            # truth: how far below baseline the inverse-temperature dropped.
            true_degr = max(0.0, (self.base_beta - beta) / self.base_beta)
            dp.context["true_degradation"] = true_degr

            move = self._softmax_pick(ref.candidate_values, beta)
            spent = self.think_time_at(ply, clock, beta)
            clock = max(0.0, clock - spent + increment)

            decisions.append(dp)
            observations.append(Observation(move=move, time_spent=spent))
            played_qualities.append(pos.move_qualities[move])

        won = self.game.game_won(played_qualities)
        return GeneratedGame(
            decisions=decisions,
            observations=observations,
            won=won,
            hidden_state=hidden,
        )

    def play_session(
        self,
        n_games: int,
        starting_clock: float = 60.0,
    ) -> list[GeneratedGame]:
        """Play a session of ``n_games``, carrying the outcome stream."""
        stream = OutcomeStream(recent=[], session_position=0)
        games: list[GeneratedGame] = []
        for g in range(n_games):
            stream.session_position = g
            gg = self.play_game(stream, starting_clock=starting_clock)
            games.append(gg)
            # Append this game's outcome so later games see it.
            avg_q = sum(
                d.engine_reference.candidate_values[o.move]
                for d, o in zip(gg.decisions, gg.observations)
            ) / max(1, len(gg.decisions))
            stream.recent.append(
                Outcome(
                    won=gg.won,
                    margin=avg_q,
                    blunders=sum(
                        1
                        for d, o in zip(gg.decisions, gg.observations)
                        if (d.engine_reference.loss_of(o.move) or 0) > 30
                    ),
                )
            )
        return games


def _phase(ply: int, total: int) -> str:
    frac = ply / max(1, total)
    if frac < 0.33:
        return "opening"
    if frac < 0.66:
        return "middlegame"
    return "endgame"


# --------------------------------------------------------------------------
# Concrete mechanisms
# --------------------------------------------------------------------------
@dataclass
class TiltPlayer(SyntheticPlayer):
    """Plays cleanly until a loss, then degrades for ``tilt_games`` games.

    Mechanism: after a loss, ``beta`` drops by ``tilt_drop`` for the next
    ``tilt_games`` games, then recovers. This is the post-loss tilt the
    dynamic latent should recover and the static model should miss.
    """

    tilt_drop: float = 3.0
    tilt_games: int = 2

    def __init__(self, player_id: str, game, **kw) -> None:
        self.tilt_drop = kw.pop("tilt_drop", 3.0)
        self.tilt_games = kw.pop("tilt_games", 2)
        super().__init__(player_id, game, **kw)

    def beta_at(self, ply, time_remaining, stream) -> float:
        beta = self.base_beta
        # Count games since the most recent loss within the stream.
        since_loss = None
        for i, o in enumerate(reversed(stream.recent)):
            if o.won is False:
                since_loss = i
                break
        if since_loss is not None and since_loss < self.tilt_games:
            beta -= self.tilt_drop
        return max(0.5, beta)


@dataclass
class TimePressurePlayer(SyntheticPlayer):
    """Move quality degrades below ``threshold`` seconds remaining.

    Also speeds up under pressure, so the timing head sees the mechanism too.
    """

    threshold: float = 10.0
    pressure_drop: float = 4.0

    def __init__(self, player_id: str, game, **kw) -> None:
        self.threshold = kw.pop("threshold", 10.0)
        self.pressure_drop = kw.pop("pressure_drop", 4.0)
        super().__init__(player_id, game, **kw)

    def beta_at(self, ply, time_remaining, stream) -> float:
        beta = self.base_beta
        if time_remaining <= self.threshold:
            # Sharper degradation the lower the clock.
            severity = 1.0 - (time_remaining / self.threshold)
            beta -= self.pressure_drop * severity
        return max(0.5, beta)

    def think_time_at(self, ply, time_remaining, beta) -> float:
        mu = self.base_log_time
        if time_remaining <= self.threshold:
            mu -= 1.0  # blitz out moves under pressure
        return math.exp(self._rng.uniform(mu - 0.3, mu + 0.3))


@dataclass
class FatiguePlayer(SyntheticPlayer):
    """Quality degrades after game ``onset`` within a session (fatigue)."""

    onset: int = 15
    fatigue_drop: float = 2.5

    def __init__(self, player_id: str, game, **kw) -> None:
        self.onset = kw.pop("onset", 15)
        self.fatigue_drop = kw.pop("fatigue_drop", 2.5)
        super().__init__(player_id, game, **kw)

    def beta_at(self, ply, time_remaining, stream) -> float:
        beta = self.base_beta
        if stream.session_position >= self.onset:
            over = stream.session_position - self.onset
            beta -= self.fatigue_drop * min(1.0, over / 5.0)
        return max(0.5, beta)
