"""Chess-shaped synthetic players for the E-C *positive control*.

The Phase-0 players in :mod:`gps.synthetic.players` emit toy positions
(``state="toy:ply=N"``) and choose moves by softmax over an engine reference --
right for :class:`~gps.policy.diff_policy.DiffMovePolicy`, but the board-native
backbone reads the *board* (FEN) and has no oracle. To validate the E-C2 driver
(:func:`gps.experiments.ec.run_ec`) end to end, we need a player whose
decisions look like real chess (a real FEN + UCI legal moves) yet carry a
**hidden dynamic** the evolving latent should recover and the memoryless twin
should not.

:class:`HiddenTiltChessPlayer` is exactly that -- the chess analog of
:class:`~gps.synthetic.players.HysteresisTiltPlayer`:

* A small pool of **distinct real FENs**, each with a designated *good* and
  *bad* UCI move. Distinct placements give distinct board planes, so the board
  identifies the position (and thus which move is good); the move set is
  declared, not engine-derived (synthetic, no oracle).
* A **hidden leaky-integral tilt** ``h`` over the *entire ordered* loss
  stream::

      h_g       = rho * h_{g-1} + (1 - rho) * loss_indicator(g-1)
      P(good)_g = sigmoid(calm_logit - tilt_scale * h_g)

  ``h`` is constant within a game and evolves across the session. It is **not**
  reconstructable from the shared
  :func:`~gps.latent.structured.history_features` (a 5-game unordered win rate
  + a 3-game-capped loss recency), for the same reason as the toy hysteresis
  player -- so a memoryless reader has irreducible error and an evolving latent
  that integrates the stream does not.
* Game outcome feeds back: worse moves -> more likely a loss -> higher ``h`` ->
  worse moves. That feedback makes the tilt *persist* (hysteresis), the regime
  where cross-game memory pays off.

This is a **positive control**: ``run_ec`` on a population of these players
should show arm D (persist) beat arm B (memoryless) -- proving the driver
detects a true effect, the complement of its correct null on board-determined
targets. Pure stdlib (numpy-free); deterministic given ``seed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gps.interface import (
    DecisionPoint,
    EngineReference,
    Game,
    Outcome,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import Observation
from gps.train.base import Trajectory, TrajectoryDataset
from gps.util.rng import LCG

#: Distinct FENs (white to move), each with several UCI moves and a *quality*
#: per move (higher = better). Moves are synthetic labels (not engine-verified
#: legality) -- only the from/to squares must be on-board so the factored head
#: can encode them; distinct placements give distinct board planes so the
#: position (and thus its quality ranking) is identifiable from the board. The
#: player picks via ``softmax(beta * quality)``, so a *lower* beta (more tilt)
#: flattens toward blunders -- exactly the toy ``HysteresisTiltPlayer`` knob,
#: now on a real board the board-native head can read.
DEFAULT_POOL: tuple[tuple[str, tuple[str, ...], tuple[float, ...]], ...] = (
    (
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        ("e2e4", "e2e3", "e1d1", "e1f1"),
        (1.0, 0.55, 0.2, 0.0),
    ),
    (
        "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
        ("d5d6", "d5c6", "e1d2", "e1f2"),
        (1.0, 0.5, 0.25, 0.0),
    ),
    (
        "4k3/8/8/8/8/5N2/8/4K3 w - - 0 1",
        ("f3e5", "f3d4", "f3g1", "f3h4"),
        (1.0, 0.6, 0.3, 0.0),
    ),
    (
        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        ("a1a8", "a1a7", "a1a4", "a1b1"),
        (1.0, 0.65, 0.3, 0.0),
    ),
    (
        "4k3/6B1/8/8/8/8/8/4K3 w - - 0 1",
        ("g7e5", "g7f6", "g7h6", "g7f8"),
        (1.0, 0.55, 0.25, 0.0),
    ),
)


def _phase(ply: int, total: int) -> str:
    frac = ply / max(1, total)
    if frac < 0.33:
        return "opening"
    if frac < 0.66:
        return "middlegame"
    return "endgame"


@dataclass
class HiddenTiltChessPlayer:
    """A chess-shaped player whose move sharpness is a hidden loss integral."""

    player_id: str
    seed: int = 0
    rho: float = 0.9
    # base_beta must be low enough that the player *loses often* -- otherwise
    # it rarely loses, h stays ~0, and there is no tilt dynamic to detect (the
    # base_beta=6 regime gave a spurious null). At base_beta=3 the leaky loss
    # integral h spans ~[0, 0.6], the regime where the evolving latent wins.
    base_beta: float = 3.0
    tilt_scale: float = 6.0  # h~0.5 -> beta ~0 (full blunders) when tilted
    plies_per_game: int = 8
    starting_clock: float = 60.0
    pool: tuple[tuple[str, tuple[str, ...], tuple[float, ...]], ...] = field(
        default_factory=lambda: DEFAULT_POOL
    )

    def __post_init__(self) -> None:
        self._rng = LCG(self.seed)

    def _h_from(self, recent: list[Outcome]) -> float:
        h = 0.0
        for o in recent:  # oldest -> newest
            loss = 1.0 if o.won is False else 0.0
            h = self.rho * h + (1.0 - self.rho) * loss
        return h

    def _pick(self, qualities: tuple[float, ...], beta: float) -> int:
        weights = [math.exp(beta * q) for q in qualities]
        return self._rng.categorical(weights)

    def build_trajectory(self, n_games: int) -> Trajectory:
        """Play ``n_games`` and assemble one time-ordered Trajectory."""
        decisions: list[DecisionPoint] = []
        observations: list[Observation] = []
        prior: list[Outcome] = []  # completed-game outcomes (oldest -> newest)

        for g in range(n_games):
            # Freeze the history as of this game (constant within the game,
            # never aliased -- mirrors gps.data.lichess.build_trajectory).
            history = OutcomeStream(recent=list(prior), session_position=g)
            h = self._h_from(prior)
            beta = max(0.3, self.base_beta - self.tilt_scale * h)

            clock = self.starting_clock
            quality_sum = 0.0
            for ply in range(self.plies_per_game):
                fen, moves, quals = self.pool[ply % len(self.pool)]
                idx = self._pick(quals, beta)
                quality_sum += quals[idx]
                # Attach the (synthetic) engine reference too, so this player
                # is dual-use: the board-native backbone ignores it, but
                # DiffMovePolicy can read it -- enabling the head-to-head that
                # isolates *where* the dynamic latent has leverage (engine
                # sharpness vs a from-scratch board model).
                cand = {m: q for m, q in zip(moves, quals)}
                best = max(cand, key=cand.get)
                ref = EngineReference(
                    candidate_values=cand,
                    best_move=best,
                    best_value=cand[best],
                    unit="synthetic",
                )
                # Tilted players also dawdle a touch (a weak, latent-correlated
                # timing signal; not the headline).
                spent = math.exp(self._rng.uniform(0.6, 1.2) + 0.4 * h)
                clock = max(1.0, clock - spent)
                decisions.append(
                    DecisionPoint(
                        game=Game.CHESS,
                        player_id=self.player_id,
                        state=fen,
                        legal_actions=moves,
                        engine_reference=ref,
                        time_signal=TimeSignal(
                            time_remaining=clock,
                            increment=0.0,
                            time_spent=spent,
                            move_number=ply,
                            phase=_phase(ply, self.plies_per_game),
                        ),
                        recent_outcomes=history,
                        context={"synthetic": True, "hidden_h": h},
                    )
                )
                observations.append(
                    Observation(move=moves[idx], time_spent=spent)
                )

            # Outcome feeds back into the accumulator: worse moves -> likelier
            # loss -> higher h next game (the hysteresis loop).
            good_frac = quality_sum / self.plies_per_game
            won = self._rng.uniform(0.0, 1.0) < good_frac
            prior.append(Outcome(won=won))

        return Trajectory(self.player_id, decisions, observations)


def build_hidden_tilt_dataset(
    n_players: int = 16,
    n_games: int = 24,
    *,
    seed: int = 0,
    plies_per_game: int = 8,
    rho: float = 0.9,
    base_beta: float = 3.0,
    tilt_scale: float = 6.0,
) -> TrajectoryDataset:
    """A population of hidden-tilt chess players as a ``TrajectoryDataset``.

    Each player gets a distinct seed (so their loss streams, and thus hidden
    trajectories, differ). The dataset is the *full* sessions; ``run_ec`` makes
    the per-player temporal split.
    """
    trajectories = [
        HiddenTiltChessPlayer(
            player_id=f"hidden-tilt-{i}",
            seed=seed + 1 + i,
            rho=rho,
            base_beta=base_beta,
            tilt_scale=tilt_scale,
            plies_per_game=plies_per_game,
        ).build_trajectory(n_games)
        for i in range(n_players)
    ]
    return TrajectoryDataset(trajectories)
