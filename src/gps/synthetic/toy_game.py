"""A minimal perfect-information toy game for Phase-0.

We do not need real chess/Go to validate the *mechanism-recovery* claims --
we need a game with (a) a clear notion of move quality (so an engine oracle
is trivial and exact), and (b) enough moves per game to expose within-game
dynamics. This toy supplies both with zero dependencies.

Position
--------
A position offers ``branching`` legal moves labelled ``"m0".."m{b-1}"``.
Each move has a fixed *quality* in ``[0, 1]`` (1 == optimal). The oracle
value of a move is just ``quality`` expressed in "centipawn-like" units, so
points-lost is exact and known. A game lasts a fixed number of plies; the
result is decided by the average quality of the moves actually played
(noisy threshold), which lets a synthetic player's degraded play translate
into more losses -- the signal post-loss tilt feeds on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gps.interface import EngineReference
from gps.util.rng import LCG


@dataclass(frozen=True)
class ToyPosition:
    """One toy position: a fixed set of moves with known qualities."""

    ply: int
    move_qualities: dict[str, float]

    @property
    def legal_moves(self) -> tuple[str, ...]:
        return tuple(self.move_qualities.keys())

    def engine_reference(self, scale: float = 100.0) -> EngineReference:
        """Exact oracle: value == quality * scale (centipawn-like)."""
        values = {m: q * scale for m, q in self.move_qualities.items()}
        best = max(values, key=values.get)
        return EngineReference(
            candidate_values=values,
            best_move=best,
            best_value=values[best],
            unit="toy-centipawn",
            depth=None,
        )


@dataclass
class ToyGame:
    """Generates reproducible toy positions and scores games.

    ``seed`` makes the whole game deterministic; synthetic-player runs vary
    behaviour by *index*, never by wall-clock randomness, so experiments
    replay exactly.
    """

    branching: int = 5
    plies: int = 20
    seed: int = 0
    _rng: LCG = field(init=False)

    def __post_init__(self) -> None:
        self._rng = LCG(self.seed)

    def position(self, ply: int) -> ToyPosition:
        """A position at ``ply`` with deterministic per-move qualities."""
        # Derive a stable sub-seed from (seed, ply) so the same ply in the
        # same game always yields the same position.
        rng = LCG(self.seed * 1_000_003 + ply * 97 + 1)
        qualities: dict[str, float] = {}
        for i in range(self.branching):
            qualities[f"m{i}"] = round(rng.uniform(0.0, 1.0), 4)
        # Guarantee at least one clearly-best move for a well-posed oracle.
        best_move = f"m{rng.randint(0, self.branching - 1)}"
        qualities[best_move] = 1.0
        return ToyPosition(ply=ply, move_qualities=qualities)

    def game_won(
        self, played_qualities: list[float], center: float = 0.85
    ) -> bool:
        """Decide a game outcome from the qualities actually played.

        Win probability is a logistic in the *average move quality*, centred
        at ``center`` (calibrated so a strong-but-imperfect player wins
        clearly more than half but still loses with realistic frequency --
        without losses, post-loss tilt could never fire). Deterministic
        given the qualities and a draw from the game RNG, so a fixed
        trajectory replays identically.
        """
        if not played_qualities:
            return False
        avg = sum(played_qualities) / len(played_qualities)
        # P(win) = sigmoid(slope * (avg - center)); higher quality wins more.
        slope = 12.0
        p_win = 1.0 / (1.0 + math.exp(-slope * (avg - center)))
        return self._rng.random() < p_win
