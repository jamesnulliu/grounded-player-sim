"""Shared decision-point interface (proposal section 4.1).

Both chess and Go reduce every decision to the *same* record. This is what
licenses the "single framework, two games" claim: the latent-state injector
and the policy only ever see :class:`DecisionPoint`s, never game-specific
types. Game code lives behind :mod:`gps.games` and is responsible for
producing these records.

Design notes
------------
* Pure stdlib (``dataclasses`` + ``typing``) so this module -- and the
  Phase-0 / eval code that depends on it -- imports with zero third-party
  packages. Heavy deps stay in backend modules behind lazy imports.
* ``state`` is intentionally an opaque ``object``: a board-native backbone
  wants a tensor, an LLM backbone wants a string (FEN/SGF). Each
  :class:`~gps.games.base.Game` documents what it puts here; policies
  declare what they consume. Keeping it opaque is what lets one interface
  serve very different backbones.
* Move encoding is a plain ``str`` (UCI for chess, GTP-ish for Go). The
  legal-action set is carried explicitly so prediction heads can normalise
  over exactly the legal moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Game(str, Enum):
    """Which domain a decision point belongs to.

    Named ``Game`` for historical reasons (chess/Go were first), but the
    interface is domain-agnostic: a non-game oracle domain such as knowledge
    tracing reuses the same :class:`DecisionPoint` schema (RQ5, design.md §11),
    swapping only the encoder + oracle head.
    """

    CHESS = "chess"
    GO = "go"
    KNOWLEDGE_TRACING = "knowledge_tracing"


@dataclass(frozen=True)
class EngineReference:
    """Engine-oracle view of a position (Stockfish / KataGo).

    This is what turns "style" and "mistake" from vague labels into a
    measurable per-decision quantity: the target is the player's *deviation
    from optimal*, not optimal play itself.

    Attributes
    ----------
    candidate_values:
        Map from legal move (UCI/GTP) to the engine's value for playing it,
        in a per-game unit (chess: centipawns from the mover's side; Go:
        score/winrate from the mover's side). Higher is better for the mover.
    best_move:
        The engine's top move. Convenience; equals the argmax of
        ``candidate_values`` when that map is dense.
    best_value:
        Value of ``best_move`` in the same unit as ``candidate_values``.
    unit:
        Human-readable unit tag, e.g. ``"centipawn"`` or ``"score"``.
    depth:
        Search budget used (Stockfish depth / KataGo visits). Recorded
        because centipawn-loss is settings-dependent and must be reported.
    """

    candidate_values: dict[str, float]
    best_move: str | None = None
    best_value: float | None = None
    unit: str = "centipawn"
    depth: int | None = None

    def loss_of(self, move: str) -> float | None:
        """Points lost by ``move`` vs. the best move (>= 0), or ``None``.

        Returns ``None`` when we lack a value for either ``move`` or the
        reference best, so downstream code can distinguish "no blunder" from
        "unknown".
        """
        ref = self.best_value
        if ref is None and self.candidate_values:
            ref = max(self.candidate_values.values())
        chosen = self.candidate_values.get(move)
        if ref is None or chosen is None:
            return None
        return max(0.0, ref - chosen)


@dataclass(frozen=True)
class TimeSignal:
    """Per-move timing context -- the clearest behavioral fingerprint.

    All times are in seconds. Go-specific byo-yomi fields are optional and
    left ``None`` for chess.
    """

    time_remaining: float | None = None
    increment: float | None = None
    time_spent: float | None = None
    move_number: int = 0
    # Go byo-yomi: periods left + length of each period (seconds).
    byo_yomi_periods_left: int | None = None
    byo_yomi_period_length: float | None = None
    # Coarse game phase tag ("opening" / "middlegame" / "endgame" / ...),
    # filled by the game encoder when cheap to compute.
    phase: str | None = None

    @property
    def in_time_trouble(self) -> bool:
        """Heuristic flag used by state-recovery probes (proposal RQ2).

        Conservative: only true when we positively know remaining time is
        low. Threshold is deliberately crude; probes treat it as a noisy
        indicator, not ground truth.
        """
        return self.time_remaining is not None and self.time_remaining <= 10.0


@dataclass(frozen=True)
class Outcome:
    """One completed game in a player's recent stream."""

    won: bool | None
    # Result margin in game units (centipawns / score). Sign is from the
    # tracked player's perspective; ``None`` if unknown.
    margin: float | None = None
    # Engine-scored swing magnitude over the game (volatility proxy).
    engine_swing: float | None = None
    blunders: int = 0
    time_scramble: bool = False
    # Seconds between the end of this game and the start of the next, used
    # to segment sessions. ``None`` for the most recent game.
    gap_to_next_seconds: float | None = None


@dataclass
class OutcomeStream:
    """Running stream of recent results, within and across a session.

    Ordered oldest -> newest. The latent-state injector reads this to model
    cross-game effects (post-loss tilt, win momentum, late-session fatigue).
    """

    recent: list[Outcome] = field(default_factory=list)
    # Index of the current game within its session (0 == first game). This
    # is a *derived* quantity; see ``gps.data.sessions`` for segmentation.
    session_position: int = 0

    def last(self) -> Outcome | None:
        return self.recent[-1] if self.recent else None

    def recent_win_rate(self, k: int = 5) -> float | None:
        """Win rate over the last ``k`` decided games (foil for B7)."""
        decided = [o.won for o in self.recent[-k:] if o.won is not None]
        if not decided:
            return None
        return sum(decided) / len(decided)


@dataclass(frozen=True)
class DecisionPoint:
    """One move-choice opportunity, game-agnostic (proposal section 4.1).

    A policy is asked: given this player, this position, the engine
    reference, the timing context, and how the session has gone so far --
    what move does *this* player make, and how long do they take?
    """

    game: Game
    player_id: str
    state: object  # opaque board encoding; game-specific (see module doc)
    legal_actions: tuple[str, ...]
    engine_reference: EngineReference | None
    time_signal: TimeSignal
    recent_outcomes: OutcomeStream
    # Free-form game context: time_control, rating_gap, color, etc. Kept as
    # a dict so games can attach extras without widening the schema.
    context: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.legal_actions:
            raise ValueError(
                "DecisionPoint must have at least one legal action"
            )
