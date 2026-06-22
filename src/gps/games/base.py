"""Game backend interface + engine-oracle protocol.

A :class:`Game` knows how to (a) encode a native position into the opaque
``state`` carried by :class:`~gps.interface.DecisionPoint`, (b) enumerate
legal moves, and (c) consult an :class:`EngineOracle` for the per-move value
reference. Keeping all three behind one interface is what lets chess and Go
share the simulator, the latent injector, and the eval harness unchanged.
"""

from __future__ import annotations

import abc
from typing import Protocol

from gps.interface import EngineReference
from gps.interface import Game as GameId


class EngineOracle(Protocol):
    """A per-position value/policy reference (Stockfish / KataGo).

    Implementations may be expensive (subprocess to an engine) so callers
    are expected to cache. The protocol is intentionally tiny: given a
    native position and its legal moves, return an
    :class:`~gps.interface.EngineReference`.
    """

    def evaluate(
        self, position: object, legal_moves: tuple[str, ...]
    ) -> EngineReference: ...


class Game(abc.ABC):
    """Turns native records into shared decision points."""

    game_id: GameId

    @abc.abstractmethod
    def encode_state(self, position: object) -> object:
        """Encode a native position into the opaque ``state`` field.

        What goes here depends on the intended backbone: an LLM backbone
        wants a string (FEN / SGF + move list); a board-native backbone
        wants a tensor. A game may return a small struct carrying both and
        let the backbone pick.
        """

    @abc.abstractmethod
    def legal_moves(self, position: object) -> tuple[str, ...]:
        """Legal moves in this position, as UCI (chess) / GTP (Go) strings."""

    @abc.abstractmethod
    def apply_move(self, position: object, move: str) -> object:
        """Return the position after ``move`` (does not mutate input)."""

    def oracle(self) -> EngineOracle | None:
        """The engine oracle for this game, or ``None`` if not configured."""
        return None
