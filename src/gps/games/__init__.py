"""Game-specific encoders and engine oracles.

Each game implements :class:`~gps.games.base.Game` to turn its native
records into the shared :class:`~gps.interface.DecisionPoint`. Chess is the
concrete first target; Go conforms to the same interface and is filled in
behind it (proposal Risk 1: Go per-move timing coverage is the riskier data
source, so chess carries the timing-specific claims first).
"""

from gps.games.base import Game as GameBackend

__all__ = ["GameBackend"]
