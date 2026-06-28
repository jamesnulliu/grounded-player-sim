"""Data ingestion: native game records -> shared trajectories.

This package turns *raw* game archives (Lichess PGN, later SGF) into the
game-agnostic :class:`~gps.train.base.Trajectory` records the trainer and eval
harness consume. The contract is deliberately narrow so the rest of the system
never learns it is looking at chess:

* :mod:`gps.data.sessions` -- segment a timestamp stream into sessions (the
  unlabeled-construct decision from design.md section 6; gap threshold is an
  explicit, ablatable parameter).
* :mod:`gps.data.lichess` -- stream a ``.pgn(.zst)`` archive, bucket games by
  player, extract per-move clocks, and assemble per-player trajectories. All
  chess/network/heavy dependencies (``python-chess``, ``zstandard``) are
  imported lazily so the pure-Python assembly logic stays unit-testable
  without them.

Keeping every chess-specific assumption inside this package (plus
:mod:`gps.games`) is what makes the framework portable to a non-game oracle
domain later: a new domain only writes a new adapter that emits the same
``Trajectory`` records (design.md section 11).
"""
