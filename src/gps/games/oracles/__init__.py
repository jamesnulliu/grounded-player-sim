"""Engine oracles -- the per-decision quality reference (the project's moat).

Both oracles satisfy the :class:`~gps.games.base.EngineOracle` protocol
(``evaluate(position, legal_moves) -> EngineReference``) so the data pipeline
and eval harness consume them interchangeably. The point of having two is to
**decide eval-set-vs-Stockfish empirically** on a small sample before
committing CPU to a full-pool eval (design.md / milestone_a.md section 7):

* :class:`~gps.games.oracles.stockfish.StockfishOracle` -- runs Stockfish
  locally. Dense (covers every position) but CPU-bound; on a 12-core box only
  feasible over a filtered player subset.
* :class:`~gps.games.oracles.lichess_eval.LichessEvalOracle` -- looks positions
  up in the published ``lichess_db_eval`` set. Free (no compute) but
  **partial coverage** -- the set skews to analyzed/popular positions, so an
  arbitrary blitz position is often absent.

The decision procedure: collect the FENs from a small cohort, measure
:func:`~gps.games.oracles.lichess_eval.eval_set_coverage`, and compare the two
oracles' values on the covered overlap. If coverage is high enough, use the
eval set and skip Stockfish; otherwise self-run Stockfish on the subset.
"""

from gps.games.oracles.lichess_eval import (
    LichessEvalOracle,
    eval_set_coverage,
)
from gps.games.oracles.stockfish import StockfishOracle

__all__ = ["StockfishOracle", "LichessEvalOracle", "eval_set_coverage"]
