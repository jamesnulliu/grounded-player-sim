"""Stockfish engine oracle (dense, CPU-bound).

Wraps a local Stockfish via ``python-chess``'s UCI engine bridge. Produces an
:class:`~gps.interface.EngineReference` whose ``candidate_values`` are
centipawns from the mover's side, with the search ``depth`` recorded (centipawn
loss is settings-dependent and must be reported, per the proposal).

Cost note (the reason :class:`LichessEvalOracle` exists): a full per-move,
all-legal-moves eval over millions of plies is the dominant cost of the whole
pipeline and is CPU-, not GPU-, bound. On a 12-core box this is only feasible
over a *filtered player subset*. ``evaluate`` uses MultiPV, so by default it
scores the top-``multipv`` moves; the *played* move's loss is what we actually
need, so :meth:`move_loss` scores a specific move precisely (a blunder is, by
definition, outside the top-PV set, so MultiPV alone would miss its size).

Heavy deps (``python-chess`` + a Stockfish binary) are imported/launched
lazily; constructing the oracle is cheap and import-safe.
"""

from __future__ import annotations

from gps.interface import EngineReference


class StockfishOracle:
    """A per-position Stockfish reference. Launches the engine on first use."""

    def __init__(
        self,
        engine_path: str = "stockfish",
        *,
        depth: int = 12,
        multipv: int = 8,
        threads: int = 1,
        hash_mb: int = 64,
    ) -> None:
        self.engine_path = engine_path
        self.depth = depth
        self.multipv = multipv
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            try:
                import chess.engine
            except ImportError as e:  # pragma: no cover - env-dependent
                raise ImportError(
                    "python-chess is required for StockfishOracle; install "
                    "the 'chess' extra."
                ) from e
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(
                    self.engine_path
                )
            except FileNotFoundError as e:  # pragma: no cover - env-dependent
                raise FileNotFoundError(
                    f"Stockfish binary not found at {self.engine_path!r}. "
                    "Install it (apt/conda/brew) or pass engine_path=..., or "
                    "use LichessEvalOracle to skip self-running Stockfish."
                ) from e
            self._engine.configure(
                {"Threads": self.threads, "Hash": self.hash_mb}
            )
        return self._engine

    @staticmethod
    def _board(position):
        import chess

        return (
            position
            if isinstance(position, chess.Board)
            else chess.Board(position)
        )

    def _cp(self, score, board) -> float:
        """Centipawns from the mover's view (mate -> large cp)."""
        return float(score.pov(board.turn).score(mate_score=100_000))

    def evaluate(
        self, position: object, legal_moves: tuple[str, ...]
    ) -> EngineReference:
        """MultiPV reference: values for the top-``multipv`` legal moves."""
        import chess
        import chess.engine

        board = self._board(position)
        engine = self._ensure_engine()
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=min(self.multipv, max(1, board.legal_moves.count())),
        )
        candidates: dict[str, float] = {}
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            candidates[pv[0].uci()] = self._cp(info["score"], board)
        best_move = max(candidates, key=candidates.get) if candidates else None
        return EngineReference(
            candidate_values=candidates,
            best_move=best_move,
            best_value=candidates.get(best_move) if best_move else None,
            unit="centipawn",
            depth=self.depth,
        )

    def move_loss(self, position: object, move: str) -> float | None:
        """Precise centipawn loss of a *specific* played move (>= 0).

        Two evals -- the position (best value) and the position after ``move``
        (the move's value, negated to the mover's view) -- so a blunder outside
        the top PV is still measured. This is the per-decision deviation target
        the trajectory builder ultimately needs.
        """
        import chess
        import chess.engine

        board = self._board(position)
        engine = self._ensure_engine()
        limit = chess.engine.Limit(depth=self.depth)
        best = self._cp(engine.analyse(board, limit)["score"], board)
        board.push(chess.Move.from_uci(move))
        # After the move it is the opponent's turn; negate to the mover's view.
        after = -self._cp(engine.analyse(board, limit)["score"], board)
        return max(0.0, best - after)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> StockfishOracle:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
