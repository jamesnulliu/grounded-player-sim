"""Lichess evaluation-set oracle (free, partial coverage).

Backed by the published ``lichess_db_eval.jsonl(.zst)`` (~388M positions,
https://database.lichess.org/#evals): one JSON object per line mapping a FEN
to one or more Stockfish ``evals`` (each a list of ``pvs`` with ``cp`` /
``mate`` and a UCI ``line``, plus ``depth`` / ``knodes``).

The catch this oracle exists to *measure*: the set is keyed by position and
skews to analyzed/popular positions, so an arbitrary blitz position is often
absent. Coverage is therefore partial, and ``evaluate`` returns an
``EngineReference`` with empty ``candidate_values`` (so ``loss_of`` is
``None``, i.e. "unknown") for an uncovered position rather than a fake zero.

Because the full file is far too large to hold in memory, the practical entry
point is :meth:`from_subset`: collect the FENs your small cohort actually
reaches, stream the file once, and keep only matching positions. Pair it with
:func:`eval_set_coverage` to get the hit rate that decides
eval-set-vs-Stockfish (see :mod:`gps.games.oracles`).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Iterator

from gps.interface import EngineReference

#: cp magnitude a forced mate maps to (sign = who mates), matching
#: StockfishOracle so the two oracles are comparable on the overlap.
MATE_CP = 100_000


def normalize_fen(fen: str) -> str:
    """Key on the position-defining first four FEN fields.

    Drops the half-move clock and full-move number so transpositions and
    differing move counters still match the eval set.
    """
    return " ".join(fen.split()[:4])


@contextlib.contextmanager
def _open_text(path: str):
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "zstandard is required to read .jsonl.zst; pip install "
                "zstandard."
            ) from e
        import io

        fh = open(path, "rb")
        try:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        finally:
            fh.close()
    else:
        fh = open(path, encoding="utf-8", errors="replace")
        try:
            yield fh
        finally:
            fh.close()


def iter_eval_lines(path: str) -> Iterator[dict]:
    """Yield parsed JSON objects from the eval archive (lazy, streaming)."""
    with _open_text(path) as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def _best_eval(entry: dict) -> dict | None:
    """The deepest ``evals`` block for a position (most trustworthy)."""
    evals = entry.get("evals") or entry.get("eval")
    if not evals:
        return None
    return max(evals, key=lambda e: e.get("depth", 0))


def _cp_white_pov(pv: dict) -> float:
    if "cp" in pv:
        return float(pv["cp"])
    if "mate" in pv:
        m = pv["mate"]
        return float(MATE_CP if m > 0 else -MATE_CP)
    return 0.0


class LichessEvalOracle:
    """Position -> :class:`EngineReference` from the cached eval subset."""

    def __init__(
        self,
        index: dict[str, dict],
        *,
        white_pov: bool = True,
    ) -> None:
        #: normalized-FEN -> {"moves": {uci: cp_white_pov}, "depth": int}
        self._index = index
        #: Lichess cp is stored white-positive; we flip to mover-positive in
        #: :meth:`evaluate` to honour EngineReference's "higher == better for
        #: the mover" contract. Exposed as a flag so the empirical comparison
        #: against StockfishOracle can confirm the convention.
        self.white_pov = white_pov

    @classmethod
    def from_subset(
        cls,
        eval_path: str,
        wanted_fens: Iterable[str],
        *,
        white_pov: bool = True,
    ) -> LichessEvalOracle:
        """Stream the archive once, keeping only ``wanted_fens``."""
        wanted = {normalize_fen(f) for f in wanted_fens}
        index: dict[str, dict] = {}
        for entry in iter_eval_lines(eval_path):
            fen = entry.get("fen")
            if not fen:
                continue
            key = normalize_fen(fen)
            if key not in wanted or key in index:
                continue
            best = _best_eval(entry)
            if not best:
                continue
            moves = {}
            for pv in best.get("pvs", []):
                line = pv.get("line") or ""
                first = line.split(" ", 1)[0] if line else None
                if first:
                    moves[first] = _cp_white_pov(pv)
            index[key] = {"moves": moves, "depth": best.get("depth")}
            if len(index) == len(wanted):
                break
        return cls(index, white_pov=white_pov)

    def coverage(self, wanted_fens: Iterable[str]) -> float:
        """Fraction of ``wanted_fens`` present in the index (0..1)."""
        wanted = {normalize_fen(f) for f in wanted_fens}
        if not wanted:
            return 0.0
        return sum(1 for f in wanted if f in self._index) / len(wanted)

    def evaluate(
        self, position: object, legal_moves: tuple[str, ...]
    ) -> EngineReference:
        fen = position if isinstance(position, str) else str(position)
        rec = self._index.get(normalize_fen(fen))
        if rec is None:
            # Uncovered: empty reference -> loss_of(...) is None ("unknown").
            return EngineReference(
                candidate_values={}, unit="centipawn", depth=None
            )
        black_to_move = (
            fen.split()[1] == "b" if len(fen.split()) > 1 else False
        )
        flip = self.white_pov and black_to_move
        candidates = {
            uci: (-cp if flip else cp) for uci, cp in rec["moves"].items()
        }
        best_move = max(candidates, key=candidates.get) if candidates else None
        return EngineReference(
            candidate_values=candidates,
            best_move=best_move,
            best_value=candidates.get(best_move) if best_move else None,
            unit="centipawn",
            depth=rec.get("depth"),
        )


def eval_set_coverage(
    eval_path: str, wanted_fens: Iterable[str]
) -> tuple[int, int, float]:
    """``(covered, total, fraction)`` for ``wanted_fens`` against the archive.

    The empirical input to the eval-set-vs-Stockfish decision: high coverage
    -> use the (free) eval set; low coverage -> self-run Stockfish on the
    subset. Streams the file once.
    """
    oracle = LichessEvalOracle.from_subset(eval_path, wanted_fens)
    wanted = {normalize_fen(f) for f in wanted_fens}
    covered = sum(1 for f in wanted if f in oracle._index)
    total = len(wanted)
    return covered, total, (covered / total if total else 0.0)
