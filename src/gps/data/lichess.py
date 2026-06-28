"""Lichess PGN -> per-player :class:`~gps.train.base.Trajectory`.

The Lichess open database (https://database.lichess.org/, CC0) ships one
``.pgn.zst`` per month of *all* rated games, ordered chronologically -- **not**
by player. So ingestion is: stream the archive once, parse each game into a
light stdlib :class:`GameRecord`, bucket records by player, then assemble each
player's records into a time-ordered ``Trajectory`` (one ``DecisionPoint`` +
``Observation`` per move *that player* made).

Design boundary (so the framework stays portable, design.md section 11)
----------------------------------------------------------------------
The only function that touches ``python-chess`` / ``zstandard`` is the parser
(:func:`iter_game_records` / :func:`open_pgn`); it emits :class:`GameRecord` /
:class:`PlyRecord`, which are **pure stdlib**. Everything downstream
(:func:`bucket_by_player`, :func:`player_stats`, :func:`build_trajectory`)
works on those records with no heavy import, so the assembly logic -- the part
with the subtle correctness traps (history aliasing, session position, clock
arithmetic) -- is unit-testable on any box. A non-game domain reuses the
downstream half wholesale and only writes a new parser.

The engine oracle is **not** attached here. Per-move centipawn loss is
expensive and we are still deciding eval-set-vs-Stockfish empirically
(:mod:`gps.games.oracles`), so ``build_trajectory`` takes an *optional* oracle
and leaves ``engine_reference=None`` when none is given. Filtered trajectories
without oracle values are cheap to produce and re-usable.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

from gps.data.sessions import (
    DEFAULT_GAP_THRESHOLD_SECONDS,
    segment_sessions,
)
from gps.interface import (
    DecisionPoint,
    Game,
    Outcome,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import Observation
from gps.train.base import Trajectory, TrajectoryDataset

# --------------------------------------------------------------------------- #
# The stdlib record boundary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlyRecord:
    """One half-move within a game, from the moving side's view.

    All fields are plain Python so records survive without ``python-chess``.
    ``clock_before`` / ``time_spent`` are ``None`` when the archive carried no
    ``[%clk]`` for this ply.
    """

    fullmove_number: int
    side: str  # "w" | "b"
    uci: str
    fen_before: str
    legal_actions: tuple[str, ...]
    clock_before: float | None = None  # seconds on the mover's clock pre-move
    time_spent: float | None = None  # seconds the mover used on this move
    increment: float | None = None


@dataclass(frozen=True)
class GameRecord:
    """One parsed game, enough to build either player's perspective."""

    white: str
    black: str
    result: str  # "1-0" | "0-1" | "1/2-1/2" | "*"
    plies: tuple[PlyRecord, ...]
    white_elo: int | None = None
    black_elo: int | None = None
    white_is_bot: bool = False
    black_is_bot: bool = False
    time_control: str | None = None
    utc_start: float | None = None  # epoch seconds (UTCDate + UTCTime)

    def color_of(self, player: str) -> str | None:
        if player == self.white:
            return "white"
        if player == self.black:
            return "black"
        return None

    def opponent_of(self, player: str) -> str | None:
        if player == self.white:
            return self.black
        if player == self.black:
            return self.white
        return None

    def is_bot(self, player: str) -> bool:
        if player == self.white:
            return self.white_is_bot
        if player == self.black:
            return self.black_is_bot
        return False

    def elo_of(self, player: str) -> int | None:
        if player == self.white:
            return self.white_elo
        if player == self.black:
            return self.black_elo
        return None

    def won_by(self, player: str) -> bool | None:
        """``True`` win / ``False`` loss / ``None`` draw or unknown."""
        color = self.color_of(player)
        if color is None or self.result not in ("1-0", "0-1"):
            return None
        return (color == "white") == (self.result == "1-0")

    def duration_seconds(self) -> float:
        """Approximate wall-clock length: sum of both sides' think time.

        Used only to derive a game ``end`` for session segmentation. Falls
        back to ``0`` when clocks are absent (consecutive games then segment
        on start-to-start spacing, which is the conservative behaviour).
        """
        return sum(p.time_spent or 0.0 for p in self.plies)


# --------------------------------------------------------------------------- #
# Time-control parsing (stdlib)
# --------------------------------------------------------------------------- #


def parse_time_control(tc: str | None) -> tuple[float | None, float | None]:
    """``"180+2"`` -> ``(180.0, 2.0)``. Returns ``(None, None)`` for ``"-"``.

    The base clock seeds the first ply's ``clock_before``; the increment feeds
    the per-move time arithmetic.
    """
    if not tc or tc == "-":
        return None, None
    base, _, inc = tc.partition("+")
    try:
        return float(base), float(inc or 0.0)
    except ValueError:
        return None, None


# --------------------------------------------------------------------------- #
# The parser (the only python-chess / zstandard dependent code)
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def open_pgn(path: str):
    """Open a ``.pgn`` or ``.pgn.zst`` as a decoded text stream.

    ``.zst`` is decompressed in a stream (``zstandard``) so a multi-GB monthly
    archive is never expanded to disk or held in memory. Plain ``.pgn`` is
    opened directly. ``python-chess`` reads from the yielded text handle.
    """
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "zstandard is required to read .pgn.zst archives; "
                "install it (pip install zstandard)."
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


def _parse_utc(headers) -> float | None:
    date = headers.get("UTCDate")  # "2026.06.01"
    time = headers.get("UTCTime")  # "12:34:56"
    if not date or not time or "?" in date:
        return None
    try:
        dt = datetime.strptime(f"{date} {time}", "%Y.%m.%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        return dt.timestamp()
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "", "?") else None
    except ValueError:
        return None


def iter_game_records(
    stream, *, max_games: int | None = None
) -> Iterator[GameRecord]:
    """Parse a PGN text stream into :class:`GameRecord`s (lazy).

    Reads one game at a time with ``python-chess`` so memory stays flat over a
    full month. For each ply it records the FEN *before* the move, the legal
    move set in that position, and -- when ``[%clk]`` is present -- the mover's
    clock before the move and the time they spent (``prev_clock - clk +
    increment``, the standard Lichess reconstruction).
    """
    try:
        import chess
        import chess.pgn
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "python-chess is required to parse PGN; install the 'chess' "
            "extra (pip install python-chess)."
        ) from e

    count = 0
    while True:
        if max_games is not None and count >= max_games:
            return
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        h = game.headers
        base, inc = parse_time_control(h.get("TimeControl"))
        # Per-side clock carried forward to compute think time.
        prev_clock = {chess.WHITE: base, chess.BLACK: base}

        board = game.board()
        plies: list[PlyRecord] = []
        for node in game.mainline():
            mover = board.turn  # side to move == side that made node.move
            fen_before = board.fen()
            legal = tuple(sorted(m.uci() for m in board.legal_moves))
            clk = (
                node.clock()
            )  # mover's remaining time AFTER the move, or None
            clock_before = prev_clock[mover]
            time_spent = None
            if clk is not None and clock_before is not None:
                time_spent = max(0.0, clock_before - clk + (inc or 0.0))
            if clk is not None:
                prev_clock[mover] = clk
            plies.append(
                PlyRecord(
                    fullmove_number=board.fullmove_number,
                    side="w" if mover == chess.WHITE else "b",
                    uci=node.move.uci(),
                    fen_before=fen_before,
                    legal_actions=legal,
                    clock_before=clock_before,
                    time_spent=time_spent,
                    increment=inc,
                )
            )
            board.push(node.move)

        yield GameRecord(
            white=h.get("White", "?"),
            black=h.get("Black", "?"),
            result=h.get("Result", "*"),
            plies=tuple(plies),
            white_elo=_to_int(h.get("WhiteElo")),
            black_elo=_to_int(h.get("BlackElo")),
            white_is_bot=h.get("WhiteTitle") == "BOT",
            black_is_bot=h.get("BlackTitle") == "BOT",
            time_control=h.get("TimeControl"),
            utc_start=_parse_utc(h),
        )
        count += 1


# --------------------------------------------------------------------------- #
# Bucketing + cohort selection (stdlib)
# --------------------------------------------------------------------------- #


def bucket_by_player(
    records: Iterable[GameRecord],
    *,
    players: set[str] | None = None,
    include_bots: bool = False,
    exclude_bot_opponents: bool = True,
) -> dict[str, list[GameRecord]]:
    """Map each human player -> the list of their games (both colours).

    A game appears under *both* of its players (each will be read from their
    own perspective downstream). ``players``, when given, restricts buckets to
    that cohort so a full-month pass stays cheap. Bot accounts are dropped as
    *tracked* players (``include_bots``) and, by default, games against a bot
    are dropped too (``exclude_bot_opponents``) since bot opponents distort the
    human dynamics we model.
    """
    buckets: dict[str, list[GameRecord]] = {}
    for rec in records:
        sides = (
            (rec.white, rec.white_is_bot, rec.black_is_bot),
            (rec.black, rec.black_is_bot, rec.white_is_bot),
        )
        for name, is_bot, opp_is_bot in sides:
            if name == "?":
                continue
            if players is not None and name not in players:
                continue
            if is_bot and not include_bots:
                continue
            if opp_is_bot and exclude_bot_opponents:
                continue
            buckets.setdefault(name, []).append(rec)
    return buckets


@dataclass(frozen=True)
class PlayerStats:
    """Volume/longitudinality summary used for cohort selection."""

    player_id: str
    n_games: int
    n_sessions: int
    span_days: float


def player_stats(
    buckets: dict[str, list[GameRecord]],
    *,
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
) -> dict[str, PlayerStats]:
    """Per-player game/session counts + career span (days)."""
    out: dict[str, PlayerStats] = {}
    for player, games in buckets.items():
        timed = sorted(
            (g for g in games if g.utc_start is not None),
            key=lambda g: g.utc_start,
        )
        spans = [
            (g.utc_start, g.utc_start + g.duration_seconds()) for g in timed
        ]
        n_sessions = len(segment_sessions(spans, gap_threshold_seconds))
        span_days = (
            (timed[-1].utc_start - timed[0].utc_start) / 86400.0
            if len(timed) >= 2
            else 0.0
        )
        out[player] = PlayerStats(
            player_id=player,
            n_games=len(games),
            n_sessions=n_sessions,
            span_days=span_days,
        )
    return out


def select_players(
    stats: dict[str, PlayerStats],
    *,
    min_games: int = 50,
    min_sessions: int = 3,
) -> list[str]:
    """Cohort of players with enough volume *and* multi-session history.

    Both gates matter: the future-behavior split (E-C3) needs several sessions
    per player, not just many games in one sitting.
    """
    chosen = [
        s.player_id
        for s in stats.values()
        if s.n_games >= min_games and s.n_sessions >= min_sessions
    ]
    chosen.sort(key=lambda p: stats[p].n_games, reverse=True)
    return chosen


# --------------------------------------------------------------------------- #
# Trajectory assembly (stdlib)
# --------------------------------------------------------------------------- #


def _phase_for(fullmove: int) -> str:
    if fullmove <= 10:
        return "opening"
    if fullmove <= 30:
        return "middlegame"
    return "endgame"


def build_trajectory(
    player: str,
    games: Iterable[GameRecord],
    *,
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
    oracle=None,
    game_id: Game = Game.CHESS,
) -> Trajectory:
    """Assemble one player's games into a time-ordered ``Trajectory``.

    One ``DecisionPoint`` (+ matching ``Observation``) per move *this player*
    made. ``recent_outcomes`` is **snapshotted per game** -- never the shared
    mutable stream -- to avoid the aliasing bug recorded in design.md section 5
    (every decision otherwise seeing the final session history). ``oracle``,
    if given, is consulted per decision for the ``engine_reference``; otherwise
    it is left ``None`` and attached later.
    """
    ordered = sorted(
        games, key=lambda g: (g.utc_start is None, g.utc_start or 0.0)
    )
    spans = [
        (
            g.utc_start or float(i),
            (g.utc_start or float(i)) + g.duration_seconds(),
        )
        for i, g in enumerate(ordered)
    ]
    sessions = segment_sessions(spans, gap_threshold_seconds)
    pos_in_session = [0] * len(ordered)
    for session in sessions:
        for pos, idx in enumerate(session):
            pos_in_session[idx] = pos

    decisions: list[DecisionPoint] = []
    observations: list[Observation] = []
    prior: list[Outcome] = []  # this player's completed games so far

    for idx, game in enumerate(ordered):
        color = game.color_of(player)
        if color is None:
            continue
        side = "w" if color == "white" else "b"
        # One immutable snapshot of "how the session has gone so far",
        # shared by every decision *within* this game (constant within a game)
        # but never mutated -- so no two games alias the same history.
        stream = OutcomeStream(
            recent=list(prior), session_position=pos_in_session[idx]
        )
        elo = game.elo_of(player)
        opp_elo = game.elo_of(game.opponent_of(player) or "")
        for ply in game.plies:
            if ply.side != side:
                continue
            ref = None
            if oracle is not None:
                ref = oracle.evaluate(ply.fen_before, ply.legal_actions)
            decisions.append(
                DecisionPoint(
                    game=game_id,
                    player_id=player,
                    state=ply.fen_before,
                    legal_actions=ply.legal_actions,
                    engine_reference=ref,
                    time_signal=TimeSignal(
                        time_remaining=ply.clock_before,
                        increment=ply.increment,
                        time_spent=ply.time_spent,
                        move_number=ply.fullmove_number,
                        phase=_phase_for(ply.fullmove_number),
                    ),
                    recent_outcomes=stream,
                    context={
                        "color": color,
                        "player_elo": elo,
                        "opponent_elo": opp_elo,
                        "time_control": game.time_control,
                    },
                )
            )
            observations.append(
                Observation(move=ply.uci, time_spent=ply.time_spent)
            )
        # Game finished: append its outcome for subsequent games to see.
        prior.append(
            Outcome(
                won=game.won_by(player),
                time_scramble=any(
                    p.clock_before is not None
                    and p.side == side
                    and p.clock_before <= 10.0
                    for p in game.plies
                ),
            )
        )

    return Trajectory(
        player_id=player, decisions=decisions, observations=observations
    )


def build_dataset(
    buckets: dict[str, list[GameRecord]],
    *,
    players: Iterable[str] | None = None,
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
    oracle=None,
    min_decisions: int = 1,
) -> TrajectoryDataset:
    """Build a ``TrajectoryDataset`` for ``players`` (default: all buckets).

    Players whose assembled trajectory has fewer than ``min_decisions`` moves
    are dropped (they cannot support a temporal split).
    """
    names = list(players) if players is not None else list(buckets)
    trajectories = []
    for name in names:
        traj = build_trajectory(
            name,
            buckets.get(name, []),
            gap_threshold_seconds=gap_threshold_seconds,
            oracle=oracle,
        )
        if len(traj.decisions) >= min_decisions:
            trajectories.append(traj)
    return TrajectoryDataset(trajectories=trajectories)
