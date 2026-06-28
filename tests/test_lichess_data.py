"""Lichess ingestion tests.

The assembly half (records -> trajectories, bucketing, sessions, the eval
oracle) is pure stdlib and tested directly. The PGN parser needs
``python-chess`` and is smoke-tested behind ``importorskip`` so the suite still
runs without it.
"""

import json

import pytest

from gps.data.lichess import (
    GameRecord,
    PlyRecord,
    bucket_by_player,
    build_trajectory,
    parse_time_control,
    player_stats,
    select_players,
)
from gps.data.sessions import segment_sessions, session_positions
from gps.games.oracles.lichess_eval import LichessEvalOracle, eval_set_coverage
from gps.latent.structured import history_features

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _ply(side, uci, fen, *, clock=None, spent=None, fullmove=1):
    return PlyRecord(
        fullmove_number=fullmove,
        side=side,
        uci=uci,
        fen_before=fen,
        legal_actions=("e2e4", "d2d4", "g1f3", "g8f6", "e7e5"),
        clock_before=clock,
        time_spent=spent,
        increment=2.0,
    )


def _g1():  # alice (white) beats bob; alice makes 2 moves
    return GameRecord(
        white="alice",
        black="bob",
        result="1-0",
        white_elo=1500,
        black_elo=1480,
        time_control="180+2",
        utc_start=1000.0,
        plies=(
            _ply("w", "e2e4", START, clock=180.0, spent=3.0, fullmove=1),
            _ply("b", "e7e5", START, clock=180.0, spent=2.0, fullmove=1),
            _ply("w", "g1f3", START, clock=179.0, spent=4.0, fullmove=2),
        ),
    )


def _g2():  # alice (black) loses to carol; alice makes 1 move
    return GameRecord(
        white="carol",
        black="alice",
        result="1-0",
        white_elo=1490,
        black_elo=1500,
        time_control="180+2",
        utc_start=1015.0,
        plies=(
            _ply("w", "d2d4", START, clock=180.0, spent=2.0, fullmove=1),
            _ply("b", "g8f6", START, clock=180.0, spent=5.0, fullmove=1),
        ),
    )


def test_parse_time_control():
    assert parse_time_control("180+2") == (180.0, 2.0)
    assert parse_time_control("600+0") == (600.0, 0.0)
    assert parse_time_control("-") == (None, None)
    assert parse_time_control(None) == (None, None)


def test_session_positions_matches_segments():
    games = [(0.0, 100.0), (200.0, 300.0), (4000.0, 4100.0)]
    assert segment_sessions(games, 1800.0) == [[0, 1], [2]]
    assert session_positions(games, 1800.0) == [0, 1, 0]


def test_build_trajectory_one_decision_per_own_move():
    traj = build_trajectory("alice", [_g1(), _g2()])
    # alice: 2 moves in g1 + 1 move in g2 = 3 decisions/observations.
    assert len(traj.decisions) == 3
    assert [o.move for o in traj.observations] == ["e2e4", "g1f3", "g8f6"]


def test_build_trajectory_snapshots_history_without_aliasing():
    traj = build_trajectory("alice", [_g1(), _g2()])
    d_g1_first, _d_g1_second, d_g2 = traj.decisions
    # g1 decisions saw no prior games; the later g2 outcome must NOT leak back.
    assert d_g1_first.recent_outcomes.recent == []
    assert d_g1_first.recent_outcomes.session_position == 0
    # g2 decision sees exactly the g1 result (alice won g1) and is later in
    # the same session.
    assert len(d_g2.recent_outcomes.recent) == 1
    assert d_g2.recent_outcomes.recent[0].won is True
    assert d_g2.recent_outcomes.session_position == 1


def test_build_trajectory_clock_and_context():
    traj = build_trajectory("alice", [_g1(), _g2()])
    d0 = traj.decisions[0]
    assert d0.time_signal.time_remaining == 180.0
    assert d0.context["color"] == "white"
    assert d0.context["player_elo"] == 1500
    assert d0.context["opponent_elo"] == 1480
    # The injector's feature extractor must accept a real built decision point.
    feats = history_features(d0)
    assert set(feats) == {"time_pressure", "post_loss", "fatigue", "momentum"}


def test_build_trajectory_attaches_oracle_reference():
    class _StubOracle:
        def evaluate(self, position, legal_moves):
            from gps.interface import EngineReference

            return EngineReference(
                candidate_values={"e2e4": 30.0, "g1f3": 5.0},
                best_move="e2e4",
                best_value=30.0,
                depth=12,
            )

    traj = build_trajectory("alice", [_g1()], oracle=_StubOracle())
    ref = traj.decisions[0].engine_reference
    assert ref is not None and ref.loss_of("g1f3") == 25.0


def test_bucket_excludes_bots_and_bot_opponents():
    bot_game = GameRecord(
        white="sf-bot",
        black="alice",
        result="0-1",
        white_is_bot=True,
        utc_start=900.0,
        plies=(_ply("b", "g8f6", START, clock=180.0, spent=1.0),),
    )
    buckets = bucket_by_player([_g1(), _g2(), bot_game])
    assert "sf-bot" not in buckets  # bots are never tracked players
    # alice's bot game is dropped (default exclude_bot_opponents=True).
    assert {g.utc_start for g in buckets["alice"]} == {1000.0, 1015.0}


def test_player_stats_and_selection():
    buckets = bucket_by_player([_g1(), _g2()])
    stats = player_stats(buckets)
    assert stats["alice"].n_games == 2
    assert stats["alice"].n_sessions == 1  # both games one sitting
    # Gates filter low-volume players out.
    assert select_players(stats, min_games=2, min_sessions=1) == ["alice"]
    assert select_players(stats, min_games=5, min_sessions=1) == []


def test_lichess_eval_oracle_coverage_and_values(tmp_path):
    pos4 = " ".join(START.split()[:4])  # 4-field key as the eval set stores it
    path = tmp_path / "evals.jsonl"
    path.write_text(
        json.dumps(
            {
                "fen": pos4,
                "evals": [
                    {
                        "depth": 30,
                        "pvs": [
                            {"cp": 31, "line": "e2e4 e7e5"},
                            {"cp": 20, "line": "d2d4 d7d5"},
                        ],
                    }
                ],
            }
        )
        + "\n"
    )
    other = "8/8/8/8/8/8/8/8 w - - 0 1"
    covered, total, frac = eval_set_coverage(str(path), [START, other])
    assert (covered, total) == (1, 2) and frac == 0.5

    oracle = LichessEvalOracle.from_subset(str(path), [START])
    ref = oracle.evaluate(START, ("e2e4", "d2d4"))
    assert ref.best_move == "e2e4"
    assert ref.loss_of("d2d4") == 11.0  # 31 - 20, white to move (no flip)
    # Uncovered position -> "unknown", not a fabricated zero.
    assert oracle.evaluate(other, ()).loss_of("a1a2") is None


def test_pgn_parser_smoke():
    chess_pgn = pytest.importorskip("chess.pgn")
    import io

    from gps.data.lichess import iter_game_records

    pgn = (
        '[White "alice"]\n'
        '[Black "bob"]\n'
        '[Result "1-0"]\n'
        '[UTCDate "2026.06.01"]\n'
        '[UTCTime "12:00:00"]\n'
        '[WhiteElo "1500"]\n'
        '[TimeControl "180+2"]\n\n'
        "1. e4 { [%clk 0:03:00] } e5 { [%clk 0:03:00] } "
        "2. Nf3 { [%clk 0:02:58] } 1-0\n"
    )
    assert chess_pgn is not None  # importorskip guard
    (rec,) = list(iter_game_records(io.StringIO(pgn)))
    assert rec.white == "alice" and rec.white_elo == 1500
    assert rec.utc_start is not None
    assert len(rec.plies) == 3
    p1 = rec.plies[0]
    assert p1.side == "w" and p1.uci == "e2e4"
    assert "e2e4" in p1.legal_actions and len(p1.legal_actions) == 20
    assert p1.clock_before == 180.0 and p1.time_spent == 2.0  # 180-180+2
    # White's 2nd move: 180 -> 178 with +2 increment => 4s spent.
    assert rec.plies[2].time_spent == 4.0
