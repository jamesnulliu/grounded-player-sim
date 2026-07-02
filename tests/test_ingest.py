"""Ingest pipeline tests: serialization, speed class, splitting, 2-pass driver.

The serialization + speed/split helpers are pure stdlib and tested directly.
The PGN-parsing paths (summaries, parallel parse, end-to-end ingest) need
``python-chess`` and are guarded behind ``importorskip`` so the suite still
runs without it.
"""

import io
import json

import pytest

from gps.data.lichess import (
    GameRecord,
    PlyRecord,
    build_trajectory,
    speed_class,
    split_pgn_games,
)
from gps.data.store import (
    load_dataset,
    save_dataset,
    trajectory_from_dict,
    trajectory_to_dict,
)
from gps.train.base import TrajectoryDataset

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --------------------------------------------------------------------------- #
# Fixtures (mirror tests/test_lichess_data.py)
# --------------------------------------------------------------------------- #


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


class _StubOracle:
    def evaluate(self, position, legal_moves):
        from gps.interface import EngineReference

        return EngineReference(
            candidate_values={"e2e4": 30.0, "g1f3": 5.0},
            best_move="e2e4",
            best_value=30.0,
            depth=12,
        )


# --------------------------------------------------------------------------- #
# speed_class
# --------------------------------------------------------------------------- #


def test_speed_class():
    assert speed_class("15+0") == "ultrabullet"  # est 15
    assert speed_class("60+0") == "bullet"  # est 60
    assert speed_class("120+1") == "bullet"  # est 160
    assert speed_class("180+2") == "blitz"  # est 260
    assert speed_class("300+0") == "blitz"  # est 300
    assert speed_class("600+0") == "rapid"  # est 600
    assert speed_class("1800+0") == "classical"  # est 1800
    assert speed_class("-") == "correspondence"
    assert speed_class(None) == "correspondence"
    # Increment can promote a short base across a boundary (40*inc).
    assert speed_class("60+5") == "blitz"  # est 60 + 200 = 260


# --------------------------------------------------------------------------- #
# split_pgn_games
# --------------------------------------------------------------------------- #


def _pgn_game(white, black, result, *, utctime, tc="180+2"):
    return (
        '[Event "Rated Blitz game"]\n'
        f'[White "{white}"]\n'
        f'[Black "{black}"]\n'
        f'[Result "{result}"]\n'
        '[UTCDate "2026.06.01"]\n'
        f'[UTCTime "{utctime}"]\n'
        '[WhiteElo "1500"]\n'
        '[BlackElo "1480"]\n'
        f'[TimeControl "{tc}"]\n\n'
        "1. e4 { [%clk 0:03:00] } e5 { [%clk 0:03:00] } "
        "2. Nf3 { [%clk 0:02:58] } 1-0\n\n"
    )


def test_split_pgn_games_counts_and_reparse():
    chess_pgn = pytest.importorskip("chess.pgn")
    from gps.data.lichess import iter_game_records

    pgn = (
        _pgn_game("alice", "bob", "1-0", utctime="12:00:00")
        + _pgn_game("carol", "alice", "0-1", utctime="12:10:00")
        + _pgn_game("alice", "dave", "1-0", utctime="13:00:00")
    )
    chunks = list(split_pgn_games(io.StringIO(pgn)))
    assert len(chunks) == 3
    assert chess_pgn is not None
    # Every chunk is independently parseable into exactly one game.
    for chunk in chunks:
        recs = list(iter_game_records(io.StringIO(chunk)))
        assert len(recs) == 1


def test_split_pgn_ignores_trailing_whitespace():
    assert list(split_pgn_games(io.StringIO("\n\n  \n"))) == []


# --------------------------------------------------------------------------- #
# Header-only summaries + parallel parse
# --------------------------------------------------------------------------- #


def test_iter_game_summaries():
    pytest.importorskip("chess.pgn")
    from gps.data.lichess import iter_game_summaries

    pgn = _pgn_game("alice", "bob", "1-0", utctime="12:00:00")
    (s,) = list(iter_game_summaries(io.StringIO(pgn)))
    assert s.white == "alice" and s.black == "bob"
    assert s.white_elo == 1500 and s.time_control == "180+2"
    assert s.utc_start is not None
    assert s.duration_seconds() == 0.0  # headers only


def test_parallel_parse_serial_and_pooled_agree():
    pytest.importorskip("chess.pgn")
    import os
    import tempfile

    from gps.data.lichess import iter_game_records_parallel

    pgn = (
        _pgn_game("alice", "bob", "1-0", utctime="12:00:00")
        + _pgn_game("carol", "dave", "0-1", utctime="12:10:00", tc="600+0")
        + _pgn_game("alice", "dave", "1-0", utctime="13:00:00")
    )
    fd, path = tempfile.mkstemp(suffix=".pgn")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(pgn)

        # Serial, blitz only -> the two alice blitz games (the 600+0 is rapid).
        serial = list(
            iter_game_records_parallel(path, speed="blitz", workers=1)
        )
        assert {(r.white, r.black) for r in serial} == {
            ("alice", "bob"),
            ("alice", "dave"),
        }

        # Cohort filter: only games involving carol.
        carol = list(
            iter_game_records_parallel(path, players={"carol"}, workers=1)
        )
        assert {(r.white, r.black) for r in carol} == {("carol", "dave")}

        # Pooled path returns the same set (order not guaranteed).
        pooled = list(
            iter_game_records_parallel(
                path, speed="blitz", workers=2, batch_size=1
            )
        )
        assert {(r.white, r.black) for r in pooled} == {
            (r.white, r.black) for r in serial
        }
    finally:
        os.remove(path)


# --------------------------------------------------------------------------- #
# Trajectory serialization round-trip
# --------------------------------------------------------------------------- #


def test_trajectory_dict_roundtrip_with_oracle():
    traj = build_trajectory("alice", [_g1(), _g2()], oracle=_StubOracle())
    back = trajectory_from_dict(trajectory_to_dict(traj))
    # Structural equality via the canonical dict form.
    assert trajectory_to_dict(back) == trajectory_to_dict(traj)
    # Spot-check the engine reference survived with working loss arithmetic.
    ref = back.decisions[0].engine_reference
    assert ref is not None and ref.loss_of("g1f3") == 25.0
    assert ref.depth == 12
    # Observations + timing preserved.
    assert [o.move for o in back.observations] == ["e2e4", "g1f3", "g8f6"]
    assert back.decisions[0].time_signal.time_remaining == 180.0


def test_store_preserves_per_game_stream_sharing():
    # g1 has two alice moves -> both decisions must share ONE stream object,
    # both before and after a round-trip (the dedup/realias contract).
    traj = build_trajectory("alice", [_g1(), _g2()])
    d0, d1, _d2 = traj.decisions
    assert d0.recent_outcomes is d1.recent_outcomes  # built shared
    back = trajectory_from_dict(trajectory_to_dict(traj))
    b0, b1, b2 = back.decisions
    assert b0.recent_outcomes is b1.recent_outcomes  # re-aliased on load
    assert b0.recent_outcomes is not b2.recent_outcomes  # different game
    # And the serialized form stored each unique stream once.
    d = trajectory_to_dict(traj)
    assert len(d["streams"]) == 2  # one per game alice played


def test_save_load_dataset_plain_and_gzip(tmp_path):
    traj = build_trajectory("alice", [_g1(), _g2()], oracle=_StubOracle())
    ds = TrajectoryDataset(trajectories=[traj])
    for name in ("dataset.jsonl", "dataset.jsonl.gz"):
        path = str(tmp_path / name)
        save_dataset(ds, path)
        loaded = load_dataset(path)
        assert len(loaded) == 1
        assert loaded.trajectories[0].player_id == "alice"
        assert trajectory_to_dict(loaded.trajectories[0]) == (
            trajectory_to_dict(traj)
        )


def test_dataset_header_schema_mismatch_raises(tmp_path):
    path = str(tmp_path / "bad.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema_version": 999}) + "\n")
    with pytest.raises(ValueError, match="schema"):
        load_dataset(path)


# --------------------------------------------------------------------------- #
# End-to-end ingest driver
# --------------------------------------------------------------------------- #


def test_run_ingest_end_to_end(tmp_path):
    pytest.importorskip("chess.pgn")
    from gps.data.ingest import run_ingest

    # alice plays 3 blitz games; opponents appear once each. One rapid game
    # for carol must be excluded by the blitz filter.
    pgn = (
        _pgn_game("alice", "bob", "1-0", utctime="12:00:00")
        + _pgn_game("carol", "alice", "0-1", utctime="12:10:00")
        + _pgn_game("alice", "dave", "1-0", utctime="13:00:00")
        + _pgn_game("carol", "dave", "1-0", utctime="14:00:00", tc="600+0")
    )
    archive = tmp_path / "mini.pgn"
    archive.write_text(pgn)
    out = tmp_path / "out"

    manifest = run_ingest(
        str(archive),
        str(out),
        speed="blitz",
        min_games=2,
        min_sessions=1,
        workers=1,
        verbose=False,
    )

    # Only alice clears min_games=2 within blitz.
    assert manifest["n_players_selected"] == 1
    assert manifest["players"]["alice"]["n_games"] == 3
    assert manifest["n_decisions_total"] > 0

    # Files exist and the dataset reloads to the same cohort.
    assert (out / "manifest.json").exists()
    ds = load_dataset(str(out / "dataset.jsonl.gz"))
    assert ds.players() == {"alice"}
    # Decision count in the dataset matches the manifest.
    assert (
        sum(len(t.decisions) for t in ds.trajectories)
        == (manifest["n_decisions_total"])
    )
