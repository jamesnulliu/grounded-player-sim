"""``load_kt_csv`` builds a valid real-KT ``TrajectoryDataset`` from the
standard 5-column preprocessed export (tested on a tiny inline CSV -- no
network, no external data)."""

from __future__ import annotations

from gps.data.kt_csv import load_kt_csv
from gps.interface import Game


def _write_csv(path):
    # user_id, item_id, timestamp, correct, skill_id  (tab-separated)
    rows = [
        "user_id\titem_id\ttimestamp\tcorrect\tskill_id",
    ]
    # student A: 6 responses, skill 1 (easy) mostly correct, skill 2 harder
    for i, (skill, correct) in enumerate(
        [(1, 1), (1, 1), (2, 0), (2, 1), (1, 1), (2, 0)]
    ):
        rows.append(f"A\t{100 + i}\t{i}\t{correct}\t{skill}")
    # student B: 5 responses
    for i, (skill, correct) in enumerate(
        [(1, 0), (2, 0), (1, 1), (2, 0), (1, 1)]
    ):
        rows.append(f"B\t{200 + i}\t{i}\t{correct}\t{skill}")
    # student C: only 2 responses -> filtered by min_responses
    rows.append("C\t300\t0\t1\t1")
    rows.append("C\t301\t1\t1\t1")
    path.write_text("\n".join(rows) + "\n")


def test_load_kt_csv_builds_dataset(tmp_path):
    csv_path = tmp_path / "kt.csv"
    _write_csv(csv_path)
    ds = load_kt_csv(
        str(csv_path), n_students=10, min_responses=5, max_len=200
    )

    # C is dropped (2 < 5); A and B kept.
    assert len(ds.trajectories) == 2
    ids = {t.player_id for t in ds.trajectories}
    assert ids == {"A", "B"}

    a = next(t for t in ds.trajectories if t.player_id == "A")
    assert len(a.decisions) == len(a.observations) == 6

    d0 = a.decisions[0]
    assert d0.game is Game.KNOWLEDGE_TRACING
    assert d0.legal_actions == ("correct", "incorrect")
    assert d0.context == {"synthetic": False}
    # single difficulty feature in [0, 1]
    assert len(d0.state) == 1
    assert 0.0 <= d0.state[0] <= 1.0
    # temporal structure: move_number and session_position grow with index
    assert a.decisions[0].time_signal.move_number == 0
    assert a.decisions[3].time_signal.move_number == 3
    assert a.decisions[3].recent_outcomes.session_position == 3
    # recent_outcomes accumulates the student's own past (i priors at step i)
    assert len(a.decisions[3].recent_outcomes.recent) == 3
    # observations are the correct/incorrect labels
    assert a.observations[0].move == "correct"
    assert a.observations[2].move == "incorrect"


def test_load_kt_csv_respects_n_students(tmp_path):
    csv_path = tmp_path / "kt.csv"
    _write_csv(csv_path)
    ds = load_kt_csv(str(csv_path), n_students=1, min_responses=5)
    assert len(ds.trajectories) == 1


def test_load_kt_csv_skill_difficulty(tmp_path):
    csv_path = tmp_path / "kt.csv"
    _write_csv(csv_path)
    ds = load_kt_csv(
        str(csv_path), n_students=10, min_responses=5, train_frac=0.7
    )
    # A and B are evaluated (>= min_responses), so only their *training*
    # prefix (round(0.7*n), matching split_indices) feeds the difficulty
    # estimate -- their own held-out suffix is excluded. C is filtered out
    # (2 < 5) and so never evaluated; its full sequence is safe to use.
    #   skill 1: A[:4]=(1,1) + B[:4]=(0,1) + C=(1,1) -> 5/6 correct -> 0.167
    #   skill 2: A[:4]=(0,1) + B[:4]=(0,0)            -> 1/4 correct -> 0.75
    diffs = {
        round(d.state[0], 3) for t in ds.trajectories for d in t.decisions
    }
    assert 0.167 in diffs  # easy skill
    assert 0.75 in diffs  # hard skill


def test_load_kt_csv_difficulty_leakage_safe(tmp_path):
    """Flipping a student's own held-out (eval-suffix) responses must not
    change the difficulty feature fed into that student's *train-period*
    decisions -- regression test for the leakage this loader used to have
    (difficulty was fit over the whole file, including eval-suffix rows)."""

    def _rows(a_tail):
        rows = ["user_id\titem_id\ttimestamp\tcorrect\tskill_id"]
        # student A: 10 responses, skill 1 throughout. train_frac=0.7 ->
        # boundary=round(0.7*10)=7, so only the first 7 rows should ever
        # influence a difficulty estimate used by A's own decisions.
        a_seq = [1, 1, 0, 1, 0, 1, 1] + a_tail
        for i, correct in enumerate(a_seq):
            rows.append(f"A\t{100 + i}\t{i}\t{correct}\t1")
        # student B: filler so skill 1 has another contributor too.
        for i, correct in enumerate([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]):
            rows.append(f"B\t{200 + i}\t{i}\t{correct}\t1")
        return "\n".join(rows) + "\n"

    path_low = tmp_path / "low.csv"
    path_high = tmp_path / "high.csv"
    path_low.write_text(_rows([0, 0, 0]))  # held-out suffix: all wrong
    path_high.write_text(_rows([1, 1, 1]))  # held-out suffix: all correct

    ds_low = load_kt_csv(
        str(path_low), n_students=10, min_responses=5, train_frac=0.7
    )
    ds_high = load_kt_csv(
        str(path_high), n_students=10, min_responses=5, train_frac=0.7
    )
    a_low = next(t for t in ds_low.trajectories if t.player_id == "A")
    a_high = next(t for t in ds_high.trajectories if t.player_id == "A")

    # Train-period decision (index 0, inside the [0, 7) prefix): identical
    # difficulty feature regardless of A's own held-out future responses.
    assert a_low.decisions[0].state == a_high.decisions[0].state
