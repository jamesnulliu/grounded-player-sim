"""Load a *real* knowledge-tracing cohort from a preprocessed CSV.

Turns the standard 5-column KT export -- ``user_id, item_id, timestamp,
correct, skill_id`` (tab-separated; the format shared by ASSISTments 2009/2017
and the KDD-Cup Cognitive-Tutor sets in
``theophilee/learner-performance-prediction``) -- into the same
:class:`~gps.train.base.TrajectoryDataset` the synthetic
:func:`~gps.experiments.kt.build_kt_dataset` produces, so the *identical* RQ5 /
Milestone-F pipeline runs on real students.

The item feature is each skill's **empirical difficulty**
(``1 - mean(correct)``), an IRT-style single-number stand-in. Each student
becomes one time-ordered
``Trajectory``; ``recent_outcomes`` / ``session_position`` are rebuilt from the
row order (the exports are already time-sorted per student), so the future/
temporal split is well defined.

Most KT exports carry no response time, so ``time_spent`` defaults to a
constant ``1.0`` (the correctness channel only). When a file *does* carry a
per-row response time (e.g. ASSISTments' ``ms_first_response``, appended as a
6th tab-separated column by ``scripts/prepare_kt_data.py``), pass
``response_time_col=5`` to feed it as the real timing target -- this is what
makes the RQ5 *when-not-what* asymmetry testable on real (non-chess,
non-synthetic) response times.

Difficulty is fit **leakage-safe**: for a student who will actually be
evaluated (survives ``min_responses``/``n_students``), only their
*training-prefix* rows -- the same ``[0, boundary)`` cut
``BoardNativeBackbone.split_indices`` applies downstream -- contribute to the
estimate. A student's own held-out future responses never leak into the
difficulty feature fed to their own held-out decisions. Rows from students
outside the evaluated cohort are never scored, so their full sequences are
safe to use in full.
"""

from __future__ import annotations

import csv
from collections import defaultdict

from gps.interface import (
    DecisionPoint,
    Game,
    Outcome,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import Observation
from gps.train.base import Trajectory, TrajectoryDataset


def load_kt_csv(
    path: str,
    *,
    n_students: int = 500,
    min_responses: int = 50,
    max_len: int = 200,
    train_frac: float = 0.7,
    delimiter: str = "\t",
    response_time_col: int | None = None,
    rt_clip: tuple[float, float] = (0.5, 300.0),
) -> TrajectoryDataset:
    """Read a preprocessed KT CSV into a :class:`TrajectoryDataset`.

    ``n_students`` caps the cohort (first eligible students in file order),
    ``min_responses`` filters students with too little history, and ``max_len``
    truncates very long students (keeps padding sane). ``train_frac`` must
    match the value passed to ``run_kt``/``split_indices`` downstream -- it is
    used only to keep the difficulty estimate leakage-safe (see module
    docstring), not to split the returned dataset itself.

    ``response_time_col``, if given, is the 0-indexed column holding a
    per-row response time in *milliseconds* (e.g. 5 for a 6th column appended
    after ``skill_id``); it is converted to seconds and clipped to
    ``rt_clip`` to guard against known artifacts in raw exports (negative
    values from clock skew, multi-hour idle gaps that aren't real
    "thinking time"). Rows with a non-numeric value in that column are
    dropped (a handful of ``NA``/blank entries in some exports).
    """
    rows: list[tuple[str, str, int, float | None]] = []
    with open(path) as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        next(reader)  # header
        for parts in reader:
            user_id, _item_id, _ts, correct, skill_id = parts[:5]
            rt = None
            if response_time_col is not None:
                try:
                    rt_ms = float(parts[response_time_col])
                except (ValueError, IndexError):
                    continue
                lo, hi = rt_clip
                rt = min(max(rt_ms / 1000.0, lo), hi)
            rows.append((user_id, skill_id, int(correct), rt))

    by_user: dict[str, list[tuple[str, int, float | None]]] = defaultdict(list)
    for user_id, skill, correct, rt in rows:
        by_user[user_id].append((skill, correct, rt))

    students = [u for u, seq in by_user.items() if len(seq) >= min_responses][
        :n_students
    ]
    evaluated = set(students)

    # Leakage-safe difficulty: mirrors BoardNativeBackbone.split_indices's
    # per-trajectory boundary exactly, so an evaluated student's held-out
    # rows never contribute. Non-evaluated students are never scored, so
    # their full (truncated-to-max_len) sequence is safe to use.
    by_skill: dict[str, list[int]] = defaultdict(list)
    for user_id, seq in by_user.items():
        seq = seq[:max_len]
        if user_id in evaluated:
            n = len(seq)
            boundary = (
                min(n - 1, max(1, round(train_frac * n))) if n >= 2 else n
            )
            seq = seq[:boundary]
        for skill, correct, _rt in seq:
            by_skill[skill].append(correct)
    difficulty = {
        skill: 1.0 - sum(vals) / len(vals) for skill, vals in by_skill.items()
    }
    # A skill with no training-prefix observations anywhere (rare, only
    # possible for very small cohorts) falls back to the neutral prior.
    default_difficulty = 0.5

    trajectories = []
    for user_id in students:
        seq = by_user[user_id][:max_len]
        decisions, observations, prior = [], [], []
        for i, (skill, correct, rt) in enumerate(seq):
            spent = rt if rt is not None else 1.0
            decisions.append(
                DecisionPoint(
                    game=Game.KNOWLEDGE_TRACING,
                    player_id=user_id,
                    state=(difficulty.get(skill, default_difficulty),),
                    legal_actions=("correct", "incorrect"),
                    engine_reference=None,
                    time_signal=TimeSignal(
                        time_remaining=None,
                        time_spent=spent,
                        move_number=i,
                    ),
                    recent_outcomes=OutcomeStream(
                        recent=list(prior), session_position=i
                    ),
                    context={"synthetic": False},
                )
            )
            observations.append(
                Observation(
                    move="correct" if correct else "incorrect",
                    time_spent=spent,
                )
            )
            prior.append(Outcome(won=bool(correct)))
        trajectories.append(Trajectory(user_id, decisions, observations))
    return TrajectoryDataset(trajectories=trajectories)
