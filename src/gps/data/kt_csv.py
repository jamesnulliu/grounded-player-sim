"""Load a *real* knowledge-tracing cohort from a preprocessed CSV.

Turns the standard 5-column KT export -- ``user_id, item_id, timestamp,
correct, skill_id`` (tab-separated; the format shared by ASSISTments 2009/2017
and the KDD-Cup Cognitive-Tutor sets in
``theophilee/learner-performance-prediction``) -- into the same
:class:`~gps.train.base.TrajectoryDataset` the synthetic
:func:`~gps.experiments.kt.build_kt_dataset` produces, so the *identical* RQ5 /
Milestone-F pipeline runs on real students.

The item feature is each skill's **empirical difficulty**
(``1 - mean(correct)``), an IRT-style single-number stand-in (these exports
carry no response time, so this is the correctness channel only). Each student
becomes one time-ordered
``Trajectory``; ``recent_outcomes`` / ``session_position`` are rebuilt from the
row order (the exports are already time-sorted per student), so the future/
temporal split is well defined.
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
    delimiter: str = "\t",
) -> TrajectoryDataset:
    """Read a preprocessed KT CSV into a :class:`TrajectoryDataset`.

    ``n_students`` caps the cohort (first eligible students in file order),
    ``min_responses`` filters students with too little history, and ``max_len``
    truncates very long students (keeps padding sane). Skill difficulty is
    computed over the *whole* file before filtering.
    """
    rows: list[tuple[str, str, int]] = []
    with open(path) as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        next(reader)  # header
        for user_id, _item_id, _ts, correct, skill_id in reader:
            rows.append((user_id, skill_id, int(correct)))

    by_skill: dict[str, list[int]] = defaultdict(list)
    for _u, skill, correct in rows:
        by_skill[skill].append(correct)
    difficulty = {
        skill: 1.0 - sum(vals) / len(vals) for skill, vals in by_skill.items()
    }

    by_user: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for user_id, skill, correct in rows:
        by_user[user_id].append((skill, correct))

    students = [u for u, seq in by_user.items() if len(seq) >= min_responses][
        :n_students
    ]

    trajectories = []
    for user_id in students:
        seq = by_user[user_id][:max_len]
        decisions, observations, prior = [], [], []
        for i, (skill, correct) in enumerate(seq):
            decisions.append(
                DecisionPoint(
                    game=Game.KNOWLEDGE_TRACING,
                    player_id=user_id,
                    state=(difficulty[skill],),
                    legal_actions=("correct", "incorrect"),
                    engine_reference=None,
                    time_signal=TimeSignal(
                        time_remaining=None,
                        time_spent=1.0,
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
                    time_spent=1.0,
                )
            )
            prior.append(Outcome(won=bool(correct)))
        trajectories.append(Trajectory(user_id, decisions, observations))
    return TrajectoryDataset(trajectories=trajectories)
