"""Session segmentation from a wall-clock timestamp stream.

Lichess gives a timestamp stream, not session labels (design.md section 6).
We segment by the wall-clock gap between consecutive games, and we make the
gap threshold an **explicit, ablatable parameter** rather than a silent
default -- because "what counts as one sitting" is an unlabeled construct and
the result (post-loss tilt, late-session fatigue) can depend on it. Sweep it
as an ablation (TODO Milestone C).

Pure stdlib: this module never imports a chess/torch dependency, so it is
testable on any box.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Default gap (seconds) above which two consecutive games start a new
#: session. 30 minutes is a conventional starting point; sweep it.
DEFAULT_GAP_THRESHOLD_SECONDS = 1800.0


def segment_sessions(
    games: Sequence[tuple[float, float]],
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
) -> list[list[int]]:
    """Group game indices into sessions by inter-game idle time.

    Parameters
    ----------
    games:
        ``(start, end)`` epoch-second pairs, **chronologically ordered** (the
        caller sorts; we do not, so the returned indices line up with the
        input order). ``end`` is when a game finished; ``start`` when the next
        began. The idle gap is ``next.start - current.end``.
    gap_threshold_seconds:
        A gap of at least this many seconds starts a new session.

    Returns
    -------
    A list of sessions, each a list of indices into ``games`` (oldest first).
    Empty input yields an empty list.
    """
    if not games:
        return []
    sessions: list[list[int]] = [[0]]
    for i in range(1, len(games)):
        gap = games[i][0] - games[i - 1][1]
        if gap >= gap_threshold_seconds:
            sessions.append([i])
        else:
            sessions[-1].append(i)
    return sessions


def session_positions(
    games: Sequence[tuple[float, float]],
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
) -> list[int]:
    """Per-game 0-based position within its session.

    Convenience over :func:`segment_sessions` for filling
    :attr:`~gps.interface.OutcomeStream.session_position`: the first game of
    each sitting is ``0``, the second ``1``, and so on. ``len`` matches
    ``games``.
    """
    positions = [0] * len(games)
    for session in segment_sessions(games, gap_threshold_seconds):
        for pos, idx in enumerate(session):
            positions[idx] = pos
    return positions
