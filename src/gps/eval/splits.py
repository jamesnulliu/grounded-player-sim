"""Temporal split helpers (proposal RQ3 / section Phase 1-2 'Split').

The decisive test that a dynamic model generalises forward rather than
memorising a stable habit is a *strict temporal* split per player: train on
earlier sessions, validate on middle, test on later. A random split leaks
stable habits and inflates results, so it is deliberately not offered here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class TemporalSplit:
    """Indices/items for a per-player chronological split."""

    train: list
    val: list
    test: list


def temporal_split(
    items: Sequence[T],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> TemporalSplit:
    """Split an already chronologically-ordered sequence by position.

    ``items`` MUST be ordered oldest -> newest (e.g. a player's sessions or
    games in time order). Test fraction is the remainder. Raises if the
    split would leave any partition empty, since an empty test set silently
    invalidates the RQ3 claim.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("fractions must be in (0,1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = list(items[:n_train])
    val = list(items[n_train : n_train + n_val])
    test = list(items[n_train + n_val :])

    if not (train and val and test):
        raise ValueError(
            f"temporal split left an empty partition for n={n}; "
            "need more games/sessions per player"
        )
    return TemporalSplit(train=train, val=val, test=test)
