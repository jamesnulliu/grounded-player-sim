"""Tests for the over-players bootstrap CI (Milestone-A significance tool)."""

import pytest

from gps.eval.bootstrap import bootstrap_ci


def test_all_negative_excludes_zero():
    # A clean signed effect: every unit negative -> CI entirely below 0.
    ci = bootstrap_ci([-0.5, -0.4, -0.6, -0.55, -0.45], seed=0)
    assert ci.point < 0
    assert ci.high < 0
    assert ci.excludes_zero
    assert ci.p_below_zero == 1.0
    assert ci.n_units == 5


def test_centered_does_not_exclude_zero():
    # Symmetric around 0 -> CI straddles 0, not significant.
    ci = bootstrap_ci([-1.0, -0.5, 0.0, 0.5, 1.0], seed=0)
    assert abs(ci.point) < 1e-9
    assert ci.low < 0 < ci.high
    assert not ci.excludes_zero
    assert 0.2 < ci.p_below_zero < 0.8


def test_point_is_sample_mean_and_bounds_order():
    vals = [0.1, -0.2, 0.3, -0.4, 0.05, -0.15]
    ci = bootstrap_ci(vals, seed=3)
    assert ci.point == pytest.approx(sum(vals) / len(vals))
    assert ci.low <= ci.point <= ci.high


def test_reproducible_given_seed():
    a = bootstrap_ci([-0.1, -0.2, 0.05, -0.3], seed=7)
    b = bootstrap_ci([-0.1, -0.2, 0.05, -0.3], seed=7)
    assert (a.low, a.high, a.p_below_zero) == (b.low, b.high, b.p_below_zero)


def test_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])
