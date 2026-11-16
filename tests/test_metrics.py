"""Metric tests: numpy only, no simulator, no robot model needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import metrics  # noqa: E402


def test_perfect_robot_scores_zero():
    target = np.array([0.5, -0.2, 1.3])
    points = np.tile(target, (10, 1))
    assert metrics.position_accuracy(points, target) == pytest.approx(0.0, abs=1e-12)
    assert metrics.position_repeatability(points) == pytest.approx(0.0, abs=1e-12)


def test_accuracy_is_distance_of_the_mean():
    target = np.zeros(3)
    points = np.array([[0.003, 0.004, 0.0], [0.003, 0.004, 0.0]])
    assert metrics.position_accuracy(points, target) == pytest.approx(0.005)


def test_repeatability_matches_iso_formula_by_hand():
    # Four points at distance 1 and four at distance 3 from their common mean.
    pts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                    [3, 0, 0], [-3, 0, 0], [0, 3, 0], [0, -3, 0]], dtype=float)
    l = np.array([1, 1, 1, 1, 3, 3, 3, 3], dtype=float)
    expected = l.mean() + 3 * l.std(ddof=1)
    assert metrics.position_repeatability(pts) == pytest.approx(expected)


def test_accurate_and_repeatable_are_independent():
    rng = np.random.default_rng(0)
    target = np.zeros(3)
    tight_but_off = np.array([0.010, 0, 0]) + rng.normal(0, 1e-5, (30, 3))
    centred_but_loose = rng.normal(0, 3e-3, (3000, 3))
    assert metrics.position_accuracy(tight_but_off, target) > 100 * metrics.position_repeatability(tight_but_off)
    assert metrics.position_repeatability(centred_but_loose) > 10 * metrics.position_accuracy(centred_but_loose, target)


def test_repeatability_ignores_a_common_shift():
    rng = np.random.default_rng(1)
    pts = rng.normal(0, 1e-3, (20, 3))
    assert metrics.position_repeatability(pts + 5.0) == pytest.approx(metrics.position_repeatability(pts))


def test_multidirectional_variation_is_largest_gap():
    b = np.array([[0, 0, 0], [0.003, 0, 0], [0, 0.004, 0]], dtype=float)
    assert metrics.multidirectional_variation(b) == pytest.approx(0.005)
    assert metrics.multidirectional_variation(b[:1]) == 0.0


def test_repeatability_needs_two_points():
    with pytest.raises(ValueError):
        metrics.position_repeatability(np.zeros((1, 3)))


def test_success_rate_needs_both_tolerances():
    pos = np.array([1.0, 1.0, 9.0, 1.0])
    ori = np.array([0.5, 5.0, 0.5, 0.5])
    assert metrics.success_rate(pos, ori, 5.0, 2.0) == pytest.approx(0.5)


def test_rmse():
    assert metrics.rmse(np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))


def test_mean_error_is_not_accuracy():
    """AP cancels opposing errors, mean error does not. Two trials 5 mm either
    side of the target have AP = 0 but mean error = 5."""
    target = np.zeros(3)
    points = np.array([[0.005, 0, 0], [-0.005, 0, 0]])
    err = metrics.per_trial_error(points, target)
    assert metrics.position_accuracy(points, target) == pytest.approx(0.0, abs=1e-12)
    assert metrics.mean_error(err) == pytest.approx(0.005)


def test_mean_error_never_below_accuracy():
    rng = np.random.default_rng(3)
    target = np.zeros(3)
    for _ in range(20):
        points = rng.normal(0.004, 0.002, (12, 3))
        err = metrics.per_trial_error(points, target)
        assert metrics.mean_error(err) >= metrics.position_accuracy(points, target) - 1e-12


def test_spread_is_zero_on_a_sphere_but_scatter_is_not():
    """Error magnitude and spatial scatter are different things: six points on a
    sphere of radius 5 mm all have the same error, so spread is 0, yet they are
    spread across 10 mm of space."""
    target = np.zeros(3)
    r = 0.005
    points = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]])
    err = metrics.per_trial_error(points, target)
    assert metrics.error_spread(err) == pytest.approx(0.0, abs=1e-12)
    assert metrics.position_repeatability(points) > 0.004


def test_rmse_relates_to_mean_and_spread():
    """RMSE^2 = mean^2 + population variance."""
    rng = np.random.default_rng(4)
    err = np.abs(rng.normal(0.01, 0.003, 50))
    expected = np.sqrt(metrics.mean_error(err) ** 2 + err.var(ddof=0))
    assert metrics.rmse(err) == pytest.approx(expected)


def test_rmse_punishes_an_outlier_more_than_the_mean():
    clean = np.full(10, 2.0)
    with_outlier = np.concatenate([np.full(9, 2.0), [20.0]])
    mean_ratio = metrics.mean_error(with_outlier) / metrics.mean_error(clean)
    rmse_ratio = metrics.rmse(with_outlier) / metrics.rmse(clean)
    assert rmse_ratio > mean_ratio


def test_spread_needs_two_trials():
    with pytest.raises(ValueError):
        metrics.error_spread(np.array([1.0]))


def test_axis_std_by_hand():
    pts = np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [4.0, 0.0, 5.0]])
    assert metrics.axis_std(pts) == pytest.approx([2.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        metrics.axis_std(pts[:1])


def test_radius_percentile_contains_that_share_of_points():
    rng = np.random.default_rng(6)
    pts = rng.normal(0, 1e-3, (2000, 3))
    r95 = metrics.radius_percentile(pts, 95.0)
    inside = (metrics.radii(pts) <= r95).mean()
    assert inside == pytest.approx(0.95, abs=0.002)
    assert metrics.radius_percentile(pts, 100.0) == pytest.approx(metrics.radii(pts).max())


def test_percentiles_by_hand():
    out = metrics.percentiles(np.arange(1.0, 102.0), (50.0, 95.0))      # 1 .. 101
    assert out == pytest.approx({"p50": 51.0, "p95": 96.0})
