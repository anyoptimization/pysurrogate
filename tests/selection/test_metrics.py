"""Tests for the metrics registry: plain error, ranking, selection, and calibration."""

import numpy as np

from pysurrogate.selection import metrics


def test_plain_metrics_zero_on_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.calc_metric("mse", y, y) == 0.0
    assert metrics.calc_metric("rmse", y, y) == 0.0
    assert metrics.calc_metric("mae", y, y) == 0.0
    assert metrics.calc_metric("max_error", y, y) == 0.0
    assert metrics.calc_metric("r2", y, y) == 1.0


def test_direction_flags():
    assert metrics.greater_is_better("r2") is True
    assert metrics.greater_is_better("spear") is True
    assert metrics.greater_is_better("rmse") is False
    assert metrics.greater_is_better("nlpd") is False


def test_nlpd_rewards_honest_uncertainty():
    y = np.zeros(3)
    y_hat = np.ones(3)  # off by 1.0 everywhere
    confident = metrics.calc_metric("nlpd", y, y_hat, sigma=np.full(3, 0.1))
    honest = metrics.calc_metric("nlpd", y, y_hat, sigma=np.full(3, 1.0))
    assert confident > honest


def test_probabilistic_metric_requires_sigma():
    y, y_hat = np.zeros(3), np.ones(3)
    try:
        metrics.calc_metric("nlpd", y, y_hat)
    except ValueError:
        pass
    else:
        raise AssertionError("nlpd without sigma should raise")


def test_ranking_is_monotone_invariant():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(metrics.calc_metric("spear", y, y), 1.0)
    # a monotone transform preserves ranking even though the error is large
    assert np.isclose(metrics.calc_metric("spear", y, y**2 + 10), 1.0)


def test_evaluate_groups_by_family_and_gates_on_sigma():
    rng = np.random.RandomState(0)
    y = rng.random(20)
    y_hat = y + rng.normal(0, 0.05, 20)

    point_only = metrics.evaluate(y, y_hat)
    assert "accuracy" in point_only and "ranking" in point_only
    assert "calibration" not in point_only  # no sigma -> no calibration metrics

    with_sigma = metrics.evaluate(y, y_hat, sigma=np.full(20, 0.05))
    assert "calibration" in with_sigma and "nlpd" in with_sigma["calibration"]


def test_evaluate_explicit_names_drop_probabilistic_without_sigma():
    # explicit names use the SAME computability predicate as the default path: a probabilistic
    # metric without sigma is silently dropped, not raised on -- one consistent point of truth
    rng = np.random.RandomState(1)
    y = rng.random(15)
    y_hat = y + rng.normal(0, 0.05, 15)
    out = metrics.evaluate(y, y_hat, names=["rmse", "nlpd"])  # nlpd needs sigma, none given
    families = {m for fam in out.values() for m in fam}
    assert "rmse" in families
    assert "nlpd" not in families  # dropped, not an error
