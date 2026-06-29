"""Tests for the core metrics: plain error, uncertainty-aware (NLPD/MSLL), and ranking."""

import numpy as np

from pysurrogate.core import metrics


def test_plain_metrics_zero_on_perfect_fit():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert metrics.mse(y, y) == 0.0
    assert metrics.rmse(y, y) == 0.0
    assert metrics.mae(y, y) == 0.0
    assert metrics.max_error(y, y) == 0.0
    assert metrics.r2(y, y) == 1.0


def test_rmse_is_sqrt_of_mse_and_shape_agnostic():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([[1.5], [2.0], [2.0]])  # (n, 1) vs (n,) must agree after ravel
    assert np.isclose(metrics.rmse(yt, yp), np.sqrt(metrics.mse(yt, yp)))
    assert np.isclose(metrics.mse(yt, yp), np.mean([0.25, 0.0, 1.0]))


def test_nlpd_rewards_honest_uncertainty():
    yt = np.array([0.0, 0.0, 0.0])
    yp = np.array([1.0, 1.0, 1.0])  # all off by 1.0
    confident = metrics.nlpd(yt, yp, var=np.full(3, 0.01))  # wrong AND sure
    honest = metrics.nlpd(yt, yp, var=np.full(3, 1.0))  # wrong but uncertain
    # being confidently wrong is penalized far more than being honestly uncertain
    assert confident > honest


def test_msll_negative_when_model_beats_baseline():
    rng = np.random.RandomState(0)
    y_train = rng.normal(0, 1, size=200)
    yt = rng.normal(0, 1, size=50)
    good = metrics.msll(yt, y_pred=yt, var=np.full(50, 1e-3), y_train=y_train)  # near-perfect mean
    assert good < 0.0


def test_ranking_metrics_perfect_and_monotonic():
    yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(metrics.spearman(yt, yt), 1.0)
    assert np.isclose(metrics.kendall_tau(yt, yt), 1.0)
    # a monotone-increasing transform preserves ranking (Spearman stays 1) even though MSE is large
    yp = yt**2 + 10.0
    assert np.isclose(metrics.spearman(yt, yp), 1.0)
    assert metrics.mse(yt, yp) > 0.0
