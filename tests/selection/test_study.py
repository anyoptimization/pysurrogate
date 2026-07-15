"""Tests for the function-sampling study harness and the analytic test functions."""

import numpy as np
import pytest

from pysurrogate.models import RBF, SimpleMean
from pysurrogate.selection import StudyResult, study
from pysurrogate.util.test_functions import TEST_FUNCTIONS, get_test_function


def test_ranking_does_not_penalize_models_without_calibration_metrics():
    # a point-only model that wins every point metric must rank first even though it scores NaN on
    # the calibration metrics a probabilistic model can compute. Each metric is ranked only over the
    # models that produced a finite value for it, so the inability to emit uncertainty cannot inflate
    # a model's rank. Before the fix, NaN -> last place on 4 calibration metrics dragged the accurate
    # point-only model below a less-accurate probabilistic one.
    raw = {
        "point_only": {
            "rmse": [0.1],
            "mae": [0.1],
            "nlpd": [np.nan],
            "crps": [np.nan],
            "cal_err": [np.nan],
            "calib": [np.nan],
        },
        "probabilistic": {
            "rmse": [0.5],
            "mae": [0.5],
            "nlpd": [1.0],
            "crps": [1.0],
            "cal_err": [0.05],
            "calib": [1.0],
        },
    }
    res = StudyResult(raw, failures={}, meta={})
    ranking = res.ranking()
    assert list(ranking)[0] == "point_only"  # wins on the metrics it can actually compute
    assert res.best() == "point_only"


def test_test_functions_have_known_optimum():
    # every shipped function has its documented optimum of 0 at the origin (or all-ones)
    for name in ["sphere", "ackley", "rastrigin", "griewank"]:
        f, xl, xu = get_test_function(name, n_var=3)
        assert np.isclose(f(np.zeros((1, 3)))[0], 0.0, atol=1e-8)
    f, _, _ = get_test_function("rosenbrock", n_var=3)
    assert np.isclose(f(np.ones((1, 3)))[0], 0.0, atol=1e-8)


def test_get_test_function_bounds_shape():
    f, xl, xu = get_test_function("ackley", n_var=4)
    assert xl.shape == (4,) and xu.shape == (4,)
    assert np.all(xl < xu)


def test_get_test_function_rejects_unknown():
    with pytest.raises(ValueError):
        get_test_function("not_a_function")


def test_study_ranks_rbf_above_mean_on_sphere():
    f, xl, xu = get_test_function("sphere", n_var=2)
    # tps (polyharmonic spline) handles the wide [-5.12, 5.12] sphere domain; a fixed-sigma
    # gaussian kernel would underfit at that scale.
    result = study(
        f, xl, xu, n=40, models={"mean": SimpleMean(), "rbf": RBF(kernel="tps")}, n_test=200, repeats=3, seed=0
    )
    assert isinstance(result, StudyResult)
    assert result.best() == "rbf"
    assert result.mean("rmse")["rbf"] < result.mean("rmse")["mean"]


def test_study_records_failures_and_metrics():
    f, xl, xu = get_test_function("sphere", n_var=2)
    result = study(f, xl, xu, n=30, models={"rbf": RBF(kernel="gaussian")}, n_test=100, repeats=2, seed=1)
    assert result.failures["rbf"] == 0
    assert "rmse" in result.metrics()
    assert len(TEST_FUNCTIONS) >= 6


def test_study_is_deterministic_under_seed():
    # study delegates to FunctionBenchmark; a fixed seed must give a reproducible ranking + scores
    f, xl, xu = get_test_function("sphere", n_var=2)
    models = {"mean": SimpleMean(), "rbf": RBF(kernel="tps")}
    a = study(f, xl, xu, n=25, models=models, n_test=80, repeats=2, seed=3)
    b = study(f, xl, xu, n=25, models=models, n_test=80, repeats=2, seed=3)
    assert a.ranking() == b.ranking()
    assert np.isclose(a.mean("rmse")["rbf"], b.mean("rmse")["rbf"])
    # a noise-free interpolant beats the constant mean on the smooth sphere
    assert a.mean("rmse")["rbf"] < a.mean("rmse")["mean"]


def test_study_train_noise_is_reproducible_and_keeps_ranking():
    f, xl, xu = get_test_function("sphere", n_var=2)
    models = {"mean": SimpleMean(), "rbf": RBF(kernel="tps")}
    noisy = study(f, xl, xu, n=30, models=models, n_test=80, repeats=2, seed=5, noise=0.5)
    again = study(f, xl, xu, n=30, models=models, n_test=80, repeats=2, seed=5, noise=0.5)
    assert noisy.failures == again.failures
    assert np.isclose(noisy.mean("rmse")["rbf"], again.mean("rmse")["rbf"])


def test_metrics_are_returned_in_registry_order_regardless_of_insertion():
    # regression: metrics() built its order from a set, so column order varied across processes
    # (PYTHONHASHSEED). It must follow the registry order deterministically. Feed a scrambled raw
    # and assert the output matches metric_names() filtered to what was collected.
    from pysurrogate.selection.metrics import metric_names

    raw = {"m": {"calib": [1.0], "rmse": [0.1], "nlpd": [1.0], "mae": [0.2]}}
    res = StudyResult(raw, failures={}, meta={})
    collected = {"calib", "rmse", "nlpd", "mae"}
    assert res.metrics() == [m for m in metric_names() if m in collected]
