"""Tests for the function-sampling study harness and the analytic test functions."""

import numpy as np

from pysurrogate.models import RBF, SimpleMean
from pysurrogate.selection import StudyResult, study
from pysurrogate.util.test_functions import TEST_FUNCTIONS, get_test_function


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
    try:
        get_test_function("not_a_function")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown test function should raise")


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
