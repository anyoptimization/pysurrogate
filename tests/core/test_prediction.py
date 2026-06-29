"""Tests for the unified Prediction type: var canonical, mse/sigma aliases."""

import numpy as np

from pysurrogate.core import Prediction


def test_mse_aliases_var():
    var = np.array([[4.0], [9.0]])
    var_grad = np.array([[1.0, 2.0]])
    p = Prediction(y=np.zeros((2, 1)), var=var, grad=None, var_grad=var_grad)

    # mse / mse_grad are read-only aliases of var / var_grad (DACE-literature names)
    assert p.mse is p.var
    assert p.mse_grad is p.var_grad


def test_sigma_is_sqrt_of_var_clamped():
    p = Prediction(y=np.zeros((2, 1)), var=np.array([[4.0], [-1.0]]))
    # sqrt of the variance, with negatives clamped to 0 so std never goes NaN
    assert np.allclose(p.sigma, np.array([[2.0], [0.0]]))


def test_aliases_none_when_var_absent():
    p = Prediction(y=np.zeros((1, 1)))
    assert p.var is None
    assert p.mse is None
    assert p.sigma is None
    assert p.mse_grad is None
