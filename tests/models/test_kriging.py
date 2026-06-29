"""Tests for the Kriging Model adapter over the Dace engine."""

import numpy as np

from pysurrogate.core import Prediction
from pysurrogate.dace import Gaussian, LinearRegression
from pysurrogate.models import Kriging


def _branin_like():
    rng = np.random.RandomState(1)
    X = rng.random((30, 2))
    y = np.sin(3 * X[:, [0]]) + (X[:, [1]] - 0.5) ** 2
    return X, y


def test_kriging_interpolates_training_points():
    X, y = _branin_like()
    model = Kriging(regr=LinearRegression(), corr=Gaussian()).fit(X, y)
    pred = model.predict(X)
    # an interpolating kriging model reproduces the training targets closely
    assert np.allclose(pred.y, y, atol=1e-3)


def test_kriging_returns_shared_prediction_with_var():
    X, y = _branin_like()
    model = Kriging(corr=Gaussian()).fit(X, y)
    pred = model.predict(X[:5], var=True)
    assert isinstance(pred, Prediction)
    assert pred.var is not None and pred.var.shape == (5, 1)
    assert pred.mse is pred.var  # alias intact through the lifecycle
    assert np.all(pred.var >= 0.0)


def test_kriging_eliminates_duplicates_by_default():
    X, y = _branin_like()
    Xd = np.vstack([X, X[:3]])
    yd = np.vstack([y, y[:3]])
    # duplicate points would make the correlation matrix singular; the adapter drops them
    model = Kriging(corr=Gaussian()).fit(Xd, yd)
    assert model.success and len(model.X) == len(X)
