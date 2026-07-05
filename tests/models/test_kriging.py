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


def test_ard_broadcasts_theta_even_with_no_bounds():
    # Kriging(ARD=True, theta_bounds=None) must fit one length-scale per input dimension -- an
    # unbounded ARD search -- and NOT silently collapse to a single isotropic length-scale. Frozen
    # (optimize=False) so no optimizer is needed: the broadcast alone proves ARD was honored.
    rng = np.random.RandomState(0)
    X = rng.random((30, 3))
    y = np.sin(3 * X[:, 0]) + 0.2 * X[:, 1]
    m = Kriging(ARD=True, theta_bounds=None)
    m.fit(X, y, optimize=False)
    assert np.ravel(m.model.model["theta"]).shape == (3,)  # per-dimension, not (1,)

    # the isotropic default still yields a single length-scale
    iso = Kriging(ARD=False, theta_bounds=None)
    iso.fit(X, y, optimize=False)
    assert np.ravel(iso.model.model["theta"]).shape == (1,)


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
