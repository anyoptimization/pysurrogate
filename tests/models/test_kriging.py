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


def test_eliminate_duplicates_can_be_overridden_without_typeerror():
    # regression: the backend passed eliminate_duplicates=True alongside **kwargs, so a user
    # override collided ("multiple values for keyword argument"). setdefault makes it overridable.
    assert Kriging(eliminate_duplicates=False).eliminate_duplicates is False
    assert Kriging().eliminate_duplicates is True  # default stays on


def test_refit_resets_variance_calibration():
    # regression: the warm-started Dace-backed refit bypasses fit(), so it used to keep a stale
    # calibration scale. refit must reset it to the identity, like the base (fit-rebuilding) refit.
    X, y = _branin_like()
    model = Kriging().fit(X[:20], y[:20])
    model.calibrate(X[20:], y[20:])
    assert model._calibration != 1.0  # calibration took effect
    model.refit(X[20:25], y[20:25])
    assert model._calibration == 1.0  # absorbing points reset the stale scale


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


def test_kriging_respects_active_dims_on_fit_refit_and_predict():
    # regression (DaceBackedModel._refit): the target uses only inputs 0 and 2. With active_dims the
    # engine trains on the 2 selected columns; refit previously bypassed preprocess and handed the
    # engine full-width raw inputs, crashing on the active-dims mismatch. It must now fit/refit/predict.
    rng = np.random.RandomState(3)
    X = rng.uniform(-1, 1, size=(40, 4))
    y = np.sin(3 * X[:, [0]]) + X[:, [2]] ** 2
    model = Kriging(corr=Gaussian(), active_dims=[0, 2]).fit(X, y, optimize=False)
    assert model.X.shape[1] == 2  # engine saw only the 2 active dims
    Xnew = rng.uniform(-1, 1, size=(8, 4))
    ynew = np.sin(3 * Xnew[:, [0]]) + Xnew[:, [2]] ** 2
    model.refit(Xnew, ynew, optimize=False)  # must not raise despite full-width raw inputs
    assert np.all(np.isfinite(model.predict(X[:5]).y))
