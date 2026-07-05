"""Behavior tests for Dace.refit — append new points and re-fit, warm-started."""

import numpy as np
import pytest

from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.regr import ConstantRegression
from pysurrogate.optimizer import LBFGS, Boxmin

_DEFAULT = object()  # local sentinel: "let Dace pick its default optimizer"


def _fun(X):
    return np.sum(np.sin(X * 2 * np.pi), axis=1)


def _model(theta=1.0, theta_bounds=(1e-5, 100.0), optimizer=_DEFAULT):
    kw = {} if optimizer is _DEFAULT else {"optimizer": optimizer}
    return Dace(
        regr=ConstantRegression(),
        corr=Gaussian(),
        theta=theta,
        theta_bounds=theta_bounds,
        **kw,
    )


def test_refit_before_fit_raises():
    rng = np.random.default_rng(0)
    X = rng.random((5, 1))
    with pytest.raises(Exception, match="requires a prior fit"):
        _model().refit(X, _fun(X))


def test_refit_appends_and_matches_cold_fit_on_combined_data():
    # refit takes only the new points; an MLE refit (validate=False) with the same
    # optimizer must equal a cold fit on the full combined set -- warm starting changes
    # the path, not the destination. (The refit *defaults* differ -- warm LBFGS +
    # validate=True -- so this round-trip opts back into the cold fit's semantics.)
    rng = np.random.default_rng(0)
    X0 = rng.random((15, 1))
    X_new = rng.random((10, 1))
    X_all = np.vstack([X0, X_new])

    cold = _model()
    cold.fit(X_all, _fun(X_all))

    warm = _model()
    warm.fit(X0, _fun(X0))
    warm.refit(X_new, _fun(X_new), validate=False)  # only the additions, same configured optimizer

    x_test = np.linspace(0, 1, 50)[:, None]
    assert np.allclose(cold.predict(x_test).y, warm.predict(x_test).y, atol=1e-5)


def test_refit_grows_the_stored_training_set():
    rng = np.random.default_rng(3)
    X0 = rng.random((12, 1))
    model = _model()
    model.fit(X0, _fun(X0))
    assert model.model["X"].shape[0] == 12

    X_new = rng.random((7, 1))
    model.refit(X_new, _fun(X_new))
    assert model.model["X"].shape[0] == 19
    assert model.model["nX"].shape[0] == 19  # the fit actually used all points


def test_refit_warm_starts_from_previous_theta():
    # the search seeds from self.theta; after fit() that should be the previous
    # optimized theta, not the original initial guess of 1.0.
    rng = np.random.default_rng(1)
    X0 = rng.random((12, 1))
    model = _model(theta=1.0)
    model.fit(X0, _fun(X0))
    optimized = model.model["theta"].copy()

    X_new = rng.random((6, 1))
    model.refit(X_new, _fun(X_new))
    assert np.allclose(model.theta, optimized)
    assert not np.allclose(model.theta, 1.0)


def test_refit_optimizer_none_freezes_theta_but_uses_new_data():
    # optimizer=None freezes theta (the replacement for the old Fixed()): theta must not
    # move, yet the new points are still incorporated (prediction changes because the data
    # grew, not because theta did).
    rng = np.random.default_rng(7)
    X0 = rng.random((12, 1))
    model = _model()
    model.fit(X0, _fun(X0))
    theta_before = model.model["theta"].copy()

    x_test = np.linspace(0, 1, 30)[:, None]
    pred_before = model.predict(x_test).y

    X_new = rng.random((6, 1))
    model.refit(X_new, _fun(X_new), optimize=False)

    assert np.allclose(model.model["theta"], theta_before)  # theta frozen
    assert model.model["X"].shape[0] == 18  # data still appended
    assert not np.allclose(pred_before, model.predict(x_test).y)  # fit changed


def test_refit_restores_configured_optimizer():
    # optimize=False frees theta for the duration of the call; the model's configured optimizer
    # must be restored afterward (it is not mutated).
    rng = np.random.default_rng(8)
    X0 = rng.random((10, 1))
    configured = Boxmin()
    model = _model(optimizer=configured)
    model.fit(X0, _fun(X0))

    model.refit(rng.random((4, 1)), _fun(rng.random((4, 1))), optimize=False)
    assert model.optimizer is configured  # restored after the call


def test_refit_warm_with_configured_optimizer_produces_valid_fit():
    # a warm refit uses the model's configured optimizer (here LBFGS): it appends the data,
    # keeps theta within bounds, and yields finite predictions.
    rng = np.random.default_rng(5)
    X0 = rng.random((15, 1))
    X_new = rng.random((8, 1))

    model = _model(optimizer=LBFGS())
    model.fit(X0, _fun(X0))
    model.refit(X_new, _fun(X_new))  # warm, uses the configured LBFGS

    assert model.model["X"].shape[0] == 23
    theta = np.ravel(model.model["theta"])
    assert np.all((theta >= 1e-5) & (theta <= 100.0))
    assert np.all(np.isfinite(model.predict(np.linspace(0, 1, 40)[:, None]).y))


def test_lbfgs_as_configured_optimizer():
    # the generic LBFGS works as the model-level configured optimizer too.
    rng = np.random.default_rng(6)
    X0 = rng.random((12, 1))
    model = _model(optimizer=LBFGS())
    model.fit(X0, _fun(X0))
    # the generic layer records its search on model.optimization (theta/noise/f/n_evals)
    assert model.optimization is not None
    assert model.optimization["n_evals"] >= 1
    pred = model.predict(np.linspace(0, 1, 10)[:, None]).y
    assert np.all(np.isfinite(pred))


def test_refit_without_optimization_reuses_theta():
    # optimizer=None -> no hyperparameter search; refit keeps the last theta frozen.
    rng = np.random.default_rng(2)
    X0 = rng.random((10, 1))
    model = Dace(regr=ConstantRegression(), corr=Gaussian(), theta=2.0, optimizer=None)
    model.fit(X0, _fun(X0))

    X_new = rng.random((5, 1))
    model.refit(X_new, _fun(X_new), optimize=False)  # freeze on refit too
    assert np.allclose(model.model["theta"], 2.0)
