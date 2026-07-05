"""Held-out theta selection (Dace.refit held-out search + ValidationSelection); Model.validate."""

import numpy as np
import pytest

from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.regr import ConstantRegression
from pysurrogate.models import Kriging
from pysurrogate.optimizer import LBFGS, Boxmin

_DEFAULT = object()  # local sentinel: "let Dace pick its default optimizer"


def _fun(X):
    return np.sum(np.sin(X * 2 * np.pi), axis=1)


def _model(optimizer=_DEFAULT):
    kw = {} if optimizer is _DEFAULT else {"optimizer": optimizer}
    return Dace(regr=ConstantRegression(), corr=Gaussian(), theta=1.0, theta_bounds=(1e-5, 100.0), **kw)


# --- Dace.refit: held-out theta selection (returns nothing; scoring lives on Model) -------------


def test_refit_validate_holds_out_new_points_and_appends():
    # refit(validate=True): the new points are the held-out set that steers the theta search,
    # AND are appended afterwards. The configured optimizer is restored. Dace.refit returns None.
    rng = np.random.default_rng(5)
    X0 = rng.random((15, 1))

    configured = Boxmin()
    m = _model(optimizer=configured)
    m.fit(X0, _fun(X0))

    Xn = rng.random((8, 1))
    assert m.refit(Xn, _fun(Xn), validate=True) is None  # engine refit reports no score

    assert m.optimizer is configured  # restored after the call
    assert m.model["X"].shape[0] == 23  # new points appended -> all rows in the final model
    theta = np.ravel(m.model["theta"])
    assert np.all((theta >= 1e-5) & (theta <= 100.0))
    assert np.all(np.isfinite(m.predict(np.linspace(0, 1, 30)[:, None]).y))


def test_refit_validate_false_selects_by_likelihood():
    rng = np.random.default_rng(7)
    X0 = rng.random((20, 1))
    m = _model(optimizer=LBFGS())
    m.fit(X0, _fun(X0))

    Xn = rng.random((6, 1))
    m.refit(Xn, _fun(Xn), validate=False)

    theta = np.ravel(m.model["theta"])
    assert m.model["X"].shape[0] == 26
    assert np.all((theta >= 1e-5) & (theta <= 100.0))
    assert np.all(np.isfinite(m.predict(X0).y))


def test_refit_validate_handles_matrix_Y():
    rng = np.random.default_rng(6)
    X0 = rng.random((20, 2))

    def _multi(X):
        return np.column_stack([np.sum(np.sin(X * 3.0), axis=1), np.sum(np.cos(X * 2.0), axis=1)])

    m = _model(optimizer=Boxmin())
    m.fit(X0, _multi(X0))
    m.refit(rng.random((6, 2)), _multi(rng.random((6, 2))), validate=True)

    pred = m.predict(X0).y
    assert pred.shape == (20, 2)
    assert np.all(np.isfinite(pred))


def test_fit_has_no_validation_or_append_kwargs():
    # the held-out mask lived on fit() and moved entirely to refit(); fit() must reject it.
    m = _model()
    X = np.random.default_rng(0).random((12, 1))
    with pytest.raises(TypeError):
        m.fit(X, _fun(X), validation=np.zeros(12, dtype=bool))
    with pytest.raises(TypeError):
        m.fit(X, _fun(X), append=False)


# --- Model.validate: the generic predict + metric scorer ---------------------------------------


def _rmse(a, b):
    a = a if a.ndim == 2 else a[:, None]
    b = b if b.ndim == 2 else b[:, None]
    return float(np.sqrt(np.mean(np.square(a - b))))


def test_model_validate_scores_points_against_current_model():
    # Model.validate(X, y) predicts X with the current fit and returns a multi-metric score dict.
    rng = np.random.RandomState(0)
    X = rng.random((30, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    m = Kriging().fit(X, y)

    Xv = rng.random((8, 2))
    yv = np.sin(3 * Xv[:, 0]) + Xv[:, 1] ** 2
    score = m.validate(Xv, yv)
    assert isinstance(score, dict)
    assert score["rmse"] == pytest.approx(_rmse(m.predict(Xv).y, yv[:, None]))
    assert np.isfinite(score["mae"]) and "r2" in score  # the whole registry, not one metric
    assert np.isfinite(score["nlpd"])  # calibration metric present (Kriging has sigma)
    # restrict to a subset when you want
    assert set(m.validate(Xv, yv, metrics=["rmse", "mae"])) == {"rmse", "mae"}
    # scoring mutates nothing
    assert m.X.shape[0] == 30


def test_model_validate_requires_a_fit():
    with pytest.raises(Exception, match="requires a fitted model"):
        Kriging().validate(np.zeros((3, 2)), np.zeros(3))


# --- Model.calibrate: the variance-scaling sibling of validate ---------------------------------


def _cal_data():
    rng = np.random.RandomState(0)
    X = rng.random((40, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    Xv = rng.random((15, 2))
    yv = np.sin(3 * Xv[:, 0]) + Xv[:, 1] ** 2
    return X, y, Xv, yv


def test_calibrate_apply_true_rescales_variance():
    X, y, Xv, yv = _cal_data()
    m = Kriging().fit(X, y)
    var_before = m.predict(Xv, var=True).var.copy()

    s = m.calibrate(Xv, yv, apply=True)
    var_after = m.predict(Xv, var=True).var
    assert s > 0
    assert np.allclose(var_after, var_before * s)  # variance scaled by exactly s
    # a second calibrate on the now-calibrated model is ~identity and leaves the scale put
    assert m.calibrate(Xv, yv, apply=True) == pytest.approx(1.0, rel=1e-6)


def test_calibrate_apply_false_returns_scale_without_changing_model():
    X, y, Xv, yv = _cal_data()
    m = Kriging().fit(X, y)
    var_before = m.predict(Xv, var=True).var.copy()

    s = m.calibrate(Xv, yv, apply=False)
    assert s > 0
    assert np.allclose(m.predict(Xv, var=True).var, var_before)  # unchanged


def test_calibrate_raises_without_variance():
    from pysurrogate.models import SVR

    X, y, Xv, yv = _cal_data()
    m = SVR().fit(X, y)  # SVR has no predictive variance
    with pytest.raises(Exception, match="no predictive variance"):
        m.calibrate(Xv, yv)


def test_fit_resets_calibration():
    X, y, Xv, yv = _cal_data()
    m = Kriging().fit(X, y)
    m.calibrate(Xv, yv, apply=True)
    assert m._calibration != 1.0
    m.fit(X, y)  # a fresh fit clears the calibration
    assert m._calibration == 1.0
