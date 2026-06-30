"""Behavior tests for held-out theta selection (Dace.fit(validation=mask, append=...))."""

import numpy as np
import pytest

from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.regr import ConstantRegression
from pysurrogate.optimizer import LBFGS, Boxmin

_DEFAULT = object()  # local sentinel: "let Dace pick its default optimizer"


def _fun(X):
    return np.sum(np.sin(X * 2 * np.pi), axis=1)


def _model(optimizer=_DEFAULT):
    kw = {} if optimizer is _DEFAULT else {"optimizer": optimizer}
    return Dace(
        regr=ConstantRegression(),
        corr=Gaussian(),
        theta=1.0,
        theta_bounds=(1e-5, 100.0),
        **kw,
    )


def _rmse(a, b):
    a = a if a.ndim == 2 else a[:, None]
    b = b if b.ndim == 2 else b[:, None]
    return float(np.sqrt(np.mean(np.square(a - b))))


def _norm(m, X, y):
    # standardize raw rows with a fitted model's TRAINING stats (what selection uses)
    nX = (X - m.model["mX"]) / m.model["sX"]
    y = y[:, None] if y.ndim == 1 else y
    nY = (y - m.model["mY"]) / m.model["sY"]
    return nX, nY


def test_validation_none_matches_default():
    # the opt-out path must be byte-for-byte the historical MLE behavior: no mask
    # (or validation=None) selects theta by likelihood.
    rng = np.random.default_rng(0)
    X = rng.random((20, 1))
    y = _fun(X)

    default = _model()
    default.fit(X, y)
    explicit = _model()
    explicit.fit(X, y, validation=None)

    assert np.allclose(default.model["theta"], explicit.model["theta"])
    xt = np.linspace(0, 1, 50)[:, None]
    assert np.allclose(default.predict(xt).y, explicit.predict(xt).y)


def test_val_error_is_normalized_space_rmse():
    # _val_error scores in normalized Y space: it must equal predict()'s original-space
    # error divided by the training sY (per output), with nothing destandardized.
    rng = np.random.default_rng(1)
    X = rng.random((18, 1))
    y = _fun(X)
    Xv = rng.random((9, 1))
    yv = _fun(Xv)

    m = _model()
    m.fit(X, y)

    nXv, nYv = _norm(m, Xv, yv)
    internal = m._val_error(m.model, nXv, nYv)
    # normalized-space RMSE == original-space predict RMSE divided by the training sY
    manual = _rmse(m.predict(Xv).y, yv) / m.model["sY"].item()
    assert internal == pytest.approx(manual, rel=1e-9, abs=1e-12)


def test_validation_mask_predicts_and_stays_in_bounds():
    # a fit with a held-out mask still yields a valid model: theta in bounds, finite
    # predictions, and (append=True default) trained on every row.
    rng = np.random.default_rng(2)
    X = rng.random((24, 1))
    y = _fun(X)
    mask = np.zeros(24, dtype=bool)
    mask[::3] = True  # ~1/3 held out

    m = _model()
    m.fit(X, y, validation=mask)

    theta = np.ravel(m.model["theta"])
    assert np.all((theta >= 1e-5) & (theta <= 100.0))
    assert m.model["X"].shape[0] == 24  # append=True -> all rows in the final model
    assert np.all(np.isfinite(m.predict(np.linspace(0, 1, 30)[:, None]).y))


def test_append_false_keeps_model_on_training_rows_only():
    # append=False must leave the final model fit on the 0-rows only; the held-out
    # rows never enter the stored training data (and so never enter predict).
    rng = np.random.default_rng(3)
    X = rng.random((24, 1))
    y = _fun(X)
    mask = np.zeros(24, dtype=bool)
    mask[:8] = True  # 8 held out, 16 train

    m = _model()
    m.fit(X, y, validation=mask, append=False)

    assert m.model["X"].shape[0] == 16
    assert m.model["nX"].shape[0] == 16
    assert np.all(np.isfinite(m.predict(X).y))


def test_generic_validation_selects_and_stays_in_bounds():
    # the generic layer's held-out selection (ValidationSelection callback) yields a valid
    # train-only model: theta in bounds, finite predictions. The deeper "picks the lowest
    # held-out error" contract is covered in test_problem.py.
    rng = np.random.default_rng(7)
    X = rng.random((25, 1))
    y = _fun(X) + 0.08 * rng.standard_normal(25)
    mask = np.zeros(25, dtype=bool)
    mask[::5] = True

    m = _model(optimizer=LBFGS())
    m.fit(X, y, validation=mask, append=False)

    theta = np.ravel(m.model["theta"])
    assert np.all((theta >= 1e-5) & (theta <= 100.0))
    assert np.all(np.isfinite(m.predict(X).y))


def test_refit_validation_uses_new_points_and_appends():
    # refit(validation=True): the new points steer the theta search (they are the held-out set)
    # AND are appended afterwards. The model's configured optimizer is restored after the call.
    rng = np.random.default_rng(5)
    X0 = rng.random((15, 1))

    configured = Boxmin()
    m = _model(optimizer=configured)
    m.fit(X0, _fun(X0))

    Xn = rng.random((8, 1))
    m.refit(Xn, _fun(Xn), validation=True)  # warm, uses the configured optimizer

    assert m.optimizer is configured  # restored after the call
    assert m.model["X"].shape[0] == 23  # new points appended
    assert np.all(np.isfinite(m.predict(np.linspace(0, 1, 30)[:, None]).y))


def test_degenerate_mask_raises_clearly():
    # a mask that holds out nothing (all 0) or everything (all 1) is a user error -- it
    # must raise a clear message, not fail deep in the search on an empty array.
    rng = np.random.default_rng(8)
    X = rng.random((12, 1))
    y = _fun(X)
    m = _model()
    for bad in (np.zeros(12, dtype=bool), np.ones(12, dtype=bool)):
        with pytest.raises(Exception, match="hold out some rows but not all"):
            m.fit(X, y, validation=bad)
    # wrong length is also rejected
    with pytest.raises(Exception, match="one entry per row"):
        m.fit(X, y, validation=np.zeros(5, dtype=bool))


def test_validation_mask_handles_matrix_Y():
    # multi-output: Y is (n, q). Selection scores in normalized space across all
    # outputs, and the mask still yields a valid multi-output model.
    rng = np.random.default_rng(6)
    X = rng.random((24, 2))
    Y = np.column_stack([np.sum(np.sin(X * 3.0), axis=1), np.sum(np.cos(X * 2.0), axis=1)])
    mask = np.zeros(24, dtype=bool)
    mask[::4] = True

    m = _model(optimizer=Boxmin())
    m.fit(X, Y, validation=mask)

    pred = m.predict(X).y
    assert pred.shape == (24, 2)
    assert np.all(np.isfinite(pred))
