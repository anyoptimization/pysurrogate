"""Tests for the Model base lifecycle: preprocessing, normalization, postprocess scaling."""

import numpy as np
import pytest

from pysurrogate.core import Model, Prediction, Standardization


class LinearBackend(Model):
    """Minimal backend: least-squares plane, used to exercise the lifecycle, not the math."""

    def _fit(self, X, y, **kwargs):
        A = np.hstack([X, np.ones((len(X), 1))])
        self.model, *_ = np.linalg.lstsq(A, y, rcond=None)

    def _predict(self, X, var=False, grad=False):
        A = np.hstack([X, np.ones((len(X), 1))])
        y = A @ self.model
        grad_val = np.tile(self.model[:-1].T, (len(X), 1)) if grad else None
        var_val = np.zeros((len(X), 1)) if var else None
        return Prediction(y=y, var=var_val, grad=grad_val)


def _data():
    rng = np.random.RandomState(0)
    X = rng.random((40, 2))
    y = (3.0 * X[:, [0]] - 2.0 * X[:, [1]] + 1.0) * 10.0
    return X, y


def test_fit_predict_recovers_plane():
    X, y = _data()
    model = LinearBackend().fit(X, y)
    pred = model.predict(X)
    assert np.allclose(pred.y, y, atol=1e-6)
    assert model.has_been_fitted and model.success


def test_eliminate_duplicates_drops_repeats():
    X, y = _data()
    Xd = np.vstack([X, X[:5]])
    yd = np.vstack([y, y[:5]])
    model = LinearBackend(eliminate_duplicates=True).fit(Xd, yd)
    # the 5 duplicate rows are removed before fitting
    assert len(model.X) == len(X)


def test_grad_unnormalized_through_standardization():
    X, y = _data()
    # with output standardization, the backend sees normalized y; postprocess must carry the
    # gradient back to original units via the affine scale -- so grad matches the raw-fit grad.
    raw = LinearBackend().fit(X, y).predict(X[:3], grad=True)
    std = LinearBackend(norm_y=Standardization()).fit(X, y).predict(X[:3], grad=True)
    assert np.allclose(raw.grad, std.grad, atol=1e-6)
    assert np.allclose(raw.y, std.y, atol=1e-6)


def test_unknown_constructor_kwarg_raises():
    # a typo'd keyword (e.g. norm_x= instead of norm_X=) was silently swallowed, hiding a
    # misconfigured model; unknown kwargs no subclass consumed must now raise.
    with pytest.raises(TypeError, match="unexpected keyword"):
        LinearBackend(norm_x=Standardization())  # note the lowercase typo


def test_predict_before_fit_and_after_failed_fit_have_distinct_messages():
    # never fitted -> "has not been fitted"; a fit that ran but failed -> "error while fitting".
    never = LinearBackend()
    with pytest.raises(RuntimeError, match="has not been fitted"):
        never.predict(np.array([[0.1, 0.2]]))

    class _Fails(Model):
        def _fit(self, X, y, **kwargs):
            raise RuntimeError("boom")

        def _predict(self, X, var=False, grad=False):
            return Prediction(y=np.zeros((len(X), 1)))

    failed = _Fails(raise_exception_while_fitting=False)
    failed.fit(*_data())
    assert not failed.has_been_fitted  # a failed fit does not count as fitted
    with pytest.raises(RuntimeError, match="error while fitting"):
        failed.predict(np.array([[0.1, 0.2]]))


def test_predict_promotes_1d_point_before_slicing_active_dims():
    # regression: a 1-D query with active_dims must be promoted to one row FIRST, then have its
    # columns sliced -- slicing a 1-D point would pick coordinates as rows and crash / mis-shape.
    X, y = _data()
    model = LinearBackend(active_dims=[0]).fit(X, y)
    pred = model.predict(np.array([0.3, 0.7]))  # a single 1-D point of width 2
    assert pred.y.shape == (1, 1) and np.all(np.isfinite(pred.y))
