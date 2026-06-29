"""Tests for the Model base lifecycle: preprocessing, normalization, postprocess scaling."""

import numpy as np

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
