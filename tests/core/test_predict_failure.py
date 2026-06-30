"""Regression tests for Model.predict on a model whose fit failed."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction


class _AlwaysFails(Model):
    """A model whose fit always raises, to exercise the failed-fit predict path."""

    def _fit(self, X, y, **kwargs):
        raise RuntimeError("boom")

    def _predict(self, X, var=False, grad=False):
        return Prediction(y=np.zeros((len(X), 1)))


def test_failed_fit_predict_returns_nan_when_prediction_not_raising():
    model = _AlwaysFails(raise_exception_while_fitting=False, raise_exception_while_prediction=False)
    model.fit(np.random.RandomState(0).random((10, 2)), np.random.RandomState(0).random((10, 1)))

    # a single 1-D query point is ONE row, not d rows
    pred = model.predict(np.array([0.3, 0.7]))

    assert pred.y.shape == (1, 1)
    assert np.all(np.isnan(pred.y))


def test_failed_fit_predict_honors_prediction_flag():
    model = _AlwaysFails(raise_exception_while_fitting=False, raise_exception_while_prediction=True)
    model.fit(np.random.RandomState(0).random((10, 2)), np.random.RandomState(0).random((10, 1)))

    try:
        model.predict(np.array([[0.3, 0.7]]))
    except Exception:
        return
    raise AssertionError("predict should have raised when raise_exception_while_prediction=True")
