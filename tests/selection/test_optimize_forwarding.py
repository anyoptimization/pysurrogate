"""Tests that the fit-time optimize flag is forwarded through the selection layer and RBF."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.models import RBF
from pysurrogate.selection import AutoModel, Benchmark


class _RecordingModel(Model):
    """A trivial model recording every ``optimize`` flag seen, via a shared class-level log.

    The log is a class attribute so it survives the ``copy.deepcopy`` the benchmark makes of each
    prototype (instance attributes would be copied per fold and lost).
    """

    fit_log: list = []

    def _fit(self, X, y, optimize=True, **kwargs):
        _RecordingModel.fit_log.append(optimize)
        self.model = float(np.mean(y))

    def _predict(self, X, var=False, grad=False):
        return Prediction(y=np.full((len(X), 1), self.model))


def _data(n=40):
    rng = np.random.RandomState(0)
    X = rng.random((n, 2))
    y = (X[:, [0]] + X[:, [1]]).reshape(-1, 1)
    return X, y


def test_model_selection_forwards_optimize_false():
    X, y = _data()
    _RecordingModel.fit_log = []
    sel = AutoModel({"rec": _RecordingModel()}, sorted_by="rmse")
    sel.fit(X, y, optimize=False)

    # every fit the benchmark folds + winner refit performed must have seen optimize=False
    assert _RecordingModel.fit_log
    assert all(flag is False for flag in _RecordingModel.fit_log)


def test_benchmark_forwards_optimize_false():
    X, y = _data()
    _RecordingModel.fit_log = []
    Benchmark({"rec": _RecordingModel()}, metrics=["rmse"]).do(X, y, optimize=False)
    assert _RecordingModel.fit_log and all(flag is False for flag in _RecordingModel.fit_log)


def test_rbf_honors_fit_time_optimize_false():
    X, y = _data(60)

    tuned = RBF(kernel="gaussian", optimize=True)
    tuned.fit(X, y, optimize=True)

    frozen = RBF(kernel="gaussian", optimize=True)
    frozen.fit(X, y, optimize=False)

    # optimize=False must skip the sigma grid and keep the constructor sigma (1.0)
    assert frozen.model["kwargs"]["sigma"] == 1.0
