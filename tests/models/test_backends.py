"""Tests for the model backends: fit/predict shape, accuracy, variance, and analytic gradients."""

import numpy as np
import pytest

from pysurrogate.core import Prediction
from pysurrogate.models import (
    KNN,
    RBF,
    SVR,
    InverseDistanceWeighting,
    PolynomialRegression,
    RandomForest,
    SimpleMean,
)


def _data(n=60, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.random((n, 2))
    y = (np.sin(3 * X[:, [0]]) + (X[:, [1]] - 0.5) ** 2).reshape(-1, 1)
    return X, y


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


ALL_BACKENDS = [
    lambda: RBF(kernel="cubic"),
    lambda: SVR(),
    lambda: KNN(n_nearest=8),
    lambda: InverseDistanceWeighting(),
    lambda: PolynomialRegression(degree=3),
    lambda: RandomForest(n_estimators=30),
]


@pytest.mark.parametrize("make", ALL_BACKENDS, ids=lambda m: m().__class__.__name__)
def test_fit_predict_shape_and_type(make):
    X, y = _data()
    model = make().fit(X, y)
    pred = model.predict(X[:7])
    assert isinstance(pred, Prediction)
    assert pred.y.shape == (7, 1)
    assert np.all(np.isfinite(pred.y))


@pytest.mark.parametrize("make", ALL_BACKENDS, ids=lambda m: m().__class__.__name__)
def test_beats_constant_mean_baseline(make):
    X, y = _data()
    Xte, yte = _data(n=40, seed=7)

    baseline = _rmse(SimpleMean().fit(X, y).predict(Xte).y, yte)
    model_err = _rmse(make().fit(X, y).predict(Xte).y, yte)
    # every real backend should beat predicting the global mean everywhere
    assert model_err < baseline


def test_simple_mean_is_constant():
    X, y = _data()
    pred = SimpleMean().fit(X, y).predict(_data(n=5, seed=3)[0])
    assert np.allclose(pred.y, y.mean())


@pytest.mark.parametrize("model_with_var", [KNN(n_nearest=8), RandomForest(n_estimators=30)], ids=["KNN", "Forest"])
def test_variance_is_returned_and_nonnegative(model_with_var):
    X, y = _data()
    pred = model_with_var.fit(X, y).predict(X[:5], var=True)
    assert pred.var is not None and pred.var.shape == (5, 1)
    assert np.all(pred.var >= 0.0)


@pytest.mark.parametrize(
    "make",
    # gaussian is well-conditioned, so the analytic gradient is FD-verifiable; the pure-power
    # kernels (cubic on squared distance) are too ill-conditioned for a reliable FD check.
    [lambda: RBF(kernel="gaussian"), lambda: InverseDistanceWeighting(p=3.0), lambda: PolynomialRegression(degree=3)],
    ids=["RBF", "IDW", "PolynomialRegression"],
)
def test_analytic_gradient_matches_finite_difference(make):
    X, y = _data()
    model = make().fit(X, y)

    q = np.array([[0.37, 0.62]])
    g = model.predict(q, grad=True).grad
    assert g.shape == (1, 2)

    eps = 1e-6
    fd = np.zeros((1, 2))
    for k in range(2):
        qp, qm = q.copy(), q.copy()
        qp[0, k] += eps
        qm[0, k] -= eps
        fd[0, k] = (model.predict(qp).y[0, 0] - model.predict(qm).y[0, 0]) / (2 * eps)

    assert np.allclose(g, fd, atol=1e-3)


@pytest.mark.parametrize(
    "make",
    [
        lambda: SVR(),
        lambda: InverseDistanceWeighting(),
        lambda: PolynomialRegression(degree=2),
        lambda: RandomForest(),
        lambda: RBF(),
    ],
    ids=["SVR", "IDW", "PolynomialRegression", "RandomForest", "RBF"],
)
def test_single_output_backends_reject_multi_output(make):
    # these backends fit one output; a multi-output y previously got silently truncated to y[:, 0].
    # A clear ValueError beats a silent wrong answer -- fit one model per output instead.
    rng = np.random.RandomState(0)
    X = rng.random((30, 2))
    Y = np.column_stack([np.sin(X.sum(1)), (X**2).sum(1)])  # two outputs
    with pytest.raises(ValueError, match="single output"):
        make().fit(X, Y)
