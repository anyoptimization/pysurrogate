"""The RBF polynomial tail is the shared core regression basis (one implementation, not a copy)."""

import numpy as np
import pytest

from pysurrogate.core.regression import (
    ConstantRegression,
    LinearRegression,
    QuadraticRegression,
)
from pysurrogate.models import RBF
from pysurrogate.models.rbf import _tail_grad, rbf_kernel


@pytest.mark.parametrize(
    "tail,basis",
    [
        ("constant", ConstantRegression()),
        ("linear", LinearRegression()),
        ("quadratic", QuadraticRegression()),
        ("linear+quadratic", QuadraticRegression()),
    ],
)
def test_tail_columns_equal_basis(tail, basis):
    rng = np.random.RandomState(0)
    X = rng.random((6, 3))
    phi = rng.random((6, 6))
    appended = rbf_kernel(X, phi, tail=tail)[:, phi.shape[1] :]
    assert np.allclose(appended, basis(X))


def test_no_tail_appends_nothing():
    X = np.zeros((4, 2))
    phi = np.ones((4, 4))
    assert rbf_kernel(X, phi, tail=None).shape == (4, 4)


def test_tail_grad_matches_basis_grad():
    rng = np.random.RandomState(1)
    X = rng.random((5, 3))
    basis = QuadraticRegression()
    c = rng.random(basis(X).shape[1])
    assert np.allclose(_tail_grad(X, "quadratic", c), basis.grad(X) @ c)


def test_unknown_tail_raises():
    with pytest.raises(ValueError, match="Unknown tail"):
        rbf_kernel(np.zeros((2, 2)), np.zeros((2, 2)), tail="bogus")


def test_quadratic_tail_gradient_matches_finite_difference():
    # exercises the new basis.grad path end-to-end through a fitted RBF with a quadratic tail
    rng = np.random.RandomState(2)
    X = rng.random((40, 2))
    y = (X[:, [0]] ** 2 + X[:, [1]]).reshape(-1, 1)
    model = RBF(kernel="gaussian", tail="quadratic").fit(X, y)

    q = np.array([[0.4, 0.6]])
    g = model.predict(q, grad=True).grad
    eps = 1e-6
    fd = np.zeros((1, 2))
    for k in range(2):
        qp, qm = q.copy(), q.copy()
        qp[0, k] += eps
        qm[0, k] -= eps
        fd[0, k] = (model.predict(qp).y[0, 0] - model.predict(qm).y[0, 0]) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-3)
