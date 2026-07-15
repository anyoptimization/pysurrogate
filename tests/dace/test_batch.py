"""Tests for the batched theta primitive (objective + analytic gradient + feasibility)."""

import numpy as np
import pytest

from pysurrogate.dace.corr import Gaussian, Matern
from pysurrogate.dace.fit import _cholesky_batch, batch_obj_grad, fit
from pysurrogate.dace.regr import ConstantRegression

GAUSS = Gaussian()
CONSTANT = ConstantRegression()


def test_noise_grad_requires_with_grad():
    # df/d(noise) needs the gradient block; requesting it with with_grad=False used to silently
    # return an all-zero dnoise. It must now raise rather than mislead the caller.
    nX, nY = _standardized(0, 2)
    with pytest.raises(ValueError, match="noise_grad requires with_grad"):
        batch_obj_grad(nX, nY, CONSTANT, GAUSS, np.array([[1.0, 1.0]]), noise=0.1, with_grad=False, noise_grad=True)


def _standardized(seed, d, n=18, q=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    cols = [np.sum(np.sin(X * (3.0 + i)), axis=1) for i in range(q)]
    y = np.column_stack(cols)
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    nY = (y - y.mean(0)) / y.std(0, ddof=1)
    return nX, nY


def _fd_grad(nX, nY, regr, kernel, theta, h=1e-6):
    """Central-difference gradient of the batched objective at a single theta (theta-space)."""
    theta = np.atleast_1d(np.asarray(theta, float))
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        fp = batch_obj_grad(nX, nY, regr, kernel, tp[None], with_grad=False)[0][0]
        fm = batch_obj_grad(nX, nY, regr, kernel, tm[None], with_grad=False)[0][0]
        g[k] = (fp - fm) / (2 * h)
    return g


# --- the keystone: batch_obj_grad's objective == fit()'s objective ---


def test_batch_obj_matches_fit_isotropic():
    nX, nY = _standardized(0, 3)
    thetas = np.array([[0.2], [0.7], [1.5], [5.0]])
    obj, _, feasible = batch_obj_grad(nX, nY, CONSTANT, GAUSS, thetas)

    assert feasible.all()
    for i, t in enumerate(thetas):
        assert np.isclose(obj[i], fit(nX, nY, CONSTANT, GAUSS, t)["obj"], rtol=1e-10, atol=1e-12)


def test_batch_obj_matches_fit_ard():
    nX, nY = _standardized(1, 3)
    thetas = np.array([[0.5, 1.3, 0.9], [2.0, 0.3, 4.0], [1.0, 1.0, 1.0]])
    obj, _, feasible = batch_obj_grad(nX, nY, CONSTANT, GAUSS, thetas)

    assert feasible.all()
    for i, t in enumerate(thetas):
        assert np.isclose(obj[i], fit(nX, nY, CONSTANT, GAUSS, t)["obj"], rtol=1e-10, atol=1e-12)


# --- the analytic batch gradient matches finite differences of the batch objective ---


def test_batch_grad_matches_fd_isotropic():
    nX, nY = _standardized(2, 3)
    thetas = np.array([[0.3], [1.1], [4.0]])
    _, grad, _ = batch_obj_grad(nX, nY, CONSTANT, GAUSS, thetas)

    for i, t in enumerate(thetas):
        assert np.allclose(grad[i], _fd_grad(nX, nY, CONSTANT, GAUSS, t), rtol=1e-5, atol=1e-7)


def test_batch_grad_matches_fd_ard():
    nX, nY = _standardized(3, 3)
    thetas = np.array([[0.5, 1.3, 0.9], [2.0, 0.3, 4.0]])
    _, grad, _ = batch_obj_grad(nX, nY, CONSTANT, GAUSS, thetas)

    for i, t in enumerate(thetas):
        assert np.allclose(grad[i], _fd_grad(nX, nY, CONSTANT, GAUSS, t), rtol=1e-5, atol=1e-7)


def test_batch_matches_for_a_product_kernel_and_multioutput():
    # exercise the default (looping) kernel.batch path + theta_grad, plus q > 1
    matern = Matern(nu=2.5)
    nX, nY = _standardized(4, 2, q=2)
    thetas = np.array([[0.4, 1.2], [1.0, 1.0], [3.0, 0.5]])
    obj, grad, feasible = batch_obj_grad(nX, nY, CONSTANT, matern, thetas)

    assert feasible.all()
    for i, t in enumerate(thetas):
        model = fit(nX, nY, CONSTANT, matern, t)
        assert np.isclose(obj[i], model["obj"], rtol=1e-10, atol=1e-12)
        assert np.allclose(grad[i], _fd_grad(nX, nY, CONSTANT, matern, t), rtol=1e-5, atol=1e-7)


# --- Gaussian.batch == the looping default ---


def test_gaussian_batch_matches_loop():
    rng = np.random.default_rng(5)
    D = rng.normal(size=(40, 3))
    for thetas in (np.array([[0.5], [2.0]]), np.array([[0.5, 1.0, 2.0], [3.0, 0.1, 0.7]])):
        got = GAUSS.batch(D, thetas)
        want = np.stack([GAUSS(D, t) for t in thetas])
        assert np.allclose(got, want, rtol=1e-12, atol=1e-14)


# --- feasibility: non-PD slices are masked, not raised ---


def test_cholesky_batch_masks_non_pd_slice():
    pd = np.eye(3)
    non_pd = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # indefinite
    feasible, C = _cholesky_batch(np.stack([pd, non_pd, pd]))

    assert feasible.tolist() == [True, False, True]
    assert np.allclose(C[0], np.eye(3))  # feasible factor is filled in
    assert np.allclose(C[2], np.eye(3))


def test_batch_obj_grad_feasible_thetas_never_raise():
    nX, nY = _standardized(6, 2)
    thetas = np.array([[0.1, 0.1], [1.0, 1.0], [50.0, 50.0]])
    obj, grad, feasible = batch_obj_grad(nX, nY, CONSTANT, GAUSS, thetas)

    assert feasible.all()
    assert np.all(np.isfinite(obj))
    assert np.all(np.isfinite(grad))
