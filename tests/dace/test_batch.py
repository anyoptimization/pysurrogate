"""Tests for the batched theta primitive and the VectorizedAdam optimizer."""

import numpy as np

from pysurrogate.dace.corr import Gaussian, Matern
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.fit import _cholesky_batch, batch_obj_grad, fit
from pysurrogate.dace.optimizers import VectorizedAdam, objective_gradient
from pysurrogate.dace.regr import ConstantRegression

GAUSS = Gaussian()
CONSTANT = ConstantRegression()


def _standardized(seed, d, n=18, q=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    cols = [np.sum(np.sin(X * (3.0 + i)), axis=1) for i in range(q)]
    y = np.column_stack(cols)
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    nY = (y - y.mean(0)) / y.std(0, ddof=1)
    return nX, nY


# --- the keystone: batch_obj_grad == loop of fit + objective_gradient ---


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


def test_batch_grad_matches_objective_gradient_isotropic():
    nX, nY = _standardized(2, 3)
    thetas = np.array([[0.3], [1.1], [4.0]])
    _, grad, _ = batch_obj_grad(nX, nY, CONSTANT, GAUSS, thetas)

    for i, t in enumerate(thetas):
        model = fit(nX, nY, CONSTANT, GAUSS, t)
        ref = objective_gradient(nX, model, t, GAUSS.theta_grad)
        assert np.allclose(grad[i], ref, rtol=1e-8, atol=1e-10)


def test_batch_grad_matches_objective_gradient_ard():
    nX, nY = _standardized(3, 3)
    thetas = np.array([[0.5, 1.3, 0.9], [2.0, 0.3, 4.0]])
    _, grad, _ = batch_obj_grad(nX, nY, CONSTANT, GAUSS, thetas)

    for i, t in enumerate(thetas):
        model = fit(nX, nY, CONSTANT, GAUSS, t)
        ref = objective_gradient(nX, model, t, GAUSS.theta_grad)
        assert np.allclose(grad[i], ref, rtol=1e-8, atol=1e-10)


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
        ref = objective_gradient(nX, model, t, matern.theta_grad)
        assert np.allclose(grad[i], ref, rtol=1e-8, atol=1e-10)


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


# --- VectorizedAdam end to end ---


def test_vectorized_adam_improves_objective_and_predicts():
    rng = np.random.default_rng(2)
    X = rng.random((25, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    model = Dace(
        regr=CONSTANT,
        corr=GAUSS,
        theta=np.array([1.0, 1.0]),
        thetaL=[1e-4, 1e-4],
        thetaU=[50.0, 50.0],
        optimizer=VectorizedAdam(pop_size=6, steps=40),
    )
    model.fit(X, y)

    start_obj = fit(model.model["nX"], model.model["nY"], CONSTANT, GAUSS, np.array([1.0, 1.0]))["f"]
    assert model.model["f"] <= start_obj + 1e-9
    assert np.all(np.isfinite(model.predict(rng.random((5, 2))).y))
    assert model.optimization["pop_size"] == 6


def test_vectorized_adam_isotropic_theta():
    rng = np.random.default_rng(7)
    X = rng.random((22, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    model = Dace(regr=CONSTANT, corr=GAUSS, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=VectorizedAdam())
    model.fit(X, y)

    assert np.ravel(model.model["theta"]).shape == (1,)
    assert np.all(np.isfinite(model.predict(rng.random((4, 2))).y))


def test_vectorized_adam_escapes_a_bad_starting_basin():
    # from the lower-bound plateau a single descent stays stuck; a diverse population
    # must reach the good optimum (mirrors the LBFGS-restarts test).
    rng = np.random.default_rng(3)
    X = rng.random((30, 1))
    y = np.sin(X[:, 0] * 12.0)

    model = Dace(
        regr=CONSTANT,
        corr=GAUSS,
        theta=1e-4,
        thetaL=1e-4,
        thetaU=100.0,
        optimizer=VectorizedAdam(pop_size=12),
    )
    model.fit(X, y)

    assert model.model["f"] < 1e-2
    assert np.ravel(model.model["theta"])[0] > 1e-4  # moved off the plateau


def test_vectorized_adam_works_in_refit_with_validation():
    rng = np.random.default_rng(4)
    X = rng.random((20, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    model = Dace(
        regr=CONSTANT,
        corr=GAUSS,
        theta=np.array([1.0, 1.0]),
        thetaL=[1e-4, 1e-4],
        thetaU=[50.0, 50.0],
    )
    model.fit(X, y)

    Xn, yn = rng.random((4, 2)), np.sum(np.sin(rng.random((4, 2)) * 3.0), axis=1)
    model.refit(Xn, yn, optimizer=VectorizedAdam(pop_size=5, steps=20))

    assert model.model["X"].shape[0] == 24  # the new points were appended
    assert np.all(np.isfinite(model.predict(rng.random((5, 2))).y))
