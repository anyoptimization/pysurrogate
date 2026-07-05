"""Multi-output (q>1) coverage: shared-hyperparameter GP gradient + refit round-trip.

The per-output sum in the objective (``sum_j sigma2_j``) and its envelope-theorem
gradient term are where a multi-output bug would hide, so the analytic batched gradient
(``batch_obj_grad``) is checked against finite differences with a matrix ``Y``; ``refit``
is exercised end-to-end with matrix targets.
"""

import numpy as np

from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.fit import batch_obj_grad, fit
from pysurrogate.dace.regr import ConstantRegression
from pysurrogate.optimizer import LBFGS

GAUSS = Gaussian()
CONSTANT = ConstantRegression()


def _analytic_grad(nX, nY, theta):
    """Analytic theta-gradient of the objective at a single theta via the batched primitive."""
    return batch_obj_grad(nX, nY, CONSTANT, GAUSS, np.atleast_1d(theta)[None])[1][0]


def _standardized_multi(seed, d=2, q=2, n=20):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    cols = [np.sin(X[:, 0] * 3.0 + j) + np.cos(X[:, 1] * 2.0 - j) for j in range(q)]
    Y = np.column_stack(cols)
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    nY = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    return nX, nY


def _obj(nX, nY, theta):
    return fit(nX, nY, CONSTANT, GAUSS, theta)["f"]


def _fd_grad(nX, nY, theta, eps=1e-6):
    theta = np.atleast_1d(np.array(theta, dtype=float))
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps
        tm[k] -= eps
        g[k] = (_obj(nX, nY, tp) - _obj(nX, nY, tm)) / (2 * eps)
    return g


def test_objective_gradient_matches_fd_with_matrix_Y():
    # the analytic gradient sums tr/quad terms over a single shared factorization but
    # the objective sums sigma2 over all q outputs -- FD pins that the per-output sum
    # and the envelope-theorem cancellation are handled correctly for q>1.
    nX, nY = _standardized_multi(0, d=2, q=3)
    for theta in (np.array([0.6, 0.6]), np.array([1.2, 0.4]), np.array([3.0, 2.0])):
        analytic = _analytic_grad(nX, nY, theta)
        finite = _fd_grad(nX, nY, theta)
        assert analytic.shape == (2,)
        assert np.allclose(analytic, finite, rtol=1e-4, atol=1e-6), theta


def test_refit_roundtrip_with_matrix_Y():
    # refit on matrix targets must append the new points, keep the (m, q) prediction
    # shape, and land essentially on the cold fit of the combined data. It is not
    # bit-identical: the shared-theta multi-output likelihood is flatter, so Boxmin's
    # pattern search settles in a marginally different basin from a warm vs cold start
    # (a ~3e-4 prediction difference here) -- the destination matches to a loose tol.
    rng = np.random.default_rng(1)
    X0 = rng.random((16, 2))
    Xn = rng.random((8, 2))
    X_all = np.vstack([X0, Xn])

    def _Y(X):
        return np.column_stack([np.sum(np.sin(X * 3.0), axis=1), np.sum(np.cos(X * 2.0), axis=1)])

    def _model():
        return Dace(regr=CONSTANT, corr=GAUSS, theta=1.0, theta_bounds=(1e-4, 50.0))

    cold = _model()
    cold.fit(X_all, _Y(X_all))

    warm = _model()
    warm.fit(X0, _Y(X0))
    # opt into the cold fit's semantics (MLE over all data); the refit default validate=True
    # optimizes a different objective. Same configured optimizer as the cold fit.
    warm.refit(Xn, _Y(Xn), validate=False)

    assert warm.model["X"].shape[0] == 24
    xt = rng.random((10, 2))
    assert warm.predict(xt).y.shape == (10, 2)
    assert np.all(np.isfinite(warm.predict(xt).y))
    assert np.allclose(cold.predict(xt).y, warm.predict(xt).y, atol=5e-3)


def test_lbfgs_matrix_Y_end_to_end():
    # the analytic-gradient (generic) LBFGS path must drive a matrix-Y fit to a valid optimum:
    # at the converged interior theta the objective gradient is ~0.
    rng = np.random.default_rng(2)
    X = rng.random((22, 2))
    Y = np.column_stack([np.sum(np.sin(X * 3.0), axis=1), np.sum(np.cos(X * 2.0), axis=1)])

    model = Dace(
        regr=CONSTANT,
        corr=GAUSS,
        theta=np.array([1.0, 1.0]),
        theta_bounds=([1e-4, 1e-4], [50.0, 50.0]),
        optimizer=LBFGS(options={"gtol": 1e-8, "ftol": 1e-12}),
    )
    model.fit(X, Y)

    theta = np.ravel(model.model["theta"])
    grad = _analytic_grad(model.model["nX"], model.model["nY"], theta)
    interior = (theta > 1e-4 * 1.01) & (theta < 50.0 * 0.99)
    assert np.all(np.abs(grad[interior]) < 1e-3)
    assert model.predict(rng.random((5, 2))).y.shape == (5, 2)


def test_multioutput_variance_is_output_mean_not_sum():
    # the shared predictive variance AVERAGES the per-output sigma2 -- it must not SUM it (a sum
    # grows with the number of outputs and is a meaningless scale). With q identical output columns
    # the shared variance must equal the single-output variance, not q times it. theta is frozen
    # (optimizer=None) so the multi- and single-output fits are identical up to the output count.
    rng = np.random.default_rng(3)
    X = rng.random((18, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)
    Y = np.column_stack([y, y, y])  # 3 identical outputs
    theta = 0.7 * np.ones(2)

    multi = Dace(regr=CONSTANT, corr=GAUSS, theta=theta, optimizer=None)
    multi.fit(X, Y)
    single = Dace(regr=CONSTANT, corr=GAUSS, theta=theta, optimizer=None)
    single.fit(X, y)

    q = rng.random((5, 2))
    vm = multi.predict(q, var=True).var
    vs = single.predict(q, var=True).var
    assert vm.shape == (5, 1)  # one shared variance per point, not (5, 3)
    assert np.allclose(vm, vs)  # mean of identical outputs == single-output variance (a sum -> 3x)


def test_predict_grad_and_mse_grad_shapes_for_multioutput():
    # predict(grad=True) must work for q>1 (it used to crash on the sY broadcast): the
    # mean gradient is (m, q, d) -- one gradient per output -- and each output's gradient
    # matches a finite difference. The shared-correlation mse and its gradient stay (m, 1)
    # / (m, d). Single-output keeps the historical (m, d) mean-gradient shape.
    rng = np.random.default_rng(7)
    X = rng.random((16, 2))
    Y = np.column_stack([np.sum(np.sin(X * 3.0), axis=1), np.sum(np.cos(X * 2.0), axis=1)])
    model = Dace(
        regr=CONSTANT,
        corr=GAUSS,
        theta=0.5 * np.ones(2),
        theta_bounds=(0.05 * np.ones(2), 10.0 * np.ones(2)),
    )
    model.fit(X, Y)

    q = np.array([[0.4, 0.7]])
    p = model.predict(q, mse=True, grad=True)
    assert p.grad.shape == (1, 2, 2)  # (m, q, d)
    assert p.mse.shape == (1, 1)  # shared across outputs
    assert p.mse_grad.shape == (1, 2)  # (m, d)

    # each output's mean gradient matches central finite differences
    eps = 1e-6
    for j in range(2):
        fd = np.zeros(2)
        for k in range(2):
            qp, qm = q.copy(), q.copy()
            qp[0, k] += eps
            qm[0, k] -= eps
            fd[k] = (model.predict(qp).y[0, j] - model.predict(qm).y[0, j]) / (2 * eps)
        assert np.allclose(p.grad[0, j], fd, rtol=1e-4, atol=1e-6)

    # single-output keeps the (m, d) mean-gradient shape (no per-output axis)
    single = Dace(
        regr=CONSTANT,
        corr=GAUSS,
        theta=0.5 * np.ones(2),
        theta_bounds=(0.05 * np.ones(2), 10.0 * np.ones(2)),
    )
    single.fit(X, Y[:, 0])
    assert single.predict(q, grad=True).grad.shape == (1, 2)
