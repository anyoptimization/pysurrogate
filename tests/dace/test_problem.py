"""Tests for DaceProblem: the DACE likelihood as a generic Problem, noise folded into the vector."""

import numpy as np

from pysurrogate.core.optimizer import Callback
from pysurrogate.core.sampling import LHS, Sampling
from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.fit import batch_obj_grad, fit
from pysurrogate.dace.problem import DaceProblem
from pysurrogate.dace.regr import ConstantRegression
from pysurrogate.optimizer import LBFGS, PatternSearch

GAUSS = Gaussian()
CONST = ConstantRegression()


def _data(n=18, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n, d))
    y = np.sum(np.sin(X), axis=1, keepdims=True)
    # standardize, as the optimizer layer expects
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    y = (y - y.mean(0)) / y.std(0, ddof=1)
    return X, y


def _fd_grad(problem, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (float(problem(xp[None]).f[0]) - float(problem(xm[None]).f[0])) / (2 * eps)
    return g


# --- gradient correctness (the keystone) -------------------------------------------------


def test_noise_gradient_matches_finite_difference():
    # df/d(noise) from batch_obj_grad's Rk=I term vs a finite difference on the objective
    X, y = _data()
    thetas = np.array([[1.3, 0.7]])
    nz = 1e-3
    _, _, _, dnoise = batch_obj_grad(X, y, CONST, GAUSS, thetas, noise=nz, noise_grad=True)

    def obj(noise):
        o, _, _ = batch_obj_grad(X, y, CONST, GAUSS, thetas, noise=noise)
        return float(o[0])

    fd = (obj(nz + 1e-7) - obj(nz - 1e-7)) / 2e-7
    assert np.isclose(dnoise[0], fd, rtol=1e-4, atol=1e-6)


def test_problem_theta_gradient_matches_fd():
    X, y = _data()
    prob = DaceProblem(X, y, CONST, GAUSS, theta_bounds=(np.full(2, 0.01), np.full(2, 20.0)))
    x = np.log10(np.array([1.5, 0.6]))
    assert np.allclose(prob(x[None]).grad[0], _fd_grad(prob, x), rtol=1e-4, atol=1e-5)


def test_problem_joint_theta_noise_gradient_matches_fd():
    X, y = _data()
    prob = DaceProblem(X, y, CONST, GAUSS, theta_bounds=(np.full(2, 0.01), np.full(2, 20.0)), noise_bounds=(1e-6, 1e-1))
    x = np.log10(np.array([1.2, 0.8, 1e-3]))  # [logθ1, logθ2, log noise]
    assert prob.n_var == 3
    assert np.allclose(prob(x[None]).grad[0], _fd_grad(prob, x), rtol=1e-4, atol=1e-5)


# --- fitting through the generic optimizers ----------------------------------------------


def test_lbfgs_fits_theta_only():
    X, y = _data()
    prob = DaceProblem(X, y, CONST, GAUSS, theta_bounds=(0.01, 20.0))  # isotropic -> p = 1
    res = LBFGS(sampling=Sampling(3)).minimize(prob, x0=np.zeros(1))
    # the generic optimum matches a direct fit at the same theta AND nugget
    theta, noise = prob.decode(res.x)
    direct = fit(X, y, CONST, GAUSS, theta, noise=noise)
    assert res.f <= direct["f"] + 1e-9
    assert np.isfinite(res.f)


def test_sampling_spreads_the_noise_coordinate_not_pinned_at_zero():
    # when noise is a learned coordinate, Sampling draws it across noise_bounds in log space --
    # starts are NOT all noise=0; each gets its own nugget (and never exactly 0).
    X, y = _data()
    prob = DaceProblem(X, y, CONST, GAUSS, theta_bounds=(np.full(2, 0.01), np.full(2, 20.0)), noise_bounds=(1e-6, 1e-1))
    starts = Sampling(20, LHS()).sample(prob.bounds, np.random.default_rng(0))
    noises = 10.0 ** starts[:, prob.p]  # the last coordinate is log10(noise)
    assert (noises > 0).all()  # never literally zero (lower bound is floored away from 0)
    assert noises.min() < 1e-3 < noises.max()  # genuinely spread across the range


def test_noise_is_just_another_coordinate():
    # learning the nugget only changes the vector length; same optimizer, no special path
    X, y = _data()
    prob = DaceProblem(X, y, CONST, GAUSS, theta_bounds=(np.full(2, 0.01), np.full(2, 20.0)), noise_bounds=(1e-6, 1e-1))
    res = LBFGS(sampling=Sampling(4)).minimize(prob)
    theta, noise = prob.decode(res.x)
    assert theta.shape == (2,)
    assert 1e-6 <= noise <= 1e-1
    assert np.isfinite(res.f)


def test_pattern_search_fits_without_gradient_path():
    X, y = _data()
    prob = DaceProblem(X, y, CONST, GAUSS, theta_bounds=(0.01, 20.0))  # isotropic -> p = 1
    res = PatternSearch(tol=1e-5).minimize(prob, x0=np.zeros(1))
    assert np.isfinite(res.f)
    # pattern search (no gradient) lands near L-BFGS (gradient) on the same likelihood
    ref = LBFGS(sampling=Sampling(3)).minimize(prob, x0=np.zeros(1))
    assert res.f <= ref.f + 0.05


def test_never_raises_on_degenerate_theta():
    # a tiny length-scale makes R near-singular; the problem must report infeasible, not raise
    X, y = _data()
    prob = DaceProblem(X, y, CONST, GAUSS, theta_bounds=(1e-3, 50.0))
    ev = prob(np.log10(np.array([[1e-3, 1e-3], [2.0, 2.0]])))
    assert ev.f.shape == (2,)
    assert np.all(np.isfinite(ev.f) | ~ev.feasible)  # infeasible -> inf, never an exception


def test_validation_selection_picks_lowest_held_out_error():
    from pysurrogate.dace.selection import ValidationSelection

    X, y = _data(n=24)
    Xtr, ytr, Xv, yv = X[:18], y[:18], X[18:], y[18:]
    prob = DaceProblem(Xtr, ytr, CONST, GAUSS, theta_bounds=(np.full(2, 0.01), np.full(2, 20.0)))

    cb = ValidationSelection(prob, Xv, yv)
    res = LBFGS(sampling=Sampling(5)).minimize(prob, callback=cb)
    assert res.x is not None and np.isfinite(cb.best_score)

    # the validation-selected theta is no worse on held-out than the MLE pick over the same search
    mle = Callback()
    LBFGS(sampling=Sampling(5)).minimize(prob, callback=mle)
    assert cb.best_score <= cb.score(mle.best, mle.best_f, None) + 1e-9


def test_validation_style_selection_runs_through_callback():
    # selection by a non-likelihood score still flows through the callback unchanged
    X, y = _data()
    prob = DaceProblem(X, y, CONST, GAUSS, theta_bounds=(0.01, 20.0))

    class PreferLargeTheta(Callback):
        def score(self, x, f, info):
            return -float(np.sum(x))  # contrived: prefer larger log-theta

    cb = PreferLargeTheta()
    LBFGS(sampling=Sampling(5)).minimize(prob, callback=cb)
    assert cb.best is not None
