"""MAP prior on the length-scale search: pure-MLE default, regularization, and its gradient."""

import numpy as np

from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.problem import DaceProblem
from pysurrogate.dace.regr import ConstantRegression
from pysurrogate.models import Kriging


def _data(seed=0, n=25, d=2):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sum(np.sin(3.0 * X), axis=1)
    return X, y


def test_prior_lambda_zero_matches_pure_mle():
    # theta_prior with lam=0 adds a zero penalty, so it must reproduce the MLE fit exactly (the same
    # default deterministic optimizer runs on an identical objective) -- the golden-safe guarantee.
    X, y = _data()
    mle = Dace(corr=Gaussian(), theta=1.0, theta_bounds=(1e-3, 1e3), theta_prior=None)
    mle.fit(X, y)  # Dace.fit returns None (engine, not a Model) -- read the fitted dict after
    zero = Dace(corr=Gaussian(), theta=1.0, theta_bounds=(1e-3, 1e3), theta_prior=(0.0, 0.0))
    zero.fit(X, y)
    assert np.allclose(np.ravel(mle.model["theta"]), np.ravel(zero.model["theta"]))


def test_strong_prior_collapses_theta_to_mean():
    # a dominant MAP prior swamps the likelihood -> the fitted log10(theta) collapses onto `mean`
    X, y = _data()
    mean = 1.5
    m = Dace(corr=Gaussian(), theta=1.0, theta_bounds=(1e-3, 1e3), theta_prior=(mean, 1e8))
    m.fit(X, y)
    z = np.log10(np.ravel(m.model["theta"]))
    assert np.allclose(z, mean, atol=0.05)


def test_kriging_forwards_theta_prior():
    # Kriging exposes theta_prior and forwards it to its Dace engine
    X, y = _data()
    mle = Kriging(corr=Gaussian(), theta_bounds=(1e-3, 1e3)).fit(X, y)
    mapf = Kriging(corr=Gaussian(), theta_bounds=(1e-3, 1e3), theta_prior=(1.5, 1e8)).fit(X, y)
    z_mle = np.log10(np.ravel(mle.model.model["theta"]))
    z_map = np.log10(np.ravel(mapf.model.model["theta"]))
    assert np.allclose(z_map, 1.5, atol=0.05)  # pulled onto the prior mean
    assert not np.allclose(z_mle, 1.5, atol=0.05)  # MLE lands elsewhere


def test_penalized_objective_gradient_matches_finite_difference():
    # the analytic gradient of (likelihood + MAP penalty) must match central finite differences,
    # so a gradient optimizer (LBFGS/Adam) descends the regularized objective correctly.
    X, y = _data(seed=3, n=20, d=2)
    prob = DaceProblem(X, y, ConstantRegression(), Gaussian(), ([1e-3, 1e-3], [1e3, 1e3]), theta_prior=(0.3, 0.7))
    z = np.array([[0.1, -0.2]])  # a log10-theta point (p = 2)
    g = prob(z).grad[0]

    eps = 1e-6
    fd = np.zeros(2)
    for k in range(2):
        zp, zm = z.copy(), z.copy()
        zp[0, k] += eps
        zm[0, k] -= eps
        fd[k] = (prob(zp).f[0] - prob(zm).f[0]) / (2 * eps)
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_screen_and_call_share_the_penalized_objective():
    # Restart screens with screen() and polishes with __call__(); both must include the prior so the
    # screen ranks candidates by the same objective the gradient path then optimizes.
    X, y = _data(seed=4, n=18, d=2)
    prob = DaceProblem(X, y, ConstantRegression(), Gaussian(), ([1e-3, 1e-3], [1e3, 1e3]), theta_prior=(0.5, 2.0))
    Z = np.array([[0.2, -0.4], [1.0, 0.3]])
    assert np.allclose(prob.screen(Z), prob(Z).f)
