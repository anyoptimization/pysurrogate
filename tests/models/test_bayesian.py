"""FullyBayesianGP: the HMC sampler, the GP log-posterior gradient, and the calibration payoff."""

import numpy as np
import pytest

from pysurrogate.dace import ConstantRegression, Gaussian, GaussianPrior
from pysurrogate.dace.hmc import hmc_sample
from pysurrogate.models import FullyBayesianGP, Kriging
from pysurrogate.models.bayesian import _GPPosterior


def _toy(seed=0, n=40, d=4, noise=0.25):
    """A noisy, mildly nonlinear single-output problem: (train X, train y, test X, clean test y)."""
    r = np.random.RandomState(seed)
    f = lambda Z: np.sin(2 * Z[:, 0]) + Z[:, 1] ** 2  # noqa: E731
    Xtr = r.uniform(-1.5, 1.5, (n, d))
    ytr = (f(Xtr) + noise * r.standard_normal(n)).reshape(-1, 1)
    Xte = r.uniform(-1.5, 1.5, (600, d))
    yte = f(Xte).reshape(-1, 1)
    return Xtr, ytr, Xte, yte


def _nlpd(mu, var, t):
    """Mean negative log predictive density under a Gaussian -- lower is better; punishes overconfidence."""
    var = np.clip(var, 1e-8, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (t - mu) ** 2 / var))


# --------------------------------------------------------------------------------------- the sampler


def test_hmc_recovers_a_known_gaussian_posterior():
    # HMC on U(x) = 1/2 (x-mu)^T P (x-mu) must reproduce mean=mu and cov=P^-1 (P = precision).
    mu = np.array([1.0, -2.0, 0.5])
    P = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, -0.2], [0.0, -0.2, 3.0]])
    cov = np.linalg.inv(P)

    def potential(x):
        d = x - mu
        return 0.5 * d @ P @ d

    def grad(x):
        return P @ (x - mu)

    samples, accept = hmc_sample(potential, grad, np.zeros(3), n_samples=4000, n_warmup=600, n_leapfrog=20, seed=0)
    assert 0.6 < accept < 0.99
    assert np.allclose(samples.mean(axis=0), mu, atol=0.06)
    assert np.allclose(np.cov(samples.T), cov, atol=0.15)


def test_hmc_is_deterministic_given_the_seed():
    potential = lambda x: 0.5 * x @ x  # noqa: E731
    grad = lambda x: x  # noqa: E731
    a, _ = hmc_sample(potential, grad, np.zeros(2), n_samples=200, n_warmup=100, seed=3)
    b, _ = hmc_sample(potential, grad, np.zeros(2), n_samples=200, n_warmup=100, seed=3)
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------------------ the GP log-posterior


def test_gp_log_posterior_gradient_matches_finite_difference():
    # the potential reuses DaceProblem's obj/grad through a log transform + a log-space prior; the
    # analytic gradient must match a central finite difference (this is the whole correctness claim).
    r = np.random.RandomState(0)
    X = r.uniform(-1, 1, (18, 3))
    y = (np.sin(2 * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1)
    sX = np.std(X, axis=0, ddof=1)
    sY = np.std(y, axis=0, ddof=1)
    nX, nY = (X - X.mean(0)) / sX, (y - y.mean(0)) / sY
    post = _GPPosterior(
        nX,
        nY,
        ConstantRegression(),
        Gaussian(ard=True),
        GaussianPrior(0.0, 0.1),
        (np.full(3, 1e-6), np.full(3, 1e6)),
        (1e-8, 1.0),
    )
    x = np.append(np.array([0.1, -0.3, 0.2]), -2.5)  # [log10 theta (3), log10 noise]
    eps = 1e-6
    fd = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (post.potential(xp) - post.potential(xm)) / (2 * eps)
    assert np.max(np.abs(post.grad(x) - fd)) < 1e-4


class _StubProblem:
    """A DaceProblem stand-in that reports a fixed ``(obj, feasible)`` -- to drive the guard branches."""

    def __init__(self, obj, feasible):
        self._obj, self._feasible = obj, feasible
        self.p = 2

    def __call__(self, X):
        from types import SimpleNamespace

        return SimpleNamespace(f=np.array([self._obj]), feasible=np.array([self._feasible]), grad=np.ones((1, 3)))


@pytest.mark.parametrize("obj,feasible", [(1.5, False), (-1.0, True), (np.inf, True)])
def test_guard_maps_a_bad_objective_to_infinite_potential_and_zero_gradient(obj, feasible):
    # an infeasible candidate (non-PD R), a non-positive objective, or a non-finite one must all give
    # U=+inf and grad=0, so the leapfrog integrator rejects and coasts through instead of crashing.
    post = _GPPosterior.__new__(_GPPosterior)
    post.problem = _StubProblem(obj, feasible)
    post.n, post.p, post.prior = 10, 2, None
    x = np.array([0.0, 0.0, -3.0])
    assert not np.isfinite(post.potential(x))
    np.testing.assert_array_equal(post.grad(x), np.zeros_like(x))


# ------------------------------------------------------------------------------------- the backend


def test_fit_predict_shapes_positivity_and_determinism():
    X, y, Xte, _ = _toy(seed=0, n=30, d=3)
    m = FullyBayesianGP(n_samples=60, n_warmup=250, random_state=0).fit(X, y)
    assert len(m.gps_) == 30  # n_samples // thin
    assert 0.5 < m.accept_rate_ <= 1.0
    p = m.predict(Xte, var=True, grad=True)
    assert p.y.shape == (len(Xte), 1)
    assert p.var.shape == (len(Xte), 1) and np.all(p.var > 0)
    assert p.grad.shape == (len(Xte), 3)
    # same seed -> byte-identical predictions (the whole fit is deterministic)
    m2 = FullyBayesianGP(n_samples=60, n_warmup=250, random_state=0).fit(X, y)
    np.testing.assert_array_equal(p.y, m2.predict(Xte, var=True).y)


def test_bma_variance_exceeds_the_averaged_conditional_variance():
    # the between-sample spread of the means is non-negative, so the model-averaged variance is
    # always at least the mean conditional variance -- the extra term IS the hyperparameter uncertainty.
    X, y, Xte, _ = _toy(seed=1, n=30, d=3)
    m = FullyBayesianGP(n_samples=60, n_warmup=250, random_state=1).fit(X, y)
    total = m.predict(Xte, var=True).var
    within = np.stack([gp.predict(Xte, var=True).var for gp in m.gps_]).mean(axis=0)
    assert np.all(total >= within - 1e-12)
    assert np.any(total > within + 1e-8)  # some genuine between-sample disagreement exists


def test_rejects_multi_output():
    r = np.random.RandomState(0)
    X = r.uniform(-1, 1, (20, 2))
    Y = r.uniform(-1, 1, (20, 2))  # two outputs
    with pytest.raises(ValueError, match="single-output"):
        FullyBayesianGP(n_samples=20, n_warmup=100).fit(X, Y)


@pytest.mark.slow
def test_calibrates_better_than_interpolating_kriging_on_noisy_data():
    # Scope: on a SMOOTH, low-effective-dimensional noisy problem, the Bayesian model-average reports
    # honest uncertainty, whereas an INTERPOLATING MLE-Kriging (ARD, no learned nugget) forced through
    # the noise is overconfident. This is the specific failure mode FBGP avoids -- not a universal win:
    # against a nugget-learning Kriging, or on rugged multimodal landscapes, FBGP is only comparable
    # (and sometimes worse). Asserted on the aggregate NLPD across seeds (robust), not per-seed.
    bayes, krig = [], []
    for s in range(5):
        Xtr, ytr, Xte, yte = _toy(seed=s, n=40, d=4)
        bp = FullyBayesianGP(random_state=s).fit(Xtr, ytr).predict(Xte, var=True)
        # deliberately a no-nugget interpolation of noisy data -- the overconfident baseline.
        kp = Kriging(ARD=True).fit(Xtr, ytr).predict(Xte, var=True)
        bayes.append(_nlpd(bp.y, bp.var, yte))
        krig.append(_nlpd(kp.y, kp.var, yte))
    # empirically ~0.19 vs ~0.99 against this no-nugget baseline; the loose bound guards against
    # float/seed jitter while still failing loudly if the marginalization ever stops calibrating.
    assert np.mean(bayes) < 0.6 * np.mean(krig)
