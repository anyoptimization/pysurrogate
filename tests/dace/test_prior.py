"""The length-scale Prior objects: GaussianPrior reproduces the (mean, lam) tuple; resolve_prior."""

import numpy as np

from pysurrogate.dace import MAP, ConstantRegression, Dace, Gaussian, GaussianPrior, Prior
from pysurrogate.dace.prior import resolve_prior


def test_resolve_prior_handles_none_tuple_and_object():
    assert resolve_prior(None) is None
    g = resolve_prior((0.5, 0.02))  # tuple -> GaussianPrior
    assert isinstance(g, GaussianPrior) and g.mean == 0.5 and g.lam == 0.02
    p = GaussianPrior(1.0, 0.1)
    assert resolve_prior(p) is p  # a Prior passes through unchanged


def test_gaussian_prior_matches_the_tuple_penalty_and_gradient():
    # GaussianPrior must be the ridge the old (mean, lam) tuple encoded, byte-for-byte
    rng = np.random.RandomState(0)
    Z = rng.standard_normal((5, 3))
    mean, lam = 0.3, 0.07
    g = GaussianPrior(mean, lam)
    np.testing.assert_array_equal(g.penalty(Z), lam * np.sum((Z - mean) ** 2, axis=1))
    np.testing.assert_array_equal(g.grad(Z), 2.0 * lam * (Z - mean))


def test_gaussian_prior_gradient_matches_finite_difference():
    rng = np.random.RandomState(1)
    Z = rng.standard_normal((4, 3))
    g = GaussianPrior(0.2, 0.05)
    eps = 1e-6
    fd = np.zeros_like(Z)
    for j in range(Z.shape[1]):
        zp, zm = Z.copy(), Z.copy()
        zp[:, j] += eps
        zm[:, j] -= eps
        fd[:, j] = (g.penalty(zp) - g.penalty(zm)) / (2 * eps)
    assert np.allclose(g.grad(Z), fd, atol=1e-6)


def test_dace_accepts_a_prior_object_equivalently_to_the_tuple():
    # passing GaussianPrior(mean, lam) must fit identically to passing the (mean, lam) tuple
    rng = np.random.RandomState(2)
    X = rng.uniform(-1, 1, size=(30, 2))
    y = np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2

    def theta(prior):
        m = Dace(regr=ConstantRegression(), corr=Gaussian(), theta=1.0, theta_bounds=(0.01, 100.0), theta_prior=prior)
        m.fit(X, y)
        return m.model["theta"]

    np.testing.assert_array_equal(theta((0.0, 0.01)), theta(GaussianPrior(0.0, 0.01)))


def test_map_selection_uses_a_gaussian_prior_under_the_hood():
    # the MAP strategy is just a Gaussian length-scale prior (resolved from its tuple)
    prior = resolve_prior(MAP(mean=0.5, lam=0.1).theta_prior)
    assert isinstance(prior, GaussianPrior) and prior.mean == 0.5 and prior.lam == 0.1


def test_prior_base_is_abstract():
    p = Prior()
    for method in (p.penalty, p.grad):
        try:
            method(np.zeros((1, 2)))
        except NotImplementedError:
            continue
        raise AssertionError("Prior base methods should be abstract")
