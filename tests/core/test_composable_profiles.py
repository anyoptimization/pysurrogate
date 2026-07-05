"""Composable radial profiles: put a Rational-Quadratic or Matern curve on ANY metric.

The Metric x Profile split lets a profile ride a rotated (:class:`ProjectedSquare`) or reduced
(:class:`SquareThenMix`) squared metric -- kernels the separable product forms cannot express. These
tests check the values against the closed forms and the analytic gradients (spatial and theta)
against finite differences across every metric the profiles compose with.
"""

import numpy as np
import pytest

from pysurrogate.core.kernel import (
    ComposedKernel,
    MaternProfile,
    ProjectedSquare,
    RationalQuadraticProfile,
    SquareThenMix,
    WeightedSquare,
)
from pysurrogate.dace import ConstantRegression, Dace


def _metric(kind, d, h=2):
    rng = np.random.default_rng(0)
    if kind == "weighted":
        return WeightedSquare(ard=True), d
    if kind == "rotated":
        return ProjectedSquare(rng.standard_normal((d, d))), d  # full-rank rotation (Mahalanobis)
    return SquareThenMix(np.square(rng.standard_normal((d, h)))), h  # low-rank reduction (KPLS)


PROFILES = [RationalQuadraticProfile(0.25), RationalQuadraticProfile(2.0), MaternProfile(1.5), MaternProfile(2.5)]
METRICS = ["weighted", "rotated", "reduced"]


def test_rq_profile_matches_closed_form_over_weighted_square():
    rng = np.random.RandomState(1)
    D = rng.standard_normal((20, 3))
    theta = rng.random(3) + 0.2
    k = ComposedKernel(WeightedSquare(ard=True), RationalQuadraticProfile(0.7))
    s = np.sum(np.square(D) * theta, axis=1)
    assert np.allclose(k(D, theta), (1.0 + s / 0.7) ** (-0.7))


@pytest.mark.parametrize("nu,coef", [(1.5, np.sqrt(3.0)), (2.5, np.sqrt(5.0))])
def test_matern_profile_matches_closed_form_over_weighted_square(nu, coef):
    rng = np.random.RandomState(2)
    D = rng.standard_normal((20, 3))
    theta = rng.random(3) + 0.2
    k = ComposedKernel(WeightedSquare(ard=True), MaternProfile(nu))
    r = np.sqrt(np.sum(np.square(D) * theta, axis=1))
    poly = (1 + coef * r) if nu == 1.5 else (1 + coef * r + (5.0 / 3.0) * r**2)
    assert np.allclose(k(D, theta), poly * np.exp(-coef * r))


@pytest.mark.parametrize("profile", PROFILES, ids=repr)
@pytest.mark.parametrize("metric_kind", METRICS)
def test_spatial_and_theta_gradients_match_finite_differences(profile, metric_kind):
    d = 4
    metric, p = _metric(metric_kind, d)
    kernel = ComposedKernel(metric, profile)
    rng = np.random.default_rng(3)
    theta = rng.random(p) + 0.3
    D = rng.standard_normal((10, d)) + 0.8  # away from 0 so the FD stencil is clean
    eps = 1e-6

    # theta-gradient
    ana_t = kernel.theta_grad(D, theta)
    fd_t = np.zeros_like(ana_t)
    for j in range(p):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fd_t[:, j] = (kernel(D, tp) - kernel(D, tm)) / (2 * eps)
    assert np.allclose(ana_t, fd_t, atol=1e-5)

    # spatial gradient at a single point
    x = rng.standard_normal((1, d)) + 0.8
    ana_x = kernel.grad(x, theta)[0]
    fd_x = np.zeros(d)
    for j in range(d):
        xp, xm = x.copy(), x.copy()
        xp[0, j] += eps
        xm[0, j] -= eps
        fd_x[j] = (kernel(xp, theta)[0] - kernel(xm, theta)[0]) / (2 * eps)
    assert np.allclose(ana_x, fd_x, atol=1e-5)


def test_matern_profile_gradient_is_finite_at_a_coincident_point():
    # the sqrt cancels out of df/ds, so the gradient stays finite at s = 0 (D = 0)
    k = ComposedKernel(WeightedSquare(ard=True), MaternProfile(2.5))
    D = np.zeros((1, 3))
    assert np.all(np.isfinite(k.grad(D, np.array([1.0, 1.0, 1.0]))))
    assert np.all(np.isfinite(k.theta_grad(D, np.array([1.0, 1.0, 1.0]))))


def test_matern_profile_rejects_unsupported_nu():
    with pytest.raises(ValueError, match="nu in"):
        MaternProfile(0.5)


@pytest.mark.parametrize("profile", PROFILES, ids=repr)
@pytest.mark.parametrize("metric_kind", METRICS)
def test_dace_predicts_mean_variance_and_gradient(profile, metric_kind):
    # every profile x metric drives the full Dace engine and returns a valid mean, a non-negative
    # predictive variance, and a gradient -- a kernel the separable product forms cannot express (a
    # rotated / reduced RQ or Matern) is a first-class GP here, not just a point predictor.
    d = 3
    rng = np.random.RandomState(5)
    X = rng.random((30, d))
    y = np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2 - X[:, [2]]
    metric, p = _metric(metric_kind, d)
    kernel = ComposedKernel(metric, profile)
    model = Dace(regr=ConstantRegression(), corr=kernel, theta=np.full(p, 0.5), optimizer=None)
    model.fit(X, y)

    pred = model.predict(rng.random((6, d)), var=True, grad=True)
    assert pred.y.shape == (6, 1) and np.all(np.isfinite(pred.y))
    assert pred.var is not None and pred.var.shape == (6, 1)
    assert np.all(pred.var >= -1e-9) and np.all(np.isfinite(pred.var))  # non-negative GP variance
    assert pred.grad is not None and pred.grad.shape == (6, d) and np.all(np.isfinite(pred.grad))

    # exact interpolation: the predictive variance collapses to ~0 at the training sites
    assert np.all(model.predict(X, var=True).var < 1e-6)
