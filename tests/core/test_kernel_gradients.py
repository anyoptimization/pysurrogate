"""Exhaustive analytic-gradient net: every kernel's grad and theta_grad vs finite differences.

Gradient math is the historical source of kernel bugs (see ``ProductKernel``). This one net checks
the spatial gradient ``grad`` (used by ``predict(grad=True)``) and, where analytic, the length-scale
gradient ``theta_grad`` (used by the likelihood search) against central finite differences for the
*entire* zoo -- covariances, compact-support product kernels, the reduced/rotated metrics, and the
conditionally-PD radial bases. Adding a new kernel means adding one line to ``CASES``.

Points are drawn in a small positive box so the scaled distance ``t = theta*|D|`` stays in the smooth
interior of the compact-support kernels (away from the ``t = 1`` clamp, where the factor has a kink).
"""

import numpy as np
import pytest

from pysurrogate.core.kernel import (
    Cubic,
    CubicRadial,
    Exponential,
    Gaussian,
    GeneralizedExponential,
    KPLSKernel,
    Linear,
    LinearRadial,
    Mahalanobis,
    Matern,
    Multiquadric,
    RationalQuadratic,
    Spherical,
    Spline,
    ThinPlateSpline,
)

_D = 3
_RNG = np.random.default_rng(7)
_A = _RNG.standard_normal((_D, 2))  # rotation for Mahalanobis (rank 2)
_W2 = np.square(_RNG.standard_normal((_D, 2)))  # KPLS squared PLS weights (rank 2)

# (id, kernel, theta) -- theta shaped for the kernel. Length-scales are kept <= 0.6 and points are
# drawn in [0.2, 0.8] below, so compact-support kernels stay strictly inside t < 1.
CASES = [
    ("gauss/iso", Gaussian(), np.array([0.5])),
    ("gauss/ard", Gaussian(ard=True), np.array([0.4, 0.5, 0.6])),
    ("exp/iso", Exponential(), np.array([0.5])),
    ("exp/ard", Exponential(ard=True), np.array([0.4, 0.5, 0.6])),
    ("matern05/iso", Matern(nu=0.5), np.array([0.5])),
    ("matern15/ard", Matern(nu=1.5, ard=True), np.array([0.4, 0.5, 0.6])),
    ("matern25/iso", Matern(nu=2.5), np.array([0.5])),
    ("rq/iso", RationalQuadratic(alpha=1.0), np.array([0.5])),
    ("rq/ard", RationalQuadratic(alpha=0.5, ard=True), np.array([0.4, 0.5, 0.6])),
    ("cubic/iso", Cubic(), np.array([0.5])),
    ("spline/ard", Spline(ard=True), np.array([0.4, 0.5, 0.6])),
    ("spherical/iso", Spherical(), np.array([0.5])),
    ("linear/ard", Linear(ard=True), np.array([0.4, 0.5, 0.6])),
    ("expg/iso", GeneralizedExponential(), np.array([0.5, 1.5])),  # (length-scale, power)
    ("expg/ard", GeneralizedExponential(ard=True), np.array([0.4, 0.5, 0.6, 1.5])),
    ("mahalanobis", Mahalanobis(_A), np.array([0.4, 0.6])),
    ("kpls/gauss", KPLSKernel(Gaussian(), _W2), np.array([0.4, 0.6])),
    ("kpls/exp", KPLSKernel(Exponential(), _W2), np.array([0.4, 0.6])),
]

# radial bases: spatial gradient only (no theta search). Points kept away from 0 so TPS's log and
# the sqrt bases are smooth.
RADIAL = [
    ("tps", ThinPlateSpline(), 1.0),
    ("mq", Multiquadric(), 1.3),
    ("linear-radial", LinearRadial(), 1.0),
    ("cubic-radial", CubicRadial(), 1.0),
]

_EPS = 1e-6


def _points(n=8):
    return _RNG.uniform(0.2, 0.8, size=(n, _D))  # positive, moderate -> smooth interior everywhere


def _fd_spatial(kernel, D, theta):
    fd = np.zeros(D.shape)
    for j in range(D.shape[1]):
        dp, dm = D.copy(), D.copy()
        dp[:, j] += _EPS
        dm[:, j] -= _EPS
        fd[:, j] = (kernel(dp, theta) - kernel(dm, theta)) / (2 * _EPS)
    return fd


def _fd_theta(kernel, D, theta):
    fd = np.zeros((D.shape[0], len(theta)))
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += _EPS
        tm[j] -= _EPS
        fd[:, j] = (kernel(D, tp) - kernel(D, tm)) / (2 * _EPS)
    return fd


@pytest.mark.parametrize("name,kernel,theta", CASES, ids=[c[0] for c in CASES])
def test_spatial_gradient_matches_finite_difference(name, kernel, theta):
    D = _points()
    assert np.allclose(kernel.grad(D, theta), _fd_spatial(kernel, D, theta), atol=1e-5)


@pytest.mark.parametrize("name,kernel,theta", CASES, ids=[c[0] for c in CASES])
def test_theta_gradient_matches_finite_difference(name, kernel, theta):
    assert kernel.has_theta_grad, f"{name} should advertise an analytic theta-gradient"
    D = _points()
    ana = kernel.theta_grad(D, theta)
    fd = _fd_theta(kernel, D, theta)
    assert ana.shape == (D.shape[0], len(theta))
    assert np.allclose(ana, fd, atol=1e-5)


@pytest.mark.parametrize("name,kernel,theta", RADIAL, ids=[c[0] for c in RADIAL])
def test_radial_basis_spatial_gradient_matches_finite_difference(name, kernel, theta):
    D = _points()  # away from 0 so the log / sqrt bases are smooth
    assert np.allclose(kernel.grad(D, theta), _fd_spatial(kernel, D, theta), atol=1e-5)
    assert kernel.has_theta_grad is False  # radial bases are not theta-searched


def test_every_dace_searchable_kernel_is_covered():
    # guard against a new searchable kernel slipping in without a gradient check: the covered set
    # must include one instance of each concrete searchable kernel class in the zoo.
    covered = {type(k).__name__ for _, k, _ in CASES}
    expected = {
        "Gaussian",
        "Exponential",
        "Matern",
        "RationalQuadratic",
        "Cubic",
        "Spline",
        "Spherical",
        "Linear",
        "GeneralizedExponential",
        "Mahalanobis",
        "KPLSKernel",
    }
    assert expected <= covered, f"uncovered searchable kernels: {expected - covered}"
