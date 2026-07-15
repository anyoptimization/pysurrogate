"""The unified core kernel zoo: ard semantics, the Correlation alias, and the radial bases."""

import numpy as np
import pytest

from pysurrogate.core.kernel import (
    Correlation,
    Gaussian,
    GeneralizedExponential,
    Kernel,
    Mahalanobis,
    Matern,
    Multiquadric,
    ThinPlateSpline,
)


def test_mahalanobis_identity_projection_equals_ard_gaussian():
    # A = I makes the metric diagonal, so Mahalanobis must reduce EXACTLY to ARD-Gaussian on every
    # method (value, batch, both gradients) -- the fixed reference point for the reparameterization.
    rng = np.random.default_rng(0)
    d, n = 4, 20
    D = rng.standard_normal((n, d))
    theta = rng.random(d) + 0.3
    thetas = rng.random((5, d)) + 0.2
    mah, g = Mahalanobis(np.eye(d)), Gaussian()
    assert np.allclose(mah(D, theta), g(D, theta))
    assert np.allclose(mah.batch(D, thetas), g.batch(D, thetas))
    assert np.allclose(mah.theta_grad(D, theta), g.theta_grad(D, theta))
    assert np.allclose(mah.grad(D, theta), g.grad(D, theta))
    assert mah.n_theta(d) == d and mah.has_theta_grad


def test_mahalanobis_realizes_the_quadratic_form_and_reduces_rank():
    # the kernel must equal exp(-dᵀ M d) with M = A diag(theta) Aᵀ, including the off-diagonal
    # (rotation) cross terms a diagonal kernel cannot express; a rank-h A optimizes only h scales.
    rng = np.random.default_rng(1)
    d, h, n = 5, 3, 15
    A = rng.standard_normal((d, h))  # a genuinely rotated, rank-3 projection
    theta = rng.random(h) + 0.2
    D = rng.standard_normal((n, d))
    M = A @ np.diag(theta) @ A.T
    assert np.allclose(Mahalanobis(A)(D, theta), np.exp(-np.einsum("ni,ij,nj->n", D, M, D)))
    assert Mahalanobis(A).n_theta(d) == h  # only h length-scales, not d


def test_mahalanobis_gradients_match_finite_differences():
    # analytic theta- and spatial-gradients must match central differences so LBFGS/Adam and
    # predict(grad=True) are exact on a rotated low-rank metric.
    rng = np.random.default_rng(2)
    d, h = 4, 3
    A = rng.standard_normal((d, h))
    k = Mahalanobis(A)
    theta = rng.random(h) + 0.2
    D = rng.standard_normal((12, d))
    eps = 1e-6

    ana = k.theta_grad(D, theta)
    fd = np.zeros_like(ana)
    for j in range(h):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fd[:, j] = (k(D, tp) - k(D, tm)) / (2 * eps)
    assert np.allclose(ana, fd, atol=1e-6)

    x = rng.standard_normal((1, d))
    g = k.grad(x, theta)[0]
    fdx = np.zeros(d)
    for j in range(d):
        xp, xm = x.copy(), x.copy()
        xp[0, j] += eps
        xm[0, j] -= eps
        fdx[j] = (k(xp, theta)[0] - k(xm, theta)[0]) / (2 * eps)
    assert np.allclose(g, fdx, atol=1e-6)


def test_generalized_exponential_n_theta_counts_the_power():
    # GeneralizedExponential's theta is (length_scale(s)..., power), so n_theta must add 1 for the
    # shared exponent -- the base Kernel.n_theta would under-report by one.
    assert GeneralizedExponential(ard=False).n_theta(5) == 2  # one length-scale + power
    assert GeneralizedExponential(ard=True).n_theta(5) == 6  # five length-scales + power


def test_generalized_exponential_split_rejects_wrong_length_theta():
    # theta must be (length_scale, power) [len 2] or (length_scales..., power) [len d+1]; anything
    # else is a caller error and must raise ValueError (was a bare Exception; len() also failed on 0-d).
    ge = GeneralizedExponential()
    D = np.zeros((4, 3))  # d = 3, so a valid theta is length 2 or length 4
    with pytest.raises(ValueError, match="length 2 or d\\+1"):
        ge(D, np.array([1.0, 1.0, 1.0]))  # length 3: neither 2 nor d+1


def test_correlation_is_kernel_alias():
    # the DACE layer's historical name must remain the same class object
    assert Correlation is Kernel
    assert isinstance(Gaussian(), Correlation)


@pytest.mark.parametrize("ard,d,expected", [(False, 4, 1), (True, 4, 4), (False, 1, 1), (True, 1, 1)])
def test_n_theta_reflects_ard(ard, d, expected):
    assert Gaussian(ard=ard).n_theta(d) == expected
    assert Matern(nu=1.5, ard=ard).n_theta(d) == expected


def test_ard_false_shared_theta_matches_ard_true_equal_per_dim():
    # ard is a declaration; the math is shape-driven, so a shared scalar theta and a
    # per-dimension theta with all-equal entries must give identical correlations.
    rng = np.random.RandomState(0)
    D = rng.standard_normal((20, 3))
    t = 0.7
    iso = Gaussian(ard=False)(D, np.array([t]))
    ard = Gaussian(ard=True)(D, np.array([t, t, t]))
    assert np.allclose(iso, ard)


def test_ard_true_matches_per_dim_dace_path():
    # per-dimension theta must weight each coordinate independently
    rng = np.random.RandomState(1)
    D = rng.standard_normal((15, 2))
    theta = np.array([0.3, 1.9])
    expected = np.exp(np.sum(np.square(D) * -theta, axis=1))
    assert np.allclose(Gaussian(ard=True)(D, theta), expected)


def test_thin_plate_spline_value():
    rng = np.random.RandomState(2)
    D = rng.standard_normal((10, 3))
    r2 = np.sum(D**2, axis=1)
    r = np.sqrt(np.maximum(r2, np.finfo(float).eps))
    assert np.allclose(ThinPlateSpline()(D, None), r**2 * np.log(r))


@pytest.mark.parametrize("kernel,theta", [(ThinPlateSpline(), 1.0), (Multiquadric(), 1.3)])
def test_radial_spatial_gradient_matches_finite_difference(kernel, theta):
    rng = np.random.RandomState(3)
    D = rng.standard_normal((6, 2)) + 1.5  # away from 0 so TPS log is smooth
    g = kernel.grad(D, theta)
    eps = 1e-6
    fd = np.zeros_like(D)
    for k in range(D.shape[1]):
        Dp, Dm = D.copy(), D.copy()
        Dp[:, k] += eps
        Dm[:, k] -= eps
        fd[:, k] = (kernel(Dp, theta) - kernel(Dm, theta)) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-4)


def test_multiquadric_value():
    rng = np.random.RandomState(4)
    D = rng.standard_normal((8, 2))
    r2 = np.sum(D**2, axis=1)
    assert np.allclose(Multiquadric()(D, 1.5), np.sqrt(r2 + 1.5**2))


def test_radial_kernels_have_no_theta_grad():
    # conditionally-PD radial bases are not theta-searched
    assert ThinPlateSpline().has_theta_grad is False
    assert Multiquadric().has_theta_grad is False
