"""RBF is rebuilt on the shared core kernel zoo: interpolation, gradients, and one distance path."""

import numpy as np
import pytest

from pysurrogate.core.kernel import CubicRadial, Gaussian, Kernel, LinearRadial, Multiquadric, ThinPlateSpline
from pysurrogate.models import RBF


@pytest.mark.parametrize(
    "name,cls",
    [
        ("linear", LinearRadial),
        ("cubic", CubicRadial),
        ("gaussian", Gaussian),
        ("mq", Multiquadric),
        ("tps", ThinPlateSpline),
    ],
)
def test_rbf_uses_the_shared_core_kernels(name, cls):
    # the whole point of the rebuild: RBF's kernels ARE the framework's kernel objects, not a
    # private scalar-distance zoo. gaussian reuses the covariance kernel directly.
    model = RBF(kernel=name)
    assert isinstance(model.kernel, Kernel)
    assert isinstance(model.kernel, cls)


def test_unknown_kernel_raises():
    with pytest.raises(ValueError, match="Unknown kernel function"):
        RBF(kernel="not-a-kernel")


@pytest.mark.parametrize("kernel", ["tps", "cubic", "gaussian", "mq", "linear"])
def test_rbf_interpolates_training_points(kernel):
    # an exact RBF interpolant (tiny ridge) reproduces the training targets it was fit on.
    rng = np.random.RandomState(0)
    X = rng.random((25, 2))
    y = np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2
    model = RBF(kernel=kernel, rho=0.0).fit(X, y)
    assert np.allclose(model.predict(X).y, y, atol=1e-6)


def test_tune_sigma_skips_grid_for_sigma_free_kernel():
    # tps depends on the radius only and ignores sigma, so tune_sigma must NOT run the LOOCV grid
    # (it would refit the identical model 30 times) -- the fitted sigma stays the constructor value.
    rng = np.random.RandomState(0)
    X = rng.random((25, 2))
    y = np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2
    m = RBF(kernel="tps", tune_sigma=True).fit(X, y)
    assert m.model["sigma"] == 1.0
    # a sigma-using kernel (gaussian) does search the grid and lands on a different sigma
    g = RBF(kernel="gaussian", tune_sigma=True).fit(X, y)
    assert g.model["sigma"] != 1.0


def test_optimize_is_a_backcompat_alias_for_tune_sigma():
    # `optimize=` was the former constructor name for `tune_sigma=`; it must still drive the search.
    rng = np.random.RandomState(0)
    X = rng.random((25, 2))
    y = np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2
    assert RBF(kernel="gaussian", optimize=True).tune_sigma is True
    g = RBF(kernel="gaussian", optimize=True).fit(X, y)
    assert g.model["sigma"] != 1.0


def test_rbf_gaussian_is_no_longer_the_quartic_bug():
    # the old kernel_gaussian applied exp(-sigma * r**2) to the ALREADY-squared distance r, giving a
    # quartic exp(-sigma * ||.||**4). Rebuilt on the covariance Gaussian, a single center evaluates as
    # exp(-sigma * ||x - c||**2). Check the fitted single-point kernel column matches that directly.
    c = np.zeros((1, 2))
    q = np.array([[0.5, 0.4]])
    sigma = 1.3
    from pysurrogate.core.kernel import calc_kernel_matrix

    val = calc_kernel_matrix(q, c, Gaussian(), sigma)[0, 0]
    r2 = float((q**2).sum())
    assert np.isclose(val, np.exp(-sigma * r2))
    assert not np.isclose(val, np.exp(-sigma * r2**2))  # not the old quartic


def test_svd_inv_truncates_singular_values_instead_of_amplifying_them():
    # regression: a singular system used to hit 1/S -> inf. Truncating tiny singular values yields a
    # finite least-norm pseudo-inverse (A @ A_inv @ A == A) rather than an inf-poisoned matrix.
    from pysurrogate.models.rbf import svd_inv

    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])  # row 1 = 2*row 0 -> rank 2
    A_inv, _ = svd_inv(A)
    assert np.all(np.isfinite(A_inv))
    np.testing.assert_allclose(A @ A_inv @ A, A, atol=1e-9)
