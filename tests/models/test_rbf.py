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
