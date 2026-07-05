"""The Parameter concept: encodings, ParameterSpace layout, and the parameters(d) kernels build."""

import numpy as np
import pytest

from pysurrogate.core.kernel import (
    CubicRadial,
    Exponential,
    Gaussian,
    GeneralizedExponential,
    KPLSKernel,
    Mahalanobis,
    Matern,
    ThinPlateSpline,
)
from pysurrogate.core.parameter import Log10, Parameter, ParameterSpace


def test_log10_encoding_decodes_and_transforms_bounds():
    enc = Log10()
    np.testing.assert_allclose(enc.to_value(np.array([-2.0, 0.0, 3.0])), [0.01, 1.0, 1000.0])
    lo, hi = enc.bounds(0.001, 1000.0)
    assert (lo, hi) == (-3.0, 3.0)


def test_parameter_space_bounds_and_decode():
    space = ParameterSpace(
        [
            Parameter("theta", size=3, bounds=(0.001, 1000.0), encoding=Log10()),
            Parameter("power", size=1, bounds=(0.1, 10.0), encoding=Log10()),
        ]
    )
    lo, hi = space.bounds()
    np.testing.assert_allclose(lo, [-3, -3, -3, -1.0])
    np.testing.assert_allclose(hi, [3, 3, 3, 1.0])

    x = np.array([0.0, 1.0, -1.0, 0.0])
    dec = space.decode(x)
    np.testing.assert_allclose(dec["theta"], [1.0, 10.0, 0.1])
    np.testing.assert_allclose(dec["power"], [1.0])


@pytest.mark.parametrize("ard", [False, True])
def test_parameters_sizes_match_n_theta(ard):
    d = 4
    for kernel in [Gaussian(ard=ard), Exponential(ard=ard), Matern(nu=1.5, ard=ard), GeneralizedExponential(ard=ard)]:
        total = sum(p.size for p in kernel.parameters(d))
        assert total == kernel.n_theta(d)


def test_composed_parameters_are_metric_then_profile():
    # the composed kernel's declaration is literally the metric's params followed by the profile's
    params = Gaussian(ard=True).parameters(5)
    assert [p.name for p in params] == ["theta"]  # Exp profile contributes none
    assert params[0].size == 5 and params[0].fill


def test_generalized_exponential_declares_power_after_theta():
    params = GeneralizedExponential(ard=True).parameters(3)
    assert [p.name for p in params] == ["theta", "power"]
    assert params[0].size == 3 and params[0].fill  # length-scales are caller-sized
    assert params[1].size == 1 and not params[1].fill  # the exponent is a fixed shape coordinate


def test_reduced_and_rotated_metrics_declare_h_length_scales():
    d, h = 5, 2
    w2 = np.square(np.random.RandomState(0).standard_normal((d, h)))
    A = np.random.RandomState(1).standard_normal((d, h))
    assert sum(p.size for p in KPLSKernel(Gaussian(), w2).parameters(d)) == h
    assert sum(p.size for p in Mahalanobis(A).parameters(d)) == h


def test_radial_bases_declare_no_searchable_parameters():
    assert CubicRadial().parameters(3) == []
    assert ThinPlateSpline().parameters(2) == []
