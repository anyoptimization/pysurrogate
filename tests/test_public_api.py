"""The reusable kernel, composition, regression, and parameter components are importable at the root."""

import importlib

import pytest

import pysurrogate


@pytest.mark.parametrize(
    "name",
    [
        # kernel zoo
        "Gaussian",
        "Exponential",
        "Matern",
        "RationalQuadratic",
        "GeneralizedExponential",
        "Cubic",
        "Spline",
        "Spherical",
        "ThinPlateSpline",
        "Multiquadric",
        "LinearRadial",
        "CubicRadial",
        # composition seam
        "Kernel",
        "Metric",
        "Profile",
        "ComposedKernel",
        "ProductKernel",
        "RadialKernel",
        "ReducedMetric",
        "Exp",
        "WeightedSquare",
        "WeightedAbs",
        "ProjectedSquare",
        "SquareThenMix",
        "Mahalanobis",
        "KPLSKernel",
        # regression + parameters
        "Regression",
        "ConstantRegression",
        "LinearRegression",
        "QuadraticRegression",
        "Parameter",
        "ParameterSpace",
        "Encoding",
        "Log10",
    ],
)
def test_component_is_importable_from_root(name):
    assert hasattr(pysurrogate, name), f"pysurrogate.{name} is not exported"
    assert name in pysurrogate.__all__, f"{name} missing from __all__"


def test_all_names_resolve():
    # every name promised in __all__ must actually exist on the package (no dangling promises)
    for name in pysurrogate.__all__:
        assert hasattr(pysurrogate, name), f"__all__ lists {name} but it is not defined"


def test_star_import_exposes_the_composition_primitives():
    ns = {}
    exec("from pysurrogate import *", ns)  # noqa: S102 - exercising the public star-import surface
    for name in ["Gaussian", "ComposedKernel", "WeightedSquare", "Exp", "Mahalanobis", "Parameter"]:
        assert name in ns


def test_root_gaussian_is_the_core_gaussian():
    # the root export is the same object as the canonical core kernel, not a copy
    core = importlib.import_module("pysurrogate.core.kernel")
    assert pysurrogate.Gaussian is core.Gaussian
    assert pysurrogate.ComposedKernel is core.ComposedKernel
