"""DACE Kriging engine: the ``Dace`` model plus its regression and correlation parts."""

from pysurrogate.core.prediction import Prediction
from pysurrogate.dace.corr import (
    Cubic,
    Exponential,
    Gaussian,
    GeneralizedExponential,
    Linear,
    Matern,
    RationalQuadratic,
    Spherical,
    Spline,
)
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.fit import DaceFitError
from pysurrogate.dace.regr import (
    ConstantRegression,
    LinearRegression,
    QuadraticRegression,
)
from pysurrogate.dace.selection import (
    MAP,
    HeldOut,
    MaximumLikelihood,
    Selection,
)

__all__ = [
    "Dace",
    "Prediction",
    "DaceFitError",
    # hyperparameter-selection strategies
    "Selection",
    "MaximumLikelihood",
    "MAP",
    "HeldOut",
    # regression trends
    "ConstantRegression",
    "LinearRegression",
    "QuadraticRegression",
    # correlation kernels
    "Gaussian",
    "Cubic",
    "Exponential",
    "Linear",
    "GeneralizedExponential",
    "Spline",
    "Spherical",
    "RationalQuadratic",
    "Matern",
]
