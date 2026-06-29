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
from pysurrogate.dace.optimizers import (
    LBFGS,
    Boxmin,
    Fixed,
    Optimizer,
    ScreenedLBFGS,
    VectorizedAdam,
)
from pysurrogate.dace.regr import (
    ConstantRegression,
    LinearRegression,
    QuadraticRegression,
)

__all__ = [
    "Dace",
    "Prediction",
    "DaceFitError",
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
    # theta optimizers
    "Optimizer",
    "Boxmin",
    "Fixed",
    "LBFGS",
    "ScreenedLBFGS",
    "VectorizedAdam",
]
