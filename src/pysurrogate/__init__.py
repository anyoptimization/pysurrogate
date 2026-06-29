"""pysurrogate — a unified surrogate-modeling toolkit (sampling, fitting, selection)."""

from pysurrogate.core import Model, Prediction
from pysurrogate.dace import Dace
from pysurrogate.models import Kriging
from pysurrogate.selection import Benchmark, ModelSelection, cartesian

__version__ = "0.1.0"

__all__ = ["Dace", "Kriging", "Model", "Prediction", "Benchmark", "ModelSelection", "cartesian", "__version__"]
