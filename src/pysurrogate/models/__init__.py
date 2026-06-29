"""Model backends built on the core fit/predict lifecycle."""

from pysurrogate.models.forest import RandomForest
from pysurrogate.models.idw import InverseDistanceWeighting
from pysurrogate.models.knn import KNN
from pysurrogate.models.kriging import Kriging
from pysurrogate.models.mean import SimpleMean
from pysurrogate.models.rbf import RBF
from pysurrogate.models.regression import PolynomialRegression
from pysurrogate.models.svr import SVR

__all__ = [
    "Kriging",
    "RBF",
    "SVR",
    "KNN",
    "InverseDistanceWeighting",
    "SimpleMean",
    "PolynomialRegression",
    "RandomForest",
]
