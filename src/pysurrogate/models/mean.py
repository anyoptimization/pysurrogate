"""Constant-mean baseline surrogate model."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction


class SimpleMean(Model):
    """Baseline that predicts the training-target mean everywhere (a reference for selection)."""

    def _fit(self, X, y, **kwargs):
        self.model = np.mean(y, axis=0)

    def _predict(self, X, var=False, grad=False):
        return Prediction(y=np.full((len(X), 1), self.model))
