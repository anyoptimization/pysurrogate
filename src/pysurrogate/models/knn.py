"""k-nearest-neighbors surrogate model with inverse-distance weighting."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.util.dist import calc_dist


class KNN(Model):
    """Predicts each point as the inverse-distance-weighted mean of its ``n_nearest`` neighbors."""

    def __init__(self, n_nearest=10, p=2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_nearest = n_nearest
        self.p = p

    def _fit(self, X, y, **kwargs):
        pass

    def _predict(self, X, var=False, grad=False):
        D = calc_dist(X, self.X)

        idx = D.argsort(axis=1)[:, : self.n_nearest]

        d = np.take_along_axis(D, idx, axis=1) ** self.p
        d[d == 0] = 1e-64
        w = 1 / d
        w = w / w.sum(axis=1)[:, None]

        neighbors = np.take_along_axis(self.y, idx, axis=0)
        y = (w * neighbors).sum(axis=1)

        # local uncertainty = the inverse-distance-weighted variance of the k neighbor targets
        # about the weighted mean, reusing the same weights. grad/var_grad stay None.
        v = None
        if var:
            v = (w * (neighbors - y[:, None]) ** 2).sum(axis=1)[:, None]

        return Prediction(y=y[:, None], var=v)
