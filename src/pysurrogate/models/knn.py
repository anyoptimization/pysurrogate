"""k-nearest-neighbors surrogate model with inverse-distance weighting."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.util.dist import calc_dist


class KNN(Model):
    """Predicts each point as a distance-weighted mean of its ``n_nearest`` neighbors.

    Weights are ``1 / d**p`` where ``d`` is the **squared** Euclidean distance (the model layer's
    :func:`~pysurrogate.util.dist.calc_dist`). So the effective exponent on the *true* distance is
    ``2p`` -- note this differs from :class:`~pysurrogate.models.idw.InverseDistanceWeighting`,
    whose ``p`` is the exponent on the true (un-squared) distance.

    Args:
        n_nearest: Number of nearest neighbors averaged for each query point.
        p: Exponent applied to the squared distance in the inverse-distance weights.
    """

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

        # neighbors: (m, k, q) via fancy indexing -- works for any number of outputs q (a plain
        # take_along_axis only lined up for q == 1). Weights broadcast over the output axis.
        neighbors = self.y[idx]  # (m, k, q)
        wk = w[:, :, None]  # (m, k, 1)
        y = (wk * neighbors).sum(axis=1)  # (m, q)

        # local uncertainty = the inverse-distance-weighted variance of the k neighbor targets
        # about the weighted mean, reusing the same weights. Collapsed to one shared value per
        # point (mean over outputs) to match the shared-variance Prediction contract. grad stays None.
        v = None
        if var:
            per_output = (wk * (neighbors - y[:, None, :]) ** 2).sum(axis=1)  # (m, q)
            v = per_output.mean(axis=1, keepdims=True)  # (m, 1)

        return Prediction(y=y, var=v)
