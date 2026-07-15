"""Inverse distance weighting (IDW / Shepard) surrogate model."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.util.dist import euclidean_dist


class InverseDistanceWeighting(Model):
    """Shepard interpolation: each point is a distance**(-p)-weighted mean of all training targets."""

    def __init__(self, p=3.0, eps=1e-32, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p = p
        self.eps = eps

    def _fit(self, X, y, **kwargs):
        if y.shape[1] != 1:
            raise ValueError(
                f"InverseDistanceWeighting supports a single output, got {y.shape[1]}; fit one model per output."
            )

    def _predict(self, X, var=False, grad=False):
        _y = self.y[:, 0]

        D = euclidean_dist(X, self.X)
        D[D <= self.eps] = self.eps

        w = 1 / D**self.p

        # at an exact data hit, pin the surface to that target (weight 1 there, 0 elsewhere)
        is_zero = D <= self.eps  # (m, n)
        hit = is_zero.any(axis=1)
        w[hit] = is_zero[hit].astype(float)

        w = w / w.sum(axis=1)[:, None]
        y = (_y * w).sum(axis=1)

        g = None
        if grad:
            # Shepard gradient: d/dx of (sum_i w_i y_i / sum_i w_i) with w_i = D_i**-p and
            # dD_i/dx = (x - x_i)/D_i, so dw_i/dx = -p D_i**(-p-2) (x - x_i). It collapses to
            # (1/W) sum_i (y_i - yhat) dw_i/dx. At an exact data hit the surface is pinned, so
            # the gradient there is set to 0 (consistent with the hit branch above).
            diff = X[:, None, :] - self.X[None, :, :]
            raw = D ** (-self.p)
            coef = -self.p * D ** (-self.p - 2.0)
            resid = _y[None, :] - y[:, None]
            g = np.einsum("mn,mnd->md", coef * resid, diff) / raw.sum(axis=1)[:, None]
            g[hit] = 0.0

        return Prediction(y=y[:, None], grad=g)
