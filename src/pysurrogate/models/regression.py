"""Polynomial regression surrogate model with an analytic gradient."""

import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # type: ignore[import-untyped]

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction


class PolynomialRegression(Model):
    """Least-squares polynomial fit of the given ``degree`` (poly-features -> scaler -> linear)."""

    def __init__(self, degree=2, fail_if_not_enough_points=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.degree = degree
        self.fail_if_not_enough_points = fail_if_not_enough_points

    def _fit(self, X, y, **kwargs):
        n_min_points = PolynomialFeatures(self.degree).fit_transform(X).shape[1]

        if self.fail_if_not_enough_points and len(X) < n_min_points:
            raise ValueError(
                f"Polynomial regression of degree {self.degree} needs at least {n_min_points} points, got {len(X)}."
            )

        model = make_pipeline(PolynomialFeatures(self.degree), StandardScaler(), LinearRegression())
        model.fit(X, y[:, 0])
        self.model = model

    def _predict(self, X, var=False, grad=False):
        g = _poly_grad(self.model, X) if grad else None
        return Prediction(y=self.model.predict(X)[:, None], grad=g)


def _poly_grad(pipeline, X):
    """Analytic gradient of the (poly-features -> scaler -> linear) pipeline, shape ``(m, d)``.

    Each polynomial feature is a monomial ``prod_i x_i**p_i`` (powers from
    ``PolynomialFeatures.powers_``); its derivative w.r.t. ``x_j`` lowers that exponent by one
    and multiplies by the old exponent. The scaler and linear layer are affine, so they fold in
    as the constant factor ``coef_k / scale_k``.
    """
    poly = pipeline.named_steps["polynomialfeatures"]
    scaler = pipeline.named_steps["standardscaler"]
    linreg = pipeline.named_steps["linearregression"]

    powers = poly.powers_  # (F, d) exponent of each input in each feature
    w_over_scale = linreg.coef_ / scaler.scale_  # (F,) dy/d(feature)

    m, d = X.shape
    grad = np.zeros((m, d))
    for j in range(d):
        exponent = powers[:, j].astype(float)  # (F,)
        reduced = powers.copy()
        reduced[:, j] = np.maximum(powers[:, j] - 1, 0)
        mono = np.prod(X[:, None, :] ** reduced[None, :, :], axis=2)  # (m, F)
        grad[:, j] = (mono * (exponent * w_over_scale)[None, :]).sum(axis=1)
    return grad
