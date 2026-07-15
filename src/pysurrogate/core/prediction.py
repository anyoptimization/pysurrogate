"""The single, backend-agnostic result type for every surrogate prediction."""

from dataclasses import dataclass

import numpy as np


def sigma_from_var(var):
    """Predictive standard deviation ``sqrt(max(var, 0))`` -- the one clamp-then-root of variance.

    Clamps the variance non-negative before the square root so round-off (or a backend returning a
    tiny negative kriging MSE) yields ``0`` rather than a ``NaN``.

    Args:
        var: Predictive variance array.

    Returns:
        ``sqrt(clip(var, 0, None))``, same shape as ``var``.
    """
    return np.sqrt(np.clip(var, 0.0, None))


@dataclass(frozen=True)
class Prediction:
    """The result of any ``predict`` call: the mean plus any requested extras, named.

    Every backend in pysurrogate -- Kriging (``Dace``), RBF, SVR, ... -- returns this one
    type, so callers read fields instead of guessing tuple positions and a Kriging result
    and an RBF result share a single interface. ``y`` is always populated; ``var``, ``grad``
    and ``var_grad`` are ``None`` unless the flags that produce them were set on the call.

    The canonical name for the predictive uncertainty is ``var`` (the predictive variance),
    because that term is meaningful for every surrogate. The Kriging/DACE literature calls
    the same quantity the *mean squared error* -- so ``mse`` and ``mse_grad`` are provided as
    read-only aliases of ``var`` and ``var_grad`` for that audience (and for code written
    against the original DACE engine).

    Attributes:
        y: Predicted mean, shape ``(m, q)``.
        var: Predictive variance, shape ``(m, 1)``, or ``None`` when not requested. Shared
            across outputs for a multi-output model, so it stays ``(m, 1)``.
        grad: Gradient of the mean w.r.t. the query point, or ``None``. ``(m, d)`` for a
            single-output model; ``(m, q, d)`` (one gradient per output) for multi-output.
        var_grad: Gradient of the predictive variance w.r.t. the query point, ``(m, d)``, or
            ``None``. Populated only when both ``var`` and ``grad`` were requested and the
            backend supports it (currently Kriging). Lets a caller form ``grad(std)`` as
            ``var_grad / (2*sqrt(var))`` -- e.g. for gradient-based Expected Improvement.
    """

    y: np.ndarray
    var: np.ndarray | None = None
    grad: np.ndarray | None = None
    var_grad: np.ndarray | None = None

    @property
    def sigma(self) -> np.ndarray | None:
        """Predictive standard deviation ``sqrt(var)`` (clamped non-negative), or ``None``.

        Derived from ``var`` so calibration metrics and callers that think in std-dev keep a
        single accessor; the stored quantity is the variance.

        Returns:
            ``sqrt(max(var, 0))`` with the same shape as ``var``, or ``None`` when ``var`` was
            not requested.
        """
        if self.var is None:
            return None
        return sigma_from_var(self.var)

    @property
    def mse(self) -> np.ndarray | None:
        """DACE-literature alias of ``var`` (the kriging mean squared error)."""
        return self.var

    @property
    def mse_grad(self) -> np.ndarray | None:
        """DACE-literature alias of ``var_grad`` (gradient of the kriging MSE)."""
        return self.var_grad


def predictions_frame(X, y_true, pred, **labels):
    """Build the tidy per-point predictions DataFrame from a prediction -- the common schema.

    One row per predicted point (per output): the given ``**labels`` columns first (e.g.
    ``model=...``, ``role=...``, ``epoch=...``), then ``i`` (point index), ``output``, ``y_true``,
    ``y`` (prediction), ``var``, ``sigma`` (NaN when the model reports no uncertainty), and the
    input coordinates ``x0..xd``. This is the schema the benchmark layer emits and ``score``
    consumes, and the one a :meth:`~pysurrogate.core.model.Model.refit` loop accumulates, so every
    source of predictions speaks one DataFrame format.

    Multi-output (``q > 1``) gives one row per ``(point, output)``. The predictive ``var``/``sigma``
    is taken as *shared across outputs* -- the kriging variance is one value per point, repeated
    for every output -- so per-output probabilistic metrics use that shared sigma against each
    output's residual. This assumes a shared-variance model (as all current backends are); a
    backend reporting per-output variance would need a different layout.

    Args:
        X: Inputs that were predicted, shape ``(m, d)``.
        y_true: True targets for ``X``, shape ``(m,)`` or ``(m, q)``.
        pred: A :class:`Prediction` for ``X``.
        **labels: Constant label columns to prepend (broadcast over the rows).

    Returns:
        The long predictions DataFrame.
    """
    import pandas as pd  # type: ignore[import-untyped]

    Xa = np.atleast_2d(X)
    m = Xa.shape[0]
    y_hat = np.asarray(pred.y).reshape(m, -1)  # (m, q)
    yt = np.asarray(y_true).reshape(m, -1)  # (m, q)
    q = y_hat.shape[1]
    # variance/sigma are shared across outputs; NaN when the model reports no uncertainty
    var = np.full(m, np.nan) if pred.var is None else np.asarray(pred.var).ravel()
    sigma = sigma_from_var(var)

    df = pd.DataFrame(
        {
            **labels,
            "i": np.tile(np.arange(m), q),
            "output": np.repeat(np.arange(q), m),
            "y_true": yt.T.ravel(),  # output-major to match i/output tiling
            "y": y_hat.T.ravel(),
            "var": np.tile(var, q),
            "sigma": np.tile(sigma, q),
        }
    )
    for j in range(Xa.shape[1]):  # input coordinates always stored -- to reproduce errors later
        df[f"x{j}"] = np.tile(Xa[:, j], q)
    return df
