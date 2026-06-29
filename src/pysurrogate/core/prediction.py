"""The single, backend-agnostic result type for every surrogate prediction."""

from dataclasses import dataclass

import numpy as np


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
        return np.sqrt(np.clip(self.var, 0.0, None))

    @property
    def mse(self) -> np.ndarray | None:
        """DACE-literature alias of ``var`` (the kriging mean squared error)."""
        return self.var

    @property
    def mse_grad(self) -> np.ndarray | None:
        """DACE-literature alias of ``var_grad`` (gradient of the kriging MSE)."""
        return self.var_grad
