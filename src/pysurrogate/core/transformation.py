"""Invertible input/output transformations used by the model lifecycle (normalization, plog)."""

from abc import ABC, abstractmethod

import numpy as np


class Transformation(ABC):
    """Invertible map applied to inputs or outputs before/after fitting.

    A transform normalizes data on the way into a backend (``forward``) and un-normalizes
    predictions on the way out (``backward``). Affine transforms also expose ``scale`` -- the
    constant Jacobian diagonal -- so a prediction's ``var``/``grad`` can be carried back to
    original units through the chain rule without the query point.
    """

    @abstractmethod
    def forward(self, X):
        """Map data into the transformed space."""

    @abstractmethod
    def backward(self, X):
        """Map data back to the original space (inverse of ``forward``)."""

    def reset(self):
        """Drop any statistics estimated from data, restoring the as-constructed state.

        A data-fitted transform (e.g. :class:`Standardization`) caches its statistics on the first
        ``forward`` and would otherwise reuse them forever -- so a re-``fit`` on grown or different
        data (the model ``refit`` lifecycle) would normalize with stale statistics. ``Model.fit``
        calls this at the start of every fresh fit; the identity/affine transforms with no estimated
        state override it as a no-op (the base is already a no-op).
        """

    def scale(self):
        """Per-dimension affine scale of ``backward`` (the constant Jacobian diagonal).

        Used to un-normalize a prediction's ``var``/``grad``/``var_grad`` through the chain
        rule: for an affine ``backward(z) = s * z + offset`` the factor ``s`` is constant, so
        it carries normalized outputs back to original units without the query point.

        Returns:
            The multiplicative factor ``s``, broadcastable against the per-dimension axis.

        Raises:
            NotImplementedError: If the transform is not affine and thus has no constant scale
                (un-normalizing variance/gradients through it would need a per-point Jacobian,
                which is not supported).
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not affine and exposes no constant scale; predictive "
            "var/grad cannot be un-normalized through it."
        )


class NoNormalization(Transformation):
    """Identity transform: leaves data untouched (the default for a ``Model``)."""

    def forward(self, X):
        return X

    def backward(self, X):
        return X

    def scale(self):
        return 1.0


class Standardization(Transformation):
    """Zero-mean / unit-variance standardization, fit from the data on first ``forward``."""

    def __init__(self, mean=None, std=None) -> None:
        # remember the as-constructed values so reset() can distinguish user-provided statistics
        # (kept) from data-estimated ones (dropped and re-estimated on the next fit).
        self._mean0, self._std0 = mean, std
        self.mean = mean
        self.std = std

    def reset(self):
        self.mean, self.std = self._mean0, self._std0

    def forward(self, X):
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
        if self.std is None:
            self.std = np.std(X, axis=0)
        # a constant dimension has std 0; map it to scale 1 (matching ZeroToOneNormalization and
        # the Dace fit) so forward centers it to 0 instead of dividing by zero into NaN/inf
        self.std = np.where(np.asarray(self.std) == 0.0, 1.0, self.std)
        return (X - self.mean) / self.std

    def backward(self, X):
        return (X * self.std) + self.mean

    def scale(self):
        return self.std


class ZeroToOneNormalization(Transformation):
    """Min-max normalization to ``[0, 1]``, with bounds estimated from the data by default."""

    def __init__(self, xl=None, xu=None, estimate_bounds=True) -> None:
        self._xl0, self._xu0 = xl, xu  # as-constructed bounds, restored by reset()
        self.xl = xl
        self.xu = xu
        self.estimate_bounds = estimate_bounds

    def reset(self):
        self.xl, self.xu = self._xl0, self._xu0

    def forward(self, X):
        if self.estimate_bounds:
            if self.xl is None:
                self.xl = np.min(X, axis=0)
            if self.xu is None:
                self.xu = np.max(X, axis=0)

        denom = self.xu - self.xl
        # avoid divide-by-zero on a constant dimension
        denom = denom + (denom == 0) * 1e-32
        return (X - self.xl) / denom

    def backward(self, X):
        return X * (self.xu - self.xl) + self.xl

    def scale(self):
        return self.xu - self.xl


class Plog(Transformation):
    """Signed-log transform ``sign(y) * log(1 + |y|)`` for heavy-tailed outputs.

    Non-affine, so it has no constant ``scale``: requesting predictive variance/gradient on
    a ``Plog``-transformed output raises (inherited ``Transformation.scale``).
    """

    def forward(self, y):
        yp = np.zeros_like(y, dtype=float)
        larger = y >= 0
        yp[larger] = np.log(1 + y[larger])
        yp[~larger] = -np.log(1 - y[~larger])
        return yp

    def backward(self, yp):
        y = np.zeros_like(yp, dtype=float)
        larger = yp >= 0
        y[larger] = np.exp(yp[larger]) - 1
        y[~larger] = 1 - np.exp(-yp[~larger])
        return y
