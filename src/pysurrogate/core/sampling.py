"""Sampling: generate starting points in a box, with guaranteed-included points."""

from abc import ABC, abstractmethod

import numpy as np


class SamplingMethod(ABC):
    """How to fill the unit hypercube ``[0, 1]^p`` with ``n`` points."""

    @abstractmethod
    def __call__(self, n, p, rng):
        """Return ``(n, p)`` points in the unit cube, drawn with ``rng``."""


class Random(SamplingMethod):
    """Uniform random sampling in the unit cube."""

    def __call__(self, n, p, rng):
        return rng.random((n, p))


class LHS(SamplingMethod):
    """Latin hypercube: one stratified, jittered point per stratum per dimension."""

    def __call__(self, n, p, rng):
        unit = np.empty((n, p))
        edges = np.linspace(0, 1, n + 1)
        for j in range(p):
            unit[:, j] = rng.permutation(edges[:n] + rng.random(n) * (edges[1] - edges[0]))
        return unit


class Sampling:
    """Generate ``n`` starting points in a box, always including the forced points.

    The single start-generation strategy any optimizer uses to seed itself, so the logic lives
    in one place rather than duplicated across optimizers. Forced points -- a warm ``x0`` or the
    previous iteration's optimum -- are *guaranteed* to be in the returned set and count toward
    ``n``, which is why a known ``x0`` is not a special case: it is just one more guaranteed
    sample. The remaining points fill the box by ``method`` (Latin hypercube by default). Forced
    points come from two sources, merged: ``include`` set at construction (e.g. prior optima the
    user knows) and ``include`` passed to :meth:`sample` at runtime (the ``x0`` the optimizer
    injects, which the user did not have).

    Args:
        n: Total number of points to return (clamped up to the number of forced points).
        method: How to fill the box -- :class:`LHS` (default) or :class:`Random`.
        include: Points (each shape ``(p,)``) that must appear in every sample.
    """

    def __init__(self, n, method=None, include=None):
        self.n = n
        self.method = method if method is not None else LHS()
        self.include = [np.atleast_1d(np.asarray(x, float)) for x in (include or [])]

    def sample(self, bounds, rng=None, include=None):
        """Draw the start points inside ``bounds``.

        Args:
            bounds: ``(lo, hi)`` box, each shape ``(p,)``.
            rng: A ``numpy`` Generator for reproducibility; ``None`` makes a fresh one.
            include: Extra forced points for this call (e.g. the optimizer's ``x0``), merged with
                the construction-time ``include``.

        Returns:
            An array of start points, shape ``(max(n, #forced), p)``, every forced point present
            (clipped into the box) followed by the space-filling draws.
        """
        lo, hi = (np.atleast_1d(np.asarray(b, float)) for b in bounds)
        p = len(lo)
        rng = np.random.default_rng() if rng is None else rng

        forced = [
            np.clip(x, lo, hi) for x in (self.include + [np.atleast_1d(np.asarray(x, float)) for x in (include or [])])
        ]
        n_fill = max(self.n - len(forced), 0)
        pts = list(forced)
        if n_fill > 0:
            pts.extend(lo + self.method(n_fill, p, rng) * (hi - lo))
        # keep the (k, p) shape even when there is nothing to return, so callers can index [:, j]
        return np.array(pts) if pts else np.empty((0, p))
