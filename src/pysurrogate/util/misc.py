"""Miscellaneous array helpers shared across the surrogate lifecycle."""

import numpy as np

from pysurrogate.util.dist import euclidean_dist


def at_least2d(x, expand="c"):
    """Promote a 1-D array to 2-D, leaving 2-D (or higher) input untouched.

    Args:
        x: The array to promote.
        expand: ``"c"`` to add a trailing column axis (``(n,) -> (n, 1)``), ``"r"`` to add a
            leading row axis (``(n,) -> (1, n)``).

    Returns:
        ``x`` with the requested axis added when it was 1-D, otherwise ``x`` unchanged.
    """
    if x.ndim == 1:
        if expand == "c":
            return x[:, None]
        elif expand == "r":
            return x[None, :]
    return x


def is_duplicate(X, eps=1e-16):
    """Boolean mask marking each row of ``X`` that duplicates an earlier row.

    The first occurrence of a point is kept (marked ``False``); every later row within
    ``eps`` of an earlier one is marked ``True``. Used to drop repeated design points before
    a Kriging fit, where they make the correlation matrix singular.

    Args:
        X: Points, shape ``(n, d)``.
        eps: Euclidean distance below which two rows count as the same point.

    Returns:
        A length-``n`` boolean array, ``True`` for rows that repeat an earlier row.
    """
    D = euclidean_dist(X, X)
    D[np.triu_indices(len(X))] = np.inf
    return np.any(D < eps, axis=1)


def discretize(X, n_partitions, xl=None, xu=None):
    """Bin each coordinate of ``X`` into one of ``n_partitions`` equal-width bins per dimension.

    Used by the random-forest backend to collapse a continuous design space onto a grid.

    Args:
        X: Points to discretize, shape ``(n, d)``.
        n_partitions: Number of bins per dimension.
        xl: Per-dimension lower bound, or ``None`` to take the column minima of ``X``.
        xu: Per-dimension upper bound, or ``None`` to take the column maxima of ``X``.

    Returns:
        Integer bin indices in ``[0, n_partitions)``, same shape as ``X``. Values below ``xl``
        (or above ``xu``) clamp to the first (or last) bin; a zero-range dimension
        (``xl == xu``) maps everything to bin 0.
    """
    X = np.asarray(X, dtype=float)
    xl = X.min(axis=0) if xl is None else np.asarray(xl, dtype=float)
    xu = X.max(axis=0) if xu is None else np.asarray(xu, dtype=float)

    span = xu - xl
    span = np.where(span > 0, span, 1.0)  # zero-range dim: avoid 0/0, everything lands in bin 0
    bins = np.floor((X - xl) / span * n_partitions).astype(int)
    return np.clip(bins, 0, n_partitions - 1)
