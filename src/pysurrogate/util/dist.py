"""Euclidean distance matrices between two sets of points."""

import numpy as np
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]


def calc_dist(X, Z):
    """Squared euclidean distance matrix between the rows of ``X`` and ``Z``.

    The *squared* distance (no final square root) is what the RBF and KNN backends use as
    their kernel argument, so it is the default distance throughout the model layer.

    Args:
        X: Query points, shape ``(m, d)``.
        Z: Reference points, shape ``(n, d)``.

    Returns:
        The matrix ``D`` with ``D[i, j] = ||X[i] - Z[j]||**2``, shape ``(m, n)``.
    """
    return np.asarray(cdist(X, Z, "sqeuclidean"))


def euclidean_dist(X, Z):
    """Euclidean (un-squared) distance matrix between the rows of ``X`` and ``Z``.

    Args:
        X: Query points, shape ``(m, d)``.
        Z: Reference points, shape ``(n, d)``.

    Returns:
        The matrix ``D`` with ``D[i, j] = ||X[i] - Z[j]||``, shape ``(m, n)``.
    """
    return np.asarray(cdist(X, Z))
