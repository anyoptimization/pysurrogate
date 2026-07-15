"""Shared numeric helpers used across the landscape criterion families."""

import numpy as np


def _safe_float(x):
    """Coerce a value to a finite Python float, mapping non-finite results to ``np.nan``.

    Args:
        x: Any value convertible (or not) to ``float``.

    Returns:
        The value as a plain ``float``, or ``np.nan`` when conversion fails or the result is
        non-finite (the landscape feature contract: no ``inf``, undefined reads as ``nan``).
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _corr(a, b):
    """Pearson correlation of two 1-D arrays; ``nan`` when either has no spread or <3 points.

    Args:
        a: First value array.
        b: Second value array (same length as ``a``).

    Returns:
        The correlation coefficient as a float, or ``np.nan`` when undefined.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return np.nan
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    try:
        return _safe_float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return np.nan


def _gini(x):
    """Gini coefficient of a non-negative array (0 == perfectly equal, ->1 == concentrated).

    Negative inputs are shifted to non-negative before the computation.

    Args:
        x: Values whose concentration is measured.

    Returns:
        The Gini coefficient as a float, or ``np.nan`` when empty or the total vanishes.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        x = x - x.min()
    s = x.sum()
    if s <= 1e-12:
        return np.nan
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return _safe_float((np.sum((2 * idx - n - 1) * xs)) / (n * s))


def _skewness(y):
    """Fisher (moment) skewness of ``y``; ``nan`` when variance is ~zero or fewer than 3 points.

    Args:
        y: A 1-D value array.

    Returns:
        The sample skewness as a float, or ``np.nan`` when undefined.
    """
    n = y.size
    if n < 3:
        return np.nan
    m = np.mean(y)
    s = np.std(y)
    if s <= 1e-12:
        return np.nan
    return _safe_float(np.mean(((y - m) / s) ** 3))


def _adj_r2(r2, n, p):
    """Adjusted R² that deflates an in-sample fit by its parameter count, clipped to ``[0, 1]``.

    Penalizes the raw coefficient of determination for the ``p`` fitted degrees of freedom so a
    richer model whose extra terms only capture noise does not look better than a lean one.

    Args:
        r2: In-sample R² of the fit.
        n: Number of samples.
        p: Number of (non-intercept) fitted parameters.

    Returns:
        The adjusted R² clipped to ``[0, 1]``, or ``np.nan`` when it is not defined.
    """
    if not np.isfinite(r2) or n - p - 1 <= 0:
        return np.nan
    adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - p - 1.0)
    return float(np.clip(adj, 0.0, 1.0))
