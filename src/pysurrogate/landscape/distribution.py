"""ELA-style features of the y-value distribution: shape, tails, entropy, and multimodality."""

import numpy as np

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0 and removed; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


def _safe_float(x):
    """Coerce a value to a finite Python float, mapping non-finite results to ``np.nan``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _skewness(y):
    """Fisher (moment) skewness of ``y``; ``nan`` when variance is ~zero."""
    n = y.size
    if n < 3:
        return np.nan
    m = np.mean(y)
    s = np.std(y)
    if s <= 1e-12:
        return np.nan
    return _safe_float(np.mean(((y - m) / s) ** 3))


def _excess_kurtosis(y):
    """Excess kurtosis (normal == 0) of ``y``; ``nan`` when variance is ~zero."""
    n = y.size
    if n < 4:
        return np.nan
    m = np.mean(y)
    s = np.std(y)
    if s <= 1e-12:
        return np.nan
    return _safe_float(np.mean(((y - m) / s) ** 4) - 3.0)


def _kde_grid(y, gridsize=256):
    """Evaluate a Gaussian KDE of ``y`` on a fine grid; returns ``(grid, density, bandwidth)``.

    Returns ``(None, None, None)`` when a KDE is not well defined (too few points or no spread).
    """
    n = y.size
    if n < 5:
        return None, None, None
    s = np.std(y)
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    scale = min(s, iqr / 1.349) if iqr > 0 else s
    if scale <= 1e-12:
        return None, None, None
    # Silverman's rule of thumb bandwidth.
    bw = 0.9 * scale * n ** (-1.0 / 5.0)
    if bw <= 1e-12:
        return None, None, None
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = 3.0 * bw
    grid = np.linspace(lo - pad, hi + pad, gridsize)
    # Vectorized Gaussian mixture density.
    u = (grid[:, None] - y[None, :]) / bw
    dens = np.exp(-0.5 * u * u).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))
    return grid, dens, bw


def _n_modes(grid, dens):
    """Count interior local maxima of a density curve using a prominence threshold."""
    if grid is None or dens is None or dens.size < 3:
        return np.nan
    peak = float(np.max(dens))
    if peak <= 0:
        return np.nan
    thresh = 0.05 * peak  # ignore tiny ripples in the tails
    count = 0
    for i in range(1, dens.size - 1):
        if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1] and dens[i] >= thresh:
            count += 1
    return float(count) if count > 0 else 1.0


def _differential_entropy(grid, dens):
    """Differential (continuous) entropy of a KDE density via trapezoidal integration."""
    if grid is None or dens is None:
        return np.nan
    p = np.clip(dens, 1e-12, None)
    integrand = -p * np.log(p)
    return _safe_float(_trapezoid(integrand, grid))


def _tail_index(y):
    """Heavy-tail asymmetry index ``(p99 - p50) / (p50 - p1)`` on the raw ``y``."""
    try:
        p1, p50, p99 = np.percentile(y, [1, 50, 99])
    except Exception:
        return np.nan
    upper = p99 - p50
    lower = p50 - p1
    if lower <= 1e-12:
        return np.nan
    return _safe_float(upper / lower)


def compute(ctx) -> dict:
    """Structural features of the y-value distribution, ignoring input positions.

    Args:
        ctx: A landscape :class:`Context` wrapping the labelled cloud ``(X, y)``.

    Returns:
        A flat dict of float features describing the shape (skewness, excess kurtosis),
        spread (dynamic range, coefficient of variation, IQR-to-range ratio), tails
        (heavy-tail index, tail ratio), information content (differential entropy), and
        multimodality (KDE peak count, peak-mass concentration) of the objective values.
    """
    keys = [
        "skewness",
        "excess_kurtosis",
        "diff_entropy",
        "n_modes",
        "tail_index",
        "dynamic_range",
        "coef_variation",
        "iqr_range_ratio",
        "peak_concentration",
        "median_skew",
    ]
    out = {k: np.nan for k in keys}

    try:
        y = np.asarray(ctx.y, dtype=float).ravel()
        y = y[np.isfinite(y)]
    except Exception:
        return out

    n = y.size
    if n == 0:
        return out

    # Shape of the distribution.
    out["skewness"] = _skewness(y)
    out["excess_kurtosis"] = _excess_kurtosis(y)

    # KDE-based features (entropy + multimodality).
    try:
        grid, dens, _ = _kde_grid(y)
        out["diff_entropy"] = _differential_entropy(grid, dens)
        out["n_modes"] = _n_modes(grid, dens)
        # Fraction of probability mass within one bandwidth of the tallest peak: high => a
        # single dominant basin of y-values, low => mass spread across a plateau/multimodal.
        if grid is not None and dens is not None:
            total = _trapezoid(dens, grid)
            if total > 1e-12:
                pk = int(np.argmax(dens))
                span = grid[-1] - grid[0]
                half = 0.05 * span
                mask = np.abs(grid - grid[pk]) <= half
                out["peak_concentration"] = _safe_float(_trapezoid(dens[mask], grid[mask]) / total)
    except Exception:
        pass

    # Tail heaviness / asymmetry.
    out["tail_index"] = _tail_index(y)

    # Spread / dynamic range.
    try:
        ymin, ymax = float(np.min(y)), float(np.max(y))
        rng = ymax - ymin
        out["dynamic_range"] = _safe_float(rng)
        mean_abs = abs(float(np.mean(y)))
        s = float(np.std(y))
        if mean_abs > 1e-12:
            out["coef_variation"] = _safe_float(s / mean_abs)
        if rng > 1e-12:
            iqr = float(np.subtract(*np.percentile(y, [75, 25])))
            out["iqr_range_ratio"] = _safe_float(iqr / rng)
            # Nonparametric skew: (mean - median) / range, robust and bounded.
            out["median_skew"] = _safe_float((float(np.mean(y)) - float(np.median(y))) / rng)
    except Exception:
        pass

    return {k: _safe_float(v) for k, v in out.items()}
