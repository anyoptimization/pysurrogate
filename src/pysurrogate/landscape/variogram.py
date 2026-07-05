"""Geostatistical variogram features: spatial smoothness, correlation length, and noise nugget."""

import numpy as np


def _safe_slope_loglog(h, gamma):
    """Least-squares slope of ``log(gamma)`` vs ``log(h)`` over positive points.

    Args:
        h: Distances (bin centers), strictly positive entries are used.
        gamma: Semivariance at each distance.

    Returns:
        The log-log slope as a float, or ``np.nan`` when fewer than two usable points remain.
    """
    try:
        m = (h > 0) & (gamma > 0) & np.isfinite(h) & np.isfinite(gamma)
        if int(np.count_nonzero(m)) < 2:
            return float("nan")
        lh = np.log(h[m])
        lg = np.log(gamma[m])
        if float(np.ptp(lh)) <= 1e-12:
            return float("nan")
        slope = np.polyfit(lh, lg, 1)[0]
        return float(slope)
    except Exception:
        return float("nan")


def _estimate_sill(gamma, ys_var):
    """Estimate the variogram sill (plateau semivariance).

    Args:
        gamma: Semivariance values across bins.
        ys_var: Variance of the standardized outputs (theoretical sill).

    Returns:
        A positive sill estimate as a float, or ``np.nan`` when undefined.
    """
    try:
        g = gamma[np.isfinite(gamma)]
        if g.size == 0:
            return float("nan")
        # blend the empirical plateau (upper-quantile of gamma) with the total variance;
        # the far bins approximate the sill, ys_var (~1 for standardized y) anchors it.
        emp = float(np.quantile(g, 0.9))
        anchor = float(ys_var) if np.isfinite(ys_var) and ys_var > 0 else emp
        sill = max(emp, 0.5 * anchor)
        return sill if sill > 0 else float("nan")
    except Exception:
        return float("nan")


def _nugget(h, gamma):
    """Extrapolate the near-origin variogram to ``h=0`` to estimate the nugget.

    Args:
        h: Bin-center distances (ascending).
        gamma: Semivariance at each bin.

    Returns:
        A non-negative nugget estimate as a float, or ``np.nan`` when undefined.
    """
    try:
        m = np.isfinite(h) & np.isfinite(gamma)
        hh, gg = h[m], gamma[m]
        if hh.size < 2:
            return float("nan")
        order = np.argsort(hh)
        hh, gg = hh[order], gg[order]
        k = min(4, hh.size)
        if float(np.ptp(hh[:k])) <= 1e-12:
            return float("nan")
        intercept = np.polyfit(hh[:k], gg[:k], 1)[1]
        return float(max(intercept, 0.0))
    except Exception:
        return float("nan")


def compute(ctx) -> dict:
    """Compute geostatistical variogram features of a labelled point cloud.

    Fits the empirical semivariogram (semivariance vs pairwise distance on normalized inputs and
    standardized outputs) and extracts smoothness, spatial-correlation, and noise descriptors.

    Args:
        ctx: A landscape ``Context`` exposing ``variogram()``, ``ys``, and the unit-box geometry.

    Returns:
        A flat dict mapping feature names to floats (or ``np.nan`` where undefined):
        ``smoothness_exp``, ``range_rel``, ``nugget_ratio``, ``sill``, ``sill_ratio``,
        ``monotonicity``, ``near_origin_convexity``, ``noise_slope_ratio``.
    """
    keys = [
        "smoothness_exp",
        "range_rel",
        "nugget_ratio",
        "sill",
        "sill_ratio",
        "monotonicity",
        "near_origin_convexity",
        "noise_slope_ratio",
    ]
    out = {k: float("nan") for k in keys}

    try:
        h, gamma = ctx.variogram()
    except Exception:
        return out

    try:
        h = np.asarray(h, dtype=float).ravel()
        gamma = np.asarray(gamma, dtype=float).ravel()
    except Exception:
        return out

    if h.size == 0 or gamma.size == 0 or h.size != gamma.size:
        return out

    # ascending in distance for stable near-origin/plateau reasoning
    order = np.argsort(h)
    h, gamma = h[order], gamma[order]

    try:
        ys_var = float(np.var(ctx.ys))
    except Exception:
        ys_var = float("nan")

    # --- smoothness exponent: near-origin log-log slope (~2 smooth/Gaussian, ~1 rough/exponential)
    k_near = min(4, h.size)
    out["smoothness_exp"] = _safe_slope_loglog(h[:k_near], gamma[:k_near])

    # --- sill (plateau) and its ratio to the total output variance
    sill = _estimate_sill(gamma, ys_var)
    out["sill"] = sill
    if np.isfinite(sill) and np.isfinite(ys_var) and ys_var > 0:
        out["sill_ratio"] = float(sill / ys_var)

    # --- correlation range: distance to reach 95% of sill, relative to the observed distance span
    try:
        if np.isfinite(sill) and sill > 0 and h.size >= 1:
            thresh = 0.95 * sill
            hit = np.where(gamma >= thresh)[0]
            span = float(h[-1]) if h[-1] > 0 else float("nan")
            if hit.size > 0 and np.isfinite(span) and span > 0:
                out["range_rel"] = float(h[hit[0]] / span)
            elif np.isfinite(span) and span > 0:
                # never reaches the sill within the sampled span -> long-range correlation
                out["range_rel"] = 1.0
    except Exception:
        pass

    # --- nugget-to-sill ratio: extrapolated noise floor relative to sill (0 clean, ->1 pure noise)
    nug = _nugget(h, gamma)
    if np.isfinite(nug) and np.isfinite(sill) and sill > 0:
        out["nugget_ratio"] = float(np.clip(nug / sill, 0.0, 1.0))

    # --- monotonicity: fraction of adjacent bins that increase (1 well-behaved, ~0.5 periodic/noisy)
    try:
        if gamma.size >= 2:
            diffs = np.diff(gamma)
            good = np.isfinite(diffs)
            if np.any(good):
                out["monotonicity"] = float(np.mean(diffs[good] > 0))
    except Exception:
        pass

    # --- near-origin convexity: sign of second difference near the origin (>0 Gaussian-like knee,
    #     <0 concave saturation). Normalized to [-1, 1] by the local semivariance scale.
    try:
        if h.size >= 3:
            g3 = gamma[:3]
            if np.all(np.isfinite(g3)):
                second = g3[0] - 2.0 * g3[1] + g3[2]
                scale = float(np.mean(np.abs(g3))) + 1e-12
                out["near_origin_convexity"] = float(np.clip(second / scale, -1.0, 1.0))
    except Exception:
        pass

    # --- noise slope ratio: near-origin secant slope vs overall slope. A large intercept (noise)
    #     flattens the near-origin rise relative to the full range; ratio ~1 means smooth structure.
    try:
        if h.size >= 2 and h[-1] > h[0] and np.isfinite(sill) and sill > 0:
            near_rise = (gamma[min(1, h.size - 1)] - gamma[0]) / max(h[min(1, h.size - 1)] - h[0], 1e-12)
            full_rise = (gamma[-1] - gamma[0]) / max(h[-1] - h[0], 1e-12)
            if np.isfinite(near_rise) and np.isfinite(full_rise) and abs(full_rise) > 1e-12:
                out["noise_slope_ratio"] = float(np.clip(near_rise / full_rise, -5.0, 5.0))
    except Exception:
        pass

    return out
