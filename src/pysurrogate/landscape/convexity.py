"""Convexity-probing features: chord-vs-midpoint tests over random point pairs (ELA convexity)."""

import numpy as np


def _sample_pairs(ctx, n_pairs):
    """Draw random distinct index pairs ``(a, b)`` from the cloud.

    Args:
        ctx: The shared context (uses ``ctx.rng`` and ``ctx.n``).
        n_pairs: Desired number of pairs to draw.

    Returns:
        ``(a, b)`` two integer arrays of equal length with ``a != b`` element-wise, or two empty
        arrays when fewer than two points exist.
    """
    n = ctx.n
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    a = ctx.rng.integers(0, n, size=n_pairs)
    b = ctx.rng.integers(0, n, size=n_pairs)
    ok = a != b
    return a[ok], b[ok]


def compute(ctx) -> dict:
    """Convexity-probing landscape features from midpoint-vs-chord comparisons.

    Samples many random pairs of sample points. For each pair the geometric midpoint (in the
    normalized input space ``Xn``) is approximated by the nearest actual sample point, and its
    objective value is compared against the linear interpolation (the chord) of the two
    endpoints. A midpoint that sits *below* the chord is locally convex; *above* is concave. The
    features aggregate the sign and magnitude of that convexity gap across pairs, giving a global
    read on whether the landscape behaves like a convex bowl, a concave dome, or a rugged,
    multimodal surface where both appear. Gaps are measured in standardized-``y`` units so they
    are scale-free.

    Args:
        ctx: The shared :class:`Context` wrapping one labelled point cloud.

    Returns:
        A flat dict of convexity features keyed by short names, each a float or ``np.nan``.
    """
    keys = [
        "convex_frac",
        "concave_frac",
        "linear_frac",
        "net_convexity",
        "mean_gap",
        "median_gap",
        "gap_std",
        "convex_intensity",
        "concave_intensity",
        "gap_skew",
        "midpoint_approx_error",
        "usable_frac",
    ]
    out = {k: np.nan for k in keys}

    try:
        n = ctx.n
        if n < 3 or not np.isfinite(ctx.ys).all():
            return out

        # Constant landscape: perfectly linear everywhere, no convex/concave signal.
        if float(np.std(ctx.y)) <= 0.0:
            out.update(
                convex_frac=0.0,
                concave_frac=0.0,
                linear_frac=1.0,
                net_convexity=0.0,
                mean_gap=0.0,
                median_gap=0.0,
                gap_std=0.0,
                convex_intensity=0.0,
                concave_intensity=0.0,
                gap_skew=0.0,
                midpoint_approx_error=np.nan,
                usable_frac=0.0,
            )
            return out

        Xn, ys = ctx.Xn, ctx.ys
        n_pairs = int(np.clip(30 * n, 200, 4000))
        a, b = _sample_pairs(ctx, n_pairs)
        if a.size == 0:
            return out

        # Geometric midpoints, and the nearest actual sample to each (its value approximates the
        # true midpoint value). Distances are computed against every sample point at once.
        mids = 0.5 * (Xn[a] + Xn[b])
        # squared distances mid -> all points: (m, n)
        d2 = np.sum(mids**2, axis=1, keepdims=True) - 2.0 * mids @ Xn.T + np.sum(Xn**2, axis=1)
        nearest = np.argmin(d2, axis=1)
        approx_dist = np.sqrt(np.maximum(d2[np.arange(a.size), nearest], 0.0))

        pair_dist = np.linalg.norm(Xn[a] - Xn[b], axis=1)

        # Usable pairs: the nearest sample to the midpoint must be a genuine interior probe, not
        # one of the endpoints themselves (which would make the test degenerate).
        usable = (nearest != a) & (nearest != b)
        usable_frac = float(np.mean(usable))
        if not np.any(usable):
            out["usable_frac"] = usable_frac
            return out

        a, b, nearest = a[usable], b[usable], nearest[usable]
        approx_dist, pair_dist = approx_dist[usable], pair_dist[usable]

        chord = 0.5 * (ys[a] + ys[b])
        f_mid = ys[nearest]
        gap = chord - f_mid  # > 0 : midpoint below chord -> convex

        # Threshold scaled by the local spread of the pair so tiny numerical wiggles count as
        # "linear" rather than as weak convexity/concavity.
        scale = np.maximum(np.abs(ys[a] - ys[b]), 1e-9)
        eps = 1e-3 * scale
        convex = gap > eps
        concave = gap < -eps
        linear = ~(convex | concave)

        convex_frac = float(np.mean(convex))
        concave_frac = float(np.mean(concave))
        linear_frac = float(np.mean(linear))

        pos = gap[gap > 0]
        neg = gap[gap < 0]
        g_std = float(np.std(gap))
        # Sample skewness of the gap distribution (rugged surfaces have heavy two-sided tails;
        # a clean bowl is right-skewed toward convexity).
        if g_std > 1e-12:
            gap_skew = float(np.mean(((gap - np.mean(gap)) / g_std) ** 3))
        else:
            gap_skew = 0.0

        # Midpoint-approximation error relative to the pair separation: how well a real sample
        # actually stands in for the geometric midpoint (low = the probe is trustworthy).
        rel_err = approx_dist / np.maximum(pair_dist, 1e-9)

        out.update(
            convex_frac=convex_frac,
            concave_frac=concave_frac,
            linear_frac=linear_frac,
            net_convexity=convex_frac - concave_frac,
            mean_gap=float(np.mean(gap)),
            median_gap=float(np.median(gap)),
            gap_std=g_std,
            convex_intensity=float(np.mean(pos)) if pos.size else 0.0,
            concave_intensity=float(np.mean(-neg)) if neg.size else 0.0,
            gap_skew=gap_skew,
            midpoint_approx_error=float(np.median(rel_err)),
            usable_frac=usable_frac,
        )
        return out
    except Exception:
        return out
