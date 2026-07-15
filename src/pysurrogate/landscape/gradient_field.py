"""Gradient-field / Lipschitz features: slope bounds, coherence, and curvature of the field."""

import numpy as np

_TINY = 1e-12


def _pair_slopes(ctx):
    """Per-pair secant slopes ``|Δys| / |Δx|`` over all distinct point pairs (on ``Xn``, ``ys``).

    Args:
        ctx: The shared context (uses its cached distance matrix and standardized outputs).

    Returns:
        A 1-D array of finite slopes for pairs separated by a non-negligible distance, or an empty
        array when no such pair exists.
    """
    try:
        D = ctx.distances()
        iu = np.triu_indices(ctx.n, k=1)
        h = D[iu]
        dy = np.abs(ctx.ys[iu[0]] - ctx.ys[iu[1]])
        m = h > _TINY
        if not np.any(m):
            return np.array([])
        s = dy[m] / h[m]
        return s[np.isfinite(s)]
    except Exception:
        return np.array([])


def _grad_norms(G):
    """Row-wise Euclidean norms of a gradient array.

    Args:
        G: Per-point gradients, shape ``(n, d)``.

    Returns:
        The ``(n,)`` array of gradient magnitudes.
    """
    return np.sqrt(np.sum(np.asarray(G, dtype=float) ** 2, axis=1))


def _skew(x):
    """Sample skewness of a 1-D array (0 for fewer than three points or zero spread).

    Args:
        x: Values.

    Returns:
        The skewness as a float, or ``np.nan`` if undefined.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    sd = float(np.std(x))
    if sd <= _TINY:
        return 0.0
    return float(np.mean(((x - float(np.mean(x))) / sd) ** 3))


def compute(ctx) -> dict:
    """Gradient-field and Lipschitz features of the landscape (gradient-field / Lipschitz analysis).

    Uses the local-linear gradients ``gᵢ`` (and all-pairs secant slopes) to characterize how the
    function *changes*. A robust high-quantile of the secant slopes estimates the Lipschitz
    constant (worst-case steepness) without letting a single near-duplicate pair explode it; the
    magnitude distribution of the field (mean, coefficient of variation, skew) says whether the
    landscape is uniformly steep or has isolated cliffs. Gradient *coherence* -- the mean cosine
    similarity between neighboring points' gradients -- is high for a smooth, coherent field
    (bowl/plane) and low for a rugged, multimodal one whose gradients point every which way. The
    normalized variation of neighboring gradients is a mesh-free curvature proxy: near-zero for a
    linear ramp, large for a wiggly high-curvature surface.

    Args:
        ctx: The shared :class:`Context` wrapping one labelled point cloud.

    Returns:
        A flat dict of gradient-field features keyed by short names, each a float or ``np.nan``.
    """
    keys = [
        "lipschitz",
        "lipschitz_max",
        "lipschitz_peakiness",
        "grad_mag_mean",
        "grad_mag_cv",
        "grad_mag_skew",
        "grad_coherence",
        "grad_curvature",
        "grad_curv_cv",
    ]
    out = {k: np.nan for k in keys}

    # -- Lipschitz constant from all-pairs secant slopes ---------------------------------------
    slopes = _pair_slopes(ctx)
    if slopes.size >= 1:
        out["lipschitz"] = float(np.quantile(slopes, 0.95))
        out["lipschitz_max"] = float(np.max(slopes))
        med = float(np.median(slopes))
        if med > _TINY:
            out["lipschitz_peakiness"] = float(np.quantile(slopes, 0.99) / med)

    # -- gradient-magnitude distribution -------------------------------------------------------
    try:
        G = np.asarray(ctx.local_gradients(), dtype=float)
    except Exception:
        G = None

    if G is not None and G.ndim == 2 and G.shape[0] >= 1 and np.all(np.isfinite(G)):
        norms = _grad_norms(G)
        mean_norm = float(np.mean(norms))
        out["grad_mag_mean"] = mean_norm
        if mean_norm > _TINY and norms.size >= 2:
            out["grad_mag_cv"] = float(np.std(norms) / mean_norm)
        out["grad_mag_skew"] = _skew(norms)

        # -- coherence & curvature from neighbor gradient comparisons --------------------------
        k = min(ctx.default_k(), max(1, ctx.n - 1))
        idx, _ = ctx.knn(k)
        unit = G / np.maximum(norms[:, None], _TINY)
        cos_acc, curv_acc = [], []
        for i in range(ctx.n):
            nb = idx[i]
            if nb.size == 0:
                continue
            if norms[i] > _TINY:
                cos_i = unit[nb] @ unit[i]
                good = norms[nb] > _TINY
                if np.any(good):
                    cos_acc.append(float(np.mean(cos_i[good])))
            diff = G[nb] - G[i]
            curv_acc.append(float(np.mean(np.sqrt(np.sum(diff**2, axis=1)))))
        if cos_acc:
            out["grad_coherence"] = float(np.mean(cos_acc))
        if curv_acc:
            curv = np.asarray(curv_acc, dtype=float)
            if mean_norm > _TINY:
                out["grad_curvature"] = float(np.mean(curv) / mean_norm)
                cm = float(np.mean(curv))
                if cm > _TINY and curv.size >= 2:
                    out["grad_curv_cv"] = float(np.std(curv) / cm)

    return out
