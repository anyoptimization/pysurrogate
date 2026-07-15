"""Nearest-better-clustering features: nn/nb distance ratios and funnel structure (Kerschke & Preuss)."""

import numpy as np

from ._util import _corr, _gini, _safe_float


def _nn_nb_distances(ctx):
    """Compute per-point nearest-neighbor, nearest-better distances and the nearest-better edge.

    Args:
        ctx: The landscape context.

    Returns:
        A tuple ``(nn, nb, nb_edge)`` where ``nn`` is the distance to the closest other point,
        ``nb`` the distance to the closest strictly-better point (``nan`` for the global best or
        when no better point exists), and ``nb_edge`` the index of that better point (``-1`` when
        undefined). Each array has length ``n``.
    """
    D = ctx.distances()
    y = ctx.y
    n = ctx.n
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)

    nn = Dm.min(axis=1)

    nb = np.full(n, np.nan)
    nb_edge = np.full(n, -1, dtype=int)
    for i in range(n):
        better = y < y[i]
        if not np.any(better):
            continue
        d = np.where(better, Dm[i], np.inf)
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            nb[i] = d[j]
            nb_edge[i] = j
    return nn, nb, nb_edge


def compute(ctx) -> dict:
    """Nearest-better-clustering (NBC) structural features of the labelled cloud.

    For each point the distance to its nearest neighbor (``nn``) and to its nearest strictly
    better point (``nb``) are compared. In a single-funnel (unimodal) landscape a point's nearest
    neighbor is usually also its nearest better point, so ``nb/nn`` stays near one; in a
    multi-funnel (multimodal) landscape, local optima must reach across a valley to find something
    better, inflating ``nb`` and spreading the ratio distribution. The nearest-better graph's
    indegree concentration further separates one dominant basin from many competing ones.

    Args:
        ctx: A landscape :class:`Context` wrapping the labelled cloud ``(X, y)``.

    Returns:
        A flat dict of float features: central tendency and dispersion of the ``nb/nn`` ratio,
        the standard-deviation and mean ratios of the two distance sets, the ``nn``-``nb``
        correlation, correlations of ``nb`` distance / ratio / graph indegree with fitness, and
        indegree-concentration funnel indicators (max indegree and Gini).
    """
    keys = [
        "mean_ratio",
        "median_ratio",
        "ratio_cv",
        "sd_ratio",
        "mean_dist_ratio",
        "nn_nb_cor",
        "nb_fitness_cor",
        "ratio_fitness_cor",
        "indegree_fitness_cor",
        "indegree_max",
        "indegree_gini",
        "funnel_frac",
    ]
    out = {k: np.nan for k in keys}

    try:
        n = ctx.n
        if n < 3:
            return out

        nn, nb, nb_edge = _nn_nb_distances(ctx)
        y = ctx.y

        valid = np.isfinite(nb) & np.isfinite(nn) & (nn > 1e-12)
        if np.count_nonzero(valid) >= 2:
            ratio = nb[valid] / nn[valid]
            out["mean_ratio"] = _safe_float(np.mean(ratio))
            out["median_ratio"] = _safe_float(np.median(ratio))
            mr = np.mean(ratio)
            if mr > 1e-12:
                out["ratio_cv"] = _safe_float(np.std(ratio) / mr)
            # Fraction of points whose nearest better is much farther than their nearest
            # neighbor -> they sit at the rim of a distinct funnel (multimodal signature).
            out["funnel_frac"] = _safe_float(np.mean(ratio > 2.0))

        nb_ok = np.isfinite(nb)
        nn_ok = np.isfinite(nn)
        both = nb_ok & nn_ok
        if np.count_nonzero(both) >= 2:
            nn_sd = np.std(nn[both])
            nb_sd = np.std(nb[both])
            if nn_sd > 1e-12:
                out["sd_ratio"] = _safe_float(nb_sd / nn_sd)
            nn_mean = np.mean(nn[both])
            if nn_mean > 1e-12:
                out["mean_dist_ratio"] = _safe_float(np.mean(nb[both]) / nn_mean)
            out["nn_nb_cor"] = _corr(nn[both], nb[both])

        # Fitness correlations (over points that have a defined nearest-better distance).
        if np.count_nonzero(nb_ok) >= 3:
            out["nb_fitness_cor"] = _corr(nb[nb_ok], y[nb_ok])
            good = nb_ok & (nn > 1e-12)
            out["ratio_fitness_cor"] = _corr(nb[good] / nn[good], y[good])

        # Nearest-better graph indegree: how many points point at each node.
        indeg = np.zeros(n, dtype=float)
        for j in nb_edge:
            if j >= 0:
                indeg[j] += 1.0
        total_edges = float(indeg.sum())
        if total_edges > 0:
            out["indegree_max"] = _safe_float(indeg.max() / total_edges)
            out["indegree_gini"] = _gini(indeg)
            out["indegree_fitness_cor"] = _corr(indeg, y)

    except Exception:
        return {k: _safe_float(v) for k, v in out.items()}

    return {k: _safe_float(v) for k, v in out.items()}
