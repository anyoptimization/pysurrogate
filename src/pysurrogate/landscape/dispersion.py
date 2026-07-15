"""Global-structure features: Lunacek dispersion of the elite set and fitness-distance correlation."""

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]

from ._util import _corr, _safe_float


def _spearman(a, b):
    """Spearman (rank) correlation via Pearson on ranks; robust to monotone nonlinearity."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return np.nan
    try:
        return _corr(rankdata(a), rankdata(b))
    except Exception:
        return np.nan


def _mean_pairwise(D, idx):
    """Mean of the off-diagonal pairwise distances among the rows/cols selected by ``idx``.

    Args:
        D: The full ``(n, n)`` distance matrix.
        idx: 1-D array of point indices defining the subset.

    Returns:
        The mean of the upper-triangle distances within the subset, or ``nan`` when fewer than
        two points are selected.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size < 2:
        return np.nan
    sub = D[np.ix_(idx, idx)]
    iu = np.triu_indices(idx.size, k=1)
    vals = sub[iu]
    if vals.size == 0:
        return np.nan
    return _safe_float(np.mean(vals))


def _elite_indices(y, q):
    """Indices of the best (lowest-``y``) fraction ``q`` of points, at least two when possible.

    Args:
        y: The objective values (minimization; lower is better).
        q: The elite fraction in ``(0, 1]``.

    Returns:
        A 1-D integer array of the elite point indices (ordered from best to worst).
    """
    n = y.size
    m = int(round(q * n))
    m = int(np.clip(m, 2, n))
    order = np.argsort(y, kind="mergesort")
    return order[:m]


def compute(ctx) -> dict:
    """Global-structure (funnel) features from elite-set dispersion and fitness-distance coupling.

    Two classical global diagnostics are combined. The Lunacek **dispersion** metric compares how
    spread out the best ``q%`` of points are against the spread of the whole cloud: if the elite
    points are *more* dispersed than average (positive dispersion) the good regions are scattered
    across the domain -- a multi-funnel / multi-basin landscape; if they are tighter than average
    (negative dispersion) the good points collapse into one region -- a single global funnel. The
    trend of dispersion as the elite fraction shrinks sharpens this: a single funnel grows steadily
    more concentrated as ``q`` drops, a multi-funnel landscape stays scattered. The **fitness-
    distance correlation** (Jones & Forrest) measures how tightly objective value tracks distance
    to the incumbent best point: a strong positive correlation means "closer is better" everywhere
    (a searchable unimodal bowl), while a weak or negative one signals deception / competing optima.

    All distances use the normalized inputs (``ctx.distances()`` on ``Xn``), so dispersion values
    are comparable across problems and are additionally reported as a scale-free ratio.

    Args:
        ctx: A landscape :class:`Context` wrapping the labelled cloud ``(X, y)``.

    Returns:
        A flat dict of float features: signed Lunacek dispersion for elite fractions
        ``{2, 5, 10, 25}%`` (``disp_02..disp_25``), a scale-free dispersion ratio for the 10%
        elite (``disp_ratio_10``), the dispersion trend across fractions (``disp_slope``), the
        Pearson and Spearman fitness-distance correlations to the best point (``fdc``,
        ``fdc_spearman``), and the normalized slope of fitness against distance-to-best
        (``fdc_slope``).
    """
    keys = [
        "disp_02",
        "disp_05",
        "disp_10",
        "disp_25",
        "disp_ratio_10",
        "disp_slope",
        "fdc",
        "fdc_spearman",
        "fdc_slope",
    ]
    out = {k: np.nan for k in keys}

    try:
        n = ctx.n
        if n < 3:
            return out

        D = ctx.distances()
        y = ctx.y
        ys = ctx.ys

        # -- Lunacek dispersion: elite spread minus (or over) whole-cloud spread ----------------
        iu_all = np.triu_indices(n, k=1)
        all_vals = D[iu_all]
        mean_all = _safe_float(np.mean(all_vals)) if all_vals.size else np.nan

        fractions = [0.02, 0.05, 0.10, 0.25]
        fkeys = ["disp_02", "disp_05", "disp_10", "disp_25"]
        disp_vals = []
        elite_means = {}
        for q, key in zip(fractions, fkeys):
            idx = _elite_indices(y, q)
            mean_elite = _mean_pairwise(D, idx)
            elite_means[q] = mean_elite
            if np.isfinite(mean_elite) and np.isfinite(mean_all):
                out[key] = _safe_float(mean_elite - mean_all)
                disp_vals.append((q, mean_elite))

        # scale-free ratio for the 10% elite (elite spread / whole spread - 1)
        mean_e10 = elite_means[0.10]
        if np.isfinite(mean_e10) and np.isfinite(mean_all) and mean_all > 1e-12:
            out["disp_ratio_10"] = _safe_float(mean_e10 / mean_all - 1.0)

        # dispersion trend: slope of elite spread vs log(fraction). Negative -> spread shrinks as
        # the elite set tightens (single funnel); ~0 or positive -> stays scattered (multifunnel).
        if len(disp_vals) >= 2 and np.isfinite(mean_all) and mean_all > 1e-12:
            qs = np.log(np.array([q for q, _ in disp_vals]))
            sv = np.array([s for _, s in disp_vals]) / mean_all
            if np.std(qs) > 1e-12:
                slope = np.polyfit(qs, sv, 1)[0]
                out["disp_slope"] = _safe_float(slope)

        # -- Fitness-distance correlation to the incumbent best --------------------------------
        d_best = D[ctx.best]
        mask = np.ones(n, dtype=bool)
        mask[ctx.best] = False
        db = d_best[mask]
        yy = y[mask]
        yys = ys[mask]
        if db.size >= 3 and np.std(db) > 1e-12 and np.std(yy) > 1e-12:
            out["fdc"] = _corr(db, yy)
            out["fdc_spearman"] = _spearman(db, yy)
            # normalized slope: change in standardized fitness per unit distance to best.
            try:
                slope = np.polyfit(db, yys, 1)[0]
                out["fdc_slope"] = _safe_float(slope)
            except Exception:
                out["fdc_slope"] = np.nan

    except Exception:
        return {k: _safe_float(v) for k, v in out.items()}

    return {k: _safe_float(v) for k, v in out.items()}
