"""Global-curvature features from the fitted quadratic Hessian: bowl vs saddle vs flat."""

import numpy as np

from ._util import _safe_float


def _eigvals(A):
    """Real, sorted-descending eigenvalues of a symmetric matrix (empty on failure).

    Args:
        A: A square, symmetric matrix.

    Returns:
        A 1-D array of finite eigenvalues sorted from most positive to most negative, or an
        empty array when the decomposition fails or yields no finite value.
    """
    try:
        w = np.linalg.eigvalsh(np.asarray(A, dtype=float))
    except Exception:
        return np.array([])
    w = w[np.isfinite(w)]
    return np.sort(w)[::-1]


def compute(ctx) -> dict:
    """Global-curvature landscape features from the quadratic Hessian ``A``.

    Fits (via the shared context) a global quadratic ``y ~ c + bᵀx + ½ xᵀ A x`` and reads the
    curvature purely from the symmetric Hessian ``A``. Its eigenvalues are the principal
    curvatures of the landscape: all-positive marks a convex bowl, mixed signs a saddle, and
    near-zero magnitudes a flat/ridged landscape. The features summarize the magnitude,
    conditioning, sign structure, and definiteness of that curvature, plus how much curvature
    matters relative to the linear trend.

    Args:
        ctx: The shared :class:`Context` wrapping one labelled point cloud.

    Returns:
        A flat dict of curvature features keyed by short names, each a float or ``np.nan``.
    """
    keys = [
        "eig_abs_mean",
        "eig_abs_max",
        "eig_abs_min",
        "condition_number",
        "flat_frac",
        "convex_frac",
        "neg_curv_frac",
        "definiteness",
        "mean_curvature",
        "curv_energy",
        "curv_anisotropy",
        "curv_linear_ratio",
        "curv_reliable",
    ]
    out = {k: np.nan for k in keys}

    try:
        # A constant landscape (or non-finite outputs) has no curvature to speak of: report a
        # flat, degenerate signature rather than a spurious fit.
        if not np.isfinite(ctx.ys).all() or float(np.std(ctx.y)) <= 0.0:
            out.update(
                {
                    "eig_abs_mean": 0.0,
                    "eig_abs_max": 0.0,
                    "eig_abs_min": 0.0,
                    "flat_frac": 1.0,
                    "convex_frac": np.nan,
                    "neg_curv_frac": np.nan,
                    "definiteness": 0.0,
                    "mean_curvature": 0.0,
                    "curv_energy": 0.0,
                    "curv_reliable": 0.0,
                }
            )
            return out

        q = ctx.quadratic()
        out["curv_reliable"] = float(bool(q.reliable))

        w = _eigvals(q.hessian)
        if w.size == 0:
            return out

        aw = np.abs(w)
        d = float(w.size)

        out["eig_abs_mean"] = float(np.mean(aw))
        out["eig_abs_max"] = float(np.max(aw))
        out["eig_abs_min"] = float(np.min(aw))

        # scale threshold: eigenvalues far below the dominant curvature count as "flat" and are
        # excluded from sign / conditioning summaries so numerical dust does not dominate.
        emax = float(np.max(aw))
        tol = max(1e-9, 1e-6 * emax)
        nontiny = aw[aw > tol]

        # fraction of flat (near-zero-curvature) directions: ~0 for a full-rank bowl, high for a
        # ridge/plateau whose Hessian is rank-deficient.
        out["flat_frac"] = float(np.mean(aw <= tol))

        # condition number over non-tiny curvatures only: near 1 = isotropic bowl, huge = a stiff
        # valley/ridge with one dominant direction. Flat directions are deliberately excluded
        # (their prevalence is reported separately as ``flat_frac``), so a rank-deficient ridge
        # can still read as well-conditioned within its curved subspace.
        if nontiny.size >= 1:
            out["condition_number"] = float(emax / np.min(nontiny))
        else:
            out["condition_number"] = np.nan

        # sign structure over the meaningful (non-tiny) curvatures.
        signif = w[aw > tol]
        m = float(signif.size)
        if m >= 1:
            n_pos = float(np.sum(signif > 0))
            n_neg = float(np.sum(signif < 0))
            out["convex_frac"] = n_pos / m
            out["neg_curv_frac"] = n_neg / m
            # definiteness indicator in [-1, 1]: +1 positive-definite bowl, -1 concave dome,
            # values near 0 an indefinite saddle.
            if n_neg == 0:
                out["definiteness"] = 1.0
            elif n_pos == 0:
                out["definiteness"] = -1.0
            else:
                out["definiteness"] = float((n_pos - n_neg) / m)
        else:
            # all curvatures negligible -> effectively flat / indefinite-undetermined.
            out["convex_frac"] = np.nan
            out["neg_curv_frac"] = np.nan
            out["definiteness"] = 0.0

        # mean curvature = trace(A)/d = average principal curvature (signed): >0 net-convex bowl,
        # <0 net-concave, ~0 balanced saddle or flat.
        out["mean_curvature"] = float(np.sum(w) / d)

        # total curvature energy: Frobenius-style sum of squared curvatures (overall bendiness).
        out["curv_energy"] = float(np.sum(w**2))

        # anisotropy: spread of curvature magnitudes normalized by their mean (0 = perfectly
        # isotropic, larger = a few directions carry the curvature). Uses non-tiny values.
        if nontiny.size >= 2:
            mu = float(np.mean(nontiny))
            out["curv_anisotropy"] = float(np.std(nontiny) / mu) if mu > 0 else np.nan
        elif nontiny.size == 1:
            out["curv_anisotropy"] = 0.0
        else:
            out["curv_anisotropy"] = np.nan

        # curvature-vs-linear balance: how much the second-order term explains beyond the plane.
        # read from the fit's R² gain, mapped to a bounded [0, 1] weight (1 = curvature-dominated).
        lin_r2 = float(np.clip(q.linear_r2, -1e6, 1.0))
        quad_r2 = float(np.clip(q.r2, -1e6, 1.0))
        gain = quad_r2 - lin_r2
        denom = abs(quad_r2) + abs(lin_r2)
        if denom > 1e-12:
            out["curv_linear_ratio"] = float(np.clip(gain / denom, 0.0, 1.0))
        else:
            out["curv_linear_ratio"] = 0.0
    except Exception:
        return out

    # final scrub: plain floats only; non-finite (incl. inf) reads as nan per the feature contract.
    return {k: _safe_float(v) for k, v in out.items()}
