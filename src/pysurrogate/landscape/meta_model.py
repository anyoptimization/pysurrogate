"""Surrogate-quality meta-model features: how well simple linear/quadratic models fit the cloud."""

import numpy as np


def _adj_r2(r2, n, p):
    """Adjusted R² correcting a fit's R² for its ``p`` predictors and ``n`` samples.

    Args:
        r2: In-sample coefficient of determination.
        n: Number of samples.
        p: Number of predictors (excluding the intercept).

    Returns:
        The adjusted R², or ``np.nan`` when there are too few degrees of freedom.
    """
    denom = n - p - 1
    if denom <= 0:
        return np.nan
    return float(1.0 - (1.0 - r2) * (n - 1) / denom)


def compute(ctx) -> dict:
    """Meta-model landscape features from global linear and quadratic surrogate fits.

    Fits a plain linear model and a full quadratic (with interactions) to the standardized
    outputs and summarizes how much of the landscape each explains, how much the curvature adds,
    and the shape/conditioning of the fitted coefficients. High linear R² marks a near-linear
    (planar) landscape; a large curvature gain marks a smooth bowl; a large linear-coefficient
    spread marks a strongly anisotropic / few-active-variable landscape.

    Args:
        ctx: The shared :class:`Context` wrapping one labelled point cloud.

    Returns:
        A flat dict of meta-model features keyed by short names, each a float or ``np.nan``.
    """
    keys = [
        "lin_r2",
        "lin_r2_adj",
        "quad_r2",
        "quad_r2_adj",
        "curv_gain",
        "quad_improve_ratio",
        "lin_coef_min",
        "lin_coef_max",
        "lin_coef_spread",
        "quad_curv_cond",
        "intercept_abs",
        "is_linear",
        "quad_reliable",
    ]
    out = {k: np.nan for k in keys}

    try:
        # A constant landscape has no structure to model: every meta-model feature is undefined
        # except that a (degenerate) linear model trivially "explains" it.
        if not np.isfinite(ctx.ys).all() or float(np.std(ctx.y)) <= 0.0:
            out["is_linear"] = 1.0
            return out

        q = ctx.quadratic()
        n, d = ctx.n, ctx.d

        lin_r2 = float(np.clip(q.linear_r2, -1e6, 1.0))
        quad_r2 = float(np.clip(q.r2, -1e6, 1.0))
        out["lin_r2"] = lin_r2
        out["quad_r2"] = quad_r2

        # predictor counts: linear -> d ; full quadratic -> d (linear) + d (squares) + pairs.
        p_lin = d
        p_quad = 2 * d + d * (d - 1) // 2
        out["lin_r2_adj"] = _adj_r2(lin_r2, n, p_lin)
        out["quad_r2_adj"] = _adj_r2(quad_r2, n, p_quad)

        # curvature gain: extra variance the second-order terms capture over the plane.
        gain = quad_r2 - lin_r2
        out["curv_gain"] = float(gain)
        remaining = 1.0 - lin_r2
        if remaining > 1e-9:
            out["quad_improve_ratio"] = float(np.clip(gain / remaining, -1e6, 1.0))
        else:
            # linear already perfect -> curvature can add nothing.
            out["quad_improve_ratio"] = 0.0

        # shape of the linear coefficients (anisotropy / dominance of a few directions).
        b = np.abs(np.asarray(q.linear, dtype=float).ravel())
        b = b[np.isfinite(b)]
        if b.size:
            bmax = float(np.max(b))
            nz = b[b > 1e-12]
            bmin = float(np.min(nz)) if nz.size else 0.0
            out["lin_coef_min"] = bmin
            out["lin_coef_max"] = bmax
            if bmin > 1e-12:
                out["lin_coef_spread"] = float(bmax / bmin)
            elif bmax > 1e-12:
                # some coefficient is (numerically) zero -> unbounded spread.
                out["lin_coef_spread"] = np.inf
            else:
                out["lin_coef_spread"] = np.nan

        # conditioning of the pure curvature: ratio of largest to smallest |square-term| coef.
        # hessian diagonal = 2 * square-term coefficients.
        curv = np.abs(np.diag(np.asarray(q.hessian, dtype=float))) / 2.0
        curv = curv[np.isfinite(curv)]
        cnz = curv[curv > 1e-12]
        if cnz.size >= 1:
            cmax = float(np.max(curv))
            cmin = float(np.min(cnz))
            out["quad_curv_cond"] = float(cmax / cmin) if cmin > 1e-12 else np.inf
        else:
            out["quad_curv_cond"] = np.nan

        out["intercept_abs"] = float(abs(q.intercept))

        # a linear model "already explains" y when it captures almost all variance and curvature
        # adds essentially nothing.
        out["is_linear"] = float(lin_r2 >= 0.99 and gain < 0.01)

        out["quad_reliable"] = float(bool(q.reliable))
    except Exception:
        return out

    # final scrub: coerce anything odd to float / nan (inf is allowed as a meaningful signal).
    for k in keys:
        v = out[k]
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = np.nan
    return out
