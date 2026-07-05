"""Additive-separability features via functional ANOVA: sum-of-1D-functions vs coupled interactions."""

import numpy as np


def _adj_r2(r2, n, p):
    """Adjusted R² that deflates an in-sample fit by its parameter count.

    Penalizes the raw coefficient of determination for the ``p`` fitted degrees of freedom so a
    richer model whose extra terms only capture noise does not look better than a lean one.

    Args:
        r2: In-sample R² of the fit.
        n: Number of samples.
        p: Number of (non-intercept) fitted parameters.

    Returns:
        The adjusted R² clipped to ``[0, 1]``, or ``np.nan`` when it is not defined.
    """
    try:
        if not np.isfinite(r2) or n - p - 1 <= 0:
            return np.nan
        adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - p - 1.0)
        return float(np.clip(adj, 0.0, 1.0))
    except Exception:
        return np.nan


def _r2(y, y_hat):
    """In-sample R² of a prediction against ``y`` (0 when ``y`` is constant).

    Args:
        y: Observed targets.
        y_hat: Predicted targets.

    Returns:
        The coefficient of determination as a float, 0.0 when ``y`` has no variance.
    """
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-300:
        return 0.0
    return float(1.0 - np.sum((y - y_hat) ** 2) / ss_tot)


def _poly_columns(x, deg):
    """Centered monomial columns ``[x, x², ..., x^deg]`` for one coordinate.

    Args:
        x: A 1-D coordinate vector.
        deg: Highest polynomial degree to include.

    Returns:
        A ``(n, deg)`` array whose columns are mean-centered powers of ``x`` (empty when the
        coordinate is constant).
    """
    cols = []
    for p in range(1, deg + 1):
        c = x**p
        c = c - c.mean()
        if np.std(c) > 1e-12:
            cols.append(c)
    if not cols:
        return np.zeros((x.shape[0], 0))
    return np.stack(cols, axis=1)


def _fit_r2(design, ys):
    """Least-squares R² of ``ys`` on a centered design plus an intercept.

    Args:
        design: A ``(n, p)`` feature matrix (columns already centered); may be empty.
        ys: The standardized target vector.

    Returns:
        A tuple ``(r2, p)`` of the in-sample R² and the number of design columns used.
    """
    n = ys.shape[0]
    if design.shape[1] == 0:
        return 0.0, 0
    D = np.concatenate([np.ones((n, 1)), design], axis=1)
    coef, *_ = np.linalg.lstsq(D, ys, rcond=None)
    return _r2(ys, D @ coef), design.shape[1]


def compute(ctx) -> dict:
    """Additive-separability landscape features from a functional-ANOVA decomposition.

    A separable function is a sum of one-dimensional pieces ``f(x) = Σ_j f_j(x_j)`` with no
    cross-coordinate coupling; a non-separable one (rotated, multiplicative, or ridge-coupled)
    needs interaction terms. This family fits per-coordinate main effects (low-degree polynomial
    smoothers on ``Xn → ys``), measures how much variance the sum of those main effects explains,
    then compares against a model that adds pairwise interaction terms. The ratio of additive to
    additive-plus-interaction explained variance is the separability index. The fitted quadratic
    Hessian supplies a complementary quadratic proxy: energy on its off-diagonal is exactly the
    bilinear coupling ``x_i x_j`` that separable functions lack. Main-effect variances across
    coordinates additionally reveal how the separable signal is spread over the input dimensions.

    Args:
        ctx: The shared :class:`Context` wrapping one labelled point cloud.

    Returns:
        A flat dict of separability features keyed by short names, each a float or ``np.nan``.
    """
    keys = [
        "separability_index",
        "main_effect_r2",
        "interaction_r2_gain",
        "residual_interaction_ratio",
        "hessian_offdiag_ratio",
        "hessian_diag_dominance",
        "max_pair_coupling",
        "main_effect_participation",
        "main_effect_gini",
        "top_dim_share",
        "hessian_reliable",
    ]
    out = {k: np.nan for k in keys}

    try:
        Xn = np.asarray(ctx.Xn, dtype=float)
        ys = np.asarray(ctx.ys, dtype=float)
        n, d = ctx.n, ctx.d

        finite = np.isfinite(ys).all() and np.isfinite(Xn).all()
        has_var = float(np.std(ctx.y)) > 0.0

        # ---- Hessian off-diagonal energy: quadratic proxy for coordinate coupling -------------
        # A pure sum of 1D functions has a diagonal Hessian; any off-diagonal mass is exactly the
        # bilinear x_i*x_j interaction that a separable function cannot have.
        try:
            q = ctx.quadratic()
            A = np.asarray(q.hessian, dtype=float)
            out["hessian_reliable"] = float(bool(q.reliable))
            if A.size and np.isfinite(A).all():
                off = A.copy()
                np.fill_diagonal(off, 0.0)
                total = float(np.sum(A**2))
                off_energy = float(np.sum(off**2))
                if total > 1e-300:
                    ratio = float(np.clip(off_energy / total, 0.0, 1.0))
                    out["hessian_offdiag_ratio"] = ratio
                    out["hessian_diag_dominance"] = 1.0 - ratio
                    # strongest single coupling relative to the overall curvature magnitude.
                    out["max_pair_coupling"] = float(np.clip(np.max(np.abs(off)) / np.sqrt(total), 0.0, 1.0))
                elif d >= 1:
                    # flat landscape: no curvature -> no detectable coupling either.
                    out["hessian_offdiag_ratio"] = 0.0 if d > 1 else 0.0
                    out["hessian_diag_dominance"] = 1.0
                    out["max_pair_coupling"] = 0.0
        except Exception:
            pass

        if not finite or not has_var or n < 3:
            # No usable variance to decompose; leave structural indices as nan but keep the
            # (possibly computable) Hessian-proxy features already set above.
            if d == 1 and finite and has_var:
                # a single coordinate is trivially separable.
                out["separability_index"] = 1.0
            return out

        # degrees chosen so the additive design stays well under-parameterized.
        deg_main = int(np.clip((n - 1) // max(1, d) - 1, 1, 3))

        # ---- per-dimension main effects (functional-ANOVA main terms) -------------------------
        blocks = []
        block_var = np.zeros(d)
        for j in range(d):
            cols = _poly_columns(Xn[:, j], deg_main)
            blocks.append(cols)

        add_design = (
            np.concatenate([b for b in blocks if b.shape[1] > 0], axis=1)
            if any(b.shape[1] > 0 for b in blocks)
            else np.zeros((n, 0))
        )
        r2_add, p_add = _fit_r2(add_design, ys)
        adj_add = _adj_r2(r2_add, n, p_add)
        out["main_effect_r2"] = float(np.clip(r2_add, 0.0, 1.0))

        # per-dimension explained variance: fit each coordinate's 1D effect alone (Sobol main
        # effect proxy) to see how the separable signal spreads across the input axes.
        for j in range(d):
            if blocks[j].shape[1] > 0:
                rj, _ = _fit_r2(blocks[j], ys)
                block_var[j] = max(0.0, rj)
        s = float(np.sum(block_var))
        if s > 1e-12:
            w = block_var / s
            # participation ratio: effective number of coordinates carrying main-effect signal,
            # rescaled to [0, 1] (1 = signal evenly spread, ~0 = one dominant axis).
            pr = float((np.sum(w) ** 2) / np.sum(w**2))
            out["main_effect_participation"] = float(np.clip((pr - 1.0) / (d - 1.0), 0.0, 1.0)) if d > 1 else 1.0
            # Gini concentration of the per-dim variances (0 = uniform, ->1 = concentrated).
            sw = np.sort(w)
            idx = np.arange(1, d + 1)
            gini = float((np.sum((2 * idx - d - 1) * sw)) / (d)) if d > 0 else 0.0
            out["main_effect_gini"] = float(np.clip(gini, 0.0, 1.0))
            out["top_dim_share"] = float(np.clip(np.max(w), 0.0, 1.0))
        else:
            out["main_effect_participation"] = np.nan if d > 1 else 1.0
            out["main_effect_gini"] = 0.0
            out["top_dim_share"] = np.nan

        # ---- full model = main effects + pairwise interactions --------------------------------
        if d == 1:
            # nothing to couple with: perfectly separable by construction.
            out["separability_index"] = 1.0
            out["interaction_r2_gain"] = 0.0
            out["residual_interaction_ratio"] = 0.0
            return out

        # centered bilinear cross terms; cap the count so the fit stays under-parameterized.
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        budget = max(0, (n - 2) - p_add)
        inter_cols: list[np.ndarray] = []
        for i, j in pairs:
            if len(inter_cols) >= budget:
                break
            c = Xn[:, i] * Xn[:, j]
            c = c - c.mean()
            if np.std(c) > 1e-12:
                inter_cols.append(c)
        inter = np.stack(inter_cols, axis=1) if inter_cols else np.zeros((n, 0))

        full_design = np.concatenate([add_design, inter], axis=1) if inter.shape[1] > 0 else add_design
        r2_full, p_full = _fit_r2(full_design, ys)

        # compare with adjusted R² so interaction terms that only fit noise do not inflate the
        # apparent non-separability.
        adj_full = _adj_r2(r2_full, n, p_full)
        gain = 0.0
        if adj_add is not None and adj_full is not None and np.isfinite(adj_add) and np.isfinite(adj_full):
            gain = max(0.0, float(adj_full - adj_add))
            denom = adj_full if adj_full > 1e-9 else np.nan
            if np.isfinite(denom):
                # separability index: additive share of the total explained variance.
                out["separability_index"] = float(np.clip(adj_add / adj_full, 0.0, 1.0))
            else:
                # neither model explains anything -> treat as separable (no interaction found).
                out["separability_index"] = 1.0
        out["interaction_r2_gain"] = float(gain)

        # of the variance the main effects leave unexplained, how much is structured interaction
        # (vs irreducible noise): high => genuine coupling, low => additive up to noise.
        if np.isfinite(adj_add):
            residual = 1.0 - adj_add
            out["residual_interaction_ratio"] = float(np.clip(gain / residual, 0.0, 1.0)) if residual > 1e-9 else 0.0
    except Exception:
        pass

    # final scrub: everything a plain float or nan.
    for k in keys:
        v = out[k]
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = np.nan
    return out
