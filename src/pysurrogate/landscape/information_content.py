"""Information-content features of the fitness landscape via symbolized nearest-neighbor walks (Munoz et al. 2015)."""

import numpy as np


def _safe_float(x):
    """Coerce a value to a finite Python float, mapping non-finite results to ``np.nan``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _nn_tour(D, start, n):
    """Greedy nearest-neighbor tour over the distance matrix starting at ``start``.

    Args:
        D: The ``(n, n)`` pairwise distance matrix.
        start: Index of the first visited point.
        n: Number of points.

    Returns:
        An integer array of length ``n`` giving the visit order (a space-filling-ish walk that
        keeps consecutive samples close so the objective sequence reflects local landscape moves).
    """
    used = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=int)
    cur = int(start)
    used[cur] = True
    tour[0] = cur
    for i in range(1, n):
        d = D[cur].copy()
        d[used] = np.inf
        nxt = int(np.argmin(d))
        tour[i] = nxt
        used[nxt] = True
        cur = nxt
    return tour


def _ratios(ctx, tour):
    """Per-step objective slope ``(y_{i+1}-y_i)/dist`` along a tour, using standardized outputs.

    Args:
        ctx: The landscape context.
        tour: Visit order (length ``n``).

    Returns:
        A length ``n-1`` array of slopes; steps with zero spatial displacement are dropped so the
        ratio is always finite.
    """
    D = ctx.distances()
    a = tour[:-1]
    b = tour[1:]
    dist = D[a, b]
    dy = ctx.ys[b] - ctx.ys[a]
    ok = dist > 1e-12
    if not np.any(ok):
        return np.zeros(0)
    return dy[ok] / dist[ok]


def _symbolize(r, eps):
    """Map slopes to the alphabet ``{-1, 0, 1}`` at sensitivity ``eps`` (dead-zone ``|r|<=eps``)."""
    phi = np.zeros(r.shape[0], dtype=np.int8)
    phi[r > eps] = 1
    phi[r < -eps] = -1
    return phi


def _information_content(phi):
    """Shannon information content of the symbol string over its non-equal consecutive pairs.

    Blocks of equal successive symbols carry no information; the entropy is taken over the (up to
    six) distinct ordered pairs ``(p, q)`` with ``p != q`` and normalized by ``log 6`` so the
    result lies in ``[0, 1]`` (1 == maximally rugged / unpredictable transitions).

    Args:
        phi: The symbol sequence in ``{-1, 0, 1}``.

    Returns:
        The normalized information content in ``[0, 1]``.
    """
    if phi.shape[0] < 2:
        return 0.0
    a = phi[:-1]
    b = phi[1:]
    diff = a != b
    m = int(np.count_nonzero(diff))
    if m == 0:
        return 0.0
    # encode each differing ordered pair as one of the six symbols and count them
    code = (a[diff] + 1) * 3 + (b[diff] + 1)
    counts = np.bincount(code, minlength=9)
    counts = counts[counts > 0]
    p = counts / m
    return float(-np.sum(p * np.log(p)) / np.log(6.0))


def _partial_information(phi, denom):
    """Partial information content: density of slope-sign alternations after dropping flats.

    Zeros are removed, then consecutive duplicates collapsed; the number of surviving alternations
    (local optima along the walk) is normalized by the original sequence length ``denom``.

    Args:
        phi: The symbol sequence in ``{-1, 0, 1}``.
        denom: Length of the original (pre-filter) symbol sequence.

    Returns:
        The partial information content in ``[0, 1]``.
    """
    nz = phi[phi != 0]
    if nz.shape[0] < 2 or denom <= 0:
        return 0.0
    changes = int(np.count_nonzero(nz[:-1] != nz[1:]))
    return float(changes / denom)


def compute(ctx) -> dict:
    """Information-content features of the labelled cloud (ELA information content, Munoz 2015).

    The sample is ordered along one or more greedy nearest-neighbor walks; the sequence of
    objective changes is symbolized into ``{-1, 0, 1}`` under a swept sensitivity ``eps``, and two
    curves are traced: the information content ``H(eps)`` (entropy of consecutive symbol-pair
    transitions, a ruggedness measure) and the partial information ``M(eps)`` (density of slope-sign
    alternations, a modality measure). Summaries of these curves distinguish smooth unimodal
    landscapes (low ruggedness, information concentrated at a single scale) from rugged, multimodal,
    or noisy ones (high ``H_max``, high initial modality, information spread across scales). Walk
    curves are averaged over several random starts for stability, and ``eps`` values are reported on
    a ``log10`` scale relative to the objective-slope magnitude.

    Args:
        ctx: A landscape :class:`Context` wrapping the labelled cloud ``(X, y)``.

    Returns:
        A flat dict of float features: ``h_max`` (maximum information content / peak ruggedness),
        ``eps_max`` (``log10`` sensitivity at that peak), ``eps_s`` (settling sensitivity where
        ``H`` finally drops below a small threshold), ``eps_ratio`` (informative-scale width),
        ``h0`` (information content at full sensitivity), ``h_auc`` (mean ``H`` across scales /
        multi-scale ruggedness), ``m0`` (initial partial information / modality), ``m_max`` (peak
        partial information), and ``flat_frac`` (fraction of near-flat steps / neutrality).
    """
    keys = [
        "h_max",
        "eps_max",
        "eps_s",
        "eps_ratio",
        "h0",
        "h_auc",
        "m0",
        "m_max",
        "flat_frac",
    ]
    out = {k: np.nan for k in keys}

    try:
        n = ctx.n
        if n < 4:
            return out

        D = ctx.distances()

        # Collect per-step slopes from several nearest-neighbor walks for a stable curve estimate.
        n_walks = int(min(5, n))
        try:
            starts = ctx.rng.choice(n, size=n_walks, replace=False)
        except Exception:
            starts = np.arange(n_walks)

        rlist = []
        for s in starts:
            tour = _nn_tour(D, int(s), n)
            r = _ratios(ctx, tour)
            if r.shape[0] >= 2:
                rlist.append(r)
        if not rlist:
            return out

        rmax = max(float(np.max(np.abs(r))) for r in rlist)
        settle = 0.05

        # --- full-sensitivity (eps = 0) features: modality and fine-scale ruggedness ---
        h0_vals, m0_vals, flat_vals = [], [], []
        for r in rlist:
            phi = _symbolize(r, 0.0)
            h0_vals.append(_information_content(phi))
            m0_vals.append(_partial_information(phi, r.shape[0]))
        out["h0"] = _safe_float(np.mean(h0_vals))
        out["m0"] = _safe_float(np.mean(m0_vals))

        # near-flat step fraction at a tiny relative sensitivity (neutrality / plateaus)
        if rmax > 0:
            eps_tiny = rmax * 1e-3
            for r in rlist:
                flat_vals.append(float(np.mean(np.abs(r) <= eps_tiny)))
            out["flat_frac"] = _safe_float(np.mean(flat_vals))
        else:
            # constant landscape: every step flat, no information at any scale
            out["h_max"] = 0.0
            out["h0"] = 0.0
            out["h_auc"] = 0.0
            out["m0"] = 0.0
            out["m_max"] = 0.0
            out["flat_frac"] = 1.0
            return out

        # --- sweep eps on a log grid spanning fine to coarse relative to the slope magnitude ---
        n_eps = 200
        grid = rmax * np.power(10.0, np.linspace(-5.0, 0.5, n_eps))

        H = np.zeros(n_eps)
        M = np.zeros(n_eps)
        for k, eps in enumerate(grid):
            hk, mk = [], []
            for r in rlist:
                phi = _symbolize(r, eps)
                hk.append(_information_content(phi))
                mk.append(_partial_information(phi, r.shape[0]))
            H[k] = np.mean(hk)
            M[k] = np.mean(mk)

        out["h_max"] = _safe_float(np.max(H))
        out["m_max"] = _safe_float(np.max(M))
        out["h_auc"] = _safe_float(np.mean(H))

        log_grid = np.log10(grid)

        # sensitivity at maximum information content (characteristic ruggedness scale)
        kmax = int(np.argmax(H))
        out["eps_max"] = _safe_float(log_grid[kmax])

        # settling sensitivity: the coarsest eps at which H is still non-trivial, i.e. just past the
        # last grid point whose information exceeds the settling threshold
        above = np.where(H >= settle)[0]
        if above.size == 0:
            eps_s = _safe_float(log_grid[0])
        else:
            last = int(above[-1])
            eps_s = _safe_float(log_grid[min(last + 1, n_eps - 1)])
        out["eps_s"] = eps_s

        # width of the informative sensitivity band (settling minus peak location)
        if np.isfinite(eps_s) and np.isfinite(out["eps_max"]):
            out["eps_ratio"] = _safe_float(eps_s - out["eps_max"])

    except Exception:
        return {k: _safe_float(v) for k, v in out.items()}

    return {k: _safe_float(v) for k, v in out.items()}
