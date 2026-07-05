"""Basin-count / multimodality features from sample local-optima counting on the point cloud."""

import numpy as np


def _safe_float(x):
    """Coerce a value to a finite Python float, mapping non-finite results to ``np.nan``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _local_min_mask(ys, idx):
    """Boolean mask of points strictly below every one of their neighbors in ``idx``.

    Args:
        ys: Standardized outputs, shape ``(n,)``.
        idx: Neighbor index array, shape ``(n, k)``.

    Returns:
        Boolean array shape ``(n,)``; ``True`` where the point is a strict local minimum.
    """
    neigh = ys[idx]  # (n, k)
    return np.all(ys[:, None] < neigh, axis=1)


def _local_max_mask(ys, idx):
    """Boolean mask of points strictly above every one of their neighbors in ``idx``."""
    neigh = ys[idx]
    return np.all(ys[:, None] > neigh, axis=1)


def _k_sweep(ctx):
    """A small ascending sweep of neighborhood sizes, valid for the sample size.

    Returns:
        A sorted list of distinct ``k`` values in ``[2, n-1]`` spanning small to moderate
        neighborhoods (empty when the cloud is too small to define any neighborhood).
    """
    n = ctx.n
    if n < 4:
        return []
    base = ctx.default_k()
    cand = {2, 3, 5, base, base * 2, max(2, int(round(np.sqrt(n))))}
    ks = sorted(k for k in cand if 2 <= k <= n - 1)
    return ks


def compute(ctx) -> dict:
    """Multimodality / basin-count features from counting sample-level local minima.

    A sample point counts as a local minimum when its objective value lies strictly below all of
    its ``k`` nearest neighbors. Sweeping ``k`` makes the count robust to spurious minima from
    sampling noise (small ``k`` over-counts; larger ``k`` retains only deeper basins). The
    fraction of such points, the extrapolated basin count, the mean basin size, and how quickly
    the count decays with ``k`` together diagnose how rugged / multimodal the landscape is.

    Args:
        ctx: A landscape :class:`Context` wrapping the labelled cloud ``(X, y)``.

    Returns:
        A flat dict of float features: the local-minimum fraction (at the default ``k`` and
        averaged over the sweep), an estimated basin count, local-minimum density, a mean
        basin-size proxy, the decay of the count with ``k``, the local-maximum fraction, the mean
        relative depth of the detected minima, and their spatial dispersion.
    """
    keys = [
        "local_min_frac",
        "local_min_frac_mean",
        "n_basins_est",
        "local_min_density",
        "mean_basin_size",
        "min_frac_k_decay",
        "local_max_frac",
        "min_depth_mean",
        "basin_dispersion",
        "min_max_ratio",
    ]
    out = {k: np.nan for k in keys}

    try:
        ys = np.asarray(ctx.ys, dtype=float).ravel()
        n = ys.size
    except Exception:
        return out

    if n < 4 or not np.all(np.isfinite(ys)):
        return out

    # Constant (or near-constant) objective => no strict minima anywhere; report zeros so the
    # landscape reads as flat/unimodal rather than undefined.
    if np.ptp(ys) <= 1e-12:
        out.update(
            {
                "local_min_frac": 0.0,
                "local_min_frac_mean": 0.0,
                "n_basins_est": 0.0,
                "local_min_density": 0.0,
                "mean_basin_size": _safe_float(n),
                "min_frac_k_decay": 0.0,
                "local_max_frac": 0.0,
                "min_depth_mean": 0.0,
                "basin_dispersion": 0.0,
                "min_max_ratio": np.nan,
            }
        )
        return out

    ks = _k_sweep(ctx)
    if not ks:
        return out

    fracs, max_fracs = [], []
    per_k_masks = {}
    try:
        for k in ks:
            idx, _ = ctx.knn(k)
            mn = _local_min_mask(ys, idx)
            mx = _local_max_mask(ys, idx)
            per_k_masks[k] = mn
            fracs.append(float(np.mean(mn)))
            max_fracs.append(float(np.mean(mx)))
    except Exception:
        return out

    fracs_arr = np.asarray(fracs, dtype=float)
    max_fracs_arr = np.asarray(max_fracs, dtype=float)

    # Reference features at the default neighborhood size.
    k0 = ctx.default_k()
    k0 = int(np.clip(k0, 2, max(2, n - 1)))
    try:
        idx0, _ = ctx.knn(k0)
        mask0 = _local_min_mask(ys, idx0)
    except Exception:
        mask0 = per_k_masks[ks[len(ks) // 2]]

    n_min0 = int(np.sum(mask0))
    out["local_min_frac"] = _safe_float(np.mean(mask0))
    out["local_min_frac_mean"] = _safe_float(np.mean(fracs_arr))
    out["local_max_frac"] = _safe_float(np.mean(max_fracs_arr))

    # Estimated basin count: extrapolate the observed local-minimum fraction to the whole space.
    # Averaging over the sweep suppresses noise-driven spurious minima at small k.
    frac_robust = float(np.mean(fracs_arr))
    out["n_basins_est"] = _safe_float(max(1.0, round(frac_robust * n)) if frac_robust > 0 else 0.0)

    # Local-minimum density = minima per sample (already a fraction, kept as its own key for
    # interpretability); mean basin size proxy = samples per basin.
    out["local_min_density"] = _safe_float(frac_robust)
    n_min_robust = frac_robust * n
    out["mean_basin_size"] = _safe_float(n / n_min_robust) if n_min_robust > 1e-9 else _safe_float(n)

    # Decay of the minimum-fraction as k grows: large positive => most minima are shallow noise
    # (fraction collapses with a wider window); ~0 => minima persist => genuine multimodality.
    if fracs_arr.size >= 2 and fracs_arr[0] > 1e-12:
        out["min_frac_k_decay"] = _safe_float((fracs_arr[0] - fracs_arr[-1]) / fracs_arr[0])
    else:
        out["min_frac_k_decay"] = 0.0

    # Symmetry of ruggedness: ratio of local-min to local-max fraction (≈1 for symmetric ripple).
    mm_mean = float(np.mean(max_fracs_arr))
    if mm_mean > 1e-12:
        out["min_max_ratio"] = _safe_float(float(np.mean(fracs_arr)) / mm_mean)
    else:
        out["min_max_ratio"] = np.nan

    # Depth of the default-k minima: how far each sits below its neighborhood mean (in std units).
    try:
        if n_min0 > 0:
            neigh_mean = ys[idx0].mean(axis=1)
            depths = neigh_mean[mask0] - ys[mask0]
            out["min_depth_mean"] = _safe_float(np.mean(depths))
        else:
            out["min_depth_mean"] = 0.0
    except Exception:
        out["min_depth_mean"] = np.nan

    # Spatial dispersion of the detected minima: mean pairwise distance between minima relative to
    # the overall cloud scale. High => basins spread across the domain; low/nan => few or clustered.
    try:
        if n_min0 >= 2:
            D = ctx.distances()
            mi = np.where(mask0)[0]
            sub = D[np.ix_(mi, mi)]
            iu = np.triu_indices(mi.size, k=1)
            mean_min_dist = float(np.mean(sub[iu]))
            overall = float(np.mean(D[np.triu_indices(n, k=1)]))
            out["basin_dispersion"] = _safe_float(mean_min_dist / overall) if overall > 1e-12 else np.nan
        else:
            out["basin_dispersion"] = 0.0
    except Exception:
        out["basin_dispersion"] = np.nan

    return {k: _safe_float(v) for k, v in out.items()}
