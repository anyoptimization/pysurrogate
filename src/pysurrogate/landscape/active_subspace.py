"""Active-subspace features: effective dimensionality from the gradient-covariance spectrum."""

import numpy as np

_TINY = 1e-12


def _spectrum(M):
    """Non-negative eigenvalues of a symmetric PSD matrix, sorted descending.

    Args:
        M: A ``(d, d)`` symmetric matrix (the active-subspace / gradient-covariance matrix).

    Returns:
        Eigenvalues clipped at 0 and sorted in descending order, shape ``(d,)``, or ``None`` if
        the decomposition fails or the input is not finite.
    """
    try:
        M = 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)
        if not np.all(np.isfinite(M)):
            return None
        w = np.linalg.eigvalsh(M)
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        return np.sort(w)[::-1]
    except Exception:
        return None


def _participation_ratio(w):
    """Participation ratio ``(Σλ)² / Σλ²`` -- the effective number of active directions.

    Args:
        w: Non-negative eigenvalues.

    Returns:
        A value in ``[1, d]`` (near ``d`` when energy spreads over all directions, near ``1`` when
        a single direction dominates), or ``np.nan`` when the spectrum vanishes.
    """
    s1 = float(np.sum(w))
    s2 = float(np.sum(w**2))
    if s2 <= _TINY or s1 <= _TINY:
        return np.nan
    return float(np.clip(s1 * s1 / s2, 1.0, float(w.size)))


def _energy_dim(w, frac=0.9):
    """Number of leading eigenvalues needed to reach ``frac`` of the cumulative energy.

    Args:
        w: Eigenvalues sorted descending.
        frac: Cumulative-energy threshold in ``(0, 1]``.

    Returns:
        The (integer, as float) count of top directions, or ``np.nan`` when energy vanishes.
    """
    s = float(np.sum(w))
    if s <= _TINY:
        return np.nan
    cum = np.cumsum(w) / s
    return float(int(np.searchsorted(cum, frac) + 1))


def _decay_slope(w):
    """Slope of ``log(λ)`` regressed on rank -- how fast the spectrum decays.

    Args:
        w: Eigenvalues sorted descending.

    Returns:
        A (typically negative) slope; steeper (more negative) means faster decay and a lower
        effective dimension. ``np.nan`` when fewer than two positive eigenvalues exist.
    """
    pos = w[w > _TINY]
    if pos.size < 2:
        return np.nan
    try:
        ranks = np.arange(pos.size, dtype=float)
        slope = float(np.polyfit(ranks, np.log(pos), 1)[0])
        return slope
    except Exception:
        return np.nan


def _spectral_entropy(w):
    """Normalized Shannon entropy of the eigenvalue distribution.

    Args:
        w: Non-negative eigenvalues.

    Returns:
        Entropy in ``[0, 1]`` (1 = energy uniformly spread over all directions -> high effective
        dimension; 0 = one direction carries everything), or ``np.nan`` when ``d < 2`` or the
        spectrum vanishes.
    """
    if w.size < 2:
        return np.nan
    s = float(np.sum(w))
    if s <= _TINY:
        return np.nan
    p = w / s
    p = p[p > _TINY]
    if p.size < 1:
        return np.nan
    ent = -float(np.sum(p * np.log(p)))
    return float(np.clip(ent / np.log(w.size), 0.0, 1.0))


def _gini(x):
    """Gini coefficient of non-negative values (spread/inequality across coordinates).

    Args:
        x: Non-negative values.

    Returns:
        Gini in ``[0, 1]`` (0 = all equal, ->1 = concentrated in one entry), or ``np.nan`` when
        fewer than two entries or the total vanishes.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 2:
        return np.nan
    s = float(np.sum(x))
    if s <= _TINY:
        return np.nan
    idx = np.arange(1, n + 1)
    return float(np.clip((2.0 * np.sum(idx * x)) / (n * s) - (n + 1.0) / n, 0.0, 1.0))


def compute(ctx) -> dict:
    """Active-subspace / effective-dimensionality features (Constantine active subspaces).

    Decomposes the gradient-covariance matrix ``C = mean_i gᵢ gᵢᵀ`` -- whose eigenvectors are the
    directions the function actually varies along -- and reads the *shape* of its eigenvalue
    spectrum to estimate how many dimensions the landscape effectively uses. A nearly flat spectrum
    means every input direction matters (a genuinely high-dimensional, isotropic function); a fast
    decay with one dominant eigenvalue means variation collapses onto a low-dimensional active
    subspace (a ridge/embedded function). Per-coordinate sensitivity spread (from the diagonal of
    ``C``) separates functions where a few inputs dominate from ones where all inputs contribute
    equally.

    Args:
        ctx: The shared :class:`Context` wrapping one labelled point cloud.

    Returns:
        A flat dict of active-subspace features keyed by short names, each a float or ``np.nan``.
    """
    keys = [
        "participation_ratio",
        "intrinsic_dim_frac",
        "energy_dim_90",
        "energy_dim_frac",
        "spectral_decay_slope",
        "top_eig_frac",
        "spectral_entropy",
        "sensitivity_gini",
        "sensitivity_cv",
    ]
    out = {k: np.nan for k in keys}

    try:
        C = ctx.gradient_covariance()
    except Exception:
        return out

    d = int(getattr(ctx, "d", None) or np.asarray(C).shape[0])

    # -- eigenvalue-spectrum (active-subspace) features ----------------------------------------
    w = _spectrum(C)
    if w is not None and w.size >= 1:
        pr = _participation_ratio(w)
        out["participation_ratio"] = pr
        if np.isfinite(pr) and d >= 1:
            out["intrinsic_dim_frac"] = float(np.clip(pr / d, 0.0, 1.0))

        ed = _energy_dim(w, 0.9)
        out["energy_dim_90"] = ed
        if np.isfinite(ed) and d >= 1:
            out["energy_dim_frac"] = float(np.clip(ed / d, 0.0, 1.0))

        out["spectral_decay_slope"] = _decay_slope(w)

        s = float(np.sum(w))
        out["top_eig_frac"] = float(w[0] / s) if s > _TINY else np.nan

        out["spectral_entropy"] = _spectral_entropy(w)

    # -- per-coordinate sensitivity spread (diagonal of C = mean squared gradient per input) ---
    diag = np.clip(np.diag(np.asarray(C, dtype=float)), 0.0, None)
    out["sensitivity_gini"] = _gini(diag)
    if diag.size >= 2:
        m = float(np.mean(diag))
        if m > _TINY:
            out["sensitivity_cv"] = float(np.std(diag) / m)

    return out
