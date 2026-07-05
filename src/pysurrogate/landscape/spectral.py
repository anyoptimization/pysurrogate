"""Spectral-signature features: graph-Laplacian eigen-decomposition of the output signal."""

import numpy as np

_KEYS = [
    "rayleigh",
    "spectral_centroid",
    "low_energy_frac",
    "high_energy_frac",
    "spectral_entropy",
    "dominant_freq",
    "spectral_rolloff",
    "participation_ratio",
]


def _nan_out():
    """Return the full feature dict with every value set to ``np.nan``."""
    return {k: float("nan") for k in _KEYS}


def _build_laplacian(ctx):
    """Build a symmetric normalized graph Laplacian from a k-NN graph on ``ctx.Xn``.

    Edges connect each point to its ``k`` nearest neighbors with Gaussian (heat-kernel) weights
    scaled by the median neighbor distance; the adjacency is symmetrized by union (``max``) so the
    graph stays connected. The returned operator is ``L = I - D^{-1/2} W D^{-1/2}`` with eigenvalues
    in ``[0, 2]``.

    Args:
        ctx: A landscape ``Context``.

    Returns:
        The ``(n, n)`` normalized Laplacian as a float array, or ``None`` when it cannot be built.
    """
    try:
        n = int(ctx.n)
        if n < 3:
            return None
        k = int(ctx.default_k())
        k = int(np.clip(k, 1, max(1, n - 1)))
        idx, dist = ctx.knn(k)
        idx = np.asarray(idx)
        dist = np.asarray(dist, dtype=float)
        if idx.size == 0:
            return None

        # bandwidth: median of finite, positive neighbor distances
        finite = dist[np.isfinite(dist)]
        pos = finite[finite > 0]
        sigma = float(np.median(pos)) if pos.size > 0 else 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        W = np.zeros((n, n), dtype=float)
        rows = np.repeat(np.arange(n), idx.shape[1])
        cols = idx.ravel()
        d = dist.ravel()
        w = np.exp(-(d**2) / (2.0 * sigma**2))
        w[~np.isfinite(w)] = 0.0
        W[rows, cols] = w
        # symmetrize by union so a directed k-NN edge makes both endpoints neighbors
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0.0)

        deg = W.sum(axis=1)
        if not np.all(np.isfinite(deg)):
            return None
        # isolated nodes -> self-normalize to identity row (Laplacian value 1 on diagonal)
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
        S = (W * inv_sqrt[:, None]) * inv_sqrt[None, :]
        L = np.eye(n) - S
        # enforce exact symmetry against float drift
        L = 0.5 * (L + L.T)
        if not np.all(np.isfinite(L)):
            return None
        return L
    except Exception:
        return None


def _spectrum(L, ys):
    """Eigen-decompose the Laplacian and project the unit-energy output signal onto its modes.

    Args:
        L: The ``(n, n)`` symmetric normalized Laplacian.
        ys: The standardized output vector, shape ``(n,)``.

    Returns:
        ``(w, energy)`` where ``w`` are eigenvalues ascending and ``energy`` the squared projection
        coefficients of ``ys/||ys||`` onto the eigenvectors (summing to ~1), or ``None`` on failure.
    """
    try:
        norm = float(np.linalg.norm(ys))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        w, U = np.linalg.eigh(L)
        order = np.argsort(w)
        w = np.clip(w[order].astype(float), 0.0, 2.0)
        U = U[:, order]
        coeffs = U.T @ (ys / norm)
        energy = coeffs**2
        s = float(energy.sum())
        if not np.isfinite(s) or s <= 1e-12:
            return None
        energy = energy / s
        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(w)):
            return None
        return w, energy
    except Exception:
        return None


def compute(ctx) -> dict:
    """Compute spectral-graph-signal features of a labelled point cloud.

    Builds a k-NN graph on the normalized inputs, forms the symmetric normalized graph Laplacian,
    and analyzes how the energy of the standardized output signal distributes across the Laplacian
    eigenmodes (graph "frequencies"). Low-frequency energy means a smooth function that varies
    slowly across the neighborhood graph; high-frequency energy means a rugged/multimodal signal.

    Args:
        ctx: A landscape ``Context`` exposing ``Xn``, ``ys``, ``knn`` and ``default_k``.

    Returns:
        A flat dict mapping feature names to floats (or ``np.nan`` where undefined):
        ``rayleigh``, ``spectral_centroid``, ``low_energy_frac``, ``high_energy_frac``,
        ``spectral_entropy``, ``dominant_freq``, ``spectral_rolloff``, ``participation_ratio``.
    """
    out = _nan_out()

    L = _build_laplacian(ctx)
    if L is None:
        return out
    try:
        ys = np.asarray(ctx.ys, dtype=float).ravel()
    except Exception:
        return out

    spec = _spectrum(L, ys)
    if spec is None:
        return out
    w, energy = spec
    m = w.size
    if m < 2:
        return out

    wmax = float(w[-1])
    wmax = wmax if wmax > 1e-12 else 1.0

    # --- Rayleigh quotient ys^T L ys / ys^T ys = energy-weighted mean eigenvalue (0 smooth -> 2 rugged)
    try:
        out["rayleigh"] = float(np.sum(w * energy))
    except Exception:
        pass

    # --- spectral centroid: Rayleigh normalized to [0, 1] by the largest graph frequency
    try:
        if np.isfinite(out["rayleigh"]):
            out["spectral_centroid"] = float(np.clip(out["rayleigh"] / wmax, 0.0, 1.0))
    except Exception:
        pass

    # Drop the trivial (near-zero) mode so fractions describe genuine variation, not the DC offset.
    try:
        nontrivial = w > 1e-9 * wmax
        if int(np.count_nonzero(nontrivial)) >= 2:
            w_nz = w[nontrivial]
            e_nz = energy[nontrivial]
            es = float(e_nz.sum())
            e_nz = e_nz / es if es > 1e-12 else e_nz
        else:
            w_nz, e_nz = w, energy
    except Exception:
        w_nz, e_nz = w, energy

    mm = w_nz.size

    # --- low-frequency energy fraction: signal energy in the smoothest 20% of the spectrum
    try:
        cut = max(1, int(np.ceil(0.2 * mm)))
        out["low_energy_frac"] = float(np.clip(np.sum(e_nz[:cut]), 0.0, 1.0))
    except Exception:
        pass

    # --- high-frequency energy fraction: signal energy in the ruggedest 20% of the spectrum
    try:
        cut = max(1, int(np.ceil(0.2 * mm)))
        out["high_energy_frac"] = float(np.clip(np.sum(e_nz[-cut:]), 0.0, 1.0))
    except Exception:
        pass

    # --- spectral entropy: Shannon entropy of the energy spread, normalized to [0, 1].
    #     ~0 when a single frequency dominates (structured/periodic), ~1 when energy is uniform (noise).
    try:
        p = e_nz[e_nz > 0]
        if p.size >= 2:
            ent = -float(np.sum(p * np.log(p)))
            out["spectral_entropy"] = float(np.clip(ent / np.log(mm), 0.0, 1.0))
        elif p.size == 1:
            out["spectral_entropy"] = 0.0
    except Exception:
        pass

    # --- dominant frequency: normalized eigenvalue of the single most energetic mode (0 smooth->1 rugged)
    try:
        j = int(np.argmax(e_nz))
        out["dominant_freq"] = float(np.clip(w_nz[j] / wmax, 0.0, 1.0))
    except Exception:
        pass

    # --- spectral rolloff: normalized frequency below which 85% of the signal energy accumulates
    try:
        order = np.argsort(w_nz)
        cum = np.cumsum(e_nz[order])
        hit = np.where(cum >= 0.85)[0]
        if hit.size > 0:
            out["spectral_rolloff"] = float(np.clip(w_nz[order][hit[0]] / wmax, 0.0, 1.0))
    except Exception:
        pass

    # --- participation ratio: effective fraction of modes carrying signal (inverse Simpson / m).
    #     Small -> energy concentrated in a few frequencies (low-rank/periodic structure);
    #     large -> energy spread across many frequencies (broadband / noisy landscape).
    try:
        denom = float(np.sum(e_nz**2))
        if denom > 1e-12:
            out["participation_ratio"] = float(np.clip((1.0 / denom) / mm, 0.0, 1.0))
    except Exception:
        pass

    return out
