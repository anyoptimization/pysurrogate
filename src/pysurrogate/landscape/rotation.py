"""Rotation & anisotropy features: eigenframe alignment of the Hessian and active subspace."""

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

_TINY = 1e-12
_ISO_EPS = 0.05  # below this anisotropy the spectrum is ~isotropic and rotation is undefined


def _sym_eig(M):
    """Symmetric eigendecomposition of ``M`` returning ascending eigenvalues and unit columns.

    Args:
        M: A ``(d, d)`` matrix; it is symmetrized before decomposition.

    Returns:
        ``(w, V)`` where ``w`` are eigenvalues (ascending) and ``V`` has the corresponding
        unit eigenvectors as columns, or ``(None, None)`` if the decomposition fails.
    """
    try:
        M = 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)
        if not np.all(np.isfinite(M)):
            return None, None
        w, V = np.linalg.eigh(M)
        return w, V
    except Exception:
        return None, None


def _weighted_offaxis(w, V):
    """Eigenvalue-weighted off-axis energy ``1 - Σ_i ŵ_i max_j V[j,i]²`` of the eigenframe.

    Each eigenvector's alignment to its nearest coordinate axis is ``max_j V[j,i]²`` (1 when
    axis-aligned, ``1/d`` when maximally tilted). Weighting by ``|eigenvalue|`` emphasizes the
    directions that actually carry curvature/variation.

    Args:
        w: Eigenvalues, shape ``(d,)``.
        V: Eigenvectors as columns, shape ``(d, d)``.

    Returns:
        Off-axis energy in ``[0, 1)``; 0 for a perfectly axis-aligned frame.
    """
    d = V.shape[1]
    if d < 2:
        return 0.0
    maxsq = np.max(V**2, axis=0)  # per eigenvector (column)
    a = np.abs(w)
    s = float(np.sum(a))
    weights = a / s if s > _TINY else np.full(d, 1.0 / d)
    return float(np.clip(1.0 - float(np.sum(weights * maxsq)), 0.0, 1.0))


def _assignment_offaxis(V):
    """Off-axis energy under the optimal one-to-one axis<->eigenvector assignment.

    The squared-component matrix ``P[j,i] = V[j,i]²`` is doubly (near-)stochastic; the Hungarian
    assignment picks the best axis for each eigenvector without double-counting an axis (a flaw of
    the plain ``max``). The complement of the mean matched weight is the residual rotation.

    Args:
        V: Eigenvectors as columns, shape ``(d, d)``.

    Returns:
        Assignment-based off-axis energy in ``[0, 1)``; 0 for an (up to permutation) axis-aligned
        frame.
    """
    d = V.shape[1]
    if d < 2:
        return 0.0
    try:
        P = V**2
        ri, ci = linear_sum_assignment(-P)
        return float(np.clip(1.0 - float(np.mean(P[ri, ci])), 0.0, 1.0))
    except Exception:
        return np.nan


def _anisotropy(w):
    """Spectral spread ``(|λ|max - |λ|min) / (|λ|max + |λ|min)`` of the eigenvalues.

    Args:
        w: Eigenvalues, shape ``(d,)``.

    Returns:
        Anisotropy in ``[0, 1]`` (0 = isotropic/sphere, ->1 = one dominant direction), or
        ``np.nan`` when ``d < 2`` or the spectrum vanishes.
    """
    a = np.abs(np.asarray(w, dtype=float))
    if a.size < 2:
        return np.nan
    lmax, lmin = float(np.max(a)), float(np.min(a))
    if lmax + lmin <= _TINY:
        return np.nan
    return float((lmax - lmin) / (lmax + lmin))


def _rot_score(w, V):
    """Isotropy-aware rotation score: assignment off-axis, gated to ``NaN`` on an isotropic spectrum.

    Rotation is undefined for a sphere (any frame is an eigenframe), so the score is returned as
    ``NaN`` when anisotropy is below :data:`_ISO_EPS` or the spectrum is degenerate. Otherwise it is
    the assignment-based off-axis energy scaled by how anisotropic the spectrum is.

    Args:
        w: Eigenvalues, shape ``(d,)``.
        V: Eigenvectors as columns, shape ``(d, d)``.

    Returns:
        A rotation score in ``[0, 1]``, or ``np.nan`` when rotation is undefined.
    """
    aniso = _anisotropy(w)
    if not np.isfinite(aniso) or aniso < _ISO_EPS:
        return np.nan
    off = _assignment_offaxis(V)
    if not np.isfinite(off):
        return np.nan
    return float(np.clip(off * aniso, 0.0, 1.0))


def _dominant_alignment(wa, Va, wb, Vb):
    """Squared cosine between the dominant (largest-``|λ|``) eigenvectors of two frames.

    Measures whether the Hessian and the active-subspace matrix agree on the principal direction
    of the landscape (1 = same axis, 0 = orthogonal).

    Args:
        wa: Eigenvalues of the first frame.
        Va: Eigenvectors (columns) of the first frame.
        wb: Eigenvalues of the second frame.
        Vb: Eigenvectors (columns) of the second frame.

    Returns:
        ``cos²`` between the two dominant eigenvectors in ``[0, 1]``, or ``np.nan`` if undefined.
    """
    try:
        va = Va[:, int(np.argmax(np.abs(wa)))]
        vb = Vb[:, int(np.argmax(np.abs(wb)))]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na <= _TINY or nb <= _TINY:
            return np.nan
        c = float(np.dot(va, vb) / (na * nb))
        return float(np.clip(c * c, 0.0, 1.0))
    except Exception:
        return np.nan


def compute(ctx) -> dict:
    """Rotation & anisotropy features from the Hessian and active-subspace eigenframes.

    Decomposes both the global quadratic Hessian and the gradient-covariance (active subspace)
    matrix, then quantifies how far each eigenframe is rotated away from the coordinate axes and how
    anisotropic its spectrum is. Together these separate the three canonical shapes: a **sphere**
    (isotropic -> anisotropy ~0, rotation ``NaN``), an **axis-aligned ellipsoid** (anisotropic but
    off-axis ~0, rotation ~0), and a **rotated ellipsoid** (anisotropic *and* high off-axis energy,
    rotation ->1). A cross-frame alignment feature checks whether curvature and variation agree on
    the principal direction.

    Args:
        ctx: The shared :class:`Context` wrapping one labelled point cloud.

    Returns:
        A flat dict of rotation/anisotropy features keyed by short names, each a float or ``np.nan``.
    """
    keys = [
        "hess_offaxis",
        "hess_aniso",
        "hess_rot",
        "grad_offaxis",
        "grad_aniso",
        "grad_rot",
        "rot_align",
        "rot_consensus",
    ]
    out = {k: np.nan for k in keys}

    # -- Hessian eigenframe --------------------------------------------------------------------
    wh = Vh = None
    try:
        q = ctx.quadratic()
        wh, Vh = _sym_eig(q.hessian)
        if wh is not None and float(np.max(np.abs(wh))) > _TINY:
            out["hess_offaxis"] = _weighted_offaxis(wh, Vh)
            out["hess_aniso"] = _anisotropy(wh)
            out["hess_rot"] = _rot_score(wh, Vh)
    except Exception:
        pass

    # -- Active-subspace (gradient-covariance) eigenframe --------------------------------------
    wg = Vg = None
    try:
        C = ctx.gradient_covariance()
        wg, Vg = _sym_eig(C)
        if wg is not None and float(np.max(np.abs(wg))) > _TINY:
            out["grad_offaxis"] = _weighted_offaxis(wg, Vg)
            out["grad_aniso"] = _anisotropy(wg)
            out["grad_rot"] = _rot_score(wg, Vg)
    except Exception:
        pass

    # -- Cross-frame agreement & consensus -----------------------------------------------------
    try:
        if wh is not None and wg is not None:
            out["rot_align"] = _dominant_alignment(wh, Vh, wg, Vg)
    except Exception:
        pass

    try:
        vals = [v for v in (out["hess_rot"], out["grad_rot"]) if np.isfinite(v)]
        out["rot_consensus"] = float(np.mean(vals)) if vals else np.nan
    except Exception:
        pass

    return out
