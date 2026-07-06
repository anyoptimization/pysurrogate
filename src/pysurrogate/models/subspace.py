"""Active-subspace estimation: a Mahalanobis rotation learned from the data's gradient covariance."""

import numpy as np
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

from pysurrogate.core.kernel import Mahalanobis
from pysurrogate.core.model import Model
from pysurrogate.dace import Dace, LinearRegression


def _standardize(X):
    """Z-score the columns of ``X`` (``ddof=1``), matching the Dace engine; constant columns -> /1."""
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0, ddof=1)
    return (X - m) / np.where(s > 0, s, 1.0)


def _local_gradients(Xs, ys, k):
    """Per-point local-linear gradient of ``ys`` over the ``k`` nearest neighbors in ``Xs``.

    For each point a linear model (through the origin, on the neighbor *differences*) is fit to its
    neighborhood, giving a numerical gradient without the true gradient. Shape ``(n, d)``.
    """
    n = Xs.shape[0]
    D = cdist(Xs, Xs)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    G = np.zeros_like(Xs)
    for i in range(n):
        nb = idx[i]
        g, *_ = np.linalg.lstsq(Xs[nb] - Xs[i], ys[nb] - ys[i], rcond=None)
        G[i] = g
    return G


def active_subspace(X, y, n_components=None, k=None):
    """Estimate the active subspace of ``(X, y)`` -- the directions the function varies along most.

    Standardizes ``X`` to zero-mean/unit-variance (``ddof=1``) -- the space the Dace/Kriging engine
    fits in -- estimates a per-point local-linear gradient over the ``k`` nearest neighbors, forms
    the gradient-covariance matrix ``C = mean_i g_i g_iᵀ`` (Constantine's active-subspace matrix),
    and returns its leading eigenvectors (the rotation) with the full eigenvalue spectrum (the
    variation energy per direction). The eigenvectors feed :class:`~pysurrogate.core.kernel.Mahalanobis`;
    the spectrum's decay is the effective dimensionality (few large eigenvalues -> a low-dim ridge).

    Args:
        X: Inputs, shape ``(n, d)``.
        y: Outputs, shape ``(n,)`` or ``(n, 1)``.
        n_components: Number of leading directions to return; ``None`` returns all ``d`` (a full
            rotation). A smaller value restricts the metric to the top active subspace (rank ``h``).
        k: Neighbors for the local-linear gradient; defaults to ``min(2*(d+1), n-1)`` floored at
            ``d+1`` (a determined local fit needs at least ``d`` neighbors).

    Returns:
        ``(A, eigvals)`` -- ``A`` shape ``(d, n_components)`` with orthonormal columns ordered by
        decreasing variation energy, and ``eigvals`` the full descending ``(d,)`` spectrum of ``C``
        (clipped to ``>= 0``).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    k = int(np.clip(int(k or min(2 * (d + 1), n - 1)), min(d + 1, n - 1), max(1, n - 1)))

    Xs = _standardize(X)
    sy = np.std(y)
    ys = (y - np.mean(y)) / (sy if sy > 0 else 1.0)

    G = _local_gradients(Xs, ys, k)
    C = (G.T @ G) / max(1, n)
    C = 0.5 * (C + C.T)  # symmetrize against round-off before the symmetric eigensolver

    w, V = np.linalg.eigh(C)  # ascending
    order = np.argsort(w)[::-1]  # by decreasing energy
    w, V = np.clip(w[order], 0.0, None), V[:, order]

    h = int(np.clip(n_components or d, 1, d))
    return V[:, :h], w


class RotatedKriging(Model):
    """Kriging over a Mahalanobis metric whose rotation is *learned from the data*.

    Ordinary ARD Kriging can only stretch the coordinate axes -- it cannot represent a function whose
    variation runs along an off-axis (rotated) direction, or that lives on a low-dimensional active
    subspace of the inputs. This backend removes the "supply the rotation yourself" step of
    :class:`~pysurrogate.core.kernel.Mahalanobis`: at fit time it estimates the active subspace of the
    training data (:func:`active_subspace`, the eigenvectors of the gradient-covariance matrix) and
    fits a Kriging model whose metric is rotated into that frame. With ``n_components < d`` the metric
    is restricted to the top active subspace -- a rotated, low-rank cousin of :class:`KPLS`.

    Mechanically it mirrors :class:`KPLS`: the rotation is a data-dependent reparameterization
    computed in the engine's standardized space, and the ``Dace`` engine then fits an ordinary
    length-scale search over the ``n_components`` rotated coordinates -- so the likelihood, predictive
    variance, and gradients are unchanged. The rotation is fixed at the first fit and reused on refit.

    Args:
        regr: Regression trend (default :class:`LinearRegression`).
        n_components: Rotated coordinates to keep; ``None`` keeps all ``d`` (a full rotation).
        k: Neighbors for the local-gradient estimate (see :func:`active_subspace`).
        theta: Starting length-scale for every rotated coordinate.
        theta_bounds: ``(lo, hi)`` length-scale bounds, or ``None`` for an unbounded search.
        theta_prior: Optional ``(mean, lam)`` MAP prior on the length-scales (see ``Dace``).
    """

    def __init__(
        self, regr=None, n_components=None, k=None, theta=1.0, theta_bounds=(0.0, 100.0), theta_prior=None, **kwargs
    ) -> None:
        super().__init__(eliminate_duplicates=True, **kwargs)
        self.regr = regr if regr is not None else LinearRegression()
        self.n_components = n_components
        self.k = k
        self.theta = theta
        self.theta_bounds = theta_bounds
        self.theta_prior = theta_prior

    def _engine(self, X, y):
        """Build the ``Dace`` engine with a data-estimated Mahalanobis rotation and a length-``h`` theta."""
        A, _ = active_subspace(X, y, self.n_components, self.k)
        h = A.shape[1]
        theta = np.full(h, self.theta)
        theta_bounds = self.theta_bounds
        if theta_bounds is not None:
            lo, hi = theta_bounds
            theta_bounds = (np.full(h, lo), np.full(h, hi))
        return Dace(
            regr=self.regr, corr=Mahalanobis(A), theta=theta, theta_bounds=theta_bounds, theta_prior=self.theta_prior
        )

    def _fit(self, X, y, optimize=True, **kwargs):
        self.model = self._engine(X, y)
        self.model.fit(X, y, optimize=optimize)

    def _refit(self, X, y, optimize=True):
        # incremental warm-started re-fit at the fixed rotation; the generic Model.refit handles the
        # out-of-sample scoring and record. optimize warm-starts / freezes the length-scale search.
        self.model.refit(X, y, optimize=optimize)

    def _predict(self, X, var=False, grad=False):
        return self.model.predict(X, var=var, grad=grad)
