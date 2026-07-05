"""Shared context: the precomputed primitives every landscape-criterion family builds on.

A :class:`Context` wraps one labelled point cloud ``(X, y)`` and exposes the common, expensive
building blocks -- normalized inputs, standardized outputs, pairwise distances, k-nearest
neighbors, a fitted global quadratic, local-linear gradients, the gradient-covariance (active
subspace) matrix, and an empirical variogram -- each computed once and cached. Every criterion
module takes a ``Context`` and returns a flat ``{feature_name: value}`` dict, so families stay
independent yet never recompute a shared primitive.

Convention: ``y`` is a minimization objective (lower is better); ``best`` is ``argmin(y)``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]


@dataclass(frozen=True)
class QuadraticFit:
    """A global second-order model ``y ~ c + bᵀx + ½ xᵀ A x`` least-squares-fit on the cloud.

    Attributes:
        intercept: The constant term ``c``.
        linear: The linear coefficients ``b``, shape ``(d,)``.
        hessian: The symmetric curvature matrix ``A``, shape ``(d, d)`` (``A_ii`` from the pure
            square terms, ``A_ij`` from the interaction terms).
        r2: In-sample coefficient of determination of the full quadratic.
        linear_r2: In-sample ``R²`` of a plain linear fit (the curvature-free baseline).
        reliable: ``False`` when there were fewer samples than quadratic coefficients (the fit is
            underdetermined / min-norm and its structure should be trusted only weakly).
    """

    intercept: float
    linear: np.ndarray
    hessian: np.ndarray
    r2: float
    linear_r2: float
    reliable: bool


def _r2(y, y_hat):
    """In-sample R² of a prediction against ``y`` (0 when ``y`` is constant)."""
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-300:
        return 0.0
    return float(1.0 - np.sum((y - y_hat) ** 2) / ss_tot)


class Context:
    """Precomputed, cached primitives for one labelled point cloud ``(X, y)``.

    Args:
        X: Inputs, shape ``(n, d)``.
        y: Outputs, shape ``(n,)`` (minimization objective; lower is better).
        seed: Seed for any randomized feature (random walks, subsampling).
    """

    def __init__(self, X, y, seed=0):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.rng = np.random.default_rng(seed)

        # inputs min-max normalized to [0, 1] per dimension (constant dims -> 0), so distances and
        # rotation are measured in the natural unit box; outputs standardized to zero-mean/unit-var.
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        self.Xn = (X - lo) / span
        sd = float(np.std(y))
        self.ys = (y - float(np.mean(y))) / (sd if sd > 0 else 1.0)

        self.best = int(np.argmin(y))
        self.worst = int(np.argmax(y))

        self._dist = None
        self._knn: dict = {}
        self._quad = None
        self._grad: dict = {}
        self._gradcov: dict = {}

    # -- distances / neighborhoods -------------------------------------------------------------

    def distances(self):
        """The ``(n, n)`` Euclidean distance matrix on the normalized inputs ``Xn`` (cached)."""
        if self._dist is None:
            self._dist = cdist(self.Xn, self.Xn)
        return self._dist

    def default_k(self):
        """A sensible neighborhood size: ``~2*(d+1)`` neighbors, clipped to ``[3, n-1]``."""
        return int(np.clip(2 * (self.d + 1), 3, max(3, self.n - 1)))

    def knn(self, k=None):
        """Return ``(idx, dist)`` of the ``k`` nearest neighbors of each point (self excluded).

        Args:
            k: Neighbors per point; defaults to :meth:`default_k`.

        Returns:
            ``idx`` shape ``(n, k)`` neighbor indices (nearest first), ``dist`` shape ``(n, k)``
            their distances.
        """
        k = int(k or self.default_k())
        k = int(np.clip(k, 1, max(1, self.n - 1)))
        if k not in self._knn:
            D = self.distances().copy()
            np.fill_diagonal(D, np.inf)
            idx = np.argsort(D, axis=1)[:, :k]
            dist = np.take_along_axis(D, idx, axis=1)
            self._knn[k] = (idx, dist)
        return self._knn[k]

    # -- second-order structure ----------------------------------------------------------------

    def quadratic(self):
        """Fit (and cache) the global :class:`QuadraticFit` of ``ys`` on ``Xn``.

        Uses a least-squares (min-norm when underdetermined) solve over the design
        ``[1, xᵢ, xᵢ², xᵢxⱼ]``. Curvature/rotation/separability all read the returned Hessian.
        """
        if self._quad is None:
            self._quad = self._fit_quadratic()
        return self._quad

    def _fit_quadratic(self):
        Xn, ys, d, n = self.Xn, self.ys, self.d, self.n
        cols = [np.ones((n, 1)), Xn, Xn**2]
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        if pairs:
            cols.append(np.stack([Xn[:, i] * Xn[:, j] for i, j in pairs], axis=1))
        design = np.concatenate(cols, axis=1)
        n_feat = design.shape[1]
        coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
        y_hat = design @ coef

        intercept = float(coef[0])
        linear = coef[1 : 1 + d].copy()
        sq = coef[1 + d : 1 + 2 * d]
        A = np.diag(2.0 * sq)
        off = coef[1 + 2 * d :]
        for (i, j), c in zip(pairs, off):
            A[i, j] = A[j, i] = c

        lin_design = np.concatenate([np.ones((n, 1)), Xn], axis=1)
        lin_coef, *_ = np.linalg.lstsq(lin_design, ys, rcond=None)
        linear_r2 = _r2(ys, lin_design @ lin_coef)

        return QuadraticFit(intercept, linear, A, _r2(ys, y_hat), linear_r2, reliable=n >= n_feat)

    def local_gradients(self, k=None):
        """Per-point gradient of ``ys`` estimated by a local linear fit over ``k`` neighbors.

        For each point a linear model is fit to its neighborhood (in ``Xn``), giving a numerical
        gradient without needing the true gradient. Shape ``(n, d)``; the active subspace and the
        gradient-field features build on these. Cached per ``k``.
        """
        k = int(k or max(self.default_k(), self.d + 1))
        k = int(np.clip(k, self.d + 1, max(self.d + 1, self.n - 1)))
        if k in self._grad:
            return self._grad[k]
        idx, _ = self.knn(k)
        G = np.zeros((self.n, self.d))
        for i in range(self.n):
            nb = idx[i]
            dX = self.Xn[nb] - self.Xn[i]
            dy = self.ys[nb] - self.ys[i]
            g, *_ = np.linalg.lstsq(dX, dy, rcond=None)  # through the origin
            G[i] = g
        self._grad[k] = G
        return G

    def gradient_covariance(self, k=None):
        """The ``(d, d)`` active-subspace matrix ``C = mean_i gᵢ gᵢᵀ`` from local gradients.

        Its eigenvectors are the directions the function actually varies along (rotation), and its
        eigenvalue decay is the effective dimensionality. Cached per ``k``.
        """
        k = int(k or max(self.default_k(), self.d + 1))
        if k not in self._gradcov:
            G = self.local_gradients(k)
            self._gradcov[k] = (G.T @ G) / max(1, G.shape[0])
        return self._gradcov[k]

    # -- spatial correlation -------------------------------------------------------------------

    def variogram(self, n_bins=15):
        """Empirical semivariance ``γ(h) = ½·mean[(yᵢ−yⱼ)²]`` vs distance ``h`` (on ``Xn``, ``ys``).

        Args:
            n_bins: Number of distance bins between 0 and the median pairwise distance.

        Returns:
            ``(h, gamma)`` -- bin-center distances and their semivariance (empty bins dropped).
        """
        D = self.distances()
        iu = np.triu_indices(self.n, k=1)
        h = D[iu]
        g = 0.5 * (self.ys[iu[0]] - self.ys[iu[1]]) ** 2
        hi = float(np.median(h)) if h.size else 1.0
        edges = np.linspace(0.0, max(hi, 1e-9), n_bins + 1)
        which = np.digitize(h, edges)
        hs, gs = [], []
        for b in range(1, n_bins + 1):
            m = which == b
            if np.any(m):
                hs.append(0.5 * (edges[b - 1] + edges[b]))
                gs.append(float(np.mean(g[m])))
        return np.array(hs), np.array(gs)
