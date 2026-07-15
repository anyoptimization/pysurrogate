"""Radial basis function (RBF) surrogate model with an analytic gradient."""

import numpy as np

from pysurrogate.core.kernel import (
    CubicRadial,
    Gaussian,
    LinearRadial,
    Multiquadric,
    ThinPlateSpline,
    calc_kernel_matrix,
    pairwise_diffs,
)
from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.core.regression import (
    ConstantRegression,
    LinearRegression,
    QuadraticRegression,
)

# RBF now shares the framework's one kernel zoo (:mod:`pysurrogate.core.kernel`) -- the same kernel
# objects the Dace engine uses, evaluated on coordinate differences ``D`` (not a private
# scalar-distance path). The radial bases are conditionally positive-definite interpolation kernels;
# ``gaussian`` reuses the covariance Gaussian directly (so ``sigma`` is its length-scale). ``sigma``
# maps to the kernel's ``theta`` shape parameter; the pure-power bases ignore it.
_KERNELS = {
    "linear": LinearRadial(),
    "cubic": CubicRadial(),
    "gaussian": Gaussian(),
    "mq": Multiquadric(),
    "tps": ThinPlateSpline(),
}

# the pure-power bases depend on the radius only -- sigma is ignored, so a sigma grid search
# would fit the identical model 30 times over; these skip straight to the single fixed fit.
_SIGMA_FREE = frozenset({"linear", "cubic", "tps"})

# The polynomial tail is the shared core regression basis -- one implementation of "build a
# polynomial design matrix P(X) (+ gradient)" for the whole framework. "quadratic" and
# "linear+quadratic" both resolve to the full quadratic basis (intercept + linear + pairwise).
_TAILS = {
    None: None,
    "constant": ConstantRegression(),
    "linear": LinearRegression(),
    "quadratic": QuadraticRegression(),
    "linear+quadratic": QuadraticRegression(),
}


def _tail_basis(tail):
    """Resolve a tail name to its core regression basis (``None`` for no tail)."""
    if tail not in _TAILS:
        raise ValueError(f"Unknown tail: {tail!r}.")
    return _TAILS[tail]


class RBF(Model):
    """RBF interpolant: a kernel expansion over the training points plus a polynomial tail.

    The kernel is one of the shared :mod:`pysurrogate.core.kernel` bases (``linear``/``cubic`` radial
    powers, ``gaussian`` covariance, ``mq`` multiquadric, ``tps`` thin-plate spline), evaluated on the
    coordinate differences between points. ``sigma`` is the kernel's shape parameter (the Gaussian
    length-scale, the multiquadric offset); the pure-power bases ignore it. With ``tune_sigma=True``
    ``sigma`` is selected by minimizing leave-one-out cross-validation error over a small grid.
    """

    def __init__(
        self,
        kernel="tps",
        tail="linear",
        sigma=1.0,
        rho=1e-6,
        normalized=False,
        tune_sigma=False,
        optimize=None,
        **kwargs,
    ) -> None:
        # default duplicate elimination on, but via setdefault so a user override (e.g.
        # eliminate_duplicates=False) does not collide with an explicit positional pass-through.
        kwargs.setdefault("eliminate_duplicates", True)
        kwargs.setdefault("eliminate_duplicates_eps", 1e-8)
        super().__init__(**kwargs)
        self.tail = tail
        self.rho = rho
        self.sigma = sigma
        self.normalized = normalized
        # whether the fit searches the sigma grid. `tune_sigma` is the current name; `optimize=` is
        # its former spelling, kept as a back-compat alias so it no longer visually collides with the
        # separate fit-time `optimize=` lever (which switches the search off per-fit).
        self.tune_sigma = tune_sigma if optimize is None else optimize

        if kernel not in _KERNELS:
            raise ValueError(f"Unknown kernel function: {kernel!r}. Choose one of {sorted(_KERNELS)}.")
        self.kernel_name = kernel
        self.kernel = _KERNELS[kernel]

    def _fit(self, X, y, optimize=True, **kwargs):
        if y.shape[1] != 1:
            raise ValueError(f"RBF supports a single output, got {y.shape[1]}; fit one model per output.")
        rho, tail, kernel, sigma, normalized = self.rho, self.tail, self.kernel, self.sigma, self.normalized

        # tune sigma only when the model is configured to, the fit-time flag allows it, AND the
        # kernel actually uses sigma -- the pure-power bases ignore it, so a grid would re-fit the
        # identical model 30 times (optimize=False also forces the cheap fixed-sigma screening fit).
        if self.tune_sigma and optimize and self.kernel_name not in _SIGMA_FREE:
            sigmas = np.linspace(0.0001, 20, 30)
            models = [rbf_fit(X, y, kernel, sigma=s, tail=tail, rho=rho, normalized=normalized) for s in sigmas]
            f = np.array([model["loocv"] for model in models])
            cond = np.array([model["cond"] for model in models])
            f[cond > 1e12] = np.inf
            self.model = models[f.argmin()]
        else:
            self.model = rbf_fit(X, y, kernel, tail=tail, rho=rho, sigma=sigma, normalized=normalized)

    def _predict(self, X, var=False, grad=False):
        g = rbf_grad(self.model, X) if grad else None
        return Prediction(y=rbf_predict(self.model, X), grad=g)


def rbf_kernel(X, phi, tail="linear", **kwargs):
    """Append the polynomial-tail columns for ``tail`` to the kernel block ``phi``."""
    basis = _tail_basis(tail)
    P = np.zeros((X.shape[0], 0)) if basis is None else basis(X)
    return np.column_stack([phi, P])


def rbf_fit(X, y, kernel, Xp=None, sigma=1.0, tail="linear", rho=0.0, normalized=False):
    """Solve the RBF + polynomial-tail system and return the fitted model dict.

    Args:
        X: Training inputs, shape ``(n, d)``.
        y: Training targets, shape ``(n, 1)``.
        kernel: A :mod:`pysurrogate.core.kernel` kernel evaluated on coordinate differences.
        Xp: Kernel centers; defaults to ``X`` (a full interpolant).
        sigma: The kernel shape parameter (passed as the kernel's ``theta``).
        tail: Polynomial-tail name (see :data:`_TAILS`).
        rho: Ridge added to the kernel diagonal (``rho**2``) to regularize a near-singular system.
        normalized: Whether to column-normalize the kernel block by its column sums.

    Returns:
        The fitted model dict consumed by :func:`rbf_predict` and :func:`rbf_grad`.
    """
    if Xp is None:
        Xp = X

    phi = calc_kernel_matrix(X, Xp, kernel, sigma)

    phi_norm = 1.0
    if normalized:
        phi_norm = phi.sum(axis=0)[None, :]
        phi = phi / phi_norm

    if rho is not None:
        phi = phi + np.eye(len(phi)) * (rho**2)

    K = rbf_kernel(X, phi, tail=tail)
    n, m = K.shape

    lhs = np.zeros((m, m))
    lhs[:n, :m] = K
    lhs[n:, :n] = K[:, n:].T

    rhs = np.zeros((m, 1))
    rhs[:n] = y

    A_inv, cond = svd_inv(lhs)
    coef = A_inv @ rhs

    c, Kinv = coef[:n, 0], A_inv[:n, :n]
    e = c / (np.diag(Kinv) + 1e-128)
    loocv = (e**2).sum()
    gcv = (c**2).sum() / (np.diag(Kinv).mean() ** 2)

    return dict(
        X=X, kernel=kernel, sigma=sigma, tail=tail, phi_norm=phi_norm, coef=coef, cond=cond, e=e, loocv=loocv, gcv=gcv
    )


def rbf_predict(model, X):
    """Evaluate a fitted RBF model at the query points ``X``."""
    phi = calc_kernel_matrix(X, model["X"], model["kernel"], model["sigma"])
    phi = phi / model["phi_norm"]
    phi = rbf_kernel(X, phi, tail=model["tail"])
    return phi @ model["coef"]


def svd_inv(A):
    """Pseudo-inverse of ``A`` via SVD, returning ``(A_inv, condition_number)``.

    Singular values below a relative tolerance are truncated (their reciprocal set to 0), so a
    near-singular system yields a stable least-norm solution instead of amplifying round-off through
    ``1/S``. A well-conditioned system keeps every singular value, so the result is unchanged there.
    The reported ``cond`` is the full untruncated ratio (the sigma-grid uses it to reject fits).
    """
    U, S, V = np.linalg.svd(A)
    cond = np.abs(np.max(S)) / np.abs(np.min(S))
    # numpy's default pinv cutoff: keep S_i whose ratio to S_max exceeds machine-eps * max(shape).
    tol = S.max() * max(A.shape) * np.finfo(float).eps
    S_inv = np.where(S > tol, 1.0 / S, 0.0)
    A_inv = V.T @ np.diag(S_inv) @ U.T
    return A_inv, cond


def _tail_grad(X, tail, c_tail):
    """Gradient of the polynomial tail ``P(x) @ c_tail`` w.r.t. ``x``, shape ``(m, d)``.

    Delegates to the shared core basis: ``basis.grad(X)`` is the per-row ``(d, p)`` Jacobian,
    contracted with the tail coefficients ``c_tail``.
    """
    basis = _tail_basis(tail)
    if basis is None:
        return np.zeros(X.shape)
    return basis.grad(X) @ c_tail  # (m, d, p) @ (p,) -> (m, d)


def rbf_grad(model, X):
    """Analytic gradient of the RBF prediction w.r.t. the query points, shape ``(m, d)``.

    The kernel expansion contributes ``sum_i w_i * grad phi(x - x_i)``, where ``grad phi`` is the
    shared kernel's own spatial gradient and ``w_i`` folds in the fit coefficient and any column
    normalization; the polynomial tail adds its basis Jacobian.
    """
    Xc = model["X"]
    kernel = model["kernel"]
    sigma = model["sigma"]
    m, d = X.shape
    n = Xc.shape[0]

    # componentwise differences in the calc_kernel_matrix layout, so kernel.grad gives d phi / d x
    # for every (query, center) pair; reshape to (m, n, d) and contract with the center weights.
    D = pairwise_diffs(X, Xc)  # (m*n, d)
    G = kernel.grad(D, sigma).reshape(m, n, d)

    coef = model["coef"][:, 0]
    w = coef[:n] / np.asarray(model["phi_norm"]).ravel()  # center weights incl. column normalization
    grad = np.einsum("n,mnd->md", w, G)

    return grad + _tail_grad(X, model["tail"], coef[n:])
