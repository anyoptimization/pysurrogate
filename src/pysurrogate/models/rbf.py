"""Radial basis function (RBF) surrogate model with an analytic gradient."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.util.dist import calc_dist


class RBF(Model):
    """RBF interpolant: a kernel expansion over the training points plus a polynomial tail.

    The kernel argument is the *squared* distance ``calc_dist`` returns. With ``optimize=True``
    the shape parameter ``sigma`` is selected by minimizing leave-one-out cross-validation
    error over a small grid.
    """

    def __init__(
        self, kernel="tps", tail="linear", sigma=1.0, rho=1e-6, normalized=False, optimize=False, **kwargs
    ) -> None:
        super().__init__(eliminate_duplicates=True, eliminate_duplicates_eps=1e-8, **kwargs)
        self.tail = tail
        self.rho = rho
        self.sigma = sigma
        self.normalized = normalized
        self.optimize = optimize

        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel function: {kernel!r}. Choose one of {sorted(KERNELS)}.")
        self.kernel = KERNELS[kernel]

    def _fit(self, X, y, optimize=True, **kwargs):
        rho, tail, kernel, sigma, normalized = self.rho, self.tail, self.kernel, self.sigma, self.normalized

        # tune sigma only when the model is configured to AND the fit-time flag allows it, so
        # optimize=False forces the cheap fixed-sigma fit for model-selection screening (like Kriging)
        if self.optimize and optimize:
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
    n, _ = X.shape

    if tail is None:
        P = np.zeros((n, 0))
    elif tail == "constant":
        P = np.ones((n, 1))
    elif tail == "linear":
        P = np.column_stack((np.ones(n), X))
    elif tail == "quadratic":
        P = np.column_stack((np.ones(n), X**2))
    elif tail == "linear+quadratic":
        P = np.column_stack((np.ones(n), X, X**2))
    else:
        raise ValueError(f"Unknown tail: {tail!r}.")

    return np.column_stack([phi, P])


def rbf_fit(X, y, func, Xp=None, rho=0.0, normalized=False, **kwargs):
    """Solve the RBF + polynomial-tail system and return the fitted model dict."""
    if Xp is None:
        Xp = X

    phi = func(calc_dist(X, Xp), **kwargs)

    phi_norm = 1.0
    if normalized:
        phi_norm = phi.sum(axis=0)[None, :]
        phi = phi / phi_norm

    if rho is not None:
        phi = phi + np.eye(len(phi)) * (rho**2)

    K = rbf_kernel(X, phi, **kwargs)
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

    return dict(X=X, cond=cond, e=e, loocv=loocv, gcv=gcv, coef=coef, func=func, phi_norm=phi_norm, kwargs=kwargs)


def rbf_predict(model, X):
    """Evaluate a fitted RBF model at the query points ``X``."""
    phi = model["func"](calc_dist(X, model["X"]), **model["kwargs"])
    phi = phi / model["phi_norm"]
    phi = rbf_kernel(X, phi, **model["kwargs"])
    return phi @ model["coef"]


def kernel_linear(r, sigma=1.0, **kwargs):
    """Linear RBF kernel ``sigma * r`` over the squared distance ``r``."""
    return sigma * r


def kernel_quadratic(r, sigma=1.0, **kwargs):
    """Quadratic RBF kernel ``(sigma * r)**2`` over the squared distance ``r``."""
    return (sigma * r) ** 2


def kernel_cubic(r, sigma=1.0, **kwargs):
    """Cubic RBF kernel ``(sigma * r)**3`` over the squared distance ``r``."""
    return (sigma * r) ** 3


def kernel_gaussian(r, sigma=None, **kwargs):
    """Gaussian-shaped RBF kernel ``exp(-sigma * r**2)`` over the squared distance ``r``."""
    return np.exp(-(sigma * r**2))


def kernel_periodic(r, sigma=1.0, **kwargs):
    """Periodic RBF kernel (fixed period) over the squared distance ``r``."""
    return (sigma**2) * np.exp(-2 * np.sin((np.pi * r) / 5) ** 2)


def kernel_multi_quadr(r, sigma=1.0, **kwargs):
    """Multiquadric RBF kernel ``sqrt(r**2 + sigma**2)`` over the squared distance ``r``."""
    return ((r**2) + (sigma**2)) ** 0.5


def kernel_tps(r, **kwargs):
    """Thin-plate-spline RBF kernel ``r**2 * log(r)`` (clamped) over the squared distance ``r``."""
    r = np.where(r < np.finfo(float).eps, np.finfo(float).eps, r)
    return (r**2) * np.log(r)


KERNELS = {
    "linear": kernel_linear,
    "quadratic": kernel_quadratic,
    "cubic": kernel_cubic,
    "gaussian": kernel_gaussian,
    "mq": kernel_multi_quadr,
    "tps": kernel_tps,
    "periodic": kernel_periodic,
}


def svd_inv(A):
    """Pseudo-inverse of ``A`` via SVD, returning ``(A_inv, condition_number)``."""
    U, S, V = np.linalg.svd(A)
    A_inv = V.T @ np.diag(1 / S) @ U.T
    cond = np.abs(np.max(S)) / np.abs(np.min(S))
    return A_inv, cond


# --- analytic gradient -------------------------------------------------------------------
# The kernels take r = calc_dist(...) which is the *squared* Euclidean distance D = ||x-xi||^2.
# So each dkernel below is d(kernel)/dD, and the chain to the query point is dD/dx_j =
# 2 (x_j - xi_j) -- no 1/r singularity at a center.


def dkernel_linear(r, sigma=1.0, **kwargs):
    """Derivative ``d(kernel_linear)/dD`` w.r.t. the squared distance ``r``."""
    return np.full_like(r, float(sigma))


def dkernel_quadratic(r, sigma=1.0, **kwargs):
    """Derivative ``d(kernel_quadratic)/dD`` w.r.t. the squared distance ``r``."""
    return 2 * sigma**2 * r


def dkernel_cubic(r, sigma=1.0, **kwargs):
    """Derivative ``d(kernel_cubic)/dD`` w.r.t. the squared distance ``r``."""
    return 3 * sigma**3 * r**2


def dkernel_gaussian(r, sigma=None, **kwargs):
    """Derivative ``d(kernel_gaussian)/dD`` w.r.t. the squared distance ``r``."""
    return -2 * sigma * r * np.exp(-(sigma * r**2))


def dkernel_periodic(r, sigma=1.0, **kwargs):
    """Derivative ``d(kernel_periodic)/dD`` w.r.t. the squared distance ``r``."""
    return -(sigma**2) * (2 * np.pi / 5) * np.sin(2 * np.pi * r / 5) * np.exp(-2 * np.sin((np.pi * r) / 5) ** 2)


def dkernel_multi_quadr(r, sigma=1.0, **kwargs):
    """Derivative ``d(kernel_multi_quadr)/dD`` w.r.t. the squared distance ``r``."""
    return r / np.sqrt((r**2) + (sigma**2))


def dkernel_tps(r, **kwargs):
    """Derivative ``d(kernel_tps)/dD`` w.r.t. the squared distance ``r``."""
    r = np.where(r < np.finfo(float).eps, np.finfo(float).eps, r)
    return 2 * r * np.log(r) + r


KERNEL_GRADS = {
    kernel_linear: dkernel_linear,
    kernel_quadratic: dkernel_quadratic,
    kernel_cubic: dkernel_cubic,
    kernel_gaussian: dkernel_gaussian,
    kernel_periodic: dkernel_periodic,
    kernel_multi_quadr: dkernel_multi_quadr,
    kernel_tps: dkernel_tps,
}


def _tail_grad(X, tail, c_tail):
    """Gradient of the polynomial tail ``P(x) @ c_tail`` w.r.t. ``x``, shape ``(m, d)``."""
    m, d = X.shape
    g = np.zeros((m, d))
    if tail == "linear":
        g = g + c_tail[1 : 1 + d][None, :]
    elif tail == "quadratic":
        g = g + 2 * X * c_tail[1 : 1 + d][None, :]
    elif tail == "linear+quadratic":
        g = g + c_tail[1 : 1 + d][None, :] + 2 * X * c_tail[1 + d : 1 + 2 * d][None, :]
    # tail is None or "constant" -> constant term, zero gradient
    return g


def rbf_grad(model, X):
    """Analytic gradient of the RBF prediction w.r.t. the query points, shape ``(m, d)``."""
    Xc = model["X"]
    kwargs = model["kwargs"]
    n = Xc.shape[0]

    diff = X[:, None, :] - Xc[None, :, :]  # (m, n, d)
    D = (diff**2).sum(axis=2)  # (m, n) squared distance = the kernel argument

    dpsi = KERNEL_GRADS[model["func"]](D, **kwargs)  # (m, n) d(kernel)/dD
    coef = model["coef"][:, 0]

    # center term: sum_i (c_i / phi_norm_i) * psi'(D) * dD/dx, with dD/dx = 2 * diff
    w = (coef[:n][None, :] / model["phi_norm"]) * dpsi  # (m, n)
    grad = 2.0 * np.einsum("mn,mnd->md", w, diff)

    return grad + _tail_grad(X, kwargs.get("tail"), coef[n:])
