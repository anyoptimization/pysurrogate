"""Generalized least-squares fit of the Dace Kriging model for a fixed theta."""

import numpy as np
from numpy.linalg import LinAlgError

from pysurrogate.dace.corr import calc_kernel_matrix, calc_kernel_tensor


class DaceFitError(Exception):
    """A fit is infeasible at this theta (non-positive-definite R, or ill-conditioned F).

    Raised instead of a bare ``Exception`` so theta optimizers can ``except`` it and
    treat the offending theta as an infinite objective -- rejecting the step instead
    of letting the failure abort the whole search. An infeasible theta should *score
    badly*, not crash.
    """


def fit(X, Y, regr, kernel, theta, noise=0.0):
    """Generalized least-squares fit of the Dace Kriging model at a fixed theta.

    Builds the correlation matrix, Cholesky-factorizes it (with the deliberate ``noise``
    nugget on the unit diagonal), and solves the GLS trend/residual system, returning the
    full model dict consumed by prediction and theta selection.

    Args:
        X: Standardized design sites, shape ``(n, d)``.
        Y: Standardized targets, shape ``(n,)`` or ``(n, q)``.
        regr: Regression trend basis.
        kernel: Correlation kernel.
        theta: Length-scale parameters for the kernel.
        noise: Deliberate noise-to-signal ratio added to the unit diagonal (``0`` interpolates,
            ``>0`` smooths), on top of the always-present machine-epsilon floor.

    Returns:
        The fitted model dict (``C``, ``beta``, ``gamma``, ``obj``, ``_sigma2``, ...).

    Raises:
        DaceFitError: ``R`` is not positive-definite at the requested noise, or ``F`` is too
            ill-conditioned on these design sites.
    """
    # number of sample points (rows of the design matrix)
    n_sample = X.shape[0]

    # baseline float-level jitter against near-singularity, always added -- the invisible
    # numerical floor (machine epsilon scaled by sample size), not a modeling choice
    base = (10 + n_sample) * 2.220446049250313e-16
    R0 = calc_kernel_matrix(X, X, kernel, theta)

    # do the cholesky decomposition. The diagonal carries the DELIBERATE observation
    # `noise` (a noise-to-signal ratio on R's unit diagonal: 0 -> interpolate, >0 ->
    # regression GP that smooths through points), always added on top of the `base`
    # machine-eps floor. There is no silent auto-repair climb: if R is not positive-
    # definite at the requested noise, we raise -- and the theta optimizers read that as
    # an infeasible theta. To regularize a non-PD fit, raise `noise` / `noise_bounds`
    # explicitly; we never add hidden nugget behind the caller's back.
    R = R0 + np.eye(n_sample) * (base + noise)
    try:
        C = np.linalg.cholesky(R)
    except LinAlgError as e:
        raise DaceFitError("R is not positive-definite at the requested noise -- increase noise / noise_bounds.") from e

    # fit the least squares for regression
    F = regr(X)
    Ft = np.linalg.lstsq(C, F, rcond=None)[0]
    Q, G = np.linalg.qr(Ft)
    # rcond is the reciprocal condition number of G (small => ill-conditioned). Guard
    # fires when G is near-singular, i.e. the regression basis is degenerate on these
    # sites; otherwise the bad G would silently corrupt beta and the predictive variance.
    rcond = 1.0 / np.linalg.cond(G)
    if rcond < 1e-15:
        raise DaceFitError("F is too ill conditioned: Poor combination of regression model and design sites")
    Yt = np.linalg.solve(C, Y)
    beta = np.linalg.lstsq(G, Q.T @ Yt, rcond=None)[0]

    # calculate the residual to fit with gaussian process and calculate objective function
    rho = Yt - Ft @ beta
    sigma2 = np.sum(np.square(rho), axis=0) / n_sample
    detR = np.prod(np.power(np.diag(C), (2 / n_sample)))
    obj = np.sum(sigma2) * detR

    # finally gamma to predict values
    gamma = np.linalg.solve(C.T, rho)

    if type(theta) is not np.ndarray:
        theta = np.array([theta])

    return {
        "kernel": kernel,
        "regr": regr,
        "theta": theta,
        "R": R,
        "C": C,
        "F": F,
        "Ft": Ft,
        "Q": Q,
        "G": G,
        "Yt": Yt,
        "beta": beta,
        "rho": rho,
        "_sigma2": sigma2,
        "obj": obj,
        "f": obj,
        "gamma": gamma,
        "noise": noise,
    }


def _cholesky_batch(R):
    """Batched Cholesky with a per-slice feasibility fallback.

    Tries the whole stack at once (the common case, since the optimizers descend from
    feasible starts). If any slice is not positive-definite -- which makes the batched
    call raise -- it falls back to a per-slice loop, marking the non-PD slices
    infeasible instead of failing the whole batch.

    Args:
        R: Stacked correlation matrices, shape ``(J, n, n)``.

    Returns:
        ``(feasible, C)`` -- a boolean mask of the positive-definite slices and the
        lower Cholesky factors (the rows for infeasible slices are unused).
    """
    try:
        return np.ones(R.shape[0], dtype=bool), np.linalg.cholesky(R)
    except LinAlgError:
        feasible = np.zeros(R.shape[0], dtype=bool)
        C = np.zeros_like(R)
        for j in range(R.shape[0]):
            try:
                C[j] = np.linalg.cholesky(R[j])
                feasible[j] = True
            except LinAlgError:
                pass
        return feasible, C


def batch_obj_grad(X, Y, regr, kernel, thetas, noise=0.0, with_grad=True, noise_grad=False):
    """Objective and theta-gradient of the Dace likelihood for a population of theta.

    The batched, lock-step counterpart of a loop over ``fit``: it builds the stacked
    ``(J, n, n)`` correlation tensor and runs one batched Cholesky / GLS solve, so J theta
    candidates cost a single set of LAPACK calls instead of J Python-level fits.
    Non-positive-definite candidates are reported as infeasible (an infinite objective and a
    zero gradient) rather than raising, exactly as a theta search treats them. The objective
    matches ``fit(...)["obj"]`` slice-for-slice, and the gradient is the analytic derivative of
    that likelihood -- this is a pure vectorization of the already-tested ``fit``.

    Args:
        X: Standardized design sites, shape ``(n, d)``.
        Y: Standardized targets, shape ``(n,)`` or ``(n, q)``.
        regr: Regression trend.
        kernel: Correlation kernel.
        thetas: Population of length-scales, shape ``(J, p)``.
        noise: Deliberate noise-to-signal ratio added to the unit diagonal -- a scalar, or a
            per-candidate array ``(J,)`` to score different nugget levels in one batch (the
            learned-nugget path, where ``noise`` is a search coordinate driven by
            ``noise_bounds``). No climbing; a non-PD candidate at this noise is simply infeasible.
        with_grad: Whether to compute the theta-gradient. ``False`` skips the gradient
            work entirely (the dominant cost: forming ``Rinv``, the per-theta kernel
            derivatives, and the trace terms) and returns a zero gradient -- used for the
            cheap objective-only screening of a large candidate pool.
        noise_grad: Also return ``df/d(noise)`` per candidate. The nugget enters as
            ``R = R0(theta) + noise * I``, so its derivative is the ``Rk = I`` special case of
            the theta-gradient: ``df/d(noise) = (detR/n) * (s * tr(Rinv) - sum_q ||gamma_q||^2)``.
            This is what lets an optimizer treat the noise as just another coordinate of the
            search vector. Implies the gradient block (requires ``with_grad``); the extra cost
            is one trace per candidate since ``Rinv`` and ``gamma`` are already formed.

    Returns:
        ``(obj, grad, feasible)`` -- objectives ``(J,)`` (``inf`` where infeasible),
        gradients ``(J, p)`` (``0`` where infeasible, or everywhere if ``with_grad`` is
        False), and the feasibility mask ``(J,)``. When ``noise_grad`` is set, a fourth
        element ``dnoise`` ``(J,)`` (``0`` where infeasible) is appended.
    """
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    J, p = thetas.shape
    n = X.shape[0]
    if Y.ndim == 1:
        Y = Y[:, None]

    obj = np.full(J, np.inf)
    grad = np.zeros((J, p))
    dnoise = np.zeros(J)

    # underdetermined trend (more regression-basis columns than design points): there is no GLS
    # fit at ANY theta, and a nugget regularizes R, not F -- so report every candidate infeasible
    # rather than letting the QR/solve raise. Honors the never-raise contract for a degenerate
    # regression x design (e.g. QuadraticRegression on too few points).
    if regr(X).shape[1] > n:
        feas = np.zeros(J, dtype=bool)
        return (obj, grad, feas, dnoise) if noise_grad else (obj, grad, feas)

    base = (10 + n) * 2.220446049250313e-16
    # noise may be a scalar (same nugget for all) or a per-candidate (J,) array; reshape so
    # it broadcasts onto the (J, n, n) stack as a diagonal add.
    diag = (base + np.asarray(noise, dtype=float)).reshape(-1, 1, 1)
    R = calc_kernel_tensor(X, X, kernel, thetas) + np.eye(n) * diag

    feasible, C = _cholesky_batch(R)
    idx = np.flatnonzero(feasible)
    if idx.size == 0:
        return (obj, grad, feasible, dnoise) if noise_grad else (obj, grad, feasible)

    # work on the feasible subset only -- a singular C would break the batched solves
    R, C, th = R[idx], C[idx], thetas[idx]
    m = idx.size

    F = regr(X)
    Fb = np.broadcast_to(F, (m, n, F.shape[1]))
    Yb = np.broadcast_to(Y, (m, n, Y.shape[1]))

    # C is the lower-triangular Cholesky factor, so np.linalg.solve is an exact triangular
    # solve; the GLS below mirrors fit() but stacked over the population.
    Ft = np.linalg.solve(C, Fb)
    Q, G = np.linalg.qr(Ft)
    Yt = np.linalg.solve(C, Yb)
    beta = np.linalg.solve(G, np.swapaxes(Q, 1, 2) @ Yt)
    rho = Yt - Ft @ beta

    sigma2 = np.sum(np.square(rho), axis=1) / n  # (m, q)
    detR = np.prod(np.diagonal(C, axis1=1, axis2=2) ** (2.0 / n), axis=1)  # (m,)
    s = np.sum(sigma2, axis=1)  # (m,)
    obj[idx] = s * detR

    if with_grad:
        # gradient: df/d(theta_k) = (detR/n) * (s * tr(Rinv Rk) - sum_q g^T Rk g).
        gamma = np.linalg.solve(np.swapaxes(C, 1, 2), rho)  # (m, n, q)
        # Rinv = C^-T C^-1, reusing the Cholesky we already have rather than refactorizing
        # R from scratch (np.linalg.inv would redo an LU). Formed once per member and reused
        # for all p theta-components, where each trace term is then a cheap elementwise sum --
        # so one O(n^3) inverse amortizes over p, cheaper than a triangular solve per k.
        Cinv = np.linalg.solve(C, np.broadcast_to(np.eye(n), C.shape))
        Rinv = np.swapaxes(Cinv, 1, 2) @ Cinv  # (m, n, n)

        if noise_grad:
            # df/d(noise) is the Rk = I special case: tr(Rinv * I) = tr(Rinv), and the
            # quadratic term sum_q gamma_q^T I gamma_q = sum_q ||gamma_q||^2.
            tr_Rinv = np.einsum("mii->m", Rinv)  # (m,)
            gamma_sq = np.sum(gamma**2, axis=(1, 2))  # sum over n and q, (m,)
            dnoise[idx] = (detR / n) * (s * tr_Rinv - gamma_sq)

        D = np.repeat(X, n, axis=0) - np.tile(X, (n, 1))  # (n*n, d), the kernel-matrix layout
        # Loop over the (small) population rather than pre-stacking a single (m, n, n, p)
        # derivative tensor -- that tensor grows as m*n^2*p (tens of MB by n=120) and the
        # allocation/cache traffic, not the FLOPs, was the cost. Each member touches only
        # its own (n, n, p) slice; both contractions over p stay vectorized per member.
        g = np.zeros((m, p))
        for j in range(m):
            dKj = kernel.theta_grad(D, th[j]).reshape(n, n, p)
            tr = np.einsum("ab,abk->k", Rinv[j], dKj, optimize=True)  # tr(Rinv Rk); both symmetric
            quad = np.einsum("nq,nmk,mq->k", gamma[j], dKj, gamma[j], optimize=True)  # sum_q g^T Rk g
            g[j] = (detR[j] / n) * (s[j] * tr - quad)
        grad[idx] = g

    # a non-PD pocket survived as a non-finite objective (e.g. a near-singular G): drop it
    # the same way -- infeasible, zero gradient -- so callers never see NaN.
    bad = ~np.isfinite(obj)
    grad[bad] = 0.0
    dnoise[bad] = 0.0
    feasible = feasible & ~bad
    return (obj, grad, feasible, dnoise) if noise_grad else (obj, grad, feasible)
