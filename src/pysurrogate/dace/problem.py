"""DaceProblem: the DACE likelihood as a generic Problem, with the nugget folded into the vector."""

import numpy as np

from pysurrogate.core.optimizer import Evaluation, Problem
from pysurrogate.core.parameter import Log10, Parameter, ParameterSpace
from pysurrogate.dace.fit import batch_obj_grad, fit
from pysurrogate.dace.prior import resolve_prior

_LN10 = np.log(10.0)
_FLOOR = 1e-12  # keep log10 of a zero/near-zero bound finite
# finite log10 window used to seed starts when a theta bound is infinite (an unbounded search):
# sample length-scales in [1e-3, 1e3], then let the local descent leave the window upward.
_SAMPLE_LO, _SAMPLE_HI = -3.0, 3.0


class DaceProblem(Problem):
    """The DACE profile-likelihood as a backend-free :class:`Problem` over ``log10`` parameters.

    The search vector is ``x = [log10 theta_1, ..., log10 theta_p]`` and -- when a nugget is
    learned -- one extra coordinate ``log10 noise`` appended on the end. This is the concrete
    payoff of "noise is just another coordinate": with ``noise_bounds`` set, every generic
    optimizer learns the nugget jointly with the length-scales through the *same* vector and the
    *same* analytic gradient (the nugget's derivative is the ``Rk = I`` term from
    :func:`~pysurrogate.dace.fit.batch_obj_grad`); with ``noise_bounds=None`` the coordinate is
    absent and the fixed ``noise`` (``0`` for ``None``) is used.

    ``__call__`` evaluates a whole population in one batched GLS solve and **never raises** -- a
    non-positive-definite candidate comes back ``feasible=False`` with ``f=+inf`` -- so an
    ill-conditioned region is something the search steps around, not something that crashes it.

    Args:
        X: Standardized design sites, shape ``(n, d)``.
        Y: Standardized targets, shape ``(n,)`` or ``(n, q)``.
        regr: Regression trend.
        kernel: Correlation kernel (must expose ``theta_grad`` for the analytic gradient).
        theta_bounds: ``(lo, hi)`` length-scale bounds in *original* (not log) space, each
            broadcastable to the ``p`` optimized length-scales. An infinite ``hi`` (e.g.
            ``np.inf``) makes the search *unbounded above* in that coordinate -- the length-scale
            stays positive (floored) but has no ceiling; starts are still seeded from a finite
            window (see :attr:`sampling_bounds`).
        noise: The fixed nugget when not learning it. ``None`` (default) fixes it at ``0`` and does
            **not** optimize over it (exact interpolation, the MATLAB regime); a float fixes it at
            that value. Ignored when ``noise_bounds`` is given.
        noise_bounds: ``(lo, hi)`` to *learn* the nugget as an extra log-space coordinate of the
            search vector, or ``None`` (default) to keep ``noise`` fixed. Learning by likelihood
            alone tends to drive the nugget back toward 0 on clean data -- it pays off mainly with
            genuinely noisy data or validation selection.
        theta_prior: ``(mean, lam)`` for a MAP prior on the length-scales, or ``None`` (default) for
            pure maximum likelihood. When set, ``lam * sum((log10(theta) - mean)**2)`` is added to
            the objective (and its analytic gradient) -- a Gaussian prior on the *encoded* (log10)
            length-scales only (never the nugget coordinate), regularizing the search toward
            ``10**mean`` and away from short-length-scale over-fitting. ``None`` -> zero penalty, so
            the objective is bit-for-bit the plain likelihood.
    """

    def __init__(self, X, Y, regr, kernel, theta_bounds, noise=None, noise_bounds=None, theta_prior=None):
        self.X, self.Y = X, Y
        self.regr, self.kernel = regr, kernel

        # optional MAP prior on the encoded length-scales: (mean, lam) or None for pure MLE.
        self._prior = resolve_prior(theta_prior)  # None, or a Prior (a (mean, lam) tuple -> GaussianPrior)

        # the componentwise differences are theta-independent, so build them once here and reuse
        # them across every objective/gradient evaluation of the search (instead of rebuilding the
        # (n*n, d) matrix per call). Passing this one array also lets a reducing kernel (KPLS) cache
        # its per-fit distance projection, keyed on this exact object.
        n = X.shape[0]
        self._D = np.repeat(X, n, axis=0) - np.tile(X, (n, 1))

        # The search layout is the kernel's own declaration: the concatenated length-scale / shape
        # coordinates it exposes via parameters(d), sized from the data dimensionality. The user's
        # theta_bounds are distributed across those coordinates by position (in original space, with
        # the machine-eps floor), then the nugget is appended as one more coordinate when learned.
        # This replaces the old hard-coded "[theta..., noise]" assumption -- e.g. GeneralizedExponential
        # declares its own trailing `power` coordinate here instead of it being smuggled into theta.
        lo, hi = theta_bounds
        lo = np.maximum(np.atleast_1d(np.asarray(lo, float)), _FLOOR)
        hi = np.atleast_1d(np.asarray(hi, float))
        tlo, thi = np.broadcast_arrays(lo, hi)
        p_total = len(thi)

        # the length-scale ("fill") coordinates are caller-sized -- their ARD count comes from the
        # supplied bounds, not the kernel's ard flag -- so the one fill parameter absorbs whatever is
        # left after the kernel's fixed shape coordinates (e.g. GeneralizedExponential's exponent).
        kparams = kernel.parameters(X.shape[1])
        fixed = sum(kp.size for kp in kparams if not kp.fill)
        params, i = [], 0
        for kp in kparams:
            size = (p_total - fixed) if kp.fill else kp.size
            params.append(
                Parameter(
                    kp.name,
                    size=size,
                    bounds=(tlo[i : i + size], thi[i : i + size]),
                    encoding=kp.encoding,
                    fill=kp.fill,
                )
            )
            i += size
        if i != p_total:
            raise ValueError(f"theta_bounds has {p_total} coordinates but the kernel declares {i}.")
        self.p = p_total
        self._theta_names = [p.name for p in params]

        # noise_bounds set -> learn the nugget as a trailing coordinate; else fixed (None -> 0).
        self.learn_noise = noise_bounds is not None
        if self.learn_noise:
            nlo, nhi = noise_bounds
            self.noise_bounds = (max(float(nlo), _FLOOR), float(nhi))  # value-space, for start clamping
            params.append(Parameter("noise", size=1, bounds=self.noise_bounds, encoding=Log10()))
            self.noise = None
        else:
            self.noise_bounds = None
            self.noise = 0.0 if noise is None else float(noise)

        self.space = ParameterSpace(params)

    @property
    def bounds(self):
        # the encoded (log10) coordinate bounds for the whole search vector, assembled by the
        # ParameterSpace from each parameter's value-space bounds and encoding (+inf survives where a
        # theta coordinate is unbounded above).
        return self.space.bounds()

    @property
    def sampling_bounds(self):
        # finite seeding region: clamp any infinite hard bound to the default window, leave all
        # finite bounds untouched (so a fully bounded problem samples exactly as before).
        lo, hi = self.bounds
        lo = np.where(np.isfinite(lo), lo, _SAMPLE_LO)
        hi = np.where(np.isfinite(hi), hi, _SAMPLE_HI)
        return lo, hi

    def decode(self, x):
        """Split a log10 search vector into ``(theta, noise)`` in original space.

        Args:
            x: A point in the search space, shape ``(p,)`` or ``(p + 1,)``.

        Returns:
            ``(theta, noise)`` -- the length-scale vector and the scalar nugget (the fixed
            ``noise`` when this problem does not learn it).
        """
        values = self.space.decode(x)
        # the kernel consumes its length-scale / shape coordinates as one positional vector, in the
        # order they were declared; the nugget (when learned) is the trailing coordinate.
        theta = np.concatenate([np.atleast_1d(values[name]) for name in self._theta_names])
        noise = float(values["noise"][0]) if self.learn_noise else self.noise
        return theta, float(noise)

    def fit(self, x):
        """Commit a full :func:`~pysurrogate.dace.fit.fit` at the parameters encoded by ``x``."""
        theta, noise = self.decode(x)
        return fit(self.X, self.Y, self.regr, self.kernel, theta, noise=noise)

    def _prior_penalty(self, Z):
        """The MAP prior's penalty per candidate over the log10 length-scales ``Z``.

        Args:
            Z: The encoded length-scale coordinates of the population, shape ``(J, p)``.

        Returns:
            The per-candidate penalty ``(J,)``, or ``0.0`` when no prior is set.
        """
        return 0.0 if self._prior is None else self._prior.penalty(Z)

    def screen(self, X):
        """Cheap objective-only ranking (no gradient) -- the fast path for Restart's screen."""
        X = np.atleast_2d(np.asarray(X, float))
        thetas = 10.0 ** X[:, : self.p]
        noise = 10.0 ** X[:, self.p] if self.learn_noise else self.noise
        obj, _, _ = batch_obj_grad(
            self.X, self.Y, self.regr, self.kernel, thetas, noise=noise, with_grad=False, D=self._D
        )
        # add the MAP penalty so the screen ranks by the SAME objective the gradient path polishes
        # (infeasible candidates keep obj=+inf: inf + finite penalty = inf).
        return obj + self._prior_penalty(X[:, : self.p])

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        thetas = 10.0 ** X[:, : self.p]
        if self.learn_noise:
            noise = 10.0 ** X[:, self.p]  # per-candidate nugget, (J,)
            obj, g_theta, feasible, dnoise = batch_obj_grad(
                self.X, self.Y, self.regr, self.kernel, thetas, noise=noise, with_grad=True, noise_grad=True, D=self._D
            )
        else:
            obj, g_theta, feasible = batch_obj_grad(
                self.X, self.Y, self.regr, self.kernel, thetas, noise=self.noise, with_grad=True, D=self._D
            )

        # chain rule from the original parameter to its log10 coordinate: d/d(log10 v) = v*ln10 * d/dv
        grad = g_theta * thetas * _LN10
        if self.learn_noise:
            grad = np.hstack([grad, (dnoise * noise * _LN10)[:, None]])

        # MAP prior on the encoded length-scales: add the prior's penalty to the objective and its
        # gradient directly in log10 space (the search coordinate). Applied to the theta coordinates
        # only, and only on feasible rows (infeasible keep obj=+inf, grad=0). The Prior object supplies
        # the penalty and its gradient (e.g. GaussianPrior's ridge lam*(z-mean)^2).
        if self._prior is not None:
            Z = X[:, : self.p]
            obj = obj + self._prior.penalty(Z)
            grad[:, : self.p] += feasible[:, None] * self._prior.grad(Z)

        return Evaluation(f=obj, feasible=feasible, grad=grad, info=None)
