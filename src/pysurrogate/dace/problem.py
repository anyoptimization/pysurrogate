"""DaceProblem: the DACE likelihood as a generic Problem, with the nugget folded into the vector."""

import numpy as np

from pysurrogate.core.optimizer import Evaluation, Problem
from pysurrogate.dace.fit import batch_obj_grad, fit

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
    """

    def __init__(self, X, Y, regr, kernel, theta_bounds, noise=None, noise_bounds=None):
        self.X, self.Y = X, Y
        self.regr, self.kernel = regr, kernel

        lo, hi = theta_bounds
        lo = np.maximum(np.atleast_1d(np.asarray(lo, float)), _FLOOR)
        hi = np.atleast_1d(np.asarray(hi, float))
        self._tlo, self._thi = np.broadcast_arrays(lo, hi)
        self.p = len(self._thi)

        # noise_bounds set -> learn the nugget as a coordinate; else fixed (None -> 0).
        self.learn_noise = noise_bounds is not None
        if self.learn_noise:
            nlo, nhi = noise_bounds
            self._nlo, self._nhi = max(float(nlo), _FLOOR), float(nhi)
            self.noise = None
        else:
            self.noise = 0.0 if noise is None else float(noise)

    @property
    def bounds(self):
        lo = np.log10(self._tlo)
        hi = np.log10(self._thi)  # +inf where a theta coordinate is unbounded above
        if self.learn_noise:
            lo = np.append(lo, np.log10(self._nlo))
            hi = np.append(hi, np.log10(self._nhi))
        return lo, hi

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
        x = np.atleast_1d(np.asarray(x, float))
        theta = 10.0 ** x[: self.p]
        noise = 10.0 ** x[self.p] if self.learn_noise else self.noise
        return theta, float(noise)

    def fit(self, x):
        """Commit a full :func:`~pysurrogate.dace.fit.fit` at the parameters encoded by ``x``."""
        theta, noise = self.decode(x)
        return fit(self.X, self.Y, self.regr, self.kernel, theta, noise=noise)

    def screen(self, X):
        """Cheap objective-only ranking (no gradient) -- the fast path for Restart's screen."""
        X = np.atleast_2d(np.asarray(X, float))
        thetas = 10.0 ** X[:, : self.p]
        noise = 10.0 ** X[:, self.p] if self.learn_noise else self.noise
        obj, _, _ = batch_obj_grad(self.X, self.Y, self.regr, self.kernel, thetas, noise=noise, with_grad=False)
        return obj

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        thetas = 10.0 ** X[:, : self.p]
        if self.learn_noise:
            noise = 10.0 ** X[:, self.p]  # per-candidate nugget, (J,)
            obj, g_theta, feasible, dnoise = batch_obj_grad(
                self.X, self.Y, self.regr, self.kernel, thetas, noise=noise, with_grad=True, noise_grad=True
            )
        else:
            obj, g_theta, feasible = batch_obj_grad(
                self.X, self.Y, self.regr, self.kernel, thetas, noise=self.noise, with_grad=True
            )

        # chain rule from the original parameter to its log10 coordinate: d/d(log10 v) = v*ln10 * d/dv
        grad = g_theta * thetas * _LN10
        if self.learn_noise:
            grad = np.hstack([grad, (dnoise * noise * _LN10)[:, None]])

        return Evaluation(f=obj, feasible=feasible, grad=grad, info=None)
