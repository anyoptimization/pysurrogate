"""Generic bounded L-BFGS-B optimizer over a Problem, with optional multi-start."""

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from pysurrogate.core.optimizer import Optimizer

# returned to scipy for an infeasible point: a large but finite penalty (np.inf breaks
# L-BFGS-B's line search and finite-difference gradients), so the search steps away from it.
_INFEASIBLE = 1e25


class _Stop(Exception):
    """Internal signal: the callback asked to stop, unwind out of scipy's loop."""


class LBFGS(Optimizer):
    """Bounded quasi-Newton (L-BFGS-B) over any :class:`Problem`.

    A *local* optimizer: from a warm start it converges in a few evaluations, which makes it
    the natural choice for a refit. When the problem returns an analytic gradient it is used as
    the exact Jacobian (what makes it fast); otherwise scipy falls back to a finite-difference
    gradient. The problem is evaluated one point at a time (``J = 1``).

    Multi-start is delegated to :class:`~pysurrogate.core.sampling.Sampling` rather than being a
    bespoke L-BFGS feature: pass a ``sampling`` and L-BFGS runs one local descent from each
    sampled start, the ``x0`` (when known) force-included so the warm start always competes; the
    *callback* keeps the best across all of them. With no ``sampling``, it is a single descent
    from ``x0`` (or the box center if ``x0`` is unknown). One iteration is one full local descent,
    so multi-start is many iterations -- which lets a driver race it.

    Args:
        sampling: A :class:`~pysurrogate.core.sampling.Sampling` that generates the starts, or
            ``None`` for a single start. The DACE-style likelihood is multi-modal, so a cold fit
            benefits from ``Sampling(8, LHS())``.
        random_state: Seed for the sampling, so multi-start runs are reproducible.
        options: Options forwarded to ``scipy.optimize.minimize(method="L-BFGS-B")``. The
            defaults ``{"gtol": 1e-6, "ftol": 1e-9, "maxfun": 200}`` let the polish actually
            converge: the DACE log-likelihood is often very flat near the optimum (a small
            gradient over a long valley), so a loose ``gtol`` would stop the descent almost
            immediately and leave the result to the screen. Anything passed here overrides or
            extends them.
    """

    def __init__(self, sampling=None, random_state=0, options=None):
        super().__init__()
        self.sampling = sampling
        self.random_state = random_state
        self.options = {"gtol": 1e-6, "ftol": 1e-9, "maxfun": 200, **(options or {})}

    def _setup(self):
        # one ITERATION here is one full local descent from one start; multi-start = many iters.
        lo, hi, _, _ = self._box()
        # scipy L-BFGS-B wants None (not +/-inf) for an absent bound -- an inf passed through can
        # stall the bounded descent -- so translate each non-finite hard bound to None.
        self._bounds = [
            (lo_i if np.isfinite(lo_i) else None, hi_i if np.isfinite(hi_i) else None) for lo_i, hi_i in zip(lo, hi)
        ]
        # seed starts from the FINITE sampling region (an infinite hard box cannot be sampled);
        # the descent below is still free to leave it, constrained only by the hard bounds.
        self._starts = list(self._seed_starts(self.sampling, self.random_state))
        self._next = 0
        self._jac = self.problem.has_grad

    def _fun(self, x):
        X = np.atleast_2d(np.asarray(x, float))
        ev = self.problem(X)
        self.n_evals += 1
        f, feasible = float(ev.f[0]), bool(ev.feasible[0])
        grad = ev.grad[0] if ev.grad is not None else None
        # only feasible candidates reach the callback (selection sees real fits only); an
        # infeasible point is handed back to scipy as a finite penalty so it retreats.
        if self._emit_batch(X, ev):
            raise _Stop
        if grad is None:
            return f if feasible else _INFEASIBLE
        return (f if feasible else _INFEASIBLE), (grad if feasible else np.zeros_like(grad))

    def _advance(self):
        if self._next >= len(self._starts):  # no starts at all (e.g. an empty sampling)
            self.message = "converged"
            return False
        s = self._starts[self._next]
        self._next += 1
        try:
            minimize(self._fun, s, method="L-BFGS-B", jac=self._jac, bounds=self._bounds, options=self.options)
        except _Stop:
            return False  # callback asked to stop; _emit already flagged is_done
        if self._next >= len(self._starts):
            self.message = "converged"
            return False
        return True  # more starts -> more iterations
