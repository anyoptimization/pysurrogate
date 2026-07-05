"""Generic derivative-free pattern (compass) search over a bounded Problem."""

import numpy as np

from pysurrogate.core.optimizer import Optimizer


class PatternSearch(Optimizer):
    """Generalized pattern (compass) search -- derivative-free, batch-friendly, robust.

    At each iteration it probes the ``2p`` axial neighbors ``x +/- step * e_i`` in one batched
    problem evaluation, moves to the best improving neighbor (growing the step on success) and
    otherwise shrinks the step. It uses no gradient, so it works on any problem -- including one
    whose objective is noisy or non-smooth -- and converges to a stationary point on smooth
    objectives as the step contracts. This is the generic, backend-free descendant of the
    classic DACE box-min search.

    Selection and early stopping are the callback's job: every feasible probe is reported, so
    the callback keeps the best (by likelihood, validation, ...) and may stop the search.

    Args:
        init_step: Initial step as a fraction of each coordinate's box width. Default ``0.25``.
        shrink: Step contraction factor on an unsuccessful iteration. Default ``0.5``.
        grow: Step expansion factor on a successful iteration. Default ``1.0`` (no growth).
        tol: Stop once the (fractional) step shrinks below this. Default ``1e-4``.
        max_iter: Hard cap on iterations (each is one batched evaluation). Default ``200``.
    """

    def __init__(self, init_step=0.25, shrink=0.5, grow=1.0, tol=1e-4, max_iter=200):
        super().__init__()
        self.init_step = init_step
        self.shrink = shrink
        self.grow = grow
        self.tol = tol
        self.max_iter = max_iter

    def _setup(self):
        # hard bounds clip the moves (may be +/-inf); the finite sampling region sets the step
        # scale and the center fallback, since an infinite box has no width or midpoint.
        lo, hi, slo, shi = self._box()
        self._lo, self._hi, self._width, self._p = lo, hi, shi - slo, len(lo)
        # no warm start -> begin at the sampling-region center (this is a local search; it needs a point)
        self._x = 0.5 * (slo + shi) if self.x0 is None else np.clip(self.x0, lo, hi)
        self._step = self.init_step
        # evaluate the incumbent; if it is infeasible we still march -- a neighbor may be feasible.
        ev = self.problem(np.atleast_2d(self._x))
        self.n_evals += 1
        self._f = float(ev.f[0])
        if bool(ev.feasible[0]):
            self._emit(self._x, self._f, ev.info[0] if ev.info is not None else None)
        self.message = "converged (step < tol)"

    def _advance(self):
        # one iteration = one poll of the 2p axial neighbors, scored in a single batched call
        if self._step < self.tol:
            return False
        if self.n_iter >= self.max_iter:
            self.message = "stopped (max_iter)"
            return False

        offsets = np.vstack([np.eye(self._p), -np.eye(self._p)]) * (self._step * self._width)
        cand = np.clip(self._x + offsets, self._lo, self._hi)
        ev = self.problem(cand)
        self.n_evals += len(cand)

        best_f, best_x = self._f, None
        for i in range(len(cand)):
            if not bool(ev.feasible[i]):
                continue
            fi = float(ev.f[i])
            if self._emit(cand[i], fi, ev.info[i] if ev.info is not None else None):
                return False
            if fi < best_f:
                best_f, best_x = fi, cand[i]

        if best_x is not None:
            self._x, self._f = best_x, best_f  # successful poll -> move, optionally grow the step
            self._step *= self.grow
        else:
            self._step *= self.shrink  # no improvement -> contract toward the incumbent
        return self._step >= self.tol
