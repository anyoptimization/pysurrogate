"""Generic Hooke & Jeeves pattern search -- the MATLAB DACE Boxmin algorithm over any Problem."""

import numpy as np

from pysurrogate.core.optimizer import Optimizer

_LOG10_2 = np.log10(2.0)


class Boxmin(Optimizer):
    """Hooke & Jeeves pattern search -- a faithful, backend-free port of MATLAB DACE's Boxmin.

    The original Boxmin searches the length-scale multiplicatively (``theta *= D``). On a
    log-space :class:`~pysurrogate.core.optimizer.Problem` -- such as ``DaceProblem``, whose
    coordinates are ``log10 theta`` -- the equivalent *additive* moves (``x += log10(D)``)
    reproduce that exact trajectory, since multiplying ``theta`` by ``D`` is adding ``log10(D)``
    to ``log10(theta)``. So this single generic optimizer reproduces the original DACE Boxmin
    bit-for-bit on the log-space DACE problem while remaining a plain ``Optimizer`` over any
    bounded problem.

    The step schedule (``D = 2 ** (k / (p + 2))``), the explore/move pattern and the per-sweep
    step updates all match Boxmin exactly; the only difference is additive-in-coordinate vs
    multiplicative-in-theta, which the log transform makes identical. Every evaluated candidate is
    reported to the callback (which owns selection); the full visited trajectory is kept on
    the contract attribute ``self.visited`` (declared and reset on the :class:`Optimizer` base)
    for inspection (e.g. the golden theta-trajectory snapshots).
    """

    requires_x0 = True  # a pattern search needs a starting point (the warm theta)

    def _setup(self):
        lo, hi = (np.atleast_1d(np.asarray(b, float)) for b in self.problem.bounds)
        self._lo, self._hi = lo, hi
        p = len(lo)
        self._p = p
        x = np.clip(np.array(self.x0, float), lo, hi)
        # additive per-coordinate step == log10(Boxmin's D), with D = 2 ** (arange(1, p+1) / (p+2))
        s = (np.arange(1, p + 1) / (p + 2)) * _LOG10_2
        eq = lo == hi  # an equality bound pins that coordinate (Boxmin's D[ee] = 1, theta = upper)
        s[eq] = 0.0
        x[eq] = hi[eq]
        self._s = s
        self._ne = np.flatnonzero(s != 0.0)  # coordinates that are actually searched
        # self.visited is reset by Optimizer.setup() before this hook runs; we just append to it.
        # relocate the start toward the upper bound until R is positive-definite (Boxmin's _start
        # relocation); the climb probes are feasibility-only and are not part of the trajectory.
        x = self._relocate(x)
        self._x = x
        self._f = self._record(x)
        self._kmax = 2 if p <= 2 else min(p, 4)
        self._k = 0
        self._ok = np.isfinite(self._f)  # only run the pattern moves from a feasible start
        self.message = "completed"

    def _relocate(self, x):
        """Move the start up toward ``hi`` (``theta *= 2`` per step) until it is feasible."""
        t = x.copy()
        for _ in range(64):
            if bool(self.problem(np.atleast_2d(t)).feasible[0]):
                return t
            nxt = np.minimum(t + _LOG10_2, self._hi)  # theta *= 2 toward the upper bound
            if np.all(nxt == t):
                return t
            t = nxt
        return t

    def _record(self, x):
        """Evaluate ``x``, append it to the trajectory, report a feasible point to the callback."""
        ev = self.problem(np.atleast_2d(x))
        self.n_evals += 1
        self.visited.append(np.array(x, float))
        f = float(ev.f[0])
        if bool(ev.feasible[0]):
            self._emit(x, f, ev.info[0] if ev.info is not None else None)
            return f
        return np.inf

    def _try(self, tt):
        """Evaluate a candidate; adopt it as the incumbent if it lowers the (MLE) objective."""
        f = self._record(tt)
        if f < self._f:
            self._x, self._f = np.array(tt, float), f
            return True
        return False

    def _advance(self):
        if not self._ok or self._k >= self._kmax:
            return False
        self._k += 1
        last_x = self._x.copy()
        self._explore()
        self._move(last_x)
        return self._k < self._kmax

    def _explore(self):
        # probe each searched coordinate up by one step (halved at a bound), then down if that did
        # not improve -- the Hooke & Jeeves exploratory move, refreshed against the current best.
        for j in self._ne:
            t = self._x.copy()
            step = self._s[j]
            tt = t.copy()
            if t[j] == self._hi[j]:
                tt[j] = t[j] - 0.5 * step
            elif t[j] == self._lo[j]:
                tt[j] = t[j] + 0.5 * step
            else:
                tt[j] = min(self._hi[j], t[j] + step)
            improved = self._try(tt)
            if not improved and t[j] != self._hi[j] and t[j] != self._lo[j]:
                tt = t.copy()
                tt[j] = max(self._lo[j], t[j] - step)
                self._try(tt)

    def _move(self, last_x):
        # pattern move: extrapolate along the exploratory gain (doubling the step on each success),
        # then shrink the step schedule for the next sweep (the perm/0.2/0.25 update from Boxmin).
        perm = np.concatenate([np.arange(1, self._p), [0]])
        v = self._x - last_x
        if np.all(v == 0.0):
            self._s = 0.2 * self._s[perm]
            return
        cont = True
        while cont:
            tt = np.clip(self._x + v, self._lo, self._hi)
            improved = self._try(tt)
            if improved:
                v = v * 2.0
            cont = improved
            if np.any((tt == self._lo) | (tt == self._hi)):
                cont = False
        self._s = 0.25 * self._s[perm]
