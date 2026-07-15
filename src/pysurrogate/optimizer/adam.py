"""Generic population Adam optimizer: a batch of points descended in lock-step by gradient."""

import numpy as np

from pysurrogate.core.optimizer import Optimizer


class Adam(Optimizer):
    """Population Adam over the box -- ``pop_size`` points descended in lock-step by gradient.

    One iteration is one Adam step over the whole population, which is a single batched problem
    evaluation -- so it stays cheap to batch and is a natural fit for :meth:`~Optimizer.advance`
    and racing. The population is seeded from ``x0`` (when given) plus random points, so it is a
    semi-global search that still benefits from a warm start. Requires a problem that returns an
    analytic gradient.

    Args:
        pop_size: Number of points descended together (used only when no ``sampling`` is given).
        steps: Number of Adam steps (iterations) -- a ceiling; the callback may stop sooner.
        lr: Adam learning rate.
        sampling: A :class:`~pysurrogate.core.sampling.Sampling` that seeds the initial
            population (its ``n`` sets the population size, ``x0`` force-included). ``None`` seeds
            ``pop_size`` points from ``x0`` plus random fill.
        random_state: Seed for the population sampling.
    """

    def __init__(self, pop_size=8, steps=50, lr=0.1, sampling=None, random_state=0):
        super().__init__()
        self.pop_size = pop_size
        self.steps = steps
        self.lr = lr
        self.sampling = sampling
        self.random_state = random_state

    def _setup(self):
        # Adam descends by gradient, so fail fast (like requires_x0) when the problem exposes none,
        # rather than only discovering it on the first _advance.
        if not self.problem.has_grad:
            raise ValueError("Adam requires a problem that returns an analytic gradient.")
        # hard bounds clip the descent (may be +/-inf); seed the population from the finite
        # sampling region, since an infinite box cannot be uniformly sampled.
        lo, hi, _, _ = self._box()
        self._lo, self._hi = lo, hi
        self._pop = self._seed_starts(self.sampling, self.random_state, n=self.pop_size, fill="uniform")
        self._m = np.zeros_like(self._pop)
        self._v = np.zeros_like(self._pop)

    def _advance(self):
        ev = self.problem(self._pop)
        self.n_evals += len(self._pop)
        # gradient support was verified in _setup (fail-fast), so ev.grad is present here.
        if self._emit_batch(self._pop, ev):
            return False

        # one Adam step over the population. Zero the gradient of infeasible candidates -- the
        # Evaluation contract does not promise a finite grad for an infeasible row, and a NaN there
        # would poison the shared momentum/variance state. Note this does not fully freeze such a
        # point: population members are independent (no interacting neighbors), and accumulated
        # momentum (self._m) can still nudge it -- which is fine, it drifts under inertia until a
        # step lands it back in the feasible region.
        b1, b2, eps, t = 0.9, 0.999, 1e-8, self.n_iter
        g = np.where(ev.feasible[:, None], ev.grad, 0.0)
        self._m = b1 * self._m + (1 - b1) * g
        self._v = b2 * self._v + (1 - b2) * (g * g)
        mhat = self._m / (1 - b1**t)
        vhat = self._v / (1 - b2**t)
        self._pop = np.clip(self._pop - self.lr * mhat / (np.sqrt(vhat) + eps), self._lo, self._hi)
        if self.n_iter >= self.steps:
            self.message = "completed (max steps)"
            return False
        return True
