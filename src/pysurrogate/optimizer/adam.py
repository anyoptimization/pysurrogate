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
        # hard bounds clip the descent (may be +/-inf); seed the population from the finite
        # sampling region, since an infinite box cannot be uniformly sampled.
        lo, hi = (np.atleast_1d(np.asarray(b, float)) for b in self.problem.bounds)
        slo, shi = (np.atleast_1d(np.asarray(b, float)) for b in self.problem.sampling_bounds)
        self._lo, self._hi = lo, hi
        rng = np.random.default_rng(self.random_state)
        if self.sampling is not None:
            extra = [self.x0] if self.x0 is not None else []
            self._pop = self.sampling.sample((slo, shi), rng, include=extra)
        else:
            pop = [] if self.x0 is None else [np.clip(self.x0, lo, hi)]
            while len(pop) < self.pop_size:
                pop.append(rng.uniform(slo, shi))
            self._pop = np.array(pop[: self.pop_size])
        self._m = np.zeros_like(self._pop)
        self._v = np.zeros_like(self._pop)
        self.message = "completed (max steps)"

    def _advance(self):
        ev = self.problem(self._pop)
        self.n_evals += len(self._pop)
        if ev.grad is None:
            raise ValueError("Adam requires a problem that returns an analytic gradient.")

        for i in range(len(self._pop)):
            if bool(ev.feasible[i]):
                info = ev.info[i] if ev.info is not None else None
                if self._emit(self._pop[i], float(ev.f[i]), info):
                    return False

        # one Adam step over the population. Mask infeasible candidates to a zero gradient so they
        # genuinely do not move (the Evaluation contract does not promise a zero/finite grad for an
        # infeasible row), mirroring LBFGS -- until a neighbor pulls them back into a feasible region.
        b1, b2, eps, t = 0.9, 0.999, 1e-8, self.n_iter
        g = np.where(ev.feasible[:, None], ev.grad, 0.0)
        self._m = b1 * self._m + (1 - b1) * g
        self._v = b2 * self._v + (1 - b2) * (g * g)
        mhat = self._m / (1 - b1**t)
        vhat = self._v / (1 - b2**t)
        self._pop = np.clip(self._pop - self.lr * mhat / (np.sqrt(vhat) + eps), self._lo, self._hi)
        return self.n_iter < self.steps
