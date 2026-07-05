"""Restart: run an inner optimizer from several sampled starts, keeping the best continuously."""

import numpy as np

from pysurrogate.core.optimizer import Optimizer


class Restart(Optimizer):
    """Run an inner optimizer from several sampled starts, keeping the best across all of them.

    Composes with *any* inner optimizer. :class:`~pysurrogate.core.sampling.Sampling` generates
    the candidate starts (the ``x0`` force-included at runtime); the inner optimizer polishes
    each, **sharing this Restart's callback** so the running best is tracked continuously across
    every start. One iteration is one inner run from one start.

    With ``screen`` set, the sampled candidates are first ranked by a cheap objective-only pass
    (:meth:`Problem.screen`) and trimmed to the best ``screen`` *before* polishing -- sample
    broadly, polish few. So the old screen-and-polish search is just
    ``Restart(LBFGS(), Sampling(64, LHS()), screen=8)``; with ``screen=None`` every sampled
    candidate is polished (plain multi-start).

    Args:
        inner: The :class:`~pysurrogate.core.optimizer.Optimizer` to run from each start (it is
            re-``setup`` per start, so a single instance is reused).
        sampling: A :class:`~pysurrogate.core.sampling.Sampling` for the candidate starts.
        screen: Keep only this many best candidates (by the cheap screen) as starts; ``None``
            polishes every sampled candidate.
        random_state: Seed for the sampling.
    """

    def __init__(self, inner, sampling, screen=None, random_state=0):
        super().__init__()
        self.inner = inner
        self.sampling = sampling
        self.screen = screen
        self.random_state = random_state

    def _setup(self):
        # seed candidates from the FINITE sampling region; the inner optimizer's descent is
        # constrained by the problem's hard bounds, which may be unbounded above.
        _, _, slo, shi = self._box()
        rng = np.random.default_rng(self.random_state)
        extra = [self.x0] if self.x0 is not None else []
        cand = self.sampling.sample((slo, shi), rng, include=extra)

        if self.screen is not None and self.screen < len(cand):
            # cheap objective-only rank, then keep the best `screen` as the starts to polish
            f = np.asarray(self.problem.screen(cand), float)
            self.n_evals += len(cand)
            cand = cand[np.argsort(f)[: self.screen]]

        self._starts = list(cand)
        self._next = 0
        self.message = "completed"

    def _advance(self):
        if self._next >= len(self._starts):
            return False
        start = self._starts[self._next]
        self._next += 1
        # polish this start with the inner optimizer, SHARING our callback so the best is kept
        # across all starts; the inner optimizer never knows it is one of several.
        run = self.inner.setup(self.problem, x0=start, callback=self.callback)
        run.run()
        self.n_evals += run.n_evals
        return self._next < len(self._starts)
