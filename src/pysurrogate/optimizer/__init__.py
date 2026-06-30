"""Generic, backend-free optimizers over a bounded :class:`~pysurrogate.core.optimizer.Problem`.

These strategies know nothing about surrogates -- they minimize any ``Problem`` (a box plus a
never-raising ``__call__``), drive selection and early stopping through a
:class:`~pysurrogate.core.optimizer.Callback`, and generate their starts from a shared
:class:`~pysurrogate.core.sampling.Sampling`. So they are reusable for hyper-parameter fitting,
acquisition maximization, or any bounded search:

- :class:`LBFGS` -- bounded quasi-Newton; uses the problem's analytic gradient when present.
- :class:`PatternSearch` -- derivative-free compass search; robust on non-smooth objectives.
- :class:`Boxmin` -- Hooke & Jeeves pattern search; a faithful generic port of MATLAB DACE's
  Boxmin (reproduces it bit-for-bit on the log-space DACE problem).
- :class:`Adam` -- a population descended in lock-step by gradient (vectorized).
- :class:`Restart` -- wrap any inner optimizer, run it from sampled starts (optionally screened),
  keeping the best continuously.
"""

from pysurrogate.optimizer.adam import Adam
from pysurrogate.optimizer.boxmin import Boxmin
from pysurrogate.optimizer.lbfgs import LBFGS
from pysurrogate.optimizer.pattern import PatternSearch
from pysurrogate.optimizer.restart import Restart

__all__ = ["LBFGS", "PatternSearch", "Boxmin", "Adam", "Restart"]
