"""Theta-optimization strategies for the Dace model.

End users pick a strategy and pass an instance to ``Dace(optimizer=...)`` or
``model.refit(optimizer=...)``:

- ``Boxmin()``  -- Hooke & Jeeves pattern search (the constructor default; reproduces
  MATLAB Dace exactly, the port-fidelity contract).
- ``ScreenedLBFGS()`` -- a batched cheap screen feeding a few analytic-gradient L-BFGS
  polishes; the fast *and* high-quality search, and ``refit``'s default warm refiner.
- ``LBFGS()``   -- bounded quasi-Newton with an analytic gradient; a plain local refine.
- ``VectorizedAdam()`` -- a population of theta descended in lock-step (batched).
- ``Fixed()``   -- no search, fit at the current theta (the cheapest refit).

All are subclasses of ``Optimizer`` and obtain their *committed* fits through
``fit_feasible``, so they consistently honor the model's noise / ``max_noise`` policy
(the per-step search fits stay strict -- a non-PD theta is simply infeasible).
"""

from pysurrogate.dace.optimizers.adam import VectorizedAdam
from pysurrogate.dace.optimizers.base import Optimizer, fit_feasible
from pysurrogate.dace.optimizers.boxmin import Boxmin
from pysurrogate.dace.optimizers.fixed import Fixed
from pysurrogate.dace.optimizers.lbfgs import LBFGS, objective_gradient
from pysurrogate.dace.optimizers.screened import ScreenedLBFGS

__all__ = [
    "Optimizer",
    "fit_feasible",
    "Boxmin",
    "Fixed",
    "LBFGS",
    "ScreenedLBFGS",
    "VectorizedAdam",
    "objective_gradient",
]
