---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# The optimizer layer

Fitting Kriging requires searching for the length-scale that maximizes the likelihood, so
pysurrogate ships a small, **surrogate-agnostic** optimization layer. It is a clean contract
you can reuse for any bounded problem — acquisition maximization, hyperparameter tuning, or
plain function minimization.

The contract has four pieces:

- `Problem` — a bounded minimization over a box that *never raises*: an infeasible
  candidate is reported with `feasible=False` and `f=+inf`, so the search steps away instead
  of crashing.
- `Optimizer` — a search strategy with a `setup` → `advance` → `run` lifecycle
  (`minimize` is the one-shot shortcut).
- `Callback` — the selector *and* the stopping rule in one object.
- `Evaluation` / `Result` — the value objects passed in and out.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from pysurrogate.core.optimizer import Problem, Evaluation

plt.rcParams["figure.figsize"] = (7, 4)

class Sphere(Problem):
    """A bounded 2-D sphere with an analytic gradient — minimum 0 at the origin."""

    def __init__(self, lo, hi):
        self._lo, self._hi = np.asarray(lo, float), np.asarray(hi, float)

    @property
    def bounds(self):
        return self._lo, self._hi

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        f = np.sum(X**2, axis=1)          # (J,)
        grad = 2.0 * X                    # (J, p)
        feasible = np.ones(len(X), dtype=bool)
        return Evaluation(f=f, feasible=feasible, grad=grad)

problem = Sphere([-5.0, -5.0], [5.0, 5.0])
problem.n_var, problem.has_grad
```

## Minimize with each strategy

Every strategy shares the same `minimize(problem, x0=...)` entry point and returns a `Result`
(`x`, `f`, `n_evals`, `message`):

```{code-cell} ipython3
from pysurrogate import LBFGS, PatternSearch, Boxmin, Adam, Restart, Sampling, LHS

start = np.array([4.0, -4.0])
strategies = {
    "LBFGS": (LBFGS(), start),
    "PatternSearch": (PatternSearch(), start),
    "Boxmin": (Boxmin(), start),
    "Adam": (Adam(), start),
    "Restart+LBFGS": (Restart(LBFGS(), Sampling(20, LHS())), None),
}

import pandas as pd
rows = []
for name, (opt, x0) in strategies.items():
    res = opt.minimize(problem, x0=x0)
    rows.append((name, res.x[0], res.x[1], res.f, res.n_evals))
pd.DataFrame(rows, columns=["strategy", "x0*", "x1*", "f*", "evals"])
```

All five converge to the origin from different mechanics: a quasi-Newton local descent
(`LBFGS`), derivative-free compass searches (`PatternSearch`, `Boxmin`), a population
gradient method (`Adam`), and a multi-start wrapper (`Restart`).

## Trajectories

The `visited` list is part of the `Optimizer` contract, but recording it is *optional* — a
strategy that keeps a trajectory (like the `Boxmin` pattern search) appends to it; others
leave it empty:

```{code-cell} ipython3
for name, (opt, x0) in strategies.items():
    opt.minimize(problem, x0=x0)
    print(f"{name:>14}: {len(opt.visited)} points recorded")
```

`Boxmin` walks downhill on the sphere's contours, one poll at a time:

```{code-cell} ipython3
gx = np.linspace(-5, 5, 100)
GX, GY = np.meshgrid(gx, gx)
Z = GX**2 + GY**2

opt = Boxmin()
opt.minimize(problem, x0=start)
path = np.array(opt.visited)

plt.contour(GX, GY, Z, levels=15, cmap="Greys", alpha=0.6)
plt.plot(path[:, 0], path[:, 1], "o-", ms=4, color="C3", label="Boxmin path")
plt.plot(*start, "ks", ms=9, label="start"); plt.plot(0, 0, "g*", ms=16, label="optimum")
plt.legend(); plt.title(f"Boxmin — {len(path)} steps to the origin");
```

## How Kriging uses it

The DACE theta search is *literally* one of these problems: the length-scale vector is the
parameter, the negative log-likelihood is the objective, and the box is `theta_bounds`. That
is why the default engine optimizer is

```python
from pysurrogate import Dace, Restart, LBFGS, Sampling, LHS
Dace(optimizer=Restart(LBFGS(), Sampling(16, LHS()), screen=4))
```

— sample many length-scales, screen to the best few, polish each with L-BFGS — and why
`Boxmin()` reproduces the classic MATLAB DACE search exactly. See [kriging](kriging.ipynb).

## Reuse it standalone

Because the layer knows nothing about surrogates, you can minimize *any* bounded objective —
for instance, maximizing a surrogate's predictive uncertainty (an active-learning
acquisition) by minimizing its negative:

```{code-cell} ipython3
from pysurrogate import Kriging

Xtr = Sampling(8, LHS()).sample((np.array([0.0]), np.array([5.0])), rng=np.random.default_rng(0))
ytr = np.sin(3 * Xtr).ravel()
model = Kriging().fit(Xtr, ytr)

class MaxSigma(Problem):
    """Minimize -sigma(x): find where the surrogate is least certain."""

    @property
    def bounds(self):
        return np.array([0.0]), np.array([5.0])

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        sigma = model.predict(X, var=True).sigma.ravel()
        return Evaluation(f=-sigma, feasible=np.ones(len(X), bool))

res = Restart(LBFGS(), Sampling(30, LHS())).minimize(MaxSigma())
print(f"most-uncertain x = {res.x[0]:.3f}   sigma = {-res.f:.3f}")
```

## See also

- [kriging](kriging.ipynb) — the theta search these optimizers drive.
