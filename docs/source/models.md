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

# The model zoo

Every backend in pysurrogate implements the same `Model` contract, so they are
interchangeable:

- `fit(X, y, optimize=True)` — train on a design `X` (shape `(n, d)`) and targets `y`.
- `predict(X, var=False, grad=False)` — return a `Prediction` (mean, and optionally
  variance and gradient).
- `refit(X_new, y_new)` — score new points out-of-sample, then absorb them.

The base class also handles input normalization (`norm_X` / `norm_y`), NaN/inf filtering,
optional duplicate elimination, and a `records()` validation log — so backends only
implement the math.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from pysurrogate import (
    Kriging, RBF, SVR, KNN, InverseDistanceWeighting,
    SimpleMean, PolynomialRegression, RandomForest, Sampling, LHS,
)
from pysurrogate.selection.metrics import calc_metric

plt.rcParams["figure.figsize"] = (7, 4)

def truth(x):
    return np.sin(3 * x) + 0.4 * x

xl, xu = np.array([0.0]), np.array([5.0])
X = Sampling(14, LHS()).sample((xl, xu), rng=np.random.default_rng(0))
y = truth(X).ravel()
grid = np.linspace(0, 5, 300).reshape(-1, 1)
```

## One fit per backend

Each backend brings a different inductive bias — interpolating, smoothing, piecewise, or a
flat baseline:

```{code-cell} ipython3
models = {
    "Kriging": Kriging(),
    "RBF": RBF(),
    "SVR": SVR(),
    "KNN": KNN(n_nearest=4),
    "InverseDistanceWeighting": InverseDistanceWeighting(),
    "SimpleMean": SimpleMean(),
    "PolynomialRegression": PolynomialRegression(degree=3),
    "RandomForest": RandomForest(),
}

fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
for ax, (name, model) in zip(axes.ravel(), models.items()):
    model.fit(X, y)
    mu = model.predict(grid).y.ravel()
    ax.plot(grid, truth(grid), "k--", lw=1)
    ax.plot(grid, mu, "C0")
    ax.scatter(X, y, c="k", s=12, zorder=5)
    ax.set_title(name, fontsize=9)
fig.tight_layout()
```

## Side-by-side accuracy

Fit every backend on the same design and rank them by test RMSE on a dense grid:

```{code-cell} ipython3
y_grid = truth(grid).ravel()
rows = []
for name, model in models.items():
    model.fit(X, y)
    rmse = calc_metric("rmse", y_grid, model.predict(grid).y.ravel())
    rows.append((name, rmse))

import pandas as pd
pd.DataFrame(rows, columns=["model", "test RMSE"]).sort_values("test RMSE").reset_index(drop=True)
```

```{code-cell} ipython3
plt.plot(grid, truth(grid), "k--", lw=1.5, label="truth")
for name, model in models.items():
    model.fit(X, y)
    plt.plot(grid, model.predict(grid).y.ravel(), lw=1, label=name)
plt.scatter(X, y, c="k", zorder=5)
plt.legend(fontsize=8, ncol=2, loc="upper left")
plt.title("all backends overlaid");
```

## Which backends report uncertainty?

Only some backends produce a predictive **variance** (`predict(..., var=True)`):

| Backend | `var` | `grad` | note |
|---|:--:|:--:|---|
| `Kriging` | ✓ | ✓ | full GP — calibrated variance & gradients |
| `RandomForest` | ✓ | — | ensemble disagreement across trees |
| `KNN` | ✓ | — | variance of the nearest neighbors |
| `RBF` | — | ✓ | interpolating radial basis + polynomial tail |
| `InverseDistanceWeighting` | — | ✓ | Shepard interpolation |
| `PolynomialRegression` | — | ✓ | analytic gradient of the polynomial |
| `SVR` | — | — | mean only |
| `SimpleMean` | — | — | constant baseline |

```{code-cell} ipython3
# The variance-bearing backends give an honest band; the others return None for var.
for name in ["Kriging", "RandomForest", "KNN", "SVR"]:
    p = models[name].predict(grid[:3], var=True)
    print(f"{name:>14}: var is {'available' if p.var is not None else 'None'}")
```

This split is exactly what the [selection](selection.ipynb) page uses when a calibration metric needs
`sigma` — models without a variance are simply skipped for those metrics.

## See also

- [kriging](kriging.ipynb) — the deepest backend, in full.
- [selection](selection.ipynb) — benchmark the whole zoo and let `AutoModel` pick a winner.
