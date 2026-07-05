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

# Getting started

This page walks the **end-to-end surrogate workflow** as one narrative: sample a design,
fit a model, read its predictions and uncertainty, then grow the design point-by-point in
an active-learning loop and measure how the error shrinks.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from pysurrogate import Kriging, Sampling, LHS

plt.rcParams["figure.figsize"] = (7, 4)
rng = np.random.default_rng(1)
```

## 1. Sample a design

`Sampling(n, method)` fills a box with `n` points. The method is a space-filling strategy
— `LHS()` (Latin Hypercube) or `Random()`. Bounds are passed as a `(lower, upper)` pair of
arrays, and reproducibility comes from a **local** generator (`rng`) — pysurrogate never
touches NumPy's global RNG.

```{code-cell} ipython3
def truth(x):
    """A wiggly 1-D target to approximate."""
    return np.sin(3 * x) + 0.5 * x

xl, xu = np.array([0.0]), np.array([5.0])
X = Sampling(12, LHS()).sample((xl, xu), rng=rng)
y = truth(X).ravel()
X.shape, y.shape
```

## 2. Fit and predict

Every backend shares the same contract: `fit(X, y)` then `predict(X, var=, grad=)`. Ask
for the predictive variance and the mean gradient in one call — Kriging returns both from a
single Cholesky solve.

```{code-cell} ipython3
model = Kriging().fit(X, y)

grid = np.linspace(0, 5, 300).reshape(-1, 1)
pred = model.predict(grid, var=True, grad=True)
```

## 3. The `Prediction` object

`predict` returns a `Prediction` — read fields **by name**, never by tuple position:

| field | meaning |
|---|---|
| `pred.y` | predicted mean, shape `(m, q)` |
| `pred.var` | predictive variance, shape `(m, 1)` |
| `pred.sigma` | `sqrt(var)` — the standard deviation |
| `pred.grad` | gradient of the mean, shape `(m, d)` |
| `pred.var_grad` | gradient of the variance (when both `var` and `grad` are requested) |

```{code-cell} ipython3
mu = pred.y.ravel()
sigma = pred.sigma.ravel()
dmu = pred.grad.ravel()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

ax1.plot(grid, truth(grid), "k--", lw=1, label="truth")
ax1.plot(grid, mu, "C0", label="mean")
ax1.fill_between(grid.ravel(), mu - 2 * sigma, mu + 2 * sigma, color="C0", alpha=0.2, label="±2σ")
ax1.scatter(X, y, c="k", zorder=5, label="samples")
ax1.legend(loc="upper left"); ax1.set_title("mean & confidence band")

ax2.plot(grid, 3 * np.cos(3 * grid.ravel()) + 0.5, "k--", lw=1, label="true slope")
ax2.plot(grid, dmu, "C3", label="predicted gradient")
ax2.legend(loc="upper left"); ax2.set_title("mean gradient")
fig.tight_layout()
```

## 4. Active learning — grow the design

`refit(X_new, y_new)` does two things: it first **scores the new points against the current
model** (returning that out-of-sample `Prediction`), then folds them in and warm-starts the
fit. Collecting those out-of-sample predictions gives an honest, *prequential* error curve —
each point is predicted before the model ever sees it.

```{code-cell} ipython3
# Start small and add points one at a time.
rng = np.random.default_rng(3)
X0 = Sampling(6, LHS()).sample((xl, xu), rng=rng)
model = Kriging().fit(X0, truth(X0).ravel())

errors = []
for _ in range(15):
    x_new = Sampling(1, LHS()).sample((xl, xu), rng=rng)
    oos = model.refit(x_new, truth(x_new).ravel())   # score-then-absorb
    errors.append(abs(oos.y.ravel()[0] - truth(x_new).ravel()[0]))

plt.figure()
plt.plot(range(1, len(errors) + 1), errors, "o-")
plt.xlabel("points added"); plt.ylabel("out-of-sample |error|")
plt.title("prequential error as the design grows");
```

Everything the loop saw is available as a tidy frame via `records()`:

```{code-cell} ipython3
model.history().head()
```

## 5. Evaluate on a held-out set

Score the final model on fresh test points with the metrics registry:

```{code-cell} ipython3
from pysurrogate.selection.metrics import calc_metric

X_test = Sampling(500, LHS()).sample((xl, xu), rng=np.random.default_rng(99))
y_test = truth(X_test).ravel()
y_hat = model.predict(X_test).y.ravel()

for m in ["rmse", "mae", "r2"]:
    print(f"{m:>5}: {calc_metric(m, y_test, y_hat):.4f}")
```

## Where to go next

- [models](models.ipynb) — the full backend zoo behind the same contract.
- [kriging](kriging.ipynb) — everything the Kriging/DACE engine can do.
- [selection](selection.ipynb) — let `AutoModel` pick the best model for you.
