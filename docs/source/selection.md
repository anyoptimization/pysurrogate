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

# Benchmarking & model selection

Which surrogate is best for *your* data? pysurrogate answers this with cross-validated
benchmarking, an auto-selecting drop-in model, a study harness for known functions, and a
direction-aware metrics registry.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from pysurrogate import Kriging, RBF, KNN, Sampling, LHS
from pysurrogate.util.test_functions import get_test_function

f, xl, xu = get_test_function("ackley", n_var=2)
X = Sampling(60, LHS()).sample((xl, xu), rng=np.random.default_rng(0))
y = f(X)
fleet = {"KNN": KNN(), "RBF": RBF(), "Kriging": Kriging()}
```

## `Benchmark` — cross-validate a fleet

`Benchmark` runs k-fold cross-validation for every model on one dataset and ranks them by a
metric. `.frame()` returns the per-model metric means:

```{code-cell} ipython3
from pysurrogate import Benchmark

bench = Benchmark(fleet, metrics=["rmse", "mae", "r2"])
bench.do(X, y)
bench.frame(sorted_by="rmse")
```

## `AutoModel` — the self-selecting surrogate

`AutoModel` *is* a `Model`: it benchmarks a fleet under the hood, picks the winner, refits it
on the full data, and then behaves like any other backend — same `fit` / `predict`, same
normalization lifecycle:

```{code-cell} ipython3
from pysurrogate import AutoModel

auto = AutoModel(models=fleet, sorted_by="rmse")
auto.fit(X, y)

Xte = Sampling(500, LHS()).sample((xl, xu), rng=np.random.default_rng(1))
pred = auto.predict(Xte, var=True)
print("selected fleet scores (best first):")
for name, sc in auto.statistics().items():
    print(f"  {name:>8}: {sc:.4f}")
```

## `study` — sweep a known function

When the truth is a known function, `study` resamples the design many times and aggregates,
so the ranking is not an artifact of one lucky draw. `StudyResult` carries the tidy numbers:

```{code-cell} ipython3
from pysurrogate import study

result = study(f, xl, xu, n=40, models=fleet, repeats=5, seed=1)
result.frame()
```

```{code-cell} ipython3
print("best model:", result.best())
print("mean ranks:", result.ranking())
```

Add label noise to the training targets (the test targets stay clean) to see which models
are robust to noisy observations:

```{code-cell} ipython3
noisy = study(f, xl, xu, n=40, models=fleet, repeats=5, seed=1, noise=1.0)
noisy.frame()
```

`FunctionBenchmark` is the lower-level engine behind `study` — it returns the raw tidy
predictions frame (one row per test point, per replication, per model) if you want to
compute your own aggregates with `score`:

```{code-cell} ipython3
from pysurrogate import FunctionBenchmark, score

fb = FunctionBenchmark(f, xl, xu, models=fleet, train=Sampling(40, LHS()),
                       test=Sampling(400, LHS()), replications=2, random_state=0)
df = fb.run()
score(df, ["rmse", "mae"], by=("model",))
```

## The metrics registry

Metrics are organized into families — accuracy, fit, ranking, selection, and calibration —
each knowing its own *direction* (lower- or higher-is-better). Compute one with `calc_metric`,
list them with `metric_names`, or evaluate a whole family with `evaluate`:

```{code-cell} ipython3
from pysurrogate.selection.metrics import calc_metric, metric_names, metric_sort_key

y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_hat = np.array([1.1, 1.9, 3.3, 3.7])

print("accuracy metrics:", metric_names(family="accuracy"))
print("rmse =", round(calc_metric("rmse", y_true, y_hat), 4))
print("r2   =", round(calc_metric("r2", y_true, y_hat), 4))
```

`metric_sort_key` makes ranking direction-agnostic — it returns a key where **smaller is
always better**, so `r2` (higher-is-better) is negated automatically:

```{code-cell} ipython3
print("sort key for rmse=0.2 :", metric_sort_key("rmse", 0.2))   # as-is
print("sort key for r2=0.85  :", metric_sort_key("r2", 0.85))    # negated
```

Calibration metrics need a `sigma`, so they only apply to variance-bearing backends
([models](models.ipynb)):

```{code-cell} ipython3
sigma = np.array([0.2, 0.2, 0.3, 0.3])
print("nlpd =", round(calc_metric("nlpd", y_true, y_hat, sigma=sigma), 4))
```

## Building fleets — `cartesian` and `as_named`

`cartesian` instantiates a whole grid of a model's hyperparameters as a named `{name: model}`
fleet, ready to drop into any of the tools above:

```{code-cell} ipython3
from pysurrogate import cartesian, as_named
from pysurrogate.dace.corr import Gaussian, Matern, RationalQuadratic

kriging_fleet = cartesian(Kriging, corr={
    "gauss": Gaussian(),
    "matern": Matern(nu=2.5),
    "rq": RationalQuadratic(0.25),
})
list(kriging_fleet)
```

`as_named` normalizes a bare list into the same `{name: model}` form (disambiguating repeats):

```{code-cell} ipython3
as_named([KNN(), RBF(), KNN()])
```

## Test functions

The `pysurrogate.util.test_functions` registry provides standard analytic landscapes over
their conventional boxes — handy for studies and quick checks:

```{code-cell} ipython3
from pysurrogate.util.test_functions import TEST_FUNCTIONS, get_test_function

print("available:", sorted(TEST_FUNCTIONS))
f, xl, xu = get_test_function("rastrigin", n_var=3)
f(np.zeros((1, 3)))   # optimum is 0 at the origin
```

## See also

- [models](models.ipynb) — the backends being selected among.
- [getting_started](getting_started.ipynb) — the single-model workflow these tools automate.
