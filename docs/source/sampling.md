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

# Sampling, partitioning & transforms

The plumbing around a fit: where to place your design points, how to split them for
cross-validation, and how to normalize inputs and outputs. All three use **local** random
generators — pysurrogate never mutates NumPy's global RNG.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from pysurrogate import Sampling, LHS, Random

plt.rcParams["figure.figsize"] = (7, 4)
```

## Sampling a design

`Sampling(n, method)` fills a box with `n` points using a space-filling `method`. `LHS()`
(Latin Hypercube) stratifies every axis for even coverage; `Random()` draws uniformly.
Reproducibility comes from the `rng` you pass to `.sample`:

```{code-cell} ipython3
lo, hi = np.array([0.0, 0.0]), np.array([1.0, 1.0])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (name, method) in zip(axes, [("LHS", LHS()), ("Random", Random())]):
    X = Sampling(60, method).sample((lo, hi), rng=np.random.default_rng(0))
    ax.scatter(X[:, 0], X[:, 1], s=20)
    ax.set_title(f"{name}  (n={len(X)})"); ax.set_aspect("equal")
fig.tight_layout()
```

Same seed, same points — no global state touched:

```{code-cell} ipython3
a = Sampling(5, LHS()).sample((lo, hi), rng=np.random.default_rng(42))
b = Sampling(5, LHS()).sample((lo, hi), rng=np.random.default_rng(42))
np.allclose(a, b)
```

You can also force specific points into every design (e.g. the box corners or a current
incumbent) with `include`:

```{code-cell} ipython3
forced = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
X = Sampling(10, LHS(), include=forced).sample((lo, hi), rng=np.random.default_rng(1))
X[:2]   # the two forced points come first
```

## Partitioning for cross-validation

`CrossvalidationPartitioning` splits the row indices into `k` folds; each `.do(X)` returns a
list of `Split(train, test, valid)` index sets. `default_partitioning` is the shared default
(`DEFAULT_CV_FOLDS`-fold):

```{code-cell} ipython3
from pysurrogate import CrossvalidationPartitioning, RandomPartitioning
from pysurrogate.core.partitioning import default_partitioning, DEFAULT_CV_FOLDS

X = np.arange(20)
splits = CrossvalidationPartitioning(k_folds=5, seed=0).do(X)

print(f"DEFAULT_CV_FOLDS = {DEFAULT_CV_FOLDS}")
for i, s in enumerate(splits):
    print(f"fold {i}: test rows = {np.sort(s.test)}")
```

Every fold holds out a disjoint block, and together the test sets cover all rows exactly
once. `RandomPartitioning` instead draws independent train/test splits:

```{code-cell} ipython3
rp = RandomPartitioning(perc_train=0.7, n_sets=3, seed=0).do(X)
for i, s in enumerate(rp):
    print(f"split {i}: {len(s.train)} train / {len(s.test)} test")
```

## Transformations

Transformations normalize inputs or outputs and always support a `forward` / `backward`
round-trip. They are what the `Model` lifecycle uses internally (`norm_X` / `norm_y`), but
you can use them directly:

```{code-cell} ipython3
from pysurrogate.core.transformation import (
    Standardization, ZeroToOneNormalization, Plog, NoNormalization,
)

data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])

std = Standardization()
z = std.forward(data)                       # zero mean, unit variance (auto-fit)
print("standardized mean/std:", z.mean(0).round(3), z.std(0).round(3))
print("round-trip exact:", np.allclose(std.backward(z), data))

mm = ZeroToOneNormalization()
u = mm.forward(data)                        # min-max to [0, 1] per column
print("min/max:", u.min(0), u.max(0))
```

`Plog` is a non-affine signed-log — it compresses heavy-tailed targets so a surrogate can fit
them, and inverts exactly on the way back:

```{code-cell} ipython3
y = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
plog = Plog()
print("plog:    ", plog.forward(y).round(3))
print("inverted:", plog.backward(plog.forward(y)).round(3))
```

`NoNormalization` is the identity — the default when a model should see raw values.

## See also

- [getting_started](getting_started.ipynb) — sampling and fitting together in the workflow.
- [selection](selection.ipynb) — partitioning powers the cross-validated benchmarks.
