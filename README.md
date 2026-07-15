# pysurrogate

A unified surrogate-modeling toolkit for Python — sampling, fitting, and model
selection in one place.

Every backend speaks the same `Model` contract — `fit` / `predict` / `refit` —
so you can swap a Kriging model for a random forest without touching the
surrounding code:

- **Kriging / DACE** — a faithful DACE engine (`Dace`, `Kriging`, `KPLS`) with a
  correlation-kernel zoo, regression trends, ARD, learned noise, calibrated
  variance, and analytic gradients.
- **Model zoo** — `RBF`, `SVR`, `KNN`, `InverseDistanceWeighting`,
  `RandomForest`, `PolynomialRegression`, `DeepKernelGP`, `RotatedKriging`, and
  a `SimpleMean` baseline.
- **Generic optimizers** — a small `Problem`/`Optimizer` layer (LBFGS,
  PatternSearch, Boxmin, Adam, Restart) reusable far beyond theta tuning.
- **Model selection** — `Benchmark`, `AutoModel`, `study`, and a
  direction-aware metrics registry.
- **Landscape analysis** — model-free structural fingerprints of a labelled
  point cloud (`pysurrogate.landscape`).

## Install

```bash
pip install pysurrogate
```

## Quick start

```python
import numpy as np
from pysurrogate import Kriging

X = np.random.random((20, 2))
y = (X**2).sum(axis=1)

model = Kriging().fit(X, y)

pred = model.predict(np.random.random((5, 2)), var=True)
print(pred.y, pred.sigma)
```

The lower-level `Dace` engine is available directly:

```python
from pysurrogate import Dace, Gaussian, ConstantRegression

model = Dace(regr=ConstantRegression(), corr=Gaussian(), theta=1.0, theta_bounds=(1e-3, 20.0))
model.fit(X, y.reshape(-1, 1))
pred = model.predict(np.random.random((5, 2)), var=True)
```

## Documentation

The full documentation (getting started, the model zoo, Kriging internals,
optimizers, sampling, selection, and the API reference) lives under `docs/` and
builds with `pyclawd docs build`.

## Development

This project is driven by [pyclawd](https://github.com/julian/pyclawd). See
`AGENTS.md` for the command contract; the short version:

```bash
pyclawd check     # format + lint + typecheck + descriptions + tests
pyclawd test      # run the suite
```
