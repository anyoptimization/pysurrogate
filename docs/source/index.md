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

# pysurrogate

**A unified surrogate-modeling toolkit — sampling, Kriging/DACE, a model zoo, a generic
optimizer layer, and automatic model selection.**

pysurrogate builds cheap, differentiable *surrogates* (a.k.a. metamodels or response
surfaces) that stand in for an expensive function. Every backend speaks the same
`Model` contract — `fit` / `predict` / `refit` — so you can swap a Kriging model for a
random forest without touching the surrounding code, ask any model for a calibrated
uncertainty estimate, and let `AutoModel` pick the best one for your data.

## Install

```bash
pip install -U pysurrogate
```

## Hero example — Kriging with a confidence band

Sample a 1-D function, fit a `Kriging` model, and predict the mean together with its
predictive uncertainty (`±2σ`):

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from pysurrogate import Kriging

plt.rcParams["figure.figsize"] = (7, 4)

def f(x):
    return np.sin(2 * np.pi * x) + 0.3 * x

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=(8, 1))         # 8 training points
y = f(X).ravel()

model = Kriging().fit(X, y)                 # fit the surrogate

grid = np.linspace(0, 1, 200).reshape(-1, 1)
pred = model.predict(grid, var=True)        # mean + predictive variance
mu, sigma = pred.y.ravel(), pred.sigma.ravel()

plt.plot(grid, f(grid), "k--", lw=1, label="true function")
plt.plot(grid, mu, "C0", label="Kriging mean")
plt.fill_between(grid.ravel(), mu - 2 * sigma, mu + 2 * sigma,
                 color="C0", alpha=0.2, label="±2σ")
plt.scatter(X, y, c="k", zorder=5, label="samples")
plt.legend(loc="upper right")
plt.title("Kriging surrogate with a predictive confidence band");
```

The band is wide where data is sparse and pinches to zero at the training points — the
hallmark of an interpolating Gaussian-process surrogate.

## What's inside

| Area | Highlights |
|---|---|
| **Kriging / DACE** | A faithful DACE engine: correlation-kernel zoo, regression trends, ARD, learned noise, calibrated variance, analytic gradients. |
| **Model zoo** | `Kriging`, `KPLS`, `RotatedKriging`, `DeepKernelGP`, `RBF`, `SVR`, `KNN`, `InverseDistanceWeighting`, `SimpleMean`, `PolynomialRegression`, `RandomForest` — one `Model` contract. |
| **Uncertainty** | Predictive variance, `sigma`, and `calibrate()` to make intervals honest. |
| **Generic optimizers** | A small `Problem`/`Optimizer` layer — LBFGS, PatternSearch, Boxmin, Adam, Restart — reusable far beyond theta tuning. |
| **Sampling** | Latin-Hypercube and random designs with local RNGs (no global-state pollution). |
| **Model selection** | `Benchmark`, `AutoModel`, `study`, and a direction-aware metrics registry. |
| **Landscape analysis** | `Landscape` — model-free structural fingerprints (rotation, modality, separability, smoothness, ...) of a labelled point cloud. |

## Explore

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. toctree::
   :maxdepth: 1

   getting_started
   models
   kriging
   optimizers
   sampling
   selection
   landscape
   api
```

## About

pysurrogate is part of [anyoptimization](https://anyoptimization.com) by
[Julian Blank](https://julianblank.com).

- **Source:** [github.com/anyoptimization/pysurrogate](https://github.com/anyoptimization/pysurrogate)
- **Issues:** [github.com/anyoptimization/pysurrogate/issues](https://github.com/anyoptimization/pysurrogate/issues)
