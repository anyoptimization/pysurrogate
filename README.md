# pysurrogate

A unified surrogate-modeling toolkit for Python — sampling, fitting, and model
selection in one place.

`pysurrogate` consolidates a DACE Kriging engine and a multi-backend model layer
into a single, consistently-styled package:

- **`pysurrogate.dace`** — the DACE Kriging engine: the `Dace` model with
  pluggable regression trends, correlation kernels, and theta optimizers.
- *(more backends — RBF, SVR, KNN, IDW, ... — and the model-selection layer are
  being folded in.)*

## Install

```bash
pip install pysurrogate
```

## Quick start

```python
import numpy as np
from pysurrogate import Dace
from pysurrogate.dace import Gaussian, ConstantRegression

X = np.random.random((20, 2))
y = (X ** 2).sum(axis=1, keepdims=True)

model = Dace(regr=ConstantRegression(), corr=Gaussian(), theta=1.0, thetaL=1e-3, thetaU=20.0)
model.fit(X, y)

pred = model.predict(np.random.random((5, 2)), mse=True)
print(pred.y, pred.mse)
```

## Development

This project is driven by [pyclawd](https://github.com/julian/pyclawd). See
`AGENTS.md` for the command contract; the short version:

```bash
pyclawd check     # format + lint + typecheck + descriptions + tests
pyclawd test      # run the suite
```
