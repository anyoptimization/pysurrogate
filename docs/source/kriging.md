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

# Kriging / DACE

Kriging (a.k.a. DACE — *Design and Analysis of Computer Experiments*) is the flagship of
pysurrogate: a Gaussian-process interpolator that fits a **regression trend** plus a
**correlated residual**, and returns a calibrated predictive **variance** and analytic
**gradients** alongside the mean. This page tours everything the engine can do.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from pysurrogate import Kriging, Dace, Sampling, LHS

plt.rcParams["figure.figsize"] = (7, 4)

def truth(x):
    return np.sin(3 * x) + 0.4 * x

xl, xu = np.array([0.0]), np.array([5.0])
rng = np.random.default_rng(0)
X = Sampling(10, LHS()).sample((xl, xu), rng=rng)
y = truth(X).ravel()
grid = np.linspace(0, 5, 300).reshape(-1, 1)
```

## Correlation kernels

The kernel sets how quickly correlation decays with distance — and therefore how smooth the
surrogate is. pysurrogate ships a full zoo, all importable from `pysurrogate.dace.corr`:

```{code-cell} ipython3
from pysurrogate.dace.corr import (
    Gaussian, Exponential, Matern, RationalQuadratic, Cubic, Spline, Spherical,
)

kernels = {
    "Gaussian": Gaussian(),
    "Exponential": Exponential(),
    "Matern(1.5)": Matern(nu=1.5),
    "Matern(2.5)": Matern(nu=2.5),
    "RationalQuadratic": RationalQuadratic(0.25),
    "Cubic": Cubic(),
    "Spline": Spline(),
    "Spherical": Spherical(),
}

plt.plot(grid, truth(grid), "k--", lw=1, label="truth")
for name, corr in kernels.items():
    mu = Kriging(corr=corr).fit(X, y).predict(grid).y.ravel()
    plt.plot(grid, mu, lw=1, label=name)
plt.scatter(X, y, c="k", zorder=5)
plt.legend(loc="upper left", fontsize=8, ncol=2)
plt.title("the same data, eight correlation kernels");
```

The rougher kernels (`Exponential`, `Spherical`) give a jagged fit; the smooth ones
(`Gaussian`, `Matern(2.5)`) are gentle. `RationalQuadratic(0.25)` is the default — the best
all-round performer on pysurrogate's benchmark.

## ARD — anisotropic length-scales

By default a single length-scale is shared across all inputs. `ARD=True` fits **one
length-scale per dimension**, which matters when the function varies at different rates
along different axes:

```{code-cell} ipython3
# A 2-D function that wiggles fast in x0 and slowly in x1.
def aniso(X):
    return np.sin(3 * X[:, 0]) + 0.1 * X[:, 1]

lo, hi = np.array([0.0, 0.0]), np.array([5.0, 5.0])
rng = np.random.default_rng(2)
X2 = Sampling(40, LHS()).sample((lo, hi), rng=rng)
y2 = aniso(X2)
X2t = Sampling(1000, LHS()).sample((lo, hi), rng=np.random.default_rng(3))
y2t = aniso(X2t)

from pysurrogate.selection.metrics import calc_metric
for ard in (False, True):
    m = Kriging(ARD=ard).fit(X2, y2)
    rmse = calc_metric("rmse", y2t, m.predict(X2t).y.ravel())
    print(f"ARD={ard!s:>5}  test RMSE={rmse:.4f}  theta={np.ravel(m.model.model['theta'])}")
```

With `ARD=True` the two fitted `theta` entries differ — a short length-scale for the fast
axis, a long one for the slow axis — and the test error drops.

## Regression trends

The trend is the deterministic part the GP models residuals around. Choose
`ConstantRegression`, `LinearRegression` (the default), or `QuadraticRegression`:

```{code-cell} ipython3
from pysurrogate.dace.regr import ConstantRegression, LinearRegression, QuadraticRegression

def trended(x):
    return 0.5 * x**2 - 2 * x + np.sin(4 * x)   # strong quadratic trend + wiggle

Xr = Sampling(8, LHS()).sample((xl, xu), rng=np.random.default_rng(5))
yr = trended(Xr).ravel()

plt.plot(grid, trended(grid), "k--", lw=1, label="truth")
for name, regr in [("Constant", ConstantRegression()),
                   ("Linear", LinearRegression()),
                   ("Quadratic", QuadraticRegression())]:
    mu = Kriging(regr=regr).fit(Xr, yr).predict(grid).y.ravel()
    plt.plot(grid, mu, label=name)
plt.scatter(Xr, yr, c="k", zorder=5)
plt.legend(loc="upper left"); plt.title("regression trend: extrapolation differs");
```

The trend dominates **extrapolation** where there is no data to correlate against — the
quadratic trend keeps rising sensibly past the last point.

## Theta optimization

Fitting Kriging means searching for the length-scale `theta` that maximizes the likelihood.
The search strategy is a pluggable [optimizer](optimizers.ipynb) — configured on the `Dace`
engine directly (the `Kriging` wrapper uses the default). Freeze `theta` entirely with
`optimize=False`.

```{code-cell} ipython3
from pysurrogate.optimizer import Boxmin, LBFGS, Restart

Xo = Sampling(15, LHS()).sample((xl, xu), rng=np.random.default_rng(7))
yo = truth(Xo).ravel()
Xot = np.linspace(0, 5, 500).reshape(-1, 1)
yot = truth(Xot).ravel()

configs = {
    "frozen (optimize=False)": (Dace(), False),
    "Boxmin (MATLAB DACE)": (Dace(optimizer=Boxmin()), True),
    "Restart+LBFGS (default)": (Dace(optimizer=Restart(LBFGS(), Sampling(16, LHS()))), True),
}
for name, (engine, opt) in configs.items():
    engine.fit(Xo, yo, optimize=opt)
    rmse = calc_metric("rmse", yot, engine.predict(Xot).y.ravel())
    print(f"{name:>26}  theta={float(np.ravel(engine.model['theta'])[0]):8.3f}  RMSE={rmse:.4f}")
```

`Restart(LBFGS(), ...)` — many sampled starts, each polished by gradient descent — is the
default and the most robust; `Boxmin` reproduces the classic MATLAB DACE search.

## Noise / nugget — from interpolation to regression

With `noise=0` Kriging **interpolates** (the mean passes exactly through every point). Add a
nugget — a fixed `noise=` or a learned `noise_bounds=` range — and it **regresses**, smoothing
through noisy observations:

```{code-cell} ipython3
rng = np.random.default_rng(11)
Xn = Sampling(25, LHS()).sample((xl, xu), rng=rng)
yn = truth(Xn).ravel() + rng.normal(0, 0.25, size=len(Xn))   # noisy labels

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, (title, engine) in zip(axes, {
    "noise=0 (interpolate)": Dace(noise=0.0),
    "noise=0.1 (fixed nugget)": Dace(noise=0.1),
    "noise_bounds (learned)": Dace(noise_bounds=(1e-4, 1.0)),
}.items()):
    engine.fit(Xn, yn)
    mu = engine.predict(grid).y.ravel()
    ax.plot(grid, truth(grid), "k--", lw=1)
    ax.plot(grid, mu, "C0")
    ax.scatter(Xn, yn, c="k", s=12, zorder=5)
    ax.set_title(title, fontsize=9)
fig.tight_layout()
```

## Predictive variance & calibration

The kriging variance is theoretically exact but empirically **overconfident**. `calibrate()`
fits one scalar multiplier by cross-validation so the intervals become honest, returning the
scale (`> 1` means it was overconfident):

```{code-cell} ipython3
engine = Dace()
engine.fit(Xn, yn)
sigma_before = engine.predict(grid, var=True).sigma.ravel()

scale = engine.calibrate()          # mutates engine.scale, rescales variance
sigma_after = engine.predict(grid, var=True).sigma.ravel()
print(f"calibration scale = {scale:.3f}")

mu = engine.predict(grid).y.ravel()
plt.plot(grid, truth(grid), "k--", lw=1, label="truth")
plt.plot(grid, mu, "C0", label="mean")
plt.fill_between(grid.ravel(), mu - 2*sigma_before, mu + 2*sigma_before, color="C1", alpha=0.2, label="±2σ before")
plt.fill_between(grid.ravel(), mu - 2*sigma_after, mu + 2*sigma_after, color="C0", alpha=0.2, label="±2σ after")
plt.scatter(Xn, yn, c="k", s=12, zorder=5)
plt.legend(loc="upper left", fontsize=8); plt.title("calibration widens the band to be honest");
```

## Gradients

Kriging returns the analytic gradient of the mean (and of the variance) from the same solve.
Here it matches a finite-difference check:

```{code-cell} ipython3
model = Kriging().fit(X, y)
pred = model.predict(grid, var=True, grad=True)

eps = 1e-5
fd = (model.predict(grid + eps).y.ravel() - model.predict(grid - eps).y.ravel()) / (2 * eps)

plt.plot(grid, pred.grad.ravel(), "C0", lw=2, label="analytic grad")
plt.plot(grid, fd, "k--", lw=1, label="finite difference")
plt.legend(); plt.title("mean gradient: analytic vs finite-difference");
```

`pred.var_grad` (returned when both `var=True` and `grad=True`) is the gradient of the
variance — the quantity active-learning acquisition functions differentiate.

## Multi-output

Stack targets column-wise and Kriging fits every output with the shared kernel:

```{code-cell} ipython3
Y = np.column_stack([truth(X).ravel(), np.cos(2 * X).ravel()])
multi = Kriging().fit(X, Y)
pred = multi.predict(grid)
print("prediction shape:", pred.y.shape)   # (m, 2)

for j in range(2):
    plt.plot(grid, pred.y[:, j], label=f"output {j}")
plt.scatter(np.tile(X, 2), Y.ravel(), c="k", s=12, zorder=5)
plt.legend(); plt.title("one Kriging model, two outputs");
```

## `Kriging` vs `Dace`

- `Kriging` is the `Model`-lifecycle wrapper — use it for the uniform `fit`/`predict`
  contract, normalization, duplicate elimination, and to sit inside [model selection](selection.ipynb). It uses the default theta optimizer.
- `Dace` is the underlying engine — reach for it when you need to configure the theta
  `optimizer=`, fixed/learned `noise=`, or call `calibrate()` directly.

## See also

- [optimizers](optimizers.ipynb) — the theta search is just a `Problem` for the generic optimizer layer.
- [models](models.ipynb) — how Kriging compares to the other backends.
