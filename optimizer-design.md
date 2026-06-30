# Generic optimizer layer — design & status

Built this session in `pysurrogate`. All green: `pyclawd check` (200 tests) + golden (19), clean.
Replaces the DACE-specific optimizer zoo with a backend-free, composable contract.

## The contract (`core/optimizer.py`)

```
Problem    bounds (a box) + __call__(X) -> Evaluation     # what to minimize; NEVER raises
           screen(X) -> objective only                    # cheap rank hook (default = __call__)
Optimizer  construct -> setup(problem, x0, callback) -> advance()* / run() -> Result
Callback   __call__(x, f, info) -> stop?                   # selection AND termination, one hook
```

### Lifecycle — construct / setup / advance
Three phases owned by different parties (why a single `minimize(problem, x0, …)` felt wrong):
1. **construct** — the *user* picks the strategy + hyperparameters: `LBFGS(sampling=…)`.
2. **`setup(problem, x0, callback)`** — the *framework* binds runtime context the user doesn't
   have (problem, the warm `x0`, the selection callback). `requires_x0` raises here if a pure
   local optimizer got no start. `minimize(...)` is one-shot sugar for `setup(...).run()`.
3. **`advance()`** — the steppable primitive: **one iteration** (a poll / a gradient step / one
   local descent). `run()` loops it. Exposing the single step lets a driver interleave several
   optimizers and **race** them on `callback.best_score` — no optimizer knows it's in a race.

### Callback = selection + termination
Tracks the best by an overridable `score(x, f, info)` (default `f` = MLE; override for
validation/MAP) **and** returns `True` to stop after `patience` stale steps. Search value ≠
selection value: the optimizer descends `f`, the callback may re-score from `info`.

## Sampling (`core/sampling.py`)
One start-generation strategy, injected wherever starts are needed (no duplication):
`Sampling(n, method=LHS()|Random(), include=[…])` → `.sample(bounds, rng, include=[x0])`. Forced
points (prior optima, and the runtime `x0`) are **guaranteed members** of the sample — so a known
`x0` is just one more guaranteed sample, not a special case.

## Generic optimizers (`optimizer/`)
- **`LBFGS(sampling=…)`** — quasi-Newton; analytic gradient when present. Multi-start = a
  `Sampling` (replaces the old `n_restarts`); one descent per start.
- **`PatternSearch`** — derivative-free compass search; one poll per iteration.
- **`Adam(sampling=…)`** — vectorized population descended in lock-step; one batched step per
  iteration; population seeded by `Sampling`.
- **`Restart(inner, sampling, screen=k)`** — wrap *any* inner optimizer, run it from sampled
  starts, keep the best continuously. `screen=k` = cheap-rank the pool (`Problem.screen`) and
  polish only the best k. `Restart(LBFGS(), Sampling(64, LHS()), screen=8)` ≡ old ScreenedLBFGS.
- **No `Fixed`** — "don't search" is just `optimizer=None` at the model layer (cleaner).

## DACE binding (`dace/problem.py`)
`DaceProblem` maps `x = [log θ…, (log noise)]` → batched GLS likelihood (`batch_obj_grad`) →
objective + analytic gradient, never raising. Noise folds in via `noise_bounds` (the `Rk = I`
gradient term, FD-verified). `batch_obj_grad` gained an additive `noise_grad=False` — golden
unchanged.

**Noise default = `1e-6` (pinned), not 0.** A tiny fixed nugget gives PD-robustness + softens
variance over-confidence at zero search cost. Learning (`noise_bounds`) is opt-in and only pays
off with noisy data or a validation callback — likelihood alone drives the nugget back to ~0.

## Still open (deliberate)
1. **Wire `Dace`/`Kriging` to drive this layer** (it lives alongside `dace/optimizers/`, nothing
   ripped out yet). The model supplies `x0` (current θ) and the selection callback at fit time.
2. **A real validation `Callback`** that predicts held-out from `info` (DaceProblem `info=None`).
3. **MAP** as a third `Callback` — unverified-here, not disproven.
4. **Graceful-commit** — fold `fit_feasible` relocate/climb into the never-raise contract.

## Files
- `core/optimizer.py` — Problem / Optimizer (lifecycle + advance) / Callback / Evaluation / Result
- `core/sampling.py` — Sampling / LHS / Random
- `optimizer/{lbfgs,pattern,adam,restart}.py` — generic optimizers
- `dace/problem.py` — DaceProblem (+ `screen` cheap hook); `dace/fit.py` — `batch_obj_grad` noise grad
- `tests/core/test_optimizer.py`, `tests/dace/test_problem.py` — ~40 tests
