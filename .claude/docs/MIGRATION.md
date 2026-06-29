# pysurrogate — migration status

Consolidation of **pydacefit** (DACE Kriging engine) and **ezmodel** (model layer) into a
single, pyclawd-driven package. This file tracks what landed and what is intentionally left.

_Sources: `~/workspace/pydacefit`, `~/workspace/ezmodel`. Target: `~/workspace/pysurrogate`
(`github.com/anyoptimization/pysurrogate`)._

## Status: core merge complete ✅

Both libraries are merged and green: `pyclawd check` passes (format / lint / typecheck /
descriptions / tests), 165 unit tests + 19 golden snapshots.

### Package layout

```
src/pysurrogate/
  dace/         DACE Kriging engine (was pydacefit). Class DACE -> Dace.
                corr, regr, fit, optimizers/ (Boxmin, LBFGS, ScreenedLBFGS, Adam, Fixed)
  core/         the backend-agnostic contract
    prediction.py     single Prediction (var canonical; mse/sigma/var_grad aliases)
    model.py          Model fit/predict lifecycle (normalize -> _fit/_predict -> postprocess)
    transformation.py NoNormalization, Standardization, ZeroToOneNormalization, Plog
    metrics.py        metric registry: accuracy/fit/ranking/selection/calibration
    partitioning.py   Split(train/test/valid), CV + random hold-out, valid_frac
  models/       Model backends: Kriging, RBF, SVR, KNN, InverseDistanceWeighting,
                SimpleMean, PolynomialRegression, RandomForest
  selection/    cartesian factory, Benchmark (CV scoring), ModelSelection (pick+refit),
                study() function-sampling harness + StudyResult
  util/         dist.py (squared/euclidean), misc.py (at_least2d/is_duplicate/discretize),
                test_functions.py (sphere/ackley/rastrigin/griewank/rosenbrock/sine)
```

### Key design decisions

- **Single `Prediction`.** `dace` depends on `core` and re-exports it. `var` is canonical;
  `mse` / `mse_grad` are alias properties so DACE-literature code (and the golden tests) keep
  reading `.mse`.
- **`Dace` is the Kriging.** No duplicate Kriging implementation — `models/kriging.py` is a thin
  `Model` adapter over the engine (adds duplicate-elimination + the uniform lifecycle).
- **Optimizers stay in `dace`.** They are Kriging-theta-MLE solvers bound to the engine, not
  general minimizers — promoting them to `core` would be a speculative abstraction (YAGNI).
- **Metrics are one registry.** Direction-aware (`greater_is_better`), family-grouped, point vs.
  probabilistic; `Benchmark`/`ModelSelection`/`study` all sort and group through it.
- **`study()` vs `Benchmark`.** Renamed ezmodel's top-level `benchmark()` to `study()` to avoid a
  case-collision with the dataset-CV `Benchmark` class. Rename back if you prefer the old name.

## Remaining — optional / peripheral

None of this blocks a v0.1.0; it is the "do you want the extra surfaces" list.

### 1. Heavy optional backends — deferred to `[extras]`
Source: `ezmodel/models/{gpy,gpflow,smt}.py` (and the original `pysotrbf`).
- Need external deps (GPy, GPflow, SMT, pySOT). Port as lazy-imported optional extras:
  `pip install pysurrogate[gpy]`, etc., each a `Model` subclass mirroring the others.
- Suggested: add `[project.optional-dependencies]` groups + `models/_optional/` with import guards.

### 2. Niche utilities — port only if used
- `ezmodel/util/aggregate/{clearing,front_wise,grid}.py` — multi-objective aggregation helpers.
  Unclear if still used downstream; confirm before porting (else drop).
- `ezmodel/util/sample_from_func.py` — small dataset-from-function helper. Largely superseded by
  `study()`'s internal sampling + `util/test_functions.py`. Port only if a public sampler is wanted.

### 3. Dropped on purpose
- `ezmodel/experimental/*` — scratch/experimental code.
- `ezmodel/fit.py` (top-level) — superseded by the `Model`/`Dace` fit paths.
- pydacefit `usage.py` — example script (its golden/correctness coverage was ported as real tests).

## Packaging TODO before a public release
- [ ] Decide final name for `study()` (keep, or rename to `benchmark`).
- [ ] Add `[project.optional-dependencies]` for the heavy backends (if wanted).
- [ ] Fill out `README.md` usage beyond Kriging (selection + study examples).
- [ ] Tag `v0.1.0` and publish to PyPI (`pysurrogate` name is reserved/available).
- [ ] Retire the old `pydacefit` / `ezmodel` repos (point their READMEs here).
