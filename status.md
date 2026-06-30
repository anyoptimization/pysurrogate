# Optimizer migration — status & handoff

## The goal

Replace the DACE-specific optimizer zoo with **one generic, composable optimizer layer** that any
surrogate can reuse, and delete the deprecated code. This is now essentially DONE.

- Optimizers are **generic** — they minimize an abstract bounded `Problem`, not a `Dace` object.
- Three concerns are **separated**: search (Optimizer) / objective (Problem) / selection+stop (Callback).
- **Never raise** on ill-conditioning — a bad candidate is `feasible=False`, the search steps around it.
- **noise is just another coordinate** of the search vector (folded in via `noise_bounds`).
- `noise` / `noise_bounds` mirror `theta` / `theta_bounds` (value = fixed/start, bounds = optimize).
- **MATLAB goldens must NOT drift** — now anchored by the *generic* `Boxmin` (a faithful port).

## Current state — green, 1 golden case awaiting a human bless

- `pyclawd check`: **green** (format / lint / typecheck / descriptions / 167 tests).
- `pyclawd doctor`: green.
- `pyclawd golden`: **18/19 bit-identical**. The one drift is `matern25/const/opt-scalar`, off by
  ~3e-9 in a single gradient element — pure log-space round-trip noise (the generic path commits
  `10**log10(theta)`, ≈theta ± 1e-16, and Matérn's gradient is the most sensitive). theta_traj,
  pred and mse all match. This is an *intended* consequence of switching golden to the generic
  Boxmin, so it needs a human `pyclawd golden update -k matern25` + commit. **Not yet blessed.**

## Semantics (decided this session)

- **`optimizer` decides whether to search**, not the bounds:
  - unset (default) → the generic `Restart(LBFGS(), Sampling(16, LHS()), screen=4)` → `Dace()` optimizes.
  - `optimizer=None` → **freeze theta** (no search, single GLS solve at the current theta). The
    replacement for the old `Fixed()`. A sentinel (`_DEFAULT_OPTIMIZER`) keeps "unset" ≠ "None".
- **`theta_bounds` is just the box**:
  - a pair → bounded search; `None` → **unbounded search** (the old "None = freeze" meaning is GONE).
  - Unbounded = positive (floored) but no ceiling. The generic layer seeds starts from a finite
    window (`Problem.sampling_bounds`) and the local descent leaves it freely (hard `bounds` may be
    `+inf`; scipy gets `None`, not `inf`, for an absent bound).
- **`noise` / `noise_bounds` mirror `theta` / `theta_bounds`**: `noise` is the fixed value (or the
  start when learning); `noise_bounds=(lo,hi)` learns the nugget jointly (its own coordinate +
  analytic gradient). `noise='auto'` is GONE. Learning needs an optimizer (a search).

## DONE this session

- **Removed `max_noise`** entirely (the silent auto-repair climb): `fit()` does a single Cholesky
  else raises `DaceFitError`; dropped the param everywhere (`fit`, `DaceProblem.fit`, `Dace`).
- **Deleted the legacy `dace/optimizers/` package entirely** (boxmin, base, screened, lbfgs, adam,
  fixed + `fit_feasible`, `objective_gradient`, `_select`, `supports_auto_noise`).
- **Generic `Boxmin`** (`pysurrogate/optimizer/boxmin.py`) — a faithful port of MATLAB DACE's
  Hooke & Jeeves. Boxmin searches theta multiplicatively; on the log-space `DaceProblem` the
  equivalent *additive* moves reproduce its trajectory **bit-for-bit** (multiply-θ-by-D ==
  add-log10(D)-to-log10θ). Verified ≤1e-16 on theta/trajectory across kernels.
- **`PatternSearch` left untouched** (general compass search; Sphere tests intact) — Boxmin is a
  separate dedicated class.
- **`sampling_bounds`** added to `Problem` (default = `bounds`); `DaceProblem` clamps infinite
  bounds to a finite seeding window. LBFGS/Restart/Adam/PatternSearch seed from `sampling_bounds`,
  constrain to `bounds`.
- **LBFGS default tolerances tightened** (`gtol=1e-6, ftol=1e-9, maxfun=200`) — the old `1e-3`
  stopped on the flat DACE likelihood after one eval; now the polish actually converges (verified
  it reaches the brute-force grid optimum).
- **`Dace.fit` unified**: one generic search path (`_optimize_generic`) + one frozen path; the two
  legacy optimizer branches removed. The visited trajectory is exposed as
  `optimization["models"]` (decoded from the optimizer's `visited`) for the golden/correctness
  theta-trajectory snapshots.
- **Tests migrated**: golden + correctness use the generic `Boxmin`; validation/refit/multioutput
  use generic `Boxmin`/`LBFGS`; `objective_gradient` checks replaced by FD on `batch_obj_grad`;
  `theta_bounds=None`-as-fixed → `optimizer=None`; cubic `max_noise` climb tests deleted;
  `noise='auto'` gone. Deleted `test_screened.py`, `test_optimizers.py`.

## Remaining

1. **Bless the one golden drift**: `pyclawd golden update -k matern25`, review the `git diff`
   (should be ~1e-9), commit. (Human step — agents compare, humans bless.)
2. Optional polish: a real validation `Callback` that predicts from `info` without re-fitting
   (currently `ValidationSelection` re-fits per candidate — correct but O(visited) fits).
3. Optional: trace the benign `DLASCL parameter number 4 had an illegal value` LAPACK stderr line
   in some degenerate tests (no test fails).

## Key files
```
core/optimizer.py        Problem (+ sampling_bounds) / Optimizer / Callback / Evaluation / Result
core/sampling.py         Sampling / LHS / Random
optimizer/*.py           LBFGS / PatternSearch / Boxmin (generic) / Adam / Restart
dace/problem.py          DaceProblem (log-space; unbounded clamp; learn-noise coordinate)
dace/selection.py        ValidationSelection
dace/fit.py              fit() (no max_noise) + batch_obj_grad
dace/dace.py             Dace — generic search path, optimizer/None + theta_bounds/noise_bounds semantics
```
