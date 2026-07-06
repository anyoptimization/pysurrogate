# Refactor handoff — completed work (proposal.md backlog)

All 9 backlog tasks in `proposal.md` were worked. **Gate is green at every step and now:**
`pyclawd check` ✓ (format/lint/typecheck/descriptions/test) · `pyclawd golden` 19/19
byte-identical · `pyclawd doctor` exit 0 · `pyclawd test all` 248 passed (was 211).
**Nothing was committed or pushed** (per AGENTS.md "ask first").

> Dev-env note: I ran `pip install -e .` so `src/` is the live source. The package was a
> non-editable site-packages copy that only some commands synced, which made verification
> unreliable. The editable install is the standard dev setup; revert with a plain
> `pip install .` if you prefer.

## Task-by-task

| # | Task | Status |
|---|------|--------|
| 1 | Surface public API at package root | ✅ Done |
| 2 | `core/regression.py` — one polynomial basis | ✅ Done (1 deliberate scope call) |
| 3 | `core/kernel.py` — one kernel zoo + `ard` | ✅ Done; **RBF rebuild deferred for your bless** |
| 4 | One function-benchmark engine | ✅ Done |
| 5 | Align `Dace` with the `Model` contract | ✅ Done (public API — see below) |
| 6 | Make `AutoModel` a genuine `Model` | ✅ Done |
| 7 | Promote optimizer `visited` into the contract | ✅ Done |
| 8 | `Partitioning` local RNG | ✅ Done green (chose RandomState over default_rng — see below) |
| 9 | Opportunistic cleanups | ✅ Most done; 3 sub-items skipped w/ rationale |

## Decisions that need your eye (public API / judgment calls)

1. **Task 5 — `Dace.predict` signature changed (public API).** `predict(_X, mse=…)` →
   `predict(X, var=…, grad=…)` to match the `Model` vocabulary; `Kriging` no longer
   translates. `mse=` is kept as a **backward-compatible alias** (the `Prediction` type
   already keeps both names), so existing `predict(mse=True)` callers/tests still work.
   `Dace.fit` grew an `optimize=True` lever (freeze vs search per-fit), mirroring `Model`.

2. **Task 3 — RBF rebuild DEFERRED for a human decision.** The kernel zoo now lives once
   in `core/kernel.py` (DACE kernels moved verbatim → golden byte-identical; `ard` flag +
   `n_theta`; `ThinPlateSpline`/`Multiquadric` added in the unified `k(D,θ)` style). I did
   **not** rebuild/retire `RBF` because: (a) it changes *all* RBF predictions with **no
   golden anchor**, and (b) RBF's radial `"cubic"/"linear"/"quadratic"` names **collide**
   with DACE's compact-support `Cubic/Linear` (different math). The proposal makes RBF's
   fate a "decide with the user" step. RBF stays functional on its current path; its
   polynomial *tail* already shares `core/regression.py` (Task 2). **Tell me retire vs
   rebuild and I'll finish it** (a rebuild will also fold in the Task-9 RBF
   gaussian-quartic correctness fix, which changes numbers).

3. **Task 8 — chose local `RandomState`/`random.Random`, not `default_rng`.** This fixes
   the actual defect (global RNG reseeding / concurrency interference) **with zero
   sequence drift**, so golden/tests stayed green and **no bless was needed**. The proposal
   suggested `default_rng`, which *would* shift fold assignments and need a human bless —
   which I can't do autonomously. Migrating to `default_rng` later is a one-line change
   that **will** require you to bless new baselines.

4. **Task 2 — `PolynomialRegression` left on sklearn.** I collapsed the two *hand-rolled*
   polynomial duplications (DACE trend + RBF tail) onto `core/regression.py`. I did **not**
   move `PolynomialRegression` off sklearn's `PolynomialFeatures` — that would cap it at
   degree ≤2 and drop the `StandardScaler` conditioning (a capability/numerics regression).

## Task 9 — skipped sub-items (with rationale)

- **RBF `kernel_gaussian` quartic bug**: tied to the deferred RBF rebuild (changes RBF
  numbers, no golden anchor). I did do the safe `kernel_periodic` `period` kwarg fix.
- **Normalized-prediction formula (3rd copy in `predict`/`_val_error`/`ValidationSelection`)**:
  factoring it touches the golden-critical `Dace.predict` path for low value — left as-is.
- **`Dace._val_error` / `Kernel.has_theta_grad`**: kept as documented standalone utilities
  (the lowest-risk of the "keep or delete" options).

## New files
`core/kernel.py`, `core/regression.py`, and tests: `tests/core/test_kernel.py`,
`tests/dace/test_model_contract.py`, `tests/models/test_rbf_tail.py`,
`tests/selection/test_automodel_lifecycle.py` (plus additions to existing test files).
Every behavior change got a regression test.
