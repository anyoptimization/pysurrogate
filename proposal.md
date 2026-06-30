# pysurrogate — refactor handoff

**You are an agent picking up a refactor of the `pysurrogate` framework.** This document is
self-contained: read it top-to-bottom, then execute the tasks in **Backlog** order. Each task says
what to change, which files, how to verify, and when it's done. You do **not** need any prior
conversation — everything you need is here or in `AGENTS.md`.

---

## 1. Orientation (read first, ~5 min)

`pysurrogate` is a surrogate-modeling toolkit (sampling, fitting, model selection). It is driven by
**pyclawd** — read `AGENTS.md` in the repo root for the full operational contract. The layering:

```
core/        backend-agnostic primitives
  model.py          Model: the fit/predict/refit lifecycle (pre/postprocess, normalization)
  prediction.py     Prediction result type + predictions_frame (tidy DataFrame)
  optimizer.py      generic Problem / Optimizer / Callback contract  (NEW layer)
  sampling.py       Sampling / LHS / Random
  transformation.py normalization transforms
  partitioning.py   cross-validation splits
dace/         DACE Kriging engine (the flagship surrogate)
  dace.py           Dace: fit / predict / theta search
  fit.py            GLS fit() + batch_obj_grad (likelihood + analytic gradient)
  corr.py           correlation kernels (Gaussian/Exponential/Matern/Cubic/...)
  regr.py           regression trend basis (Constant/Linear/Quadratic)
  problem.py        DaceProblem: likelihood as a generic Problem
  selection.py      ValidationSelection callback
optimizer/    generic optimizers over a Problem: lbfgs / boxmin / adam / pattern / restart
models/       Model backends: kriging / rbf / idw / knn / forest / svr / regression / mean
selection/    benchmark / metrics / study / factory  (benchmarking + AutoModel)
util/         dist / misc / test_functions
```

Mental model after a recent refactor: the **generic optimizer layer** (`core/optimizer.py`,
`core/sampling.py`, `optimizer/*`, `dace/problem.py`, `dace/selection.py`) is deliberate and good —
do **not** undo it. `Dace` drives it: `optimizer=None` freezes theta, an optimizer present searches,
`theta_bounds=None` means an *unbounded* search (not "freeze"), noise is just another search
coordinate.

---

## 2. Working agreement (non-negotiable)

1. **Run Python only via `pyclawd python`.** Never bare `python`/`pip`.
2. **The gate after every change — both must stay green:**
   - `pyclawd check` → format-check, lint, typecheck, descriptions, tests.
   - `pyclawd golden` → 19 DACE behavior snapshots. **These numbers must not drift.** golden is the
     behavior oracle: a clean refactor can still move a number, and golden is what catches it.
3. **If golden drifts:** that is a *real regression* until proven intended. Fix the cause. Do **not**
   run `pyclawd golden update` to paper over it — blessing a baseline is a **human** step
   (AGENTS.md: "agents compare, humans bless"). If you believe a drift is genuinely intended, stop
   and surface it to the user with the `git diff` of the baseline.
4. **Small, verified batches.** One coherent change → run the gate → only then continue. Add a
   regression test for every behavior change.
5. **Do not `git commit` / `git push` unless the user explicitly asks** (AGENTS.md: ask first). Leave
   work in the tree; tell the user it's ready.
6. **Match existing style.** Google-style docstrings (no types in the docstring — annotations carry
   them); every module opens with a one-line docstring (the `descriptions` gate enforces it). Read a
   neighbor file before writing a new one.
7. **When a task touches the public API or a default,** make the change but call it out clearly in
   your summary so the user can veto.

Verification cheatsheet:
```bash
pyclawd check                 # full gate
pyclawd golden                # behavior oracle (must stay 19/19)
pyclawd test -k EXPR          # run a subset by keyword
pyclawd test path::node       # run one test
pyclawd test all              # include slow tier before declaring a big task done
```

---

## 3. Already done — do NOT redo

A quality + review pass already landed (all gated green, with regression tests). Treat this as the
baseline; don't re-discover it:

- **Bugs fixed:** `SimpleMean` multi-output; `Standardization` zero-std guard; `Plog` int truncation;
  `Model.predict` failed-fit row-count + exception-flag; `Sampling` empty `(0,p)` shape; `Benchmark`
  target-aware ranking via the shared `metrics.metric_sort_key` (also used by `StudyResult`);
  `AutoModel`/`Benchmark` forward the `optimize` flag; `RBF` honors the fit-time `optimize`; `Adam`
  masks infeasible gradients.
- **Hygiene:** `Transformation` is now `ABC`; `kernel_tps` no longer mutates input; dead `calc_grad`
  and a no-op `super().__init__()` removed; stale docstrings corrected; distance helper / re-export
  centralized; a per-fold `set()` hoisted.
- **Rename done:** `ModelSelection` → **`AutoModel`** (clean, no alias — nothing is published yet).
  Exported from `pysurrogate` and `pysurrogate.selection`.

---

## 4. Backlog — execute in this order

Each task is independent enough to do and verify on its own. Severity: **[H]** high value · **[M]** ·
**[L]** opportunistic. Stop and ask the user before starting any **[H]** task that changes the public
API (Tasks 5 and 6 are flagged).

---

### Task 1 — Surface the public API at the package root  ·  [M] · low risk

**Why.** `pysurrogate/__init__.py` exports only `Dace, Kriging, Model, Prediction, Benchmark,
AutoModel, cartesian`. The generic optimizer layer (`Optimizer`, `Problem`, `Callback`, `Sampling`,
`LBFGS`/`PatternSearch`/`Boxmin`/`Adam`/`Restart`) and the study/selection front-ends
(`study`, `FunctionBenchmark`, `score`) are unreachable from the root — yet `Dace`'s own docstring
tells users to pass `optimizer=Boxmin()` / `LBFGS()` by bare name. The headline abstraction is
undiscoverable.

**Do.** Add the optimizer-layer names and the study/selection front-ends to `pysurrogate/__init__.py`
(and its `__all__`). Either export at the root or via documented `pysurrogate.optimizer` /
`pysurrogate.selection` namespaces the docstrings point to. At minimum, every name the `Dace`
docstring references must be importable as written.

**Files.** `src/pysurrogate/__init__.py` (+ check `optimizer/__init__.py`, `selection/__init__.py`,
`core/__init__.py` re-export what you need).

**Verify.** `pyclawd python -c "import pysurrogate; from pysurrogate import LBFGS, Boxmin, Optimizer, Problem, Sampling"`;
`pyclawd check`. (golden unaffected.)

**Done when.** Every optimizer/sampling/selection name a user needs is importable from `pysurrogate`,
`__all__` lists them, `pyclawd check` green.

---

### Task 2 — `core/regression.py`: one polynomial basis  ·  ★ [H] · low risk · **do this first of the structural tasks**

**Why.** "Build a polynomial design matrix `P(X)` (+ gradient)" exists three times:
`dace/regr.py` (`Constant/Linear/QuadraticRegression`, the clean first-class version),
`models/rbf.py` (the polynomial *tail*, re-implemented inline as string keys in `rbf_kernel`, with a
separate hand-written `_tail_grad`), and `models/regression.py` (`PolynomialRegression`). One
implementation should serve all three.

**Do.**
1. Create `core/regression.py` and move `dace/regr.py`'s `Regression` base + `Constant/Linear/
   Quadratic` there verbatim (the API is already right: `__call__(X) -> (m,p)`, `grad(X) -> (m,d,p)`).
   Keep the `dace` names working (re-export from `dace/regr.py` or update `dace` imports).
2. Rebuild `models/rbf.py`'s polynomial tail on it: the tail columns become `Linear()(X)` /
   `Quadratic()(X)`, and `_tail_grad` becomes the basis `.grad(X)`. Delete the bespoke copies.
3. Have `models/regression.py` build its features from the same basis.

**Files.** new `core/regression.py`; `dace/regr.py`, `dace/dace.py` (imports); `models/rbf.py`,
`models/regression.py`; `core/__init__.py` (export).

**Verify.** `pyclawd golden` (DACE trend is golden-critical — must stay byte-identical);
`pyclawd check`; `pyclawd test -k "rbf or regression"`. Add a small test asserting the RBF tail
columns equal the basis output.

**Done when.** Three basis implementations collapse to one in `core/`, golden 19/19, check green.

**Risk.** Low — pure relocation + call-site swaps, same numbers.

---

### Task 3 — `core/kernel.py`: one kernel object, `ard` toggle  ·  ★ [H] · medium risk

**Why.** Kernels are defined twice: `dace/corr.py` (`k(D, theta)` style, per-dim `theta`, analytic
`theta`-gradient, valid GP covariances) and `models/rbf.py` (`kernel_*` on a scalar squared distance,
scalar `sigma`, `dkernel_*` derivatives, includes conditionally-PD radial bases `tps`/`mq`). They are
the *same idea* at different parametrizations.

**Key design (decided with the user):** ARD is just "one length-scale per dimension" vs "one shared."
So a **single** kernel object covers both via an `ard` flag — `ard=False` is the isotropic/RBF use,
`ard=True` the per-dimension/DACE use. And `tps`/`mq` get written in the **same `k(D, theta)` style**
(not as RBF scalar-distance functions), living in the one zoo with `ard=False` as their natural
setting.

```python
# core/kernel.py — target shape
class Kernel:
    def __init__(self, ard=False): ...   # False: shared theta; True: per-dim theta
    def __call__(self, D, theta): ...    # value on pairwise coordinate differences
    def dtheta(self, D, theta): ...      # d/dtheta — likelihood optimization (DACE)
    def dr(self, D, theta): ...          # spatial derivative — predict(grad=True)

class Gaussian(Kernel): ...      # valid GP covariance
class Exponential(Kernel): ...
class Cubic(Kernel): ...
class Matern(Kernel): ...
class ThinPlateSpline(Kernel): ...   # conditionally-PD radial basis
class Multiquadric(Kernel): ...
```

**Context (decided):** `RBF` is a **temporary** implementation — `Dace` is always the better
surrogate. So this task is **not** about preserving RBF. It is about giving the kernel zoo one clean
home in `core/`. `RBF` should then either be **retired** or rebuilt as a thin "shared kernel +
polynomial tail (Task 2), no theta-search" `Model`. Because RBF's separate scalar-distance path is
not being preserved, the kernel just takes coordinate differences `D` (what DACE already passes) —
there's no second representation to reconcile.

**Do.**
1. Define `core/kernel.py` with the `Kernel` protocol above (port `dace/corr.py`'s kernels; add the
   `ard` flag — `ard=True` reproduces today's per-dim DACE behavior exactly).
2. Point `Dace` at `core.kernel` (keep `dace/corr.py` names re-exported, or update imports).
3. Decide RBF's fate with the user (retire vs rebuild). If rebuilt, express its kernels (incl.
   `tps`/`mq`) as `core.kernel` kernels with `ard=False` + the Task-2 polynomial tail.
4. While here, fix the kernel correctness items from Task 9's RBF notes *as part of defining the zoo*
   (a correct Gaussian, a `period`-parametrized periodic kernel, real defaults).

**Files.** new `core/kernel.py`; `dace/corr.py`, `dace/dace.py`, `dace/problem.py`; `models/rbf.py`;
`core/__init__.py`.

**Verify.** `pyclawd golden` is the anchor — migrate **one kernel at a time** and re-run golden after
each; the 19 snapshots prove the DACE kernels are byte-identical after the move. `pyclawd check`;
`pyclawd test -k "kernel or corr or matern or cubic"`. Add tests that `ard=False` with equal thetas
matches the old isotropic behavior and `ard=True` matches the per-dim DACE path.

**Done when.** Kernels live once in `core/kernel.py`; `Dace` uses them with golden 19/19 byte-identical;
RBF retired or rebuilt on the shared zoo; check green.

**Risk.** Medium — touches the golden-critical DACE kernel path. Golden makes it safe if you migrate
incrementally.

---

### Task 4 — One function-benchmark engine  ·  [H] · medium risk

**Why.** `selection/study.py` (`study()` / `StudyResult`) and `selection/benchmark.py`
(`FunctionBenchmark` / `score`) are two independent implementations of one job: sample a known
function over a box, fit a model fleet across draws, score with the metric registry, rank
direction-aware. They share nothing — `study._sample` is a hand-rolled copy of `core.sampling`,
`study._predict_with_sigma` re-implements `predictions_frame`'s NaN-sigma clamp. A user faces two
unrelated APIs for one task.

**Do.** Make `study()` a thin front-end over `FunctionBenchmark`: it builds the benchmark, runs it,
and reduces the resulting predictions DataFrame via `score` / groupby. Delete `study._sample`
(use `core.sampling.Sampling`) and `_predict_with_sigma` (use `predictions_frame`). Keep
`StudyResult` only as a reporting view over the shared frame if its `__str__`/`ranking` output is
valued (its `ranking` already uses the shared `metric_sort_key`).

**Files.** `src/pysurrogate/selection/study.py`, `src/pysurrogate/selection/benchmark.py`.

**Verify.** `pyclawd test -k "study or function_benchmark"`; `pyclawd check`. Pin the new `study()`
output against the old (a golden-style snapshot test of a small deterministic study is worth adding).

**Done when.** One sampler, one sigma policy, one scoring path; `study()` delegates to
`FunctionBenchmark`; tests green.

---

### Task 5 — Align `Dace` with the `Model` contract  ·  [H] · **PUBLIC API — confirm with user first**

**Why.** `Dace` (flagship) doesn't match the `Model` contract every other backend follows, so
`Kriging` partly exists to translate:

| op | `Model` / backends | `Dace` |
|---|---|---|
| predict | `predict(X, var=False, grad=False)` | `predict(_X, mse=False, grad=False)` |
| fit lever | `fit(X, y, optimize=True)` | `fit(X, Y, validation, append)` + `optimizer=` |
| refit | `refit(X, y, optimize)` | `refit(X, Y, optimize, validation=True)` |

The leading-underscore `_X` is a public param that reads as private; `mse=` vs `var=` forces
`Kriging._predict` to translate.

**Do.** Rename `Dace.predict(_X, mse=...)` → `predict(X, var=...)`; reconcile the "don't search" lever
so `optimize=` / `optimizer=None` / `validation` mean the same across `Dace` and `Kriging`/`Model`
(ideally `Dace.fit` grows `optimize=True` mapping to "use the configured optimizer vs freeze", with
`optimizer=` remaining the *strategy* choice). Update `Kriging` to drop the translation shim.

**Files.** `src/pysurrogate/dace/dace.py`, `src/pysurrogate/models/kriging.py`, plus the many DACE
tests that call `predict(mse=...)` / `fit(...)`.

**Verify.** `pyclawd golden` (numbers must not move — this is a signature change, not a math change);
`pyclawd check`; full `pyclawd test`.

**Done when.** `Dace` and `Kriging` share the `Model` vocabulary; `Kriging` no longer translates;
golden 19/19, check green.

---

### Task 6 — Make `AutoModel` a *genuine* `Model`  ·  [M] · **behavior-adjacent — confirm with user**

**Why.** `AutoModel` subclasses `Model` but overrides `fit`/`predict`/`refit`/`records` and never
implements `_fit`/`_predict`, so none of the lifecycle (normalization, nan/inf filtering, duplicate
elimination, active-dims, postprocess) actually runs — the `is-a Model` claim is faked. The drop-in
`fit`/`predict` UX is the *good* part and must stay; the issue is honesty. (The name is already fixed:
it's `AutoModel`, "a surrogate that auto-selects".)

**Do.** Implement `_fit` = run the benchmark + store the winning prototype, `_predict` = delegate to
the winner, so the `Model` lifecycle genuinely owns pre/postprocess. Remove the ad-hoc overrides that
duplicate `Model`.

**Files.** `src/pysurrogate/selection/benchmark.py` (the `AutoModel` class).

**Verify.** `pyclawd test -k "selection or auto"`; `pyclawd check`. Confirm the existing AutoModel
tests still pass and add one asserting the lifecycle runs (e.g. normalization is applied).

**Done when.** `AutoModel` is a real `Model` with the same convenient interface; tests green.

---

### Task 7 — Promote the optimizer trajectory into the contract  ·  [M] · low risk

**Why.** Only `Boxmin` defines `self.visited`; `Dace` reaches in via
`getattr(self.optimizer, "visited", None)` (`dace/dace.py`). So the theta-trajectory snapshot silently
exists for `Boxmin` and vanishes for every other optimizer — an undocumented side-channel, not part of
the `Optimizer` contract.

**Do.** Declare `visited` (default `None`/empty) on the `Optimizer` base in `core/optimizer.py` with
documented semantics, or have the base `_emit` optionally append to a base-level trajectory list. Then
`Dace` consumes a *documented* attribute and any optimizer can opt in.

**Files.** `src/pysurrogate/core/optimizer.py`, `src/pysurrogate/optimizer/boxmin.py`,
`src/pysurrogate/dace/dace.py`.

**Verify.** `pyclawd golden` (trajectory feeds golden theta-trajectory snapshots — must stay
identical); `pyclawd check`; `pyclawd test -k "optimizer or boxmin"`.

**Done when.** `visited` is contractual; the `Dace` `getattr` reach-in is gone; golden 19/19.

---

### Task 8 — `Partitioning`: local RNG instead of reseeding globals  ·  [M] · **changes reproducible fold assignments**

**Why.** `core/partitioning.py` `do()` calls `random.seed(self.seed)` + `np.random.seed(self.seed)`
and the subclasses use the module-global `random.shuffle` / `np.random.permutation`. This perturbs
any other code relying on global RNG state, and concurrent partitionings interfere. `core/sampling.py`
does the right thing with a threaded `Generator`.

**Do.** Switch to a local `np.random.default_rng(self.seed)` (and a local `random.Random(self.seed)`
if Python-`random` shuffling is kept), threaded through `_folds`.

**Important caveat.** `default_rng(seed).permutation` produces a *different* sequence than
`np.random.seed(seed); np.random.permutation`, so **fold assignments shift**. Any test or golden that
pins CV outputs will change. This is a deliberate reproducibility change → run the gate, and if golden
moves, **stop and ask the user to bless** the new baselines (do not self-bless).

**Files.** `src/pysurrogate/core/partitioning.py`.

**Verify.** `pyclawd check`; `pyclawd golden`. Expect possible golden/test movement → escalate, don't
auto-update.

**Done when.** No global RNG reseed; the gate is green (after a human blesses any intended drift).

---

### Task 9 — Opportunistic cleanups  ·  [L] · low risk (bundle)

Small, independent polish. Do any subset; each is self-verifying with `pyclawd check` + `pyclawd
golden`.

- **Optimizer accounting (`optimizer/{lbfgs,adam,boxmin}.py`):** `n_evals` undercounts
  (`Boxmin._relocate` probes, `LBFGS`'s gradient-detection probe are uncounted). Add a
  `Problem.has_grad` flag (or one counted helper on the `Optimizer` base) so gradient support is
  detected once and consistently; count the probes in `n_evals` or document them as excluded; move
  `Adam`'s "requires gradient" check from `_advance` to `_setup` (fail fast, like `requires_x0`).
- **Bounds-extraction boilerplate (`optimizer/{lbfgs,adam,pattern,restart}.py`):** the
  `lo, hi = (np.atleast_1d(np.asarray(b, float)) for b in self.problem.bounds)` + `sampling_bounds`
  unpack is copy-pasted four times. Add `Optimizer._box() -> (lo, hi, slo, shi)` and call it.
- **`KNN` vs `IDW` `p` (`models/knn.py`, `models/idw.py`):** same param name, different meaning —
  `IDW` uses true distance `1/D**p` (default 3), `KNN` raises *squared* distance to `**p` (default 2),
  so KNN's effective exponent on true distance is `2p`. Make the distance basis consistent or
  rename/document; at minimum correct the KNN docstring (it weights by squared distance).
- **RBF kernel correctness (`models/rbf.py`)** — *fold into Task 3 if RBF is rebuilt; skip if retired:*
  `kernel_gaussian` computes `exp(-sigma*r**2) = exp(-sigma*||x||**4)` (a quartic, not a Gaussian; a
  Gaussian over squared-distance `r` is `exp(-sigma*r)`); `kernel_periodic` hard-codes period `5` as a
  bare literal (make it a named kwarg); `kernel_gaussian(sigma=None)` has no usable default.
- **`Dace._val_error` / `Correlation.has_theta_grad` (`dace/dace.py`, `dace/corr.py`):** both are
  unused in production (only pinned by tests). Decide per item: keep as a documented standalone
  utility, or delete (and its test). There's a third copy of the normalized-prediction formula
  (`predict` / `_val_error` / `ValidationSelection.score`) worth factoring into one helper.
- **CV default mismatch:** `Benchmark.do` defaults `CrossvalidationPartitioning(k_folds=3, seed=1)`
  while `Dace.calibrate` defaults `k_folds=5`. Share one default (a named constant / factory).
- **`evaluate()` redundant guard (`selection/metrics.py`):** the default-`names` filter and the
  in-loop `PROBABILISTIC and sigma is None: continue` encode the same intent twice with subtly
  different consequences (explicit names silently drop probabilistic metrics when sigma is missing,
  rather than erroring). Pick one point of truth.

---

## 5. Definition of done (per task and overall)

- `pyclawd check` green (format-check, lint, typecheck, descriptions, tests).
- `pyclawd golden` 19/19 — and any intended drift was **blessed by a human**, not self-updated.
- `pyclawd doctor` exits 0.
- Every behavior change has a regression test.
- Public-API / default changes are called out to the user.
- You did **not** commit or push unless the user asked.

When in doubt, prefer the smaller change, keep golden green, and ask.
