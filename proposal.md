# pysurrogate — architecture review & redesign proposal

A design-level companion to a code-quality pass over the whole framework. The pass was driven by
a multi-agent review (one reviewer per subsystem + a cross-cutting architecture reviewer), with
every finding adversarially re-checked against the real source. This document collects the
**architecture-level** findings — the ones that are *design decisions*, not mechanical fixes — and
proposes concrete redesigns. The smaller, safe fixes were already applied (see the last section);
everything below is left for you to decide, because each changes a public surface, a default, or a
reproducible output.

The framework is in good shape: layering is mostly clean (`core` → `dace`/`models`/`optimizer` →
`selection`), the new generic optimizer layer is a genuine improvement, and `pyclawd check` + 19
golden snapshots are green. The items below are about making the *seams* consistent.

Severity legend: **[H]** worth doing soon · **[M]** worth doing · **[L]** opportunistic.

---

## ★ [H] Core unification: one kernel family, one regression basis

The framework currently defines *kernels* in two places and *polynomial bases* in three. Both should
move to `core/` and be reused everywhere. This is the headline structural cleanup.

### A. `core/regression.py` — the polynomial basis (low-risk, do first)

The same "build a polynomial design matrix `P(X)` (+ its gradient)" exists three times:

| where | form |
|---|---|
| `dace/regr.py` | `Constant/Linear/QuadraticRegression` → `F(X)` `(m,p)` + `grad(X)` `(m,d,p)` — the clean version, already first-class objects |
| `models/rbf.py` | the polynomial **tail**, re-implemented inline as string keys (`"linear"`/`"quadratic"`/`"linear+quadratic"`) in `rbf_kernel`, with a *separate* hand-written `_tail_grad` |
| `models/regression.py` | `PolynomialRegression` builds polynomial features yet again |

`dace/regr.py`'s `Regression` base **is already the right abstraction**. Promote it verbatim to
`core/regression.py`:

```python
# core/regression.py
class Regression:
    def __call__(self, X): ...   # design matrix F, shape (m, p)
    def grad(self, X): ...       # Jacobian per row, shape (m, d, p)

class Constant(Regression): ...
class Linear(Regression): ...
class Quadratic(Regression): ...
```

Then:
- `dace/` imports trends from `core.regression` (drop `dace/regr.py`, keep names re-exported).
- `RBF`'s tail becomes "append `Linear()(X)` columns"; `_tail_grad` becomes `Linear().grad(X)` —
  two bespoke copies deleted.
- `PolynomialRegression` builds its features from the same `Quadratic`/`Linear`.

**Risk:** low. Pure relocation + call-site swaps; golden stays green (same numbers). This is the safe
first step.

> **Context (decided):** `RBF` is a *temporary* implementation — the `Dace` engine is always the
> better surrogate. So the kernel unification is **not** about preserving RBF; it is about giving the
> kernel zoo a single clean home in `core/`, with the option to either **retire `RBF`** or re-express
> it as a thin "DACE kernels + polynomial tail, no theta-search" configuration. Because we are not
> preserving RBF's separate scalar-distance path, the two "real details" below mostly evaporate (see
> the note after the sketch). The `ard` design still matters: it is how *one* kernel object serves
> both the isotropic and the per-dimension use without duplication.

### B. `core/kernel.py` — one kernel object, `ard` toggles isotropic vs per-dimension

The key insight (yours): **ARD is just "one length-scale per dimension" vs "one shared."** So a single
kernel object covers both — `ard=False` is the isotropic/RBF use, `ard=True` is the per-dimension/DACE
use. And `tps`/`mq`/`linear`/etc. can be written in the **DACE `k(D, theta)` style** too, not only as
RBF scalar-distance functions:

```python
# core/kernel.py
class Kernel:
    def __init__(self, ard=False): ...      # False: shared theta (RBF); True: per-dim theta (DACE)
    def __call__(self, D, theta): ...       # value on pairwise coordinate differences
    def dtheta(self, D, theta): ...         # d/dtheta — for likelihood optimization (DACE)
    def dr(self, D, theta): ...             # spatial derivative — for predict(grad=True) (RBF + DACE)

class Gaussian(Kernel): ...      # valid GP covariance
class Exponential(Kernel): ...   # valid GP covariance
class Cubic(Kernel): ...
class Matern(Kernel): ...
class ThinPlateSpline(Kernel): ...   # radial basis; conditionally-PD (needs the polynomial tail)
class Multiquadric(Kernel): ...      # radial basis; conditionally-PD
```

`Dace` uses the GP-covariance kernels (typically `ard=True`); `RBF` can use **any** of them
(typically `ard=False`) plus its polynomial tail. The kernel zoo is defined **once**.

**The two former "details", now that RBF is not being preserved:**
1. **ARD needs per-dimension information.** A kernel feeds on coordinate differences `D` (what DACE
   already passes), so `ard=True` scales each axis by its own theta and `ard=False` shares one theta.
   Since RBF's separate scalar-`r` path is going away (RBF retired or rebuilt on the DACE
   representation), there is no second representation to reconcile — the kernel just takes `D`.
2. **Not every kernel is a valid GP covariance.** `tps`/`mq` are only conditionally positive-definite,
   so as a raw Kriging correlation they can give a non-PD `R`. Already handled by the *never-raise /
   infeasible* contract (`DaceFitError` → the optimizer steps around it). If a tail-based RBF-style
   model is kept, the polynomial tail makes them well-posed. Either way, **whether a given model
   accepts a given kernel stays the model's concern** — we don't pretend `tps` is a covariance.

**Risk:** medium — touches the golden-critical DACE kernel path, so each kernel migrates with golden
as the anchor (the existing 19 snapshots prove the DACE kernels are byte-identical after the move).
Writing `tps`/`mq` in the `k(D, theta)` style (per your note) lets them live in the same zoo with
`ard=False` as their natural setting.

### Why this is worth it
- One kernel zoo and one basis instead of 2× and 3× copies — bug-fixes and new kernels land once.
- The `core` layer becomes the real home of the modeling primitives, with `dace`/`models` as
  assemblies on top — matching the layering the rest of the framework already follows.
- It clarifies the path for `RBF`: either delete it, or make it a thin "shared kernel + polynomial
  tail, no theta-search" `Model` — no longer a parallel kernel implementation.

---

## 1. [H] Two parallel function-benchmark stacks

**Where:** `selection/study.py` (`study()` / `StudyResult`) vs `selection/benchmark.py`
(`FunctionBenchmark` / `score`).

**Problem.** These are two independent implementations of one job: *sample a known function over a
box, fit a model fleet across repeated draws, score with the metric registry, rank
direction-aware*. They share nothing:

| concern        | `study()` / `StudyResult`            | `FunctionBenchmark` / `score`             |
|----------------|--------------------------------------|-------------------------------------------|
| sampling       | private `_sample` (`'lhs'`/`'random'`) | `core.sampling.Sampling(LHS()/Random())`  |
| sigma handling | private `_predict_with_sigma`        | `predictions_frame` (already clamps)      |
| result shape   | nested dicts + bespoke `StudyResult` | tidy predictions DataFrame + `score`      |
| ranking        | `StudyResult.ranking` (now shared key) | `Benchmark`-style                       |

A user faces two unrelated APIs for one task, and every metric/bugfix must be applied twice.
`study._sample`'s LHS branch is byte-for-byte `core.sampling.LHS.__call__`; `_predict_with_sigma`
re-implements the NaN-sigma clamp `predictions_frame` already owns.

**Proposal.** Collapse to one engine. Make `study()` a thin front-end that builds a
`FunctionBenchmark`, runs it, and reduces the resulting predictions DataFrame via `score` /
groupby. `StudyResult` survives only as a reporting view (its `__str__`/`ranking` over the shared
frame) if its console output is valued. Net: delete `_sample` and `_predict_with_sigma`, one
sampler, one sigma policy, one scoring path. This also resolves the two duplication findings
(`study._sample` re-implements `core.sampling`; `_predict_with_sigma` re-implements
`predictions_frame`).

---

## 2. [H] `Dace`'s public API diverges from the `Model` contract

**Where:** `dace/dace.py` vs `core/model.py` (and `models/kriging.py`, which bridges them).

**Problem.** `Dace` is the flagship engine, but its surface does not match the `Model` contract
that every other backend follows, so `Kriging` exists partly as a translation shim:

| operation | `Model` (and all backends)            | `Dace`                                      |
|-----------|---------------------------------------|---------------------------------------------|
| predict   | `predict(X, var=False, grad=False)`   | `predict(_X, mse=False, grad=False)`        |
| fit lever | `fit(X, y, optimize=True)`            | `fit(X, Y, validation, append)` + `optimizer=`|
| refit     | `refit(X, y, optimize)`               | `refit(X, Y, optimize, validation=True)`    |

Three different signatures for fit/predict/refit between `Dace` and `Kriging`/`Model`. The leading
underscore on `Dace.predict(_X, ...)` is a public parameter that reads as private, and `mse=` vs
`var=` forces `Kriging._predict` to translate.

**Proposal.** Align `Dace` to the `Model` vocabulary:
- rename `predict(_X, mse=...)` → `predict(X, var=...)`. Keep `mse=` as a deprecated alias if the
  DACE-literate audience expects it (mirror `Prediction.mse`, which already aliases `var`).
- reconcile the "don't search" lever so `optimize=` / `optimizer=None` / `validation` mean the same
  thing across the two front-ends — ideally `Dace.fit` grows an `optimize=True` that maps to
  "use the configured optimizer vs freeze", matching `Model.fit`, with `optimizer=` remaining the
  *strategy* choice.

This is the change most likely to reduce long-term friction, but it touches the public API — hence
proposal, not auto-applied.

---

## 3. [M] `AutoModel` (was `ModelSelection`) — renamed ✓; still bypasses the lifecycle

> **Done:** renamed `ModelSelection` → **`AutoModel`** (clean rename, no alias — nothing is
> published yet). The remaining work below is making the inheritance *honest*.


**Where:** `selection/benchmark.py`.

**Problem.** `AutoModel` subclasses `Model` yet overrides `fit`/`predict`/`refit`/`records`
and never implements `_fit`/`_predict`. None of the machinery `Model` advertises runs —
normalization, nan/inf filtering, duplicate elimination, active-dims, exception capture,
postprocess un-normalization. It borrows only `Model.__init__`'s `_validation`/`_epoch` fields. The
"is-a `Model`" claim is currently faked: it is really a *composition* facade that delegates to a
chosen sub-model.

**The drop-in `fit`/`predict` is a feature, not the problem.** Being able to hand an `AutoModel` to
anything that expects a model — `fit`, then `predict` — is genuinely convenient UX and worth
keeping. So the real issue is **honesty**, not the model-like interface. (The name is now `AutoModel`,
which says "a surrogate that auto-selects" — done.)

**Remaining proposal — make it a *genuine* `Model`:** implement `_fit` = run the benchmark + store
the winning prototype, `_predict` = delegate to the winner. The lifecycle then *actually* owns
pre/postprocess, so the `is-a Model` claim becomes true and the convenient `fit`/`predict` surface is
real, not faked.

(The earlier "honest facade that doesn't inherit `Model`" option is *rejected* — it would throw away
the convenient drop-in interface, which is the thing worth preserving.)

---

## 4. [M] The package root hides the headline abstractions

**Where:** `pysurrogate/__init__.py`.

**Problem.** The root exports only `Dace, Kriging, Model, Prediction, Benchmark, ModelSelection,
cartesian`. The deliberately-built generic optimizer layer (`Optimizer`, `Problem`, `Callback`,
`Sampling`, `LBFGS`/`PatternSearch`/`Boxmin`/`Adam`/`Restart`) is **not** reachable from the root,
nor are `study` / `FunctionBenchmark` / `score`. Yet `Dace`'s own docstring tells users to pass
`optimizer=Boxmin()` / `LBFGS()` by bare name — names only importable via
`from pysurrogate.optimizer import ...`. The newest, most reusable contract is the least
discoverable.

**Proposal.** Surface the optimizer layer and the study/selection front-ends at the root (or via
documented `pysurrogate.optimizer` / `pysurrogate.selection` namespaces the docstrings point to).
At minimum, export the optimizer names the `Dace` docstring already references.

---

## 5. [M] Optimizer trajectory (`visited`) is an ad-hoc Boxmin↔Dace side-channel

**Where:** `optimizer/boxmin.py` (defines `self.visited`) ↔ `dace/dace.py`
(`getattr(self.optimizer, "visited", None)`).

**Problem.** Only `Boxmin` records `visited`; `LBFGS`/`PatternSearch`/`Adam`/`Restart` do not. `Dace`
reaches in via `getattr`, so the theta-trajectory snapshot silently exists for `Boxmin` and
silently vanishes for every other optimizer. The base `Optimizer` contract
(`core/optimizer.py`) never mentions a trajectory hook — it is an undocumented protocol between two
classes.

**Proposal.** Promote it into the contract: declare `visited` (default `None`/empty) on the
`Optimizer` base with documented semantics, or have the base `_emit` optionally append to a
base-level trajectory list. Then `Dace` consumes a *documented* attribute, and any optimizer can
opt in. (Boxmin/pattern searches populate it; gradient methods may leave it empty — that's fine, as
long as it's contractual.)

---

## 6. [M] Two core randomness conventions: `Partitioning` reseeds globals, `Sampling` threads a Generator

**Where:** `core/partitioning.py` (`random.seed` + `np.random.seed`, then global `random.shuffle` /
`np.random.permutation`) vs `core/sampling.py` (explicit `np.random.Generator`).

**Problem.** `Partitioning.do` reseeds the **process-global** RNGs and the subclasses consume the
module globals. So calling `do()` perturbs any other code relying on global numpy/random state, and
concurrent partitionings interfere. The sampling layer does the right thing with a threaded
`Generator`. Two opposite conventions for the framework's two randomness consumers.

**Proposal.** Switch `Partitioning` to a local `np.random.default_rng(self.seed)` (and a local
`random.Random(self.seed)` if Python-`random` shuffling is kept), threaded through `_folds`,
matching `Sampling`.

**Caveat (why this was *not* auto-applied):** `default_rng(seed).permutation` produces a *different*
sequence than `np.random.seed(seed); np.random.permutation`, so fold assignments shift. Any test or
golden that pins specific CV outputs would need re-blessing. Worth doing, but it is a deliberate
reproducibility change for a human to bless.

---

## 7. [L] Per-optimizer accounting & gradient-detection are inconsistent

**Where:** `optimizer/{lbfgs,adam,boxmin}.py`.

Three small contract inconsistencies in the otherwise-clean optimizer layer:
- **`n_evals` undercounts.** `Boxmin._relocate` (up to 64 feasibility probes) and `LBFGS`'s
  gradient-detection probe both call `self.problem(...)` without incrementing `n_evals`. `n_evals`
  is surfaced on `Result`, so it under-reports the true cost — and for a `DaceProblem` each probe is
  a full GLS solve.
- **Gradient detection differs.** `LBFGS` spends a throwaway problem evaluation in `setup` just to
  learn whether `.grad` exists; `Adam` checks lazily in `_advance`. Two conventions for one question.
- **Fail-fast timing differs.** `Adam` raises "requires analytic gradient" mid-run in `_advance`,
  whereas `requires_x0` fails fast in `setup`.

**Proposal.** Add a `Problem.has_grad` flag (or a single counted helper on the `Optimizer` base)
so gradient support is detected once, cheaply, and consistently; count relocation/detection probes
in `n_evals` (or document them as deliberately excluded); move Adam's gradient precondition to
`_setup`. None changes search results — only honesty of `n_evals` and where the error surfaces.

---

## 8. [L] Bounds-extraction boilerplate duplicated across four optimizers

**Where:** `optimizer/{lbfgs,adam,pattern,restart}.py`.

`lo, hi = (np.atleast_1d(np.asarray(b, float)) for b in self.problem.bounds)` and the parallel
`sampling_bounds` unpack are copy-pasted verbatim in four optimizers. **Proposal:** a small
`Optimizer._box()` helper returning `(lo, hi, slo, shi)` as float arrays, called from each
`_setup`. Pure dedup, easy win — left out of the applied pass only to avoid touching all four
optimizers without your sign-off.

---

## 9. [L] `KNN` and `IDW` give the same name `p` two different meanings

**Where:** `models/knn.py` vs `models/idw.py`.

Both expose a power parameter `p`, but `IDW` weights `1/D**p` over the **true** Euclidean distance
(`euclidean_dist`, default `p=3`), while `KNN` raises **squared** distance (`calc_dist`) to `**p`
(default `p=2`) — so KNN's effective exponent on true distance is `2p`, and its
"inverse-distance-weighting" docstring is really inverse-*squared*-distance. The identically named
knob means different things between two sibling models.

**Proposal.** Make the distance basis consistent (use `euclidean_dist` in KNN, or rename/document
the exponent). At minimum, correct the KNN docstring to say it weights by squared distance.

---

## 10. [L] RBF kernel library has three sharp edges

> Note: `RBF` is a temporary implementation slated for retirement or a rebuild on the shared
> `core/kernel.py` (★B). If it is removed, these are moot; if it is rebuilt, they are fixed for free
> while defining the unified kernel zoo. Listed here for completeness.


**Where:** `models/rbf.py`.

- **`kernel_gaussian` is a quartic, not a Gaussian.** Kernels receive `r = calc_dist(...)` = the
  *squared* distance `D`. `kernel_gaussian` computes `exp(-sigma * r**2) = exp(-sigma * ||x-xi||**4)`.
  A Gaussian in the squared-distance argument is `exp(-sigma * r)`. The gradient is internally
  consistent with the quartic, and `tps` is the default so golden doesn't exercise it — but the
  `'gaussian'` label is mathematically wrong. **Proposal:** either use `exp(-sigma * r)` (and update
  `dkernel_gaussian`) or rename the kernel to reflect what it is.
- **`kernel_periodic` hard-codes period 5** as a bare literal (and again in `dkernel_periodic`),
  unconfigurable, with `r` being squared distance so the geometric meaning is unclear. **Proposal:**
  expose the period as a named kwarg with a documented default.
- **`kernel_gaussian(sigma=None)`** default would raise on a bare call; it only works because the
  RBF model always passes `sigma`. **Proposal:** give it a real default.

These are semantic/numeric decisions, so they belong here rather than in the mechanical pass — and
they are naturally resolved while rewriting the kernels into the shared `core/kernel.py`
(★ section): a correct `Gaussian`, a `period`-parametrized periodic kernel, and real defaults all
land as part of defining the unified kernel zoo once.

---

## 11. [L] Decide the fate of `Dace._val_error` and `Correlation.has_theta_grad`

**Where:** `dace/dace.py`, `dace/corr.py`.

- `Dace._val_error` is unused in production (held-out theta selection now flows through
  `ValidationSelection.score`); it survives only because a test pins it. Its docstring was corrected
  in this pass to stop claiming the optimizer uses it. **Decision:** keep it as a documented
  standalone scorer, or delete it (and its test) and let `ValidationSelection` own the formula
  outright. There is a third copy of the normalized-prediction formula (`predict` /
  `_val_error` / `ValidationSelection.score`) worth factoring into one helper.
- `Correlation.has_theta_grad` has no consumer in `src` (only a test). `DaceProblem` always calls
  `kernel.theta_grad` unconditionally. **Decision:** either wire `DaceProblem`/`LBFGS` to consult it
  and fall back to finite differences for kernels without an analytic theta-gradient, or remove the
  property and its now-overstated docstring.

---

## 12. [L] Inconsistent cross-validation defaults

**Where:** `selection/benchmark.py` (`CrossvalidationPartitioning(k_folds=3, seed=1)`) vs
`dace/dace.py` `calibrate` (`k_folds=5`).

The same conceptual choice ("how many folds to estimate held-out performance") has two different
defaults in one framework, so benchmark ranking and variance calibration disagree on what "default
CV" means. **Proposal:** one shared default (a named constant or a default-partitioning factory)
referenced by both.

---

## Applied in this pass (already committed to the working tree, all gated green)

These were safe enough to apply directly — each was verified by `pyclawd check` (format / lint /
typecheck / descriptions / tests) **and** the 19 golden snapshots after every batch, so observable
behavior is provably unchanged for the covered paths. New regression tests were added for the
behavioral bug fixes.

**Correctness bugs**
- `SimpleMean` now returns `(m, q)` for multi-output targets (previously crashed for `q > 1`).
- `Standardization.forward` guards a zero-std (constant) dimension instead of dividing into NaN/inf
  (matches `ZeroToOneNormalization` and the Dace fit).
- `Plog.forward`/`backward` allocate float output, so integer-typed targets are no longer truncated.
- `Model.predict` on a failed fit uses the *promoted* row count (a 1-D query point is one row, not
  `d`), and honors `raise_exception_while_prediction` (not the fitting flag) on the failed-fit path.
- `Sampling.sample` returns shape `(0, p)` (not `(0,)`) on the empty edge case.
- `Benchmark.results`/`frame` now rank **target-aware** (e.g. calibration metrics rank by distance
  to their target), via a new shared `metrics.metric_sort_key` that `StudyResult.ranking` also uses —
  so the two rankers agree. `results()` also raises a clear error if `sorted_by` isn't a computed
  metric (was a `KeyError`).
- `ModelSelection.fit`/`do` now forward `optimize` (was silently dropped, making the cheap-screen
  path unreachable through the public API).
- `RBF._fit` honors the fit-time `optimize` flag (gates the sigma grid on
  `self.optimize and optimize`), matching Kriging — so `optimize=False` screening is cheap for RBF.
- `Adam` masks infeasible candidates' gradients to zero before the step (the comment claimed this
  but nothing enforced it; `LBFGS` already did).

**Hygiene / consistency**
- `Transformation` now inherits `ABC`, so its `@abstractmethod`s are actually enforced.
- `kernel_tps` no longer mutates its input array (uses `np.where`, matching `dkernel_tps`).
- Deleted dead `corr.calc_grad`; removed a no-op `super().__init__()` in `Dace`.
- Corrected stale docstrings/comments: `theta_bounds=None` (unbounded search, not "freeze"),
  `batch_obj_grad` (`noise='auto'` and `objective_gradient` are gone), `Dace._val_error`
  (no longer used by the optimizer). Added the missing `fit()` docstring and one-line docstrings for
  the RBF `kernel_*`/`dkernel_*` helpers.
- Routed `is_duplicate` through `util.dist.euclidean_dist`; imported `predictions_frame` from its
  home module; hoisted a per-fold `set()` out of an inner loop in `CrossvalidationPartitioning`.

---

## Suggested order of attack

1. **#4** (exports) — trivial, immediately improves discoverability of the optimizer layer.
2. **★A** (`core/regression.py` basis) — low-risk, deletes 3→1 copies, golden stays green. The first
   concrete unification step.
3. **★B** (`core/kernel.py`, `ard` toggle) — medium-risk, golden-anchored; makes `RBF` and `Dace`
   two configurations of one kernel family.
4. **#1** (one benchmark engine) — biggest reduction in duplicated surface.
5. **#2** (`Dace` ↔ `Model` API alignment) + **#3** (`AutoModel`: genuine `Model` + rename) — biggest
   reduction in long-term user friction.
6. **#5** (trajectory contract), **#6** (RNG) — clean up the optimizer/core seams.
7. The **[L]** items as opportunistic polish (several fold into ★B automatically).
