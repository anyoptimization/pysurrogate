# pysurrogate — Documentation Plan

The blueprint for pysurrogate's online documentation. This file is the **spec**: what to
build, how it's wired, and exactly what every page contains. Execute it top-to-bottom.

Decisions (locked): **medium ~8-page site**, **full autodoc API reference**, **every feature
shown as an executed + plotted notebook cell**. Stack and layout **mirror pysampling and pymoo**
(both in `~/workspace`) — no new tooling is introduced.

---

## 1. Goals

- Explain **every public feature** of pysurrogate with a runnable, plotted example.
- Match the anyoptimization house style (Sphinx + notebook sources + `sphinx-book-theme`).
- Publish to `https://anyoptimization.com/projects/pysurrogate/`.
- Stay comprehensive but far leaner than pymoo (~8 pages, not ~25 sections).

---

## 2. Stack & toolchain (identical to pysampling)

| Concern | Choice |
|---|---|
| Renderer | **Sphinx** + **nbsphinx** (`nbsphinx_execute = "never"`) |
| Theme | **sphinx-book-theme** (light mode, GitHub buttons) |
| Page source | **MyST Markdown** (`.md`) paired to `.ipynb` via **jupytext** |
| Execution | Out-of-band via `docs/cli.py`, cached by **jupyter-cache** |
| API ref | **autodoc + numpydoc** (Google-style docstrings already in the code) |
| Copy button | **sphinx-copybutton** |
| Driver | **pyclawd `DocsConfig`** → `python docs/cli.py` |
| Hosting | S3 `blankjul` + CloudFront `E3O32PKOWAIOFZ` (anyoptimization.com) |

**Pipeline:** `compile` (jupytext `.md`→`.ipynb`) → `run` (execute notebooks in parallel, cache
successes) → `render` (Sphinx renders hydrated `.ipynb`). Execution and rendering are separate,
so a text fix re-renders in seconds.

---

## 3. Files to create

```
docs/
  cli.py                      # the build runner (port from pysampling/docs/cli.py verbatim)
  README.md                   # "how to build these docs" (port from pysampling)
  .gitignore                  # build/, .jupyter_cache/executed/, *.ipynb (generated)
  source/
    conf.py                   # Sphinx config (see §5)
    jupytext.toml             # formats: ipynb,md:myst
    llms.txt                  # LLM-readable one-page project summary (shipped to site root)
    _static/
      custom.css              # port pysampling's minimal CSS
      pysurrogate.png         # logo (see §9 — needs an asset)
    _templates/
      layout.html             # port pysampling's (analytics/meta if any)
    index.md                  # Page 1 — Home
    getting_started.md        # Page 2
    models.md                 # Page 3
    kriging.md                # Page 4 (flagship)
    optimizers.md             # Page 5
    sampling.md               # Page 6 (sampling + partitioning + transforms)
    selection.md              # Page 7 (benchmark + AutoModel + study + metrics)
    api.md                    # Page 8 — API reference (autodoc via MyST eval-rst)
```

Everything under `docs/` is **excluded from ruff** (`extend-exclude = ["docs"]`) — it is a
self-contained builder, like pysampling's.

---

## 4. Wiring pyclawd + pyproject

**`.pyclawd/config.py`** — add a `DocsConfig` to `Project(...)`:

```python
from pyclawd import DocsConfig  # add to the import
# ...
docs=DocsConfig(
    runner=["python", "docs/cli.py"],
    source_dir="docs/source",
    cache_dir="docs/.jupyter_cache",
    cache_db="docs/.jupyter_cache/global.db",
    build_html="docs/build/html",
    branch="main",              # pysurrogate's default branch
),
```

**`pyproject.toml`** — add a PEP 735 `docs` dependency group (NOT a shipped extra), copied
from pysampling:

```toml
[dependency-groups]
docs = [
  "sphinx>=7,<9", "nbsphinx>=0.9,<1.0", "sphinx-book-theme>=1.1,<2.0",
  "sphinx-copybutton", "numpydoc",
  "jupytext", "jupyter-cache", "nbclient", "nbconvert", "nbformat",
  "matplotlib", "ipython", "ipykernel",
]
```

And exclude docs from ruff: `extend-exclude = ["docs"]`.

Install once: `pip install -e . --group docs`. Then `pyclawd docs build` / `serve` work.

---

## 5. `conf.py` specifics (from pysampling, retargeted)

- `project = "pysurrogate"`, `author = "Julian Blank"`, `release = pysurrogate.__version__`.
- `sys.path.insert(0, "../../src")`; `import pysurrogate`.
- `extensions = ["nbsphinx", "sphinx_copybutton", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "numpydoc"]`.
  (pysampling is single-page so ships only `nbsphinx`; we add autodoc/napoleon/numpydoc for Page 8.)
- `html_theme = "sphinx_book_theme"`, `html_logo = "_static/pysurrogate.png"`, `html_title = "pysurrogate"`.
- `html_baseurl = "https://anyoptimization.com/projects/pysurrogate/"`.
- `html_theme_options`: `default_mode="light"`, repository buttons →
  `https://github.com/anyoptimization/pysurrogate`, `repository_branch="main"`,
  `use_repository_button/use_issues_button/use_source_button=True`.
- **Multi-page** (unlike pysampling): keep the normal left nav (a real toctree in `index.md`),
  and let the right sidebar show the in-page TOC (do NOT copy pysampling's single-page
  `html_sidebars`/`secondary_sidebar_items:[]` override).
- `nbsphinx_execute = "never"`; `html_extra_path = ["llms.txt"]`; `html_copy_source = False`.
- `napoleon_google_docstring = True` (our docstrings are Google-style, no types).

---

## 6. Shared conventions across pages

To keep examples comparable and cheap to execute:

- **Reproducibility:** every stochastic call takes a fixed `random_state` / `seed`. Never touch
  global RNG (mirror pysampling's messaging — our `Sampling` and `Partitioning` use local RNGs).
- **A single plotting helper**, defined once per page (or a short repeated snippet): a 1-D
  "predict-with-confidence-band" plot (`y ± 2σ` shaded) and a 2-D contour helper. This is the
  visual backbone — most pages reuse it.
- **Test functions from `pysurrogate.util.test_functions`** (`get_test_function`): use `sphere`
  (smooth), `ackley`/`rastrigin` (multimodal) so plots are recognizable and cheap.
- **Small designs** (n≈20–60) so notebooks execute in seconds and cache well.
- Each page opens with a one-paragraph "what you'll learn" and ends with a "see also" linking
  sibling pages (`{doc}` refs).

---

## 7. Page-by-page content (the 8 pages)

Every ✦ bullet is a runnable, plotted (where visual) notebook cell.

### Page 1 — `index.md` (Home)
- Tagline: *"A unified surrogate-modeling toolkit — sampling, Kriging/DACE, a model zoo,
  a generic optimizer layer, and model selection."*
- Badges (python, license), `pip install -U pysurrogate`.
- ✦ **Hero example** (≤20 lines): sample a 1-D function → `Kriging().fit(X, y)` →
  `predict(var=True)` → plot mean + 2σ confidence band + training points.
- **Feature grid** (cards): Kriging/DACE · Model zoo · Uncertainty & calibration · Generic
  optimizers · Sampling · Model selection (AutoModel) · Metrics.
- Cite/about block (anyoptimization, Julian Blank), links to GitHub/issues.
- `toctree` listing pages 2–8.

### Page 2 — `getting_started.md`
The end-to-end surrogate workflow as one narrative.
- ✦ Sample a design over a box (`Sampling(30, LHS())` or `sample`), label with a test function.
- ✦ Fit `Kriging`, `predict(var=True, grad=True)`; plot mean, confidence band, and the gradient.
- ✦ The `Prediction` type — read `y`, `var`, `sigma`, `grad` (not tuple positions).
- ✦ **Active-learning loop**: `refit(X_new, y_new)` returns the out-of-sample `Prediction`
  (prequential); plot error shrinking as points are added; show `records()` as a tidy frame.
- ✦ Evaluate: `score`/metrics on a held-out set.
- "Where to go next" → Models, Kriging, Selection.

### Page 3 — `models.md` (the backend zoo)
- The **`Model` contract**: `fit(X, y, optimize=)`, `predict(X, var=, grad=)`, `refit`,
  normalization (`norm_X`/`norm_y`), nan/inf filtering, duplicate elimination, `records()`.
- ✦ One 1-D fit + plot **per backend**: `Kriging`, `RBF`, `SVR`, `KNN`,
  `InverseDistanceWeighting`, `SimpleMean`, `PolynomialRegression`, `RandomForest`.
- ✦ **Side-by-side comparison** on a test function: a small table of RMSE (via `Benchmark`
  or `score`) + an overlay plot of every model's prediction.
- Note which backends report **uncertainty** (`var`) vs not (feeds the Selection page).
- `Prediction` / `predictions_frame` cross-ref to API.

### Page 4 — `kriging.md` (DACE — the flagship, deepest page)
- What Kriging/DACE is (GP interpolation + trend + correlation).
- ✦ **Correlation kernels** (the zoo): plot the kernel shapes and resulting fits for
  `Gaussian`, `Exponential`, `Matern(nu=…)`, `RationalQuadratic(alpha=…)`, `Cubic`, `Spline`,
  `Spherical`, `GeneralizedExponential`. Show `Multiquadric`/`ThinPlateSpline` as the radial
  bases too.
- ✦ **ARD** (`ard=True`): anisotropic length-scales on a 2-D function with unequal axis scales.
- ✦ **Regression trends**: `ConstantRegression`/`LinearRegression`/`QuadraticRegression`.
- ✦ **Theta optimization**: `optimizer=` choice (LBFGS/Restart/Boxmin), `theta_bounds`,
  freeze via `optimize=False` / `optimizer=None`; show the fitted length-scale and likelihood.
- ✦ **Noise / nugget**: `noise=` fixed vs `noise_bounds=` learned; interpolation → regression GP.
- ✦ **Predictive variance + `calibrate()`**: show the calibration ratio before/after; plot
  intervals tightening/widening honestly.
- ✦ **Gradients**: mean gradient and variance gradient (`var_grad`) vs finite differences.
- ✦ **Multi-output** fit/predict.
- Note: `Kriging` (Model wrapper) vs `Dace` (engine) — when to use which.

### Page 5 — `optimizers.md` (the generic optimizer layer)
- The contract: `Problem` (bounded, never-raises), `Optimizer` (setup/advance/run),
  `Callback` (selection + early stop), `Evaluation`/`Result`.
- ✦ Define a toy `Problem` (e.g. a bounded sphere) and minimize it with each strategy:
  `LBFGS`, `PatternSearch`, `Boxmin`, `Adam`, `Restart` — plot the trajectories (`visited`).
- ✦ How Dace uses it: the theta search IS a `Problem`; show `Restart(LBFGS(), Sampling(...))`
  as the default and `Boxmin()` as the MATLAB-DACE anchor.
- ✦ Reusing the layer standalone for any bounded optimization (acquisition maximization, etc.).

### Page 6 — `sampling.md` (sampling · partitioning · transforms)
- ✦ **Sampling**: `Sampling(n, LHS())` vs `Random()`; plot the point spread; reproducibility
  via local RNG (`random_state`), no global-state pollution.
- ✦ **Partitioning**: `CrossvalidationPartitioning`, `RandomPartitioning`, the `Split` type,
  `default_partitioning` / `DEFAULT_CV_FOLDS`; show fold coverage.
- ✦ **Transformations**: `Standardization`, `ZeroToOneNormalization`, `Plog`, `NoNormalization`
  — forward/backward round-trip and effect on a fit.

### Page 7 — `selection.md` (benchmarking & selection)
- ✦ **`Benchmark`**: cross-validate a fleet on one dataset, rank by a metric; show `.frame()`.
- ✦ **`AutoModel`**: a drop-in `Model` that auto-selects — `fit`/`predict` like any model;
  show the chosen winner and that the lifecycle (normalization) runs.
- ✦ **`FunctionBenchmark` + `study`**: sweep a known function, aggregate over repeats; show the
  tidy predictions frame and `StudyResult` ranking; the train-noise option.
- ✦ **Metrics registry**: the taxonomy (accuracy / fit / ranking / selection / calibration),
  `score`, `evaluate`, `metric_sort_key` (direction-aware); a table across models.
- ✦ **Test functions**: `get_test_function`, `TEST_FUNCTIONS`.
- `cartesian` / `as_named` for building model fleets.

### Page 8 — `api.md` (API reference — full autodoc)
MyST `eval-rst` blocks with `automodule`/`autoclass` (`:members:`), grouped to match the pages:
- **Core**: `Model`, `Prediction`, `predictions_frame`, transforms, partitioning, sampling.
- **Kriging/DACE**: `Dace`, kernels, regression trends, `Kriging`, `DaceFitError`.
- **Models**: RBF, SVR, KNN, IDW, SimpleMean, PolynomialRegression, RandomForest.
- **Optimizers**: `Problem`, `Optimizer`, `Callback`, `Evaluation`, `Result`, LBFGS/
  PatternSearch/Boxmin/Adam/Restart.
- **Selection**: `Benchmark`, `AutoModel`, `FunctionBenchmark`, `study`, `StudyResult`,
  `score`, metrics registry, `cartesian`/`as_named`.

---

## 8. `llms.txt`

A single plaintext page (shipped to site root via `html_extra_path`) summarizing: what
pysurrogate is, install, the headline API (`Kriging`, `Dace`, `AutoModel`, `study`), and links
to the 8 pages. Mirror pysampling's `llms.txt` format.

---

## 9. Assets still needed

- **Logo** `_static/pysurrogate.png` — a project logo in the anyoptimization visual family
  (pysampling/pymoo have one each). Placeholder acceptable for first build; ask Julian for the
  final asset.
- Confirm the **GitHub repo URL** (`anyoptimization/pysurrogate`) and default branch (`main`)
  for the theme's repository buttons.

---

## 10. Build & deploy

**Local build**
```bash
pip install -e . --group docs
pyclawd docs build          # compile → execute (cached) → render
pyclawd docs serve          # preview docs/build/html
pyclawd docs failures       # debug any notebook that failed to execute
```

**Deploy** (per CLAUDE.md — S3 + CloudFront):
```bash
aws s3 sync docs/build/html/ s3://blankjul/web/anyoptimization.com/html/projects/pysurrogate/ --delete
aws cloudfront create-invalidation --distribution-id E3O32PKOWAIOFZ --paths "/projects/pysurrogate/*"
```
Served at `https://anyoptimization.com/projects/pysurrogate/`.

---

## 11. Execution order (suggested)

1. **Scaffold**: port `docs/cli.py`, `README.md`, `conf.py`, `jupytext.toml`, `_static`,
   `_templates` from pysampling; add `DocsConfig` + `docs` group; `pip install -e . --group docs`;
   confirm an empty `pyclawd docs build` renders.
2. **Page 1 (index)** + toctree → get the hero example executing and the site navigable.
3. **Pages 2–7** in order; run `pyclawd docs build` after each so the cache stays warm.
4. **Page 8 (API)** once the narrative is stable.
5. `llms.txt`, logo, polish (`custom.css`), then **deploy**.

**Definition of done:** `pyclawd docs build` is clean (no execution failures), every public
feature in §7 has a runnable example, the API page renders the full surface, and the site is
live at the anyoptimization URL.
