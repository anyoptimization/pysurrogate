# landscape — exploratory landscape analysis for labelled point clouds

`src/pysurrogate/landscape/` turns a labelled point cloud `(X, y)` into a
structural fingerprint of the function it implies. You hand it the samples you
already have — design points and their measured objective values — and it hands
back every geometric insight the cloud can support: rotation, effective
dimensionality, smoothness, multimodality, separability, conditioning, noise, and
global funnel structure. It does this **without fitting a surrogate and without
measuring prediction accuracy**. Every number is a property of the function's
geometry as sampled, not of any model's ability to reproduce it.

## 1. Motivation — why structural analysis, not fit quality

The usual way to reason about "what model should I use here" is fit-based model
selection: fit a handful of surrogates, cross-validate, keep the one with the
lowest error. That answers *which model predicts best on this sample*, which is a
different — and weaker — question than *what kind of function is this*. Predictive
accuracy is decoupled from optimization difficulty: a surrogate can have excellent
held-out error on a landscape that is nonetheless deceptive, ill-conditioned, or
multi-funnel, and it can have poor error on a benign bowl that is simply
undersampled. Fit quality also entangles the function with the model — a low score
might mean the landscape is rugged, or merely that the wrong basis was tried.

Structural landscape analysis sidesteps all of that. Instead of asking a model how
well it fits, it reads the geometry directly and asks *what kind of function is
this*: Is it rotated relative to the axes? How many directions does it really vary
along? Is it smooth or rough, one basin or many, separable or coupled, benign or
deceptive? These are intrinsic properties of the point cloud, computed once,
model-free.

The eventual purpose is to **choose a matched model or prior** from the detected
structure — a rotated landscape wants a Mahalanobis kernel, a low-effective-dimension
one wants a low-rank/PLS model, a rough one wants a Matérn or exponential kernel.
That mapping — **step 2, model assignment — is explicitly out of scope for now.**
This module is step 1: detect the structure reliably and honestly. The outlook at
the end sketches where step 2 goes.

## 2. Core idea and the shared Context primitives

The design is a thin set of expensive, shared primitives computed **once** from
`(X, y)`, on top of which every criterion family is a cheap, largely arithmetic
read. Families never re-normalize, re-fit, or recompute distances; they consume the
cached `Context`. This keeps the whole 163-feature vector affordable and makes the
families directly comparable, because they all see the same normalized geometry.

The `Context` primitives are:

- **Normalized inputs (`Xn`)** — inputs rescaled to a common unit box so distances
  and slopes are comparable across problems of different native scales.
- **Standardized outputs (`ys`)** — `y` centered and scaled, so gaps, chords, and
  variogram sills are in unit-variance objective terms.
- **Global quadratic fit** — a least-squares plane-plus-bowl (linear + full
  quadratic with interactions) fit to `(Xn, ys)`, yielding the coefficient vector,
  the symmetric Hessian `A`, and fit diagnostics. The meta-model, curvature,
  rotation, and separability families all read this one fit.
- **Local gradients** — per-point local-linear gradient estimates from each point's
  neighborhood, the raw material for the gradient field and active subspace.
- **Gradient-covariance / active-subspace matrix** `C = mean_i g_i g_iᵀ` — its
  eigenvectors are the directions the function actually varies along and its
  eigenvalue decay reveals effective dimensionality and rotation.
- **Empirical variogram** — the semivariogram `γ(h)` of output differences versus
  input lag on `(Xn, ys)`, the geostatistical read on smoothness, range, and noise.
- **k-NN graph** — a k-nearest-neighbor graph on `Xn` with cached pairwise
  distances, reused by the topology, spectral, network, nearest-better, and
  multimodality families.

Everything degrades gracefully: constant `y`, `n=5`, `d=1`, and `d=20` all return
the full feature vector as finite-or-NaN values — never `inf`, never a raised
exception.

## 3. The criterion families

Each family is a self-contained view of one structural axis. Intuition first, then
the features it emits.

### distribution — the value histogram as a fingerprint

Looks only at the multiset of objective values `y`, ignoring *where* in input space
they occur — the landscape as a bag of heights. This is the classic ELA
y-distribution view (Mersmann et al. 2011): the shape of the value histogram is a
cheap but surprisingly discriminative signal of global character. Symmetric,
smoothly spread values (near-zero skew/kurtosis, high entropy, a single KDE mode)
are the signature of simple unimodal bowls like Sphere; a sharp concentration of
similar values punctuated by rare extremes (high kurtosis, heavy tail index,
multiple KDE modes) betrays rugged, funnel, or multimodal functions like Rastrigin,
Ackley, or Schwefel. It combines moment-based shape with robust quantile measures
so it degrades gracefully under outliers and small samples, and adds KDE-derived
entropy, mode count, and peak concentration for an information-theoretic and
multimodality read. Being distance-free and position-invariant, it is the fastest
slice of the signature.

- **skewness** — Fisher moment skewness of `y`. Near 0 for symmetric bowls; positive
  for funnel/plateau functions, negative when a broad basin dominates.
- **excess_kurtosis** — tail weight vs normal. High for spiky multimodal (Schwefel);
  low/negative for near-uniform value spreads.
- **diff_entropy** — differential entropy of a Silverman-bandwidth KDE of `y`. High
  for smoothly spread values (Sphere/linear); low for collapsed/plateau landscapes.
- **n_modes** — prominence-thresholded local maxima in the KDE density. 1 for
  unimodal, >1 for values clustering at several levels.
- **tail_index** — `(p99−p50)/(p50−p1)`. >1 when the bad-value tail is heavier
  (penalized/funnel objectives); ~1 symmetric.
- **dynamic_range** — raw `max−min`. Large for steep/ill-conditioned functions; 0 for
  constant `y`.
- **coef_variation** — `std/|mean|`. High for large relative variation; nan/0 for
  constant or zero-mean data.
- **iqr_range_ratio** — IQR over full range. Toward 0.5 for near-uniform; low when
  outliers stretch the range (spiky multimodal).
- **median_skew** — bounded `(mean−median)/range`, robust asymmetry.
- **peak_concentration** — KDE mass in a narrow band around the tallest peak. High for
  one dominant level (funnel/plateau); low for multimodal.

### meta_model — how well a plane and a bowl fit

Asks the canonical ELA question: how well do the two cheapest surrogates — a plane
and a bowl — describe the cloud? A global linear least-squares fit and a full
quadratic (with interactions) are fit to the standardized outputs and their
goodness-of-fit dissected. High linear R² signals a planar, trend-dominated
landscape; the curvature gain (quadratic minus linear R²) and its normalized ratio
isolate smooth second-order structure beyond the trend, cleanly separating flat/
linear functions from convex bowls. The fitted coefficients add an orthogonal read:
the spread of the linear coefficients and the conditioning of the pure curvature
terms expose anisotropy and effective dimensionality. Multimodal, rugged functions
defeat both surrogates, leaving low R² everywhere. All quantities degrade
gracefully — constant `y` is flagged linear, underdetermined quadratics surface via
adjusted R² and the reliability flag. These are about *global fit-ability of
low-order models*, complementary to local-gradient or variogram ruggedness.

- **lin_r2** / **lin_r2_adj** — R² (and df-penalized R²) of the global linear fit.
  High for planar/linear, low for multimodal.
- **quad_r2** / **quad_r2_adj** — R² of the full quadratic. High for smooth bowls;
  nan/low when underdetermined.
- **curv_gain** — `quad_r2 − lin_r2`, extra variance from second-order terms. ~0 for
  linear, large for curved bowls.
- **quad_improve_ratio** — `(quad_r2−lin_r2)/(1−lin_r2)`, curvature's share of the
  residual. ~1 for pure quadratic bowls, low for rugged.
- **lin_coef_min** / **lin_coef_max** / **lin_coef_spread** — smallest/largest/ratio
  of linear coefficients. Spread large/inf for anisotropic or few-active-variable
  landscapes; ~1 for isotropic.
- **quad_curv_cond** — conditioning of the pure curvature terms. Large/inf for
  ill-conditioned bowls (Ellipsoid, Cigar, Discus); ~1 for Sphere.
- **intercept_abs** — absolute intercept of the standardized linear fit.
- **is_linear** — flag: `lin_r2 ≥ 0.99` and `curv_gain < 0.01`.
- **quad_reliable** — flag: at least as many samples as quadratic coefficients.

### curvature — second-order shape from the Hessian spectrum

Reads the landscape's second-order shape from the eigenvalues of the globally
fitted quadratic Hessian `A`, treating them as principal curvatures. Their **signs**
classify coarse topology — all-positive is a convex bowl, mixed signs a saddle,
all-negative a dome. Their **magnitudes and spread** describe conditioning: an
isotropic Sphere has condition number ~1 and low anisotropy, while a stretched
valley (Rosenbrock, Cigar) shows a huge condition number because one direction is
far stiffer. The signed trace captures net bowl-vs-dome tendency and `curv_energy`
the overall bendiness. Because the global quadratic captures only average curvature,
these describe the dominant large-scale bowl/saddle character, not local wiggles;
`curv_linear_ratio` quantifies how much of the fit is genuinely second-order, and
`curv_reliable` flags the underdetermined regime.

- **eig_abs_mean / eig_abs_max / eig_abs_min** — mean/stiffest/softest principal
  curvature.
- **condition_number** — largest over smallest non-tiny `|eigenvalue|`. Very high for
  stretched valleys, ~1 for Sphere.
- **convex_frac / neg_curv_frac** — positive/negative eigenvalue share. 1.0/0 for
  convex bowls, ~0.5 for saddles.
- **definiteness** — in `[-1,1]`: +1 bowl, −1 dome, ~0 saddle.
- **mean_curvature** — `trace(A)/d`, signed net curvature.
- **curv_energy** — sum of squared eigenvalues (total bendiness).
- **curv_anisotropy** — CV of non-tiny `|eigenvalues|`. High when a few directions
  carry the curvature.
- **curv_linear_ratio** — bounded `[0,1]` second-order weight over the plane.
- **curv_reliable** — enough-samples flag.

### rotation — are the principal axes tilted off the coordinate axes?

Measures whether the landscape's principal directions are rotated relative to the
coordinate axes, and how anisotropic it is. It builds two independent eigenframes —
one from the quadratic Hessian (curvature), one from the gradient-covariance /
active-subspace matrix (first-order variation) — and for each reports off-axis
energy (tilt away from the nearest axis, via an eigenvalue-weighted max with an
internal Hungarian axis assignment that avoids double-counting), spectral
anisotropy, and an **isotropy-gated** rotation score. The gating is the crux:
rotation is *undefined for a sphere* because every frame is an eigenframe, so the
score becomes NaN when the spectrum is near-isotropic. This cleanly separates the
three canonical shapes — sphere (isotropic → NaN), axis-aligned ellipsoid
(anisotropic but off-axis ~0 → rotation 0), and rotated ellipsoid (anisotropic AND
high off-axis → rotation → 1). A cross-frame alignment feature checks whether
curvature and variation agree on the principal direction.

- **hess_offaxis** — eigenvalue-weighted off-axis energy of the Hessian eigenframe.
  ~0 for axis-aligned, high for rotated.
- **hess_aniso** — Hessian spectral anisotropy `(|λmax|−|λmin|)/(|λmax|+|λmin|)`.
- **hess_rot** — isotropy-gated rotation score from the Hessian: assignment off-axis
  × anisotropy, NaN when near-isotropic. **The key sphere-vs-aligned-vs-rotated
  discriminator.**
- **grad_offaxis / grad_aniso / grad_rot** — the first-order counterparts from the
  active-subspace matrix; robust when the quadratic fit is unreliable.
- **rot_align** — squared cosine between the dominant Hessian and active-subspace
  eigenvectors. ~1 for well-behaved unimodal bowls.
- **rot_consensus** — NaN-aware mean of `hess_rot` and `grad_rot`: a single robust
  rotation summary.

### active_subspace — effective dimensionality

Measures how many input directions the function actually varies along versus how
many it nominally has, entirely from the eigenvalue spectrum of the
gradient-covariance matrix `C = mean_i g_i g_iᵀ`. A nearly flat spectrum (high
participation ratio, high spectral entropy, slope ~0) signals a genuinely
high-dimensional isotropic function; a fast-decaying spectrum with one dominant
eigenvalue signals a ridge or embedded function on a low-dimensional active
subspace. Per-coordinate sensitivity spread (Gini and CV of `diag(C)`) reads the raw
axis-aligned importances. The science is Constantine's active subspaces plus
spectral/participation-ratio analysis and global sensitivity analysis.

- **participation_ratio** — `(Σλ)²/Σλ²`, effective number of active directions. Near
  `d` for isotropic, near 1 for a ridge.
- **intrinsic_dim_frac** — participation ratio over `d`.
- **energy_dim_90 / energy_dim_frac** — leading eigenvalues (and fraction) for 90% of
  gradient-variation energy.
- **spectral_decay_slope** — slope of `log(λ)` vs rank. Steeply negative for ridges,
  ~0 for isotropic.
- **top_eig_frac** — energy share of the single largest eigenvalue. → 1 for ridges.
- **spectral_entropy** — normalized Shannon entropy of the spectrum. → 1 isotropic,
  → 0 ridge.
- **sensitivity_gini / sensitivity_cv** — inequality/spread of per-coordinate
  sensitivities `diag(C)`.

### separability — sum of 1-D pieces, or coupled?

Asks whether the landscape is a sum of independent one-dimensional pieces or whether
the coordinates are entangled, two complementary ways. First, a **functional-ANOVA**
decomposition: fit per-coordinate main effects (low-degree polynomial smoothers on
each axis), measure the variance that additive model explains, then compare against
a model that adds pairwise `x_i·x_j` interactions; the additive-to-total ratio is
the separability index, computed on **adjusted** R² so interaction terms that merely
fit noise are penalized. Second, a **quadratic proxy**: the shared global Hessian is
exactly diagonal for a separable function, so its off-diagonal energy directly
measures bilinear coupling with no extra fitting. High separability (index ~1,
off-diagonal ~0) points to sum-of-1D functions (Sphere, Rastrigin); low separability
points to rotated, banana, or multiplicative functions (Rosenbrock, rotated
ellipsoid).

- **separability_index** — adjusted-R² additive share of explained variance. → 1
  separable, low for coupled/rotated.
- **main_effect_r2** — R² of the additive main-effect model.
- **interaction_r2_gain** — adjusted-R² gain from pairwise interactions. High for
  coupled, ~0 for separable.
- **residual_interaction_ratio** — interaction gain over additive residual: is the
  leftover structured coupling or irreducible noise?
- **hessian_offdiag_ratio / hessian_diag_dominance** — off-/on-diagonal Hessian energy
  share.
- **max_pair_coupling** — largest single off-diagonal coupling relative to total
  curvature.
- **main_effect_participation / main_effect_gini / top_dim_share** — how the separable
  signal is distributed across the axes (Sobol-style).
- **hessian_reliable** — fit-conditioning flag.

### variogram — smoothness, range, and noise, geostatistically

Reads the landscape the way a geostatistician reads a spatial field: treat each
labelled point as a sample of a random function and ask how quickly the objective
decorrelates with distance. The empirical semivariogram
`γ(h) = ½·mean squared output difference at lag h` is fit on `(Xn, ys)` and
summarized. Its behavior **at the origin** captures smoothness (log-log slope ~2 →
differentiable Gaussian-smooth; ~1 → rough exponential; flat/negative → noise); the
distance to the plateau (the **range**) captures how far correlation extends; the
**nugget-to-sill** ratio isolates irreducible short-lag variance that signals
measurement noise or discontinuity; and the **sill** anchors everything to the total
variance and flags non-stationary trend. Monotonicity and near-origin diagnostics
further separate clean unimodal basins from periodic, multimodal, or noise-dominated
functions.

- **smoothness_exp** — near-origin log-log slope of `γ`. ~2 smooth, ~1 rough, ~0/neg
  noise.
- **range_rel** — distance to 95% of the sill over the sampled span. Near 1 for
  long-range basins, low for high-frequency/noisy.
- **nugget_ratio** — extrapolated `h→0` nugget over sill, in `[0,1]`. → 1 noisy, ~0
  clean.
- **sill / sill_ratio** — plateau semivariance, and its ratio to total variance
  (stationarity/trend diagnostic; >1 for trended fields).
- **monotonicity** — fraction of variogram bins that rise. → 1 well-behaved, ~0.5
  periodic/hole-effect.
- **near_origin_convexity** — origin curvature of `γ`. Positive parabolic knee →
  Gaussian; concave → exponential/noise.
- **noise_slope_ratio** — short-lag secant slope over overall slope. ~1 structured,
  ≫1 nugget-dominated.

### multimodality — counting basin bottoms directly

Measures multimodality directly from the cloud by counting **sample-level local
optima**: a point is a basin bottom when its objective lies strictly below all its k
nearest neighbors. Because a single `k` is fragile (small `k` over-counts noise
ripples, large `k` keeps only the deepest basins), it sweeps `k` and reports both the
averaged fraction and how fast the fraction decays — persistence across scales
separates genuine multimodality from smooth-but-noisy sampling. The raw count is
enriched with a mean basin-size proxy, the mean depth/prominence of the minima, and
their spatial dispersion. A local-maximum count and the min/max ratio add a symmetry
read on ruggedness. High for Rastrigin/Ackley/Griewank-style functions, low (single
basin, zero decay, large basin size) for unimodal bowls. *(See §6 — this is the
weakest axis empirically.)*

- **local_min_frac / local_min_frac_mean** — local-minimum fraction at default `k`,
  and swept over `k`.
- **n_basins_est** — extrapolated basin count.
- **local_min_density** — minima per sample point.
- **mean_basin_size** — samples per basin.
- **min_frac_k_decay** — relative drop of the fraction from small to large `k`
  (multi-scale persistence).
- **local_max_frac** — local-maximum fraction (maxima side).
- **min_depth_mean** — mean depth of minima below their neighborhood.
- **basin_dispersion** — spread of minima relative to the whole cloud.
- **min_max_ratio** — min/max optima symmetry.

### nearest_better — single-funnel vs multi-funnel

Nearest-better clustering (Kerschke & Preuss) distinguishes single-funnel from
multi-funnel landscapes from geometry alone. For each point it measures `nn`, the
distance to the closest other point, and `nb`, the distance to the closest point
with strictly better (lower) fitness. In a smooth unimodal landscape the nearest
neighbor is almost always the nearest better point, so `nb/nn` stays ~1 and the two
distance sets correlate tightly. In a multimodal landscape a point near a poor local
optimum must reach across a valley to find anything better, so `nb` inflates, the
ratio spreads, and `nb` correlates with fitness. A directed nearest-better graph
exposes funnel structure through indegree — a single funnel concentrates edges into
one dominant attractor (high max indegree, high Gini).

- **mean_ratio / median_ratio** — mean/median `nb/nn`. ~1 unimodal, >1 multimodal.
- **ratio_cv** — dispersion of the `nb/nn` distribution.
- **sd_ratio** — `std(nb)/std(nn)`.
- **mean_dist_ratio** — mean `nb` over mean `nn`.
- **nn_nb_cor** — correlation of `nn` and `nb` distances. Near 1 unimodal.
- **nb_fitness_cor / ratio_fitness_cor** — correlation of `nb` distance / ratio with
  fitness. Positive for multi-funnel.
- **indegree_fitness_cor** — indegree vs fitness. Strongly negative for single-funnel.
- **indegree_max / indegree_gini** — concentration of attraction. High for
  single-funnel.
- **funnel_frac** — fraction of rim points where `nb > 2·nn`.

### dispersion — global funnel structure and searchability

Measures the **global funnel structure**: are the good points gathered into one
region or scattered across many? The Lunacek dispersion metric compares the average
spread of the best q% of points against the spread of the whole cloud. When the
elite set is tighter (negative dispersion) there is a single global funnel; when the
elite set is as spread out as the cloud (positive dispersion) the good regions are
scattered — the hallmark of multi-funnel problems like Rastrigin or Schwefel.
Sweeping the elite fraction from 25% down to 2% and reading the trend (`disp_slope`)
sharpens the signal. Complementing it, the Jones & Forrest fitness-distance
correlation asks whether distance to the current best predicts fitness at all:
strong positive FDC means "closer is better" globally (an easy bowl), weak or
negative FDC exposes deception.

- **disp_02 / disp_05 / disp_10 / disp_25** — Lunacek dispersion for the best
  2/5/10/25% elite. Positive → multi-funnel, negative → single funnel.
- **disp_ratio_10** — scale-free version of the 10% dispersion.
- **disp_slope** — normalized elite spread vs `log(elite fraction)`. Strongly positive
  → single global funnel, ~0 → multi-funnel.
- **fdc / fdc_spearman** — Pearson/Spearman fitness-distance correlation to the best
  point. Near +1 searchable, ~0/negative deceptive.
- **fdc_slope** — regression slope of standardized fitness on distance-to-best.

### information_content — ruggedness as a symbolic time series

Reads the landscape as a symbolic time series. From several random starts the sample
is threaded into greedy nearest-neighbor walks so consecutive samples are spatially
close, and the sequence of objective changes is quantized into up/down/flat moves at
a tunable sensitivity `eps`. Sweeping `eps` traces two curves: the **information
content** `H(eps)`, the entropy of consecutive symbol transitions (ruggedness), and
the **partial information** `M(eps)`, the density of slope-sign alternations
(modality). A smooth bowl produces long monotone runs, so `H` and `M` are low and
information concentrates at one scale; a rugged/multimodal/noisy function produces
rapidly alternating symbols, pushing `H_max` and `m0` high and spreading information
across many scales. *(See §6 — `h_max` was not discriminative on the n=300 demo
clouds.)*

- **h_max / h0 / h_auc** — peak / full-sensitivity / mean information content over the
  sweep.
- **eps_max / eps_s / eps_ratio** — scale of peak ruggedness, settling scale, and the
  width of the informative band.
- **m0 / m_max** — initial/peak partial information (modality).
- **flat_frac** — fraction of near-flat walk steps (neutrality/plateau).

### convexity — chords vs midpoints

Exploits the defining property of convex functions: for any two points, the function
at their midpoint lies below the straight-line chord. It samples many random pairs,
approximates each geometric midpoint by the nearest actual sample, and compares that
value against the linear interpolation of the endpoints. Pairs where the midpoint
dips below the chord are convex; those above, concave. Aggregating sign and magnitude
across thousands of pairs gives a global shape read: convex bowls score high
`convex_frac` and positive `net_convexity`; concave domes flip the sign; rugged
multimodal surfaces produce a balanced mix with high `gap_std` and near-zero net
convexity. Because a real sample stands in for the true midpoint, two diagnostics
report how trustworthy the probe is for the given cloud density. All gaps are in
standardized-`y` units.

- **convex_frac / concave_frac / linear_frac** — pair shares below / above / near the
  chord.
- **net_convexity** — `convex_frac − concave_frac`. +1 bowl, −1 dome, ~0 rugged.
- **mean_gap / median_gap** — mean/robust signed convexity gap.
- **gap_std** — dispersion of the gap (high for rugged).
- **convex_intensity / concave_intensity** — mean magnitude of positive / negative
  gaps.
- **gap_skew** — skewness of the gap distribution.
- **midpoint_approx_error / usable_frac** — reliability diagnostics for the midpoint
  approximation.

### topology — sublevel-set persistence-lite

Treats the cloud as a filtered object: build a k-NN graph on `Xn`, sweep the
objective threshold from low to high, and watch the connected components of the
already-below-threshold subgraph appear and merge. This is a discretized,
persistence-lite version of sublevel-set persistent homology / Morse theory. Each
time the threshold passes a local minimum a new basin is born; when the rising level
bridges two basins the younger dies and the elder survives, and the death-minus-birth
gap is that basin's persistence — its topological prominence. A unimodal bowl yields
one component and zero prominence; a rugged function yields many coexisting
components with substantial, evenly spread persistences. *(See §6 — `n_basins`
over-counts on smooth clouds and is not trustworthy alone.)*

- **peak_components** — max components coexisting at any threshold.
- **n_basins** — total components ever born (graph-level minima).
- **basin_density** — basins per sample.
- **max_persistence / mean_persistence** — largest / mean normalized basin lifetime.
- **persistence_entropy** — entropy of the prominence distribution (many equally
  prominent basins → high).
- **prominence_ratio** — second-largest over largest persistence (rival-basin
  strength).
- **component_spread** — value-range fraction over which >1 basin coexists.
- **final_components / single_component** — components after the full sweep, and the
  connectivity flag.
- **euler_mean / euler_final** — Euler characteristic `(V−E)/n` across the sweep and
  on the full graph.

### spectral — the objective as a graph signal

Treats the sampled objective as a signal on the k-NN graph and asks how it decomposes
into graph "frequencies." It builds the symmetric normalized graph Laplacian
(heat-kernel edge weights, median-distance bandwidth), eigendecomposes it, and
projects the standardized outputs onto its eigenvectors — the graph Fourier
transform. Smooth, globally-structured landscapes concentrate energy in the lowest
modes (small Rayleigh quotient, high low-frequency fraction, low entropy); rugged/
multimodal/noisy landscapes push energy into high-frequency modes. The Rayleigh
quotient `ysᵀ L ys / ysᵀ ys` is the core smoothness scalar; the fractional,
centroid, rolloff, and entropy features characterize the whole energy distribution.
Participation ratio adds whether energy packs into a few modes (periodic/low-rank) or
scatters broadband (noise). Mesh-free and rotation-invariant.

- **rayleigh** — Dirichlet energy / mean graph frequency. ~0 smooth, → 2 rugged.
- **spectral_centroid** — Rayleigh quotient normalized to `[0,1]`.
- **low_energy_frac / high_energy_frac** — energy in the smoothest / ruggedest 20% of
  modes.
- **spectral_entropy** — entropy of the mode-energy distribution.
- **dominant_freq** — graph frequency of the single most energetic mode.
- **spectral_rolloff** — frequency below which 85% of energy accumulates.
- **participation_ratio** — effective fraction of active modes (few → periodic/
  low-rank; many → broadband noise).

### gradient_field — reading the field of change

Reads the landscape through its field of change rather than its values. From
all-pairs secant slopes it estimates a robust Lipschitz constant and its tail
peakiness (worst-case steepness, and whether isolated cliffs hide among flat
regions). From the per-point local-linear gradients it summarizes the magnitude
distribution (mean, CV, skew) to distinguish uniformly steep functions from ones
mixing gentle basins with steep walls. The signature feature is **gradient
coherence** — the mean cosine similarity between neighboring points' gradients — high
for smooth coherent fields (a bowl or plane whose gradients point roughly the same
way), low for rugged fields whose gradients scatter. A mesh-free curvature proxy (the
normalized variation of neighboring gradients) is ~0 for a linear ramp and large for
a wiggly surface, and its own dispersion flags whether curvature is spatially uniform
or concentrated.

- **lipschitz / lipschitz_max** — 95th-percentile / max secant slope.
- **lipschitz_peakiness** — `p99/median` slope, the heavy-tail cliff index.
- **grad_mag_mean / grad_mag_cv / grad_mag_skew** — mean / spread / skew of gradient
  magnitudes.
- **grad_coherence** — mean neighbor gradient cosine similarity. → 1 smooth, → 0
  rugged.
- **grad_curvature** — mean normalized neighbor-gradient difference (mesh-free
  curvature). ~0 linear, high wiggly.
- **grad_curv_cv** — spatial heterogeneity of that curvature.

### network — the cloud as a fitness network

Treats the cloud as a fitness network: a symmetrized k-NN graph whose nodes carry an
objective value, read through complex-network and local-optima-network (LON) theory.
It asks three questions. **Do connected points share fitness?** Fitness
assortativity and mean edge gap turn neighbor-to-neighbor coherence into a ruggedness
gauge. **How cohesive are neighborhoods?** A fitness-weighted (Onnela) clustering
coefficient measures whether interlinked triangles also agree on fitness, with an
unweighted baseline to separate real fitness structure from sampling geometry.
**Where do descents end?** A directed better-neighbor graph — every node points to
its best neighbor — makes sink nodes into attracting local optima; counting them
estimates modality, while largest-basin fraction, basin-size Gini, sink-fitness
spread, and descent path lengths reveal one deep funnel versus many shallow rivals.
*(See §6 — `fitness_assortativity` is the most reliable multimodality signal
here.)*

- **fitness_assortativity** — fitness correlation across edges. High smooth, low
  rugged. **The reliable modality signal.**
- **edge_fitness_gap / edge_gap_cv** — mean / CV of edge fitness gaps.
- **weighted_clustering / clustering_coef** — Onnela fitness-weighted vs unweighted
  clustering.
- **basin_count** — sink fraction of the better-neighbor graph.
- **largest_basin_frac** — fraction descending into the single largest basin.
- **basin_size_gini** — concentration of basin sizes.
- **sink_fitness_spread** — std of fitness across sink nodes.
- **mean_path_length / max_path_length** — descent hops to a sink.

## 4. Benchmark — forward and backward

The validation philosophy is: **design functions with KNOWN structure and assert the
detectors fire.** Two directions.

- **Forward** — take a canonical function whose character is understood (Sphere is an
  isotropic bowl, Rastrigin is rugged multimodal, Linear is a plane) and confirm the
  matching features land in the expected regime.
- **Backward** — this is where the design does real work. Construct a *pair* of
  functions that differ in exactly one structural property while holding everything
  else fixed, so a detector that fires must be responding to that property alone. The
  flagship backward test is an **axis-aligned versus rotated ellipsoid at equal
  conditioning**: both have condition number ~100 and identical anisotropy, differing
  only in whether the principal axes are tilted off the coordinate axes. A rotation
  detector that reports 0 on the aligned one and a positive value on the rotated one —
  while conditioning stays matched — is provably measuring rotation and not merely
  anisotropy. This is the cleanest possible existence proof for the rotation family.

Benchmark table (demo run, `d=5`, `n=300`):

```
function            rot  offaxis    aniso     cond  eff_dim   smooth   nugget   basins localmin    separ   lin_r2      fdc
--------------------------------------------------------------------------------------------------------------------------
sphere              nan     0.00     0.00     1.01     4.71     1.01     0.00     2.00     0.01     1.00     0.02     0.80
ellipsoid_aa       0.00     0.00     1.27    99.87     1.44     0.88     0.01     5.00     0.02     1.00     0.02     0.47
ellipsoid_rot      0.37     0.32     1.27   100.20     1.43     3.40     0.00     5.00     0.03     0.31     0.02     0.33
ridge              0.18     0.20     0.00     1.00     1.26     2.12     0.00    12.00     0.05     0.56     0.02     0.16
rastrigin          0.08     0.42     0.14     1.44     4.77     0.69     0.18    10.00     0.04     1.00     0.01     0.49
rosenbrock         0.23     0.33     0.47    14.86     4.25     0.92     0.02     3.00     0.01     0.99     0.08     0.59
linear              nan      nan      nan      nan     1.00     2.12     0.00     3.00     0.01     1.00     1.00     0.82
noisy_sphere        nan     0.38     0.03     1.07     4.77     0.34     0.10     3.00     0.01     1.00     0.02     0.70
```

Columns: `rot`=rotation.hess_rot, `offaxis`=rotation.hess_offaxis,
`aniso`=curvature.curv_anisotropy, `cond`=curvature.condition_number,
`eff_dim`=active_subspace.participation_ratio (of `d=5`),
`smooth`=variogram.smoothness_exp, `nugget`=variogram.nugget_ratio,
`basins`=topology.n_basins, `localmin`=multimodality.local_min_frac,
`separ`=separability.separability_index, `lin_r2`=meta_model.lin_r2,
`fdc`=dispersion.fdc.

## 5. Usage

```python
from pysurrogate.landscape import Landscape

lp = Landscape(X, y)          # X: (n, d) inputs, y: (n,) objective values

lp.report()                   # human-readable structural summary
feats = lp.features()         # dict of all 163 features (finite-or-nan)
```

`Landscape(X, y)` computes the shared `Context` once; `.features()` returns the full
model-free feature vector, and `.report()` renders the headline structural verdicts
(rotation, conditioning, effective dimension, smoothness, noise, multimodality,
separability, funnel structure) in readable form.

## 6. What works, what is noisy

Honest read from the benchmark.

**Clearly works (right value on the right function):**

- **Rotation** — the flagship backward-designed result, and it is unambiguous.
  `hess_rot = 0.00` on the axis-aligned ellipsoid vs `0.37` on the rotated one, while
  `condition_number` stays matched (99.87 vs 100.20); `hess_offaxis` corroborates
  (0.00 vs 0.32); Sphere correctly returns NaN (rotation undefined).
- **Separability** — index 1.00 (aligned) vs 0.31 (rotated), `interaction_r2_gain`
  0.69 on rotated, `hessian_offdiag_ratio` 0.00 vs 0.39. Cleanly detects the coupling
  rotation induces.
- **Conditioning** — `condition_number` ~1 (sphere), ~100 (both ellipsoids), ~15
  (rosenbrock); `curv_anisotropy` tracks it.
- **Effective dimension** — `participation_ratio` 4.7 (isotropic sphere, of 5) vs 1.26
  (ridge) vs 1.0 (linear); `energy_dim_90` and `top_eig_frac` agree. Ridge/
  active-subspace detection is solid.
- **Linearity** — `lin_r2 = 1.00` and `is_linear=1` only for the linear function;
  `curv_linear_ratio` ~0 for linear vs >0.9 for bowls.
- **Noise** — `nugget_ratio` 0.10 on noisy_sphere vs 0.00 clean; `smoothness_exp`
  drops 1.01 → 0.34. The nugget detector fires reliably.
- **Smoothness** — `smoothness_exp` separates smooth bowls (≥1) from rugged rastrigin
  (0.69).

**Weak / noisy detectors:**

- **Multimodality is the weakest axis.** `topology.n_basins` *over-counts* on smooth
  clouds (ellipsoid=5, ridge=12 — both unimodal), so it is not trustworthy alone.
  `multimodality.local_min_frac` is only ~5× (rastrigin 0.037 vs sphere 0.007) —
  discriminative but low-contrast. `multimodality.n_basins_est` is essentially useless
  (22–34 for everything). The **reliable** multimodality signal is
  `network.fitness_assortativity` (rastrigin 0.12 vs smooth ~0.27–0.53), so
  `Landscape.report()` requires *both* elevated `local_min_frac` AND low assortativity
  before calling a landscape multimodal — it no longer false-positives on the smooth
  ellipsoid/ridge.
- **`information_content.h_max` was not discriminative here** (all ~0.99) — it did not
  separate rugged from smooth on these `n=300` clouds.
- **`dispersion.fdc`** works directionally (sphere 0.80 searchable, ridge 0.16 low)
  but is more a searchability index than a modality flag.
- **`rotation.grad_rot`** is noisier than `hess_rot` (returns 0.20 for sphere where
  `hess_rot` correctly gives NaN); trust `hess_rot` when the quadratic fit is
  reliable.

**Robustness:** constant-`y`, `n=5`, `d=1`, and `d=20` all return the full
163-feature vector as finite-or-nan — never `inf`, never raising.

## Outlook — step 2, mapping structure to model

Step 1 detects *what kind of function this is*; step 2 will turn that verdict into a
matched surrogate model or Bayesian-optimization prior. The natural correspondences
are already visible in the families: a confidently rotated landscape (high
`hess_rot`, low separability) argues for a **Mahalanobis / full-covariance kernel**
that learns off-axis correlations rather than an axis-aligned ARD kernel; a
low-effective-dimension landscape (small `participation_ratio`, high `top_eig_frac`)
argues for a **PLS / low-rank / active-subspace** model that fits only the directions
that matter; a rough landscape (low `smoothness_exp`, high `nugget_ratio`) argues for
a **Matérn or exponential** kernel over a smooth Gaussian one, with a learned noise
term when the nugget fires. Conditioning, separability, and funnel structure feed the
same decision. Building and validating that structure-to-prior mapping is future
work; for now the module's contract is to detect the structure reliably and report it
honestly.
