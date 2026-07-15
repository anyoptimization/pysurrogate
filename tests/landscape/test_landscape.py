"""Forward+backward benchmark: analytic functions with known structure vs the fired detectors."""

import numpy as np
import pytest

from pysurrogate.landscape import FAMILIES, Landscape
from pysurrogate.landscape import distribution as distribution_mod
from pysurrogate.landscape import variogram as variogram_mod
from pysurrogate.landscape._context import Context
from pysurrogate.landscape.information_content import _information_content

# --------------------------------------------------------------------------------------------
# Sampling designs and analytic benchmark functions with KNOWN structure.
# --------------------------------------------------------------------------------------------

N = 300
D = 5
COND = 100.0  # shared condition number for the axis-aligned vs rotated ellipsoid pair


def lhs(n, d, seed, lo=-5.0, hi=5.0):
    """A Latin-hypercube design on ``[lo, hi]^d`` (each 1D margin stratified once)."""
    rng = np.random.default_rng(seed)
    cuts = np.linspace(0.0, 1.0, n + 1)
    pts = np.zeros((n, d))
    for j in range(d):
        u = rng.uniform(size=n)
        pts[:, j] = (cuts[:-1] + u * (cuts[1:] - cuts[:-1]))[rng.permutation(n)]
    return lo + pts * (hi - lo)


def rotation_matrix(d, seed):
    """A random orthogonal ``(d, d)`` rotation via QR of a Gaussian matrix."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q


def ellipsoid_weights(d, cond):
    """Geometric per-axis weights spanning condition number ``cond`` (min 1, max ``cond``)."""
    return np.logspace(0.0, np.log10(cond), d)


# One fixed axis-aligned cloud shared by ALL functions, so structure -- not the design -- drives
# the contrast. The rotated ellipsoid keeps this same cloud and rotates only inside y (its Hessian
# becomes Qᵀ diag(w) Q); rotating the cloud too would cancel the rotation in its own coordinates.
_X = lhs(N, D, seed=1)
_W = ellipsoid_weights(D, COND)
_Q = rotation_matrix(D, seed=7)


def f_sphere(X):
    """Isotropic unimodal separable smooth bowl, NOT rotated."""
    return np.sum(X**2, axis=1)


def f_ellipsoid_axis_aligned(X):
    """Ill-conditioned bowl stretched along the coordinate axes (separable, not rotated)."""
    return np.sum(_W * X**2, axis=1)


def f_ellipsoid_rotated(X):
    """Same condition number as the axis-aligned ellipsoid but rotated off the axes (coupled)."""
    XR = X @ _Q.T
    return np.sum(_W * XR**2, axis=1)


def f_ridge(X):
    """Low effective dimension: value depends on one linear combination of inputs only."""
    w = np.array([1.0, 0.5, 0.0, 0.0, 0.0])
    return (X @ w) ** 2


def f_rastrigin(X):
    """Highly multimodal, separable, rugged."""
    d = X.shape[1]
    return 10.0 * d + np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)


def f_rosenbrock(X):
    """Non-separable curved banana valley."""
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


def f_linear(X):
    """Planar trend: linear R2 ~ 1, ~zero curvature."""
    return X @ np.arange(1.0, X.shape[1] + 1.0)


def f_noisy_sphere(X):
    """Sphere plus additive Gaussian noise: variogram nugget/noise detector should fire."""
    rng = np.random.default_rng(3)
    base = np.sum(X**2, axis=1)
    return base + rng.normal(0.0, 0.3 * float(np.std(base)), size=X.shape[0])


BENCH = {
    "sphere": f_sphere,
    "ellipsoid_aa": f_ellipsoid_axis_aligned,
    "ellipsoid_rot": f_ellipsoid_rotated,
    "ridge": f_ridge,
    "rastrigin": f_rastrigin,
    "rosenbrock": f_rosenbrock,
    "linear": f_linear,
    "noisy_sphere": f_noisy_sphere,
}


def build(name):
    """Construct the :class:`Landscape` for one benchmark function on the shared axis-aligned cloud."""
    return Landscape(_X, BENCH[name](_X), seed=0)


@pytest.fixture(scope="module")
def L():
    """Cache one Landscape per benchmark function for the whole module."""
    return {name: build(name) for name in BENCH}


# --------------------------------------------------------------------------------------------
# Forward: does the right detector fire on the function it was designed for?
# --------------------------------------------------------------------------------------------


def test_all_families_wired(L):
    """Every family contributes features and the flat dict is namespaced ``family.feature``."""
    feats = L["sphere"].features()
    assert len(feats) > 120
    for fam in FAMILIES:
        assert any(k.startswith(f"{fam}.") for k in feats), f"missing family {fam}"


def test_linear_is_linear(L):
    """The linear function reads as planar: high linear R2 and the is_linear flag set."""
    lin = L["linear"]
    assert lin.get("meta_model.lin_r2") > 0.95
    assert lin.get("meta_model.is_linear") >= 0.5
    # non-linear functions must not be flagged linear
    assert L["sphere"].get("meta_model.is_linear") < 0.5
    assert L["rastrigin"].get("meta_model.is_linear") < 0.5


def test_linear_has_near_zero_curvature(L):
    """A plane has essentially no curvature relative to its linear signal."""
    assert L["linear"].get("curvature.curv_linear_ratio") < 0.1
    # a bowl, by contrast, is dominated by curvature
    assert L["sphere"].get("curvature.curv_linear_ratio") > 0.5


def test_sphere_isotropic_unimodal(L):
    """Sphere: isotropic (condition ~1), fully active, and single-funnel (high FDC)."""
    sph = L["sphere"]
    assert sph.get("curvature.condition_number") < 3.0
    assert sph.get("curvature.curv_anisotropy") < 0.2
    assert sph.get("active_subspace.participation_ratio") > 0.7 * D
    assert sph.get("dispersion.fdc") > 0.6


def test_ridge_low_effective_dimension(L):
    """The ridge collapses onto one direction: low participation ratio, dominant top eigenvalue."""
    ridge = L["ridge"]
    sph = L["sphere"]
    assert ridge.get("active_subspace.participation_ratio") < 2.0
    assert ridge.get("active_subspace.participation_ratio") < sph.get("active_subspace.participation_ratio")
    assert ridge.get("active_subspace.top_eig_frac") > 0.7
    assert ridge.get("active_subspace.energy_dim_90") <= 2


def test_rastrigin_multimodal_vs_sphere(L):
    """Rastrigin is rugged/multimodal: more basins, rougher, lower fitness assortativity."""
    ras = L["rastrigin"]
    sph = L["sphere"]
    assert ras.get("topology.n_basins") > sph.get("topology.n_basins")
    assert ras.get("multimodality.local_min_frac") > sph.get("multimodality.local_min_frac")
    assert ras.get("network.fitness_assortativity") < sph.get("network.fitness_assortativity")
    assert ras.get("variogram.nugget_ratio") > sph.get("variogram.nugget_ratio")


def test_rosenbrock_curved_and_conditioned(L):
    """Rosenbrock is a curved, ill-conditioned valley -- more anisotropic/curved than a sphere."""
    ros = L["rosenbrock"]
    sph = L["sphere"]
    assert ros.get("curvature.condition_number") > sph.get("curvature.condition_number")
    assert ros.get("curvature.curv_anisotropy") > sph.get("curvature.curv_anisotropy")


def test_noisy_sphere_nugget_fires(L):
    """Additive noise lifts the variogram nugget and lowers the smoothness exponent vs clean."""
    noisy = L["noisy_sphere"]
    clean = L["sphere"]
    assert noisy.get("variogram.nugget_ratio") > clean.get("variogram.nugget_ratio")
    assert noisy.get("variogram.nugget_ratio") > 0.05
    assert noisy.get("variogram.smoothness_exp") < clean.get("variogram.smoothness_exp")


# --------------------------------------------------------------------------------------------
# Backward: the key rotation pair -- SAME conditioning, only rotation differs.
# --------------------------------------------------------------------------------------------


def test_condition_number_matches_across_rotation_pair(L):
    """The rotation must be isolated: axis-aligned and rotated ellipsoids share conditioning."""
    aa = L["ellipsoid_aa"].get("curvature.condition_number")
    rot = L["ellipsoid_rot"].get("curvature.condition_number")
    assert aa > 20 and rot > 20  # both genuinely ill-conditioned
    assert abs(aa - rot) / max(aa, rot) < 0.15  # ...at essentially the SAME condition number


def test_rotation_detector_distinguishes_the_pair(L):
    """THE backward-designed test: rotation score is high on rotated, ~0 on axis-aligned."""
    aa = L["ellipsoid_aa"]
    rot = L["ellipsoid_rot"]
    # Hessian-eigenframe rotation: axis-aligned ~0, rotated clearly positive.
    aa_hr = aa.get("rotation.hess_rot")
    rot_hr = rot.get("rotation.hess_rot")
    assert aa_hr < 0.1 or np.isnan(aa_hr)
    assert rot_hr > 0.2
    assert rot_hr > (0.0 if np.isnan(aa_hr) else aa_hr) + 0.15
    # Off-axis energy corroborates: rotated tilts curvature axes off the coordinates.
    assert rot.get("rotation.hess_offaxis") > aa.get("rotation.hess_offaxis") + 0.15


def test_rotated_ellipsoid_is_non_separable_vs_aligned(L):
    """Rotation induces coupling: the rotated ellipsoid reads non-separable, the aligned one not."""
    aa = L["ellipsoid_aa"]
    rot = L["ellipsoid_rot"]
    assert aa.get("separability.separability_index") > 0.85
    assert rot.get("separability.separability_index") < 0.7
    assert rot.get("separability.hessian_offdiag_ratio") > aa.get("separability.hessian_offdiag_ratio") + 0.1
    assert rot.get("separability.interaction_r2_gain") > 0.2


def test_sphere_rotation_undefined(L):
    """For an isotropic sphere the Hessian rotation is undefined (nan), not a false positive."""
    assert np.isnan(L["sphere"].get("rotation.hess_rot"))


# --------------------------------------------------------------------------------------------
# Robustness: degenerate clouds must yield finite-or-nan features and never raise.
# --------------------------------------------------------------------------------------------


def _all_finite_or_nan(feats):
    """True when every value is a plain float (finite or nan), never inf and never a raise."""
    return all(isinstance(v, float) and not np.isinf(v) for v in feats.values())


def _constant_y():
    """A cloud with a constant objective (zero variance)."""
    return np.random.default_rng(0).uniform(-1, 1, (60, 3)), np.ones(60)


def _tiny_n():
    """A cloud with only five points (fewer than the quadratic needs)."""
    rng = np.random.default_rng(1)
    return rng.uniform(-1, 1, (5, 3)), rng.normal(size=5)


@pytest.mark.parametrize("make", [_constant_y, _tiny_n], ids=["constant_y", "n=5"])
def test_robustness_smoke(make):
    """Constant-y and tiny-n clouds still produce a full finite-or-nan feature vector."""
    X, y = make()
    feats = Landscape(X, y).features()
    assert len(feats) > 120
    assert _all_finite_or_nan(feats)


def test_robustness_d1():
    """A 1D cloud runs end to end and yields finite-or-nan features."""
    x = np.linspace(-3, 3, 60).reshape(-1, 1)
    feats = Landscape(x, (x[:, 0] ** 2)).features()
    assert _all_finite_or_nan(feats)


def test_robustness_d20():
    """A high-dimensional (d=20) sphere runs without raising."""
    X = np.random.default_rng(5).uniform(-1, 1, (80, 20))
    lp = Landscape(X, np.sum(X**2, axis=1))
    assert _all_finite_or_nan(lp.features())
    # report must render for a high-d cloud too
    assert isinstance(lp.report(), str)


def test_report_renders(L):
    """The human report mentions the headline structural axes for a representative function."""
    text = L["ellipsoid_rot"].report()
    for token in ("Rotation", "Curvature", "Eff. dim", "Modality", "Separable", "Noise"):
        assert token in text


# --------------------------------------------------------------------------------------------
# Regression tests for reviewed fixes.
# --------------------------------------------------------------------------------------------


def test_information_content_entropy_counts_all_pairs():
    """Entropy denominator is ALL consecutive pairs (Munoz et al. 2015), not differing ones only.

    A long flat run must dilute the entropy: two differing pairs among twenty carry far less
    information than two among two.
    """
    phi = np.array([1] * 10 + [-1] + [1] * 10, dtype=np.int8)
    # 20 consecutive pairs total; exactly two differ: (1,-1) and (-1,1), once each.
    p = np.array([1.0, 1.0]) / (phi.size - 1)
    expected = float(-np.sum(p * np.log(p)) / np.log(6.0))
    assert _information_content(phi) == pytest.approx(expected)
    # the old counts/m normalization reported ln(2)/ln(6) ~ 0.387 regardless of the flat run.
    assert _information_content(phi) < 0.25


def test_rng_for_is_order_independent():
    """Per-family generators are deterministic and immune to other families' draws."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (30, 2))
    y = np.sum(X**2, axis=1)
    c1 = Context(X, y, seed=0)
    c1.rng_for("convexity").integers(0, 1000, size=50)  # drain another family's stream first
    a = c1.rng_for("information_content").integers(0, 1000, size=50)
    c2 = Context(X, y, seed=0)
    b = c2.rng_for("information_content").integers(0, 1000, size=50)
    assert np.array_equal(a, b)
    # distinct families get distinct streams
    assert not np.array_equal(a, Context(X, y, seed=0).rng_for("convexity").integers(0, 1000, size=50))


def test_family_failure_warns_and_strict_raises(monkeypatch):
    """A failing family warns and is skipped by default, but re-raises under ``strict=True``."""
    import pysurrogate.landscape.convexity as convexity_mod

    def boom(ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(convexity_mod, "compute", boom)
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (40, 2))
    y = np.sum(X**2, axis=1)
    with pytest.warns(UserWarning, match="convexity failed"):
        lp = Landscape(X, y)
    assert lp.groups()["convexity"] == {}
    assert np.isnan(lp.get("convexity.convex_frac"))
    with pytest.raises(RuntimeError, match="boom"):
        Landscape(X, y, strict=True)


def test_curvature_flat_frac_detects_ridge(L):
    """flat_frac exposes rank-deficient (ridge) Hessians the condition number deliberately skips."""
    assert L["ridge"].get("curvature.flat_frac") > 0.5
    assert L["sphere"].get("curvature.flat_frac") < 0.2


def test_variogram_near_origin_convexity_requires_uniform_bins():
    """The second-difference probe only fires on a near-uniform bin grid, else reads nan."""

    class _Stub:
        """A minimal context stub exposing exactly what ``variogram.compute`` reads."""

        ys = np.linspace(0.0, 1.0, 10)

        def __init__(self, h):
            self._h = np.asarray(h, dtype=float)

        def variogram(self):
            """Return the stubbed ``(h, gamma)`` curve."""
            return self._h, np.array([0.1, 0.2, 0.5, 0.7])

    uniform = variogram_mod.compute(_Stub([0.1, 0.2, 0.3, 0.4]))
    assert np.isfinite(uniform["near_origin_convexity"])
    # a dropped bin leaves a gap between the second and third centers -> probe undefined
    gapped = variogram_mod.compute(_Stub([0.1, 0.2, 0.5, 0.6]))
    assert np.isnan(gapped["near_origin_convexity"])


def test_peak_concentration_uses_kde_bandwidth():
    """Peak mass is measured within the KDE bandwidth, not a fraction of the (outlier-prone) span."""
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0.0, 1.0, 300), [100.0]])
    X = rng.uniform(-1, 1, (y.size, 2))
    out = distribution_mod.compute(Context(X, y))
    # the old 5%-of-span half-width spanned ~5 units here and reported ~all mass at the peak.
    assert out["peak_concentration"] < 0.6


def test_context_rejects_empty_cloud():
    """An empty point cloud fails fast with an explicit error, not a cryptic argmin failure."""
    with pytest.raises(ValueError, match="empty point cloud"):
        Context(np.empty((0, 3)), np.empty(0))


def test_network_basin_count_renamed_to_basin_frac(L):
    """The sink fraction is exposed as ``basin_frac``; the misleading old name is gone."""
    sph = L["sphere"]
    assert not np.isnan(sph.get("network.basin_frac"))
    assert 0.0 < sph.get("network.basin_frac") <= 1.0
    assert "basin_count" not in sph.groups()["network"]
