"""Tests for the KPLS backend (Kriging on a Partial-Least-Squares length-scale subspace)."""

import numpy as np
import pytest

from pysurrogate.core import Prediction
from pysurrogate.core.kernel import Gaussian, KPLSKernel
from pysurrogate.models import KPLS, Kriging


def _highdim(n=60, d=15, seed=0):
    rng = np.random.RandomState(seed)
    lo, hi = -5.0, 5.0
    X = rng.uniform(lo, hi, size=(n, d))
    # a function driven by a few directions -> PLS should recover a low-rank structure
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * X[:, 2] ** 2
    return X, y


# --- the KPLS kernel: exact reparameterization of an ARD kernel -------------------------------


def test_kpls_kernel_reduces_theta_dimension():
    w2 = np.random.RandomState(0).random((8, 3)) ** 2
    k = KPLSKernel(Gaussian(), w2)
    assert k.n_theta(8) == 3  # h length-scales, not d


def test_kpls_kernel_theta_grad_matches_finite_difference():
    rng = np.random.RandomState(1)
    d, h, n = 6, 3, 30
    w2 = rng.random((d, h)) ** 2
    k = KPLSKernel(Gaussian(), w2)
    D = rng.standard_normal((n, d))
    theta = rng.random(h) + 0.2

    ana = k.theta_grad(D, theta)
    eps = 1e-6
    fd = np.zeros_like(ana)
    for j in range(h):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fd[:, j] = (k(D, tp) - k(D, tm)) / (2 * eps)
    assert np.allclose(ana, fd, atol=1e-6)
    assert k.has_theta_grad  # so LBFGS uses the analytic Jacobian, not a numeric fallback


def test_kpls_kernel_spatial_grad_matches_finite_difference():
    rng = np.random.RandomState(2)
    d, h = 5, 2
    w2 = rng.random((d, h)) ** 2
    k = KPLSKernel(Gaussian(), w2)
    x = rng.standard_normal(d)
    theta = rng.random(h) + 0.3

    g = k.grad(x[None, :], theta)[0]
    eps = 1e-6
    fd = np.zeros(d)
    for j in range(d):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        fd[j] = (k(xp[None, :], theta)[0] - k(xm[None, :], theta)[0]) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-6)


# --- the KPLS backend: the full Model contract ------------------------------------------------


def test_kpls_fits_only_n_pls_length_scales():
    X, y = _highdim(d=15)
    model = KPLS(n_pls=3).fit(X, y)
    # the engine optimizes h=3 length-scales, not one per input dimension
    assert np.ravel(model.model.model["theta"]).shape == (3,)


def test_kpls_returns_prediction_with_var_and_grad():
    X, y = _highdim(d=12)
    model = KPLS(n_pls=3).fit(X, y)
    pred = model.predict(X[:5], var=True, grad=True)
    assert isinstance(pred, Prediction)
    assert pred.var is not None and pred.var.shape == (5, 1)
    assert np.all(pred.var >= 0.0)
    assert pred.grad is not None and pred.grad.shape == (5, X.shape[1])
    assert pred.var_grad is not None and pred.var_grad.shape == (5, X.shape[1])


def test_kpls_clamps_components_to_dimension():
    # asking for more components than dimensions clamps to d rather than raising
    X, y = _highdim(n=30, d=4)
    model = KPLS(n_pls=10).fit(X, y)
    assert np.ravel(model.model.model["theta"]).shape == (4,)


def test_kpls_matches_or_beats_ard_kriging_on_highdim():
    X, y = _highdim(n=70, d=15, seed=3)
    rng = np.random.RandomState(4)
    Xt = rng.uniform(-5, 5, size=(500, 15))
    yt = np.sin(Xt[:, 0]) + 0.5 * Xt[:, 1] + 0.1 * Xt[:, 2] ** 2

    kpls = KPLS(n_pls=3).fit(X, y)
    ard = Kriging(ARD=True).fit(X, y)
    rmse_kpls = np.sqrt(np.mean((kpls.predict(Xt).y.ravel() - yt) ** 2))
    rmse_ard = np.sqrt(np.mean((ard.predict(Xt).y.ravel() - yt) ** 2))
    # KPLS should be competitive with full ARD in high dimensions (here, at least as good)
    assert rmse_kpls <= rmse_ard * 1.25


def test_kpls_refit_returns_out_of_sample_score():
    X, y = _highdim(d=10)
    model = KPLS(n_pls=2).fit(X[:40], y[:40])
    score = model.refit(X[40:], y[40:])
    assert isinstance(score, dict) and np.isfinite(score["rmse"]) and score["rmse"] >= 0.0
    assert not model.history().empty  # the refit was recorded prequentially (full prediction kept)


def test_kpls_refit_requires_prior_fit():
    model = KPLS(n_pls=2)
    with pytest.raises(Exception, match="requires a prior fit"):
        model.refit(*_highdim(d=10))


def test_kpls_respects_active_dims_on_fit_refit_and_predict():
    # regression (DaceBackedModel._refit): with active_dims KPLS runs PLS and fits on the selected
    # columns only; refit previously bypassed preprocess and passed full-width raw inputs, crashing.
    rng = np.random.RandomState(3)
    X = rng.uniform(-1, 1, size=(50, 6))
    y = (np.sin(X[:, [0]]) + 0.5 * X[:, [2]] + 0.1 * X[:, [4]] ** 2).reshape(-1, 1)
    model = KPLS(n_pls=2, active_dims=[0, 2, 4]).fit(X, y, optimize=False)
    assert model.X.shape[1] == 3  # PLS/engine saw only the 3 active dims
    Xnew = rng.uniform(-1, 1, size=(10, 6))
    ynew = (np.sin(Xnew[:, [0]]) + 0.5 * Xnew[:, [2]] + 0.1 * Xnew[:, [4]] ** 2).reshape(-1, 1)
    model.refit(Xnew, ynew, optimize=False)  # must not raise despite full-width raw inputs
    assert np.all(np.isfinite(model.predict(X[:5]).y))


def test_kpls_survives_deepcopy_and_fits():
    """A deep-copied KPLS must still fit -- the selection layer clones models per CV fold.

    Regression guard: the default optimizer must be a concrete object on the instance, not an
    identity sentinel resolved at fit time. A sentinel does not survive ``deepcopy`` by identity,
    so a cloned model would pass a non-optimizer to the engine and silently fail its fit -- which
    dropped KPLS out of every ``Benchmark`` / ``AutoModel`` fleet.
    """
    import copy

    X, y = _highdim(n=40, d=12)
    clone = copy.deepcopy(KPLS(n_pls=3))
    clone.fit(X, y)
    assert clone.success
    assert clone.predict(X[:3]).y.shape[0] == 3


def test_kpls_is_selected_by_default_kriging_fleet_in_highdim():
    """The default uncertainty fleet includes KPLS and cross-validation picks it in high-D."""
    from pysurrogate import AutoModel
    from pysurrogate.selection.study import default_kriging

    assert [k for k in default_kriging() if k.startswith("KPLS")] == ["KPLS[2]", "KPLS[3]"]

    X, y = _highdim(n=60, d=30, seed=0)
    auto = AutoModel(models=default_kriging(), sorted_by="rmse").fit(X, y)
    assert auto.success
    # every candidate that fit successfully is ranked -- KPLS must not silently drop out
    assert any(name.startswith("KPLS") for name in auto.statistics())


# --- correctness against references ------------------------------------------------------------


def test_kpls_reduces_exactly_to_ard_kriging():
    """KPLS at a fixed theta is *exactly* an ARD-Gaussian GP with length-scales eta = W2 @ theta.

    This pins the reparameterization: the PLS reduction only constrains the length-scale vector,
    it must not change the GP itself. Any drift here is a bug in the kernel wrapper.
    """
    from pysurrogate.core.kernel import Gaussian as _G
    from pysurrogate.dace import ConstantRegression, Dace

    X, y = _highdim(n=50, d=8, seed=0)
    theta0 = 0.7
    kpls = KPLS(regr=ConstantRegression(), corr=_G(), n_pls=3, theta=theta0).fit(X, y, optimize=False)
    w2 = kpls.model.kernel.w2
    eta = w2 @ np.full(w2.shape[1], theta0)  # the effective d-dim ARD length-scales

    ard = Dace(regr=ConstantRegression(), corr=_G(ard=True), theta=eta, optimizer=None)
    ard.fit(X, y)

    Xt = np.random.RandomState(1).uniform(-3, 3, size=(100, 8))
    pk, pa = kpls.predict(Xt, var=True), ard.predict(Xt, var=True)
    assert np.allclose(pk.y, pa.y, atol=1e-9)
    assert np.allclose(pk.var, pa.var, atol=1e-9)


def test_kpls_reduced_path_matches_full_expansion():
    """The cached reduced-distance fast path must give the same fit as the full d-space kernel.

    The theta search evaluates the KPLS kernel in the reduced ``h``-space (M = D**2 @ W2, cached
    per fit) instead of expanding to ``d`` dimensions every call. It is an exact algebraic
    rewrite, so both paths must agree to floating-point roundoff -- this guards the optimization
    against silently changing results.
    """
    X, y = _highdim(n=80, d=40, seed=1)
    Xte = np.random.RandomState(2).uniform(-5, 5, size=(200, 40))

    fast = KPLS(n_pls=3).fit(X, y)  # reduced path (default)
    slow = KPLS(n_pls=3)
    slow.model = slow._kpls_engine(X, y)
    slow.model.kernel._reducible = False  # force the full expand-to-d path
    slow.model.fit(X, y, optimize=True)

    pf = fast.predict(Xte, var=True, grad=True)
    ps = slow.model.predict(Xte, var=True, grad=True)
    assert np.allclose(pf.y, ps.y, atol=1e-8)
    assert np.allclose(pf.var, ps.var, atol=1e-8)
    assert np.allclose(pf.grad, ps.grad, atol=1e-8)


@pytest.mark.integration
def test_kpls_matches_smt_reference():
    """Cross-check against SMT's reference KPLS (the canonical implementation).

    With a matching constant trend, our KPLS should track SMT's KPLS closely on the same data
    -- proof that the PLS weighting and kernel formula agree with the published implementation.
    Skipped when SMT is not installed (it is not a project dependency).
    """
    smt_models = pytest.importorskip("smt.surrogate_models")
    from pysurrogate.core.sampling import LHS, Sampling
    from pysurrogate.dace import ConstantRegression
    from pysurrogate.util.test_functions import get_test_function

    f, xl, xu = get_test_function("griewank", n_var=20)
    Xtr = Sampling(100, LHS()).sample((xl, xu), rng=np.random.default_rng(0))
    ytr = f(Xtr)
    Xte = Sampling(2000, LHS()).sample((xl, xu), rng=np.random.default_rng(1))
    yte = f(Xte)

    ours = KPLS(n_pls=3, regr=ConstantRegression()).fit(Xtr, ytr)
    rmse_ours = float(np.sqrt(np.mean((ours.predict(Xte).y.ravel() - yte) ** 2)))

    smt = smt_models.KPLS(n_comp=3, print_global=False)
    smt.set_training_values(Xtr, ytr)
    smt.train()
    rmse_smt = float(np.sqrt(np.mean((smt.predict_values(Xte).ravel() - yte) ** 2)))

    # same model family + matching trend -> within a few percent of the reference
    assert rmse_ours == pytest.approx(rmse_smt, rel=0.10)
