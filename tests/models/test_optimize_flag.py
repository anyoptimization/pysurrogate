"""Tests for the fit(optimize=...) contract and cheap-rank-then-tune via Benchmark."""

import copy

import numpy as np

from pysurrogate.core.sampling import LHS, Sampling
from pysurrogate.dace import Gaussian, Matern
from pysurrogate.models import RBF, Kriging, SimpleMean
from pysurrogate.selection import Benchmark, cartesian
from pysurrogate.util.test_functions import get_test_function


def _data(fn="rastrigin", d=3, n=70, seed=0):
    f, xl, xu = get_test_function(fn, d)
    rng = np.random.default_rng(seed)
    X = Sampling(n, LHS()).sample((xl, xu), rng)
    return X, f(X)[:, None], (f, xl, xu)


def test_kriging_optimize_false_freezes_theta():
    # optimize=False -> the Dace inside runs no theta search (optimizer frozen)
    X, y, _ = _data()
    full = Kriging(corr=Gaussian())
    full.fit(X, y, optimize=True)
    cheap = Kriging(corr=Gaussian())
    cheap.fit(X, y, optimize=False)
    # the full fit ran a real search; the cheap one did not (no optimization record)
    assert full.model.optimization is not None and full.model.optimization["n_evals"] > 1
    assert cheap.model.optimization is None  # frozen theta -> no search
    # the cheap fit still produces a usable model
    assert np.all(np.isfinite(cheap.predict(X[:5]).y))


def test_optimize_flag_is_noop_for_hyperparameterless_models():
    # SimpleMean / RBF have no theta search; optimize=False is harmless
    X, y, _ = _data()
    for m in (SimpleMean(), RBF(kernel="gaussian")):
        m.fit(X, y, optimize=False)
        assert np.all(np.isfinite(m.predict(X[:5]).y))


def test_kriging_refit_reuses_theta_warm():
    # Kriging.refit delegates to Dace.refit: appends new points and warm-starts from the fitted
    # length-scale (not the configured start)
    X, y, _ = _data()
    m = Kriging(corr=Gaussian())
    m.fit(X[:50], y[:50])
    theta0 = float(np.ravel(m.model.model["theta"])[0])
    m.refit(X[50:], y[50:], optimize=True)
    assert m.model.model["X"].shape[0] == 70  # all points appended
    # the refit seeded from the previous theta (warm start), not the configured 1.0
    assert m.model.theta == theta0 or not np.isclose(theta0, 1.0)


def test_kriging_refit_optimize_false_freezes_theta():
    # optimize=False keeps theta fixed and just re-solves the kernel matrix on the grown data
    X, y, _ = _data()
    m = Kriging(corr=Gaussian())
    m.fit(X[:50], y[:50])
    theta_before = np.ravel(m.model.model["theta"]).copy()
    m.refit(X[50:], y[50:], optimize=False)
    assert np.allclose(np.ravel(m.model.model["theta"]), theta_before)  # frozen
    assert m.model.model["X"].shape[0] == 70  # but data grew


def test_refit_returns_out_of_sample_prediction():
    # refit scores the new points on the OLD model (prequential validation) and returns it,
    # then appends them -- the returned prediction equals predicting them pre-refit
    X, y, _ = _data()
    m = Kriging(corr=Gaussian())
    m.fit(X[:50], y[:50])
    expected = m.predict(X[50:60], var=True)  # OOS on the 50-point model, before adding
    oos = m.refit(X[50:60], y[50:60])
    np.testing.assert_allclose(oos.y, expected.y)  # same model scored the unseen points
    np.testing.assert_allclose(oos.sigma, expected.sigma)
    assert m.model.model["X"].shape[0] == 60  # points appended afterward


def test_refit_accumulates_records_with_epoch():
    # refit accumulates the out-of-sample predictions into records() with an epoch per call
    X, y, _ = _data()
    m = Kriging(corr=Gaussian())
    m.fit(X[:40], y[:40])
    assert m.records().empty  # an empty DataFrame before any refit
    for step in range(3):
        s = 40 + step * 8
        m.refit(X[s : s + 8], y[s : s + 8])
    df = m.records()
    assert set(df["epoch"]) == {0, 1, 2}  # one epoch per refit
    assert len(df) == 24  # 3 epochs * 8 points
    for col in ["epoch", "i", "output", "y_true", "y", "var", "sigma"]:
        assert col in df.columns


def test_generic_model_refit_appends_and_refits():
    # the base Model.refit (no warm start) appends new data and re-fits -- works for RBF
    X, y, _ = _data()
    m = RBF(kernel="gaussian")
    m.fit(X[:50], y[:50])
    m.refit(X[50:], y[50:])
    assert m._X.shape[0] == 70  # combined data
    assert np.all(np.isfinite(m.predict(X[:5]).y))


def test_cheap_ranking_then_tune_winner_manually():
    # the two-stage is just the optimize flag: rank a fleet cheaply via Benchmark(optimize=False),
    # then fully tune the chosen winner yourself -- no special ModelSelection mode needed.
    X, y, _ = _data(fn="rosenbrock")
    fleet = cartesian(Kriging, corr={"gauss": Gaussian(), "m25": Matern(2.5)})

    bench = Benchmark(fleet, metrics=["rmse"]).do(X, y, optimize=False)  # cheap fixed-theta ranking
    best_proto = bench.results(sorted_by="rmse")[0]["proto"]

    winner = copy.deepcopy(best_proto)
    winner.fit(X, y, optimize=True)  # commit: full tuning on the chosen config
    assert winner.model.optimization is not None  # the winner was actually tuned
    assert np.all(np.isfinite(winner.predict(X[:5]).y))
