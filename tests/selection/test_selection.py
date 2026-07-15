"""Tests for the selection layer: factory, benchmark ranking, and model selection."""

import numpy as np
import pytest

from pysurrogate.models import RBF, SVR, Kriging, SimpleMean
from pysurrogate.selection import AutoModel, Benchmark, cartesian
from pysurrogate.selection.metrics import metric_names


def _data(n=60, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.random((n, 2))
    y = (np.sin(3 * X[:, 0]) + (X[:, 1] - 0.5) ** 2).reshape(-1, 1)
    return X, y


def test_cartesian_names_and_count():
    models = cartesian(RBF, kernel=["cubic", "gaussian"], tail=["linear", "constant"])
    assert len(models) == 4
    assert "RBF[cubic,linear]" in models
    assert all(isinstance(m, RBF) for m in models.values())


def test_cartesian_rejects_duplicate_tokens():
    with pytest.raises(ValueError):
        cartesian(RBF, kernel=["cubic", "cubic"])


def test_benchmark_ranks_real_model_above_mean_baseline():
    X, y = _data()
    bench = Benchmark({"mean": SimpleMean(), "rbf": RBF(kernel="gaussian")}).do(X, y)
    ranking = bench.results(sorted_by="rmse")
    assert [r["label"] for r in ranking][0] == "rbf"  # rbf beats the constant mean
    assert ranking[0]["performance"]["rmse"]["mean"] < ranking[-1]["performance"]["rmse"]["mean"]


def test_benchmark_records_per_fold_runs():
    X, y = _data()
    bench = Benchmark({"rbf": RBF(kernel="gaussian")}).do(X, y)
    rec = bench.records["rbf"]
    assert rec["n_runs"] == 3 and rec["n_success"] == 3
    assert rec["performance"]["mae"]["values"].shape == (3,)


def test_benchmark_scores_probabilistic_metrics_with_sigma():
    """A probabilistic metric (needs sigma) must be scored, not silently NaN'd.

    Regression guard: ``Benchmark`` predicted mean-only and never passed ``sigma`` to the metric,
    so every calibration metric raised "requires sigma" and collapsed the whole benchmark. It must
    now request the variance and pass it through.
    """
    X, y = _data()
    bench = Benchmark({"kriging": Kriging()}, metrics=["rmse", "nlpd", "calib"]).do(X, y)
    perf = bench.records["kriging"]["performance"]
    assert np.isfinite(perf["nlpd"]["mean"])  # actually computed, not NaN
    assert np.isfinite(perf["calib"]["mean"])


def test_benchmark_probabilistic_metric_skips_models_without_sigma():
    """A no-sigma model (SVR) scores NaN on a probabilistic metric but keeps its point metrics."""
    X, y = _data()
    bench = Benchmark({"svr": SVR()}, metrics=["rmse", "nlpd"]).do(X, y)
    perf = bench.records["svr"]["performance"]
    assert np.isfinite(perf["rmse"]["mean"])  # point metric still works
    assert np.isnan(perf["nlpd"]["mean"])  # probabilistic metric skipped, model not crashed


# Every registered metric must be computable through the whole selection stack -- the coverage gap
# that let the "Benchmark never passes sigma" and "AutoModel never computes the sorted_by metric"
# bugs through. Parametrizing over the live registry means a new metric is auto-covered.
@pytest.mark.parametrize("metric", metric_names())
def test_benchmark_computes_every_registered_metric(metric):
    X, y = _data()
    # Kriging provides sigma, so even probabilistic metrics resolve to a finite score.
    bench = Benchmark({"kriging": Kriging()}, metrics=[metric]).do(X, y)
    assert np.isfinite(bench.records["kriging"]["performance"][metric]["mean"])


@pytest.mark.parametrize("metric", metric_names())
def test_automodel_selects_by_every_registered_metric(metric):
    X, y = _data()
    auto = AutoModel(models={"kriging": Kriging(), "rbf": RBF(kernel="gaussian")}, sorted_by=metric).fit(X, y)
    assert auto.success
    assert list(auto.statistics())  # a winner was actually chosen by this metric


def test_model_selection_picks_and_refits_best():
    X, y = _data()
    sel = AutoModel({"mean": SimpleMean(), "rbf": RBF(kernel="gaussian"), "svr": SVR()}, sorted_by="rmse")
    sel.fit(X, y)

    assert sel.best["label"] == "rbf"
    # the selection is itself a usable model, refit on all data and predicting through the winner
    pred = sel.predict(X[:5])
    assert pred.y.shape == (5, 1) and np.all(np.isfinite(pred.y))
    assert list(sel.statistics())[0] == "rbf"


def test_automodel_statistics_empty_before_fit():
    # before fit there is no ranking yet -> an empty dict, not None (callers can iterate safely)
    assert AutoModel({"mean": SimpleMean()}).statistics() == {}


def test_model_selection_defaults_to_recommended_fleet():
    # AutoModel() with no args selects over the recommended default fleet
    X, y = _data()
    sel = AutoModel(sorted_by="rmse")
    sel.fit(X, y)
    assert sel.best is not None and len(sel.ranking) > 1  # a real fleet was benchmarked
    assert np.all(np.isfinite(sel.predict(X[:5]).y))


def test_model_selection_refit_refits_winner_only():
    # refit refits the SELECTED winner on new data (no re-selection); the winner is unchanged
    X, y = _data()
    sel = AutoModel({"mean": SimpleMean(), "rbf": RBF(kernel="gaussian"), "svr": SVR()}, sorted_by="rmse")
    sel.fit(X[:40], y[:40])
    winner_before = sel.best["label"]
    n_before = sel.model._X.shape[0]

    sel.refit(X[40:], y[40:])  # append the rest to the winner

    assert sel.best["label"] == winner_before  # not re-selected
    assert sel.model._X.shape[0] == n_before + (len(X) - 40)  # winner grew by the new points
    assert np.all(np.isfinite(sel.predict(X[:5]).y))


def test_model_selection_refit_best_false_keeps_fold_model():
    # the renamed refit_best flag still controls whether the winner is refit on all data
    from pysurrogate.core.partitioning import RandomPartitioning

    X, y = _data()
    sel = AutoModel(
        {"rbf": RBF(kernel="gaussian")},
        sorted_by="rmse",
        refit_best=False,
        partitioning=RandomPartitioning(perc_train=0.7, n_sets=1, seed=0),
    )
    sel.fit(X, y)
    assert np.all(np.isfinite(sel.predict(X[:5]).y))
