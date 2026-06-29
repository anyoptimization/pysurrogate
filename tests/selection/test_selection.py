"""Tests for the selection layer: factory, benchmark ranking, and model selection."""

import numpy as np

from pysurrogate.models import RBF, SVR, SimpleMean
from pysurrogate.selection import Benchmark, ModelSelection, cartesian


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
    try:
        cartesian(RBF, kernel=["cubic", "cubic"])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate axis tokens should raise")


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


def test_model_selection_picks_and_refits_best():
    X, y = _data()
    sel = ModelSelection({"mean": SimpleMean(), "rbf": RBF(kernel="gaussian"), "svr": SVR()}, sorted_by="rmse")
    sel.fit(X, y)

    assert sel.best["label"] == "rbf"
    # the selection is itself a usable model, refit on all data and predicting through the winner
    pred = sel.predict(X[:5])
    assert pred.y.shape == (5, 1) and np.all(np.isfinite(pred.y))
    assert list(sel.statistics())[0] == "rbf"
