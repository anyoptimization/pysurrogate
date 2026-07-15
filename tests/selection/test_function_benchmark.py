"""Tests for FunctionBenchmark: predictions-DataFrame schema, roles, and groupby scoring."""

import numpy as np
import pytest

from pysurrogate.core.sampling import LHS, Random, Sampling
from pysurrogate.dace import Gaussian
from pysurrogate.models import Kriging, SimpleMean
from pysurrogate.selection import FunctionBenchmark, score
from pysurrogate.selection.metrics import calc_metric
from pysurrogate.util.test_functions import get_test_function


def _bench(**kw):
    f, xl, xu = get_test_function("sphere", n_var=2)
    models = {"krig": Kriging(corr=Gaussian()), "mean": SimpleMean()}
    defaults = dict(train=Sampling(30, LHS()), valid=Sampling(15, LHS()), test=Sampling(200, Random()), random_state=0)
    defaults.update(kw)
    return FunctionBenchmark(f, xl, xu, models, **defaults)


def test_run_predictions_schema_and_roles():
    df = _bench().run()
    # the fixed columns plus input coordinates are present
    for col in ["rep", "model", "role", "i", "output", "y_true", "y", "var", "sigma", "x0", "x1"]:
        assert col in df.columns
    # every model predicted on every role
    assert set(df["role"]) == {"train", "valid", "test"}
    assert set(df["model"]) == {"krig", "mean"}
    # row count = sum over roles of (n_points * n_models), single output
    assert (df["role"] == "test").sum() == 200 * 2
    assert (df["role"] == "train").sum() == 30 * 2


def test_no_valid_omits_role():
    df = _bench(valid=None).run()
    assert set(df["role"]) == {"train", "test"}


def test_score_matches_handcomputed_rmse():
    df = _bench().run()
    table = score(df.query("role == 'test'"), ["rmse"], by=["model"])
    g = df.query("role == 'test' and model == 'krig'")
    expected = calc_metric("rmse", g["y_true"].to_numpy(), g["y"].to_numpy())
    assert np.isclose(table.loc["krig", "rmse"], expected)


def test_uncertainty_model_has_sigma_baseline_does_not():
    df = _bench().run()
    krig = df.query("model == 'krig'")
    mean = df.query("model == 'mean'")
    assert np.isfinite(krig["sigma"]).all()  # Kriging reports uncertainty
    assert mean["sigma"].isna().all()  # SimpleMean does not
    # a probabilistic metric is finite for Kriging, NaN for the baseline
    tbl = score(df.query("role == 'test'"), ["nlpd"], by=["model"])
    assert np.isfinite(tbl.loc["krig", "nlpd"])
    assert np.isnan(tbl.loc["mean", "nlpd"])


def test_reproducible_under_seed():
    a = _bench(random_state=7).run()
    b = _bench(random_state=7).run()
    np.testing.assert_array_equal(a["y_true"].to_numpy(), b["y_true"].to_numpy())
    np.testing.assert_allclose(a["y"].to_numpy(), b["y"].to_numpy())


def test_replications_axis():
    df = _bench(replications=3).run()
    assert set(df["rep"]) == {0, 1, 2}
    # independent re-draws -> the test designs differ across replications
    t0 = df.query("rep == 0 and role == 'test' and model == 'krig'")["x0"].to_numpy()
    t1 = df.query("rep == 1 and role == 'test' and model == 'krig'")["x0"].to_numpy()
    assert not np.array_equal(t0, t1)


def test_predict_failure_is_guarded_like_fit_failure():
    # regression: a model that fits but then fails to PREDICT used to abort the whole run;
    # it must count as a failure for that replication and leave the other models' rows intact.
    class PredictFails(SimpleMean):
        def _predict(self, X, var=False, grad=False):
            raise RuntimeError("boom")

    f, xl, xu = get_test_function("sphere", n_var=2)
    models = {"bad": PredictFails(), "mean": SimpleMean()}
    bench = FunctionBenchmark(f, xl, xu, models, train=Sampling(20, LHS()), test=Sampling(50, Random()), random_state=0)
    df = bench.run()
    assert bench.failures["bad"] == 1
    assert bench.failures["mean"] == 0
    assert set(df["model"]) == {"mean"}  # no partial rows from the failing model

    with pytest.raises(RuntimeError, match="boom"):
        FunctionBenchmark(f, xl, xu, {"bad": PredictFails()}, train=Sampling(20, LHS()), raise_exception=True).run()


def test_generalization_gap_diagnostic():
    # train RMSE should be far below test RMSE for an interpolating Kriging on a smooth function
    df = _bench().run()
    by_role = score(df.query("model == 'krig'"), ["rmse"], by=["role"])
    assert by_role.loc["train", "rmse"] < by_role.loc["test", "rmse"]
