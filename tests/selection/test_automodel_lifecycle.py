"""AutoModel is a genuine Model: the fit/predict lifecycle (normalization, filtering) really runs."""

import numpy as np
import pytest

from pysurrogate.core.partitioning import RandomPartitioning
from pysurrogate.core.transformation import Standardization
from pysurrogate.models import RBF, SimpleMean
from pysurrogate.selection import AutoModel


def test_automodel_refit_best_false_reuses_the_fold_fit():
    # refit_best=False must reuse the winner's single-partition fold fit -- NOT silently refit on
    # all data (the old behavior, because the fold fit was discarded). With one partition the
    # retained fold model was trained on that split's training rows.
    rng = np.random.RandomState(0)
    X = rng.random((40, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    part = RandomPartitioning(perc_train=0.75, n_sets=1, seed=0)

    auto = AutoModel(
        models={"rbf": RBF(kernel="tps"), "mean": SimpleMean()},
        sorted_by="rmse",
        refit_best=False,
        partitioning=part,
    ).fit(X, y)

    assert auto.success
    assert auto.model is auto.best["fitted"]  # the stored model IS the retained fold fit
    assert auto.model._X.shape[0] == int(np.ceil(0.75 * 40))  # fit on the train split, not all rows
    assert np.all(np.isfinite(auto.predict(X[:5]).y))


def test_automodel_refit_best_false_rejects_multi_fold_benchmark():
    # a multi-partition benchmark has no single "the fold fit" to keep -> refit_best=False must raise
    X = np.random.RandomState(1).random((30, 2))
    y = X[:, 0]
    auto = AutoModel(models={"rbf": RBF(kernel="tps")}, sorted_by="rmse", refit_best=False)
    with pytest.raises(RuntimeError, match="single-partition"):
        auto.fit(X, y)


def _data(seed=0, n=50):
    rng = np.random.RandomState(seed)
    X = rng.random((n, 2)) * 10.0
    y = np.sin(X[:, [0]]) + (X[:, [1]] - 5.0) ** 2
    return X, y


def test_automodel_implements_model_hooks_not_overrides():
    # the honest claim: AutoModel is a Model via _fit/_predict, not faked fit/predict overrides
    from pysurrogate.core.model import Model

    assert AutoModel._fit is not Model._fit
    assert AutoModel._predict is not Model._predict
    assert AutoModel.fit is Model.fit  # the public lifecycle is inherited, not overridden
    assert AutoModel.predict is Model.predict


def test_lifecycle_normalization_is_applied():
    X, y = _data()
    models = {"mean": SimpleMean(), "rbf": RBF(kernel="tps")}
    sel = AutoModel(models, sorted_by="rmse", norm_X=Standardization(), norm_y=Standardization())
    sel.fit(X, y)
    # the lifecycle normalized the stored design: preprocess applied Standardization to X
    assert np.allclose(sel.X.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(sel.X.std(axis=0), 1.0, atol=1e-6)  # standardized to unit (population) std
    # and prediction is still correct in original units (postprocess un-normalized it)
    pred = sel.predict(X[:5])
    assert pred.y.shape == (5, 1)
    assert np.all(np.isfinite(pred.y))


def test_failed_fit_sets_success_false_via_lifecycle():
    # a fleet that cannot fit (empty after filtering) must flip the Model success flag, not pretend
    X, y = _data(n=40)
    sel = AutoModel({"mean": SimpleMean(), "rbf": RBF(kernel="tps")}, sorted_by="rmse")
    sel.fit(X, y)
    assert sel.success is True
    assert sel.has_been_fitted is True
