"""Dace speaks the Model vocabulary: predict(X, var=), the mse alias, and the fit optimize lever."""

import numpy as np

from pysurrogate.dace import Dace, Gaussian


def _data(seed=0, n=25):
    rng = np.random.RandomState(seed)
    X = rng.random((n, 2))
    y = np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2
    return X, y


def test_predict_var_matches_mse_alias():
    X, y = _data()
    model = Dace(corr=Gaussian(), optimizer=None)
    model.fit(X, y)
    q = _data(seed=1, n=5)[0]
    a = model.predict(q, var=True)
    b = model.predict(q, mse=True)
    assert np.allclose(a.y, b.y)
    assert np.allclose(a.var, b.var)
    # var is the canonical name; mse is the read alias on the Prediction
    assert np.allclose(a.var, a.mse)


def test_fit_optimize_false_freezes_theta():
    X, y = _data()
    frozen = Dace(corr=Gaussian(), theta=2.0, theta_bounds=(0.01, 100.0))
    frozen.fit(X, y, optimize=False)
    # theta untouched by a search, and no optimization record was produced
    assert np.allclose(frozen.model["theta"], 2.0)
    assert frozen.optimization is None


def test_fit_optimize_true_searches_theta():
    X, y = _data()
    searched = Dace(corr=Gaussian(), theta=2.0, theta_bounds=(0.01, 100.0))
    searched.fit(X, y, optimize=True)
    assert searched.optimization is not None
    # the search moved theta off its start (the smooth data prefers a different length-scale)
    assert not np.allclose(searched.model["theta"], 2.0)


def test_predict_positional_query_argument():
    X, y = _data()
    model = Dace(corr=Gaussian(), optimizer=None)
    model.fit(X, y)
    # the query argument is now plainly `X` (was the private-looking `_X`)
    pred = model.predict(_data(seed=2, n=4)[0])
    assert pred.y.shape == (4, 1)
