"""Regression tests for the SimpleMean baseline, including the multi-output path."""

import numpy as np

from pysurrogate.models import SimpleMean


def test_simple_mean_single_output_shape_and_value():
    rng = np.random.RandomState(0)
    X = rng.random((20, 3))
    y = rng.random((20, 1))

    model = SimpleMean().fit(X, y)
    pred = model.predict(rng.random((5, 3)))

    assert pred.y.shape == (5, 1)
    np.testing.assert_allclose(pred.y, np.full((5, 1), y.mean()))


def test_simple_mean_multi_output_does_not_break_broadcasting():
    # regression: a (q,) per-output mean previously failed to broadcast into (m, 1)
    rng = np.random.RandomState(1)
    X = rng.random((30, 2))
    y = rng.random((30, 3))

    model = SimpleMean().fit(X, y)
    pred = model.predict(rng.random((7, 2)))

    assert pred.y.shape == (7, 3)
    np.testing.assert_allclose(pred.y, np.tile(y.mean(axis=0), (7, 1)))
