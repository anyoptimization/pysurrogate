"""Tests for the Sampling start-point generator, including the empty edge case."""

import numpy as np

from pysurrogate.core.sampling import LHS, Random, Sampling


def test_sampling_includes_forced_points_and_fills():
    rng = np.random.default_rng(0)
    x0 = np.array([0.5, 0.5])
    pts = Sampling(8, LHS()).sample(([0, 0], [1, 1]), rng, include=[x0])

    assert pts.shape == (8, 2)
    assert np.any(np.all(np.isclose(pts, x0), axis=1))


def test_sampling_empty_keeps_column_shape():
    # regression: n=0 with no forced points must still return shape (0, p), not (0,)
    pts = Sampling(0, Random()).sample(([0, 0, 0], [1, 1, 1]), np.random.default_rng(0))

    assert pts.shape == (0, 3)
    # indexing a column must not raise
    assert pts[:, 1].shape == (0,)
