"""Tests for util.misc: discretize binning edges and duplicate-row detection."""

import numpy as np

from pysurrogate.util.misc import discretize, is_duplicate


def test_discretize_interior_points_hit_expected_bins():
    X = np.array([[0.05], [0.35], [0.75]])
    bins = discretize(X, n_partitions=4, xl=np.array([0.0]), xu=np.array([1.0]))
    assert bins.tolist() == [[0], [1], [3]]


def test_discretize_upper_bound_maps_to_top_bin():
    # regression: X == xu used to fall through argmax-of-all-False and land in bin 0
    X = np.array([[1.0], [0.0]])
    bins = discretize(X, n_partitions=4, xl=np.array([0.0]), xu=np.array([1.0]))
    assert bins.tolist() == [[3], [0]]


def test_discretize_out_of_range_clamps_to_edge_bins():
    # regression: above-range points used to map to bin 0 instead of the top bin
    X = np.array([[2.0], [-1.0]])
    bins = discretize(X, n_partitions=5, xl=np.array([0.0]), xu=np.array([1.0]))
    assert bins.tolist() == [[4], [0]]


def test_discretize_zero_range_dimension_maps_to_bin_zero():
    # a degenerate xl == xu dimension must not divide by zero; everything lands in bin 0
    X = np.array([[0.5, 0.2], [0.5, 0.9]])
    bins = discretize(X, n_partitions=3, xl=np.array([0.5, 0.0]), xu=np.array([0.5, 1.0]))
    assert bins[:, 0].tolist() == [0, 0]
    assert bins[:, 1].tolist() == [0, 2]


def test_discretize_defaults_bounds_from_data():
    X = np.array([[0.0], [5.0], [10.0]])
    bins = discretize(X, n_partitions=2)
    assert bins.tolist() == [[0], [1], [1]]


def test_is_duplicate_marks_later_repeats_only():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    mask = is_duplicate(X)
    assert mask.tolist() == [False, False, True, False, True]
