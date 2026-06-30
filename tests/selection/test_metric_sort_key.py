"""Tests for the shared, target-aware metric ranking key used by Benchmark and StudyResult."""

import numpy as np

from pysurrogate.selection.metrics import metric_sort_key


def test_lower_is_better_metric_orders_ascending():
    # rmse: smaller value is better -> smaller key
    assert metric_sort_key("rmse", 0.1) < metric_sort_key("rmse", 0.5)


def test_greater_is_better_metric_orders_descending():
    # r2: larger value is better -> smaller key for the larger value
    assert metric_sort_key("r2", 0.9) < metric_sort_key("r2", 0.2)


def test_target_metric_ranks_by_distance_to_target():
    # calib has target=1.0: a ratio of 0.9 (|0.1|) must beat 1.5 (|0.5|), even though
    # greater_is_better=False would otherwise wrongly prefer the smaller raw value
    assert metric_sort_key("calib", 0.9) < metric_sort_key("calib", 1.5)
    assert metric_sort_key("calib", 1.0) < metric_sort_key("calib", 0.5)


def test_non_finite_value_sorts_last():
    assert metric_sort_key("rmse", np.nan) == np.inf
    assert metric_sort_key("rmse", np.inf) == np.inf
    assert metric_sort_key("rmse", None) == np.inf
