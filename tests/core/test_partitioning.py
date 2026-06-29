"""Tests for partitioning: k-fold coverage, random hold-out, and validation slices."""

import numpy as np

from pysurrogate.core import CrossvalidationPartitioning, RandomPartitioning, Split


def test_kfold_covers_every_point_once_as_test():
    splits = CrossvalidationPartitioning(k_folds=4, seed=0).do(20)
    assert len(splits) == 4
    test_union = np.concatenate([s.test for s in splits])
    assert sorted(test_union.tolist()) == list(range(20))  # every index tested exactly once


def test_kfold_train_and_test_are_disjoint():
    for s in CrossvalidationPartitioning(k_folds=3, seed=1).do(15):
        assert set(s.train.tolist()).isdisjoint(s.test.tolist())
        assert s.valid is None  # no validation requested


def test_validation_slice_is_carved_from_training():
    splits = CrossvalidationPartitioning(k_folds=4, seed=2, valid_frac=0.25).do(40)
    for s in splits:
        assert s.valid is not None
        # train / valid / test are mutually disjoint and valid came out of the old training pool
        assert set(s.train.tolist()).isdisjoint(s.valid.tolist())
        assert set(s.test.tolist()).isdisjoint(s.valid.tolist())
        assert len(s.valid) >= 1


def test_random_partitioning_split_sizes():
    splits = RandomPartitioning(perc_train=0.7, n_sets=3, seed=0).do(100)
    assert len(splits) == 3
    for s in splits:
        assert isinstance(s, Split)
        assert len(s.train) + len(s.test) == 100
        assert abs(len(s.train) - 70) <= 1


def test_seed_is_reproducible():
    a = CrossvalidationPartitioning(k_folds=5, seed=7).do(30)
    b = CrossvalidationPartitioning(k_folds=5, seed=7).do(30)
    assert all(np.array_equal(x.test, y.test) for x, y in zip(a, b))
