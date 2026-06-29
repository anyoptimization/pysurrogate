"""Train/validation/test partitioning strategies for cross-validated model evaluation."""

import math
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    """One fold's index sets: training, test, and an optional validation slice.

    ``valid`` is ``None`` unless a validation fraction was requested -- carved out of the training
    rows for hyperparameter tuning / early stopping (e.g. the ``Dace`` engine's theta-selection
    validation), leaving ``test`` purely held out for scoring.

    Attributes:
        train: Integer indices of the training rows.
        test: Integer indices of the held-out test rows.
        valid: Integer indices of the validation rows, or ``None`` when none were reserved.
    """

    train: np.ndarray
    test: np.ndarray
    valid: np.ndarray | None = None


class Partitioning:
    """Base class: turns a dataset size (or array) into a list of :class:`Split` folds.

    Subclasses implement ``_folds`` returning ``(train_idx, test_idx)`` pairs; this base seeds the
    RNGs for reproducibility and, when ``valid_frac > 0``, reserves a validation slice out of each
    fold's training indices.
    """

    def __init__(self, seed=None, valid_frac=0.0) -> None:
        self.seed = seed
        self.valid_frac = valid_frac

    def do(self, X):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        splits = []
        for trn, tst in self._folds(X):
            trn = np.asarray(trn, dtype=int)
            tst = np.asarray(tst, dtype=int)
            valid = None
            if self.valid_frac > 0 and len(trn) > 1:
                n_valid = max(1, int(round(self.valid_frac * len(trn))))
                perm = np.random.permutation(len(trn))
                valid, trn = trn[perm[:n_valid]], trn[perm[n_valid:]]
            splits.append(Split(train=trn, test=tst, valid=valid))
        return splits

    def _folds(self, X):
        raise NotImplementedError


class CrossvalidationPartitioning(Partitioning):
    """k-fold cross-validation: each fold holds out one of ``k_folds`` disjoint test blocks."""

    def __init__(self, k_folds=5, randomize=True, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.k_folds = k_folds

    def _folds(self, X):
        n = X if isinstance(X, int) else len(X)
        assert n > 1

        k_folds = min(self.k_folds, n)

        indices = list(range(n))
        if self.randomize:
            random.shuffle(indices)

        tst = [[] for _ in range(k_folds)]
        for k in range(n):
            tst[k % k_folds].append(indices[k])

        return [([j for j in indices if j not in set(fold)], fold) for fold in tst]


class RandomPartitioning(Partitioning):
    """Random hold-out: ``n_sets`` independent splits, each with a ``perc_train`` training fraction."""

    def __init__(self, perc_train=0.7, n_sets=1, **kwargs):
        super().__init__(**kwargs)
        self.n_sets = n_sets
        self.perc_train = perc_train

    def _folds(self, X):
        n = X if isinstance(X, int) else len(X)
        n_train = math.ceil(self.perc_train * n)

        folds = []
        for _ in range(self.n_sets):
            perm = np.random.permutation(n)
            folds.append((perm[:n_train], perm[n_train:]))
        return folds
