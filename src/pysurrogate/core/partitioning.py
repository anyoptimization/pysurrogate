"""Train/validation/test partitioning strategies for cross-validated model evaluation."""

import math
import random
from dataclasses import dataclass

import numpy as np

# Canonical default number of cross-validation folds, shared by the benchmark and calibration
# defaults so the choice lives in one named place instead of scattered bare literals.
DEFAULT_CV_FOLDS = 5


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

    Subclasses implement ``_folds(X, rng, pyrng)`` returning ``(train_idx, test_idx)`` pairs,
    drawing all randomness from the two **local** generators this base threads through -- a
    ``numpy.random.RandomState`` and a ``random.Random``, both seeded from ``self.seed``. Using
    local generators (never the module globals) keeps partitioning from perturbing any other code's
    RNG state and lets concurrent partitionings run without interfering. When ``valid_frac > 0`` a
    validation slice is reserved out of each fold's training indices (from the same local ``rng``).
    """

    def __init__(self, seed=None, valid_frac=0.0) -> None:
        self.seed = seed
        self.valid_frac = valid_frac

    def do(self, X):
        # local generators seeded per call -- no global reseed. RandomState / random.Random (rather
        # than the modern default_rng) are used deliberately: they reproduce the legacy global-seed
        # sequences exactly, so fold assignments are unchanged while the global state is left alone.
        rng = np.random.RandomState(self.seed)
        pyrng = random.Random(self.seed)

        splits = []
        for trn, tst in self._folds(X, rng, pyrng):
            trn = np.asarray(trn, dtype=int)
            tst = np.asarray(tst, dtype=int)
            valid = None
            if self.valid_frac > 0 and len(trn) > 1:
                # clamp so at least one training row survives: a valid_frac near 1 (or a tiny fold)
                # would otherwise carve the whole training set into validation and leave train empty.
                n_valid = min(max(1, int(round(self.valid_frac * len(trn)))), len(trn) - 1)
                perm = rng.permutation(len(trn))
                valid, trn = trn[perm[:n_valid]], trn[perm[n_valid:]]
            splits.append(Split(train=trn, test=tst, valid=valid))
        return splits

    def _folds(self, X, rng, pyrng):
        raise NotImplementedError


class CrossvalidationPartitioning(Partitioning):
    """k-fold cross-validation: each fold holds out one of ``k_folds`` disjoint test blocks."""

    def __init__(self, k_folds=5, randomize=True, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.k_folds = k_folds

    def _folds(self, X, rng, pyrng):
        n = X if isinstance(X, int) else len(X)
        assert n > 1

        k_folds = min(self.k_folds, n)

        indices = list(range(n))
        if self.randomize:
            pyrng.shuffle(indices)

        tst = [[] for _ in range(k_folds)]
        for k in range(n):
            tst[k % k_folds].append(indices[k])

        folds = []
        for fold in tst:
            held = set(fold)
            folds.append(([j for j in indices if j not in held], fold))
        return folds


def default_partitioning(k_folds=DEFAULT_CV_FOLDS, seed=None):
    """The framework's default cross-validation scheme: ``k_folds``-fold :class:`CrossvalidationPartitioning`.

    The single place the default CV is constructed, so the (otherwise scattered) fold count and
    seed are named rather than bare literals. ``DEFAULT_CV_FOLDS`` (5) is the canonical choice --
    enough holdout to be honestly out-of-sample without the cost of leave-one-out. A caller with a
    different cost/accuracy trade-off (e.g. cheap model-selection *screening* over many candidates)
    may still pass a smaller ``k_folds`` explicitly.

    Args:
        k_folds: Number of cross-validation folds.
        seed: Seed for the local fold-assignment RNG (``None`` for an unseeded shuffle).

    Returns:
        A configured :class:`CrossvalidationPartitioning`.
    """
    return CrossvalidationPartitioning(k_folds=k_folds, seed=seed)


class RandomPartitioning(Partitioning):
    """Random hold-out: ``n_sets`` independent splits, each with a ``perc_train`` training fraction."""

    def __init__(self, perc_train=0.7, n_sets=1, **kwargs):
        super().__init__(**kwargs)
        self.n_sets = n_sets
        self.perc_train = perc_train

    def _folds(self, X, rng, pyrng):
        n = X if isinstance(X, int) else len(X)
        n_train = math.ceil(self.perc_train * n)

        folds = []
        for _ in range(self.n_sets):
            perm = rng.permutation(n)
            folds.append((perm[:n_train], perm[n_train:]))
        return folds
