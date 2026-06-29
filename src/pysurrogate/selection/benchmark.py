"""Benchmark: fit and score a set of surrogate models on the same data via cross-validation."""

import copy

import numpy as np

from pysurrogate.core.metrics import POINT_METRICS, calc_metric, greater_is_better
from pysurrogate.core.partitioning import CrossvalidationPartitioning
from pysurrogate.selection.factory import as_named
from pysurrogate.util.misc import at_least2d


def _aggregate(values):
    """Summarize a per-fold metric list, ignoring non-finite folds."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, values=arr)
    return dict(
        mean=float(finite.mean()), std=float(finite.std()), min=float(finite.min()), max=float(finite.max()), values=arr
    )


class Benchmark:
    """Cross-validate a collection of model prototypes on one data set and rank them by a metric.

    Each model is deep-copied and fit on every training fold, then scored on the held-out fold
    with the requested metrics; the per-fold scores are aggregated to mean/std/min/max per model.
    """

    def __init__(self, models, metrics=None, partitioning=None, raise_exception=False):
        self.models = as_named(models)
        self.metrics = list(POINT_METRICS) if metrics is None else list(metrics)
        self.partitioning = partitioning
        self.raise_exception = raise_exception
        self.records = None

    def do(self, X, y, partitions=None):
        X = at_least2d(np.asarray(X, dtype=float), expand="r")
        y = at_least2d(np.asarray(y, dtype=float), expand="c")[:, 0]
        assert len(X) == len(y)

        if partitions is None:
            partitioning = self.partitioning or CrossvalidationPartitioning(k_folds=3, seed=1)
            partitions = partitioning.do(len(X))

        records = {}
        for name, proto in self.models.items():
            per_metric: dict = {m: [] for m in self.metrics}
            n_success = 0
            for split in partitions:
                trn, tst = split.train, split.test
                try:
                    model = copy.deepcopy(proto)
                    model.fit(X[trn], y[trn])
                    y_hat = model.predict(X[tst]).y[:, 0]
                    for m in self.metrics:
                        per_metric[m].append(calc_metric(m, y[tst], y_hat))
                    n_success += 1
                except Exception:
                    if self.raise_exception:
                        raise
                    for m in self.metrics:
                        per_metric[m].append(np.nan)

            records[name] = dict(
                label=name,
                proto=proto,
                n_runs=len(partitions),
                n_success=n_success,
                success=n_success > 0,
                performance={m: _aggregate(per_metric[m]) for m in self.metrics},
            )

        self.records = records
        return self

    def results(self, sorted_by="mae", only_successful=True):
        """Return the per-model records as a list, sorted by a metric's mean (direction-aware).

        Args:
            sorted_by: Metric name to rank by; its registry direction sets ascending/descending.
            only_successful: Drop models that failed on every fold.

        Returns:
            A list of per-model record dicts, best first. A non-finite mean sorts last.
        """
        if self.records is None:
            raise RuntimeError("call do(X, y) before results().")

        rows = [r for r in self.records.values() if (not only_successful or r["success"])]
        gib = greater_is_better(sorted_by)

        def key(record):
            val = record["performance"][sorted_by]["mean"]
            if not np.isfinite(val):
                return np.inf
            return -val if gib else val

        return sorted(rows, key=key)

    def frame(self, sorted_by="mae"):
        """Return a pandas DataFrame of per-model metric means, rows sorted by ``sorted_by``."""
        import pandas as pd  # type: ignore[import-untyped]

        rows = self.results(sorted_by=sorted_by, only_successful=False)
        data = {r["label"]: {m: r["performance"][m]["mean"] for m in self.metrics} for r in rows}
        return pd.DataFrame(data).T
