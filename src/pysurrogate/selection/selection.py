"""Model selection: benchmark candidate surrogates and return the best, refit on all data."""

import copy

from pysurrogate.core.model import Model
from pysurrogate.selection.benchmark import Benchmark


class ModelSelection(Model):
    """Pick the best surrogate from a set of candidates by cross-validated benchmark.

    Wraps a :class:`~pysurrogate.selection.benchmark.Benchmark`: ``fit`` runs the benchmark on the
    data, ranks the candidates by ``sorted_by``, and (by default) refits the winning prototype on
    the full data set. ``predict`` then delegates to that chosen model. It is itself a ``Model``, so
    a selection can be dropped in anywhere a single surrogate is expected.
    """

    def __init__(self, models, sorted_by="mae", refit=True, partitioning=None):
        super().__init__()
        self.benchmark = models if isinstance(models, Benchmark) else Benchmark(models, partitioning=partitioning)
        self.sorted_by = sorted_by
        self.refit = refit
        self.best = None
        self.ranking = None

    def do(self, X, y):
        self.benchmark.do(X, y)
        ranking = self.benchmark.results(sorted_by=self.sorted_by, only_successful=True)
        if not ranking:
            raise RuntimeError("No candidate model could be fitted successfully.")

        self.ranking = ranking
        self.best = ranking[0]

        if self.refit:
            model = copy.deepcopy(self.best["proto"])
            model.fit(X, y)
        else:
            if self.best["n_runs"] > 1:
                raise RuntimeError("refit=False needs a single-partition benchmark (n_runs == 1).")
            model = copy.deepcopy(self.best["proto"]).fit(X, y)

        self.model = model
        return model

    def fit(self, X, y, **kwargs):
        self.do(X, y)
        self.has_been_fitted = True
        self.success = True
        return self

    def predict(self, X, var=False, grad=False):
        return self.model.predict(X, var=var, grad=grad)

    def statistics(self):
        """Return a ``{model_name: score}`` dict of the ranking metric, best first."""
        if self.ranking is None:
            return None
        return {r["label"]: r["performance"][self.sorted_by]["mean"] for r in self.ranking}
