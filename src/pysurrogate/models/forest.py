"""Random forest surrogate model over a discretized design space."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.util.misc import discretize


class RandomForest(Model):
    """Random forest fit on a per-dimension grid, keeping the best target per occupied cell."""

    def __init__(self, n_partitions=15, n_estimators=100, xl=None, xu=None, random_state=42, **kwargs) -> None:
        super().__init__(**kwargs)
        self.xl, self.xu = xl, xu
        self.n_partitions = n_partitions
        self.n_estimators = n_estimators
        self.random_state = random_state  # fixed default keeps fits deterministic; overridable

    def _fit(self, X, y, **kwargs):
        # resolve the grid bounds per fit into fit-local attributes; do NOT overwrite the
        # constructor's xl/xu. Overwriting them froze the bounds after the first fit, so a later
        # fit / refit on grown or shifted data silently re-used the original range and mis-binned
        # the new points (discretize collapses anything outside the range).
        self._xl = self.xl if self.xl is not None else X.min(axis=0)
        self._xu = self.xu if self.xu is not None else X.max(axis=0)

        if y.shape[1] != 1:
            raise ValueError(f"RandomForest supports a single output, got {y.shape[1]}; fit one model per output.")
        y = y[:, 0]
        X = discretize(X, self.n_partitions, self._xl, self._xu)

        # collapse duplicate grid cells, keeping the best (minimum) target per cell. Exact row
        # equality via np.unique -- the old str(x) key truncated wide rows through numpy's print
        # summarization and could merge distinct cells.
        Xg, inv = np.unique(X, axis=0, return_inverse=True)
        yg = np.full(len(Xg), np.inf)
        np.minimum.at(yg, inv, y)

        rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        rf.fit(Xg, yg)
        self.model = rf

    def _predict(self, X, var=False, grad=False):
        Xd = discretize(X, self.n_partitions, self._xl, self._xu)
        y = self.model.predict(Xd)[:, None]

        # forest uncertainty = spread of the per-tree predictions (the ensemble disagreement).
        # grad/var_grad stay None: a forest surface is piecewise-constant, so its gradient is 0
        # almost everywhere and uninformative.
        v = None
        if var:
            per_tree = np.array([tree.predict(Xd) for tree in self.model.estimators_])
            v = per_tree.var(axis=0)[:, None]

        return Prediction(y=y, var=v)
