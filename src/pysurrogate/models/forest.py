"""Random forest surrogate model over a discretized design space."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.util.misc import discretize


class RandomForest(Model):
    """Random forest fit on a per-dimension grid, keeping the best target per occupied cell."""

    def __init__(self, n_partitions=15, n_estimators=100, xl=None, xu=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.xl, self.xu = xl, xu
        self.n_partitions = n_partitions
        self.n_estimators = n_estimators

    def _fit(self, X, y, **kwargs):
        if self.xl is None:
            self.xl = X.min(axis=0)
        if self.xu is None:
            self.xu = X.max(axis=0)

        y = y[:, 0]
        X = discretize(X, self.n_partitions, self.xl, self.xu)

        # collapse duplicate grid cells, keeping the best (minimum) target per cell
        cells: dict[str, dict] = {}
        for i, x in enumerate(X):
            s = str(x)
            if s not in cells:
                cells[s] = dict(X=x, y=y[i], n=1)
            else:
                cells[s] = dict(X=x, y=min(cells[s]["y"], y[i]), n=cells[s]["n"] + 1)

        Xg = np.zeros((len(cells), X.shape[1]))
        yg = np.zeros(len(cells))
        for i, e in enumerate(cells.values()):
            Xg[i] = e["X"]
            yg[i] = e["y"]

        rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        rf.fit(Xg, yg)
        self.model = rf

    def _predict(self, X, var=False, grad=False):
        Xd = discretize(X, self.n_partitions, self.xl, self.xu)
        y = self.model.predict(Xd)[:, None]

        # forest uncertainty = spread of the per-tree predictions (the ensemble disagreement).
        # grad/var_grad stay None: a forest surface is piecewise-constant, so its gradient is 0
        # almost everywhere and uninformative.
        v = None
        if var:
            per_tree = np.array([tree.predict(Xd) for tree in self.model.estimators_])
            v = per_tree.var(axis=0)[:, None]

        return Prediction(y=y, var=v)
