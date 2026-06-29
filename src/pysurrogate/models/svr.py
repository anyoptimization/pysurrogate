"""Support vector regression (SVR) surrogate model."""

from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from sklearn.svm import SVR as _SVR  # type: ignore[import-untyped]

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction


class SVR(Model):
    """Support vector regression, fit through a standardizing scikit-learn pipeline."""

    def __init__(self, kernel="rbf", eps=0.1, C=10.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel
        self.eps = eps
        self.C = C

    def _fit(self, X, y, **kwargs):
        svr = _SVR(kernel=self.kernel, epsilon=self.eps, C=self.C, gamma="scale", degree=3, tol=0.001, shrinking=True)
        regr = make_pipeline(StandardScaler(), svr)
        regr.fit(X, y[:, 0])
        self.model = regr

    def _predict(self, X, var=False, grad=False):
        return Prediction(y=self.model.predict(X)[:, None])
