"""Kriging surrogate: the ``Dace`` engine dressed in the ``Model`` lifecycle."""

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.dace import Dace, LinearRegression, RationalQuadratic


class Kriging(Model):
    """A ``Model``-lifecycle Kriging built on the ``Dace`` engine.

    This is a thin adapter, not a second Kriging implementation: all the math lives in
    ``Dace``. It exists so Kriging can sit beside the other ``Model`` backends (uniform
    ``fit``/``predict``, normalization, model selection) and so duplicate design points --
    which make the correlation matrix singular -- are eliminated before fitting.

    ``regr`` and ``corr`` are ``Dace`` objects passed straight through. The default kernel is
    ``RationalQuadratic(0.25)`` (the best all-round performer across the test-function
    benchmark) with a linear regression trend.
    """

    def __init__(self, regr=None, corr=None, ARD=False, theta=1.0, thetaL=1e-5, thetaU=100.0, **kwargs) -> None:
        super().__init__(eliminate_duplicates=True, **kwargs)
        self.regr = regr if regr is not None else LinearRegression()
        self.corr = corr if corr is not None else RationalQuadratic(0.25)
        self.ARD = ARD
        self.theta = theta
        self.thetaL = thetaL
        self.thetaU = thetaU

    def _fit(self, X, y, **kwargs):
        theta, thetaL, thetaU = self.theta, self.thetaL, self.thetaU

        # ARD (one length-scale per input dimension) broadcasts the scalar start/bounds to a
        # per-dimension vector so the theta search tunes each dimension independently.
        if self.ARD and self.thetaL is not None and self.thetaU is not None:
            _, m = X.shape
            theta = np.full(m, theta)
            thetaL = np.full(m, thetaL)
            thetaU = np.full(m, thetaU)

        self.model = Dace(regr=self.regr, corr=self.corr, theta=theta, thetaL=thetaL, thetaU=thetaU)
        self.model.fit(X, y)

    def _predict(self, X, var=False, grad=False):
        # Dace.predict already returns the shared Prediction (its mse/mse_grad are aliases of
        # var/var_grad), so this is a direct pass-through -- the mean, variance and gradients
        # all share Dace's single Cholesky solve. The Model lifecycle un-normalizes them.
        return self.model.predict(X, mse=var, grad=grad)
