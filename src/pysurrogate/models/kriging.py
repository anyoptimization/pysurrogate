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

    def __init__(self, regr=None, corr=None, ARD=False, theta=1.0, theta_bounds=(0.0, 100.0), **kwargs) -> None:
        super().__init__(eliminate_duplicates=True, **kwargs)
        self.regr = regr if regr is not None else LinearRegression()
        self.corr = corr if corr is not None else RationalQuadratic(0.25)
        self.ARD = ARD
        self.theta = theta
        self.theta_bounds = theta_bounds

    def _fit(self, X, y, optimize=True, **kwargs):
        theta, theta_bounds = self.theta, self.theta_bounds

        # ARD (one length-scale per input dimension) broadcasts the scalar start/bounds to a
        # per-dimension vector so the theta search tunes each dimension independently.
        if self.ARD and theta_bounds is not None:
            _, m = X.shape
            lo, hi = theta_bounds
            theta = np.full(m, theta)
            theta_bounds = (np.full(m, lo), np.full(m, hi))

        # optimize=False -> freeze theta (optimizer=None): the cheap fixed-length-scale fit used
        # for model-selection screening and frozen-theta loop refits. optimize=True -> the default
        # theta search.
        extra = {} if optimize else {"optimizer": None}
        self.model = Dace(regr=self.regr, corr=self.corr, theta=theta, theta_bounds=theta_bounds, **extra)
        self.model.fit(X, y)

    def refit(self, X, y, optimize=True):
        """Score the new points out-of-sample, then incrementally refit (warm-started theta).

        Like :meth:`Model.refit`, the new points are first predicted by the current model and that
        out-of-sample :class:`~pysurrogate.core.prediction.Prediction` is returned (prequential
        validation -- collect it against ``y``). The re-fit then delegates to :meth:`Dace.refit`,
        which appends the points and reuses the previously fitted theta: ``optimize=True``
        warm-starts the length-scale search from it, ``optimize=False`` freezes it and only
        re-solves the kernel matrix on the grown data.

        Args:
            X: The new input points to add (only the additions).
            y: The targets for the new points.
            optimize: ``True`` warm-starts the theta search; ``False`` freezes theta.

        Returns:
            The out-of-sample :class:`~pysurrogate.core.prediction.Prediction` of ``X`` from the
            model *before* the new points were added.

        Raises:
            Exception: If called before a successful :meth:`fit`.
        """
        if self.model is None:
            raise Exception("refit() requires a prior fit(); call fit() first.")
        out_of_sample = self.predict(X, var=True)  # OLD model scores the unseen points
        self._record(X, y, out_of_sample)  # accumulate into self.validation (epoch-stamped)
        self.model.refit(X, y, optimize=optimize)
        return out_of_sample

    def _predict(self, X, var=False, grad=False):
        # Dace.predict already returns the shared Prediction (its mse/mse_grad are aliases of
        # var/var_grad), so this is a direct pass-through -- the mean, variance and gradients
        # all share Dace's single Cholesky solve. The Model lifecycle un-normalizes them.
        return self.model.predict(X, mse=var, grad=grad)
