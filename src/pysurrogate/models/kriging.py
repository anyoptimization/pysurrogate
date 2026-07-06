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

    ``theta_prior=(mean, lam)`` turns on a MAP prior on the length-scale search (default ``None`` =
    maximum likelihood); see :class:`~pysurrogate.dace.dace.Dace`. Prefer it over pure MLE when the
    surrogate drives an optimization loop -- MLE over-fits short length-scales on small biased
    designs and the resulting over-confidence poisons acquisition; ``lam~=0.01`` is a good start.
    """

    def __init__(
        self,
        regr=None,
        corr=None,
        ARD=False,
        theta=1.0,
        theta_bounds=(0.0, 100.0),
        theta_prior=None,
        selection=None,
        **kwargs,
    ) -> None:
        super().__init__(eliminate_duplicates=True, **kwargs)
        self.regr = regr if regr is not None else LinearRegression()
        self.corr = corr if corr is not None else RationalQuadratic(0.25)
        self.ARD = ARD
        self.theta = theta
        self.theta_bounds = theta_bounds
        self.theta_prior = theta_prior
        # optional hyperparameter-selection strategy (MLE / MAP / held-out), the same object Dace
        # takes; when given it supplies the optimizer / prior / nugget policy (see Dace `selection`).
        self.selection = selection

    def _fit(self, X, y, optimize=True, **kwargs):
        theta, theta_bounds = self.theta, self.theta_bounds

        # ARD (one length-scale per input dimension) broadcasts the scalar start (and bounds, when
        # finite) to a per-dimension vector so the theta search tunes each dimension independently.
        # The start is broadcast even with theta_bounds=None -- Dace reads the ARD dimension from
        # the start vector for an unbounded search, so ARD must not silently collapse to isotropic.
        if self.ARD:
            _, m = X.shape
            theta = np.full(m, theta)
            if theta_bounds is not None:
                lo, hi = theta_bounds
                theta_bounds = (np.full(m, lo), np.full(m, hi))

        # `optimize` is the shared Model-contract lever: forwarded straight to Dace.fit, where
        # optimize=False freezes theta (the cheap fixed-length-scale fit used for model-selection
        # screening and frozen-theta loop refits) and optimize=True runs the configured search.
        self.model = Dace(
            regr=self.regr,
            corr=self.corr,
            theta=theta,
            theta_bounds=theta_bounds,
            theta_prior=self.theta_prior,
            selection=self.selection,
        )
        self.model.fit(X, y, optimize=optimize)

    def _refit(self, X, y, optimize=True):
        # incremental warm-started re-fit via the Dace engine (append + reuse the fitted theta):
        # optimize=True warm-starts the length-scale search, optimize=False freezes it. The generic
        # Model.refit handles the out-of-sample scoring and record; this is just the absorb step.
        self.model.refit(X, y, optimize=optimize)

    def _predict(self, X, var=False, grad=False):
        # Dace.predict now speaks the Model vocabulary (var=, grad=), so this is a direct
        # pass-through with no name translation -- the mean, variance and gradients all share
        # Dace's single Cholesky solve. The Model lifecycle un-normalizes them.
        return self.model.predict(X, var=var, grad=grad)
