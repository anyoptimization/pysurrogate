"""Kriging surrogate: the ``Dace`` engine dressed in the ``Model`` lifecycle."""

from pysurrogate.dace import Dace, LinearRegression, RationalQuadratic
from pysurrogate.models._dace_backed import DaceBackedModel


class Kriging(DaceBackedModel):
    """A ``Model``-lifecycle Kriging built on the ``Dace`` engine.

    This is a thin adapter, not a second Kriging implementation: all the math lives in
    ``Dace``. It exists so Kriging can sit beside the other ``Model`` backends (uniform
    ``fit``/``predict``, normalization, model selection) and so duplicate design points --
    which make the correlation matrix singular -- are eliminated before fitting.

    ``regr`` and ``corr`` are ``Dace`` objects passed straight through. The default kernel is
    ``RationalQuadratic(0.25)`` (the best all-round performer across the test-function
    benchmark). The default trend is ``LinearRegression`` -- note this differs from
    :class:`~pysurrogate.models.kpls.KPLS` and :class:`~pysurrogate.models.neural.DeepKernelGP`,
    which default to a constant trend; pass ``regr=ConstantRegression()`` to match them.

    ``theta_prior=(mean, lam)`` turns on a MAP prior on the length-scale search (default ``None`` =
    maximum likelihood); see :class:`~pysurrogate.dace.dace.Dace`. Prefer it over pure MLE when the
    surrogate drives an optimization loop -- MLE over-fits short length-scales on small biased
    designs and the resulting over-confidence poisons acquisition; ``lam~=0.01`` is a good start.
    """

    default_regr = LinearRegression

    @staticmethod
    def default_corr():
        """The default kernel: ``RationalQuadratic(0.25)`` (heavy, robust tails)."""
        return RationalQuadratic(0.25)

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
        super().__init__(regr=regr, corr=corr, selection=selection, **kwargs)
        self.ARD = ARD
        self.theta = theta
        self.theta_bounds = theta_bounds
        self.theta_prior = theta_prior

    def _fit(self, X, y, optimize=True, **kwargs):
        theta, theta_bounds = self.theta, self.theta_bounds

        # ARD (one length-scale per input dimension) broadcasts the scalar start (and bounds, when
        # finite) to a per-dimension vector so the theta search tunes each dimension independently.
        if self.ARD:
            theta, theta_bounds = self._broadcast_theta(theta, theta_bounds, X.shape[1])

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
