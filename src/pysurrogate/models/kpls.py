"""KPLS: high-dimensional Kriging whose length-scales live on a Partial-Least-Squares subspace."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression  # type: ignore[import-untyped]

from pysurrogate.core.kernel import KPLSKernel
from pysurrogate.core.transformation import standardize
from pysurrogate.dace import ConstantRegression, Dace, Gaussian
from pysurrogate.models._dace_backed import DaceBackedModel
from pysurrogate.optimizer import Adam

# sentinel: "optimizer not specified" -> use the KPLS default below. Distinct from optimizer=None,
# which Dace reads as "freeze theta (no search)".
_DEFAULT_OPTIMIZER = object()


def _default_optimizer():
    """A small population Adam -- the right-sized default for the KPLS theta search.

    KPLS optimizes only ``n_pls`` (2-4) length-scales over a smooth space, so the engine's
    16-restart screen-and-polish is wasteful. With the reduced-distance kernel each batched
    likelihood evaluation is cheap, so a small Adam population converges in a few steps -- ~2x
    faster than the restart search at the same accuracy across dimensions.
    """
    return Adam(pop_size=8, steps=40)


class KPLS(DaceBackedModel):
    """Kriging with Partial Least Squares -- Kriging that scales to high input dimensions.

    Ordinary ARD Kriging tunes one length-scale per input dimension, so the ``theta`` search
    becomes intractable past ~20 dimensions. KPLS (Bouhlel et al., 2016) keeps the Kriging
    engine but constrains the ``d`` length-scales to a rank-``n_pls`` linear subspace found by
    Partial Least Squares -- reducing the search to ``n_pls`` hyperparameters (typically 2--4)
    regardless of ``d``. The result trains far faster in high dimensions while keeping the
    predictive variance and gradients of a full Gaussian process.

    Mechanically KPLS is a reparameterization, not a new engine: PLS gives a squared-weight
    matrix ``W2`` (shape ``(d, n_pls)``), and the effective per-dimension length-scales are
    ``eta = W2 @ theta`` for the ``n_pls`` optimized ``theta``. For the ``Gaussian`` kernel this
    equals the standard KPLS kernel exactly, so this backend wraps the configured kernel in a
    :class:`~pysurrogate.core.kernel.KPLSKernel` and hands the ``Dace`` engine a length-``n_pls``
    ``theta`` -- the likelihood, theta search, ``predict`` and ``calibrate`` are unchanged.

    The PLS weights are computed in the same standardized space the ``Dace`` engine fits in, so
    the subspace the kernel sees matches the data the model is trained on. The base kernel should
    be product-exponential for the factorization to be exact -- ``Gaussian`` (default) or
    ``Exponential``.
    """

    # Constant trend by default: it is the canonical KPLS choice (matches the SMT reference)
    # and more accurate in high dimensions, where a linear trend's d+1 extra GLS parameters
    # tend to overfit. Pass regr=LinearRegression() to override.
    default_regr = ConstantRegression
    default_corr = Gaussian

    def __init__(
        self,
        regr=None,
        corr=None,
        n_pls=3,
        theta=1.0,
        theta_bounds=(0.0, 100.0),
        optimizer=_DEFAULT_OPTIMIZER,
        selection=None,
        **kwargs,
    ) -> None:
        super().__init__(regr=regr, corr=corr, selection=selection, **kwargs)
        self.n_pls = n_pls
        self.theta = theta
        self.theta_bounds = theta_bounds
        # Which strategy searches the (small, 2-4 parameter) theta space. Unset -> a small
        # population Adam (see _default_optimizer): the reduced-space kernel makes each batched
        # likelihood evaluation cheap, so a small population converges this low-dimensional, smooth
        # problem in few steps -- faster than a restart search at equal accuracy. Pass any
        # core.optimizer.Optimizer to override, or optimizer=None to freeze theta.
        #
        # Resolve the sentinel to a concrete optimizer HERE, not lazily at fit time: the selection
        # layer deep-copies models per fold, and a bare object() sentinel does not survive deepcopy
        # by identity -- storing the real Adam keeps `is`-checks out of the hot path entirely.
        self.optimizer = _default_optimizer() if optimizer is _DEFAULT_OPTIMIZER else optimizer

    def _pls_weights(self, X, y, h):
        """Squared PLS weights ``(d, h)`` in the engine's standardized space.

        The ``Dace`` engine standardizes inputs to zero mean / unit variance (``ddof=1``) before
        fitting, so the kernel operates in that space; PLS is run there too so the length-scale
        subspace matches. Returns ``x_rotations_ ** 2`` -- the per-dimension, per-component
        squared weights the KPLS kernel mixes.
        """
        nX, _, _ = standardize(X)
        nY, _, _ = standardize(y)
        pls = PLSRegression(n_components=h, scale=False).fit(nX, nY)
        return np.square(pls.x_rotations_)  # (d, h)

    def _kpls_engine(self, X, y):
        """Build the ``Dace`` engine with a PLS-reduced kernel and a length-``h`` theta."""
        n, d = X.shape
        # PLS needs 1 <= n_components <= min(n_samples, n_features); clamp the request into range.
        h = max(1, min(self.n_pls, d, n))
        w2 = self._pls_weights(X, y, h)
        kernel = KPLSKernel(self.corr, w2)

        theta, theta_bounds = self._broadcast_theta(self.theta, self.theta_bounds, h)
        # self.optimizer is already a concrete optimizer (or None to freeze), resolved in __init__.
        return Dace(
            regr=self.regr,
            corr=kernel,
            theta=theta,
            theta_bounds=theta_bounds,
            optimizer=self.optimizer,
            selection=self.selection,
        )

    def _fit(self, X, y, optimize=True, **kwargs):
        self.model = self._kpls_engine(X, y)
        self.model.fit(X, y, optimize=optimize)
