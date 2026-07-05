"""KPLS: high-dimensional Kriging whose length-scales live on a Partial-Least-Squares subspace."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression  # type: ignore[import-untyped]

from pysurrogate.core.kernel import KPLSKernel
from pysurrogate.core.model import Model
from pysurrogate.dace import ConstantRegression, Dace, Gaussian
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


class KPLS(Model):
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

    def __init__(
        self,
        regr=None,
        corr=None,
        n_pls=3,
        theta=1.0,
        theta_bounds=(0.0, 100.0),
        optimizer=_DEFAULT_OPTIMIZER,
        **kwargs,
    ) -> None:
        super().__init__(eliminate_duplicates=True, **kwargs)
        # Constant trend by default: it is the canonical KPLS choice (matches the SMT reference)
        # and more accurate in high dimensions, where a linear trend's d+1 extra GLS parameters
        # tend to overfit. Pass regr=LinearRegression() to override.
        self.regr = regr if regr is not None else ConstantRegression()
        self.corr = corr if corr is not None else Gaussian()
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
        mX, sX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
        mY, sY = np.mean(y, axis=0), np.std(y, axis=0, ddof=1)
        sX = np.where(sX == 0.0, 1.0, sX)
        sY = np.where(sY == 0.0, 1.0, sY)
        nX, nY = (X - mX) / sX, (y - mY) / sY
        pls = PLSRegression(n_components=h, scale=False).fit(nX, nY)
        return np.square(pls.x_rotations_)  # (d, h)

    def _kpls_engine(self, X, y):
        """Build the ``Dace`` engine with a PLS-reduced kernel and a length-``h`` theta."""
        n, d = X.shape
        # PLS needs 1 <= n_components <= min(n_samples, n_features); clamp the request into range.
        h = max(1, min(self.n_pls, d, n))
        w2 = self._pls_weights(X, y, h)
        kernel = KPLSKernel(self.corr, w2)

        lo, hi = self.theta_bounds
        theta = np.full(h, self.theta)
        theta_bounds = (np.full(h, lo), np.full(h, hi))
        # self.optimizer is already a concrete optimizer (or None to freeze), resolved in __init__.
        return Dace(regr=self.regr, corr=kernel, theta=theta, theta_bounds=theta_bounds, optimizer=self.optimizer)

    def _fit(self, X, y, optimize=True, **kwargs):
        self.model = self._kpls_engine(X, y)
        self.model.fit(X, y, optimize=optimize)

    def _refit(self, X, y, optimize=True):
        # incremental warm-started re-fit via the Dace engine at the fixed PLS subspace; the generic
        # Model.refit handles out-of-sample scoring and record. optimize warm-starts / freezes theta.
        self.model.refit(X, y, optimize=optimize)

    def _predict(self, X, var=False, grad=False):
        return self.model.predict(X, var=var, grad=grad)
