"""Deep-kernel Gaussian process: a neural feature map with a GP in the learned feature space."""

import numpy as np
from sklearn.neural_network import MLPRegressor  # type: ignore[import-untyped]

from pysurrogate.core.model import Model
from pysurrogate.core.prediction import Prediction
from pysurrogate.core.transformation import Standardization
from pysurrogate.dace import ConstantRegression, Dace, Gaussian
from pysurrogate.util.misc import at_least2d

# smooth activations and their derivatives expressed in terms of the *post*-activation value ``a``
# (so the feature Jacobian needs only the activations it already computed). ``tanh`` is the default:
# it is smooth everywhere, so the analytic input gradient matches finite differences.
_ACT = {
    "tanh": (np.tanh, lambda a: 1.0 - a**2),
    "logistic": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a: a * (1.0 - a)),
    "relu": (lambda z: np.maximum(z, 0.0), lambda a: (a > 0).astype(float)),
    "identity": (lambda z: z, lambda a: np.ones_like(a)),
}

# sentinel: "optimizer not specified" -> let the Dace engine pick its default search (distinct from
# optimizer=None, which Dace reads as "freeze the length-scale -- no search").
_DEFAULT_SEARCH = object()


class DeepKernelGP(Model):
    """Deep-kernel Gaussian process: a neural feature map ``phi(x)`` with a GP in feature space.

    A small MLP is trained to regress ``y``; its last hidden layer becomes a learned nonlinear
    feature map ``phi``, and an ordinary Gaussian-process engine (:class:`~pysurrogate.dace.Dace`) is
    fit on ``phi(X)``. This is the nonlinear generalization of the linear feature maps already in the
    zoo -- :class:`~pysurrogate.core.kernel.Mahalanobis` (a rotation) and :class:`KPLS` (a low-rank
    projection) replace their matrix with a neural net -- and, being a GP in feature space, it returns
    a predictive **mean and variance** plus an analytic gradient (chain rule: the NN feature Jacobian
    times the GP's feature-space gradient).

    The NN and the GP are trained **sequentially** (feature extraction, then a GP head), not jointly
    through the GP marginal likelihood -- a pragmatic deep kernel with no autodiff dependency. Inputs
    and outputs are standardized by the model lifecycle, so the MLP trains on well-scaled data and the
    returned variance/gradients are un-scaled automatically. **Single output only** -- the shared
    feature map and shared-variance GP head make a multi-output deep kernel ill-defined; fit one model
    per output.

    Overfitting of the feature map is guarded by **early stopping**: the MLP holds out a
    ``validation_fraction`` of the training data and stops once the held-out score fails to improve for
    ``n_iter_no_change`` epochs (rather than always running to ``max_iter``). ``alpha`` (L2) is a
    second regularizer. Early stopping is on by default and is skipped only when the design is too
    small to spare a validation point.

    Args:
        hidden_layer_sizes: MLP hidden widths; the last one is the feature-space dimension.
        activation: Hidden activation (``tanh`` default -- smooth, so the analytic gradient is exact).
        alpha: L2 regularization for the MLP.
        max_iter: Max MLP training iterations (the ceiling; early stopping usually stops sooner).
        early_stopping: Hold out a validation split and stop when its score plateaus (default ``True``).
        validation_fraction: Fraction of the training data held out for early stopping.
        n_iter_no_change: Epochs of no validation improvement before stopping.
        random_state: Seed for the MLP, incl. its validation split (fixed -> deterministic fit).
        regr: GP regression trend on the features (default :class:`ConstantRegression`).
        theta: Starting length-scale of the feature-space Gaussian kernel.
        theta_bounds: ``(lo, hi)`` length-scale bounds for the GP search, or ``None`` (unbounded).
        noise: The nugget on the GP diagonal. When ``noise_bounds`` is set this is the *start* of the
            learned nugget; otherwise it is fixed. The small default doubles as a collapse floor --
            deep kernels can map distinct points to near-identical features (saturated units), which a
            zero nugget would leave singular.
        noise_bounds: ``(lo, hi)`` to **learn** the nugget jointly with the length-scale (it becomes
            an extra log-space search coordinate, optimized by the same likelihood/gradient), or
            ``None`` to keep it fixed at ``noise``. Learned by default: a deep kernel's imperfect
            feature map induces effective noise, so a fitted nugget smooths appropriately (and the
            ``lo`` floor keeps a collapsed feature map positive-definite). ``optimize=False`` fits at
            the fixed ``noise`` start instead of searching.
        optimizer: The search strategy for the GP head's hyperparameters -- **the same object you
            would pass to** :class:`~pysurrogate.dace.Dace` (``LBFGS``, ``Restart``, ``Adam``,
            ``Boxmin``, ...). Unset uses the Dace default; ``None`` freezes the length-scale.
        theta_prior: ``(mean, lam)`` MAP prior on ``log10(length-scale)`` forwarded to the GP head --
            the same regularizer :class:`~pysurrogate.dace.Dace` takes. ``None`` (default) is pure MLE.

    The GP head's hyperparameters (length-scale, and the nugget when ``noise_bounds`` is set) are
    selected by **maximum likelihood** -- the DACE profile likelihood, exactly as :class:`Kriging` --
    tunable via the same ``optimizer`` / ``theta_prior`` / ``noise_bounds`` you would pass to Dace. The
    MLP feature map, separately, is trained to minimize squared error (with early stopping).
    """

    def __init__(
        self,
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        alpha=1e-4,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=0,
        regr=None,
        theta=1.0,
        theta_bounds=(0.0, 100.0),
        noise=1e-8,
        noise_bounds=(1e-8, 1e-1),
        optimizer=_DEFAULT_SEARCH,
        theta_prior=None,
        **kwargs,
    ) -> None:
        # standardize inputs (the MLP needs scaled features) and outputs; the lifecycle un-scales the
        # returned mean, variance, and gradient through these affine transforms.
        super().__init__(norm_X=Standardization(), norm_y=Standardization(), eliminate_duplicates=True, **kwargs)
        if activation not in _ACT:
            raise ValueError(f"Unknown activation {activation!r}; choose one of {sorted(_ACT)}.")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.regr = regr if regr is not None else ConstantRegression()
        self.theta = theta
        self.theta_bounds = theta_bounds
        self.noise = noise
        self.noise_bounds = noise_bounds
        self.optimizer = optimizer
        self.theta_prior = theta_prior

    def _features(self, X, jac=False):
        """Map inputs to the last-hidden-layer features ``phi(X)``; optionally also ``d phi / d x``.

        Args:
            X: Inputs in the model's standardized space, shape ``(m, d)``.
            jac: Whether to also return the feature Jacobian.

        Returns:
            ``Z`` shape ``(m, F)`` (``F`` = last hidden width). When ``jac`` is set, also ``J`` shape
            ``(m, F, d)`` with ``J[:, f, k] = d Z_f / d x_k``, propagated forward through the layers.
        """
        act, dact = _ACT[self.activation]
        a = np.asarray(X, dtype=float)
        J = np.broadcast_to(np.eye(a.shape[1]), (a.shape[0], a.shape[1], a.shape[1])).copy() if jac else None
        # hidden layers only (coefs_[:-1]); the MLP's final layer is the discarded regression head.
        for W, b in zip(self._coefs[:-1], self._intercepts[:-1]):
            if jac:
                J = np.einsum("mpk,pu->muk", J, W)  # d(a@W)/dx before the activation
            a = act(a @ W + b)
            if jac:
                J = dact(a)[:, :, None] * J  # chain in the elementwise activation derivative
        return (a, J) if jac else a

    def _fit(self, X, y, optimize=True, **kwargs):
        # early stopping needs at least ~2 held-out points to judge a plateau; on a design too small
        # to spare them, fall back to plain training (alpha still regularizes) rather than crash.
        early = self.early_stopping and int(len(X) * self.validation_fraction) >= 2
        nn = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=early,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        # single output only: the shared feature map and the shared-variance GP head make a
        # multi-output deep kernel ill-defined here (the Prediction contract carries one variance per
        # point, and standardized per-output y-scales cannot un-scale a shared variance/gradient).
        # Fit one DeepKernelGP per output instead.
        if y.shape[1] != 1:
            raise ValueError(f"DeepKernelGP supports a single output, got {y.shape[1]}; fit one model per output.")
        nn.fit(X, y[:, 0])
        self.nn_ = nn  # keep the fitted MLP for introspection (validation curve, iterations, ...)
        self._coefs, self._intercepts = nn.coefs_, nn.intercepts_

        # fit the GP head on the learned features: an isotropic Gaussian in feature space (the NN
        # already handles per-input relevance, so one shared length-scale keeps the search cheap). The
        # length-scale + nugget are selected by MLE via the same Dace knobs (optimizer/theta_prior/
        # noise_bounds) the user would pass to a Dace/Kriging; the nugget floor keeps the fit PD.
        search = {} if self.optimizer is _DEFAULT_SEARCH else {"optimizer": self.optimizer}
        # the nugget can only be *learned* while the length-scale is searched; when the search is
        # frozen (optimizer=None, or optimize=False for model-selection screening) fall back to the
        # fixed `noise` floor rather than asking Dace to learn a nugget it will not search.
        searching = optimize and self.optimizer is not None
        noise_bounds = self.noise_bounds if searching else None
        self.gp = Dace(
            regr=self.regr,
            corr=Gaussian(),
            theta=self.theta,
            theta_bounds=self.theta_bounds,
            noise=self.noise,
            noise_bounds=noise_bounds,
            theta_prior=self.theta_prior,
            **search,
        )
        self.gp.fit(self._features(X), y, optimize=optimize)

    def _refit(self, X, y, optimize=True):
        # freeze the learned feature map (and the fitted input/output standardization): only the cheap
        # GP head absorbs the new points, warm-started through the Dace engine. The expensive NN
        # training is NOT repeated -- amortizing the representation is what makes refit fast, so an
        # adaptive/Bayesian-optimization loop can add points without retraining the network each round.
        Xr, yr = at_least2d(X, expand="r"), at_least2d(y, expand="c")
        # run the new points through the SAME preprocessing the fit path uses -- active-dimension
        # selection, duplicate/nan filtering, and the fitted (frozen) standardization -- so the frozen
        # feature map sees inputs in exactly the space it was trained on.
        Xp, yp = self.preprocess(Xr, yr)
        self.gp.refit(self._features(Xp), yp, optimize=optimize)
        self._X = np.vstack([self._X, Xr])
        self._y = np.vstack([self._y, yr])

    def _predict(self, X, var=False, grad=False):
        if not grad:
            p = self.gp.predict(self._features(X), var=var)
            return Prediction(y=p.y, var=p.var)
        Z, J = self._features(X, jac=True)
        p = self.gp.predict(Z, var=var, grad=True)
        # chain rule (single output): d y / d x = (d y / d Z) . (d Z / d x), with p.grad shape (m, F).
        g = np.einsum("mf,mfk->mk", p.grad, J)
        return Prediction(y=p.y, var=p.var, grad=g)
