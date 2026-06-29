"""Base ``Model`` class: the fit/predict lifecycle with pre- and post-processing."""

import time

import numpy as np

from pysurrogate.core.prediction import Prediction
from pysurrogate.core.transformation import NoNormalization
from pysurrogate.util.misc import at_least2d, is_duplicate


class Model:
    """Backend-agnostic surrogate with a shared fit/predict lifecycle.

    A concrete surrogate subclasses ``Model`` and implements the hooks ``_fit`` and
    ``_predict`` (and optionally ``_optimize`` / ``_preprocess`` / ``_postprocess``). The
    public ``fit`` and ``predict`` wrap those hooks with the common machinery every backend
    shares: input promotion to 2-D, normalization, nan/inf filtering, duplicate elimination,
    active-dimension selection, exception capture, and un-normalizing the returned
    ``Prediction`` back to original units.
    """

    def __init__(
        self,
        norm_X=None,
        norm_y=None,
        active_dims=None,
        filter_nan_and_inf=True,
        eliminate_duplicates=False,
        eliminate_duplicates_eps=1e-16,
        raise_exception_while_fitting=True,
        raise_exception_while_prediction=True,
        verbose=False,
        **kwargs,
    ):
        self.norm_X = norm_X if norm_X is not None else NoNormalization()
        self.norm_y = norm_y if norm_y is not None else NoNormalization()

        self.eliminate_duplicates = eliminate_duplicates
        self.eliminate_duplicates_eps = eliminate_duplicates_eps

        self.active_dims = active_dims
        self.filter_nan_and_inf = filter_nan_and_inf
        self.verbose = verbose

        self.time = None
        self.model = None
        self.X, self.y = None, None
        self._X, self._y = None, None
        self.success = None
        self.data = {}

        self.raise_exception_while_fitting = raise_exception_while_fitting
        self.raise_exception_while_prediction = raise_exception_while_prediction
        self.exception = None

        self.has_been_fitted = False

    def preprocess(self, X, y, **kwargs):
        if self.active_dims is not None:
            X = X[:, self.active_dims]

        if self.eliminate_duplicates:
            keep = ~is_duplicate(X, eps=self.eliminate_duplicates_eps)
            X, y = X[keep], y[keep]

        if self.filter_nan_and_inf:
            X_I = np.all(~np.isnan(X) & ~np.isinf(X), axis=1)
            y_I = np.all(~np.isnan(y) & ~np.isinf(y), axis=1)
            X, y = X[X_I & y_I], y[X_I & y_I]

        X, y = self.norm_X.forward(X), self.norm_y.forward(y)
        X, y = self._preprocess(X, y, **kwargs)
        return X, y

    def postprocess(self, pred):
        y = self.norm_y.backward(pred.y)
        var, grad, var_grad = pred.var, pred.grad, pred.var_grad

        # var/grad/var_grad come back in the (norm_X, norm_y) space the backend was handed;
        # carry them to original units via the affine chain rule -- output scale s_y, input
        # scale s_X: var ~ s_y**2, grad ~ s_y / s_X, var_grad ~ s_y**2 / s_X. Skipped entirely
        # when no extras were requested, so a plain mean prediction is untouched, and identity
        # norms make every factor 1.
        if var is not None or grad is not None or var_grad is not None:
            s_y, s_X = self.norm_y.scale(), self.norm_X.scale()
            if var is not None:
                var = var * s_y**2
            if grad is not None:
                grad = grad * s_y / s_X
            if var_grad is not None:
                var_grad = var_grad * s_y**2 / s_X

        pred = Prediction(y=y, var=var, grad=grad, var_grad=var_grad)
        return self._postprocess(pred)

    def fit(self, X, y, **kwargs):
        X, y = at_least2d(X, expand="r"), at_least2d(y, expand="c")
        assert len(X) == len(y)
        self._X, self._y = X, y

        self.X, self.y = self.preprocess(X, y)

        start = time.time()
        try:
            self._fit(self.X, self.y, **kwargs)
            self._optimize(**kwargs)
            self.success = True
        except Exception as ex:
            self.success = False
            self.exception = ex
            if self.raise_exception_while_fitting:
                raise ex

        self.has_been_fitted = True
        self.time = time.time() - start
        return self

    def predict(self, X, var=False, grad=False):
        """Predict the mean (and optionally variance/gradient) for ``X``.

        Args:
            X: Query points, shape ``(m, d)``.
            var: Also return the predictive variance (``Prediction.var``; ``Prediction.sigma``
                is its std-dev view).
            grad: Also return the gradient of the mean w.r.t. ``X`` (``Prediction.grad``).
                When ``var`` and ``grad`` are both set, a backend that supports it (Kriging)
                additionally returns ``Prediction.var_grad``.

        Returns:
            A :class:`~pysurrogate.core.prediction.Prediction` whose ``y`` is always set;
            ``var``/``grad``/``var_grad`` are populated only when their flag is requested (and
            the model supports them, else ``None``).

        Note:
            ``var``/``grad``/``var_grad`` are returned in the model's original output space:
            ``postprocess`` un-normalizes them through the affine scales of ``norm_X``/``norm_y``
            (``var`` by ``s_y**2``, ``grad`` by ``s_y / s_X``). A non-affine transform (e.g.
            ``Plog``) has no constant scale and raises when uncertainty/gradients are requested.
        """
        q = self._y.shape[1] if self._y is not None else 1

        if not self.success:
            if self.raise_exception_while_fitting:
                raise Exception("There was an error while fitting the model.")
            return Prediction(y=np.full((len(X), q), np.nan))

        Xq = X[:, self.active_dims] if self.active_dims is not None else X
        Xq = self.norm_X.forward(at_least2d(Xq, expand="r"))

        try:
            pred = self._predict(Xq, var=var, grad=grad)
            pred = self.postprocess(pred)
        except Exception as e:
            if self.raise_exception_while_prediction:
                raise e
            pred = Prediction(y=np.full((len(X), q), np.inf))

        return pred

    def _preprocess(self, X, y, **kwargs):
        return X, y

    def _postprocess(self, pred):
        return pred

    def _fit(self, X, y, **kwargs):
        pass

    def _predict(self, X, var=False, grad=False):
        raise NotImplementedError

    def _optimize(self, **kwargs):
        pass
