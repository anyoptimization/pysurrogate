"""Base ``Model`` class: the fit/predict lifecycle with pre- and post-processing."""

import time

import numpy as np

from pysurrogate.core.prediction import Prediction, predictions_frame
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

        # prequential (out-of-sample) validation log accumulated across refit() calls -- the tidy
        # predictions DataFrame (with an ``epoch`` column), exposed via records(). epoch counts
        # refit calls.
        self._validation = None
        self._epoch = 0

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

    def fit(self, X, y, optimize=True, **kwargs):
        """Fit the model. ``optimize`` toggles hyperparameter tuning.

        Args:
            X: Training inputs, shape ``(n, d)``.
            y: Training targets, shape ``(n,)`` or ``(n, q)``.
            optimize: ``True`` (default) tunes the model's hyperparameters (e.g. the Kriging
                length-scale search); ``False`` does the cheap fit at fixed/default
                hyperparameters -- the same model, no inner search. Models without tunable
                hyperparameters ignore it. This is the lever for cheap model-selection screening
                (rank with ``optimize=False``, refit the winner with ``optimize=True``) and for a
                frozen-hyperparameter refit in an iterative loop.
            **kwargs: Backend-specific fit options, forwarded to ``_fit``.
        """
        X, y = at_least2d(X, expand="r"), at_least2d(y, expand="c")
        assert len(X) == len(y)
        self._X, self._y = X, y

        self.X, self.y = self.preprocess(X, y)

        start = time.time()
        try:
            self._fit(self.X, self.y, optimize=optimize, **kwargs)
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

    def refit(self, X, y, optimize=True):
        """Score the new points out-of-sample, then append them and re-fit.

        The new points are first predicted by the **current** model -- which has not seen them --
        and that out-of-sample :class:`~pysurrogate.core.prediction.Prediction` is **returned**.
        This is prequential validation: in an online/refit loop, scoring each batch on the
        model-before-it-saw-them is honest held-out generalization with no leakage, exactly the
        record to collect. Only after scoring are the points appended and the model re-fit.

        Generic behavior stacks the new ``(X, y)`` onto the data the model was last fit on and
        re-fits; ``optimize`` is forwarded to :meth:`fit`. Backends with a warm start (e.g. Kriging)
        override the re-fit step for an incremental update; the base re-fits fresh.

        Args:
            X: The new input points to add (only the additions, not the full set).
            y: The targets for the new points.
            optimize: Forwarded to :meth:`fit` (``True`` tunes hyperparameters, ``False`` the cheap
                fixed-hyperparameter fit).

        Returns:
            The out-of-sample :class:`~pysurrogate.core.prediction.Prediction` of ``X`` from the
            model *before* the new points were added (collect it against ``y`` to validate).

        Raises:
            Exception: If called before a successful :meth:`fit`.
        """
        if not self.has_been_fitted or self._X is None:
            raise Exception("refit() requires a prior fit(); call fit() first.")
        out_of_sample = self.predict(X, var=True)  # the OLD model scores the unseen points
        self._record(X, y, out_of_sample)
        X = np.vstack([self._X, at_least2d(X, expand="r")])
        y = np.vstack([self._y, at_least2d(y, expand="c")])
        self.fit(X, y, optimize=optimize)
        return out_of_sample

    def records(self):
        """Return the prequential validation log accumulated across :meth:`refit` calls.

        Each :meth:`refit` scores its new points out-of-sample and appends them here, stamped with
        an ``epoch`` (the refit index). The result is the tidy predictions DataFrame -- the same
        schema the benchmark layer emits -- so it feeds :func:`~pysurrogate.selection.score`
        directly, e.g. grouped by ``epoch`` to track generalization over the loop.

        Returns:
            The accumulated predictions DataFrame; an empty DataFrame before the first refit.
        """
        import pandas as pd  # type: ignore[import-untyped]

        return self._validation if self._validation is not None else pd.DataFrame()

    def _record(self, X, y, pred):
        """Append one refit's out-of-sample prediction to the validation log (stamped ``epoch``).

        Builds the tidy predictions block (the schema :func:`~pysurrogate.selection.score` consumes)
        and accumulates it, then advances the epoch counter.

        Args:
            X: The new points that were scored out-of-sample.
            y: Their true targets.
            pred: The model's out-of-sample :class:`~pysurrogate.core.prediction.Prediction`.
        """
        import pandas as pd  # type: ignore[import-untyped]

        block = predictions_frame(X, y, pred, epoch=self._epoch)
        self._validation = (
            block if self._validation is None else pd.concat([self._validation, block], ignore_index=True)
        )
        self._epoch += 1

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
        # row count from the promoted input -- a 1-D point of shape (d,) is one row, not d rows,
        # so both early-return shapes below stay (m, q) regardless of how X was passed
        m = len(at_least2d(X, expand="r"))

        if not self.success:
            # this is a predict() call, so honor the prediction toggle (not the fitting one)
            if self.raise_exception_while_prediction:
                raise Exception("There was an error while fitting the model.")
            return Prediction(y=np.full((m, q), np.nan))

        Xq = X[:, self.active_dims] if self.active_dims is not None else X
        Xq = self.norm_X.forward(at_least2d(Xq, expand="r"))

        try:
            pred = self._predict(Xq, var=var, grad=grad)
            pred = self.postprocess(pred)
        except Exception as e:
            if self.raise_exception_while_prediction:
                raise e
            pred = Prediction(y=np.full((m, q), np.inf))

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
