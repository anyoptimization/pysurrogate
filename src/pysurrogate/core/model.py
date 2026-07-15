"""Base ``Model`` class: the fit/predict lifecycle with pre- and post-processing."""

import numpy as np

from pysurrogate.core.prediction import Prediction, predictions_frame
from pysurrogate.core.transformation import NoNormalization
from pysurrogate.util.misc import at_least2d, is_duplicate


class Model:
    """Backend-agnostic surrogate with a shared fit/predict lifecycle.

    A concrete surrogate subclasses ``Model`` and implements the hooks ``_fit`` and
    ``_predict`` (and optionally ``_preprocess`` / ``_postprocess``). The
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
        **kwargs,
    ):
        # unknown keyword arguments are typos (e.g. `norm_x=`), not configuration -- swallowing
        # them silently hid misconfigured models, so reject anything no subclass consumed.
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unexpected keyword argument(s): {sorted(kwargs)}")

        self.norm_X = norm_X if norm_X is not None else NoNormalization()
        self.norm_y = norm_y if norm_y is not None else NoNormalization()

        self.eliminate_duplicates = eliminate_duplicates
        self.eliminate_duplicates_eps = eliminate_duplicates_eps

        self.active_dims = active_dims
        self.filter_nan_and_inf = filter_nan_and_inf

        self.model = None
        self.X, self.y = None, None
        self._X, self._y = None, None
        self.success = None

        self.raise_exception_while_fitting = raise_exception_while_fitting
        self.raise_exception_while_prediction = raise_exception_while_prediction
        self.exception = None

        self.has_been_fitted = False

        # prequential (out-of-sample) validation log accumulated across refit() calls -- the tidy
        # predictions DataFrame (with an ``epoch`` column), exposed via history(). epoch counts
        # refit calls.
        self._validation = None
        self._epoch = 0

        # scalar multiplier applied to the predictive variance, set by calibrate(apply=True). 1.0 is
        # the identity (uncalibrated); a fresh fit resets it.
        self._calibration = 1.0

    def preprocess(self, X, y, **kwargs):
        """Run the shared input pipeline: active-dims, duplicate/nan filtering, normalization.

        Args:
            X: Training inputs, shape ``(n, d)``.
            y: Training targets, shape ``(n, q)``.
            **kwargs: Forwarded to the backend's ``_preprocess`` hook.

        Returns:
            The prepared ``(X, y)`` in the normalized space the backend fits in.
        """
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
        """Carry a backend prediction back to original units through the affine norm scales.

        Args:
            pred: The backend's :class:`~pysurrogate.core.prediction.Prediction` in normalized space.

        Returns:
            The un-normalized prediction, after the backend's ``_postprocess`` hook.
        """
        y = self.norm_y.backward(pred.y)
        var, grad, var_grad = pred.var, pred.grad, pred.var_grad

        # var/grad/var_grad come back in the (norm_X, norm_y) space the backend was handed;
        # carry them to original units via the affine chain rule -- output scale s_y, input
        # scale s_X: var ~ s_y**2, grad ~ s_y / s_X, var_grad ~ s_y**2 / s_X. Skipped entirely
        # when no extras were requested, so a plain mean prediction is untouched, and identity
        # norms make every factor 1.
        if var is not None or grad is not None or var_grad is not None:
            s_y = self.norm_y.scale()
            # the calibration scale (calibrate()) multiplies the variance and its gradient -- 1.0
            # until calibrated, so an uncalibrated model is untouched. The mean gradient is unscaled.
            if var is not None:
                var = var * s_y**2 * self._calibration
            # the input scale is needed only for the gradients; fetching it lazily keeps a
            # var-only predict working under a non-affine norm_X (e.g. Plog), whose scale() raises.
            if grad is not None or var_grad is not None:
                s_X = self.norm_X.scale()
                if grad is not None:
                    grad = grad * s_y / s_X
                if var_grad is not None:
                    var_grad = var_grad * s_y**2 / s_X * self._calibration

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
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}.")
        self._X, self._y = X, y

        # a fresh fit invalidates any prior variance calibration -- reset to the identity.
        self._calibration = 1.0
        # drop any statistics a data-fitted normalization estimated on a previous fit, so this fit
        # (e.g. the refit lifecycle on grown data) re-estimates from the current data instead of
        # reusing stale train statistics. No-op for the identity/user-fixed transforms.
        self.norm_X.reset()
        self.norm_y.reset()
        self.X, self.y = self.preprocess(X, y)

        try:
            self._fit(self.X, self.y, optimize=optimize, **kwargs)
            self.success = True
            self.has_been_fitted = True  # only a successful fit counts as fitted
        except Exception as ex:
            self.success = False
            self.exception = ex
            if self.raise_exception_while_fitting:
                raise ex

        return self

    def refit(self, X, y, optimize=True, metrics=None):
        """Score the new points out-of-sample, then append them and re-fit -- ``validate`` + absorb.

        The new points are first predicted by the **current** model -- which has not seen them -- and
        that out-of-sample prediction is scored and **returned** as a multi-metric ``{metric: value}``
        dict (exactly what :meth:`validate` returns). ``refit`` additionally logs the full prediction
        (see :meth:`history`) and then absorbs the points. Prequential validation: scoring each batch
        on the model-before-it-saw-them is honest held-out generalization with no leakage.

        The absorb step delegates to :meth:`_refit`; ``optimize`` is forwarded. Backends with a warm
        start (e.g. Kriging) override :meth:`_refit` for an incremental update; the base re-fits fresh.

        Args:
            X: The new input points to add -- **only the additions, not the full set**.
            y: The targets for the new points.
            optimize: Forwarded to the re-fit (``True`` tunes hyperparameters, ``False`` the cheap
                fixed-hyperparameter fit).
            metrics: Restrict the returned score to these metric names (default: all computable ones),
                matching :meth:`validate`.

        Returns:
            The out-of-sample multi-metric score of the new points -- the same ``{metric: value}``
            :meth:`validate` would return on them *before* the model saw them.

        Raises:
            RuntimeError: If called before a successful :meth:`fit`.
        """
        if not self.has_been_fitted or self._X is None:
            raise RuntimeError("refit() requires a prior fit(); call fit() first.")
        out_of_sample = self.predict(X, var=True)  # the OLD model scores the unseen points
        self._record(X, y, out_of_sample)  # keep the full prediction for history()
        score = self._score(at_least2d(y, expand="c"), out_of_sample, metrics)
        self._refit(X, y, optimize=optimize)
        return score

    def _refit(self, X, y, optimize=True):
        """Re-fit hook: absorb the new points and update the fitted model.

        The base stacks the new ``(X, y)`` onto the data the model was last fit on and re-fits from
        scratch via :meth:`fit`. Backends with an incremental / warm-started update (e.g. Kriging via
        the Dace engine) override just this step to append cheaply; the generic :meth:`refit`
        (out-of-sample scoring + record) stays shared.
        """
        X = np.vstack([self._X, at_least2d(X, expand="r")])
        y = np.vstack([self._y, at_least2d(y, expand="c")])
        self.fit(X, y, optimize=optimize)

    def validate(self, X, y, metrics=None):
        """Score points against the current model across the metric registry -- a multi-metric score.

        Predicts ``X`` and evaluates it against ``y`` over every applicable metric (calibration
        metrics included when the model has a predictive variance), returning a ``{metric: value}``
        dict -- so you pick the metric *after* the fact instead of committing to one up front. When
        ``X`` is data the model has not seen -- e.g. the new points in :meth:`refit` -- this is a
        genuine out-of-sample (prequential) score. Generic: just ``predict`` + the metrics registry.

        Args:
            X: Points to score, shape ``(m, d)``.
            y: Their targets, shape ``(m,)`` or ``(m, q)``.
            metrics: Restrict to these metric names (default: all computable ones).

        Returns:
            ``{metric_name: value}`` over the registry metrics -- e.g. ``score["rmse"]``,
            ``score["nlpd"]``.

        Raises:
            RuntimeError: If called before a successful :meth:`fit`.
        """
        if not self.has_been_fitted:
            raise RuntimeError("validate() requires a fitted model; call fit() first.")
        pred = self.predict(X, var=True)
        return self._score(at_least2d(y, expand="c"), pred, metrics)

    @staticmethod
    def _score(y, pred, metrics=None):
        """Flatten :func:`~pysurrogate.selection.metrics.evaluate` into a ``{metric: value}`` score."""
        from pysurrogate.selection.metrics import evaluate

        families = evaluate(y, pred.y, sigma=pred.sigma, names=metrics)
        return {name: value for family in families.values() for name, value in family.items()}

    def calibrate(self, X, y, apply=True):
        """Fit the predictive-variance scale on held-out ``(X, y)`` -- the sibling of :meth:`validate`.

        Predicts ``X`` with the current model and fits the single scalar ``s = mean[(y - yhat)^2 / var]``
        that makes the predictive variance match the observed errors on the held-out set. ``s > 1``
        means the model was overconfident (errors larger than ``var`` implies), ``s < 1``
        underconfident, ``1`` calibrated. Only the *variance* is rescaled; the mean is untouched, so
        accuracy metrics do not change. Like :meth:`validate`, it takes an explicit held-out set --
        it scores the *current* model, so there is no cross-validation or re-fitting.

        Args:
            X: Held-out points, shape ``(m, d)``.
            y: Their targets, shape ``(m,)`` or ``(m, q)``.
            apply: ``True`` (default) *keeps* the scale -- future ``predict(var=True)`` multiplies the
                variance (and its gradient) by it; ``False`` just returns it without changing the model.

        Returns:
            The fitted variance scale ``s`` for this held-out set (relative to the current predictions).

        Raises:
            RuntimeError: If called before a fit.
            ValueError: If the model has no predictive variance to scale.
        """
        if not self.has_been_fitted:
            raise RuntimeError("calibrate() requires a fitted model; call fit() first.")
        pred = self.predict(X, var=True)
        if pred.var is None:
            raise ValueError("model has no predictive variance to calibrate.")
        resid2 = np.square(at_least2d(y, expand="c") - pred.y)
        s = float(np.mean(resid2 / np.maximum(pred.var, 1e-300)))
        if apply:
            # accumulate: pred.var already carries the current scale, so multiplying keeps calibrate()
            # idempotent (a second call on a calibrated model returns ~1 and leaves the scale put).
            self._calibration *= s
        return s

    def history(self):
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
        # promote first: a 1-D point of shape (d,) is ONE row, not d rows -- and active_dims must
        # slice columns of a 2-D array (slicing a 1-D point would pick coordinates as rows).
        X = at_least2d(X, expand="r")
        m = len(X)

        if not self.success:
            # this is a predict() call, so honor the prediction toggle (not the fitting one).
            # success is None -> fit() was never called; False -> the fit itself failed.
            if self.raise_exception_while_prediction:
                if self.success is None:
                    raise RuntimeError("model has not been fitted; call fit() first.")
                raise RuntimeError("There was an error while fitting the model.") from self.exception
            return Prediction(y=np.full((m, q), np.nan))

        Xq = X[:, self.active_dims] if self.active_dims is not None else X
        Xq = self.norm_X.forward(Xq)

        try:
            pred = self._predict(Xq, var=var, grad=grad)
            pred = self.postprocess(pred)
        except Exception as e:
            if self.raise_exception_while_prediction:
                raise e
            # the same NaN sentinel as the failed-fit path above -- one failure convention
            pred = Prediction(y=np.full((m, q), np.nan))

        return pred

    def _preprocess(self, X, y, **kwargs):
        return X, y

    def _postprocess(self, pred):
        return pred

    def _fit(self, X, y, **kwargs):
        pass

    def _predict(self, X, var=False, grad=False):
        raise NotImplementedError
