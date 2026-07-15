"""Benchmark surrogate models and select the best: cross-validation, function sweeps, selection."""

import copy
import logging

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pysurrogate.core.model import Model
from pysurrogate.core.partitioning import default_partitioning
from pysurrogate.core.prediction import predictions_frame
from pysurrogate.core.sampling import LHS, Random, Sampling
from pysurrogate.selection.factory import as_named
from pysurrogate.selection.metrics import (
    POINT_METRICS,
    PROBABILISTIC,
    calc_metric,
    get_metric,
    metric_sort_key,
)
from pysurrogate.util.logging import get_logger
from pysurrogate.util.misc import at_least2d

log = get_logger("selection")

# ---------------------------------------------------------------------------------------------
# FunctionBenchmark -- sample a known function, fit models, emit a tidy predictions DataFrame
# ---------------------------------------------------------------------------------------------


class FunctionBenchmark:
    """Benchmark surrogate models on a known function by sampling train/valid/test partitions.

    Each role draws its own points from the box domain via its own :class:`Sampling`, the function
    labels them, and every model is fit on ``train`` and predicted on every role. One row per
    predicted point is recorded -- ``rep``, ``model``, ``role``, ``i`` (point index), ``output``,
    ``y_true``, ``y`` (prediction), ``var`` (predictive variance), ``sigma`` (its square root), and
    the input coordinates ``x0..xd`` -- so the returned DataFrame is the durable source of truth:
    any metric is a groupby over it (see :func:`score`), and a failing prediction is reproducible
    from its stored inputs.

    Models are fit on ``train`` only; ``valid`` is an extra held-out partition that is *scored*
    (not fed to the model), so per-role diagnostics like the generalization gap (train vs test) or
    calibration drift (valid vs test) fall out of a groupby. With ``replications > 1`` every role
    is re-drawn under a new seed, giving the ``rep`` axis honest independent test sets.

    Args:
        f: The function under test; maps ``X`` of shape ``(m, d)`` to ``(m,)`` (or ``(m, q)``).
        xl: Lower bounds of the box domain, shape ``(d,)``.
        xu: Upper bounds of the box domain, shape ``(d,)``.
        models: A ``{name: model}`` dict or a list of models (named via the factory).
        train: :class:`Sampling` for the training design. Defaults to ``Sampling(50, LHS())``.
        valid: :class:`Sampling` for the validation partition, or ``None`` to omit it.
        test: :class:`Sampling` for the test partition. Defaults to ``Sampling(2000, Random())``.
        replications: Number of independent re-draws of all partitions (the ``rep`` axis).
        random_state: Base RNG seed; replication ``r`` uses ``random_state + r``.
        noise: Std-dev of Gaussian noise added to the *training* labels only (the held-out
            roles keep the clean ground truth). ``0.0`` (default) trains on the exact function.
        raise_exception: If ``True``, a model that fails to fit aborts the run; if ``False``
            (default) the failure is counted in :attr:`failures` and that model is skipped for
            the offending replication.
    """

    def __init__(
        self,
        f,
        xl,
        xu,
        models,
        train=None,
        valid=None,
        test=None,
        replications=1,
        random_state=0,
        noise=0.0,
        raise_exception=False,
    ):
        self.f = f
        self.xl = np.asarray(xl, dtype=float)
        self.xu = np.asarray(xu, dtype=float)
        self.models = as_named(models)
        self.train = train if train is not None else Sampling(50, LHS())
        self.valid = valid
        self.test = test if test is not None else Sampling(2000, Random())
        self.replications = replications
        self.random_state = random_state
        self.noise = noise
        self.raise_exception = raise_exception
        self.failures = {name: 0 for name in self.models}

    def run(self) -> pd.DataFrame:
        """Fit every model on every replication and return the tidy predictions DataFrame."""
        bounds = (self.xl, self.xu)
        self.failures = {name: 0 for name in self.models}
        blocks = []
        for rep in range(self.replications):
            rng = np.random.default_rng(self.random_state + rep)
            # draw each role's design, then label with the function (train first so its RNG draw
            # is stable regardless of whether valid is present)
            data = {"train": self.train.sample(bounds, rng)}
            if self.valid is not None:
                data["valid"] = self.valid.sample(bounds, rng)
            data["test"] = self.test.sample(bounds, rng)
            data = {role: (X, np.asarray(self.f(X))) for role, X in data.items()}

            # noisy observations, clean ground truth: perturb only the labels the model trains on
            Xtr, ytr = data["train"]
            ytr_fit = ytr + rng.normal(0.0, self.noise, size=ytr.shape) if self.noise > 0 else ytr

            for name, proto in self.models.items():
                model = copy.deepcopy(proto)
                # one guard around fit AND predict: a model that fits but cannot predict is just as
                # failed for this replication -- count it and skip, contributing no partial rows.
                try:
                    model.fit(Xtr, ytr_fit)
                    rows = [
                        predictions_frame(X, y, model.predict(X, var=True), rep=rep, model=name, role=role)
                        for role, (X, y) in data.items()
                    ]
                except Exception:
                    if self.raise_exception:
                        raise
                    self.failures[name] += 1
                    continue
                blocks.extend(rows)
        if not blocks:
            # every model failed to fit on every replication -> nothing to concatenate. Raise a
            # clear message instead of pandas' opaque "No objects to concatenate" from concat([]).
            raise RuntimeError("no predictions produced: every model failed to fit on every replication.")
        return pd.concat(blocks, ignore_index=True)


def score(df, metrics, by=("model", "role")):
    """Compute registry metrics over a predictions DataFrame, grouped.

    Each group's rows supply ``y_true``, ``y`` and (for probabilistic metrics) ``sigma`` to
    :func:`~pysurrogate.selection.metrics.calc_metric`. Probabilistic metrics on a model without
    uncertainty (all-NaN ``sigma``) yield ``NaN``. Grouping by ``("model", "role")`` gives the
    common views at once -- per-model test scores and the train/valid/test diagnostics.

    Args:
        df: A predictions DataFrame from :meth:`FunctionBenchmark.run` or
            :meth:`~pysurrogate.core.model.Model.records`.
        metrics: Metric names to compute (point and/or probabilistic).
        by: Columns to group by before scoring, e.g. ``["epoch"]`` for a prequential record.
            ``None`` (or an empty list) computes one row over the whole DataFrame.

    Returns:
        A DataFrame with one column per metric: one row per group, or a single ``"overall"`` row
        when ``by`` is ``None``.
    """

    def _row(g):
        y_true, y_hat, sig = g["y_true"].to_numpy(), g["y"].to_numpy(), g["sigma"].to_numpy()
        return {
            name: calc_metric(name, y_true, y_hat, sigma=sig if get_metric(name).kind == PROBABILISTIC else None)
            for name in metrics
        }

    if not by:  # no grouping -> a single overall row
        return pd.DataFrame([_row(df)], index=["overall"])

    by = list(by)
    rows = [{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **_row(g)} for k, g in df.groupby(by, sort=False)]
    return pd.DataFrame(rows).set_index(by)


# ---------------------------------------------------------------------------------------------
# Benchmark -- cross-validate model prototypes on one fixed data set and rank them
# ---------------------------------------------------------------------------------------------


def _aggregate(values):
    """Summarize a per-fold metric list, ignoring non-finite folds."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, values=arr)
    return dict(
        mean=float(finite.mean()), std=float(finite.std()), min=float(finite.min()), max=float(finite.max()), values=arr
    )


class Benchmark:
    """Cross-validate a collection of model prototypes on one data set and rank them by a metric.

    Each model is deep-copied and fit on every training fold, then scored on the held-out fold
    with the requested metrics; the per-fold scores are aggregated to mean/std/min/max per model.
    """

    def __init__(self, models, metrics=None, partitioning=None, raise_exception=False):
        self.models = as_named(models)
        self.metrics = list(POINT_METRICS) if metrics is None else list(metrics)
        self.partitioning = partitioning
        self.raise_exception = raise_exception
        self.records = None

    def do(self, X, y, partitions=None, optimize=True):
        """Cross-validate every prototype on ``(X, y)`` and record the per-fold metric scores.

        Args:
            X: Input points, shape ``(n, d)`` (a 1-D input is promoted to one row).
            y: Target values, shape ``(n,)`` (a 2-D column is flattened).
            partitions: Explicit train/test splits to use; ``None`` uses the configured
                partitioning (default: 3-fold cross-validation).
            optimize: Forwarded to every model's ``fit`` (``False`` skips hyperparameter search).

        Returns:
            ``self``, with :attr:`records` populated -- so ``do(...)`` chains into
            :meth:`results` / :meth:`frame`.

        Raises:
            ValueError: If ``X`` and ``y`` disagree in length.
        """
        X = at_least2d(np.asarray(X, dtype=float), expand="r")
        y = at_least2d(np.asarray(y, dtype=float), expand="c")[:, 0]
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} vs {len(y)}.")

        if partitions is None:
            # 3-fold (not the canonical 5) on purpose: model-SELECTION screens many candidates, so
            # the cheaper, lower-variance-budget 3-fold is the deliberate speed trade-off here.
            partitioning = self.partitioning or default_partitioning(k_folds=3, seed=1)
            partitions = partitioning.do(len(X))

        # probabilistic metrics (nlpd, crps, calibration) need the predictive sigma, so ask the
        # model for the variance whenever any requested metric is probabilistic -- point-only metric
        # sets keep the cheaper mean-only predict.
        need_sigma = any(get_metric(m).kind == PROBABILISTIC for m in self.metrics)

        records = {}
        for name, proto in self.models.items():
            per_metric: dict = {m: [] for m in self.metrics}
            n_success = 0
            fitted = None  # last successfully fitted fold model (used by AutoModel(refit_best=False))
            for split in partitions:
                trn, tst = split.train, split.test
                try:
                    model = copy.deepcopy(proto)
                    model.fit(X[trn], y[trn], optimize=optimize)
                    pred = model.predict(X[tst], var=need_sigma)
                    y_hat = pred.y[:, 0]
                    fitted = model
                    # a backend with no predictive variance (SVR, mean, ...) returns sigma=None;
                    # probabilistic metrics then score NaN for it (skipped), not crash the model.
                    sigma = pred.sigma[:, 0] if (need_sigma and pred.sigma is not None) else None
                    n_success += 1
                except Exception:
                    if self.raise_exception:
                        raise
                    for m in self.metrics:
                        per_metric[m].append(np.nan)
                    continue
                # score each metric independently: a probabilistic metric on a no-sigma model
                # fails only that metric, leaving the model's point metrics intact.
                for m in self.metrics:
                    try:
                        per_metric[m].append(calc_metric(m, y[tst], y_hat, sigma=sigma))
                    except Exception:
                        if self.raise_exception:
                            raise
                        per_metric[m].append(np.nan)

            records[name] = dict(
                label=name,
                proto=proto,
                fitted=fitted,
                n_runs=len(partitions),
                n_success=n_success,
                success=n_success > 0,
                performance={m: _aggregate(per_metric[m]) for m in self.metrics},
            )

        self.records = records
        return self

    def results(self, sorted_by="mae", only_successful=True):
        """Return the per-model records as a list, sorted by a metric's mean (direction-aware).

        Args:
            sorted_by: Metric name to rank by; its registry direction sets ascending/descending.
            only_successful: Drop models that failed on every fold.

        Returns:
            A list of per-model record dicts, best first. A non-finite mean sorts last.
        """
        if self.records is None:
            raise RuntimeError("call do(X, y) before results().")
        if sorted_by not in self.metrics:
            raise ValueError(f"sorted_by={sorted_by!r} is not a computed metric; available: {self.metrics}")

        rows = [r for r in self.records.values() if (not only_successful or r["success"])]

        def key(record):
            # target- and direction-aware, shared with StudyResult so the two rankers agree
            return metric_sort_key(sorted_by, record["performance"][sorted_by]["mean"])

        return sorted(rows, key=key)

    def frame(self, sorted_by="mae"):
        """Return a pandas DataFrame of per-model metric means, rows sorted by ``sorted_by``."""
        rows = self.results(sorted_by=sorted_by, only_successful=False)
        data = {r["label"]: {m: r["performance"][m]["mean"] for m in self.metrics} for r in rows}
        return pd.DataFrame(data).T


# ---------------------------------------------------------------------------------------------
# AutoModel -- pick the best candidate by a cross-validated Benchmark, refit on all data
# ---------------------------------------------------------------------------------------------


class AutoModel(Model):
    """A surrogate that auto-selects its implementation from a set of candidates.

    Wraps a :class:`Benchmark`: ``fit`` runs the benchmark on the data, ranks the candidates by
    ``sorted_by``, and (by default) refits the winning prototype on the full data set. ``predict``
    then delegates to that chosen model. It is itself a ``Model``, so an ``AutoModel`` can be dropped
    in anywhere a single surrogate is expected.

    ``AutoModel()`` with no arguments works out of the box: it selects over the recommended
    fleet (:func:`~pysurrogate.selection.study.default_models` -- the mean/KNN/IDW/SVR/RBF baselines
    plus the Kriging kernel zoo). If the chosen model must report uncertainty (e.g. to drive an
    acquisition function), pass the uncertainty-only fleet
    :func:`~pysurrogate.selection.study.default_kriging` instead, since the baselines report no
    ``sigma`` and accuracy-ranking tends to pick them.

    Args:
        models: A ``{name: model}`` dict, a list of models, or a pre-built :class:`Benchmark`.
            ``None`` (default) uses :func:`~pysurrogate.selection.study.default_models`.
        sorted_by: Metric to rank by (e.g. ``"rmse"``, ``"mae"``); its registry direction sets
            ascending/descending.
        refit_best: Refit the winner on the full data (``True``) or keep its fold fit (``False``).
        partitioning: Cross-validation scheme; ``None`` uses the :class:`Benchmark` default.
    """

    def __init__(self, models=None, sorted_by="mae", refit_best=True, partitioning=None, **kwargs):
        # forward Model lifecycle options (norm_X/norm_y, active_dims, filtering, ...) to the base
        # so AutoModel is configured like any other Model -- the lifecycle genuinely runs.
        super().__init__(**kwargs)
        if models is None:
            from pysurrogate.selection.study import default_models

            models = default_models()
        if isinstance(models, Benchmark):
            self.benchmark = models
        else:
            # ensure the selection metric is actually computed: the point-metric default omits the
            # probabilistic ones (crps/nlpd/calib), so selecting by one of those needs it added to
            # the benchmark's metric set -- otherwise `sorted_by` has nothing to rank on.
            metrics = list(POINT_METRICS)
            if sorted_by not in metrics:
                metrics.append(sorted_by)
            self.benchmark = Benchmark(models, metrics=metrics, partitioning=partitioning)
        self.sorted_by = sorted_by
        self.refit_best = refit_best
        self.best = None
        self.ranking = None

    def _fit(self, X, y, optimize=True, **kwargs):
        # a genuine Model hook: the public Model.fit has already promoted, normalized, filtered
        # and de-duplicated (X, y), so the benchmark and the winner all see the same prepared data
        # and the Model lifecycle owns pre/postprocess. We run the cross-validated benchmark, pick
        # the winner by `sorted_by`, and store the winner *fitted on the full data* as self.model.
        self.benchmark.do(X, y, optimize=optimize)
        ranking = self.benchmark.results(sorted_by=self.sorted_by, only_successful=True)
        if not ranking:
            raise RuntimeError("No candidate model could be fitted successfully.")

        self.ranking = ranking

        def _mean(entry):
            return entry["performance"][self.sorted_by]["mean"]

        self.best = ranking[0]
        log.info(
            "AutoModel: selected %r by %s=%.6g out of %d candidate(s)",
            self.best["label"],
            self.sorted_by,
            _mean(self.best),
            len(ranking),
        )
        if log.isEnabledFor(logging.DEBUG):
            for rank, entry in enumerate(ranking):
                log.debug("AutoModel ranking #%d: %r %s=%.6g", rank, entry["label"], self.sorted_by, _mean(entry))

        if self.refit_best:
            # refit the winning prototype on the FULL data set (the common case).
            winner = copy.deepcopy(self.best["proto"])
            winner.fit(X, y, optimize=optimize)
        else:
            # keep the winner's fold fit -- only well-defined for a single-partition benchmark,
            # where that one fitted model was trained on (essentially) all the data.
            if self.best["n_runs"] > 1:
                raise RuntimeError("refit_best=False needs a single-partition benchmark (n_runs == 1).")
            winner = self.best["fitted"]
            if winner is None:
                raise RuntimeError("refit_best=False but the winner has no retained fold fit.")
        self.model = winner

    def _predict(self, X, var=False, grad=False):
        # delegate to the chosen winner; Model.predict wraps this with the shared postprocess
        # (un-normalization) and the failed-fit guard, so AutoModel needs no bespoke predict.
        return self.model.predict(X, var=var, grad=grad)

    def refit(self, X, y, optimize=True):
        """Refit the *selected* winner on new data -- it does **not** re-select.

        Delegates to the chosen model's :meth:`~pysurrogate.core.model.Model.refit` (appending the
        new points), reusing whatever warm start that model has (e.g. Kriging keeps its fitted
        length-scale). The selection is *not* revisited: in an iterative/online loop the data is
        adaptively sampled and biased, so re-selecting on it is unreliable -- you commit to the
        winner chosen on the representative initial design and just refit it. To re-select, call
        :meth:`fit` again.

        Like any :meth:`~pysurrogate.core.model.Model.refit`, the winner scores the new points
        out-of-sample first and that :class:`~pysurrogate.core.prediction.Prediction` is returned
        (prequential validation -- collect it against ``y``).

        Args:
            X: The new input points to add (only the additions).
            y: The targets for the new points.
            optimize: Forwarded to the winner's refit (``True`` warm-starts hyperparameters,
                ``False`` keeps them fixed).

        Returns:
            The out-of-sample :class:`~pysurrogate.core.prediction.Prediction` of ``X`` from the
            winner *before* the new points were added.

        Raises:
            Exception: If called before a successful :meth:`fit`.
        """
        if self.model is None:
            raise Exception("refit() requires a prior fit(); call fit() first.")
        return self.model.refit(X, y, optimize=optimize)  # winner records its own prequential log

    def history(self):
        """Return the winner's prequential validation log (see :meth:`Model.history`)."""
        return pd.DataFrame() if self.model is None else self.model.history()

    def statistics(self):
        """Return a ``{model_name: score}`` dict of the ranking metric, best first (empty before fit)."""
        if self.ranking is None:
            return {}
        return {r["label"]: r["performance"][self.sorted_by]["mean"] for r in self.ranking}
