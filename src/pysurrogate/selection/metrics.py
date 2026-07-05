"""Holistic surrogate-model evaluation metrics (accuracy, fit, ranking, selection, calibration).

A single registry of metrics, each tagged with the data it consumes (point vs. probabilistic),
the question it answers (accuracy/fit/ranking/selection/calibration), and its direction of
improvement -- so ``Benchmark`` and ``AutoModel`` can sort and group results without any
metric-specific code. ``evaluate`` runs the whole suite; ``get_metric``/``calc_metric`` resolve
and compute one by name.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.stats import kendalltau, norm, spearmanr  # type: ignore[import-untyped]
from sklearn.metrics import r2_score  # type: ignore[import-untyped]

# --------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------
# what a metric consumes
POINT = "point"  # (y, y_hat)
PROBABILISTIC = "probabilistic"  # (y, y_hat, sigma)

# which question a metric answers
ACCURACY = "accuracy"  # magnitude of the error
FIT = "fit"  # scale-free goodness of fit
RANKING = "ranking"  # is the ordering of points correct
SELECTION = "selection"  # would optimizing the surrogate pick good points
CALIBRATION = "calibration"  # is the predicted uncertainty trustworthy


@dataclass(frozen=True)
class Metric:
    """A single evaluation metric plus the metadata needed to use it correctly.

    Attributes:
        name: Short registry key (e.g. ``"rmse"``).
        func: Implementation: ``func(y, y_hat)`` for point metrics, ``func(y, y_hat, sigma)`` for
            probabilistic ones.
        greater_is_better: Direction of improvement. Lets ``Benchmark``/``AutoModel`` sort
            without a hand-passed ``ascending`` flag. Ignored when ``target`` is set.
        family: One of ``ACCURACY``/``FIT``/``RANKING``/``SELECTION``/``CALIBRATION``.
        kind: ``POINT`` or ``PROBABILISTIC`` (whether ``sigma`` is required).
        description: One-line human description.
        target: Ideal value for a "closer is better" metric (e.g. ``1.0`` for a calibration ratio).
            When set, ``greater_is_better`` is moot -- better means nearer ``target``.
    """

    name: str
    func: Callable
    greater_is_better: bool
    family: str
    kind: str = POINT
    description: str = ""
    target: Optional[float] = None

    def __call__(self, y, y_hat, sigma=None) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_hat = np.asarray(y_hat, dtype=float).ravel()
        if y.shape != y_hat.shape:
            raise ValueError(f"y and y_hat must share a shape, got {y.shape} vs {y_hat.shape}")
        if self.kind == PROBABILISTIC:
            if sigma is None:
                raise ValueError(f"metric '{self.name}' is probabilistic and requires sigma")
            sigma = np.asarray(sigma, dtype=float).ravel()
            return float(self.func(y, y_hat, sigma))
        return float(self.func(y, y_hat))


# --------------------------------------------------------------
# Accuracy — magnitude of the error (lower is better)
# --------------------------------------------------------------


def mse(y, y_hat):
    """Mean squared error."""
    return np.mean((y - y_hat) ** 2)


def rmse(y, y_hat):
    """Root mean squared error (same units as y)."""
    return np.sqrt(np.mean((y - y_hat) ** 2))


def mae(y, y_hat):
    """Mean absolute error."""
    return np.mean(np.abs(y - y_hat))


def medae(y, y_hat):
    """Median absolute error (robust to outliers)."""
    return np.median(np.abs(y - y_hat))


def max_error(y, y_hat):
    """Largest absolute error (worst case)."""
    return np.max(np.abs(y - y_hat))


def nrmse(y, y_hat):
    """RMSE normalized by the range of y (scale-free, comparable across problems)."""
    rng = np.ptp(y)
    denom = rng if rng > 0 else (np.abs(np.mean(y)) or 1.0)
    return np.sqrt(np.mean((y - y_hat) ** 2)) / denom


# --------------------------------------------------------------
# Fit — scale-free goodness of fit (higher is better)
# --------------------------------------------------------------


def r2(y, y_hat):
    """Coefficient of determination R^2."""
    return r2_score(y, y_hat)


# --------------------------------------------------------------
# Ranking — is the ordering correct (higher is better)
# --------------------------------------------------------------


def spear(y, y_hat):
    """Spearman rank correlation (undefined -> NaN for a constant predictor)."""
    if np.ptp(y) == 0 or np.ptp(y_hat) == 0:
        return np.nan
    return spearmanr(y, y_hat).correlation


def kendall(y, y_hat):
    """Kendall's tau rank correlation (undefined -> NaN for a constant predictor)."""
    if np.ptp(y) == 0 or np.ptp(y_hat) == 0:
        return np.nan
    return kendalltau(y, y_hat).correlation


# --------------------------------------------------------------
# Selection — would optimizing the surrogate pick good points
# Convention: smaller y is better (minimization), matching optimization use.
# --------------------------------------------------------------


def simple_regret(y, y_hat):
    """Return the true gap of the surrogate-best point: ``y[argmin(y_hat)] - min(y)``."""
    return y[int(np.argmin(y_hat))] - np.min(y)


def precision_at_k(y, y_hat, k):
    """Fraction of the true best-k points (lowest y) recovered in the predicted best-k."""
    k = max(1, min(int(k), len(y)))
    true_best = set(np.argsort(y)[:k].tolist())
    pred_best = set(np.argsort(y_hat)[:k].tolist())
    return len(true_best & pred_best) / k


def _prec_at(k):
    return lambda y, y_hat: precision_at_k(y, y_hat, k)


# --------------------------------------------------------------
# Calibration — is the predicted uncertainty trustworthy (need sigma)
# --------------------------------------------------------------


def _safe_sigma(sigma):
    return np.clip(sigma, 1e-12, None)


def nlpd(y, y_hat, sigma):
    """Negative log predictive density under a Gaussian (lower is better).

    The metric that scores the predicted variance, not just the mean: a confident wrong
    prediction (small sigma, large residual) is penalized heavily, honest uncertainty is not.
    """
    s = _safe_sigma(sigma)
    return np.mean(0.5 * np.log(2 * np.pi) + np.log(s) + 0.5 * ((y - y_hat) / s) ** 2)


def crps_gaussian(y, y_hat, sigma):
    """Continuous ranked probability score for a Gaussian predictive (lower is better)."""
    s = _safe_sigma(sigma)
    z = (y - y_hat) / s
    return np.mean(s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


def coverage(y, y_hat, sigma, level=0.9):
    """Empirical fraction of points inside the central ``level`` predictive interval."""
    s = _safe_sigma(sigma)
    half = norm.ppf(0.5 + level / 2) * s
    return np.mean(np.abs(y - y_hat) <= half)


def calibration_error(y, y_hat, sigma, level=0.9):
    """Absolute gap between empirical and nominal interval coverage (lower is better)."""
    return np.abs(coverage(y, y_hat, sigma, level) - level)


def calib(y, y_hat, sigma):
    """Mean standardized squared error ``mean(((y - y_hat) / sigma)^2)`` (ideal 1.0).

    The ratio of actual squared error to the variance the model reports. ``1`` is honest; ``> 1``
    is over-confident (errors larger than ``sigma`` implies); ``< 1`` under-confident. Unlike
    interval coverage this is unbounded above, so it flags severe over-confidence that a clipped
    coverage statistic would saturate on.
    """
    s = _safe_sigma(sigma)
    return np.mean(((y - y_hat) / s) ** 2)


# --------------------------------------------------------------
# Registry
# --------------------------------------------------------------

_REGISTRY: dict = {}


def register(metric: Metric) -> Metric:
    """Add a metric to the global registry (keyed by ``metric.name``)."""
    _REGISTRY[metric.name] = metric
    return metric


for _m in [
    # accuracy
    Metric("mse", mse, False, ACCURACY, description="mean squared error"),
    Metric("rmse", rmse, False, ACCURACY, description="root mean squared error"),
    Metric("mae", mae, False, ACCURACY, description="mean absolute error"),
    Metric("medae", medae, False, ACCURACY, description="median absolute error (robust)"),
    Metric("max_error", max_error, False, ACCURACY, description="worst-case absolute error"),
    Metric("nrmse", nrmse, False, ACCURACY, description="range-normalized RMSE"),
    # fit
    Metric("r2", r2, True, FIT, description="coefficient of determination"),
    # ranking
    Metric("spear", spear, True, RANKING, description="Spearman rank correlation"),
    Metric("kendall", kendall, True, RANKING, description="Kendall tau rank correlation"),
    # selection (minimization convention)
    Metric("regret", simple_regret, False, SELECTION, description="true gap of surrogate-best point"),
    Metric("prec@5", _prec_at(5), True, SELECTION, description="precision of the best-5 set"),
    Metric("prec@10", _prec_at(10), True, SELECTION, description="precision of the best-10 set"),
    # calibration (need sigma)
    Metric("nlpd", nlpd, False, CALIBRATION, PROBABILISTIC, "negative log predictive density"),
    Metric("crps", crps_gaussian, False, CALIBRATION, PROBABILISTIC, "Gaussian CRPS"),
    Metric("cal_err", calibration_error, False, CALIBRATION, PROBABILISTIC, "|empirical - nominal| coverage"),
    Metric("calib", calib, False, CALIBRATION, PROBABILISTIC, "mean standardized squared error (ideal 1.0)", 1.0),
]:
    register(_m)


# --------------------------------------------------------------
# Public API
# --------------------------------------------------------------

# Point metrics safe to compute from (y, y_hat) alone — the holistic benchmark default.
POINT_METRICS = [name for name, m in _REGISTRY.items() if m.kind == POINT]


def get_metric(metric) -> Metric:
    """Resolve a metric name (or pass through a ``Metric``) to a ``Metric`` object."""
    if isinstance(metric, Metric):
        return metric
    if metric not in _REGISTRY:
        raise ValueError(f"Metric '{metric}' is not known. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[metric]


def greater_is_better(metric) -> bool:
    """Whether a larger value of ``metric`` means a better model."""
    return get_metric(metric).greater_is_better


def metric_sort_key(metric, value) -> float:
    """Ranking sort key for ``metric`` where smaller is always better (best-first ascending).

    The single source of truth for ranking direction, used by both ``Benchmark`` and
    ``StudyResult`` so they agree. ``target``-anchored metrics (e.g. a calibration ratio whose
    ideal is ``1.0``) rank by distance to the target; otherwise the sign flips for
    greater-is-better metrics. A missing or non-finite value sorts last.

    Args:
        metric: Metric name or :class:`Metric`.
        value: The metric value to turn into a sort key.

    Returns:
        A float key; sorting ascending puts the best model first.
    """
    m = get_metric(metric)
    if value is None or not np.isfinite(value):
        return np.inf
    if m.target is not None:
        return abs(value - m.target)
    return -value if m.greater_is_better else value


def calc_metric(metric, y, y_hat, sigma=None) -> float:
    """Compute a single metric by name."""
    return get_metric(metric)(y, y_hat, sigma=sigma)


def metric_names(family: Optional[str] = None, kind: Optional[str] = None) -> list:
    """List registered metric names, optionally filtered by family and/or kind."""
    return [
        name
        for name, m in _REGISTRY.items()
        if (family is None or m.family == family) and (kind is None or m.kind == kind)
    ]


def evaluate(y, y_hat, sigma=None, names=None) -> dict:
    """Evaluate a model holistically, grouped by metric family.

    Args:
        y: Ground-truth target values.
        y_hat: Predicted mean values.
        sigma: Predictive standard deviations. Calibration metrics are only computed when given.
        names: Restrict to these metric names. Defaults to all metrics whose inputs are available
            (point metrics always; calibration metrics only when ``sigma`` is provided).

    Returns:
        ``{family: {metric_name: value}}`` for every evaluated metric.
    """

    # one point of truth for "is this metric computable here": a probabilistic metric needs sigma.
    # The same predicate drives the default name set AND filters an explicit `names` list, so
    # passing a probabilistic metric without sigma silently drops it rather than raising
    # inconsistently with the default path.
    def _computable(m):
        return m.kind == POINT or sigma is not None

    if names is None:
        names = [name for name, m in _REGISTRY.items() if _computable(m)]

    out: dict = {}
    for name in names:
        m = get_metric(name)
        if not _computable(m):
            continue
        out.setdefault(m.family, {})[name] = m(y, y_hat, sigma=sigma)
    return out
