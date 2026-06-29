"""Metrics for scoring surrogate predictions: plain error, uncertainty-aware, and ranking."""

import numpy as np
from scipy.stats import kendalltau, spearmanr  # type: ignore[import-untyped]


def _ravel(*arrays):
    """Flatten each input to a 1-D float array (so ``(n, 1)`` and ``(n,)`` are interchangeable)."""
    return [np.asarray(a, dtype=float).ravel() for a in arrays]


# --- plain accuracy metrics --------------------------------------------------------------


def mse(y_true, y_pred):
    """Mean squared error between targets and predictions."""
    yt, yp = _ravel(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    yt, yp = _ravel(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def medae(y_true, y_pred):
    """Median absolute error (robust to outliers)."""
    yt, yp = _ravel(y_true, y_pred)
    return float(np.median(np.abs(yt - yp)))


def max_error(y_true, y_pred):
    """Largest absolute residual."""
    yt, yp = _ravel(y_true, y_pred)
    return float(np.max(np.abs(yt - yp)))


def r2(y_true, y_pred):
    """Coefficient of determination ``R^2`` (1.0 is perfect; can go negative)."""
    yt, yp = _ravel(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# --- uncertainty-aware metrics (use the predicted variance) ------------------------------


def nlpd(y_true, y_pred, var, eps=1e-12):
    """Negative log predictive density under a Gaussian predictive distribution (lower is better).

    Rewards a model whose predictive variance matches its actual error: a confident wrong
    prediction (small ``var``, large residual) is penalized heavily, while honest uncertainty
    is not. This is the metric that scores the ``var`` output, not just the mean.

    Args:
        y_true: Observed targets.
        y_pred: Predicted means (``Prediction.y``).
        var: Predicted variances (``Prediction.var``), per point.
        eps: Floor applied to ``var`` so a zero variance does not produce a non-finite score.

    Returns:
        The mean per-point negative log density ``0.5*log(2*pi*var) + 0.5*(y-mu)**2/var``.
    """
    yt, yp, v = _ravel(y_true, y_pred, var)
    v = np.maximum(v, eps)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * v) + 0.5 * (yt - yp) ** 2 / v))


def msll(y_true, y_pred, var, y_train, eps=1e-12):
    """Mean standardized log loss: NLPD relative to a trivial Gaussian baseline (lower is better).

    Subtracts the NLPD of a baseline that predicts the training mean with the training variance,
    so the score is calibrated against doing nothing: negative means the model beats the naive
    Gaussian, ``0`` means it ties it.

    Args:
        y_true: Observed targets.
        y_pred: Predicted means.
        var: Predicted variances, per point.
        y_train: Training targets, used for the baseline mean and variance.
        eps: Floor applied to variances so a zero variance stays finite.

    Returns:
        The mean standardized log loss.
    """
    yt, yp, v, ytr = _ravel(y_true, y_pred, var, y_train)
    v = np.maximum(v, eps)
    mu0, var0 = ytr.mean(), max(float(ytr.var()), eps)

    model = 0.5 * np.log(2.0 * np.pi * v) + 0.5 * (yt - yp) ** 2 / v
    baseline = 0.5 * np.log(2.0 * np.pi * var0) + 0.5 * (yt - mu0) ** 2 / var0
    return float(np.mean(model - baseline))


# --- ranking metrics ---------------------------------------------------------------------


def spearman(y_true, y_pred):
    """Spearman rank correlation between targets and predictions (1.0 is a perfect ranking).

    Measures whether the surrogate orders points the same way as the truth -- the quantity that
    matters for optimization, where the *ranking* of candidates drives selection more than the
    absolute error.
    """
    yt, yp = _ravel(y_true, y_pred)
    return float(spearmanr(yt, yp).statistic)


def kendall_tau(y_true, y_pred):
    """Kendall's tau rank correlation (fraction of concordant minus discordant pairs)."""
    yt, yp = _ravel(y_true, y_pred)
    return float(kendalltau(yt, yp).statistic)
