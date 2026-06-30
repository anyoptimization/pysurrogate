"""Validation selection: a Callback that picks theta by held-out error instead of likelihood."""

import numpy as np

from pysurrogate.core.optimizer import Callback
from pysurrogate.dace.corr import calc_kernel_matrix
from pysurrogate.dace.fit import DaceFitError, fit


class ValidationSelection(Callback):
    """Select theta by **held-out prediction error**, while the optimizer still searches by MLE.

    The DACE analogue of "search by likelihood, pick by validation": the optimizer descends the
    profile likelihood, but this callback re-scores every visited candidate on a held-out set and
    keeps the one with the lowest held-out RMSE. It is the regularizer against theta over-fitting
    on a sparse design -- the same role the old ``_select`` played, now as a pluggable
    :class:`~pysurrogate.core.optimizer.Callback`.

    The candidate model is re-fit on the training rows at the decoded ``(theta, noise)`` and used
    to predict the held-out rows; both are in the standardized space the problem was built on, so
    the error is scale-free. An infeasible re-fit scores ``+inf`` (it simply will not be picked).

    Args:
        problem: The :class:`~pysurrogate.dace.problem.DaceProblem` being optimized -- supplies the
            *training* design ``X`` / ``Y``, the trend, the kernel and the ``decode`` map.
        x_val: Held-out inputs, standardized with the training stats, shape ``(m, d)``.
        y_val: Held-out targets, standardized, shape ``(m,)`` or ``(m, q)``.
        patience: Early-stop after this many non-improving evaluations (``None`` = never).
    """

    def __init__(self, problem, x_val, y_val, patience=None):
        super().__init__(patience)
        self.problem = problem
        self.x_val = np.atleast_2d(np.asarray(x_val, float))
        yv = np.asarray(y_val, float)
        self.y_val = yv[:, None] if yv.ndim == 1 else yv

    def score(self, x, f, info):
        p = self.problem
        theta, noise = p.decode(x)
        try:
            model = fit(p.X, p.Y, p.regr, p.kernel, theta, noise=noise)
        except DaceFitError:
            return float("inf")
        # predict the held-out rows from the candidate fit (normalized space)
        F = p.regr(self.x_val)
        R = calc_kernel_matrix(self.x_val, p.X, p.kernel, theta)
        y_hat = F @ model["beta"] + (model["gamma"].T @ R.T).T
        return float(np.sqrt(np.mean(np.square(y_hat - self.y_val))))
