"""Hyperparameter-selection strategies and the held-out-error selection Callback."""

import numpy as np

from pysurrogate.core.optimizer import Callback
from pysurrogate.dace.corr import calc_kernel_matrix
from pysurrogate.dace.fit import DaceFitError, fit

# sentinel for a Selection's optimizer: "not specified -> let the engine pick its default search"
# (distinct from optimizer=None, which the engine reads as "freeze the length-scale -- no search").
_UNSET = object()


class Selection:
    """A GP hyperparameter-selection strategy: how the length-scale (and learned nugget) are chosen.

    One reusable object, passed to a :class:`~pysurrogate.dace.Dace` engine (``selection=``) or to any
    GP model backend, that configures the whole selection in one place: the search ``optimizer``, the
    objective (maximum likelihood vs a MAP prior), the nugget policy (fixed vs learned), and -- for
    held-out selection -- an internal train/validation split. It is a plain config object; the engine
    reads its fields. Subclasses set the objective; all share the ``optimizer`` and ``noise_bounds``.

    Args:
        optimizer: The search strategy (a ``core.optimizer.Optimizer``, or ``None`` to freeze the
            length-scale); unset lets the engine use its default search.
        noise_bounds: ``(lo, hi)`` to *learn* the nugget jointly with the length-scale, or ``None`` to
            keep it fixed at the engine's ``noise``.
        theta_prior: ``(mean, lam)`` MAP prior on ``log10(theta)``, or ``None`` for pure likelihood.
    """

    def __init__(self, optimizer=_UNSET, noise_bounds=None, theta_prior=None):
        self.optimizer = optimizer
        self.noise_bounds = noise_bounds
        self.theta_prior = theta_prior
        self.patience = None  # early-stop after this many non-improving held-out evals (HeldOut sets it)

    def holdout(self, n):
        """``(train_idx, val_idx)`` to hold out for validation-based selection, or ``None``.

        ``None`` (the default) means the candidates are selected by the likelihood objective on all
        rows; a strategy that returns indices selects by held-out error instead.
        """
        return None

    def __repr__(self):
        return type(self).__name__


class MaximumLikelihood(Selection):
    """Select by maximum likelihood -- the DACE profile likelihood (the engine default, like Kriging)."""


class MAP(Selection):
    """Maximum likelihood plus a Gaussian prior on ``log10(length-scale)`` -- a Tikhonov regularizer.

    Pulls the fit toward smoother length-scales (``10**mean``), curbing the short-length-scale
    over-fitting that pure maximum likelihood falls into on small, biased designs.

    Args:
        mean: Prior centre on ``log10(theta)`` (``0`` centres on unit length-scale).
        lam: Prior strength (larger regularizes harder; ``~0.01`` is a good start).
        optimizer: As :class:`Selection`.
        noise_bounds: As :class:`Selection`.
    """

    def __init__(self, mean=0.0, lam=0.01, optimizer=_UNSET, noise_bounds=None):
        super().__init__(optimizer=optimizer, noise_bounds=noise_bounds, theta_prior=(float(mean), float(lam)))


class HeldOut(Selection):
    """Optimize the training objective (MLE or MAP) but early-stop and select on held-out error.

    The optimizer descends the ``objective`` -- maximum likelihood, or a MAP prior when the objective
    carries one -- on a training split; every visited candidate is re-scored on the held-out split;
    the search **stops once that held-out error stops improving** for ``patience`` evaluations; and the
    **best-on-validation** hyperparameters are returned (then the final GLS fit is committed on all
    rows). The fit-time analogue of early stopping: descend the training objective, but keep the
    hyperparameters that generalize best rather than the ones that maximize the training likelihood.
    The split is deterministic in ``seed`` so a fit stays reproducible.

    Args:
        objective: The training objective to descend -- :class:`MaximumLikelihood` (default) or
            :class:`MAP`. Its optimizer, prior, and nugget policy are inherited.
        fraction: Fraction of the training rows held out for selection.
        patience: Stop after this many consecutive non-improving held-out evaluations. ``None`` never
            early-stops -- it descends the whole objective and just keeps the best-on-validation seen.
        seed: Seed for the (deterministic) split.
    """

    def __init__(self, objective=None, fraction=0.25, patience=10, seed=0):
        base = objective if objective is not None else MaximumLikelihood()
        super().__init__(optimizer=base.optimizer, noise_bounds=base.noise_bounds, theta_prior=base.theta_prior)
        self.objective = base
        self.fraction = float(fraction)
        self.patience = patience
        self.seed = int(seed)

    def holdout(self, n):
        m = int(np.clip(round(self.fraction * n), 1, max(1, n - 1)))  # held-out count, >=1 and < n
        perm = np.random.RandomState(self.seed).permutation(n)
        return perm[m:], perm[:m]  # (train_idx, val_idx)


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
