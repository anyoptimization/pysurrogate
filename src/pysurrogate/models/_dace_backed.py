"""Shared base for the Model adapters over the ``Dace`` engine (Kriging, KPLS, RotatedKriging)."""

from typing import Callable, Optional

import numpy as np

from pysurrogate.core.model import Model
from pysurrogate.util.misc import at_least2d


class DaceBackedModel(Model):
    """A ``Model`` whose backend is a ``Dace`` engine stored in ``self.model``.

    Centralizes what every Dace adapter shares so it lives in one place instead of being
    repeated per backend:

    - duplicate elimination on by default (duplicate design points make the correlation
      matrix singular), while remaining user-overridable via ``eliminate_duplicates=``;
    - the ``regr``/``corr`` default-if-``None`` resolution (per-subclass factories);
    - the per-dimension theta / theta-bounds broadcast used by every ARD-style search;
    - ``_refit`` routed through the **Model preprocessing pipeline** (active-dims selection,
      nan/duplicate filtering, the fitted normalization) before the engine's warm-started
      ``refit`` -- so a refit sees inputs in exactly the space the engine was trained on --
      plus the ``self._X``/``self._y`` bookkeeping the generic lifecycle relies on;
    - the pass-through ``_predict`` (the Dace engine speaks the ``Prediction`` vocabulary).

    Subclasses set the class attributes ``default_regr`` / ``default_corr`` (factories, or
    ``None`` when the subclass builds its kernel itself) and implement ``_fit``.
    """

    # factory for the trend when regr=None (subclass supplies); None -> no default
    default_regr: Optional[Callable[[], object]] = None
    # factory for the kernel when corr=None; None -> subclass builds its own
    default_corr: Optional[Callable[[], object]] = None

    def __init__(self, regr=None, corr=None, selection=None, **kwargs):
        # default on, but a user override wins -- passing eliminate_duplicates=False must not
        # collide with the explicit default (kwargs.setdefault, not a positional pass-through).
        kwargs.setdefault("eliminate_duplicates", True)
        super().__init__(**kwargs)
        self.regr = regr if regr is not None else (self.default_regr() if self.default_regr else None)
        self.corr = corr if corr is not None else (self.default_corr() if self.default_corr else None)
        # optional hyperparameter-selection strategy (MLE / MAP / held-out), the same object Dace
        # takes; when given it supplies the optimizer / prior / nugget policy (see Dace `selection`).
        self.selection = selection

    @staticmethod
    def _broadcast_theta(theta, theta_bounds, h):
        """Broadcast a scalar theta start (and finite bounds) to ``h`` per-dimension coordinates.

        The start is broadcast even with ``theta_bounds=None`` -- Dace reads the ARD dimension
        from the start vector for an unbounded search, so an ARD/reduced search must not silently
        collapse to isotropic.

        Args:
            theta: Scalar starting length-scale.
            theta_bounds: ``(lo, hi)`` scalar bounds, or ``None`` for an unbounded search.
            h: Number of length-scale coordinates.

        Returns:
            ``(theta, theta_bounds)`` with length-``h`` vectors (bounds kept ``None`` if unset).
        """
        theta = np.full(h, theta)
        if theta_bounds is not None:
            lo, hi = theta_bounds
            theta_bounds = (np.full(h, lo), np.full(h, hi))
        return theta, theta_bounds

    def _refit(self, X, y, optimize=True):
        # incremental warm-started re-fit via the Dace engine (append + reuse the fitted theta):
        # optimize=True warm-starts the length-scale search, optimize=False freezes it. The generic
        # Model.refit handles the out-of-sample scoring and record; this is just the absorb step.
        Xr, yr = at_least2d(X, expand="r"), at_least2d(y, expand="c")
        # run the new points through the SAME preprocessing the fit path uses -- active-dimension
        # selection, duplicate/nan filtering, and the fitted normalization -- so the engine sees
        # inputs in exactly the space it was trained on (a raw pass-through crashed on active_dims
        # and mixed normalized with un-normalized data).
        Xp, yp = self.preprocess(Xr, yr)
        self.model.refit(Xp, yp, optimize=optimize)
        self._X = np.vstack([self._X, Xr])
        self._y = np.vstack([self._y, yr])

    def _predict(self, X, var=False, grad=False):
        # Dace.predict speaks the Model vocabulary (var=, grad=), so this is a direct pass-through
        # -- the mean, variance and gradients all share Dace's single Cholesky solve. The Model
        # lifecycle un-normalizes them.
        return self.model.predict(X, var=var, grad=grad)
