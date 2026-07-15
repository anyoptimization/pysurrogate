"""Dace Kriging surrogate: fit, predict (with MSE/gradients), and theta optimization."""

import numpy as np

from pysurrogate.core.kernel import pairwise_diffs
from pysurrogate.core.optimizer import Callback
from pysurrogate.core.optimizer import Optimizer as GenericOptimizer
from pysurrogate.core.partitioning import Partitioning, default_partitioning
from pysurrogate.core.prediction import Prediction
from pysurrogate.core.sampling import LHS, Sampling
from pysurrogate.core.transformation import standardize
from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.fit import DaceFitError, fit
from pysurrogate.dace.problem import DaceProblem
from pysurrogate.dace.regr import ConstantRegression
from pysurrogate.dace.selection import _UNSET as _SELECTION_UNSET
from pysurrogate.dace.selection import ValidationSelection
from pysurrogate.optimizer import LBFGS, Restart

# sentinel for the `optimizer` argument: distinguishes "caller said nothing -> use the
# default optimizer" from an explicit ``optimizer=None``, which means "do not search -- freeze
# theta and just fit at its current value". (None cannot itself be the default, since None is
# the no-search request.)
_DEFAULT_OPTIMIZER = object()


def _default_optimizer():
    # screen-and-polish over the generic layer: a Latin-hypercube screen (the warm theta
    # competes as a forced sample) trimmed to the best few, each refined with L-BFGS. The
    # generic replacement for the old ScreenedLBFGS default.
    return Restart(LBFGS(), Sampling(16, LHS()), screen=4)


class Dace:
    def __init__(
        self,
        regr=None,
        corr=None,
        theta=1.0,
        theta_bounds=(0.0, 100.0),
        optimizer=_DEFAULT_OPTIMIZER,
        noise=0.0,
        noise_bounds=None,
        theta_prior=None,
        selection=None,
    ):
        """Construct the model with the given regression and correlation types.

        It can be initialized with different regression and correlation types, and
        whether hyperparameter optimization is used is controlled by the theta bounds.

        Args:
            regr: Regression trend instance: ConstantRegression(), LinearRegression() or QuadraticRegression().
                Defaults to ConstantRegression().
            corr: Correlation (kernel) instance, e.g. Gaussian(), Cubic(), Exponential(),
                RationalQuadratic(alpha=...). Defaults to Gaussian().
            theta: Initial value of theta. Can be a vector or a float
            theta_bounds: ``(lower, upper)`` box for the theta search, each a float or a
                per-dimension vector. ``None`` makes the search *unbounded above* (theta stays
                floored positive but has no ceiling) -- it does **not** freeze theta. To freeze
                theta, pass ``optimizer=None``. Mirrors ``noise`` / ``noise_bounds`` -- the value
                is the start, the bounds shape the search box.
            optimizer: Strategy used to optimize theta when bounds are given. Unset (the
                default) uses a generic ``Restart(LBFGS(), Sampling(16, LHS()), screen=4)`` --
                a Latin-hypercube screen feeding a few analytic-gradient L-BFGS polishes (better
                surrogate than Boxmin, lower cost) -- so a plain ``Dace()`` *optimizes* theta.
                Pass any ``core.optimizer.Optimizer`` (``LBFGS``, ``PatternSearch``, ``Adam``,
                ``Restart`` from ``pysurrogate.optimizer``), or ``Boxmin()`` for the legacy
                MATLAB-Dace pattern search (the port-fidelity anchor). Pass ``optimizer=None``
                to *disable* the search: theta is frozen at its value and only the GLS solve
                runs (same effect as ``theta_bounds=None``).
            noise: Deliberate observation noise added to the diagonal of the correlation
                matrix on every fit. Because that diagonal is unit, it is a noise-to-signal
                ratio: noise=0.1 models 10% noise. 0.0 (default) interpolates the data;
                noise>0 makes a regression GP that smooths through the points instead. When
                ``noise_bounds`` is given this is the *start* of the learned nugget instead of a
                fixed value -- exactly as ``theta`` is the start when ``theta_bounds`` is given.
            noise_bounds: ``(lower, upper)`` to *learn* the nugget jointly with theta (it becomes
                one more coordinate of the search vector, with its own analytic gradient), or
                ``None`` (default) to keep the nugget fixed at ``noise``. Mirrors
                ``theta`` / ``theta_bounds``: the value is the start, the bounds enable the
                search. Learning the nugget requires a search (an ``optimizer``) and the generic
                optimizer layer (the legacy ``Boxmin`` cannot learn it).
            theta_prior: ``(mean, lam)`` for a MAP prior (Tikhonov regularizer) on the length-scale
                search, or ``None`` (default) for plain maximum likelihood. When set, the theta
                search minimizes ``obj_MLE + lam * sum((log10(theta) - mean)**2)`` -- a Gaussian
                prior on the *encoded* (``log10``) length-scales that pulls the fit toward smoother,
                humbler length-scales (``10**mean``) and away from the short-length-scale
                over-confidence pure MLE falls into on small, biased designs. It shapes only which
                theta is *selected*; the committed GLS fit at that theta is unchanged. ``mean=0``
                centers on unit length-scale; larger ``lam`` regularizes harder (``lam~=0.01`` is a
                good starting point; retune per objective scale). ``None`` reproduces the MLE fit
                exactly, so it is the golden-safe library default -- turn it on for downstream
                optimization (Bayesian optimization), where calibrated-conservative uncertainty
                matters more than raw fit accuracy.
            selection: A :class:`~pysurrogate.dace.selection.Selection` strategy
                (:class:`~pysurrogate.dace.selection.MaximumLikelihood`,
                :class:`~pysurrogate.dace.selection.MAP`, :class:`~pysurrogate.dace.selection.HeldOut`)
                that bundles the hyperparameter selection in one object -- the search ``optimizer``, the
                MAP ``theta_prior``, the ``noise_bounds`` nugget policy, and (for a held-out strategy)
                an internal train/validation split. When given it supplies those, overriding the
                individual arguments. ``None`` (default) leaves the explicit arguments in force -- the
                historical behavior, byte-for-byte, so the default fit path is unchanged.
        """
        # a Selection strategy bundles the search configuration in one object; when given it supplies
        # the optimizer, the MAP prior, and the nugget policy (overriding those individual knobs) and
        # -- for a held-out strategy -- an internal train/validation split consulted in fit(). Default
        # (selection=None) leaves the explicit params in force: the historical behavior, byte-for-byte.
        self._selection = selection
        if selection is not None:
            optimizer = _DEFAULT_OPTIMIZER if selection.optimizer is _SELECTION_UNSET else selection.optimizer
            theta_prior = selection.theta_prior
            noise_bounds = selection.noise_bounds

        self.regr = regr if regr is not None else ConstantRegression()
        self.kernel = corr if corr is not None else Gaussian()

        # most of the model will be stored here
        self.model = None

        # the hyperparameter can be defined (coerce a list like the bounds below, so
        # a vector theta reaches the kernel as an array and not a Python list)
        self.theta = np.asarray(theta) if isinstance(theta, (list, tuple)) else theta

        # lower and upper bound that shape the search box (theta_bounds=None -> unbounded above,
        # NOT frozen; freezing is optimizer=None). Internal names stay tl/tu (the optimizers read them).
        if theta_bounds is None:
            self.tl, self.tu = None, None
        else:
            lo, hi = theta_bounds
            self.tl = np.asarray(lo) if isinstance(lo, (list, tuple)) else lo
            self.tu = np.asarray(hi) if isinstance(hi, (list, tuple)) else hi

        # strategy that optimizes theta within the bounds. Unset -> a generic screen-and-polish
        # (Restart(LBFGS(), Sampling(16, LHS()), screen=4)) that reaches a better surrogate than
        # the original Boxmin pattern search at lower cost, so a plain Dace() optimizes. An
        # explicit optimizer=None means "do not search" (freeze theta), distinct from the unset
        # default -- the sentinel keeps the two apart. Pass Boxmin() for an exact MATLAB-Dace
        # trajectory.
        self.optimizer = _default_optimizer() if optimizer is _DEFAULT_OPTIMIZER else optimizer

        # deliberate diagonal noise-to-signal term, always added on every fit (>0 -> a
        # regression GP that smooths through points). 0.0 -> strict interpolation. There is
        # no auto-repair climb: a non-PD fit raises, fix it by raising noise / noise_bounds.
        # When noise_bounds is given, `noise` is the START of the learned nugget instead.
        self.noise = noise

        # lower/upper bound if the nugget should be LEARNED (noise_bounds=None -> fixed at
        # `noise`). Mirrors tl/tu for theta: the value is the start, the bounds enable the
        # search. Internal names nl/nu match the tl/tu convention.
        if noise_bounds is None:
            self.nl, self.nu = None, None
        else:
            self.nl, self.nu = noise_bounds

        # optional MAP prior (mean, lam) on the encoded (log10) length-scales -- a Tikhonov
        # regularizer folded into the theta-search objective (see DaceProblem). None -> pure MLE.
        self.theta_prior = theta_prior

        # record of the hyperparameter optimization (search trajectory + diagnostics),
        # populated by the optimizer on fit; None until then / for a fixed theta.
        self.optimization = None

        # scalar multiplier on the predictive variance, set only by the standalone calibrate()
        # via cross-validation. 1.0 is the identity (uncalibrated): fit never calibrates, so
        # predict returns the raw kriging variance until calibrate() is called explicitly.
        self.scale = 1.0

    def fit(self, X, Y, optimize=True):
        """Fit the model: standardize, select theta (search or freeze), and solve the GLS system.

        Theta is selected by **maximum likelihood** over all rows by default. A held-out
        :class:`~pysurrogate.dace.selection.Selection` (e.g.
        :class:`~pysurrogate.dace.selection.HeldOut`) instead carves an internal train/validation
        split and selects on validation error; :meth:`refit` (``validate=True``) is the other
        held-out path, holding out the newly appended points.

        Args:
            X: Training inputs, shape ``(n, d)``.
            Y: Training targets, shape ``(n,)`` or ``(n, q)`` for multi-output.
            optimize: Whether to *run* the theta search on this fit. ``True`` (default) uses the
                configured ``optimizer`` (the strategy); ``False`` freezes theta at its current
                value and only re-solves the GLS system. This is the ``Model``-contract lever --
                ``optimize=False`` is equivalent to having been built with ``optimizer=None`` but
                decided per-fit, so the same instance can search on a cold fit and freeze on a
                screening fit. ``optimizer=`` chooses *which* search; ``optimize`` chooses *whether*.
        """
        if len(Y.shape) == 1:
            Y = Y[:, None]
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must have the same number of rows.")

        # a fresh fit invalidates any prior variance calibration -- reset to the identity so a
        # stale scale never leaks onto a new model; recalibrate against held-out data if wanted.
        self.scale = 1.0

        nX, nY, stats = self._standardize(X, Y)
        effective_optimizer, optimize_theta = self._search_config(optimize)
        if optimize_theta:
            # a held-out Selection searches on the training split and selects by error on the val split;
            # every other case (the default) searches over all rows by likelihood -- byte-identical.
            val = self._fit_holdout(nX, nY)
            if val is None:
                theta, noise, self.optimization = self._optimize_generic(nX, nY, effective_optimizer)
            else:
                (trX, trY), (vX, vY) = val
                # the search descends the objective on the train split; the held-out callback selects
                # by val error and early-stops after `patience` non-improving evals (the Selection's).
                theta, noise, self.optimization = self._optimize_generic(
                    trX, trY, effective_optimizer, val=(vX, vY), patience=self._selection.patience
                )
        else:
            # frozen theta (optimizer=None or optimize=False): commit at the current theta and noise.
            theta, noise, self.optimization = self.theta, self.noise, None
        # commit the final GLS fit on ALL rows at the selected (theta, noise), even under a held-out split.
        self._commit(nX, nY, X, Y, stats, theta, noise, wrap_pd_error=optimize_theta)

    def _fit_holdout(self, nX, nY):
        """The ``(train, val)`` standardized split for a held-out Selection, or ``None`` for MLE/MAP.

        Returns ``None`` unless a :class:`~pysurrogate.dace.selection.Selection` with a held-out policy
        is configured -- so the default fit path is untouched.
        """
        if self._selection is None:
            return None
        idx = self._selection.holdout(nX.shape[0])
        if idx is None:
            return None
        tr, va = idx
        return (nX[tr], nY[tr]), (nX[va], nY[va])

    def _standardize(self, X, Y):
        """Standardize inputs and targets to zero mean / unit variance (``ddof=1``); return ``(nX, nY, stats)``.

        A constant column/output has zero std, which would divide to NaN -- the shared
        :func:`~pysurrogate.core.transformation.standardize` guards it to 1 so the constant maps to
        all-zeros after centering and the fit degrades to a constant predictor.
        """
        nX, mX, sX = standardize(X)
        nY, mY, sY = standardize(Y)
        return nX, nY, {"mX": mX, "sX": sX, "mY": mY, "sY": sY}

    def _search_config(self, optimize):
        """Resolve and validate ``(effective_optimizer, optimize_theta)`` for this fit.

        The effective optimizer is the configured one only when ``optimize`` is True; ``optimize=False``
        (or ``optimizer=None``) freezes theta -- no search. Learning the nugget needs a search: a
        *permanent* ``optimizer=None`` with ``noise_bounds`` is a contradiction and raises, but a
        per-fit ``optimize=False`` screen just freezes the nugget at ``noise`` (no raise) so the same
        model can learn the nugget on a cold fit and screen cheaply with it frozen.
        """
        effective_optimizer = self.optimizer if optimize else None
        optimize_theta = effective_optimizer is not None
        optimize_noise = self.nl is not None and self.nu is not None
        if optimize_theta and not isinstance(effective_optimizer, GenericOptimizer):
            raise Exception(
                f"{type(effective_optimizer).__name__} is not a core.optimizer.Optimizer -- pass a generic "
                "optimizer (LBFGS, PatternSearch, Boxmin, ...) or optimizer=None / optimize=False to freeze theta."
            )
        if optimize_noise and self.optimizer is None:
            raise Exception(
                "noise_bounds needs an optimizer to learn the nugget -- build with one, or drop noise_bounds "
                "to fix the nugget at `noise`."
            )
        return effective_optimizer, optimize_theta

    def _commit(self, nX, nY, X, Y, stats, theta, noise, wrap_pd_error=False):
        """Solve the GLS system at ``(theta, noise)`` on all rows and store the fitted model dict.

        ``wrap_pd_error`` re-raises a :class:`DaceFitError` from the commit with a hint to add noise
        -- used only after a theta *search*, where a non-PD commit usually means the selected theta
        needs a nugget. A frozen fit lets the raw error through (it may be a singular regression
        design, not a correlation-matrix issue), so its message is not overwritten.
        """
        try:
            model = fit(nX, nY, self.regr, self.kernel, theta, noise=noise)
        except DaceFitError as e:
            if wrap_pd_error:
                raise DaceFitError(
                    "No positive-definite correlation matrix for the selected theta at the requested "
                    "noise; set noise / noise_bounds to regularize the committed fit."
                ) from e
            raise
        # keep the raw (destandardized) training data so refit() can append to it
        self.model = {**model, "X": X, "Y": Y, **stats, "nX": nX, "nY": nY}
        # a single shared predictive-variance scale (the Prediction contract carries one variance
        # per point, shared across outputs). Average the destandardized per-output sigma2 rather
        # than summing it -- a sum grows with the number of outputs and is meaningless as a scale;
        # single-output is unaffected (mean of one value). A genuinely per-output predictive
        # variance would need the shared-variance layout in Prediction/predictions_frame relaxed.
        self.model["sigma2"] = float(np.mean(np.square(stats["sY"]) * self.model["_sigma2"]))

    def _theta_bounds_for_problem(self):
        """Theta ``(lo, hi)`` for the DaceProblem -- the finite bounds, or an unbounded box.

        With ``theta_bounds=None`` (tl/tu None) the search is unbounded: keep the length-scale
        positive (floored) but put no ceiling on it (``+inf``). The number of coordinates ``p``
        then comes from the start ``self.theta`` -- a scalar is isotropic (``p=1``), a vector is
        ARD -- since there are no bounds to read it from.
        """
        if self.tl is not None and self.tu is not None:
            return (self.tl, self.tu)
        p = len(np.atleast_1d(np.asarray(self.theta, dtype=float)))
        return (np.full(p, 1e-12), np.full(p, np.inf))

    def _optimize_generic(self, nX, nY, optimizer, val=None, patience=None):
        """Run a generic optimizer over a DaceProblem on ``(nX, nY)``; return ``(theta, noise, record)``.

        With ``val=(nXv, nYv)`` the candidates are *selected* by held-out error on that set
        (:class:`ValidationSelection`) while the optimizer still searches by likelihood; ``patience``
        early-stops the search after that many non-improving held-out evaluations (``None`` = never).
        Without ``val``, selection is pure maximum likelihood. With ``noise_bounds`` the nugget is a
        learned coordinate. Forming the held-out split is the caller's concern (:meth:`refit` holds out
        the newly appended points, a :class:`HeldOut` selection an internal split) -- this method just
        wires the selection callback.
        """
        # noise_bounds set (nl/nu) -> learn the nugget; else fix it at self.noise. Mirrors theta.
        noise_kw = {"noise_bounds": (self.nl, self.nu)} if self.nl is not None else {"noise": self.noise}
        theta_bounds = self._theta_bounds_for_problem()
        problem = DaceProblem(nX, nY, self.regr, self.kernel, theta_bounds, theta_prior=self.theta_prior, **noise_kw)
        callback = ValidationSelection(problem, val[0], val[1], patience=patience) if val is not None else Callback()

        x0 = self._encode_start(problem)
        res = optimizer.setup(problem, x0=x0, callback=callback).run()
        # no PD candidate at the search noise -> fall back to the start theta; the committing
        # fit() then either succeeds there or raises (no hidden repair climb). Fix a persistent
        # failure by setting noise / noise_bounds, not silently.
        x_best = res.x if res.x is not None else x0
        theta, noise = problem.decode(x_best)
        record = {
            "theta": theta,
            "noise": noise,
            "x": x_best,
            "f": res.f,
            "message": res.message,
            "n_evals": res.n_evals,
        }
        # expose the visited theta trajectory when the optimizer records one (pattern searches do):
        # decode each visited point back to theta space, mirroring the old optimization["models"].
        # `visited` is a documented Optimizer-contract attribute (empty for strategies that keep no
        # trajectory), so this reads it directly -- no getattr reach-in -- and only records a
        # trajectory when one is actually present.
        if optimizer.visited:
            record["models"] = [{"theta": problem.decode(x)[0]} for x in optimizer.visited]
        return theta, noise, record

    def _encode_start(self, problem):
        """Encode the warm start ``self.theta`` (+ the nugget start ``self.noise`` when learned) as a log10 x0."""
        theta0 = np.broadcast_to(np.atleast_1d(np.asarray(self.theta, dtype=float)), (problem.p,))
        x0 = np.log10(np.maximum(theta0, 1e-12))
        if problem.learn_noise:
            # self.noise is the START of the learned nugget; clamp it into the noise bounds (a
            # 0.0 start lands on the lower bound) so its log10 is finite and inside the box.
            noise0 = np.clip(float(self.noise), *problem.noise_bounds)
            x0 = np.append(x0, np.log10(noise0))
        return x0

    def refit(self, X, Y, optimize=True, validate=True):
        """Append new observations and re-fit, reusing the fitted theta as the warm start.

        Takes only the *new* points, appends them to the data the model was last fit on, and re-fits
        on the combined set. Theta is always **seeded from the previous fit** (it barely moves when a
        few points are added); ``optimize`` decides whether to search, and ``validate`` decides how
        theta is selected when it does:

        - ``optimize=True`` (default) -- warm-start the model's optimizer from the previous optimum.
          With ``validate=True`` (default) the **new points are held out** and theta is chosen by how
          well the old data predicts them (:class:`~pysurrogate.dace.selection.ValidationSelection`) --
          a regularizer against theta over-fitting on a sparse, adaptively-sampled design. With
          ``validate=False`` theta is selected by likelihood over all rows.
        - ``optimize=False`` -- freeze theta at its previous value and just re-solve the kernel matrix
          on the grown data (no search). ``validate`` is moot.

        The new points are always appended and the final model is fit on **all** rows.

        Args:
            X: The new input points to add (only the additions, not the full set).
            Y: The target values corresponding to the new points ``X``.
            optimize: ``True`` warm-starts the theta search from the previous fit; ``False`` keeps
                theta fixed and only re-solves the kernel matrix on the grown data.
            validate: When searching, hold out the newly appended points to select theta (``True``,
                default) instead of selecting by likelihood over all rows (``False``). Ignored when
                ``optimize=False``.

        Raises:
            Exception: If called before any successful ``fit``.
        """
        if self.model is None:
            raise Exception("refit() requires a prior fit(); call fit() first.")

        # match fit's reshape so a 1d Y appends cleanly onto the stored 2d targets
        if len(Y.shape) == 1:
            Y = Y[:, None]

        # append the new observations to the data the model was last fit on
        n_old = self.model["X"].shape[0]
        X = np.vstack([self.model["X"], X])
        Y = np.vstack([self.model["Y"], Y])

        # reuse the previously optimized theta AND nugget -- the warm start (search) or frozen value
        # (no search). Warm-starting the nugget from the previous optimum mirrors calibrate(); without
        # it a learned-nugget refit would silently reset the nugget to the constructor start.
        self.theta = self.model["theta"]
        self.noise = float(self.model["noise"])
        self.scale = 1.0

        nX, nY, stats = self._standardize(X, Y)
        effective_optimizer, optimize_theta = self._search_config(optimize)
        if optimize_theta and validate:
            # select theta by how well the OLD rows predict the appended NEW rows (held-out selection);
            # forward the Selection's early-stopping patience (as fit() does), guarding for no selection.
            patience = self._selection.patience if self._selection is not None else None
            theta, noise, self.optimization = self._optimize_generic(
                nX[:n_old], nY[:n_old], effective_optimizer, val=(nX[n_old:], nY[n_old:]), patience=patience
            )
        elif optimize_theta:
            theta, noise, self.optimization = self._optimize_generic(nX, nY, effective_optimizer)
        else:
            theta, noise, self.optimization = self.theta, self.noise, None
        self._commit(nX, nY, X, Y, stats, theta, noise, wrap_pd_error=optimize_theta)

    def predict(self, X, var=False, grad=False, mse=None):
        """Predict the mean, optionally the variance and the mean's gradient, in one pass.

        Mean and variance share the kernel matrix and the Cholesky solve, so computing
        them together is cheaper than two calls -- this is why both live on one method.

        Args:
            X: Query inputs, shape ``(m, d)``.
            var: Also return the predictive variance (the kriging MSE), shape ``(m, 1)``. For a
                multi-output model the variance is *shared* across outputs (the kernel and
                theta are shared), so it stays ``(m, 1)`` regardless of the number of outputs.
                This is the ``Model``-contract name; ``mse`` is the DACE-literature alias.
            grad: Also return the gradient of the mean w.r.t. the query point. Single-output
                models return ``(m, d)``; multi-output models return ``(m, q, d)`` -- one
                gradient per output. This is what a gradient-based optimizer searches over.
            mse: Deprecated DACE-literature alias of ``var``; when given it overrides ``var``.

        Returns:
            ``y`` (always), plus ``var`` / ``grad`` when their flag is set (else None).
            When ``var`` and ``grad`` are *both* set, ``var_grad`` (the variance gradient,
            ``(m, d)``) is also returned -- it shares the variance and mean-gradient terms,
            so the extra cost is one triangular solve per point.
        """
        if self.model is None:
            raise Exception("predict() requires a prior fit(); call fit() first.")

        # `mse` is the DACE-literature alias of `var` (mirrors Prediction.mse/var); honor it.
        if mse is not None:
            var = mse

        mX, sX, nX = self.model["mX"], self.model["sX"], self.model["nX"]
        mY, sY = self.model["mY"], self.model["sY"]
        regr, corr, theta = self.regr, self.kernel, self.model["theta"]
        beta, gamma = self.model["beta"], self.model["gamma"]

        # normalize the query inputs with the mX/sX fitted before (distinct from the training nX)
        _nX = (X - mX) / sX

        # pairwise differences x_i - t_j, shape (m*n, d): shared by the kernel matrix here
        # and, when grad is requested, its gradient -- built once so predict(grad=True) does
        # not rebuild it. _F is the regression design, _R the kernel matrix reshaped (m, n).
        m, d = _nX.shape
        n = nX.shape[0]
        _D = pairwise_diffs(_nX, nX)
        _F = regr(_nX)
        _R = corr(_D, theta).reshape(m, n)

        # predict and destandardize
        _sY = _F @ beta + (gamma.T @ _R.T).T
        _Y = (_sY * sY) + mY

        _mse = None
        _mse_clamped = None  # mask of points whose variance was clamped (for mse_grad)
        if var:
            # C is the square, full-rank Cholesky factor, so a direct solve is exact and
            # ~50x faster than lstsq's SVD on the (n, m) right-hand side.
            rt = np.linalg.solve(self.model["C"], _R.T)
            Ginv = np.linalg.inv(self.model["G"])
            u = (self.model["Ft"].T @ rt).T - _F
            v = u @ Ginv
            _mse = (self.model["sigma2"] * (1 + np.sum(v**2, axis=1) - np.sum(rt**2, axis=0)))[:, None]
            # the kriging variance is non-negative by theory; negatives are rounding near
            # training points (the cubic/spline kernels can dip moderately negative over a
            # region). Clamp at 0 so downstream sqrt(mse) (std, EI) never returns NaN.
            _mse_clamped = (_mse < 0.0).ravel()
            _mse = np.maximum(_mse, 0.0)
            # apply the validation-fitted variance scale (1.0 = uncalibrated identity). Baked
            # into the returned variance so sigma = sqrt(var) and every downstream metric is
            # calibrated without knowing about the scale; the gradient below scales identically.
            _mse = self.scale * _mse

        # mse_grad is available alongside both the variance and the mean gradient, since
        # it reuses their terms (rt, v, Ginv and the batched dR/dF). sigma2 is already
        # destandardized and the dimensionless bracket is in normalized space, so the chain
        # rule to the original input is a single 1/sX scaling per dimension.
        want_mse_grad = var and grad

        _grad = None
        _mse_grad = None
        if grad:
            # Fully vectorized over the m query points -- no per-point Python loop. dR is
            # the kernel gradient d r(x_i, t_j)/d x_i for every query/train pair from a
            # single corr.grad call; dF is the regression-basis Jacobian per query point.
            q = _sY.shape[1]
            dR = corr.grad(_D, theta).reshape(m, n, d)  # (m, n, d): reuses the diff matrix _D
            dF = self.regr.grad(_nX)  # (m, d, p)

            # mean gradient (dF @ beta) + (gamma^T @ dR) per point -> (m, q, d), then
            # destandardize per output (sY) and per input dimension (1/sX). Single-output
            # keeps the historical (m, d) shape; multi-output is (m, q, d).
            mean_grad = np.einsum("idp,pq->iqd", dF, beta) + np.einsum("nq,ind->iqd", gamma, dR)
            mean_grad = mean_grad * sY[None, :, None] / sX[None, None, :]
            _grad = mean_grad[:, 0, :] if q == 1 else mean_grad

            if want_mse_grad:
                # variance gradient of the bracket 1 + ||v||^2 - ||rt||^2. One batched solve
                # gives drt = C^-1 dR for every point; du/dv reuse the mean's Ft, Ginv, rt, v.
                # C is the (square, full-rank) Cholesky factor, so a direct solve is exact
                # and ~75x faster than lstsq's SVD on the (n, m*d) right-hand side.
                b = dR.transpose(1, 0, 2).reshape(n, m * d)
                drt = np.linalg.solve(self.model["C"], b).reshape(n, m, d).transpose(1, 0, 2)
                du = np.einsum("np,ind->idp", self.model["Ft"], drt) - dF  # (m, d, p)
                dv = np.einsum("idp,pk->idk", du, Ginv)  # (m, d, p)
                d_bracket = 2.0 * (np.einsum("idp,ip->id", dv, v) - np.einsum("ind,ni->id", drt, rt))
                _mse_grad = self.model["sigma2"] * d_bracket / sX  # (m, d)
                # where the variance was clamped to 0 the clamped surface is flat -> 0 grad
                _mse_grad[_mse_clamped] = 0.0
                _mse_grad = self.scale * _mse_grad  # same scale as the variance it differentiates

        return Prediction(y=_Y, var=_mse, grad=_grad, var_grad=_mse_grad)

    def calibrate(self, partitioning=None):
        """Fit a scalar variance multiplier by cross-validation so the intervals are calibrated.

        The kriging variance is systematically too small (overconfident) by a roughly constant
        factor; this corrects it with the one scalar ``s = mean[(y - yhat)**2 / var]`` -- the
        maximum-likelihood scale that drives the calibration ratio to 1. Only the predictive
        *variance* is rescaled (``var -> s * var``); the mean predictions are untouched, so accuracy
        metrics do not change. A later :meth:`fit` resets the scale back to the identity.

        The ``(yhat, var)`` pairs are gathered **out-of-sample**: each split re-fits at the model's
        already-chosen theta/noise (``optimizer=None``, no re-search) and predicts its held-out
        rows, so reusing :meth:`predict` keeps the fold variance identical in form to the deployed
        model's. With cross-validation every training row contributes one out-of-fold ratio, no
        separate validation set to spend -- so the scale pools all ``n`` rows and stays usable even
        on small designs. k-fold is the default rather than leave-one-out: LOO leaves the fit almost
        unchanged, so its errors look too small and the scale comes out optimistic; k-fold's larger
        holdouts are more honestly out-of-sample (and ``k`` re-fits).

        Args:
            partitioning: How to form the held-out splits. ``None`` (default) uses 5-fold
                cross-validation. Pass a :class:`~pysurrogate.core.partitioning.Partitioning`
                (e.g. ``RandomPartitioning`` for repeated hold-out, a higher ``k_folds`` for
                LOO-like behavior), or a **boolean mask** over the training rows -- a single split
                whose ``True`` rows are the held-out validation block and the rest are re-fit.

        Returns:
            The fitted scale ``s`` (also stored on ``self.scale``); ``> 1`` means the model was
            overconfident, ``< 1`` underconfident.

        Raises:
            Exception: If called before a successful :meth:`fit`.
            ValueError: If a mask is malformed (wrong length, or holds out all / none of the rows).
        """
        if self.model is None:
            raise Exception("calibrate() requires a prior fit(); call fit() first.")
        X, Y = self.model["X"], self.model["Y"]  # the original (un-normalized) training data

        # each split is a standalone model at the FIXED theta/noise (optimizer=None -> no search),
        # so its prediction on the held-out rows is genuinely out-of-sample.
        ratios = []
        for train, test in self._calibration_splits(partitioning, X):
            sub = Dace(
                regr=self.regr, corr=self.kernel, theta=self.model["theta"], noise=self.model["noise"], optimizer=None
            )
            sub.fit(X[train], Y[train])
            pred = sub.predict(X[test], var=True)
            resid2 = np.square(Y[test] - pred.y)  # (m, q): per-output squared error
            var = np.maximum(pred.var, 1e-300)  # (m, 1): shared across outputs, guard /0
            ratios.append((resid2 / var).ravel())

        # the scale that makes the pooled out-of-fold calibration ratio exactly 1
        self.scale = float(np.mean(np.concatenate(ratios)))
        return self.scale

    def _calibration_splits(self, partitioning, X):
        """Resolve the ``calibrate`` argument into a list of ``(train_idx, test_idx)`` pairs.

        Args:
            partitioning: ``None`` (default 5-fold CV), a ``Partitioning``, or a boolean mask over
                the training rows (a single held-out split).
            X: The training inputs, used for the row count and to drive the partitioning.

        Returns:
            A list of ``(train_idx, test_idx)`` integer-index pairs.
        """
        if partitioning is None:
            partitioning = default_partitioning()  # the canonical k-fold default (DEFAULT_CV_FOLDS)
        if isinstance(partitioning, Partitioning):
            return [(s.train, s.test) for s in partitioning.do(X)]

        # otherwise a boolean mask: True rows are the held-out validation block (like fit's mask)
        mask = np.asarray(partitioning, dtype=bool).ravel()
        if mask.shape[0] != X.shape[0]:
            raise ValueError(
                f"calibration mask must have one entry per training row (got {mask.shape[0]} for {X.shape[0]})."
            )
        held = int(mask.sum())
        if held == 0 or held == mask.shape[0]:
            raise ValueError(f"calibration mask must hold out some rows but not all (got {held} of {mask.shape[0]}).")
        return [(np.flatnonzero(~mask), np.flatnonzero(mask))]
