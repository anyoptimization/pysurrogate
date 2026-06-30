"""Dace Kriging surrogate: fit, predict (with MSE/gradients), and theta optimization."""

import numpy as np

from pysurrogate.core.optimizer import Callback
from pysurrogate.core.optimizer import Optimizer as GenericOptimizer
from pysurrogate.core.partitioning import CrossvalidationPartitioning, Partitioning
from pysurrogate.core.prediction import Prediction
from pysurrogate.core.sampling import LHS, Sampling
from pysurrogate.dace.corr import Gaussian, calc_kernel_matrix
from pysurrogate.dace.fit import DaceFitError, fit
from pysurrogate.dace.problem import DaceProblem
from pysurrogate.dace.regr import ConstantRegression
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
        """
        self.regr = regr if regr is not None else ConstantRegression()
        self.kernel = corr if corr is not None else Gaussian()

        # most of the model will be stored here
        self.model = None

        # the hyperparameter can be defined (coerce a list like the bounds below, so
        # a vector theta reaches the kernel as an array and not a Python list)
        self.theta = np.array(theta) if type(theta) is list else theta

        # lower and upper bound that shape the search box (theta_bounds=None -> unbounded above,
        # NOT frozen; freezing is optimizer=None). Internal names stay tl/tu (the optimizers read them).
        if theta_bounds is None:
            self.tl, self.tu = None, None
        else:
            lo, hi = theta_bounds
            self.tl = np.array(lo) if type(lo) is list else lo
            self.tu = np.array(hi) if type(hi) is list else hi

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

        # record of the hyperparameter optimization (search trajectory + diagnostics),
        # populated by the optimizer on fit; None until then / for a fixed theta.
        self.optimization = None

        # scalar multiplier on the predictive variance, set only by the standalone calibrate()
        # via cross-validation. 1.0 is the identity (uncalibrated): fit never calibrates, so
        # predict returns the raw kriging variance until calibrate() is called explicitly.
        self.scale = 1.0

    def fit(self, X, Y, validation=None, append=True):
        """Fit the model, optionally selecting theta on a held-out subset of the rows.

        Args:
            X: Training inputs, shape ``(n, d)``.
            Y: Training targets, shape ``(n,)`` or ``(n, q)`` for multi-output.
            validation: Optional binary mask over the rows of ``X`` (one entry per row), ``None``
                by default. A truthy entry marks a row as *held out for theta selection*:
                theta candidates are fit on the ``0`` rows and scored on the held-out rows
                (in normalized space), and the theta with the lowest held-out error is
                chosen instead of the maximum-likelihood one. ``None`` keeps the pure MLE
                behavior. Has no effect without theta bounds -- there is no search to steer.
            append: What the *final* model is fit on once theta is chosen, when a mask is given.
                ``True`` (default) refits on all rows, so the held-out rows rejoin and
                ``predict`` uses every label. ``False`` keeps the model fit on the ``0``
                rows only, so it never saw the held-out rows (useful when their error is
                reported separately). Ignored when ``validation`` is ``None``.
        """
        # the targets should be a 2d array
        if len(Y.shape) == 1:
            Y = Y[:, None]

        # check if for each observation a target values exist
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must have the same number of rows.")

        # a fresh fit invalidates any prior variance calibration -- reset to the identity so a
        # stale scale never leaks onto a new model; recalibrate against held-out data if wanted.
        self.scale = 1.0

        # save the mean and standard deviation of the input. Stats are over all rows,
        # so the held-out validation rows share the training normalization -- selection
        # then scores in this one normalized space (no destandardization, scale-free).
        mX, sX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
        mY, sY = np.mean(Y, axis=0), np.std(Y, axis=0, ddof=1)

        # guard zero-variance columns/outputs (a constant degenerates the normalization,
        # dividing by zero and poisoning the fit with NaN). Setting std to 1 maps the
        # constant to all-zeros after centering, so the fit stays finite and a constant
        # target degrades gracefully to a constant predictor: predict -> its mean, since
        # destandardizing 0 gives 0*sY + mY = mY.
        sX = np.where(sX == 0.0, 1.0, sX)
        sY = np.where(sY == 0.0, 1.0, sY)

        # standardize the input
        nX = (X - mX) / sX
        nY = (Y - mY) / sY

        stats = {"mX": mX, "sX": sX, "mY": mY, "sY": sY}
        # whether to search is decided by the OPTIMIZER, not the bounds: optimizer=None means
        # "freeze theta" (no search), an optimizer present means search. The bounds only shape the
        # box -- None bounds make the search unbounded (the generic layer seeds from a finite
        # window and descends without a ceiling), they no longer mean "don't search".
        optimize_theta = self.optimizer is not None

        # the nugget is LEARNED exactly when noise_bounds was given (nl/nu set) -- mirrors
        # optimize_theta. Learning it needs a search (an optimizer).
        optimize_noise = self.nl is not None and self.nu is not None

        # every optimizer is a generic core.optimizer.Optimizer now (the legacy DACE-specific
        # optimizer package is gone) -- guard against anything else with a clear message.
        if optimize_theta and not isinstance(self.optimizer, GenericOptimizer):
            raise Exception(
                f"{type(self.optimizer).__name__} is not a core.optimizer.Optimizer -- pass a generic "
                "optimizer (LBFGS, PatternSearch, Boxmin, ...) or optimizer=None to freeze theta."
            )
        if optimize_noise and not optimize_theta:
            raise Exception("noise_bounds requires an optimizer -- the nugget is learned jointly with theta.")

        if optimize_theta:
            # optimize over a DaceProblem; selection is a Callback (held-out when a validation mask
            # is given, maximum-likelihood otherwise). The nugget is a learned coordinate when
            # noise_bounds was set. An unbounded box (theta_bounds=None) is an unbounded search.
            theta, noise, self.optimization, train = self._optimize_generic(nX, nY, validation)
            if validation is not None and not append:
                Xf, Yf, nXf, nYf = X[train], Y[train], nX[train], nY[train]
            else:
                Xf, Yf, nXf, nYf = X, Y, nX, nY
            try:
                self.model = fit(nXf, nYf, self.regr, self.kernel, theta, noise=noise)
            except DaceFitError as e:
                raise DaceFitError(
                    "No positive-definite correlation matrix for the selected theta at the requested "
                    "noise; set noise / noise_bounds to regularize the committed fit."
                ) from e

        else:
            # optimizer=None -> freeze theta: a single GLS solve at the current theta and noise.
            self.model = fit(nX, nY, self.regr, self.kernel, self.theta, noise=self.noise)
            self.optimization = None
            Xf, Yf, nXf, nYf = X, Y, nX, nY

        # keep the raw (destandardized) training data so refit() can append to it
        self.model = {**self.model, "X": Xf, "Y": Yf, **stats, "nX": nXf, "nY": nYf}
        self.model["sigma2"] = np.square(sY) @ self.model["_sigma2"]

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

    def _optimize_generic(self, nX, nY, validation):
        """Run a generic optimizer over a DaceProblem; return ``(theta, noise, record, train_mask)``.

        With a validation mask the candidates train on the unmasked rows and are selected by
        held-out error (:class:`ValidationSelection`); otherwise the whole set trains and selection
        is maximum likelihood. With ``noise_bounds`` the nugget is a learned coordinate.
        """
        # noise_bounds set (nl/nu) -> learn the nugget; else fix it at self.noise. Mirrors theta.
        noise_kw = {"noise_bounds": (self.nl, self.nu)} if self.nl is not None else {"noise": self.noise}
        theta_bounds = self._theta_bounds_for_problem()

        if validation is not None:
            mask = np.asarray(validation, dtype=bool)
            if mask.shape[0] != nX.shape[0]:
                raise Exception("validation mask must have one entry per row of X.")
            n_held = int(mask.sum())
            if n_held == 0 or n_held == mask.shape[0]:
                raise Exception(
                    f"validation mask must hold out some rows but not all (got {n_held} of {mask.shape[0]})."
                )
            train = ~mask
            problem = DaceProblem(nX[train], nY[train], self.regr, self.kernel, theta_bounds, **noise_kw)
            callback = ValidationSelection(problem, nX[mask], nY[mask])
        else:
            train = np.ones(nX.shape[0], dtype=bool)
            problem = DaceProblem(nX, nY, self.regr, self.kernel, theta_bounds, **noise_kw)
            callback = Callback()

        x0 = self._encode_start(problem)
        res = self.optimizer.setup(problem, x0=x0, callback=callback).run()
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
        visited = getattr(self.optimizer, "visited", None)
        if visited is not None:
            record["models"] = [{"theta": problem.decode(x)[0]} for x in visited]
        return theta, noise, record, train

    def _encode_start(self, problem):
        """Encode the warm start ``self.theta`` (+ the nugget start ``self.noise`` when learned) as a log10 x0."""
        theta0 = np.broadcast_to(np.atleast_1d(np.asarray(self.theta, dtype=float)), (problem.p,))
        x0 = np.log10(np.maximum(theta0, 1e-12))
        if problem.learn_noise:
            # self.noise is the START of the learned nugget; clamp it into the noise bounds (a
            # 0.0 start lands on the lower bound) so its log10 is finite and inside the box.
            noise0 = np.clip(float(self.noise), problem._nlo, problem._nhi)
            x0 = np.append(x0, np.log10(noise0))
        return x0

    def refit(self, X, Y, optimize=True, validation=True):
        """Append new observations to the training data and re-fit, reusing the fitted theta.

        Takes only the *new* points, appends them to the data the model was last fit on, and
        re-fits on the combined set. Theta is always **seeded from the previous fit** (it barely
        moves when a few points are added); ``optimize`` then decides what happens from there:

        - ``optimize=True`` (default) -- **warm-start the model's optimizer** from the previous
          optimum (``self.optimizer`` -- the same method configured for the cold fit, now seeded at
          the reused theta), so theta adapts to the grown data at low cost.
        - ``optimize=False`` -- **freeze theta** at its previous value and just re-solve the kernel
          matrix on the grown data (a single GLS solve, no search). Cheapest, and it keeps theta
          from drifting on biased/adaptively-sampled data.

        The new points are always appended (that is what refit means); ``validation`` only steers
        the *search* (so it is moot when ``optimize=False``).

        Args:
            X: The new input points to add (only the additions, not the full set).
            Y: The target values corresponding to the new points ``X``.
            optimize: ``True`` warm-starts the theta search from the previous fit; ``False`` keeps
                theta fixed and only re-solves the kernel matrix on the grown data.
            validation: When ``optimize`` is ``True``, whether the *new* points are held out to
                select theta. ``True`` (default) makes the new points the validation set so theta
                is chosen by how well the old data predicts them; ``False`` re-fits by likelihood
                over all data. Ignored when ``optimize=False``.

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

        # reuse the previously optimized theta -- the warm start (optimize=True) or the frozen
        # value (optimize=False)
        self.theta = self.model["theta"]

        # the appended rows steer theta only when we actually search
        mask = None
        if validation and optimize:
            mask = np.zeros(X.shape[0], dtype=bool)
            mask[n_old:] = True

        # optimize gates WHETHER we search; the optimizer (the method) is the model's own
        # self.optimizer, now warm-started from the reused theta. optimize=False freezes theta
        # (no search), re-solving the kernel matrix only. The configured optimizer is restored
        # afterwards either way.
        configured = self.optimizer
        self.optimizer = configured if optimize else None
        try:
            self.fit(X, Y, validation=mask)
        finally:
            self.optimizer = configured

    def _val_error(self, model, nXv, nYv):
        """Root-mean-square error of a candidate fit on the held-out rows.

        A standalone scoring helper kept for tests and ad-hoc use; the production theta search
        now scores held-out candidates through :class:`~pysurrogate.dace.selection.ValidationSelection`
        (the optimizer's selection ``Callback``), which applies the same normalized-space formula.
        The candidate was fit on the training rows; here it predicts the held-out rows and
        the error is measured in *normalized* space. The held-out inputs and targets
        arrive already standardized with the training stats (the mask is applied to the
        same ``nX`` / ``nY`` the candidate trained on), so there is nothing to
        destandardize and the criterion is scale-free across outputs.

        Args:
            model: A fit() result (carries beta, gamma, theta, kernel, regr), fit on the
                training rows.
            nXv: Held-out inputs, standardized, shape ``(m, d)``.
            nYv: Held-out targets, standardized, shape ``(m,)`` or ``(m, q)``.

        Returns:
            The RMSE in normalized Y space.
        """
        nX = self.model["nX"]  # the training rows the candidate was fit on

        _F = model["regr"](nXv)
        _R = calc_kernel_matrix(nXv, nX, model["kernel"], model["theta"])

        # predicted normalized targets, compared directly to the normalized held-out Y
        _sYhat = _F @ model["beta"] + (model["gamma"].T @ _R.T).T

        nYv = nYv[:, None] if nYv.ndim == 1 else nYv
        return float(np.sqrt(np.mean(np.square(_sYhat - nYv))))

    def predict(self, _X, mse=False, grad=False):
        """Predict the mean, optionally the variance and the mean's gradient, in one pass.

        Mean and variance share the kernel matrix and the Cholesky solve, so computing
        them together is cheaper than two calls -- this is why both live on one method.

        Args:
            _X: Query inputs, shape ``(m, d)``.
            mse: Also return the predictive variance (kriging MSE), shape ``(m, 1)``. For a
                multi-output model the variance is *shared* across outputs (the kernel and
                theta are shared), so it stays ``(m, 1)`` regardless of the number of outputs.
            grad: Also return the gradient of the mean w.r.t. the query point. Single-output
                models return ``(m, d)``; multi-output models return ``(m, q, d)`` -- one
                gradient per output. This is what a gradient-based optimizer searches over.

        Returns:
            ``y`` (always), plus ``mse`` / ``grad`` when their flag is set (else None).
            When ``mse`` and ``grad`` are *both* set, ``mse_grad`` (the variance gradient,
            ``(m, d)``) is also returned -- it shares the variance and mean-gradient terms,
            so the extra cost is one triangular solve per point.
        """
        mX, sX, nX = self.model["mX"], self.model["sX"], self.model["nX"]
        mY, sY = self.model["mY"], self.model["sY"]
        regr, corr, theta = self.regr, self.kernel, self.model["theta"]
        beta, gamma = self.model["beta"], self.model["gamma"]

        # normalize the input given the mX and sX that was fitted before
        # NOTE: For the values to predict the _ is added to clarify its not the data fitted before
        _nX = (_X - mX) / sX

        # pairwise differences x_i - t_j, shape (m*n, d): shared by the kernel matrix here
        # and, when grad is requested, its gradient -- built once so predict(grad=True) does
        # not rebuild it. _F is the regression design, _R the kernel matrix reshaped (m, n).
        m, d = _nX.shape
        n = nX.shape[0]
        _D = np.repeat(_nX, n, axis=0) - np.tile(nX, (m, 1))
        _F = regr(_nX)
        _R = corr(_D, theta).reshape(m, n)

        # predict and destandardize
        _sY = _F @ beta + (gamma.T @ _R.T).T
        _Y = (_sY * sY) + mY

        _mse = None
        _mse_clamped = None  # mask of points whose variance was clamped (for mse_grad)
        if mse:
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
        want_mse_grad = mse and grad

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
                whose ``True`` rows are the held-out validation block and the rest are re-fit
                (mirrors ``fit``'s ``validation`` mask).

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
            pred = sub.predict(X[test], mse=True)
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
            partitioning = CrossvalidationPartitioning(k_folds=5)
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
