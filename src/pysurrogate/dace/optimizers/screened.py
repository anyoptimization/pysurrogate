"""Screened-start L-BFGS: a batched cheap screen feeding a few analytic-gradient polishes."""

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from pysurrogate.dace.fit import DaceFitError, batch_obj_grad, fit
from pysurrogate.dace.optimizers.base import Optimizer, fit_feasible

# a failed fit returns a large but finite penalty (np.inf breaks L-BFGS-B's line search),
# so the search steps away from an infeasible theta instead of stalling.
_INFEASIBLE = 1e25


def _logspace_bounds(dace):
    """Theta start and bounds brought to a common 1d shape, with the lower bound floored.

    Returns the warm theta, the positive (floored) bounds and their log10, so the search
    can run in log10 space -- length-scales vary multiplicatively, and Dace's default
    thetaL is 0.0, whose log10 = -inf would poison every sample.

    Args:
        dace: The model being fit (supplies ``theta`` and the bounds ``tl`` / ``tu``).

    Returns:
        ``(theta0, lo_pos, up_pos, llo, lup)`` -- the start and bounds in theta space
        (``lo_pos`` floored away from zero) and the log10 bounds.
    """
    theta0 = np.atleast_1d(np.array(dace.theta, dtype=float))
    lo = np.broadcast_to(np.atleast_1d(np.asarray(dace.tl, dtype=float)), theta0.shape)
    up = np.broadcast_to(np.atleast_1d(np.asarray(dace.tu, dtype=float)), theta0.shape)
    lo_pos = np.maximum(lo, 1e-12)
    up_pos = np.maximum(up, lo_pos)
    return theta0, lo_pos, up_pos, np.log10(lo_pos), np.log10(up_pos)


def _unit_samples(sampler, n, p, rng):
    """Generate ``n`` points in the unit hypercube ``[0, 1]^p`` for the screen.

    Args:
        sampler: ``"lhs"`` (a seeded Latin Hypercube -- one stratified, jittered point per
            stratum per dimension, so the candidates fill the box evenly), ``"random"``
            (plain uniform), or a callable ``f(n, p) -> (n, p)`` array in ``[0, 1]`` (e.g. a
            ``pysampling`` sampler for Halton / Sobol / maximin-LHS / Riesz). A space-filling
            sampler finds the likelihood basin more reliably per candidate than uniform
            random, especially for higher-dimensional (ARD) theta.
        n: Number of points.
        p: Dimension (the theta length).
        rng: Seeded generator for the built-in samplers (reproducibility).

    Returns:
        Points of shape ``(n, p)`` in ``[0, 1]``.
    """
    if callable(sampler):
        return np.asarray(sampler(n, p), dtype=float).reshape(n, p)
    if sampler == "lhs":
        u = np.empty((n, p))
        for k in range(p):
            u[:, k] = (rng.permutation(n) + rng.uniform(size=n)) / n
        return u
    if sampler == "random":
        return rng.uniform(size=(n, p))
    raise ValueError(f"sampler must be 'lhs', 'random' or a callable, got {sampler!r}")


def _farthest_point(cand, k):
    """Indices of ``k`` spread-out rows: the first, then greedily the farthest from those chosen.

    The candidates arrive sorted best-first, so taking the top ``k`` by objective collapses
    the starts into a single basin (measured: that does not help). Farthest-point selection
    keeps the best candidate but spreads the rest across the log10 space, so the polishes
    explore distinct basins of the multi-modal likelihood. Returning indices lets a paired
    per-candidate nugget ride along with the selected theta.

    Args:
        cand: Candidate starts sorted best-first, shape ``(N, p)``.
        k: How many to select.

    Returns:
        The selected row indices, shape ``(min(k, N),)``.
    """
    if len(cand) <= k:
        return np.arange(len(cand))
    chosen = [0]
    while len(chosen) < k:
        dist = np.min([np.sum((cand - cand[c]) ** 2, axis=1) for c in chosen], axis=0)
        dist[chosen] = -1.0
        chosen.append(int(np.argmax(dist)))
    return np.array(chosen)


class ScreenedLBFGS(Optimizer):
    """Batched cheap screen, then a few analytic-gradient L-BFGS polishes -- fast *and* global-ish.

    The fast theta optimizer. It splits the search into the two parts batching is good at
    and the part it is not:

    1. **Screen (batched, no Cholesky-per-candidate Python loop).** ``n_cand`` space-filling
       candidates (Latin Hypercube by default) -- plus the model's own (warm) theta in one
       slot, so a refit's existing optimum always competes -- are ranked in a *single*
       ``batch_obj_grad`` call with the gradient turned off. This is the cheap part: one
       stacked Cholesky and no gradient block (the dominant cost), so it covers the whole box
       for a fraction of one ordinary fit. At larger ``n`` the screen runs on a row
       *subsample* (``screen_rows``) -- the basin's location barely needs every row, so this
       cuts the screen cost where it would otherwise grow as ``n_cand * O(n^3)``.
    2. **Polish (gradient, the few O(n^3) fits that matter).** The best ``k_starts`` screened
       candidates -- spread by farthest-point so they do not collapse into one basin -- seed
       ``k_starts`` L-BFGS-B descents with the *exact* theta-gradient, on the *full* data.
       This is the fewest expensive fits of any global method: the screen already found good
       basins, so each descent converges in a handful of steps.

    Measured against the derivative-free ``Boxmin`` (the original Dace search) this reaches
    a *better* surrogate (lower held-out error -- ``Boxmin`` underfits smooth responses) at
    equal or better wall-clock; the default (``k_starts=1``, LHS screen, subsampled) is
    ~2-3x faster than ``Boxmin``. Against multi-start ``LBFGS`` / ``VectorizedAdam`` it
    matches their quality for a fraction of the fits, because the cheap screen replaces their
    expensive exploration. It is ``refit``'s default warm refiner; ``Boxmin`` is kept
    available for an exact MATLAB-Dace trajectory.

    The search is pure maximum likelihood; the regularizer against theta over-fitting on a
    sparse design is held-out selection (pass a validation set -- what ``refit`` does -- so
    the final pick minimizes held-out error over every visited theta).

    **Learned nugget (``noise='auto'``).** When the model is built with ``Dace(noise='auto')``
    the nugget (observation-noise term) is learned *jointly with theta* by the same marginal
    likelihood: the screen scores every theta candidate at a log grid of nuggets over
    ``nugget_range``, so one batched screen ranks the whole ``theta x nugget`` grid and each
    theta keeps its marginal-likelihood-optimal nugget (which is then fixed through that
    theta's polish). The marginal likelihood drives the nugget to ~0 for a deterministic
    (noiseless) objective and to the right level for a noisy one -- the proper, problem-adaptive
    way to get calibrated predictive variance, far stronger than biasing theta. With a fixed
    numeric ``noise`` the nugget is simply that value, unchanged.

    Args:
        n_cand: Number of candidates to screen (the warm theta takes one slot). Larger covers
            the box better; the screen is cheap, so this is generous by default.
        k_starts: How many screened basins to polish with L-BFGS. ``1`` (default) is the
            fastest and already strong because the screen finds a good basin; raise it for a
            multi-modal likelihood where a single descent may miss the global optimum.
        sampler: How the candidates fill the box -- ``"lhs"`` (default, a seeded Latin
            Hypercube), ``"random"`` (uniform), or a callable ``f(n, p) -> (n, p)`` in
            ``[0, 1]`` (e.g. ``lambda n, p: pysampling.sample("halton", n, p)``).
        screen_rows: Cap on the number of rows used for the (basin-finding) screen; the
            polish always uses every row. ``None`` screens on all rows. The default trims the
            screen cost at larger ``n`` with no measured loss of quality, since the screen
            only ranks theta *scales*. Ignored when the design has fewer rows than the cap.
        nugget_range: ``(lo, hi)`` noise-to-signal range for the learned-nugget screen, used
            only when the model's ``noise`` is ``'auto'``. The screen samples log-uniform in
            this range; ``lo`` doubles as the near-zero floor a deterministic fit selects.
        diversify: Whether to spread the ``k_starts`` by farthest-point (default True). False
            takes the top ``k_starts`` by objective -- only sensible when ``k_starts`` is 1.
        maxfun: Cap on objective evaluations per L-BFGS descent (each an O(n^3) fit); a
            safety margin, since a screened warm start converges well inside it.
        random_state: Seed for the candidate sampling, so runs are reproducible.
    """

    supports_auto_noise = True

    def __init__(
        self,
        n_cand=48,
        k_starts=1,
        sampler="lhs",
        screen_rows=64,
        nugget_range=(1e-6, 0.3),
        diversify=True,
        maxfun=60,
        random_state=0,
    ):
        super().__init__()
        self.n_cand = n_cand
        self.k_starts = k_starts
        self.sampler = sampler
        self.screen_rows = screen_rows
        self.nugget_range = nugget_range
        self.diversify = diversify
        self.maxfun = maxfun
        self.random_state = random_state

    def optimize(self, dace, validation=None):
        """Screen, polish the best basins, then select; return ``(best_model, optimization)``."""
        nX, nY = dace.model["nX"], dace.model["nY"]
        regr, kernel = dace.regr, dace.kernel
        theta0, lo_pos, up_pos, llo, lup = _logspace_bounds(dace)
        p = theta0.shape[0]
        rng = np.random.default_rng(self.random_state)
        auto_noise = isinstance(dace.noise, str) and dace.noise == "auto"

        # screen candidates: space-filling theta in log10 space, with the warm theta dropped
        # into slot 0 so it competes without enlarging the set.
        n_cand = max(self.n_cand, 1)
        cand = llo + (lup - llo) * _unit_samples(self.sampler, n_cand, p, rng)
        cand[0] = np.log10(np.clip(theta0, lo_pos, up_pos))

        # nugget grid: a single fixed value, or (auto) a log grid over nugget_range so each
        # theta is scored at *every* nugget level (not one random partner, which would
        # confound theta and noise). The grid includes the near-zero floor so a deterministic
        # objective can pick "no nugget".
        lo_nz, hi_nz = self.nugget_range
        nz_grid = np.geomspace(lo_nz, hi_nz, 8) if auto_noise else np.array([float(dace.noise)])

        # the screen only locates the basin, so it can run on a row subsample at larger n;
        # the polish below always uses the full data. One batched objective-only call (the
        # cheap part -- no gradient, one stacked Cholesky) ranks the whole theta x nugget grid.
        sX, sY = nX, nY
        if self.screen_rows is not None and self.screen_rows < nX.shape[0]:
            sub = rng.choice(nX.shape[0], size=self.screen_rows, replace=False)
            sX, sY = nX[sub], nY[sub]
        screen_theta = np.clip(10.0**cand, lo_pos, up_pos)
        grid_theta = np.repeat(screen_theta, len(nz_grid), axis=0)  # (n_cand*G, p)
        grid_nz = np.tile(nz_grid, n_cand)  # (n_cand*G,)
        obj, _, _ = batch_obj_grad(sX, sY, regr, kernel, grid_theta, noise=grid_nz, with_grad=False)
        obj = obj.reshape(n_cand, len(nz_grid))
        best_nz = nz_grid[obj.argmin(axis=1)]  # each theta's marginal-likelihood-optimal nugget
        theta_obj = obj.min(axis=1)  # ... and the obj it achieves there

        order = np.argsort(theta_obj)  # best theta first
        cand, best_nz = cand[order], best_nz[order]
        sel = _farthest_point(cand, self.k_starts) if self.diversify else np.arange(min(self.k_starts, len(cand)))
        starts, start_nz = cand[sel], best_nz[sel]
        bounds = list(zip(llo, lup))

        # for held-out selection we rank every theta the descents visit (each is already a
        # fit, so recording it costs nothing); for the MLE path we never look, so don't pay.
        record = validation is not None
        history = []

        def make_fun(nz):
            # descend theta in log10 space (chain rule d theta / d log10 theta = theta * ln 10)
            # at this start's fixed nugget; reuse batch_obj_grad on a single theta so screen
            # and polish share one objective. An infeasible theta returns a finite penalty.
            def fun(u):
                theta = np.clip(10.0 ** np.asarray(u, dtype=float), lo_pos, up_pos)
                o, g, feasible = batch_obj_grad(nX, nY, regr, kernel, theta[None], noise=nz)
                if not feasible[0] or not np.isfinite(o[0]):
                    return _INFEASIBLE, np.zeros_like(theta)
                if record:
                    try:
                        history.append(fit(nX, nY, regr, kernel, theta, noise=nz))
                    except DaceFitError:
                        pass
                return float(o[0]), g[0] * theta * np.log(10.0)

            return fun

        # polish each screened start at its paired nugget; fit_feasible applies the shared
        # feasibility / max_noise safety net at the converged theta and nugget.
        optima = []
        for u0, nz in zip(starts, start_nz):
            res = minimize(
                make_fun(nz),
                u0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxfun": self.maxfun, "gtol": 1e-5},
            )
            _, model = fit_feasible(dace, np.clip(10.0**res.x, lo_pos, up_pos), relocate=True, noise=nz)
            optima.append(model)
            if record:
                history.append(model)

        # MLE selects the best polished optimum; held-out selection ranks the full visited
        # history (so the pick is meaningful even with a single descent), via the shared _select.
        models = history if record else optima
        best = self._select(dace, models, validation)
        optimization = {
            "best": best,
            "models": models,
            "k_starts": len(starts),
            "n_cand": len(cand),
            "noise": best.get("noise"),
        }
        return best, optimization
