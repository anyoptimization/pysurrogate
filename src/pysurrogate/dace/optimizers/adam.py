"""Vectorized-Adam optimizer: a population of theta descended in lock-step."""

import numpy as np

from pysurrogate.dace.fit import DaceFitError, batch_obj_grad, fit
from pysurrogate.dace.optimizers.base import Optimizer, fit_feasible


class VectorizedAdam(Optimizer):
    """Adam descent over a population of theta, all starts advancing in lock-step.

    A multi-start gradient method tailored to batching. Where ``LBFGS`` cannot be
    batched across restarts (each scipy run has its own line search and converges in a
    different number of steps, so they desync), Adam with a *fixed* step schedule keeps
    the whole population synchronized: every iteration is one ``batch_obj_grad`` call --
    a single stacked ``(J, n, n)`` Cholesky -- evaluating all ``J`` starts at once.

    The search runs in log10(theta) space (length-scales are multiplicative, and the
    lower bound is floored away from zero, like ``LBFGS``'s restart sampling). Init
    coverage is the main thing that decides whether the population beats ``Boxmin``,
    especially for higher-dimensional (ARD) theta, so the starts are *screened*:
    ``n_candidates`` log-uniform candidates are evaluated in a single batched objective
    call (cheap -- that is what the batching is for) and the best ``pop_size`` of them
    become the descent starts. The model's existing (warm) theta is injected into that
    candidate set, so a good previous theta is simply kept and a poor one is replaced by
    better random draws -- without enlarging the set. Each start then contributes its
    best-seen theta to the final pick, which goes through the shared ``_select`` -- by
    maximum likelihood, or by held-out error when a validation set is given (which is the
    regularizing lever against over-fitting theta, and what ``refit`` supplies).

    Args:
        pop_size: Number of theta descended together (the population; the best of the
            screened candidates).
        n_candidates: How many candidates to screen down to the population. ``None``
            (default) uses ``10 * pop_size``; any value is clamped up to at least
            ``pop_size``. The existing theta takes one candidate slot, so it always competes.
        steps: Maximum lock-step Adam iterations (each one batched fit); the search stops
            early once the population's best objective stalls, so this is a ceiling.
        lr: Adam step size in log10(theta) space.
        random_state: Seed for the candidate sampling, so runs are reproducible.
    """

    def __init__(self, pop_size=8, n_candidates=None, steps=50, lr=0.1, random_state=0):
        super().__init__()
        self.pop_size = pop_size
        self.n_candidates = n_candidates
        self.steps = steps
        self.lr = lr
        self.random_state = random_state

    def optimize(self, dace, validation=None):
        """Descend the population, then select the best theta; return ``(best_model, optimization)``."""
        nX, nY = dace.model["nX"], dace.model["nY"]
        regr, kernel = dace.regr, dace.kernel

        # bring theta and bounds to a common 1d shape (theta drives the dimension)
        theta0 = np.atleast_1d(np.array(dace.theta, dtype=float))
        lo = np.broadcast_to(np.atleast_1d(np.asarray(dace.tl, dtype=float)), theta0.shape)
        up = np.broadcast_to(np.atleast_1d(np.asarray(dace.tu, dtype=float)), theta0.shape)
        p = theta0.shape[0]

        # log10 space; floor the lower bound away from zero (thetaL defaults to 0.0, and
        # log10(0) = -inf would poison every sample) and keep up >= lo.
        lo_pos = np.maximum(lo, 1e-12)
        up_pos = np.maximum(up, lo_pos)
        llo, lup = np.log10(lo_pos), np.log10(up_pos)

        # screened init: log-uniform candidates with the existing theta dropped into one
        # slot (so it competes without enlarging the set), evaluated in ONE batched
        # objective; the best pop_size become the descent starts.
        rng = np.random.default_rng(self.random_state)
        n_cand = max(
            self.n_candidates if self.n_candidates is not None else 10 * self.pop_size,
            self.pop_size,
        )
        cand = rng.uniform(llo, lup, size=(n_cand, p))
        cand[0] = np.log10(np.clip(theta0, lo_pos, up_pos))
        screen, _, _ = batch_obj_grad(
            nX,
            nY,
            regr,
            kernel,
            np.clip(10.0**cand, lo_pos, up_pos),
            noise=dace.noise,
            with_grad=False,
        )
        L = cand[np.argsort(screen)[: self.pop_size]]  # (J, p)
        J = L.shape[0]

        # standard Adam state and the per-start best-seen theta (the candidate pool)
        m = np.zeros((J, p))
        v = np.zeros((J, p))
        b1, b2, eps = 0.9, 0.999, 1e-8
        best_obj = np.full(J, np.inf)
        best_theta = np.clip(10.0**L, lo_pos, up_pos)

        # ``steps`` is a ceiling: stop early once the population's best objective has not
        # improved for ``patience`` iterations. The expensive part is steps x pop batched
        # fits, so ending as soon as the population converges is the main speed lever --
        # and the whole population stays lock-step, so batching is unaffected.
        patience = max(5, self.steps // 5)
        best_overall, stall, step = np.inf, 0, 0
        for step in range(1, self.steps + 1):
            theta = np.clip(10.0**L, lo_pos, up_pos)
            obj, g_theta, feasible = batch_obj_grad(nX, nY, regr, kernel, theta, noise=dace.noise)

            improved = obj < best_obj
            best_obj = np.where(improved, obj, best_obj)
            best_theta = np.where(improved[:, None], theta, best_theta)

            cur = float(best_obj.min())
            if np.isfinite(cur) and (not np.isfinite(best_overall) or cur < best_overall - 1e-6 * max(abs(cur), 1.0)):
                best_overall, stall = cur, 0
            else:
                stall += 1
            if stall >= patience:
                break

            # chain rule to log10 space: d theta / d log10(theta) = theta * ln(10).
            # Infeasible starts get a zero gradient so they simply hold (and are dropped
            # at selection if they never reached a feasible theta).
            g_log = g_theta * theta * np.log(10.0)
            g_log[~feasible] = 0.0

            m = b1 * m + (1 - b1) * g_log
            v = b2 * v + (1 - b2) * g_log**2
            mhat = m / (1 - b1**step)
            vhat = v / (1 - b2**step)
            L = np.clip(L - self.lr * mhat / (np.sqrt(vhat) + eps), llo, lup)

        # build one model per start's best-seen theta (strict fits, like the other
        # searches); skip any that turned out infeasible. The final pick is the shared
        # selection (MLE, or held-out error when a validation set is given).
        models = []
        for t in best_theta:
            try:
                models.append(fit(nX, nY, regr, kernel, t, noise=dace.noise))
            except DaceFitError:
                pass
        if not models:
            # no feasible candidate anywhere -> fall back to a feasibility-guaranteed fit
            # at the warm theta (honoring the model's max_noise policy), like a cold start.
            _, model = fit_feasible(dace, theta0, relocate=True)
            models = [model]

        best = self._select(dace, models, validation)
        optimization = {"best": best, "models": models, "pop_size": J, "steps": step}
        return best, optimization
