"""Tests for the generic optimizer contract: search, callback selection, early stop, never-raise."""

import numpy as np
import pytest

from pysurrogate.core.optimizer import Callback, Evaluation, Optimizer, Problem
from pysurrogate.core.sampling import LHS, Sampling
from pysurrogate.optimizer import LBFGS, Adam, PatternSearch, Restart


class Sphere(Problem):
    """f(x) = sum((x - center)^2) on a box; analytic gradient; minimum at ``center``."""

    def __init__(self, dim=3, center=None, lo=-5.0, hi=5.0, grad=True):
        self.center = np.zeros(dim) if center is None else np.asarray(center, float)
        self._lo = np.full(dim, lo)
        self._hi = np.full(dim, hi)
        self._grad = grad

    @property
    def bounds(self):
        return self._lo, self._hi

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        d = X - self.center
        f = np.sum(d**2, axis=1)
        g = 2.0 * d if self._grad else None
        return Evaluation(f=f, feasible=np.ones(len(X), bool), grad=g, info=list(X))


class Constrained(Problem):
    """Sphere where half-space x[0] < 0 is INFEASIBLE -- to test the never-raise contract."""

    @property
    def bounds(self):
        return np.array([-5.0, -5.0]), np.array([5.0, 5.0])

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        feasible = X[:, 0] >= 0.0
        f = np.where(feasible, np.sum(X**2, axis=1), np.inf)
        return Evaluation(f=f, feasible=feasible, grad=None, info=None)


# --- lifecycle: construct (user) -> setup (framework binds context) -> run -------------


def test_setup_then_run_matches_minimize_sugar():
    prob = Sphere(dim=3, center=[1.0, -1.0, 2.0])
    staged = LBFGS().setup(prob, x0=np.zeros(3)).run()
    oneshot = LBFGS().minimize(prob, x0=np.zeros(3))
    assert np.allclose(staged.x, oneshot.x, atol=1e-6)


def test_setup_returns_self_for_chaining():
    opt = PatternSearch()
    assert opt.setup(Sphere(dim=2)) is opt


def test_run_before_setup_raises():
    with pytest.raises(RuntimeError, match="before setup"):
        LBFGS().run()


def test_requires_x0_raises_when_missing():
    class LocalOnly(Optimizer):
        requires_x0 = True

        def _advance(self):
            self._emit(self.x0, float(self.problem(self.x0[None]).f[0]), None)
            return False

    with pytest.raises(ValueError, match="requires an explicit x0"):
        LocalOnly().setup(Sphere(dim=2))  # no x0
    # but it runs fine when x0 is supplied
    res = LocalOnly().minimize(Sphere(dim=2, center=[0.0, 0.0]), x0=np.zeros(2))
    assert res.x is not None


# --- advance(): one iteration at a time, and racing -------------------------------------


def test_advance_steps_and_has_next_terminates():
    opt = PatternSearch(tol=1e-3).setup(Sphere(dim=2, center=[1.0, -1.0]))
    steps = 0
    while opt.has_next():
        opt.advance()
        steps += 1
        assert steps < 10_000  # must terminate
    assert steps == opt.n_iter
    assert opt.result().f < 1e-2


def test_advance_is_noop_after_done():
    opt = PatternSearch(tol=1e-2).setup(Sphere(dim=2))
    opt.run()
    iters = opt.n_iter
    opt.advance().advance()
    assert opt.n_iter == iters  # no further work once done


def test_result_is_available_mid_run():
    opt = LBFGS(sampling=Sampling(6)).setup(Sphere(dim=2, center=[1.0, 1.0]))
    opt.advance()  # one local descent
    mid = opt.result()
    assert mid.x is not None and np.isfinite(mid.f)


def test_racing_two_optimizers_by_interleaving_advance():
    # drive two bound optimizers a step at a time and keep whoever's callback is ahead --
    # the orchestration `advance()` is meant to enable. Neither optimizer knows it is racing.
    prob = Sphere(dim=3, center=[2.0, -1.0, 0.5])
    a = PatternSearch(tol=1e-6).setup(prob, x0=np.zeros(3))
    b = LBFGS(sampling=Sampling(3)).setup(prob, x0=np.full(3, 4.0))
    while a.has_next() or b.has_next():
        if a.has_next():
            a.advance()
        if b.has_next():
            b.advance()
    winner = min((a, b), key=lambda o: o.result().f)
    assert winner.result().f < 1e-6


def test_lbfgs_reaches_minimum_with_gradient():
    res = LBFGS().minimize(Sphere(dim=4, center=[1.0, -2.0, 0.5, 3.0]), x0=np.zeros(4))
    assert np.allclose(res.x, [1.0, -2.0, 0.5, 3.0], atol=1e-3)
    assert res.f < 1e-6


def test_pattern_search_reaches_minimum_without_gradient():
    res = PatternSearch(tol=1e-6).minimize(Sphere(dim=3, center=[2.0, -1.0, 0.0], grad=False))
    assert np.allclose(res.x, [2.0, -1.0, 0.0], atol=1e-2)
    assert res.f < 1e-3


def test_lbfgs_falls_back_to_finite_difference_without_gradient():
    res = LBFGS().minimize(Sphere(dim=2, center=[1.0, 1.0], grad=False), x0=np.zeros(2))
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-2)


def test_default_start_is_box_center():
    # with no x0 the search starts at the center of the box; sphere centered there is solved
    res = PatternSearch().minimize(Sphere(dim=2, center=[0.0, 0.0]))
    assert res.f < 1e-2


def test_callback_selects_best_seen():
    cb = Callback()
    LBFGS().minimize(Sphere(dim=2, center=[1.0, 1.0]), x0=np.zeros(2), callback=cb)
    assert cb.best is not None
    assert np.allclose(cb.best, [1.0, 1.0], atol=1e-3)
    assert cb.best_f == cb.best_score  # MLE: selection score is the objective


def test_callback_early_stops_on_patience():
    # patience=1 stops after the first non-improving evaluation -> far fewer evals than free run
    free = PatternSearch().minimize(Sphere(dim=3), x0=np.full(3, 4.0))
    stopped = PatternSearch().minimize(Sphere(dim=3), x0=np.full(3, 4.0), callback=Callback(patience=1))
    assert stopped.n_evals < free.n_evals
    assert "callback" in stopped.message


def test_validation_style_callback_picks_by_its_own_score():
    # a callback may select by a score OTHER than the objective the optimizer descends:
    # here "score" rewards being near x=2 even though the objective is minimized at 0.
    class NearTwo(Callback):
        def score(self, x, f, info):
            return float(np.sum((x - 2.0) ** 2))

    cb = NearTwo()
    PatternSearch().minimize(Sphere(dim=2, center=[0.0, 0.0]), x0=np.full(2, 2.0), callback=cb)
    assert np.allclose(cb.best, [2.0, 2.0], atol=0.5)  # picked by validation score, not objective


def test_never_raises_on_infeasible_region():
    # the optimizer must traverse an infeasible half-space without throwing, and land feasible
    res = LBFGS(sampling=Sampling(4)).minimize(Constrained(), x0=np.array([-3.0, -3.0]))
    assert res.x is not None
    assert res.x[0] >= -1e-6  # converged into the feasible half
    assert np.isfinite(res.f)


# --- Sampling: start generation with forced points -------------------------------------


def test_sampling_includes_forced_points_and_size():
    pts = Sampling(5, method=LHS(), include=[[1.0, 2.0]]).sample((np.zeros(2), np.full(2, 10.0)))
    assert pts.shape == (5, 2)
    assert any(np.allclose(p, [1.0, 2.0]) for p in pts)  # the forced point is present


def test_sampling_injects_runtime_x0():
    pts = Sampling(4).sample((np.zeros(2), np.ones(2)), include=[[0.3, 0.7]])
    assert any(np.allclose(p, [0.3, 0.7]) for p in pts)  # x0 injected at sample time


def test_sampling_never_drops_a_forced_point():
    # more forced points than n -> all forced points still returned (n is a floor, not a cap here)
    pts = Sampling(1, include=[[0.1], [0.9]]).sample((np.zeros(1), np.ones(1)))
    assert len(pts) == 2


# --- Adam (vectorized population) -------------------------------------------------------


def test_adam_population_reaches_minimum():
    res = Adam(pop_size=12, steps=200, lr=0.3).minimize(Sphere(dim=2, center=[1.0, -1.0]))
    assert res.f < 1e-2


def test_adam_requires_gradient():
    with pytest.raises(ValueError, match="analytic gradient"):
        Adam(steps=1).minimize(Sphere(dim=2, grad=False))


def test_adam_fails_fast_in_setup_not_mid_search():
    # the gradient requirement is detected up front (in setup, like requires_x0), before any
    # _advance runs -- so the error surfaces from setup(), not after a partial search
    opt = Adam(steps=5)
    with pytest.raises(ValueError, match="analytic gradient"):
        opt.setup(Sphere(dim=2, grad=False))


def test_problem_has_grad_property():
    assert Sphere(dim=2, grad=True).has_grad is True
    assert Sphere(dim=2, grad=False).has_grad is False


# --- Restart: multi-start over any inner, with optional screen ---------------------------


def test_restart_runs_inner_from_each_sampled_start():
    prob = Sphere(dim=2, center=[2.0, -2.0])
    r = Restart(LBFGS(), Sampling(5, LHS())).minimize(prob)
    assert r.f < 1e-6  # best across the 5 polished starts


def test_restart_screen_polishes_only_the_best_k():
    prob = Sphere(dim=2, center=[1.0, 1.0])
    opt = Restart(LBFGS(), Sampling(20, LHS()), screen=3).setup(prob)
    opt.run()
    assert opt.n_iter == 3  # 20 sampled -> screened to 3 -> 3 inner polishes (iterations)
    assert opt.result().f < 1e-6


def test_restart_screen_uses_problem_screen_hook():
    calls = {"screen": 0}

    class Counting(Sphere):
        def screen(self, X):
            calls["screen"] += 1
            return super().screen(X)

    Restart(LBFGS(), Sampling(16, LHS()), screen=2).minimize(Counting(dim=2))
    assert calls["screen"] == 1  # the cheap screen is consulted once over the whole pool


def test_restart_wraps_pattern_search_too():
    # Restart composes with ANY inner, not just L-BFGS
    r = Restart(PatternSearch(tol=1e-5), Sampling(4, LHS())).minimize(Sphere(dim=2, center=[0.0, 3.0]))
    assert r.f < 1e-2


def test_infeasible_everywhere_yields_no_pick():
    class Empty(Problem):
        @property
        def bounds(self):
            return np.array([0.0]), np.array([1.0])

        def __call__(self, X):
            X = np.atleast_2d(X)
            return Evaluation(f=np.full(len(X), np.inf), feasible=np.zeros(len(X), bool))

    res = PatternSearch().minimize(Empty())
    assert res.x is None  # nothing feasible was ever selected


def test_visited_is_a_contract_attribute_present_on_every_optimizer():
    # `visited` is declared on the Optimizer base, so it is present (empty) before setup and
    # after a run on strategies that keep no trajectory -- never absent, no getattr needed.
    for opt in (LBFGS(), PatternSearch(), Adam()):
        assert opt.visited == []
    res_opt = LBFGS()
    res_opt.minimize(Sphere(dim=2))
    assert isinstance(res_opt.visited, list)  # still a list (empty) after a run


def test_boxmin_populates_the_visited_trajectory():
    from pysurrogate.optimizer import Boxmin

    box = Boxmin()
    box.minimize(Sphere(dim=2, center=[1.0, -1.0]), x0=np.array([0.5, 0.5]))
    # a pattern search records its trajectory on the shared contract attribute
    assert len(box.visited) > 1
    assert all(isinstance(x, np.ndarray) for x in box.visited)


def test_setup_resets_visited_between_runs():
    from pysurrogate.optimizer import Boxmin

    box = Boxmin()
    box.minimize(Sphere(dim=2), x0=np.array([0.5, 0.5]))
    first = len(box.visited)
    assert first > 0
    box.minimize(Sphere(dim=2), x0=np.array([0.5, 0.5]))  # fresh setup() must reset, not accumulate
    assert len(box.visited) == first  # deterministic run -> identical trajectory length, not doubled
