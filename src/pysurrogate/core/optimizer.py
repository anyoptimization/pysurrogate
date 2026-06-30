"""Abstract optimization contract: Problem (what to minimize), Optimizer (how to search), Callback (select / stop)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Evaluation:
    """The outcome of evaluating a :class:`Problem` at a population of candidate vectors.

    Attributes:
        f: Objective value per candidate, shape ``(J,)``; ``+inf`` where infeasible.
        feasible: Boolean mask, shape ``(J,)`` -- ``False`` for candidates the problem could
            not score (e.g. a non-positive-definite fit), which carry ``f = +inf``.
        grad: Analytic objective gradient, shape ``(J, p)``, or ``None`` when the problem
            exposes no gradient (a derivative-free search ignores it either way).
        info: Opaque per-candidate payload, length ``J``, or ``None``. A surrogate problem
            puts the fitted model here so a :class:`Callback` can re-score the candidate
            (e.g. on held-out data) without re-fitting.
    """

    f: np.ndarray
    feasible: np.ndarray
    grad: np.ndarray | None = None
    info: list | None = None


class Problem(ABC):
    """A bounded minimization problem over a box in ``R^p`` -- the thing an Optimizer searches.

    Backend-free by design: a surrogate supplies a concrete problem that maps a parameter
    vector to a model fit and its objective, but the optimizer only ever sees ``bounds`` and
    ``__call__``. Parameters that are conceptually different -- a kernel length-scale and a
    noise nugget -- are simply different *coordinates of the same vector*, each with its own
    bound. So "optimize theta and noise jointly" is not a special case; it is one longer
    vector whose extra coordinate happens to be the nugget. Coordinates live in whatever space
    the problem chooses (typically ``log10`` for positive scales like length-scales and noise).
    """

    @property
    @abstractmethod
    def bounds(self):
        """The hard search box as ``(lo, hi)``, each an array of shape ``(p,)``.

        These are the constraints the optimizer must respect; a coordinate may be unbounded
        (``+/- inf``) for a free search. Use :attr:`sampling_bounds` -- not these -- to seed
        starting points or scale steps, since an infinite box cannot be sampled.
        """

    @property
    def sampling_bounds(self):
        """A *finite* ``(lo, hi)`` region for seeding starts and scaling steps.

        Defaults to :attr:`bounds`, so a fully bounded problem is unchanged. A problem with
        infinite hard bounds overrides this to clamp the infinities to a finite window: the
        optimizer seeds its starts inside that window but the local descent is still free to
        leave it (the hard :attr:`bounds` are what constrain it). Splitting the two is what
        makes an unbounded search possible without trying to sample an infinite box.
        """
        return self.bounds

    @abstractmethod
    def __call__(self, X):
        """Evaluate the objective at a population of candidates -- without ever raising.

        Args:
            X: Candidate vectors, shape ``(J, p)`` (``J`` may be 1), one row per candidate.

        Returns:
            An :class:`Evaluation` carrying the objective, the feasibility mask, an optional
            analytic gradient and optional per-candidate ``info``.

        An infeasible candidate is reported with ``feasible=False`` and ``f=+inf`` rather than
        an exception, so the search steps away from it instead of crashing. This is the single
        contract that makes the whole layer robust to ill-conditioning.
        """

    @property
    def n_var(self):
        """Number of optimized coordinates ``p`` (length of the bounds)."""
        lo, _ = self.bounds
        return len(np.atleast_1d(lo))

    def screen(self, X):
        """Cheap objective-only evaluation of a population, for ranking candidates.

        Used by :class:`~pysurrogate.optimizer.restart.Restart` to filter a large sampled pool
        down to a few promising starts before the expensive polish. The default just delegates to
        :meth:`__call__` and returns its objectives; override to skip the costly part (e.g. the
        gradient) so the screen is genuinely cheaper than a full evaluation.

        Args:
            X: Candidate vectors, shape ``(J, p)``.

        Returns:
            Objective values, shape ``(J,)`` (``+inf`` where infeasible).
        """
        return self(X).f


class Callback:
    """Observer of a search that doubles as the *selector* and the *stopping rule*.

    The optimizer calls the callback at every evaluated, feasible candidate with the point,
    its objective and the problem's ``info`` for that point. One object decides two things --
    which is the whole point, because in this design selection and termination are the same
    mechanism:

    - **selection** -- it keeps the best candidate seen *by its own score*, which need not be
      the objective the optimizer descends. Maximum-likelihood selection scores by ``f``;
      validation selection re-scores from ``info`` (the fitted model) on held-out data; a MAP
      selection adds a prior term. The optimizer stays ignorant of which is in use.
    - **termination** -- it returns ``True`` to ask the optimizer to stop, e.g. after
      ``patience`` consecutive non-improving evaluations. Early-stopping on a *validation*
      score is itself a regularizer, so this hook is how a cheap fit avoids over-fitting.

    The base tracks the best by an overridable :meth:`score` (default: the objective) and stops
    after ``patience`` stale steps; ``patience=None`` never stops early (pure selection).
    Subclass and override :meth:`score` for validation or MAP selection.
    """

    def __init__(self, patience=None):
        self.patience = patience
        self.best = None  # best parameter vector seen
        self.best_f = np.inf  # the optimizer's objective at ``best``
        self.best_score = np.inf  # the *selection* score at ``best`` (== best_f for MLE)
        self.best_info = None  # the problem payload at ``best`` (e.g. the fitted model)
        self.n_seen = 0
        self._stale = 0

    def score(self, x, f, info):
        """Selection score for a candidate -- lower is better. Override for validation / MAP.

        Args:
            x: The candidate parameter vector.
            f: The optimizer's objective at ``x`` (what the search descends).
            info: The problem payload at ``x`` (e.g. the fitted model), or ``None``.

        Returns:
            The score this callback selects by. The default returns ``f`` (maximum-likelihood
            selection); a validation callback returns a held-out error computed from ``info``.
        """
        return f

    def __call__(self, x, f, info):
        """Record a candidate and report whether the search should stop.

        Args:
            x: The evaluated candidate vector.
            f: Its objective value.
            info: Its problem payload (or ``None``).

        Returns:
            ``True`` to request early termination (``patience`` exhausted), else ``False``.
        """
        self.n_seen += 1
        s = self.score(x, f, info)
        if s < self.best_score:
            self.best, self.best_f, self.best_score, self.best_info, self._stale = x, f, s, info, 0
        else:
            self._stale += 1
        return self.patience is not None and self._stale >= self.patience


@dataclass
class Result:
    """What :meth:`Optimizer.minimize` returns -- the callback's pick plus search metadata.

    Attributes:
        x: The selected parameter vector (the callback's best), or ``None`` if nothing
            feasible was ever evaluated.
        f: The objective at ``x``.
        info: The problem payload at ``x`` (e.g. the fitted model), or ``None``.
        n_evals: Total candidate evaluations the search performed.
        message: Human-readable note on why the search stopped (converged / patience / budget).
    """

    x: np.ndarray | None
    f: float
    info: object = None
    n_evals: int = 0
    message: str = ""


class Optimizer(ABC):
    """A search strategy over a bounded :class:`Problem` -- generic, with no surrogate knowledge.

    Its lifecycle separates the three things that happen at different times and are owned by
    different parties -- which is why a single ``minimize(problem, x0, ...)`` call felt wrong:

    1. **construction** -- the *user* picks the strategy and its hyperparameters
       (``LBFGS(n_restarts=4)``). No problem, no ``x0``: the user does not have them yet.
    2. :meth:`setup` -- the *framework* binds the runtime context it (not the user) knows: the
       :class:`Problem`, the optional ``x0`` (e.g. a model's current theta for a warm refit) and
       the selection :class:`Callback`. The ``requires_x0`` check and any problem-dependent
       preparation (the :meth:`_setup` hook -- a screen, a seeded population) happen here.
    3. :meth:`run` -- execute, returning the callback's pick as a :class:`Result`.

    The unit of execution is :meth:`advance` -- *one iteration* of the search. :meth:`run`
    just loops ``advance`` until :meth:`has_next` is false. Exposing the single step is what
    makes external orchestration possible: a driver can interleave ``advance`` across several
    bound optimizers, compare their ``callback.best_score`` between steps, and drop the laggards
    -- i.e. **race** them -- without any optimizer knowing it is in a race. :meth:`minimize` is
    one-shot sugar for ``setup(...).run()``.

    Concrete strategies implement :meth:`_advance` (one iteration) and optionally :meth:`_setup`
    (problem-dependent preparation), reading the bound ``self.problem`` / ``self.x0`` /
    ``self.callback``. What "one iteration" means is the optimizer's choice -- a single poll for
    pattern search, one gradient step for a population method, one local descent for L-BFGS.
    They report every feasible candidate via :meth:`_emit` and never pick a best themselves --
    selection is the callback's job.

    Class attributes:
        requires_x0: ``True`` for an optimizer that cannot run without an explicit start (a pure
            local refiner). :meth:`setup` then raises when ``x0`` is ``None`` instead of guessing
            one. Global / population strategies leave it ``False``.
    """

    requires_x0 = False

    def __init__(self):
        self.problem = None
        self.x0 = None
        self.callback = None
        self.is_setup = False
        self.is_done = False
        self.n_iter = 0
        self.n_evals = 0
        self.message = ""

    def setup(self, problem, x0=None, callback=None):
        """Bind the runtime context for one run; return ``self`` so calls can chain.

        Called by the fitting framework, not the end user -- it supplies the problem, the
        optional warm start and the selection callback that the user (who only constructed the
        optimizer) does not have. Resets the iteration state, so a constructed optimizer is a
        reusable spec that ``setup`` turns into a fresh run.

        Args:
            problem: The bounded :class:`Problem` to minimize.
            x0: Optional warm start, shape ``(p,)``. A local optimizer starts from it; a
                global/population one ignores it. Required only when :attr:`requires_x0` is set.
            callback: The selector/stopper. ``None`` uses a plain maximum-likelihood
                :class:`Callback` (selects the lowest objective seen, no early stopping).

        Returns:
            ``self``, bound and ready for :meth:`advance` / :meth:`run`.

        Raises:
            ValueError: If ``x0`` is ``None`` but the optimizer sets :attr:`requires_x0`.
        """
        if self.requires_x0 and x0 is None:
            raise ValueError(f"{type(self).__name__} requires an explicit x0 (a starting point); none was given.")
        self.problem = problem
        self.x0 = None if x0 is None else np.atleast_1d(np.asarray(x0, float))
        self.callback = callback if callback is not None else Callback()
        self.is_done = False
        self.n_iter = self.n_evals = 0
        self.message = ""
        self._setup()
        self.is_setup = True
        return self

    def has_next(self):
        """Whether another :meth:`advance` would do work (set up and not yet finished)."""
        return self.is_setup and not self.is_done

    def advance(self):
        """Advance the search by exactly **one iteration**; return ``self``.

        The steppable primitive an external driver loops -- or interleaves across several
        optimizers to race them. A no-op once :meth:`has_next` is false. The optimizer marks
        itself done when its :meth:`_advance` reports completion or when the callback asks to stop
        (via :meth:`_emit`).

        Returns:
            ``self``, so steps can be chained or polled.

        Raises:
            RuntimeError: If called before :meth:`setup`.
        """
        if not self.is_setup:
            raise RuntimeError("advance() called before setup(); call setup(problem, ...) first.")
        if self.is_done:
            return self
        self.n_iter += 1
        if self._advance() is False:
            self.is_done = True
        return self

    def run(self):
        """Loop :meth:`advance` to completion and return the callback-selected :class:`Result`."""
        if not self.is_setup:
            raise RuntimeError("run() called before setup(); call setup(problem, ...) first.")
        while self.has_next():
            self.advance()
        return self.result()

    def result(self):
        """The callback's current pick as a :class:`Result` -- valid at any point during a run."""
        cb = self.callback
        return Result(x=cb.best, f=float(cb.best_f), info=cb.best_info, n_evals=self.n_evals, message=self.message)

    def minimize(self, problem, x0=None, callback=None):
        """One-shot convenience: ``setup(problem, x0, callback).run()``."""
        return self.setup(problem, x0=x0, callback=callback).run()

    def _emit(self, x, f, info):
        """Report one feasible candidate to the callback; flips :attr:`is_done` if it asks to stop.

        Returns:
            ``True`` if the callback requested an early stop (the caller should also bail out of
            the current iteration), else ``False``.
        """
        if self.callback(np.asarray(x, float), float(f), info):
            self.is_done = True
            self.message = "stopped early (callback)"
            return True
        return False

    def _setup(self):
        """Optional hook for problem-dependent preparation, run at the end of :meth:`setup`.

        Reads the freshly bound ``self.problem`` / ``self.x0`` and stashes whatever the run will
        reuse (a candidate screen, a seeded population, cached bounds). The base does nothing.
        """

    @abstractmethod
    def _advance(self):
        """Perform one iteration of the search over ``self.problem``.

        Evaluate the problem (in batches where possible), report each feasible candidate via
        ``self._emit(x, f, info)``, and bail out of the iteration if ``_emit`` returns ``True``.
        Return ``False`` when the search is complete (converged / budget exhausted); any other
        return (including ``None``) means "more iterations remain". Selection is *not* the
        optimizer's job -- the callback owns it.
        """
