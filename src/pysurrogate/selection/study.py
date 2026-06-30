"""Function-sampling study: benchmark surrogate models on a known function over a box domain."""

import copy

import numpy as np

from pysurrogate.dace import Exponential, Gaussian, Matern, RationalQuadratic
from pysurrogate.models import KNN, RBF, SVR, InverseDistanceWeighting, Kriging, SimpleMean
from pysurrogate.selection.factory import as_named, cartesian
from pysurrogate.selection.metrics import POINT_METRICS, evaluate, metric_sort_key


def default_kriging():
    """Return the Kriging kernel zoo -- the *uncertainty-providing* surrogates.

    Every model here returns a predictive variance (and gradient), so this is the fleet to select
    over when the chosen model must drive an acquisition function (EI/UCB) -- unlike the broader
    :func:`default_models`, whose Mean/KNN/IDW/SVR/RBF baselines report no ``sigma``. Kernels:
    Gaussian, Exponential, the Matern family (nu=1.5/2.5; nu=0.5 equals Exponential), and
    Rational-Quadratic swept over alpha (heavier tails as alpha shrinks).
    """
    return cartesian(
        Kriging,
        corr={
            "gauss": Gaussian(),
            "exp": Exponential(),
            **{f"matern[{nu}]": Matern(nu) for nu in (1.5, 2.5)},
            **{f"rq[{a}]": RationalQuadratic(a) for a in (0.1, 0.25, 0.5, 1.0)},
        },
    )


def default_models():
    """Return a broad fleet of ready-to-use surrogate prototypes (no optional dependencies).

    The mean/KNN/IDW/SVR/RBF baselines plus the :func:`default_kriging` kernel zoo -- good for
    accuracy comparison. For a model that must report uncertainty (e.g. a BO surrogate), select
    over :func:`default_kriging` instead, since the baselines return no ``sigma``.
    """
    return {
        "Mean": SimpleMean(),
        "KNN": KNN(),
        "IDW": InverseDistanceWeighting(),
        "SVR": SVR(),
        "RBF[tps]": RBF(kernel="tps", tail="linear"),
        "RBF[mq]": RBF(kernel="mq", tail="linear"),
        **default_kriging(),
    }


def _sample(n, xl, xu, rng, method):
    """Draw ``n`` points in the box ``[xl, xu]`` (``"lhs"`` Latin hypercube or ``"random"``)."""
    xl, xu = np.asarray(xl, dtype=float), np.asarray(xu, dtype=float)
    d = len(xl)
    if method == "random":
        unit = rng.random((n, d))
    else:  # latin hypercube — one point per stratum per dimension
        unit = np.empty((n, d))
        edges = np.linspace(0, 1, n + 1)
        for j in range(d):
            unit[:, j] = rng.permutation(edges[:n] + rng.random(n) * (edges[1] - edges[0]))
    return xl + unit * (xu - xl)


def _predict_with_sigma(model, X):
    """Predict means and, if the model exposes it, predictive sigma; else ``(y_hat, None)``."""
    try:
        pred = model.predict(X, var=True)
        y_hat = np.asarray(pred.y).flatten()
        sigma = np.asarray(pred.sigma).flatten() if pred.sigma is not None else None
        if sigma is not None and not np.all(np.isfinite(sigma)):
            sigma = None
        return y_hat, sigma
    except Exception:
        return np.asarray(model.predict(X).y).flatten(), None


class StudyResult:
    """Holistic study outcome: per-model metric distributions over repeated samples.

    Attributes:
        raw: ``{model_name: {metric: [value_per_repeat, ...]}}``.
        failures: ``{model_name: n_failed_fits}``.
        meta: Run settings (``n``, ``repeats``, ``dim``, ...).
    """

    def __init__(self, raw, failures, meta):
        self.raw = raw
        self.failures = failures
        self.meta = meta

    @staticmethod
    def _finite(values):
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _agg(arr, fn):
        return float(fn(arr)) if arr.size else np.nan

    def mean(self, metric):
        """Mean of ``metric`` per model across repeats (NaN for no finite values)."""
        return {name: self._agg(self._finite(v.get(metric, [])), np.mean) for name, v in self.raw.items()}

    def std(self, metric):
        """Std of ``metric`` per model across repeats (NaN for no finite values)."""
        return {name: self._agg(self._finite(v.get(metric, [])), np.std) for name, v in self.raw.items()}

    def metrics(self):
        """All metric names that were collected (point metrics first)."""
        seen = {m for v in self.raw.values() for m in v}
        ordered = [m for m in POINT_METRICS if m in seen]
        return ordered + [m for m in seen if m not in ordered]

    def ranking(self):
        """Mean rank of each model across all metrics (direction-aware; lower is better)."""
        names = list(self.raw)
        ranks: dict = {n: [] for n in names}
        for metric in self.metrics():
            means = self.mean(metric)
            # shared direction logic with Benchmark: smaller key = better (target- and direction-aware)
            order = sorted(names, key=lambda n: metric_sort_key(metric, means[n]))
            for rank, n in enumerate(order):
                ranks[n].append(rank)
        avg = {n: float(np.mean(r)) if r else np.nan for n, r in ranks.items()}
        return dict(sorted(avg.items(), key=lambda kv: kv[1]))

    def best(self):
        """Name of the model with the lowest mean rank."""
        return next(iter(self.ranking()))

    def frame(self):
        """Return a pandas DataFrame of per-model metric means (rows sorted by overall rank)."""
        import pandas as pd  # type: ignore[import-untyped]

        order = list(self.ranking())
        data = {metric: self.mean(metric) for metric in self.metrics()}
        return pd.DataFrame(data, index=order)

    def __str__(self):
        m = self.meta
        ranking = self.ranking()
        head = f"pysurrogate study — {m['dim']}D · n={m['n']} · {m['repeats']} repeats · n_test={m['n_test']}"
        lines = [head, "=" * len(head), "", "overall ranking (mean rank across all metrics, lower = better):"]
        for name, r in ranking.items():
            lines.append(f"  {r:5.2f}  {name}")
        lines.append("")
        lines.append(f"best model: {self.best()}")
        fails = {k: v for k, v in self.failures.items() if v}
        if fails:
            lines.append("failed fits: " + ", ".join(f"{k} {v}/{m['repeats']}" for k, v in fails.items()))
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()


def study(f, xl, xu, n, models=None, n_test=1000, repeats=11, noise=0.0, seed=1, sampling="lhs"):
    """Benchmark surrogate models on a known function sampled over a box domain.

    For each of ``repeats`` independent draws, ``n`` training points are sampled in ``[xl, xu]``
    (and labelled by ``f`` plus optional Gaussian ``noise``), every model is fitted, and its
    predictions on a fresh ``n_test``-point cloud are scored against the true function with the
    holistic metric suite. Results are aggregated to mean +/- std so a single lucky/unlucky split
    cannot dominate.

    Args:
        f: Black-box function mapping ``X`` of shape ``(m, d)`` to values of shape ``(m,)``.
        xl: Lower bound of the box domain (length ``d``).
        xu: Upper bound of the box domain (length ``d``).
        n: Number of training points sampled per repeat (the evaluation budget).
        models: Models to benchmark, as ``{name: model}`` or a list of instances. Defaults to
            :func:`default_models`.
        n_test: Size of the held-out test cloud used for scoring.
        repeats: Number of independent train/test resamples to average over.
        noise: Std-dev of Gaussian noise added to the training labels (test labels stay noise-free).
        seed: Base random seed; repeat ``i`` uses ``seed + i``.
        sampling: ``"lhs"`` (Latin hypercube) or ``"random"`` (uniform) training-point sampling.

    Returns:
        A :class:`StudyResult` with per-model metric distributions and a direction-aware ranking.
    """
    protos = as_named(default_models() if models is None else models)
    xl, xu = np.asarray(xl, dtype=float), np.asarray(xu, dtype=float)
    dim = len(xl)

    raw: dict = {name: {} for name in protos}
    failures = {name: 0 for name in protos}

    for i in range(repeats):
        rng = np.random.default_rng(seed + i)
        X = _sample(n, xl, xu, rng, sampling)
        y = np.asarray(f(X)).flatten()
        if noise > 0:
            y = y + rng.normal(0, noise, len(y))
        X_test = _sample(n_test, xl, xu, rng, "random")
        y_test = np.asarray(f(X_test)).flatten()

        for name, proto in protos.items():
            model = copy.deepcopy(proto)
            try:
                model.fit(X, y)
                y_hat, sigma = _predict_with_sigma(model, X_test)
                if not np.all(np.isfinite(y_hat)):
                    raise ValueError("non-finite predictions")
            except Exception:
                failures[name] += 1
                continue
            for family in evaluate(y_test, y_hat, sigma=sigma).values():
                for metric, value in family.items():
                    raw[name].setdefault(metric, []).append(value)

    meta = dict(n=n, repeats=repeats, n_test=n_test, dim=dim, noise=noise, sampling=sampling, seed=seed)
    return StudyResult(raw, failures, meta)
