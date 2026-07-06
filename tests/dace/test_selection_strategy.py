"""Selection strategies (MLE / MAP / HeldOut) as one reusable object for Dace and the GP backends."""

import numpy as np
import pytest

from pysurrogate.dace import MAP, ConstantRegression, Dace, Gaussian, HeldOut, MaximumLikelihood, Selection
from pysurrogate.models import Kriging


def _data(n=40, d=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    y = np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2
    return X, y


@pytest.mark.parametrize(
    "selection",
    [MaximumLikelihood(), MAP(lam=0.05), HeldOut(fraction=0.3), MaximumLikelihood(noise_bounds=(1e-8, 1e-1))],
    ids=["mle", "map", "heldout", "mle+nugget"],
)
def test_every_strategy_fits_and_predicts_through_dace(selection):
    X, y = _data()
    m = Dace(regr=ConstantRegression(), corr=Gaussian(), theta=1.0, theta_bounds=(0.01, 100.0), selection=selection)
    m.fit(X, y)
    p = m.predict(X[:4], var=True, grad=True)
    assert p.y.shape == (4, 1) and np.all(np.isfinite(p.y))
    assert p.var is not None and np.all(p.var >= -1e-9)


def _fit_theta(X, y, selection):
    m = Dace(regr=ConstantRegression(), corr=Gaussian(), theta=1.0, theta_bounds=(0.01, 100.0), selection=selection)
    m.fit(X, y)
    return float(np.atleast_1d(m.model["theta"])[0])


def test_map_prior_pulls_toward_a_smoother_length_scale_than_mle():
    X, y = _data(seed=1)
    mle = _fit_theta(X, y, MaximumLikelihood())
    mapped = _fit_theta(X, y, MAP(mean=1.0, lam=0.5))
    assert mapped > mle  # the prior toward 10**mean yields a larger (smoother) length-scale


def test_maximum_likelihood_can_learn_the_nugget():
    rng = np.random.RandomState(2)
    X = rng.uniform(-1, 1, size=(60, 3))
    f = np.sin(3 * X[:, [0]])

    def fitted_nugget(data):
        m = Dace(
            regr=ConstantRegression(),
            corr=Gaussian(),
            theta=1.0,
            theta_bounds=(0.01, 100.0),
            noise=1e-8,
            selection=MaximumLikelihood(noise_bounds=(1e-8, 1e-1)),
        )
        m.fit(X, data)
        return m.model["noise"]

    clean = fitted_nugget(f)
    noisy = fitted_nugget(f + 0.3 * rng.standard_normal((60, 1)))
    assert noisy > 10 * clean  # the likelihood learns a larger nugget on noisy data


def test_heldout_split_is_deterministic_and_sized():
    sel = HeldOut(fraction=0.25, seed=3)
    tr, va = sel.holdout(40)
    assert len(va) == 10 and len(tr) == 30
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(range(40))
    tr2, va2 = HeldOut(fraction=0.25, seed=3).holdout(40)
    np.testing.assert_array_equal(va, va2)  # deterministic in the seed


def test_base_selection_is_pure_likelihood_no_holdout():
    assert Selection().holdout(50) is None  # MLE/MAP select on all rows; no held-out split


def test_one_selection_object_serves_both_dace_and_a_model_backend():
    # the SAME strategy object configures the low-level engine and a high-level backend identically
    X, y = _data(seed=4)
    sel = MAP(mean=0.5, lam=0.1)
    engine = Dace(regr=ConstantRegression(), corr=Gaussian(), theta=1.0, theta_bounds=(0.01, 100.0), selection=sel)
    engine.fit(X, y)
    backend = Kriging(corr=Gaussian(), theta_bounds=(0.01, 100.0), selection=sel).fit(X, y)
    assert np.all(np.isfinite(engine.predict(X[:3]).y))
    assert np.all(np.isfinite(backend.predict(X[:3]).y))
