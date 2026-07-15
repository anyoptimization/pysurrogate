"""RandomForest surrogate: the discretization grid bounds must track the data across re-fits."""

import numpy as np

from pysurrogate.models import RandomForest


def test_random_forest_refit_updates_grid_bounds():
    """A second fit on a shifted range must re-derive the grid bounds, not reuse the first fit's.

    ``_fit`` previously overwrote the constructor's ``xl``/``xu``, freezing them after the first
    fit -- so a later fit / refit on grown or shifted data silently re-used the original range and
    mis-binned every out-of-range point. The resolved bounds now live in fit-local attributes and
    the constructor attributes stay untouched.
    """
    rng = np.random.RandomState(0)
    X1 = rng.uniform(0.0, 1.0, (40, 2))
    m = RandomForest(n_partitions=8, n_estimators=20).fit(X1, X1[:, 0])
    xu_first = m._xu.copy()

    X2 = rng.uniform(5.0, 6.0, (40, 2))  # a disjoint, shifted range
    m.fit(X2, X2[:, 0])

    assert not np.allclose(m._xu, xu_first)  # bounds moved to the new data's range
    assert np.all(m._xu >= 4.0)  # reflect the shifted [5, 6] domain, not the frozen [0, 1]
    assert m.xl is None and m.xu is None  # constructor attributes never overwritten
    assert np.all(np.isfinite(m.predict(X2[:5]).y))


def test_random_forest_respects_explicit_bounds():
    # explicit constructor bounds are honored and preserved across a fit
    rng = np.random.RandomState(1)
    X = rng.uniform(0.0, 1.0, (30, 2))
    xl, xu = np.zeros(2), np.full(2, 2.0)
    m = RandomForest(n_partitions=6, n_estimators=15, xl=xl, xu=xu).fit(X, X[:, 0])
    assert np.allclose(m._xl, xl) and np.allclose(m._xu, xu)
    assert np.all(np.isfinite(m.predict(X[:5]).y))


def test_random_forest_keeps_the_best_target_per_grid_cell():
    # two designs landing in the same coarse cell collapse to ONE (via np.unique row keying), keeping
    # the minimum target (np.minimum.at). With a single occupied cell the forest predicts that min.
    # explicit wide bounds so both points fall in the same coarse bin (grid cell 0 in each dim)
    X = np.array([[0.10, 0.10], [0.15, 0.12]])
    y = np.array([7.0, 2.0])
    m = RandomForest(n_partitions=2, xl=np.zeros(2), xu=np.full(2, 10.0), n_estimators=10).fit(X, y)
    assert np.allclose(m.predict(np.array([[1.0, 1.0]])).y, 2.0)  # the minimum of the two targets


def test_random_forest_random_state_is_exposed_and_deterministic():
    # random_state is a real constructor knob (was hard-coded); the same seed gives an identical fit.
    rng = np.random.RandomState(0)
    X = rng.uniform(0.0, 1.0, (40, 3))
    y = X[:, 0] + 0.5 * X[:, 1]
    a = RandomForest(random_state=7, n_estimators=15).fit(X, y).predict(X[:5]).y
    b = RandomForest(random_state=7, n_estimators=15).fit(X, y).predict(X[:5]).y
    assert np.allclose(a, b)
