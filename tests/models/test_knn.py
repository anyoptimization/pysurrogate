"""KNN surrogate: single- and multi-output prediction with inverse-distance weighting."""

import numpy as np

from pysurrogate.models import KNN


def test_knn_single_output_shapes_and_finiteness():
    rng = np.random.RandomState(0)
    X = rng.random((40, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    m = KNN(n_nearest=8).fit(X, y)
    pred = m.predict(X[:6], var=True)
    assert pred.y.shape == (6, 1)
    assert pred.var.shape == (6, 1)
    assert np.all(np.isfinite(pred.y)) and np.all(pred.var >= 0.0)


def test_knn_supports_multioutput_targets():
    # KNN used to only line up for q == 1 (take_along_axis broadcast); a genuine (n, q) target must
    # now predict per output, with a single shared predictive variance per point (m, 1).
    rng = np.random.RandomState(1)
    X = rng.random((50, 2))
    Y = np.column_stack([np.sin(3 * X[:, 0]), (X[:, 1] - 0.5) ** 2, X[:, 0] + X[:, 1]])  # q = 3
    m = KNN(n_nearest=10).fit(X, Y)

    pred = m.predict(X[:7], var=True)
    assert pred.y.shape == (7, 3)  # one column per output
    assert pred.var.shape == (7, 1)  # shared predictive variance per point
    assert np.all(np.isfinite(pred.y)) and np.all(pred.var >= 0.0)

    # a query at a training point recovers that point's targets (its own weight dominates)
    at_train = m.predict(X[:1]).y
    assert np.allclose(at_train, Y[:1], atol=1e-6)
