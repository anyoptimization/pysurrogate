"""Regression tests for transformation edge cases (constant dims, integer inputs)."""

import numpy as np

from pysurrogate.core.transformation import Plog, Standardization


def test_standardization_constant_dimension_is_finite():
    # regression: a zero-std (constant) column previously produced NaN/inf via /0
    X = np.column_stack([np.linspace(0, 1, 10), np.full(10, 3.0)])

    t = Standardization()
    Z = t.forward(X)

    assert np.all(np.isfinite(Z))
    # the constant column centers to all-zeros and round-trips back to its constant
    np.testing.assert_allclose(Z[:, 1], 0.0)
    np.testing.assert_allclose(t.backward(Z), X)


def test_plog_does_not_truncate_integer_input():
    # regression: np.zeros_like(int array) truncated the log/exp results back to int
    y = np.array([0, 1, 5, -3], dtype=int)

    t = Plog()
    forward = t.forward(y)

    assert forward.dtype.kind == "f"
    np.testing.assert_allclose(forward[1], np.log(2))
    np.testing.assert_allclose(t.backward(forward), y.astype(float), atol=1e-12)
