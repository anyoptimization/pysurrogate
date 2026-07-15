"""Regression tests for transformation edge cases (constant dims, integer inputs)."""

import numpy as np
import pytest

from pysurrogate.core.transformation import Plog, Standardization, ZeroToOneNormalization


def test_standardization_reset_reestimates_on_next_forward():
    # a data-fitted Standardization caches its stats on first forward; reset() must drop them so a
    # later forward (the model refit lifecycle) re-estimates from the new data, not stale stats.
    s = Standardization()
    s.forward(np.array([[0.0], [2.0]]))  # estimates mean 1.0
    assert s.mean is not None

    s.reset()
    assert s.mean is None and s.std is None  # estimated stats dropped

    s.forward(np.array([[10.0], [20.0]]))  # re-estimates on the new data
    np.testing.assert_allclose(s.mean, 15.0)


def test_reset_keeps_user_provided_statistics():
    # explicitly provided stats are configuration, not estimated -- reset() must preserve them
    s = Standardization(mean=np.array([5.0]), std=np.array([2.0]))
    s.forward(np.array([[1.0], [9.0]]))
    s.reset()
    np.testing.assert_allclose(s.mean, 5.0)
    np.testing.assert_allclose(s.std, 2.0)

    z = ZeroToOneNormalization(xl=np.array([0.0]), xu=np.array([10.0]))
    z.reset()
    np.testing.assert_allclose(z.xl, 0.0)
    np.testing.assert_allclose(z.xu, 10.0)


def test_model_refit_reestimates_normalization():
    # end-to-end: a model with a data-fitted normalization must re-normalize to the CURRENT data on
    # a fresh fit, not reuse the first fit's statistics (the refit lifecycle on grown/shifted data).
    from pysurrogate.models import RBF

    rng = np.random.RandomState(0)
    XA = rng.random((30, 2))
    m = RBF(kernel="gaussian", norm_X=Standardization()).fit(XA, XA[:, 0])
    XB = rng.random((30, 2)) * 100.0 + 50.0  # a shifted, wider range
    m.fit(XB, XB[:, 0])
    # the stored (normalized) design reflects XB's statistics -> mean 0, not XA's stale stats
    np.testing.assert_allclose(m.X.mean(axis=0), 0.0, atol=1e-8)


def test_standardization_constant_dimension_is_finite():
    # regression: a zero-std (constant) column previously produced NaN/inf via /0
    X = np.column_stack([np.linspace(0, 1, 10), np.full(10, 3.0)])

    t = Standardization()
    Z = t.forward(X)

    assert np.all(np.isfinite(Z))
    # the constant column centers to all-zeros and round-trips back to its constant
    np.testing.assert_allclose(Z[:, 1], 0.0)
    np.testing.assert_allclose(t.backward(Z), X)


def test_zero_to_one_constant_dimension_is_finite():
    # regression: a constant column has range 0; dividing by it produced inf/NaN. It must map to
    # scale 1 (like Standardization) so forward/backward/scale stay finite and round-trip.
    X = np.column_stack([np.linspace(0, 1, 10), np.full(10, 3.0)])
    t = ZeroToOneNormalization()
    Z = t.forward(X)
    assert np.all(np.isfinite(Z))
    assert np.all(np.isfinite(t.scale()))
    np.testing.assert_allclose(t.backward(Z), X)


def test_zero_to_one_estimate_bounds_false_requires_explicit_bounds():
    # estimate_bounds=False with no xl/xu left them None -> a TypeError deep in forward(); validate
    # up front with a clear message instead.
    with pytest.raises(ValueError, match="requires explicit xl and xu"):
        ZeroToOneNormalization(estimate_bounds=False)


def test_plog_does_not_truncate_integer_input():
    # regression: np.zeros_like(int array) truncated the log/exp results back to int
    y = np.array([0, 1, 5, -3], dtype=int)

    t = Plog()
    forward = t.forward(y)

    assert forward.dtype.kind == "f"
    np.testing.assert_allclose(forward[1], np.log(2))
    np.testing.assert_allclose(t.backward(forward), y.astype(float), atol=1e-12)
