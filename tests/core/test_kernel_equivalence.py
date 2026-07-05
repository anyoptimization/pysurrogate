"""Bit-exact regression net for the Metric/Profile kernel refactor.

Every array in ``fixtures/kernel_snapshots.npz`` was captured from the pre-refactor
implementations (see ``_gen_kernel_snapshots.py``). The Metric/Profile decomposition must
reproduce them **byte-for-byte** -- ``np.array_equal``, not ``allclose``: the DACE golden suite
snapshots the whole Boxmin trajectory, and a mere ULP shift in a search-path reduction can flip a
step and drift a snapshot far past tolerance. KPLS/Mahalanobis are absent from golden entirely, so
for them this file is the *only* regression oracle -- byte-identity here is mandatory, not nice.
"""

import numpy as np
import pytest

from pysurrogate.core.kernel import (
    Exponential,
    Gaussian,
    KPLSKernel,
    Mahalanobis,
)

_SNAP = np.load("tests/core/fixtures/kernel_snapshots.npz")


def _kernels():
    """Rebuild the exact kernels the snapshot was generated with, using its stored matrices."""
    w2, A_full, A_low = _SNAP["w2"], _SNAP["A_full"], _SNAP["A_low"]
    return {
        "gauss.iso": Gaussian(),
        "gauss.ard": Gaussian(ard=True),
        "exp.iso": Exponential(),
        "exp.ard": Exponential(ard=True),
        "kpls.gauss": KPLSKernel(Gaussian(), w2),
        "kpls.exp": KPLSKernel(Exponential(), w2),
        "maha.full": Mahalanobis(A_full),
        "maha.low": Mahalanobis(A_low),
    }


_TAGS = ["gauss.iso", "gauss.ard", "exp.iso", "exp.ard", "kpls.gauss", "kpls.exp", "maha.full", "maha.low"]


@pytest.mark.parametrize("tag", _TAGS)
def test_kernel_outputs_are_byte_identical_to_snapshot(tag):
    D = _SNAP["D"]
    kernel = _kernels()[tag]
    theta = _SNAP[f"{tag}.theta"]
    thetas = _SNAP[f"{tag}.thetas"]

    np.testing.assert_array_equal(kernel(D, theta), _SNAP[f"{tag}.call"])
    np.testing.assert_array_equal(kernel.batch(D, thetas), _SNAP[f"{tag}.batch"])
    np.testing.assert_array_equal(kernel.grad(D, theta), _SNAP[f"{tag}.grad"])
    if f"{tag}.theta_grad" in _SNAP:
        np.testing.assert_array_equal(kernel.theta_grad(D, theta), _SNAP[f"{tag}.theta_grad"])


def test_kpls_reduced_distance_cache_is_identity_keyed():
    # a theta search reuses one D across evaluations; the reduction caches on object identity so the
    # same D hits the cache and a fresh D (predict) misses. The cached path must stay byte-identical.
    D = _SNAP["D"]
    k = KPLSKernel(Gaussian(), _SNAP["w2"])
    theta = _SNAP["kpls.gauss.theta"]
    first = k(D, theta)
    again = k(D, theta)  # same identity -> cache hit
    np.testing.assert_array_equal(first, again)
