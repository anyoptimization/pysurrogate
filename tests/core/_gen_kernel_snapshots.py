"""Generate the bit-exact kernel snapshot fixture from the CURRENT implementations.

Run once (via ``pyclawd python``) *before* the Metric/Profile refactor, while the legacy
``Gaussian``/``Exponential``/``KPLSKernel``/``Mahalanobis`` are still the live code. The refactor
must reproduce every array here **byte-for-byte** (``np.array_equal``); ``tests/core/
test_kernel_equivalence.py`` is the gate. This exists because KPLS/Mahalanobis are absent from the
DACE golden suite, so golden cannot see a regression in them -- the frozen snapshot is their oracle.
"""

import numpy as np

from pysurrogate.core.kernel import (
    Exponential,
    Gaussian,
    KPLSKernel,
    Mahalanobis,
)

OUT = "tests/core/fixtures/kernel_snapshots.npz"


def _cases():
    rng = np.random.default_rng(20240705)
    d, n = 4, 18
    D = rng.standard_normal((n * n, d))  # kernel-matrix layout (n_pairs, d)
    snaps = {"D": D}

    def add(tag, kernel, theta, thetas):
        snaps[f"{tag}.theta"] = np.asarray(theta, float)
        snaps[f"{tag}.thetas"] = np.asarray(thetas, float)
        snaps[f"{tag}.call"] = kernel(D, theta)
        snaps[f"{tag}.batch"] = kernel.batch(D, thetas)
        snaps[f"{tag}.grad"] = kernel.grad(D, theta)
        if kernel.has_theta_grad:
            snaps[f"{tag}.theta_grad"] = kernel.theta_grad(D, theta)

    # isotropic + ARD length-scales, and a small theta population for batch
    t_iso = np.array([0.7])
    ts_iso = rng.random((5, 1)) + 0.2
    t_ard = rng.random(d) + 0.3
    ts_ard = rng.random((5, d)) + 0.2

    add("gauss.iso", Gaussian(), t_iso, ts_iso)
    add("gauss.ard", Gaussian(ard=True), t_ard, ts_ard)
    add("exp.iso", Exponential(), t_iso, ts_iso)
    add("exp.ard", Exponential(ard=True), t_ard, ts_ard)

    # KPLS: reducible (Gaussian base, the GEMM/reduced path) and non-reducible (Exponential base,
    # the expand-to-eta fallback). h = 2 reduced coordinates. Store the construction matrices in the
    # fixture so the test rebuilds the exact kernels without replaying the RNG draw sequence.
    h = 2
    w2 = np.square(rng.standard_normal((d, h)))  # squared PLS-like weights
    A_full = rng.standard_normal((d, d))  # full-rank rotation
    A_low = rng.standard_normal((d, h))  # rank-deficient projection
    snaps["w2"], snaps["A_full"], snaps["A_low"] = w2, A_full, A_low

    add("kpls.gauss", KPLSKernel(Gaussian(), w2), rng.random(h) + 0.2, rng.random((5, h)) + 0.2)
    add("kpls.exp", KPLSKernel(Exponential(), w2), rng.random(h) + 0.2, rng.random((5, h)) + 0.2)
    add("maha.full", Mahalanobis(A_full), rng.random(d) + 0.2, rng.random((5, d)) + 0.2)
    add("maha.low", Mahalanobis(A_low), rng.random(h) + 0.2, rng.random((5, h)) + 0.2)

    return snaps


def main():
    snaps = _cases()
    np.savez(OUT, **snaps)
    print(f"wrote {OUT} with {len(snaps)} arrays")


if __name__ == "__main__":
    main()
