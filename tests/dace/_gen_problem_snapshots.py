"""Capture DaceProblem's observable outputs from the CURRENT code, before the ParameterSpace rewire.

Run once (via ``pyclawd python``) before routing DaceProblem's search vector through a
ParameterSpace. The rewire must reproduce every array here byte-for-byte (``np.array_equal``);
``tests/dace/test_problem_equivalence.py`` is the gate. DaceProblem is golden-critical: its
objective feeds the Boxmin trajectory the DACE golden suite snapshots, so a ULP shift in ``bounds`` /
``decode`` / ``screen`` / the objective can drift a snapshot far past tolerance.
"""

import numpy as np

from pysurrogate.dace import ConstantRegression, Gaussian, GeneralizedExponential
from pysurrogate.dace.problem import DaceProblem

OUT = "tests/dace/fixtures/problem_snapshots.npz"


def _cases():
    rng = np.random.default_rng(424242)
    n, d = 12, 2
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal(n)
    regr = ConstantRegression()
    snaps = {"X": X, "Y": Y}

    def add(tag, prob, pop):
        snaps[f"{tag}.lo"], snaps[f"{tag}.hi"] = prob.bounds
        snaps[f"{tag}.slo"], snaps[f"{tag}.shi"] = prob.sampling_bounds
        # decode a couple of points
        for j, x in enumerate(pop):
            theta, noise = prob.decode(x)
            snaps[f"{tag}.decode{j}.theta"] = np.asarray(theta, float)
            snaps[f"{tag}.decode{j}.noise"] = np.asarray(noise, float)
        snaps[f"{tag}.pop"] = pop
        snaps[f"{tag}.screen"] = prob.screen(pop)
        ev = prob(pop)
        snaps[f"{tag}.f"] = ev.f
        snaps[f"{tag}.grad"] = ev.grad
        snaps[f"{tag}.feasible"] = ev.feasible

    # scalar Gaussian (p = 1), no learned noise
    p1 = np.array([[-1.0], [0.0], [0.5], [1.2]])
    add("gauss_scalar", DaceProblem(X, Y, regr, Gaussian(), theta_bounds=(0.05, 10.0)), p1)

    # ARD Gaussian (p = 2)
    p2 = np.array([[-1.0, 0.5], [0.0, 0.0], [0.7, -0.3], [1.1, 0.9]])
    add(
        "gauss_ard",
        DaceProblem(X, Y, regr, Gaussian(ard=True), theta_bounds=(np.array([0.05, 0.05]), np.array([10.0, 10.0]))),
        p2,
    )

    # GeneralizedExponential: theta = (length-scale, power), p = 2, per-coordinate bounds
    add(
        "expg",
        DaceProblem(X, Y, regr, GeneralizedExponential(), theta_bounds=(np.array([0.05, 1.0]), np.array([10.0, 3.0]))),
        p2,
    )

    # scalar Gaussian learning the nugget: search vector gains a trailing noise coordinate (p+1 = 2)
    add(
        "gauss_noise",
        DaceProblem(X, Y, regr, Gaussian(), theta_bounds=(0.05, 10.0), noise_bounds=(1e-6, 1e-1)),
        p2,
    )

    # scalar Gaussian with a MAP prior on the encoded length-scale
    add(
        "gauss_prior",
        DaceProblem(X, Y, regr, Gaussian(), theta_bounds=(0.05, 10.0), theta_prior=(0.0, 0.01)),
        p1,
    )

    # an unbounded-above coordinate: hi = inf exercises the sampling-window clamp
    add(
        "gauss_unbounded",
        DaceProblem(X, Y, regr, Gaussian(), theta_bounds=(0.05, np.inf)),
        p1,
    )

    return snaps


def main():
    np.savez(OUT, **_cases())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
