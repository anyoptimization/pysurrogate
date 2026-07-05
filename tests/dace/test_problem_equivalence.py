"""Bit-exact regression net for routing DaceProblem's search vector through a ParameterSpace.

Every array in ``fixtures/problem_snapshots.npz`` was captured from the pre-rewire DaceProblem (see
``_gen_problem_snapshots.py``). Routing the search layout through
:class:`~pysurrogate.core.parameter.ParameterSpace` must reproduce ``bounds`` / ``sampling_bounds`` /
``decode`` / ``screen`` / the objective + gradient **byte-for-byte** -- ``np.array_equal``, not
``allclose``. DaceProblem's objective feeds the Boxmin trajectory the golden suite snapshots, so a
ULP shift here can drift a golden baseline far past tolerance; this file catches it directly.
"""

import numpy as np
import pytest

from pysurrogate.dace import ConstantRegression, Gaussian, GeneralizedExponential
from pysurrogate.dace.problem import DaceProblem

_SNAP = np.load("tests/dace/fixtures/problem_snapshots.npz")


def _problems():
    X, Y = _SNAP["X"], _SNAP["Y"]
    regr = ConstantRegression()
    return {
        "gauss_scalar": DaceProblem(X, Y, regr, Gaussian(), theta_bounds=(0.05, 10.0)),
        "gauss_ard": DaceProblem(
            X, Y, regr, Gaussian(ard=True), theta_bounds=(np.array([0.05, 0.05]), np.array([10.0, 10.0]))
        ),
        "expg": DaceProblem(
            X, Y, regr, GeneralizedExponential(), theta_bounds=(np.array([0.05, 1.0]), np.array([10.0, 3.0]))
        ),
        "gauss_noise": DaceProblem(X, Y, regr, Gaussian(), theta_bounds=(0.05, 10.0), noise_bounds=(1e-6, 1e-1)),
        "gauss_prior": DaceProblem(X, Y, regr, Gaussian(), theta_bounds=(0.05, 10.0), theta_prior=(0.0, 0.01)),
        "gauss_unbounded": DaceProblem(X, Y, regr, Gaussian(), theta_bounds=(0.05, np.inf)),
    }


_TAGS = ["gauss_scalar", "gauss_ard", "expg", "gauss_noise", "gauss_prior", "gauss_unbounded"]


@pytest.mark.parametrize("tag", _TAGS)
def test_dace_problem_is_byte_identical_to_snapshot(tag):
    prob = _problems()[tag]
    pop = _SNAP[f"{tag}.pop"]

    lo, hi = prob.bounds
    np.testing.assert_array_equal(lo, _SNAP[f"{tag}.lo"])
    np.testing.assert_array_equal(hi, _SNAP[f"{tag}.hi"])

    slo, shi = prob.sampling_bounds
    np.testing.assert_array_equal(slo, _SNAP[f"{tag}.slo"])
    np.testing.assert_array_equal(shi, _SNAP[f"{tag}.shi"])

    for j, x in enumerate(pop):
        theta, noise = prob.decode(x)
        np.testing.assert_array_equal(np.asarray(theta, float), _SNAP[f"{tag}.decode{j}.theta"])
        np.testing.assert_array_equal(np.asarray(noise, float), _SNAP[f"{tag}.decode{j}.noise"])

    np.testing.assert_array_equal(prob.screen(pop), _SNAP[f"{tag}.screen"])

    ev = prob(pop)
    np.testing.assert_array_equal(ev.f, _SNAP[f"{tag}.f"])
    np.testing.assert_array_equal(ev.grad, _SNAP[f"{tag}.grad"])
    np.testing.assert_array_equal(ev.feasible, _SNAP[f"{tag}.feasible"])
