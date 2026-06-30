"""Regression test for the cubic-kernel non-positive-definite Cholesky crash.

On standardized data the cubic correlation matrix is genuinely non-positive-definite at
the boxmin *start* theta (min eigenvalue ~ -0.012, far below what the tiny fit() jitter
can repair). The start fit must relocate to a feasible theta rather than crashing in
Cholesky, since larger thetas (5, 10, 50, 100) give a perfectly positive-definite matrix
that the search could have used.
"""

import numpy as np
import pytest

from pysurrogate.dace.corr import Cubic
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.regr import LinearRegression


def _cubic_dataset():
    # fixed 40-point, 2-D sample from a smooth function (matches the repro)
    rng = np.random.default_rng(42)
    X = rng.random((40, 2))
    y = (np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1])).reshape(-1, 1)
    return X, y


def test_cubic_kernel_fits_despite_non_pd_start_theta():
    # the cubic correlation matrix is non-PD at the default start theta (=1.0).
    # boxmin must relocate to a feasible (positive-definite) theta and fit EXACTLY
    # (no added noise), rather than crashing in Cholesky.
    from pysurrogate.dace.corr import calc_kernel_matrix

    X, y = _cubic_dataset()
    model = Dace(regr=LinearRegression(), corr=Cubic(), theta=1.0, theta_bounds=(1e-5, 100.0))
    model.fit(X, y)

    assert np.all(np.isfinite(model.predict(X).y))
    assert model.model["noise"] == 0.0  # relocation found a feasible theta -> exact fit

    # the theta it settled on must give a genuinely positive-definite matrix
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    R = calc_kernel_matrix(nX, nX, Cubic(), theta=model.model["theta"])
    assert np.linalg.eigvalsh(R).min() > -1e-8


def test_no_feasible_theta_raises_by_default():
    # [0.5, 1.0] is an entirely non-PD bracket for cubic on this data, needing ~1.2%
    # noise. There is no auto-repair noise climb, so a search that finds no positive-
    # definite theta must surface the infeasibility loudly (fix it by setting noise).
    X, y = _cubic_dataset()
    model = Dace(regr=LinearRegression(), corr=Cubic(), theta=0.7, theta_bounds=(0.5, 1.0))
    with pytest.raises(Exception, match="positive-definite"):
        model.fit(X, y)


def test_no_feasible_theta_fits_when_noise_is_set():
    # the supported way to fit an otherwise-infeasible bracket: set a deliberate noise
    # large enough to regularize R (a regression GP), instead of any hidden climb.
    X, y = _cubic_dataset()
    model = Dace(regr=LinearRegression(), corr=Cubic(), theta=0.7, theta_bounds=(0.5, 1.0), noise=0.05)
    model.fit(X, y)
    assert np.all(np.isfinite(model.predict(X).y))
    np.linalg.cholesky(model.model["R"])  # the deliberate noise made R positive-definite


def test_cubic_correlation_matrix_is_not_pd_at_start_theta():
    # pin the root cause itself: the standardized cubic R at theta=1.0 has a clearly
    # negative eigenvalue, so this is a real non-PD matrix, not float jitter.
    from pysurrogate.dace.corr import calc_kernel_matrix

    X, _ = _cubic_dataset()
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    R = calc_kernel_matrix(nX, nX, Cubic(), theta=1.0)
    assert np.linalg.eigvalsh(R).min() < -1e-3
