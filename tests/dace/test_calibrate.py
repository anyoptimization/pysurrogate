"""Tests for cross-validated scalar variance calibration (Dace.calibrate)."""

import numpy as np
import pytest

from pysurrogate.core.partitioning import CrossvalidationPartitioning, RandomPartitioning
from pysurrogate.core.sampling import LHS, Sampling
from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.dace import Dace
from pysurrogate.selection.metrics import calc_metric
from pysurrogate.util.test_functions import get_test_function


def _noisy(seed=0, n_train=60, noise_pct=0.10):
    f, xl, xu = get_test_function("rosenbrock", n_var=2)
    rng = np.random.default_rng(seed)
    Xtr = Sampling(n_train, LHS()).sample((xl, xu), rng)
    sd = noise_pct * np.std(f(Xtr))
    ytr = (f(Xtr) + rng.normal(0, sd, n_train))[:, None]
    return Xtr, ytr, f, xl, xu, sd, rng


def test_scale_defaults_to_identity():
    # an uncalibrated model has scale 1.0 and predict is unchanged by it
    Xtr, ytr, *_ = _noisy()
    m = Dace(corr=Gaussian())
    m.fit(Xtr, ytr)
    assert m.scale == 1.0


def test_calibrate_uses_only_training_data():
    # calibrate needs no external set -- it cross-validates the training rows
    Xtr, ytr, *_ = _noisy()
    m = Dace(corr=Gaussian(), noise_bounds=(1e-8, 1.0), noise=1e-4)
    m.fit(Xtr, ytr)
    s = m.calibrate()  # no arguments
    assert s > 1.0  # the kriging variance was overconfident


def test_calibrate_improves_test_calibration():
    # the CV scale drives calibration on a large independent test set toward 1
    Xtr, ytr, f, xl, xu, sd, rng = _noisy()
    Xte = xl + (xu - xl) * rng.random((4000, 2))
    yte = f(Xte) + rng.normal(0, sd, 4000)
    m = Dace(corr=Gaussian(), noise_bounds=(1e-8, 1.0), noise=1e-4)
    m.fit(Xtr, ytr)
    p0 = m.predict(Xte, mse=True)
    before = calc_metric("calib", yte, p0.y.ravel(), sigma=p0.sigma.ravel())
    m.calibrate()
    p1 = m.predict(Xte, mse=True)
    after = calc_metric("calib", yte, p1.y.ravel(), sigma=p1.sigma.ravel())
    assert before > 1.5  # clearly overconfident before
    assert abs(np.log(after)) < abs(np.log(before))  # closer to 1 in log-scale


def test_calibrate_scales_variance_not_mean():
    # only the variance moves; the mean predictions are identical before/after
    Xtr, ytr, f, xl, xu, sd, rng = _noisy()
    Xq = xl + (xu - xl) * rng.random((50, 2))
    m = Dace(corr=Gaussian())
    m.fit(Xtr, ytr)
    y_before = m.predict(Xq).y.copy()
    p0 = m.predict(Xq, mse=True)
    m.calibrate()
    p1 = m.predict(Xq, mse=True)
    np.testing.assert_allclose(m.predict(Xq).y, y_before)  # mean untouched
    np.testing.assert_allclose(p1.var, m.scale * p0.var)  # variance scaled by exactly s


def test_calibrate_accepts_a_partitioning():
    # an explicit partitioning is honored (random hold-out instead of the default k-fold)
    Xtr, ytr, *_ = _noisy()
    m = Dace(corr=Gaussian())
    m.fit(Xtr, ytr)
    s = m.calibrate(RandomPartitioning(perc_train=0.7, n_sets=3, seed=0))
    assert s > 0.0 and m.scale == s


def test_calibrate_accepts_a_mask():
    # a boolean mask = one held-out split: True rows are validation, the rest are re-fit
    Xtr, ytr, *_ = _noisy()
    m = Dace(corr=Gaussian())
    m.fit(Xtr, ytr)
    mask = np.zeros(Xtr.shape[0], dtype=bool)
    mask[-20:] = True  # hold out the last 20 rows
    s = m.calibrate(mask)
    assert s > 0.0 and m.scale == s


def test_calibrate_mask_validation():
    # malformed masks are rejected: wrong length, or holding out all / none
    Xtr, ytr, *_ = _noisy()
    m = Dace(corr=Gaussian())
    m.fit(Xtr, ytr)
    with pytest.raises(ValueError, match="one entry per training row"):
        m.calibrate(np.ones(5, dtype=bool))
    with pytest.raises(ValueError, match="some rows but not all"):
        m.calibrate(np.ones(Xtr.shape[0], dtype=bool))  # all held out -> empty train


def test_fit_resets_scale():
    # a fresh fit invalidates a prior calibration
    Xtr, ytr, *_ = _noisy()
    m = Dace(corr=Gaussian())
    m.fit(Xtr, ytr)
    m.calibrate()
    assert m.scale != 1.0
    m.fit(Xtr, ytr)
    assert m.scale == 1.0


def test_calibrate_works_on_small_design():
    # cross-validation pools all rows, so calibrate stays usable on a small design (no min guard)
    Xtr, ytr, *_ = _noisy(n_train=8)
    m = Dace(corr=Gaussian())
    m.fit(Xtr, ytr)
    s = m.calibrate(CrossvalidationPartitioning(k_folds=4))
    assert s > 0.0 and m.scale == s


def test_fit_never_calibrates():
    # calibration is standalone-only: a plain fit leaves the variance scale at the identity
    Xtr, ytr, *_ = _noisy()
    m = Dace(corr=Gaussian(), noise_bounds=(1e-8, 1.0), noise=1e-4)
    m.fit(Xtr, ytr)
    assert m.scale == 1.0  # fit did not calibrate
    m.calibrate()  # only the explicit call changes it
    assert m.scale != 1.0


def test_calibrate_before_fit_raises():
    m = Dace(corr=Gaussian())
    with pytest.raises(Exception, match="requires a prior fit"):
        m.calibrate()
