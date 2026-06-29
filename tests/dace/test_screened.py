"""Behavior tests for ScreenedLBFGS -- the batched-screen + gradient-polish theta search."""

import numpy as np

from pysurrogate.dace.corr import Gaussian, Matern
from pysurrogate.dace.dace import Dace
from pysurrogate.dace.fit import fit
from pysurrogate.dace.optimizers import Boxmin, ScreenedLBFGS
from pysurrogate.dace.regr import ConstantRegression

GAUSS, CONSTANT = Gaussian(), ConstantRegression()


def _xy(seed, n, d):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sum(np.sin(X * 3.0), axis=1)
    return X, y


def test_screened_improves_objective_and_predicts():
    X, y = _xy(0, 30, 2)
    model = Dace(
        regr=CONSTANT,
        corr=GAUSS,
        theta=np.array([1.0, 1.0]),
        thetaL=[1e-4, 1e-4],
        thetaU=[50.0, 50.0],
        optimizer=ScreenedLBFGS(n_cand=48, k_starts=2),
    )
    model.fit(X, y)
    start = fit(model.model["nX"], model.model["nY"], CONSTANT, GAUSS, np.array([1.0, 1.0]))["f"]
    assert model.model["f"] <= start + 1e-9
    assert np.all(np.isfinite(model.predict(np.random.default_rng(1).random((5, 2))).y))


def test_screened_isotropic_theta_shape():
    X, y = _xy(7, 24, 2)
    model = Dace(regr=CONSTANT, corr=GAUSS, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=ScreenedLBFGS())
    model.fit(X, y)
    assert np.ravel(model.model["theta"]).shape == (1,)
    assert np.all(np.isfinite(model.predict(np.random.default_rng(2).random((4, 2))).y))


def test_screened_escapes_bad_starting_basin():
    # from the lower-bound plateau a single warm descent stays stuck; the screen must
    # find the good basin regardless of the (poor) configured start theta.
    X = np.random.default_rng(3).random((30, 1))
    y = np.sin(X[:, 0] * 12.0)
    model = Dace(regr=CONSTANT, corr=GAUSS, theta=1e-4, thetaL=1e-4, thetaU=100.0, optimizer=ScreenedLBFGS())
    model.fit(X, y)
    assert model.model["f"] < 1e-2
    assert np.ravel(model.model["theta"])[0] > 1e-4  # moved off the plateau


def test_screened_beats_or_matches_default_boxmin_on_smooth():
    # on a smooth response the MLE basin is well-defined: screened-LBFGS must reach an
    # objective no worse than Boxmin's (it underfits smooth responses).
    X, y = _xy(5, 60, 5)
    bounds = dict(theta=np.full(5, 1.0), thetaL=np.full(5, 1e-3), thetaU=np.full(5, 20.0))
    boxmin = Dace(regr=CONSTANT, corr=GAUSS, **bounds)
    boxmin.fit(X, y)
    screened = Dace(regr=CONSTANT, corr=GAUSS, optimizer=ScreenedLBFGS(n_cand=48, k_starts=2), **bounds)
    screened.fit(X, y)
    assert screened.model["f"] <= boxmin.model["f"] + 1e-6


def test_auto_noise_learns_a_nugget_and_improves_calibration():
    # noisy targets: the jointly-learned nugget must be a real positive value (more than the
    # interpolating fit's) and must improve the predictive density (NLPD) on clean test points
    # -- the whole point of learning the noise.
    rng = np.random.default_rng(11)
    X = rng.random((60, 4))
    yc = np.sum(np.sin(X * 3.0), axis=1)
    y = yc + 0.30 * np.std(yc) * rng.standard_normal(len(yc))
    Xte = rng.random((400, 4))
    yte = np.sum(np.sin(Xte * 3.0), axis=1)

    def nlpd(m):
        p = m.predict(Xte, mse=True)
        var = np.maximum(p.mse.ravel(), 1e-10)
        return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (yte - p.y.ravel()) ** 2 / var))

    bounds = dict(theta=np.full(4, 1.0), thetaL=np.full(4, 1e-3), thetaU=np.full(4, 50.0))
    interp = Dace(corr=GAUSS, noise=0.0, **bounds)
    interp.fit(X, y)
    auto = Dace(corr=GAUSS, noise="auto", **bounds)
    auto.fit(X, y)
    assert auto.model["noise"] > interp.model["noise"]  # a real nugget was learned
    assert nlpd(auto) < nlpd(interp)  # better-calibrated predictive variance


def test_auto_noise_stays_near_zero_on_clean_data():
    # deterministic smooth target: marginal likelihood should prefer ~no nugget (interpolate).
    rng = np.random.default_rng(12)
    X = rng.random((40, 2))
    y = np.sum(X**2, axis=1)
    m = Dace(
        corr=GAUSS,
        theta=np.full(2, 1.0),
        thetaL=np.full(2, 1e-3),
        thetaU=np.full(2, 50.0),
        noise="auto",
    )
    m.fit(X, y)
    assert m.model["noise"] < 0.02  # essentially interpolating
    assert np.all(np.isfinite(m.predict(rng.random((4, 2))).y))


def test_auto_noise_requires_bounds_and_supporting_optimizer():
    rng = np.random.default_rng(13)
    X = rng.random((10, 2))
    y = np.sum(X, axis=1)
    # no theta bounds -> nothing to search the nugget alongside
    import pytest

    with pytest.raises(Exception, match="requires theta bounds"):
        Dace(corr=GAUSS, theta=1.0, thetaL=None, thetaU=None, noise="auto").fit(X, y)
    # an optimizer that does not support joint nugget learning
    with pytest.raises(Exception, match="not supported"):
        Dace(
            corr=GAUSS,
            theta=np.full(2, 1.0),
            thetaL=np.full(2, 1e-3),
            thetaU=np.full(2, 50.0),
            noise="auto",
            optimizer=Boxmin(),
        ).fit(X, y)


def test_screened_refit_with_validation_appends_and_warm_starts():
    X, y = _xy(4, 20, 2)
    model = Dace(
        regr=CONSTANT,
        corr=GAUSS,
        theta=np.array([1.0, 1.0]),
        thetaL=[1e-4, 1e-4],
        thetaU=[50.0, 50.0],
    )
    model.fit(X, y)
    rng = np.random.default_rng(9)
    Xn = rng.random((6, 2))
    model.refit(Xn, np.sum(np.sin(Xn * 3.0), axis=1))  # default optimizer is ScreenedLBFGS
    assert model.model["X"].shape[0] == 26
    assert np.all(np.isfinite(model.predict(rng.random((5, 2))).y))


def test_screened_product_kernel_ard():
    X, y = _xy(6, 28, 2)
    matern = Matern(nu=2.5)
    model = Dace(
        regr=CONSTANT,
        corr=matern,
        theta=np.array([1.0, 1.0]),
        thetaL=[1e-3, 1e-3],
        thetaU=[20.0, 20.0],
        optimizer=ScreenedLBFGS(k_starts=2),
    )
    model.fit(X, y)
    theta = np.ravel(model.model["theta"])
    assert np.all((theta >= 1e-3) & (theta <= 20.0))
    assert np.all(np.isfinite(model.predict(np.random.default_rng(8).random((4, 2))).y))


def test_screened_is_the_default_optimizer():
    # ScreenedLBFGS is now the Dace constructor default for a bounded fit.
    X, y = _xy(0, 24, 2)
    model = Dace(corr=GAUSS, theta=np.full(2, 1.0), thetaL=np.full(2, 1e-4), thetaU=np.full(2, 50.0))
    assert isinstance(model.optimizer, ScreenedLBFGS)
    model.fit(X, y)
    assert "k_starts" in model.optimization  # the screened-search record landed
    assert np.all(np.isfinite(model.predict(np.random.default_rng(1).random((4, 2))).y))


def test_screened_subsample_screen_runs_on_larger_n():
    # screen_rows < n triggers the row-subsample screen; the polish still uses all rows.
    X = np.random.default_rng(2).random((90, 3))
    y = np.sum(X**2, axis=1)
    model = Dace(
        corr=GAUSS,
        theta=np.full(3, 1.0),
        thetaL=np.full(3, 1e-3),
        thetaU=np.full(3, 20.0),
        optimizer=ScreenedLBFGS(screen_rows=48),
    )
    model.fit(X, y)
    assert model.model["nX"].shape[0] == 90  # full data fitted
    assert np.all(np.isfinite(model.predict(np.random.default_rng(3).random((5, 3))).y))


def test_screened_accepts_random_sampler_and_custom_callable():
    X, y = _xy(4, 26, 2)
    bounds = dict(theta=np.full(2, 1.0), thetaL=np.full(2, 1e-3), thetaU=np.full(2, 20.0))
    Dace(corr=GAUSS, optimizer=ScreenedLBFGS(sampler="random"), **bounds).fit(X, y)
    # a custom sampler: any f(n, p) -> [0, 1]^(n, p) (e.g. a pysampling method) plugs in
    calls = []

    def my_sampler(n, p):
        calls.append((n, p))
        return np.random.default_rng(0).random((n, p))

    m = Dace(corr=GAUSS, optimizer=ScreenedLBFGS(sampler=my_sampler), **bounds)
    m.fit(X, y)
    assert calls and calls[0][1] == 2  # the callable was invoked with p=2
    assert np.all(np.isfinite(m.predict(np.random.default_rng(5).random((4, 2))).y))
