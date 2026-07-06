"""Deep-kernel GP: a neural feature map with a GP head -- mean, variance, and analytic gradient."""

import numpy as np
import pytest

from pysurrogate.models import DeepKernelGP, SimpleMean


def _data(n, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    y = (np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2).reshape(-1, 1)
    return X, y


def test_fit_predict_shapes_and_variance():
    X, y = _data(60)
    model = DeepKernelGP(hidden_layer_sizes=(16, 8)).fit(X, y)
    q = np.random.RandomState(1).uniform(-1, 1, size=(5, 2))
    pred = model.predict(q, var=True)
    assert pred.y.shape == (5, 1) and np.all(np.isfinite(pred.y))
    assert pred.var is not None and pred.var.shape == (5, 1)
    assert np.all(pred.var >= -1e-9) and np.all(np.isfinite(pred.var))  # a real GP posterior variance


def test_beats_constant_mean_baseline():
    Xtr, ytr = _data(80, seed=2)
    Xte, yte = _data(200, seed=3)
    dkl = float(np.mean((DeepKernelGP().fit(Xtr, ytr).predict(Xte).y - yte) ** 2))
    base = float(np.mean((SimpleMean().fit(Xtr, ytr).predict(Xte).y - yte) ** 2))
    assert dkl < base


def test_variance_grows_away_from_the_data():
    # a GP head -> predictive variance is small near the training features and larger far away
    X, y = _data(50)
    model = DeepKernelGP(hidden_layer_sizes=(16, 8)).fit(X, y)
    near = model.predict(X, var=True).var  # at the training sites
    far = model.predict(np.full((5, 2), 5.0), var=True).var  # well outside the design
    assert np.mean(near) < np.mean(far)


def test_analytic_gradient_matches_finite_difference():
    # the chain-rule gradient (NN feature Jacobian x GP feature-gradient) vs central differences;
    # tanh keeps the map smooth so the analytic gradient is exact.
    X, y = _data(60, seed=4)
    model = DeepKernelGP(hidden_layer_sizes=(16, 8), activation="tanh").fit(X, y)
    q = np.array([[0.3, -0.2]])
    g = model.predict(q, grad=True).grad
    assert g.shape == (1, 2)

    eps = 1e-6
    fd = np.zeros((1, 2))
    for k in range(2):
        qp, qm = q.copy(), q.copy()
        qp[0, k] += eps
        qm[0, k] -= eps
        fd[0, k] = (model.predict(qp).y[0, 0] - model.predict(qm).y[0, 0]) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-3)


def test_feature_map_is_the_last_hidden_layer():
    X, y = _data(40)
    model = DeepKernelGP(hidden_layer_sizes=(16, 8)).fit(X, y)
    Z = model._features(model.norm_X.forward(X))
    assert Z.shape == (40, 8)  # last hidden width is the feature-space dimension


def test_deterministic_under_fixed_seed():
    X, y = _data(50)
    q = np.random.RandomState(5).uniform(-1, 1, size=(6, 2))
    a = DeepKernelGP(random_state=7).fit(X, y).predict(q).y
    b = DeepKernelGP(random_state=7).fit(X, y).predict(q).y
    np.testing.assert_allclose(a, b)


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        DeepKernelGP(activation="not-an-activation")


def test_rejects_multi_output():
    X, _ = _data(40)
    Y = np.column_stack([np.sin(X.sum(1)), (X**2).sum(1)])  # two outputs
    with pytest.raises(ValueError, match="single output"):
        DeepKernelGP().fit(X, Y)


def test_respects_active_dims_on_fit_refit_and_predict():
    # the target depends only on inputs 0 and 2; restricting active_dims must fit, refit, and predict
    # consistently (the feature map only ever sees the selected columns, in fit and in refit).
    rng = np.random.RandomState(3)
    X = rng.uniform(-1, 1, size=(60, 4))
    y = np.sin(3 * X[:, [0]]) + X[:, [2]] ** 2
    model = DeepKernelGP(hidden_layer_sizes=(12, 8), active_dims=[0, 2]).fit(X, y)
    assert model._coefs[0].shape[0] == 2  # the NN input layer sees only the 2 active dims
    Xnew = rng.uniform(-1, 1, size=(10, 4))
    ynew = np.sin(3 * Xnew[:, [0]]) + Xnew[:, [2]] ** 2
    model.refit(Xnew, ynew)  # must not raise despite full-width raw inputs
    assert np.all(np.isfinite(model.predict(X[:5]).y))


def test_small_nugget_keeps_a_collapsed_feature_map_fittable():
    # a near-constant target drives the MLP toward a near-constant (collapsed) feature map, which
    # would make a zero-nugget GP correlation matrix singular. The default nugget keeps it PD.
    rng = np.random.RandomState(4)
    X = rng.uniform(-1, 1, size=(50, 3))
    y = np.full((50, 1), 2.0) + 1e-9 * rng.standard_normal((50, 1))
    model = DeepKernelGP(hidden_layer_sizes=(6,)).fit(X, y)  # must not raise
    assert np.all(np.isfinite(model.predict(X[:5]).y))


def test_early_stopping_is_on_by_default_and_holds_out_a_validation_set():
    X, y = _data(120)
    model = DeepKernelGP().fit(X, y)
    # sklearn only records a validation curve when early stopping held out a validation split
    assert model.nn_.validation_scores_ is not None and len(model.nn_.validation_scores_) > 0
    assert model.nn_.n_iter_ <= model.max_iter


def test_early_stopping_can_be_disabled():
    X, y = _data(120)
    model = DeepKernelGP(early_stopping=False).fit(X, y)
    assert model.nn_.validation_scores_ is None  # no held-out validation when disabled


def test_early_stopping_curbs_overfitting_on_a_noisy_small_design():
    # noisy training targets: without a validation guard the MLP chases the noise; with early stopping
    # the held-out test error is no worse (here: better).
    rng = np.random.RandomState(11)
    Xtr = rng.uniform(-1, 1, size=(60, 2))
    ytr = (np.sin(3 * Xtr[:, [0]]) + Xtr[:, [1]] ** 2) + rng.normal(0, 0.3, size=(60, 1))
    Xte, yte = _data(400, seed=12)  # clean test set (the true signal)

    def rmse(model):
        p = model.fit(Xtr, ytr).predict(Xte).y
        return float(np.sqrt(np.mean((p - yte) ** 2)))

    # fix the GP nugget (noise_bounds=None) so the *feature map's* overfitting is what's tested --
    # otherwise the learned GP nugget also absorbs the noise and masks the NN early-stopping effect.
    guarded = rmse(DeepKernelGP(early_stopping=True, random_state=0, noise_bounds=None))
    unguarded = rmse(DeepKernelGP(early_stopping=False, random_state=0, noise_bounds=None))
    assert guarded <= unguarded * 1.05  # early stopping is competitive-or-better; generous margin


def test_nugget_is_learned_by_maximum_likelihood():
    # the GP head's nugget is optimized (MLE), like the length-scale: on clean data it stays near the
    # floor (near-interpolation); on noisy data the likelihood learns a larger nugget to smooth.
    rng = np.random.RandomState(6)
    X = rng.uniform(-1, 1, size=(80, 3))
    f = np.sin(3 * X[:, [0]]) + X[:, [1]] ** 2
    clean = DeepKernelGP().fit(X, f)
    noisy = DeepKernelGP().fit(X, f + 0.3 * rng.standard_normal((80, 1)))
    assert clean.gp.model["noise"] < 1e-4  # near the floor -> near-interpolation
    assert noisy.gp.model["noise"] > 10 * clean.gp.model["noise"]  # learned to smooth the noise


def test_forwards_selection_knobs_to_the_gp_head():
    X, y = _data(50)
    # noise_bounds=None keeps the nugget fixed at `noise` (verifies the knob is forwarded)
    fixed = DeepKernelGP(noise=5e-3, noise_bounds=None).fit(X, y)
    assert fixed.gp.model["noise"] == 5e-3
    # optimizer=None freezes the length-scale at its start (the same meaning Dace gives it)
    frozen = DeepKernelGP(optimizer=None, theta=0.7).fit(X, y)
    assert np.allclose(frozen.gp.model["theta"], 0.7)


def test_screening_fit_without_search_does_not_crash():
    # model-selection screening calls fit(optimize=False); the learn-nugget default must fall back to
    # a fixed nugget rather than ask Dace to learn a nugget without a search (which would raise).
    X, y = _data(40)
    model = DeepKernelGP().fit(X, y, optimize=False)
    assert np.all(np.isfinite(model.predict(X[:3]).y))


def test_refit_freezes_the_feature_map_and_absorbs_points():
    X, y = _data(60)
    model = DeepKernelGP(hidden_layer_sizes=(16, 8)).fit(X, y)
    coefs_before = [c.copy() for c in model._coefs]

    Xnew, ynew = _data(15, seed=9)
    score = model.refit(Xnew, ynew)  # prequential: scores the new points before absorbing them

    # the NN feature map is NOT retrained on refit -- only the GP head grows (this is the fast path)
    assert all(np.array_equal(a, b) for a, b in zip(coefs_before, model._coefs))
    assert isinstance(score, dict)
    # the added points are now known to the GP head: their variance is below that of far-away points
    near = np.mean(model.predict(Xnew, var=True).var)
    far = np.mean(model.predict(np.full_like(Xnew, 9.0), var=True).var)
    assert near < far and np.all(np.isfinite(model.predict(Xnew).y))


def test_falls_back_gracefully_when_too_small_to_validate():
    # a tiny design cannot spare a validation point; fit must not crash (early stopping auto-skips)
    X, y = _data(6)
    model = DeepKernelGP(early_stopping=True, validation_fraction=0.1).fit(X, y)
    assert model.nn_.validation_scores_ is None  # skipped
    assert np.all(np.isfinite(model.predict(X).y))
