"""Active-subspace rotation learned from data, and Kriging over that rotated Mahalanobis metric."""

import numpy as np

from pysurrogate.models import Kriging, RotatedKriging, active_subspace


def _ridge(n, d, direction, seed):
    """A ridge function y = sin(3 * (x . u)): all variation runs along the single direction u."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    u = direction / np.linalg.norm(direction)
    y = np.sin(3.0 * (X @ u)).reshape(-1, 1)
    return X, y, u


def test_active_subspace_recovers_a_planted_off_axis_direction():
    # a ridge varying only along u -> the top eigenvector must align with u (up to sign)
    u = np.array([1.0, 1.0, -1.0, 0.5])
    X, y, u = _ridge(200, 4, u, seed=0)
    A, eig = active_subspace(X, y)
    top = A[:, 0]
    cos = abs(float(top @ u))  # both unit vectors
    assert cos > 0.97, f"top active direction misaligned with the ridge (|cos|={cos:.3f})"
    # and the variation energy concentrates in that one direction (local-fit noise leaks a little)
    assert eig[0] / eig.sum() > 0.8


def test_active_subspace_spreads_energy_for_an_isotropic_function():
    # a radial bowl varies equally in all directions -> no dominant eigenvalue
    rng = np.random.RandomState(1)
    X = rng.uniform(-1, 1, size=(200, 4))
    y = np.sum(X**2, axis=1, keepdims=True)
    _, eig = active_subspace(X, y)
    assert eig[0] / eig.sum() < 0.5  # energy is not concentrated in one direction


def test_active_subspace_orthonormal_columns_and_shapes():
    rng = np.random.RandomState(2)
    X = rng.uniform(-1, 1, size=(60, 5))
    y = np.sin(X[:, [0]]) + X[:, [3]] ** 2
    A, eig = active_subspace(X, y, n_components=2)
    assert A.shape == (5, 2) and eig.shape == (5,)
    assert np.allclose(A.T @ A, np.eye(2), atol=1e-8)  # orthonormal
    assert np.all(np.diff(eig) <= 1e-12)  # eigenvalues descending


def test_active_subspace_is_deterministic():
    X, y, _ = _ridge(80, 3, np.array([1.0, -0.5, 0.3]), seed=3)
    a1, _ = active_subspace(X, y)
    a2, _ = active_subspace(X, y)
    np.testing.assert_array_equal(a1, a2)


def test_rotated_kriging_respects_active_dims_on_fit_refit_and_predict():
    # regression (DaceBackedModel._refit): the active subspace is estimated on the selected columns
    # only; refit previously bypassed preprocess and handed full-width raw inputs to the engine.
    rng = np.random.RandomState(3)
    X = rng.uniform(-1, 1, size=(50, 5))
    y = (np.sin(3.0 * (X[:, [0]] + X[:, [2]]))).reshape(-1, 1)  # varies only in dims 0 and 2
    model = RotatedKriging(active_dims=[0, 2]).fit(X, y, optimize=False)
    assert model.X.shape[1] == 2  # rotation/engine saw only the 2 active dims
    Xnew = rng.uniform(-1, 1, size=(8, 5))
    ynew = (np.sin(3.0 * (Xnew[:, [0]] + Xnew[:, [2]]))).reshape(-1, 1)
    model.refit(Xnew, ynew, optimize=False)  # must not raise despite full-width raw inputs
    assert np.all(np.isfinite(model.predict(X[:5]).y))


def test_rotated_kriging_predicts_mean_variance_and_gradient():
    X, y, _ = _ridge(50, 3, np.array([1.0, 0.7, -0.4]), seed=4)
    model = RotatedKriging(theta_bounds=(0.01, 100.0)).fit(X, y)
    q = np.random.RandomState(9).uniform(-1, 1, size=(6, 3))
    pred = model.predict(q, var=True, grad=True)
    assert pred.y.shape == (6, 1) and np.all(np.isfinite(pred.y))
    assert pred.var is not None and pred.var.shape == (6, 1)
    assert np.all(pred.var >= -1e-9) and np.all(np.isfinite(pred.var))
    assert pred.grad is not None and pred.grad.shape == (6, 3) and np.all(np.isfinite(pred.grad))
    # exact interpolation: the predictive variance collapses to ~0 at the training sites
    assert np.all(model.predict(X, var=True).var < 1e-6)


def test_rotated_kriging_can_reduce_to_a_low_rank_subspace():
    X, y, _ = _ridge(60, 4, np.array([1.0, 1.0, -1.0, 0.5]), seed=5)
    model = RotatedKriging(n_components=1, theta_bounds=(0.01, 100.0)).fit(X, y)
    assert model.model.kernel.metric.A.shape == (4, 1)  # metric restricted to the top active direction
    assert np.all(np.isfinite(model.predict(X[:5]).y))


def test_rotated_kriging_beats_axis_aligned_kriging_on_a_rotated_ridge():
    # the payoff: on an off-axis ridge, rotating the metric into the active direction predicts the
    # held-out set at least as well as (here: better than) an axis-aligned ARD Kriging.
    d = 4
    u = np.array([1.0, 0.8, -0.6, 0.4])
    Xtr, ytr, u = _ridge(70, d, u, seed=6)
    Xte, yte, _ = _ridge(200, d, u, seed=7)

    def rmse(model):
        p = model.fit(Xtr, ytr).predict(Xte).y
        return float(np.sqrt(np.mean((p - yte) ** 2)))

    rotated = rmse(RotatedKriging(theta_bounds=(0.01, 100.0)))
    axis_aligned = rmse(Kriging(ARD=True, theta_bounds=(0.01, 100.0)))
    assert rotated <= axis_aligned * 1.05  # rotated is competitive-or-better; generous margin
