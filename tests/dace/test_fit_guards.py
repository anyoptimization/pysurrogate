"""Guards on the batched Dace likelihood: a degenerate trend design must not crash the search."""

import numpy as np

from pysurrogate.dace.corr import Gaussian
from pysurrogate.dace.fit import batch_obj_grad
from pysurrogate.dace.regr import LinearRegression, QuadraticRegression

GAUSS = Gaussian()


def test_batch_obj_grad_survives_singular_trend_design():
    """A singular trend design (``p <= n``) must be reported infeasible, not crash the search.

    ``batch_obj_grad`` solves the GLS trend system in one batched ``np.linalg.solve(G, ...)``. Here
    the design sites lie on a 1-D line embedded in 2-D (``X = [t, t]``), so a *quadratic* trend has
    exactly collinear basis columns (``x0**2 == x1**2 == x0*x1``): ``G`` is exactly singular and the
    un-guarded batched solve raises ``LinAlgError`` for the whole batch -- violating
    ``DaceProblem``'s never-raise contract and aborting the theta search. ``p = 6 <= n = 8``, so the
    ``regr(X).shape[1] > n`` guard does not catch it; the condition-number guard must.
    """
    n = 8
    t = np.linspace(0.0, 1.0, n)
    X = np.column_stack([t, t])  # 1-D data embedded in 2-D -> quadratic basis is rank-deficient
    y = np.sin(3.0 * t)
    thetas = np.array([[0.5, 0.5], [1.0, 2.0]])

    obj, grad, feasible = batch_obj_grad(X, y, QuadraticRegression(), GAUSS, thetas)

    assert not feasible.any()  # every candidate flagged infeasible, not crashed
    assert np.all(np.isinf(obj))  # infinite objective (rejected), never a NaN or a finite garbage value
    assert np.allclose(grad, 0.0)  # infeasible -> zero gradient


def test_batch_obj_grad_full_rank_trend_is_feasible():
    """A well-posed design at the same shape stays feasible -- the guard does not over-reject."""
    rng = np.random.default_rng(0)
    X = rng.random((12, 2))
    y = np.sin(3.0 * X[:, 0]) + X[:, 1]
    obj, grad, feasible = batch_obj_grad(X, y, LinearRegression(), GAUSS, np.array([[0.5, 0.5]]))
    assert feasible.all()
    assert np.all(np.isfinite(obj))
