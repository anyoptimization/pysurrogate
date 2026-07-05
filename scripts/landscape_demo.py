"""Runnable demo: print a compact structural-feature table across benchmark landscapes."""

import numpy as np

from pysurrogate.landscape import Landscape


def lhs(n, d, seed, lo=-5.0, hi=5.0):
    """A Latin-hypercube design on ``[lo, hi]^d``."""
    rng = np.random.default_rng(seed)
    cuts = np.linspace(0.0, 1.0, n + 1)
    pts = np.zeros((n, d))
    for j in range(d):
        u = rng.uniform(size=n)
        pts[:, j] = (cuts[:-1] + u * (cuts[1:] - cuts[:-1]))[rng.permutation(n)]
    return lo + pts * (hi - lo)


def rotation_matrix(d, seed):
    """A random orthogonal rotation matrix."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q


def build_benchmarks(n=300, d=5, cond=100.0):
    """Return ``{name: (X, y)}`` for a handful of landscapes with known structure."""
    X = lhs(n, d, seed=1)
    w = np.logspace(0.0, np.log10(cond), d)
    Q = rotation_matrix(d, seed=7)
    rng = np.random.default_rng(3)
    base = np.sum(X**2, axis=1)
    # Rotated ellipsoid: SAME axis-aligned cloud, rotation lives only in y (Hessian Qᵀ diag(w) Q).
    return {
        "sphere": (X, base),
        "ellipsoid_aa": (X, np.sum(w * X**2, axis=1)),
        "ellipsoid_rot": (X, np.sum(w * (X @ Q.T) ** 2, axis=1)),
        "ridge": (X, (X @ np.array([1.0, 0.5, 0.0, 0.0, 0.0])) ** 2),
        "rastrigin": (X, 10.0 * d + np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)),
        "rosenbrock": (X, np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)),
        "linear": (X, X @ np.arange(1.0, d + 1.0)),
        "noisy_sphere": (X, base + rng.normal(0.0, 0.3 * float(np.std(base)), size=n)),
    }


# The headline structural features (short label -> namespaced feature key).
COLUMNS = [
    ("rot", "rotation.hess_rot"),
    ("offaxis", "rotation.hess_offaxis"),
    ("aniso", "curvature.curv_anisotropy"),
    ("cond", "curvature.condition_number"),
    ("eff_dim", "active_subspace.participation_ratio"),
    ("smooth", "variogram.smoothness_exp"),
    ("nugget", "variogram.nugget_ratio"),
    ("basins", "topology.n_basins"),
    ("localmin", "multimodality.local_min_frac"),
    ("separ", "separability.separability_index"),
    ("lin_r2", "meta_model.lin_r2"),
    ("fdc", "dispersion.fdc"),
]


def main():
    """Build the benchmarks, compute features, and print the compact table."""
    benches = build_benchmarks()
    landscapes = {name: Landscape(X, y, seed=0) for name, (X, y) in benches.items()}

    n_feats = len(next(iter(landscapes.values())).features())
    print(f"pysurrogate landscape demo -- {len(landscapes)} functions x {n_feats} features each\n")

    header = f"{'function':14s}" + "".join(f"{label:>9s}" for label, _ in COLUMNS)
    print(header)
    print("-" * len(header))
    for name, lp in landscapes.items():
        row = f"{name:14s}"
        for _, key in COLUMNS:
            v = lp.get(key)
            row += "      nan" if np.isnan(v) else f"{v:9.2f}"
        print(row)

    print("\nRead: rot/offaxis/aniso -> rotation & anisotropy; cond -> conditioning;")
    print("eff_dim -> active dimensions; smooth/nugget -> smoothness & noise;")
    print("basins/localmin -> multimodality; separ -> separability; lin_r2 -> linear trend;")
    print("fdc -> global searchability (high = single funnel).\n")

    print("=" * 60)
    print("Full structural report for the rotated ill-conditioned ellipsoid:")
    print("=" * 60)
    print(landscapes["ellipsoid_rot"].report())


if __name__ == "__main__":
    main()
