"""Sublevel-set topology features: basins born/merging along a threshold sweep of a k-NN graph."""

import numpy as np

from ._util import _safe_float


def _persistence(ys, adj):
    """Zero-dimensional sublevel-set persistence of ``ys`` on the neighbor graph ``adj``.

    Points are inserted in ascending objective order (union-find, elder rule). A point with no
    already-inserted neighbor opens a new component (a basin is born); a point bridging distinct
    components merges them, killing the younger one and recording its lifetime ``death - birth``.

    Args:
        ys: Standardized outputs, shape ``(n,)``.
        adj: Neighbor adjacency list (from :meth:`Context.adjacency`).

    Returns:
        A tuple ``(persistences, n_basins, final_components, alive_seq, order)`` where
        ``persistences`` are the finite bar lengths (raw ``ys`` units), ``n_basins`` the number of
        components ever born, ``final_components`` those still alive at the end, ``alive_seq`` the
        live-component count after each insertion (aligned with ``order``), and ``order`` the
        ascending-value insertion order of the point indices.
    """
    n = ys.size
    order = np.argsort(ys, kind="stable")
    parent = np.full(n, -1, dtype=int)
    birth = np.zeros(n, dtype=float)
    added = np.zeros(n, dtype=bool)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    persistences = []
    alive_seq = np.zeros(n, dtype=int)
    alive = 0
    n_basins = 0

    for step, p in enumerate(order):
        p = int(p)
        added[p] = True
        v = ys[p]
        nbrs = [q for q in adj[p] if added[q]]
        if not nbrs:
            parent[p] = p
            birth[p] = v
            alive += 1
            n_basins += 1
        else:
            parent[p] = find(nbrs[0])
            for q in nbrs:
                rp, rq = find(p), find(q)
                if rp != rq:
                    # elder rule: the component born later (higher birth value) dies now.
                    if birth[rp] <= birth[rq]:
                        elder, younger = rp, rq
                    else:
                        elder, younger = rq, rp
                    persistences.append(v - birth[younger])
                    parent[younger] = elder
                    alive -= 1
        alive_seq[step] = alive

    return persistences, n_basins, alive, alive_seq, order


def _euler_curve(ys, adj, order):
    """Normalized Euler-characteristic (``V - E``) curve of the sublevel sets along the sweep.

    Args:
        ys: Standardized outputs, shape ``(n,)``.
        adj: Neighbor adjacency list.
        order: Ascending-value insertion order of point indices.

    Returns:
        A tuple ``(euler_mean, euler_final)`` -- the mean and final Euler characteristic across the
        threshold sweep, each divided by ``n``. High positive values mean many disconnected
        components (basins); negative values mean the graph closes up into cycles/loops.
    """
    n = ys.size
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    # edge "appears" at the rank of its later (higher-value) endpoint.
    e_at = np.zeros(n, dtype=float)
    for i in range(n):
        for j in adj[i]:
            if j > i:
                e_at[max(rank[i], rank[j])] += 1.0
    e_cum = np.cumsum(e_at)
    v_cum = np.arange(1, n + 1, dtype=float)
    chi = v_cum - e_cum
    return float(np.mean(chi)) / n, float(chi[-1]) / n


def compute(ctx) -> dict:
    """Sublevel-set topology features: how basins are born and merge over a threshold sweep.

    A k-NN graph is built on the normalized inputs; the objective threshold is swept from low to
    high and, at each level, the connected components of the points below the threshold are tracked
    via zero-dimensional persistent homology (union-find, elder rule). The number and prominence of
    the resulting basins, the range over which several coexist, and an Euler-characteristic summary
    together diagnose how multimodal / topologically rugged the landscape is.

    Args:
        ctx: A landscape :class:`Context` wrapping the labelled cloud ``(X, y)``.

    Returns:
        A flat dict of float features: the peak number of coexisting components, the number of
        basins born and their density, the largest / mean normalized basin persistence, the
        persistence entropy and the second-to-first prominence ratio, the threshold span over which
        multiple basins coexist, the number of final (infinite) components with a single-component
        flag, and the mean / final normalized Euler characteristic.
    """
    keys = [
        "peak_components",
        "n_basins",
        "basin_density",
        "max_persistence",
        "mean_persistence",
        "persistence_entropy",
        "prominence_ratio",
        "component_spread",
        "final_components",
        "single_component",
        "euler_mean",
        "euler_final",
    ]
    out = {k: np.nan for k in keys}

    try:
        ys = np.asarray(ctx.ys, dtype=float).ravel()
        n = ys.size
    except Exception:
        return out

    if n < 4 or not np.all(np.isfinite(ys)):
        return out

    try:
        k = int(np.clip(ctx.default_k(), 2, max(2, n - 1)))
        adj = ctx.adjacency(k)
    except Exception:
        return out

    yrange = float(np.ptp(ys))

    try:
        pers, n_basins, final_comp, alive_seq, order = _persistence(ys, adj)
    except Exception:
        return out

    out["n_basins"] = _safe_float(n_basins)
    out["basin_density"] = _safe_float(n_basins / n) if n > 0 else np.nan
    out["peak_components"] = _safe_float(np.max(alive_seq)) if alive_seq.size else np.nan
    out["final_components"] = _safe_float(final_comp)
    out["single_component"] = 1.0 if final_comp <= 1 else 0.0

    # Normalized finite persistences (basin prominences relative to the objective range).
    pers = np.asarray([p for p in pers if np.isfinite(p) and p >= 0.0], dtype=float)
    if yrange > 1e-12 and pers.size:
        norm_pers = np.clip(pers / yrange, 0.0, 1.0)
        out["max_persistence"] = _safe_float(np.max(norm_pers))
        out["mean_persistence"] = _safe_float(np.mean(norm_pers))
        # Persistence entropy: how evenly basin prominences are spread (1 = many equal basins).
        total = float(np.sum(norm_pers))
        if total > 1e-12 and norm_pers.size >= 2:
            probs = norm_pers / total
            probs = probs[probs > 0]
            ent = -float(np.sum(probs * np.log(probs)))
            out["persistence_entropy"] = _safe_float(ent / np.log(norm_pers.size))
        else:
            out["persistence_entropy"] = 0.0
        # Prominence ratio: second-largest bar / largest bar (near 1 => rival basins => multimodal).
        srt = np.sort(norm_pers)[::-1]
        out["prominence_ratio"] = _safe_float(srt[1] / srt[0]) if srt.size >= 2 and srt[0] > 1e-12 else 0.0
    else:
        out["max_persistence"] = 0.0
        out["mean_persistence"] = 0.0
        out["persistence_entropy"] = 0.0
        out["prominence_ratio"] = 0.0

    # Threshold span over which more than one basin coexists (proxy for basin separation).
    try:
        if yrange > 1e-12:
            ys_sorted = ys[order]
            dv = np.diff(ys_sorted)
            multi = alive_seq[:-1] > 1
            out["component_spread"] = _safe_float(float(np.sum(dv[multi])) / yrange)
        else:
            out["component_spread"] = 0.0
    except Exception:
        out["component_spread"] = np.nan

    try:
        em, ef = _euler_curve(ys, adj, order)
        out["euler_mean"] = _safe_float(em)
        out["euler_final"] = _safe_float(ef)
    except Exception:
        pass

    return {kk: _safe_float(v) for kk, v in out.items()}
