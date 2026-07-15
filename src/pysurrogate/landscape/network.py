"""Fitness-network features: assortativity, weighted clustering, and local-optima basin counts."""

import numpy as np

from ._util import _corr, _gini, _safe_float


def _edges(adj):
    """Unique undirected edges ``(i, j)`` with ``i < j`` from an adjacency list."""
    e = []
    for i, nb in enumerate(adj):
        for j in nb:
            if i < j:
                e.append((i, j))
    return e


def _weighted_clustering(adj, w):
    """Mean Onnela weighted clustering coefficient with per-edge weights ``w``.

    Each closed triangle contributes the geometric mean of its three edge weights; a node's
    coefficient normalizes by the number of neighbor pairs. Weights encode fitness similarity, so
    the score is high when tightly interconnected points also share similar fitness.

    Args:
        adj: Adjacency list of neighbor sets.
        w: Callable ``w(i, j)`` returning the ``[0, 1]`` similarity weight of edge ``i~j``.

    Returns:
        The mean weighted clustering coefficient over nodes with at least two neighbors, or
        ``np.nan`` when no such node exists.
    """
    vals = []
    for i, nb in enumerate(adj):
        nbl = list(nb)
        deg = len(nbl)
        if deg < 2:
            continue
        acc = 0.0
        for a in range(deg):
            ja = nbl[a]
            for b in range(a + 1, deg):
                jb = nbl[b]
                if jb in adj[ja]:
                    acc += (w(i, ja) * w(i, jb) * w(ja, jb)) ** (1.0 / 3.0)
        vals.append(2.0 * acc / (deg * (deg - 1)))
    if not vals:
        return np.nan
    return _safe_float(np.mean(vals))


def _mean_clustering(adj):
    """Mean unweighted local clustering coefficient over nodes with degree >= 2."""
    vals = []
    for i, nb in enumerate(adj):
        nbl = list(nb)
        deg = len(nbl)
        if deg < 2:
            continue
        links = 0
        for a in range(deg):
            for b in range(a + 1, deg):
                if nbl[b] in adj[nbl[a]]:
                    links += 1
        vals.append(2.0 * links / (deg * (deg - 1)))
    if not vals:
        return np.nan
    return _safe_float(np.mean(vals))


def _better_neighbor_graph(idx, y):
    """Directed "descend to best neighbor" graph over the k-NN structure.

    Args:
        idx: Neighbor-index array of shape ``(n, k)``.
        y: Fitness values, shape ``(n,)`` (minimization).

    Returns:
        ``succ`` of length ``n``: for each node the index of its strictly-best neighbor (lowest
        ``y``), or the node itself when it beats all neighbors (a sink / local optimum).
    """
    n = idx.shape[0]
    succ = np.arange(n)
    for i in range(n):
        nb = idx[i]  # self is already excluded by Context.knn
        if nb.size == 0:
            continue
        j = int(nb[np.argmin(y[nb])])
        if y[j] < y[i]:
            succ[i] = j
    return succ


def _descend(succ):
    """Follow the better-neighbor graph to a sink from every node.

    Args:
        succ: Successor array where ``succ[i] == i`` marks a sink.

    Returns:
        A tuple ``(basin, steps)`` where ``basin[i]`` is the sink node reached from ``i`` and
        ``steps[i]`` the number of descent hops taken (0 at a sink). Cycles are broken defensively.
    """
    n = succ.shape[0]
    basin = np.full(n, -1, dtype=int)
    steps = np.zeros(n, dtype=float)
    for i in range(n):
        path = []
        cur = i
        guard = 0
        while succ[cur] != cur and basin[cur] < 0 and guard <= n:
            path.append(cur)
            cur = int(succ[cur])
            guard += 1
        # ``cur`` is now a sink or a node with a known basin.
        sink = basin[cur] if basin[cur] >= 0 else cur
        base = steps[cur] if basin[cur] >= 0 else 0.0
        for offset, node in enumerate(reversed(path)):
            basin[node] = sink
            steps[node] = base + offset + 1
        if basin[i] < 0:
            basin[i] = sink
            steps[i] = base
    return basin, steps


def compute(ctx) -> dict:
    """Fitness-network (local-optima-network) structural features of the labelled cloud.

    A symmetrized k-NN graph turns the point cloud into a network whose nodes carry fitness. Three
    lenses read its geometry. (1) *Assortativity*: the Pearson correlation of fitness across
    connected edges -- in a smooth landscape neighbors share similar fitness (high), in a rugged one
    they do not. (2) *Weighted clustering*: how tightly interlinked neighborhoods are, weighting
    each triangle by the fitness similarity of its edges. (3) *Basins*: a directed graph pointing
    each node to its best neighbor turns sinks into attracting local optima -- counting them
    estimates modality, while descent-path lengths and basin-size concentration describe the
    funnel structure the optima carve out.

    Args:
        ctx: A landscape :class:`Context` wrapping the labelled cloud ``(X, y)``.

    Returns:
        A flat dict of float features describing fitness assortativity and edge smoothness, plain
        and fitness-weighted clustering, the fraction of points that are local-optima sinks
        (``basin_frac``) and the concentration of their basins, and the length of gradient-descent
        paths to those optima.
    """
    keys = [
        "fitness_assortativity",
        "edge_fitness_gap",
        "edge_gap_cv",
        "weighted_clustering",
        "clustering_coef",
        "basin_frac",
        "largest_basin_frac",
        "basin_size_gini",
        "sink_fitness_spread",
        "mean_path_length",
        "max_path_length",
    ]
    out = {k: np.nan for k in keys}

    try:
        n = ctx.n
        if n < 3:
            return out

        idx, _ = ctx.knn()
        ys = ctx.ys
        y = ctx.y

        adj = ctx.adjacency()
        edges = _edges(adj)

        # -- assortativity & edge smoothness ---------------------------------------------------
        if edges:
            ei = np.fromiter((e[0] for e in edges), dtype=int, count=len(edges))
            ej = np.fromiter((e[1] for e in edges), dtype=int, count=len(edges))
            # Symmetric edge assortativity: correlate over both orientations so it is unbiased.
            a = np.concatenate([ys[ei], ys[ej]])
            b = np.concatenate([ys[ej], ys[ei]])
            out["fitness_assortativity"] = _corr(a, b)

            gaps = np.abs(ys[ei] - ys[ej])
            out["edge_fitness_gap"] = _safe_float(np.mean(gaps))
            mg = np.mean(gaps)
            if mg > 1e-12:
                out["edge_gap_cv"] = _safe_float(np.std(gaps) / mg)

            # Fitness-similarity edge weights in [0, 1] for the weighted clustering coefficient.
            span = float(np.max(gaps))
            if span > 1e-12:
                wmap = {}
                for i, j, g in zip(ei, ej, gaps):
                    key = (int(i), int(j)) if i < j else (int(j), int(i))
                    wmap[key] = 1.0 - float(g) / span

                def _w(u, v, _wmap=wmap):
                    """Fitness-similarity weight of edge ``u~v`` (1 == identical fitness)."""
                    return _wmap.get((u, v) if u < v else (v, u), 0.0)

                out["weighted_clustering"] = _weighted_clustering(adj, _w)

        out["clustering_coef"] = _mean_clustering(adj)

        # -- better-neighbor (local-optima) network --------------------------------------------
        succ = _better_neighbor_graph(idx, y)
        basin, steps = _descend(succ)

        sinks = np.where(succ == np.arange(n))[0]
        n_sinks = int(sinks.size)
        # fraction of points that are sinks (local optima) of the better-neighbor graph.
        out["basin_frac"] = _safe_float(n_sinks / n)

        if n_sinks > 0:
            sizes = np.array([int(np.count_nonzero(basin == s)) for s in sinks], dtype=float)
            out["largest_basin_frac"] = _safe_float(sizes.max() / n)
            out["basin_size_gini"] = _gini(sizes)
            if n_sinks >= 2 and np.std(ys[sinks]) > 0:
                out["sink_fitness_spread"] = _safe_float(np.std(ys[sinks]))
            else:
                out["sink_fitness_spread"] = 0.0

        out["mean_path_length"] = _safe_float(np.mean(steps))
        out["max_path_length"] = _safe_float(np.max(steps))

    except Exception:
        return {k: _safe_float(v) for k, v in out.items()}

    return {k: _safe_float(v) for k, v in out.items()}
