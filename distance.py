"""Exact distance of a surface code via shortest homologically nontrivial cycles.

d_Z: shortest cycle in the primal graph (vertices, edges) not in im d2   (a Z-logical).
d_X: shortest cycle in the dual graph (faces, edges)    not in im d1^T (an X-logical).
d = min(d_X, d_Z).

Method: for a root vertex v, take a BFS tree T_v. Every non-tree edge (a, b) closes the
cycle P(v, a) + (a, b) + P(b, v). Homological nontriviality over F_2 satisfies the
3-path condition, so the shortest nontrivial cycle is one of these for any root v lying
on it. The map is vertex-transitive (G acts transitively on vertices and on faces), so a
single root suffices; `all_roots=True` checks every root as a consistency test.

The candidate is the F_2 sum of the edge sets, so if the two tree paths overlap, the
overlap cancels and its weight is that of a genuine cycle.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from gf2 import RowSpace


@dataclass
class Logical:
    weight: int
    support: np.ndarray  # edge indices (qubits)
    root: int


def _endpoints(incidence: np.ndarray) -> np.ndarray:
    """incidence: nodes x edges, each column weight 2. Returns E x 2 endpoint array."""
    ends = np.array([np.nonzero(incidence[:, e])[0] for e in range(incidence.shape[1])])
    if ends.ndim != 2 or ends.shape[1] != 2:
        raise ValueError("every edge must have exactly two distinct endpoints")
    return ends


def _bfs(adj: list[list[tuple[int, int]]], root: int):
    """Returns (parent node, parent edge, depth) per node."""
    n = len(adj)
    par = [-1] * n
    par_edge = [-1] * n
    depth = [-1] * n
    depth[root] = 0
    q = deque([root])
    while q:
        u = q.popleft()
        for w, e in adj[u]:
            if depth[w] < 0:
                depth[w] = depth[u] + 1
                par[w], par_edge[w] = u, e
                q.append(w)
    return par, par_edge, depth


def _path_edges(par, par_edge, node: int) -> list[int]:
    out = []
    while par[node] >= 0:
        out.append(par_edge[node])
        node = par[node]
    return out


def shortest_nontrivial(incidence: np.ndarray, boundaries: np.ndarray,
                        all_roots: bool = False) -> Logical:
    """Shortest cycle in the graph given by `incidence` not in the row space of `boundaries`.

    incidence:  nodes x edges (each column weight 2)
    boundaries: rows spanning the trivial cycles (in edge space)
    """
    n_nodes, n_edges = incidence.shape
    ends = _endpoints(incidence)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for e, (a, b) in enumerate(ends):
        adj[a].append((b, e))
        adj[b].append((a, e))

    trivial = RowSpace(boundaries)
    best: Logical | None = None
    for root in (range(n_nodes) if all_roots else [0]):
        par, par_edge, depth = _bfs(adj, root)
        tree = set(par_edge) - {-1}
        for e, (a, b) in enumerate(ends):
            if e in tree:
                continue
            # No pruning on depth[a] + depth[b] + 1: overlapping tree paths cancel,
            # so the true weight can be smaller than the walk length.
            v = np.zeros(n_edges, dtype=np.uint8)
            for f in _path_edges(par, par_edge, a) + _path_edges(par, par_edge, b) + [e]:
                v[f] ^= 1
            w = int(v.sum())
            if v in trivial:
                continue
            if best is None or w < best.weight:
                best = Logical(w, np.nonzero(v)[0], root)
    if best is None:
        raise ValueError("no nontrivial cycle: k = 0?")
    return best


def distances(cx, all_roots: bool = False) -> tuple[Logical, Logical]:
    """Returns (shortest Z-logical, shortest X-logical) for a built Complex."""
    z = shortest_nontrivial(cx.d1, cx.H_Z, all_roots)    # primal cycles mod face boundaries
    x = shortest_nontrivial(cx.d2.T, cx.H_X, all_roots)  # dual cycles mod vertex coboundaries
    return z, x
