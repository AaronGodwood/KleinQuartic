"""Cells of the regular {r, s} map as left cosets, and the cellular chain complex over F_2.

Each group element g is a directed edge. With x of order 2, y of order s, xy of order r:
    vertex of g = g<y>,  edge of g = g<x>,  face of g = g<xy>.
Two cells are incident iff their cosets share an element.

Qubits on edges:  H_X = d1 (vertex checks),  H_Z = d2^T (face checks).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from psl import PSL2, Elem, find_generating_pair


def coset_labels(G: PSL2, elems: list[Elem], h: Elem) -> dict[Elem, int]:
    """Map each element to the index of its left coset g<h>."""
    H = G.cyclic(h)
    label: dict[Elem, int] = {}
    nxt = 0
    for g in elems:
        if g in label:
            continue
        for k in H:
            label[G.mul(g, k)] = nxt
        nxt += 1
    return label


@dataclass
class Complex:
    p: int
    r: int
    s: int
    x: Elem
    y: Elem
    flags: list[tuple[int, int, int]]  # (face, edge, vertex) per dart
    d1: np.ndarray  # V x E
    d2: np.ndarray  # E x F
    d1_mult: np.ndarray  # incidence with multiplicity, for degeneracy checks
    d2_mult: np.ndarray

    @property
    def V(self) -> int:
        return self.d1.shape[0]

    @property
    def E(self) -> int:
        return self.d1.shape[1]

    @property
    def F(self) -> int:
        return self.d2.shape[1]

    @property
    def H_X(self) -> np.ndarray:
        return self.d1

    @property
    def H_Z(self) -> np.ndarray:
        return self.d2.T


def build(p: int = 7, r: int = 7, s: int = 3) -> Complex:
    G = PSL2(p)
    elems = G.elements()
    x, y = find_generating_pair(G, r, s)
    face = coset_labels(G, elems, G.mul(x, y))
    edge = coset_labels(G, elems, x)
    vert = coset_labels(G, elems, y)

    flags = [(face[g], edge[g], vert[g]) for g in elems]
    V, E, F = len(set(vert.values())), len(set(edge.values())), len(set(face.values()))

    # Each edge has two darts, g and gx. Count incidences per dart: an entry of 2 would mean
    # a loop (both ends at one vertex) or an edge with the same face on both sides, which
    # cancels mod 2. For the Klein quartic every entry should be 0 or 1 (checked in tests).
    d1_mult = np.zeros((V, E), dtype=np.int64)
    d2_mult = np.zeros((E, F), dtype=np.int64)
    for f, e, v in flags:
        d1_mult[v, e] += 1
        d2_mult[e, f] += 1
    d1 = (d1_mult % 2).astype(np.uint8)
    d2 = (d2_mult % 2).astype(np.uint8)
    return Complex(p, r, s, x, y, flags, d1, d2, d1_mult, d2_mult)
