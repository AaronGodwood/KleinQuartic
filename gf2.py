"""Linear algebra over F_2. Don't use numpy.linalg.matrix_rank as it works over real numbers only."""
from __future__ import annotations

import numpy as np


def rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form over F_2. Returns (nonzero rows, pivot columns)."""
    A = (M.copy() % 2).astype(np.uint8)
    rows, cols = A.shape
    r = 0
    pivots: list[int] = []

    for c in range(cols):
        if r == rows:
            break

        nz = np.nonzero(A[r:, c])[0]

        if nz.size == 0:
            continue

        piv = r + nz[0]
        A[[r, piv]] = A[[piv, r]]
        others = np.nonzero(A[:, c])[0]
        others = others[others != r]
        A[others] ^= A[r]
        pivots.append(c)
        r += 1
    return A[:r], pivots


def rank(M: np.ndarray) -> int:
    return len(rref(M)[1])


class RowSpace:
    """Membership test for the row space of M over F_2."""

    def __init__(self, M: np.ndarray):
        self.R, self.pivots = rref(M)


    def reduce(self, v: np.ndarray) -> np.ndarray:
        v = (v.copy() % 2).astype(np.uint8)
        for row, c in zip(self.R, self.pivots):
            if v[c]:
                v ^= row
        return v


    def __contains__(self, v: np.ndarray) -> bool:
        return not self.reduce(v).any()
