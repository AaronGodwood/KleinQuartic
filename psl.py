"""PSL(2, p) for prime p, as 2x2 matrices over F_p modulo ±I.

Elements are canonical 4-tuples (a, b, c, d): the lexicographically smaller of m and -m."""

from __future__ import annotations
from dataclasses import dataclass
from itertools import product

Elem = tuple[int, int, int, int]


@dataclass(frozen=True)
class PSL2:
    p: int



    def canon(self, m: Elem) -> Elem:
        neg = tuple((-v) % self.p for v in m)
        return min(tuple(v % self.p for v in m), neg)  # type: ignore[return-value]

    @property
    def identity(self) -> Elem:
        return self.canon((1, 0, 0, 1))

    def mul(self, m: Elem, n: Elem) -> Elem:
        a, b, c, d = m
        e, f, g, h = n
        return self.canon((a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h))


    def inv(self, m: Elem) -> Elem:
        a, b, c, d = m  # det = 1 so inverse is the adjugate
        return self.canon((d, -b, -c, a))



    def elements(self) -> list[Elem]:
        p = self.p
        return sorted({self.canon(m) for m in product(range(p), repeat=4)
                       if (m[0] * m[3] - m[1] * m[2]) % p == 1})



    def order(self, g: Elem) -> int:
        k, h, e = 1, g, self.identity
        while h != e:
            h = self.mul(h, g)
            k += 1
        return k
