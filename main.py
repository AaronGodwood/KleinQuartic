"""Main running script for getting output of the calculations involving the klein  quartic"""


import gf2
from cells import build
from distance import distances

p = 7
r = 7
s = 3

cx = build(p, r, s) #repeated defaults for clarity
V, E, F = cx.V, cx.E, cx.F
chi = V - E + F
rankx, rankz= gf2.rank(cx.H_X), gf2.rank(cx.H_Z)
k = E - rankx - rankz

z, x = distances(cx, all_roots=True)

print(f"PSL(2,{p}), {{{r},{s}}} map")
print(f"  generators: x = {cx.x}, y = {cx.y}")
print(f"  V = {V}, E = {E}, F = {F}, chi = {chi}, genus = {(2 - chi) // 2}")
print(f"  d1 d2 = 0 mod 2: {not (cx.d1.astype(int) @ cx.d2 % 2).any()}")
print(f"  max incidence multiplicity: d1 {cx.d1_mult.max()}, d2 {cx.d2_mult.max()}")
print(f"  rank H_X = {rankx}, rank H_Z = {rankz}")
print(f"  d_Z = {z.weight}  (shortest Z-logical, edges {z.support.tolist()})")
print(f"  d_X = {x.weight}  (shortest X-logical, edges {x.support.tolist()})")
print(f"  [[n, k, d]] = [[{E}, {k}, {min(z.weight, x.weight)}]]")
