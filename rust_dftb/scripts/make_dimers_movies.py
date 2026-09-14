#!/usr/bin/env python3
"""Generate 20x20 .xyz movies for the pyridone and diazaphenalene 2D scans,
using the RELAXED donor-form scaffold.
"""
import numpy as np, os

D = "/home/prokop/git/dftbplus/data/xyz"
NP = 20
SYS = [
    ("pyridone",   f"{D}/pylac_relaxed.xyz", [(16,20,6),(5,23,17)], 1.0, 1.85),
    ("dzp",        f"{D}/dzp_relaxed.xyz",   [(9,20,23),(30,41,2)], 1.0, 2.05),
]
def load(p):
    L = open(p).read().splitlines()
    n = int(L[0]); el, xyz = [], []
    for ln in L[2:2+n]:
        t = ln.split(); el.append(t[0]); xyz.append([float(x) for x in t[1:4]])
    return el, np.array(xyz)

for name, path, J, d_lo, d_hi in SYS:
    el, xyz = load(path)
    ax = []
    for dn, pr, ac in J:
        v = xyz[ac] - xyz[dn]; ax.append(v / np.linalg.norm(v))
    out = f"{D}/{name}_2d_scan_{NP}x{NP}.xyz"
    with open(out, "w") as f:
        for i1 in range(NP):
            d1 = d_lo + i1 * (d_hi - d_lo) / (NP - 1)
            for i2 in range(NP):
                d2 = d_lo + i2 * (d_hi - d_lo) / (NP - 1)
                g = xyz.copy()
                g[J[0][1]] = xyz[J[0][0]] + d1 * ax[0]
                g[J[1][1]] = xyz[J[1][0]] + d2 * ax[1]
                f.write(f"{len(el)}\nframe {i1*NP+i2}: d1={d1:.3f} d2={d2:.3f}\n")
                for e, p in zip(el, g):
                    f.write(f"{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    print("wrote", out)
