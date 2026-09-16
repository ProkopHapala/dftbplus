#!/usr/bin/env python3
"""Build the transferred-tautomer endpoint for diazaphenalene (only the donor
form exists) and plot all endpoint geometries with junction arrows.

Junctions (0-based), from make_dimers_2d_scan.py:
  pyridone_dimer (lactam):   J1 N16-H20..O6 ; J2 N5-H23..O17
  pyridone_isodimer (lactim): J1 O17-H22..N5 ; J2 O6-H23..N16
  diazaphenalene_dimer:      J1 N9-H20..N23 ; J2 N30-H41..N2
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/prokop/git/dftbplus/data/xyz"

def load(p):
    L = open(p).read().splitlines()
    n = int(L[0]); el, xyz = [], []
    for ln in L[2:2 + n]:
        t = ln.split(); el.append(t[0]); xyz.append([float(t[1]), float(t[2]), float(t[3])])
    return el, np.array(xyz)

def save(p, el, xyz, comment):
    with open(p, "w") as f:
        f.write(f"{len(el)}\n{comment}\n")
        for e, q in zip(el, xyz):
            f.write(f"{e:2s} {q[0]:12.6f} {q[1]:12.6f} {q[2]:12.6f}\n")

# --- diazaphenalene transferred tautomer: each proton -> 1.0 A from acceptor
el, xyz = load(f"{D}/diazaphenalene_dimer.xyz")
J = [(9, 20, 23), (30, 41, 2)]
g = xyz.copy()
for dn, pr, ac in J:
    a = (xyz[ac] - xyz[dn]); a /= np.linalg.norm(a)
    g[pr] = xyz[ac] - 1.0 * a
save(f"{D}/diazaphenalene_transferred.xyz", el, g, "both protons transferred (constructed)")

# --- plot endpoints with junction arrows
SYS = [
    ("pyridone_dimer",   f"{D}/pyridone_dimer.xyz",    [(16, 20, 6), (5, 23, 17)]),
    ("pyridone_isodimer",f"{D}/pyridone_isodimer.xyz", [(17, 22, 5), (6, 23, 16)]),
    ("diazaphenalene",   f"{D}/diazaphenalene_dimer.xyz", [(9, 20, 23), (30, 41, 2)]),
    ("diazaphenalene*",  f"{D}/diazaphenalene_transferred.xyz", [(9, 20, 23), (30, 41, 2)]),
]
cmap = {"C": "#666", "N": "#2b6cb0", "O": "#c53030", "H": "#bbb"}
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
for ax, (name, path, J) in zip(axes, SYS):
    el, xyz = load(path)
    for i in range(len(el)):
        for j in range(i + 1, len(el)):
            if np.linalg.norm(xyz[i] - xyz[j]) < 1.6:
                ax.plot([xyz[i,0], xyz[j,0]], [xyz[i,1], xyz[j,1]], "-", c="#999", lw=0.8, zorder=2)
    for e, p in zip(el, xyz):
        ax.scatter(p[0], p[1], s=140 if e != "H" else 45, c=cmap[e], edgecolors="k", lw=0.4, zorder=3)
    for (dn, pr, ac), col in zip(J, ["#d62728", "#1f77b4"]):
        ax.annotate("", xy=xyz[ac,:2], xytext=xyz[dn,:2],
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2), zorder=4)
        for k in (dn, pr, ac):
            ax.scatter(*xyz[k,:2], s=260, facecolors="none", edgecolors=col, lw=1.8, zorder=5)
            ax.annotate(f"{el[k]}{k}", xyz[k,:2], textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=col)
    ax.set_title(name, fontsize=11); ax.set_aspect("equal")
    ax.set_xlabel("x [A]"); ax.set_ylabel("y [A]")
fig.tight_layout()
out = "/home/prokop/git/dftbplus/debug/dimers_endpoints.png"
fig.savefig(out, dpi=150)
print("wrote", f"{D}/diazaphenalene_transferred.xyz")
print("wrote", out)
