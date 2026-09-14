#!/usr/bin/env python3
"""Generate the 7-azaindole-dimer 2D proton-transfer scan inputs.

Two equivalent N..H..N junctions, one independent distance DOF each:
  J1: donor N6 - H21 ... acceptor N10   (d1 = r(N6-H21), axis N6->N10)
  J2: donor N15 - H27 ... acceptor N1   (d2 = r(N15-H27), axis N15->N1)
(0-based atom indices in azaindol_dimer.xyz; monomer A = atoms 0-8+18-23
 heavy+H, monomer B = 9-17+24-29.)

Outputs:
  data/xyz/azaindol_2d_scan_20x20.xyz  - 400-frame movie (frame i1*20+i2)
  debug/azaindol_2d_scan_dofs.png      - molecule + junction arrows + grid
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

XYZ = "/home/prokop/git/dftbplus/data/xyz/azaindol_dimer.xyz"
OUT_XYZ = "/home/prokop/git/dftbplus/data/xyz/azaindol_2d_scan_20x20.xyz"
OUT_PNG = "/home/prokop/git/dftbplus/rust_dftb/debug/azaindol_2d_scan_dofs.png"

NP = 20          # points per axis
D0, DSTEP = 1.0, 0.05   # Angstrom — same convention as the GC 1D scan

# 0-based indices: (donor, proton, acceptor)
J1 = (6, 21, 10)
J2 = (15, 27, 1)

def load(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0])
    el, xyz = [], []
    for ln in lines[2:2 + n]:
        t = ln.split()
        el.append(t[0]); xyz.append([float(t[1]), float(t[2]), float(t[3])])
    return el, np.array(xyz)

el, xyz = load(XYZ)
assert len(el) == 30

def axis(donor, acceptor):
    v = xyz[acceptor] - xyz[donor]
    return v / np.linalg.norm(v)

a1 = axis(J1[0], J1[2])   # N6 -> N10
a2 = axis(J2[0], J2[2])   # N15 -> N1
dNN1 = np.linalg.norm(xyz[J1[2]] - xyz[J1[0]])
dNN2 = np.linalg.norm(xyz[J2[2]] - xyz[J2[0]])
print(f"J1: N{J1[0]}-H{J1[1]}...N{J1[2]}  N..N={dNN1:.3f} A")
print(f"J2: N{J2[0]}-H{J2[1]}...N{J2[2]}  N..N={dNN2:.3f} A")

# ---- 400-frame movie ----------------------------------------------------
os.makedirs(os.path.dirname(OUT_XYZ), exist_ok=True)
with open(OUT_XYZ, "w") as f:
    for i1 in range(NP):
        d1 = D0 + i1 * DSTEP
        for i2 in range(NP):
            d2 = D0 + i2 * DSTEP
            g = xyz.copy()
            g[J1[1]] = xyz[J1[0]] + d1 * a1
            g[J2[1]] = xyz[J2[0]] + d2 * a2
            f.write(f"30\nframe {i1*NP+i2}: d1={d1:.3f} d2={d2:.3f}\n")
            for e, p in zip(el, g):
                f.write(f"{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
print(f"wrote {OUT_XYZ} ({NP*NP} frames)")

# ---- figure: molecule with junction arrows + scan grid ------------------
fig, (axm, axg) = plt.subplots(1, 2, figsize=(13, 6))

cmap = {"C": "#666666", "N": "#2b6cb0", "H": "#bbbbbb"}
for e, p in zip(el, xyz):
    axm.scatter(p[0], p[1], s=180 if e != "H" else 60,
                c=cmap[e], edgecolors="k", linewidths=0.5, zorder=3)
# bonds (rough, distance < 1.6 A)
for i in range(len(el)):
    for j in range(i + 1, len(el)):
        if np.linalg.norm(xyz[i] - xyz[j]) < 1.6:
            axm.plot([xyz[i,0], xyz[j,0]], [xyz[i,1], xyz[j,1]],
                     "-", c="#999999", lw=1, zorder=2)
for idx, (dn, pr, ac), col, tag in [(0, J1, "#d62728", "J1: N6-H21..N10"),
                                    (1, J2, "#1f77b4", "J2: N15-H27..N1")]:
    axm.annotate("", xy=xyz[ac,:2], xytext=xyz[dn,:2],
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5), zorder=4)
    axm.scatter(*xyz[dn,:2], s=300, facecolors="none", edgecolors=col, lw=2, zorder=5)
    axm.scatter(*xyz[ac,:2], s=300, facecolors="none", edgecolors=col, lw=2, zorder=5)
    axm.scatter(*xyz[pr,:2], s=140, facecolors="none", edgecolors=col, lw=2, zorder=5)
    mid = 0.5*(xyz[dn,:2]+xyz[ac,:2]) + np.array([0.15, 0.15])
    axm.text(*mid, tag, color=col, fontsize=10, weight="bold")
    for k, name in [(dn, f"N{dn}"), (pr, f"H{pr}"), (ac, f"N{ac}")]:
        axm.annotate(name, xyz[k,:2], textcoords="offset points",
                     xytext=(6, 6), fontsize=8, color=col)
axm.set_title("7-azaindole dimer — two scanned H-bond junctions")
axm.set_xlabel("x [A]"); axm.set_ylabel("y [A]"); axm.set_aspect("equal")

# grid panel: every (d1,d2) replica
d = D0 + np.arange(NP) * DSTEP
X, Y = np.meshgrid(d, d, indexing="ij")
axg.scatter(X, Y, s=18, c="#2b6cb0")
axg.axhline(dNN1 - 1.0, color="#d62728", ls="--", lw=1)
axg.axvline(dNN2 - 1.0, color="#1f77b4", ls="--", lw=1)
axg.text(D0, dNN1 - 0.96, "donor..acceptor midpoint", color="#888", fontsize=8)
axg.set_xlabel("d1 = r(N6-H21) [A]"); axg.set_ylabel("d2 = r(N15-H27) [A]")
axg.set_title(f"{NP}x{NP} = {NP*NP} replicas, d={D0}..{d[-1]:.2f} A")
axg.set_aspect("equal")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=150)
print(f"wrote {OUT_PNG}")
