#!/usr/bin/env python3
"""Per-system DOF figures for the new dimers, same layout as
azaindol_2d_scan_dofs.png: molecule with J1/J2 arrows + 20x20 replica grid.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/prokop/git/dftbplus/data/xyz"
NP = 20
SYS = [
    ("pyridone", f"{D}/pylac_relaxed.xyz", [(16,20,6),(5,23,17)], 1.0, 1.85,
     ("J1: N16-H20..O6", "J2: N5-H23..O17"), ("d1 = r(N16-H20)", "d2 = r(N5-H23)")),
    ("diazaphenalene", f"{D}/dzp_relaxed.xyz", [(9,20,23),(30,41,2)], 1.0, 2.05,
     ("J1: N9-H20..N23", "J2: N30-H41..N2"), ("d1 = r(N9-H20)", "d2 = r(N30-H41)")),
]
cmap = {"C": "#666666", "N": "#2b6cb0", "O": "#c53030", "H": "#bbbbbb"}

for name, path, J, d_lo, d_hi, tags, labs in SYS:
    L = open(path).read().splitlines()
    n = int(L[0]); el, xyz = [], []
    for ln in L[2:2+n]:
        t = ln.split(); el.append(t[0]); xyz.append([float(x) for x in t[1:4]])
    xyz = np.array(xyz)

    fig, (axm, axg) = plt.subplots(1, 2, figsize=(13, 6))
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(xyz[i]-xyz[j]) < 1.6:
                axm.plot([xyz[i,0],xyz[j,0]],[xyz[i,1],xyz[j,1]],"-",c="#999999",lw=1,zorder=2)
    for e,p in zip(el,xyz):
        axm.scatter(p[0],p[1],s=180 if e!="H" else 60,c=cmap[e],edgecolors="k",linewidths=0.5,zorder=3)
    for idx,(dn,pr,ac),col,tag in [(0,J[0],"#d62728",tags[0]),(1,J[1],"#1f77b4",tags[1])]:
        axm.annotate("",xy=xyz[ac,:2],xytext=xyz[dn,:2],
                     arrowprops=dict(arrowstyle="-|>",color=col,lw=2.5),zorder=4)
        for k, role in [(dn,"donor"),(pr,"H"),(ac,"acc")]:
            axm.scatter(*xyz[k,:2],s=300,facecolors="none",edgecolors=col,lw=2,zorder=5)
            axm.annotate(f"{el[k]}{k}",xyz[k,:2],textcoords="offset points",xytext=(6,6),fontsize=8,color=col)
        mid = 0.5*(xyz[dn,:2]+xyz[ac,:2]) + np.array([0.15,0.15])
        axm.text(*mid,tag,color=col,fontsize=10,weight="bold")
    axm.set_title(f"{name} dimer — two scanned H-bond junctions (relaxed scaffold)")
    axm.set_xlabel("x [A]"); axm.set_ylabel("y [A]"); axm.set_aspect("equal")

    d = np.linspace(d_lo, d_hi, NP)
    X, Y = np.meshgrid(d, d, indexing="ij")
    axg.scatter(X, Y, s=18, c="#2b6cb0")
    axg.set_xlabel(labs[0]+" [A]"); axg.set_ylabel(labs[1]+" [A]")
    axg.set_title(f"{NP}x{NP} = {NP*NP} replicas, d={d_lo}..{d_hi:.2f} A")
    axg.set_aspect("equal")
    fig.tight_layout()
    out = f"/home/prokop/git/dftbplus/rust_dftb/debug/{name}_2d_scan_dofs.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
