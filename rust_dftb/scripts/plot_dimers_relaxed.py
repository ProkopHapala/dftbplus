#!/usr/bin/env python3
"""Plot the four relaxed endpoint structures with junction arrows."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/prokop/git/dftbplus/data/xyz"
SYS = [
    ("pyridone lactam  E=-32.35907",   f"{D}/pylac_relaxed.xyz", [(16,20,6),(5,23,17)]),
    ("pyridone lactim  E=-32.36804",   f"{D}/pyiso_relaxed.xyz", [(17,22,5),(6,23,16)]),
    ("diazaphenalene   E=-53.41990",   f"{D}/dzp_relaxed.xyz",   [(9,20,23),(30,41,2)]),
    ("diazaphenalene*  E=-53.41984",   f"{D}/dzpT_relaxed.xyz",  [(9,20,23),(30,41,2)]),
]
cmap = {"C": "#666", "N": "#2b6cb0", "O": "#c53030", "H": "#bbb"}
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
for ax, (name, path, J) in zip(axes, SYS):
    L = open(path).read().splitlines()
    n = int(L[0]); el, xyz = [], []
    for ln in L[2:2+n]:
        t = ln.split(); el.append(t[0]); xyz.append([float(x) for x in t[1:4]])
    xyz = np.array(xyz)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(xyz[i]-xyz[j]) < 1.6:
                ax.plot([xyz[i,0],xyz[j,0]],[xyz[i,1],xyz[j,1]],"-",c="#999",lw=0.8,zorder=2)
    for e,p in zip(el,xyz):
        ax.scatter(p[0],p[1],s=140 if e!="H" else 45,c=cmap[e],edgecolors="k",lw=0.4,zorder=3)
    for (dn,pr,ac),col in zip(J,["#d62728","#1f77b4"]):
        ax.annotate("",xy=xyz[ac,:2],xytext=xyz[dn,:2],
                    arrowprops=dict(arrowstyle="-|>",color=col,lw=2),zorder=4)
        for k in (dn,pr,ac):
            ax.scatter(*xyz[k,:2],s=260,facecolors="none",edgecolors=col,lw=1.8,zorder=5)
            ax.annotate(f"{el[k]}{k}",xyz[k,:2],textcoords="offset points",xytext=(5,5),fontsize=7,color=col)
    ax.set_title(name,fontsize=11); ax.set_aspect("equal")
    ax.set_xlabel("x [A]"); ax.set_ylabel("y [A]")
fig.tight_layout()
out = "/home/prokop/git/dftbplus/debug/dimers_relaxed.png"
fig.savefig(out,dpi=150); print("wrote",out)
