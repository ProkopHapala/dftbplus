#!/usr/bin/env python3
"""Render the azaindole 2D scan energy surface from the rhai output.

Reads lines "d1 d2 E" from stdin (or the file given as argv[1]) and writes
debug/azaindol_2d_scan_Emap.png — filled contours + min/saddle annotations.
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
pts = []
in_map = False
for ln in src.splitlines():
    if "ENERGY MAP" in ln: in_map = True; continue
    if "=== done" in ln: in_map = False
    if in_map:
        t = ln.split()
        if len(t) == 3:
            try: pts.append(tuple(float(x) for x in t))
            except ValueError: pass
pts = np.array(pts)
NP = int(round(np.sqrt(len(pts))))
assert NP * NP == len(pts), f"expected square grid, got {len(pts)}"
d1 = pts[:, 0].reshape(NP, NP); d2 = pts[:, 1].reshape(NP, NP)
E = pts[:, 2].reshape(NP, NP)
Ek = (E - E.min()) * 627.509474  # kcal/mol above minimum

fig, ax = plt.subplots(figsize=(7.5, 6.5))
levels = np.linspace(0, min(Ek.max(), 40), 41)
cf = ax.contourf(d1, d2, Ek, levels=levels, cmap="viridis", extend="max")
cs = ax.contour(d1, d2, Ek, levels=np.arange(0, 41, 5), colors="w",
                linewidths=0.5, alpha=0.6)
ax.clabel(cs, fmt="%.0f", fontsize=7)
im = np.unravel_index(np.argmin(Ek), Ek.shape)
ax.scatter(*[d1[im], d2[im]], marker="*", s=300, c="gold", edgecolors="k", zorder=5)
ax.annotate(f"min {Ek[im]:.1f}", (d1[im], d2[im]), textcoords="offset points",
            xytext=(8, -12), fontsize=9, weight="bold")
# diagonal = synchronous path
ax.plot([d1.min(), d1.max()], [d1.min(), d1.max()], "w--", lw=1, alpha=0.7)
ax.text(d1.min() + 0.02, d1.min() + 0.06, "synchronous\nd1=d2", color="w", fontsize=8, alpha=0.9)
ax.set_xlabel("d1 = r(N6-H21) [A]"); ax.set_ylabel("d2 = r(N15-H27) [A]")
ax.set_title(f"7-azaindole dimer SCC PES, {NP}x{NP} batch, kcal/mol above min")
fig.colorbar(cf, label="E - E_min [kcal/mol]")
fig.tight_layout()
out = "/home/prokop/git/dftbplus/debug/azaindol_2d_scan_Emap.png"
fig.savefig(out, dpi=150)
print("wrote", out)
print(f"E_min={E.min():.6f} Ha at d1={d1[im]:.2f} d2={d2[im]:.2f}; "
      f"E_max={E.max():.6f} Ha; span={Ek.max():.2f} kcal/mol")
