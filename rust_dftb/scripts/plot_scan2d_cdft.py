#!/usr/bin/env python3
"""Render neutral + CDFT charge-transfer 2D scan maps side by side.

Parses "=== MAP <tag> ===" blocks of "d1 d2 E" lines from stdin or argv[1].
Panels: each map in kcal/mol above its own min, plus E_CT - E_neutral
difference maps. Writes debug/<out>.png (argv[2], default cdft_maps).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
out_name = sys.argv[2] if len(sys.argv) > 2 else "pyridone_2d_scan_cdft"

maps = {}
cur = None
for ln in src.splitlines():
    if ln.startswith("=== MAP"):
        cur = ln.split()[2]
        maps[cur] = []
        continue
    if cur and ln.startswith("==="):
        cur = None
        continue
    if cur:
        t = ln.split()
        if len(t) == 3:
            try:
                maps[cur].append(tuple(float(x) for x in t))
            except ValueError:
                pass

tags = list(maps)
assert tags, "no MAP blocks found"
NP = int(round(np.sqrt(len(maps[tags[0]]))))
grids = {}
for tag, pts in maps.items():
    p = np.array(pts)
    assert len(p) == NP * NP, f"{tag}: {len(p)} != {NP}x{NP}"
    grids[tag] = (p[:, 0].reshape(NP, NP), p[:, 1].reshape(NP, NP),
                  p[:, 2].reshape(NP, NP))
d1, d2, _ = grids[tags[0]]

n_map = len(tags)
ct_tags = [t for t in tags if t != "NEUTRAL"]
ref = grids.get("NEUTRAL")
n_pan = n_map + (len(ct_tags) if ref is not None else 0)
fig, axes = plt.subplots(1, n_pan, figsize=(5.6 * n_pan, 5.4),
                         constrained_layout=True)
if n_pan == 1:
    axes = [axes]

for ax, tag in zip(axes, tags):
    _, _, E = grids[tag]
    Ek = (E - E.min()) * 627.509474
    cf = ax.contourf(d1, d2, Ek, levels=np.linspace(0, min(Ek.max(), 40), 41),
                     cmap="viridis", extend="max")
    cs = ax.contour(d1, d2, Ek, levels=np.arange(0, 41, 5), colors="w",
                    linewidths=0.5, alpha=0.6)
    ax.clabel(cs, fmt="%.0f", fontsize=7)
    im = np.unravel_index(np.argmin(Ek), Ek.shape)
    ax.scatter(d1[im], d2[im], marker="*", s=280, c="gold", edgecolors="k", zorder=5)
    ax.plot([d1.min(), d1.max()], [d1.min(), d1.max()], "w--", lw=1, alpha=0.6)
    ax.set_title(f"{tag}  (min={E.min():.4f} Ha)", fontsize=10)
    ax.set_xlabel("d1 = r(N16-H20) [A]"); ax.set_ylabel("d2 = r(N5-H23) [A]")

# difference panels: each CT map minus neutral at the same grid point
for ax, tag in zip(axes[n_map:], ct_tags):
    dE = (grids[tag][2] - ref[2]) * 627.509474
    vlim = np.percentile(np.abs(dE), 98)
    cf = ax.contourf(d1, d2, dE, levels=np.linspace(0, vlim, 41),
                     cmap="YlOrRd")
    step = max(5, int(round(vlim / 8)))
    cs = ax.contour(d1, d2, dE, levels=np.arange(0, vlim + step, step),
                    colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, fmt="%.0f", fontsize=7)
    ax.plot([d1.min(), d1.max()], [d1.min(), d1.max()], "w--", lw=1, alpha=0.6)
    ax.set_title(f"{tag} − NEUTRAL [kcal/mol]\n(excitation energy)", fontsize=10)
    fig.colorbar(cf, ax=ax, shrink=0.85)
    ax.set_xlabel("d1 [A]"); ax.set_ylabel("d2 [A]")

out = f"/home/prokop/git/dftbplus/debug/{out_name}.png"
fig.savefig(out, dpi=150)
print("wrote", out)
for tag in tags:
    E = grids[tag][2]
    print(f"{tag}: E_min={E.min():.6f} E_max={E.max():.6f} span={(E.max()-E.min())*627.5:.1f} kcal")
