#!/usr/bin/env python
"""Plot poly(p-phenylene): geometry with bond-order coloring + band structure.

Reuses test_dftbcore_ppp (DFTBcore run + real-space bond-order analysis).
Top row: 3 unit cells of the RELAXED geometry (geo_end.gen), bond width/color
  proportional to Mulliken BO, para carbons marked.
Bottom row: folded band structure, X at +-0.5 on BOTH zone edges, Fermi zoom.

Usage:  python plot_ppp.py [--noShow]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from test_dftbcore_ppp import (run_ppp, RING_BONDS, A0, WORK, HA2EV)

NC_SHOW = 3   # unit cells drawn


def kx_centered(kpts, E):
    """Center fractional kx on [-0.5, 0.5] and duplicate the physical X point
    (kx=0.5) at BOTH boundaries, like plot_peierls.py."""
    kx = kpts[:, 0]
    kx_c = np.where(kx > 0.5, kx - 1.0, kx)
    iX = np.where(np.isclose(kx, 0.5))[0][0]
    kx_p = np.concatenate([kx_c, [-0.5]])
    E_p = np.concatenate([E, E[iX:iX + 1]], axis=0)
    order = np.argsort(kx_p)
    kx_p, E_p = kx_p[order], E_p[order]
    assert np.isclose(kx_p[0], -0.5) and np.isclose(kx_p[-1], 0.5), "BZ edges missing"
    assert np.max(np.abs(E_p[0] - E_p[-1])) < 1e-10, "edge eigenvalues differ"
    return kx_p, E_p


def draw_cell(ax, xyz, bo, xoff, bomax):
    """Draw one cell: C/H atoms + BO-weighted bonds. xyz: relaxed (10,3)."""
    C = xyz[:6] + [xoff, 0, 0]
    H = xyz[6:] + [xoff, 0, 0]
    bonds = []
    for i, j in RING_BONDS:
        bonds.append((C[i], C[j], bo[(i, j)] / bomax))
    # inter-ring: para-right (atom0) of this cell -> para-left (atom3) of next
    bonds.append((C[0], xyz[3] + [xoff + A0, 0, 0], bo['inter'] / bomax))
    for h, i in zip(H, [1, 2, 4, 5]):
        bonds.append((C[i], h, None))
    for p1, p2, w in bonds:
        if w is None:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', lw=1.0, zorder=2)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=plt.cm.viridis(w), lw=1 + 8 * w, zorder=3,
                    solid_capstyle='round')
    ax.scatter(C[:, 0], C[:, 1], s=18, c='k', zorder=5)
    ax.scatter(H[:, 0], H[:, 1], s=8, c='steelblue', zorder=4)
    ax.scatter([C[0, 0], C[3, 0]], [C[0, 1], C[3, 1]], s=18, c='crimson',
               marker='s', zorder=6)   # para (fixed) carbons as small squares


def main(noShow=False):
    sk_path = os.environ.get('DFTB_SK_PATH', os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/'))
    deltas = [0.0, 0.10, 0.15]
    nk = 32
    results = [run_ppp(f"ppp_plot_d{d:.2f}", sk_path, d, nk=nk) for d in deltas]
    bomax = max(max(r['bo'][b] for b in RING_BONDS) for r in results)
    bomax = max(bomax, max(r['bo_inter'] for r in results))

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5),
                             gridspec_kw={'height_ratios': [1, 2.2]})
    for col, r in enumerate(results):
        axg, axb = axes[0, col], axes[1, col]
        for m in range(NC_SHOW):
            draw_cell(axg, r['xyz'], r['bo'], m * A0, bomax)
            axg.axvline(m * A0, color='k', ls=':', lw=0.7)
        axg.axvline(NC_SHOW * A0, color='k', ls=':', lw=0.7)
        axg.set_aspect('equal'); axg.axis('off')
        axg.set_xlim(-0.4, NC_SHOW * A0 + 0.4)
        axg.set_ylim(-3.4, 3.4)
        axg.set_title(f"$\delta$={r['delta']:.2f} A   "
                      f"BO int/para/mid: {r['bo_inter']:.2f}/{r['bo_para']:.2f}/{r['bo_mid']:.2f}",
                      fontsize=10)

        kx, E = kx_centered(r['kpts'], r['E'] * HA2EV)
        nocc = 14
        axb.plot(kx, E, 'b-', lw=0.9)
        axb.axhline(0.5 * (E[:, nocc - 1].max() + E[:, nocc].min()),
                    color='r', ls='--', lw=0.8)
        e_lo = E[:, nocc - 3].min() - 1
        e_hi = E[:, nocc + 3].max() + 1
        axb.set_ylim(e_lo, e_hi)
        axb.set_xlim(-0.5, 0.5)
        axb.set_xticks([-0.5, 0, 0.5]); axb.set_xticklabels(['-X', '$\Gamma$', 'X'])
        axb.set_title(f"E gap = {r['gap']:.2f} eV", fontsize=10)
        if col == 0:
            axb.set_ylabel('E [eV]')

    fig.suptitle(f'PPP Peierls distortion (para C fixed, moved apart along x) '
                 f'— DFTB 3ob-3-1, nk={nk}')
    out = WORK / 'ppp_peierls.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"saved {out}")
    if not noShow:
        plt.show()


if __name__ == '__main__':
    main(noShow='--noShow' in sys.argv)
