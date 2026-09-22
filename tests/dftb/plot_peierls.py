#!/usr/bin/env python
"""Plot Peierls/SSH results for the 2-atom carbon chain.

Produces one figure with:
  top    - chain geometry with bonds colored/thickened by Mulliken bond order
           (from real-space P(R)*S(R) blocks of the exported k-dependent DM)
  bottom - band structure E(k) for each dimerization

Reuses run_chain() from test_dftbcore_peierls.py.
Saves to work/peierls.png.  Run from tests/dftb/:
    python plot_peierls.py [--noShow]
"""

import sys
import os
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from test_dftbcore_peierls import run_chain, HA2EV

WORK = Path(__file__).parent / 'work'


def plot_bonds(ax, a, d_intra, bo_intra, bo_inter, ncell=2):
    """Draw the chain: atoms as dots, bonds with lw/color ~ bond order."""
    bo_max = 1.0
    cmap = plt.cm.viridis
    for m in range(-1, ncell + 1):
        x1, x2 = m * a, m * a + d_intra          # intra-cell bond
        ax.plot([x1, x2], [0, 0], lw=1 + 9 * bo_intra / bo_max,
                color=cmap(bo_intra / bo_max), solid_capstyle='round')
        ax.plot([x2, x2 + (a - d_intra)], [0, 0], lw=1 + 9 * bo_inter / bo_max,
                color=cmap(bo_inter / bo_max), solid_capstyle='round')
    xs = [m * a + dx for m in range(-1, ncell + 1) for dx in (0, d_intra)]
    ax.scatter(xs, np.zeros(len(xs)), s=180, c='k', zorder=5)
    ax.set_xlim(-0.6 * a, ncell * a + 0.6 * a); ax.set_ylim(-1, 1)
    ax.set_yticks([]); ax.set_aspect('equal')
    ax.set_title(f"d_intra={d_intra:.2f} A   BO: {bo_intra:.3f} / {bo_inter:.3f}",
                 fontsize=9)


def main():
    sk_path = os.environ.get('DFTB_SK_PATH', os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/'))
    a = 2.60
    d_list = [a / 2, 1.20, 1.10]           # uniform, moderate, strong dimerization
    nk_band = 32                            # denser mesh for smooth bands
    nk_bo = 8                               # same mesh as the test for bond orders

    fig, axes = plt.subplots(2, len(d_list), figsize=(4.2 * len(d_list), 7.0),
                             gridspec_kw={'height_ratios': [1, 1.6]})
    kx_frac = np.arange(nk_band) / nk_band  # folded BZ 0..1 (edge at 0.5)
    kx_plot = np.where(kx_frac > 0.5, kx_frac - 1.0, kx_frac)  # recenter to [-.5,.5]

    for j, d in enumerate(d_list):
        r = run_chain(f"plot{d:.2f}", sk_path, d, nk=nk_band, a=a)
        E_eV = r['E'] * HA2EV
        # extend to include BOTH zone edges: E(-X)=E(+X) since kx=0.5 == -0.5
        # kx_plot already contains +0.5; add the -0.5 image with the same eigenvalues
        kx_ext = np.concatenate([kx_plot, [-0.5]])
        E_ext = np.vstack([E_eV, E_eV[nk_band // 2][None, :]])
        order = np.argsort(kx_ext)
        axb = axes[1, j]
        for ib in range(E_ext.shape[1]):
            axb.plot(kx_ext[order], E_ext[order, ib], 'b-', lw=1.0)
        axb.axvline(0.5, color='gray', ls=':', lw=0.8)
        axb.axhline(0.5 * (E_eV[:, 3].max() + E_eV[:, 4].min()), color='r', ls='--', lw=0.8)
        axb.set_xlim(-0.5, 0.5); axb.set_ylim(-18, 15)  # zoom to Fermi region (pi bands)
        axb.set_xticks([-0.5, 0, 0.5])
        axb.set_xticklabels(['-X', 'Γ', 'X'])
        axb.set_ylabel('E [eV]' if j == 0 else None)
        axb.set_title(f"E gap = {r['gap']:.2f} eV", fontsize=9)

        plot_bonds(axes[0, j], a, d, r['bo_intra'], r['bo_inter'])

    fig.suptitle(f"Peierls distortion of carbon chain (a={a} A) — DFTB 3ob-3-1, nk={nk_band}")
    out = WORK / 'peierls.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"saved {out}")
    if '--noShow' not in sys.argv:
        plt.show()


if __name__ == '__main__':
    main()
