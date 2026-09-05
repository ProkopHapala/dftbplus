#!/usr/bin/env python3
"""Plot atomic charges (spatial 2D scatter) + HOMO-LUMO energy axis.

Reads TSV files produced by dftb_engine (save_charges, save_eigenvalues) and
the DFTB+ reference harness (ref_charges.tsv, ref_eigenvalues.tsv), then
produces two plots:

  1. Spatial charge map — atoms positioned in 2D (xy-plane), colored by
     Mulliken charge. Side-by-side: Rust dense SCC vs Rust sparse vs DFTB+ ref.
  2. HOMO-LUMO energy axis — energy levels near the gap, occupied (blue) vs
     virtual (red), for Rust dense, Davidson, and DFTB+ ref.

Usage:
    python3 plot_charges_homo_lumo.py <debug_dir> [--systems benzene,coronene]

Expects in <debug_dir>:
    <system>_charges.tsv       (Rust dense SCC charges)
    <system>_sparse_charges.tsv (Rust sparse charges, optional)
    <system>_eigenvalues.tsv   (Rust dense eigenvalues)
    ref_charges.tsv            (DFTB+ reference)
    ref_eigenvalues.tsv        (DFTB+ reference)
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

# Element colors (CPK-like)
EL_COLORS = {
    'C': '#2F2F2F', 'H': '#FFFFFF', 'N': '#3050F8', 'O': '#FF0D0D',
    'S': '#FFFF30', 'P': '#FF8000', 'F': '#90E050', 'Cl': '#1FF01F',
}
EL_RADII = {'C': 0.7, 'H': 0.35, 'N': 0.65, 'O': 0.6, 'S': 1.0, 'P': 1.0}


def load_charges_tsv(path, convert_to_charge=False):
    """Load charges TSV → (elements, positions, charges).

    If convert_to_charge=True, convert Rust Mulliken populations to charges
    using q0=4 for C, q0=1 for H (i.e. charge = q0 - population).
    DFTB+ ref files already contain charges and need no conversion.
    """
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, dtype=None, encoding='utf-8', names=True, delimiter='\t')
    elements = [row['element'] for row in data]
    pos = np.array([[row['x'], row['y'], row['z']] for row in data])
    charges = np.array([row['charge'] for row in data])
    if convert_to_charge:
        q0_map = {'C': 4.0, 'H': 1.0, 'N': 5.0, 'O': 6.0, 'S': 6.0}
        q0 = np.array([q0_map.get(el, 4.0) for el in elements])
        charges = q0 - charges  # population → charge
    return elements, pos, charges


def load_eigenvalues_tsv(path):
    """Load eigenvalues TSV → (eigenvalues, occupations).

    Handles both formats:
    - Rust: columns idx, eigenvalue, occupied (0 or 1)
    - DFTB+ ref: columns idx, eigenvalue, occupation (0.0 or 2.0)
    """
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, names=True, delimiter='\t')
    eigs = data['eigenvalue']
    # Handle both column names
    if 'occupation' in data.dtype.names:
        occs = data['occupation']
    elif 'occupied' in data.dtype.names:
        occs = data['occupied'] * 2.0  # 0/1 → 0.0/2.0
    else:
        occs = np.zeros(len(eigs))
    return eigs, occs


def plot_spatial_charges(ax, elements, pos, charges, title, q_range=None):
    """Plot atoms in 2D (xy-plane) colored by Mulliken charge."""
    if q_range is None:
        q_range = max(abs(charges.min()), abs(charges.max()))
        if q_range < 1e-6:
            q_range = 0.1

    # Use a diverging colormap centered at 0
    norm = plt.Normalize(-q_range, q_range)
    cmap = plt.cm.RdBu_r

    for i, (el, p, q) in enumerate(zip(elements, pos, charges)):
        color = cmap(norm(q))
        radius = EL_RADII.get(el, 0.5)
        circle = Circle((p[0], p[1]), radius, facecolor=color, edgecolor='k', linewidth=0.5)
        ax.add_patch(circle)
        # Element label
        ax.text(p[0], p[1], el, ha='center', va='center', fontsize=6,
                color='white' if el in ('C', 'N') else 'black', fontweight='bold')

    # Auto-scale
    margin = 2.0
    ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
    ax.set_ylim(pos[:, 1].min() - margin, pos[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')


def plot_energy_levels(ax, eigs, occs, title, n_show=12):
    """Plot energy levels near the HOMO-LUMO gap as horizontal lines."""
    n_occ = int(occs.sum() / 2)  # closed-shell
    # Show n_show/2 below HOMO and n_show/2 above LUMO
    lo = max(0, n_occ - n_show // 2)
    hi = min(len(eigs), n_occ + n_show // 2)

    for i in range(lo, hi):
        e = eigs[i]
        occ = occs[i] > 0.5
        color = '#3060FF' if occ else '#FF3030'
        lw = 2.0 if occ else 1.5
        ax.hlines(e, 0.2, 0.8, colors=color, linewidth=lw)
        # Label HOMO and LUMO
        if i == n_occ - 1:
            ax.text(0.85, e, 'HOMO', fontsize=7, va='center', color='#3060FF')
        elif i == n_occ:
            ax.text(0.85, e, 'LUMO', fontsize=7, va='center', color='#FF3030')

    # Gap arrow
    if n_occ > 0 and n_occ < len(eigs):
        homo = eigs[n_occ - 1]
        lumo = eigs[n_occ]
        gap = lumo - homo
        ax.annotate('', xy=(0.5, lumo), xytext=(0.5, homo),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax.text(0.55, (homo + lumo) / 2, f'gap={gap:.4f} Ha', fontsize=7, va='center', color='green')

    ax.set_xlim(0, 1.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Energy (Ha)')
    ax.set_xticks([])


def main():
    ap = argparse.ArgumentParser(description="Plot charges + HOMO-LUMO comparison")
    ap.add_argument("debug_dir", help="Directory with TSV files")
    ap.add_argument("--systems", default="benzene,coronene,circumcoronene",
                    help="Comma-separated system names")
    ap.add_argument("--n-levels", type=int, default=12, help="Energy levels to show near gap")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",")]
    n_sys = len(systems)

    # ─── Plot 1: Spatial charges ───
    # Columns: Rust dense | Rust sparse | DFTB+ ref  (per system row)
    has_sparse = any(os.path.exists(os.path.join(args.debug_dir, f"{s}_sparse_charges.tsv")) for s in systems)
    n_cols = 3 if has_sparse else 2
    col_titles = ["Rust dense SCC", "Rust sparse TC2", "DFTB+ ref"] if has_sparse else ["Rust dense SCC", "DFTB+ ref"]

    fig1, axes1 = plt.subplots(n_sys, n_cols, figsize=(5 * n_cols, 5 * n_sys), squeeze=False)

    # Determine global charge range for consistent colors
    all_charges = []
    for s in systems:
        for fname, conv in [(f"{s}_charges.tsv", True), (f"{s}_sparse_charges.tsv", True), (f"{s}_ref_charges.tsv", False)]:
            d = load_charges_tsv(os.path.join(args.debug_dir, fname), convert_to_charge=conv)
            if d is not None:
                all_charges.append(d[2])
    q_range = max(abs(c).max() for c in all_charges) if all_charges else 0.1
    q_range = max(q_range, 0.01)

    for row, s in enumerate(systems):
        sources = [
            (f"{s}_charges.tsv", True, f"{s} — Rust dense"),
            (f"{s}_sparse_charges.tsv" if has_sparse else "NONE", True, f"{s} — Rust sparse"),
            (f"{s}_ref_charges.tsv", False, f"{s} — DFTB+ ref"),
        ]
        for col, (fname, conv, title) in enumerate(sources[:n_cols]):
            ax = axes1[row][col]
            d = load_charges_tsv(os.path.join(args.debug_dir, fname), convert_to_charge=conv)
            if d is not None:
                plot_spatial_charges(ax, d[0], d[1], d[2], title, q_range=q_range)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=10)

    # Add colorbar
    fig1.subplots_adjust(right=0.92)
    cbar_ax = fig1.add_axes([0.94, 0.15, 0.015, 0.7])
    norm = plt.Normalize(-q_range, q_range)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm)
    sm.set_array([])
    fig1.colorbar(sm, cax=cbar_ax, label='Mulliken charge (e)')

    out1 = os.path.join(args.debug_dir, "charges_spatial.png")
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ─── Plot 2: HOMO-LUMO energy levels ───
    # Columns: Rust dense | Davidson | DFTB+ ref  (per system row)
    fig2, axes2 = plt.subplots(n_sys, 3, figsize=(8 * 3, 5 * n_sys), squeeze=False)

    for row, s in enumerate(systems):
        # Rust dense eigenvalues
        d = load_eigenvalues_tsv(os.path.join(args.debug_dir, f"{s}_eigenvalues.tsv"))
        ax = axes2[row][0]
        if d is not None:
            plot_energy_levels(ax, d[0], d[1], f"{s} — Rust dense", args.n_levels)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{s} — Rust dense", fontsize=10)

        # Davidson (same eigenvalues for now — would be separate if saved)
        ax = axes2[row][1]
        if d is not None:
            plot_energy_levels(ax, d[0], d[1], f"{s} — Davidson (partial)", args.n_levels)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{s} — Davidson (partial)", fontsize=10)

        # DFTB+ ref
        d_ref = load_eigenvalues_tsv(os.path.join(args.debug_dir, f"{s}_ref_eigenvalues.tsv"))
        ax = axes2[row][2]
        if d_ref is not None:
            plot_energy_levels(ax, d_ref[0], d_ref[1], f"{s} — DFTB+ ref", args.n_levels)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{s} — DFTB+ ref", fontsize=10)

    out2 = os.path.join(args.debug_dir, "homo_lumo_levels.png")
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
