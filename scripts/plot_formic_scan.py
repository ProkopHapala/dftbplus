#!/usr/bin/env python3
"""Plot formic dimer proton-transfer scan results.

Reads TSV files from debug/formic_dimer_scan/ produced by the Rust test
formic_scan_plots.rs. Generates:

  1. 1D energy profile: E(t) CPU vs GPU, with barrier annotation
  2. 1D Mulliken charges: q(t) for H-bond atoms (H, donor O, acceptor O)
  3. 2D energy contour: E(t1, t2) with synchronous path overlay
  4. 2D Mulliken charge contour: q_H1(t1, t2)

Usage:
  python3 scripts/plot_formic_scan.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "rust_dftb", "debug", "formic_dimer_scan")
OUT_DIR = DATA_DIR

def load_1d(path):
    """Load 1D scan TSV. Returns dict of arrays."""
    data = np.loadtxt(path, comments='#')
    t = data[:, 0]
    e_cpu = data[:, 1]
    e_gpu = data[:, 2]
    # Columns: q_cpu, q_gpu for each of 6 atoms (H1, D1, A1, H2, D2, A2)
    q_cpu = {}
    q_gpu = {}
    labels = ['H1', 'D1', 'A1', 'H2', 'D2', 'A2']
    for k, label in enumerate(labels):
        q_cpu[label] = data[:, 3 + 2*k]
        q_gpu[label] = data[:, 4 + 2*k]
    return t, e_cpu, e_gpu, q_cpu, q_gpu

def load_2d(path, n):
    """Load 2D scan TSV. Returns t1, t2 grids and value matrix.
    Handles NaN values for unconverged points."""
    data = np.loadtxt(path, comments='#', dtype=float)
    t1 = np.zeros((n, n))
    t2 = np.zeros((n, n))
    vals = np.full((n, n), np.nan)  # default NaN (unconverged)
    for row in data:
        i, j = int(row[0]), int(row[1])
        t1[i, j] = row[2]
        t2[i, j] = row[3]
        val = row[4]
        if not np.isnan(val):
            vals[i, j] = val
    return t1, t2, vals

def plot_1d_energy(t, e_cpu, e_gpu, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, e_cpu, 'b-o', ms=3, lw=1.5, label='CPU (f64, DIIS)')
    ax.plot(t, e_gpu, 'r--s', ms=3, lw=1.5, label='GPU (f32, DIIS)')
    ax.set_xlabel('Proton transfer coordinate t', fontsize=12)
    ax.set_ylabel('Total SCC energy (Ha)', fontsize=12)
    ax.set_title('Formic dimer 1D proton-transfer PES', fontsize=13)
    # Barrier annotation
    e0 = e_cpu[0]
    e_ts = e_cpu[len(t)//2]
    barrier = e_ts - e0
    ax.annotate(f'Barrier: {barrier*27.211:.2f} eV ({barrier:.4f} Ha)',
                xy=(1.0, e_ts), xytext=(1.3, e_ts + 0.02),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

def plot_1d_charges(t, q_cpu, q_gpu, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # H-bond 1: H1, D1(O), A1(O)
    ax = axes[0]
    for label, color, ls in [('H1', 'red', '-'), ('D1', 'blue', '--'), ('A1', 'green', '--')]:
        ax.plot(t, q_gpu[label], color=color, ls=ls, lw=1.5, label=f'{label} (GPU)')
        ax.plot(t, q_cpu[label], color=color, ls=ls, lw=0.8, alpha=0.5, label=f'{label} (CPU)')
    ax.set_ylabel('Mulliken charge (|e|)', fontsize=11)
    ax.set_title('H-bond 1: H[4] — O[3](donor) → O[7](acceptor)', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color='gray', ls=':', alpha=0.5)

    # H-bond 2: H2, D2(O), A2(O)
    ax = axes[1]
    for label, color, ls in [('H2', 'red', '-'), ('D2', 'blue', '--'), ('A2', 'green', '--')]:
        ax.plot(t, q_gpu[label], color=color, ls=ls, lw=1.5, label=f'{label} (GPU)')
        ax.plot(t, q_cpu[label], color=color, ls=ls, lw=0.8, alpha=0.5, label=f'{label} (CPU)')
    ax.set_xlabel('Proton transfer coordinate t', fontsize=12)
    ax.set_ylabel('Mulliken charge (|e|)', fontsize=11)
    ax.set_title('H-bond 2: H[9] — O[8](donor) → O[2](acceptor)', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color='gray', ls=':', alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

def plot_1d_parity(t, e_cpu, e_gpu, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    de = (e_gpu - e_cpu) * 27.211  # eV
    ax.plot(t, de, 'k-o', ms=3, lw=1)
    ax.set_xlabel('Proton transfer coordinate t', fontsize=12)
    ax.set_ylabel('E(GPU) − E(CPU)  (eV)', fontsize=12)
    ax.set_title('Energy parity: GPU vs CPU along 1D scan', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', ls='-', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

def plot_2d_energy(t1, t2, e, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    # Mask NaN (unconverged) regions
    e_masked = np.ma.masked_invalid(e)
    n_conv = e_masked.count()
    n_total = e.size
    # Shift to relative energy (eV)
    e_rel = (e_masked - e_masked.min()) * 27.211
    # Show unconverged regions as grey
    ax.pcolormesh(t1, t2, np.ma.masked_invalid(np.zeros_like(e)), cmap='gray',
                  alpha=0.5, shading='auto')
    im = ax.contourf(t1, t2, e_rel, levels=30, cmap='viridis')
    ax.contour(t1, t2, e_rel, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(im, ax=ax, label='Relative energy (eV)')
    # Overlay synchronous path t1=t2
    ax.plot([0, 2], [0, 2], 'r--', lw=1.5, label='Synchronous (t1=t2)')
    ax.set_xlabel('Proton 1 transfer: t1', fontsize=12)
    ax.set_ylabel('Proton 2 transfer: t2', fontsize=12)
    ax.set_title(f'Formic dimer 2D PES (GPU SCC) — {n_conv}/{n_total} converged', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

def plot_2d_charge(t1, t2, q, out_path, atom_label='H1'):
    fig, ax = plt.subplots(figsize=(8, 7))
    q_masked = np.ma.masked_invalid(q)
    n_conv = q_masked.count()
    n_total = q.size
    im = ax.contourf(t1, t2, q_masked, levels=30, cmap='RdBu_r',
                     norm=Normalize(vmin=q_masked.min(), vmax=q_masked.max()))
    ax.contour(t1, t2, q_masked, levels=15, colors='black', alpha=0.2, linewidths=0.5)
    plt.colorbar(im, ax=ax, label=f'Mulliken charge {atom_label} (|e|)')
    ax.plot([0, 2], [0, 2], 'k--', lw=1, alpha=0.5, label='Synchronous')
    ax.set_xlabel('Proton 1 transfer: t1', fontsize=12)
    ax.set_ylabel('Proton 2 transfer: t2', fontsize=12)
    ax.set_title(f'Formic dimer 2D Mulliken charge: {atom_label} — {n_conv}/{n_total} converged', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

def main():
    # 1D plots
    path_1d = os.path.join(DATA_DIR, "scan_1d.tsv")
    if not os.path.exists(path_1d):
        print(f"ERROR: {path_1d} not found. Run the Rust test first:")
        print("  RUST_DFTB_SK_DIR=/path/to/mio-1-1 cargo test --test formic_scan_plots -- --ignored --nocapture")
        sys.exit(1)

    print("=== Loading 1D scan data ===")
    t, e_cpu, e_gpu, q_cpu, q_gpu = load_1d(path_1d)
    print(f"  {len(t)} points, t=[{t[0]:.2f}, {t[-1]:.2f}]")

    print("=== Plotting 1D ===")
    plot_1d_energy(t, e_cpu, e_gpu, os.path.join(OUT_DIR, "1d_energy.png"))
    plot_1d_charges(t, q_cpu, q_gpu, os.path.join(OUT_DIR, "1d_charges.png"))
    plot_1d_parity(t, e_cpu, e_gpu, os.path.join(OUT_DIR, "1d_parity.png"))

    # 2D plots
    path_2d_e = os.path.join(DATA_DIR, "scan_2d_energy.tsv")
    path_2d_q = os.path.join(DATA_DIR, "scan_2d_charge_H1.tsv")
    if os.path.exists(path_2d_e):
        print("\n=== Loading 2D scan data ===")
        # Read n from header
        with open(path_2d_e) as f:
            for line in f:
                if 'n_t1' in line:
                    n = int(line.split('n_t2=')[1].strip())
                    break
        t1, t2, e_2d = load_2d(path_2d_e, n)
        print(f"  {n}x{n} grid")

        print("=== Plotting 2D ===")
        plot_2d_energy(t1, t2, e_2d, os.path.join(OUT_DIR, "2d_energy.png"))

        if os.path.exists(path_2d_q):
            _, _, q_2d = load_2d(path_2d_q, n)
            plot_2d_charge(t1, t2, q_2d, os.path.join(OUT_DIR, "2d_charge_H1.png"))
    else:
        print(f"\n(Skipping 2D plots: {path_2d_e} not found)")

    print(f"\n=== All plots saved to {OUT_DIR}/ ===")

if __name__ == '__main__':
    main()
