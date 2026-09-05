#!/usr/bin/env python3
"""Plot H-bond switching scan results from hbond_ref CSV output.

Usage:
  python3 plot_hbond.py <csv_file> [--charges <data_dir>] [--out <prefix>]

For 1D scans: plots energy vs t, and Mulliken charges vs t (if --charges given).
For 2D scans: plots 2D PES contour, and charge heatmaps (if --charges given).
"""
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_csv(path):
    with open(path) as f:
        header = f.readline().strip().split(',')
        data = []
        for line in f:
            if line.strip():
                data.append([float(x) for x in line.strip().split(',')])
    return header, np.array(data)

def plot_1d(header, data, out_prefix, charges_dir=None):
    t = data[:, 1]
    e_total = data[:, 4]
    e_elec = data[:, 2]
    e_rep = data[:, 3]
    n_iter = data[:, 5].astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Energy plot
    ax = axes[0]
    ax.plot(t, e_total, 'b-o', label='Total energy', markersize=4)
    ax.plot(t, e_elec, 'g--', label='Electronic', alpha=0.7)
    ax.plot(t, e_rep, 'r--', label='Repulsive', alpha=0.7)
    ax.set_ylabel('Energy (Hartree)')
    ax.legend()
    ax.set_title(f'H-bond switching 1D scan')
    # Mark minimum
    imin = np.argmin(e_total)
    ax.axvline(t[imin], color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'min at t={t[imin]:.3f}\nE={e_total[imin]:.6f}',
                xy=(t[imin], e_total[imin]), fontsize=8,
                xytext=(t[imin]+0.05, e_total[imin]+0.01),
                arrowprops=dict(arrowstyle='->', color='gray'))

    # SCC iterations plot
    ax = axes[1]
    ax.plot(t, n_iter, 'k-o', markersize=4)
    ax.set_xlabel('t (proton transfer coordinate)')
    ax.set_ylabel('SCC iterations')
    ax.set_yscale('log')

    plt.tight_layout()
    out = f'{out_prefix}_energy.png'
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')
    plt.close()

    # Charges plot
    if charges_dir and os.path.isdir(charges_dir):
        n_atoms = None
        all_charges = []
        all_q0 = None
        for i in range(len(data)):
            pt_dir = os.path.join(charges_dir, f'pt_{i:04d}')
            ch_file = os.path.join(pt_dir, 'charges.txt')
            if not os.path.exists(ch_file):
                continue
            charges = []
            q0 = []
            with open(ch_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        charges.append(float(parts[1]))
                        q0.append(float(parts[2]))
            if n_atoms is None:
                n_atoms = len(charges)
                all_q0 = q0
            all_charges.append(charges)

        if all_charges:
            all_charges = np.array(all_charges)
            fig, ax = plt.subplots(figsize=(10, 6))
            for a in range(n_atoms):
                # Only plot atoms whose charge changes significantly
                if np.ptp(all_charges[:, a]) > 0.01:
                    ax.plot(t, all_charges[:, a], '-o', markersize=3,
                           label=f'atom {a}')
            ax.set_xlabel('t (proton transfer coordinate)')
            ax.set_ylabel('Mulliken charge')
            ax.set_title('Mulliken charges along 1D scan')
            ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            out = f'{out_prefix}_charges.png'
            plt.savefig(out, dpi=150)
            print(f'Saved: {out}')
            plt.close()

def plot_2d(header, data, out_prefix, charges_dir=None):
    n = int(np.sqrt(len(data)))
    # Detect grid: t1 = column 1, t2 = column 2
    t1_vals = np.unique(data[:, 1])
    t2_vals = np.unique(data[:, 2])
    n1 = len(t1_vals)
    n2 = len(t2_vals)

    # Reshape energy to 2D grid
    e_total = np.full((n1, n2), np.nan)
    e_elec = np.full((n1, n2), np.nan)
    for row in data:
        i = np.argmin(np.abs(t1_vals - row[1]))
        j = np.argmin(np.abs(t2_vals - row[2]))
        e_total[i, j] = row[4]
        e_elec[i, j] = row[2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, E, title in [(axes[0], e_total, 'Total energy'),
                          (axes[1], e_elec, 'Electronic energy')]:
        im = ax.pcolormesh(t1_vals, t2_vals, E.T, shading='auto', cmap='RdYlBu_r')
        ax.set_xlabel('t1 (H4: O3→O7)')
        ax.set_ylabel('t2 (H9: O8→O2)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Hartree')
        # Mark minima and saddle
        imin = np.unravel_index(np.nanargmin(E), E.shape)
        ax.plot(t1_vals[imin[0]], t2_vals[imin[1]], 'k*', markersize=15,
                label=f'min ({t1_vals[imin[0]]:.2f},{t2_vals[imin[1]]:.2f})')
        ax.legend()

    plt.tight_layout()
    out = f'{out_prefix}_pes.png'
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')
    plt.close()

    # Energy relative to minimum
    e_min = np.nanmin(e_total)
    e_rel = (e_total - e_min) * 627.509  # Hartree to kcal/mol
    fig, ax = plt.subplots(figsize=(8, 7))
    levels = np.linspace(0, np.nanmax(e_rel), 30)
    cs = ax.contourf(t1_vals, t2_vals, e_rel.T, levels=levels, cmap='hot_r')
    ax.contour(t1_vals, t2_vals, e_rel.T, levels=10, colors='black', linewidths=0.5)
    ax.set_xlabel('t1 (H4: O3→O7)')
    ax.set_ylabel('t2 (H9: O8→O2)')
    ax.set_title('2D PES (kcal/mol above minimum)')
    plt.colorbar(cs, ax=ax, label='kcal/mol')
    plt.tight_layout()
    out = f'{out_prefix}_pes_kcal.png'
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='CSV file from hbond_ref')
    parser.add_argument('--charges', help='data dir with per-point charges', default=None)
    parser.add_argument('--out', help='output prefix', default='hbond')
    args = parser.parse_args()

    header, data = parse_csv(args.csv)
    print(f'Header: {header}')
    print(f'Points: {len(data)}')

    if len(header) > 6:  # 2D: idx,t1,t2,e_elec,e_rep,e_total,n_iter
        plot_2d(header, data, args.out, args.charges)
    else:  # 1D: idx,t,e_elec,e_rep,e_total,n_iter
        plot_1d(header, data, args.out, args.charges)

if __name__ == '__main__':
    main()
