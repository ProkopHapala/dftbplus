#!/usr/bin/env python3
"""Compare Rust DFTB (dense) vs DFTB+ Fortran reference for charges + HOMO-LUMO.

Reads TSV files from a debug directory and prints a numerical parity report.
"""
import argparse
import os
import sys
import numpy as np

Q0_MAP = {'C': 4.0, 'H': 1.0, 'N': 5.0, 'O': 6.0, 'S': 6.0}


def load_charges(path, convert_to_charge=False):
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, dtype=None, encoding='utf-8', names=True, delimiter='\t')
    elements = [row['element'] for row in data]
    pos = np.array([[row['x'], row['y'], row['z']] for row in data])
    charges = np.array([row['charge'] for row in data])
    if convert_to_charge:
        q0 = np.array([Q0_MAP.get(el, 4.0) for el in elements])
        charges = q0 - charges
    return elements, pos, charges


def load_eigs(path):
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, names=True, delimiter='\t')
    eigs = data['eigenvalue']
    if 'occupation' in data.dtype.names:
        occs = data['occupation']
    elif 'occupied' in data.dtype.names:
        occs = data['occupied'] * 2.0
    else:
        occs = np.zeros(len(eigs))
    return eigs, occs


def homo_lumo(eigs, occs):
    n_occ = int(occs.sum() / 2)
    if n_occ == 0 or n_occ >= len(eigs):
        return None, None, None
    return eigs[n_occ - 1], eigs[n_occ], eigs[n_occ] - eigs[n_occ - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("debug_dir")
    ap.add_argument("--systems", default="benzene,coronene,circumcoronene")
    args = ap.parse_args()
    systems = [s.strip() for s in args.systems.split(",")]

    print("=" * 90)
    print("Rust DFTB (dense) vs DFTB+ Fortran reference — parity report")
    print("=" * 90)

    for s in systems:
        print(f"\n{'─' * 70}")
        print(f"  {s}")
        print(f"{'─' * 70}")

        # Charges
        rust = load_charges(os.path.join(args.debug_dir, f"{s}_charges.tsv"), convert_to_charge=True)
        ref = load_charges(os.path.join(args.debug_dir, f"{s}_ref_charges.tsv"), convert_to_charge=False)
        sparse = load_charges(os.path.join(args.debug_dir, f"{s}_sparse_charges.tsv"), convert_to_charge=True)
        if rust is not None and ref is not None:
            n = len(rust[2])
            dq = rust[2] - ref[2]
            print(f"  Charges (dense vs DFTB+): {n} atoms")
            print(f"    max|Δq|  = {np.abs(dq).max():.6e} e")
            print(f"    RMS|Δq|  = {np.sqrt((dq**2).mean()):.6e} e")
            print(f"    sum|Δq|  = {np.abs(dq).sum():.6e} e (should be ~0)")
            worst = np.argsort(np.abs(dq))[-3:][::-1]
            for i in worst:
                print(f"    atom {i:3d} ({rust[0][i]}): Rust={rust[2][i]:+.6f}  DFTB+={ref[2][i]:+.6f}  Δ={dq[i]:+.6e}")
        if sparse is not None and ref is not None:
            dq_sp = sparse[2] - ref[2]
            print(f"  Charges (sparse TC2 vs DFTB+):")
            print(f"    max|Δq|  = {np.abs(dq_sp).max():.6e} e")
            print(f"    RMS|Δq|  = {np.sqrt((dq_sp**2).mean()):.6e} e")
        if sparse is not None and rust is not None:
            dq_ds = sparse[2] - rust[2]
            print(f"  Charges (sparse vs dense):")
            print(f"    max|Δq|  = {np.abs(dq_ds).max():.6e} e")
        else:
            print("  Charges: missing data")

        # Eigenvalues
        rust_e = load_eigs(os.path.join(args.debug_dir, f"{s}_eigenvalues.tsv"))
        ref_e = load_eigs(os.path.join(args.debug_dir, f"{s}_ref_eigenvalues.tsv"))
        if rust_e is not None and ref_e is not None:
            rust_h, rust_l, rust_g = homo_lumo(*rust_e)
            ref_h, ref_l, ref_g = homo_lumo(*ref_e)
            print(f"  HOMO-LUMO:")
            if rust_h is not None and ref_h is not None:
                print(f"    HOMO:  Rust={rust_h:+.10f}  DFTB+={ref_h:+.10f}  Δ={rust_h-ref_h:+.6e} Ha")
                print(f"    LUMO:  Rust={rust_l:+.10f}  DFTB+={ref_l:+.10f}  Δ={rust_l-ref_l:+.6e} Ha")
                print(f"    gap:   Rust={rust_g:.10f}  DFTB+={ref_g:.10f}  Δ={rust_g-ref_g:+.6e} Ha")
            # All eigenvalues
            de = rust_e[0] - ref_e[0]
            print(f"  All eigenvalues ({len(de)}):")
            print(f"    max|Δε|  = {np.abs(de).max():.6e} Ha")
            print(f"    RMS|Δε|  = {np.sqrt((de**2).mean()):.6e} Ha")
        else:
            print("  Eigenvalues: missing data")

    print(f"\n{'=' * 90}")
    print("Done.")


if __name__ == "__main__":
    main()
