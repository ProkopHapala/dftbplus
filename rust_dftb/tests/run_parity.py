#!/usr/bin/env python3
"""Universal parity runner: Fortran DFTB+ vs Rust for arbitrary XYZ molecules.

Usage:
    python3 run_parity.py /path/to/molecule.xyz [--scc] [--delta-q a,b,c,...]

Generates Fortran reference data, then invokes:
    cargo test --test parity_universal parity_universal_from_env
"""

import argparse
import os
import subprocess
import sys
import tempfile

from test_utils import (
    parse_xyz, write_gen, write_hsd, run_dftb, parse_square_matrix,
    unique_species, SK_DIR, RUST_DIR, HUBBARD_U,
)


def parse_delta_q_from_debug(work_dir):
    """Parse deltaQAtom from Fortran scc_debug_1.txt."""
    path = os.path.join(work_dir, "scc_debug_1.txt")
    with open(path) as f:
        lines = f.readlines()
    delta_q = []
    in_section = False
    for line in lines:
        if "deltaQAtom" in line:
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("##"):
                break
            try:
                delta_q.append(float(line.strip()))
            except ValueError:
                continue
    return delta_q


def run_non_scc(work_dir, species, coords):
    """Generate and run non-SCC reference."""
    gen = os.path.join(work_dir, "geometry.gen")
    write_gen(gen, species, coords)
    write_hsd(os.path.join(work_dir, "dftb_in.hsd"), gen, species, scc=False, write_hs=True)
    run_dftb(work_dir)
    h_ref = os.path.join(work_dir, "ref_h0.dat")
    s_ref = os.path.join(work_dir, "ref_s.dat")
    os.rename(os.path.join(work_dir, "hamsqr1.dat"), h_ref)
    os.rename(os.path.join(work_dir, "oversqr.dat"), s_ref)
    return h_ref, s_ref


def run_scc_fixed(work_dir, species, coords, initial_charges):
    """Generate and run SCC (MaxSccIterations=1) reference.
    Returns path to H_scc ref and the ACTUAL deltaQ from Fortran debug output."""
    gen = os.path.join(work_dir, "geometry.gen")
    write_gen(gen, species, coords)
    write_hsd(os.path.join(work_dir, "dftb_in.hsd"), gen, species,
              scc=True, initial_charges=initial_charges, max_scc_iterations=1, write_hs=True)
    run_dftb(work_dir)
    h_ref = os.path.join(work_dir, "ref_h_scc.dat")
    os.rename(os.path.join(work_dir, "hamsqr1.dat"), h_ref)
    delta_q = parse_delta_q_from_debug(work_dir)
    return h_ref, delta_q


def main():
    parser = argparse.ArgumentParser(description="Run parity test for an XYZ molecule")
    parser.add_argument("xyz", help="Path to XYZ file")
    parser.add_argument("--scc", action="store_true", help="Also run fixed-charge SCC parity")
    parser.add_argument("--delta-q", help="Comma-separated deltaQ values (e.g. 0.1,-0.1)")
    parser.add_argument("--tol", type=float, default=1e-7, help="H0/S tolerance")
    parser.add_argument("--tol-scc", type=float, default=1e-6, help="SCC tolerance")
    args = parser.parse_args()

    species, coords = parse_xyz(args.xyz)
    name = os.path.splitext(os.path.basename(args.xyz))[0]

    with tempfile.TemporaryDirectory(prefix=f"parity_{name}_") as work:
        print(f"[{name}] Running non-SCC reference...")
        ref_h, ref_s = run_non_scc(work, species, coords)
        n, _ = parse_square_matrix(ref_h)
        print(f"  H0: {n}x{n}")

        env = {
            "RUST_DFTB_SK_DIR": SK_DIR,
            "RUST_DFTB_SPECIES": ",".join(species),
            "RUST_DFTB_COORDS": ",".join(str(c) for atom in coords for c in atom),
            "RUST_DFTB_REF_H": ref_h,
            "RUST_DFTB_REF_S": ref_s,
            "RUST_DFTB_TOLERANCE": str(args.tol),
        }

        if args.scc:
            if args.delta_q:
                user_delta_q = [float(x) for x in args.delta_q.split(",")]
            else:
                user_delta_q = [0.0] * len(species)
            # Fortran InitialCharges sign convention is complex;
            # we pass user values as initial guess and read ACTUAL deltaQ from debug.
            print(f"[{name}] Running SCC reference (requested deltaQ={user_delta_q})...")
            ref_h_scc, actual_delta_q = run_scc_fixed(work, species, coords, user_delta_q)
            n_scc, _ = parse_square_matrix(ref_h_scc)
            print(f"  H_scc: {n_scc}x{n_scc}")
            print(f"  Actual Fortran deltaQ: {actual_delta_q}")
            env["RUST_DFTB_REF_H_SCC"] = ref_h_scc
            env["RUST_DFTB_DELTA_Q"] = ",".join(str(x) for x in actual_delta_q)
            env["RUST_DFTB_TOLERANCE_SCC"] = str(args.tol_scc)

        uq = unique_species(species)
        hubbard = [str(HUBBARD_U.get(s, 0.4)) for s in uq]
        env["RUST_DFTB_HUBBARD_U"] = ",".join(hubbard)
        env["RUST_DFTB_SPECIES_U"] = ",".join(uq)

        cmd = ["cargo", "test", "--test", "parity_universal", "parity_universal_from_env", "--", "--nocapture"]
        print(f"[{name}] Running Rust parity test...")
        result = subprocess.run(cmd, cwd=RUST_DIR, env={**os.environ, **env})
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
