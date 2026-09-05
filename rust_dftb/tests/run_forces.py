#!/usr/bin/env python3
"""DFTB+ force parity driver (Agent_3, Wave 1).

Generates a DFTB+ reference for the total forces on a molecule, then runs
the Rust parity test and reports the maximum absolute discrepancy.

Usage:
    python3 tests/run_forces.py data/xyz/H2O.xyz --no-scc
    python3 tests/run_forces.py data/xyz/H2O.xyz --scc
    python3 tests/run_forces.py data/xyz/HCN.xyz --scc --tol 1e-5

The driver:
  1. Parses the XYZ file.
  2. Writes a DFTB+ GenFormat geometry + HSD input with PrintForces = Yes.
  3. Runs DFTB+ (SCC on or off).
  4. Parses the "Total Forces" block from detailed.out (Hartree/Bohr).
  5. Writes the reference forces to a temp file.
  6. Runs `cargo test --test parity_forces` with env vars pointing at the
     XYZ, the reference file, and the SK directory.
  7. Reports the max absolute discrepancy.

Environment:
  RUST_DFTB_SK_DIR – optional override for the SK directory.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

# Make sure we can import test_utils from the same directory.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_utils import (
    parse_xyz, write_gen, run_dftb,
    SK_DIR, RUST_DIR, DFTB_BIN,
)


def write_hsd_forces(path, gen_file, species, *, scc=False,
                     max_scc_iterations=None):
    """Write a DFTB+ HSD input that enables force printing."""
    uq = []
    for s in species:
        if s not in uq:
            uq.append(s)
    ang_lines = []
    for s in uq:
        ang = "p" if s in ("C", "N", "O", "F", "S", "P") else "s"
        ang_lines.append(f'    {s} = "{ang}"')
    ang_block = "\n".join(ang_lines)

    if max_scc_iterations is None:
        max_scc_iterations = 200 if scc else 1

    scc_block = ""
    if scc:
        scc_block = (f"  SCCTolerance = 1.0E-010\n"
                     f"  MaxSCCIterations = {max_scc_iterations}\n")

    hsd = f"""Geometry = GenFormat {{
  <<< "{gen_file}"
}}

Hamiltonian = DFTB {{
  SCC = {"Yes" if scc else "No"}
  {scc_block}  MaxAngularMomentum = {{
{ang_block}
  }}
  SlaterKosterFiles = Type2Filenames {{
    Prefix = "{SK_DIR}/"
    Separator = "-"
    Suffix = ".skf"
  }}
  Filling = Fermi {{
    Temperature [Kelvin] = 0.0
  }}
}}

Options {{
  WriteDetailedOut = Yes
}}

Analysis {{
  PrintForces = Yes
}}

ParserOptions {{
  ParserVersion = 13
}}
"""
    with open(path, "w") as f:
        f.write(hsd)


def parse_total_forces(work_dir):
    """Parse the 'Total Forces' block from detailed.out.

    Returns a list of [fx, fy, fz] in Hartree/Bohr.
    """
    path = os.path.join(work_dir, "detailed.out")
    with open(path) as f:
        text = f.read()

    # DFTB+ detailed.out format:
    #   Total Forces
    #       1      0.000000000000      0.018731270219     -0.000000000000
    #       2      0.000777420807     -0.009365635110      0.000000000000
    #       ...
    #   (blank line or next section)
    forces = []
    in_block = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("total forces"):
            in_block = True
            continue
        if in_block:
            if stripped == "" or stripped.startswith("-"):
                if forces:
                    break
                else:
                    continue
            parts = stripped.split()
            # Format: atom_index fx fy fz  (4 columns)
            if len(parts) == 4:
                try:
                    _idx = int(parts[0])
                    fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                    forces.append([fx, fy, fz])
                except ValueError:
                    if forces:
                        break
            elif len(parts) == 3:
                # Fallback: some DFTB+ versions omit the atom index.
                try:
                    fx, fy, fz = float(parts[0]), float(parts[1]), float(parts[2])
                    forces.append([fx, fy, fz])
                except ValueError:
                    if forces:
                        break
            else:
                if forces:
                    break
    return forces


def main():
    parser = argparse.ArgumentParser(description="DFTB+ force parity driver")
    parser.add_argument("xyz", help="Path to XYZ molecule file")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--no-scc", action="store_true", help="Non-SCC calculation")
    g.add_argument("--scc", action="store_true", help="SCC calculation")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Tolerance for force comparison (default 1e-5)")
    parser.add_argument("--release", action="store_true",
                        help="Use cargo --release")
    args = parser.parse_args()

    species, coords = parse_xyz(args.xyz)
    name = os.path.splitext(os.path.basename(args.xyz))[0]
    scc = args.scc

    print(f"[{name}] Loaded {len(species)} atoms from {args.xyz}")
    print(f"[{name}] Mode: {'SCC' if scc else 'non-SCC'}")
    print(f"[{name}] Tolerance: {args.tol}")

    sk_dir = os.environ.get("RUST_DFTB_SK_DIR", SK_DIR)

    with tempfile.TemporaryDirectory(prefix=f"forces_parity_{name}_") as work:
        gen = os.path.join(work, "geometry.gen")
        write_gen(gen, species, coords)
        write_hsd_forces(os.path.join(work, "dftb_in.hsd"), gen, species,
                         scc=scc)

        print(f"[{name}] Running DFTB+ ({'SCC' if scc else 'non-SCC'})...")
        result = run_dftb(work)
        if result.returncode != 0:
            print(f"DFTB+ failed (rc={result.returncode}):")
            print("--- stdout ---")
            print(result.stdout)
            print("--- stderr ---")
            print(result.stderr)
            sys.exit(1)

        ref_forces = parse_total_forces(work)
        if not ref_forces:
            print("ERROR: could not parse 'Total Forces' from detailed.out")
            print("--- detailed.out (first 80 lines) ---")
            with open(os.path.join(work, "detailed.out")) as f:
                for i, line in enumerate(f):
                    if i >= 80:
                        break
                    print(line.rstrip())
            sys.exit(1)

        if len(ref_forces) != len(species):
            print(f"ERROR: ref force count {len(ref_forces)} != atom count {len(species)}")
            sys.exit(1)

        print(f"[{name}] Reference forces (Hartree/Bohr):")
        for i, f in enumerate(ref_forces):
            print(f"  atom {i} {species[i]:>2}: [{f[0]:+.10e} {f[1]:+.10e} {f[2]:+.10e}]")

        # Write reference forces file for the Rust test.
        ref_path = os.path.join(work, "ref_forces.txt")
        with open(ref_path, "w") as f:
            for force in ref_forces:
                f.write(f"{force[0]:.15e} {force[1]:.15e} {force[2]:.15e}\n")

        # Run the Rust parity test.
        scc_flag = "1" if scc else "0"
        env = {
            **os.environ,
            "RUST_DFTB_SK_DIR": sk_dir,
            "RUST_DFTB_FORCES_XYZ": os.path.abspath(args.xyz),
            "RUST_DFTB_FORCES_REF": ref_path,
            "RUST_DFTB_FORCES_SCC": scc_flag,
            "RUST_DFTB_FORCES_TOL": str(args.tol),
        }

        cmd = ["cargo", "test"]
        if args.release:
            cmd.append("--release")
        cmd += ["--test", "parity_forces", "--", "--nocapture"]
        print(f"\n[{name}] Running Rust parity test: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=RUST_DIR, env=env)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
