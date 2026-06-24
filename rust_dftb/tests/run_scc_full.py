#!/usr/bin/env python3
"""Run full SCC DFTB+ reference and compare with Rust implementation.

Runs DFTB+ with SCC=Yes (full convergence) on an XYZ molecule, extracts:
  - Total energy from detailed.out
  - Converged Mulliken charges from detailed.out
  - H_scc from hamsqr1.dat
  - S from oversqr.dat
  - Eigenvalues from band.out

Then runs the Rust SCC test with these references via environment variables.

Usage:
    python3 run_scc_full.py /path/to/molecule.xyz
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

from test_utils import (
    parse_xyz, write_gen, write_hsd, run_dftb,
    SK_DIR, RUST_DIR,
)


def parse_detailed_out(work_dir):
    """Extract total energy and charges from detailed.out."""
    path = os.path.join(work_dir, "detailed.out")
    with open(path) as f:
        text = f.read()

    # Total electronic energy (band energy + SCC correction, no repulsive pair)
    # This matches our Rust energy = e_band + e_rep
    m = re.search(r'Total Electronic energy:\s+([-\d.E+]+)\s+H', text)
    total_energy = float(m.group(1)) if m else None

    # Fallback: "Total energy" (includes repulsive pair potential)
    if total_energy is None:
        m = re.search(r'Total energy:\s+([-\d.E+]+)\s+H', text)
        total_energy = float(m.group(1)) if m else None

    # Mermin free energy (at T=0 this equals total energy)
    if total_energy is None:
        m = re.search(r'Mermin free energy.*?([-\d.E+]+)\s+H', text, re.DOTALL)
        total_energy = float(m.group(1)) if m else None

    # Charges: parse atom populations (electron count per atom) to match Rust convention
    charges = []
    in_pop = False
    for line in text.split('\n'):
        if 'Atom populations' in line:
            in_pop = True
            continue
        if in_pop:
            if line.strip().startswith('-') or line.strip() == '' or 'Nr.' in line:
                in_pop = False
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    charges.append(float(parts[-1]))
                except ValueError:
                    if len(charges) > 0:
                        in_pop = False

    return total_energy, charges


def parse_band_out(work_dir):
    """Extract eigenvalues from band.out (in eV, convert to Hartree)."""
    path = os.path.join(work_dir, "band.out")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    eigs = []
    for line in text.split('\n'):
        parts = line.split()
        # Format: "index  eigenvalue_eV  occupation"
        if len(parts) == 3:
            try:
                idx = int(parts[0])
                eig_ev = float(parts[1])
                eigs.append(eig_ev / 27.211386)  # eV -> Hartree
            except ValueError:
                pass
    return eigs if eigs else None


def main():
    parser = argparse.ArgumentParser(description="Run full SCC parity test")
    parser.add_argument("xyz", help="Path to XYZ file")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for comparisons")
    parser.add_argument("--release", action="store_true", help="Use cargo --release")
    args = parser.parse_args()

    species, coords = parse_xyz(args.xyz)
    name = os.path.splitext(os.path.basename(args.xyz))[0]

    with tempfile.TemporaryDirectory(prefix=f"scc_parity_{name}_") as work:
        gen = os.path.join(work, "geometry.gen")
        write_gen(gen, species, coords)
        write_hsd(os.path.join(work, "dftb_in.hsd"), gen, species,
                  scc=True, max_scc_iterations=200, write_band_out=True)

        print(f"[{name}] Running DFTB+ SCC (full convergence)...")
        result = run_dftb(work)
        if result.returncode != 0:
            print(f"DFTB+ failed:\n{result.stderr}")
            sys.exit(1)

        # Parse results
        total_energy, charges = parse_detailed_out(work)
        eigs = parse_band_out(work)

        print(f"  Total energy: {total_energy}")
        print(f"  Charges:      {charges}")
        print(f"  Eigenvalues:  {eigs}")

        # Set up environment for Rust test
        env = {
            **os.environ,
            "RUST_DFTB_SK_DIR": SK_DIR,
            "RUST_DFTB_SCC_XYZ": os.path.abspath(args.xyz),
            "RUST_DFTB_SCC_REF_CHARGES": ",".join(str(c) for c in charges) if charges else "",
            "RUST_DFTB_SCC_REF_EIGS": ",".join(str(e) for e in eigs) if eigs else "",
            "RUST_DFTB_SCC_REF_ENERGY": str(total_energy) if total_energy else "",
            "RUST_DFTB_SCC_TOL": str(args.tol),
        }

        print(f"\n[{name}] Running Rust SCC test with references...")
        cmd = ["cargo", "test"]
        if args.release:
            cmd.append("--release")
        cmd += ["--test", "parity_scc", "--", "--nocapture"]
        result = subprocess.run(cmd, cwd=RUST_DIR, env=env)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
