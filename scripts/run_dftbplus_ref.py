#!/usr/bin/env python3
"""DFTB+ Fortran comparison harness.

Runs the upstream Fortran DFTB+ binary on a given XYZ geometry with the mio-1-1
SK set, then parses detailed.out (Mulliken charges) and band.out (eigenvalues +
occupations) into TSV files for plotting alongside the Rust results.

Conventions:
    - detailed.out: charges in electrons (deltaQ = q0 - q_elec, positive = deficit),
      Fermi level + total energy in Hartree.
    - band.out: eigenvalues in eV — converted to Hartree (1 Ha = 27.211386 eV).
    - Rust SccResult.charges stores Mulliken POPULATIONS (q_elec); convert to
      charge via q = q0 - population for comparison.

Usage:
    python3 run_dftbplus_ref.py <xyz_file> <output_dir> [--sk-dir <path>] [--scc-tol 1e-8] [--max-iter 100]

Outputs (written to <output_dir>):
    ref_charges.tsv   — atom_idx, element, x, y, z, charge
    ref_eigenvalues.tsv — idx, eigenvalue_Ha, occupation
    ref_dftbplus.log  — full DFTB+ stdout
"""
import argparse
import os
import subprocess
import sys
import tempfile

DFTBPLUS_BIN = "/home/prokophapala/git/dftbplus/_build/app/dftb+/dftb+"
DEFAULT_SK_DIR = "/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1"

# MaxAngularMomentum for mio-1-1 set
MAX_ANG_MOM = {
    "H": "s", "C": "p", "N": "p", "O": "p", "S": "d",
    "P": "d", "F": "p", "Cl": "p", "Br": "d", "I": "d",
    "Si": "d", "Na": "p", "Mg": "p", "Al": "p", "K": "p",
    "Ca": "d", "Zn": "d",
}


def parse_xyz(path):
    """Parse an XYZ file → (elements, coords). coords in Angstrom."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    elements = []
    coords = []
    for i in range(n):
        parts = lines[2 + i].split()
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elements, coords


def gen_dftb_in(xyz_path, sk_dir, scc_tol, max_iter):
    """Generate a dftb_in.hsd string for SCC DFTB with the given XYZ."""
    # Build MaxAngularMomentum block from species in the XYZ
    species = set(parse_xyz(xyz_path)[0])
    max_ang_lines = []
    for sp in sorted(species):
        mom = MAX_ANG_MOM.get(sp)
        if mom is None:
            print(f"WARNING: unknown max angular momentum for {sp}, skipping", file=sys.stderr)
            continue
        max_ang_lines.append(f'    {sp} = "{mom}"')
    max_ang_block = "\n".join(max_ang_lines)

    return f"""Geometry = xyzFormat {{
  <<< "{xyz_path}"
}}

Driver = {{}}

Hamiltonian = DFTB {{
  SCC = Yes
  SCCTolerance = {scc_tol:.1e}
  MaxSCCIterations = {max_iter}
  MaxAngularMomentum {{
{max_ang_block}
  }}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_dir}/"
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
  WriteEigenvectors = Yes
}}
"""


def parse_detailed_out(path):
    """Parse detailed.out → (charges, fermi_level, total_energy).

    Returns:
        charges: list of (atom_idx, charge) — 1-based atom indices.
        fermi_level: float in Hartree.
        total_energy: float in Hartree.
    """
    charges = []
    fermi = None
    total_energy = None
    in_charges = False
    with open(path) as f:
        for line in f:
            line_s = line.strip()
            if line_s.startswith("Atomic gross charges"):
                in_charges = True
                continue
            if in_charges:
                if line_s.startswith("Atom") and "Charge" in line_s:
                    continue  # header
                parts = line_s.split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0])
                        q = float(parts[1])
                        charges.append((idx, q))
                    except ValueError:
                        in_charges = False
                        continue
                elif len(parts) == 0:
                    in_charges = False
            if line_s.startswith("Fermi level:"):
                # "Fermi level:     0.0745171374 H     2.0277 eV"
                parts = line_s.split()
                for i, p in enumerate(parts):
                    if p == "H" and i > 0:
                        fermi = float(parts[i - 1])
            if line_s.startswith("Total energy:"):
                parts = line_s.split()
                for i, p in enumerate(parts):
                    if p == "H" and i > 0:
                        total_energy = float(parts[i - 1])
    return charges, fermi, total_energy


def parse_band_out(path):
    """Parse band.out → list of (idx, eigenvalue_Ha, occupation).

    Note: band.out eigenvalues are in eV; we convert to Hartree (1 Ha = 27.2114 eV).
    """
    EV_TO_HA = 1.0 / 27.211386
    eigs = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                try:
                    idx = int(parts[0])
                    eig_eV = float(parts[1])
                    occ = float(parts[2])
                    eigs.append((idx, eig_eV * EV_TO_HA, occ))
                except ValueError:
                    continue
    return eigs


def parse_eigenvec_bin(path, nstates, norb):
    """Parse DFTB+ eigenvec.bin → (nstates, norb) float64 array.

    Format: [4-byte identity int] [nstates * norb * 8-byte float64]
    Fortran stores eigvecsReal(nOrb, nStates) column-major; the .bin file is
    a raw Fortran memory dump. We reshape with order='F' and transpose to get
    (nStates, nOrb) in C order — same as DFTBcore.get_eigvecs_dense().
    """
    import struct
    import numpy as np
    with open(path, 'rb') as f:
        raw = f.read()
    identity = struct.unpack_from('i', raw, 0)[0]
    evecs = np.frombuffer(raw[4:], dtype=np.float64).reshape(norb, nstates, order='F').T.copy()
    return evecs


def main():
    ap = argparse.ArgumentParser(description="Run DFTB+ Fortran reference for comparison")
    ap.add_argument("xyz_file", help="Input XYZ geometry")
    ap.add_argument("output_dir", help="Output directory for TSV files")
    ap.add_argument("--sk-dir", default=DEFAULT_SK_DIR, help="SK file directory")
    ap.add_argument("--scc-tol", type=float, default=1e-8, help="SCC tolerance")
    ap.add_argument("--max-iter", type=int, default=100, help="Max SCC iterations")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    elements, coords = parse_xyz(args.xyz_file)
    print(f"=== DFTB+ reference: {args.xyz_file} ({len(elements)} atoms) ===")

    # Write dftb_in.hsd into a temp working directory
    work = tempfile.mkdtemp(prefix="dftbplus_ref_")
    hsd = gen_dftb_in(os.path.abspath(args.xyz_file), args.sk_dir, args.scc_tol, args.max_iter)
    hsd_path = os.path.join(work, "dftb_in.hsd")
    with open(hsd_path, "w") as f:
        f.write(hsd)
    print(f"  dftb_in.hsd: {hsd_path}")

    # Run DFTB+
    print(f"  running DFTB+ ...")
    result = subprocess.run(
        [DFTBPLUS_BIN],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log_path = os.path.join(args.output_dir, "ref_dftbplus.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        print(f"  ERROR: DFTB+ exited with code {result.returncode}")
        print(f"  See log: {log_path}")
        # Print last few lines of stdout for diagnosis
        tail = result.stdout.strip().split("\n")[-10:]
        for line in tail:
            print(f"    {line}")
        sys.exit(1)

    # Parse outputs
    detailed_path = os.path.join(work, "detailed.out")
    band_path = os.path.join(work, "band.out")

    if not os.path.exists(detailed_path):
        print(f"  ERROR: detailed.out not found at {detailed_path}")
        sys.exit(1)

    charges, fermi, total_energy = parse_detailed_out(detailed_path)
    print(f"  Fermi level: {fermi:.10f} Ha" if fermi else "  Fermi level: N/A")
    print(f"  Total energy: {total_energy:.10f} Ha" if total_energy else "  Total energy: N/A")
    print(f"  Charges: {len(charges)} atoms")

    # Write charges TSV (with geometry from XYZ)
    charges_tsv = os.path.join(args.output_dir, "ref_charges.tsv")
    with open(charges_tsv, "w") as f:
        f.write("atom_idx\telement\tx\ty\tz\tcharge\n")
        for i, ((el, pos), (_, q)) in enumerate(zip(zip(elements, coords), charges)):
            f.write(f"{i}\t{el}\t{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{q:.6f}\n")
    print(f"  saved: {charges_tsv}")

    # Parse and write eigenvalues
    eigs = []
    if os.path.exists(band_path):
        eigs = parse_band_out(band_path)
        eigs_tsv = os.path.join(args.output_dir, "ref_eigenvalues.tsv")
        with open(eigs_tsv, "w") as f:
            f.write("idx\teigenvalue\toccupation\n")
            for idx, eig, occ in eigs:
                f.write(f"{idx}\t{eig:.10f}\t{occ:.5f}\n")
        print(f"  saved: {eigs_tsv} ({len(eigs)} eigenvalues)")

        # HOMO-LUMO from band.out
        occ_eigs = [e for _, e, o in eigs if o > 0.5]
        virt_eigs = [e for _, e, o in eigs if o < 0.5]
        if occ_eigs and virt_eigs:
            homo = max(occ_eigs)
            lumo = min(virt_eigs)
            print(f"  HOMO={homo:.10f}, LUMO={lumo:.10f}, gap={lumo-homo:.10f} Ha")
    else:
        print("  WARNING: band.out not found")

    # Parse and write eigenvectors (eigenvec.bin)
    eigvec_path = os.path.join(work, "eigenvec.bin")
    if os.path.exists(eigvec_path) and eigs:
        import numpy as np
        nstates = len(eigs)
        norb = nstates  # square matrix
        evecs = parse_eigenvec_bin(eigvec_path, nstates, norb)
        n_occ = sum(1 for _, _, o in eigs if o > 0.5)
        eigvec_tsv = os.path.join(args.output_dir, "ref_eigenvectors.tsv")
        with open(eigvec_tsv, "w") as f:
            f.write(f"# natoms={len(elements)} norb={norb} n_occ={n_occ}\n")
            f.write("# atom_idx\telement\tx\ty\tz\n")
            for i, (el, pos) in enumerate(zip(elements, coords)):
                f.write(f"{i}\t{el}\t{pos[0]:.10f}\t{pos[1]:.10f}\t{pos[2]:.10f}\n")
            f.write("# eigenvector matrix (norb x norb), columns are MOs\n")
            f.write("mo_idx\torb_idx\tcoeff\n")
            for mo in range(nstates):
                for orb in range(norb):
                    f.write(f"{mo}\t{orb}\t{evecs[mo, orb]:.12e}\n")
            f.write("# eigenvalues\n")
            f.write("idx\teigenvalue\toccupied\n")
            for idx, eig, occ in eigs:
                occ_flag = 1 if occ > 0.5 else 0
                f.write(f"{idx}\t{eig:.10f}\t{occ_flag}\n")
        print(f"  saved: {eigvec_tsv} ({len(elements)} atoms, {norb} orbitals, n_occ={n_occ})")
    else:
        print("  WARNING: eigenvec.bin not found (eigenvectors not saved)")

    print(f"  log: {log_path}")
    print(f"=== DFTB+ reference done ===")


if __name__ == "__main__":
    main()
