"""Shared utilities for DFTB+ parity test drivers."""

import os
import subprocess

DFTB_BIN = "/home/prokophapala/git/dftbplus/_build/app/dftb+/dftb+"
SK_DIR = "/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1"
RUST_DIR = os.path.dirname(os.path.abspath(__file__))

HUBBARD_U = {"H": 0.4195, "C": 0.3647, "N": 0.4309, "O": 0.4954, "F": 0.4500, "S": 0.3200, "P": 0.3500}


def parse_xyz(path):
    """Parse standard XYZ; optional 5th column is charge (ignored for geometry)."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    species, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        species.append(parts[0].capitalize())
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return species, coords


def unique_species(species):
    """Return unique species list preserving first-seen order."""
    uq = []
    for s in species:
        if s not in uq:
            uq.append(s)
    return uq


def max_ang_string(species):
    """Build the MaxAngularMomentum HSD block for a list of species."""
    uq = unique_species(species)
    lines = []
    for s in uq:
        ang = "p" if s in ("C", "N", "O", "F", "S", "P") else "s"
        lines.append(f'    {s} = "{ang}"')
    return "\n".join(lines)


def write_gen(path, species, coords):
    """Write DFTB+ GenFormat file."""
    uq = unique_species(species)
    with open(path, "w") as f:
        f.write(f"{len(species)}  C\n")
        f.write(" ".join(uq) + "\n")
        for i, (s, c) in enumerate(zip(species, coords)):
            sp_idx = uq.index(s) + 1
            f.write(f"{i + 1}  {sp_idx}  {c[0]:.10E}  {c[1]:.10E}  {c[2]:.10E}\n")


def write_hsd(path, gen_file, species, *, scc=False, initial_charges=None,
              max_scc_iterations=None, write_hs=False, write_band_out=False):
    """Write DFTB+ input HSD.

    Args:
        scc: Enable SCC.
        initial_charges: Optional list of per-atom charges for InitialCharges block.
        max_scc_iterations: MaxSCCIterations value (default: 1 for fixed-charge, 200 for full SCC).
        write_hs: Write hamsqr1.dat / oversqr.dat.
        write_band_out: Write band.out and eigenvectors.
    """
    ang_block = max_ang_string(species)

    if max_scc_iterations is None:
        max_scc_iterations = 200 if scc else 1

    ic_block = ""
    if initial_charges:
        ic_block = "  InitialCharges = {\n"
        for i, dq in enumerate(initial_charges, 1):
            ic_block += f"    AtomCharge {{\n      Atoms = {{ {i} }}\n      ChargePerAtom = {dq}\n    }}\n"
        ic_block += "  }\n"

    scc_block = ""
    if scc:
        scc_block = f"  SCCTolerance = 1.0E-010\n  MaxSCCIterations = {max_scc_iterations}\n"

    options_lines = ['  WriteDetailedOut = Yes']
    if write_hs:
        options_lines.append('  WriteHS = Yes')
    if write_band_out:
        options_lines.append('  WriteCharges = Yes')
    options_block = "\n".join(options_lines)

    analysis_block = ""
    if write_band_out:
        analysis_block = """
Analysis {
  WriteBandOut = Yes
  WriteEigenvectors = Yes
}
"""

    hsd = f"""Geometry = GenFormat {{
  <<< "{gen_file}"
}}

Hamiltonian = DFTB {{
  SCC = {"Yes" if scc else "No"}
  {scc_block}{ic_block}  MaxAngularMomentum = {{
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
{options_block}
}}{analysis_block}

ParserOptions {{
  ParserVersion = 13
}}
"""
    with open(path, "w") as f:
        f.write(hsd)


def run_dftb(work_dir):
    """Run DFTB+ in work_dir."""
    return subprocess.run(
        [DFTB_BIN, os.path.join(work_dir, "dftb_in.hsd")],
        cwd=work_dir, capture_output=True, text=True,
    )


def parse_square_matrix(path):
    """Parse DFTB+ hamsqr/oversqr file."""
    with open(path) as f:
        lines = f.readlines()
    data = []
    found = False
    for line in lines:
        if line.strip().startswith("# MATRIX"):
            found = True
            continue
        if found:
            for tok in line.split():
                data.append(float(tok))
    n = int(len(data) ** 0.5)
    return n, data
