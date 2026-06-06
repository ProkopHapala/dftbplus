#!/usr/bin/env python3
"""Generate DFTB+ reference data for H2 using direct dftb+ binary."""

import os
import subprocess
import numpy as np

ANG2BOHR = 1.0 / 0.52917720859
sk_dir = '/home/prokophapala/git_SW/dftbplus/external/slakos/origin/mio-1-1'
dftb_bin = '/home/prokophapala/git/dftbplus/_build/app/dftb+/dftb+'

out_dir = os.path.dirname(os.path.abspath(__file__))

# HSD with MaxSCCIterations=1: H_scc built from initial charges, diagonalize once, no mixing.
hsd_content = f"""Geometry = GenFormat {{
  2  C
  H
  1  1  0.0000000000E+00  0.0000000000E+00  0.0000000000E+00
  2  1  {0.74 * ANG2BOHR:.10E}  0.0000000000E+00  0.0000000000E+00
}}

Hamiltonian = DFTB {{
  SCC = Yes
  SCCTolerance = 1.0E-010
  MaxSCCIterations = 1
  Charge = 0.0
  InitialCharges = {{
    AtomCharge = {{
      Atoms = {{ 1 }}
      ChargePerAtom = -0.1
    }}
    AtomCharge = {{
      Atoms = {{ 2 }}
      ChargePerAtom = 0.1
    }}
  }}
  MaxAngularMomentum = {{
    H = "s"
  }}
  SlaterKosterFiles = Type2Filenames {{
    Prefix = "{sk_dir}/"
    Separator = "-"
    Suffix = ".skf"
  }}
  Filling = Fermi {{
    Temperature [Kelvin] = 0.0
  }}
}}

Options {{
  WriteHS = Yes
  WriteDetailedOut = Yes
}}

ParserOptions {{
  ParserVersion = 13
}}
"""

hsd_path = os.path.join(out_dir, 'dftb_in.hsd')
with open(hsd_path, 'w') as f:
    f.write(hsd_content)

# Run DFTB+ binary directly
# NOTE: WriteHS=Yes causes DFTB+ to call ERROR STOP after writing matrices.
# This is expected behavior, not a real error.
print("Running DFTB+ with fixed charges...")
result = subprocess.run([dftb_bin, hsd_path], capture_output=True, text=True, cwd=out_dir)
print(result.stdout)
has_ref_files = (
    os.path.exists(os.path.join(out_dir, 'hamsqr1.dat')) and
    os.path.exists(os.path.join(out_dir, 'oversqr.dat'))
)
if result.returncode != 0 and not has_ref_files:
    print("STDERR:", result.stderr)
    raise RuntimeError(f"DFTB+ failed with code {result.returncode} and no reference files written")

def parse_square_matrix(path):
    """Parse DFTB+ square matrix file (hamsqr*.dat, oversqr.dat)."""
    with open(path) as f:
        lines = f.readlines()
    # Skip header lines, find '# MATRIX'
    data_lines = []
    found_matrix = False
    for line in lines:
        if line.strip().startswith('# MATRIX'):
            found_matrix = True
            continue
        if found_matrix:
            data_lines.append(line)
    if not found_matrix:
        raise ValueError(f"Could not find '# MATRIX' in {path}")
    values = []
    for line in data_lines:
        for token in line.split():
            values.append(float(token))
    n = int(np.sqrt(len(values)))
    if n * n != len(values):
        raise ValueError(f"Expected square matrix, got {len(values)} values in {path}")
    # Fortran writes arrays column-major, but each line has n values.
    # For a symmetric matrix, transpose doesn't matter.
    arr = np.array(values).reshape((n, n), order='F')
    return arr

hamsqr_path = os.path.join(out_dir, 'hamsqr1.dat')
if not os.path.exists(hamsqr_path):
    raise FileNotFoundError(f"{hamsqr_path} not found. Did WriteHS work?")
H_scc = parse_square_matrix(hamsqr_path)

soversqr_path = os.path.join(out_dir, 'oversqr.dat')
if not os.path.exists(soversqr_path):
    raise FileNotFoundError(f"{soversqr_path} not found.")
S = parse_square_matrix(soversqr_path)

# Save as reference
np.savetxt(os.path.join(out_dir, 'ref_h_scc.dat'), H_scc, fmt='%.16e')
np.savetxt(os.path.join(out_dir, 'ref_s.dat'), S, fmt='%.16e')

print(f"Saved reference data to {out_dir}")
print(f"  H_scc shape: {H_scc.shape}")
print(f"  S shape: {S.shape}")

# Also try to parse scc_debug_1.txt for shifts and charges
debug_path = os.path.join(out_dir, 'scc_debug_1.txt')
if os.path.exists(debug_path):
    with open(debug_path) as f:
        print("\n--- SCC Debug (iteration 1) ---")
        print(f.read())
else:
    print("Warning: scc_debug_1.txt not found")
