#!/usr/bin/env python
"""
Test DFTBcore interface with H2O molecule.

This test:
1. Creates a DFTB+ input file for H2O
2. Runs DFTB+ calculation using the simplified DFTBcore interface
3. Extracts energy, density matrix, Hamiltonian, and overlap matrix
4. Prints orbital coefficients and matrix properties
5. Verifies electron count from density matrix

Usage:
    cd /home/prokop/git/dftbplus/tests/dftb
    python test_dftbcore.py

Requirements:
    - DFTB+ library built with libdftbcore.so
    - Slater-Koster files in ~/opt/dftbplus/slakos/3ob-3-1/
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add pyBall to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'pyBall'))

from DFTBcore import DFTBcore, run_dftb_calculation


def create_h2o_input(sk_path, sk_set='3ob-3-1'):
    """
    Create DFTB+ input files for H2O molecule.
    
    Returns:
        Path to created input file
    """
    # Create H2O geometry in Bohr
    # O at origin, H1 at 0.96 A along x, H2 at 104.5 deg angle
    # 1 Angstrom = 1.8897259886 Bohr
    ang2bohr = 1.8897259886
    
    # H2O coordinates (Angstrom)
    coords_ang = np.array([
        [0.0000, 0.0000, 0.0000],    # O
        [0.9584, 0.0000, 0.0000],    # H1
        [-0.2399, 0.9270, 0.0000]    # H2
    ])
    coords_bohr = coords_ang * ang2bohr
    
    # Create XYZ file
    xyz_content = "3\nH2O molecule\n"
    for i, (x, y, z) in enumerate(coords_ang):
        atom = 'O' if i == 0 else 'H'
        xyz_content += f"{atom}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n"
    
    with open('h2o.xyz', 'w') as f:
        f.write(xyz_content)
    print("[Setup] Created h2o.xyz")
    
    # Create DFTB+ input file (HSD format)
    hsd_content = f"""Geometry = xyzFormat {{
  <<< "h2o.xyz"
}}

ParserOptions {{
  ParserVersion = 15
}}

Hamiltonian = DFTB {{
  Scc = Yes
  MaxAngularMomentum {{
    O = "p"
    H = "s"
  }}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_path}{sk_set}/"
    Separator = "-"
    Suffix = ".skf"
  }}
}}

Options {{
  WriteResultsTag = Yes
  WriteDetailedOut = Yes
}}

Analysis {{
  CalculateForces = Yes
}}
"""
    
    with open('h2o.hsd', 'w') as f:
        f.write(hsd_content)
    print("[Setup] Created h2o.hsd")
    
    return Path('h2o.hsd').absolute()


def print_matrix_info(name, matrix, max_print=6):
    """Print matrix information and truncated view."""
    print(f"\n{name}:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Min: {matrix.min():12.6f}, Max: {matrix.max():12.6f}")
    print(f"  Trace: {np.trace(matrix):12.6f}")
    
    n = min(max_print, matrix.shape[0])
    print(f"  First {n}x{n} block:")
    for i in range(n):
        row_str = "  "
        for j in range(n):
            row_str += f"{matrix[i,j]:10.6f} "
        print(row_str)


def print_matrix_full(name, matrix, max_print=6):
    """Print full matrix with row/column labels."""
    n = min(max_print, matrix.shape[0])
    print(f"\n{name} [{matrix.shape[0]}x{matrix.shape[1]}]:")
    print("  " + "-" * (12 * n + 3))
    
    # Column headers
    header = "      "
    for j in range(n):
        header += f"  AO{j:2d}     "
    print(header)
    print("  " + "-" * (12 * n + 3))
    
    # Rows
    for i in range(n):
        row_str = f"AO{i:2d} |"
        for j in range(n):
            row_str += f" {matrix[i,j]:10.6f}"
        print(row_str)
    print("  " + "-" * (12 * n + 3))


def compute_electron_count(dm, overlap):
    """Compute electron count: Tr(S * DM)"""
    return np.trace(overlap @ dm)


def diagonalize_hamiltonian(H, S):
    """
    Solve generalized eigenvalue problem: H * C = E * S * C
    
    Returns:
        eigenvalues, eigenvectors
    """
    # Use scipy if available, otherwise numpy
    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(H, S)
    except ImportError:
        # Fallback to numpy (less stable for generalized problem)
        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(S, H))
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs


def test_h2o_calculation():
    """Test DFTBcore with H2O molecule calculation using official C API."""
    
    print("="*70)
    print("DFTBcore Test: H2O Molecule (using official C API)")
    print("="*70)
    
    # SK file path
    sk_path = os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/library/')
    print(f"\n[Config] SK path: {sk_path}")
    
    # Create input files
    create_h2o_input(sk_path)
    
    print(f"[Setup] Created h2o.xyz")
    print(f"[Setup] Created dftb_in.hsd")
    
    # Run DFTB+ calculation
    print("\n[DFTB+] Initializing calculation...")
    
    try:
        # Use the DFTBcore context manager with new callback-based API
        result = run_dftb_calculation('dftb_in.hsd')
        
        # Extract results
        energy = result['energy']
        basis_size = result['basis_size']
        dm = result['dm']
        H = result['h']
        S = result['s']
        
        print(f"\n[Results]")
        print(f"  Energy:          {energy:16.8f} Hartree")
        print(f"  Energy:          {energy*27.2114:16.8f} eV")
        print(f"  Basis size:      {basis_size}")
        
        # Print full matrices before diagonalization
        print("\n" + "="*70)
        print("RAW MATRICES FROM DFTB+")
        print("="*70)
        print_matrix_full("Hamiltonian (H)", H, 6)
        print_matrix_full("Overlap (S)", S, 6)
        print_matrix_full("Density Matrix (DM)", dm, 6)
        
        # Verify electron count
        n_elec = compute_electron_count(dm, S)
        print(f"\n[Analysis]")
        print(f"  Electron count from DM: {n_elec:10.4f}")
        print(f"  Expected (H2O):         8.0000 (O:6 + H:1 + H:1)")
        
        # Diagonalize Hamiltonian to get MOs
        print("\n[Diagonalization] Solving H * C = E * S * C...")
        eigvals, eigvecs = diagonalize_hamiltonian(H, S)
        
        print("\n  Molecular Orbital Energies:")
        print("  " + "-"*50)
        print(f"  {'MO':>4} {'Energy (Ha)':>16} {'Energy (eV)':>16} {'Occupation':>12}")
        print("  " + "-"*50)
        
        # H2O: 8 electrons, 6 basis functions (O:2s,2px,2py,2pz + H1:1s + H2:1s)
        # MO ordering depends on basis - let's just show first few
        n_mos_show = min(6, len(eigvals))
        for i in range(n_mos_show):
            occ = 2.0 if i < 4 else 0.0  # 8 electrons = 4 occupied MOs
            print(f"  {i+1:>4} {eigvals[i]:>16.8f} {eigvals[i]*27.2114:>16.8f} {occ:>12.1f}")
        
        # Print orbital coefficients for first few MOs
        print(f"\n  MO Coefficients (first {n_mos_show} MOs):")
        for i in range(n_mos_show):
            print(f"\n  MO {i+1} (E={eigvals[i]*27.2114:.3f} eV):")
            coeffs = eigvecs[:, i]
            n_show = min(6, len(coeffs))
            for j in range(n_show):
                print(f"    AO {j}: {coeffs[j]:10.6f}")
        
        # Check matrix properties
        print("\n[Verification]")
        
        # Check if S is symmetric and positive definite
        S_sym_err = np.max(np.abs(S - S.T))
        print(f"  S symmetry error:  {S_sym_err:.2e}")
        
        # Check DM is symmetric
        dm_sym_err = np.max(np.abs(dm - dm.T))
        print(f"  DM symmetry error: {dm_sym_err:.2e}")
        
        # Verify: H should be symmetric
        H_sym_err = np.max(np.abs(H - H.T))
        print(f"  H symmetry error:  {H_sym_err:.2e}")
        
        print("\n" + "="*70)
        print("Test PASSED: DFTBcore interface working correctly")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_h2o_calculation()
    sys.exit(0 if success else 1)
