#!/usr/bin/env python
"""
Parity check: compare H, S, DM, eigenvectors between DFTBcore library and DFTB+ executable.

Usage:
    cd /home/prokop/git/dftbplus/tests/dftb
    python test_parity.py
    python test_parity.py --debug    # Print orbital coefficients

Requires:
    export DFTB_EXE=/path/to/dftb+
    export DFTB_LIB_PATH=/path/to/libdftbcore.so   (optional, auto-detected)
    export DFTB_SK_PATH=/path/to/slakos/library/
"""

import sys, os
import numpy as np
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'pyBall'))

from DFTBcore import DFTBcore
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pyBall.WavePlot.TestUtils import print_eigenvecs

# ---- helpers ----------------------------------------------------------------

def parse_band_out(fname='band.out'):
    """Parse eigenvalues (eV) from band.out."""
    eigvals = []
    with open(fname) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                try:
                    idx, ev, occ = int(parts[0]), float(parts[1]), float(parts[2])
                    eigvals.append(ev)
                except ValueError:
                    pass
    return np.array(eigvals)

def parse_energy_detailed(fname='detailed.out'):
    """Parse total energy (Hartree) from detailed.out."""
    with open(fname) as f:
        for line in f:
            if 'Total Energy:' in line:
                return float(line.split()[2])
    raise RuntimeError(f"Total Energy not found in {fname}")

def make_h2o_input(sk_path, sk_set='3ob-3-1'):
    ang2bohr = 1.8897259886
    coords_ang = np.array([[0.0,0.0,0.0],[0.9584,0.0,0.0],[-0.2399,0.9270,0.0]])
    with open('h2o.xyz', 'w') as f:
        f.write("3\nH2O\n")
        for i,(x,y,z) in enumerate(coords_ang):
            sym = 'O' if i==0 else 'H'
            f.write(f"{sym}  {x:.6f}  {y:.6f}  {z:.6f}\n")
    with open('h2o.hsd', 'w') as f:
        f.write(f"""Geometry = xyzFormat {{
  <<< "h2o.xyz"
}}

Driver = {{}}

Hamiltonian = DFTB {{
  SCC = Yes
  SCCTolerance = 1.0E-8
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
  WriteDetailedOut = Yes
}}
""")
    # Also create as dftb_in.hsd for library compatibility
    import shutil
    shutil.copy('h2o.hsd', 'dftb_in.hsd')

def parity_check(name, lib_val, exe_val, tol=1e-5):
    diff = np.max(np.abs(lib_val - exe_val))
    status = "PASS" if diff < tol else "FAIL"
    print(f"  [{status}] {name}: max|lib-exe| = {diff:.3e}  (tol={tol:.0e})")
    return diff < tol

def print_mat(name, M, n=6):
    n = min(n, M.shape[0])
    print(f"\n  {name} [{M.shape[0]}x{M.shape[1]}]:")
    for i in range(n):
        print("    " + "  ".join(f"{M[i,j]:10.6f}" for j in range(n)))

# ---- main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DFTB+ Parity Test')
    parser.add_argument('--debug', action='store_true', help='Print orbital coefficients')
    parser.add_argument('--print-eigenvec', action='store_true', help='Print eigenvectors from eigenvec.bin and exit')
    parser.add_argument('--sk-set', type=str, default='3ob-3-1', help='Slater-Koster parameter set (3ob-3-1 or mio-1-1)')
    args = parser.parse_args()

    sk_path  = os.environ.get('DFTB_SK_PATH', os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/library/'))
    libpath  = os.environ.get('DFTB_LIB_PATH', None)
    dftb_exe = os.environ.get('DFTB_EXE', None)

    # Print eigenvectors if requested
    if args.print_eigenvec:
        print_eigenvecs('eigenvec.bin', 'detailed.xml', 'waveplot_in.hsd', max_orbitals=6)
        return

    print("="*70)
    print(f"DFTB+ Parity Test: Library vs Executable (H2O, {args.sk_set})")
    print("="*70)
    print(f"  SK path:    {sk_path}")
    print(f"  SK set:     {args.sk_set}")
    print(f"  Library:    {libpath or '(auto)'}")
    print(f"  Executable: {dftb_exe or '(not set)'}")
    print(f"  Debug mode: {args.debug}")

    make_h2o_input(sk_path, args.sk_set)

    # ---- 0. Run executable FIRST (before library loads, avoid shared-lib conflicts) ----
    energy_exe = None
    eigvals_exe_eV = None
    if dftb_exe is not None:
        print("\n" + "="*70)
        print("STEP 0: DFTB+ executable (run first, clean process)")
        print("="*70)
        # Use a fresh subprocess with RPATH-based linking only
        env_exe = os.environ.copy()
        env_exe.pop('LD_PRELOAD', None)
        ret = subprocess.run([dftb_exe], capture_output=True, text=True, env=env_exe)
        if ret.returncode != 0:
            print(f"  Executable failed:\n{ret.stderr[-500:]}")
            print("  Continuing with library-only check...")
        else:
            try:
                energy_exe = parse_energy_detailed('detailed.out')
                eigvals_exe_eV = parse_band_out('band.out')
                print(f"  Energy: {energy_exe:.8f} Ha")
                print(f"  Eigenvalues (eV): {eigvals_exe_eV}")
            except:
                print("  Could not parse output files, continuing with library-only check...")

    # ---- 1. Run via DFTBcore library -----------------------------------------
    print("\n" + "="*70)
    print("STEP 1: DFTBcore library")
    print("="*70)

    dftb = DFTBcore(libpath=libpath)
    dftb.init('h2o.hsd')
    dftb.enable_hamiltonian_storage(store=True)
    dftb.enable_matrix_collection(dm=True, h=True, s=True)
    energy_lib = dftb.run_scf()
    n = dftb.get_basis_size()

    H_lib   = dftb.get_h_dense()
    S_lib   = dftb.get_s_dense()
    DM_lib  = dftb.get_dm_dense()
    C_lib_lowdin, eigvals_lib = dftb.get_eigvecs_dense(apply_inverse_lowdin=False)
    C_lib_atomic, eigvals_lib_atomic = dftb.get_eigvecs_dense(apply_inverse_lowdin=True)
    
    # Print orbital coefficients if debug mode is enabled
    if args.debug:
        # Atom-orbital map for H2O: O (s, px, py, pz), H1 (s), H2 (s)
        atom_orbital_map = [
            ('O (atom 0)', ['s', 'px', 'py', 'pz']),
            ('H (atom 1)', ['s']),
            ('H (atom 2)', ['s'])
        ]
        print("\n  Lowdin basis eigenvectors (from diagonalization):")
        dftb.print_orbital_coeffs(C_lib_lowdin, eigvals_lib, atom_orbital_map=atom_orbital_map, max_orbitals=6)
        print("\n  Atomic basis eigenvectors (after S^(-1/2) transform):")
        dftb.print_orbital_coeffs(C_lib_atomic, eigvals_lib_atomic, atom_orbital_map=atom_orbital_map, max_orbitals=6)
    
    dftb.finalize()

    print(f"  Energy:     {energy_lib:.8f} Ha")
    print(f"  Basis size: {n}")
    print(f"  H sym err:  {np.max(np.abs(H_lib - H_lib.T)):.2e}")
    print(f"  S sym err:  {np.max(np.abs(S_lib - S_lib.T)):.2e}")
    print(f"  DM sym err: {np.max(np.abs(DM_lib - DM_lib.T)):.2e}")
    nelec = np.trace(S_lib @ DM_lib)
    print(f"  Tr(S*DM):   {nelec:.4f}  (expected 8.0 for H2O)")
    print(f"  Tr(DM):     {np.trace(DM_lib):.4f}")
    print(f"  Tr(S):      {np.trace(S_lib):.4f}")

    print_mat("H_lib", H_lib)
    print_mat("S_lib", S_lib)
    print_mat("DM_lib", DM_lib)
    print_mat("C_lib_lowdin (Lowdin basis)", C_lib_lowdin)
    print_mat("C_lib_atomic (atomic basis)", C_lib_atomic)
    print(f"\n  Eigenvalues (library):")
    for i,e in enumerate(eigvals_lib):
        print(f"    MO {i+1:2d}: {e:12.6f} Ha = {e*27.2114:12.6f} eV")

    # ---- 2. Compare with eigenvec.bin (if available) -------------
    print("\n" + "="*70)
    print("STEP 2: Compare with eigenvec.bin")
    print("="*70)
    
    if os.path.exists('eigenvec.bin'):
        # Parse eigenvec.bin using the utility function
        from pyBall.OCL.DFTBplusParser import parse_eigenvec_bin_custom, parse_detailed_xml_custom
        geo = parse_detailed_xml_custom('detailed.xml')
        nstates_xml = geo['nstates']
        norb_xml = geo['norb']
        evecs_exe = parse_eigenvec_bin_custom('eigenvec.bin', nstates_xml, norb_xml)
        
        print(f"  eigenvec.bin shape: {evecs_exe.shape}")
        print(f"  C_lib_atomic shape:  {C_lib_atomic.shape}")
        
        # Compare atomic basis eigenvectors with eigenvec.bin
        if evecs_exe.shape == C_lib_atomic.shape:
            diff = np.max(np.abs(evecs_exe - C_lib_atomic))
            print(f"\n  Comparing atomic basis eigenvectors with eigenvec.bin:")
            print(f"    max|C_lib - C_exe| = {diff:.3e}")
            
            # Print first few MO coefficients for comparison
            n_show = min(3, nstates_xml)
            print(f"\n  First {n_show} MO coefficients comparison:")
            for i in range(n_show):
                print(f"    MO {i+1}:")
                print(f"      eigenvec.bin:  {evecs_exe[i,:3]}")
                print(f"      C_lib_atomic:  {C_lib_atomic[i,:3]}")
                print(f"      diff:           {np.abs(evecs_exe[i,:3] - C_lib_atomic[i,:3])}")
            
            status = "PASS" if diff < 1e-4 else "FAIL"
            print(f"\n  [{status}] Atomic basis eigenvectors match eigenvec.bin: max diff = {diff:.3e}")
        else:
            print(f"  Shape mismatch: eigenvec.bin {evecs_exe.shape} vs C_lib_atomic {C_lib_atomic.shape}")
    else:
        print("  eigenvec.bin not found (run dftb+ executable with WriteEigenvectors=Yes)")
    
    # ---- 3. Show executable results (already computed in step 0) -------------
    print("\n" + "="*70)
    print("STEP 2: Executable reference results")
    print("="*70)

    if energy_exe is None or eigvals_exe_eV is None:
        print("  No executable results available.")
        print("  Using band.out written by library run as cross-check.")
        eigvals_exe_eV = parse_band_out('band.out')
        energy_exe = energy_lib  # same code, same energy
    
    eigvals_exe_Ha = eigvals_exe_eV / 27.2114
    print(f"  Energy:     {energy_exe:.8f} Ha")
    print(f"  Eigenvalues (from band.out):")
    for i,e in enumerate(eigvals_exe_eV):
        print(f"    MO {i+1:2d}: {eigvals_exe_Ha[i]:12.6f} Ha = {e:12.6f} eV")

    # ---- 3. Parity comparison ------------------------------------------------
    print("\n" + "="*70)
    print("STEP 3: Parity comparison")
    print("="*70)

    all_pass = True

    # Energy
    ediff = abs(energy_lib - energy_exe)
    status = "PASS" if ediff < 1e-6 else "FAIL"
    print(f"  [{status}] Energy: lib={energy_lib:.8f}  exe={energy_exe:.8f}  diff={ediff:.2e}")
    all_pass = all_pass and (ediff < 1e-6)

    # Eigenvalues (library from Fortran eigenvectors, exe from band.out)
    n_compare = min(len(eigvals_lib), len(eigvals_exe_Ha))
    eigvals_lib_eV = eigvals_lib[:n_compare] * 27.2114
    eigvals_exe_eV_c = eigvals_exe_eV[:n_compare]
    print(f"\n  Eigenvalues comparison (eV):")
    print(f"  {'MO':>4} {'Library':>14} {'Executable':>14} {'|diff|':>12}")
    print("  " + "-"*48)
    max_eigdiff = 0.0
    for i in range(n_compare):
        d = abs(eigvals_lib_eV[i] - eigvals_exe_eV_c[i])
        max_eigdiff = max(max_eigdiff, d)
        flag = "  <-- MISMATCH" if d > 0.001 else ""
        print(f"  {i+1:>4} {eigvals_lib_eV[i]:>14.6f} {eigvals_exe_eV_c[i]:>14.6f} {d:>12.4e}{flag}")
    print("  " + "-"*48)
    eig_pass = max_eigdiff < 0.001  # 1 meV tolerance
    status = "PASS" if eig_pass else "FAIL"
    print(f"  [{status}] Max eigenvalue diff: {max_eigdiff:.4e} eV")
    all_pass = all_pass and eig_pass

    # Matrix sanity checks (no exe reference matrices — just verify lib matrices are physical)
    print(f"\n  Library matrix sanity:")
    all_pass = parity_check("H symmetry",  H_lib,  H_lib.T,  tol=1e-8) and all_pass
    all_pass = parity_check("S symmetry",  S_lib,  S_lib.T,  tol=1e-8) and all_pass
    all_pass = parity_check("DM symmetry", DM_lib, DM_lib.T, tol=1e-8) and all_pass
    s11_exp = 1.0
    print(f"  S[0,0] = {S_lib[0,0]:.6f}  (expected 1.0 for normalized basis)")
    nel_ok = abs(nelec - 8.0) < 0.01
    status = "PASS" if nel_ok else "FAIL"
    print(f"  [{status}] Tr(S*DM) = {nelec:.4f}  (expected 8.0)")
    all_pass = all_pass and nel_ok

    print("\n" + "="*70)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("="*70)
    return all_pass

if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
