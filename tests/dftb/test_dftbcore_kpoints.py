#!/usr/bin/env python
"""
Test DFTBcore complex (k-point) interface with a periodic graphene unit cell.

This test:
1. Creates a DFTB+ input file for a 2-atom graphene cell with a 2x2x1 k-mesh
   (non-Gamma k-points force the complex Hamiltonian path, tRealHS=.false.)
2. Runs SCF via the DFTBcore ctypes interface
3. Extracts complex H(k), S(k), DM(k), eigenvectors and eigenvalues per k-point
4. Verifies:
   - Hermiticity of H(k), S(k), P(k)
   - Re-diagonalization of (H(k), S(k)) reproduces stored eigenvalues
   - Stored eigenvectors satisfy H(k) C = E S(k) C
   - Electron count: sum_k w_k * Tr(S(k) P(k)) = 8 (2 C atoms, filling=2)
   - Parity of eigenvalues vs band.out written by the dftb+ executable

Usage:
    cd /home/prokop/git/dftbplus/tests/dftb
    python test_dftbcore_kpoints.py

Requirements:
    - libdftbcore.so built (cmake --build _build --target dftbcore)
    - Slater-Koster files (DFTB_SK_PATH env var, default ~/SIMULATIONS/dftbplus/slakos/)
    - optional: DFTB_EXE env var or _build/app/dftb+/dftb+ for band.out parity
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'pyBall'))

from DFTBcore import DFTBcore

WORK = Path(__file__).parent / 'work'
A2B = 1.8897259886  # Angstrom -> Bohr


def create_graphene_input(sk_path, sk_set='3ob-3-1'):
    """Create gen + hsd input for a 2-atom graphene unit cell, 2x2x1 k-mesh."""
    WORK.mkdir(exist_ok=True)
    a = 2.46  # graphene lattice constant [Angstrom]

    # gen 'S' takes Cartesian coords [Angstrom] ('F' is fractional);
    # atom 2 at (a1+a2)/3 -> C-C distance 1.42 A
    gen_content = f"""2  S
C
 1  1   0.000000000   0.000000000   0.000000000
 2  1   {a*0.5:10.6f}   {a*0.288675135:10.6f}   0.000000000
 0.000000000   0.000000000   0.000000000
 {a:10.6f}   0.000000000   0.000000000
 {a/2:10.6f}   {a*0.866025404:10.6f}   0.000000000
 0.000000000   0.000000000  20.000000000
"""
    (WORK / 'graphene.gen').write_text(gen_content)

    hsd_content = f"""Geometry = GenFormat {{
  <<< "graphene.gen"
}}

ParserOptions {{
  ParserVersion = 15
}}

Hamiltonian = DFTB {{
  Scc = Yes
  MaxAngularMomentum {{
    C = "p"
  }}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_path}{sk_set}/"
    Separator = "-"
    Suffix = ".skf"
  }}
  KPointsAndWeights = {{
     0.0  0.0  0.0   0.25
     0.5  0.0  0.0   0.25
     0.0  0.5  0.0   0.25
     0.5  0.5  0.0   0.25
  }}
}}

Options {{
  WriteResultsTag = Yes
  WriteDetailedOut = Yes
}}

Analysis {{
  WriteBandOut = Yes
}}
"""
    (WORK / 'graphene.hsd').write_text(hsd_content)
    print(f"[Setup] Created {WORK/'graphene.gen'} and {WORK/'graphene.hsd'}")
    return WORK / 'graphene.hsd'


def parse_band_out(path):
    """Parse band.out -> (weights[nk], evals[nk, nlev] in eV) for spin channel 1.

    Format: ' KPT   1  SPIN   1  KWEIGHT   0.25' header, then 'idx  E[eV]  occ' lines.
    """
    weights, evals = [], []
    cur_vals, in_block = [], False
    for line in open(path):
        if line.strip().startswith('KPT'):
            if in_block and cur_vals:
                evals.append(cur_vals)
            weights.append(float(line.split()[-1]))
            cur_vals, in_block = [], True
        elif in_block:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    cur_vals.append(float(parts[1]))
                except ValueError:
                    in_block = False
            else:
                if cur_vals:
                    evals.append(cur_vals)
                in_block = False
    if in_block and cur_vals:
        evals.append(cur_vals)
    return np.array(weights), np.array(evals)


def test_graphene_kpoints():
    print("=" * 70)
    print("DFTBcore Test: graphene unit cell, 2x2x1 k-mesh (complex path)")
    print("=" * 70)

    sk_path = os.environ.get('DFTB_SK_PATH', os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/'))
    dftb_exe = os.environ.get('DFTB_EXE',
                              str(Path(__file__).parent.parent.parent / '_build/app/dftb+/dftb+'))
    print(f"\n[Config] SK path: {sk_path}")
    print(f"[Config] DFTB+ exe: {dftb_exe}")

    create_graphene_input(sk_path)
    os.chdir(WORK)

    ok = True
    dftb = DFTBcore()
    dftb.init('graphene.hsd')
    dftb.enable_matrix_collection(dm=True, h=True, s=True)
    energy = dftb.run_scf()

    norb, nks, nkpts, nspin = dftb.get_cplx_dims()
    print(f"\n[Results] E = {energy:.8f} Ha = {energy*27.2114:.6f} eV")
    print(f"[Dims] norb={norb} nks={nks} nkpts={nkpts} nspin={nspin}")
    assert nkpts == 4 and nks == 4 and nspin == 1, "expected 4 k-points, 1 spin"
    assert norb == 8, f"expected 8 orbitals (2x C with s+p), got {norb}"

    kpts, kw = dftb.get_kpoints()
    print(f"[Kpoints]\n{kpts}\n  weights: {kw}")
    assert np.isclose(kw.sum(), 1.0), "k-weights must sum to 1"

    H = dftb.get_h_cplx()          # [iks, i, j]
    S = dftb.get_s_cplx()
    P = dftb.get_dm_cplx()
    C, E = dftb.get_eigvecs_cplx()  # C[iks, mo, orb], E[iks, mo]
    print(f"[Shapes] H{H.shape} S{S.shape} P{P.shape} C{C.shape} E{E.shape}")

    # 1) Hermiticity
    for name, M in [('H', H), ('S', S), ('P', P)]:
        err = max(np.abs(M[k] - M[k].conj().T).max() for k in range(nks))
        print(f"  {name}(k) Hermiticity error: {err:.3e}")
        assert err < 1e-12, f"{name}(k) not Hermitian"

    # 2) stored eigenvectors satisfy H C = E S C  (C[mo, orb] rows are MOs)
    from scipy.linalg import eigh as geigh
    max_res, max_eval_diff = 0.0, 0.0
    for iks in range(nks):
        Ck = C[iks].conj().T  # [orb, mo] columns are MOs
        res = np.abs(H[iks] @ Ck - S[iks] @ Ck * E[iks][None, :]).max()
        ev, _ = geigh(H[iks], S[iks])
        d = np.abs(np.sort(ev) - np.sort(E[iks])).max()
        max_res, max_eval_diff = max(max_res, res), max(max_eval_diff, d)
        print(f"  iks={iks}: eigvec residual={res:.3e}  |E_stored-E_rediag|max={d:.3e}")
    assert max_res < 1e-8, "eigvec residual too large"
    assert max_eval_diff < 1e-8, "stored eigenvalues do not match re-diagonalization"

    # 3) electron count: sum_k w_k Tr(S(k) P(k)) = N_e = 8
    ne = sum(kw[ik] * np.trace(S[iks] @ P[iks]).real
             for iks, ik in enumerate([ik for ik in range(nkpts)]))
    # iks = ik + ispin*nkpts; nspin=1 so iks==ik
    print(f"  Electron count sum_k w_k Tr(S P) = {ne:.6f}  (expect 8)")
    assert abs(ne - 8.0) < 1e-6, "electron count wrong"

    # 4) parity vs dftb+ executable (band.out)
    if os.path.exists(dftb_exe):
        import subprocess, shutil
        # dftb+ executable always reads 'dftb_in.hsd' in the working directory
        shutil.copy('graphene.hsd', 'dftb_in.hsd')
        r = subprocess.run([dftb_exe], capture_output=True, text=True)
        print(f"[dftb+ exe] returncode={r.returncode}")
        assert r.returncode == 0, r.stdout[-2000:]
        bw, be = parse_band_out('band.out')
        be = be / 27.2114  # eV -> Hartree
        print(f"[band.out] {be.shape[0]} k-points x {be.shape[1]} levels")
        for iks in range(nks):
            d = np.abs(np.sort(be[iks]) - np.sort(E[iks])).max()
            print(f"  iks={iks} k={kpts[iks]}: max |E_lib - E_exe| = {d:.3e}")
            ok = ok and d < 1e-5  # band.out precision is 4 decimals eV (~2e-6 Ha)
        assert ok, "eigenvalue parity vs band.out failed"
    else:
        print("[SKIP] dftb+ executable not found")

    dftb.finalize()
    print("\n" + "=" * 70)
    print("Test PASSED: complex k-point matrices correct")
    print("=" * 70)
    return True


if __name__ == '__main__':
    success = test_graphene_kpoints()
    sys.exit(0 if success else 1)
