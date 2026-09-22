#!/usr/bin/env python
"""
Peierls distortion in a carbon chain (SSH-model physics) via DFTBcore.

System: 2-atom periodic carbon chain (polyyne), cell a along x.
  - uniform    : d_intra = a/2          -> equal bonds  -> metal (gap ~ 0)
  - dimerized  : d_intra != a/2         -> alternating bonds -> gap opens

Analysis (the point of this test):
  Export the k-space density matrix P(k) and overlap S(k) via DFTBcore's
  complex getters, inverse-Fourier-transform them to real-space cell pairs

      A_{j(R),i(0)} = sum_k w_k * exp(-i k . R) * A(k)[j,i]

  (consistent with unpackHS: A(k)[j,i] = sum_R e^{ikR} A_{j(R),i(0)}),
  then form the Mulliken bond order between atom A in cell 0 and atom B
  in cell R:

      BO_AB(R) = sum_{mu in A} sum_{nu in B} Re[ P_{nu,mu}(R) * S_{nu,mu}(R) ]

  intra-cell bond: A=C1(0), B=C2(0)    -> block [4:8, 0:4] at R=0
  inter-cell bond: A=C2(0), B=C1(+1)   -> block [0:4, 4:8] at R=+1
  (equivalently  [4:8, 0:4] at R=-1 by Hermiticity)

Checks:
  - uniform:   BO_intra ~= BO_inter,  gap small (metal)
  - dimerized: BO_intra  > BO_inter,  gap opens (Peierls / SSH)

Usage:
    cd /home/prokop/git/dftbplus/tests/dftb
    python test_dftbcore_peierls.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'pyBall'))

from DFTBcore import DFTBcore

WORK = Path(__file__).parent / 'work'
HA2EV = 27.2114


def make_chain_input(workdir, sk_path, d_intra, nk=8, a=2.60, sk_set='3ob-3-1'):
    """2-atom carbon chain along x; atoms at x=0 and x=d_intra.

    NOTE: gen 'S' format takes CARTESIAN coordinates in Angstrom
    ('F' is the fractional variant) -- verified empirically that 'S'
    does not apply the lattice-vector conversion.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    gen_content = f"""2  S
C
 1  1   0.000000000   7.500000000   7.500000000
 2  1   {d_intra:.9f}   7.500000000   7.500000000
 0.000000000   0.000000000   0.000000000
 {a:10.6f}   0.000000000   0.000000000
 0.000000000  15.000000000   0.000000000
 0.000000000   0.000000000  15.000000000
"""
    (workdir / 'chain.gen').write_text(gen_content)

    klines = "".join(f"  {i/nk:10.7f}  0.0  0.0   {1.0/nk:.7f}\n" for i in range(nk))
    hsd_content = f"""Geometry = GenFormat {{
  <<< "chain.gen"
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
{klines}  }}
}}

Options {{
  WriteResultsTag = Yes
  WriteDetailedOut = Yes
}}
"""
    (workdir / 'chain.hsd').write_text(hsd_content)


def fourier_to_realspace(Ak, kpts, weights, R_cells):
    """A_{j(R),i(0)} = sum_k w_k e^{-i k.R} A(k)[j,i].

    Ak: (nks, norb, norb) complex; kpts: (nk,3) fractional; R_cells: list of int
    cell indices along x (chain direction). Returns dict R -> (norb, norb).
    """
    out = {}
    kx = kpts[:, 0]  # chain along x -> only x-component of fractional k matters
    for R in R_cells:
        ph = np.exp(-2j * np.pi * kx * R)
        out[R] = np.einsum('k,kij->ij', weights * ph, Ak)
    return out


def bond_order(P_R, S_R, R, row_orb, col_orb):
    """Mulliken bond order between atoms: rows = B orbitals in cell R,
    cols = A orbitals in cell 0.  BO = sum Re[P_nu,mu(R) * S_nu,mu(R)]."""
    blk_P = P_R[R][np.ix_(row_orb, col_orb)]
    blk_S = S_R[R][np.ix_(row_orb, col_orb)]
    return np.sum((blk_P * blk_S).real)


def run_chain(name, sk_path, d_intra, nk=8, a=2.60):
    """Run one geometry; return dict with E, gap, bond orders."""
    wd = WORK / name
    make_chain_input(wd, sk_path, d_intra, nk=nk, a=a)
    cwd = os.getcwd()
    os.chdir(wd)
    try:
        dftb = DFTBcore()
        dftb.init('chain.hsd')
        dftb.enable_matrix_collection(dm=True, h=True, s=True)
        Etot = dftb.run_scf()
        norb, nks, nkpts, nspin = dftb.get_cplx_dims()
        assert nkpts == nk and norb == 8, f"unexpected dims norb={norb} nk={nkpts}"
        kpts, kw = dftb.get_kpoints()
        H = dftb.get_h_cplx(); S = dftb.get_s_cplx(); P = dftb.get_dm_cplx()
        C, E = dftb.get_eigvecs_cplx()
        dftb.finalize()
    finally:
        os.chdir(cwd)

    # --- inverse FT to real-space cell pairs ---
    R_cells = [-1, 0, 1]
    P_R = fourier_to_realspace(P, kpts, kw, R_cells)
    S_R = fourier_to_realspace(S, kpts, kw, R_cells)

    # C1 = atom 0 (orbs 0..3), C2 = atom 1 (orbs 4..7)
    o1, o2 = np.arange(0, 4), np.arange(4, 8)
    bo_intra = bond_order(P_R, S_R, 0, o2, o1)   # C2(0) <-> C1(0)
    bo_inter = bond_order(P_R, S_R, +1, o1, o2)  # C1(+1) <-> C2(0)
    # cross-check: same inter bond seen from the other side, R=-1
    bo_inter_m = bond_order(P_R, S_R, -1, o2, o1)

    # --- band gap: 8 electrons/cell -> 4 doubly-occupied bands ---
    nocc = 4
    homo_max = E[:, nocc - 1].max()
    lumo_min = E[:, nocc].min()
    gap = (lumo_min - homo_max) * HA2EV
    gap_edge = (E[:, nocc] - E[:, nocc - 1]) * HA2EV  # direct gap per k

    return dict(name=name, d=d_intra, d_inter=a - d_intra, Etot=Etot,
                gap=gap, gap_edge=gap_edge,
                bo_intra=bo_intra, bo_inter=bo_inter, bo_inter_m=bo_inter_m,
                kpts=kpts, E=E)


def main():
    sk_path = os.environ.get('DFTB_SK_PATH', os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/'))
    a = 2.60
    nk = 8
    print("=" * 78)
    print("Peierls / SSH test: carbon chain, 2-atom cell, a = %.2f A, nk = %d" % (a, nk))
    print("=" * 78)

    results = []
    for d in [a / 2, 1.25, 1.20, 1.15, 1.10]:
        name = f"d{d:.2f}"
        r = run_chain(name, sk_path, d, nk=nk, a=a)
        results.append(r)
        print(f"\n--- d_intra={r['d']:.3f} A  d_inter={r['d_inter']:.3f} A ---")
        print(f"  E_total        = {r['Etot']:.8f} Ha")
        print(f"  gap(global)    = {r['gap']:.4f} eV   gap(k=edge min) = {r['gap_edge'].min():.4f} eV")
        print(f"  BO_intra(R=0)  = {r['bo_intra']:.4f}")
        print(f"  BO_inter(R=+1) = {r['bo_inter']:.4f}   (R=-1 check: {r['bo_inter_m']:.4f})")

    print("\n" + "=" * 78)
    print(f"{'d_intra':>8} {'d_inter':>8} {'E[Ha]':>12} {'gap[eV]':>9} {'BO_intra':>9} {'BO_inter':>9} {'ratio':>7}")
    for r in results:
        ratio = r['bo_intra'] / max(r['bo_inter'], 1e-12)
        print(f"{r['d']:8.3f} {r['d_inter']:8.3f} {r['Etot']:12.6f} {r['gap']:9.4f} "
              f"{r['bo_intra']:9.4f} {r['bo_inter']:9.4f} {ratio:7.3f}")

    uni, dim = results[0], results[-1]
    ok = True
    ok &= abs(uni['bo_intra'] - uni['bo_inter']) < 0.05 * uni['bo_intra']
    ok &= dim['bo_intra'] > dim['bo_inter']
    ok &= dim['gap'] > uni['gap']
    ok &= abs(dim['bo_inter'] - dim['bo_inter_m']) < 1e-6  # Hermiticity cross-check
    print("\n[Checks]")
    print(f"  uniform: |BO_intra - BO_inter| = {abs(uni['bo_intra']-uni['bo_inter']):.4f} (expect ~0)")
    print(f"  dimerized BO ratio = {dim['bo_intra']/dim['bo_inter']:.3f} (expect > 1)")
    print(f"  gap: uniform {uni['gap']:.3f} eV -> dimerized {dim['gap']:.3f} eV")
    print("=" * 78)
    print("Test PASSED" if ok else "Test FAILED")
    print("=" * 78)
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
