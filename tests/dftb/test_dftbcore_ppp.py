#!/usr/bin/env python
"""
Peierls-type distortion of poly(p-phenylene) (PPP) via DFTBcore.

System: C6H4 per cell, para-linked along x, cell a = 4.28 A.
  - aromatic (delta=0): uniform ring 1.40 A bonds, inter-ring 1.48 A
  - distorted (delta>0): the two para (linking, non-hydrogenated) carbons are
    moved APART along x by +-delta and FIXED; the remaining 4 ring carbons
    and 4 hydrogens relax (GeometryOptimisation + MovedAtoms).
    With fixed cell this shortens the inter-ring bond (quinoid direction)
    and elongates the ring para-para axis.

Bond orders from real-space P(R),S(R) as in test_dftbcore_peierls.py:
  inter-ring : paraC(0, atom idx 0) - paraC(+1, atom idx 3)
  ring para-adjacent: bonds (0,1),(2,3),(3,4),(5,0)   [0-based C indices]
  ring middle (top/bottom): bonds (1,2),(4,5)

Atom order in gen (1-based): 1=C(para,right) 2,3=C(top) 4=C(para,left)
  5,6=C(bottom) 7-10=H.  MovedAtoms = "2 3 5 6 7 8 9 10".

Orbitals (3ob): C -> 4 (s,px,py,pz), H -> 1 (s). norb = 28, nocc = 14.

Usage:
    cd /home/prokop/git/dftbplus/tests/dftb
    python test_dftbcore_ppp.py
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'pyBall'))

from DFTBcore import DFTBcore

WORK = Path(__file__).parent / 'work'
HA2EV = 27.2114

R_RING = 1.40    # in-ring C-C [A] (regular hexagon edge = radius)
D_INTER = 1.48   # inter-ring C-C at delta=0 [A]
D_CH = 1.09      # C-H [A]
A0 = 2 * R_RING + D_INTER   # 4.28 A

# 0-based ring indices: 0=para-right, 1=upper-right, 2=upper-left,
#                       3=para-left, 4=lower-left, 5=lower-right
RING_BONDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
PARA_ADJ = [(0, 1), (2, 3), (3, 4), (5, 0)]   # bonds touching a para C
MID_BONDS = [(1, 2), (4, 5)]                # top/bottom horizontal bonds


def ppp_coords(delta):
    """C (6,3) and H (4,3) Cartesian coords; para-right moved +2*delta along x,
    para-left stays at x=0 (cell boundary, cell-0 assignment verified at delta=0).
    Keeping all coords in [0,a) avoids DFTB+ folding atoms to other cells, which
    would shift the corresponding bond blocks to P(R=+-1) instead of R=0."""
    ang = np.deg2rad([0, 60, 120, 180, 240, 300])
    C = np.zeros((6, 3))
    C[:, 0] = R_RING * np.cos(ang) + R_RING   # shift leftmost vertex to x=0
    C[:, 1] = R_RING * np.sin(ang)
    C[0, 0] += 2 * delta    # para-right moved +x (para-para grows by 2*delta)
    H = np.zeros((4, 3))
    h_ang = np.deg2rad([60, 120, 240, 300])
    H[:, 0] = (R_RING + D_CH) * np.cos(h_ang) + R_RING
    H[:, 1] = (R_RING + D_CH) * np.sin(h_ang)
    return C, H


def make_ppp_input(workdir, sk_path, delta, nk=8, relax=True, sk_set='3ob-3-1'):
    """PPP cell, 'S' gen = periodic CARTESIAN Angstrom."""
    workdir.mkdir(parents=True, exist_ok=True)
    C, H = ppp_coords(delta)
    lines = ["10  S", "C  H"]
    for i, c in enumerate(C):
        lines.append(f" {i+1}  1   {c[0]:.9f}   {c[1]:.9f}   {c[2]:.9f}")
    for j, h in enumerate(H):
        lines.append(f" {7+j}  2   {h[0]:.9f}   {h[1]:.9f}   {h[2]:.9f}")
    lines += ["0.000000000   0.000000000   0.000000000",
              f" {A0:10.6f}   0.000000000   0.000000000",
              "0.000000000  15.000000000   0.000000000",
              "0.000000000   0.000000000  15.000000000"]
    (workdir / 'ppp.gen').write_text("\n".join(lines) + "\n")

    klines = "".join(f"  {i/nk:10.7f}  0.0  0.0   {1.0/nk:.7f}\n" for i in range(nk))
    driver = """Driver = GeometryOptimisation {
  Optimiser = LBFGS { Memory = 20 }
  MovedAtoms = 2 3 5 6 7 8 9 10     # para carbons 1,4 fixed
  Convergence { GradElem = 1e-4 }
  MaxSteps = 100
  OutputPrefix = "geo_end"
}
""" if relax else ""
    hsd = f"""{driver}Geometry = GenFormat {{
  <<< "ppp.gen"
}}

ParserOptions {{
  ParserVersion = 15
}}

Hamiltonian = DFTB {{
  Scc = Yes
  MaxAngularMomentum {{
    C = "p"
    H = "s"
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
    (workdir / 'ppp.hsd').write_text(hsd)


def read_geo_end(path):
    """Parse geo_end.gen ('S' format) -> (natom, species_idx, coords)."""
    toks = path.read_text().split()
    natom = int(toks[0]); assert toks[1].upper() == 'S'
    nsp = 2  # C H
    pos = 2 + nsp
    spec = np.zeros(natom, int); xyz = np.zeros((natom, 3))
    for i in range(natom):
        spec[i] = int(toks[pos + 1]); xyz[i] = [float(toks[pos + 2 + k]) for k in range(3)]
        pos += 5
    return spec, xyz


def fourier_to_realspace(Ak, kpts, weights, R_cells):
    out = {}
    kx = kpts[:, 0]
    for R in R_cells:
        ph = np.exp(-2j * np.pi * kx * R)
        out[R] = np.einsum('k,kij->ij', weights * ph, Ak)
    return out


def orb(atom):
    """0-based orbital slice: C atoms 0-5 -> 4 orbs; H atoms 6-9 -> 1 orb."""
    return np.arange(4 * atom, 4 * atom + 4) if atom < 6 else np.array([24 + atom - 6])


def bond_order(P_R, S_R, R, a_row, a_col):
    blk_P = P_R[R][np.ix_(orb(a_row), orb(a_col))]
    blk_S = S_R[R][np.ix_(orb(a_row), orb(a_col))]
    return np.sum((blk_P * blk_S).real)


def run_ppp(name, sk_path, delta, nk=8, relax=True):
    wd = WORK / name
    make_ppp_input(wd, sk_path, delta, nk=nk, relax=relax)
    cwd = os.getcwd()
    os.chdir(wd)
    try:
        dftb = DFTBcore()
        dftb.init('ppp.hsd')
        dftb.enable_matrix_collection(dm=True, h=True, s=True)
        Etot = dftb.run_scf()
        norb, nks, nkpts, nspin = dftb.get_cplx_dims()
        assert nkpts == nk and norb == 28, f"unexpected dims norb={norb} nk={nkpts}"
        kpts, kw = dftb.get_kpoints()
        P = dftb.get_dm_cplx(); S = dftb.get_s_cplx()
        C, E = dftb.get_eigvecs_cplx()
        dftb.finalize()
        geo_file = wd / 'geo_end.gen'
        spec, xyz = read_geo_end(geo_file) if geo_file.exists() else (None, None)
    finally:
        os.chdir(cwd)

    R_cells = [-1, 0, 1]
    P_R = fourier_to_realspace(P, kpts, kw, R_cells)
    S_R = fourier_to_realspace(S, kpts, kw, R_cells)

    bo = {}
    for i, j in RING_BONDS:
        bo[(i, j)] = bond_order(P_R, S_R, 0, j, i)          # ring bond, same cell
    bo['inter'] = bond_order(P_R, S_R, +1, 3, 0)            # para-left(+1) - para-right(0)
    bo['inter_m'] = bond_order(P_R, S_R, -1, 0, 3)          # hermiticity cross-check
    bo_para = np.mean([bo[b] for b in PARA_ADJ])
    bo_mid = np.mean([bo[b] for b in MID_BONDS])

    nocc = 14
    gap = (E[:, nocc].min() - E[:, nocc - 1].max()) * HA2EV
    return dict(name=name, delta=delta, Etot=Etot, gap=gap,
                bo_inter=bo['inter'], bo_para=bo_para, bo_mid=bo_mid,
                bo=bo, kpts=kpts, E=E, xyz=xyz, spec=spec)


def main():
    sk_path = os.environ.get('DFTB_SK_PATH', os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/'))
    nk = 8
    print("=" * 82)
    print(f"PPP Peierls test: C6H4 cell, a = {A0:.2f} A, para C displaced +-delta, nk = {nk}")
    print("=" * 82)

    results = []
    for delta in [0.0, 0.05, 0.10, 0.15]:
        r = run_ppp(f"ppp_d{delta:.2f}", sk_path, delta, nk=nk)
        results.append(r)
        print(f"\n--- delta={r['delta']:.2f} A  (para-para {2*R_RING+2*delta:.2f} A, "
              f"inter-ring start {D_INTER-2*delta:.2f} A) ---")
        print(f"  E_total     = {r['Etot']:.8f} Ha")
        print(f"  gap         = {r['gap']:.4f} eV")
        print(f"  BO_inter    = {r['bo_inter']:.4f}   (cross-check {r['bo']['inter_m']:.4f})")
        print(f"  BO_para-adj = {r['bo_para']:.4f}   BO_mid = {r['bo_mid']:.4f}")
        print("  ring bonds  : " + "  ".join(f"{i}-{j}:{r['bo'][(i,j)]:.3f}" for i, j in RING_BONDS))

    print("\n" + "=" * 82)
    print(f"{'delta':>6} {'E[Ha]':>12} {'gap[eV]':>9} {'BO_int':>8} {'BO_para':>8} {'BO_mid':>8}")
    for r in results:
        print(f"{r['delta']:6.2f} {r['Etot']:12.6f} {r['gap']:9.4f} "
              f"{r['bo_inter']:8.4f} {r['bo_para']:8.4f} {r['bo_mid']:8.4f}")

    ok = all(abs(r['bo_inter'] - r['bo']['inter_m']) < 1e-6 for r in results)
    print("\n[Hermiticity cross-check inter vs R=-1]:", "OK" if ok else "FAILED")
    print("=" * 82)
    return results


if __name__ == '__main__':
    main()
