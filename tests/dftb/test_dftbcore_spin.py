#!/usr/bin/env python
"""Verify density-matrix export for SPIN-POLARIZED (open-shell) runs.

Bug being tested: the real-path store overwrote storedDM per spin channel
(kept only the last), and the complex path stores PER-SPIN DM slots which must
be summed over spins to get the physical density.

Systems:
  1. Single H atom, cluster, Colinear UnpairedElectrons=1 -> N_e = 1
  2. H chain, 1 atom/cell a=2.0 A, nk=4, spin-polarized -> N_e = 1/cell

Check: Tr(S . P_total) == N_e  (each spin channel separately would give less)
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'pyBall'))
from DFTBcore import DFTBcore

WORK = Path(__file__).parent / 'work'
SK = os.environ.get('DFTB_SK_PATH', os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/'))


def write_hsd(wd, name, gen, spin_block, klines=""):
    wd.mkdir(parents=True, exist_ok=True)
    (wd / f'{name}.gen').write_text(gen)
    (wd / f'{name}.hsd').write_text(f"""Geometry = GenFormat {{
  <<< "{name}.gen"
}}
ParserOptions {{ ParserVersion = 15 }}
Hamiltonian = DFTB {{
  Scc = Yes
  MaxAngularMomentum {{ H = "s" }}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{SK}3ob-3-1/"
    Separator = "-"
    Suffix = ".skf"
  }}
{spin_block}
{klines}}}
""")


def test_h_atom():
    """Single H atom, 1 unpaired electron, cluster (real path)."""
    wd = WORK / 'spin_h_atom'
    gen = """1  C
H
 1  1   0.000000000   0.000000000   0.000000000
"""
    spin = """  SpinPolarisation = Colinear { UnpairedElectrons = 1.0 }
  SpinConstants { H = { -0.05 } }
"""
    write_hsd(wd, 'hatom', gen, spin)
    cwd = os.getcwd(); os.chdir(wd)
    try:
        d = DFTBcore(); d.init('hatom.hsd')
        d.enable_matrix_collection(dm=True, h=True, s=True)
        E = d.run_scf()
        P = d.get_dm_dense(); S = d.get_s_dense()
        d.finalize()
    finally:
        os.chdir(cwd)
    ne = np.trace(P @ S).real
    print(f"H atom: E={E:.6f} Ha  Tr(SP)={ne:.6f}  (expect 1.0)")
    ok = abs(ne - 1.0) < 1e-8
    print("  -> " + ("OK (total DM)" if ok else "FAIL: only one spin channel stored"))
    return ok


def test_h_chain():
    """H chain, 1 e-/cell, spin-polarized -> nks = 2*nk slots."""
    nk = 4
    wd = WORK / 'spin_h_chain'
    gen = """1  S
H
 1  1   0.000000000   0.000000000   0.000000000
 0.000000000   0.000000000   0.000000000
 2.000000000   0.000000000   0.000000000
 0.000000000  15.000000000   0.000000000
 0.000000000   0.000000000  15.000000000
"""
    spin = """  SpinPolarisation = Colinear { UnpairedElectrons = 1.0 }
  SpinConstants { H = { -0.05 } }
"""
    kl = "  KPointsAndWeights = {\n" + "".join(
        f"  {i/nk:10.7f}  0.0  0.0   {1.0/nk:.7f}\n" for i in range(nk)) + "  }\n"
    write_hsd(wd, 'hchain', gen, spin, kl)
    cwd = os.getcwd(); os.chdir(wd)
    try:
        d = DFTBcore(); d.init('hchain.hsd')
        d.enable_matrix_collection(dm=True, h=True, s=True)
        E = d.run_scf()
        norb, nks, nkpts, nspin = d.get_cplx_dims()
        kpts, w = d.get_kpoints()
        P = d.get_dm_cplx(); S = d.get_s_cplx()
        Pt = d.get_dm_cplx_total()
        d.finalize()
    finally:
        os.chdir(cwd)

    print(f"H chain: nks={nks} nkpts={nkpts} nspin={nspin} (expect nks=2*nk)")
    ne_spin = [sum(w[ik] * np.trace(S[ik + s * nkpts] @ P[ik + s * nkpts]).real
                   for ik in range(nkpts)) for s in range(nspin)]
    ne_tot = sum(w[ik] * np.trace(S[ik] @ Pt[ik]).real for ik in range(nkpts))
    print(f"  electrons per spin channel: {np.round(ne_spin, 6)}")
    print(f"  electrons total (get_dm_cplx_total): {ne_tot:.6f}  (expect 1.0)")
    ok = (nks == 2 * nkpts and abs(ne_tot - 1.0) < 1e-8
          and abs(ne_spin[0] + ne_spin[1] - ne_tot) < 1e-10)
    print("  -> " + ("OK" if ok else "FAIL"))
    return ok


if __name__ == '__main__':
    ok1 = test_h_atom()
    ok2 = test_h_chain()
    print("=" * 60)
    print("SPIN TEST " + ("PASSED" if (ok1 and ok2) else "FAILED"))
    sys.exit(0 if (ok1 and ok2) else 1)
