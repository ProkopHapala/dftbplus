#!/usr/bin/env python3
"""Near-EF Bloch LDOS of the relaxed vacuum ribbons (mio-1-1).

Reads geom.out.gen from the SPAMMM enum (periodic along x) and projects
DFTBcore C(k) with project_bloch_points. The map is Σ 2 w_k |ψ|² for
states within 0.5 eV of the Fermi level. One cell is enough for that
sum (it is lattice-periodic). A separate hue plot of the single state
closest to EF, drawn on three cells, shows the e^{ikx} phase.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pyBall.DFTBcore import DFTBcore
from pyBall.OCL.DFTBplusGridProjector import DFTBplusGridProjector
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang
from pyBall.plotUtils import plot_complex_hsv, plot_ldos_row

HA2EV = 27.211386
Z_ABOVE = 1.0
DX = 0.12
HALF_WIN = 0.5
RIBBON_ROOT = Path('/home/prokop/git/SPAMMM/debug/ribbon_mio')
WFC = Path(__file__).parent / 'dftb_ptcda' / 'wfc.mio-1-1.hsd'
WORK = Path(__file__).parent / 'work' / 'ribbon_bloch'
OUT = REPO / 'debug' / 'ribbon_bloch'
ANG = {'H': 's', 'C': 'p', 'N': 'p', 'O': 'p'}
STATES = ['0H', '1H-p', '1H-d', '2H-ss-adj', '2H-ss-sep', '2H-os-adj', '2H-os-sep']
WIDTHS = (1, 2, 3, 4, 5)
# Protonation series on the narrowest ribbon, plus the bare edge at every width.
CASES = [(chem, 1, st) for chem in ('C', 'N', 'O') for st in STATES]
CASES += [(chem, w, '0H') for chem in ('C', 'N', 'O') for w in WIDTHS if w != 1]


def read_gen(path):
    lines = Path(path).read_text().splitlines()
    head = lines[0].split()
    natoms, kind = int(head[0]), head[1]
    if kind != 'S':
        raise RuntimeError(f'{path}: expected supercell gen (S), got {kind}')
    elems = lines[1].split()
    pos = np.zeros((natoms, 3))
    sym = []
    for i in range(natoms):
        p = lines[2 + i].split()
        ispe = int(p[1]) - 1
        if ispe < 0 or ispe >= len(elems):
            raise RuntimeError(f'{path}: atom {i+1} species index {ispe+1} outside {elems}')
        sym.append(elems[ispe])
        pos[i] = [float(p[2]), float(p[3]), float(p[4])]
    origin = np.array([float(x) for x in lines[2 + natoms].split()])
    if np.linalg.norm(origin) > 1e-8:
        raise RuntimeError(f'{path}: cell origin {origin} is not zero')
    lat = np.array([[float(x) for x in lines[3 + natoms + i].split()] for i in range(3)])
    return sym, pos, lat


def kpoints_text(hsd_text, src):
    m = re.search(r'KPointsAndWeights\s*=\s*SupercellFolding\s*\{([^}]*)\}', hsd_text)
    if not m:
        raise RuntimeError(f'{src}: no SupercellFolding k-mesh')
    return m.group(1).strip('\n')


def sk_prefix(hsd_text, src):
    m = re.search(r'(?<![\w])Prefix\s*=\s*(\S+)', hsd_text)
    if not m:
        raise RuntimeError(f'{src}: no Slater-Koster Prefix')
    return m.group(1)


def write_case(wd, sym, pos, lat, kblock, prefix):
    wd.mkdir(parents=True, exist_ok=True)
    elems = []
    for s in sym:
        if s not in elems:
            elems.append(s)
    idx = {e: i + 1 for i, e in enumerate(elems)}
    body = [f'{len(sym)}  S', ' '.join(elems)]
    for i, (s, r) in enumerate(zip(sym, pos)):
        body.append(f' {i+1:4d}  {idx[s]:2d}  {r[0]:16.8f}  {r[1]:16.8f}  {r[2]:16.8f}')
    body.append('  0.00000000  0.00000000  0.00000000')
    for v in lat:
        body.append(f'  {v[0]:16.8f}  {v[1]:16.8f}  {v[2]:16.8f}')
    (wd / 'geom.gen').write_text('\n'.join(body) + '\n')
    ang = []
    for e in elems:
        if e not in ANG:
            raise RuntimeError(f'no angular momentum for element {e}')
        ang.append(f'    {e} = "{ANG[e]}"')
    hsd = f"""Geometry = GenFormat {{
  <<< "geom.gen"
}}
ParserOptions {{ ParserVersion = 15 }}
Hamiltonian = DFTB {{
  Scc = Yes
  MaxAngularMomentum {{
{chr(10).join(ang)}
  }}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = {prefix}
    Separator = "-"
    Suffix = ".skf"
  }}
  KPointsAndWeights = SupercellFolding {{
{kblock}
  }}
  SCCTolerance = 1.0e-8
  MaxSccIterations = 200
  Mixer = DIIS {{ Generations = 8 }}
  Filling = Fermi {{ Temperature [K] = 300 }}
}}
"""
    (wd / 'dftb_in.hsd').write_text(hsd)


def fermi_ev(path):
    text = Path(path).read_text()
    hits = re.findall(r'Fermi level:\s+([+-]?\d+\.\d+)\s+H\s+([+-]?\d+\.\d+)\s+eV', text)
    if len(hits) != 1:
        raise RuntimeError(f'{path}: expected one Fermi level, found {hits}')
    return float(hits[0][1])


def run_dftb(dftb, hsd_name):
    dftb.init(hsd_name, 'dftb.log')
    energy = dftb.run_scf()
    C, E = dftb.get_eigvecs_cplx()
    kpts, kw = dftb.get_kpoints()
    dftb.finalize()
    return energy, C, E, kpts, kw


def select_states(C, E, kpts, kw, ef):
    """States with |E−EF| < HALF_WIN. If that window is empty, the frontier orbital on each side, at every k."""
    ev = E * HA2EV
    elo, ehi = ef - HALF_WIN, ef + HALF_WIN
    pick = (ev >= elo) & (ev < ehi)
    if not np.any(pick):
        pick = np.zeros(ev.shape, dtype=bool)
        for ik in range(ev.shape[0]):
            below = np.where(ev[ik] < ef)[0]
            above = np.where(ev[ik] >= ef)[0]
            if below.size == 0 or above.size == 0:
                raise RuntimeError(
                    f'k {ik} has no state on both sides of EF={ef:.3f}; E={ev[ik].min():.3f}..{ev[ik].max():.3f}')
            pick[ik, below[np.argmax(ev[ik, below])]] = True
            pick[ik, above[np.argmin(ev[ik, above])]] = True
        print(f'  [window] nothing inside ±{HALF_WIN} eV; frontier pair at each k')
    rows, ks, ws, es = [], [], [], []
    for ik, mo in zip(*np.nonzero(pick)):
        rows.append(C[ik, mo])
        ks.append(kpts[ik])
        ws.append(2.0 * kw[ik])
        es.append(ev[ik, mo])
    return np.array(rows), np.array(ks), np.array(ws), np.array(es)


def cells_x(a1, n):
    cart, frac = [], []
    for n1 in range(-n, n + 1):
        cart.append(n1 * a1)
        frac.append((float(n1), 0.0, 0.0))
    return np.array(cart), np.array(frac)


def plane(pos, a1, nrep, heavy):
    z = float(np.median(pos[heavy, 2])) + Z_ABOVE
    y0 = float(pos[heavy, 1].min()) - 1.5
    y1 = float(pos[heavy, 1].max()) + 1.5
    x0 = -0.4
    x1 = nrep * float(a1[0]) + 0.4
    xs = np.arange(x0, x1 + 0.5 * DX, DX)
    ys = np.arange(y0, y1 + 0.5 * DX, DX)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    pts = np.stack([xx, yy, np.full(xx.shape, z)], axis=-1).reshape(-1, 3)
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    return pts, extent, xx.shape


def overlay(sym, pos, a1, extent, nrep):
    chunks_r, chunks_e = [], []
    for n in range(nrep + 1):
        r = pos + n * a1
        inside = (
            (r[:, 0] >= extent[0] - 0.3) & (r[:, 0] <= extent[1] + 0.3) &
            (r[:, 1] >= extent[2] - 0.3) & (r[:, 1] <= extent[3] + 0.3)
        )
        chunks_r.append(r[inside])
        chunks_e.extend([sym[i] for i in np.nonzero(inside)[0]])
    return np.concatenate(chunks_r), chunks_e


def main():
    if not WFC.is_file():
        raise RuntimeError(f'missing basis {WFC}')
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'basis.hsd').write_text(
        'Basis {\n  Resolution = 0.1\n  <<+ "%s"\n}\n' % os.path.relpath(WFC, WORK)
    )
    species = parse_basis_hsd_ang(str(WORK / 'basis.hsd'))
    by_name = {sp['name']: i for i, sp in enumerate(species)}
    basis = {'species': species}
    proj = DFTBplusGridProjector(verbosity=0)
    proj.load_basis_dftb(basis)
    dftb = DFTBcore()
    cwd0 = os.getcwd()
    proton = {c: [] for c in ('C', 'N', 'O')}
    bare = {c: [] for c in ('C', 'N', 'O')}

    for chem, width, state in CASES:
        tag = f'enumv_{chem}_r{width}_{state}'
        src = RIBBON_ROOT / f'enumv_{chem}_r{width}' / state
        gen = src / 'geom.out.gen'
        hsd_src = src / 'dftb_in.hsd'
        if not gen.is_file() or not hsd_src.is_file():
            raise RuntimeError(f'missing relaxed input in {src}')
        sym, pos, lat = read_gen(gen)
        a1, a2, a3 = lat
        if max(abs(a1[1]), abs(a1[2]), abs(a2[0]), abs(a2[2])) > 1e-3:
            raise RuntimeError(f'{tag}: lattice is tilted, this slice assumes an orthogonal cell\n{lat}')
        text = hsd_src.read_text()
        wd = WORK / tag
        write_case(wd, sym, pos, lat, kpoints_text(text, hsd_src), sk_prefix(text, hsd_src))
        os.chdir(wd)
        try:
            energy, C, E, kpts, kw = run_dftb(dftb, 'dftb_in.hsd')
            ef = fermi_ev(wd / 'detailed.out')
        finally:
            os.chdir(cwd0)
        ispec = np.array([by_name[s] for s in sym], dtype=np.int32)
        atoms = proj.prepare_atoms_dftb(pos, ispec, basis)
        norb = int(atoms['i0orb'][-1] + atoms['norb'][-1])
        if norb != C.shape[2]:
            raise RuntimeError(f'{tag}: basis norb {norb} != DFTB norb {C.shape[2]}')
        coeff, ks, ws, es = select_states(C, E, kpts, kw, ef)
        print(f'[{tag}] E={energy:.6f} Ha  natoms={len(sym)}  norb={norb}  nk={len(kpts)}  '
              f'EF={ef:.3f} eV  nstate={len(es)}  E=[{es.min():.3f},{es.max():.3f}]', flush=True)

        heavy = np.array([s != 'H' for s in sym])
        pts, extent, shape = plane(pos, a1, nrep=1, heavy=heavy)
        cell_cart, cell_n = cells_x(a1, n=2)
        rho, _ = proj.project_bloch_points(pts, atoms, cell_cart, cell_n, coeff, ks, ws, write_psi=False)
        if not np.isfinite(rho).all() or float(rho.max()) <= 0.0:
            raise RuntimeError(f'{tag}: LDOS max={float(np.max(rho))}')
        ny, nx = shape
        at, en = overlay(sym, pos, a1, extent, nrep=1)
        panel = {
            'rho': rho.reshape(ny, nx),
            'extent': extent,
            'atoms': at,
            'enames': en,
            'title': f'{state}  r{width}\n{es.min():.2f}..{es.max():.2f} eV\nn={len(es)}',
        }
        if width == 1:
            proton[chem].append(panel)
        if state == '0H':
            bare[chem].append(panel)

        if state == '0H' and width in (1, 5):
            j = int(np.argmin(np.abs(es - ef)))
            pts3, ext3, sh3 = plane(pos, a1, nrep=3, heavy=heavy)
            psi, = _one_state(proj, pts3, atoms, cell_cart, cell_n, coeff[j], ks[j])
            at3, en3 = overlay(sym, pos, a1, ext3, nrep=3)
            ny3, nx3 = sh3
            span = ext3[1] - ext3[0]
            height = ext3[3] - ext3[2]
            phase_path = OUT / f'{tag}_phase.png'
            plot_complex_hsv(
                psi.reshape(ny3, nx3), ext3, at3,
                f'{tag}  closest to EF  E={es[j]:.3f} eV  k=[{ks[j,0]:.3f} {ks[j,1]:.3f} {ks[j,2]:.3f}]  z=+{Z_ABOVE:.1f} Å',
                str(phase_path), enames=en3, figsize=(max(8.0, 3.2 * span / height), 3.6),
            )
            print(f'  phase {phase_path}', flush=True)

    for chem in ('C', 'N', 'O'):
        path = OUT / f'enumv_{chem}_r1_ldos.png'
        plot_ldos_row(
            proton[chem], str(path),
            f'enumv_{chem} r1  mio-1-1  Σ 2 w_k |ψ|²   near EF (±{HALF_WIN:.1f} eV, else frontier)   z=+{Z_ABOVE:.1f} Å',
        )
        print(f'[plot] {path}', flush=True)
        path = OUT / f'enumv_{chem}_0H_widths.png'
        plot_ldos_row(
            bare[chem], str(path),
            f'enumv_{chem} 0H  widths r1–r5  Σ 2 w_k |ψ|²   near EF   z=+{Z_ABOVE:.1f} Å',
        )
        print(f'[plot] {path}', flush=True)


def _one_state(proj, pts, atoms, cell_cart, cell_n, coeff, k):
    rho, psi = proj.project_bloch_points(
        pts, atoms, cell_cart, cell_n,
        coeff.reshape(1, -1), k.reshape(1, 3), np.array([1.0]), write_psi=True,
    )
    return (psi[0],)


if __name__ == '__main__':
    main()
