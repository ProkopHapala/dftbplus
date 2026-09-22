#!/usr/bin/env python3
"""Systematic near-EF Tersoff-Hamann LDOS + unfolded bands for the vacuum ribbons.

Extends test_ribbon_bloch.py (same DFTBcore C(k) -> project_bloch_points path;
see doc/prokop/userguide/bloch_slice.md) to the full enum:

  chem {C,N,O} x width r1..r5 x 7 protonation states (0H, 1H-p/d, 2H-ss/os-adj/sep)

Differences vs test_ribbon_bloch.py:
  * dense k-mesh along the ribbon (NKX, default 48 supercell points = 24 stored
    after the +/-k fold) instead of copying the 8x1x1 relaxation mesh
  * TWO half-windows per state: occupied [EF-W, EF) and empty [EF, EF+W]
    (~ STM at -W / +W bias); per-side frontier fallback for real gaps
  * band structure unfolded to the primitive (1-cell) BZ via SPAMMM's
    unfold_spectral_weights; pristine 0H unfolded bands overlaid as reference
  * green '+' marks the switched sites (heavy atom whose H-count differs
    from the 0H cell)

Outputs (debug/ribbon_stm_sys/):
  enumv_{chem}_r{w}_{occ,unocc}_ldos.png   7 states in a row, per-panel max
  enumv_{chem}_r{w}_bands.png             unfolded bands, one row per state
  index.html                              gallery

Usage:
  python test_ribbon_stm_sys.py                      # everything (cache-aware)
  python test_ribbon_stm_sys.py --chem C --widths 1  # subset
  python test_ribbon_stm_sys.py --plot-only          # replot from work/*.npz

Per-case cache lives in tests/grid/work/ribbon_stm_sys/*.npz (gitignored).
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, '/home/prokop/git/SPAMMM')          # spammm builders + unfold helpers

from pyBall.DFTBcore import DFTBcore
from pyBall.OCL.DFTBplusGridProjector import DFTBplusGridProjector
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang
from pyBall.plotUtils import plot_ldos_row, scatter_atoms

from test_ribbon_bloch import (read_gen, sk_prefix, write_case, fermi_ev,
                               run_dftb, cells_x, plane, overlay, HA2EV)

from spammm.topology.ribbon_pbc import build_edge_switch_cell, junction_state_strings, xscan_state_strings
from spammm.quantum.pi_bond_order import subcell_group_indices, unfold_spectral_weights

RIBBON_ROOT = Path('/home/prokop/git/SPAMMM/debug/ribbon_mio')
WFC = Path(__file__).parent / 'dftb_ptcda' / 'wfc.mio-1-1.hsd'
WORK = Path(__file__).parent / 'work' / 'ribbon_stm_sys'
OUT = REPO / 'debug' / 'ribbon_stm_sys'

NKX = 48          # supercell mesh along x (24 irreducible -> 96 primitive channels)
HALF_WIN = 0.5    # eV, each side of EF
BAND_WIN = 1.8    # eV, unfolded-bands y-range around EF
Z_ABOVE = 1.0     # slice height over the molecular plane (pz is zero at z=0)
DX = 0.12
NCELLS = 4        # supercell = 4 primitive cells (ncells of the enum; 8 for --xscan)
TAG = 'enumv'     # enum dir prefix (xscanv for --xscan)
STATES = ['0H', '1H-p', '1H-d', '2H-ss-adj', '2H-ss-sep', '2H-os-adj', '2H-os-sep']
STATE_STRS = junction_state_strings(NCELLS)
WIDTHS = (1, 2, 3, 4, 5)


def dense_kblock(nkx):
    """SupercellFolding text: nkx along x, half-shifted (same convention as the enum inputs)."""
    return f'{nkx} 0 0\n    0 1 0\n    0 0 1\n    0.5 0.0 0.0'


def ideal_geom(chem, width, state):
    """Builder geometry (same atom order as the enum's gen) for subcell grouping."""
    st = STATE_STRS[state].split('|')[0]
    atoms, lvs, _ = build_edge_switch_cell(2 * (width + 1), NCELLS, st, chem=chem)
    return atoms, lvs


def switched_sites_xy(atoms0, atoms1, lvs, rch=1.35):
    """x,y of heavy atoms whose H-count differs from the 0H cell (min-image in x)."""
    apos0, apos1 = np.asarray(atoms0.apos), np.asarray(atoms1.apos)
    hvy0 = np.array([not e.startswith('H') for e in atoms0.enames])
    hvy1 = np.array([not e.startswith('H') for e in atoms1.enames])
    p0, p1 = apos0[hvy0], apos1[hvy1]
    h0, h1 = apos0[~hvy0], apos1[~hvy1]
    Lx = float(lvs[0, 0])

    def n_h(ph, hh):
        if len(hh) == 0:
            return np.zeros(len(ph), int)
        d = ph[:, None, :] - hh[None, :, :]
        d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
        return (np.linalg.norm(d, axis=-1) < rch).sum(axis=1)

    c0, c1 = n_h(p0, h0), n_h(p1, h1)
    dx = p1[:, 0, None] - p0[None, :, 0]
    dx -= Lx * np.round(dx / Lx)
    dy = p1[:, 1, None] - p0[None, :, 1]
    j0 = np.argmin(dx * dx + dy * dy, axis=1)
    return p1[c1 != c0[j0], :2]


def select_window(C, E, kpts, kw, lo, hi, side, tag):
    """States with E in [lo,hi); if empty take the single frontier state on `side` per k."""
    ev = E * HA2EV
    pick = (ev >= lo) & (ev < hi)
    if not np.any(pick):
        pick = np.zeros(ev.shape, dtype=bool)
        for ik in range(ev.shape[0]):
            m = np.where(ev[ik] < hi)[0] if side == 'occ' else np.where(ev[ik] >= lo)[0]
            if m.size == 0:
                raise RuntimeError(f'{tag}: k{ik} no state on {side} side of [{lo:.2f},{hi:.2f}]')
            pick[ik, m[np.argmax(ev[ik, m])] if side == 'occ' else m[np.argmin(ev[ik, m])]] = True
        print(f'  [window] {side} empty; frontier state at each k', flush=True)
    rows, ks, ws, es = [], [], [], []
    for ik, mo in zip(*np.nonzero(pick)):
        rows.append(C[ik, mo])
        ks.append(kpts[ik])
        ws.append(2.0 * kw[ik])
        es.append(ev[ik, mo])
    return np.array(rows), np.array(ks), np.array(ws), np.array(es)


def run_primitive(dftb, chem, width, nkx):
    """Primitive-cell (x1) bands of the pristine 0H ribbon -> clean reference lines.

    Ideal builder geometry at ncells=1 (cutting relaxed subcells is fragile for
    N/O chem); eigenvalues on a dense primitive mesh, mirrored to the full BZ.
    """
    tag = f'{TAG}_{chem}_r{width}_prim'
    src = RIBBON_ROOT / f'{TAG}_{chem}_r{width}' / '0H'
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    from spammm.topology.ribbon_pbc import EDGE_SWITCH, A_CC
    b = MoleculeEditorBackend(a_CC=A_CC)
    base = EDGE_SWITCH[chem][0]
    b.build_zigzag_ribbon(width_chains=2 * (width + 1), length_cells=1,
                          passivation_bottom=base, passivation_top=base, bPeriodicX=True)
    b._sync_sys()
    apos = np.asarray(b.sys.apos, float).copy()
    hvy = np.array([e.split('_')[0] != 'H' for e in b.sys.enames])
    h = apos[hvy, 1].ptp()
    apos[:, 1] -= apos[hvy, 1].min() - 6.0
    lvs = np.array([[2.0 * A_CC * np.cos(np.pi / 6.0), 0.0, 0.0], [0.0, h + 12.0, 0.0], [0.0, 0.0, 20.0]])
    sym = [e.split('_')[0] for e in b.sys.enames]
    wd = WORK / tag
    write_case(wd, sym, apos, lvs, dense_kblock(nkx),
               sk_prefix((src / 'dftb_in.hsd').read_text(), src / 'dftb_in.hsd'))
    cwd0 = os.getcwd()
    os.chdir(wd)
    try:
        energy, C, E, kpts, kw = run_dftb(dftb, 'dftb_in.hsd')
        ef = fermi_ev(wd / 'detailed.out')
    finally:
        os.chdir(cwd0)
    G = 2.0 * np.pi / float(lvs[0, 0])
    kf = np.concatenate([kpts[:, 0], -kpts[:, 0]])            # half-BZ -> full
    ev = np.concatenate([E, E], axis=0) * HA2EV
    return {'pkx': kf * G, 'pev': ev, 'pef': ef, 'pE': energy, 'G': G}


def run_case(dftb, proj, basis, by_name, chem, width, state, nkx, win, dx):
    """DFTB SP on the relaxed gen + LDOS maps + unfolded bands -> cache dict."""
    tag = f'{TAG}_{chem}_r{width}_{state}'
    src = RIBBON_ROOT / f'{TAG}_{chem}_r{width}' / state
    gen, hsd_src = src / 'geom.out.gen', src / 'dftb_in.hsd'
    if not gen.is_file() or not hsd_src.is_file():
        raise RuntimeError(f'missing relaxed input in {src}')
    sym, pos, lat = read_gen(gen)
    a1 = lat[0]
    if max(abs(a1[1]), abs(a1[2]), abs(lat[1, 0]), abs(lat[1, 2])) > 1e-3:
        raise RuntimeError(f'{tag}: non-orthogonal cell\n{lat}')
    wd = WORK / tag
    write_case(wd, sym, pos, lat, dense_kblock(nkx), sk_prefix(hsd_src.read_text(), hsd_src))
    cwd0 = os.getcwd()
    os.chdir(wd)
    try:
        energy, C, E, kpts, kw = run_dftb(dftb, 'dftb_in.hsd')
        ef = fermi_ev(wd / 'detailed.out')
    finally:
        os.chdir(cwd0)
    ispec = np.array([by_name[s] for s in sym], dtype=np.int32)
    atoms_d = proj.prepare_atoms_dftb(pos, ispec, basis)
    norb = int(atoms_d['i0orb'][-1] + atoms_d['norb'][-1])
    if norb != C.shape[2]:
        raise RuntimeError(f'{tag}: basis norb {norb} != DFTB norb {C.shape[2]}')
    heavy = np.array([s != 'H' for s in sym])
    pts, extent, shape = plane(pos, a1, nrep=1, heavy=heavy)
    cell_cart, cell_n = cells_x(a1, n=2)
    rhos, spans, nst = {}, {}, {}
    for side, (lo, hi) in (('occ', (ef - win, ef)), ('unocc', (ef, ef + win))):
        cs, ks, ws, es = select_window(C, E, kpts, kw, lo, hi, side, tag)
        rho, _ = proj.project_bloch_points(pts, atoms_d, cell_cart, cell_n, cs, ks, ws, write_psi=False)
        if not np.isfinite(rho).all() or float(rho.max()) <= 0.0:
            raise RuntimeError(f'{tag}/{side}: LDOS max={float(np.max(rho))}')
        rhos[side] = rho.reshape(shape)
        spans[side] = (float(es.min()), float(es.max()))
        nst[side] = len(es)

    # unfolded bands: C[nk,norb,nband] (get_eigvecs_cplx gives [nk,band,orb])
    atoms_i, lvs = ideal_geom(chem, width, state)
    IDX = subcell_group_indices(np.asarray(atoms_i.apos), np.asarray(lvs), list(atoms_i.enames), NCELLS, strict=False)
    q, W = unfold_spectral_weights(np.ascontiguousarray(C.transpose(0, 2, 1)), kpts[:, 0], IDX, NCELLS)
    ev = E * HA2EV
    nkb = ev.shape[1]
    qq = np.tile(q[:, :, None], (1, 1, nkb)).ravel()          # [nk,nc,nband] -> flat
    ee = np.tile(ev[:, None, :], (1, NCELLS, 1)).ravel()
    ww = W.ravel()
    qq = np.concatenate([qq, (-qq) % 1.0])                    # stored k is half-BZ; mirror by time reversal
    ee = np.concatenate([ee, ee])
    ww = np.concatenate([ww, ww])
    G = 2.0 * np.pi / (float(a1[0]) / NCELLS)                 # primitive G [1/A]
    kx = ((qq % 1.0 + 0.5) % 1.0 - 0.5) * G                   # centered primitive BZ
    keep = (np.abs(ee - ef) < BAND_WIN) & (ww > 0.02)
    a0, _ = ideal_geom(chem, width, '0H')
    sites = switched_sites_xy(a0, atoms_i, lvs)
    at, en = overlay(sym, pos, a1, extent, nrep=1)
    return {
        'rho_occ': rhos['occ'], 'rho_unocc': rhos['unocc'], 'extent': np.array(extent),
        'span_occ': np.array(spans['occ']), 'span_unocc': np.array(spans['unocc']),
        'nst_occ': nst['occ'], 'nst_unocc': nst['unocc'], 'ef': ef, 'E': energy,
        'kx': kx[keep], 'be': ee[keep], 'bw': ww[keep], 'G': G,
        'atoms': at, 'enames': np.array(en), 'sites': sites,
    }


def bands_figure(chem, width, caches, prim, win):
    """One row per state: unfolded weight (red, size) + pristine primitive bands (black lines)."""
    import matplotlib.pyplot as plt
    pkx, pev, pef = prim['pkx'], prim['pev'], prim['pef']
    order = np.argsort(pkx)
    pkx, pev = pkx[order], pev[order]
    fig, axes = plt.subplots(len(STATES), 1, figsize=(9.5, 2.15 * len(STATES)), sharex=True)
    ref = caches['0H']
    rk, re_, rw = ref['kx'], ref['be'], ref['bw']
    sharp0 = rw > 0.5
    for ax, st in zip(axes, STATES):
        c = caches[st]
        for nb in range(pev.shape[1]):
            ax.plot(pkx, pev[:, nb] - pef, c='0.15', lw=0.8, zorder=1)
        ax.scatter(rk[sharp0], re_[sharp0] - ref['ef'], s=4, c='0.55', lw=0, zorder=2)
        ax.scatter(c['kx'], c['be'] - c['ef'], s=90 * c['bw'], c='tab:red',
                   alpha=0.6, lw=0, zorder=3)
        ax.axhspan(-win, 0.0, color='tab:blue', alpha=0.10, lw=0)
        ax.axhspan(0.0, win, color='tab:orange', alpha=0.10, lw=0)
        ax.axhline(0.0, color='0.4', lw=0.7)
        ax.set_ylabel('E−EF [eV]', fontsize=8)
        ax.set_ylim(-BAND_WIN, BAND_WIN)
        ax.tick_params(labelsize=7)
        frac = float((c['bw'] > 0.5).mean())
        ax.set_title(f'{st}   unfolded (red, size=W; sharp {frac * 100:.0f}%)  '
                     f'vs  pristine x1 bands (black) + relaxed 0H unfold (grey)', fontsize=8, loc='left')
    axes[-1].set_xlabel('kx [1/Å]  primitive BZ')
    axes[-1].set_xlim(-prim['G'] / 2, prim['G'] / 2)
    fig.suptitle(f'{TAG}_{chem} r{width}  mio-1-1  x{NCELLS} supercell unfolded -> primitive BZ; '
                 f'shaded = STM windows ±{win} eV', fontsize=10)
    fig.tight_layout()
    path = OUT / f'{TAG}_{chem}_r{width}_bands.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[plot] {path}', flush=True)
    return path


def ldos_figure(chem, width, caches, win):
    """2 rows (occ top, unocc bottom) x 7 states, shared normalization over all
    14 panels: lin = shared max;  sat5 = shared/5;  log = LogNorm 3 decades, plasma."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    tag = f'{TAG}_{chem}_r{width}'
    vmax_all = max(float(np.max(caches[st][f'rho_{side}']))
                   for st in STATES for side in ('occ', 'unocc'))
    if not np.isfinite(vmax_all) or vmax_all <= 0.0:
        raise RuntimeError(f'{tag}: shared vmax={vmax_all}')
    variants = [('lin', 'viridis', 1.0), ('sat5', 'viridis', 0.2), ('log', 'plasma', None)]
    paths = []
    for vname, cmap, vmaxf in variants:
        fig, axes = plt.subplots(2, len(STATES), figsize=(2.7 * len(STATES), 7.8), squeeze=False)
        for irow, side in enumerate(('occ', 'unocc')):
            for ax, st in zip(axes[irow], STATES):
                c = caches[st]
                rho = np.asarray(c[f'rho_{side}'])
                if vname == 'log':
                    ax.imshow(np.clip(rho, vmax_all * 1e-3, None), origin='lower', cmap=cmap,
                              extent=list(c['extent']), interpolation='nearest',
                              norm=LogNorm(vmin=vmax_all * 1e-3, vmax=vmax_all))
                else:
                    ax.imshow(rho, origin='lower', cmap=cmap, extent=list(c['extent']),
                              interpolation='nearest', vmin=0.0, vmax=vmax_all * vmaxf)
                scatter_atoms(ax, c['atoms'], list(c['enames']), s=8)
                if len(c['sites']):
                    ax.plot(c['sites'][:, 0], c['sites'][:, 1], '+', color='lime', ms=5, mew=1.0, zorder=11)
                lo, hi = c[f'span_{side}']
                ax.set_title(f"{st}  {lo:.2f}..{hi:.2f} eV  n={c[f'nst_{side}']}\n"
                             f"max {float(np.max(rho)):.3g}", fontsize=7)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=6)
        for irow, side in enumerate(('occ', 'unocc')):
            lab = {'occ': f'occ EF−{win}..EF', 'unocc': f'unocc EF..EF+{win}'}[side]
            axes[irow, 0].set_ylabel(lab + '   y [Å]', fontsize=8)
        mode = {'lin': 'shared max', 'sat5': 'vmax=shared/5 saturated', 'log': 'log10 3 decades'}[vname]
        fig.suptitle(f'{tag}  mio-1-1  Σ 2w_k|ψ|²  z=+{Z_ABOVE:.1f} Å  '
                     f'[{mode}, shared vmax={vmax_all:.3g}]  (green + = switched sites)', fontsize=10)
        fig.tight_layout()
        path = OUT / f'{tag}_ldos_{vname}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f'[plot] {path}', flush=True)
        paths.append(path)
    return paths


def write_index(figs):
    rows = '\n'.join(f'<p><a href="{f.name}">{f.name}</a><br><img src="{f.name}" width="1100"></p>'
                     for f in sorted(figs))
    (OUT / 'index.html').write_text(
        f'<html><body><h1>ribbon_stm_sys — near-EF TH-LDOS + unfolded bands</h1>'
        f'<p>mio-1-1, nkx={NKX}, window ±{HALF_WIN} eV, z=+{Z_ABOVE} Å, dx={DX} Å. '
        f'bands: red scatter = unfolded weight W (size), black = pristine 0H.</p>{rows}</body></html>')


def main():
    global NCELLS, TAG, STATES, STATE_STRS, WORK, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--chem', default='C,N,O')
    ap.add_argument('--widths', default=','.join(map(str, WIDTHS)))
    ap.add_argument('--states', default=None)
    ap.add_argument('--nkx', type=int, default=NKX)
    ap.add_argument('--win', type=float, default=HALF_WIN)
    ap.add_argument('--dx', type=float, default=DX)
    ap.add_argument('--xscan', action='store_true',
                    help='separation-scan set (xscanv_* dirs, ncells=8, os/ss-dK states)')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.xscan:
        NCELLS, TAG = 8, 'xscanv'
        STATE_STRS = xscan_state_strings(NCELLS)
        STATES = list(STATE_STRS)
        WORK = Path(__file__).parent / 'work' / 'ribbon_stm_xscan'
        OUT = REPO / 'debug' / 'ribbon_stm_xscan'
    if not WFC.is_file():
        raise RuntimeError(f'missing basis {WFC}')
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    chems = args.chem.split(',')
    widths = [int(x) for x in args.widths.split(',')]
    states = args.states.split(',') if args.states else list(STATES)
    figs = []

    if not args.plot_only:
        (WORK / 'basis.hsd').write_text(
            'Basis {\n  Resolution = 0.1\n  <<+ "%s"\n}\n' % os.path.relpath(WFC, WORK))
        species = parse_basis_hsd_ang(str(WORK / 'basis.hsd'))
        by_name = {sp['name']: i for i, sp in enumerate(species)}
        basis = {'species': species}
        proj = DFTBplusGridProjector(verbosity=0)
        proj.load_basis_dftb(basis)
        dftb = DFTBcore()
        for chem in chems:
            for w in widths:
                pnpz = WORK / f'{TAG}_{chem}_r{w}_prim.npz'
                if not pnpz.is_file() or args.force:
                    d = run_primitive(dftb, chem, w, args.nkx * NCELLS)
                    np.savez_compressed(pnpz, **d)
                    print(f'[{TAG}_{chem}_r{w}_prim] E={d["pE"]:.6f} Ha  EF={d["pef"]:.3f} eV', flush=True)
                for st in states:
                    tag = f'{TAG}_{chem}_r{w}_{st}'
                    npz = WORK / f'{tag}.npz'
                    if npz.is_file() and not args.force:
                        continue
                    d = run_case(dftb, proj, basis, by_name, chem, w, st, args.nkx, args.win, args.dx)
                    np.savez_compressed(npz, **d)
                    print(f'[{tag}] E={d["E"]:.6f} Ha  EF={d["ef"]:.3f} eV  '
                          f'occ n={d["nst_occ"]} [{d["span_occ"][0]:.2f},{d["span_occ"][1]:.2f}]  '
                          f'unocc n={d["nst_unocc"]} [{d["span_unocc"][0]:.2f},{d["span_unocc"][1]:.2f}]',
                          flush=True)

    for chem in chems:
        for w in widths:
            caches = {}
            for st in STATES:
                npz = WORK / f'{TAG}_{chem}_r{w}_{st}.npz'
                if not npz.is_file():
                    raise RuntimeError(f'missing cache {npz} — run without --plot-only first')
                caches[st] = dict(np.load(npz, allow_pickle=True))
            prim = dict(np.load(WORK / f'{TAG}_{chem}_r{w}_prim.npz'))
            figs.append(bands_figure(chem, w, caches, prim, args.win))
            figs.extend(ldos_figure(chem, w, caches, args.win))
    write_index(figs)
    print(f'[plot] {OUT / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
