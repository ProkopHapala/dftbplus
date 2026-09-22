#!/usr/bin/env python3
"""Bloch wavefunction and |ψ|² slice of an infinite graphene sheet.

Runs a 2-atom cell through DFTBcore, then projects C(k) with
project_bloch_points (pyBall/OCL/cl/DFTBplusGrid.cl) onto a plane 1 Å
above the sheet. A few points are checked against the same sum on the CPU.
Images go to debug/graphene_bloch/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pyBall.DFTBcore import DFTBcore
from pyBall.OCL.DFTBplusGridProjector import DFTBplusGridProjector
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang
from pyBall.plotUtils import plot_2d_array, plot_complex_hsv, plot_bands_and_maps

A2 = 2.46
Z_SLICE = 1.0  # Angstrom above the nuclear plane (pz vanishes at z=0)
PREF_S = np.float32(0.28209479)
PREF_P = np.float32(0.48860251)
WORK = Path(__file__).parent / 'work' / 'graphene_bloch'
OUT = REPO / 'debug' / 'graphene_bloch'


def graphene_geometry():
    a1 = np.array([A2, 0.0, 0.0])
    a2 = np.array([A2 / 2, A2 * np.sqrt(3) / 2, 0.0])
    pos = np.array([
        [0.0, 0.0, 0.0],
        [A2 / 2, A2 * np.sqrt(3) / 6, 0.0],
    ], dtype=np.float64)
    return pos, a1, a2


def write_geometry(sk_path):
    WORK.mkdir(parents=True, exist_ok=True)
    pos, a1, a2 = graphene_geometry()
    gen = f"""2  S
C
 1  1   {pos[0,0]:.8f}  {pos[0,1]:.8f}  0.00000000
 2  1   {pos[1,0]:.8f}  {pos[1,1]:.8f}  0.00000000
 0.00000000  0.00000000  0.00000000
 {a1[0]:.8f}  {a1[1]:.8f}  0.00000000
 {a2[0]:.8f}  {a2[1]:.8f}  0.00000000
 0.00000000  0.00000000  20.00000000
"""
    (WORK / 'graphene.gen').write_text(gen)
    wfc = Path(__file__).parent / 'dftb_ptcda' / 'wfc.3ob-3-1.hsd'
    (WORK / 'basis.hsd').write_text(
        'Basis {\n  Resolution = 0.1\n  <<+ "%s"\n}\n' % os.path.relpath(wfc, WORK)
    )
    return sk_path


def write_hsd(name, kpts, weights, sk_path):
    lines = [f"  {k[0]:.8f}  {k[1]:.8f}  {k[2]:.8f}   {w:.8e}"
             for k, w in zip(kpts, weights)]
    hsd = f"""Geometry = GenFormat {{
  <<< "graphene.gen"
}}
ParserOptions {{ ParserVersion = 15 }}
Hamiltonian = DFTB {{
  Scc = Yes
  MaxAngularMomentum {{ C = "p" }}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_path}3ob-3-1/"
    Separator = "-"
    Suffix = ".skf"
  }}
  KPointsAndWeights = {{
{chr(10).join(lines)}
  }}
}}
"""
    (WORK / name).write_text(hsd)


def mesh_kpoints(n):
    w = 1.0 / (n * n)
    kpts = [(i / n, j / n, 0.0) for i in range(n) for j in range(n)]
    return np.array(kpts), np.full(len(kpts), w)


def band_path(nseg=24):
    """Γ–M–K–Γ. K = (2/3, 1/3) for this 2-atom cell."""
    verts = [
        ('Γ', np.array([0.0, 0.0, 0.0])),
        ('M', np.array([0.5, 0.0, 0.0])),
        ('K', np.array([2.0 / 3.0, 1.0 / 3.0, 0.0])),
        ('Γ', np.array([0.0, 0.0, 0.0])),
    ]
    ks, ticks = [], []
    for i in range(len(verts) - 1):
        a, b = verts[i][1], verts[i + 1][1]
        ts = np.linspace(0.0, 1.0, nseg, endpoint=True)
        if i > 0:
            ts = ts[1:]
        ticks.append((len(ks), verts[i][0]))
        for t in ts:
            ks.append((1.0 - t) * a + t * b)
    ticks.append((len(ks) - 1, verts[-1][0]))
    return np.array(ks), ticks


def run_dftb(hsd_name):
    dftb = DFTBcore()
    dftb.init(hsd_name)
    dftb.enable_matrix_collection(dm=True, h=True, s=True)
    energy = dftb.run_scf()
    C, E = dftb.get_eigvecs_cplx()
    kpts, kw = dftb.get_kpoints()
    dftb.finalize()
    return energy, C, E, kpts, kw


def kpath_distance(k_frac, a1, a2):
    a3 = np.array([0.0, 0.0, 20.0])
    A = np.column_stack([a1, a2, a3])
    B = 2.0 * np.pi * np.linalg.inv(A).T
    kc = k_frac @ B.T
    step = np.linalg.norm(np.diff(kc, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def states_in_window(C, E, kpts, kw, elo, ehi):
    rows, ks, ws = [], [], []
    for ik in range(len(kpts)):
        for mo in range(E.shape[1]):
            ev = E[ik, mo] * 27.2114
            if elo <= ev < ehi:
                rows.append(C[ik, mo])
                ks.append(kpts[ik])
                ws.append(2.0 * kw[ik])
    if not rows:
        return np.zeros((0, C.shape[2])), np.zeros((0, 3)), np.zeros(0)
    return np.array(rows), np.array(ks), np.array(ws)


def cells(a1, a2, n=3):
    cart, frac = [], []
    for n1 in range(-n, n + 1):
        for n2 in range(-n, n + 1):
            cart.append(n1 * a1 + n2 * a2)
            frac.append((n1, n2, 0.0))
    return np.array(cart), np.array(frac)


def spline_radial(r, y, d2, dr):
    r = np.float32(r)
    n = y.shape[0]
    if r >= np.float32((n - 1) * dr):
        return np.float32(0)
    x = r / np.float32(dr)
    i = int(np.floor(float(x)))
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    t = np.float32(x - np.float32(i))
    a = np.float32(1) - t
    b = t
    h2_6 = np.float32(dr * dr * (1.0 / 6.0))
    corr = ((a * a * a - a) * d2[i] + (b * b * b - b) * d2[i + 1]) * h2_6
    return a * y[i] + b * y[i + 1] + corr


def ylm(l, m, dri, ri):
    if ri < np.float32(1e-10):
        return PREF_S if (l == 0 and m == 0) else np.float32(0)
    if l == 0:
        return PREF_S
    inv = np.float32(1) / ri
    if m == -1:
        return PREF_P * dri[1] * inv
    if m == 0:
        return PREF_P * dri[2] * inv
    if m == 1:
        return PREF_P * dri[0] * inv
    return np.float32(0)


def bloch_ref(points, atoms, packed, dr, cell_cart, cell_n, coeffs, k_frac, weights):
    """Same sum as project_bloch_points, in float32, for a handful of points."""
    pos = atoms['pos'].astype(np.float32)
    rcut = atoms['Rcut'].astype(np.float32)
    typ = atoms['type'].astype(np.int32)
    i0 = atoms['i0orb'].astype(np.int32)
    norb_a = atoms['norb'].astype(np.int32)
    c32 = coeffs.astype(np.complex64)
    npoints = len(points)
    nstate = len(weights)
    psi = np.zeros((nstate, npoints), dtype=np.complex64)
    rho = np.zeros(npoints, dtype=np.float32)
    twopi = np.float32(2 * np.pi)
    for ip, p in enumerate(np.asarray(points, dtype=np.float32)):
        acc = np.float32(0)
        for ist in range(nstate):
            z = np.complex64(0)
            k = k_frac[ist].astype(np.float32)
            for ic in range(len(cell_n)):
                th = twopi * np.float32(np.dot(k, cell_n[ic].astype(np.float32)))
                phase = np.complex64(complex(np.cos(float(th)), -np.sin(float(th))))
                R = cell_cart[ic].astype(np.float32)
                for ia in range(len(pos)):
                    dri = p - (pos[ia] + R)
                    ri2 = np.float32(np.dot(dri, dri))
                    if ri2 >= rcut[ia] * rcut[ia]:
                        continue
                    ri = np.float32(np.sqrt(float(ri2)))
                    orb = int(i0[ia])
                    ish = 0
                    if norb_a[ia] > 0:
                        Rs = spline_radial(ri, packed[typ[ia], ish, :, 0], packed[typ[ia], ish, :, 1], dr)
                        amp = np.float32(Rs) * ylm(0, 0, dri, ri)
                        z += c32[ist, orb] * phase * np.complex64(amp)
                        orb += 1
                        ish += 1
                    if norb_a[ia] > 1:
                        Rp = spline_radial(ri, packed[typ[ia], ish, :, 0], packed[typ[ia], ish, :, 1], dr)
                        for m, take in ((-1, orb < i0[ia] + norb_a[ia]), (0, True), (1, True)):
                            if orb < int(i0[ia] + norb_a[ia]):
                                Y = ylm(1, m, dri, ri)
                                z += c32[ist, orb] * phase * np.complex64(np.float32(Rp) * Y)
                            orb += 1
            psi[ist, ip] = z
            acc += np.float32(weights[ist]) * np.float32(z.real * z.real + z.imag * z.imag)
        rho[ip] = acc
    return rho, psi


def gauge_real(c):
    i = int(np.argmax(np.abs(c)))
    ph = c[i]
    if abs(ph) < 1e-14:
        return c
    return c * np.conj(ph) / abs(ph)


def main():
    sk = os.environ.get('DFTB_SK_PATH', os.path.expanduser('~/SIMULATIONS/dftbplus/slakos/'))
    if not sk.endswith('/'):
        sk += '/'
    write_geometry(sk)
    os.chdir(WORK)
    print(f'[setup] SK prefix {sk}3ob-3-1/')

    kmesh, wmesh = mesh_kpoints(6)
    write_hsd('graphene.hsd', kmesh, wmesh, sk)
    energy, C, E, kpts, kw = run_dftb('graphene.hsd')
    print(f'[dftb] E={energy:.6f} Ha  C{C.shape}  nk={len(kpts)}')

    pos, a1, a2 = graphene_geometry()
    species = parse_basis_hsd_ang(str(WORK / 'basis.hsd'))
    iC = next(i for i, sp in enumerate(species) if sp['atomic_number'] == 6)
    basis = {'species': species}
    proj = DFTBplusGridProjector(verbosity=0)
    packed = proj.load_basis_dftb(basis)
    atoms = proj.prepare_atoms_dftb(pos, np.array([iC, iC], dtype=np.int32), basis)
    cell_cart, cell_n = cells(a1, a2, n=3)
    print(f'[grid] atoms norb={atoms["norb"]} i0={atoms["i0orb"]}  ncell={len(cell_n)}  '
          f'dr={proj.basis_meta["dr"]:.4f} A  rcut={atoms["Rcut"][0]:.3f} A')

    n_occ = 4
    pz = np.asarray(atoms['i0orb'] + 2, dtype=int)  # s, py, pz, px
    ikG = int(np.argmin(np.linalg.norm(kpts, axis=1)))

    def pz_weight(Ck):
        # Ck is (mo, orb). Index the AO axis explicitly; Ck[:, pz] keeps MOs as rows.
        return np.sum(np.abs(Ck[:, pz]) ** 2, axis=1)

    pz_w = pz_weight(C[ikG])
    moG = int(np.argmax(pz_w[:n_occ]))
    print(f'[pick] Gamma k={kpts[ikG]}  mo={moG}  E={E[ikG, moG]*27.2114:.3f} eV  '
          f'pz weight={pz_w[moG]:.3f}')
    print(f'       E[eV]={np.round(E[ikG]*27.2114, 3)}  pz_w={np.round(pz_w, 3)}')
    cG = gauge_real(C[ikG, moG].copy())

    # Occupied π-like states: pz weight above 0.3
    rows, ks, ws = [], [], []
    for ik in range(len(kpts)):
        wpz = pz_weight(C[ik])
        for mo in range(n_occ):
            if wpz[mo] > 0.3:
                rows.append(C[ik, mo])
                ks.append(kpts[ik])
                ws.append(2.0 * kw[ik])
    coeffs = np.array(rows)
    k_sel = np.array(ks)
    w_sel = np.array(ws)
    print(f'[pick] π-occupied states in the |ψ|² sum: {len(rows)}')

    # Parity points: on atoms, between them, one image, and far outside the cutoff
    samples = np.array([
        [pos[0, 0], pos[0, 1], Z_SLICE],
        [pos[1, 0], pos[1, 1], Z_SLICE],
        [A2, A2 * np.sqrt(3) / 6, Z_SLICE],
        [(pos[0, 0] + pos[1, 0]) / 2, (pos[0, 1] + pos[1, 1]) / 2, Z_SLICE],
        [-12.0, -12.0, Z_SLICE],
    ], dtype=np.float64)

    rho_g, psi_g = proj.project_bloch_points(
        samples, atoms, cell_cart, cell_n, cG[None, :], kpts[ikG][None, :], np.array([1.0]),
        write_psi=True)
    rho_c, psi_c = bloch_ref(
        samples, atoms, packed, np.float32(proj.basis_meta['dr']),
        cell_cart, cell_n, cG[None, :], kpts[ikG][None, :], np.array([1.0]))
    dpsi = np.max(np.abs(psi_g[0] - psi_c[0]))
    drho = np.max(np.abs(rho_g - rho_c))
    print(f'[parity Γ π] max|Δψ|={dpsi:.3e}  max|Δ|ψ|²|={drho:.3e}')
    print(f'         ψ_gpu={np.array2string(psi_g[0], precision=4)}')
    print(f'         ψ_cpu={np.array2string(psi_c[0], precision=4)}')
    assert dpsi < 2e-4, f'wavefunction mismatch {dpsi}'
    assert drho < 2e-4, f'density mismatch {drho}'
    assert abs(psi_g[0, -1]) < 1e-6, 'far point should be outside every cutoff'
    assert np.max(np.abs(psi_g[0, :4])) > 1e-3, 'π orbital vanished on the sheet'

    rho_g2, _ = proj.project_bloch_points(
        samples, atoms, cell_cart, cell_n, coeffs, k_sel, w_sel, write_psi=False)
    rho_c2, _ = bloch_ref(
        samples, atoms, packed, np.float32(proj.basis_meta['dr']),
        cell_cart, cell_n, coeffs, k_sel, w_sel)
    dsum = np.max(np.abs(rho_g2 - rho_c2))
    print(f'[parity sum] max|Δρ|={dsum:.3e}  ρ_gpu={np.array2string(rho_g2, precision=4)}')
    assert dsum < 5e-4, f'summed density mismatch {dsum}'
    assert rho_g2[-1] < 1e-8
    assert rho_g2[0] > rho_g2[3] * 0.2

    # Image: two primitive cells, z = 1 Å
    xs = np.linspace(-0.4, 2 * A2 + 0.4, 96)
    ys = np.linspace(-0.4, A2 * np.sqrt(3) + 0.4, 80)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    plane = np.stack([xx, yy, np.full_like(xx, Z_SLICE)], axis=-1).reshape(-1, 3)
    psi_plane_rho, psi_plane = proj.project_bloch_points(
        plane, atoms, cell_cart, cell_n, cG[None, :], kpts[ikG][None, :], np.array([1.0]),
        write_psi=True)
    rho_plane, _ = proj.project_bloch_points(
        plane, atoms, cell_cart, cell_n, coeffs, k_sel, w_sel, write_psi=False)
    ny, nx = len(ys), len(xs)
    re_img = psi_plane[0].real.astype(np.float64).reshape(ny, nx)
    abs_img = psi_plane_rho.astype(np.float64).reshape(ny, nx)
    den_img = rho_plane.astype(np.float64).reshape(ny, nx)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    # overlay every image atom that falls inside the frame
    overlay = []
    for n1 in range(-1, 4):
        for n2 in range(-1, 4):
            shift = n1 * a1 + n2 * a2
            for p in pos:
                q = p + shift
                if xs[0] <= q[0] <= xs[-1] and ys[0] <= q[1] <= ys[-1]:
                    overlay.append(q)
    overlay = np.array(overlay)
    OUT.mkdir(parents=True, exist_ok=True)
    # Γ bonding π is real and in phase on both sublattices: one hue, no in-plane node.
    # Γ π* flips sign between the two sublattices. K is not time-reversal invariant, so arg(ψ) winds.
    mo_star = n_occ + int(np.argmax(pz_w[n_occ:]))
    gap = E[:, n_occ] - E[:, n_occ - 1]
    ik_dirac = int(np.argmin(gap))
    EF = 0.5 * (E[ik_dirac, n_occ - 1] + E[ik_dirac, n_occ]) * 27.2114
    kK = np.array([2.0 / 3.0, 1.0 / 3.0, 0.0])
    ikK = int(np.argmin(np.linalg.norm(kpts - kK, axis=1)))
    moK = int(np.argmin(np.abs(E[ikK] * 27.2114 - EF)))
    print(f'[pick] Γ π* mo={mo_star} E={E[ikG, mo_star]*27.2114:.3f} eV  pz={pz_w[mo_star]:.3f}')
    print(f'[pick] Dirac k={kpts[ik_dirac]} gap={gap[ik_dirac]*27.2114:.3f} eV  EF={EF:.3f} eV')
    print(f'[pick] K-map k={kpts[ikK]} mo={moK} E={E[ikK, moK]*27.2114:.3f} eV')

    def project_one(ik, mo):
        _, psi = proj.project_bloch_points(
            plane, atoms, cell_cart, cell_n,
            C[ik, mo][None, :], kpts[ik][None, :], np.array([1.0]), write_psi=True)
        return psi[0].reshape(ny, nx)

    psi_star = project_one(ikG, mo_star)
    psi_K = project_one(ikK, moK)

    # Bloch check at K: ψ(r+m a1) must equal ψ(r) * exp(-i 2π kx m), so |ψ|² is periodic
    # even though the hue advances by 120° per cell (kx = 2/3).
    cell_wide, n_wide = cells(a1, a2, n=7)
    probe = np.array([[x, y, Z_SLICE]
                      for x in np.linspace(0.4, A2 - 0.4, 4)
                      for y in np.linspace(0.4, A2 * np.sqrt(3) / 2 - 0.4, 4)])
    shifted = np.concatenate([probe + m * a1 for m in range(4)])
    _, psi_shift = proj.project_bloch_points(
        shifted, atoms, cell_wide, n_wide,
        C[ikK, moK][None, :], kpts[ikK][None, :], np.array([1.0]), write_psi=True)
    psi_shift = psi_shift[0].reshape(4, -1)
    kx = kpts[ikK, 0]
    bloch_err, dens_err = [], []
    for m in range(1, 4):
        expect = np.exp(-1j * 2 * np.pi * kx * m) * psi_shift[0]
        bloch_err.append(np.max(np.abs(psi_shift[m] - expect)))
        dens_err.append(np.max(np.abs(np.abs(psi_shift[m])**2 - np.abs(psi_shift[0])**2)))
    print(f'[bloch K] max|ψ(r+m a1) - e^{{-i 2π kx m}} ψ(r)|  m=1,2,3: '
          f'{bloch_err[0]:.3e} {bloch_err[1]:.3e} {bloch_err[2]:.3e}')
    print(f'[bloch K] max||ψ|²(r+m a1)-|ψ|²(r)|  m=1,2,3: '
          f'{dens_err[0]:.3e} {dens_err[1]:.3e} {dens_err[2]:.3e}')
    assert max(bloch_err) < 5e-3, 'K state is not a Bloch wave — cell phase is wrong'
    assert max(dens_err) < 1e-4

    # Continuity across the primitive-cell edge (a jump here would be unphysical).
    y_mid = 0.5 * A2 * np.sqrt(3) / 2
    edge = np.array([[A2 + s, y_mid, Z_SLICE] for s in (-0.05, -0.01, 0.01, 0.05)])
    _, psi_edge = proj.project_bloch_points(
        edge, atoms, cell_wide, n_wide,
        C[ikK, moK][None, :], kpts[ikK][None, :], np.array([1.0]), write_psi=True)
    d_edge = np.abs(np.diff(psi_edge[0]))
    print(f'[bloch K] |Δψ| across x=a1 in steps of 0.04 Å: {np.array2string(d_edge, precision=4)}')
    # Middle step is the one that crosses the cell edge. A seam would make it larger than its neighbors.
    assert d_edge[1] <= 1.5 * max(d_edge[0], d_edge[2]) + 1e-5
    plot_complex_hsv(
        psi_star, extent, overlay,
        f'Γ π*   hue = arg(ψ), brightness = |ψ|   z={Z_SLICE:.1f} Å   {E[ikG, mo_star]*27.2114:.2f} eV',
        str(OUT / 'gamma_pistar_phase.png'))
    plot_complex_hsv(
        psi_K, extent, overlay,
        f'K   hue = arg(ψ), brightness = |ψ|   z={Z_SLICE:.1f} Å   k={np.round(kpts[ikK], 3)}   {E[ikK, moK]*27.2114:.2f} eV',
        str(OUT / 'K_phase.png'))
    plot_2d_array(abs_img, extent, overlay,
                  f'graphene Γ π  |ψ|²   z={Z_SLICE:.1f} Å',
                  str(OUT / 'gamma_pi_abs2.png'), cmap='viridis')
    print(f'[phase] Γ π* |ψ| max={np.abs(psi_star).max():.4f}  '
          f'phase span={np.ptp(np.angle(psi_star[np.abs(psi_star) > 0.2*np.abs(psi_star).max()])):.2f} rad')
    print(f'[phase] K    |ψ| max={np.abs(psi_K).max():.4f}  '
          f'phase span={np.ptp(np.angle(psi_K[np.abs(psi_K) > 0.2*np.abs(psi_K).max()])):.2f} rad')

    # Horizontal cuts through the bands: bottom σ, bonding-π region, Dirac window, empty bands.
    # Each map is scaled on its own, so a narrow window is not hidden by a wider one.
    edges = [-28.0, -16.0, -8.0, -1.0, 28.0]
    colors = ['#1a5276', '#117a65', '#b9770e', '#6c3483']
    windows = []
    for elo, ehi, col in zip(edges[:-1], edges[1:], colors):
        cw, ksw, ww = states_in_window(C, E, kpts, kw, elo, ehi)
        print(f'[window] {elo:.2f} … {ehi:.2f} eV   nstate={len(ww)}')
        if len(ww) == 0:
            continue
        rho_w, _ = proj.project_bloch_points(
            plane, atoms, cell_cart, cell_n, cw, ksw, ww, write_psi=False)
        windows.append({
            'elo': elo, 'ehi': ehi, 'color': col, 'nstate': len(ww),
            'rho': rho_w.reshape(ny, nx),
        })
    assert len(windows) >= 3

    k_path, ticks = band_path(24)
    write_hsd('bands.hsd', k_path, np.full(len(k_path), 1.0 / len(k_path)), sk)
    _, _, Epath, k_got, _ = run_dftb('bands.hsd')
    # DFTB+ may reorder k-points; match each requested point back to the path order.
    order = [int(np.argmin(np.linalg.norm(k_got - k, axis=1))) for k in k_path]
    Epath = Epath[order] * 27.2114
    kdist = kpath_distance(k_path, a1, a2)
    plot_bands_and_maps(
        kdist, Epath, [kdist[i] for i, _ in ticks], [lab for _, lab in ticks],
        windows, extent, overlay, str(OUT / 'bands_windows.png'))
    # Larger patch: 4×4 cells. One K state shows the e^{ikR} hue (period 3 cells).
    # The Fermi-window map sums |ψ|², so those phases cancel and the density is 1-cell periodic.
    nrep = 4
    xsL = np.linspace(-0.3, nrep * A2 + 0.3, 48 * nrep)
    ysL = np.linspace(-0.3, nrep * A2 * np.sqrt(3) / 2 + 0.3, 40 * nrep)
    xxL, yyL = np.meshgrid(xsL, ysL, indexing='xy')
    planeL = np.stack([xxL, yyL, np.full_like(xxL, Z_SLICE)], axis=-1).reshape(-1, 3)
    nyL, nxL = len(ysL), len(xsL)
    extentL = [xsL[0], xsL[-1], ysL[0], ysL[-1]]
    overlayL = []
    for n1 in range(-1, nrep + 1):
        for n2 in range(-1, nrep + 1):
            shift = n1 * a1 + n2 * a2
            for p in pos:
                q = p + shift
                if xsL[0] <= q[0] <= xsL[-1] and ysL[0] <= q[1] <= ysL[-1]:
                    overlayL.append(q)
    overlayL = np.array(overlayL)
    _, psiKL = proj.project_bloch_points(
        planeL, atoms, cell_wide, n_wide,
        C[ikK, moK][None, :], kpts[ikK][None, :], np.array([1.0]), write_psi=True)
    psiKL = psiKL[0].reshape(nyL, nxL)
    plot_2d_array(np.abs(psiKL)**2, extentL, overlayL,
                  f'one K state  |ψ|²  (same in every cell)   {E[ikK, moK]*27.2114:.2f} eV',
                  str(OUT / 'K_abs2_large.png'), cmap='viridis')
    plot_complex_hsv(
        psiKL, extentL, overlayL,
        f'one K state  hue=arg(ψ) advances 120° per cell, repeats every 3   {E[ikK, moK]*27.2114:.2f} eV',
        str(OUT / 'K_phase_large.png'))

    eloF, ehiF = EF - 1.2, EF + 1.2
    cF, kF, wF = states_in_window(C, E, kpts, kw, eloF, ehiF)
    print(f'[fermi] {eloF:.2f} … {ehiF:.2f} eV   nstate={len(wF)}')
    assert len(wF) > 0
    rhoF, _ = proj.project_bloch_points(
        planeL, atoms, cell_wide, n_wide, cF, kF, wF, write_psi=False)
    plot_2d_array(rhoF.reshape(nyL, nxL), extentL, overlayL,
                  f'near EF   Σ 2 w_k |ψ|²   {eloF:.2f} … {ehiF:.2f} eV   {len(wF)} states   z={Z_SLICE:.1f} Å',
                  str(OUT / 'fermi_ldos_large.png'), cmap='viridis')
    print(f'[image] wrote {OUT}/gamma_pistar_phase.png  K_phase_large.png  '
          f'K_abs2_large.png  fermi_ldos_large.png  bands_windows.png')
    assert abs_img.max() > 1e-4
    assert den_img.max() > den_img.min()
    assert np.ptp(np.angle(psi_star[np.abs(psi_star) > 0.25 * np.abs(psi_star).max()])) > 2.0
    print('PASS')


if __name__ == '__main__':
    main()
