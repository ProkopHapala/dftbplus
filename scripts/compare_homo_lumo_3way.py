#!/usr/bin/env python3
"""compare_homo_lumo_3way.py — 3-way HOMO/LUMO comparison: Rust dense, Rust sparse (TC2), DFTB+ Fortran.

Compares:
1. Eigenvalues (HOMO, LUMO, gap) from all 3 methods.
2. Wavefunction plots (HOMO, LUMO) from all 3 methods side-by-side.

For the sparse path, the TC2 density matrix K is diagonalized as K·S to obtain
natural orbitals (occupation numbers + eigenvectors). The HOMO is the natural
orbital with occupation closest to 2 from below, LUMO closest to 0 from above.

Usage:
    python3 compare_homo_lumo_3way.py debug/graphene_sparse/ --wfc tests/grid/dftb_ptcda/wfc.mio-1-1.hsd
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyBall.OCL.DFTBplusParser import parse_wfc_hsd, evec_to_kernel_coeffs
from pyBall.OCL.Grid import GridProjector, setup_gridprojector_from_dftb, evaluate_mos_on_points

BOHR2ANG = 0.5291772109
ANG2BOHR = 1.0 / BOHR2ANG


def load_wfc_basis(wfc_path):
    wfc_path = Path(wfc_path)
    with open(wfc_path) as f:
        content = f.read()
    if '<<+' in content:
        from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang
        return parse_basis_hsd_ang(wfc_path)
    basis_data = parse_wfc_hsd(str(wfc_path))
    B = BOHR2ANG
    species_list = []
    for sp_name, sp_data in basis_data.items():
        orbitals = []
        for orb in sp_data['orbitals']:
            l = orb['AngularMomentum']
            exps_b = np.asarray(orb['Exponents'], dtype=np.float64)
            coeffs_b = np.asarray(orb['Coefficients'], dtype=np.float64)
            if coeffs_b.ndim == 1:
                coeffs_b = coeffs_b.reshape(1, -1)
            nPow = coeffs_b.shape[0]
            scale_factors = np.array([B ** (l + j) for j in range(nPow)])
            coeffs_scaled = coeffs_b / scale_factors[:, None]
            orbitals.append({'l': l, 'cutoff': orb['Cutoff'] * B,
                            'exponents': exps_b / B, 'coefficients': coeffs_scaled})
        species_list.append({'name': sp_name, 'atomic_number': sp_data['AtomicNumber'],
                            'orbitals': orbitals, 'resolution': 0.04 * B})
    return species_list


def parse_eigenvectors_tsv(path):
    species, coords, evecs, eigs, n_occ = [], [], None, [], 0
    natoms = norb = 0
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('# natoms='):
            parts = line.split()
            natoms = int(parts[1].split('=')[1])
            norb = int(parts[2].split('=')[1])
            n_occ = int(parts[3].split('=')[1])
            evecs = np.zeros((norb, norb), dtype=np.float64)
            i += 1
            if i < len(lines) and lines[i].strip().startswith('# atom_idx'):
                i += 1
            for _ in range(natoms):
                parts = lines[i].strip().split('\t')
                species.append(parts[1])
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                i += 1
        elif line.startswith('# eigenvector') or line.startswith('# eigenvector matrix'):
            i += 1
            if i < len(lines) and lines[i].strip().startswith('mo_idx'):
                i += 1
            # Read until next # section or EOF — sparse files have only 2 MOs
            max_mo = 0
            while i < len(lines) and not lines[i].strip().startswith('#'):
                parts = lines[i].strip().split('\t')
                if len(parts) >= 3:
                    mo = int(parts[0]); orb = int(parts[1]); coeff = float(parts[2])
                    if mo >= evecs.shape[0]:
                        # Need to resize evecs for sparse (only 2 MOs)
                        new_evecs = np.zeros((mo + 1, norb), dtype=np.float64)
                        new_evecs[:evecs.shape[0], :] = evecs
                        evecs = new_evecs
                    evecs[mo, orb] = coeff
                    max_mo = max(max_mo, mo)
                i += 1
        elif line.startswith('# eigenvalues'):
            i += 1
            if i < len(lines) and lines[i].strip().startswith('idx'):
                i += 1
            while i < len(lines) and not lines[i].strip().startswith('#'):
                parts = lines[i].strip().split('\t')
                if len(parts) >= 2:
                    eigs.append(float(parts[1]))
                i += 1
        else:
            i += 1
    return species, np.array(coords, dtype=np.float64), evecs, np.array(eigs), n_occ


def project_mo(evecs_sel, species, coords_ang, species_list_ang, plane='xy', z_offset=0.0,
               npoints=128, margin=3.0):
    """Project MOs onto a 2D grid using OpenCL GridProjector."""
    natoms = len(species)
    unique_species = sorted(set(species))
    species_to_idx = {sp: i for i, sp in enumerate(unique_species)}
    species_per_atom = np.array([species_to_idx[sp] for sp in species], dtype=np.int32)
    coords_bohr = coords_ang * ANG2BOHR
    dftb_data = {'coords_bohr': coords_bohr, 'species_per_atom': species_per_atom,
                 'species_names': unique_species}
    projector, atoms_dict = setup_gridprojector_from_dftb(dftb_data, species_list_ang, verbosity=0)

    if plane == 'xy':
        rmin = float(coords_ang[:, 0].min()) - margin
        rmax = float(coords_ang[:, 0].max()) + margin
        rmin2 = float(coords_ang[:, 1].min()) - margin
        rmax2 = float(coords_ang[:, 1].max()) + margin
        u = np.linspace(rmin, rmax, npoints)
        v = np.linspace(rmin2, rmax2, npoints)
        uu, vv = np.meshgrid(u, v, indexing='ij')
        points = np.column_stack([uu.ravel(), vv.ravel(), np.full(uu.size, z_offset)])
        extent = [rmin, rmax, rmin2, rmax2]
    else:
        raise NotImplementedError(f"plane={plane} not supported in this function")

    nstates = evecs_sel.shape[0]
    mo_indices = list(range(nstates))
    norb_per_atom = np.array([4] * natoms, dtype=np.int32)
    vals = evaluate_mos_on_points(
        projector, mo_indices, points.astype(np.float32),
        evecs_sel, natoms, species_per_atom, unique_species, species_list_ang,
        norb_per_atom, atoms_dict
    )
    return np.array(vals), extent


def main():
    p = argparse.ArgumentParser(description='3-way HOMO/LUMO comparison',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('debug_dir', type=str, help='Directory with *_eigenvectors.tsv and *_ref_eigenvectors.tsv')
    p.add_argument('--wfc', type=str, default='tests/grid/dftb_ptcda/wfc.mio-1-1.hsd')
    p.add_argument('--systems', type=str, default='benzene,coronene,circumcoronene')
    p.add_argument('--npoints', type=int, default=128)
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    debug_dir = Path(args.debug_dir)
    wfc_path = REPO_ROOT / args.wfc if not os.path.isabs(args.wfc) else Path(args.wfc)
    systems = args.systems.split(',')

    print("=== 3-way HOMO/LUMO comparison ===")
    print(f"  systems: {systems}")
    print(f"  wfc: {wfc_path}")

    species_list_ang = load_wfc_basis(wfc_path)
    print(f"  basis species: {[sp['name'] for sp in species_list_ang]}")

    # ─── Eigenvalue comparison table ───
    print("\n─── Eigenvalue comparison ───")
    print(f"{'System':<20} {'Method':<12} {'HOMO (Ha)':<16} {'LUMO (Ha)':<16} {'Gap (Ha)':<14}")
    print("-" * 80)

    all_data = {}
    for sys_name in systems:
        all_data[sys_name] = {}
        # Dense
        dense_path = debug_dir / f"{sys_name}_eigenvectors.tsv"
        if dense_path.exists():
            sp, coords, evecs, eigs, n_occ = parse_eigenvectors_tsv(dense_path)
            homo = eigs[n_occ - 1]
            lumo = eigs[n_occ]
            print(f"{sys_name:<20} {'dense':<12} {homo:<16.10f} {lumo:<16.10f} {lumo-homo:<14.10f}")
            all_data[sys_name]['dense'] = {'sp': sp, 'coords': coords, 'evecs': evecs, 'eigs': eigs, 'n_occ': n_occ}

        # Fortran ref
        ref_path = debug_dir / f"{sys_name}_ref_eigenvectors.tsv"
        if ref_path.exists():
            sp_r, coords_r, evecs_r, eigs_r, n_occ_r = parse_eigenvectors_tsv(ref_path)
            homo_r = eigs_r[n_occ_r - 1]
            lumo_r = eigs_r[n_occ_r]
            print(f"{'':<20} {'fortran':<12} {homo_r:<16.10f} {lumo_r:<16.10f} {lumo_r-homo_r:<14.10f}")
            all_data[sys_name]['fortran'] = {'sp': sp_r, 'coords': coords_r, 'evecs': evecs_r, 'eigs': eigs_r, 'n_occ': n_occ_r}

        # Sparse (Chebyshev filter + Ritz iterative eigensolver)
        sparse_eig_path = debug_dir / f"{sys_name}_sparse_eigenvectors.tsv"
        if sparse_eig_path.exists():
            sp_s, sp_c, sp_evecs, sp_eigs, sp_n_occ = parse_eigenvectors_tsv(sparse_eig_path)
            # sparse_eigenvectors.tsv has only 2 MOs (HOMO, LUMO)
            homo_sp = sp_eigs[0]
            lumo_sp = sp_eigs[1]
            gap_sp = lumo_sp - homo_sp
            print(f"{'':<20} {'sparse':<12} {homo_sp:<16.10f} {lumo_sp:<16.10f} {gap_sp:<14.10f}")
            all_data[sys_name]['sparse'] = {'sp': sp_s, 'coords': sp_c, 'evecs': sp_evecs, 'eigs': sp_eigs, 'n_occ': sp_n_occ}
        print()

    # ─── Wavefunction comparison plots ───
    print("─── Wavefunction comparison plots ───")
    for sys_name in systems:
        data = all_data.get(sys_name, {})
        if 'dense' not in data or 'fortran' not in data:
            print(f"  {sys_name}: missing dense/fortran data, skipping plot")
            continue

        dense = data['dense']
        fortran = data['fortran']
        sparse = data.get('sparse')
        n_occ = dense['n_occ']

        mo_labels = ['HOMO', 'LUMO']
        coords = dense['coords']
        species = dense['sp']

        # Project dense HOMO/LUMO
        print(f"  {sys_name}: projecting dense HOMO/LUMO...")
        evecs_dense_sel = dense['evecs'][[n_occ - 1, n_occ], :]
        vals_dense, extent = project_mo(evecs_dense_sel, species, coords, species_list_ang,
                                         plane='xy', npoints=args.npoints)

        # Project fortran HOMO/LUMO
        print(f"  {sys_name}: projecting fortran HOMO/LUMO...")
        evecs_for_sel = fortran['evecs'][[n_occ - 1, n_occ], :]
        vals_for, _ = project_mo(evecs_for_sel, species, coords, species_list_ang,
                                  plane='xy', npoints=args.npoints)

        # Project sparse HOMO/LUMO (Chebyshev+Ritz)
        vals_sparse = None
        if sparse is not None:
            print(f"  {sys_name}: projecting sparse HOMO/LUMO...")
            # sparse evecs has only 2 columns (HOMO, LUMO)
            evecs_sparse_sel = sparse['evecs'][[0, 1], :]
            vals_sparse, _ = project_mo(evecs_sparse_sel, species, coords, species_list_ang,
                                         plane='xy', npoints=args.npoints)

        # Plot: 3 columns (dense, sparse, fortran) × 2 rows (HOMO, LUMO)
        n_cols = 3 if vals_sparse is not None else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
        if n_cols == 2:
            axes = axes.reshape(2, 2)
        methods = [('dense', vals_dense, dense['eigs'], [n_occ - 1, n_occ])]
        if vals_sparse is not None:
            methods.append(('sparse(Cheb+Ritz)', vals_sparse, sparse['eigs'], [0, 1]))
        methods.append(('fortran', vals_for, fortran['eigs'], [n_occ - 1, n_occ]))
        grid_shape = (args.npoints, args.npoints)

        for row, (mo_label, _) in enumerate(zip(mo_labels, [0, 1])):
            for col, (method_name, vals, eigs, mo_idx_list) in enumerate(methods):
                ax = axes[row, col]
                dat = vals[row].reshape(grid_shape)
                clim = max(abs(dat.min()), abs(dat.max())) or 1e-10
                im = ax.imshow(dat.T, origin='lower', extent=extent, cmap='RdBu_r',
                              vmin=-clim, vmax=clim, interpolation='bilinear')
                ax.scatter(coords[:, 0], coords[:, 1], c='black', s=15, zorder=5,
                          edgecolors='white', linewidths=0.3)
                e_val = eigs[mo_idx_list[row]]
                ax.set_title(f"{method_name} {mo_label}\nE={e_val:.6f} Ha", fontsize=10)
                ax.set_xlabel('x (Å)')
                ax.set_ylabel('y (Å)')
                plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"{sys_name} — HOMO/LUMO wavefunctions: dense vs sparse(Cheb+Ritz) vs DFTB+ Fortran\n"
                     f"(OpenCL GridProjector + STO basis)", fontsize=12)
        fig.tight_layout()
        out = debug_dir / f"{sys_name}_homo_lumo_3way.png"
        fig.savefig(str(out), dpi=args.dpi)
        print(f"  Saved: {out}")

    # ─── Eigenvalue difference table ───
    print("\n─── Eigenvalue differences ───")
    print(f"{'System':<20} {'ΔHOMO dense-for':<16} {'ΔLUMO dense-for':<16} {'ΔHOMO sparse-dense':<20} {'ΔLUMO sparse-dense':<20}")
    print("-" * 100)
    for sys_name in systems:
        data = all_data.get(sys_name, {})
        if 'dense' not in data or 'fortran' not in data:
            continue
        d = data['dense']
        f = data['fortran']
        dh_df = d['eigs'][d['n_occ'] - 1] - f['eigs'][f['n_occ'] - 1]
        dl_df = d['eigs'][d['n_occ']] - f['eigs'][f['n_occ']]
        if 'sparse' in data:
            s = data['sparse']
            dh_sd = s['eigs'][0] - d['eigs'][d['n_occ'] - 1]
            dl_sd = s['eigs'][1] - d['eigs'][d['n_occ']]
            print(f"{sys_name:<20} {dh_df:<16.2e} {dl_df:<16.2e} {dh_sd:<20.2e} {dl_sd:<20.2e}")
        else:
            print(f"{sys_name:<20} {dh_df:<16.2e} {dl_df:<16.2e} {'(no sparse)':<20} {'':<20}")


if __name__ == '__main__':
    main()
