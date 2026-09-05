#!/usr/bin/env python3
"""plot_wavefunctions.py — project Rust DFTB MOs onto a 2D grid using pyBall OpenCL GridProjector.

Reads eigenvectors saved by `rhai_save_eigenvectors` (Rust SCC result) and uses
the existing `pyBall/OCL/Grid.py::GridProjector` to evaluate each MO on a 2D
plane (default: molecular xy-plane at z=0). Produces a grid of contour plots
showing HOMO, LUMO, and a few orbitals around the gap.

The STO basis is loaded from a `wfc.mio-1-1.hsd` file (same as waveplot uses).
Orbital ordering in Rust matches DFTB+: [s, py, pz, px] per atom (tesseral
spherical harmonics ordered by magnetic quantum number m).

Usage:
    python3 plot_wavefunctions.py <eigenvectors.tsv> [--wfc <wfc.mio-1-1.hsd>]
        [--plane xy|xz|yz] [--z-offset 0.0] [--npoints 128]
        [--nmo 6] [--mo-range START END] [--margin 3.0] [--out <dir>]

Example:
    python3 scripts/plot_wavefunctions.py debug/graphene_sparse/benzene_eigenvectors.tsv \
        --wfc tests/grid/dftb_ptcda/wfc.mio-1-1.hsd --plane xy --npoints 128 --nmo 6
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
# Insert dftbplus repo root FIRST so our local pyBall/ (with OCL/Grid.py) shadows
# any installed pyBall (e.g. FireCore's, which lacks GridProjector).
sys.path.insert(0, str(REPO_ROOT))

from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang, parse_wfc_hsd, evec_to_kernel_coeffs
from pyBall.OCL.Grid import GridProjector, setup_gridprojector_from_dftb, evaluate_mos_on_points

BOHR2ANG = 0.5291772109
ANG2BOHR = 1.0 / BOHR2ANG


def load_wfc_basis(wfc_path):
    """Load STO basis from a wfc.*.hsd file, converting Bohr → Angstrom.

    Handles both standalone wfc files (no Basis{} wrapper) and waveplot_in.hsd
    files with `<<+ "wfc.mio-1-1.hsd"` include directives.
    """
    wfc_path = Path(wfc_path)
    # If it's a waveplot_in.hsd with include, use parse_basis_hsd_ang
    with open(wfc_path) as f:
        content = f.read()
    if '<<+' in content:
        return parse_basis_hsd_ang(wfc_path)

    # Standalone wfc file — parse directly and convert Bohr → Angstrom
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
            cutoff_b = orb['Cutoff']
            nPow = coeffs_b.shape[0]
            scale_factors = np.array([B ** (l + j) for j in range(nPow)])
            coeffs_scaled = coeffs_b / scale_factors[:, None]
            orbitals.append({
                'l': l,
                'cutoff': cutoff_b * B,
                'exponents': exps_b / B,
                'coefficients': coeffs_scaled,
            })
        species_list.append({
            'name': sp_name,
            'atomic_number': sp_data['AtomicNumber'],
            'orbitals': orbitals,
            'resolution': 0.04 * B,
        })
    return species_list


def parse_eigenvectors_tsv(path):
    """Parse the TSV produced by rhai_save_eigenvectors.

    Returns: species (list[str]), coords_ang (natoms,3), evecs (norb,norb),
             eigs (norb,), n_occ (int).
    """
    species = []
    coords = []
    evecs = None
    eigs = []
    n_occ = 0
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
            # Skip header line starting with '# atom_idx'
            if i < len(lines) and lines[i].strip().startswith('# atom_idx'):
                i += 1
            # Read atom lines
            for _ in range(natoms):
                parts = lines[i].strip().split('\t')
                species.append(parts[1])
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                i += 1
        elif line.startswith('# eigenvector matrix'):
            i += 1
            # Skip header line
            if i < len(lines) and lines[i].strip().startswith('mo_idx'):
                i += 1
            # Read norb*norb lines
            for _ in range(norb * norb):
                parts = lines[i].strip().split('\t')
                mo = int(parts[0])
                orb = int(parts[1])
                coeff = float(parts[2])
                evecs[mo, orb] = coeff  # evecs[mo_idx, orb_idx]
                i += 1
        elif line.startswith('# eigenvalues'):
            i += 1
            if i < len(lines) and lines[i].strip().startswith('idx'):
                i += 1
            for _ in range(norb):
                parts = lines[i].strip().split('\t')
                eigs.append(float(parts[1]))
                i += 1
        else:
            i += 1

    coords_ang = np.array(coords, dtype=np.float64)
    # Rust stores coords in Angstrom (from NanoStructure.positions)
    return species, coords_ang, evecs, np.array(eigs), n_occ


def main():
    p = argparse.ArgumentParser(description='Project Rust DFTB MOs onto 2D grid',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('eigvec_tsv', type=str, help='Path to *_eigenvectors.tsv from Rust')
    p.add_argument('--wfc', type=str, default='tests/grid/dftb_ptcda/wfc.mio-1-1.hsd',
                   help='Path to wfc.mio-1-1.hsd (STO basis for waveplot)')
    p.add_argument('--plane', choices=['xy', 'xz', 'yz'], default='xy',
                   help='2D plane for projection')
    p.add_argument('--z-offset', type=float, default=0.0,
                   help='Fixed coordinate for out-of-plane axis (Angstrom)')
    p.add_argument('--npoints', type=int, default=128, help='Grid resolution')
    p.add_argument('--nmo', type=int, default=6, help='Number of MOs around HOMO')
    p.add_argument('--mo-range', type=int, nargs=2, default=None, metavar=('START', 'END'),
                   help='1-based inclusive MO index range (overrides --nmo)')
    p.add_argument('--margin', type=float, default=3.0, help='Grid margin around molecule (Angstrom)')
    p.add_argument('--out', type=str, default=None, help='Output directory (default: same as TSV)')
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    tsv_path = Path(args.eigvec_tsv)
    assert tsv_path.exists(), f"File not found: {tsv_path}"
    system_name = tsv_path.stem.replace('_eigenvectors', '')

    out_dir = Path(args.out) if args.out else tsv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    wfc_path = REPO_ROOT / args.wfc if not os.path.isabs(args.wfc) else Path(args.wfc)
    assert wfc_path.exists(), f"WFC basis file not found: {wfc_path}"

    print(f"=== Wavefunction projection: {system_name} ===")
    print(f"  eigenvectors: {tsv_path}")
    print(f"  wfc basis: {wfc_path}")
    print(f"  plane: {args.plane}  z-offset: {args.z_offset} Å  npoints: {args.npoints}")

    # 1. Parse eigenvectors + geometry from Rust TSV
    species, coords_ang, evecs, eigs, n_occ = parse_eigenvectors_tsv(tsv_path)
    natoms = len(species)
    norb = evecs.shape[1]
    print(f"  natoms={natoms}  norb={norb}  n_occ={n_occ}")
    print(f"  HOMO idx={n_occ-1}  E={eigs[n_occ-1]:.6f} Ha")
    print(f"  LUMO idx={n_occ}    E={eigs[n_occ]:.6f} Ha")

    # 2. Parse STO basis
    species_list_ang = load_wfc_basis(wfc_path)
    sp_names_in_basis = [sp['name'] for sp in species_list_ang]
    print(f"  basis species: {sp_names_in_basis}")

    # Check all species are in the basis
    for sp in set(species):
        assert sp in sp_names_in_basis, f"Species '{sp}' not in wfc basis ({sp_names_in_basis})"

    # 3. Build species mapping
    unique_species = sorted(set(species))
    species_to_idx = {sp: i for i, sp in enumerate(unique_species)}
    species_per_atom = np.array([species_to_idx[sp] for sp in species], dtype=np.int32)

    # 4. Setup OpenCL projector
    coords_bohr = coords_ang * ANG2BOHR
    dftb_data = {
        'coords_bohr': coords_bohr,
        'species_per_atom': species_per_atom,
        'species_names': unique_species,
    }
    print("  Setting up OpenCL GridProjector...")
    projector, atoms_dict = setup_gridprojector_from_dftb(dftb_data, species_list_ang, verbosity=0)

    # 5. MO selection
    if args.mo_range:
        mo_start = max(1, args.mo_range[0])
        mo_end = min(norb, args.mo_range[1])
    else:
        mo_start = max(1, n_occ - args.nmo // 2)
        mo_end = min(norb, n_occ + args.nmo // 2)
    nstates = mo_end - mo_start + 1
    print(f"  Evaluating MOs {mo_start}–{mo_end}  ({nstates} total)")

    # evecs is (norb, norb) with columns as MOs. We need (nstates, norb) rows.
    evecs_sel = evecs[mo_start - 1:mo_end, :]  # (nstates, norb)

    # 6. Build 2D point grid
    if args.plane == 'xy':
        rmin = float(coords_ang[:, 0].min()) - args.margin
        rmax = float(coords_ang[:, 0].max()) + args.margin
        rmin2 = float(coords_ang[:, 1].min()) - args.margin
        rmax2 = float(coords_ang[:, 1].max()) + args.margin
        u = np.linspace(rmin, rmax, args.npoints)
        v = np.linspace(rmin2, rmax2, args.npoints)
        uu, vv = np.meshgrid(u, v, indexing='ij')
        ww = np.full_like(uu, args.z_offset)
        points = np.column_stack([uu.ravel(), vv.ravel(), ww.ravel()])
        ax_labels = ('x (Å)', 'y (Å)')
        extent = [rmin, rmax, rmin2, rmax2]
    elif args.plane == 'xz':
        rmin = float(coords_ang[:, 0].min()) - args.margin
        rmax = float(coords_ang[:, 0].max()) + args.margin
        rmin2 = float(coords_ang[:, 2].min()) - args.margin
        rmax2 = float(coords_ang[:, 2].max()) + args.margin
        u = np.linspace(rmin, rmax, args.npoints)
        v = np.linspace(rmin2, rmax2, args.npoints)
        uu, vv = np.meshgrid(u, v, indexing='ij')
        points = np.column_stack([uu.ravel(), np.full(uu.size, args.z_offset), vv.ravel()])
        ax_labels = ('x (Å)', 'z (Å)')
        extent = [rmin, rmax, rmin2, rmax2]
    else:  # yz
        rmin = float(coords_ang[:, 1].min()) - args.margin
        rmax = float(coords_ang[:, 1].max()) + args.margin
        rmin2 = float(coords_ang[:, 2].min()) - args.margin
        rmax2 = float(coords_ang[:, 2].max()) + args.margin
        u = np.linspace(rmin, rmax, args.npoints)
        v = np.linspace(rmin2, rmax2, args.npoints)
        uu, vv = np.meshgrid(u, v, indexing='ij')
        points = np.column_stack([np.full(uu.size, args.z_offset), uu.ravel(), vv.ravel()])
        ax_labels = ('y (Å)', 'z (Å)')
        extent = [rmin, rmax, rmin2, rmax2]

    print(f"  Grid: {args.npoints}×{args.npoints} = {len(points)} points")

    # 7. Evaluate MOs on the grid using OpenCL
    mo_indices = list(range(nstates))
    print("  Evaluating MOs on grid (OpenCL)...")
    vals_list = evaluate_mos_on_points(
        projector, mo_indices, points.astype(np.float32),
        evecs_sel, natoms, species_per_atom, unique_species, species_list_ang,
        # norb_per_atom and atoms_dict are computed inside
        np.array([4] * natoms, dtype=np.int32),  # all C → 4 orbitals (s+p)
        atoms_dict
    )
    vals = np.array(vals_list)  # (nstates, npoints)
    print(f"  Done. max|ψ|={np.abs(vals).max():.4e}")

    # 8. Plot
    n_cols = min(4, nstates)
    n_rows = (nstates + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    grid_shape = (args.npoints, args.npoints)

    for idx in range(nstates):
        ax = axes[idx]
        mo_abs = mo_start + idx
        dat = vals[idx].reshape(grid_shape)
        clim = max(abs(dat.min()), abs(dat.max())) or 1e-10
        im = ax.imshow(dat.T, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=-clim, vmax=clim, interpolation='bilinear')

        # Mark atom positions
        if args.plane == 'xy':
            ax_pts = coords_ang[:, 0]
            ay_pts = coords_ang[:, 1]
        elif args.plane == 'xz':
            ax_pts = coords_ang[:, 0]
            ay_pts = coords_ang[:, 2]
        else:
            ax_pts = coords_ang[:, 1]
            ay_pts = coords_ang[:, 2]
        ax.scatter(ax_pts, ay_pts, c='black', s=20, zorder=5, edgecolors='white', linewidths=0.5)

        tag = " HOMO" if mo_abs == n_occ else (" LUMO" if mo_abs == n_occ + 1 else "")
        ax.set_title(f"MO{mo_abs}{tag}\nE={eigs[mo_abs - 1]:.4f} Ha", fontsize=9)
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        plt.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[nstates:]:
        ax.set_visible(False)

    fig.suptitle(f"{system_name} — wavefunctions ({args.plane} plane, z={args.z_offset:.2f} Å)\n"
                 f"Rust DFTB SCC → OpenCL GridProjector", fontsize=11)
    fig.tight_layout()
    out = out_dir / f"{system_name}_wavefunctions_{args.plane}_z{args.z_offset:.2f}_MO{mo_start}-{mo_end}.png"
    fig.savefig(str(out), dpi=args.dpi)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
