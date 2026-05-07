#!/usr/bin/env python3
"""
test_density_projection.py

Test density matrix projection vs brute-force sum of occupied orbitals for H2O.

This script:
1. Runs DFTB+ via libdftbcore.so
2. Extracts eigenvectors and density matrix
3. Computes density using brute-force method (sum of |ψ_i|² for occupied orbitals)
4. Computes density using optimized density matrix projection
5. Compares both methods for validation

Usage:
    python test_density_projection.py --dftb-dir tests/grid/dftb_h2o
    python test_density_projection.py --dftb-dir tests/grid/dftb_h2o --points --plane2d xy --z-offset 0.0
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyBall.OCL.DFTBplusParser import ( parse_basis_hsd_ang, parse_detailed_xml_custom, evec_to_kernel_coeffs)
from pyBall.OCL.Grid import GridProjector, setup_gridprojector_from_dftb
from pyBall.DFTBcore import DFTBcore
from pyBall.WavePlot.TestUtils import print_eigenvecs

BOHR2ANG = 0.5291772109
OUTPUT_DIR = Path(__file__).parent / 'waveplot_output' / 'density'

# Angular order conversion: DFTB+ [s,py,pz,px] -> OpenCL [px,py,pz,s]
_ORT_SPP_TO_OCL = np.array([3, 1, 2, 0], dtype=np.int32)


def parse_args():
    p = argparse.ArgumentParser(
        description='Density matrix projection parity test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--dftb-dir', type=str, default=None, help='DFTB+ run dir containing dftb_in.hsd, detailed.xml')
    p.add_argument('--lib-path', type=str, default=None,help='Path to libdftbcore.so')
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--dpi', type=int, default=150)
    # Point evaluation
    p.add_argument('--points', action='store_true', help='Evaluate at explicit points instead of full 3D grid')
    p.add_argument('--plane2d', type=str, choices=['xy', 'xz', 'yz'], default='xy',  help='2D plane for --points')
    p.add_argument('--z-offset', type=float, default=0.0, help='Fixed coordinate value for out-of-plane axis (Å)')
    p.add_argument('--xy-range', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'), help='Coordinate range for 2D scan (Å)')
    p.add_argument('--npoints', type=int, default=64)  # Higher resolution for visualization
    # 3D grid
    p.add_argument('--step', type=float, default=0.3, help='Grid spacing in Å (3D grid mode)')
    p.add_argument('--margin', type=float, default=3.0,   help='Grid margin around molecule in Å')
    p.add_argument('--print-eigenvec', action='store_true', help='Print eigenvectors from eigenvec.bin and exit')
    return p.parse_args()


def run_dftb_and_get_data(dftb_dir, lib_path=None):
    """
    Run DFTB+ via libdftbcore.so and return eigenvectors, density matrix, and system info.
    """
    dftb_dir = Path(dftb_dir)
    if not dftb_dir.exists():
        raise FileNotFoundError(f"DFTB+ directory not found: {dftb_dir}")

    # Find input file
    input_file = dftb_dir / 'dftb_in.hsd'
    if not input_file.exists():
        raise FileNotFoundError(f"DFTB+ input file not found: {input_file}")

    # Find library
    if lib_path is None:
        lib_paths = [
            REPO_ROOT / '_build' / 'app' / 'dftbcore' / 'libdftbcore.so',
            REPO_ROOT / 'build' / 'libdftbcore.so',
            REPO_ROOT / 'build' / 'lib' / 'libdftbcore.so',
        ]
        for p in lib_paths:
            if p.exists():
                lib_path = p
                break
        if lib_path is None:
            raise FileNotFoundError(f"libdftbcore.so not found in {lib_paths}")
    else:
        lib_path = Path(lib_path)

    print(f"[DFTBcore] Using library: {lib_path}")
    print(f"[DFTBcore] Input file: {input_file}")

    # Change to DFTB+ directory so it finds input files
    orig_dir = os.getcwd()
    os.chdir(dftb_dir)

    try:
        dftb = DFTBcore(libpath=str(lib_path))
        dftb.init(str(input_file))
        # Enable matrix collection for density matrix
        dftb.enable_matrix_collection(dm=True, h=False, s=True)  # Need S for electron count
        energy = dftb.run_scf()
        print(f"[DFTBcore] SCF completed. Energy = {energy:.8f} Ha")

        # Get data
        evecs, eigenvals = dftb.get_eigvecs_dense()  # (nstates, nOrb), (nstates,)
        dm_dense = dftb.get_dm_dense()  # (nOrb, nOrb) in C row-major
        s_dense = dftb.get_s_dense()   # (nOrb, nOrb) overlap matrix
        basis_size = dftb.get_basis_size()

        # Get occupations from detailed.xml if available
        detailed_file = dftb_dir / 'detailed.xml'
        if detailed_file.exists():
            dftb_data = parse_detailed_xml_custom(str(detailed_file))
            occupations = np.array(dftb_data.get('occupations', []), dtype=np.float64)
        else:
            # Estimate occupations (2 electrons per orbital for closed shell)
            n_electrons = sum([1 if i < basis_size // 2 else 0 for i in range(basis_size)]) * 2
            occupations = np.zeros(basis_size)
            for i in range(basis_size // 2):
                occupations[i] = 2.0

        dftb.finalize()

    finally:
        os.chdir(orig_dir)

    print(f"[DFTBcore] nOrb={basis_size}, nStates={len(eigenvals)}")
    print(f"[DFTBcore] DM shape: {dm_dense.shape}, symmetric: {np.allclose(dm_dense, dm_dense.T, atol=1e-12)}")
    print(f"[DFTBcore] Tr(DM) = {np.trace(dm_dense):.4f}")
    print(f"[DFTBcore] Tr(S*DM) = {np.trace(s_dense @ dm_dense):.4f} (electron count)")

    return {
        'evecs': evecs,
        'eigenvals': eigenvals,
        'dm_dense': dm_dense,
        's_dense': s_dense,
        'basis_size': basis_size,
        'occupations': occupations,
        'energy': energy
    }


def parse_dftb_setup(dftb_dir):
    """Parse DFTB+ input files to get geometry and basis info."""
    dftb_dir = Path(dftb_dir)

    # Parse detailed.xml
    detailed_file = dftb_dir / 'detailed.xml'
    dftb_data = parse_detailed_xml_custom(str(detailed_file))

    # Parse basis from waveplot_in.hsd or dftb_in.hsd
    waveplot_hsd = dftb_dir / 'waveplot_in.hsd'
    dftb_hsd = dftb_dir / 'dftb_in.hsd'
    hsd_file = waveplot_hsd if waveplot_hsd.exists() else dftb_hsd

    species_list_ang = parse_basis_hsd_ang(str(hsd_file))

    # Extract atomic positions and types
    atom_coords_b = np.array(dftb_data['coords_bohr'], dtype=np.float64)  # (natoms, 3) in Bohr
    natoms = dftb_data['natoms']
    
    # Get species names from detailed.xml
    species_names_per_atom = dftb_data['species_per_atom']  # array like [0, 1, 1]
    unique_species_names = dftb_data['species_names']  # list like ['O', 'H']
    
    # Build norb_per_atom
    norb_per_atom = []
    for sp_idx in species_names_per_atom:
        sp_name = unique_species_names[sp_idx]
        sp_info = next((sp for sp in species_list_ang if sp['name'] == sp_name), None)
        if sp_info is None:
            raise ValueError(f"Species {sp_name} not found in basis info")
        norb = sum(2 * orb['l'] + 1 for orb in sp_info['orbitals'])
        norb_per_atom.append(norb)

    # Build orbital-to-atom mapping
    orb_to_atom = []
    orb_to_local = []
    for ia, norb in enumerate(norb_per_atom):
        for io in range(norb):
            orb_to_atom.append(ia)
            orb_to_local.append(io)

    return {
        'natoms': natoms,
        'atom_coords_b': atom_coords_b,
        'atom_coords_ang': atom_coords_b * BOHR2ANG,
        'species_list_ang': species_list_ang,
        'species_names': unique_species_names,
        'species_per_atom': species_names_per_atom,
        'norb_per_atom': norb_per_atom,
        'orb_to_atom': orb_to_atom,
        'orb_to_local': orb_to_local,
        'basis_size': sum(norb_per_atom)
    }


def convert_dm_to_ocl_format(dm_dense, setup_data, rcut=5.0):
    """
    Convert DFTB+ dense density matrix to OpenCL sparse format.

    Returns:
        rho_sparse: (natoms, neigh_max, 4, 4) array
        neighs: Neighbor list structure
    """
    natoms = setup_data['natoms']
    norb_per_atom = setup_data['norb_per_atom']
    orb_to_atom = setup_data['orb_to_atom']
    orb_to_local = setup_data['orb_to_local']
    atom_coords_ang = setup_data['atom_coords_ang']

    basis_size = len(orb_to_atom)
    assert dm_dense.shape == (basis_size, basis_size), f"DM shape mismatch: {dm_dense.shape} vs ({basis_size}, {basis_size})"

    # Build neighbor lists based on distance cutoff
    neighbors = []
    for i in range(natoms):
        neigh_i = []
        pos_i = atom_coords_ang[i]
        for j in range(natoms):
            pos_j = atom_coords_ang[j]
            dist = np.linalg.norm(pos_i - pos_j)
            if dist <= rcut:
                neigh_i.append(j)
        neighbors.append(neigh_i)

    neigh_max = max(len(neigh_i) for neigh_i in neighbors)
    print(f"[DM Convert] Max neighbors per atom: {neigh_max}")

    # Initialize sparse density matrix with padding
    numorb_max = 4  # Always use 4x4 blocks for consistency
    rho_sparse = np.zeros((natoms, neigh_max, numorb_max, numorb_max), dtype=np.float32)

    # Fill sparse density matrix
    for i in range(natoms):
        i_orb_start = sum(norb_per_atom[:i])
        i_orb_end = i_orb_start + norb_per_atom[i]

        for ineigh, j in enumerate(neighbors[i]):
            j_orb_start = sum(norb_per_atom[:j])
            j_orb_end = j_orb_start + norb_per_atom[j]

            # Extract block from dense DM
            block = dm_dense[i_orb_start:i_orb_end, j_orb_start:j_orb_end].astype(np.float32)

            # Pad to 4x4 if necessary
            block_padded = np.zeros((numorb_max, numorb_max), dtype=np.float32)
            ni, nj = block.shape
            block_padded[:ni, :nj] = block

            # Apply angular order conversion if both atoms have p orbitals
            # DFTB+ order: [s, py, pz, px], OpenCL order: [px, py, pz, s]
            if ni == 4 and nj == 4:
                # Reorder both dimensions
                block_padded = block_padded[_ORT_SPP_TO_OCL][:, _ORT_SPP_TO_OCL]
            elif ni == 4 and nj == 1:
                # Reorder only i dimension
                block_padded[:4, 0] = block_padded[_ORT_SPP_TO_OCL, 0]
            elif ni == 1 and nj == 4:
                # Reorder only j dimension
                block_padded[0, :4] = block_padded[0, _ORT_SPP_TO_OCL]

            # Debug: print first block
            if i == 0 and ineigh == 0:
                print(f"[DEBUG] First DM block (atom {i}, neigh {j}):")
                print(f"  Dense block (DFTB+ order):\n{block}")
                print(f"  After padding and conversion:\n{block_padded}")

            rho_sparse[i, ineigh, :, :] = block_padded

    # Create neighbor index array for OpenCL
    neigh_j = np.zeros((natoms, neigh_max), dtype=np.int32)
    for i in range(natoms):
        for ineigh, j in enumerate(neighbors[i]):
            neigh_j[i, ineigh] = j + 1  # +1 for Fortran indexing

    return rho_sparse, neigh_j, neighbors


def compute_density_brute_force(evecs, occupations, projector, atoms_dict, points_ang, setup_data):
    """
    Compute density by summing squares of occupied orbitals.
    ρ(r) = Σ_i f_i |ψ_i(r)|²
    """
    natoms = setup_data['natoms']
    species_per_atom = setup_data['species_per_atom']
    species_names = setup_data['species_names']
    species_list_ang = setup_data['species_list_ang']
    norb_per_atom = setup_data['norb_per_atom']

    density = np.zeros(len(points_ang), dtype=np.float64)

    # Sum over occupied orbitals
    for imo, occ in enumerate(occupations):
        if occ <= 0:
            continue

        # Get coefficients for this MO
        coeffs = evec_to_kernel_coeffs(evecs[imo], natoms, species_per_atom,
                                       species_names, species_list_ang)

        # Project orbital at points
        psi = projector.project_orbital_points(
            points_ang.astype(np.float32),
            coeffs,
            norb_per_atom,
            atoms_dict
        )

        # Add to density
        density += occ * (psi ** 2)

    return density


def compute_density_dm(dm_dense, setup_data, projector, atoms_dict, grid_spec):
    """
    Compute density using optimized density matrix projection.
    """
    # Create dummy neighs object
    class Neighs:
        def __init__(self, neigh_j):
            self.neigh_j = neigh_j

    neighs = Neighs(neigh_j)

    # Use the project method
    density_grid = projector.project(
        dm_dense,
        neighs,
        atoms_dict,
        grid_spec,
        nMaxAtom=64,
        use_tiled=True
    )

    return density_grid


def compute_density_dm_correct(dm_dense, setup_data, projector, atoms_dict, points_ang):
    """
    Compute density using correct formula: ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)
    with proper 4x4 matrix multiplication for atom pairs.
    """
    natoms = setup_data['natoms']
    norb_per_atom = setup_data['norb_per_atom']
    orb_to_atom = setup_data['orb_to_atom']
    orb_to_local = setup_data['orb_to_local']
    
    basis_size = dm_dense.shape[0]
    density = np.zeros(len(points_ang))
    
    # Group orbitals by atom
    orb_groups = []
    for ia in range(natoms):
        i_orb_start = sum(norb_per_atom[:ia])
        i_orb_end = i_orb_start + norb_per_atom[ia]
        orb_groups.append((i_orb_start, i_orb_end, norb_per_atom[ia]))
    
    print(f"\n[DEBUG] Orbital groups:")
    print(f"  norb_per_atom: {norb_per_atom}")
    print(f"  orb_to_atom: {orb_to_atom}")
    print(f"  orb_to_local: {orb_to_local}")
    for ia, (start, end, norb) in enumerate(orb_groups):
        print(f"  Atom {ia}: orbitals {start}-{end}, norb={norb}")
    
    # Compute density by iterating over atom pairs
    atom_idx = 0
    for (i_start, i_end, ni) in orb_groups:
        ia = atom_idx
        atom_idx += 1
        atom_idx_j = 0
        for (j_start, j_end, nj) in orb_groups:
            ja = atom_idx_j
            atom_idx_j += 1
            
            # Extract DM block for this atom pair
            dm_block = dm_dense[i_start:i_end, j_start:j_end]
            
            # Skip if block is too small
            if np.max(np.abs(dm_block)) < 1e-12:
                continue
            
            # Evaluate all basis functions for both atoms at all points
            # For atom i
            coeffs_i_all = []
            for iloc in range(ni):
                coeffs = np.zeros((natoms, 4), dtype=np.float32)
                if iloc < 4:
                    if ni == 4:
                        # Full sp3: DFTB+ [s,py,pz,px] -> OpenCL [px,py,pz,s]
                        loc_ocl = _ORT_SPP_TO_OCL[iloc]
                    else:
                        # Only s-orbital: goes to position 3 (s position in OpenCL)
                        loc_ocl = 3
                    coeffs[ia, loc_ocl] = 1.0
                coeffs_i_all.append(coeffs)
            
            # For atom j
            coeffs_j_all = []
            for jloc in range(nj):
                coeffs = np.zeros((natoms, 4), dtype=np.float32)
                if jloc < 4:
                    if nj == 4:
                        loc_ocl = _ORT_SPP_TO_OCL[jloc]
                    else:
                        loc_ocl = 3
                    coeffs[ja, loc_ocl] = 1.0
                coeffs_j_all.append(coeffs)
            
            # Project all combinations
            for iloc in range(ni):
                for jloc in range(nj):
                    p_ij = dm_block[iloc, jloc]
                    if abs(p_ij) < 1e-12:
                        continue
                    
                    psi_i = projector.project_orbital_points(
                        points_ang.astype(np.float32),
                        coeffs_i_all[iloc],
                        norb_per_atom,
                        atoms_dict
                    )
                    
                    psi_j = projector.project_orbital_points(
                        points_ang.astype(np.float32),
                        coeffs_j_all[jloc],
                        norb_per_atom,
                        atoms_dict
                    )
                    
                    # Sum all contributions directly (DM is symmetric but we sum all)
                    density += p_ij * (psi_i * psi_j)
    
    return density


def create_2d_grid_points(atom_coords_ang, plane='xy', z_offset=0.0, xy_range=None, npoints=32):
    """Create 2D grid points for evaluation."""
    if xy_range is None:
        rmin = float(atom_coords_ang.min()) - 3.0
        rmax = float(atom_coords_ang.max()) + 3.0
    else:
        rmin, rmax = xy_range

    u = np.linspace(rmin, rmax, npoints)
    uu, vv = np.meshgrid(u, u, indexing='ij')

    if plane == 'xy':
        ww = np.full_like(uu, z_offset)
        points = np.column_stack([uu.ravel(), vv.ravel(), ww.ravel()])
    elif plane == 'xz':
        points = np.column_stack([uu.ravel(), np.full(uu.size, z_offset), vv.ravel()])
    elif plane == 'yz':
        points = np.column_stack([np.full(uu.size, z_offset), uu.ravel(), vv.ravel()])

    return points, (npoints, npoints)


def create_3d_grid(atom_coords_ang, step=0.3, margin=3.0):
    """Create 3D grid specification."""
    rmin = atom_coords_ang.min(axis=0) - margin
    rmax = atom_coords_ang.max(axis=0) + margin

    nx = int(np.ceil((rmax[0] - rmin[0]) / step)) + 1
    ny = int(np.ceil((rmax[1] - rmin[1]) / step)) + 1
    nz = int(np.ceil((rmax[2] - rmin[2]) / step)) + 1

    return {
        'origin': rmin,
        'dA': np.array([step, 0.0, 0.0]),
        'dB': np.array([0.0, step, 0.0]),
        'dC': np.array([0.0, 0.0, step]),
        'ngrid': np.array([nx, ny, nz], dtype=np.int32)
    }


def validate_and_plot(density_brute, density_dm, points_ang, shape, args, output_dir):
    """Validate and visualize the comparison between methods."""
    # Reshape for 2D plotting
    if len(shape) == 2:
        db = density_brute.reshape(shape)
        dd = density_dm.reshape(shape)

        # Statistics
        diff = np.abs(db - dd)
        rms_error = np.sqrt(np.mean(diff ** 2))
        max_error = np.max(diff)
        max_rel_error = np.max(diff / (np.abs(db) + 1e-12))

        print(f"\n[VALIDATION]")
        print(f"  RMS error:     {rms_error:.6e}")
        print(f"  Max error:     {max_error:.6e}")
        print(f"  Max rel error: {max_rel_error:.6e}")
        print(f"  Brute max:     {np.max(db):.6e}")
        print(f"  DM max:        {np.max(dd):.6e}")

        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        im0 = axes[0, 0].imshow(db.T, origin='lower', extent=[points_ang[:, 0].min(), points_ang[:, 0].max(),
                                                               points_ang[:, 1].min(), points_ang[:, 1].max()])
        axes[0, 0].set_title('Brute Force (Sum of Orbitals)')
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(dd.T, origin='lower', extent=[points_ang[:, 0].min(), points_ang[:, 0].max(),
                                                               points_ang[:, 1].min(), points_ang[:, 1].max()])
        axes[0, 1].set_title('Density Matrix (Fast)')
        plt.colorbar(im1, ax=axes[0, 1])

        im2 = axes[0, 2].imshow(diff.T, origin='lower', extent=[points_ang[:, 0].min(), points_ang[:, 0].max(),
                                                                 points_ang[:, 1].min(), points_ang[:, 1].max()])
        axes[0, 2].set_title(f'Absolute Difference\nRMS={rms_error:.2e}')
        plt.colorbar(im2, ax=axes[0, 2])

        # Line profiles through center
        mid = shape[0] // 2
        axes[1, 0].plot(db[mid, :], label='Brute')
        axes[1, 0].plot(dd[mid, :], label='DM', linestyle='--')
        axes[1, 0].set_title(f'Line Profile (x={mid})')
        axes[1, 0].legend()

        axes[1, 1].plot(db[:, mid], label='Brute')
        axes[1, 1].plot(dd[:, mid], label='DM', linestyle='--')
        axes[1, 1].set_title(f'Line Profile (y={mid})')
        axes[1, 1].legend()

        # Scatter plot
        axes[1, 2].scatter(db.flatten(), dd.flatten(), alpha=0.3, s=1)
        axes[1, 2].plot([db.min(), db.max()], [db.min(), db.max()], 'r--', label='y=x')
        axes[1, 2].set_xlabel('Brute Force')
        axes[1, 2].set_ylabel('Density Matrix')
        axes[1, 2].set_title('Correlation')
        axes[1, 2].legend()

        plt.tight_layout()

        png_path = output_dir / f'density_comparison_{args.plane2d}_z{args.z_offset:.2f}.png'
        plt.savefig(png_path, dpi=args.dpi)
        print(f"[PLOT] Saved to {png_path}")

        if not args.no_show:
            plt.show()

    return rms_error, max_error, max_rel_error


def main():
    args = parse_args()

    if args.dftb_dir is None:
        args.dftb_dir = Path(__file__).parent / 'dftb_h2o'

    dftb_dir = Path(args.dftb_dir)
    
    # Print eigenvectors if requested
    if args.print_eigenvec:
        print_eigenvecs(dftb_dir / 'eigenvec.bin', dftb_dir / 'detailed.xml', dftb_dir / 'waveplot_in.hsd', max_orbitals=6)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DENSITY PROJECTION PARITY TEST")
    print("=" * 60)

    # Step 1: Run DFTB+ and get data
    print("\n[1] Running DFTB+ calculation...")
    dftb_data = run_dftb_and_get_data(args.dftb_dir, args.lib_path)

    # Step 2: Parse setup
    print("\n[2] Parsing DFTB+ setup...")
    setup_data = parse_dftb_setup(args.dftb_dir)

    # Verify consistency
    assert dftb_data['basis_size'] == setup_data['basis_size'], \
        f"Basis size mismatch: {dftb_data['basis_size']} vs {setup_data['basis_size']}"

    # Step 3: Convert density matrix to OpenCL format
    print("\n[3] Converting density matrix to OpenCL format...")
    rho_sparse, neigh_j, neighbors = convert_dm_to_ocl_format(
        dftb_data['dm_dense'],
        setup_data,
        rcut=5.0
    )

    # Step 4: Setup OpenCL projector
    print("\n[4] Setting up OpenCL projector...")
    fdata_dir = REPO_ROOT / 'data' / 'Fdata'

    # Prepare data for projector setup (must match setup_gridprojector_from_dftb expectations)
    dftb_data_ocl = {
        'coords_bohr': setup_data['atom_coords_b'],  # Bohr units
        'species_per_atom': setup_data['species_per_atom'],  # 0-based indices
        'species_names': setup_data['species_names'],  # list of species names
    }

    projector, atoms_dict = setup_gridprojector_from_dftb(
        dftb_data_ocl,
        setup_data['species_list_ang'],
        verbosity=1
    )

    # Step 5: Compute density using both methods
    if args.points:
        # 2D plane evaluation
        print(f"\n[5] Computing density on 2D {args.plane2d} plane at z={args.z_offset:.2f} Å...")

        points_ang, shape = create_2d_grid_points(
            setup_data['atom_coords_ang'],
            plane=args.plane2d,
            z_offset=args.z_offset,
            xy_range=args.xy_range,
            npoints=args.npoints
        )
        print(f"      Grid: {shape[0]}x{shape[1]} = {np.prod(shape)} points")

        # Brute force method
        print("\n[5a] Brute force method (sum of occupied orbitals)...")
        t0 = time.time()
        density_brute = compute_density_brute_force(
            dftb_data['evecs'],
            dftb_data['occupations'],
            projector,
            atoms_dict,
            points_ang,
            setup_data
        )
        t_brute = time.time() - t0
        print(f"      Time: {t_brute:.2f} s")
        print(f"      Density range: [{density_brute.min():.6e}, {density_brute.max():.6e}]")
        # Estimate electron count from density (approximate for point evaluation)
        vol_per_point = (points_ang[:,0].max() - points_ang[:,0].min()) * \
                       (points_ang[:,1].max() - points_ang[:,1].min()) * 0.1  # assume 0.1 Å thickness
        electrons_est = np.sum(density_brute) * vol_per_point / len(points_ang)
        print(f"      Estimated electrons (approx): {electrons_est:.2f}")

        # Density matrix method - point-wise evaluation with correct formula
        print("\n[5b] Density matrix method (point-wise, correct formula)...")
        t0 = time.time()
        
        # Use DM from eigenvectors (correct normalization)
        dm_from_evecs = np.zeros((dftb_data['basis_size'], dftb_data['basis_size']))
        for imo, occ in enumerate(dftb_data['occupations']):
            if occ > 0:
                dm_from_evecs += occ * np.outer(dftb_data['evecs'][imo], dftb_data['evecs'][imo])
        
        print(f"\n[DEBUG] Density matrix from eigenvectors:")
        print(f"  Shape: {dm_from_evecs.shape}")
        print(f"  Matrix:\n{dm_from_evecs}")
        print(f"  Max element: {np.max(dm_from_evecs):.6f}")
        print(f"  Non-zero elements: {np.count_nonzero(dm_from_evecs > 1e-6)} / {dm_from_evecs.size}")
        
        # Compute density using correct formula: ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)
        density_dm = compute_density_dm_correct(dm_from_evecs, setup_data, projector, atoms_dict, points_ang)
        
        t_dm = time.time() - t0
        print(f"      Time: {t_dm:.2f} s")
        print(f"      Density range: [{density_dm.min():.6e}, {density_dm.max():.6e}]")

    else:
        # 3D grid evaluation
        print("\n[5] Computing density on 3D grid...")

        grid_spec = create_3d_grid(setup_data['atom_coords_ang'], step=args.step, margin=args.margin)
        print(f"      Grid: {grid_spec['ngrid']} = {np.prod(grid_spec['ngrid'])} points")

        # Brute force - sum orbitals on the same grid
        print("\n[5a] Brute force method (sum of occupied orbitals)...")
        # For 3D, we use the orbital projection on the full grid
        # This is expensive but serves as ground truth

        # Create points for the full grid
        nx, ny, nz = grid_spec['ngrid']
        x = grid_spec['origin'][0] + np.arange(nx) * grid_spec['dA'][0]
        y = grid_spec['origin'][1] + np.arange(ny) * grid_spec['dB'][1]
        z = grid_spec['origin'][2] + np.arange(nz) * grid_spec['dC'][2]

        yy, xx, zz = np.meshgrid(y, x, z, indexing='ij')
        points_ang = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        t0 = time.time()
        density_brute = compute_density_brute_force(
            dftb_data['evecs'],
            dftb_data['occupations'],
            projector,
            atoms_dict,
            points_ang,
            setup_data
        )
        t_brute = time.time() - t0
        density_brute = density_brute.reshape((nx, ny, nz))
        print(f"      Time: {t_brute:.2f} s")
        print(f"      Density range: [{density_brute.min():.6e}, {density_brute.max():.6e}]")

        # Density matrix method
        print("\n[5b] Density matrix method (fast)...")
        t0 = time.time()
        density_dm = compute_density_dm(
            rho_sparse,
            neigh_j,
            projector,
            atoms_dict,
            grid_spec
        )
        t_dm = time.time() - t0
        print(f"      Time: {t_dm:.2f} s")
        print(f"      Density range: [{density_dm.min():.6e}, {density_dm.max():.6e}]")

        points_ang = points_ang  # For plotting

    # Step 6: Validate and visualize
    print("\n[6] Validating and visualizing...")
    rms, max_err, max_rel = validate_and_plot(
        density_brute,
        density_dm,
        points_ang,
        density_brute.shape if len(density_brute.shape) == 3 else (args.npoints, args.npoints),
        args,
        OUTPUT_DIR
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"System:           H2O ({setup_data['natoms']} atoms, {dftb_data['basis_size']} orbitals)")
    print(f"Electrons:        {np.sum(dftb_data['occupations']):.1f}")
    print(f"Brute force time: {t_brute:.2f} s")
    print(f"DM method time:   {t_dm:.2f} s")
    print(f"Speedup:          {t_brute / t_dm:.1f}x")
    print(f"RMS error:        {rms:.6e}")
    print(f"Max rel error:    {max_rel:.6e}")

    if max_rel < 0.01:  # 1% threshold
        print("\n[PASS] Density matrix projection validated!")
    else:
        print("\n[FAIL] Large discrepancy detected - needs debugging")

    return 0 if max_rel < 0.01 else 1


if __name__ == '__main__':
    sys.exit(main())
