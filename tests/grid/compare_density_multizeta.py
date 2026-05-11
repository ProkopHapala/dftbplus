#!/usr/bin/env python3
"""
Parity check between libwaveplot and pyOpenCL for density projection.

For a given basis set (mio-1-1 or 3ob-3-1), compare:
- Molecular orbitals: libwaveplot vs pyOpenCL
- Density: libwaveplot (sum) vs pyOpenCL (sum) vs pyOpenCL (density matrix)

This is a parity check to ensure pyOpenCL produces the same results as libwaveplot.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyBall.OCL.DFTBplusParser import (
    parse_basis_hsd_ang, parse_detailed_xml_custom, parse_eigenvec_bin_custom, evec_to_kernel_coeffs, build_wp_basis
)
from pyBall.OCL.Grid import setup_gridprojector_from_dftb
from pyBall.WavePlot.TestUtils import generate_2d_point_grid
from pyBall.WavePlot.WavePlot import WavePlot
from pyBall.DFTBcore import DFTBcore
from pyBall import dftb_utils
import argparse

BOHR2ANG = 0.5291772109

def get_libwaveplot_results(dftb_dir, points_ang):
    """Get MO and density results from libwaveplot using orb2points."""
    dftb_dir = Path(dftb_dir)
    
    # Parse detailed.xml for geometry and occupations
    detailed = parse_detailed_xml_custom(str(dftb_dir / 'detailed.xml'))
    occupations = np.array(detailed['occupations']).flatten()
    
    # Parse basis from waveplot_in.hsd (multi-zeta)
    basis = parse_basis_hsd_ang(str(dftb_dir / 'waveplot_in.hsd'))
    species_used = set(detailed['species_names'])
    basis = [sp for sp in basis if sp['name'] in species_used]
    
    # Convert species names to integer indices for WavePlot (1-based)
    sp_name_to_idx = {sp['name']: i+1 for i, sp in enumerate(basis)}
    species_wp = np.array([sp_name_to_idx[detailed['species_names'][si]] for si in detailed['species_per_atom']], dtype=np.int32)
    
    # Build WavePlot basis format
    sp_names_hsd = [sp['name'] for sp in basis]
    wp_basis, resoln_b = build_wp_basis(basis, sp_names_hsd)
    
    # Get occupied orbitals
    occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
    
    # Parse eigenvec.bin for eigenvectors
    eigenvec_file = dftb_dir / 'eigenvec.bin'
    norb = detailed['norb']
    nstates = detailed['occupations'].shape[0]
    evecs = parse_eigenvec_bin_custom(str(eigenvec_file), nstates, norb)
    
    # Initialize WavePlot
    LIB_PATH = Path(__file__).parent.parent.parent / '_build' / 'app' / 'waveplot' / 'libwaveplot.so'
    wp = WavePlot(str(LIB_PATH))
    
    # Set geometry and basis (using same format as compare_waveplot_lib.py)
    wp.set_geometry(detailed['coords_bohr'], species_wp, is_periodic=False)
    wp.set_basis(wp_basis, resolution=resoln_b)
    wp.set_eigenvectors(evecs)
    
    # Convert points to Bohr for WavePlot
    points_bohr = points_ang / BOHR2ANG
    
    # Get occupied orbitals using orb2points
    npoints = int(np.sqrt(len(points_ang)))
    mo_values = []
    for imo in occupied_idx:
        mo_vals = wp.orb2points(imo + 1, points_bohr)  # WavePlot is 1-indexed
        mo_values.append(mo_vals.reshape(npoints, npoints))
    
    # Compute density manually from orbitals (sum of occupied orbitals squared)
    density = np.zeros(len(points_ang), dtype=np.float64)
    for i, (imo, occ) in enumerate(zip(occupied_idx, occupations[occupied_idx])):
        density += occ * (mo_values[i].ravel() ** 2)
    density = density.reshape(npoints, npoints)
    
    return {
        'mo_values': mo_values,
        'density': density,
        'occupations': occupations,
        'occupied_idx': occupied_idx,
        'atom_coords': detailed['coords_bohr'] * BOHR2ANG
    }


def get_pyopencl_results(dftb_dir, points_ang, step=0.1, system_name='test', z_offset=0.0):
    """Get MO and density results from pyOpenCL using DFTBcore."""
    from pathlib import Path
    
    dftb_dir = Path(dftb_dir)
    
    # Run DFTB+ calculation using dftb_utils
    dftb_data = dftb_utils.run_dftb_scf(dftb_dir, dm=True, h=False, s=True)
    evecs = dftb_data['evecs']
    dm_dense = dftb_data['dm_dense']
    s_dense = dftb_data['s_dense']
    detailed = dftb_data['detailed']
    occupations = dftb_data['occupations']
    basis_size = dftb_data['basis_size']
    
    # Parse basis from waveplot_in.hsd (multi-zeta)
    basis = parse_basis_hsd_ang(str(dftb_dir / 'waveplot_in.hsd'))
    species_used = set(detailed['species_names'])
    basis = [sp for sp in basis if sp['name'] in species_used]
    
    # Calculate extent based on molecular geometry
    atom_coords_ang = detailed['coords_bohr'] * BOHR2ANG
    rmin = float(atom_coords_ang.min()) - 3.0
    rmax_r = float(atom_coords_ang.max()) + 3.0
    extent = [rmin, rmax_r, rmin, rmax_r]
    
    # Setup OpenCL projector with d-orbital support
    dftb_data = {
        'coords_bohr': detailed['coords_bohr'],
        'species_per_atom': detailed['species_per_atom'],
        'species_names': detailed['species_names'],
    }
    
    # Compute norb_per_atom and max_shells from basis
    sp_by_name = {sp['name']: sp for sp in basis}
    natoms = len(detailed['coords_bohr'])
    norb_per_atom = []
    max_l = 0
    for ia in range(natoms):
        sp_name = detailed['species_names'][detailed['species_per_atom'][ia]]
        sp_info = sp_by_name[sp_name]
        norb = sum(2*orb['l']+1 for orb in sp_info['orbitals'])
        for orb in sp_info['orbitals']:
            max_l = max(max_l, orb['l'])
        norb_per_atom.append(norb)
    norb_per_atom = np.array(norb_per_atom, dtype=np.int32)
    max_shells = max_l + 1  # shells 0..max_l
    
    # Compute orbital offsets for dense matrix indexing
    orb_offsets = np.zeros(natoms + 1, dtype=np.int32)
    orb_offsets[1:] = np.cumsum(norb_per_atom)
    norb_total = int(orb_offsets[-1])
    
    projector, atoms_dict = setup_gridprojector_from_dftb(dftb_data, basis, verbosity=0, max_shells=max_shells)
    
    species_per_atom = detailed['species_per_atom']
    species_names = detailed['species_names']
    npoints = int(np.sqrt(len(points_ang)))
    
    # Get occupied orbitals
    occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
    
    # -----------------------------------------------------------
    # OLD CODE PATH: project each occupied orbital using sparse kernel
    # -----------------------------------------------------------
    mo_values_old = []
    for imo in occupied_idx:
        coeffs = evec_to_kernel_coeffs(evecs[imo], natoms, species_per_atom, species_names, basis)
        psi = projector.project_orbital_points(points_ang.astype(np.float32), coeffs, np.array(norb_per_atom, dtype=np.int32), atoms_dict).reshape(npoints, npoints)
        mo_values_old.append(psi)
    
    # Compute density using sum of orbitals (old path)
    density_sum_old = np.zeros(len(points_ang), dtype=np.float64)
    for i, (imo, occ) in enumerate(zip(occupied_idx, occupations[occupied_idx])):
        density_sum_old += occ * (mo_values_old[i].ravel() ** 2)
    density_sum_old = density_sum_old.reshape(npoints, npoints)
    
    # -----------------------------------------------------------
    # NEW CODE PATH: dense MO coefficient vector projection at points
    # -----------------------------------------------------------
    print(f"  [Dense] Testing orbital projection at points with max_shells={max_shells}, norb_total={norb_total}")
    mo_values_dense = []
    for imo in occupied_idx:
        coeffs_dense = evecs[imo].astype(np.float32)  # (norb_total,) in Fortran order
        psi = projector.project_orbital_dense_points(
            points_ang.astype(np.float32), coeffs_dense,
            norb_per_atom, orb_offsets, atoms_dict
        ).reshape(npoints, npoints)
        mo_values_dense.append(psi)
    
    # Compare old vs dense orbital projection
    for i, imo in enumerate(occupied_idx):
        diff = mo_values_old[i] - mo_values_dense[i]
        maxdiff = np.max(np.abs(diff))
        print(f"    MO{imo+1} old-vs-dense maxdiff = {maxdiff:.6e}")
    
    # Compute density using sum of orbitals (dense path)
    density_sum_dense = np.zeros(len(points_ang), dtype=np.float64)
    for i, (imo, occ) in enumerate(zip(occupied_idx, occupations[occupied_idx])):
        density_sum_dense += occ * (mo_values_dense[i].ravel() ** 2)
    density_sum_dense = density_sum_dense.reshape(npoints, npoints)
    
    # -----------------------------------------------------------
    # NEW CODE PATH: dense density matrix projection at points
    # -----------------------------------------------------------
    print(f"  [Dense] Testing density matrix projection at points")
    dm32 = dm_dense.astype(np.float32)
    density_dm_dense = projector.project_density_dense_points(
        points_ang.astype(np.float32), dm32,
        norb_per_atom, orb_offsets, atoms_dict
    ).reshape(npoints, npoints)
    
    # Compare sum-of-orbitals vs dense DM
    diff_sum_dm = density_sum_dense - density_dm_dense
    print(f"    dense-sum vs dense-DM maxdiff = {np.max(np.abs(diff_sum_dm)):.6e}")
    
    # -----------------------------------------------------------
    # NEW CODE PATH: 3D grid tests (if grid_spec provided)
    # -----------------------------------------------------------
    # Build a small 3D grid for testing
    grid_spec = {
        'origin': np.array([extent[0], extent[2], 0.0], dtype=np.float32),
        'dA': np.array([step, 0.0, 0.0], dtype=np.float32),
        'dB': np.array([0.0, step, 0.0], dtype=np.float32),
        'dC': np.array([0.0, 0.0, 1.0], dtype=np.float32),
        'ngrid': np.array([npoints, npoints, 1], dtype=np.int32),
    }
    print(f"  [Dense] Testing orbital projection on 3D grid")
    psi_grid_dense = projector.project_orbital_dense(
        evecs[occupied_idx[0]].astype(np.float32), norb_per_atom, orb_offsets, atoms_dict, grid_spec
    )
    print(f"    Grid shape: {psi_grid_dense.shape}, max = {np.max(np.abs(psi_grid_dense)):.6e}")
    
    print(f"  [Dense] Testing density matrix projection on 3D grid")
    rho_grid_dense = projector.project_density_dense(
        dm32, norb_per_atom, orb_offsets, atoms_dict, grid_spec
    )
    print(f"    Grid shape: {rho_grid_dense.shape}, max = {np.max(rho_grid_dense):.6e}")
    
    # Use dense results as primary output
    density_dm = density_dm_dense
    mo_values = mo_values_dense
    density_sum = density_sum_dense
    
    # Save dense kernel comparison results to separate file
    with open(f'dense_kernel_comparison_{system_name}_z{z_offset:.1f}.txt', 'w') as f:
        f.write(f"Dense Kernel Test Results: {system_name}\n")
        f.write(f"max_shells={max_shells}, norb_total={norb_total}\n\n")
        f.write("OLD vs DENSE Orbital Projection (points):\n")
        for i, imo in enumerate(occupied_idx):
            diff = mo_values_old[i] - mo_values_dense[i]
            maxdiff = np.max(np.abs(diff))
            f.write(f"  MO{imo+1}: maxdiff = {maxdiff:.6e}\n")
        f.write(f"\nDENSE-SUM vs DENSE-DM Density Projection (points):\n")
        f.write(f"  maxdiff = {np.max(np.abs(diff_sum_dm)):.6e}\n")
        f.write(f"\n3D Grid Tests:\n")
        f.write(f"  Orbital grid shape: {psi_grid_dense.shape}, max = {np.max(np.abs(psi_grid_dense)):.6e}\n")
        f.write(f"  Density grid shape: {rho_grid_dense.shape}, max = {np.max(rho_grid_dense):.6e}\n")
    print(f"  Dense kernel comparison saved to: dense_kernel_comparison_{system_name}_z{z_offset:.1f}.txt")
    
    return {
        'mo_values': mo_values,
        'mo_values_old': mo_values_old,
        'density_sum': density_sum,
        'density_dm': density_dm,
        'occupations': occupations,
        'occupied_idx': occupied_idx,
        'extent': extent,
        'atom_coords': atom_coords_ang,
        'points_ang': points_ang,
        'projector': projector,
        'atoms_dict': atoms_dict,
        'norb_per_atom': norb_per_atom,
        'orb_offsets': orb_offsets,
        'basis': basis,
        'evecs': evecs,
        'dm_dense': dm_dense,
        'grid_spec': grid_spec,
        'psi_grid_dense': psi_grid_dense,
        'rho_grid_dense': rho_grid_dense,
    }


def create_mo_comparison_figure(lib_data, ocl_data, output_path, dpi=150):
    """
    Create Figure 1: MO comparison for occupied orbitals.
    Layout: norb rows (occupied orbitals), 3 columns (libwaveplot vs pyOpenCL vs diff)
    """
    import matplotlib.pyplot as plt
    
    occupied_idx = lib_data['occupied_idx']
    n_occ = len(occupied_idx)
    
    if n_occ == 0:
        print("  No occupied orbitals found")
        return
    
    extent = ocl_data['extent']
    
    fig, axes = plt.subplots(n_occ, 3, figsize=(15, 5*n_occ))
    if n_occ == 1:
        axes = axes[np.newaxis, :]
    
    for row, idx in enumerate(occupied_idx):
        occ = lib_data['occupations'][idx]
        
        # libwaveplot MO
        psi_lib = lib_data['mo_values'][row]
        
        # pyOpenCL MO
        psi_ocl = ocl_data['mo_values'][row]
        
        # Difference
        diff = psi_lib - psi_ocl
        clim = max(np.abs(psi_lib).max(), np.abs(psi_ocl).max())
        
        # Plot libwaveplot
        im0 = axes[row, 0].imshow(psi_lib, origin='lower', cmap='RdBu_r', vmin=-clim, vmax=clim, extent=extent)
        axes[row, 0].set_title(f'libwaveplot MO{idx+1} (occ={occ:.1f})')
        plt.colorbar(im0, ax=axes[row, 0])
        
        # Plot pyOpenCL
        im1 = axes[row, 1].imshow(psi_ocl, origin='lower', cmap='RdBu_r', vmin=-clim, vmax=clim, extent=extent)
        axes[row, 1].set_title(f'pyOpenCL MO{idx+1} (occ={occ:.1f})')
        plt.colorbar(im1, ax=axes[row, 1])
        
        # Plot difference
        im2 = axes[row, 2].imshow(diff, origin='lower', cmap='bwr', vmin=-clim*0.1, vmax=clim*0.1, extent=extent)
        axes[row, 2].set_title(f'Diff (max|diff|={np.max(np.abs(diff)):.2e})')
        plt.colorbar(im2, ax=axes[row, 2])
        
        # Add atoms to all plots
        for col in range(3):
            axes[row, col].scatter(ocl_data['atom_coords'][:, 0], ocl_data['atom_coords'][:, 1], 
                                   c='black', marker='.', s=10, alpha=0.5, zorder=10)
    
    plt.suptitle('MO Parity Check: libwaveplot vs pyOpenCL (Occupied Orbitals)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"  Figure 1 saved to: {output_path}")


def create_density_comparison_figure(lib_data, ocl_data, output_path, dpi=150):
    """
    Create Figure 2: Density comparison with OLD vs NEW clearly distinguished.
    If lib_data is None, show only pyOpenCL results (dense kernel test).
    """
    import matplotlib.pyplot as plt
    
    extent = ocl_data['extent']
    
    if lib_data is None:
        # Dense-only mode: show pyOpenCL OLD (sparse), NEW (dense sum), NEW (dense DM)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Reconstruct OLD density from old orbital values
        if 'mo_values_old' in ocl_data:
            density_sum_old = np.zeros_like(ocl_data['density_sum'])
            for i, (imo, occ) in enumerate(zip(ocl_data['occupied_idx'], ocl_data['occupations'][ocl_data['occupied_idx']])):
                density_sum_old += occ * (ocl_data['mo_values_old'][i] ** 2)
            im0 = axes[0].imshow(density_sum_old, origin='lower', cmap='viridis', extent=extent)
            axes[0].set_title('pyOpenCL OLD (sparse)')
            plt.colorbar(im0, ax=axes[0])
        else:
            axes[0].text(0.5, 0.5, 'OLD data not available', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('pyOpenCL OLD (sparse) - N/A')
        
        # pyOpenCL NEW (dense sum)
        im1 = axes[1].imshow(ocl_data['density_sum'], origin='lower', cmap='viridis', extent=extent)
        axes[1].set_title('pyOpenCL NEW (dense sum)')
        plt.colorbar(im1, ax=axes[1])
        
        # pyOpenCL NEW (dense DM)
        im2 = axes[2].imshow(ocl_data['density_dm'], origin='lower', cmap='viridis', extent=extent)
        axes[2].set_title('pyOpenCL NEW (dense DM)')
        plt.colorbar(im2, ax=axes[2])
        
        # Add atoms to all plots
        for col in range(3):
            if axes[col].get_visible():
                axes[col].scatter(ocl_data['atom_coords'][:, 0], ocl_data['atom_coords'][:, 1], 
                                   c='black', marker='.', s=10, alpha=0.5, zorder=10)
        
        plt.suptitle('Density Projection: pyOpenCL OLD (sparse) vs NEW (dense)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        print(f"  Figure 2 saved to: {output_path}")
        return
    
    # Full comparison mode with libwaveplot reference
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 1: Four density calculations
    # libwaveplot sum (reference)
    im00 = axes[0, 0].imshow(lib_data['density'], origin='lower', cmap='viridis', extent=extent)
    axes[0, 0].set_title('libwaveplot (reference)')
    plt.colorbar(im00, ax=axes[0, 0])
    
    # pyOpenCL OLD (sparse) - need to compute this
    if 'mo_values_old' in ocl_data:
        # Reconstruct OLD density from old orbital values
        density_sum_old = np.zeros_like(lib_data['density'])
        for i, (imo, occ) in enumerate(zip(ocl_data['occupied_idx'], ocl_data['occupations'][ocl_data['occupied_idx']])):
            density_sum_old += occ * (ocl_data['mo_values_old'][i] ** 2)
        im01 = axes[0, 1].imshow(density_sum_old, origin='lower', cmap='viridis', extent=extent)
        axes[0, 1].set_title('pyOpenCL OLD (sparse)')
        plt.colorbar(im01, ax=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, 'OLD data not available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('pyOpenCL OLD (sparse) - N/A')
    
    # pyOpenCL NEW (dense sum)
    im02 = axes[0, 2].imshow(ocl_data['density_sum'], origin='lower', cmap='viridis', extent=extent)
    axes[0, 2].set_title('pyOpenCL NEW (dense sum)')
    plt.colorbar(im02, ax=axes[0, 2])
    
    # pyOpenCL NEW (dense DM)
    im03 = axes[0, 3].imshow(ocl_data['density_dm'], origin='lower', cmap='viridis', extent=extent)
    axes[0, 3].set_title('pyOpenCL NEW (dense DM)')
    plt.colorbar(im03, ax=axes[0, 3])
    
    # Row 2: Differences
    # diff: libwaveplot vs OLD
    if 'mo_values_old' in ocl_data:
        diff_lib_old = lib_data['density'] - density_sum_old
        clim1 = np.max(np.abs(diff_lib_old))
        im10 = axes[1, 0].imshow(diff_lib_old, origin='lower', cmap='RdBu_r', vmin=-clim1, vmax=clim1, extent=extent)
        axes[1, 0].set_title(f'libwaveplot vs OLD\nmax|diff|={clim1:.6e}')
        plt.colorbar(im10, ax=axes[1, 0])
    else:
        axes[1, 0].axis('off')
    
    # diff: libwaveplot vs NEW (dense sum)
    diff_lib_new = lib_data['density'] - ocl_data['density_sum']
    clim2 = np.max(np.abs(diff_lib_new))
    im11 = axes[1, 1].imshow(diff_lib_new, origin='lower', cmap='RdBu_r', vmin=-clim2, vmax=clim2, extent=extent)
    axes[1, 1].set_title(f'libwaveplot vs NEW (sum)\nmax|diff|={clim2:.6e}')
    plt.colorbar(im11, ax=axes[1, 1])
    
    # diff: libwaveplot vs NEW (dense DM)
    diff_lib_dm = lib_data['density'] - ocl_data['density_dm']
    clim3 = np.max(np.abs(diff_lib_dm))
    im12 = axes[1, 2].imshow(diff_lib_dm, origin='lower', cmap='RdBu_r', vmin=-clim3, vmax=clim3, extent=extent)
    axes[1, 2].set_title(f'libwaveplot vs NEW (DM)\nmax|diff|={clim3:.6e}')
    plt.colorbar(im12, ax=axes[1, 2])
    
    # diff: OLD vs NEW (dense sum) - validation
    if 'mo_values_old' in ocl_data:
        diff_old_new = density_sum_old - ocl_data['density_sum']
        clim4 = np.max(np.abs(diff_old_new))
        im13 = axes[1, 3].imshow(diff_old_new, origin='lower', cmap='RdBu_r', vmin=-clim4, vmax=clim4, extent=extent)
        axes[1, 3].set_title(f'OLD vs NEW (sum)\nmax|diff|={clim4:.6e}')
        plt.colorbar(im13, ax=axes[1, 3])
    else:
        axes[1, 3].axis('off')
    
    # Add atoms to all plots
    for row in range(2):
        for col in range(4):
            if axes[row, col].get_visible():
                axes[row, col].scatter(ocl_data['atom_coords'][:, 0], ocl_data['atom_coords'][:, 1], 
                                       c='black', marker='.', s=10, alpha=0.5, zorder=10)
    
    plt.suptitle('Density Parity Check: libwaveplot vs pyOpenCL OLD (sparse) vs NEW (dense)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"  Figure 2 saved to: {output_path}")




def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Parity check between libwaveplot and pyOpenCL')
    parser.add_argument('--system', choices=['h2o-mio', 'h2o-3ob', 'ptcda-mio', 'ptcda-3ob', 'tbtap-3ob'], required=True, help='System to use (h2o-mio, h2o-3ob, ptcda-mio, ptcda-3ob, or tbtap-3ob)')
    parser.add_argument('--step', type=float, default=0.1, help='Grid step in Angstrom (default: 0.1)')
    parser.add_argument('--z-offsets', type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], help='Z offsets in Angstrom for XY planes (default: 0.0 0.5 1.0 1.5 2.0 2.5)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output images (default: 150)')
    parser.add_argument('--no-mo-plot', action='store_true', help='Skip MO comparison figure (useful for systems with many orbitals)')
    args = parser.parse_args()
    
    # Set directory based on system
    if args.system == 'h2o-mio':
        dftb_dir = 'dftb_h2o'
        basis_name = 'mio-1-1'
    elif args.system == 'h2o-3ob':
        dftb_dir = 'dftb_h2o_3ob'
        basis_name = '3ob-3-1'
    elif args.system == 'ptcda-mio':
        dftb_dir = 'dftb_ptcda'
        basis_name = 'mio-1-1'
    elif args.system == 'ptcda-3ob':
        dftb_dir = 'dftb_ptcda_3ob'
        basis_name = '3ob-3-1'
    else:  # tbtap-3ob
        dftb_dir = 'dftb_tbtap'
        basis_name = '3ob-3-1'
    
    print("=" * 70)
    print(f"PARITY CHECK: libwaveplot vs pyOpenCL ({args.system})")
    print("=" * 70)
    
    # Parse detailed.xml to get geometry for extent calculation
    detailed = parse_detailed_xml_custom(str(Path(dftb_dir) / 'detailed.xml'))
    atom_coords_ang = detailed['coords_bohr'] * BOHR2ANG
    rmin = float(atom_coords_ang.min()) - 3.0
    rmax_r = float(atom_coords_ang.max()) + 3.0
    
    # Calculate number of points from step size
    range_size = rmax_r - rmin
    npoints = int(np.ceil(range_size / args.step))
    print(f"  Grid step: {args.step} Å, range: [{rmin:.2f}, {rmax_r:.2f}] Å")
    print(f"  Number of points: {npoints}x{npoints}")
    print(f"  Z offsets: {args.z_offsets}")
    
    all_fig2_paths = []
    
    # Loop over all z-offsets
    for z_offset in args.z_offsets:
        print(f"\n{'='*70}")
        print(f"Z offset: {z_offset} Å")
        print(f"{'='*70}")
        
        # Generate 2D grid (same for both libwaveplot and pyOpenCL)
        points_ang, extent = generate_2d_point_grid('xy', npoints, z_offset, (rmin, rmax_r))
        print(f"  Extent = {extent}")
        
        # Get libwaveplot results (using orb2points) - skip for systems without wfc file
        lib_data = None
        if args.system != 'tbtap-3ob':  # tbtap doesn't have wfc file yet
            print(f"\n--- Getting libwaveplot results from {dftb_dir} ---")
            lib_data = get_libwaveplot_results(dftb_dir, points_ang)
            print(f"  Found {len(lib_data['occupied_idx'])} occupied orbitals")
        else:
            print(f"\n--- Skipping libwaveplot for {args.system} (no wfc file) ---")
        
        # Get pyOpenCL results (using same points)
        print(f"\n--- Getting pyOpenCL results from {dftb_dir} ---")
        ocl_data = get_pyopencl_results(dftb_dir, points_ang, step=args.step, system_name=args.system, z_offset=z_offset)
        print(f"  Computed {len(ocl_data['occupied_idx'])} occupied orbitals")
        
        # Compute statistics (only if libwaveplot data available)
        if lib_data:
            mo_errors = []
            for i, (lib_mo, ocl_mo) in enumerate(zip(lib_data['mo_values'], ocl_data['mo_values'])):
                diff = lib_mo - ocl_mo
                max_err = np.max(np.abs(diff))
                mo_errors.append(max_err)
            
            # Density parity
            diff_density = lib_data['density'] - ocl_data['density_sum']
            density_max_err = np.max(np.abs(diff_density))
            density_corr = np.corrcoef(lib_data['density'].flatten(), ocl_data['density_sum'].flatten())[0, 1]
        else:
            mo_errors = []
            density_max_err = None
            density_corr = None
        
        print(f"\n--- Parity Statistics ---")
        if lib_data:
            print(f"  MO max errors: {mo_errors}")
            print(f"  Density max error: {density_max_err:.6e}")
            print(f"  Density correlation: {density_corr:.6f}")
        else:
            print(f"  No libwaveplot comparison (dense kernel only)")
        
        # Create Figure 1: MO comparison (skip if --no-mo-plot or too many orbitals)
        fig1_path = None
        if lib_data and not args.no_mo_plot and len(lib_data['occupied_idx']) <= 10:
            print("\n--- Creating Figure 1: MO comparison ---")
            fig1_path = Path(f'density_parity_{args.system}_z{z_offset:.1f}_fig1_mo.png')
            create_mo_comparison_figure(lib_data, ocl_data, fig1_path, dpi=args.dpi)
        elif lib_data:
            print(f"\n--- Skipping Figure 1 (MO comparison) - {len(lib_data['occupied_idx'])} orbitals, use --no-mo-plot to suppress this message ---")
        else:
            print(f"\n--- Skipping Figure 1 (no libwaveplot data) ---")
        
        # Create Figure 2: Density comparison (with or without libwaveplot data)
        print("\n--- Creating Figure 2: Density comparison ---")
        fig2_path = Path(f'density_parity_{args.system}_z{z_offset:.1f}_fig2_density.png')
        create_density_comparison_figure(lib_data, ocl_data, fig2_path, dpi=args.dpi)
        all_fig2_paths.append(fig2_path)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"System: {args.system}")
    print(f"  Z offsets: {args.z_offsets}")
    if lib_data:
        print(f"  Occupied orbitals: {len(lib_data['occupied_idx'])}")
        print(f"  MO max errors: {mo_errors}")
        print(f"  Density max error: {density_max_err:.6e}")
        print(f"  Density correlation: {density_corr:.6f}")
    else:
        print(f"  Occupied orbitals: {len(ocl_data['occupied_idx'])}")
        print(f"  Dense kernel test completed (no libwaveplot reference)")
    print(f"\nOutput files:")
    if fig1_path:
        print(f"  Figure 1 (MO comparison): {fig1_path.absolute()}")
    for fig2_path in all_fig2_paths:
        print(f"  Figure 2 (Density comparison): {fig2_path.absolute()}")
    print(f"  Dense kernel comparison: dense_kernel_comparison_{args.system}_z{args.z_offsets[-1]:.1f}.txt")
    
    if lib_data and density_max_err < 0.01:
        print("\n[PASS] Parity achieved between libwaveplot and pyOpenCL")
    elif lib_data:
        print(f"\n[INFO] Parity not achieved (max error = {density_max_err:.6e})")
    else:
        print(f"\n[INFO] Dense kernel test completed for {args.system}")


if __name__ == '__main__':
    main()
