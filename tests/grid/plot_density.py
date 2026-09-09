#!/usr/bin/env python3
"""
Plot DFTB+ density and orbitals using pyOpenCL projection.

This is a simplified script that uses only:
- pyBall.DFTBcore (fast library)
- pyBall.OCL.Grid (OpenCL projector)
- pyBall.GridUtils (grid generation)
- pyBall.ProjectionUtils (projection setup)
- pyBall.PlotUtils (plotting)

No dependency on libwaveplot or dftb_utils.
"""

import sys
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyBall.GridUtils import (
    compute_bbox_margin, estimate_npoints_from_step, generate_2d_point_grid, build_grid_spec_2d
)
from pyBall.ProjectionUtils import (
    run_dftb_calculation, setup_projector, get_orbital_indices,
    project_orbital_at_points, project_density_at_points,
    density_from_orbitals
)
from pyBall.PlotUtils import (
    plot_orbital_2d, plot_density_2d, plot_orbitals_grid, plot_density_comparison
)
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang

BOHR2ANG = 0.5291772109


def main():
    parser = argparse.ArgumentParser(description='Plot DFTB+ density and orbitals')
    parser.add_argument('--xyz', type=str, required=True, help='Path to .xyz file')
    parser.add_argument('--basis', choices=['mio-1-1', '3ob-3-1'], required=True, help='Basis set')
    parser.add_argument('--work-dir', type=str, default='.', help='Working directory for DFTB+ calc')
    parser.add_argument('--mode', choices=['orbitals', 'density', 'both'], default='density',
                        help='What to plot: orbitals, density, or both')
    parser.add_argument('--mo', type=str, nargs='+', default=None,
                        help='MO indices (1-based) or HOMO/HOMO-1/LUMO etc.')
    parser.add_argument('--method', choices=['dense', 'sparse'], default='dense',
                        help='Projection method: dense (supports d-orbitals) or sparse (sp only)')
    parser.add_argument('--z-offsets', type=float, nargs='+', default=[0.0],
                        help='Z offsets in Angstrom for XY planes (default: 0.0)')
    parser.add_argument('--step', type=float, default=0.1, help='Grid step in Angstrom (default: 0.1)')
    parser.add_argument('--margin', type=float, default=4.0, help='Bounding box margin in Angstrom (default: 4.0)')
    parser.add_argument('--output-prefix', type=str, default='plot', help='Output file prefix')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output images')
    parser.add_argument('--lib-path', type=str, default=None, help='Path to libdftbcore.so')
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DFTB+ Density/Orbital Plotter")
    print("=" * 70)
    print(f"  XYZ: {args.xyz}")
    print(f"  Basis: {args.basis}")
    print(f"  Work dir: {work_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Method: {args.method}")
    print(f"  Step: {args.step} A")
    print(f"  Margin: {args.margin} A")
    print(f"  Z offsets: {args.z_offsets}")
    
    # Copy xyz to work dir if needed
    xyz_path = Path(args.xyz)
    geom_path = work_dir / 'geom.xyz'
    if not geom_path.exists() or geom_path.read_text() != xyz_path.read_text():
        geom_path.write_text(xyz_path.read_text())
    
    # Write dftb_in.hsd if not exists
    dftb_in = work_dir / 'dftb_in.hsd'
    if not dftb_in.exists():
        write_dftb_input(work_dir, args.basis)
    
    # Run DFTB+ calculation
    print("\n--- Running DFTB+ calculation ---")
    dftb_data = run_dftb_calculation(work_dir, lib_path=args.lib_path)
    detailed = dftb_data['detailed']
    occupations = dftb_data['occupations']
    evecs = dftb_data['evecs']
    dm_dense = dftb_data['dm_dense']
    
    print(f"  Energy: {dftb_data['energy']:.6f} Hartree")
    print(f"  Basis size: {dftb_data['basis_size']}")
    print(f"  Occupied orbitals: {sum(occupations > 0)}")
    
    # Parse basis from waveplot_in.hsd (which includes wfc file via <<+)
    # Always write waveplot_in.hsd to ensure <<+ include is correct
    wfc_file = work_dir / f'wfc.{args.basis}.hsd'
    wp_file = work_dir / 'waveplot_in.hsd'
    if wfc_file.exists():
        wp_in = f"""Options = {{
    PlottedLevels = {{ 1 }}
    PlottedKPoints = {{ 1 }}
    PlottedSpins = {{ 1 }}
    PlottedRegion = {{
        Box [Angstrom] = {{
            10.0     0.0     0.0
            0.0     10.0     0.0
            0.0     0.0     10.0
        }}
        Origin [Angstrom] = {{ -5.0 -5.0 -5.0 }}
    }}
    NrOfPoints = {{ 50 50 50 }}
    RealComponent = Yes
}}

DetailedXML = "detailed.xml"
EigenvecBin = "eigenvec.bin"

GroundState = Yes

Basis = {{
    Resolution = 0.04
    <<+ "wfc.{args.basis}.hsd"
}}
"""
        with open(wp_file, 'w') as f:
            f.write(wp_in)
    
    basis = parse_basis_hsd_ang(str(wp_file))
    
    # Setup projector
    print("\n--- Setting up OpenCL projector ---")
    projector, atoms_dict, norb_per_atom, orb_offsets, max_shells = setup_projector(
        detailed, basis, max_shells=None, verbosity=0
    )
    print(f"  max_shells: {max_shells}")
    print(f"  norb_total: {orb_offsets[-1]}")
    
    # Compute grid extent
    atom_coords_ang = detailed['coords_bohr'] * BOHR2ANG
    rmin, rmax = compute_bbox_margin(detailed['coords_bohr'], margin=args.margin)
    npoints = estimate_npoints_from_step(rmax - rmin, args.step)
    print(f"\n  Grid: {npoints}x{npoints}, range: [{rmin:.2f}, {rmax:.2f}] A")
    
    # Process each z-offset
    for z_offset in args.z_offsets:
        print(f"\n{'='*70}")
        print(f"Z offset: {z_offset} A")
        print(f"{'='*70}")
        
        points_ang, extent = generate_2d_point_grid('xy', npoints, z_offset, (rmin, rmax))
        
        if args.mode in ('orbitals', 'both'):
            plot_orbitals(args, detailed, occupations, evecs, projector, atoms_dict,
                          norb_per_atom, orb_offsets, points_ang, extent, atom_coords_ang, z_offset)
        
        if args.mode in ('density', 'both'):
            plot_density(args, occupations, evecs, dm_dense, projector, atoms_dict,
                         norb_per_atom, orb_offsets, points_ang, extent, atom_coords_ang, z_offset)
    
    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


def plot_orbitals(args, detailed, occupations, evecs, projector, atoms_dict,
                  norb_per_atom, orb_offsets, points_ang, extent, atom_coords_ang, z_offset):
    """Plot selected orbitals."""
    
    # Get MO indices
    if args.mo is None:
        # Default: plot HOMO
        occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
        mo_indices = [occupied_idx[-1] + 1] if occupied_idx else []
    else:
        mo_indices = get_orbital_indices(occupations, mo_list=args.mo)
    
    if not mo_indices:
        print("  No orbitals to plot")
        return
    
    print(f"  Plotting orbitals: {[f'MO{i+1}' for i in mo_indices]}")
    
    # Project each orbital
    mo_values = []
    for imo in mo_indices:
        if args.method == 'dense':
            coeffs = evecs[imo].astype(np.float32)
        else:
            from pyBall.OCL.DFTBplusParser import evec_to_kernel_coeffs
            coeffs = evec_to_kernel_coeffs(
                evecs[imo], len(detailed['coords_bohr']),
                detailed['species_per_atom'], detailed['species_names'],
                parse_basis_hsd_ang(str(Path(args.work_dir) / 'waveplot_in.hsd'))
            )
        
        psi = project_orbital_at_points(
            projector, points_ang, coeffs, norb_per_atom, orb_offsets, atoms_dict,
            method=args.method
        )
        npoints = int(np.sqrt(len(points_ang)))
        mo_values.append(psi.reshape(npoints, npoints))
    
    # Plot individual orbitals
    for i, imo in enumerate(mo_indices):
        title = f'MO{imo+1} (occ={occupations[imo]:.1f}) z={z_offset:.1f}A'
        output = Path(args.work_dir) / f'{args.output_prefix}_MO{imo+1}_z{z_offset:.1f}.png'
        plot_orbital_2d(mo_values[i], extent, atom_coords_ang, title, output, dpi=args.dpi)
    
    # Plot grid of orbitals if multiple
    if len(mo_indices) > 1:
        output = Path(args.work_dir) / f'{args.output_prefix}_orbitals_z{z_offset:.1f}.png'
        plot_orbitals_grid(mo_values, [i+1 for i in mo_indices], extent, atom_coords_ang, output, dpi=args.dpi)


def plot_density(args, occupations, evecs, dm_dense, projector, atoms_dict,
                 norb_per_atom, orb_offsets, points_ang, extent, atom_coords_ang, z_offset):
    """Plot density using sum of orbitals or density matrix."""
    
    print(f"  Computing density projection (method={args.method})")
    
    occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
    
    if args.method == 'dense':
        # Use density matrix projection (the new dense method)
        density_dm = project_density_at_points(
            projector, points_ang, dm_dense, norb_per_atom, orb_offsets, atoms_dict,
            method='dense'
        )
        npoints = int(np.sqrt(len(points_ang)))
        density_dm = density_dm.reshape(npoints, npoints)
        
        # Also compute sum of orbitals for comparison
        mo_values = []
        for imo in occupied_idx:
            coeffs = evecs[imo].astype(np.float32)
            psi = project_orbital_at_points(
                projector, points_ang, coeffs, norb_per_atom, orb_offsets, atoms_dict,
                method='dense'
            )
            mo_values.append(psi.reshape(npoints, npoints))
        
        density_sum = density_from_orbitals(mo_values, occupied_idx, occupations)
        
        # Plot comparison
        output = Path(args.work_dir) / f'{args.output_prefix}_density_z{z_offset:.1f}.png'
        plot_density_comparison(
            [density_sum, density_dm],
            ['Sum of orbitals (dense)', 'Density matrix (dense)'],
            extent, atom_coords_ang, output, dpi=args.dpi
        )
        
        # Also plot individual densities
        output_sum = Path(args.work_dir) / f'{args.output_prefix}_density_sum_z{z_offset:.1f}.png'
        plot_density_2d(density_sum, extent, atom_coords_ang,
                        f'Total density (sum) z={z_offset:.1f}A', output_sum, dpi=args.dpi)
        
        output_dm = Path(args.work_dir) / f'{args.output_prefix}_density_dm_z{z_offset:.1f}.png'
        plot_density_2d(density_dm, extent, atom_coords_ang,
                        f'Total density (DM) z={z_offset:.1f}A', output_dm, dpi=args.dpi)
        
        maxdiff = np.max(np.abs(density_sum - density_dm))
        print(f"    Sum vs DM maxdiff: {maxdiff:.6e}")
        
    else:
        # Sparse method: sum of orbitals only
        mo_values = []
        for imo in occupied_idx:
            from pyBall.OCL.DFTBplusParser import evec_to_kernel_coeffs
            coeffs = evec_to_kernel_coeffs(
                evecs[imo], len(detailed['coords_bohr']),
                detailed['species_per_atom'], detailed['species_names'],
                parse_basis_hsd_ang(str(Path(args.work_dir) / 'waveplot_in.hsd'))
            )
            psi = project_orbital_at_points(
                projector, points_ang, coeffs, norb_per_atom, orb_offsets, atoms_dict,
                method='sparse'
            )
            npoints = int(np.sqrt(len(points_ang)))
            mo_values.append(psi.reshape(npoints, npoints))
        
        density = density_from_orbitals(mo_values, occupied_idx, occupations)
        
        output = Path(args.work_dir) / f'{args.output_prefix}_density_z{z_offset:.1f}.png'
        plot_density_2d(density, extent, atom_coords_ang,
                        f'Total density (sparse) z={z_offset:.1f}A', output, dpi=args.dpi)


def write_dftb_input(work_dir, basis_set):
    """Write minimal dftb_in.hsd for orbital/density calculation."""
    import os
    import shutil
    import re
    sk_path = os.environ.get('DFTB_SK_PATH', '/home/prokop/SIMULATIONS/dftbplus/slakos/library/')
    sk_dir = f"{sk_path}/{basis_set}/"
    
    # Read xyz to get elements
    xyz_file = work_dir / 'geom.xyz'
    with open(xyz_file) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    elements = set()
    for i in range(n_atoms):
        parts = lines[2 + i].split()
        elements.add(parts[0])
    
    # Copy wfc file from existing test directories FIRST (needed for max angular momentum)
    wfc_file = work_dir / f'wfc.{basis_set}.hsd'
    wfc_copied = False
    if not wfc_file.exists():
        repo_root = Path(__file__).parent.parent.parent
        search_paths = [
            repo_root / 'tests' / 'grid' / 'dftb_ptcda_3ob' / f'wfc.{basis_set}.hsd',
            repo_root / 'tests' / 'grid' / 'dftb_h2o_3ob' / f'wfc.{basis_set}.hsd',
        ]
        for src in search_paths:
            if src.exists():
                shutil.copy(str(src), str(wfc_file))
                print(f"  Copied wfc file: {src} -> {wfc_file}")
                wfc_copied = True
                break
        else:
            print(f"  WARNING: wfc file not found, searching in {search_paths}")
    
    # Get max angular momentum from wfc file (basis set) if available
    if wfc_file.exists():
        max_ang = {}
        with open(wfc_file) as f:
            content = f.read()
        for elem in elements:
            elem_pattern = rf'{elem}\s*{{'
            elem_match = re.search(elem_pattern, content)
            if elem_match:
                start = elem_match.end()
                next_elem = re.search(r'\n\w+\s*{{', content[start:])
                if next_elem:
                    block = content[start:start+next_elem.start()]
                else:
                    block = content[start:]
                ams = re.findall(r'AngularMomentum\s*=\s*(\d+)', block)
                if ams:
                    max_l = max(int(m) for m in ams)
                    l_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
                    max_ang[elem] = l_map.get(max_l, 's')
                else:
                    from pyBall import elements as elem_module
                    max_ang[elem] = elem_module.ELEMENT_DICT[elem][4]
            else:
                from pyBall import elements as elem_module
                max_ang[elem] = elem_module.ELEMENT_DICT[elem][4]
    else:
        from pyBall import elements as elem_module
        max_ang = {e: elem_module.ELEMENT_DICT[e][4] for e in elements}
    max_ang_str = '\n'.join([f'        {e} = "{max_ang[e]}"' for e in sorted(elements)])
    
    dftb_in = f"""Geometry = xyzFormat {{
    <<< "geom.xyz"
}}

Driver = {{}}

Hamiltonian = DFTB {{
  SCC = Yes
  SCCTolerance = 1.0E-8
  MaxSCCIterations = 100
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_dir}"
    Separator = "-"
    Suffix = ".skf"
  }}
  MaxAngularMomentum {{
{max_ang_str}
  }}
}}

Analysis {{
  WriteEigenvectors = Yes
}}

Options {{
  WriteDetailedXml = Yes
}}
"""
    
    with open(work_dir / 'dftb_in.hsd', 'w') as f:
        f.write(dftb_in)
    print(f"  Written dftb_in.hsd")
    
    # Write waveplot_in.hsd with wfc include if available
    if wfc_file.exists():
        wp_in = f"""Options = {{
    PlottedLevels = {{ 1 }}
    PlottedKPoints = {{ 1 }}
    PlottedSpins = {{ 1 }}
    PlottedRegion = {{
        Box [Angstrom] = {{
            10.0     0.0     0.0
            0.0     10.0     0.0
            0.0     0.0     10.0
        }}
        Origin [Angstrom] = {{ -5.0 -5.0 -5.0 }}
    }}
    NrOfPoints = {{ 50 50 50 }}
    RealComponent = Yes
}}

DetailedXML = "detailed.xml"
EigenvecBin = "eigenvec.bin"

GroundState = Yes

Basis = {{
    Resolution = 0.04
    <<+ "wfc.{basis_set}.hsd"
}}
"""
    else:
        wp_in = """Options = {
    PlottedLevels = { 1 }
    PlottedKPoints = { 1 }
    PlottedSpins = { 1 }
    PlottedRegion = {
        Box [Angstrom] = {
            10.0     0.0     0.0
            0.0     10.0     0.0
            0.0     0.0     10.0
        }
        Origin [Angstrom] = { -5.0 -5.0 -5.0 }
    }
    NrOfPoints = { 50 50 50 }
    RealComponent = Yes
}

DetailedXML = "detailed.xml"
EigenvecBin = "eigenvec.bin"

GroundState = Yes

Basis = {
    Resolution = 0.04
}
"""
    
    with open(work_dir / 'waveplot_in.hsd', 'w') as f:
        f.write(wp_in)
    print(f"  Written waveplot_in.hsd")


if __name__ == '__main__':
    main()
