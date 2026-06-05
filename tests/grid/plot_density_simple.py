#!/usr/bin/env python3
"""
Simple density/orbital projection script using pyOpenCL.

Uses only:
- pyBall.DFTBcore (fast library)
- pyBall.OCL.Grid (OpenCL projector)
- pyBall.WavePlot.TestUtils.generate_2d_point_grid
- pyBall.dftb_utils.run_dftb_scf
- pyBall.plotUtils.plot_2d_array

No dependency on libwaveplot or pyBall.dftb_utils.py.
"""

import sys
import os
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyBall.WavePlot.TestUtils import generate_2d_point_grid
from pyBall import plotUtils
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang, parse_detailed_xml_custom
from pyBall.OCL.Grid import setup_gridprojector_from_dftb
from pyBall.DFTBcore import DFTBcore

BOHR2ANG = 0.5291772109


def run_dftb_scf(work_dir, lib_path=None, dm=True, h=False, s=True):
    """Run DFTB+ SCF calculation and return eigenvectors, DM, occupations, detailed."""
    import os
    work_dir = Path(work_dir)
    
    if lib_path is None:
        lib_paths = [
            Path(__file__).parent.parent.parent / '_build' / 'app' / 'dftbcore' / 'libdftbcore.so',
            Path(__file__).parent.parent.parent / 'build' / 'libdftbcore.so',
            Path(__file__).parent.parent.parent / 'build' / 'lib' / 'libdftbcore.so',
        ]
        for p in lib_paths:
            if p.exists():
                lib_path = str(p)
                break
        if lib_path is None:
            raise FileNotFoundError(f"libdftbcore.so not found in {lib_paths}")
    
    orig_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        dftb = DFTBcore(libpath=str(lib_path))
        input_file = work_dir / 'dftb_in.hsd'
        dftb.init(str(input_file))
        dftb.enable_matrix_collection(dm=dm, h=h, s=s)
        energy = dftb.run_scf()
        evecs, eigenvals = dftb.get_eigvecs_dense()
        dm_dense = dftb.get_dm_dense() if dm else None
        s_dense = dftb.get_s_dense() if s else None
        basis_size = dftb.get_basis_size()
        dftb.finalize()
    finally:
        os.chdir(orig_dir)
    
    detailed = parse_detailed_xml_custom(str(work_dir / 'detailed.xml'))
    occupations = np.array(detailed['occupations']).flatten()
    
    return {
        'evecs': evecs,
        'dm_dense': dm_dense,
        's_dense': s_dense,
        'occupations': occupations,
        'detailed': detailed,
        'energy': energy,
        'basis_size': basis_size,
    }


def write_dftb_input(elements, basis, work_dir):
    """Write simple dftb_in.hsd and waveplot_in.hsd for orbital analysis."""
    import shutil
    work_dir = Path(work_dir)
    
    # Copy wfc file first and parse it for correct max angular momentum
    wfc_file = f"wfc.{basis}.hsd"
    wfc_dest = work_dir / wfc_file
    
    if not wfc_dest.exists():
        # Search for wfc file in existing test directories
        search_paths = [
            Path(__file__).parent / f"dftb_h2o_3ob" / wfc_file,
            Path(__file__).parent / f"dftb_ptcda_3ob" / wfc_file,
            Path(__file__).parent / f"dftb_tbtap" / wfc_file,
        ]
        for p in search_paths:
            if p.exists():
                shutil.copy(p, wfc_dest)
                print(f"  Copied {wfc_file} from {p}")
                break
        if not wfc_dest.exists():
            raise FileNotFoundError(f"{wfc_file} not found in search paths")
    
    # Parse wfc file to get max angular momentum per element
    import re
    max_ang_from_wfc = {}
    with open(wfc_dest) as f:
        content = f.read()
    
    for el in elements:
        # Find the element block and extract max angular momentum
        pattern = rf'{el}\s*{{'
        match = re.search(pattern, content)
        if match:
            start = match.end()
            # Find the next element block or end of file
            next_match = re.search(r'\n\w+\s*{', content[start:])
            if next_match:
                block = content[start:start+next_match.start()]
            else:
                block = content[start:]
            
            # Extract all AngularMomentum values
            ams = re.findall(r'AngularMomentum\s*=\s*(\d+)', block)
            if ams:
                max_l = max(int(am) for am in ams)
                max_ang_from_wfc[el] = {0: 's', 1: 'p', 2: 'd'}.get(max_l, 'p')
            else:
                # Fallback to default
                max_ang_from_wfc[el] = 'p'
        else:
            max_ang_from_wfc[el] = 'p'
    
    # Map basis to Slater-Koster library path
    sk_path = os.environ.get('DFTB_SK_PATH', '/home/prokop/SIMULATIONS/dftbplus/slakos/library/')
    # Remove trailing slash to avoid double slashes
    sk_path = sk_path.rstrip('/')
    # Add basis set directory
    sk_path = f"{sk_path}/{basis}"
    
    # Build MaxAngularMomentum block from parsed wfc
    max_l_block = '  MaxAngularMomentum {\n'
    for el in elements:
        max_l_block += f'    {el} = "{max_ang_from_wfc.get(el, "p")}"\n'
    max_l_block += '  }'
    
    # Write dftb_in.hsd
    dftb_in = f'''Geometry = xyzFormat {{
    <<< "geom.xyz"
}}

Options {{
  WriteDetailedXML = Yes
}}

Analysis {{
  WriteEigenvectors = Yes
}}

Hamiltonian = DFTB {{
  Scc = Yes
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_path}/"
    Separator = "-"
    Suffix = ".skf"
  }}
{max_l_block}
  SCCTolerance = 1e-6
  MaxSCCIterations = 200
}}
'''
    (work_dir / 'dftb_in.hsd').write_text(dftb_in)
    
    # Write waveplot_in.hsd with wfc file
    waveplot_in = f'''Options {{
  TotalChargeDensity = Yes
  ChargeDensity = Yes
  RealComponent = Yes
  PlottedSpins = 1 -1
  PlottedLevels = 1:-1
  PlottedRegion = OptimalCuboid {{}}
  NrOfPoints = 50 50 50
  NrOfCachedGrids = -1
  Verbose = Yes
}}

DetailedXml = "detailed.xml"
EigenvecBin = "eigenvec.bin"

Basis {{
  Resolution = 0.01
  <<+ "{wfc_file}"
}}
'''
    (work_dir / 'waveplot_in.hsd').write_text(waveplot_in)


def main():
    parser = argparse.ArgumentParser(description='Plot DFTB+ density and orbitals')
    parser.add_argument('--xyz', type=str, required=True, help='Path to .xyz file')
    parser.add_argument('--basis', choices=['mio-1-1', '3ob-3-1'], required=True, help='Basis set')
    parser.add_argument('--work-dir', type=str, default='.', help='Working directory for DFTB+ calc')
    parser.add_argument('--mode', choices=['orbitals', 'density', 'both'], default='density',help='What to plot: orbitals, density, or both')
    parser.add_argument('--mo', type=str, nargs='+', default=None, help='MO indices (1-based) or HOMO/HOMO-1/LUMO etc.')
    parser.add_argument('--method', choices=['dense', 'sparse'], default='dense', help='Projection method: dense (supports d-orbitals) or sparse (sp only)')
    parser.add_argument('--z-offsets', type=float, nargs='+', default=[0.0], help='Z offsets in Angstrom for XY planes (default: 0.0)')
    parser.add_argument('--step', type=float, default=0.1, help='Grid step in Angstrom (default: 0.1)')
    parser.add_argument('--margin', type=float, default=4.0, help='Bounding box margin in Angstrom (default: 4.0)')
    parser.add_argument('--output-prefix', type=str, default='plot', help='Output file prefix')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output images')
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
    
    # Copy xyz to work dir
    xyz_path = Path(args.xyz)
    geom_path = work_dir / 'geom.xyz'
    if not geom_path.exists() or geom_path.read_text() != xyz_path.read_text():
        geom_path.write_text(xyz_path.read_text())
    
    # Write dftb_in.hsd if not exists
    dftb_in = work_dir / 'dftb_in.hsd'
    if not dftb_in.exists():
        # Read xyz to get elements
        with open(geom_path) as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        elements = set()
        for i in range(n_atoms):
            parts = lines[2 + i].split()
            elements.add(parts[0])
        write_dftb_input(list(elements), args.basis, work_dir)
        print(f"  Written dftb_in.hsd")
    
    # Run DFTB+ calculation
    print("\n--- Running DFTB+ calculation ---")
    dftb_data = run_dftb_scf(work_dir, dm=True, h=False, s=True)
    detailed = dftb_data['detailed']
    occupations = dftb_data['occupations']
    evecs = dftb_data['evecs']
    dm_dense = dftb_data['dm_dense']
    
    print(f"  Energy: {dftb_data['energy']:.6f} Hartree")
    print(f"  Basis size: {dftb_data['basis_size']}")
    print(f"  Occupied orbitals: {sum(occupations > 0)}")
    
    # Parse basis
    basis = parse_basis_hsd_ang(str(work_dir / 'waveplot_in.hsd'))
    
    # Setup projector
    print("\n--- Setting up OpenCL projector ---")
    dftb_data_proj = {
        'coords_bohr': detailed['coords_bohr'],
        'species_per_atom': detailed['species_per_atom'],
        'species_names': detailed['species_names'],
    }
    
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
    max_shells = max_l + 1
    
    orb_offsets = np.zeros(natoms + 1, dtype=np.int32)
    orb_offsets[1:] = np.cumsum(norb_per_atom)
    
    projector, atoms_dict = setup_gridprojector_from_dftb(dftb_data_proj, basis, verbosity=0, max_shells=max_shells)
    print(f"  max_shells: {max_shells}")
    print(f"  norb_total: {orb_offsets[-1]}")
    
    # Compute grid extent
    atom_coords_ang = detailed['coords_bohr'] * BOHR2ANG
    rmin = float(atom_coords_ang.min()) - args.margin
    rmax = float(atom_coords_ang.max()) + args.margin
    npoints = int(np.ceil((rmax - rmin) / args.step))
    print(f"\n  Grid: {npoints}x{npoints}, range: [{rmin:.2f}, {rmax:.2f}] A")
    
    # Process each z-offset
    for z_offset in args.z_offsets:
        print(f"\n{'='*70}")
        print(f"Z offset: {z_offset} A")
        print(f"{'='*70}")
        
        points_ang, extent = generate_2d_point_grid('xy', npoints, z_offset, (rmin, rmax))
        
        if args.mode in ('orbitals', 'both'):
            plot_orbitals(args, detailed, occupations, evecs, projector, atoms_dict,
                          norb_per_atom, orb_offsets, points_ang, extent, atom_coords_ang, z_offset, basis)
        
        if args.mode in ('density', 'both'):
            plot_density(args, occupations, evecs, dm_dense, projector, atoms_dict,
                         norb_per_atom, orb_offsets, points_ang, extent, atom_coords_ang, z_offset)
    
    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


def plot_orbitals(args, detailed, occupations, evecs, projector, atoms_dict,
                  norb_per_atom, orb_offsets, points_ang, extent, atom_coords_ang, z_offset, basis):
    """Plot selected orbitals."""
    
    # Get MO indices
    if args.mo is None:
        occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
        homo_idx = occupied_idx[-1] if occupied_idx else 0
        # Default: HOMO-4 to LUMO+4
        mo_indices = list(range(max(0, homo_idx - 4), min(len(occupations), homo_idx + 5)))
    else:
        mo_indices = []
        for mo in args.mo:
            mo = mo.strip().upper()
            if mo == 'HOMO':
                occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
                mo_indices.append(occupied_idx[-1])
            elif mo == 'LUMO':
                occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
                mo_indices.append(occupied_idx[-1] + 1)
            elif mo.startswith('HOMO-'):
                offset = int(mo.split('-')[1])
                occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
                mo_indices.append(occupied_idx[-1] - offset)
            elif mo.startswith('LUMO+'):
                offset = int(mo.split('+')[1])
                occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
                mo_indices.append(occupied_idx[-1] + 1 + offset)
            else:
                mo_indices.append(int(mo) - 1)
    
    if not mo_indices:
        print("  No orbitals to plot")
        return
    
    # Get HOMO index for relative labeling
    occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
    homo_idx = occupied_idx[-1] if occupied_idx else 0
    
    # Create labels
    mo_labels = []
    for imo in mo_indices:
        if imo == homo_idx:
            label = f'MO{imo+1} (HOMO)'
        elif imo == homo_idx + 1:
            label = f'MO{imo+1} (LUMO)'
        elif imo < homo_idx:
            label = f'MO{imo+1} (HOMO-{homo_idx-imo})'
        else:
            label = f'MO{imo+1} (LUMO+{imo-homo_idx-1})'
        mo_labels.append(label)
    
    print(f"  Plotting orbitals: {mo_labels}")
    
    # Project each orbital
    mo_values = []
    for imo in mo_indices:
        if args.method == 'dense':
            coeffs = evecs[imo].astype(np.float32)
            psi = projector.project_orbital_dense_points(
                points_ang.astype(np.float32), coeffs,
                norb_per_atom, orb_offsets, atoms_dict
            )
        else:
            from pyBall.OCL.DFTBplusParser import evec_to_kernel_coeffs
            coeffs = evec_to_kernel_coeffs(
                evecs[imo], len(detailed['coords_bohr']),
                detailed['species_per_atom'], detailed['species_names'], basis
            )
            psi = projector.project_orbital_points(
                points_ang.astype(np.float32), coeffs,
                np.array(norb_per_atom, dtype=np.int32), atoms_dict
            )
        npoints = int(np.sqrt(len(points_ang)))
        mo_values.append(psi.reshape(npoints, npoints))
    
    # Plot individual orbitals
    for i, imo in enumerate(mo_indices):
        title = f'{mo_labels[i]} (occ={occupations[imo]:.1f}) z={z_offset:.1f}A'
        output = Path(args.work_dir) / f'{args.output_prefix}_MO{imo+1}_z{z_offset:.1f}.png'
        plotUtils.plot_2d_array(mo_values[i], extent, atom_coords_ang, title, output, dpi=args.dpi, cmap='RdBu_r', symmetric=True)


def plot_density(args, occupations, evecs, dm_dense, projector, atoms_dict,
                 norb_per_atom, orb_offsets, points_ang, extent, atom_coords_ang, z_offset):
    """Plot density using sum of orbitals or density matrix."""
    
    print(f"  Computing density projection (method={args.method})")
    
    occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
    npoints = int(np.sqrt(len(points_ang)))
    
    if args.method == 'dense':
        # Use density matrix projection
        density_dm = projector.project_density_dense_points(
            points_ang.astype(np.float32), dm_dense.astype(np.float32),
            norb_per_atom, orb_offsets, atoms_dict
        ).reshape(npoints, npoints)
        
        # Also compute sum of orbitals for comparison
        mo_values = []
        for imo in occupied_idx:
            coeffs = evecs[imo].astype(np.float32)
            psi = projector.project_orbital_dense_points(
                points_ang.astype(np.float32), coeffs,
                norb_per_atom, orb_offsets, atoms_dict
            )
            mo_values.append(psi.reshape(npoints, npoints))
        
        density_sum = np.zeros_like(mo_values[0])
        for i, imo in enumerate(occupied_idx):
            density_sum += occupations[imo] * (mo_values[i] ** 2)
        
        # Plot both
        output_sum = Path(args.work_dir) / f'{args.output_prefix}_density_sum_z{z_offset:.1f}.png'
        plotUtils.plot_2d_array(density_sum, extent, atom_coords_ang,
                                f'Total density (sum) z={z_offset:.1f}A', output_sum, dpi=args.dpi, cmap='magma')
        
        output_dm = Path(args.work_dir) / f'{args.output_prefix}_density_dm_z{z_offset:.1f}.png'
        plotUtils.plot_2d_array(density_dm, extent, atom_coords_ang,
                                f'Total density (DM) z={z_offset:.1f}A', output_dm, dpi=args.dpi, cmap='magma')
        
        maxdiff = np.max(np.abs(density_sum - density_dm))
        print(f"    Sum vs DM maxdiff: {maxdiff:.6e}")
        
    else:
        # Sparse method: sum of orbitals only
        from pyBall.OCL.DFTBplusParser import evec_to_kernel_coeffs
        mo_values = []
        for imo in occupied_idx:
            coeffs = evec_to_kernel_coeffs(
                evecs[imo], len(detailed['coords_bohr']),
                detailed['species_per_atom'], detailed['species_names'], basis
            )
            psi = projector.project_orbital_points(
                points_ang.astype(np.float32), coeffs,
                np.array(norb_per_atom, dtype=np.int32), atoms_dict
            )
            mo_values.append(psi.reshape(npoints, npoints))
        
        density = np.zeros_like(mo_values[0])
        for i, imo in enumerate(occupied_idx):
            density += occupations[imo] * (mo_values[i] ** 2)
        
        output = Path(args.work_dir) / f'{args.output_prefix}_density_z{z_offset:.1f}.png'
        plotUtils.plot_2d_array(density, extent, atom_coords_ang,
                                f'Total density (sparse) z={z_offset:.1f}A', output, dpi=args.dpi, cmap='magma')


if __name__ == '__main__':
    main()
