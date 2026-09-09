"""
Core projection utilities for DFTB+ density and orbital projection.

This module encapsulates the DFTB+ calculation, OpenCL projector setup,
and projection logic so that CLI scripts remain simple.
"""

import os
import numpy as np
from pathlib import Path

from .OCL.DFTBplusParser import (
    parse_basis_hsd_ang, parse_detailed_xml_custom, evec_to_kernel_coeffs
)
from .OCL.Grid import setup_gridprojector_from_dftb

BOHR2ANG = 0.5291772109

def find_libdftbcore():
    """Find libdftbcore.so in standard locations."""
    lib_paths = [
        Path(__file__).parent.parent / '_build' / 'app' / 'dftbcore' / 'libdftbcore.so',
        Path(__file__).parent.parent / 'build' / 'libdftbcore.so',
        Path(__file__).parent.parent / 'build' / 'lib' / 'libdftbcore.so',
    ]
    for p in lib_paths:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"libdftbcore.so not found in {lib_paths}")

def run_dftb_calculation(work_dir, lib_path=None):
    """Run DFTB+ SCF calculation and return eigenvectors, DM, occupations, geometry.
    
    Args:
        work_dir: Path to directory with dftb_in.hsd and geometry
        lib_path: Optional path to libdftbcore.so
        
    Returns:
        dict with keys: evecs, dm_dense, s_dense, occupations, detailed, energy
    """
    from .DFTBcore import DFTBcore
    
    work_dir = Path(work_dir)
    if lib_path is None:
        lib_path = find_libdftbcore()
    
    orig_dir = os.getcwd()
    os.chdir(work_dir)
    
    try:
        dftb = DFTBcore(libpath=str(lib_path))
        input_file = work_dir / 'dftb_in.hsd'
        dftb.init(str(input_file))
        dftb.enable_matrix_collection(dm=True, h=False, s=True)
        energy = dftb.run_scf()
        
        evecs, eigenvals = dftb.get_eigvecs_dense()
        dm_dense = dftb.get_dm_dense()
        s_dense = dftb.get_s_dense()
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

def setup_projector(detailed, basis, max_shells=None, verbosity=0):
    """Setup OpenCL projector from DFTB+ data.
    
    Args:
        detailed: dict from parse_detailed_xml_custom
        basis: list of basis dicts from parse_basis_hsd_ang
        max_shells: optional max shells (auto-detected if None)
        verbosity: int verbosity level
        
    Returns:
        projector, atoms_dict, norb_per_atom, orb_offsets, max_shells
    """
    dftb_data = {
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
    if max_shells is None:
        max_shells = max_l + 1
    
    orb_offsets = np.zeros(natoms + 1, dtype=np.int32)
    orb_offsets[1:] = np.cumsum(norb_per_atom)
    
    projector, atoms_dict = setup_gridprojector_from_dftb(
        dftb_data, basis, verbosity=verbosity, max_shells=max_shells
    )
    
    return projector, atoms_dict, norb_per_atom, orb_offsets, max_shells

def get_orbital_indices(occupations, mo_list=None, relative_to_homo=False):
    """Get list of orbital indices from user input.
    
    Args:
        occupations: (norb,) array of occupations
        mo_list: list of indices or strings like 'HOMO-1', 'HOMO', 'LUMO'
        relative_to_homo: if True, interpret mo_list as relative to HOMO
        
    Returns:
        list of 0-based orbital indices
    """
    occupied_idx = [i for i, occ in enumerate(occupations) if occ > 0]
    if not occupied_idx:
        return []
    homo_idx = occupied_idx[-1]
    
    if mo_list is None:
        return occupied_idx
    
    result = []
    for mo in mo_list:
        if isinstance(mo, str):
            mo = mo.strip().upper()
            if mo == 'HOMO':
                result.append(homo_idx)
            elif mo == 'LUMO':
                result.append(homo_idx + 1)
            elif mo.startswith('HOMO-'):
                offset = int(mo.split('-')[1])
                result.append(homo_idx - offset)
            elif mo.startswith('LUMO+'):
                offset = int(mo.split('+')[1])
                result.append(homo_idx + 1 + offset)
            else:
                result.append(int(mo) - 1)  # 1-based to 0-based
        else:
            idx = int(mo)
            if relative_to_homo:
                result.append(homo_idx + idx)
            else:
                result.append(idx - 1)  # 1-based to 0-based
    
    return result

def project_orbital_at_points(projector, points_ang, coeffs, norb_per_atom, orb_offsets, atoms_dict, method='dense'):
    """Project a single orbital at given points.
    
    Args:
        projector: OpenCL projector
        points_ang: (npoints, 3) array in Angstrom
        coeffs: orbital coefficients (dense or sparse format)
        norb_per_atom: (natoms,) array
        orb_offsets: (natoms+1,) array
        atoms_dict: atoms dict for projector
        method: 'dense' or 'sparse'
        
    Returns:
        (npoints,) array of orbital values
    """
    if method == 'dense':
        return projector.project_orbital_dense_points(
            points_ang.astype(np.float32), coeffs.astype(np.float32),
            norb_per_atom, orb_offsets, atoms_dict
        )
    else:
        return projector.project_orbital_points(
            points_ang.astype(np.float32), coeffs,
            np.array(norb_per_atom, dtype=np.int32), atoms_dict
        )

def project_density_at_points(projector, points_ang, dm, norb_per_atom, orb_offsets, atoms_dict, method='dense'):
    """Project density matrix at given points.
    
    Args:
        projector: OpenCL projector
        points_ang: (npoints, 3) array in Angstrom
        dm: density matrix (norb, norb)
        norb_per_atom: (natoms,) array
        orb_offsets: (natoms+1,) array
        atoms_dict: atoms dict for projector
        method: 'dense' or 'sparse'
        
    Returns:
        (npoints,) array of density values
    """
    if method == 'dense':
        return projector.project_density_dense_points(
            points_ang.astype(np.float32), dm.astype(np.float32),
            norb_per_atom, orb_offsets, atoms_dict
        )
    else:
        raise NotImplementedError("Sparse density projection not implemented")

def project_orbital_on_grid(projector, coeffs, norb_per_atom, orb_offsets, atoms_dict, grid_spec, method='dense'):
    """Project orbital on a 3D grid.
    
    Args:
        projector: OpenCL projector
        coeffs: orbital coefficients
        norb_per_atom: (natoms,) array
        orb_offsets: (natoms+1,) array
        atoms_dict: atoms dict for projector
        grid_spec: dict with origin, dA, dB, dC, ngrid
        method: 'dense' or 'sparse'
        
    Returns:
        (nx, ny, nz) array of orbital values
    """
    if method == 'dense':
        return projector.project_orbital_dense(
            coeffs.astype(np.float32), norb_per_atom, orb_offsets, atoms_dict, grid_spec
        )
    else:
        raise NotImplementedError("Sparse grid projection not implemented")

def project_density_on_grid(projector, dm, norb_per_atom, orb_offsets, atoms_dict, grid_spec, method='dense'):
    """Project density matrix on a 3D grid.
    
    Args:
        projector: OpenCL projector
        dm: density matrix (norb, norb)
        norb_per_atom: (natoms,) array
        orb_offsets: (natoms+1,) array
        atoms_dict: atoms dict for projector
        grid_spec: dict with origin, dA, dB, dC, ngrid
        method: 'dense' or 'sparse'
        
    Returns:
        (nx, ny, nz) array of density values
    """
    if method == 'dense':
        return projector.project_density_dense(
            dm.astype(np.float32), norb_per_atom, orb_offsets, atoms_dict, grid_spec
        )
    else:
        raise NotImplementedError("Sparse grid projection not implemented")

def density_from_orbitals(mo_values, occupied_idx, occupations):
    """Compute density from orbital values and occupations.
    
    Args:
        mo_values: list of (npoints,) or (nx, ny) arrays
        occupied_idx: list of orbital indices
        occupations: (norb,) array of occupations
        
    Returns:
        density array with same shape as mo_values[0]
    """
    density = np.zeros_like(mo_values[0])
    for i, imo in enumerate(occupied_idx):
        occ = occupations[imo]
        density += occ * (mo_values[i] ** 2)
    return density
