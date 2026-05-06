"""
Test utilities for waveplot comparison and debugging.

Provides high-level functions for comparing libwaveplot and OpenCL evaluations,
including plotting utilities and RMS error calculations.
"""

import numpy as np
from pathlib import Path


def compute_rms_error(vals1, vals2):
    """
    Compute RMS error between two arrays.
    
    Args:
        vals1: array of values
        vals2: array of values (same shape)
    
    Returns:
        rms: float, root-mean-square error
        max_err: float, maximum absolute error
    """
    diff = vals1 - vals2
    rms = np.sqrt(np.mean(diff**2))
    max_err = np.max(np.abs(diff))
    return rms, max_err


def compare_point_evaluations(wp_vals, ocl_vals, mo_indices, energies, homo):
    """
    Compare libwaveplot and OpenCL point evaluations.
    
    Args:
        wp_vals: list of arrays, each [npoints] from libwaveplot
        ocl_vals: list of arrays, each [npoints] from OpenCL
        mo_indices: list of MO indices
        energies: list of energies (eV)
        homo: HOMO index
    
    Returns:
        results: list of dicts with rms, max_err for each MO
    """
    results = []
    for i, (wp, ocl) in enumerate(zip(wp_vals, ocl_vals)):
        rms, max_err = compute_rms_error(wp, ocl)
        mo_idx = mo_indices[i]
        tag = " [HOMO]" if mo_idx == homo else (" [LUMO]" if mo_idx == homo+1 else "")
        results.append({
            'mo_index': mo_idx,
            'tag': tag,
            'energy': energies[i],
            'rms': rms,
            'max_err': max_err
        })
    return results


def compare_grid_evaluations(grid_ref, grid_lib, mo_indices, energies, homo):
    """
    Compare reference cube and libwaveplot grid evaluations.
    
    Args:
        grid_ref: list of arrays, each (nx, ny, nz) from reference cube
        grid_lib: list of arrays, each (nx, ny, nz) from libwaveplot
        mo_indices: list of MO indices
        energies: list of energies (eV)
        homo: HOMO index
    
    Returns:
        results: list of dicts with rms, max_err for each MO
    """
    results = []
    for i, (ref, lib) in enumerate(zip(grid_ref, grid_lib)):
        rms, max_err = compute_rms_error(ref, lib)
        mo_idx = mo_indices[i]
        tag = " [HOMO]" if mo_idx == homo else (" [LUMO]" if mo_idx == homo+1 else "")
        results.append({
            'mo_index': mo_idx,
            'tag': tag,
            'energy': energies[i],
            'rms': rms,
            'max_err': max_err
        })
    return results


def print_comparison_results(results, method_name=""):
    """
    Print comparison results in a formatted table.
    
    Args:
        results: list of dicts from compare_*_evaluations
        method_name: string to identify the method (e.g., "orb2points")
    """
    print(f"\n  RMS ({method_name}):")
    for r in results:
        print(f"    MO {r['mo_index']:3d}{r['tag']:8s} E={r['energy']:8.3f}eV  RMS={r['rms']:.3e}  max={r['max_err']:.3e}")


def generate_2d_point_grid(plane='xy', npoints=64, z_offset=0.0, xy_range=None):
    """
    Generate 2D grid of points for orbital evaluation.
    
    Args:
        plane: 'xy', 'xz', or 'yz'
        npoints: number of points per axis
        z_offset: fixed coordinate for out-of-plane axis (Å)
        xy_range: optional (min, max) for in-plane axes (Å)
    
    Returns:
        points: (npoints*npoints, 3) array in Angstrom
        extent: [xmin, xmax, ymin, ymax] for plotting
    """
    if xy_range is None:
        xy_range = (-3.0, 3.0)
    
    x = np.linspace(xy_range[0], xy_range[1], npoints)
    y = np.linspace(xy_range[0], xy_range[1], npoints)
    X, Y = np.meshgrid(x, y)
    
    if plane == 'xy':
        points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_offset)])
        extent = [xy_range[0], xy_range[1], xy_range[0], xy_range[1]]
    elif plane == 'xz':
        points = np.column_stack([X.ravel(), np.full(X.size, z_offset), Y.ravel()])
        extent = [xy_range[0], xy_range[1], xy_range[0], xy_range[1]]
    elif plane == 'yz':
        points = np.column_stack([np.full(X.size, z_offset), X.ravel(), Y.ravel()])
        extent = [xy_range[0], xy_range[1], xy_range[0], xy_range[1]]
    else:
        raise ValueError(f"Unknown plane: {plane}")
    
    return points, extent


def generate_1d_z_scan(npoints=100, z_range=(-3.0, 3.0)):
    """
    Generate 1D scan along z-axis.
    
    Args:
        npoints: number of points
        z_range: (zmin, zmax) in Angstrom
    
    Returns:
        points: (npoints, 3) array with x=y=0
        z_vals: (npoints,) z coordinates
    """
    z = np.linspace(z_range[0], z_range[1], npoints)
    points = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    return points, z
