"""
Grid generation utilities for DFTB+ density and orbital projection.
"""

import numpy as np

BOHR2ANG = 0.5291772109

def compute_bbox_margin(coords_bohr, margin=4.0):
    """Compute bounding box from atomic coordinates with margin.
    
    Args:
        coords_bohr: (natoms, 3) array in Bohr
        margin: margin in Angstrom (default 4.0)
        
    Returns:
        rmin, rmax: float bounds in Angstrom
    """
    coords_ang = coords_bohr * BOHR2ANG
    rmin = float(coords_ang.min()) - margin
    rmax = float(coords_ang.max()) + margin
    return rmin, rmax

def estimate_npoints_from_step(bbox_range, step):
    """Estimate number of points from step size and range.
    
    Args:
        bbox_range: float total range in Angstrom
        step: float step size in Angstrom
        
    Returns:
        int: number of points
    """
    return int(np.ceil(bbox_range / step))

def generate_2d_point_grid(plane, npoints, z_offset, extent_range):
    """Generate a 2D grid of points in the XY plane at a given z offset.
    
    Args:
        plane: 'xy', 'xz', or 'yz' plane
        npoints: int, number of points in each direction
        z_offset: float, z offset in Angstrom
        extent_range: (rmin, rmax) tuple in Angstrom
        
    Returns:
        points_ang: (npoints*npoints, 3) array of points in Angstrom
        extent: [rmin, rmax, rmin, rmax] for matplotlib
    """
    rmin, rmax = extent_range
    x = np.linspace(rmin, rmax, npoints)
    y = np.linspace(rmin, rmax, npoints)
    X, Y = np.meshgrid(x, y)
    
    if plane == 'xy':
        points_ang = np.column_stack([X.ravel(), Y.ravel(), np.full(npoints*npoints, z_offset)])
    elif plane == 'xz':
        points_ang = np.column_stack([X.ravel(), np.full(npoints*npoints, z_offset), Y.ravel()])
    else:  # yz
        points_ang = np.column_stack([np.full(npoints*npoints, z_offset), X.ravel(), Y.ravel()])
    
    extent = [rmin, rmax, rmin, rmax]
    return points_ang.astype(np.float32), extent

def build_grid_spec_2d(extent, step, z_offset):
    """Build a grid_spec dict for 2D XY plane projection.
    
    Args:
        extent: [rmin, rmax, rmin, rmax] in Angstrom
        step: float grid step in Angstrom
        z_offset: float z offset in Angstrom
        
    Returns:
        grid_spec dict for OpenCL projection
    """
    rmin = extent[0]
    rmax = extent[1]
    npoints = int(np.ceil((rmax - rmin) / step))
    return {
        'origin': np.array([rmin, rmin, z_offset], dtype=np.float32),
        'dA': np.array([step, 0.0, 0.0], dtype=np.float32),
        'dB': np.array([0.0, step, 0.0], dtype=np.float32),
        'dC': np.array([0.0, 0.0, 1.0], dtype=np.float32),
        'ngrid': np.array([npoints, npoints, 1], dtype=np.int32),
    }
