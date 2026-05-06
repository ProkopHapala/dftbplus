#!/usr/bin/env python3
"""Compare OpenCL waveplot output with WAVEPLOT Fortran cube files."""
import numpy as np
import struct
from pathlib import Path

BOHR = 0.5291772109  # Å/Bohr

def read_cube(fname):
    """Parse Gaussian cube file."""
    with open(fname) as f:
        lines = f.readlines()
    
    # Header
    comment1 = lines[0].strip()
    comment2 = lines[1].strip()
    
    # Line 3: natoms, origin
    natoms, ox, oy, oz = map(float, lines[2].split())
    natoms = int(natoms)
    origin = np.array([ox, oy, oz]) * BOHR  # Convert Bohr to Å
    
    # Lines 4-6: grid definition
    npts = []
    vecs = []
    for i in range(3):
        parts = list(map(float, lines[3+i].split()))
        npts.append(int(parts[0]))
        vecs.append(np.array(parts[1:]) * BOHR)
    npts = np.array(npts)
    vecs = np.array(vecs)  # shape (3,3)
    
    # Atom lines
    atoms = []
    for i in range(natoms):
        parts = lines[6+i].split()
        Z = int(parts[0])
        x, y, z = map(float, parts[1:4])
        atoms.append({'Z': Z, 'pos': np.array([x, y, z]) * BOHR})
    
    # Grid values (rest of file, 6 values per line)
    grid_vals = []
    for line in lines[6+natoms:]:
        grid_vals.extend(map(float, line.split()))
    grid = np.array(grid_vals).reshape(npts[0], npts[1], npts[2], order='F')  # Fortran order
    
    return {
        'comment': comment1,
        'origin': origin,
        'npts': npts,
        'vecs': vecs,
        'atoms': atoms,
        'grid': grid
    }

def read_opencl_grid(grid_file):
    """Read OpenCL grid from numpy save."""
    return np.load(grid_file)

def interpolate_opencl_to_cube(opencl_grid, opencl_spec, cube_spec):
    """Interpolate OpenCL grid to cube coordinates using trilinear interpolation."""
    # OpenCL grid: uniform cubic with origin and step
    o_origin = opencl_spec['origin']
    o_step = opencl_spec['step']
    o_ngrid = opencl_spec['ngrid']
    
    # Cube grid: possibly non-orthogonal, defined by origin + npts * vecs
    c_origin = cube_spec['origin']
    c_npts = cube_spec['npts']
    c_vecs = cube_spec['vecs']
    
    # Generate cube coordinates
    ix, iy, iz = np.meshgrid(np.arange(c_npts[0]), np.arange(c_npts[1]), np.arange(c_npts[2]), indexing='ij')
    c_coords = c_origin + (ix[:,None,None] * c_vecs[0] + 
                           iy[:,None,None] * c_vecs[1] + 
                           iz[:,None,None] * c_vecs[2])
    
    # Map to OpenCL grid indices
    rel_pos = (c_coords - o_origin) / o_step
    grid_idx = rel_pos.astype(int)
    
    # Check bounds
    valid = (grid_idx[:,0] >= 0) & (grid_idx[:,0] < o_ngrid[0]) & \
            (grid_idx[:,1] >= 0) & (grid_idx[:,1] < o_ngrid[1]) & \
            (grid_idx[:,2] >= 0) & (grid_idx[:,2] < o_ngrid[2])
    
    # Nearest-neighbor interpolation (for simplicity)
    interpolated = np.zeros(c_npts)
    interpolated[valid] = opencl_grid[grid_idx[valid,0], grid_idx[valid,1], grid_idx[valid,2]]
    
    return interpolated, valid

def compare_grids(opencl_grid, cube_grid, valid_mask, name):
    """Compute error metrics."""
    # Only compare valid points
    o_vals = opencl_grid[valid_mask]
    c_vals = cube_grid[valid_mask]
    
    diff = o_vals - c_vals
    rms = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))
    corr = np.corrcoef(o_vals, c_vals)[0,1] if len(o_vals) > 1 else 1.0
    
    print(f"\n{name}:")
    print(f"  Valid points: {np.sum(valid_mask)}/{valid_mask.size}")
    print(f"  OpenCL range: [{o_vals.min():.6e}, {o_vals.max():.6e}]")
    print(f"  WAVEPLOT range: [{c_vals.min():.6e}, {c_vals.max():.6e}]")
    print(f"  RMS error: {rms:.6e}")
    print(f"  Max absolute error: {max_abs:.6e}")
    print(f"  Correlation: {corr:.6f}")
    
    return {'rms': rms, 'max_abs': max_abs, 'corr': corr}

def main():
    cube_dir = Path('dftb_h2o')
    opencl_dir = Path('waveplot_output/H2O')
    
    # OpenCL grid spec (from test output)
    opencl_spec = {
        'origin': np.array([-4.725, -4.125, -3.525]),
        'step': 0.15,
        'ngrid': np.array([64, 56, 48])
    }
    
    print("=" * 60)
    print("WAVEPLOT vs OpenCL Grid Comparison")
    print("=" * 60)
    
    # Compare all 6 MOs
    all_errors = []
    for imo in range(1, 7):
        cube_file = cube_dir / f'wp-1-1-{imo}-real.cube'
        opencl_file = opencl_dir / f'opencl_MO{imo}.npy'
        
        if not cube_file.exists():
            print(f"\nSkipping MO{imo}: cube file not found")
            continue
        if not opencl_file.exists():
            print(f"\nSkipping MO{imo}: OpenCL .npy not found")
            continue
        
        cube_data = read_cube(cube_file)
        opencl_grid = np.load(opencl_file)
        
        print(f"\n{'='*60}")
        print(f"MO{imo}: {cube_file.name}")
        print(f"  WAVEPLOT origin (Å): {cube_data['origin']}")
        print(f"  OpenCL origin (Å):   {opencl_spec['origin']}")
        print(f"  Grid size: {cube_data['npts']}")
        
        # Direct comparison (grids should have same size and origin)
        if not np.allclose(cube_data['origin'], opencl_spec['origin'], atol=1e-3):
            print(f"  WARNING: Origin mismatch!")
            print(f"    Diff: {cube_data['origin'] - opencl_spec['origin']}")
        
        if not np.array_equal(cube_data['npts'], opencl_spec['ngrid']):
            print(f"  WARNING: Grid size mismatch!")
            print(f"    WAVEPLOT: {cube_data['npts']}")
            print(f"    OpenCL:   {opencl_spec['ngrid']}")
            continue
        
        # Cube grid is in Fortran order (x fastest), OpenCL is C order (z fastest)
        # Reshape cube to match
        cube_grid = cube_data['grid']  # Already reshaped to (nx, ny, nz) with order='F'
        
        # Try both orders
        print(f"  Trying C-order comparison...")
        errors_c = compare_grids(opencl_grid, cube_grid, np.ones_like(opencl_grid, dtype=bool), f"MO{imo} (C-order)")
        
        print(f"  Trying transposed comparison...")
        # Transpose to match Fortran order: (nx, ny, nz) -> (nz, ny, nx)
        opencl_transposed = opencl_grid.transpose(2, 1, 0)  # (nx, ny, nz) -> (nz, ny, nx)
        # Create valid mask for transposed shape
        valid_t = np.ones_like(opencl_transposed, dtype=bool)
        errors_f = compare_grids(opencl_transposed, cube_grid, valid_t, f"MO{imo} (transposed)")
        
        # Use whichever has better correlation
        if errors_f['corr'] > errors_c['corr']:
            print(f"  Using transposed (correlation {errors_f['corr']:.6f} > {errors_c['corr']:.6f})")
            errors = errors_f
        else:
            print(f"  Using C-order (correlation {errors_c['corr']:.6f} > {errors_f['corr']:.6f})")
            errors = errors_c
        
        all_errors.append((imo, errors))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for imo, err in all_errors:
        print(f"MO{imo}: RMS={err['rms']:.6e}, Max={err['max_abs']:.6e}, Corr={err['corr']:.6f}")
    
    avg_rms = np.mean([e['rms'] for _, e in all_errors])
    avg_max = np.mean([e['max_abs'] for _, e in all_errors])
    avg_corr = np.mean([e['corr'] for _, e in all_errors])
    print(f"\nAverage: RMS={avg_rms:.6e}, Max={avg_max:.6e}, Corr={avg_corr:.6f}")
    
    # Tolerance check
    if avg_rms < 1e-5 and avg_max < 1e-4 and avg_corr > 0.99:
        print("\n✓ PASS: OpenCL matches WAVEPLOT within tolerance")
    else:
        print("\n✗ FAIL: OpenCL differs from WAVEPLOT beyond tolerance")

if __name__ == '__main__':
    main()
