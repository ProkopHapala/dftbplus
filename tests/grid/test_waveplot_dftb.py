#!/usr/bin/env python3
"""
DFTB+ OpenCL Waveplot Test Script

Tests orbital projection and density computation using OpenCL.
Supports H2O (cluster) and PTCDA (periodic) test systems.

Usage:
    python test_waveplot_dftb.py --system H2O --step 0.2
    python test_waveplot_dftb.py --system PTCDA --z-height 2.0 --step 0.15
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pyBall.OCL.DFTBplusParser import DFTBplusParser, compute_sto_radial, parse_basis_hsd_ang
from pyBall.OCL.Grid import GridProjector
from pyBall import plotUtils as pu


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DFTB+ OpenCL Waveplot Test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # System selection
    parser.add_argument('--system', type=str, default='H2O',
                       choices=['H2O', 'PTCDA'],
                       help='Test system to run')
    parser.add_argument('--xyz', type=str, default=None,
                       help='Path to XYZ file (overrides --system)')
    
    # Grid parameters
    parser.add_argument('--step', type=float, default=0.15,
                       help='Grid spacing in Angstrom')
    parser.add_argument('--margin', type=float, default=3.0,
                       help='Margin around molecule in Angstrom')
    parser.add_argument('--z-height', type=float, default=None,
                       help='Height above molecular plane for 2D slice (PTCDA only)')
    parser.add_argument('--ngrid', type=int, nargs=3, default=None, metavar=('NX', 'NY', 'NZ'),
                       help='Explicit grid dimensions (overrides auto)')
    
    # Point evaluation (Method 2)
    parser.add_argument('--points', action='store_true',
                       help='Method 2: evaluate at points along z-axis instead of 3D grid')
    parser.add_argument('--z-range', type=float, nargs=2, default=[-3.0, 3.0],
                       metavar=('ZMIN', 'ZMAX'), help='Z-range for point evaluation (Å)')
    parser.add_argument('--npoints', type=int, default=301, help='Number of points for evaluation')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='waveplot_output',
                       help='Output directory for plots')
    parser.add_argument('--prefix', type=str, default='dftb_waveplot',
                       help='Filename prefix for output files')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for saved figures')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    
    # Plotting options
    parser.add_argument('--plot-mos', type=int, nargs='+', default=None,
                       help='MO indices to plot (default: HOMO-2 to LUMO+2)')
    parser.add_argument('--plot-density', action='store_true',
                       help='Plot total electron density')
    parser.add_argument('--plot-slices', action='store_true',
                       help='Plot 3 orthogonal slices through 3D grid')
    parser.add_argument('--cmap-orbital', type=str, default='RdBu_r',
                       help='Colormap for orbital plots (signed)')
    parser.add_argument('--cmap-density', type=str, default='hot',
                       help='Colormap for density plots (positive)')
    
    # OpenCL options
    parser.add_argument('--nmax-atom', type=int, default=64,
                       help='Max atoms per task block')
    parser.add_argument('--gpu-tasks', action='store_true',
                       help='Use GPU-based task builder')
    
    # Validation options
    parser.add_argument('--vmin-min', type=float, default=1e-6,
                       help='Minimum expected absolute value')
    parser.add_argument('--vmax-max', type=float, default=1e6,
                       help='Maximum expected absolute value')
    parser.add_argument('--check-nan', action='store_true', default=True,
                       help='Check for NaN values')
    
    # Verbosity
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase verbosity (use -v, -vv, or -vvv)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def validate_grid(grid, name, args):
    """
    Validate grid values for numerical issues.
    
    Args:
        grid: numpy array to validate
        name: identifier for error messages
        args: command line arguments
    
    Returns:
        vmin, vmax: value range
    
    Raises:
        ValueError: if validation fails
    """
    vmin = float(grid.min())
    vmax = float(grid.max())
    
    print(f"  {name}: vmin={vmin:.6e}, vmax={vmax:.6e}")
    
    # Check NaN
    if args.check_nan and (np.isnan(vmin) or np.isnan(vmax)):
        raise ValueError(f"{name}: NaN values detected! Grid is invalid.")
    
    # Check for all zeros
    if abs(vmin) < args.vmin_min and abs(vmax) < args.vmin_min:
        raise ValueError(f"{name}: All values too small (|v| < {args.vmin_min}). Grid may be empty.")
    
    # Check for unreasonably large values
    if abs(vmin) > args.vmax_max or abs(vmax) > args.vmax_max:
        raise ValueError(f"{name}: Values too large (|v| > {args.vmax_max}). Possible overflow.")
    
    # Check for Inf
    if np.isinf(vmin) or np.isinf(vmax):
        raise ValueError(f"{name}: Inf values detected! Grid is invalid.")
    
    return vmin, vmax


def compute_grid_extent(coords, margin=3.0, axes=(0, 1)):
    """
    Compute plot extent for given axes with margin.
    
    Args:
        coords: (natoms, 3) atomic coordinates
        margin: margin to add around molecule
        axes: which axes to use (e.g., (0, 1) for xy)
    
    Returns:
        extent: [xmin, xmax, ymin, ymax] for imshow
    """
    ax1, ax2 = axes
    xmin = coords[:, ax1].min() - margin
    xmax = coords[:, ax1].max() + margin
    ymin = coords[:, ax2].min() - margin
    ymax = coords[:, ax2].max() + margin
    return [xmin, xmax, ymin, ymax]


def plot_orbital_2d(grid_2d, coords, extent, title, fname, args, is_density=False):
    """
    Plot 2D orbital or density map with atoms.
    
    Args:
        grid_2d: (nx, ny) 2D grid data
        coords: (natoms, 3) atomic coordinates
        extent: plot extent [xmin, xmax, ymin, ymax]
        title: plot title
        fname: output filename
        args: command line arguments
        is_density: if True, use density colormap (always positive)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmin, vmax = grid_2d.min(), grid_2d.max()
    
    if is_density:
        # Density is always positive
        cmap = args.cmap_density
        im = ax.imshow(grid_2d.T, origin='lower', extent=extent, 
                      cmap=cmap, aspect='equal', vmin=0, vmax=vmax)
    else:
        # Orbital is signed
        cmap = args.cmap_orbital
        vabs = max(abs(vmin), abs(vmax))
        if vabs < 1e-30:
            vabs = 1.0
        im = ax.imshow(grid_2d.T, origin='lower', extent=extent,
                      cmap=cmap, aspect='equal', vmin=-vabs, vmax=vabs)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if is_density:
        cbar.set_label('Electron Density (e/Å³)')
    else:
        cbar.set_label('Wavefunction Amplitude')
    
    # Plot atoms
    ax1, ax2 = 0, 1  # xy plane
    pu.plotAtoms(apos=coords, es=None, sizes=100, colors='gray', 
                marker='o', axes=(ax1, ax2))
    
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_title(title)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    fig.tight_layout()
    fig.savefig(fname, dpi=args.dpi, bbox_inches='tight')
    print(f"  Saved: {fname}")
    
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def plot_orbital_slices(grid_3d, coords, grid_spec, title_prefix, fname_prefix, args):
    """
    Plot 3 orthogonal slices through 3D grid.
    
    Args:
        grid_3d: (nx, ny, nz) 3D grid data
        coords: atomic coordinates
        grid_spec: grid specification dict
        title_prefix: prefix for plot titles
        fname_prefix: prefix for output filenames
        args: command line arguments
    """
    nx, ny, nz = grid_3d.shape
    origin = grid_spec['origin']
    dA, dB, dC = grid_spec['dA'], grid_spec['dB'], grid_spec['dC']
    
    # Slice indices (middle)
    ix, iy, iz = nx // 2, ny // 2, nz // 2
    
    # Extract slices
    slice_xy = grid_3d[:, :, iz]  # z = const
    slice_xz = grid_3d[:, iy, :]  # y = const
    slice_yz = grid_3d[ix, :, :]  # x = const
    
    # Compute extents
    extent_xy = [
        origin[0], origin[0] + nx * dA[0],
        origin[1], origin[1] + ny * dB[1]
    ]
    extent_xz = [
        origin[0], origin[0] + nx * dA[0],
        origin[2], origin[2] + nz * dC[2]
    ]
    extent_yz = [
        origin[1], origin[1] + ny * dB[1],
        origin[2], origin[2] + nz * dC[2]
    ]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin, vmax = grid_3d.min(), grid_3d.max()
    vabs = max(abs(vmin), abs(vmax))
    if vabs < 1e-30:
        vabs = 1.0
    
    # XY slice
    im0 = axes[0].imshow(slice_xy.T, origin='lower', extent=extent_xy,
                        cmap=args.cmap_orbital, vmin=-vabs, vmax=vabs)
    axes[0].set_title(f'{title_prefix} - XY slice (z={origin[2] + iz*dC[2]:.2f}Å)')
    axes[0].set_xlabel('x (Å)')
    axes[0].set_ylabel('y (Å)')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # XZ slice
    im1 = axes[1].imshow(slice_xz.T, origin='lower', extent=extent_xz,
                        cmap=args.cmap_orbital, vmin=-vabs, vmax=vabs)
    axes[1].set_title(f'{title_prefix} - XZ slice (y={origin[1] + iy*dB[1]:.2f}Å)')
    axes[1].set_xlabel('x (Å)')
    axes[1].set_ylabel('z (Å)')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # YZ slice
    im2 = axes[2].imshow(slice_yz.T, origin='lower', extent=extent_yz,
                        cmap=args.cmap_orbital, vmin=-vabs, vmax=vabs)
    axes[2].set_title(f'{title_prefix} - YZ slice (x={origin[0] + ix*dA[0]:.2f}Å)')
    axes[2].set_xlabel('y (Å)')
    axes[2].set_ylabel('z (Å)')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    fig.tight_layout()
    fname = f"{fname_prefix}_slices.png"
    fig.savefig(fname, dpi=args.dpi, bbox_inches='tight')
    print(f"  Saved: {fname}")
    
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def extract_plane_at_z(grid_3d, grid_spec, z_target):
    """
    Extract 2D plane from 3D grid at given z height.
    
    Uses linear interpolation between grid planes.
    
    Args:
        grid_3d: (nx, ny, nz) 3D grid
        grid_spec: grid specification
        z_target: target z coordinate in Angstrom
    
    Returns:
        grid_2d: (nx, ny) 2D slice
    """
    nx, ny, nz = grid_3d.shape
    origin = grid_spec['origin']
    dC = grid_spec['dC']
    
    # Find z index
    z_grid = origin[2] + np.arange(nz) * dC[2]
    
    # Find surrounding planes
    if z_target <= z_grid[0]:
        return grid_3d[:, :, 0]
    if z_target >= z_grid[-1]:
        return grid_3d[:, :, -1]
    
    iz = np.searchsorted(z_grid, z_target) - 1
    if iz < 0:
        iz = 0
    if iz >= nz - 1:
        iz = nz - 2
    
    # Linear interpolation
    z0, z1 = z_grid[iz], z_grid[iz + 1]
    t = (z_target - z0) / (z1 - z0)
    
    grid_2d = (1 - t) * grid_3d[:, :, iz] + t * grid_3d[:, :, iz + 1]
    
    return grid_2d


def run_h2o_test(args):
    """Run H2O cluster test using real DFTB+ eigenvectors."""
    print("=" * 60)
    print("Running H2O Test (Cluster)")
    print("=" * 60)
    
    # DFTB+ calculation directory (contains eigenvec.bin + detailed.xml)
    dftb_dir = Path(__file__).parent / 'dftb_h2o'
    output_dir = Path(args.output_dir) / 'H2O'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Geometry: read directly from dftb_h2o/geom.xyz (Angstrom) ----
    # DFTB+ uses Bohr internally; detailed.xml stores Bohr coords
    # We use Angstrom throughout (geom.xyz was in Angstrom)
    BOHR = 0.5291772109  # Å/Bohr
    geom_xyz = dftb_dir / 'geom.xyz'
    with open(geom_xyz) as f:
        lines = f.readlines()
    natoms = int(lines[0])
    atom_symbols = []
    coords = []
    for i in range(2, 2 + natoms):
        parts = lines[i].split()
        atom_symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    coords = np.array(coords, dtype=np.float64)  # Angstrom
    print(f"\n[Geometry]  {natoms} atoms: {atom_symbols}")
    for i, (s, r) in enumerate(zip(atom_symbols, coords)):
        print(f"  {i} {s}  {r[0]:8.4f} {r[1]:8.4f} {r[2]:8.4f} Å")

    # ---- Read eigenvectors from eigenvec.bin ----
    # Format: 4-byte int32 identity + (nstates * norb) float64 row-major
    # H2O basis: O{s,py,pz,px}, H1{s}, H2{s} => 6 AOs, 6 MOs
    import struct
    with open(dftb_dir / 'eigenvec.bin', 'rb') as f:
        raw = f.read()
    identity = struct.unpack_from('i', raw, 0)[0]
    norb_dftb = 6   # O:4 + H:1 + H:1
    nstates   = 6
    evecs = np.frombuffer(raw[4:], dtype=np.float64).reshape(nstates, norb_dftb)
    # evecs[mo, ao] — DFTB+ AO order: O:s,py,pz,px | H1:s | H2:s
    energies_ev = [-23.0459, -11.0052, -8.7497, -7.0553, 9.0574, 13.8781]
    occs         = [2, 2, 2, 2, 0, 0]
    print(f"\n[Eigenvectors] identity={identity}")
    for i in range(nstates):
        print(f"  MO{i+1} E={energies_ev[i]:7.3f} eV occ={occs[i]}  {evecs[i]}")

    # ---- Build per-atom padded coeffs in kernel order [px,py,pz,s] ----
    # Atom layout:  ia=0 O  (AOs 0..3: s,py,pz,px)
    #               ia=1 H1 (AO  4: s)
    #               ia=2 H2 (AO  5: s)
    # Kernel slot:  [px, py, pz, s]  = indices [3,1,2,0] in DFTB order for sp atoms
    # H only has s; px,py,pz slots stay 0
    numorb_max = 4
    def dftb_to_kernel_coeffs(evec, atom_symbols):
        """Convert DFTB+ MO coeffs (flat AO order) to (natoms,4) [px,py,pz,s]."""
        na = len(atom_symbols)
        c = np.zeros((na, numorb_max), dtype=np.float32)
        ao = 0
        for ia, sym in enumerate(atom_symbols):
            if sym == 'O':
                # DFTB order: s, py, pz, px
                s_c, py_c, pz_c, px_c = evec[ao:ao+4]
                c[ia] = [px_c, py_c, pz_c, s_c]  # [px,py,pz,s]
                ao += 4
            elif sym == 'H':
                c[ia] = [0, 0, 0, evec[ao]]  # H only has s
                ao += 1
            elif sym == 'C':
                # DFTB order for C in PTCDA: s, py, pz, px
                s_c, py_c, pz_c, px_c = evec[ao:ao+4]
                c[ia] = [px_c, py_c, pz_c, s_c]
                ao += 4
            elif sym == 'N':
                s_c, py_c, pz_c, px_c = evec[ao:ao+4]
                c[ia] = [px_c, py_c, pz_c, s_c]
                ao += 4
            elif sym == 'O':
                s_c, py_c, pz_c, px_c = evec[ao:ao+4]
                c[ia] = [px_c, py_c, pz_c, s_c]
                ao += 4
        return c

    # ---- Load STO basis from waveplot_in.hsd (authoritative source, Bohr -> Ang conversion inside) ----
    hsd_path = dftb_dir / 'waveplot_in.hsd'
    species_list_sto = parse_basis_hsd_ang(hsd_path)
    print(f"\n[STO basis from {hsd_path}]")
    for sp in species_list_sto:
        for orb in sp['orbitals']:
            print(f"  {sp['name']} l={orb['l']}  alpha={orb['exponents']}  cutoff={orb['cutoff']:.3f} Å  coeff={orb['coefficients'].ravel()}")

    # ---- Initialize projector ----
    print(f"\n[Initializing OpenCL projector]")
    projector = GridProjector(fdata_dir=None, verbosity=args.verbose)
    projector.load_basis_sto(species_list_sto)

    sym_to_nz = {'H': 1, 'O': 8}
    atom_nz = np.array([sym_to_nz[s] for s in atom_symbols], dtype=np.int32)
    # rcut per atom: max orbital cutoff for that species (already in Ang from parse_basis_hsd_ang)
    nz_to_rcut = {sp['atomic_number']: max(o['cutoff'] for o in sp['orbitals']) for sp in species_list_sto}
    rcut = np.array([nz_to_rcut[nz] for nz in atom_nz], dtype=np.float32)
    atoms_dict = {'pos': coords.astype(np.float32), 'Rcut': rcut, 'type': atom_nz}
    norb_per = np.full(natoms, numorb_max, dtype=np.int32)

    # ---- Method 2: Point evaluation along z-axis (if --points) ----
    if args.points:
        print(f"\n[Method 2: Point evaluation along z-axis]")
        z_vals = np.linspace(args.z_range[0], args.z_range[1], args.npoints)
        x_vals = np.zeros_like(z_vals)
        y_vals = np.zeros_like(z_vals)
        points_ang = np.column_stack([x_vals, y_vals, z_vals])  # (npoints, 3)
        print(f"  Evaluating at {args.npoints} points along z-axis from {z_vals[0]:.2f} to {z_vals[-1]:.2f} Å")
        
        # For OpenCL: generate fine 3D grid and interpolate (project_orbital_points is buggy)
        # Use smaller step for better interpolation accuracy
        step_fine = 0.05  # Å
        origin_fine = np.array([-4.725, -4.125, -3.525])
        ngrid_fine = np.ceil((coords.max(axis=0) - coords.min(axis=0) + 2*args.margin) / step_fine).astype(int)
        dA = np.array([step_fine, 0., 0.])
        dB = np.array([0., step_fine, 0.])
        dC = np.array([0., 0., step_fine])
        grid_spec_fine = {'origin': origin_fine, 'dA': dA, 'dB': dB, 'dC': dC, 'ngrid': ngrid_fine}
        
        print(f"  Generating fine 3D grid: {ngrid_fine} with step={step_fine} Å")
        
        # Evaluate each MO on fine grid
        ocl_vals = np.zeros((nstates, args.npoints), dtype=np.float32)
        for imo in range(nstates):
            coeffs_k = dftb_to_kernel_coeffs(evecs[imo], atom_symbols)
            grid_3d = projector.project_orbital(coeffs_k, norb_per, atoms_dict, grid_spec_fine, nMaxAtom=args.nmax_atom)
            # Interpolate at points
            idx = ((points_ang - origin_fine) / step_fine).astype(int)
            idx = np.clip(idx, [0,0,0], [grid_3d.shape[0]-1, grid_3d.shape[1]-1, grid_3d.shape[2]-1])
            for i, (ix, iy, iz) in enumerate(idx):
                ocl_vals[imo, i] = grid_3d[ix, iy, iz]
            print(f"  MO{imo+1} max = {np.abs(ocl_vals[imo]).max():.6f}")
        
        # Plot comparison (just OpenCL values since we don't have libwaveplot reference here)
        fig, axes = plt.subplots(nstates, 2, figsize=(14, 3*nstates))
        if nstates == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for imo in range(nstates):
            ax_lin = axes[2*imo]
            ax_log = axes[2*imo+1]
            
            ax_lin.plot(z_vals, ocl_vals[imo], 'r-', lw=2, label='OpenCL')
            ax_lin.axhline(0, c='gray', lw=0.5)
            ax_lin.set_xlabel('z (Å)')
            ax_lin.set_ylabel('ψ')
            ax_lin.set_title(f'MO{imo+1} — linear scale')
            ax_lin.legend(fontsize=8)
            
            mask = z_vals >= 0
            eps = 1e-12
            y_oc = np.abs(ocl_vals[imo, mask]).clip(eps)
            ax_log.semilogy(z_vals[mask], y_oc, 'r-', lw=2, label='OpenCL')
            ax_log.set_xlabel('z (Å)')
            ax_log.set_ylabel('|ψ| (log)')
            ax_log.set_title(f'MO{imo+1} — log scale')
            ax_log.legend(fontsize=8)
        
        fig.tight_layout()
        out_file = output_dir / f'points_n{args.npoints}.png'
        fig.savefig(str(out_file), dpi=args.dpi)
        print(f"\nSaved: {out_file}")
        if not args.no_show:
            plt.show()
        return

    # ---- Setup 3D grid ----
    print(f"\n[Setting up grid]")

    '''
    # ORIGINAL GRID SETUP *- do not mofidy or delete
    pos_min = coords.min(axis=0) - args.margin
    pos_max = coords.max(axis=0) + args.margin
    span = pos_max - pos_min
    ngrid = np.ceil(span / args.step).astype(int)
    block = 8
    ngrid = ((ngrid + block - 1) // block) * block
    '''

    # NEW Grid setup to match WAVEPLOT
    # Match WAVEPLOT cube file origin exactly for comparison
    # WAVEPLOT origin from cube: [-4.7249995, -4.12499986, -3.52499969] Å
    origin = np.array([-4.725, -4.125, -3.525], dtype=np.float64)
    ngrid = np.array([64, 56, 48], dtype=np.int32)  # Match WAVEPLOT
    dA = np.array([args.step, 0.0, 0.0])
    dB = np.array([0.0, args.step, 0.0])
    dC = np.array([0.0, 0.0, args.step])
    grid_spec = {'origin': origin, 'dA': dA, 'dB': dB, 'dC': dC, 'ngrid': ngrid}
    print(f"  Grid: {ngrid},  origin: {origin},  step: {args.step}")

    # ---- Project and plot all MOs ----
    mo_indices = list(range(nstates)) if args.plot_mos is None else [i-1 for i in args.plot_mos]
    print(f"\n[Projecting {len(mo_indices)} MOs]")

    def best_slice(grid_3d):
        """Return (data_2d, axis_label, coord_label, extent) for the slice with max |psi|."""
        nx_, ny_, nz_ = grid_3d.shape
        # XY at best z
        iz_ = np.argmax([np.abs(grid_3d[:,:,iz]).max() for iz in range(nz_)])
        xy = grid_3d[:, :, iz_]
        z_val = origin[2] + iz_ * args.step
        ext_xy = [origin[0], origin[0]+nx_*args.step, origin[1], origin[1]+ny_*args.step]
        # XZ at best y
        iy_ = np.argmax([np.abs(grid_3d[:,iy,:]).max() for iy in range(ny_)])
        xz = grid_3d[:, iy_, :]
        y_val = origin[1] + iy_ * args.step
        ext_xz = [origin[0], origin[0]+nx_*args.step, origin[2], origin[2]+nz_*args.step]
        # YZ at best x
        ix_ = np.argmax([np.abs(grid_3d[ix,:,:]).max() for ix in range(nx_)])
        yz = grid_3d[ix_, :, :]
        x_val = origin[0] + ix_ * args.step
        ext_yz = [origin[1], origin[1]+ny_*args.step, origin[2], origin[2]+nz_*args.step]

        slices = [
            (xy, 'XY', f'z={z_val:.2f}Å', ext_xy, 'x (Å)', 'y (Å)'),
            (xz, 'XZ', f'y={y_val:.2f}Å', ext_xz, 'x (Å)', 'z (Å)'),
            (yz, 'YZ', f'x={x_val:.2f}Å', ext_yz, 'y (Å)', 'z (Å)'),
        ]
        # Pick the slice with largest max |psi|
        best = max(slices, key=lambda s: np.abs(s[0]).max())
        return best

    def plot_mo_slice(data_2d, atom_coords, atom_syms, plane_name, coord_str, extent,
                      xlabel, ylabel, title, fname):
        """Plot a 2D orbital slice with atoms projected onto the plane."""
        fig_i, ax_i = plt.subplots(figsize=(5, 4))
        clim = max(abs(data_2d.min()), abs(data_2d.max()))
        clim = clim if clim > 1e-10 else 1.0
        im = ax_i.imshow(data_2d.T, origin='lower', extent=extent,
                         cmap='RdBu_r', vmin=-clim, vmax=clim, interpolation='bilinear')
        plt.colorbar(im, ax=ax_i, label='ψ amplitude')
        # Project atoms onto the 2D plane
        ax_map = {'XY': (0,1), 'XZ': (0,2), 'YZ': (1,2)}
        a0, a1 = ax_map[plane_name]
        for ia, sym in enumerate(atom_syms):
            ax_i.scatter(atom_coords[ia, a0], atom_coords[ia, a1], c='k', s=40, zorder=5)
            ax_i.annotate(sym, (atom_coords[ia,a0], atom_coords[ia,a1]),
                          textcoords='offset points', xytext=(4,4), fontsize=9)
        ax_i.set_xlabel(xlabel); ax_i.set_ylabel(ylabel)
        ax_i.set_title(title)
        fig_i.tight_layout(); fig_i.savefig(fname, dpi=args.dpi); plt.close(fig_i)

    # Combined figure: all MOs
    n_cols = min(3, len(mo_indices))
    n_rows = (len(mo_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if len(mo_indices) == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for plot_idx, imo in enumerate(mo_indices):
        coeffs_k = dftb_to_kernel_coeffs(evecs[imo], atom_symbols)
        try:
            grid_3d = projector.project_orbital(
                coeffs_k, norb_per, atoms_dict, grid_spec, nMaxAtom=args.nmax_atom
            )
        except Exception as e:
            print(f"  MO{imo+1}: FAILED {e}")
            continue

        gmax = np.abs(grid_3d).max()
        occ_str = f"occ={occs[imo]}"
        print(f"  MO{imo+1} E={energies_ev[imo]:7.3f} eV {occ_str}  |ψ|max={gmax:.4e}")
        if gmax < 1e-8:
            print(f"  WARNING: MO{imo+1} grid is essentially zero — kernel issue?")
            continue

        # Save grid for WAVEPLOT comparison
        npy_path = str(output_dir / f"opencl_MO{imo+1}.npy")
        np.save(npy_path, grid_3d)
        print(f"    Saved OpenCL grid: {npy_path}")

        # Save individual figure with all 3 orthogonal slices
        nx_, ny_, nz_ = grid_3d.shape
        # Slices through O atom position (ia=0), nearest grid index
        o_pos = coords[0]
        iz_o = int(round((o_pos[2] - origin[2]) / args.step)); iz_o = max(0, min(nz_-1, iz_o))
        iy_o = int(round((o_pos[1] - origin[1]) / args.step)); iy_o = max(0, min(ny_-1, iy_o))
        ix_o = int(round((o_pos[0] - origin[0]) / args.step)); ix_o = max(0, min(nx_-1, ix_o))
        slices_to_plot = [
            (grid_3d[:,:,iz_o], 'XY', f'z={origin[2]+iz_o*args.step:.2f}Å',
             [origin[0], origin[0]+nx_*args.step, origin[1], origin[1]+ny_*args.step], 'x (Å)', 'y (Å)'),
            (grid_3d[:,iy_o,:], 'XZ', f'y={origin[1]+iy_o*args.step:.2f}Å',
             [origin[0], origin[0]+nx_*args.step, origin[2], origin[2]+nz_*args.step], 'x (Å)', 'z (Å)'),
            (grid_3d[ix_o,:,:], 'YZ', f'x={origin[0]+ix_o*args.step:.2f}Å',
             [origin[1], origin[1]+ny_*args.step, origin[2], origin[2]+nz_*args.step], 'y (Å)', 'z (Å)'),
        ]
        fig_3, axes_3 = plt.subplots(1, 3, figsize=(14, 4))
        ax_map = {'XY': (0,1), 'XZ': (0,2), 'YZ': (1,2)}
        clim_3 = max(abs(grid_3d.min()), abs(grid_3d.max())) or 1e-10
        for ax_3, (d2, pname, cstr, ext, xl, yl) in zip(axes_3, slices_to_plot):
            im3 = ax_3.imshow(d2.T, origin='lower', extent=ext,
                              cmap='RdBu_r', vmin=-clim_3, vmax=clim_3, interpolation='bilinear')
            plt.colorbar(im3, ax=ax_3, label='ψ')
            a0_, a1_ = ax_map[pname]
            ax_3.scatter(coords[:, a0_], coords[:, a1_], c='k', s=30, zorder=5)
            for ia_, sym_ in enumerate(atom_symbols):
                ax_3.annotate(sym_, (coords[ia_,a0_], coords[ia_,a1_]),
                              textcoords='offset points', xytext=(3,3), fontsize=8)
            ax_3.set_xlabel(xl); ax_3.set_ylabel(yl)
            ax_3.set_title(f"{pname} {cstr}")
        fig_3.suptitle(f"H2O MO{imo+1} E={energies_ev[imo]:.3f} eV ({occ_str})")
        fig_3.tight_layout()
        fname = str(output_dir / f"{args.prefix}_h2o_MO{imo+1}.png")
        fig_3.savefig(fname, dpi=args.dpi); plt.close(fig_3)
        data_2d, plane, coord_str, extent, xlabel, ylabel = best_slice(grid_3d)
        print(f"    best slice: {plane} at {coord_str},  range [{data_2d.min():.4e}, {data_2d.max():.4e}]")
        print(f"    Saved: {fname}")

        # Add to combined figure
        ax = axes[plot_idx]
        clim = max(abs(data_2d.min()), abs(data_2d.max())) or 1e-10
        ax.imshow(data_2d.T, origin='lower', extent=extent,
                  cmap='RdBu_r', vmin=-clim, vmax=clim, interpolation='bilinear')
        ax_map = {'XY': (0,1), 'XZ': (0,2), 'YZ': (1,2)}
        a0, a1 = ax_map[plane]
        ax.scatter(coords[:, a0], coords[:, a1], c='k', s=15, zorder=5)
        ax.set_title(f"MO{imo+1} {energies_ev[imo]:.2f}eV {occ_str}\n{plane} {coord_str}", fontsize=8)

    for ax in axes[len(mo_indices):]:
        ax.set_visible(False)

    fig.suptitle("H2O Molecular Orbitals", fontsize=12)
    fig.tight_layout()
    combined_path = str(output_dir / f"{args.prefix}_h2o_all_MOs.png")
    fig.savefig(combined_path, dpi=args.dpi)
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f"\n  Saved combined: {combined_path}")

    print(f"\n[Outputs saved to: {output_dir}]")
    return True


def run_ptcda_test(args):
    """Run PTCDA periodic test."""
    print("=" * 60)
    print("Running PTCDA Test (Periodic)")
    print("=" * 60)
    
    # Setup paths
    xyz_path = _REPO_ROOT / 'data' / 'xyz' / 'PTCDA.xyz'
    output_dir = Path(args.output_dir) / 'PTCDA'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Setup]")
    print(f"  XYZ file: {xyz_path}")
    print(f"  Output dir: {output_dir}")
    
    # Read XYZ
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    
    natoms = int(lines[0].strip())
    
    # Parse lattice vectors if present
    lvs = None
    line2 = lines[1].strip()
    if line2.startswith('lvs'):
        parts = line2.split()[1:]
        lvs = np.array([float(x) for x in parts]).reshape(3, 3)
        coord_start = 2
    else:
        coord_start = 2
    
    atom_symbols = []
    coords = []
    
    for i in range(coord_start, coord_start + natoms):
        parts = lines[i].split()
        atom_symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    
    coords = np.array(coords)
    print(f"  Atoms: {natoms}")
    print(f"  Species: {set(atom_symbols)}")
    if lvs is not None:
        print(f"  Lattice: {lvs[0]} / {lvs[1]} / {lvs[2]}")
    
    # Determine z-height for plane
    if args.z_height is not None:
        z_target = args.z_height
    else:
        # Default: 2.0 Å above molecular plane
        z_target = coords[:, 2].max() + 2.0
    
    print(f"  Target Z plane: {z_target:.2f} Å")
    
    # Initialize projector
    print(f"\n[Initializing OpenCL projector]")
    projector = GridProjector(fdata_dir=None, verbosity=args.verbose)
    
    # Load STO basis from waveplot_in.hsd if available, else raise — no hardcoded fallback
    ptcda_dftb_dir = Path(__file__).parent / 'dftb_ptcda'
    hsd_candidates = [ptcda_dftb_dir / 'waveplot_in.hsd', Path(args.output_dir).parent / 'waveplot_in.hsd']
    hsd_path_ptcda = next((p for p in hsd_candidates if p.exists()), None)
    if hsd_path_ptcda is None:
        raise FileNotFoundError(f"No waveplot_in.hsd found for PTCDA. Checked: {hsd_candidates}")
    print(f"\n[Loading STO basis from {hsd_path_ptcda}]")
    species_list_sto = parse_basis_hsd_ang(hsd_path_ptcda)
    for sp in species_list_sto:
        for orb in sp['orbitals']:
            print(f"  {sp['name']} l={orb['l']}  alpha={orb['exponents']}  cutoff={orb['cutoff']:.3f} Å")
    projector.load_basis_sto(species_list_sto)
    
    # Map symbols -> atomic numbers
    sym_to_nz = {'H': 1, 'C': 6, 'O': 8}
    unknown_symbols = set(atom_symbols) - set(sym_to_nz.keys())
    if unknown_symbols:
        raise ValueError(f"Unknown species in PTCDA: {unknown_symbols}")
    atom_nz = np.array([sym_to_nz[s] for s in atom_symbols], dtype=np.int32)
    natoms = len(atom_nz)
    
    nz_to_rcut = {sp['atomic_number']: max(o['cutoff'] for o in sp['orbitals']) for sp in species_list_sto}
    rcut = np.array([nz_to_rcut[nz] for nz in atom_nz], dtype=np.float32)
    
    atoms_dict = {'pos': coords.astype(np.float32), 'Rcut': rcut, 'type': atom_nz}
    
    if args.verbose > 0:
        print(f"  natoms={natoms}, unique species: {sorted(set(atom_nz))}")
    
    # Setup grid for 2D plane at z_target
    print(f"\n[Setting up grid]")
    
    # For 2D plane, we need to extend in x,y around the molecule
    margin = args.margin
    xmin, xmax = coords[:, 0].min() - margin, coords[:, 0].max() + margin
    ymin, ymax = coords[:, 1].min() - margin, coords[:, 1].max() + margin
    
    span_x = xmax - xmin
    span_y = ymax - ymin
    
    nx = int(np.ceil(span_x / args.step))
    ny = int(np.ceil(span_y / args.step))
    nz = 8  # Minimum 8 for kernel block structure (8x8x8 blocks)
    
    # Round to block size
    block = 8
    nx = ((nx + block - 1) // block) * block
    ny = ((ny + block - 1) // block) * block
    
    ngrid = np.array([nx, ny, nz])
    
    # Grid vectors
    dA = np.array([args.step, 0.0, 0.0])
    dB = np.array([0.0, args.step, 0.0])
    dC = np.array([0.0, 0.0, args.step])  # Not really used for 2D
    
    origin = np.array([xmin, ymin, z_target - 0.5 * args.step])
    
    grid_spec = {
        'origin': origin,
        'dA': dA,
        'dB': dB,
        'dC': dC,
        'ngrid': ngrid
    }
    
    print(f"  Grid: {ngrid}")
    print(f"  Origin: {origin}")
    print(f"  Step: {args.step}")
    print(f"  Extent: [{xmin:.2f}, {xmax:.2f}] x [{ymin:.2f}, {ymax:.2f}]")
    
    # Test projection
    print(f"\n[Test: Orbital Projection at z={z_target:.2f}Å]")
    numorb_max = 4
    norb_per = np.full(natoms, numorb_max, dtype=np.int32)
    print(f"  natoms={natoms}, total orbs={natoms*numorb_max} (padded)")
    
    # Random test orbital in [px,py,pz,s] per atom; p-channels zeroed for H
    np.random.seed(42)
    coeffs_raw = np.random.randn(natoms, numorb_max).astype(np.float32)
    for ia in range(natoms):
        if atom_nz[ia] == 1:
            coeffs_raw[ia, :3] = 0.0
    coeffs_raw /= (np.linalg.norm(coeffs_raw) + 1e-10)
    
    try:
        nmax_atom_ptcda = max(args.nmax_atom, 64)
        grid_3d = projector.project_orbital(
            coeffs_raw, norb_per, atoms_dict, grid_spec, nMaxAtom=nmax_atom_ptcda
        )
        
        # Extract z=0 plane (z_target)
        grid_2d = grid_3d[:, :, 0]
        
        vmin, vmax = validate_grid(grid_2d, "Test Orbital (PTCDA plane)", args)
        print(f"  ✓ Projection successful!")
        
        # Plot with atoms
        extent = [xmin, xmax, ymin, ymax]
        plot_orbital_2d(grid_2d, coords, extent,
                       f"PTCDA Test Orbital (z={z_target:.1f}Å)",
                       str(output_dir / f"{args.prefix}_ptcda_z{z_target:.1f}.png"),
                       args)
        
        # If requested, also compute and plot density
        if args.plot_density:
            print(f"\n[Test: Density]")
            # Simple test: square of orbital
            density = grid_2d ** 2
            plot_orbital_2d(density, coords, extent,
                           f"PTCDA Test Density (z={z_target:.1f}Å)",
                           str(output_dir / f"{args.prefix}_ptcda_density_z{z_target:.1f}.png"),
                           args, is_density=True)
        
    except Exception as e:
        print(f"  ✗ Projection failed: {e}")
        raise
    
    print(f"\n[Outputs saved to: {output_dir}]")
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    print("DFTB+ OpenCL Waveplot Test")
    print("=" * 60)
    print(f"Verbosity: {args.verbose}")
    print(f"System: {args.system}")
    print(f"Step: {args.step} Å")
    print(f"Margin: {args.margin} Å")
    
    try:
        if args.system == 'H2O':
            run_h2o_test(args)
        elif args.system == 'PTCDA':
            run_ptcda_test(args)
        else:
            print(f"Unknown system: {args.system}")
            return 1
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        return 0
        
    except ValueError as e:
        print(f"\n[VALIDATION ERROR] {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
