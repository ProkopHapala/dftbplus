#!/usr/bin/env python3
"""
DFTBplusGridProjector - OpenCL grid projector for DFTB+ waveplot.

Adapts Fireball's GridProjector to work with DFTB+ Slater-type orbitals.
"""

import numpy as np
import pyopencl as cl
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyBall.OCL.Grid import GridProjector
from pyBall.OCL.DFTBplusParser import precompute_sto_grid, compute_sto_radial


class DFTBplusGridProjector(GridProjector):
    """
    OpenCL grid projector adapted for DFTB+ Slater-type orbitals.
    
    Replaces Fireball's numerical basis functions with DFTB+ STO parameters.
    """
    
    def __init__(self, ctx=None, queue=None, nloc=32, verbosity=0):
        # Initialize OpenCL context first
        self.nloc = nloc
        self.verbosity = verbosity
        self.ctx = ctx
        self.queue = queue
        
        # Set debug flags (required by parent class _load_kernels)
        self.debug_early_exit = False
        self.debug_clear_only = False
        self.debug_return0 = False
        self.debug_read_task = False
        self.debug_read_grid = False
        
        if ctx:
            if queue is None:
                self.queue = cl.CommandQueue(self.ctx)
        else:
            # Initialize OpenCL context - prefer NVIDIA
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
            
            # Try to find NVIDIA platform first
            nvidia_platform = None
            for platform in platforms:
                if 'NVIDIA' in platform.name:
                    nvidia_platform = platform
                    break
            
            if nvidia_platform:
                self.ctx = cl.Context(devices=nvidia_platform.get_devices())
                if self.verbosity > 0:
                    print(f"[DFTBplusGridProjector] Using NVIDIA platform: {nvidia_platform.name}")
            else:
                # Fall back to first available platform
                devices = platforms[0].get_devices()
                if not devices:
                    raise RuntimeError("No OpenCL devices found")
                self.ctx = cl.Context(devices=[devices[0]])
                if self.verbosity > 0:
                    print(f"[DFTBplusGridProjector] Using platform: {platforms[0].name}")
            
            self.queue = cl.CommandQueue(self.ctx)
        
        self.task_dtype = [
            ('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('w', 'i4'),
            ('na', 'i4'), ('nj', 'i4'), ('pad1', 'i4'), ('pad2', 'i4')
        ]
        self.task_dtype_np = np.dtype(self.task_dtype)
        
        # DFTB+ specific data
        self.sto_basis = {}  # STO parameters per species
        self.basis_data = {}  # Will store precomputed grids
        
        self._load_kernels_dftb()
    
    def _load_kernels_dftb(self):
        """Load DFTB+ specific OpenCL kernels."""
        kernel_path = os.path.join(os.path.dirname(__file__), 'cl', 'DFTBplusGrid.cl')
        with open(kernel_path, 'r') as f:
            kernel_src = f.read()
        
        self.prg = cl.Program(self.ctx, kernel_src).build()
    
    def load_basis_dftb(self, basis_data):
        """
        Load DFTB+ STO basis and precompute on uniform grid.
        
        Args:
            basis_data: dict from DFTBplusParser with 'species' list
        """
        species_list = basis_data['species']
        
        if self.verbosity > 0:
            print(f"[DFTBplusGridProjector] Loading {len(species_list)} species")
        
        # Find finest resolution and largest cutoff across all species/orbitals
        all_resolutions = []
        all_cutoffs = []
        
        for sp in species_list:
            all_resolutions.append(sp['resolution'])
            for orb in sp['orbitals']:
                all_cutoffs.append(orb['cutoff'])
        
        dr = min(all_resolutions)
        rc_max = max(all_cutoffs)
        n_nodes = int(np.ceil(rc_max / dr)) + 2
        
        if self.verbosity > 0:
            print(f"[DFTBplusGridProjector] Common grid: dr={dr:.6f} Å, rc_max={rc_max:.3f} Å, n_nodes={n_nodes}")
        
        # Determine max shells per species
        max_shells = max(len(sp['orbitals']) for sp in species_list)
        n_species = len(species_list)
        
        # Pack basis data: (n_species, max_shells, n_nodes, 2) for (value, d2)
        packed_basis = np.zeros((n_species, max_shells, n_nodes, 2), dtype=np.float32)
        
        self.species_nz = []  # Atomic numbers in order
        
        for i_spec, sp in enumerate(species_list):
            atomic_number = sp['atomic_number']
            self.species_nz.append(atomic_number)
            
            for i_orb, orb in enumerate(sp['orbitals']):
                aa = orb['coefficients']  # (nPow, nAlpha)
                alpha = orb['exponents']  # (nAlpha,)
                ll = orb['l']
                cutoff = orb['cutoff']
                
                # Precompute STO on grid with common n_nodes and dr
                grid_vals, grid_d2, _ = precompute_sto_grid(aa, alpha, ll, cutoff, dr, n_nodes=n_nodes, dr=dr)
                
                packed_basis[i_spec, i_orb, :, 0] = grid_vals
                packed_basis[i_spec, i_orb, :, 1] = grid_d2
                
                if self.verbosity > 0:
                    nAlpha = len(alpha)
                    nPow = aa.shape[0] if aa.ndim > 1 else 1
                    print(f"[DFTBplusGridProjector]   Species Z={atomic_number} shell {i_orb} (l={ll}): nAlpha={nAlpha}, nPow={nPow}, cutoff={cutoff:.2f}")
        
        # Upload to GPU
        self.d_basis = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=packed_basis)
        self.basis_meta = {
            'n_species': n_species,
            'max_shells': max_shells,
            'n_nodes': n_nodes,
            'dr': dr,
            'nz_map': {nz: i for i, nz in enumerate(self.species_nz)}
        }
        
        # Store STO parameters for direct evaluation if needed
        self.sto_basis = basis_data
        
        return packed_basis
    
    def prepare_atoms_dftb(self, coords, species, basis_data, rcut_margin=0.5):
        """
        Prepare atom data for DFTB+ with per-orbital cutoffs.
        
        Args:
            coords: (natoms, 3) positions
            species: (natoms,) species indices
            basis_data: basis dict with orbital info
            rcut_margin: margin added to max orbital cutoff
        
        Returns:
            atoms_dict compatible with GridProjector
        """
        natoms = len(coords)
        species_list = basis_data['species']
        
        # Compute per-atom properties
        pos_list = []
        rcut_list = []
        type_list = []
        i0orb_list = []
        norb_list = []
        
        orb_offset = 0
        
        for ia in range(natoms):
            ispec = species[ia]
            sp = species_list[ispec]
            
            pos_list.append(coords[ia])
            
            # Max cutoff across all orbitals for this species
            max_cutoff = max(orb['cutoff'] for orb in sp['orbitals'])
            rcut_list.append(max_cutoff + rcut_margin)
            
            type_list.append(ispec)
            
            # Count orbitals: sum over shells of (2*l+1)
            norb = sum(2*orb['l'] + 1 for orb in sp['orbitals'])
            i0orb_list.append(orb_offset)
            norb_list.append(norb)
            
            orb_offset += norb
        
        atoms_dict = {
            'pos': np.array(pos_list, dtype=np.float32),
            'Rcut': np.array(rcut_list, dtype=np.float32),
            'type': np.array(type_list, dtype=np.int32),
            'i0orb': np.array(i0orb_list, dtype=np.int32),
            'norb': np.array(norb_list, dtype=np.int32)
        }
        
        return atoms_dict
    
    def project_orbital(self, coeffs, atoms_dict, grid_spec, norb_total, 
                        cell_vecs=None, kpoint=None, task_list=None, nMaxAtom=64):
        """
        Project a single molecular orbital to the grid.
        
        Args:
            coeffs: (norb_total,) or (norb_total, 2) for complex
            atoms_dict: atom data from prepare_atoms_dftb
            grid_spec: dict with 'origin', 'dA', 'dB', 'dC', 'ngrid'
            norb_total: total number of orbitals
            cell_vecs: (ncells, 3) periodic cell translations (optional)
            kpoint: (3,) k-point for phase factors (optional)
            task_list: pre-built task list (optional)
            nMaxAtom: max atoms per task
        
        Returns:
            grid_data: (nx, ny, nz) array of orbital values
        """
        ngrid_arr = grid_spec['ngrid']
        nx, ny, nz = int(ngrid_arr[0]), int(ngrid_arr[1]), int(ngrid_arr[2])
        npoints = nx * ny * nz
        
        # Handle complex coefficients
        is_complex = coeffs.ndim > 1 and coeffs.shape[1] == 2
        if is_complex:
            coeffs_cl = np.zeros(norb_total * 2, dtype=np.float32)
            coeffs_cl[0::2] = coeffs[:, 0]  # Real
            coeffs_cl[1::2] = coeffs[:, 1]  # Imag
        else:
            coeffs_cl = coeffs.astype(np.float32)
        
        if self.verbosity > 0:
            print(f"[DEBUG] norb_total={norb_total}, coeffs_cl.shape={coeffs_cl.shape}")
            print(f"[DEBUG] coeffs_cl min={coeffs_cl.min():.3f}, max={coeffs_cl.max():.3f}, sum={coeffs_cl.sum():.3f}")
            print(f"[DEBUG] atoms_dict norb={atoms_dict['norb']}, i0orb={atoms_dict['i0orb']}")
        
        # Build tasks if not provided
        if task_list is None:
            tasks_np, task_atoms_np = self.build_tasks_gpu(atoms_dict, grid_spec, nMaxAtom=nMaxAtom)
        else:
            tasks_np, task_atoms_np = task_list
        
        n_tasks = len(tasks_np)
        
        # Prepare grid spec buffer (GridSpec struct in kernel)
        # struct layout: float4 origin, float4 dA, float4 dB, float4 dC, int4 ngrid
        # Total: 16 floats + 4 ints = 80 bytes, but we need proper alignment
        # Actually float4[4] + int4 = 64 + 16 = 80 bytes
        # Use structured approach: 20 floats to cover everything (padded)
        grid_buffer = np.zeros(20, dtype=np.float32)
        grid_buffer[0:3] = grid_spec['origin']
        grid_buffer[4:7] = grid_spec['dA']
        grid_buffer[8:11] = grid_spec['dB']
        grid_buffer[12:15] = grid_spec['dC']
        # Store ngrid as floats which will be reinterpreted as ints by kernel
        # Or we need to create a byte buffer with proper layout
        
        # Better approach: Create a bytes buffer with exact layout
        import struct
        grid_bytes = b''
        # float4 origin
        grid_bytes += struct.pack('4f', *list(grid_spec['origin']) + [0.0])
        # float4 dA
        grid_bytes += struct.pack('4f', *list(grid_spec['dA']) + [0.0])
        # float4 dB
        grid_bytes += struct.pack('4f', *list(grid_spec['dB']) + [0.0])
        # float4 dC
        grid_bytes += struct.pack('4f', *list(grid_spec['dC']) + [0.0])
        # int4 ngrid
        grid_bytes += struct.pack('4i', nx, ny, nz, 0)
        
        d_grid = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid_bytes)
        d_tasks = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=tasks_np)
        d_task_atoms = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=task_atoms_np)
        
        # Atom data
        atom_pos = np.zeros((len(atoms_dict['pos']), 4), dtype=np.float32)
        atom_pos[:, 0:3] = atoms_dict['pos']
        atom_pos[:, 3] = atoms_dict['Rcut']
        d_atoms = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=atom_pos)
        
        # Coefficients
        d_coeffs = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coeffs_cl)
        
        # Output grid - initialize to zero (kernel uses += accumulation)
        if is_complex:
            out_grid = np.zeros(npoints * 2, dtype=np.float32)
        else:
            out_grid = np.zeros(npoints, dtype=np.float32)
        d_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=out_grid)
        
        # Species info buffer
        species_info = np.zeros((len(self.species_nz), 4), dtype=np.int32)
        for i_spec, at_num in enumerate(self.species_nz):
            sp = self.sto_basis['species'][i_spec]
            species_info[i_spec, 0] = at_num
            species_info[i_spec, 1] = len(sp['orbitals'])
            species_info[i_spec, 2] = sp['orbitals'][0]['l'] if sp['orbitals'] else 0
            species_info[i_spec, 3] = 0
        
        if self.verbosity > 0:
            print(f"[DEBUG] species_info shape: {species_info.shape}")
            print(f"[DEBUG] species_info:\n{species_info}")
            print(f"[DEBUG] basis_meta: n_species={self.basis_meta['n_species']}, max_shells={self.basis_meta['max_shells']}, n_nodes={self.basis_meta['n_nodes']}")
        
        d_species_info = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=species_info)
        
        # Prepare kernel arguments
        n_basis_params = np.array([
            self.basis_meta['n_species'],
            self.basis_meta['max_shells'],
            self.basis_meta['n_nodes'],
            0  # pad
        ], dtype=np.int32)
        
        # Get numorb_max from atoms
        numorb_max = int(atoms_dict['norb'].max())
        
        # Launch kernel with all required arguments
        # The kernel uses local_size=32 (threads per task) and each thread processes multiple voxels
        local_size = (32, 1, 1)  # threads_per_task from kernel
        global_size = (n_tasks * 32, 1, 1)
        
        kernel = self.prg.project_orbital_dftb
        kernel(
            self.queue, global_size, local_size,
            d_grid,
            np.int32(n_tasks),
            d_tasks,
            d_atoms,
            d_task_atoms,
            d_coeffs,
            self.d_basis,
            np.int32(self.basis_meta['n_nodes']),
            np.float32(self.basis_meta['dr']),
            np.int32(self.basis_meta['max_shells']),
            np.int32(nMaxAtom),
            d_out
        )
        
        # Read back results
        cl.enqueue_copy(self.queue, out_grid, d_out)
        
        # Reshape output
        if is_complex:
            out_complex = np.zeros(npoints, dtype=np.complex64)
            out_complex.real = out_grid[0::2]
            out_complex.imag = out_grid[1::2]
            return out_complex.reshape((nx, ny, nz))
        else:
            return out_grid.reshape((nx, ny, nz))
    
    def compute_density(self, eigenvecs, occupations, atoms_dict, grid_spec, norb_total,
                        cell_vecs=None, kpoints=None, k_indexes=None, nMaxAtom=64):
        """
        Compute electron density from occupied eigenvectors.
        
        Args:
            eigenvecs: (norb, nstates, nkpoints, nspin) array
            occupations: (nstates, nkpoints, nspin) array
            atoms_dict: atom data
            grid_spec: grid specification
            norb_total: total orbitals
            cell_vecs: periodic cell vectors
            kpoints: k-point coordinates
            k_indexes: k-point index per state
            nMaxAtom: max atoms per task
        
        Returns:
            density: (nx, ny, nz) electron density
        """
        norb, nstates, nkpoints, nspin = eigenvecs.shape
        nx, ny, nz = grid_spec['ngrid']
        density = np.zeros((nx, ny, nz), dtype=np.float64)
        
        for ispin in range(nspin):
            for ik in range(nkpoints):
                for istate in range(nstates):
                    occ = occupations[istate, ik, ispin]
                    if occ < 1e-10:
                        continue
                    
                    # Get MO coefficients
                    if eigenvecs.dtype == np.complex128:
                        coeffs = eigenvecs[:, istate, ik, ispin]
                        is_complex = True
                    else:
                        coeffs = eigenvecs[:, istate, ik, ispin]
                        is_complex = False
                    
                    # Project orbital
                    if is_complex and kpoints is not None:
                        # Need phase factors - use periodic projection
                        orb_grid = self.project_orbital_periodic(
                            coeffs, atoms_dict, grid_spec, norb_total,
                            cell_vecs, kpoints[:, ik], nMaxAtom=nMaxAtom
                        )
                    else:
                        orb_grid = self.project_orbital(
                            coeffs, atoms_dict, grid_spec, norb_total,
                            nMaxAtom=nMaxAtom
                        )
                    
                    # Accumulate density
                    density += occ * (orb_grid.real**2 + orb_grid.imag**2)
        
        return density
    
    def project_orbital_periodic(self, coeffs, atoms_dict, grid_spec, norb_total,
                                  cell_vecs, kpoint, nMaxAtom=64):
        """
        Project orbital with periodic boundary conditions and k-point phase.
        
        TODO: Implement full periodic support with phase factors.
        For now, calls simple projection.
        """
        # This is a placeholder - full implementation would handle cell images and phases
        return self.project_orbital(coeffs, atoms_dict, grid_spec, norb_total, nMaxAtom=nMaxAtom)


class DFTBplusWaveplotRunner:
    """
    High-level runner for DFTB+ waveplot OpenCL calculations.
    """
    
    def __init__(self, work_dir='.', verbosity=0):
        self.work_dir = Path(work_dir)
        self.verbosity = verbosity
        self.parser = None
        self.projector = None
        
    def initialize(self):
        """Initialize parser and projector."""
        from pyBall.OCL.DFTBplusParser import DFTBplusParser
        
        self.parser = DFTBplusParser(self.work_dir, self.verbosity)
        self.projector = DFTBplusGridProjector(verbosity=self.verbosity)
        
    def run_calculation(self, xyz_path, basis_path=None, 
                       ngrid=None, origin=None, box_vecs=None,
                       margin=3.0, step=0.2,
                       plot_levels=None, plot_kpoints=None,
                       output_dir='waveplot_output'):
        """
        Run full waveplot calculation.
        
        Args:
            xyz_path: Path to XYZ file with geometry
            basis_path: Path to basis HSD file
            ngrid: (nx, ny, nz) grid dimensions (or None for auto)
            origin: (3,) grid origin (or None for auto)
            box_vecs: (3, 3) box vectors (or None for auto)
            margin: Margin around molecule (Angstrom)
            step: Grid spacing (Angstrom)
            plot_levels: List of MO indices to plot (or None for HOMO/LUMO)
            plot_kpoints: List of k-point indices to plot
            output_dir: Output directory for plots
        
        Returns:
            results dict with grids and metadata
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.parser.parse_detailed_xml()
        self.parser.parse_basis_hsd(basis_path)
        
        # Setup projector with STO basis
        self.projector.load_basis_dftb(self.parser.basis)
        
        # Prepare atoms
        atoms_dict = self.projector.prepare_atoms_dftb(
            self.parser.geometry['coords'],
            self.parser.geometry['species'],
            self.parser.basis
        )
        
        # Determine grid
        if ngrid is None or origin is None:
            coords = self.parser.geometry['coords']
            pos_min = coords.min(axis=0) - margin
            pos_max = coords.max(axis=0) + margin
            span = pos_max - pos_min
            
            if ngrid is None:
                ngrid = np.ceil(span / step).astype(int)
                # Round to block size (8)
                block = 8
                ngrid = ((ngrid + block - 1) // block) * block
            
            if origin is None:
                center = 0.5 * (pos_min + pos_max)
                total_span = ngrid * step
                origin = center - 0.5 * total_span
            
            dA = np.array([step, 0.0, 0.0])
            dB = np.array([0.0, step, 0.0])
            dC = np.array([0.0, 0.0, step])
        else:
            dA = box_vecs[0] / ngrid[0] if box_vecs is not None else np.array([step, 0.0, 0.0])
            dB = box_vecs[1] / ngrid[1] if box_vecs is not None else np.array([0.0, step, 0.0])
            dC = box_vecs[2] / ngrid[2] if box_vecs is not None else np.array([0.0, 0.0, step])
        
        grid_spec = {
            'origin': origin,
            'dA': dA,
            'dB': dB,
            'dC': dC,
            'ngrid': ngrid
        }
        
        if self.verbosity > 0:
            print(f"[WaveplotRunner] Grid: {ngrid}, origin: {origin}")
            print(f"  Step: {step}, span: {ngrid * step}")
        
        # Load eigenvectors
        self.parser.parse_eigenvec_bin()
        
        # Determine which levels to plot
        if plot_levels is None:
            # Plot HOMO-2 to LUMO+2
            occ = self.parser.occupations[:, 0, 0]  # First k-point, first spin
            homo = np.where(occ > 0.5)[0]
            if len(homo) > 0:
                homo = homo[-1]
            else:
                homo = len(occ) // 2 - 1
            plot_levels = list(range(max(0, homo-2), min(len(occ), homo+3)))
        
        # Project orbitals
        results = {
            'grids': {},
            'density': None,
            'geometry': self.parser.geometry,
            'grid_spec': grid_spec
        }
        
        norb_total = self.parser.basis['total_norb']
        
        for level in plot_levels:
            if self.verbosity > 0:
                print(f"[WaveplotRunner] Projecting MO {level}")
            
            # Get coefficients (first k-point, first spin for now)
            if self.parser.t_real:
                coeffs = self.parser.eigenvectors[:, level, 0, 0]
            else:
                coeffs = self.parser.eigenvectors[:, level, 0, 0]
            
            # Project
            grid = self.projector.project_orbital(
                coeffs, atoms_dict, grid_spec, norb_total
            )
            
            results['grids'][level] = grid
            
            # Check for issues
            self._validate_grid(grid, f"MO {level}")
        
        # Compute density
        if self.verbosity > 0:
            print("[WaveplotRunner] Computing total density")
        
        density = self.projector.compute_density(
            self.parser.eigenvectors,
            self.parser.occupations,
            atoms_dict,
            grid_spec,
            norb_total
        )
        results['density'] = density
        self._validate_grid(density, "Density")
        
        return results
    
    def _validate_grid(self, grid, name):
        """Validate grid values for issues."""
        vmin, vmax = grid.min(), grid.max()
        
        if np.isnan(vmin) or np.isnan(vmax):
            raise ValueError(f"{name}: NaN values detected in grid")
        
        if abs(vmin) < 1e-6 and abs(vmax) < 1e-6:
            raise ValueError(f"{name}: Grid values too small (<1e-6)")
        
        if abs(vmin) > 1e6 or abs(vmax) > 1e6:
            raise ValueError(f"{name}: Grid values too large (>1e6)")
        
        if self.verbosity > 0:
            print(f"  {name}: range [{vmin:.6e}, {vmax:.6e}]")
