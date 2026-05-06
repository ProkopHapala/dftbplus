#!/usr/bin/env python3
"""
DFTBplusParser - Parse DFTB+ output files for waveplot OpenCL reimplementation.

Handles:
- detailed.xml: Geometry, occupations, k-points
- eigenvec.bin: Binary eigenvector data (Fortran unformatted)
- Basis HSD: Slater-type orbital parameters
"""

import numpy as np
import struct
import xml.etree.ElementTree as ET
from pathlib import Path


class DFTBplusParser:
    """Parser for DFTB+ waveplot input files."""
    
    def __init__(self, work_dir='.', verbosity=0):
        self.work_dir = Path(work_dir)
        self.verbosity = verbosity
        self.geometry = None
        self.eigenvectors = None
        self.basis = None
        self.kpoints = None
        self.occupations = None
        self.t_real = True
        self.identity = None
        
    def parse_detailed_xml(self, xml_path=None):
        """Parse detailed.xml for geometry, occupations, k-points."""
        if xml_path is None:
            xml_path = self.work_dir / 'detailed.xml'
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get identity
        self.identity = int(root.find('Identity').text)
        
        # Parse geometry
        geo = root.find('Geometry')
        atoms = []
        species_names = []
        species_dict = {}
        
        for atom in geo.findall('Atom'):
            coords = [float(x) for x in atom.get('coords').split()]
            species = atom.get('species')
            if species not in species_dict:
                species_dict[species] = len(species_names)
                species_names.append(species)
            atoms.append({
                'coords': np.array(coords),
                'species': species_dict[species],
                'species_name': species
            })
        
        natoms = len(atoms)
        coords = np.array([a['coords'] for a in atoms])
        species = np.array([a['species'] for a in atoms], dtype=np.int32)
        
        # Check for periodic
        t_periodic = False
        lat_vecs = None
        periodic = root.find('Periodic')
        if periodic is not None:
            t_periodic = True
            lat_vecs = np.zeros((3, 3))
            for i, vec in enumerate(periodic.findall('LatVec')):
                lat_vecs[i] = [float(x) for x in vec.text.split()]
        
        self.geometry = {
            'natoms': natoms,
            'coords': coords,
            'species': species,
            'species_names': species_names,
            't_periodic': t_periodic,
            'lat_vecs': lat_vecs
        }
        
        # Parse k-points
        kp_node = root.find('KPoints')
        if kp_node is not None:
            nkp = int(kp_node.get('number'))
            kpoints = np.zeros((3, nkp))
            for i, kp in enumerate(kp_node.findall('KPoint')):
                kpoints[:, i] = [float(x) for x in kp.text.split()]
            self.kpoints = kpoints
        else:
            self.kpoints = np.zeros((3, 1))
            
        # Parse occupations
        occ_node = root.find('Occupations')
        if occ_node is not None:
            nspin = int(occ_node.get('nSpin'))
            nstate = int(occ_node.get('nStates'))
            nkpt = int(occ_node.get('nKPoints'))
            self.occupations = np.zeros((nstate, nkpt, nspin))
            for spin in range(nspin):
                for ik in range(nkpt):
                    occ_str = occ_node.get(f'spin{spin+1}_k{ik+1}')
                    if occ_str:
                        self.occupations[:, ik, spin] = [float(x) for x in occ_str.split()]
        
        # Real or complex
        real_node = root.find('Real')
        if real_node is not None:
            self.t_real = real_node.text.lower() == 'true'
            
        if self.verbosity > 0:
            print(f"[DFTBplusParser] Parsed detailed.xml:")
            print(f"  Identity: {self.identity}")
            print(f"  Atoms: {natoms}, Species: {len(species_names)}")
            print(f"  K-points: {self.kpoints.shape[1]}")
            print(f"  Real: {self.t_real}")
            
        return self.geometry
    
    def parse_eigenvec_bin(self, bin_path=None, nstates=None, nkpoints=None, nspin=None, norb=None):
        """Parse binary eigenvector file (Fortran unformatted)."""
        if bin_path is None:
            bin_path = self.work_dir / 'eigenvec.bin'
        
        with open(bin_path, 'rb') as f:
            # Read identity (Fortran record marker + int + record marker)
            rec_start = struct.unpack('i', f.read(4))[0]
            identity = struct.unpack('i', f.read(4))[0]
            rec_end = struct.unpack('i', f.read(4))[0]
            
            if self.verbosity > 0:
                print(f"[DFTBplusParser] eigenvec.bin identity: {identity}")
            
            # Determine dimensions from detailed.xml if available
            if nstates is None and self.occupations is not None:
                nstates = self.occupations.shape[0]
                nkpoints = self.occupations.shape[1]
                nspin = self.occupations.shape[2]
            
            if norb is None and self.basis is not None:
                norb = sum(self.basis['norb_per_atom'])
            
            if any(x is None for x in [nstates, nkpoints, nspin, norb]):
                raise ValueError("Must provide nstates, nkpoints, nspin, norb or parse detailed.xml and basis first")
            
            # Read eigenvectors
            if self.t_real:
                eigenvecs = np.zeros((norb, nstates, nkpoints, nspin), dtype=np.float64)
            else:
                eigenvecs = np.zeros((norb, nstates, nkpoints, nspin), dtype=np.complex128)
            
            for ispin in range(nspin):
                for ik in range(nkpoints):
                    for istate in range(nstates):
                        # Read record markers and data
                        rec_start = struct.unpack('i', f.read(4))[0]
                        if self.t_real:
                            data = np.fromfile(f, dtype=np.float64, count=norb)
                        else:
                            data = np.fromfile(f, dtype=np.float64, count=2*norb)
                            data = data[0::2] + 1j * data[1::2]
                        rec_end = struct.unpack('i', f.read(4))[0]
                        eigenvecs[:, istate, ik, ispin] = data
            
        self.eigenvectors = eigenvecs
        
        if self.verbosity > 0:
            print(f"[DFTBplusParser] Parsed eigenvec.bin: shape {eigenvecs.shape}")
            
        return eigenvecs
    
    def parse_basis_hsd(self, hsd_path=None):
        """Parse basis definition from HSD format."""
        if hsd_path is None:
            # Try to find in work_dir
            candidates = list(self.work_dir.glob('*.hsd')) + list(self.work_dir.glob('wfc.*.hsd'))
            if candidates:
                hsd_path = candidates[0]
        
        if hsd_path is None:
            raise FileNotFoundError("Could not find basis HSD file")
            
        hsd_path = Path(hsd_path)
        
        # Simple HSD parser (not full implementation, just basis extraction)
        species_basis = []
        current_species = None
        current_orbitals = []
        
        with open(hsd_path, 'r') as f:
            content = f.read()
            
        # Parse basis block
        import re
        
        # Find Basis block
        basis_match = re.search(r'Basis\s*{([^}]*)}', content, re.DOTALL)
        if not basis_match:
            raise ValueError("No Basis block found in HSD")
        
        basis_content = basis_match.group(1)
        
        # Parse species entries
        species_pattern = r'(\w+)\s*{([^}]*)}'
        for match in re.finditer(species_pattern, basis_content):
            species_name = match.group(1)
            species_block = match.group(2)
            
            # Parse atomic number
            an_match = re.search(r'AtomicNumber\s*=\s*(\d+)', species_block)
            atomic_number = int(an_match.group(1)) if an_match else 0
            
            # Parse resolution
            res_match = re.search(r'Resolution\s*=\s*(\S+)', species_block)
            resolution = float(res_match.group(1)) if res_match else 0.02
            
            # Parse orbitals
            orbitals = []
            orb_pattern = r'Orbital\s*{([^}]*)}'
            for orb_match in re.finditer(orb_pattern, species_block):
                orb_block = orb_match.group(1)
                
                # Parse angular momentum
                l_match = re.search(r'AngularMomentum\s*=\s*(\d+)', orb_block)
                l = int(l_match.group(1)) if l_match else 0
                
                # Parse occupation
                occ_match = re.search(r'Occupation\s*=\s*(\S+)', orb_block)
                occ = float(occ_match.group(1)) if occ_match else 0.0
                
                # Parse cutoff
                cut_match = re.search(r'Cutoff\s*=\s*(\S+)', orb_block)
                cutoff = float(cut_match.group(1)) if cut_match else 10.0
                
                # Parse exponents
                exp_match = re.search(r'Exponents\s*=\s*{([^}]*)}', orb_block)
                if exp_match:
                    exponents = [float(x) for x in exp_match.group(1).split()]
                else:
                    exponents = []
                
                # Parse coefficients
                coef_match = re.search(r'Coefficients\s*=\s*{([^}]*)}', orb_block)
                if coef_match:
                    all_coefs = [float(x) for x in coef_match.group(1).split()]
                    nexp = len(exponents)
                    npow = len(all_coefs) // nexp if nexp > 0 else 0
                    coefficients = np.array(all_coefs).reshape(nexp, npow).T  # (nPow, nAlpha)
                else:
                    coefficients = np.array([])
                
                orbitals.append({
                    'l': l,
                    'occupation': occ,
                    'cutoff': cutoff,
                    'exponents': np.array(exponents),
                    'coefficients': coefficients,  # (nPow, nAlpha)
                    'nAlpha': len(exponents),
                    'nPow': coefficients.shape[0] if coefficients.size > 0 else 0
                })
            
            species_basis.append({
                'name': species_name,
                'atomic_number': atomic_number,
                'resolution': resolution,
                'orbitals': orbitals
            })
        
        self.basis = {
            'species': species_basis,
            'nspecies': len(species_basis)
        }
        
        # Compute orbital layout per atom
        if self.geometry is not None:
            norb_per_atom = []
            for ia in range(self.geometry['natoms']):
                ispec = self.geometry['species'][ia]
                orb_count = sum(2*orb['l'] + 1 for orb in species_basis[ispec]['orbitals'])
                norb_per_atom.append(orb_count)
            
            orb_offsets = np.zeros(self.geometry['natoms'] + 1, dtype=np.int32)
            orb_offsets[1:] = np.cumsum(norb_per_atom)
            
            self.basis['norb_per_atom'] = np.array(norb_per_atom, dtype=np.int32)
            self.basis['orb_offsets'] = orb_offsets
            self.basis['total_norb'] = orb_offsets[-1]
        
        if self.verbosity > 0:
            print(f"[DFTBplusParser] Parsed basis: {len(species_basis)} species")
            for sp in species_basis:
                print(f"  {sp['name']} (Z={sp['atomic_number']}): {len(sp['orbitals'])} orbitals")
        
        return self.basis
    
    def load_all(self, work_dir=None):
        """Load all DFTB+ output files."""
        if work_dir is not None:
            self.work_dir = Path(work_dir)
        
        self.parse_detailed_xml()
        self.parse_basis_hsd()
        self.parse_eigenvec_bin()
        
        return {
            'geometry': self.geometry,
            'basis': self.basis,
            'eigenvectors': self.eigenvectors,
            'kpoints': self.kpoints,
            'occupations': self.occupations,
            't_real': self.t_real
        }


BOHR2ANG = 0.5291772109  # Bohr -> Angstrom


def parse_basis_hsd_ang(hsd_path):
    """
    Parse waveplot_in.hsd Basis block and return species_list ready for load_basis_sto().

    All parameters in the HSD are in Bohr (DFTB+ internal units). This function
    converts them to Angstrom for use with OCL kernels that work in Angstrom:
      alpha_Ang  = alpha_Bohr  / BOHR2ANG
      cutoff_Ang = cutoff_Bohr * BOHR2ANG
      res_Ang    = res_Bohr    * BOHR2ANG
      coeff_Ang  = coeff_Bohr  / BOHR2ANG^l   (r^l carries units)

    Returns list of species dicts compatible with GridProjector.load_basis_sto().
    """
    import re
    B = BOHR2ANG
    with open(hsd_path, 'r') as f:
        content = f.read()

    # Parse top-level Resolution
    res_match = re.search(r'Basis\s*\{[^}]*Resolution\s*=\s*(\S+)', content, re.DOTALL)
    res_bohr = float(res_match.group(1)) if res_match else 0.04

    # Find Basis block (handles nested braces via simple depth tracking)
    start = content.find('Basis')
    depth = 0; basis_content = ''
    for i, ch in enumerate(content[start:], start):
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0: basis_content = content[start:i+1]; break

    species_list = []
    # Match species blocks: 'Name = {' or 'Name{' — skip HSD keywords
    _SKIP = {'Basis', 'Orbital', 'Options', 'PlottedRegion', 'Box', 'Origin',
             'Resolution', 'AtomicNumber', 'AngularMomentum', 'Occupation',
             'Cutoff', 'Exponents', 'Coefficients', 'NrOfPoints', 'RealComponent',
             'Verbose', 'PlottedLevels', 'PlottedKPoints', 'PlottedSpins', 'GroundState'}
    sp_pat = re.compile(r'\b([A-Z][a-zA-Z0-9]*)\s*=\s*\{')  # species: capital letter, then = {
    i = basis_content.find('{') + 1  # skip opening Basis {
    while i < len(basis_content) - 1:
        m = sp_pat.search(basis_content, i)
        if not m: break
        sp_name = m.group(1)
        if sp_name in _SKIP: i = m.end(); continue
        # Extract species block (the { starting at m.end()-1)
        b_start = m.end() - 1
        depth = 0; b_end = b_start
        for j, ch in enumerate(basis_content[b_start:], b_start):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0: b_end = j; break
        sp_block = basis_content[b_start:b_end+1]
        i = b_end + 1

        an_m = re.search(r'AtomicNumber\s*=\s*(\d+)', sp_block)
        atomic_number = int(an_m.group(1)) if an_m else 0

        orbitals = []
        orb_pat = re.compile(r'Orbital\s*=\s*\{')
        oi = 0
        while True:
            om = orb_pat.search(sp_block, oi)
            if not om: break
            od = 0; oe = om.start()
            for j, ch in enumerate(sp_block[om.start():], om.start()):
                if ch == '{': od += 1
                elif ch == '}':
                    od -= 1
                    if od == 0: oe = j; break
            orb_block = sp_block[om.start():oe+1]
            oi = oe + 1

            l_m   = re.search(r'AngularMomentum\s*=\s*(\d+)', orb_block)
            cut_m = re.search(r'Cutoff\s*=\s*(\S+)', orb_block)
            exp_m = re.search(r'Exponents\s*=\s*\{([^}]*)\}', orb_block)
            cof_m = re.search(r'Coefficients\s*=\s*\{([^}]*)\}', orb_block)

            l      = int(l_m.group(1)) if l_m else 0
            cutoff_b = float(cut_m.group(1)) if cut_m else 10.0
            exps_b = np.array([float(x) for x in exp_m.group(1).split()]) if exp_m else np.array([1.0])
            if cof_m:
                all_c = np.array([float(x) for x in cof_m.group(1).split()])
                nexp  = len(exps_b)
                npow  = len(all_c) // nexp
                coef_b = all_c.reshape(nexp, npow).T  # (nPow, nAlpha)
            else:
                coef_b = np.ones((1, len(exps_b)))

            # Convert Bohr -> Angstrom
            orbitals.append({
                'l':            l,
                'cutoff':       cutoff_b * B,
                'exponents':    exps_b   / B,
                'coefficients': coef_b   / (B ** l),   # r^l unit correction
            })

        species_list.append({
            'name':          sp_name,
            'atomic_number': atomic_number,
            'resolution':    res_bohr * B,
            'orbitals':      orbitals,
        })

    return species_list


def compute_sto_radial(r, aa, alpha, ll):
    """
    Compute Slater-type orbital radial part analytically.
    
    Formula: R_l(r) = sum_{i=1}^{nAlpha} [ sum_{j=1}^{nPow} aa(j,i) * r^{l+j-1} ] * exp(-alpha_i * r)
    
    Args:
        r: radial distance (scalar or array)
        aa: (nPow, nAlpha) coefficients
        alpha: (nAlpha,) exponents
        ll: angular momentum
    
    Returns:
        STO radial value
    """
    aa = np.asarray(aa)
    alpha = np.asarray(alpha)
    r = np.asarray(r)
    
    nAlpha = len(alpha)
    nPow = aa.shape[0] if aa.ndim > 1 else 1
    
    result = np.zeros_like(r, dtype=np.float64)
    
    for i_alpha in range(nAlpha):
        exp_term = np.exp(-alpha[i_alpha] * r)
        
        for i_pow in range(nPow):
            coef = aa[i_pow, i_alpha] if aa.ndim > 1 else aa[i_alpha]
            power = ll + i_pow
            
            # Handle r=0 case for l=0, p=1 (r^0 = 1)
            if power == 0:
                r_pow = np.ones_like(r)
            else:
                r_pow = r ** power
            
            result += coef * r_pow * exp_term
    
    return result


def precompute_sto_grid(aa, alpha, ll, cutoff, resolution, n_nodes=None, dr=None):
    """
    Precompute STO values on uniform grid with spline second derivatives.
    
    Args:
        aa: (nPow, nAlpha) coefficients
        alpha: (nAlpha,) exponents
        ll: angular momentum
        cutoff: cutoff radius
        resolution: grid spacing (used if n_nodes/dr not provided)
        n_nodes: optional number of grid points (overrides cutoff/resolution)
        dr: optional grid spacing (overrides resolution)
    
    Returns:
        grid_values: (nNodes,) STO values
        grid_d2: (nNodes,) second derivatives for cubic spline
        dr: grid spacing
    """
    if n_nodes is None or dr is None:
        n_nodes = int(np.ceil(cutoff / resolution)) + 2
        dr = resolution
    
    r = np.arange(n_nodes) * dr
    
    # Evaluate STO at each grid point
    grid_values = compute_sto_radial(r, aa, alpha, ll)
    grid_values = grid_values.astype(np.float32)
    
    # Compute second derivatives (natural cubic spline)
    grid_d2 = _spline_d2_uniform(grid_values, dr)
    
    return grid_values, grid_d2, dr


def _spline_d2_uniform(y, h):
    """Natural cubic spline second derivatives for uniform grid."""
    n = len(y)
    if n < 3:
        return np.zeros(n, dtype=np.float32)
    
    a = np.ones(n-3, dtype=np.float64)
    b = np.full(n-2, 4.0, dtype=np.float64)
    c = np.ones(n-3, dtype=np.float64)
    rhs = np.zeros(n-2, dtype=np.float64)
    rhs[:] = 6.0 * (y[2:] - 2.0*y[1:-1] + y[:-2]) / (h*h)
    
    # Thomas algorithm
    for i in range(1, n-2):
        w = a[i-1] / b[i-1]
        b[i] -= w * c[i-1]
        rhs[i] -= w * rhs[i-1]
    
    d2_inner = np.zeros(n-2, dtype=np.float64)
    d2_inner[-1] = rhs[-1] / b[-1]
    for i in range(n-4, -1, -1):
        d2_inner[i] = (rhs[i] - c[i] * d2_inner[i+1]) / b[i]
    
    d2 = np.zeros(n, dtype=np.float32)
    d2[1:-1] = d2_inner.astype(np.float32)
    
    return d2
