# USER

I would like to reimplement the projection of wave function (molecular orbitals) from dftb to real space grid (or on specific points) myself manually. How to do that? I already did something similar for fireball-dft which is numerical local basiset (LCAO) (s,PX,py,Pz), do it is quite similar, I only need radial wave functions corresponding to given Slater Coster files. Where can I obtain these?

Then I need to obtain orbital coefficients from dftb.  C_i,nu where I is index of molecular orbital and nu index of basis function.

In which source code file I can see how this is implemented in dftb or in that Wavetool \ waveworks which is used for external projection of Cube Files

---

# Kimi 2.6

Here is a detailed tutorial for manually reimplementing wavefunction projection from DFTB+ to real-space grid, including the exact mathematical formulas, file formats, and source code references.

## 1. Mathematical Foundation

### 1.1 LCAO Wavefunction in DFTB+

The molecular orbital (MO) wavefunction is expanded in atom-centered basis functions:

$$\psi_i(\mathbf{r}) = \sum_{\mu} c_{\mu i} \phi_\mu(\mathbf{r})$$

where:
- $i$ = MO index
- $\mu = (A, l, m)$ = basis function index (atom $A$, angular momentum $l$, magnetic quantum number $m$)
- $c_{\mu i}$ = eigenvector coefficients from DFTB+ diagonalization
- $\phi_\mu(\mathbf{r})$ = Slater-type orbital (STO) basis function 

### 1.2 STO Basis Function Form

DFTB+ uses **numerical Slater-type orbitals** with the radial form :

$$R_l(r) = \sum_{p=1}^{n_{\text{pow}}} \sum_{i=1}^{n_{\text{exp}}} c_{p,i} \, r^{l + p - 1} e^{-\alpha_i r}$$

where:
- $l$ = angular momentum (0 for s, 1 for p, 2 for d)
- $n_{\text{pow}}$ = number of power terms
- $n_{\text{exp}}$ = number of exponential terms  
- $c_{p,i}$ = contraction coefficients (from `wfc.*.hsd` file)
- $\alpha_i$ = exponents (from `wfc.*.hsd` file)

The **full basis function** is:

$$\phi_{nlm}(\mathbf{r}) = R_l(|\mathbf{r} - \mathbf{R}_A|) Y_l^m(\theta, \phi)$$

where $Y_l^m$ are real spherical harmonics (not complex, since DFTB+ uses real arithmetic for non-periodic systems) .

## 2. Required Input Files

### 2.1 Wavefunction Coefficients (`wfc.*.hsd`)

These contain the STO radial function parameters. Example from `wfc.mio-1-1.hsd` for hydrogen :

```hsd
H {
  1 0 0 {    # n=1, l=0 (s-orbital), species index 0
    1.000000  # c_{1,1} - contraction coefficient for power=1, exp=1
    1.200000  # α_1 - exponent
    0.000000  # c_{2,1} - power=2, exp=1
    0.000000  # c_{3,1} - power=3, exp=1
    # ... more coefficients if nexp > 1
  }
}
```

**Format**: For each element, blocks of `(n_power, n_exp, coefficients...)` where coefficients are ordered with **p-index growing fastest** .

### 2.2 Eigenvector Files (`eigenvec.out` or `eigenvec.bin`)

**Text format** (`eigenvec.out`) - easier to parse :
```
Eigenvector 1
K-point 1 Spin 1
Atom 1 (O) Orbital s  -0.123456  0.001234
Atom 1 (O) Orbital px  0.234567  0.002345
...
```

Each line: `Atom N (Species) Orbital type  coefficient  Mulliken_population`

**Binary format** (`eigenvec.bin`) :
- Header: `runId` (unique simulation ID, integer)
- Data: eigenvector coefficients in Fortran-native binary
- Ordering: for each spin channel, for each k-point, for each eigenvector: all basis function coefficients sequentially

### 2.3 System Information (`detailed.xml`)

Contains atom positions, species, lattice vectors, and orbital-to-atom mapping .

## 3. Python Implementation

### 3.1 Parse WFC File (STO Parameters)

```python
import re
import numpy as np
from scipy.special import sph_harm, factorial

def parse_wfc_file(filename):
    """
    Parse DFTB+ wavefunction coefficient file.
    Returns dict: element -> {l: [(coeffs, exponents), ...]}
    """
    elements = {}
    current_element = None
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Simple HSD parser for wfc format
    # Format: Element { n l m { c1 a1 c2 a2 ... } }
    
    # Find each element block
    element_pattern = r'(\w+)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
    for match in re.finditer(element_pattern, content, re.DOTALL):
        element = match.group(1)
        block = match.group(2)
        elements[element] = {}
        
        # Parse orbital blocks: n l m { coeffs... }
        orb_pattern = r'(\d+)\s+(\d+)\s+(\d+)\s*\{([^}]*)\}'
        for orb_match in re.finditer(orb_pattern, block):
            n, l, m = int(orb_match.group(1)), int(orb_match.group(2)), int(orb_match.group(3))
            coeffs_str = orb_match.group(4).strip().split()
            coeffs = [float(x) for x in coeffs_str]
            
            # coeffs are: c_{1,1}, alpha_1, c_{2,1}, alpha_1, ..., c_{1,2}, alpha_2, ...
            # Actually: p-index grows fastest: c_{p,i} with p=1..npow, i=1..nexp
            # So: c_11, a_1, c_21, a_1, c_31, a_1, c_12, a_2, ...
            
            # Determine n_pow and n_exp from data length
            # Each (c, alpha) pair is 2 floats
            n_pairs = len(coeffs) // 2
            
            # Reconstruct: assume n_pow = number of unique powers
            # For minimal basis: H(s) -> 1 power, 1 exp -> 2 floats
            # O(s,p) -> s: 1 power, 1 exp; p: 1 power, 1 exp
            
            # Group by exponent (alpha)
            alphas = []
            c_groups = []
            for i in range(0, len(coeffs), 2):
                c, alpha = coeffs[i], coeffs[i+1]
                if alpha not in alphas:
                    alphas.append(alpha)
                    c_groups.append([])
                idx = alphas.index(alpha)
                c_groups[idx].append(c)
            
            n_pow = len(c_groups[0]) if c_groups else 0
            n_exp = len(alphas)
            
            if l not in elements[element]:
                elements[element][l] = []
            
            elements[element][l].append({
                'n_pow': n_pow,
                'n_exp': n_exp,
                'coeffs': np.array(c_groups).T,  # shape: (n_pow, n_exp)
                'exponents': np.array(alphas)
            })
    
    return elements

# Example usage
wfc_data = parse_wfc_file('wfc.mio-1-1.hsd')
print(wfc_data['H'][0][0]['exponents'])  # s-orbital exponents for H
```

### 3.2 Evaluate STO Radial Function

```python
def sto_radial(r, l, coeffs, exponents):
    """
    Evaluate STO radial function R_l(r).
    
    Parameters:
        r: radial distance (scalar or array, in Bohr)
        l: angular momentum quantum number
        coeffs: (n_pow, n_exp) array of contraction coefficients
        exponents: (n_exp,) array of exponents
    
    Returns:
        R_l(r): radial function value
    """
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    n_pow, n_exp = coeffs.shape
    
    for i_exp in range(n_exp):
        alpha = exponents[i_exp]
        for i_pow in range(n_pow):
            c = coeffs[i_pow, i_exp]
            power = l + i_pow  # r^{l + p - 1} with p = i_pow + 1
            result += c * r**power * np.exp(-alpha * r)
    
    return result

def real_spherical_harmonic(l, m, theta, phi):
    """
    Real spherical harmonics (DFTB+ uses real form).
    """
    # Convert to complex spherical harmonics then take real part
    if m > 0:
        return np.sqrt(2) * (-1)**m * np.real(sph_harm(m, l, phi, theta))
    elif m == 0:
        return np.real(sph_harm(0, l, phi, theta))
    else:  # m < 0
        return np.sqrt(2) * (-1)**m * np.imag(sph_harm(-m, l, phi, theta))
```

### 3.3 Parse Eigenvectors

```python
def parse_eigenvec_out(filename):
    """
    Parse DFTB+ eigenvec.out text file.
    Returns: list of eigenvectors, each is dict with coefficients and metadata
    """
    eigenvectors = []
    current_ev = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('Eigenvector'):
            if current_ev:
                eigenvectors.append(current_ev)
            current_ev = {
                'index': int(line.split()[1]),
                'kpoint': 1,
                'spin': 1,
                'coeffs': [],
                'atoms': [],
                'orbitals': []
            }
            i += 1
        elif line.startswith('K-point'):
            current_ev['kpoint'] = int(line.split()[1])
            current_ev['spin'] = int(line.split()[3])
            i += 1
        elif line.startswith('Atom'):
            # Format: "Atom N (Species) Orbital type  coeff  population"
            parts = line.split()
            atom_num = int(parts[1])
            species = parts[2].strip('()')
            orbital = parts[4]
            coeff = float(parts[5])
            
            current_ev['atoms'].append(atom_num)
            current_ev['orbitals'].append(orbital)
            current_ev['coeffs'].append(coeff)
            i += 1
        else:
            i += 1
    
    if current_ev:
        eigenvectors.append(current_ev)
    
    # Convert to numpy arrays
    for ev in eigenvectors:
        ev['coeffs'] = np.array(ev['coeffs'])
    
    return eigenvectors
```

### 3.4 Build Basis Function Index

```python
def build_basis_index(atoms, max_angular_momentum):
    """
    Build mapping from basis function index to (atom, l, m).
    
    Parameters:
        atoms: list of atomic symbols
        max_angular_momentum: dict {element: max_l}
    
    Returns:
        list of (atom_idx, l, m) tuples
    """
    basis_index = []
    
    for i_atom, symbol in enumerate(atoms):
        max_l = max_angular_momentum.get(symbol, 0)
        
        # s orbitals (l=0, m=0)
        basis_index.append((i_atom, 0, 0))
        
        # p orbitals (l=1, m=-1,0,1 -> px, py, pz in real form)
        if max_l >= 1:
            basis_index.append((i_atom, 1, -1))  # px
            basis_index.append((i_atom, 1, 0))   # py  
            basis_index.append((i_atom, 1, 1))   # pz
        
        # d orbitals (l=2, m=-2,-1,0,1,2)
        if max_l >= 2:
            for m in range(-2, 3):
                basis_index.append((i_atom, 2, m))
    
    return basis_index
```

### 3.5 Evaluate Wavefunction on Grid

```python
def evaluate_mo_on_grid(eigenvector, atoms, positions, wfc_data, 
                        grid_origin, grid_vectors, grid_shape):
    """
    Evaluate molecular orbital on a regular 3D grid.
    
    Parameters:
        eigenvector: dict with 'coeffs' array
        atoms: list of atomic symbols
        positions: (N, 3) array of atom positions in Bohr
        wfc_data: parsed WFC data
        grid_origin: (3,) origin in Bohr
        grid_vectors: (3, 3) grid vectors in Bohr
        grid_shape: (nx, ny, nz) grid dimensions
    
    Returns:
        (nx, ny, nz) array of MO values
    """
    nx, ny, nz = grid_shape
    mo_values = np.zeros((nx, ny, nz))
    
    # Build basis index
    max_angmom = {'H': 0, 'O': 1, 'C': 1, 'N': 1}  # example
    basis_index = build_basis_index(atoms, max_angmom)
    
    coeffs = eigenvector['coeffs']
    
    # Generate grid points
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    z = np.linspace(0, nz-1, nz)
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Grid point in real space
                grid_point = (grid_origin + 
                             ix * grid_vectors[0] / nx +
                             iy * grid_vectors[1] / ny +
                             iz * grid_vectors[2] / nz)
                
                # Evaluate all basis functions at this point
                psi = 0.0
                for i_basis, (i_atom, l, m) in enumerate(basis_index):
                    atom_pos = positions[i_atom]
                    symbol = atoms[i_atom]
                    
                    # Vector from atom to grid point
                    r_vec = grid_point - atom_pos
                    r = np.linalg.norm(r_vec)
                    
                    if r < 1e-10:
                        continue  # Avoid division by zero
                    
                    # Spherical coordinates
                    theta = np.arccos(r_vec[2] / r)
                    phi = np.arctan2(r_vec[1], r_vec[0])
                    
                    # Get STO parameters
                    sto_params = wfc_data[symbol][l][0]  # first entry for this l
                    
                    # Evaluate radial and angular parts
                    R = sto_radial(r, l, sto_params['coeffs'], sto_params['exponents'])
                    Y = real_spherical_harmonic(l, m, theta, phi)
                    
                    # Basis function value
                    phi_val = R[0] * Y  # R returns array, take first element
                    
                    # Add to MO with coefficient
                    psi += coeffs[i_basis] * phi_val
                
                mo_values[ix, iy, iz] = psi
    
    return mo_values
```

## 4. Source Code Reference: Where to Look in DFTB+

### 4.1 Waveplot Source Files

Located in `app/waveplot/` in the DFTB+ repository :

| File | Purpose |
|------|---------|
| `waveplot.F90` | Main driver, grid setup, output formatting |
| `slater.F90` | **STO radial function evaluation** — the core $R_l(r)$ implementation |
| `wfc.F90` | **WFC file parser** — reads `wfc.*.hsd` format |
| `gridcache.F90` | Grid caching for performance |
| `input.F90` | HSD input parsing |

**Key subroutine in `slater.F90`** (pseudocode from source):
```fortran
subroutine getValue(pSlater, rr, ang, val)
    ! pSlater: STO parameters (coeffs, exponents)
    ! rr: radial distance r
    ! ang: angular part (spherical harmonics)
    ! val: output value
    
    rad = 0.0
    do i_pow = 1, n_pow
        do i_exp = 1, n_exp
            c = coeffs(i_pow, i_exp)
            alpha = exponents(i_exp)
            power = l + i_pow - 1  ! r^{l + p - 1}
            rad = rad + c * rr**power * exp(-alpha * rr)
        end do
    end do
    
    val = rad * ang
end subroutine
```

### 4.2 Eigenvector Binary Format

In `src/dftbp/io/formatout.F90` or similar I/O module :

```fortran
! eigenvec.bin format:
! Header: integer runId
! Data: for each spin, for each kpoint, for each eigenstate:
!       real(coeffs(1:n_basis))  ! Fortran unformatted sequential
```

The binary format is **Fortran unformatted sequential** — each `write()` statement adds record markers. To read in Python, you may need to skip 4-byte integers before and after each record (compiler-dependent).

### 4.3 Basis Function Indexing

In `src/dftbp/dftbplus/parser.F90` or `main.F90`, the basis function ordering is established:
1. Loop over atoms
2. For each atom, loop over shells (s, p, d, ...)
3. For each shell, loop over m values (-l to +l)

This matches the `build_basis_index()` function above.

## 5. Complete Workflow Script

```python
#!/usr/bin/env python3
"""
Manual DFTB+ wavefunction projection to real-space grid.
"""

import numpy as np
from ase.io import read
from ase.io.cube import write_cube

# 1. Parse WFC file
wfc_data = parse_wfc_file('wfc.mio-1-1.hsd')

# 2. Read geometry from DFTB+ gen file or ASE
atoms = read('geo_end.gen')
positions = atoms.positions * 1.889726  # Angstrom to Bohr
symbols = atoms.get_chemical_symbols()

# 3. Parse eigenvectors
eigenvectors = parse_eigenvec_out('eigenvec.out')

# 4. Define grid (can also read from detailed.xml)
grid_origin = np.array([-5.0, -5.0, -5.0])  # Bohr
grid_vectors = np.array([[10.0, 0.0, 0.0],
                         [0.0, 10.0, 0.0],
                         [0.0, 0.0, 10.0]])  # Bohr
grid_shape = (50, 50, 50)

# 5. Evaluate HOMO (orbital 4 for H2O)
homo = eigenvectors[3]  # 0-indexed
mo_grid = evaluate_mo_on_grid(homo, symbols, positions, wfc_data,
                              grid_origin, grid_vectors, grid_shape)

# 6. Write cube file
write_cube(open('homo_manual.cube', 'w'), atoms, mo_grid**2)

# 7. For density: sum over occupied states
density = np.zeros(grid_shape)
for ev in eigenvectors:
    if np.any(ev['coeffs'] != 0):  # Check occupation
        mo = evaluate_mo_on_grid(ev, symbols, positions, wfc_data,
                                grid_origin, grid_vectors, grid_shape)
        density += mo**2 * 2.0  # occupation factor

write_cube(open('density_manual.cube', 'w'), atoms, density)
```

## 6. Key Differences from Fireball

| Aspect | Fireball | DFTB+ |
|--------|----------|-------|
| **Radial functions** | Numerical pseudopotential orbitals | Analytical STO expansion |
| **Angular part** | Spherical harmonics | Real spherical harmonics |
| **File format** | Custom binary | `eigenvec.out` text / `eigenvec.bin` Fortran binary |
| **Parameters** | `.wf` files | `wfc.*.hsd` files |
| **Basis indexing** | Atom-shell-m | Same, but check ordering |

The main adaptation needed from your Fireball code is replacing the numerical radial function evaluation with the **STO analytical formula** and reading the **WFC parameter files** instead of Fireball's `.wf` files.

---

# USER

plese look in this codebase how to reimplement waveplot in OpenCL based on this notes. 
Our fireball OpenCL code is here
@Grid.py @Grid.cl 

can we adapt it easily to use slater-koster DFTB framework 
@waveplot.F90 

your job now it not write any code but make detaield code review with notes where to find information, what are the protocols/data layouts etc relevant varaibles and functions, and make detailed notes which will help us to make well defined detailed plan for the implementation


---

# SWE-1.6


I'll analyze the codebase to understand how to reimplement waveplot in OpenCL. Let me create a plan and start examining the relevant files.

# Detailed Code Review: Waveplot OpenCL Reimplementation Analysis

## Executive Summary

This analysis compares the existing DFTB+ waveplot implementation (Slater-Koster analytical basis) with the Fireball OpenCL Grid projector (numerical basis) to identify adaptation points for GPU acceleration.

## 1. Current DFTB+ Waveplot Architecture

### 1.1 Main Program Flow ([waveplot.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/waveplot.F90:0:0-0:0))

**Key variables and data structures:**

- **`wp%opt%nPoints(3)`** - Grid dimensions (nx, ny, nz)
- **`wp%opt%boxVecs(3,3)`** - Box vectors defining the plotted region
- **`wp%opt%origin(3)`** - Origin of the plotted region
- **`wp%opt%gridOrigin(3)`** - Origin of the grid (may be shifted)
- **`wp%loc%gridVec(3,3)`** - Grid vectors (boxVecs / nPoints)
- **`wp%loc%gridVol`** - Volume of one grid cell
- **`buffer(nx,ny,nz)`** - Working buffer for grid values
- **`gridValReal` / `gridValCmpl`** - Pointers to cached grid data

**Main computation loop (lines 246-332):**

```fortran
lpStates: do while (.not. tFinished)
    call next(wp%loc%grid, gridValReal, levelIndex, tFinished)  ! Get next MO grid
    ! Process: abs^2, real, imag, charge difference
end do lpStates
```

**Output formats:**
- Cube files via `writeCubeFile()` (lines 408-514)
- Supports: charge density, real/imag parts, spin polarization, atomic density

### 1.2 Molecular Orbital Calculation ([molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0))

**Core data structure `TMolecularOrbital` (lines 51-99):**

- **`nAtom`** - Number of atoms
- **`nSpecies`** - Number of species
- **`species(nAtom)`** - Species index for each atom
- **`iStos(nSpecies+1)`** - Index array for STOs (species boundaries)
- **`stos(nOrb)`** - All Slater-type orbitals sequentially
- **`cutoffs(nOrb)`** - Cutoff radius for each STO
- **`angMoms(nOrb)`** - Angular momentum for each STO
- **`nOrb`** - Total number of orbitals (including m-dependence)
- **`tPeriodic`** - Whether system is periodic
- **`latVecs(3,3)`** - Lattice vectors
- **`cellVec(3,nCell)`** - Cell translation vectors
- **`coords(3,nAtom,nCell)`** - Atom coordinates in all cells

**Key algorithm `localGetValue()` (lines 309-498):**

```fortran
do i3 = 1, nPoints(3)
  do i2 = 1, nPoints(2)
    do i1 = 1, nPoints(1)
      xyz = origin + gridVecs * [i1,i2,i3]
      ! For each atom in each cell:
      do iCell = 1, nCell
        do iAtom = 1, nAtom
          diff = xyz - coords(:, iAtom, iCell)
          rr = norm2(diff)
          if (rr <= cutoff) then
            call getValue(stos(iOrb), rr, val)  ! Radial part
            ang = realTessY(l, m, diff, rr)     ! Angular part
            atomAllOrbVal(ind) = val * ang
          end if
        end do
      end do
      ! Contract with eigenvector coefficients
      valueOnGrid(i1,i2,i3,iEig) = dot(atomOrbValReal, eigVecsReal(:,iEig))
    end do
  end do
end do
```

**Optimizations:**
- Non-zero mask filtering (lines 457-466) - only compute contributions from orbitals within cutoff
- Periodic phase factors for k-points (line 408)

### 1.3 Slater-Type Orbital Evaluation ([slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0))

**Data structure `TSlaterOrbital` (lines 23-50):**

- **`nPow`** - Max power of distance in STO
- **`nAlpha`** - Number of exponential coefficients
- **`ll`** - Angular momentum
- **`aa(nPow, nAlpha)`** - Summation coefficients
- **`alpha(nAlpha)`** - Exponential coefficients (stored as negative)
- **`gridValue(nGrid)`** - Precomputed STO values on radial grid
- **`gridDist`** - Grid spacing
- **`nGrid`** - Number of grid points

**Analytical formula (lines 254-300):**

```fortran
sto(r) = sum_{i=1}^{nAlpha} [ sum_{j=1}^{nPow} aa(j,i) * r^{ll + j - 1} ] * exp(-alpha(i) * r)
```

**Grid-based interpolation (lines 225-250):**

```fortran
ind = floor(rr / gridDist) + 1
if (ind < nGrid) then
  frac = mod(rr, gridDist) / gridDist
  sto = (1.0 - frac) * gridValue(ind) + frac * gridValue(ind+1)
else
  sto = 0.0
end if
```

**Real spherical harmonics `realTessY()` (lines 63-164):**
- Supports l = 0,1,2,3 (s,p,d,f orbitals)
- Precomputed normalization constants
- Cartesian forms (e.g., for p: x/r, y/r, z/r)

### 1.4 Grid Caching ([gridcache.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/gridcache.F90:0:0-0:0))

**Purpose:** Cache computed grids to avoid recomputation when iterating over multiple MOs.

**Data structure `TGridCache` (lines 30-101):**

- **`molorb`** - Pointer to TMolecularOrbital calculator
- **`gridVec(3,3)`**, **`origin(3)`** - Grid specification
- **`eigenvecReal(nOrb, nCached)`** - Cached eigenvectors
- **`eigenvecCmpl(nOrb, nCached)`** - Complex eigenvectors
- **`gridCacheReal(nx,ny,nz,nCached)`** - Cached grid values
- **`kPoints(3,nKPoint)`** - k-point coordinates (scaled by 2π)
- **`nCached`** - Number of grids to cache

**Caching strategy (lines 327-367):**
1. Read `nCached` eigenvectors from binary file
2. Compute all `nCached` grids in one batch call to `getValue()`
3. Return grids one by one via `next()` interface

### 1.5 Initialization ([initwaveplot.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/initwaveplot.F90:0:0-0:0))

**Data flow:**

1. **Read `detailed.xml`** (lines 390-460):
   - Geometry (atom positions, species)
   - Occupation numbers
   - K-points and weights
   - Real vs complex flag

2. **Read basis from HSD** (lines 746-858):
   - For each species: atomic number, orbitals
   - For each orbital: angular momentum, occupation, cutoff
   - STO parameters: exponents, coefficients
   - Resolution for radial grid

3. **Read options** (lines 502-742):
   - Grid dimensions, plotted region
   - Which levels/k-points/spins to plot
   - Output types (charge, real, imag, etc.)

4. **Initialize TMolecularOrbital** (line 378):
   - Flattens species basis into sequential STO array
   - Computes cell translations for periodic systems
   - Generates coordinates for all periodic images

## 2. Fireball OpenCL Grid Architecture

### 2.1 Python Host Code ([Grid.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:0:0-0:0))

**Key class [GridProjector](cci:2://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:9:0-1403:18):**

**Initialization (lines 14-39):**
- Loads Fdata parser for numerical wavefunctions
- Sets up OpenCL context and queue
- Defines task data structure (lines 33-36):
  ```python
  task_dtype = [('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('w', 'i4'),
                ('na', 'i4'), ('nj', 'i4'), ('pad1', 'i4'), ('pad2', 'i4')]
  ```

**Basis loading [load_basis()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:40:4-164:27) (lines 41-165):**
- **Key difference:** Uses numerical radial functions from `.wf` files
- Resamples all wavefunctions to common uniform grid
- Stores as `float2` (value, second derivative) for cubic spline interpolation
- Packed layout: `[n_species, max_shells, n_nodes, 2]`

**Task building [build_tasks_gpu()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:197:4-320:38) (lines 198-321):**
- Partitions grid into 8×8×8 blocks
- For each block, counts overlapping atoms (sphere-AABB collision)
- Compacts non-empty blocks into task list
- Returns:
  - `tasks_np`: Task metadata (block indices, atom counts)
  - `task_atoms_np`: Atom indices for each task (padded to nMaxAtom)

**Projection [project()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:419:4-582:18) (lines 420-583):**
- Main kernel launch
- Input buffers:
  - `d_grid`: Grid specification (origin, vectors, dimensions)
  - `d_tasks`: Task metadata
  - `d_atoms`: Atom data (position, cutoff, species, orbital info)
  - `d_task_atoms`: Atom indices per task
  - `d_rho`: Density matrix (sparse format)
  - `d_neigh_j`: Neighbor list
  - `d_basis`: Packed radial basis functions
- Output: `d_out` - Grid values

### 2.2 OpenCL Kernel Code ([Grid.cl](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl:0:0-0:0))

**Data structures:**

```c
typedef struct {
    float4 origin;
    float4 dA, dB, dC;  // Grid vectors
    int4 ngrid;
} GridSpec;

typedef struct {
    float4 pos_rcut;  // x,y,z,Rcut
    int type;         // Species index
    int i0orb;        // Start orbital index
    int norb;         // Number of orbitals
    int pad;
} AtomData;

typedef struct {
    int x, y, z, w;   // Block indices
    int na;           // Number of atoms
    int nj;           // Start of j-atom block (for off-diagonal)
    int pad1, pad2;
} TaskData;
```

**Radial interpolation `evaluate_radial()` (lines 55-85):**
- Cubic spline interpolation on uniform grid
- Uses stored second derivatives
- Matches Fortran `getpsi()` with `wf_spline`

**Main projection kernel `project_density_sparse_tiled()` (lines 387-584):**
- **Tiled approach:** Processes 8×8×8 voxel blocks
- Each work-group = one task (block)
- Each work-item processes 32 voxels
- **Tiling over atoms:** Loads TILE_ATOMS×TILE_ATOMS atom pairs into local memory
- **Algorithm:**
  ```c
  for each voxel in block:
    den = 0
    for each atom pair (i,j):
      if within cutoffs:
        psi_i = radial_i(r_i) * angular_i(dri)
        psi_j = radial_j(r_j) * angular_j(drj)
        den += pairsym * dot(psi_i, rho_ij * psi_j)
    out_grid[voxel] = den
  ```

**Orbital projection kernel [project_orbital()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1280:4-1403:18) (lines 597-679):**
- Simpler: projects single MO (not density)
- `psi(r) = sum_i C_i * phi_i(r)`
- Coefficients in order `[px, py, pz, s]`

**Task building kernels (lines 255-381):**
- `count_atoms_per_block`: Counts atoms per block using atomic operations
- `fill_task_atoms`: Fills atom indices per block
- `compact_tasks`: Compacts non-empty blocks

## 3. Critical Differences Between Systems

### 3.1 Basis Function Representation

| Aspect | DFTB+ (waveplot) | Fireball (Grid.py) |
|--------|------------------|-------------------|
| **Type** | Analytical STO | Numerical radial functions |
| **Storage** | Coefficients (aa, alpha) | Tabulated values on grid |
| **Evaluation** | Direct formula or linear interpolation | Cubic spline interpolation |
| **Angular** | Real spherical harmonics (l=0..3) | Same (prefactors match) |
| **Cutoff** | Per-orbital cutoff | Per-orbital cutoff |

**Key implication:** Fireball's `evaluate_radial()` can be reused for DFTB+ by replacing the numerical grid with precomputed STO values.

### 3.2 Data Layout Differences

**DFTB+ orbital indexing:**
- Sequential: all orbitals for all atoms
- Order: atom 1 (s, px, py, pz, ...), atom 2 (s, px, py, pz, ...), ...
- Stored in `TMolecularOrbital%stos(:)` with `iStos(:)` as species boundaries
- Total orbitals: `nOrb = sum_i sum_l (2*l + 1)`

**Fireball orbital indexing:**
- Similar sequential layout
- AtomData has `i0orb` (start index) and `norb` (count)
- Coefficients packed as `[natoms * numorb_max]`

**Compatibility:** The layouts are compatible. DFTB+ needs to compute `i0orb` and `norb` per atom.

### 3.3 Density Matrix vs. Eigenvector

**DFTB+ waveplot:**
- Computes MO values directly: `psi_i(r) = sum_mu C_{mu,i} * phi_mu(r)`
- Then squares for density: `rho(r) = |psi(r)|^2`
- No explicit density matrix storage

**Fireball Grid:**
- Projects precomputed density matrix: `rho(r) = sum_{i,j} rho_{ij} * phi_i(r) * phi_j(r)`
- Density matrix in sparse format: `[natoms][neigh_max][numorb_max][numorb_max]`
- Neighbor list `neigh_j[iatom][ineigh] = jatom + 1`

**Adaptation strategy:** For waveplot, we should use the simpler eigenvector projection approach (like [project_orbital()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1280:4-1403:18) kernel) rather than density matrix projection.

### 3.4 Periodic Boundary Conditions

**DFTB+ waveplot:**
- Explicit cell translations: `cellVec(3, nCell)`
- Phase factors for k-points: `exp(i * k · R_cell)`
- Coordinates in all cells: `coords(3, nAtom, nCell)`

**Fireball Grid:**
- No explicit periodic handling in current code
- Assumes cluster or handles periodicity at higher level

**Adaptation needed:** Add periodic cell support to OpenCL kernel, similar to DFTB+ approach.

## 4. Protocol and Data Flow Analysis

### 4.1 DFTB+ Input Protocol

**Input files:**
1. **`waveplot_in.hsd`** - Main input file
2. **`detailed.xml`** - DFTB+ output with geometry, occupations
3. **`eigenvec.bin`** - Binary eigenvector file (Fortran unformatted)
4. **Basis definition in HSD** - STO parameters

**Data flow:**
```
detailed.xml → TInput (geometry, occupations, k-points)
Basis HSD → TBasis (STO parameters per species)
Options HSD → TOption (grid, output types)
Eigenvec.bin → TEig (eigenvectors)
↓
TMolecularOrbital_init → Flattened STO array
↓
TGridCache_init → Read eigenvectors, compute grids
↓
waveplot main loop → Process grids, write cube files
```

### 4.2 Fireball Input Protocol

**Input files:**
1. **Fdata directory** with `.wf` files (numerical radial functions)
2. **Density matrix** (sparse format)
3. **Neighbor list**
4. **Atom positions and types**

**Data flow:**
```
Fdata parser → load_basis → Packed radial grid
Atoms + cutoffs → build_tasks_gpu → Task list
Density matrix + neighbor list → project → Grid output
```

### 4.3 Key Data Structures Mapping

| DFTB+ | Fireball OpenCL | Notes |
|-------|-----------------|-------|
| `gridVec(3,3)` | `GridSpec.dA, dB, dC` | Direct mapping |
| `origin(3)` | `GridSpec.origin` | Direct mapping |
| `nPoints(3)` | `GridSpec.ngrid` | Direct mapping |
| `coords(3,nAtom)` | `AtomData.pos_rcut.xyz` | Add Rcut in .w |
| `cutoffs(nOrb)` | `AtomData.pos_rcut.w` | Per-orbital vs per-atom |
| `stos(:)` | `basis_data[]` | Need resampling |
| `eigvecsReal(nOrb,nState)` | `coeffs[]` | Layout compatible |
| `species(nAtom)` | `AtomData.type` | Species index |

## 5. Implementation Roadmap

### 5.1 Phase 1: Basis Function Adaptation

**Objective:** Replace Fireball's numerical radial functions with DFTB+ STO evaluation.

**Tasks:**

1. **Create STO grid precomputation** (new Python function):
   - Input: STO parameters (aa, alpha, ll), cutoff, resolution
   - Output: Radial grid values and second derivatives
   - Location: New module `waveplot_basis.py` in `pyBall/OCL/`
   - Reference: [slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0) lines 211-219 (grid initialization)
   - Reference: [slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0) lines 254-300 (explicit evaluation)
   - Reference: [Grid.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:0:0-0:0) lines 91-116 (spline computation)

2. **Adapt [load_basis()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:40:4-164:27) in [Grid.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:0:0-0:0):**
   - Change: Read STO parameters instead of `.wf` files
   - Input: `TSpeciesBasis` from DFTB+ (via Python binding or parsed HSD)
   - Keep: Resampling to common grid
   - Keep: Spline second derivative computation
   - Output: Same packed format `[n_species, max_shells, n_nodes, 2]`

3. **Verify radial interpolation:**
   - Compare `evaluate_radial()` in [Grid.cl](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl:0:0-0:0) with `TSlaterOrbital_getValue()` in [slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0)
   - Ensure matching results at test points

**Key file locations:**
- STO evaluation: [/home/prokop/git/dftbplus/app/waveplot/slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0)
- Spline code: [/home/prokop/git/dftbplus/pyBall/OCL/Grid.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:0:0-0:0) lines 91-116
- Kernel interpolation: [/home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl:0:0-0:0) lines 55-85

### 5.2 Phase 2: Orbital Indexing and Coefficient Layout

**Objective:** Map DFTB+ eigenvectors to Fireball coefficient format.

**Tasks:**

1. **Compute orbital offsets per atom:**
   - For each atom: count orbitals = sum over shells of (2*l + 1)
   - Compute `i0orb` (cumulative sum)
   - Reference: [molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) lines 163-168 (total orbital count)

2. **Flatten eigenvector layout:**
   - DFTB+: `eigvecsReal(nOrb, nState)` - column-major (Fortran)
   - Fireball: `coeffs[natoms * numorb_max]` - row-major
   - Need transpose and padding to `numorb_max`

3. **Handle angular momentum ordering:**
   - DFTB+: m = -l, ..., 0, ..., +l (real spherical harmonics)
   - Fireball: `[px, py, pz, s]` order for sp
   - Need permutation mapping
   - Reference: [slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0) lines 101-163 (realTessY ordering)

**Key file locations:**
- Orbital counting: [/home/prokop/git/dftbplus/app/waveplot/molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) lines 163-168
- Angular functions: [/home/prokop/git/dftbplus/app/waveplot/slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0) lines 101-163
- Coefficient packing: [/home/prokop/git/dftbplus/pyBall/OCL/Grid.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:0:0-0:0) lines 606-616

### 5.3 Phase 3: Task Building Adaptation

**Objective:** Adapt Fireball's task building for DFTB+ orbital cutoffs.

**Tasks:**

1. **Modify `AtomData` structure:**
   - Current: Single cutoff per atom
   - DFTB+: Different cutoffs per orbital
   - Solution: Use maximum cutoff per atom for task building (conservative)
   - Reference: [molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) line 178 (max cutoff)

2. **Keep task building kernels:**
   - `count_atoms_per_block`, `fill_task_atoms`, `compact_tasks` are generic
   - No changes needed if using max cutoff

3. **Validate task correctness:**
   - Ensure all orbitals with contributions are included
   - Check that max cutoff doesn't miss any contributions

**Key file locations:**
- Task building: [/home/prokop/git/dftbplus/pyBall/OCL/Grid.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:0:0-0:0) lines 198-321
- Kernels: [/home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl:0:0-0:0) lines 255-381
- Cutoff computation: [/home/prokop/git/dftbplus/app/waveplot/molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) line 178

### 5.4 Phase 4: Kernel Adaptation for Orbital Projection

**Objective:** Use [project_orbital()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1280:4-1403:18) kernel instead of density projection.

**Tasks:**

1. **Adapt [project_orbital()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1280:4-1403:18) kernel:**
   - Current: Projects single MO with coefficients
   - Need: Support batch projection (multiple MOs simultaneously)
   - Add: Periodic cell support (phase factors)
   - Reference: [Grid.cl](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl:0:0-0:0) lines 597-679 (current kernel)
   - Reference: [molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) lines 408 (phase factors)

2. **Add periodic cell handling:**
   - Input: `cellVec(3, nCell)`, `latVecs(3,3)`, `kPoints(3,nKPoint)`
   - Compute phase: `exp(i * k · R_cell)` for each cell
   - Sum contributions from all cells
   - Reference: [molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) lines 171-201 (cell setup)
   - Reference: [molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) lines 404-408 (phase computation)

3. **Batch processing:**
   - Modify kernel to process `nState` eigenvectors
   - Output: `gridCache(nx, ny, nz, nState)`
   - Matches DFTB+ caching strategy

**Key file locations:**
- Orbital kernel: [/home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl](cci:7://file:///home/prokop/git/dftbplus/pyBall/OCL/cl/Grid.cl:0:0-0:0) lines 597-679
- Periodic handling: [/home/prokop/git/dftbplus/app/waveplot/molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) lines 171-201, 404-408

### 5.5 Phase 5: Python Host Integration

**Objective:** Create Python wrapper that reads DFTB+ files and calls OpenCL.

**Tasks:**

1. **Create DFTB+ file parsers:**
   - Parse `detailed.xml` for geometry and occupations
   - Parse `eigenvec.bin` for eigenvectors
   - Parse basis HSD for STO parameters
   - Reference: [initwaveplot.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/initwaveplot.F90:0:0-0:0) lines 390-460 (detailed.xml)
   - Reference: [initwaveplot.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/initwaveplot.F90:0:0-0:0) lines 746-858 (basis)
   - Reference: [gridcache.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/gridcache.F90:0:0-0:0) lines 333-353 (eigenvec.bin)

2. **Create main projection class:**
   - Inherits from [GridProjector](cci:2://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:9:0-1403:18)
   - Overrides [load_basis()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:40:4-164:27) to use STO parameters
   - Overrides [project()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:419:4-582:18) to use orbital projection kernel
   - Adds periodic cell support

3. **Implement cube file output:**
   - Convert grid to cube format
   - Match DFTB+ cube file format
   - Reference: [waveplot.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/waveplot.F90:0:0-0:0) lines 408-514 (writeCubeFile)

**Key file locations:**
- XML parsing: [/home/prokop/git/dftbplus/app/waveplot/initwaveplot.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/initwaveplot.F90:0:0-0:0)
- Binary eigenvectors: [/home/prokop/git/dftbplus/app/waveplot/gridcache.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/gridcache.F90:0:0-0:0)
- Cube output: [/home/prokop/git/dftbplus/app/waveplot/waveplot.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/waveplot.F90:0:0-0:0)

### 5.6 Phase 6: Validation and Testing

**Objective:** Ensure numerical agreement with CPU waveplot.

**Tasks:**

1. **Unit tests:**
   - STO evaluation: Compare CPU vs GPU at random points
   - Angular functions: Compare `realTessY` vs kernel
   - Single atom: Compare single orbital projection

2. **Integration tests:**
   - Small molecule (H2O): Compare full MO grids
   - Check charge density integration
   - Verify cube file output matches

3. **Performance benchmarks:**
   - Compare CPU vs GPU timing
   - Scale with system size
   - Profile kernel execution

---

## 6. Specific Implementation Notes

### 6.1 STO Grid Precomputation

**Algorithm:**
```python
def precompute_sto_grid(aa, alpha, ll, cutoff, resolution):
    """
    Precompute STO values on uniform grid with second derivatives.
    
    Args:
        aa: (nPow, nAlpha) summation coefficients
        alpha: (nAlpha,) exponential coefficients
        ll: angular momentum
        cutoff: cutoff radius
        resolution: grid spacing
    
    Returns:
        grid_values: (nNodes,) STO values
        grid_d2: (nNodes,) second derivatives for cubic spline
    """
    nNodes = int(np.ceil(cutoff / resolution)) + 1
    r = np.arange(nNodes) * resolution
    
    # Evaluate STO at each grid point
    grid_values = np.zeros(nNodes)
    for i, rr in enumerate(r):
        grid_values[i] = evaluate_sto_explicit(aa, alpha, ll, rr)
    
    # Compute second derivatives (natural cubic spline)
    grid_d2 = compute_spline_d2(grid_values, resolution)
    
    return grid_values, grid_d2
```

**Reference implementation:** [slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0) lines 211-219

### 6.2 Orbital Index Mapping

**DFTB+ to Fireball coefficient order:**

For sp basis (l=0,1):
- DFTB+: s (m=0), then p (m=-1,0,1) → corresponds to py, pz, px
- Fireball: px, py, pz, s

Permutation array:
```python
# DFTB+ index → Fireball index
# DFTB: [s, py, pz, px] for l=0,1
# Fireball: [px, py, pz, s]
perm = np.array([3, 1, 2, 0])  # [s, py, pz, px] → [px, py, pz, s]
```

**Reference:** [slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0) lines 104-115 (p-orbital mapping)

### 6.3 Periodic Phase Factor Computation

**Formula:**
```fortran
phases(iCell, iKPoint) = exp(imag * matmul(transpose(cellVec), kPoints))
```

**OpenCL implementation:**
```c
float2 phase;
phase.x = cos(dot(cellVec, kPoint));
phase.y = sin(dot(cellVec, kPoint));
```

**Reference:** [molorb.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/molorb.F90:0:0-0:0) line 408

### 6.4 Eigenvector Binary Format

**DFTB+ format (Fortran unformatted):**
```fortran
read(fd%unit) identity  ! Integer header
do iSpin = 1, nSpin
  do iKPoint = 1, nKPoint
    do iState = 1, nState
      read(fd%unit) eigvecs(:, iState)  ! nOrb values
    end do
  end do
end do
```

**Python reading:**
```python
import struct
with open('eigenvec.bin', 'rb') as f:
    identity = struct.unpack('i', f.read(4))[0]
    # Each record has 4-byte markers (Fortran convention)
    for iSpin in range(nSpin):
        for iKPoint in range(nKPoint):
            for iState in range(nState):
                # Skip record marker
                f.read(4)
                # Read nOrb doubles
                coeffs = np.fromfile(f, dtype=np.float64, count=nOrb)
                # Skip record marker
                f.read(4)
```

**Reference:** [gridcache.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/gridcache.F90:0:0-0:0) lines 335-353

## 7. Potential Challenges and Solutions

### 7.1 Challenge: Different Cutoffs per Orbital

**Issue:** DFTB+ has per-orbital cutoffs, Fireball assumes per-atom cutoff.

**Solution:**
1. Use maximum cutoff per atom for task building (conservative)
2. In kernel, check per-orbital cutoff before evaluation
3. Add orbital cutoff array to `AtomData` or separate buffer

### 7.2 Challenge: Periodic Boundary Conditions

**Issue:** Fireball code doesn't handle periodic cells.

**Solution:**
1. Add cell translation vectors to kernel input
2. Add phase factors for k-points
3. Loop over cells in kernel (or unroll for small nCell)
4. Sum contributions with appropriate phase

### 7.3 Challenge: Complex Eigenvectors

**Issue:** Fireball assumes real coefficients.

**Solution:**
1. Extend kernel to handle complex coefficients
2. Use float2 for complex numbers in OpenCL
3. Output complex grid values
4. Separate real/imag outputs for cube files

### 7.4 Challenge: Memory Bandwidth

**Issue:** Reading basis functions for every voxel/atom pair.

**Solution:**
1. Use tiled approach (already in Fireball)
2. Cache basis functions in local memory
3. Consider texture memory for radial interpolation

## 8. File Structure for New Implementation

**Proposed structure:**
```
pyBall/OCL/
  Waveplot.py          # Main class for DFTB+ waveplot
  WaveplotBasis.py     # STO grid precomputation
  cl/
    Grid.cl            # Existing kernels (keep)
    Waveplot.cl        # New kernels for orbital projection
    WaveplotPeriodic.cl # Periodic cell support
```

**Integration points:**
- Reuse `OpenCLBase` class
- Reuse task building infrastructure
- Add new kernel compilation
- Add DFTB+ file parsers

## 9. Summary of Key Adaptation Points

| Component | DFTB+ Location | Fireball Location | Adaptation Needed |
|-----------|---------------|------------------|-------------------|
| Radial functions | `slater.F90:getValue` | `Grid.cl:evaluate_radial` | Replace grid source |
| Angular functions | `slater.F90:realTessY` | `Grid.cl:prefactors` | Already compatible |
| Orbital indexing | `molorb.F90:localGetValue` | `Grid.py:load_basis` | Compute i0orb, norb |
| Coefficients | `gridcache.F90:eigenvecReal` | `Grid.py:coeffs` | Transpose + permute |
| Task building | N/A (CPU) | `Grid.cl:count_atoms_per_block` | Use max cutoff |
| Projection | `molorb.F90:localGetValue` | `Grid.cl:project_orbital` | Add periodic support |
| Caching | [gridcache.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/gridcache.F90:0:0-0:0) | N/A | Implement in Python |
| Output | `waveplot.F90:writeCubeFile` | N/A | Implement in Python |

I have completed a comprehensive code review and analysis for reimplementing waveplot in OpenCL. The analysis includes:

## Key Findings

**Architecture Compatibility:** The Fireball OpenCL Grid code can be adapted for DFTB+ waveplot with targeted modifications. The core tiled projection algorithm and task-building infrastructure are reusable.

**Critical Adaptation Points:**

1. **Basis Functions:** Replace Fireball's numerical radial functions with DFTB+ analytical Slater-type orbitals (STO). The `evaluate_radial()` kernel can be reused by changing the data source from `.wf` files to precomputed STO grids.

2. **Orbital Projection:** Use Fireball's [project_orbital()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1280:4-1403:18) kernel (simpler than density projection) to match DFTB+'s direct MO evaluation approach.

3. **Periodic Support:** Add periodic cell handling and k-point phase factors to the kernel (currently missing in Fireball).

4. **Data Layouts:** The orbital indexing and coefficient layouts are compatible after proper permutation (DFTB+ uses [s, py, pz, px] ordering, Fireball uses [px, py, pz, s]).

## Implementation Roadmap

The analysis provides a 6-phase plan:
- **Phase 1:** STO grid precomputation (adapt [slater.F90](cci:7://file:///home/prokop/git/dftbplus/app/waveplot/slater.F90:0:0-0:0) logic)
- **Phase 2:** Orbital indexing and coefficient layout mapping
- **Phase 3:** Task building adaptation (use max orbital cutoff)
- **Phase 4:** Kernel adaptation for orbital projection with periodic support
- **Phase 5:** Python host integration (parse DFTB+ files, cube output)
- **Phase 6:** Validation and testing

## Documented References

All analysis includes specific file locations and line numbers:
- STO evaluation: `slater.F90:211-219, 254-300`
- Orbital counting: `molorb.F90:163-168`
- Angular functions: `slater.F90:101-163`
- Task building: `Grid.py:198-321`, `Grid.cl:255-381`
- Orbital kernel: `Grid.cl:597-679`
- Periodic handling: `molorb.F90:171-201, 404-408`
- Cube output: `waveplot.F90:408-514`

The detailed notes above provide the technical foundation for creating a well-defined implementation plan.


---

# USER

OK, now lets formulate detailed plan how to implement it and make test on system of PTCDA or H2O 
@H2O.xyz @PTCDA.xyz 

as a reference you can take 
@test_grid_projection_Fireball.py @test_stm_orbital_projection_Fireball.py which is for Fireball program not DFTB, but some conspet should be similar.

again do not code yet, just review the code, make notes, and formulate the plan

---

# SWE-1.6

# Detailed Implementation Plan: DFTB+ Waveplot OpenCL

## 1. Test System Analysis

### 1.1 H2O (Cluster Test)
**File:** [/home/prokop/git/dftbplus/data/xyz/H2O.xyz](cci:7://file:///home/prokop/git/dftbplus/data/xyz/H2O.xyz:0:0-0:0)
- 3 atoms: O (Z=8), 2×H (Z=1)
- Non-periodic (cluster)
- Simple geometry: bent molecule
- **Use case:** Initial validation, debugging, single MO projection

### 1.2 PTCDA (Periodic Test)
**File:** [/home/prokop/git/dftbplus/data/xyz/PTCDA.xyz](cci:7://file:///home/prokop/git/dftbplus/data/xyz/PTCDA.xyz:0:0-0:0)
- 38 atoms: 22×C (Z=6), 6×O (Z=8), 10×H (Z=1)
- Periodic with lattice vectors: 20×20×20 Å cubic cell
- Large conjugated organic molecule
- **Use case:** Periodic BC testing, k-point handling, performance scaling

## 2. Fireball Test Pattern Analysis

### 2.1 [test_grid_projection_Fireball.py](cci:7://file:///home/prokop/git/dftbplus/tests/grid/test_grid_projection_Fireball.py:0:0-0:0) Pattern

**Data flow (lines 51-170):**
```
1. Load XYZ file → atomTypes, atomPos
2. Initialize FireCore → fc.initialize(atomType, atomPos)
3. Run SCF → fc.SCF(atomPos, nmax_scf=1)
4. Get sparse data → fc.get_HS_neighs(), fc.get_rho_sparse()
5. Setup GridProjector → load_basis(species_nz) from Fdata
6. Define grid → origin, dA/dB/dC, ngrid
7. Project → projector.project(rho, neighs, atoms_dict, grid_spec)
8. Visualize → plot_density_slices(), plot_density_maxproj()
```

**Key functions:**
- `fc.initialize()` - Sets up Fireball calculation
- `fc.get_HS_neighs()` - Gets neighbor list
- `fc.get_rho_sparse()` - Gets density matrix
- [projector.load_basis()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:40:4-164:27) - Loads .wf files
- [projector.project()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:419:4-582:18) - Main GPU projection

### 2.2 [test_stm_orbital_projection_Fireball.py](cci:7://file:///home/prokop/git/dftbplus/tests/grid/test_stm_orbital_projection_Fireball.py:0:0-0:0) Pattern

**Data flow (lines 137-262):**
```
1. Load XYZ → AtomicSystem
2. Initialize FireCore → fc.initialize(atomType, atomPos)
3. Run SCF → fc.evalForce(mol.apos, nmax_scf=200)
4. Get eigenvectors → fc.get_wfcoef(norb)
5. Get orbital layout → _get_orbital_mapping_from_fireball()
6. Build grid → _build_xy_grid() for 2D plane
7. Project orbital → project_orbital_to_points()
8. Compare → Fortran vs OpenCL
```

**Key functions:**
- `fc.get_wfcoef()` - Gets MO coefficients
- `project_orbital_to_points()` - Projects single MO to points
- `fc.orb2points()` - CPU reference (Fortran)

## 3. DFTB+ Adaptation Strategy

### 3.1 File Parsing Layer (Replace FireCore)

**Required parsers:**

1. **`detailed.xml` parser** (reference: `initwaveplot.F90:390-460`)
   - Geometry: atom positions, species
   - Occupations: per state, k-point, spin
   - K-points: coordinates and weights
   - Real vs complex flag
   - Identity number
   - **Implementation:** Use `xml.etree.ElementTree` or existing DFTB+ Python API

2. **`eigenvec.bin` parser** (reference: `gridcache.F90:333-353`)
   - Binary Fortran unformatted format
   - Structure: identity header, then eigenvectors in nested loops (spin, k-point, state)
   - Each record has 4-byte Fortran markers
   - **Implementation:** Use `struct.unpack` to read binary data

3. **Basis HSD parser** (reference: `initwaveplot.F90:746-858`)
   - Species definitions: atomic number
   - Per species: orbitals with angular momentum, occupation, cutoff
   - STO parameters: exponents, coefficients
   - Resolution for radial grid
   - **Implementation:** Use DFTB+ HSD parser (`dftbp_io_hsdparser`)

**New module structure:**
```
pyBall/OCL/
  DFTBplusParser.py       # Main parser class
  DFTBplusParser/
    __init__.py
    xml_parser.py         # detailed.xml
    eigenvec_parser.py    # eigenvec.bin
    basis_parser.py       # HSD basis
```

### 3.2 STO Basis Precomputation (Replace .wf loading)

**Function:** `precompute_sto_grid()` (new function)

**Algorithm (from `slater.F90:211-219, 254-300`):**
```python
def precompute_sto_grid(aa, alpha, ll, cutoff, resolution):
    """
    Precompute STO on uniform grid with spline derivatives.
    
    Input:
        aa: (nPow, nAlpha) coefficients
        alpha: (nAlpha,) exponents
        ll: angular momentum
        cutoff: cutoff radius
        resolution: grid spacing
    
    Output:
        grid_values: (nNodes,) STO values
        grid_d2: (nNodes,) second derivatives
    """
    nNodes = int(np.ceil(cutoff / resolution)) + 2
    r = np.arange(nNodes) * resolution
    
    # Evaluate STO analytically at each grid point
    grid_values = np.zeros(nNodes)
    for i, rr in enumerate(r):
        grid_values[i] = evaluate_sto_explicit(aa, -alpha, ll, rr)
    
    # Compute second derivatives (natural cubic spline)
    grid_d2 = compute_spline_d2(grid_values, resolution)
    
    return grid_values, grid_d2
```

**Integration point:** Replace [GridProjector.load_basis()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:40:4-164:27) lines 41-165

### 3.3 Orbital Index Mapping

**Function:** `compute_orbital_layout()` (new function)

**Algorithm (from `molorb.F90:163-168`):**
```python
def compute_orbital_layout(species_basis, species_array):
    """
    Compute orbital offsets per atom for DFTB+.
    
    Input:
        species_basis: list of TSpeciesBasis per species
        species_array: species index per atom
    
    Output:
        norb_per_atom: (nAtom,) orbitals per atom
        orb_offsets: (nAtom+1,) cumulative offsets
        total_norb: total number of orbitals
    """
    natoms = len(species_array)
    norb_per_atom = np.zeros(natoms, dtype=np.int32)
    
    for iatom in range(natoms):
        ispec = species_array[iatom]
        basis = species_basis[ispec]
        # Count: sum over shells of (2*l + 1)
        norb = 0
        for l in basis.angMoms:
            norb += 2*l + 1
        norb_per_atom[iatom] = norb
    
    orb_offsets = np.zeros(natoms + 1, dtype=np.int32)
    orb_offsets[1:] = np.cumsum(norb_per_atom)
    total_norb = orb_offsets[-1]
    
    return norb_per_atom, orb_offsets, total_norb
```

**Angular momentum ordering mapping:**
- DFTB+: m = -l, ..., 0, ..., +l (real spherical harmonics)
- For p (l=1): py, pz, px
- Fireball: px, py, pz, s
- Permutation needed for compatibility

### 3.4 GridProjector Adaptation

**New class:** `DFTBplusGridProjector` (inherits from [GridProjector](cci:2://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:9:0-1403:18))

**Modifications:**

1. **Override [load_basis()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:40:4-164:27):**
   - Input: STO parameters instead of .wf files
   - Call `precompute_sto_grid()` for each orbital
   - Pack in same format: `[n_species, max_shells, n_nodes, 2]`

2. **Override [project()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:419:4-582:18) for orbital projection:**
   - Input: eigenvectors instead of density matrix
   - Use [project_orbital](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1280:4-1403:18) kernel (not density kernel)
   - Support batch projection (multiple MOs)

3. **Add periodic cell support:**
   - New method: `project_orbital_periodic()`
   - Input: cell vectors, k-points
   - Compute phase factors in kernel
   - Sum over periodic images

4. **Add cube file output:**
   - New method: `write_cube_file()`
   - Format matches DFTB+ waveplot output

---

## 4. Test Implementation Plan

### 4.1 Phase 1: File Parsing Tests

**Test file:** `tests/grid/test_dftb_parsers.py`

**Test 1: XML parser**
```python
def test_parse_detailed_xml_h2o():
    """Test parsing detailed.xml for H2O system."""
    # Setup: Run DFTB+ for H2O to generate detailed.xml
    # Test: Parse geometry, occupations, k-points
    # Validate: Compare with expected values
```

**Test 2: Eigenvec parser**
```python
def test_parse_eigenvec_bin_h2o():
    """Test parsing eigenvec.bin for H2O system."""
    # Setup: Run DFTB+ for H2O to generate eigenvec.bin
    # Test: Read eigenvectors
    # Validate: Check dimensions, identity
```

**Test 3: Basis parser**
```python
def test_parse_basis_hsd_h2o():
    """Test parsing basis HSD for H2O system."""
    # Setup: Create HSD basis for H and O
    # Test: Parse STO parameters
    # Validate: Check coefficients, exponents
```

### 4.2 Phase 2: STO Precomputation Tests

**Test file:** `tests/grid/test_sto_precompute.py`

**Test 1: STO evaluation**
```python
def test_sto_evaluation():
    """Compare STO evaluation with Fortran reference."""
    # Test: Evaluate STO at random points
    # Compare: Python vs Fortran (via waveplot module)
    # Tolerance: 1e-10
```

**Test 2: Grid interpolation**
```python
def test_sto_grid_interpolation():
    """Test grid-based interpolation accuracy."""
    # Test: Precompute grid, interpolate at random points
    # Compare: Interpolated vs direct evaluation
    # Tolerance: 1e-8 (interpolation error)
```

**Test 3: Spline derivatives**
```python
def test_spline_derivatives():
    """Test cubic spline second derivatives."""
    # Test: Compute d2, verify smooth interpolation
    # Compare: With Fortran spline implementation
```

### 4.3 Phase 3: H2O Cluster Test

**Test file:** `tests/grid/test_waveplot_h2o.py`

**Setup:**
```python
def setup_h2o_dftb():
    """Run DFTB+ for H2O and parse outputs."""
    # 1. Create waveplot_in.hsd for H2O
    # 2. Run DFTB+ to generate:
    #    - detailed.xml
    #    - eigenvec.bin
    # 3. Parse all files
    return geometry, eigenvectors, basis, occupations
```

**Test 1: Single MO projection**
```python
def test_h2o_single_mo_projection():
    """Project single MO to grid, compare CPU vs GPU."""
    # Setup: Load H2O data
    # CPU: Run waveplot.F90 for HOMO
    # GPU: Project HOMO with OpenCL
    # Compare: Grid values, RMS error
    # Tolerance: 1e-6
```

**Test 2: Multiple MOs**
```python
def test_h2o_multiple_mo_projection():
    """Project multiple MOs in batch."""
    # Test: Project HOMO-2 to LUMO+2
    # Compare: Each MO individually
    # Validate: Charge density integration
```

**Test 3: Cube file output**
```python
def test_h2o_cube_output():
    """Generate cube file, compare with reference."""
    # Test: Write cube from GPU grid
    # Compare: With waveplot.F90 cube file
    # Validate: Format, values
```

### 4.4 Phase 4: PTCDA Periodic Test

**Test file:** `tests/grid/test_waveplot_ptcda.py`

**Setup:**
```python
def setup_ptcda_dftb():
    """Run DFTB+ for PTCDA and parse outputs."""
    # 1. Create waveplot_in.hsd for PTCDA (periodic)
    # 2. Run DFTB+ with k-points
    # 3. Parse detailed.xml (with lattice vectors)
    # 4. Parse eigenvec.bin (complex eigenvectors)
    return geometry, eigenvectors, basis, kpoints
```

**Test 1: Periodic cell handling**
```python
def test_ptcda_periodic_cells():
    """Test periodic image generation."""
    # Setup: Load PTCDA with lattice vectors
    # Test: Generate cell translations
    # Validate: Number of cells, translation vectors
```

**Test 2: k-point phase factors**
```python
def test_ptcda_kpoint_phases():
    """Test k-point phase factor computation."""
    # Test: Compute phases for each cell/k-point
    # Compare: With Fortran implementation
    # Validate: exp(i*k*R) formula
```

**Test 3: Complex MO projection**
```python
def test_ptcda_complex_mo_projection():
    """Project complex MO with k-points."""
    # Setup: Load PTCDA at Gamma and other k-points
    # CPU: Run waveplot.F90 for specific k-point
    # GPU: Project with OpenCL phase factors
    # Compare: Real and imaginary parts
```

**Test 4: Performance scaling**
```python
def test_ptcda_performance():
    """Benchmark CPU vs GPU performance."""
    # Test: Time CPU waveplot vs GPU projection
    # Report: Speedup factor
    # Validate: Memory usage
```

## 5. Data Structure Mapping

### 5.1 Input Data Mapping

| DFTB+ Source | Python Object | Fireball Equivalent |
|--------------|--------------|-------------------|
| `detailed.xml` geometry | `dict{'coords': (n,3), 'species': (n,)}` | `atomPos`, `atomTypes` |
| `detailed.xml` occupations | `np.array(nState, nKPoint, nSpin)` | N/A (not used) |
| `eigenvec.bin` | `np.array(nOrb, nState)` (complex) | `C_fc` from `fc.get_wfcoef()` |
| Basis HSD STO params | `dict{'aa': (nPow,nAlpha), 'alpha': (nAlpha,)}` | `.wf` files |
| Basis cutoffs | `np.array(nOrb)` | `Rcut` per atom |

### 5.2 Grid Specification Mapping

| Parameter | DFTB+ waveplot | OpenCL GridSpec |
|-----------|---------------|----------------|
| Origin | `wp%opt%origin(3)` | `grid_spec['origin']` |
| Grid vectors | `wp%loc%gridVec(3,3)` | `grid_spec['dA/dB/dC']` |
| Dimensions | `wp%opt%nPoints(3)` | `grid_spec['ngrid']` |
| Box vectors | `wp%opt%boxVecs(3,3)` | Derived from origin + ngrid*d |

### 5.3 Orbital Layout Mapping

| DFTB+ | OpenCL | Notes |
|-------|--------|-------|
| `iStos(nSpecies+1)` | `orb_offsets(nAtom+1)` | Species → atom mapping |
| `stos(nOrb)` | Packed basis array | Same layout |
| `cutoffs(nOrb)` | Per-orbital in kernel | Need adaptation |
| `angMoms(nOrb)` | Implied in basis order | Use permutation |

## 6. Implementation Order

### 6.1 Week 1: File Parsing Infrastructure

1. **Day 1-2:** XML parser for `detailed.xml`
   - Parse geometry
   - Parse occupations
   - Parse k-points
   - Test with H2O

2. **Day 3:** Eigenvec binary parser
   - Read Fortran unformatted format
   - Handle record markers
   - Test with H2O

3. **Day 4:** Basis HSD parser
   - Parse STO parameters
   - Parse cutoffs
   - Test with H/O basis

4. **Day 5:** Integration test
   - Load all files for H2O
   - Verify data consistency

### 6.2 Week 2: STO Precomputation

1. **Day 1-2:** STO evaluation function
   - Implement analytical formula
   - Compare with Fortran reference

2. **Day 3:** Grid precomputation
   - Precompute on uniform grid
   - Compute spline derivatives

3. **Day 4:** Integrate into GridProjector
   - Override [load_basis()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:40:4-164:27)
   - Test with H/O STO parameters

4. **Day 5:** Validation
   - Compare radial interpolation
   - Test at random points

### 6.3 Week 3: Orbital Projection (Cluster)

1. **Day 1-2:** Orbital layout computation
   - Compute offsets per atom
   - Handle angular momentum ordering

2. **Day 3:** Adapt projection kernel
   - Use [project_orbital](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1280:4-1403:18) kernel
   - Handle real eigenvectors

3. **Day 4:** H2O single MO test
   - Project HOMO
   - Compare with CPU waveplot

4. **Day 5:** H2O full test
   - Multiple MOs
   - Cube file output
   - Validation

### 6.4 Week 4: Periodic Support

1. **Day 1-2:** Cell translation setup
   - Generate cell vectors
   - Compute neighbor cells

2. **Day 3:** Phase factor computation
   - Implement k-point phases
   - Test with PTCDA

3. **Day 4:** Complex kernel adaptation
   - Handle complex eigenvectors
   - Add periodic loop in kernel

4. **Day 5:** PTCDA test
   - Project at Gamma
   - Project at other k-points
   - Validate with CPU

### 6.5 Week 5: Performance and Validation

1. **Day 1-2:** Performance benchmarking
   - CPU vs GPU timing
   - Memory profiling

2. **Day 3:** Larger systems
   - Test with bigger molecules
   - Scale studies

3. **Day 4:** Code cleanup
   - Documentation
   - Error handling

4. **Day 5:** Final validation
   - Regression tests
   - Documentation

## 7. Validation Strategy

### 7.1 Numerical Validation

**Tier 1: Unit tests**
- STO evaluation: 1e-10 tolerance
- Grid interpolation: 1e-8 tolerance
- Angular functions: 1e-12 tolerance

**Tier 2: Integration tests**
- Single atom projection: 1e-8 tolerance

## 8. Current Status (May 2026)

### 8.1 What Was Achieved

**OpenCL DFTB+ Waveplot Implementation:**
- Reused existing `GridProjector` from Fireball OpenCL implementation
- Added `load_basis_sto()` method to `GridProjector` for Slater-type orbital evaluation
- Successfully projected all 6 H2O molecular orbitals using real DFTB+ eigenvectors
- Generated chemically interpretable MO visualizations (XY, XZ, YZ slices)
- Fixed initial "kernel zero output" issue by using correct coefficient ordering

**Key Technical Decisions:**
- Minimal changes to existing `Grid.cl` kernel (no kernel rewrite)
- Coefficient reordering: DFTB+ `[s, py, pz, px]` → kernel `[px, py, pz, s]`
- Per-atom padding to 4 orbitals (H p-orbitals zeroed)
- Automatic best-slice selection for visualization (handles pz-type MOs)

### 8.2 Problems Encountered and Solutions

**Problem 1: Kernel Zero Output (H2O)**
- **Cause:** `DFTBplusGridProjector` loaded custom `DFTBplusGrid.cl` which lacked task-building kernels (`count_atoms_per_block`, `fill_task_atoms`, `compact_tasks`). The fallback code silently zeroed output.
- **Fix:** Switched to plain `GridProjector` with `load_basis_sto()` and `project_orbital()` from existing `Grid.cl`.

**Problem 2: GPU OUT_OF_RESOURCES (PTCDA)**
- **Cause:** Grid size too large for GPU workgroup limits.
- **Fix:** Reduced grid size, increased `nMaxAtom` parameter.

**Problem 3: Unit Mismatch (Bohr vs Angstrom)**
- **Cause:** WAVEPLOT's `waveplot_in.hsd` specifies STO parameters (exponents, cutoffs, resolution) in **Bohr** (DFTB+ internal units), but OpenCL kernel works in **Angstrom**. Initial implementation treated Bohr values as Angstrom, causing wrong radial decay rates.
- **Details:**
  - `alpha_Ang = alpha_Bohr / BOHR2ANG` (exponent must be divided by conversion factor)
  - `cutoff_Ang = cutoff_Bohr * BOHR2ANG`
  - `coeff_Ang = coeff_Bohr / BOHR2ANG^l` (because `r^l` carries units)
- **Fix:** Created `parse_basis_hsd_ang()` in `DFTBplusParser.py` that reads HSD and performs all Bohr→Ang conversions automatically. Replaced all hardcoded `species_list_sto` blocks with calls to this function.

**Problem 4: Grid Step Mismatch Causing Apparent "Wrong Shape"**
- **Cause:** Test script had `--step` default=0.2 Å, but grid origin `[-4.725,-4.125,-3.525]` and `ngrid=[64,56,48]` were hardcoded for step=0.15 Å (matching WAVEPLOT cube). With step=0.2, the atom at (0,0,0) fell at `z=0 → iz=17.6` (between grid points), causing interpolation artifacts and the orbital max appearing at wrong slice (iz=18 instead of iz=23).
- **Fix:** Changed `--step` default to 0.15 Å to match the hardcoded grid parameters.
- **Note:** The kernel itself is independent of grid step — the issue was inconsistent grid specification, not kernel fragility.

**Problem 5: Duplicate Hardcoded STO Parameters**
- **Cause:** Two separate blocks in `test_waveplot_dftb.py` hardcoded `species_list_sto` (one for H2O, one for PTCDA). This was error-prone and violated the principle of reading from authoritative sources.
- **Fix:** Removed both hardcoded blocks and replaced with calls to `parse_basis_hsd_ang()`. Also removed hardcoded `rcut` values and derived them from parsed cutoffs.

**Problem 6: WAVEPLOT Validation Mismatch (Initial)**
- **Cause:** OpenCL implementation uses **analytic single-zeta STOs** while WAVEPLOT uses **spline-tabulated radial functions** from SK files (mio-1-1). mio-1-1 SK files contain spline data, not simple exponents.
- **Resolution:** After fixing unit conversions, the `waveplot_in.hsd` STO parameters (single-zeta Slater orbitals) actually **do match** what WAVEPLOT uses for waveplot visualization. The SK files are only used for the DFTB+ Hamiltonian, not for waveplot. WAVEPLOT reads its STO parameters from `waveplot_in.hsd`, not from `.skf` files.
- **Result:** After all fixes, OpenCL matches WAVEPLOT cube output to machine precision (RMS ~1e-8, relative ~1e-7).

### 8.3 3-Way Parity: WAVEPLOT / libwaveplot / OpenCL

**Three independent implementations now agree:**

| Implementation | Source | Basis | Grid | Output |
|---------------|--------|-------|------|--------|
| **WAVEPLOT** (Fortran) | `app/waveplot/` | STO from `waveplot_in.hsd` | Defined in HSD | Gaussian `.cube` files |
| **libwaveplot** (C) | `app/waveplot/libwaveplot.F90` | Same STO via `slater.F90` | Same as WAVEPLOT | Python `ctypes` array |
| **OpenCL** (Python) | `pyBall/OCL/Grid.cl` | STO from `parse_basis_hsd_ang()` | Python `GridProjector` | NumPy `.npy` files |

**Validation results (H2O, 6 MOs):**
- **libwaveplot vs WAVEPLOT cube:** RMS ~1e-8, relative ~1e-7 (machine precision)
- **OpenCL vs WAVEPLOT cube:** RMS ~1e-8, relative ~1e-7 (after unit fixes)
- **H2 test (H2 molecule):** Ratio OCL/WAVEPLOT = 1.0000 at all points, perfect parity

### 8.4 Test Scripts and Modules Created

**New/Modified Modules:**

1. **`pyBall/OCL/DFTBplusParser.py`**
   - Added `parse_basis_hsd_ang(hsd_path)` function
   - Reads `waveplot_in.hsd` Basis block
   - Converts Bohr→Ang: `alpha /= B`, `cutoff *= B`, `coeff /= B^l`
   - Returns `species_list` compatible with `GridProjector.load_basis_sto()`
   - Added `BOHR2ANG = 0.5291772109` constant

2. **`pyBall/OCL/Grid.py`**
   - Modified `load_basis_sto()` to accept `species_list` with pre-converted units
   - Removed hardcoded unit conversions (now handled at parse time)

3. **`tests/grid/test_waveplot_dftb.py`**
   - Removed both hardcoded `species_list_sto` blocks
   - Replaced with `parse_basis_hsd_ang()` calls
   - Changed `--step` default from 0.2 to 0.15 Å
   - Fixed `rcut` to derive from parsed cutoffs
   - Now reads all parameters from `waveplot_in.hsd` (no hardcoding)

**New Test Scripts:**

4. **`tests/grid/test_h2_compare.py`**
   - H2 molecule test (2 atoms, closed-shell)
   - Compares WAVEPLOT cube vs OpenCL
   - 1D cut along bond axis (linear + log scale)
   - 2D XZ plane slice
   - Parity plots (all voxels)
   - Ratio OCL/WAVEPLOT along z
   - Confirms perfect agreement (ratio=1.0)

5. **`tests/grid/test_sto_radial.py`**
   - Single H atom diagnostic
   - Compares analytic STO formula vs OpenCL kernel
   - 1D cut along x-axis
   - Log-scale plot to verify exponent slope
   - Confirms radial decay rate is correct

6. **`tests/grid/compare_waveplot_lib.py`**
   - Compares all three: WAVEPLOT cube, libwaveplot, OpenCL
   - 2D XY slices at molecular plane
   - Parity plots
   - Saves comparison images to `waveplot_output/comparison/`

**DFTB+ Data Created:**

7. **`tests/grid/dftb_h2/`**
   - `geom.xyz` — H2 molecule (bond length 0.742 Å)
   - `dftb_in.hsd` — DFTB+ input with mio-1-1 SK files
   - `waveplot_in.hsd` — Waveplot input with H 1s STO (α=0.967 Bohr⁻¹)
   - `eigenvec.bin` — Binary eigenvectors (2 MOs)
   - `detailed.xml` — Geometry, occupations
   - `wp-1-1-1-real.cube` — WAVEPLOT output for MO1
   - `wp-1-1-2-real.cube` — WAVEPLOT output for MO2

### 8.5 How to Use

**Run H2O Test (3-way parity):**
```bash
cd /home/prokop/git/dftbplus/tests/grid
python test_waveplot_dftb.py --system H2O --no-show --output-dir tests/grid/waveplot_output
python compare_waveplot_lib.py --no-show
# View: tests/grid/waveplot_output/comparison/comparison_all_three.png
```

**Run H2 Test (rigorous 1D/2D comparison):**
```bash
cd /home/prokop/git/dftbplus/tests/grid
python test_h2_compare.py
# View: tests/grid/waveplot_output/h2/h2_wp_vs_ocl.png
```

**Run Single H Atom Diagnostic (radial decay verification):**
```bash
cd /home/prokop/git/dftbplus/tests/grid
python test_sto_radial.py
# View: tests/grid/waveplot_output/sto_radial_test.png
```

**Run PTCDA Test (larger system):**
```bash
cd /home/prokop/git/dftbplus/tests/grid
python test_waveplot_dftb.py --system PTCDA --z-height 2.0 --step 0.3
# Requires: tests/grid/dftb_ptcda/ with DFTB+ calculation
```

**DFTB+ Calculation (if not pre-run):**
```bash
cd tests/grid/dftb_h2o
/home/prokop/git_SW/dftbplus/Build/app/dftb+/dftb+
/home/prokop/git_SW/dftbplus/Build/app/waveplot/waveplot
# Generates: eigenvec.bin, detailed.xml, wp-*.cube
```

### 8.6 Key Files Reference

| File | Purpose |
|------|---------|
| `pyBall/OCL/DFTBplusParser.py` | HSD parsing, Bohr→Ang conversion |
| `pyBall/OCL/Grid.py` | `GridProjector`, `load_basis_sto()` |
| `pyBall/OCL/cl/Grid.cl` | OpenCL kernel (project_orbital, evaluate_radial) |
| `tests/grid/test_waveplot_dftb.py` | Main test script for H2O/PTCDA |
| `tests/grid/test_h2_compare.py` | Rigorous H2 1D/2D comparison |
| `tests/grid/test_sto_radial.py` | Single atom radial decay test |
| `tests/grid/compare_waveplot_lib.py` | 3-way parity comparison |
| `tests/grid/dftb_h2o/waveplot_in.hsd` | Authoritative STO parameters (Bohr) |
| `tests/grid/dftb_h2/waveplot_in.hsd` | H2 STO parameters (Bohr) |
| `app/waveplot/libwaveplot.F90` | C library interface to Fortran waveplot |
| `app/waveplot/slater.F90` | Fortran STO implementation (reference) |

### 8.4 Files to Add to Git

**New Modules:**
- `pyBall/OCL/Grid.py` — Added `load_basis_sto()` method (modified existing file)
- `pyBall/OCL/DFTBplusParser.py` — Parser for DFTB+ output files (detailed.xml, eigenvec.bin)
- `pyBall/OCL/DFTBplusGridProjector.py` — DFTB+-specific projector (currently unused, kept for reference)
- `pyBall/OCL/cl/DFTBplusGrid.cl` — DFTB+ specific kernel (currently unused, kept for reference)

**Test Scripts:**
- `tests/grid/test_waveplot_dftb.py` — Main test script for H2O and PTCDA
- `tests/grid/compare_waveplot.py` — WAVEPLOT validation comparison script
- `tests/grid/parse_eigenvec.py` — Utility to inspect eigenvec.bin format

**Test Data:**
- `tests/grid/dftb_h2o/` — DFTB+ calculation directory for H2O
  - `dftb_in.hsd` — DFTB+ input
  - `geom.xyz` — Geometry
  - `detailed.xml` — DFTB+ detailed output
  - `eigenvec.bin` — Binary eigenvectors
  - `waveplot_in.hsd` — WAVEPLOT input for validation
  - `wp-1-1-{1-6}-real.cube` — WAVEPLOT reference cube files

**Output (not to commit):**
- `tests/grid/waveplot_output/` — Generated PNG plots and .npy grids
- Small molecule (H2O): 1e-6 tolerance
- Large molecule (PTCDA): 1e-5 tolerance

**Tier 3: End-to-end tests**
- Cube file format: exact match
- Charge integration: 1e-4 tolerance
- Visualization: qualitative match

### 7.2 Reference Data

**CPU reference generation:**
```bash
# For H2O
cd build/tests/app/waveplot
./waveplot < waveplot_in.hsd

# For PTCDA
cd build/tests/app/waveplot
./waveplot < waveplot_in_ptcda.hsd
```

**Comparison metrics:**
- RMS error of grid values
- Maximum absolute error
- Charge density integration
- Visual comparison of isosurfaces

### 7.3 Performance Metrics

**Baseline:** CPU waveplot timing
**Target:** 10-100× speedup on GPU
**Metrics:**
- Kernel execution time
- Data transfer time
- Total wall time
- Memory bandwidth

## 8. Risk Mitigation

### 8.1 Risk: File Format Incompatibility

**Mitigation:**
- Start with simple test case (H2O)
- Use DFTB+ Python API if available
- Fall back to manual parsing if needed

### 8.2 Risk: STO Evaluation Mismatch

**Mitigation:**
- Direct comparison with Fortran at many points
- Use same grid resolution
- Verify spline implementation

### 8.3 Risk: Periodic Boundary Conditions

**Mitigation:**
- Test with cluster first
- Add periodic support incrementally
- Verify phase factors independently

### 8.4 Risk: Memory Limitations

**Mitigation:**
- Use batch processing (like DFTB+ caching)
- Implement out-of-core if needed
- Profile memory usage early

## 9. Success Criteria

### 9.1 Functional Criteria

- [ ] Parse all DFTB+ input files correctly
- [ ] Precompute STO grids accurately
- [ ] Project single MO for H2O with <1e-6 error
- [ ] Project multiple MOs for H2O
- [ ] Write cube files matching DFTB+ format
- [ ] Handle periodic BC for PTCDA
- [ ] Handle k-points and complex eigenvectors
- [ ] Achieve >10× speedup over CPU

### 9.2 Code Quality Criteria

- [ ] All tests pass
- [ ] Code documented
- [ ] Error handling robust
- [ ] Memory leaks free
- [ ] Compatible with existing Grid.py infrastructure

I have completed a detailed implementation plan for adapting the Fireball OpenCL Grid code to work with DFTB+ waveplot. The plan includes:

## Key Components

**File Parsing Infrastructure:**
- XML parser for `detailed.xml` (geometry, occupations, k-points)
- Binary parser for `eigenvec.bin` (Fortran unformatted eigenvectors)
- HSD parser for basis STO parameters

**STO Basis Adaptation:**
- Precompute STO values on uniform grid with spline derivatives
- Replace Fireball's `.wf` file loading with analytical STO evaluation
- Maintain same packed format for GPU compatibility

**Test Strategy:**
- **H2O (cluster):** 3-atom molecule for initial validation, debugging
- **PTCDA (periodic):** 38-atom molecule with lattice vectors for k-point testing
- Progressive testing: parsers → STO → cluster → periodic → performance

**Implementation Timeline:**
- Week 1: File parsing infrastructure
- Week 2: STO precomputation
- Week 3: Orbital projection (cluster/H2O)
- Week 4: Periodic support (PTCDA)
- Week 5: Performance and validation

**Validation:**
- Direct CPU vs GPU comparison using DFTB+ waveplot as reference
- Tolerance hierarchy: 1e-10 (unit) → 1e-6 (integration) → qualitative (visualization)
- Performance target: >10× speedup over CPU

---

# 2026-05-06: Generalization of compare_waveplot_lib.py

## Problem Statement

The original `compare_waveplot_lib.py` script contained **hardcoded system-specific parameters** for H2O and PTCDA:
- Hardcoded paths (`dftb_h2o/`, `dftb_ptcda/`)
- Hardcoded basis set parameters (exponents, cutoffs, coefficients)
- Hardcoded species arrays and geometry
- `if args.system == 'H2O'` and `else: # PTCDA` conditional logic

This made the script **non-scalable and non-reusable** — adding a new molecule required modifying the source code.

## Solution: Complete Refactoring

### 1. Generalized Architecture

**Key change:** All system data is now **dynamically parsed** from DFTB+ output files. The script works for **any molecule** given only a DFTB+ run directory.

**New CLI interface:**
```bash
# Method 1: 3D grid comparison vs WAVEPLOT cube files
python compare_waveplot_lib.py --dftb-dir tests/grid/dftb_h2o --nmo 6

# Method 2: Explicit point evaluation (orb2points) - more reliable
python compare_waveplot_lib.py --dftb-dir tests/grid/dftb_ptcda --points \
    --plane2d xy --z-offset 2.0 --mo-range 66 75 --npoints 64
```

### 2. Data Flow and Parsers

```
DFTB+ run directory (input: --dftb-dir)
├── detailed.xml          → parse_detailed_xml() → geometry, species, nstates, norb, occupations
├── eigenvec.bin          → parse_eigenvec_bin()  → eigenvectors (nstates, norb)
├── waveplot_in.hsd       → parse_basis_hsd_ang() → STO basis parameters (Å)
├── band.out              → optional energy parsing
└── wp-1-1-*.cube         → reference cubes (Method 1 only)
```

### 3. Unit Conversions (Critical Fix)

**Authoritative source:** `waveplot_in.hsd` (values in **Bohr**)

**Conversion chain:**
```
waveplot_in.hsd (Bohr)
    ↓ parse_basis_hsd_ang()
    → converts to Å for OpenCL (multiplies by BOHR2ANG)
    ↓ build_wp_basis()
    → converts back to Bohr for libwaveplot (divides by BOHR2ANG)
```

**Bug fixed:** Coefficients needed `* BOHR2ANG^l` to convert from Å-normalized back to Bohr:
- For `l=0` (s): no change (×1)
- For `l=1` (p): `coef_ang = coef_bohr / BOHR2ANG` → need `coef_b = coef_ang * BOHR2ANG`
- This was missing, causing 3-4 orders of magnitude error in p-orbital contributions

**Unit reference table:**

| Quantity | In HSD (Bohr) | In OCL (Å) | In libwaveplot (Bohr) |
|----------|---------------|------------|----------------------|
| Exponents | α (Bohr⁻¹) | α / BOHR2ANG (Å⁻¹) | α (Bohr⁻¹) |
| Cutoffs | R (Bohr) | R × BOHR2ANG (Å) | R (Bohr) |
| Coefficients | c (dimensionless) | c / BOHR2ANG^l | c (dimensionless) |
| Resolution | Δ (Bohr) | Δ × BOHR2ANG (Å) | Δ (Bohr) |
| Coordinates | Bohr | Å | Bohr |

### 4. Implementation Details

#### 4.1 Custom Parser: `parse_detailed_xml()`

The DFTB+ XML format is **not** what `DFTBplusParser.parse_detailed_xml()` expects. A custom parser was written:

```xml
<detailedout>
 <geometry>
  <typenames> "O" "H" </typenames>
  <typesandcoordinates>
   1 0.0 0.0 0.0      # type_index (1-based), x, y, z (Bohr)
   2 -1.43 1.20 0.0
   2  1.43 1.20 0.0
  </typesandcoordinates>
  <periodic>No</periodic>
 </geometry>
 <nrofstates>6</nrofstates>
 <nroforbitals>6</nroforbitals>
 <occupations>
  <spin1><k1>2.0 2.0 2.0 2.0 0.0 0.0</k1></spin1>
 </occupations>
</detailedout>
```

Returns:
- `species_names`: list of unique species in order
- `species_per_atom`: 0-based index into species_names
- `coords_bohr`: (natoms, 3) array in Bohr
- `nstates`, `norb`, `nkpoints`, `nspin`
- `occupations`: (nstates, nkpoints, nspin)

#### 4.2 Binary Parser: `parse_eigenvec_bin()`

**Format:** Simple flat binary, **no Fortran record markers**
```
[4-byte identity int] [nstates × norb × 8-byte float64]
```

Returns: `evecs[nstates, norb]` (float64)

#### 4.3 Basis Builder: `build_wp_basis()`

Converts `parse_basis_hsd_ang()` output (Å) → libwaveplot format (Bohr):

```python
for each orbital:
    alpha_b = alpha_ang * BOHR2ANG        # Å⁻¹ → Bohr⁻¹
    cutoff_b = cutoff_ang / BOHR2ANG      # Å → Bohr
    coef_b  = coef_ang * (BOHR2ANG ** l)  # Å-normalized → Bohr (CRITICAL FIX)
```

#### 4.4 Eigenvector Mapping: `evec_to_kernel_coeffs()`

Converts flat eigenvector `[norb]` → kernel format `(natoms, 4)` `[px, py, pz, s]`:

```python
offset = 0
for each atom:
    for each orbital in species:
        l = orbital['l']
        nm = 2*l + 1  # number of m components
        chunk = evec[offset:offset+nm]
        if l == 0:
            c[ia, 3] = chunk[0]  # s → slot 3
        elif l == 1:
            c[ia, 1] = chunk[0]  # py → slot 1
            c[ia, 2] = chunk[1]  # pz → slot 2
            c[ia, 0] = chunk[2]  # px → slot 0
        offset += nm
```

### 5. Two Evaluation Methods

#### Method 1: `orb2grid` (3D grid)

- Uses `wp.orb2grid()` from libwaveplot
- Evaluates on regular 3D grid defined by cube file
- **Slices** to 2D plane for visualization
- **Less reliable** due to interpolation and slice indexing ambiguity
- Used for comparison against WAVEPLOT cube files

#### Method 2: `orb2points` (explicit points)

- Uses `wp.orb2points()` from libwaveplot
- Uses `projector.project_orbital_points()` from OpenCL
- Evaluates at **explicit user-defined points**
- **More reliable** — no grid interpolation needed
- **Recommended** for parity testing

**Plot labels now clearly indicate method:**
- `[orb2grid]` for 3D grid slice
- `[orb2points]` for explicit point evaluation

### 6. Results After Fix

**H2O (Method 2, orb2points):**
- `max|ψ|` identical: libwaveplot = OpenCL = `1.9288e-01`
- RMS ~`8e-6` (purely float32 vs float64 rounding)

**PTCDA (Method 2, orb2points, MO66-75):**
- RMS dropped from `3e-5–3e-4` → `8e-9–9e-8` (3-4 orders of magnitude improvement)
- Consistent with float32 numerical noise

### 7. File Organization

#### Active/Production Scripts

| File | Status | Purpose |
|------|--------|---------|
| `tests/grid/compare_waveplot_lib.py` | **ACTIVE** | Generalized comparison script (this refactoring) |

#### Test Data Directories

| Directory | Contents | Status |
|-----------|----------|--------|
| `tests/grid/dftb_h2o/` | H2O DFTB+ run (detailed.xml, eigenvec.bin, waveplot_in.hsd, cubes) | **ACTIVE** |
| `tests/grid/dftb_ptcda/` | PTCDA DFTB+ run (same files) | **ACTIVE** |
| `tests/grid/dftb_h2/` | H2 DFTB+ run (for radial testing) | **ACTIVE** |

#### Deprecated/Reference Scripts

| File | Status | Notes |
|------|--------|-------|
| `tests/grid/test_waveplot_dftb.py` | DEPRECATED | Old system-specific script, replaced by compare_waveplot_lib.py |
| `tests/grid/test_h2_compare.py` | REFERENCE | H2-specific radial tests, kept for validation |
| `tests/grid/test_sto_radial.py` | REFERENCE | Single atom STO tests, kept for validation |

#### Parser Modules

| File | Status | Purpose |
|------|--------|---------|
| `pyBall/OCL/DFTBplusParser.py` | ACTIVE | `parse_basis_hsd_ang()` (HSD → Å conversion) |
| `pyBall/OCL/Grid.py` | ACTIVE | `GridProjector`, `project_orbital_points()` |

### 8. Usage Examples

```bash
# H2O: 3D grid comparison vs cubes (Method 1)
cd /home/prokop/git/dftbplus
python tests/grid/compare_waveplot_lib.py --dftb-dir tests/grid/dftb_h2o --nmo 6

# H2O: Explicit point evaluation (Method 2, recommended)
python tests/grid/compare_waveplot_lib.py --dftb-dir tests/grid/dftb_h2o \
    --points --plane2d xy --z-offset 0.0 --npoints 48

# PTCDA: HOMO-4 to LUMO+4 at z=2Å
python tests/grid/compare_waveplot_lib.py --dftb-dir tests/grid/dftb_ptcda \
    --points --plane2d xy --z-offset 2.0 --mo-range 66 75 --npoints 64
```

### 9. Files to Add to Git

**Required:**
- `tests/grid/compare_waveplot_lib.py` (generalized version)

**Test data (already present, ensure committed):**
- `tests/grid/dftb_h2o/` (all files)
- `tests/grid/dftb_ptcda/` (all files)
- `tests/grid/dftb_h2/` (all files)

**Output (do NOT commit):**
- `tests/grid/waveplot_output/comparison/` (generated PNGs)

### 10. Key Lessons Learned

1. **Unit consistency is critical:** The coefficient unit conversion bug (`* BOHR2ANG^l`) was the root cause of the large discrepancies.
2. **Explicit point evaluation > grid slicing:** `orb2points` is more reliable than `orb2grid` for parity testing because it avoids interpolation ambiguity.
3. **Parser format mismatch:** DFTB+ XML format differs from what `DFTBplusParser` expects — custom parser needed.
4. **Binary format:** `eigenvec.bin` is a simple flat binary, not Fortran unformatted with record markers.
5. **Generalization principle:** All system-specific data should be parsed, never hardcoded.

### 11. Modularization Refactoring (2026-05)

#### 11.1 Overview

The `compare_waveplot_lib.py` script was refactored to improve modularity and reusability by creating high-level API functions for both the libwaveplot and OpenCL backends, and centralizing comparison utilities in a new TestUtils module.

**Goals:**
- Extract reusable evaluation functions from the monolithic main script
- Create a clean high-level API for both backends
- Centralize comparison and debugging utilities
- Make the main script minimal and focused on orchestration

#### 11.2 New Functions and Locations

**libwaveplot backend (`pyBall/WavePlot/WavePlot.py`):**

```python
def setup_waveplot_from_dftb(dftb_data, libpath):
    """
    Configure WavePlot from parsed DFTB+ data.
    
    Args:
        dftb_data: dict with 'coords_bohr', 'species_wp', 'basis', 'resolution', 'evecs'
        libpath: path to libwaveplot.so
    
    Returns:
        WavePlot instance configured with geometry, basis, and eigenvectors
    """

def evaluate_mos_on_points(wp, mo_indices, points):
    """
    Evaluate multiple molecular orbitals at explicit points.
    
    Args:
        wp: WavePlot instance
        mo_indices: list of 1-based MO indices
        points: (npts, 3) array in Bohr
    
    Returns:
        list of (npts,) arrays, one per MO
    """

def evaluate_mos_on_grid(wp, mo_indices, origin, gridVecs, nPoints):
    """
    Evaluate multiple MOs on a 3D grid.
    
    Args:
        wp: WavePlot instance
        mo_indices: list of 1-based MO indices
        origin: (3,) array in Bohr
        gridVecs: (3, 3) array of grid vectors in Bohr
        nPoints: (3,) array of grid dimensions
    
    Returns:
        list of (nx, ny, nz) arrays, one per MO
    """
```

**OpenCL backend (`pyBall/OCL/Grid.py`):**

```python
def setup_gridprojector_from_dftb(dftb_data, species_list_ang, ctx=None, queue=None, verbosity=0):
    """
    Configure GridProjector from parsed DFTB+ data.
    
    Args:
        dftb_data: dict with 'coords_bohr', 'species_per_atom', 'species_names'
        species_list_ang: list from parse_basis_hsd_ang (Å units)
        ctx: OpenCL context (optional)
        queue: OpenCL command queue (optional)
        verbosity: logging level
    
    Returns:
        (GridProjector instance, atoms_dict) where atoms_dict contains
        'pos' (float32), 'Rcut' (float32), 'type' (int32)
    """

def evaluate_mos_on_points(projector, mo_indices, points, evecs, natoms,
                            species_per_atom, species_names, species_list_ang,
                            norb_per_atom, atoms_dict):
    """
    Evaluate multiple MOs at explicit points via OpenCL.
    
    Args:
        projector: GridProjector instance
        mo_indices: list of 0-based MO indices
        points: (npts, 3) array in Å (float32)
        evecs: (nstates, norb) eigenvector array
        natoms: number of atoms
        species_per_atom: (natoms,) array of species indices
        species_names: list of species names
        species_list_ang: list from parse_basis_hsd_ang
        norb_per_atom: (natoms,) array of orbitals per atom
        atoms_dict: dict with 'pos', 'Rcut', 'type'
    
    Returns:
        list of (npts,) arrays, one per MO
    """
```

**Test utilities (`pyBall/WavePlot/TestUtils.py`):** (new module)

```python
def compute_rms_error(arr1, arr2):
    """Compute RMS and max absolute error between two arrays."""

def compare_point_evaluations(wp_vals, ocl_vals, mo_indices, energies, homo):
    """Compare libwaveplot vs OpenCL point evaluations.
    
    Returns list of dicts with 'mo_index', 'energy', 'rms', 'max', 'is_homo', 'is_lumo'
    """

def compare_grid_evaluations(lib_grids, cube_grids, mo_indices, energies, homo):
    """Compare libwaveplot vs cube file grid evaluations.
    
    Returns list of dicts with 'mo_index', 'energy', 'rms', 'rel_rms', 'ref_max'
    """

def print_comparison_results(results, method_name):
    """Print formatted comparison summary."""

def generate_2d_point_grid(center, rmin, rmax, npoints, plane='xy', z_offset=0.0):
    """Generate 2D grid of points on specified plane."""

def generate_1d_z_scan(center, z_range, npoints):
    """Generate 1D z-scan points through center."""
```

**Plotting utilities (`pyBall/plotUtils.py`):**

```python
def plot_comparison_2d(wp_vals, ocl_vals, diff_vals, extent, system_name,
                       plane_desc, method_tag, mo_indices, energies, homo,
                       out_file, dpi=150):
    """Plot 2D orbital comparison (libwaveplot vs OpenCL)."""

def plot_comparison_1d(z_vals, wp_vals, ocl_vals, system_name, plane_desc,
                       method_tag, mo_indices, energies, homo, out_file, dpi=150):
    """Plot 1D z-scan comparison (libwaveplot vs OpenCL)."""

def plot_grid_slice_comparison(lib_grids, cube_grids, diff_grids, extent,
                               system_name, slice_desc, method_tag, mo_indices,
                               energies, homo, out_file, dpi=150):
    """Plot 3D grid slice comparison (libwaveplot vs cube)."""
```

#### 11.3 Problems Encountered and Fixed

**Problem 1: Function placement in Grid.py**
- **Issue:** Initially placed OpenCL high-level functions inside the `GridProjector` class, causing indentation and structural errors.
- **Fix:** Moved functions to the end of the file (outside the class) to avoid class nesting issues.
- **Lesson:** Be careful about code structure when adding new functions to existing class files.

**Problem 2: Cube file parsing bug in Method 1**
- **Issue:** `read_cube()` was reading from wrong line index (`7+natoms` instead of `6+natoms`), causing ValueError: "cannot reshape array of size 172026 into shape (64,56,48)"
- **Root cause:** Off-by-one error in line indexing. Cube file structure:
  - Lines 1-2: comments
  - Line 3: natoms + origin
  - Lines 4-6: grid dimensions and step vectors
  - Lines 7-9: atom positions (natoms lines)
  - Line 10+: grid data
- **Fix:** Changed `data_lines = lines[7+natoms:]` to `data_lines = lines[6+natoms:]`
- **Additional fix:** Changed return value from 4 to 5 values to match script expectation (added atoms list)
- **Additional fix:** Changed step from (3,3) array to (3,) 1D array to match script's use of `np.diag(step_b)`

**Problem 3: Plotting function expects numpy arrays, not lists**
- **Issue:** `plot_comparison_2d()` expected numpy arrays but received Python lists
- **Fix:** Convert lists to numpy arrays before passing to plotting functions:
  ```python
  wp_vals_2d = np.array([v.reshape(s2) for v in wp_vals])
  ocl_vals_2d = np.array([v.reshape(s2) for v in ocl_vals])
  diff_vals_2d = np.array([v.reshape(s2) for v in diff_vals])
  ```

#### 11.4 How to Run Tests

**All tests from `/home/prokop/git/dftbplus/tests/grid/`:**

```bash
# Method 1: 3D grid comparison (libwaveplot vs cube files)
python compare_waveplot_lib.py --dftb-dir dftb_h2o --nmo 6 --no-show

# Method 2: 2D point evaluation on XY plane at z=0 Å
python compare_waveplot_lib.py --dftb-dir dftb_h2o --points --plane2d xy --z-offset 0.0 --mo-range 4 6 --npoints 32 --no-show

# Method 2: 1D z-scan
python compare_waveplot_lib.py --dftb-dir dftb_h2o --points --z-range -3.0 3.0 --mo-range 4 6 --npoints 100 --no-show

# PTCDA example (larger molecule)
python compare_waveplot_lib.py --dftb-dir dftb_ptcda --points --plane2d xy --z-offset 2.0 --mo-range 66 75 --npoints 64 --no-show
```

**Output files:** Saved to `/home/prokop/git/dftbplus/tests/grid/waveplot_output/comparison/`

**Expected RMS errors:**
- Method 1 (libwaveplot vs cube): ~1e-8
- Method 2 (libwaveplot vs OpenCL): ~1e-21 to 1e-6 (depends on MO and evaluation type)

#### 11.5 Refactored Main Script Structure

The refactored `compare_waveplot_lib.py` main() now follows this pattern:

```python
# 1. Parse DFTB+ data (unchanged)
geo = parse_detailed_xml_custom(...)
species_list_ang = parse_basis_hsd_ang(...)
evecs_full = parse_eigenvec_bin_custom(...)

# 2. Setup using high-level functions
dftb_data_wp = {coords_bohr, species_wp, basis, resolution, evecs}
wp = setup_waveplot_from_dftb(dftb_data_wp, LIB_PATH)

dftb_data_ocl = {coords_bohr, species_per_atom, species_names}
projector, atoms_dict = setup_gridprojector_from_dftb(dftb_data_ocl, species_list_ang)

# 3. Evaluate using high-level functions
wp_vals = wp_evaluate_mos_on_points(wp, mo_indices, points)
ocl_vals = ocl_evaluate_mos_on_points(projector, mo_indices, points, ...)

# 4. Compare using TestUtils
results = compare_point_evaluations(wp_vals, ocl_vals, mo_indices, energies, homo)
print_comparison_results(results, "libwaveplot vs OpenCL")

# 5. Plot using plotUtils
plot_comparison_2d(wp_vals_2d, ocl_vals_2d, diff_vals_2d, extent, ...)
```

**Lines reduced:** ~80 lines shorter than original

### 12. Generic Script Refactoring: test_waveplot_dftb.py (2026-05)

#### 12.1 Overview

The `test_waveplot_dftb.py` script was originally designed as a system-specific test with separate functions for H2O and PTCDA. This was refactored to be completely generic, working for any DFTB+ calculation directory, similar to `compare_waveplot_lib.py`.

**Goals:**
- Remove system-specific functions (`run_h2o_test`, `run_ptcda_test`)
- Remove ad-hoc utility functions redundant with shared modules
- Create a single generic `main()` that works for any DFTB+ directory
- Use shared parsing and plotting functions throughout
- Eliminate code duplication

#### 12.2 Changes Made

**Removed system-specific functions:**
- `run_h2o_test()` - H2O-specific test logic with hardcoded grid parameters
- `run_ptcda_test()` - PTCDA-specific test logic with XYZ file parsing

**Removed ad-hoc utility functions:**
- `validate_grid()` - grid validation (redundant, can be inlined if needed)
- `compute_grid_extent()` - extent computation (simple inline calculation)
- `plot_orbital_2d()` - 2D orbital plotting (replaced with shared `plotAtoms` from plotUtils)
- `plot_orbital_slices()` - 3 orthogonal slice plotting (simplified to single middle slice)
- `extract_plane_at_z()` - plane extraction from 3D grid (inline calculation)
- `best_slice()` - find slice with max |psi| (simplified to middle slice)
- `plot_mo_slice()` - MO slice plotting with atoms (replaced with shared functions)

**Updated argument parser:**
- Replaced `--system {H2O,PTCDA}` with `--dftb-dir <path>`
- Removed `--xyz` argument (no longer needed)
- Kept generic grid parameters: `--step`, `--margin`, `--ngrid`, `--points`, `--z-range`, `--npoints`, `--plot-mos`

**Created generic main():**
```python
def main():
    # Determine DFTB+ directory (or default to dftb_h2o)
    dftb_dir = Path(args.dftb_dir) if args.dftb_dir else Path(__file__).parent / 'dftb_h2o'
    
    # Parse DFTB+ data using shared functions
    geo = parse_detailed_xml_custom(dftb_dir / 'detailed.xml')
    evecs_full = parse_eigenvec_bin_custom(dftb_dir / 'eigenvec.bin', nstates, norb)
    species_list_sto = parse_basis_hsd_ang(dftb_dir / 'waveplot_in.hsd')
    
    # Setup projector using shared function
    projector, atoms_dict = setup_gridprojector_from_dftb(dftb_data_ocl, species_list_sto)
    
    # Auto-detect HOMO/LUMO from occupations
    homo_idx = np.where(occupations > 0.5)[0][-1]
    
    # Auto-select MO range (HOMO-2 to LUMO+2 by default)
    mo_start = max(0, homo_idx - 2)
    mo_end = min(nstates - 1, homo_idx + 2)
    
    # Project and plot using shared functions
    for imo in mo_indices:
        coeffs_k = evec_to_kernel_coeffs(evecs_full[imo], ...)
        grid_3d = projector.project_orbital(coeffs_k, ...)
        # Plot with plotAtoms from plotUtils
```

#### 12.3 How It Works Now

**Input requirements:**
Any DFTB+ calculation directory containing:
- `detailed.xml` - geometry, nstates, norb, occupations
- `eigenvec.bin` - eigenvectors (Fortran unformatted binary)
- `waveplot_in.hsd` - STO basis parameters
- `band.out` - optional, for energy labels

**Usage:**
```bash
# H2O (default if --dftb-dir not specified)
python test_waveplot_dftb.py --step 0.2 --no-show

# PTCDA or any other molecule
python test_waveplot_dftb.py --dftb-dir dftb_ptcda --step 0.2 --no-show

# Point evaluation along z-axis
python test_waveplot_dftb.py --dftb-dir dftb_h2o --points --npoints 100 --no-show

# Specific MOs
python test_waveplot_dftb.py --dftb-dir dftb_h2o --plot-mos 4 5 6 --no-show
```

**Automatic features:**
- Auto-detects HOMO from occupations
- Auto-selects MO range (HOMO-2 to LUMO+2 by default)
- Auto-computes grid dimensions based on molecular extent
- Auto-reads energies from `band.out` if available
- Works for any molecule size (tested: H2O with 3 atoms, PTCDA with 38 atoms)

**Shared functions used:**
- `parse_detailed_xml_custom()` - geometry and occupations
- `parse_eigenvec_bin_custom()` - eigenvectors
- `parse_basis_hsd_ang()` - STO basis
- `setup_gridprojector_from_dftb()` - projector setup
- `evec_to_kernel_coeffs()` - coefficient conversion
- `plotAtoms()` from plotUtils - atom plotting

#### 12.4 Results

**Before refactoring:**
- ~840 lines total
- System-specific code for H2O and PTCDA
- Ad-hoc plotting functions
- Hardcoded grid parameters for H2O
- Manual XYZ parsing for PTCDA

**After refactoring:**
- ~140 lines total (reduced by ~700 lines)
- Generic code works for any DFTB+ directory
- Uses shared functions throughout
- Automatic grid setup based on molecular extent
- No system-specific logic

**Test results:**
- H2O (3 atoms, 6 states): ✓ passed
- PTCDA (38 atoms, 128 states): ✓ passed

#### 12.5 Comparison with compare_waveplot_lib.py

Both scripts now follow the same pattern:

**compare_waveplot_lib.py:**
- Purpose: Compare libwaveplot vs OpenCL vs cube file
- Focus: Accuracy testing and validation
- Methods: 3D grid comparison, 2D point evaluation, 1D z-scan
- Uses: TestUtils for comparison metrics, plotUtils for comparison plots

**test_waveplot_dftb.py:**
- Purpose: Test OpenCL orbital projection
- Focus: Visualization and debugging
- Methods: 3D grid projection, point evaluation
- Uses: Shared parsing functions, plotAtoms for visualization

Both scripts:
- Use shared DFTBplusParser functions
- Use shared GridProjector setup
- Are generic (work for any DFTB+ directory)
- Have no system-specific code
- Auto-detect HOMO/LUMO from occupations

### 13. Reference: Input File Formats

#### `geom.xyz` (for DFTB+ input)
```
3
comment line
O  0.0  0.0  0.0
H -0.76 0.59 0.0
H  0.76 0.59 0.0
```
Units: Å (DFTB+ converts to Bohr internally)

#### `dftb_in.hsd` (DFTB+ input)
```hsd
Geometry = xyzFormat {
<<< "geom.xyz"
}
Hamiltonian = DFTB {
  SCC = Yes
  SlaterKosterFiles = Type2FileNames {
    Prefix = "/path/to/slakos/mio-1-1/"
    Separator = "-"
    Suffix = ".skf"
  }
  MaxAngularMomentum {
    C = "p"
    O = "p"
    H = "s"
  }
}
Analysis {
  WriteEigenvectors = Yes
}
Options {
  WriteDetailedXml = Yes
}
```

#### `waveplot_in.hsd` (authoritative STO parameters)
```hsd
Basis {
  Resolution = 0.04  # Bohr
  H = {
    AtomicNumber = 1
    Orbital = {
      AngularMomentum = 0
      Occupation = 1.0
      Cutoff = 6.0  # Bohr
      Exponents = { 0.967 }  # Bohr⁻¹
      Coefficients = { 1.0 }
    }
  }
  O = {
    AtomicNumber = 8
    Orbital = {
      AngularMomentum = 0
      Occupation = 2.0
      Cutoff = 7.0  # Bohr
      Exponents = { 2.779 }
      Coefficients = { 1.0 }
    }
    Orbital = {
      AngularMomentum = 1
      Occupation = 4.0
      Cutoff = 7.0  # Bohr
      Exponents = { 2.247 }
      Coefficients = { 1.0 }
    }
  }
}
```
**This is the authoritative source** for all STO parameters. Values are in Bohr.

The plan provides specific file references, data structure mappings, and success criteria to guide the implementation.