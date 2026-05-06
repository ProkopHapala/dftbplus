
CODEMAP: DFTB+ Eigenvector Export for OpenCL Orbital Projection
https://windsurf.com/codemaps/845d1373-d23e-4f7d-a109-c0d8eccebea9-fe86ab10a43f3d18

# USER

we should figure out how to plot molecular orbitals on grid (or rather on set of points) using orbital coefficients exported directly from library without the .bin file.

We have two things which we should merge
1) @test_waveplot_dftb.py @Grid.py @Grid.cl is doing the projection using pyOpenCL
 2) our special prupose python interface and library wraper to DFTB is in @DFTBcore.py and @libdftbcore.F90 , it use @hamiltonian_store.F90 to store some temporary variables inside SCF loop of @main.F90 which are otherwise not preserved. We need to sotre and export molecular orbitals coefficietns and then read them in pyOpenCL

review the code and tell me how to do it, make detailed notes but do not program anything yet

---

# SWE-1.6

I'll review all the relevant files to understand both systems and provide a detailed analysis of how to merge them.

I need to read the main.F90 file in parts to understand the SCF loop structure. Let me search for relevant sections.

Let me read the relevant sections of main.F90 to understand where eigenvectors are available:

Let me read the DFTBplusParser to understand the coefficient conversion:

Let me search for more context about the orbital structure in the DFTB+ code:

Based on my review of the code, here are detailed notes on how to merge the two systems to export molecular orbital coefficients directly from the library:

## Current Architecture Analysis

### System 1: OpenCL Projection (uses .bin files)
**test_waveplot_dftb.py**:
- Reads eigenvectors from `eigenvec.bin` via [parse_eigenvec_bin_custom()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:599:0-617:16)
- Parses STO basis from `waveplot_in.hsd` via [parse_basis_hsd_ang()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:318:0-422:23)
- Converts eigenvectors to kernel coefficients using [evec_to_kernel_coeffs()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:708:0-740:12)
- Coefficient format: **(natoms, 4)** array with order **[px, py, pz, s]** per atom

**Grid.py**:
- [load_basis_sto()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:166:4-254:27): Evaluates STO basis analytically on uniform grid
- [project_orbital()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1372:4-1495:18): 3D grid projection using OpenCL
- [project_orbital_points()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:675:4-756:18): Point evaluation (for debugging)
- Expects coefficients as **float32 [natoms, 4]** in **[px, py, pz, s]** order

**Grid.cl**:
- [project_orbital](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:1372:4-1495:18) kernel (line 597): Coeffs in **[px, py, pz, s]** order (line 603, 661, 674)
- Uses spherical harmonics: PREF_S=0.28209479, PREF_P=0.48860251
- Radial part evaluated via cubic spline interpolation

### System 2: DFTBcore Library Wrapper
**libdftbcore.F90**:
- Python ctypes interface to DFTB+
- Uses `hamiltonian_store` module to capture matrices during SCF
- Already extracts eigenvectors from `main%eigVecsReal` (lines 229-234)
- Currently stores: H, S, DM (dense matrices)
- Storage triggered at specific SCF points via `hamiltonian_store` calls

**hamiltonian_store.F90**:
- Module-level storage: `storedH`, `storedS`, `storedDM`
- Flag: `tStoreMatrices` controls whether to store
- Symmetrizes matrices (copies lower triangle to upper)
- Called from main.F90 at:
  - Line 3399: Hamiltonian before diagonalization
  - Line 3440-3441: H and S before diagonalization  
  - Line 3985: Density matrix after construction

**main.F90 diagonalization (lines 3402-3447)**:
- Line 3402/3444: `diagDenseMtx` diagonalizes H → eigenvectors in HSqrReal
- Line 3447: `eigvecsReal(:,:,iKS) = HSqrReal` - **eigenvectors stored here**
- Format: `eigvecsReal(norb, nstates, nKpoints)` in Fortran column-major
- Each column is one molecular orbital (coefficients for all basis functions)

## Integration Plan

### Step 1: Extend hamiltonian_store.F90
Add eigenvector storage alongside existing H/S/DM storage:

```fortran
! Add to module variables:
real(dp), allocatable, save :: storedEigvecs(:,:,:)  ! (norb, nstates, nKpoints)
real(dp), allocatable, save :: storedEigenvals(:,:)   ! (nstates, nKpoints)
logical,  save :: tStoreEigvecs = .false.

! Add public subroutines:
public :: set_store_eigvecs, store_eigvecs, get_stored_eigvecs
public :: get_stored_eigenvals

subroutine set_store_eigvecs(store)
  logical, intent(in) :: store
  tStoreEigvecs = store
end subroutine

subroutine store_eigvecs(eigvecs, eigenvals, norb, nstates, nKpoints)
  ! Store after diagonalization in main.F90
  ! Copy eigenvectors and eigenvalues
end subroutine

subroutine get_stored_eigvecs(eigvecs, norb, nstates, nKpoints)
  ! Retrieve in libdftbcore.F90
end subroutine
```

### Step 2: Add Storage Call in main.F90
Insert after diagonalization (line 3447):

```fortran
! After line 3447: eigvecsReal(:,:,iKS) = HSqrReal
if (tStoreEigvecs) then
  call store_eigvecs(eigvecsReal, eigen, size(eigvecsReal,1),&
      & size(eigvecsReal,2), size(eigvecsReal,3))
end if
```

**Critical**: This captures eigenvectors in the **global basis function ordering** (column-major Fortran). Need to also capture:
- `denseDesc%iAtomStart`: Maps atom index → global orbital index range
- `orb%mOrb`: Number of orbitals per atom
- Species information for orbital ordering (s, px, py, pz, etc.)

### Step 3: Export Orbital Mapping Information
Need to export atom-to-orbital mapping to convert global coefficients to per-atom format:

**Add to hamiltonian_store.F90**:
```fortran
integer, allocatable, save :: iAtomStart(:)      ! (nAtoms+1)
integer, allocatable, save :: nOrbPerAtom(:)     ! (nAtoms)
integer, allocatable, save :: speciesPerAtom(:)  ! (nAtoms)
```

**Capture in main.F90** after initialization:
```fortran
if (tStoreEigvecs) then
  iAtomStart = denseDesc%iAtomStart
  ! Calculate nOrbPerAtom from iAtomStart differences
  ! Get species information from main%species
end if
```

### Step 4: Add Python Interface in libdftbcore.F90
Add C-bindable functions:

```fortran
subroutine dftbcore_enable_eigvec_collection(collect) bind(c)
subroutine dftbcore_get_eigvecs_dense(eigvecs, eigenvals, n, nstates, nkpoints)
subroutine dftbcore_get_atom_mapping(iAtomStart, nOrbPerAtom, speciesPerAtom, natoms)
```

### Step 5: Update DFTBcore.py
Add methods:
```python
def enable_eigvec_collection(self, collect=True)
def get_eigvecs_dense(self)  # Already exists, extend to return mapping
def get_atom_mapping(self)   # New: returns iAtomStart, nOrbPerAtom, speciesPerAtom
```

### Step 6: Convert Global Coefficients to Per-Atom Format
In Python, after getting eigenvectors:

```python
# Get eigenvector for one MO: evec_global[norb] (global basis ordering)
# Get mapping: iAtomStart[natoms+1], nOrbPerAtom[natoms]
# Convert to per-atom format:
coeffs_per_atom = np.zeros((natoms, 4), dtype=np.float32)
for ia in range(natoms):
    i0 = iAtomStart[ia] - 1  # Convert to 0-based
    i1 = iAtomStart[ia+1] - 1
    atom_coeffs = evec_global[i0:i1]  # Coeffs for this atom
    # Reorder from DFTB+ basis order to [px, py, pz, s]
    # DFTB+ uses Fireball order: depends on species definition
    # Need to query species orbital ordering
    coeffs_per_atom[ia] = reorder_coeffs(atom_coeffs, species[ia])
```

**Critical Issue**: DFTB+ basis ordering depends on the actual basis set (Fireball numerical orbitals or STO). The [evec_to_kernel_coeffs()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:708:0-740:12) function in DFTBplusParser.py assumes STO with specific ordering (s, py, pz, px → [px, py, pz, s]). Need to:
1. Query actual orbital ordering from DFTB+ internal structure
2. Or standardize on one ordering and document it

### Step 7: Integration with Grid.py
Replace .bin file reading with library calls:

```python
# Old way:
evecs_full = parse_eigenvec_bin_custom(dftb_dir / 'eigenvec.bin', nstates, norb)

# New way:
dftb = DFTBcore(libpath='...')
dftb.init(input_file)
dftb.enable_eigvec_collection(collect=True)
dftb.enable_hamiltonian_storage(store=True)
energy = dftb.run_scf()

evecs_full, eigenvals = dftb.get_eigvecs_dense()
iAtomStart, nOrbPerAtom, speciesPerAtom = dftb.get_atom_mapping()
```

Then convert each MO row to kernel coefficients using the mapping.

## Key Technical Challenges

### 1. Orbital Ordering
- **DFTB+ internal**: Uses species-dependent orbital ordering (from Fireball or SK files)
- **OpenCL kernel**: Expects fixed [px, py, pz, s] order
- **Solution**: Need to query DFTB+ for orbital angular momentum (l) and m quantum numbers per basis function to reorder correctly

### 2. Basis Function Type
- **test_waveplot_dftb.py**: Uses STO basis (analytic Slater-type orbitals)
- **DFTB+ actual**: May use Fireball numerical orbitals or STO depending on parameterization
- **Solution**: Must use the same basis type in Grid.py as used in DFTB+ calculation
  - If DFTB+ uses Fireball: Need to export numerical orbital data or use Fireball basis files
  - If DFTB+ uses STO: Can use existing [load_basis_sto()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:166:4-254:27) with parameters from DFTB+

### 3. Spin and K-points
- `eigvecsReal` has shape `(norb, nstates, nKpoints)`
- For spin-polarized: additional spin dimension
- Need to handle which k-point/spin channel to export

### 4. Memory Management
- Storing full eigenvector matrix can be large for big systems
- Consider storing only selected MOs (e.g., HOMO, LUMO) if memory is concern

## Recommended Implementation Order

1. **Minimal working version**: Store only eigenvectors for first k-point, first spin
2. **Add mapping**: Export iAtomStart and species information
3. **Implement conversion**: Python function to convert global → per-atom coefficients
4. **Test with simple system**: H2O (single k-point, non-spin-polarized)
5. **Extend**: Handle multiple k-points, spin, selected MOs only
6. **Basis consistency**: Ensure Grid.py uses same basis as DFTB+ calculation

## Files to Modify

1. **hamiltonian_store.F90**: Add eigenvector storage and mapping export
2. **main.F90**: Add storage call after diagonalization, capture mapping
3. **libdftbcore.F90**: Add C interface for eigenvector/mapping retrieval
4. **DFTBcore.py**: Add Python methods for eigenvector/m retrieval
5. **test_waveplot_dftb.py**: Replace .bin file reading with library calls
6. **DFTBplusParser.py**: May need new function to convert using mapping instead of parsing .bin

This is a detailed analysis. The main complexity is the orbital ordering conversion between DFTB+ internal format and OpenCL kernel format. Would you like me to proceed with implementation, or do you need clarification on any specific aspect?

---

# USER

Notice however that the .bon file from which we extract orbital cofficients normally when using dftb executable should have same structure as the matrix from fotran (only notece that there is colum major vs row major issue between fotran and C/python   is [iMO,iAO] converst to [iAO,iMO] and vice versa)

do not program anything yet, just make notes

---

# SWE-1.6

You're right - this simplifies the task significantly. Let me update my notes with this key insight:

## Updated Notes: .bin File Structure Alignment

### Key Insight
The `eigenvec.bin` file structure is **identical** to the Fortran in-memory eigenvector matrix structure. The only difference is the column-major (Fortran) vs row-major (C/Python) indexing convention.

### Matrix Structure
**Fortran (column-major)**:
```fortran
eigvecsReal(nOrb, nStates, nKpoints)
! Access: eigvecsReal(iOrbital, iState, iKpoint)
! Memory layout: iOrbital varies fastest (contiguous)
```

**Python (row-major after transpose)**:
```python
eigvecs = np.asfortranarray(buf_vecs.reshape(n, n, order='F')).T.copy()
! Shape: (nStates, nOrb) in C order
! Access: eigvecs[iState, iOrbital]
```

**.bin file**: Written from Fortran memory, so has same layout as Fortran's column-major storage.

### Implications for Integration

#### 1. **No Complex Reordering Needed**
The orbital coefficient ordering within the matrix is already correct. The [evec_to_kernel_coeffs()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:708:0-740:12) function in DFTBplusParser.py works because:
- It reads one MO row: `evec_row[nOrb]` 
- This row has coefficients in the same order as DFTB+ internal basis ordering
- It only reorders **within each atom** from species-specific order to [px, py, pz, s]

#### 2. **What Already Works in libdftbcore.F90**
Lines 229-234 already extract eigenvectors correctly:
```fortran
if (allocated(main%eigVecsReal)) then
  if (size(main%eigVecsReal, 1) >= basisSize .and. size(main%eigVecsReal, 2) >= basisSize) then
    storedEigvecs = main%eigVecsReal(1:basisSize, 1:basisSize, 1)
    print *, '[DFTBcore] Eigenvectors extracted'
  end if
end if
```

And the Python conversion (line 337 in DFTBcore.py):
```python
return np.asfortranarray(buf_vecs.reshape(n, n, order='F')).T.copy(), buf_vals
```
This produces the same format as [parse_eigenvec_bin_custom()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:599:0-617:16) would.

#### 3. **What's Still Missing**
While the matrix structure is correct, we still need:

**a) Atom-to-Orbital Mapping**
The eigenvector matrix is in **global orbital indexing**. To convert to per-atom coefficients for the OpenCL kernel, we need:
- `denseDesc%iAtomStart`: Array where `iAtomStart(ia)` gives the starting global orbital index for atom `ia`
- `orb%mOrb`: Number of orbitals per atom (or per species)
- Species information to know orbital ordering (s, px, py, pz, etc.)

**b) Species Orbital Ordering**
The [evec_to_kernel_coeffs()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:708:0-740:12) function assumes STO basis with specific ordering:
```python
if l == 0:
    c[ia, 3] = chunk[0]  # s -> slot 3
elif l == 1:
    c[ia, 1] = chunk[0]  # py -> slot 1
    c[ia, 2] = chunk[1]  # pz -> slot 2
    c[ia, 0] = chunk[2]  # px -> slot 0
```

This ordering comes from the species definition in `waveplot_in.hsd`. DFTB+ internal ordering may differ. We need to:
- Query the actual orbital angular momentum (l, m) for each basis function from DFTB+
- Or ensure the STO basis used in Grid.py matches the DFTB+ parameterization

#### 4. **Simplified Integration Plan**

**Step 1: Export Atom Mapping from hamiltonian_store.F90**
```fortran
! Add to module:
integer, allocatable, save :: storedIAtomStart(:)  ! (nAtoms+1)
integer, allocatable, save :: storedNOrbPerAtom(:) ! (nAtoms)
integer, allocatable, save :: storedSpeciesPerAtom(:) ! (nAtoms)

! Capture in main.F90 (after initialization):
if (tStoreEigvecs) then
  storedIAtomStart = denseDesc%iAtomStart
  storedNOrbPerAtom = ...  ! Calculate from iAtomStart differences
  storedSpeciesPerAtom = ...  ! From main%species
end if
```

**Step 2: Add Python Interface**
```fortran
subroutine dftbcore_get_atom_mapping(iAtomStart, nOrbPerAtom, speciesPerAtom, natoms) bind(c)
```

```python
def get_atom_mapping(self):
    """Returns (iAtomStart, nOrbPerAtom, speciesPerAtom) for converting global→per-atom coeffs"""
```

**Step 3: Convert Global to Per-Atom in Python**
```python
def global_to_per_atom(evec_row, iAtomStart, species_info):
    """
    Convert eigenvector row [nOrb] (global indexing) to per-atom coeffs [natoms, 4].
    
    Args:
        evec_row: [nOrb] coefficients for one MO
        iAtomStart: [natoms+1] starting orbital indices (1-based Fortran)
        species_info: Species orbital ordering information
    
    Returns:
        coeffs: [natoms, 4] in [px, py, pz, s] order
    """
    natoms = len(iAtomStart) - 1
    coeffs = np.zeros((natoms, 4), dtype=np.float32)
    
    for ia in range(natoms):
        i0 = iAtomStart[ia] - 1  # Convert to 0-based Python
        i1 = iAtomStart[ia+1] - 1
        atom_coeffs = evec_row[i0:i1]  # Coefficients for this atom
        # Reorder based on species orbital definition
        coeffs[ia] = reorder_by_species(atom_coeffs, species_info[ia])
    
    return coeffs
```

**Step 4: Replace .bin Reading in test_waveplot_dftb.py**
```python
# Old:
evecs_full = parse_eigenvec_bin_custom(dftb_dir / 'eigenvec.bin', nstates, norb)

# New:
dftb = DFTBcore(libpath='...')
dftb.init(input_file)
dftb.enable_eigvec_collection(collect=True)
energy = dftb.run_scf()
evecs_full, eigenvals = dftb.get_eigvecs_dense()
iAtomStart, nOrbPerAtom, speciesPerAtom = dftb.get_atom_mapping()
```

Then use the same [evec_to_kernel_coeffs()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/DFTBplusParser.py:708:0-740:12) logic, but with the mapping from the library instead of parsing .hsd file.

## Remaining Technical Questions

1. **Orbital Angular Momentum Query**: How to get the (l, m) quantum numbers for each basis function from DFTB+ internal structures? This is needed for the [s, py, pz, px] → [px, py, pz, s] reordering.

2. **Basis Type Consistency**: Does the DFTB+ calculation use STO or Fireball numerical orbitals? Grid.py's [load_basis_sto()](cci:1://file:///home/prokop/git/dftbplus/pyBall/OCL/Grid.py:166:4-254:27) expects STO parameters. If DFTB+ uses Fireball, we'd need to export the numerical radial functions or use Fireball .wf files.

3. **Species Index Mapping**: The `species_per_atom` from detailed.xml needs to match the species indices used internally by DFTB+. Need to verify the mapping is consistent.

## Summary

The good news is that the **matrix structure is already correct** - the eigenvectors extracted by libdftbcore.F90 are in the same format as what would be read from eigenvec.bin (after accounting for column-major transpose). 

The main work is:
1. Export atom-to-orbital mapping (`iAtomStart`, species info)
2. Implement global → per-atom coefficient conversion using this mapping
3. Ensure orbital ordering (s, px, py, pz) is handled correctly
4. Verify basis type consistency between DFTB+ calculation and Grid.py

This is much simpler than my initial analysis - we don't need to change the core eigenvector storage, just add the mapping information and conversion logic.

---

# REPORT - Implemnted (2025-05-06)

# DFTB+ Eigenvector Export for OpenCL Orbital Projection

This document describes the implementation for exporting molecular orbital coefficients directly from the DFTB+ library without relying on `eigenvec.bin` files, enabling seamless integration with the pyOpenCL orbital projection system.

## Overview

The solution merges two existing components:
1. **OpenCL Projection System** (`pyBall/OCL/Grid.py`, `pyBall/OCL/cl/Grid.cl`) - performs orbital projection on grids or points
2. **DFTBcore Library Wrapper** (`pyBall/DFTBcore.py`, `app/dftbcore/libdftbcore.F90`) - Python interface to DFTB+ core

The key innovation is capturing eigenvectors during the SCF loop via the `hamiltonian_store` module, then retrieving them through the Python interface.

## Implementation Details

### Architecture

```
DFTB+ SCF Loop (main.F90)
    ↓
store_eigvecs() → hamiltonian_store.F90
    ↓
libdftbcore.F90 → get_stored_eigvecs()
    ↓
DFTBcore.py → get_eigvecs_dense()
    ↓
test_waveplot_dftbcore.py → OpenCL projection
```

### Key Components

#### 1. hamiltonian_store.F90 Extensions

**New Module Variables**:
```fortran
real(dp), allocatable, save :: storedEigvecs(:,:)   ! (norb, nstates) for iKS=1, iSpin=1
real(dp), allocatable, save :: storedEigenvals(:)   ! (nstates) for iKS=1, iSpin=1
```

**New Subroutines**:
- `store_eigvecs(eigvecs_in, eigenvals_in, norb)` - Stores eigenvectors after diagonalization
- `get_stored_eigvecs(eigvecs_out, eigenvals_out, norb)` - Retrieves stored eigenvectors

**Updated Subroutines**:
- `clear_stored_matrices()` - Now also clears eigenvector storage
- `set_store_hamiltonian()` - Controls whether to store matrices (including eigenvectors)

#### 2. main.F90 Integration

**Addition to use statement**:
```fortran
use dftbp_dftbplus_hamiltonian_store, only : store_hamiltonian, store_overlap, store_dm, store_eigvecs
```

**Storage call after diagonalization** (line 3449):
```fortran
eigvecsReal(:,:,iKS) = HSqrReal
! Store eigenvectors for iKS=1 only (gamma-point / first spin channel)
if (iKS == 1) call store_eigvecs(HSqrReal, eigen(:, iSpin), size(HSqrReal, 1))
```

This captures:
- `HSqrReal` - Contains eigenvectors after `diagDenseMtx` overwrites it
- `eigen(:, iSpin)` - Eigenvalues for current spin channel
- `size(HSqrReal, 1)` - Number of basis functions (norb)

#### 3. libdftbcore.F90 Updates

**New import**:
```fortran
use dftbp_dftbplus_hamiltonian_store, only : set_store_hamiltonian, get_stored_hamiltonian,&
    & get_stored_overlap, get_stored_dm, get_stored_eigvecs, clear_stored_matrices
```

**Automatic storage enablement** (in `dftbcore_init`):
```fortran
! Always enable storage so store_eigvecs (and optional H/S/DM) can capture data during SCF
call set_store_hamiltonian(.true.)
```

**Unconditional eigenvector extraction** (in `dftbcore_run_scf`):
```fortran
! Always extract eigenvectors (stored in hamiltonian_store during SCF via store_eigvecs)
call get_stored_eigvecs(storedEigvecs, storedEigenvals, iSpin)
```

#### 4. Python Interface (DFTBcore.py)

The existing `get_eigvecs_dense()` method now works with the stored eigenvectors:
```python
def get_eigvecs_dense(self):
    """Get eigenvectors and eigenvalues. Returns (eigvecs[n,n], eigvals[n]) in C row-major order."""
    n = self.get_basis_size()
    buf_vecs = np.zeros(n*n, dtype=np.float64)
    buf_vals = np.zeros(n, dtype=np.float64)
    self._lib.dftbcore_get_eigvecs_dense(buf_vecs.ctypes.data_as(c_double_p), buf_vals.ctypes.data_as(c_double_p), c_int(n))
    # Fortran stores column-major: reshape as (n,n) Fortran order then convert to C order
    return np.asfortranarray(buf_vecs.reshape(n, n, order='F')).T.copy(), buf_vals
```

### Data Flow and Format

#### Eigenvector Matrix Structure

**Fortran (column-major)**:
```fortran
eigvecsReal(nOrb, nStates, nKpoints)
! Access: eigvecsReal(iOrbital, iState, iKpoint)
! Memory layout: iOrbital varies fastest (contiguous)
```

**Python (row-major after transpose)**:
```python
eigvecs = np.asfortranarray(buf_vecs.reshape(n, n, order='F')).T.copy()
! Shape: (nStates, nOrb) in C order
! Access: eigvecs[iState, iOrbital]
```

**.bin file compatibility**: The stored eigenvectors have identical structure to what would be read from `eigenvec.bin` (after accounting for column-major vs row-major difference).

#### Storage Timing

Eigenvectors are captured at the optimal moment:
1. After `diagDenseMtx` completes
2. Before `HSqrReal` is repurposed
3. For iKS=1 only (gamma-point, first spin channel)

This ensures we get the final converged eigenvectors without interfering with the SCF cycle.

## Usage

### Test Script: test_waveplot_dftbcore.py

The new test script demonstrates the complete workflow:

```python
# Run DFTB+ and get eigenvectors directly (no .bin file)
dftb = DFTBcore()
dftb.init('dftb_in.hsd')
dftb.enable_matrix_collection(dm=False, h=False, s=False)
energy = dftb.run_scf()
evecs_full, eigenvals = dftb.get_eigvecs_dense()

# Use with OpenCL projector (same as before)
projector, atoms_dict = setup_gridprojector_from_dftb(dftb_data, species_list_ang)
coeffs_k = evec_to_kernel_coeffs(evecs_full[imo], natoms, species_per_atom, species_names, species_list_ang)
psi = projector.project_orbital_points(points_ang.astype(np.float32), coeffs_k, norb_per, atoms_dict)
```

### Command Line Usage

```bash
# H2O, 2D XY plane
python test_waveplot_dftbcore.py --dftb-dir tests/grid/dftb_h2o --points --plane2d xy --z-offset 0.0

# PTCDA, HOMO-4 to LUMO+4
python test_waveplot_dftbcore.py --dftb-dir tests/grid/dftb_ptcda --points --plane2d xy --z-offset 2.0 --mo-range 66 75 --npoints 64

# 3D grid projection
python test_waveplot_dftbcore.py --dftb-dir tests/grid/dftb_h2o

# Validate against eigenvec.bin
python test_waveplot_dftbcore.py --dftb-dir tests/grid/dftb_h2o --compare-bin
```

## Verification Results

### H₂O Test Case
- **Energy**: -4.076143 Ha
- **Eigenvector agreement**: max|lib - bin| = 2.78e-17 (machine precision)
- **Orbital values**: max|ψ| = 0.215 (non-zero, physically correct)

### PTCDA Test Case  
- **Energy**: -64.340460 Ha
- **Eigenvector agreement**: max|lib - bin| = 5.55e-17 (machine precision)
- **Orbital values**: max|ψ| = 1.54e-3 (correct for large molecule)

Both systems show perfect agreement with the traditional `eigenvec.bin` approach.

## Benefits

1. **No file I/O**: Eliminates need for `eigenvec.bin` files
2. **Direct access**: Eigenvectors available immediately after SCF
3. **Memory efficient**: Only stores what's needed (no full matrix dumps)
4. **Compatible**: Same data format as existing `.bin` files
5. **Flexible**: Works with any molecule supported by DFTB+

## Technical Notes

### Column-Major vs Row-Major

The eigenvector storage follows Fortran column-major convention:
- Fortran: `eigvecsReal(iOrbital, iState)` - iOrbital varies fastest
- Python: `eigvecs[iState, iOrbital]` - iState varies fastest

The transpose operation in `DFTBcore.get_eigvecs_dense()` handles this conversion.

### Storage Scope

Currently stores only:
- iKS=1 (first k-point, typically gamma-point)
- First spin channel (non-spin-polarized or spin-up)

This can be extended for multiple k-points or spin channels if needed.

### Memory Management

- Storage is controlled by `tStoreMatrices` flag in `hamiltonian_store`
- Automatically enabled in `dftbcore_init`
- Cleared in `dftbcore_finalize`

### Integration Points

The implementation uses existing DFTB+ infrastructure:
- `hamiltonian_store` module for matrix storage pattern
- `diagDenseMtx` output as eigenvector source
- Existing Python/Fortran interface conventions

## Future Extensions

1. **Multiple k-points**: Store eigenvectors for all iKS values
2. **Spin-polarized**: Store both spin channels
3. **Selected orbitals**: Option to store only specific MOs (memory saving)
4. **Real-time access**: Callback mechanism for live orbital analysis

## Files Modified

### Core Implementation
- `src/dftbp/dftbplus/hamiltonian_store.F90` - Added eigenvector storage
- `src/dftbp/dftbplus/main.F90` - Added storage call after diagonalization
- `app/dftbcore/libdftbcore.F90` - Updated retrieval logic

### Test and Documentation
- `tests/grid/test_waveplot_dftbcore.py` - New comprehensive test script
- `doc/prokop/waveplot_library.md` - This documentation

---