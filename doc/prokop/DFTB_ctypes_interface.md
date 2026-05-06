
CODEMAP: DFTB+ Calculation Flow: Standalone Program, C API, and Python Wrapper
https://windsurf.com/codemaps/2c157118-9d28-4a7c-a234-a49a3d464424-fe86ab10a43f3d18

# USER

. Do you think we can make our own interface similat to this we made for fireball @libFireCore.f90 @FireCore.py or for waveplot
@libwaveplot.F90 @WavePlot.py 

think what is the minimum ammout of function we need to make to be able to run DFT initialize it, solve the SCF and export the density matrix. 

do not program anythin yet, just make notes

---

# SWE-1.6

Looking at the Fireball and WavePlot interfaces, here are my notes for a minimal DFTB+ interface:

## Pattern Analysis

**Fireball approach:**
- State module (`firecore_options`, internal modules) holds persistent data
- C-bindable subroutines with simple signatures (arrays passed as pointers)
- Python ctypes wrapper loads library and sets argtypes/restype
- Functions: [preinit()](cci:1://file:///home/prokop/git/dftbplus/pyBall/Fireball/FireCore.py:249:0-250:33) → [init()](cci:1://file:///home/prokop/git/dftbplus/pyBall/Fireball/FireCore.py:146:0-153:57) → `set_*()` → [SCF()](cci:1://file:///home/prokop/git/dftbplus/pyBall/Fireball/FireCore.py:302:0-303:58) → `get_*()`

**WavePlot approach:**
- State module (`libwaveplot_state`) holds geometry, basis, eigenvectors
- C-bindable interface with `bind(c, name='...')`
- Python wrapper handles Fortran/C array ordering (transpose to Fortran order)

## Minimal DFTB+ Interface Functions

### Core Requirements
1. **Initialize** - Create DFTB+ instance, load input
2. **Set geometry** - Atomic positions and species
3. **Run SCF** - Self-consistent field calculation
4. **Get density matrix** - Extract DM (dense or sparse)

### Proposed Fortran Interface (`libdftbcore.F90`)

```fortran
! State module to hold DFTB+ instance
module libdftbcore_state
  use dftbp_dftbplus_main
  type(TDftbPlus), public, save :: dftbInstance
  logical, public, save :: tInitialized = .false.
end module

! C-bindable functions
subroutine dftbcore_init(inputFile, outputFile) bind(c)
  ! Initialize DFTB+ from input file
  character(c_char) :: inputFile(*), outputFile(*)
end subroutine

subroutine dftbcore_set_geometry(natoms, coords, species) bind(c)
  ! Set atomic geometry (if not in input file)
  integer(c_int), value :: natoms
  real(c_double) :: coords(3, natoms)
  integer(c_int) :: species(natoms)
end subroutine

subroutine dftbcore_run_scf(energy) bind(c)
  ! Run SCF calculation
  real(c_double) :: energy  ! output: total energy
end subroutine

subroutine dftbcore_get_dm_size(nOrb) bind(c)
  ! Get basis size for DM allocation
  integer(c_int) :: nOrb  ! output
end subroutine

subroutine dftbcore_get_dm_dense(dm) bind(c)
  ! Get density matrix in dense format
  real(c_double) :: dm(nOrb, nOrb)  ! output
end subroutine

subroutine dftbcore_get_dm_sparse(nelem, i, j, val) bind(c)
  ! Get density matrix in sparse format
  integer(c_int) :: nelem  ! output: number of non-zero elements
  integer(c_int) :: i(nelem), j(nelem)  ! output: indices
  real(c_double) :: val(nelem)  ! output: values
end subroutine

subroutine dftbcore_finalize() bind(c)
  ! Clean up DFTB+ instance
end subroutine
```

### Key Considerations

1. **Simplify vs existing C API**: The existing [dftbplus.h](cci:7://file:///home/prokop/git/dftbplus/src/dftbp/api/mm/dftbplus.h:0:0-0:0) API is complex with callbacks. A simpler wrapper could:
   - Use internal DFTB+ routines directly
   - Store DM in a module variable instead of callback
   - Provide simple get functions after SCF

2. **Dense vs Sparse DM**: 
   - Dense is simpler but memory-intensive for large systems
   - Sparse matches DFTB+ internal storage
   - Fireball uses sparse (`firecore_get_rho_sparse`)

3. **Input handling**:
   - Option A: Load from input file (like waveplot loads from DFTB+ output)
   - Option B: Set everything programmatically (like Fireball)
   - Hybrid: Input file for parameters, programmatic for geometry

4. **Array ordering**: Need to handle Fortran column-major vs Python row-major (transpose in Python wrapper)

5. **Minimal function set**:
   - `dftbcore_init(inputFile)` - or `dftbcore_init_empty()` + setters
   - `dftbcore_set_geometry(coords, species)` - optional if in input
   - `dftbcore_run_scf()` - returns energy
   - `dftbcore_get_dm_*()` - extract density matrix
   - `dftbcore_finalize()` - cleanup

### Alternative: Use Existing C API Directly

Instead of new Fortran wrapper, could fix the Python ctypes wrapper to:
- Add the missing `matrix_descr` structure
- Fix callback signature (6 parameters instead of 5)
- Simplify the callback handling

This would be less code but depends on the complex existing C API.

### Recommendation

For a **minimal, clean interface** similar to Fireball/WavePlot:
- Create new `libdftbcore.F90` with state module and simple C-bindable functions
- Use internal DFTB+ routines (not the MM-API callbacks)
- Store DM in module after SCF, provide simple getter
- Python wrapper similar to WavePlot pattern

This avoids the callback complexity and provides a clean, testable interface.

---

# USER

form where we can see what sequence of function calls is required to run DFTB? Can you find program entry point or example how it uste the C_API library so we know what we should exactly ut into our simplified fotran/ctypes library interface?

---

# SWE-1.6





Perfect! I found the exact sequence. Here are the notes:

## DFTB+ C API Function Call Sequence

### From [test_fileinitc.c](cci:7://file:///home/prokop/git/dftbplus/test/src/dftbp/api/mm/testers/test_fileinitc.c:0:0-0:0) (Basic Usage - No Callbacks)

```c
// 1. Initialize DFTB+ instance
dftbp_init(&calculator, NULL);

// 2. Load input from file
dftbp_get_input_from_file(&calculator, "dftb_in.hsd", &input);

// 3. Process input (parse HSD, initialize internal structures)
dftbp_process_input(&calculator, &input);

// 4. Finalize input (free input tree)
dftbp_input_final(&input);

// 5. Set coordinates (optional - can override from input file)
dftbp_set_coords(&calculator, coords);
// OR for periodic:
dftbp_set_coords_and_lattice_vecs(&calculator, coords, latvecs);

// 6. Run SCF calculation and get energy
dftbp_get_energy(&calculator, &mermin_energy);

// 7. Extract results
dftbp_get_gradients(&calculator, gradients);
dftbp_get_gross_charges(&calculator, charges);
dftbp_get_cm5_charges(&calculator, cm5_charges);
dftbp_get_stress_tensor(&calculator, stress_tensor);

// 8. Cleanup
dftbp_final(&calculator);
```

### From [test_expdmhs.c](cci:7://file:///home/prokop/git/dftbplus/test/src/dftbp/api/mm/testers/test_expdmhs.c:0:0-0:0) (With Callbacks for DM/HS Export)

```c
// 1-4. Same as above (init, load input, process, finalize input)
dftbp_init(&calculator, NULL);
dftbp_get_input_from_file(&calculator, "dftb_in.hsd", &input);
dftbp_process_input(&calculator, &input);
dftbp_input_final(&input);

// 5. Set geometry
dftbp_set_coords_and_lattice_vecs(&calculator, coords, latvecs);

// 6. Register callbacks BEFORE calculation
dftbp_register_dm_callback(&calculator, dm_callback, 0);
dftbp_register_s_callback(&calculator, s_callback, 0);
dftbp_register_h_callback(&calculator, h_callback, 0);

// 7. Run calculation - callbacks invoked during SCF
dftbp_get_energy(&calculator, &mermin_energy);

// 8. Cleanup
dftbp_final(&calculator);
```

### Callback Signature (Critical - This was the bug in [dftb_lib.py](cci:7://file:///home/prokop/git/dftbplus/pyBall/dftb_lib.py:0:0-0:0))

```c
void dm_callback(void *aux_ptr, int iK, int iS, int *blacs_descr,
                 const void *blacs_data, DftbPlusMatrixDescr *matrix_descr) {
  // iK: k-point index (1-based)
  // iS: spin index (1-based)
  // blacs_descr: BLACS descriptor (NULL if no SCALAPACK)
  // blacs_data: pointer to matrix data (real or complex)
  // matrix_descr: descriptor with storage_type (TRIL, FULL, etc.)
}
```

### From Fortran Main Program ([app/dftb+/dftbplus.F90](cci:7://file:///home/prokop/git/dftbplus/app/dftb+/dftbplus.F90:0:0-0:0))

```fortran
call initGlobalEnv()
call parseHsdInput(input)
call TEnvironment_init(env)
allocate(main)
call main%initProgramVariables(input, env)
deallocate(input)
call runDftbPlus(main, env)  // This runs the full calculation
call main%destructProgramVariables()
deallocate(main)
call env%destruct()
call destructGlobalEnv()
```

## Minimal Simplified Interface Design

Based on this, the minimal functions needed for a simplified Fortran/ctypes interface:

### Fortran (`libdftbcore.F90`)

```fortran
subroutine dftbcore_init(inputFile, outputFile) bind(c)
  ! Initialize DFTB+ instance
end subroutine

subroutine dftbcore_process_input() bind(c)
  ! Process input (after init)
end subroutine

subroutine dftbcore_set_geometry(natoms, coords, species) bind(c)
  ! Set atomic geometry (optional override)
end subroutine

subroutine dftbcore_register_dm_callback(callback_ptr) bind(c)
  ! Register callback for density matrix export
end subroutine

subroutine dftbcore_run_scf(energy) bind(c)
  ! Run SCF calculation
end subroutine

subroutine dftbcore_finalize() bind(c)
  ! Cleanup
end subroutine
```

### Key Observations

1. **Input file is required** - The C API expects an HSD input file with all parameters (Hamiltonian, SCC, Slater-Koster files, etc.)
2. **Callbacks must be registered BEFORE `dftbp_get_energy`** - The calculation triggers the callbacks
3. **The missing `matrix_descr` parameter** was the bug - it's needed to know storage format (TRIL vs FULL)
4. **Geometry can be overridden** after loading input via `dftbp_set_coords*` functions
5. **No direct "get DM" function** - DM is only available via callback during calculation

### Alternative: Simplify Further

If we want to avoid callbacks entirely (like Fireball), we could:
1. Store DM in a module variable after SCF
2. Provide a simple getter function
3. This requires modifying internal DFTB+ code to expose the DM directly

The callback approach is cleaner but requires fixing the Python wrapper signature.

---

# IMPLEMENTATION: Matrix Extraction Interface

## Overview

This section documents the complete implementation of a DFTB+ matrix extraction interface that provides access to Hamiltonian (H), Overlap (S), and Density Matrix (DM) from Python via ctypes. The implementation follows the FireCore pattern and successfully achieves parity between the library and executable results.

## Architecture

```
Python (ctypes) → Fortran C-bindable interface → DFTB+ internal matrices
     ↓                    ↓                           ↓
DFTBcore.py         libdftbcore.F90          main.F90 + hamiltonian_store.F90
```

## Files Added/Modified

### 1. New Fortran Module: `src/dftbp/dftbplus/hamiltonian_store.F90`

**Purpose**: Stores dense H, S, DM matrices at well-defined points inside the SCF loop where they are valid and fully populated.

**Key Features**:
- Uses `save` variables to persist matrices across subroutine calls
- Handles lower-triangle filling from `unpackHS`/`herk` by explicit mirroring
- Provides clean getter functions for the library interface

```fortran
module dftbp_dftbplus_hamiltonian_store
  real(dp), allocatable, save :: storedH(:,:), storedS(:,:), storedDM(:,:)
  logical,  save :: tStoreMatrices = .false.
  integer,  save :: storedSize = 0
end module
```

**Critical Functions**:
- `store_hamiltonian(H, sizeH)` - stores H before diagonalization
- `store_overlap(S, sizeS)` - stores S before diagonalization overwrites it  
- `store_dm(DM, sizeDM)` - stores DM from `getDensityMatrix` output
- `get_stored_*()` - retrieve matrices for library interface
- `clear_stored_matrices()` - cleanup

**Important Implementation Detail**:
```fortran
! Explicit lower-triangle mirroring (unpackHS/herk only fill lower triangle)
do j = 1, sizeH
  do i = j, sizeH
    storedH(i,j) = H(i,j)
    storedH(j,i) = H(i,j)  ! Mirror to upper triangle
  end do
end do
```

### 2. Modified Core File: `src/dftbp/dftbplus/main.F90`

**Purpose**: Added calls to store matrices at the exact points where they are valid.

**Changes Made**:

#### Import Addition (line ~36):
```fortran
use dftbp_dftbplus_hamiltonian_store, only : store_hamiltonian, store_overlap, store_dm
```

#### Matrix Storage in `buildAndDiagDenseRealHam` (lines ~3439-3441):
```fortran
! Store H and S before diagonalization (SSqrReal gets overwritten below)
call store_hamiltonian(HSqrReal, size(HSqrReal, 1))
call store_overlap(SSqrReal, size(SSqrReal, 1))
```

**Location**: Immediately before `diagDenseMtx` which overwrites `SSqrReal` with eigenvectors.

#### DM Storage in `getDensityFromRealEigvecs` (lines ~3984-3985):
```fortran
! Store dense DM at the point it is valid (before packing to sparse)
call store_dm(work, size(work, 1))
```

**Location**: Immediately after `densityMatrix%getDensityMatrix(work, eigvecs, filling)` fills `work` with the converged density matrix.

**Why These Locations?**:
- H and S are valid right after `unpackHS` from sparse to dense, but before diagonalization
- DM is valid right after `getDensityMatrix` computes it from eigenvectors and occupations
- Storing at these points avoids the complexity of extracting from sparse formats later

### 3. Library Interface: `app/dftbcore/libdftbcore.F90`

**Purpose**: C-bindable interface that exposes DFTB+ functionality to Python ctypes.

**Key Functions**:
```fortran
subroutine dftbcore_init(inputFile, outputFile) bind(c)
subroutine dftbcore_enable_hamiltonian_storage(store) bind(c)
subroutine dftbcore_enable_matrix_collection(dm, h, s) bind(c)
subroutine dftbcore_run_scf(energy, ierr) bind(c)
subroutine dftbcore_get_h_dense(h, n) bind(c)
subroutine dftbcore_get_s_dense(s, n) bind(c)
subroutine dftbcore_get_dm_dense(dm, n) bind(c)
subroutine dftbcore_get_eigvecs_dense(eigvecs, eigvals, n) bind(c)
subroutine dftbcore_finalize() bind(c)
```

**Interface Pattern (FireCore-style)**:
- Explicit-size arrays with `n` passed by `value`
- Flat buffers (`(*)`) for ctypes compatibility
- Error handling via output parameters

### 4. Python Wrapper: `pyBall/DFTBcore.py`

**Purpose**: Python ctypes wrapper that provides clean Python API.

**Key Methods**:
```python
def init(self, input_file, output_file=None)
def enable_hamiltonian_storage(self, store=True)
def enable_matrix_collection(self, dm=True, h=True, s=True)
def run_scf(self)
def get_h_dense(self)
def get_s_dense(self)
def get_dm_dense(self)
def get_eigvecs_dense(self)
def finalize(self)
```

**Array Handling**:
- Uses flat buffers for ctypes calls
- Converts Fortran column-major to Python row-major
- Helper `_get_matrix()` for consistent interface

### 5. Test Suite: `tests/dftb/test_parity.py`

**Purpose**: Comprehensive parity test between library and executable.

**Test Flow**:
1. Run DFTB+ executable first (avoid shared library conflicts)
2. Initialize library with same input
3. Enable matrix storage and collection
4. Run SCF via library
5. Extract H, S, DM, eigenvectors, eigenvalues
6. Compare with executable results
7. Verify physical properties (electron count, symmetry)

**Key Checks**:
- Energy parity (exact match)
- Eigenvalue parity (< 0.1 meV difference)
- Matrix symmetry (all matrices symmetric)
- Electron count: `Tr(S·DM) = 8.0` for H₂O
- Overlap normalization: diagonal elements ≈ 1.0

## Integration Points

### Build System
- Added `hamiltonian_store.F90` to `src/dftbp/dftbplus/CMakeLists.txt`
- Library target: `app/dftbcore` builds `libdftbcore.so`

### Data Flow
```
Sparse matrices (ints%hamiltonian, ints%overlap) 
    ↓ unpackHS (lower triangle only)
Dense H, S (HSqrReal, SSqrReal)
    ↓ store_hamiltonian/store_overlap
hamiltonian_store module
    ↓ get_stored_*
libdftbcore.F90 (C interface)
    ↓ ctypes
DFTBcore.py (Python)
```

For DM:
```
Eigenvectors + occupations
    ↓ getDensityMatrix (herk - lower triangle only)
Dense DM (work array)
    ↓ store_dm (explicit mirroring)
hamiltonian_store module
    ↓ get_stored_dm
libdftbcore.F90
    ↓ ctypes
DFTBcore.py
```

## Critical Technical Details

### Lower-Triangle Issue
The core technical challenge was that `unpackHS` and BLAS `herk` only fill the **lower triangle** of matrices. The upper triangle contains garbage values. This caused:
- Asymmetric matrices when using naive symmetrization
- Wrong electron counts (`Tr(S·DM) ≠ expected`)

**Solution**: Explicit mirroring in storage routines:
```fortran
do j = 1, n
  do i = j, n
    stored(i,j) = input(i,j)
    stored(j,i) = input(i,j)  ! Mirror
  end do
end do
```

### Timing of Storage
Storing matrices at the wrong time leads to:
- H containing eigenvectors instead of Hamiltonian
- S being overwritten by diagonalization
- DM being from intermediate SCC iteration

**Solution**: Store at exact points where matrices are valid:
- H/S: after `unpackHS`, before `diagDenseMtx`
- DM: after `getDensityMatrix`, before `packHS`

### Library vs Executable Conflicts
Running both executable and library in same process causes segfaults due to:
- Shared Fortran global state
- Memory allocation conflicts

**Solution**: Run executable first in subprocess, then library.

## Verification Results

For H₂O test case (3ob-3-1 Slater-Koster):
```
[PASS] Energy: lib=-4.05893301  exe=-4.05893301  diff=0.00e+00
[PASS] Max eigenvalue diff: 6.1650e-05 eV
[PASS] H symmetry: max|lib-exe| = 0.000e+00
[PASS] S symmetry: max|lib-exe| = 0.000e+00  
[PASS] DM symmetry: max|lib-exe| = 0.000e+00
[PASS] Tr(S*DM) = 8.0000  (expected 8.0)
OVERALL: ALL CHECKS PASSED
```

## New Eigenvector Export Functions (2025-05-06)

### Overview

Added functionality to export molecular orbital coefficients directly from the DFTB+ library without requiring `eigenvec.bin` files. This enables seamless integration with OpenCL orbital projection and other post-processing tools.

### New Functions in hamiltonian_store.F90

#### Module Variables
```fortran
real(dp), allocatable, save :: storedEigvecs(:,:)   ! (norb, nstates) for iKS=1, iSpin=1
real(dp), allocatable, save :: storedEigenvals(:)   ! (nstates) for iKS=1, iSpin=1
```

#### store_eigvecs
```fortran
subroutine store_eigvecs(eigvecs_in, eigenvals_in, norb)
  real(dp), intent(in) :: eigvecs_in(:,:)
  real(dp), intent(in) :: eigenvals_in(:)
  integer,  intent(in) :: norb
```
**Purpose**: Stores eigenvectors and eigenvalues after diagonalization
**Called from**: `main.F90` after `diagDenseMtx` completes
**Stores**: 
- `eigvecs_in(1:norb, 1:norb)` - eigenvector matrix (columns are MOs)
- `eigenvals_in(1:norb)` - eigenvalue array
- Updates `storedSize = norb`

#### get_stored_eigvecs
```fortran
subroutine get_stored_eigvecs(eigvecs_out, eigenvals_out, norb)
  real(dp), intent(out) :: eigvecs_out(:,:)
  real(dp), intent(out) :: eigenvals_out(:)
  integer,  intent(out) :: norb
```
**Purpose**: Retrieves stored eigenvectors and eigenvalues
**Returns**: 
- `eigvecs_out` - eigenvector matrix
- `eigenvals_out` - eigenvalue array  
- `norb` - matrix dimension (0 if not stored)

#### Updated Functions
- `clear_stored_matrices()` - Now also clears `storedEigvecs` and `storedEigenvals`
- `set_store_hamiltonian()` - Controls eigenvector storage via `tStoreMatrices` flag

### Integration Points in main.F90

#### Use Statement Addition
```fortran
use dftbp_dftbplus_hamiltonian_store, only : store_hamiltonian, store_overlap, store_dm, store_eigvecs
```

#### Storage Call Location
**File**: `src/dftbp/dftbplus/main.F90`  
**Subroutine**: `buildAndDiagDenseRealHam`  
**Line**: 3449 (after diagonalization)

```fortran
eigvecsReal(:,:,iKS) = HSqrReal
! Store eigenvectors for iKS=1 only (gamma-point / first spin channel)
if (iKS == 1) call store_eigvecs(HSqrReal, eigen(:, iSpin), size(HSqrReal, 1))
```

**Timing**: Captures eigenvectors at optimal moment:
- After `diagDenseMtx` completes
- Before `HSqrReal` is reused
- For first k-point only (iKS=1)

**Data Captured**:
- `HSqrReal` - Contains eigenvectors (overwrites original H)
- `eigen(:, iSpin)` - Eigenvalues for current spin
- `size(HSqrReal, 1)` - Number of basis functions

### libdftbcore.F90 Updates

#### New Import
```fortran
use dftbp_dftbplus_hamiltonian_store, only : set_store_hamiltonian, get_stored_hamiltonian,&
    & get_stored_overlap, get_stored_dm, get_stored_eigvecs, clear_stored_matrices
```

#### Automatic Storage Enablement
**Location**: `dftbcore_init` subroutine
```fortran
! Always enable storage so store_eigvecs (and optional H/S/DM) can capture data during SCF
call set_store_hamiltonian(.true.)
```

#### Eigenvector Extraction
**Location**: `dftbcore_run_scf` subroutine
```fortran
! Always extract eigenvectors (stored in hamiltonian_store during SCF via store_eigvecs)
call get_stored_eigvecs(storedEigvecs, storedEigenvals, iSpin)
```

**Behavior**:
- Unconditional extraction (not dependent on H/S/DM flags)
- Returns zero matrices if not stored
- Provides debug output about availability

### Python Interface

The existing `DFTBcore.get_eigvecs_dense()` method now retrieves from storage:

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

### Data Format

#### Matrix Structure
- **Fortran storage**: `storedEigvecs(norb, nstates)` column-major
- **Python output**: `(nstates, norb)` row-major after transpose
- **Compatibility**: Identical to `eigenvec.bin` format (after transpose)

#### Storage Scope
Currently stores:
- iKS=1 (first k-point, typically gamma-point)
- First spin channel (non-spin-polarized or spin-up)
- Full eigenvector matrix for all states

### Usage Example

```python
from pyBall.DFTBcore import DFTBcore

# Initialize and run calculation
dftb = DFTBcore()
dftb.init('h2o.hsd')
dftb.enable_matrix_collection(dm=False, h=False, s=False)  # Optional
energy = dftb.run_scf()

# Extract eigenvectors directly (no .bin file needed)
evecs, eigenvals = dftb.get_eigvecs_dense()

print(f"Energy: {energy:.8f} Ha")
print(f"Eigenvectors shape: {evecs.shape}")  # (nstates, norb)
print(f"Eigenvalues shape: {eigenvals.shape}")  # (nstates,)

dftb.finalize()
```

### Verification Results

Tested with H₂O and PTCDA systems:

| System | Energy (Ha) | max|lib - bin| | max|ψ| |
|--------|-------------|--------------|---------|
| H₂O | -4.076143 | 2.78e-17 | 0.215 |
| PTCDA | -64.340460 | 5.55e-17 | 1.54e-3 |

Machine-precision agreement with traditional `eigenvec.bin` approach.

### Integration Notes

#### Minimal Impact
- Uses existing `hamiltonian_store` infrastructure
- No changes to core DFTB+ algorithms
- Backward compatible with existing functionality

#### Storage Control
- Controlled by `tStoreMatrices` flag
- Automatically enabled in `dftbcore_init`
- Can be disabled if not needed

#### Memory Usage
- Stores one additional `(norb, norb)` matrix
- For typical systems: < 1MB for small molecules, few MB for large ones
- Much less than full matrix dumps

### Future Extensions

Potential enhancements:
1. **Multiple k-points**: Store for all iKS values
2. **Spin channels**: Store both spin-up and spin-down
3. **Selective storage**: Option to store only specific MOs
4. **Real-time access**: Callback mechanism during SCF

### Files Modified

- `src/dftbp/dftbplus/hamiltonian_store.F90` - Added eigenvector storage
- `src/dftbp/dftbplus/main.F90` - Added storage call after diagonalization  
- `app/dftbcore/libdftbcore.F90` - Updated retrieval logic
- `tests/grid/test_waveplot_dftbcore.py` - New test script

## Future Considerations

### Upstream Integration
These modifications could potentially be upstreamed to DFTB+:
1. **hamiltonian_store.F90** - Generic matrix storage utility
2. **Storage calls in main.F90** - Optional via compile flag
3. **libdftbcore.F90** - Additional API layer

### Potential Conflicts
When updating from upstream DFTB+ repository:
- Check for changes in `buildAndDiagDenseRealHam` around H/S storage calls
- Check for changes in `getDensityFromRealEigvecs` around DM storage call  
- Verify `hamiltonian_store.F90` module name doesn't conflict
- Ensure `use` statements still match available functions

### Alternative Approaches
- Use existing C API with callbacks (more complex, no code changes)
- Store matrices in existing object members instead of separate module
- Add compile-time flags to enable/disable storage functionality

## Usage Example

```python
from pyBall.DFTBcore import DFTBcore
import numpy as np

# Initialize and run calculation
dftb = DFTBcore()
dftb.init('h2o.hsd')
dftb.enable_hamiltonian_storage(store=True)
dftb.enable_matrix_collection(dm=True, h=True, s=True)
energy = dftb.run_scf()

# Extract matrices
H = dftb.get_h_dense()
S = dftb.get_s_dense()  
DM = dftb.get_dm_dense()
C, eigvals = dftb.get_eigvecs_dense()

# Verify properties
print(f"Energy: {energy:.8f} Ha")
print(f"Electron count: {np.trace(S @ DM):.4f}")
print(f"Eigenvalues (eV): {eigvals * 27.2114}")

dftb.finalize()
```

This implementation provides a clean, robust interface for extracting DFTB+ matrices while maintaining full parity with the executable results.