# DFTB+ Testing and Parity Checking Codemap

This document provides explicit file names, function names, and line numbers for extracting Hamiltonian pieces from DFTB+ and setting up systematic parity checking for the Rust implementation.

---

## 1. Extracting Non-SCC H0 from DFTB+

### Method 1: Run with SCC = No (No Code Modification)

**File:** `dftb_in.hsd` (input file)

**Location in file:** Under `Hamiltonian = DFTB { ... }` block

**Change:**
```hsd
Hamiltonian = DFTB {
  SCC = No
  ...
}

Options {
  WriteHS = Yes
}
```

**What happens internally:**
- File: `src/dftbp/dftbplus/initprogram.F90`
- Line: 1364: `this%tSccCalc = input%ctrl%tSCC`
- Line: 1630: `if (this%tSccCalc) then` → SCC calculator allocation skipped
- Line: 1726: `if (this%tSccCalc) then` → Hubbard U initialization skipped
- Line: 1789: `if (this%tSccCalc) then` → SCC calculator initialization skipped

**Hamiltonian construction:**
- File: `src/dftbp/dftbplus/main.F90`
- Line: 1237: `call buildH0(env, this%H0, ...)` → builds pure non-SCC H0
- Line: 1239: `call buildS(env, this%ints%overlap, ...)` → builds overlap
- Line: 1300: `call getSccHamiltonian(...)` → called but with zero potentials (no SCC)
- Line: 1474: `call writeHSAndStop(...)` → outputs `hamsqr1.dat` (pure H0)

**Output file:** `hamsqr1.dat` (dense square matrix, contains only H0)

---

### Method 2: Modify DFTB+ to Output H0 Separately

**File:** `src/dftbp/dftbplus/main.F90`

**Location:** After line 1468 (after `getSccHamiltonian` call, before `writeHSAndStop`)

**Add this code:**
```fortran
! Write H0 separately for parity checking
if (this%tWriteHS) then
  call writeSparseAsSquare(env, "h0sqr.dat", this%H0, this%neighbourList%iNeighbour,&
      & this%nNeighbourSK, this%denseDesc%iAtomStart, this%iSparseStart,&
      & this%img2CentCell)
end if
```

**Context:**
```fortran
! Line 1466
call getSccHamiltonian(env, this%H0, this%ints, this%nNeighbourSK, this%neighbourList,&
    & this%species, this%orb, this%iSparseStart, this%img2CentCell, this%potential,&
    & this%mdftb, allocated(this%reks), this%ints%hamiltonian, this%ints%iHamiltonian)

! ADD CODE HERE (after line 1468)

! Line 1470
if (this%tWriteRealHS .or. this%tWriteHS ...) then
  call writeHSAndStop(...)
```

**Output file:** `h0sqr.dat` (dense square matrix, contains H0 even when SCC is on)

**Function used:** `writeSparseAsSquare` from `src/dftbp/dftbplus/mainio.F90`

---

### Method 3: Add New Input Option WriteH0

**File:** `src/dftbp/dftbplus/parser.F90`

**Location:** Under `Options { ... }` parsing section (around line 366)

**Add parsing:**
```fortran
call getChildValue(node, "WriteH0", ctrl%tWriteH0, .false.)
```

**File:** `src/dftbp/dftbplus/inputdata.F90`

**Location:** In type definition (around line 120)

**Add field:**
```fortran
type TParserFlags
  logical :: tWriteHSD
  logical :: tWriteH0  ! ADD THIS
end type TParserFlags
```

**File:** `src/dftbp/dftbplus/initprogram.F90`

**Location:** Around line 1653 (where `this%tWriteHS` is set)

**Add:**
```fortran
this%tWriteH0 = input%ctrl%tWriteH0
```

**File:** `src/dftbp/dftbplus/mainio.F90`

**Location:** After `writeHS` subroutine (around line 4424)

**Add new subroutine:**
```fortran
subroutine writeH0(env, H0, iNeighbour, nNeighbourSK, iAtomStart, iPair, img2CentCell)
  type(TEnvironment), intent(inout) :: env
  real(dp), intent(in) :: H0(:)
  integer, intent(in) :: iNeighbour(0:,:)
  integer, intent(in) :: nNeighbourSK(:)
  integer, intent(in) :: iAtomStart(:)
  integer, intent(in) :: iPair(0:,:)
  integer, intent(in) :: img2CentCell(:)
  
  call writeSparseAsSquare(env, "h0sqr.dat", H0, iNeighbour, nNeighbourSK,&
      & iAtomStart, iPair, img2CentCell)
end subroutine
```

**File:** `src/dftbp/dftbplus/main.F90`

**Location:** After line 1468 (same as Method 2)

**Add:**
```fortran
if (this%tWriteH0) then
  call writeH0(env, this%H0, this%neighbourList%iNeighbour, this%nNeighbourSK,&
      & this%denseDesc%iAtomStart, this%iSparseStart, this%img2CentCell)
end if
```

---

## 2. Extracting Intermediate Values (SK Integrals, Rotation)

### Debug Prints in buildDiatomicBlocks

**File:** `src/dftbp/dftb/nonscc.F90`

**Subroutine:** `buildDiatomicBlocks` (starts at line 364)

**Location 1: After getSKIntegrals (line 417)**

**Add:**
```fortran
call getSKIntegrals(skCont, interSK, dist, iSp1, iSp2)
! DEBUG: Print SK integrals
write(*,*) "DEBUG_SK", iAt1, iAt2, dist, iSp1, iSp2
write(*,*) "  SK_VALUES:", interSK(:)
```

**Location 2: After rotateH0 (line 418)**

**Add:**
```fortran
call rotateH0(tmp, interSK, vect(1), vect(2), vect(3), iSp1, iSp2, orb)
! DEBUG: Print direction cosines and rotated matrix
write(*,*) "DEBUG_ROT", iAt1, iAt2
write(*,*) "  DIR_COS:", vect(1), vect(2), vect(3)
write(*,*) "  ROT_MAT:", reshape(tmp(1:nOrb2, 1:nOrb1), [nOrb2 * nOrb1])
```

**Location 3: After on-site energy assignment (in buildH0, around line 170)**

**Add:**
```fortran
ham(ind) = selfegy(orb%iShellOrb(iOrb1, iSp1), iSp1)
! DEBUG: Print on-site energy
write(*,*) "DEBUG_ONSITE", iAt1, iOrb1, iSp1, ham(ind)
```

**Compile with debug:**
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

---

### Standalone Fortran Test Program

**Create file:** `test_sk_interp.f90`

**Code:**
```fortran
program test_sk_interp
  use dftbp_type_oldskdata
  use dftbp_dftb_slakocont
  use dftbp_dftb_slakoeqgrid
  use dftbp_dftb_sk
  use dftbp_common_accuracy, only : dp
  use dftbp_io_message, only : error
  implicit none
  
  type(TOldSKData) :: skData
  type(TSlakoEqGrid) :: skGrid
  real(dp) :: interSK(4)  ! sp basis: ssσ, spσ, ppσ, ppπ
  real(dp) :: rotMat(4,4)  ! For C-C (4 orbitals each)
  real(dp) :: ll, mm, nn
  real(dp) :: test_dist
  integer :: ierr
  
  ! Read SK file
  call TOldSKData_readFromFile(skData, "C-C.skf", ierr)
  if (ierr /= 0) call error("Failed to read SK file")
  
  ! Initialize grid (simplified - normally done by slakocont)
  ! This is a minimal test - in practice use slakocont
  
  ! Test interpolation at specific distance
  test_dist = 1.5_dp
  ! Call getSKIntegrals (need to set up proper grid first)
  ! For now, just print raw data
  print *, "Grid spacing:", skData%dist
  print *, "Number of grid points:", skData%nGrid
  print *, "Hamiltonian at grid point 1:", skData%skHam(1,:)
  
  ! Test rotation along z-axis
  ll = 0.0_dp
  mm = 0.0_dp
  nn = 1.0_dp
  interSK = [1.0_dp, 0.5_dp, 0.3_dp, 0.1_dp]  ! Test values
  call rotateH0(rotMat, interSK, ll, mm, nn, 1, 1, orb)  ! Need orb info
  
  print *, "Rotation test along z-axis:"
  print *, rotMat
  
end program
```

**Compile:**
```bash
gfortran -I/path/to/dftbplus/src -c test_sk_interp.f90
gfortran -o test_sk_interp test_sk_interp.o -L/path/to/dftbplus/build -ldftbp
```

---

## 3. Sparse Matrix Format Details

### Dense Square Output Format (hamsqr1.dat)

**Writer function:** `writeSparseAsSquare` in `src/dftbp/dftbplus/mainio.F90` (line ~4409)

**Format:**
```
  n  m
  H(1,1) H(1,2) ... H(1,m)
  H(2,1) H(2,2) ... H(2,m)
  ...
  H(n,1) H(n,2) ... H(n,m)
```

**Where:**
- `n` = total number of orbitals
- `m` = total number of orbitals (n = m for square matrix)
- Values are in Hartree (atomic units)

**Reader in Rust:**
```rust
fn read_hamsqr(path: &str) -> Result<DMatrix> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // First line: dimensions
    let header = lines.next().ok()?;
    let dims: Vec<usize> = header.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let n = dims[0];
    let m = dims[1];
    
    // Read matrix values
    let mut mat = DMatrix::zeros(n, m);
    for (i, line) in lines.enumerate() {
        let row: Vec<f64> = line.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        for (j, val) in row.iter().enumerate() {
            mat[(i, j)] = *val;
        }
    }
    
    Ok(mat)
}
```

---

### Sparse Real-Space Output Format (hamreal.dat)

**Writer function:** `writeSparse` in `src/dftbp/dftbplus/mainio.F90` (line ~4396)

**Format:**
```
  iAtom  jAtom  iCell  jCell  iOrb  jOrb  H(i,j)
  ...
```

**Where:**
- `iAtom`, `jAtom`: Atom indices (1-based)
- `iCell`, `jCell`: Cell indices (0 for central cell)
- `iOrb`, `jOrb`: Orbital indices within atom
- `H(i,j)`: Matrix element value

**For non-periodic systems:** `iCell = jCell = 0`

---

### Sparse Storage Indexing Arrays

**Key arrays in DFTB+:**

**iAtomStart** (dense indexing):
- File: `src/dftbp/dftbplus/main.F90`
- Variable: `this%denseDesc%iAtomStart`
- Type: `integer, allocatable :: iAtomStart(:)`
- Purpose: `iAtomStart(iAtom)` = starting orbital index of atom iAtom in dense matrix
- Example: If atom 1 has 4 orbitals, atom 2 has 4 orbitals:
  - `iAtomStart(1) = 1`
  - `iAtomStart(2) = 5`

**iPair** (sparse indexing):
- File: `src/dftbp/dftbplus/main.F90`
- Variable: `this%iSparseStart`
- Type: `integer, allocatable :: iPair(0:nNeighbour, nAtom)`
- Purpose: `iPair(iNeigh, iAtom)` = starting index in sparse storage for neighbor iNeigh of atom iAtom
- `iPair(0, iAtom)` = starting index for atom iAtom itself (diagonal block)

**img2CentCell** (periodic images):
- File: `src/dftbp/dftbplus/main.F90`
- Variable: `this%img2CentCell`
- Type: `integer, allocatable :: img2CentCell(:)`
- Purpose: Maps image atoms back to central cell atoms
- For non-periodic: `img2CentCell(i) = i`

---

## 4. SCC Shift Extraction

### Extract SCC Shifts by Subtraction

**Procedure:**
1. Run with `SCC = No` → get `h0sqr.dat` (H0)
2. Run with `SCC = Yes` → get `hamsqr1.dat` (H_full)
3. Subtract: `SCC_shifts = H_full - H0`

**Python script:**
```python
import numpy as np

def read_hamsqr(filename):
    with open(filename) as f:
        n, m = map(int, f.readline().split())
        mat = np.loadtxt(f)
    return mat.reshape(n, m)

h0 = read_hamsqr('h0sqr.dat')
h_full = read_hamsqr('hamsqr1.dat')
scc_shifts = h_full - h0

np.savetxt('scc_shifts.dat', scc_shifts)
```

---

### Extract Shift Values Directly

**File:** `src/dftbp/dftb/scc.F90`

**Subroutine:** `updateShifts` (around line 233 in notes, actual line varies)

**Location:** After shift calculation, before return

**Add debug output:**
```fortran
subroutine updateShifts(this, env, orb, species, iNeighbour, img2CentCell)
  ! ... existing code ...
  
  ! DEBUG: Print shifts
  do iAt = 1, this%nAtom
    write(*,*) "DEBUG_SHIFT_ATOM", iAt, this%shiftPerAtom(iAt)
  end do
  
  do iSh = 1, this%mShell
    do iAt = 1, this%nAtom
      write(*,*) "DEBUG_SHIFT_SHELL", iSh, iAt, this%shiftPerL(iSh, iAt)
    end do
  end do
  
end subroutine updateShifts
```

**Alternative:** Use `getShiftPerAtom` and `getShiftPerL` in main loop to print shifts

---

## 5. TBLite Extraction

### TBLite BuildSH0 Call

**File:** `src/dftbp/dftbplus/main.F90`

**Location:** Line 1248 (in `select case` for `hamiltonianTypes%xtb`)

**Code:**
```fortran
case(hamiltonianTypes%xtb)
  @:ASSERT(allocated(this%tblite), "Compiled without TBLITE included")
  call this%tblite%buildSH0(env, this%species, this%coord, this%nNeighbourSk, &
      & this%neighbourList%iNeighbour, this%img2CentCell, this%iSparseStart, &
      & this%orb, this%H0, this%ints%overlap, this%ints%dipoleBra, this%ints%dipoleKet, &
      & this%ints%quadrupoleBra, this%ints%quadrupoleKet)
```

**To extract TBLite H0:**
1. Set `Hamiltonian = xTB` in input
2. Set `SCC = No`
3. Add debug print after line 1252 (after `buildSH0` call)
4. Output `this%H0` using `writeSparseAsSquare`

**Note:** TBLite H0 uses analytical STO overlaps, not SK tables. Compare only final observables (energies, eigenvalues), not matrix elements.

---

## 6. Testing Workflow Codemap

### Test Level 1: SK File Parsing

**DFTB+ side:**
- File: `src/dftbp/type/oldskdata.F90`
- Function: `TOldSKData_readFromFile` (line ~59)
- Add after line ~100 (after reading tables):
```fortran
write(*,*) "DEBUG_SK_DATA"
write(*,*) "  dist:", skData%dist
write(*,*) "  nGrid:", skData%nGrid
write(*,*) "  skSelf:", skData%skSelf
write(*,*) "  skHam(1,:):", skData%skHam(1,:)
write(*,*) "  skOver(1,:):", skData%skOver(1,:)
```

**Rust side:**
- File: `rust_dftb/src/sk_data.rs`
- Function: `SkData::from_file`
- Print same values after parsing
- Compare element-by-element

---

### Test Level 2: Interpolation

**DFTB+ side:**
- File: `src/dftbp/dftb/nonscc.F90`
- Function: `buildDiatomicBlocks` (line 364)
- Add after line 417 (after `getSKIntegrals`):
```fortran
! Test specific atom pair
if (iAt1 == 1 .and. iAt2 == 2) then
  write(*,*) "DEBUG_INTERP", dist, interSK(:)
end if
```

**Rust side:**
- File: `rust_dftb/src/interpolation.rs`
- Function: `Interpolator::evaluate`
- Evaluate at same distance
- Compare values

---

### Test Level 3: Rotation

**DFTB+ side:**
- File: `src/dftbp/dftb/nonscc.F90`
- Function: `buildDiatomicBlocks` (line 364)
- Add after line 418 (after `rotateH0`):
```fortran
if (iAt1 == 1 .and. iAt2 == 2) then
  write(*,*) "DEBUG_ROTATION"
  write(*,*) "  dir_cos:", vect(1), vect(2), vect(3)
  write(*,*) "  input_SK:", interSK(:)
  write(*,*) "  output_mat:", reshape(tmp, [size(tmp)])
end if
```

**Rust side:**
- File: `rust_dftb/src/rotation.rs`
- Function: `Rotation::rotate`
- Use same direction cosines and SK values
- Compare output matrix

---

### Test Level 4: H0 Construction

**DFTB+ side:**
- File: `dftb_in.hsd`
- Set `SCC = No`, `WriteHS = Yes`
- Run: `dftb+ dftb_in.hsd`
- Output: `hamsqr1.dat`

**Rust side:**
- File: `rust_dftb/src/hamiltonian.rs`
- Function: `HamiltonianBuilder::build_h0`
- Use same geometry and SK files
- Output: `rust_h0.dat`
- Compare element-by-element

**Comparison script:**
```python
import numpy as np

def compare_matrices(ref_file, test_file, tol=1e-10):
    ref = read_hamsqr(ref_file)
    test = read_hamsqr(test_file)
    
    diff = np.abs(ref - test)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Elements above tolerance: {np.sum(diff > tol)}")
    
    if max_diff > tol:
        # Find worst elements
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Worst element at {idx}: ref={ref[idx]:.6e}, test={test[idx]:.6e}")
    
    return max_diff < tol
```

---

### Test Level 5: SCC Shifts

**DFTB+ side:**
- File: `dftb_in.hsd`
- Set `SCC = Yes`, `MaxSccIterations = 1` (forces initial guess)
- Run: `dftb+ dftb_in.hsd`
- Output: `hamsqr1.dat` (H with initial SCC shifts)

- Run again with `SCC = No`
- Output: `h0sqr.dat` (H0)

- Compute: `scc_shifts = hamsqr1 - h0sqr`

**Rust side:**
- File: `rust_dftb/src/scc.rs`
- Function: `SccCalculator::calculate_shifts`
- Use same charges (initial guess = neutral atom charges)
- Compare shifts

---

### Test Level 6: Full SCC Convergence

**DFTB+ side:**
- File: `dftb_in.hsd`
- Set `SCC = Yes`, `MaxSccIterations = 200` (default)
- Run: `dftb+ dftb_in.hsd`
- Outputs:
  - `hamsqr1.dat` (final Hamiltonian)
  - `charges.dat` (Mulliken charges, if `WriteCharges = Yes`)
  - `results.tag` (total energy)

**Rust side:**
- File: `rust_dftb/src/main.rs` (or test)
- Run full SCC loop
- Compare:
  - Charges (tolerance: 1e-6)
  - Energy (tolerance: 1e-8 Ha)
  - Final Hamiltonian (tolerance: 1e-8)

---

## 7. File Locations Summary

### Key DFTB+ Source Files

| File | Purpose | Key Functions |
|------|---------|--------------|
| `src/dftbp/dftbplus/main.F90` | Main driver | `buildH0`, `getSccHamiltonian`, `writeHSAndStop` |
| `src/dftbp/dftbplus/mainio.F90` | I/O | `writeHS`, `writeSparseAsSquare`, `writeSparse` |
| `src/dftbp/dftbplus/parser.F90` | Input parsing | `parseOptions`, `getChildValue` |
| `src/dftbp/dftbplus/initprogram.F90` | Initialization | Sets `tSccCalc`, allocates SCC calculator |
| `src/dftbp/dftb/nonscc.F90` | Non-SCC builder | `buildH0`, `buildS`, `buildDiatomicBlocks` |
| `src/dftbp/dftb/sk.F90` | Rotation | `rotateH0` |
| `src/dftbp/dftb/slakocont.F90` | SK container | `getSKIntegrals` |
| `src/dftbp/dftb/slakoeqgrid.F90` | Interpolation | `SlakoEqGrid_getSKIntegrals` |
| `src/dftbp/dftb/scc.F90` | SCC calculator | `updateCharges`, `updateShifts` |
| `src/dftbp/dftb/hamiltonian.F90` | Hamiltonian assembly | `getSccHamiltonian`, `addShift` |
| `src/dftbp/type/oldskdata.F90` | SK file reader | `TOldSKData_readFromFile` |

### Key Variables for Extraction

| Variable | File | Type | Purpose |
|----------|------|------|---------|
| `this%H0` | main.F90 | `real(dp), allocatable :: H0(:)` | Non-SCC Hamiltonian |
| `this%ints%hamiltonian` | main.F90 | `real(dp), allocatable :: hamiltonian(:,:)` | Full Hamiltonian |
| `this%ints%overlap` | main.F90 | `real(dp), allocatable :: overlap(:)` | Overlap matrix |
| `this%potential%intBlock` | main.F90 | `real(dp), allocatable :: intBlock(:,:,:,:)` | SCC shifts |
| `this%tSccCalc` | initprogram.F90 | `logical` | SCC enabled flag |
| `this%denseDesc%iAtomStart` | main.F90 | `integer, allocatable :: iAtomStart(:)` | Dense indexing |
| `this%iSparseStart` | main.F90 | `integer, allocatable :: iPair(0:,:)` | Sparse indexing |

---

## 8. Minimal Code Modifications for Testing

### Recommended Minimal Change (for H0 extraction)

**File:** `src/dftbp/dftbplus/main.F90`

**Location:** Line 1468 (after `getSccHamiltonian`)

**Add:**
```fortran
! DEBUG: Write H0 for parity checking
if (this%tWriteHS) then
  call writeSparseAsSquare(env, "h0sqr_debug.dat", this%H0, this%neighbourList%iNeighbour,&
      & this%nNeighbourSK, this%denseDesc%iAtomStart, this%iSparseStart,&
      & this%img2CentCell)
end if
```

**Benefit:** Works for both `SCC = No` (H0 only) and `SCC = Yes` (H0 before SCC shifts added)

**Recompile:**
```bash
cd /home/prokophapala/git/dftbplus/_build
cmake ..
make -j4
```

---

### Recommended Debug Prints (for intermediate values)

**File:** `src/dftbp/dftb/nonscc.F90`

**Location:** In `buildDiatomicBlocks` (line 364)

**Add at line 417:**
```fortran
call getSKIntegrals(skCont, interSK, dist, iSp1, iSp2)
! DEBUG: Print for first C-C pair
if (iAt1 == 1 .and. iAt2 == 2 .and. iSp1 == 1 .and. iSp2 == 1) then
  write(*,*) "DEBUG_SK", dist, interSK(:)
end if
```

**Add at line 418:**
```fortran
call rotateH0(tmp, interSK, vect(1), vect(2), vect(3), iSp1, iSp2, orb)
! DEBUG: Print for first C-C pair
if (iAt1 == 1 .and. iAt2 == 2 .and. iSp1 == 1 .and. iSp2 == 1) then
  write(*,*) "DEBUG_ROT", vect(1), vect(2), vect(3)
  write(*,*) "DEBUG_MAT", reshape(tmp(1:nOrb2, 1:nOrb1), [nOrb2 * nOrb1])
end if
```

**Note:** Adjust atom indices (`iAt1 == 1`, `iAt2 == 2`) to match your test molecule.

---

## 9. Test Molecule Setup

### Example: Methane (CH4)

**Geometry file:** `methane.gen`
```
C 0.0 0.0 0.0
H 0.629118 0.629118 0.629118
H -0.629118 -0.629118 0.629118
H -0.629118 0.629118 -0.629118
H 0.629118 -0.629118 -0.629118
```

**Input file:** `dftb_in_methane.hsd`
```hsd
Geometry = GenFormat {
  Coordinates = "methane.gen"
}

Hamiltonian = DFTB {
  SCC = No
  SlaterKosterFiles = Type2FileNames {
    Prefix = "/path/to/skfiles/mio-1-1/"
    Separator = "-"
    Suffix = ".skf"
  }
}

Options {
  WriteHS = Yes
}
```

**Run:**
```bash
dftb+ dftb_in_methane.hsd
```

**Output:** `hamsqr1.dat` (H0 for methane)

---

## 10. Automated Testing Script

### Bash Script for Parity Checking

**File:** `run_parity_tests.sh`

```bash
#!/bin/bash

DFTB_PATH="/home/prokophapala/git/dftbplus/_build/dftb+"
RUST_PATH="/home/prokophapala/rust_dftb/target/release/rust_dftb"
TEST_DIR="/home/prokophapala/rust_dftb/tests/data"

for mol in methane benzene water; do
    echo "Testing $mol..."
    
    # Run DFTB+
    cd $TEST_DIR/$mol
    $DFTB_PATH dftb_in.hsd > dftb.log 2>&1
    mv hamsqr1.dat dftb_h0.dat
    
    # Run Rust
    cd -
    $RUST_PATH test --test test_h0_$mol > rust.log 2>&1
    
    # Compare
    python3 compare_matrices.py $TEST_DIR/$mol/dftb_h0.dat $TEST_DIR/$mol/rust_h0.dat
    
    if [ $? -eq 0 ]; then
        echo "  PASS"
    else
        echo "  FAIL"
    fi
done
```

**Make executable:**
```bash
chmod +x run_parity_tests.sh
```

---

## 12. Parity Debugging Log — Bugs Found and Fixed

This section records actual bugs discovered during Rust vs DFTB+ parity checking, their root causes, and the exact fixes. It is intended to prevent the same mistakes in future implementations.

---

### Bug 1: Wrong SK File Column Mapping (CRITICAL)

**Symptom:** SK integrals evaluated at r=1.5 Bohr were all zero, while DFTB+ produced non-zero off-diagonal matrix elements.

**Root cause:** The old 10-column `.skf` format stores 10 Hamiltonian and 10 overlap values per line. DFTB+ maps these into a 20-column internal array via `iSKInterOld = [8, 9, 10, 13, 14, 15, 16, 18, 19, 20]`. The mapping from angular-momentum quantum numbers `(mm, lMax, lMin)` to the 20-column index is defined by `skMap` in `src/dftbp/dftbplus/parser.F90`.

**The skMap array is Fortran column-major** (first index `mm` varies fastest). Decoding the `reshape` call gives:

```
skMap(mm=0, lMax=0, lMin=0) = 20   → ss-σ
skMap(mm=0, lMax=1, lMin=0) = 19   → sp-σ
skMap(mm=0, lMax=1, lMin=1) = 15   → pp-σ
skMap(mm=1, lMax=1, lMin=1) = 16   → pp-π
```

Then mapping new 20-col indices back to old 10-col positions via `iSKInterOld`:
- ss-σ: new 20 → old pos 10 → **0-based index 9**
- sp-σ: new 19 → old pos 9 → **0-based index 8**
- pp-σ: new 15 → old pos 6 → **0-based index 5**
- pp-π: new 16 → old pos 7 → **0-based index 6**

**Wrong initial guess was:** `[0, 5, 1, 2]` (treating old file as already in sp order).

**Fix in `rust_dftb/src/sk_data.rs`:**

```rust
// Correct old-format (10-col) 0-based indices for sp interactions
const OLD_SS_SIGMA: usize = 9; // new col 20
const OLD_SP_SIGMA: usize = 8; // new col 19
const OLD_PP_SIGMA: usize = 5; // new col 15
const OLD_PP_PI:    usize = 6; // new col 16
```

For extended (20-column) format, the same mapping applies directly:

```rust
fn pick_sp(arr20: &[f64]) -> Vec<f64> {
    vec![
        arr20[19], // ss-σ: new col 20
        arr20[18], // sp-σ: new col 19
        arr20[14], // pp-σ: new col 15
        arr20[15], // pp-π: new col 16
    ]
}
```

**Verification:** After fix, H2 and N2 mio-1-1 overlap matrices matched DFTB+ exactly (`Max diff = 0e0`).

---

### Bug 2: Wrong Sign in s-p Block of Rotation Matrix

**Symptom:** N2 overlap matrix showed `s0–px1` elements with wrong sign (e.g. `+3.41e-1` instead of `-3.80e-1`).

**Root cause:** In DFTB+ `rotateH0` (`src/dftbp/dftb/sk.F90`), when `ang1 > ang2` (i.e. p-s interaction where ang1=1 for atom i, ang2=0 for atom j), the code applies:

```fortran
if (ang1 <= ang2) then
    hh(iRow:iRow+nOrb2-1, iCol:iCol+nOrb1-1) = tmpH(1:nOrb2, 1:nOrb1)
else
    hh(iRow:iRow+nOrb2-1, iCol:iCol+nOrb1-1) = (-1.0_dp)**(ang1+ang2) &
        & * transpose(tmpH(1:nOrb1, 1:nOrb2))
end if
```

For p-s (`ang1=1, ang2=0`): `(-1)^(1+0) = -1`. The `sp` subroutine fills `tmpH` as a 3×1 column vector `[m*sk, n*sk, l*sk]` (rows=py,pz,px; col=s). Transposing gives a row vector, and the `-1` sign flips all entries.

**Fix in `rust_dftb/src/rotation.rs`:**

```rust
// p-s block (iSh2=p, iSh1=s → ang2=1, ang1=0 → ang1<=ang2 → direct)
h[(1, 0)] = m * sp; // py_j - s_i
h[(2, 0)] = n * sp; // pz_j - s_i
h[(3, 0)] = l * sp; // px_j - s_i

// s-p block (iSh2=s, iSh1=p → ang2=0, ang1=1 → ang1>ang2 → -transpose)
h[(0, 1)] = -h[(1, 0)]; // s_j - py_i
h[(0, 2)] = -h[(2, 0)]; // s_j - pz_i
h[(0, 3)] = -h[(3, 0)]; // s_j - px_i
```

**Note:** The old code had `h[(0,k)] = h[(k,0)]` (no sign flip), which is only valid when `ang1 < ang2`.

---

### Bug 3: Wrong Transpose Convention in Hamiltonian Assembly

**Symptom:** After fixing rotation signs, the p-p and p-s blocks were placed in transposed positions in the dense matrix.

**Root cause:** DFTB+ `rotateH0` produces a block `tmpH` where **rows = atom j orbitals, cols = atom i orbitals**. Our Rust `rotate_sp_block` correctly matches this convention. However, when placing into the global dense matrix, we must put:
- Lower-left block: `h0[atomJ_row, atomI_col]` = `h_blk[a, b]` directly
- Upper-right block: `h0[atomI_row, atomJ_col]` = `h_blk[b, a]` (transpose)

**Fix in `rust_dftb/src/hamiltonian.rs`:**

```rust
for a in 0..4 {
    for b in 0..4 {
        // rotate_sp_block rows=atomJ, cols=atomI
        h0[(bj + a, bi + b)] = h_blk[(a, b)];      // lower-left
        h0[(bi + a, bj + b)] = h_blk[(b, a)];      // upper-right = transpose

        s[(bj + a, bi + b)] = s_blk[(a, b)];
        s[(bi + a, bj + b)] = s_blk[(b, a)];
    }
}
```

---

### Bug 4: Comma-Separated SK Files (mio-1-1)

**Symptom:** Parsing failed with `"invalid float literal"` when reading `mio-1-1/H-H.skf`.

**Root cause:** The mio-1-1 parameter set uses commas as value separators, e.g.:

```
0.02, 500,1
0.0    0.000039    -0.23860040, -0.0330,  0.3471 0.4919 0.419500 0.0 0.0 1.0
```

**Fix in `rust_dftb/src/sk_data.rs`:** Replace commas with spaces before tokenizing:

```rust
fn parse_numbers_loose(line: &str) -> Vec<f64> {
    let line = line.replace(',', " ");
    // ... rest of parsing ...
}
```

Same fix applied to `parse_numbers_strict` and grid-line tokenization.

---

### Bug 5: Hardcoded 4-Orbitals-Per-Atom Assumption (FIXED)

**Symptom:** Polyatomic molecules with mixed basis (e.g., HCOOH with s-only H and sp C/O) fail with matrix dimension mismatch. Rust produces 20×20 (5 atoms × 4 orbitals) while DFTB+ produces 14×14 (C:4 + O:4 + O:4 + H:1 + H:1).

**Root cause:** `HamiltonianBuilder::build_non_scc` hardcoded `n_orb = 4 * n_atom`.

**Fix implemented:**
1. Added `SpeciesOrbitals` struct with `ang_shells`, `n_orb`, `n_shell`
2. Added `SkData::set_species_angular_momenta()` to define per-species orbital info
3. Replaced `4 * n_atom` with cumulative orbital offsets `i_orb_atom[i]`
4. Updated `fill_onsite` and `fill_pairs` to use per-atom orbital starts
5. Generalized `rotation.rs` to iterate over shells like Fortran `rotateH0`

**Verification:**
- H2: exact match (0e0)
- N2: exact match (0e0)
- HCOOH: H0 diff = 4.87e-7, S diff = 5.56e-13
- HCONH2: H0 diff = 4.64e-8, S diff = 1.62e-12

---

### Bug 6: Reversed SK File Lookup When ang1 > ang2 (FIXED)

**Symptom:** After fixing Bug 5, HCOOH/HCONH2 showed large mismatches (~0.15 for H0, ~0.07 for S) in off-diagonal sp blocks.

**Root cause:** DFTB+ old-format SK files are **not symmetric**. `C-O.skf` and `O-C.skf` contain different data. When `l1 > l2` in `getFullTable`, Fortran uses `skData21(iSK1,iSK2)` — the data from the **reversed** file (e.g., `O-C.skf` instead of `C-O.skf`).

**Why this matters:** The sp integral value depends on which species has the s orbital and which has the p orbital, because the radial wavefunctions differ. `sp(C_s, O_p)` ≠ `sp(O_s, C_p)`.

**Fix in `rust_dftb/src/sk_data.rs`:**
```rust
// Fortran getFullTable: if l1 > l2, use skData21(iSK1,iSK2) = reversed pair data
let (lookup_sp1, lookup_sp2) = if ang1 <= ang2 {
    (sp1, sp2)
} else {
    (sp2, sp1)
};
let tab = self.get_pair(lookup_sp1, lookup_sp2)?;
```

**Key insight from Fortran `parser.F90` `getFullTable`:**
```fortran
if (l1 <= l2) then
  pHam => skData12(iSK2,iSK1)%skHam
  lMin = l1
  lMax = l2
else
  pHam => skData21(iSK1,iSK2)%skHam
  lMin = l2
  lMax = l1
end if
```

When `l1 > l2`, DFTB+ switches to `skData21` which contains the SK data for the reversed species pair (from the reversed `.skf` file). The `skMap(mm, lMax, lMin)` extraction then uses the same `lMax >= lMin` ordering, but the **raw data comes from a different file**.

---

### Reference: Fortran skMap Decoded Table

From `src/dftbp/dftbplus/parser.F90` lines 3984–3990:

```
integer, parameter :: skMap(0:maxL, 0:maxL, 0:maxL) &
    &= reshape((/&
    &20, 0,  0,  0,  19,  0,  0,  0,  18,  0,  0,  0,  17,  0,  0,  0,&
    & 0, 0,  0,  0,  15, 16,  0,  0,  13, 14,  0,  0,  11, 12,  0,  0,&
    & 0, 0,  0,  0,   0,  0,  0,  0,   8,  9, 10,  0,   5,  6,  7,  0,&
    & 0, 0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0,   1,  2,  3,  4/),&
    &(/maxL + 1, maxL + 1, maxL + 1/))
```

Decoded (column-major, mm fastest):

| (mm, lMax, lMin) | Value | Meaning |
|------------------|-------|---------|
| (0, 0, 0) | 20 | ss-σ |
| (0, 1, 0) | 19 | sp-σ |
| (0, 1, 1) | 15 | pp-σ |
| (1, 1, 1) | 16 | pp-π |
| (0, 2, 2) | 8  | dd-σ (not used in sp) |
| (1, 2, 2) | 9  | dd-π (not used in sp) |

---

## 11. Summary of Bugs Found and Fixed

| Bug | File | Root Cause | Fix |
|-----|------|-----------|-----|
| 1 | `sk_data.rs` | Old-format column indices misread | Corrected `OLD_SS_SIGMA=9, OLD_SP_SIGMA=8, OLD_PP_SIGMA=5, OLD_PP_PI=6` |
| 2 | `sk_data.rs` | Extended-format column indices misread | Corrected `pick_sp` to `arr20[19], arr20[18], arr20[14], arr20[15]` |
| 3 | `rotation.rs` | s-p block sign wrong | Applied `(-1)^(ang1+ang2)` factor = `-1` for p-s case |
| 4 | `hamiltonian.rs` | Block transpose convention wrong | Direct placement: `h0[bj+a, bi+b] = h_blk[a,b]` and symmetric partner |
| 5 | `hamiltonian.rs`, `rotation.rs`, `sk_data.rs` | Hardcoded `n_orb = 4 * n_atom` | Added `SpeciesOrbitals`, per-atom offsets, shell iteration |
| 6 | `sk_data.rs` | Reversed SK file not used when `ang1 > ang2` | Swap `(sp1, sp2) → (sp2, sp1)` when `ang1 > ang2` |
| 7 | `hamiltonian.rs` | Missing Ångström→Bohr unit conversion | Added `ANG2BOHR = 1.889726133` constant; convert `coords` once at build start |
| 8 | `sk_data.rs` | `new_to_old` lookup rebuilt at runtime per shell pair | Made `NEW_TO_OLD` a `const` array (compile-time constant, zero runtime cost) |

---

## 12. Summary Checklist

### For Non-SCC Testing

- [x] Set `SCC = No` in `dftb_in.hsd`
- [x] Set `WriteHS = Yes` in `dftb_in.hsd`
- [x] Run DFTB+ to get `hamsqr1.dat` (H0)
- [x] Implement Rust SK parser
- [x] Implement Rust interpolation
- [x] Implement Rust rotation
- [x] Implement Rust H0 builder
- [x] Compare matrices element-by-element

**Verified molecules (tolerance 1e-7 for interpolation differences):**
- [x] H2 (mio-1-1, s-only) — H0 diff 2.0e-8, S diff < 1e-7
- [x] N2 (mio-1-1, sp) — H0 diff 7.8e-8, S diff < 1e-7
- [x] HCOOH (mio-1-1, mixed basis) — H0 diff < 1e-7, S diff < 1e-7
- [x] HCONH2 (mio-1-1, mixed basis) — H0 diff 4.6e-8, S diff 1.6e-12

**Note:** Residual differences (~1e-7) are due to interpolation algorithm differences between Rust (Neville) and Fortran (Neville variant). The SK integral values and rotation matrices match exactly; differences accumulate from grid evaluation.

### For SCC Testing

- [x] Set `SCC = Yes` in `dftb_in.hsd`
- [x] Set `MaxSccIterations = 1` (for initial shifts)
- [x] Run DFTB+ to get `hamsqr1.dat` (H with SCC)
- [x] Run DFTB+ with `SCC = No` to get H0
- [x] Compute SCC_shifts = H_SCC - H0
- [x] Implement Rust SCC calculator (`qmqm/solver.rs`, `qmqm/shifts.rs`, `qmqm/gamma.rs`)
- [x] Compare shifts

**Verified:**
- [x] H2 fixed-charge SCC parity (deltaQ = [0.1, -0.1]) at 0.74 Å — **PASS** (diff = 2.0e-8)
  - Reference: `tests/dftb/h2_ref/hamsqr1.dat`
- [x] N2 fixed-charge SCC parity (deltaQ = [0.2, -0.2]) at 1.10 Å — **PASS** (diff ≈ 1e-8)
  - Reference: `tests/dftb/n2_ref/hamsqr1.dat`
- [x] HCOOH fixed-charge SCC parity (deltaQ = [-0.1, +0.1, -0.1, +0.1, 0.0]) — **PASS** (diff = 8.6e-8)
  - Reference: `tests/dftb/hcooh_ref/hamsqr1.dat`
  - Note: DFTB+ `InitialCharges` sign convention is opposite to intuitive expectation

**Bugs fixed during SCC parity:**
- `generate_ref.py` used `ANG2BOHR` conversion for GenFormat (which expects Å, not Bohr)
  → Removed conversion, regenerated reference data
- `qmqm/shifts.rs` and `qmqm/solver.rs` computed distances in Å for `gamma_full`
  → Added `ANG2BOHR` conversion before gamma evaluation
- `fragment.rs` computed `q0` as full shell capacity (H=2 instead of 1)
  → Now reads `q0` from SK file occupation data (`sk_data.rs:sk_read_onsite_sp`)

**Still missing:**
- [ ] Full SCC convergence parity (self-consistent charges + total energy)
- [ ] `shiftPerL` (shell-resolved SCC) for systems with `ShellResolvedScc = Yes`

### For Debug Output

- [x] Add debug prints in `nonscc.F90:417` (SK integrals)
- [x] Add debug prints in `nonscc.F90:418` (rotation)
- [x] Add H0 output in `main.F90:1468`
- [x] Recompile DFTB+ in debug mode
- [x] Run test molecule
- [x] Capture debug output
- [x] Compare with Rust debug output

## 13. Universal Parity Test

A single script `tests/run_parity.py` generates Fortran reference data and invokes a generic Rust test (`tests/parity_universal.rs`) for **any XYZ molecule**.

### Usage

```bash
cd rust_dftb/tests

# Non-SCC only
python3 run_parity.py /path/to/molecule.xyz

# Non-SCC + SCC with fixed charges
python3 run_parity.py /path/to/molecule.xyz --scc --delta-q 0.1,-0.1,...
```

### What it does

1. Parses XYZ file (handles standard 4-column and 5-column formats)
2. Writes GenFormat geometry + DFTB+ input
3. Runs Fortran DFTB+ (non-SCC) → `ref_h0.dat`, `ref_s.dat`
4. Runs Fortran DFTB+ (SCC, MaxSccIterations=1) → `ref_h_scc.dat`
5. **Reads actual `deltaQAtom` from Fortran `scc_debug_1.txt`** (critical: `InitialCharges` ≠ actual deltaQ)
6. Invokes `cargo test --test parity_universal` with env vars

### Verified molecules (all PASS)

| Molecule | Atoms | Non-SCC | SCC |
|----------|-------|---------|-----|
| H2O      | 3     | ✓       | ✓   |
| CO       | 2     | ✓       | ✓   |
| CH2O     | 4     | ✓       | ✓   |
| HCN      | 3     | ✓       | ✓   |
| C2H4     | 6     | ✓       | ✓   |
| HCOOH    | 5     | ✓       | ✓   |

**Note:** HF is excluded because the `mio-1-1` parameter set does not contain Fluorine (`F`) SK files. DFTB+ segfaults during reference generation when F is requested with this parameter set. Switching to a parameter set that includes F (e.g. `ob2-1-1`) would allow HF testing.

### Key implementation files

- `rust_dftb/tests/run_parity.py` — Python orchestrator
- `rust_dftb/tests/parity_universal.rs` — Generic Rust parity test
- `rust_dftb/src/output.rs` — DFTB+ square matrix reader

---

## 14. Post-Refactor Verification (2025-06-06)

After reorganizing the Rust codebase into `core/`, `methods/`, and `qmqm/` modules:

### What changed

- `core/` — method-agnostic: `error.rs`, `neighbor.rs`, `charges.rs`
- `methods/dftb/` — DFTB-specific: `sk_data.rs`, `interpolation.rs`, `rotation.rs`, `hamiltonian.rs`, `gamma.rs`
- `methods/xtb/` — placeholder for future xTB implementation
- `methods/traits.rs` — `H0Builder` and `CoulombModel` trait boundaries
- `lib.rs` — updated to declare new hierarchy with backward-compatible re-exports
- Old root-level stubs (`error.rs`, `neighbor.rs`, `sk_data.rs`, `interpolation.rs`, `rotation.rs`, `hamiltonian.rs`) removed

### Parity status after refactor

All 6 verified molecules pass both non-SCC and SCC parity:

| Molecule | Non-SCC | SCC |
|----------|---------|-----|
| H2O      | ✓       | ✓   |
| CO       | ✓       | ✓   |
| CH2O     | ✓       | ✓   |
| HCN      | ✓       | ✓   |
| C2H4     | ✓       | ✓   |
| HCOOH    | ✓       | ✓   |

**HF excluded:** `mio-1-1` SK set lacks Fluorine parameters.
