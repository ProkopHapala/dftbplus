# DFTB+ Hamiltonian Assembly - Detailed Investigation Notes

## Overview

This document details the Hamiltonian assembly process in DFTB+, focusing on:
1. Slater-Koster (SK) file reading
2. Matrix element rotation
3. Interpolation
4. SCC (Self-Consistent Charge) charge-dependent term evaluation

## Directory Structure

Key directories:
- `/home/prokophapala/git/dftbplus/src/dftbp/dftb/` - Core DFTB implementation
- `/home/prokophapala/git/dftbplus/src/dftbp/dftbplus/` - Main program and API
- `/home/prokophapala/git/dftbplus/external/tblite/origin/` - External tblite library
- `/home/prokophapala/git/dftbplus/external/slakos/` - Empty (SK files are external)

## Key Files for Hamiltonian Assembly

### Core Hamiltonian Files
- `src/dftbp/dftb/hamiltonian.F90` - Main Hamiltonian assembly
- `src/dftbp/dftb/nonscc.F90` - Non-SCC H0 and S matrix construction
- `src/dftbp/dftb/sk.F90` - Slater-Koster rotation routines
- `src/dftbp/dftb/scc.F90` - SCC charge-dependent terms
- `src/dftbp/dftb/potentials.F90` - Potential data structures

### Slater-Koster Files
- `src/dftbp/dftb/slakoeqgrid.F90` - Equidistant grid SK interpolation
- `src/dftbp/dftb/slakocont.F90` - SK container for all species pairs
- `src/dftbp/type/oldskdata.F90` - Old SK file format reader

### SCC Electrostatics Files
- `src/dftbp/dftb/shortgamma.F90` - Short-range gamma function
- `src/dftbp/dftb/coulomb.F90` - Long-range 1/R Coulomb interactions
- `src/dftbp/dftb/shortgammafuncs.F90` - Gamma function implementations

## 1. Slater-Koster File Reading

### File Format (oldskdata.F90)

The SK file format is defined in `src/dftbp/type/oldskdata.F90`:

**Structure:**
- Line 1: Optional "@" for extended format (f-orbitals)
- Line 2: Grid spacing (dist) and number of grid points (nGrid)
- Line 3 (homonuclear): Atomic eigenvalues, Hubbard U, occupations for each shell
- Line 3 (heteronuclear): Polynomial coefficients for repulsive potential
- Subsequent lines: SK integral tables (H and S) for each grid point
- Optional: Spline repulsive potential section
- Optional: Hybrid XC-functional parameters

**Data stored in TOldSKData type:**
```fortran
type TOldSKData
    real(dp) :: dist              ! Grid spacing
    integer :: nGrid              ! Number of grid points
    real(dp) :: skSelf(4)          ! Atomic eigenvalues (s,p,d,f)
    real(dp) :: skHubbU(4)        ! Hubbard U parameters
    real(dp) :: skOcc(4)          ! Occupations
    real(dp) :: mass              ! Atomic mass
    real(dp), allocatable :: skHam(:,:)   ! Hamiltonian table [nGrid, nSKInter]
    real(dp), allocatable :: skOver(:,:)  ! Overlap table [nGrid, nSKInter]
end type
```

**Number of interactions:**
- Extended format (f-orbitals): 20 interactions
- Old format (spd): 10 interactions

**Interaction mapping (old to new):**
```fortran
integer, parameter :: iSKInterOld(nSKInterOld) = [8, 9, 10, 13, 14, 15, 16, 18, 19, 20]
```

## 2. Slater-Koster Interpolation

### Equidistant Grid Interpolation (slakoeqgrid.F90)

**Data structure:**
```fortran
type TSlakoEqGrid
    integer :: nGrid              ! Number of grid points
    integer :: nInteg             ! Number of integrals
    real(dp) :: dist              ! Grid spacing
    real(dp), allocatable :: skTab(:,:)  ! SK table [nGrid, nInteg]
    integer :: skIntMethod        ! Interpolation method
end type
```

**Two interpolation methods:**

1. **Old method (skEqGridOld)**: Uses 3-point polynomial interpolation
2. **New method (skEqGridNew)**: Uses 8-point polynomial interpolation

**New method interpolation (SlakoEqGrid_interNew_):**
- For distances within grid: 8-point polynomial fit using `polyInterUniform`
- For distances beyond grid: 5th-order polynomial extrapolation to zero
- Uses derivatives at grid boundary for smooth extrapolation
- Extrapolation formula: `poly5ToZero(y1, y1p, y1pp, dr, -distFudge, invDistFudge)`

**Old method interpolation (SlakoEqGrid_interOld_):**
- For distances within grid: 3-point polynomial fit
- Between penultimate and last point: Free cubic spline
- Beyond grid: Free cubic spline extrapolation to zero

**Cutoff calculation:**
```fortran
cutoff = real(nGrid, dp) * dist + distFudge
```
where `distFudge` is a small extension beyond the last grid point.

## 3. Slater-Koster Rotation

### Rotation Theory (sk.F90)

The rotation transforms matrix elements from the SK parameterization orientation (along z-axis) to the actual bond direction.

**Main routine:**
```fortran
subroutine rotateH0(hh, skIntegs, ll, mm, nn, iSp1, iSp2, orb)
```

**Inputs:**
- `skIntegs`: SK integrals for the species pair
- `ll, mm, nn`: Direction cosines of the bond vector
- `iSp1, iSp2`: Species indices
- `orb`: Orbital information

**Process:**
1. Loop over shells of atom 1 and atom 2
2. For each shell pair (ang1, ang2), select appropriate rotation routine
3. Apply rotation based on angular momentum (s, p, d, f)
4. Handle symmetry: if ang1 > ang2, transpose with sign factor

**Rotation routines available:**
- `ss` - s-s interaction
- `sp` - s-p interaction
- `sd` - s-d interaction
- `sf` - s-f interaction
- `pp` - p-p interaction
- `pd` - p-d interaction
- `pf` - p-f interaction
- `dd` - d-d interaction
- `df` - d-f interaction
- `ff` - f-f interaction

**Example (sp rotation):**
```fortran
subroutine sp(hh, ll, mm, nn, sk)
    hh(1,1) = mm*sk(1)    ! s-py
    hh(2,1) = nn*sk(1)    ! s-pz
    hh(3,1) = ll*sk(1)    ! s-px
end subroutine
```

**Maximum angular momentum:** Currently supports up to f-orbitals (l=3)

## 4. Non-SCC Hamiltonian Construction

### Building H0 (nonscc.F90)

**Main routine:**
```fortran
subroutine buildH0(env, ham, skHamCont, selfegy, coords, nNeighbourSK, iNeighbours, species, iPair, orb)
```

**Process:**
1. **On-site energies:** Set diagonal elements from atomic eigenvalues
   ```fortran
   ham(ind) = selfegy(orb%iShellOrb(iOrb1, iSp1), iSp1)
   ```

2. **Diatomic blocks:** For each atom and its neighbors:
   - Calculate distance and direction vector
   - Get SK integrals via interpolation: `getSKIntegrals(skCont, interSK, dist, iSp1, iSp2)`
   - Rotate to bond direction: `rotateH0(tmp, interSK, vect(1), vect(2), vect(3), iSp1, iSp2, orb)`
   - Store in sparse format

**Sparse format:**
- Uses `iPair` indexing for efficient storage
- Only stores non-zero blocks for atom pairs within cutoff

### Building Overlap Matrix (nonscc.F90)

**Main routine:**
```fortran
subroutine buildS(env, over, skOverCont, coords, nNeighbourSK, iNeighbours, species, iPair, orb)
```

**Process:**
1. Set diagonal elements to 1.0 (orthonormal basis)
2. Build diatomic blocks similar to H0 but using overlap SK integrals

## 5. SCC Charge-Dependent Terms

### SCC Module Structure (scc.F90)

**Main SCC type:**
```fortran
type TScc
    integer :: nAtom, nSpecies, mShell, mOrb
    real(dp), allocatable :: shiftPerAtom(:)      ! Shift per atom
    real(dp), allocatable :: shiftPerL(:,:)         ! Shift per shell
    real(dp), allocatable :: deltaQ(:,:)          ! Orbital charge differences
    real(dp), allocatable :: deltaQShell(:,:)      ! Shell charge differences
    real(dp), allocatable :: deltaQAtom(:)        ! Atomic charge differences
    
    ! Electrostatic calculators
    type(TCoulomb), allocatable :: coulomb         ! Long-range 1/R
    type(TShortGamma), allocatable :: shortGamma   ! Short-range gamma
    type(TPoisson), allocatable :: poisson         ! Poisson solver
    
    ! Optional corrections
    type(TChrgPenalty), allocatable :: chrgPenalties
    type(TChrgPenalty), allocatable :: thirdOrder
    type(TExtCharges), allocatable :: extCharges
end type
```

### SCC Workflow

**1. Update charges:**
```fortran
subroutine updateCharges(this, env, qOrbital, orb, species, q0)
```
- Calculate charge differences: `deltaQ = q0 - qOrbital`
- Sum over shells to get shell and atomic charges
- Update both shortGamma and coulomb calculators

**2. Update shifts:**
```fortran
subroutine updateShifts(this, env, orb, species, iNeighbour, img2CentCell)
```
- Calculate potential shifts from charge distribution
- For gamma-electrostatics:
  - Coulomb: `shiftPerAtom = invRMat * deltaQAtom`
  - ShortGamma: `shiftPerL = -sum(shortGamma * deltaQShell)`

**3. Get shifts:**
```fortran
subroutine getShiftPerAtom(this, shift)
subroutine getShiftPerL(this, shift)
```

### Short-Range Gamma (shortgamma.F90)

**Purpose:** Calculate short-range part of electron-electron interaction

**Data structure:**
```fortran
type TShortGamma
    type(TUniqueHubbard), allocatable :: hubbU_     ! Hubbard U parameters
    real(dp), allocatable :: shortCutoffs_(:,:,:,:)  ! Cutoffs for each U pair
    real(dp), allocatable :: shortGamma_(:,:,:,:)    ! Cached gamma values
    real(dp), allocatable :: shiftShell_(:,:)         ! Shell-resolved shifts
end type
```

**Gamma function:**
```fortran
expGamma(r, U2, U1) = erf(sqrt(U1*U2/(U1+U2)) * r) / r
```

**Damping option:**
```fortran
expGammaDamped(r, U2, U1, exponent)
```

**Shift calculation:**
```fortran
shiftShell(iSh1, iAt1) = -sum over neighbors (shortGamma * deltaQShell(iSh2, iAt2))
```

**Force calculation:**
```fortran
force = -deltaQ1 * deltaQ2 * expGammaPrime(r) * (r_vec / r)
```

### Long-Range Coulomb (coulomb.F90)

**Purpose:** Calculate long-range 1/R Coulomb interactions

**Data structure:**
```fortran
type TCoulomb
    real(dp) :: alpha                    ! Ewald parameter
    real(dp), allocatable :: invRMat(:,:)  ! 1/R matrix
    real(dp), allocatable :: shiftPerAtom_(:)
    real(dp), allocatable :: deltaQAtom_(:)
    
    ! Periodic system data
    real(dp) :: latVecs_(3,3), recVecs_(3,3), volume_
    real(dp), allocatable :: gLatPoints_(:,:)   ! Reciprocal lattice points
    type(TDynNeighList), allocatable :: neighList_
end type
```

**Non-periodic case:**
```fortran
invRMat(i,j) = 1.0_dp / |R_i - R_j|
shiftPerAtom = invRMat * deltaQAtom
```

**Periodic case (Ewald summation):**
- Real space: `erfc(alpha * r) / r`
- Reciprocal space: `4*pi/V * exp(-g^2/(4*alpha^2)) / g^2 * exp(i*g*R)`
- Self-interaction correction: `-2*alpha/sqrt(pi)`

**Shift calculation:**
```fortran
call hemv(shiftPerAtom, invRMat, deltaQAtom, 'L')  ! Matrix-vector multiply
```

### Total SCC Hamiltonian Assembly (hamiltonian.F90)

**Main routine:**
```fortran
subroutine getSccHamiltonian(env, H0, ints, nNeighbourSK, neighbourList, species, orb, iSparseStart, img2CentCell, potential, mDftb, isREKS, ham, iHam)
```

**Process:**
1. **Reset Hamiltonian:** `ham(:,:) = 0.0_dp` (unless REKS)

2. **Add SCC shifts to Hamiltonian:**
   ```fortran
   call addShift(env, ham, ints%overlap, nNeighbourSK, neighbourList%iNeighbour, species, orb, iSparseStart, nAtom, img2CentCell, potential%intBlock, .not. isREKS)
   ```

3. **Add non-SCC Hamiltonian:**
   ```fortran
   ham(:,1) = ham(:,1) + h0
   ```

4. **Add on-site shifts (external fields):**
   ```fortran
   call addOnSiteShift(ham, ints%overlap, species, orb, iSparseStart, nAtom, potential%intOnSiteAtom)
   ```

5. **Add dipole/quadrupole terms:**
   ```fortran
   call addAtomicMultipoleShift(ham, ints%dipoleBra, ints%dipoleKet, nNeighbourSK, neighbourList%iNeighbour, species, orb, iSparseStart, nAtom, img2CentCell, atomFieldDeriv)
   ```

6. **Add multipole expansion (if MDFTB):**
   ```fortran
   call mdftb%addMultiExpanHamiltonian(ham, ints%overlap, nNeighbourSK, neighbourList%iNeighbour, species, orb, iSparseStart, nAtom, img2CentCell)
   ```

### Adding Charge Potentials (hamiltonian.F90)

**Routine:**
```fortran
subroutine addChargePotentials(env, sccCalc, tblite, updateScc, qInput, q0, chargePerShell, orb, multipole, species, neighbourList, img2CentCell, spinW, solvation, thirdOrd, dispersion, potential, errStatus)
```

**Contributions added:**
1. **SCC (self-consistent charges):**
   - Update charges if needed
   - Get shifts from SCC calculator
   - Add to `potential%intAtom` and `potential%intShell`

2. **Dispersion:**
   - Update charges
   - Add dispersion potential

3. **TBLite (external library):**
   - Update charges including multipoles
   - Get shifts (atom, shell, dipole, quadrupole)

4. **Third order SCC:**
   - Update charges
   - Add third order potential

5. **Solvation:**
   - Update charges
   - Add solvation potential

6. **Spin:**
   - Add spin-dependent shifts

**Potential hierarchy:**
```fortran
call totalShift(potential%intShell, potential%intAtom, orb, species)  ! Shell to atom
call totalShift(potential%intBlock, potential%intShell, orb, species)  ! Atom to block
```

## 6. Potential Data Structures (potentials.F90)

**Main type:**
```fortran
type TPotentials
    ! Internal potentials (require 0.5 scaling for double counting)
    real(dp), allocatable :: intAtom(:,:)      ! [nAtom, nSpin]
    real(dp), allocatable :: intShell(:,:,:)    ! [mShell, nAtom, nSpin]
    real(dp), allocatable :: intBlock(:,:,:,:)  ! [mOrb, mOrb, nAtom, nSpin]
    
    ! External potentials (no double counting)
    real(dp), allocatable :: extAtom(:,:)
    real(dp), allocatable :: extShell(:,:,:)
    real(dp), allocatable :: extBlock(:,:,:,:)
    real(dp), allocatable :: extGrad(:,:)       ! Gradient of external potential
    
    ! Special potentials
    real(dp), allocatable :: orbitalBlock(:,:,:,:)  ! DFTB+U, pSIC
    real(dp), allocatable :: iorbitalBlock(:,:,:,:) ! Imaginary parts (spin-orbit)
    real(dp), allocatable :: coulombShell(:,:,:)    ! Electrostatic for contact calc
    real(dp), allocatable :: intOnSiteAtom(:,:)     ! On-site internal
    real(dp), allocatable :: extOnSiteAtom(:,:)     ! On-site external
    real(dp), allocatable :: dipoleAtom(:,:)        ! Dipole contribution
    real(dp), allocatable :: quadrupoleAtom(:,:)    ! Quadrupole contribution
end type
```

## 7. Integration with Main Program

The main assembly flow in `src/dftbp/dftbplus/main.F90`:

1. **Initialization:** Read SK files, initialize containers
2. **Geometry setup:** Build neighbor lists
3. **Non-SCC construction:** Build H0 and S matrices
4. **SCC loop:**
   - Calculate charges from density matrix
   - Update SCC shifts
   - Build full Hamiltonian: H = H0 + SCC_shifts
   - Diagonalize
   - Check convergence
5. **Final output:** Energies, forces, etc.

## 8. Key Algorithms Summary

### Rotation Algorithm
1. For each atom pair within cutoff
2. Calculate bond vector and direction cosines (ll, mm, nn)
3. Interpolate SK integrals at distance
4. Select rotation routine based on angular momenta
5. Apply rotation formulas (analytic expressions from Podolskiy & Vogl)
6. Store in Hamiltonian/overlap matrix

### Interpolation Algorithm
1. Find grid index: `ind = floor(distance / grid_spacing)`
2. If within grid:
   - Select nInter points around ind
   - Fit polynomial through these points
   - Evaluate at target distance
3. If beyond grid:
   - Calculate value and derivatives at last grid point
   - Extrapolate to zero using 5th-order polynomial

### SCC Algorithm
1. Calculate charge differences from reference
2. Short-range part:
   - For each neighbor pair, calculate gamma function
   - Sum over neighbors: `shift = -sum(gamma * deltaQ_neighbor)`
3. Long-range part:
   - Build 1/R matrix (Ewald for periodic)
   - Matrix-vector multiply: `shift = invRMat * deltaQ`
4. Add shifts to Hamiltonian diagonal and off-diagonal elements

## 9. Important Constants and Parameters

**Grid parameters:**
- `distFudge`: Small extension beyond grid for smooth cutoff
- `nInterNew_ = 8`: Number of points for new interpolation
- `nInterOld_ = 3`: Number of points for old interpolation
- `deltaR_ = 1e-5_dp`: Displacement for derivative calculation

**Angular momentum limits:**
- `mAngRot_ = 3`: Maximum angular momentum (f-orbitals)

**Cutoffs:**
- SK cutoff: `nGrid * dist + distFudge`
- Short gamma cutoff: `expGammaCutoff(U1, U2)`

## 10. External Library Integration

**TBLite:**
- Located in `external/tblite/origin/`
- Provides alternative SCC implementation
- Integrated via `src/dftbp/extlibs/tblite.F90`
- Can handle multipole moments (dipole, quadrupole)

## 11. File I/O for Hamiltonian

**Export options (from DFTB_docs.md):**
```hsd
Options {
  WriteHS = Yes           # Square H and S matrices
  WriteRealHS = Yes       # Real-space sparse H and S
}
```

**Output files:**
- `hamsqr1.dat` - Zero-charge Hamiltonian
- `oversqr.dat` - Overlap matrix
- `hamreal.dat` - Real-space Hamiltonian
- `overreal.dat` - Real-space overlap

## 12. Detailed File-by-File Contents

### src/dftbp/dftb/slakoeqgrid.F90
**Purpose:** Equidistant grid Slater-Koster interpolation

**Key components:**
- `TSlakoEqGrid` type: Stores SK table data
  - `nGrid`: Number of grid points
  - `nInteg`: Number of integrals (10 for spd, 20 for spdf)
  - `dist`: Grid spacing
  - `skTab`: 2D array [nGrid, nInteg] with SK values
  - `skIntMethod`: Interpolation method selector

- `SlakoEqGrid_init`: Initialize from grid data
- `SlakoEqGrid_getSKIntegrals`: Get interpolated values at distance
- `SlakoEqGrid_getNIntegrals`: Return number of integrals
- `SlakoEqGrid_getCutoff`: Return cutoff distance

- `SlakoEqGrid_interNew_`: New interpolation method
  - Uses 8-point polynomial interpolation within grid
  - Uses 5th-order polynomial extrapolation beyond grid
  - Calls `polyInterUniform` and `poly5ToZero` from dftbp_math_interpolation

- `SlakoEqGrid_interOld_`: Old interpolation method
  - Uses 3-point polynomial interpolation within grid
  - Uses free cubic spline near boundary
  - Uses free cubic spline extrapolation beyond grid

**Constants:**
- `skEqGridOld = 0`, `skEqGridNew = 1`
- `distFudge`, `distFudgeOld`: Cutoff extension parameters

### src/dftbp/dftb/slakocont.F90
**Purpose:** Container for all Slater-Koster tables

**Key components:**
- `TSlakoCont` type: Container for all species pairs
  - `slakos(:,:)`: 2D array of TSlaKo_ types for each species pair
  - `nSpecies`: Number of species
  - `mInt`: Maximum number of integrals
  - `cutoff`: Overall cutoff distance
  - `tDataOK`: Data validity flag

- `SlakoCont_init`: Initialize container
- `SlakoCont_addTableEqGrid`: Add equidistant grid table for species pair
- `SlakoCont_getMIntegrals`: Get maximum number of integrals
- `SlakoCont_getCutoff`: Get overall cutoff
- `SlakoCont_getSKIntegrals`: Get SK integrals for species pair at distance

**Internal type TSlaKo_:**
- `pSlakoEqGrid`: Pointer to TSlakoEqGrid
- `iSp1`, `iSp2`: Species indices

### src/dftbp/dftb/nonscc.F90
**Purpose:** Non-SCC Hamiltonian and overlap matrix construction

**Key components:**
- `TNonSccDiff` type: Differentiation settings
  - `method`: Differentiation method (finite difference, Richardson)
  - `deltaR`: Displacement for derivatives
  - `nOrder`: Order of derivative

- `buildH0`: Build non-SCC Hamiltonian in sparse format
  - Input: SK container, coordinates, neighbor lists, species, orbital info
  - Output: Sparse Hamiltonian matrix
  - Process: Set on-site energies, build diatomic blocks

- `buildS`: Build overlap matrix in sparse format
  - Similar to buildH0 but for overlap integrals
  - Diagonal elements set to 1.0

- `buildDiatomicBlocks`: Helper for diatomic block construction
  - Calculates distance and direction vector
  - Calls getSKIntegrals and rotateH0
  - Stores in sparse format

- `getH0Derivative`: Calculate derivative of H0
  - Supports finite difference and Richardson extrapolation
  - Can calculate first and second derivatives

**Constants:**
- `diffAnalytic = 0`, `diffFiniteDiff = 1`, `diffRichardson = 2`

### src/dftbp/type/oldskdata.F90
**Purpose:** Old SK file format reader

**Key components:**
- `TOldSKData` type: SK file data structure
  - `dist`: Grid spacing
  - `nGrid`: Number of grid points
  - `skSelf(4)`: Atomic eigenvalues (s,p,d,f)
  - `skHubbU(4)`: Hubbard U parameters
  - `skOcc(4)`: Occupations
  - `mass`: Atomic mass
  - `skHam(:,:)`: Hamiltonian table [nGrid, nSKInter]
  - `skOver(:,:)`: Overlap table [nGrid, nSKInter]

- `TOldSKData_readFromFile`: Read SK file
  - Detects extended format (f-orbitals) via "@" on first line
  - Reads grid spacing and number of points
  - Reads atomic parameters (homonuclear) or repulsive coefficients (heteronuclear)
  - Reads H and S tables
  - Handles both standard (10 interactions) and extended (20 interactions) formats

- `TOldSKData_readSplineRep`: Read spline repulsive potential
- `parseHybridXcTag`: Parse hybrid XC functional parameters

**Constants:**
- `nSKInterOld = 10`: Old format interactions
- `nSKInterExt = 20`: Extended format interactions
- `iSKInterOld`: Mapping array [8,9,10,13,14,15,16,18,19,20]

### src/dftbp/dftb/shortgamma.F90
**Purpose:** Short-range gamma function for SCC

**Key components:**
- `TShortGammaInput` type: Input parameters
  - `hubbU`: Hubbard U values
  - `shortCutoffs`: Cutoff distances
  - `damping`: Damping parameters

- `TShortGamma` type: Main calculator
  - `nSpecies_`: Number of species
  - `mShell_`: Maximum shells per species
  - `nAtom_`: Number of atoms
  - `hubbU_`: Hubbard U parameters (contracted)
  - `shortCutoffs_(:,:,:,:)`: Cutoffs [mHubbU, mHubbU, nSpecies, nSpecies]
  - `nNeigh_(:,:,:,:)`: Number of neighbors [mHubbU, mHubbU, nSpecies, nAtom]
  - `deltaQShell_(:,:)`: Net charges per shell
  - `deltaQUniqU_(:,:)`: Charges summed over equivalent Hubbard U
  - `h5Correction_`: H5 correction calculator
  - `damping_`: Gamma damping calculator
  - `shiftShell_(:,:)`: Shell-resolved shifts
  - `shortGamma_(:,:,:,:)`: Cached gamma values

- `TShortGamma_init`: Initialize calculator
- `updateCoords`: Update coordinates and recalculate gamma
- `updateCharges`: Update shell-resolved charges
- `updateShifts`: Calculate shift vectors
- `addAtomMatrix`: Add contributions to atomic matrix
- `addGradient`: Add gradient contributions
- `addStress`: Add stress tensor contributions

**Key functions:**
- `expGamma(r, U2, U1)`: erf(sqrt(U1*U2/(U1+U2)) * r) / r
- `expGammaDamped(r, U2, U1, exponent)`: Damped version
- `expGammaPrime(r, U2, U1)`: Derivative for forces

### src/dftbp/dftb/coulomb.F90
**Purpose:** Long-range Coulomb interactions

**Key components:**
- `TCoulombInput` type: Input parameters
  - `alpha`: Ewald parameter
  - `tolEwald`: Ewald tolerance
  - `autoEwald`: Auto-evaluate Ewald parameter

- `TCoulomb` type: Main calculator
  - `alpha`: Ewald parameter
  - `invRMat(:,:)`: 1/R matrix
  - `nAtom_`: Number of atoms
  - `boundaryCond_`: Boundary condition (cluster or periodic)
  - `latVecs_(3,3)`: Lattice vectors
  - `recVecs_(3,3)`: Reciprocal lattice vectors
  - `volume_`: Cell volume
  - `coords_(:,:)`: Atomic coordinates
  - `gLatPoints_(:,:)`: Reciprocal lattice points for Ewald
  - `rLatPoints_(:,:)`: Real lattice points for Ewald
  - `neighList_`: Dynamic neighbor list
  - `shiftPerAtom_(:)`: Shift per atom
  - `deltaQAtom_(:)`: Net atomic charges

- `TCoulomb_init`: Initialize calculator
- `updateCoords`: Update coordinates and recompute invRMat
- `updateLatVecs`: Update lattice vectors and Ewald parameters
- `updateCharges`: Update atomic charges
- `updateShifts`: Calculate shifts from charges
- `addEnergy`: Add energy contribution
- `addGradient`: Add gradient contribution
- `addStress`: Add stress contribution

**Key functions:**
- `invRCluster`: Compute 1/R matrix for non-periodic systems
- `invRPeriodic`: Compute 1/R matrix for periodic systems using Ewald
- `invRPeriodicBLACS`: Parallel version for ScaLAPACK

### src/dftbp/dftb/potentials.F90
**Purpose:** Potential data structure container

**Key components:**
- `TPotentials` type: Central potential container
  - `intAtom(:,:)`: Internal atom-resolved potential [nAtom, nSpin]
  - `intShell(:,:,:)`: Internal shell-resolved potential [mShell, nAtom, nSpin]
  - `intBlock(:,:,:,:)`: Internal block-resolved potential [mOrb, mOrb, nAtom, nSpin]
  - `extAtom(:,:)`: External atom-resolved potential
  - `extShell(:,:)`: External shell-resolved potential
  - `extBlock(:,:,:,:)`: External block-resolved potential
  - `extGrad(:,:)`: Gradient of external potential
  - `orbitalBlock(:,:,:,:)`: DFTB+U, pSIC potentials
  - `iorbitalBlock(:,:,:,:)`: Imaginary parts (spin-orbit)
  - `coulombShell(:,:,:)`: Electrostatic for contact calculations
  - `intOnSiteAtom(:,:)`: Internal on-site potential
  - `extOnSiteAtom(:,:)`: External on-site potential
  - `dipoleAtom(:,:)`: Dipole contribution
  - `quadrupoleAtom(:,:)`: Quadrupole contribution
  - `extDipoleAtom(:,:)`: External dipole
  - `extQuadrupoleAtom(:,:)`: External quadrupole

- `TPotentials_init`: Initialize and allocate based on system size
- `TPotentials_zero`: Zero all potentials
- `TPotentials_destroy`: Deallocate

**Note:** Internal potentials require 0.5 scaling for double counting in energy evaluation.

### src/dftbp/dftb/sk.F90
**Purpose:** Slater-Koster rotation routines

**Key components:**
- `rotateH0`: Main rotation routine
  - Input: SK integrals, direction cosines (ll, mm, nn), species, orbital info
  - Output: Rotated Hamiltonian block
  - Process: Loops over shells, selects rotation routine, applies transformation

**Rotation routines (one for each angular momentum pair):**
- `ss`: s-s interaction (1 element)
- `sp`: s-p interaction (3 elements)
- `sd`: s-d interaction (5 elements)
- `sf`: s-f interaction (7 elements)
- `pp`: p-p interaction (9 elements)
- `pd`: p-d interaction (15 elements)
- `pf`: p-f interaction (21 elements)
- `dd`: d-d interaction (25 elements)
- `df`: d-f interaction (35 elements)
- `ff`: f-f interaction (49 elements)

**Constants:**
- `mAngRot_ = 3`: Maximum angular momentum (f-orbitals)
- `nAngRot_ = 4`: Number of angular momentum types (s,p,d,f)

### src/dftbp/dftb/scc.F90
**Purpose:** SCC charge-dependent term coordinator

**Key components:**
- `TScc` type: Main SCC calculator
  - `nAtom`, `nSpecies`, `mShell`, `mOrb`: System dimensions
  - `shiftPerAtom(:)`: Shift per atom
  - `shiftPerL(:,:)`: Shift per shell
  - `deltaQ(:,:)`: Orbital charge differences
  - `deltaQShell(:,:)`: Shell charge differences
  - `deltaQAtom(:)`: Atomic charge differences
  - `coulomb`: TCoulomb instance
  - `shortGamma`: TShortGamma instance
  - `poisson`: TPoisson instance
  - `chrgPenalties`: Charge penalty calculator
  - `thirdOrder`: Third-order SCC calculator
  - `extCharges`: External charges calculator

- `TScc_init`: Initialize SCC calculator
- `updateCharges`: Update orbital charges
- `updateShifts`: Update potential shifts
- `getShiftPerAtom`: Get atom-resolved shifts
- `getShiftPerL`: Get shell-resolved shifts
- `addEnergy`: Add SCC energy contribution
- `addGradient`: Add SCC gradient contribution
- `addStress`: Add SCC stress contribution

### src/dftbp/dftb/hamiltonian.F90
**Purpose:** Main Hamiltonian assembly coordinator

**Key components:**
- `getSccHamiltonian`: Build full SCC Hamiltonian
  - Combines H0, SCC shifts, on-site shifts, multipole terms
  - Handles REKS special case
  - Calls addShift, addOnSiteShift, addAtomicMultipoleShift

- `addShift`: Add SCC shifts to Hamiltonian
  - Uses overlap matrix to distribute shifts
  - Handles both atom and block-resolved potentials

- `addOnSiteShift`: Add on-site potential shifts
- `addAtomicMultipoleShift`: Add dipole/quadrupole contributions

- `addChargePotentials`: Add all charge-dependent potentials
  - Coordinates SCC, dispersion, TBLite, third-order, solvation, spin
  - Updates charges if needed
  - Accumulates contributions in TPotentials

### src/dftbp/dftb/shortgammafuncs.F90
**Purpose:** Gamma function implementations

**Key components:**
- Various gamma function implementations:
  - `expGamma`: Basic erf/r function
  - `expGammaDamped`: Damped version
  - `expGammaPrime`: Derivative
  - `expGammaCutoff`: Cutoff calculation
  - `gammaKlopmanOhno`: Klopman-Ohno form
  - `gammaMatagaNishimoto`: Mataga-Nishimoto form

## 13. Reimplementation Checklist

For reimplementation, you will need:

1. **SK file parser:**
   - Read grid spacing and number of points
   - Read atomic eigenvalues and Hubbard U
   - Read H and S tables
   - Handle both old and extended formats

2. **Interpolation module:**
   - Implement polynomial interpolation
   - Implement smooth extrapolation to zero
   - Handle both old and new methods

3. **Rotation module:**
   - Implement all rotation routines (ss, sp, sd, sf, pp, pd, pf, dd, df, ff)
   - Handle direction cosine calculation
   - Implement symmetry handling

4. **Non-SCC builder:**
   - Build sparse Hamiltonian with on-site terms
   - Build sparse overlap matrix
   - Handle neighbor lists

5. **SCC module:**
   - Implement short-range gamma function
   - Implement long-range Coulomb (1/R for cluster, Ewald for periodic)
   - Calculate charge-dependent shifts
   - Add shifts to Hamiltonian

6. **Data structures:**
   - Orbital information (shells, angular momenta)
   - Neighbor lists
   - Sparse matrix storage
   - Potential containers

7. **Integration:**
   - Main assembly loop
   - SCC convergence
   - Energy and force calculation

---

## 14. Extracting Hamiltonian Pieces from DFTB+ for Parity Checking

### 14.1 DFTB+ Output Options (Existing)

DFTB+ provides two built-in output options in the input file (`dftb_in.hsd`):

```hsd
Options {
  WriteHS = Yes           # Dense square H and S matrices
  WriteRealHS = Yes       # Sparse real-space H and S
}
```

**Output files generated:**
- `hamsqr1.dat` — Dense square Hamiltonian (spin channel 1)
- `oversqr.dat` — Dense square overlap matrix
- `hamreal1.dat` — Sparse real-space Hamiltonian (spin channel 1)
- `overreal.dat` — Sparse real-space overlap matrix

**Critical limitation:** These outputs are written AFTER `getSccHamiltonian()` is called in `main.F90:1466`. Therefore, the Hamiltonian output ALWAYS contains SCC shifts when `SCC = Yes`.

### 14.2 How to Extract Non-SCC H0 Separately

#### Method 1: Run DFTB+ with SCC = No (Recommended)

In `dftb_in.hsd`:
```hsd
Hamiltonian = DFTB {
  SCC = No
  ...
}

Options {
  WriteHS = Yes
}
```

**What happens:**
- `this%tSccCalc = .false.` in `initprogram.F90`
- `maxSccIter = 1` (only one iteration, no charge mixing)
- `buildH0()` is called at `main.F90:1237` to construct `this%H0`
- `getSccHamiltonian()` is called at `main.F90:1300` but with zero potentials (no SCC calculator allocated)
- The output `hamsqr1.dat` contains ONLY the non-SCC Hamiltonian H0

**Verification:** When `SCC = No`, `tSccCalc = .false.` at `initprogram.F90:1364`. The SCC calculator is never allocated (`if (this%tSccCalc)` blocks at lines 1726, 1789, etc. are skipped).

#### Method 2: Run DFTB+ Twice and Subtract

1. **Run 1:** `SCC = No` → get H0
2. **Run 2:** `SCC = Yes` → get H_full
3. **Compute:** SCC_shifts = H_full - H0 (element-wise subtraction)

**Advantage:** Completely non-invasive, no code modification.
**Disadvantage:** Two runs required, need to match sparse indexing exactly.

#### Method 3: Modify DFTB+ to Output H0

Add code in `src/dftbp/dftbplus/main.F90` at line ~1466 (after `getSccHamiltonian`):

```fortran
! Write H0 separately for parity checking
if (this%tWriteHS) then
  call writeSparseAsSquare(env, "h0sqr.dat", this%H0, iNeighbour, nNeighbourSK,&
      & iAtomStart, iPair, img2CentCell)
end if
```

**Location:** After line 1468 in `main.F90`, before line 1470.

#### Method 4: Add New Input Option `WriteH0 = Yes`

In `src/dftbp/dftbplus/parser.F90`, add parsing for `WriteH0`:
```fortran
call getChildValue(node, "WriteH0", ctrl%tWriteH0, .false.)
```

Then in `mainio.F90`:
```fortran
subroutine writeH0(env, H0, iNeighbour, nNeighbourSK, iAtomStart, iPair, img2CentCell)
  call writeSparseAsSquare(env, "h0sqr.dat", H0, iNeighbour, nNeighbourSK,&
      & iAtomStart, iPair, img2CentCell)
end subroutine
```

### 14.3 Extracting Intermediate Values (SK Integrals, Rotated Values)

To debug the assembly pipeline, we need to extract:
1. Raw SK integrals at specific distances
2. Direction cosines (l, m, n)
3. Rotated matrix elements
4. On-site energies

#### Option A: Add Debug Prints in DFTB+ Source

**Location:** `src/dftbp/dftb/nonscc.F90:417-418` in `buildDiatomicBlocks`:

```fortran
! After getSKIntegrals
call getSKIntegrals(skCont, interSK, dist, iSp1, iSp2)
write(*,*) "DEBUG SK", iAt1, iAt2, dist, interSK(:)

! After rotateH0
call rotateH0(tmp, interSK, vect(1), vect(2), vect(3), iSp1, iSp2, orb)
write(*,*) "DEBUG ROT", iAt1, iAt2, vect(:)
write(*,*) "DEBUG MAT", reshape(tmp, [size(tmp)])
```

**Compile DFTB+ in debug mode** (CMake option `-DCMAKE_BUILD_TYPE=Debug`) to enable these prints.

#### Option B: Use a Minimal Test Fortran Program

Write a standalone Fortran program that:
1. Reads SK files directly using `TOldSKData_readFromFile`
2. Calls `getSKIntegrals` at specific distances
3. Calls `rotateH0` with known direction cosines
4. Prints results for comparison

```fortran
program test_sk
  use dftbp_type_oldskdata
  use dftbp_dftb_slakocont
  use dftbp_dftb_sk
  ! ... read SK file, test interpolation and rotation
end program
```

#### Option C: Use DFTB+ Python API (pyBall)

The `pyBall` directory in DFTB+ might expose Hamiltonian building functions. Check if the Python interface allows:
- Accessing `this%H0` directly
- Calling `buildH0()` independently

### 14.4 Sparse Matrix Format Details

DFTB+ uses a **block-sparse format** with the following indexing:

**Key arrays:**
- `iAtomStart(iAtom)`: Start index of atom iAtom in the dense matrix (1-based)
- `iPair(0:nNeighbour, nAtom)`: Start index in sparse storage for each neighbor
- `img2CentCell(nAtom)`: Maps image atoms back to central cell atoms

**Dense square output format (`hamsqr1.dat`):**
```
  n  m
  H(1,1) H(1,2) ... H(1,m)
  H(2,1) H(2,2) ... H(2,m)
  ...
```
Where n = m = total number of orbitals.

**Sparse real-space output format (`hamreal.dat`):**
```
  iAtom  jAtom  iCell  jCell  iOrb  jOrb  H(i,j)
```
Where iCell, jCell are cell indices for periodic systems.

**For parity checking:** Read `hamsqr1.dat` as a dense matrix and compare element-by-element with the Rust implementation's dense matrix output.

### 14.5 Extracting from TBLite / TBLight

**TBLite** (external library, `external/tblite/`):
- Builds H0 using analytical STO overlaps, not SK tables
- For parity, use TBLite's `buildSH0` API at `main.F90:1248`
- Can be called with `hamiltonianTypes%xtb`

**TBLight** (if available as standalone):
- Likely has its own output format
- Check if it supports similar `WriteHS` options

**Key difference:** TBLite H0 ≠ DFTB+ H0 because:
- TBLite uses analytical STO overlap integrals
- DFTB+ uses tabulated SK integrals
- For parity, compare within the same method only

---

## 15. Systematic Testing Strategy for Rust Implementation

### 15.1 Test Levels (Bottom-Up)

#### Level 1: SK File Parsing Test

**Goal:** Verify Rust parser reads the same values as DFTB+.

**Method:**
1. Pick a test SK file (e.g., `C-C.skf` from mio set)
2. In DFTB+: Add a debug print after `TOldSKData_readFromFile` to dump `skHam` and `skOver` tables
3. In Rust: Parse the same file, print tables
4. Compare element-by-element at all grid points

**Expected precision:** Exact match (within floating-point rounding, ~1e-15)

#### Level 2: Interpolation Test

**Goal:** Verify interpolation at arbitrary distances matches DFTB+.

**Method:**
1. Choose a species pair (e.g., C-C)
2. In DFTB+: Modify `buildDiatomicBlocks` to print `interSK` for a specific distance
3. In Rust: Evaluate the interpolator at the same distance
4. Compare all 4 interaction values (ssσ, spσ, ppσ, ppπ)

**Test distances:**
- Within grid: 1.0 Å, 2.0 Å, 3.0 Å
- Near boundary: cutoff - 0.1 Å, cutoff + 0.1 Å
- Extrapolation region: beyond cutoff

**Expected precision:** ~1e-12 (interpolation introduces small errors)

#### Level 3: Rotation Test

**Goal:** Verify rotation formulas for known bond directions.

**Method:**
1. Fix SK integrals: set `interSK = [1.0, 0.5, 0.3, 0.1]` (arbitrary)
2. Test bond directions:
   - Along z-axis: `(0, 0, 1)` → should match raw SK values
   - Along x-axis: `(1, 0, 0)` → pz → px swap
   - Along y-axis: `(0, 1, 0)` → pz → py swap
   - 45° in xz-plane: `(1/√2, 0, 1/√2)`
3. Compare with DFTB+ `rotateH0` output for same inputs

**Expected precision:** Exact match (~1e-15)

#### Level 4: H0 Construction Test (Non-SCC)

**Goal:** Verify full H0 matrix matches DFTB+ with `SCC = No`.

**Method:**
1. Create a small test molecule (e.g., CH4, H2O, benzene)
2. Run DFTB+ with `SCC = No` and `WriteHS = Yes`
3. Read `hamsqr1.dat` from DFTB+
4. Run Rust code on same geometry
5. Compare element-by-element

**Tolerance:**
- On-site energies: exact match
- Off-diagonal: ~1e-10 (accumulated interpolation errors)

**Debugging failures:**
- If off by constant factor: check sign conventions in rotation
- If off-diagonal pattern is wrong: check neighbor list or orbital ordering
- If specific pairs are wrong: check direction cosines or species mapping

#### Level 5: SCC Shift Test

**Goal:** Verify SCC charge-dependent shifts.

**Method:**
1. Run DFTB+ with `SCC = Yes`, `MaxSccIterations = 1` (forces initial guess)
2. Output H_full (full Hamiltonian)
3. Run DFTB+ with `SCC = No` to get H0
4. Compute ΔH = H_full - H0
5. Run Rust SCC module, compare shifts

**Alternative:**
1. Run DFTB+ with `SCC = Yes`, converged
2. Modify code to output `shiftPerAtom` and `shiftPerL`
3. Compare with Rust SCC calculator outputs

#### Level 6: Full SCC Convergence Test

**Goal:** Verify iterative SCC loop produces same charges and energy.

**Method:**
1. Run full DFTB+ SCC calculation to convergence
2. Output final charges, energy, Hamiltonian
3. Run Rust with same parameters
4. Compare:
   - Mulliken charges (tolerance: 1e-6)
   - Total energy (tolerance: 1e-8 Ha)
   - Final Hamiltonian (tolerance: 1e-8)

### 15.2 Regression Test Suite Structure

```
tests/
├── data/
│   ├── sk_files/          # Test SK files (mio set)
│   ├── geometries/        # .gen files for test molecules
│   └── reference/         # Expected outputs from DFTB+
│       ├── h0_methane.dat
│       ├── h0_benzene.dat
│       ├── scc_methane.dat
│       └── shifts_methane.dat
├── test_sk_parser.rs
├── test_interpolation.rs
├── test_rotation.rs
├── test_h0.rs
├── test_scc.rs
└── test_full.rs
```

### 15.3 Continuous Parity Checking Workflow

**Step 1: Generate reference data**
```bash
# For each test molecule:
cd tests/data/reference/
dftb+ dftb_in.hsd   # Run with SCC=No, WriteHS=Yes
mv hamsqr1.dat h0_molecule.dat
dftb+ dftb_in_scc.hsd  # Run with SCC=Yes, WriteHS=Yes
mv hamsqr1.dat scc_molecule.dat
```

**Step 2: Rust test reads reference**
```rust
#[test]
fn test_h0_methane() {
    let ref_h0 = read_dense_matrix("tests/data/reference/h0_methane.dat");
    let rust_h0 = build_h0(&geometry, &sk_tables);
    assert_matrix_close(&ref_h0, &rust_h0, 1e-10);
}
```

**Step 3: Automated comparison script**
```bash
#!/bin/bash
# run_parity_check.sh
for mol in methane benzene water; do
    cargo test test_h0_${mol}
    cargo test test_scc_${mol}
done
```

### 15.4 TBLite Parity Checking

For xTB/tblite comparison:
1. Use `tblite` Python package or standalone binary
2. Extract H0 and overlap using tblite's API
3. Note: xTB H0 uses different formalism (analytical STOs, not SK tables)
4. Compare only at the level of:
   - Final eigenvalues (should match ~1e-4 eV)
   - Total energy
   - Forces (if implemented)

---

## 16. Recommended Minimal Test Molecules

| Molecule | Basis | Atoms | Tests |
|----------|-------|-------|-------|
| H2 | s only | 2 | On-site, ssσ bond |
| CH4 | sp | 5 | sp hybridization, tetrahedral angles |
| C2H2 | sp | 4 | Linear, sp σ + π bonds |
| C2H4 | sp | 6 | Planar, sp2 hybridization |
| Benzene (C6H6) | sp | 12 | Aromatic system, delocalized π |
| H2O | sp | 3 | Bent geometry, lone pairs |
| NH3 | sp | 4 | Pyramidal, lone pair |

These cover:
- Homonuclear (C-C, H-H) and heteronuclear (C-H) bonds
- Multiple bond directions (linear, tetrahedral, planar)
- Different angular momentum combinations
- Small enough for fast testing

---

# USER

OK, to debug interpolation of simplest possible example use just H2 molecule.

ad sparse vs dense hamiltonain - Can you extract sparse formof hamiltionian form dftb+? If yes we should also generate this as it would be more direct comparison. In the and we want the sparse form anyway. We exported dense only because it was easy.

Do not guess orbital ordering by arbitrary trial and error, read the fotran reference carefully.

One tio - S-matrix should be simpler to debug as it depend only on one component (overlap) hole hamiltonian has more components kinetic energy, electrostatics, pseudopotentiasl etc.

We must achieve parity in overlap matrix fist.

Make detailed paln and then implemen it

---

# Kimi-2.6

