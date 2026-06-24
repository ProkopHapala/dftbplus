# Multipole Integral Debug Investigation Notes

## Problem Statement
Rust implementation of GFN2 multipole integrals shows large errors compared to tblite Fortran reference:
- Dipole integrals: errors ~0.1-1.0 (should be ~1e-4)
- Quadrupole integrals: errors ~0.01-0.1 (should be ~1e-4)
- Overlap integrals: good parity (errors ~1e-6)

## Investigation Status

### Files Reviewed
1. `/external/tblite/origin/src/tblite/integral/multipole.f90` - Fortran reference
2. `/rust_dftb/src/methods/xtb/multipole_integrals.rs` - Rust implementation
3. `/external/tblite/origin/src/tblite/basis/slater.f90` - Slater-to-Gauss expansions
4. `/external/tblite/origin/src/tblite/integral/trafo.f90` - Cartesian to spherical transformations
5. `/external/tblite/origin/src/tblite/xtb/h0.f90` - How multipole_cgto is called
6. `/external/tblite/origin/src/tblite/basis/type.f90` - Integral cutoff logic
7. `/external/tblite/origin/src/tblite/api/result.f90` - C API for dipole/quadrupole integrals

### Key Findings

#### 1. Integral Kernel (multipole_3d) - VERIFIED CORRECT
- Rust implementation matches Fortran line-by-line
- Polynomial coefficient initialization: `vi[li[k]] = 1.0`, `vj[lj[k]] = 1.0` ✓
- horizontal_shift logic matches ✓
- form_product logic matches ✓
- 1D integral accumulation (overlap, dipole, quadrupole) matches ✓
- 3D assembly (product of dimensions) matches ✓

#### 2. Slater-to-Gauss Parameters - VERIFIED CORRECT
- Compared pAlpha3/pCoeff3, pAlpha4/pCoeff4, pAlpha5/pCoeff5, pAlpha6/pCoeff6
- All exponents and coefficients match exactly between Fortran and Rust
- Normalization formula uses dfactorial in both ✓

#### 3. Integral Cutoff - POTENTIAL ISSUE
- Fortran: `intcut = integral_cutoff(acc)` where `acc` is accuracy parameter
- Formula: `intcut = clip(max_intcut - 10*log10(clip(acc, min_acc, max_acc)), min_intcut, max_intcut)`
  - max_intcut = 25.0, min_intcut = 5.0
  - max_acc = 1.0e-4, min_acc = 1.0e+3
- Rust: hardcoded `intcut = 40.0`
- **This is likely WRONG** - Rust uses 40.0 while Fortran uses 5.0-25.0 depending on accuracy

#### 4. CGTO Normalization - NEEDS VERIFICATION
- Fortran slater_to_gauss_array applies normalization including:
  - `prefactor = (2.0_wp*n + 1.0_wp) * dfactorial(2*n - 1) / (dfactorial(2*n) * 4.0_wp * n)`
  - Final: `cgto%coeff(iprim) = prefactor * coeff(iprim) * sqrt(pref)**n`
- Rust slater_to_gauss applies similar formula but needs verification
- **This is a prime suspect for the bug**

#### 5. Transformations - VERIFIED CORRECT
- For s and p orbitals, transformations are identity or simple linear combinations
- Rust trafo0, trafo1 match Fortran transform0, transform1

#### 6. Trace Removal - VERIFIED CORRECT
- Rust removes trace from quadrupole: `q[0] -= trace/3.0`, `q[2] -= trace/3.0`, `q[5] -= trace/3.0`
- Matches Fortran logic

## Suspected Root Causes (Priority Order)

### HIGH PRIORITY: Integral Cutoff
- Rust uses `intcut = 40.0` (hardcoded)
- Fortran uses `intcut = 5.0-25.0` (accuracy-dependent)
- This affects which shell pairs are included in the calculation
- **Action**: Check if cutoff is causing shell pairs to be skipped incorrectly

### HIGH PRIORITY: CGTO Normalization
- The normalization formula is complex and involves double factorials
- Small errors in normalization propagate through all integrals
- **Action**: Add debug output to compare individual primitive coefficients

### MEDIUM PRIORITY: 1D Integral Array Bounds
- Rust: `s1d` array needs size `max_l + 3` for dipole/quadrupole
- Fortran: `s1d(max_l+2)` is allocated
- **Action**: Verify array sizing in overlap_1d

## Next Steps

1. **Fix integral cutoff**: Change from 40.0 to match Fortran logic (use ~25.0 for high accuracy)
2. **Add diagnostic unit test**: Compute single dipole integral for H2 s-s pair at known distance
3. **Compare with tblite C helper**: Use existing C helper to get reference value for same shell pair
4. **Debug normalization**: Print individual primitive coefficients before/after normalization
5. **Iterate**: Fix issues found, re-test until parity achieved

## Test Results

### Unit Test: H2 Dipole Integral
- H2 at 1.4 Bohr bond length
- Tblite dipole[0,1][0] = 0.3973299102325661
- Rust dipole[0,1][0] = 0.4642599862530868
- Error = 0.067 (17% relative error)
- Pattern is correct (off-diagonal non-zero, diagonal zero)

### CGTO Coefficients Debug Output
Rust computed for H (zeta=1.23):
- alpha = [3.3702276975e0, 6.1389118221e-1, 1.6614291148e-1]
- coeff = [2.7359110589e-1, 2.6460540654e-1, 8.2465947534e-2]

Verification:
- Raw alpha from table: [2.227660584, 0.4057711562, 0.1098175104]
- Scaled by zeta^2=1.5129: [3.370, 0.614, 0.166] ✓ (matches)
- Normalization formula matches Fortran exactly ✓

**Conclusion**: CGTO normalization appears correct. Bug must be elsewhere.

## Basis Parameters Audit (COMPLETED)

### Fortran gfn2.f90 arrays vs Rust params_gfn2.rs:
- **nshell**: H=1 ✓ (matches)
- **ang_shell**: H=[0,0,0] (1s) ✓ (matches)
- **principal_qn**: H=[1,0,0] (n=1) ✓ (matches)
- **number_of_primitives**: H=[3,0,0] (3 primitives) ✓ (matches)
- **slater_exponent**: H=1.230000 ✓ (matches)

**CONCLUSION**: All basis parameters match exactly between Fortran and Rust.

## BUG FOUND: UNIT CONVERSION MISMATCH!

### Root Cause:
The C helper (`tblite_helper.c`) expects coordinates in **Angstrom** and internally converts to Bohr. Test was passing Bohr.

### Fix:
Pass Angstrom coordinates to the C helper while keeping Bohr coordinates for Rust.

### After Fix:
- H2 dipole integral test PASSES ✓

## BUG FOUND: DFACTORIAL INDEXING ERROR!

### Root Cause:
Fortran `dfactorial` array is **1-indexed**: `dfactorial(1)=1.0, dfactorial(2)=1.0, dfactorial(3)=3.0`
Rust `DFACTORIAL` array is **0-indexed**: `DFACTORIAL[0]=1.0, DFACTORIAL[1]=1.0, DFACTORIAL[2]=3.0`

Fortran uses `dfactorial(l+1)` in normalization, so Rust must use `DFACTORIAL[l]`.
But Rust code used `DFACTORIAL[l+1]` — **off by one!**

For p orbitals (l=1):
- Fortran: `sqrt(dfactorial(2)) = sqrt(1.0) = 1.0`
- Broken Rust: `sqrt(DFACTORIAL[2]) = sqrt(3.0) = 1.732`
- **p-orbital normalization was too large by √3!**

### Impact:
- p-p on-site overlap wrong by factor of **3** (1.0 vs 0.333)
- p-orbital dipole/quadrupole integrals wrong by **√3** or **3**
- This explains ALL the N2 integral discrepancies!

### Fix:
Changed `DFACTORIAL[l + 1]` to `DFACTORIAL[l]` in `slater_to_gauss`.

### After Fix:
- **N dipole/quadrupole/overlap integral test PASSES!** ✓ (all errors < 1e-5)
- **N2 eigenvalue error dropped from 2.6% to 0.36%!** (3.59e-3)

### Remaining Work:
N2 eigenvalue error is 3.6e-3 (0.36%), still above 1e-4 target. This may be due to:
1. SCF convergence differences (39 vs 36 iterations)
2. Remaining Hamiltonian construction issues
3. Integral cutoff (25.0 vs Fortran dynamic)
4. Need to verify H2 SCC Hamiltonian still passes

## Current Test Status (After All Fixes):

| Test | Status | Notes |
|------|--------|-------|
| H2 dipole integral | **PASS** | s-s parity achieved |
| H2 SCC Hamiltonian | **PASS** | Full SCF with multipoles works |
| N dipole/quadrupole/overlap | **PASS** | p-orbital parity achieved |
| N2 SCC charges | **PASS** (1e-10 error) | Charges essentially perfect |
| N2 SCC eigenvalues | **FAIL** (0.36% error) | Down from 2.6%, needs investigation |
| HCOOH SCC charges | **FAIL** (~1% error) | Small systematic shift |
| HCOOH SCC eigenvalues | Close (~0.006 diff) | Very close but charges fail first |

### Next Steps:
1. Verify if remaining errors are in SCF loop or Hamiltonian construction
2. Compare effective Hamiltonian matrices element-wise for N2
3. Check Coulomb matrix / gamma matrix construction
4. Consider if integral cutoff needs dynamic computation

---

## CRITICAL BUG FOUND: Coordination Number Formula (2025-06-07)

### Problem:
`compute_coordination_numbers` used a completely wrong single-exponential formula:
```rust
// WRONG (old Rust)
countf = exp(-3.0 * (r/rc - 1.0))
```

### Fortran Reference (tblite ncoord/gfn.f90):
Uses a **double exponential** counting function:
```fortran
countf = exp_count(ka, r, rc) * exp_count(kb, r, rc + r_shift)
where exp_count(k, r, r0) = 1.0 / (1.0 + exp(-k*(r0/r - 1.0)))
ka = 10.0, kb = 20.0, r_shift = 2.0
```

### Impact:
- N2 CN: old=3.515, correct=0.999 (3.5x too large!)
- N2 mrad: old=2.088, correct=1.900
- HCOOH CN similarly wrong
- This propagated to multipole damping radii, making `amat_dd`, `amat_sd`, `amat_sq` all wrong
- This caused dipole/charge potentials to be off by 10-30%

### Fix:
Rewrote `compute_coordination_numbers` to match Fortran exactly.

### After Fix:
| Test | Before | After | Improvement |
|------|--------|-------|-------------|
| N2 eff. Hamiltonian | 2.94e-3 | **1.11e-4** | 26x better |
| HCOOH eff. Hamiltonian | 1.51e-2 | **9.09e-4** | 16x better |
| HCOOH charges | ~1% error | **~0.03% error** | 30x better |
| N2 potentials (ref moments) | 10-30% off | **dipole/quadrupole match exactly** | fixed |

### Remaining Issues:
- N2 eff. Hamiltonian still 1.1e-4 off (0.026% relative) at position (2,2)
- HCOOH eff. Hamiltonian still 9.1e-4 off (0.21% relative) at position (13,13)
- HCOOH charges still off by ~0.001 (0.03%)
- Charge potential still has small residual error (~1e-4 for N2)
- Need to investigate: gamma matrix, third-order potential, or subtle multipole differences
