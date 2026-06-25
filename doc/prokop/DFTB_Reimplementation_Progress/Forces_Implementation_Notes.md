# DFTB Forces Implementation — Review & Plan

**Created:** 2026-06-25
**Goal:** Implement forces in `rust_dftb`, achieving parity with Fortran DFTB+

---

## 1. Force Decomposition in DFTB+

The total DFTB force is the negative gradient of the total energy:

```
F_total = -dE/dR = F_nonSCC + F_SCC_shift + F_SCC_dc + F_rep
```

The main force assembly happens in `getGradients()` at
`@/home/prokop/git/dftbplus/src/dftbp/dftbplus/main.F90:6938`.

### 1.1 Non-SCC Electronic Force (F_nonSCC)

**Source:** `derivative_nonSCC` in `@/home/prokop/git/dftbplus/src/dftbp/dftb/forces.F90:45`

This is the "band structure force" from the density matrix (DM) and energy-weighted density matrix (EDM):

```
F_i = 2 * ( sum(DM_ij · dH0/dx_i) - sum(EDM_ij · dS/dx_i) )
F_j = -F_i   (Newton's third law)
```

- **DM** = density matrix = 2 · C_occ · C_occ^T (closed-shell)
- **EDM** = energy-weighted density matrix = 2 · C_occ · diag(eps_occ) · C_occ^T
- **dH0/dx** and **dS/dx** are computed by **finite differences** (displacing one atom of the pair by ±deltaX)
- Factor of 2 accounts for implicit summation over lower triangle of DM

**Key detail:** DFTB+ uses finite differences for dH0/dR and dS/dR, NOT analytic derivatives.
The `TNonSccDiff` type in `@/home/prokop/git/dftbplus/src/dftbp/dftb/nonscc.F90:39` supports:
- `diffTypes%finiteDiff` (default): central finite difference with `deltaXDiff = epsilon(1.0_dp)^0.25`
- `diffTypes%richardson`: Richardson extrapolation

The finite difference code (`getFirstDerivFiniteDiff` at `nonscc.F90:428`) simply:
1. Displaces atomJ by ±deltaX along each Cartesian direction
2. Rebuilds the diatomic block (interpolate SK + rotate)
3. Takes central difference: `(block(+) - block(-)) / (2·deltaX)`

**This is charge-independent** — present even without SCC. Only needs DM and EDM from diagonalization of H0/S.

### 1.2 SCC Shift Force (F_SCC_shift)

**Source:** `derivative_block` in `@/home/prokop/git/dftbplus/src/dftbp/dftb/forces.F90:325`

When SCC is enabled, the Hamiltonian includes potential shifts: H = H0 + S·shift. The force from the shift derivative adds to the non-SCC part:

```
F_i += 2 * sum( shiftSprime · DM )
```

where:
```
shiftSprime = 0.5 * ( S'·shift_atom1 + shift_atom2·S' )
```

- `S'` = dS/dx (same finite-difference derivative as non-SCC)
- `shift_atom1`, `shift_atom2` = block-resolved SCC potential shifts for atoms i and j
- `DM` = density matrix (spin-resolved)

This is the "Pulay-like" term from the SCC potential depending on the overlap matrix.

### 1.3 SCC Double-Counting Force (F_SCC_dc)

**Source:** `scc%addForceDc` in `@/home/prokop/git/dftbplus/src/dftbp/dftb/scc.F90:759`

This is the direct Coulomb/gamma force from charge-charge interactions. It is NOT covered by the shift derivative above.

**Short-range gamma** (`shortGamma%addGradientsDc` at `@/home/prokop/git/dftbplus/src/dftbp/dftb/shortgamma.F90:348`):
```
F_i = -deltaQ_i · deltaQ_j · gamma'(r_ij) · r_vec_ij / r_ij
F_j = +deltaQ_i · deltaQ_j · gamma'(r_ij) · r_vec_ij / r_ij
```

where `gamma'(r)` = `expGammaPrime(r, U_a, U_b)` — derivative of the short-range gamma function
(see `@/home/prokop/git/dftbplus/src/dftbp/dftb/shortgammafuncs.F90:153`).

**Long-range Coulomb 1/R** (`coulomb%addGradients` → `addInvRPrimeCluster` at
`@/home/prokop/git/dftbplus/src/dftbp/dftb/coulomb.F90:1135`):
```
F_i = -deltaQ_i · deltaQ_j / r_ij³ · r_vec_ij
F_j = +deltaQ_i · deltaQ_j / r_ij³ · r_vec_ij
```

Note: For non-periodic systems, the total gamma already includes 1/R, so the short-range gamma'
plus the 1/R' gives the full d(gamma)/dR. In DFTB+ these are split into `shortGamma` (erf part)
and `coulomb` (1/R part) for cutoff/neighbor-list reasons.

### 1.4 Repulsive Pair Force (F_rep)

**Source:** `repulsive%getGradients` → `getTwoBodyGradients_` at
`@/home/prokop/git/dftbplus/src/dftbp/dftb/repulsive/twobodyrep.F90:273`

```
F_i = dE_rep/dr · r_vec_ij / r_ij
F_j = -dE_rep/dr · r_vec_ij / r_ij
```

The repulsive energy E_rep(r) is a spline polynomial read from the SK file.
Its derivative `dE_rep/dr` is obtained from `TSplineRep_getValue` with `dEnergy` argument
(`@/home/prokop/git/dftbplus/src/dftbp/dftb/repulsive/splinerep.F90:136`).

The spline has:
- Exponential head for r < xStart(1)
- Cubic spline segments in the middle
- 5th-order polynomial tail near cutoff
- Zero beyond cutoff

### 1.5 Force Assembly Flow in main.F90

```
getGradients() [main.F90:6938]
├── if (.not. SCC):
│   └── derivative_nonSCC()           # F_nonSCC only
├── if (SCC):
│   ├── derivative_block()            # F_nonSCC + F_SCC_shift
│   ├── scc%addForceDc()              # F_SCC_dc (gamma + Coulomb)
│   ├── thirdOrd%addGradientDc()      # third-order SCC (if enabled)
│   └── ... (other optional contributions)
├── repulsive%getGradients()          # F_rep (always, if repulsive allocated)
├── dispersion%addGradients()         # D3/D4/MBD (if enabled)
└── derivs = sum of all contributions
```

Forces are output as `-derivs` (negated gradient = force).

---

## 2. How to Extract Force References from DFTB+

### 2.1 Using PrintForces

In `dftb_in.hsd`, add to Options:
```hsd
Options {
  WriteDetailedOut = Yes
  PrintForces = Yes
}
```

This triggers force calculation (`tForces = .true.`) and prints forces in `detailed.out`:
```
Total Forces
    1   -0.001234567890   0.002345678901   0.003456789012
    2    0.001234567890  -0.002345678901  -0.003456789012
```

Format: `I5, 3F20.12` (atom index, fx, fy, fz) in atomic units (Hartree/Bohr).

**Note:** Forces are `-derivs` (negated gradient).

### 2.2 Non-SCC Forces (charge-independent)

Run with `SCC = No` + `PrintForces = Yes`:
```hsd
Hamiltonian = DFTB {
  SCC = No
  ...
}
Options {
  WriteDetailedOut = Yes
  PrintForces = Yes
}
```

This gives: **F_nonSCC + F_rep** (no SCC contributions).

**Verification:** When `SCC = No`, `tSccCalc = .false.` in `initprogram.F90:1364`.
The code path at `main.F90:7132` takes the `if (.not. (tSccCalc .or. isExtField))` branch,
calling only `derivative_nonSCC`. No `addForceDc` is called. Repulsive is always added.

### 2.3 Full SCC Forces

Run with `SCC = Yes` + `PrintForces = Yes`:
```hsd
Hamiltonian = DFTB {
  SCC = Yes
  SCCTolerance = 1.0E-010
  MaxSCCIterations = 200
  ...
}
Options {
  WriteDetailedOut = Yes
  PrintForces = Yes
}
```

This gives: **F_nonSCC + F_SCC_shift + F_SCC_dc + F_rep** (full forces).

### 2.4 Alternative: MaxSCCIterations = 1

Setting `MaxSCCIterations = 1` with `SCC = Yes` would run one SCC iteration.
With initial charges q = q0 (deltaQ = 0), the SCC shifts are zero and addForceDc = 0.
So this should give the same as SCC = No, BUT the code path still goes through
`derivative_block` with zero shifts (which reduces to `derivative_nonSCC`).
This is equivalent but less clean than SCC = No.

### 2.5 Parsing Forces from detailed.out

```python
def parse_forces(work_dir):
    """Parse Total Forces from detailed.out."""
    path = os.path.join(work_dir, "detailed.out")
    with open(path) as f:
        text = f.read()
    forces = []
    in_forces = False
    for line in text.split('\n'):
        if 'Total Forces' in line:
            in_forces = True
            continue
        if in_forces:
            parts = line.split()
            if len(parts) == 4:
                try:
                    idx = int(parts[0])
                    fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                    forces.append([fx, fy, fz])
                except ValueError:
                    in_forces = False
            elif line.strip() == '':
                if forces:
                    break
    return forces
```

### 2.6 Parsing from results.tag

The `results.tag` file contains `force_tot` as a tagged array:
```
force_tot : real : -3, 2 :
  -0.001234567890  0.002345678901  0.003456789012
   0.001234567890 -0.002345678901 -0.003456789012
```

Shape is `(3, nAtom)`, values are `-derivs` (actual forces).

---

## 3. Implementation Plan for rust_dftb

### Phase 1: Non-SCC Forces (charge-independent)

**What's needed:**
1. Density matrix DM (already available from `SccResult.density`)
2. Energy-weighted density matrix EDM = 2 · C_occ · diag(eps_occ) · C_occ^T
3. Finite-difference derivatives dH0/dx and dS/dx for each atom pair
4. Repulsive pair potential derivative dE_rep/dr

**Steps:**

1. **Add EDM to SccResult** — compute during diagonalization:
   ```rust
   let edm = &c_occ * DMatrix::from_diagonal(&eps_occ) * c_occ.transpose() * 2.0;
   ```

2. **Implement finite-difference dH0/dx, dS/dx** — for each atom pair (i,j):
   - Displace atom j by ±delta along x, y, z
   - Rebuild diatomic block (interpolate SK + rotate)
   - Central difference: `(block(+) - block(-)) / (2·delta)`
   - This reuses existing `Rotation::rotate_diatomic_block_into()` and interpolation code
   - `delta = f64::EPSILON.powf(0.25)` ≈ 1.2e-4 (matching DFTB+)

3. **Implement repulsive force** — read spline repulsive from SK file:
   - Parse spline coefficients from SK file (currently not parsed in `sk_data.rs`)
   - Evaluate `dE_rep/dr` at each pair distance
   - `F_rep = dE_rep/dr * r_hat`

4. **Assemble non-SCC force:**
   ```
   for each pair (i,j):
     F_i += 2 * (sum(DM_ij · dH0/dx) - sum(EDM_ij · dS/dx))
     F_j -= same
     F_i += dE_rep/dr * r_hat
     F_j -= dE_rep/dr * r_hat
   ```

**Test:** Run DFTB+ with SCC=No, PrintForces=Yes, compare forces.

### Phase 2: SCC-Dependent Forces

**What's needed:**
1. Block-resolved SCC shifts (need to expose from SCC loop)
2. Gamma derivative `gamma'(r)` (not yet in `gamma.rs`)
3. 1/R² derivative for Coulomb part

**Steps:**

1. **Implement gamma'(r)** in `gamma.rs`:
   - `expGammaPrime(r, Ua, Ub)` — derivative of the short-range gamma
   - See Fortran: `@/home/prokop/git/dftbplus/src/dftbp/dftb/shortgammafuncs.F90:153`
   - For same-U case: polynomial × exp(-tau·r) derivative
   - For different-U case: `gammaSubExprnPrime_` for each term

2. **SCC double-counting force:**
   ```
   for each pair (i,j):
     gamma_r = gamma_full(r, U_i, U_j)
     gamma_prime = gamma_prime_full(r, U_i, U_j)  # d(gamma)/dr
     F_i += -deltaQ_i · deltaQ_j · gamma_prime · r_hat
     F_j += +deltaQ_i · deltaQ_j · gamma_prime · r_hat
   ```
   Note: `gamma_full = 1/r - short_range`, so `gamma_prime = -1/r² - short_range_prime`.
   The total is equivalent to `d(gamma_full)/dr`.

3. **SCC shift force (Pulay-like):**
   - Need block-resolved shifts: `shift_block[mu, nu, atom]` for each atom
   - Currently the Rust code uses atom-resolved shifts only
   - The shift force requires: `0.5 * (S'·shift_i + shift_j·S')` contracted with DM
   - This is more complex — may need to refactor SCC to expose block shifts

**Test:** Run DFTB+ with SCC=Yes, PrintForces=Yes, compare forces.

### Phase 3: Analytic Derivatives (optional optimization)

DFTB+ uses finite differences for dH0/dR and dS/dR. For the Rust implementation,
we could eventually implement analytic derivatives of the SK interpolation + rotation,
which would be faster and more accurate. But for parity, finite differences are sufficient.

---

## 4. Key Fortran Files for Forces

| File | Purpose | Key routines |
|------|---------|-------------|
| `src/dftbp/dftb/forces.F90` | Electronic force assembly | `derivative_nonSCC`, `derivative_block`, `derivative_iBlock` |
| `src/dftbp/dftb/nonscc.F90` | H0/S derivatives | `getFirstDerivBlock`, `getFirstDerivFiniteDiff`, `getFirstDerivRichardson` |
| `src/dftbp/dftb/scc.F90` | SCC dc force coordinator | `addForceDc` |
| `src/dftbp/dftb/shortgamma.F90` | Short-range gamma force | `addGradientsDc` |
| `src/dftbp/dftb/shortgammafuncs.F90` | Gamma math functions | `expGamma`, `expGammaPrime`, `gammaSubExprnPrime_` (different-U case) |
| `src/dftbp/dftb/coulomb.F90` | 1/R Coulomb force | `addGradients`, `addInvRPrimeCluster` |
| `src/dftbp/dftb/repulsive/repulsive.F90` | Abstract TRepulsive interface | `getGradients`, `getEnergy`, `getStress` (deferred procedures) |
| `src/dftbp/dftb/repulsive/repulsivecont.F90` | Container for multiple repulsives | `getGradients` (delegates to each repulsive) |
| `src/dftbp/dftb/repulsive/pairrepulsive.F90` | Abstract TPairRepulsive interface | `getValue` (returns energy + dEnergy + d2Energy) |
| `src/dftbp/dftb/repulsive/twobodyrep.F90` | Two-body repulsive implementation | `getGradients`, `getTwoBodyGradients_`, `getTwoBodyEnergy_` |
| `src/dftbp/dftb/repulsive/splinerep.F90` | Spline repulsive evaluation | `TSplineRep_getValue` (with `dEnergy`), `getSpline`, `getExponentialHead`, `getPolynomialTail` |
| `src/dftbp/dftb/repulsive/polyrep.F90` | Polynomial repulsive (alternative) | `TPolyRep_getValue` |
| `src/dftbp/dftb/repulsive/chimesrep.F90` | ChIMES repulsive (alternative) | `getGradients` |
| `src/dftbp/dftb/stress.F90` | Stress tensor (uses same derivative infra) | `derivative_nonSCC_stress`, `derivative_block_stress` |
| `src/dftbp/dftbplus/main.F90` | Main force driver | `getGradients` (line 6938), `postprocessDerivs`, `printMaxForces` |
| `src/dftbp/dftbplus/mainio.F90` | Force output | `writeDetailedOut4` (line 3471), `writeResultsTag` (line 2041), `writeMdOut2` |
| `src/dftbp/dftbplus/mainapi.F90` | Public API for forces | `getGradients` (line 184, wraps main%getGradients) |
| `src/dftbp/dftbplus/initprogram.F90` | Force flag initialization | `tPrintForces`, `tForces`, `tDerivs` (line 2016) |
| `src/dftbp/dftbplus/inputdata.F90` | Input data types | `TControl%tPrintForces` (line 279), `TControl%tForces` (line 270) |
| `src/dftbp/dftbplus/parser.F90` | HSD parser for force keywords | `PrintForces` (line 5350), `CalculateForces` (old name, line 870) |
| `src/dftbp/type/oldskdata.F90` | SK file parsing (repulsive spline) | Parses spline section from SK files |

---

## 5. Key Rust Files to Modify/Create

| File | Action | Content |
|------|--------|---------|
| `src/methods/dftb/gamma.rs` | Modify | Add `gamma_prime_full(r, U1, U2)` |
| `src/methods/dftb/hamiltonian.rs` | Modify | Add `compute_forces()` method, add EDM to SccResult |
| `src/methods/dftb/sk_data.rs` | Modify | Parse repulsive spline coefficients from SK file |
| `src/methods/dftb/forces.rs` | **Create** | Force computation module |
| `src/methods/dftb/mod.rs` | Modify | Add `pub mod forces;` |
| `tests/parity_forces.rs` | **Create** | Force parity tests (non-SCC and SCC) |
| `tests/run_forces.py` | **Create** | Python driver: XYZ → DFTB+ forces → Rust comparison |

---

## 6. Test Strategy

### 6.1 Non-SCC Force Parity

**Driver:** `tests/run_forces.py molecule.xyz --no-scc`

1. Generate DFTB+ input with `SCC = No`, `PrintForces = Yes`
2. Run DFTB+, parse forces from `detailed.out`
3. Run Rust `build_non_scc()`, diagonalize, compute DM + EDM, compute forces
4. Compare force vectors with tolerance ~1e-6

**Test file:** `tests/parity_forces.rs`
```rust
#[test]
fn non_scc_forces_from_xyz() {
    // Env: RUST_DFTB_SK_DIR, RUST_DFTB_FORCES_XYZ, RUST_DFTB_FORCES_REF
    // 1. Build H0, S
    // 2. Diagonalize: H0·C = S·C·eps
    // 3. Compute DM, EDM
    // 4. Compute F_nonSCC + F_rep
    // 5. Compare with Fortran reference
}
```

### 6.2 Full SCC Force Parity

**Driver:** `tests/run_forces.py molecule.xyz --scc`

1. Generate DFTB+ input with `SCC = Yes`, `PrintForces = Yes`
2. Run DFTB+, parse forces from `detailed.out`
3. Run Rust `build_scc()`, compute all force contributions
4. Compare with tolerance ~1e-5

### 6.3 Test Molecules

Start with small molecules where forces are non-zero (asymmetric geometries):
- H2O (bent, asymmetric)
- HCN (linear, asymmetric)
- CH4 (tetrahedral, but use distorted geometry)
- HCOOH (asymmetric)

**Important:** For symmetric molecules at equilibrium, forces are zero — use distorted geometries
or non-equilibrium bond lengths to get non-trivial force values.

---

## 7. Notes on the Repulsive Potential

The repulsive potential is stored in the SK file as spline coefficients.
Currently `sk_data.rs` in the Rust code parses the SK tables (H, S integrals) but
may NOT parse the repulsive spline section.

**SK file structure (heteronuclear):**
1. Line 1: grid spacing, nGrid
2. Line 2: polynomial coefficients for repulsive (5 coefficients)
3. Lines 3+: SK integral tables (H, S)
4. Optional: spline repulsive section (starts with `Spline` keyword)

**SK file structure (homonuclear):**
1. Line 1: grid spacing, nGrid
2. Line 2: atomic eigenvalues, Hubbard U, occupations
3. Lines 3+: SK integral tables
4. Optional: spline repulsive section

The spline repulsive section format:
```
Spline [nSpline]
xStart(1) xStart(2) ... xStart(nSpline)
spCoeffs(1,1) spCoeffs(2,1) spCoeffs(3,1) spCoeffs(4,1)
...
spLastCoeffs(1) ... spLastCoeffs(6)
expCoeffs(1) expCoeffs(2) expCoeffs(3)
cutoff
```

See `@/home/prokop/git/dftbplus/src/dftbp/type/oldskdata.F90` for parsing details,
and `@/home/prokop/git/dftbplus/src/dftbp/dftb/repulsive/splinerep.F90` for evaluation.
The abstract interface is in `@/home/prokop/git/dftbplus/src/dftbp/dftb/repulsive/pairrepulsive.F90`,
and the container that manages multiple repulsive contributions is in
`@/home/prokop/git/dftbplus/src/dftbp/dftb/repulsive/repulsivecont.F90`.

---

## 8. Summary: What to Extract from Fortran DFTB+

| Force component | Fortran source | What to extract |
|----------------|---------------|-----------------|
| F_nonSCC | `forces.F90:derivativeNonSccEuclidian` | Formula: `2*(DM·dH0' - EDM·dS')`, finite diff for dH0'/dS' |
| F_SCC_shift | `forces.F90:derivative_blockEuclidean` | Formula: `+2*shiftSprime·DM`, needs block shifts |
| F_SCC_dc (gamma) | `shortgamma.F90:addGradientsDc` | Formula: `-dQ_i·dQ_j·gamma'(r)·r_hat` |
| F_SCC_dc (Coulomb) | `coulomb.F90:addInvRPrimeCluster` | Formula: `-dQ_i·dQ_j/r³·r_vec` |
| F_rep | `twobodyrep.F90:getTwoBodyGradients_` | Formula: `dE_rep/dr·r_hat`, spline from SK file |
| Force output | `mainio.F90:writeDetailedOut4` | Parse "Total Forces" from detailed.out |
| Force trigger | `parser.F90:5350` | `PrintForces = Yes` in Options block |
