CODEMAPS:

tblite GFN1-xTB Implementation: Calculator Construction, Hamiltonian Assembly, and SCF Convergence
https://windsurf.com/codemaps/4038e764-8ad4-4bce-83b4-5bf48fd6bbc3-fe86ab10a43f3d18

tblite xTB Implementation: GFN1 Hamiltonian Construction and C API
https://windsurf.com/codemaps/37991fa2-164c-442e-a750-ac463aca56d6-fe86ab10a43f3d18


---

# USER

could you plese review very carefully the reference tblight so we can try to reimplement this in rust?

then wer should also adjust the tests so we can checkcthe hamiltonian party wrt tblight. But we do not have access to tblight internal, maybe we must create libtblightcore or somethinbg like that?

something similar to @libdftbcore.F90@DFTBcore.py ?

OK I prefer optionm A WITH_API TRUE, but hopefully it will work, for dftb+ it was not working.

I do not care about dispersion or other corrections right now...

I wonder that xTB is more complex than DFTB+ I was thinking the opposite.

Anyways make detailed review note all relevant files and function we need to implement non-SCC and SCC part of it. list the relevant files and functions here in chat which short single sentece descripotion of the purpose and maybe some secheme how they are interacting, we will use it to generate code map

---

# Kimi-2.6

Here is the comprehensive review of tblite's xTB implementation based on my reading of the source code:

## tblite xTB Code Review — Files, Functions & Interaction Scheme

### Architecture Overview

xTB is **more complex than DFTB+** in several key ways:
- **Integrals:** DFTB+ uses pre-tabulated Slater-Koster tables (lookup); xTB computes analytical CGTO integrals on-the-fly
- **Self-energy:** xTB has CN-dependent (`kcn`) and charge-dependent (`kq1`, `kq2`) shifts; DFTB+ has fixed onsite values
- **Basis:** xTB uses contracted Gaussian type orbitals (CGTOs) derived from STO exponents; DFTB+ uses implicit valence orbitals
- **Electrostatics:** xTB is shell-resolved with multipoles (dipole, quadrupole); DFTB+ is atom-resolved γ-function only
- **Hamiltonian scaling:** xTB uses Pauling EN differences and valence/core distinctions; DFTB+ uses fixed SK tables

### 1. PARAMETER TABLES (Static Data)

| File | Key Content | Purpose |
|------|-------------|---------|
| [xtb/gfn1.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/gfn1.f90:0:0-0:0) | `p_selfenergy`, `p_kcn`, `p_shpoly`, `slater_exponent`, `nshell`, `ang_shell`, `hubbard_parameter`, `shell_hubbard`, `p_hubbard_derivs`, `rep_alpha`, `rep_zeff`, `halogen_bond` | **GFN1-xTB parameters** — 86 elements, max 3 shells (s,p,d), all hardcoded static arrays |
| `xtb/gfn2.f90` | Same structure as GFN1 but different values | **GFN2-xTB parameters** — uses Klopman-Ohno instead of γ-function for electrostatics |
| `xtb/ipea1.f90` | Same structure | **IPEA1-xTB parameters** — tuned for ionization potentials/electron affinities |

**Key GFN1 parameters:**
- `p_selfenergy(max_shell, max_elem)` — atomic level energies in eV (converted to Hartree)
- `p_kcn(max_shell, max_elem)` — CN-dependent shift coefficients
- `p_shpoly(0:2, max_elem)` — polynomial parameters for distance-dependent Hamiltonian scaling
- `slater_exponent(max_shell, max_elem)` — STO exponents for CGTO construction
- `hubbard_parameter(max_elem)` — atomic hardnesses
- `shell_hubbard(0:2, max_elem)` — shell-dependent scaling of hardness
- `kshell(k,l)` — shell-pair Hamiltonian scaling matrix (s-s, s-p, p-p, etc.)

### 2. BASIS SET CONSTRUCTION

| File | Key Types/Functions | Purpose |
|------|---------------------|---------|
| `basis/type.f90` | `cgto_type` (ang, nprim, alpha(12), coeff(12)), `basis_type` (nsh, nao, cgto(:,:), mappings) | Defines contracted Gaussian basis and shell/AO indexing |
| [basis/slater.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/basis/slater.f90:0:0-0:0) | `slater_to_gauss(ng, pqn, l, zeta, cgto, norm, stat)` | Converts STO to CGTO using Stewart's STO-NG coefficients |
| `basis/ortho.f90` | `orthogonalize(cgto1, cgto2)` | Orthogonalizes shells with same angular momentum (e.g. core vs valence) |

**Basis construction flow (from `calculator.f90:add_basis`):**
1. For each element, get `nshell`, `ang_shell`, `number_of_primitives`, `slater_exponent`
2. Call `slater_to_gauss` for each shell to create `cgto_type`
3. Call `orthogonalize` for duplicate angular momenta (e.g. 2s/3s both with l=0)
4. Call `new_basis` to build `basis_type` with all index mappings (`sh2at`, `ao2sh`, `iao_sh`, etc.)

### 3. INTEGRAL EVALUATION

| File | Key Functions | Purpose |
|------|--------------|---------|
| [integral/overlap.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/integral/overlap.f90:0:0-0:0) | `overlap_cgto(cgto_i, cgto_j, r2, vec, intcut, stmp, dtmp, qtmp)` | Computes overlap + dipole + quadrupole integrals for a CGTO pair |
| [integral/multipole.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/integral/multipole.f90:0:0-0:0) | `multipole_cgto(...)` | Same as overlap but with full multipole output |
| [integral/dipole.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/integral/dipole.f90:0:0-0:0) | `dipole_cgto(...)` | Dipole-specific integrals |
| [integral/trafo.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/integral/trafo.f90:0:0-0:0) | `transform0`, `transform1`, `transform2` | Cartesian-to-spherical harmonic transformations |

**`overlap_cgto` / `multipole_cgto` signature:**
```fortran
subroutine multipole_cgto(cgto_i, cgto_j, r2, vec, intcut, &
     & overlap, dipole, quadrupole)
```
- `cgto_i`, `cgto_j` — the two CGTOs
- `r2` — squared distance between centers
- `vec(3)` — displacement vector (j - i)
- `intcut` — integral cutoff threshold
- Outputs: `overlap(msao)**2`, `dipole(3, msao**2)`, `quadrupole(6, msao**2)`

These are **analytical** Gaussian product integrals — no lookup tables.

### 4. HAMILTONIAN SPECIFICATION

| File | Key Types/Functions | Purpose |
|------|---------------------|---------|
| `xtb/spec.f90` | `tb_h0spec` (abstract) | Abstract parameter generator for Hamiltonian components |
| [xtb/h0.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/h0.f90:0:0-0:0) | `tb_hamiltonian`, `new_hamiltonian`, `get_selfenergy`, `get_hamiltonian` | Core Hamiltonian builder |

**`tb_h0spec` abstract methods:**
- `get_selfenergy(mol, bas, selfenergy)` — atomic level energies per shell
- `get_cnshift(mol, bas, kcn)` — CN-dependent shift coefficients
- `get_q1shift(mol, bas, kq1)` — linear charge dependence
- `get_q2shift(mol, bas, kq2)` — quadratic charge dependence
- `get_hscale(mol, bas, hscale)` — shell-pair scaling factors
- `get_rad(mol, bas, rad)` — atomic radii for distance scaling
- `get_shpoly(mol, bas, shpoly)` — polynomial coefficients
- `get_reference_occ(mol, bas, refocc)` — reference occupations

**`tb_hamiltonian` type stores:**
- `selfenergy(mshell, nid)` — base atomic levels
- `kcn(mshell, nid)` — CN dependence
- `kq1(mshell, nid)`, `kq2(mshell, nid)` — charge dependence
- `hscale(mshell, mshell, nid, nid)` — shell-pair scaling
- `shpoly(mshell, nid)` — polynomial parameters
- `rad(nid)` — atomic radii
- `refocc(mshell, nid)` — reference occupations

### 5. HAMILTONIAN CONSTRUCTION

| File | Key Function | Purpose |
|------|--------------|---------|
| [xtb/h0.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/h0.f90:0:0-0:0) | `get_hamiltonian(mol, trans, list, bas, h0, selfenergy, overlap, dpint, qpint, hamiltonian)` | Builds H₀ and S matrices |

**`get_hamiltonian` algorithm:**
1. Loop over atom pairs (iat, jat) via neighbor list
2. For each shell pair (ish, jsh):
   a. Call `multipole_cgto` → overlap, dipole, quadrupole integrals
   b. Compute `shpoly = (1 + shpoly_i*rr) * (1 + shpoly_j*rr)` where `rr = sqrt(r)/sum(rad)`
   c. Compute `hij = 0.5 * (selfenergy_i + selfenergy_j) * hscale * shpoly`
   d. Accumulate: `H += overlap * hij`, `S += overlap`
   e. Accumulate dipole/quadrupole integrals
3. Loop over atoms for onsite terms (iat == jat)

**Self-energy formula:**
```
ε_ish = selfenergy(ish, izp) - kcn(ish, izp)*CN(iat) - kq1(ish, izp)*q(iat) - kq2(ish, izp)*q(iat)²
```

### 6. COULOMB / SCC ELECTROSTATICS

| File | Key Types/Functions | Purpose |
|------|---------------------|---------|
| [coulomb/charge/type.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/coulomb/charge/type.f90:0:0-0:0) | `coulomb_charge_type` (abstract) | Base class for isotropic electrostatics |
| [coulomb/charge/gamma.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/coulomb/charge/gamma.f90:0:0-0:0) | `gamma_coulomb`, `new_gamma_coulomb`, `get_coulomb_matrix` | **DFTB γ-functional** — used by GFN1 |
| [coulomb/charge/effective.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/coulomb/charge/effective.f90:0:0-0:0) | `effective_coulomb`, `new_effective_coulomb` | **Klopman-Ohno** — used by GFN2 |
| `coulomb/multipole.f90` | `damped_multipole`, `new_damped_multipole` | Anisotropic dipole/quadrupole interactions |
| `coulomb/thirdorder.f90` | `onsite_thirdorder`, `new_onsite_thirdorder` | Third-order Hubbard corrections |
| [xtb/coulomb.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/coulomb.f90:0:0-0:0) | `tb_coulomb` (container) | Combines es2 + aes2 + es3 into one object |

**GFN1 uses:** `effective_coulomb` (Klopman-Ohno with harmonic averaging) + `onsite_thirdorder`
**GFN2 uses:** `gamma_coulomb` (DFTB γ) + `damped_multipole` (multipole interactions)

### 7. COORDINATION NUMBER

| File | Key Function | Purpose |
|------|--------------|---------|
| [ncoord/gfn.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/ncoord/gfn.f90:0:0-0:0) | `new_gfn_ncoord`, `get_coordination_number` | Double-exponential counting function for CN |

**CN formula:**
```
CN_i = Σ_j 1 / (1 + exp(-ka*(r_ij/rcov - 1))) * 1 / (1 + exp(-kb*(r_ij/rcov - r_shift)))
```
- `ka = 10.0`, `kb = 20.0`, `r_shift = 2.0`
- Used to modify self-energies: `ε_ish -= kcn(ish) * CN_i`

### 8. SCF LOOP

| File | Key Function | Purpose |
|------|--------------|---------|
| [scf/iterator.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/scf/iterator.f90:0:0-0:0) | `next_scf(iscf, mol, bas, wfn, solver, mixer, info, coulomb, dispersion, interactions, ints, pot, cache, dcache, icache, energies, error)` | **One SCF iteration** |
| [scf/potential.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/scf/potential.f90:0:0-0:0) | `potential_type`, `new_potential`, `add_pot_to_h1` | Container for charge-dependent potential shifts |
| `scf/mixer/*.f90` | Broyden mixer, etc. | Convergence acceleration |
| [scf/solver.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/scf/solver.f90:0:0-0:0) | `solver_type` | LAPACK eigensolver wrapper |

**`next_scf` steps:**
1. Compute Mulliken charges from density matrix → `qat`, `qsh`
2. Compute atomic multipoles from density → `dpat`, `qpat`
3. Call `coulomb%get_potential(mol, cache, wfn, pot)` → charge-dependent shifts
4. Call `add_pot_to_h1(ints%hamiltonian, pot)` → apply shifts to H
5. Solve generalized eigenvalue problem: `H·c = E·S·c`
6. Update occupations (Fermi smearing if `kt > 0`)
7. Build new density matrix
8. Mixer combines old/new density

### 9. WAVEFUNCTION

| File | Key Type/Functions | Purpose |
|------|-------------------|---------|
| [wavefunction/type.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/wavefunction/type.f90:0:0-0:0) | `wavefunction_type` (density, coeff, emo, focc, qat, qsh, dpat, qpat) | Stores all density-related quantities |
| `wavefunction/mulliken.f90` | `get_mulliken_shell_charges`, `get_mulliken_atomic_multipoles` | Population analysis |
| `wavefunction/fermi.f90` | `get_fermi_filling` | Fermi-Dirac occupation smearing |

### 10. SINGLE POINT CALCULATION (Main Entry Point)

| File | Key Function | Purpose |
|------|--------------|---------|
| [xtb/singlepoint.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/singlepoint.f90:0:0-0:0) | `xtb_singlepoint(ctx, mol, calc, wfn, accuracy, energy, gradient, sigma, verbosity, results)` | **Main driver** |

**`xtb_singlepoint` flow:**
1. Compute repulsion, halogen, dispersion energies (non-SCF, geometry-dependent)
2. `new_potential(pot, mol, calc%bas, wfn%nspin)`
3. `get_occupation` → reference occupations → number of electrons
4. `get_selfenergy(calc%h0, ...)` with CN if available
5. `get_hamiltonian(...)` → builds H₀, S, dipole, quadrupole integrals
6. SCF loop: `while(not converged) call next_scf(...)`
7. Save results (overlap, Hamiltonian, bond orders) if requested

### 11. CALCULATOR CONSTRUCTION

| File | Key Function | Purpose |
|------|--------------|---------|
| [xtb/calculator.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/calculator.f90:0:0-0:0) | `new_xtb_calculator(calc, mol, param, error)` | Builds calculator from param records |
| [xtb/gfn1.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/gfn1.f90:0:0-0:0) | `new_gfn1_calculator(calc, mol)` | Hardcoded GFN1 constructor |

**`new_xtb_calculator` builds:**
1. `add_basis` → CGTOs from param records
2. `add_ncoord` → coordination number evaluator
3. `add_hamiltonian` → `tb_hamiltonian` from `param_h0spec`
4. `add_repulsion` → pairwise repulsion
5. `add_halogen` → halogen bonding correction
6. `add_dispersion` → D3/D4 dispersion
7. `add_coulomb` → electrostatics (effective/γ + multipole + thirdorder)

---

### 12. C API (For Parity Testing)

| File | Key Functions | Purpose |
|------|--------------|---------|
| `include/tblite/calculator.h` | `tblite_new_gfn1_calculator`, `tblite_new_gfn2_calculator`, `tblite_set_calculator_save_integrals` | Create calculator, enable integral storage |
| `include/tblite/result.h` | `tblite_get_result_overlap_matrix`, `tblite_get_result_hamiltonian_matrix`, `tblite_get_result_density_matrix`, `tblite_get_result_orbital_coefficients`, `tblite_get_result_charges` | Extract matrices for parity testing |
| `include/tblite/calculator.h` | `tblite_get_singlepoint` | Run calculation |

## Interaction Scheme (Flow Diagram)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   gfn1.f90      │────▶│  calculator.f90  │────▶│   basis/type    │
│  (parameters)     │     │ (build calc)     │     │  (cgto, basis)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │
         │              ┌────────┴────────┐
         │              │                 │
         │         ┌────▼────┐      ┌────▼────┐
         │         │  h0.f90 │      │ ncoord  │
         │         │(Ham spec)│      │  (CN)   │
         │         └────┬────┘      └────┬────┘
         │              │                 │
         │              └────────┬────────┘
         │                       │
         │              ┌────────▼────────┐
         │              │  singlepoint.f90│
         │              │  (main driver)  │
         │              └────────┬────────┘
         │                       │
    ┌────┴────┐         ┌────────▼────────┐      ┌─────────────┐
    │coulomb  │◀────────│    scf loop     │◀────▶│  scf/iterator │
    │(es2,es3)│         │  next_scf()     │      │  (mixer)      │
    └─────────┘         └────────┬────────┘      └─────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌────▼────┐  ┌───▼────┐  ┌────▼────┐
              │ integral │  │wavefn  │  │potential │
              │ overlap  │  │mulliken│  │add_pot   │
              │multipole │  │fermi   │  │to H      │
              └─────────┘  └────────┘  └─────────┘
```

## For Rust Implementation: Minimum Viable xTB

To implement **non-SCC xTB** in Rust, you need:

1. **`params.rs`** — Hardcode GFN1 parameters for H, C, N, O (selfenergy, kcn, shpoly, slater, hubbard, shell_hubbard, hscale matrix)
2. **`basis.rs`** — `Cgto` type + `slater_to_gauss` (STO-NG coefficients from Stewart 1970)
3. **`integrals.rs`** — `overlap_cgto` (analytical Gaussian overlap + dipole + quadrupole)
4. **`hamiltonian.rs`** — `XtbBuilder` implementing `H0Builder`:
   - Build basis from elements
   - Compute CN if needed
   - Compute selfenergy = p_selfenergy - kcn*CN
   - Loop over shell pairs, call `multipole_cgto`, scale by `hscale * shpoly`
   - Return H and S

To add **SCC**, also need:

5. **`coulomb.rs`** — Implement `CoulombModel`:
   - GFN1: `effective_coulomb` (Klopman-Ohno with harmonic average)
   - GFN2: `gamma_coulomb` (DFTB γ-function)
   - Third-order: `onsite_thirdorder`
6. **`solver.rs`** — Generic SCF loop (reuse existing `qmqm/` infrastructure)

## For Parity Testing via C API

**Steps:**
1. Change [external/tblite/CMakeLists.txt](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/CMakeLists.txt:0:0-0:0): `set(WITH_API TRUE)`
2. Rebuild tblite (or full DFTB+)
3. Create Python ctypes wrapper or Rust FFI bindings
4. Call sequence:
   ```c
   tblite_context ctx = tblite_new_context();
   tblite_structure mol = tblite_new_structure(ctx, nat, nums, xyz, ..., 0);
   tblite_calculator calc = tblite_new_gfn1_calculator(ctx, mol);
   tblite_set_calculator_save_integrals(ctx, calc, 1);
   tblite_result res = tblite_new_result();
   tblite_get_singlepoint(ctx, mol, calc, res);
   
   // Extract matrices for parity
   tblite_get_result_overlap_matrix(error, res, smat);
   tblite_get_result_hamiltonian_matrix(error, res, hmat);
   tblite_get_result_density_matrix(error, res, pmat);
   tblite_get_result_charges(error, res, charges);
   ```

**Next step:** Want me to try enabling `WITH_API TRUE` and rebuilding tblite, or start implementing the Rust xTB module?

---

# USER

yes plese run it with WITH_API TRUE try if it compiles and run, figure out what fine-griend information about SCC and non-SCC components of hamiltonian we can get. can we get sparse form? or individual blocks for pairs of atoms ?

---

# Kimi-2.6

## C API Data Extraction — What Is Actually Available

### What `save_integrals` Gives You

**Mechanism:**
- C API: `tblite_set_calculator_save_integrals(ctx, calc, 1)`
- Fortran: sets `calc%ptr%save_integrals = .true.`
- In [singlepoint.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/singlepoint.f90:0:0-0:0) line 325:
  ```fortran
  if (calc%save_integrals .and. present(results)) then
     call move_alloc(ints%overlap, results%overlap)
     call move_alloc(ints%hamiltonian, results%hamiltonian)
  end if
  ```
- These are then exposed via C API in [api/result.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/api/result.f90:0:0-0:0):
  - `tblite_get_result_overlap_matrix(error, res, smat)` → dense `nao × nao`
  - `tblite_get_result_hamiltonian_matrix(error, res, hmat)` → dense `nao × nao`

### Critical Finding: Saved Hamiltonian is **NON-SCC H₀ ONLY**

The `hamiltonian` saved is the one built by `get_hamiltonian()` in [xtb/h0.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/h0.f90:0:0-0:0). It uses:
- `selfenergy = p_selfenergy - kcn*CN` (CN-dependent but charge-independent)
- `hscale` (Pauling EN-based shell-pair scaling)
- `shpoly` (distance-dependent polynomial scaling)
- `overlap_cgto` / `multipole_cgto` integrals

The **SCC potential is added LATER** in `next_scf()` via `add_pot_to_h1(ints%hamiltonian, pot)`. This modifies H in-place during each SCF iteration. The saved matrix is the **original H₀ before SCC**.

**Consequence for parity testing:**
- We can directly compare our Rust `H₀` (non-SCC) against tblite's saved `hamiltonian`
- We **cannot** directly get the final SCC-modified H from tblite via C API
- For SCC parity, we must either:
  a. Reconstruct H_scc from our Rust code and compare eigenvalues/density/charges
  b. Or extract tblite's final density matrix and charges, then verify our H_scc produces the same

### Full List of C API Extractable Quantities

From [include/tblite/result.h](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/include/tblite/result.h:0:0-0:0) and [api/result.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/api/result.f90:0:0-0:0):

| Function | Data | Notes |
|----------|------|-------|
| `tblite_get_result_overlap_matrix` | `S` (nao×nao) | Full dense |
| `tblite_get_result_hamiltonian_matrix` | `H₀` (nao×nao) | **Non-SCC only** |
| `tblite_get_result_density_matrix` | `P` (nspin×nao×nao) | Final converged density |
| `tblite_get_result_charges` | `qat(nat)` | Atomic Mulliken charges |
| `tblite_get_result_orbital_energies` | `emo(nao)` | Eigenvalues |
| `tblite_get_result_orbital_coefficients` | `coeff(nspin×nao×nao)` | MO coefficients |
| `tblite_get_result_bond_orders` | `mbo(nat×nat×nspin)` | Mayer bond orders |
| `tblite_get_result_energies` | `energies(nat)` | Atom-resolved energies |
| `tblite_get_result_dipole` | `dipole(3)` | Molecular dipole |
| `tblite_get_result_quadrupole` | `quadrupole(6)` | Traceless quadrupole |

### Basis Metadata for Matrix Slicing

From [include/tblite/calculator.h](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/include/tblite/calculator.h:0:0-0:0) — we CAN get the basis structure:

| Function | Purpose |
|----------|---------|
| `tblite_get_calculator_shell_count` | Number of shells `nsh` |
| `tblite_get_calculator_shell_map` | `sh2at(nsh)` — which atom each shell belongs to |
| `tblite_get_calculator_angular_momenta` | `am(nsh)` — angular momentum of each shell |
| `tblite_get_calculator_orbital_count` | Number of AOs `nao` |
| `tblite_get_calculator_orbital_map` | `ao2sh(nao)` — which shell each AO belongs to |

**With these mappings, we CAN extract atom-pair blocks ourselves** by slicing the full dense matrix. No sparse form is natively provided.

### What Is **NOT** Available via C API

- **No sparse matrices** — everything is dense `nao × nao`
- **No atom-pair blocks directly** — must slice using shell_map + orbital_map
- **No CN values** — coordination numbers are internal only
- **No shell selfenergies** — `selfenergy(nsh)` is internal
- **No SCC potential** — the potential shifts added to H are not exposed
- **No multipole integrals separately** — dipole/quadrupole integrals computed in `get_hamiltonian()` are not saved or exposed
- **No shell-resolved charges** — only atomic charges `qat` are exposed (not `qsh`)

### Recommended Parity Testing Strategy

**Non-SCC parity (direct H₀ comparison):**
1. `tblite_set_calculator_save_integrals(ctx, calc, 1)`
2. `tblite_get_singlepoint(ctx, mol, calc, res)`
3. `tblite_get_result_hamiltonian_matrix(res, hmat)` → compare with Rust `H₀`
4. `tblite_get_result_overlap_matrix(res, smat)` → compare with Rust `S`

**SCC parity (indirect):**
1. Run tblite singlepoint (normal, no need for save_integrals)
2. `tblite_get_result_charges(res, charges)` → compare with Rust Mulliken charges
3. `tblite_get_result_orbital_energies(res, emo)` → compare eigenvalues
4. `tblite_get_result_density_matrix(res, pmat)` → compare density matrix
5. For H_scc: our Rust code computes H_scc = H₀ + V_scc. Since we can't get V_scc from tblite, we verify by checking that diagonalizing our H_scc with our S produces the same eigenvalues/vectors.

## Next Step: Enable WITH_API TRUE and Build

Now I will modify the tblite CMakeLists.txt and attempt to compile. However, I notice there is **no build directory** in the workspace. I need to check how the project is normally built.

**Issue:** I'm in Ask mode, so I cannot directly edit files or run commands. I need to tell the user to switch to Code mode, or ask them to make the change and run the build themselves.

Let me provide the exact change needed and a verification command.### Summary of C API Findings (before attempting build):

**Key discovery:** The saved `hamiltonian` from `save_integrals` is the **non-SCC H₀ only** — the SCC potential is added later in the SCF loop and is not exposed. This is actually good for us: we can do direct H₀ parity.

**We CAN get:**
- Full dense `S`, `H₀`, `P`, MO coeffs, eigenvalues, charges, bond orders
- Basis metadata (shell→atom, AO→shell, angular momenta) to slice matrices into atom blocks ourselves

**We CANNOT get:**
- Sparse matrices, individual atom-pair blocks, CN values, shell selfenergies, SCC potential shifts, or separate dipole/quadrupole integrals

### Build Attempt

There is **no existing build directory** in the workspace. To proceed, I need you to either:

1. **Switch to Code mode** so I can edit [external/tblite/CMakeLists.txt](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/CMakeLists.txt:0:0-0:0) and run the build commands, OR
2. **Run the following yourself:**

Change line 10 in [/home/prokophapala/git/dftbplus/external/tblite/CMakeLists.txt](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/CMakeLists.txt:0:0-0:0):
```cmake
# Old:
set(WITH_API FALSE)
# New:
set(WITH_API TRUE)
```

Then configure and build:
```bash
cd /home/prokophapala/git/dftbplus
mkdir -p build && cd build
cmake .. -DWITH_TBLITE=TRUE  [plus your other config options]
make -j$(nproc)
```

Which do you prefer — should you switch to Code mode, or will you run the build manually?