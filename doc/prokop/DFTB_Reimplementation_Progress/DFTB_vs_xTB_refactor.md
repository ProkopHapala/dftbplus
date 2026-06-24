

# USER

read @NonCovalent_DFTB_Approx_xTB_Fukui.md and figure out how we can implement such system using dftb+ in current codebase and using what we have in rust @Testing_and_Parity_Codemap.md @rust_QMQM.md 

do we have there the xTB hamiltonian with the analytical expressions somewhere?

---

# Kimi-2.6

## xTB Hamiltonian with analytical expressions — YES, it exists

The xTB Hamiltonian is implemented in the `tblite` submodule at:

- `@/home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/h0.f90:157-299` — `get_hamiltonian`
- `@/home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/gfn1.f90` — GFN1-xTB parameters

**The analytical form** (from [h0.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/h0.f90:0:0-0:0)) is:

```fortran
hij = 0.5_wp * (selfenergy_i + selfenergy_j) * hscale(i,j) * shpoly(i,j)
H_μν = S_μν * hij
```

Where:
- `S_μν` — analytical overlap integrals from contracted Gaussian-type orbitals (`multipole_cgto` / `overlap_cgto`)
- `selfenergy` — shell-resolved atomic level: `ε = base - kcn*CN - kq1*q - kq2*q²`
- `shpoly = (1 + p_shpoly_i * rr) * (1 + p_shpoly_j * rr)` with `rr = √(R / (rad_i + rad_j))`
- `hscale` — pairwise shell scaling factors (includes electronegativity difference scaling in GFN2)

This is **not** Slater–Köster table interpolation. Everything is computed on-the-fly from STO exponents and the parameters in [gfn1.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/gfn1.f90:0:0-0:0).

## How to implement the non-covalent / xTB / Fukui system

You have **two paths**, depending on whether you want to stay inside DFTB+ or use your Rust QM/QM solver.

### Path A: Use DFTB+ directly (for reference / validation)

DFTB+ already supports xTB via the `tblite` interface:

```fortran
@/home/prokophapala/git/dftbplus/src/dftbp/dftbplus/main.F90:1246-1252
case(hamiltonianTypes%xtb)
  call this%tblite%buildSH0(env, this%species, this%coord, ...)
```

- Set `Hamiltonian = xTB` in the HSD input
- `buildSH0` calls `tblite` to build H0 and S analytically
- SCC proceeds normally via DFTB+’s `scc.F90`

For extracting the Fukui function, you would need to post-process the eigenvectors from DFTB+ output (or patch [main.F90](cci:7://file:///home/prokophapala/git/dftbplus/src/dftbp/dftbplus/main.F90:0:0-0:0) to print HOMO/LUMO coefficients).

### Path B: Use your Rust QM/QM solver (recommended for speed)

Your [rust_dftb/src/qmqm/](cci:9://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm:0:0-0:0) module is **already architected for exactly this problem**. Status check:

| Component | File | Status |
|-----------|------|--------|
| [GammaTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:108:0-115:1) + [gamma_full()](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:29:0-66:1) | [gamma.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:0:0-0:0) | **Complete** — DFTB γ(R,U₁,U₂) with exp-screening |
| [Fragment](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/fragment.rs:104:0-129:1) + [FragmentTemplate](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/fragment.rs:27:0-52:1) | [fragment.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/fragment.rs:0:0-0:0) | **Complete** — owns H0, S, diagonalizes via Cholesky |
| [compute_intra_shifts](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/shifts.rs:14:0-51:1) | [shifts.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/shifts.rs:0:0-0:0) | **Complete** — intra-fragment SCC shifts |
| [MultiSystemSolver::solve_scc()](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:246:4-296:5) | [solver.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:0:0-0:0) | **Complete** — global Jacobi SCC loop with V_ext |
| `DIIS` / `SimpleMixer` | [mixer.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/mixer.rs:0:0-0:0) | **Complete** |
| `FragmentNeighborList` | [neighbor.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/neighbor.rs:0:0-0:0) | **Complete** — cell-list spatial hashing |

**The current Rust solver uses DFTB-style SK tables for intra-fragment H0**, not xTB analytical integrals. For a multi-fragment non-covalent solver, this is actually fine — you get exact fragment quantum mechanics and the inter-fragment coupling is already correct γ-electrostatics.

## What you actually need for the Fukui/AFM model

From the notes in [NonCovalent_DFTB_Approx_xTB_Fukui.md](cci:7://file:///home/prokophapala/git/dftbplus/doc/prokop/NonCovalent_DFTB_Approx_xTB_Fukui.md:0:0-0:0), the fast AFM interaction model uses three atomic descriptors per fragment:

- **χ** (electronegativity) = chemical potential = `-dE/dN` ≈ orbital energy of HOMO/LUMO midpoint
- **η** (hardness) = Hubbard U parameter (already in `GammaTable::hubbard_u`)
- **f** (Fukui function) = frontier orbital density condensed per atom

### Concrete implementation plan

1. **Keep the existing Rust [qmqm](cci:9://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm:0:0-0:0) solver** for fragment-level DFTB. The SK-based H0 is accurate enough for fragments.

2. **Extract descriptors after diagonalization** ([fragment.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/fragment.rs:0:0-0:0)):
   ```rust
   // After frag.diagonalize():
   let homo_idx = n_orbs - n_occ - 1;
   let lumo_idx = homo_idx + 1;
   // f^-(A) = Σ_{μ∈A} c_μ,HOMO * (D·S)_μμ  (simplified)
   // χ_A ≈ -(ε_HOMO + ε_LUMO)/2  (fragment-averaged, or per-atom from shift)
   // η_A = GammaTable::u(species_A)
   ```

3. **For the fast pairwise AFM force field**, bypass the full SCC loop and use the analytical formula from the notes:
   ```
   E_int(R) = E_rep(R) + E_vdW(R) - ½ (χ_T - χ_S)² / (η_T + η_S - 2γ(R)) · (f_T · f_S)
   ```
   This uses the same [gamma_full()](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:29:0-66:1) already in [qmqm/gamma.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:0:0-0:0).

4. **If you truly need xTB intra-fragment H0 in Rust** (no SK files), you would need to port:
   - `overlap_cgto` (contracted Gaussian overlap integrals)
   - `multipole_cgto` (dipole/quadrupole integrals)
   - `get_selfenergy` with CN and charge dependence
   - `new_hamiltonian` parameter loader for GFN1/2

   This is doable but a few hundred lines of analytical integral code. The `tblite` Fortran source is the reference.

## Bottom line

- **xTB analytical Hamiltonian exists** in `external/tblite/`. DFTB+ calls it via `buildSH0` when `Hamiltonian = xTB`.
- **Your Rust QM/QM solver is ready** for multi-fragment non-covalent calculations with γ-electrostatic coupling. The SCC loop, mixer, and neighbor list are all wired.
- **For the Fukui/AFM fast model**, you don't need to port the full xTB Hamiltonian to Rust. Use the existing fragment solver to get eigenvectors, then compute condensed Fukui indices and feed them into the analytical pairwise formula using the [gamma.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:0:0-0:0) you already have.

---

# USER

OK, to keep it more organized, we should perhaps make our RUST module more like sub modules of sone semiemeprical LCAO solver. with modules QMQM (for multi system), xTB (or GFN) for grime baed analytical pair mixing methods, and dftb/SKtable based interpolation methods. these should be encapsulated sub modules. and than we should have shared module like neighbors, diagonalization, orbital coefs to density-matrix, etc. not sure if SCC should be shared beterrn xTB and dftb submodules? sk_data are specific for dftb right? xTB should have its own params reader right?

analyse the code and figure out answers to this questions, then sketch ideal refoactoring/reorganization so that the code is clean and split into sub-modules resp sub forlders/sub-creates for cargo with clearly defined respoinsibilites and minimal cross references (so it is modular and not spageti) 

---

# Kimi-2.6

## Answers to your specific questions

**1. Should SCC be shared between xTB and DFTB?**
Yes — the SCC *iteration loop* is identical for both methods:
1. Build charge-dependent shifts
2. Apply to H0: `H = H0 + S * shift`
3. Diagonalize each fragment
4. Compute Mulliken charges
5. Mix and check convergence

Only **step 1** (the shift builder) differs:
- DFTB / GFN1: atom-resolved γ with exponential screening (`expGamma`)
- GFN2: Klopman–Ohno effective Coulomb (different screening, optionally shell-resolved)

**The right design:** put the loop in [qmqm/solver.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:0:0-0:0) and abstract the shift model behind a `CoulombModel` trait. DFTB and xTB each provide their own implementation.

**2. Is `sk_data` specific to DFTB?**
**Yes, 100%.** [sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0) is only about parsing `.skf` files, old/extended format conventions, [sk_map](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:407:0-417:1), and [EqGridTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:16:0-19:1) Neville interpolation. xTB has no SK files at all — it uses hardcoded parameter arrays and analytical Gaussian integrals. `sk_data` must move under `methods/dftb/`.

**3. Should xTB have its own params reader?**
**xTB does not need a "reader"** in the SK sense. GFN1/GFN2 parameters are hardcoded static arrays (translated from the Fortran `parameter` statements in [gfn1.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/gfn1.f90:0:0-0:0) / [gfn2.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/gfn2.f90:0:0-0:0)). The "input" is just selecting the parameter set (`GFN1`, `GFN2`, `IPEA1`, or custom). Optional: loading external tblite JSON via `serde`, but that's a bonus, not a core requirement.

## Current dependency analysis

These files are **DFTB-only** and xTB will never touch them:

| File | Lines | DFTB-only reason |
|------|-------|------------------|
| [sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0) | 419 | `.skf` parsing, old/extended format, [sk_map](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:407:0-417:1) |
| [interpolation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:0:0-0:0) | 233 | Neville interpolation for SK grid tables |
| [rotation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:0:0-0:0) | 257 | Rotates SK integrals from local to global frame via direction cosines |
| [hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0) | 417 | Calls [Rotation](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:45:0-45:20) + [SkTableSp::eval_shell_integrals_into](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:142:4-191:5) |

These are **method-agnostic** and should be shared:

| File | Lines | Why shared |
|------|-------|------------|
| [qmqm/charges.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/charges.rs:0:0-0:0) | 124 | Mulliken analysis works for any LCAO method |
| [qmqm/mixer.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/mixer.rs:0:0-0:0) | — | Anderson/DIIS/Broyden operate on charge vectors only |
| [qmqm/neighbor.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/neighbor.rs:0:0-0:0) | — | Spatial cell-list hashing, no QM knowledge |
| [neighbor.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/neighbor.rs:0:0-0:0) (top) | 35 | Atom-pair neighbor finding (generic geometry) |
| [error.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/error.rs:0:0-0:0) | — | Generic error types |

These are **currently DFTB-coupled** but conceptually generic:

| File | Problem | Fix |
|------|---------|-----|
| [qmqm/fragment.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/fragment.rs:0:0-0:0) | Hardcodes [HamiltonianBuilder::new(sk)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:54:4-90:5) | Make [FragmentTemplate](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/fragment.rs:27:0-52:1) generic over `H0Builder` trait |
| [qmqm/solver.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:0:0-0:0) | Hardcodes [GammaTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:108:0-115:1) (DFTB γ) | Make [MultiSystemSolver](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:20:0-51:1) generic over `CoulombModel` trait |
| [qmqm/shifts.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/shifts.rs:0:0-0:0) | Uses [GammaTable](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:108:0-115:1) directly | Use `CoulombModel` trait instead |

## Proposed directory structure

Keep **one crate** for now (avoid workspace overhead at ~3K LOC), but organize modules as if they were independent crates. Extraction to workspace members is trivial later because boundaries are clean.

```
rust_dftb/src/
  lib.rs                              # Public re-exports only
  
  core/                               # Method-agnostic primitives
    mod.rs
    error.rs                          # DftbError (can rename to LcaoError later)
    system.rs                         # OrbitalSystem: atom/orbital indexing, species lookup
    neighbor.rs                       # Spatial neighbor finding (moved from top-level)
    linalg.rs                         # Cholesky diagonalization, density-matrix builders
    charges.rs                        # Mulliken / Lowdin population analysis (moved from qmqm)
  
  methods/                            # Specific Hamiltonian methods
    mod.rs
    traits.rs                         # H0Builder, CoulombModel traits
    
    dftb/                             # SK-table DFTB
      mod.rs
      sk_data.rs                      # SkData, file parsing, EqGridTable
      interpolation.rs                # Neville interpolation (SK-grid specific)
      rotation.rs                     # DirectionCosines, Rotation (SK specific)
      hamiltonian.rs                  # DftbBuilder: implements H0Builder
      gamma.rs                        # ExpGamma / GammaTable (GFN1 reuses this)
      
    xtb/                              # Analytical xTB
      mod.rs
      params.rs                       # GFN1/GFN2 static parameter tables
      basis.rs                        # CGTO construction from STO exponents
      integrals.rs                    # overlap_cgto, multipole_cgto
      hamiltonian.rs                  # XtbBuilder: implements H0Builder
      coulomb.rs                      # GFN1 γ (re-export) + GFN2 Klopman–Ohno
  
  qmqm/                               # Multi-system solver (method-agnostic)
    mod.rs
    mixer.rs                          # Mixers (already generic)
    fragment.rs                       # Fragment<T: H0Builder>
    neighbor.rs                       # FragmentNeighborList
    solver.rs                         # MultiSystemSolver<T: CoulombModel>
    shifts.rs                         # Generic shift computation via CoulombModel trait
```

## The trait boundaries (zero-cost, no vtables in hot loops)

These two traits are the **only** cross-module references between [qmqm/](cci:9://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm:0:0-0:0) and `methods/`:

```rust
// methods/traits.rs
use nalgebra::DMatrix;

/// Builds H0 and S. Both DFTB and xTB implement this.
pub trait H0Builder: Clone {
    fn cutoff(&self) -> f64;
    fn build_system(&self, species: &[String]) -> Result<<OrbitalSystem>;
    fn fill_onsite(&self, sys: &OrbitalSystem, h0: &mut DMatrix<f64>) -> Result<<()>;
    fn fill_pairs(&self, sys: &OrbitalSystem, neigh: &NeighborList,
                  coords: &[[f64; 3]], h0: &mut DMatrix<f64>, s: &mut DMatrix<f64>) -> Result<<()>;
}

/// Computes SCC shifts. DFTB and xTB provide different implementations.
pub trait CoulombModel: Clone {
    fn compute_shifts(&self, coords: &[[f64; 3]], species: &[u8],
                      delta_q: &[f64], out: &mut [f64]);
}
```

With generics, Rust monomorphizes at compile time. The hot [fill_pairs](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:248:4-308:5) loop and `compute_shifts` loop have **zero trait-object overhead**.

## What moves where (concrete)

**Move into `core/`:**
- [error.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/error.rs:0:0-0:0) → `core/error.rs`
- [neighbor.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/neighbor.rs:0:0-0:0) (top-level) → `core/neighbor.rs`
- [qmqm/charges.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/charges.rs:0:0-0:0) → `core/charges.rs`
- New: `core/system.rs` (extract `OrbitalSystem` from current [hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0))
- New: `core/linalg.rs` (Cholesky wrapper, density matrix helpers)

**Move into `methods/dftb/`:**
- [sk_data.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:0:0-0:0) → `methods/dftb/sk_data.rs`
- [interpolation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/interpolation.rs:0:0-0:0) → `methods/dftb/interpolation.rs`
- [rotation.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:0:0-0:0) → `methods/dftb/rotation.rs`
- [hamiltonian.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:0:0-0:0) (current SK logic) → `methods/dftb/hamiltonian.rs`
- [qmqm/gamma.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/gamma.rs:0:0-0:0) → `methods/dftb/gamma.rs` (DFTB γ; GFN1 can delegate to it)

**Create in `methods/xtb/`:**
- `methods/xtb/params.rs` — static arrays from [gfn1.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/gfn1.f90:0:0-0:0) / [gfn2.f90](cci:7://file:///home/prokophapala/git/dftbplus/external/tblite/origin/src/tblite/xtb/gfn2.f90:0:0-0:0)
- `methods/xtb/basis.rs` — CGTO definition, contraction from STO exponents
- `methods/xtb/integrals.rs` — analytical Cartesian Gaussian overlap / multipole integrals
- `methods/xtb/hamiltonian.rs` — `XtbBuilder` implementing `H0Builder`
- `methods/xtb/coulomb.rs` — `CoulombModel` for GFN1 (wraps dftb gamma) and GFN2 (Klopman–Ohno)

**Refactor in [qmqm/](cci:9://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm:0:0-0:0):**
- [fragment.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/fragment.rs:0:0-0:0): replace [FragmentTemplate::new(sk: &SkData, ...)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:54:4-90:5) with [FragmentTemplate::new<B: H0Builder>(builder: &B, ...)](cci:1://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:54:4-90:5)
- [solver.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:0:0-0:0): replace `gamma: GammaTable` with `coulomb: C` where `C: CoulombModel`
- [shifts.rs](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/shifts.rs:0:0-0:0): replace `gamma_tbl: &GammaTable` with `model: &dyn CoulombModel` (or generic `M: CoulombModel`)

## Why this is clean (no spaghetti)

| Rule | Enforcement |
|------|-------------|
| `core/` never imports from `methods/` or [qmqm/](cci:9://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm:0:0-0:0) | It only knows about geometry, orbitals, and linear algebra |
| `methods/dftb/` never imports from `methods/xtb/` | These are sibling alternatives |
| `methods/xtb/` never imports from `methods/dftb/` | xTB has no SK tables, no rotation, no interpolation |
| [qmqm/](cci:9://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm:0:0-0:0) only imports `core::` and `methods::traits` | It does not know about SK files, CGTOs, or rotation matrices |
| `methods/traits.rs` is the **only** bridge | Two traits, ~15 lines total |

## Bottom line

- **SCC loop:** Shared in [qmqm/](cci:9://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm:0:0-0:0), parameterized by `CoulombModel`.
- **`sk_data`:** DFTB-only. Move to `methods/dftb/`.
- **xTB params:** No "reader" needed — static arrays in `methods/xtb/params.rs`, selected by enum (`GFN1` / `GFN2`).
- **Refactor order:** (1) Extract `core/` and `methods::traits`, (2) Move DFTB-specific files into `methods/dftb/`, (3) Make [FragmentTemplate](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/fragment.rs:27:0-52:1) and [MultiSystemSolver](cci:2://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm/solver.rs:20:0-51:1) generic, (4) Implement `methods/xtb/` without touching anything in `methods/dftb/` or [qmqm/](cci:9://file:///home/prokophapala/git/dftbplus/rust_dftb/src/qmqm:0:0-0:0) beyond the trait.

---

## What was done (2025-06-06)

### Phase 1: `core/` extraction

- `src/core/error.rs` — moved `DftbError`, `Result`
- `src/core/neighbor.rs` — moved `NeighborList`, `NeighborBuilder`
- `src/core/charges.rs` — moved Mulliken charge analysis from `qmqm/charges.rs`
- `src/core/mod.rs` — declares the three submodules

### Phase 2: `methods/` extraction

- `src/methods/traits.rs` — created `H0Builder` and `CoulombModel` trait stubs
- `src/methods/dftb/` — moved all DFTB-specific code:
  - `sk_data.rs` — SK file parsing, `SkData`, `SkTableSp`
  - `interpolation.rs` — `EqGridTable`, Neville interpolation
  - `rotation.rs` — `DirectionCosines`, `Rotation`
  - `hamiltonian.rs` — `HamiltonianBuilder`, `SystemContext`, `HWorkspace`
  - `gamma.rs` — `gamma_full`, `GammaTable`
- `src/methods/dftb/mod.rs` — declares DFTB submodules
- `src/methods/xtb/mod.rs` — placeholder for future xTB implementation
- `src/methods/mod.rs` — declares `traits`, `dftb`, `xtb`

### Phase 3: `lib.rs` update and cleanup

- `lib.rs` now declares `pub mod core; pub mod methods; pub mod qmqm;`
- Re-exports public API directly from new locations for backward compatibility
- Removed 6 old root-level stub files (`error.rs`, `neighbor.rs`, `sk_data.rs`, `interpolation.rs`, `rotation.rs`, `hamiltonian.rs`)
- Fixed all internal imports in `qmqm/fragment.rs`, `qmqm/solver.rs`, `scc.rs`, `output.rs`

### Verification

- `cargo check` — clean
- `cargo test --test parity_universal` — passes
- All 6 molecules (H2O, CO, CH2O, HCN, C2H4, HCOOH) pass both non-SCC and SCC parity
- HF excluded due to missing Fluorine parameters in `mio-1-1`

### Still pending

- Phase 3 proper: make `FragmentTemplate` and `MultiSystemSolver` generic over `H0Builder` and `CoulombModel`
- Implement `H0Builder` for `HamiltonianBuilder` and `CoulombModel` for `GammaTable`
- `methods/xtb/` actual implementation (params, basis, integrals, Hamiltonian, coulomb)