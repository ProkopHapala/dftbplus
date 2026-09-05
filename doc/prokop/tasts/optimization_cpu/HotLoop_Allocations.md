# Hot-Loop Allocation & Redundancy Audit — Fix Plan

**Status:** spec only — implementation not started.
**Prerequisite:** commit current working state first (may break compilation temporarily).
**Reference:** `doc/prokop/chats/CPU_Optimization.chat.md` (GPT 5.6 review)
**Files touched:**
- `rust_dftb/src/methods/dftb/dftb_cpu.rs` (main)
- `rust_dftb/src/methods/dftb/forces.rs` (force path)
- `rust_dftb/src/methods/dftb/hamiltonian.rs` (SystemContext, build_non_scc)
- `rust_dftb/src/qmqm/mixer.rs` (DIIS)
- `rust_dftb/examples/hbond_ref.rs` (driver)

**Verification:** run `hbond_ref --mode optimize` 100 steps, compare energy trajectory and final energy to current baseline (-3.2112994148e1 Hartree). Energies must match to ≥10 digits. Timing should improve.

---

## Problem inventory (by severity)

### CRITICAL — allocations or clones every SCC iteration / geometry step

#### C1: `update_geometry` clones entire SkData every geometry step
**File:** `dftb_cpu.rs:230`
**Code:** `let builder = HamiltonianBuilder::new(self.sk.clone());`
**Problem:** `self.sk.clone()` deep-copies all SK tables (every pair table, every grid, every spline). This is the single most expensive clone in the entire pipeline, called once per geometry step.
**Fix:** Build H0/S directly from `self.ctx` (which already owns the pair tables and species data). Either:
- (a) Add a method `build_h0_s_from_ctx(&ctx, coords) -> (DMatrix, DMatrix)` that uses `SystemContextStatic` directly, bypassing `HamiltonianBuilder`, OR
- (b) Make `HamiltonianBuilder` borrow `&SkData` instead of owning it, then store a `&'static SkData` or `Rc<SkData>` in `DftbCpu`.
**Prefer (a)** — it eliminates the `HamiltonianBuilder` dependency entirely from the hot path.

#### C2: `as_ctx()` clones 6 Vecs + collects 2 more every call
**File:** `dftb_cpu.rs:120-134`
**Code:** `atom_species: self.atom_species.clone(), atom_n_orb: ..., atom_orb_off: ..., species_n_orb: ..., pair_lut: ..., pair_tables: ...`
**Problem:** `compute_forces` calls `as_ctx()` **twice** (lines 553 and 572), each time cloning 6 vectors and collecting 2 slices. That's 12 allocations + 4 collects per force evaluation.
**Fix:** Restructure `SystemContext` to borrow from `SystemContextStatic`:
```rust
impl SystemContextStatic {
    pub fn as_ctx_ref(&self) -> SystemContextRef<'_> {
        SystemContextRef {
            n_atoms: self.n_atoms,
            atom_species: &self.atom_species,
            atom_n_orb: &self.atom_n_orb,
            atom_orb_off: &self.atom_orb_off,
            species_n_orb: &self.species_n_orb,
            species_ang: &self.species_ang,  // &Vec<Vec<i32>>
            species_onsite: &self.species_onsite,
            pair_lut: &self.pair_lut,
            pair_tables: &self.pair_tables,
        }
    }
}
```
Then change all force functions to accept `SystemContextRef<'_>` (or make `SystemContext` generic over owned/borrowed, or just use `&SystemContextStatic` directly). The force functions only need `atom_n_orb`, `atom_orb_off`, `atom_species`, `pair_table()`, `species_ang`.

#### C3: SCC loop allocates `dq: Vec<f64>` every iteration
**File:** `dftb_cpu.rs:296`
**Code:** `let dq: Vec<f64> = (0..nat).map(|i| self.charges[i] - self.q0[i]).collect();`
**Fix:** Add field `dq: Vec<f64>` to `DftbCpu`, preallocated in `new()`. In `solve_scc`, compute in-place:
```rust
for i in 0..nat { self.dq[i] = self.charges[i] - self.q0[i]; }
```

#### C4: SCC loop clones entire L matrix every iteration
**File:** `dftb_cpu.rs:311`
**Code:** `let mut b = l.clone();`
**Problem:** Clones N² f64 to build `B = V_orb · L`. Then `b` is consumed by `solve_lower_triangular`.
**Fix:** Add field `b_mat: DMatrix<f64>` to `DftbCpu`. Copy L into it once, then each iteration only reapply the row scaling:
```rust
self.b_mat.copy_from(&self.cholesky_l);
for mu in 0..n {
    let atom = self.orb_to_atom(mu);
    let v = self.v_shift[atom];
    for j in 0..n { self.b_mat[(mu, j)] *= v; }
}
```
Note: `solve_lower_triangular` consumes its argument. Need to either:
- (a) Clone `b_mat` into a scratch before the solve (still 1 clone, but we can reuse a preallocated scratch), OR
- (b) Use LAPACK `dtrsm` (triangular solve with multiple RHS) which works in-place on a preallocated buffer — this is P4.

#### C5: SCC loop clones H0' every iteration
**File:** `dftb_cpu.rs:322`
**Code:** `self.h_prime = self.h0_prime.clone();`
**Fix:** Don't clone — add the SCC correction in-place:
```rust
self.h_prime.copy_from(&self.h0_prime);
for i in 0..n {
    for j in 0..n {
        self.h_prime[(i,j)] += 0.5 * (x[(i,j)] + x[(j,i)]);
    }
}
```
`copy_from` is a memcpy, not a heap allocation. The current `.clone()` also allocates a new matrix every time.

#### C6: SCC loop allocates `h_prime_data: Vec<f64>` every iteration
**File:** `dftb_cpu.rs:330`
**Code:** `let mut h_prime_data = self.h_prime.as_slice().to_vec();`
**Problem:** LAPACK needs a contiguous `&mut [f64]`. `DMatrix::as_slice()` already gives `&[f64]`, but LAPACK needs `&mut`. The `.to_vec()` allocates N² f64 every iteration.
**Fix:** Add field `h_prime_data: Vec<f64>` to `DftbCpu`, preallocated as `vec![0.0; n_orbs * n_orbs]`. Copy into it:
```rust
self.h_prime_data.copy_from_slice(self.h_prime.as_slice());
```
Then pass `&mut self.h_prime_data` to `dsyevd`.

#### C7: SCC loop allocates `eig: Vec<f64>` every iteration
**File:** `dftb_cpu.rs:331`
**Fix:** Add field `eig_tmp: Vec<f64>` to `DftbCpu`, preallocated as `vec![0.0; n_orbs]`.

#### C8: Force path allocates 6 Vecs per pair per direction in `pair_block_derivative`
**File:** `forces.rs:484-506`
**Code:** `h_plus, s_plus, h_minus, s_minus, dh, ds` — each `vec![0.0; block_size]`
**Problem:** For 20 atoms with ~190 pairs, 3 directions: 190×3×6 = 3420 allocations per force evaluation.
**Fix:** Preallocate scratch buffers in `DftbCpu` (or a `ForceWorkspace` struct):
```rust
struct ForceWorkspace {
    h_plus: Vec<f64>, s_plus: Vec<f64>,
    h_minus: Vec<f64>, s_minus: Vec<f64>,
    dh: Vec<f64>, ds: Vec<f64>,
    sqr_dm: Vec<f64>, sqr_edm: Vec<f64>,
}
```
Sized to `max_block = max_n_orb²`. Pass `&mut ForceWorkspace` to force functions.

#### C9: DIIS mixer allocates 2 Vecs every iteration
**File:** `mixer.rs:165-166`
**Code:**
```rust
let q_in_copy: Vec<f64> = q_inout.to_vec();
let res_copy: Vec<f64> = residual.to_vec();
```
**Fix:** Replace `VecDeque<Vec<f64>>` with a ring buffer of preallocated arrays:
```rust
struct DiisMixer {
    q_in_bufs: Vec<Vec<f64>>,   // max_history preallocated
    res_bufs: Vec<Vec<f64>>,    // max_history preallocated
    buf_idx: usize,             // ring buffer write position
    n_filled: usize,            // how many valid entries
}
```
On `mix()`, write into `q_in_bufs[buf_idx]` via `copy_from_slice`, advance `buf_idx = (buf_idx + 1) % max_history`.

---

### HIGH — redundant computation

#### H1: `scc_shift_force` recomputes dS already computed by `non_scc_electronic_force`
**File:** `forces.rs:625`
**Problem:** Both functions iterate over the same neighbor pairs and call `pair_block_derivative` for dS. The dS is identical. This doubles the finite-difference SK evaluations.
**Fix:** Fuse into a single `electronic_force_fused` function that computes dH and dS once per pair per direction, then applies both the non-SCC and SCC-shift contractions:
```rust
for p in &neigh.pairs {
    let (sqr_dm, sqr_edm) = extract_pair_dm_edm(...);
    for dir in 0..3 {
        let (dh, ds) = pair_block_derivative(...);
        // non-SCC: F += 2 * Σ (DM·dH - EDM·dS)
        // SCC shift: F += 2 * avg_shift * Σ (DM·dS)
        // Combined in one pass over the block
    }
}
```
This halves the number of SK evaluations in the force path.

#### H2: `scc_shift_force` re-extracts DM block already extracted by `non_scc_electronic_force`
**File:** `forces.rs:622`
**Fix:** Eliminated by H1 (fused loop).

#### H3: `scc_double_counting_force_cached` ignores precomputed `gamma_mat`
**File:** `dftb_cpu.rs:594-627`
**Code:** `fn scc_double_counting_force_cached(..., _gamma_mat: &[f64], gamma_table: &GammaTable, ...)`
**Problem:** The function receives `gamma_mat` but ignores it (note the `_` prefix). It recomputes distances, calls `gamma_table.u()`, and calls `gamma_prime_full()` from scratch for every pair.
**Fix:** Precompute `gamma_prime_over_r: Vec<f64>` in `update_geometry()` — for each pair (a,b), store `γ'(R_ab) / R_ab` (the force coefficient). Then the force is just:
```rust
let coef = -delta_q[i] * delta_q[j] * gamma_prime_over_r[i*n+j];
forces[i] += coef * dr;  // dr = coords[i] - coords[j] (Å)
forces[j] -= coef * dr;
```
No sqrt, no gamma function, no distance computation in the force loop.

#### H4: `build_result` has dead code — H_scc built twice
**File:** `dftb_cpu.rs:440-472`
**Problem:** Lines 442-457 build `h_scc` with a wrong formula (double-counts), then lines 463-472 build it again correctly, overwriting the first. The first block is dead code.
**Fix:** Delete lines 440-457. Keep only the correct block (463-472).

#### H5: `compute_forces` allocates `coords_bohr` but never uses it
**File:** `dftb_cpu.rs:544-546`
**Problem:** Dead allocation. The neighbor list is already in Bohr from `update_geometry`, and the force functions take coords in Å.
**Fix:** Delete lines 544-546.

#### H6: `build_result` clones 6 matrices/vectors into `CpuSccResult`
**File:** `dftb_cpu.rs:491-498`
**Problem:** `h0, s, h_scc, density, edm, eigenvalues, eigenvectors, charges, q0` — all cloned.
**Fix:** This is called once per geometry (not per SCC iteration), so it's lower priority. However, if forces don't need `h0` or `h_scc` in the result (they use cached state), we can remove those fields from `CpuSccResult`. The density and EDM are needed. Consider returning references instead of clones, or moving the data out of the solver into the result (solver rebuilds them next geometry anyway).

#### H7: `orb_to_atom` does linear scan every call
**File:** `dftb_cpu.rs:405-415`
**Problem:** Called O(N) times per SCC iteration (in the V_orb loop and H_scc build). Linear scan through `atom_orb_off`.
**Fix:** Precompute `orb_to_atom_lut: Vec<u8>` (length `n_orbs`) in `new()`:
```rust
orb_to_atom_lut: (0..n_orbs).map(|mu| {
    (0..n_atoms).find(|&a| {
        mu < (ctx.atom_orb_off[a] + ctx.atom_n_orb[a]) as usize
    }).unwrap() as u8
}).collect();
```
Then `orb_to_atom(mu)` is just `self.orb_to_atom_lut[mu] as usize` — O(1).

#### H8: `extract_pair_dm_edm` allocates 2 Vecs per pair
**File:** `forces.rs:518-534`
**Fix:** Eliminated by C8 (preallocated `ForceWorkspace` with `sqr_dm`, `sqr_edm`).

---

### MEDIUM — smaller but still wasteful

#### M1: `update_geometry` allocates `coords_bohr` Vec
**File:** `dftb_cpu.rs:236`
**Fix:** Preallocate `coords_bohr: Vec<[f64;3]>` in `DftbCpu`, fill in `update_geometry`.

#### M2: `update_geometry` clones S for Cholesky
**File:** `dftb_cpu.rs:242`
**Code:** `nalgebra::linalg::Cholesky::new(self.s.clone())`
**Problem:** `Cholesky::new` consumes the matrix. We need to keep `self.s` for forces.
**Fix:** Either:
- (a) Use LAPACK `dpotrf` which works in-place on a copy (preallocated), OR
- (b) Preallocate `s_copy: DMatrix<f64>` and do `s_copy.copy_from(&self.s); Cholesky::new(s_copy)`.

#### M3: SCC loop allocates `y_occ`, `c_occ`, `sc_occ` matrices
**File:** `dftb_cpu.rs:354-356`
**Problem:** Three N×N_occ matrix allocations per SCC iteration.
**Fix:** Preallocate in `DftbCpu`:
```rust
y_occ: DMatrix<f64>,    // n_orbs × n_occ
c_occ: DMatrix<f64>,    // n_orbs × n_occ
sc_occ: DMatrix<f64>,   // n_orbs × n_occ
```
Fill via `copy_from` or direct computation each iteration.

#### M4: `compute_forces` allocates `delta_q: Vec<f64>`
**File:** `dftb_cpu.rs:563`
**Fix:** Preallocate `dq_force: Vec<f64>` in `DftbCpu`, or reuse `self.dq` from the SCC loop (it's the same thing).

#### M5: `build_result` allocates `eps_occ: Vec<f64>`
**File:** `dftb_cpu.rs:430`
**Fix:** Preallocate `eps_occ: Vec<f64>` in `DftbCpu` (length `n_occ`).

---

## Implementation order

Each step must be verified by running the 100-step optimization and checking the energy matches before proceeding to the next.

### Phase 1: SCC loop — eliminate all per-iteration allocations
1. **C3**: Add `dq: Vec<f64>` field, compute in-place
2. **C5**: Use `copy_from` instead of `clone` for h_prime
3. **C6**: Add `h_prime_data: Vec<f64>` field, `copy_from_slice` instead of `to_vec`
4. **C7**: Add `eig_tmp: Vec<f64>` field
5. **C4**: Add `b_mat: DMatrix<f64>` field, `copy_from` instead of `clone` (still need 1 clone for the triangular solve, or switch to `dtrsm` — defer to P4)
6. **M3**: Add `y_occ`, `c_occ`, `sc_occ` preallocated matrices
7. **H7**: Add `orb_to_atom_lut: Vec<u8>` for O(1) lookup
8. **Verify**: 100-step optimization, energy must match baseline

### Phase 2: update_geometry — eliminate SK clone
9. **C1**: Write `build_h0_s_from_ctx()` that uses `SystemContextStatic` directly, no `HamiltonianBuilder`, no SK clone
10. **M1**: Preallocate `coords_bohr`
11. **M2**: Preallocate `s_copy` for Cholesky
12. **Verify**: 100-step optimization

### Phase 3: Force path — eliminate per-pair allocations and redundancy
13. **C2**: Create `SystemContextRef<'_>` (zero-clone borrow from `SystemContextStatic`), change force functions to accept it
14. **C8**: Create `ForceWorkspace` struct with preallocated scratch buffers, change force functions to accept `&mut ForceWorkspace`
15. **H1/H2**: Fuse `non_scc_electronic_force` and `scc_shift_force` into one loop — compute dH/dS once per pair per direction
16. **H5**: Delete dead `coords_bohr` allocation in `compute_forces`
17. **H4**: Delete dead double H_scc build in `build_result`
18. **Verify**: 100-step optimization, forces must match

### Phase 4: Coulomb force — use precomputed gamma'
19. **H3**: Precompute `gamma_prime_over_r: Vec<f64>` in `update_geometry`, use in `scc_double_counting_force_cached`
20. **Verify**: 100-step optimization

### Phase 5: DIIS mixer — ring buffer
21. **C9**: Replace `VecDeque<Vec<f64>>` with ring buffer of preallocated arrays
22. **Verify**: 100-step optimization, SCC iteration count must not change

### Phase 6: build_result cleanup (lowest priority)
23. **H6**: Remove unnecessary fields from `CpuSccResult` (h0, h_scc if forces don't need them)
24. **M4/M5**: Preallocate `dq_force` and `eps_occ`
25. **Verify**: 100-step optimization

---

## What is explicitly NOT in this task

- **P4 (LAPACK dpotrf/dsygst/dtrsm):** deferred — the nalgebra triangular solves are correct; replacing them is a separate numerical-parity task
- **P7 (all SK channels once per pair):** deferred — requires restructuring `Rotation::rotate_diatomic_block_into`
- **P8 (analytic sp derivatives + polynomial interpolation):** deferred — requires implementing analytic SK derivatives, a physics task
- **P5 (fused dH/dS with analytic derivatives):** the *allocation* fusion is in Phase 3 (H1), but replacing finite differences with analytic derivatives is deferred

These are algorithmic changes, not allocation fixes. They belong in a separate task after the allocation issues are resolved.
