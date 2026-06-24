# Rust DFTB/xTB Reimplementation — Test & Status Overview

**Last updated:** 2026-06-24
**Location:** `rust_dftb/tests/`

---

## 1. DFTB+ Hamiltonian Assembly (CPU)

### 1.1 Non-SCC H0 and S — COMPLETE

The non-SCC Hamiltonian builder is fully implemented and parity-verified against Fortran DFTB+.

**Key source files:**
- `src/methods/dftb/hamiltonian.rs` — `HamiltonianBuilder`
- `src/methods/dftb/rotation.rs` — diatomic rotation matrices
- `src/methods/dftb/interpolation.rs` — Neville 8-point SK interpolation
- `src/methods/dftb/sk_data.rs` — SK file I/O and shell integral extraction

**Two code paths:**

| Path | Purpose | Status |
|------|---------|--------|
| `build_non_scc()` | Generic, arbitrary angular momentum (s, p, d) | Complete |
| `build_non_scc_sp_only()` | Hand-unrolled fast path for sp-only systems | Complete, parity-checked against generic |

**Parity tolerance:** H0 and S match Fortran reference to `< 1e-8` max absolute difference.

### 1.2 SCC (Self-Consistent Charges) — COMPLETE

Full self-consistent charge DFTB+ calculations are implemented and parity-verified against Fortran DFTB+ across a wide range of molecules.

**What works:**
- `HamiltonianBuilder::build_scc()` — standalone API that iterates to charge self-consistency
- Gamma-matrix construction (`GammaTable::from_sk_data`) — auto-extracts Hubbard U from SK files
- Full SCC loop: H_scc → diagonalize → Mulliken charges → mixer → converge
- DIIS mixer with warmup (5 iterations simple mixing, then Anderson acceleration)
- SCC energy computation matching DFTB+ "Total Electronic energy"
- Cholesky factorization caching across SCC iterations
- Efficient Mulliken charge computation (diagonal-only D·S, O(N²·n_occ))

**Parity results (all PASS):**

| Molecule | Atoms | Orbs | SCC Iters | Charge Diff | Eigval Diff | Energy Diff |
|----------|-------|------|-----------|-------------|-------------|-------------|
| H2O | 3 | 6 | ~10 | <1e-8 | <1e-8 | <1e-8 |
| CH4 | 5 | 9 | ~10 | <1e-8 | <1e-8 | <1e-8 |
| O2 | 2 | 8 | ~10 | <1e-8 | <1e-8 | <1e-8 |
| HCN | 3 | 10 | ~10 | <1e-8 | <1e-8 | <1e-8 |
| C2H4 | 6 | 12 | ~15 | <1e-8 | <1e-8 | <1e-8 |
| NH3 | 4 | 8 | ~10 | <1e-8 | <1e-8 | <1e-8 |
| HCOOH | 5 | 13 | ~15 | <1e-8 | <1e-8 | <1e-8 |
| Benzene | 12 | 30 | ~15 | <1e-8 | <1e-8 | <1e-8 |
| Uracil | 12 | 36 | ~15 | <1e-8 | <1e-8 | <1e-8 |
| Pyridine | 11 | 29 | ~15 | <1e-8 | <1e-8 | <1e-8 |
| Pentacene | 36 | 102 | ~20 | <1e-7 | <1e-6 | <1e-6 |
| Porphyrin | 38 | 110 | ~20 | <1e-7 | <1e-6 | <1e-6 |
| PTCDA | 38 | 128 | 19 | 5.3e-8 | 8.0e-6 | 1.7e-6 |

**Key bug fixed during SCC development:**
- SK table interpolation `r_max` was computed as `(nGrid-1)*dr` instead of `nGrid*dr + distFudge`, and lacked a hard cutoff beyond `r_max`. The `poly5_to_zero` tail extrapolation grows as r³ for large r, producing spurious H0 elements for distant atom pairs. Fixed by matching DFTB+ Fortran: hard zero for `r >= rMax` where `rMax = nGrid*dr + distFudge`.

---

## 2. DFTB+ Hamiltonian Assembly (GPU / OpenCL)

### 2.1 Status — DESIGN & KERNELS DONE, NO RUNTIME

| Component | Status | Notes |
|-----------|--------|-------|
| Design document | Complete | `doc/prokop/DFTB_Hassembly_OpenCL.md` |
| OpenCL kernels | Complete | `src/methods/dftb/dftb_hamiltonian.cl` |
| Host-side batch prep | Complete | `src/qmqm/gpu_prep.rs` (new, not yet wired to mod.rs) |
| OpenCL runtime integration | **MISSING** | No device init, buffer upload, or enqueue |
| End-to-end GPU test | **MISSING** | Kernels have never been launched |

**Kernel design summary:**
- **Pair-bucket sorted launch:** Pairs are grouped by species-pair + block type (1x1, 1x4, 4x4). Each workgroup loads one compact SK table into `__local` memory.
- **B-spline interpolation:** Replaced Neville with cubic B-spline for GPU-friendliness.
- **Fragment metadata:** Supports heterogeneous fragments (different atom counts) via prefix-sum offsets.
- **Precision:** f32 throughout (GPU is ~10x slower in f64).

**Next step to make this runnable:**
1. Wire `gpu_prep.rs` into `src/qmqm/mod.rs`
2. Add an OpenCL driver module (buffer allocation, kernel compilation, enqueue)
3. Write a smoke test that builds a batch, uploads it, runs `assemble_pairs`, and reads back H/S

---

## 3. GFN1-xTB and GFN2-xTB Hamiltonian Assembly

### 3.1 GFN1 — COMPLETE

- Non-SCC H0 and overlap built from Slater-type orbitals → Gaussian expansion → Cartesian integrals
- SCC loop with shell charges, Coulomb matrix, and third-order onsite correction
- Parity achieved against `tblite` reference for H2, N2, HCOOH

### 3.2 GFN2 — MOSTLY COMPLETE (small residual errors)

GFN2 adds shell-resolved Coulomb, multipole integrals, dipole/quadrupole potentials, and coordination-number-dependent damping.

**Major bugs found and fixed (documented in `INTEGRAL_DEBUG_NOTES.md`):**

| Bug | Impact | Fix |
|-----|--------|-----|
| `dfactorial` off-by-one indexing | p-orbital normalization wrong by sqrt(3) | Changed `DFACTORIAL[l+1]` to `DFACTORIAL[l]` |
| Coordination number formula (single vs double exponential) | CN 3.5x too large, propagated to multipole damping | Rewrote to match Fortran exactly |
| Unit conversion in C helper (Bohr vs Angstrom) | Dipole integrals 17% off | Pass Angstrom to helper, Bohr to Rust |

**Current parity (after fixes):**

| System | H0/S | SCC Charges | SCC Eigenvalues | Notes |
|--------|------|-------------|-----------------|-------|
| H2 | **PASS** | **PASS** | **PASS** | Reference: tblite GFN1 |
| N2 | **PASS** | **PASS** (1e-10) | Close (0.36% error) | Down from 2.6% pre-fix |
| HCOOH | **PASS** | Close (~0.03% error) | Close (~0.006 diff) | H1 residual ~9e-4 |

**Remaining work:**
- N2 effective Hamiltonian: 1.1e-4 residual (0.026% relative) at position (2,2)
- HCOOH effective Hamiltonian: 9.1e-4 residual (0.21% relative)
- HCOOH charges: ~0.001 absolute error (~0.03%)
- Suspected causes: subtle gamma-matrix difference, third-order potential, or integral cutoff

---

## 4. Test Inventory

### 4.1 DFTB+ CPU Tests

| Test | File | Requires | Status |
|------|------|----------|--------|
| `parity_h0_methane_example` | `tests/parity_non_scc.rs` | `RUST_DFTB_SK_DIR`, `RUST_DFTB_REF_H`, `RUST_DFTB_REF_S` | Env-based |
| `parity_sp_only_vs_generic` | `tests/parity_non_scc.rs` | `RUST_DFTB_SK_DIR` | **PASS** |
| `benchmark_parity_sp_only` | `tests/parity_non_scc.rs` | `RUST_DFTB_SK_DIR` | **PASS** |
| `parity_universal_from_env` | `tests/parity_universal.rs` | Env vars + optional Fortran refs | **PASS** (13 molecules) |
| `scc_convergence_from_xyz` | `tests/parity_scc.rs` | `RUST_DFTB_SK_DIR`, `RUST_DFTB_SCC_XYZ` | **PASS** (13 molecules) |
| `fragment_h2_matches_full_system_non_scc` | `tests/qmqm_integration.rs` | `RUST_DFTB_SK_DIR` | **PASS** |
| `fragment_n2_matches_full_system_non_scc` | `tests/qmqm_integration.rs` | `RUST_DFTB_SK_DIR` | **PASS** |
| `fragment_hcooh_matches_full_system_non_scc` | `tests/qmqm_integration.rs` | `RUST_DFTB_SK_DIR` | **PASS** |
| `h2_fixed_charge_scc_parity` | `tests/qmqm_integration.rs` | `RUST_DFTB_SK_DIR` | **PASS** |
| `n2_fixed_charge_scc_parity` | `tests/qmqm_integration.rs` | `RUST_DFTB_SK_DIR`, `RUST_DFTB_REF_H`, `RUST_DFTB_REF_S` | **PASS** |
| `hcooh_fixed_charge_scc_parity` | `tests/qmqm_integration.rs` | `RUST_DFTB_SK_DIR`, `RUST_DFTB_REF_H`, `RUST_DFTB_REF_S` | **PASS** |
| `gamma_self_consistency` | `tests/qmqm_integration.rs` | None | **PASS** |
| `fragment_h2_diagonalization` | `tests/qmqm_integration.rs` | `RUST_DFTB_SK_DIR` | **PASS** |
| `fragment_h2_fixed_neutral_charges` | `tests/qmqm_integration.rs` | `RUST_DFTB_SK_DIR` | **PASS** |

### 4.2 DFTB+ GPU Tests

| Test | File | Status |
|------|------|--------|
| *(none)* | — | **MISSING** |

### 4.3 xTB (GFN1/GFN2) Tests

| Test | File | Requires | Status |
|------|------|----------|--------|
| `test_h2_gfn1_parity` | `tests/xtb_parity.rs` | `tblite_helper` binary | **PASS** |
| `test_n2_gfn1_parity` | `tests/xtb_parity.rs` | `tblite_helper` binary | **PASS** |
| `test_hcooh_gfn1_parity` | `tests/xtb_parity.rs` | `tblite_helper` binary | **PASS** |
| `test_h2_scc_parity` | `tests/xtb_parity.rs` | `tblite_helper` binary | **PASS** |
| `test_n2_scc_parity` | `tests/xtb_parity.rs` | `tblite_helper` binary | Charges PASS, eigenvalues 0.36% off |
| `test_hcooh_scc_parity` | `tests/xtb_parity.rs` | `tblite_helper` binary | Charges ~0.03% off, H1 ~9e-4 off |

**Test helper:**
- `tests/tblite_helper.c` — C program linking against `tblite` library; outputs JSON with Hamiltonian, overlap, charges, eigenvalues, multipole integrals
- `tests/tblite_helper` — compiled binary (must exist for xTB tests to run; tests skip gracefully if missing)

### 4.4 Test Automation

- `tests/run_parity.py` — Universal DFTB+ parity runner:
  1. Parses XYZ file
  2. Generates DFTB+ HSD input
  3. Runs Fortran DFTB+ to produce `hamsqr1.dat` / `oversqr.dat`
  4. Invokes `cargo test --test parity_universal` with env vars
  5. Optional SCC mode reads actual deltaQ from Fortran debug output

**Usage:**
```bash
python3 tests/run_parity.py molecule.xyz
python3 tests/run_parity.py molecule.xyz --scc --delta-q 0.1,-0.1
```

---

## 5. How to Reproduce

### 5.1 DFTB+ Non-SCC Parity (H2, N2, HCOOH)

```bash
# Set SK directory
export RUST_DFTB_SK_DIR=/path/to/mio-1-1

# Run all qmqm integration tests
cargo test --test qmqm_integration -- --nocapture

# Or run specific test
cargo test fragment_h2_matches_full_system_non_scc -- --nocapture
```

### 5.2 DFTB+ Universal Parity (any XYZ)

```bash
# Requires compiled DFTB+ binary at DFTB_BIN path inside run_parity.py
python3 tests/run_parity.py /path/to/molecule.xyz --tol 1e-7
```

### 5.3 DFTB+ Full SCC Parity

```bash
# Run full SCC convergence parity for any molecule
python3 tests/run_scc_full.py /path/to/molecule.xyz --release

# Tested molecules (all PASS):
# H2O, CH4, O2, HCN, C2H4, NH3, HCOOH, benzene, uracil, pyridine,
# pentacene, porphyrin, PTCDA
```

### 5.4 xTB Parity

```bash
# Compile tblite helper first
gcc -o tests/tblite_helper tests/tblite_helper.c -ltblite $(pkg-config --cflags --libs tblite)

# Run xTB tests
cargo test --test xtb_parity -- --nocapture
```

### 5.5 GPU (not yet runnable)

```bash
# No automated test exists yet.
# Manual steps would be:
# 1. Compile OpenCL kernels
# 2. Wire gpu_prep.rs into qmqm/mod.rs
# 3. Write a driver that calls clCreateBuffer / clEnqueueNDRangeKernel
# 4. Compare output H/S against CPU reference
```

---

## 6. Known Gaps & Next Steps

| # | Gap | Priority | Effort |
|---|-----|----------|--------|
| 1 | **No CI / automated test runner** | High | Low — add GitHub Actions or a `justfile` |
| 2 | **DFTB+ GPU has no OpenCL runtime** | High | Medium — needs driver module + smoke test |
| 3 | ~~DFTB+ standalone SCC convergence~~ | ~~Medium~~ | ~~Done~~ — `build_scc()` implemented, parity verified on 13 molecules up to PTCDA (38 atoms) |
| 4 | **GFN2 N2/HCOOH residual errors (~0.1%)** | Medium | High — needs deep debugging of gamma / third-order / multipole |
| 5 | **gpu_prep.rs not wired into module tree** | Low | Trivial — add `pub mod gpu_prep;` to `qmqm/mod.rs` |
| 6 | **No single-command reproduction for all tests** | Medium | Low — create Makefile/justfile that builds helper + runs everything |

---

## 7. File Map

| File | Purpose |
|------|---------|
| `tests/parity_non_scc.rs` | Non-SCC parity (methane example, sp-only vs generic, benchmark) |
| `tests/parity_universal.rs` | Universal env-driven parity for arbitrary molecules |
| `tests/qmqm_integration.rs` | Fragment vs full-system parity, fixed-charge SCC, gamma consistency |
| `tests/xtb_parity.rs` | GFN1/GFN2 parity against tblite reference |
| `tests/parity_scc.rs` | Full SCC convergence parity (charges, eigenvalues, energy vs Fortran) |
| `tests/run_scc_full.py` | Python driver: XYZ → Fortran SCC → Rust SCC comparison |
| `tests/run_parity.py` | Python driver: XYZ → Fortran → Rust comparison |
| `tests/tblite_helper.c` | C helper extracting JSON data from tblite library |
| `INTEGRAL_DEBUG_NOTES.md` | Detailed debug log for GFN2 multipole / CN bugs |
| `doc/prokop/DFTB_Hassembly_OpenCL.md` | GPU kernel design document |
| `src/qmqm/gpu_prep.rs` | Host-side GPU batch preparation (new, not yet wired) |
