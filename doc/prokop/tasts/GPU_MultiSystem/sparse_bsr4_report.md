# Sparse BSR4 Purification — Implementation Report

**Date:** 2026-09-05
**Source spec:** [`doc/prokop/chats/SparseLargeSystemOpenCL.chat.md`](../../chats/SparseLargeSystemOpenCL.chat.md)
**Kernels:** [`rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl`](../../../rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl)
**Module:** `rust_dftb/src/methods/sparse/`
**Tests:** `rust_dftb/tests/gpu_sparse_bsr4.rs` — **13/13 pass**

---

## 1. Goal

Implement and test the block-CSR (BSR4, 4×4 atom blocks) sparse density-kernel
purification route for large DFTB systems, **independently of the `qmqm`
dense/fragment solver**. The motivation is memory: for nanocrystals with
thousands of atoms the dense eigenvector matrix is O(N²) while the localized
density kernel is O(N·z) with z ≈ constant for gapped systems.

The architecture (from the chat doc):

```
atom-block CSR (4×4 blocks)
   ↓
non-orthogonal density kernel K:  KSK = K,  N_occ = Tr(KS)
   ↓
masked sparse products  C = P_M(A·B)   (never densify)
   ↓
purification:  McWeeny (3KSK−2KSKSK)  or  TC2
   ↓
Mulliken charges, energy, (eventually) forces  — no eigenvectors
```

No S⁻¹ or S⁻¹ᐟ² is ever constructed for the density solve; matrix polynomials
are evaluated as sequences of masked sparse products so every intermediate
stays O(N·z).

---

## 2. What was built

### 2.1 Module structure

```
rust_dftb/src/methods/sparse/
├── mod.rs                       module root + re-exports
├── bsr4.rs                      host BSR4 data structure + helpers
├── gpu_sparse.rs                GPU wrapper (SparseBsr4Gpu)
└── sparse_bsr4_purification.cl  OpenCL kernels (from chat doc, unchanged)
```

Registered in `methods/mod.rs` as `pub mod sparse`. The module depends only on
the shared `GpuRuntime` (OpenCL context/queue) — **not** on any `qmqm` solver
logic. No second OpenCL context is created.

### 2.2 Host data structure (`bsr4.rs`)

`Bsr4Matrix`: atom-block CSR with 4×4 blocks.
- `row_ptr[n_atom+1]`, `col_idx[nblock]` (sorted per row), `values[nblock*16]`.
- Stores **both** (i,j) and (j,i) for symmetric matrices → all GPU kernels are
  gather-only (no scatter, no atomics).
- Methods: `find` (binary search), `block`, `set_block`, `to_dense` (for CPU
  reference checks), `from_structure`, `from_parts`.

Helpers:
- `build_geometric_mask(pos, cutoff)` — Euclidean radius mask.
- `build_full_mask(n_atom)` — dense mask (for parity tests).
- `build_product_mask(M_K, M_HS)` — **boolean structural product** M_T = M_K ∘ M_HS
  (block (i,j) exists iff ∃ k: (i,k)∈M_K ∧ (k,j)∈M_HS). This is the exact
  possible support of T=KS, not an arbitrary radius.
- `build_identity(n_atom, mask)` — 4×4 identity on diagonal blocks.
- `transpose_block_map`, `diag_block_map` — precomputed index maps for the GPU
  symmetrize/trace/Mulliken kernels.
- `symmetrize_host` — CPU reference for the GPU symmetrize kernel.
- `gershgorin_bounds(B)` — orbital-level spectral bounds (emin, emax).
- `inf_norm(S)` — max orbital row sum (for Newton-Schulz α = 1/‖S‖_∞).
- `dense_matmul`, `dense_max_abs_diff`, `dense_frobenius`, `dense_trace` — CPU
  reference utilities.

### 2.3 GPU wrapper (`gpu_sparse.rs`)

`SparseBsr4Gpu`: compiles `sparse_bsr4_purification.cl` with configurable
`-D WG`, `-D MAX_LEFT_BLOCKS`, `-D REDUCE_WG` defines.

**Low-level kernel launches** (buffer-in/buffer-out):
`spgemm_masked`, `spgemm_masked_bsym`, `axpby`, `zero`, `mcweeny`, `tc2`,
`symmetrize`, `mulliken_ks`, `trace_ks` (recursive reduction), `idempotency_err`.

**High-level `Bsr4Matrix` methods:**
- `matmul_masked`, `matmul_masked_bsym` — masked SpGEMM returning `Bsr4Matrix`.
- `ksk(K, S, k_mask, t_mask)` — Q = K·S·K via two Bsym products.
- `mcweeny_step` — one step of 3KSK−2KSKSK.
- `tc2_step` — one TC2 step (Q if Tr>Nocc else 2K−Q).
- `symmetrize_mat`, `mulliken`, `trace_ks`, `idempotency_err`, `frobenius_norm`.

**P0 solver pieces (per chat doc §1-§2):**
- `newton_schulz_inverse(S, k_mask, t_mask, ...)` — Z ≈ S⁻¹ via
  Z_{n+1} = 2Z_n − Z_n S Z_n. Monitors R_Z = ‖I−ZS‖_F/√N. Stops on tolerance
  or stall.
- `build_k0(H, S, Z, emin, emax)` — K₀ = (εmax·Z − Z·H·Z)/(εmax−εmin).
- `spectral_bounds(H, Z, t_mask, padding)` — Gershgorin on B=ZH + padding.
- `hamiltonian_residual(H, K, S, mask)` — R_H = ‖HKS − SKH‖_F.
- `tc2_purify(K₀, S, Nocc, ...)` — full TC2 loop with monitoring.

### 2.4 Tests (`gpu_sparse_bsr4.rs`) — 13 tests

| # | Test | Validates | Result |
|---|------|-----------|--------|
| 1 | `spgemm_masked_vs_dense` | Generic masked SpGEMM vs dense matmul | 2.4e-7 |
| 2 | `spgemm_masked_bsym_vs_dense` | Symmetric-right SpGEMM (two-pointer intersection) | 2.4e-7 |
| 3 | `spgemm_mask_truncation` | Only masked blocks computed (10/16 blocks) | 2.4e-7 |
| 4 | `trace_ks_vs_cpu` | Tr(KS) partial reduction + recursive reduce | 2.4e-7 |
| 5 | `symmetrize_vs_host` | GPU symmetrize vs host reference + symmetry check | 0.0 |
| 6 | `mulliken_vs_cpu` | Mulliken charges q_A = 2·Tr((KS)_AA) | 1.2e-7 |
| 7 | `idempotency_exact_kernel` | K from CPU eigensolve: KSK=K, Tr(KS)=Nocc | 1.3e-7 |
| 8 | `mcweeny_convergence` | 3KSK−2KSKSK from perturbed K₀ → idempotent | 2.6e-7 (5 steps) |
| 9 | `tc2_convergence` | Metric TC2 from α·K_exact → idempotent | 4.1e-6 (4 steps) |
| 10 | `boolean_product_mask` | M_T = M_K∘M_HS correctness + omission detection | 0 mismatches |
| 11 | `newton_schulz_inverse` | Z ≈ S⁻¹: R_Z → 0, Z matches dense S⁻¹ | 5 iters, R_Z=0 |
| 12 | `k0_and_tc2_vs_dense_projector` | **Full P0 chain**: H,S→Z→K₀→TC2→K vs dense projector + R_H | 2.4e-7 |
| 13 | `rh_distinguishes_projectors` | R_H ≈ 0 for eigenvector-aligned K, large for random K | ratio 9.9×10⁶ |

All tests skip gracefully if no OpenCL device is available.

---

## 3. What works

### 3.1 Sparse algebra kernels — correct to f32 round-off

Both SpGEMM variants match dense CPU references to ~2.4e-7 (f32 epsilon):
- `bsr4_spgemm_masked` (generic, binary search for B_kj)
- `bsr4_spgemm_masked_Bsym` (symmetric B, two-pointer intersection using
  B_kj = transpose(B_jk))

The Bsym kernel is mathematically correct: C_ij[r,c] += A_ik[r,m]·B_jk[c,m].
Trace, Mulliken, symmetrize, idempotency, and Frobenius-norm reductions all
match CPU references.

### 3.2 Full P0 solver chain — end-to-end parity

The headline result (test 12):

```
H,S (random, well-conditioned, nocc=3)
  → Z ≈ S⁻¹     (Newton-Schulz, 7 iters, R_Z = 0)
  → spectral bounds  (Gershgorin on ZH, emin=-2.72, emax=7.59)
  → K₀ = (emax·Z − ZHZ)/Δε   (||K₀ − K_ref|| = 0.46)
  → TC2 purification  (30 iters, R_I → 2.4e-7, Tr(KS) → 3.000000)
  → K_final  (||K_final − K_ref|| = 2.4e-7)
  → R_H = ||HKS−SKH||_F = 3.0e-7  (vs K_ref: 7.6e-8)
```

The sparse route produces the **same density kernel** as a dense generalized
eigensolve, to f32 precision, using only masked sparse products.

### 3.3 Newton-Schulz inverse converges quadratically

Z₀ = α·I with α = 1/‖S‖_∞. For well-conditioned S (I + 0.15·offdiag):
R_Z: 0.39 → 0.17 → 0.037 → 0.0023 → 0 (5 iterations). Matches dense S⁻¹ to
<1e-2.

### 3.4 Boolean product mask is exact

M_T = M_K ∘ M_HS contains exactly the structurally possible support of T=KS,
verified against a dense boolean product reference (0 mismatches). This is
better than an arbitrary R_T radius: it includes every possible contribution
but no blocks that cannot occur because of the actual atomic graph.

### 3.5 R_H diagnostic works

R_H = ‖HKS − SKH‖_F is ~0 for eigenvector-aligned projectors and ~1.1 for a
random symmetric matrix (ratio 9.9×10⁶). See §5.2 for an important subtlety
discovered during testing.

---

## 4. Problems encountered and how they were corrected

### 4.1 TC2 divergence from noisy K₀

**Problem:** Test 9 (TC2 convergence) initially diverged — trace moved *away*
from Nocc (3 → 2.84 → 2.61 → 1.85 → −1.13 → −18.5 → −335).

**Root cause:** Adding symmetric noise of 0.03 to an exact projector broke the
spectral structure. The eigenvalues of KS left the [0,1] basin, and TC2
(which pushes eigenvalues toward 0 or 1 based on trace direction) amplified
the error instead of correcting it.

**Correction:** The chat doc explicitly warns: "TC2 does not magically discover
the Hamiltonian eigenvectors... the starting K₀ must already be a spectral
function of the generalized eigenproblem." The correct spectrally-valid
perturbation is K₀ = α·K_exact (eigenvalues of KS are {α, 0} ⊂ [0,1]). With
α=0.8, TC2 converges in 4 steps (0.25 → 4.1e-6, trace 2.4 → 2.99999).

**Lesson:** This is not a test artifact — it's a real algorithmic requirement.
A production solver **must** construct K₀ from the current Hamiltonian (the
Z → K₀ route), not reuse an old density with noise. This motivated
implementing the full Newton-Schulz + K₀ chain (P0 items 2-3).

### 4.2 Masked truncation test — wrong CPU reference

**Problem:** Test 3 (masked truncation) failed with max|dC| = 1.8.

**Root cause:** The CPU reference used the *full* dense random matrices for the
matmul, but the GPU only sees mask-projected A/B (blocks outside the mask are
zero in the BSR4 representation). The GPU correctly computed P_M(A_proj ·
B_proj), but the reference computed A_full · B_full.

**Correction:** Project A and B onto the mask first (via `to_dense()`), then
multiply. After fix: max|dC| = 2.4e-7.

### 4.3 Boolean product mask — omission detection logic

**Problem:** Test 10 initially asserted that omitting a block from M_T would
change the *remaining* blocks (diff over the intersection). It didn't —
max|dT| = 0.

**Root cause:** Each output block C_ij is computed independently in the SpGEMM
kernel. Removing block (0,j) from M_T doesn't affect other blocks — it only
means block (0,j) is absent from the output. The omission is visible only in
the *missing* block itself.

**Correction:** Check that the omitted block has nonzero norm in the full result
(0.50) and is absent from the truncated result. This correctly detects the
omission.

### 4.4 R_H test — wrong expectation about what R_H detects

**Problem:** Test 13 initially compared R_H for the correct projector (lowest
nocc eigenvectors) vs a "wrong" projector (highest nocc eigenvectors). Both
had R_H ≈ 0 — the test failed because R_H did not distinguish them.

**Root cause:** R_H = ‖HKS − SKH‖_F = 0 for **any** projector onto eigenvectors
of the generalized eigenproblem, not just the occupied ones. This is because
Hc_i = Sc_i ε_i holds for *every* eigenvector, so K = c_i c_i^T gives
HKS = SKH regardless of which c_i are chosen. R_H detects "is K aligned with
the eigenvector basis?" not "is it the right subspace?"

**Correction:** Compare an eigenvector-aligned projector against a *random
symmetric matrix* (not eigenvector-aligned). R_H correctly distinguishes these
(ratio 9.9×10⁶). The right-subspace check is Tr(KS) = N_occ, not R_H.

**Lesson:** The three diagnostics R_I, R_N, R_H have distinct roles:
- R_I = ‖KSK−K‖ — idempotency
- R_N = |Tr(KS)−Nocc| — electron count
- R_H = ‖HKS−SKH‖ — eigenvector alignment

All three are needed: an idempotent K with the right trace but not aligned with
H's eigenvectors is still wrong. See §5.2.

### 4.5 OpenCL `ProgramBuilder` API

**Problem:** Initial code used `ProgramBuilder::cmplr_def(...)` as a static
method and `.bo(...)` chaining. The `ocl` crate's `cmplr_def` is a `&mut self`
method, not a static.

**Correction:** Use a mutable builder:
```rust
let mut builder = ProgramBuilder::new();
builder.devices(device);
builder.src(SOURCE);
builder.cmplr_def("WG", config.wg);
builder.cmplr_def("MAX_LEFT_BLOCKS", config.max_left_blocks);
let program = builder.build(&context)?;
```

### 4.6 `dense_frobenius` name collision

**Problem:** `dense_frobenius` was defined both in `bsr4.rs` (newly added) and
in the test file → `E0255: name defined multiple times`.

**Correction:** Removed the duplicate from the test file; import from `bsr4.rs`.

---

## 5. Open issues and problems

### 5.1 SCC integration (P0 items 6-7) — not yet done

The sparse solver pieces (Z, K₀, TC2, Mulliken, R_H) are ready and tested in
isolation, but **not yet connected to the existing DFTB Hamiltonian builder**
(`HamiltonianBuilder`, `SkData`, `gamma_full`). The remaining P0 work:

1. Wrap `bsr4_build_Hscc` (already compiled, not yet wrapped):
   H_AB = H⁰_AB + ½(V_A + V_B)·S_AB
2. Build the SCC loop:
   ```
   q → dq → V = γ·dq → Hscc → K₀(Hscc,S) → TC2 → KS → q_new → mixer → repeat
   ```
3. Match dense CPU charges and energies for small systems with full masks.

**Key cautions from the chat doc (§4, §10):**
- Use the **same sign convention** as the existing dense/CPU DFTB for Δq.
  Do not infer the sign from the sparse code.
- Port the **total-energy expression** exactly (band + SCC charge + repulsive).
  The electronic solver producing the correct density is not enough.
- Long-range γ remains N² for now (FMM/mesh is a separate later concern).
- TC2 must regenerate K₀ from the *current* H each SCC iteration — cannot
  reuse the previous density as the only starting point (§10.1).

### 5.2 R_H subtlety — does not detect wrong subspace

As discovered in §4.4, R_H = 0 for any eigenvector projector. For non-degenerate
systems, R_I + R_N + R_H together ensure correctness. But for **degenerate**
systems (e.g. metallic graphene at the Fermi level), there can be multiple
projectors with the same trace that are all eigenvector-aligned — R_H won't
distinguish them. The energy comparison (or a direct K vs K_ref check) remains
the ultimate correctness test.

### 5.3 Multi-mask not yet tested with distinct radii

The API supports separate M_HS, M_K, M_T (the `k_mask`/`t_mask` parameters are
threaded through all methods), and `build_product_mask` constructs M_T
correctly. But all tests currently use `build_full_mask` (zero truncation
error). The convergence-vs-R_K study (chat doc §8) — varying R_K = 4, 6, 8, ...
Å and monitoring ΔE, RMS(Δq), R_I, R_N, R_H — is **not yet done**. This is the
key scientific test that determines whether fixed geometric masks are
sufficient or dynamic sparsification is needed.

### 5.4 Newton-Schulz truncation floor not characterized

With a fixed sparse mask, R_Z will eventually hit a truncation floor (the
inverse is not perfectly localized). The stall detection (stop when
R_Z^{n+1} > 0.9 R_Z^n for 3 iters) is implemented, but we haven't yet
characterized how R_Z floor vs R_K behaves for realistic DFTB overlap matrices.
A rapidly diverging/stagnating inverse is an important warning: either S is
poorly conditioned or the mask is too short.

### 5.5 Fused KSK kernel — not implemented (P1)

The chat doc (§5) describes a row-local fused KSK kernel that materializes
T_i = (KS)_i in local memory and computes Q_ij from it, eliminating the global
T buffer. The doc explicitly says: **"Do not optimize fused products before
the P0 chain works."** P0 works now, so this is the next performance
optimization. The doc warns against a naive triple sparse loop — the correct
approach is row-local materialization with degree-specialized paths (small
rows → fused, large rows → two-kernel).

### 5.6 LNV solver — kernels compiled but not wrapped/tested

`bsr4_lnv_gradient` and `bsr4_gradient_step` are in the .cl file and compile,
but no Rust wrapper or test exists. The chat doc (§6) says to validate the
gradient by finite differences first, and notes the electron-number difficulty
(grand-canonical LNV needs a μ search for fixed-N DFTB). LNV is an alternative
solver branch, not currently a replacement for TC2.

### 5.7 `bsr4_drop_small` — diagnostic not wrapped

The block-thresholding diagnostic kernel (§7) is compiled but not wrapped. It
zeroes blocks with ‖K_ij‖_F < ε but does NOT compact CSR (no memory saving).
Its purpose is to answer: "how many blocks are numerically negligible?" before
investing in dynamic sparsification. Should be wrapped with pair-symmetric
dropping (decide once per (i,j)/(j,i) pair via transpose map) and never drop
diagonal blocks.

### 5.8 BSR4 and hydrogen

Current BSR4 assumes every atom has exactly 4 orbitals (s,p). Graphene ribbons
need H-passivated edges (H has 1 orbital). The chat doc (§2a, §10) says either
assert "all atoms have 4 orbitals" or support `norb_atom[A]` with padded 4×4
storage (inactive rows/cols stay zero; identity/trace/residual kernels must
ignore inactive orbitals). **Not yet addressed.** Do not discover this only
after the sparse solver is otherwise finished.

### 5.9 Periodic systems

The current branch assumes finite clusters / Γ-point real matrices. General
k-point DFTB introduces complex blocks and is a separate extension (§10).
Should be asserted explicitly.

### 5.10 Forces

The chat doc (§10) notes forces are possible without eigenvectors via
W = KHK (energy-weighted density), giving dE_band/dR = 2·Tr(K·dH/dR) −
2·Tr(W·dS/dR) + SCC + repulsive terms. Not yet implemented. The density
representation must not make construction of sparse W impossible — current
K storage is fine.

### 5.11 f32 precision

All calculations are float. The chat doc (§"Where single precision enters")
suggests the dominant controllable error will often be localization/truncation
rather than f32 rounding, and recommends the hierarchy:
```
increase iterations → if no improvement, increase R_T/R_K → if still no improvement, investigate f32/conditioning
```
rather than immediately introducing doubles. Not yet benchmarked on real
systems.

---

## 6. Recommended next steps (per chat doc §"Recommended concrete implementation sequence")

Done: items 1-5 (multi-mask, Newton-Schulz Z, K₀, TC2 parity, R_H).

**Next:**
6. Wrap `build_Hscc` and integrate one complete sparse SCC calculation using
   full masks.
7. Match dense CPU charges and energies.
8. Enable finite R_K and perform systematic convergence tests (§5.3).
9. Benchmark/implement fused row-local KSK (§5.5).
10. Validate LNV gradient and explore LNV as the H-dependent/warm-start solver.
11. Use `drop_small` to decide whether dynamic sparsity is worth the complexity.

---

## 7. Files changed

**New:**
- `rust_dftb/src/methods/sparse/mod.rs`
- `rust_dftb/src/methods/sparse/bsr4.rs`
- `rust_dftb/src/methods/sparse/gpu_sparse.rs`
- `rust_dftb/tests/gpu_sparse_bsr4.rs`

**Modified:**
- `rust_dftb/src/methods/mod.rs` — added `pub mod sparse`

**Unchanged (from chat doc):**
- `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl` — the OpenCL
  kernels, used as-is.

---

## 8. Graphene DFTB sparse-vs-dense parity (2026-09-06)

### 8.1 Setup

A shared Rust engine CLI (`rust_dftb/src/bin/dftb_engine.rs`) was built to
drive the full workflow from a Rhai script
(`rust_dftb/scripts/test_graphene_sparse.rhai`):

1. Generate graphene PAH geometry (carbon-only, 4 orbs/atom via `build_non_scc_sp_only`).
2. Run dense DFTB non-SCC → reference H, S, density D = 2·C_occ·C_occ^T.
3. Convert dense H, S → BSR4 atom-block format (full mask).
4. Newton–Schulz Z ≈ S⁻¹.
5. Spectral bounds (emin, emax) via Gershgorin on ZH.
6. K₀ = (emax·Z − ZHZ)/(emax − emin).
7. TC2 purification with divergence safeguard.
8. Compare 2·K vs D (spin convention: K has Tr(KS)=N_occ, D has Tr(DS)=2·N_occ).
9. Compare Mulliken charges.

### 8.2 Test systems

| System            | C atoms | Orbitals | N_occ | Dense E (Ha) |
|-------------------|---------|----------|-------|--------------|
| Benzene           | 6       | 24       | 12    | -10.1546     |
| Coronene          | 24      | 96       | 48    | -42.0238     |
| Circumcoronene    | 54      | 216      | 108   | -95.5796     |

### 8.3 Results

| System            | TC2 iters | R_I (best)   | R_H (norm)   | ‖2K−D‖_max | ‖2K−D‖_rms | max\|dq\|  |
|-------------------|-----------|--------------|--------------|------------|------------|-----------|
| Benzene           | 48        | 2.24e-7      | 5.38e-6      | 2.58e-5    | 4.91e-6    | 1.48e-5   |
| Coronene          | 38        | 9.07e-7      | 8.25e-7      | 6.25e-6    | 4.44e-7    | 1.14e-5   |
| Circumcoronene    | 45        | 1.58e-6      | 5.56e-7      | 2.42e-5    | 7.48e-7    | 2.44e-5   |

**Key findings:**

- **Dense-sparse parity at f32 roundoff:** ‖2K−D‖_max ≈ 2.5e-5 across all
  three systems. This is consistent with f32 accumulation error in the GPU
  kernels and the TC2 stopping at R_I ≈ 1e-6–1e-7.
- **Mulliken charges match to ~1e-5:** max\|dq\| ≤ 2.4e-5 for all systems.
  The sparse route reproduces the dense DFTB charge distribution.
- **Electron count exact:** Tr(KS) converges to N_occ to 6 decimal places
  before divergence sets in.
- **TC2 divergence safeguard is essential:** TC2 in f32 converges to a
  minimum R_I (~1e-6 to 1e-7) then diverges. The `tc2_purify` function now
  tracks the best K and returns it when R_I grows by >10×, preventing NaN
  blowup. This is expected f32 behavior for TC2 near machine precision.
- **Hamiltonian commutator R_H is small:** normalized R_H ≈ 5e-7 to 5e-6,
  confirming [H,K]_S ≈ 0 at the purified solution.

### 8.4 Convergence plots

Plot saved at: `debug/graphene_sparse/convergence.png`

Shows R_I (log scale) and Tr(KS) versus TC2 iteration for all three systems.
The characteristic TC2 pattern is visible: oscillatory convergence to a
minimum, followed by divergence. The safeguard returns the best iterate.

### 8.5 Convention note

The sparse density kernel K uses the convention Tr(KS) = N_occ (no spin
factor). The dense DFTB density matrix D = 2·C_occ·C_occ^T uses the spin-
degenerate convention Tr(DS) = 2·N_occ. The correct comparison is **2·K vs D**.
The Mulliken charges are unaffected because the factor of 2 cancels in the
charge formula q_A = Tr_A(KS) vs q_A = ½·Tr_A(DS).

### 8.6 Limitations and next steps

- **Carbon-only:** The current BSR4 layout assumes 4 orbitals/atom. Hydrogen
  (1 orbital) requires a variable-block extension or a separate H block map.
  All tests use bare carbon PAHs (no passivation).
- **Non-SCC only:** The current test uses H₀ (non-SCC). SCC integration
  requires iterating the sparse density with the SCC charge self-consistency.
- **Full mask:** All tests use the full (dense) mask. Geometric cutoff masks
  need validation for sparsity-vs-accuracy tradeoff.
- **f32 precision:** The ~2.5e-5 error is f32 roundoff. An f64 kernel path
  or mixed-precision accumulation would improve this.
- **Graphene gaplessness:** TC2 converges despite the small/near-zero gap of
  graphene, but the oscillatory behavior is more pronounced than for gapped
  molecular systems.
