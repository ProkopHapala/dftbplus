# Task 2: Sparse Linear-Scaling Vibrations of Si/Diamond Nanocrystals

**Created:** 2026-09-07
**Status:** planning / manifest
**Owner:** prokop / Devin

---

## 1. Goal

Optimize large silicon and diamond nanocrystals (500-1000 atoms) as fast as
possible using sparse linear-scaling DFTB on GPU (single precision), then
compute the Hessian and extract vibrational spectra. Verify that:

1. The sparse BSR4 method is **actually sparse** — matrices do not densify
   through the purification iterations, and the geometric mask provides real
   O(N) scaling, not just a constant-factor speedup over dense.
2. The method provides **real performance benefits** — measure wall time vs
   system size and confirm near-linear scaling.
3. The Hessian computed with f32 GPU sparse DFTB matches a reference (f64 CPU
   dense DFTB or DFTB+ Fortran) for smaller systems at the **same geometry**
   (i.e. without relaxation, to isolate Hessian accuracy from geometry
   differences).
4. The resulting vibrational spectra match the reference — no pathological
   imaginary modes (negative Hessian eigenvalues) caused by poor convergence
   or insufficient force tolerance.

This is a **single-system process** (not multi-system batched). The challenge
is making one large system fast via sparsity and GPU, not batching many small
systems.

---

## 2. Motivation & Physics

### 2.1 Why Si and diamond nanocrystals?

Silicon and diamond nanocrystals are:
- **Covalent network solids** with tetrahedral bonding (diamond cubic /
  zincblende structure). Each atom has 4 nearest neighbors → naturally sparse
  Hamiltonian and overlap matrices.
- **Technologically relevant:** Si nanocrystals for photovoltaics, quantum
  dots; diamond nanocrystals for NV centers, abrasives, optics.
- **Well-characterized:** DFTB SK files exist for Si (siband-1-1, matsci-0-3,
  pbc-0-3) and C (matsci-0-3, pbc-0-3). DFTB+ has phonon test cases for Si.
- **Good test of sparsity:** diamond cubic has a fixed coordination number
  (4), so the number of non-zero blocks in H/S scales linearly with N. The
  question is whether purification preserves this sparsity.

### 2.2 Vibrational spectra from Hessian

The vibrational spectrum is obtained by:
1. **Optimize** the geometry to a local minimum (all forces < tolerance).
2. **Compute the Hessian** H_ij = d²E/dx_i dx_j (3N × 3N matrix).
   - Method: central finite differences of forces. For each of the 3N
     coordinates: evaluate forces at x_i + h and x_i - h, then
     H_ij = (F_i(x+h) - F_i(x-h)) / (2h). Requires 6N force evaluations.
   - For N=1000: 6000 force evaluations. Each force evaluation requires a
     full SCC solve. This is where linear scaling is critical.
3. **Mass-weight** the Hessian: H_mw = M^{-1/2} H M^{-1/2} where M = diag(m_i).
4. **Diagonalize** H_mw → eigenvalues ω². Frequencies ω = sqrt(ω²).
   - 3N × 3N diagonalization. For N=1000: 3000 × 3000 — feasible with LAPACK
     or iterative methods.
5. **Remove 6 zero modes** (translations + rotations) — these should be
   ~0 if the geometry is at a true minimum.
6. **Check for imaginary modes** — negative eigenvalues of H_mw → imaginary
   frequencies. These indicate:
   - The geometry is NOT at a minimum (saddle point or unconverged).
   - The Hessian is inaccurate (force noise from f32 or poor SCC convergence).
   - The finite-difference step h is too small (numerical noise) or too large
     (higher-order terms).

### 2.3 The sparsity question

The BSR4 sparse format stores H and S as 4×4 atom blocks in CSR format. For
diamond cubic with a geometric mask (only blocks within r_max cutoff):
- **H0, S:** O(N) non-zero blocks — each atom interacts with ~4-12 neighbors
  (1st, 2nd, 3rd shell depending on r_max).
- **K0 (initial density guess):** same sparsity as H0 (scaled by S^{-1}).
- **Purification iterations (TC2/McWeeny):** K → 3KSK - 2KSKSK. The product
  KSK has the sparsity of K·S intersected with K — if both are sparse, the
  product is sparse. But repeated products can **fill in** (densify) if the
  mask is not truncated.
- **The key test:** does the masked SpGEMM (with geometric mask) keep the
  matrix sparse through 20-50 purification iterations? Or does the mask need
  to be expanded, losing the O(N) benefit?

### 2.4 Accuracy vs speed tradeoff

We explicitly accept **single precision (f32) on GPU** for speed. The
questions are:
- Is f32 sufficient for **geometry optimization** (forces need to be accurate
  enough to reach a minimum)?
- Is f32 sufficient for **Hessian computation** (second derivatives amplify
  force errors — the Hessian is the derivative of forces, so force noise
  becomes Hessian noise)?
- How tight must the **SCC convergence** be for the Hessian to be free of
  pathological imaginary modes? (Rule of thumb: SCC tolerance should be
  10-100× tighter than the force tolerance, because the Hessian is the
  derivative of forces.)

---

## 3. What Is Already Implemented

### 3.1 Sparse BSR4 matrices and GPU purification

- **BSR4 layout** — `methods/sparse/bsr4.rs::Bsr4Matrix`. 4×4 atom blocks,
  CSR, symmetric storage. Dense ↔ BSR4 conversion. Geometric + full masks.
- **GPU TC2 purification** — `methods/sparse/gpu_sparse.rs`. Newton-Schulz
  Z ≈ S⁻¹, spectral bound estimation, TC2 purification loop with idempotency
  residual R_I. Convergence history export.
- **Device-resident sparse workspace** — `GpuBsrStructure`, `GpuBsrMatrix`,
  `SparsePurifyWorkspace` with persistent device buffers, cached kernel
  handles. Exactly 2 SpGEMMs per TC2 iteration, 0 matrix host transfers per
  iteration, 1 scalar read for trace/branch. Bitwise-identical parity vs
  old host-roundtrip path.
- **Sparse Mulliken charges** — from KS diagonal blocks. Parity vs dense +
  DFTB+ on benzene/coronene/circumcoronene (max|Δq| < 6.5e-5 e).
- **Jacobi stopping fix** — separated PAIR_SKIP_TOL (1e-12) from JACOBI_TOL
  (1e-7), 3-sweep stagnation detection.
- **OpenCL kernels** — `sparse_bsr4_purification.cl`: SpGEMM (masked,
  symmetric-right), axpby, zero, mcweeny, tc2, symmetrize, mulliken_ks,
  trace_ks, idempotency_err, identity_residual, matmul_masked.

### 3.2 CPU forces (dense)

- **Analytic DFTB forces** — `methods/dftb/forces.rs::compute_scc_forces`.
  Four components: F_nonSCC, F_SCC_shift, F_SCC_dc, F_rep. Parity vs Fortran
  <7e-8. Analytic dH0/dx, dS/dx.
- **FIRE optimizer** — `examples/hbond_ref.rs::FireOptimizer`. Single-system,
  CPU, with LAPACK eigensolver.

### 3.3 Geometry builder (carbon sp² only)

- `rust_dftb/src/geometry/mod.rs` — builds graphene sheets, ribbons, PAHs,
  flakes. **No diamond cubic / zincblende / tetrahedral builder.** This is a
  gap — needs to be written.

### 3.4 SK files for Si and C

- **siband-1-1:** Si-Si, Si-H, H-Si, H-H. Specialized for Si band structure.
  Located at `/home/prokop/SIMULATIONS/dftbplus/slakos/siband-1-1/`.
- **matsci-0-3:** Si-Si, Si-H, Si-C, Si-O, C-C, C-H, C-N, C-O, etc.
  General materials science set. Located at
  `/home/prokop/SIMULATIONS/dftbplus/slakos/matsci-0-3/`.
- **pbc-0-3:** Si-Si, Si-H, Si-C, C-C, C-H, etc. Periodic boundary conditions
  set. Located at `/home/prokop/SIMULATIONS/dftbplus/slakos/pbc-0-3/`.
- **DFTB+ phonon test:** `test/app/phonons/Si/` — Si₂ dimer phonon reference.
  Has `hessian.out`, `Si-Si.skf`, `dftb_in.hsd`.

### 3.5 DFTB+ Fortran phonon reference

- `app/phonons/` — DFTB+ phonon calculation executable.
- `src/dftbp/` — Fortran reference for Hessian computation (finite-difference
  forces), phonon postprocessing.
- `test/app/phonons/Si/` — reference Si phonon test with `hessian.out`.

### 3.6 Davidson partial eigensolver (for frontier orbitals, not needed for purification)

- `methods/sparse/davidson.rs` — generalized Davidson for HC=SCε. Works on
  benzene, fails on coronene/circumcoronene (diagonal preconditioner
  insufficient for degenerate frontier). Not needed for purification-based
  SCC (which avoids diagonalization entirely), but relevant if we want
  HOMO-LUMO gaps for the nanocrystals.

---

## 4. What Needs to Be Done

### 4.1 Diamond cubic / zincblende geometry generation (use FireCore — do NOT rebuild)

**Policy:** FireCore has a rich, well-tested nanocrystal generator (both JS
and Python) that produces Si and diamond nanocrystals with H-passivation,
Miller-plane facets, Wulff shapes, bridge defects, and silyl passivation.
We do NOT want to duplicate this machinery in dftbplus. Instead, generate
geometries in FireCore and export `.xyz` / `.mol2` files into `data/xyz/`
here. See §8 below for the full FireCore tool inventory and workflow.

**What we need in dftbplus:** only a thin `.xyz` loader (already exists:
`rust_dftb/src/io.rs::parse_xyz`) to consume the generated geometries. No
diamond cubic builder in Rust.

**Sizing** (achievable with FireCore generators):
| Shape | Radius (Å) | N_Si | N_H (passivation) | N_total | N_orbs (Si=4, H=1) |
|---|---|---|---|---|---|
| Small | 5 | ~30 | ~30 | ~60 | ~150 |
| Medium | 10 | ~200 | ~100 | ~300 | ~900 |
| Large | 15 | ~600 | ~200 | ~800 | ~2600 |
| X-large | 20 | ~1200 | ~400 | ~1600 | ~5200 |

### 4.2 Sparse SCC solve (energy + charges, no forces yet)

**Current state:** the sparse purification path computes the density matrix K
from H0 and S, then Mulliken charges. But it does NOT compute the total SCC
energy. The energy requires:
- E_band = Tr(D · H0) (Frobenius trace — kernel exists:
  `frobenius_trace_batched` but for dense; need sparse version)
- E_scc = 0.5 · Σ Δq · V (gamma contribution — need sparse gamma matvec or
  dense gamma for N_atom × N_atom which is small)
- E_rep = Σ repulsive spline (pairwise, O(N) with neighbor list)

**What's needed:**
1. **Sparse Frobenius trace** Tr(K · H0) — sum over non-zero blocks of
   element-wise product. New kernel: `sparse_frobenius_trace` in
   `sparse_bsr4_purification.cl`.
2. **SCC energy assembly** — E_total = Tr(K·H0) + 0.5·ΣΔq·V + E_rep.
   The gamma matvec V = G·Δq is dense (N_atom × N_atom) but N_atom is small
   (500-1000) → can be done on host or with a dense GPU kernel.
3. **SCC loop with sparse purification** — the full SCC cycle:
   ```
   for scc_iter in 0..max_scc_iters:
       1. Build H_scc = H0 + 0.5·S·(V_i + V_j)  (sparse elementwise)
       2. Purify K from H_scc and S  (TC2/McWeeny, sparse)
       3. q = Mulliken(K, S)  (sparse, exists)
       4. V = G · Δq  (dense gamma matvec, small)
       5. Residual + mix → q_next
       6. Check convergence
   ```
   This requires a sparse H_scc update kernel (elementwise addition on the
   sparse structure — similar to `axpby` but with the S·(V_i+V_j) term).

### 4.3 Sparse forces (BLOCKING for optimization and Hessian)

**Current state:** analytic DFTB forces exist only for dense CPU
(`methods/dftb/forces.rs`). The sparse path has no force computation.

**What's needed:** forces in the sparse BSR4 framework. The four components:
1. **F_nonSCC = 2·(DM·dH0' − EDM·dS')** — requires dH0/dx, dS/dx. In sparse
   BSR4, this means computing derivatives of the 4×4 blocks with respect to
   atomic positions. The derivative blocks have the same sparsity as H0/S
   (only neighbor pairs contribute). This is a sparse contraction: for each
   pair (i,j), compute the 3×4×4 derivative tensor and contract with the
   density matrix blocks.
   - **Complexity:** O(N · n_neigh · 4² · 3) = O(N) for fixed coordination.
2. **F_rep = dE_rep/dr · r_hat** — pairwise, O(N · n_neigh). Same as dense
   but using the neighbor list from the BSR4 structure.
3. **F_SCC_dc (gamma + 1/R Coulomb)** — gamma' matvec. The gamma matrix is
   dense (N_atom × N_atom) but N_atom is small. Can be done on host or with
   a dense GPU kernel.
4. **F_SCC_shift (Pulay-like)** — dS/dx contracted with density. Same sparse
   structure as F_nonSCC component 1.

**Alternative: finite-difference forces with sparse SCC.** For the Hessian,
we need forces at 6N displaced geometries. Each requires:
- Rebuild H0/S at displaced geometry (sparse, O(N) for local displacement —
  only a few blocks change)
- Sparse SCC solve (purification, O(N) per iteration)
- Extract forces by finite differencing the energy

**For geometry optimization:** analytic forces are much faster (one SCC solve
per step vs 6N+1). For Hessian: finite-difference forces are the standard
method (6N force evals regardless of analytic vs finite-difference, because
the Hessian IS the finite difference of forces).

**Recommendation:**
- For **optimization:** implement sparse analytic forces (one SCC solve per
  step, O(N) force evaluation).
- For **Hessian:** use finite-difference of forces (6N SCC solves). Each SCC
  solve is O(N) with sparse purification → total O(N²) for the Hessian.
  For N=1000: 6000 SCC solves × O(1000) per solve = 6M operations — feasible
  on GPU.

### 4.4 Geometry optimization with sparse forces

**Need:** FIRE (or L-BFGS) optimizer using sparse DFTB forces.
- Single system, not batched.
- Each step: sparse SCC solve → sparse forces → FIRE step → update geometry.
- Warm-start charges from previous step (3-4 SCC iters instead of 20+).
- Convergence: max|F| < f_tol (tight: 1e-4 Ha/Å for Hessian quality).
- **Critical:** the geometry must be at a TRUE minimum for the Hessian to
  have no imaginary modes. This requires tight force convergence.

**Performance target:** 500-atom Si nanocrystal optimized in <30 minutes on
GPU. 1000-atom in <2 hours. (These are guesses — need to measure.)

### 4.5 Hessian computation

**Method:** central finite differences of forces.
```
For each atom i, each Cartesian direction α ∈ {x, y, z}:
    1. Displace x_{i,α} by +h
    2. Rebuild H0/S (only blocks involving atom i change — O(1) blocks)
    3. Sparse SCC solve → forces F+
    4. Displace x_{i,α} by -h
    5. Rebuild H0/S → SCC → forces F-
    6. H_{j,β, i,α} = -(F+_{j,β} - F-_{j,β}) / (2h)  for all j, β
```

**Parameters:**
- Displacement step h: DFTB+ uses `deltaXDiff = epsilon(1.0)^(1/4) ≈ 1.19e-4`
  Bohr ≈ 6.3e-5 Å. For f32, this is too small (below f32 precision). Need
  h ~1e-3 to 1e-2 Å for f32 forces.
- **Question:** what is the optimal h for f32 forces? Too small: numerical
  noise dominates. Too large: higher-order terms contaminate. Need to
  benchmark.

**Parallelization:** the 6N force evaluations are independent → can batch
them. But each is a large sparse SCC solve (N=1000), so we can't fit many
on the GPU simultaneously. Options:
- (a) Sequential: 6N SCC solves, one at a time. Simple but slow.
- (b) Batched: pack K displaced geometries into one batch, solve K SCC
  systems simultaneously. Limited by GPU memory (each system needs its own
  sparse workspace). For N=1000, each workspace is ~N·n_neigh·16·4 bytes
  ≈ 1000·12·64 = 768 KB → can fit ~100 systems on a 4 GB GPU.
- (c) Hybrid: batch 10-50 displaced geometries, iterate through 6N/50 batches.

**Recommendation:** Option (c) — batch 10-50 displaced geometries per GPU
call. For N=1000: 6000 evals / 50 per batch = 120 batches. Each batch takes
~50× SCC time. Total: 120 × (50 × SCC_time) = 6000 × SCC_time.

### 4.6 Vibrational frequency computation

**After Hessian:**
1. Mass-weight: H_mw[i,j] = H[i,j] / sqrt(m_i · m_j)
2. Diagonalize H_mw (3N × 3N) → eigenvalues λ_k
3. Frequencies: ω_k = sqrt(λ_k) (in cm⁻¹ after unit conversion)
4. Remove 6 zero modes (3 translations + 3 rotations)
5. Check for imaginary modes (λ_k < 0 → ω_k imaginary)

**For N=1000:** 3000×3000 dense symmetric eigensolve. LAPACK `dsyevd` handles
this in ~seconds. Or use iterative methods (Lanczos) if only low-frequency
modes are needed.

**Implementation:** `rust_dftb/src/methods/phonon.rs` (new) or
`rust_dftb/src/core/hessian.rs` (new).

### 4.7 Sparsity monitoring and linear-scaling verification

**Critical deliverable:** prove (or disprove) that the sparse method is
actually O(N) and that matrices stay sparse.

**Measure per system size (N = 60, 300, 800, 1600):**
1. **nnz(N):** number of non-zero blocks in H0, S, K0, K_final. Plot nnz vs
   N. Should be linear if sparse.
2. **nnz through purification:** track nnz(K) at each TC2 iteration. Does it
   grow? If the geometric mask truncates, it should stay constant.
3. **Fill ratio:** nnz / N². Should decrease as 1/N for truly sparse.
4. **Time per SCC iteration:** plot vs N. Should be linear if sparse.
5. **Total optimization time:** plot vs N. Should be ~linear (dominated by
   SCC iterations × O(N) per iteration).
6. **Hessian computation time:** plot vs N. Should be ~O(N²) (6N force evals
   × O(N) per eval).
7. **GPU memory usage:** plot vs N. Should be ~linear.

**Output:** `debug/nanocrystal_vib/scaling_study.tsv` + plots:
- `debug/nanocrystal_vib/nnz_vs_N.png`
- `debug/nanocrystal_vib/time_vs_N.png`
- `debug/nanocrystal_vib/fill_ratio_vs_N.png`
- `debug/nanocrystal_vib/purification_nnz_history.png`

---

## 5. Open Questions and Challenges

### 5.1 Does TC2 purification preserve sparsity?

- The TC2 step is K → 3KSK - 2KSKSK. The product K·S has the sparsity of
  K's row pattern intersected with S's column pattern. With a geometric
  mask (fixed cutoff), the product should stay within the mask.
- **But:** the McWeeny variant (3KSK - 2KSKSK) involves KSKSK — a triple
  product. If the mask is not truncated at each step, fill-in propagates.
- **Current implementation:** uses masked SpGEMM (`spgemm_masked`) which
  truncates the product to the output mask. So sparsity IS preserved by
  construction. The question is whether this truncation **degrades accuracy**
  — are important blocks being dropped?
- **Test:** compare sparse-purified K vs dense-purified K for a medium
  system (N=300). Measure max|ΔK| and max|Δq|. If the truncated product
  loses accuracy, we need a wider mask.

### 5.2 f32 precision for Hessian

- The Hessian is the second derivative of energy. f32 forces have ~1e-4
  relative precision. The finite-difference Hessian amplifies this:
  H ~ ΔF / (2h). If h = 1e-3 Å and ΔF ~ 1e-4 Ha/Å, then H ~ 0.1 Ha/Å²
  with noise ~1e-1/1e-3 = 100 — wait, that's wrong. Let me think again.
  - Force noise: δF ~ 1e-4 Ha/Å (f32 SCC)
  - Hessian element: H = (F+ - F-) / (2h) ≈ dF/dx
  - Hessian noise: δH ~ δF / h = 1e-4 / 1e-3 = 0.1 Ha/Å²
  - This is HUGE — comparable to actual Hessian elements (~1-10 Ha/Å²).
  - **Solution:** use larger h (h=1e-2 → δH ~ 0.01, marginal) or use f64
    forces (host-side) for the Hessian.
- **Question:** can we use f32 SCC + f64 force post-processing? The SCC
  solve is the expensive part (GPU, f32). The force contraction (Tr(DM·dH0')
  etc.) is cheaper and can be done in f64 on host from the f32 density matrix.
  This would give f64 forces from f32 density → much better Hessian.
- **Alternative:** use f64 on GPU (`cl_khr_fp64`). Most NVIDIA GPUs support
  it but it's 2-4× slower than f32. For the Hessian (6N force evals), the
  extra precision may be worth the speed cost.

### 5.3 SCC convergence tightness for Hessian

- DFTB+ recommends SCC tolerance 1e-8 for phonon calculations. Our default
  is 1e-7. For f32, the achievable tolerance is limited to ~1e-6 (f32 has
  ~7 digits).
- **Question:** is 1e-6 SCC tolerance sufficient for a Hessian without
  imaginary modes? Need to test on small systems first.
- **Symptom of poor convergence:** low-frequency imaginary modes (~50-200
  cm⁻¹). These are the acoustic/optical modes that are most sensitive to
  force noise.

### 5.4 H-passivation and mixed species in BSR4

- H atoms have 1 orbital (1s), Si has 4 (3s3p). The BSR4 format uses 4×4
  blocks. For H atoms, 3 of the 4 orbitals are empty (zero padding).
- **Current state:** BSR4 assumes 4 orbitals per atom. H atoms are padded
  to 4 with zeros. This wastes 75% of the H-atom blocks but keeps the
  structure uniform.
- **Open issue (from roadmap):** "BSR4: variable block size or dense fallback
  for H atoms" — not yet implemented. For Si nanocrystals with ~25% H atoms,
  the waste is ~19% of total blocks (0.25 × 0.75). Acceptable for now.
- **Question:** does the zero padding cause numerical issues in purification?
  The zero rows/columns in S make it singular. Need to ensure the S^{-1}
  computation (Newton-Schulz) handles this correctly.

### 5.5 Geometry optimization to a true minimum

- Large nanocrystals have many soft modes (surface reconstruction, breathing
  modes). FIRE may get stuck in a local minimum that is not the global one.
- **Question:** how many FIRE steps are needed for N=500? N=1000? The
  formic+azaindole dimer (56 orbs) took 100 steps and didn't fully converge
  (max|F| = 0.049, target 0.005). For 1000 atoms, this could be 1000+ steps.
- **Potential solution:** L-BFGS (better for large systems) or CG. Or:
  optimize in stages (first rough with small f_tol, then tight).

### 5.6 Hessian diagonalization for large N

- For N=1000: 3000×3000 dense symmetric eigensolve. LAPACK `dsyevd` can do
  this in ~1-10 seconds. Not a bottleneck.
- For N=10000 (future): 30000×30000 — too large for dense. Would need
  iterative methods (Lanczos) to get only low-frequency modes.
- **Current scope:** N≤1000, dense LAPACK eigensolve is fine.

---

## 6. Contracts and Tests

### 6.1 Contract: diamond cubic geometry builder

**Test:** `tests/nanocrystal_geometry.rs::test_diamond_cubic_si` (new)
- Build Si nanocrystal, radius=5 Å, H-passivated
- Assert:
  - All Si atoms have 4 neighbors (bulk) or <4 (surface, passivated by H)
  - All H atoms have 1 neighbor (Si)
  - No atoms within MIN_NEIGH_DIST of each other
  - Si-Si distances in [2.2, 2.5] Å (bulk nearest neighbor = 2.35 Å)
  - Si-H distances in [1.4, 1.6] Å (typical Si-H = 1.48 Å)
  - Atom count matches expected (~30 Si + ~30 H for R=5 Å)
- **Diagnostic:** print structure summary (N_Si, N_H, coordination histogram,
  radius of gyration). Save XYZ to `debug/nanocrystal_vib/si_R5.xyz`.

### 6.2 Contract: sparse SCC parity on Si nanocrystal

**Test:** `tests/sparse_scc_nanocrystal.rs::test_sparse_scc_si30` (new)
- System: Si₃₀H₃₀ (small nanocrystal, ~60 atoms, ~150 orbs)
- Run: GPU sparse SCC (purification route)
- Reference: CPU dense SCC (`HamiltonianBuilder::build_scc`) with LAPACK
- Tolerances (f32):
  - Energy: |dE| < 1e-3 Ha
  - Charges: max|dq| < 1e-2 e
  - Density: max|ΔK| < 1e-2 (Frobenius)
- **Diagnostic:** print convergence history (TC2 iterations, idempotency
  residual, trace). Print nnz before/after purification.

### 6.3 Contract: sparsity verification

**Test:** `tests/sparse_scaling.rs::test_sparsity_preserved` (new)
- Systems: Si nanocrystals, N = 60, 300, 800 atoms (3 sizes)
- For each: build H0/S (sparse), run purification, measure:
  - nnz(H0), nnz(S), nnz(K0), nnz(K_final)
  - nnz at each TC2 iteration (convergence history)
  - Fill ratio = nnz / N²
- **Assert:**
  - nnz(K_final) ≤ nnz(K0) × C (C = 1.5 — allow some growth but not densification)
  - Fill ratio decreases with N (scales as ~1/N for 3D sparse)
  - No NaN/Inf in any matrix
- **Output:** `debug/nanocrystal_vib/sparsity_report.tsv`

### 6.4 Contract: linear scaling verification

**Test:** `tests/sparse_scaling.rs::test_linear_scaling` (new, `--ignored`)
- Systems: Si nanocrystals, N = 60, 300, 800, 1600 atoms
- For each: measure wall time for:
  - H0/S assembly
  - One SCC iteration (purification)
  - Full SCC solve (to convergence)
  - One force evaluation (when implemented)
- **Assert:**
  - Time per SCC iteration scales as O(N^α) with α < 1.5 (target α ≈ 1.0)
  - If α > 2: FAIL — method is not linear scaling, report loudly
- **Output:** `debug/nanocrystal_vib/scaling_study.tsv` +
  `debug/nanocrystal_vib/time_vs_N.png` (log-log plot with fit line)

### 6.5 Contract: Hessian parity (small system, same geometry)

**Test:** `tests/hessian_parity.rs::test_hessian_si_small` (new)
- System: Si₈H₈ (or Si₂ from DFTB+ phonon test, ~8-20 atoms)
- Geometry: use the DFTB+ reference geometry (NOT relaxed — same geometry
  for both paths to isolate Hessian accuracy)
- Run 1: Rust sparse DFTB, f32 GPU → Hessian H_gpu
- Run 2: Rust dense DFTB, f64 CPU → Hessian H_cpu
- Run 3 (optional): DFTB+ Fortran → Hessian H_fortran
- Compare:
  - max|H_gpu - H_cpu| per element
  - Eigenvalues of mass-weighted Hessian: |dω| per mode
  - Frequency spectrum: plot ω_gpu vs ω_cpu
- **Tolerances:**
  - Hessian elements: max|dH| < 1e-2 Ha/Å² (f32 vs f64)
  - Frequencies: max|dω| < 10 cm⁻¹ for modes > 100 cm⁻¹
  - Low modes (< 100 cm⁻¹): max|dω| < 50 cm⁻¹ (more sensitive)
- **Assert:**
  - No pathological imaginary modes in H_gpu that are not in H_cpu
  - (If H_cpu has imaginary modes too: both are wrong, report)
  - Number of imaginary modes in H_gpu ≤ number in H_cpu + 1
- **Output:**
  - `debug/nanocrystal_vib/hessian_parity.tsv` (per-mode comparison)
  - `debug/nanocrystal_vib/frequency_spectrum.png` (overlayed spectra)

### 6.6 Contract: no imaginary modes after tight optimization

**Test:** `tests/nanocrystal_vib.rs::test_no_imaginary_modes_si300` (new, long)
- System: Si nanocrystal, ~300 atoms, H-passivated
- Step 1: Optimize geometry with sparse forces, f_tol = 1e-4 Ha/Å (tight)
- Step 2: Compute Hessian (finite differences, h=1e-3 Å or tuned)
- Step 3: Diagonalize mass-weighted Hessian
- **Assert:**
  - All 6 zero modes have |ω| < 10 cm⁻¹ (translations + rotations)
  - All other modes have ω > 0 (no imaginary modes)
  - If imaginary modes exist: report them loudly with frequency and
    eigenvector (which atoms are involved)
- **Diagnostic output:**
  - `debug/nanocrystal_vib/si300_optimized.xyz`
  - `debug/nanocrystal_vib/si300_hessian.dat`
  - `debug/nanocrystal_vib/si300_spectrum.png`
  - `debug/nanocrystal_vib/si300_optimization.png` (energy + max|F| vs step)
- **Note:** if imaginary modes appear, this is a DIAGNOSTIC, not a failure
  to hide. Report: (a) which modes are imaginary, (b) likely cause (force
  noise, insufficient SCC convergence, finite-difference step too small).

### 6.7 Contract: large system performance

**Test:** `tests/nanocrystal_perf.rs::bench_si1000_optimization` (new, `--ignored`)
- System: Si nanocrystal, ~1000 atoms, H-passivated
- Measure:
  - Optimization wall time (to f_tol = 1e-3)
  - Hessian computation wall time
  - Frequency diagonalization wall time
  - Peak GPU memory usage
  - SCC iterations per optimization step (with warm-start)
- **Output:** `debug/nanocrystal_vib/si1000_performance.tsv`
- **Goal:** optimization < 2 hours, Hessian < 1 hour (on NVIDIA GPU).

### 6.8 Monitoring: fail-loud invariants

Throughout all tests:
- All energies, forces, Hessian elements finite (no NaN/Inf)
- Hessian is symmetric: max|H_ij - H_ji| < 1e-4 (finite-difference symmetry)
- Mass-weighted Hessian eigenvalues: 6 modes near zero (|λ| < threshold),
  rest positive. If negative: report as imaginary mode with context.
- Sparse matrix nnz does not explode: nnz(K_iter) / nnz(K_0) < 10 throughout
  purification. If it does: report densification, the method is not working.
- SCC convergence: residual monotonically decreasing (or report oscillation).
- No silent fallback to dense (if sparse path fails, crash with context).

---

## 7. Implementation Plan (phased)

### Phase 0: Diamond cubic geometry builder (2-3 days)
- Implement `build_diamond_cubic_nanocrystal` in `geometry/mod.rs`
- H-passivation logic
- Test: Contract 6.1
- Build geometries for N = 60, 300, 800, 1600 and save to `debug/nanocrystal_vib/`

### Phase 1: Sparse SCC energy and convergence (3-5 days)
- Implement sparse Frobenius trace kernel (Tr(K·H0))
- Implement sparse H_scc update (elementwise on BSR4 structure)
- Implement full sparse SCC loop (H_scc → purify → charges → gamma → mix)
- Test: Contract 6.2 (parity on small Si nanocrystal)
- Test: Contract 6.3 (sparsity verification)

### Phase 2: Sparse forces (5-7 days)
- Implement sparse analytic force kernels (F_nonSCC, F_rep, F_SCC_dc, F_SCC_shift)
- Or: implement finite-difference energy forces (simpler, slower)
- Test: force parity vs CPU dense on small Si system
- Benchmark: sparse force time vs N

### Phase 3: Geometry optimization (3-5 days)
- Implement FIRE/L-BFGS with sparse forces
- Warm-start charges between optimization steps
- Test: optimize Si₃₀H₃₀ to f_tol=1e-4
- Benchmark: Contract 6.4 (linear scaling verification)

### Phase 4: Hessian and vibrational spectra (5-7 days)
- Implement finite-difference Hessian (batched displaced geometries)
- Implement mass-weighting and diagonalization (LAPACK)
- Implement frequency extraction + zero mode removal
- Test: Contract 6.5 (Hessian parity, small system, same geometry)
- Test: Contract 6.6 (no imaginary modes, Si₃₀₀)
- Plot: frequency spectra

### Phase 5: Large system production run (3-5 days)
- Optimize Si₁₀₀₀ nanocrystal
- Compute Hessian and vibrational spectrum
- Benchmark: Contract 6.7 (performance)
- Plot: scaling study, sparsity study, frequency spectrum

### Phase 6: Accuracy investigation (ongoing)
- Compare f32 vs f64 Hessians
- Sweep finite-difference step h
- Sweep SCC tolerance
- Identify minimum parameters for imaginary-mode-free spectra
- Compare with DFTB+ Fortran phonon reference (Si₂ dimer, small clusters)

---

## 8. Geometry Generation — External Repos (do NOT duplicate here)

**Policy:** We do NOT want to pollute the dftbplus repo with nanocrystal
geometry-building machinery that already exists in FireCore. Instead,
generate geometries **in FireCore** and export `.xyz` / `.mol2` / `.npz`
files into `data/xyz/` here. This section documents where the tools live
and how to use them.

### 8.1 FireCore — Si/diamond nanocrystal generator (JS + Python)

**Repo:** `/home/prokop/git/FireCore`
**Codemap:** `FireCore/CODEMAP.md` (entry point)
**Topical audit:** `FireCore/doc/topical_audit/Nanocrystal_Vibrations.md`
**Working hub:** `FireCore/tests/tSiNCs/README.md` + `AGENTS.md`

FireCore has a very rich nanocrystal generator supporting spherical cuts,
Miller-plane facets, Wulff shapes, H-passivation, bridge defects, silyl
passivation, and ensemble batch generation. Both JavaScript (feature-complete)
and Python (spherical cuts) CLIs are available.

**Key files:**

| File | Role |
|---|---|
| `web/molgui_webgpu/Nanocrystals.js` | **Core JS library**: CIF → cuts → prune → H-cap → bridges / `silyl100Prob` / `silyl111Prob` / fuse. Wulff shapes, Miller planes, defect operators. |
| `web/molgui_webgpu/EditableMolecule.js` | Molecular graph with editing ops |
| `web/molgui_webgpu/CrystalUtils.js` | Crystal symmetry, primitive cell ops |
| `web/common_js/npzIO.js` | NPZ I/O (crystal arrays, topology) |
| `web/common_js/nanocrystalSvg.js` | SVG export, ring detection viz |
| `tests/tSiNCs/nanocrystals.mjs` | **Unified CLI**: `generate`, `ensemble`, `topology`, `audit`, `nonbond`, `rings` subcommands |
| `tests/tSiNCs/gen_nanocrystals.py` | **Python CLI**: spherical cuts native; Miller planes delegate to Node |
| `pyBall/nanocrystal_gen.py` | **Python sphere-cut builder**: `build_spherical_nanoparticle`, `save_xyz`, `find_cap_hh_pairs`. Parity target for JS. |
| `pyBall/nanocrystal_pipeline.py` | **NPZ pipeline CLI**: `relax` → `hessian` → `spectrum` → `accumulate` (stages 01–05) |
| `pyBall/FTIR.py` | Vibrational spectra post-processing: `build_hessian_from_linear_topology`, rigid-mode projection, mass matrix |
| `tests/tSiNCs/crosscheck_nanocrystal_generators.py` | JS vs Python generator parity verification |
| `tests/tSiNCs/chem_atlas.json` | Atlas config for batch ensemble generation |

**Crystal primitive cells** (in `cpp/common_resources/crystals/`):
- `Si_primitive.xyz`, `Si_primitive.cif`, `Si-sym.cif`, `Si_conventional.xyz`
- `diamond_primitive.xyz`, `diamond_primitive.cif`, `C_diamond_sym.cif`,
  `diamond_conventional.xyz`

**Pre-built fixtures** (in `tests/tSiNCs/fixtures/`):
- `si_1nm_passivation/` — nine-crystal NPZ pipeline gallery (stages 01–05
  per crystal: init mol2, relaxed, topology, hessian, spectrum)
- `npz_viewer/` — minimal viewer smoke fixtures
- `vibration_benchmarks/` — benchmark NPZ structures

**Workflow to generate Si nanocrystal geometries:**

1. **Spherical cut (Python, simplest):**
   ```bash
   cd /home/prokop/git/FireCore
   python3 tests/tSiNCs/gen_nanocrystals.py \
       --cutMode sphere --element Si --sphere-r 10.0 --sphere-nrep 5 \
       --caps H --outDir tests/tSiNCs/OUT_nanocrystals_py
   # → writes Si_sphere_R10.0_nrep5_natXXX.xyz
   ```

2. **Miller-plane facets (JS, feature-complete):**
   ```bash
   cd /home/prokop/git/FireCore
   node tests/tSiNCs/nanocrystals.mjs generate \
       --cif cpp/common_resources/crystals/Si-sym.cif \
       --cutMode planes --planeTemplates a111 \
       --nx-range 3,3 --ny-range 3,3 --nz-range 3,3 \
       --caps H --outDir tests/tSiNCs/OUT_nanocrystals
   # → writes mol2 + xyz + viewer JSON
   ```

3. **Wulff shape (chemically realistic facets):**
   ```bash
   node tests/tSiNCs/nanocrystals.mjs generate \
       --cif cpp/common_resources/crystals/Si-sym.cif \
       --wulffShape octahedron --wulffNLat 1.0 \
       --caps H --outDir tests/tSiNCs/OUT_nanocrystals
   ```

4. **Batch ensemble (atlas):**
   ```bash
   node tests/tSiNCs/nanocrystals.mjs ensemble \
       --atlas tests/tSiNCs/chem_atlas.json \
       --output-dir tests/tSiNCs/OUT_chem_atlas
   ```

5. **Copy generated `.xyz` to dftbplus:**
   ```bash
   cp /home/prokop/git/FireCore/tests/tSiNCs/OUT_nanocrystals_py/Si_sphere_R10.0_*.xyz \
      /home/prokop/git/dftbplus/data/xyz/
   ```

6. **Verify generator parity** (JS vs Python):
   ```bash
   cd /home/prokop/git/FireCore/tests/tSiNCs
   python3 crosscheck_nanocrystal_generators.py
   ```

**For diamond (C) nanocrystals:** same tools, just use `--element C` and
`--cif cpp/common_resources/crystals/C_diamond_sym.cif`.

### 8.2 FireCore — Hessian and vibration pipeline (reference)

FireCore also has a complete Hessian + vibration pipeline (using classical
force fields, not DFTB). This is useful as a **reference** for our DFTB-based
Hessian:

| File | Role |
|---|---|
| `pyBall/nanocrystal_pipeline.py` | NPZ pipeline: `relax` (MMFF) → `hessian` (topology-linear K) → `spectrum` (eigh) → `accumulate` |
| `pyBall/FTIR.py` | `build_hessian_from_linear_topology`, rigid-mode projection, `vibration_spectrum_from_modes` |
| `pyBall/MMFF.py` | `getHessian3Nx3N(inds, dx)` — 3N×3N Hessian via central FD in C++ |
| `spammm/dynamics/Vibrations.py` | (in SPAMMM) Normal-mode analysis: DFTB/UFF/SPFF Hessian, rigid-mode projection, mode analysis |

**Key lesson from FireCore** (documented in
`doc/Topics/FTIR_Nanocrystals/Hessian_at_own_minimum.md`):
> Harmonic spectrum = Hessian at that method's **own minimum**. Relax with
> the **same** potential until f_max < f_conv, then Hessian. DFTB geometry +
> MMFF Hessian is FFfit only, never a spectrum. Smoking gun: MMFF modes
> piled from ~18 cm⁻¹ or hundreds of large imaginaries.

This directly applies to our task: the Hessian must be computed at the
DFTB-optimized geometry, not at a geometry from another method.

### 8.3 SPAMMM — Vibrational analysis (reference)

**Repo:** `/home/prokop/git/SPAMMM`
**Topical audit:** `SPAMMM/doc/Topics/Vibrations.md`

SPAMMM has a `Vibrations.py` module for normal-mode analysis from molecular
Hessians (DFTB or GPU force-field FD). Useful as reference for:
- Rigid-body mode projection (translations + rotations removed)
- In-plane vs out-of-plane mode classification
- Unit conversion (cm⁻¹, meV, THz, kcal/mol)
- DFTB+ Hessian I/O (`write_dftb_input_hessian`, `read_hessian`)

| File | Role |
|---|---|
| `spammm/dynamics/Vibrations.py` | `run_vibrations(mol, backend=...)` — Hessian assembly, rigid-mode projection, mode analysis |
| `spammm/dynamics/VibrationPlot.py` | Top-view mode plots (in-plane arrows + z circles) |
| `spammm/quantum/DFTB_utils.py` | `write_dftb_input_hessian`, `read_hessian`, `hessian_hartree_bohr_to_eV_angstrom` |

### 8.4 What NOT to build in dftbplus

- **No diamond cubic / Wulff / Miller-plane builder** — use FireCore's
  `Nanocrystals.js` / `gen_nanocrystals.py`
- **No H-passivation logic** — FireCore handles Si-H, C-H, silyl, bridge defects
- **No MMFF Hessian pipeline** — FireCore's `nanocrystal_pipeline.py` does this
  (classical); our task uses DFTB Hessian
- **No rigid-mode projection** — can reference SPAMMM's `Vibrations.py` or
  FireCore's `FTIR.py` for the algorithm; implement in Rust if needed
- **No NPZ I/O** — use `.xyz` files as the interchange format

The dftbplus Rust crate should only **consume** the generated `.xyz` files
and compute DFTB energies, forces, Hessians, and spectra. All geometry
building stays in FireCore.

---

## 9. File Ownership

| File | Status | Owner |
|---|---|---|
| `rust_dftb/src/methods/sparse/gpu_sparse.rs` | exists, extend with energy/forces | this task |
| `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl` | exists, extend with trace/force kernels | this task |
| `rust_dftb/src/methods/sparse/sparse_scc.rs` | new (sparse SCC loop) | this task |
| `rust_dftb/src/methods/sparse/sparse_forces.rs` | new (sparse analytic forces) | this task |
| `rust_dftb/src/core/hessian.rs` | new (finite-difference Hessian) | this task |
| `rust_dftb/src/core/phonon.rs` | new (mass-weight, diagonalize, frequencies) | this task |
| `rust_dftb/tests/nanocrystal_geometry.rs` | new (validates imported XYZ from FireCore) | this task |
| `rust_dftb/tests/sparse_scc_nanocrystal.rs` | new | this task |
| `rust_dftb/tests/sparse_scaling.rs` | new | this task |
| `rust_dftb/tests/hessian_parity.rs` | new | this task |
| `rust_dftb/tests/nanocrystal_vib.rs` | new | this task |
| `rust_dftb/tests/nanocrystal_perf.rs` | new | this task |
| `rust_dftb/examples/nanocrystal_optimize.rs` | new (production CLI) | this task |
| `rust_dftb/examples/nanocrystal_phonons.rs` | new (production CLI) | this task |
| `scripts/plot_nanocrystal_spectrum.py` | new | this task |
| `scripts/plot_scaling_study.py` | new | this task |
| `data/xyz/si_nanocrystal_*.xyz` | generated by FireCore, imported | external |
| `data/xyz/diamond_nanocrystal_*.xyz` | generated by FireCore, imported | external |

---

## 10. SK File Notes

- **siband-1-1:** best for pure Si (Si-Si, Si-H). Does NOT have C-C, C-H.
  Use for Si nanocrystals.
- **matsci-0-3:** has Si-Si, Si-C, C-C, C-H, Si-H. Use for diamond (C)
  nanocrystals and mixed Si-C systems.
- **pbc-0-3:** has Si-Si, Si-C, C-C, C-H, Si-H. Designed for periodic
  systems but works for clusters too. May have better repulsive potentials
  for bulk-like environments.
- **RUST_DFTB_SK_DIR** must point to the appropriate slako directory.
- **Recommendation:** use siband-1-1 for Si nanocrystals, matsci-0-3 for
  diamond nanocrystals. Benchmark both on small systems to see which gives
  better phonon agreement with experiment/DFT.

---

## 11. Related Documents

- `doc/prokop/tasts/GPU_Sparse_Resident_Integration/task.md` — prior task
  that integrated the device-resident sparse purification path.
- `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` —
  section 7.5 tracks sparse BSR4 purification status.
- `doc/prokop/topical_audit/sparse_tc2_purification.md` — technical audit
  of the TC2 purification implementation.
- `doc/prokop/topical_audit/davidson_eigensolver.md` — Davidson eigensolver
  (not needed for purification, but relevant for frontier orbitals).
- `test/app/phonons/Si/` — DFTB+ Fortran phonon reference for Si₂.
- `doc/prokop/AGENTS/guidelines/efficiency.md` — efficiency rules (no
  allocation in hot loops, three-tier data lifetime, etc.).

---

## 12. Key Differences from Task 1 (H-Bond Relaxed Scan)

| Aspect | Task 1 (H-bond scan) | Task 2 (Nanocrystal vib) |
|---|---|---|
| System type | Nucleobase pairs (compact, dense) | Si/diamond nanocrystals (extended, sparse) |
| System size | ~30 atoms, ~87 orbs | 500-1000 atoms, 2000-4000 orbs |
| Number of systems | Many (batched, 100-1000) | One (single system) |
| GPU strategy | Batched dense SCC, N>64 extension | Sparse BSR4 purification, O(N) |
| Key bottleneck | N>64 eigensolver, GPU forces | Sparsity preservation, force evaluation |
| Forces | Analytic (port to GPU) | Analytic (sparse) or finite-difference |
| Post-processing | Relaxed 2D PES, barrier analysis | Hessian, vibrational frequencies |
| Accuracy concern | SCC convergence on polar systems | f32 Hessian noise → imaginary modes |
| SK set | mio-1-1 (H,C,N,O) | siband-1-1 (Si) or matsci-0-3 (C,Si) |
| Geometry source | SPAMMM (ASCII art + coordinate_scan) | FireCore (Nanocrystals.js + gen_nanocrystals.py) |
