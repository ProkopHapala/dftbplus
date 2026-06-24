# USER

sorry there was connection autage, I repeate:

plese review @dftb_hamiltonian.cl form point of view of parallel GPU efficiency, find all possible problems and bottleneks and possibility of improving speed.

consider following
1) unnecessary branching (if) especially tread divergnece
2) global memory access patters
3) efficient use of workgroups and group shared memory
4) pre-calcualtion of things (do not recalculate many tomes some complex clalculations iw we can cache the results in shared or private memoryu and reuse it )
5) efficienect saturation of GPU consiering we have multiple system

do not program yet, rather write review and suggest possible improvements

---

# GLM-5.2

# GPU Efficiency Review: [dftb_hamiltonian.cl](cci:7://file:///home/prokophapala/git/dftbplus/rust_dftb/src/methods/dftb/dftb_hamiltonian.cl:0:0-0:0)

## 1. Branching & Thread Divergence

### Critical: `gamma_full` has 3 divergent branches (L78, L86-87, L87-91)
```
if (r < 1.0e-10f)          // self-interaction, rare but divergent
if (du < 0.0f) du = -du;   // fabs — use fabs() builtin instead
if (du < 1.0e-4f)           // same-U vs different-U path
```
**Problem:** In `onsite_and_va`, all threads in a workgroup process different atoms with different neighbor lists. The `du < 1e-4` branch will diverge frequently — same-element pairs take the cheap `exp_gamma_same_u` path, mixed pairs take the expensive `gamma_sub_exprn` path (2× exp calls). Within a single warp, some threads may take 2 `exp()` calls while others take 1.

**Suggestions:**
- Replace `if (du < 0.0f) du = -du;` with `du = fabs(du);` — this is a single instruction on most GPUs, no branch.
- Pre-sort neighbor lists by species-pair similarity so warps process homogeneous (same-U vs different-U) neighbors together. Or precompute a boolean `same_u` flag per neighbor pair on the host and store it in the CSR list, eliminating the runtime branch entirely.
- The `r < 1e-10` branch is extremely rare (only self-interaction). Consider handling self-interaction separately on the host (it's just `0.5 * (u1 + u2)`) and removing it from the kernel entirely.

### Minor: `cubic_interp_params` has 3 branches (L111, L117, L118)
```
if (r < 0.0f || r >= ...) return -1;
if (i < 1) i = 1;
if (i > n_grid - 3) i = n_grid - 3;
```
**Problem:** The range check at L111 should never trigger if pairs are pre-filtered by cutoff on the host. The clamp branches at L117-118 are predictable but still cost instruction slots.

**Suggestion:** Remove the range check entirely (guarantee correctness by host-side filtering). Replace the two clamp branches with `clamp()` builtin: `i = clamp(i, 1, n_grid - 3);` — single instruction.

### Minor: `interp_sk_*_indexed` all check `if (base < 0)` (L128, L137, L152)
**Problem:** These branches are dead code if the range check is removed from `cubic_interp_params`. They force a comparison + branch on every interpolation call.

**Suggestion:** Remove once host-side filtering guarantees valid `base`.

### Good: `block_type` branch in `assemble_pairs` (L387-415)
**Already mitigated** — the comment at L11 correctly notes this is uniform per launch. No divergence here since all threads in a workgroup take the same path. ✓

## 2. Global Memory Access Patterns

### Critical: `onsite_and_va` — uncoalesced H_out writes (L305-308)
```
H_out[base_H + off * n_orbs + off]     = e.x;  // diagonal
H_out[base_H + (off+1)*n_orbs + (off+1)] = e.y;  // strided
H_out[base_H + (off+2)*n_orbs + (off+2)] = e.y;  // strided
H_out[base_H + (off+3)*n_orbs + (off+3)] = e.y;  // strided
```
**Problem:** Consecutive threads (`a = tid, tid+1, ...`) write to positions separated by `n_orbs` stride. This is fully uncoalesced — each thread triggers a separate 128-byte cache line fetch/store. For a 20-orbital fragment with 64 threads, this is 64 separate cache lines for 64 writes.

**Suggestion:** Since only diagonal elements are written, precompute the diagonal on the host (it's just `onsite_es_ep` placed at known positions) and skip this entirely in the kernel. Alternatively, use a separate small kernel that writes contiguous diagonal values, or store diagonals in a separate compact array and scatter them with a dedicated pass.

### Critical: `assemble_pairs` — `fragments[p.replica]` is random-access (L373)
**Problem:** Each thread reads `fragments[p.replica]` where `replica` varies per pair. Pairs are sorted by replica within a bucket, so adjacent threads likely access the same fragment — but this is not guaranteed. Each `Fragment` is 16 bytes, so a 128-byte cache line holds 8 fragments. If pairs span many fragments, this causes cache thrashing.

**Suggestion:** Pre-load fragment metadata into `__local` memory at kernel start (it's small — `n_frags * 16` bytes). For typical workloads with <100 fragments, this is <1.6KB, well within local memory budget.

### Moderate: `assemble_pairs` — `V_a[atom_off + p.atom_i]` random access (L378-379)
**Problem:** `V_a` is accessed with per-pair indices. Adjacent pairs in the same bucket may belong to different fragments, causing uncoalesced reads.

**Suggestion:** If pairs are sorted by replica (already done on host), adjacent threads will read nearby `V_a` entries. Ensure the host sort is stable and groups by replica first, then by atom_i within replica.

### Moderate: `onsite_and_va` — `charges[gb]` and `atom_species[gb]` in inner loop (L325-326)
**Problem:** For each neighbor, we read `charges[gb]` and `atom_species[gb]` from global memory. These are accessed with `b_local` indices which are arbitrary (CSR neighbor order).

**Suggestion:** Pre-load `charges` and `atom_species` for the current fragment into `__local` memory at kernel start. A fragment with N atoms needs `N * 8` bytes (4 for charge, 4 for species). For N=50, that's 400 bytes — trivial. This converts the inner loop from global to local memory reads.

### Good: SK table caching into `__local` (L362-365) ✓
Already done. But see issue below about local memory waste.

## 3. Workgroups & Shared Memory

### Critical: SK_GRID_MAX * 4 * 2 = 8192 floats = 32KB per workgroup (L359-360)
```
__local float l_sk_h[SK_GRID_MAX * 4];  // 4096 * 4 = 16KB
__local float l_sk_s[SK_GRID_MAX * 4];  // 16KB
```
**Problem:** This always allocates for the worst case (4-channel, 1024 grid points). For block_type 0 (1-channel), only 1024 floats (4KB) are used per table — **75% waste**. This limits occupancy: many GPUs have 32-48KB local memory per workgroup. With 32KB used, only 1-2 workgroups can be resident per compute unit, severely limiting latency hiding.

**Suggestion:** 
- Use a kernel compile-time `#define N_SK_COLS` and compile 3 variants of the kernel (one per block type). Then allocate `__local float l_sk_h[SK_GRID_MAX * N_SK_COLS]` — exact size, no waste.
- Or use `__local float l_sk_h[SK_GRID_MAX * 4]` only for block_type 2, and separate kernels with smaller arrays for block_type 0 and 1.
- Also consider whether `SK_GRID_MAX=1024` is realistic. Typical SK tables have 20-100 grid points. If max is 100, that's 800 bytes per table — 50× reduction.

### Moderate: `onsite_and_va` uses no shared memory at all
**Problem:** The kernel reads `charges`, `atom_species`, `hubbard_u`, `neigh_j`, `neigh_r` all from global memory with no caching. For small fragments (10-50 atoms), all per-atom data fits easily in shared memory.

**Suggestion:** Load `charges[atom_off..atom_off+n_atoms]`, `atom_species[atom_off..atom_off+n_atoms]`, and `hubbard_u[0..n_species]` into `__local` at kernel start. This eliminates repeated global reads in the neighbor loop.

### Moderate: Workgroup size not specified
**Problem:** No `__attribute__((reqd_work_group_size(X, 1, 1)))` or host-side size hint. The optimal size depends on fragment size and occupancy considerations.

**Suggestion:** For `onsite_and_va`, use workgroup size = `min(64, next_pow2(max_atoms_per_fragment))`. For `assemble_pairs`, use 64 or 128 depending on pair count. Consider auto-tuning.

## 4. Pre-calculation & Caching

### Critical: `rotate_4x4` recomputes `v.yzw * v.yzw * diff4.yzw + sk.w * v.yzw` 3× (L190, L195, L200)
**Problem:** The TODO comment at L185-187 already identifies this. The pp outer product `v.yzw * v.yzw * diff + sk.w * v.yzw` is identical for rows 1, 2, and 3 — only the `.x` component changes (it's `sk.y * m`, `sk.y * n`, `sk.y * l` respectively). This wastes 6 mul + 3 add per rotation.

**Suggestion:** Compute the pp part once:
```c
float3 pp = v.yzw * v.yzw * diff + sk.w * v.yzw;
// Then just set .x per row and reuse pp for .yzw
```

### Moderate: `gamma_full` recomputes `exp(-tau*r)` potentially 2× in different-U path (L90-91)
**Problem:** `gamma_sub_exprn(r, tau1, tau2)` computes `exp(-tau1 * r)` and `gamma_sub_exprn(r, tau2, tau1)` computes `exp(-tau2 * r)`. These are 2 separate `exp()` calls. The `exp_gamma_same_u` path also calls `exp()` once.

**Suggestion:** For the different-U path, both `exp(-tau1*r)` and `exp(-tau2*r)` are needed. These could be computed once and passed in, but the function signature would change. Alternatively, accept this as unavoidable — `exp` is already relatively fast on modern GPUs.

### Moderate: `h1_factor` computed but `sk_h + h1_factor * sk_s` done after interpolation (L401, L413)
**Problem:** For block_type 1 and 2, we interpolate `sk_s` and `sk_h` separately, then add `h1_factor * sk_s`. Since interpolation is linear, we could pre-compute `sk_h + h1_factor * sk_s` on the table values before rotation, saving one rotation call.

**Current code already does this partially** (L401: `rotate_1x4(..., sk_h + h1_factor * sk_s, &blk)`) — good. But for block_type 2 (L407-413), `sk_s` is interpolated, rotated, and written to `S_out`, then `sk_h` is interpolated separately and `sk_h + h1_factor * sk_s` is rotated again. The `sk_s` interpolation result is reused (good), but two separate rotations are done.

**Suggestion:** Since `rotate_4x4` is linear, compute `sk_combined = sk_h + h1_factor * sk_s` after both interpolations, then rotate once for H. The current code does `rotate(sk_h + h1_factor * sk_s)` which is correct and only does one rotation for H — but it also does a separate rotation for S. This is unavoidable since we need both H and S. ✓ (already optimal)

### Good: `cubic_interp_params` computed once per pair (L385) ✓
The index and weights are computed once and reused for both H and S interpolation. ✓

## 5. GPU Saturation with Multiple Systems

### Critical: `onsite_and_va` — one workgroup per fragment (L289)
**Problem:** If fragments are small (e.g., 5-20 atoms), a single workgroup of 64 threads has most threads idle. With 10 fragments, that's only 10 workgroups — not enough to saturate a modern GPU with 20-80 compute units.

**Suggestion:** 
- **Option A:** Launch multiple workgroups per fragment, each handling a subset of atoms. Use atomics or a reduction for `V_out` (since multiple workgroups contribute to the same atom's V_A). This increases parallelism but adds synchronization overhead.
- **Option B:** Pack multiple small fragments into a single workgroup. Thread 0-15 handles fragment 0, threads 16-31 handle fragment 1, etc. This requires careful indexing but avoids atomics.
- **Option C:** For very small fragments, process all atoms in a single warp (32 threads) and launch 2 fragments per workgroup.

### Moderate: `assemble_pairs` — one kernel launch per bucket (L339)
**Problem:** Each species-pair bucket is a separate kernel launch. With 3 block types × N² species pairs, this could be dozens of launches. Each launch has ~5-10μs overhead. For small buckets (few pairs), the launch overhead dominates.

**Suggestion:**
- **Merge small buckets:** Combine multiple species-pair buckets with the same block_type into a single launch. Add `sk_table_idx` to `PairEntry` and load the appropriate SK table per pair. This requires either: (a) loading all SK tables into local memory (may not fit), or (b) reading SK from global memory per pair (slower but avoids launch overhead).
- **Persistent kernel:** Launch one kernel with enough workgroups to fill the GPU, and have each workgroup pull bucket indices from a global queue. This eliminates all launch overhead but requires atomic counter for work distribution.

### Moderate: No occupancy control for `assemble_pairs`
**Problem:** With 32KB of local memory per workgroup (see issue #3), only 1-2 workgroups can be resident per compute unit. This means the GPU cannot hide memory latency by switching between workgroups.

**Suggestion:** Reduce local memory usage (see #3) to allow 4+ workgroups per compute unit. Target <8KB local memory per workgroup for good occupancy on most GPUs.

### Good: Pairs sorted by replica within bucket ✓
This improves coalescing for `fragments[]` and `V_a[]` accesses. ✓

## Summary: Priority-ranked improvements

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| **P0** | SK local memory over-allocation (32KB → 4-8KB) | 2-4× occupancy | Low — compile-time variants |
| **P0** | `rotate_4x4` redundant pp computation (3×) | ~30% of rotation cost | Low — already identified |
| **P0** | `onsite_and_va` uncoalesced diagonal writes | 64× cache line waste | Low — move to host or separate kernel |
| **P1** | `gamma_full` branch divergence (du < 1e-4) | Warp serialization | Medium — pre-sort or pre-flag |
| **P1** | No shared memory caching in `onsite_and_va` | Global memory latency | Medium — load fragment data to local |
| **P1** | Small-fragment under-saturation in `onsite_and_va` | GPU idle | Medium — multi-fragment workgroups |
| **P2** | Multiple kernel launches for small buckets | Launch overhead | Medium — merge or persistent kernel |
| **P2** | `fragments[]` random access in `assemble_pairs` | Cache thrashing | Low — preload to local |
| **P2** | Dead branch checks in interpolation (base < 0) | Minor instruction waste | Low — remove |
| **P3** | `fabs()` instead of branch for `du` | Single instruction | Trivial |
| **P3** | `clamp()` instead of branches in interp params | Single instruction | Trivial |

---

# USER

we should think deeply how we can do SCF cycle on GPU efficiently for multiple fragments in parralel. Consider we have hamiltoonians of decent size from N= 100 to 1000 dimension, and we have many such systems, we should do follogin operations
1) assemble hamiltonians of all fragments
2) solve gneralized eigenvalue HC=SCe (can we use jacobi rotation?), we perhpas can precompute lowdin transform
3) compute density matrix
4) perhaps also some matrix multiplications
... we should do all of this in tiled apporach considering worgroupsize and loading tiles to local mmeory

do not program rather thing it throgh, analyse problem, decomposed it to subplroblems, and adress each of it in details. 

---

# GLM 5.2

# GPU SCF Cycle Analysis for Batched DFTB Fragments

## Problem Overview

We need to run the SCF cycle for many independent fragments (N=100–1000) simultaneously on GPU. The key operations per SCF iteration are:

1. Assemble H_scc = H0 + shift contributions
2. Solve generalized eigenvalue problem HC = SCε (or equivalent)
3. Compute density matrix D
4. Compute charges and update shifts

The critical insight: **S and H0 are constant across SCF iterations**. Only the shift contributions change. This allows precomputing expensive one-time transforms.

## Subproblem 1: Generalized Eigenvalue Problem

### The Core Difficulty

The generalized eigenvalue problem HC = SCε is the bottleneck. Three approaches:

### Option A: Löwdin Orthogonalization + Standard Eigenvalue Problem

Transform to standard form using X such that X^T S X = I:
- **Symmetric orthogonalization**: X = S^{-1/2} (requires eigendecomposition of S)
- **Cholesky-based**: X = L^{-1} where S = L L^T (cheaper, GPU-friendly)

Then: H' = X^T H X, solve H'C' = C'ε, back-transform C = X C'.

**Cholesky-based is preferred** because:
- Cholesky is O(N³/3) vs eigendecomposition O(N³) with larger constant
- Triangular operations map better to tiled GPU algorithms
- S is constant → **L^{-1} computed once, reused every SCF iteration**

The cost per SCF iteration becomes: 2 batched matmuls (L^{-1} @ H @ L^{-T}) + standard eigenvalue solve.

But we still need to solve the standard eigenvalue problem...

### Option B: Jacobi Rotation (Parallelizable Diagonalization)

Classical Jacobi iteratively zeros off-diagonal elements via 2×2 rotations.

**Parallel Jacobi on GPU:**
- Each sweep: N/2 independent rotations can be applied simultaneously
- For N=100: 50 rotations per sweep, ~5-10 sweeps → 250-500 rotation rounds
- For N=1000: 500 rotations per sweep, ~10-15 sweeps → 5000-7500 rounds

**Problem**: Each rotation modifies 2 rows/columns, creating dependencies. Parallel Jacobi uses non-overlapping rotation pairs, but:
- Need careful scheduling (Brent-Luk ordering)
- Each rotation is a tiny 2×2 problem → low arithmetic intensity per thread
- Global memory traffic for row/column updates is high
- Convergence is O(N²) total rotations → O(N⁴) work for full diagonalization

**Verdict**: Works for N≤200 but doesn't scale to N=1000 efficiently.

### Option C: Density Matrix Purification (NO diagonalization at all) ⭐

**This is the most GPU-friendly approach.** Instead of computing eigenvalues/eigenvectors, directly compute the density matrix using only matrix multiplication.

**Palser-Manolopoulos purification** (for gapped systems, which DFTB molecules are):

```
1. Scale H': H_norm = (H' - μ_min I) / (μ_max - μ_min)  → eigenvalues in [0,1]
2. Initial guess: D_0 = I - H_norm  (eigenvalues > 0.5 → occupied)
   - Or: D_0 = fermi(-H'/kT) with bisection on μ for correct trace
3. Purify: D_{n+1} = 3D_n² - 2D_n³  (quadratically convergent to idempotent D)
4. Repeat ~10-15 times until ||D² - D|| < tol
```

**Why this is ideal for GPU:**
- **Only matrix multiplication** — the single best-optimized operation on GPU
- No diagonalization, no eigendecomposition
- All fragments processed in batched matmul
- Matrix multiply has high arithmetic intensity (O(N³) FLOPs / O(N²) data)
- Purification converges quadratically: ~10 iterations for 10⁻⁸ accuracy

**Cost per SCF iteration:**
- 2 matmuls (Löwdin transform) + ~20 matmuls (purification, 2 per iteration × 10) = **22 batched matmuls**

**μ_min, μ_max estimation:**
- Gershgorin circle theorem: μ_min ≥ min_i(H'_ii - Σ_{j≠i}|H'_ij|), μ_max ≤ max_i(H'_ii + Σ_{j≠i}|H'_ij|)
- Or reuse from previous SCF iteration (warm restart)
- Row sums computed in parallel (one reduction per fragment)

### Recommended: Option C (Purification)

```
Precompute (once per fragment):
  S → Cholesky → L → L^{-1}

Per SCF iteration:
  H_scc = H0 + shifts              (element-wise, O(N²))
  H' = L^{-1} @ H_scc @ L^{-T}    (2 batched matmuls, O(N³))
  estimate μ_min, μ_max            (row sums + reduction, O(N²))
  D = purify(H', μ_min, μ_max)     (~10 × 2 batched matmuls, O(N³))
  q_A = diag(D @ S)_atoms          (1 batched matmul + reduction, O(N³))
  update shifts                    (element-wise, O(N²))
```

## Subproblem 2: Tiled Batched Matrix Multiplication

This is the workhorse — called ~22 times per SCF iteration. Must be highly optimized.

### Tiling Strategy

For C = A @ B where A, B, C are N×N:

**Tile size T** determined by local memory budget:
- Need 3 tiles (A, B, C) of T×T floats: 3 × T² × 4B ≤ local_mem
- With 48KB: T² ≤ 4096 → **T = 64** (but 49KB, borderline) or **T = 32** (safe, 12KB)
- T=32 is the sweet spot for occupancy

**Workgroup mapping:**
- Each workgroup computes one T×T output tile
- NDRange: `(ceil(N/T), ceil(N/T), n_fragments)` — 3D, fragment_id in z-dimension
- Workgroup size: T×T = 1024 threads (or T×T/4 = 256 with each thread doing 2×2 elements)

**Kernel structure (per workgroup):**
```
__local float tileA[T][T];
__local float tileB[T][T];
float accum[T_PER_THREAD] = {0};

for (int kt = 0; kt < ceil(N/T); kt++) {
    // Load tiles from global memory (coalesced)
    tileA[ty][tx] = A[frag][row][kt*T + tx];
    tileB[ty][tx] = B[frag][kt*T + ty][col];
    barrier();
    
    // Multiply within local memory
    for (int k = 0; k < T; k++)
        accum += tileA[ty][k] * tileB[k][tx];
    barrier();
}
C[frag][row][col] = accum;
```

### Size-Dependent Strategies

| N range | T | Tiles per matmul | Strategy |
|---------|---|-----------------|----------|
| ≤64 | 32 | 4 | Single tile per fragment, one workgroup |
| 65–128 | 32 | 16 | Few tiles, high fragment parallelism |
| 129–512 | 32 | 16–256 | Medium tiling, good occupancy |
| 513–1000 | 32 | 256–1024 | Many tiles, needs global mem coordination |

### Batched Layout

Store all fragment matrices contiguously:
```
A_global = [frag0_A (N0×N0) | frag1_A (N1×N1) | ... | fragK_A (NK×NK)]
```
Each workgroup knows its fragment from `get_global_id(2)`, looks up N and offset from a parameter array.

**For mixed N sizes**: pad to nearest multiple of T (32). Wasted compute is at most ~30%.

## Subproblem 3: Cholesky Factorization of S (One-Time)

S is symmetric positive definite. Tiled Cholesky:

```
for (int i = 0; i < n_tiles; i++) {
    // 1. Diagonal tile: L_ii = chol(A_ii - Σ_{k<i} L_ik L_ik^T)
    //    → Serial (one workgroup), but small (T×T)
    
    // 2. Off-diagonal tiles in column i: L_ij = (A_ij - Σ_{k<i} L_ik L_jk^T) L_ii^{-T}
    //    → Parallel across j (n_tiles - i - 1 workgroups)
    
    // 3. Update trailing matrix: A_jk -= L_ji L_ki^T
    //    → Parallel across (j,k) pairs
}
```

**Dependency chain**: Sequential along diagonal (n_tiles steps). But:
- This is a **one-time cost** (S is constant across SCF)
- Could be done on host and uploaded, or on GPU with multiple kernel launches
- For N=1000, T=32: 32 sequential steps, each with up to 32² = 1024 parallel tile updates

**L^{-1}**: Triangular inverse — simpler than Cholesky, fully parallel per column of tiles.

### Alternative: Skip Cholesky, Use S^{-1/2} via Eigendecomposition on Host

For initial implementation: compute S^{-1/2} on host (LAPACK), upload to GPU. Move to GPU Cholesky later as optimization.

## Subproblem 4: Density Matrix Purification Details

### Initial Guess

**First SCF iteration**: Use Fermi function with bisection for chemical potential:
```
D_0[i][j] = 1 / (1 + exp((H'[i][j] - μ) / kT))
```
Wait — this isn't right. The Fermi function applies to eigenvalues, not matrix elements. The correct approach:

**Palser-Manolopoulos** (canonical):
```
1. Compute μ_min, μ_max from Gershgorin circles
2. Scale: H_s = (H' - μ_min I) / (μ_max - μ_min)  → eigenvalues in [0, 1]
3. D_0 = I - H_s  → eigenvalues in [0, 1], occupied states have D ≈ 1
4. Trace correction: if Tr(D) > N_occ, apply D = D - D²(D - I) etc.
5. Purify: D = 3D² - 2D³ until idempotent
```

**Key**: Need `Tr(D) = 2 * N_occ` (closed shell). After each purification step, check trace and apply correction:
- If Tr(D) > 2*N_occ: D = D - D²(D - I)  (removes electron)
- If Tr(D) < 2*N_occ: D = D + D²(D - I)  (adds electron)

### Subsequent SCF Iterations

Reuse D from previous iteration as starting point. Just re-purify after updating H':
```
D_new = purify(D_old, H'_new)  // warm restart, converges in ~5 iterations
```

### Purification Kernel

Each purification step = 2 matmuls:
```
D² = D @ D          (batched matmul)
D³ = D² @ D         (batched matmul)
D = 3*D² - 2*D³     (element-wise)
trace correction    (reduction)
```

**Total: ~10 purification steps × 2 matmuls = 20 batched matmuls**

## Subproblem 5: Charge Computation

Charges per atom: `q_A = Z_A - Σ_{i∈A} (D·S)_{ii}`

This requires:
1. Compute P = D @ S (one batched matmul — or D @ S_A for atom blocks, cheaper)
2. Extract diagonal elements belonging to each atom
3. Sum per atom

**Optimization**: Don't compute full D@S. Only need diagonal blocks:
- For each atom A with orbital range [i_A, j_A]: `q_A = Z_A - Σ_{i=i_A}^{j_A} (D @ S)[i][i]`
- Compute only diagonal: `(D @ S)[i][i] = Σ_k D[i][k] * S[k][i]`
- This is a matrix-vector-like operation: for each row i, dot product with column i of S
- Can be done with one workgroup per atom, loading D row and S column into local memory

**Cost**: O(N²) per fragment — negligible compared to matmuls.

## Subproblem 6: Shift Update

```
shift_A = U_A * Δq_A + Σ_B γ_AB * Δq_B
```

- U_A: Hubbard parameter (scalar per atom, constant)
- Δq_A: charge difference (scalar per atom, from previous step)
- γ_AB: gamma matrix (constant, precomputed)

This is a **matrix-vector multiply** (γ @ Δq) plus element-wise addition. For N_atoms per fragment:
- O(N_atoms²) — trivially parallel
- One workgroup per fragment, each thread handles one atom
- For N_atoms=38: 38 threads, one workgroup — tiny but fast

## Subproblem 7: Convergence Check & DIIS

### Convergence Check

- Compute `||D_new - D_old||` per fragment (Frobenius norm of difference)
- Reduction across matrix elements, then across fragments
- One small kernel launch, returns bool converged

### DIIS on GPU

DIIS for density matrix:
- Store history: D_0, D_1, ..., D_{k-1} and residuals R_i = D_i - F(D_i)
- Build B matrix: B_ij = <R_i, R_j> (Frobenius inner product = element-wise multiply + reduce)
- Solve small linear system (k×k, k≤10) — trivial, can do on host or with tiny GPU kernel
- Extrapolate: D_new = Σ c_i D_i

**Memory cost**: k × N² × 4B per fragment
- N=100, k=10: 400KB per fragment — fine
- N=1000, k=10: 40MB per fragment — significant, may need to limit DIIS history for large N

**Practical approach**: Apply DIIS only to the charge vector (N_atoms) rather than full D matrix:
- Store charge history (k × N_atoms × 4B) — tiny
- DIIS on charges is common in DFTB+ and works well
- Avoids storing D history entirely

## Overall SCF Cycle Architecture

```
┌─────────────────────────────────────────────────────┐
│ HOST: Upload H0, S, gamma, Hubbard U for all frags  │
│ GPU: Cholesky(S) → L, L^{-1}  (one-time, batched)   │
└──────────────────────┬──────────────────────────────┘
                       │
     ┌─────────────────▼──────────────────┐
     │     SCF ITERATION (on GPU)          │
     │                                     │
     │  1. H_scc = H0 + shifts             │  ← element-wise kernel
     │  2. H' = L^{-1} @ H_scc @ L^{-T}   │  ← 2 batched matmul kernels
     │  3. μ_min, μ_max (Gershgorin)       │  ← reduction kernel
     │  4. D = purify(H', D_prev)          │  ← ~10× (2 matmul + element-wise)
     │  5. q = charges(D, S)               │  ← matmul + reduction
     │  6. shifts = update(q, gamma, U)    │  ← matvec kernel
     │  7. DIIS on charges (optional)      │  ← small linear system
     │  8. converged? ||ΔD|| < tol         │  ← reduction kernel
     └─────────────────┬──────────────────┘
                       │ not converged → loop
                       │ converged ↓
     ┌─────────────────▼──────────────────┐
     │  Read back: charges, energy, D      │
     │  (or keep on GPU for next stage)    │
     └────────────────────────────────────┘
```

**Total kernel launches per SCF iteration**: ~25 (mostly batched matmuls)
**All data stays on GPU** between iterations — no host transfer except final results.

## Performance Estimates

| N | Fragments | Matmuls/iter | FLOPs/iter | Time @ 5 TFLOPS |
|---|-----------|-------------|------------|-----------------|
| 100 | 200 | 22 | 8.8×10¹⁰ | ~18ms |
| 300 | 100 | 22 | 1.2×10¹¹ | ~24ms |
| 500 | 100 | 22 | 5.8×10¹¹ | ~116ms |
| 1000 | 50 | 22 | 2.2×10¹² | ~440ms |

For 20 SCF iterations: N=100 finishes in 0.36s, N=1000 in 8.8s.

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Eigenvalue problem | **Purification** (no diagonalization) | Only needs matmul; GPU-optimal; works for gapped DFTB systems |
| Orthogonalization | **Cholesky-based** (L^{-1}) | Cheaper than S^{-1/2}; one-time cost; tiled GPU algorithm |
| DIIS | **On charge vector** (not D matrix) | Avoids storing D history (N² per step); works well for DFTB |
| Tiling | **T=32, workgroup=256** | Fits local memory; good occupancy; handles N=100–1000 |
| Batching | **3D NDRange (tiles_x, tiles_y, frag_id)** | All fragments processed simultaneously; saturates GPU |
| Mixed N | **Pad to multiple of T=32** | Simple; ≤30% wasted compute; avoids variable-size complexity |
| μ estimation | **Gershgorin + warm restart** | O(N²); no iteration needed; reuses previous SCF step |

## Implementation Priority

1. **Batched tiled matmul kernel** — the foundation; everything depends on it
2. **Purification loop** — uses matmul + element-wise + trace correction
3. **Charge computation** — matmul + reduction
4. **Shift update** — matvec
5. **Cholesky on GPU** — can start with host-computed L^{-1}
6. **DIIS on charges** — small linear system
7. **Convergence check** — reduction

The batched matmul is by far the most critical piece. Get that right and everything else follows.

---

# USER

Are you sure that jacobi rotation is so slow? really N^4 ?

try to look here
/home/prokophapala/git/NumericalMathPlayground/topics/LinearAlgebra/FastDirectSolvers/nested_solver.py

I would prefer to use direct solver rhater than density purification. I mean for SCC we can use disnty matrix purification, but finally I want to have orbitals.

Also for diagonalization of the overlap matrix it would be good to use that Jacobi. I was thinking that thanks to its efficient paralelization it is actually fast on GPU. Cannot we have some tiled version of jacobi which is faster as it use blocks of e.g. 32x32 or 16x16 or 8x8 to fit out workgroupsize ?

---

# GLM 5.2

You're right to push back. Let me re-examine this carefully.

## Correcting the Complexity Analysis

My earlier O(N⁴) claim was **wrong**. Let me recalculate properly:

**Classical cyclic Jacobi:**
- Rotations per sweep: N(N-1)/2 ≈ N²/2
- Work per rotation: O(N) (update 2 rows + 2 columns)
- Work per sweep: O(N² × N) = **O(N³)**
- Sweeps to convergence: **empirically 6–12** (not O(N)), due to quadratic convergence
- Total: **O(N³) with constant ~5–10** — same asymptotic class as QR/LAPACK

**Parallel Jacobi (Brent-Luk ordering):**
- N/2 non-overlapping rotations per parallel step
- Each step: O(N) work, but N/2 rotations in parallel → O(N) parallel time per step  
- Steps per sweep: N-1
- Parallel time per sweep: **O(N²)**
- Total parallel time: **O(N² × sweeps) ≈ O(N² × 10)**

For N=1000: ~10⁷ parallel steps — easily fast on GPU. I was confusing total work with parallel depth.

## Block/Tiled Jacobi — The Key Insight

Your idea of using **block rotations** instead of 2×2 rotations is the crucial optimization. Instead of zeroing a single off-diagonal element A[p,q], we zero an entire T×T off-diagonal block A_{pq}.

### How Block Jacobi Works

Partition the N×N matrix into (N/T)×(N/T) grid of T×T blocks. A single **block rotation** on block-pair (p,q):

```
1. Extract the 2T×2T compound block:
   M = [A_pp  A_pq]     (2T × 2T symmetric)
       [A_qp  A_qq]

2. Diagonalize M → M = V Λ V^T
   (V is 2T×2T, Λ is diagonal)
   This is a SMALL problem — T=32 → 64×64

3. Apply the rotation to ALL block-columns k:
   [A_pk']     [V^T  0 ] [A_pk]
   [A_qk']  =  [0   V^T] [A_qk]
   
   This is a batched matmul: for each k, multiply 2T×T block by 2T×2T rotation
   (N/T - 1) independent block-columns → fully parallel

4. Also update eigenvector accumulator:
   [V_pk']     [V^T  0 ] [V_pk]
   [V_qk']  =  [0   V^T] [V_qk]
```

### Why This Is Better Than 2×2 Jacobi on GPU

| Aspect | 2×2 Jacobi | Block Jacobi (T=32) |
|--------|-----------|---------------------|
| Rotation kernel | 2×2 eigenproblem (trivial) | 64×64 eigenproblem (fits in local memory) |
| Update per rotation | 2N scalars | 2N×T scalars (matmul, high intensity) |
| Rotations per sweep | N²/2 | (N/T)²/2 |
| Parallel rotations per step | N/2 | N/(2T) |
| Work per parallel step | O(N) | O(N×T) |
| Parallel depth per sweep | O(N) | O(N/T) |
| **Total parallel time** | O(N² × sweeps) | O(N²/T × sweeps) |

The block version has **T× less parallel depth** because each rotation does T²× more work but with T× more parallelism per rotation (the matmul tiles). The arithmetic intensity jumps from ~1 FLOP/byte (scalar updates) to ~T FLOP/byte (tiled matmul), which is exactly what GPU needs.

### Concrete Numbers for N=1000, T=32

- Blocks per dimension: 1000/32 ≈ 32
- Block pairs per sweep: 32×31/2 = 496
- Parallel block rotations per step: 16
- Steps per sweep: 31
- Each block rotation: 64×64 diagonalization + 31 × (64×32 matmul updates)
- Sweeps: ~8–12
- **Total parallel steps**: ~31 × 10 = 310 kernel launches
- Each launch: 16 parallel block rotations × (many fragments in z-dim)

### The 2T×2T Diagonalization

This is the inner kernel. For T=32, it's a 64×64 symmetric eigenproblem. Options:

**Option 1: Single-workgroup classical Jacobi** (your existing `block_jacobi` kernel)
- 64×64 = 4096 floats for A + 4096 for V = 32KB local memory — fits
- Single thread does cyclic sweeps (64×63/2 = 2016 rotations × ~8 sweeps)
- ~16K rotations × 64 work each ≈ 10⁶ FLOPs — takes ~0.1ms on one workgroup
- Simple, proven (you already have this code)

**Option 2: Parallel Jacobi within the workgroup** (your `block_jacobi_padded` kernel)
- 64 threads, each handles one row of the 64×64 matrix
- 64/2 = 32 parallel rotations per step, 63 steps per sweep
- ~8 sweeps → ~500 barrier synchronizations
- Faster but more complex

**Option 3: Recursive block Jacobi** (T=32 → 4×4 blocks of 8×8)
- Apply the same block Jacobi idea recursively
- 8×8 blocks → 16×16 diagonalization → even smaller inner problem
- Could go down to 2×2 at the leaf level

**Recommendation**: Start with Option 1 (your existing kernel), optimize later.

### The Update Step — Where the GPU Shines

After each 2T×2T diagonalization, we need to update all other block-columns. For block-column k:

```
[A_pk_new]     [V_11  V_12] [A_pk]
[A_qk_new]  =  [V_21  V_22] [A_qk]
```

This is a (2T × 2T) × (2T × T) matrix multiply — **exactly a tiled GEMM**. And there are (N/T - 2) independent block-columns to update, all parallel.

With multiple fragments: each fragment has its own matrix, so the z-dimension of the NDRange adds another layer of parallelism. For 100 fragments of N=1000: 100 × 30 = 3000 independent 64×32 matmuls per block rotation step.

### Mapping to GPU Workgroups

```
NDRange: (N/T, N/T, n_fragments)    — 3D
Workgroup: (T, T) = (32, 32) = 1024 threads

Each workgroup:
  1. Load the 2T×2T compound block into local memory (collaborative)
  2. Thread 0 does the 64×64 Jacobi diagonalization (or use parallel version)
  3. Store V (2T×2T) in local memory
  4. For each block-column k (loop):
     - Load A_pk and A_qk tiles from global memory
     - Multiply by V (in local memory)
     - Write back A_pk' and A_qk'
```

Wait — this doesn't quite work because step 2 is serial (one thread) while the rest is parallel. Better decomposition:

**Two-kernel approach:**
1. **Kernel 1 (diagonalize)**: One workgroup per block-pair per fragment. Loads 2T×2T, diagonalizes in local memory, writes V to global memory. Small kernel, fast.
2. **Kernel 2 (apply rotation)**: Tiled batched matmul. For each block-column k, apply the 2T×2T rotation. This is a standard GEMM kernel — the same one we need for everything else.

This separation is cleaner: the GEMM kernel is reused for all matmul operations in the SCF cycle, and the diagonalization kernel is specialized but small.

### For the Overlap Matrix S

S is symmetric positive definite, constant across SCF iterations. We need S = V Λ V^T to compute S^{-1/2} = V Λ^{-1/2} V^T.

- One-time cost per fragment
- Block Jacobi works perfectly here
- Can be done on GPU or host (for initial implementation)
- Result: S^{-1/2} uploaded once, reused every SCF iteration

### Revised Architecture: Direct Diagonalization

```
Precompute (once per fragment):
  S → block Jacobi → eigenvalues Λ_S, eigenvectors V_S
  S^{-1/2} = V_S Λ_S^{-1/2} V_S^T    (2 batched matmuls)

Per SCF iteration:
  H_scc = H0 + shifts                  (element-wise, O(N²))
  H' = S^{-1/2} @ H_scc @ S^{-1/2}    (2 batched matmuls, O(N³))
  H' → block Jacobi → ε, C'            (block Jacobi diagonalization)
  C = S^{-1/2} @ C'                    (1 batched matmul)
  D = 2 * C_occ @ C_occ^T              (1 batched matmul)
  q = charges(D, S)                    (matmul + reduction)
  shifts = update(q, gamma, U)         (matvec)
```

**Total per SCF iteration:**
- 4 batched matmuls (transform + density)
- 1 block Jacobi diagonalization (~10 sweeps × 2 kernels)
- ~25–35 kernel launches total

### Hybrid Approach: Best of Both Worlds

You could also do a **hybrid**: use purification for SCF iterations (fast, only matmuls), then do one final full diagonalization at the end to get orbitals:

```
SCF iterations:
  H' = transform(H_scc)
  D = purify(H')           ← fast, only matmuls, ~20 matmuls
  q = charges(D, S)
  shifts = update(q)

Final step (once, after convergence):
  H' → block Jacobi → ε, C'   ← get orbitals
  C = S^{-1/2} @ C'
```

This gives you ~20 matmuls per SCF iteration (fast) + 1 diagonalization at the end (for orbitals). The diagonalization cost is amortized over all SCF iterations.

### Summary of Corrected Analysis

| Method | Complexity | GPU-friendly? | Gives orbitals? |
|--------|-----------|---------------|-----------------|
| 2×2 Jacobi (parallel) | O(N³) total, O(N²) parallel | Poor (low intensity) | ✅ |
| **Block Jacobi (T=32)** | **O(N³) total, O(N²/T) parallel** | **Excellent (matmul + small diag)** | **✅** |
| Density purification | O(N³) total, O(N³/GPU) parallel | Excellent (pure matmul) | ❌ (only D) |
| QR (LAPACK) | O(N³) | Poor (sequential, branchy) | ✅ |

**Block Jacobi is the right choice.** It:
- Has the same O(N³) total work as everything else
- Parallelizes to O(N²/T) depth — very fast
- Uses matmul as the inner workhorse (high arithmetic intensity)
- Gives you eigenvalues AND eigenvectors
- Your existing `block_jacobi` kernel already handles the 2T×2T subproblem
- The update step is just a tiled GEMM — the same kernel you need anyway

The key realization I missed: **block Jacobi transforms the problem from many tiny serial rotations into few large parallel matmuls**, which is exactly what GPUs excel at. The 2T×2T diagonalization is small enough to fit in local memory and can use your existing single-workgroup Jacobi kernel.

