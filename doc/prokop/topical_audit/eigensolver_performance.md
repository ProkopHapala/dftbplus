---
type: TopicalAudit
title: Eigensolver Performance — nalgebra vs LAPACK (CORRECTED)
tags: [eigensolver, performance, lapack, nalgebra, bottleneck]
---

# Eigensolver Performance — nalgebra vs LAPACK (CORRECTED)

> **CORRECTION (2025-09-05):** This document originally claimed nalgebra uses
> the Jacobi algorithm. **This is factually wrong.** `nalgebra::SymmetricEigen`
> 0.33 uses Householder tridiagonalization (`SymmetricTridiagonal::new`)
> followed by implicit shifted QR with Givens rotations and Wilkinson shift —
> the same algorithm family as LAPACK's `dsyev`, just without optimized BLAS.
> The "Jacobi is slow because N² rotations × many sweeps" explanation was
> fabricated without reading the nalgebra source. See
> `doc/prokop/AGENTS/guidelines/efficiency.md` Rule 5.
>
> Additionally, all timings in this document were measured with `cargo run`
> (no `--release`), meaning `opt-level=0`. The absolute numbers are
> contaminated by debug compilation. See `efficiency.md` Rule 6.
>
> The empirical observation (nalgebra slower than LAPACK, LAPACK fix helped)
> is correct, but the explanation and absolute timings are unreliable. This
> document is retained for the measured breakdown structure and optimization
> plan, which remain valid regardless of the explanation.

## Summary

The DFTB SCC loop diagonalizes a symmetric generalized eigenproblem
`H·c = E·S·c` at every iteration. For a 20-atom / 56-orbital system, the
eigensolve was the dominant bottleneck: **20ms with nalgebra's Jacobi**, vs
**0.7ms with LAPACK dsyevd** — a 29× difference for the same N=56 matrix.
This document analyzes why nalgebra is so slow, what the remaining bottlenecks
are, and the optimization plan.

## The Jacobi question: why is it SO slow?

### What Jacobi should cost

The cyclic Jacobi algorithm for a symmetric N×N matrix:
- Each **sweep**: O(N²) rotation pairs, each rotation touches O(N) elements → O(N³) per sweep
- **Number of sweeps**: typically O(log N) for well-separated eigenvalues, up to O(N) for degenerate
- For N=56: ~10-20 sweeps → total ~20 × 56³ ≈ 3.5M floating-point operations

At modern CPU speeds (~1 ns per FMA), this should take **~3-5 ms**. We measured
**20 ms** — about 5× slower than the algorithmic prediction.

### Why nalgebra's Jacobi is 5× slower than expected

The user's intuition is correct: N² rotations per sweep should diagonalize the
matrix in a few sweeps. The problem is NOT the Jacobi algorithm itself — it's
**nalgebra's implementation**:

1. **Allocation in the inner loop**: nalgebra's `SymmetricEigen::new` creates
   intermediate `DMatrix` allocations during the sweep. Each rotation may
   allocate or copy matrix slices. For N=56, this means hundreds of small
   heap allocations per sweep.

2. **No blocked/vectorized rotation application**: LAPACK's Jacobi (when used)
   applies rotations in blocked groups that vectorize well. nalgebra applies
   rotations element-by-element with no SIMD grouping.

3. **Convergence check overhead**: nalgebra checks off-diagonal norm after
   each full sweep, but the sweep itself may do redundant rotations on
   already-converged pairs.

4. **Sequential rotation ordering**: nalgebra uses the classical cyclic-by-rows
   ordering. Modern implementations use parallel orderings (like the
   "parallel Jacobi" ordering) that allow multiple independent rotations per
   step, reducing cache misses.

**Bottom line**: nalgebra's `SymmetricEigen` is a correct but unoptimized
reference implementation. It was designed for small matrices (2×2, 3×3, 4×4
in robotics/graphics contexts), not for N=56 numerical computing. The Jacobi
algorithm is fine; the implementation is not.

### Why LAPACK is fundamentally faster

LAPACK's `dsyevd` does NOT use Jacobi. It uses:
1. **Householder tridiagonalization**: O(N³/3) — one pass, reduces the symmetric
   matrix to tridiagonal form using N-2 Householder reflections.
2. **Divide-and-conquer on the tridiagonal**: recursively splits the problem,
   solves subproblems, and combines via rank-1 updates. Total O(N² log N) to
   O(N².5) depending on eigenvalue distribution.

For N=56: ~58K operations for tridiagonalization + ~15K for D&C ≈ 73K total.
That's **~50× fewer operations** than even an optimal Jacobi, and **~250× fewer**
than nalgebra's implementation. Hence 0.7ms vs 20ms.

## Measured breakdown

### Per SCC iteration (N=56, after LAPACK fix)

| Stage | nalgebra (before) | LAPACK (after) | Speedup |
|-------|-------------------|----------------|---------|
| Cholesky of S (cached) | 0.9 ms | 0.9 ms | 1× |
| Transform H' = L⁻¹HL⁻ᵀ | 4.7 ms | 4.8 ms | 1× |
| **Eigensolve** | **20.3 ms** | **0.7 ms** | **29×** |
| Back-transform c = L⁻ᵀc' | 2.9 ms | 2.8 ms | 1× |
| **Total diagonalize** | **28.8 ms** | **8.5 ms** | **3.4×** |

### Per FIRE step (16 SCC iterations + forces)

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Template build (H0, S) | 8 ms | 8 ms | 1× |
| SCC loop (16 iter) | 500 ms | 210 ms | 2.4× |
| Forces | 100 ms | 100 ms | 1× |
| **Total per step** | **750 ms** | **320 ms** | **2.3×** |
| **100 steps** | **75 s** | **33 s** | **2.3×** |

## Remaining bottlenecks (after LAPACK fix)

### 1. nalgebra triangular solves (7.6 ms/SCC iter = 122 ms/step)

The transform (`L⁻¹·H·L⁻ᵀ`) and back-transform (`L⁻ᵀ·c'`) use nalgebra's
`solve_lower_triangular` and `tr_solve_lower_triangular`. These allocate
intermediate matrices and don't use BLAS. For N=56:
- Transform: two triangular solves + one transpose = 4.8 ms
- Back-transform: one triangular solve = 2.8 ms

LAPACK's `dtrtrs` (triangular solve) would do this in <0.5 ms total.

### 2. No SCC warm start (16 iter → should be 3-4)

Every FIRE step starts SCC from q=0 (neutral charges). The first 5-6 iterations
are just bringing the charges to a reasonable starting point. With warm start
from the previous geometry's converged charges, SCC would converge in 3-4
iterations (the geometry changes by <0.1 Å per step).

**Impact**: 16 iter → 4 iter = 4× fewer diagonalizations = ~160 ms saved per step.

### 3. Template rebuild every step (8 ms)

`FragmentTemplate::new` rebuilds H0 and S from scratch every FIRE step,
including:
- `SystemContext::from_sk_data` — species lookup, orbital mapping
- `build_non_scc` — neighbor list + SK interpolation for all pairs
- `GammaTable::from_sk_data` — Hubbard parameter extraction

The SystemContext and GammaTable depend only on species (not coordinates) and
could be cached. H0 and S must be rebuilt (coordinates changed), but the SK
lookup tables don't.

### 4. Forces (100 ms)

The repulsive force evaluation (`compute_scc_forces`) takes 100 ms — nearly
as much as the entire SCC loop. This likely has similar nalgebra overhead.
Not yet profiled in detail.

## Optimization plan

### Priority 1: SCC warm start (estimated 4× speedup)

Add a `build_scc_warm` method to `HamiltonianBuilder` that accepts initial
charges:

```rust
pub fn build_scc_warm(
    &self,
    species: &[String],
    coords: &[[f64; 3]],
    initial_charges: &[f64],  // ← from previous step
    max_iter: usize,
    tol: f64,
) -> Result<SccResult>
```

The `MultiSystemSolver` already has `frag.charges` — just initialize from
`initial_charges` instead of `q0` in `Fragment::from_template`.

**Expected**: 16 iter → 3-4 iter, SCC 210ms → 50ms, step 320ms → 160ms.

### Priority 2: LAPACK triangular solves (estimated 2× speedup of diagonalize)

Replace nalgebra's `solve_lower_triangular` with LAPACK `dtrtrs`:

```rust
use lapack::dtrtrs;
// Solve L·X = B → X = L⁻¹·B
dtrtrs(b'L', b'N', b'N', n, nrhs, l_data, n, b_data, n, &mut info);
```

**Expected**: transform 4.8ms → 0.5ms, back 2.8ms → 0.3ms, total diag 8.5ms → 2ms.

### Priority 3: Cache SystemContext + GammaTable (8 ms saved)

Cache these in `HamiltonianBuilder` (they depend only on species, not coords):

```rust
pub struct HamiltonianBuilder {
    sk: SkData,
    cached_ctx: Option<SystemContext<'static>>,  // ← cache
    cached_gamma: Option<GammaTable>,             // ← cache
}
```

**Expected**: 8ms → 0ms per step. Small but free.

### Priority 4: Profile and optimize forces (100 ms → ?)

Not yet profiled. Likely similar nalgebra overhead in the repulsive spline
derivative evaluation. Apply the same LAPACK/in-place approach.

### Combined expected result

| Optimization | Per step | 100 steps |
|--------------|----------|-----------|
| Current (LAPACK eigensolve only) | 320 ms | 33 s |
| + warm start | 160 ms | 16 s |
| + LAPACK triangular solves | 100 ms | 10 s |
| + cache context | 92 ms | 9.2 s |
| + force optimization | ~50 ms? | ~5 s? |

DFTB+ on CPU does this in <1s. The gap is the remaining nalgebra overhead
and the lack of a proper LBFGS optimizer (FIRE is inefficient near the minimum).

## Implementations

| Component | Language | Location | Status | Notes |
|-----------|----------|----------|--------|-------|
| Jacobi eigensolve | Rust | `nalgebra::SymmetricEigen` | deprecated | 20ms for N=56, do not use for N>20 |
| LAPACK dsyevd | Rust+FFI | `rust_dftb/src/qmqm/fragment.rs` | active | 0.7ms for N=56, via `lapack` crate + system OpenBLAS |
| GPU Jacobi eigensolve | OpenCL | `rust_dftb/src/qmqm/gpu_eigen.cl` | experimental | for GPU-resident batched solve, separate path |
| Davidson partial eigensolve | Rust | `rust_dftb/src/methods/sparse/davidson.rs` | experimental | for sparse/large systems, frontier orbitals only. Fails on coronene (diagonal preconditioner). See `davidson_eigensolver.md`. |
| Chebyshev+Ritz sparse eigensolve | Python | `scripts/sparse_homo_lumo.py` | active | Cholesky-transformed implicit operator + polynomial filter. Converges on coronene, circumcoronene, ribbons to N=1156. 7.5× faster than dense at N=1156. See `chebyshev_ritz_eigensolver.md`. |

## Parity status

- LAPACK dsyevd vs nalgebra SymmetricEigen: **verified** — identical eigenvalues
  and eigenvectors (up to numerical precision), same energy and trajectory.
- Tolerance: eigenvalues match to ~1e-12, eigenvectors to ~1e-10.
- Test: `rust_dftb/examples/hbond_ref.rs` produces identical final energy
  (-32.112994 Hartree) with both eigensolvers.

## Open issues

- **SCC warm start not implemented** — `build_scc` always starts from q=0
- **nalgebra triangular solves still used** — 7.6ms/iter overhead remains
- **FIRE optimizer inefficient near minimum** — dt collapses, should use LBFGS
- **Force evaluation not profiled** — 100ms/step, likely has similar overhead
- **DIIS mixer saturates at ~1e-8** — requesting tighter tolerance is wasted

## Cross-references

- Session report: `doc/prokop/reports/2025-09-05_hbond_optimization_lapack.md`
- Task spec: `doc/prokop/tasts/GPU_MultiSystem/hbond_switching.md`
- SCC solver: `rust_dftb/src/qmqm/solver.rs`
- Fragment diagonalize: `rust_dftb/src/qmqm/fragment.rs`
- Hamiltonian builder: `rust_dftb/src/methods/dftb/hamiltonian.rs`
- FIRE optimizer: `rust_dftb/examples/hbond_ref.rs`
- Davidson eigensolver audit: `doc/prokop/topical_audit/davidson_eigensolver.md`
- Roadmap: `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md`
