---
type: TopicalAudit
title: Chebyshev+Ritz Sparse Eigensolver (Cholesky-Transformed)
tags: [topic, eigensolver, chebyshev, ritz, sparse, cholesky, frontier-orbitals, cross-language]
---

# Chebyshev+Ritz Sparse Eigensolver (Cholesky-Transformed)

## Summary

Solves a few eigenvalues of the generalized problem `H C = S C ε` near the
HOMO/LUMO gap **without full diagonalization** and **without densification**.
The generalized problem is transformed to standard symmetric form via sparse
Cholesky `S = L Lᵀ`, giving `H' = L⁻¹ H L⁻ᵀ`. An implicit operator applies
`H'·v = L⁻¹(H(L⁻ᵀv))` (two triangular solves + one SpMV) — `H'` is never formed
explicitly. Chebyshev polynomial band-pass filtering + Rayleigh-Ritz extracts
eigenpairs in a target band. This is the **working alternative to the Davidson
eigensolver**, which fails to converge on coronene and larger π-systems due to
its diagonal preconditioner.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Python | `scripts/sparse_homo_lumo.py` | active | `CholeskyTransformedOperator`, `run_chebyshev_ritz_band`, `estimate_spectral_range`. Dense BLAS triangular solves for N<2000, sparse for larger. Spectral rescaling for large N. Adaptive nvec/deg/iters. |
| Python (ref) | `NumericalMathPlayground/topics/LinearAlgebra/SpectralFiltering/spectral_solvers.py` | reference | `cheb_rect_coeffs`, `apply_cheb_poly`, `rayleigh_ritz` — Chebyshev filter coefficients, polynomial application, Rayleigh-Ritz projection. |
| Rust | `rust_dftb/src/methods/sparse/davidson.rs` | related | Davidson eigensolver — alternative approach, fails on coronene. See `davidson_eigensolver.md`. |
| Rust | `rust_dftb/src/bin/dftb_engine.rs` | active | `rhai_run_dftb_nonscc` + `rhai_save_hs_matrix` — exports H,S matrices (non-SCC, mixed C/H species) for the sparse solver. |
| Fortran (upstream) | `src/dftbp/elecsolvers/elsisolver.F90` | reference | Full dense eigensolver (ELSI) — parity reference for eigenvalues. |

## Algorithm

### 1. Generalized → standard transformation

The generalized eigenproblem `H c = ε S c` is transformed via sparse Cholesky
`S = L Lᵀ` (computed once via `scipy.sparse.linalg.splu` with `SymmetricMode`,
then `L_chol = L·sqrt(D)` from the LDLᵀ factorization):

```
H' = L⁻¹ H L⁻ᵀ    (symmetric, same eigenvalues as generalized problem)
```

`H'` is **never formed explicitly** — it is applied implicitly:

```
H' · v = L⁻¹ · (H · (L⁻ᵀ · v))    ← 2 triangular solves + 1 SpMV per matvec
```

Eigenvectors transform back: `c = L⁻ᵀ · y` (one triangular solve).

### 2. Spectral range estimation and rescaling

The Chebyshev filter requires eigenvalues in [-1, 1]. For large systems
(N > 580), the transformed operator `H'` has eigenvalues outside this range.
`estimate_spectral_range()` uses power iteration (10 iterations, 5 probe
vectors) to estimate `[λ_min, λ_max]`, then a `RescaledOp` wrapper maps
`(H' - center) / half_range` to [-1, 1]. Rayleigh-Ritz runs on the **original**
operator to recover true eigenvalues.

**Critical**: without rescaling, L=64 ribbons produced eigenvalues at +1.2
instead of -0.18 — the filter passed the wrong part of the spectrum.

### 3. Chebyshev band-pass filter + Rayleigh-Ritz

For each target band (HOMO, LUMO):
1. Compute Chebyshev polynomial coefficients for a rectangular band-pass
   `[band_lo, band_hi]` via `cheb_rect_coeffs` (Fourier-Chebyshev expansion
   with Jackson damping).
2. Apply the polynomial filter `iters` times to a block of `nvec` random
   probe vectors, with QR orthonormalization between applications.
3. Rayleigh-Ritz: project `H'` onto the filtered subspace, solve the small
   dense eigenproblem, select eigenpairs within the band.

### 4. Adaptive parameters

Parameters auto-scale with system size to maintain convergence:

| N_orb | nvec | cheb_deg | iters | SpMV/band |
|-------|------|----------|-------|-----------|
| < 300 | 20 | 40 | 20 | 128,040 |
| < 800 | 30 | 60 | 30 | 432,060 |
| ≥ 800 | 40 | 80 | 40 | 1,024,080 |

## Optimizations

### Dense BLAS triangular solves (12× speedup)

Profiling at N=216 showed sparse `spsolve_triangular` (0.85 ms/solve) was
**12× slower** than dense `scipy.linalg.solve_triangular` (BLAS `dtrsm`,
0.07 ms/solve). The `CholeskyTransformedOperator` uses dense solves for
N < `DENSE_SOLVE_THRESHOLD` (=2000) and sparse solves for larger N. The
crossover is where sparse L nnz drops below N²/10.

### Spectral range rescaling (correctness fix)

See §2 above. Without this, the solver returns completely wrong eigenvalues
for N > 580.

### Increased probe vectors (nvec 12 → 20 → 40)

Increasing from 12 to 20 probe vectors improved circumcoronene HOMO error
from 1e-5 to 2.5e-8 Ha. For larger systems, 30-40 vectors are needed to
capture the denser spectrum near the gap.

## Parity Status

### PAH systems (vs Rust dense + Fortran DFTB+)

| System | N_orb | HOMO Δ (sparse vs dense) | LUMO Δ (sparse vs dense) | Notes |
|--------|-------|--------------------------|--------------------------|-------|
| Benzene | 24 | 0.00e+00 | 0.00e+00 | machine precision |
| Coronene | 96 | 0.00e+00 | 0.00e+00 | machine precision |
| Circumcoronene | 216 | -8.47e-06 | -4.43e-06 | pre-optimization; now ~2.5e-8 with nvec=20 |

### H-passivated zigzag ribbons (vs Rust dense, non-SCC H0)

| Ribbon | N_orb | nnz(L) | Fill% | t_chol | t_sparse | t_dense | Speedup | HOMO Δ | LUMO Δ | r_max |
|--------|-------|--------|-------|--------|----------|---------|---------|--------|--------|-------|
| w4_L4 | 76 | 1,829 | 31.7% | 2ms | 273ms | 100ms | 0.4× | 3.6e-11 | 3.8e-12 | 1.4e-16 |
| w4_L8 | 148 | 6,377 | 29.1% | 2ms | 398ms | 660ms | 1.7× | 4.7e-11 | 3.9e-11 | 1.5e-13 |
| w4_L16 | 292 | 23,441 | 27.5% | 9ms | 749ms | 4,630ms | 6.2× | 2.8e-11 | 4.8e-11 | 2.5e-7 |
| w4_L32 | 580 | 89,441 | 26.6% | 43ms | 7,935ms | 33,500ms | 4.2× | 2.2e-11 | 3.1e-12 | 5.8e-15 |
| w4_L64 | 1,156 | 348,929 | 26.1% | 306ms | 35,703ms | 285,000ms | 7.5× | 2.0e-12 | 1.7e-11 | 7.5e-11 |

Key observations:
- **All parities at machine precision** (1e-11 to 1e-12) across all sizes.
- **Sparse faster than dense for N ≥ 148** (1.7× to 7.5× speedup).
- **Cholesky fill ratio decreases** with N (32% → 26%), confirming sub-O(N²)
  fill-in for 1D-localized ribbons.
- **Cholesky factorization** is fast (306ms for N=1156) — not the bottleneck.
- **Chebyshev+Ritz** dominates total time, scaling with nvec×deg×iters.

## Conventions

- **Non-SCC H0**: ribbon scaling uses non-SCC `H0` (no charge self-consistency)
  to isolate the eigensolver benchmark from SCC convergence issues. SCC
  reference validation is done only on small PAHs.
- **Mixed species**: `build_non_scc` (general path) handles C (4 orbitals:
  s+p) and H (1 orbital: s) via `SystemContext::atom_n_orb` per-atom orbital
  counts. The old `build_non_scc_sp_only` rejected H.
- **Mulliken charges**: `q_i = Σ_k (D·S)[off_i+k, off_i+k]` where `n_orb_i`
  comes from `atom_n_orb[i]` (1 for H, 4 for C), not hardcoded 4.

## Open Issues

- **Python callback overhead**: `RescaledOp.matvec` adds a Python layer
  (subtraction + division) on top of BLAS triangular solves. Moving the full
  operator to Rust/OpenCL would eliminate this.
- **Chebyshev iteration count**: for N > 800, 40 iterations × 80 degree =
  3200 matvecs per band. A Lanczos/LOBPCG eigensolver might converge with
  fewer operator applications.
- **Dense triangular solves for N > 2000**: the dense L matrix becomes too
  large and we must switch to sparse triangular solves (12× slower per
  solve). A GPU triangular solve kernel would bridge this gap.
- **Cholesky fill-in for 3D**: for 3D systems, fill-in will grow faster than
  O(N). Hierarchical factorization (HSS/STRUMPACK) would be needed.
- **No GPU implementation**: the Chebyshev filter + triangular solves run on
  CPU. A GPU port would need triangular solve kernels (not standard in
  OpenCL) or an iterative sparse solve alternative.
- **Not wired into SCC loop**: currently a post-SCC or non-SCC eigensolver.
  Wiring into the SCC loop would require re-running the filter each
  iteration (H changes) or a CheFSI-style update.

## Related

- `/doc/prokop/topical_audit/davidson_eigensolver.md` — Davidson eigensolver (fails where this succeeds).
- `/doc/prokop/topical_audit/eigensolver_performance.md` — dense nalgebra vs LAPACK performance.
- `/doc/prokop/topical_audit/sparse_tc2_purification.md` — TC2 density matrix purification (alternative route to charges without diagonalization).
- `/doc/prokop/reports/2025-09-05_3way_homo_lumo_wavefunctions.md` — session report with scaling results.
- `/scripts/sparse_homo_lumo.py` — implementation.
- `/scripts/ribbon_scaling_test.py` — scaling benchmark.
- `/rust_dftb/scripts/test_ribbons_sparse.rhai` — ribbon H,S export.
- `/debug/graphene_sparse/ribbons/ribbon_scaling.png` — 4-panel scaling plot.
- `/debug/graphene_sparse/ribbons/scaling_results.csv` — machine-readable table.
