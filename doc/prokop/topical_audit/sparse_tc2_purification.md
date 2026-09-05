---
type: TopicalAudit
title: Sparse TC2 Density Matrix Purification
tags: [topic, sparse, tc2, purification, density-matrix, gpu, cross-language]
---

# Sparse TC2 Density Matrix Purification

## Summary

Computes the density matrix `K` (single-particle density kernel) from a sparse
Hamiltonian `H` and overlap `S` via TC2 (trace-correcting) purification, avoiding
full diagonalization. The kernel satisfies `K S K = K`, `Tr(KS) = N_occ`. Mulliken
charges are derived from the diagonal blocks of `KS`. Implemented on GPU (OpenCL)
with BSR4 sparse matrices (4 orbitals/atom, s+p basis).

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Rust | `rust_dftb/src/methods/sparse/gpu_sparse.rs` | active | GPU driver: Newton-Schulz `Z≈S⁻¹`, spectral bounds, TC2 loop, `KS`, Mulliken. |
| Rust | `rust_dftb/src/methods/sparse/bsr4.rs` | active | BSR4 sparse matrix layout (atom-block CSR, 4×4 blocks, symmetric storage). |
| OpenCL | `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl` | active | GPU kernels: matmul, purification, trace, residual. |
| Rust | `rust_dftb/src/methods/sparse/davidson.rs` | related | Partial eigensolver — alternative route to frontier orbitals. |
| Fortran (ref) | `src/dftbp/elecsolvers/` | reference | Full diagonalization (ELSI) — parity reference for charges/eigenvalues. |

## Parity Status

Tested on pure-C PAHs (benzene, coronene, circumcoronene) vs Rust dense and DFTB+
Fortran. All sparse runs used `tol=1e-6` for TC2 idempotency.

| System | atoms | TC2 iters | sparse vs dense max\|Δq\| (e) | sparse vs DFTB+ max\|Δq\| (e) |
|--------|-------|-----------|-------------------------------|-------------------------------|
| Benzene | 6 | 46 | 1.5e-5 | 1.5e-5 |
| Coronene | 24 | 36 | 9.0e-6 | 9.0e-6 |
| Circumcoronene | 54 | 51 | 6.5e-5 | 6.5e-5 |

Dense charges match DFTB+ to machine precision (~1e-16 e). Sparse TC2 charges
match to ~1e-5–1e-4 e, consistent with the TC2 tolerance.

## Conventions

- **Sparse `K`**: `Tr(KS) = N_occ` (no spin factor). Mulliken charges from
  diagonal blocks of `KS`.
- **Dense `D`**: `D = 2·C_occ·C_occᵀ`, `Tr(DS) = 2·N_occ`. Comparison uses
  `2K` vs `D`.
- **Rust `SccResult.charges`**: stores Mulliken **populations** (`q_electronic`).
  Actual charge = `q0 - population` (opposite sign to DFTB+ convention
  `deltaQ = q0 - q_elec`).

## Open Issues

- **BSR4 requires 4 orbitals/atom** — rejects H (1s only). Pure-C systems only.
  Variable block size or dense fallback for H needed for realistic systems.
  **Note**: the Chebyshev+Ritz sparse eigensolver (`chebyshev_ritz_eigensolver.md`)
  handles mixed C/H systems via the general `build_non_scc` path with
  per-atom orbital counts from `SystemContext::atom_n_orb`. H-passivated
  ribbons up to N=1156 (388 atoms, 132 H) are validated. The BSR4 TC2 path
  remains pure-C only.
- **TC2 convergence depends on spectral bounds** — the Newton-Schulz `Z≈S⁻¹`
  step must converge first; if it fails, TC2 diverges. Fail-loud checks present.
- **No sparse matvec operator for Davidson** — the partial eigensolver currently
  densifies `H` and `S`. A sparse matvec would enable Davidson on large systems
  without densification. **Note**: the Chebyshev+Ritz solver already uses a
  sparse implicit operator (`CholeskyTransformedOperator`) with sparse SpMV
  and triangular solves — see `chebyshev_ritz_eigensolver.md`.

## Related

- `/doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md` — session report.
- `/doc/prokop/topical_audit/chebyshev_ritz_eigensolver.md` — sparse Chebyshev+Ritz eigensolver (handles mixed C/H, alternative to TC2 for frontier orbitals).
- `/rust_dftb/scripts/test_charges_homo_lumo.rhai` — end-to-end test.
- `/debug/graphene_sparse/` — plots, TSVs, convergence history.
