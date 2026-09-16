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
  **Update (Gate D):** the sparse nanocrystal vibrations task
  (`sparse_nanocrystal_vibrations.md`) solved this for Si/H systems by padding
  H atoms with 3 dummy orbitals (S_dd=1, H_dd=E_dummy), making the padded
  overlap nonsingular while preserving the physical electron count. See
  `tests/sih_padded_basis.rs`.
- **TC2 convergence depends on spectral bounds** — the Newton-Schulz `Z≈S⁻¹`
  step must converge first; if it fails, TC2 diverges. Fail-loud checks present.
- **No sparse matvec operator for Davidson** — the partial eigensolver currently
  densifies `H` and `S`. A sparse matvec would enable Davidson on large systems
  without densification. **Note**: the Chebyshev+Ritz solver already uses a
  sparse implicit operator (`CholeskyTransformedOperator`) with sparse SpMV
  and triangular solves — see `chebyshev_ritz_eigensolver.md`.

## Update (2026-09-16) — two-phase purify + DMM warm update

- **Two-phase production policy** (`sparse_system.rs::tc2_purify`, report
  §15.15): Phase A = bounded f32 TC2 (budget = caller's `tc2_max`, production
  default 30) with a floor-stop on `trace_locked && best_ri < ff_switch &&
  5-check stall`; Phase B = optional terminal **FF32 McWeeny polish**
  (`tc2_hiacc` / `RUST_DFTB_TC2_FF`, ≤5 steps, `PolishedFF` status). FF
  kernels: `FF_LO_GLOBAL` + `FF_ACC2` default-on → 27.4 ms/step ≈ 4.0× an
  f32 iter at deg330. rk40: floor 2.7e-7 → R_I64 2.8e-8; rk20 stays
  mask-limited ~3e-5 — by design.
- **DMM warm update** (`dmm_descend`, `RUST_DFTB_VIB_DMUPD`): for small
  geometry displacements (FD Hessians), purification from a warm seed is
  *repelling* (masked fixed point) — the correct warm update minimizes
  ‖[K,H]‖: `δK=−η(X+Xᵀ−2Y)`, `X=(Z·H)·K`, `Y=(K·S)·X`, Z=S⁻¹, plus
  McWeeny retraction. Validated 0.30% ΔF vs cold at ~25% lower cost.
  Report: `reports/2026-09-16_sparse_dmm_warm_density_hessian.md`.
- **Note:** workspace `Z` is **S⁻¹** (Newton `Z←2Z−ZSZ`), not S⁻¹ᐟ² — a
  derivation assuming the root produces an ascent direction (measured).

## Related

- `/doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md` — session report.
- `/doc/prokop/topical_audit/chebyshev_ritz_eigensolver.md` — sparse Chebyshev+Ritz eigensolver (handles mixed C/H, alternative to TC2 for frontier orbitals).
- `/doc/prokop/topical_audit/sparse_nanocrystal_vibrations.md` — Si/H nanocrystal vibrations: padded BSR4 basis, analytic forces, SpGEMM plans, Hessian plateau.
- `/rust_dftb/scripts/test_charges_homo_lumo.rhai` — end-to-end test.
- `/debug/graphene_sparse/` — plots, TSVs, convergence history.
