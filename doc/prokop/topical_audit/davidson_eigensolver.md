---
type: TopicalAudit
title: Davidson / Partial Generalized Eigensolver
tags: [topic, eigensolver, davidson, frontier-orbitals, cross-language]
---

# Davidson / Partial Generalized Eigensolver

## Summary

Solves a few eigenvalues of the generalized problem `H C = S C ε` around the
SCC occupation gap (HOMO/LUMO region) without full diagonalization. Used to
extract frontier orbitals after SCC convergence, and eventually to avoid
densification on large sparse systems.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Rust | `rust_dftb/src/methods/sparse/davidson.rs` | active | Generalized Davidson: S-orthonormalization, Rayleigh-Ritz, diagonal preconditioner with regularization, subspace restart. Unit test vs dense. |
| Python (ref) | `NumericalMathPlayground/topics/LinearAlgebra/FastDirectSolvers/Davidson_Eigensolver.py` | reference | Standard Davidson example — subspace iteration, projected diagonalization, residual expansion. |
| Python (ref) | `NumericalMathPlayground/topics/LinearAlgebra/FastDirectSolvers/test_gpu_ritz.py` | reference | GPU Ritz example — projected similarity transforms, residual norms, subspace overlap robust to sign/degeneracy. |
| Fortran (upstream) | `src/dftbp/elecsolvers/elsisolver.F90` | reference | Full electronic solver (ELSI interface) — not a partial solver, but the parity reference for eigenvalues. |

## Parity Status

- **Rust Davidson vs Rust dense** (`test_davidson_vs_dense_small`): HOMO/LUMO/gap
  match to 1e-10 Ha on a small test matrix. Test passes.
- **Rust Davidson on benzene** (24 orbitals): converges in 3 iterations, matches
  dense HOMO=-0.245236, LUMO=-0.243239, gap=0.001998 Ha.
- **Rust Davidson on coronene** (96 orbitals): **does not converge** — the dense
  manifold of near-degenerate π states near the gap makes the diagonal
  preconditioner ineffective. Open issue.

## Open Issues

- **Diagonal preconditioner insufficient for coronene/circumcoronene.** The
  denominator `H_ii - θ·S_ii` vanishes for near-degenerate states, causing the
  correction vectors to constantly introduce new directions that don't isolate
  the LUMO. Regularization (`|denom| < eps → sign·eps`) prevents blow-up but
  doesn't fix convergence. Proposed fixes:
  - SSOR / ILU preconditioner (captures off-diagonal coupling).
  - Shift-invert Davidson (robust for interior eigenvalues, but needs `(H-σS)⁻¹`).
  - Chebyshev-filtered subspace iteration (CheFSI) — see
    `NumericalMathPlayground/topics/LinearAlgebra/LinearScalingQM/CheFSI/`.
  - Lanczos with spectral transformation.
- **Not wired to sparse BSR4 operator.** Currently uses dense `H_scc` and `S`
  from `SccResult`. For large systems, a sparse matvec operator is needed to
  avoid densification.
- **No GPU implementation yet.** The `qmqm/gpu_eigen.cl` path is a full
  eigensolver, not a partial one.

## Related

- `/doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md` — session report.
- `/rust_dftb/src/methods/sparse/davidson.rs` — implementation.
- `dftb_engine.rs` Rhai function `davidson_homo_lumo(name, n_target)`.
