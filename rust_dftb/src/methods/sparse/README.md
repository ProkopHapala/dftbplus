# rust_dftb/src/methods/sparse/

BSR4 sparse matrices, GPU sparse density-matrix purification (TC2), and a partial
generalized eigensolver (Davidson) for frontier orbitals. All GPU kernels are
single-precision f32; CPU reference is f64.

- **bsr4.rs** — BSR4 sparse matrix layout: atom-block CSR with 4×4 blocks
  (s+p basis, 4 orbitals/atom). Symmetric matrices store both (i,j) and (j,i)
  blocks. Requires all atoms to have 4 orbitals — rejects H.
- **gpu_sparse.rs** — GPU driver for sparse purification: Newton-Schulz
  `Z≈S⁻¹`, spectral bound estimation, TC2 purification loop, `KS` product,
  Mulliken charges from `KS` diagonal blocks. Returns `SparseResult` with
  idempotency residual `R_I`, Hamiltonian residual `R_H`, `Tr(KS)`, Mulliken
  charges, and TC2 convergence history.
- **davidson.rs** — generalized Davidson eigensolver for `H C = S C ε`.
  S-orthonormalization (modified Gram-Schmidt under the S metric), Rayleigh-Ritz
  projection, diagonal preconditioner with regularization for near-degenerate
  states, subspace restart. Selects `n_target` occupied + `n_target` virtual
  eigenvalues around the Fermi gap. Unit test vs dense `SymmetricEigen`.
  **Limitation:** the diagonal preconditioner is insufficient for systems with
  dense near-degenerate frontier manifolds (e.g. coronene) — see
  `doc/prokop/topical_audit/davidson_eigensolver.md`.
- **sparse_bsr4_purification.cl** — OpenCL kernels: BSR4 matmul, TC2 purification
  steps, trace, residual norms. Gather-only operations, workgroup size ~32.
- **mod.rs** — module exports.
