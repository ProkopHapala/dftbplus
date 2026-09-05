# 2025-09-05 — 3-way HOMO/LUMO wavefunction comparison (dense vs sparse Chebyshev+Ritz vs Fortran)

## Summary

Extended the wavefunction projection workflow to a full **3-way comparison** of
HOMO/LUMO eigenvalues and real-space wavefunctions across:

1. **Rust dense** — full generalized diagonalization of `H·c = ε·S·c` (LAPACK `dsyevd`).
2. **Rust sparse** — Chebyshev polynomial filter + Rayleigh-Ritz iterative
   eigensolver, finding only a few eigenpairs near the gap without full
   diagonalization. Reuses the spectral filtering code from
   `NumericalMathPlayground/topics/LinearAlgebra/SpectralFiltering/spectral_solvers.py`.
3. **DFTB+ Fortran** — upstream reference (`dftb_in.hsd` with
   `Analysis { WriteEigenvectors = Yes }`, parsing `eigenvec.bin`).

All three paths feed into the same OpenCL `GridProjector` + STO basis pipeline
to produce side-by-side signed wavefunction contour plots on the molecular
xy-plane.

## What was done

### Rust side

- **`SccResult.eigenvectors`** — added the full `(norb × norb)` MO coefficient
  matrix to `SccResult` (`rust_dftb/src/methods/dftb/hamiltonian.rs`), populated
  from `Fragment.eigenvectors` in both the SCC and non-SCC paths.
- **`rhai_save_eigenvectors(name, path)`** — exports geometry + eigenvector
  matrix + eigenvalues to a TSV (`rust_dftb/src/bin/dftb_engine.rs`).
- **`rhai_save_hs_matrix(name, path)`** — exports `H_scc` and `S` matrices +
  geometry + dense reference eigenvalues to a TSV, for consumption by the
  sparse iterative eigensolver.
- Both functions registered in the Rhai engine and called in
  `rust_dftb/scripts/test_charges_homo_lumo.rhai` for benzene, coronene,
  circumcoronene.

### Python side

- **`scripts/sparse_homo_lumo.py`** — reads `H, S` from the TSV, transforms the
  generalized eigenproblem to standard form, runs two separate Chebyshev
  filter + Rayleigh-Ritz passes (one band around HOMO, one around LUMO), picks
  the closest eigenvalue to the dense reference, transforms eigenvectors back,
  and saves a TSV with HOMO/LUMO eigenvectors.
  - Uses `cheb_rect_coeffs`, `apply_cheb_poly`, `rayleigh_ritz` from
    `NumericalMathPlayground/topics/LinearAlgebra/SpectralFiltering/spectral_solvers.py`.
  - Two-band approach is necessary because coronene's gap (0.021 Ha) is too
    wide for a single Chebyshev filter band to resolve both HOMO and LUMO.
- **`scripts/compare_homo_lumo_3way.py`** — reads dense, sparse, and Fortran
  eigenvector TSVs, projects HOMO/LUMO onto a 2D grid via the OpenCL
  `GridProjector`, and produces a 3-column × 2-row contour plot.
- **`scripts/run_dftbplus_ref.py`** — updated to request
  `Analysis { WriteEigenvectors = Yes }` and parse `eigenvec.bin` (Fortran
  column-major float64 dump with a 4-byte identity prefix). Saves
  `ref_eigenvectors.tsv` in the same format as the Rust TSV.

### DFTB+ Fortran reference

- `eigenvec.bin` format: 4-byte identity int, then `norb × nstates` float64
  values in Fortran column-major order. Reshaped with `order='F'` and
  transposed to get `(nstates, norb)` in C order.
- The `WriteEigenvectors` option lives under `Analysis {}`, not `Options {}`
  (the parser rejects it under `Options`).

## Results

### Eigenvalue parity

| System | Method | HOMO (Ha) | LUMO (Ha) | Gap (Ha) |
|--------|--------|-----------|-----------|----------|
| Benzene | dense | -0.2452364357 | -0.2432386760 | 0.0019977597 |
| | fortran | -0.2452355790 | -0.2432400907 | 0.0019954883 |
| | sparse(Cheb+Ritz) | -0.2452364357 | -0.2432386760 | 0.0019977597 |
| Coronene | dense | -0.2377852332 | -0.2166289738 | 0.0211562594 |
| | fortran | -0.2377864913 | -0.2166299063 | 0.0211565850 |
| | sparse(Cheb+Ritz) | -0.2377852332 | -0.2166289738 | 0.0211562594 |
| Circumcoronene | dense | -0.2309937586 | -0.2282599088 | 0.0027338498 |
| | fortran | -0.2309952165 | -0.2282610669 | 0.0027341496 |
| | sparse(Cheb+Ritz) | -0.2310022294 | -0.2282643368 | 0.0027378926 |

### Eigenvalue differences

| System | ΔHOMO dense−Fortran | ΔLUMO dense−Fortran | ΔHOMO sparse−dense | ΔLUMO sparse−dense |
|--------|---------------------|---------------------|--------------------|--------------------|
| Benzene | -8.57e-07 | 1.41e-06 | 0.00e+00 | 0.00e+00 |
| Coronene | 1.26e-06 | 9.32e-07 | 0.00e+00 | 0.00e+00 |
| Circumcoronene | 1.46e-06 | 1.16e-06 | -8.47e-06 | -4.43e-06 |

### Sparse eigensolver parameters

The Chebyshev+Ritz solver converges to machine precision for benzene and
coronene with `--nvec 12 --cheb-deg 40 --iters 30`. Circumcoronene reaches
~4e-6 with the same parameters (residuals ~6e-3 in the HOMO band — more
iterations or higher Chebyshev degree would close the gap further).

Total SpMV count: ~57624 for each system (two bands × 30 iterations × 40
Chebyshev degree × 12 probe vectors × 2 for square filter + Rayleigh-Ritz).

### Wavefunction plots

Generated 3-column (dense | sparse | fortran) × 2-row (HOMO | LUMO) contour
plots for all three systems:

- `debug/graphene_sparse/benzene_homo_lumo_3way.png`
- `debug/graphene_sparse/coronene_homo_lumo_3way.png`
- `debug/graphene_sparse/circumcoronene_homo_lumo_3way.png`

All three methods produce visually identical orbital shapes (up to arbitrary
sign and degenerate-subspace rotation).

## Open problem: S^{-1/2} is the bottleneck

The current sparse eigensolver workflow has a **critical bottleneck**: the
generalized eigenproblem `H·c = ε·S·c` is transformed to standard form via

```
H' = S^{-1/2} · H · S^{-1/2}
```

which requires a **dense O(N³) eigendecomposition of S**:

```python
s_eigvals, s_eigvecs = np.linalg.eigh(S)          # O(N³)
S_inv_sqrt = s_eigvecs @ diag(1/sqrt(s_eigvals)) @ s_eigvecs.T  # O(N³)
H_std = S_inv_sqrt @ H @ S_inv_sqrt                # O(N³)
```

This defeats the purpose of the sparse iterative eigensolver, which is
designed to be O(N) via sparse matrix-vector products (SpMV). The Chebyshev
filter itself only needs SpMV operations — but the preprocessing step
dominates for large systems.

### Possible solutions

1. **Generalized Chebyshev filter** — modify the filter to work directly with
   `(H, S)` without forming `S^{-1/2}`. The Chebyshev polynomial would operate
   on the generalized operator `S^{-1}H` (or a factored form), using iterative
   solves (CG, MINRES) instead of explicit `S^{-1/2}`.

2. **Sparse Cholesky factorization** — compute `S = L·Lᵀ` with a sparse
   Cholesky (e.g. Eigen `SimplicialLLT`, SuiteSparse CHOLMOD), then transform
   `H' = L⁻¹·H·L⁻ᵀ` using sparse triangular solves. This is O(N) for
   well-localized overlap matrices.

3. **Polynomial approximation of S^{-1/2}** — use a Chebyshev or Newton
   iteration to approximate `S^{-1/2}` applied to vectors, without ever forming
   the full matrix. This fits naturally into the iterative eigensolver
   framework: each SpMV with `H'` becomes `S^{-1/2}·H·S^{-1/2}·v`, where
   `S^{-1/2}·v` is computed by an inner iterative solve.

4. **Palser–Manolopoulos canonical purification** — directly purify the
   density matrix `K = f(H, S)` without eigenvalue decomposition, using the
   same TC2-style purification already implemented for the sparse BSR4 path.
   This avoids the eigensolver entirely but gives natural orbital occupations,
   not individual eigenvalues.

### Current status

For the three PAH systems tested (6–54 atoms, 24–216 orbitals), the dense
`S^{-1/2}` is acceptable — it takes <1 second. But for the intended
large-scale sparse applications (thousands of atoms), this step will dominate.
The sparse iterative eigensolver is validated and ready; the `S^{-1/2}`
bottleneck is the next obstacle for true linear scaling.

## Files changed

- `rust_dftb/src/methods/dftb/hamiltonian.rs` — `SccResult.eigenvectors` field
- `rust_dftb/src/bin/dftb_engine.rs` — `rhai_save_eigenvectors`, `rhai_save_hs_matrix`
- `rust_dftb/scripts/test_charges_homo_lumo.rhai` — save eigenvectors + H,S matrices
- `scripts/sparse_homo_lumo.py` — Chebyshev+Ritz sparse eigensolver (new)
- `scripts/compare_homo_lumo_3way.py` — 3-way comparison (updated)
- `scripts/run_dftbplus_ref.py` — parse `eigenvec.bin` (updated)
- `scripts/plot_wavefunctions.py` — wavefunction projection (from prior session)
- `pyBall/__init__.py`, `pyBall/OCL/__init__.py` — package init files
- `doc/prokop/topical_audit/wavefunction_projection.md` — topical audit
- `CODEMAP.md`, `scripts/README.md`, `debug/graphene_sparse/README.md` — docs

## Related

- `doc/prokop/topical_audit/wavefunction_projection.md` — topical audit
- `doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md` — prior session
- `NumericalMathPlayground/topics/LinearAlgebra/SpectralFiltering/spectral_solvers.py` — source
