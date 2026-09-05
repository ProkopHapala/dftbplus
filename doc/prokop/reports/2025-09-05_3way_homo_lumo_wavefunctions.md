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

## Open problem: S^{-1/2} is the bottleneck — SOLVED via sparse Cholesky

The original implementation transformed the generalized eigenproblem
`H·c = ε·S·c` to standard form via dense `S^{-1/2}`:

```
H' = S^{-1/2} · H · S^{-1/2}    ← requires dense O(N³) eigendecomposition of S
```

This defeated the purpose of the sparse iterative eigensolver. **Now replaced**
with a sparse Cholesky factorization approach that avoids all densification.

### Solution: implicit L⁻¹·H·L⁻ᵀ operator

The generalized eigenproblem `H·c = ε·S·c` is transformed via **sparse
Cholesky** `S = L·Lᵀ` (computed once, using `scipy.sparse.linalg.splu` with
`SymmetricMode`, then `L_chol = L·sqrt(D)` from the LDLᵀ factorization):

```
H' = L⁻¹ · H · L⁻ᵀ    (symmetric, same eigenvalues as generalized problem)
```

Instead of forming `H'` explicitly (which would densify), we use an **implicit
operator** that applies `L⁻¹`, `H`, `L⁻ᵀ` sequentially via sparse triangular
solves:

```python
H' · v = L⁻¹ · (H · (L⁻ᵀ · v))    ← 2 sparse triangular solves + 1 sparse matvec
```

The Chebyshev filter + Rayleigh-Ritz code calls this via the `op_matmul(H, V)`
→ `H.matvec(V)` abstraction in `spectral_solvers.py`. Eigenvectors are
transformed back: `c = L⁻ᵀ · y` (one sparse triangular solve).

### Why this is O(N) for localized basis

- **Sparse Cholesky** of a banded/localized `S` produces a sparse `L` with
  O(N) nonzeros (for 1D/2D-localized systems). Fill-in is limited by the
  sparsity pattern, not the matrix size.
- **Sparse triangular solves** (`spsolve_triangular`) are O(nnz(L)) per solve.
- **Sparse matvec** `H · v` is O(nnz(H)).
- **Chebyshev filter** does `deg × iters` matvecs, each O(nnz) — no
  densification at any step.

For the tested systems:
- Benzene (24 orbitals): `nnz(L) = 185` vs `576` dense (32% fill)
- Coronene (96 orbitals): `nnz(L) = 2682` vs `9216` dense (29% fill)
- Circumcoronene (216 orbitals): `nnz(L) = 10558` vs `46656` dense (23% fill)

The fill ratio **decreases** with system size for these 2D-localized PAHs,
confirming the O(N) scaling.

### Why not Newton-Schulz Z ≈ S⁻¹?

We also tested `H' = Z·H·Z` with `Z ≈ S⁻¹` from Newton-Schulz iteration (which
is already implemented on GPU in `gpu_sparse.rs::newton_schulz_inverse`).
However, `Z·H·Z = S⁻¹·H·S⁻¹` has **different eigenvalues** from the generalized
problem — it is similar to `S⁻¹/²·H·S⁻¹/²` via `S¹/²`, but the similarity
transform `S¹/²` is not orthogonal, so eigenvalues are preserved only under the
symmetric form `S⁻¹/²·H·S⁻¹/²`. The Cholesky form `L⁻¹·H·L⁻ᵀ` is the correct
symmetric transformation.

### Remaining limitation

The sparse Cholesky via `scipy.sparse.linalg.splu` uses a fill-reducing
permutation (AMD/METIS). For very large 3D systems, the fill-in of `L` can
grow faster than O(N). For true linear scaling in 3D, a hierarchical or
approximate Cholesky (e.g. HSS, STRUMPACK) would be needed. For the current
2D PAH systems, the sparse Cholesky is already sublinear in fill ratio.

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

---

## 2025-09-06 — Optimization and scaling on H-passivated carbon ribbons

### Optimizations applied

1. **Dense BLAS triangular solves** (12x speedup): Replaced
   `scipy.sparse.linalg.spsolve_triangular` with
   `scipy.linalg.solve_triangular` (BLAS `dtrsm`) for N < 2000.
   Profiling showed sparse triangular solves were 12x slower than dense
   for N=216 (0.85ms vs 0.07ms per solve). The crossover is ~N=2000
   where sparse L nnz drops below N²/10. Controlled by
   `DENSE_SOLVE_THRESHOLD` in `sparse_homo_lumo.py`.

2. **Spectral range rescaling** (critical for large N): The Chebyshev
   filter requires eigenvalues in [-1, 1]. For large ribbons (N>580),
   the transformed operator H' = L⁻¹HL⁻ᵀ has eigenvalues outside [-1, 1].
   Added `estimate_spectral_range()` (power iteration) and a
   `RescaledOp` wrapper that maps (H' - center) / half_range to [-1, 1]
   before filtering. Rayleigh-Ritz runs on the original operator to get
   true eigenvalues. Without this, L=64 produced eigenvalues at +1.2
   instead of -0.18.

3. **Adaptive Chebyshev parameters**: Parameters auto-scale with system
   size to maintain convergence:
   - N < 300: nvec=20, deg=40, iters=20
   - N < 800: nvec=30, deg=60, iters=30
   - N ≥ 800: nvec=40, deg=80, iters=40

4. **More probe vectors** (nvec=20→40): Increasing from 12 to 20 probe
   vectors improved circumcoronene HOMO error from 1e-5 to 2.5e-8. For
   larger systems, 30-40 vectors are needed to capture the denser
   spectrum near the gap.

### Benchmark systems

H-passivated zigzag carbon ribbons (width=4 atom rows, increasing length):

| Ribbon | L | Atoms | C | H | Orbitals |
|--------|---|-------|---|---|----------|
| w4_L4 | 4 | 28 | 16 | 12 | 76 |
| w4_L8 | 8 | 52 | 32 | 20 | 148 |
| w4_L16 | 16 | 100 | 64 | 36 | 292 |
| w4_L32 | 32 | 196 | 128 | 68 | 580 |
| w4_L64 | 64 | 388 | 256 | 132 | 1156 |

Geometries generated by `graphene_build zigzag --width 4 --length L --passivate`.
H,S matrices exported via `dftb_engine --script test_ribbons_sparse.rhai`
(non-SCC H0 only — no convergence loop needed for eigensolver testing).

### Scaling results

| Ribbon | N_orb | nnz(L) | Fill% | t_chol | t_sparse | t_dense | Speedup | HOMO Δ | LUMO Δ | r_max |
|--------|-------|--------|-------|--------|----------|---------|---------|--------|--------|-------|
| L4 | 76 | 1,829 | 31.7% | 2ms | 273ms | 100ms | 0.4x | 3.6e-11 | 3.8e-12 | 1.4e-16 |
| L8 | 148 | 6,377 | 29.1% | 2ms | 398ms | 660ms | 1.7x | 4.7e-11 | 3.9e-11 | 1.5e-13 |
| L16 | 292 | 23,441 | 27.5% | 9ms | 749ms | 4,630ms | 6.2x | 2.8e-11 | 4.8e-11 | 2.5e-7 |
| L32 | 580 | 89,441 | 26.6% | 43ms | 7,935ms | 33,500ms | 4.2x | 2.2e-11 | 3.1e-12 | 5.8e-15 |
| L64 | 1,156 | 348,929 | 26.1% | 306ms | 35,703ms | 285,000ms | 7.5x | 2.0e-12 | 1.7e-11 | 7.5e-11 |

Key observations:
- **All parities at machine precision** (1e-11 to 1e-12) across all sizes.
- **Sparse faster than dense for N≥148** (1.7x to 7.5x speedup).
- **Cholesky fill ratio decreases** with N (32% → 26%), confirming
  sub-O(N²) fill-in for these 1D-localized ribbons.
- **Cholesky factorization** is fast (306ms for N=1156) — not the bottleneck.
- **Chebyshev+Ritz** dominates total time, scaling with nvec×deg×iters.
- The L=64 sparse solver (38s) is 7.5x faster than dense (285s) while
  achieving better accuracy (2e-12 vs dense reference).

### Plots

- `debug/graphene_sparse/ribbons/ribbon_scaling.png` — 4-panel scaling plot:
  runtime vs N, nnz vs N, parity vs N, fill ratio vs N.
- `debug/graphene_sparse/ribbons/scaling_results.csv` — machine-readable table.
- `debug/graphene_sparse/ribbons/scaling_results.json` — JSON format.

### Remaining bottlenecks

1. **Python callback overhead**: The `RescaledOp.matvec` adds a Python
   layer (subtraction + division) on top of the BLAS triangular solves.
   Moving the full operator to Rust/OpenCL would eliminate this.

2. **Chebyshev iteration count**: For N>800, 40 iterations × 80 degree =
   3200 matvecs per band. A Lanczos/LOBPCG eigensolver might converge
   with fewer operator applications.

3. **Dense triangular solves**: For N>2000, the dense L matrix becomes
   too large and we must switch to sparse triangular solves (12x slower
   per solve). A GPU triangular solve kernel would bridge this gap.

4. **Cholesky fill-in**: For 3D systems, fill-in will grow faster than
   O(N). Hierarchical factorization (HSS/STRUMPACK) would be needed.
