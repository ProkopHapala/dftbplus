# doc/prokop/topical_audit/

Cross-implementation topic maps — one file per scientific topic, connecting
implementations of the same concept across Rust, Fortran, Python, and OpenCL.

- **davidson_eigensolver.md** — partial generalized eigensolver for frontier
  orbitals (`H C = S C ε`). Rust Davidson vs Python reference vs Fortran ELSI.
  Fails on coronene (diagonal preconditioner); Chebyshev+Ritz is the working
  alternative.
- **chebyshev_ritz_eigensolver.md** — sparse Chebyshev+Ritz eigensolver with
  Cholesky-transformed implicit operator `H' = L⁻¹HL⁻ᵀ`. Converges on all PAHs
  and H-passivated ribbons to N=1156. 7.5× faster than dense at N=1156.
  Optimizations: dense BLAS triangular solves, spectral rescaling, adaptive
  parameters.
- **eigensolver_performance.md** — nalgebra Jacobi vs LAPACK dsyevd performance
  analysis. Why nalgebra is 29× slower, remaining bottlenecks, optimization plan.
- **sparse_tc2_purification.md** — BSR4 sparse density-matrix purification via
  TC2 on GPU. Rust + OpenCL vs Fortran dense reference.
- **dftbplus_parity_harness.md** — Python harness running the Fortran DFTB+
  binary for parity validation of Rust results.
- **scc_mulliken_charges.md** — atomic Mulliken charges after SCC: dense, sparse,
  and DFTB+ reference. Sign convention documentation.
- **gpu_scc_pipeline.md** — device-resident GPU SCC loop (gamma matvec, H_scc
  update, GEMM, Jacobi, density, Mulliken, DIIS). Batched homogeneous templates,
  warm-start, best-effort mode. 1D/2D formic dimer scan validation.
- **wavefunction_projection.md** — projecting MOs onto a real-space grid using
  pyBall OpenCL GridProjector + STO basis. Rust eigenvectors → 2D contour plots.
- **sk_interpolation.md** — SK integral interpolation: Fortran Neville 8-point,
  Rust cubic Hermite spline (CPU), Rust cubic B-spline (GPU). Parity, performance,
  analytic derivative paths.
