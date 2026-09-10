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
  warm-start, best-effort mode. 1D/2D formic dimer scan validation. Analytic GPU
  forces in `qmqm/gpu_forces.rs`/`.cl` (1×4 crash fixed by `vload2`).
- **wavefunction_projection.md** — projecting MOs onto a real-space grid using
  pyBall OpenCL GridProjector + STO basis. Rust eigenvectors → 2D contour plots.
- **sk_interpolation.md** — SK radial interpolation. Production is C² cubic
  B-spline (CPU f64 reference, GPU f32). 2026-09-09: Neville `poly5_to_zero`
  tail removed (it exploded on H–H). Stopgap = blunt extra zero *samples* on
  the right + phantom control on the left. Next: general extra-control fitter
  (solve for pad points; do not hardcode zeros).
- **f32_floor_dense_hbond.md** — dense H-bond GPU: bugs vs method stopgaps vs
  measured floors. Package 2: AT `|dE|` tracks `δ_CH` (band vs `CᵀHC`), not
  frozen `δε_occ` (~1e-6). Löwdin Newton kept; f32 GEMM Kahan did not cut `δ_CH`.
  Honest two-tier test contract. Do not require AT `|dE|<1e-5`.
- **f32_floor_sparse.md** — sparse BSR4 Si/H: bugs (device NS N4) vs SK-q0
  misconception vs interpolator fitter vs **missing SparseDftb pipeline** vs
  measured Hessian floor (Gate G \|\|ΔH\|\|_F/\|\|H\|\|_F ≈ 0.11%). Manifest §0.
- **sparse_nanocrystal_vibrations.md** — sparse GPU DFTB for vibrational
  calculations on Si/H nanocrystals. See
  `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md` §0.
