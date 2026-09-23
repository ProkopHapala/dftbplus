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
- **eigensolver_performance.md** — eigensolver performance map. CPU: nalgebra
  vs LAPACK dsyevd (why nalgebra is 29× slower). GPU: batched Jacobi variants
  (Brent-Luk n≤64, streaming direct n≤256, resident-A+deferred-V n≲96,
  block n>128) + `eigsolver_kind` dispatch; measured digest
  `tasts/HBond_Relaxed_Scan_GPU/Measured_Facts_Jacobi_Sweeps.md`.
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
  pyBall OpenCL + STO basis. Two packings: Γ-only Fireball order in `Grid.cl`,
  and k-resolved DFTB+ order in `DFTBplusGrid.cl` (`project_bloch_points`).
  User guide: `userguide/bloch_slice.md`.
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
- **sparse_nanocrystal_vibrations.md** — sparse GPU DFTB vibrations.
  2026-09-23: frozen-density C330 spectrum (no imaginary modes) and
  FIRE-step Hessian vs DFTB+ on adamantane and Si₁₀H₁₆. User guide
  `userguide/sparse_vibrations.md` §0.
- **hessian_eval_bottleneck.md** — per-eval cost breakdown of the sparse
  FD Hessian (SCC = >98%); mode ladder A (frozen) / B (fixed-q) / C (DMM
  warm update — implemented, 0.3% ΔF at ~25% vs cold); batch-parallel
  columns is the product-level fix.
- **cdft_constraints.md** — constrained DFT (fragment Mulliken-charge
  constraints) on the dense GPU solver: the λ-shift enters `h_scc` as
  `½λ·S·(w_μ+w_ν)`, so one kernel + an outer-λ host loop give
  charge-localized diabatic states with correct constrained-surface
  forces. Per-replica targets = diabatic ladder in one batch.
- **gpu_pbc_hbond_scans.md** — 2-D proton-transfer scans on a periodic
  H-bond wire via `GpuPbc` (`pbc_*` rhai bindings). QX/HQ chain built
  from ascii-art with herringbone tilt; one junction crosses the cell
  boundary. Batched 400-replica SCC: d1↔d2 symmetry 9e-6 Ha, degenerate
  endpoints, stepwise-wins mechanism.
