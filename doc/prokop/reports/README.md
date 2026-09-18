# doc/prokop/reports/

Written session reports — what was done, problems encountered, how they were
overcome, and open issues. Chronological order.

- **2025-09-05_scc_charges_davidson_parity.md** — SCC Mulliken charges exposure,
  HOMO-LUMO extraction, Davidson partial eigensolver implementation, DFTB+
  Fortran parity harness, plotting, and end-to-end validation on benzene /
  coronene / circumcoronene.
- **2025-09-05_hbond_optimization_lapack.md** — H-bond switching FIRE
  optimization, SCC convergence analysis (DIIS floor ~1e-8), bottleneck
  diagnosis (nalgebra Jacobi 20ms vs LAPACK 0.7ms), LAPACK dsyevd replacement,
  2.3× speedup (75s → 33s for 100 steps).
- **2025-09-05_3way_homo_lumo_wavefunctions.md** — 3-way HOMO/LUMO comparison
  (dense vs sparse Chebyshev+Ritz vs DFTB+ Fortran): eigenvalue parity, real-space
  wavefunction contour plots, and the S^{-1/2} dense transformation bottleneck
  for linear-scaling sparse eigensolving.
- **2025-09-06_gpu_scc_benchmarks.md** — **GPU dense multi-system SCC** performance
  benchmarks (batched Jacobi eigensolver + DIIS). Formic acid dimer, batch sizes
  1–441. Jacobi bottleneck (50–65%), throughput scaling, GPU vs CPU 60–120× speedup.
  Small-system path (N<100, many replicas).
- **2025-09-06_sparse_cholesky_ritz_scaling.md** — **sparse Chebyshev+Ritz eigensolver**
  scaling benchmarks on H-passivated carbon ribbons (N=76..1156). Cholesky-transformed
  implicit operator, dense BLAS triangular solves, spectral rescaling, adaptive
  parameters. 7.5× faster than dense at N=1156, all parities at machine precision.
  Large-system path (PAHs, ribbons, flakes).
- **2025-09-06_gpu_hs_assembly_bugfix.md** — GPU H0/S assembly bugfix (sp rotation
  kernel, ss×sp block indexing).
- **2025-09-06_scan_plots_and_gamma_fix.md** — formic dimer 2D scan plots and
  gamma function fix.
- **2026-09-16_dense_gpu_pes_forces_benchmark.md** — **dense-multi GPU PES/force
  parity vs CPU f64** (GC N–H···N 19-pt proton-transfer scan: shape error
  0.10 meV, barrier err 0.068 meV, max|ΔF| 5.5e-6 Ha/Å; CPU DIIS limit-cycle
  floor ~1e-8 diagnosed via convergence plots) + **20×20-scan saturation
  benchmark** on 7 systems (H2O→DTH, N=6–246, batch 1–400): 52–342k sys/s,
  8.5–157× vs sequential CPU; Jacobi O(N³) bound at large N.
- **2026-09-16_sparse_dmm_warm_density_hessian.md** — **sparse DMM warm-density
  update for FD Hessians** (R10, 330 Si): 3-SpGEMM generalized-commutator step
  `δK=−η(X+Xᵀ−2Y)` + planned McWeeny retraction, seeded from the central
  projector. 0.30% ΔF vs cold at ~0.3 s/eval (~25% faster). Bugs fixed: Z=S⁻¹
  (not S⁻¹ᐟ²) ascent-direction root cause; bsym plan on asymmetric operand
  (silent T·Xᵀ). Honest accounting of why warm ≠ 10× yet + open Tier-1 plan.
- **2026-09-16_dense_gpu_pes_forces_benchmark_UPDATED.md** — updated version of
  the dense-GPU PES/forces benchmark (post-T03 numbers).
- **2026-09-17_resident_jacobi_eigensolver.md** — **resident-memory Jacobi
  kernels** (`jacobi_resident_batched`, T08b): A in `__local` across all
  sweeps + per-sweep deferred-V apply via a global rotation log. ~2.2× vs
  best streaming-direct at N=86/batch=400, bit-identical accuracy; GC SCC
  4.01→2.19 ms/iter end-to-end. Auto-dispatched for n≤128 when lA fits
  `local_mem_size` (48 KB device → n≲96). Measured facts:
  `tasts/HBond_Relaxed_Scan_GPU/Measured_Facts_Jacobi_Sweeps.md`.
