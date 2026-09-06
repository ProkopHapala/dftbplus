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
