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
