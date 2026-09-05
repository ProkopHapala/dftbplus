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
