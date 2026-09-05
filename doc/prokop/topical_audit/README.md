# doc/prokop/topical_audit/

Cross-implementation topic maps — one file per scientific topic, connecting
implementations of the same concept across Rust, Fortran, Python, and OpenCL.

- **davidson_eigensolver.md** — partial generalized eigensolver for frontier
  orbitals (`H C = S C ε`). Rust Davidson vs Python reference vs Fortran ELSI.
- **eigensolver_performance.md** — nalgebra Jacobi vs LAPACK dsyevd performance
  analysis. Why nalgebra is 29× slower, remaining bottlenecks, optimization plan.
- **sparse_tc2_purification.md** — BSR4 sparse density-matrix purification via
  TC2 on GPU. Rust + OpenCL vs Fortran dense reference.
- **dftbplus_parity_harness.md** — Python harness running the Fortran DFTB+
  binary for parity validation of Rust results.
- **scc_mulliken_charges.md** — atomic Mulliken charges after SCC: dense, sparse,
  and DFTB+ reference. Sign convention documentation.
- **wavefunction_projection.md** — projecting MOs onto a real-space grid using
  pyBall OpenCL GridProjector + STO basis. Rust eigenvectors → 2D contour plots.
