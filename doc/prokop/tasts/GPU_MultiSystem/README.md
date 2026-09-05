# doc/prokop/tasts/GPU_MultiSystem/

Task specifications for the GPU multi-system DFTB project — batched DFTB on
consumer GPUs for many small independent systems (scans, NEB, MD ensembles).

- **task_master.md** — master ledger: agent assignments, wave structure,
  acceptance status, cross-references.
- **agent01_gpu_driver.md** — GPU driver: OpenCL context, batched H0/S assembly.
- **agent02_cpu_multifrag.md** — CPU multi-fragment solver: `MultiSystemSolver`,
  `Fragment`, `FragmentTemplate`, DIIS mixer.
- **agent03_forces.md** — SCC forces: finite-difference dH0/dx, repulsive spline,
  gamma derivative.
- **agent04_gpu_scc.md** — GPU-resident SCC: device-side H_scc update, charge
  mixing, convergence.
- **agent05_scan_neb.md** — Scan/NEB: rigid and relaxed PES, nudged elastic band.
- **agent06_scc_kernels.md** — GPU SCC kernels: GEMM, Jacobi, purification.
- **hbond_switching.md** — H-bond switching benchmark: formic/azaindole dimer,
  1D/2D PES, computation modes, atom mappings, status (reactant optimization done).
- **sparse_bsr4_report.md** — BSR4 sparse purification report.
