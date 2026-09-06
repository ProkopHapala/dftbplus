# rust_dftb/src/qmqm/

Multi-fragment QM/QM solver with OpenCL GPU offload — the orchestration layer
for batched DFTB on many independent systems.

- **fragment.rs** — `Fragment` / `FragmentTemplate` encapsulation. Each fragment
  owns H0, S, charges, eigenvectors, and the SCC Hamiltonian. `diagonalize()`
  solves the generalized eigenproblem `H·c = E·S·c` via Cholesky transform +
  LAPACK `dsyevd` (was nalgebra Jacobi — 29× slower, see
  `doc/prokop/topical_audit/eigensolver_performance.md`).
- **solver.rs** — `MultiSystemSolver` with flattened global charge vector.
  `solve_scc` runs the SCC loop: build H_scc → diagonalize → Mulliken charges →
  DIIS mix → converge. Zero-allocation hot loop (pre-allocated buffers).
  Verbose per-iteration output via `RUST_DFTB_SCC_VERBOSE=1`.
- **mixer.rs** — DIIS mixer with 5-iteration simple-mixing warmup. Saturates at
  RMS ~1e-8 for typical systems (do not request tighter tolerance).
- **shifts.rs** — intra- and inter-fragment potential shifts.
- **gamma.rs** — `GammaTable` for the multi-fragment Coulomb model.
- **charges.rs** — charge gathering/scattering between global vector and fragments.
- **neighbor.rs** — `FragmentNeighborList` (cell-list, O(N) centroid neighbor finding).
- **gpu_driver.rs** — OpenCL driver for batched H0/S assembly on GPU.
- **gpu_runtime.rs** — OpenCL context/queue/program management.
- **gpu_matrix.rs** — `GpuMatrixContext`: batched GEMM, Jacobi, purification kernels.
- **gpu_prep.rs** — GPU system preparation (orbital mapping, species packing).
- **gpu_eigen.rs** / **gpu_eigen.cl** — GPU-resident Jacobi eigensolver (Brent-Luk). Pair-block update uses two barriers/round; `JACOBI_BLOCK_UPDATE=0` retains the two-pass reference. FMA rotation-normalization refinement preserves orthogonality (`JACOBI_NORMALIZE_ROTATION=0` for original diagnostic reference). Release tests: `cargo test --release --lib jacobi_block -- --include-ignored --nocapture --test-threads=1`; select NVIDIA explicitly with `OCL_DEFAULT_PLATFORM_IDX` after checking `clinfo -l`. `jacobi_sweep_diagnostic` is an ignored per-sweep numerical diagnostic. Measurements and remaining convergence/assembly caveats: `doc/prokop/chats/GPU_Optimization.chat.md` (repo root).
- **gpu_matrix_ops.cl** — OpenCL kernels: GEMM, Jacobi, density purification.
