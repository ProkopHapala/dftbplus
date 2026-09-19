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
- **gpu_eigen.rs** + **gpu_eigen.cl** / **gpu_tiled_jacobi.cl** / **gpu_block_jacobi.cl** — GPU-resident batched Jacobi eigensolvers, one workgroup per system, dispatched by `eigsolver_kind(n, local_mem)` / `RUST_DFTB_EIGSOLVER`={auto,direct,block,resident,resident_av} (fail-loud on unknown/unfit). Variants: Brent-Luk full-local (n≤64, gpu_eigen.cl); `jacobi_cyclic_global_batched` "direct" (n≤256, streams A/V — bandwidth-bound); **`jacobi_resident_batched`** "res-defV/res-AV" (n≤128-if-fits: A `__local`-resident across all sweeps + per-sweep deferred-V apply via global rotlog — ~2.2× vs direct at N=86, auto-default when lA+16KB ≤ local_mem); `block_jacobi_1wg` (n>128, compound-pivot, B/IMAX/ITOL via `RUST_DFTB_BJ_*`). FMA rotation-normalization preserves orthogonality. Sweeps: `cargo test --release --test gpu_tiled_jacobi <sweep> -- --ignored --nocapture`; measured digest `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Measured_Facts_Jacobi_Sweeps.md`, report `doc/prokop/reports/2026-09-17_resident_jacobi_eigensolver.md`.
- **gpu_purify.rs** + **gpu_purify.cl** — batched dense TC2 density-matrix
  purification (alternative eigensolve path, separate from Jacobi). One
  workgroup per system; fused step kernel computes D², idempotency error,
  trace, branch decision (contract `D←D²` / expand `D←2D−D²`), and a
  device-side `done[]` freeze per launch — zero host syncs inside the
  iteration loop. Palser–Gershgorin init (`tc2_init_batched`). GEMM
  interior: `RUST_DFTB_PURIFY_GEMM`=2 register-tiled symmetric-square
  (default — 22×11thr × 8×4reg, As_t staging serves both operands,
  6.7 TFLOPS ≈ 19 % peak isolated; fused branch/write from `acc`
  registers → **8.46 ms/solve at n=86/b400 cold**, beats Jacobi warm
  9.7 ms); =0 row·row reference; =1 tiled (broken — do not use).
  `RUST_DFTB_PURIFY_WG` / `_TILE` / `_TK` knobs. Warm-start + δK0
  occupied-subspace rotation (sparse `k_seed_shift` recipe) is the
  designed next step for SCC. Design + measured status:
  `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Alternative_Dense_Multi_Eigensolve.md`;
  parity/bench: `tests/gpu_purify.rs`; GEMM sweep: `tests/gpu_gemm.rs`.
- **gpu_gemm.rs** + **gpu_gemm.cl** — standalone batched-GEMM variant
  sweep + bench. Kernels `gemm_regtile` (register-tiled, compile-time
  `GEMM_{TX,TY,RTX,RTY,TK,SPLIT_M,SPLIT_N,SQ}` shape — split workgroups
  + symmetric-square `/sq` mode), `gemm_1elem` (floor), `gemm_fulla`.
  `GemmVariant` enum → `gemm_kernel` builder. Best: reg22x11/r8x4
  /sq = 6.2–7.8 TFLOPS ≈ 17–22 % peak (batch-scaled). The purify GEMM
  interior was tuned here first; parity + bench: `tests/gpu_gemm.rs`.
- **gpu_matrix_ops.cl** — OpenCL kernels: GEMM, Jacobi, density purification.
