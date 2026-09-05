---
type: Task
title: GPU Multi-System DFTB — Agent_04 Brent-Luk Jacobi + S^{-1/2}
tags: [parallel-agents, worker-task, jacobi, eigensolver, opencl]
---

# Agent_04: Brent-Luk parallel cyclic Jacobi + S^{-1/2} on GPU

- **Master:** [`task_master.md`](task_master.md)
- **Agent ID:** `Agent_4`
- **Required contract version:** 2
- **Status authority:** coordinator only

Read the master completely before working. The master overrides this file. Execute
only this packet; other worker scopes are context, not optional work.

## Goal and boundary

- **Goal:** Implement a Brent-Luk parallel cyclic Jacobi eigensolver for batched
  symmetric matrices (N≤64, full-local in `__local` memory), and an S^{-1/2} kernel
  that reuses the Jacobi eigensolver. This is the central technical challenge of the
  revised architecture — it replaces the sequential-rotation `local_jacobi_blocks_parallel`
  with N/2 independent rotations per round and one barrier per round.
- **In scope:** `qmqm/gpu_eigen.cl` (new OpenCL kernel file), `qmqm/gpu_eigen.rs`
  (new Rust host module), `tests/gpu_eigenproblem.rs` (new tests + benchmarks).
- **Out of scope:** Modifying `gpu_matrix_ops.cl`, `gpu_matrix.rs`, `gpu_driver.rs`,
  `dftb_hamiltonian.cl`, or any existing kernel. SCC loop integration (coordinator).
  GEMM (Agent_6). Scan/NEB (Agent_5).

## Design reference

See `GPU_MultiSystem_Design.md` §4.1 (Brent-Luk Jacobi), §4.2 (S^{-1/2}), §3.1
(local memory budget), D8, D9, D10 (f32 precision).

Key design points:
- **N≤64:** A and V (eigenvectors) fully in `__local` memory. ~35 KiB/workgroup for N=64.
- **N+1 leading dimension** (`JLD = N+1`) to avoid power-of-two bank conflicts.
- **N+1 padding for odd N:** pad with dummy state (huge diagonal, zero off-diagonal).
  One kernel works for N=63 and N=64.
- **Round-robin schedule** computed in-kernel (no upload needed):
  ```c
  inline ushort2 jacobi_pair(int round, int ipair) {
      if (ipair == 0) return (ushort2)(JN-1, round);
      int m = JN-1;
      return (ushort2)((round+ipair)%m, (round+m-ipair)%m);
  }
  ```
- **N/2 independent rotations per round.** Each work-item owns one 2×2 block update.
  No atomics, no races. One barrier per round. N-1 rounds per sweep.
- **f32 only.** Monitor λ_min(S) for precision.
- **Workgroup sizes:** N≤32 → WG=128, N~32-64 → WG=256. Query
  `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` if possible.
- **Compile-time specialization:** Use text substitution for `JN` (N+1 padded),
  `JLD` (N+1), `JPAIR` (N/2), `JROUND` (N-1), `WG`. The existing Rust matrix
  infrastructure already specializes kernels by text substitution.

## Inputs and preconditions

- **Coordinator pre-work DONE:** `GpuRuntime` exists in `qmqm/gpu_runtime.rs`.
  `pub mod gpu_eigen;` wired into `qmqm/mod.rs`. `gpu_eigen.rs` is a stub with
  `unimplemented!()` — replace the stub with your implementation.
- Frozen inputs/fixtures:
  - `RUST_DFTB_SK_DIR` env var pointing to mio-1-1 SK files
  - Test molecules: H2 (N=2), N2 (N=8), H2O (N=6), CH4 (N=8)
  - CPU reference: `HamiltonianBuilder::build_non_scc()` + `build_scc()` for eigenvalues
- Interfaces consumed:
  - `qmqm/gpu_runtime.rs::GpuRuntime` (from coordinator) — shared OpenCL context/queue/program.
    Key API: `GpuRuntime::new() -> Result<Self>`, `rt.build_program(&mut self, source) -> Result<Program>`
    (note: `&mut self` because of program cache), `rt.buffer_from_slice(&self, &[T]) -> Result<Buffer<T>>`,
    `rt.zero_buffer<T: Default>(&self, len) -> Result<Buffer<T>>`, `rt.read_buffer(&self, &Buffer<T>, &mut [T])`,
    `rt.queue() -> &Queue`, `rt.context() -> &Context`, `rt.caps() -> &GpuCapabilities`.
  - `qmqm/gpu_driver.rs::GpuDriver::gpu_assemble_batched()` (from Agent_1, read-only) —
    to get H0/S on GPU for testing. Note: `GpuDriver` has its own separate context/queue;
    for your tests you can either use `GpuDriver` to produce H/S and then copy to host and
    re-upload to `GpuRuntime` buffers, or just build H/S on CPU and upload via `GpuRuntime`.
  - CPU `nalgebra` symmetric eigendecomposition for reference comparison
- **Important:** The `cargo test` wrapper crashes with LLVM/clang. Use
  `cargo test --test gpu_eigenproblem --no-run` then run the binary directly
  from `~/.cargo-target-shared/debug/deps/`.

## Exclusive ownership

- **May write:** `qmqm/gpu_eigen.cl` (new), `qmqm/gpu_eigen.rs` (new), `tests/gpu_eigenproblem.rs` (new)
- **Read-only:** all other `src/` files
- **Must not:** edit any file outside your ownership. If `GpuRuntime` API is insufficient, stop and report.

## Work and verification

### 1. Write `qmqm/gpu_eigen.cl` — Brent-Luk Jacobi kernel

Implement `jacobi_cyclic_local_batched`:
- Input: `__global const float* A` (batched symmetric matrices), `__global float* V`
  (eigenvectors, initialized to identity), `int n`, `int batch`
- Output: A overwritten with eigenvalues on diagonal, V holds eigenvectors
- Layout: `A[batch][i][j]` at `batch*N*N + i*N + j`, same for V
- One workgroup per system (`get_group_id(0) = system_index`)
- Load A and V into `__local` with leading dimension `JLD = N+1`
- For each round (0..N-2):
  - N/2 work-items compute rotation parameters for their (p,q) pair
  - `barrier(CLK_LOCAL_MEM_FENCE)`
  - All work-items update 2×2 blocks of A: `B_ab = G_a^T · A_ab · G_b`
  - All work-items update V: `V <- V · G`
  - `barrier(CLK_LOCAL_MEM_FENCE)`
- Store A (diagonal = eigenvalues) and V back to global
- Multiple sweeps until off-diagonal norm < tolerance (or fixed 5-10 sweeps)

Also implement `build_inv_sqrt_from_eig`:
- Input: eigenvectors U, eigenvalues λ
- Output: X = U · diag(rsqrt(λ)) · U^T
- Load U into `__local`, compute `X_ij = sum_k U_ik * rsqrt(lambda_k) * U_jk`
- Return λ_min per system for precision monitoring

### 2. Write `qmqm/gpu_eigen.rs` — Rust host module

- `pub fn jacobi_cyclic_local_batched(rt: &GpuRuntime, a_buf: &Buffer<f32>, v_buf: &Buffer<f32>, n: usize, batch: usize) -> Result<()>`
  - Specializes kernel source by text substitution for given N
  - Compiles program via `rt.build_program()`
  - Launches with `global_size = batch * wg`, `local_size = wg`
- `pub fn build_inv_sqrt(rt: &GpuRuntime, s_buf: &Buffer<f32>, n: usize, batch: usize) -> Result<(Buffer<f32>, Buffer<f32>)>`
  - First calls `jacobi_cyclic_local_batched` on S
  - Then launches `build_inv_sqrt_from_eig` kernel
  - Returns (X_buf, lambda_min_buf)
- Handle N padding: if N is odd, pad to N+1 with dummy state

### 3. Write `tests/gpu_eigenproblem.rs` — tests + benchmarks

Tests:
- `test_jacobi_parity_h2` — H2 Hamiltonian (2×2), compare eigenvalues + eigenvectors vs CPU
- `test_jacobi_parity_n2` — N2 Hamiltonian (8×8), compare vs CPU
- `test_jacobi_parity_h2o` — H2O Hamiltonian (6×6), compare vs CPU
- `test_jacobi_parity_ch4` — CH4 Hamiltonian (8×8), compare vs CPU
- `test_inv_sqrt_h2` — S^{-1/2} for H2, compare vs CPU
- `test_inv_sqrt_n2` — S^{-1/2} for N2, compare vs CPU
- `test_inv_sqrt_h2o` — S^{-1/2} for H2O, compare vs CPU
- `test_lambda_min_reporting` — verify λ_min is returned and reasonable
- `test_batched_jacobi` — 10× H2O at different geometries, all in one launch
- Skip gracefully if no OpenCL device or no SK dir

Benchmarks:
- `bench_jacobi_vs_old` — compare `jacobi_cyclic_local_batched` vs existing
  `local_jacobi_blocks_parallel` at N=8,16,32,48,64 and batch=1,10,100,1000
- Print timing per system for each configuration

Tolerances:
- Eigenvalues: < 1e-4 vs CPU (f32)
- Eigenvectors: < 1e-3 vs CPU (f32, sign-insensitive)
- S^{-1/2}: < 1e-3 vs CPU (f32)

### 4. Run tests

```bash
export RUST_DFTB_SK_DIR=/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1
cargo build
# Run binary directly (cargo test wrapper has LLVM crash issue):
/home/prokophapala/.cargo-target-shared/debug/deps/gpu_eigenproblem-<hash> --nocapture
# Or: cargo test --test gpu_eigenproblem -- --nocapture
```

**Expected deliverables:**
- `qmqm/gpu_eigen.cl` — Brent-Luk Jacobi + S^{-1/2} kernels
- `qmqm/gpu_eigen.rs` — Rust host module with public API
- `tests/gpu_eigenproblem.rs` — all tests pass, benchmark results recorded
- Evidence: eigenvalue/eigenvector parity vs CPU for H2, N2, H2O, CH4

## Handoff to coordinator

When finished, write your handoff report into `## Agent reports` at the bottom of
the **master** file. Check your checkbox `[ ]` → `[x]` in the dispatch checklist.
Your report must include:

1. Contract version and baseline used.
2. Changed files and concise rationale.
3. Exact commands plus full pass/fail results.
4. Benchmark results (Jacobi vs old, timing per system).
5. Produced API: function signatures, kernel names, specialization parameters.
6. Worst discrepancy (eigenvalues, eigenvectors, S^{-1/2}), λ_min values observed.
7. Assumptions, unresolved risks, requested coordinator edits.

You MAY edit only your own checkbox and your own report in the master file.
