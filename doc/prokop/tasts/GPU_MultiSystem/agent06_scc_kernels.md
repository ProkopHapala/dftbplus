---
type: Task
title: GPU Multi-System DFTB — Agent_06 GEMM + SCC component kernels
tags: [parallel-agents, worker-task, gemm, scc, opencl]
---

# Agent_06: Full-local batched GEMM + SCC component kernels

- **Master:** [`task_master.md`](task_master.md)
- **Agent ID:** `Agent_6`
- **Required contract version:** 2
- **Status authority:** coordinator only

Read the master completely before working. The master overrides this file. Execute
only this packet; other worker scopes are context, not optional work.

## Goal and boundary

- **Goal:** Implement (1) a full-local batched GEMM kernel for N≤64 that loads both
  matrices into `__local` memory once, and (2) the individual SCC component kernels
  (gamma matvec, H_scc_update, Mulliken charges, residual+mixer) that the coordinator
  will wire into the device-resident SCC loop. These are all independent of the Jacobi
  eigensolver (Agent_4) — they're separate kernels with no dependency.
- **In scope:** `qmqm/gpu_matrix_ops.cl` (extend with new kernels), `qmqm/gpu_matrix.rs`
  (extend with host code), `tests/gpu_scc_kernels.rs` (new tests).
- **Out of scope:** Jacobi/eigensolver (Agent_4). SCC loop driver/integration
  (coordinator). Scan/NEB (Agent_5). Editing `gpu_eigen.cl`, `gpu_eigen.rs`,
  `gpu_driver.rs`, `dftb_hamiltonian.cl`.

## Design reference

See `GPU_MultiSystem_Design.md` §4.3 (full-local GEMM), §4.4 (H_scc_update),
D15 (gamma precompute), D16 (H0/S separated from SCC), D11 (device-resident SCC).

Key design points:
- **Full-local GEMM:** One WG/system. Load both A and B into `__local` (N+1 leading
  dimension). Each thread computes N²/WG elements. For N=64, WG=256: 16 elements/thread.
  Memory access: `LA[i,k]` broadcast within warp, `LB[k,j]` consecutive across j.
- **Gamma matvec:** V_A = sum_B G_AB · Δq_B. G is dense N_atom×N_atom. One WG/system,
  each thread handles one atom.
- **H_scc_update:** H = H0 + 0.5·S·(V_i + V_j). Elementwise, cache V_atom in `__local`.
  Uses `orb_atom[]` mapping (orbital → atom index).
- **Mulliken charges:** q_A = sum_μ (D·S)_μμ where μ belongs to atom A. One WG/system,
  reduction over orbitals per atom.
- **Residual + mix:** RMS = ||q_new - q_old||, q_mixed = α·q_new + (1-α)·q_old.
  One WG/system, reduction for RMS.
- **f32 throughout.** All kernels use `float`, not `double`.

## Inputs and preconditions

- **Coordinator pre-work DONE:** `GpuRuntime` exists in `qmqm/gpu_runtime.rs`.
  Key API: `GpuRuntime::new() -> Result<Self>`, `rt.build_program(&mut self, source) -> Result<Program>`
  (note: `&mut self` because of program cache), `rt.buffer_from_slice(&self, &[T]) -> Result<Buffer<T>>`,
  `rt.zero_buffer<T: Default>(&self, len) -> Result<Buffer<T>>`, `rt.read_buffer(&self, &Buffer<T>, &mut [T])`,
  `rt.queue() -> &Queue`, `rt.caps() -> &GpuCapabilities`.
- Frozen inputs/fixtures:
  - `RUST_DFTB_SK_DIR` env var
  - Test molecules: H2, N2, H2O, CH4
  - CPU reference: `HamiltonianBuilder::build_non_scc()`, `build_scc()`, `gamma_full()`
- Interfaces consumed:
  - `qmqm/gpu_runtime.rs::GpuRuntime` (from coordinator) — shared OpenCL context/queue
  - Existing `gpu_matrix_ops.cl` kernels (read-only reference for style/conventions)
  - `qmqm/gamma.rs::gamma_full` (CPU reference for gamma matvec test)
- **Important:** The `cargo test` wrapper crashes with LLVM/clang. Use
  `cargo test --test gpu_scc_kernels --no-run` then run the binary directly
  from `~/.cargo-target-shared/debug/deps/`.
- **Important:** Existing `gpu_matrix.rs::GpuMatrixContext` has its own context/queue.
  Your new functions should use `GpuRuntime` instead. Do NOT modify existing
  `GpuMatrixContext` methods — only append new functions.

## Exclusive ownership

- **May write:** `qmqm/gpu_matrix_ops.cl` (extend — append new kernels, do not modify
  existing ones), `qmqm/gpu_matrix.rs` (extend — append new functions, do not modify
  existing ones), `tests/gpu_scc_kernels.rs` (new)
- **Read-only:** all other `src/` files
- **Must not:** edit `gpu_eigen.cl`, `gpu_eigen.rs`, `gpu_driver.rs`,
  `dftb_hamiltonian.cl`, `gpu_runtime.rs`. If `GpuRuntime` API is insufficient, stop and report.

## Work and verification

### 1. Full-local batched GEMM kernel

Add `matmul_full_local_batched` to `gpu_matrix_ops.cl`:
```c
__kernel void matmul_full_local_batched(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int N_ORB, int WG
){
    int sid = get_group_id(0);
    int lid = get_local_id(0);
    __local float LA[N_ORB*(N_ORB+1)];
    __local float LB[N_ORB*(N_ORB+1)];
    // global → local once
    // C = A·B
}
```
- Specialize via text substitution for N_ORB and WG
- One WG per system, full matrices in `__local`
- Benchmark vs existing `batched_gemm` (tiled) at N=8,16,32,48,64, batch=1,10,100,1000

Add host function to `gpu_matrix.rs`:
- `pub fn matmul_full_local_batched(rt: &GpuRuntime, a_buf: &Buffer<f32>, b_buf: &Buffer<f32>, c_buf: &Buffer<f32>, n: usize, batch: usize) -> Result<()>`

### 2. Gamma matvec kernel

Add `gamma_matvec_batched` to `gpu_matrix_ops.cl`:
- Input: G (dense `[batch][Na*Na]`), Δq (`[batch][Na]`)
- Output: V (`[batch][Na]`) = G · Δq
- One WG/system, each thread computes one V_A = sum_B G[AB] * dq[B]
- Naive reduction (Na ≤ 100, so ≤100 elements per thread)

Add host function: `pub fn gamma_matvec_batched(rt: &GpuRuntime, g_buf: &Buffer<f32>, dq_buf: &Buffer<f32>, v_buf: &Buffer<f32>, n_atoms: usize, batch: usize) -> Result<()>`

### 3. H_scc_update kernel

Add `h_scc_update_batched` to `gpu_matrix_ops.cl`:
- Input: H0, S, V_atom (`[batch][Na]`), orb_atom (`[batch][N]` orbital→atom map)
- Output: H = H0 + 0.5·S·(V[atom_i] + V[atom_j])
- Cache V_atom in `__local`, elementwise over N²
- See `GPU_MultiSystem_Design.md` §4.4 for pseudocode

Add host function: `pub fn h_scc_update_batched(rt: &GpuRuntime, h0_buf: &Buffer<f32>, s_buf: &Buffer<f32>, v_buf: &Buffer<f32>, h_buf: &Buffer<f32>, orb_atom_buf: &Buffer<i32>, n: usize, n_atoms: usize, batch: usize) -> Result<()>`

### 4. Mulliken charges kernel

Add `mulliken_charges_batched` to `gpu_matrix_ops.cl`:
- Input: D, S, orb_atom map
- Output: q (`[batch][Na]`) = per-atom Mulliken charges
- q_A = sum_{μ ∈ atom A} (D·S)_μμ
- One WG/system, compute D·S diagonal, reduce per atom

Add host function: `pub fn mulliken_charges_batched(rt: &GpuRuntime, d_buf: &Buffer<f32>, s_buf: &Buffer<f32>, q_buf: &Buffer<f32>, orb_atom_buf: &Buffer<i32>, n: usize, n_atoms: usize, batch: usize) -> Result<()>`

### 5. Residual + simple mixer kernel

Add `residual_and_mix_batched` to `gpu_matrix_ops.cl`:
- Input: q_new, q_old, alpha
- Output: q_mixed = alpha*q_new + (1-alpha)*q_old, rms = ||q_new - q_old||
- One WG/system, reduction for RMS

Add host function: `pub fn residual_and_mix_batched(rt: &GpuRuntime, q_new_buf: &Buffer<f32>, q_old_buf: &Buffer<f32>, q_mixed_buf: &Buffer<f32>, rms_buf: &Buffer<f32>, alpha: f32, n_atoms: usize, batch: usize) -> Result<()>`

### 6. Write `tests/gpu_scc_kernels.rs`

Tests (all vs CPU reference, f32 tolerance):
- `test_matmul_full_local_h2` — 2×2 GEMM vs CPU
- `test_matmul_full_local_n2` — 8×8 GEMM vs CPU
- `test_matmul_full_local_h2o` — 6×6 GEMM vs CPU
- `test_matmul_full_local_batched` — 10× H2O GEMM in one launch
- `test_gamma_matvec_h2o` — gamma matvec vs CPU `gamma_full` for H2O
- `test_h_scc_update_h2o` — H_scc = H0 + 0.5·S·(V_i+V_j) vs CPU for H2O
- `test_mulliken_charges_h2o` — Mulliken charges vs CPU `Fragment::compute_charges`
- `test_residual_and_mix` — residual + mixing vs CPU
- `test_batched_scc_kernels` — 10× H2O, all kernels batched
- Skip gracefully if no OpenCL device or no SK dir

Benchmark:
- `bench_gemm_full_local_vs_tiled` — compare at N=8,16,32,48,64, batch=1,10,100,1000

Tolerances:
- GEMM: < 1e-4 vs CPU
- Gamma matvec: < 1e-4 vs CPU
- H_scc_update: < 1e-4 vs CPU
- Mulliken charges: < 1e-4 vs CPU
- Residual+mix: < 1e-6 vs CPU (exact arithmetic)

### 7. Run tests

```bash
export RUST_DFTB_SK_DIR=/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1
cargo build
# Run binary directly (cargo test wrapper has LLVM crash issue):
/home/prokophapala/.cargo-target-shared/debug/deps/gpu_scc_kernels-<hash> --nocapture
```

**Expected deliverables:**
- `qmqm/gpu_matrix_ops.cl` — 5 new kernels appended (do not modify existing kernels)
- `qmqm/gpu_matrix.rs` — 5 new host functions appended
- `tests/gpu_scc_kernels.rs` — all tests pass, benchmark results recorded
- Evidence: each kernel matches CPU reference < 1e-4

## Handoff to coordinator

When finished, write your handoff report into `## Agent reports` at the bottom of
the **master** file. Check your checkbox `[ ]` → `[x]` in the dispatch checklist.
Your report must include:

1. Contract version and baseline used.
2. Changed files and concise rationale.
3. Exact commands plus full pass/fail results.
4. Benchmark results (full-local GEMM vs tiled, timing per system).
5. Produced API: all 5 function signatures + kernel names.
6. Worst discrepancy per kernel, assumptions, unresolved risks.
7. Requested coordinator edits (if any).

You MAY edit only your own checkbox and your own report in the master file.
