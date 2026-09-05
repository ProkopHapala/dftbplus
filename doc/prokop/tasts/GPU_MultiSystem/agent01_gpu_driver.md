---
type: Task
title: GPU Multi-System DFTB — Agent_01 GPU driver & H-assembly runtime
tags: [parallel-agents, worker-task, gpu, opencl]
---

# Agent_01: GPU driver & H-assembly runtime

- **Master:** [`task_master.md`](task_master.md)
- **Agent ID:** `Agent_1`
- **Required contract version:** 1
- **Status authority:** coordinator only

Read the master completely before working. The master overrides this file. Execute
only this packet; other worker scopes are context, not optional work.

## Goal and boundary

- **Goal:** Write an OpenCL driver module that compiles `dftb_hamiltonian.cl`, uploads `GpuBatch` data to device buffers, launches `onsite_and_va` + `assemble_pairs` kernels, reads back H and S matrices, and verifies parity against CPU `HamiltonianBuilder::build_non_scc()`.
- **In scope:** `qmqm/gpu_driver.rs` (new file), `tests/gpu_hamiltonian.rs` (new test file). Device init, buffer management, kernel compilation, kernel launch, H/S readback, parity comparison.
- **Out of scope:** SCC loop (Agent_4), scan/NEB driver (Agent_5), modifying kernels or `gpu_prep.rs` or `gpu_matrix.rs`.

## Inputs and preconditions

- Frozen inputs/fixtures:
  - `RUST_DFTB_SK_DIR` env var pointing to mio-1-1 SK files
  - Test molecules: H2 (2 atoms, 2 orbs), N2 (2 atoms, 8 orbs) from `data/xyz/` or hardcoded coords
  - CPU reference: `HamiltonianBuilder::build_non_scc()` (already parity-verified vs Fortran)
- Upstream gate: none (Wave 1, independent)
- Interfaces consumed:
  - `qmqm/gpu_prep.rs::GpuBatch::from_fragments()` — produces flat arrays for OpenCL
  - `qmqm/gpu_matrix.rs::GpuMatrixContext` — existing OpenCL context/buffer API (reuse for device init)
  - `methods/dftb/dftb_hamiltonian.cl` — kernel source (read at runtime, compile via `ocl`)
  - `methods/dftb/hamiltonian.rs::HamiltonianBuilder::build_non_scc()` — CPU reference for parity

## Exclusive ownership

- **May write:** `qmqm/gpu_driver.rs` (new), `tests/gpu_hamiltonian.rs` (new)
- **Read-only:** `qmqm/gpu_matrix.rs`, `qmqm/gpu_prep.rs`, `qmqm/gpu_matrix_ops.cl`, `methods/dftb/dftb_hamiltonian.cl`, `methods/dftb/hamiltonian.rs`, all other `src/` files
- **Must not:** edit `lib.rs`, `qmqm/mod.rs` (coordinator will wire `pub mod gpu_driver;`), edit kernel files, edit `gpu_matrix.rs` or `gpu_prep.rs`. If a change is needed in any of these, stop and report.

If any required edit falls outside ownership, stop and propose it to the coordinator.

## Work and verification

1. **Study existing code:** Read `gpu_matrix.rs::GpuMatrixContext` to understand how OpenCL context/buffer/program are created. Read `gpu_prep.rs::GpuBatch` to understand the flat array layout. Read `dftb_hamiltonian.cl` kernel signatures (`onsite_and_va`, `assemble_pairs`) to know exact argument order and types.

2. **Write `qmqm/gpu_driver.rs`:**
   - `pub struct GpuDriver { ctx: GpuMatrixContext, ham_program: ocl::Program, ... }`
   - `pub fn new() -> Result<Self>` — init OpenCL context (reuse `GpuMatrixContext::new` or create separate), compile `dftb_hamiltonian.cl` from source (embed with `include_str!` or read from path)
   - `pub fn gpu_assemble_batched(&self, batch: &GpuBatch) -> Result<(Vec<f32>, Vec<f32>)>` — upload pair/fragment/SK/V_a buffers, launch `onsite_and_va` then `assemble_pairs` for each bucket, read back H_out and S_out
   - Handle the pair bucket structure: `GpuBatch` contains multiple `GpuPairBucket` entries, each needs a separate `assemble_pairs` launch with its own `block_type` and SK table
   - Convert f32 GPU results to f64 for comparison

3. **Write `tests/gpu_hamiltonian.rs`:**
   - `test_gpu_assemble_pairs_smoke` — build a GpuBatch from 1× H2, launch kernels, verify no panic, H/S have expected shape (2×2)
   - `test_gpu_hs_parity_h2` — compare GPU H/S vs `HamiltonianBuilder::build_non_scc()` for H2, max abs diff < 1e-5
   - `test_gpu_hs_parity_n2` — same for N2 (8×8)
   - `test_gpu_multi_replica` — 10× H2 at different bond lengths, all in one batch, verify each matches CPU
   - Skip gracefully if no OpenCL device (same pattern as `tests/gpu_diagonalization.rs::try_gpu_ctx`)

4. **Run tests and verify:**
   ```bash
   export RUST_DFTB_SK_DIR=/path/to/mio-1-1
   cargo test --test gpu_hamiltonian -- --nocapture
   ```

**Commands:**

```bash
export RUST_DFTB_SK_DIR=/path/to/mio-1-1
cargo test --test gpu_hamiltonian -- --nocapture
```

**Expected deliverables:**
- `qmqm/gpu_driver.rs` — working OpenCL driver with `GpuDriver::new()` and `gpu_assemble_batched()`
- `tests/gpu_hamiltonian.rs` — 4 tests passing (smoke, H2 parity, N2 parity, multi-replica)
- H/S parity < 1e-5 max abs diff (f32 GPU vs f64 CPU)

## Handoff to coordinator/consumer

When finished, write your handoff report directly into the `## Agent reports` section at
the bottom of the **master** file (not this worker file). Check your checkbox `[ ]` → `[x]`
in the master's dispatch checklist. Your report must include:

1. Contract version and baseline used.
2. Changed files and concise rationale.
3. Exact commands plus full pass/fail results.
4. Artifact and `REVIEW:` paths.
5. Produced interface/schema and downstream usage notes (Agent_4 will use `GpuDriver`).
6. Worst discrepancy, assumptions, unresolved risks, and requested coordinator edits.

You MAY edit only your own checkbox and your own report in the master file. Do not edit
anything else in the master. Do not mark the aggregate task fixed/resolved/done.
