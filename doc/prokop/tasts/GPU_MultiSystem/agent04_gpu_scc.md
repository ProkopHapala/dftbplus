---
type: Task
title: GPU Multi-System DFTB — Agent_04 GPU batched SCC (independent replicas)
tags: [parallel-agents, worker-task, gpu, scc]
---

# Agent_04: GPU batched SCC (independent replicas)

- **Master:** [`task_master.md`](task_master.md)
- **Agent ID:** `Agent_4`
- **Required contract version:** 1
- **Status authority:** coordinator only

Read the master completely before working. The master overrides this file. Execute
only this packet; other worker scopes are context, not optional work.

## Goal and boundary

- **Goal:** Extend `GpuDriver` (from Agent_1) with a host-driven SCC loop that runs all replicas in parallel on GPU: assemble H → diagonalize → compute charges → mix → converge. Save per-replica data (H0, S, H_scc, C, ε, D, q, E) to disk.
- **In scope:** `qmqm/gpu_driver.rs` (extend with SCC methods), `tests/gpu_scc.rs` (new). Mulliken charge computation on GPU or host; mixer on host; convergence check on host.
- **Out of scope:** Modifying OpenCL kernels (`dftb_hamiltonian.cl`, `gpu_matrix_ops.cl`). GPU-internal mega-kernel (D2b — future). QM/QM inter-fragment coupling on GPU (future). Scan/NEB driver (Agent_5).

## Inputs and preconditions

- Frozen inputs/fixtures:
  - `RUST_DFTB_SK_DIR` env var pointing to mio-1-1 SK files
  - Test molecules: 10× H2O at different geometries
  - CPU reference: `HamiltonianBuilder::build_scc()` (already parity-verified on 13 molecules)
- Upstream gate: **Agent_1 must be accepted** — `GpuDriver::new()` and `gpu_assemble_batched()` must work and pass `tests/gpu_hamiltonian.rs`.
- Interfaces consumed:
  - `qmqm/gpu_driver.rs::GpuDriver` (from Agent_1) — device, buffer management, H-assembly
  - `qmqm/gpu_matrix.rs::GpuMatrixContext::lowdin_transform()`, `local_jacobi_blocks()` — batched diagonalization
  - `qmqm/mixer.rs::SimpleMixer`, `DiisMixer` — host-side mixing
  - `qmqm/gamma.rs::GammaTable::from_sk_data()` — gamma for intra-fragment shifts
  - `qmqm/shifts.rs::compute_intra_shifts()` — intra-fragment SCC shifts (host-side)
  - `methods/dftb/hamiltonian.rs::HamiltonianBuilder::build_scc()` — CPU reference

## Exclusive ownership

- **May write:** `qmqm/gpu_driver.rs` (extend — Agent_1's code is the base), `tests/gpu_scc.rs` (new)
- **Read-only:** `methods/dftb/dftb_hamiltonian.cl`, `qmqm/gpu_matrix.rs`, `qmqm/gpu_matrix_ops.cl`, `qmqm/gpu_prep.rs`, `qmqm/mixer.rs`, `qmqm/shifts.rs`, `qmqm/gamma.rs`, `methods/dftb/hamiltonian.rs`
- **Must not:** edit kernel files, edit `gpu_matrix.rs` or `gpu_prep.rs` or `mixer.rs` or `shifts.rs`. If a change is needed, stop and report.

If any required edit falls outside ownership, stop and propose it to the coordinator.

## Work and verification

1. **Study Agent_1's handoff:** Read `GpuDriver` API from Agent_1's code. Understand `gpu_assemble_batched()` return type and buffer management. Read `gpu_matrix.rs` batched diagonalization API (`lowdin_transform`, `local_jacobi_blocks`).

2. **Implement `gpu_diagonalize_batched`:**
   - Input: H_buf, S_buf (GPU buffers from `gpu_assemble_batched`), n (matrix size), batch (n_replicas)
   - Compute S^{-1/2} on host (read S back, eigendecompose, reconstruct) — or on GPU if Agent_1 already does this
   - Löwdin transform: H' = X^T·H·X via `lowdin_transform` (two batched GEMMs)
   - Diagonalize H' via `local_jacobi_blocks` (batched, one workgroup per replica)
   - Back-transform: C = X·C' (one batched GEMM)
   - Read back eigenvalues (sorted ascending) and MO coefficients
   - Return: eps_buf, C_buf (GPU buffers or host Vec)

3. **Implement Mulliken charges (host-side, per replica):**
   - For each replica: D = 2·C_occ·C_occ^T (occupied = first n_electrons/2 columns)
   - q_atom[i] = sum_j D[i,j]·S[i,j] (diagonal of D·S)
   - delta_q[i] = q_atom[i] - q0[i]
   - This can be done on host after reading back C and S — simpler than a GPU kernel

4. **Implement `gpu_solve_scc_batched`:**
   ```rust
   pub fn gpu_solve_scc_batched(
       &self,
       geometries: &[(Vec<String>, Vec<[f64;3]>)],  // N replicas
       max_iter: usize,
       tol: f64,
   ) -> Result<SccBatchResult>
   ```
   - For each SCC iteration:
     a. Build `GpuBatch` from all replicas' current charges (host-side `gpu_prep`)
     b. `gpu_assemble_batched` → H, S on GPU
     c. `gpu_diagonalize_batched` → eps, C on GPU
     d. Read back C, eps to host
     e. Compute Mulliken charges on host (per replica)
     f. Compute residual = q_out - q_in
     g. Mix: `SimpleMixer::mix` or `DiisMixer::mix` (host-side, per replica or global)
     h. Check convergence: RMS residual < tol for all replicas
     i. If not converged, update charges and repeat
   - Return: `SccBatchResult { energies, charges, eigenvalues, n_iters, ... }`

5. **Implement `save_replica_data`:**
   - For each replica, write binary file: `{dir}/replica_{i}.bin`
   - Contents: H0 (f64, N²), S (f64, N²), H_scc (f64, N²), C (f64, N²), eps (f64, N), D (f64, N²), q (f64, N_atoms), E (f64, 1)
   - Simple format: magic number + N + N_atoms + data arrays (no external dependency)

6. **Write `tests/gpu_scc.rs`:**
   - `test_gpu_scc_parity_h2` — 1× H2, compare converged charges + energy vs CPU `build_scc`, tol 1e-5
   - `test_gpu_scc_independent_h2o` — 10× H2O at different geometries, all in one batch, each matches CPU `build_scc` within 1e-5 energy, 1e-5 charges
   - `test_gpu_save_replica_data` — run SCC, save data, read back, verify shapes and values
   - Skip gracefully if no OpenCL device

7. **Run tests:**
   ```bash
   export RUST_DFTB_SK_DIR=/path/to/mio-1-1
   cargo test --test gpu_scc -- --nocapture
   ```

**Commands:**

```bash
export RUST_DFTB_SK_DIR=/path/to/mio-1-1
cargo test --test gpu_scc -- --nocapture
```

**Expected deliverables:**
- `qmqm/gpu_driver.rs` extended with `gpu_diagonalize_batched`, `gpu_solve_scc_batched`, `save_replica_data`
- `tests/gpu_scc.rs` — 3 tests passing
- SCC parity < 1e-5 energy, < 1e-5 charges (f32 GPU vs f64 CPU)
- Binary data files saved correctly

## Handoff to coordinator/consumer

When finished, write your handoff report directly into the `## Agent reports` section at
the bottom of the **master** file (not this worker file). Check your checkbox `[ ]` → `[x]`
in the master's dispatch checklist. Your report must include:

1. Contract version and baseline used.
2. Changed files and concise rationale.
3. Exact commands plus full pass/fail results.
4. Artifact and `REVIEW:` paths.
5. Produced interface: `SccBatchResult` struct, `gpu_solve_scc_batched` signature, `save_replica_data` format (for Agent_5).
6. Worst discrepancy, assumptions, unresolved risks, and requested coordinator edits.

You MAY edit only your own checkbox and your own report in the master file. Do not edit
anything else in the master. Do not mark the aggregate task fixed/resolved/done.
