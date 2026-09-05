---
type: Task
title: GPU Multi-System DFTB — Agent_05 Scan/NEB driver (CPU backend)
tags: [parallel-agents, worker-task, scan, neb]
---

# Agent_05: Scan/NEB driver (CPU backend)

- **Master:** [`task_master.md`](task_master.md)
- **Agent ID:** `Agent_5`
- **Required contract version:** 2
- **Status authority:** coordinator only

Read the master completely before working. The master overrides this file. Execute
only this packet; other worker scopes are context, not optional work.

## Goal and boundary

- **Goal:** Write a rigid coordinate scan driver and a NEB driver that use the EXISTING CPU `HamiltonianBuilder::build_scc()` to evaluate many geometries. The scan driver sweeps one coordinate and plots the energy curve. The NEB driver interpolates between two endpoints and optimizes the band. This is INDEPENDENT of Agent_4 — no GPU dependency. The coordinator will later swap the CPU backend for Agent_4's GPU batched SCC (one API call change).
- **In scope:** `examples/scan.rs` (new), `examples/neb.rs` (new), `tests/scan.rs` (new). Geometry generation (rigid scan, linear interpolation), CPU SCC call per geometry, energy curve output, basic NEB spring forces + image update (host-side).
- **Out of scope:** Modifying any `src/` file. GPU internals. Force computation (use Agent_3's `forces.rs` API if available; if not, NEB uses energy-only finite-difference forces). Kernel work.

## Inputs and preconditions

- Frozen inputs/fixtures:
  - `RUST_DFTB_SK_DIR` env var pointing to mio-1-1 SK files
  - Test molecules: H2 (bond scan 0.5–3.0 Å), H2O (angle scan)
  - CPU reference: `HamiltonianBuilder::build_scc()` — this IS the backend, not just a reference
- Upstream gate: **Agent_3 gate (optional):** if forces are available, NEB can use them; if not, NEB uses energy-only finite-difference forces. **NO dependency on Agent_4.**
- Interfaces consumed:
  - `methods/dftb/hamiltonian.rs::HamiltonianBuilder::build_scc()` — CPU SCC backend (already exists, parity-verified)
  - `methods/dftb/forces.rs` (from Agent_3, if available) — forces for NEB
  - `qmqm/fragment.rs::Fragment` / `FragmentTemplate` — for constructing per-geometry fragments

## Exclusive ownership

- **May write:** `examples/scan.rs` (new), `examples/neb.rs` (new), `tests/scan.rs` (new)
- **Read-only:** all `src/` files
- **Must not:** edit any `src/` file. If a change is needed in `hamiltonian.rs` API, stop and report.

If any required edit falls outside ownership, stop and propose it to the coordinator.

## Work and verification

1. **Study the CPU SCC API:** Read `HamiltonianBuilder::build_scc()` signature and `SccResult` struct. Understand input format (species, coords, SK data). Read `Fragment`/`FragmentTemplate` for constructing per-geometry fragments.

2. **Write `examples/scan.rs` — rigid coordinate scan:**
   - Input: molecule XYZ, scan coordinate definition (e.g. "bond 0 1 from 0.5 to 3.0 Å, 20 points")
   - Generate 20 geometries by modifying the specified coordinate
   - Call `HamiltonianBuilder::build_scc()` for each geometry (serial CPU loop)
   - Output: energy curve (scan_point, energy) to stdout or CSV file
   - Also save full per-replica data (H, S, D, C, eigenvalues, charges, energy) to disk
   - Example usage: `cargo run --example scan -- --xyz data/xyz/h2.xyz --bond 0 1 --from 0.5 --to 3.0 --n 20`

3. **Write `examples/neb.rs` — nudged elastic band:**
   - Input: two endpoint XYZ files, number of images (default 20), spring constant, max iterations
   - Generate initial band: linear interpolation between endpoints
   - NEB iteration loop:
     a. Call `build_scc()` for each image → energies + forces (if Agent_3 available)
     b. If forces available: use real forces. If not: finite-difference forces from energy (central difference on image positions)
     c. Compute spring forces: F_spring = k·(R_{i+1} - 2·R_i + R_{i-1})
     d. NEB projection: F_perp = F_real - (F_real·τ)·τ; F_parallel = F_spring·τ; F_total = F_perp + F_parallel (τ = tangent)
     e. Update images: R_i += step·F_total (simple gradient descent or quick-min)
     f. Convergence: max |F_total| < tol or max_iter reached
   - Output: converged band (image positions + energies) to stdout or file
   - Example usage: `cargo run --example neb -- --start data/xyz/reactant.xyz --end data/xyz/product.xyz --images 20 --k 0.1 --maxiter 100`

4. **Write `tests/scan.rs`:**
   - `test_h2_bond_scan` — scan H2 bond 0.5–3.0 Å, 20 points, verify energy curve is smooth and physically reasonable (minimum near 0.74 Å, dissociation at large R)
   - `test_scan_saves_data` — verify per-replica data files are written and readable
   - Skip gracefully if no SK dir

5. **Run tests:**
   ```bash
   export RUST_DFTB_SK_DIR=/path/to/mio-1-1
   cargo test --test scan -- --nocapture
   cargo run --example scan -- --xyz data/xyz/h2.xyz --bond 0 1 --from 0.5 --to 3.0 --n 20
   ```

**Commands:**

```bash
export RUST_DFTB_SK_DIR=/path/to/mio-1-1
cargo test --test scan -- --nocapture
cargo run --example scan -- --xyz data/xyz/h2.xyz --bond 0 1 --from 0.5 --to 3.0 --n 20
```

**Expected deliverables:**
- `examples/scan.rs` — working rigid scan driver (CPU backend)
- `examples/neb.rs` — working NEB driver (energy-only if forces not ready)
- `tests/scan.rs` — tests passing, energy curve physically reasonable
- Sample energy curve output for H2 bond scan

## Handoff to coordinator/consumer

When finished, write your handoff report directly into the `## Agent reports` section at
the bottom of the **master** file (not this worker file). Check your checkbox `[ ]` → `[x]`
in the master's dispatch checklist. Your report must include:

1. Contract version and baseline used.
2. Changed files and concise rationale.
3. Exact commands plus full pass/fail results.
4. Artifact and `REVIEW:` paths (energy curve CSV, NEB band output).
5. Produced interface: example CLI usage, output file format.
6. Worst discrepancy, assumptions, unresolved risks, and requested coordinator edits.
7. **Note for coordinator:** the CPU backend call site is isolated in one function — swapping to `gpu_solve_scc_batched` later is a one-function replacement.

You MAY edit only your own checkbox and your own report in the master file. Do not edit
anything else in the master. Do not mark the aggregate task fixed/resolved/done.
