---
type: Task
title: GPU Multi-System DFTB — Agent_03 DFTB forces (CPU)
tags: [parallel-agents, worker-task, forces, dftb]
---

# Agent_03: DFTB forces (CPU)

- **Master:** [`task_master.md`](task_master.md)
- **Agent ID:** `Agent_3`
- **Required contract version:** 1
- **Status authority:** coordinator only

Read the master completely before working. The master overrides this file. Execute
only this packet; other worker scopes are context, not optional work.

## Goal and boundary

- **Goal:** Implement DFTB forces (non-SCC + SCC) on CPU, achieving parity with Fortran DFTB+ `PrintForces` output. Forces are needed for relaxed scan and NEB spring-force projection.
- **In scope:** `methods/dftb/forces.rs` (new module), `tests/parity_forces.rs` (new test), `tests/run_forces.py` (new Python driver). Finite-difference dH0/dx, dS/dx; repulsive spline; gamma derivative; SCC double-counting + shift forces.
- **Out of scope:** GPU work (Agent_1/4). QM/QM multi-fragment (Agent_2). Modifying `hamiltonian.rs` (report if `SccResult` needs new fields).

## Inputs and preconditions

- Frozen inputs/fixtures:
  - `RUST_DFTB_SK_DIR` env var pointing to mio-1-1 SK files
  - Fortran DFTB+ binary (path in `tests/run_forces.py`)
  - Test molecules with non-zero forces (distorted geometries): H2O (bent), HCN (linear asymmetric), HCOOH
  - Design doc: `doc/prokop/DFTB_Reimplementation_Progress/Forces_Implementation_Notes.md` (full force decomposition, Fortran file map, implementation plan)
- Upstream gate: none (Wave 1, independent)
- Interfaces consumed:
  - `methods/dftb/hamiltonian.rs::HamiltonianBuilder::build_non_scc()` — H0, S, density matrix
  - `methods/dftb/hamiltonian.rs::HamiltonianBuilder::build_scc()` — converged SCC result
  - `methods/dftb/hamiltonian.rs::SccResult` — eigenvalues, eigenvectors, charges, energy
  - `methods/dftb/rotation.rs::Rotation::rotate_diatomic_block_into()` — for finite-diff dH0/dx
  - `methods/dftb/interpolation.rs::EqGridTable::eval_into()` — for finite-diff dH0/dx
  - `methods/dftb/gamma.rs::gamma_full()` — for SCC double-counting force
  - `methods/dftb/sk_data.rs` — for repulsive spline parsing (may need extension)

## Exclusive ownership

- **May write:** `methods/dftb/forces.rs` (new), `tests/parity_forces.rs` (new), `tests/run_forces.py` (new)
- **Read-only:** all other `src/` files
- **Must not:** edit `hamiltonian.rs` (if `SccResult` needs a `density` field or `forces` method, stop and request coordinator edit). Edit `gamma.rs` (if `gamma_prime_full` is needed, request coordinator edit or implement in `forces.rs` as a local function). Edit `sk_data.rs` (if spline parsing is needed, request coordinator edit). Edit `lib.rs`, `mod.rs`.

If any required edit falls outside ownership, stop and propose it to the coordinator.

## Work and verification

1. **Read the design doc:** `doc/prokop/DFTB_Reimplementation_Progress/Forces_Implementation_Notes.md` — it contains the complete force decomposition, Fortran source file map, and implementation plan. Follow the Phase 1 (non-SCC) → Phase 2 (SCC) plan.

2. **Phase 1 — Non-SCC forces:**
   - Compute density matrix DM = 2·C_occ·C_occ^T from `build_non_scc()` result
   - Compute energy-weighted density matrix EDM = 2·C_occ·diag(eps_occ)·C_occ^T
   - Implement finite-difference dH0/dx, dS/dx: displace atom j by ±delta (delta = f64::EPSILON^0.25 ≈ 1.2e-4) along each axis, rebuild diatomic block via `rotate_diatomic_block_into` + `eval_into`, central difference
   - Parse repulsive spline from SK file (if `sk_data.rs` doesn't expose it, implement parsing in `forces.rs` as a local function and request coordinator to move it later)
   - F_nonSCC = 2·(DM·dH0' − EDM·dS') per pair; F_rep = dE_rep/dr · r_hat
   - Assemble total non-SCC force per atom

3. **Phase 2 — SCC forces:**
   - Implement `gamma_prime_full(r, U1, U2)` (derivative of `gamma_full`) — mirror Fortran `shortgammafuncs.F90::expGammaPrime`. If same-U and different-U cases, handle both.
   - F_SCC_dc = −dQ_i·dQ_j·gamma'(r)·r_hat (gamma + 1/R Coulomb part)
   - F_SCC_shift = 2·shiftSprime·DM (Pulay-like; needs block-resolved shifts — if `shifts.rs` doesn't expose them, use atom-resolved as approximation and document)
   - Total SCC force = F_nonSCC + F_SCC_shift + F_SCC_dc + F_rep

4. **Write `tests/run_forces.py`:**
   - Generate DFTB+ HSD input with `SCC = No` (or `Yes`), `PrintForces = Yes`
   - Run Fortran DFTB+, parse "Total Forces" from `detailed.out`
   - Run Rust force computation
   - Compare and report max abs diff

5. **Write `tests/parity_forces.rs`:**
   - `non_scc_forces_from_xyz` — env-driven: `RUST_DFTB_FORCES_XYZ`, `RUST_DFTB_FORCES_REF`
   - `scc_forces_from_xyz` — same but with SCC
   - Compare force vectors, tolerance 1e-5

6. **Run tests:**
   ```bash
   export RUST_DFTB_SK_DIR=/path/to/mio-1-1
   python3 tests/run_forces.py data/xyz/h2o.xyz --no-scc
   cargo test --test parity_forces -- --nocapture
   ```

**Commands:**

```bash
export RUST_DFTB_SK_DIR=/path/to/mio-1-1
python3 tests/run_forces.py data/xyz/h2o.xyz --no-scc
python3 tests/run_forces.py data/xyz/h2o.xyz --scc
cargo test --test parity_forces -- --nocapture
```

**Expected deliverables:**
- `methods/dftb/forces.rs` — complete force computation (non-SCC + SCC)
- `tests/parity_forces.rs` — parity tests passing
- `tests/run_forces.py` — Python driver
- Force parity < 1e-5 vs Fortran for H2O, HCN, HCOOH (distorted geometries)

## Handoff to coordinator/consumer

When finished, write your handoff report directly into the `## Agent reports` section at
the bottom of the **master** file (not this worker file). Check your checkbox `[ ]` → `[x]`
in the master's dispatch checklist. Your report must include:

1. Contract version and baseline used.
2. Changed files and concise rationale.
3. Exact commands plus full pass/fail results.
4. Artifact and `REVIEW:` paths.
5. Produced interface: `forces.rs` public API (function signatures for Agent_5 to use).
6. Worst discrepancy, assumptions, unresolved risks, and requested coordinator edits.
7. **If `SccResult` or `sk_data.rs` or `gamma.rs` needed changes:** list exact requested edits for coordinator.

You MAY edit only your own checkbox and your own report in the master file. Do not edit
anything else in the master. Do not mark the aggregate task fixed/resolved/done.
