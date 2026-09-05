---
type: Task
title: GPU Multi-System DFTB — Agent_02 CPU multi-fragment validation
tags: [parallel-agents, worker-task, qmqm, testing]
---

# Agent_02: CPU multi-fragment validation (correctness oracle)

- **Master:** [`task_master.md`](task_master.md)
- **Agent ID:** `Agent_2`
- **Required contract version:** 1
- **Status authority:** coordinator only

Read the master completely before working. The master overrides this file. Execute
only this packet; other worker scopes are context, not optional work.

## Goal and boundary

- **Goal:** Write tests that exercise the `MultiSystemSolver` with **2+ fragments**, validating inter-fragment electrostatic coupling (`compute_v_ext`), charge conservation, and polarization convergence. This is the correctness oracle for the GPU QM/QM path.
- **In scope:** Append new test functions to `tests/qmqm_integration.rs`. Read and analyze `qmqm/solver.rs`, `qmqm/fragment.rs`, `qmqm/neighbor.rs` to understand the API.
- **Out of scope:** Modifying any `src/` file (unless a bug is found — then report, don't fix). GPU work (Agent_1/4). Forces (Agent_3).

## Inputs and preconditions

- Frozen inputs/fixtures:
  - `RUST_DFTB_SK_DIR` env var pointing to mio-1-1 SK files
  - Test molecules: H2O (3 atoms), two H2O at varying separation
  - CPU reference: single-fragment `HamiltonianBuilder::build_scc()` (already parity-verified)
- Upstream gate: none (Wave 1, independent)
- Interfaces consumed:
  - `qmqm/solver.rs::MultiSystemSolver::new(fragments, neighbors, gamma, mixer)` — multi-frag solver
  - `qmqm/solver.rs::MultiSystemSolver::solve_scc(max_iter, tol)` — SCC convergence
  - `qmqm/solver.rs::MultiSystemSolver::compute_v_ext()` — inter-fragment polarization
  - `qmqm/fragment.rs::Fragment::from_template()` — fragment construction
  - `qmqm/neighbor.rs::FragmentNeighborList::build()` — fragment neighbor list
  - `qmqm/gamma.rs::GammaTable::from_sk_data()` — gamma table

## Exclusive ownership

- **May write:** `tests/qmqm_integration.rs` (append new test functions only — do not modify existing tests)
- **Read-only:** all `src/` files
- **Must not:** edit any `src/` file. If a bug is found in `solver.rs` or `fragment.rs`, stop and report it in your handoff. Do not fix it yourself. Do not edit `lib.rs` or `mod.rs`.

If any required edit falls outside ownership, stop and propose it to the coordinator.

## Work and verification

1. **Study existing tests:** Read `tests/qmqm_integration.rs` — note that ALL existing tests use `vec![frag]` (single fragment). Read `qmqm/solver.rs::compute_v_ext` and `solve_scc` to understand the inter-fragment coupling code path.

2. **Write `two_fragment_independent_scc` test:**
   - Two H2O molecules, far apart (e.g. 20 Å between centroids)
   - At large separation, inter-fragment coupling ≈ 0
   - Each fragment should converge to the same charges as a standalone H2O SCC
   - Verify: both fragments converge, charges match standalone within 1e-6

3. **Write `two_fragment_polarization` test:**
   - Two H2O molecules, close together (e.g. 3 Å between centroids, O···H hydrogen-bond geometry)
   - Inter-fragment coupling is active
   - Verify: both fragments converge, total charge conserved (sum of all dQ = 0 within 1e-10)
   - Verify: charges differ from standalone (polarization occurred)
   - Compare total energy vs two standalone H2O (should differ by inter-fragment interaction energy)

4. **Write `charge_conservation_multi_frag` test:**
   - 3× H2O at various separations
   - Verify sum of all atomic charges = sum of q0 (neutral) within 1e-10 at every SCC iteration
   - This is a critical invariant — if it fails, there's a bug in `compute_v_ext` or `gather/scatter_charges`

5. **Write `two_fragment_vs_single_combined` test (optional, if feasible):**
   - Two H2O close together as 2 fragments via `MultiSystemSolver`
   - Same geometry as 1 fragment (all 6 atoms) via `HamiltonianBuilder::build_scc()`
   - The QM/QM approximation should give similar but not identical results
   - Document the difference (this is the QM/QM approximation error)

6. **Run tests:**
   ```bash
   export RUST_DFTB_SK_DIR=/path/to/mio-1-1
   cargo test --test qmqm_integration two_fragment -- --nocapture
   cargo test --test qmqm_integration charge_conservation -- --nocapture
   ```

**Commands:**

```bash
export RUST_DFTB_SK_DIR=/path/to/mio-1-1
cargo test --test qmqm_integration two_fragment -- --nocapture
cargo test --test qmqm_integration charge_conservation -- --nocapture
```

**Expected deliverables:**
- 3–4 new test functions in `tests/qmqm_integration.rs`, all passing
- Evidence that `compute_v_ext` works correctly for 2+ fragments
- Evidence that charge conservation holds across fragments
- If bugs found in `solver.rs`, documented in handoff (not fixed)

## Handoff to coordinator/consumer

When finished, write your handoff report directly into the `## Agent reports` section at
the bottom of the **master** file (not this worker file). Check your checkbox `[ ]` → `[x]`
in the master's dispatch checklist. Your report must include:

1. Contract version and baseline used.
2. Changed files and concise rationale.
3. Exact commands plus full pass/fail results.
4. Artifact and `REVIEW:` paths.
5. Produced interface/schema and downstream usage notes.
6. Worst discrepancy, assumptions, unresolved risks, and requested coordinator edits.
7. **If bugs found in `solver.rs`/`fragment.rs`:** describe the bug, the failing assertion, and the suspected root cause. Do not fix it — request coordinator edit.

You MAY edit only your own checkbox and your own report in the master file. Do not edit
anything else in the master. Do not mark the aggregate task fixed/resolved/done.
