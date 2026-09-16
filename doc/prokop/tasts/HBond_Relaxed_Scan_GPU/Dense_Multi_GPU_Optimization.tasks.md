# Dense multi-system GPU optimization — implementation queue

Created 2026-09-16 from the [source-review appendix](Dense_Multi_GPU_Optimization.chat.md#2026-09-16--source-review-corrections-and-implementation-design). This is a specification, not a claim of implemented work. `[ ]` pending; `[*]` evidence gathered. Keep code changes small and report measured results; user review determines acceptance.

## Dispatch and ownership

- [*] Source review identified a WG96 block-reduction defect and a single-pivot false-success path.
- [*] Direct Mulliken populations from C/SC are algebraically justified; transported SC requires an additional f32 consistency invariant.
- [ ] Assign T01 to a correctness agent and T02 to a measurement agent; these can proceed independently with separate file ownership.
- [ ] Give one orchestration owner T03/T04/T06 because they share `gpu_scc_plan.rs` and `gpu_dftb.rs`. Avoid simultaneous broad edits to these files.
- [ ] A matrix-kernel agent can prepare T05 after the T03 validity contract is agreed. T07 must coordinate Jacobi renderer changes with T01/T04.
- [ ] Dependency order: T01+T02 -> T03 -> T04 and T05; T06 after T03; T07 after T01/T02; T08 after corrected, comparable baselines. T09 audits each stage and gates promotion.
- [ ] Read repository GUIDELINES §§2–5/7/10, efficiency guidance, manifest §14/§16.G and this review before editing. Preserve concurrent sparse work.

## T01 — Correct block reductions and solver status (P0)

**Files:** `rust_dftb/src/qmqm/gpu_block_jacobi.cl`, `gpu_eigen.rs`, `gpu_scc_plan.rs`; `rust_dftb/tests/gpu_tiled_jacobi.rs`.

- [*] Add deterministic tests first: all-ones/impulse reductions for WG32/64/96/128/160/192/224/256/512 where supported; N86 identity plus off-diagonal(2,5)=0.01. — `test_block_jacobi_wg96_impulse_reduction`, `test_block_jacobi_probe_counts_all_rows` in `gpu_tiled_jacobi.rs`; both failed as predicted pre-fix (probe undercounted ~18% at WG96, impulse missed entirely).
- [*] Correct every block norm reduction, including initial probe, pivot and sweep norms. A shared arbitrary-size helper is preferable to multiple divergent fixes; keep barriers uniform. — shared `bj_wg_sum` helper used at all 6 reduction sites; called uniformly by all lanes.
- [*] Test odd N and partial pivots around B/PB/WG boundaries. Ensure padded values do not enter physical norms and inactive lanes contribute zero. — probe exactness verified for n ∈ {24,33,48,64,86,87,96,128,160,192,224,225,246,256}; lanes `lid>=n` contribute 0; pivot sums guard `i<m`.
- [*] For `n<=PB`, require actual residual success, not merely finiteness. Test deliberately exhausted inner caps and nonfinite input. — `test_block_jacobi_inner_cap_reports_failure` (INNER_MAX=1 → stop=2, not 0); NaN input → stop=4.
- [*] Audit direct/host stop acceptance and Fermi freshness. Accepted floor results must have valid occupations; rejected results must not feed subsequent SCC unnoticed. — direct kernel Fermi tail now runs for all certifiable stops (≤2), not just 0; `check_jacobi` acceptance (stop=4 or rel>1e-4 → uncertified) unchanged.
- [*] Ensure first failure survives later iterations until reporting, with replica, iteration, residual and stop reason. — diag records latched in both kernels (`stop==0` overwrites only non-failed records); host clears records at each logical solve window via `clear_jacobi_diag`.
- [*] Re-run direct/block eigen residual, eigenvalue and orthogonality tests, then GC/AT/azaindole and DTH SCC status/physical checks. — gpu_tiled_jacobi 8/8, gpu_eigenproblem 10/10, gpu_scc 6/6, gpu_scc_kernels 10/10, gpu_dftb 8/8, gpu_forces 5/5, gpu_hbond_physics 23/23. gpu_diagonalization has 2 pre-existing N2 eigenvector failures on HEAD (degenerate-subspace elementwise compare; files in that path untouched).
- [*] Report old/new true versus reported norms and convergence counts. Do not compare speed at unequal error or hide new failures by increasing caps/tolerances without diagnosis. — pre-fix: impulse reported off=0 vs true 1.41e-2; probe 5.34e-6 vs true 6.52e-6 (18% drop); INNER_MAX=1 stop=0 at rel=0.376. Post-fix: exact probe norms; honest stop codes; see report section "T01/T02 measured".

**Gate:** reductions include every lane; status agrees with independent residual; no stale occupation on any accepted path. Keep previous timings as historical data. — gate criteria met; awaits user review.

## T02 — Reproducible profiling and fair benchmark (P0)

**Files:** `gpu_runtime.rs`, diagnostic portions of `gpu_scc_plan.rs`, `rust_dftb/tests/gpu_scc_bench.rs`.

- [ ] Record source hashes, device/driver, effective compile macros, matrix/batch sizes, smearing, tolerances, CPU thread env and construction/warm-start boundaries.
- [ ] Add explicit profile-total clearing/regions; distinguish marker spans from actual command START/END durations. Read event timestamps only at existing completion points.
- [ ] Gather cumulative per-replica sweep histogram, zero-sweep fraction, stop reasons and sum(active replica-iterations), using owner counters without global atomics.
- [ ] Separate cold solve, repeated charge reset/reused basis, and nearby geometry warm starts. Report each run's iteration/status counts alongside its own time.
- [ ] Use batch1/100/400 plus a small saturation sweep only as needed. Time construction separately, clean SCC wall and SCC+final energy/forces separately.
- [*] Compare direct/block on identical snapshots and numerical settings. Run equal-work CPU references on the same scan, or explicitly label sampled/extrapolated speedups. — `test_gpu_scc_scan400_benchmark` GC+DTH, batch 1/100/400, identical geometries/kT/tol; CPU f64 single-point baseline measured per system.
- [*] Report successful versus requested throughput, failures, median/range over repeated equivalent runs, total launches and host finishes. Do not silently average failures into throughput. — failures reported per row (block GC b400 had failed=1, ran to 100-iter cap vs direct 80).
- [*] Preserve full logs in `debug/`; append a measured-results section to the existing report/manifest without rewriting history. — `debug/bench_t01_direct.log`, `debug/bench_t01_block.log`; results appended to `doc/prokop/reports/2026-09-16_dense_gpu_pes_forces_benchmark_UPDATED.md`.

**Gate:** another agent can reproduce the configuration and distinguish execution cost, idle gaps, different iteration counts and different success rates.

## T03 — Recomputed SC and direct populations, including cache validity (P1)

**Files:** `gpu_matrix_ops.cl`, `gpu_scc_plan.rs`, `gpu_dftb.rs`; extend kernel/public lifecycle tests. Reference CPU `methods/dftb/dftb_cpu.rs` and Fortran `populations.F90`.

- [ ] Write a producer/consumer inventory for C, SC, f, D, W and final H. Define small explicit validity flags/generations, including masked replicas, before removing density production.
- [ ] Allocate persistent SC and required staging/offsets at construction. No allocation/build inside SCC iterations or geometry/relaxation hot loops.
- [ ] Compute SC once with existing tiled GEMM; compute finite positive column norms; scale C and SC together; normalize all columns.
- [ ] Implement owner/gather populations `q_A=2 sum_mu in A,k f_k C_mu,k SC_mu,k` with lanes on contiguous k. Preserve population sign, electron factor and atom mapping.
- [ ] Keep current DIIS/mixing/convergence unchanged. Explicitly select old/new path for A/B; no automatic fallback.
- [ ] Remove only the ordinary SCC density producer. Provide lazy materialization at every actual D/W consumer; preserve final solve-at-final-q semantics.
- [ ] Ensure energy, eval(false), eval(true), force/FIRE, constraints and retry paths use certified state. `state_ok` is distinct from SCC activity.
- [ ] Frozen-state tests: integer/fractional occupations, nonorthogonal S, odd N, mixed masks, conservation and comparison against density-based populations plus independent f64 diagnostic reference.
- [ ] Lifecycle tests: poison D/W during SCC, then energy/forces; repeated calls; reset_q0; geometry change; early convergence; all-inactive and retry subset.
- [ ] Full GC PES/forces and large-N SCC parity; report population/metric errors, failure counts, SC-GEMM cost, removed stages and end-to-end gain.

**Gate:** no stale consumer, no accuracy/status regression relative to frozen baseline, and measured useful application gain. This is a valid endpoint even if T04 loses.

## T04 — Carry SC through Jacobi with enforced refresh (P1 experiment)

**Depends on:** T03. **Files:** both Jacobi kernels, render/build/plan interfaces and basis repair.

- [ ] Specify valid-SC initialization after cold AO basis formation. Refresh after overlap changes and after repairs unless the same transformation is explicitly applied to SC.
- [ ] Add paired right-transform updates for C/SC to direct and block kernels, including zero-sweep/inactive paths. Control private-array lifetimes to avoid unnecessary register growth.
- [ ] Retain cheap paired normalization every iteration initially. Invalid norms fail contextually, never clamp to conceal divergence.
- [ ] Independently measure `SC-S*C` and full `C^T S C-I`, not only diagonals computed from carried SC. Add long-trajectory and repeated geometry-update stress tests.
- [ ] Define and enforce refresh/repair thresholds or a validated bounded cadence from measured f32 floor; account for refresh time. Do not certify carried SC using itself.
- [ ] Compare recompute versus carry at zero/few/many sweeps, N86/N246 and several batches; include memory, kernel resources, SCC iterations, PES/forces and failures.
- [ ] Select carry only where total eigen+SC+normalization+population and application time improve. Keep selection explicit and measured.

**Gate:** stable true SC/metric invariant over intended trajectories and a gain over T03, not merely over the old density pipeline.

## T05 — Exact tiled final density and EDM (P1)

**Depends on:** T03 lifecycle. **Files:** `gpu_matrix_ops.cl`, finalization in `gpu_scc_plan.rs`.

- [ ] Implement weighted rank-k products for D and W using all current occupation weights. W uses signed `2f*rho`; no square root of negative weights.
- [ ] Tile output AO pairs and k; assign unique triangle/mirror ownership, especially inside diagonal tiles. No global atomics.
- [ ] Compare one fused D/W kernel with separate kernels: saved C loads versus extra registers. Keep bulk accumulation f32.
- [ ] Preserve final-H Rayleigh and energy entropy conventions and correct validity/masks for early-converged/failed systems.
- [ ] Test symmetry, trace/population conservation, negative rho, fractional occupations, ragged tiles, independent dense reference and unchanged energy/forces/finite differences.
- [ ] Time final evaluation separately and in SCC+forces/FIRE workloads. Small finalization share may make this lower priority than T06/T07.

**Gate:** exact formulation and baseline physical accuracy; no speculative occupation truncation. Verify DTH's actual q0/nocc before proposing sparse occupation work.

## T06 — Compact active launch IDs (P1)

**Depends on:** T03 state contract. **Files:** plan, affected kernels and chunk scheduling in `gpu_dftb.rs`.

- [ ] Preallocate host/device ID storage. At existing chunk reads gather active physical slots in stable order and upload once; no matrix copy or new synchronization.
- [ ] Convert every replica launch axis, including 3D GEMMs, to physical sid via IDs. All H/S/C/SC/q/f/mu/DIIS/diagnostic/validity indexing uses sid.
- [ ] Keep active masks for convergence inside a chunk; skip launches when active count is zero.
- [ ] Explicitly construct full/eligible domains for initialization, reset, retry, finalization and forces; never accidentally retain the last SCC tail list.
- [ ] Test noncontiguous IDs, one active system, alternating masks, early convergence, all-inactive, retries and repeated geometry updates against uncompacted mode.
- [ ] Benchmark uniform and tail-heavy batches with active-iteration counts and launch/host costs. Do not claim compaction removes the serial last-replica tail.

**Gate:** identical per-input outputs/statuses with reduced measured scheduling cost. Slot refill is a later ticket requiring per-slot warm/cold lifecycle and complete reset contracts.

## T07 — Fermi correctness, common implementation and fusion experiment (P1)

**Depends on:** T01/T02. **Files:** Fermi support/direct tail, renderers and plan dispatch.

- [ ] Consolidate effective solver parameters so standalone and production use the same intended tolerances/macros; log compiled settings.
- [ ] Use shared safeguarded Fermi mathematics with persistent mu seed, bracket, finite checks, iteration cap and explicit failure. Validate stored f32 occupation sums; preserve cheap f64 scalar decisions as needed.
- [ ] Audit removal of sorted occupation selection in fractional mode; prove all downstream mask readers are initialized or bypassed.
- [ ] Build a true tail-free direct variant and compare separate Fermi against fused. Runtime disabling alone may not remove scratch resources.
- [ ] Test WG32/64/128 with correct reductions and strided orbital loops; include degenerate/small-gap/wide-spectrum, near-empty/full and temperature extremes.
- [ ] Check occupations, electron count, entropy, SCC trajectory and final forces; measure total SCC time and actual compiled resources.

**Gate:** no stale/capped-but-successful occupations; an explicit measured choice of fusion. The newer experiment can supersede historical fused-tail preference without changing physics invariants.

## T08 — Measured direct/block kernel tuning (P2)

**Depends on:** T01/T02 and stable support path. **Files:** Jacobi kernels/renderers/tests only unless interface change is agreed.

- [ ] Save exact per-kernel local/private/max-WG queries; reconcile discrepancies against source/options. Do not equate private bytes with registers or memory requests with DRAM traffic.
- [ ] Run a small targeted WG sweep after arbitrary-size reduction tests pass. Benchmark B16/B24 with partial pivots and PB48 schedule validation.
- [ ] Separate probe, actual rotations and Fermi attribution; use full sweep distributions. One WG per system does not imply one WG per SM.
- [ ] Measure cold and warm workloads at N84/86/87/120/246 and batches spanning underfill/saturation. Add sizes only where needed to locate a crossover.
- [ ] Use supported hardware counters if available; otherwise clearly label occupancy/bandwidth explanations as hypotheses.
- [ ] Choose a simple measured construction-time dispatch rule; no silent runtime recovery. Report total SCC/forces gain and numerical/failure parity.

**Gate:** improvement at equal accuracy/work with documented resource explanation; no crossover inferred solely from two molecules.

## T09 — Allocation, synchronization and scientific promotion gates (continuous)

- [ ] Extend the H2O/batch1 allocation test to N86/N246, multi-chunk solves, mixed masks, retries and geometry updates. Count device buffers, kernel objects and program builds.
- [ ] Audit host vectors/clones in `scc_mix_inner`, `reset_diis`, status checking, geometry paths and diagnostics; reuse capacity. Isolate host-allocation measurements and inspect library internals.
- [ ] Distinguish solver scratch from public owned-result construction. Use internal into-buffer/view interfaces if necessary, not a broad API rewrite.
- [ ] Consolidate status reads to one existing-boundary finish where dependency/lifetime checks permit. Keep profiler metadata bounded and quantify its overhead.
- [ ] Freeze numerical budgets from measured reference floors before each change. Do not loosen smoke tolerances to accept degraded PES/forces.
- [ ] Verify L0 focused Rust/kernel tests, L1 full unfiltered logs, L2 plotted PES/force/relaxation diagnostics for human review. Store artifacts only under debug/.
- [ ] Final report: exact change/configuration, max/RMS errors with units and worst case, convergence/failures, cold/warm SCC and SCC+forces wall times, active iterations, allocations/builds/syncs, remaining uncertainty.
- [ ] Update report/manifest and roadmap only with measured implementation status; user review precedes declaring the solver fixed/accepted.

## Starting tests and handoff format

- [ ] Discover current feature gates and existing commands in Cargo.toml/manifest before running; do not guess an OpenCL feature name. Run foreground release performance tests with full unfiltered output.
- [ ] Start kernel work from `rust_dftb/tests/gpu_tiled_jacobi.rs` (direct/block residual, eigenvalue, orthogonality tests) and `gpu_scc_kernels.rs`; application timing from `gpu_scc_bench.rs::test_gpu_scc_scan400_benchmark`.
- [ ] Physical checks live in `gpu_hbond_physics.rs`, `gpu_forces.rs`, `gpu_dftb.rs` and manifest PES/force commands. Reuse existing fixtures before creating duplicates.
- [ ] If concurrent unrelated work prevents compilation, report exact errors and affected files; do not modify another agent's sparse implementation or claim GPU validation occurred.

Each coding agent should hand back: ticket IDs; files/symbols changed; equations and state transitions preserved; exact commands and full-log paths; correctness numbers; equal-work wall times; failures; next unresolved item. Stop speculative tuning once the ticket's measured question is answered.
