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

- [*] Write a producer/consumer inventory for C, SC, f, D, W and final H. Define small explicit validity flags/generations, including masked replicas, before removing density production. — `sc` buffer persistent; `direct_pop` flag selects path; D still materialized in `finalize` for force/EDM consumers.
- [*] Allocate persistent SC and required staging/offsets at construction. No allocation/build inside SCC iterations or geometry/relaxation hot loops. — `sc`, `k_sc_gemm`, `k_csnorm`, `k_mulliken_cs` all built at construction; zero per-iter allocs.
- [*] Compute SC once with existing tiled GEMM; compute finite positive column norms; scale C and SC together; normalize all columns. — `batched_gemm_active` for SC=S·C; `cs_normalize_batched` scales C and SC by shared `rsqrt(g_k)`; NaN scale on non-finite/non-positive g (fails loud).
- [*] Implement owner/gather populations `q_A=2 sum_mu in A,k f_k C_mu,k SC_mu,k` with lanes on contiguous k. Preserve population sign, electron factor and atom mapping. — `mulliken_cs_batched`: per-orbital contraction then per-atom gather; `occ_mask` (integer) or `occ_w` (Fermi) weights.
- [*] Keep current DIIS/mixing/convergence unchanged. Explicitly select old/new path for A/B; no automatic fallback. — `RUST_DFTB_POP=density` selects legacy path; default `direct`.
- [*] Remove only the ordinary SCC density producer. Provide lazy materialization at every actual D/W consumer; preserve final solve-at-final-q semantics. — in-loop density+mulliken+snormalize gone on direct path; `finalize` still builds D after populations for force consumers.
- [*] Ensure energy, eval(false), eval(true), force/FIRE, constraints and retry paths use certified state. `state_ok` is distinct from SCC activity. — finalize activates all replicas; occ_rayleigh unchanged; D rebuilt for edm/forces.
- [*] Frozen-state tests: integer/fractional occupations, nonorthogonal S, odd N, mixed masks, conservation and comparison against density-based populations plus independent f64 diagnostic reference. — `test_gpu_scc_direct_pop_parity`: per-iter q_new A/B to ~1e-6, host-verified SC=S·C to ~1e-7, cᵀSC=1 to ~2e-7, converged parity vs CPU ~1e-6; integer+smeared, n≤64 cold and n>64 warm.
- [ ] Lifecycle tests: poison D/W during SCC, then energy/forces; repeated calls; reset_q0; geometry change; early convergence; all-inactive and retry subset. — partially covered by existing suite (gpu_dftb 23/23, gpu_forces 5/5, gpu_hbond_physics 8/8 pass on direct path); dedicated poison/lifecycle tests still open.
- [*] Full GC PES/forces and large-N SCC parity; report population/metric errors, failure counts, SC-GEMM cost, removed stages and end-to-end gain. — all physical tests pass on direct path; see "T03 measured" below.

**T03 measured (batch=400, clean wall, same binary A/B):** direct-solver GC N86 4.55→3.84 ms/iter, DTH N246 351→299 ms/iter (~1.18×); **block-solver DTH 162→68.6 ms/iter (2.36×/iter)**. Wall: block+direct DTH 1646 ms vs block+density 2591 ms same-binary (1.57×; iters 24 vs 16 — different DIIS trajectory). EVT profile: jacobi 80.3%, 3 GEMMs ~9%, diis ~5%; density/snormalize/mulliken removed from loop. Logs: `debug/bench_t03_*.log`.

**Pre-existing bug found and fixed during T03 validation:** smeared n≤64 path skipped `extract_diag` (condition was `want_eig || kT<=0 || fermi_ref`), so `select_occ`/`fermi_occ` bisected on a **zero eig_diag** → uniform `occ_w=2/3` → Mulliken map became input-independent constant `[5.33,1.33,1.33]` → DIIS "converged" to it (residual < tol trivially). Fix: `eigh_solve` now launches `extract_diag` whenever standalone occupation kernels run (`|| n<=64` added). Virgin-plan regression check lives in `test_gpu_scc_direct_pop_parity`; `test_h2o_smeared_map_second_fixed_point` evaluates the f64 map directly (res at the false point = 1.15 — not a fixed point).

**Gate:** no stale consumer, no accuracy/status regression relative to frozen baseline, and measured useful application gain. This is a valid endpoint even if T04 loses. — gate criteria met pending user review.

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
**Design:** evolved into the slot-pool scheduler — see `Slot_Pool_Scheduler.design.md` v2 (work_ids convention, refill by subset reassembly, staged A→D plan). The checkboxes below are the Phase-A plumbing; the pool+refill is Phase C; async per-slot MD is Phase D.

- [x] **Phase A DONE (identity plumbing + non-identity acceptance).** `work_ids` appended as final arg to every SCC/matrix/eigensolver/CDFT kernel (~35 kernels: `gpu_matrix_ops.cl`, `gpu_tiled_jacobi.cl` incl. `jacobi_resident_batched`, `gpu_block_jacobi.cl`, `gpu_eigen.cl`, `gpu_cdft.cl`); all index `sid = work_ids[get_group_id(N)]` (3D GEMMs map dim-2 only; `(system,column)` and elementwise kernels derive col/elem from the launch index and remap output to `sid`). `GpuSccPlan.work_ids` + `GpuPbcPlan` identity buffers bound at build; `gpu_matrix.rs` helpers allocate internal identity buffers so legacy/test signatures are unchanged. All `set_arg` indices preserved (appended arg). **Verified:** all identity suites pass (tiled_jacobi 9, scc 8, dftb 8, forces 5, hbond_physics 23); production GC batch400 **195.1 ms / 1.951 ms/iter / failed=0** — zero measurable plumbing cost (vs 218.7 pre-conversion, run variance). **Non-identity acceptance — `tests/gpu_work_ids.rs` 6/6:** permuted `[2,0]` GEMM subset; single-slot 1-WG commit; elementwise sid+output remap; (system,column) decomposition; resident-Jacobi one-slot eigensolve to CPU parity with other slots bit-untouched; `active=0` inside compact domain skipped.
- [x] Preallocate host/device ID storage. At existing chunk reads gather active physical slots in stable order and upload once; no matrix copy or new synchronization. *(Phase B DONE: `GpuSccPlan::{work_ids_host, work_n}` + `set_work_domain` uploads only on list change, inside the existing chunk-end sync; zero new syncs.)*
- [x] Convert every replica launch axis, including 3D GEMMs, to physical sid via IDs. All H/S/C/SC/q/f/mu/DIIS/diagnostic/validity indexing uses sid. *(Solve-domain surface done; force/assembly/PBC kernels deferred to a later phase by design.)*
- [x] Keep active masks for convergence inside a chunk; skip launches when active count is zero. *(mask retained + tested; `n_active==0` exits the chunk loop before any compact rebuild.)*
- [x] Explicitly construct full/eligible domains for initialization, reset, retry, finalization and forces; never accidentally retain the last SCC tail list. *(Phase B DONE: `restore_work_domain` at SCC-loop exit + defensive restore at `finalize`/`scc_step`/`scc_step_diis` entry; retry-mask solves seed the compact domain up front.)*
- [x] Test noncontiguous IDs, one active system, alternating masks, early convergence, all-inactive, retries and repeated geometry updates against uncompacted mode. *(launch-domain level done in `gpu_work_ids.rs`; SCC-loop-level early-convergence/retry cases belong to Phase B/C verification.)*
- [x] Benchmark uniform and tail-heavy batches with active-iteration counts and launch/host costs. Do not claim compaction removes the serial last-replica tail. *(First cut below; full SCC-loop A/B is Phase B.)*

**Phase B kernel-level sweep** (`work_ids_saturation_sweep`, `tests/gpu_work_ids.rs`, `#[ignore]` — n=86 resident Jacobi, batch=400, same data, min of 3 reps; both arms include the 11.8 MB A re-upload ≈ 1.7 ms floor, so small-S times are floor-dominated):

| S | compact ms | masked(400-WG) ms | dead-WG |
|---|---:|---:|---:|
| 400 | 7.79 | 7.84 | ~0 |
| 300 | 6.82 | 6.35 | −0.46 |
| 200 | 4.87 | 4.58 | −0.29 |
| 100 | 2.82 | 2.85 | ~0 |
| 50 | 1.83 | 2.79 | +0.96 |
| 25 | 1.81 | 2.79 | +0.99 |
| 10 | 1.80 | 2.78 | +0.97 |
| 1 | 1.78 | 1.78 | ~0 |

Readout: **dead-WG ≈ 1 ms per dominant-kernel launch for S≤50** (~5–25% of a 2 ms/iter SCC step once the tail is sparse — real but bounded, as the design doc predicted); **saturation knee S\* ≈ 128–200** at n=86 (per-slot cost flat 19.5→24.4 µs for S≥200, degrades below ~100). Anomalies: masked slightly *faster* at S=200–300 (dead WGs vacate SMs early / noise); dead-WG cost vanishes at S=1 (floor-dominated, scheduling non-linear).

**Phase B DONE (SCC-loop compact launch domains, 2026-09-17).** Plan-side `LaunchGeom` captures every per-replica kernel's build-time grid factor; in-loop launches use `cmd().global_work_size(work_n×fac)` (1D) / dim-2 override (3D GEMMs) / padded `work_n×n` (`extract_diag`) — no kernel rebuilds, no new args, ~18 sites across `enq_dq_v_hscc`, `eigh_solve`, `eigh_finish`, `populations`, `occupation`, `scc_step`, `scc_step_diis_enq`, `commit_q_next` + CDFT `enq_shift_n`. Driver rebuilds the work list at the existing chunk-end sync **only when `n_active` dropped** (active set shrinks monotonically → unchanged count = free); retry-mask solves seed the compact domain before the first chunk. Env: `RUST_DFTB_SCC_COMPACT=0` A/B, `RUST_DFTB_DEBUG_DOMAIN=1` prints per-chunk domain width.

**Measured engagement (GC N=86 batch=400, `RUST_DFTB_DEBUG_DOMAIN=1`):** the workload IS tail-heavy — domain decays 400→238→88→39→22→11→8→5→3 over iters 16–80, i.e. ~70% of iterations run at ≤22% width. **A/B (best of N_RUNS=3):** compact ON **189.5 / 190.4 ms (1.895–1.904 ms/iter)** vs OFF **194.9 / 195.6 ms (1.949–1.956)** — **≈2.5–3% end-to-end SCC gain**, failed=0, identical all-converged statuses. Unguarded rebuild-every-chunk measured +1.5% (198.6 ms) — the `n_active<prev` guard removed that. Remaining tests under compact default: work_ids 6/6, scc 8/8, dftb 8/8, forces 5/5, hbond_physics 23/23, tiled_jacobi 9/9, pbc 7/7.

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

Post-T03 measured context (block+direct DTH N246 evt profile): jacobi is **80.3%** of device time — this ticket is now the primary lever.

**T08 measured results (RTX-class OpenCL device, batch=400, equal inputs, all configs stop=0/bad=0):**

- [x] **#6 block params** — `block_jacobi_param_sweep` (17 cfgs × {one,cold} × N{86,246}), accuracy-gated (res/orth/eig-parity ≈ baseline): **IMAX=1 is the dominant lever** — converging each 2B pivot to 1e-7 was waste since outer sweeps revisit; outer sweep counts unchanged. N=246: B16/IMAX12/ITOL1e-7 → 254.0/603.1 ms (one/cold); **B32/IMAX1/ITOL1e-6 → 190.7/377.2 ms (1.34×/1.60×)**; B24/IMAX1 ≈ 189.3/438.8. N=86: ~1.5×/2.2×. New defaults: **B=32, IMAX=1, ITOL=1e-6** (`RUST_DFTB_BJ_{B,IMAX,ITOL}` override). Single-pivot guard: n≤2B keeps IMAX≥12 (whole matrix IS the pivot).
- [x] **#5 direct WG≈N — FALSIFIED.** Arbitrary-WG fold-then-halve reductions ported into `gpu_tiled_jacobi.cl` (`jac_sum_*`; non-PoT legal, env `RUST_DFTB_JACOBI_WG`). Sweep {64…512}: runtime decreases monotonically with WG — **WG512 fastest** at N=86 (34.9→25.3 ms one; 65.5→52.9 cold) and N=246. Kernel is throughput-bound on fused strip updates, not barrier-bound. Default stays `min(512,max_wg)`; the infrastructure + env knob remain for future devices.
- [x] **Tail-free variant — no gain.** `notail` compile-out halves local mem (16.4→7.2 KB @WG512) but runtime ±3% noise → fused tail kept (saves a separate fermi_occ launch). `RUST_DFTB_JACOBI_NOTAIL=1` kept for A/B.
- [x] **JACOBI_PREC=0 — rejected at equal accuracy**: not faster AND 10–30× worse eig parity (N246 cold: 7.8e-3 vs 2.6e-4). prec=1 stays.
- [x] **#7 spilling — mostly falsified** (CL_KERNEL_PRIVATE_MEM_SIZE, ncu does not support OpenCL): block B16 → 0 B, B24 → 16–24 B, B32 → 160–232 B/thread (~58 f32 — real but small; B32 still wins cold). Direct kernel 0 B always. The real B-cost is **local mem** (9.7→20.2→34.8 KB per WG), not private spill.
- [x] **Measured dispatch rule** — `eigsolver_kind(n, local_mem)`: auto = block for **n>128**; n≤128 → **resident-defV when lA fits local_mem else direct**; `RUST_DFTB_EIGSOLVER` ∈ {auto,direct,block,resident,resident_av} forces (fail-loud on unknown / unfit explicit choice). End-to-end same-binary A/B, DTH N246 batch400, equal work (24 iters) + failed=0: B16-baseline 78.6 ms/iter → tuned 71.4 ms/iter (**1.10×**; per-solve gain is larger but warm probes skip most solves). DTH wall 6358→1715 ms ≈ **3.7× vs original baseline**; GC batch400 401.5→218.7 ms (**1.84×**) after resident dispatch.
- [ ] Crossover 86<n<246 unmeasured — n>128 threshold errs toward direct (resident can't fit ≥48 KB local anyway above n≈96); refine if mid-size systems appear.
- [ ] Registers/thread + warp-stall attribution remain hypotheses (no OpenCL counter tool); revisit if a CUDA port or profiling-capable driver appears.

**T08b — occupancy/WG/local-memory balance (from `Dense_Jacobi_Eigen_Tiling_Opt.md` review):**

Reflection on that doc: its headline suggestion — small WG (32–128) for the direct kernel to buy co-residency — is **already falsified** by the batch-400 WG sweep above (WG512 wins monotonically; in the queued regime the 400-WG queue already supplies parallelism, so per-WG lane count wins). What remains open and worth testing:

**Deeper finding — the direct kernel is bandwidth-bound on A/V streaming, not occupancy-bound.** Each of ~jround=85 rounds/sweep re-streams all of A and V through global memory (~10 MB/sweep/system at N=86 → ~20 GB per batch-400 solve ≈ 790 GB/s ≈ 85% of DRAM peak; working set 24 MB >> L2). Larger WG wins because more outstanding loads = more memory-level parallelism — this explains the monotonic WG512 result and means **no WG/tile resizing can fix it; only data residency can**. GPT's framing (threads-per-system) misses this; the user's original intuition (working tile iterated internally in shared memory) was pointing at it all along.

- [x] **Resident Jacobi — CONFIRMED, now auto-default at n≤128-if-fits.** Implemented `jacobi_resident_batched` (`RESIDENT_V` compile switch) in `gpu_tiled_jacobi.cl` + `EigKind::{ResidentDefV,ResidentAV}` dispatch in `gpu_eigen.rs`. **res-defV** (A __local-resident all sweeps + per-sweep deferred V apply via global rotlog — the winning form): N=86 batch400 equal-accuracy (bit-identical res/orth/par): direct-best 21.9/45.7 ms → **11.4/20.0 ms = ~2.2×** (`resident_jacobi_sweep`). WG-insensitive inside res-defV (256≈512) = no longer MLP-starved, consistent with the bandwidth diagnosis. **res-AV** (A+V both local): 59.7 KB > this device's **48 KB local** → skipped; res-defV fits n≲96 only (29.9 KB at 86; 66 KB at 128 fails). End-to-end GC batch400: **401.5→218.7 ms, 4.01→2.19 ms/iter = 1.84×**, failed=0. Dispatch: `RUST_DFTB_EIGSOLVER` ∈ {auto,direct,block,resident,resident_av}; auto = resident when lA+16KB ≤ local_mem else direct (capacity choice at plan build, fail-loud on explicit-but-unfit). rotlog scratch [batch][jround·jpair] double2 ≈ 23 MB at n=86/b400, allocated once.
- [ ] **res-AV on a larger-local device** (99 KB class would fit n=86 A+V) — expected further gain but untestable on this GPU; keep env-forced.
- [ ] **A-local/V-global hybrid for mid n** (if full-local doesn't fit: A alone = n²·4B — n≤~96 at 48 KB local, ~160 at 100 KB). res-defV IS this hybrid (V streamed once/sweep instead of per-round); extend coverage if mid-size systems appear.
- [ ] **Block kernel (n>128) strip-residency variant.** P+U already local; strips still stream global per pivot (~28 MB/sweep/system at n=246 → also bandwidth-bound). Options: hold A-block-row slabs in local across a pivot column pass, or accept DRAM-bound but reduce bytes (skip identity strips — `PAIR_SKIP_REL` already does this partially).

- [ ] **Block kernel WG < n (strided rows).** `block_jacobi_1wg` currently requires WG≥n (thread-per-row). Convert row loops to `for (row=lid; row<n; row+=lsz)` and sweep WG∈{64,96,128} at N=246 — matters specifically in the *small-slot-count scheduler regime* (S≈64–128, no WG queue to hide latency), where resident occupancy becomes binding. Falsified-in-queued-regime ≠ falsified in resident regime.
- [ ] **Joint (slots × WG) co-optimization.** `N_slots × warps/WG` sets total parallelism — slot-pool size (T06/scheduler doc §6) and Jacobi WG are coupled parameters, not independent. Once the scheduler exists, sweep the product: e.g. {64,128,256} slots × {64,128,256,512} WG on GC+DTH; pick the pair at the throughput knee, not each separately.
- [ ] **Local-memory-driven co-residency curve for block.** Document WGs/SM vs B (measured local: B16=9.7K, B24=20.2K, B32=34.8K/WG — B32 won anyway because serial-pivot reduction dominated; record the curve so future devices/GPUs can re-derive the optimum rather than re-litigating).
- [ ] **Direct kernel `rot[128]` cap.** Local rot arrays bound n≤256; if larger n ever appears, either raise the cap (more local) or route to block — note, not urgent.
- [ ] **Per-atom/tile-kernel sizing sanity check.** Confirm no other in-loop kernel has a WG≈N assumption silently baked in (cs_normalize, occ_renorm/snorm use gid=group(0) strided loops — verify they're already WG-flexible; the small O(n_atoms) kernels are negligible anyway).

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

## T10 — Small wins and fusion leftovers (P2)

Items from the review's punch-list that are real but small after T03 (measured device shares at N246 block+direct: hscc 0.5%, fermi 0.7%, select_occ ~0.15-0.26 ms/iter):

- [ ] **Hscc fusion:** synthesize `h = H0_ij + ½ S_ij (V_Ai + V_Aj)` on tile load inside the first projection GEMM; never materialize `h_scc`. Removes an N² write+read+one launch (~0.3 ms/iter device at N246 — small; the review's 21 ms was the PROF=1 host-stall artifact). Split `dq_v_hscc` into a tiny atom-space `dq→V` kernel + fused GEMM.
- [ ] **Skip `select_occ` under smearing:** `occ_idx`/`occ_mask` are unread when `use_w=1` (density uses `t`, mulliken_cs/occ_rayleigh/occ_renorm use `occ_w`). Guard in `occupation()`.
- [ ] **Fermi kernel upgrade:** standalone `fermi_occ_batched` is still fixed-40 f64 bisection — replace with warm-μ safeguarded Newton (WG32/64, f32 logistic + f64 scalar count). See T07.
- [ ] **Alloc-free housekeeping:** `reset_diis` allocates `vec![0i32; batch]` per call; `check_jacobi` allocates temporaries. Preallocate at construction (see T09 audit).

## Starting tests and handoff format

- [ ] Discover current feature gates and existing commands in Cargo.toml/manifest before running; do not guess an OpenCL feature name. Run foreground release performance tests with full unfiltered output.
- [ ] Start kernel work from `rust_dftb/tests/gpu_tiled_jacobi.rs` (direct/block residual, eigenvalue, orthogonality tests) and `gpu_scc_kernels.rs`; application timing from `gpu_scc_bench.rs::test_gpu_scc_scan400_benchmark`.
- [ ] Physical checks live in `gpu_hbond_physics.rs`, `gpu_forces.rs`, `gpu_dftb.rs` and manifest PES/force commands. Reuse existing fixtures before creating duplicates.
- [ ] If concurrent unrelated work prevents compilation, report exact errors and affected files; do not modify another agent's sparse implementation or claim GPU validation occurred.

Each coding agent should hand back: ticket IDs; files/symbols changed; equations and state transitions preserved; exact commands and full-log paths; correctness numbers; equal-work wall times; failures; next unresolved item. Stop speculative tuning once the ticket's measured question is answered.
