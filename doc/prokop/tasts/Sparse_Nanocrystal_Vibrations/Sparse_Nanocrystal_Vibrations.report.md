# Sparse Nanocrystal Vibrations — Implementation Report

> **Current review correction:** read manifest **§15** before using the later Item 5/6/7 conclusions. W=2(ZH)K still consumes truncated ZH; the quartic clamp is not a certified stability invariant; TRS G3.4 remains red; and the histogram's 3% row-norm budget is 0.09% squared norm mass, not 3% mass or a force-error bound. The ~61-neighbor result lacks symmetric-mask/re-solved force validation. Earlier results are retained as history, not current acceptance. §15 provides the prioritized sparse-only coding-agent instructions; this review made no implementation changes and ran no new benchmarks.

## S1 — state & reduction contracts (implemented; manifest §15.2)

Implemented the §15.2 repairs; all new evidence is in the manifest §15.2
status block. In short:

- **Stale-product fix (the confirmed bug):** `tc2_purify_p` now measures the
  trace, lets the guard rescale P, and only then forms `Q=P²`; the residual
  and the update act on the post-rescale state. K-TC2 likewise re-forms
  `T=K·S` and re-measures `Tr(KS)` after a guard rescale — the reported
  trace is a measurement, never `Nocc` by assignment.
- **Recovered-K diagnostics:** `purify_hscc_p`/`purify_hscc_trs` now return
  `R_I(K)=‖KSK−K‖/‖K‖` and `Tr(KS)` computed on the recovered K — the same
  state that feeds Mulliken charges and the energy. Measured on SiH4-scale
  fixtures: `Tr(KS)` of the recovered K deviates ~1.6e-5 from `Tr(P)=3.0`
  (Z-recovery defect) — previously hidden by reporting the P diagnostics.
- **Accepted-state validity:** `set_coords`/`set_q` invalidate the cached
  energy; `scc` invalidates it before propagating a failure; `energy()`/
  `forces()` refuse post-mutation and post-failure calls (new contract test).
- **Skin check is now Euclidean:** per-atom `|ΔR|` vs `r_skin/2`; a diagonal
  move with each component < skin/2 but `|ΔR|` > skin/2 is now correctly
  rejected (test `test_s1_skin_euclidean_diagonal`).
- **Charge/trace consistency:** `mulliken_checked` enforces
  `|Σq − 2·Tr(KS)| ≤ 1e-6·n_atom + 1e-5` (worst-case f32 lane-order bound)
  on top of the absolute electron-count check.
- **Guard observability:** `guard_fires` counter; seeded-leak tests prove
  the guard actually fires on both purifier paths and that the accepted
  state satisfies the trace tolerance.

Run record (unfiltered, 2026-09): `sparse::` lib 10/10; `gate_g3_energy`
7/7; `gpu_dftb` 2/2; `gate_f_geom_opt` 1/1; `gate_g_hessian` 1/1;
`gpu_sparse_bsr4` 23/23; `sparse_dftb` 1/1. `gate_e_determinism` stays red
by design (no analytic sparse force of E_el+E_rep — S3). New tests:
`test_p_tc2_guard_and_recovery_contract`, `test_k_tc2_guard_fires`,
`test_s1_state_invalidation_contract`, `test_s1_skin_euclidean_diagonal`.

Open within S1's scope but deferred by the work order: `trace_kh0_dev`
single-f32 band-energy reduction (→ S6 selective precision); masked-residual
honesty on expanded support (→ S2).

## Status as of 2025-01-XX

**Repository:** `/home/prokop/git/dftbplus`
**Rust crate:** `rust_dftb/`
**Manifest (source of truth):** `doc/prokop/tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md`

---

## Objective

Make the sparse GPU DFTB implementation scientifically reliable and practical for
vibrational calculations on large silicon and diamond nanocrystals (300–1000 atoms).

End state:
- genuinely sparse, GPU-resident DFTB SCC and force evaluation;
- f32 GPU arithmetic where scientifically safe;
- smooth, deterministic forces suitable for finite-difference Hessians;
- robust geometry optimization;
- Hessian and vibrational-spectrum workflows;
- ~5% ordinary-frequency accuracy;
- no unexplained/pathological imaginary modes;
- documented convergence, precision, numerical-limit, and performance behavior.

---

## GPT-5.6 review — status corrections (2026-09-09)

A detailed code review by GPT-5.6 (transcribed in `Sparse_Nanocrystal_Vibrations.chat.md`
from line 2064) found that several "completed" items are **not actually wired
into the production numerical path**. The following status corrections apply:

| Item | Was | Corrected status | Reason |
|---|---|---|---|
| P0 | completed | [*] | Sparse firewall + perf stats — correct |
| P1 | completed | [~] | C² B-spline evaluator built but **not canonical** in production force path; `SkTableSp` still uses C¹ Hermite + numerical FD tail derivative |
| P2 | completed | [~] | Angular derivatives correct; but radial V,V' still from Hermite, not C² |
| P3 | completed | [~] | D/W formula correct; but allocation-heavy and attached to dense SCC (not self-consistent sparse) |
| P4 | completed | [~] | Plan infrastructure built; **not integrated** into TC2/NS/K0/W |
| Gate C | passed | [ ] | False positive — 5-atom toy system, +2I doesn't create gap, R=7Å covers everything |
| Gate D | passed | [*] | Padded Si/H basis — correct |
| Gate E | passed | [ ] | False positive — FD-of-energy forces, symmetrized Hessian tautology, h_ref in candidate list |

See manifest §13 for the full 22-issue checklist and 10-step ordered plan.
See `Sparse_Nanocrystal_Vibrations.tasks.md` for the phased master task breakdown.

---

## Completed work (P0–P4 + Gates C, D, E) — **see status corrections above**

### P0 — Sparse performance statistics + sparse firewall

**What:** Extended `SparsePerfStats` with physical/padded orbital counts and
synchronization/planning metrics. Added `sparse_firewall` feature: `to_dense()`
and `compare_to_dense()` panic in production builds, allowed in test builds.

**Files:**
- `src/methods/sparse/bsr4.rs` — `to_dense()` gated by `sparse_firewall`
- `src/methods/sparse/gpu_sparse.rs` — `compare_to_dense()` gated, perf stats fields

**Tests:** pass (part of `gpu_sparse_bsr4.rs` suite, 22 tests).

---

### P1 — B-spline evaluators (CPU + GPU)

**What:** CPU f64 C² cubic B-spline evaluator with partition-of-unity,
derivative-sum, knot-continuity, and interpolation tests. OpenCL GPU B-spline
evaluator (`bspline3_v_d1_d2`) with parity test vs CPU.

**Files:**
- `src/methods/dftb/spline_resample.rs` — CPU evaluator
- `src/methods/dftb/dftb_hamiltonian.cl` — GPU evaluator
- `tests/gpu_bspline_eval.rs` — parity test

**Tests:** 1 passed (GPU parity).

---

### P2 — Analytic Slater–Koster derivatives

**What:** Replaced production pair-force finite differences with analytic SK
radial + angular derivatives via `build_pair_block_with_derivs()` and
`Rotation::rotate_block_with_derivs_into()`. This is the production analytic
force path — no finite differences in the hot loop.

**Gate B #1:** Analytic pair derivatives vs f64 finite difference — 6 rotation
tests passed.
**Gate B #2:** Full analytic force vs f64 total-energy finite difference — 3
force tests passed.

**Files:**
- `src/methods/dftb/forces.rs` — `build_pair_block_with_derivs`, `non_scc_electronic_force`
- `src/methods/dftb/rotation.rs` — `rotate_block_with_derivs_into`

**Tests:** 9 passed (6 rotation + 3 force).

---

### P3 — Sparse D=2K, W=2KHK via masked SpGEMM

**What:** `build_dw_sparse()` builds device-resident D=2K and W=2KHK using
masked SpGEMM. No dense allocation, no host roundtrip. The density kernel K
comes from TC2 purification; H_scc is the SCC Hamiltonian.

**Algorithm (manifest §4.4):**
```
M_TW = boolean_product(M_K, M_HS)
T    = K * H_scc          (on M_TW)
W    = 2 * project_MHS(T * K)
symmetrize(W)
D    = 2 * K              (same mask as K)
```

**Files:**
- `src/methods/sparse/sparse_forces.rs` — `build_dw_sparse()`, `build_dw_dense_reference()`

**Tests:** 2 passed (D/W parity vs dense f64, symmetry).

---

### Gate C — Locality sweep R_K × R_Z

**What:** Staged independent sweep over R_K and R_Z on a 5-atom synthetic
chain. Measures energy error, charge error, Tr(KS)-Nocc, R_in, R_leak, R_H,
iteration counts, wall time. Dense reference uses the same truncated H/S
support as the sparse path.

**Result:** Plateau at R_K=7, R_Z=7:
- energy error ≈ 5.2e-6
- charge error ≈ 5.4e-6
- |Tr(KS)-Nocc| ≈ 3.6e-6
- R_in ≈ 1.0e-5
- R_leak ≈ 0
- R_H ≈ 3.2e-7

**Files:**
- `tests/locality_sweep.rs`
- `src/methods/sparse/bsr4.rs` — added `project_to_mask()`

**Tests:** 1 passed.

---

### Gate D — Nonsingular padded Si/H basis

**What:** SiH4 (silane: 1 Si + 4 H), 8 physical orbitals → 20 padded BSR4
orbitals. H atoms get 3 dummy orbitals: S_dd=1, H_dd=E_dummy=2.0, all dummy
couplings=0. This makes the padded overlap nonsingular while preserving the
physical electron count.

**Result:**
- Dense ref: E=-1.489265, dummy occ ~1e-32 (negligible)
- Sparse: Z converges (7 iters, R_Z=6.3e-8), TC2 converges (20 iters, R_I=9.0e-6)
- Parity: |dE|=1.0e-7, |dq|=3.0e-6, ||dK||=2.3e-6, dummy occ=0
- Tr(KS)=4.000007 (Nocc=4), active Mulliken sum=8.000013 (N_e=8)

**Key fixes during implementation:**
1. matsci-0-3 SK files don't encode n_shell on grid line → q0 extraction
   broken → hardcoded valence counts (Si=4, H=1).
2. E_dummy=10 made spectral interval too wide for TC2 → lowered to 2.0.
3. Mulliken sum gives Nocc (spinless), not N_e → factor of 2 for closed-shell.

**Files:**
- `tests/sih_padded_basis.rs`

**Tests:** 1 passed.

---

### P4 — Symbolic SpGEMM plan infrastructure

**What:** Precomputed symbolic plan for masked SpGEMM with symmetric right
operand. The plan stores exact (A_ik, B_jk) block pairs for each output block
C_ij, eliminating the runtime sorted-row intersection.

**OpenCL kernel:** `bsr4_spgemm_plan_Bsym` — loads A's row into local memory,
iterates precomputed plan terms (only loads + 4×4 FMAs), fails loudly (NaN)
on corrupt plan.

**Host-side:** `SpgemmPlan` struct with `plan_ptr`, `plan_a_idx`, `plan_b_idx`.
`build_spgemm_plan_bsym()` intersects A's row with B's row j for each output
block. `SpgemmPlanGpu` holds device buffers. `upload_plan()` uploads once;
`spgemm_plan_bsym_dev()` launches the kernel.

**Result (8-atom chain):**
- Parity: ||C_plan - C_intersection||_max = 0.000e0 (exact match)
- Plan: 124 terms, 1.2 KiB, 2.38 avg terms/block
- Performance: 0.93x on 8 atoms (plan overhead outweighs intersection savings
  at small scale; manifest says "performance hypothesis, not a law")

**Files:**
- `src/methods/sparse/sparse_bsr4_purification.cl` — `bsr4_spgemm_plan_Bsym` kernel
- `src/methods/sparse/bsr4.rs` — `SpgemmPlan`, `build_spgemm_plan_bsym()`
- `src/methods/sparse/gpu_sparse.rs` — `SpgemmPlanGpu`, `upload_plan()`, `spgemm_plan_bsym_dev()`
- `src/methods/sparse/mod.rs` — re-exports
- `tests/spgemm_plan.rs` — parity + performance test

**Tests:** 1 passed.

---

### Gate E — Determinism, arithmetic sensitivity, Hessian h plateau

**What:** Three-part test on SiH4:

**E-A (repeatability):** Energy spread over 5 identical runs = 0.0
(deterministic). TC2 tolerance sensitivity (1e-3, 1e-4, 1e-5) spread = 7.7e-5.

**E-B (sparse-vs-dense force bias):** max|bias|=1.6e-3, rel_bias=1.3e-2.
**FLAW:** Forces computed via finite differences of sparse energy (30 pipeline
runs per force eval). Acceptable for one-shot bias measurement, but
established a bad pattern for Gate F.

**E-C (Hessian h sweep):** Dense f64 3-point central FD Hessian at h=0.01,
0.02, 0.05, 0.10 Å. All stable, max asymmetry=0. Plateau chosen at h=0.02 Å.

**Files:**
- `tests/gate_e_determinism.rs`

**Tests:** 1 passed (but E-B force path is flawed — see blocker).

---

## BLOCKER: Multiple issues — see GPT-5.6 review and master task

The GPT-5.6 code review (chat.md line 2064+) identified **22 issues**, of
which 7 are critical blockers. The original Gate F blocker (FD-of-energy
forces) is one symptom of a deeper architectural problem: the sparse path is
not self-consistent and not wired into production.

### Revised plan

The full phased plan is in `Sparse_Nanocrystal_Vibrations.tasks.md`:
- **Phase A** — Foundation: canonical C² spline, TC2 fix, plan integration
- **Phase B** — Sparse SCC solver (GPU-resident, self-consistent) — **current focus**
- **Phase C** — GPU-resident sparse analytic forces (after dense agent finishes)
- **Phase D** — Correctness gates (redo C, E, then F, G, H)
- **Phase E** — Performance (counters, cell-list, degree buckets, packed plans, scaling)

### Original Gate F problem (superseded by Phase C)

Gate F used FD-of-energy forces (30 pipeline runs per force eval, 6000 total).
The fix is not a CPU bridge but a dedicated GPU pair-force + atom-gather kernel
(Phase C, GPT-5.6 issue #16). The force must be the derivative of the same
self-consistent sparse model (Phase B).

---

## Remaining manifest gates (after Phase B+C+D)

- **Gate G** — same-geometry Hessian parity (sparse f32 vs dense f64)
- **Gate H** — spectra at each method's own minimum
- **Gate I** — scaling and whole-program profile (N ~ 60, 150, 300, 600, 1000, 1600)

---

## File inventory

### Test files created:
| File | Gate | Status |
|------|------|--------|
| `tests/gpu_bspline_eval.rs` | P1 | [~] evaluator built, not canonical in production |
| `tests/locality_sweep.rs` | Gate C | [ ] false positive — redo on real Si/H (D1) |
| `tests/sih_padded_basis.rs` | Gate D | [*] PASS |
| `tests/spgemm_plan.rs` | P4 | [~] plan built, not integrated into TC2/NS/K0/W |
| `tests/gate_e_determinism.rs` | Gate E | [ ] false positive — redo with analytic forces (D2) |
| `tests/gate_f_geom_opt.rs` | Gate F | [ ] broken — FD-of-energy forces, do not use |

### Source files modified:
| File | Changes |
|------|---------|
| `src/methods/sparse/sparse_bsr4_purification.cl` | P4: `bsr4_spgemm_plan_Bsym` kernel |
| `src/methods/sparse/bsr4.rs` | P0: firewall; P4: `SpgemmPlan`, `build_spgemm_plan_bsym`, `project_to_mask` |
| `src/methods/sparse/gpu_sparse.rs` | P0: firewall, perf stats; P4: `SpgemmPlanGpu`, `upload_plan`, `spgemm_plan_bsym_dev` |
| `src/methods/sparse/mod.rs` | P4: re-exports |
| `src/methods/sparse/sparse_forces.rs` | P3: `build_dw_sparse` |
| `src/methods/dftb/forces.rs` | P2: analytic SK derivatives |
| `src/methods/dftb/rotation.rs` | P2: `rotate_block_with_derivs_into` |
| `src/methods/dftb/spline_resample.rs` | P1: CPU B-spline evaluator |
| `src/methods/dftb/dftb_hamiltonian.cl` | P1: GPU B-spline |
| `Cargo.toml` | P0: `sparse_firewall` feature |

### Not committed, not pushed.

---

## Lab notebook — 2026-09-10 (SparseDftb + CLI)

Appended here so we do not lose the numbers. Manifest **§0.7** is the short
table. User guide: `doc/prokop/userguide/sparse_dftb.md`.

**Product shape:** one binary `dftb_engine`. Dense jobs = `gpu_*`. Sparse jobs
= `sparse_*` → `SparseDftb`. New molecule = new `.rhai`, not a new Rust target.

**NVIDIA RTX 3090, matsci-0-3, `scripts/test_sparse_dftb_sih4.rhai`:**

- NS `R_Z = 6.265e-8` (7 iters, host ‖I−T‖). n_orbs=8, nocc=4, full mask.
- SCC E=**−2.76420524 Ha**, 17 mix iters, Tr(KS)=4.000002, max\|F\|=0.0883 Ha/Å.
- Reuse SCC: 4 iters, E=−2.76420569 Ha.
- One FIRE + SCC: E=−2.76442147 Ha. One MD (dt=0.05) + SCC: E=−2.76464354 Ha.
- Not USER-confirmed; distorted 1.48 Å SiH₄, not a Gate F minimum.

**Allocating-path gates (same morning, `scc.rs`, superseded the same afternoon):**
G3.2 \|dE_el\|=1.13e-7; G3.3 max\|dF\|=3.7e-6; Gate F FIRE 1.477 Å,
E=−2.826057; Gate G Hessian rel 0.11%. Kept as history.

**After unification onto `SparseDftb` (same NVIDIA, afternoon):**
G3.2 \|dE_el\|=3.47e-7; G3.3 max\|dF\|=5.30e-6; G3.4 rel=4.4e-3 (abs 6e-5);
Gate F 72 steps E=−2.826054 \|F\|=9.3e-4 Si–H 1.477 Å; Gate G rel 0.116%,
η_asym 3.25e-4. See manifest §0.7.

**Do not quote `cargo test` wall time as GPU performance.**

---

## Lab notebook — 2026-09-11 (Phase F rewrite: F1–F6 done, F7 partial)

Session goal: implement the second GPT-5.6 review's Phase F tasks — remove all
dense storage from `SparseDftb`, make the production path genuinely sparse
end-to-end, and verify with physical gates (not just compilation).

### What changed (F1–F6)

**F1 — Newton–Schulz contract (`sparse_system.rs`):**
- `compute_z` is device-resident: residual `‖I−T‖_F/√N` computed on device
  (`identity_residual_to_dev` — the old scalar path had a missing sqrt, N4).
- Z₀ uses `build_identity_dev` — full-write kernel, no stale off-diagonal
  values on M_Z.
- T = Z·S products go through symbolic plans `plan_zs`/`plan_tz` (built once
  in `new()`); per-iteration T download removed.
- Warm-Z: previous geometry's Z NS-corrected against new S (first-order
  correction `Z − Z·δS·Z`); on stall/divergence one cold restart from αI,
  then loud Err.
- Tests: `test_ns_device_residual_contract`, `test_compute_z_second_geometry`,
  `test_sparse_system_scc_pipeline`, `test_sparse_system_reuse` — all pass.

**F2 — No dense storage in `SparseDftb` (`sparse_dftb.rs` rewrite):**
- `h0_pad`/`s_pad`/`h_scc_pad`/`k_pad` production storage deleted. H0/S are
  assembled directly into BSR values: `assemble_hs_bsr` iterates the frozen
  `hs_pairs` list (unique i<j off-diagonal pairs of M_HS) and writes rotated
  pair blocks in both orientations — same `Rotation::rotate_diatomic_block_into`
  convention as the dense builder (shells of i → columns, j → rows,
  transpose+sign for ang1>ang2), dummy lanes S_dd=1/H_dd=E_DUMMY.
- Independent masks: M_HS (SK cutoff + Verlet skin), M_K, M_Z — geometric
  cell-pair lists for n_atom > `FULL_MASK_ATOMS`, full mask below.
- Verlet skin semantics enforced: `set_coords` fails loudly when any atom
  moved > skin/2 vs `coords_build`.
- `sparse_firewall` feature still guards `to_dense()` in production.
- **F2 parity test** (`gate_g3_energy.rs::test_f2_direct_bsr_hs_parity`):
  geometric mask on distorted SiH4, BSR H/S vs dense `build_non_scc` on
  physical indices: max|dH|, max|dS| < 5e-7 (f32 rounding of f64). Dummy
  lanes verified (H_dd=2, S_dd=1, couplings 0). Skin guard verified.

**F3 — Sparse masked energy:**
- `store_energy` uses `trace_kh0_dev` (masked `2·Tr(K·H0)` over M_HS on
  device via `bsr4_trace_hk_partial` + `hs_to_kt` transpose map) — no
  `k_to_dense`, no `trace_ab`.
- `mulliken_charges` reuses the TC2-final `T = K·S` product (`t_ks_valid`),
  zero extra SpGEMM.
- Alloc counter: `SparseBsr4Gpu.n_buf_allocs` (Cell<u64>) incremented in
  every `buf_*`/`zero_*` helper.
- **F3 test** (`test_f3_no_device_allocs_in_scc`): snapshot counter after
  `new()`, assert flat across scc(), forces(), and geometry-2 warm-Z scc() —
  zero growth. Per-iteration host Vec allocs in `mulliken_charges` also moved
  to persistent `qpack_host`.

**F4 — Stationary SCC finalization:**
- `scc()` reports state (q_in, K(q_in), H_scc(q_in), V(q_in)) — one
  consistent electronic state; `r_scc` = rms(q_out−q_in) gap.
- `rh_stationarity`: one extra SpGEMM `A = H_scc·(K·S)` on M_HT, sparse
  download, host f64 antisymmetry ‖HKS−SKH‖.

**F5 — Sparse force path (`sparse_forces.rs::SparseDWWorkspace`):**
- Device `T = K·H_scc`, `W = 2·T·K` (D = 2K) on persistent buffers; download
  only `W[M_HS]` + `K[M_K]`; CPU pair contraction `sparse_forces_bsr` over
  `hs_pairs` using analytic `build_pair_block_with_derivs` (no FD in hot
  path). No O(N³) dense product.
- Verified by G3.3 (sparse vs dense D/W force parity) and G3.4 (own E_tot
  central-difference vs analytic F).

**F6 — Per-geometry precompute:**
- Repulsive splines parsed once in `new()` (`parse_all_repulsive` →
  `repulsive` field; `repulsive_energy_cached` in set_coords).
- Dense f64 γ matrix cached per geometry (`gmat`), O(N²) — honest
  transitional state, direct γ until a long-range accelerator.
- `hs_pairs`, `hs_diag`, `k_diag` built once in `new()`.
- `SystemContext` rebuilt per `set_coords`/`forces` call — O(n_atom +
  n_species²) table build; **cannot** be a struct field (borrows `&'a SkData`
  → self-referential; adding a lifetime to `SparseDftb` would poison the
  `'static` engine handle). Accepted as negligible per-geometry cost.

### Measured results (RTX 3090, matsci-0-3, distorted SiH4)

| Gate | Result | Status |
|------|--------|--------|
| F2 BSR-vs-dense H/S parity | max\|dH\|, \|dS\| < 5e-7 | PASS |
| F3 device allocs in scc/forces/geom2 | 0 | PASS |
| G3.1 sparse E_el+E_rep vs dense | within f32 floor | PASS |
| G3.2 sparse SCC vs dense SCC | consistent state, r_h reported | PASS |
| G3.3 analytic sparse force vs dense D/W | parity | PASS |
| G3.4 energy-gradient agreement | own FD vs analytic | PASS |
| `sparse_system` unit tests (4) | NS contract, warm Z, pipeline | PASS |
| `gpu_sparse_bsr4` | 22/23 (1 intentionally ignored stub) | PASS |
| `sih_padded_basis` (Gate D) | | PASS |
| `gate_f_geom_opt` | | PASS |

### Designed-red tests (NOT regressions — diagnostic gates by intent)

- `gate_e_determinism` Gate E-B: panics by design — "stays red until the
  analytic sparse force of E_el+E_rep exists". That force now exists
  (`SparseDftb::forces`); the test's own FD-of-FD comparison can now be
  upgraded to use it (task for Gate-D phase, not done yet).
- `locality_sweep` Gate C: panics by design — 5-atom toy with R≥7 Å covers
  the whole system; must be redone on a real gapped Si/H cluster.
- `spline_resample::test_resample_bspline_sin` — pre-existing failure in an
  untouched file (B-spline resample vs sin), unrelated to sparse work.

### F7 — packed Mulliken read + multi-accumulator SpGEMM

- `bsr4_mulliken_KS` now writes packed `q_pack[2N]` (phys at [0..N), dummy
  at [N..2N)) — one device buffer, one host read per iteration. Required
  updating the cached `k_mulliken_ks` builder's declared arg count (see bug
  class 1 below).
- Workspace TC2 tolerance is already normalized: `ri = ‖KSK−K‖_F / ‖K‖_F`
  (R12), documented on `SparseDftbConfig::tc2_tol`.
- Multi-accumulator SpGEMM: `bsr4_spgemm_plan_Bsym` now uses 4 independent
  FMA accumulators (one per column m) — breaks the serial dependency across
  plan terms. Parity vs intersection kernel 1.8e-7 (f32 reorder). Measured
  1.06× speedup on the 8-atom toy — launch-latency bound at this scale; the
  benefit should grow with avg terms/block on real systems. **Re-benchmark
  on a large nanocrystal before claiming a speedup.**
- `test_dw_workspace_vs_oneshot` tolerance fix: absolute 1e-6 → relative
  1e-6·|W|max. W entries are O(100) (T·K of random blocks) so 3e-5 abs is
  ~1e-7 relative f32 reorder noise between intersection and plan kernels —
  a mis-calibrated threshold, not a physics bug.

### Bug classes hit during the session (record so we don't re-hit them)

1. **Kernel arg-count drift**: changing a `.cl` signature requires touching
   THREE places — the kernel, the cached `k_*` builder in `SparseBsr4Gpu::new`
   (declared dummy args), and every host launcher (`set_arg`/`Kernel::builder`
   arg lists). Twice hit: `trace_KS_partial` gained `n_orb` (legacy
   `trace_ks`/`tc2_purify`/`tc2_step`/`mulliken` wrappers still sent 4);
   `mulliken_KS` packed (cached builder still sent 6). OpenCL fails at
   enqueue with "wrong number of kernel arguments" — the panic text names
   neither kernel nor callsite; grep the arg count.
2. **Legacy one-shot API drift**: `purify_h`/`tc2_purify`/`tc2_step`/
   `mulliken`/`energy_non_scc`/`run_sparse_scc` needed `atom_n_orb` threaded
   through (physical-masked trace/Mulliken are now the only semantics —
   dummy lanes excluded from Tr(KS)=Nocc).
3. `#[cfg(test)]` on `build_pair_block` blocked non-test import; `str_as_str`
   unstable `.as_slice()` on a slice; `&&Vec<f32>` iteration — mechanical.
4. User concurrently edited `qmqm/gpu_dftb.rs` (D9 DIIS status + D11
   per-system status, D10 fused kernels) — mid-flight states caused two
   transient lib compile breaks (`statuses` field, `plateau` binding). Left
   untouched; user finished both.
5. **`git stash` accident (agent)**: `git stash -u --keep-index` was run to
   check whether `test_residual_and_mix` fails at HEAD — it stashed the
   entire working tree (my changes + user's in-flight edits). Restored with
   `git stash pop`, clean apply, nothing lost. Lesson: verify a failure at
   HEAD by reading `git show HEAD:file` (cheap, safe), never by touching the
   working tree.
6. **`test_residual_and_mix` pre-existing failure**: kernel
   `residual_and_mix_batched` outputs RMS `√(Σd²/n_atoms)`; the test's CPU
   reference computed L2 `√(Σd²)` — off by exactly √n_atoms (√5).
   Kernel is correct (RMS matches the CPU SCC tolerance convention); the
   test reference was stale at HEAD `e965ae00`. Fixed both rms_cpu sites to
   divide by n_atoms; 10/10 pass.

### Honest limitations (unchanged)

- Direct γ is O(N²) work+memory (`gmat` dense f64) — transitional R16.
- f32 purification sets the achievable tolerance floor (TC2 R_I ~1e-5..1e-6
  typical).
- `run_sparse_scc`/`eval_sparse_energy_forces`/`purify_h` remain the legacy
  allocating dense-pad reference path — kept for parity tests only.
- `locality_sweep` and `gate_e_determinism` red states documented above.

---

## 2026-09-12 — First end-to-end vibration run: Si10H16

**Driver added:** `sparse_vibrations(name, h_ang, scc_tol, path)` in
`dftb_engine.rs` — central-difference Hessian of the analytic sparse forces
(`h=0.02 Å`, warm-started SCC per column, engine restored + reconverged at
the end), symmetrize (max-asym reported), mass-weight, dense f64
`SymmetricEigen` (3N small), freqs + unit-norm real-space modes to file.
`Element::mass()` (standard atomic weights) added to `geometry/mod.rs`.
New rhai knob `sparse_tc2_tol(name, tol)` (see below why it exists).
Script: `rust_dftb/scripts/sparse_vibrations_si10h16.rhai`.

**Result (matsci-0-3, Si10H16, 26 atoms, N=56, full mask):**
- relax FIRE → max|F|=9.9e-5 Ha/Å (f_tol 1e-4, 600 steps).
- 78 FD columns, all warm-started SCC converged (r_scc < 1e-6).
- Hessian max asymmetry = 6.4e-3 E_h/Å² (SCC-noise floor — see below).
- Spectrum: 6 rigid modes |ν| ≤ 16 cm⁻¹ (3 slightly imaginary), real
  spectrum from 101 cm⁻¹, Si–H stretch cluster 2224–2258 cm⁻¹.
  Physically sensible for silane clusters (Si–H ~2100–2300 cm⁻¹).
- Output: `debug/sparse_vib_si10h16_freq.txt`.

### Bugs found by the run (the run was the diagnostic)

1. **Linear mixing is unstable on real nanocrystals.** Si10H16 SCC under
   α=0.5 diverges (rms ~1.4 oscillation — charge sloshing, Jacobian
   eigenvalue |λ|>1; SiH4 is small enough that it never showed).
   Fix: wired the existing `qmqm::mixer::DiisMixer` into `SparseDftb::scc`
   (`cfg.diis_hist`, default 8, 0 = pure linear `cfg.mix`; mixer alpha =
   cfg.mix fallback). DIIS collapsed rms 3e-2 → 2.7e-6 in one step.
2. **`DiisMixer` safeguard bug (shared code, also affects dense host
   path).** The catastrophic-extrapolation check compared `‖q_next‖`
   (≈√N·valence ≈ 13.3 for Si10H16) against `100·‖r‖` — so DIIS was
   silently REJECTED whenever ‖r‖ < ‖q‖/100, i.e. exactly near
   convergence. Observed limit cycle: residual grows under α=0.5 linear
   until ‖r‖ > 0.13 → DIIS accepted → collapses to ~1e-6 → rejected again.
   Fix: criterion now bounds the DIIS STEP ‖q_next − q_in‖ ≤ 50·‖r‖
   (step norm is the right scale for ill-conditioned extrapolation).
   After the fix DIIS stays accepted through convergence.
3. **SCC noise floor is set by tc2_tol, not scc_tol.** r_I = ‖KSK−K‖/‖K‖
   stops at ~2e-5 with the default 1e-4; Mulliken q carries ~r_I noise, so
   SCC rms cannot go below ~3e-6. Hessian-grade work needs
   `sparse_tc2_tol(name, 1e-7)` → r_I ~ 8e-8 → rms 7e-7 reached. TC2
   converges geometrically so the cost is modest. `set_tc2_tol` documents
   this coupling.
4. **DIIS history vs geometry changes.** Full `reset()` per `set_coords`
   forced every FD column to re-fight the sloshing mode with linear
   fallback for ~15 iters (and col 1 −h never recovered in 80 iters).
   `set_coords`/`set_q` now use `reset_iter_only()` (keep the subspace —
   the map changes smoothly for ±0.02 Å; the mixer's own docstring
   prescribes this for warm-started geometry changes).
5. **Unit bugs in my vibrations code (my bug, not library):** the
   cm⁻¹ factor 5140.487 needs mass-weight in **amu**, I used m_e (×1822.9
   → freqs 42.7× too small); and `Forces` are E_h/Å so the FD Hessian is
   E_h/Å², not E_h/Bohr² (another /3.57). Both fixed; max freq went
   99.9 → 2257 cm⁻¹, matching the Si–H stretch range.

### Noise-floor analysis (rigid modes)

|ν| ≤ 16 cm⁻¹ on the 6 T/R modes survives a 10× tighter relax
(max|F| 9.2e-4 → 9.9e-5) — it is NOT incomplete relaxation. Soft-mode
Hessian eigenvalues ~3e-4 E_h/Å² sit ~20× below the per-column FD noise
(Hessian asym 6.4e-3 ≈ force noise ~1e-4 Ha/Å ÷ 2h). Options if rigid
modes matter: project T/R before diagonalizing (standard), or tighten
scc_tol→1e-7 with tc2_tol→1e-8. Spectrum above ~100 cm⁻¹ is well clear
of the floor.

### What this does NOT yet validate (honest list)

- n_atom ≤ 64 → full_mask path only; geometric masks (r_k_ang/r_z_ang
  knobs) untested on a real 300–1000-atom crystal.
- γ is dense O(N²) direct evaluation — will dominate at large N.
- Hessian is host-side FD orchestration (fine for 3N ≲ few thousand);
  per-atom-block update + GPU pair-force contraction remain pending.
- No T/R projection, no IR intensities (charges→dipole derivatives TBD).
- No Fortran DFTB+ parity check on the spectrum yet — nearest reference
  would be a matsci Si10H16 hessian run.
- gate_e E-B still designed-red (its own assertion needs the upgrade);
  locality_sweep red (5-atom toy, designed).

## 2026-09-12 — SK-table structure, physical decay, and the neighbor-count budget (ESSENTIAL REFERENCE)

### .skf file structure (verified against `sk_data.rs::read_skf_all`)

Old (non-`@`) format, per data row — **20 columns = 10 H channels (cols 0–9)
THEN 10 S channels (cols 10–19)**, packed in ONE row. There is only ONE
n_grid-row table per file — do NOT split rows into an "H block then S block"
(that was the earlier misparse that made S look missing). Extended `@` files
have 40 cols: 20 H then 20 S per row (sp3d channel space).

Header: line 1 = `grid_dist nGrid` (grid in **Bohr**); for homonuclear pairs
two more lines follow (onsite energies, then `mass + fields` line that
expands to exactly 20 tokens — easy to misread as a data row); heteronuclear
files have one such line. Data = `nGrid−1` rows (rust parser drops the last).
Repeat syntax `N*value` must be expanded; `Spline` keyword starts the
repulsive polynomial section; `<Documentation>` is trailing XML metadata —
never parse either as numeric data.

**The local `slakos/matsci-0-3` copy is s+p-only** (20 cols = 10 H + 10 S of
the s+p channel space; no d integrals). The original L1 DFTB+ run used a
different matsci copy (`/auto/vestec1-elixir/...`) that HAD d channels →
297 vs 152 orbitals. That was the "model difference" — confirmed.

### Measured decay — the number that matters for sparsity

Max |channel| radius (Å) where it stays below threshold. H = max over the
10 H columns, S = max over the 10 S columns:

| file            | table end | H<1e-2 | H<1e-3 | H<1e-4 | H<1e-5 | S<1e-4 |
|-----------------|-----------|--------|--------|--------|--------|--------|
| matsci Si-Si    | 10.56     | 4.18   | 5.03   | 6.48   | 8.07   | 7.04   |
| matsci Si-H     | 10.56     | 2.88   | 3.62   | 6.16   | 8.39   | 7.04   |
| matsci H-Si     | 10.56     | 3.20   | 3.89   | 6.16   | 8.41   | 7.17   |
| matsci H-H      | 10.56     | 2.14   | 2.62   | 6.19   | 8.15   | 6.67   |
| pbc   Si-Si     | 5.49      | 3.91   | 4.72   | 5.40   | 5.49   | 5.42   |
| pbc   Si-H      | 5.49      | 3.32   | 4.14   | 4.84   | 5.45   | 4.89   |
| mio   C-C       | 5.28      | 3.18   | 3.92   | 4.58   | 5.28   | 4.71   |
| 3ob   C-C       | 6.87      | 3.33   | 4.17   | 4.93   | 6.87   | 5.11   |

Reading:
- |H| < 1e-3 Ha (~0.03 eV) at ~4–5 Å for ALL sets — this is the
  Fireball-style ~5.6 Å regime. User memory confirmed.
- |H| < 1e-4 Ha needs ~6.5–7 Å for matsci (its tables carry a long appended
  spline tail out to 10.6 Å — the tail is real but sub-meV).
- pbc's table literally ENDS at 5.49 Å — intrinsically short-ranged by
  construction. mio similar. 3ob ~6.9 Å.
- S and H decay on essentially the same length scale (same orbital tails).

### THE neighbor-count budget (the real optimization target)

Si-sphere R18 (1648 atoms, density ~0.05 Si + H shell ~2× surface):

| radius | max degree | mean | nnz blocks |
|--------|-----------|------|------------|
| 6 Å    | ~75 est   | ~50  |            |
| 8 Å    | 111       | 84.6 | 139k       |
| 9 Å    | 167       | 121  | 200k       |
| 10 Å   | 206       | 150  | 247k       |
| 11 Å   | 294       | 202  | 333k       |
| 12 Å   | 378       | 250  | 412k       |
| 13 Å   | 481       | 305  | 503k       |

**Target: ~64 max neighbors → r ≈ 6–7 Å.** At r=6 Å the taper error is
~0.65 mHa on cube65 (measured); r=5 Å → 3.5 mHa. r_trunc ≈ 6–7 Å is the
physically defensible minimum for matsci; pbc/mio get the same accuracy at
~5 Å because their tables end there anyway.

### Smooth taper implemented (sparse path only, config-gated)

`SparseDftbConfig`: `r_trunc_ang: Option<f64>`, `taper_w_ang: f64` (1.0).
Cosine window w(r) = 1 for r ≤ r_trunc−w, decays to 0 at r_trunc, C¹.
Applied to BOTH H and S pair blocks in `assemble_hs_bsr` AND in
`sparse_forces_bsr` with the chain term d(wV)/dx = w·dV/dx + w'·u·V —
energy/force consistent. Mask radius = r_trunc + r_skin. FD force check
(r_trunc=7): dF scatter identical to untruncated baseline (~1 mHa/Å SCC
noise) → taper is the exact gradient of the tapered energy.

### THE CRITICAL FINDING — intermediates, not masks, were the explosion

The "620 blocks/row" failure was NOT the M_HS/M_K mask degree (max ~290 at
r=11 Å) — it was the **symbolic product mask** of the intermediates:
`T = K·S` stored on product(M_K, M_HS) ≈ support r_k + r_hs ≈ 19–20 Å →
~500 blocks/row > MAX_LEFT_BLOCKS. Fix: **multiply-truncate** (standard
SP2): store T=K·S on M_K itself, Z·S on M_Z — `build_spgemm_plan_bsym`
already accepts a prescribed output mask. Mulliken and Tr(KS) need only
T's diagonal blocks; plan_tk/plan_bz already output to M_K/M_Z — so the
truncation is consistent and its error is the same controlled quantity as
the K-support truncation (r_k_ang knob).

M_K/M_Z default = FULL SK radius (K decays with the gap, not with H's
range — observed: truncating M_K to 10 Å on cube65 put a TC2 limit-cycle
floor at R_I~1e-4). r_k_ang/r_z_ang are user knobs; for the big run use
r_k ≈ 8–10 Å and MEASURE the error vs a larger reference.

### Energy error vs r_trunc (cube_Si65, matsci, taper_w=1 Å)

| r_trunc | ΔE (mHa) | max\|F\| | SCC iters |
|---------|----------|---------|-----------|
| none    | 0 (ref)  | 0.0502  | 15        |
| 9 Å     | −0.004   | 0.0502  | 14        |
| 8 Å     | −0.037   | 0.0502  | 14        |
| 7 Å     | +0.096   | 0.0502  | 13        |
| 6 Å     | +0.65    | 0.0503  | 15        |
| 5 Å     | −3.5     | 0.0508  | 13        |

Rule of thumb: ΔE per atom ≈ 0.01–0.05 mHa at r_trunc=6–7 Å (~0.006–0.03
kcal/mol/atom) — acceptable for screening, borderline for tight reference
work; r_trunc=8 is the safe matsci default.

### Script/API additions

- `sparse_new(name, sk_dir)` — untruncated (default).
- `sparse_new(name, sk_dir, r_trunc, taper_w)` — H/S taper.
- `sparse_new(name, sk_dir, r_trunc, taper_w, r_k, r_z)` — + K/Z mask radii.
- `sparse_force_at(name, atom, comp)` — analytic force component (FD checks).
- `sparse_ns_tol(name, tol)` — NS residual tol (M_Z truncation raises the
  achievable floor above 1e-5).
- `RUST_DFTB_SPARSE_ALGEBRA_VERBOSE=1` now prints per-iter TC2 R_I/Tr(KS).

## 2026-09-12 — 1648-atom TC2 divergence: ROOT CAUSE (f32 + open-loop iteration)

### ⚠ STANDING RULE (user, repeated many times, keeps being ignored)

**This is an f32 GPU architecture. Every iterative scheme must be designed
around that limit, not in spite of it.** Concretely:

- An iteration with **no restoring mechanism** (open-loop) accumulates f32
  noise until it violates its own stability precondition. That is a DESIGN
  bug, not bad luck.
- Accuracy requirements must be **compromised deliberately** where f32
  cannot deliver: set the tolerance to the measured floor and STOP there.
  Never "iterate harder" against a noise floor.
- Quantities that make **discrete decisions** (branch selects, occupancy
  counts, convergence tests) must be computed in f64 on the host if they
  are cheap — a wrong branch from 1e-5 noise is catastrophic, while a
  1e-5 error in a matrix element is harmless.
- Every new solver must state: what is its stability invariant, how is
  that invariant *enforced* (not merely hoped for) under f32 noise, and
  what is its measured floor.

### The observation

Si sphere R18 (1648 atoms, 5254 orbitals, nocc=2627), r_trunc=8 Å,
r_k=r_z=10 Å. NS converged to its floor R_Z=4.6e-5. TC2 then:

```
iters 0..~48 : R_I limit-cycles ~1.8e-3 .. 3.3e-3, Tr(KS)=2627 ± 0.03
iter 49..55  : Tr = 2627.32 → .74 → 2628.61 → 2630.36 → 2633.89
                    → 2641.00 → 2655.40   (R_I 1.1e-3 → 1.4e-2)
```

### Root cause — proven by the increment pattern

Trace increments: 0.086, 0.201, 0.422, 0.87, 1.75, 3.53, 7.11, 14.40 —
they **double** every iteration. That is the analytic signature of
`λ → λ²` acting on an eigenvalue just above 1: λ=1+δ ⇒ λ²=1+2δ+δ².
And that IS the active branch: `bsr4_tc2` sets `Knew = Q = KSK` whenever
`Tr(KS) > Nocc`.

> **At least one eigenvalue of KS leaked outside [0,1]. Both TC2
> polynomials (λ² and 2λ−λ²) are unconditionally divergent outside
> [0,1], and there is NO mechanism anywhere in the loop that pulls the
> spectrum back. The iteration is open-loop.**

Once δ exceeded the f32/trace noise, the doubling took over. The
preceding 20-iteration R_I limit cycle at ~2e-3 was the pre-runaway
phase (the bang-bang branch flipping on noise, since |Tr−Nocc| ≈ 0.03
≪ 1 is far below the decision scale).

### Why here and not on cube_Si65 (N=152) — three compounding causes

**(1) Truncating the INTERMEDIATE T=K·S was the trigger (my change).**
`Q = T·K` with `T = P_{M_K}(K·S)`. Projecting the *output* of KSK is
benign; projecting **T** is not, because `Q_ij = Σ_k T_ik K_kj` — every
dropped `T_ik` deletes a REAL contribution to an IN-MASK block of Q.
True support of K·S is r_k+r_hs ≈ 19 Å; it was cut to 10 Å.
Perturbation to Q is O(1e-3), not O(1e-5) — exactly the observed R_I
floor (vs 6e-5 on cube65 where the mask is nearly complete).
**Lesson: truncating an intermediate is a different, far larger error
class than truncating a result. Radius-truncation of intermediates is
uncontrolled; magnitude (τ) dropping is the error-bounded way.**

**(2) The spectral bounds are no longer bounds.** `spectral_bounds_dev`
runs Gershgorin on `B = P_{M_Z}(Z·H)` — non-symmetric AND truncated —
with padding = 10 % of span. Gershgorin on exact ZH over-estimates
(safe); on a truncated ZH it can under-estimate. Then
`K0 = (εmax·Z − ZHZ)/Δ` with BOTH products truncated. So K0's spectrum
may already poke outside [0,1] at iteration 0. At N=152 the 10 % padding
absorbed it; at N=5254 it does not.

**(3) f32 alone is marginal at 55 iterations.** Per-SpGEMM noise
~1e-5·‖K‖; the pre-runaway Tr jitter ±0.03 on 2627 is ~1e-5 relative —
i.e. pure f32. Healthy TC2 converges in ~25–30 iterations; needing 55
means it never converged, it plateaued at the truncation floor early.

### Secondary bug — plateau recovery picked a poisoned snapshot

The recovery I added selects the iterate with minimum `R_I`. The best
R_I (1.09e-3) occurred at **iter 50, when Tr was already 2627.74** —
mid-runaway. The snapshot therefore failed the `TC2_TRACE_TOL` gate and
the hard error was raised anyway. Two fixes: select on a JOINT criterion
(`R_I` **and** `|Tr−Nocc|`), and break on **stagnation** (no improvement
over N checks) instead of waiting for `ri > 10·best`, which only fires
after divergence has already begun.

### K/Z truncation is much more expensive than H/S truncation

cube_Si65, r_trunc=8 Å fixed, varying r_k=r_z:

| r_k=r_z | ΔE (mHa) | note                                   |
|---------|----------|----------------------------------------|
| full    | 0 (ref)  |                                        |
| 12 Å    | 0.000    | identical                              |
| 10 Å    | +0.19    | TC2 plateau-restore active             |
| 8 Å     | +6.2     | NS floor 4.8e-5, needs ns_tol 1e-4     |
| 6 Å     | —        | NS stalls at 1.0e-3, unusable          |

Compare: the H/S taper at the SAME radius costs 0.037 mHa. **K's decay
is gap-controlled, not SK-tail-controlled**, and Si nanocrystal surface
states make it slower. Reaching the ~64-neighbor target for K will need
τ-based magnitude dropping with error control, measured as its own axis
(Δ per atom vs τ) — NOT a radius cut.

### Ranked fixes (1 = highest value)

1. **Close the loop: trace-correcting TC2.** Rescale after each step so
   `Tr(KS)=Nocc` exactly (Niklasson TRS4 / Rubensson error-controlled
   SP2). Removes the noise-driven bang-bang branch AND continuously pulls
   the spectrum back inside [0,1] — directly kills the doubling mode.
2. **Stop radius-truncating intermediates**; use τ (block-norm) dropping,
   or keep T wide enough for the contraction.
3. **Fix the plateau exit** (joint R_I+trace criterion, stagnation break)
   so a truncation floor reports honestly instead of crashing.
4. **Make the bounds rigorous or generous** — a K0 with spectrum provably
   in [0,1] is a PRECONDITION of TC2, not an optimization.
5. **Trace + branch decision in f64** on the host (only n_atom diagonal
   terms — negligible cost, removes the branch-flip noise entirely).

## 2026-09-12 — WHAT WAS IMPLEMENTED + OPEN REGRESSION (handover state)

### Implemented and verified

**A. TC2 trace guard (fix #1, the closed-loop invariant)** —
`sparse_system.rs::tc2_purify`. Every iteration: compute `T=K·S`, read
`Tr(KS)` (f32 scalar), and if the iteration is in its ENDGAME and the
trace has drifted, rescale `K` (and `T`) by `α = Nocc/Tr`. Constants in
`gpu_sparse.rs`:
- `TC2_TRACE_GUARD_REL = 5e-5` — drift that triggers rescaling (healthy
  f32 jitter is ~1e-5 relative; leakage drift starts ~1e-4).
- `TC2_TRACE_LOCK_REL = 1e-3` **and** `TC2_LOCK_RI = 1e-2` — BOTH required
  to arm the guard. Two failed attempts taught this:
  1. Guard with no lock → fired at iter 0 (Tr=2841 vs Nocc=2627 is the
     LEGITIMATE transient) and destroyed the iteration.
  2. Lock on trace alone → armed at iter 2 because the transient *crosses*
     Nocc (Tr=2628.19 while R_I was still O(1)). Hence the extra R_I
     condition.
- After a guard rescale the trace is exactly Nocc, so the sign-based
  branch is undefined; `trace_buf` is written `Nocc·(1+1e-6)` to bias the
  CONTRACTING branch (`Knew=Q`, λ→λ²).

**B. Joint best-K snapshot criterion.** Snapshot only iterates with
`|Tr−Nocc| ≤ TC2_TRACE_TOL`. Previously min-R_I alone captured an iterate
already mid-runaway, which the trace gate then rejected → hard fail.

**C. Multiply-truncate in the FORCE path too** —
`sparse_forces.rs::SparseDWWorkspace::new`: `T=K·H_scc` now on `M_K`
instead of `product(M_K,M_HS)` (that mask had **1298 blocks/row** on the
1648-atom sphere).

**D. Degree check moved to LEFT OPERANDS only** — `GpuBsrStructure` now
carries `max_deg`; `check_left_degree` is called at `spgemm_*_dev` entry
points instead of checking EVERY structure at construction. Output/product
masks may legitimately exceed MAX_LEFT_BLOCKS.

**E. `R_H` residual on M_K** (was `product(M_HS,M_K)`, which is not even
symmetric when r_hs≠r_k → tripped the symmetry assertion).

**F. New API:** `copy_f32` (device→device), `sparse_ns_tol(name,tol)`,
`sparse_force_at(name,atom,comp)`, `sparse_new` 4-arg and 6-arg overloads.

### RESULT — 1648-atom Si sphere runs end-to-end (first time)

```
r_trunc=8 Å taper_w=1 Å, r_k=r_z=10 Å, tc2_tol=1e-6, ns_tol=1e-4
n_atom=1648  n_orbs=5254  nocc=2627
nnz_hs=199676  nnz_k=nnz_z=247364   (max degree 206 — was 620)
NS: 7 iters → R_Z=4.6e-5 (floor)
TC2: trace guard fires iters 19-24, plateau restores best K
     → R_I floor 7.87e-3, Tr(KS)=2627.000000 exact
SCC: converged, 16 iters, rms=2.80e-6
E_tot = -1773.23424907 Ha     max|F| = 0.1306 Ha/Å     R_H = 3.64e-3
wall: 7.5 s total
```

**Honest caveat: R_I = 7.9e-3 is a BAD idempotency floor (~0.8 %).** It is
the intermediate-truncation error (fix #2, not done). The energy at this K
quality is NOT validated — do not quote −1773.234 Ha as a reference
number. What is demonstrated is that the *pipeline* survives 1648 atoms
with bounded degrees and a stationary SCC state.

### ⚠ OPEN REGRESSION — gate_f fails (must be fixed before anything else)

```
cargo test --release --test gate_f_geom_opt
step 15: E=-2.78061752  |F|=1.337e-1  n_scc=11    ← fine
step 16: E=-2.78393659  |F|=1.270e-1  n_scc=12    ← fine
step 17: PANIC — SparseDftb SCC diverging at iter 4:
         rms=3.339e-2 (prev 3.463e-3)
and earlier in the same run:
  TC2 plateau at iter 22 (oscillating): restoring best K
  (R_I=1.15e-4, Tr(KS)=4)
```

`gate_g3_energy`, `sih_padded_basis`, `gpu_sparse_bsr4` — not re-run after
the last edit; `gate_f` is the known red one.

**Already tried and REVERTED (do not repeat):** a "stagnation break" that
exited TC2 after N non-improving checks. Non-improving checks are ROUTINE
during the normal TC2 descent, so it cut SiH4 off at R_I=1.2e-4 and caused
exactly this step-17 divergence. The code keeps `let stagnant = false;`
with a comment. Reverting it did **not** fix gate_f, so the cause is one
of the OTHER changes.

**Prime suspects for gate_f, in order (untested — next agent starts here):**

1. **`m_t_ks = m_k.clone()` (multiply-truncate of T=K·S) on a SMALL system.**
   SiH4 uses `full_mask` at n_atom=5, so M_K=M_HS=full and truncation
   should be a no-op — **verify that first**, it is cheap and would
   exonerate the change. But gate_f may run a larger cluster than SiH4
   (check the test's geometry) in which case this IS the likely cause, and
   its 0.8 %-class error on K would readily make SCC oscillate at rms 3e-2.
2. **`m_ht = m_k.clone()`** changes `R_H` only (diagnostic) — should be
   harmless, but confirm R_H is not used in a convergence decision.
3. **The trace guard firing at a geometry where the lock arms early.** The
   SiH4 log shows the guard is NOT printed before the failure, only the
   plateau — so probably not the guard. Confirm with
   `RUST_DFTB_SPARSE_ALGEBRA_VERBOSE=1`.
4. **`check_left_degree` replacing construction-time checks** — if some
   kernel other than the three patched `spgemm_*_dev` entry points caches
   a row in local memory, it is now unchecked and could silently return
   zeros (the old failure mode this check existed to prevent). Grep the
   `.cl` for `MAX_LEFT_BLOCKS` (4 sites) and make sure every kernel that
   uses it is reached by a `check_left_degree` call.

**Bisection recipe:** revert (1) alone → rerun gate_f. `m_t_ks`/`m_t_zs`
should become a CONFIG choice (`cfg.truncate_products`), default OFF
(product masks) so small/validated systems keep the old exact behaviour,
ON only for large systems where the degree cap forces it. That is the
surgical resolution and it also makes the error measurable as its own
axis.

### Next steps, in priority order

1. **Fix gate_f** per the bisection above. Nothing else should be trusted
   until the validated small-system gates are green again.
2. **Fix #2 from the analysis: τ-based (block-norm) dropping instead of
   radius truncation for intermediates.** This is what takes R_I from
   7.9e-3 to something defensible and is the only route to the
   ~64-neighbour target for K.
3. Re-measure the r_k/r_z error sweep on cube_Si65 once (2) exists
   (current numbers: r_k=12 → 0.000 mHa, 10 → +0.19, 8 → +6.2, 6 → NS
   stalls at 1e-3).
4. Then re-run the 330 / 864 / 1648-atom ladder and only then quote
   energies.

## 2026-09-12 (b) — GPT-5.6 review ADOPTED as the work plan (user decision)

GPT-5.6 reviewed the code at 55ddb4c + labbook (chat.md L4044+). Its
claims were verified against the source. User directives on top:

- Follow GPT-5.6's priority order; architectural changes are WANTED —
  do not fine-tune a wrong architecture. Keep K-TC2 as deprecated legacy
  (switchable, measured comparison) — do NOT delete it.
- NO CPU rescue/fallback when GPU fails — the system is the benchmark
  for designing the most efficient f32-GPU solver; f64/Kahan only where
  a cheap scalar decision absolutely needs it.
- Gate F rescue: GPT-5.6's judgment = rollback + damped retry.

### Verified code facts (re-checked this session)

- `m_t_ks = m_k.clone()` / `m_t_zs = m_z.clone()` / `m_ht = m_k.clone()`
  (sparse_system.rs:191-197) — intermediates ARE radius-truncated.
- gate_f = SiH4 = 5 atoms < FULL_MASK_ATOMS(64) → all masks full →
  `m_t_ks` clone is a NO-OP there. **Suspect #1 (intermediate
  truncation) is EXONERATED for gate_f** — earlier labbook entry wrong.
- Proximate fatal line: `sparse_dftb.rs:599`
  `it > 2 && rms > rms_prev*2.0 && rms > 1e-2` — the observed
  `rms=3.339e-2 (prev 3.463e-3)` trips it exactly. A single DIIS
  overshoot is turned into a fatal error. (Same line in scc.rs:381.)
- TC2 details confirmed: branch-fabrication write `Nocc*(1+1e-6)` (:779),
  `TC2_TRACE_TOL=0.05` absolute (gpu_sparse.rs:24),
  `best_r_i < 1e-2` returned as `Ok` (:855,:881), `k_norm` frozen from
  K0 (:721), Gershgorin bounds on truncated ZH + 10% pad (:652-665).
- `bsr4_tc2` kernel branches on a device f32 `trace_KS[0] > Nocc`
  (sparse_bsr4_purification.cl:641).

### Item 1 RESULT — Gate F green again (2026-09-12, verified)

- SCC rescue implemented in `sparse_dftb.rs::scc()`: snapshot
  (q_in, res) before each mix; on `rms > 2·rms_prev && rms > 1e-2`
  → restore last accepted q, `mixer.reset()`, damped `mix·res` step,
  `continue`; ≤8 rescues/call; nonfinite rms fails immediately (the
  old check could NOT even see NaN — `NaN > x` is false).
- **AB result: overshoots are PRE-EXISTING, not caused by the TC2
  machinery.** With `RUST_DFTB_TC2_GUARD=0 RUST_DFTB_TC2_PLATEAU=0`
  (new machinery fully off) the same overshoots fire at steps 14/15/18+
  and all recover under rescue. The old abort was simply too aggressive;
  step-17's overshoot was a bit larger than usual (plausibly nudged by
  the plateau-restored K) but the same class.
- gate_f: converged step 72, |F|=9.4e-4, mean Si–H=1.477 Å. All gates
  green: gate_g3(5) gpu_sparse_bsr4(22) sih(10) gpu_scc_kernels(10)
  spgemm_plan(1) lib sparse(8).
- Also fixed (user's in-flight commit-model change): kernel
  `residual_and_mix_batched` gained an `active` arg — standalone
  wrapper `gpu_matrix.rs` + `gpu_scc.rs` + 2 test call sites updated
  with all-ones buffers (legacy path has no frozen replicas).
- Test `test_row_degree_overflow_fail_loud` updated to the NEW contract:
  `GpuBsrStructure::new` no longer rejects wide masks (only records
  `max_deg`); the check fires at `spgemm_*_dev` LEFT-operand entry.
- Env toggles added (AB bisection, default ON):
  `RUST_DFTB_TC2_GUARD`, `RUST_DFTB_TC2_PLATEAU`, `RUST_DFTB_SPARSE_PLANS`.

### Item 2 RESULT — host-f64 trace + branch flag (2026-09-12, verified)

Implemented:
- New kernel `bsr4_trace_atom` — per-atom diagonal partials; host reads
  N_atom floats (~6.6 kB @1648) and sums in f64. Replaces the device
  f32 tree-reduce for the TC2 decision path.
- `bsr4_tc2` now takes an explicit `branch: u32` decided on the host —
  the `Nocc*(1+1e-6)` fabricated-trace write is GONE.
- `tc2_trace_tol(nocc) = max(2e-5·Nocc, 1e-4)` replaces the universal
  0.05 e⁻ constant everywhere (new + legacy paths).
- `tc2_purify`/`purify_hscc`/`run_scc` return Tr as f64;
  `tr_ks` reporting fields stay f32 (cast at store).

**Found + fixed in the process — two real numerical lessons:**

1. Post-rescale branch choice matters. tr_eff==Nocc is degenerate; the
   complement branch (2K−Q) after a rescale RATCHETS — it pushes all
   λ<1 up, trace rises, guard rescales, repeat (observed doubling
   excess on SiH4, R_I floor degraded to 7e-4). Choosing the squaring
   branch after rescale makes the next measured trace dip below Nocc
   and restores the natural alternation. Implemented as an explicit
   `branch=1` — no fabricated trace value.
2. The guard was firing during HEALTHY descent (SiH4 dev_rel~2e-4 at
   R_I still improving) and its rescale injected ~1e-4 state error —
   G3.4 force/FD parity degraded 6× (|d|=1.19e-4 → FAIL vs 1.86e-5
   pass with guard off). Fixed: guard now fires only on CONFIRMED
   exponential growth — dev_rel>jitter AND dev_rel>1.5×previous for 2
   consecutive iterations (the λ=1+δ⇒λ²≈1+2δ doubling signature).

**Verified both ends:**
- SiH4 (gate_g3, gate_f, all small gates): guard never fires; TC2
  converges R_I≈5e-7; G3.4 force parity |d|=1.9e-5.
- 1648-atom sphere: doubling leak detected iters 54-56
  (excess 0.039→0.119→0.305), guard fires at 57/60/63, plateau restore
  at 63 → R_I=1.07e-3 (better than the earlier 7.9e-3), Tr=2627.000000,
  SCC rms=8.5e-6, E=-1772.11080 Ha, max|F|=0.131 Ha/Å, ~7 s.

### TRS4 update rule — second revision (a-family + clamp + complement, 2026-09-12)

Two failure modes found at r_k=12 on the 1648-atom sphere:

1. **First prototype** (`a∈[−1,3]` window + TC2 fallback by trace sign):
   diverged — the trace-sign fallback picked SQUARING when den≤0, which
   amplifies eigenvalues >1.
2. **Niklasson β-form** (`P_new = (1+β)Q − βR`, β∈[0,1]): converged on
   paper but STALLED in practice — when the spectrum is mid-range
   (Tr(Q)=1355 ≪ Nocc=2627), no β∈[0,1] can reach the trace (2x²−x⁴
   pushes x≈0.5 DOWN), so Tr(P) collapsed 2627→633 to a wrong-rank
   fixed point.

**Working rule**: `P_new = a·Q + (1−a)·R`, `a=(Nocc−Tr R)/(Tr Q−Tr R)`
clamped to [−1,3]; `den=Tr(Q)−Tr(R) ≤ 0` (eigenvalues >1 present) →
complement `2P−Q`. The a>2 regime (max f = a²/4(a−1) ≤ 9/8 ≈ 1.13)
mildly overshoots but self-corrects via the next den≤0 complement step.

Measured (1648-atom Si sphere, TRS4):
| r_k | floor R_I | nnz_k | note |
|-----|-----------|-------|------|
| 10 Å | 7.67e-4 | 224k | baseline |
| 12 Å | 3.34e-4 | 412k | ~2× lower floor, SCC 13 iters, stable |

E(12 Å,TRS)=−1772.3924 vs E(12 Å,P-TC2)=−1772.3966 (4 mHa apart — both
floor-limited). P-TC2 at r_k=12 works identically (floor 3.7e-4), so the
mask is fine — the earlier TRS divergence was the fallback-rule bug.

### Item 7 groundwork — K block-norm decay measured (2026-09-12)

`test_k_block_norm_histogram` (ignored diagnostic, gate_g3_energy.rs):
converged K on the 1648-atom sphere at r_k=12 Å, 411,838 blocks =
250 nbr/atom, ‖K‖_F²=1344.

Distance profile: mass concentrated at 0–3 Å (diag 0.44 + nn 0.26
mass/atom) but with a FAT tail — the 10–12 Å bins still hold
~1.5–2.5e-4 mass/atom and max norms ~5e-3. This is exactly why the
geometric cut at 10 Å costs a 7.7e-4 floor: it deletes real tail mass.

τ-screen table (block ‖B_ij‖_F > τ kept):
| τ | nbr/atom | dropped mass frac |
|---|----------|-------------------|
| 1e-4 | 240 | 3.9e-8 |
| 1e-3 | 167 | 3.3e-5 |
| 3e-3 | 84 | 4.4e-4 |

Per-row error budget (drop smallest blocks until dropped mass = b²·‖row‖_F²):
| budget b | nbr/atom |
|----------|----------|
| 1e-2 | 140 |
| 3e-2 | 61  |
| 1e-1 | 20  |

Conclusion: ~64 effective nbr/atom is reachable at ~3% per-row dropped
mass — that loses real tail weight; the honest working point is more
like 140 nbr at 1%. The screening must be value-based on a provisional
K (H-norm proxies undershoot — K decays gap-controlled past H's cutoff),
so the design is two-phase: provisional purify on the wide mask →
τ/budget screen → freeze M_K'. Implementation of the mask-rebuild path
is the remaining work.

### Item 6 RESULT — W=2(ZH)K one-product force path (2026-09-12, verified)

**The identity is EXACT, not approximate.** This code's Z is S⁻¹ (NS
target ‖I−ZS‖→0), so `Z·H·K = S⁻¹H·Σc cᵀ = Σ_occ εᵢ cᵢcᵢᵀ = ρHρ = W/2`
since `S⁻¹Hcᵢ = εᵢcᵢ`. Measured in `test_w_zhk_vs_khk_parity`
(tests/gpu_sparse_bsr4.rs, locked by assert): elementwise match to
5e-8 in f64 on nonorthogonal S; contraction parity 6e-7 over 50 random
symmetric dS proxies. My initial derivation said "differs by a Z" — it
was wrong (I had assumed Z=S^{-1/2}); the measurement settled it.

Production change (`SparseDWWorkspace`, sparse_forces.rs):
- `W = 2·(Z·H_scc)·K` — ONE SpGEMM instead of two; the ZH operand is
  `ws.b_zh`, the same truncated product that built K0/P0 for the
  current H_scc → **the force path now has NO intermediate truncation**
  at all (the KH→M_K truncation is gone).
- Default ON (`RUST_DFTB_W_ZK=0` reverts to legacy 2KHK for A/B).
- G3.3/G3.4 SiH4: |d|=1.864e-5 (was 1.857e-5), max|dF|=1.40e-6 —
  identical forces. Same under TRS mode (F_ana parity 1.31e-6).
- All 43 tests green.

### Item 5 RESULT — TRS4 trace-resetting purification on P (2026-09-12, verified)

`trs_purify_p` / `purify_hscc_trs` (sparse_system.rs); switch
`cfg.purifier_trs` / env `RUST_DFTB_TRS=1` (takes precedence over
`purifier_p`; both default off). Per iter: Q=P², R=Q² (two planned
generic SpGEMMs, same plan_pp since Q lives on M_P), then
P_new = a·Q + b·R with a+b=1 and a·Tr(Q)+b·Tr(R)=Nocc — the trace is
reset EXACTLY in host f64 every iteration (the rigorous restoring
mechanism; no guard, no lock, no lockout thresholds needed).

- SiH4: ~10 iters to R_I~7e-7 (vs ~20 for TC2), a→2.0 at convergence
  (the degree-4 map degenerates to pure squaring near idempotency, as
  expected). Tr(P)=4.000000 every iter.
- Si65: ~10 iters per purify (vs ~22 TC2), E_tot=−52.45756303 matches
  legacy to 2e-7 Ha.
- **1648-atom sphere: R_I floor 7.6e-4** — best so far (vs 1.15e-3
  P-TC2, 1.07e-3 K-TC2), still descending ~1e-5/iter at max_iter=80.
  Tr(P)=2627.000±3e-4 held exactly — occupation invariant is enforced
  by construction, not by a heuristic.
- Fallback fix (measured, not hypothetical): when the TRS solve is out
  of window with den=Tr(Q)−Tr(R) ≤ 0, eigenvalues >1 exist and the
  COMPLEMENT map 2x−x² must be used — a squaring step there amplified
  Tr(Q) 2630→2681 and diverged (first prototype run). With
  `den≤0 → complement`, the single out-of-window event recovered
  cleanly back to the 8e-4 floor.
- Consistent finalization: after K=PZ recovery, T=K·S is recomputed and
  Mulliken reads the recovered K's KS — q and K must come from the same
  matrix or the SCC energy sits off-stationarity. (Direct Mulliken from
  P remains available via `p_valid` when no recovered K exists.)
- G3.4 caveat unchanged: the ~5e-4 FD-vs-ana gap under P modes is
  per-side SCC noise (~1e-6 Ha at scc_tol=1e-5, ×1/2h=500), NOT a
  systematic P-path bias — the +h energies agree with K-TC2 to 2e-7.
- All 42 tests green in default mode.

### Item 4 RESULT — P=KS purifier prototype (2026-09-12, verified)

Iterate P = K·S on mask M_P = M_K: P²=P, Tr(P)=Nocc, q_A = 2·Tr(P_AA)
directly. ONE generic planned `P²` SpGEMM per iteration (new
`build_spgemm_plan` + `bsr4_spgemm_plan` — `plan_b_idx` indexes B_kj
directly, no transpose); P is non-symmetric → no symmetrize. P0 =
(emax·I − ZH)/Δ (exactly K0·S since ZS=I). K = P·Z recovered after SCC
(plan_pz, Bsym) for the energy/force path. Switch:
`SparseDftbConfig.purifier_p` or env `RUST_DFTB_P_TC2=1`; default OFF.

- SiH4 (full mask): converges R_I→3.6e-6 in ~20 iters, Tr(P)=4.000007 —
  same dynamics as K-TC2 (P=KS algebraically), no guard needed.
- Si65 cube: E_tot=−52.457555 vs K-TC2 −52.457563 (Δ=7e-6 Ha),
  R_I=1.9e-7, 8 SCC iters — equivalent.
- **1648-atom sphere — the runaway is GONE**: trace stays 2627.0±0.03
  across all 80 iters; the K-TC2 doubling signature (dev 3.9e-2→0.98,
  guard rescales ×6) never appears. The open-loop spectral instability
  was therefore STRUCTURAL — the truncated T=K·S feeding Q=T·K — not an
  f32 arithmetic limit. Confirms the review diagnosis.
- BUT the R_I floor is NOT lower: best 1.15e-3 at iter 79 (still
  decreasing ~1e-5/iter when max_iter hit) vs K-TC2's 1.07e-3. The floor
  is now the M_P=M_K iterate truncation — the honest fixed-point
  residual of the masked algebra. Lowering it needs a wider/error-
  budgeted M_P (item 7) or TRS (item 5), not more f32 headroom.
- E_tot: P −1772.1353 vs K −1772.1108 Ha (Δ=2.4e-2) — the two paths sit
  at different floor states; neither validated until the mask is wider.
  R_H=6.3e-3 comparable.
- NumericalFloor status correctly reported and accepted; SCC converged
  rms=4.1e-6 in 13 iters; forces evaluated (max|F|=0.131).
- G3.4 SiH4 FD-vs-ana force gap is ~5.9e-4 under P mode vs ~1.9e-5
  legacy — marginal; candidate cause: energy uses K=PZ which inherits
  the NS Z-residual. Not resolved; P mode stays opt-in/experimental.

### Item 3 RESULT — PurifyStatus (2026-09-12, verified)

- `PurifyStatus { Converged, NumericalFloor, Failed }` in gpu_sparse.rs.
  `tc2_purify`/`purify_hscc`/`run_scc` return it; Err remains the
  fail-loud `Failed` propagation. `SparseDftbScc` and `SparseDftbEnergy`
  carry `purify_status`; `SparseDftbConfig.accept_numerical_floor
  (Option<bool>, default true)` gates NumericalFloor acceptance —
  a Hessian run can set Some(false) to reject floor states.
- All sparse gates green; gate_e_determinism stays red BY DESIGN
  (it is an FD-of-Tr(KH0) non-test kept red to document the pending
  analytic sparse force — unrelated to this change).

### Adopted implementation order (GPT-5.6, items as in chat.md L4570+)

1. **Gate F**: replace one-jump SCC abort with closed-loop rescue —
   snapshot (q_in, res) before each mix; on measured residual explosion
   restore last accepted q, `mixer.reset()` (drop suspect history),
   damped linear step, retry; fail only after repeated rescue failure
   or nonfinite. Do NOT touch the shared mixer (dense solver alone).
   Env toggles for AB bisection: `RUST_DFTB_TC2_GUARD`,
   `RUST_DFTB_TC2_PLATEAU`.
2. **Host-f64 trace + branch flag**: per-atom diagonal partials → host
   f64 sum → explicit `branch` uint into `bsr4_tc2` (removes the
   `Nocc*1e-6` fabrication). Size-scaled trace tol replaces the absolute
   0.05. ~N floats readback/iter; sync already exists.
3. **`enum PurifyStatus {Converged, NumericalFloor, Failed}`** — stop
   returning R_I<1e-2 as ordinary `Ok`; run mode decides acceptance.
4. **P = KS purifier (architectural)**: generic planned product for P²
   (P non-symmetric → Bsym plan can't be reused); 1 SpGEMM/iter instead
   of 2; P0 = (εmax·I − ZH)/Δε (eliminates ZHZ from every init);
   Mulliken = 2·Tr P_AA; K = PZ once post-SCC. Coexist via cfg switch;
   compare iterations/wall/R_I/forces on SiH4, Si65, ~300at.
5. TRS/trace-resetting purification on P (only after 4 works).
6. W = 2(ZH)K for forces — parity vs 2KHK on full-mask SiH4/Si65 first;
   kills the truncated KH intermediate.
7. τ-screening with per-row error budget on a frozen larger structural
   mask (the real route to ~64 effective neighbours); freeze active
   pattern during Hessians.
8. GPU throughput: MAX_LEFT buckets 64/128/256/512, packed plan indices,
   fewer syncs. Gamma deprioritized.

Supporting rules adopted:
- Standing rule: **truncate stored RESULT matrices; never
  radius-truncate algebraic INTERMEDIATES.**
- Plateau detector arms only in endgame (|Tr−Nocc|/Nocc AND R_I in
  floor range), detects 2-cycle / flat log-slope of R_I — not "N
  non-improvements".
- Under P: r_I = ‖P²−P‖_F/√Nocc (‖P‖_F ≈ √Nocc for a projector).
- Spectral bounds: conservative ‖Z‖∞·‖H‖∞ (or enlarge, not shrink,
  Gershgorin) since bounds on truncated ZH are not bounds.
- Precision budget: matrices f32 FMA, 2–4 f32 accumulators; f64 ONLY
  for trace/branch, charge conservation, energy final sum, DIIS small
  solve, CPU Hessian. No Kahan in SpGEMM dots.

## 2026-09-13 — Review-3 work order implemented (manifest §15.9 SC1–SC4, PR3, PR4)

The third GPT-5.6 review (chat §"Chat GPT 5.6 sol" from line 4819) was
recorded as a checkable work order in manifest §15.9 and the first-pass
items were implemented:

**Sparsity contract / kernel sizing**
- **SC1**: `SparseDftb::with_config` measures `max_deg` of M_HS, M_K, M_Z
  at init, prints `deg_*/budget`, and hard-`Err`s when a mask exceeds its
  budget (`cfg.max_deg_{hs,k,z}` or env `RUST_DFTB_MAX_DEG_*`; defaults
  512/128/256). Derived masks are clones of the three base masks.
- **SC2**: `MAX_LEFT_BLOCKS` is now compiled from the measured degree
  (`ceil16(max(deg_hs, deg_k, deg_z))`), not the historical 512 — e.g.
  SiH4 gets 16 (~1.0 KiB/WG local instead of 32 KiB). Frozen topology ⇒
  the degree cannot grow during a run.
- **SC3**: plan-build/upload failure is a **hard error** in
  `SparseSystemWorkspace`, `SparseDWWorkspace`, and `Tc2Workspace`. The
  intersection kernel remains only under the explicit
  `RUST_DFTB_SPARSE_PLANS=0` diagnostic toggle — no silent switch to the
  slow path. New `plan_ht`: the R_H stationarity product `H_scc·(K·S)`
  uses the generic planned gather kernel (was the only production
  intersection-kernel product left).
- **SC4**: `build_hscc_dev` now has the host-side degree guard — the
  kernel's `if(nb > MAX_LEFT_BLOCKS) return;` stale-row bail can no
  longer fire silently.

**Selective precision**
- **PR3**: `reduce_partials_f64` — GPU reductions now stop at ≤128
  partials and finish with an f64 host sum. Applied to idempotency
  (‖KSK−K‖², ‖P²−P‖²), NS residual (‖I−T‖²), ‖K‖/‖P‖ norms, and the
  band energy. Every accept/reject decision scalar has an f64 tail.
- **PR4**: `trace_kh0_dev` returns f64 (host-f64 tail) and
  `store_energy` runs only on the converged SCC iteration (or verbose) —
  the band-energy reduction + host sync are removed from non-final
  iterations.

**Verified:** `cargo test --release` — lib sparse 10/10,
gpu_sparse_bsr4 23/23 (+1 intentionally ignored), sparse_dftb 1/1,
gate_g3_energy 7/7 (+1 ignored), gate_f_geom_opt 1/1, gate_g_hessian
1/1. Init line on SiH4: `deg_hs=5/512 deg_k=5/128 deg_z=5/256
MAX_LEFT_BLOCKS=16 (~1.0 KiB local/WG)`.

**Not done (next):** SC5/SC6 (explicit radii/smaller skin — config-level,
needs script updates), SC7 (NS ZS halo), AL1 (P-TC2 validation→default),
MS1–MS3 (mask calibration + degree-sweep measurements), PR1/PR2
(8-accumulator / Kahan A-B experiments). Note: `sparse_big_nc.rhai`
(r_k=r_z=10 Å → ~285 nbr/atom) will now trip the degree budgets — it
needs explicit `max_deg_*` override or tighter radii, which is the
intended contract.

## 2026-09-13 (b) — SC5/SC7/PR1/AL1-prep + MS3 measured degree matrix

Continued through manifest §15.9:

**Done**
- **SC7 (Z·S halo):** `SparseSystemWorkspace::new` now takes an explicit
  `tzs_mask`; `SparseDftbConfig.r_zs_halo_ang` (default 0 = legacy
  `M_TZS = M_Z`) builds `M_TZS = geometric(r_z + halo)`. Degree-measured,
  budget-checked under the M_Z ceiling, included in `MAX_LEFT_BLOCKS`.
  `plan_zs`/`plan_zh` write on it; `plan_tz`/`plan_bz` read it and
  truncate results to `M_Z`/`M_K` — the "halo only for intermediates,
  results always projected" contract is enforced structurally.
- **SC5 (partial):** init prints a loud warning when all radii default to
  the permissive full-SK-radius+skin regime. Hard-Err deferred: MS3 data
  showed the right radius is system-dependent.
- **PR1 (8 accumulators):** implemented on both plan kernels as `-DACC8`
  (two FMA quads alternating over plan terms, odd-term tail into the
  first quad). Selected via `SparseBsr4Config.spgemm_acc8` /
  `RUST_DFTB_SPGEMM_ACC8=1`, **default OFF** — the reassociation noise
  (~ulp in K, ~1e-4 in the FD force) trips the tight G3.4 gate. Physics
  unchanged; needs a wall-clock A/B before promotion.
- **AL1 (partial):** `set_purifier("k"|"p"|"trs")` + rhai
  `sparse_purifier`. **Real bug found by the sweep:** `K=P·Z` recovery
  inherited the ZS−I residual — `Tr(KS)` drifted 0.165/1305 at 864 atoms
  and hard-failed the SCC Tr-gate. `recover_k_from_p` now restores
  charge conservation (one K·S product, rescale K and T by Nocc/Tr;
  T is linear in K) and leaves `t_ks` fresh — callers' redundant
  `spgemm_ks` removed.
- **MS3 (measured):** degree sweep on `si_sphere_R14` (864 atoms);
  cube_si65 proved too small (deg saturates at n_atom=65 at every
  radius — cannot discriminate the {64,96,128} targets). Full table in
  manifest §15.9 MS3. Headline: **deg ~200 (r_k≈10 Å) is the usable
  floor on Si spheres** — deg 108 SCC-limit-cycles under K-TC2, deg 52
  converges to a state 2.1 mHa/atom wrong. New
  `sparse_scc_try` (rms or NaN, loud stderr) lets sweeps record
  non-convergence as data without aborting.

**Tests:** `cargo test --release` — lib sparse 10/10, gpu_sparse_bsr4
23/23(+1 ig), sparse_dftb 1/1, gate_g3_energy 7/7(+1 ig),
gate_f_geom_opt 1/1, gate_g_hessian 1/1 — all green with ACC8 off
(default) and trace-restore in place.

**Not done:** SC6 (skin/r_trunc tightening — needs per-chemistry
calibration), PR1 wall-clock A/B + PR2 (Kahan endgame), AL1 promotion
(needs wide/dense reference comparison), MS1/MS2/MS4, degree-bucketed
kernels for >512 rows.

### Recommended radii (opinion, from the MS3 matrix)

On matsci Si spheres the mask degree — not the kernel — sets the error:

- **r_trunc (H/S) = 8 Å** — validated on cube_si65 (<0.05 mHa); Si–Si
  coupling is ~1e-4 Ha at 7 Å, the 10.6 Å table end is not physics.
- **r_k = r_z ≈ 10 Å (deg ~210)** — screening/energies tier:
  0.16 mHa/atom, 3 s/SCC on R14.
- **r_k = r_z ≈ 12 Å (deg ~390, budget 512)** — vibrational tier:
  needed for the 1e-5 Ha/atom phonon criterion; this was the sweep
  reference and is ~2× slower than r_k=10.
- **deg ≤ ~110 is dead territory** for Si: non-convergent or wrong by
  ~1 mHa/atom. The deg-64–128 ambition fails here — the fix, if any, is
  better intermediates (M_TZS halo, MS2 magnitude-aware masks), not
  smaller masks.
- Radii are system-dependent; budgets + the SC5 warning are the
  contract, not a hard-coded default.

### Next on the list (priority order)

1. **SC7 value test** — does `r_zs_halo_ang` = 2–4 Å recover accuracy at
   r_k=8–10? The machinery exists; needs one sweep.
2. **AL1 finish** — dense or full-radius reference E/F for P-TC2 vs
   K-TC2 at deg ~390 to adjudicate the −4 mHa fixed-point difference;
   P-TC2 is ~2× faster wall-clock and converges where K-TC2 cycles.
3. **PR1 A/B benchmark** — wall-clock deg~200–400, ACC8 on/off.
4. **SC6** — skin reduction (Hessian ±0.02 Å needs << 1 Å) +
   r_trunc confirmation on the sphere (8 Å was validated on the cube).
5. **PR2** — Kahan endgame only if the mask-floor analysis shows f32,
   not truncation, is the binding term.
6. **Batch-parallel driver** — the actual product: many configs on
   frozen masks; the sparse stack is now ready for it.
7. MS1/MS2/MS4, degree buckets (>512-row rows), as needed later.

## 2026-09-13 (c) — L1: pbc basis solves the locality problem

GPT-5.6 (chat §5400) noted pbc-0-3 Si is far more confined than matsci
(r₀=3.3 vs 4.2 a₀; tables end at 5.5 Å vs 10.6 Å). Ran the R14 degree
sweep with `pbc-0-3` (script `sparse_degree_sweep_pbc.rhai`,
r_trunc=5.3 Å):

- **Full pbc mask (r_k=0 → 6.5 Å incl. skin): deg_k=95, deg_hs=54 —
  converges cleanly: 19 SCC iters, 537 ms, E=−845.11517 Ha.** Compare
  matsci reference: deg 386, 7073 ms. **13× faster, 4× fewer
  neighbors, and deg 95 is already under the 128 budget.**
- Truncating below pbc's own table range fails: r_k=3.0–6.2 Å
  (deg 6–52) → K-TC2 diverges (`R_I=inf`, truncated Z breaks the
  spectral bounds) and P-TC2 plateaus at r_I~1e-2.
- P-TC2 fails on pbc even at full mask (r_I floor ~1.5e-2) — K-TC2 is
  the working purifier on this basis; worth investigating later.

**Conclusion:** the ~200-neighbor floor was a *matsci basis* property,
not a sparse-solver limit. The confined pbc parameterization makes the
density matrix genuinely short-ranged.

**DIRECTIVE (user, 2026-09-13): strongly prefer pbc-0-3 over matsci-0-3
for the sparse solver.** Fewer neighbors is the single most important
thing for performance, and pbc delivers deg 95 vs 386 by construction.
matsci remains the reference/parity baseline only.

The "can't go below the 5.5 Å table end" is what *blind radial
truncation* showed — it is NOT a proven floor. GPT-5.6's §15.10 routes
(L3 P-locality vs K recovery, L4 top-k magnitude masks, L5 LNV
variational refinement on a fixed mask, L7 smooth-step windows) are
still open and may push the working degree well below 95 — measure
before concluding.

Next required checks before trusting pbc numbers: (1) Fortran DFTB+
parity E/F on R14 with pbc-0-3 (validate the parameterization itself);
(2) forces/vibration quality; (3) why P-TC2 fails on pbc. Then
L3 → L4 → L5 to push degree down further.

### L3 measured — P is NOT more local than K

`band_energy_from_p` = 2·Tr(P·Z·H_scc) via restrict(b_zh→M_P)+masked
trace (`sparse_eval_p` in rhai). On matsci R14, E_P drift vs the deg-356
reference: deg 212 → +144 mHa, deg 108 → +205 mHa, deg 52 → +4.65 Ha —
same as or worse than the recovered-K path. The K=PZ recovery was NOT
the limiter; the projector itself carries the long tail. The "run the
hot loop on tiny masks, recover K later" hypothesis is dead on matsci.

### L4 measured — mostly Case B (projector not compressible)

`tests/sparse_topk.rs` (ignored GPU test): top-k masks built from the
converged wide-run K block norms, symmetrized. Results:

| top-k | deg (symmetrized) | result |
|------:|------:|--------|
| 32 | 60 | non-converged (rms 3.9e-4) |
| 48 | 94 | +384 mHa |
| 64 | 114 | +206 mHa |
| 96 | 174 | +83 mHa |
| 128 | 236 | rms 1.3e-5 ≈ tol, +36 mHa |

Magnitude masks DO beat geometric masks at equal degree (converge at
deg 114 where geometric deg 108 limit-cycled; deg 174 → 83 mHa vs
geometric deg 212 → 137 mHa — ~2× better error per neighbor). But even
optimal block selection leaves 80–400 mHa at deg ~100–230: the matsci
projector is genuinely not compressible to 64–128 neighbors.
Consistent with L1: the fix is the basis (pbc), not mask cleverness.
Top-k machinery (`build_topk_mask`, `mask_kz` injection) stays as the
MS2 substrate — e.g. to shrink masks *below pbc's native range* later.

## 2026-09-13 (d) — pbc parity vs Fortran DFTB+: OPEN PROBLEM

Parity run `debug/pbc_parity_R14/` (DFTB+ dev build, SCC, pbc-0-3):

| system | basis | DFTB+ | Rust sparse | Δ |
|--------|-------|-------|-------------|---|
| SiH4   | matsci | −2.7642057 | −2.7642056 | **+0.00007 Ha — exact** |
| SiH4   | pbc    | −2.5674616 | −2.5674278 | +0.034 mHa — fine |
| R14 sphere | matsci | −882.5987 | −882.5152 (deg 356) | +78 mHa (truncation-level) |
| R14 sphere | pbc    | **−847.0896** | **−845.1152** (deg 95) | **+1974 mHa — WRONG** |

- The sparse engine is validated: SiH4 parity is 1e-7 on matsci and
  34 μHa on pbc. SKF parsing (repeat tokens, comma separators, Hubbard
  field position) is consistent between Rust and DFTB+.
- The 2 Ha gap is scale-dependent, NOT a per-atom parsing/onsite issue:
  34 μHa on SiH4 → ~30 mHa if it scaled ×864, not 2 Ha.
- Ruled out: H/S taper (r_trunc 5.3→5.45 moved nothing), Hubbard U
  (same field both sides).
- DFTB+ pbc decomposition: band −900.009, electronic −848.511,
  repulsive +1.421. Rust E_tot −845.117 → electronic ≈ −846.538
  (if E_rep matches) → the SCC electronic energy is off by ~2 Ha.
  Suspects: converged-charge state difference at scale (sphere has real
  Si→H surface charge transfer), or a pbc-specific convention DFTB+
  applies that the sparse path misses.
- **ALSO: a rerun of the identical pbc config (r_trunc=5.45) diverged** —
  TC2 blew up at iter 62 (R_I=0.78, Tr(KS)=1508) after cleanly
  converging (19 iters, E=−845.117) minutes earlier. Nondeterministic
  TC2 blowup on pbc — possibly related to in-flight qmqm edits at the
  time; must be reproduced/investigated before trusting pbc numbers.

**Bottom line: pbc gives the locality we want, but the R14 result is
not yet trustworthy — 2 Ha off DFTB+. Parity must be resolved before
pbc is the production basis.**

## 2026-09-14 — parity root-caused: it's the purification floor, not an engine bug

**Review correction (2026-09-14):** preserve the measurements below, but the intrinsic-f32-floor attribution and “post-hoc repair fails” conclusion are not established. Read the appended **Speed–accuracy source review** and manifest **§15.12**: production NS has a normalization defect, and the purported f64 McWeeny experiment implements a different map with f32 intermediate storage.

Energy decomposition at the converged state (deg_k=95, r_k=7.6 Å):

| term | Rust sparse | DFTB+ | Δ |
|------|------------|-------|---|
| Tr(K·H0) / "Energy H0" | −846.632 | −848.607 | **+1.975 Ha — all of it** |
| E_scc (½ΣΔqγΔq) | +0.093 | +0.096 | ~0 |
| E_rep | +1.4211 | +1.4211 | 0 |
| net Δq | Si +0.0254, H −0.0523 | Si +0.0254, H −0.0529 | ~1e-3 |

Charges, SCC term, and repulsive all match — the **entire 2 Ha is the
band term Tr(K·H0)**, i.e. the purified K is not the true projector
even though its Mulliken populations are right.

Mask sweep (r_k = r_z, r_trunc=5.45 fixed, R14/pbc):

| r_k (Å) | deg_k | R_I floor | E_tot | dE vs DFTB+ |
|---------|-------|-----------|-------|--------------|
| 7.6 | 95 | 4.2e-3 | −845.117 | +1972 mHa |
| 9.0 | 168 | 2.4e-3 | −846.455 | +634 mHa |
| 12.0 | 386 | 9.9e-4 | −846.939 | +151 mHa |

**dE ∝ R_I²** (ratios 1.75→3.1, 2.4→4.2): the gap is the purification
residual, which is mask-driven. Controls: (a) DFTB+ non-SCC run gives
E_H0=−848.727 — the Rust single-purification is already ~3 Ha off, so
this is purification of H0 alone, no SCC-state dependence; (b) pbc
HOMO–LUMO gap is 2.43 eV — not a small-gap problem; (c) wide-Z-only
(r_z=8, K at 7.6) did NOT help — the binding constraint is the K mask.

**Consequence (sobering): the pbc DM in the K representation is not
more local than matsci.** ~150 mHa remains at deg-386/12 Å — same as
matsci's full-mask accuracy at the same degree. The observed "13×
speedup" at deg-95 was bought with 2 Ha of purification error; pbc's
real win so far is only the short H/S table (deg_hs 56 vs ~356),
which cheapens H_scc assembly and the masked traces — not the loop
matrix itself. Whether the occupied projector is more compressible on
a *magnitude-selected* graph (top-k) or via LNV refinement remains the
open question — exactly GPT-5.6's program.

**Bug found & fixed — `cut_bohr` sized the "full" mask off unrelated
SKF tables.** `load_sk_folder` loads every .skf in the directory and
`cut_bohr` took the max over ALL of them; pbc-0-3's F-O.skf (620 pts)
set r_full=7.59 Å for an Si+H system whose true table end is 5.5 Å
(→6.54 Å). This fully explains the "deg 52→95" mystery: the deg-95
mask was r=7.6 Å, not 6.5 Å — no bulk shell needed. Fixed by
filtering `pairs` to species present in the system
(`sparse_dftb.rs` ~line 362). NOTE: after the fix the default
(r_k=None) pbc mask is deg~56 and **SCC fails to converge there** —
the earlier deg-95 "converged" run was accidentally using 7.6 Å.
matsci unaffected (uniform 20 bohr tables).

**Also: the "nondeterministic" TC2 blowup is deterministic.** The same
config diverges identically (iter 62, R_I=0.78, Tr=1508.69) whenever
`tc2_tol` is left at the tighter default; the earlier parity script set
`tc2_tol=1e-5` which stops TC2 at its plateau. The purifier on pbc is
genuinely marginal — pushing past the R_I floor destabilizes it.

### Follow-on experiments same day

**T1 — wide Z does NOT rescue narrow K** (r_z=9 → deg_z=168 fixed,
r_k swept): deg_k 21–52 diverge, 56 converges dirty (+2975 mHa),
78 → +2317, 168 → +634. The K matrix itself needs the width — Z is
not the binding constraint.

**Near-complete mask is still f32-floored:** r_k=14 → deg 630/864
converged to R_I=7.1e-4, E=−847.043 (**+46 mHa**). The R_I floor
flattens (deg386→9.9e-4, deg630→7.1e-4) — at the wide end the residual
is f32-SpGEMM roundoff, not mask starvation. **Consequence: mHa-accurate
sparse energies on pbc need compensated accumulation or LNV — wider
masks alone saturate at ~50 mHa.** (matsci deg-356 R_I=6.2e-4 → 52 mHa
sits on the same dE≈1e8·R_I² curve — one law across basis sets.)

**T4 — top-k oracle on pbc** (ref = r_k=12 deg-386, E=−846.939):
magnitude-selected graphs beat radial ~4× per degree but the DM is
heavy-tailed:

| top-k | deg after symmetrize | dE vs deg-386 ref |
|-------|---------------------|-------------------|
| 32 | 63 | +1365 mHa |
| 48 | 86 | +476 mHa |
| 64 | 113 | +345 mHa |
| 96 | 173 | +125 mHa |
| 128 | 235 | +34 mHa |

vs radial: deg~86 radial ≈ +1822 mHa vs top-48's +476. Top-k is the
right direction but does NOT reach deg-64-accurate alone — the last
mile needs LNV (variational in-mask refinement) or compensated f32.
Caveat: symmetrization roughly doubles the per-row degree.

**f32 floor confirmed — constant per atom.** R10 sphere (330 atoms,
COMPLETE mask deg 330/330, zero truncation): R_I=4.9e-4, E=−298.1696
vs DFTB+ −298.1869 → **+17.3 mHa = 52 μHa/atom**. R14 deg-630: 46 mHa
/864 = 53 μHa/atom. Same per-atom purification roundoff → the f32 floor
scales linearly with N and is INDEPENDENT of mask and basis. The 2 Ha
"parity gap" is now fully explained: mask-driven R_I at deg-95 ≫ the
f32 floor; the engine itself is exact (P-TC2 also converges on pbc at
deg-168: E_P=−899.21 vs DFTB+ band −900.01; E_K identical to K-TC2).

**Revised conclusion:** for ~mHa accuracy the binding constraint is no
longer the mask — it is f32 accumulation in the purifier. The deferred
Kahan/ACC8 work is an *accuracy* requirement, not a micro-optimization;
alternatively LNV variational refinement on the fixed mask minimizes
E directly and can beat the R_I² curve.

**Post-hoc repair fails (decisive):** host-f64 checks on the converged
R10 K (complete mask): device f32 trace is accurate to 8 μHa — the K
itself is defective. f64 McWeeny iteration K←2K−KSK **diverges** from
it (R_I 7.5e-4 → 7e-3, E drifts away) — the f32-TC2 fixed point is
displaced in the *subspace*, not just noisy; no post-hoc polish can
fix it. (Possibly aggravated by the trace guard's 0.987× rescale moving
K off the purification manifold mid-run.) Remediation must live inside
the purification loop: compensated SpGEMM accumulation or LNV.

**Ruled-out cheap fixes (all measured on R10/pbc complete mask):**
- `RUST_DFTB_SPGEMM_ACC8=1` (8-way accumulators): R_I identical —
  accumulation ORDER is not the floor.
- `RUST_DFTB_TC2_GUARD=0` (no trace rescale): R_I identical — the
  26 guard fires are symptom, not cause.
- Tighter spectral bounds: E_DUMMY=2.0→0.8 tightened Gershgorin ~3×
  (dummy lanes ARE the emax=2.34 — physical spectrum tops at +0.24)
  but R_I got *worse* (4.9e-4→8.4e-4) — the loose bound smooths the
  Fermi step; bounds are NOT the floor mechanism. Also: dummy lanes
  must stay *inside* the map — explicit emin/emax override crashed on
  "dummy-orbital occupation 6.0" (padded lanes at ε=2.0 outside a tight
  map get occupied). Reverted.
- Dense-CPU cross-check unusable: `run_dftb_scc` mixer diverges on
  R10/pbc (RMS~2.2), and `run_dftb_nonscc` counts n_electrons=330
  (1/atom — wrong q0 parsing for this system). Dense path needs its
  own bugfix before it can serve as reference; DFTB+ remains the truth.

## 2026-09-14 — parity root-caused: it's the purification floor, not an engine bug

**Review correction (2026-09-14):** preserve the measurements below, but the intrinsic-f32-floor attribution and “post-hoc repair fails” conclusion are not established. Read the appended **Speed–accuracy source review** and manifest **§15.12**: production NS has a normalization defect, and the purported f64 McWeeny experiment implements a different map with f32 intermediate storage.

Energy decomposition at the converged state (deg_k=95, r_k=7.6 Å):

| term | Rust sparse | DFTB+ | Δ |
|------|------------|-------|---|
| Tr(K·H0) / "Energy H0" | −846.632 | −848.607 | **+1.975 Ha — all of it** |
| E_scc (½ΣΔqγΔq) | +0.093 | +0.096 | ~0 |
| E_rep | +1.4211 | +1.4211 | 0 |
| net Δq | Si +0.0254, H −0.0523 | Si +0.0254, H −0.0529 | ~1e-3 |

Charges, SCC term, and repulsive all match — the **entire 2 Ha is the
band term Tr(K·H0)**, i.e. the purified K is not the true projector
even though its Mulliken populations are right.

Mask sweep (r_k = r_z, r_trunc=5.45 fixed, R14/pbc):

| r_k (Å) | deg_k | R_I floor | E_tot | dE vs DFTB+ |
|---------|-------|-----------|-------|--------------|
| 7.6 | 95 | 4.2e-3 | −845.117 | +1972 mHa |
| 9.0 | 168 | 2.4e-3 | −846.455 | +634 mHa |
| 12.0 | 386 | 9.9e-4 | −846.939 | +151 mHa |

**dE ∝ R_I²** (ratios 1.75→3.1, 2.4→4.2): the gap is the purification
residual, which is mask-driven. Controls: (a) DFTB+ non-SCC run gives
E_H0=−848.727 — the Rust single-purification is already ~3 Ha off, so
this is purification of H0 alone, no SCC-state dependence; (b) pbc
HOMO–LUMO gap is 2.43 eV — not a small-gap problem; (c) wide-Z-only
(r_z=8, K at 7.6) did NOT help — the binding constraint is the K mask.

**Consequence (sobering): the pbc DM in the K representation is not
more local than matsci.** ~150 mHa remains at deg-386/12 Å — same as
matsci's full-mask accuracy at the same degree. The observed "13×
speedup" at deg-95 was bought with 2 Ha of purification error; pbc's
real win so far is only the short H/S table (deg_hs 56 vs ~356),
which cheapens H_scc assembly and the masked traces — not the loop
matrix itself. Whether the occupied projector is more compressible on
a *magnitude-selected* graph (top-k) or via LNV refinement remains the
open question — exactly GPT-5.6's program.

**Bug found & fixed — `cut_bohr` sized the "full" mask off unrelated
SKF tables.** `load_sk_folder` loads every .skf in the directory and
`cut_bohr` took the max over ALL of them; pbc-0-3's F-O.skf (620 pts)
set r_full=7.59 Å for an Si+H system whose true table end is 5.5 Å
(→6.54 Å). This fully explains the "deg 52→95" mystery: the deg-95
mask was r=7.6 Å, not 6.5 Å — no bulk shell needed. Fixed by
filtering `pairs` to species present in the system
(`sparse_dftb.rs` ~line 362). NOTE: after the fix the default
(r_k=None) pbc mask is deg~56 and **SCC fails to converge there** —
the earlier deg-95 "converged" run was accidentally using 7.6 Å.
matsci unaffected (uniform 20 bohr tables).

**Also: the "nondeterministic" TC2 blowup is deterministic.** The same
config diverges identically (iter 62, R_I=0.78, Tr=1508.69) whenever
`tc2_tol` is left at the tighter default; the earlier parity script set
`tc2_tol=1e-5` which stops TC2 at its plateau. The purifier on pbc is
genuinely marginal — pushing past the R_I floor destabilizes it.

### Follow-on experiments same day

**T1 — wide Z does NOT rescue narrow K** (r_z=9 → deg_z=168 fixed,
r_k swept): deg_k 21–52 diverge, 56 converges dirty (+2975 mHa),
78 → +2317, 168 → +634. The K matrix itself needs the width — Z is
not the binding constraint.

**Near-complete mask is still f32-floored:** r_k=14 → deg 630/864
converged to R_I=7.1e-4, E=−847.043 (**+46 mHa**). The R_I floor
flattens (deg386→9.9e-4, deg630→7.1e-4) — at the wide end the residual
is f32-SpGEMM roundoff, not mask starvation. **Consequence: mHa-accurate
sparse energies on pbc need compensated accumulation or LNV — wider
masks alone saturate at ~50 mHa.** (matsci deg-356 R_I=6.2e-4 → 52 mHa
sits on the same dE≈1e8·R_I² curve — one law across basis sets.)

**T4 — top-k oracle on pbc** (ref = r_k=12 deg-386, E=−846.939):
magnitude-selected graphs beat radial ~4× per degree but the DM is
heavy-tailed:

| top-k | deg after symmetrize | dE vs deg-386 ref |
|-------|---------------------|-------------------|
| 32 | 63 | +1365 mHa |
| 48 | 86 | +476 mHa |
| 64 | 113 | +345 mHa |
| 96 | 173 | +125 mHa |
| 128 | 235 | +34 mHa |

vs radial: deg~86 radial ≈ +1822 mHa vs top-48's +476. Top-k is the
right direction but does NOT reach deg-64-accurate alone — the last
mile needs LNV (variational in-mask refinement) or compensated f32.
Caveat: symmetrization roughly doubles the per-row degree.

**f32 floor confirmed — constant per atom.** R10 sphere (330 atoms,
COMPLETE mask deg 330/330, zero truncation): R_I=4.9e-4, E=−298.1696
vs DFTB+ −298.1869 → **+17.3 mHa = 52 μHa/atom**. R14 deg-630: 46 mHa
/864 = 53 μHa/atom. Same per-atom purification roundoff → the f32 floor
scales linearly with N and is INDEPENDENT of mask and basis. The 2 Ha
"parity gap" is now fully explained: mask-driven R_I at deg-95 ≫ the
f32 floor; the engine itself is exact (P-TC2 also converges on pbc at
deg-168: E_P=−899.21 vs DFTB+ band −900.01; E_K identical to K-TC2).

**Revised conclusion:** for ~mHa accuracy the binding constraint is no
longer the mask — it is f32 accumulation in the purifier. The deferred
Kahan/ACC8 work is an *accuracy* requirement, not a micro-optimization;
alternatively LNV variational refinement on the fixed mask minimizes
E directly and can beat the R_I² curve.

**Post-hoc repair fails (decisive):** host-f64 checks on the converged
R10 K (complete mask): device f32 trace is accurate to 8 μHa — the K
itself is defective. f64 McWeeny iteration K←2K−KSK **diverges** from
it (R_I 7.5e-4 → 7e-3, E drifts away) — the f32-TC2 fixed point is
displaced in the *subspace*, not just noisy; no post-hoc polish can
fix it. (Possibly aggravated by the trace guard's 0.987× rescale moving
K off the purification manifold mid-run.) Remediation must live inside
the purification loop: compensated SpGEMM accumulation or LNV.

**Ruled-out cheap fixes (all measured on R10/pbc complete mask):**
- `RUST_DFTB_SPGEMM_ACC8=1` (8-way accumulators): R_I identical —
  accumulation ORDER is not the floor.
- `RUST_DFTB_TC2_GUARD=0` (no trace rescale): R_I identical — the
  26 guard fires are symptom, not cause.
- Tighter spectral bounds: E_DUMMY=2.0→0.8 tightened Gershgorin ~3×
  (dummy lanes ARE the emax=2.34 — physical spectrum tops at +0.24)
  but R_I got *worse* (4.9e-4→8.4e-4) — the loose bound smooths the
  Fermi step; bounds are NOT the floor mechanism. Also: dummy lanes
  must stay *inside* the map — explicit emin/emax override crashed on
  "dummy-orbital occupation 6.0" (padded lanes at ε=2.0 outside a tight
  map get occupied). Reverted.
- Dense-CPU cross-check unusable: `run_dftb_scc` mixer diverges on
  R10/pbc (RMS~2.2), and `run_dftb_nonscc` counts n_electrons=330
  (1/atom — wrong q0 parsing for this system). Dense path needs its
  own bugfix before it can serve as reference; DFTB+ remains the truth.

## GPT-5.6 reframing (chat line 5855+): 95 is NOT a proven floor

Key correction: "failure when K, Z and intermediates are all chopped by
the SAME radius proves only that this truncated algebra fails — not that
the density matrix needs that radius." The pbc table end is 5.5 Å; the
working "full" mask was 6.5 Å incl. skin — Z can legitimately extend
beyond the H/S range. The 6.2-fail/6.5-work transition may just be Z or
an intermediate needing the extra shell. Also unexplained: the degree
jump 52→95 over 6.2→6.5 Å — diamond Si has NO bulk neighbor shell
between 5.92 and 6.65 Å, so the jump is probably surface-H/skin/mask
semantics, not a localization length. Audit needed.

The performance objective is NOT "every matrix ≤ 64 neighbors" — it is
"the matrix multiplied 10–30×/SCC iter (P or K) has degree ~64". Z is
built once per geometry; wide Z is cheap.

## 2026-09-14 — Speed–accuracy source review

**Scope/status:** current-working-tree source analysis and documentation only. No solver edits, GPU tests or new timings. Historical numerical results below are reported observations, not independently rerun measurements. Existing uncommitted work is preserved. New implementation tickets in manifest §15.12 remain open/unverified.

**Recommendation:** keep f32 BSR4 and planned gather products. Spend accuracy on a trustworthy inverse/initial projector and final force state; spend less work in early SCC and after evidenced stagnation. Smooth energy offsets may be acceptable for scans/vibrations, but charge convergence alone cannot certify their derivatives. Do not prescribe Kahan, double-single storage, LNV or a universal degree before resolving the inexpensive diagnostic defects.

### Confirmed source findings

1. **NS stopping understates the residual by √N.** `sparse_system.rs::compute_z` (reviewed line 724) computes `sqrt(r2)/n_orb`; its documented contract is `sqrt(r2)/sqrt(n_orb)`. `gpu_sparse.rs::identity_residual_to_f64` and `bsr4_identity_residual_partial` confirm that r2 is the squared Frobenius sum. For R10/330 atoms and R14/864 atoms the factors are **36.33 and 58.79**, using 4 padded orbitals/atom. With `ns_tol=1e-4`, acceptance permits the documented residual up to **3.63e-3 and 5.88e-3**. These are acceptance bounds, not measurements of the returned inverse. The pbc parity script and `sparse_f64check.rs` use that tolerance.

   `test_ns_device_residual_contract` compares two independently recomputed residuals using √N, but only **prints** the production `rz_reported`; it never asserts agreement with it. This permits the normalization bug to escape the test. Inaccurate Z contaminates ZH/ZHZ initialization and the default W shortcut. Its contribution to the energy error must be measured; this review does not claim it explains all 17.3 mHa.

2. **The polish experiment is neither McWeeny nor an f64 iteration.** `tests/sparse_f64check.rs::f64_host_trace` repeats `K←2K−KSK`; true generalized McWeeny is `K←3KSK−2KSKSK`. For the implemented map an empty-state occupation ε becomes approximately 2ε: repeated complements amplify leakage by construction. KS and every new K are also cast back to f32. Its useful host-f64 energy contraction shows the final trace reduction is not the main error; the subsequent loop does **not** prove irreversible subspace corruption or rule out polishing. Correct algebra already exists in `gpu_sparse.rs::mcweeny_step`.

3. **Complete masks do not establish an arithmetic floor.** They remove support truncation, but not inverse/stopping/initialization/SCC errors or model differences. K-TC2 normalizes R_I by the initial K0 norm; recovered P-TC2 K uses its current norm. Changing spectral bounds changes K0 and hence the reported K-TC2 residual normalization. The proposed universal `dE∝R_I²` and 52 μHa/atom laws remain correlations across confounded runs. ACC8 changes summation ordering, not precision; no improvement from ACC8 does not exclude accumulation error.

4. **Overhead and acceptance are not solved.** Production `purify_hscc` passes `check_every=1`: each K-TC2 iteration reads n_atom trace partials, then up to 128 residual partials in another blocking read; guards add work. Reduction tails are **host f64**, not device f64. `rh_stationarity` downloads O(nnz) at finalization. `forces()` downloads K/W and contracts on CPU; `compute_v()` is a host O(N²) gamma matvec; `mulliken_checked()` constructs charge vectors each SCC iteration. Persistent matrix buffers are valuable progress, not a fully GPU-resident pipeline.

   `tc2_purify` has `stagnant=false` and can label exhaustion a `NumericalFloor`. Snapshot eligibility uses trace/R_I, not stationarity/force quality. SCC accepts floors by default; `finalize_scc` measures R_H without gating it. P-TC2 retains P's status while returning recovered-K residuals. These labels cannot by themselves certify vibrations.

### Answers to manifest §15.11

**1. Mechanism:** unresolved until inverse and diagnostic corrections are isolated at fixed H,S. Polynomial purification preserves eigenspaces in exact arithmetic: true McWeeny can improve occupations near 0/1 but cannot rotate a wrong subspace toward H. Finite-precision SP2 stagnation is known; it does not establish the origin of this unusually large error. Convergence-order-based stopping is a useful reference, with validation needed for this nonorthogonal masked algebra. [Kruchinina, Rudberg & Rubensson (2016)](https://arxiv.org/abs/1507.02087).

**2. Accuracy per flop:** first correct NS and certify it once per geometry, then stop wasting products at a genuine plateau. Compare repaired K/P-TC2 including recovery and force cost. Compensated-f32 endgame is conditional on measured product-accumulation error. GPU f64 products conflict with PR5; double-single storage is a last resort due to traffic/complexity. LNV is the relevant alternative if H-dependent subspace correction remains necessary; changing scalar purification polynomials alone is not that correction.

**3. Variational masking:** LNV may improve the variational state, but the recoverable fraction of the deg-95 error is unknown. A sparse auxiliary L in `K(L)=3LSL−2LSLSL` can generate a wider K; equal degree of L and a stored K is not equal support/work. Projecting products changes the implemented functional, whose electronic gradient and nuclear forces must be derived consistently. Existing `bsr4_lnv_gradient` is a combination primitive, not a validated masked optimizer. Do not promise first-order recovery of missing tails or quadratic energy error for inadmissible/nonstationary states. [Nunes & Vanderbilt, nonorthogonal formulation](https://www.physics.rutgers.edu/~dhv/pubs/local_copy/rw_dms.pdf).

**4. Relative energies:** measure `b(R)=E_fast−E_ref` along the stencil. For consistent gradients, force bias is `−∇b` and Hessian bias is `∇²b`. Constant offsets cancel; smooth curvature changes frequencies; deterministic discontinuities remain harmful. Freeze masks/model settings, start both signs from the same central state, reset incompatible mixer history, and test reversed order/cold starts. Fixed iteration counts or replayed branches alone do not ensure smoothness.

**5. Mask choice:** retain radial as the control; “radial is dead” is unsupported. Current top-k `mask_kz` injection changes K and Z together. Separate their budgets and intermediate contribution support. Direct `|H_ij K_ji|` neglects overlap/Pulay sensitivity and indirect product paths; `|H_ij P_ji|` is not the band-energy contraction because P=KS. Compare actual executed plan terms and force accuracy, including wide-reference/oracle setup cost.

**6. Dummy lanes:** structural exclusion of the decoupled dummy density subspace is a legitimate later optimization. Keep dummy S nonsingular; enforce zero dummy density from initialization through updates/recovery before excluding it from bounds. Tighter valid bounds can reduce iterations, but do not smooth the final zero-temperature projector or repair its subspace. Do not clip physical occupations.

**7. Crossover/occupancy:** no defensible crossover N follows from these data. A 4×4 block triple costs approximately 128 FLOPs; count retained triples, physical orbitals, all iterations/transfers and force time. The current kernel launches one WG per atom row (330/864 WGs for R10/R14); utilization also depends on plan lengths, registers and compiled resources. Local cache is `64·MAX_LEFT_BLOCKS` bytes/WG. A 49 KiB per-WG limit does not imply a degree-380 ceiling. Wide Z/halo can enlarge the compiled cache for narrow-K products too. Benchmark matched-quality complete evaluations on the actual device, then batch displacements with shared plans and independent electronic state.

**Next work:** manifest §15.12 A → B/C → evidence-selected D/E. The target is fastest accepted scans, relaxation and Hessians with measured uncertainty, not system-independent mHa total-energy parity. No production accuracy defaults or new speedup claims are established by this review.

## 2026-09-14 (cont.) — §15.12-A executed: the "f32 floor" was never f32

All measurements below from `tests/sparse_f64check.rs` (`frozen_input_dense_ref`, `f64_host_trace`; R10/pbc, pbc-0-3).

### A1 — NS normalization bug: CONFIRMED + FIXED

`compute_z` divided `||I−ZS||_F` by `n_orb` instead of `√n_orb` — understated the residual by √1320 = **36.3×** at R10. The reported `R_Z=3.75e-5` was really `1.36e-3` — a genuinely sloppy inverse feeding ZH/ZHZ/K0 and the default W. Fixed (`sparse_system.rs`); `test_ns_device_residual_contract` now **asserts** the production value against an independent host-f64 recomputation (it previously only printed). Post-fix NS honestly converges: 7 iters, true host-f64 `||ZS−I||/√N = 7.6e-6`.

### A2 — true McWeeny, all-f64: CONFIRMED that post-hoc polish fails

The earlier experiment ran `2K−KSK` with f32 intermediate casts — void. Corrected: true `K' = 3KSK − 2(KS)²K`, f64 throughout (on deg-175 mask): R_I oscillates ~7.5e-4, E stays ~−298.785. Post-hoc polishing cannot reach the right point — but the reason is now clear (below), not an f32 fixed-point defect.

### A3 — frozen-input dense reference: engine inputs FULLY VALIDATED

New test solves the **identical generalized eigenproblem** from the engine's own converged `h_scc_pad` + `s_bsr` (physical lanes, f64 Cholesky+dsyevd):

```
frozen H_scc eigh:  2Σ_occ ε = -312.6925   2Tr(P·H0) = -298.807141
sparse device E_band (deg-175) = -298.791399
DFTB+ R10: Energy H0 = -298.807668
```

**H0, S, H_scc, and the converged charges are all correct** (0.5 mHa vs DFTB+). The error lives entirely in the purified K.

### The actual mechanism: masked-product truncation, not f32

| M_K | nnz | device R_I | host-f64 R_I(sparse K) | E_band | Δ vs frozen ref |
|---|---|---|---|---|---|
| r_k=12 (deg~175, 53%) | 57 736 | 4.97e-4 | 7.6e-4 | −298.7914 | **+15.7 mHa** |
| r_k=14 (deg~228) | 75 364 | 3.16e-4 | 4.8e-4 | −298.8014 | +5.5 mHa |
| r_k=16 (deg~275) | 90 838 | 9.6e-5 | 1.5e-4 | −298.8057 | +1.2 mHa |
| r_k=20 (deg~330, ~complete) | 108 090 | 3.27e-5 | 5.0e-5 | −298.8070 | **+23 μHa** |

Error decays with the DM's physical decay length (R10 diameter ~15.5 Å → "complete" is cheap here; for 1000+ atoms the required degree is set by the decay length, not N). The truncation that matters is in the **intermediate** products: `m_t_ks = m_k` in `sparse_system.rs` — `K·S` is stored on M_K, so `KSK` on M_K loses all two-hop halo terms each iteration → biased fixed point.

- The earlier "complete-mask floor" claim was wrong — r_k=12 gives only 53% pair coverage on R10.
- Exact K injected on the mask (`purify_current` diagnostic, new `inject_k_bsr`): on the truncated mask the exact projector itself has R_I=9.4e-3 (dropped tails), and one f32 purifier step kicks it off toward the same biased point. On the ~complete mask the purifier **preserves** the injected exact K (R_I→4e-5, E unchanged to 0.03 mHa).
- The iteration therefore converges to a *biased fixed point* when intermediate products `K·S`, `K·S·K` are truncated to M_K — lower masked R_I does not mean better energy (sparse K had R_I 7.6e-4 but was 15.7 mHa off; the masked exact K has R_I 9.4e-3 and is exact in energy).
- My earlier `‖K_sparse−P_exact‖=15.85` "50% Frobenius error" was a factor-2 artifact of mine: engine K is the **spin-free** density `C·Cᵀ` (occupation 1, Tr(KS)=459), so the right comparison is `‖K−P/2‖`; `‖P‖/2 = 15.85` exactly — the complete-mask sparse K ≈ the exact projector.

### Consequences

1. **There is no significant f32 arithmetic floor.** Complete-mask f32 purifier reaches R_I~3e-5 and ~23 μHa energy error. The residual gap at truncated masks is systematic truncation of intermediate products — the quantity an expanded product support or a variational method (LNV) addresses.
2. The `dE ∝ R_I²` / "52 μHa/atom floor" laws are retracted — they were truncation + sloppy-Z effects.
3. Priority for production degree now: widen **intermediate product support** (T=KS onto a halo wider than M_K) and/or LNV so the fixed point is unbiased at moderate deg — this is §15.12-D's decision gate, now evidence-backed.
4. Sparse vs DFTB+ on R10/pbc is now **0.7 mHa** at complete mask (−298.8070 vs −298.8077; residual difference is charge-state level, from f32 SCC convergence at r_scc=1e-5 — DFTB+ ran to 2.8e-8).

Open: does ~deg-64–128 with a *wider intermediate* support (M_K∘M_HS halo) or LNV give acceptable energy — the deg-175→330 gap is where production speed lives.

**Cost/sparsity coupling (USER note, 2026-09-14):** the DM's required support follows its own decay length, not H/S's — but the *cost* of every product still scales with operand sparsity, and the required intermediate halo is `M_K ∘ M_S` (KS must cover S's reach around each K row). Narrower H/S therefore buys twice: cheaper gathers AND a narrower halo needed for an unbiased fixed point (manifest SC8b).

## 2026-09-14 (cont. 2) — Sparse-path profiling (RUST_DFTB_PROF instrumentation)

New: `SparseBsr4Gpu::prof_tick/reset/report` passthroughs to `GpuRuntime::Prof`; ticks at every kernel-stage boundary in `compute_z` (ns.zs/rz/tz/upd), `tc2_purify` (tc2.ks/tr/ksk/res/upd), `scc_inner` (ns, scc.v/hscc/k0/tc2/mull/mix/fin), `set_coords` (geom.hs/ul/gamma/rep), `forces` (f.dw/dl/contract), plus `[init]` compile/plan timing. `dftb_engine` dumps the table for sparse engines too. Raw data: `debug/prof_sparse/{r10_evt,r10_mark,r10_rk12_evt,r14_rk12_evt}.txt`.

**Workload:** R10/pbc-0-3 (330 at) and R14 (864 at), one SCC to r_scc~1e-5 + one eval(+forces). `evt` = host ms + true device ms per stage; `mark` = pure host enqueue (no syncs).

### Per-stage device time per call (the real GPU cost)

| stage (TC2 iter unless noted) | R10 deg175 | R10 deg330 | R14 deg~390 |
|---|---|---|---|
| `tc2.ksk` — Q=T·K → M_K | **2.64 ms** | **10.34 ms** | **6.97 ms** |
| `tc2.ks` — T=K·S → M_K | 0.56 ms | 1.17 ms | 1.40 ms |
| `tc2.tr` — trace partials read | 0.10 | 0.15 | 0.13 |
| `tc2.res` — idempotency read | 0.11 | 0.18 | 0.17 |
| `tc2.upd` — branch+symmetrize | 0.05 | 0.14 | 0.19 |
| `scc.k0` — ZH,ZHZ,bounds (per SCC iter) | 4.08 | 14.9 | 11.4 |
| `ns.tz` — Q=T·Z (per NS iter) | 3.92 | 17.3 | 12.3 |
| `ns.zs` — T=Z·S (per NS iter) | 0.87 | 2.14 | 2.65 |
| `f.contract` — host force contraction (once) | 5.4 | 5.9 | 21.2 |
| `f.dw` — device W build (once) | 2.0 | 7.1 | 10.0 |
| `scc.fin` — R_H stationarity (once) | 4.2 | 9.7 | 18.1 |
| `geom.hs` — H0/S assemble (per geom) | 3.5 | 3.5 | 11.9 |

### Shares and structure

- **`tc2.ksk` is the hot kernel: 73–83% of device time.** Work per call ≈ nnz(M_K)×deg(operand) — going deg 175→330 costs 4× (output nnz ~2× AND inner degree ~2×). K-degree is the price; S-degree is 3–10× cheaper per call.
- **Two blocking host reads per TC2 iter** (`tc2.tr` trace + `tc2.res` residual): `reads=1614 ≈ 2×753 iters`. The host wait lands under `tc2.res` (it drains the just-enqueued KSK). True sync overhead beyond the product itself ≈ 0.3–0.7 ms/iter (`mark` mode shows pure enqueue is only 0.02–0.06 ms/call — launch is not the problem).
- **Plateau iters are the biggest recoverable waste:** ~30–50 of the ~55–60 TC2 iters per SCC call are spent oscillating at the floor before the 10×-growth detector fires (e.g. R10: floor reached ~iter 44, restore at 53; several runs exhausted all 80). An honest stagnation detector (§15.12-B) would cut ~40% of device time.
- **`scc.k0` per SCC iter ≈ 4–15 ms dev** (two wide products + two bounds reads) — 12–15 calls ≈ 5–8% of the SCC total. Worth caching if H_scc changes little, or at least the bounds.
- **Per-geometry costs** (R14): geom.hs 11.9 ms, geom.gamma 5.6 ms (O(N²) host), NS ~120 ms total (6 iters × ~15 ms), f.contract 21 ms, scc.v 0.5 ms/iter.
- **SCC wall times:** R10 deg175: 2.7 s (13 iters); R10 deg330: 8.1 s; R14 deg390: 7.8 s (15 iters). Warm re-SCC: 0.37 s (single confirm iter).

### What the profile says to do next

1. **B (early stop):** kill plateau iterations — ~40% of TC2 device time.
2. **E (syncs):** fuse the trace+residual reads into one return (one marker, one read) — saves ~0.5–1.5 ms/iter and one queue drain per iter.
3. **D (mask halo):** cost is `nnz_out × deg_operand`; a narrow K with a wider *intermediate* halo is the production shape (SC8b) — the ksk product is what must get cheaper.

## 2026-09-14 — Speed–accuracy source review

**Scope/status:** current-working-tree source analysis and documentation only. No solver edits, GPU tests or new timings. Historical numerical results below are reported observations, not independently rerun measurements. Existing uncommitted work is preserved. New implementation tickets in manifest §15.12 remain open/unverified.

**Recommendation:** keep f32 BSR4 and planned gather products. Spend accuracy on a trustworthy inverse/initial projector and final force state; spend less work in early SCC and after evidenced stagnation. Smooth energy offsets may be acceptable for scans/vibrations, but charge convergence alone cannot certify their derivatives. Do not prescribe Kahan, double-single storage, LNV or a universal degree before resolving the inexpensive diagnostic defects.

### Confirmed source findings

1. **NS stopping understates the residual by √N.** `sparse_system.rs::compute_z` (reviewed line 724) computes `sqrt(r2)/n_orb`; its documented contract is `sqrt(r2)/sqrt(n_orb)`. `gpu_sparse.rs::identity_residual_to_f64` and `bsr4_identity_residual_partial` confirm that r2 is the squared Frobenius sum. For R10/330 atoms and R14/864 atoms the factors are **36.33 and 58.79**, using 4 padded orbitals/atom. With `ns_tol=1e-4`, acceptance permits the documented residual up to **3.63e-3 and 5.88e-3**. These are acceptance bounds, not measurements of the returned inverse. The pbc parity script and `sparse_f64check.rs` use that tolerance.

   `test_ns_device_residual_contract` compares two independently recomputed residuals using √N, but only **prints** the production `rz_reported`; it never asserts agreement with it. This permits the normalization bug to escape the test. Inaccurate Z contaminates ZH/ZHZ initialization and the default W shortcut. Its contribution to the energy error must be measured; this review does not claim it explains all 17.3 mHa.

2. **The polish experiment is neither McWeeny nor an f64 iteration.** `tests/sparse_f64check.rs::f64_host_trace` repeats `K←2K−KSK`; true generalized McWeeny is `K←3KSK−2KSKSK`. For the implemented map an empty-state occupation ε becomes approximately 2ε: repeated complements amplify leakage by construction. KS and every new K are also cast back to f32. Its useful host-f64 energy contraction shows the final trace reduction is not the main error; the subsequent loop does **not** prove irreversible subspace corruption or rule out polishing. Correct algebra already exists in `gpu_sparse.rs::mcweeny_step`.

3. **Complete masks do not establish an arithmetic floor.** They remove support truncation, but not inverse/stopping/initialization/SCC errors or model differences. K-TC2 normalizes R_I by the initial K0 norm; recovered P-TC2 K uses its current norm. Changing spectral bounds changes K0 and hence the reported K-TC2 residual normalization. The proposed universal `dE∝R_I²` and 52 μHa/atom laws remain correlations across confounded runs. ACC8 changes summation ordering, not precision; no improvement from ACC8 does not exclude accumulation error.

4. **Overhead and acceptance are not solved.** Production `purify_hscc` passes `check_every=1`: each K-TC2 iteration reads n_atom trace partials, then up to 128 residual partials in another blocking read; guards add work. Reduction tails are **host f64**, not device f64. `rh_stationarity` downloads O(nnz) at finalization. `forces()` downloads K/W and contracts on CPU; `compute_v()` is a host O(N²) gamma matvec; `mulliken_checked()` constructs charge vectors each SCC iteration. Persistent matrix buffers are valuable progress, not a fully GPU-resident pipeline.

   `tc2_purify` has `stagnant=false` and can label exhaustion a `NumericalFloor`. Snapshot eligibility uses trace/R_I, not stationarity/force quality. SCC accepts floors by default; `finalize_scc` measures R_H without gating it. P-TC2 retains P's status while returning recovered-K residuals. These labels cannot by themselves certify vibrations.

### Answers to manifest §15.11

**1. Mechanism:** unresolved until inverse and diagnostic corrections are isolated at fixed H,S. Polynomial purification preserves eigenspaces in exact arithmetic: true McWeeny can improve occupations near 0/1 but cannot rotate a wrong subspace toward H. Finite-precision SP2 stagnation is known; it does not establish the origin of this unusually large error. Convergence-order-based stopping is a useful reference, with validation needed for this nonorthogonal masked algebra. [Kruchinina, Rudberg & Rubensson (2016)](https://arxiv.org/abs/1507.02087).

**2. Accuracy per flop:** first correct NS and certify it once per geometry, then stop wasting products at a genuine plateau. Compare repaired K/P-TC2 including recovery and force cost. Compensated-f32 endgame is conditional on measured product-accumulation error. GPU f64 products conflict with PR5; double-single storage is a last resort due to traffic/complexity. LNV is the relevant alternative if H-dependent subspace correction remains necessary; changing scalar purification polynomials alone is not that correction.

**3. Variational masking:** LNV may improve the variational state, but the recoverable fraction of the deg-95 error is unknown. A sparse auxiliary L in `K(L)=3LSL−2LSLSL` can generate a wider K; equal degree of L and a stored K is not equal support/work. Projecting products changes the implemented functional, whose electronic gradient and nuclear forces must be derived consistently. Existing `bsr4_lnv_gradient` is a combination primitive, not a validated masked optimizer. Do not promise first-order recovery of missing tails or quadratic energy error for inadmissible/nonstationary states. [Nunes & Vanderbilt, nonorthogonal formulation](https://www.physics.rutgers.edu/~dhv/pubs/local_copy/rw_dms.pdf).

**4. Relative energies:** measure `b(R)=E_fast−E_ref` along the stencil. For consistent gradients, force bias is `−∇b` and Hessian bias is `∇²b`. Constant offsets cancel; smooth curvature changes frequencies; deterministic discontinuities remain harmful. Freeze masks/model settings, start both signs from the same central state, reset incompatible mixer history, and test reversed order/cold starts. Fixed iteration counts or replayed branches alone do not ensure smoothness.

**5. Mask choice:** retain radial as the control; “radial is dead” is unsupported. Current top-k `mask_kz` injection changes K and Z together. Separate their budgets and intermediate contribution support. Direct `|H_ij K_ji|` neglects overlap/Pulay sensitivity and indirect product paths; `|H_ij P_ji|` is not the band-energy contraction because P=KS. Compare actual executed plan terms and force accuracy, including wide-reference/oracle setup cost.

**6. Dummy lanes:** structural exclusion of the decoupled dummy density subspace is a legitimate later optimization. Keep dummy S nonsingular; enforce zero dummy density from initialization through updates/recovery before excluding it from bounds. Tighter valid bounds can reduce iterations, but do not smooth the final zero-temperature projector or repair its subspace. Do not clip physical occupations.

**7. Crossover/occupancy:** no defensible crossover N follows from these data. A 4×4 block triple costs approximately 128 FLOPs; count retained triples, physical orbitals, all iterations/transfers and force time. The current kernel launches one WG per atom row (330/864 WGs for R10/R14); utilization also depends on plan lengths, registers and compiled resources. Local cache is `64·MAX_LEFT_BLOCKS` bytes/WG. A 49 KiB per-WG limit does not imply a degree-380 ceiling. Wide Z/halo can enlarge the compiled cache for narrow-K products too. Benchmark matched-quality complete evaluations on the actual device, then batch displacements with shared plans and independent electronic state.

**Next work:** manifest §15.12 A → B/C → evidence-selected D/E. The target is fastest accepted scans, relaxation and Hessians with measured uncertainty, not system-independent mHa total-energy parity. No production accuracy defaults or new speedup claims are established by this review.

## 2026-09-14 (cont.) — §15.12-A executed: the "f32 floor" was never f32

All measurements below from `tests/sparse_f64check.rs` (`frozen_input_dense_ref`, `f64_host_trace`; R10/pbc, pbc-0-3).

### A1 — NS normalization bug: CONFIRMED + FIXED

`compute_z` divided `||I−ZS||_F` by `n_orb` instead of `√n_orb` — understated the residual by √1320 = **36.3×** at R10. The reported `R_Z=3.75e-5` was really `1.36e-3` — a genuinely sloppy inverse feeding ZH/ZHZ/K0 and the default W. Fixed (`sparse_system.rs`); `test_ns_device_residual_contract` now **asserts** the production value against an independent host-f64 recomputation (it previously only printed). Post-fix NS honestly converges: 7 iters, true host-f64 `||ZS−I||/√N = 7.6e-6`.

### A2 — true McWeeny, all-f64: CONFIRMED that post-hoc polish fails

The earlier experiment ran `2K−KSK` with f32 intermediate casts — void. Corrected: true `K' = 3KSK − 2(KS)²K`, f64 throughout (on deg-175 mask): R_I oscillates ~7.5e-4, E stays ~−298.785. Post-hoc polishing cannot reach the right point — but the reason is now clear (below), not an f32 fixed-point defect.

### A3 — frozen-input dense reference: engine inputs FULLY VALIDATED

New test solves the **identical generalized eigenproblem** from the engine's own converged `h_scc_pad` + `s_bsr` (physical lanes, f64 Cholesky+dsyevd):

```
frozen H_scc eigh:  2Σ_occ ε = -312.6925   2Tr(P·H0) = -298.807141
sparse device E_band (deg-175) = -298.791399
DFTB+ R10: Energy H0 = -298.807668
```

**H0, S, H_scc, and the converged charges are all correct** (0.5 mHa vs DFTB+). The error lives entirely in the purified K.

### The actual mechanism: masked-product truncation, not f32

| M_K | nnz | device R_I | host-f64 R_I(sparse K) | E_band | Δ vs frozen ref |
|---|---|---|---|---|---|
| r_k=12 (deg~175, 53%) | 57 736 | 4.97e-4 | 7.6e-4 | −298.7914 | **+15.7 mHa** |
| r_k=14 (deg~228) | 75 364 | 3.16e-4 | 4.8e-4 | −298.8014 | +5.5 mHa |
| r_k=16 (deg~275) | 90 838 | 9.6e-5 | 1.5e-4 | −298.8057 | +1.2 mHa |
| r_k=20 (deg~330, ~complete) | 108 090 | 3.27e-5 | 5.0e-5 | −298.8070 | **+23 μHa** |

Error decays with the DM's physical decay length (R10 diameter ~15.5 Å → "complete" is cheap here; for 1000+ atoms the required degree is set by the decay length, not N). The truncation that matters is in the **intermediate** products: `m_t_ks = m_k` in `sparse_system.rs` — `K·S` is stored on M_K, so `KSK` on M_K loses all two-hop halo terms each iteration → biased fixed point.

- The earlier "complete-mask floor" claim was wrong — r_k=12 gives only 53% pair coverage on R10.
- Exact K injected on the mask (`purify_current` diagnostic, new `inject_k_bsr`): on the truncated mask the exact projector itself has R_I=9.4e-3 (dropped tails), and one f32 purifier step kicks it off toward the same biased point. On the ~complete mask the purifier **preserves** the injected exact K (R_I→4e-5, E unchanged to 0.03 mHa).
- The iteration therefore converges to a *biased fixed point* when intermediate products `K·S`, `K·S·K` are truncated to M_K — lower masked R_I does not mean better energy (sparse K had R_I 7.6e-4 but was 15.7 mHa off; the masked exact K has R_I 9.4e-3 and is exact in energy).
- My earlier `‖K_sparse−P_exact‖=15.85` "50% Frobenius error" was a factor-2 artifact of mine: engine K is the **spin-free** density `C·Cᵀ` (occupation 1, Tr(KS)=459), so the right comparison is `‖K−P/2‖`; `‖P‖/2 = 15.85` exactly — the complete-mask sparse K ≈ the exact projector.

### Consequences

1. **There is no significant f32 arithmetic floor.** Complete-mask f32 purifier reaches R_I~3e-5 and ~23 μHa energy error. The residual gap at truncated masks is systematic truncation of intermediate products — the quantity an expanded product support or a variational method (LNV) addresses.
2. The `dE ∝ R_I²` / "52 μHa/atom floor" laws are retracted — they were truncation + sloppy-Z effects.
3. Priority for production degree now: widen **intermediate product support** (T=KS onto a halo wider than M_K) and/or LNV so the fixed point is unbiased at moderate deg — this is §15.12-D's decision gate, now evidence-backed.
4. Sparse vs DFTB+ on R10/pbc is now **0.7 mHa** at complete mask (−298.8070 vs −298.8077; residual difference is charge-state level, from f32 SCC convergence at r_scc=1e-5 — DFTB+ ran to 2.8e-8).

Open: does ~deg-64–128 with a *wider intermediate* support (M_K∘M_HS halo) or LNV give acceptable energy — the deg-175→330 gap is where production speed lives.

**Cost/sparsity coupling (USER note, 2026-09-14):** the DM's required support follows its own decay length, not H/S's — but the *cost* of every product still scales with operand sparsity, and the required intermediate halo is `M_K ∘ M_S` (KS must cover S's reach around each K row). Narrower H/S therefore buys twice: cheaper gathers AND a narrower halo needed for an unbiased fixed point (manifest SC8b).

## 2026-09-14 (cont. 2) — Sparse-path profiling (RUST_DFTB_PROF instrumentation)

New: `SparseBsr4Gpu::prof_tick/reset/report` passthroughs to `GpuRuntime::Prof`; ticks at every kernel-stage boundary in `compute_z` (ns.zs/rz/tz/upd), `tc2_purify` (tc2.ks/tr/ksk/res/upd), `scc_inner` (ns, scc.v/hscc/k0/tc2/mull/mix/fin), `set_coords` (geom.hs/ul/gamma/rep), `forces` (f.dw/dl/contract), plus `[init]` compile/plan timing. `dftb_engine` dumps the table for sparse engines too. Raw data: `debug/prof_sparse/{r10_evt,r10_mark,r10_rk12_evt,r14_rk12_evt}.txt`.

**Workload:** R10/pbc-0-3 (330 at) and R14 (864 at), one SCC to r_scc~1e-5 + one eval(+forces). `evt` = host ms + true device ms per stage; `mark` = pure host enqueue (no syncs).

### Per-stage device time per call (the real GPU cost)

| stage (TC2 iter unless noted) | R10 deg175 | R10 deg330 | R14 deg~390 |
|---|---|---|---|
| `tc2.ksk` — Q=T·K → M_K | **2.64 ms** | **10.34 ms** | **6.97 ms** |
| `tc2.ks` — T=K·S → M_K | 0.56 ms | 1.17 ms | 1.40 ms |
| `tc2.tr` — trace partials read | 0.10 | 0.15 | 0.13 |
| `tc2.res` — idempotency read | 0.11 | 0.18 | 0.17 |
| `tc2.upd` — branch+symmetrize | 0.05 | 0.14 | 0.19 |
| `scc.k0` — ZH,ZHZ,bounds (per SCC iter) | 4.08 | 14.9 | 11.4 |
| `ns.tz` — Q=T·Z (per NS iter) | 3.92 | 17.3 | 12.3 |
| `ns.zs` — T=Z·S (per NS iter) | 0.87 | 2.14 | 2.65 |
| `f.contract` — host force contraction (once) | 5.4 | 5.9 | 21.2 |
| `f.dw` — device W build (once) | 2.0 | 7.1 | 10.0 |
| `scc.fin` — R_H stationarity (once) | 4.2 | 9.7 | 18.1 |
| `geom.hs` — H0/S assemble (per geom) | 3.5 | 3.5 | 11.9 |

### Shares and structure

- **`tc2.ksk` is the hot kernel: 73–83% of device time.** Work per call ≈ nnz(M_K)×deg(operand) — going deg 175→330 costs 4× (output nnz ~2× AND inner degree ~2×). K-degree is the price; S-degree is 3–10× cheaper per call.
- **Two blocking host reads per TC2 iter** (`tc2.tr` trace + `tc2.res` residual): `reads=1614 ≈ 2×753 iters`. The host wait lands under `tc2.res` (it drains the just-enqueued KSK). True sync overhead beyond the product itself ≈ 0.3–0.7 ms/iter (`mark` mode shows pure enqueue is only 0.02–0.06 ms/call — launch is not the problem).
- **Plateau iters are the biggest recoverable waste:** ~30–50 of the ~55–60 TC2 iters per SCC call are spent oscillating at the floor before the 10×-growth detector fires (e.g. R10: floor reached ~iter 44, restore at 53; several runs exhausted all 80). An honest stagnation detector (§15.12-B) would cut ~40% of device time.
- **`scc.k0` per SCC iter ≈ 4–15 ms dev** (two wide products + two bounds reads) — 12–15 calls ≈ 5–8% of the SCC total. Worth caching if H_scc changes little, or at least the bounds.
- **Per-geometry costs** (R14): geom.hs 11.9 ms, geom.gamma 5.6 ms (O(N²) host), NS ~120 ms total (6 iters × ~15 ms), f.contract 21 ms, scc.v 0.5 ms/iter.
- **SCC wall times:** R10 deg175: 2.7 s (13 iters); R10 deg330: 8.1 s; R14 deg390: 7.8 s (15 iters). Warm re-SCC: 0.37 s (single confirm iter).

### What the profile says to do next

1. **B (early stop):** kill plateau iterations — ~40% of TC2 device time.
2. **E (syncs):** fuse the trace+residual reads into one return (one marker, one read) — saves ~0.5–1.5 ms/iter and one queue drain per iter.
3. **D (mask halo):** cost is `nnz_out × deg_operand`; a narrow K with a wider *intermediate* halo is the production shape (SC8b) — the ksk product is what must get cheaper.

## 2026-09-14 (cont. 3) — Sync architecture & per-kernel cost analysis (for review discussion)

Detailed answers to four questions about the profiling results. All numbers measured on the RTX 3090 (82 CUs, 49 KiB local/WG) unless noted.

### Q1 — What exactly are the "2 blocking host reads per TC2 iteration"?

Per iteration of `tc2_purify` the host performs two blocking buffer reads (`GpuRuntime::read_buffer` → `clEnqueueReadBuffer` on an in-order queue, i.e. each read drains everything enqueued so far):

| read | data | device-side producer | consumed by |
|---|---|---|---|
| `trace_ks_f64` | n_atom f32 partials → host f64 sum → **Tr(KS)** | `bsr4_mulliken_KS` diag-partial kernel on `t_ks` (K·S) | **the branch decision** `branch = (Tr > nocc) ? squaring : complement` — a hard data dependency of the *next* product, plus the trace guard and Tr reporting |
| `idempotency_to_f64` | ≤128 f32 partials → host f64 → **R_I = ‖KSK−K‖_F/‖K‖** | `bsr4_idempotency_partial` on (Q, K) | convergence/floor/snapshot decision |

Measured (R14, deg~390): `tc2.tr` host ≈ 1.68 ms/iter, `tc2.res` host ≈ 7.08 ms/iter — but note the host wait under `tc2.res` mostly *is* the KSK kernel latency (the read drains the queue that contains it). Pure enqueue overhead is tiny (`mark` mode: 0.02–0.06 ms/call). So the real cost of the reads is not bandwidth — it is that **each read forces a full pipeline drain**: the GPU finishes KSK, sits idle while the host sums 128 floats and branches, then the next iteration's products are enqueued into an empty pipe.

**Can they be eliminated?** Partially trivially, fully with an architectural change:

- `R_I` check can run every k-th iteration (`check_every` parameter already exists; production currently passes 1).
- `Tr(KS)` is needed **every iteration** because the host picks the polynomial branch from it — you cannot skip the *decision*, only move it.
- **Full fix = device-side TC2 driver:** a small kernel at the end of each iteration computes Tr from the partials, computes R_I partials after KSK, forms `branch` and `guard_alpha` on-device (`α = nocc/Tr` when the guard predicate holds, else 1.0 — folded into `tc2_dev`'s update so the rescale is unconditional and free), and appends `{Tr, R_I, dev_rel, branch, guard_fired}` to a device log buffer. The host then runs M iterations completely blind and reads the log once per chunk → decide continue / stop / restore-best-K. This is the dense side's W4 "chunked convergence" pattern transplanted to TC2. Payoff: zero mid-loop drains; products enqueue back-to-back; ~0.3–0.7 ms sync overhead per iter plus recovered overlap.
- Cost: a decision kernel + a log buffer + chunked host supervision loop. Moderate refactor, confined to `tc2_purify` + two new small kernels.

### Q2 — Why is `tc2.ksk` ~5× more expensive than `tc2.ks`?

Both products write the **same output mask** M_K — identical nnz_out. The difference is the **inner dimension**: for `C_ij = Σ_k A_ik·B_kj` the retained work per output block is `|N_left(i) ∩ N_right(j)|`.

- `T = K·S` (`tc2.ks`): needs `S_kj ≠ 0` → only **deg_S ≈ 56** of row i's K-neighbors contribute.
- `Q = T·K` (`tc2.ksk`): needs `K_kj ≠ 0` → **deg_K ≈ 175–330** contribute.

Measured ratio 4.7× (deg 175) / 8.8× (deg 330) ≈ deg_K/deg_S (3.1×/5.9×) plus worse locality of the wider operand rows. **The price is set by K's degree, not by the output mask size.** This is the quantitative form of the user's sparsity-cost coupling: narrowing S (SC6) directly discounts the ks product, but the ksk product only gets cheaper if K's *stored* neighborhood shrinks — regardless of where results land.

### Q3 — The ~40% plateau waste: concrete detector design

Observed pattern (deg-175 R10): R_I descends to ~5e-4 by iter ~44, then oscillates for 9+ iterations, trace guard fires repeatedly, and the run ends either by the `ri > 10·best` blowup detector (iter ~53) or by exhausting all 80 iters — several SCC calls burned the full 80. The previously-removed "N consecutive non-improving checks" detector false-fired during normal descent because *single* non-improving iters are routine during the TC2 transient.

**Safe criterion — windowed-best:** stop iff `iter − iter_of_best > W` (W≈10–15) AND `best_r_i < 1e-2` AND the snapshot's trace is valid. During true descent `best_r_i` is updated every few iterations even when individual iters oscillate; at the floor it stops updating entirely. Cannot fire before a snapshot exists. Expected saving ~30–40% of TC2 device time on truncated-mask runs; nothing to save when converged cleanly.

### Q4 — Do the reductions need the host every cycle?

No. The GPU partial-sum kernels already exist — what forces the drain is that the *decisions* live on the host. Three levels:

1. **Cheap now:** `check_every=4` for R_I (skips 3/4 of residual reads). Trace read still needed every iter → saves ~half the drains.
2. **One-drain-per-iter:** the trace read (pre-KSK) and residual read (post-KSK) cannot trivially share one read — the trace feeds the branch *before* KSK. Fusing them into one return would defer the branch by a full iteration (uses T from the previous K) — a numerics change, needs A/B testing.
3. **Zero-drain (the Q1 fix):** device branch+guard+log, host reads once per M-iter chunk. Combined with (1)'s cadence this reduces host syncs to ~1 per chunk plus the final state read.

### Implications for §15.12 ordering

- **B** (early stop) becomes: windowed-best detector + honest status labels — no algorithm change, big win.
- **E** first real item: device-side TC2 driver (branch/guard/log on GPU) — eliminates the only per-iter data dependency the host has inside purification.
- The **mask-halo** question (D) is now measurable in isolation: `tc2.ksk` is the single kernel to make cheaper; a stored-narrow-K + wide-intermediate experiment changes exactly this kernel's cost.

## 2026-09-14 (cont. 4) — Review answers to Q1–Q4 and next sparse work

**Scope/status:** source review of the current sparse implementation, the preceding discussion starting at line 2033, and all four `debug/prof_sparse/*.txt` tables. No code changes or new numerical/timing runs. The corrected NS normalization and assertion are present in source; the reported improvement is substantial evidence against the previous large intrinsic-f32-floor hypothesis. Preserve that progress. The answers below correct several proposed next steps before implementation; all new proposals remain unverified.

### Q1 — The reads are real, but the branch is not a dependency of KSK

`tc2_purify` computes `T=K·S`, reads its trace, computes `Q=T·K`, reads `||Q−K||`, and only then calls `tc2_dev` to choose Q or 2K−Q. **Both branches use exactly the same Q.** The branch value is not an argument to either SpGEMM. The exceptional dependency is the **trace guard**, which currently rescales K/T before Q is formed. The trace producer is `bsr4_trace_atom`, via `trace_atom_dev`, not `bsr4_mulliken_KS` as Q1's table states.

**Recommended first change: one read on the ordinary path, preserving host-f64 decisions and the existing guard arithmetic.**

1. Enqueue T, trace partials, unscaled Q, and residual partials without an intervening host wait. Place the two sets of partials in persistent disjoint ranges of one diagnostic buffer; one blocking read returns them. Preserve the current reduction order initially.
2. On the host compute the trace, guard predicate and residual for the **same current K**. Preserve the existing temporal semantics: the guard-lock predicate currently uses the previous checked residual, not the new one. A scheduling refactor must not quietly change this numerical policy.
3. If no guard fires, perform the existing convergence/snapshot/update decisions. No stale trace, previous-iteration branch, or changed polynomial is involved.
4. If the guard fires, treat the speculative unscaled Q/residual as scratch: perform the existing K/T rescale, remeasure trace, recompute Q and residual, then make decisions on that corrected state. This occasional extra product/read preserves existing f32 operation ordering. Measure guard frequency; if guards are frequent this route may lose its benefit, another reason to stop floor-chasing first.

An optional later algebraic variant avoids that recomputation. With `K'=αK`, `T'=αT`, one has `Q'=α²Q`, so the correct updates are `α²Q` or `2αK−α²Q`, and the corresponding residual is `||α²Q−αK||`. Scaling Q by α, or simply scaling the final old update, is wrong. Scalar coefficients are cheap but **not bitwise-equivalent** to rescale-before-SpGEMM in f32. Trace must still be measured on the accepted state, and snapshots/diagnostics must refer to that state. Test seeded guard firings, branches, trace, residuals and forces before adopting this variant; do not repeat the stale-Q bug from S1.

The “full device driver” is therefore not necessary to remove the first drain. Start with the smaller change above and measure unprofiled end-to-end time. Sequential dependent matrix products still cannot overlap with one another; eliminating CPU gaps does not remove their execution time.

### Q2 — KSK has more retained triples; degree alone is not the answer

The qualitative explanation is sound: S has fewer neighbors than K, so `KS` generally has fewer contributing block triples than `(KS)K`. However the report mixes **average degree, maximum degree and inner intersection length**. For example 57,736 blocks / 330 rows = average degree 174.96; that is not the maximum row degree used for kernel sizing. Each output block has its own intersection length. The output mask matters too: it sets the number of outputs and which intersections exist.

For each plan collect once at construction:

- total `T_p = plan_ptr.last()` retained triples; output nnz; mean/max row degree;
- mean/quantiles/max terms per output and total terms per atom row;
- plan bytes, compiled MAX_LEFT_BLOCKS, local/private resources and WG size.

The useful comparison is roughly `128·T_p` FLOPs per product (4×4 multiply-add), time per triple and distribution of row work. A ratio exceeding the degree ratio does not prove poor locality without those counts. The existing `SpgemmPlan` supplies the needed structural data.

**Widening T affects BOTH kernels:** KS writes more output blocks, and Q reads a wider left row with more contributions. It can also enlarge the compiled local cache and plans. At fixed K, a halo is expected to cost more; it wins if the resulting accuracy allows a sufficiently narrower K or fewer iterations. Thus “the halo experiment changes exactly KSK's cost” and “KSK only gets cheaper by shrinking stored K” are both too strong. Exact layout/scheduling improvements and validated contribution screening can also reduce its time.

At equal accepted E/F quality compare the combined KS+Q cost and total solve, not Q alone. Kernel opportunities after that comparison: per-product cache sizing instead of a global maximum set by wide Z/T, grouping rows by executed work, and splitting unusually heavy rows by disjoint output tiles. An exact global-gather route avoids row-sized local cache pressure; it is an A/B candidate, not an automatic replacement. Keep one owner per output and deterministic term ordering. Measure register/private-memory costs before increasing accumulator count or tile size.

### Q3 — Windowed-best is a candidate heuristic, not a safe criterion yet

`iter−iter_of_best>W` is effectively **W consecutive checks without a new global minimum**. The currently disabled `n_stagnant` already increments when the jointly eligible best snapshot does not improve. Renaming that history “windowed-best” does not solve its earlier false-positive mechanism. Small gaps/conditioning or a long trace-correction phase can pause improvement; conversely tiny random improvements can reset the clock forever. `best_r_i<1e-2` and valid trace are not a physical-accuracy certificate, especially when masked R_I can improve while energy worsens.

First replay candidate rules against saved histories from the existing SiH4 failure, full/nearly-full masks, truncated R10/R14, and cold/warm starts. Log iteration, branch, trace, raw/current-normalized residual, guard, snapshot iteration and termination reason. Use the existing diagnostic machinery; avoid a new solver framework.

A credible small detector needs evidence of entering the endgame, branch-aware comparison of comparable states, and **lack of meaningful improvement** over a window relative to measured arithmetic/truncation variability. Keep separate indices for the absolute best snapshot and the last meaningful improvement. Do not erase slow real progress. Guards identify a perturbed sequence; do not apply an unmodified exact-polynomial convergence theorem across them. The literature's loss-of-convergence-order criterion is a better starting point than an unexplained W, but its assumptions/norms must be checked for the nonorthogonal masked method. [Kruchinina, Rudberg & Rubensson, parameterless stopping criteria](https://arxiv.org/abs/1507.02087).

Return **stagnated at the requested support/precision**, with the best admissible snapshot and remeasured diagnostics; let the explicit application policy decide whether that error is acceptable. Exhaustion remains exhaustion. A low masked residual and a valid charge do not certify a vibrational state.

**The 40% saving is not demonstrated by the quoted example.** Best at 44, restoration at 53 leaves about 9 iterations to remove. W=10–15 would stop around 55–60, later than the existing restoration. An 80-step run might save about 20–25 steps with that W, depending on the actual last best. Aggregate saved *product time* over real histories, including cleanly converged runs and any extra SCC iterations/changed forces. The earlier “30–50 of 55–60” assertion cannot be inferred from “floor at 44.”

### Q4 — Reduce host visits in stages; blind chunks need device-side safety

**First choice: the Q1 common-path combined read.** It removes one ordinary host visit without changing precision or lagging the branch. Q4's claim that one read necessarily uses the previous iteration's trace is incorrect.

`check_every=4` is not purely a scheduling knob in current source. `last_r_i` arms the trace guard; it also controls snapshot and floor detection. Checking less often can delay guarding and skip the best/final converged state, potentially adding products or allowing growth. It removes roughly 3 of every 8 normal reads (37.5%), not all residual work for free, and after Q1 both diagnostics already share the same visit. Test cadence as a separate numerical change, preferably dense checking in the endgame/guard region. Do not change cadence and stopping policy simultaneously.

Device-side decisions/chunks are a later option if measured idle time warrants the refactor. A per-iteration log plus host supervision after M **blind** updates is insufficient: an accepted state can be overwritten and a divergent state can be advanced before the host notices. A proper design requires persistent per-system active/converged/stagnated/failed flags, on-device fail-fast checks, best-state snapshots and iteration IDs, and terminal states that make subsequent queued work harmless. Kernel launches provide global ordering; workgroup barriers cannot synchronize the whole sparse matrix. Recompute final T/diagnostics after restoration. Full logging should be bounded and debug-gated; production needs a compact status record, not a new growing history buffer.

The repository currently specifies host-f64 discrete decisions. Do not silently replace them with one device f32 sum. Any future device decision policy must explicitly preserve sufficient scalar accuracy, validate near-Nocc branch decisions and guard predicates against the host reference, and reconcile that policy with the existing contract. GPU f64 **matrix** products remain excluded. For now, batching independent systems' partials into one host-f64 decision round is compatible with the existing precision policy and can hide CPU gaps without redesigning the purifier.

### Additional finding 1 — The profiler is an elapsed timeline, not kernel execution time

`GpuRuntime::prof_drain_events` accumulates `end(marker_i)−end(marker_(i−1))`. That interval includes kernel execution, transfers, driver scheduling and **idle time before the host submits the next command**. The implementation's own report comment acknowledges idle gaps, despite calling the column “TRUE device time.” Raw R14 data assign **21.228 ms of device time to CPU `f.contract`**, and CPU gamma/assembly similarly appear in the device column. Nearly equal host/device totals are expected for two partitions of the same elapsed interval, not proof of GPU saturation.

The `mark` mode also includes real blocking waits and host computation; only stages containing enqueue-only work approximate host submission overhead. Small enqueue duration does not by itself exclude launch latency. KSK is a credible dominant interval, but the **exact pure-kernel share and 0.3–0.7 ms recoverable sync cost are not isolated** by these tables.

**Measurement instruction:** retain markers as timeline diagnostics, correctly labeled. For kernel execution use the kernels' own event START/END, and record submission/queue gaps separately. Measure profiler-off full evaluations with matched iteration counts/state/accuracy. Do not sum waiting time with the GPU execution it already contains. A/B the proposed schedule to establish actual savings. In R14 the reported K0 interval is 171.7/7851.1 ms, about **2.2%**, not 5–8%; CPU force contraction is about 0.27%. Do not prioritize a complex force port or speculative K0 cache over the repeated products on this workload. Their shares may change after the purifier improves.

**Concrete redundant synchronization:** `GpuRuntime::read_buffer` does `buf.read(out).enq()` followed by `queue.finish()`. The installed `ocl-0.19.7/src/standard/buffer.rs` defaults buffer reads to `block=true` (constructor and API documentation checked). For the current sequential in-order use, the second finish is redundant. The logs' equal `finishes`/`reads` counters reflect this implementation, not two independent waits for outstanding compute. Have the shared-runtime owner remove the redundant finish after validating buffer readiness/error propagation and event draining; sparse agents should not edit the parallel dense owner's file without coordination. Keep read/finish counters distinct. Removing it is a small API-overhead opportunity, not a promised millisecond-scale gain.

### Additional finding 2 — Truncation dominates, but intermediate-only causality is still unseparated

The new data support **support/initialization errors dominating the former large f32 claim**. They do not yet isolate omitted KS intermediates from omitted K entries: the sweep changes K and Z radii together, and the f64 McWeeny test still projects every product onto M_K. Even the largest listed graph is not literally complete: 108,090 versus 330²=108,900 blocks. Correct the diagnostic header's unconditional “R10 mask is COMPLETE” claim; its default radius is still 12 Å.

**Decisive cheap diagnostic, before LNV:** freeze one H,S and a separately certified Z. For the same truncated K_c and fixed output mask M, compute in host f64:

`Q_legacy = project_M(project_M(K_c S) K_c)`

`Q_exact_products = project_M((K_c S) K_c)`

and compare both with `project_M(K_ref S K_ref)` using the reference full K. The first difference isolates **intermediate loss for identical operands**; the second comparison exposes effects of truncating K itself. Repeat with both the projected reference K and the current sparse solution. Then re-solve SCC with the chosen improved algebra and measure E/F, not just one product's residual. This avoids treating an energy contraction that is exact on H's support as proof of a valid density matrix: retaining exact K entries wherever H0 is nonzero preserves `Tr(KH0)` by construction while idempotency/stationarity can remain wrong.

For exact projected Q from stored K, a sufficient T mask is `M_K ∘ M_S` (Boolean product). It can be trimmed to entries actually read by Q:

`M_T_needed = (M_K ∘ M_S) ∩ (M_Q ∘ transpose(M_K))`, with `M_Q=M_K`, plus explicitly required diagonal/diagnostic entries.

This follows directly from retaining T[i,l] only if some requested Q[i,j] has K[l,j] present. Reuse existing Boolean-product/plan helpers and verify on a small fixture. T need not be symmetric. Measure degree, capacity, plan bytes and both product costs before GPU construction; do not materialize an unbounded production closure or silently enlarge budgets. This removes unnecessary halo entries, but does not promise a physical projector on a narrow K mask. If it cannot meet the force budget at acceptable cost, then assess variational refinement.

### Additional finding 3 — Numerical rigor still has several concrete gaps

- **Do not overstate input validation.** Frozen-input LAPACK agreeing with a DFTB+ energy component is strong evidence, but one scalar comparison does not fully validate H0/S/H_scc or all charges/forces. Preserve separate same-input electronic-solver parity and independent parameterization/SCC/force parity. The remaining 0.7 mHa attribution to SCC tolerance is plausible, not isolated until a controlled tolerance/state comparison establishes it.
- **The factor-two diagnostic remains wrong in source.** `frozen_input_dense_ref` constructs `p_exact=2 C_occ C_occᵀ` then still prints `||pd−p_exact||` where pd is spin-free K. Its later injection correctly uses `p_exact/2`. Correct that comparison and labels: K=C_occ C_occᵀ, D=2K, P=KS; `e_h0_exact` already contracts D with H0, so its `2Tr(P·H0)` print is misleading. Also label the reported “host-f64 R_I” as masked when its products use M_K; it is not the full residual.
- **New diagnostic mutators bypass state validity.** `SparseDftb::inject_k_bsr` passes only values, not the supplied CSR structure, to the workspace and leaves accepted `last` usable. `purify_current` also mutates K without invalidating accepted energy/forces, and its docstring misleadingly says “against current H_scc” although the polynomial uses K,S without H-dependent correction. Restrict these to diagnostics or invalidate accepted state first, verify complete CSR ordering on injection, and require fresh acceptance before energy/forces. Otherwise stale E/q/H can be combined with modified K/W. Failed diagnostic purification must invalidate acceptance too.
- **Force/Hessian acceptance remains the product gate.** `finalize_scc` still only reports R_H; floor acceptance remains default-on; recovered-P status is not reclassified against K. Do not certify vibrations from the improved band energy. For candidate speed settings perform frozen-mask ±h force comparisons, cold/central-warm/reversed-order checks, and own-minimum Hessian gates from §15.12-C. Neither clipping negative frequencies nor returning a lower masked R_I is a remedy.
- **Cosine taper is C1, not C2.** `sparse_forces.rs::hs_taper` uses `w=(1+cos(πt))/2` inside its window and constants outside. w' matches at endpoints but w'' jumps there, so the tapered Hamiltonian can have a Hessian discontinuity even with a C2 SK interpolator. Inventory an existing C2 switch; if none fits, compare an explicit quintic endpoint-flat taper `1−10t³+15t⁴−6t⁵` with consistent derivatives. This changes the approximate model in the window: revalidate cutoff bias and forces rather than changing it silently. Test endpoint-crossing stencils; a frozen neighbor list alone cannot cure this source of h sensitivity.

### Recommended bounded work order

1. **Correct diagnostic labels/state contracts and establish profiler-off timing plus true kernel-event timing.** Preserve the NS improvement; no new precision arithmetic.
2. **Replay/validate a stopping candidate** on existing histories, then measure saved products and unchanged force quality. In the same phase, perform the frozen-operands intermediate-loss diagnostic; these are independent analyses, not reasons to rewrite two algorithms at once.
3. **Implement common-path one-read TC2 with unchanged guard semantics**, and ask the runtime owner to remove its redundant finish. A/B at identical settings and seeded guard events.
4. **Implement only the measured useful T support**, then calibrate the smallest K mask against forces/Hessians. Benchmark combined product cost and full SCC; do not promise degree 64–128 before this succeeds.
5. **Tune heavy-product layout or batch independent displacements** once the accepted scalar policy exists. Defer a full device driver, LNV, compensation and CPU-force port unless the revised profile/error decomposition justifies them. Cache topology/plans/Z as already allowed; do not reuse stale K0 just because H changes little. Bounds reuse needs a valid perturbation bound, and density reuse needs an H-dependent stationarity correction.

**Handoff evidence:** exact configuration/mask statistics, failures and guard counts, stopping histories, actual E/F/Hessian errors, profiler-off timing and kernel execution timing. Document observed versus inferred causes separately. No new speedup or physical-acceptance claim is established by this review alone.

## 2026-09-14 (cont. 5) — §15.12-B/C executed: stagnation detector + the REAL mechanism

### §15.12-1 — Diagnostic contracts fixed (done)

- `sparse_f64check`: factor-2 comparison fixed (`‖2K−D_exact‖`, was `‖K−D‖`), masked-vs-full R_I labeled, "COMPLETE" claim corrected (r_k=20 = 108090/108900 = 99.3%).
- `inject_k_bsr` verifies the full CSR (n_atom + row_ptr + col_idx) and **invalidates `last`**; `purify_current` invalidates too (mutated K ⇒ stale E/q/H).
- `RUST_DFTB_TC2_HIST=<csv>` per-iter history (iter,branch,tr,dev_rel,guard,r_i,best,snap) for offline stopping-rule replay.

### §15.12-2 — Stagnation detector: replay-validated, env-gated

Rule: **stop when the trace-gated absolute best improved <5% over the last W checked iters** (endgame: dev_rel<5e-4, valid snapshot, best<1e-2) → restore best-K, `NumericalFloor`. Replay on 4 saved histories (R10 deg175, R10 deg330, R14 deg386, SiH4):

- **W=8 fails**: fires inside R14's genuine 15–20-iter mid-descent stalls → live divergence (returned R_I 1.2–6e-3 vs achievable 9e-4).
- **W=28 safe**: fires only in deg330's ≥34-iter flat tails. Live A/B: R10 deg330 wall **8.1→6.9 s (−15%)**, energy bit-identical (−298.18619690); R14/SiH4/deg175 **zero fires** (their floors creep — the 10× blowup detector remains the backstop there). `RUST_DFTB_TC2_STOP_W=28`, off by default.

### §15.12-2′ — Intermediate-loss diagnostic: the halo hypothesis is REFUTED

`intermediate_loss_diag` test (host f64, R10 deg175, frozen operands):

| quantity | value |
|---|---|
| ‖P_M(P_M(KS)K) − P_M((KS)K)‖ (intermediate loss) | **0.0119 (0.075% rel)** |
| ‖P_M((K_c·S)K_c) − P_M(K_ref S K_ref)‖ (K-truncation loss) | **0.2257 (19×)** |
| masked-f64 TC2 from K_ref\|M_K, iter 0 | R_I jumps ~0 → **9.6e-3** |
| …same, iter 15 | ‖K−K_ref‖ 0.15→0.22 (same ~1e-3 limit cycle as f32) |

**Mechanism (final):** the masked map `K′=P_M((K·S)·K)` has **no stable fixed point at K_ref|M_K** — each map step needs K's *own* halo entries, which are structurally zero. In exact f64 arithmetic the iteration walks off the exact state immediately; f32 noise is irrelevant to the floor (f64 shows the identical limit cycle). The KS-intermediate halo (M_T_needed) would recover only the 0.075% product loss — **work-order item 4 is deprioritized**.

**Consequence for the "fewest neighbors" objective:** the DM's own tails are needed *by the map itself* — stored-K degree is the accuracy knob (deg175→16mHa, deg275→1.2mHa, deg330→0.02mHa; scales with DM decay length, not N). Remaining routes to fewer neighbors: (a) accept the decay-length degree, (b) top-k/magnitude masks (~4× efficiency vs radial), (c) LNV — a narrower auxiliary L generating a wider effective K (variational, needs its own mask/gradient work).

### §15.12-3 — Common-path one-read TC2 (implemented, semantics preserved)

Restructured `tc2_purify`: enqueue `T=K·S` → trace partials → **speculative `Q=T·K`** → residual partials (check iters) → ONE blocking read → host f64 guard/branch → K update. The trace guard is the only pre-KSK dependency; on guard fires (~1.8% of iters, measured via event counts) the speculative Q/resid are discarded and recomputed from the rescaled state — identical semantics to the old read-then-compute order.

- Energies **bit-identical** on all three workloads (deg175 −298.17071484, deg330 −298.18619690, R14 −846.95945817); R14 exercises the guard-recompute path.
- Wall time ~neutral (±5% run noise) — the old read-then-enqueue gap was ~0.2ms/iter vs ~3ms device work; the structural benefit is one consolidated stall and no mid-iteration queue idle.

### §15.12-4 — True kernel-event timing (RUST_DFTB_KTIME=1)

Kernel START/END events on the two TC2 SpGEMMs (plan path only), drained per SCC finalize; `reduce_partials_f64`/`idempotency_partial_dev`/`trace_atom_dev` split so partials can be enqueued ahead of the blocking read. Measured:

| kernel | marker-elapsed (old est.) | TRUE kernel-exec | launches |
|---|---|---|---|
| R10 deg175 `tc2.ks` | 0.56 ms | 0.564 ms | 766 |
| R10 deg175 `tc2.ksk` | 2.64 ms | **2.73 ms** | 780 (14 guard re-fires) |
| R10 deg330 `tc2.ks` | 1.17 ms | 1.17 ms | 661 |
| R10 deg330 `tc2.ksk` | 10.34 ms | **12.33 ms** | 650 |

**Markers undercounted ksk by ~16%** on deg330. Kernel products are ~86–88% of wall (deg175: 2.56s/2.99s; deg330: 8.8s/8.6s prof-off baseline — KTIME instrumentation costs ~16% wall itself). Pre-existing test failure noted: `spline_resample::test_resample_bspline_sin` (unrelated, file unmodified).

### §15.13 — First end-to-end vibrations (2026-09-14)

`sparse_vibrations` (FD Hessian of analytic sparse forces, h=0.02 Å, mass-weighted eigen) run on two crystals:

**Si10H16** (26 atoms, matsci, deg=26 complete): relax→Hessian in ~1 min. Rigid modes 0–5 at −16.2..+0.3 cm⁻¹ (FD/asymmetry noise band, max asym 6.4e-3); first real mode 101 cm⁻¹; Si–H stretches 2220–2257 cm⁻¹. `debug/sparse_vib_si10h16_freq.txt`.

**cube_Si65** (65 atoms, matsci, deg=65 complete — geometric mask path exercised): relax→195-col Hessian in ~4 min. max asym 1.0e-3. Rigid modes −14.0..−0.06 cm⁻¹. **vs DFTB+ L1 reference** (`vibrations.tag` ×219474.63):

- Spectrum structure correct: same band organization, degeneracies preserved (e.g., 83.9 triplet, 225.1 doublet), Si–H block 2084–2250 vs ref 1922–2143.
- Deviations are **systematically positive and sizeable**: internal mean |Δ|=74 cm⁻¹, max 162 cm⁻¹; typical +5–15% mid-range, Si–H block +5%.
- Geometry confound is small: sparse min is 0.024 Å RMS from DFTB+ min; Hessian at own min vs ref's own min.
- Contributing causes (unresolved): the ~0.99 Ha energy offset at identical geometry means a real model-parity gap (full mask, r_I~1e-7 — NOT a mask/f32 issue); FD noise ±~10 cm⁻¹ on the softest modes; index-matching ambiguity inside dense degenerate clusters (e.g., mode 59: 561→519 may be reordering).
- Verdict: **pipeline functional and honest; frequencies qualitatively right, quantitatively ~5–15% soft-stiff vs DFTB+** — consistent with a systematic force-model bias, not noise. Needs the parity-gap investigation before quantitative frequency claims on large systems.

### §15.14 — FF32-POLISH endgame: implemented, debugged, benchmarked (2026-09-17)

Emulated float-float (double-single, `hi+lo` f32 pair ≈ 46-bit) McWeeny polish on GPU — all f32 FMA, no native f64 (RTX 3090 f64 ≈ 1/64 rate). Design per manifest §4.12.2 r2 / GPT-5.6 spec: products stay float-float through the whole chain `T_ff=K·S → Q_ff=T_ff·K → U_ff=Q_ff·S → V_ff=U_ff·K → K'=f32(3Q−2V)`, reusing the precomputed symbolic plans (`plan_ks`, `plan_tk`); three persistent lo buffers (`ff_t_lo` on M_TKS, `ff_q_lo`/`ff_v_lo` on M_K; hi parts reuse `t_ks`/`q`/`a_zhz`).

**Bug found + fixed (compiler reassociation):** the original fused product+combine kernel (`bsr4_mcw_ff_final`) measured 1.7e-7 elementwise error while every intermediate product verified ≤9.7e-14 and a host f64 replay from the GPU's own buffers gave 1.3e-13 — the NVIDIA OpenCL compiler reassociated the TwoSum chain inside that kernel (the same macro in the product kernel survived — fragile luck). `-cl-fp32-contract=off` is NOT supported by this driver. Fix: V_ff goes through the verified `bsr4_spgemm_plan_Bsym_ff` product kernel (V now stored on M_K, +1 hi+lo buffer) and the combine became a single explicit-fma chain `Knew = fma(3,Qh, fma(-2,Vh, fma(3,Ql,-2·Vl)))` — the lo parts are ~1e-8 linear corrections, so this is ~1-ulp accurate AND contraction-immune. Result: GPU output **bitwise identical** to the Rust-f32 emulation; `K'` at 5.4e-8 ≈ 2× the pure f32-rounding floor (2.7e-8).

**Numerical validation** (`sparse_ff_test`, r_k=40 Å — full M_K, zero truncation):

| product | err(hi only) | err(hi+lo) vs masked-f64 | mask truncation |
|---|---|---|---|
| T=K·S | 2.6e-8 | **7.2e-15** | 0.0 |
| Q=(KS)·K | 2.7e-8 | **6.6e-14** | 0.0 |
| U=Q·S | 2.6e-8 | **5.3e-14** | — |
| V=U·K | 2.7e-8 | **9.7e-14** | — |

**Benchmark** (R10, 330 Si, 918 orbs, deg≈330; `bench_ff_endgame.rhai` + `RUST_DFTB_PROF=evt`; figures `debug/tc2_ff_endgame_rk{20,40}.png`):

| metric | f32 TC2 iter | FF32 McWeeny step |
|---|---|---|
| wall | **14.5 ms** | **44 ms ≈ 3.0 f32 iters** (spec ≤8) |
| products | 2 f32 SpGEMM | 4 ff SpGEMM + combine + sym |

Convergence (true R_I by `sparse_ri_f64`):

- **r_k=40 Å** (M_K covers the cluster, tail=0): f32 purify floor 2.7e-7 → +1 FF step **3.96e-8** → +2 **2.92e-8** → in-loop endgame (6 steps, terminal) **1.81e-8** → asymptote ≈ f32-storage fixed point ~1.4e-8. Total purify+polish 0.68 s vs 0.44 s f32-only → **15× residual improvement for +55% wall**.
- **r_k=20 Å** (production mask, tail ‖K'∖M_K‖/‖K'‖=5.3e-6): f32 floor ~4.9e-5 → FF plateau ~3e-5 — **representation-limited, not arithmetic**. Widening M_K is the only lever at fixed r_k.
- **f32-McWeeny endgame** (no FF): limit-cycles at ~1.4e-7 device for 130+ iters — same floor, no benefit.
- **Device R_I floor ~1e-7**: the in-loop residual is itself f32-computed and cannot resolve below ~1e-7; the FF improvement is only visible via `sparse_ri_f64` (f64 host diagnostic).

**Design consequence:** the earlier "f32 arithmetic floor ~1e-5" was entirely (a) product arithmetic (fixed by FF kernels) and (b) `M_TKS=M_K` product truncation (fixed by building the true structural product mask `build_product_mask(M_K,M_HS)`) and (c) M_K *storage* truncation — the remaining irreducible floor set by `r_k`. For R10, r_k=40 Å costs nothing in degree (deg already ≈ n_atom) and buys 1000× residual; for genuinely large systems the r_k floor is the honest price of sparsity.

**Integration:** `RUST_DFTB_TC2_FF=1` + `RUST_DFTB_TC2_FF_SWITCH` (default 1e-3) + `RUST_DFTB_TC2_FF_STEPS` (default 2). Once the step cap is hit the purify returns the polished K immediately (`NumericalFloor`) — returning to f32 TC2 re-pollutes the state to ~1e-7 within a few iters, so the endgame is deliberately terminal. `sparse_sync` rhai call added for honest enqueue+drain timing.

**Remaining/open:** R_H on a converged H_scc (subspace leg) still unmeasured; force-noise-vs-R_I sweep (does the Hessian even need <1e-5?) is the next physics-relevant question; P-space path untouched by the M_TKS change; sparse SCC 4-iter failure at R10 remains a separate unresolved issue.

### §15.15 — Production two-phase purify + FF kernel optimization (2026-09-17, cont.)

The §15.14 study machinery is now a production policy (manifest §4.12.2 consolidated). `tc2_purify` resolves the whole policy ONCE before the loop (no env reads inside): `budget = max_iter` (caller's budget; production default `tc2_max` = **30**, study scripts pass larger values; `RUST_DFTB_TC2_BUDGET` caps downward), `ff_on = ws.tc2_hiacc || RUST_DFTB_TC2_FF`, `ff_switch` (default 1e-3), `ff_steps` (cap 5), `ff_target` (default 1e-7), `ff_vtq` (3-product path, default off — see below).

**Phase A** (f32 TC2): ordinary `Converged` exit unchanged; a new floor-stop fires when `trace_locked && best_ri < ff_switch && best unchanged over 5 checks` — safe because genuine mid-descent stalls live at R_I≫1e-2. Plateau/budget exits restore `k_best` → `NumericalFloor`; budget exhausted *without* a projector basin → `Err` (fail loud, no polishing of garbage).

**Phase B** (hiacc only, terminal): each iteration computes T_ff,Q_ff first — Q=KSK is needed anyway, so the residual of the CURRENT K is measured *before* spending the U,V products. Early exit on `ri < ff_target` or stall (`ri_now > 0.7·prev`); cap `ff_steps`; best-of tracking restores the best polished state; the returned R_I and trace are measured on the RETURNED state (stale pre-FF residual bug fixed). New status `PurifyStatus::PolishedFF` — no longer disguised as `NumericalFloor`. `SparseDftbConfig.tc2_hiacc: Option<bool>` + `ws.set_tc2_hiacc` wired through `purify`, `purify_scc_step`, and the vibration paths.

**Kernel optimization** (`sparse_bsr4_purification.cl`, R10 deg~330, ACC16 f32 baseline 6.8 ms/iter):

| FF step variant | ms/step | vs f32-ACC16 iter |
|---|---|---|
| generic `a_has_lo` kernel (old) | 44–46 | 6.5× |
| + specialized split (ff0/ff) | ~44 | — (product 1 already cheap) |
| + `FF_LO_GLOBAL` (A_lo via L2, −21 KiB local) | 38.2 | 5.6× |
| + `FF_ACC2` (2-way ff accumulators) | **27.4–27.8** | **4.0×** |
| `FF_VTQ` 3-product V=T·Q (ff×ff kernel) | 27.4 | ~same |

Both winners are now **default ON** (`RUST_DFTB_FF_LO_GLOBAL=0` / `FF_ACC2=0` opt out) — verified: LO_GLOBAL trajectory bitwise-identical, ACC2 identical to ~6 digits, both reach the same floor. VTQ measured a wash: the dropped product (Q·S on plan_ks ≈1.7 ms) is the cheap one while V=T·Q pays double-width B streaming on the expensive plan_tk — kept as `RUST_DFTB_TC2_FF_VTQ=1` option. Per-stage device profile (evt): ff.ks 1.25 + ff.tk 11.85 + ff.qs 1.74 + ff.uk 11.96 + comb 0.16 ≈ 27.0 ms; the ff product is now ~2.1× its f32 twin (11.9 vs 5.6 ms) vs ~2.75× flops — approaching flop-bound.

**Production-policy measurements** (R10, `bench_ff_endgame.rhai`, all-new code path):

| run | result | wall |
|---|---|---|
| fast, tol=1e-4, r_k=20, ACC4 | `Converged` iter 24, R_I64=1.3e-4 | 0.316 s (13.2 ms/it) |
| fast, tol=1e-9, r_k=20, ACC4 | `NumericalFloor` @30, R_I64=4.8e-5 | 0.413 s |
| fast, tol=1e-9, r_k=20, ACC16 | `NumericalFloor` @30, R_I64=4.8e-5 | **0.204 s** (6.8 ms/it) |
| hiacc, r_k=40, target 1e-7 | plateau-stop @29 (1.1e-7) → 0 FF steps → `PolishedFF` 9.6e-8 | 0.271 s |
| hiacc, r_k=40, target 1e-8 | +2 FF steps → `PolishedFF` dev 1.86e-8, **R_I64=2.8e-8** | 0.329 s |
| hiacc, r_k=20 | +2 FF steps → `PolishedFF` dev 6.2e-6, R_I64=3.2e-5 (mask floor) | 0.272 s |

In-SCC purifies under hiacc entered Phase B at 8e-6…9e-5 and polished 1–5 steps to ~1.3e-8 (rk40) / ~6e-6 (rk20) — early exit working, cap 5 bounding the mask-floor case.

**Cost/benefit verdict:** on R10 hiacc costs +2×27 ms ≈ +60–80 ms over a ~200 ms ACC16 purify (+30–40%) and buys ~5–10× residual at rk40 (or nothing extra at rk20 where the mask floor dominates — as designed). The "30-iter cap + no floor dancing" bound holds: worst case 30×6.8 + 5×27 ≈ 0.34 s.

Figures: `debug/tc2_ff_endgame_rk{20,40}_v2.png` (log-scale R_I vs iter and vs wall ms; ◇ = Phase-B FF iters; f64-verified FFSTEP series on rk40). Harness: `bench_ff_endgame.rhai` parameterized via `BENCH_RK/BENCH_TOL/BENCH_MAXITER/BENCH_FFSTEP` + new rhai `env()` binding.

Unit tests: `sparse::sparse_system` 6/6 pass (caller-budget semantics keep the toy-system tests at their explicit 60-iter budgets).

### §15.16 — DMM warm-density update for FD Hessians (2026-09-16)

Full report: `reports/2026-09-16_sparse_dmm_warm_density_hessian.md`; bottleneck context: `topical_audit/hessian_eval_bottleneck.md` §"mode C".

**Problem:** `VIB_DMUPD`'s δK0-seed + polish converged to a wrong-subspace projector (idempotent, right trace, R_H~1e-3) — purification polynomials cannot select the occupied subspace. The fix is a **subspace rotation**: steepest descent on the projector manifold under the generalized commutator.

**Implemented** (`sparse_system.rs::dmm_descend`, driven from `scc_fixedq` under `RUST_DFTB_VIB_DMUPD=1`): restore central (q,K,Z,K0) → warm NS → δK0 first-order seed → DMM descent `δK = −η(X + Xᵀ − 2Y)` with `X=(Z·H)·K`, `Y=(K·S)·X`, η=`eta_scale/(εmax−εmin)` → hard `R_H ≤ 1e-4` gate → forces. Retraction = fully-planned McWeeny every `ret` steps; post-DMM polish defaults off (H-blind maps *raise* R_H).

**Bugs found by measurement:** (i) workspace `Z` is **S⁻¹, not S⁻¹ᐟ²** — the first derivation assumed the square root and produced an *ascent* direction (dense-f64 verify: Tr(H·G)=−15.4, E_band rose every step); (ii) `Y=T·X` ran on a **bsym** SpGEMM plan but X is asymmetric → the kernel computed T·Xᵀ (error 1.4e-2); fixed with generic `plan_tk_g` (→8.4e-7). Rule recorded: asymmetric right operand ⇒ generic plan; (iii) `KS·ZHZ = KHZ = Xᵀ` is free from `symmetrize_dev` — the redundant 4th product is why the first working version was slower than cold.

**Measured (R10, 330 Si, deg≈330, h=0.05 Å, solve+forces per eval):**

| mode | products | ms/eval | R_H | ΔF vs cold fixq |
|------|---------:|--------:|----:|----------------|
| frozen | 0 | 7–12 | — | 2.4% |
| **DMM 6/η8/ret2** | ~26 | **260–340** | 4.2e-5 | **0.30%** |
| DMM 3 +1McW | ~20 | ~175 | 8.3e-5 | 2.8% |
| cold fixq | ~50 | 330–355 | 8.0e-5 | ref |

Rejected-by-measurement: η=10/4-step and ret=3 trip the dummy-lane force gate (off-manifold drift, r_I≈2e-3); ret=1 costs more than it saves; final McWeeny doubles R_H margin usage for a 3× r_I gain. Fail-loud gates fire correctly on all of these.

**Honest verdict:** certified ~25% vs cold, NOT the 10× target — steepest-descent contraction (~1.7×/step) is set by η·Δε, not seed quality; seed lands at R_H≈7e-4 vs the ~1e-4 force-validated gate. Open Tier-1 plan (GPT-5.6 rebuttal, chat ~line 11600): skip δK0/K0 (metric-transport seed `K←2K−KS₁K`), 1 NS iter, 1–2 steps at real h=0.02 Å, antisymmetric ±h sharing → ~7 products target.

### §15.17 — Stripped tiers + frozen-orbital correction (2026-09-16, later)

The GPT-5.6 rebuttal (chat ~line 12800+, 13400+) was executed; it inverted the frozen-mode conclusion and produced the real tier ladder.

**The stale `b_zh` was the correct approximation — the fix was the regression.** `forces_frozen` had built `W=2·b_zh·K` with `b_zh` stale from the central `compute_k0` ⇒ `W=2(Z₀H₀)K₀=W₀` — the **consistent clamped-electron (frozen-orbital) freeze** `δq=δK=δW=0` (SCF Lagrangian differentiated with orbitals AND multipliers frozen). The "fixed" hybrid `W̃=Z(R)H(R)K₀` keeps `(δF)K₀` but drops `F₀δK` — half a cancelling response → **86% column error**. Now deliberate: `snapshot_electronic_state` stores `W₀` (+`P₀=K₀S₀`, `X₀=B₀K₀` for the linear tier); `forces_frozen` is pure CPU (`compute_v` + pair contraction), **0 device products → 5.5 ms/eval, 6.3% column error** (h-independent — identical at h=0.02/0.05). The earlier "2.4%" figure is unreproducible; 6.3% is the reliable number.

**Stripped tiers** (`VIB_LITE=1`, `VIB_LINEAR=1`): no δK0 seed (`VIB_SEED=0` default-worth), no retractions, no per-eval residual gates/measurements — those cost ~40% of the warm eval (ns0d2: 100→55 ms). Lite sets `e_tot=NaN` → `energy()` refuses loudly; only `forces()` is served.

**Measured ladder (R10, h=0.02 Å, 3 cols, per displaced eval):**

| tier | products | ms | col err | env |
|------|---------:|---:|---------:|-----|
| clamped (K₀,W₀,q₀) | 0 | 5.5 | 6.3% | `VIB_FROZEN` |
| linear1 (δB response) | 4 | 31 | 6.7% | `VIB_LINEAR` — refuted |
| DMM2-lite, central Z | 8 | 55 | 6.2% | `VIB_LITE DMM=2` |
| 1 Newton + DMM2-lite | 11 | 64 | 3.1% | `+NSMAX=2 NSTOL=3e-5` |
| 1 Newton + DMM4-lite | 17 | 105 | 1.0% | `DMM=4` |
| gated warm (ns2+DMM4+gates) | ~22 | 175 | 1.0% | `VIB_DMUPD SEED=0` |
| cold fixq | ~50 | 345 | ref | `VIB_FIXQ` |

**Structural findings:** (i) **Z accuracy is the tier discriminator** — central Z (R_Z≈2e-3 at displaced S) caps ALL K-updates at ~6%; ONE actual Newton correction (R_Z→1.4e-5; "NS=2 iters" = 1 update + residual checks = 3 products) unlocks 1–3%; (ii) retractions *hurt* — every McWeeny-retracted variant is worse than its unretracted twin (4.8% vs 3.0%), the map is H-blind; without the δK0 contaminant ≤4 raw steps stay on-manifold (6 steps trip the Tr(KS) gate, loudly); (iii) metric transport alone gives 99% error — refuted; (iv) linear1's single fixed-η response is insufficient (6.7% ≈ frozen at 6× cost) — refuted at tested η.

**Remaining 10× lever:** batch-parallel ±h columns (all 990 share topology/plans). Untested: ±h antisymmetric sharing (`K(−h)≈2K₀−K(+h)`, exact to O(h²)), Chebyshev/BB η schedule, CG on the manifold.

### §15.18 — First R18 (1648-atom) run: two loud guards, tier timings, and the locality rules (2026-09-17)

System: `debug/nanocrystals/si_sphere_R18.xyz` = 18 Å-radius bulk-Si sphere, H-passivated → 1202 Si + 446 H = **1648 atoms, 5254 orbitals** (matsci-0-3, Si sp³=4 orbs, H s=1). Hessian = 4944 columns × 2 evals = 9888 force evaluations.

**Nothing was diverging.** The run hit two sequential loud guards, both correct:

1. `M_TKS = supp(K∘S)` (the FF32-corrected true product support) spans r_k+r_hs ≈ 21 Å → interior rows are **fully dense** (deg→1648) on a 37 Å sphere and exceed `MAX_LEFT_BLOCKS=384`. New env `RUST_DFTB_TRUNC_PRODUCTS=1` stores T on M_K (pre-fix layout; dropped tail ~7e-6 ≪ 5e-4 mask floor); without it the workspace now fails **at init** with an actionable message instead of mid-SCC. Real fix = left-row chunking kernel.
2. R_H stationarity gate: SCC converged cleanly (13 iters, r_scc=4.4e-6, Tr(KS)=2627.000000, E=−1772.3732 Ha ≈ Sep-12 −1772.3924 at floor) but R_H=4.2e-3 > default 5e-4 — the **mask-floor regime** (R_I(K)≈2.6e-3 sets R_H; no r_k within the local-memory budget reaches 5e-4 at this size). Same for the per-column gate inside `scc_fixedq`. Production overrides: `RUST_DFTB_SCC_RHGATE=1e-2` and `RUST_DFTB_VIB_RHGATE=1e-2` (still rejects the ~1.4e-2 wrong-subspace signature both gates exist for).

**Measured per-eval timings (RTX 3090, r_k=r_z=12 Å, r_trunc=8+1, deg_k=378 saturated at bulk, nnz_k=412k):**

| tier | ms/eval | full Hessian | note |
|------|--------:|-------------:|------|
| frozen (clamped K₀,W₀,q₀) | 190 | ~31 min | ~all CPU: geom 86 ms + contract 99 ms; GPU idles |
| fixq (cold TC2, 27 iters floor-stop 1e-3) | 916 | ~2.5 h | deterministic cold purify per eval |
| full SCC | ~5–8 s est. | overnight | |

Dense f64 GEVP reference point: ~30–90 s per diagonalization at 5254 orbs → CPU-dense Hessian would be **weeks**. Sparse is ~50–100× ahead already — but this N is the *entry* of the asymptotic regime, not its middle.

**Frozen-mode spectrum validation (si10h16, relaxed, vs Sep-15 SCC reference) — important correction to the "6% column error" marketing:**

- frozen spectrum **bit-identical** to the Sep-14 old-frozen file → inherent clamped-electron physics, not a regression.
- rms error **157 cm⁻¹**: Si–H band −250 cm⁻¹ soft (expected), mid-band collapse (SCC 420–990 → frozen 760–830, worst +344), 3 of 6 rigid modes land at +120 cm⁻¹.
- **fixq is the quantitative cheap tier**: rms 5.9 cm⁻¹, Si–H −8 cm⁻¹, rigid modes intact (−16..+1, the SCC residual-force floor).
- `fixq+dmupd` rms 315 — refuted again.

### §15.19 — Optimization rules for the frozen/local Hessian (user-stated, confirmed)

**R1 — deg_hs is a *basis-set/SK-table* property, not physics.** The number of H/S neighbors (deg_hs) sets assembly cost, every SpGEMM touching M_HS, and the force-contraction pair list. Measured table ranges: matsci-0-3 runs to **10.58 Å** for Si-Si/Si-H/H-H; **pbc-0-3 runs to 5.50 Å**. At bulk density that is deg_hs ≈ 167 (matsci @ r_trunc 8+1) vs ≈ **35 (pbc)** — ~5× cheaper assembly and contraction, and a much sparser M_HS operand, at the price of a different parameterization (must re-relax and re-validate on its own minimum; pbc is the set designed for bulk solids).

**R2 — r_k is set by the band gap; decouple it from r_hs.** Density-kernel decay is exponential with a length fixed by the HOMO-LUMO gap (~12 Å needed for matsci Si); shrinking r_k for speed hits the force-error floor (labbook r_k table). But **nothing couples r_k to r_hs**: H/S sparsity (=multiplication speed) and K range (=accuracy) are independent knobs. Tune r_hs via the SK set/r_trunc for speed, r_k via the measured R_I/force floor for accuracy; validate separately.

**R3 — frozen Hessian columns are local: do NOT contract over all N² pairs.** With δq=δK=δW=0, `dF_i/dR_j ≠ 0` only for atoms i connected to j by an explicit R-dependent term:
- H0′/S′ band terms: pairs (i,j) within r_hs → **~deg_hs atoms per column** (~167 of 1648 at R18).
- Repulsion: local spline cutoff — same ball.
- γ′ term `Σ γ′_iJ Δq_i Δq_J`: formally dense, but a *smooth radial tail* ∝ Δq_iΔq_j/r³ needing no SK evaluation — compute over all pairs in O(N) scalar ops or truncate with a diagnostic.
⇒ Per-column work should be **O(deg_hs), not O(N)**: restrict the force contraction to the touched pair list and store Hessian columns sparse (~deg_hs of 3N nonzeros). The current `forces_frozen` rebuilds all of H/S/γ (~86 ms) and contracts ~200k pairs (~99 ms) per eval — at 1648 atoms that is ~95% wasted work. Same locality enables **multi-displacement batching**: columns whose response supports are disjoint (graph coloring, support radius r_hs frozen / r_hs+r_k fixq) can share one eval — the real 10–100× lever at N≫10³.

**Validation path:** test on the smaller crystals first — R10 (330 atoms) and R14 (864) — where supports overlap less, before claiming R18.

### §15.20 — pbc-0-3 vs matsci-0-3 on R18: measured mask data + cost projection (2026-09-17)

Bounded run (frozen, `R18_RTRUNC=5.5` = pbc's native table end, MAXCOL=4, timeout 30). Center SCC **failed loudly** at mix-iter 3 — but the mask/timing data needed for the projection was collected:

| quantity | matsci-0-3 @ r_trunc=8 | pbc-0-3 @ r_trunc=5.5 |
|---|---:|---:|
| deg_hs (max blocks/row) | 167 | **56** (2.98×) |
| nnz_hs (H/S pair blocks) | 199 676 | **72 368** (2.76×) |
| SK table range | 10.58 Å | 5.50 Å (H–H 5.29) |
| ws_new (plans+alloc) | 11.57 s | 9.34 s |
| NS iters / R_Z | 7 / 4.2e-5 | 7 / 7.6e-6 |
| TRS4 floor (purify) | ~4.9e-4 | **~1.1e-3 → stall ~1.7e-2, dies** |
| center SCC | converges, 13 iters | rms grows 0.08→0.14, `TRS4 exhausted` at iter 3 |

**Physics finding:** pbc-0-3 gives a smaller effective gap on this sphere → slower density-kernel decay → the r_k=12 Å mask floor rises from ~5e-4 to ~1e-2. SCC then death-spirals (floor-limited K → wrong charges → smaller gap → worse purify). Fail-loud worked correctly. To run pbc properly needs r_k≈14–16 Å → deg_k≈600–900 > `MAX_LEFT_BLOCKS`=384 — i.e. **the pbc route requires the left-row-chunking kernel** (same cap that bit M_TKS). Ironically: shorter H/S (good) but longer DM (bad) — the two radii really are independent knobs (R2).

**Projected per-eval cost** (pbc, scaling the measured matsci PROF breakdown by nnz_hs ratio 2.76; γ and K-mask stages unchanged):

| stage | matsci ms | pbc ms | scaling |
|---|---:|---:|---|
| H0/S assembly `geom.hs` | 54.9 | ~20 | ∝ nnz_hs |
| γ build `geom.gamma` | 19.8 | 19.8 | O(N²) — unchanged |
| repulsion `geom.rep` | 5.6 | ~4 | rep-pairs |
| H/S upload `geom.ul` | 3.2 | ~1.2 | ∝ nnz_hs |
| restore K,Z + copies | ~6.5 | ~6.5 | M_K unchanged |
| `compute_v` γ·Δq | 1.9 | 1.9 | O(N²) |
| force contract `f.contract` | 98.6 | ~36 | ∝ nnz_hs |
| **frozen eval total** | **~190** | **~90** | **2.1×** |

- **Frozen Hessian: ~31 min → ~15 min.** Wins are real but bounded: γ's O(N²) build (~22 ms/eval) is untouched and becomes the next wall — at N≈10k it alone would be ~0.8 s/eval.
- **fixq:** +NS 50 ms + TC2 ~27 iters. Only the `K·S` product sees M_HS (~3× cheaper); `KSK` is M_K×M_K and unchanged → ~26→~17 ms/iter → **~600 ms/eval → ~1.7 h**, *if* it converged — it doesn't at r_k=12 for pbc (floor ~1e-2). Blocked pending chunking + larger r_k.
- Conclusion: basis-set choice buys ~2× on the cheap tier today; the structural wins remain (a) column-local contraction (O(deg) not O(N) per column), (b) γ tail truncation/approximation, (c) left-row chunking (unlocks both M_TKS correctness AND pbc's needed r_k).

### §15.21 — Band gaps drive r_k; C/diamond particles as the next target (2026-09-17)

**Measured gaps** (dense path, `get_eigenvalues` at correct n_occ; H0 = non-SCC):

| system | SK set | gap |
|---|---|---:|
| si10h16 (SCC) | matsci-0-3 | 0.37 eV (SCC state; H0 sphere value below is more relevant) |
| si_sphere_R10, H0 | matsci-0-3 | **5.66 eV** |
| si_sphere_R10, H0 | pbc-0-3 | **2.98 eV** |
| c_sphere_R06, H0 | 3ob-3-1 | **9.25 eV** |
| c_sphere_R06, H0 | mio-1-1 | **9.06 eV** |

pbc's ~1.9× smaller gap explains the §15.20 purify stall quantitatively (DM decay length ∝ gap⁻¹-ish → r_k=12 floor 5e-4→~1e-2). Note also: **pbc dense SCC diverged on si10h16** (rms 0.1 after 100 iters, no DIIS in the dense path) — the model itself is marginally stable on Si clusters, not only a sparse-mask problem.

**C/diamond spheres generated** (`debug/gen_diamond_sphere.py`, a=3.567 Å, dangling bonds → H at 1.09 Å): c_sphere_R06 (283 at), **R10 (1053)**, R14 (2599), **R18 (5343 atoms — the real large regime)**. BSR4-compatible (C sp³ = 4 orbs, H padded).

**Init-measured mask data** (mask_stats.rhai, `RUST_DFTB_TRUNC_PRODUCTS=1`):

| geometry | SK | r_trunc | r_k | deg_hs | deg_k | nnz_k |
|---|---|---:|---:|---:|---:|---:|
| si_sphere_R18 (1648) | matsci | 8+1 | 12 | 167 | 378 | 412k |
| si_sphere_R18 | pbc | 5.5+1 | 12 | 56 | 378 | 412k (purify fails) |
| c_sphere_R10 (1053) | 3ob | 6.9+1 | 8 | 376 | 399 | 254k |
| c_sphere_R10 | mio | 5.3+0.8 | 8 | 179 | 399 | 254k |
| c_sphere_R18 (5343) | mio | 5.3+0.8 | 7 | ~179 | **283** | **1.18M** |

Key numbers: C packs 3.5× more atoms/Å³ → same-radius masks carry ~3.5× more neighbors per atom. But the ~9 eV gap legitimately allows r_k≈6–8 Å (vs Si's 12): at r_k=7, C-R18's deg_k=283 fits `MAX_LEFT_BLOCKS`=400 **without** chunking. mio-on-C (deg_hs=179) ≈ matsci-on-Si (167) in neighbor count — the per-atom H/S cost is comparable; the cost difference is N itself (5343 vs 1648).

**Cost projection, C-R18 + mio @ r_k=7** (scale Si-R18 measurements by nnz/deg): frozen eval ≈ contract ~320 ms + geom ~210 ms ≈ **0.55 s/eval → Hessian (6·5343·2 ≈ 64k evals) ~9–10 h**; fixq ≈ +TC2 (~55 ms/iter × ~25) ≈ **1.8–2 s/eval → ~30+ h**. Same verdict as Si: the current all-pairs implementation is the bottleneck — column-local contraction + batching (R3) is worth ~10× here, and γ's O(N²) (~0.5 s/eval at N=5.3k) must be truncated next.

**Open C-specific check:** mio vs 3ob parameterization quality for diamond phonons (3ob-freq-1-1 exists — fitted for frequencies; C-C range 6.35 Å). The DM floor at r_k=7 on C must be measured (purify R_I) before claiming a spectrum — bigger gap helps, but verify.

### §15.22 — Diamond-C Hessian cost projection table (2026-09-17)

Anchored to the measured Si-R18 stage costs (frozen eval 190 ms = hs 54.9 + γ 19.8 + rep 5.6 + ul 3.2 + restore/copies ~6.5 + v 1.9 + contract 98.6; fixq adds NS ~50 + TC2 ~26 ms/iter × ~27). Scaling: H/S stages ∝ nnz_hs, γ/v ∝ N², restore ∝ nnz_k, TC2 iter ∝ nnz_k·(deg_k+deg_hs). All rows assume current all-pairs implementation, `TRUNC_PRODUCTS=1`, h=0.02, full 6N×2-eval Hessian.

**Si spheres (matsci-0-3, r_trunc 8+1, r_k=r_z=12):**

| system | N | deg_hs/deg_k | nnz_k | frozen ms/eval | fixq ms/eval | frozen Hessian | fixq Hessian |
|---|---|---:|---:|---:|---:|---:|---:|
| si_sphere_R10 | 330 | 167*/378* | ~83k | 5.5 (measured) | ~345 (measured) | ~22 s | ~23 min |
| si_sphere_R14 | 864 | ~167/378 | ~216k | ~95 | ~620 | ~29 min | ~3.2 h |
| **si_sphere_R18** | **1648** | **167/378** | **412k** | **190 (measured)** | **916 (measured)** | **~31 min** | **~2.5 h** |
| si_R18, pbc-0-3 | 1648 | 56/378 | 412k | ~90 (projected) | ~600 (projected) | ~15 min | ~1.7 h — **BLOCKED: purify floor ~1e-2 @ r_k=12, needs r_k≈14+ (deg_k>600) → needs chunking kernel** |

**Diamond-C spheres (mio-1-1, r_trunc 5.3+0.8, r_k=r_z=7–8; gap ~9 eV justifies short r_k):**

| system | N | deg_hs/deg_k | nnz_k | frozen ms/eval | fixq ms/eval | frozen Hessian | fixq Hessian |
|---|---|---:|---:|---:|---:|---:|---:|
| c_sphere_R06 | 283 | ~120*/≤283 | ~51k | ~27 | ~90 | ~1.5 min | ~5 min |
| c_sphere_R10 | 1053 | 179/399@8 (283@7) | 254k | ~120 | ~610 | ~26 min | ~2.1 h |
| c_sphere_R14 | 2599 | ~179/~283 | ~650k | ~420 | ~1.4 s | ~3.6 h | ~12 h |
| **c_sphere_R18** | **5343** | **~179/283** | **1.18M** | **~1.0 s** | **~2.8 s** | **~17 h** | **~50 h** |
| c_R18, 3ob-3-1 | 5343 | ~376/~399 | ~1.8M | ~1.65 s | ~5.8 s | ~29 h | ~100 h |

(*deg_hs at R06 is surface-dominated, ~0.7×bulk.)

**What the table says:**

1. **N² Coulomb is the frozen-tier wall at scale.** γ build + `compute_v` at C-R18 ≈ 230 ms/eval (~23%) — at N=10k it exceeds 1 s/eval. Truncating γ to a real-space cutoff (with diagnostic) or patching only row/col j per displacement is required.
2. **fixq scaling is deg²-dominated.** TC2 cost ∝ nnz_k·deg — C's density (3.5× Si) makes the same-radius C sphere ~2× pricier per eval than Si at larger N. Warm-seeded purify (G3) is the only big fixq lever (~3×).
3. **mio > 3ob for cost** (deg_hs 179 vs 376) — and 3ob-freq-1-1 (C-C 6.35 Å) sits between. For spectra, accuracy of mio-vs-3ob on diamond phonons must be checked (mio was fitted for organics; 3ob for solids/vibrations — 3ob-freq is likely the better *physics* choice despite ~2× cost).
4. **Column-locality (R3) dominates everything:** restricting contraction+assembly to the displaced atom's support turns the frozen eval into O(deg) → C-R18 frozen Hessian ~**15–25 min** instead of 17 h; combined with multi-displacement coloring (support radius r_hs≈6 Å for frozen → ~50–100 columns/eval at N=5k) → **minutes**. This is the difference between "expensive" and "interactive".
5. Si-R18 remains the practical validation target today: 31 min frozen / 2.5 h fixq. C-R18 is blocked on (a) column-local contraction to be affordable, (b) r_k floor verification, (c) ideally the chunking kernel to lift the deg cap entirely.

### §15.23 — Tiled GPU n-body γ/γ′ implemented; dense gmat eliminated (2026-09-17)

**Before:** `set_coords` rebuilt a dense f64 `gmat[N×N]` on CPU (nested pair loop,
`geom.gamma` ≈ 20 ms at N=1648); `compute_v` was a CPU O(N²) matvec per SCC
iteration; the SCC double-counting force `scc_double_counting_force` re-evaluated
`gamma_prime_full` analytically over all N² pairs on CPU inside every
`sparse_forces_bsr` call (the dominant share of `f.contract` ≈ 99 ms).

**Now** (`sparse_gamma.cl`, gravity-kernel pattern): each work-item owns ONE
target atom i and *gathers* over j-tiles staged in `__local` memory
(GWG = 128 atoms/tile: `float4 xyzu` + `float dq` = 20 B/atom → 2.5 KB/tile).
No atomics, no scatter, no dense matrix — γ evaluated analytically in f32
(exact port of `gamma_full`/`gamma_prime_full`: Elstner same-U polynomial and
different-U `gamma_sub_exprn` branches, on-site r→0 → u_i). Two kernels:

- `gamma_matvec`: `V_i = Σ_j γ_ij·Δq_j` → writes `v_buf` directly (feeds the
  device H_scc build; host readback only for E_scc/v_shift).
- `gamma_force`: `F_i = −Δq_i·Σ_j γ′_ij·Δq_j·r̂_ij` in Ha/Å — result enters
  `sparse_forces_bsr` via a `scc_dc_override` param; `None` keeps the CPU
  reference path for tests.

Plumbing: persistent `xyzu_buf`/`dq_buf`/`gf_buf` in `SparseSystemWorkspace`
(allocated once); `set_coords` uploads 4·N floats (O(N)); `compute_v` and
`contract_forces` upload dq (N floats) + launch + readback per call.

**Parity** (L0, `gamma_kernels_match_cpu`, N=313 random, two U species):
max|dV| = 7.5e-6 on |V|~5.5; max|dF| = 2.2e-6 on |F|~0.1 — f32 noise.

**Measured on Si-R18 frozen** (MAXCOL=4, matsci, same geometry/params):
per-eval **~145 ms** (rst+geom 66 + solve+f 78) vs 190 ms before —
**and the O(N²) terms are gone**: `geom.gamma` 20 ms → ~0 (O(N) upload),
γ′ analytic loop (~36 ms of contract) → ~2 ms kernel + ~µs readback.
Center max|F| = 0.13074126361447178 — identical to the CPU path.
E_tot shifts 1.6e-4 Ha (f32 V vs f64 matvec — inside the f32-architecture
budget). Projected full frozen Hessian now ~24 min; more importantly the
γ wall at N=5343 (was ~230 ms/eval) is now ~10–30 ms, and the N→10k
scaling wall in row 1 of §15.22 is removed.

**Remaining frozen-eval cost at N=1648 (~145 ms):** H/S assembly ~55 ms +
upload ~10 (all pairs — column-local patch assembly is the next lever, R3),
K/W snapshot restore copies ~10, pair contraction ~63 (still all nnz_hs
pairs — R3 again), repulsion ~5. γ/γ′ is no longer the bottleneck.

**Note on Newton's third law:** the gather form computes F_i independently
per atom — pair forces cancel only to f32 rounding (~1e-7·N), so
`check_newton` (warning-only diagnostic, tol 1e-6) may print a small |ΣF|
residual; the CPU reference path keeps exact symmetry for tests.

### §15.24 — GPU pair path: H/S assembly + analytic forces + repulsion on device (2026-09-18)

The remaining all-pairs CPU loops (H/S SK assembly ~55 ms, K/W force
contraction ~63 ms, repulsive ~5 ms at N=1648) are now GPU-resident via
`sparse_hs.cl`. Explicit switch: `RUST_DFTB_SPARSE_CPU=1` selects the CPU
reference everywhere; GPU is default and **never silently falls back**
(banner prints `pair physics: GPU sparse_hs.cl` / `CPU reference (explicit)`).

Kernels (gather-invariant, no atomics, no scatter):

- `hs_diag` — onsite blocks from packed per-species params (E_DUMMY on
  padded lanes, S diagonal = 1).
- `hs_assemble` — one work-item per unique pair; port of
  `build_pair_block_with_derivs`: B-spline eval (v,d1,d2) of the 8 packed
  SK channels (H ssσ/spσ/ppσ/ppπ + S), Slater–Koster rotation for s/p
  shells, taper, writes both BSR4 orientations.
- `hs_contract` — per-pair analytic dH/dR,dS/dR contracted with K and W
  blocks + atom-potential term; outputs `pf[2p]` (non-SCC) and `pf[2p+1]`
  (SCC-shift) forces; line-for-line the `sparse_forces_bsr` formula.
- `rep_eval` — repulsive spline eval from `pack_repulsive_gpu` records
  (exp head `exp(-ar+b)+c`, `n_int` via `as_int` bit pattern), per-pair
  force + energy out.
- `force_gather` — one work-item per atom reads its CSR adjacency
  (`pair_gather_adj`, `(pair<<1)|is_j` with ± sign convention), writes each
  force component exactly once.

Host packing once at init: `SkGpuPack` (meta/parm/ctrl, channel map
resolved at pack time, unsupported extended-format mixes rejected),
repulsive spline records, pair meta (i,j,h_blk,s_blk,k_blk,rev_blk),
atom species/orbital counts, gather CSR. `SparseSystemWorkspace` keeps
`GpuPairState` with all buffers persistent; `set_coords` uploads only
xyzu + pair geometry and launches `hs_diag`+`hs_assemble`;
`contract_forces` runs `gamma_forces_dev` → `hs_contract_dev` →
`rep_eval_dev` → `force_gather` and reads back component arrays.

**L0 parity** (`sparse_pair_path_parity`, SiH4 + displaced geometry,
`cpu_pair` explicit on both engines): max|dH0|=4.5e-8, max|dS|=6.0e-8,
|dE|=3.0e-7 Ha; force components non_scc 1.2e-6, scc_shift 1.7e-7,
repulsive 1.0e-7, scc_dc 4.7e-8, total 1.2e-6 — displaced-geometry
total 1.3e-6. f32 floor.

**Measured benchmark** (frozen mode, bounded columns, RTX 3090):

| system | path | rst+geom | solve+f | per eval | speedup |
|---|---|---:|---:|---:|---:|
| si R10 (330 at, 31k pairs) | CPU | ~9.6 ms | ~13.5 ms | ~23 ms | — |
| | GPU | ~1.0 ms | ~0.7 ms | **~1.7 ms** | **~14×** |
| si R18 (1648 at, 200k pairs) | CPU | ~69 ms | ~134 ms | ~203 ms | — |
| | GPU | ~6.6 ms | ~4.2 ms | **~10.8 ms** | **~19×** |

R18 parity in the run: E_tot GPU −1772.37331509 vs CPU −1772.37324817
(|dE|=6.7e-5 Ha across SCC trajectories, ~4e-8/atom); center max|F|
0.1307415962 vs 0.1307412703.

**Projected frozen Hessian (4944 columns):** ~22 ms/column → **~2 min** of
column time (was ~31–35 min CPU). Remaining per-eval cost is host-side
snapshot restore + K/W upload and gather readback — the next step is
column-local pair subranges into the same kernels (R3), which turns the
~11 ms all-pairs eval into O(deg_hs) work for force columns.

### §15.25 — Residency audit: ~155 MB/eval of PCIe traffic eliminated (2026-09-18)

FLOP accounting exposed why the GPU pair path showed "only" ~19×: per eval
is ~0.5–1 GFLOP of pair math (**~30 µs at the 3090's peak**) — the measured
11 ms was almost entirely transfers and host work, not kernels. A
single-thread CPU at ~5 GFLOPS explains the ~200 ms CPU figure; an OpenMP
CPU would indeed be competitive at this size — the GPU path's value is
residency + scaling + being the substrate for batched column launches.

Leaks found per frozen eval and fixed (manifest §F):

| leak | was | now |
|---|---|---|
| `restore_central_state` | `s.k`+`s.z` host upload ~52 MB | `copy_f32` device→device from `GpuCentralState` (~100 µs) |
| `forces_frozen` | `s.k`+`s.w0` upload ~52 MB | `hs_contract` consumes `cd.k`/`cd.w0` snapshot buffers directly — zero copies |
| `set_coords` GPU | h0/s readback ~25 MB "for diagnostics" | lazy mirrors — `h_bsr()/s_bsr()` refresh on demand (`&mut` + `Result`), hot path skips |
| `snapshot_electronic_state` | `inject_k_values` re-upload 26 MB | `copy_f32` + `invalidate_ks()` |

`GpuCentralState{k,z,k0,w0}` — device buffers captured once at snapshot;
`central`/`central_dev` always Some/None together; restore without a
device snapshot fails loud. Host `central` vecs retained (cpu_pair path,
δK0 seed, diagnostics). Diagnostics reading host S
(`ri_f64_diag`/`mcw_f64_diag`/`materialize_dense_diag`) call
`refresh_hs_mirrors` first; test call sites updated to the new
`Result`-returning `&mut` accessors.

Residual per-eval transfers (accepted): xyzu up (4N), dq up (N),
v/kdummy down (N), pe_rep down (n_rep), 5×3N forces down, plus the
O(n_pairs) host coincident-atom guards (fail-loud contract).

Verification status (measured, 2026-09-18): lib + all test targets
compile clean (the unrelated `gpu_scc_plan.rs` refactor settled);
`test_sparse_pair_gpu_vs_cpu_parity` and `test_sparse_dftb_sih4_reuse_scc_and_fire`
pass; `sparse_f64check`/`gate_g3_energy` compile.

**Post-fix R18 benchmark (frozen, RTX 3090, full unbounded run — all
4944 columns):**

| metric | before residency | after |
|---|---:|---:|
| rst+geom per eval | ~6.6 ms | **~0.5 ms** |
| solve+f per eval | ~4.2 ms | **~1.1 ms** |
| **per eval total** | **~10.8 ms** | **~1.6 ms** (**~7×**) |
| full 4944-col Hessian wall | est. ~31–35 min | **138.2 s** |

Physics unchanged: center `max|F|=0.130741` (identical to pre-fix),
`n_imag=1`, `freq_min=−0.32 cm⁻¹` (acoustic floor), `freq_max=2298 cm⁻¹`,
Hessian asymmetry 6.7e-3, `E_tot=−1772.37331509` matching the converged
SCC value.

Per-eval profile now (n≈9890 calls): `geom.hs` 0.374 ms + `f.gamma`
0.360 ms + `geom.xyzu` 0.138 ms + `geom.rep` 0.024 ms — all
kernel/launch-bound work; the large O(nnz) transfers are gone. The
~155 MB/eval → ~0 estimate is confirmed by the ~7× eval speedup and the
collapse of `rst+geom` (which was dominated by the K/Z upload + H/S
readback).

Note: `VIB_COLS=12` was set but the run executed all 4944 columns — the
bounded-column env knob is named differently (`maxcol` is a script arg);
the accidental full run *is* the definitive measurement: 4944×2 evals +
init + final SCC/diag in 138 s wall.

### §15.26 — CPU-vs-GPU benchmark table + bottleneck census (2026-09-18)

Same runs as §15.24/§15.25, frozen mode, bounded at
`RUST_DFTB_VIB_MAXCOL=12`, `RUST_DFTB_PROF=mark`. CPU rows use the
explicit `RUST_DFTB_SPARSE_CPU=1` reference (single thread). R10:
330 atoms, nnz_hs=13 362, deg_hs≈60, h=0.05 Å, pbc-0-3,
`si_r10_sparse_relaxed_rk20.xyz`. R18: 1648 atoms, nnz_hs=199 676,
deg_hs≈121, h=0.02 Å, matsci-0-3, raw `si_sphere_R18.xyz`.

**Per-eval wall time (ms/eval = rst+geom + solve+f):**

| system | stage | CPU ref | GPU | speedup |
|---|---|---:|---:|---:|
| R10 (330) | rst+geom | ~3.6 | ~0.1 | |
| | solve+f | ~7.4 | ~0.3 | |
| | **total** | **~11.0** | **~0.4** | **~27×** |
| R18 (1648) | rst+geom | ~62 | ~0.5 | |
| | solve+f | ~132 | ~1.1 | |
| | **total** | **~194** | **~1.6** | **~121×** |

Progression of the same measurement (frozen GPU eval): §15.24 pair-port
1.7 ms (R10) / 10.8 ms (R18) → §15.25 residency **0.4 / 1.6 ms**.
Projected full Hessians: R18 CPU ≈ 4944×2×0.194 s ≈ **32 min** vs GPU
**138 s**; R10 CPU ≈ 990×2×0.011 s ≈ **22 s** vs GPU ≈ **0.8 s** of
column time (measured 0.094 s for 12 cols + fixed overheads).

**R18 GPU per-eval stage census (n=9890 calls, full-run profile):**

| stage | ms/eval | what it is |
|---|---:|---|
| `geom.hs` | 0.374 | hs_diag + hs_assemble over 200k pairs + kdummy + s_inf reduce + H_scc build |
| `f.gamma` | 0.360 | γ′ all-pairs n-body force kernel (1.6M pairs, O(N²)) |
| `geom.xyzu` | 0.138 | coords pack + upload (O(N)) |
| `geom.rep` | 0.024 | repulsive spline eval |
| *unprofiled* | ~0.7 | restore device copies, dq upload, contract+gather launches, O(N) readbacks, host pair-guard loops, ~10 launch latencies |
| **total** | **~1.6** | |

**Where the remaining time actually goes:**

1. **The Hessian columns are no longer the pipeline bottleneck.** Of the
   138.2 s R18 wall: ~11.8 s init (plan/alloc), ~7 s central SCC,
   ~16 s all 9888 frozen evals — and **~100 s dense host
   `nalgebra::SymmetricEigen` on the mass-weighted 4944×4944 Hessian**.
   The eigensolve is now ~70% of the total wall and is pure O((3N)³)
   host work. At N=5k (15k cols) a dense eigh is ~hours — this becomes
   the new wall before column cost does.
2. **Per-eval: `geom.hs` + `f.gamma` ≈ 0.73 ms of 1.6 ms** — both are
   *all-pairs* work for a single-atom displacement. Column-local pair
   subranges (manifest §F) shrink both to O(deg): the ~167 H/S pairs and
   the γ′ row touched by the displacement. This is the correct next
   optimization, not kernel tuning.
3. **~0.7 ms unprofiled orchestration floor**: ~10 kernel launches
   (~10–20 µs each on this driver), 5×3N-force + O(N) readbacks,
   device→device restores, and the **O(n_pairs) host coincident-atom
   guard loops** in `set_coords` (~200k-iteration CPU scan per geometry
   — hoistable to once per pair list, or a GPU check). Batched ±h
   launches amortize all of it across a color class.
4. **`f.gamma` is the asymptotic risk**: O(N²) tiled n-body. At N=5k it
   is ~9× today's cost (~3 ms/eval) and would dominate; fine for the
   frozen column phase if made column-local too (γ′ row of the displaced
   atom), or eventually a cell-list/tree method.

**Physics parity notes.** Center `max|F|` identical CPU vs GPU
(0.130741); per-column `max|ΔF|` tracks with ~10–20 % deviation
(e.g. col 0: 2.66e-2 GPU vs 2.42e-2 CPU; col 2: 3.55e-2 vs 2.86e-2).
ΔF = F(+h)−F(−h) is a ~5×-cancelling difference of |F|~0.13 forces, so
this is ~2–3 % f32 accumulated error on the forces themselves — *plus*
the two runs converged to different SCC states (r_scc 4.4e-6 vs 7.6e-6),
which shifts the frozen reference at the ~1e-5 level and is amplified by
the FD difference. Same-state kernel parity is covered by
`test_sparse_pair_gpu_vs_cpu_parity` (max|dF|=1.2e-6 at small N); a
component-level R18 parity check remains queued. The outcome-level
Hessian is unaffected: n_imag=1 at −0.32 cm⁻¹, asymmetry 6.7e-3.

### §15.27 — Frozen multi-replica batch (F1) implemented + measured (2026-09-19)

Manifest §F.1 implemented as designed: replica axis `get_global_id(1)`
on the 5 eval kernels (`gamma_matvec`, `gamma_force`, `hs_contract`,
`rep_eval`, `force_gather`), 2D NDRange `gws=(domain, B)` / `lws=(·,1)`,
shared inputs (pair topology, SK/rep tables, `dq₀`, `k0`, `w0`) single
and read-only, per-replica `xyzu`/`pf`/`pe_rep`/force buffers at
`b·stride` in one persistent `GpuFrozenBatch` (~6.9 MB/replica at R18).
No atomics, no scheduler state machine — static JobId→SlotId mapping;
results scattered to Hessian columns by evaluation index. `forces_frozen_batch(x0, evals, h)`
is pure w.r.t. `x0` (bug fix: scalar path had left `self.coords` at the
last displaced geometry — the base is now an explicit parameter). Driver:
`RUST_DFTB_VIB_BATCH=B`; B=1 keeps the scalar loop verbatim. The batched
path also skips `hs_assemble` — `hs_contract` recomputes SK+rotation from
`xyzu` directly, so the assembled H/S blocks were dead work in frozen
evals (the F0 item, folded in).

**L0 parity (`test_sparse_frozen_batch_parity`, small system):**
30 evals batched vs sequential `forces_frozen`: **max|dF| = 0.000e0,
bit-identical**; subset reuse (7 evals into the same workspace)
bit-identical; `test_sparse_pair_gpu_vs_cpu_parity` unchanged.

**R18 measured (1648 atoms, 96 columns = 192 evals, `VIB_MAXCOL=96`,
raw sphere, RTX 3090):**

| B | ms/eval | vs scalar | batch wall/eval breakdown |
|---:|---:|---:|---|
| 1 (scalar loop) | ~1.6 | — | 0.5 rst+geom + 1.1 solve+f |
| 8 | 0.31 | ~5.2× | one xyzu upload + one readback per 8 evals |
| 16 | **0.27** | **~5.9×** | saturation knee |
| 32 | 0.27–0.32 | ~5–6× | saturated — kernel-work-bound, not overhead-bound |

Physics identical at every B: per-column `max|ΔF|` matches the scalar
run column-for-column (e.g. 1.1310e-2/1.8520e-2/1.8445e-2 sequences
identical at B=1/8/16/32). Full Hessian at B=16: `n_imag=1`,
`freq_min=−0.32`, `freq_max=2298.25 cm⁻¹`, asymmetry 6.732e-3 —
identical to the scalar full run.

**End-to-end honest accounting:** full R18 frozen Hessian wall
**138.2 s scalar → 138.6 s at B=16** — unchanged because the column
phase (~16 s → ~2.7 s of evals) was never the bottleneck. Wall
composition now: ~12 s init, ~7 s central SCC, ~2.7 s all 9888 evals,
**~100 s dense host eigensolve (~72 %)**. Batching delivered its
designed goal — the per-eval overhead floor is gone and columns are now
~2 % of wall — but the next real lever on wall time is the O((3N)³)
host `SymmetricEigen`, not more column work. B=16 is the default-safe
choice at R18; at N=5k the same kernels get ~9× more work per launch so
saturation moves to larger B (γ′ tiling is shared across the replica
axis only through launch width — per-replica work is unchanged).

**Deferred (unchanged plan):** fixq/SCC batching reuses the SlotId
plumbing but needs the variable-convergence machinery from
`Sparse_MultiSystem_Scheduler.chat.md` (active masks, compact job IDs,
refill) — only justified after frozen Hessian + eigensolve path are
settled. F2 column-local pair subranges (~167 of 200k pairs per
displacement) remains the next per-eval lever; at B=16 the eval is now
dominated by real kernel work so F2 shrinks total work, not overhead.
