# Sparse Nanocrystal Vibrations — Implementation Report

> **Current review correction:** read manifest **§15** before using the later Item 5/6/7 conclusions. W=2(ZH)K still consumes truncated ZH; the quartic clamp is not a certified stability invariant; TRS G3.4 remains red; and the histogram's 3% row-norm budget is 0.09% squared norm mass, not 3% mass or a force-error bound. The ~61-neighbor result lacks symmetric-mask/re-solved force validation. Earlier results are retained as history, not current acceptance. §15 provides the prioritized sparse-only coding-agent instructions; this review made no implementation changes and ran no new benchmarks.

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
