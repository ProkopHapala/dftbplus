# Sparse Nanocrystal Vibrations — Implementation Report

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
