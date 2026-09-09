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
