---
type: TopicalAudit
title: Sparse Nanocrystal Vibrations (Si/H)
tags: [topic, sparse, bsr4, vibrations, hessian, forces, gpu, nanocrystal, si, h]
---

# Sparse Nanocrystal Vibrations (Si/H)

## Summary

Sparse GPU DFTB implementation for vibrational calculations on large silicon
and diamond nanocrystals (300–1000 atoms). Uses BSR4 sparse matrices (4×4 atom
blocks), f32 GPU arithmetic, TC2 density-matrix purification, analytic
Hellmann-Feynman forces, and finite-difference Hessians. **Everything
GPU-resident** — no CPU bridges, no hybrid paths.

**Manifest (source of truth):**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md`
**Master task:**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.tasks.md`
**Implementation report:**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md`
**GPT-5.6 review:**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.chat.md` line 2064+

## Status (revised after GPT-5.6 review, 2026-09-09)

GPT-5.6 reviewed commit `b269ab6` and found 22 issues (7 critical blockers).
Several "completed" items are not actually wired into the production numerical
path. See manifest §13 for the full checklist and `tasks.md` for the phased
master task breakdown.

## Implementations

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| Sparse firewall | `methods/sparse/bsr4.rs`, `gpu_sparse.rs` | [*] P0 PASS | `sparse_firewall` feature |
| CPU/GPU B-spline | `methods/dftb/spline_resample.rs`, `dftb_hamiltonian.cl` | [~] P1 | Evaluator built, **not canonical** in production force path |
| Analytic SK derivatives | `methods/dftb/forces.rs`, `rotation.rs` | [~] P2 | Angular correct; radial V,V' still from Hermite, not C² |
| Sparse D=2K, W=2KHK | `methods/sparse/sparse_forces.rs` | [~] P3 | Formula correct; allocation-heavy, attached to dense SCC |
| Symbolic SpGEMM plan | `methods/sparse/bsr4.rs`, `gpu_sparse.rs`, `sparse_bsr4_purification.cl` | [~] P4 | Plan built, **not integrated** into TC2/NS/K0/W |
| BSR4 padded Si/H basis | `tests/sih_padded_basis.rs` | [*] Gate D PASS | H dummy orbitals: S_dd=1, H_dd=E_dummy=2.0 |
| Locality sweep R_K×R_Z | `tests/locality_sweep.rs` | [ ] Gate C | False positive — 5-atom toy, +2I doesn't create gap |
| Determinism + Hessian h | `tests/gate_e_determinism.rs` | [ ] Gate E | False positive — FD-of-energy forces, symmetrized Hessian |
| Geometry optimization | `tests/gate_f_geom_opt.rs` | [ ] Gate F | Broken — FD-of-energy forces, do not use |
| Hessian parity | — | NOT STARTED | Gate G |
| Vibrational spectra | — | NOT STARTED | Gate H |

## Gate Status

| Gate | Description | Status |
|------|-------------|--------|
| B | Analytic force vs finite difference | [~] PASS (9 tests) — but radial V,V' from Hermite, not C² |
| C | Locality sweep R_K × R_Z | [ ] false positive — redo on real Si/H (task D1) |
| D | Nonsingular padded Si/H basis | [*] PASS |
| E | Determinism, arithmetic sensitivity, Hessian h plateau | [ ] false positive — redo with analytic forces (task D2) |
| F | Geometry optimization | [ ] broken — needs GPU sparse force kernel (Phase C) |
| G | Same-geometry Hessian parity | NOT STARTED |
| H | Spectra at each method's own minimum | NOT STARTED |
| I | Scaling and whole-program profile | NOT STARTED |

## Blockers (GPT-5.6 review, 7 critical issues)

1. **#1 — C² spline not canonical in production force path.** `SkTableSp`
   still uses `EqGridTable` (C¹ Hermite + numerical FD tail derivative).
2. **#2 — No sparse SCC solver.** Sparse purification attached to dense
   `SccResult`. K from dense charges, not self-consistent sparse.
3. **#3 — TC2 does ~4 SpGEMMs/iter instead of 2.** Convergence check
   recomputes KS+KSK after update. Trace read to host every iteration.
4. **#4 — P4 plans not integrated.** Plan kernel exists but TC2/build_dw_sparse
   still use intersection kernel.
5. **#8 — `build_dw_sparse()` allocation-heavy.** No persistent workspace,
   separate D allocation, scale kernels.
6. **#10 — Gate C false positive.** 5-atom toy system, +2I doesn't create gap.
7. **#13 — Gate E false positive.** FD-of-energy forces, symmetrized Hessian
   tautology, h_ref in candidate list.

## Current plan (Phases A–E, see `tasks.md`)

- **Phase A** — Foundation: canonical C² spline (A1), TC2 fix (A2), plan
  integration (A3)
- **Phase B** — Sparse SCC solver, GPU-resident, self-consistent — **current
  focus** (B1: workspace, B2: H0/S, B3: gamma, B4: Hscc, B5: K0+bounds, B6:
  SCC loop, B7: Mulliken)
- **Phase C** — GPU-resident sparse analytic forces (pair-force + atom-gather
  kernel, after dense agent finishes)
- **Phase D** — Correctness gates: redo C (D1), E (D2), then F (D3), G (D4),
  H (D5)
- **Phase E** — Performance: real counters, cell-list, degree buckets, packed
  plans, lane benchmark, symmetrization policy, scaling, production

## Cross-References

- **Sparse TC2 purification:** `topical_audit/sparse_tc2_purification.md`
- **SK interpolation:** `topical_audit/sk_interpolation.md`
- **GPU SCC pipeline:** `topical_audit/gpu_scc_pipeline.md`
- **SCC Mulliken charges:** `topical_audit/scc_mulliken_charges.md`
- **Roadmap:** `DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` §7.6
- **Manifest:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md` (§13 checklist)
- **Master task:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.tasks.md`
- **Report:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md`
- **GPT-5.6 review:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.chat.md` line 2064+
