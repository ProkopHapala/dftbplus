---
type: TopicalAudit
title: Sparse Nanocrystal Vibrations (Si/H)
tags: [topic, sparse, bsr4, vibrations, hessian, forces, gpu, nanocrystal, si, h]
---

# Sparse Nanocrystal Vibrations (Si/H)

## Summary

Sparse GPU DFTB for Si/H nanocrystal vibrations (300–1000 atoms): BSR4,
f32 TC2, analytic forces, FD Hessians. **Physics gates G3/F/G run on NVIDIA
with a CPU H0/S + host-NS + CPU force bridge.** Production owner `SparseDftb`
is coded; drive it with `dftb_engine` + `scripts/test_sparse_dftb_sih4.rhai`
(userguide `sparse_dftb.md`). Floor vs bugs: `f32_floor_sparse.md`.

**Manifest (source of truth):**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md` **§0**
**Bugs vs floor vs pipeline:** `topical_audit/f32_floor_sparse.md`
**Interpolator fitter:** `topical_audit/sk_interpolation.md`
**Master task:**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.tasks.md`
**Implementation report:**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md`
**GPT-5.6 review:**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.chat.md` line 2064+

## Status (revised 2026-09-10 — read `f32_floor_sparse.md` + manifest §0)

Physics gates G3/F/G exist and were run on NVIDIA. They are **investigating**,
not done. Device NS (N4) stays **red**. There is **no** persistent `SparseDftb`
production loop — `cargo test` is not the product.

GPT-5.6 (commit `b269ab6`) found 22 issues; several are now stale (B-spline
eval is production; sparse SCC exists in `scc.rs`). The remaining split is
bugs / fitter / floor / missing pipeline — do not mix them.

## Implementations

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| Sparse firewall | `methods/sparse/bsr4.rs`, `gpu_sparse.rs` | [*] P0 PASS | `sparse_firewall` feature |
| CPU/GPU B-spline eval | `interpolation.rs`, force V' | [*] eval | Extra-control **fitter** still stopgap zeros |
| Sparse D=2K, W=2KHK | `sparse_forces.rs` + CPU `compute_forces_from_dw` | [~] G3.3 | Allocation-heavy; not GPU-resident |
| Sparse SCC (real mix loop) | `scc.rs::run_sparse_scc` + `SparseDftb::scc` | [~] G3.2 / production owner | G3 still allocating. `SparseDftb` is the persistent loop |
| Geometry optimization | `tests/gate_f_geom_opt.rs` | [~] | FIRE 1.477 Å on NVIDIA; not marked done |
| Hessian parity | `tests/gate_g_hessian.rs` | [~] | Unsymmetrized FD-of-F; 0.11% vs dense; not marked done |
| Vibrational spectra | — | NOT STARTED | Gate H |
| Production `SparseDftb` lifetime | `sparse_dftb.rs` | **coded, not confirmed** | NVIDIA SiH4 reuse+FIRE; see `f32_floor_sparse.md` |

## Gate Status

| Gate | Description | Status |
|------|-------------|--------|
| B | Analytic force vs finite difference | [~] G3.3/G3.4 — analytic F vs dense 3.7e-6; energy-FD rel 4e-3 at h=1e-3 is floor |
| C | Locality sweep R_K × R_Z | [ ] false positive — redo on real Si/H (task D1) |
| D | Nonsingular padded Si/H basis | [*] PASS |
| E | Determinism, arithmetic sensitivity, Hessian h plateau | [ ] false positive — redo with analytic forces (task D2) |
| F | Geometry optimization | [~] investigating — FIRE 1.477 Å; not done |
| G | Same-geometry Hessian parity | [~] investigating — 0.11% vs dense; not done |
| H | Spectra at each method's own minimum | NOT STARTED |
| I | Scaling and whole-program profile | NOT STARTED — illegal until `SparseDftb` exists |

## Remaining split (do not treat GPT-5.6 2026-09-09 list as current)

Stale: Hermite-as-production, “no sparse SCC”, Gate F 0.93 Å as the only F.
Current: `f32_floor_sparse.md` ID table. Still true from that review: Gate C/E
false positives; device NS N4; no persistent production owner; D/W not GPU-resident.

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
- **SK interpolation:** `topical_audit/sk_interpolation.md` (extra-control fitter)
- **Sparse floor vs bugs vs pipeline:** `topical_audit/f32_floor_sparse.md`
- **GPU SCC pipeline:** `topical_audit/gpu_scc_pipeline.md`
- **SCC Mulliken charges:** `topical_audit/scc_mulliken_charges.md`
- **Roadmap:** `DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` §7.6
- **Manifest:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md` (§13 checklist)
- **Master task:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.tasks.md`
- **Report:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md`
- **GPT-5.6 review:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.chat.md` line 2064+
