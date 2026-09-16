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

## Status (revised 2026-09-16 — read `f32_floor_sparse.md` + manifest §0)

`SparseDftb` is now the persistent production loop (`sparse_dftb.rs`):
`set_coords`/`scc`/`forces`/`fire_step`/`relax`/`sparse_vibrations` all run
through one engine with frozen topology and prebuilt SpGEMM plans. End-to-end
FD Hessians were produced (si10h16, cube_Si65, R10). The per-eval mode ladder
is now a measured hierarchy (R10, h=0.02 Å, Hessian-column error vs cold
fixq): **clamped frozen-orbital** (`VIB_FROZEN=1` — explicit `K₀,W₀,q₀`
snapshot, ZERO device products) **5.5 ms @ 6.3%** → `VIB_LITE` 1-Newton +
DMM2 **64 ms @ 3.1%** → 1-Newton + DMM4 **105 ms @ 1.0%** → cold fixq ~345 ms.
The gated/certified variants (`VIB_DMUPD=1`) cost ~40% more in per-eval
residual checks — validation runs only. See
`reports/2026-09-16_sparse_dmm_warm_density_hessian.md` and
`topical_audit/hessian_eval_bottleneck.md`. Cube_Si65 absolute frequencies
remain ~5–15% stiff vs DFTB+ (energy-parity gap under investigation).

Device NS (N4) status per `f32_floor_sparse.md`; the force-validated gates
(R_H stationarity, dummy-lane occupation) fail loudly rather than falling back.

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
| Vibrational spectra | `sparse_vibrations` rhai → FD Hessian | [~] produced | si10h16/cube65/R10; ~5–15% stiff vs DFTB+ — not benchmark-grade |
| Production `SparseDftb` lifetime | `sparse_dftb.rs` | [*] production | Persistent engine; central-state snapshot per column |
| FD-Hessian warm update | `sparse_system.rs::dmm_descend` / `linear_response`, `forces_frozen` | [*] validated | tier ladder: clamped 5.5 ms/6.3%, lite 64 ms/3.1%, 105 ms/1.0% |

## Gate Status

| Gate | Description | Status |
|------|-------------|--------|
| B | Analytic force vs finite difference | [~] G3.3/G3.4 — analytic F vs dense 3.7e-6; energy-FD rel 4e-3 at h=1e-3 is floor |
| C | Locality sweep R_K × R_Z | [ ] false positive — redo on real Si/H (task D1) |
| D | Nonsingular padded Si/H basis | [*] PASS |
| E | Determinism, arithmetic sensitivity, Hessian h plateau | [ ] false positive — redo with analytic forces (task D2) |
| F | Geometry optimization | [~] investigating — FIRE 1.477 Å; not done |
| G | Same-geometry Hessian parity | [~] investigating — 0.11% vs dense; FD columns validated 0.3% ΔF vs cold fixq |
| H | Spectra at each method's own minimum | [~] produced — cube65 ~5–15% stiff vs DFTB+ reference |
| I | Scaling and whole-program profile | [~] bottleneck measured — `hessian_eval_bottleneck.md`; batching not yet done |

## Remaining split (do not treat GPT-5.6 2026-09-09 list as current)

Stale: Hermite-as-production, “no sparse SCC”, Gate F 0.93 Å as the only F.
Current: `f32_floor_sparse.md` ID table. Still true from that review: Gate C/E
false positives; device NS N4; no persistent production owner; D/W not GPU-resident.

## Current plan (Phases A–E, see `tasks.md`)

- **Phase A** — Foundation: canonical C² spline (A1), TC2 fix (A2), plan
  integration (A3) — done
- **Phase B** — Sparse SCC solver, GPU-resident, self-consistent — done
  (`SparseDftb` production owner)
- **Phase C** — GPU-resident sparse analytic forces — done (sparse D/W
  contraction + pair-force kernel on device)
- **Phase D** — Correctness gates: redo C (D1), E (D2), then F (D3), G (D4),
  H (D5) — in progress (see gate table)
- **Phase E** — Performance: real counters, cell-list, degree buckets, packed
  plans, lane benchmark, symmetrization policy, scaling, production —
  **current focus**: stripped warm tiers landed (Phase E.1 — clamped
  5.5 ms / lite 64–105 ms per eval); next is batch-parallel ±h columns —
  the remaining order-of-magnitude lever

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
- **GPT-5.6 review:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.chat.md` line 2064+ (DMM discussion ~10605+, Tier-1 rebuttal ~11600+)
- **FD-Hessian bottleneck:** `topical_audit/hessian_eval_bottleneck.md`
- **DMM warm-update report:** `reports/2026-09-16_sparse_dmm_warm_density_hessian.md`
