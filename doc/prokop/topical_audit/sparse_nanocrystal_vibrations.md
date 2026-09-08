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
Hellmann-Feynman forces, and finite-difference Hessians.

**Manifest (source of truth):**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md`
**Implementation report:**
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md`

## Implementations

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| BSR4 padded Si/H basis | `tests/sih_padded_basis.rs` | Gate D PASS | H dummy orbitals: S_dd=1, H_dd=E_dummy=2.0 |
| Sparse D=2K, W=2KHK | `methods/sparse/sparse_forces.rs` | P3 PASS | Masked SpGEMM, device-resident |
| Symbolic SpGEMM plan | `methods/sparse/bsr4.rs`, `gpu_sparse.rs`, `sparse_bsr4_purification.cl` | P4 PASS | Precomputed intersection, parity exact |
| Locality sweep R_K×R_Z | `tests/locality_sweep.rs` | Gate C PASS | Plateau R_K=7, R_Z=7 |
| Determinism + Hessian h | `tests/gate_e_determinism.rs` | Gate E PASS | Plateau h=0.02 Å |
| Analytic SK derivatives | `methods/dftb/forces.rs`, `rotation.rs` | P2 PASS | Production force path |
| CPU/GPU B-spline | `methods/dftb/spline_resample.rs`, `dftb_hamiltonian.cl` | P1 PASS | C² cubic, parity verified |
| Sparse firewall | `methods/sparse/bsr4.rs`, `gpu_sparse.rs` | P0 PASS | `sparse_firewall` feature |
| Geometry optimization | `tests/gate_f_geom_opt.rs` | **BLOCKED** | Needs sparse analytic force bridge |
| Hessian parity | — | **NOT STARTED** | Gate G, depends on Gate F |
| Vibrational spectra | — | **NOT STARTED** | Gate H, depends on Gate G |

## Gate Status

| Gate | Description | Status |
|------|-------------|--------|
| B | Analytic force vs finite difference | PASS (9 tests) |
| C | Locality sweep R_K × R_Z | PASS |
| D | Nonsingular padded Si/H basis | PASS |
| E | Determinism, arithmetic sensitivity, Hessian h plateau | PASS (E-B force path flawed) |
| F | Geometry optimization | **BLOCKED** — needs sparse analytic force |
| G | Same-geometry Hessian parity | NOT STARTED |
| H | Spectra at each method's own minimum | NOT STARTED |
| I | Scaling and whole-program profile | NOT STARTED |

## Blocker: Sparse Analytic Force Bridge

Gate F is blocked because the sparse force path uses finite differences of the
sparse energy (30 pipeline runs per force evaluation), which is:
1. Horrendously slow (~5 min for 5 atoms).
2. Against the performance policy (rebuilding buffers in hot loop).
3. Scientifically wrong — analytic force infrastructure already exists.

The correct path:
1. One sparse pipeline run per geometry → K → `build_dw_sparse()` → D, W.
2. Analytic force contraction: `F = 2·ANG2BOHR·Σ(D·dH - W·dS)` per pair, using
   `build_pair_block_with_derivs()` (already implemented in P2).
3. The missing piece: bridge padded BSR4 D/W (4 orb/atom) to variable-orbital
   `non_scc_electronic_force` (Si=4, H=1). Dummy rows/columns in D/W are zero
   (verified Gate D), so extraction is straightforward but not yet implemented.

## Cross-References

- **Sparse TC2 purification:** `topical_audit/sparse_tc2_purification.md`
- **SK interpolation:** `topical_audit/sk_interpolation.md`
- **GPU SCC pipeline:** `topical_audit/gpu_scc_pipeline.md`
- **SCC Mulliken charges:** `topical_audit/scc_mulliken_charges.md`
- **Roadmap:** `DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` §7.5
- **Manifest:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md`
- **Report:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md`
