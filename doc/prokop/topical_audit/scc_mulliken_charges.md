---
type: TopicalAudit
title: SCC Mulliken Charges
tags: [topic, scc, mulliken, charges, cross-language]
---

# SCC Mulliken Charges

## Summary

After SCC convergence, atomic Mulliken charges are computed from the density
matrix `D` (dense) or density kernel `K` (sparse). These charges drive the SCC
self-consistency and are the primary observable for parity validation against
DFTB+.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Rust (dense) | `rust_dftb/src/methods/dftb/hamiltonian.rs` | active | `SccResult.charges` = Mulliken populations from `D·S` diagonal. `q0` stored separately. |
| Rust (sparse) | `rust_dftb/src/methods/sparse/gpu_sparse.rs` | active | Mulliken from diagonal blocks of `K·S` (BSR4). `SparseResult.mulliken`. |
| Rust (xtb) | `rust_dftb/src/methods/xtb/mulliken.rs` | active | xTB Mulliken/CM5 charges. |
| Rust (engine) | `rust_dftb/src/bin/dftb_engine.rs` | active | Rhai: `get_charges`, `save_charges`, `get_sparse_charges`, `save_sparse_charges`. |
| Rust (GPU) | `rust_dftb/src/qmqm/gpu_scc.rs` | active | `GpuSccResult.charges` — Mulliken from device-resident D·S diagonal, batched. Parity <6e-6 vs CPU on formic dimer. |
| Python | `scripts/compare_rust_vs_dftbplus.py` | active | Parity report: converts Rust populations → charges for comparison. |
| Python | `scripts/plot_charges_homo_lumo.py` | active | Spatial charge map (Rust dense vs sparse vs DFTB+). |
| Fortran (ref) | `src/dftbp/scc/` | reference | Upstream SCC + Mulliken analysis. `detailed.out` reports `deltaQ = q0 - q_elec`. |

## Sign convention (critical)

- **Rust**: `SccResult.charges` = Mulliken **populations** (`q_electronic`).
  Charge transfer: `delta_q = q - q0` (positive = electron gain).
- **DFTB+**: `detailed.out` reports `deltaQ = q0 - q_elec` (positive = electron
  deficit).
- **Conversion**: `charge_DFTB+ = q0 - population_Rust`.
  Example: Rust C population = 4.095 → DFTB+ charge = 4.0 - 4.095 = -0.095.

This is documented in `rust_dftb/src/methods/dftb/forces.rs:16-18`.

## Parity Status

| System | dense vs DFTB+ max\|Δq\| (e) | sparse TC2 vs DFTB+ max\|Δq\| (e) |
|--------|------------------------------|-------------------------------|
| Benzene | 0 | 1.5e-5 |
| Coronene | 4.2e-16 | 9.0e-6 |
| Circumcoronene | 3.9e-16 | 6.5e-5 |

Dense charges match DFTB+ to machine precision. Sparse TC2 charges match to
~1e-5–1e-4 e (consistent with TC2 tolerance).

## Open Issues

- **BSR4 rejects H atoms** — 4 orbitals/atom required. Pure-C systems only for
  sparse charges. Variable block size or dense fallback needed.
- **Population vs charge convention** — easy to confuse. The plotting/comparison
  scripts convert; direct Rhai output shows populations.

## Related

- `/doc/prokop/topical_audit/sparse_tc2_purification.md` — sparse charge route.
- `/doc/prokop/topical_audit/gpu_scc_pipeline.md` — GPU device-resident SCC.
- `/doc/prokop/topical_audit/dftbplus_parity_harness.md` — parity validation.
- `/doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md` — session report.
