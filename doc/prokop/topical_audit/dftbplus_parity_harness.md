---
type: TopicalAudit
title: DFTB+ Fortran Parity Harness
tags: [topic, parity, fortran, reference, python]
---

# DFTB+ Fortran Parity Harness

## Summary

A Python harness that runs the upstream Fortran DFTB+ binary on the same geometry
and SK set as the Rust implementation, then parses the output (`detailed.out`,
`band.out`) into TSV files for numerical comparison and plotting. Enables
apples-to-apples parity validation of Rust dense and Rust sparse results against
the Fortran reference.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Python | `scripts/run_dftbplus_ref.py` | active | Generates HSD, runs DFTB+, parses `detailed.out` + `band.out` → TSV. |
| Python | `scripts/compare_rust_vs_dftbplus.py` | active | Numerical parity report: charges, eigenvalues, gaps. |
| Python | `scripts/plot_charges_homo_lumo.py` | active | Spatial charge map + HOMO-LUMO levels (Rust vs DFTB+). |
| Fortran | `_build/app/dftb+/dftb+` | reference | The built DFTB+ binary. |
| Fortran | `src/dftbp/` | reference | Source code (ground truth). |

## Parity Status

Validated on benzene (6 C), coronene (24 C), circumcoronene (54 C) with the
mio-1-1 SK set, SCC enabled, `SCCTolerance=1e-8`.

| System | dense max\|Δq\| vs DFTB+ (e) | dense max\|Δε\| vs DFTB+ (Ha) | gap Δ (Ha) |
|--------|------------------------------|-------------------------------|------------|
| Benzene | 0 | 2.06e-6 | 2.27e-6 |
| Coronene | 4.2e-16 | 2.04e-6 | 3.26e-7 |
| Circumcoronene | 3.9e-16 | 2.06e-6 | 3.00e-7 |

- **Charges**: machine-precision match (dense).
- **Eigenvalues**: ~2e-6 Ha difference from Fermi smearing temperature mismatch
  (Rust uses `T=0`, DFTB+ uses a small nonzero electronic temperature).

## Output format notes

- `detailed.out`: Fermi level and total energy in **Hartree**; Mulliken charges
  in **electrons** (sign convention: `deltaQ = q0 - q_elec`, positive = electron
  deficit).
- `band.out`: eigenvalues in **eV** (not Hartree!), occupations as 0.0/2.0.
  Conversion: `eig_Ha = eig_eV / 27.211386`.
- `dftb_in.hsd`: `WriteDetailedOut = Yes` is needed for charges. `WriteBandOut`
  is not accepted by the current parser version; `band.out` is written by default.

## Open Issues

- **Fermi temperature mismatch** — Rust uses `T=0` filling; DFTB+ uses a small
  nonzero temperature. This causes ~2e-6 Ha eigenvalue differences. Could align
  by matching the temperature in both codes.
- **`band.out` overwritten per run** — the harness copies `ref_charges.tsv` and
  `ref_eigenvalues.tsv` to per-system names (`<sys>_ref_*.tsv`) to avoid
  clobbering.
- **HSD parser version** — the built DFTB+ uses parser version 15; some older
  HSD keywords (e.g. `WriteBandOut`) are rejected. The harness uses only
  version-15-compatible options.
- **No SK set auto-detection** — the user must pass `--sk-dir`. The default
  points to `mio-1-1`.

## Related

- `/doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md` — session report.
- `/scripts/run_dftbplus_ref.py` — harness.
- `/debug/graphene_sparse/` — generated TSVs and plots.
