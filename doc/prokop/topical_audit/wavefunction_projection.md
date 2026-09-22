---
type: TopicalAudit
title: Wavefunction Projection onto Real-Space Grid
tags: [topic, wavefunction, grid, opencl, sto, visualization, cross-language]
---

# Wavefunction Projection onto Real-Space Grid

## Summary

Projects LCAO coefficients onto real-space points with Slater-type orbitals.
Two kernels must not be mixed. `Grid.cl` is Γ-only and packs Fireball
`[px, py, pz, s]`. `DFTBplusGrid.cl` `project_bloch_points` takes complex
`C(k)` from DFTBcore in DFTB+ order (`s`, `py`, `pz`, `px`), repeats the home
cell over an explicit image list, and applies `exp(−i 2π k·n)` itself.
A density map is `Σ w |ψ|²` (Tersoff–Hamann), not a sum of the complex waves
and not a density-matrix contraction, so the Bloch phase does not appear in `ρ`.
How to run it: `doc/prokop/userguide/bloch_slice.md`.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Python+OpenCL | `pyBall/OCL/Grid.py::GridProjector` | active | GPU projection via OpenCL. `project_orbital_dense()` for dense MO coefficient vectors. `evaluate_mos_on_points()` for arbitrary point sets. |
| Python+OpenCL | `pyBall/OCL/Grid.py::setup_gridprojector_from_dftb` | active | Configures GridProjector from DFTB+ data (coords, species, STO basis). |
| Python | `pyBall/OCL/DFTBplusParser.py::parse_wfc_hsd` | active | Parses `wfc.mio-1-1.hsd` STO basis files (Bohr units). |
| Python | `pyBall/OCL/DFTBplusParser.py::evec_to_kernel_coeffs` | active | Converts eigenvector row → (natoms, 4) kernel coeffs [px,py,pz,s]. |
| OpenCL | `pyBall/OCL/cl/Grid.cl` | active | GPU kernels for STO evaluation and projection. |
| Python | `scripts/plot_wavefunctions.py` | active | End-to-end: reads Rust eigenvector TSV → GridProjector → 2D contour plot. |
| Python | `tests/grid/test_waveplot_dftbcore.py` | reference | DFTBcore library → eigenvectors → GridProjector. The reference workflow. |
| Fortran | `src/dftbp/waveplot/` | reference | Upstream waveplot (Fortran) — writes cube files. Phase convention matched by `project_bloch_points` (`exp(−ikr)` after the Fortran `dot_product`). |
| OpenCL | `pyBall/OCL/cl/DFTBplusGrid.cl` `project_bloch_points` | active | k-resolved. One work-item per point. s+p only (`norb ≤ 4` per atom). |
| Python | `pyBall/OCL/DFTBplusGridProjector.py` `project_bloch_points` | active | Host side. `C` is `(nstate, norb)` complex, raw home-cell eigenvectors. |
| Python | `tests/grid/test_graphene_bloch2d.py` | active | 2-atom graphene, 3ob-3-1. CPU spline parity, K Bloch check, band-window maps. |
| Python | `tests/grid/test_ribbon_bloch.py` | active | SPAMMM vacuum ribbons, mio-1-1. Near-EF |\ψ|² and one-state hue plots. Orthogonal cells only. |

## Data flow

```
Rust SCC → SccResult.eigenvectors (norb × norb, columns = MOs)
  → Rhai save_eigenvectors(name, path) → TSV file
  → Python plot_wavefunctions.py
    → parse_eigenvectors_tsv() → species, coords, evecs, eigs
    → load_wfc_basis(wfc.mio-1-1.hsd) → STO basis (Angstrom)
    → setup_gridprojector_from_dftb() → GridProjector (OpenCL)
    → evaluate_mos_on_points() → ψ(r) on 2D grid
    → matplotlib contour plot → PNG
```

## Orbital ordering (critical)

DFTB+ and the Rust solver store real spherical harmonics as:

- l=0 (s): `[s]`
- l=1 (p): `[py (m=-1), pz (m=0), px (m=+1)]`

`project_bloch_points` consumes that order directly. Hydrogen is `s` only.

`Grid.cl` does **not**. `evec_to_kernel_coeffs()` (`DFTBplusParser.py`)
repacks a DFTB+ eigenvector into Fireball `[px, py, pz, s]` for that older
kernel. Sending the repacked vector to `project_bloch_points`, or the raw
DFTB+ vector to `Grid.cl`, swaps every p lobe.

## Parity Status

- **Benzene** (6 C, 24 orbitals): HOMO (MO12) and LUMO (MO13) visualized.
  The HOMO shows the expected doubly-degenerate π orbital pattern.
- **Coronene** (24 C, 96 orbitals): HOMO/LUMO visualized.
- **Circumcoronene** (54 C, 216 orbitals): HOMO/LUMO visualized.

### 3-way eigenvalue parity (dense vs sparse-Chebyshev+Ritz vs Fortran)

| System | ΔHOMO dense-Fortran | ΔLUMO dense-Fortran | ΔHOMO sparse-dense | ΔLUMO sparse-dense |
|--------|---------------------|---------------------|--------------------|--------------------|
| Benzene | -8.57e-07 | 1.41e-06 | 0.00e+00 | 0.00e+00 |
| Coronene | 1.26e-06 | 9.32e-07 | 0.00e+00 | 0.00e+00 |
| Circumcoronene | 1.46e-06 | 1.16e-06 | -8.47e-06 | -4.43e-06 |

The sparse Chebyshev+Ritz eigensolver achieves machine-precision parity with
dense diagonalization for benzene and coronene, and ~1e-5 for circumcoronene
(with 30 filter iterations, Chebyshev degree 40).

## Parity — k-resolved slice

`test_graphene_bloch2d.py`, 3ob-3-1, plane at z = 1 Å:

- GPU `project_bloch_points` vs an independent float32 CPU spline sum: max |\Δψ| ~ 1e-8 (Γ π, and a sum of occupied π states).
- Bloch theorem at K = (2/3, 1/3), shifts of 1, 2, 3 cells along `a1`: max |ψ(r+ma1) − exp(−i 2π k_x m) ψ(r)| is 2e-8, 6e-8, 2e-7. |\ψ|² repeats to ~1e-8. The step of ψ across the cell edge is smaller than the interior steps.
- Ribbon mirror pairs `1H-p` / `1H-d` match in total energy (Hamiltonian check, not a grid check): C −30.944143 Ha, N −33.075399 Ha, O −56.864447 Ha.

## Open Issues

- **No cube-file comparison for the k-resolved kernel.** The check above is against a CPU copy of the same sum, not against `app/waveplot` cube output.
- **No DFTB+ cube file comparison** — the Fortran waveplot writes `.cube` files;
  we could compare the Rust-projected grid against those for numerical parity.
- **STO basis only** — the GridProjector uses STO basis from `wfc.mio-1-1.hsd`.
  For non-mio SK sets, the corresponding wfc file must be provided.
- **2D only** — currently only 2D plane projections. 3D grid projection is
  supported by `project_orbital_dense()` but not wired into the plotting script.
- **`project_bloch_points` is s+p only.** An atom with d orbitals raises. The point list is arbitrary, so a 3D sample is possible, but the drivers only cut a plane.
- **`project_orbital_periodic` is a stub.** It ignores k. The k-resolved entry point is `project_bloch_points`.
- **Ribbon driver is orthogonal vacuum cells.** Tilted self-junction stacks (`enumsj_*`) are refused. Complex eigenvectors are stored on the serial dense DFTBcore path only.
- **SPAMMM `compute_stm` is a different STM.** Γ-only, optional exponential tail, no k-phase. Do not treat those maps as this projector.
- **H atoms rejected by BSR4** — the sparse TC2 path requires 4 orbitals/atom,
  so H-containing systems can only use the dense eigenvector path for
  wavefunction projection.
- **Sparse Chebyshev+Ritz convergence** — circumcoronene needs ~30 iterations
  with Chebyshev degree 40 to reach ~1e-5 parity. The transformation now uses
  sparse Cholesky `S = L·Lᵀ` with an implicit `L⁻¹·H·L⁻ᵀ` operator (2 sparse
  triangular solves + 1 SpMV per matvec, no densification). For very large 3D
  systems, hierarchical Cholesky would be needed to control fill-in.

## Related

- `/doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md` — session report.
- `/scripts/plot_wavefunctions.py` — the plotting script.
- `/rust_dftb/src/methods/dftb/hamiltonian.rs::SccResult` — eigenvectors field.
- `/rust_dftb/src/bin/dftb_engine.rs::rhai_save_eigenvectors` — Rhai export.
- `/tests/grid/test_waveplot_dftbcore.py` — Γ-only reference workflow (DFTBcore lib, `Grid.cl`).
- `/doc/prokop/userguide/bloch_slice.md` — how to run the k-resolved slice.
- `/tests/grid/test_graphene_bloch2d.py`, `/tests/grid/test_ribbon_bloch.py` — the two drivers.
