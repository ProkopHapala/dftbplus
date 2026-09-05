---
type: TopicalAudit
title: Wavefunction Projection onto Real-Space Grid
tags: [topic, wavefunction, grid, opencl, sto, visualization, cross-language]
---

# Wavefunction Projection onto Real-Space Grid

## Summary

Projects molecular orbital coefficients (from the LCAO eigenvector matrix) onto a
real-space 3D grid or 2D plane using Slater-type orbital (STO) basis functions.
The OpenCL GPU kernel evaluates `ψ(r) = Σ_i c_i · φ_i(r)` for each MO at each
grid point, where `φ_i` are the atomic STO basis functions. Used to visualize
HOMO, LUMO, and frontier orbitals after SCC convergence.

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
| Fortran | `src/dftbp/waveplot/` | reference | Upstream waveplot (Fortran) — writes cube files. |

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

Both Rust and DFTB+ use tesseral/real spherical harmonics ordered by magnetic
quantum number m:
- l=0 (s): `[s]`
- l=1 (p): `[py (m=-1), pz (m=0), px (m=+1)]` — NOT px,py,pz!

This is documented in `rust_dftb/src/methods/dftb/rotation.rs:29-33` and handled
by `evec_to_kernel_coeffs()` in `DFTBplusParser.py:1052-1084`.

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

## Open Issues

- **No DFTB+ cube file comparison** — the Fortran waveplot writes `.cube` files;
  we could compare the Rust-projected grid against those for numerical parity.
- **STO basis only** — the GridProjector uses STO basis from `wfc.mio-1-1.hsd`.
  For non-mio SK sets, the corresponding wfc file must be provided.
- **2D only** — currently only 2D plane projections. 3D grid projection is
  supported by `project_orbital_dense()` but not wired into the plotting script.
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
- `/tests/grid/test_waveplot_dftbcore.py` — reference workflow (DFTBcore lib).
