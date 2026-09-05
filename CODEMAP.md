# CODEMAP — Repo Navigation Router

Big-picture map of this repo: what lives where, what we implemented, where the
references are. Start here when you need to find something.

## What this repo is

A fork of **DFTB+** (Fortran, upstream) plus a from-scratch **Rust reimplementation**
(`rust_dftb/`) of the semi-empirical LCAO solvers (DFTB, xTB) and a multi-system
QM/QM fragment solver with OpenCL GPU offload. Python utilities (`pyBall/`,
`tools/`) wrap/extend both.

## Top-level layout

- `src/dftbp/` — **Fortran reference** (upstream DFTB+). Subdirs: `dftb/`, `xtb/`,
  `math/`, `geometry/`, `io/`, `md/`, `elecsolvers/`, `poisson/`, `solvation/`,
  `transport/`, `timedep/`, `reks/`, `mixer/`, `api/`, … Use this as the
  ground-truth reference for parity tests.
- `rust_dftb/` — **our Rust reimplementation** (see below).
- `external/` — vendored upstream deps (submodules): `tblite/`, `dftd4/`,
  `s-dftd3/`, `mctc-lib/`, `multicharge/`, `slakos/`, `gbsa/`, `mbd/`, `libnegf/`,
  `mpifx/`, `scalapackfx/`, `fortuno/`, `fypp/`, …
- `app/` — Fortran executables (`dftb+`, `dftbcore`, `waveplot`, `modes`,
  `phonons`, `transporttools`, `misc`).
- `tools/` — upstream Python tooling: `dptools/`, `pythonapi/`, `misc/`.
- `pyBall/` — Prokop's Python layer (`AtomicSystem.py`, `DFTBcore.py`, `OCL/`,
  `WavePlot/`, …).
- `data/` — shared molecular inputs: `data/xyz/` (XYZ geometries), `data/mol/`
  (mol2).
- `tests/` — upstream-style test reference data: `tests/dftb/*_ref/`,
  `tests/grid/`. Inputs + reference outputs only — **no debug dumps**.
- `scripts/` — **reusable kept scripts** (Python/Bash) shared across tasks.
  Outputs go to `debug/`, never here. Key scripts:
  - `run_dftbplus_ref.py` — runs the Fortran DFTB+ binary on an XYZ, parses
    `detailed.out` + `band.out` → TSV charges/eigenvalues.
  - `plot_charges_homo_lumo.py` — spatial charge map + HOMO-LUMO energy levels
    (Rust dense vs sparse vs DFTB+ ref).
  - `plot_wavefunctions.py` — projects Rust DFTB MOs onto a 2D grid using the
    pyBall OpenCL `GridProjector` (STO basis from `wfc.mio-1-1.hsd`). Produces
    contour plots of HOMO, LUMO, and nearby orbitals.
  - `compare_homo_lumo_3way.py` — 3-way HOMO/LUMO comparison: Rust dense vs
    Rust sparse (Chebyshev+Ritz) vs DFTB+ Fortran (eigenvalues + wavefunction
    contour plots side-by-side).
  - `sparse_homo_lumo.py` — finds HOMO/LUMO via Chebyshev filter + Rayleigh-Ritz
    iterative eigensolver (from NumericalMathPlayground). Reads H,S matrices
    exported by Rust, transforms to standard form via sparse Cholesky L⁻¹HL⁻ᵀ,
    finds few eigenvalues near the gap without full diagonalization. Uses dense
    BLAS triangular solves for N<2000, spectral range rescaling for large N,
    adaptive Chebyshev parameters (nvec/deg/iters scale with system size).
  - `ribbon_scaling_test.py` — systematic scaling benchmark on H-passivated
    zigzag carbon ribbons (L=4..64, N=76..1156). Produces timing table, nnz
    counts, parity, and 4-panel scaling plot.
  - `compare_rust_vs_dftbplus.py` — numerical parity report (charges, eigenvalues,
    gaps).
  - `plot_formic_scan.py` — plots 1D/2D proton-transfer scan data from
    `rust_dftb/debug/formic_dimer_scan/*.tsv`. Produces energy PES, Mulliken
    charge, parity, and 2D contour plots (handles NaN for unconverged points).
  - `geometry_engine.py`, `plot_hbond.py`, `run_formic_dimer_*.sh`.
- `debug/` — **all debug artifacts** (PNGs, scratch CSVs, SCC dumps, one-off
  plots). Organized as `debug/<topic>/`. **Never commit anything here.**
- `doc/` — documentation (see below).
- `utils/`, `sys/`, `cmake/` — build/system helpers.

## `rust_dftb/` — the Rust crate

- `build.rs` — links system OpenBLAS (`-lopenblas`) for LAPACK eigensolver
  (`dsyevd` in `fragment.rs`). Required since `lapack` crate only provides FFI
  declarations, not the library itself.
- `src/core/` — method-agnostic primitives: `error.rs`, `neighbor.rs`,
  `charges.rs`.
- `src/geometry/` — nanostructure builder (graphene flakes/sheets/zigzag/armchair,
  PAHs, edge passivation with VSEPR sp² geometry).
- `src/methods/` — Hamiltonian methods + `traits.rs` (`H0Builder`, `CoulombModel`).
  - `dftb/` — DFTB SK-tables: `sk_data.rs`, `interpolation.rs`, `rotation.rs`,
    `hamiltonian.rs`, `gamma.rs`, `forces.rs`, `spline_resample.rs`,
    `dftb_hamiltonian.cl` (OpenCL).
  - `xtb/` — xTB analytical: `basis.rs`, `hamiltonian.rs`, `integrals.rs`,
    `coulomb.rs`, `mulliken.rs`, `scf.rs`, `params_gfn2.rs`,
    `multipole_integrals.rs`.
  - `sparse/` — BSR4 sparse + GPU sparse purification + partial eigensolver:
    `bsr4.rs`, `gpu_sparse.rs`, `sparse_bsr4_purification.cl`,
    `davidson.rs` (generalized Davidson for `H C = S C ε`, frontier orbitals).
- `src/qmqm/` — multi-fragment QM/QM solver + GPU runtime: `fragment.rs`,
  `solver.rs`, `mixer.rs`, `shifts.rs`, `gamma.rs`, `charges.rs`,
  `gpu_driver.rs`, `gpu_runtime.rs`, `gpu_matrix.rs`, `gpu_prep.rs`,
  `gpu_eigen.rs`/`.cl`, `gpu_matrix_ops.cl`, `gpu_scc.rs` (device-resident
  SCC loop driver with DIIS, warm-start, best-effort mode, per-system RMS
  diagnostics).
- `src/bin/` — executables: `dftb_engine.rs`, `graphene_build.rs`.
- `examples/` — runnable demos: `hbond_ref.rs`, `scan.rs`, `neb.rs`, `test_h2.rs`,
  `debug_h2.rs`, `debug_sk.rs`.
- `tests/` — Rust integration tests: `parity_*.rs` (vs Fortran), `gpu_*.rs`,
  `qmqm_integration.rs`, `xtb_parity.rs`, `scan.rs`, `formic_scan_plots.rs`
  (1D/2D proton-transfer scan with plotting, `--ignored`), `hbond_gpu_scc.rs`
  (H-bond GPU SCC parity) + Python runners (`run_parity.py`, `run_scc_full.py`,
  `run_forces.py`).
- `tools/` — crate-local Python tools: `sk_compress/` (SK table compression),
  `make_diatomic_hsd.py`, `run_dftbcore_dump.py`.
- `scripts/` — Rhai test scripts (e.g. `test_graphene_sparse.rhai`).

## `doc/` — documentation

- `doc/dftb+/`, `doc/dptools/`, `doc/waveplot/`, `doc/api/` — upstream docs.
- `doc/prokop/` — **our work**:
  - `DFTB_Reimplementation_Progress/` — design notes & status: `OVERVIEW_Roadmap.md`
    (master status checklist), `GPU_MultiSystem_Design.md`, `DFTB_Hassembly_OpenCL.md`,
    `Forces_Implementation_Notes.md`, `xTB_reimplementation.md`, …
  - `AGENTS/guidelines/` — repo-specific efficiency and coding guidelines,
    derived from code reviews. `efficiency.md` has 12 rules + 6 general
    principles (three-tier data lifetime, no allocation in hot loops, no
    strings/HashMaps in hot paths, verify before claiming, benchmark in
    release mode, exploit math structure, precompute polynomials, analytic
    derivatives, check units, warm starts). Referenced from `AGENTS.md`.
  - `tasts/<task>/` — **task specs only** (Markdown). e.g.
    `tasts/GPU_MultiSystem/` → `task_master.md` + `agent*.md`. No scripts/artifacts
    here; specs may *reference* `debug/...` paths.
  - `chats/` — design chat logs (`MultiSystemOpenCL.chat.md`, …).
  - `reports/` — written session reports (e.g.
    `2025-09-05_scc_charges_davidson_parity.md`,
    `2025-09-05_hbond_optimization_lapack.md`,
    `2025-09-06_gpu_hs_assembly_bugfix.md`,
    `2025-09-06_scan_plots_and_gamma_fix.md`,
    `2025-09-06_gpu_scc_benchmarks.md`).
  - `topical_audit/` — cross-implementation topic maps (one `.md` per topic:
    `davidson_eigensolver.md`, `eigensolver_performance.md`,
    `sparse_tc2_purification.md`, `dftbplus_parity_harness.md`,
    `scc_mulliken_charges.md`).

## Folder & artifact policy (concise)

- **`debug/`** — all debug output goes here as `debug/<topic>/`. **Never `git add`
  anything under `debug/`.** Not gitignored (kept navigable); rule is by convention.
  Stage with `git add -A -- . ':!debug/'` and review `git status` before committing.
- **`scripts/`** — reusable kept scripts. Write outputs to `debug/`, never into
  `scripts/` or task folders.
- **`doc/prokop/tasts/<task>/`** — Markdown specs only. No `scripts/`/`artifacts/`
  subfolders.
- **`tests/`** — reference data + inputs only. Debug dumps → `debug/scc/`.
- Before every commit: confirm nothing under `debug/` is staged, no large/regenerable
  files (`.png`, `.csv`, `.xyz`, `.log`) staged unless intended.
