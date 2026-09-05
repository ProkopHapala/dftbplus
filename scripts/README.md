# scripts/

Reusable kept scripts (Python/Bash) shared across tasks. Outputs go to `debug/`,
never into this folder.

- **run_dftbplus_ref.py** — runs the Fortran DFTB+ binary on an XYZ, parses
  `detailed.out` (Mulliken charges, Fermi level, total energy) and `band.out`
  (eigenvalues in eV → Ha, occupations) → TSV files. Usage:
  `python3 run_dftbplus_ref.py <xyz> <out_dir> --sk-dir <path>`.
- **plot_charges_homo_lumo.py** — generates two plots from TSVs: spatial 2D
  charge map (Rust dense vs sparse vs DFTB+) and HOMO-LUMO energy-level diagram.
  Converts Rust Mulliken populations to charges (`q = q0 - population`).
- **plot_wavefunctions.py** — projects Rust DFTB MOs onto a 2D grid using the
  pyBall OpenCL `GridProjector` (`pyBall/OCL/Grid.py`). Reads eigenvectors from
  `*_eigenvectors.tsv` (saved by Rhai `save_eigenvectors`), loads the STO basis
  from `wfc.mio-1-1.hsd`, and evaluates MOs on a 2D plane (xy/xz/yz) via OpenCL.
  Produces contour plots with atom positions marked. Usage:
  `python3 plot_wavefunctions.py <eigenvectors.tsv> --wfc <wfc.mio-1-1.hsd> --plane xy`.
- **sparse_homo_lumo.py** — finds HOMO/LUMO via Chebyshev filter + Rayleigh-Ritz
  iterative eigensolver (from `NumericalMathPlayground/topics/LinearAlgebra/
  SpectralFiltering/spectral_solvers.py`). Reads H,S matrices from Rust
  (`*_hs_matrix.tsv`), transforms the generalized problem to standard form via
  S^{-1/2}, then applies Chebyshev polynomial filtering + Rayleigh-Ritz to
  extract eigenpairs in two bands (around HOMO and LUMO) without full
  diagonalization. Usage:
  `python3 sparse_homo_lumo.py <hs_matrix.tsv> --nvec 12 --cheb-deg 40 --iters 30`.
- **compare_homo_lumo_3way.py** — 3-way HOMO/LUMO comparison: Rust dense vs
  Rust sparse (Chebyshev+Ritz) vs DFTB+ Fortran. Prints eigenvalue table
  (HOMO, LUMO, gap differences) and generates side-by-side wavefunction contour
  plots using the OpenCL GridProjector. Usage:
  `python3 compare_homo_lumo_3way.py <debug_dir> --wfc <wfc.mio-1-1.hsd>`.
- **compare_rust_vs_dftbplus.py** — numerical parity report: per-atom charge
  residuals, HOMO/LUMO/gap differences, eigenvalue max/RMS residuals.
- **geometry_engine.py** — geometry building utility (wraps the Rust
  `graphene_build` binary for PAH/flake/ribbon generation).
- **plot_hbond.py** — H-bond analysis plot (formic acid dimer scan).
- **run_formic_dimer_1d.sh** — 1D formic acid dimer scan runner.
- **run_formic_dimer_2d.sh** — 2D formic acid dimer scan runner.
