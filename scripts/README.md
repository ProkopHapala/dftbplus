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
- **compare_rust_vs_dftbplus.py** — numerical parity report: per-atom charge
  residuals, HOMO/LUMO/gap differences, eigenvalue max/RMS residuals.
- **geometry_engine.py** — geometry building utility (wraps the Rust
  `graphene_build` binary for PAH/flake/ribbon generation).
- **plot_hbond.py** — H-bond analysis plot (formic acid dimer scan).
- **run_formic_dimer_1d.sh** — 1D formic acid dimer scan runner.
- **run_formic_dimer_2d.sh** — 2D formic acid dimer scan runner.
