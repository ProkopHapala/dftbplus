# rust_dftb/scripts/

Input scripts for **`dftb_engine`** (the user CLI). How to write one:
[`doc/prokop/userguide/dftb_engine.md`](../../doc/prokop/userguide/dftb_engine.md).

```bash
cargo run --release --bin dftb_engine -- --script scripts/<file>.rhai --sk-dir <mio-1-1>
```

- **test_gpu_dftb_molecules.rhai** — production `GpuDftb`: H2O batch=1 and 4,
  AT batch=1 and 2, GC, 7-azaindole dimer. Replica energies must agree.
  NVIDIA required. Run `--release`.
- **test_gpu_dftb_measure.rhai** — Package 2 diagnostics on `GpuDftb`: frozen-H
  Jacobi residual, mixer 0/1/2 on H2O, H2O forces vs CPU, formic-dimer relative
  ΔE scan. NVIDIA required. Run `--release`.
- **test_charges_homo_lumo.rhai** — end-to-end test: builds benzene, coronene,
  circumcoronene (pure-C PAHs), runs dense SCC, sparse TC2 purification, and
  Davidson partial eigensolver. Saves charges, eigenvalues, sparse charges, and
  convergence history to `debug/graphene_sparse/`. See
  `/doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md`.
- **test_graphene_sparse.rhai** — earlier sparse purification test on graphene
  flakes.
- **scan2d_pyridone.rhai** — pyridone-dimer 20×20 double-proton-transfer scan
  (fixed scaffold, batch=400, neutral ground state).
- **scan2d_pyridone_cdft.rhai** — same grid × 3 electronic states: neutral,
  Q₁=−1 e (M1⁺M2⁻), Q₁=+1 e (M1⁻M2⁺) via `gpu_cdft*`. The CDFT worked
  example — see `doc/prokop/userguide/cdft.md` and
  `doc/prokop/topical_audit/cdft_constraints.md`.
- **plot_scan2d_cdft.py** — renders multi-`MAP` blocks from a scan log into
  a side-by-side PNG (per-map contours + CT−neutral excitation panels),
  e.g. `debug/pyridone_2d_scan_cdft.png`.
- **make_qxhq_chain.py** — builds the periodic quinoxaline/
  dihydroquinoxaline (QX/HQ) chain cell from ASCII art (SPAMMM
  `ascii_art_heterocycle`): herringbone tilt about the N–N spine, wraps
  y into [0, Ly), exports `data/xyz/qxhq_chain_cell.xyz` + `.gen`, plots
  `debug/qxhq_chain.png`.
- **scan2d_qxhq_pbc.rhai** — periodic 2-D proton-transfer scan on that
  cell via the `pbc_*` engine functions (`GpuPbc`, nk=4 along y): J1 is
  intracell, J2's acceptor is in the next cell. 20×20 grid, batch=400.
  See `doc/prokop/userguide/hbond_pbc_scans.md`.
- **plot_scan2d_pbc.py** — renders the PBC scan log's `ENERGY MAP` block
  into `debug/qxhq_pbc_Emap.png`.
- **sparse_vib_c330.rhai** — frozen-density H/S-block Hessian of the
  relaxed 330-atom carbon particle (3ob). Guide
  `doc/prokop/userguide/sparse_vibrations.md` §0.
- **sparse_vib_ref.rhai** — same Hessian at a geometry given by
  `RUST_DFTB_XYZ`. Used for the adamantane and Si₁₀H₁₆ DFTB+ comparison.
- **plot_vib_spectrum.py** — stick spectrum of a `sparse_vibrations` file.
- **plot_vib_parity.py** — spectrum, Hessian heatmap, and frequency
  correlation, including a DFTB+ `hessian.out`.
