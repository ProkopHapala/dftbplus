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
