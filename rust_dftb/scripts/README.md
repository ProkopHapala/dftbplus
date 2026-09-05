# rust_dftb/scripts/

Rhai test scripts for the `dftb_engine` binary. Run with:
`cargo run --bin dftb_engine -- --script <file.rhai> --sk-dir <path>`.

- **test_charges_homo_lumo.rhai** — end-to-end test: builds benzene, coronene,
  circumcoronene (pure-C PAHs), runs dense SCC, sparse TC2 purification, and
  Davidson partial eigensolver. Saves charges, eigenvalues, sparse charges, and
  convergence history to `debug/graphene_sparse/`. See
  `/doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md`.
- **test_graphene_sparse.rhai** — earlier sparse purification test on graphene
  flakes.
