# rust_dftb/

From-scratch Rust reimplementation of DFTB+/xTB semi-empirical LCAO solvers
with OpenCL GPU offload. Designed for batched many-small-system computation
on consumer GPUs.

## Build

```bash
# Requires system OpenBLAS (for LAPACK eigensolver)
# Ubuntu: sudo apt install libopenblas-dev

CARGO_TARGET_DIR=/home/prokophapala/.cargo-target-shared cargo build
```

## Dependencies

- **nalgebra** 0.33 — matrix algebra (small matrices, triangular solves)
- **lapack** 0.20 + **openblas-src** 0.10 — LAPACK FFI bindings, linked to system
  OpenBLAS. Used for `dsyevd` eigensolver in `qmqm/fragment.rs`. See
  `doc/prokop/topical_audit/eigensolver_performance.md` for why nalgebra's
  Jacobi is 29× slower.
- **ocl** 0.19 — OpenCL bindings for GPU kernels
- **rhai** 1.26 — scripting for test scenarios
- **ndarray**, **serde**, **serde_json**, **thiserror**, **log**, **env_logger**

## Run

```bash
# H-bond optimization example (needs SK files):
RUST_DFTB_SK_DIR=/path/to/slakos/mio/mio-1-1 \
cargo run --example hbond_ref -- \
  --xyz ../data/xyz/formic_azaindole_dimer.xyz \
  --mode optimize \
  --h1 14 --donor1 6 --acceptor1 17 \
  --h2 19 --donor2 18 --acceptor2 1 \
  --out ../debug/hbond_switching/reactant_opt.xyz \
  --data-dir ../debug/hbond_switching \
  --opt-max-iter 100 --opt-f-tol 5e-3 \
  --max-iter 50 --tol 1e-7

# Timing breakdown:
RUST_DFTB_TIMING=1 cargo run --example hbond_ref -- ...
# SCC per-iteration residual:
RUST_DFTB_SCC_VERBOSE=1 cargo run --example hbond_ref -- ...
```

## Layout

- `src/core/` — error types, neighbor lists, charge utilities
- `src/geometry/` — nanostructure builder (graphene, PAHs)
- `src/methods/dftb/` — DFTB SK tables, H0/S assembly, SCC, forces
- `src/methods/xtb/` — xTB analytical integrals, GFN1/2
- `src/methods/sparse/` — BSR4 sparse + GPU purification + Davidson eigensolver
- `src/qmqm/` — multi-fragment QM/QM solver + GPU runtime
- `src/bin/` — executables (`dftb_engine`, `graphene_build`)
- `examples/` — `hbond_ref`, `scan`, `neb`, `test_h2`, `debug_sk`
- `tests/` — parity tests vs Fortran DFTB+, GPU tests, integration tests
- `build.rs` — links system OpenBLAS for LAPACK
