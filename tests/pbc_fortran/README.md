# pbc_fortran — Fortran DFTB+ reference for the GPU PBC path

Finite-precision numerical parity between `GpuPbc` (complex k-point,
OpenCL) and the upstream Fortran `dftb+` binary.

## Test system

`dftb_in.hsd`: periodic C–O chain — C at (0,0,0), O at (1.2,0,0) Å,
lattice 3.0 Å along x, 20×20 Å transverse, 4 explicit fractional
k-points (Γ, ±0.25, 0.5), mio-1-1 SK set, SCC to 1e-8.

Heteropolar cell chosen deliberately: net Mulliken transfer
(±0.096 e) means the SCC loop actually iterates — a homonuclear
cell converges trivially. Intercell C–O gap (1.8 Å) is inside the
SK cutoff, so the run exercises image-cell SK blocks, Ewald γ, and
the Bloch fold at generic k (±0.25) and at the BZ edge (0.5).

## Files

| file | role |
|---|---|
| `dftb_in.hsd` | Fortran input (static SCC, explicit k-points) |
| `run_reference.sh` | runs the Fortran binary in `work/`, regenerates `reference.txt` |
| `extract_reference.py` | parses `detailed.out` + `band.out` → `reference.txt` |
| `reference.txt` | checked-in reference: net charges, per-k eigenvalues (Ha), energies (Ha) |
| `work/` | scratch dir created by the script — do not commit artifacts |

The Rust side is `rust_dftb/tests/gpu_pbc_fortran.rs`; it mirrors the geometry,
lattice, and k-list from `dftb_in.hsd` — keep them in sync.

## Reproduce

```bash
cd tests/pbc_fortran
./run_reference.sh                          # regenerate reference.txt
cd ../../rust_dftb && cargo test --release --test gpu_pbc_fortran -- --nocapture
```

Needs `_build/app/dftb+/dftb+` (override with `DFTB_BIN=...`) and the
mio-1-1 SK dir (path in `dftb_in.hsd` / `RUST_DFTB_SK_DIR`).

## What is compared (finite precision — never binary equality)

| quantity | rust source | fortran source | tol | measured |
|---|---|---|---|---|
| Mulliken populations | `read_charges` (q_gpu) | `detailed.out` gross charges (q0 − net) | 1e-3 e | ~1.7e-4 e |
| eigenvalues per k | `read_eigenvalues` | `band.out` (eV→Ha) | 2e-3 Ha | ~1.9e-5 Ha |
| band energy | `e_scal_host[0]` | `detailed.out` "Band energy" | 5e-3 Ha | ~7e-5 Ha |
| electronic energy | `compute_energy` = e_band − ½Δq·v − q0·v | "Total Electronic energy" | 5e-3 Ha | ~1.8e-6 Ha |

Tolerances sit ~an order of magnitude above the measured floor —
they catch real regressions (wrong phase convention, missing image
shells, γ sign) without f32 noise trips. band.out is itself printed
at ~4e-6 Ha resolution.
