#!/bin/bash
# Regenerate the Fortran DFTB+ reference for the GPU PBC parity test.
#
# Runs _build/app/dftb+/dftb+ on dftb_in.hsd (periodic C-O chain,
# 4 explicit k-points, SCC) inside ./work/, then extracts
# charges / energies / per-k eigenvalues into reference.txt.
#
# Usage:  ./run_reference.sh            (from tests/pbc_fortran/)
# Needs:  the built Fortran binary and the mio-1-1 SK set
#         (SK path is hard-coded in dftb_in.hsd -- edit there).
#
# After regenerating, re-run the Rust side:
#   cargo test --release --test gpu_pbc_fortran -- --nocapture
set -euo pipefail
cd "$(dirname "$0")"

DFTB_BIN="${DFTB_BIN:-/home/prokop/git/dftbplus/_build/app/dftb+/dftb+}"

mkdir -p work
cp dftb_in.hsd work/
( cd work && "$DFTB_BIN" dftb_in.hsd )
python3 extract_reference.py work reference.txt
