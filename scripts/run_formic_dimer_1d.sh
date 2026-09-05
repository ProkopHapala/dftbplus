#!/bin/bash
# Formic acid dimer (HCOOH)2 — 1D synchronous double proton transfer scan
# 10 atoms, 28 orbitals (fits N<=64 one workgroup)
#
# H(4) transfers O(3) -> O(7)  (left to right)
# H(9) transfers O(8) -> O(2)  (right to left, opposite direction)
# Both move with same parameter t in [0,1]
#
# Usage: ./run_formic_dimer_1d.sh [N_POINTS]
set -e

N=${1:-21}
SK_DIR=${RUST_DFTB_SK_DIR:-/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1}
ROOT=/home/prokophapala/git/dftbplus
OUT=$ROOT/debug/gpu_multisystem/formic_dimer_1d_scc
BIN=/home/prokophapala/.cargo-target-shared/debug/examples/hbond_ref

export RUST_DFTB_SK_DIR=$SK_DIR

echo "=== Formic dimer 1D SCC scan ($N points) ==="
$BIN \
    --xyz $ROOT/data/xyz/formic_dimer.xyz \
    --mode scc --scan 1d --n $N \
    --h1 4 --donor1 3 --acceptor1 7 \
    --h2 9 --donor2 8 --acceptor2 2 \
    --tol 1e-8 \
    --out $OUT.csv --data-dir $OUT

echo "=== Done. CSV: $OUT.csv ==="
