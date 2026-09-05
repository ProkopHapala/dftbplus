#!/bin/bash
# Formic acid dimer (HCOOH)2 — 2D asynchronous double proton transfer PES
# 10 atoms, 28 orbitals (fits N<=64 one workgroup)
#
# t1 controls H(4): O(3) -> O(7)
# t2 controls H(9): O(8) -> O(2)
# Independent — reveals concerted vs stepwise mechanism
#
# Usage: ./run_formic_dimer_2d.sh [N1] [N2]
set -e

N1=${1:-21}
N2=${2:-21}
SK_DIR=${RUST_DFTB_SK_DIR:-/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1}
ROOT=/home/prokophapala/git/dftbplus
OUT=$ROOT/debug/gpu_multisystem/formic_dimer_2d_scc
BIN=/home/prokophapala/.cargo-target-shared/debug/examples/hbond_ref

export RUST_DFTB_SK_DIR=$SK_DIR

echo "=== Formic dimer 2D SCC PES ($N1 x $N2 = $((N1*N2)) points) ==="
$BIN \
    --xyz $ROOT/data/xyz/formic_dimer.xyz \
    --mode scc --scan 2d --n $N1 --n2 $N2 \
    --h1 4 --donor1 3 --acceptor1 7 \
    --h2 9 --donor2 8 --acceptor2 2 \
    --tol 1e-8 \
    --out $OUT.csv --data-dir $OUT

echo "=== Done. CSV: $OUT.csv ==="
