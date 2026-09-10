#!/usr/bin/env bash
# Sparse solver L0 review entry point (Gate 0.6).
# NVIDIA + matsci-0-3, --test-threads=1 --nocapture, tee to debug/sparse_review/.
# Non-zero exit on test failure OR a harness skip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/rust_dftb"

export RUST_DFTB_SK_DIR="${RUST_DFTB_SK_DIR:-/home/prokop/SIMULATIONS/dftbplus/slakos/matsci-0-3}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-full}"
export OPENBLAS_NUM_THREADS=1

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$ROOT/debug/sparse_review/${STAMP}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/cargo_test.log"

echo "=== sparse review ${STAMP} ==="
echo "SK dir: $RUST_DFTB_SK_DIR"
echo "log:    $LOG"

# Algebra + physics gates (G3, F, G). Do not filter stdout; a skip in the log is a failed review.
run_one() {
  cargo test "$@" -- --test-threads=1 --nocapture 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

set +e
run_one \
  --test gpu_sparse_bsr4 \
  --test gpu_bspline_eval \
  --test spgemm_plan \
  --test locality_sweep \
  --test sih_padded_basis \
  --test gate_g3_energy \
  --test gate_e_determinism \
  --test gate_f_geom_opt \
  --test gate_g_hessian
STATUS_INT=$?
run_one --lib sparse_
STATUS_LIB=$?
set -e

STATUS=0
if [[ "$STATUS_INT" -ne 0 || "$STATUS_LIB" -ne 0 ]]; then
  STATUS=1
fi

if grep -E "Skipping (Gate|GPU sparse|sparse GPU|locality|P3|P4)" "$LOG" >/dev/null; then
  echo "FAIL: harness skip detected in $LOG (G0.3/G0.6)"
  exit 1
fi

if [[ "$STATUS" -ne 0 ]]; then
  echo "FAIL: cargo test exit $STATUS"
  exit "$STATUS"
fi

echo "=== sparse review finished (exit 0). Inspect $LOG — green is not proof of physics. ==="
exit 0
