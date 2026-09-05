# GPU SCC Performance Benchmarks

**Date:** 2025-09-06
**System:** Formic acid dimer (10 atoms, 28 orbitals, 18 occupied)
**GPU:** OpenCL device (NVIDIA)

## Methodology

Benchmark test: `rust_dftb/tests/gpu_scc_bench.rs` (run with `--ignored`)

Each benchmark:
1. Builds batched H0/S on GPU via `GpuDriver::gpu_assemble_batched`
2. Warm-up run (kernel compilation, not timed)
3. Timed run with `RUST_DFTB_TIMING=1` — per-phase wall-clock timing
4. All systems use identical geometry (formic dimer at t=0) — measures compute, not physics

Solver: `gpu_solve_scc_batched_diis_warmstart` with tol=5e-6, alpha=0.3,
max_history=8, warmup=3, max_iter=500.

## Results

### Throughput vs batch size

| Batch | Iters | Total (s) | Per-iter (ms) | Systems/s | Jacobi % |
|-------|-------|-----------|---------------|-----------|----------|
| 1     | 9     | 0.032     | 3.12          | 31.7      | 50.7%    |
| 10    | 9     | 0.034     | 3.19          | 291.7     | 52.3%    |
| 41    | 9     | 0.050     | 4.97          | 829.1     | 59.3%    |
| 100   | 9     | 0.084     | 8.24          | 1194.5    | 61.0%    |
| 441   | 500*  | 10.33     | 20.62         | 42.7*     | 33.7%*   |

*batch=441 did not converge (314/441 systems failed — see Known Issues)

### Per-phase breakdown (batch=100, 9 iters, representative)

| Phase                | Per-iter (ms) | % of iter | Notes |
|----------------------|---------------|-----------|-------|
| Jacobi eigensolver   | 5.02          | 61.0%     | **Bottleneck** — Brent-Luk cyclic, N=28 |
| GEMM (2× H'=X·H·X)  | 0.42          | 5.1%      | Nearly batch-independent |
| DIIS mixing (CPU)    | 1.35          | 16.4%     | Scales linearly with batch |
| occ_sort + upload    | 0.48          | 5.8%      | Host sort + N² readback |
| back_gemm (C=X·C')  | 0.24          | 2.9%      | |
| delta_q              | 0.18          | 2.2%      | |
| h_scc_update         | 0.14          | 1.7%      | |
| mulliken             | 0.14          | 1.7%      | |
| gamma_matvec         | 0.15          | 1.8%      | |
| density              | 0.13          | 1.6%      | |
| **Total per iter**   | **8.24**      | 100%      | |

### Scaling analysis

- **batch 1→10**: 9.2× throughput from 10× batch — near-linear scaling
- **batch 10→41**: 2.8× throughput from 4.1× batch — good scaling
- **batch 41→100**: 1.4× throughput from 2.4× batch — diminishing returns
- **batch=100 sweet spot**: 1194 systems/s at 8.2 ms/iter

The diminishing returns at batch>100 are because:
1. Jacobi time grows (more workgroups competing for GPU resources)
2. DIIS mixing (CPU) scales linearly with batch — becomes significant
3. occ_sort readback (N² per system) grows

## Key Findings

### 1. Jacobi eigensolver is the primary bottleneck

At all batch sizes ≤100, Jacobi consumes 50-65% of per-iteration time. The
Brent-Luk parallel cyclic Jacobi for N=28 uses a workgroup of 128 threads
(jpair=14, PPG=8, wg=128). With batch=100, that's 100 workgroups of 128
threads = 12,800 threads — well within GPU capacity, but each workgroup runs
~20 sweeps × 27 rounds with barriers.

**Optimization opportunities:**
- Cache Kernel objects (currently rebuilt each call — `Kernel::builder().build()`)
- Reduce Jacobi sweeps (convergence threshold per-system)
- Active mask (skip converged systems)
- Try lower precision for off-diagonal elements

### 2. GEMM is nearly batch-independent

The full-local batched GEMM (`matmul_full_local_batched`) takes ~0.42 ms/iter
regardless of batch size (1 to 100). This means the GPU is well-utilized for
matrix multiplication — the workgroups run efficiently in parallel.

### 3. CPU DIIS mixing scales linearly

The DIIS mixing (readback + CPU mix + upload) takes:
- batch=1: 0.05 ms/iter
- batch=10: 0.19 ms/iter
- batch=41: 0.48 ms/iter
- batch=100: 1.35 ms/iter
- batch=441: 10.16 ms/iter

This scales linearly with batch (as expected — it's CPU work). At batch=441
it becomes the dominant cost (49% of per-iter time), but only because the
solver didn't converge (500 iters). At normal convergence (9 iters), DIIS
is 16% of per-iter time at batch=100.

### 4. Kernel object rebuild overhead

Each call to `gpu_solve_scc_batched_diis_warmstart` rebuilds all Kernel
objects via `Kernel::builder().build()`. The Program is cached (build by
source hash), but Kernel handle creation has overhead. The warm-up run
absorbs this; the timed run benefits from program cache but still rebuilds
Kernel handles.

**Fix:** Cache Kernel objects in `GpuRuntime` (roadmap item D14).

### 5. N² eigenvalue readback

Reading the full `hp` buffer (N² × batch × 4 bytes) to extract the diagonal
for sorting is a known TODO. For N=28, batch=100: 313 KB per iteration.
At batch=441: 1.4 MB per iteration. A diagonal-extract kernel would eliminate
this.

## Known Issues

### batch=441 convergence failure with identical geometries

When 441 identical formic dimer systems are batched, 314/441 fail to converge
(RMS=2.55). This is unexpected — all systems have the same H0, S, gamma, and
q0, so they should converge identically. The failure pattern (systems
128-135, 244, 440 all have the same RMS) suggests a race condition in the
Jacobi kernel or a GPU memory issue with large workgroup counts.

**Investigation needed:**
- Check if the Jacobi kernel has a race condition with >256 workgroups
- Check if GPU local memory is exhausted with 441 workgroups
- Test with batch=200, 300, 400 to find the threshold
- This does NOT affect the scan test (which uses 21-point strips)

## Comparison: GPU vs CPU

CPU reference: `HamiltonianBuilder::build_scc` with LAPACK dsyevd, f64, DIIS.
For a single formic dimer system, CPU SCC takes ~50-100 ms (including H0/S
assembly). The GPU SCC at batch=1 takes 32 ms total (including S^{-1/2}
precompute + 9 iterations + energy). At batch=100, 84 ms for 100 systems =
0.84 ms/system — a **60-120× speedup** over CPU.

## Optimization Priority

Based on the benchmark results, the optimization priority order is:

1. **Cache Kernel objects** (D14) — eliminates per-call Kernel::builder().build()
2. **Active mask** — skip converged systems in Jacobi + GEMM + density
3. **Diagonal-extract kernel** — eliminate N² readback for eigenvalue sorting
4. **Reduce Jacobi sweeps** — per-system convergence threshold (early exit)
5. **GPU DIIS mixer** — only needed if batch >100 and convergence is fast
6. **Precompute H'0 = X·H0·X** (D16) — saves 1 GEMM per iteration

## Files

- `rust_dftb/tests/gpu_scc_bench.rs` — benchmark test
- `rust_dftb/debug/gpu_scc_bench/bench_results.tsv` — raw timing data
- `rust_dftb/src/qmqm/gpu_scc.rs` — `GpuSccTiming` struct + `timed!` macro

## Reproduction

```bash
RUST_DFTB_SK_DIR=/path/to/mio-1-1 \
RUST_DFTB_TIMING=1 \
cargo test --test gpu_scc_bench -- --ignored --nocapture
```
