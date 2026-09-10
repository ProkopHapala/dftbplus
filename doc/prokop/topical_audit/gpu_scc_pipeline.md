---
type: TopicalAudit
title: GPU SCC Pipeline
tags: [topic, gpu, scc, opencl, batched, cross-language]
---

# GPU SCC Pipeline

## Summary

Device-resident self-consistent charge (SCC) loop on OpenCL GPU. All heavy
linear algebra (gamma matvec, H_scc update, GEMM, Jacobi eigensolver, density
build, Mulliken charges) runs as GPU kernels orchestrated by a host loop. The
only PCIe traffic per iteration is `batch` RMS floats (convergence check) and
`batch*n_atoms` charge floats (DIIS mixing). CPU-driven DIIS is allowed because
the charge vectors are tiny (n_atoms ≤ ~100) compared to the N² matrices.

## Architecture

Per SCC iteration (all on device):
1. Δq = q − q0 (elementwise)
2. V = G · Δq (gamma matvec, batched)
3. H_scc = H0 + 0.5·S·(V_i + V_j) (elementwise, orbital→atom map)
4. H' = X · H_scc · X (2 full-local batched GEMMs, X = S^{-1/2} precomputed once)
5. Jacobi(H') → ε (diag), C' (eigenvectors) (Brent-Luk parallel cyclic, N ≤ 64)
6. Read diag(H') → host sort → occ_mask → upload (N² readback, TODO: diagonal kernel)
7. C = X · C' (back-transform, 1 GEMM)
8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T (masked density build)
9. q_new = Mulliken(D, S) (device-resident)
10. CPU DIIS mixing → upload mixed charges

After convergence: `E = 2 Σ_{occ} ε − ½ Δq·V − q0·V` (algebraically the same as
`Tr(D·H0)+½Δq·V`; band form avoids the noisy f32 projector). See
`f32_floor_dense_hbond.md`.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Rust+OpenCL | `rust_dftb/src/qmqm/gpu_scc.rs` | active | `gpu_solve_scc_batched` (simple mix), `gpu_solve_scc_batched_diis` (DIIS), `gpu_solve_scc_batched_diis_warmstart` (warm-start + best-effort) |
| OpenCL kernels | `rust_dftb/src/qmqm/gpu_matrix_ops.cl` | active | gamma_matvec, h_scc_update, mulliken, residual_and_mix, build_density_masked, frobenius_trace, dot — all batched |
| OpenCL eigensolver | `rust_dftb/src/qmqm/gpu_eigen.rs`/`.cl` | active | `jacobi_cyclic_local_batched` (Brent-Luk), `build_inv_sqrt` (S^{-1/2}) |
| OpenCL forces | `rust_dftb/src/qmqm/gpu_forces.rs`/`.cl` | active | Analytic GPU forces (nonSCC + shift + gamma' + rep). H2O vs CPU rel ~3e-5 (`gpu_hbond_physics.rs`). 1×4 crash fixed (`vload2`). |
| OpenCL GEMM | `rust_dftb/src/qmqm/gpu_matrix.rs` | active | `matmul_full_local_batched` (both matrices in __local, N ≤ 64) |
| Rust (CPU ref) | `rust_dftb/src/methods/dftb/hamiltonian.rs` | reference | `HamiltonianBuilder::build_scc` — f64, LAPACK dsyevd, DIIS |
| Rust (CPU ref) | `rust_dftb/src/qmqm/solver.rs` | reference | `MultiSystemSolver::solve_scc` — multi-fragment |
| Test | `rust_dftb/tests/gpu_scc.rs` | active | H2O/N2/10×H2O parity <1e-6 |
| Test | `rust_dftb/tests/hbond_gpu_scc.rs` | active | Formic dimer 1D scan (21 pts), |dE|<2e-5, |dq|<6e-6 |
| Test | `rust_dftb/tests/formic_scan_plots.rs` | active | 1D (41 pts) + 2D (21×21) scan with plots, `--ignored` |
| Benchmark | `rust_dftb/tests/gpu_scc_bench.rs` | active | Timing at batch=1,10,41,100,441; `--ignored` |
| Plotting | `scripts/plot_formic_scan.py` | active | 1D energy/charge/parity + 2D contour plots |

## Key Design Decisions

- **f32 on GPU, with documented f64 islands** — N≤64 energies ~1e-6 vs CPU f64.
  N~90 (tiled Jacobi) hits an occupied-ε floor `|dE|~2.6e-5`. γ' forces already
  use f64. Map: `f32_floor_dense_hbond.md`.
- **Brent-Luk parallel cyclic Jacobi** — N/2 independent rotations per round,
  one barrier per round, A+V in `__local` for N ≤ 64. Workgroup size derived
  from N (power of 2, 32–1024).
- **S^{-1/2} via Jacobi eigendecomposition** — X = U·diag(rsqrt(λ))·U^T,
  computed once per geometry, reused every SCC iteration.
- **Full-local batched GEMM** — both A and B loaded into `__local` once per
  workgroup. One workgroup per system. N+1 leading dimension avoids
  power-of-two bank conflicts.
- **CPU-driven DIIS** — the mixer runs on CPU but only touches charge vectors
  (n_atoms per system). All matrix operations stay on GPU. This is the
  user-approved compromise.
- **Warm-start** — `gpu_solve_scc_batched_diis_warmstart` accepts separate
  `init_q_buf` distinct from `q0_buf`, enabling continuation from neighbouring
  converged charges.
- **Best-effort mode** — when `best_effort=true`, returns results with
  per-system RMS even if not all systems converged. Caller checks `rms[i] < tol`.
  Used for 2D scans where some asymmetric geometries don't converge (CPU also
  fails — physics limitation, not a bug).
- **Per-system RMS diagnostics** — on nonconvergence, reports the 10 worst
  systems with indices and RMS values, plus converged/failed count.

## Parity Status

| System | N_orbs | Batch | |dE| (Ha) | |dq| (e) | n_iters | Test |
|--------|--------|-------|----------|---------|---------|------|
| H2O | 6 | 1 | <3e-7 | <1e-6 | — | `gpu_scc.rs` |
| N2 | 8 | 1 | <1e-6 | <1e-6 | — | `gpu_scc.rs` |
| 10× H2O | 6 | 10 | <1e-6 | <1e-6 | — | `gpu_scc.rs` |
| Formic dimer (t=0) | 28 | 1 | 3.6e-6 | 2.9e-6 | 13 | `hbond_gpu_scc.rs` |
| Formic dimer 1D (21 pts) | 28 | 21 | <1.5e-5 | <6.2e-6 | 46 total | `hbond_gpu_scc.rs` |
| Formic dimer 1D (41 pts) | 28 | 41 | <3.6e-6 | <1e-5 | 13 | `formic_scan_plots.rs` |
| Formic dimer 2D (441 pts) | 28 | 21×21 | — | — | 147/441 conv | `formic_scan_plots.rs` |
| H2O full-chain (GPU H/S) | 6 | 1 | **3e-7** | 4e-7 | ~20 | `gpu_hbond_physics.rs` G3.4 |
| AT full-chain (GPU H/S) | 87 | 1 | **2.6e-5** (eigen floor) | 1.2e-5 | plateau rms~7e-6 | `gpu_hbond_physics.rs` G3.4 |

**2D nonconvergence:** 294/441 points unconverged. CPU also fails on these
highly asymmetric geometries (|t1−t2| > ~0.5). Root cause: SCC fixed-point
oscillation between competing charge transfer states. Not a GPU bug.

## Open Issues

- **SK interpolator stopgap (2026-09-09)** — blunt extra zero *samples* on
  the right + left phantom knot. Kills Neville-tail explosion. **Not** the
  intended BC: extra controls must be *fitted*. See `sk_interpolation.md`.
- **AT/GC GPU SCC leftover is `δ_CH`, not a mixer bug and not occupied-ε 2.5e-5** —
  Package 2 frozen-H: `max|δε_occ|~1e-6`; `|dE|~3e-5` tracks `E_band−2ΣCᵀHC`.
  Löwdin Newton + f32 GEMM Kahan in; Kahan does not cut `δ_CH`. Honest contract:
  N~90 `|dE|<1e-4` (regression), not `<1e-5`. SSOT: `f32_floor_dense_hbond.md` §3.1.
- **Kernel objects rebuilt each call** — `Kernel::builder().build()` per
  iteration. Program cache hits, but Kernel handle creation is a performance
  TODO. Target: cache Kernel objects in `GpuRuntime`.
- **Full N² readback for eigenvalues** — `hp` (N² per system) read back to
  extract diagonal for sorting. For N≤64, batch≤1000 this is ≤256 KB —
  acceptable but a diagonal-extract kernel would eliminate it.
- **No active mask** — all systems run all iterations even if some converged.
  Target: `gpu_scc.rs` active mask early-return.
- **2D scan convergence** — 67% of 2D points don't converge (CPU also fails).
  Possible fixes: Broyden mixing, level shifting, smaller alpha for asymmetric
  geometries, strip-by-strip propagation with neighbouring 2D warm-start.
- **No GPU DIIS** — DIIS runs on CPU. Could be ported to GPU for very large
  batches, but current bottleneck is Jacobi eigensolver, not mixing.

## Related

- `/doc/prokop/topical_audit/scc_mulliken_charges.md` — Mulliken charge sign convention
- `/doc/prokop/topical_audit/eigensolver_performance.md` — CPU eigensolver performance
- `/doc/prokop/reports/2025-09-06_gpu_hs_assembly_bugfix.md` — H0/S assembly bug fixes
- `/doc/prokop/reports/2025-09-06_scan_plots_and_gamma_fix.md` — scan validation, gamma fix
- `/doc/prokop/reports/2025-09-06_gpu_scc_benchmarks.md` — timing benchmarks
- `/doc/prokop/topical_audit/sk_interpolation.md` — SK B-spline BCs (stopgap vs intended fitter)
- `/doc/prokop/topical_audit/f32_floor_dense_hbond.md` — bugs vs measured f32 arithmetic floor
- `/doc/prokop/DFTB_Reimplementation_Progress/GPU_MultiSystem_Design.md` — design doc
