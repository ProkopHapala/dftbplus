# Dense_Multi_PBC — architecture notes

Companion to `Dense_Multi_PBC.chat.md` (prototype kernel). Implementation
landed in `rust_dftb/src/qmqm/`:

| file | role |
|---|---|
| `gpu_hermitian_jacobi.cl` | complex Hermitian cyclic Jacobi, one WG per flat (replica,k) system |
| `gpu_zmatrix_ops.cl` | complex GEMM (N/T/H ops), zdq_v, zhscc, kpoint_occ, zsc_mulliken, kpoint_qreduce, zsnormalize, zscale_eigenvectors, zkpoint_energy_tail |
| `gpu_hermitian.rs` | source renderers + standalone driver helpers for tests |
| `gpu_pbc_plan.rs` | `GpuPbcPlan` — persistent plan, the complex sibling of `GpuSccPlan` |
| `pbc_cell.rs` | host cell math: lattice/reciprocal vectors, R/G lists, Ewald α/cutoff autotuning, image-pair enumeration, f64 Ewald reference |
| `gpu_pbc.cl` | `assemble_pairs_img` (per-slot SK blocks), `kpoint_phase_sum_batched` (Bloch fold), `ewald_invr_batched`, `gamma_pbc_batched` |
| `gpu_pbc.rs` | `GpuPbc` driver — cell + Ewald params + all lists/buffers/kernels built once at construction; `set_geometry` re-runs assemble→fold→γ |
| `tests/gpu_hermitian_jacobi.rs` | standalone validation |
| `tests/gpu_pbc.rs` | PBC validation (Ewald analytic targets, Bloch symmetries, end-to-end SCC) |

## Real-valued path (what we mirror)

`GpuSccPlan` (`gpu_scc_plan.rs`) owns every scratch buffer and every
`Kernel` handle for a fixed `(n, n_atoms, batch)`. Kernels are built once
at construction with the io buffers already bound; per-iteration work is
`kernel.enq()` only. Per-solve scalars (n_occ, kT, alpha, rms_tol) are
re-bound via `bind_solve_params`/`bind_mix_params` — never per iteration.
SCC iterations are enqueued in chunks (`scc_step_diis_enq` does zero host
I/O); `read_chunk_status` is the only sync point, once per chunk. DIIS
clears `active[b]` on the device when a replica converges, so all later
kernels in the same chunk gate on the mask — a done replica stops doing
work with no host round-trip. `check_jacobi` defers Jacobi certification
to a single readback at solve end.

## Complex/PBC differences

- Matrices are `float2` `(re,im)` row-major, `[n_sys][n*n]` where
  `n_sys = n_rep·nk` and `sid = rep·nk + kpt`. Charges/γ/DIIS stay REAL,
  per replica — one `active_r[n_rep]` mask, flat kernels read
  `active[sid/nk]`.
- Hermiticity is a *restoring invariant*: `A ← (A+A†)/2` once per sweep
  (one extra round, O(n²) — cheap vs the O(n³) sweep). f32 complex
  rotations otherwise accumulate an anti-Hermitian drift.
- Jacobi: same Brent–Luk cyclic schedule and update structure as
  `jacobi_cyclic_global_batched`, but complex pair rotations
  (`e^{iφ}c`, `e^{iφ}s` — the phase-fix convention `A'_pq ∈ ℝ`). Warm
  path rotates `c` in place exactly like the real kernel (init_v=1).
- `S^{-1/2}` per k: Jacobi(S) → `zscale` (V·rsqrt λ) → `zgemm` (X = Vs·V†,
  op_b=H). No Newton/Löwdin reuse across geometries in v1 — set_geometry
  is a full rebuild (marked, fail-loud on λ_min ≤ 1e-6).
- Occupation is a *k-point reduction*: `kpoint_occ_batched` runs a shared
  μ solve (f64 bracket+Newton, 48 iters) over all nk bands of a replica,
  writes `occ_w = w_k·f`, `mu`, `e_band`, `mts`. kT=0 falls back to
  bisection on the weighted band count — one kernel covers both modes.
- Mulliken without materializing D: `zgemm` computes `SC = S·C`, then
  `zsc_mulliken_batched` contracts `qk_μ = 2·Re Σ_t w_t C_μt conj(SC_μt)`
  and `kpoint_qreduce_batched` sums `q_new = Σ_k qk`. Per-k charges are
  real because Σ over the BZ of a Hermitian-consistent set is real.
- Energy tail is real-only (`zkpoint_energy_tail_batched`): e_scal =
  {e_band, mts, dq·v, q0·v} in f64, one readback; E assembled on host.

## Periodic assembly (landed — `gpu_pbc.rs`/`gpu_pbc.cl`/`pbc_cell.rs`)

- `GpuPbc::new` freezes the pair/R-image SET at construction (margin
  `PAIR_MARGIN` = 2 Bohr): one shared integer-cell table covers
  max(SK cutoff, Ewald maxR, γ cutoff) + extent. Geometry updates only
  re-evaluate distances.
- Per geometry: zero-fill H0(k)/S(k) → per SK bucket `assemble_pairs_img`
  (one WI per (replica,slot), k-independent real blocks) →
  `kpoint_phase_sum_batched` (one WI per (rep,out-pair,k,element);
  `diag` adds onsite+I, `herm` writes the conj side) →
  `ewald_invr_batched` (per-pair CSR real sum + half-space G sum +
  −π/Vα² + −2α/√π diag; f64 accumulator) → `gamma_pbc_batched`
  (invr − Σ_short expGamma, includes the onsite slot).
- Phase convention: `e^{+i·k_cart·R_cart}` with k_cart = rec·k_frac
  (matches Fortran `exp(imag·2π·k_frac·R)`).
- Ordering constraint: `GpuPbcPlan::new` runs the S^{-1/2} pipeline at
  construction, so the driver runs one full assembly chain first —
  buf_s must hold valid S(k).
- Real-space Ewald filters by PAIR distance |r_ij+R| < maxR (neighbor-
  list convention), not by |R| < maxR — an origin-centered R-ball
  misses far-side images (~2.6e-4 Madelung error, caught by the NaCl
  test).

## Deferred (v1 limits — loud, not silent)

- No complex Newton warm-basis repair across geometries.
- No PBC forces/stress (Ewald + SK derivatives investigated, not wired).
- No EDM/forces (zbuild_density exists but is unused on the charge path).
- No occ_mask/occ_idx integer-occupation sort path — `kpoint_occ` covers
  both integer and smeared cases via the shared-μ solve.
- No Fortran `dftb+` binary parity test yet (gated, `_build` exists).

## Measured (tests/gpu_hermitian_jacobi.rs, RTX-class device)

- Residual ‖AV−VΛ‖/‖A‖ ~ 1e-6, unitarity ‖V†V−I‖/N ~ 1e-7 for N ≤ 128.
- Eigenvalue parity vs `zheevd` inside the Weyl bound everywhere.
- ε(−k) = ε(k) to 0.0 on the Bloch-invariant test.
- Warm start: 2 sweeps vs 6 cold on the projected problem.
- ‖X·S·X − I‖ = 6.4e-6 (S^{-1/2} pipeline, f32 floor).
- alloc_count delta = 0 inside the repeated-solve loop.
