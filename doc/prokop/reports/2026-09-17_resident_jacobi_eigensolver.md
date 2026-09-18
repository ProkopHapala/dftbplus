---
type: Report
title: Resident-memory Jacobi eigensolver — local-A + deferred-V rotation log
tags: [gpu, jacobi, eigensolver, local-memory, bandwidth, t08]
timestamp: 2026-09-17
---

# Resident Jacobi Eigensolver (T08b)

Session report: implementation, measurement, and production integration of the
resident-memory Jacobi kernels proposed in
`tasts/HBond_Relaxed_Scan_GPU/Dense_Jacobi_Eigen_Tiling_Opt.md` (§540+).
Full measured tables: `tasts/HBond_Relaxed_Scan_GPU/Measured_Facts_Jacobi_Sweeps.md`.
Roadmap state: `tasts/HBond_Relaxed_Scan_GPU/Dense_Multi_GPU_Optimization.tasks.md` T08.

## Background — why this kernel exists

The previous streaming direct kernel `jacobi_cyclic_global_batched` keeps A and
V in global memory and re-streams *all* of both matrices every one of
~jround≈n−1 rounds per sweep. A roofline estimate (≈10 MB/sweep/system at n=86,
~20 GB per batch-400 solve ≈ 790 GB/s ≈ 85% of DRAM peak; 24 MB working set
≫ L2) diagnosed it as **bandwidth-bound on A/V streaming, not occupancy-bound**.
That explains the falsified "small WG for co-residency" hypothesis: WG512 wins
monotonically because more lanes = more memory-level parallelism — no WG/tile
resizing can fix a residency problem. The fix is keeping the high-reuse matrix
resident in `__local`.

## What was implemented

New kernel `jacobi_resident_batched` in `rust_dftb/src/qmqm/gpu_tiled_jacobi.cl`
(one workgroup per system, same contract as the direct kernel), with a compile
switch `RESIDENT_V`:

- **res-defV** (`RESIDENT_V=0`, the shipping variant): A loaded once into a
  dynamic `__local` array `lA[n·(n+1)]` (stride n+1 kills bank conflicts) and
  kept resident across **all** sweeps — every rotation parameter and every A
  update runs at local-memory bandwidth. V stays in global memory, but instead
  of the per-round update each rotation (c,s,p,q) is **logged** to a global
  `rotlog[batch][jround·jpair] double2` scratch and replayed against V **once
  per sweep** in an apply epoch — V's whole footprint (~30 KB at n=86) is
  L1-resident during that epoch, so global V traffic drops ~n×.
- **res-AV** (`RESIDENT_V=1`, experimental): V also `__local`
  (lA+lV ≈ 59.7 KB at n=86). Exceeds this device's 48 KB `local_mem_size` →
  gated by a host capacity check, untested here, kept for larger-local devices.

Semantics preserved: `active` mask, `diag[batch][4]` latch (off, off/‖A‖,
stop code, sweep count) with all stop codes incl. nonfinite=4, fused Fermi
tail reading the `lA` diagonal, warm `init_v` seeding from `c`, and the
residue-on-failure write-back contract (`lA` → `gA` once at exit, so
`extract_diag` and downstream paths are unchanged).

Rotation-order correctness: Jacobi pairs within a round are disjoint but do
not commute across rounds — the apply epoch replays rounds **sequentially in
logged order**, making the deferred product V·R₁·R₂… bit-identical to the
per-round update. Verified empirically: res/orth/eig-parity identical to the
direct kernel to 7 significant digits.

Rust side (`gpu_eigen.rs`): `tiled_render_source_cfg` gains `RESIDENT_V`,
`build_resident_jacobi_kernel` binds the bound-handle (tail scalars kept at
the same arg indices as the direct kernel so `bind_scc_params` is shared),
standalone `resident_jacobi_batched` runner for tests. `gpu_scc_plan.rs`
allocates the rotlog once (`[batch][jround·jpair] double2` ≈ 23 MB at
n=86/batch=400) and builds cold/warm resident handles.

Dispatch (`eigsolver_kind(n, local_mem)`, `RUST_DFTB_EIGSOLVER`):
`auto` → n>128 `block`; n≤128 → `ResidentDefV` when lA+16 KB headroom ≤
device local_mem else `direct`. Explicit `resident`/`resident_av` selections
that don't fit **panic** (fail-loud, not silent fallback).

## Measured results

Device: RTX-class OpenCL, `max_wg=1024`, **`local_mem_size=48 KB`** — this
caps residency at n≲96 (lA = 29.9 KB at n=86, 66 KB at n=128 fails).

`resident_jacobi_sweep`, N=86, batch=400, equal inputs — every config stop=0,
bad=0, and res/orth/par **identical to 7 digits** (same rotation math):

| kernel | WG | tail | "one" ms | "cold" ms |
|---|---:|---|---:|---:|
| direct | 256 | yes | 27.67 | 58.45 |
| direct | 512 | yes | 24.53 | 48.62 |
| direct | 512 | no | 21.92 | 45.67 |
| res-defV | 256 | yes | 11.50 | 24.03 |
| res-defV | 512 | yes | 12.61 | 23.52 |
| **res-defV** | 512 | no | **11.36** | **19.98** |
| res-AV | — | — | SKIP: 59.7 KB > 48 KB local cap | |

N=128: all resident variants skip (66 KB > 48 KB); direct-only regime.

**→ ~2.0–2.3× over the best streaming-direct config at equal accuracy.**
WG-insensitivity inside res-defV (256 ≈ 512, ±10%) is itself evidence the
kernel is no longer memory-level-parallelism-starved. `notail` now gives a
real ~10% (tail scratch costs occupancy once A is resident) — kept as an
env option, not the default.

End-to-end SCC (`gpu_scc_bench::test_gpu_scc_scan400`, no env overrides):

| system | solver (auto) | wall | iters | ms/iter | sys/s | failed |
|---|---|---:|---:|---:|---:|---:|
| GC N86 batch400 | res-defV | **218.7 ms** | 100 | **2.186** | 1829 | 0 |
| GC N86 batch400 | direct (previous) | 401.5 ms | 100 | 4.01 | — | 0 |
| DTH N246 batch400 | block B32 (unaffected) | 1745 ms | 24 | 72.7 | 229 | 0 |

GC same-work A/B: **4.01 → 2.19 ms/iter = 1.84× end-to-end** from the
eigensolver swap alone (diluted vs the standalone 2.2× because warm probes
skip most solves). Cumulative GC improvement vs the original baseline ≈ 8×;
DTH 6358 → 1745 ms ≈ 3.6×.

## Verification

- `test_resident_jacobi_parity` — residual/orthogonality/eigenvalue parity
  vs direct at n=86 and n=128-sized-where-fits. Pass.
- `gpu_tiled_jacobi` 9/9 non-ignored, `gpu_scc` 8/8, `gpu_dftb` 8/8,
  `gpu_forces` 5/5, `gpu_hbond_physics` 23/23 — all under new auto-dispatch
  (N=86 SCC tests exercise the resident path).
- Reproduce:
  `cargo test --release --test gpu_tiled_jacobi resident_jacobi_sweep -- --ignored --nocapture`
  and `RUST_DFTB_SK_DIR=<mio-1-1> RUST_DFTB_BENCH_SYSTEMS=GC cargo test
  --release --test gpu_scc_bench test_gpu_scc_scan400 -- --ignored --nocapture`.

## Open issues / next

- **res-AV untested** — needs a device with >60 KB local (a 99 KB-class part
  would fit n=86 A+V). Expected further gain (V traffic → zero during
  sweeps); env `resident_av` is ready.
- **n>128 residency** — the block kernel at n=246 is bandwidth-bound the same
  way (strips stream global per pivot) but A alone is 242 KB. The analogue
  is strip-residency inside `block_jacobi_1wg`, not full residency.
- **Scheduler coupling** — slot-pool design (`Slot_Pool_Scheduler.design.md`)
  must co-optimize slot count × WG × resident footprint; resident kernels
  shrink the per-slot local budget question to lA+scratch.
- ~~Rotlog is f64 (double2) per logged pair~~ → DONE 2026-09-18: prec-gated
  `jlog2_t` — float2 at prec=0 (production; c,s are f32 anyway), double2 at
  prec≥1 (accuracy reference). 23→11.5 MB buffer at n=86/b400.

## Addendum 2026-09-18 — packed-triangular lA

`lA` now stores only the lower triangle via `lat(r,c) = max(r,c)·(max+1)/2 +
min(r,c)` — 14.6 KB at n=86 (was 29.9 KB square). Invariants:

- Phase-2 iterates only pair-blocks `a≤b`: blocks (a,b) and transpose (b,a)
  map to the SAME packed slots, so the transpose block is skipped (it would
  rewrite identical values — `new M_ba = (new M_ab)ᵀ` — and race). This also
  halves the phase-2 block count vs square.
- `red2`/`dred2` scratch buffers removed — the reductions are strictly
  sequential, one buffer per dtype suffices.
- Measured `CL_KERNEL_LOCAL_MEM_SIZE` (incl. arg_local): **24 224 B/WG at
  WG512+tail → 2×24 224 = 48 448 ≤ 49 152 → 2 WG/SM now fits** (was hard
  1 WG/SM). res-defV capacity extended to n≤127.
- Verified: parity unchanged (par=1.73e-6, bad=0; resident sweep, work-ids
  subset, gpu_scc 8/8). Speed: ~5–9 % (WG512 notail one 10.67 ms, cold
  18.18 ms) — NOT 2×; the kernel is not occupancy-bound, V-replay traffic
  and barrier serialization dominate (see Measured_Facts §6).

## Files touched

`rust_dftb/src/qmqm/gpu_tiled_jacobi.cl` (kernel), `gpu_eigen.rs` (renderer,
runner, `EigKind`, dispatch), `gpu_scc_plan.rs` (rotlog buffer, bound
handles, dispatch), `tests/gpu_tiled_jacobi.rs` (parity + sweep),
`tasts/HBond_Relaxed_Scan_GPU/Measured_Facts_Jacobi_Sweeps.md` (§1b),
`Dense_Multi_GPU_Optimization.tasks.md` (T08b).
