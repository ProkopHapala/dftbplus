---
type: report
title: "GPU H-bond scan: Phase 4 analytic forces hand-off report"
tags: [hbond, gpu, forces, phase4]
timestamp: 2026-09-09
---

# HBond_Relaxed_Scan_GPU — Phase 4 hand-off report

> **2026-09-09 addendum (read this first).** The 1×4 `CL_OUT_OF_RESOURCES`
> crash below is **stale** — fixed by `vload2` in both force and assembly
> kernels. AT/GC `max|dH|~0.4` was **not** GPU assembly: CPU Neville
> `poly5_to_zero` exploded on the H–H tail (Hss −0.4 Ha at 10.39 Bohr).
>
> **Stopgap:** extra **zero function samples** on the right + phantom control
> `c_{-1}=2c_0−c_1` on the left. That is **not** the intended BC. Next
> interpolator work is a general extra-control fitter (solve for pad points
> so the valid-domain polynomial is preserved and V,V'→0 at cutoff). Do not
> re-Neville. Spec: `doc/prokop/topical_audit/sk_interpolation.md`.
>
> Honest tests: `rust_dftb/tests/gpu_hbond_physics.rs`. H2O GPU forces match
> CPU (rel ~3e-5). AT/GC GPU SCC rms plateaus `~1e-5` — **hypothesis: f32
> floor** (`~1e-8` relative × values `~100` → abs `~1e-6`–`1e-5`), not a
> proven mixer/Jacobi failure. Manifest §3.0.1. Do not edit sparse.

## Scope

This report documents the state of the `HBond_Relaxed_Scan_GPU` task, specifically
Phase 4 (analytic GPU forces). It is written for the next agent because the P4
work is currently blocked on a real-SK 1×4 s-p kernel crash.

---

## Phases 0–3: completed

`doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.manifest..md`
lists the phases. The work up to the end of Phase 3 was completed before this
session:

- **P0:** GPU harness cleanup (persistent `GpuSccPlan`, reusable buffers/kernels,
  event profiling).
- **P1/P2:** General tiled GEMM and one-WG tiled/block Jacobi for `N > 64`,
  tested on dimensions `63/64/65/87/88/96/97/128`.
- **P3:** Complete `N > 64` SCC: tiled Jacobi for `H'` and overlap eigensolves,
  `S^(-1/2)` reconstruction on GPU, device occupation/index selection, `P` and
  `W` built on GPU, AT/GC/azaindole SCC parity and fixed-geometry PES parity.

These phases deliver a working batched GPU SCC solver with validated CPU parity
for nucleobase-sized systems.

---

## Phase 4: analytic GPU forces — current state

### New files added

| File | Role |
|------|------|
| `rust_dftb/src/qmqm/gpu_forces.cl` | OpenCL force kernel `force_pairs` (non-SCC electronic part only) |
| `rust_dftb/src/qmqm/gpu_forces.rs` | Rust driver `GpuForceDriver` — compiles kernel, uploads batch, launches per-species-pair buckets |
| `rust_dftb/src/qmqm/mod.rs` | Registers `gpu_forces` module |
| `rust_dftb/tests/gpu_forces.rs` | GPU-vs-CPU force parity tests for synthetic H2/sp3 and real-SK H2O/formic-dimer |

### Sub-phases status

| Sub-phase | Status | Notes |
|-----------|--------|-------|
| P4a: inventory CPU force code | done | `rust_dftb/src/methods/dftb/forces.rs::non_scc_electronic_force` and `build_pair_block_with_derivs` documented |
| P4b: validate CPU analytic H/S derivatives vs finite differences | done | CPU analytic path already validated; rel_err ~1e-10 for H2, tilted H2, sp3 |
| P4c: port analytic derivatives to OpenCL | partial | Kernel implements s-s, s-p, p-p analytic derivatives; gamma and repulsive derivatives not ported |
| P4d: GPU-vs-CPU analytic force parity | partial | Synthetic H2/H2-tilted/sp3 pass; real-SK H2O crashes on 1×4 bucket |
| P4e: Newton-law test | implicit | Sum of forces is ~0 for passing synthetic tests; no standalone test added |

### What the kernel does

- Per `PairEntry` bucket (block_type 1×1, 1×4, 4×4), one work item reads
  `p.l, p.m, p.n, p.r`, interpolates SK values and radial derivatives from the
  compact per-species-pair SK table, forms the rotated `dH`/`dS` blocks in
  registers, and contracts them on the fly with `DM` and `EDM`.
- Contraction formula:
  `F_i[a] += 2 * ANG2BOHR * Σ_{μ∈i,ν∈j} (DM[μ,ν]·dH[μ,ν]/dR_a - EDM[μ,ν]·dS[μ,ν]/dR_a)`
  `F_j[a] -= same`
- Forces are accumulated into a global `float` buffer using a CAS-loop float
  atomic (`atomic_cmpxchg` on `unsigned int` reinterpretation) because OpenCL
  1.2 does not provide a float `atomic_add`.

### Bug fixes already made during this session

1. **OpenCL float atomic build error.** `atomic_add(&forces[...], f)` failed to
   compile. Replaced with `atomic_add_f32` using an `atomic_cmpxchg` CAS loop.
2. **4×4 p-p block missing angular term.** The pp-pp derivative initially only
   differentiated w.r.t. the row (atom j) direction cosine. Added the column
   (atom i) angular term; this fixed the `sp3` test (rel_err dropped to ~1.6e-3).
3. **SK table stride mismatch.** The kernel originally sampled channels with
   stride `SK_GRID_MAX`; the GPU SK tables are interleaved with stride
   `n_sk_cols`. Rewrote `interp_sk_2`, `interp_sk_5`, and derivative variants to
   use the interleaved layout.
4. **1×4 `float2` local alignment.** Changed `(__local float2*)tab` casts to
   `vload2(0, tab + 2*(base+k))` to avoid relying on 8-byte local memory alignment.

### Current blocker: real-SK H2O 1×4 bucket crashes

Command that reproduces the crash:

```bash
RUST_DFTB_SK_DIR=/home/prokop/git/SPAMMM/debug/AFM_CLI_FDBM/BAK/adenine-uracil/dftb_work_stock \
  cargo test --test gpu_forces test_gpu_force_parity_h2o -- --nocapture
```

Output:

```text
H2O: total_atoms=3 total_h=36 n_frags=1 pair_buckets=2
  bucket 0: block_type=0 n_pairs=1 n_grid=256 n_sk_cols=1 dr=0.043149605
  bucket 1: block_type=1 n_pairs=2 n_grid=256 n_sk_cols=2 dr=0.043149605
...
Status error code: CL_INVALID_COMMAND_QUEUE (-36)
```

Isolation (one `queue.finish()` per bucket) shows:

- **Bucket 0 (1×1 s-s, H–H):** returns OK.
- **Bucket 1 (1×4 s-p, two O–H pairs):** crashes with `CL_INVALID_COMMAND_QUEUE`.

`CL_INVALID_COMMAND_QUEUE` at `clFinish` means the `force_pairs` kernel crashed
during execution on the GPU. It is **not** a host-side compile error. The
synthetic `sp3` (4×4) and `H2` (1×1) tests pass, so the crash is triggered by
real SK data, not the s-p logic in isolation.

### Already verified

- OpenCL device: `NVIDIA CUDA` / `NVIDIA GeForce RTX 3090`.
- `n_grid=256` is within `SK_GRID_MAX=512`.
- H2O pair entries are correct (atom/orbital offsets, direction cosines, `r`).
- DM/EDM indices for the 1×4 block are within the 6×6 matrices.
- `r` values are ~1.87–2.87 Bohr, not near zero.

### Strongest remaining hypotheses

1. **The CAS-loop float atomic is not robust for the real 1×4 data.** The
   `atomic_cmpxchg` loop may hit alignment issues, live-lock, or race patterns
   when multiple O–H pairs write to the same atoms. Replacing it with a
   per-pair output buffer plus a separate reduction is the first recommended
   diagnostic step.
2. **A subtle out-of-bounds or uninitialized read in the real-SK 1×4 path.**
   Despite bounds checks, the `vload2`/`vload4` access or the B-spline control
   point conversion may still read past the valid range for `n_grid=256`.
3. **NaN/Inf produced by the interpolated SK values for O–H, causing later
   invalid operations.** The `.skf` files contain finite numbers, but the
   B-spline resampling step may produce problematic control points near the
   table head/tail.

---

## What the next agent should do

1. **Remove the atomic accumulation entirely.** Write per-pair `float3`
   contributions to a global `pair_forces[3 * n_pairs]` buffer, then reduce on
   the host or with a second simple kernel. This eliminates the CAS loop as a
   crash source and provides a clean diagnostic.
2. **Re-run the H2O test.** If it still crashes, the bug is in the 1×4
   derivative/interpolation path, not the atomics. If it passes, re-implement
   a safe reduction and re-enable full parity.
3. **Add standalone gamma and repulsive spline force contributions.** The current
   kernel only computes the non-SCC electronic term. `forces.rs` shows the
   full `F_total = F_nonSCC + F_SCC_shift + F_SCC_dc + F_rep` decomposition.
4. **Run full GPU-vs-CPU total-force parity** for H2O and formic-dimer once the
   non-SCC path is stable.
5. **Add explicit Newton's-third-law / torque tests** if not already covered.
6. **Remove `eprintln!` debug prints** added to `gpu_forces.rs` and
   `tests/gpu_forces.rs` once the work is validated.

---

## Links

- Manifest: `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.manifest..md`
- CPU reference: `rust_dftb/src/methods/dftb/forces.rs`
- CPU analytic derivatives: `rust_dftb/src/methods/dftb/rotation.rs`
- GPU kernel: `rust_dftb/src/qmqm/gpu_forces.cl`
- GPU driver: `rust_dftb/src/qmqm/gpu_forces.rs`
- GPU tests: `rust_dftb/tests/gpu_forces.rs`
- Related topical audit: `doc/prokop/topical_audit/gpu_scc_pipeline.md`
- SK interpolator SSOT: `doc/prokop/topical_audit/sk_interpolation.md`
- Honest physics tests: `rust_dftb/tests/gpu_hbond_physics.rs`
