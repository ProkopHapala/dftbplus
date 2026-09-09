---
type: review
title: "HBond_Relaxed_Scan_GPU — implementation audit + review protocol"
tags: [hbond, gpu, review, verification, dense-scc, forces]
timestamp: 2026-09-09
status: audit complete / interpolator stopgap measured 2026-09-09; B1 assembly crash is stale
---

# HBond_Relaxed_Scan_GPU — what is actually implemented, and how to review it

> **2026-09-09 — do not restart from B1.** The formic/H2O `assemble_pairs`
> `CL_OUT_OF_RESOURCES` (float2 cast) was fixed with `vload2`. AT/GC H/S
> mismatch of 0.4 Ha was CPU Neville tail explosion, not GPU assembly.
> Stopgap spline BC: blunt extra **zero samples** on the right, phantom
> `c_{-1}=2c_0−c_1` on the left. **Next interpolator work is not more
> zeros** — fit extra controls before/after so the valid-domain polynomial
> stays accurate. Spec:
> `doc/prokop/topical_audit/sk_interpolation.md`.
> Honest suite: `tests/gpu_hbond_physics.rs`. AT/GC GPU SCC rms `~1e-5` is a
> **f32-floor hypothesis** (manifest §3.0.1), not a proven mixer bug. Sparse
> = other agent.

This document has two halves:

- **Part A — Audit.** What the working tree actually contains, measured on the
  real GPU today, versus what `HBond_Relaxed_Scan_GPU.manifest..md` §11 and
  `.report.md` claim.
- **Part B — Review protocol.** The gates, commands, thresholds and
  "what counts as proof" rules by which this task should be reviewed from here
  on. This is the part that was requested; Part A exists because a review
  protocol that is not anchored to measured facts is just another checklist.

## How this audit was produced

| | |
|---|---|
| Repo state | working tree on top of `b269ab63` (uncommitted: +3226/−308 over 17 files) |
| Device | `NVIDIA GeForce RTX 3090`, OpenCL platform 0 (`NVIDIA CUDA`) |
| SK set | `RUST_DFTB_SK_DIR=/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1` |
| Build | `cargo test --no-run --tests` — clean, no errors |
| Suites run | `gpu_tiled_jacobi`, `gpu_scc`, `gpu_forces`, `gpu_hamiltonian`, `hbond_gpu_scc` |
| Mode | debug profile, `--test-threads=1 --nocapture`, unfiltered output |

Everything below marked *measured* is from a run today. Everything marked
*read* is from the source. Nothing here is confirmed as fixed — per
`AGENTS.md`, status stays "investigating" until the USER confirms.

---

# Part A — Audit

## A.0 Three blockers, in priority order

### B1 — The GPU H/S assembly kernel is broken for multi-species systems

`assemble_pairs` (`src/methods/dftb/dftb_hamiltonian.cl`) faults reproducibly
on the formic acid dimer, **at batch=1 as well as batch=41**:

```
cd rust_dftb && RUST_DFTB_SK_DIR=.../mio-1-1 \
  cargo test --test hbond_gpu_scc test_formic_dimer_single_point_scc -- --nocapture

thread 'test_formic_dimer_single_point_scc' panicked at tests/hbond_gpu_scc.rs:162:65:
  OpenCL error: clEnqueueNDRangeKernel("assemble_pairs")
  Status error code: CL_OUT_OF_RESOURCES (-5)
```

Same failure at `hbond_gpu_scc.rs:316` for the 41-point 1D scan. Meanwhile
`cargo test --test gpu_hamiltonian` passes all four tests with `max|dH|~1e-8`,
`max|dS|~1e-8` — but those cover only **H2 and N2**, i.e. one species and no
mixed `1×4` bucket.

So the failure is species/block-type dependent. **Leading hypothesis** (read,
not yet proven): `interp_sk_2` still contains the exact construct that the
Phase-4 report lists as bug fix #4 in the *force* kernel:

```205:213:/home/prokop/git/dftbplus/rust_dftb/src/methods/dftb/dftb_hamiltonian.cl
// 2-channel: float2 cast (1x4: ss, sp)
inline float2 interp_sk_2(__local const float* tab, int base, float4 w) {
    __local float2* tab2 = (__local float2*)tab;
    float2 v0 = tab2[base    ];
```

`gpu_forces.cl` replaced this with `vload2(0, tab + 2*(base+k))` to stop
relying on 8-byte `__local` alignment, and that fix resolved the reported H2O
`1×4` crash. The twin in `dftb_hamiltonian.cl` was never fixed. The `1×4`
bucket is exactly what a H/C/O system produces and what H2/N2 do not.

**Why this is blocker #1:** the formic dimer scan *is* the Phase-0 baseline
workload and the only test in the repo that exercises GPU-assembled H/S
feeding GPU SCC. While it faults, there is no verified path from coordinates to
energy anywhere in the dense pipeline.

### B2 — The harness converts GPU faults into green tests

The fault above poisons the CUDA context for the rest of the process. Every
subsequent `clCreateContext` returns `CL_NV_INVALID_MEM_ACCESS (-9999)`, and
the test helper interprets that as "no OpenCL device" and **skips**:

```
test test_formic_dimer_1d_scan_gpu_vs_cpu ... FAILED
test test_gc_scc_parity ... Skipping GPU test: no OpenCL device (... CL_NV_INVALID_MEM_ACCESS ...) ok
test result: FAILED. 3 passed; 1 failed
```

The two most important tests in the entire task — AT and GC parity at N≈86–87 —
were counted as **passed without executing**. Run in isolation, both actually
pass with good numbers (see A.2). The same skip-on-absence pattern makes the
whole SCC suite green with zero physics when the SK path is unset:

```
cargo test --test gpu_scc          # RUST_DFTB_SK_DIR unset
test test_gpu_scc_parity_n64_h2o_cluster ... Skipping: RUST_DFTB_SK_DIR not set ... ok
test result: ok. 6 passed; 0 failed
```

Additionally, `GpuRuntime::new()` uses `Platform::default()` / `Device::first`
and **no test asserts the device**. On this machine platform 0 is NVIDIA from a
normal shell, but PoCL CPU from a restricted/sandboxed shell (verified with
`clinfo -l` in both). PoCL hides precisely the bug classes this review flags:
missing global-memory fences, `l_frags[128]` out-of-bounds reads, atomic
races, and local-memory limits. Only one benchmark in `gpu_eigen.rs:401`
asserts `NVIDIA`.

This violates `AGENTS.md` "Fail Fast" more severely than any numerical issue in
the codebase, because it makes every other review statement untrustworthy.

### B3 — Nothing is verified end-to-end; the validated slice is narrow

Read from the tests: **every** passing SCC parity test uploads the CPU
reference's own matrices to the GPU.

```443:448:/home/prokop/git/dftbplus/rust_dftb/tests/hbond_gpu_scc.rs
    let h0_buf = rt.buffer_from_slice(&h0_flat).unwrap();
    let s_buf = rt.buffer_from_slice(&s_flat).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
```

`h0_flat`/`s_flat` come from `scc.h0`/`scc.s` (CPU), and γ from CPU
`build_gamma_matrix`. The same is true of all six tests in `tests/gpu_scc.rs`.
The force tests likewise feed CPU-computed DM/EDM and CPU-built
`GpuPairEntry{r,l,m,n}`.

So the verified GPU chain is only:

```
[CPU H0, S, γ] → SCC iteration → ε, q, E_el
[CPU DM, EDM, pair geometry] → non-SCC electronic force component
```

Untested seams: assembly→SCC (and currently broken, B1), SCC→forces,
the other three force components, E_rep, and anything device-resident.

## A.1 R1–R20 status: claimed vs verified

Legend: **✓** verified by measurement or unambiguous code read · **~** partially
done · **✗** not done · **!** claim in the manifest is stronger than the code.

| # | Item | Manifest claim | Audit verdict |
|---|---|---|---|
| R1 | Jacobi: don't zero pivot residual | FIXED | **~ !** Pivot store is correct (`gpu_tiled_jacobi.cl:280` stores `lA[i*PLD+j]`). But the kernel still zeroes the whole global off-diagonal unconditionally at exit (`:411–416`), on *every* exit path including sweep-cap and a stagnation heuristic. The manifest says this "only happens after convergence is established" — it does not. See N6. |
| R2 | Brent–Luk parallel inner pivot | FIXED | **✓** 32 pairs × 8 threads, all 256 lanes active (`:196–271`). Residual/parity improvements confirmed by measurement (A.2). Two efficiency gaps remain: fixed 20 inner sweeps with no early exit, and dummy tail lanes still scheduled. See N7. |
| R3 | Global memory fences | FIXED | **✓** `CLK_GLOBAL_MEM_FENCE` present at the three dependency boundaries (`:118`, `:296`, `:380`); the other 14 barriers are correctly local-only. |
| R4 | Restore strict test contracts | FIXED (Jacobi), pending (SCC) | **~** Jacobi asserts 1e-5/1e-5/1e-4 and *passes with 10–100× margin*. N>64 SCC still asserts **1e-2** in `gpu_scc.rs:513–515` and `hbond_gpu_scc.rs:464–466`, while measuring 3e-5–7e-5. See N12. |
| R5 | E_rep in the total energy | PARTIALLY FIXED | **~** Kernel + `set_repulsive_splines` exist; **no caller anywhere** in tests, examples or bins. Every energy this pipeline reports today, including the AT/GC parity numbers, is electronic-only. See N10. |
| R6 | All four force components | PARTIALLY FIXED | **~** All 4 kernels + 4 Rust drivers exist. **3 of 4 have zero test coverage**; no total-force test. See N11. |
| R7 | Energy/forces on the same charge state | PARTIALLY FIXED | **~** `finalize()` exists and `compute_energy` calls it. No force path calls it (`GpuSccPlan` is not even referenced from `gpu_forces.rs`). The inconsistency R7 was raised about — energy gradient parity — is therefore still open. |
| R8 | `GpuSccPlan` owns persistent kernels/buffers | FIXED | **~** 15 pre-built `Kernel` fields; no `Kernel::builder()` in `scc_step`/`finalize`/`compute_energy`. But `set_geometry()` still calls `build_inv_sqrt_batched`, which allocates fresh N² buffers and replaces `x_buf` — and `set_geometry` is the per-step hot path of a relaxation. See N13. |
| R9 | Occupation + DIIS on GPU | PARTIALLY FIXED | **✓** for both, better than claimed: occupation is a device bitonic sort, and `diis_step_batched` is a real GPU DIIS. One blocking `read_buffer` of the `[batch]` RMS vector per iteration remains, and `read_buffer` does a full `queue.finish()`. See N14. |
| R10 | Force driver: shared runtime, no per-bucket churn | PARTIALLY FIXED | **~** Shared `GpuRuntime` ✓, per-bucket `finish()` removed ✓. Still per-bucket: `Kernel::builder()` + SK table uploads inside the bucket loop. `gpu_repulsive_force_batched` calls `rt.build_program()` **and** `Kernel::builder()` on every call. |
| R11 | `l_frags[128]` batch>128 bug | unfixed | **✗** Present in `gpu_forces.cl:322` and `:480`, **and also in `dftb_hamiltonian.cl:491`** (plus `l_charges[256]`, `l_species[256]`, `l_u[64]` in the H1 kernel). Planned batches are 200–1000. |
| R12 | Replace atomic accumulation | unfixed | **✗** CAS-loop `atomic_add_f32`, 6 per pair, in all four force kernels. |
| R13 | Device-resident geometry | unfixed | **✗** and now *inconsistent*: `force_pairs`/`force_pairs_scc_shift` read host-precomputed `r,l,m,n` from `GpuPairEntry`, while `force_gamma_deriv_batched`/`force_repulsive_batched` take a device `coords` buffer. Half-migrated is worse than either end state. |
| R14 | RMS vs L2 norm | unfixed | **✗** `sqrt(Σ r²)` in both `residual_and_mix_batched` (`gpu_matrix_ops.cl:842`) and `diis_step_batched` (`:1295`). `tol=1e-5` is ~3× stricter for AT (30 atoms) than for H2O (3 atoms). |
| R15 | GEMM test tolerances + timing | unfixed | **✗** `tol = max(N²·1e-5, 1e-3)` → 0.164 at N=128 (`gpu_tiled_gemm.rs:95`). The "full-local vs tiled" test contains no timing at all. |
| R16 | Precompute `rsqrt(λ)` | unfixed | **✗** for the N>64 path: `scale_eigenvectors_batched` recomputes `rsqrt` per matrix *element* (`gpu_eigen.cl:472–478`). The N≤64 path does precompute (`:414–417`). |
| R17 | Don't hide invalid overlap | unfixed | **✗** `rsqrt(fmax(lam, LAMBDA_FLOOR=1e-7))` remains; `λ_min` is computed by the kernel and thrown away at both call sites (`let (x_buf, _lambda_min) = ...`). |
| R18 | Fuse Δq→V→H_SCC | unfixed | **✗** three separate kernel launches. |
| R19 | Build W only after convergence | unfixed | **✗** |
| R20 | Validate on real AT/azaindole, not 12×H2O | unfixed | **✓ done, better than claimed.** `tests/hbond_gpu_scc.rs` has `test_at_scc_parity` (N=87) and `test_gc_scc_parity` (N=86) and both pass with good numbers. The 12×H2O test also still exists. This item can be closed — but note B2: these two tests silently skip if any earlier test faults the GPU. |

## A.2 Measured numbers (RTX 3090, today)

Tiled Jacobi, random symmetric, batch=2, f32 storage:

| N | residual ‖AV−VΛ‖/‖A‖ | orthogonality ‖VᵀV−I‖/N | eigenvalue parity vs LAPACK |
|---|---|---|---|
| 65 | 1.3e-6 | 1.3e-7 | 3.4e-5 |
| 87 | 1.3e-6 | 1.4e-7 | 4.2e-5 |
| 96 | 1.4e-6 | 1.5e-7 | 4.2e-5 |
| 97 | 1.6e-6 | 1.7e-7 | 5.3e-5 |
| 128 | 1.7e-6 | 1.7e-7 | 7.6e-5 |

The solver is genuinely good: residual and orthogonality beat their 1e-5 targets
by 10–100×. Eigenvalue parity, however, **grows monotonically with N** against a
fixed 1e-4 assertion — 2.4× headroom at N=87, 1.3× at N=128 (N18).

SCC parity (GPU vs CPU, CPU-supplied H0/S/γ):

| System | N | asserted | measured \|dE\| | measured \|dq\| | measured \|dε\| | iters |
|---|---|---|---|---|---|---|
| H2O | 6 | 1e-3 | 1.6e-7 Ha | 2.9e-6 e | 8.3e-7 Ha | 24 |
| N2 | 8 | 1e-3 | 1.1e-6 Ha | 0 | 4.2e-7 Ha | 1 |
| 12×H2O | 72 | **1e-2** | 3.2e-5 Ha | 5.3e-6 e | 7.5e-6 Ha | **93** |
| **AT** | **87** | **1e-2** | **6.8e-5 Ha** | 1.9e-5 e | 2.0e-5 Ha | 26 |
| **GC** | **86** | **1e-2** | **7.2e-5 Ha** | 1.6e-5 e | 2.3e-5 Ha | 26 |

AT/GC already satisfy the manifest §6.2 acceptance limits (1e-4 Ha / 1e-3 e /
1e-4 Ha) with margin. The assertions are 150× looser than the achieved accuracy.

Non-SCC electronic force parity (CPU-supplied DM/EDM), asserted `rel_err < 5e-3`:

| Case | max\|F\| | max\|err\| | rel_err | Σ F |
|---|---|---|---|---|
| H2 / H2 tilted | 1.8e-1 | 5.5e-8 | 3.1e-7 | 0 |
| sp3 (1×4) | 6.9e-2 | 3.4e-8 | 4.9e-7 | 0 |
| **H2O (real SK, 1×4)** | 3.8e-1 | 1.5e-6 | **4.0e-6** | 8.5e-10 |
| formic dimer | — | — | passes | — |

**The blocker described in `.report.md` is resolved.** The real-SK H2O `1×4`
bucket no longer crashes and matches the CPU to 4e-6 relative. That report
section is stale and should be marked as such.

## A.3 New findings not in the GPT 5.6 review

**N6 — Unconditional off-diagonal zeroing + unreachable convergence criterion.**
`gpu_tiled_jacobi.cl:411–416` zeroes the global off-diagonal on every exit path.
The exits are: relative norm `< JACOBI_TOL = 1e-9`, a 3-strike stagnation
heuristic (`off_cur > 0.9f*prev_off`), or `MAX_SWEEPS = 50`. A **relative**
Frobenius ratio of 1e-9 is below what f32 accumulation can deliver, so the
designed criterion is effectively unreachable and the real exit is always the
stagnation heuristic or the sweep cap. The kernel returns no sweep count, no
final residual and no status, so the host cannot tell a converged system from a
stalled one — and the zeroing then makes the output *look* converged. R1 is
half-fixed: the local shortcut was removed, the global one remains.

**N7 — Inner pivot does ~20× more work than needed and rotates dummy lanes.**
`INNER_SWEEPS=20` × `JROUND_INNER=63` = 1260 rounds per pivot, unconditionally,
with 3 barriers each. There is no per-round check of the pivot's own
off-diagonal norm. Separately, `ipair < JPAIR_INNER` always schedules all 32
pairs over the full `PB=64`, so for N=87's tail pivot (m=55 active) the kernel
rotates 9 dummy lanes every round. The manifest claims active-lane scheduling;
the code does not do it.

**N8 — f64 in the hottest inner loop, on a GPU where f64 is 1/64 rate.**
`gpu_tiled_jacobi.cl:234–251` runs the 1024-block-per-round pivot update in
`double`. Rotation *parameters* in f64 is cheap and defensible (32 per round);
the block update is the innermost loop of the whole solver. This contradicts
manifest §4.10 verbatim ("do not assume FP64 is free on an RTX 3090-class
consumer GPU"). There is also no `#pragma OPENCL EXTENSION cl_khr_fp64 : enable`.

**N9 — Local memory at ~90% with no kernel-level query.** lA 16.6 kB + lU
16.6 kB + strip 8.2 kB + reduce 1 kB + rotation arrays ≈ **43.3 kB of 48 kB**.
Manifest §4.3 explicitly required querying `CL_DEVICE_LOCAL_MEM_SIZE` *and* the
built kernel's `CL_KERNEL_LOCAL_MEM_SIZE`; neither is queried. Consequence: the
B∈{24,32} × WG∈{256,512} sweep the manifest asks for is not runnable at
B=32/WG=512, and B=24 has never been measured. `CL_OUT_OF_RESOURCES` is the
error NVIDIA returns for local-memory overcommit, which makes this worth ruling
in or out while chasing B1.

**N12 — Tolerances 100–1000× looser than measured accuracy.** This is a
symmetric problem to the one GPT 5.6 raised: a 1e-2 assertion on a quantity
measured at 7e-5 will not catch a 100× regression. Tightening is free and is the
cheapest regression net available.

**N13 — Allocation in the relaxation hot loop.** `set_geometry()` →
`build_inv_sqrt_batched()` allocates `a_work`, `v_buf`, `v_scaled`,
`lambda_min_buf` (all N²·batch) and replaces `self.x_buf` on every call. For
Phase 5, `set_geometry` runs once per FIRE step. Direct violation of the
`AGENTS.md` hard rule. Needs a `build_inv_sqrt_into(&mut self, ...)` writing
into buffers owned by the plan.

**N14 — One full pipeline flush per SCC iteration.** `rt.read_buffer` ends with
`queue.finish()` (`gpu_runtime.rs:148–149`) and is called once per iteration to
fetch the `[batch]` RMS vector. For a many-small-system workload this
synchronization is the dominant cost, not the arithmetic.

**N17 — No event profiling in the production runtime.** `GpuRuntime`'s queue is
created without `CL_QUEUE_PROFILING_ENABLE`; OpenCL events are used only inside
a `gpu_eigen.rs` unit test. `SccStats` is host wall-clock. The Phase-0 box
"event-based profiling" is not satisfied for the pipeline, and no new baseline
report exists after the cleanup (newest is `2026-09-08_gpu_force_resolution.md`).

**N19 — The 72-orbital cluster SCC is numerically unhealthy.** Trace from the
run: RMS 3.0e-1 → **1.3e0** → 5.3e-1 → … → 3.5e-5 (iter 30) → **7.1e-2**
(iter 40) → 3.9e-4 (50) → 9.9e-8 (60) → **2.1e-4** (70), converging in 93
iterations, versus 26 for AT/GC with DIIS. The printed trace comes from
`src/qmqm/solver.rs:286` and the 93 is the GPU count, so attribution between
reference and GPU still has to be established. Either way, four-order-of-
magnitude non-monotonicity on a weakly-coupled cluster is the warning sign for
manifest §5.5 (SCC behaviour at zwitterionic proton-transfer points), and it
should be diagnosed before the relaxed scan depends on it.

---

# Part B — Review protocol

The organising principle: **this project's review problem is not "are the
numbers good" — the numbers that exist are good. It is "do the green checkmarks
mean anything".** Today they do not, for three independent reasons (B1/B2/B3).
So the gates below are ordered to fix observability first, then coverage, then
accuracy contracts, then performance. Do not review out of order; a performance
number from a harness that skips on error is noise.

## Gate 0 — Make the harness incapable of reporting a false pass

Nothing else in this document is reviewable until these hold. All five are
small.

| ID | Requirement | Accept when |
|---|---|---|
| G0.1 | Missing SK data is a hard failure, not a skip. Either resolve `RUST_DFTB_SK_DIR` to a repo default or `panic!` with the searched paths. | `cargo test --test gpu_scc` with the variable unset **fails**, and the message names the paths tried. |
| G0.2 | Device is asserted and banner-printed once per suite: name, vendor, `CL_DEVICE_LOCAL_MEM_SIZE`, compute units. Non-NVIDIA aborts unless `RUST_DFTB_ALLOW_CPU_CL=1` is set explicitly. | Test log begins with the banner; forcing `OCL_DEFAULT_PLATFORM_IDX=1` (PoCL) makes the suite fail with a clear message. |
| G0.3 | No OpenCL error is ever mapped to "skip". Absence of *any* device may skip; a device that errors must fail. | Grep shows no `Skipping GPU test` reachable from an `Err` carrying a `CL_*` status. |
| G0.4 | A memory-access fault aborts the whole test binary instead of letting later tests skip into green. | Injecting a fault into test 1 makes the run exit non-zero **and** report the remaining tests as not-run, never as passed. |
| G0.5 | One entry point: `scripts/run_gpu_review.sh` — fixed env, device banner, unfiltered `tee` to `debug/hbond_review/<date>/<suite>.log`, non-zero exit on any failure *or any skip*. | Running it produces one directory of logs an L1 reviewer can read top to bottom, and its exit code alone is a valid gate. |

Rationale is `AGENTS.md` §"Fail Fast, Fix the Physics" applied to the harness
itself. B2 is a live demonstration that this is not hypothetical: the AT and GC
tests were reported as passed while never running.

## Gate 1 — Fix and then properly cover the H/S assembly (unblocks everything)

**Step 1.1 — root-cause B1.** Confirm or refute the `interp_sk_2` alignment
hypothesis by porting the `vload2` fix from `gpu_forces.cl`. Rule out the
alternative (local-memory overcommit) by printing
`CL_KERNEL_LOCAL_MEM_SIZE` for the specialized `assemble_pairs` and comparing
to the device limit — this is required by N9 anyway. Report which it was; do not
"try things until it stops crashing".

**Step 1.2 — build the assembly test matrix that would have caught it.** The
existing coverage (H2, N2) misses every dimension along which the bug lives:

| Dimension | Must include |
|---|---|
| species count | 1, 2, 3 (H2 · H2O · formic dimer / AT) |
| block types | 1×1, 1×4, 4×1, 4×4 |
| SK pair ordering | homonuclear and **both** heteronuclear orderings (X-Y and Y-X tables) |
| batch | 1, 41, **200**, 500 — 200 and 500 are the R11 `l_frags[128]` probes |
| metric | `max|dH|`, `max|dS|` vs CPU `HamiltonianBuilder`, target < 1e-6 |

The batch>128 entries are the point: R11 is not a hypothetical, and a test at
batch=200 is the cheapest possible detector.

## Gate 2 — Tighten every contract to measured accuracy (free regression net)

Derive thresholds from A.2 with roughly 3× margin. Per `AGENTS.md`, a red test
here is a diagnostic to investigate, not a threshold to relax again.

| Test | Now | Proposed | Measured today |
|---|---|---|---|
| `gpu_scc.rs` N>64 cluster `|dE|` | 1e-2 | **1e-4** | 3.2e-5 |
| `gpu_scc.rs` N>64 cluster `|dq|` / `|dε|` | 1e-2 | **1e-4** | 5.3e-6 / 7.5e-6 |
| `hbond_gpu_scc.rs` AT/GC `|dE|` | 1e-2 | **2e-4** | 6.8e-5 / 7.2e-5 |
| `hbond_gpu_scc.rs` AT/GC `|dq|` / `|dε|` | 1e-2 | **1e-4** | 1.9e-5 / 2.3e-5 |
| `gpu_scc.rs` H2O/N2 `|dE|` | 1e-3 | **1e-5** | 1.6e-7 / 1.1e-6 |
| `gpu_forces.rs` rel_err | 5e-3 | **1e-4** | 4.0e-6 |
| `gpu_tiled_gemm.rs` | `N²·1e-5` (0.164 @128) | relative Frobenius **1e-5** | not reported |
| Jacobi eigenvalue parity | 1e-4 | keep 1e-4, **add trend assertion** | 3.4e-5 → 7.6e-5 (N18) |

Two additions rather than just tightenings:

- **GEMM must test the transpose combinations actually used** by the Löwdin
  transform and S^(−1/2) reconstruction, on physical H/S-sized matrices, not
  just square random ones (R15).
- **Jacobi eigenvalue parity must be tested at N=160 and N=200** to see whether
  the N-trend in A.2 is f32 accumulation in the strip updates (fixable) or
  inherent. Claiming meV barrier resolution requires knowing which.

## Gate 3 — Close the end-to-end seam (this is what makes Phase 4/5 reviewable)

B3 is the deepest structural gap: every component is validated against a CPU
reference that also *supplies its inputs*. Four tests close it, in order.

**G3.1 — E_rep parity (unblocks all energy claims).** Feed real SK spline data
through `set_repulsive_splines` and compare `E_rep` per system to the CPU
repulsive energy. Until this test exists, R5 must not be marked done and **no
PES produced by this pipeline is physically meaningful** — it is missing the
term that varies most as a proton moves between donor and acceptor.

**G3.2 — Per-component force parity.** One test per component (b) SCC shift,
(c) γ′, (d) repulsive, against the corresponding CPU term in
`methods/dftb/forces.rs`, then the **total** `F = F_nonSCC + F_shift + F_dc +
F_rep` against `compute_scc_forces`. Three of four components are currently
untested code (R6/N11).

**G3.3 — Energy-gradient consistency (the only test that catches R7).**
Central-difference `dE/dR` for 3–5 selected Cartesian components at ~1e-3 Å
against the analytic GPU force, relative agreement < 1e-3. Component-wise
parity against a CPU reference cannot detect that `E` and `F` were evaluated at
different charge states — only differentiating the pipeline's *own* energy can.
This is also the test that will tell you whether `finalize()` needs to be called
from the force path (it currently is not).

**G3.4 — Full chain from coordinates.** `coords → GPU H/S → GPU SCC → GPU P/W →
GPU forces → E_tot, F` versus CPU on identical coordinates, for AT and GC.
Assert energy, charges, forces, **and** ΣF ≈ 0 and translation invariance
(translation invariance has no test today; the Newton sum does).

Do not begin Phase 5 (constrained FIRE) before G3.4 is green. A relaxation
driven by forces whose relationship to the reported energy is unverified will
produce a PES that cannot be debugged.

## Gate 4 — Make the eigensolver report its own status

The solver's numbers are good; its *self-reporting* is the problem (N6).

- Return per-system `sweeps` and final `off_cur/off0` from the kernel. Assert
  convergence directly instead of inferring it from a post-hoc residual.
- Only zero the global off-diagonal when the convergence test actually passed.
  On non-convergence, fail loud with `(system_id, sweeps, off_cur/off0, N)` —
  per `AGENTS.md`, informative messages carry where, what, and the values.
- Replace `JACOBI_TOL = 1e-9` with a threshold reachable in f32 (`~1e-6`
  relative), so that the stagnation heuristic becomes a genuine fallback rather
  than the primary exit path.
- **Add the generalized-problem diagnostics, which do not exist anywhere
  today:** `‖HC − SCε‖_F/‖H‖_F` and `‖CᵀSC − I‖_F/N` at N=86/87. The standard
  eigenproblem residual is not the quantity DFTB depends on.
- Test adversarial and *physical* spectra, not just random symmetric matrices:
  clustered/near-degenerate eigenvalues, an overlap with `λ_min ~ 1e-5`, and the
  actual `H′`/`S` extracted from AT and GC.
- Report `λ_min` and `λ_min/λ_max` instead of discarding them, and fail on
  non-positive overlap rather than clipping with `LAMBDA_FLOOR` (R17).

## Gate 5 — Performance review (only meaningful after Gates 0–4)

Prerequisites: profiling-enabled queue in `GpuRuntime`, per-kernel event
timings, and a device banner in every log (N17, G0.2). **PoCL timings are not
GPU timings** and must never appear in a report.

Baseline to re-establish and store as a report under `doc/prokop/reports/`:
per-kernel GPU time for H/S assembly · S^(−1/2) · Löwdin GEMMs · Jacobi
(+ sweep counts) · occupation · P/W · mixing · forces, plus host wall time,
PCIe bytes, blocking-synchronization count, and systems/s versus batch, for
N = 28 / 72 / 87 at batch = 1, 10, 50, 100, 200, 500, 1000.

Then run these as measured experiments, each reported as before/after kernel
event time on the same device and batch:

| Experiment | Hypothesis | From |
|---|---|---|
| f64 → f32 in the pivot block update | large speedup, small parity cost | N8 |
| early exit on pivot off-diagonal norm | up to ~20× less inner work | N7 |
| skip dummy tail pairs | ~15% at N=87 | N7 |
| B=24 vs B=32 (needs the local-mem query first) | unknown; never measured | N9 |
| CAS atomics vs per-pair buffer + gather | atomics assumed cheap, unverified | R12 |
| fuse Δq→V→H_SCC into one WG/system kernel | 3 launches → 1 | R18 |
| build W only after convergence | removes N² work per iteration | R19 |
| RMS readback every k iterations | removes a pipeline flush per iteration | N14 |
| `build_inv_sqrt_into` preallocated | removes N²·batch allocation per FIRE step | N13 |

## Gate 6 — L2 human review artifacts

Per `AGENTS.md`, plots to `debug/`, reviewed by a human. The set that would make
the state of this task legible at a glance:

1. SCC residual traces (AT, GC, 12×H2O) — one line per system, log y. Makes N19
   immediately visible.
2. Jacobi eigenvalue parity and residual vs N (65…200) — makes N18 a trend, not
   an anecdote.
3. systems/s vs batch, one curve per N (28/72/87).
4. Stacked per-kernel time breakdown at batch=100.
5. **The physics money plot:** the 1D proton-transfer PES with and without
   E_rep, on the same axes. If those two curves have different barrier heights —
   and they will — that single figure justifies G3.1 better than any argument.

## What counts as proof (rules for whoever reports on this next)

These exist because the current manifest §11 contains `[*] FIXED` marks whose
code does not fully support them (R1, R4, R8), and a `.report.md` describing a
blocker that is in fact resolved. Both directions of drift are expensive.

1. **A "FIXED" claim requires the exact command and its unfiltered output**,
   run on the NVIDIA device, with the assertion at the *target* tolerance — not
   a loosened one, and not with the relevant test skipping.
2. **A performance claim requires before/after OpenCL event times** on the same
   device, batch and N. Wall-clock around a queue with a `finish()` in it is not
   a kernel time.
3. **Partial work is marked partial, and the remaining item is named.** `[~]`
   with "remaining: X" is worth more than `[*]`, and the audit above shows why.
4. **A red test stays red until the physics is understood.** If a threshold is
   genuinely wrong, change it and say so explicitly with the justification
   in the same commit message.
5. **Never mark fixed/done without USER confirmation** — a code change is not
   proof, and a green suite is not proof either while Gate 0 is open.

## Suggested order of work

1. **Gate 0** (harness honesty) — small, and everything downstream depends on it.
2. **B1 / Gate 1** (assembly fault + coverage matrix) — restores the baseline.
3. **Gate 2** (tighten tolerances) — free, and locks in what already works.
4. **G3.1** (E_rep parity) — the largest *physics* gap; every energy is currently
   incomplete.
5. **Gate 4** (eigensolver status reporting) — small, removes the last silent
   fallback in the numerics.
6. **G3.2–G3.4** (force components, gradient consistency, full chain) — the
   prerequisite for Phase 5.
7. **Gate 5** (performance) — with R11/R12/R13 fixed as part of it, since
   batch>128 correctness and device-resident geometry are both blocking for
   the relaxation workload anyway.

## Related documents

- `HBond_Relaxed_Scan_GPU.manifest..md` — task spec; §11 status marks need
  reconciling with Part A.1.
- `HBond_Relaxed_Scan_GPU.report.md` — Phase-4 hand-off; its "current blocker"
  section is **stale**, the H2O 1×4 force crash is resolved (A.2).
- `doc/prokop/reports/2026-09-08_gpu_force_resolution.md`
- `doc/prokop/topical_audit/gpu_scc_pipeline.md`
- `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md`
