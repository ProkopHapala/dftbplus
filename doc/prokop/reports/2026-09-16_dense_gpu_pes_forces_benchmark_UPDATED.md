---
type: report
title: "Dense-multi GPU: PES/force parity vs CPU f64 + 20×20-scan throughput benchmark"
tags: [gpu, dense-multi, f32, parity, pes, forces, benchmark]
timestamp: 2026-09-16
---

# Dense GPU solver — PES-shape/force parity and saturation benchmark

Hardware: NVIDIA RTX 3090 (82 CUs). Build: `--release`. SK set: mio-1-1.
Engine: production `GpuDftb` (`rust_dftb/src/qmqm/gpu_dftb.rs`), f32 bulk,
Fermi smearing kT = 0.002 Ha, GPU DIIS mix=0.

## Part 1 — PES shape and forces: GC proton-transfer scan vs CPU f64

The physically relevant accuracy question is not absolute energy but the
*shape* of the energy surface and the forces along it. Test:
`tests/gpu_hbond_physics.rs::test_gc_ptscan_pes_forces_vs_cpu`.

- System: guanine–cytosine base pair (29 atoms, N=86).
- Rigid scan: H13 displaced along the N8→N20 axis, d(N8–H13) = 1.00→1.90 Å,
  19 points — the same coordinate as `scripts/test_gpu_ptscan_gc.rhai`.
- GPU: one `GpuDftb` batch of 19 replicas, `scc(100, 1e-6)` — all converged
  in 32 batch iterations (per-replica rms 0.3–1.0e-6).
- CPU f64: per-point `DftbCpu`, `solve_scc(200, 1e-8)` + analytic
  `compute_forces` + spline repulsive term.

### Results

| metric | value |
|---|---|
| max \|E_gpu−E_cpu\| | 3.5e-6 Ha |
| **max \|ΔΔE\| (PES-shape error vs pt 0)** | **3.7e-6 Ha = 0.100 meV** |
| rms \|ΔΔE\| | 1.65e-6 Ha |
| rigid-scan barrier (39.70 kcal/mol) error | 0.068 meV |
| max \|ΔF\| (all Cartesian components, all atoms) | 5.5e-6 Ha/Å |
| max \|ΔF_scan\| (F(H13)·û) | 2.1e-6 Ha/Å |
| FD-vs-analytic consistency (h = 0.05 Å) | 4.3e-3 Ha/Å — **identical GPU and CPU** → pure O(h²) truncation, not solver noise |

Interpretation: the f32 error is *mostly* a smooth offset but **not**
constant — \|dE\| varies 1.4e-7→3.5e-6 along the scan, so ~0.1 meV of
genuine shape distortion exists. For a 40 kcal/mol proton-transfer
coordinate this is ~400× below chemical accuracy; the force profile
(including through the near-degenerate midpoint, HOMO/LUMO gap ~0.5 mHa)
tracks the f64 reference to ~5e-6 Ha/Å.

### Convergence diagnostics (the *right* way to handle tight-tolerance failure)

The CPU reference at tol=1e-9 "fails" at d=1.65 with last rms 1.66e-8.
Per-iteration trace (`debug/ptscan_conv.log`, plot `debug/ptscan_conv.png`)
shows all 19 trajectories contracting geometrically (15–32 iters), with the
mid-transfer point entering a **DIIS limit-cycle jitter around ~1e-8** —
a mixer floor, not slow convergence. Increasing `max_iter` only burns time
on the jitter; tol=1e-8 charge-rms is already ~100× tighter than the f32
quantities being measured. **Do not raise iteration counts to chase a
numerical floor** — plot the residual trajectory first.

Plots: `debug/ptscan_pes.png` (E_rel overlay + shape error + force profiles),
`debug/ptscan_conv.png` (all 19 CPU f64 trajectories + GPU final rms).

## Part 2 — 20×20-scan saturation benchmark (all tested systems)

Test: `tests/gpu_scc_bench.rs::test_gpu_scc_scan400_benchmark` (`--ignored`).
Each system is run as a batch of distinct scan geometries (two H-bond
proton-transfer coordinates, 20×20 grid → up to 400 replicas; smaller
batches tile the same grid prefix). CPU baseline = single-point `DftbCpu`
scc+eval timed on the same machine (sequential reference, not per-replica
parallel).

| system | atoms | orbs | batch | scc ms | iters | ms/iter | sys/s | evalF ms | CPU 1-pt ms | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H2O | 3 | 6 | 1 | 0.66 | 8 | 0.08 | 1525 | 0.26 | 0.46 | 0.7× |
| H2O | 3 | 6 | 400 | 1.17 | 8 | 0.15 | **342k** | 0.33 | 0.46 | 157× |
| formic dimer | 10 | 28 | 1 | 3.46 | 8 | 0.43 | 289 | 0.58 | 2.49 | 0.7× |
| formic dimer | 10 | 28 | 400 | 26.3 | 32 | 0.82 | **15.2k** | 1.82 | 2.49 | 38× |
| azaindole dimer | 30 | 84 | 1 | 12.8 | 16 | 0.80 | 78 | 1.18 | 12.2 | 0.9× |
| azaindole dimer | 30 | 84 | 400 | 218 | 56 | 3.9 | **1834** | 4.64 | 12.2 | 22× |
| guanine–cytosine | 29 | 86 | 1 | 17.3 | 16 | 1.08 | 58 | 0.60 | 12.9 | 0.7× |
| guanine–cytosine | 29 | 86 | 400 | 434 | 80 | 5.4 | **923** | 7.48 | 12.9 | 12× |
| adenine–thymine | 30 | 87 | 1 | 17.4 | 16 | 1.09 | 57 | 1.27 | 13.5 | 0.8× |
| adenine–thymine | 30 | 87 | 400 | 253 | 48 | 5.3 | **1578** | 7.78 | 13.5 | 21× |
| diazaphenalene dimer | 42 | 120 | 1 | 47.2 | 16 | 2.95 | 21 | 2.50 | 23.8 | 0.5× |
| diazaphenalene dimer | 42 | 120 | 400 | 838 | 40 | 21.0 | **477** | 17.5 | 23.8 | 11× |
| DiTetraceno-helicene | 84 | 246 | 1 | 283 | 16 | 17.7 | 3.5 | 6.0 | 164 | 0.6× |
| DiTetraceno-helicene | 84 | 246 | 400 | 7719 | 16 | 482 | **52** | 274 | 164 | 8.5× |

`iters` = batch iterations = convergence of the *slowest* replica (2D-scan
far corners are the hard points: GC needs 80, azaindole 56 vs 16–32 for
single points). `sys/s` counts one SCC+eval per replica; `evalF ms` is the
additional energy+force evaluation pass.

### Performance audit — the raw 8–38× numbers are not satisfactory, and the first interpretation was too weak

The first interpretation ("both GPU Jacobi and CPU LAPACK are O(N³)") is not
enough. The measurements contain much more diagnostic information. In
particular, the N≈84–87 results are almost a fingerprint of the current
**one-workgroup-per-matrix** Jacobi decomposition.

#### 2.1 The published `speedup` column is not yet a clean GPU-vs-one-CPU-core number

Two benchmark-definition issues must be fixed before treating the last column
as a hardware speedup:

1. **Sequential replicas ≠ single CPU thread.** `DftbCpu` calls LAPACK
   `dsyevd`, and the Rust crate links to the system OpenBLAS. The benchmark
   does not currently record or force the OpenBLAS thread count. Therefore
   the CPU reference is sequential *over geometries*, but the eigensolver may
   still use several CPU threads internally. A true one-core reference must
   explicitly run with `OPENBLAS_NUM_THREADS=1` (or equivalent runtime API)
   and record the effective BLAS thread count/library.
2. **The CPU and GPU rows do not represent the same SCC work.** The GPU
   batch contains 400 distinct scan points and runs until the slowest point
   finishes (e.g. GC 80 batch iterations, AZA 56, AT 48), whereas `CPU 1-pt
   ms` is one single geometry. Multiplying that one-point CPU time by 400 is
   therefore not the time required to solve the same 400-point grid
   sequentially. The active GPU mask also means that `max iterations × 400`
   is not the actual amount of GPU work.
3. The table describes the CPU reference as `scc+eval`, while the displayed
   speedup numerically uses the GPU `scc ms` column; `evalF ms` is reported
   separately. This is small for N≈85 but large for tiny systems and must be
   made apples-to-apples.

The required fair throughput benchmark is therefore: **the identical list of
400 geometries**, CPU sequential and GPU batched, same convergence criterion
and smearing, with CPU BLAS threads explicitly set to 1 (and separately to
"normal/full CPU" if desired). Report both SCC-only and SCC+energy+forces.

Also record

\[
N_{\rm active\ iter}=\sum_{\rm SCC\ iter} N_{\rm active},
\]

not only the maximum SCC iteration count. This is the natural unit of work
for the masked batch solver.

The current numbers can still be used diagnostically. Multiplying the raw
speedup by `batch_iters / batch1_iters` gives a **rough iteration-normalized
heuristic**, not a valid benchmark: formic ≈152×, AZA ≈77×, GC ≈60×,
AT ≈63×, diazaphenalene ≈28×, DTH ≈8.5×. The fact that these numbers
match the occupancy analysis below is informative, but they must not be
reported as measured speedups.

#### 2.2 N≈84–87 scales almost exactly as an ~82-slot machine

Define the batch-throughput gain relative to the batch-1 SCC iteration as

\[
G_{400} = {400\,t_{1}\over t_{400}},
\]

where `t1` and `t400` are the table's `ms/iter`. If one matrix occupies one
effective GPU execution slot and the RTX 3090 exposes about 82 such slots,
400 matrices should pass in about `ceil(400/82)=5` waves.

| system | N | batch-1 ms/iter | batch-400 ms/iter | `G400` | 82-slot prediction `5*t1` | measured / prediction |
|---|---:|---:|---:|---:|---:|---:|
| H2O | 6 | 0.08 | 0.15 | 213× | 0.40 ms | 0.38 |
| formic dimer | 28 | 0.43 | 0.82 | 210× | 2.15 ms | 0.38 |
| azaindole dimer | 84 | 0.80 | 3.9 | **82.1×** | 4.00 ms | **0.98** |
| guanine–cytosine | 86 | 1.08 | 5.4 | **80.0×** | 5.40 ms | **1.00** |
| adenine–thymine | 87 | 1.09 | 5.3 | **82.3×** | 5.45 ms | **0.97** |
| diazaphenalene dimer | 120 | 2.95 | 21.0 | 56.2× | 14.75 ms | 1.42 |
| DiTetraceno-helicene | 246 | 17.7 | 482 | 14.7× | 88.5 ms | **5.45** |

For N=84–87 the agreement is too close to ignore. It is strongly consistent
with the current Jacobi architecture: **one workgroup owns one matrix**. The
batch-400 job is not "400 matrices executing simultaneously"; it behaves much
more like roughly five waves of ~82 matrix workgroups. Exact workgroup
residency still needs to be confirmed from kernel resource/occupancy data, but
the wall-time scaling itself is already a strong clue.

This also explains why the raw result feels disappointing. At batch=1 the
GPU is not faster than the CPU for N≈85:

- AZA: CPU/GPU single-matrix ratio ≈ 12.2/12.8 = 0.95;
- GC: 12.9/17.3 = 0.75;
- AT: 13.5/17.4 = 0.78.

So, to first order, **all of the useful GPU speedup is currently coming from
running many matrices on different SMs, not from evaluating one matrix
efficiently on the GPU**. Multiplying those per-matrix ratios by the observed
~80–82-way batch gain gives ≈78× (AZA), 60× (GC), 64× (AT), almost exactly
the iteration-normalized heuristic above.

That is the key architectural diagnosis. The multi-system scheduler is
probably doing its job; the per-matrix solver is not exploiting the GPU
aggressively enough.

The N=120 and especially N=246 rows show a second failure mode: they scale
*worse* than the simple 82-slot wave model. N=246 takes ~5.45× longer than
that already-conservative prediction. This is a genuine large-N throughput
collapse, consistent with increasing global-memory traffic, synchronization,
register/resource pressure and the one-WG design.

Small N behaves differently: H2O/formic show >82× batch gain because batch-1
is dominated by launch/latency and small kernels can potentially co-reside;
they are not useful models for the N≈85 production regime.

#### 2.3 The dominant `jacobi` stage often performs **zero Jacobi sweeps**

This is the most suspicious fact found in the existing profiling notes.

For a distinct-256 production run, the stage profile reports approximately:

- `scc.jacobi`: **74.8% of device time**, ~1.84 ms/call;
- warm Jacobi launches: **mean 0.0 sweeps, p90=0**, initial relative
  off-diagonal norm ~4×10⁻⁷.

Earlier profiling found the same behavior: warm `BᵀHB` is already within the
Jacobi exit tolerance, so the common SCC iteration does no Jacobi rotations.
The kernel still has to execute its fixed path: off/Frobenius check, diagnostic
bookkeeping and the fused Fermi-occupation tail.

Therefore the statement "`jacobi` dominates" is currently misleading:
**the dominant cost in the common warm SCC iteration may be the machinery
around deciding that no diagonalization is needed.** That is exactly the kind
of overhead this project is supposed to eliminate.

The current source also gives a concrete hypothesis. The fused Fermi tail
can execute up to 24 safeguarded-Newton iterations and uses workgroup-wide
FP64 reductions for electron count/derivative. The labbook attributes much
of the remaining zero-sweep fixed cost after the R8c prologue fusion to this
tail plus diagnostics. An earlier A/B, however, claimed that adding the
Fermi tail cost only ~1 µs/call, so the evidence is not yet internally
consistent. This must be remeasured at **production batch=256/400**, with
normal event timing, by separating:

1. warm off-norm/check only;
2. Fermi occupation only;
3. actual Jacobi rotations.

Do not optimize rotation arithmetic until these three costs are known.

#### 2.4 High-priority hypotheses for the missing speed

**H1 — WG=512 was tuned for single-matrix latency, not batched throughput
(high confidence).**

The production objective is hundreds of simultaneous small matrices. A
workgroup size that minimizes batch=1 latency can be the wrong choice if it
reduces resident workgroups per SM. The near-82× scaling at N≈85 strongly
suggests only about one *effective* matrix workgroup per SM.

The important A/B is therefore **WG=128/256/512 at batch 82, 164, 328 and
400**, not another batch=1 benchmark. A WG=256 kernel may be somewhat slower
for one matrix yet 1.5–2× faster in production if two matrix workgroups can
reside/usefully overlap on one SM. Record achieved resident blocks/warps if
the profiler exposes them.

**H2 — the zero-sweep warm path should not pay for the full Jacobi kernel
(high confidence).**

If warm SCC is already diagonal to the requested f32 tolerance in almost
every iteration, invoking the heavyweight full Jacobi kernel is likely the
wrong granularity. The compiler still has to provision the registers/local
storage/control flow of the complete solver even on the zero-sweep branch.

A lightweight warm "check + diagonal/Fermi" path, with the full rotation
kernel activated only for systems that actually need a sweep, is a prime
architectural candidate. This should be evaluated as a throughput design,
not rejected merely because it adds one small launch: at batch=400 launch
latency can be much cheaper than reserving heavyweight Jacobi resources for
400 no-op diagonalizations.

**H3 — the fused Fermi tail may be over-parallelized and over-precise
(medium/high confidence; needs clean A/B).**

For N≈86 it solves one scalar chemical potential from only 86 eigenvalues.
A workgroup-wide FP64 reduction repeated up to 24 times can easily become a
barrier/FP64 latency machine. μ is warm-started, so the normal iteration
count should be measured; if it commonly approaches the cap, the solver or
stopping criterion is wrong for this use. Bulk occupations should remain
f32; only a tiny scalar correction, if demonstrably needed for parity, merits
f64.

Also check whether fusing Fermi scratch into the Jacobi kernel increases
local-memory/register pressure enough to reduce workgroup residency. Kernel
fusion is not automatically a win if it destroys occupancy.

**H4 — actual cold Jacobi rounds have poor memory locality (high confidence
for N≥120; secondary for zero-sweep N≈85).**

The direct kernel stores both `A` and `V` in global memory and applies every
Brent–Luk round there. The pair schedule gives scattered row/column accesses,
and every round has a workgroup synchronization boundary. This is much less
compute-dense than a blocked CPU eigensolver despite having the same formal
O(N³) scaling.

For N≈85 there is enough local memory to investigate much more aggressive
specialization. A full f32 A matrix is only ~30 kB at N=86; alternatively
the symmetric upper triangle is ~15 kB. Keeping A local while V remains
global, or packing A and reusing the local scratch, could eliminate most
round-by-round A traffic. Whether the complete local-A/V variant fits the
actual OpenCL per-workgroup local-memory limit must be measured, not assumed.
This is mainly relevant to the rare one-sweep/cold path; it will not fix a
zero-sweep overhead problem by itself.

**H5 — N≈246 should not use the same one-WG direct solver (very high
confidence).**

The measured 5.45× slowdown relative even to the simple 82-slot wave model
shows that the current direct one-WG/global-memory Jacobi has left its useful
regime by N≈246. A blocked/multi-WG eigensolver or the sparse path is not just
a future scalability nicety; it is required if this size matters.

**H6 — benchmark against a vendor batched eigensolver as a diagnostic ceiling
(high value, no requirement to adopt it).**

NVIDIA cuSOLVER provides batched Jacobi eigensolvers (`syevjBatched`, and the
Hermitian complex counterpart). A one-off benchmark with the same N,
batch and tolerance would answer an important question:

- if cuSOLVER is several times faster, the custom Jacobi implementation has
  substantial headroom;
- if it is similar, the limitation is more fundamental to this algorithm/
  matrix size and a different decomposition is needed.

This is a reference benchmark, not a proposal to make CUDA/cuSOLVER a
production dependency.

#### 2.5 What **not** to focus on first

- Forces are not the explanation for the poor SCC speedup here; `evalF` is
  small compared with SCC for N≈85.
- Do not spend effort shaving a few percent from Hscc/GEMM bookkeeping before
  resolving the zero-sweep Jacobi fixed cost and workgroup residency.
- Do not use more SCC iterations to hide convergence floors.
- Do not interpret "10k CUDA cores/threads" as 10k independent matrix
  processors. With one WG per matrix, the relevant question is how many
  matrix WGs can reside per SM and how efficiently one WG uses that SM.

### Revised scaling interpretation

- **N≈84–87 is already close to saturation for the *current*
  one-WG-per-matrix decomposition.** The ~80–82× batch-throughput factor is
  almost exactly the 82-SM count. This is not evidence that the GPU is fully
  exploited; it is evidence of the current architectural ceiling.
- **The per-matrix GPU implementation is the problem:** batch=1 N≈85 is
  roughly CPU-speed or slower. The present ~60–80× iteration-normalized
  throughput comes almost entirely from SM-level replication.
- **The reported 12–22× raw scan speedups at N≈85 are additionally depressed
  by an unfair CPU baseline / unequal convergence workload.** Rebenchmark the
  exact same 400 geometries with CPU BLAS threads pinned before quoting a
  hardware speedup.
- **N≥120 deteriorates beyond the one-WG/SM wave model; N=246 is clearly
  outside the efficient regime of the direct solver.**
- **The highest-value immediate investigation is the zero-sweep Jacobi path
  and batch-throughput WG occupancy**, followed by the actual cold-rotation
  kernel only after those fixed costs are removed.
- **Forces are comparatively cheap** in this benchmark and are not the first
  target for explaining the SCC throughput gap.

### Source/code facts used in this audit

- Current production direct Jacobi explicitly uses **one workgroup per
  system** and keeps A/V in global memory.
- `DftbCpu` diagonalizes through LAPACK `dsyevd`; the Rust build links
  `openblas-src` with the system OpenBLAS, and the repository does not
  explicitly pin OpenBLAS to one thread.
- Existing production profiling records warm H-Jacobi at zero sweeps for the
  overwhelming majority of launches, while the enclosing `scc.jacobi` stage
  remains the dominant device-time bucket.
- NVIDIA documents a batched symmetric/Hermitian Jacobi eigensolver
  (`syevjBatched`/`heevjBatched`), useful as an external performance control.

### Robustness note

In an earlier run of this benchmark the GC batch=100 pass had **one**
replica at the extreme far corner (d1=1.0, d2=1.95 — both protons
maximally asymmetric) fail to reach tol=1e-6 in 100 iters, plateauing at
rms~2.2e-4. In the present run all 400 GC replicas converged — this point
is *marginal*, not deterministically broken. The benchmark reports
`failed=N` per line rather than aborting; a non-zero count is a finding
about solver robustness on asymmetric PT geometries, not a benchmark bug.
The same corner is hard for the CPU f64 solver too (needs the most DIIS
iterations, limit-cycle floor ~1e-8). If this matters for production,
diagnose the residual trajectory — do not raise max_iter blindly.

## Test/inventory changes

- `tests/gpu_hbond_physics.rs::test_gc_ptscan_pes_forces_vs_cpu` (new;
  moved here from `gpu_dftb.rs` per repo convention — that file stays
  H2O-smoke only).
- `tests/gpu_scc_bench.rs::test_gpu_scc_scan400_benchmark` (new, `--ignored`):
  multi-system scan benchmark above + per-replica failure reporting.
- `debug/plot_ptscan_conv.py`, `debug/plot_ptscan_pes.py` — plot scripts;
  outputs `debug/ptscan_conv.png`, `debug/ptscan_pes.png`.

## SSOT updates

- `doc/prokop/topical_audit/f32_floor_dense_hbond.md` §3.2 — PES/force
  measurement added.
- `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` §6.4 —
  dated single-point table + PES/throughput lines.
- `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.report.md`
  — 2026-09-16 addendum (closes "real distinct-geometry batch ≥200 run").

## Open items

1. **Rebuild the benchmark baseline before quoting CPU speedup:** run the exact
   same 400 geometries sequentially on CPU; record OpenBLAS version/thread
   count; include an explicit `OPENBLAS_NUM_THREADS=1` result; compare the same
   SCC / energy / force work on both devices; record `Σ active replicas` over
   SCC iterations.
2. **Measure production-batch Jacobi occupancy/throughput:** A/B WG
   128/256/512 at batch 82/164/328/400. The existing batch-1 WG tuning is not
   sufficient for a multisolver.
3. **Decompose the zero-sweep `jacobi` stage:** off-norm/prologue vs Fermi tail
   vs real rotation work with event timing. Warm launches currently perform
   zero sweeps yet dominate device time in existing profiling; this is the
   highest-priority performance anomaly.
4. **A/B a lightweight warm fast path** versus always invoking the full direct
   Jacobi kernel. Measure total SCC throughput, not isolated launch latency.
5. **Use cuSOLVER `syevjBatched` as a diagnostic reference ceiling** at
   N=28/84/86/87/120/246 and production batch sizes. No production dependency
   implied.
6. **Investigate local-A / packed-A specialization around N≈80–100** for the
   rare cold/one-sweep path; current A/V-global round updates are structurally
   memory/synchronization heavy.
7. **N=246 requires a different path** if it matters: current one-WG direct
   Jacobi is ~5.45× slower than even the simple 82-slot wave prediction.
8. GC far-corner marginal nonconvergence — residual-trajectory diagnosis
   before any iteration-limit change.





---


# USER

OK, so what should we do ? How can we improve it? inst it still memory bound before we do not have sufficiently tiled the processing using local memeory? Do you think that using multi-workgroup approach for the jacobi solver would help it? Also the problem is that the system wait for the slowest replica (80 iterations) while other replicas converge in 16-30 and remain idle...

consider therefore imporvement both to the kernels (see attached code) and to the harness

---

analysis for my codeing agent:

Good question — let me look at the actual kernel structure for facts (workgroup size, shared memory, serialization), then do the arithmetic on the measured numbers.

One workgroup per replica, 256 threads, **A and V in global memory**, ~3 barriers per round — Jacobi is latency-bound, not FLOP-bound. And I see `JACOBI_PREC=2` (broad f64!) as the default — let me check what production actually compiles.

I have the facts now. Let me do the arithmetic and update the report.

**Key numbers (from measured table):**

- GC, batch=400: 5.42 ms/iter ÷ 400 = **13.5 µs/replica-iter** vs CPU 12.94 ms ÷ 16 iters = **0.81 ms/replica-iter** → GPU is already **~60× faster per replica-iteration**. The end-to-end 12× is eaten by *lockstep*: the batch pays the slowest replica's 80 iters while each CPU solve pays its own 16–32.
- DTH N=246: GPU 482 ms/iter ÷ 400 = 1.2 ms/replica-iter vs CPU 10.2 ms → only **~8.4×** — advantage collapses with N.

**Kernel facts (from `gpu_eigen.rs` / `gpu_tiled_jacobi.cl`):**
- One workgroup (512 threads) per replica — `jacobi_cyclic_global_batched`, A and V in **global memory**, ~3655 *serially dependent* rotations per sweep at N=86. So "10k threads" = 400 narrow latency-bound pipelines, not a wide SIMD op.
- Production JACOBI_PREC=0 (pure f32) — the f64 disaster mode isn't active.
- Warm-path prologue (V-init + off-norm) is ~58% of device time already (R8b).
- FLOP utilization: GC solve ≈ 80 iters × ~4·N³ flops × 400 replicas / 0.434 s ≈ **~190 GFLOPS ≈ 0.5% of the 3090's 35 TFLOPS** — completely latency-bound.

Let me update the report:

Report updated — `reports/2026-09-16_dense_gpu_pes_forces_benchmark.md` now has a "Why is 400-way GPU batching only 8–38× over sequential CPU?" section. The short answer:

**The 10k-threads framing is wrong — nothing here is compute-bound.**

Facts:
- Per replica-iteration the GPU is **already ~60× faster** than CPU (GC: 13.5 µs vs 0.81 ms/replica-iter). The end-to-end gap collapses because the batch runs **lockstep to the slowest replica** — 80 iters for GC vs 16–32 per CPU solve, a ~4× penalty.
- The eigensolver is `jacobi_cyclic_global_batched`: **one workgroup (≤512 threads) per replica**, A/V in global memory, ~3655 serially-dependent rotations/sweep at N=86. So the "10k threads" are actually 400 narrow latency-bound pipelines — that's why ms/iter scales linearly with batch: more replicas = more serial waves, not wider parallelism.
- Effective FLOPs ≈ **190 GFLOPS vs 35 TFLOPS peak — ~0.5% utilization**. Two orders of magnitude from compute-bound; it's all barriers, global-memory latency, and launch latency.
- The CPU baseline is LAPACK `dsyevd` (D&C + BLAS3, f64, cache-resident) — a much better O(N³) constant than cyclic Jacobi, so the per-iteration comparison isn't even algorithm-for-algorithm.
- R8b already measured it: with the warm basis, Jacobi exits in ~0–2 sweeps and the **fixed prologue (V-init + off-norm) is ~58% of device time** — the sweeps aren't the cost.
- JACOBI_PREC=0 in production — the f64 Jacobi disaster mode is not active.

Hypotheses ranked by payoff (all testable, none change numerics): **H1** warm-path prologue early-out (~2× device time), **H2** tail-aware batching so parked replicas stop consuming schedule, **H3** within-replica Jacobi parallelism (block/multi-WG or batched one-vector eigensolver) once batch×WG exceeds SM capacity, **H4** fuse the ~8–10 kernels/iteration chain, **H5** reframe: the honest metric is fraction-of-peak, and it says latency-bound.

Bottom line in the report: the batch engine is doing the right thing — the limit is the sequential-per-replica eigensolver plus kernel-chain latency. More batch doesn't fix it; the lever is the warm-path early-out and within-replica parallelism.

One flag stands from before: the lib doesn't compile right now — your in-progress `sparse_dftb.rs` calls `ws.trace_ks_to_f64()`/`ws.ri_idempotency()` which don't exist yet (I saw you renamed to `measure_projector_state` — the call site at sparse_dftb.rs:1402 still uses the old names).


---


# Chat GPT 5.6 sol

Yes. I think we now have a fairly clear direction, and it is **not simply “add more threads.”**

There are two different bottlenecks hiding under the word “Jacobi”:

* **Warm SCC, which is your common case:** often zero Jacobi sweeps. This is currently dominated by fixed reduction/Fermi machinery and synchronization.
* **Actual Jacobi sweeps:** the present direct cyclic implementation really is badly memory-structured. A and V are repeatedly streamed from global memory through ~N serial Jacobi rounds.

And separately there is a **batch scheduler/tail problem**.

### 1. Is the actual Jacobi sweep memory-bound?

Yes, very likely.

For N=86, the direct kernel has 43 independent rotations per round and **85 serial rounds per sweep**. It is not “3655 serial rotations”; it is 85 dependent rounds, each containing 43 parallel rotations. In each round the kernel updates all pair-pair blocks of A and all corresponding columns of V directly in global memory. 

A rough traffic count for N=86 is:

$$
43^2 \times (4\ {\rm reads}+4\ {\rm writes})\times4
+
86\times43\times(2\ {\rm reads}+2\ {\rm writes})\times4
\approx118\,{\rm kB/round}.
$$

With 85 rounds:

$$
\approx 10\ {\rm MB/system/sweep}.
$$

For N=246 it becomes about **226 MiB/system/sweep**.

The arithmetic intensity is only roughly **0.5–0.7 FLOP/byte**. Even with an ideal ~900 GB/s memory system, such a kernel has a roofline of only around 0.5 TFLOP/s. So comparing it with the 3090's ~35 TFLOP/s FP32 peak is misleading: this formulation can never get remotely close to that peak.

This is exactly where your intuition about local memory/tiled processing is correct.

But there is a crucial qualification:

> **Your common warm zero-sweep case is not memory-bandwidth-bound.**

The zero-sweep prologue only reads approximately 30 kB/system at N=86. Even 400 systems are only ~12 MB total, which is negligible for the 3090 memory system. Yet the measured zero-sweep “Jacobi” costs milliseconds.

The culprit is visible directly in the kernel: after the norm check, the same huge WG executes the Fermi tail, with local `double dred[WG]`, `double dred2[WG]`, min/max reductions and up to **24 Newton iterations**, each involving WG-wide barriers. 

So I would attack the two cases separately.

## What I would do, in order

1. **Split Fermi out of Jacobi immediately.**

   The previous idea of fusing Fermi into Jacobi sounded attractive because it removed a kernel launch. Looking at the actual code, I now think that fusion is counterproductive.

   Jacobi wants perhaps 256–512 threads. Fermi for 86 eigenvalues wants approximately **one warp = 32 threads/system**, perhaps 64. Fusion forces the tiny Fermi problem to use the Jacobi WG size and also permanently adds `le[256]`, `dred[WG]`, `dred2[WG]` and `lmu[]` to the Jacobi kernel's local-resource footprint. 

   Make a small `fermi_post_batched`, ideally one 32-thread subgroup/system on NVIDIA. Each lane handles `k = lane, lane+32, ...`. Use subgroup reductions if OpenCL exposes them; otherwise a WG32 local reduction is still only five reduction stages.

   Keep **μ and the decision in f64**, but compute the logistic function from the f32 eigenvalue/kT in **f32** and cast its contribution into f64 for the sum. Doing `exp(double)` on a 3090 gains essentially no input information because ε and kT were f32 to begin with.

   Also record the actual Newton iteration distribution. I strongly suspect a warm-started μ usually needs perhaps 1–4 iterations, not 24.

   This gives a useful general rule for the project:

   > **Do not fuse kernels whose natural parallel granularity differs by 16×. Kernel launch elimination is not worth turning an 86-number scalar reduction into a 512-thread synchronization problem.**

2. **Make the warm zero-sweep path truly cheap.**

   After the two warm GEMMs form

   $$
   A=C^T H C,
   $$

   run a small `jacobi_probe` that computes `||offdiag A||/||A||`. If it is already below tolerance, **do nothing to C** and take `diag(A)` as the eigenvalues. That is exactly what the current Jacobi kernel eventually decides anyway.

   The probe should contain no Fermi code and no huge scratch arrays.

   Initially I would use perhaps WG128 for N≈86 and WG256 for N≈246. Later, if this probe remains noticeable, fuse the norm calculation into the second GEMM: each GEMM output tile can write `{sum A², sum offdiag²}` to a small partial buffer, followed by one tiny per-system reduction. The current GEMM is already properly tiled in local memory and uses coalesced transpose loads and four independent FMA accumulators. 

   I would **not** complicate GEMM with that until the standalone probe is actually measured.

3. **Rebenchmark WG=128/256/512 using batch=400, not batch=1.**

   The current Rust harness chose WG512 because it beat 256 for a **single direct eigensolve**, and the comment then effectively assumes batching does not change that conclusion. That is precisely the wrong objective for this application.

   We care about throughput at batch≈400 and, even more importantly, at tail populations like 80, 40, 20, 5.

   Benchmark this matrix:

   |  WG |         active replicas |
   | --: | ----------------------: |
   | 128 | 400, 160, 80, 40, 20, 1 |
   | 256 | 400, 160, 80, 40, 20, 1 |
   | 512 | 400, 160, 80, 40, 20, 1 |

   And run two matrix types separately: near-diagonal **0-sweep** and forced **1-sweep**.

   I would not be surprised if WG512 remains best latency for one matrix while WG256 wins production throughput.

4. **Implement logical batch compaction, not physical matrix compaction.**

   This is the important harness change.

   Your current code is already better than the agent's description: DIIS clears `active[sid]` on the GPU as soon as a replica converges, and subsequent kernels early-return for it. The SCC loop already runs in chunks of 8 without host synchronization.  The GEMM gate also occurs before loading tiles. 

   Therefore the statement

   > “all 400 systems execute 80 heavy iterations”

   is **not true anymore**.

   But the launch geometry is still for all 400. For a 6×6 GEMM tile grid, 20 living systems plus 380 dead systems still dispatch roughly 14,400 WGs, most of which exist only to inspect `active[sid]` and return.

   Add persistent:

   ```text
   active_ids[batch]
   n_active
   ```

   At the existing chunk-end status read, compact the IDs on the host into the already allocated array and upload only the prefix. No new synchronization is necessary because you already synchronized there.

   Kernels become conceptually:

   ```text
   slot = get_group_id(...);
   sid  = active_ids[slot];
   ```

   and are launched with `n_active`, not `batch`.

   For GEMM:

   ```text
   sid = active_ids[get_group_id(2)]
   ```

   For one-WG/system kernels:

   ```text
   sid = active_ids[get_group_id(0)]
   ```

   For `(system,column)` kernels:

   ```text
   slot = group_id / n
   k    = group_id % n
   sid  = active_ids[slot]
   ```

   **Do not move C/H/D matrices around.** This is purely an indirection table.

   Within an 8-iteration chunk, newly converged replicas can still clear `active[sid]` and early-out exactly as now. At the next chunk boundary they disappear from `active_ids`.

5. **Record the active population curve before attributing the 12× result to “lockstep.”**

   What we actually need is

   $$
   N_\mathrm{work}=\sum_i n_i
   $$

   rather than `batch × max_iterations`.

   At each chunk boundary record:

   ```text
   iteration    n_active
        0          400
        8          397
       16          312
       24          141
       32           47
       40           18
       ...
       80            1
   ```

   Then integrate it.

   That tells us whether SCC divergence is costing 10%, 2× or 4×. Right now we do not actually know.

   The earlier “60× per replica-iteration” number is also slightly misleading: dividing a batched wall time by 400 gives a **throughput-normalized system-iteration cost**, not the latency of one GPU replica.

6. **For actual rotations, build a real multi-WG BLOCK Jacobi. Do not make the current Brent–Luk round multi-WG.**

   This distinction is critical.

   A naive multi-WG version of the current cyclic solver is ugly because round \(r+1\) depends on the whole matrix after round \(r\). OpenCL gives no safe grid-wide barrier inside a normal kernel. Splitting 85 rounds into 85–170 kernel launches per sweep would be terrible.

   Instead use **parallel block Jacobi**.

   Partition orbitals into B=16–32 blocks. In one block phase, pair blocks using a round-robin matching. All block pairs in that phase are disjoint.

   For each phase:

   ```text
   pivot kernel:
       one WG / (replica, block-pair)
       load 2B×2B pivot to local
       diagonalize locally
       write U_pair

   transform kernel:
       many WGs / replica
       each WG owns one compound A tile
       A_PQ <- U_P^T A_PQ U_Q
       local tile + local U reuse

   V transform:
       many WGs / replica
       V_P <- V_P U_P
   ```

   The transform kernels are naturally tiled GEMM-like operations.

   The important thing is that one **block phase** replaces dozens of scalar Jacobi rounds.

   For N=86 and B≈24–32 you have only 3–4 blocks, hence roughly **3 block phases per sweep**.

   Compare the memory traffic estimates:

   $$
   \text{direct cyclic N=86}\approx 10\,\mathrm{MB/sweep/system}
   $$

   while a block formulation that reads+writes A and V approximately once per phase is roughly

   $$
   3\times16N^2\approx0.35\,\mathrm{MB/sweep/system}.
   $$

   That is almost **30× less global traffic** before considering cache behavior.

   For N=246 the direct estimate is ~226 MiB/sweep. With eight B=32 blocks, seven block phases are only of order **6–7 MiB/sweep**—again roughly 30× less traffic.

   Those are approximate roofline numbers, not predicted speedups, but they explain why N=246 collapses so badly today.

7. **Do not resurrect the deprecated one-WG tiled kernel unchanged.**

   The old tiled implementation had the right idea—local pivot + U—but executed all the strip work inside one WG and accumulated enormous barrier depth. Its own comments document that the direct kernel replaced a ~16k-barrier eigensolve. 

   What I am proposing is different:

   > local block algebra + **many WGs for the global tile transforms** + kernel boundary as the block-phase global synchronization.

   That is where multi-WG makes sense.

8. **As a quick experimental proof of the memory hypothesis, try A-local / V-global for N≤96.**

   This is much simpler than multi-WG block Jacobi and could be a useful intermediate benchmark.

   At N=88 padded, full A is only about

   $$
   88\times89\times4 \approx31\,\mathrm{kB}.
   $$

   So keep A entirely in local memory across all Jacobi rounds, while V remains global.

   That immediately eliminates roughly half the direct solver's round-by-round global traffic and makes the convergence norm local.

   It will not solve N=246 and it may reduce WG residency because of ~30–40 kB local memory/WG, so I would treat it as a **performance probe first**, not automatically another permanent solver.

   If it gives 1.5–2× on a forced N=86 sweep, we have direct empirical confirmation that A traffic is killing us.

   Also query the actual NVIDIA OpenCL local-memory limit. If it is sufficiently above 48 kB, a packed-symmetric A plus local V becomes interesting; for N≈88 packed A is only ~15 kB.

9. **Use multi-WG particularly for the convergence tail.**

   This is where your two concerns meet.

   With 400 active replicas, one-WG/system already exposes plenty of WGs. Multi-WG does not help merely by creating “more threads”; its advantage must come from better block locality.

   But once only 20 hard replicas remain, one-WG/system Jacobi gives only 20 useful Jacobi WGs and leaves most of the GPU empty.

   A block Jacobi producing, say, 10–30 WGs per active matrix can fill the GPU again.

   So an eventual adaptive policy could naturally be:

   ```text
   warm projected matrix
       |
       +-- offdiag already small --> no Jacobi; tiny Fermi-post
       |
       +-- needs rotations
               |
               +-- block/multi-WG Jacobi
   ```

   I would first benchmark the block solver universally rather than hard-code an active-count threshold.

10. **Fix the remaining harness allocations while touching this code.**

They are still there.

Current `set_coords()` does:

```rust
self.scc_ok = vec![true; self.batch];
```

and `scc_mix_inner()` allocates `hist`, `stagnant`, `done`, `samples`, `q_in`, `q_out`, `res`, `q_f32`, plus clones the Jacobi mask.

These should all be persistent plan/engine scratch. They probably do not explain a factor 10, but they violate the architecture rule we explicitly established and there is no reason to leave them.

---

### What I think the likely winning architecture looks like

For the ordinary N≈86 SCC iteration:

```text
active_ids: only live replicas

1  fused dq -> V -> Hscc
2  GEMM temp = C^T H
3  GEMM A    = temp C

4  cheap eig-probe
      offdiag small?
          YES: C unchanged
          NO : block-Jacobi on only those systems

5  small WG32 Fermi + occupation
6  density/Mulliken/DIIS
```

The really common path therefore contains **no eigensolver sweep whatsoever**.

And when Jacobi is genuinely needed, we stop streaming A/V through 85 scalar rotation rounds and instead transform block tiles with local-memory reuse.

### A particularly important correction to the coding-agent analysis

I would delete these two conclusions from its current text:

> “the end-to-end 12× is eaten by lockstep because all replicas pay 80 iterations”

and

> “~190 GFLOPS proves Jacobi is latency-bound.”

The first is outdated because your device `active` mask already stops expensive work for converged replicas inside the chunk. The remaining problem is full-grid dispatch plus low active occupancy in the tail.

The second counts \(4N^3\) Jacobi work even though the profiler says most warm Jacobi launches perform **zero sweeps**. You cannot derive meaningful achieved GFLOPS by counting FLOPs the GPU did not execute.

For actual nonzero sweeps, the more informative metric is the **~0.6 FLOP/byte arithmetic intensity**, which says directly that the current formulation is bandwidth-bound.

---

Here is the condensed instruction I would give the coding agent:

The current performance problem must be attacked as three separate problems: warm-path Jacobi overhead, real-sweep Jacobi memory traffic, and shrinking-batch tail utilization.

Do NOT assume that batch=400 executes max_iter heavy iterations for every replica. The current GPU DIIS already clears active[sid] device-side and all production SCC kernels gate on active. First instrument n_active after every SCC chunk and report the active-population curve and sum of effective system-iterations.

P0 — isolate and remove the zero-sweep cost.

The production warm H-Jacobi commonly reports zero sweeps, yet jacobi remains the dominant timed stage. Split the Fermi occupation solver out of jacobi_cyclic_global_batched. The present fused tail is an architectural mismatch: Jacobi uses WG≈512, while Fermi on N≈86 naturally requires one subgroup/WG32 or WG64. It currently carries WG-sized f64 scratch and performs repeated WG-wide reductions.

Implement a small standalone Fermi post-kernel:

* one WG32 initially on NVIDIA;
* each lane handles several eigenvalues;
* persistent previous μ as Newton initial guess;
* evaluate f(x) in f32, accumulate electron count / derivative and make μ decisions in f64;
* record Newton iteration count;
* keep safeguarded bracket fallback;
* write occ_w, μ and any occupation mask/index needed downstream in the same kernel.
  Do not add host synchronization.

After removing Fermi from Jacobi, benchmark the pure zero-sweep Jacobi/probe cost separately.

P1 — benchmark workgroup size for production throughput.

The current WG512 choice was based on batch=1 latency. That is not a valid production tuning criterion. Benchmark WG={128,256,512} for active replica counts {400,160,80,40,20,1}, separately for:
(a) a near-diagonal zero-sweep matrix;
(b) a matrix forced to require at least one sweep.

Report kernel event time and throughput in system-solves/s. Do not select a WG from batch=1 alone.

P2 — logical active-list compaction.

Add persistent active_ids[batch] and n_active. Do NOT physically compact/copy N² matrices.

At the existing chunk-end readback, build active_ids from active_host using preallocated host scratch and upload the active prefix. Subsequent kernels launch only n_active logical replicas and map slot -> original sid through active_ids.

Examples:

* one-WG/system: sid = active_ids[get_group_id(0)]
* tiled GEMM: sid = active_ids[get_group_id(2)]
* system×column kernels: slot=group/n; sid=active_ids[slot].

Within a chunk, DIIS may still set active[sid]=0 immediately; kernels must retain the active test. At the next chunk boundary, remove those replicas from active_ids.

Do not rebuild kernels or allocate buffers when n_active changes. Override global work size at enqueue using persistent kernel handles.

Profile wall time against active fraction. In particular report cost at n_active={400,80,40,20,5,1}.

P3 — diagnose real-sweep memory traffic.

The present direct cyclic Jacobi stores both A and V in global memory and revisits them every Jacobi round. For N=86 there are 85 serial rounds/sweep; straightforward traffic is ~10 MB/system/sweep. For N=246 it is ~226 MiB/system/sweep. Arithmetic intensity is only ~0.6 FLOP/byte.

As a low-effort diagnostic, prototype A-local/V-global for N<=~96:

* load full padded A into local once;
* keep A local through all rounds/sweeps;
* V remains global;
* convergence norm is computed from local A;
* Fermi must remain outside this kernel.
  Benchmark only; do not automatically make it a permanent third solver.

P4 — implement a genuinely multi-WG block Jacobi for the real sweep path.

Do NOT attempt a multi-WG version of each Brent-Luk scalar round. There is no safe portable inter-WG barrier inside an OpenCL kernel, and one kernel launch per scalar round is forbidden.

Instead implement parallel block Jacobi.

Partition the matrix into B×B blocks, benchmark B={16,24,32}. For each round-robin block phase, pair blocks disjointly.

Phase A: pivot kernel

* one WG per (active replica, block pair);
* load the compound 2B×2B diagonal pivot to local memory;
* diagonalize with the existing full-local Brent-Luk machinery;
* write the compound U for that block pair.

Phase B: matrix transform

* many WGs per active replica;
* each WG owns one compound A tile;
* load tile and required U blocks into local memory;
* compute A_PQ <- U_P^T A_PQ U_Q;
* write each output tile exactly once;
* exploit symmetry where useful;
* no global atomics.

Phase C: eigenvector transform

* many WGs per active replica;
* V_P <- V_P U_P using tiled/local-memory reuse.

Kernel boundaries between block phases provide the required global synchronization. There should be O(number_of_blocks) global phases per block sweep, NOT O(N) scalar-round launches.

Accumulate off-diagonal norm as per-tile partials and reduce once per block sweep. No global floating-point atomics.

For N≈86, B≈24–32 gives only ~3 block phases/sweep. For N=246, B=32 gives ~7 phases. This should reduce global A/V traffic by roughly one to two orders of magnitude relative to the scalar-round direct kernel while exposing many WGs per active matrix.

Do not resurrect tiled_jacobi_batched unchanged; its one-WG strip processing and barrier depth were the problem.

P5 — tail behavior.

After logical compaction, explicitly profile when n_active falls below the number of SMs. This is where one-WG/system Jacobi necessarily underutilizes the GPU. The multi-WG block solver should be evaluated specifically at n_active={1,5,20,40}. It may be especially valuable for the hard convergence tail even if the simpler path remains competitive at n_active=400.

P6 — preserve the architecture invariants.

No global atomics.
No host readback per SCC iteration.
No kernel/program construction in SCC/FIRE loops.
No GPU-buffer allocation in SCC/FIRE loops.
No physical matrix repacking merely to compact the active batch.
No global spin-barriers between workgroups.
No silent CPU fallback.
No one-kernel-per-Jacobi-round design.

Remove the remaining hot-path host allocations as part of this work. In particular set_coords must not recreate scc_ok, and scc_mix_inner must reuse persistent hist/stagnant/done/samples/q scratch instead of Vec allocations.

Required benchmark output after each optimization:

* total SCC wall time;
* device time by stage;
* active-count curve versus SCC iteration;
* Jacobi zero-sweep / nonzero-sweep counts;
* Fermi Newton iteration histogram;
* WG size and active count;
* eigensolver residual/orthogonality;
* CPU parity of energy/charges;
* for Jacobi, separate timings for zero-sweep and forced one-sweep matrices at N≈86 and N≈246.

Optimize production batch throughput, not batch=1 latency.

That is the direction I would pursue. The **first three changes—remove the fused Fermi tail, retune WG for batch throughput, and logical `active_ids` compaction—are much lower risk than rewriting the eigensolver** and should tell us how much of the apparent disaster is fixed overhead/scheduling. Then the multi-WG block Jacobi addresses the genuine structural memory problem that remains, especially at N=246.


---

# USER

Another noted from my dicsusion with coding agent, reflect on them and include it to your analysis:

\---

# SWE-2

The other LLM's update is genuinely sharper than my analysis — let me verify the checkable parts before reflecting. First: is the CPU baseline actually single-threaded? Cheap test — rerun the bench with `OPENBLAS_NUM_THREADS=1` and compare the CPU 1-pt column.

Lib compiles now (the sparse edit is complete). The test is `--ignored`:

Very informative — and the other LLM's §2.1 criticism let me measure something immediately: with `OPENBLAS_NUM_THREADS=1` the CPU baseline is actually **faster** (azaindole 12.18→9.42 ms, DTH 163.9→131.3). OpenBLAS threading was *hurting* small-matrix dsyevd — so the published speedups were, if anything, slightly optimistic, and the 1-thread baseline is now the controlled one.

And the rerun confirms their sharpest claim — the **82-SM fingerprint**: recomputed `G400 = 400·t1/t400` = azaindole **92×**, GC **84×**, AT **86×** — vs diazaphenalene 57×, DTH 17×. At N≈85 the batch gain is essentially the SM count, and batch=1 GPU ≈ CPU speed (0.7–0.8×).

Let me merge their version into the canonical report with my new measured facts added:

Merged — the canonical report is now their version plus my measured follow-up. Reflection on their notes:

## What's genuinely useful (verified where checkable)

1. **The 82-SM fingerprint — the sharpest observation.** `G400 = 400·t1/t400` ≈ 82–92× for N=84–87, reproduced on my rerun. Combined with batch=1 GPU ≈ 0.7–0.95× CPU, the honest conclusion is: **all the speedup comes from ~82-way SM-level replication; one matrix per workgroup runs at roughly CPU speed.** My earlier "GPU is 60× faster per replica-iter" framing was misleading — it divided out the batch benefit.
2. **Benchmark-fairness critique — partly verified by measurement.** `OPENBLAS_NUM_THREADS=1` made CPU *faster* (threading is pure overhead at N≈85–246), so the old column was if anything optimistic by ~10–25%. Also correct that `iters×400` ≠ GPU work (park mask) and CPU 1-pt × 400 ≠ same-work baseline. The `Σ active replicas` metric is the right work unit.
3. **Zero-sweep Jacobi anomaly — most actionable.** Warm `BᵀHB` is already within tolerance → mean 0 sweeps, yet `scc.jacobi` is ~60–75% of device time. The cost is the prologue/Fermi-tail machinery *deciding* no work is needed — a lightweight "check+diag" fast path or the R8b early-out is the highest-value optimization.
4. **WG-size A/B at production batch** — WG=512 was tuned on batch=1 latency; at batch=400 it may cap residency at ~1 WG/SM. Sweeping 128/256/512 at batch 82/164/328/400 is cheap and directly tests the 82-slot model.
5. **Local-A specialization for N≈80–100** — f32 A is ~30 kB at N=86, fits the 48 kB local limit; could kill most round-by-round global traffic. Only helps the rare cold/sweep path though — secondary to the zero-sweep fix.
6. **cuSOLVER `syevjBatched` as a diagnostic ceiling** — smart use: answers "is our Jacobi leaving 3× on the table, or is the algorithm wrong for this size" without adopting a CUDA dep.
7. **N=246 is outside the direct solver's regime** — 5.45× worse than even the wave model; needs blocked/multi-WG or the sparse path.

## Caveats on their text

- "1 effective WG/SM" is a wall-time inference, not a measured occupancy — the direct kernel's local usage is only ~15 kB, so 2–3 WGs *could* fit by shared memory; the limiter is likely registers/global latency. They flagged this honestly.
- The §2.3 profiling numbers (74.8%, p90=0 sweeps) cite the labbook, not this run — consistent with R8b but worth re-confirming at batch=400.

I left `_UPDATED.md` in place since it's your file — the canonical report now contains everything from it, so it can be removed when you're ready.

\---

# USER

what the hell we are using workgroupize 512 ???!?!?! for system with 80 atoms? Do I understand it correctly? how the hell can we saturate this? The WG size mus be given by the size of the system!!!! epecially when we one workgroup per replica? this is ttoal nonsense. No wonder we canno squeeze more replicas into one compute unit. Also the size of local memory limit number of replicas processed by one computeunit! I was sayng it is essential to keep sinze of local memory reasonable it is big compromise between global memory overhead, and compute-unit occupancy. We should cache manually only something we know it will help. Otherwise the GPU have some automatic cache maybe it is better to raly on it rather than choke the gPU computeunit!

\---

# SWE-2

Let me get the actual numbers — which kernel runs, its local-memory footprint, and what residency that allows.

The user is right — let me get the exact numbers to confirm and quantify.

You've nailed the actual bug. Let me put the exact numbers on it.

**The kernel:** `jacobi_cyclic_global_batched` — one WG per replica, WG = `min(max_wg, 512)` = **512 threads** hard-coded. The comment at `gpu_scc_plan.rs:1782` even admits the tuning basis: "WG=512 ~1.55× faster than 256 ... **direct_jacobi_bench**" — that was a *single-matrix latency* benchmark, and the comment asserts "batch doesn't change per-replica time — one WG per system already parallelizes", which is exactly the fallacy: it ignores per-SM residency.

**Local memory per WG (WG=512, JACOBI_PREC=0):**

| array | size |
|---|---|
| `dred`/`dred2` f64 reductions — **Fermi tail only** | 2 × 4 KB |
| `le[256]` eigenvalue staging — **Fermi tail only** | 1 KB |
| `reduce[512]` f32 | 2 KB |
| `rot_c/s/p/q[128]` | 2 KB |
| `lmu`, `l_nrot` | ~40 B |
| **total** | **~13.3 KB** |

So ~9.2 KB of the 13.3 KB is Fermi-tail scratch that's **dead weight for the entire eigensolve** — the fusion permanently inflates every WG's footprint.

**Per-round parallelism at N=86:** jpair = 43 independent pairs/round; each pair's strip update is ~6·N ≈ 516 element updates. With 512 threads that's ~1 element/thread in the update phase, but the rotation-parameter phase runs ~43 threads of 512, and every reduction tree is 9 barrier levels. With WG=128: ~4 elements/thread, 7 barrier levels, ~7 KB local — and the SM can host many more WGs.

**Residency math (GA102: 1536 threads/SM, 64K regs, 48 KB OpenCL local):**
- WG=512: thread-limit 3 WGs/SM, local-limit 3 — observed G400 ≈ 82–92× ≈ 1 WG/SM effective → register-bound in practice.
- WG=128: thread-limit 12 WGs/SM, local-limit ~6 — even 4 replicas/SM would triple current throughput at N≈85.

The right design space, consistent with what you said:

1. **WG ∝ N** — e.g. `wg ≈ max(64, round_up(jpair·2, 32))` → 96–128 for N=86, not 512.
2. **Split the Fermi tail out of the Jacobi kernel** (or run it once at solve end) — removes ~9 KB/WG of dead local.
3. **Sub-group packing**: one WG of 512 holding 4 replicas × 128-thread subgroups — same global shape, 4× replica residency per launch; bigger rewrite but that's the "multiple systems per compute unit" you're describing.
4. For small N the extreme is warp-per-replica (32 thr) → up to 48 replicas/SM.

Let me add this to the report.

Yes, you understand correctly — and it's worse than just the WG size:

**WG=512 for an 86-orbital matrix:**
- The parallel unit is the *pair*: N=86 → **43 independent rotations per round**. 512 threads means the rotation-parameter phase uses ~43 threads; the strip update spreads ~516 elements over 512 threads (~1 elem/thread). The reduction tree is 9 barrier levels.
- The 512 came from a **batch=1 latency benchmark** (`direct_jacobi_bench`: "1.55× faster than 256") — and the code comment literally reasons "batch doesn't change per-replica time — one WG per system already parallelizes", i.e. residency was never considered.

**Local memory is also wasted:** ~13.3 KB per WG, of which **~9.2 KB is Fermi-tail scratch** (`dred`/`dred2` f64 reduction buffers, `le[256]` staging) — completely dead during the eigensolve itself. The live Jacobi footprint is only ~4 KB. That fusion (done to save one launch) is choking exactly the occupancy you're talking about.

**The residency math says it all:** GA102 = 1536 threads/SM, 48 KB local. WG=512 → ≤3 WGs/SM by threads/local, and the measured ~82× batch gain at N=85 says ~1 effective WG/SM (register-bound). WG=128 → 12 by threads, ~6 by local. Your instinct is confirmed: we built a scheduler that maps **one replica per SM**, not "as many replicas as the compute unit can hold".

Concrete directions consistent with your point, now in the report (§2.4 H1 + open item 2):

1. **WG ∝ N** — for N=86, ~128 threads is the right scale, not 512.
2. **Split the Fermi tail** out of the Jacobi kernel (or once per solve) — kills ~9 KB dead local per WG.
3. **Sub-group packing** — the real version of "more replicas per compute unit": e.g. 4 replicas × 128-thread subgroups in one WG, or warp-per-replica for small N. That changes the ceiling from ~82 concurrent replicas to several hundred.
4. On trusting the hardware cache vs manual local memory: for the warm path (0 sweeps — the common case) the whole question is moot; for cold sweeps, A fits in ~30 KB at N=86 — but only if local isn't already burned on Fermi scratch and oversized `reduce` buffers.

The immediate measurement to run: WG A/B at **production batch** (82/164/400), not batch=1 — that directly tests whether residency is the binding constraint.


---


# Chat GPT 5.6 sol

Yes. These new measurements make the picture much cleaner, and they strengthen rather than weaken the conclusion that the current mapping is leaving a lot on the table.

The most important verified fact is now this:

$$
G_{400}=\frac{400\,t_1}{t_{400}}\approx 84-92
$$

for the N≈84–87 systems, essentially the 82-SM count of the 3090. Combined with the controlled `OPENBLAS_NUM_THREADS=1` result—where CPU LAPACK actually became faster—this rules out the comforting explanation that the CPU baseline was secretly using many cores. The GPU is getting almost all its speedup from **replicating roughly CPU-speed single-matrix work across the machine**, rather than making each matrix intrinsically fast.

That is the central problem.

### WG=512: the problem is not simply “512 > 86”

I agree strongly with your reaction, but with one technical correction. The natural parallelism of the current Jacobi kernel is not just \(N\) threads. At N=86 there are 43 rotation pairs, but after calculating those rotations the kernel has

$$
43^2 + 86\times43 = 5547
$$

independent A/V update tasks in each round. The code distributes those tasks over the whole WG. 

So WG512 explains why the **batch=1** benchmark liked 512: each thread performs only about 11 of those small update tasks, whereas WG128 performs ~43.

The mistake is this:

> We optimized the latency of one matrix instead of the throughput of hundreds of matrices.

The production harness literally chooses up to WG512 based on that batch=1 benchmark.

For your actual use case, residency is at least as important as latency.

And WG512 is especially bad because it is accompanied by a needlessly fat kernel.

The attached kernel with `JACOBI_PREC=0` allocates approximately:

| local object                   |        WG512 |
| ------------------------------ | -----------: |
| rotation c/s/p/q arrays        |      ~2.0 kB |
| `reduce[512]`                  |      ~2.0 kB |
| Fermi `le[256]`                |      ~1.0 kB |
| Fermi `dred[512] + dred2[512]` |      ~8.0 kB |
| small state                    |     ~0.04 kB |
| **total**                      | **~13.0 kB** |

The Fermi tail therefore accounts for roughly **9 kB of local memory which has nothing to do with Jacobi**. 

More importantly, the fused Fermi code may increase the compiler's **register/private-memory requirement** for the entire kernel as well. Local memory alone does not explain the observed approximately one-effective-WG-per-SM behavior: 13 kB would theoretically allow several WGs under a 48 kB local-memory limit. So the 82-SM fingerprint is an empirical residency symptom, not proof that local memory alone is the limiter.

We need to query and record at construction time:

* `CL_KERNEL_LOCAL_MEM_SIZE`
* `CL_KERNEL_PRIVATE_MEM_SIZE`
* `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`
* device local-memory size
* actual WG size

and then use the batch throughput itself as the ultimate occupancy measurement.

### I would change WG selection immediately

For this direct solver a much more sensible starting heuristic is approximately

$$
WG \sim 2\,N_{\rm pair}
$$

rounded to a power of two.

Thus:

$$
N=86,\quad N_{\rm pair}=43
\Rightarrow WG\approx128
$$

and

$$
N=246,\quad N_{\rm pair}=123
\Rightarrow WG\approx256.
$$

That is already far more rational than “512 for everything N>64.”

But I would not hard-code that yet. Run:

$$
WG=\{64,128,256,512\}
$$

at

$$
N_{\rm active}=\{400,160,80,40,20,5,1\}
$$

for both zero-sweep and one-sweep cases.

I particularly want WG64 tested at N≈86. It gives only 43/64 useful lanes during rotation construction, and although each lane has considerably more update work, the GPU may hide that beautifully by having many independent replica WGs resident.

The optimum may be completely different between batch=1 and batch=400.

---

## I now agree even more strongly that Fermi must come back out of Jacobi

This fusion was optimization in the wrong dimension.

We saved one launch, but forced two completely different algorithms to have the same workgroup geometry:

* Jacobi: perhaps WG128–256;
* Fermi μ solve for 86 scalar eigenvalues: WG32 or WG64.

The current Fermi tail stages the eigenvalues, calculates min/max, then allows up to 24 Newton/bisection iterations, each with two FP64 WG reductions. 

That is particularly awful in the **zero-sweep case**, because the code effectively does:

$$
\text{“No diagonalization needed.”}
$$

followed by a fairly elaborate 512-thread scalar algorithm.

Make Fermi a separate WG32 kernel. I would use:

$$
f_k = \frac1{1+\exp(x_k)}
$$

computed in f32, while accumulating

$$
\sum f_k,\qquad \sum f_k(1-f_k)
$$

into f64 scalar reductions if we want the decision robust.

And record how many Newton iterations it actually takes with warm μ. I expect the histogram to look much closer to 1–4 than 24.

This change is attractive because it simultaneously reduces:

* local memory,
* likely register pressure,
* reduction depth,
* barrier count,
* zero-sweep Jacobi latency.

So it is not merely moving code into another launch.

---

## On manual local memory: your instinct is exactly right

I would modify my previous suggestion about “keep all A local.”

It is a useful **diagnostic**, but I would not make it the default architecture.

For N=86:

$$
A = 86^2\times4 \approx 29.6\ {\rm kB}.
$$

If you reserve ~30 kB/WG merely to cache A, you may immediately force approximately one resident WG/SM under a 48 kB local-memory limit.

That could reduce global traffic and simultaneously destroy the batch throughput we actually care about.

So the design principle should be:

> **Use local memory only when the reuse gained is worth the occupancy lost.**

The tiled GEMM you already have is a good example: it keeps only small A/B tiles locally and reuses them across the K loop. 

For Jacobi I would follow exactly the same philosophy.

Trust L1/L2 for broad matrix traffic unless we know an object will be reused heavily. Explicitly cache:

* a small \(2B\times2B\) pivot,
* its rotation U,
* one transformation tile,

not an entire matrix just because it happens to fit.

The hardware cache is there exactly for the ambiguous reuse cases.

---

# This changes my block-Jacobi recommendation too

The old B=32 design is too greedy with local memory.

Its pivot + U alone require about 33 kB. 

If we also include a B×2B strip tile, approximate local-memory footprints are:

$$
\begin{array}{c|c}
B & \text{pivot + U + strip}\\
\hline
16 & 10.3\ {\rm kB}\\
24 & 22.9\ {\rm kB}\\
32 & 40.5\ {\rm kB}
\end{array}
$$

So on a nominal 48 kB local-memory budget:

* **B=32 → effectively one WG/SM**
* B=24 → perhaps two
* **B=16 → perhaps four**

before registers/thread limits intervene.

That makes **B=16 the most interesting starting point**, not B=32.

This is a major adjustment to the old manifest reasoning. We were optimizing block size for arithmetic reuse without adequately pricing residency.

---

# Multi-WG Jacobi: yes, but for two very specific reasons

I now think a multi-WG solver is likely worthwhile.

But not because “more threads = faster.”

It helps in two different regimes.

First, an actual direct Jacobi sweep is genuinely memory-inefficient. At N≈86, by a straightforward traffic estimate, the present scalar-round algorithm moves around

$$
\sim10\ {\rm MB/system/sweep}
$$

through A/V.

At N=246 this explodes to order

$$
200+\ {\rm MB/system/sweep}.
$$

That explains beautifully why N=246 collapses.

A properly tiled block transformation can load a matrix tile once, apply a relatively large block transformation, and write it once. This dramatically raises arithmetic intensity.

Second, a multi-WG solver fixes the **tail utilization** problem.

With 400 systems active, one WG/system already exposes many independent jobs.

With only 8 difficult systems left, it gives just 8 useful WGs.

If each matrix can expose perhaps 10–30 block-transform WGs, those same 8 systems can once again occupy a large fraction of the GPU.

So I increasingly see the best architecture as:

$$
\boxed{
\text{many replicas}
\times
\text{several WGs/replica}
}
$$

rather than either extreme.

---

## But do not parallelize the current scalar Jacobi rounds across WGs

That would require synchronization after every Brent–Luk round:

$$
r=0\rightarrow1\rightarrow2\rightarrow\cdots
$$

There is no portable OpenCL grid barrier within the kernel.

Turning each round into a kernel launch would give ~85 launches per sweep at N=86. Completely unacceptable.

Instead use **block Jacobi**.

For B=16, N=86 gives roughly six orbital blocks.

A block round pairs them as, for example,

```text
round 0: (0,5) (1,4) (2,3)
round 1: (0,4) (5,3) (1,2)
...
```

Each block pair has a local 32×32 compound pivot.

A block phase would look like:

```text
pivot stage:
    one WG / replica / block-pair
    32×32 compound pivot → local
    local Jacobi → U_pair

matrix transform:
    many WGs / replica
    each WG owns one A tile
    Aij <- Ui^T Aij Uj

eigenvector transform:
    many WGs / replica
    Vi <- Vi Ui
```

Then a kernel boundary provides the global synchronization before the next block phase.

For six blocks, you need only about five block phases per sweep rather than 85 scalar Jacobi rounds.

And the matrix-transform stage looks very much like the tiled GEMM machinery you already have.

That is where I would spend substantial kernel-development effort.

---

# The “slowest replica takes 80 iterations” issue

Here I want to make one correction to both your mental picture and the coding agent's earlier report.

Your current harness already does something useful: the GPU DIIS kernel writes `active[sid]=0` as soon as an individual replica converges, and subsequent kernels check that mask. 

The SCC loop also enqueues 8 iterations at once, without host synchronization, and reads status only after the chunk.

So a replica converging at iteration 17 **does not perform the expensive electronic work all the way to iteration 80**.

That part is already fixed.

What remains inefficient is that kernels continue to be launched with the original batch geometry.

For example the GEMM has a 2D tile grid for **every batch element**, and dead replicas merely hit:

```c
if (ib >= batch || active[ib] == 0) return;
```

before loading the tile. 

Suppose N=86 gives 6×6=36 GEMM WGs/system.

At batch400:

$$
400\times36=14\,400\ {\rm WGs}.
$$

If only 10 replicas remain active, you still dispatch 14,400 WGs, of which 14,040 immediately die.

That is clearly wasteful.

---

# The harness should therefore switch from masking to compaction

Not physical matrix compaction.

Just logical IDs.

Keep persistent:

```text
active_ids[400]
n_active
```

At the existing 8-iteration chunk boundary you already synchronize/read statuses, so build:

```text
active_ids = [7, 12, 61, 88, ...]
```

in preallocated host scratch and upload perhaps 10–50 integers.

Then launch only:

```text
n_active × matrix tiles
```

and inside the kernel:

```c
int slot = get_group_id(2);
int sid  = active_ids[slot];
```

All matrix buffers stay at their original replica offsets.

This gives three benefits simultaneously:

1. no dead WGs;
2. the scheduler sees a truthful active population;
3. multi-WG-per-matrix becomes very effective in the tail.

And we don't introduce any extra synchronization because compaction occurs at the synchronization point that already exists.

---

## I would also make the SCC chunk size adaptive

Eight iterations is sensible when 400 replicas are alive because host synchronization is expensive.

When only 8 remain, waiting 8 iterations before rebuilding the active list makes less sense.

Something like:

$$
\begin{cases}
8 & n_{\rm active}>100\\
4 & 20<n_{\rm active}\le100\\
2 & n_{\rm active}\le20
\end{cases}
$$

is worth benchmarking.

Not because a finished replica continues doing expensive work—it doesn't—but because we want to compact the launch domain more quickly as the batch collapses.

---

# There is an even better harness optimization for large scans

For a 2D PES, you often have more independent geometries waiting than fit efficiently at once.

Do not think of “batch400” as a fixed group that must live and die together.

Think of it as **400 GPU slots**.

When one geometry finishes SCC/FIRE, assign a new pending geometry to that slot.

Thus:

```text
slot 17 finishes
        ↓
return result
        ↓
load next scan geometry into slot 17
        ↓
continue
```

Then the GPU remains close to saturation until the entire work queue is nearly empty.

Only the very last handful of geometries experience the true long tail.

This is especially attractive for large 2D/3D parameter scans and for the CDFT/PCET work we discussed, where there could eventually be thousands of independent states/geometries.

It requires more harness engineering than `active_ids`, so I would do logical compaction first.

---

# I would *not* implement the agent's “4 replicas inside one WG512” idea

This is one place where I disagree with SWE-2.

Packing four 128-thread replicas into one 512-thread WG sounds superficially attractive, but it throws away one of the scheduler's best features: independent workgroups can finish independently.

If four replicas share one WG and one needs two Jacobi sweeps while the others need zero, you have created the same slowest-member divergence problem **inside the workgroup**.

You would also need:

* subgroup-level barriers rather than WG barriers,
* replicated local scratch,
* divergent control flow,
* more complicated resource accounting.

Simply launching **four independent WG128 workgroups** lets the hardware do exactly that packing for us while retaining independent completion.

That is cleaner and almost certainly what we want.

---

# So my revised priority order is now

| priority   | change                                             | reason                                                      |
| ---------- | -------------------------------------------------- | ----------------------------------------------------------- |
| **P0**     | Split Fermi tail from Jacobi                       | removes ~9 kB/WG + f64/barrier machinery + likely registers |
| **P0**     | A/B WG64/128/256/512 at production active counts   | directly tests the 82-SM/residency hypothesis               |
| **P1**     | `active_ids[]` logical compaction                  | fixes shrinking-batch launch waste                          |
| **P1**     | Instrument `n_active(iter)`                        | tells us actual tail cost rather than guessing              |
| **P2**     | tiny zero-sweep probe / fast path                  | common production case should barely cost anything          |
| **P2**     | adaptive SCC chunk size                            | quicker tail compaction                                     |
| **P3**     | B≈16 multi-WG block Jacobi                         | attacks actual-sweep bandwidth + tail occupancy             |
| **P4**     | persistent work queue / refill finished scan slots | eliminates scan-level tail                                  |
| diagnostic | A-local/V-global N≈86                              | useful test of memory hypothesis, probably not final design |
| diagnostic | cuSOLVER batched Jacobi                            | external performance ceiling                                |

The most important conceptual change from my previous analysis is:

> **Do not optimize “the Jacobi kernel” as one monolithic thing.**

There are really three operating regimes:

$$
\boxed{
\begin{array}{ll}
\textbf{Warm matrix already diagonal} &
\rightarrow \text{tiny probe + Fermi} \\[2mm]
\textbf{Many active replicas requiring sweeps} &
\rightarrow \text{small WG / high residency} \\[2mm]
\textbf{Few difficult replicas requiring sweeps} &
\rightarrow \text{multi-WG tiled/block Jacobi}
\end{array}}
$$

And I would resist manual caching of entire matrices unless a benchmark proves it helps. **The performance objective is not minimum global traffic; it is minimum global traffic subject to keeping several useful workgroups resident per SM.**

That last constraint was missing from the earlier kernel design, and I think it explains a substantial part of why the 3090 currently behaves more like “82 parallel small CPUs” than like the massively parallel processor we expected.





---

# USER

I agree that the fact that Fermi occupanc solver takes local memory is reason why to rmov it from jacobi kernel, we must optimize the kernels to efficiently use shared memory(only the thing we know we reuse many times) and to allow many WGs to run on one compute unit we cannot use too much memory

I quite disagree with pussing more threads per WG than is number of atoms. The whole point of single-WG-per-system was to make the kernel architecture and memory access pattern as simple as possible to allow it optimize efficiently. I have feeling that if we have one thread per column/row of the matrix if would be simpler and better layour, less synchronization than if we try to push parallization beyond that. I also think that the idea to solve the matrix by pieces e.g. 16x16 where these blocks 16x16 can reside in local momeory and make loop over these tiles inside the krenel within one WG, could be great to keep the local memory use bound while still making sure we minimize/amortize global memory I/O
But you must formulate the actuall algorithm (pseudocode) and perhaps also sketch the kernels, and evaluate criticall what seems most worthy...

I agree that “4 replicas inside one WG512” is very bad idea

Also I like the idea to conider the replicas as "slots", we run several iterations, evaluate which converged, and then we enque kernel with different size working only on the remaining... so that there is not so many idle replicas doing nothing but blocking computeunits...

elaborate more on these ideas



---


# Chat GPT 5.6 sol

I agree with this direction. I would now make the **primary experimental eigensolver a one-WG-per-system, one-useful-thread-per-matrix-row, small-block Jacobi**, rather than jumping to multi-WG.

And by “one thread per row” I mean one per **orbital/matrix row**, so for GC \(N\approx86\): about 86 useful threads, probably WG=96 only because NVIDIA schedules in warps of 32. Not WG=512.

The architecture I would try is:

$$
\boxed{
\text{global } A,V
\quad+\quad
\text{local only }(2B\times2B)\text{ pivot and }U
\quad+\quad
\text{one thread owns one row}
}
$$

with initially \(B=16\).

That seems like the best compromise between **simple ownership, small local memory, cache-friendly repeated access, low synchronization, and multiple resident replicas per SM**.

---

## 1. Why I like the “one row = one thread” mapping

The current direct Jacobi uses WG512 and flattens roughly

$$
N_{\rm pair}^2 + N\,N_{\rm pair}
$$

small tasks over all 512 threads. It works, but it makes every phase have a different natural parallelism: only ~43 threads calculate rotations for N=86, 512 participate in matrix updates, 512 participate in reductions, etc. 

Instead I would make the fundamental ownership invariant:

```text
thread r owns row r of A and V
```

for all \(r<N\).

For N=86:

```text
WG = 96
threads 0..85   -> matrix rows
threads 86..95  -> only help cooperative local-pivot operations / otherwise idle
```

That's not wasteful in the meaningful sense: the hardware would schedule 3 warps anyway.

For N=246:

```text
WG = 256
```

This has some very nice consequences.

Each thread knows exactly what it writes. There is no complicated mapping from flattened task number to pair/pair/row. There are very few barriers. The WG size naturally tracks system size. And most importantly, **shrinking the local-memory footprint actually gives the scheduler the chance to keep several independent replicas resident on one SM**.

---

# 2. But I would not simply do scalar Jacobi with this mapping

If we keep the current scalar Brent–Luk rounds, the fundamental problem remains:

$$
85\text{ rounds/sweep at }N=86,
$$

and A/V are streamed repeatedly from global memory. The current production kernel explicitly keeps both A and V global. 

So I would combine your row ownership idea with **block Jacobi**.

The key is:

> Use local memory only for data that is reused many times before being evicted.

For a block pair \(P,Q\), that is exactly:

1. the compound pivot

   $$
   A_{PQ}=
   \begin{pmatrix}
   A_{PP} & A_{PQ}\\
   A_{QP} & A_{QQ}
   \end{pmatrix},
   $$
2. and its unitary/orthogonal transformation \(U\).

Everything else is streamed.

---

# 3. Proposed one-WG block Jacobi

Take

$$
B=16,\qquad M=2B=32.
$$

Split the 86×86 matrix into six orbital blocks:

```text
0:  0..15
1: 16..31
2: 32..47
3: 48..63
4: 64..79
5: 80..85
```

Then one block sweep visits all block pairs:

```text
(0,1) (0,2) ... (0,5)
      (1,2) ... (1,5)
             ...
                  (4,5)
```

There are only

$$
{6\choose2}=15
$$

block-pair operations.

For each pair, make a 32×32 compound pivot, diagonalize that **entire small matrix in local memory**, obtain \(U\), and then use \(U\) to update the corresponding 32 columns of the full A and V.

No multi-WG synchronization.

No global atomics.

No full matrix in local memory.

---

## 4. Local-memory budget becomes quite attractive

With padding against bank conflicts, approximately:

|      B | pivot dimension |   pivot + U | blocks for N=86 | block-pairs/sweep |
| -----: | --------------: | ----------: | --------------: | ----------------: |
|     12 |           24×24 |     ~4.8 kB |               8 |                28 |
| **16** |       **32×32** | **~8.4 kB** |           **6** |            **15** |
|     24 |           48×48 |    ~18.8 kB |               4 |                 6 |
|     32 |           64×64 |    ~33.3 kB |               3 |                 3 |

This makes B=16 look particularly good.

B=32 minimizes the number of block pairs, but essentially spends the whole local-memory budget on one replica. That defeats the occupancy goal.

B=24 is interesting and worth one A/B.

But **B=16 is where I would begin**:

$$
\sim 8.5\ {\rm kB/WG}
$$

for the expensive reusable matrices, instead of ~33 kB for B32 or ~13 kB for the present Jacobi+Fermi monster.

The old tiled design already identified the benefit of local pivot/U reuse, but B32 used about 33 kB before other workspace and kept the strip processing within one huge WG. 

---

# 5. The important simplification: no local strip buffer

This is where I would change the old tiled design.

After obtaining \(U\), each global row needs to transform only its vector of \(2B=32\) entries:

$$
a_r = A[r,P\cup Q].
$$

Then

$$
a'_r=a_r U.
$$

Likewise,

$$
v'_r=V[r,P\cup Q]U.
$$

So let thread `r` do that directly.

Conceptually:

```c
if (row < n) {

    // -------------------------------
    // A strip
    // -------------------------------

    if (!row_is_inside_pivot(row)) {

        float x[M];

        // 32 values only
        for (int k=0; k<m; k++)
            x[k] = A[row*n + idx[k]];

        for (int j=0; j<m; j++) {
            float y = 0.0f;

            for (int k=0; k<m; k++)
                y = fma(x[k], U[k*LD+j], y);

            A[row*n + idx[j]] = y;

            // exploit symmetry:
            A[idx[j]*n + row] = y;
        }
    }

    // -------------------------------
    // Eigenvector strip
    // -------------------------------

    float x[M];

    for (int k=0; k<m; k++)
        x[k] = V[row*n + idx[k]];

    for (int j=0; j<m; j++) {
        float y = 0.0f;

        for (int k=0; k<m; k++)
            y = fma(x[k], U[k*LD+j], y);

        V[row*n + idx[j]] = y;
    }
}
```

The temporary `x[32]` should live in private registers.

That is data we know is reused 32 times, so caching it is unquestionably worthwhile.

But I would **not** put it in WG local memory.

This nicely follows your rule:

> Explicitly cache only things whose reuse we understand.

Here:

* `pivot`: reused many times → local
* `U`: reused by every row → local
* current row vector: reused many times by one thread → registers
* rest of A/V: one streaming use → global/cache

That is exactly how I would want to use the memory hierarchy.

---

# 6. And hardware cache may actually handle the row access reasonably well

There is an apparent concern that row ownership gives poor warp coalescing because lane \(r\) accesses row \(r\).

But for B=16, each thread reads two contiguous 16-float chunks:

```text
A[row][P:P+16]
A[row][Q:Q+16]
```

Each is only 64 bytes.

So although a single load instruction across the warp is strided by the row pitch, each lane continues consuming the same one/two cache lines during its inner loop.

The L1/cache has a good chance to turn that into:

```text
fetch row fragment once
reuse ~16-32 times
```

rather than genuinely going to DRAM for every scalar.

I agree with you that we should **let the hardware demonstrate whether that is good enough before introducing another manually managed strip cache**.

If profiling later says those loads are bad, then we can add a tiny cooperative strip tile.

But not pre-emptively.

---

# 7. Full algorithm

The kernel becomes approximately:

```c
kernel block_jacobi_1wg(
    A,
    V,
    active_ids,
    n_active,
    n
) {
    int slot = group_id(0);
    int sid  = active_ids[slot];

    int tid = local_id(0);
    int row = tid;

    global A_sys = A + sid*n*n;
    global V_sys = V + sid*n*n;

    local float P[M][M+1];
    local float U[M][M+1];

    local float rot_c[B];
    local float rot_s[B];

    local float row_frob[WG];
    local float row_off [WG];

    // -----------------------------------
    // CHEAP WARM PROBE
    // -----------------------------------

    float fa = 0;
    float fo = 0;

    if (row < n) {
        for (int j=0; j<n; j++) {
            float x = A_sys[row*n+j];

            fa += x*x;
            if (row != j)
                fo += x*x;
        }
    }

    row_frob[tid] = fa;
    row_off [tid] = fo;

    barrier(local);

    // n~86: serial sum of 86 floats is cheap;
    // do NOT build a 9-barrier reduction tree.
    if (tid == 0) {
        float sf = 0;
        float so = 0;

        for (int r=0; r<n; r++) {
            sf += row_frob[r];
            so += row_off[r];
        }

        converged =
            sqrt(so) <= TOL * sqrt(sf);
    }

    barrier(local);

    if (converged)
        return;


    // ===================================
    // BLOCK JACOBI SWEEPS
    // ===================================

    int nb = ceil_div(n,B);

    for (int sweep=0; sweep<MAX_SWEEPS; sweep++) {

        // Simple cyclic ordering.
        // One WG owns matrix => no need for outer Brent-Luk.

        for (int bp=0; bp<nb; bp++) {
            for (int bq=bp+1; bq<nb; bq++) {

                // --------------------------------
                // 1. build index list P U Q
                // --------------------------------

                int m = build_indices(bp,bq,...);

                // --------------------------------
                // 2. load compound pivot to local
                // --------------------------------

                for (int t=tid; t<m*m; t+=WG) {
                    int i=t/m;
                    int j=t%m;

                    P[i][j] =
                        A_sys[idx[i]*n + idx[j]];

                    U[i][j] =
                        (i==j) ? 1 : 0;
                }

                barrier(local);


                // --------------------------------
                // 3. local eigensolve
                //    U^T P U = diagonal
                // --------------------------------

                jacobi_local(P,U,m);

                barrier(local);


                // --------------------------------
                // 4. transform external A rows
                //    and all V rows
                // --------------------------------

                if (row < n) {

                    if (!inside(row,idx,m))
                        transform_A_row(...);

                    transform_V_row(...);
                }


                // pivot rows write transformed pivot
                if (tid < m) {
                    for (int j=0; j<m; j++)
                        A_sys[idx[tid]*n + idx[j]]
                            = P[tid][j];
                }

                barrier(
                    local |
                    global
                );
            }
        }


        // --------------------------------
        // 5. convergence once / block sweep
        // --------------------------------

        if (row < n)
            row_off[tid] = row_offdiag_norm(A_sys,row);

        barrier(local);

        if (tid == 0) {
            float so=0;
            for (int r=0; r<n; r++)
                so += row_off[r];

            converged =
                sqrt(so) <= off_exit;
        }

        barrier(local);

        if (converged)
            break;
    }
}
```

This is much closer to what I think your GPU programming style wants.

---

# 8. The local 32×32 Jacobi can reuse what you already have

Inside `jacobi_local(P,U,32)` I would reuse your current Brent–Luk machinery.

Now the important difference is that P and U are **actually local**, so those repeated Jacobi rounds don't touch global memory.

For M=32:

$$
16\text{ independent rotations/round},
\qquad
31\text{ rounds/sweep}.
$$

We can let the whole ~96-thread WG assist with the little local matrix, because this phase is short and local.

Something like:

```c
for inner_sweep {

    for round=0..30 {

        // first 16 lanes
        if (tid < 16)
            make_rotation(...);

        barrier(local);

        // cooperative local 2x2 block update
        for (int ij=tid;
             ij<16*16;
             ij+=WG)
        {
            update_2x2_block(P,...);
        }

        // U <- U J
        for (int t=tid;
             t<32*16;
             t+=WG)
        {
            update_U(...);
        }

        barrier(local);
    }

    if (local_offdiag_small)
        break;
}
```

This part is not what is killing residency, because the arrays are only ~8 kB total and all access is local.

We should keep it simple and correct first.

---

# 9. There is an important block-Jacobi advantage beyond memory

The current scalar solver has at N=86:

$$
85
$$

globally dependent rounds per sweep.

The proposed block solver with B16 has:

$$
15
$$

compound pivot operations per block sweep.

Those aren't directly equivalent operations—each compound pivot performs a substantial local eigensolve—but globally the matrix is only streamed through **15 much larger transformations instead of 85 tiny transformations**.

And because each block participates in only five other block pairs, the full V matrix is effectively transformed about five times/block-sweep rather than ~85 scalar rounds.

A rough global-traffic estimate is therefore closer to perhaps

$$
0.8-1\ {\rm MB/system/block\ sweep}
$$

instead of roughly

$$
10\ {\rm MB/system/direct\ sweep}.
$$

I would regard this as an order-of-magnitude potential reduction in global traffic, not an order-of-magnitude promised speedup.

The local 32×32 diagonalizations now cost arithmetic, so the actual speedup will be smaller.

But importantly, **we have converted cheap global-memory operations into lots of local FMA work**, which is exactly what the GPU is good at.

---

# 10. B=16 vs B=24 is the real A/B I want

I would not waste time exploring ten block sizes.

The two interesting candidates are:

### B=16

$$
M=32,\quad L_{\rm pivot+U}\approx8.4\text{ kB}
$$

N=86:

$$
n_b=6,\quad 15\text{ pivot pairs}.
$$

Advantages:

* excellent local-memory occupancy;
* small local diagonalization;
* row private vector only 32 floats;
* likely several resident WGs/SM.

### B=24

$$
M=48,\quad L\approx18.8\text{ kB}.
$$

N=86:

$$
n_b=4,\quad6\text{ pivot pairs}.
$$

Advantages:

* much less global A/V traffic;
* fewer compound transforms.

Disadvantages:

* perhaps only ~2 WGs/SM from local memory;
* row temporary grows to 48 registers-ish;
* local 48×48 diagonalization substantially more expensive.

This is a genuinely meaningful tradeoff.

I would benchmark exactly those first.

B32 should be a reference showing what happens when we sacrifice occupancy for locality.

---

# 11. One very useful design detail: do not use an outer Brent–Luk schedule yet

Because **one WG owns the system**, block pairs cannot actually execute concurrently.

Therefore something like:

```text
round 0: (0,5) (1,4) (2,3)
```

doesn't buy us anything initially.

Just do:

```c
for (p=0; p<nb; p++)
    for (q=p+1; q<nb; q++)
        process_pair(p,q);
```

Much simpler.

The disjoint round-robin schedule becomes useful only if we later make the block transforms multi-WG.

So I would explicitly design this first solver to be simple.

---

# 12. I now think multi-WG Jacobi should be deferred

Not abandoned.

But test this first:

```text
one WG/system
WG ~= N
B=16 or 24
small P+U local cache
A,V streamed
```

because it may already solve both problems:

* good occupancy across hundreds of replicas;
* much better arithmetic intensity when a sweep is needed.

Multi-WG introduces real architectural cost because now you need kernel boundaries for global synchronization.

I would move to it only if one of these is measured:

$$
N\approx246
$$

is still much too slow, or

$$
n_{\rm active}\ll82
$$

tail execution becomes a substantial fraction of wall time.

Then multi-WG block Jacobi has an obvious natural implementation.

But there is no reason to pay that complexity prematurely.

---

# 13. Fermi should definitely become a separate tiny kernel

On this we completely agree now.

The present Jacobi kernel allocates `le[256]`, two WG-sized f64 reductions and μ state purely for the Fermi tail. 

Remove all of it.

Then our new Jacobi local footprint is almost entirely:

```text
P
U
rotations
tiny norm scratch
```

all of which directly contributes to eigensolving.

Fermi can be:

```text
one WG32 / active replica
```

and that WG should solve μ from the diagonal.

That means the zero-sweep production path becomes beautifully cheap:

```text
C^T H C
   ↓
block-jacobi kernel
   ↓
one N² norm pass
   ↓
already diagonal → return
   ↓
WG32 Fermi
```

There is no reason for zero-sweep Jacobi itself to cost much more than reading ~30 kB at N=86.

---

# 14. The active-replica slots are equally important

I think your wording of them as **slots** is helpful.

At the simplest level, don't even move any matrices.

Keep:

```text
active_ids = [
    0,1,2,...,399
]
n_active = 400
```

Run perhaps 8 SCC iterations.

Suppose afterward:

```text
n_active = 143
```

Compact the ID array:

```text
active_ids = [
    2,5,7,8,12,...   // 143 original slots
]
```

Then the next kernels launch only **143 logical systems**.

A Jacobi kernel does:

```c
int logical = get_group_id(0);
int sid     = active_ids[logical];

A += sid*n*n;
V += sid*n*n;
```

A GEMM uses:

```c
int logical = get_group_id(2);
int sid     = active_ids[logical];
```

Nothing moves.

No expensive compaction of N² matrices.

Just one indirection.

---

# 15. And retain the active flag inside a chunk

This is important.

Suppose chunk size=8 and replica 17 converges after the second iteration.

DIIS does:

```c
active[17] = 0;
```

So iterations 3–8 immediately skip its work, as your current implementation already does. 

At the chunk boundary we remove 17 from `active_ids`.

So we get both:

```text
device mask
    = immediate stopping inside chunk

active_ids
    = remove dead launch slots between chunks
```

They complement one another.

---

# 16. The scheduler should probably use dynamic chunk size

Once `active_ids` exists, I would use something simple such as:

| active fraction | chunk |
| --------------: | ----: |
|            >50% |     8 |
|          10–50% |     4 |
|            <10% |     2 |

The exact thresholds are benchmark parameters, not sacred.

The reasoning is straightforward.

With 400 live systems, synchronizing every two iterations is stupid.

With only 8 stubborn systems left, executing eight chunk iterations before compacting again is also stupid.

---

# 17. The even better “slot” concept is a streaming scan queue

This matters particularly for your PES calculations.

Suppose there are 441 geometries.

Instead of creating one monolithic batch of 441 that eventually shrinks to one pathological point, allocate perhaps:

$$
C=128\text{ or }256
$$

physical GPU slots.

Initially:

```text
slot 0   <- geometry 0
slot 1   <- geometry 1
...
slot 127 <- geometry 127
```

When slot 37 completely finishes:

```text
save result for geometry 37
reset DIIS state of slot 37
slot 37 <- geometry 128
```

Then the GPU stays full.

Conceptually:

```text
pending jobs
     |
     v
+-----------------------------+
|  GPU slot pool              |
|                             |
| [17][42][88][91]...[130]    |
+-----------------------------+
     |
     v
completed jobs
```

That is much better than treating “batch” as something immutable.

---

# 18. And it tells us how large a batch should actually be

Right now we chose 400 partly because “more GPU work = more saturation.”

But after fixing WG residency, maybe throughput saturates already at:

$$
150
$$

or

$$
250
$$

active replicas.

If so there is no reason to keep 400 physical copies of every:

* H,
* S,
* C,
* D,
* DIIS history,
* force buffer,
* etc.

Determine the saturation capacity:

$$
C_{\rm sat}
$$

from the throughput curve and make the physical slot pool around that size.

Then stream arbitrary thousands of scan points through it.

That could significantly reduce GPU memory footprint as a bonus.

---

# 19. For relaxed PES this eventually suggests an asynchronous state machine

Not necessary immediately, but architecturally it's interesting.

Different replicas don't merely take different numbers of SCC iterations; they also take different numbers of FIRE steps.

Eventually a slot could have a state:

```text
SCC
 ↓
FORCE
 ↓
FIRE
 ↓
ASSEMBLE
 ↓
SCC
 ...
 ↓
DONE
```

and the harness batches together slots currently requiring the same operation.

That's basically a tiny GPU work scheduler.

I would **not** implement that now.

First logical `active_ids` within SCC.

Then physical scan-slot refill.

Only if FIRE divergence becomes the next bottleneck would I make the whole relaxation asynchronous.

---

# 20. What I think is worth implementing now

My ordering would now be:

1. **Remove Fermi from Jacobi.**
   Make it WG32 and record μ iterations.

2. **Set Jacobi WG approximately equal to N**, not 512:

   $$
   WG=\lceil N/32\rceil32.
   $$

   So N=86 → WG96.

3. **Add `active_ids` and dynamic launch size.**
   This is simple and directly fixes the convergence tail.

4. **Make a new `jacobi_block_1wg` branch/module with B=16.**
   One WG/system, row ownership, only pivot+U local.

5. A/B **B16 versus B24** at N≈86 and N=246.

6. Only if necessary, make the block solver multi-WG.

The critical experiment matrix I would want is then:

| solver                    |   N | active systems | metric                      |
| ------------------------- | --: | -------------: | --------------------------- |
| current direct            |  86 |            400 | baseline                    |
| direct, Fermi split, WG96 |  86 |            400 | immediate occupancy gain    |
| block B16 WG96            |  86 |            400 | production throughput       |
| block B24 WG96            |  86 |            400 | locality/occupancy tradeoff |
| same variants             |  86 |             20 | tail behavior               |
| same variants             | 246 |            400 | scaling                     |
| same variants             | 246 |             20 | tail scaling                |

And separately classify matrices into:

```text
0 sweep
1 sweep
cold / several sweeps
```

because otherwise the warm zero-sweep majority hides the eigensolver behavior.

The core architectural hypothesis I would test is now very precise:

$$
\boxed{
\text{WG}\sim N,\quad
B=16,\quad
\text{only }P,U\text{ in local memory},\quad
\text{row-private strips},\quad
\text{logical active slots}
}
$$

I think that is much more consistent with the original philosophy of your multisolver than either WG512 or immediately turning one 86×86 matrix into a complicated multi-WG computation.

---

## T01/T02 measured results (2026-09-16, git 324a9e14 + T01 patch set)

Working tree = HEAD `324a9e14` + T01 changes in `gpu_block_jacobi.cl`,
`gpu_tiled_jacobi.cl`, `gpu_eigen.rs`, `gpu_scc_plan.rs`, `gpu_dftb.rs`,
`tests/gpu_tiled_jacobi.rs`. Full logs: `debug/bench_t01_direct.log`,
`debug/bench_t01_block.log`.

### T01 correctness evidence

Deterministic tests added to `tests/gpu_tiled_jacobi.rs`, all failed
pre-fix as predicted, all pass post-fix:

| test | pre-fix | post-fix |
|---|---|---|
| `wg96_impulse_reduction` (N86, I + 0.01 at (2,5)) | probe reported off=0, stop=0, 0 sweeps, res=1.5e-3 | off exact, 1 sweep, res=5.6e-9 |
| `probe_counts_all_rows` (14 sizes) | N86: 5.34e-6 vs true 6.52e-6 (−18%) | reported == host-exact for all 14 |
| `inner_cap_reports_failure` (INNER_MAX=1, N24) | stop=0 at rel=0.376 | stop=2; latch verified; NaN→stop=4 after window clear |

Fixes: shared `bj_wg_sum` arbitrary-WG reduction at all 6 sites in
`gpu_block_jacobi.cl`; `n<=PB` path requires actual tolerance; diag records
latched (first failure wins until host clears the window); direct-kernel
Fermi tail runs for all certifiable stops (≤2), closing the stale-`occ_w`
hole where `check_jacobi` accepts stop=1/2 below rel 1e-4; direct-kernel WG
clamped to a power of two below `max_work_group_size`.

Test suite after fix (release, RTX 3090, mio-1-1):
gpu_tiled_jacobi 8/8 · gpu_eigenproblem 10/10 · gpu_scc 6/6 ·
gpu_scc_kernels 10/10 · gpu_dftb 8/8 · gpu_forces 5/5 ·
gpu_hbond_physics 23/23 (incl. GC PES/forces parity, max|dF|=5.5e-6 Ha/Å).
`gpu_diagonalization` has 2 failures (`*_n2`) that are **pre-existing on
HEAD** — N=8 degenerate-subspace elementwise eigenvector compare in the
untouched n≤64 `local_jacobi_blocks` path; eigenvalues pass.

### T02-lite baseline — direct vs block, scan400 harness

Command (both modes): `RUST_DFTB_SK_DIR=<mio-1-1> RUST_DFTB_BENCH_SYSTEMS=GC,DTH
RUST_DFTB_PROF=1 cargo test --release --test gpu_scc_bench
test_gpu_scc_scan400_benchmark -- --ignored --nocapture --test-threads=1`;
block adds `RUST_DFTB_EIGSOLVER=block`. 3 scc runs averaged (per-run
`reset_q0`, warm basis), `eval(true)` timed once, CPU = measured f64
single-point.

**Direct (production default):**

| system | N | batch | scc ms | iters | ms/iter | sys/s | evalF ms | cpu 1pt ms | speedup | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| GC  |  86 |   1 |   20.5 |  16 |   1.28 |  48.8 |   0.62 | 12.7 |  0.6x | 0 |
| GC  |  86 | 100 |  144.8 |  72 |   2.01 | 690.7 |   2.02 | 12.7 |  8.8x | 0 |
| GC  |  86 | 400 |  417.9 |  80 |   5.22 | 957.2 |   5.23 | 12.7 | 12.2x | 0 |
| DTH | 246 |   1 |  268.0 |  16 |  16.75 |   3.7 |   6.02 | 132.5 |  0.5x | 0 |
| DTH | 246 | 100 | 1519.6 |  16 |  94.97 |  65.8 |  44.68 | 132.5 |  8.7x | 0 |
| DTH | 246 | 400 | 6777.3 |  16 | 423.58 |  59.0 | 267.33 | 132.5 |  7.8x | 0 |

**Block (`RUST_DFTB_EIGSOLVER=block`, post-fix):**

| system | N | batch | scc ms | iters | ms/iter | sys/s | evalF ms | cpu 1pt ms | speedup | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| GC  |  86 |   1 |   52.3 |  16 |   3.27 |  19.1 |   0.68 | 13.5 | 0.3x | 0 |
| GC  |  86 | 100 |  379.1 | 100 |   3.79 | 263.8 |   1.72 | 13.5 | 3.5x | 0 |
| GC  |  86 | 400 |  579.6 | 100 |   5.80 | 690.1 |   5.93 | 13.5 | 9.3x | **1** |
| DTH | 246 |   1 |  293.8 |  16 |  18.37 |   3.4 |   4.48 | 132.5 | 0.5x | 0 |
| DTH | 246 | 100 |  843.3 |  16 |  52.71 | 118.6 |  40.98 | 132.5 | 15.7x | 0 |
| DTH | 246 | 400 | 3136.3 |  16 | 196.02 | 127.5 | 266.23 | 132.5 | 16.9x | 0 |

Stage profile at batch=400 (host ms per launch, avg over 3 runs):

| stage | GC direct | GC block | DTH direct | DTH block |
|---|---|---|---|---|
| scc.jacobi      | 3.49 | 4.68 | 243.6 | 72.6 |
| scc.hscc        | 0.60 | 0.58 | 39.6  | 21.0 |
| scc.density     | 0.25 | 0.21 | 36.3  | 36.4 |
| scc.eigh_finish | 0.26 | 0.24 | 25.8  | 25.3 |
| scc.diis        | 0.25 | 0.26 |  0.58 | 0.54 |
| scc.fermi_occ   | — (fused) | 0.23 | — (fused) | 0.64 |
| scc.select_occ  | — | 0.15 | — | 0.26 |
| scc.gemm_th+th2 | 0.26 | 0.26 |  8.5 | 8.6 |
| scc.mulliken    | 0.10 | 0.10 |  2.3 | 2.3 |

Reading: at N246 the one-WG block eigensolve is ~3.4× faster per launch
(72.6 vs 243.6 ms) and cuts total scc wall 2.2×; at N86 it is slightly
slower per launch (4.68 vs 3.49) *and* pays extra `fermi_occ`+`select_occ`
kernels that direct fuses. Block GC also ran to the 100-iter cap with one
failed replica (direct: 80 iters, 0 failures) — iteration count is a
workload difference, not a pure speed difference; per-iter is the fair
comparison. At both sizes the eigensolve no longer dominates once block is
used: for DTH-block, `density`+`eigh_finish`+`hscc` ≈ 49% of host time —
the T03 targets.

