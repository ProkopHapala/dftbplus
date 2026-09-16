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

**Measured follow-up (OPENBLAS_NUM_THREADS=1 rerun, same test):** pinning
BLAS to one thread made the CPU baseline *faster*, not slower — AZA
12.18→9.42 ms, DTH 163.9→131.3 ms, formic 2.49→1.93 ms, GC 12.94→12.20 ms,
AT 13.45→12.15 ms. OpenBLAS threading is pure overhead at these matrix
sizes. So the published speedup column was slightly *optimistic* (CPU
handicapped ~10–25%), and the 1-thread numbers are the controlled baseline.
The 82-slot fingerprint below reproduces on the rerun: G400 = 92× (AZA),
84× (GC), 86× (AT), 57× (diazaphenalene), 17× (DTH), 208× (formic),
226× (H2O). Production Jacobi runs JACOBI_PREC=0 (pure f32) — the
FP64-precision modes exist only as A/B levers, they are not the cost.

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

**WG-size numbers at N=86 (why 512 is wrong here):** the kernel's
independent unit is the pair — jpair = jn/2 = 43 rotations per round; the
strip update is ~6·N ≈ 516 element updates spread over 512 threads
(~1 elem/thread), the rotation-parameter phase runs ~43 threads of 512, and
the packed prologue reduce is a 9-level barrier tree. At WG=128 the same
rounds give ~4 elems/thread and a 7-level tree. Local memory per WG at
WG=512 is ~13.3 KB: `dred`+`dred2` f64 (8 KB) + `le[256]` (1 KB) are
**Fermi-tail scratch only** — dead for the entire eigensolve; the live
Jacobi footprint is just `rot_*` + `reduce` ≈ 4 KB. Residency ceiling on
GA102 (1536 thr/SM, 48 KB OpenCL local): WG=512 → ≤3 WGs/SM (measured
throughput says ~1 — register-bound); WG=128 → ≤12 by threads, ~6 by local.
Splitting the Fermi tail out of the kernel removes ~9 KB of dead local per
WG. Beyond WG sizing, **sub-group packing** (e.g. 4 × 128-thread replicas
inside one 512-thread WG, or warp-per-replica for small N) is the direct
way to raise replicas/SM — the current mapping caps residency at the SM
count itself.

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

## Part 3 — Block-Jacobi solver implemented + first true device-time profile (2026-09-16 pm)

A new eigensolver architecture (`block_jacobi_1wg`, `src/qmqm/gpu_block_jacobi.cl`,
selected by `RUST_DFTB_EIGSOLVER=block`; the direct kernel stays the default) plus
the first `RUST_DFTB_PROF=evt` per-stage device-time profile of the production
SCC pipeline. The profile **changes the bottleneck ranking** — several beliefs
from Parts 1–2 need revision.

### Measured per-stage device time (marker-event spans, `RUST_DFTB_PROF=evt`)

Caveat: values accumulate over all batch sizes run (1/100/400); `dev%` shares
and per-call magnitudes are the reliable quantities. Marker spans include
inter-stage idle, so small stages are mildly inflated by queue starvation.

**GC (n=86), direct vs block:**

| stage | direct dev | direct % | block dev | block % |
|---|---:|---:|---:|---:|
| scc.jacobi | 840 ms | **79.6%** | 1255 ms | **82.4%** |
| scc.diis | 54 | 5.1 | 55 | 3.6 |
| scc.eigh_finish (warm = `snormalize_batched`) | 50 | 4.7 | 50 | 3.3 |
| scc.density (`build_density_occ_batched`) | 43 | 4.0 | 45 | 3.0 |
| scc.fermi_occ (block only; direct has fused tail) | — | — | 43 | 2.8 |
| scc.gemm_th + gemm_th2 (cᵀHc projection) | 49 | 4.6 | 49 | 3.2 |
| scc.hscc / mulliken / extract / occ | ~17 | ~1.6 | ~18 | ~1.2 |
| **wall scc b=400** | **352 ms, 80 it, 0 fail** | | **508 ms, 100 it, 2 fail** | |

**DTH (n=246), direct vs block:**

| stage | direct dev | direct % | block dev | block % |
|---|---:|---:|---:|---:|
| scc.jacobi | 13 697 ms | 77.2% | 3 872 ms (60.5/call) | **49.9%** |
| scc.density | 2 022 | 11.4 | 1 929 (30.1 ms/call) | **24.9%** |
| scc.eigh_finish (snorm) | 1 394 | 7.9 | 1 326 (20.7 ms/call) | **17.1%** |
| scc.gemm_th + gemm_th2 | 471 | 2.7 | 438 | 5.6 |
| scc.mulliken | 122 | 0.7 | 118 | 1.5 |
| scc.fermi_occ (block only) | — | — | 27 | 0.4 |
| scc.hscc / diis / rest | ~40 | ~0.2 | ~40 | ~0.6 |
| **wall scc b=400** | **5 916 ms, 16 it** | | **2 584 ms, 16 it (2.3×)** | |

### Fact 1 — the production Jacobi does REAL sweeps every iteration; the "warm = 0 sweeps" assumption was wrong for 2D scans

Standalone warm probe (exactly-diagonal input): 0.04 ms/launch (direct, n=86,
b=400). Production `scc.jacobi`: **3.13 ms/call** (direct) / **4.48 ms/call**
(block) — i.e. ~1–3 real sweeps per iteration, every iteration. Each charge
update re-excites the off-diagonals of `hp = cᵀH_scc c` by O(‖ΔH_scc‖), which is
not below `JACOBI_OFF_TOL=1e-6`. The earlier "p90 = 0 sweeps" observation held
for a different workload (near-equilibrium), not the 2D scan far corners.
**The warm path is not free — it is a small but real eigensolve each step.**

### Fact 2 — block Jacobi fixed the large-N problem, not the medium-N one

- n=246: 214 → 60.5 ms/call (3.5×); solves the global-traffic bottleneck exactly
  as designed (pivot-only local memory, row-owner writeback, ~8.4 KB/WG).
- n=86: 3.13 → 4.48 ms/call — **slower**. At n=86 the direct kernel is already
  latency- not bandwidth-bound (A fits in L2), and the block kernel pays more
  per-WG serial latency: nb=6 → 15 serial block-pair ops, each an inner
  Brent–Luk solve with barriers, while 64 of 96 threads idle during each inner
  solve. Occupancy is not the limiter here — per-WG critical path is.

### Fact 3 — after the block fix, the *supporting* O(N³) kernels are the new bottleneck at N=246

Both are memory-streaming implementations of what should be tiled GEMMs:

- **`snormalize_batched`** (warm `eigh_finish`, runs EVERY iteration): one WG per
  (replica, column) — 98 400 WGs at b=400/n=246, each streaming the **full S
  matrix (242 KB)** to compute `t_s = S·c_k` then `cᵀt_s`. ≈ **24 GB** of global
  traffic per call → measured 20.7 ms ≈ 1.2 TB/s effective — pure bandwidth.
  Exists only to arrest f32 column-norm drift of the warm basis.
- **`build_density_occ_batched` under Fermi smearing**: `use_w=1` ⇒ `n_occ=n`
  → a **full rank-N outer product** (D = 2·C·diag(w)·Cᵀ), each thread loops all
  n columns with 2 gathers/FMA → ≈12 GB/call → measured 30.1 ms ≈ ~200 GFLOPS.
  A column-scale + tiled SYRK-class kernel would be ~10× faster; and at
  kT=0.002 only ~n_occ+few columns have w>1e-6 — the occupied-index-list
  machinery (`loi`) already exists unused on this path.
- `gemm_th`/`gemm_th2` (the projection GEMMs) are already tiled and cheap
  (2.8–5.6%): the pipeline's own GEMMs show the headroom is real.

### Fact 4 — lockstep tax is real and measurable

`iters` = max over 400 replicas (GC: 80 direct / 100 block — **2 marginal
far-corner replicas drag the whole batch to the cap**). Parked replicas still
occupy WGs every launch (cheap early-exit, but the scheduler slot is spent).
GC block-mode: iters 80→100 vs direct — the stragglers cost +20 iters × 400
replicas of every kernel.

### Fact 5 — local-memory occupancy measured (pyopencl build query)

| kernel | WG | local/WG | → WGs/SM by local(48KB) | by threads(1536) |
|---|---:|---:|---:|---:|
| direct (fused Fermi tail) | 512 | 13.0 KB | 3 | 3 |
| block n=86 | 96 | 8.9 KB | 5 | 16 |
| block n=246 | 256 | 9.5 KB | 5 | 6 |

The block slimming did its job (dead Fermi scratch gone: −4.1 KB). But
`CL_KERNEL_PRIVATE_MEM_SIZE` reports 0 on this NVIDIA driver (unavailable),
and `CL_KERNEL_WORK_GROUP_SIZE` reports **256 even for the WG=512 direct
build** — either a driver quirk or the kernel is register-limited to ≤256
threads/WG (launches at 512 demonstrably succeed and give correct results, so
the report is at minimum unreliable). **Registers are probably the true
residency limiter — needs `ncu` (`launch__registers_per_thread`,
`sm__warps_active`) on a real launch to settle.** If block @n=86 gets ~4–6
WGs/SM it is near the residency ceiling and per-WG latency (Fact 2) is what
remains.

### Roofline for the "≥100× vs 1 CPU thread" target

CPU-sequential baseline for the same 400 GC points ≈ 4.9 s → 100× needs
**≤49 ms total ≈ ≤0.6 ms/iter** at ~80 iters. Measured today: 4.4–5.1 ms/iter
wall. Even a *zero-cost* eigensolver leaves ~0.9 ms/iter of non-eigen device
work at n=86 (diis 0.20 + snorm 0.18 + density 0.16 + fermi 0.15 + gemms 0.18 +
mulliken 0.03 + hscc 0.02) → ~70 ms → **~70× ceiling** with the current
supporting kernels. To actually reach ~100× at n≈85, ALL of these must hold:

1. eigensolver ≈ probe-cost or replaced by a GEMM-class method (below),
2. density + snorm at tiled-GEMM efficiency (~5–10× each),
3. tail compaction so iters ≈ mean not max.

At n=246 the same analysis gives a ~2–3× further win on top of the current
2.3× block gain, i.e. ~50× vs CPU-seq — N=246 may also want >1 WG per replica
(block-pairs are independent enough for a 2-WG split) or accept it.

### Ranked hypotheses / candidate levers (for discussion)

- **H-A (biggest single win now, N≥120):** rewrite `snormalize_batched` +
  `build_density_occ` as GEMM-class kernels. Density: column-scale `c` by
  `√w` then lower-triangle tiled GEMM — reuses `matmul_tiled` machinery.
  Est. −35–40% dev at n=246. Alternative cheaper still: cut `occ_w` at a
  threshold → `n_eff` ≈ 35–45 columns (vs 246) on the existing rank-update
  kernel — controlled error, must be measured vs PES parity.
- **H-B (cheap):** snorm drift is *slow* — amortize to every k-th iteration
  or gate it on a cheap orthonormality check, not every iteration.
- **H-C (algorithmic, the real "≥100×" candidate):** in the warm near-diagonal
  basis the SCC loop needs the *projector response*, not eigenpairs. Options:
  (i) first-order density update `ΔD_ij = (w_j−w_i)·hp_ij/(ε_j−ε_i)` — O(n²)
  elementwise, degeneracies handled via the Fermi window; (ii) 1–2
  Newton–Schulz/Chebyshev projector steps as dense GEMMs (the sparse path
  already does DMM/TC2 for exactly this reason); (iii) CheFSI/LOBPCG on the
  n_occ subspace (n²·n_occ ≪ n³). Each removes sweeps AND their barriers;
  Fermi μ comes from Tr(DS)=N_e instead of eigenvalues. Numerics must be
  re-validated against the PES/force parity test — this changes *what* is
  converged, not just how fast.
- **H-D (structural):** `active_ids` compaction — parked replicas stop
  consuming WG slots in the tail; recovers up to ~2× on straggler-bound
  batches (GC). 
- **H-E (structural, end-game):** persistent per-replica WG executing the
  WHOLE SCC iteration (or solve) without kernel boundaries — eliminates the
  ~10 boundary drains/iter AND the lockstep (each WG exits on its own
  convergence). This is what "throughput = n_replicas" actually requires.
  Cost: every stage becomes WG-local (a 96-thread GEMM is fine when 400 WGs
  run concurrently); DIIS already per-replica on device.
- **H-F (immediate measurement):** `ncu` on both kernels at production batch —
  registers/thread, achieved occupancy, warp stall reasons. Settles whether
  n=86 is latency- or residency-bound. Also A/B `B=8` pivot and `WG∝N`
  variants at fixed kernel.
- **H-G (correctness, open):** GC block-mode 2-replica marginal failure at the
  far corner — diagnose stop-code/off-norm trajectory, do NOT paper over.

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
   sufficient for a multisolver. Concrete leaks measured in the kernel:
   ~9.2 KB/WG of dead Fermi-tail local scratch (`dred`/`dred2`/`le`/`lmu`)
   inflating every eigensolve WG; WG=512 vs only 43 independent pairs/round
   at N=86. Candidate fixes: WG ∝ N, Fermi tail split or once-per-solve,
   sub-group packing (multiple replicas per WG).
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
