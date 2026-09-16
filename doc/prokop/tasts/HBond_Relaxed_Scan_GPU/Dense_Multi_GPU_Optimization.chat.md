

---

# USER

there are some updates from my coding agent, what you think?

I think he really does not push enoguh on the performance accepting that desnity-matrix build is slow, but the goal is optimize those as well, and remove the harnsss overhead, it is not acceptable to do zeros cost diagonalization and kill the performance in stupid harness or density assembly

---

Let me first check if the lib compiles now (your sparse edit was mid-flight), then get real per-kernel numbers rather than speculate — the plan already has `prof_tick` instrumentation.

Lib compiles again. Let me check the profiler modes:

The profiler exists but `prof_report` isn't called in the bench — add it to the scan400 bench so I get the real per-stage GPU breakdown:

The profiler gives real numbers. This run was **direct** mode (no `RUST_DFTB_EIGSOLVER` set) — let me get the same profile in block mode for the A/B, and check what `scc.density`/`scc.eigh_finish` actually run:

Now let me get the block-mode profile for comparison — running it, and inspecting what `density`/`eigh_finish` actually launch while it goes:

The run completed — let me get the GC profile section from the log:

Now let me see what `eigh_finish` and `density` actually enqueue — they're now the #2/#3 device costs at N=246 (30 + 21 ms/call, bigger than the Jacobi in block mode):

I have the real device-time breakdown now — and it changes the picture substantially. Let me look at where to write this in the report:

The picture is now clear and it changes the bottleneck ranking. Let me also get the actual register/local footprint of both kernels — the user's occupancy question deserves a measured answer, not inference:

Add the resource query to the A/B bench — this directly answers the occupancy question:

The lib is mid-edit again (`snapshot_p0_x0` missing). For the resource query I don't need the lib — check if pyopencl can compile the kernel directly:

pyopencl is available — I can compile the kernels directly and query resources. Let me check the source templating format:

I can replicate the rendering in pyopencl directly. Let me check the template `#define` defaults first:

Driver reports are in (NVIDIA doesn't expose private-mem via this API — `private=0` is a non-answer; `ncu` exists for later). Local memory: direct 13.0KB, block 8.9–9.5KB. Now let me write the analysis into the report — this is the factual basis for your LLM discussion:

Now I'll append the new measured analysis as Part 3 of the report:

Now update the manifest §16 checklist with the measured findings:

Analysis written to the report (`Part 3`) + manifest §16.G. Here's the honest answer — **the local-memory/occupancy theory is only part of it; the measured profile shows three bigger problems**:

## What I measured (RUST_DFTB_PROF=evt — true device time per stage, direct vs block)

**Fact 1 — the Jacobi does real sweeps EVERY iteration.** Standalone warm probe = 0.04 ms/launch, but production `scc.jacobi` = **3.1–4.5 ms/call**. The "warm ≈ 0 sweeps" assumption was wrong for scan workloads: each charge update re-excites off-diagonals of `cᵀH_scc c` above the 1e-6 tolerance → ~1–3 real sweeps per iteration, every iteration.

**Fact 2 — block Jacobi is not occupancy-starved at n=86; it's *latency*-bound.** Per-call it went **up** (4.48 vs 3.13 ms) — 15 serial block-pair ops with barriers while 64/96 threads idle. At N=246 the opposite: 214→60.5 ms/call, the traffic fix worked exactly as designed.

**Fact 3 — the new bottleneck at N=246 isn't the eigensolver at all.** After the block fix: jacobi 50%, **`density` 25% + `snormalize` 17%** — and both are memory-streaming kernels doing GEMM-class math badly:
- `snormalize_batched`: one WG per (replica, column) — each of 98,400 WGs streams the **full 242KB S matrix** → ~24 GB/call. It only exists to arrest f32 basis drift.
- `build_density_occ` under Fermi: full rank-N outer product, ~12 GB/call at ~200 GFLOPS — a tiled SYRK would be ~10× faster, and only ~40 columns have w>1e-6 anyway.

**Fact 4 — lockstep tax:** GC's 2 marginal far-corner replicas drag all 400 to the 100-iter cap (80→100 iters under block).

**Fact 5 — occupancy measured** (pyopencl, lib-independent): local/WG = 13.0KB direct → **3 WGs/SM**, 8.9–9.5KB block → **5 WGs/SM** by local, 16 by threads at n=86. NVIDIA doesn't report registers via this API and `ncu` is installed — that's the remaining measurement. So your instinct was half-right: local memory WAS wasting residency in the direct kernel (dead Fermi scratch now gone), but registers likely bind next.

## Why 100× isn't reachable yet — the arithmetic

CPU-seq 400 GC points ≈ 4.9 s → 100× needs **≤0.6 ms/iter**. Even a *zero-cost* eigensolver leaves ~0.9 ms/iter of supporting kernels → ~70× ceiling. To reach 100× all three must hold: eigensolver≈free, density+snorm at GEMM efficiency, tail compaction. That's now in the report as ranked hypotheses H-A…H-G, including the boldest one for your LLM discussion:

**H-C: the warm iteration may not need diagonalization at all** — in the near-diagonal warm basis, SCC needs the *projector response* `ΔD_ij = (w_j−w_i)·hp_ij/(ε_j−ε_i)`, or 1–2 NS/Chebyshev GEMM steps, or CheFSI on the n_occ subspace. No sweeps, no barriers, μ from Tr(DS)=N_e. That's the change that could actually get ~100× — but it changes what gets converged, so it needs PES-parity revalidation.

One flag: your in-progress `sparse_dftb.rs` edit broke the lib build again mid-run (`snapshot_p0_x0` missing at :1332) — the PES-parity check in block mode is still queued behind that.



---

# Chat GPT 5.6 sol


I agree with your criticism. The new measurements are valuable, but I would **not** accept the agent’s framing that after fixing Jacobi we are naturally left with 0.9 ms/iteration of “supporting work.” That 0.9 ms is mostly evidence of **more bad kernels**, not an irreducible floor.

The N=246 experiment is especially useful: block Jacobi improves 214 → 60.5 ms/call, yet wall time improves only 2.3× because `density` and `snormalize` immediately expand to ~42% of device time.  That is exactly what should happen during serious optimization: remove one bottleneck and expose the next one. We should keep going until the remaining cost is actual necessary arithmetic.

## The density build should probably disappear from the SCC loop entirely

This is the biggest thing I think the agent missed.

Currently it forms

$$
D_{\mu\nu}
=
2\sum_k f_k C_{\mu k}C_{\nu k}
$$

every SCC iteration, and then Mulliken contracts \(D\) with \(S\). The report correctly observes that the present implementation is an untiled full-rank outer-product and is very slow. 

But simply replacing it by SYRK is already thinking too locally.

For SCC we do **not need \(D\)**.

What SCC needs is the atomic Mulliken population:

$$
N_A
=
\sum_{\mu\in A}(DS)_{\mu\mu}.
$$

Define

$$
T = SC.
$$

Then exactly

$$
N_A
=
2
\sum_{\mu\in A}
\sum_k
f_k
C_{\mu k}T_{\mu k}.
$$

So after the eigensolve we can compute Mulliken charges directly from \(C\), \(T=SC\), and occupations:

```text
for atom A:
    pop = 0
    for mu in orbitals(A):
        for k:
            pop += 2 * occ[k] * C[mu,k] * SC[mu,k]
```

That is only

$$
O(N^2)
$$

rather than constructing an \(O(N^2)\) matrix through \(O(N^3)\) work and then immediately contracting it away.

**Build \(D\) only once, after SCC convergence, when forces/final observables genuinely need it.**

That alone should remove `scc.density` from the hot loop, not merely make it 10× faster.

---

## And `snormalize` can probably disappear as an \(O(N^3)\) operation too

This one is even more offensive.

The report says correctly that it exists only to compensate f32 drift of the warm basis, yet the present implementation streams the whole \(S\) matrix separately for every column. At N=246 it moves roughly 24 GB/call and takes 20.7 ms. 

But consider the warm basis mathematically.

We start with

$$
C^TSC=I.
$$

Jacobi applies an orthogonal transformation:

$$
C' = CU.
$$

Therefore

$$
C'^TSC'
=
U^T(C^TSC)U
=
U^TU
=
I.
$$

So in exact arithmetic **there is nothing to normalize after an SCC Jacobi step**.

It is purely an f32 drift repair.

That should never justify an \(O(N^3)\) operation every iteration.

### Maintain \(T=SC\) together with \(C\)

This gives a particularly clean solution.

At geometry initialization, calculate once:

$$
T=SC.
$$

Whenever Jacobi rotates

$$
C\leftarrow CU,
$$

also rotate

$$
T\leftarrow TU.
$$

This is cheap because the same \(U\) is already sitting in local memory. In the block kernel, each row thread simply transforms a second row strip:

```text
C[row,PQ]  = C[row,PQ]  * U
SC[row,PQ] = SC[row,PQ] * U
```

Then column S-norm is merely

$$
g_k =
\sum_\mu C_{\mu k}T_{\mu k},
$$

which costs only \(O(N^2)\).

If necessary:

$$
C_{\mu k}\leftarrow C_{\mu k}/\sqrt{g_k},
$$

$$
T_{\mu k}\leftarrow T_{\mu k}/\sqrt{g_k}.
$$

So the current expensive `snormalize_batched`

$$
O(N^3)
$$

becomes a trivial

$$
O(N^2)
$$

column dot/scale.

And the same maintained \(T=SC\) gives us the direct Mulliken formula above.

That is, one architectural change potentially eliminates **both** of the new big bottlenecks.

---

# The SCC iteration I now want

Instead of approximately

```text
Hscc
C^T H
(C^T H) C
Jacobi
S-normalize
Fermi
build D
Mulliken(D,S)
DIIS
```

I want:

```text
Hscc
  ↓
C^T H
  ↓
(C^T H) C
  ↓
Jacobi:
    C  <- C U
    SC <- SC U
  ↓
Fermi
  ↓
Mulliken directly from C, SC, occ
  ↓
DIIS
```

That's it.

No density matrix.

No \(S C\) recomputation.

No O(N³) normalization.

Then after convergence:

```text
weighted-SYRK -> D
weighted-SYRK -> W
forces / energy
```

Now the inner SCC cost is dominated by exactly what we expect:

$$
C^THC + \text{eigensolve}.
$$

That is a much healthier architecture.

---

## And the final density/W kernels should still be optimized properly

Even though they leave the SCC loop, we still shouldn't tolerate the present implementation.

For a final state:

$$
D_{ij}=2\sum_k f_k C_{ik}C_{jk}
$$

and

$$
W_{ij}=2\sum_k f_k\rho_k C_{ik}C_{jk}.
$$

Implement one generic:

```text
weighted_syrk(C, weights, Out)
```

with tiled K reuse:

```c
for K-tile:
    load C[row_tile,K] -> local A
    load C[col_tile,K] -> local B
    load weight[K]     -> local w

    acc_ij += sum_k w[k] * A[i,k] * B[j,k]
```

Compute only the lower triangle and mirror it.

That should look much more like your already-efficient tiled GEMM than the current rank-update kernel. The projection GEMMs are only 2.8–5.6% of device time at N=246, which is direct evidence that good tiled dense algebra is not the problem. 

---

# I distrust one statement in the agent's density analysis

It says:

> at kT=0.002 only ~35–45 columns have w>1e-6

That needs verification.

Your Fermi solver satisfies

$$
\sum_k f_k=n_{\rm occ},
\qquad
0\le f_k\le1.
$$

Therefore if `n_occ` is, say, 80–120, it is mathematically impossible for only ~40 orbitals to carry essentially all the occupation.

Maybe the agent meant only ~35–45 **frontier/noninteger** states, or is looking at some other array.

Before introducing any occupation cutoff, dump:

```text
n_occ
count(w > 1e-2)
count(w > 1e-4)
count(w > 1e-6)
min/max/frontier weights
```

for GC and DTH.

I would first make the exact weighted-SYRK fast. Approximate truncation should not be needed.

---

# The harness conclusion in the report is also slightly wrong

The report currently says that two GC stragglers going from 80 to 100 iterations cost

> “+20 iters × 400 replicas of every kernel.” 

That's not actually true with your current device `active` mask.

The 398 converged systems early-return from the expensive body.

What you still pay is:

* dispatching the full launch domain;
* scheduling WGs which immediately discover they're inactive;
* especially for tiled GEMMs, thousands of dead tile-WGs;
* very poor GPU occupancy because only two systems contain real work.

That distinction matters because it tells us precisely what to implement:

```text
active mask     -> stops work immediately inside a chunk
active_ids[]    -> removes dead WGs between chunks
```

I strongly support the **slot** design we discussed.

After each chunk:

```text
400 -> 213 -> 87 -> 21 -> 4
```

change the launch geometry accordingly.

Do not copy matrices. Just:

```c
sid = active_ids[logical_sid];
```

Then the remaining four hard replicas genuinely launch four systems rather than launching a 400-system grid full of no-ops.

And for scans larger than the saturation pool, refill completed physical slots with pending geometries.

---

# I would push back strongly on H-E

The agent suggests as an “end-game”:

> persistent per-replica WG executing the whole SCC iteration/solve. 

I would **not pursue this now**, and possibly never.

We have just learned the hard way that fusing unrelated work into one kernel can destroy occupancy because the kernel inherits the union of:

* registers,
* local memory,
* control paths,
* scratch arrays.

The Fermi-in-Jacobi mistake is exactly that lesson.

A giant SCC kernel would contain:

```text
Hscc
GEMM
GEMM
Jacobi
Fermi
Mulliken
DIIS
```

with radically different natural thread geometries and scratch requirements.

That could easily produce another WG512/Fermi situation at a much larger scale.

Your current philosophy is better:

> **small number of coarse kernels, each with a natural memory layout, persistent buffers, no host synchronization, active-slot compaction.**

Kernel launches are cheap compared with turning every kernel into an occupancy monster.

---

# I would also demote H-C substantially

The agent's projector-response / Newton–Schulz / CheFSI ideas are scientifically interesting. 

But they change the numerical algorithm.

Before touching them, we currently have huge exact-equivalent wins available:

1. eliminate inner-loop density construction using direct Mulliken from \(C,SC\);
2. eliminate \(O(N^3)\) S-normalization by carrying \(SC\) through Jacobi;
3. optimize final D/W as tiled weighted-SYRK;
4. active-ID compaction;
5. fix WG sizing / direct-versus-block crossover;
6. remove remaining host allocations/launch overhead.

Only after all of those should we ask whether eigendecomposition itself needs replacing.

Otherwise we won't know whether CheFSI is solving a genuine bottleneck or compensating for bad plumbing.

---

## Direct versus block Jacobi now looks clearer too

The new measurement is actually very useful:

$$
N=86:
\quad
3.13\ {\rm ms\ direct}
<
4.48\ {\rm ms\ block}
$$

while

$$
N=246:
\quad
214
\rightarrow
60.5\ {\rm ms}.
$$



So I wouldn't force one eigensolver architecture universally.

Likely something like:

```text
N ~ 65–100:
    direct cyclic
    WG ~= round_up(N,32)
    minimal local scratch

N ~ 120–300:
    block Jacobi
    B=16/24
    WG ~= round_up(N,32)
```

with the crossover measured rather than guessed.

And the N≈86 direct kernel still deserves the changes we discussed:

```text
WG512 -> ~WG96
Fermi out
tiny local footprint
production-batch tuning
```

The block solver being slower there does **not** mean N≈86 is optimized; it only means this particular block implementation doesn't win.

---

# One more thing: the report now contains contradictory history

Part 2 still says the dominant Jacobi stage often performs zero sweeps and treats that as the main production anomaly. 

Part 3 now measures that the actual 2D scan workload does **1–3 real sweeps every SCC iteration**, and explicitly says the previous assumption applied to another near-equilibrium workload. 

That's fine as a lab notebook chronology, but not fine as a canonical performance report.

I'd mark Part 2 explicitly:

```text
SUPERSEDED for scan workloads by Part 3 measurements.
Zero-sweep behavior applies to near-equilibrium/warm-probe cases only.
```

Likewise open item 3 at the end is now stale because Part 3 already resolved it. 

---

## My revised performance priority

The next optimization wave I would give the coding agent is:

1. **Add persistent `SC` alongside `C`.** Build `SC=S*C` once when geometry/basis changes. Every Jacobi rotation must rotate both C and SC.

2. **Remove `snormalize_batched` from the SCC loop.** Replace it with O(N²) `metric_diag_and_scale(C,SC)` if needed; first A/B running it every 4/8 iterations or final-only.

3. **Remove `build_density_occ_batched` from SCC.** Compute Mulliken populations directly:

   $$
   N_A=2\sum_{\mu\in A,k} f_k C_{\mu k}(SC)_{\mu k}.
   $$

4. **Implement exact tiled `weighted_syrk` for D and W**, used only on finalized states / force evaluation.

5. **Implement `active_ids` compact launch domains.** Record \(\sum N_{\rm active}\), not max-iteration×batch.

6. **Then optimize eigensolver throughput**: WG≈N, Fermi separate, direct/block crossover, B16/B24.

7. **Only after that consider H-C projector methods.**

That is much more aggressive than accepting the report's current "~70× ceiling." That ceiling is only the ceiling of the **current inefficient support pipeline**. The measured data themselves show that we still have large, obvious, mathematically exact performance wins available.

---

## 2026-09-16 — Source review: corrections and implementation design

This appendix checks the discussion against the current working tree, benchmark report Part 3 and manifest §16.G. Original text is preserved. **Analysis only: no solver edits or GPU benchmark runs were made for this review.** The reduction schedule below was reproduced arithmetically on the host. Concurrent uncommitted changes exist; record source hashes in subsequent measurements.

Implementation agents: follow [Dense_Multi_GPU_Optimization.tasks.md](Dense_Multi_GPU_Optimization.tasks.md). `[*]` denotes an established finding, not user acceptance of an implementation. `[ ]` denotes pending work.

### A. Priority decisions

- [*] The direct-population proposal is mathematically sound and removes unnecessary density construction from SCC.
- [*] Current block reductions are incorrect for WG=96 and other non-power-of-two sizes. Correctness must precede medium-N tuning.
- [ ] First implement **recomputed SC + paired normalization + direct populations** using existing tiled GEMM. This provides a clean comparison without changing Jacobi.
- [ ] Then test **transported SC** separately. Carrying an extra matrix through every sweep can cost more than one GEMM and needs a restoring invariant in f32.
- [ ] Introduce explicit D/W validity and lazy final materialization in the same patch that removes their SCC-loop producers.
- [ ] Compact active launch domains, improve Fermi and tune dispatch after comparable profiles are available.
- [ ] Defer slot refill, all-SCC persistent kernels and projector-response solvers until the exact-equivalent changes are measured.

### B. Correctness blockers in block Jacobi

`gpu_eigen.rs::block_jacobi_wg` rounds N up to a multiple of 32. The repeated reduction in `gpu_block_jacobi.cl` uses `o=lsz>>1; o>0; o>>=1` with `reduce[lid]+=reduce[lid+o]`. This requires a power-of-two workgroup.

| WG | Correct sum of ones | Current schedule |
|---:|---:|---:|
| 32 | 32 | 32 |
| 64 | 64 | 64 |
| 96 | 96 | 64 |
| 128 | 128 | 128 |
| 160 | 160 | 128 |
| 192 | 192 | 128 |
| 224 | 224 | 128 |
| 256 | 256 | 256 |

WG96 drops lanes `2,5,8,...,95`. Global probe, pivot norms and sweep stopping are affected. Random matrices can hide this because both numerator and denominator are biased.

- [ ] Regression: N86, `A=I`, `A[2,5]=A[5,2]=0.01`, V=I. Current probe can report zero off-norm/zero sweeps. True off-norm is `sqrt(2)*0.01`; unchanged-basis relative eigen residual is about `1.525e-3`.
- [ ] Correct all reductions or explicitly restrict WG to powers of two as a temporary control. Re-run N84/86/87 parity and SCC before trusting old block comparisons. N246/WG256 is unaffected by this particular defect.
- [ ] Fix `n<=PB`: `stop=isfinite(off_cur)?0:4` currently reports success even when the inner sweep cap leaves a finite residual above tolerance.
- [ ] Audit `check_jacobi` finite/residual validation and accepted floor/stall/cap outcomes. Direct fused Fermi executes for stop zero, while host logic can accept some nonzero stops: accepted bases must have freshly valid occupations.
- [ ] Keep first-failure context until checked; later solves must not erase evidence of bad output already consumed by SCC.

When editing scratch reductions, prove uniform barrier control and read-before-reuse ordering. The pivot-gather barrier already precedes the next pivot reduction overwrite; absence of a barrier immediately after the outer norm read alone is not proof of a race.

### C. What the current measurements establish

See [benchmark report](../../reports/2026-09-16_dense_gpu_pes_forces_benchmark.md), especially Part 3. Preserve historical values and append corrected measurements.

| Batch400 workload | Direct | Block |
|---|---:|---:|
| GC N86 SCC wall | 352 ms, 80 iterations, 0 failures | 508 ms, 100 iterations, 2 failures |
| DTH N246 SCC wall | 5916 ms, 16 iterations | 2584 ms, 16 iterations |
| DTH Jacobi share | 77.2% | 49.9% |
| DTH density share | 11.4% | 24.9% |
| DTH normalization share | 7.9% | 17.1% |

- [*] Large-N block speedup is promising. GC has different convergence/failure counts and a WG96 reduction defect, so it cannot establish the medium-N architecture choice.
- [*] `prof_drain_events` uses successive marker END-to-END spans, including intervening commands and queue starvation. These are not individual kernel START-to-END times. `prof_reset` resets the anchor, not all accumulated totals.
- [*] Current scan400 code constructs a new `GpuDftb` per batch size. The report's claim that a profile aggregates batch1/100/400 conflicts with this source; initialization and repeated runs within one runtime can still accumulate. Identify the exact producing revision.
- [*] `reset_q0` preserves basis/warm state and chemical-potential seed. Repeated runs are not all cold. Last-run iteration count cannot normalize mean wall time across different histories.
- [*] `check_jacobi` reports the last solve per replica, not every iteration. Final p90=0 cannot prove mostly zero sweeps; stage timing versus a diagonal probe cannot prove 1–3 sweeps every iteration either.
- [ ] Record trajectory sweep histograms, initial off-norms, stop reasons and exact active replica-iterations with per-replica owner counters; read at existing boundaries.
- [ ] Compare cold, charge-reset/reused-basis and geometry-warm workloads separately. Report successful systems/s, residuals and every failure.
- [ ] Match CPU/GPU geometry sets, tolerances and SCC/result/force work. CPU single-point time multiplied by400 is an extrapolation, not measured scan throughput.

DTH density+normalization occupy 42% of the reported block profile. Eliminating them entirely gives a conditional `1/(1-.249-.171)=1.72x` before replacement costs. The report's ~70x ceiling is likewise a conditional Amdahl estimate, not a hardware roofline. Source-level logical loads are not measured DRAM bytes; cache reuse prevents diagnosing bandwidth from those counts alone.

### D. SC algebra and f32 invariant

Use `T=S*C` with row-major `[replica][AO][orbital]`. For spin-degenerate occupations:

```text
D_mu,nu = 2 sum_k f_k C_mu,k C_nu,k
q_A     = sum_mu in A,nu D_mu,nu S_mu,nu
        = 2 sum_mu in A,k f_k C_mu,k T_mu,k
```

q here is electron population: preserve `dq=q-q0` and factor two. `methods/dftb/dftb_cpu.rs` already uses the equivalent `C=L^-T Y`, `SC=L Y` formulation. `qmqm/charges.rs` has a helper with the opposite `q0-population` convention. Fortran `populations.F90::denseMullikenReal` supplies the density-contraction reference.

- [ ] First compute SC once per iteration with tiled GEMM; compute `g_k=sum_mu C_mu,k*T_mu,k`; require finite positive g; scale both C and T by `g_k^-1/2`; form populations directly; preserve DIIS policy.
- [ ] One `(replica,atom)` output owner gathers its AO rows with lanes on contiguous k. Precompute offsets, no atomics. Normalize all columns because occupations can change.
- [ ] Use explicit kernel boundaries for cross-workgroup dependencies. A 2D metric-reduction tile can avoid strided column gathers; begin with a simple verifiable implementation.
- [ ] For transported SC apply the same right transformation to C and T. With `delta=T-S*C`, rounding gives `delta_next=delta U + error(TU)-S error(CU)`. Paired scaling does not remove drift; diagonal normalization does not repair off-diagonal metric error.
- [ ] Independently recompute SC and `C^T S C` in diagnostics. Enforce refresh/repair based on measured f32 bounds. Start normalization every iteration; do not assume every4/every8 is safe.
- [ ] Refresh T whenever S changes or basis repair does not explicitly transport it. Cold Jacobi rotates orthogonal-basis cp before forming AO C: initialize T from the resulting AO C, not an unrelated cp.
- [ ] Benchmark the entire eigen+SC maintenance+normalization+population path. Control register lifetimes when updating C/T strips; extra sweep traffic may lose to recomputation.

Persistent T costs `4*batch*N²`: batch400/N86 is 11,833,600 bytes (~11.29 MiB), N246 is 96,825,600 bytes (~92.34 MiB). It cannot alias overwritten projection/repair scratch. The charge contraction is O(N²); neither SC recomputation nor the entire SCC iteration thereby becomes O(N²).

### E. State validity and final D/W

- [ ] Split electronic/basis validity from density and energy-weighted-density validity. `state_fresh` alone becomes insufficient when SCC stops producing D.
- [ ] Inventory energy/eval(false)/eval(true), finalization, forces, FIRE, constraints, retry and debug consumers. Basis/occupation changes invalidate D/W; geometry invalidates T; parked replicas retain their own valid state.
- [ ] Preserve final solve-at-final-charge initially: DIIS can advance q beyond the last solved H. Keep force eligibility separate from SCC active mask.
- [ ] Poison D/W during diagnostic SCC runs to expose stale consumers; test repeated evaluation, reset, changed geometry and subset retry.
- [ ] Build final `D=C diag(2f) C^T`, `W=C diag(2f*rho) C^T` with exact weighted rank-k tiles. W weights can be negative: do not sqrt signed weights. Preserve Rayleigh and entropy semantics; ensure unique triangle/mirror ownership and correct masks.

DTH has 54 C and 30 H in the inspected XYZ. Neutral mio valences imply 246 electrons and `sum f=123`; verify loaded q0/charge. With `f<=1`, 35–45 orbitals cannot represent this state. At cutoff tau, retained count obeys `m >= (nocc-N*tau)/(1-tau)`; for N246/nocc123/tau1e-6 at least123 remain. Do not optimize by silently discarding occupation weight.

### F. Kernel tuning: correct models first

- [ ] Compare fused Fermi against a separately compiled-out tail; a runtime flag may retain the 512-entry scratch footprint. Share safeguarded bracketed/Newton logic, persistent mu seed and explicit failure across paths. Validate stored f32 occupation sums; cheap f64 scalar decisions remain allowed.
- [ ] Audit whether fractional mode needs `select_occupation_batched`; remove or initialize downstream mask reads before dropping its producer. Test Fermi WG32/64/128 including degenerate/small-gap/extreme-occupation cases.
- [*] B16/PB32 pivot update loops distribute `16²+32*16=768` logical items across the WG. The claim that64/96 lanes are idle throughout is incorrect; only particular phases have limited useful lanes.
- [*] N86 direct update loops have `43²+86*43=5547` logical items per round, not ~516.
- [*] B16 block local arrays total9092 bytes at WG96 and9732 at WG256: `2*PB*(PB+1)*4 +(PB/2)*16 +WG*4+4`, before compiler-specific accounting.
- [ ] Query resources for exact compiled source/settings. Private-memory bytes are not register count; one WG/matrix does not imply one resident WG/SM. Use hardware counters only if tools support the actual OpenCL configuration.
- [ ] Unify/validate duplicated renderers in `gpu_eigen.rs` and `gpu_scc_plan.rs`; standalone off-tolerance overrides are not uniformly applied to production/block compilation.
- [ ] Test WG and B16/B24 after arbitrary-size reductions and ragged pivots pass. Choose crossover from application measurements across N/batch/warmness, not two examples.

### G. Active IDs and host overhead

- [ ] Preallocate compact IDs; gather stable physical slots at existing chunk reads and upload once. Map all participating kernels, including 3D GEMMs, through sid; every persistent buffer uses physical sid.
- [ ] Retain masks for within-chunk convergence; skip zero-active launches; finalization/forces must not accidentally inherit the last tail list. No matrix movement or global atomics.
- [ ] Count true active replica-iterations. Compaction removes empty WGs but cannot fill the GPU with only two remaining systems. Defer slot refill until complete per-slot reset/lifecycle contracts exist; the batch warm flag is insufficient.
- [ ] Extend allocation checks beyond the tiny H2O/batch1 device counter. Audit host vectors/clones/status formatting, `reset_diis`, geometry updates, kernel/program creation and library enqueue internals. Reuse internal capacity; distinguish public owned results from solver scratch.
- [ ] Combine chunk reads/finishes where dependencies permit. Bound diagnostic event metadata and report profiler overhead separately.

### H. Verification contract

- [ ] Freeze baseline numerical budgets before optimization. Existing loose smoke tolerances do not justify losing the reported GC relative-PES error (~3.7e-6 Ha) or force accuracy (~5.5e-6 Ha/Å).
- [ ] Use frozen-input f64 diagnostic references, keeping production bulk f32. Validate eigen residual, full metric, SC consistency, electron count, SCC status, PES, energy/entropy, forces/finite differences and relaxation where relevant.
- [ ] Cover N6/28/84/86/87/120/246 plus synthetic odd/tile-boundary sizes, mixed masks, all-inactive, retries and geometry/evaluation sequences.
- [ ] Run release benchmarks with pinned CPU thread settings. Store full logs under `debug/`, specs here, measured conclusions in reports/manifest. Record commands, hashes, device/driver, compile settings, warm-start policy and all failures.
- [ ] Promote defaults only after focused tests, application measurements and user review. No tolerance weakening, bulk f64, silent fallback, hot-loop allocation or global atomics.
