
> **Read [`Dense_Multi_Performance.md`](Dense_Multi_Performance.md) first.**
> Later turns of this chat override earlier ones. The part that still
> steers the solver starts at the learned-Λ autopsy and the
> extrapolation repair; the opening GEMM-microkernel turns are already
> measured (`gemm_sq_iter`).

https://chatgpt.com/share/6aae6101-84d4-83eb-93e4-94ff83b2d4a1

---

# USER

plese see this commit
https://github.com/ProkopHapala/dftbplus/commit/adaa8fbf717b3b987444be8d0bcddc401b8d470e

we added new test for GPU multi system matrix multiplication for efficient GPU-multi system density-matrix purification of fermi-operator-expansion solver of SCC/SCF DFTB+

there are detailed test of different matrix multiplication versions. Can you review it, mainly
https://github.com/ProkopHapala/dftbplus/blob/adaa8fbf717b3b987444be8d0bcddc401b8d470e/rust_dftb/src/qmqm/gpu_gemm.cl

and suggest how can we improve it even more? we are now like 10-20% of GPU theoretical trashoft for the best kerlsn, can we achieve more?

another question - what we need to do we do not loose this seepd when going to real system full SCC cycle. Like there are other operations, and we must minimize the overhead of hanress, sychnornization blocking etc.

---

# Chat GPT 5.6 sol

Yes. I reviewed the commit, the tuned `gpu_gemm.cl`, its Rust wrapper/tests, the fused purification kernel, and the persistent SCC plan. The benchmark progression is convincing: you have already moved from ~0.6–1 TFLOP/s with conventional tiled GEMM to ~6–8 TFLOP/s with the specialized symmetric-square register kernel. The design note records 6.7–6.8 TFLOP/s around batch 400 and 7.83 TFLOP/s at batch 1600, i.e. roughly 19–22% of your quoted 36-TFLOP/s peak.  

My main conclusion is:

> **I would not spend much effort trying to turn the current generic full-matrix `D²` kernel from 22% into 50–70% of peak.**
>
> There are probably another ~20–40% relative gains available in that kernel, but the larger wins now come from exploiting the *mathematical structure* of purification and from keeping the whole SCC pipeline asynchronous.

## 1. The current GEMM kernel is actually quite good

The winning structure makes sense:

```text
242 threads/WG
22 × 11 thread grid
8 × 4 output registers/thread
=> 88 × 88 output
TK ~ 16–22
one WG / matrix
A symmetric => only As_t staged
```

For `n=86`, it has several very favorable properties simultaneously:

* exactly one WG owns one matrix;
* essentially no redundant output coverage;
* A is loaded once per K tile;
* because `D=Dᵀ`, the same `As_t` supplies both operands;
* 32 independent accumulators/thread give substantial FMA ILP;
* local storage is only approximately

$$
88\times16\times4 \approx 5.6\ {\rm kB}
$$

for `TK=16`, rather than ~30 kB for the entire matrix.

Your sweep also experimentally demonstrates something important: **more occupancy by itself is not the answer**. 484- and 968-thread variants get worse, while the 242-thread/32-accumulator variant wins. 

So I would keep this as the reference kernel.

---

# 2. The first new experiment I would do: compute only half of \(D^2\)

This is potentially considerably more valuable than another 10% SGEMM tuning.

You know

$$
D=D^T
$$

and therefore

$$
D^2=(D^2)^T.
$$

Yet currently you compute all \(n^2\) elements independently.

That's almost **2× redundant arithmetic**.

### The obvious triangular formulation

Compute only

$$
j\ge i
$$

and write

```c
C[i*n+j] = cij;
C[j*n+i] = cij;
```

except diagonal.

For `n=86`, instead of 7396 dot products you need

$$
86(87)/2=3741,
$$

almost exactly half.

This changes what the optimal microtile should be.

Your current `8×4` microtile is very good for a rectangular dense output, but symmetry suggests trying **square microtiles**, particularly the already-existing

```text
11 × 11 threads
8 × 8 registers
=> 88 × 88
```

except that only the 66 upper-triangular thread tiles would perform useful FMAs.

The full-square version is slower:

```text
11x11 × r8x8 / sq ≈ 4.5 TFLOPS
22x11 × r8x4 / sq ≈ 6.2–7.8 TFLOPS
```

but that comparison counts the same number of mathematical FLOPs.

If the 11×11/r8×8 triangular version executes approximately **half of them**, it does not need high nominal TFLOPS to win.

For example:

```text
current full D²:        0.082 ms
triangular at only
4.5 equivalent TFLOPS: ~0.056 ms
```

and potentially better.

### Even better: triangular register tiles

Do not necessarily deactivate 55 threads completely.

I'd test three variants:

1. **11×11 threads × 8×8 register tile, triangular thread mask**
2. same, but diagonal `8×8` tiles compute only their internal triangle
3. a flattened list of the 66 triangular tile coordinates

```c
tile = lid;                // 0..65
decode tile -> (bi,bj);    // bj >= bi
```

with maybe 64/96/128 threads.

The mirror writes are cheap compared with the avoided 86-term dot products.

This is the most promising remaining single-kernel experiment I see.

---

# 3. Your existing kernel still pays unnecessary boundary/control overhead

`n=86` is known when you build the program.

But the kernel remains fairly generic:

```c
if (m < n && gk < n)
if (gi < n)
if (gj < n)
min(GEMM_TK,n-k0)
```

and loops are runtime loops.

For the production `N=86` specialization, I'd seriously consider generating a completely specialized source:

```c
#define N 86
#define NN 7396
#define NTK 4       // e.g. TK=22
```

rather than passing `n`.

Then let the compiler fully unroll:

```c
#pragma unroll
for (int kk=0; kk<22; kk++)
```

and ideally the `8×4` outer-product loops too.

The particularly attractive value from your larger-batch data is **TK=22**. 

For `n=86`:

```text
22 + 22 + 22 + 20
```

Only four K stages.

That means only ~8 local barriers in the existing GEMM.

I would benchmark generated fixed-N kernels for:

```text
TK = 11
TK = 22
TK = 29
TK = 43
```

not because local-memory size is the dominant issue anymore, but because this is a tradeoff among:

* number of barriers;
* local-memory residency;
* scheduling;
* compiler unrolling;
* register lifetime.

Your measured TK sensitivity is quite noisy, which strongly suggests architecture/scheduling effects rather than a simple bandwidth optimum.

---

# 4. Check register occupancy with the compiler, not just thread occupancy

At 32 accumulators/thread you already have at minimum:

```text
32 acc
8 A fragments
4 B fragments
indices / pointers / temporaries
```

probably somewhere around 50-ish registers after compilation.

That may be the actual occupancy limiter now.

I'd inspect NVIDIA's generated PTX/SASS or OpenCL compiler build log if available.

Then deliberately sweep:

```text
r8×4 = 32 accum
r6×4 = 24
r8×3 = 24
r6×6 = 36
r8×5 = 40
```

with tile dimensions chosen around 86.

The optimum might not lie on powers of two at all.

Your current result already proves that:

```text
22 × 11 threads
```

is better than aesthetically nicer 8/16/32 shapes.

So I would let the matrix dimension drive the design aggressively.

---

# 5. But the bigger target is your 0.141 ms versus 0.082 ms

This is more interesting to me.

Your own measurement says:

$$
t_{\rm GEMM}\approx0.082\ {\rm ms}
$$

but

$$
t_{\rm TC2-step}\approx0.141\ {\rm ms}.
$$

So **42% of wall time isn't the matrix multiplication anymore**. 

Even a magical infinitely fast GEMM only gets:

$$
0.141\rightarrow0.059\,{\rm ms},
$$

a 2.4× bound.

Conversely, cutting the administrative 0.059 ms to ~0.02 ms while keeping the GEMM unchanged gives

$$
0.082+0.020=0.102\ {\rm ms},
$$

already a **38% speedup**.

That is where I'd work next.

---

# 6. Fuse several TC2 iterations into ONE kernel launch

This is especially attractive because of your unusual topology:

> **one workgroup owns one complete matrix/system.**

Therefore TC2 iteration \(k+1\) does not require synchronization with any other WG.

So instead of Rust doing

```text
launch TC2
launch TC2
launch TC2
launch TC2
read convergence
```

make one kernel do:

```c
for(int iter=0; iter<4; iter++){
    D²
    error reduction
    trace reduction
    TC2 decision
    write Dnext

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    swap D/Dnext logically
}
```

for, say, **4 or 8 iterations per launch**.

This is much cleaner here than for most GPU algorithms because there is no inter-system coupling whatsoever.

You still need to write the matrix to global memory between iterations unless you sacrifice occupancy and keep the entire D locally, but you eliminate:

* OpenCL launch overhead ×4/8;
* Rust `set_arg` ping-pong;
* queue bookkeeping;
* repeated kernel setup;
* opportunities for accidental synchronization.

And `done[sys]` already provides exactly the semantic mechanism needed:

```text
if converged:
    stop updating this system
```

within the same kernel.

I'd call it something like:

```c
tc2_chunk_batched(..., int niter_chunk)
```

and start with `niter_chunk = 4`.

**I think this is likely a bigger practical gain than further SGEMM micro-tuning.**

---

# 7. You can also fold the error calculation into the output pass more cheaply

Currently you calculate

$$
e^2=\sum_{ij}(T_{ij}-D_{ij})^2
$$

then perform a whole-WG reduction.

For convergence you do not necessarily need the exact Frobenius norm every iteration.

A useful optimization is:

### Cheap iterations

Do TC2 only:

```text
D²
trace
branch
write
```

### Diagnostic iteration every 4th step

Also calculate:

```text
||D²-D||F
```

and update `done`.

Since TC2 convergence is monotonic-ish until the FP32 floor region, a `CHECK_INTERVAL=2` or `4` should be fine, provided you retain your freeze logic once convergence is detected.

Your design currently checks convergence every TC2 iteration device-side because otherwise already-converged systems can become unstable in FP32. 

But you don't actually need a full Frobenius reduction every iteration to prevent that.

You could use a **cheap convergence proxy** between exact checks, e.g. trace movement or max update from thread-local values; or simply accept one or two extra TC2 iterations if tests show the FP32 instability does not develop that quickly.

This needs empirical validation because you've already observed the fixed-point instability, so I wouldn't blindly remove the check.

---

# 8. For the real SCC path: do NOT integrate the standalone `purify_tc2_batched()` literally

This is crucial.

The current standalone wrapper does:

```rust
build_program(...)
allocate d_a
allocate d_b
allocate traces
allocate errs
allocate done
...
set_arg() every iteration
...
read errs
read traces every chunk
...
read final D to host
```

That is perfectly appropriate for a benchmark/bring-up function.

It would be disastrous inside the real SCC loop.

Your production `GpuSccPlan` has already adopted the correct philosophy for the Jacobi path:

> persistent buffers, persistent kernels, buffer arguments bound once, no allocation/compile in the SCC loop.

So purification should become a **first-class resident component of `GpuSccPlan`**, not a call to this convenience wrapper.

Something like:

```rust
struct GpuPurifyPlan {
    d0: Buffer<f32>,
    d1: Buffer<f32>,
    trace: Buffer<f32>,
    err: Buffer<f32>,
    done: Buffer<i32>,
    k_init: Kernel,
    k_chunk: Kernel,
    ...
}
```

created once in `GpuSccPlan::new()`.

No allocations thereafter.

---

# 9. More importantly: keep D entirely GPU-resident

For SCC the correct pipeline should be approximately:

```text
q
 ↓
H(q)
 ↓
purification
 ↓
D
 ↓
Mulliken q'
 ↓
DIIS
 ↓
next q
```

There should be **zero density-matrix CPU transfers**.

The standalone wrapper ends with:

```rust
rt.read_buffer(final_buf, d_out)
```

which obviously must disappear.

And ideally the downstream Mulliken kernel accepts whichever ping-pong buffer currently contains `D`.

Even better, make parity known deterministically by the chunk size:

```text
4 TC2 iterations per kernel
```

means each chunk ends in the same buffer it starts in if you structure two internal matrices appropriately.

Then there is no host-side "which ping-pong buffer contains D?" complexity either.

---

# 10. Purification makes your existing direct-population path obsolete/different

Your current eigensolver SCC has a nice optimization:

```text
C
S·C
Mulliken directly
```

so it avoids constructing D in the SCC loop.

With purification, D **is the result you already have**, so Mulliken becomes extremely cheap:

$$
q_A = q_A^0-\sum_{\mu\in A,\nu}D_{\mu\nu}S_{\nu\mu}.
$$

No eigenvectors, occupation sorting, density construction, etc.

This is an important system-level reason FOE should do much better than comparing only:

```text
Jacobi time vs purification time.
```

The entire tail changes.

For orthogonal purification you unfortunately still need:

$$
P = X D X^T,
$$

which is two generic GEMMs.

And **those generic GEMMs currently run only ~10% peak**, rather than the 20% symmetric-square path. 

That could eat a meaningful fraction of your gain.

Which brings me to the next important step.

---

# 11. Generalized purification may eventually be more important than another GEMM optimization

Your own design note correctly identifies:

$$
Q = KSK
$$

directly in the nonorthogonal AO basis, with

$$
KSK=K,\qquad N_e=\mathrm{Tr}(KS).
$$

That avoids:

* Löwdin \(X=S^{-1/2}\);
* \(X^THX\);
* final \(XDX^T\);
* warm-X repair machinery.



There is an apparent downside:

```text
orthogonal TC2:   D²         = 1 GEMM/iteration
generalized TC2: KS + (KS)K = 2 GEMMs/iteration
```

so naive counting says it is worse.

But in a **full SCC/geometry iteration**, that may not be true because you eliminate several other matrix multiplications and Löwdin maintenance outside purification.

Especially with warm start, where you may need only ~3–6 purification corrections, the fixed transformations become a large fraction of total cost.

So after integration I'd benchmark:

```text
total µs / SCC iteration
```

not purification alone.

---

# 12. I would keep host synchronization at SCC-chunk boundaries only

Your current `GpuSccPlan` architecture already does this very sensibly:

```text
enqueue several SCC iterations
read rms + active once
compact active systems
continue
```

That philosophy should remain.

For purification, there are actually **two nested asynchronous loops**:

```text
SCC chunk
    purification chunk
```

I would avoid letting both cause host synchronization.

Instead:

```text
SCC iteration:
    H(q)
    purification entirely device-side
    Mulliken
    DIIS
```

The purification loop should either:

### A. Run a predetermined warm iteration count

For example:

```text
4 McWeeny/TC2 correction steps
```

with a device convergence/failure flag.

Then SCC chunk-end reads everything.

or:

### B. Device-side purification loop

One kernel executes up to:

```text
MAX_PURIFY_INNER = 6 or 8
```

and exits per WG when converged.

No CPU involvement at all.

For a warm SCC calculation, **B is particularly attractive**.

---

# 13. Warm start is probably the largest algorithmic win

The note's cold random matrices take ~60 iterations, which is intentionally ugly. In actual SCC the density changes slightly. 

If you can get a genuine warm purification down to:

```text
3–6 iterations
```

then at your current fused step speed:

$$
6\times0.14 \approx0.84\ {\rm ms}
$$

for 400 systems already.

If chunk/fusion work gets a step near 0.10 ms:

$$
4\times0.10=0.4\ {\rm ms}.
$$

That's where this becomes extremely interesting.

But I strongly agree with the lesson from your sparse implementation:

> simply taking the old converged projector is insufficient because a polynomial of the new H cannot in general rotate the old occupied subspace.

The \(\delta K_0\) shift described in the note is therefore important. 

I would prioritize getting that physically correct warm start working before spending a week chasing another 5% from FMA scheduling.

---

# 14. One thing I would change in your benchmark

Your present TFLOP number counts:

$$
2n^3
$$

useful GEMM FLOPs.

That's appropriate for comparing GEMMs, but once you add triangular symmetry it becomes misleading.

I'd report both:

```text
useful-GEMM TFLOPS = 2 n³ / t
executed/effective work estimate
wall time / matrix
```

Ultimately the thing you care about is:

```text
ns/system/TC2-step
µs/system/SCC-step
```

not maximizing an artificial TFLOPS score.

A symmetry-aware kernel could show only "12 TFLOPS equivalent" while being dramatically faster because it doesn't execute half the FMAs.

---

# My recommended order

I would proceed roughly like this:

1. **Triangular/symmetric \(D^2\)** experiment. This is the one remaining kernel optimization that plausibly offers something approaching a factor of two in arithmetic, rather than 10%.
2. **`tc2_chunk_batched` with 4–8 iterations per launch.** One WG/system makes this unusually easy and should attack the measured 0.06-ms non-GEMM overhead directly.
3. Specialize production kernels for fixed `N=86` and test `TK=11,22,29,43`, with completely unrolled microtiles/K loops.
4. Inspect actual NVIDIA register count/occupancy; tune around 24/32/40 accumulators rather than just WG size.
5. Move purification into `GpuSccPlan`: all buffers/kernels persistent, D never leaves GPU, no compile/alloc/readback.
6. Implement **δK0 warm start + McWeeny/TC2 correction** and measure real GC/DTH SCC Hamiltonians rather than random spectra.
7. Benchmark complete SCC iteration, not eigensolver/purification alone.
8. Then try generalized \(KSK\) purification and ask whether eliminating Löwdin + basis transforms wins globally.

The most important practical point is that I think **the 20–22% peak number understates how successful the kernel already is**. For an 86×86 matrix, a 242-thread WG, substantial register blocking, barriers every K tile, boundary padding, and only 400–1600 independent matrices, ~8 TFLOP/s is respectable. The next large speedup should come from doing **less mathematical work and fewer launches**, not merely making the same \(n^3\) operation issue FMAs faster.

And I think the triangular \(D^2\) variant is the most obvious piece of low-hanging fruit that the current 42-variant sweep does **not** appear to test.

---

# Devin follow-up (2026-09-19): the ideas above were MEASURED, not just reviewed

Tested as new variants in `gpu_gemm.cl` + `tests/gpu_gemm.rs` only (no
SCC-plan files touched — the SCC-loop integration is a separate work
stream). All variants CPU-f64 parity-verified. RTX 3090.

## Verdicts vs the recommendations above

| idea | verdict | data |
|---|---|---|
| §2 triangular D² | **REJECTED** | every tri variant slower: 3.3–4.9 TF vs 6.2–8.4 full-sq, both b400 and b1600 |
| §3 fixed-N + TK sweep | **works** | `r8x4/tk22/nf86`: **8.38 TF (23 %)** best single-launch GEMM at b1600; +10–50 % vs same shape generic |
| §4 register sweep | neutral | `r9x4`@220thr ≈ ties `r8x4`; r6x6/r6x4/r8x3/r7x7 all worse; 8×4 stands |
| §6 chunked launches | premise half-wrong | in-kernel global ping-pong ≈ resident ⇒ launch amortization is minor (queue already pipelines); the real win is **residency** |
| §7 cheap convergence | reduced to non-issue | merged float2 reduce costs ~nothing; the 0.059 ms gap is elsewhere |
| §8–13 integration | separate work stream | §17.6 in progress by the other agent |

## Why triangular failed (the flaw in §2's reasoning)

Triangular masking halves *total FLOPs* but not the *critical path*: at
n=88-cover, one WG owns one matrix, and threads mapped to lower-triangle
tiles just idle — every active thread still runs its full
per-thread FMA chain. Wall time ≈ unchanged → measured 0.11–0.16 ms vs
0.076 full. Shrinking tiles to 4×4 (253 active tiles on 484 thr) still
leaves each active thread at one tile — same latency, worse
load:FMA ratio. The only real fix would be K-split work redistribution;
not worth it. Dead end — but now it's *measured*, not guessed.

## The result that matters — `gemm_sq_iter` (new kernel)

Iterated `D ← scl·D²`, niter rounds in ONE launch, D resident in
`__local` as a transposed 88×88 tile (30.9 KB < 48 KB): no staging
loops, no global traffic inside, 2 barriers/iter.

| variant | ms/iter b400 | TF-eq | vs fused tc2 step (0.141 ms) |
|---|---:|---:|---:|
| r8x4 loc red0 | 0.063–0.071 | 7.2–8.1 | ~2× faster |
| **r8x8 loc red0 (121 thr, 64 acc)** | **0.047–0.054** | **9.4–10.8** | **~2.7×** |
| r8x8 loc red2 (merged reduce) | 0.051 | 10.0 | ~2.8× |
| r8x4 glob red0 (ping-pong) | 0.068–0.074 | 7.0–7.5 | launch fusion alone buys ~nothing |

At b1600 the resident kernel still wins per-iter: 0.200 ms (r8x8) vs
0.243 for the best *single* launched GEMM.

**Where the fused step's 0.059 ms actually goes** (§5 of the answer
above): mostly NOT the reduces — the er-accumulate and the fused-write
pass each re-read `Din` from global (+2·n² ≈ +66 % traffic/iter), plus
~32 fold-halve barriers. In the resident kernel both costs vanish
(Din is `Ls`, merged reduce is ~free — `red2` reproducibly *beats*
`red0`, a scheduling quirk). With global buffers kept (glob mode), the
same fixes apply: cache the Din 8×4 fragment in registers, merge
er+trd into one float2 reduce.

**For production TC2** (handoff note): `tc2_step_batched` with
D-resident local + merged reduce projects to ~0.05–0.06 ms/iter
(~2.3–2.8× over 0.141); branch/freeze/trace fit inside (trace reduce
already priced by red2). Caveat: LN=88 cover needs n ≤ LN and
lsz = TX·TY; local 30.9 KB + reduce scratch fits 48 KB. Occupancy
drops to ~3 WGs/SM — measured faster anyway.



---

# USER

OK, we tested some of your ideas, (not yet n the git)

now focus on mirco optimization of the matrix-multiplication, macro optimization o the whole SCC loop is onother topic, which the coding agent did not finished yet, now revie notes of the tst from micro-optimizing agent, and think if there is anything we can do to improve it further (but only major improvement like another 10-20% not 1%)

All data is in and consistent across runs. Let me append a measured-results addendum to the chat file (it ends at line 765 — the user has it open there), then summarize:

Done. I reviewed the ChatGPT recommendations, implemented the testable kernel-level ones as new variants in the standalone microbenchmark files only (gpu_gemm.cl, gpu_gemm.rs, tests/gpu_gemm.rs — zero overlap with the other agent's files: gpu_purify.*, gpu_scc_plan.rs, gpu_eigen.rs, gpu_scc_bench.rs), and measured everything on the RTX 3090. Full writeup appended to the chat doc.

Review verdicts — measured, not guessed

Triangular D² (their headline idea #1): REJECTED. All 5 tri variants lose: 3.3–4.9 TF vs 6.2–8.4 full-sq, at both batch 400 and 1600. The flaw in their reasoning: masking lower-triangle threads halves total FLOPs but not the critical path — each active thread still runs its full FMA chain while masked threads idle at barriers. Their own variants 1–3 all keep per-thread latency constant (or worse). Only K-split redistribution could fix this — not worth it.

Fixed-N specialization (#3): works. nf86 gives +10–50% depending on shape; r8x4/tk22/nf86 hits 8.38 TF (23%) at b1600, the best single-launch GEMM. TK=22 makes 4 uniform K-chunks of 88 — confirming their "let n drive the design" intuition.

Register sweep (#4): neutral. r9x4@220thr ties the winner; nothing beats 8×4.

Chunked launches (#6): premise half-wrong. Global ping-pong in-kernel ≈ resident → launch amortization buys ~nothing (the queue already pipelines 50 reps). The real win is residency.

The result that matters — gemm_sq_iter (new kernel)

Iterated D ← scl·D², N iterations in one launch, D held in __local as a transposed 88×88 tile (30.9 KB): no staging, no global traffic inside, 2 barriers/iter:

variant	ms/iter (b400)	TF-eq	vs fused tc2 step
r8x8 loc, no reduce (121 thr, 64 acc)	0.047–0.054	9.4–10.8	~2.7× faster
r8x8 loc + merged reduce	0.051	10.0	~2.8×
r8x4 loc, no reduce	0.063–0.071	7.2–8.1	~2×
r8x4 glob ping-pong	0.068–0.074	7.0–7.5	launch fusion ≈ nothing

And it answers their §5 question: the fused step's 0.059 ms overhead is not the reduces (merged float2 reduce is ~free — red2 reproducibly beats red0, a scheduling quirk) — it's mostly the two extra global re-reads of Din in the er-accumulate and write passes (+66% traffic/iter) plus ~32 fold-halve barriers.

Handoff implications for the other agent

A resident-mode tc2_step_batched (D in __local, merged float2 reduce for er+trd, branch/freeze inside — all priced in already) projects to ~0.05–0.06 ms/iter vs current 0.141 — roughly a 2.4× step-level win before warm-start even enters. If they keep the global ping-pong design instead, two cheap fixes still apply: cache each thread's 8×4 Din fragment in registers (kills the double global re-read) and merge the two reduces into one float2.

Caveats for production: resident needs LN ≥ n (LN=88 at n=86), lsz = TX·TY, ~34 KB local/WG (fits 48 KB, ~3 WGs/SM — measured faster anyway). Parity verified: max|D−Dref| = 1.2e-11 over 4 iterations, traces exact.

Reproduce: cargo test --release --test gpu_gemm gemm_bench -- --ignored --nocapture (env: GEMM_BATCH, GEMM_REPS, GEMM_ITER).


---

# Chat GPT 5.6 sol

I think there is **one genuinely major kernel idea still untested**, and it is importantly different from the triangular variants that failed.

The micro-agent is correct that **masking the lower triangle is useless**. But the conclusion “only K-split could exploit symmetry” does not follow. You can redistribute the triangular work **spatially**, with no K-split and no extra reduction/barrier.

## 1. The experiment I would do next: true compact SYRK, 8×4 per thread

Your winning resident kernel is essentially

```text
11 × 11 threads = 121 threads
each thread = 8 × 8 outputs = 64 accumulators
whole WG = 88 × 88 outputs
```

So per \(k\):

$$
121\times64 = 7744
$$

FMAs.

But only the upper triangle is independent:

$$
\frac{88\times89}{2}=3916.
$$

The failed triangular kernel left an active thread responsible for its same 8×8 tile, hence **64 FMAs/k remained on the critical path**.

Instead enumerate only the **66 upper-triangular 8×8 tiles**, and split each tile spatially between two threads:

```text
66 upper 8×8 tiles
× 2 threads/tile
= 132 threads

each thread computes 8×4 = 32 outputs
```

For an off-diagonal block:

```text
thread A: rows 0..7, cols 0..3
thread B: rows 0..7, cols 4..7
```

then mirror the result into the lower triangle.

That gives:

$$
66\times64\times88=371712
$$

FMAs versus

$$
88^3=681472
$$

now: **54.5% of the arithmetic**.

The remaining ~4% beyond the mathematical 50.6% comes from computing the unnecessary lower half inside the 11 diagonal 8×8 blocks. Ignore that initially.

### Why this is very different from the failed triangular test

Current resident r8×8:

```text
121 threads
64 FMAs/thread/k
16 local floats/thread/k   (8 A + 8 B)
4 scheduled warps
```

Proposed compact triangular r8×4:

```text
132 threads
32 FMAs/thread/k
12 local floats/thread/k   (8 A + 4 B)
5 scheduled warps
```

Total local reads per \(k\) also fall:

$$
121(8+8)=1936
$$

to

$$
132(8+4)=1584,
$$

about **18% less**.

So this simultaneously gives:

* ~45% fewer FMAs;
* ~18% fewer local-memory reads;
* half the accumulators/thread, 32 instead of 64;
* more warps available to hide latency;
* exactly the `r8×4` register shape that already worked well in your previous sweep;
* **no K reduction**;
* **no additional barrier**.

That is absolutely worth testing. I would expect this to have a realistic chance of another **20–40% wall-time improvement**, although of course measurement decides.

---

## 2. The kernel mapping is quite simple

Since this is fixed `LN=88`, don't derive triangular coordinates expensively at runtime.

Generate a constant list:

```c
// 66 upper triangular block coordinates
uchar2 blocks[66] = {
    {0,0},{0,1},...,{0,10},
    {1,1},{1,2},...,{1,10},
    ...
    {10,10}
};
```

Then:

```c
int tile = lid >> 1;   // 0..65
int half = lid & 1;    // left/right 4 columns

int bi = blocks[tile].x;
int bj = blocks[tile].y;

int i0 = bi * 8;
int j0 = bj * 8 + half * 4;
```

Each thread then has a natural `8×4` microkernel:

```c
float4 acc0 = 0;
float4 acc1 = 0;
...
float4 acc7 = 0;

for (int k=0; k<88; k++) {
    float8 a = vload8(... row-block bi ...);
    float4 b = vload4(... col-block bj+half ...);

    acc0 = fma((float4)a.s0, b, acc0);
    acc1 = fma((float4)a.s1, b, acc1);
    ...
    acc7 = fma((float4)a.s7, b, acc7);
}
```

I would specifically write it in this explicit vector form rather than

```c
float acc[8][4];
for(r)
    for(c)
```

because it forces exactly the code structure you want: 32 scalar accumulators, vector fragment loads, no private-array indexing surprises.

After the read phase:

```c
barrier(CLK_LOCAL_MEM_FENCE);
```

write both halves:

```text
D'[i,j] = x
D'[j,i] = x
```

for off-diagonal blocks.

For diagonal blocks only write/mirror `j>=i`.

Then the usual barrier before the next \(D^2\).

So the resident loop still has **two barriers per iteration**.

---

## 3. Don't worry initially about the 132-thread non-warp multiple

132 means five scheduled warps, with the fifth mostly empty.

At first that sounds ugly, but compare warp instruction work.

Current:

$$
4\ {\rm warps}\times64\ {\rm FMA\ instructions/k}
\approx256
$$

warp-FMA instruction groups.

Proposed:

$$
5\times32\approx160.
$$

Even paying for that mostly-empty fifth warp, this is roughly **37% fewer warp-level FMA instructions**.

And with three resident WGs/SM you go roughly from

```text
12 warps/SM
```

to

```text
15 warps/SM
```

which may actually improve latency hiding.

So I would test the straightforward 132-thread version before trying to be clever about forcing 128.

---

## 4. If it works, diagonal tiles can remove another ~8%

The 132-thread version still computes the whole 8×8 diagonal tiles although only 36/64 entries are unique.

The exact SYRK arithmetic is

$$
3916\times88=344608
$$

FMAs,

whereas the simple 132-thread implementation performs

$$
4224\times88=371712.
$$

So only another

$$
1-\frac{344608}{371712}\approx7.3\%
$$

remains.

I would **not optimize this until the 132-thread version wins**.

If it does win, then a specialized diagonal path becomes worthwhile.

---

# 5. This also gives you the correct production reduction almost for free

For TC2 you need

$$
\|D^2-D\|_F^2.
$$

If you're computing only the upper triangle, don't reconstruct the whole thing just to evaluate the norm.

Accumulate

$$
e =
\sum_i \Delta_{ii}^2
+
2\sum_{i<j}\Delta_{ij}^2.
$$

So:

```c
if (i == j)
    er += dd*dd;
else
    er += 2.0f*dd*dd;
```

Trace only uses diagonal entries as usual.

Thus the compact SYRK kernel remains mathematically identical for the convergence criterion.

---

# 6. Second experiment: explicit vector microkernel in the current resident code

Even independently of triangular redistribution, I would test replacing:

```c
float acc[8][8];
float af[8];
float bf[8];

for(r)
    for(c)
        acc[r][c] = fma(af[r],bf[c],acc[r][c]);
```

with eight `float8` accumulators:

```c
float8 c0=0, c1=0, ... c7=0;

float8 a = vload8(...);
float8 b = vload8(...);

c0 = fma((float8)(a.s0), b, c0);
...
c7 = fma((float8)(a.s7), b, c7);
```

Your fixed-N result strongly suggests that **code-generation quality matters significantly** on this kernel family.

If NVIDIA's OpenCL compiler already completely scalarizes/unrolls the arrays, this gives zero. But if it doesn't, this could plausibly be a 10% effect.

It is cheap enough to test.

For the compact kernel I would use this explicit representation from the start:

```text
8 × float4 accumulators
```

which is a particularly natural microkernel.

---

# 7. Also test modest K-loop software unrolling in the resident kernel

The resident kernel currently has the compile-time

```c
for (int kk=0; kk<88; ++kk)
```

but no explicit unrolling.

I would not fully unroll 88 iterations: code size could become ridiculous.

But test:

```text
unroll ×2
unroll ×4
```

manually:

```c
for (int k=0; k<88; k+=2) {
    load a0,b0
    load a1,b1

    update acc with a0,b0
    update acc with a1,b1
}
```

This gives the compiler more opportunity to move the next local loads ahead of the current outer-product arithmetic.

Because the resident kernel now has no global latency, local-load/FMA scheduling is one of the few things left.

I regard this as less promising than compact SYRK, but potentially >10% rather than a 1% tweak.

---

# 8. Shared-memory banking may be the next actual bottleneck

There is something suspicious about the current `r8×8` addressing.

For one fragment you effectively access:

```text
base = tx * 8
```

across threads.

With 32 four-byte shared-memory banks, a fixed vector component generates bank indices roughly

$$
8\,tx \pmod{32},
$$

so only four bank residues occur.

Broadcast helps where threads request exactly the same address, but distinct `tx` values separated by four hit the same bank at different addresses.

So I suspect there are meaningful bank conflicts in the `float8` B-fragment loads.

However, I **wouldn't immediately pad 88→89/96**, because 88×88 is 30.25 KiB and your ability to hold ~3 WGs/SM is clearly valuable. Increasing the shared footprint can suddenly reduce residency from three WGs to two, which would be disastrous.

The compact `8×4` SYRK kernel partially attacks this automatically:

* only 4 B values rather than 8;
* fewer total local loads;
* different thread/tile mapping;
* lower register pressure.

So I would measure that first.

Only afterward would I experiment with an XOR/swizzled shared layout.

---

# 9. There is an even more aggressive symmetry formulation, but I wouldn't start there

The micro-agent says:

> “Only K-split redistribution could fix triangular.”

K-split would indeed work, but I agree it is unattractive because partial 8×8 results then have to be reduced.

The spatial redistribution above is much cleaner.

If somehow compact SYRK does **not** win, only then would I consider a two-thread K-split because the resident matrix gives you an interesting trick: once all threads finish reading `Ls`, that 31-KB `Ls` storage can temporarily become reduction scratch before being overwritten by the next matrix. So K-split doesn't necessarily require another 31-KB allocation.

But that's significantly more complicated, and spatial `8×4` splitting should be tried first.

---

# 10. Tensor cores are the only remaining radically different route

If the target is strictly ordinary FP32 OpenCL, ignore this.

RTX 3090 has a vastly larger tensor-core throughput than its ~36-TFLOP/s CUDA-core FP32 throughput. So the only obvious route toward a *factor-of-several* beyond this kernel family is some TF32/tensor-core formulation.

But I see two serious problems:

1. TC2 purification wants much better numerical precision than raw TF32's ~10-bit mantissa naturally gives.
2. Standard NVIDIA OpenCL doesn't give you a nice portable WMMA programming model.

You could theoretically investigate compensated TF32 or inline/vendor-specific machinery, but that's a separate research project, not the next optimization.

For your current implementation I would stay with normal FP32 cores.

---

## So my priority is now quite different

I would give the micro-agent exactly these three tests, in this order:

1. **Compact resident SYRK:** 66 upper 8×8 blocks × 2 threads/block × `8×4` outputs = **132 threads**, mirrored write, same two barriers. This is the big one.
2. Inside that kernel use **8 explicit `float4` accumulator vectors**, not `acc[8][4]`.
3. Test K-loop software unroll 1/2/4.

I would **not** spend time on more `TX/TY/TK` brute-force sweeps, more triangular masking, reduce variants, launch fusion, or tiny local-memory tweaks. Your measurements have already largely exhausted those directions.

The key distinction is:

> The failed triangular kernels saved arithmetic by **turning threads off**.
> The compact-SYRK kernel saves arithmetic by **giving the freed threads half of an active tile**.

That is the one remaining structural microkernel change I see that has a credible chance of another **≥20%** on top of the excellent ~0.05-ms resident result.

---

# Measured results — round 2 (compact SYRK, vec accs, k-unroll)

All three suggestions implemented as `gemm_sq_iter_tri` + `GEMM_VEC`/`GEMM_KU`
knobs on `gemm_sq_iter` (same files, still zero overlap with the SCC agent).
Parity: all variants max|D−Dref| = 1.2e-11 (bit-identical across kernels —
the mirror-write/writeback is correct), traces exact.

**b1600, ms/iter (within-run comparison — cross-run clock drift is ±10%):**

| variant | ms/iter | TF-eq |
|---|---:|---:|
| r8x8 loc red0 (prev best) | 0.2229 | 9.13 |
| r8x8 loc ku4 | 0.2223 | 9.15 |
| r8x8 loc vec | 0.2239 | 9.09 |
| **tri 132-thr r8x4 + ku4** | **0.2080** | **9.79** |
| tri 132-thr r8x4 red0 | 0.2159 | 9.43 |
| tri 132-thr r8x4 red2 | 0.2440 | 8.34 |
| tri 264-thr r8x2 red0 | 0.2709 | 7.51 |

**b400:** tri r8x4 red0 ≈ 0.055 vs r8x8 red0 ≈ 0.058 (run 1: tri/ku4 0.0516 was
best overall). Ordering is stable within-run; absolute numbers drift.

## Verdicts

1. **Compact SYRK (your §1): WORKS but ~5–10%, not 20–40%.** The direction is
   right — it is now the best kernel — but the kernel is not FMA-issue-bound,
   so the 37% instruction reduction translates to much less wall time. The
   residual bound is a latency/LSU mix: all threads read fragments of the
   same `Ls` row each kk (32B/lane × 32 lanes = 1 KB/warp → 8 LSU cycles at
   128 B/clk), plus mirror writeback doubles the local-store count, and the
   diagonal-block waste (~7%) stays.
2. **264-thr r8x2 split: worse** (~0.27). 16 acc/thread underfills the
   pipeline — the 8×4 split is the sweet spot, as you predicted.
3. **vec accumulators: neutral** (±1%). The NV compiler already scalarizes
   `acc[8][8]` identically — code-gen was not the limiter.
4. **ku unroll: small positive** — helps tri most (0.216→0.208 at b1600);
   r8x8 also likes ku4 occasionally (0.0541 vs 0.0583 at b400 run 2). Keep
   `ku4` on whichever kernel wins.
5. **Reduce cost on tri is NOT free**: red2 adds ~13–17% at 132 threads
   (0.216→0.244) — with half the FMA work per thread there's less compute to
   hide the fold-halve tree under. In production, run the er/trd reduce only
   every k-th iteration or overlap it with the next iteration's reads.
6. **Bank conflicts (your §8): padding can't help here** — every thread reads
   a window of the *same* Ls row each kk, so per-row stride changes nothing.
   The ~3-way conflict on `gi0·8 mod 32 ∈ {0,8,16,24}` is real but inherent to
   row-fragment access; a block-XOR swizzle would need write-side coordination
   and is the next thing to try only if tri becomes the production kernel.
7. **Tensor cores (your §10): confirmed out of scope** — no WMMA in NVIDIA
   OpenCL and TF32's ~10-bit mantissa is marginal for idempotency at 1e-5.

Net state: best resident kernel is now `sqit12x11/r8x4/tri/ku4` ≈ **0.21 ms/iter
at b1600 / ~0.052 at b400**, ≈9.8 TF-eq — a modest but real improvement over
the r8x8 resident kernel. Diminishing returns confirmed for this microkernel
family; the remaining wins are at the SCC-loop level (warm-start, reduce
frequency, resident tc2_step integration).







---

# USER

Now bigger picture about solving SCC/SCF by density matrix purification. We maybe of stupid thing. We focus on purification itself but the critical part is minimize energy wrt hessian and overlap matrix. DM-purifucation is there just to ensure Pauli exclusion principle (right). We had problem that warm-start of purifier fix occupied and unoccupied subspace but that is exactly because we miss the charge relaxation step (downhill energy minimize). We can even afford initially relaxed porufier (violat Pauli exclusion slightly) and increase the tighten convergence criteria only as we reach close to minimum 


---

# Chat GPT 56. sol

Yes. I think this changes the whole way we should formulate the solver.

And I assume by “Hessian” you mean the **Hamiltonian \(H\)** together with overlap \(S\).

What we have been doing so far is essentially the **spectral-projection use of purification**:

$$
H \longrightarrow D_0(H)\longrightarrow
D_1\longrightarrow D_2\longrightarrow\cdots
$$

where TC2/McWeeny is expected to turn a function of \(H\) into the occupied projector.

But purification has another, older use: **as the constraint-maintenance part of direct density-matrix energy minimization**. The literature explicitly distinguishes these two uses. LNV, CG density-matrix search, and hybrid purification/minimization schemes do essentially what you are now suggesting. ([Wiley Online Library][1])

And for SCC-DFTB I think that formulation is much more natural.

## The key conceptual decomposition

Write, for the moment, in an orthogonal basis:

$$
D = U\,{\rm diag}(f_i)\,U^T.
$$

There are really **two independent things that have to relax**:

1. **Eigenvectors \(U\)** — rotate the occupied subspace toward the low-energy eigenspace of the current Hamiltonian.
2. **Occupations \(f_i\)** — push them toward the allowed occupation pattern.

Purification,

$$
D\rightarrow 3D^2-2D^3
$$

or TC2,

does essentially **only #2**.

It changes the eigenvalues \(f_i\), but it leaves the eigenvectors of \(D\) unchanged.

That is exactly why the old converged \(D\) is such a bad warm start for pure TC2 after \(H\) changes: **we were asking an eigenvalue operation to perform an eigenvector rotation. It mathematically cannot do it.**

The missing operation is an **energy-descending subspace rotation**.

---

# The natural density-matrix gradient

For an exactly idempotent projector,

$$
D^2=D,
$$

the tangent-space gradient of

$$
E_{\rm band}=\operatorname{Tr}(DH)
$$

is the double commutator.

A downhill flow is

$$
\boxed{
\dot D=-[D,[D,H]]
}
$$

or equivalently

$$
\dot D =
2DHD-DH-HD.
$$

This is very beautiful for our purpose.

Define

$$
C=[D,H].
$$

Then

$$
\frac{dE}{dt}
=
-\|[D,H]\|_F^2\le0.
$$

So it is genuinely downhill, and convergence is simply

$$
[D,H]\rightarrow0.
$$

Moreover, the continuous flow can be written as

$$
\dot D=[[D,H],D],
$$

which is an **isospectral rotation**. It rotates the eigenvectors of \(D\) but preserves its eigenvalues.

So we get almost a perfect complementary pair:

$$
\boxed{\text{double-commutator descent}}
\qquad
\text{rotates eigenvectors, preserves occupations}
$$

while

$$
\boxed{\text{purification}}
\qquad
\text{changes occupations, preserves eigenvectors.}
$$

This is, I think, the conceptual piece we had been missing.

---

## And yes: we absolutely do not need tight idempotency early

There is actually a subtle correction here which makes your idea stronger.

Pauli exclusion does **not** require

$$
D^2=D.
$$

It requires, per spin,

$$
0\le f_i\le1,
$$

plus

$$
\operatorname{Tr}D=N_{\rm occ}.
$$

Idempotency

$$
f_i\in\{0,1\}
$$

is the stronger condition for a zero-temperature pure Slater determinant.

So an early density matrix with, for example,

$$
f=(0.99,0.95,0.76,0.24,0.05,0.01)
$$

is not “violating Pauli”. It is simply an ensemble/fractionally occupied density matrix.

Only things such as

$$
f_i<0
\quad\text{or}\quad
f_i>1
$$

violate the admissible occupation bounds.

That strongly suggests exactly your proposed strategy:

> **early iterations:** concentrate on lowering the actual electronic energy; keep the density merely well behaved / weakly purified
> **late iterations:** progressively enforce idempotency to the final tolerance.

At finite electronic temperature we should not even drive idempotency to zero; fractional Fermi occupations are the correct solution.

---

# This is even more important for SCC-DFTB

For SCC-DFTB the relevant energy is not merely

$$
\operatorname{Tr}(D H)
$$

with a fixed \(H\).

Schematically,

$$
E[D]
=
2\operatorname{Tr}(D H^0)
+
\frac12\Delta q[D]^T\gamma\,\Delta q[D]
+
E_{\rm rep}.
$$

Here the Mulliken charges are linear functions of the density matrix.

Therefore

$$
\frac{\delta E}{\delta D}
$$

is precisely the SCC Hamiltonian \(H_{\rm SCC}[D]\), modulo the usual factors/conventions.

So instead of doing the traditional nested process

$$
D_n
\rightarrow H[D_n]
\rightarrow
\boxed{\text{solve }H[D_n]\text{ accurately}}
\rightarrow D_{n+1}
\rightarrow\text{mix}
$$

we can conceptually do

$$
\boxed{
D_{n+1}=D_n+\text{downhill step using }H[D_n]
}
$$

and continue.

That simultaneously relaxes:

* orbital/subspace orientation,
* charge distribution,
* SCC Hamiltonian,
* and eventually occupations/idempotency.

In other words, **we may not need an “inner eigensolve/purification until convergence” at every SCC iteration at all.**

That inner solve is probably massive oversolving far away from self-consistency.

The literature's direct density-matrix minimization methods are precisely based on this viewpoint; LNV treats purification as part of the representation/constraint rather than as the whole electronic solver. ([PubMed][2])

---

# There is already a very useful stationarity condition

In a nonorthogonal AO basis, the desired final density \(K\) satisfies three conceptually separate conditions:

$$
\boxed{\operatorname{Tr}(KS)=N_{\rm occ}}
$$

electron count,

$$
\boxed{KSK=K}
$$

zero-temperature idempotency,

and

$$
\boxed{HKS-SKH=0}
$$

energy stationarity / occupied–virtual decoupling.

The last one is the nonorthogonal analogue of

$$
[H,D]=0.
$$

This exact `HKS − SKH` condition is also the standard AO SCF commutator criterion. ([Heidelberg University Archive][3])

Notice what TC2 addresses:

$$
KSK-K\rightarrow0.
$$

It does **nothing directly** to make

$$
HKS-SKH\rightarrow0
$$

if the eigenvectors are wrong.

So we had indeed been focusing heavily on only one of the two residuals.

---

# A very simple prototype algorithm

Before implementing full LNV, I would make an extremely simple orthogonal-basis experiment.

Start with a reasonably valid \(D\).

For each electronic iteration:

$$
H=H[D].
$$

Compute

$$
A=DH
$$

and then

$$
B=DHD=A D.
$$

Because \(D,H\) are symmetric,

$$
HD=A^T.
$$

Then the downhill direction is just

$$
\boxed{
G=2B-A-A^T.
}
$$

Update

$$
D\leftarrow D+\alpha G.
$$

That is only **two dense GEMMs per relaxation step**.

Then, not necessarily every time:

$$
D\leftarrow {\cal P}(D),
$$

where \({\cal P}\) is one McWeeny/TC2 correction.

Conceptually:

```text
repeat:
    H = build_H(D)

    A = D H
    B = A D

    G = 2 B - A - A^T       # downhill subspace rotation
    D += α G

    occasionally:
        weak purification / occupation correction

    gradually tighten:
        ||H D - D H||
        ||D² - D||
        charge residual
```

That is already enough to test the central hypothesis.

---

## An especially nice diagnostic

We should separately monitor

$$
r_{\rm orbital}=\|HD-DH\|_F
$$

and

$$
r_{\rm idem}=\|D^2-D\|_F.
$$

They tell us **different things**.

For example after moving geometry:

```text
old converged projector:
    r_idem    ≈ 1e-6      excellent
    r_orbital ≈ 1e-2      bad
```

Pure TC2 does approximately:

```text
r_idem    ↓
r_orbital unchanged / poorly addressed
```

while the double-commutator step should do:

```text
r_orbital ↓↓↓
r_idem    slightly perturbed by finite step
```

Then an occasional McWeeny step repairs the latter.

I would very much like to see these two curves plotted independently. They should make the problem visually obvious.

---

# Your idea of progressive purification tolerance makes sense

I would go even further than changing a convergence tolerance.

I would change the **relative number of the two operations** during convergence.

For example, early:

```text
4–8 energy-descent steps
1 weak McWeeny correction
```

middle:

```text
2–4 descent steps
1 purification
```

late:

```text
1 descent
1–2 purification
```

and finally certify

$$
r_{\rm orbital}<\epsilon_H,
\qquad
r_{\rm idem}<\epsilon_P,
\qquad
|\operatorname{Tr}D-N|<\epsilon_N.
$$

The exact schedule should probably be residual-driven rather than hard-coded.

For example,

$$
\eta=
\frac{r_{\rm idem}}
     {r_{\rm orbital}+r_{\rm idem}}
$$

could determine whether the next operation should spend work on purification or energy descent.

If

$$
r_{\rm orbital}\gg r_{\rm idem},
$$

purifying harder is clearly wasted work.

---

# This is very close to LNV

What you have independently arrived at is essentially the motivation behind **Li–Nunes–Vanderbilt density-matrix minimization**.

LNV introduces an auxiliary matrix \(L\) and defines the physical density through McWeeny purification. In a nonorthogonal basis:

$$
\boxed{
K=3LSL-2LSLSL
}
$$

and then minimizes the energy **with respect to \(L\)**.

So purification is built into the parameterization, but the actual search direction comes from **energy minimization**, not repeated spectral projection. The nonorthogonal generalization was explicitly constructed so that only \(S\), rather than \(S^{-1}\), appears in the variational functional. ([Rutgers University][4])

There is even old work comparing pure purification and LNV that concludes they have complementary strengths and proposes a **hybrid**, exactly the direction we are discussing. ([ScienceDirect][5])

And Challacombe later derived a simplified density-matrix minimization scheme specifically to reduce matrix multiplications and shift more work to quadratically convergent McWeeny purification. ([CiNii Research][6])

So this isn't merely plausible; there is a fairly deep body of algorithms here that we should now mine rather than invent everything ourselves.

---

# One potentially very nice property specific to DFTB

DFTB may actually be unusually friendly to direct minimization.

For fixed nuclei,

$$
E[D]
=
E_0[D]+\frac12\Delta q[D]^T\gamma\Delta q[D].
$$

Since \(q[D]\) is linear in \(D\), along a **linear trial direction**

$$
D(\alpha)=D+\alpha G,
$$

the SCC energy is essentially a quadratic polynomial:

$$
E(\alpha)
=
E(0)+a\alpha+\frac12b\alpha^2.
$$

That means we may be able to get an **analytic or nearly analytic optimal step length** rather than doing an expensive line search.

This deserves a careful derivation because the projector constraint/retraction complicates it slightly, but it could be extremely useful on GPU: determine a per-system \(\alpha\) from a handful of scalar contractions, update all 400 replicas independently.

That is much more attractive than a generic conjugate-gradient line search with host intervention.

---

## The one caution

I would not let \(D\) become arbitrarily non-idempotent.

Without constraints, minimizing a linear band energy is ill-posed: occupations can go outside their physical range and collapse into the lowest state.

So “loose purification” should mean something like

$$
0-\delta \lesssim f_i \lesssim1+\delta
$$

with the correct trace, not “ignore representability entirely.”

LNV handles this carefully using the auxiliary matrix/purification functional; that's one reason I think we should study its exact functional before designing our final implementation.

But the broad idea is exactly right.

### I now think the right architecture is

$$
\boxed{
\text{energy minimizer}
+
\text{weak/adaptive N-representability enforcement}
}
$$

rather than

$$
\boxed{
\text{purifier pretending to be the electronic minimizer}.
}
$$

And the cleanest first test is probably **double-commutator descent + occasional McWeeny**, because it will tell us very quickly whether the real GC/DTH warm-SCC problem collapses from “5–15 purification GEMMs” to perhaps only a couple of subspace-relaxation steps plus one final purification.

[1]: https://onlinelibrary.wiley.com/doi/10.1002/qua.25048?utm_source=chatgpt.com "A perspective on the density matrix purification for linear scaling electronic structure calculations - Kim - 2016 - International Journal of Quantum Chemistry - Wiley Online Library"
[2]: https://pubmed.ncbi.nlm.nih.gov/10005210/?utm_source=chatgpt.com "Density-matrix electronic-structure method with linear system-size scaling."
[3]: https://archiv.ub.uni-heidelberg.de/volltextserver/30839/1/Maximilien_Ambroise_Dissertation.pdf?utm_source=chatgpt.com "Inaugural - Dissertation"
[4]: https://www.researchwithrutgers.org/en/publications/generalization-of-the-density-matrix-method-to-a-nonorthogonal-ba/?utm_source=chatgpt.com "Generalization of the density-matrix method to a nonorthogonal basis - Rutgers, The State University of New Jersey"
[5]: https://www.sciencedirect.com/science/article/abs/pii/S0010465599002210?utm_source=chatgpt.com "Density matrices in O(N) electronic structure calculations: theory and applications - ScienceDirect"
[6]: https://cir.nii.ac.jp/crid/1363107370478897536?utm_source=chatgpt.com "A simplified density matrix minimization for linear scaling self-consistent field theory | CiNii Research"



---

# USER

Ok, now we should think about computational efficiency. How to minimize number of matrix multiplications which we spent effort optimizing. Is the commutator [D,H] really the best way to implement minimizer? I understand the advantage it is tangential and approximately preserve the idempotency? Because if we just make derivative of Energy Tr(H.D) wrt density matrix elements ignoring any orthogonality constraints it would be perhaps faster just the idempotency preservation would require more steps. Than it is about what is more numerical work the nice tangential down hill step or restoration of idempotency. Can you analyse this thoroughly?


---

# Chat GPT 56. sol

Yes. Once we count **matrix multiplications rather than conceptual elegance**, the answer changes. The double-commutator is an excellent geometric minimizer, but I don't think it is obviously the fastest implementation for your GPU. There is a particularly attractive alternative: **take the cheap unconstrained energy-gradient step, then use a very cheap resident polynomial retraction back toward the projector manifold**.

This is close in spirit to the old density-matrix-minimization literature: purification and variational minimization have complementary roles, and hybrid schemes were explicitly found useful; Challacombe's simplified DMM was motivated specifically by reducing matrix multiplications and shifting more work toward quadratically convergent McWeeny purification. ([CiNii Research][1])

## 1. What the double commutator is buying us

For clarity, first use an orthogonal basis and an exact rank-\(N\) projector

$$
P^2=P,\qquad P^T=P.
$$

For fixed \(H\),

$$
E[P]=\operatorname{Tr}(PH).
$$

The unconstrained derivative is trivially

$$
\nabla E=H.
$$

But an allowed infinitesimal variation of a projector must satisfy

$$
P\,\delta P+\delta P\,P=\delta P.
$$

The tangent projection of \(H\) is

$$
G=[P,[P,H]]
  =PH+HP-2PHP.
$$

So steepest descent is

$$
P' = P-\alpha G.
$$

This has two major numerical virtues.

First, it preserves particle number to first order because

$$
\operatorname{Tr}G=0.
$$

More importantly, since \(G\) is tangent,

$$
P'^2-P'
=\alpha^2G^2.
$$

**There is no \(O(\alpha)\) idempotency error.**

That is the real reason the commutator is attractive. You can rotate the occupied subspace significantly while only damaging idempotency at second order.

And the continuous equation

$$
\dot P=-[P,[P,H]]
$$

is actually isospectral: it rotates the eigenvectors without changing occupation eigenvalues at all.

### Cost

Using symmetry you need only two full matrix multiplications:

$$
A=PH,
$$

then either

$$
B=AP=PHP,
$$

and

$$
G=A+A^T-2B.
$$

So:

$$
\boxed{\text{double commutator}=2\ {\rm general\ GEMMs}}
$$

per minimization step.

That's mathematically clean. But your **general GEMM is much more expensive than the extremely optimized resident symmetric square** \(D^2\).

That changes the engineering tradeoff.

---

# 2. What happens with the stupidly cheap raw gradient?

Suppose instead we simply do

$$
X=P-\alpha(H-\mu I).
$$

No GEMM at all. H is already available.

Write the perturbation as

$$
X=P+\alpha E.
$$

Then

$$
X^2-X
=
\alpha(PE+EP-E)+O(\alpha^2).
$$

In the basis defined by the current occupied/unoccupied subspaces,

$$
P=
\begin{pmatrix}
I&0\\
0&0
\end{pmatrix},
\qquad
H=
\begin{pmatrix}
H_{oo}&H_{ov}\\
H_{vo}&H_{vv}
\end{pmatrix}.
$$

The raw step gives

$$
\delta P=
-\alpha
\begin{pmatrix}
H_{oo}-\mu I&H_{ov}\\
H_{vo}&H_{vv}-\mu I
\end{pmatrix}.
$$

Only

$$
-\alpha
\begin{pmatrix}
0&H_{ov}\\
H_{vo}&0
\end{pmatrix}
$$

is useful physical subspace rotation.

The \(oo\) and \(vv\) pieces merely destroy the occupations.

So raw gradient is computationally free, but **a large part of the step points normal to the projector manifold**.

Consequently,

$$
X^2-X=O(\alpha),
$$

rather than \(O(\alpha^2)\).

This is exactly the tradeoff you described.

---

# 3. But here is the important observation: purification can project that raw gradient onto the correct tangent direction

Take the McWeeny map

$$
M(X)=3X^2-2X^3.
$$

Linearize it around an exact projector:

$$
X=P+\epsilon E.
$$

One obtains

$$
M(P+\epsilon E)
=
P+
\epsilon
\left(
PE(1-P)+(1-P)EP
\right)
+O(\epsilon^2).
$$

In words:

**McWeeny deletes the occupied–occupied and virtual–virtual parts of the perturbation to first order, while preserving the occupied–virtual part.**

Therefore start with

$$
E=-(H-\mu I).
$$

Then

$$
M\!\left(P-\alpha(H-\mu I)\right)
=
P
-\alpha
\left[
PH(1-P)+(1-P)HP
\right]
+O(\alpha^2).
$$

But

$$
PH(1-P)+(1-P)HP
=
[P,[P,H]].
$$

Therefore

$$
\boxed{
\text{raw gradient + McWeeny}
=
\text{double-commutator descent}
+O(\alpha^2).
}
$$

This is the crucial result.

The raw-gradient approach is not fundamentally a worse descent direction if you retract it properly. **To first order it produces exactly the same constrained gradient.**

So now it becomes purely a question of which implementation makes the retraction cheaper.

---

# 4. Unfortunately ordinary McWeeny still needs two multiplications

McWeeny requires

$$
X^2
$$

and

$$
X^3=X^2X.
$$

Thus it is still

$$
\boxed{2\ {\rm matrix\ multiplies}}.
$$

Nominally that ties the double commutator.

But they're not equivalent on your GPU:

| Scheme                   | Products         |
| ------------------------ | ---------------- |
| Double commutator        | \(PH,\;(PH)P\)   |
| McWeeny retraction       | \(X^2,\;(X^2)X\) |
| Your optimized primitive | **\(X^2\)**      |

The first McWeeny multiply is exactly the operation for which you've just built that exceptional ~0.05-ms resident kernel.

The second one, unfortunately, is again a general multiplication.

But we can do better.

---

# 5. Two quadratic purifier steps give a proper retraction using only **squares**

Define the two TC2-like quadratic maps

$$
q_0(x)=x^2
$$

and

$$
q_1(x)=2x-x^2.
$$

A single one cannot properly stabilize both 0 and 1.

But compose them:

$$
R_1(x)
=
q_0(q_1(x))
=
(2x-x^2)^2,
$$

or in the opposite order,

$$
R_2(x)
=
q_1(q_0(x))
=
2x^2-x^4.
$$

Both satisfy

$$
R(0)=0,\qquad R(1)=1
$$

and, critically,

$$
R'(0)=R'(1)=0.
$$

Therefore around a projector they have the same first-order retraction property as McWeeny:

$$
R(P+\epsilon E)
=
P+
\epsilon
\left(
PEQ+QEP
\right)
+O(\epsilon^2),
\qquad Q=I-P.
$$

So we can do

$$
\boxed{
X=P-\alpha(H-\mu I)
}
$$

followed by

$$
\boxed{
X\leftarrow 2X-X^2
}
$$

and

$$
\boxed{
P'\leftarrow X^2
}
$$

for example.

And that is **two symmetric squares and nothing else**.

This maps beautifully onto what you have just optimized.

---

# 6. On your GPU this could be substantially faster than the commutator

Imagine one resident kernel:

```text
load X = D - α(H - μI) into local memory

X = 2X - X²
X = X²

write Dnew
```

The raw energy step is just during the load:

```c
Ls[i] = D[i] - alpha * H[i];
```

The two expensive operations are exactly your resident `sq_iter`.

No generic \(D H\).

No generic \(DHD\).

No global ping-pong between the two purification operations.

Given your measured resident square of roughly

$$
0.05\ {\rm ms}
$$

per \(86\times86\) batched operation, the core of one properly retracted gradient step might be roughly

$$
\sim 0.10\ {\rm ms}
$$

plus one load/write.

That should be directly compared with **two general \(86^3\) GEMMs** required by the exact double commutator.

I would not be surprised if the raw-gradient + two-square retraction wins substantially in wall time despite having nominally the same count of matrix products.

This is probably the first experiment I'd do.

---

# 7. Can we get down to only ONE square per energy-minimization step?

This is exactly where your “don't enforce Pauli/idempotency tightly early” idea becomes interesting.

There is a mathematical obstacle to doing a proper retraction in one square.

Suppose a quadratic polynomial

$$
p(x)=ax^2+bx+c
$$

should preserve the fixed points

$$
p(0)=0,\qquad p(1)=1.
$$

For both endpoints to be attractive to first order we would want

$$
|p'(0)|<1,\qquad |p'(1)|<1.
$$

Since \(c=0\) and \(a+b=1\),

$$
p'(0)=b,
\qquad
p'(1)=2-b.
$$

You cannot simultaneously have

$$
|b|<1
$$

and

$$
|2-b|<1.
$$

They only meet at the marginal case \(b=1\), which is just \(p(x)=x\).

So:

$$
\boxed{\text{one matrix square cannot contract occupations toward both 0 and 1.}}
$$

You need at least a cubic map or a composition of two quadratics for a true local retraction.

That explains something fundamental about TC2: one branch repairs one side of the spectrum and worsens the other side.

---

# 8. But alternating one-square steps may be exactly the relaxed solver you want

We do not necessarily need a full retraction after **every SCC step**.

We could do

$$
X_n=D_n-\alpha H_n
$$

then on iteration \(n\),

$$
D_{n+1}=X_n^2,
$$

and next iteration

$$
D_{n+2}=2X_{n+1}-X_{n+1}^2.
$$

Or choose the branch from the trace as TC2 already does.

Over two iterations the combination has the proper stable 0/1 endpoints.

This would give approximately

$$
\boxed{1\ {\rm resident\ square/SCC\ iteration}}
$$

during the loose phase.

That is extremely attractive.

The price is that after every individual iteration one side of the occupation spectrum is imperfect and can actually move away from its endpoint.

But perhaps that simply doesn't matter early in SCF.

Then as the energy/charge residual becomes small:

```text
early:   raw descent + 1 TC2 square
middle:  raw descent + 2-square retraction
late:    full idempotency purification / certification
```

That is much closer to what I think you had in mind.

And because \(H[D]\) changes between those iterations, it isn't mathematically identical to composing two quadratic maps on the same matrix. So this one definitely needs empirical testing.

But the reward is large enough—potentially **half the matrix-multiply count**—that it is worth trying.

---

# 9. What if we do several completely raw \(D-\alpha H\) steps before purification?

I am less enthusiastic about that.

The normal error is \(O(\alpha)\) every step, so after \(k\) unrestricted steps you generically accumulate something like

$$
O(k\alpha)
$$

occupation distortion.

Meanwhile the physically useful subspace rotation also scales \(O(k\alpha)\).

So you aren't gaining a favorable asymptotic ratio.

Worse, once eigenvalues cross the basin boundary between occupied and virtual states, purification can restore the **wrong projector**.

I would therefore prefer **one cheap quadratic correction every iteration**, rather than completely unconstrained descent for five iterations followed by a big cleanup.

---

# 10. There is a useful middle ground requiring one general GEMM

Instead of the completely raw gradient, compute only

$$
A=DH.
$$

Then take

$$
G_\mu
=
-(A+A^T-2\mu D).
$$

Cost:

$$
\boxed{1\ {\rm general\ GEMM}}.
$$

For an exact projector this has blocks

$$
G_\mu
=
-\begin{pmatrix}
2(H_{oo}-\mu I)&H_{ov}\\
H_{vo}&0
\end{pmatrix}.
$$

Compared with raw \(-H\), it completely removes the virtual–virtual pollution.

Only the occupied–occupied normal component remains.

So this is a reasonable halfway point:

```text
raw gradient:          0 GEMM, lots of normal pollution
semi-tangent:          1 GEMM, less normal pollution
exact tangent:         2 GEMM, zero O(α) pollution
```

Whether the one-GEMM version is worthwhile depends on the occupied bandwidth. A single scalar \(\mu\) cannot remove \(H_{oo}-\mu I\), so if occupied eigenvalues span several eV the first-order idempotency drift may still be substantial.

I would test it only after the zero-GEMM + square-retraction approach.

---

# 11. A very important benchmark: count **wall time**, not GEMMs

Because for your kernel these are hugely different units:

$$
D^2
$$

is now a special operation that can keep an entire matrix resident in local memory and reuse the same matrix as both operands.

Whereas

$$
DH
$$

has two independent operands.

You cannot resident-store both full \(88\times88\) matrices in 48 KB:

$$
2\times88^2\times4
\approx 62\ {\rm KB}.
$$

So the double-commutator's two products can never exploit the exact same trick.

Thus the correct cost model isn't

```text
2 MM versus 2 MM
```

but more like

```text
2 resident symmetric squares
versus
2 ordinary GEMMs.
```

That could easily be a factor approaching two in actual time.

---

# 12. There may be an even better representation than D

There is one more possibility I think we should seriously keep in mind.

If the objective is simply to find the occupied subspace, represent it by

$$
C\in\mathbb R^{n\times m},
\qquad
C^TC=I,
\qquad m=N_{\rm occ},
$$

instead of the full projector

$$
D=CC^T.
$$

Then

$$
E=\operatorname{Tr}(C^THC).
$$

The Grassmann gradient is

$$
R=HC-C(C^THC).
$$

For \(m\simeq n/2\), its arithmetic is approximately:

$$
HC:\quad 2n^2m\approx n^3,
$$

$$
C^T(HC):\quad 2nm^2\approx\frac12 n^3,
$$

$$
C(C^THC):\quad2nm^2\approx\frac12 n^3.
$$

Total:

$$
2n^3
$$

FLOPs.

That is equivalent to **one full \(n\times n\) GEMM**, whereas density-matrix double-commutator descent costs approximately two.

Moreover,

$$
C' = C-\alpha R
$$

violates orthonormality only at

$$
O(\alpha^2),
$$

because \(C^TR=0\).

So you can again orthonormalize only periodically.

And your current SCC code already has the conceptual advantage that Mulliken populations can be evaluated directly from \(C\) and \(SC\), without ever building the full density during the SCC loop.

So if we are truly asking:

> “What is the minimum linear-algebra work necessary to rotate the occupied subspace?”

then **occupied-subspace minimization may ultimately beat density-matrix minimization.**

The old DMM literature was mostly motivated by linear scaling/sparsity, which changes that tradeoff. Your regime is rather unusual: tiny dense \(n\sim86\) matrices, hundreds of replicas, very fast custom GPU kernels.

---

## What I would benchmark now

1. **Exact reference:** double commutator, 2 general GEMMs/step. This tells us the number of true energy-descent iterations needed and gives the cleanest physics baseline.
2. **Projected raw step:** \(X=D-\alpha(H-\mu I)\), followed by **two resident quadratic squares** \(q_0\circ q_1\). This should have the same first-order direction as the commutator but be much better matched to your GPU.
3. **Aggressive loose mode:** raw step + **one TC2 square per SCC iteration**, alternating/trace-selecting the branch; switch to two-square retraction near convergence.
4. Only if those disappoint, test the one-general-GEMM semi-tangent form.
5. In parallel, make a small CPU/GPU prototype of occupied-subspace \(C\)-minimization, because its theoretical arithmetic count is roughly half that of the full-density tangent method.

The key thing I would **not** do now is assume “double commutator = best because it respects the manifold.” Geometrically it is best. Computationally, your newly discovered **resident matrix-square primitive is so disproportionately cheap** that it makes a deliberately unconstrained gradient followed by polynomial retraction potentially the better algorithm.

And there is literature precedent for precisely this philosophy: direct minimization and purification were developed as complementary approaches, with hybrid methods often outperforming either alone. ([arXiv][2]) The recent semiempirical GPU work also reinforces that dense purification becomes attractive specifically when matrix-matrix operations map exceptionally well to consumer GPUs, although that work uses purification mainly as the density build from a fixed Hamiltonian rather than the variational hybrid we're considering here. ([cambridge.org][3])

[1]: https://cir.nii.ac.jp/crid/1363107370478897536?utm_source=chatgpt.com "A simplified density matrix minimization for linear scaling self-consistent field theory | CiNii Research"
[2]: https://arxiv.org/abs/cond-mat/9810042?utm_source=chatgpt.com "Density matrices in O(N) electronic structure calculations: theory and applications"
[3]: https://www.cambridge.org/engage/chemrxiv/article-details/68076154927d1c2e663240c7?utm_source=chatgpt.com "Acceleration of semi-empirical electronic structure theory calculations on consumer-grade GPUs using mixed precision density matrix purification | Theoretical and Computational Chemistry | Cambridge Open Engage"


---

# USER

Ok do you think it make sense to merge into one kernel (do the stupid unconstrained gradient and immediately constraint\project back to tangent space at least approximately?) also think about numerical error. Remember we work in f32 accuracy (f64 is costly on gaming GPUs), and I think if we do unconstrained step and then subtract radial part we introduce large numerical noise. Therefore it would be better to do some noise-aware approximately tangential descent step inside single kernel and the correct it by one or two purification steps.


---

# Chat GPT 5.6 sol


Yes. I think merging it into **one resident kernel** makes a lot of sense, but I would change one thing in the formulation:

> I would **not** use the literal unconstrained gradient \(-H\) and rely on purification to remove a large radial component every time.
>
> Instead I would maintain a **cheap approximation to the normal/constraint force**, subtract that *before* taking the step, and let one or two resident purification squares clean up only the remaining small error.

That is both computationally attractive and much better behaved in FP32.

The general idea is consistent with density-matrix-minimization work: purification and variational minimization have complementary roles, and hybrid schemes were explicitly proposed; simplified DMM formulations were developed specifically to reduce the matrix-multiply count and shift work toward efficient purification. ([ScienceDirect][1])

## 1. Why I would avoid `D -= α H; purify(D)` literally

Suppose \(D=P\) is already a perfect projector. In the basis of its occupied/unoccupied subspaces,

$$
P=
\begin{pmatrix}
I&0\\0&0
\end{pmatrix},
\qquad
H=
\begin{pmatrix}
H_{oo}&H_{ov}\\
H_{vo}&H_{vv}
\end{pmatrix}.
$$

Only

$$
H_{\rm tan}=
\begin{pmatrix}
0&H_{ov}\\
H_{vo}&0
\end{pmatrix}
$$

actually rotates the occupied subspace.

The raw step

$$
X=P-\alpha H
$$

also introduces the much larger components

$$
-\alpha H_{oo},\qquad-\alpha H_{vv}.
$$

Near convergence this is particularly bad: \(H_{ov}\) may be tiny, while \(H_{oo},H_{vv}\) remain of ordinary electronic-energy magnitude.

Then we deliberately create a large error only to cancel it with purification.

Mathematically that works. Numerically in FP32 it is unattractive.

---

# 2. Your FP32 concern about exact tangent projection is also valid

The exact projector tangent is

$$
T_P(H)
=
[P,[P,H]]
=
PH+HP-2PHP.
$$

Geometrically this is beautiful.

Numerically, near convergence, it is something like

$$
\text{small}
=
\text{large}
+\text{large}
-\text{large}.
$$

And the large terms themselves came from length-86 FP32 dot products.

A crude error scale for a naive FP32 dot product is

$$
\gamma_{86}\sim 86\,\epsilon_{\rm f32}\sim5\times10^{-6}
$$

before considering cancellation.

That is already roughly the same regime as the projector/idempotency floors you have actually observed.

So the exact formula can paradoxically be **less useful numerically near convergence than a slightly approximate tangent direction constructed without large cancellation**.

And it costs two ordinary GEMMs.

So I would not make it the production method.

---

# 3. Think in terms of the **normal constraint force**

For an exact projector, decompose \(H\) into tangent and normal pieces:

$$
H=T+N,
$$

with

$$
T=PHQ+QHP,
\qquad Q=I-P,
$$

and

$$
N=PHP+QHQ.
$$

Notice:

$$
[P,N]=0.
$$

At the solution,

$$
[P,H]=0,
$$

therefore

$$
T=0,\qquad N=H.
$$

So instead of recomputing the expensive projection \(T_P(H)\), maintain an estimate

$$
\Lambda\approx N.
$$

Then simply use

$$
\boxed{F=H-\Lambda}
$$

as the descent force.

If \(\Lambda\) is good,

$$
F\simeq T.
$$

Then your cheap raw step

$$
X=D-\alpha F
$$

is **already approximately tangential**.

This is much more attractive.

---

# 4. The beautiful thing is that purification tells us the constraint force almost for free

Suppose

$$
F=H-\Lambda
$$

and make the trial

$$
X=D-\alpha F.
$$

Then retract/purify it:

$$
D'=R(X).
$$

For a near-projector and small step,

$$
D'
=
D-\alpha T_D(F)+O(\alpha^2).
$$

Therefore the amount removed by the retraction is

$$
D'-X
=
\alpha N_D(F)+O(\alpha^2).
$$

So the purification step itself gives us an estimate of the normal part of the force:

$$
N_D(F)
\approx
\frac{D'-X}{\alpha}.
$$

Therefore update

$$
\boxed{
\Lambda'
=
\Lambda+
\frac{D'-X}{\alpha}.
}
$$

This is the really interesting algorithmic loop.

We do **not** explicitly calculate the tangent projector.

We learn the constraint force from what purification had to remove.

---

## 5. There is an even simpler equivalent update

Because

$$
X=D-\alpha(H-\Lambda),
$$

the previous equation becomes

$$
\Lambda'
=
H+\frac{D'-D}{\alpha}.
$$

So, approximately,

$$
\boxed{
\Lambda_{n+1}
=
H_n+\frac{D_{n+1}-D_n}{\alpha}.
}
$$

This is reminiscent of recovering a Lagrange/constraint force from the difference between unconstrained and constrained motion.

And look at the fixed point:

if

$$
D'=D,
$$

then

$$
\Lambda'=H,
$$

and consequently

$$
F=H-\Lambda=0.
$$

That is extremely important numerically.

The naive

$$
D-\alpha H\rightarrow\text{purify}
$$

scheme continues making a huge radial excursion **even at the exact solution**.

This scheme does not.

At convergence the force that enters the update actually goes to zero.

That is exactly what we want in FP32.

---

# 6. Warm start becomes exceptionally natural

Suppose geometry \(R_0\) was converged:

$$
D_0,\quad H_0,
\qquad [D_0,H_0]\simeq0.
$$

Then

$$
H_0
$$

is almost entirely normal with respect to \(D_0\).

So simply initialize

$$
\boxed{\Lambda_0=H_0.}
$$

At the new geometry / SCC state,

$$
F_1=H_1-H_0.
$$

That is exactly the small perturbation we intuitively wanted.

No subtraction of orbital energies inside GEMMs.

Just one elementwise FP32 subtraction.

So instead of feeding purification

$$
D-\alpha H_1,
$$

we feed it roughly

$$
D-\alpha(H_1-H_0).
$$

But unlike the naïve "`ΔH` forever" idea, \(\Lambda\) is subsequently **updated from the constraint correction**, so the solver doesn't stall merely because \(H\) stops changing.

I like this formulation much more.

---

# 7. We can improve the approximate tangent further for essentially zero cost

Even if \(\Lambda\) is imperfect, subtract from

$$
F_0=H-\Lambda
$$

any matrix that is a polynomial in \(D\), because it commutes with \(D\) and therefore has no true tangent component.

The simplest is

$$
aI+bD.
$$

Choose \(a,b\) to minimize

$$
\|F_0-aI-bD\|_F^2.
$$

This only needs four scalar reductions:

$$
s_1=\operatorname{Tr}D,
$$

$$
s_2=\langle D,D\rangle_F,
$$

$$
h_0=\operatorname{Tr}F_0,
$$

$$
h_1=\langle D,F_0\rangle_F.
$$

Then solve

$$
\begin{pmatrix}
n&s_1\\
s_1&s_2
\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}
=
\begin{pmatrix}h_0\\h_1\end{pmatrix}.
$$

For an exact rank-\(m\) projector,

$$
s_1=s_2=m
$$

and the determinant is

$$
m(n-m),
$$

so for your roughly half-filled \(n=86,m\sim43\) problem this is very well conditioned.

Then use

$$
\boxed{
F=F_0-aI-bD.
}
$$

This guarantees

$$
\operatorname{Tr}F=0
$$

and

$$
\langle D,F\rangle_F=0.
$$

So you've cheaply removed the two largest obvious normal modes.

No matrix multiplication.

Just reductions and elementwise operations.

I would call this **commuting-gauge removal**.

---

# 8. This now gives exactly the single kernel I think you want

Something conceptually like:

```text
resident_relax(D, H, Lambda):

    # O(n²), global -> local
    F0 = H - Lambda

    reduce:
        Tr(F0)
        <D,F0>
        Tr(D)
        <D,D>

    solve tiny 2×2 system for a,b

    # approximate tangent step
    X = D - α [F0 - a I - b D]

    # D is now resident in local memory
    one or two polynomial purification steps

    D_out = X

    update Lambda from constraint correction
```

Everything expensive after the initial loads operates on one resident 88×88 matrix.

No general GEMM.

No global intermediate matrix.

No double commutator.

---

# 9. If the force is approximately tangent, idempotency violation is only second order

This is the other big advantage.

For exact projector \(P\), let \(T\) be tangent.

Set

$$
X=P-\alpha T.
$$

Because

$$
PT+TP=T,
$$

we have

$$
X^2-X
=
\alpha^2T^2.
$$

So the defect is

$$
\boxed{O(\alpha^2)}
$$

rather than

$$
O(\alpha)
$$

for a generic unconstrained gradient.

That means if our learned \(\Lambda\) is reasonably good, purification no longer has to do heavy lifting.

It is just a small retraction.

This is exactly where **one square per iteration** becomes plausible.

---

# 10. One purification square may genuinely be enough most of the time

Suppose the approximate tangent error is \(\epsilon_N\).

Then roughly

$$
\|X^2-X\|
\sim
O(\alpha\epsilon_N)
+
O(\alpha^2\|T\|^2).
$$

If \(\Lambda\) tracks the normal force, \(\epsilon_N\) should rapidly become small.

Then a single quadratic correction can keep the defect bounded while the system moves downhill.

Occasionally run the complementary correction to clean both sides of the occupation spectrum.

So I would consider:

### Fast iteration

$$
X=D-\alpha F
$$

then one resident quadratic polynomial.

### Cleanup iteration

Same descent followed by **two complementary quadratic maps**.

---

# 11. I would use the two-square SP2 pair rather than McWeeny on your GPU

The two simple maps are

$$
q_-(X)=X^2
$$

and

$$
q_+(X)=2X-X^2.
$$

Each needs exactly one of your extremely fast resident symmetric squares.

The composition

$$
q_-\circ q_+
$$

or

$$
q_+\circ q_-
$$

has

$$
R(0)=0,\qquad R(1)=1
$$

and zero derivative at both fixed points.

Thus it eliminates first-order normal errors, like a proper retraction.

And unlike McWeeny,

$$
3X^2-2X^3,
$$

it uses **only matrix squares**.

That is perfectly matched to the kernel you've just optimized.

---

# 12. A softer version is possible, but I suspect the hard SP2 pair is actually better in FP32

Consider

$$
p_a(X)
=
X+a(X-X^2)
=
(1+a)X-aX^2.
$$

Its endpoint derivatives are

$$
p_a'(0)=1+a,
\qquad
p_a'(1)=1-a.
$$

Then applying \(p_a\) followed by \(p_{-a}\) gives endpoint contraction

$$
1-a^2
$$

on both sides.

So \(a<1\) gives a gentler purifier.

That sounds attractive numerically.

But your \(a=1\) maps are simply

$$
X^2
$$

and

$$
2X-X^2,
$$

whose coefficients are exactly representable and require minimal arithmetic.

The temporary amplification is only ×2.

If our force is already almost tangent, the radial error being doubled is tiny anyway.

So my first implementation would actually use the standard hard pair.

---

# 13. Important FP32 point: don't purify below the arithmetic floor

Your measured purification already tells us the FP32 floor is around

$$
10^{-6}
$$

in the relevant norms.

An 86-term FP32 GEMM also naturally has arithmetic errors in roughly this range.

Therefore if

$$
\|D^2-D\|
\sim10^{-6},
$$

performing three more purification rounds is not “making it more physical.”

It is mostly manipulating rounding noise.

So the control logic should be something like:

```text
idempotency bad     -> two square corrections
idempotency moderate -> one correction
idempotency near FP32 floor -> no correction
```

This fits your original intuition that constraint enforcement should be adaptive.

---

# 14. Updating Λ without storing the original trial matrix

There's a practical problem:

$$
\Lambda'
=
\Lambda+\frac{D'-X}{\alpha}
$$

looks as though we'd have to retain \(X\), another 31 kB.

We don't.

During each polynomial transformation,

$$
Y=p(X),
$$

the thread responsible for \(Y_{ij}\) already has:

```text
old = X_ij
new = Y_ij
```

at the moment it overwrites local storage.

Therefore it can simultaneously do

```c
lambda[idx] += beta * (new - old) / alpha;
```

before overwriting `old`.

For two purification steps:

```text
Lambda += β (Y-X)/α
Lambda += β (Z-Y)/α
```

which sums to

$$
\Lambda
+
\beta\frac{Z-X}{\alpha}.
$$

No second full matrix resident.

No extra GEMM.

No need to reread \(D_{\rm old}\).

This is a very natural fusion into your resident square kernel.

---

# 15. I would damp the constraint-force update

I wouldn't initially use

$$
\beta=1.
$$

Because the quantity

$$
D'-X
$$

contains:

* real constraint correction;
* finite-step \(O(\alpha^2)\) error;
* FP32 GEMM error.

I'd use something like

$$
\Lambda_{n+1}
=
\Lambda_n
+
\beta\frac{D'-X}{\alpha},
\qquad
\beta\sim0.3-0.7.
$$

Then experiment.

When

$$
|D'-X|
$$

approaches the FP32 floor, stop updating \(\Lambda\), because division by \(\alpha\) would merely amplify numerical noise.

This is another reason I prefer a **fixed reasonable \(\alpha\)** rather than reducing \(\alpha\rightarrow0\) near convergence.

---

# 16. Step size should be tied to spectral width

For a small occupied–virtual rotation involving states \(i,a\),

$$
E(\theta)
\simeq
E_0+
(\epsilon_a-\epsilon_i)\theta^2.
$$

Gradient descent behaves roughly like

$$
\theta_{n+1}
\simeq
[1-\alpha(\epsilon_a-\epsilon_i)]
\theta_n.
$$

So stability suggests

$$
0<\alpha\Delta\epsilon_{\max}<2.
$$

A very sensible starting point is therefore

$$
\boxed{
\alpha \sim \frac{0.5-1}{\epsilon_{\max}-\epsilon_{\min}}.
}
$$

You already have machinery for estimating spectral bounds.

That also keeps the initial trial matrix safely within the basin where the polynomial correction behaves well.

No line search required initially.

---

# 17. Why this is much better numerically than the double commutator

I'd rank them like this.

| Method                                      | general GEMMs | squares | FP32 behavior near solution                             |
| ------------------------------------------- | ------------: | ------: | ------------------------------------------------------- |
| raw \(-H\) + purification                   |             0 |     1–2 | **bad:** large artificial radial excursion remains      |
| exact double commutator                     |             2 |       0 | geometrically perfect, but subtracts large GEMM results |
| \(H-aI-bD\) + purification                  |             0 |     1–2 | better, but block-energy dispersion remains             |
| **\(H-\Lambda\) recycled constraint force** |         **0** | **1–2** | **good:** force itself tends to zero                    |
| \(H-\Lambda-aI-bD\)                         |         **0** | **1–2** | probably best candidate                                 |

The last one is what I would implement.

---

## The resulting picture

We no longer treat purification as the solver.

Nor do we explicitly calculate the expensive projector gradient.

Instead:

$$
\boxed{
\begin{aligned}
F &= H-\Lambda\\
F &\leftarrow F-\operatorname{proj}_{\{I,D\}}F\\
X &= D-\alpha F\\
D' &= \text{1–2 resident-square retractions}(X)\\
\Lambda' &=
\Lambda+\beta\,\frac{D'-X}{\alpha}.
\end{aligned}}
$$

The interpretation is quite clean:

* \(H\) says where energy wants to go;
* \(\Lambda\) learns the part forbidden by the Pauli/projector constraint;
* \(H-\Lambda\) becomes an increasingly tangent force;
* purification handles only the remaining small constraint error;
* the correction made by purification feeds back into \(\Lambda\).

At convergence,

$$
D'=D,\qquad \Lambda=H,\qquad H-\Lambda=0,
$$

so unlike raw-gradient + purification, **there is no large cancellation cycle left running at the FP32 floor**.

I think this is sufficiently promising that I would test it **before implementing the exact double-commutator minimizer at all**. The double commutator is useful as a CPU/f64 reference to verify the direction, but computationally this learned-constraint-force formulation fits your highly optimized resident-square GPU primitive much better.

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0010465599002210?utm_source=chatgpt.com "Density matrices in O(N) electronic structure calculations: theory and applications - ScienceDirect"



---

# USER

This is very good I like it a lot! But you wrote the merged kernel very abstract pseudocode and it seems to me like multiple matrix multiplication. Can you write it really as one kernel efficiently using shared memory (which is limited we should store like small tiles 8x8 or so not whole matrix because that would allow fit only one workgroup in 48kb per compute unit Budget)?


---

# Chat GPT 5.6 sol

Yes. And with your constraint **“small local tiles, several WGs resident per SM”**, the fused kernel should look quite different from the resident-whole-matrix prototype.

The key realization is:

> **One WG should own one whole system, but it does not need to store the whole matrix locally.**
> It can stream \(K\)-slabs through ~8–12 kB of local memory, exactly like your successful `r8x4/tk22` GEMM.

That still allows several independent systems/WGs per SM.

More importantly, we can do the complete

$$
D\rightarrow X\rightarrow{\rm purification}\rightarrow D'
$$

**in place**, with no second \(N^2\) scratch matrix.

---

# 1. Concrete operation we want

Start from

$$
F_0=H-\Lambda.
$$

Cheaply remove the obvious commuting/normal components,

$$
F=F_0-aI-bD,
$$

where \(a,b\) come from the \(2\times2\) least-squares projection.

Then

$$
X=D-\alpha F.
$$

Equivalently,

$$
X=(1+\alpha b)D-\alpha(H-\Lambda)+\alpha aI.
$$

That is important computationally: **constructing \(X\) is just an elementwise fused expression.**

Then either one cheap purification:

$$
D'\equiv q_\pm(X)
$$

or the stronger complementary pair

$$
Y=q_\pm(X),
\qquad
D'=q_\mp(Y),
$$

with

$$
q_-(X)=X^2,
\qquad
q_+(X)=2X-X^2.
$$

The latter costs **two symmetric squares**, but they occur inside the same kernel.

---

# 2. No full matrix in local memory

For \(N=86\), keep your successful geometry approximately:

```text
N    = 86
LN   = 88

TX   = 22
TY   = 11
RTX  = 4
RTY  = 8

WG   = 242 threads
TK   = 22
```

One thread owns an \(8\times4\) register tile.

The entire output coverage is

$$
(11\times8)\times(22\times4)
=
88\times88.
$$

But local memory is only

$$
22\times88\times4
=
7744\ {\rm bytes}
$$

for the staged symmetric operand.

Add perhaps

```text
242 × float4 = 3872 bytes
```

for one temporary reduction buffer.

Total:

$$
\approx11.6\ {\rm kB/WG}.
$$

So a 48-kB SM can fit **four WGs by shared-memory budget**. Register/thread limits may reduce that, but local memory certainly doesn't force one WG/SM.

That is exactly the regime you wanted.

---

# 3. The nice trick: transform `D → X` while loading the first GEMM

Normally we might do:

```text
kernel 1: X = D - αF
kernel 2: T = X²
```

Absolutely unnecessary.

While staging each element into the local \(K\)-slab:

```c
float d   = D[idx];
float f0  = H[idx] - Lambda[idx];

float x = (1 + alpha*b)*d - alpha*f0;
if (i == j) x += alpha*a;

Ls[kk*LN + i] = x;
```

The same thread can also do

```c
D[idx] = x;
```

at that moment.

Each matrix element is staged exactly once during the complete square, so this is safe.

Thus by the time \(X^2\) has finished,

```text
register accumulators = X²
global D               = X
```

with **no separate X-building pass**.

That is the central fusion.

---

# 4. Then overwrite `D = Y` in place

After the first square, every thread has its \(8\times4\) part of

$$
T=X^2.
$$

Because all reads of the original \(D\) are finished, we can safely do:

```c
x = D[idx];        // D now contains X

if (expand)
    y = 2*x - t;
else
    y = t;

D[idx] = y;
```

At exactly the same point we learn how much the constraint operation removed:

```c
Lambda[idx] += beta * (y - x) / alpha;
```

Then:

```c
barrier(CLK_GLOBAL_MEM_FENCE);
```

Now the same global `D` buffer contains the complete \(Y\).

No temporary matrix.

---

# 5. If we want the second purifier, square `D=Y` immediately

Reuse exactly the same local slab:

```text
global D=Y
   ↓
small local K slab
   ↓
register GEMM
   ↓
Y²
```

Then apply the opposite TC2 branch:

```c
y = D[idx];

if (second_expand)
    z = 2*y - y2;
else
    z = y2;

Lambda[idx] += beta * (z - y) / alpha;
D[idx] = z;
```

The two Lambda updates sum to

$$
\Lambda'
=
\Lambda+
\beta\frac{Y-X}{\alpha}
+
\beta\frac{Z-Y}{\alpha}
=
\Lambda+
\beta\frac{Z-X}{\alpha}.
$$

So **we never need to keep \(X\)**.

That is particularly neat.

---

# 6. Near-real OpenCL kernel

This is close to how I would actually implement the experiment.

I am omitting only mundane bounds/debug plumbing.

```c
#define N    86
#define LN   88

#define TX   22
#define TY   11
#define RTX   4
#define RTY   8

#define TK   22
#define WG   (TX*TY)

// WG = 242

inline float4 reduce4(
    __local float4* buf,
    float4 v,
    int lid
){
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    int s = WG;
    while(s > 1){
        // ceil(s/2), important for WG=242
        int h = (s + 1) >> 1;

        if(lid < s-h){
            buf[lid] += buf[lid+h];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        s = h;
    }

    // avoid the race you already found
    float4 out = buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    return out;
}


__kernel
__attribute__((reqd_work_group_size(WG,1,1)))
void relax_purify_batched(
    __global float*       D,          // IN/OUT
    __global const float* H,
    __global float*       Lambda,     // IN/OUT
    __global const float* nocc,
    const float alpha,
    const float beta,
    const int   two_step              // 0 or 1
){
    const int sys = get_group_id(0);
    const int lid = get_local_id(0);

    const int tx = lid % TX;
    const int ty = lid / TX;

    const int r0 = ty * RTY;
    const int c0 = tx * RTX;

    const int NN = N*N;

    __global float*       Dg = D      + sys*NN;
    __global const float* Hg = H      + sys*NN;
    __global float*       Lg = Lambda + sys*NN;

    // symmetric K-slab, NOT whole matrix
    __local float Ls[TK*LN];

    // scalar-reduction scratch
    __local float4 red[WG];

    __local float l_a;
    __local float l_b;
    __local float l_cD;
    __local float l_trX;
    __local int   l_expand1;


    // ------------------------------------------------------------
    // 0. Cheap O(N²) projection:
    //
    // F0 = H - Lambda
    //
    // choose a,b minimizing
    // || F0 - a I - b D ||²
    //
    // reductions:
    //   s1 = Tr(D)
    //   s2 = <D,D>
    //   h0 = Tr(F0)
    //   h1 = <D,F0>
    // ------------------------------------------------------------

    float trD = 0.0f;
    float dd  = 0.0f;
    float trF = 0.0f;
    float dF  = 0.0f;

    for(int idx=lid; idx<NN; idx+=WG){

        int i = idx / N;
        int j = idx - i*N;

        float d  = Dg[idx];
        float f0 = Hg[idx] - Lg[idx];

        dd = fma(d,d,dd);
        dF = fma(d,f0,dF);

        if(i==j){
            trD += d;
            trF += f0;
        }
    }

    float4 rr = reduce4(
        red,
        (float4)(trD,dd,trF,dF),
        lid
    );

    if(lid==0){

        const float s1 = rr.x;
        const float s2 = rr.y;
        const float h0 = rr.z;
        const float h1 = rr.w;

        const float det = (float)N*s2 - s1*s1;

        // [N   s1][a] = [h0]
        // [s1  s2][b]   [h1]

        const float invdet = 1.0f / det;

        const float a =
            (h0*s2 - h1*s1) * invdet;

        const float b =
            ((float)N*h1 - s1*h0) * invdet;

        l_a  = a;
        l_b  = b;

        // X = cD*D - alpha*(H-Lambda) + alpha*a*I
        l_cD = fma(alpha,b,1.0f);

        // actual first-order trace after projected step
        float trFp =
            h0
          - a*(float)N
          - b*s1;

        l_trX = s1 - alpha*trFp;

        // TC2 ordering:
        // too many electrons -> contract first
        l_expand1 = (l_trX <= nocc[sys]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    // ============================================================
    // 1. FIRST SQUARE
    //
    // Construct X ON THE FLY while loading the K slabs.
    //
    // At completion:
    //
    //     registers : T = X²
    //     global D  : X
    //
    // ============================================================

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    float4 acc4 = (float4)(0.0f);
    float4 acc5 = (float4)(0.0f);
    float4 acc6 = (float4)(0.0f);
    float4 acc7 = (float4)(0.0f);

    #pragma unroll
    for(int k0=0; k0<LN; k0+=TK){

        // ---------------------------------------------
        // Cooperative staging.
        //
        // Every real matrix element is transformed
        // old D -> X exactly once.
        // ---------------------------------------------

        for(int e=lid; e<TK*LN; e+=WG){

            int m  = e / TK;
            int kk = e - m*TK;

            int k = k0 + kk;

            float x = 0.0f;

            if(m < N && k < N){

                int idx = m*N + k;

                float d  = Dg[idx];
                float f0 = Hg[idx] - Lg[idx];

                x = fma(l_cD,d,-alpha*f0);

                if(m==k)
                    x += alpha*l_a;

                // turn D into X in-place
                Dg[idx] = x;
            }

            // transposed staging:
            // Ls[k][row]
            Ls[kk*LN + m] = x;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // ---------------------------------------------
        // X² microkernel
        // ---------------------------------------------

        #pragma unroll
        for(int kk=0; kk<TK; ++kk){

            float8 av =
                vload8(0,&Ls[kk*LN + r0]);

            float4 bv =
                vload4(0,&Ls[kk*LN + c0]);

            acc0 = fma((float4)(av.s0),bv,acc0);
            acc1 = fma((float4)(av.s1),bv,acc1);
            acc2 = fma((float4)(av.s2),bv,acc2);
            acc3 = fma((float4)(av.s3),bv,acc3);
            acc4 = fma((float4)(av.s4),bv,acc4);
            acc5 = fma((float4)(av.s5),bv,acc5);
            acc6 = fma((float4)(av.s6),bv,acc6);
            acc7 = fma((float4)(av.s7),bv,acc7);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // All writes old-D -> X must now be visible to this WG.
    barrier(
        CLK_LOCAL_MEM_FENCE |
        CLK_GLOBAL_MEM_FENCE
    );


    // ------------------------------------------------------------
    // 2. Apply first quadratic retraction:
    //
    // expand:   Y = 2 X - X²
    // contract: Y = X²
    //
    // simultaneously learn Lambda correction.
    //
    // global D becomes Y.
    // ------------------------------------------------------------

    #define WRITE_ROW(R,ACC)                                      \
    {                                                             \
        int i = r0 + (R);                                         \
        if(i < N){                                                \
            float4 tv = (ACC);                                    \
            for(int cc=0; cc<RTX; ++cc){                          \
                int j = c0 + cc;                                  \
                if(j < N){                                        \
                    int idx = i*N+j;                              \
                    float x = Dg[idx];                             \
                    float t = tv[cc];                              \
                    float y = l_expand1 ?                          \
                        fma(2.0f,x,-t) : t;                        \
                                                                  \
                    float dl = beta*(y-x)/alpha;                   \
                    Lg[idx] += dl;                                 \
                    Dg[idx]  = y;                                  \
                }                                                 \
            }                                                     \
        }                                                         \
    }

    WRITE_ROW(0,acc0);
    WRITE_ROW(1,acc1);
    WRITE_ROW(2,acc2);
    WRITE_ROW(3,acc3);
    WRITE_ROW(4,acc4);
    WRITE_ROW(5,acc5);
    WRITE_ROW(6,acc6);
    WRITE_ROW(7,acc7);

    #undef WRITE_ROW


    if(!two_step)
        return;


    // All threads must see complete Y before Y².
    barrier(
        CLK_GLOBAL_MEM_FENCE |
        CLK_LOCAL_MEM_FENCE
    );


    // ============================================================
    // 3. SECOND SQUARE
    //
    // D already IS Y.
    // Reuse the same 7.7 kB local slab.
    // ============================================================

    acc0 = (float4)(0.0f);
    acc1 = (float4)(0.0f);
    acc2 = (float4)(0.0f);
    acc3 = (float4)(0.0f);
    acc4 = (float4)(0.0f);
    acc5 = (float4)(0.0f);
    acc6 = (float4)(0.0f);
    acc7 = (float4)(0.0f);

    #pragma unroll
    for(int k0=0; k0<LN; k0+=TK){

        for(int e=lid; e<TK*LN; e+=WG){

            int m  = e / TK;
            int kk = e - m*TK;
            int k  = k0 + kk;

            float y = 0.0f;

            if(m < N && k < N)
                y = Dg[m*N+k];

            Ls[kk*LN+m] = y;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(int kk=0; kk<TK; ++kk){

            float8 av =
                vload8(0,&Ls[kk*LN+r0]);

            float4 bv =
                vload4(0,&Ls[kk*LN+c0]);

            acc0 = fma((float4)(av.s0),bv,acc0);
            acc1 = fma((float4)(av.s1),bv,acc1);
            acc2 = fma((float4)(av.s2),bv,acc2);
            acc3 = fma((float4)(av.s3),bv,acc3);
            acc4 = fma((float4)(av.s4),bv,acc4);
            acc5 = fma((float4)(av.s5),bv,acc5);
            acc6 = fma((float4)(av.s6),bv,acc6);
            acc7 = fma((float4)(av.s7),bv,acc7);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    // ------------------------------------------------------------
    // 4. Complementary branch
    //
    // first expand  -> second contract
    // first contract -> second expand
    //
    // global D becomes final Z.
    // ------------------------------------------------------------

    const int expand2 = !l_expand1;

    #define WRITE_FINAL(R,ACC)                                    \
    {                                                             \
        int i = r0 + (R);                                         \
        if(i < N){                                                \
            float4 tv = (ACC);                                    \
            for(int cc=0; cc<RTX; ++cc){                          \
                int j = c0 + cc;                                  \
                if(j < N){                                        \
                    int idx = i*N+j;                              \
                    float y = Dg[idx];                             \
                    float t = tv[cc];                              \
                    float z = expand2 ?                            \
                        fma(2.0f,y,-t) : t;                        \
                                                                  \
                    Lg[idx] += beta*(z-y)/alpha;                   \
                    Dg[idx]  = z;                                  \
                }                                                 \
            }                                                     \
        }                                                         \
    }

    WRITE_FINAL(0,acc0);
    WRITE_FINAL(1,acc1);
    WRITE_FINAL(2,acc2);
    WRITE_FINAL(3,acc3);
    WRITE_FINAL(4,acc4);
    WRITE_FINAL(5,acc5);
    WRITE_FINAL(6,acc6);
    WRITE_FINAL(7,acc7);

    #undef WRITE_FINAL
}
```

That is essentially the full algorithm.

---

# 7. Notice how few genuinely expensive operations it contains

For the **one-purification-step** variant:

```text
O(N²):
    reduction for a,b
    construct X while loading

O(N³):
    one symmetric X²

O(N²):
    TC2 combination
    Lambda update
```

There is exactly:

$$
\boxed{1\ {\rm matrix\ square}}
$$

and **zero general matrix multiplications**.

For the strong two-step version:

```text
one symmetric X²
one symmetric Y²
```

so:

$$
\boxed{2\ {\rm matrix\ squares}}
$$

again with zero generic products.

---

# 8. Why the in-place sequence is safe

This deserves emphasis because initially it looks suspicious.

During the first GEMM we convert

$$
D_{\rm old}\rightarrow X
$$

as elements are staged.

This works because calculating an individual \(X_{ij}\) needs only

$$
D_{ij},\;H_{ij},\;\Lambda_{ij},
$$

not another element of \(D\).

And each \(D_{ij}\) is staged exactly once.

So there is no situation like

```text
thread A overwrites D[i,j]
thread B later needed old D[i,j]
```

The square uses the staged `Ls`, not global transpose loads.

After the complete \(X^2\) calculation, all reads of old \(D\) are finished permanently.

Then `D` can become \(Y\).

Likewise, after the complete \(Y^2\), all reads of \(Y\) are finished and `D` can become \(Z\).

---

# 9. Why one WG/system is necessary, but one WG/CU is not

These are very different requirements.

We need

```text
1 WG owns 1 matrix/system
```

because after

```text
D = Y
```

we need a global-memory synchronization before computing \(Y^2\).

OpenCL gives us a barrier only **within a workgroup**.

So if four WGs cooperated on one matrix, we could not safely do

```text
first square
global synchronization
second square
```

inside one kernel.

But nothing prevents:

```text
SM:
    WG system 13
    WG system 27
    WG system 94
    WG system 115
```

all being resident simultaneously.

And with ~12 kB local/WG, local-memory budget allows roughly four.

That's exactly the occupancy pattern we want.

---

# 10. `TK=22` is still a good choice here

I would actually **not** reduce it automatically to 8×8 local tiles.

Your measured standalone fixed-N result already says `TK=22` is excellent.

The local-memory footprint remains small:

$$
22\times88\times4=7.6\ {\rm kB}.
$$

Going to `TK=8` reduces that to

$$
2.8\ {\rm kB},
$$

but increases the number of K phases from 4 to 11 and therefore almost triples the staging barriers.

Since 7.6 kB already permits multiple WGs/SM, I would start with:

```text
TK=22
```

and only benchmark `TK=8/11/16` if register pressure causes the actual occupancy to differ.

---

# 11. One optimization I'd make immediately: separate fast and strong kernels at compile time

I wouldn't actually pass

```c
two_step
```

as runtime state in production.

Compile:

```text
relax_purify1
relax_purify2
```

or use

```c
#define NPUR 1
#define NPUR 2
```

so the compiler completely removes the unused half.

Then your algorithm can select:

```text
far from convergence:
    relax_purify1

near convergence / idempotency drifting:
    relax_purify2
```

Same source template, two specialized kernels.

---

# 12. We can probably merge the \(a,b\) reduction with useful diagnostics

That initial `float4` is currently:

```text
Tr(D)
<D,D>
Tr(H-Lambda)
<D,H-Lambda>
```

We could use `float8` and simultaneously calculate things like

$$
\|D^2-D\|
$$

only if a square result exists later, not initially.

But after the square, each thread has

$$
X^2
$$

and \(X\), so it can accumulate

$$
e_{\rm idem}
=
\sum_{ij}(X^2_{ij}-X_{ij})^2
$$

before writing \(Y\).

That is essentially free arithmetic.

Likewise after the final square.

Therefore the kernel can output per system:

```text
idempotency_before
idempotency_after
trace
||purification correction||
```

with one merged reduction.

Your previous measurements indicate reductions are not the important cost anyway.

---

# 13. An important numerical refinement

I would **not update \(\Lambda\)** if

$$
|z-y|
$$

or

$$
|y-x|
$$

is at the FP32 noise floor.

Something like:

```c
float corr = z-y;

if (fabs(corr) > corr_floor)
    Lambda[idx] += beta*corr/alpha;
```

Maybe don't do this per-element with a discontinuous threshold immediately; a better production criterion is per-system:

$$
\|D'-X\|_F > c\,\epsilon_{\rm fp32}.
$$

Otherwise stop learning \(\Lambda\).

Because near convergence,

$$
\frac{\text{roundoff}}{\alpha}
$$

would contaminate the learned constraint force.

---

## The resulting kernel architecture is actually very compact

Conceptually it is just:

```text
                 global D(old), H, Lambda
                           │
                           │ O(N²) reduction
                           ▼
                    determine a,b
                           │
                           ▼
        ┌──── stage small X K-slab ────┐
        │       and D <- X             │
        │                              │
        └──────── X² in regs ──────────┘
                           │
                           ▼
                 D <- q1(X)
                 Lambda correction
                           │
                      WG barrier
                           │
             [optional strong cleanup]
                           │
                           ▼
        ┌──── stage small D K-slab ────┐
        │                              │
        └──────── D² in regs ──────────┘
                           │
                           ▼
                 D <- q2(D)
                 Lambda correction
```

Local storage never exceeds roughly one `TK×88` slab plus reduction scratch.

So I think this gives you what you were asking for: **a genuinely single OpenCL kernel implementing approximate tangential energy descent + one/two constraint-restoration steps, while preserving the high-occupancy small-local-memory philosophy of the existing tuned GEMM.**



---

# USER

we tested it and it does not seem to work at all.Plese figure out if the problem is conceptual, or my coding agent just does not understand the pricniple and implemented somehting stupid with wrong assumtions. 

---

# Chat GPT 5.6 sol

I think the coding agent has mostly understood and implemented the proposed scheme correctly. **The major failure is conceptual: the learned matrix \(\Lambda\) was a bad idea in the form I proposed it.** Your agent has actually diagnosed the core problem rather well. The fact that the cold calculation converges to the correct variational minimum while warm starts converge to clean, idempotent, wrong projectors is almost a fingerprint of this failure mode. 

## 1. The fatal problem with learned \(\Lambda\)

Let \(P\) be an exact projector and \(Q=I-P\). Any symmetric matrix \(A\) decomposes into

$$
A=T_P(A)+N_P(A),
$$

where the tangent part is

$$
T_P(A)=PAQ+QAP
      =PA+AP-2PAP
$$

and the normal part is

$$
N_P(A)=PAP+QAQ.
$$

For a proper polynomial retraction \(R\), including your complementary two-square SP2 map, locally

$$
R(P+\epsilon A)
=
P+\epsilon T_P(A)+O(\epsilon^2).
$$

Your algorithm sets

$$
F=H-\Lambda-\text{gauge},
\qquad
X=P-\alpha F,
\qquad
P'=R(X).
$$

Therefore

$$
P'
=
P-\alpha T_P(F)+O(\alpha^2),
$$

while the part removed by purification is

$$
P'-X
=
\alpha N_P(F)+O(\alpha^2).
$$

The proposed learning rule was

$$
\Lambda'=
\Lambda+\beta\frac{P'-X}{\alpha}.
$$

Thus, to first order,

$$
\boxed{
\Lambda'=\Lambda+\beta N_P(H-\Lambda)
}
$$

and therefore

$$
\boxed{
T_P(\Lambda')=T_P(\Lambda).
}
$$

That is the bug in the *mathematics*.

**The learning channel is completely blind to tangent contamination already present in \(\Lambda\).**

Your notes identify exactly this problem: deposits are normal in the frame where they are generated, but as \(D\) rotates those old deposits acquire a tangent component; nothing in the update removes it. 

I missed this in the earlier proposal.

---

# 2. There is an even stronger proof: every projector can become an exact fixed point

This makes the problem unambiguously fatal for a standalone solver.

Take **any** clean rank-\(N_{\rm occ}\) projector \(P_*\), even a completely wrong one.

Choose

$$
\Lambda_*
=
H-aI-bP_*.
$$

Then

$$
F_0=H-\Lambda_*
=aI+bP_*.
$$

Your \(\{I,D\}\) removal finds exactly those \(a,b\), so

$$
F=0.
$$

Therefore

$$
X=P_*.
$$

And because it is already a projector,

$$
q_+(P_*)=P_*,
\qquad
q_-(P_*)=P_*.
$$

Consequently

$$
D'=P_*,
\qquad
D'-X=0,
\qquad
\Lambda'=\Lambda_*.
$$

So

$$
\boxed{\text{every rank-}N_{\rm occ}\text{ projector can be a fixed point}}
$$

regardless of whether

$$
[P_*,H]=0,
$$

and regardless of its energy.

Your notes derive the same degeneracy explicitly. 

That is not an implementation detail.

It means the iteration is solving

$$
T_D(H-\Lambda)=0
$$

rather than the physical condition

$$
\boxed{T_D(H)=0.}
$$

---

# 3. This explains the bizarre warm-start behavior perfectly

Suppose

$$
P_0
$$

is the converged ground-state projector of \(H_0\), so

$$
[P_0,H_0]=0.
$$

Then initialize, as I suggested,

$$
\Lambda_0=H_0.
$$

For

$$
H_1=H_0+\Delta H,
$$

the first force is

$$
F_0=\Delta H.
$$

That initially looks excellent. Its off-diagonal part indeed contains the perturbation which should rotate the occupied space.

But then \(P\) starts rotating.

While \(H_0\) was normal/block-diagonal in the frame of \(P_0\), it is **not** normal in the frame of the rotated \(P_1\).

It develops a tangent component

$$
T_{P_1}(H_0)\ne0.
$$

But \(\Lambda=H_0\) still contains it.

Therefore the physical force becomes

$$
T_P(H_1-\Lambda)
\approx
T_P(H_1-H_0)
=
T_P(\Delta H),
$$

rather than

$$
T_P(H_1).
$$

So the iteration tends toward an invariant subspace of \(\Delta H\), exactly as your agent observed numerically.

The measured warm run where canonicalization makes \(\Lambda\approx H_1\) and the dynamics drift catastrophically is therefore not mysterious at all. 

The agent's explanation:

> "`ΔH` forever"

is essentially correct.

---

# 4. The canonicalization kernel cannot repair this

`relax_canon_batched` adds

$$
aI+bD
$$

to \(\Lambda\).

But those matrices commute with \(D\), so they are purely normal:

$$
T_D(aI+bD)=0.
$$

Therefore this operation can change the **gauge** of \(\Lambda\), but it cannot remove the problematic quantity

$$
T_D(\Lambda).
$$

So canonicalization cannot cure the fundamental issue.

Likewise, simple decay

$$
\Lambda\leftarrow(1-\gamma)\Lambda
$$

doesn't distinguish:

* useful normal \(\Lambda\),
* harmful tangent \(\Lambda\).

It destroys both. The residual floor your agent observed with decay is therefore unsurprising. 

---

# 5. Could we salvage \(\Lambda\) by projecting it?

Yes, mathematically:

$$
\Lambda\leftarrow N_D(\Lambda)
$$

each iteration.

Equivalently remove

$$
T_D(\Lambda)
=
D\Lambda+\Lambda D-2D\Lambda D.
$$

Then by construction

$$
T_D(\Lambda)=0
$$

and hence

$$
T_D(H-\Lambda)=T_D(H).
$$

That would make the method correct.

Unfortunately this requires precisely the operations we were trying to avoid:

$$
D\Lambda
$$

and

$$
D\Lambda D.
$$

That's **two general matrix products**, not symmetric squares.

And I don't see a square-only shortcut. The information you need is specifically the *relative orientation* of \(D\) and \(\Lambda\); \(D^2\) and \(\Lambda^2\) cannot tell you their cross block \(D\Lambda Q\).

So I would **not salvage learned \(\Lambda\)**. It defeats the computational motivation.

---

# 6. Fortunately, we don't actually need \(\Lambda\)

This is the most important practical conclusion.

Keep the good part:

$$
\boxed{
F=H-aI-bD
}
$$

but make it **stateless**.

Recompute \(a,b\) from the current \(D,H\) every iteration.

Then

$$
X=D-\alpha(H-aI-bD)
$$

and retract:

$$
D'=R(X).
$$

For an exact projector,

$$
T_D(aI+bD)=0,
$$

so

$$
T_D(H-aI-bD)=T_D(H).
$$

Thus

$$
\boxed{
D'
=
D-\alpha T_D(H)+O(\alpha^2).
}
$$

This is exactly the correct constrained gradient descent to first order.

And it still uses:

* zero general GEMMs;
* O(\(N^2\)) reduction for \(a,b\);
* one or two of your optimized symmetric squares.

This was actually the essential useful idea. **The persistent learned matrix was unnecessary and poisonous.**

---

# 7. The \(aI+bD\) subtraction is still valuable for FP32

This part I would retain.

For an exact rank-\(m\) projector, minimizing

$$
\|H-aI-bD\|_F
$$

over \(a,b\) gives approximately

$$
a = \bar\epsilon_{\rm virt},
$$

$$
a+b = \bar\epsilon_{\rm occ}.
$$

So rather than stepping with the full \(H\), you're stepping with something whose normal blocks are approximately

$$
H_{vv}-\bar\epsilon_v I
$$

and

$$
H_{oo}-\bar\epsilon_o I.
$$

The large absolute energy offset and much of the occupied/virtual block average have disappeared.

That directly addresses your original FP32 concern without introducing state.

This is a much cleaner interpretation:

> \(aI+bD\) is a **fresh, cheap approximate normal-force removal**, not a learned Lagrange multiplier.

---

# 8. Why the agent's `β=0 cold fails` does NOT disprove this

This is quite important.

The mathematical argument

$$
R(D-\alpha H)
=
D-\alpha T_D(H)+O(\alpha^2)
$$

assumes \(D\) is already near the projector manifold.

Your cold Palser density is not necessarily close enough.

So the fact that the no-\(\Lambda\) cold calculation fails says almost nothing about the use case we actually care about:

> **warm relaxation from a previously converged projector.**

We already have an excellent cold solver: TC2 purification from \(H\). Its test explicitly verifies that the returned density reaches the variational minimum, not merely idempotency. 

Therefore I wouldn't force one algorithm to do both jobs.

Use:

```text
first electronic solve / genuinely cold:
    ordinary TC2 / Jacobi
```

then:

```text
warm SCC iterations / nearby geometry:
    stateless energy descent + polynomial retraction
```

That's likely the sensible architecture.

---

# 9. There ARE two important coding/testing mistakes

They aren't responsible for the fundamental failure, but I would fix them immediately.

### A. `done` does not test physical stationarity

The fused kernel declares convergence based on essentially

```text
step small
purification correction small
trace correct
```

but does **not** require

$$
[H,D]\approx0.
$$

So once \(\Lambda\) has successfully cancelled the force at a wrong \(D\), the code says “converged.”

That is precisely why you can obtain

```text
idempotency ≈ 1e-6
Tr = 43
comm ≈ 0.31
```

and still report success. 

For an experimental solver, convergence must ultimately include either

$$
\|HD-DH\|
$$

or an equivalent physical gradient criterion.

It doesn't have to be evaluated every iteration; every 4–8 iterations or at apparent convergence would suffice.

### B. The warm unit test calculates failure but doesn't assert it

The test computes

```rust
bad2 += 1;
```

for wrong warm densities.

But after the warm test there is no

```rust
assert_eq!(bad2, 0)
```

before it proceeds to the polish experiment.

At the end it only does:

```rust
assert!(diag2.converged)
```

which is exactly the biased internal convergence criterion.

So currently the test can report a successful warm solver even though its own energy/projector certification says the answer is wrong. 

That should definitely be fixed.

---

# 10. There is one actual inconsistency in the `ldecay` experiment

I noticed this in the kernel.

Pass 0 computes \(a,b\) from

$$
F_0=H-\Lambda.
$$

Then during staging you do

```c
if (ldecay > 0)
    Lambda *= 1-ldecay;
```

and only afterward construct \(X\) using the **decayed** \(\Lambda\).

Therefore the \(a,b\) coefficients and trace pin were computed from a different force than the one actually used to construct \(X\).

So with `ldecay>0`:

$$
\operatorname{Tr}X=N_{\rm occ}
$$

is no longer actually guaranteed.

That's a real implementation error in that experimental branch.

It doesn't explain the default `ldecay=0` warm failure, but I would not interpret any decay experiments quantitatively until that is fixed.

---

# 11. One more semantic oddity: trace pin + branch selection

You pin

$$
\operatorname{Tr}X=N_{\rm occ},
$$

but choose the first TC2 branch from

$$
\operatorname{Tr}D
$$

before the step.

Once \(D\) has the correct trace—which is almost always after the pin—

```c
l_expand1 = (Tr(D) <= Nocc)
```

will nearly always choose the same ordering:

$$
q_+\quad\text{then}\quad q_-.
$$

If the intent is **a complementary polynomial retraction**, that's fine: just make the order explicit and stop calling it a TC2 branch decision.

If the intent is actual trace-correcting TC2 semantics, then it isn't doing that.

I don't think this causes the warm bias, but cleaning up the conceptual semantics will make debugging easier.

---

# 12. The decisive next experiment is now very simple

I would stop changing \(\Lambda\). Remove it completely for one test.

Start from a **correct clean projector** \(D_0\) of \(H_0\).

Perturb:

$$
H_1=H_0+\delta H.
$$

Then iterate only:

$$
F=H_1-aI-bD,
$$

$$
X=D-\alpha F,
$$

$$
D'=q_-\!\left(q_+(X)\right)
$$

(or the opposite fixed complementary order).

Absolutely no \(\Lambda\).

Track every iteration:

$$
E=\operatorname{Tr}(DH),
$$

$$
r_H=\|HD-DH\|_F,
$$

$$
r_P=\|D^2-D\|_F,
$$

$$
\operatorname{Tr}D.
$$

What I predict is:

```text
energy      ↓
commutator  ↓
idempotency remains controlled
D → true new projector
```

If this fails even for small \(\delta H\), **then the retraction/minimization concept itself needs reconsideration**.

If it works—and mathematically I expect it to—then we've isolated the culprit cleanly:

$$
\boxed{\text{persistent learned }\Lambda\text{ was the mistake}.}
$$

---

## 13. We can still exploit \(H_{\rm old}\), but only as a one-step predictor

There was a good idea hidden in

$$
\Lambda_0=H_{\rm old}.
$$

The first force

$$
H_{\rm new}-H_{\rm old}
$$

is indeed a very nice small predictor.

The mistake was **keeping \(H_{\rm old}\) inside a persistent learned multiplier**.

Instead I would test:

```text
first warm step only:
    F = H_new - H_old
    remove aI+bD
    retract

all following steps:
    F = H_new
    remove aI+bD
    retract
```

No stateful \(\Lambda\).

That gives the FP32-friendly small first displacement we wanted without the “ΔH forever” pathology.

For an actual SCC loop, it may be even simpler: one such relaxation step, then recompute \(H[D]\). That is much closer to your original insight that **we should jointly relax \(D\) and SCC charge, rather than fully solve an inner fixed-\(H\) problem every time**. Your notes correctly identify that as a separate and potentially more appropriate regime. 

### So my verdict

**The agent did not implement something stupid.** The fused-square kernel appears faithful enough that the cold success is meaningful. The warm failure exposed a real flaw in my learned-\(\Lambda\) proposal.

I would now **delete persistent \(\Lambda\) from the algorithmic core**, retain the stateless \(aI+bD\) gauge removal, and test warm `raw gradient → 2-square retraction` directly. That version has the property we actually need:

$$
\boxed{\text{its first-order fixed-point condition is }[D,H]=0,}
$$

instead of

$$
[D,H-\Lambda]=0.
$$

That is the conceptual correction I would make before doing any more GPU optimization.


---

# USER

OK, so lets step back to the stanrad approach used in literature for Density matrixminimization (resp minimization of Tr(H*D), we should first implement the canonical method and only then try to invent our special speedup and hack for local memory and f32 accuracy. Standard method use the comutator [H,D]? Or what? 

Can we implement the standrd method efficiently on our GPU? minimizing the f32 noise before it emerge?


---

# Chat GPT 56. sol

Yes. I would now implement a **boringly canonical reference method first**, with no learned \(\Lambda\), no trace pinning hacks, and initially even no SCC feedback: hold \(H\) fixed and verify that the algorithm finds its lowest-\(N\) projector. Your current experimental code was explicitly trying to replace the eigensolver with a special 1–2-square minimizer, so resetting the mathematical baseline before optimizing it is sensible. 

The important distinction is:

**\([H,D]\) is the canonical stationarity residual, but it is not itself the density-matrix gradient you add to \(D\).** For an idempotent projector, the constrained steepest-descent direction is the **double commutator**. The classical LNV density-matrix-minimization method instead minimizes a McWeeny-transformed cubic functional; at an idempotent \(D\), its gradient reduces exactly to that same double commutator.

## 1. What is the actual mathematical problem?

In an orthogonal basis, at \(T=0\),

$$
D^2=D,\qquad D^T=D,\qquad \operatorname{Tr}D=N_{\rm occ},
$$

and we minimize

$$
E=\operatorname{Tr}(HD).
$$

The correct solution satisfies

$$
\boxed{[H,D]=HD-DH=0}.
$$

Together with idempotency and the correct rank, this means \(D\) projects onto an invariant subspace of \(H\); energy minimization selects the lowest \(N_{\rm occ}\) states. McWeeny already described idempotency plus commutation with the Hamiltonian as the necessary/sufficient SCF conditions. ([APS Journals][1])

But \([H,D]\) is antisymmetric for real symmetric \(H,D\), so you cannot simply do

$$
D\leftarrow D-\alpha[H,D].
$$

That would destroy symmetry.

---

# 2. The canonical constrained gradient is the double commutator

On the manifold of rank-\(N\) projectors, the projected gradient is

$$
G=[D,[D,H]].
$$

For \(D^2=D\),

$$
G=DH+HD-2DHD.
$$

The downhill equation is

$$
\boxed{
\dot D=-[D,[D,H]]
}
$$

or equivalently

$$
\boxed{
\dot D=[D,[H,D]].
}
$$

This has the properties we wanted all along:

$$
\operatorname{Tr}\dot D=0,
$$

and it is tangent to the projector manifold,

$$
D\dot D+\dot D D=\dot D.
$$

Therefore an Euler step damages idempotency only at second order:

$$
(D+\alpha\dot D)^2-(D+\alpha\dot D)
=O(\alpha^2).
$$

So yes: **the commutator is fundamental**, but the actual symmetric gradient is the double commutator.

---

# 3. Classical LNV density-matrix minimization is slightly different

The canonical Li–Nunes–Vanderbilt approach introduces an auxiliary symmetric matrix \(L\) and defines the physical density as the McWeeny polynomial

$$
D(L)=3L^2-2L^3.
$$

Then minimize the grand potential

$$
\boxed{
\Omega[L]
=
\operatorname{Tr}
\left[
(3L^2-2L^3)(H-\mu I)
\right].
}
$$

This is the classic 1993 LNV density-matrix-minimization method. Later implementations commonly use conjugate gradients; the literature explicitly distinguishes this direct minimization use of McWeeny purification from FOE purification of a fixed Hamiltonian. ([PubMed][2])

Let

$$
F=H-\mu I.
$$

Its exact matrix gradient is

$$
\boxed{
\nabla_L\Omega
=
3(LF+FL)
-
2(L^2F+LFL+FL^2).
}
$$

This is the formula standard CG-DMS/LNV implementations minimize. Conjugate-gradient implementations with analytic line minimization were developed precisely as alternatives to diagonalization. ([scuseria.rice.edu][3])

Now the nice connection:

if \(L=P\) is already idempotent,

$$
P^2=P,
$$

then

$$
\nabla\Omega
=
PH+HP-2PHP
=
[P,[P,H]].
$$

The chemical-potential term disappears from the tangent gradient.

So:

$$
\boxed{
\text{LNV gradient}
\;\xrightarrow{L^2=L}\;
\text{double-commutator gradient}.
}
$$

That explains why both viewpoints keep reappearing.

---

# 4. Which should we implement first?

I would actually implement **two reference modes**, but in this order:

1. **Exact LNV steepest descent** for fixed \(H\). This is the canonical literature DMM baseline.
2. **Double-commutator descent** starting from an idempotent warm projector. This is the canonical projector-manifold baseline.

They should converge to the same \(D_{\rm ref}\).

Do **not** initially use CG, DIIS, SCC feedback, learned multipliers, adaptive purification, etc. First establish that the gradient is correct.

Once steepest descent works reliably, replace it by nonlinear CG.

---

# 5. There is a very good FP32 form of the LNV gradient

This is where I think we can already design the implementation intelligently without changing the mathematics.

Do **not** evaluate

$$
3(LF+FL)-2(L^2F+LFL+FL^2)
$$

literally near convergence.

That is exactly your FP32 nightmare:

$$
\text{small gradient}
=
\text{several large matrices}
-
\text{several large matrices}.
$$

Instead algebraically rewrite the exact same gradient as

$$
\boxed{
\nabla\Omega
=
[L,[L,F]]
+
3\left[
(L-L^2)F
+
F(L-L^2)
\right].
}
$$

This identity is exact.

Check it:

$$
[L,[L,F]]
=
L^2F+FL^2-2LFL
$$

and

$$
3[(L-L^2)F+F(L-L^2)]
=
3LF+3FL-3L^2F-3FL^2.
$$

Adding gives exactly the LNV gradient.

### Why this is ideal for FP32

Near the answer there are two physically small residuals:

$$
C=[L,F]
$$

— failure to commute —

and

$$
R=L-L^2
$$

— failure of idempotency.

Then the gradient is built **from those small residuals**, instead of creating large terms and hoping they cancel.

That is exactly the numerical philosophy you asked for:

> minimize the noise **before it emerges**, rather than calculate noisy large quantities and subtract them afterward.

---

# 6. Very concrete GPU evaluation

For each system:

### A. Idempotency residual

Use your optimized symmetric square:

$$
L_2=L^2.
$$

Then elementwise

$$
R=L-L_2.
$$

Cost:

$$
\boxed{1\text{ symmetric square}}
$$

and \(O(n^2)\).

### B. Commutator

Compute only

$$
A=LF.
$$

One general GEMM.

Since \(L,F\) are symmetric,

$$
FL=A^T.
$$

Therefore construct

$$
\boxed{
C=A-A^T.
}
$$

No second GEMM.

And importantly, construct it explicitly antisymmetric:

```c
C[i,j] = A[i,j] - A[j,i];
C[j,i] = -C[i,j];
C[i,i] = 0;
```

So FP32 cannot gradually introduce a spurious symmetric part.

### C. Double commutator

Compute

$$
B=LC.
$$

One general GEMM.

Because \(L^T=L\) and \(C^T=-C\),

$$
(LC)^T=-CL.
$$

Therefore

$$
\boxed{
[L,C]=LC-CL=B+B^T.
}
$$

Again no second product.

Construct it explicitly symmetric:

```c
Kij = B[i,j] + B[j,i];
Kji = Kij;
```

### D. Non-idempotency correction

Compute

$$
M=RF.
$$

One general GEMM.

Since \(R,F\) are symmetric,

$$
FR=M^T.
$$

Therefore

$$
3(RF+FR)
=
3(M+M^T).
$$

Finally,

$$
\boxed{
G=(B+B^T)+3(M+M^T).
}
$$

So exact canonical LNV gradient costs

$$
\boxed{
1\ {\rm symmetric\ square}
+
3\ {\rm general\ GEMMs}.
}
$$

No approximation.

---

# 7. This form should be substantially quieter in FP32 than the textbook formula

Compare the naive evaluation near convergence:

```text
A = L F               ~ O(H)
B = L² F              ~ O(H)
C = L F L             ~ O(H)

G = 3(A+Aᵀ)-2(B+Bᵀ+C)
```

You are subtracting quantities of ordinary electronic-energy magnitude.

Instead:

```text
Ccomm = [L,F]          small
Ridem = L-L²           small

B = L*Ccomm            small
M = Ridem*F            small

G = B+Bᵀ + 3(M+Mᵀ)    small
```

There is still **one unavoidable cancellation**

$$
C=A-A^T,
$$

because physically the commutator itself is a small difference.

But after that, everything operates on the small residual.

This is much better than having three separate \(O(H)\) cancellations.

---

# 8. Shift and scale \(H\) before doing anything

I think this is essential for FP32.

Set

$$
F=
\frac{H-\mu I}{\Delta},
$$

where \(\Delta\) is roughly the spectral width.

Positive scaling does not change the minimizer, and \(\mu\) should lie in the HOMO-LUMO gap for the desired occupation.

Then eigenvalues of \(F\) are roughly \(O(1)\), ideally something like

$$
[-0.5,+0.5]
$$

or

$$
[-1,+1].
$$

Advantages:

* GEMM operands stay around unity;
* gradient magnitudes have predictable scale;
* step size becomes dimensionless;
* absolute orbital-energy offsets don't contaminate FP32;
* the commutator is completely invariant to the identity shift anyway:

$$
[H-\mu I,D]=[H,D].
$$

For the first reference test I would **cheat deliberately** and obtain \(\mu\) from the CPU/Jacobi reference:

$$
\mu=\frac{\epsilon_{\rm HOMO}+\epsilon_{\rm LUMO}}2.
$$

We are testing DMM correctness, not chemical-potential estimation.

Only after it works should we replace that with the practical estimator.

---

# 9. Use an orthogonal basis first

This is also important.

A nonorthogonal LNV formulation exists; Nunes and Vanderbilt explicitly generalized it so only \(S\), not \(S^{-1}\), appears in the functional. ([APS Journals][4])

But CG-DMS implementations commonly transform Fock and density matrices to an orthonormal representation, and Challacombe's later work reported the nonorthogonal minimization as uncompetitive with minimization in the orthogonal representation. ([SciSpace][5])

For us that also improves numerical cleanliness enormously.

So initially:

$$
\tilde H=X^THX,\qquad X^TSX=I,
$$

do the entire DMM in the orthogonal basis, and only transform the final density back.

Do **not** mix the \(S\)-metric into the first debugging exercise.

---

# 10. Canonical update: start with steepest descent

After computing

$$
G=\nabla_L\Omega,
$$

just do

$$
L_{k+1}=L_k-\alpha G.
$$

Because we've scaled \(F\) to order unity, start conservatively, e.g.

$$
\alpha\sim0.1-0.25.
$$

For a scalar eigenvalue \(l\), the LNV gradient is

$$
\frac{d\Omega}{dl}
=
6 f\,l(1-l).
$$

Near \(l=0\) or \(1\), stability is controlled by approximately \(6\alpha|f|\). So if

$$
|f|\lesssim1,
$$

then

$$
\alpha<\frac13
$$

is a sensible conservative range.

I would initially use a fixed value like

$$
\boxed{\alpha=0.2}.
$$

No line search.

No momentum.

No conjugate gradient.

If that cannot converge for a fixed random symmetric \(H\), something is wrong with the implementation.

---

# 11. Then add the actual standard acceleration: nonlinear CG

Once steepest descent is verified, use Polak–Ribière:

$$
P_k=-G_k+\beta_kP_{k-1},
$$

with

$$
\beta_k=
\max\left(
0,
\frac{\langle G_k,G_k-G_{k-1}\rangle}
     {\langle G_{k-1},G_{k-1}\rangle}
\right).
$$

Historical CG-DMS implementations use nonlinear conjugate gradient and exploit the fact that the LNV functional along a line is cubic, so an analytic line minimum reduces to solving a quadratic. ([sma.epfl.ch][6])

But that's **phase two**.

I'd first establish correct SD behavior.

---

# 12. Scalars can cheaply be more accurate than matrices

You are absolutely right that full FP64 matrix algebra is unacceptable on a 3090.

But that does **not** imply every scalar reduction must be FP32.

The \(O(n^3)\) work remains FP32.

For quantities such as

$$
\operatorname{Tr}D,
\quad
E,
\quad
\|G\|^2,
\quad
\langle G_k,G_{k-1}\rangle,
$$

there are only \(O(n^2)\) terms.

Three sensible possibilities are:

1. FP32 pairwise tree reduction;
2. compensated FP32 (`float2`: sum + correction);
3. FP64 only in the final reduction.

Even on an RTX 3090, the third is likely negligible next to three \(86^3\) GEMMs because you're doing thousands rather than millions of FP64 operations.

So I'd happily use:

```text
matrix products     FP32
matrix storage      FP32
elementwise update  FP32
scalar diagnostics  compensated FP32 or FP64
```

That is a very different proposition from FP64 GEMM.

---

# 13. The commutator implementation itself can be made exceptionally clean

This is worth emphasizing.

Never calculate separately

$$
HD
$$

and

$$
DH.
$$

For symmetric \(H,D\), compute only

$$
A=HD.
$$

Then

$$
DH=A^T.
$$

Thus

$$
[H,D]=A-A^T.
$$

This gives three advantages:

* one GEMM instead of two;
* both terms originate from exactly the same GEMM implementation;
* you can enforce perfect antisymmetry.

Likewise, to make the double commutator:

$$
C=[H,D],
$$

compute

$$
B=DC
$$

once.

Then

$$
CD=-B^T,
$$

so

$$
[D,C]=B+B^T.
$$

Thus the canonical projector gradient needs just

$$
\boxed{2\ {\rm general\ GEMMs}}
$$

rather than four.

And numerically its entire implementation is:

```text
A = H D
C = A - Aᵀ            # exact antisymmetry

B = D C
G = B + Bᵀ            # exact symmetry
```

This is probably the **cleanest possible FP32 implementation of the tangent gradient**.

---

# 14. This gives us an excellent cross-check against LNV

Take exactly idempotent \(D\).

Compute independently:

### Method A — LNV gradient

$$
G_{\rm LNV}
=
3(DH+HD)
-
2(D^2H+DHD+HD^2).
$$

### Method B — commutator route

$$
C=DH-HD,
$$

$$
G_{\rm comm}=[D,C].
$$

Since \(D^2=D\),

$$
\boxed{
G_{\rm LNV}=G_{\rm comm}.
}
$$

That is a fantastic unit test.

Generate random symmetric \(H\), make an exact rank-43 projector \(D=VV^T\), and compare both in CPU f64 and GPU f32.

That should catch sign/transposition/layout bugs immediately.

---

## 15. The baseline test sequence I would give the coding agent

I would now stop modifying the existing `relax_step_batched` and create a **separate clean reference implementation**.

1. Fixed random symmetric \(H\), \(n=86\), \(N_{\rm occ}=43\). CPU eigensolver supplies exact \(D_{\rm ref}\), \(\mu\), spectral width.
2. Orthogonal basis only.
3. Implement **`lndv_grad_batched`** using the exact residual-based identity

$$
G=[L,[L,F]]
+3[(L-L^2)F+F(L-L^2)].
$$

4. Validate every intermediate against CPU f64:

   * \(L^2\),
   * \(C=[L,F]\),
   * \(R=L-L^2\),
   * double-commutator term,
   * idempotency term,
   * full gradient.
5. Fixed-step steepest descent, say \(\alpha=0.2\) after scaling \(F\).
6. Certify every few iterations with

$$
\Delta E,\quad
\|D-D_{\rm ref}\|,\quad
\|D^2-D\|,\quad
\|[H,D]\|.
$$

7. Separately implement the simpler projector-gradient reference

$$
A=HD,\quad
C=A-A^T,\quad
B=DC,\quad
G=B+B^T.
$$

8. Starting from an exact projector, verify LNV and double-commutator gradients agree to FP32 tolerance.
9. Only after both work: nonlinear CG / analytic line search.
10. Only after **that**: SCC \(H[D]\), warm starts, kernel fusion, square-specialization tricks.

The literature's standard LNV method is variational and was specifically designed to impose the idempotency constraint during density-matrix minimization; later work found purification and minimization complementary and developed hybrids, so after the canonical baseline works we can safely revisit your extremely fast square primitive. ([ScienceDirect][7])

The part I am most optimistic about for FP32 is the exact identity

$$
\boxed{
\nabla\Omega
=
[L,[L,F]]
+
3[(L-L^2)F+F(L-L^2)].
}
$$

It is **not a new algorithm or approximation**. It is simply the textbook LNV gradient rearranged so that the GPU works on the two quantities that should physically go to zero—commutation error and idempotency error—rather than manufacturing them by cancellation of several large \(O(H)\) matrices. That is where I would start.

[1]: https://journals.aps.org/pr/abstract/10.1103/PhysRev.126.1028?utm_source=chatgpt.com "Perturbation Theory for the Fock-Dirac Density Matrix | Phys. Rev."
[2]: https://pubmed.ncbi.nlm.nih.gov/10005210/?utm_source=chatgpt.com "Density-matrix electronic-structure method with linear system-size scaling."
[3]: https://scuseria.rice.edu/list.html?utm_source=chatgpt.com "List of Publications"
[4]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.50.17611?utm_source=chatgpt.com "Generalization of the density-matrix method to a nonorthogonal basis | Phys. Rev. B"
[5]: https://scispace.com/topics/state-transition-matrix-4n4n3ris/1997?utm_source=chatgpt.com "Top 59 papers published in the topic of State-transition matrix in 1997"
[6]: https://sma.epfl.ch/~anchpcommon/students/bolliger.pdf?utm_source=chatgpt.com "mcweeny.tex"
[7]: https://www.sciencedirect.com/science/article/abs/pii/S0010465599002210?utm_source=chatgpt.com "Density matrices in O(N) electronic structure calculations: theory and applications - ScienceDirect"




---

# USER

please review this commit
https://github.com/ProkopHapala/dftbplus/commits/main/
and ttched report 

my coding agnt tried to impement your instruction but it found it either not working at all or super inefficient (require hundertsn or thousands iterations wich make is slower than jacobi and much slower than doing purification from scratch (from radnom DM)

I refuse to accept this as final results, the information sotred in density matrix from previous similar configuration (if atoms move by 0.1A, charges does not change almost) must be usefull starting point for warmstart, I suspect we do something totally silly conceptually aporach it from wrong direction

either my coding agent iterpreted your ideas in stupid way, or your ideas were stupid, reinvestigate this problem deepely again with open mind.


---

# Chat GPT 5.6 sol

I reviewed the latest commit, `359ee76cec82baefcc309bb26c6473ce33d447d5`, including the new LNV implementation, the pre-existing DMM path, `tc2_extrap_batched`, and the attached report.

I think your objection is justified. **The experiments do not show that a previous density matrix is useless. They show that “take the old projector and minimize \(\mathrm{Tr}(HD)\) by first-order gradient descent” is the wrong way to exploit it.** We were using the warm information in the weakest possible form.

There are also two important things the current report misses: one concrete problem in the extrapolation implementation, and a standard matrix-only Newton/response formulation that directly supplies the missing inverse-gap information.

## 1. The LNV result is real — I would stop working on it

Here I do **not** think the coding agent made a stupid implementation.

The new GPU LNV gradient was compared against an independent CPU/f64 implementation and agrees to roughly \(10^{-6}\) relative accuracy. The independent CPU implementation reproduces the thousands-of-iterations behavior. The earlier product-order bug was also found and fixed. 

So the conclusion

$$
\text{LNV steepest descent from }P_{\rm old}
$$

is a bad warm solver for these small dense systems is credible.

Likewise the exact tangent descent

$$
G=[P,[P,H]]
$$

is mathematically correct, but its convergence mode-by-mode is

$$
\theta_{ia}^{(k+1)}
\approx
\left[1-\alpha(\epsilon_a-\epsilon_i)\right]\theta_{ia}^{(k)}.
$$

If

$$
\kappa\sim
\frac{\Delta\epsilon_{\rm spectrum}}
     {\Delta\epsilon_{\rm HOMO-LUMO}},
$$

then plain first-order descent requires \(O(\kappa)\) iterations. The report's dense random half-filled systems deliberately generate \(\kappa\sim300-4000\), so hundreds or thousands of steps are entirely plausible. 

No GPU optimization fixes that.

So **my previous suggestion to make tangent/LNV minimization the main warm solver was conceptually poor.**

---

# 2. But the comparison to “cold purification” is misleading

This is the crucial point.

Cold TC2 is not starting from a random density matrix.

You initialize with

$$
X_0=
\frac{\lambda_{\max}I-H_{\rm new}}
     {\lambda_{\max}-\lambda_{\min}}.
$$

Therefore

$$
[X_0,H_{\rm new}]=0
$$

**exactly**.

So from iteration zero, \(X_0\) already has the exact eigenvectors of the **new Hamiltonian**.

TC2 never has to find the new occupied subspace. It only performs scalar operations

$$
x_i\rightarrow 0\ \text{or}\ 1
$$

on eigenvalues whose eigenvectors are already correct.

That is why 20–40 ridiculously cheap matrix squares beat warm gradient descent.

The report recognizes this later. 

So we have actually been comparing:

$$
\boxed{\text{exact new eigenbasis + eigenvalue cleanup}}
$$

against

$$
\boxed{\text{old eigenbasis + slow first-order rotation}}.
$$

No surprise the supposedly “cold” method wins.

The question is therefore not:

> How do we minimize from \(P_{\rm old}\) faster?

It is:

> **How do we predict the rotation of the occupied subspace before purification?**

That is a very different problem.

---

# 3. The report goes too far when it says the density matrix has thrown away the gap information

This statement in the report is only half true:

> the density matrix alone does not contain the denominators \(1/(\epsilon_a-\epsilon_i)\). 

A projector \(P\) alone indeed contains only the occupied subspace.

But we do **not** have only \(P\). We also have the previous Hamiltonian \(H_0\).

And because

$$
[P,H_0]=0,
$$

the pair

$$
\boxed{(P,H_0)}
$$

contains all of the occupied and virtual Hamiltonian blocks:

$$
H_{oo}=PH_0P,
$$

$$
H_{vv}=QH_0Q,
\qquad Q=I-P.
$$

The first-order response to

$$
H_1=H_0+\Delta H
$$

is obtained from the **Sylvester equation**

$$
\boxed{
H_{vv}X-XH_{oo}
=
-Q\,\Delta H\,P
}
$$

and

$$
\delta P=X+X^T.
$$

In the eigenbasis this becomes simply

$$
(\epsilon_a-\epsilon_i)X_{ai}
=
-\Delta H_{ai},
$$

hence

$$
X_{ai}
=
-\frac{\Delta H_{ai}}
       {\epsilon_a-\epsilon_i}.
$$

There are our missing denominators.

**No eigenvectors are required to formulate or solve this.**

This is not speculative. Density-matrix perturbation theory has been developed specifically to obtain density response without sum-over-states/eigenvectors; both recursive purification formulations and Sylvester-equation formulations exist. ([APS Journals][1])

So one major branch that the current campaign has simply not tested is:

$$
\boxed{\text{matrix-only Newton / Sylvester response}}
$$

rather than gradient descent.

---

# 4. And I found a concrete problem with the extrapolation experiment

This is potentially very significant.

Current `gpu_purify.cl::tc2_extrap_batched` does

$$
S=2P_1-P_0
$$

which is actually a good idea.

But then it computes **Gershgorin bounds of \(S\)** and, whenever they escape approximately \([0,1]\), affinely rescales the whole matrix into that interval.

For a dense projector, Gershgorin bounds are completely inappropriate for this decision.

Take a perfectly exact random rank-43 projector of size 86:

$$
P^2=P,
\qquad
\sigma(P)=\{0,1\}.
$$

As a representative numerical check, I get Gershgorin bounds around

$$
[-3.4,\;4.4]
$$

despite the exact spectrum being only 0 and 1.

So your condition

```text
lmin < -0.02 || lmax > 1.02
```

will trigger on an **absolutely perfect projector** in a dense representation.

The affine rescaling would then turn true eigenvalues roughly

$$
0\rightarrow0.44,
\qquad
1\rightarrow0.56.
$$

You have just destroyed the thing that made the warm seed valuable!

The report says this renormalization “preserves eigenvectors/ordering,” which is true, but it destroys the near-idempotency and forces purification to redo almost the whole spectral separation.

That is already bad.

But there is a second issue.

---

# 5. TC2 is the wrong retraction for an extrapolated projector

Consider two nearby projectors:

$$
P_{k-1},P_k.
$$

The natural linear extrapolation is

$$
S=2P_k-P_{k-1}.
$$

For smooth motion,

$$
P_{k-1}
=
P_k-h\dot P+\frac12h^2\ddot P+\cdots,
$$

so

$$
S
=
P_k+h\dot P-\frac12h^2\ddot P+\cdots.
$$

The true next projector is

$$
P_{k+1}
=
P_k+h\dot P+\frac12h^2\ddot P+\cdots.
$$

Therefore

$$
\boxed{S-P_{k+1}=O(h^2).}
$$

This is already a **second-order prediction** of the occupied-subspace motion.

But because \(\dot P\) is tangent, \(S\)'s eigenvalues naturally overshoot slightly:

$$
\lambda_{\rm occ}=1+O(h^2),
\qquad
\lambda_{\rm virt}=0-O(h^2).
$$

And that is completely harmless.

In fact **McWeeny is almost ideal for exactly this situation**:

$$
R(x)=3x^2-2x^3.
$$

Around both endpoints,

$$
R'(0)=R'(1)=0,
$$

so the radial \(O(h^2)\) eigenvalue errors get suppressed quadratically.

But on a tangent perturbation the Fréchet derivative is identity:

$$
DR_P[\delta P_{\rm tangent}]
=
\delta P_{\rm tangent}.
$$

So McWeeny:

> **kills the radial extrapolation error while preserving the predicted subspace rotation to first order.**

That is exactly what we need.

TC2 is not like that. One TC2 branch has derivative 2 at one endpoint:

$$
x^2:
\quad
q'(1)=2,
$$

while

$$
2x-x^2:
\quad
q'(0)=2.
$$

So if extrapolation produces one eigenvalue slightly above 1 and another slightly below 0—which it naturally does—whichever TC2 branch you pick repairs one side and makes the other worse.

That explains the report's observation that extrapolation followed by TC2 drifts or branch-misfires. 

I would therefore **reopen extrapolation immediately**. Its current failure does not falsify the idea.

---

# 6. The most promising architecture now looks like predictor → retraction → tiny correction

This is, I think, the fundamental change.

Do **not** ask DMM to generate the entire rotation from a stale \(P\).

Make history generate almost all of it.

Suppose we have

$$
P_{k-2},P_{k-1}
$$

and now know \(H_k\).

For a uniform trajectory use

$$
S=2P_{k-1}-P_{k-2}.
$$

Even better, adapt the extrapolation coefficient to the actual Hamiltonian movement:

$$
\Delta H_k=H_k-H_{k-1},
$$

$$
\Delta H_{k-1}=H_{k-1}-H_{k-2},
$$

$$
\gamma=
\frac{\langle\Delta H_k,\Delta H_{k-1}\rangle}
{\|\Delta H_{k-1}\|^2},
$$

then

$$
\boxed{
S=P_{k-1}+
\gamma(P_{k-1}-P_{k-2}).
}
$$

For constant motion, \(\gamma\simeq1\).

Then:

$$
P_{\rm pred}=3S^2-2S^3.
$$

Now calculate

$$
R_H=[H_k,P_{\rm pred}].
$$

If the predictor worked, perhaps **zero or one DMM correction** is enough.

This is exactly the kind of previous-density extrapolation that has been shown useful in geometry optimization and molecular dynamics. Niklasson et al. explicitly developed trace-correcting density-matrix extrapolation to accelerate self-consistency, and modern Grassmann extrapolation methods are based on the same observation that previous converged densities contain valuable trajectory information. ([DOI][2])

---

# 7. DFTB gives you an even better version almost for free

At fixed geometry, your SCC Hamiltonian is affine in charges:

$$
H(q)=H_0+Bq.
$$

And you already have DIIS/mixing machinery for the charge trajectory.

Suppose DIIS predicts

$$
q_{\rm next}
=
\sum_i c_i q_i,
\qquad
\sum_i c_i=1.
$$

Then exactly,

$$
H(q_{\rm next})
=
\sum_i c_i H(q_i).
$$

For nearby solutions, to first order,

$$
P(H_{\rm next})
\approx
\sum_i c_i P_i.
$$

So instead of inventing an unrelated density predictor, use **the same DIIS coefficients**:

$$
\boxed{
S=\sum_i c_i P_i
}
$$

and then retract \(S\) to the Grassmann/projector manifold.

That is almost free:

* you already computed the DIIS coefficients;
* combining 3–6 old matrices is \(O(n^2)\);
* one McWeeny retraction;
* one commutator certificate;
* perhaps one DMM correction.

This is much closer to the mathematics of the SCC trajectory than “16 fixed DMM steps from the previous projector.”

I think this deserves very high priority.

---

# 8. Product accounting becomes completely different

The report currently talks about roughly five matrix products per tangent-descent iteration. That is also slightly pessimistic.

Your actual `dmm_update_batched` is:

$$
T=HP
$$

and

$$
Y=PT=PHP,
$$

then

$$
G=T+T^T-2Y.
$$

That's **two general products**, not three.

McWeeny adds:

$$
P^2,
\qquad
P^3=P^2P,
$$

two more products.

If you retract every second DMM step, that's

$$
\frac{2+2+2}{2}
=
3
$$

matrix products per DMM step on average, not five.

Still too expensive for 30–300 steps, yes.

But predictor-corrector could look like:

| operation                    | matrix products |
| ---------------------------- | --------------: |
| extrapolate history          |               0 |
| McWeeny retract              |               2 |
| one DMM correction           |               2 |
| final commutator certificate |               1 |
| **total**                    |           **5** |

Now we are finally in the regime that can plausibly beat a ~30-square cold TC2.

That is the correct target: **0–2 correction steps, not 30–300 minimization steps.**

---

# 9. If one old projector is all we have, use a Newton response — not gradient descent

For the first geometry step, or when trajectory history is unavailable, the Sylvester formulation is the principled method.

At the old converged solution:

$$
[P,H_0]=0.
$$

For \(H_1=H_0+\Delta H\), solve

$$
\mathcal A(X)
=
-[P,[P,\Delta H]]
$$

where

$$
\boxed{
\mathcal A(X)
=
[P,[X,H_0]].
}
$$

For a tangent \(X\), in the unknown eigenbasis:

$$
\mathcal A(X)_{ia}
=
(\epsilon_a-\epsilon_i)X_{ia}.
$$

So this operator **is exactly the orbital-energy-gap Hessian** we have been missing.

This is Newton/PT1 expressed without eigenvectors.

And it is cheap to apply with your existing generic GEMM.

Given symmetric \(X,H_0,P\):

$$
M=H_0X
$$

one GEMM.

Then

$$
C=XH_0-H_0X=M^T-M.
$$

Then

$$
N=PC
$$

second GEMM.

Because \(C^T=-C\),

$$
\boxed{
\mathcal A(X)=N+N^T.
}
$$

So one Hessian-vector application costs exactly

$$
\boxed{2\text{ general GEMMs}.}
$$

Solve with CG in the tangent space.

Gradient descent convergence is \(O(\kappa)\).

CG/Sylvester convergence is roughly \(O(\sqrt\kappa)\).

For the pathological random systems with \(\kappa=1000\), this may still need tens of iterations. For a realistic gapped DFTB system with a much smaller effective condition number, it might require only 2–5.

The literature explicitly contains density-matrix-only Sylvester formulations of perturbation theory, so this is not a speculative construction. ([UCL Discovery][3])

And the older recursive DMPT work provides another matrix-only route using perturbed purification with quadratic recursions. ([arXiv][4])

I would prototype the Sylvester version in CPU/f64 before GPU optimization.

---

# 10. There is another route I now think may be even better: stop storing only \(P\)

This doesn't mean going back to eigenvectors.

The physical information you need is the **occupied subspace**, not individual canonical eigenvectors.

Represent it by any orthonormal matrix

$$
C\in\mathbb R^{n\times m},
\qquad
C^TC=I,
$$

with

$$
P=CC^T.
$$

There is no requirement that columns of \(C\) diagonalize \(H\).

Now use the previous \(C\) and apply a polynomial filter of the new Hamiltonian:

$$
C'\sim p_k(H_{\rm new})C.
$$

Then orthonormalize its columns.

This is **Chebyshev-filtered subspace iteration**.

And this is exactly an established solution to almost exactly your problem: in SCF calculations, keep the previously converged occupied subspace and refine it with polynomial filtering instead of solving a fresh eigenproblem. The original CheFSI work reports subspace updates substantially cheaper than repeated eigenvalue solves. ([ScienceDirect][5])

For your dimensions,

$$
n=86,\qquad m\approx43,
$$

one multiplication

$$
H C
$$

costs only about half a full \(86\times86\) GEMM.

A degree-4 filter therefore costs roughly two full-GEMM equivalents.

Then orthogonalizing an \(86\times43\) matrix is also much cheaper than a full eigensolve.

And here's the particularly attractive part for your existing SCC code:

**you do not necessarily have to rebuild \(P\) at all.**

You already have direct population machinery based on occupied coefficients \(C\) and \(SC\).

So a future architecture could be

$$
\boxed{
C_{\rm old}
\rightarrow
p(H_{\rm new})C_{\rm old}
\rightarrow
\text{orthonormalize}
\rightarrow
\text{Mulliken charges directly}.
}
$$

No diagonalization.

No \(P^2=P\) purification.

No DMM.

No density matrix unless some downstream operation explicitly needs it.

This is a much more serious candidate than the `C`-gradient method discussed in the report. The report's Q7 only considers

$$
HC-C(C^THC),
$$

which is again first-order gradient descent and suffers the same condition-number problem. 

**Polynomial-filtered subspace iteration was not tested.**

---

# 11. We can obtain \(C\) from your purified density without diagonalizing it

If the first calculation is TC2 and gives only \(P\), that's fine.

Choose a fixed random matrix

$$
\Omega\in\mathbb R^{n\times m}.
$$

Compute

$$
Y=P\Omega.
$$

Since \(P\) is rank \(m\),

$$
\operatorname{range}(Y)
=
\operatorname{range}(P)
$$

with probability one.

Then orthonormalize \(Y\):

$$
C=\operatorname{orth}(Y).
$$

That's a rectangular multiplication plus a small QR/CholeskyQR.

No eigensolver.

From then on carry \(C\), not just \(P\).

This is probably worth a completely separate prototype.

---

# 12. Geometry changes need another correction to the report

The report says, roughly,

$$
K_{\rm new}=B K_{\rm old}B^T,
\qquad
B=X_{\rm new}^{-1}X_{\rm old},
$$

“preserves the physical density.” 

That statement is not quite correct when atoms move.

The AO basis functions themselves move.

If we want to project the **old physical orbitals** onto the new AO basis, the natural transport includes the old–new cross overlap

$$
S_{\rm cross}
=
\langle\phi^{\rm new}_\mu|
\phi^{\rm old}_\nu\rangle.
$$

In orthogonal coordinates the transport is approximately

$$
\boxed{
B
=
X_{\rm new}^T
S_{\rm cross}
X_{\rm old}.
}
$$

Then

$$
K_{\rm new}^{\rm guess}
=
B K_{\rm old}B^T.
$$

The simple \(X_{\rm new}^{-1}X_{\rm old}\) formula preserves an AO **coefficient representation** under a particular identification of old/new AO labels; it does not literally preserve the physical orbital functions when those basis functions have moved.

For a 0.1 Å displacement this may matter.

Nonorthogonal DMPT was developed specifically to deal with perturbation-dependent bases and nuclear displacement, so there is established theory here too. ([PubMed][6])

Within one SCC loop at fixed geometry this issue disappears because \(X\) is fixed.

---

# 13. The dense-random benchmark should no longer be the performance judge

Keep it as an adversarial correctness test.

But it is almost maximally hostile to warm subspace methods:

* random dense spectrum;
* exact half filling;
* almost guaranteed tiny Fermi gaps;
* perturbation independent of physical DFTB structure.

The report itself notices that the hardest case has

$$
\mathrm{gap}/\mathrm{span}\approx2.7\times10^{-4},
$$

and explicitly asks whether real DFTB spectra are much kinder. 

That needs answering empirically.

For production performance, I would benchmark actual sequences:

* successive SCC Hamiltonians for H\(_2\)O / GC / your real molecular systems;
* geometry displacements 0.01, 0.05, 0.1 Å;
* perhaps frozen-deflection sequences relevant to the force/Hessian calculations.

Measure

$$
\|P_k-P_{k-1}\|,
$$

$$
\|[H_k,P_{k-1}]\|,
$$

$$
\|[H_k,P_{\rm pred}]\|,
$$

Fermi gap/span, and number of corrections required.

That will tell us whether the actual warm problem is easy.

---

## What I would do next

I would stop all LNV/momentum parameter work and run four very targeted experiments:

1. **Repair the existing extrapolation experiment first.** Use \(S=P_1+\gamma(P_1-P_2)\); remove the Gershgorin rescaling entirely; do **one McWeeny retraction**, not TC2; then one commutator certificate. Start with \(\gamma=1\), then try the \(\Delta H\)- or SCC-DIIS-derived coefficient. Only if the certificate is insufficient, add 1–2 existing `dmm_update_batched` steps. This is the cheapest and most likely immediate win.

2. **Test it on a real SCC trajectory**, not random \(H+\delta R\). In particular, try applying the **same DIIS coefficients to stored \(P_i\)** that you already use for charge/H extrapolation. If the predicted projector needs only one correction, we are essentially done.

3. **Prototype the Sylvester/Newton response in CPU f64.** Solve

   $$
   [P,[X,H_0]]=-[P,[P,\Delta H]]
   $$

   with CG. Count CG iterations on real DFTB perturbations. If it is 2–5, GPU implementation is highly promising; if it is 20+, don't bother.

4. **Prototype occupied-subspace CheFSI.** Extract \(C=\mathrm{orth}(P\Omega)\) once, then test degree 2/4/6 polynomial filtering of \(H_{\rm new}C\) plus orthonormalization. This is the candidate I would take most seriously for a new production architecture because it directly preserves and updates the useful object—the occupied subspace—without asking a density-matrix gradient to rediscover it.

So the revised conceptual picture is:

$$
\boxed{
\text{cold: }H\rightarrow\text{Palser+TC2}
}
$$

but for warm updates,

$$
\boxed{
\text{predict the subspace motion}
\rightarrow
\text{cheap retraction/filter}
\rightarrow
\text{at most tiny correction}.
}
$$

The mistake was treating a previous converged density as merely an **initial point for another optimization**. It is much more valuable as a point on a **smooth trajectory of occupied subspaces**. That is the information we should exploit.

[1]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.92.193001 "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.92.193001"
[2]: https://doi.org/10.1063/1.3351785 "https://doi.org/10.1063/1.3351785"
[3]: https://discovery.ucl.ac.uk/id/eprint/10114096/ "https://discovery.ucl.ac.uk/id/eprint/10114096/"
[4]: https://arxiv.org/abs/cond-mat/0311591 "https://arxiv.org/abs/cond-mat/0311591"
[5]: https://www.sciencedirect.com/science/article/abs/pii/S002199910600146X "https://www.sciencedirect.com/science/article/abs/pii/S002199910600146X"
[6]: https://pubmed.ncbi.nlm.nih.gov/16095346/ "https://pubmed.ncbi.nlm.nih.gov/16095346/"


---

# USER

I think you ideas are very good but I noticed something which I see as big problem:

"""
If
then plain first-order descent requires � iterations. The report's dense random half-filled systems deliberately generate �, so hundreds or thousands of steps are entirely plausible.
""X

This is total nonsense! Why the hell we are testing it on something which doesn't have bandgap? Half-filled state? This seem to me like sabotage! The method is ment to work for System with decent gap and very good starting guess, that is the whole point! 


---

# Chat GPT 5.6 sol

Yes. I agree strongly: **the performance conclusion drawn from that benchmark is not valid for the problem you actually care about.** The random half-filled matrices are useful as an adversarial correctness/stability test, but they are a very poor warm-start benchmark.

The report itself effectively admits this: the difficult systems have `gap/span ≈ 2.7×10⁻⁴`, and it says dense random matrices at exact half filling practically guarantee near-degenerate frontier levels.  It later explicitly questions whether this has anything to do with real DFTB spectra. 

### The real error is worse than just “small gap”

The important small parameter for a warm start is **not**

$$
\Delta=0.02
$$

in the code.

It is approximately

$$
\boxed{
\eta_{\rm warm}
=
\frac{\|Q_0\,\Delta H\,P_0\|}
     {\Delta_{\rm gap}}
}
$$

where

$$
P_0=D_{\rm old},\qquad Q_0=I-P_0.
$$

This follows directly from first-order perturbation theory:

$$
\delta P_{ai}
\sim
\frac{\Delta H_{ai}}
     {\epsilon_i-\epsilon_a}.
$$

If the denominator is absurdly tiny, a numerically “small”

$$
H_1=H_0+0.02R
$$

can correspond to an **order-one rotation of the occupied subspace**.

Then it isn't a warm-start problem anymore!

For the reported pathological case,

$$
\frac{\mathrm{gap}}{\mathrm{span}}
\simeq2.7\times10^{-4}.
$$

A perturbation of even a percent of the spectral scale can therefore be **tens of times the gap**. The old projector may have very little predictive value for the new one.

So asking

> “why doesn't the old density converge in five iterations?”

under those conditions is unfair to the method.

---

## Half filling itself isn't the sin

One nuance: \(N_{\rm occ}=43\) for \(N=86\) does not inherently mean a metallic or gapless system.

The problem is:

> **random dense symmetric matrix + choose the middle 43 eigenvalues**

There is no mechanism creating a HOMO-LUMO gap. You are cutting through the dense middle of a random spectrum, exactly where neighboring levels are plentiful.

For our target systems we expect something qualitatively more like

```text
occupied band
xxxxxxxxxxxxxxxx

        GAP

                    xxxxxxxxxxxxxxxx
                    virtual band
```

not

```text
x x xx x x xxx x xx x xx x xxx x x
                   ^
                 Nocc
```

So yes: that synthetic test was almost designed to destroy the assumption on which warm starting is based.

---

# This changes my interpretation of the 30–300 iterations

I would **not accept those numbers as evidence against DMM warm starting**.

For tangent descent, each occupied–virtual rotation behaves approximately as

$$
\theta_{ia}^{(k+1)}
=
\left[
1-\alpha(\epsilon_a-\epsilon_i)
\right]\theta_{ia}^{(k)}.
$$

If the relevant gaps range from

$$
g_{\min}
$$

to

$$
g_{\max},
$$

the condition number is roughly

$$
\kappa_{\rm eff}
=
\frac{g_{\max}}{g_{\min}}.
$$

The random benchmark constructed

$$
\kappa_{\rm eff}\gg100
$$

or even thousands.

But imagine instead that the modes actually excited by a 0.1 Å geometry/SCC perturbation have gaps clustered around, say,

$$
2-6\ {\rm eV}.
$$

Then

$$
\kappa_{\rm eff}\sim3,
$$

not 3000.

A sensible step or momentum/quasi-Newton correction could then remove the error in **a handful of iterations**.

That is precisely the regime worth testing.

---

# More importantly: the initial error is also tiny

There are two independent advantages of a genuine warm start:

1. good conditioning because there is a real gap;
2. small starting error because \(\Delta H\) is small compared with that gap.

First-order theory gives roughly

$$
\|P_1-P_0\|
\sim
\frac{\|Q_0\Delta H P_0\|}
     {g}.
$$

If atoms move only \(0.1\) Å and SCC charges barely change, this might already be, say,

$$
10^{-2}
$$

or less.

Then we aren't asking the minimizer to rotate the subspace from scratch.

We are asking it to remove a **1% error**.

That is exactly where extrapolation, quasi-Newton DMM, or even simple tangent descent could be extremely competitive.

---

# We need to redesign the benchmark before doing anything else

I would use three benchmark classes.

### A. Actual production trajectories — the decisive benchmark

Save consecutive Hamiltonians/densities from:

* real SCC iterations;
* geometry steps of \(0.01,\ 0.05,\ 0.1\) Å;
* your frozen deflection / response calculations.

For every transition measure:

$$
g=\epsilon_{\rm LUMO}-\epsilon_{\rm HOMO},
$$

$$
g/\mathrm{span}(H),
$$

$$
\|P_{\rm new}-P_{\rm old}\|_F,
$$

$$
\|[H_{\rm new},P_{\rm old}]\|_F,
$$

and particularly

$$
\boxed{
r=
\frac{\|Q_{\rm old}\Delta H P_{\rm old}\|_F}{g}.
}
$$

That last number tells us whether this is actually a warm problem.

---

### B. Controlled synthetic **gapped** benchmark

Instead of `random_symmetric()`, explicitly generate

$$
H_0=U\,\mathrm{diag}(\epsilon)\,U^T
$$

with something like

```text
43 occupied states:   [-1.0 ... -0.3]
GAP
43 virtual states:    [+0.3 ... +1.0]
```

so

$$
g/\mathrm{span}\approx0.3.
$$

Then also test harder controlled cases:

$$
g/\mathrm{span}
=
0.2,\ 0.1,\ 0.05,\ 0.02.
$$

Now make \(H_1\) by a **controlled small subspace rotation**, rather than adding arbitrary dense noise.

For example,

$$
U_1=e^A U_0,
$$

where \(A\) has only occupied–virtual blocks and produces

$$
\|P_1-P_0\|_F
=
0.001,\ 0.01,\ 0.05,\ 0.1.
$$

Then

$$
H_1=U_1\,\mathrm{diag}(\epsilon+\delta\epsilon)\,U_1^T.
$$

We know the exact answer and exactly how warm the starting point is.

That is an excellent algorithmic benchmark.

---

### C. Keep the current random matrices

But label them correctly:

> **adversarial near-metallic stress test**

Use them for:

* NaN/divergence detection;
* correctness;
* convergence certification;
* near-degeneracy robustness.

Do **not** use them to decide whether a warm-start algorithm is worthwhile for gapped molecules.

---

# And this makes the extrapolation idea much more attractive

For a smooth sequence,

$$
P_{k+1}
=
P_k+\dot P\,h+O(h^2).
$$

Then

$$
2P_k-P_{k-1}
=
P_{k+1}+O(h^2).
$$

If the actual projector changes only by

$$
\|P_{k+1}-P_k\|\sim10^{-2},
$$

the extrapolation error can naturally be on the order of

$$
10^{-4}.
$$

Then one McWeeny retraction plus perhaps **one tangent correction** might genuinely be sufficient.

Testing that on a random matrix where the occupied projector jumps dramatically because the HOMO/LUMO are nearly degenerate tells us essentially nothing.

---

# Same for the quasi-Newton mean-gap idea

The report suggested

$$
\bar\Delta
=
\frac{\operatorname{Tr}(HQ)}{n_v}
-
\frac{\operatorname{Tr}(HP)}{n_o}.
$$

Then approximately

$$
\delta P
\approx
-\frac{1}{\bar\Delta}[P,[P,H]].
$$

On the random benchmark this is poor because relevant gaps range almost from zero to the entire bandwidth.

But on a molecular system where the coupled occupied–virtual transitions lie in a relatively narrow energy range, this can be a perfectly sensible preconditioner.

Again: **we have not actually tested the regime where the approximation is supposed to work.**

---

## So I would revise the current conclusion substantially

The report currently reaches, at one point, something close to:

> D-only warm starts are fundamentally slow because they have lost inverse-gap information. 

I would replace that with:

> **Unpreconditioned D-only gradient descent is slow on nearly gapless random matrices. Its performance for the intended regime—gapped systems with a genuinely nearby previous density—has not yet been established.**

That is a much narrower conclusion.

And I would not spend another hour tuning algorithms before collecting a table like:

| system/step     | gap/span | \(\|P_1-P_0\|\) | \(\|Q\Delta HP\|/g\) | cold TC2 squares | DMM steps | extrap+DMM steps |
| --------------- | -------: | --------------: | -------------------: | ---------------: | --------: | ---------------: |
| real SCC 1→2    |        … |               … |                    … |                … |         … |                … |
| real SCC 5→6    |        … |               … |                    … |                … |         … |                … |
| geometry 0.01 Å |        … |               … |                    … |                … |         … |                … |
| geometry 0.05 Å |        … |               … |                    … |                … |         … |                … |
| geometry 0.10 Å |        … |               … |                    … |                … |         … |                … |

I suspect that table will change the whole discussion. The current `Δ=0.02 random matrix` benchmark may well turn out to be **far more violent than a real 0.1 Å molecular displacement**, despite the innocent-looking `0.02` coefficient.
