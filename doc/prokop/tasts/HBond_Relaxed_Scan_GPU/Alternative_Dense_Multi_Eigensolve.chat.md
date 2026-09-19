
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

