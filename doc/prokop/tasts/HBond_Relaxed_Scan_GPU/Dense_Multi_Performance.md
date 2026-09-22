---
type: mandate
title: Dense multi-system GPU DFTB — performance is the goal
tags: [gpu, dense, purification, jacobi, performance, dftb]
timestamp: 2026-09-22
status: standing order for all further dense-solver work
---

# Dense multi-system GPU DFTB — performance is the goal

Read this before any other file in
[`HBond_Relaxed_Scan_GPU/`](.) or before touching
`rust_dftb/src/qmqm/gpu_purify.*`, `gpu_scc_plan.rs`, or the Jacobi kernels.

Later notes inside the long `.chat.md` files override earlier ones.
This file is the synthesis of those later notes, plus the standing
order that those notes kept losing.

## 1. The goal

The dense multi-system solver exists to be the **fastest GPU DFTB that
still does real chemistry**. Performance is the primary goal of the
work that starts from here. Accuracy is a constraint on that goal, and
the constraint is usefulness:

- The PES, barriers, forces, and proton-transfer direction must be
  chemically right and publishable.
- Qualitative failures (wrong minimum, inverted barrier, charge that
  oscillates or locks onto the wrong fragment) are failures.
- Matching a single-thread f64 eigensolver to `1e-7` Ha, idempotency to
  `1e-6`, or a commutator certificate on every SCC iteration is **not**
  the goal. Those targets already produced schedules that throw away
  the only reason to be on the GPU.

**Hard bar.** A GPU result that is not at least **100× a single CPU
thread** on the production workload (many replicas, real molecules,
full SCC) is not a GPU solver anyone will run. The f32 compromises are
only justified by that speed. Today the dense path is about **25×** on
guanine–cytosine (n = 86, batch = 400) and about **21×** on
DiTetraceno-helicene (n = 246). That is a factor of four to five short
of the bar. Further Jacobi tuning cannot close it.

The sparse nanocrystal path already cleared the same bar once the
solver stopped copying matrices to the host: 1648-atom Si, **~121×**
one CPU thread
([report §15.26](../Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md)).
The dense solver has not done the equivalent.

Device for every number below: NVIDIA RTX 3090, f32 peak ~36 TFLOPS.

## Main open problem — purification is slow, and warm does not help

This is the question to answer before any more kernel tuning.
Jacobi can already drop from 16 electronic iterations to 8 per
geometry step at a charge RMS of 1e-3, with the energy moving by at
most 0.07 meV. That is noted below. It is not this problem.

**Purification takes 24–40 SCC iterations. A warm start does not
reduce that number.** On GC both warm and cold print 32. On
diazaphen and DTH warm prints more than cold (32 vs 24, 40 vs 32).
A warm start whose only job is to need fewer iterations does not do
that job. The initial guess is not being used.

Why the iteration count is large. One SCC step does not purify the
density. A warm step applies a fixed two DMM+McWeeny rounds — a small
nudge — then mixes the charges and rebuilds H. The next step does the
same two rounds. The charges therefore walk to self-consistency over
a few dozen outer iterations. Cold spends 60 TC2 steps on the first
iteration only. Every iteration after that is the same two-round
nudge. The long tail is that nudge repeated, not the 60-step block.

Why warm is not faster. Carrying the previous K only changes the
starting point of that same nudge. Two rounds do not turn a good K
into the density of the new Hamiltonian, so the outer mixer still has
to crawl. The commutator gate makes this worse: if the carried K does
not yet commute with the new H, the replica is kept alive after the
charges would already have stopped. Cold's first 60-step purification
commutes sooner, so cold is allowed to finish first. After both have
been driven to a 1e-6 charge residual, the warm density is farther
from the CPU than the cold one (GC: 7e-3 e vs 3e-3 e). The carried K
is being polished toward the fixed point of a weak two-round map. It
is not being used as a guess that shortens the solve.

Loosening the charge tolerance does not fix this. At 1e-3, DTH purify
hits the iteration cap and every replica fails, because the
commutator gate turns the replica back on after the mixer has frozen
it. GC and diazaphen move by 0.25–0.59 meV. The stop is not the bug.
The update is.

What would actually be a warm start. Not "purify the old density
until it is good." McWeeny and TC2 change occupations and leave the
eigenvectors where they are, so a purification of a rotated kernel
repairs the electron count and leaves the forces wrong. That is
measured, not a conjecture: sparse B1 (extrapolate, one McWeeny, no
commutator) brings the trace error from ~0.04 down to ~10⁻³ and the
forces stay 12–26% off a cold SCC
(`../Sparse_Nanocrystal_Vibrations/Warm_Geometry_DM.md` §8).

The rotation is the commutator step. Sparse B2/B3, the recipe that
was kept, is one geometry step and then stop:

```text
K ← previous K          (2K₁−K₀ only if it lowers R_H)
two commutator steps, η = 8
one McWeeny
damped charge update (α = 0.2), no cold TC2
```

On SiH₄, 0.1 Å moves, forces match a cold SCC to about 1%, energy to
about 1 mHa (0.1 mHa once a second history point exists). A 12-step
FIRE stayed downhill and finished 5×10⁻⁵ Ha from a cold SCC of that
geometry, at ~1.6 ms per step against 7 ms cold. The truncated-mask
nanocrystal has not been run with this recipe.

Dense warm already contains both operations, and uses them as the
thing that failed: η = 1, two rounds, then the charge mixer, repeated
until the residual is 1e-6. The geometry step is now
`GpuDftb::geom_bold_dmm`: η starts at 8, a step that raises R_H is
restored and η is halved, at most two accepted steps, then one
McWeeny. Blind η = 8 on formic raised R_H from 2.3e-2 to 6.0e-2.
The accepted step was η = 4, twice. Batch 1, against a CPU SCC of
the moved geometry. Energy and forces below are a Jacobi finalize
at the step's own Mulliken charges.

| system | move | R_H | max\|Δq\| | ΔE | max\|ΔF\| / max\|F\| | time |
|---|---|---|---|---|---|---|
| formic | 0.02 Å along F | 2.3e-2 → 3.6e-3 | 9e-3 e | −0.6 meV | 11% | 0.4 ms |
| formic | 0.10 Å, one atom | 8.8e-3 → 1.4e-3 | 6e-3 e | −0.4 meV | 4% | 1.5 ms |
| GC | 0.02 Å along F | 3.2e-2 → 4.0e-3 | 2.3e-2 e | −1.8 meV | 2.4% | 0.9 ms |
| GC | 0.10 Å, one atom | 2.1e-2 → 4.6e-3 | 1.3e-2 e | −2.1 meV | 2.2% | 0.9 ms |

Trace stays on Nocc (τ < 10⁻³). This is one geometry step, not
an SCC loop. The forces are a few percent, not the SiH₄ 1%.

Same Hamiltonian, batch 256, cap on accepted steps raised from 1 to 8
(2026-09-22, `test_geom_bold_nacc`). Each row is that many accepted
commutator steps at η = 4, then one McWeeny. `dq_H` is against one
Jacobi diagonalization of this same H. `dq_scc`, ΔE and the force are
against a Jacobi SCC at the new geometry. R_H falls every step. The
density walks toward the projector of the stale Hamiltonian, and that
projector is not the SCC density, so the forces move away from Jacobi
on the three smaller molecules. DTH's stale H is already close, so
eight steps stay near 0.5% in the force. The cap stays at two. The write-up of why this is not a convergent
solve, and what could buy the accuracy back, is
[`Bold_Step_Accuracy.md`](Bold_Step_Accuracy.md).

| system | N | R_H | dq_H | dq_scc | ΔE | force |
|---|---:|---:|---:|---:|---:|---:|
| formic | 1 | 7.6e-3 | 2.2e-2 | 8.3e-3 | −0.4 meV | 8.5% |
| formic | 2 | 3.6e-3 | 2.0e-2 | 9.2e-3 | −0.6 meV | 11% |
| formic | 8 | 8.2e-4 | 7.1e-3 | 1.7e-2 | −5.7 meV | 48% |
| GC | 1 | 8.4e-3 | 5.4e-2 | 3.0e-2 | −3.4 meV | 4.5% |
| GC | 2 | 4.0e-3 | 5.4e-2 | 2.3e-2 | −1.8 meV | 2.4% |
| GC | 8 | 8.1e-4 | 1.4e-2 | 2.8e-2 | −23 meV | 13% |
| diazaphen | 2 | 4.4e-3 | 5.2e-2 | 1.0e-2 | −2.9 meV | 4.9% |
| diazaphen | 8 | 1.1e-3 | 1.7e-2 | 3.0e-2 | −28 meV | 25% |
| DTH | 2 | 5.6e-3 | 5.3e-3 | 4.0e-3 | −1.4 meV | 0.49% |
| DTH | 8 | 1.7e-3 | 3.0e-3 | 4.9e-3 | −1.9 meV | 0.46% |

A second pass rebuilds H and rotates again. Mixing only 20% of the
new charges (α = 0.2) and repeating makes the 0.02 Å forces worse
(formic 11% → 26%, GC 2.4% → 7% at three passes). Taking the
Mulliken charges in full (α = 1) and doing one more pass:

| system | move | 1 pass ΔE, force | 2 passes, α = 1 |
|---|---|---|---|
| formic | 0.02 Å | −0.6 meV, 11% | −0.07 meV, 4.3% |
| formic | 0.10 Å | −0.4 meV, 3.6% | −0.05 meV, 0.9% |
| GC | 0.02 Å | −1.8 meV, 2.4% | −1.6 meV, 3.0% |
| GC | 0.10 Å | −2.1 meV, 2.2% | −0.8 meV, 1.5% |

One extra pass at the new charges is worth it on formic and on the
larger move. It does not bring GC's 0.02 Å force under 1%. More
damped passes are not that lever.

Throughput of both, same 0.02 Å protocol as the Jacobi curves
(`throughput_{formic,GC,diazaphen,DTH}.png`, purple). Saturated
systems/s, bold versus the Jacobi-warm row already on those figures
(that row is the 1e-6 SCC, 16 iterations on GC and larger):

| system | bold 1× | bold 2× α=1 | Jacobi warm |
|---|---:|---:|---:|
| formic, batch 1024 | 9.0×10⁵ | 3.3×10⁵ | 8.2×10⁴ |
| GC, batch 256 | 8.2×10⁴ | 4.1×10⁴ | 8.7×10³ (peak 9.6×10³ at 400) |
| diazaphen, batch 1024 | 2.9×10⁴ | 1.5×10⁴ | ~2.1×10³ |
| DTH, batch 256 | 3.6×10³ | 1.8×10³ | ~3.4×10² |

One bold step is about 8–14× the Jacobi SCC on the same batch. The
second pass costs about half the throughput and is still several
times Jacobi. Charges versus the CPU stay at the batch-1 values at
every batch (the copies are the same molecule).

**Relaxation, as measured.** Jacobi at charge RMS 1e-3 keeps the
energy of a 0.02 Å force step inside 0.1 meV (worst case DTH, 0.07
meV) and halves the electronic iterations on GC, diazaphen, and DTH
(16 → 8; formic was already 8). Forces were not differenced; the
energy is the energy of the state the forces are taken from, and the
charges stay within 5e-4 e of the CPU. Purification cannot take the
same cut: the energy moves by tenths of a meV, or the solve fails.
The number of geometry steps in a relaxation is unchanged either way.
The throughput figures are still the 1e-6 grid.
Dates are the measurement dates in the source notes.

## 2. Why Jacobi cannot get there

Cyclic Jacobi at n ≈ 86 is one workgroup per molecule. The matrix does
not fit in local memory together with the eigenvectors (A+V ≈ 60 KB,
device local cap 48 KB), so a resident-A kernel holds one workgroup per
compute unit. That caps throughput at roughly the SM count.

What was measured, and is closed:

| fact | number | where |
|---|---|---|
| Streaming Jacobi, n=86, batch=400 | ~85 % of DRAM, WG512 fastest | `Measured_Facts_Jacobi_Sweeps.md` §1 |
| Resident A (the bandwidth fix) | cold ~17–20 ms, warm-ish ~10–11 ms; ~2.2× over streaming | same, §1b |
| Packed triangle → 2 WG/SM | **+5–9 %** | `Dense_Multi_GPU_Optimization.tasks.md` T08 |
| Solve-end V replay, ~500× less V traffic | **+6–9 %** | same; `Measured_Facts` §8 |
| Arithmetic intensity | ~1.8 TFLOPS ≈ **5 % of peak**; another ~2× of Jacobi tuning → ~10 % | `Alternative_Dense_Multi_Eigensolve.md` §0 |
| Production warm Jacobi inner solve, GC batch=400 | **1.53–1.76 ms**, ~70 % of the SCC iteration | notes Part VII–VIII; design doc §7.3.3 |
| End-to-end vs one f64 CPU thread | GC **~25×** (1916 sys/s); DTH block Jacobi **~21×** | design doc §7.3.3; manifest §16.F |

The kernel is **round-latency bound**: ~85 serialized rounds per sweep,
most lanes idle, a barrier between phases. No tile size, no extra
occupancy, and no more residency gets this algorithm to 100×. Stop
tuning Jacobi. Keep it as the reference eigensolver and as the
drop-in when a caller actually needs eigenvalues.

## 3. Why purification, and what went wrong

The replacement is a density-matrix inner solve built from the tiled
symmetric square (`square_regtile` / `gemm_sq_iter`). That primitive
runs at **9–11 TFLOPS (~25–30 % of peak), 0.047–0.054 ms per square**
at n=86, batch=400. Cold trace-correcting purification (Palser start +
TC2) then beats **cold** Jacobi: **8.5–10.7 ms vs ~17 ms**. That result
was reproduced; it is real.

It is also the wrong comparison. Production Jacobi is warm: it reuses
the previous eigenvectors and finishes in 1–3 sweeps, **1.5 ms**. A
cold purification every SCC iteration is several times slower than
that, by arithmetic, before any bug.

The original reason to leave Jacobi was the warm start: the previous
density should make the next geometry, or the next SCC iteration,
cheap. That attempt failed for specific, now-understood reasons. The
failure is not "purification cannot be fast."

What was tried, in order, and what the **later** records concluded:

1. **Learned constraint force Λ.** Structurally broken. Any clean
   rank-N projector is an exact fixed point once Λ absorbs the gauge;
   the tangent part of Λ is invisible to the learning update. Cold from
   Λ=0 works. A warm Λ freezes on the wrong projector or drifts toward
   eig(ΔH). Do not revive Λ. (Notes §6–7; chat from the "fatal problem
   with learned Λ" turn.)

2. **Steepest-descent / LNV on a stale D.** Polynomials in D cannot
   rotate the occupied subspace; only the force can, and it crawls at
   a rate set by the occ–virt gaps. On **random half-filled** matrices
   the gap/span is ~2.7×10⁻⁴, so a "small" ΔH is an order-one rotation.
   Those runs (30–300 iterations) measure an adversarial metal. They
   are a stress test for NaNs and certificates. They are not a
   performance result. The user rejected them as a benchmark, and the
   review agreed (chat after "this is total nonsense", and notes
   Part III).

3. **Extrapolation, once the two bugs were removed.** Gershgorin
   rescaling maps a good {0,1} projector onto ~{0.44, 0.56} and
   destroys the seed. TC2 folds amplify the radial error of an
   extrapolated projector; **McWeeny** (`3x²−2x³`) kills it and keeps
   the predicted rotation. After both fixes, on real molecules:

   - Geometry steps 0.01 / 0.05 / 0.10 Å, charges carried over,
     AO-space `2D₁−D₀`, one McWeeny, certificate:
     **3 products, zero descent steps, ΔE ~ 1e-6 to 1e-7**, H2O and
     formic acid, all three step sizes. Cold TC2 on the same targets
     is 16–24 products. (Notes Part VI.2.)
   - Same-AO-coefficient transport is the right geometry guess.
     Cross-overlap orbital transport (`S_cross`) was measured ~10×
     worse: the basis follows the atoms; pinning orbitals to absolute
     space does not.
   - Mid-SCC with a large charge swing (rotation amplitude η_w ~ 0.1–1)
     correctly fails the certificate. The late SCC tail is only a
     ~25 % product saving. A fixed γ=1 predictor overshoots a
     decelerating SCC sequence; γ=0 (reuse the last D) is better there.

4. **The production `EIGSOLVER=purify` path never shipped (3).** It
   shipped a defensive schedule: orthogonalize, **16 DMM steps, 8
   McWeeny, a TC2 tail, a certificate computed after the work is
   already done**, then an unconditional fused-TC2 fallback, each
   product its own `batched_gemm` launch (~0.35 ms) instead of the
   resident square (~0.05 ms). On the 400-point scan that is **3–30×
   slower per iteration than warm Jacobi**, more SCC iterations, and
   hundreds of unconverged GC replicas (notes Part VII).

5. **Those GC failures are not a purification failure.** The
   certificate readback is `max_rh ≈ 1e-6` with **zero uncertified
   replicas**. TC2 returns a commuting integer projector. At
   kT = 0.002 the Jacobi path returns a Fermi-smeared density. On the
   proton-transfer geometries where the gap is ≲ kT, the integer
   density is the wrong physics, and **Jacobi at kT = 0 fails the same
   population** (232 vs 207). At equal physics (kT = 0) purification
   fails fewer replicas on every system tested, and at n = 120 it is
   already faster per iteration **with the dumb schedule still in**
   (5.4–5.7 ms vs 7.6–9.1 ms). (Notes Part VIII.)

The pattern to refuse next time: a safe iteration cap, a tight
certificate, a host branch, and a full re-solve, chosen so a test
cannot go red. That schedule is why the GPU path lost to Jacobi.

## 4. The accuracy contract

Publishable for this project means what the proton-transfer scan
already showed against CPU f64: PES **shape** error ~0.1 meV, forces
at a few 10⁻⁶ Ha/Å
([roadmap](../../DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md)
GC scan note; `topical_audit/f32_floor_dense_hbond.md`). Absolute
energy offsets of ~10⁻⁵ Ha that stay smooth along a geometry are
acceptable. A discontinuous occupation that flips the charge between
SCC iterations is not.

Allowed compromises, in order of preference:

- f32 for every O(n³) product. f64 only for O(n) decisions (trace,
  chemical potential, DIIS Gram) when a measurement says the f32
  decision is the thing that diverges.
- Integer occupation on gapped molecules. Finite-T is required only
  where the gap is comparable to kT **and** the workload uses smearing.
- Early-stopped purification as a smooth stand-in for the Fermi
  function (transition width halves each TC2 step; ~9 steps at
  kT = 0.002 on a ~1 Ha spectrum, versus ~50 steps to a hard
  projector). The fixed point will differ slightly from Fermi–Dirac.
  Accept it if the PES shape stays in the contract above. A 100-term
  Chebyshev expansion of the exact Fermi function costs ~11 ms at n=86
  and **loses to Jacobi** — rejected on cost (notes VIII.5, VIII.8).
- A commutator or energy check every few iterations, or only at SCC
  acceptance. Not every square.
- An explicit fallback (short TC2, or Jacobi) when a certificate says
  the predictor missed. `converged = false` stays a real result. A
  silent clamp, a loosened tolerance inside the test, or a 60-step
  loop "just in case" does not.

Forbidden compromises: dropping the 100× bar, hiding a host sync inside
the iteration, allocating or building kernels in the SCC loop, and
declaring success on a random half-filled matrix.

## 5. What to take from the sparse solver

Sparse work lives in
[`Sparse_Nanocrystal_Vibrations/`](../Sparse_Nanocrystal_Vibrations/).
It is a different algorithm (masked BSR SpGEMM, nanocrystal Hessians).
These pieces transfer. The rest does not.

**Take.**

- **Residency is the 100×.** R10 went ~27× and R18 ~121× only after
  ~155 MB/eval of PCIe disappeared (report §15.25–15.26). Kernel
  tweaks before that were noise. The dense analogue is: K, H, and the
  square stay in device buffers; the host sees charges and a scalar
  residual at chunk boundaries.
- **A short fixed recipe can be the fast path.** Sparse `n_dmm = 4`
  (DMM-lite) is how the Hessian columns became affordable. The dense
  analogue is the measured 3-product geometry update, not "iterate
  until rh < 1e-6".
- **The branch belongs on the device.** Sparse TC2 lost a sync per
  iteration to a 4-byte trace read. Dense TC2 had the same bug
  (`purify_trace_correcting` read the trace back to pick D² vs 2D−D²).
  The fused `tc2_step_batched` is the correction; the production
  purify schedule walked away from it.
- **Stop when the observable stops moving.** Sparse learned that a
  plateau, a truncation floor, and a real divergence are different
  events, and that running the polynomial past the floor poisons K.
  Dense should stop a replica when the density is good enough for the
  current SCC residual, and should not purify below the f32 floor.
- **Batch many replicas on one launch.** That is the whole point of
  the dense multi solver. Sparse SpGEMM is bandwidth-bound (~2 FLOP/B,
  batching gave 1.2× at 1648 atoms, report §15.28). Dense tiled GEMM
  is the opposite regime, so a batch of hundreds of n≈86–256 matrices
  is where this GPU pays off. Do not design the inner solve around one
  molecule.

**Leave there.**

- Masks, radial cutoffs, dummy-lane padding, Newton–Schulz for S⁻¹.
  Small dense molecules are not sparse, and Löwdin X is a few GEMMs.
- The sparse ordering "correctness, then rigor, then speed"
  (manifest §1.4). It was right for a solver that was still producing
  wrong forces. Applied here it recreates the 57-launch schedule.
- Any conclusion that warm density-matrix updates are limited to tiny
  force perturbations. That was true for the adversarial benchmark and
  false for 0.1 Å geometry steps once the predictor and the retraction
  were the right ones (Part VI.2).

## 6. Path forward

Order is the performance order. Each step is done when a wall-clock
ratio versus one CPU thread is written down on `scan400`, not when a
unit test matches Jacobi to 1e-6.

1. **Make the benchmark the contract.** `tests/gpu_scc_bench.rs`
   (`test_gpu_scc_scan400_benchmark`) and `tests/gpu_warm_bench.rs`,
   batch = 400, release build, one CPU thread as the denominator.
   Report sys/s and the ratio next to the chemistry check (one
   proton-transfer scan, shape error in meV). Two rows: kT = 0 and
   kT = 0.002. Random matrices stay in `tests/gpu_purify.rs` as a
   crash/NaN test only.

**Measured 2026-09-22, co-iteration only** (fallback off, 2 DMM rounds,
`scan400`, release, kT = 0, same binary). The 60-step per-iteration
TC2 is gone. Wall time for batch = 400:

| system | Jacobi | purify | failed (J / P) |
|---|---|---|---|
| formic n=28 | 36 ms, 100 iters, 0.36 ms/iter, 29× | **18 ms, 80 iters, 0.23 ms/iter, 50×** | 4 / 0 |
| GC n=86 | 492 ms, 100 iters, 4.92 ms/iter, 13× | **239 ms, 100 iters, 2.40 ms/iter, 21×** | 232 / 1 |

About 2× the Jacobi wall, and the GC failure population collapses.
Still a factor of four to five short of 100× on GC. The remaining
~2.4 ms/iter is the general `batched_gemm_active` products in the
short update (~0.21 ms each at n=86, batch=400), not a purification
loop. Next lever is those products on the resident square, then the
3-product geometry predictor so a new geometry does not pay a cold TC2.

**Measured 2026-09-22, saturation + accuracy** (same binary, kT=0,
1 run, scan window held at d∈[1.0, 1.95] Å). Throughput vs one CPU
thread on the equilibrium geometry. Mulliken `max|Δq|` is the
equilibrium SCC density read before `finalize` (which is still Jacobi).

| system | n | batch | ms/iter | sys/s | vs 1 CPU | max\|Δq\| | failed |
|---|---|---|---|---|---|---|---|
| formic | 28 | 400 | 0.30 | 24040 | 43× | 5.5e-3 e | 0 |
| formic | 28 | 900 | 0.49 | 32624 | 58× | | 0 |
| formic | 28 | 1600 | 0.90 | 31596 | 56× | | 0 |
| GC | 86 | 400 | 3.37 | 1855 | 19× | 4.4e-3 e | 0 |
| GC | 86 | 900 | 7.28 | 1933 | 20× | | 0 |
| GC | 86 | 1600 | 12.97 | 1927 | 20× | | 0 |
| diazaphen | 120 | 400 | 7.78 | 1072 | 23× | 4.8e-3 e | 0 |
| diazaphen | 120 | 900 | 11.80 | 1060 | 22× | | 0 |
| diazaphen | 120 | 1600 | 21.73 | 1023 | 22× | | 0 |
| DTH (row·row) | 246 | 400 | 194 | 43 | 5.7× | 1.9e-3 e | 0 |

n≥86 is already full at 400 replicas: 1600 takes 4× the time and the
same sys/s. formic still climbs until ~900. DTH cannot launch the
register-tiled square (62×31 = 1922 threads > 1024), so that row is
`RUST_DFTB_PURIFY_GEMM=0`.

GC device profile (`RUST_DFTB_PROF=evt`), batch 400, 64 iters, 220 ms
on device: `scc.purify` 65% (2.24 ms/iter), `scc.pur_back` 15%
(0.53 ms), `scc.pur_ortho` 12% (0.41 ms), mulliken 3%, DIIS 2%, Hbuild
2%. Same fractions at batch 1600, times ×4. Hot GEMM is
`batched_gemm_active` 16×16, K=32, 256 threads, **4168 B local of
48 KB**. OpenCL `CL_KERNEL_PRIVATE_MEM_SIZE` returns 0 for every
kernel on this driver, including Jacobi, so register spill is not
visible that way. The tuned square (`tc2_step`, RTX=4×RTY=8) is only
the cold start.

**Measured 2026-09-22, warm products on regtile** (same grid, kT=0,
batch=400). The four DMM products (H′K, KT, K², K³) use
`gemm_regtile_masked`, 11×11 threads × 8×8 regs, tk=16 — the §7.3.1
general-GEMM winner. n<64 stays on 16×16 (formic got slower on the
88-wide tile). `RUST_DFTB_PURIFY_MM=tiled` restores the old kernel.

| system | kernel | wall | iters | ms/iter | vs 1 CPU | failed | max\|Δq\| |
|---|---|---|---|---|---|---|---|
| GC n=86 | 16×16 | 225 ms | 100 | 2.25 | 20× | 1 | 4.42e-3 e |
| GC n=86 | regtile | **158 ms** | **64** | 2.47 | **28×** | **0** | 4.43e-3 e |
| diazaphen n=120 | 16×16 | 388 ms | 64 | 6.06 | 22× | 0 | 4.83e-3 e |
| diazaphen n=120 | regtile | **337 ms** | **56** | 6.03 | **27×** | 0 | 3.76e-3 e |

Device profile, GC, per call: `scc.purify` 1.52 ms → 1.38 ms. The
iteration is not 4× cheaper. Ten launches still dominate, and
orthogonalization plus the back-transform are still the 16×16 kernel.
The wall-clock gain is that the batch finishes in 64 iterations with
no stalled replica, where 16×16 ran to the cap.

**Measured 2026-09-22, DTH workgroup.** The 5.7× row was
`PURIFY_GEMM=0`. A 4×8 full-cover square is 62×31 = 1922 threads; an
8×8 full-cover square is 961 threads and the driver returns
`CL_OUT_OF_RESOURCES`. The cold square now walks 88-wide tiles on an
11×11 × 8×8 workgroup (121 threads), the same shape as the warm
GEMM. Batch 400, kT = 0:

| DTH n=246 | wall | iters | ms/iter | sys/s | vs 1 CPU | failed | max\|Δq\| |
|---|---|---|---|---|---|---|---|
| row·row | 9315 ms | 48 | 194 | 43 | 5.7× | 0 | 1.9e-3 e |
| tiled square | **2088 ms** | 40 | 52 | **192** | **27×** | 0 | 2.0e-3 e |

GC on the same run: 165 ms, 2430 sys/s, 30× one thread (12.4 ms).
Per system the GPU cost rose 13× from n=86 to n=246; the one CPU
thread rose 11× (12 ms → 141 ms). The ratio stays ~30×.

### CPU reference — 8 OS threads, 1 BLAS thread each

Measured 2026-09-22 on the Ryzen 7 5800X (8 cores). This is the
CPU number every later speedup is quoted against. It is a bench
result, not the solver default: `cargo` still links whatever
`LD_LIBRARY_PATH` puts first, and that is Spack OpenBLAS 0.3.33
(`USE_OPENMP USE_LOCKING`), which admits one BLAS call per process.
The reference uses the system pthread library:

```text
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so
```

`openblas_set_num_threads(1)` is called once on the main thread
before the workers start. Eight `DftbCpu` values, eight equilibrium
copies, kT = 0, `dsyevd`, no custom eigensolver.

| system | 1 thread | 1 solve × 8 BLAS threads | **8 threads × 1 BLAS** |
|---|---|---|---|
| GC n=86 | 11.7 ms, 86 sys/s | 12.1 ms (1.0×) | **10.6 ms, 95 sys/s (1.1×)** |
| DTH n=246 | 142 ms, 7.1 sys/s | 157 ms (0.9×) | **34 ms, 29 sys/s (4.1×)** |

Spreading one `dsyevd` across 8 BLAS threads does not help. Eight
independent solves do: 4.1× on DTH, about 1× on GC (the GC
diagonalization does not fill a core). The earlier 7.1 DTH sys/s
figure is one core. Against this 8-core rate the saturated purify
batch (400, cold scan) is ~7× on DTH (192 / 29) and ~25× on GC.

### Geometry step — same guess, four solvers

Measured 2026-09-22, same machine, same `LD_PRELOAD`, kT = 0,
`test_gpu_warm_step_benchmark`. Not a random matrix.

Procedure, every system: converge the xyz geometry on the CPU,
take one Cartesian step along the forces with rms 0.02 Å, then
time only the next SCC. All replicas share that one perturbed
geometry. CPU warm starts from the converged G0 charges
(`set_charges`, no `reset_charges`). CPU cold starts from atomic
q0. GPU Jacobi warm carries the eigenvectors and the charges
across `set_coords`. GPU purify warm carries K
(`set_coords_keep_k`); the default `set_coords` still drops K.
GPU cold is a fresh Jacobi solve, or purify after `reset_q0`.
**cpu 8×1 is eight OS threads, one BLAS thread each.** It is not
one thread. The 10.6 ms / 95 sys/s (GC) and 34 ms / 29 sys/s (DTH)
reference is this same layout. `batch` is the number of copies of
that one perturbed molecule solved together. sys/s = batch / wall.
A batch of 8 is one molecule per CPU core; it does not fill the GPU.
`max|Δq|` is replica 0
against the CPU warm charges at the new geometry. Every row
converged (`failed = 0`). The SCC timer does not include the
geometry upload (0.3–16 ms).

| system | n | batch | method | wall | iters | sys/s | vs CPU warm | max\|Δq\| |
|---|---|---|---|---|---|---|---|---|
| formic | 28 | 8 | cpu 8 threads, cold | 83 ms | 13 | 96 | | |
| formic | 28 | 8 | cpu 8 threads, warm | 69 ms | 12 | 116 | 1× | |
| formic | 28 | 8 | gpu jacobi cold | 5.0 ms | 16 | 1590 | 14× | 1.7e-6 |
| formic | 28 | 8 | gpu jacobi warm | 2.6 ms | 8 | 3090 | 27× | 8.5e-7 |
| formic | 28 | 8 | gpu purify cold | 5.4 ms | 32 | 1490 | 13× | 5.0e-3 |
| formic | 28 | 8 | gpu purify warm | 3.8 ms | 24 | 2110 | 18× | 1.9e-3 |
| GC | 86 | 8 | cpu 8 threads, cold | 73 ms | 19 | 110 | | |
| GC | 86 | 8 | cpu 8 threads, warm | 75 ms | 12 | 107 | 1× | |
| GC | 86 | 8 | gpu jacobi cold | 16 ms | 16 | 510 | 4.7× | 5.6e-6 |
| GC | 86 | 8 | gpu jacobi warm | 9.5 ms | 16 | 840 | 7.8× | 2.2e-6 |
| GC | 86 | 8 | gpu purify cold | 20 ms | 32 | 390 | 3.7× | 3.1e-3 |
| GC | 86 | 8 | gpu purify warm | 15 ms | 32 | 540 | 5.0× | 7.3e-3 |
| diazaphen | 120 | 8 | cpu 8 threads, cold | 95 ms | 15 | 85 | | |
| diazaphen | 120 | 8 | cpu 8 threads, warm | 100 ms | 13 | 80 | 1× | |
| diazaphen | 120 | 8 | gpu jacobi cold | 35 ms | 16 | 230 | 2.8× | 5.2e-6 |
| diazaphen | 120 | 8 | gpu jacobi warm | 20 ms | 16 | 400 | 5.0× | 2.1e-6 |
| diazaphen | 120 | 8 | gpu purify cold | 49 ms | 24 | 160 | 2.1× | 3.4e-3 |
| diazaphen | 120 | 8 | gpu purify warm | 42 ms | 32 | 190 | 2.4× | 4.5e-3 |
| DTH | 246 | 8 | cpu 8 threads, cold | 299 ms | 16 | 27 | | |
| DTH | 246 | 8 | cpu 8 threads, warm | 241 ms | 14 | 33 | 1× | |
| DTH | 246 | 8 | gpu jacobi cold | 219 ms | 16 | 37 | 1.1× | 3.9e-6 |
| DTH | 246 | 8 | gpu jacobi warm | 125 ms | 16 | 64 | 1.9× | 1.7e-6 |
| DTH | 246 | 8 | gpu purify cold | 187 ms | 32 | 43 | 1.3× | 2.1e-3 |
| DTH | 246 | 8 | gpu purify warm | 324 ms | 40 | 25 | 0.74× | 5.6e-4 |
| formic | 28 | 64 | cpu 8 threads, cold | 84 ms | 13 | 760 | | |
| formic | 28 | 64 | cpu 8 threads, warm | 58 ms | 12 | 1110 | 1× | |
| formic | 28 | 64 | gpu jacobi cold | 4.4 ms | 16 | 14400 | 13× | 1.7e-6 |
| formic | 28 | 64 | gpu jacobi warm | 2.2 ms | 8 | 29600 | 26× | 8.5e-7 |
| formic | 28 | 64 | gpu purify cold | 7.1 ms | 32 | 9000 | 8× | 5.0e-3 |
| formic | 28 | 64 | gpu purify warm | 3.2 ms | 24 | 19800 | 18× | 1.9e-3 |
| GC | 86 | 64 | cpu 8 threads, cold | 245 ms | 19 | 260 | | |
| GC | 86 | 64 | cpu 8 threads, warm | 158 ms | 12 | 410 | 1× | |
| GC | 86 | 64 | gpu jacobi cold | 19 ms | 16 | 3330 | 8.2× | 5.6e-6 |
| GC | 86 | 64 | gpu jacobi warm | 13 ms | 16 | 4960 | 12× | 2.2e-6 |
| GC | 86 | 64 | gpu purify cold | 27 ms | 32 | 2340 | 5.8× | 3.1e-3 |
| GC | 86 | 64 | gpu purify warm | 24 ms | 32 | 2650 | 6.5× | 7.3e-3 |
| diazaphen | 120 | 64 | cpu 8 threads, cold | 324 ms | 15 | 200 | | |
| diazaphen | 120 | 64 | cpu 8 threads, warm | 281 ms | 13 | 230 | 1× | |
| diazaphen | 120 | 64 | gpu jacobi cold | 41 ms | 16 | 1580 | 6.9× | 5.2e-6 |
| diazaphen | 120 | 64 | gpu jacobi warm | 25 ms | 16 | 2600 | 11× | 2.1e-6 |
| diazaphen | 120 | 64 | gpu purify cold | 77 ms | 24 | 830 | 3.6× | 3.4e-3 |
| diazaphen | 120 | 64 | gpu purify warm | 70 ms | 32 | 910 | 4.0× | 4.5e-3 |
| DTH | 246 | 64 | cpu 8 threads, cold | 1890 ms | 16 | 34 | | |
| DTH | 246 | 64 | cpu 8 threads, warm | 1760 ms | 14 | 36 | 1× | |
| DTH | 246 | 64 | gpu jacobi cold | 340 ms | 16 | 190 | 5.2× | 3.9e-6 |
| DTH | 246 | 64 | gpu jacobi warm | 205 ms | 16 | 310 | 8.6× | 1.7e-6 |
| DTH | 246 | 64 | gpu purify cold | 386 ms | 32 | 170 | 4.6× | 2.1e-3 |
| DTH | 246 | 64 | gpu purify warm | 625 ms | 40 | 100 | 2.8× | 5.6e-4 |

On this step Jacobi is the faster GPU solver at every size.
Carrying K helps formic and GC and costs iterations on DTH
(40 vs 32), so purify-warm is slower than purify-cold there.
Jacobi charges match the CPU to ~1e-6 e. Purify stays at a few
millielectrons. Batch 8 does not fill the GPU: DTH purify-warm
is slower than the 8 CPU threads (0.74×). At batch 64 the same
row is 2.8×, and Jacobi-warm is 8.6×. The 8-thread CPU rate on
this step matches the equilibrium reference above (GC ~9–11 ms
per system, DTH ~30–37 ms per system).

### Throughput vs batch

Same geometry step as above. The quantity is systems per second,
`batch / wall`. **batch** is how many copies of that one perturbed
molecule are solved together. It is not a random-matrix size.

Two CPU lines, both warm-started from the G0 charges:

- **cpu 1 thread** — one `dsyevd`.
- **cpu 8 threads** — eight OS threads, one BLAS thread each.
  This is the reference layout. At batch 8 it is one system per
  thread. On GC that does not beat one thread (96 vs 92 sys/s):
  the 95 sys/s quoted above *is* this point, and it is not an
  8× machine. On DTH the same point is 30 sys/s, 4× the
  1-thread rate of 7.6, which is the 29 sys/s quoted above.
  The plateau is higher once each core has several systems:
  DTH holds **52–55 sys/s** from batch 32 through 512, then
  44 sys/s at 1024.

GPU curves use one engine allocated at the largest batch. A
smaller batch is the first B replicas, launch domain shrunk to
B, density guess restored, so every point is the same step.
Figures: `debug/dense_multi/throughput_{formic,GC,diazaphen,DTH}.png`
(log batch, log throughput). CSV next to them.

| system | cpu 1 thread | cpu 8 threads, plateau | jacobi warm | purify warm | purify cold |
|---|---|---|---|---|---|
| formic n=28 | 1.1×10³ | 6.2×10³ at 1024, still up | **8.2×10⁴ from 512** | 8.0×10⁴ at 1024, still up | 3.7×10⁴ at 1024, still up |
| GC n=86 | 1.1×10² | ~7×10² from batch 128 | **9.6×10³ at 400**, 8.5×10³ at 1024 | **4.4×10³ at 256–400**, 4.0×10³ at 1024 | 3.9×10³ at 400, 3.4×10³ at 1024 |
| diazaphen n=120 | 49 | ~3.6×10² from batch 256 | **2.8×10³ at 64**, 2.1×10³ at 1024 | **1.8×10³ from 256** | **1.6×10³ at 1024** |
| DTH n=246 | 7.6 | **55 from batch 32**, 44 at 1024 | **3.4×10² at 256–512**, 3.1×10² at 1024 | **1.6×10² at 256**, 1.3×10² at 1024 | **2.4×10² at 256**, 1.9×10² at 1024 |

Bold is where that curve has stopped climbing. Formic, diazaphen,
and DTH are powers of two through 1024 (400, 900, and 1600 are
not on those figures). GC still includes batch 400, where Jacobi
peaks. Same-batch Jacobi speedup versus 8-thread warm is about
13× on formic at 1024 (the speedup panel peaks near 45× at batch
64, before the CPU has filled), 13× on GC at its peak (11× at
1024), 12× on diazaphen at batch 64 and 6× at 1024, and 7× on
DTH at 1024. Against one thread the Jacobi peaks are about 75×,
90×, 60× and 45×. Purify warm meets Jacobi on formic at batch
1024; on DTH it stays under Jacobi and only clears the 8 CPU
threads past batch 32.

Jacobi cold for n>64 is not on the figures. Clearing the warm
basis on a live engine and re-solving left charges ~0.01 e off
the CPU (formic, n=28, stays on the figure and matches to 1e-6).
A fresh Jacobi engine at batch 8, in the table above, matched
to ~1e-6. Purify's few millielectrons are unchanged.

### GC batch 256 — per-kernel time, and why the count is 32

Measured 2026-09-22, kT = 0, one 0.02 Å force step, 256 copies.
Device time is OpenCL `START→END` on each launch (`RUST_DFTB_KTIME=1`).
The quiet walls (no profiling queue) are the ones in the throughput
figure: Jacobi warm 29.4 ms, Jacobi cold 48.5 ms, purify warm 58.0 ms,
purify cold 73.3 ms. The profiled run is slower (37, 63, 70, 86 ms)
because the queue is created with profiling enabled; the split below
is from that run. `batched_gemm_active` is the 16×16 tiled product.
`gemm_regtile_masked` is the register tile, used only for the
purification products.

**16 and 32 are not a schedule.** The SCC loop stops when the charge
RMS drops below 1e-6. Purification also refuses to stop while the
subspace commutator rh = ‖H′K−KH′‖/‖H′K‖ is above 1e-3. The host
reads those two numbers only every 8 steps, so the printed count is
the next multiple of 8 after the stop. Inside that last chunk the
device clears the replica as soon as the residual crosses the line;
later launches in the chunk early-out. We do not have a per-step
timer, so we know the last chunk was required and we do not know
whether the crossing was the first step of it or the eighth.

What the chunk checks actually saw, max over all 256 copies:

| after step | Jacobi warm RMS | purify warm RMS | purify warm rh | purify cold RMS | purify cold rh |
|---|---|---|---|---|---|
| 8 | 3.7e-6 | 1.6e-2 | 1.9e-3, all fail | 8.1e-2 | 1.2e-2, all fail |
| 16 | stopped | 2.8e-3 | 9.2e-4, passed | 6.9e-2 | 5.7e-3, all fail |
| 24 | | 5.7e-4 | 9.5e-4, passed | 4.1e-3 | 1.32e-3, all fail |
| 32 | | stopped | | stopped | |

Jacobi at step 8 is still above 1e-6, so the second chunk is
required. Purify warm's commutator has already passed at step 16.
The charges have not: 5.7e-4 at step 24 is about 500× the stop, so
the fourth chunk is required. Purify cold is above both lines at
step 24. The printed 32 is that measurement, rounded up. It is not
a constant in the source.

Two inner budgets **are** fixed, and they are not checked against
the charge residual:

- Every warm purification step always launches **2** DMM + McWeeny
  rounds (`RUST_DFTB_PURIFY_CORR`, default 2). The certificate can
  make the arithmetic early-out, but the launches are still issued.
  We have not measured whether 1 round, or 3, is what keeps the SCC
  stable.
- The unseeded first step always runs **60** TC2 steps
  (`RUST_DFTB_PURIFY_COLD_STEPS`). No check in the middle. On this
  solve that block is 3.3 ms, once.

Jacobi sweeps are measured. The kernel stops when the off-diagonal
norm over ‖A‖ drops below 1e-6, with a cap of 40. The last warm step
of this run took **1** sweep. We do not have the sweep count of the
earlier steps.

#### Jacobi warm — 16 steps, kernel-exec 35.6 ms

| kernel | role | calls | ms | ms/call | share |
|---|---|---:|---:|---:|---:|
| `jacobi_resident_batched` | diagonalize cᵀHc, rotate c | 16 | 24.35 | 1.522 | 68.5% |
| `cs_normalize_batched` | renormalize C and SC | 16 | 3.31 | 0.207 | 9.3% |
| `batched_gemm_active` cᵀ·H | project H | 16 | 2.22 | 0.139 | 6.2% |
| `batched_gemm_active` S·C | overlap times C | 16 | 2.20 | 0.138 | 6.2% |
| `batched_gemm_active` (cᵀH)·c | finish the projection | 16 | 2.20 | 0.138 | 6.2% |
| `mulliken_cs_batched` | charges | 16 | 0.46 | 0.029 | 1.3% |
| `fused_dq_v_hscc_batched` | Δq → V → H_scc | 16 | 0.33 | 0.021 | 0.9% |
| `diis_step_batched` | mix, clear converged | 16 | 0.30 | 0.019 | 0.8% |
| `select_occupation_batched` | kT = 0 | 16 | 0.12 | 0.008 | 0.3% |
| `extract_diagonal_batched` | read the diagonal | 16 | 0.08 | 0.005 | 0.2% |

#### Jacobi cold — 16 steps, kernel-exec 61.2 ms

This is one cold diagonalization and then fifteen warm steps. After
the first `eigh_finish` the plan sets the warm-basis flag, so the
call counts are 1× `Xᵀ·H` and 15× `cᵀ·H`. The Jacobi total is 45.1 ms
against 24.4 ms when all 16 steps are warm, so that one cold
diagonalization is about 22 ms. The `X·C′` back-transform is 0.24 ms,
once. Charges on this in-place cold flag are 0.017 e off the CPU.

| kernel | calls | ms | ms/call | share |
|---|---:|---:|---:|---:|
| `jacobi_resident_batched` | 16 | 45.11 | 2.820 | 73.7% |
| `batched_gemm_active` cᵀ·H | 15 | 4.03 | 0.268 | 6.6% |
| `cs_normalize_batched` | 16 | 3.00 | 0.187 | 4.9% |
| `batched_gemm_active` S·C | 16 | 2.87 | 0.180 | 4.7% |
| `batched_gemm_active` (cᵀH)·c | 15 | 2.64 | 0.176 | 4.3% |
| `diis_step_batched` | 16 | 1.16 | 0.072 | 1.9% |
| `mulliken_cs_batched` | 16 | 1.00 | 0.063 | 1.6% |
| `fused_dq_v_hscc_batched` | 16 | 0.43 | 0.027 | 0.7% |
| `batched_gemm_active` Xᵀ·H | 1 | 0.24 | 0.237 | 0.4% |
| `batched_gemm_active` X·C′ | 1 | 0.24 | 0.236 | 0.4% |
| `batched_gemm_active` (XᵀH)·X | 1 | 0.24 | 0.236 | 0.4% |
| `select_occupation_batched` | 16 | 0.14 | 0.009 | 0.2% |
| `extract_diagonal_batched` | 16 | 0.08 | 0.005 | 0.1% |
| `occ_normalize_batched` | 1 | 0.07 | 0.069 | 0.1% |

#### Purify warm — 32 steps, kernel-exec 66.5 ms

Two correction rounds per step, so the inner kernels have 64 calls.
The certificate product H′·K is launched 4 times per step (seed,
two re-checks, final), 128 calls.

| kernel | role | calls | ms | ms/call | share |
|---|---|---:|---:|---:|---:|
| `batched_gemm_active` X·K | D = XK … | 32 | 9.08 | 0.284 | 13.6% |
| `batched_gemm_active` (XK)·Xᵀ | … Xᵀ | 32 | 8.78 | 0.274 | 13.2% |
| `batched_gemm_active` (XᵀH)·X | H′ = … X | 32 | 8.78 | 0.274 | 13.2% |
| `batched_gemm_active` Xᵀ·H | H′ = XᵀH … | 32 | 8.23 | 0.257 | 12.4% |
| `gemm_regtile_masked` H′·K | T for the certificate and DMM | 128 | 7.77 | 0.061 | 11.7% |
| `dmm_update_batched` | commutator descent | 64 | 6.55 | 0.102 | 9.8% |
| `gemm_regtile_masked` (K·K)·K | K³ | 64 | 3.24 | 0.051 | 4.9% |
| `gemm_regtile_masked` K·T | Y = KT | 64 | 2.90 | 0.045 | 4.4% |
| `gemm_regtile_masked` K·K | K² | 64 | 2.75 | 0.043 | 4.1% |
| `mcweeny_combine_batched` | K ← 3K²−2K³ | 64 | 1.97 | 0.031 | 3.0% |
| `comm_gate_batched` | rh | 128 | 1.81 | 0.014 | 2.7% |
| `diis_step_batched` | charge mix | 32 | 1.73 | 0.054 | 2.6% |
| `mulliken_charges_batched` | charges from D, S | 32 | 1.41 | 0.044 | 2.1% |
| `fused_dq_v_hscc_batched` | Δq → V → H_scc | 32 | 1.05 | 0.033 | 1.6% |
| `spec_span_batched` | DMM step length | 32 | 0.50 | 0.016 | 0.8% |

#### Purify cold — 32 steps, kernel-exec 82.7 ms

Step 1 is Palser + 60 `tc2_step_batched`. Steps 2–32 are the same
2-round update, so those kernels have 62 calls, not 64. H′·K has
125 calls: one final certificate on the TC2 step, four per later step.

| kernel | calls | ms | ms/call | share |
|---|---:|---:|---:|---:|
| `gemm_regtile_masked` H′·K | 125 | 11.56 | 0.092 | 14.0% |
| `dmm_update_batched` | 62 | 8.56 | 0.138 | 10.3% |
| `batched_gemm_active` X·K | 32 | 8.24 | 0.257 | 10.0% |
| `batched_gemm_active` Xᵀ·H | 32 | 7.40 | 0.231 | 8.9% |
| `gemm_regtile_masked` K·K | 62 | 7.37 | 0.119 | 8.9% |
| `batched_gemm_active` (XK)·Xᵀ | 32 | 7.06 | 0.221 | 8.5% |
| `batched_gemm_active` (XᵀH)·X | 32 | 6.21 | 0.194 | 7.5% |
| `gemm_regtile_masked` (K·K)·K | 62 | 6.09 | 0.098 | 7.4% |
| `gemm_regtile_masked` K·T | 62 | 5.74 | 0.093 | 6.9% |
| `mcweeny_combine_batched` | 62 | 4.74 | 0.076 | 5.7% |
| `tc2_step_batched` | 60 | 3.33 | 0.056 | 4.0% |
| `comm_gate_batched` | 125 | 1.96 | 0.016 | 2.4% |
| `diis_step_batched` | 32 | 1.78 | 0.055 | 2.1% |
| `mulliken_charges_batched` | 32 | 1.24 | 0.039 | 1.5% |
| `fused_dq_v_hscc_batched` | 32 | 0.96 | 0.030 | 1.2% |
| `spec_span_batched` | 31 | 0.49 | 0.016 | 0.6% |
| `tc2_init_batched` | 1 | 0.03 | 0.032 | 0.0% |

Warm's H′·K call is 0.061 ms and cold's is 0.092 ms. Cold stays
uncertified through step 24, so those launches do the full product.
Warm's certificate has passed by step 16 and the later launches
early-out.

#### The 1e-6 line is the wrong question

The outer loop does watch a residual. It is not a hard cap of 32.
What it watches is the wrong residual, and it watches it far past the
point where the answer can still change.

Charge RMS < 1e-6 means the SCC map agrees with itself. It does not
mean the density is the diagonalization. On this GC step the Jacobi
map's fixed point matches the CPU to 2e-6 e. The purify map's fixed
point, after it has passed the same 1e-6 test, is still 7e-3 e off
the CPU (warm) and 3e-3 e off (cold). The missing 1e-3 e is the bias
of two DMM rounds. The steps that take RMS from 5e-4 down to 1e-6 are
making an approximate density consistent with itself more tightly than
it agrees with the truth. §4 already set the chemistry bar at ~0.1 meV
on the scan shape and forces at a few 10⁻⁶ Ha/Å. Nothing in this solve
connects 1e-6 e of SCC residual to that bar. The threshold was inherited
from an eigensolver test.

The warm guess is doing its job on the way down, and the stop throws
that away. Same GC step:

| after step | warm RMS | cold RMS |
|---|---|---|
| 8 | 1.6e-2 | 8.1e-2 |
| 16 | 2.8e-3, rh already passed | 6.9e-2, rh still failing |
| 24 | 5.7e-4 | 4.1e-3, rh still failing |

Warm is ahead at every check. Both still miss 1e-6 at step 24, so
both print 32. A stop at the size of the solver's own bias (~1e-3 e,
which is where warm already is at step 16) would finish the warm
solve in half the steps and leave the cold solve running. That is
the only comparison in which a warm start is allowed to win. Tightening
further does not even improve the final density: after both have
"converged", cold is closer to the CPU than warm. The carried K is
not being used as a correction of a good density. Each step rebuilds
H′ and D with four full products and applies a fixed two rounds, the
same machinery as a cold step.

What to change, in that order:

1. **Stop on the energy and the forces**, or on a charge change of
   order 1e-3 e, which is the error purify has versus Jacobi even at
   RMS 1e-6. On the trace above, that is "purify warm is done at step
   16." Confirm by the §4 contract (scan shape ~0.1 meV, forces a few
   10⁻⁶ Ha/Å), not by a tighter residual. If step 16 already meets it,
   the second half of every purify solve in the throughput figure is
   waste. If it does not, add a round because the update is too weak,
   and do not lower 1e-6 to 1e-7.

2. **A warm step is a correction, then a look at the energy.** One or
   two updates of the carried K, charges refreshed, stop if the energy
   and forces moved less than the tolerance in §4. The four basis
   products (35 ms, half of purify warm) are the cost of treating every
   iteration as a fresh factorization. They are the second lever, after
   the stop. The fixed 2 rounds and the fixed 60 TC2 steps are the same
   habit as the 1e-6 line: a number chosen so a test stays green.

3. **Jacobi is already the true density.** Its waste is doing ~12–16
   full diagonalizations (1.5 ms each, 68% of the solve) to push a
   residual from 4e-6 to 1e-6. The same energy/force stop applies. The
   kernel work under that (DIIS, Mulliken, H build, the 60-step TC2
   block at 3.3 ms) does not move a geometry step.

#### Measured stop, batch 8, same 0.02 Å step

`test_scc_rms_stop`. G0 is still converged to 1e-6, so the warm guess
is a real density. Only the geometry step changes tolerance. Energy is
one finalize after the solve (the test calls `energy()` once). There
is no energy or force kernel inside the SCC loop. Charge error is the
Mulliken vector read before that finalize. ΔE is against that method's
own 1e-6 step, in meV (1 Ha = 27211 meV). §4's shape bar is ~0.1 meV.

| system | method | tol | iters | rms | max\|Δq\| vs CPU | ΔE (meV) | failed |
|---|---|---|---|---|---|---|---|
| formic | Jacobi | 1e-6 | 8 | 5.1e-7 | 8.5e-7 | 0 | 0 |
| formic | Jacobi | 1e-3 | 8 | 1.4e-4 | 1.4e-4 | +0.018 | 0 |
| formic | Jacobi | 3e-3 | 8 | 1.5e-3 | 1.1e-3 | −0.027 | 0 |
| formic | purify | 1e-6 | 24 | 9.2e-8 | 1.9e-3 | 0 | 0 |
| formic | purify | 1e-3 | 16 | 4.9e-4 | 2.1e-3 | −0.026 | 0 |
| formic | purify | 3e-3 | 8 | 2.3e-3 | 2.9e-3 | −0.182 | 0 |
| GC | Jacobi | 1e-6 | 16 | 9.4e-7 | 2.2e-6 | 0 | 0 |
| GC | Jacobi | 1e-3 | 8 | 3.4e-4 | 4.1e-4 | −0.009 | 0 |
| GC | Jacobi | 3e-3 | 8 | 2.8e-3 | 4.2e-3 | −0.513 | 0 |
| GC | purify | 1e-6 | 32 | 5.3e-8 | 7.3e-3 | 0 | 0 |
| GC | purify | 1e-3 | 24 | 5.3e-4 | 7.9e-3 | −0.255 | 0 |
| GC | purify | 3e-3 | 16 | 2.8e-3 | 1.2e-2 | −0.456 | 0 |
| diazaphen | Jacobi | 1e-6 | 16 | 6.0e-7 | 2.1e-6 | 0 | 0 |
| diazaphen | Jacobi | 1e-3 | 8 | 1.9e-4 | 2.3e-4 | −0.019 | 0 |
| diazaphen | Jacobi | 3e-3 | 8 | 2.0e-3 | 2.6e-3 | −0.337 | 0 |
| diazaphen | purify | 1e-6 | 32 | 0 | 4.5e-3 | 0 | 0 |
| diazaphen | purify | 1e-3 | 32 | 8.6e-4 | 6.5e-3 | −0.590 | 0 |
| diazaphen | purify | 3e-3 | 64 | 2.2e-3 | 6.1e-3 | −1.007 | 0 |
| DTH | Jacobi | 1e-6 | 16 | 8.3e-7 | 1.7e-6 | 0 | 0 |
| DTH | Jacobi | 1e-3 | 8 | 6.6e-4 | 4.8e-4 | −0.070 | 0 |
| DTH | Jacobi | 3e-3 | 8 | 2.0e-3 | 1.6e-3 | −0.739 | 0 |
| DTH | purify | 1e-6 | 40 | 2.6e-7 | 5.6e-4 | 0 | 0 |
| DTH | purify | 1e-3 | 100 | 9.2e-4 | 1.6e-3 | −0.336 | 8 |
| DTH | purify | 3e-3 | 100 | 1.6e-3 | 3.0e-3 | −0.451 | 8 |

Jacobi at 1e-3 stays inside the 0.1 meV bar on all four molecules
(largest move is DTH, 0.07 meV) and drops GC, diazaphen, and DTH from
16 iterations to 8. Formic was already done at 8. Jacobi at 3e-3 is
not: GC, diazaphen, and DTH move by 0.3–0.7 meV. The Jacobi warm
geometry step in the bench now uses 1e-3. G0 stays at 1e-6. The
throughput figures and `throughput.csv` are still the 1e-6 grid; a
re-sweep has not been run.

Purify does not get that stop. At 1e-3, GC moves 0.25 meV and
diazaphen 0.59 meV. DTH hits the iteration cap with every replica
Failed: the charge residual is already under 1e-3, but the commutator
gate (`rh` > 1e-3) rearms the replica after DIIS has frozen it, the
skipped steps spoil the path, and `rh` never certifies. At 3e-3,
diazaphen takes 64 iterations instead of 32. Loosening the charge RMS
on top of a fixed commutator tolerance is not a warm start. Purify
stays at 1e-6.

Diagnostics stay off the hot path. `RUST_DFTB_SCC_TRACE`,
`RUST_DFTB_KTIME`, `RUST_DFTB_PURIFY_DEBUG`, `RUST_DFTB_PROF`, and
`RUST_DFTB_JACOBI_REPORT` are read only when set. The per-replica
status dump is skipped when `RUST_DFTB_SCC_QUIET` is set. The
stability run set that and did not turn the others on.

2. **Ship the 3-product geometry update as the inner solve.**
   AO `2D₁−D₀` (or plain reuse when only one previous D exists),
   transport into the current orthogonal basis, one McWeeny, one
   commutator certificate. Persistent program and buffers. The
   resident square, not `batched_gemm_active` per product. Fallback,
   when the certificate misses, is a **short** device-side TC2
   (budget on the order of 8–12 squares, early exit, no host branch).
   A 60-step cold purification and a 16-step DMM block are deleted
   from the hot path. `converged = false` still falls back; it does
   not expand into a second algorithm "to be safe."

3. **Co-iterate K with the charges.** One to three fused squares per
   SCC iteration, K left on the device. SCC acceptance is charge RMS
   and a commutator that is small enough for the energy contract, read
   at the chunk boundary that already exists. Mid-SCC does not need a
   fully idempotent projector. This is the schedule notes Part VIII.4
   calls (A); the kernels already exist.

4. **Smearing, cheap or scoped.** First experiment: stop TC2 when the
   sigmoid width matches kT (notes VIII.8). If the PT-scan shape stays
   inside §4, that is the finite-T method. If it does not, run the
   100× chase at kT = 0 on the gapped systems (formic, AT, azaindole)
   and treat near-degenerate proton-transfer points as a separate
   problem. Do not build the 100-term Chebyshev.

5. **One launch per SCC iteration.** Fuse the short body the way
   `gemm_sq_iter` already fuses a purification step (chat §6, measured
   0.047 ms). A 3-product solve launched as three 0.35 ms GEMMs is
   still slower than Jacobi. The fusion is justified only after step 2
   shows the short schedule is the one that converges.

6. **Cover n ≈ 246 with the same tiled GEMM.** The current purify path
   falls back to a row·row kernel and spends ~900 ms/iteration on DTH.
   That fallback is a broken dispatch, not a large-n method. Block
   Jacobi at n=246 is ~21× CPU and is not the route to 100×.

7. **Only if step 2–3 miss the mid-SCC rotations** (η_w ~ 1, the
   certificate rejects most replicas): occupied-subspace CheFSI
   (carry C, filter H_new·C, orthonormalize — chat §11 / notes III.5).
   Sylvester/Newton CG only after a CPU f64 prototype shows a handful
   of matvecs on a real DFTB ΔH. LNV and learned Λ stay reference code.

Expected class of result, from measurements already in hand, not from
a new promise: a warm inner solve at a few resident squares is
**~0.2–0.5 ms** against Jacobi’s **1.5 ms** at n=86, batch=400. That
alone is not 100× end-to-end, because Jacobi is ~70 % of today’s
iteration and the iteration count is part of the product. The bar is
reached only if the inner solve shrinks, the launch tax goes away, the
SCC iteration count does not inflate, and n=246 is on the tiled kernel.
If a step does not move the scan400 ratio, it is the wrong step.

## 7. Tests that decide

| question | test | pass looks like |
|---|---|---|
| Are we faster than one CPU thread by ≥100×? | `test_gpu_scc_scan400_benchmark`, release, batch=400, vs `DftbCpu` one thread | sys/s ratio reported per molecule; GC and DTH both |
| Does a geometry step reuse D? | `tests/gpu_warm_bench.rs` geometry section | ~3 products, ΔE at the f32 floor, δ up to 0.1 Å |
| Does the SCC tail stay cheap? | same file, SCC section | late iterations certified; early ones fall back in a short budget, not 60 steps |
| Is the chemistry intact? | GC / formic proton-transfer scan vs CPU | shape error in the meV class of the existing Jacobi scan; forces same order |
| Did a kernel change break algebra? | `tests/gpu_purify.rs`, `tests/gpu_scc*.rs` | parity and the explicit `converged=false` path; random matrices are not a timing claim |

Profile with `RUST_DFTB_PROF` event times. A host timestamp that
includes a queue stall is not a kernel time. Print iteration counts,
product counts, and uncertified-replica counts. A green test that
widened its tolerance is a failure of the change.

## 8. Document map

Later text wins inside a file. The chat logs are design discussion;
the notes’ Parts VI–VIII are the measurements that settled it.

### This folder — read

| file | role |
|---|---|
| **this file** | standing order and the path |
| [`TrDH_minimization_purification_Notes.md`](TrDH_minimization_purification_Notes.md) **Parts VI–VIII** | the numbers: real-molecule warm start, production Jacobi-vs-purify table, kT=0 ablation, why the GC failures are smearing |
| same file, Parts III–V | why the random-matrix conclusion was withdrawn; the predictor/McWeeny fix; the benchmark plan that Part VI then ran |
| [`Alternative_Dense_Multi_Eigensolve.chat.md`](Alternative_Dense_Multi_Eigensolve.chat.md) **from the learned-Λ autopsy onward** (~the "fatal problem with learned Λ" turn, then the LNV stop, then the extrapolation repair) | the reasoning behind Parts III–VI. The early GEMM-microkernel turns are done; their measured outcome is `gemm_sq_iter` |
| [`Alternative_Dense_Multi_Eigensolve.md`](Alternative_Dense_Multi_Eigensolve.md) §0 and §7 | utilization gap (Jacobi at 5 % of peak) and the fused-TC2 timings |
| [`Measured_Facts_Jacobi_Sweeps.md`](Measured_Facts_Jacobi_Sweeps.md) | Jacobi ceiling. Closed. |
| [`Dense_Multi_GPU_Optimization.tasks.md`](Dense_Multi_GPU_Optimization.tasks.md) T08, T09, T11 | what Jacobi work finished, the overhead audit (harness is already clean), FOE task status |

### This folder — background, do not steer from these

| file | why it is not the guide |
|---|---|
| Notes Parts I–II | learned-Λ design and the LNV campaign. Superseded. The paradox in II.8 ("warm descent cannot beat cold TC2") is true for a stale D on a gapless matrix and false for the Part VI geometry predictor |
| Chat, first half (triangular SYRK, register tiles) | kernel experiments already measured; do not re-open |
| [`HBond_Relaxed_Scan_GPU.review.md`](HBond_Relaxed_Scan_GPU.review.md) | 2026-09-09 audit. Assembly crash and the 0.4 Ha H/S mismatch were fixed later. Do not restart from its B1 |
| [`HBond_Relaxed_Scan_GPU.manifest..md`](HBond_Relaxed_Scan_GPU.manifest..md) §16 and the early Jacobi boxes | historical campaign. CPU ratios there predate resident Jacobi. §0.7 f32 policy still holds: no f64 in the O(n³) loop |
| [`HBond_Relaxed_Scan_GPU.chat.md`](HBond_Relaxed_Scan_GPU.chat.md), [`Divide_and_Conquare_Jacobi.chat.md`](Divide_and_Conquare_Jacobi.chat.md) | Jacobi-era design. Geometric divide-and-conquer is not the sign-function route; both are behind purification |
| [`Dense_Jacobi_Eigen_Tiling_Opt.md`](Dense_Jacobi_Eigen_Tiling_Opt.md) | tiling proposals. The occupancy idea was measured and mostly falsified (T08) |
| [`Slot_Pool_Scheduler.design.md`](Slot_Pool_Scheduler.design.md) | useful later, once one replica’s solve is actually short. Scheduling a 1.5 ms Jacobi does not create a 100× solver |
| [`Dense_Multi_PBC.*`](Dense_Multi_PBC.arch_notes.md), [`Dense_Multi_CDFT.*`](Dense_Multi_CDFT.spec.md) | periodic and constrained-DFT layers on top of the same SCC. They inherit this mandate; they do not set the inner-solve choice |
| [`HBond_Relaxed_Scan_GPU.report.md`](HBond_Relaxed_Scan_GPU.report.md), [`.labbook.md`](HBond_Relaxed_Scan_GPU.labbook.md) | implementation diary of the H-bond pipeline (assembly, forces, FIRE). Solver choice has moved on |

### Sparse folder — the transferable pages

| file | take |
|---|---|
| [`Sparse_Performance.md`](../Sparse_Nanocrystal_Vibrations/Sparse_Performance.md) | the sparse standing order. Read this rather than re-deriving the Hessian ladder from the report |
| [`Sparse_Nanocrystal_Vibrations.report.md`](../Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md) §15.24–§15.28 | residency, the 27×/121× table, and why SpGEMM batching saturates. The proof that 100× is a property of staying on device |
| [`Sparse_Nanocrystal_Vibrations.manifest.md`](../Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md) §1.4 | the performance language. Reorder it as in §1 of this file before applying it to dense work |
| manifest §15.12 and report §15.14–§15.16 | early-stop, f32-vs-truncation (the "floor" was a mask bug, not a law), short DMM recipe for a nearby density |
| [`Sparse_Nanocrystal_Vibrations.review.md`](../Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.review.md) | 2026-09-09. Stale relative to the report’s September 14–19 entries |
| [`Sparse_MultiSystem_Scheduler.chat.md`](../Sparse_Nanocrystal_Vibrations/Sparse_MultiSystem_Scheduler.chat.md) | variable-convergence batching. Relevant only after the dense inner solve is a few launches long |

### Tests and code named above

- `rust_dftb/tests/gpu_purify.rs` — algebra, cold TC2 timing, gapped extrapolation
- `rust_dftb/tests/gpu_warm_bench.rs` — real-molecule SCC and geometry warm starts
- `rust_dftb/tests/gpu_scc_bench.rs` — `test_gpu_scc_scan400_benchmark`, the throughput bar
- `rust_dftb/src/qmqm/gpu_purify.cl`, `gpu_purify.rs` — TC2, McWeeny, DMM, `warm_extrap_solve`
- `rust_dftb/src/qmqm/gpu_scc_plan.rs` — production dispatch (`EigKind`, `PurifyScc`)
- `rust_dftb/src/qmqm/gpu_gemm.cl` — `gemm_sq_iter`, the fast square
- `rust_dftb/src/methods/dftb/dftb_cpu.rs` — the single-thread denominator
