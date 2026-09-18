# Measured Facts — Dense GPU Jacobi Sweeps (for review/LLM context)

Paste-ready digest of what we already measured. Everything below is **measured**
on an RTX-class OpenCL device (dev max WG=1024, ~82 SM class), batch=400 unless
stated, `--release`, equal inputs per comparison, all configs converged
(stop=0, bad=0) with residual/orthogonality/eigenvalue-parity gates.

## 0. Kernel architecture (so suggestions target the right thing)

- `jacobi_cyclic_global_batched` ("direct", n≤256): ONE workgroup per system.
  **A and V live in GLOBAL memory; every one of ~jround≈n−1 rounds per sweep
  re-streams all of A and V.** Local memory holds only rot_c/rot_s/rot_p/rot_q
  [128] + reduction scratch + Fermi-tail scratch (~7–16 KB at WG512).
- `jacobi_resident_batched` ("resident", new §1b): ONE WG per system, A loaded
  once into dynamic `__local` lA[n·(n+1)] (padded stride kills bank conflicts)
  and kept resident across ALL sweeps. Two compile variants:
  - `RESIDENT_V=0` "res-defV": V stays global; each round logs (c,s,p,q) to a
    global rotlog [batch][jround·jpair] double2; logged rotations are applied
    to V once PER SWEEP (whole V ≈30 KB at n=86 is L1-resident during the
    apply epoch). lA = 29.9 KB at n=86 → 1 WG/SM class, fits 48 KB devices.
  - `RESIDENT_V=1` "res-AV": V also __local (lA+lV ≈ 59.7 KB at n=86) —
    exceeds this device's 48 KB local → untested here, gated by capacity.
  Same contracts as direct: active mask, diag latch + stop codes, fused
  Fermi tail, warm `c` seeding, A written back once at exit (extract_diag
  unchanged).
- `block_jacobi_1wg` ("block", n>128): ONE WG per system. Local: compound pivot
  `P[PB×PB]` + `U[PB×PB]` (PB=2B). Strips still stream A/V from global per pivot.
- `batched_gemm_active` (the "3D kernel"): grid = (row_tile, col_tile, replica);
  dims 0,1 are the 16×16 output-tile decomposition of ONE system — standard
  tiled GEMM, correct as-is. Only dim 2 is the replica axis.

## 1. Direct-kernel WG sweep (arbitrary-WG reductions now legal)

N=86, batch=400, ms per batched solve:

| WG | "one" (warm-ish) | "cold" |
|---:|---:|---:|
| 64 | 34.9 | 65.5 |
| 96 | 32.1 | 65.5 |
| 128 | 30.4 | 61.6 |
| 192 | 29.1 | 59.7 |
| 384 | 25.6 | 54.1 |
| **512** | **25.3** | **52.9** |

N=246: same monotonic trend, WG512 fastest (897/1581 ms).

→ **WG≈N hypothesis FALSIFIED *for this kernel and this regime only*.**
Runtime decreases monotonically with WG. Explanation: the kernel is
bandwidth-bound on A/V streaming (~10 MB/sweep/system → ~20 GB per solve ≈
790 GB/s ≈ 85% DRAM peak; 24 MB working set >> L2). More lanes = more
outstanding memory requests = better MLP — when DRAM bandwidth is the wall,
idle threads are free.

**⚠ REGIME-CONDITIONAL RESULT — do not generalize (user-flagged 2026-09-18).**
This sweep was measured on the *streaming* (bandwidth-bound) direct kernel in
the *queued* regime (400 WGs ≫ resident capacity). It is NOT evidence about
the optimal shape of a compute/local-bound kernel: once A is resident and the
bandwidth wall is gone, "wasting" lanes is no longer free — the resident
kernel runs at 1 WG/SM (~25% thread occupancy) and the optimum WG ×
co-residency product is an OPEN question, not a settled one. Do not cite the
WG512-wins result to justify the current workgroup shape of the *resident*
kernel; it was falsified only for the kernel that no longer ships at n=86.
The same caveat applies to small-slot scheduler regime (S≈64–128, UNTESTED).

## 1b. Resident-A + deferred-V vs streaming direct (THE bandwidth fix)

N=86, batch=400, `resident_jacobi_sweep` — same inputs, identical
res/orth/par to 7 digits (same rotation math), all stop=0/bad=0:

| kernel | WG | tail | "one" ms | "cold" ms |
|---|---:|---|---:|---:|
| direct | 256 | yes | 27.67 | 58.45 |
| direct | 512 | yes | 24.53 | 48.62 |
| direct | 512 | no | 21.92 | 45.67 |
| **res-defV** | 256 | yes | **11.50** | 24.03 |
| **res-defV** | 512 | yes | 12.61 | 23.52 |
| **res-defV** | 512 | no | 11.36 | **19.98** |
| res-AV | — | — | SKIP: 59.7 KB > 48 KB local cap | |

→ **Residency hypothesis CONFIRMED: ~2.0–2.3× over the best streaming-direct
config** at equal accuracy (bit-identical rotation sequence). This is the
lever the WG sweep could not reach — per-round A/V streaming removed.
→ WG insensitivity inside res-defV (WG256 ≈ WG512, ±10%) is itself
evidence the kernel is no longer MLP-starved.
→ Device regime: local_mem_size = **48 KB** → res-defV fits n ≲ 96
(29.9 KB at 86; 66 KB at 128 fails). n=128+ stays direct/block.

## 2. Block-kernel parameter sweep (N=246 unless noted)

17 configs × {one,cold}; dominant lever is INNER_MAX (over-solved inner pivots):

| config | one ms | cold ms |
|---|---:|---:|
| B16/IMAX12/ITOL1e-7 (old default) | 254.0 | 603.1 |
| B16/IMAX12/ITOL1e-5 | 224.6 | 528.6 |
| B24/IMAX1/ITOL1e-6 | 189.3 | 438.8 |
| **B32/IMAX1/ITOL1e-6 (new default)** | 190.7 | **377.2** |

N=86: B16 baseline 19.2/45.0 → B24/IMAX1 12.9/20.3 ms. Outer sweep counts
unchanged; accuracy equal or slightly better. Single-pivot guard: n≤2B keeps
IMAX≥12 (whole matrix IS the pivot).

## 3. Resource footprint (CL_KERNEL_{LOCAL,PRIVATE}_MEM_SIZE)

| kernel | local | private |
|---|---:|---:|
| block B16 | 9.7 KB | 0 B |
| block B24 | 20.2 KB | 16–24 B |
| block B32 | 34.8 KB | 160–232 B |
| direct WG512 +Fermi tail | 16.4 KB | 0 B |
| direct WG512 no-tail | 7.2 KB | 0 B |

→ Private-array spilling hypothesis mostly falsified (0 B at B16; ≤232 B at B32
which still wins). The real B-cost is LOCAL memory. ncu cannot profile OpenCL.

## 4. Rejected variants

- **JACOBI_PREC=0** (f32 rotations): NOT faster AND 10–30× worse eigenvalue
  parity (N246 cold: 7.8e-3 vs 2.6e-4). Rejected at equal accuracy.
  **NOTE (2026-09-18): the production default was nevertheless later changed
  to PREC=0** (commit 3be6c822; `JACOBI_PREC` atomic default 0, env
  `RUST_DFTB_JACOBI_PREC`/`gpu_jacobi_prec()` override). The parity rejection
  above stands as the measured record; the default flip is a policy decision
  (f32-is-the-architecture) — prec=1 remains the accuracy-reference knob.
- **JACOBI_NO_TAIL** (compile out Fermi tail): halves local mem, runtime ±3%
  noise on the direct kernel. On res-defV standalone it IS ~10% (12.61→11.36
  ms "one") because tail scratch shares the WG's local budget — but unfusing
  changes occ values → different SCC trajectory (80 vs 100 iters), so no
  clean e2e gain. Fused tail kept; `RUST_DFTB_JACOBI_NOTAIL=1` for A/B.

## 4b. f64 audit (2026-09-18 — policy: f64 on GPU is a violation unless deeply justified)

| site | precision | verdict |
|---|---|---|
| Jacobi rotations/updates | jrot_t/jupd_t = **f32** at prec=0 | clean (prec≥1 keeps f64 as accuracy reference) |
| **rotlog scratch (res-defV)** | was `double2` | **FIXED → `jlog2_t`: float2 at prec=0, double2 at prec≥1.** Buffer 23→11.5 MB (n=86/b400); log stream 58.5→29 KB/sweep/system. Implemented + verified. |
| Fermi tail (fused in Jacobi) | f64 exp + f64 reductions | ~1–3% of solve; scalar-decision class, allowed by GUIDELINES §3 — but re-verify (see T09). |
| `fermi_occ_batched` standalone | f64, fixed-40-iter bisection | runs only n≤64/ref/block paths; measured 13–31% of dev time at n=6/28. Deferred — T10: warm-μ Newton upgrade. Re-verify indirect damage. |
| `diis_step_batched` QR | f64 | user decision: keep; barrier-bound not arithmetic-bound. Re-verify. |
| `energy_reduce_batched` | f64 accumulation | eval-only, not in SCC loop. Re-verify. |
| `jacobi_cyclic_local_batched` (n≤64) | f64 internal? check | small-N path — audit pending. |

## 5. End-to-end SCC numbers (production; auto = resident n≤128-if-fits, block n>128)

| system | path | batch=400 wall | iters | ms/iter | failed |
|---|---|---:|---:|---:|---:|
| GC N86 | direct+density (orig baseline) | — | — | ~4.55→ | 0 |
| GC N86 | direct+direct-pop | 401.5 ms | 100 | 4.01 | 0 |
| GC N86 | **res-defV + direct-pop** | **218.7 ms** | 100 | **2.19** | 0 |
| DTH N246 | direct+density (orig) | 6358 ms | — | ~351 | 0 |
| DTH N246 | **block B32 + direct-pop** | **1714.8 ms** | 24 | **71.4** | 0 |

GC same-work A/B (100 iters, failed=0, converged rms ~1e-6): 4.01→2.19
ms/iter = **1.84× end-to-end** from resident dispatch alone.

Block same-binary A/B at N246, equal work (24 iters): 78.6→71.4 ms/iter (1.10×
end-to-end; per-solve gain larger but warm probes skip most solves).
EVT profile: Jacobi = **80%** of device time; density/snormalize/mulliken
removed from the loop (T03). GC run hitting 100-iter cap while median replica
finishes ~56 = the straggler tail the slot scheduler targets.

## 6. Current open hypotheses (what we have NOT tried)

- ~~Full-local-resident Jacobi~~ → TESTED (§1b): res-defV ~2.2× confirmed;
  res-AV ~~blocked by 48 KB local cap~~ → **NOW FEASIBLE (§7): packed lA
  made it 53.8 KB — needs only warp-shuffle reductions (~6 KB scratch)
  to fit.** res-AV eliminates rotlog + the ~5 MB/sweep V-replay + ~170
  barriers/sweep — the next structural lever at n=86.
- ~~rotlog double2~~ → **DONE (2026-09-18): `jlog2_t` prec-gated** — float2
  at prec=0 (production), double2 at prec≥1 (accuracy reference keeps
  bit-exact logged rotations). 23→11.5 MB buffer, 58.5→29 KB/sweep/system.
- ~~Packed symmetric lA → 2 WG/SM (the occupancy lever).~~ → **DONE
  (2026-09-18): lA stores only the lower triangle via `lat(r,c) =
  max(r,c)·(max+1)/2 + min(r,c)`** — 14.6 KB at n=86 (was 29.9 KB).
  Phase-2 must skip transpose blocks (`b<a continue`): blocks (a,b) and
  (b,a) map to the SAME packed slots → only a≤b runs (each slot one
  writer; also halves phase-2 block count). red2/dred2 scratch aliased
  away (reductions are sequential). Measured `CL_KERNEL_LOCAL_MEM_SIZE`
  (incl. arg_local on this driver): **24 224 B/WG at WG512+tail**
  → 2×24 224 = 48 448 ≤ 49 152 → **2 WG/SM now fits** (margin ~700 B;
  was 39 KB → hard 1 WG/SM). Parity unchanged (par=1.73e-6, bad=0,
  n=86 one/cold/WG256/512, work-ids subset test, scc suite 8/8).
  **Performance: ~5–9 % faster, NOT 2×** — one WG512 notail 10.67 ms
  (was ~11.4), cold WG512 notail 18.18 ms (was ~20). Interpretation:
  crossing the 2-WG threshold did NOT produce a 2× jump → the kernel
  is not occupancy/latency-bound; the V-replay global/L2 traffic and
  barrier serialization dominate. This is the measured confirmation of
  the user's hypothesis that other bottlenecks mask occupancy effects —
  WG×co-residency remains open (§1 caveat), now with the extra datum
  that 2-WG co-residency alone is worth ≲10 %, not 2×.
  E2E GC batch=400 unchanged in the noise (196.4 ms/2.455 ms/iter,
  80 iters, failed=0); DTH unchanged (block path).
- **Block-kernel pivot skip (cheap decoupling).** `block_jacobi_1wg` runs
  the gather + inner solve + full strip writeback for EVERY (bp,bq) pair
  unconditionally — even when the off-diagonal block is already ~0
  (U≈identity → the strip writeback is wasted streaming). A pivot-level
  off-norm check during gather → skip writeback = free iterations on
  warm SCC sweeps. Unimplemented.
- **Full block decoupling → independent subproblems.** If a whole
  off-diagonal B×B block is numerically zero, the matrix splits into two
  independent eigenproblems (~4× less work at n/2, and each fits smaller
  kernels/more resident WGs). Requires permutation + detection + routing —
  not implemented; the pivot skip above is the incremental version.
- res-defV for the BLOCK kernel at n=246: A alone is 242 KB — can't fit;
  the strip-resident variant (pivot block + streamed strips) is the n>128
  analogue, unimplemented.
- Separate row-tile kernel for the deferred V apply (GPT's original form);
  sweep-epoch in-kernel apply already captured ~all the gain.
- Block kernel WG<n via strided rows (currently requires WG≥n); untested,
  matters only in the small-slot resident regime.
- Joint (N_slots × WG) co-optimization once the slot scheduler exists —
  and WG×co-residency for the resident kernel (see §1 caveat: open).
- Hscc fusion, select_occ skip, warm-μ Newton — small (~0.3–0.5 ms/iter each).

## 7. The deferred-V (rotlog+replay) design — rationale, cost, and why
##    "just accumulate one smaller rotation matrix" doesn't exist

**Why deferred-V exists.** The resident kernel's win comes from keeping A
in `__local` across all sweeps. V was left in `__global` because square lA
(29.9 KB) + square lV (29.9 KB) = ~69 KB > 48 KB cap. Deferred-V is the
workaround: don't touch V during the 85 rounds/sweep; log each round's
jpair=43 (c,s) pairs to a global `rotlog`, then replay all rounds against
gV once per sweep end.

**Why there is no smaller rotation matrix.** User question: "accumulate
the rotations into one local rotation matrix — it must be smaller than all
those rotations." It is NOT smaller, and this is a theorem, not a tuning
detail:

- A real orthogonal n×n matrix has exactly **n(n−1)/2 degrees of freedom**
  (one Givens angle per coordinate plane).
- One full Jacobi sweep applies exactly jround·jpair = 85·43 = **3655 =
  n(n−1)/2** rotations — the same count.
- rotlog/sweep = 3655 float2 = 29.2 KB; the dense accumulated product
  n×n = 29.6 KB. The log is already information-theoretically minimal —
  the product cannot compress below n(n−1)/2 parameters, only re-encode
  them redundantly.

So "smaller accumulated transform" does not exist. What DOES exist is the
accumulator itself: **the product matrix accumulated per round IS V**
(V starts at I cold / previous basis warm; folding each round's rotation
into it per round = `RESIDENT_V=1`, res-AV). Accumulating a separate U
and applying `V ← V·U` once at solve end is the same 29.6 KB of local
plus a wasted n³ matmul — strictly worse than res-AV. No variant beats
"keep V local".

**The cost of deferred-V — FALSIFIED estimate.** Pre-change estimate:
~5 MB/sweep/system of gV traffic + ~170 replay barriers/sweep ⇒
"V-replay dominates, res-AV is the fix." **Measured: wrong.** Batching
the replay to solve end (§8) cut gV traffic ~nsw× and removed ~all
replay barriers — and gained only ~6–9%, not ~1.5×. The replay was
already cheap because V (~30 KB/sys) stays L2-resident across a sweep,
so the ~85 re-streams hit L2 at streaming rate, and its barriers carry
little work to drain. **The dominant cost is the round loop itself:**
85 rounds × ~2–3 barriers × ~7 sweeps ≈ 1200–1800 barriers per solve,
each draining phase-1 (only jpair=43 of 512 lanes busy) and phase-2
(~925 halved block updates ≈ 36 flops/lane — ~nothing). ~10 µs/round
wall at batch-400 = pure latency, not bandwidth. Levers that remain are
about round latency/structure, not V traffic.

**The fix that now exists.** res-AV removes rotlog, the replay phase, and
~170 barriers/sweep in one move — V updates ride in the existing phase-2
loop on local memory. It did not fit before packed-A (69 KB); now:

  packed lA 14.6 + lV 29.9 + scratch 9.3 = 53.8 KB — over by ~5.8 KB.

  Warp-shuffle reductions (reduce[512]→~16 partials, dred likewise) cut
  scratch to ~3.5 KB → **≈48 KB, fits** (comfortably at WG256). res-AV
  is correct BY CONSTRUCTION vs replay because it applies the identical
  rotation sequence to local V. **But §8's measurement caps its upside:**
  the replay it eliminates was worth only ~6–9%, so res-AV's remaining
  value is mostly the simpler kernel structure, not a big win. Deprioritized
  behind structural round-latency work.

## 8. Solve-end V replay — IMPLEMENTED (K=4) + the latency finding

`jacobi_resident_batched` now logs `JACOBI_LOG_SWEEPS=4` sweeps of
rotations to rotlog (`[batch][4][jround·jpair]` jlog2_t = ~47 MB at
n=86/b400/prec0 — sized to stay L2-resident) and replays them in ONE
`jac_replay_V` pass at solve end (same rotation order → bit-identical V;
the off-norm trajectory never touches V). A solve exceeding 4 sweeps
flush-replays mid-solve and resets the log — worst case is K replays,
never worse than per-sweep. Dead rounds (all-identity) skip their apply
pass AND their global fence via a per-round nonzero count.

**Measured (n=86, batch 400, res-defV WG512 notail):**

| mode | per-sweep replay | K=8 log | **K=4 log** |
|---|---:|---:|---:|
| one  (sw≈3.7) | 10.67 ms | 10.15 | **9.72 ms** (−9%) |
| cold (sw≈7.0) | 18.18 ms | 19.37 | **17.10 ms** (−6%) |

K=8 *regressed* on cold: 8 sweeps × 29 KB ≈ 203 KB/sys log → ~80 MB
batch-wide, exceeding L2 — the end replay then reads the log from DRAM
instead of L2-hot. K=4 (≈117 KB/sys, 47 MB) stays L2-sized → wins
everywhere. Parity unchanged (par=1.73e-6 one / 1.91e-5 cold, bad=0);
e2e GC b400 194.6 ms ≈ neutral (warm solves log ≤1 sweep anyway).

**The finding that matters:** ~500× less V traffic + ~170 fewer
barriers/sweep bought only ~6–9%. ⇒ The kernel is **round-latency-bound**,
not bandwidth-bound and not occupancy-bound. Each round serializes:
phase-1 (43/512 lanes) → barrier → phase-2 (~36 flops/lane) → barrier.
Next levers must attack the round structure itself — co-resident WGs to
fill latency (the only occupancy that still matters), fewer/fatter
rounds (impossible at element level — 43 disjoint pairs is already
maximal), or the block-round / D&C restructure of the chat doc for n=246
where pairs stream globally and rounds CAN be fused.
