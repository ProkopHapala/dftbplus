# Alternative dense multi-system eigensolvers — beyond Jacobi

Status: design/analysis note (2026-09-18). Companion to
`Dense_Multi_GPU_Optimization.tasks.md` (T08) and
`Measured_Facts_Jacobi_Sweeps.md` §8.

## 0. Why this document exists — the utilization gap

```
Device f32 peak:                    ~36 TFLOPS
Resident Jacobi, n=86 batch=400:    ~17.6 GFLOP per batch solve
                                    (7 sweeps × 3655 rot × ~24n flops + V)
Measured:                           9.7 ms → ~1.8 TFLOPS ≈ 5 % of peak
```

Two independent measurements (packed-A occupancy gain ~5–9 %, solve-end
V-replay gain ~6–9 %) both say the same thing: **the cyclic Jacobi kernel
is round-latency-bound, not bandwidth- or occupancy-bound.** Each of the
85 rounds/sweep serializes: phase-1 (43 of 512 lanes busy) → barrier →
phase-2 (~36 flops/lane) → barrier. ~10 µs per round is latency, not
arithmetic. Inside this algorithm the remaining ceiling is ~2× (merged
barriers + more co-resident WGs) → ~10 % of peak. **No amount of Jacobi
tuning reaches 15–25 %.**

To get there the eigensolve must be expressed as **batched n³ matmuls** —
the only dense primitive that runs at 10–30 % of peak on this device for
n≈86. This document inventories the candidate algorithms, what each
costs, and the concrete plan for the chosen route.

## 1. Route A — density purification / FOE (chosen first path)

Solve for the **density matrix directly**; never produce eigenvectors in
the SCC loop. All iterations are matmuls.

### The math (orthogonal form — already implemented)

Initial guess from Palser–Manolopoulos canonical purification:
```
D₀ = α(β·I − H̃),   H̃ = Xᵀ H X  (Löwdin-orthogonalized)
```
Trace-correcting TC2 iteration (Niklasson): each step computes D² and
chooses per system
```
if Tr(D) > Nocc:  D ← D²          (contract: raise occupations' spread)
else:             D ← 2D − D²     (expand)
```
converging to the idempotent projector onto occupied states with
**exactly Tr(D) = Nocc at every step** — no chemical potential needed.
Convergence test: ‖D²−D‖_F → 0 (quadratic). McWeeny 3D²−2D³ is the
non-correcting variant (polish step).

### The generalized form (non-orthogonal — the bigger prize)

The **sparse** path already solves the generalized problem directly
(`sparse_bsr4_purification.cl`): metric TC2 on K in the AO basis
```
Q = K S K
if Tr(KS) > Nocc:  K ← Q          else:  K ← 2K − Q
invariant: K S K = K,   Ne = Tr(KS)
generalized McWeeny polish:  K ← 3KSK − 2KSKSK
```
Same equations, dense GEMMs instead of masked BSR4 matmuls. Working in
the AO basis eliminates the **eigensolve AND the Löwdin X** (whose
Newton iterations are themselves a per-SCC cost) — K goes straight to
Mulliken charges (`q_i = 1 − Σ_μ∈i (KS)_μμ`) and forces.

### What already exists (verified inventory)

`rust_dftb/src/qmqm/gpu_matrix.rs` (GpuMatrixPlan — dense, batched):
- `batched_gemm` — tiled batched n³ matmul (the workhorse)
- `purify_palser_manolopoulos` — orthogonal McWeeny loop
- `purify_trace_correcting` — orthogonal TC2, **but with a host readback
  per iteration** (trace → CPU → mode → relaunch) — a manifest §14
  violation; needs a device-side fused TC2 kernel
- `scale_density_guess` + spectral-bounds plumbing (Palser start)
- `trace`, `idempotency_error` — convergence diagnostics
- `local_jacobi_blocks` — small-m fallback

`rust_dftb/src/methods/sparse/` (BSR4 path — the *equations* to copy):
- `sparse_bsr4_purification.cl` — generalized metric TC2 + McWeeny,
  emulated-fp64 accumulation, trace reset, plateau/floor detection

### What's missing for a production dense-FOE path

1. **Device-side TC2**: one kernel does D² (or KS, KSK) then the
   per-element combine choosing the D² vs 2D−D² branch by comparing an
   on-device trace buffer to Nocc — zero host sync inside the loop.
   Convergence checked every K iters or via a device early-exit flag.
2. **Solve-path dispatch**: `SolveKind::{Eigen, Purify}` (or extend
   `EigKind`) alongside the Jacobi path — Jacobi stays default; FOE is
   env/opt-in until parity is proven. Outputs the density K; the eigen
   outputs (ε, C) are simply not produced on this path.
3. **Downstream contract**: SCC needs P (=K) and charges — both direct.
   Forces need the energy-weighted density W = K H K-terms — also pure
   GEMMs (W = P·H̃ in orthogonal form). Fermi smearing is the honest
   caveat: TC2 gives the **0 K** density (integer occupations). At
   kT = 0.002 Ha only states within ~0.01 Ha of μ deviate from 0/1;
   finite-T needs Chebyshev-FOE with μ bisection (still all-GEMM,
   ~5–10 bracket evals × ~25 matmuls) — phase 2. First version targets
   the integer-occupation case and must verify against the eigensolve
   path where smearing matters.
4. **Warm start**: K from the previous geometry/mix iteration converges
   in ~3–6 TC2 steps (vs ~15–25 cold). This is the SCC-regime win —
   analog of the warm-V Jacobi path.

### Expected performance

~10–25 TC2 iterations × 2 batched 86³ GEMMs ≈ 13–32 GFLOP per batch
solve — comparable flop count to Jacobi, but at batched-GEMM rates
(10–30 % peak ≈ 3.6–11 TFLOPS) → **~0.3–0.9 ms vs current ~10 ms** —
the ~10×. Plus the Löwdin X disappears if the generalized form is used.

## 2. Route B — spectral divide-and-conquer via the matrix sign function

**Not the same D&C as `Divide_and_Conquare_Jacobi.chat.md`.** Two
different splits:

- **Chat-doc D&C (geometric)**: Jacobi-annihilate the off-diagonal
  block between index halves, then recurse. Same rotation machinery,
  still Jacobi-latency per node — and it has the correctness caveat that
  a Jacobi-zeroed block is not an invariant separator (later rotations
  re-couple it; freezing requires a fully converged cross-block, i.e.
  block spectral D&C / Riccati — different algorithm).
- **Sign-function D&C (spectral)**: compute `sign(A − μI)` — eigenvalues
  of A map to ±1; the projectors `P± = (I ± sign(A−μI))/2` extract the
  exact invariant subspaces below/above μ. Recursively bisect the
  spectrum (~log₂n = 7 levels), block-diagonalize, read eigenvalues as
  Rayleigh quotients. **Every operation is a GEMM.**

The sign function is computed by Newton–Schulz / QDWH polar iteration:
```
X₀ = A/‖A‖,   X_{k+1} = ½(3X − X Xᵀ X)      (Newton–Schulz, quadratic)
X_∞ = U where A = U·P is the polar decomposition; for symmetric A, U = sign(A)
```
~8–12 iterations × 2 GEMMs per sign evaluation; ~7 levels × a few μ's
→ ~140 batched 86³ GEMMs ≈ 36 GFLOP/solve. At GEMM rates ≈ 3–9 ms —
the same 10× class as FOE, while **keeping the `eigh(H) → ε, C`
contract** (drop-in for the Jacobi slot, eigenvectors produced).

Caveats (why this is phase 2, not phase 1): f32 Newton–Schulz converges
fine for the projector but eigenvalue accuracy depends on Rayleigh
quotients in the extracted basis — needs a NumPy reference to validate
~1e-6-level parity, and degenerate/clustered spectra near split points
need care. Research-grade but well-defined (QDWH-eig, Nakatsukasa–
Higham spectral D&C).

## 3. Route C — one-sided Jacobi (incremental, not 10×)

Apply rotations only to **columns** of A (post-multiplication): Jacobi
orthogonalizes the columns; for symmetric A the result's columns are
eigenvectors scaled by |eigenvalues|. Advantages vs the current
two-sided element kernel: rotations are contiguous column `drot`s —
perfectly coalesced, no 2×2-block index gymnastics, V never exists
separately (folds into A). **But** it keeps the same 85-round
serialization → ~1.5–2× class, not 10×. Worth knowing; not the target.

## 4. Why not Householder/QR

Householder tridiagonalization is ~4n³/3 flops (50× less than Jacobi) —
but at n=86 the panel factorizations are ~86 sequential skinny BLAS-2
steps: same latency-bound structure as Jacobi rounds, just fewer flops
per step. The back-transformation and tridiagonal solve add complexity.
Batched syevd (Householder + D&C) is viable on paper but the tridiagonal
phase is serial-per-system and the implementation cost is highest of
all routes for the smallest structural gain. Skip.

## 5. Recommendation and order

1. **Dense FOE/purification path (Route A) — implement now**, separate
   code path, Jacobi untouched:
   a. NumPy/f64 reference: generalized TC2 on real H,S from the scan →
      parity of K vs C·f·Cᵀ, charge parity vs the eigensolve path.
   b. GPU kernels: fused `dense_tc2_batched` (KS, KSK, device-side branch
      on trace, no host sync) + idempotency check — new
      `gpu_purify.cl`/`gpu_purify.rs`, reusing `batched_gemm`.
   c. `SolveKind::Purify` dispatch in `gpu_scc_plan.rs` — opt-in env/
      config; Jacobi remains default.
   d. Parity + benchmark on GC/DTH (charges, energy, ms/iter, failed).
2. **Phase-1/phase-2 barrier merge + 3 WG/SM** in the resident Jacobi
   (banks ~2× on the fallback path, independent of FOE).
3. **Spectral D&C (Route B)** if a drop-in eigensolver replacement is
   wanted later — NumPy reference first (its own caveat).
4. Finite-T Chebyshev-FOE when smeared-occupation accuracy demands it.

Principle from §0 that gates everything: **the deliverable is flops in
GEMM shape**. Any route that keeps per-round latency serialization —
however clever its memory layout — caps at ~10 % of peak.

---

## 6. Dense FOE implementation design (2026-09-18)

Scope: batched many-small-system purification (batch=400, n≈86) — the
single-system case does not saturate the GPU and is explicitly out of
scope. Separate code path; the Jacobi eigensolver stays default.

### 6.1 Basis choice: orthogonal TC2 first, generalized second

| | orthogonal TC2 (chosen) | generalized metric TC2 |
|---|---|---|
| state | D (density in Löwdin basis) | K (density in AO basis) |
| GEMMs / iteration | **1** (`T = D·D`) | 2 (`T = K·S`, `Q = T·K`) |
| needs | H̃ = XᵀHX, X (already in pipeline) | H, S only — skips Löwdin entirely |
| output | P = X·D·Xᵀ (one GEMM pair, once) | K directly |
| start | Palser `D₀ = σ(λmax·I − H̃)`, σ fixes Tr=Nocc | generalized guess is fiddlier |

The generalized form is the documented phase-2 optimization (drops X and
H̃); orthogonal-first because it is one GEMM/iter, reuses existing X/H̃,
and has the well-understood Palser start.

### 6.2 TC2 semantics — copied from `gpu_sparse.rs::tc2_step`

Per iteration, given D with `tr = Tr(D)`:
```
T = D·D
branch = (tr > Nocc[sys])           // per-system scalar, decided on device
D' = branch ? T : 2·D − T
```
Convergence (fail-loud, same contract as sparse): accept only when
`‖D²−D‖_F < tol` AND `|Tr(D) − Nocc| ≤ trace_tol` — a converged
wrong-rank projector is a failure, not a fallback.

### 6.3 The fused step kernel — ONE launch per iteration

`tc2_step_batched`: one WG per system, ping-pong Din/Dout.

```
pass 1 (GEMM):  compute T = Din·Din tile-wise, write T into Dout;
                accumulate per-thread partials of
                  err += (T_ij − Din_ij)²        → errs[sys] = ‖D²−D‖_F
pass 2 (fuse):  read branch from traces[sys] (= Tr(Din), written by the
                previous iteration / init kernel — no recompute needed);
                Dout_ij ← branch ? T_ij : 2·Din_ij − T_ij   (in-place,
                each thread owns its elements — read/write same slot);
                accumulate Tr(Dout) on diagonal → traces[sys]
```
Zero host synchronization inside the loop. `errs`/`traces` are tiny
per-system buffers; the host reads them once per *chunk* of ~4
iterations to decide continue/stop (~5 readbacks/solve, not per iter).

### 6.4 The GEMM interior — two compile-time variants, benchmarked

The user's constraint: small local-memory footprint → WG count per SM
limited by threads, not local memory.

- **Variant R (row·row, default)**: D is symmetric ⇒ `T = D·D = D·Dᵀ` ⇒
  `T_ij = row_i · row_j` — each thread dots two *coalesced* rows (86 fma).
  **Zero barriers inside the GEMM, zero local memory** → maximal
  occupancy (threads-capped: 8 WGs/SM at WG256). Reads hit L1 (D is
  29.6 KB, fully cache-resident).
- **Variant T (cooperative tile, `PURIFY_TILE=16|32`)**: classic tiled
  GEMM — tA/tB tiles in local (16²: 2 KB; 32²: 8 KB), each thread one
  output element per tile, `nt³` tile-steps with 2 barriers each
  (TILE=32 → 54 barriers/GEMM vs R's zero). For comparison/validation of
  whether local tiling beats L1-resident streaming at n=86.

Both produce identical results; `PURIFY_TILE=0` selects R.

### 6.5 Init kernel

`tc2_init_batched`: per system, Gershgorin bounds from H̃
(`λmax_i = h_ii + Σ|offdiag|`, `λmin_i = h_ii − Σ|offdiag|`, row-reduce),
then `D₀ = σ(λmax·I − H̃)` with `σ = Nocc/(n·λmax − Tr(H̃))` so that
`Tr(D₀) = Nocc` exactly and eigenvalues ∈ [0,~1]; writes `traces[sys] =
Nocc`. Warm-start path: copy previous D (trace already = Nocc).

### 6.6 Remaining pipeline (outside the loop, one launch each)

- `P = X·D·Xᵀ` — two calls to existing `batched_gemm` (once per solve).
- Band energy: `E = 2·Tr(D·H̃)` as a Frobenius inner-product reduce —
  no GEMM needed.
- Charges/Mulliken: existing path consumes AO-density P unchanged.
- Convergence policing: chunked host check of `errs`/`traces`; any
  system past `max_iter` or violating trace tolerance → reported
  `Failed`, never silently accepted (sparse contract).

### 6.7 Integration

`SolveKind::{Eigen, Purify}` dispatch in `gpu_scc_plan.rs`
(`RUST_DFTB_SOLVER=purify` opt-in; Eigen default). New files
`gpu_purify.cl` + `gpu_purify.rs` — Jacobi files untouched. Buffers
(Din, Dout, traces, errs, nocc) allocated once at plan build.

### 6.8 Expected cost

1 GEMM/iter = 1.27 MFLOP/system → cold ~15–25 iters ≈ 8–13 GFLOP/batch,
warm ~3–6 iters ≈ 2–4 GFLOP/batch. At 10–20 % of 36 TFLOPS:
**~0.3–1.3 ms cold, ~0.1–0.4 ms warm** vs ~10 ms Jacobi — the 10× class.
Plus Fermi tail, Löwdin, and V-log machinery all vanish from the loop.

### 6.9 Honest caveats

- 0 K density only (integer occupations). kT = 0.002 Ha smearing needs
  either the eigen path for those runs or finite-T Chebyshev-FOE later —
  parity tests must compare against the kT→0 eigensolve contract.
- f32 purification: TC2 is self-correcting in trace but idempotency
  floor is ~1e-6 in f32 — parity target is charges/energy, not bitwise P.
- Degenerate HOMO/LUMO gaps slow TC2 convergence (more iterations);
  fail-loud max_iter protects the loop.

## 7. Implementation status (2026-09-18, after first bring-up)

**Files:** `src/qmqm/gpu_purify.cl` (kernels), `src/qmqm/gpu_purify.rs`
(wrapper `purify_tc2_batched`), `tests/gpu_purify.rs` (parity + probe +
bench). Separate code path — Jacobi untouched; no plan dispatch yet.

### 7.1 Correctness — VERIFIED

`test_purify_tc2_parity` (n=86, batch=16, nocc=43, tol=1e-5):
converged at 56 iters, all systems
`ΔE ≈ 1e-6 Ha · ‖D−Dref‖ ≈ 3e-5 · ‖D²−D‖ ≈ 1.2e-6 · asym = 0`.
`purify_step_probe` (env-parametrized bisect test): element-wise parity
with an f64 CPU reference ≤ ~6e-6 at every iteration count, batch 1–400.

### 7.2 Bugs found during bring-up (all fixed)

1. **Test-side:** `nalgebra::symmetric_eigen` order is NOT ascending here —
   `eigs[..nocc]`/`vecs[:,k]` picked an unsorted subset (other tests in the
   repo sort explicitly). Test now sorts eigenvalues + permutes columns.
   *This was the false "converged-but-wrong-projector" signal — the GPU
   result was correct all along.*
2. **Device-side freeze required:** the TC2 fixed point is marginally
   stable in f32 — dust eigenvalues regrow ×2/step once converged
   (measured: err 1.6e-6 → 7e-2 in 20 iters). In a *batched* loop,
   converged systems must be frozen in-kernel (`done[]` flag → mirror
   Din→Dout, skip update) or they are destroyed while others finish.
   Single-system sparse path never saw this (it exits the whole loop).
3. **The real race:** `pur_reduce*` ended with `return red[0]` after the
   last barrier — a fast thread could enter the NEXT reduce call and
   overwrite `red[0]` before stragglers read it. Corrupted `lmax/lmin`
   (span ~2–5 instead of ~90 → D₀ blowup ~1e30, Tr ~ −1700, random victim
   per run, mostly high workgroup indices). Fixed: copy `red[0]` to a
   register, barrier, then return. Deterministic ever since.
4. **`pur_reduce` silently dropped contributions for non-power-of-2 lsz**
   (found when fusing the branch/write pass — a red herring: the write
   itself was never wrong). The fold used `h = floor(s/2)` with
   `lid < h` — for odd `s` the LAST element is never folded (e.g.
   lsz=242 drops lids 120,241). The row·row path ran wg=256/512 (PoT →
   correct) and the original regtile pass accumulated `trn` over a
   *strided* element map where the dropped lids happened to own zero
   diagonal elements — so the bug was latent until the fused write moved
   `trn` accumulation to tile ownership (lids 120/241 own diagonals
   40–43,84–85 ≈ 3.1 of the trace → under-counted `trn` → wrong TC2
   branch at iter 7 → wrong-rank projector rank-46, ΔE≈64). Same drop
   also corrupted `pur_reduce_min/max` bounds at wg=242. Fixed:
   `h = ceil(s/2)`, pairs `(i, i+h)` for `i < s−h` — every element
   counted exactly once, no dest/source overlap. Verified: trn exact at
   every iteration, parity restored.
5. Kernel-side `PURIFY_DEBUG_INIT`/`PURIFY_DEBUG` env-gated diagnostics
   remain in the wrapper — they found (3) and (4) in minutes.

### 7.3 Measured performance (n=86, batch=400, seeds 2000+, cold)

| variant | ms/solve | ms/iter | conv |
|---|---:|---:|---|
| row·row WG128 | 72.3 | 1.20 | yes |
| row·row WG256 | 43.7 | 0.73 | yes |
| row·row WG512 | **34.2** | 0.57 | yes |
| row·row WG1024 | 33.6 | 0.56 | yes |
| tiled T16 WG256 | 31.7 | — | **broken** (errs never written; not debugged — secondary path) |
| **regtile-sq two-pass** (22×11thr × 8×4reg, T→C then branch pass) | 10.25 | 0.171 | yes |
| **regtile-sq fused** (same GEMM; branch applied from acc regs — no raw-T round-trip) | **8.46** | 0.141 | yes |

#### 7.3.1 Isolated batched-GEMM variant sweep (n=86, batch=400, 508.84 MFLOP/call, bug-fixed code, 2026-09-19)

`tests/gpu_gemm.rs` — `gemm_bench` (42 configs, CPU f64 parity-verified).
Naming: `reg{TX}x{TY}/r{RTY}x{RTX}/tk{TK}/s{SM}x{SN}[/sq]` = TX×TY thread
grid × RTY×RTX register micro-tile per thread, TK-deep `__local`
k-staging, SM×SN workgroups per system, `/sq` = symmetric-square mode
(B ≡ A — `As_t` transposed staging serves both operand fragments).

**Floors (naive / production baselines):**

| variant | ms/call | TFLOPS | % of ~36T |
|---|---:|---:|---:|
| fulla/wg64 (whole A in `__local`, 29.6 KB) | 1.491 | 0.34 | 0.9 % |
| batched32x32/tk8 (production tile-per-WG) | 0.858 | 0.59 | 1.6 % |
| batched16x16/tk16 | 0.666 | 0.76 | 2.1 % |
| batched8x8/tk16 | 0.603 | 0.84 | 2.3 % |
| batched16x16/tk32 | 0.524 | 0.97 | 2.7 % |
| fulla/wg128 | 0.838 | 0.61 | 1.7 % |
| reg16x16/r1x1/tk16 (tiled 1-elem) | 0.498 | 1.02 | 2.8 % |
| **1elem/wg256** (row·col dot, no tiling) | 0.417 | 1.22 | 3.4 % |
| **1elem/wg512** | 0.280 | 1.82 | 5.0 % |

**Register-tiled, non-sq (C = A·B, both operands staged):**

| variant | ms/call | TFLOPS | % of ~36T |
|---|---:|---:|---:|
| reg4x11/r8x8/tk16/s1x3 (3 WGs/sys) | 0.278 | 1.83 | 5.1 % |
| reg6x11/r8x8/tk16/s1x2 | 0.248 | 2.05 | 5.7 % |
| reg11x4/r8x8/tk16/s3x1 | 0.248 | 2.05 | 5.7 % |
| reg6x6/r8x8/tk16/s2x2 (4 WGs/sys) | 0.245 | 2.08 | 5.8 % |
| reg16x16/r4x4/tk16/s1x1 (256 thr) | 0.228 | 2.23 | 6.2 % |
| reg8x16/r4x8/tk16/s2x2 | 0.218 | 2.33 | 6.5 % |
| reg11x11/r8x8/tk32/s1x1 | 0.192 | 2.66 | 7.4 % |
| reg16x16/r4x4/tk16/s2x2 | 0.190 | 2.68 | 7.4 % |
| reg8x16/r4x8/tk16/s1x1 | 0.181 | 2.82 | 7.8 % |
| reg16x8/r8x4/tk16/s2x2 | 0.172 | 2.97 | 8.2 % |
| reg8x8/r4x8/tk16/s3x2 | 0.171 | 2.98 | 8.3 % |
| reg8x8/r8x8/tk16/s1x1 (64 thr, user-spec shape) | 0.168 | 3.04 | 8.4 % |
| reg8x16/r4x8/tk16/s2x2 | 0.157 | 3.24 | 9.0 % |
| reg11x11/r4x8/tk16/s1x1 | 0.148 | 3.44 | 9.6 % |
| reg8x16/r8x8/tk16/s1x1 (128 thr) | 0.142 | 3.58 | 9.9 % |
| reg8x8/r8x8/tk16/s2x2 | 0.142 | 3.59 | 10.0 % |
| reg8x8/r4x4/tk16/s3x2 | 0.132 | 3.85 | 10.7 % |
| reg8x8/r4x4/tk16/s3x3 (9 WGs/sys) | 0.163 | 3.13 | 8.7 % |
| **reg11x11/r8x8/tk16/s1x1** (88×88 cover, 121 thr) | 0.138 | 3.68 | 10.2 % |

**Symmetric-square (`/sq`, B ≡ A — the purify D² case; requires a
single full-coverage tile):**

| variant | ms/call | TFLOPS | % of ~36T |
|---|---:|---:|---:|
| reg22x44/r2x4/tk16/s1x1/sq (968 thr) | 0.193 | 2.64 | 7.3 % |
| reg44x22/r4x2/tk16/s1x1/sq | 0.157 | 3.23 | 9.0 % |
| reg11x11/r8x8/tk32/s1x1/sq | 0.136 | 3.74 | 10.4 % |
| reg44x11/r8x2/tk16/s1x1/sq (484 thr) | 0.134 | 3.80 | 10.6 % |
| reg11x11/r8x8/tk88/s1x1/sq (whole A, 30 KB) | 0.127 | 4.01 | 11.1 % |
| reg11x22/r4x8/tk16/s1x1/sq | 0.122 | 4.18 | 11.6 % |
| reg22x22/r4x4/tk16/s1x1/sq (484 thr) | 0.123 | 4.13 | 11.5 % |
| reg11x11/r8x8/tk16/s1x1/sq | 0.113 | 4.51 | 12.5 % |
| reg11x11/r8x8/tk44/s1x1/sq | 0.089 | 5.70 | 15.8 % |
| reg22x11/r8x4/tk44/s1x1/sq | 0.090 | 5.68 | 15.8 % |
| reg22x11/r8x4/tk22/s1x1/sq | 0.087 | 5.85 | 16.3 % |
| reg22x11/r8x4/tk8/s1x1/sq | 0.087 | 5.82 | 16.2 % |
| reg22x11/r8x4/tk32/s1x1/sq | 0.085 | 5.99 | 16.6 % |
| **reg22x11/r8x4/tk16/s1x1/sq** | **0.082** | **6.21** | **17.3 %** |

(Best-ever single measurement in this family: **0.075–0.076 ms = 6.7–6.8
TFLOPS ≈ 18.7–18.9 % of peak** — run-to-run variance ±10 %; the
22×11thr × 8×4reg / sq shape is consistently the winner.)

**Reading the sweep:**

- **Floors:** 1-elem global dot ≈ 1.2–1.8 TFLOPS; production
  tile-per-WG `batched_gemm` ≈ 0.6–1.0 TFLOPS (re-staging traffic
  dominates at 400 tiny systems); whole-A-in-local ≈ 0.3–0.6 TFLOPS.
- **Register-tiled non-sq:** rises to ~3.7 TFLOPS at the
  coverage-optimal 88×88 tile — occupancy (400 WGs × 121 thr ≈ 48 K
  threads) and per-WG traffic are the levers; split-WG configs help at
  small tiles but lose to the coverage-optimal single tile.
- **`/sq` is the big win:** `As_t` serves both operands (B ≡ A for
  symmetric D) — halves staging + B traffic → +40–80 % over the same
  shape non-sq.
- **Sweet spot = 242 thr × 32 accums (r8x4)**: more threads at fewer
  accums (484/968-thr shapes) regress — the staging/barrier cost
  outweighs occupancy; fewer accums per thread also cuts ILP.
- **Residual overhead vs isolated GEMM:** fused purify step =
  0.141 ms/iter vs 0.082 ms isolated — ~0.06 ms of barriers/reduces/
  branch bookkeeping per iteration (the next target if worth it).

#### 7.3.2 Batch-size saturation sweep (2026-09-19)

400 systems × 242 thr ≈ 97 K threads vs ~168 K GPU slots — mild
under-occupancy. Scaling batch 400 → 1600 (≈ 390 K threads, WGs queue):

**Isolated GEMM, winner family reg22x11/r8x4/sq (TFLOPS):**

| tk | b400 | b900 | b1600 |
|---:|---:|---:|---:|
| 8 | 5.82 | 6.25 | 6.87 |
| 16 | 6.21 | 5.72 | 7.02 |
| **22** | 5.85 | 6.18 | **7.83 — 21.7 % of peak** |
| 32 | 5.99 | 6.23 | 6.73 |
| 44 | 5.68 | 6.85 | 6.74 |

**Full solve (`purify_bench`, 60 iters, conv=true everywhere):**

| batch | ms/solve | ms/iter | µs/sys/iter |
|---:|---:|---:|---:|
| 400 | 8.97 | 0.150 | 0.37 |
| 900 | 17.72 | 0.295 | 0.33 |
| 1600 | 28.20 | 0.470 | 0.29 |

**Reading it:** the GEMM interior gains ~20–30 % throughput from
b400→b1600 (occupancy was genuinely the limiter at 400); **tk22/sq at
b1600 = 7.83 TFLOPS ≈ 22 % of peak**. The full solve gains ~20 %
per-system (the barrier/reduce overhead benefits less than the GEMM
interior). Floors (1elem, batched_gemm, fulla) are traffic/latency-bound
— flat vs batch. So the ~19 % figure at b400 is not the kernel's
ceiling — ~22 % is the realistic shape ceiling at full occupancy;
beyond that the barriers + reduces + non-GEMM bookkeeping dominate.

- ~60 iters for these random matrices (worst case: tiny Fermi gaps);
  `iters=60 conv=true`.
- Register-tiled symmetric-square GEMM (As_t transposed staging serves
  both operands — B ≡ A) measured **6.7 TFLOPS ≈ 19 % of 36 T peak** in
  isolation (see `tests/gpu_gemm.rs` sweep); inside the fused TC2 step
  the effective 0.141 ms/iter vs 0.075 ms isolated GEMM leaves ~0.066 ms
  for staging barriers + reduces + bookkeeping — the next target.
- **Purify now beats Jacobi even on the COLD worst case:** 8.46 ms vs
  ~9.7 ms warm-resident / ~17 ms cold Jacobi. With warm-start K
  (~5–15 iters) it projects to ~0.7–2.1 ms — >5× faster than Jacobi.
- Row·row kept as `RUST_DFTB_PURIFY_GEMM=0` reference path (0.9 TFLOPS,
  latency-bound serial dots).
- Warm-start input (initial K instead of Palser D₀) is the designed
  next step for the SCC path — not yet wired in the standalone wrapper.

#### 7.3.3 Production SCC stage profile — where purify lands (2026-09-19)

`tests/gpu_scc_bench.rs::test_gpu_scc_scan400_benchmark` with
`RUST_DFTB_PROF=evt` (true per-stage GPU time) + `=mark` (pure host
enqueue time). GC n=86, batch=400 (20×20 scan), kT=0.002, tol=1e-6,
DIIS hist=10, ~80–92 iters/solve:

**Solve-level:** scc = **208.8 ms/call → 2.61 ms/iter**, 1916 sys/s,
24.8× vs CPU f64 point. `eval(true)` forces = **5.98 ms ≈ 2.8 % of
scc** — forces are already cheap. `finishes=91, reads=91` over 3
solves ≈ ~30 syncs/solve (chunk ~7 iters) — **the harness is already
clean: GPU busy 639.5 ms ≈ wall 626 ms → GPU-bound, not sync-bound.**

**Device time per SCC iteration** (dev ms/276 calls ≈ ms/iter):

| stage | ms/iter | % dev | what |
|---|---:|---:|---|
| `scc.jacobi` | **1.76** | **76 %** | warm in-place Jacobi (`eigh_solve`) |
| `scc.diis` | 0.19 | 8 % | GPU DIIS mix (`diis_step_batched`) |
| `scc.gemm_th` | 0.11 | 4.5 % | Cᵀ·H_scc (orthonormalize) |
| `scc.sc_gemm` | 0.10 | 4.3 % | S·C (direct Mulliken input) |
| `scc.gemm_th2` | 0.09 | 3.8 % | (CᵀH)·C → hp |
| `scc.hscc` | 0.03 | 1.4 % | fused dq→V→H_scc kernel |
| eigh_finish/occ/extract_diag | ~0.02 | ~1 % | smalls |
| `scc.chunkend` | ~0.03 | 1.3 % | chunk-end sync (dev side) |
| **launch-gap residual** | **~0.3** | **~13 %** | wall 2.61 − Σdev 2.31; ~11 launches/iter ≈ ~27 µs bubble each |

**Host side** (`=mark`): enqueues are ~3 µs each — all stages ≤ 0.02
ms/call except `scc.hscc` at **0.43 ms host/call**, which is driver
backpressure (first kernel of each iter; the host stalls there when
the command queue is full), not CPU work — `enq_dq_v_hscc` is a single
prebound launch. `scc.chunkend` host = 612 ms = the blocking waits —
the host just waits on the GPU.

**What this means for purify integration:**

- The eigensolve block (jacobi + extract/occ/finish ≈ 1.78 ms) is the
  76 % target — but the *warm* Jacobi already runs 1.76 ms/iter, so
  purify wins only via **warm-start K** (δK0 shift + McWeeny + ~5–15
  TC2 steps ≈ 0.7–2 ms/solve-section, i.e. ~0.7–1.4 ms/iter replacing
  1.78 → SCC iter ~2.6 → ~1.5–1.9 ms ⇒ **~1.4–1.7× end-to-end**, not
  10×). The 10×-per-iter dream requires the eigensolve to become ~free
  AND the surrounding stages to shrink.
- **Production runs kT=0.002 smearing** — TC2 as implemented gives the
  T=0 projector only. The Jacobi tail currently produces occ_w+μ
  fused. For production-parity benchmarking, purify needs the
  finite-T path (Chebyshev/FOE) or a T=0 comparison mode. This is the
  real blocker for `SolveKind::Purify` on the scan workload.
- Generalized-metric purify (K·S·K on H,S directly) would ALSO remove
  `gemm_th`+`gemm_th2` (0.19 ms) and the whole Löwdin X path — the
  bigger architectural win.
- After the eigensolve shrinks, the next targets are: DIIS (0.19),
  the ~0.3 ms/iter launch-gap (≈11 launches/iter — kernel fusion or
  a single fat SCC-iteration kernel), and `sc_gemm` (skippable —
  Mulliken can be computed from D·S directly).
- DIIS fallbacks observed (~50–74/solve, reason=2 nonfinite at
  sid=19) — data point, not benchmark-blocking.

### 7.4 User-confirmed design directions (2026-09-18)

**A. GEMM interior — register-tiled, not row·row.** The user confirms the
matmul is the easy part to optimize and the harness is secondary: target
shape is a classic register-tiled small-GEMM — **8×8 output tiles per
thread, 64 threads/workgroup** (8×8 thread grid → 64×64 output per pass,
2 passes cover 86²). Per-thread: 8+8 fragment loads per k-step, 64 FMAs —
~4× arithmetic intensity over row·row's 2-loads-per-FMA. A/B source tiles
stage through `__local` in modest blocks (not the whole 29 KB matrix —
small tiles keep many WGs resident per CU, per the user's occupancy
requirement). This is the standard SGEMM shape (CLBlast/MaMiMag class),
not an experimental kernel — expect ≥5 TFLOPS batched at n=86.

**B. Warm start — two regimes, and the occupied-subspace rotation trick.**
The user confirms: purification's real payoff is solving the **whole SCC**
(hot start), not a single cold diagonalization — during geometry
optimization the density changes only a little. But a bare
TC2-on-old-D warm start has a known gap (learned in the sparse path,
`sparse_system.rs::k_seed_shift` Phase G3): **polynomials of H cannot
rotate eigenvectors** — when H (and S) change, the occupied subspace
must rotate into the new basin, and no purification step can express
that rotation. The sparse recipe to copy:

```
K0_old = (emax·I − H_old)/Δ_old          # saved from previous geometry
K0_new = (emax·I − H_new)/Δ_new          # rebuilt (cheap: 1 kernel)
K_seed = K_conv_old + (K0_new − K0_old)  # δK0 shift = first-order
                                         # occupied-subspace rotation
McWeeny polish (K←3KSK−2KSKSK)           # contracting toward idempotency,
                                         # no TC2 branch discontinuity —
                                         # the right polish after δK0
TC2 floor-walk to restore trace          # then converge as usual
```

Generalized (non-orthogonal) form needs `S` in the products — K·S·K;
dense equivalents are plain GEMMs. The sparse code also warns:
`RUST_DFTB_WARM_K` experiments **refuted** seeding TC2 directly with the
old converged K — it lands at the masked-map saddle, not the new basin.
The δK0 shift is mandatory, not optional.

**C. Scope note:** purification is for the batched multi-system case —
single small systems cannot saturate the GPU and stay on Jacobi/LAPACK.

### 7.5 Next steps (priority order)

1. ~~**GEMM interior:** register-tiled variant~~ **DONE** — regtile-sq
   (22×11thr × 8×4reg, 88×88 cover, As_t serves both operands) = 6.7
   TFLOPS ≈ 19 % of peak isolated; fused branch/write (no raw-T
   round-trip) → **8.46 ms/solve**, beats warm-resident Jacobi on the
   cold worst-case benchmark. Residual: ~0.066 ms/iter of
   barrier/reduce/bookkeeping overhead vs the 0.075 ms isolated GEMM —
   optional further fusion (e.g. fold the `er` reduce into the GEMM's
   last staging barrier) if worth it.
2. ~~**Warm-start input**~~ — implemented, then **measured refuted in
   all three variants** — see §7.6. The correct warm update is the
   sparse solver's DMM commutator descent, not any K-only seed.
3. ~~`SolveKind::{Eigen, Purify}`~~ — **DONE** (`RUST_DFTB_EIGSOLVER=
   purify`, `EigKind::Purify`, `PurifyScc` state in `gpu_scc_plan.rs`;
   in-loop ortho → TC2 → `D = 2·X·K·Xᵀ` → Mulliken → DIIS). Jacobi stays
   default and is still used by `finalize`/`eval`.
4. Generalized metric form (K·S·K, Tr(KS)) — drops the Löwdin X and
   makes the sparse `k_seed_shift`/`dmm_descend` recipes directly
   portable.
5. Debug or delete the broken `PURIFY_TILE>0` path — currently writes
   no diagnostics; do not let a broken config silently pass.
6. ~~**Implement the DMM commutator-descent warm update + R_H gate**~~
   — **DONE, verified** (§7.6.7): DMM descent + McWeeny retracts +
   device R_H certificate on the final K every solve. The certificate
   is computed per replica into `rh[]` every iteration (1 extra GEMM +
   reduce); a hard gate/restart policy on top of it is the remaining
   piece.
7. **PRIORITY — wire the fast GEMM interior into the DMM/back-transform
   products** (generic `batched_gemm_active` ~0.45 ms → regtile/sq
   ~0.05 ms at n=86 b400, ~9×). Without this the §7.6.7 numbers are
   meaningless. Coordinate with the in-flight `gpu_gemm` work (tri-SYRK
   for the K² retracts; general regtile for H′K products).
8. **Amortized mini-update (§7.6.10)**: 1–2 DMM + 2–4 TC2 per SCC iter
   + rh-driven early exit + full certified solve only at the end /
   gate failures → explicit cold restart.
9. Realistic warm benchmark: cross-geometry reuse in the relax/FIRE
   path (small displacement), not the cold-start SCC trajectory.

### 7.6 Warm-start inside SCC — measured failure modes and the fix that already exists

Purify was wired into the real SCC loop (`EigKind::Purify`) and every
warm-seed idea was benchmarked on GC (n=86, batch=400, kT=0.002) and
H2O, with a per-iteration **commutator certificate**
`comm = ‖H′K − KH′‖_F` read back on host (debug-only,
`RUST_DFTB_PURIFY_DEBUG=1`). For the true occupied projector comm = 0;
trace and idempotency alone **cannot** detect a wrong subspace.

#### 7.6.1 Measured results

| warm seed | replica-0 outcome | comm at "convergence" | verdict |
|---|---|---|---|
| none (always cold, 60 TC2 steps) | E = −44.918316 = Jacobi **exactly**, 16 iters | 4e-6 | **correct baseline** |
| reuse K_old (no rotation) | frozen: q_out constant for any q_in | ~0 but wrong — never responds to ΔH | no-op (K²=K is a hard fixed point) |
| β·D0 blend (β=0.3) | right basin on replica 0, but residual floor: er~1e-2, 379/400 replicas never reach rms<1e-6 | — | residual floor kills convergence |
| δK0 shift K+(D0′−D0) + Gershgorin renorm | converges to a **spurious fixed point**: q_O off by 0.23e, E = −44.976 vs −44.918 | **0.238** | wrong-subspace projector, self-consistent under the purify map but not the true map |
| XL extrapolation 2K₁−K₂ | diverges (extrapolation doubles the per-iter error) | grows monotonically → 3.6 | unstable |

Also measured: cold-path GC at kT=0 is *better* than cold Jacobi at
kT=0 (7.9 vs 10.1 ms/iter, 207 vs 231 failed/400 — GC at T=0 is just
hard; ~half the replicas exceed 100 SCC iters under BOTH solvers).

#### 7.6.2 The structural reason

TC2 (and McWeeny, and every polynomial in K) **preserves eigenvectors** —
it converges to the rank-n_occ projector of the *seed's* eigenbasis, not
of H′. Only seeds that are functions of H′ itself (the Palser seed
D0 = (λmax I − H′)/span) carry the correct occupied-subspace ordering.
Any seed built from old projectors alone must approximate the rotation,
and approximation errors do not self-correct under TC2 — they lock in
(idempotent, right trace, wrong subspace) or accumulate (extrapolation
divergence). Iterating on **H** (sign/Newton–Schulz, FOE/Chebyshev)
can never land on a wrong subspace but forfeits the warm-start advantage
entirely — every solve is ~25–60 full GEMM steps.

#### 7.6.3 The sparse solver already solved this (Phase G3, verified)

`sparse_dftb.rs` `scc_fixedq` / `sparse_system.rs` — the geometry-
displacement warm update, tuned and validated on R10/330Si:

1. `compute_k0_from_hscc` — Palser seed of the NEW H (+ bounds, b_zh)
2. `k_seed_shift` — K ← K_conv + (K0_new − K0_center) — the δ-shift.
   **Itself refuted there** (`RUST_DFTB_VIB_SEED=0`: "the first-order
   change of the spectral *initializer*, not of the occupied projector —
   pushes the good central K off-manifold"). Consistent with our
   comm=0.238 measurement.
3. `metric_transport` — K ← 2K − K·S₁·K (first-order overlap change;
   N/A inside one geometry — S fixed during SCC).
4. **`dmm_descend` — the actual corrector.** n≈6 steps of the
   Riemannian gradient descent on Tr(KH′) over the Grassmannian:
   δK = −η·Sym[(S⁻¹−K)H K], η ≈ 8/(emax−emin), McWeeny retraction
   (3K²−2K³) every 2 steps to hold idempotency. Because the step
   *multiplies K by H*, it rotates the occupied subspace — the piece
   every K-polynomial lacks.
5. Optional final TC2 — default OFF (measured: post-DMM TC2 *degrades*
   R_H; the masked-map branch re-breaks stationarity).
6. **R_H gate** — `rh_stationarity` = ‖A−Aᵀ‖_F/(2‖A‖_F) with
   A = H·(K·S) — i.e. the normalized commutator. Gate: 1e-4 seeded /
   5e-4 cold. Exceeds → **fail loud**, silent re-solve deliberately
   removed ("K is idempotent on a wrong occupied subspace").

#### 7.6.4 Translation to the dense orthogonal-basis path

Our purify already works in the Löwdin basis (H′ = XᵀH_sccX, S = I), so
the recipe simplifies — S⁻¹ = I, Z = I:

- **DMM step (2 GEMMs, not 3):** T = H′·K; Y = K·T;
  K ← K − (η/Δε)·(T + Tᵀ − 2Y); symmetrize K.
  `T + Tᵀ − 2Y = (I−K)H′K + KH′(I−K)` — the occ–virt coupling block.
- **McWeeny retract:** K ← 3K² − 2K³ (2 GEMMs: T=K² then T·K, or one
  fused kernel) every ~2 DMM steps.
- **R_H certificate is free at the last DMM step:** comm = ‖T−Tᵀ‖ since
  (H′K)ᵀ = KH′ — one elementwise kernel + reduce on the already-computed
  T, then gate per replica: fail loud or mark for explicit cold restart.
- **Bounds:** η = η_scale/(emax−emin) — emax/emin from the Palser init's
  Gershgorin bounds (tc2_init already computes them; keep them).

Cost model (n=86, batch=400, sq-GEMM ≈ 0.08 ms): 6 DMM steps ×2 GEMMs +
3 retracts ×2 GEMMs ≈ 18 GEMMs ≈ 1.5 ms/iter — comparable to warm
Jacobi (1.76 ms/iter) at THIS size; the win is the O(GEMM) scaling vs
Jacobi sweeps at larger n and the resident/fused kernels (SqIter work)
collapsing launches. Fewer DMM steps when ΔH small should amortize
further — to be tuned, not assumed.

#### 7.6.5 Design options for the dense SCC path (for discussion)

- **A — DMM warm update (recommended, sparse-proven):**
  seed = K_old (NO shift — refuted in both codes) → ~6 DMM steps →
  interleaved McWeeny retracts → R_H gate per replica. Fail loud /
  explicit cold restart on gate failure. Iterates on H′ → can never
  converge to a wrong subspace *if converged*.
- **B — certificate + cold restart only:** warm TC2 attempt, comm check,
  flagged replicas redo Palser+60 steps. Cheaper per iter but reactive —
  a drifted replica emits a wrong q that iter (DIIS pollution), and the
  recovery needs the full cold cost.
- **C — always-cold:** verified correct today; ~7.9 ms/iter on GC
  (vs 1.76 warm Jacobi) — loses at n=86, wins vs cold Jacobi.
- **D — Newton–Schulz sign on (H′−μI):** correct by construction
  (iterates on H′), but ~25–30 cubic steps/solve — strictly cold.
- **E — extrapolation/δ-shift seeds:** measured refuted (§7.6.1).

#### 7.6.6 Open questions for the design discussion

1. η calibration: sparse used η=8/Δε tuned on geometry deltas (~mHa);
   early SCC iterations have ΔH ~0.1–0.5 eV-scale charge swings — likely
   needs a smaller η_scale or a step-size schedule, plus the renorm
   guard we added for seed kernels.
2. Per-replica vs per-launch policy: the batch launch shares the step
   count; flagged replicas needing cold restart need ~60 steps vs ~6 —
   escalate the whole launch or park-and-retry like `scc_retry_failed`?
3. Is DMM alone (no TC2 at all) the right in-loop map — matching the
   sparse lite tier — with idempotency held only by retracts?
4. Whether the comm certificate should also gate *convergence credit*:
   a replica whose rms is small but comm large is spuriously converged —
   should it count as Failed for the certify path?
5. The kT=0.002 smearing parity issue is still open (purify produces the
   T=0 projector; Jacobi tail writes Fermi occupations) — GC showed
   Jacobi kT=0 vs 0.002 differ negligibly in E (−44.9183 both) but this
   needs a per-system check before equal-physics claims.

#### 7.6.7 Implementation status (2026-09, dense DMM landed)

**Wired in `purify_solve_enq`** (`gpu_scc_plan.rs`, `EigKind::Purify`).
Warm solve = seed K in place → `dmm_steps` DMM descent steps
(2 generic GEMMs + 1 update kernel each) → McWeeny retract every
`dmm_retract` steps (2 GEMMs + combine) → `tc2_tail` optional TC2 steps
→ **R_H certificate every solve** (T = H′·K_final GEMM +
`comm_gate_batched` → `rh[]` per replica). Cold path = Palser +
`steps_cold` TC2 + same certificate.

New kernels (`gpu_purify.cl`): `spec_span_batched` (Gershgorin Δε),
`dmm_update_batched` (pair-owner symmetrizing update + ‖G‖ reduce for
the trust region), `mcweeny_combine_batched` (3K²−2K³, symmetrized),
`comm_gate_batched` (rh = ‖T−Tᵀ‖/(2‖T‖)). Extrapolation machinery
(`tc2_extrap_batched`, K-history buffers) removed from the Rust side.

**The one deviation from the sparse recipe — and why:** sparse uses the
fixed step s = η/Δε with η=8, which was tuned on geometry-perturbation
ΔH (~mHa gradients). Inside SCC the first iterations have O(1) charge
swings → O(1) gradients → η=8 diverges hard (measured: comm 4.6e-7 →
0.72 → NaN in 2 iters). Gradient descent on the quartic manifold is
stable only for s ≲ 1/Δε, so the dense default is **η=1** plus a
trust-region cap s_eff = min(η/Δε, cap/‖G‖_F) (default cap=10 —
inactive except on pathological gradients). With that, DMM is stable
and correct.

**Measured performance — three solver variants, same binary, same
settings** (`test_gpu_scc_scan400_benchmark`, kT=0.002 Fermi smearing,
charge-rms tolerance 1e-6, DIIS history 10, 20×20 scan = batch 400;
columns: total SCC wall time per solve, number of SCC iterations,
milliseconds per iteration, replicas that failed to converge within
100 iterations, converged band-structure energy of replica 0).

**H2O — 3 atoms, n=6 orbitals, batch=400:**

| solver path (in-loop eigensolve) | SCC total (ms) | SCC iters | ms per SCC iter | failed replicas | E0 (Ha) |
|---|---:|---:|---:|---:|---:|
| Jacobi (production default)      | 1.92   | 16  | 0.120 | 0   | −4.075284 |
| DM-purify, cold (Palser+60 TC2)  | 5.96   | 16  | 0.373 | 0   | −4.075284 |
| DM-purify, warm (16 DMM + 4 TC2) | 36.51  | 88  | 0.415 | 0   | −4.075284 |

**GC — 29 atoms, n=86 orbitals, batch=400:**

| solver path (in-loop eigensolve) | SCC total (ms) | SCC iters | ms per SCC iter | failed replicas | E0 (Ha) |
|---|---:|---:|---:|---:|---:|
| Jacobi (production default)      | 221.2  | 80  | 2.77  | 0   | −44.918315 |
| DM-purify, cold (Palser+60 TC2)  | 753.6  | 100 | 7.54  | 207 | −44.918316 |
| DM-purify, warm (16 DMM + 4 TC2) | 2360.6 | 100 | 23.6  | 236 | −44.918315 |

**What this table says, in plain terms:**

- **Correctness: the warm-DMM path is verified.** Replica-0 energy is
  bit-for-bit the Jacobi energy (−44.918315 Ha) on every row, and the
  per-iteration commutator certificate rh ≈ 2e-7 proves the projector
  sits on the correct occupied subspace — not merely idempotent.
- **Speed: the warm-DMM path is currently ~8.5× SLOWER per iteration
  than Jacobi on GC** (23.6 ms vs 2.77 ms per SCC iteration), and ~3.5×
  slower on H2O. It is correct but not yet competitive.
- The reason is arithmetic volume, not overhead: one warm iteration
  runs ~47 dense products through the generic `batched_gemm_active`
  tiled GEMM (~0.4–0.5 ms each at n=86, batch=400): 16 DMM steps × 2
  GEMMs + 8 McWeeny retracts × 2 GEMMs + ortho (2) + back-transform (2)
  + certificate (1). The dedicated regtile/sq GEMM measured ~0.05 ms at
  this size — roughly 8–9× faster per product — so swapping the GEMM
  interior is the single biggest lever (that work is in progress in
  `gpu_gemm.cl` by a parallel effort; the DMM path deliberately still
  uses the production GEMM to avoid interference).
- **GC convergence failures are a physics gap, not a solver bug:** the
  purify path produces the T=0 projector while production runs
  kT=0.002 Fermi smearing. The scan contains near-gapless geometries
  where the T=0 SCC map is ill-conditioned — Jacobi at kT=0 shows the
  same ~200/400 failure regime, and Jacobi at kT=0.002 gets failed=0.
  Finite-T occupations (FOE/Chebyshev) are required for production
  parity on GC-like systems.
- Cold purify converges in the same 16 iterations as Jacobi on easy
  replicas but ~2.7× slower per iteration on GC (60 fused TC2 steps at
  ~0.08 ms + ortho/back-transform GEMMs); on H2O it is ~3× slower per
  iteration.

**Knobs:** `PURIFY_DMM_{STEPS,ETA,CAP,RETRACT}`, `PURIFY_TC2_TAIL`,
`PURIFY_WARM`, `PURIFY_COLD_STEPS`, `PURIFY_TOL`, `PURIFY_DEBUG`
(per-iter tr/er/rh + host comm on replica 0).

#### 7.6.8 Why the current warm-vs-cold numbers are misleading — hypotheses

The table above must NOT be read as "warm purify is 10× slower" — three
separate artifacts stack in this measurement, none of which is the
warm-start idea itself:

1. **This benchmark is the worst case for warm start, not the intended
   one.** Every SCC iteration rebuilds H_scc from a DIIS-mixed charge
   update — early iterations move q by O(0.5 e), i.e. ΔH is O(eV) —
   huge. The sparse-proven DMM recipe is designed for (and verified on)
   *small* perturbations: a converged projector + a small geometry step
   (relaxation/MD) or a nearly-converged charge update. The intended
   dense benchmark is: converge SCC at geometry A → move atoms slightly
   → reuse K across the geometry boundary (the `relax`/FIRE path), or
   at minimum the late-SCC regime where ΔH is already small. Measuring
   warm cost averaged over the full cold-start trajectory conflates the
   hard early iterations (where warm should arguably not even run —
   cold purify or Jacobi does the same work more cheaply) with the
   regime it was built for.
2. **Fixed step budget, no early exit.** The warm path currently runs a
   FIXED 16 DMM steps + 4 TC2 every iteration regardless of need. Near
   convergence (ΔH→0) the seed is already correct — rh is computable
   per step almost for free (T = H′K is already the step-1 product, and
   comm = ‖T−Tᵀ‖ needs one elementwise pass). Early-exit on rh < gate
   would collapse late iterations to ~1–2 products and should also fix
   the iteration COUNT: 88 vs 16 iters on H2O means the bounded budget
   leaves residual comm that perturbs the q-map; exiting on the
   certificate (not a step count) makes the warm map exact again.
3. **Unoptimized interior.** DMM products run on the generic
   `batched_gemm_active` tiled GEMM (~0.45 ms each at n=86/b400) while
   the dedicated regtile/sq GEMM measures ~0.05 ms (~9×). The DMM path
   deliberately avoided the in-flight GEMM work; wiring it in cuts the
   warm iteration from ~23.6 ms toward ~2.6 ms before any early-exit
   savings. Per-iter cost also scales with step count — the certificate
   should drive it.
4. **Rotation cadence.** Strictly, a subspace rotation is only needed
   when H (or S) *changes* — i.e. once per iteration here since charges
   update every iteration, but with size ∝ ΔH. The certificate gives
   exactly that signal for free: rh small ⇒ skip/shorten the descent.
   For the geometry-displacement use case (the sparse design point),
   the rotation runs once per geometry step, not per SCC iter — and a
   converged K needs no TC2 at all inside the subsequent SCC if the
   charge map is evaluated directly off K.
5. **GC failures are orthogonal** — the T=0-vs-kT=0.002 smearing gap
   (§7.3.3), shared with Jacobi at kT=0.

**Bottom line:** the warm-DMM mechanism is implemented and certified
correct (E0 exact, rh≈2e-7); its current cost numbers measure a
worst-case scenario with an unoptimized interior and a fixed step
budget. **The §7.6.7 timings are therefore NOT the deliverable
performance numbers** — they measure the generic GEMM, not the
algorithm. Absolute priority next iteration: wire the fast sq/regtile
GEMM interior (measured ~0.05 ms/product at n=86 b400 vs 0.45 ms
generic) into the DMM products, then re-measure.

#### 7.6.9 What actually runs per SCC iteration — pseudocode + constants

Current implementation (v1 — "full certified solve every iteration"):

```
every SCC iteration (charge update via DIIS):
    hp  = Xᵀ·H_scc·X                      # 2 GEMMs   (orthogonalize)
    if seeded (warm):
        spans = Gershgorin(hp)            # 1 small kernel
        repeat N_DMM = 16 times:          # per step: 2 GEMMs + 1 update
            T = hp·K ; Y = K·T            #   (generic tiled GEMM ~0.45 ms)
            K ← sym(K) − s·(T+Tᵀ−2Y)      #   s = min(η/Δε, cap/‖G‖),
                                          #   η=1.0, cap=10
            every 2nd step: retract       # +2 GEMMs + combine:
                K ← 3K²−2K³               #   (McWeeny, 8 retracts total)
        repeat N_TAIL = 4 TC2 steps       # fused sq steps ~0.08 ms
    else (cold):
        K ← Palser(hp) ; N_COLD = 60 TC2 steps
    cert: T = hp·K_final ; rh = ‖T−Tᵀ‖/2‖T‖   # 1 GEMM + 1 kernel
    D = 2·X·K·Xᵀ                          # 2 GEMMs (back-transform)
    q_new = Mulliken(D, S)                # 1 kernel
    DIIS mix → next iteration
    ≈ 47 generic GEMMs + ~10 small kernels per iteration (warm)
```

So: **16 DMM steps run every SCC iteration** — every time the charge
mix produces a new H. That is exactly the unaffordable structure; the
fix is below.

#### 7.6.10 Proposed v2 — amortized mini-update (the design to test)

The user's proposal — a few cheap steps per iteration, letting the SCC
loop itself amortize the projector convergence — is sound, because at
the SCC fixed point q stops changing → H′ freezes → the mini-updates
still drive K to the *exact* stationary projector of the final H′. The
in-loop projector may be approximate mid-flight; **correctness is
enforced by the certificate at the end** (rh gate on the last solve
before accepting charges/forces), not by per-iteration exactness.

```
every SCC iteration:
    hp = Xᵀ·H_scc·X                       # 2 GEMMs (ortho — unavoidable)
    if seeded:
        N_DMM_ITER = 1–2 descent steps    # 2–4 GEMMs — H-tracking only
        McWeeny retract every 2nd SCC iter # 2 GEMMs
        N_TC2_ITER = 2–4 cheap sq steps   # ~0.08 ms each, idempotency/trace
    else:
        cold: Palser + N_COLD TC2
    D = 2·X·K·Xᵀ ; q_new = Mulliken ; DIIS
final iteration (or when rms < tighter tol):
    full certified solve: DMM until rh < gate (early exit) → accept
    replicas failing the gate → explicit cold restart (fail-loud)
```

Per-iteration cost at the fast GEMM (~0.05 ms/product): ~6–10 products
≈ **0.3–0.5 ms/iter** — already below Jacobi's 1.76 ms/iter at n=86.
The ortho GEMMs (2) and back-transform (2) then dominate — the
generalized-metric form (K·S·K on raw H,S) removes the ortho pair.

Cheaper/riskier rotation candidates to try under the same cert gate
(even if only approximate — the certificate keeps them honest):

- **single-step linear response** (sparse `linear_response` tier — the
  lite path): exactly ONE gradient step = 2 GEMMs/iter.
- **η scheduling**: larger step early (rotation far), ~η/Δε late.
- **skip-when-stationary**: if rh(K vs new hp) < gate before stepping,
  skip DMM entirely that iteration (T from the cert doubles as the
  gradient input).
- **geometry-mode**: across a relaxation step, rotate once (the sparse
  use case), then pure TC2 inside the new SCC — needs the
  `set_geometry` hook to run DMM once instead of clearing the seed.

**Still open:** hard gating of `rh` into the replica status (currently
computed but only read under DEBUG); finite-T occupations for purify;
faster GEMM interior; adaptive step count; `PURIFY_TILE>0` path is still
broken/silent.

---

## 8. Key code locations (file:line relevance map)

For code review / LLM-assisted discussion — every kernel and entry
point of this path, exact locations (2026-09-19, will drift with edits):

### GPU kernels — `rust_dftb/src/qmqm/gpu_purify.cl`

| lines | symbol | role |
|---:|---|---|
| 57–103 | `pur_reduce` / `pur_reduce_min` / `pur_reduce_max` | workgroup fold-reduce; **the non-PoT-lsz fix lives here** (`h = ceil(s/2)` fold — see §7.2 #4) |
| 111–148 | `__kernel tc2_init_batched` | Palser–Gershgorin init: Gershgorin row bounds → `D0 = (λmax I − H)/span`, writes `traces[sys] = Tr(D0)` |
| 157–346 | `__kernel tc2_step_batched` | **the fused step — everything per iteration**: done-mirror early-return (183–186), regtile-sq GEMM interior (188–254, `As_t` serves both operands), `er` reduce + `l_trd`/`l_frozen` broadcast (305–315), branch `contract` (317), fused branch/write from `acc` regs (319–335), `trn` reduce → `traces` (344–345) |
| 257–336 | `#elif PURIFY_TILE` / `#else` | broken tiled variant + row·row reference GEMM (kept for A/B) |

### GPU kernels — `rust_dftb/src/qmqm/gpu_gemm.cl`

| lines | symbol | role |
|---:|---|---|
| 40 | `__kernel gemm_1elem` | 1-thread-per-element row·col dot — the floor |
| 82 | `__kernel gemm_regtile` | **the tuned kernel**: `GEMM_{TX,TY,RTX,RTY,TK,SPLIT_M,SPLIT_N,SQ}` compile-time shape; TX×TY thread grid × RTY×RTX register micro-tile, `As_t`/`Bs` local k-staging, `/sq` mode stages only `As_t` (B ≡ A) |
| 198 | `__kernel gemm_fulla` | whole-A-in-`__local` variant (traffic-bound, loses) |

### Rust wrapper — `rust_dftb/src/qmqm/gpu_purify.rs`

| lines | symbol | role |
|---:|---|---|
| 29–38 | `enum PurifyGemm` | GEMM interior selector (`RUST_DFTB_PURIFY_GEMM`) |
| 45–57 | `regtile_shape` | derives (tx,ty,rtx,rty,tk) from `n` — the 8×4 micro-tile choice |
| 61–80 | `render_purify_source` | template `#define` substitution (wg/tile/gemm/shape) |
| 83–93 | `struct PurifyDiag` | errs/traces/iters/converged result |
| 100–260 | `purify_tc2_batched` | **the entry point**: buffer alloc, kernel build (moved to plan-level later), init launch, chunked ping-pong loop (223–235), convergence check (250–253), final-buffer select (257) |
| 264 | `trace_tol` | `max(2e-5·Nocc, 1e-4)` — same as sparse path |

### Rust wrapper — `rust_dftb/src/qmqm/gpu_gemm.rs`

| lines | symbol | role |
|---:|---|---|
| 23–36 | `enum GemmVariant` | `OneElem` / `RegTile{tx,ty,rtx,rty,tk,split_m,split_n,sq}` / `FullA` |
| 59–82 | `render_source` | `#define` substitution per variant |
| 86–142 | `gemm_kernel` | builds the `Kernel` (global = batch·split·wg; sq needs full-coverage tile → fail-loud at 125–130) |

### Tests / benchmarks

| file | symbol | role |
|---|---|---|
| `tests/gpu_purify.rs:56` | `test_purify_tc2_parity` | parity vs sorted CPU eig (ΔE, ‖D−Dref‖, ‖D²−D‖, asym, ‖HD−DH‖) |
| `tests/gpu_purify.rs:201` | `purify_step_probe` | per-iteration element-wise bisect tool (`PURIFY_PROBE_{ITERS,BATCH,TOL,SEED,REPS}`) |
| `tests/gpu_purify.rs:304` | `purify_bench` | solve-level bench (`PURIFY_BENCH_BATCH`, µs/sys/iter) |
| `tests/gpu_gemm.rs:54` | `variants()` | the 42-config sweep list |
| `tests/gpu_gemm.rs:124` | `test_gemm_variants_parity` | all variants vs f64 CPU |
| `tests/gpu_gemm.rs:183` | `gemm_bench` | isolated ms/TFLOPS/%peak table (`GEMM_{N,BATCH,REPS}`) |

### Semantics sources (outside this path)

| file | symbol | role |
|---|---|---|
| `src/methods/sparse/gpu_sparse.rs:~1701` | `tc2_step` | sparse metric-TC2 — the branch semantics this mirrors (`Tr(KS)>Nocc → KSK else 2K−KSK`; acceptance needs idempotency AND trace) |
| `src/methods/sparse/sparse_system.rs` | `k_seed_shift` (Phase G3), `mcweeny_polish`, `dmm_descend` | the warm-start recipe to port: δK0 subspace rotation + McWeeny polish + TC2 floor-walk (§7.4-B) |
| `src/qmqm/gpu_eigen.rs` | `eigsolver_kind`, Jacobi kernels | the default eigensolve path this is an *alternative* to — untouched |
| `src/qmqm/gpu_scc_plan.rs` | SCC plan | future `SolveKind::{Eigen,Purify}` dispatch site |

### Reproduce

```bash
cd rust_dftb
cargo test --release --test gpu_purify                    # parity + probe
cargo test --release --test gpu_purify purify_bench -- --ignored --nocapture
cargo test --release --test gpu_gemm gemm_bench -- --ignored --nocapture
GEMM_BATCH=1600 GEMM_REPS=30 cargo test --release --test gpu_gemm gemm_bench -- --ignored --nocapture
PURIFY_BENCH_BATCH=1600 cargo test --release --test gpu_purify purify_bench -- --ignored --nocapture
```

