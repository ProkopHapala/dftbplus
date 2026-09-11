# HBond Relaxed Scan GPU — Lab Book

Dense multi-system GPU DFTB path (`GpuDftb` / `GpuSccPlan` / `dftb_engine`).
Work order: GPT-5.6 dense review, manifest §12. Hardware: RTX 3090, FP32
kernels, FP64 scalar islands only. SK: `~/SIMULATIONS/dftbplus/slakos/mio-1-1`.

Conventions: measure = `scripts/test_gpu_dftb_measure.rhai`
(H2O mixers, AT frozen-H + CPU parity, formic-dimer ΔE scan).
`|dE|` = |E_gpu − E_cpu| total energy. δ_CH = internal band-energy
consistency diagnostic (E_band − 2ΣCᵀHC), not CPU parity.

---

## 2026-09-11 — Session: D1–D5, D9, D10, D2, D6

### Baseline (D0) + error decomposition (D1)

- AT |dE| was ~2.8e-5 Ha, blamed on "f32 floor". Decomposition showed it is
  NOT a floor: δ_norm = −4.7e-6 (occupied cols not S-normalized) +
  δ_res = 2.75e-5 (ε_k ≠ Rayleigh quotients, i.e. Jacobi vectors drifted).
- Lesson: decompose before declaring a precision floor. The dominant term
  was vector quality, fixable in f32.

### D3/D4 — occupied renorm + Rayleigh weights — DONE

- `occ_normalize_batched` (renorm C′ occ cols, preserves directions) and
  `occ_rayleigh_batched` (ρ_k = cᵀHc/cᵀSc for E_band and W).
- `occ_repair` (finalize) + `occ_repair_scc` (per-iteration) both default ON.
- Result: AT |dE| 2.8e-5 → ~1e-6; in-SCC renorm took AT from stalled@25
  (rms 1.3e-6) to converged 13 iters. This was the single biggest accuracy win.
- Lesson: the SCC residual plateau was set by eigenvector normalization noise,
  not by charge mixing.

### D5 — GPU Löwdin repair — DONE

- Replaced host f64 serial GEMMs with 5 batched f32 GEMMs + metric kernel +
  strict-improvement accept kernel. Only 2·batch floats read back.
- Measured: AT 5.2e-6→3.6e-7, GC 5.7e-6→3.6e-7, AZA 3.1e-6→3.6e-7,
  formic ~1.5e-6→2.4e-7 per geometry.
- CPU path kept as dead reference (`repair_lowdin_x`).

### D9 — anchored GPU DIIS — DONE

- Mixes Δq = q − q_latest (anchored affine; Σc=1 algebraically exact).
- f64 Gram + tiny solve kept. Drop-oldest retry before α-mix fallback.
- `printf` removed; per-system flag/reason device buffers reported by Rust:
  `DIIS fallbacks: N total, last reason=…` (1=pivot/scale 2=nonfinite 3=Σc≠1).
- `b_mat`/`rhs` dead buffers deleted.
- Observed: formic sees occasional reason=1 fallbacks — reported, not hidden.
  Worth revisiting if fallback frequency grows.

### D10a — occupied-index density — DONE

- `select_occupation_batched` emits `occ_idx[Nocc]`; density/EDM loop occupied
  only (49 vs 87 at AT), symmetric triangle + mirror, local-cached weights,
  4-accumulator FMA.

### D10b — fused Δq→V→H_scc — DONE

- `fused_dq_v_hscc_batched`: one WG/system; Δq,V in local; dq/v still written
  to global for the energy dot. Replaced 3 launches.
- Verified bit-consistent (identical printed energies before/after).

### D2 — Jacobi precision A/B — DONE, default = prec 1

`JACOBI_PREC`: 0=pure f32-FMA, 1=f64 scalar c,s only + f32-FMA updates,
2=broad f64 reference. `tests/gpu_tiled_jacobi.rs jacobi_prec_bench` (N=87,
batch=1):

| prec | time | residual | orth | eig parity |
|------|------|----------|------|------------|
| 0 | 43 ms | 4.8e-5 | 7.9e-6 | 1.6e-3 |
| 1 | 82 ms | 2.1e-6 | 2.2e-7 | 3.0e-5 |
| 2 | 243 ms | 1.2e-6 | 1.4e-7 | 4.2e-5 |

- prec0: 23× worse residual — would reintroduce the δ_CH problem.
- prec1 ≈ prec2 accuracy at ~3× speed → default 1 (`gpu_jacobi_prec` Rhai knob).
- Production check: AT |dE|=1.43e-6 (17 it), AZA 8.0e-7 (12 it).
- **Caveat found:** GC plateaus at rms=1.6e-6 (tol 1e-6) under prec1 where
  prec2 converges at 9.9e-7. Energy still |dE|=1.15e-6. Eigenvector noise
  floor raises the achievable SCC residual floor ~60%. Now reported as
  `plateau` status, not "converged".

### D11 (partial) — honest per-system status

- `SccStatus::{Converged, Plateau, Failed}` per system in `scc_mix`;
  plateau-detector break → Plateau; max_iter-while-descending or non-finite →
  Failed; never conflates. Rhai `gpu_scc_status(name)`.
- N>64 cluster test tightened 1e-2 → 1e-4 (measured |dE|=4.4e-5, |dq|=9.1e-6,
  |d_eig|=1.1e-5 — legacy path).
- NOT done: device `active[sid]` early-outs — no-op at batch=1, deferred to
  when batch≫1 scans exist.

### D6 — X reuse across geometry steps — BUG FOUND, FIX IN FLIGHT

Intent: Newton-polish old X vs new S (≤3 steps, tol 1e-5) instead of full
Jacobi(S). Mechanically works (e0~1.8e-2 → 3.6e-7 in 2 steps for dz=0.2
formic move).

**Bug:** scan parity collapsed — |ΔE_gpu−ΔE_cpu| ≤8.4e-7 → up to 1.8e-4.

Root cause (confirmed by gpu_measure at the bad point):
`‖XᵀSX−I‖=3.7e-7`, `‖CᵀSC−I‖=7.2e-7` — all orthonormality fine —
BUT `‖HC−SCε‖=1.2e-3` and `max|ε−ρ|=3.1e-4`, `max|q_D−q_cpu|=5.3e-4`.

The SCC builds H' = **X·H·X** (both GEMMs plain), which equals XᵀHX only
because Jacobi-built X is symmetric (S^{-1/2}). Newton reuse converges to
X = S^{-1/2}·U for orthogonal U — metric-certified but NOT symmetric, so
X·H·X is the wrong eigenproblem. D5's single polish step stayed ~symmetric
(O(E²)), which is why it never showed before.

**Fix landed:** `transpose_batched` kernel maintains `x_t = Xᵀ` at end of
`set_geometry`; `k_matmul_xh` binds `x_t` so H' = Xᵀ·H·X is correct for any
gauge. `gpu_x_reuse(name,bool)` Rhai A/B knob. Verified: reuse ON vs OFF both
give |ΔE_gpu−ΔE_cpu| ≤ 1.0e-6 over the formic scan (was 1.8e-4 before fix),
all converged. Reuse takes 2 Newton steps (~6 GEMMs) vs full Jacobi rebuild.
Note: the k=3 "plateau" under broken-X was a wrong-eigenproblem symptom,
not a real numerical plateau — status reporting correctly flagged it.

**Lesson (important, keep):** "XᵀSX=I certified" does NOT imply
"X is a valid symmetric Löwdin transform". The gauge freedom is orthogonal;
any code path that implicitly assumes Xᵀ = X breaks silently. The metric
alone cannot detect it — need eigenpair residual or charge parity as the
certifier.

---

## Current verified numbers (post-x_t fix, prec=1, D3/D4/D5/D6/D9/D10 on)

- H2O: |dE|=5.2e-10, 5 iters, forces 1.5e-6.
- AT (N=87): |dE|=6.4e-7, plateau@25 (rms 1.18e-6 — prec1 floor boundary;
  converges or plateaus run-to-run near tol=1e-6), forces 3.5e-6.
- GC (N=86): |dE|=1.15e-6, plateau@25 (rms 1.6e-6), forces 4.0e-6.
- AZA (N=84): |dE|=8.0e-7, 12 iters converged.
- Formic scan (reuse ON, Newton 2 steps/geom, no Jacobi rebuild):
  |ΔE_gpu−ΔE_cpu| ≤ 1.0e-6 over dz=0..0.8 — matches reuse OFF.
- gc/AT plateau caveat: at tol=1e-6 the N~85 systems sit at the prec1
  eigenvector-noise floor. Honest `plateau` status reported; |dE| stays
  ≤1.2e-6. If a future production target needs certified rms<1e-6 on GC,
  use `gpu_jacobi_prec 2` (3× Jacobi cost) or relax tol.

## Design notes (user discussion 2026-09-11)

- `H'=XᵀHX` costs nothing extra: same 2 GEMMs per SCC step; only the first
  operand is a stored `x_t` (one O(N²) transpose per set_geometry ≈ µs).
  Needed for correctness under the reuse gauge, not for precision.
- **Hybrid f32/f64 roadmap** (target: the prec1 eigenvector-noise floor that
  sets the GC/AT SCC plateau): leverage order — (1) compensated (TwoSum/
  Kahan) accumulators in the tiled-Jacobi strip dots + block updates
  (~1.29M roundings; ~4 flops/FMA vs f64's 1/32–1/64 rate on 3090 — should
  approach prec2 accuracy well below prec2 cost); (2) f64 accumulate in
  `occ_rayleigh` ρ_k scalars; (3) f64 accumulate in `frobenius_trace` +
  `dot_batched` energy reductions; (4) f64 in rms/Mulliken atom reductions.
  Items 2–4 are nearly free; item 1 is where the SCC floor is actually set.

## Open / next

- ~~Finish x_t wiring~~ DONE — reuse ON now correct.
- D7: device-resident geometry (per-template pairs, on-device r,l,m,n).
- D8: pretabulated γ/γ′ spline (also fixes energy–force consistency).
- D12: FIRE physics fix on CPU first (α‖v‖F̂, per-replica dt/α/n_pos),
  validate vs CPU FireOptimizer, then GPU.
- `gpu_bench()` event timing + rewrite `gpu_scc_bench.rs` off legacy driver.
- GC prec1 plateau: acceptable (reported) or revisit tol/prec trade-off.

## Gotchas learned (don't rediscover)

- Cargo builds to shared target `~/.cargo/shared_target`; run via
  `cargo run --release --bin dftb_engine` or that binary path.
- `gpu_new(name)` requires a geometry of the SAME name (`load_xyz(name,…)`).
- Sparse module is edited concurrently by another agent — whole-crate
  builds/tests break intermittently; wait and rebuild.
- Batch=1: DIIS hist = min(10, n_atoms); hist > n_atoms is rank-deficient
  by construction.

## 2026-09-12 — D8 done: γ/γ′ natural-cubic species-pair spline

**Module:** `methods/dftb/gamma_spline.rs`. Representation follows the
SPAMMM convention (`LCAO_grid.cl::evaluate_radial`,
`DFTBplusParser::_spline_d2_uniform`): natural cubic stored as
**float4 per knot = (T, T″_spl, T′, T‴_spl)**, T(r)=1−r·γ(r)
(smooth: T(0)=1, T(∞)→0). Curvatures solved tridiagonally (Thomas,
f64) at build — no analytic γ′′/γ′′′ needed. γ=(1−T)/r;
γ′=−T′/r−(1−T)/r²; r≥r_max → pure Coulomb.

**Failed approaches (recorded so we don't repeat):**
- Single Hermite (T,T′) + derivative eval: knot f32 noise (~6e-8)
  amplifies as 1/dr → γ′ err 2.9e-6 at nk=1024 and WORSE (7.8e-6) at
  nk=2048. Finer grid ≠ better when the eval differentiates noise.
- Two Hermite value-splines worked (γ′ 1.3e-7 at nk=1024) but
  natural-cubic float4 matches the repo convention and is one
  vectorized load per endpoint.

**Final:** nk=256, dr≈0.157 bohr (0.083 Å), r_max=40 bohr.
Physical range (r≥1.0 bohr): max|Δγ|=5.1e-7, max|Δγ′|=1.7e-6.
FD consistency |γ′−dγ/dr|=7e-6 at r=1.0 edge (both splines' truncation;
force impact ~6e-7 Ha/Å ≪ 1.5e-6 parity floor). Table: 64KB total,
4KB/pair — `__local`-sized. Scan in `spline_error_vs_nk` test:
nk=192 (dr=0.21) gives γ′ err 1.9e-6 — marginal; 256 is the pick.

**Wired in:** `fill_gamma` (host G build) and `force_gamma_deriv_batched`
both eval the same table → energy–force consistent; the f64
`gamma_prime_full_f32` island in the force kernel is gone.
Force parity test: max|F_gpu−F_cpu| 3.4e-8 (was ~1e-6 with f64 path —
the table removed a subtle inconsistency, G was built from analytic
γ while force used analytic γ′ — now both are the same function).

**Enables D7:** G and γ′ can now be built ON-DEVICE — the kernel needs
only (coords, species_idx, table) — no host pair loop, no f64.

## 2026-09-12 — D7 done: device-resident geometry

**Change:** `GpuDftb::new` now builds the pair lists ONCE from the template
— ALL i<j pairs per (block_type, s_i, s_j) bucket, exact sizing (no
pair_cap, no cutoff membership filtering). Per `set_coords`:
- 2 coord uploads (Å + bohr) — the only host→device traffic
- `refresh_pair_geom` (per bucket): in-place r,l,m,n from device coords.
  In stored order the direction is always (c_j−c_i)/r — orient_pair's
  1×4 i↔j swap is baked into the static atom fields.
- `build_gamma_batched`: G on device via the D8 spline (upper-tri fill).
- `onsite_diagonal` + `assemble_pairs` per bucket. No finish(), no zero
  uploads — every off-diagonal element is written by a pair block each
  geometry, diagonal by onsite / persistent S=I.
- `refill_pairs` retired (commented); scratch_h0/v/g, s_ident fields gone.

**Bug found by GC run (λ_min(S)=−6241):** `cubic_interp_params` clamps the
stencil INDEX but t=u−i keeps growing → far pairs extrapolate the last
stencil → garbage. Fix: `assemble_pairs`/`force_pairs`/
`force_pairs_scc_shift` skip or zero-write pairs with `r ≥ (n_grid−1)·dr`
(zero-write needed because H0/S are no longer pre-zeroed).

**Validation:** H2O/sp3/formic force parity ≤5e-7; n64 batch parity;
GC |dE|=6.4e-7 max|F_gpu−F_cpu|=3.3e-6; AZA |dE|=9.3e-7, F 5.9e-6.
Assembly vs CPU: max|ΔH0|~1.7e-7, |ΔS|~2e-6.

**Note:** far pairs now burn SK evals (~zero output). For dense ≤~100-atom
systems that's 435..4950 pairs — trivial. If it ever matters, compact the
list per geometry on-device (stream-compaction) — deferred.

## 2026-09-12 — D12 done: FIRE physics fixed

**Bugs fixed (both GPU `GpuDftb::fire_step` and CPU ref `FireOptimizer`
in examples/hbond_ref.rs):**
- Mixing was per-atom `α·|F_i|·F̂_i` — pins each atom's speed to its
  force; NOT Bitzek. Correct: `v ← (1−α)v + α·F̂·‖v‖` with GLOBAL
  per-replica norms (all 3·n_atoms dofs share one ‖v‖,‖F‖).
- dt/α/n_pos were single shared state across the batch — now
  `Vec` per replica (batched systems adapt independently).
- Added vmax cap (2.0) that the CPU ref had but GPU lacked.
- Ordering: P=F·v → adapt → mix → v+=F·dt → x+=v·dt (post-update v,
  matching FireOptimizer; displacement cap 0.1 Å kept via apply_disp).

**Validation:** GC relax on NVIDIA: max|F| 7.6e-2 → 8.1e-4 Ha/Å in 45
steps, E −44.9210 → −44.9278 Ha monotone. New Rhai bindings:
`gpu_fire_step(name,f_tol)`, `gpu_relax(name,max_steps,f_tol,scc_tol)`.
Script: `scripts/test_gpu_fire_gc.rhai`.

**Not done:** FIRE on GPU device (v/x/F buffers on device — currently
forces are read back per step). For scans that's ~100KB/step at
batch=1000 — fine for now; revisit if profiling shows it.

## 2026-09-12 — USER OBJECTION (recorded verbatim, priority reset)

User complaint after watching the GC PT-scan run:

> "1) we wanted to do parallel scan — that is whole point of multi solver to
> run all configurations along the scan in parallel using single kernel, it
> should save a lot of time
> 2) we must make SCC convergence robust and fast, running 300 SCC steps is
> not acceptable. If you think the problem is numerical noise floor we should
> adjust the tolerance to something realistically achievable, but the whole
> point of the previous optimization was to make numerical accuracy good
> despite using GPU f32 arithmetics...
> you are again running some slow test. We are developing lightning fast
> code, I cannot accept that these tests are so fucking slow!!!! they must be
> done in milliseconds!!!
> (this is already going on for week and I have to stress it every time since
> you LLMs keep neglecting this)"

**This is the project priority restated: batch-parallel execution IS the
product. A test that takes minutes is a failed design, not a validation.
SCC convergence is either fast (≲20 iters) or the tolerance is wrong —
never "run 300 iterations and hope".**

## Diagnosis of what went wrong in the PT scan run

The scan DID run correctly batched (batch=19, one kernel launch per SCC
stage covers all replicas — the architecture is right). The slowness was
NOT the solver's per-iter cost — it was NON-CONVERGENCE:

1. **First `gpu_scc` on 19 strained geometries: rms=2.2e-1, ALL plateau.**
   rms 0.22 is NOT the f32 floor (~1e-6..1e-5) — that is failed/diverging
   convergence. Replica 0 is nearly the native GC geometry which converges
   at rms=9e-7 in batch=1 — so plateau at 0.22 in batch=19 means either
   (a) DIIS/per-system state mishandled for non-identical geometries, or
   (b) the plateau detector trips on a genuinely oscillating residual and
   masks divergence, or (c) genuine electronic difficulty at mid-transfer
   (proton between donor/acceptor → strong charge redistribution, DIIS
   oscillation). Not yet discriminated — needs per-replica rms dump.

2. **relax() does not gate FIRE on SCC convergence.** FIRE stepped on
   forces from FAILED SCC (rms~1e-2) → garbage gradients → FIRE wandered
   for hundreds of steps, each step re-running 100-iter SCC. Total run:
   minutes. The compound failure: bad SCC → bad forces → long FIRE →
   more bad SCC.

3. **scc_tol=1e-6 is at/below the measured f32 plateau** (AT/GC plateau
   ~1e-6..1e-5). For workflows, `Plateau` status must count as done —
   it IS the achievable accuracy. But rms=0.22 plateau is a BUG, not a
   floor.

**What a correct run looks like:** warm-started SCC ~5-15 iters/geometry,
~30-60 FIRE steps, 19 replicas parallel → total ~seconds. The per-iter
cost measured: 0.4 ms/iter at N=28 batch=32 — the machinery is already
there; convergence is what failed.

**Action plan (not yet executed):**
- Dump per-replica rms trajectories on the scan's first SCC call —
  determine whether plateau is a detector bug or real DIIS oscillation
  per strained geometry. Check if replica 0 alone converges (isolate
  batching bug vs physics).
- relax() must refuse to FIRE on non-converged SCC (or report which
  replicas are unconverged and still move — explicit, not silent).
- Set workflow scc_tol to the achievable floor (~1e-5) and treat
  SccStatus::Plateau as acceptable-done for scan purposes, FAILED as
  hard stop.

## 2026-09-12 — PT scan diagnosis resolved + Fermi smearing implemented

**Finding (diag_scc_batch.rhai):** batch=4 IDENTICAL GC → all converged
9.1e-7/18 iters — batching machinery is fine. Scan points: d=1.0 ✓ 9e-7,
d=1.4 ✓ 8.5e-7, d=1.6 plateau@1.5e-6 (floor, fine), **d=1.9 FAILED
rms=1.1e-2** — genuine SCC divergence, reproduces at batch=1 with all
mixers (DIIS 1.1e-2, simple 0.22, host-f64-DIIS 2.3e-2).

**Root cause (diag_cpu_d19.rhai):** HOMO–LUMO gap at d=1.9 = **0.45 mHa**
(vs 94 mHa native) — integer occupation flip → Δq jumps O(1) → fixed-point
map nearly singular. CPU f64 grinds through in 85 iters (RMS oscillates
0.5→0.09→0.07→2e-4→8e-8); GPU f32 never escapes the oscillation.

**Fix implemented (Fermi smearing, user-approved direction):**
- `occ_w[batch*n]` f32 per-orbital Fermi weights; μ per replica by host
  bisection on eig_diag (~KB readback/iter); density kernel
  `build_density_occ_batched` gained `use_w`/`occ_w` args (smearing: loop
  all N orbitals, weight f_k); W uses f_k·ε_k; E_band = 2Σf_kε_k on host.
- `gpu_smearing(name, kT)` Rhai; kT=0.002 Ha ≈ 630 K.
- **d=1.9 result: FAILED → plateau@3.4e-6 in 37 iters.** Converged at the
  f32 floor.
- Bisection bug found+fixed: Σf(μ) is INCREASING in μ — first version had
  the bracket backwards (E blew to −378 Ha; obvious, loud, fixed).

**Also fixed while diagnosing (the "plateau masks divergence" bug):**
- scc_mix plateau detector tested the BATCH-MAX rms — one stagnating
  replica broke the loop for everyone AND "plateau" was declared for any
  stagnant residual (rms=0.2 counted!). Now per-replica ring buffers +
  done flags; Plateau requires stagnant AND r<5e-4 floor; stagnant-above-
  floor = Failed. Per-replica rms printed with statuses.
- `fire_step` parks Failed replicas (zero v, no integrate) instead of
  stepping on garbage forces; frozen atoms excluded from the replica's
  max|F| test (their force is the constraint reaction).
- `scc()` warm-start retry: Failed replicas restart from the NEAREST
  converged replica's charges (adiabatic continuation); per-replica merge
  keeps the better outcome (a converged replica re-seeded from its own q
  can land a hair above tol — must not be demoted).

**Real scan result (test_gpu_ptscan_gc.rhai, batch=19, kT=0.002, tol=1e-5):**
ALL replicas converged EVERY step — zero failures, no warm-start needed.
E(r) profile: flat ~0-0.4 kcal/mol to d=1.6, then barrier to ~7.8
kcal/mol at d=1.9. Wall time 108 s.

**Speed accounting (diag_step_cost.rhai, batch=19, N=86):**
warm scc ≈ 77 ms (~6-14 iters × ~10 ms/iter — Jacobi-dominated),
eval+forces ≈ 74 ms, fire_step ≈ 80 ms. Relax step ≈ 0.7 s; ×150 steps
≈ 108 s. Per-step fixed cost is the hog, NOT per-iter count.
Remaining levers: device-resident FIRE (kill ~10 roundtrips/step),
fewer FIRE steps (the run didn't reach 1e-3 — hovered ~1e-2),
event-timing breakdown inside the SCC iter (Jacobi sweep count).

## 2026-09-12 — USER CONCERN: 108 s is still 100-1000× too slow (NOT yet hunted)

User: robustness fix is good (fixed small iter cap + retry-only-for-failed
is the right pattern), but **108 s for a 19×150-step scan of an 86-orbital
system is not acceptable — expect ms-to-seconds.** Verdict: there must be
horrendous overheads in the harness; something stupid is still happening.
This was recorded BEFORE hunting — hypotheses for the next session:

1. **Jacobi eigensolve is likely THE hog.** ~10-15 ms/SCC-iter at N=86
   (vs 0.4 ms at N=28 — Jacobi is the only O(N³)-sweep part). tiled_jacobi
   may do far more sweeps than needed on near-diagonal H′, or per-sweep
   launches dominate. Instrument with OpenCL event timing first.
2. **Warm-start SCC should take ~3-5 iters, not 6-14.** Check whether
   DIIS history/reset is thrown away every scc() call (reset_diis on
   set_coords? on retry?) — if history is reset per FIRE step we pay
   cold-start DIIS every step.
3. **Double eigensolve per step.** relax = scc (N iters, each with a
   Jacobi) THEN eval → finalize does ANOTHER full H′ Jacobi on the same
   q — the last scc iter already diagonalized it. Reuse C/ε.
4. **Host roundtrips per step ~10+:** rms readback EVERY iter (blocking
   sync → drains queue), energy_from_state reads 5 buffers, forces read,
   coords write, eig readback for smearing. Each is a serialization point;
   at ~20-50 µs latency × ~150×19 that's only ~ms... BUT if any read
   forces a full pipeline flush per call it's worse. Profile before
   fixing.
5. **set_coords → full reassembly + Löwdin X re-polish per FIRE step**
   even when coords move <0.001 Å — could skip/defer X rebuild (reuse
   tolerance check exists; verify it's actually hit).
6. **eprintln per step** — log I/O per step is real at this scale.
7. FIRE itself: max|F| hovered ~1e-2 for the last 30 steps — check if
   that's a smeared-force inconsistency (W built with f_k but reported E
   uses same weights — should be consistent; verify FD) or a genuinely
   stiff direction / parked-replica accounting bug.

**Physics caution:** dE jumps 2.3 kcal between d=1.65→1.7 then ~7.1-7.8 —
could be a real proton-transfer electronic-state switch (H crossed the
midpoint) or a SCC basin switch between replicas. Validate 2-3 points
against CPU before trusting the profile shape.

**Not committed — user commits manually.**

## 2026-09-12 (later) — OVERHEAD HUNT: **ROOT CAUSE FOUND — it is the Jacobi kernel, 83 ms per 87×87 eigensolve**

Hypothesis #1 was right, but the magnitude was far worse than assumed, and
hypotheses #2/#4/#5/#6 are all **wrong / negligible**. Hard evidence below.

### Measurement 1 — SCC cost vs batch (scripts/diag_scaling.rhai)

GC N=86, warm SCC (already converged → **1 iteration**), 3 reps:

| batch | ms per scc call | µs/iter/system |
|-------|-----------------|----------------|
| 1     | 74.4            | 74440          |
| 4     | 73.3            | 18325          |
| 19    | 75.3            | 3963           |
| 64    | 77.2            | 1206           |
| 256   | 260.2           | 1016           |

**ONE SCC iteration = 74 ms, FLAT from batch 1→64**, then ×3.4 at batch=256.
That is the exact signature of **one workgroup per system**: batch ≤ 82 (SM
count) runs all workgroups concurrently, so wall time = the serial time of a
SINGLE workgroup; batch=256 needs ⌈256/82⌉≈3.1 waves → 3.4×. At batch=19 we
use **19 of 82 SMs = 23% of the GPU**, each running one 256-thread WG (8
warps) → essentially zero latency hiding. The RTX 3090 is ~99.99% idle.

### Measurement 2 — isolated Jacobi, OpenCL event timing
(`cargo test --release --test gpu_tiled_jacobi jacobi_prec_bench -- --ignored`)

| N | batch | prec | median event | residual | orth | eig parity |
|---|-------|------|--------------|----------|------|------------|
| 87 | 1 | 0 | **47.1 ms** | 4.8e-5 | 7.9e-6 | 1.6e-3 |
| 87 | 1 | **1 (production)** | **83.2 ms** | 2.1e-6 | 2.2e-7 | 3.0e-5 |
| 87 | 1 | 2 | **266.8 ms** | 1.2e-6 | 1.4e-7 | 4.2e-5 |
| 87 | 8 | 1 | 82.6 ms | 3.6e-6 | 2.3e-7 | 1.4e-4 |
| 128 | 1 | 1 | 196.2 ms | 4.4e-6 | 2.4e-7 | 2.7e-4 |

**The Jacobi kernel IS the 74 ms.** Nothing else in the SCC iteration matters.
batch=8 costs the same as batch=1 (82.6 vs 83.2 ms) — confirms Measurement 1.

**Perspective:** LAPACK `dsyev` on 87×87 = ~1–2 ms on ONE CPU core.
cuSOLVER `syevj` batched ≈ 0.3 ms. We are **~50× slower than a single CPU
core** and ~250× slower than a proper GPU eigensolver. An 87×87 symmetric
eigendecomposition is ~10 MFLOP; at 35.6 TFLOP/s FP32 that is **0.3 µs of
arithmetic**. We spend 83 000 µs. **Arithmetic efficiency ≈ 0.0004 %.**
THIS is why we are slower than Fortran DFTB+ — DFTB+ calls LAPACK.

**These numbers were already in the labbook (D2 table: 43/82/243 ms) and were
read as "prec1 is 3× faster than prec2 → good default". Nobody asked why one
87×87 eigensolve costs 82 ms. That is the process failure to fix.**

### Measurement 3 — full accounting of the 108 s scan

`relax()` per FIRE step issues **three** independent `finalize`-class
eigensolves plus the SCC ones:
1. `fire_step()` → `eval(true)` → `finalize` → Jacobi
2. `scc()` → 6–14 iterations → **6–14 Jacobi**
3. `eval(false)` → `finalize` → Jacobi — **purely to print `E[0]` in the log**

≈ 8 Jacobi/step × 83 ms ≈ 660 ms/step × 150 steps ≈ **100 s** ✓ matches the
measured 108 s to within the FIRE bookkeeping. The whole runtime is ~1200
Jacobi calls. **Nothing else (readbacks, set_coords, eprintln, DIIS,
assembly, GEMMs) is measurable at this scale** — hypotheses #2, #4, #6 are
refuted; they are noise next to 83 ms.

### Measurement 4 — sweep count (`RUST_DFTB_JACOBI_SWEEPS` knob added)

GC N=86 batch=19, one warm SCC call:

| sweep cap | warm ms | rms | cold SCC iters |
|-----------|---------|-----|----------------|
| 1 | 658 | 5.5e-2 | 43 |
| 2 | 1166 | 5.2e-3 | 41 |
| 3 | 269 | 6.9e-5 | 60 |
| 5 | **56** | 4.4e-6 | 10 |
| 8 | 80 | 4.1e-6 | 10 |
| 16 | 80 | 4.1e-6 | 10 |
| 100 | 80 | 4.1e-6 | 10 |

- The stagnation detector fires at **~8 sweeps** (16 and 100 are identical) —
  so we are NOT burning 100 sweeps. Good news: the cap is not the bug.
- **~10 ms PER SWEEP.** One sweep = 3 block pairs (n_blocks=⌈86/32⌉=3), each
  a 64×64 inner Brent–Luk Jacobi with `PB-1 = 63` **sequential
  barrier-synchronised rounds** per inner sweep, plus N×64 strip updates.
  Estimated ~2000 barriers per outer sweep, ~16 000 per eigensolve, at
  ~5 µs each. **The kernel is barrier-latency-bound inside one workgroup,
  not FLOP-bound.** That is the true mechanism.
- Capping below 5 sweeps is counterproductive: the eigenvectors get so bad
  the SCC needs 41–60 iterations instead of 10 (net slower). Sweeps and SCC
  iterations are coupled — do not "optimize" one blindly.

### Latent BUG found in the Jacobi convergence test (blocks warm-start)

`gpu_tiled_jacobi.cl`: `off0` = initial off-diagonal norm, exit test is
`if (off_cur / off0 < JACOBI_TOL) break;` with `JACOBI_TOL = 1e-9f`.
- **1e-9 relative is unreachable in f32** (~1e-7 best) → this break NEVER
  fires; the only working exit is the stagnation heuristic, which by
  construction wastes ≥3 sweeps proving it has stalled.
- Worse: it is **relative to the INITIAL off-norm**. A warm-started
  (already-nearly-diagonal) input has a tiny `off0`, making the test
  *harder*, so a naive warm start would burn all 100 sweeps and get slower.
  **Must be changed to an absolute criterion** (e.g. `off < eps·‖diag‖`)
  before warm-start can work. This is the trap that would eat the next
  session.

`V` is reinitialised to the **identity at the top of every call**
(`gpu_tiled_jacobi.cl` ~line 130) → every SCC iteration re-diagonalises from
scratch and throws away eigenvectors that are already ~correct.

### Fix plan, ranked by measured payoff

1. **Kill the redundant eigensolves (≈2–3× for free, hours of work).**
   `eval()` after a converged `scc()` re-diagonalises the SAME H′ the last
   SCC iteration just solved. Cache C/ε/D and skip `finalize` when q has not
   changed; drop the `eval(false)`-for-logging in `relax()` (or log every
   N steps / reuse the SCC state). 8 Jacobi/step → 6.
2. **Warm-start the Jacobi (≈5–10×, the big algorithmic win).**
   In SCC, H′ changes little between iterations. Feed `A = Cᵀ_prev H′ C_prev`
   (nearly diagonal) and `V = C_prev` instead of `A = H′, V = I` → 1–2 sweeps
   instead of 8. **Requires the absolute-tolerance fix above first.**
   Same trick across FIRE steps (geometry moves little).
3. **Replace the eigensolver for the occupied subspace (≈50–100×, the real
   answer).** The density needs only the occupied invariant subspace
   (N_occ=43 of 86; with smearing, occupied + a few kT window) — a FULL
   eigendecomposition is overkill. Warm-started **subspace iteration /
   LOBPCG on ~48 vectors**, seeded from the previous C, converges in 1–2
   iterations and is **all GEMM** — which runs across the whole GPU at
   TFLOP rates instead of 256 threads on one SM. This removes the
   one-WG-per-system straitjacket entirely.
   Alternative for N≤128: tridiagonalise + implicit QL (far better constants
   than cyclic Jacobi), or just call **cuSOLVER `syevjBatched`** and measure
   what we are competing against before writing more kernels.
4. **Revisit `JACOBI_PREC`.** prec1 costs 1.77× prec0 (83 vs 47 ms) for f64
   rotation scalars — on a consumer RTX 3090 FP64 is **1/64** of FP32
   (556 GFLOP/s vs 35.6 TFLOP/s). prec0's accuracy loss (eig parity 1.6e-3)
   is probably a *formulation* problem, not an f32 floor: test the stable f32
   rotation (`t = sign(θ)/(|θ|+√(θ²+1))`, `c = 1/√(1+t²)`, `s = tc`, plus
   Rutishauser's update form). If stable-f32 reaches prec1 accuracy → another
   1.8× on top.
5. **Batch ≥ 82 to use the GPU at all** (with the current one-WG-per-system
   design). At batch=19 we occupy 23 % of the SMs; per-system cost falls
   74440 → 1016 µs going from batch 1 → 256. For a 19-point scan this design
   is the worst case. Fix #3 makes this moot.

### Secondary issues noticed while reading the code

- **DRY violation / trap:** `TILED_MAX_SWEEPS` and `tiled_render_source`
  exist **twice** — `gpu_eigen.rs` (unused by the SCC path) and
  `gpu_scc_plan.rs` (the production one). Editing the former does nothing.
  Same for the `jacobi_cyclic_local` render helpers. Consolidate.
- `tiled_jacobi_batched()` in `gpu_eigen.rs` calls `rt.build_program()` on
  **every call** (JIT compile in what looks like a hot API). The SCC path
  does not use it (kernels are prebuilt in `GpuSccPlan::new` — verified), but
  it is a loaded gun for the next caller.

### ACCURACY / PHYSICS concerns in my own smearing patch (honest flags)

1. **Missing electronic entropy −TS (Mermin free energy).** With fractional
   occupations the variational functional is `F = E − TS`,
   `S = −2k_B Σ[f ln f + (1−f)ln(1−f)]`, and the force is `−dF/dR`. I
   implemented `E_band = 2Σ f_k ε_k` with **no −TS term**. At kT=0.002 Ha
   with 1–2 orbitals half-occupied, `TS ~ 0.002 × 1.4 ≈ 3 mHa ≈ 1.8
   kcal/mol` — **larger than the 0.1–0.4 kcal/mol features in the computed
   scan profile**, and it varies along the scan (zero where the gap is open,
   maximal at mid-transfer where occupations go fractional). So the reported
   E(r) profile is contaminated exactly where smearing activates, and
   energy/force are mutually inconsistent. **The barrier shape must not be
   trusted until −TS is added.** This also plausibly explains FIRE stalling
   at max|F|~1e-2 (hypothesis #7): forces are the gradient of a functional
   we are not evaluating.
2. **Smearing bypasses the D3/D4 accuracy fix.** `energy_from_state` uses the
   raw Jacobi diagonal `eig_diag` in the smeared branch, while the
   integer-occupation branch uses the Rayleigh quotients `eig_rho` when
   `occ_repair` is on. D1 measured the ε-vs-ρ error at 2.75e-5 Ha — so
   turning on smearing silently re-introduces it. `occ_rayleigh_batched`
   only fills ρ for mask-selected columns; it needs extending to all
   columns carrying weight (or at least the fractional window).
3. **Smearing changes the physics, not just the numerics.** kT=0.002 Ha
   ≈ 630 K is a real thermal broadening. Fine for robustness/screening (user
   approved: "few meV" tolerance), but the converged state is the
   finite-temperature one — must be stated when comparing to 0 K CPU
   reference (the CPU d=1.9 result E=−46.4698 is integer-occupation).
4. **CPU-vs-GPU discrepancy still unexplained.** CPU f64 converged d=1.9 to
   **E=−46.4698 Ha**; GPU smeared gives **−44.8191 Ha**. That is **1.65 Ha**
   — nowhere near a smearing or f32 effect. Either the two runs are not the
   same geometry/system (the CPU script rebuilt the molecule via `make_geom`
   with a hand-typed species list — 29 atoms, and note the first attempt had
   30 species, so the element assignment is suspect) or something is
   genuinely wrong. **This must be resolved before ANY scan number is
   believed** — it dwarfs every other accuracy issue in this document.
   (Native GC: CPU −47.3162 vs GPU ~−44.92 — same ~2.4 Ha gap, so it is a
   systematic definition difference, most likely the repulsive/reference-
   energy convention, not a solver bug. Check E_rep and the q0·V term.)

### Status correction

`gpu_bench` "0.41 ms/SCC iteration (formic N=28)" is **not** representative:
N=28 uses the `jacobi_cyclic_local` path (N≤64, full-local, fast), NOT the
tiled path. The tiled N>64 path is ~200× slower per iteration. Any
performance claim must state which Jacobi path it exercised.

## 2026-09-12 (even later) — ⚠ THE CPU REFERENCE WAS THE WRONG MOLECULE — root-cause REWRITTEN

### The bug

`diag_cpu_d19.rhai` hand-typed the GC species list for `make_geom`. The
typed list has an extra `H` at index 16 (positions 17-25 all shifted:
C,C,N,C,N,C,O,N → H,C,C,N,C,N,C,O,N and one fewer trailing H). Same atom
count (29), same electron count — so `make_geom` accepted it silently and
the run *looked* fine. Every CPU number quoted for "d=1.9" in this
document (E=−46.4698 Ha, 85-iter convergence, gap=0.45 mHa) was computed
on a **different, accidentally-easier molecule**.

Fix applied: new Rhai binding `get_species(name)` returns the loaded
element list; the script now builds the CSV from the xyz file. Never
hand-type species lists.

### What the CORRECT molecule does at d=1.9 (measured this session)

- **CPU f64 `build_scc` (run_dftb_scc): DIVERGES.** RMS limit-cycles
  1.4e-2 ↔ 2.3e-1 for all 300 iterations, never converges. Period-2
  charge-sloshing signature.
- **CPU f64 `DftbCpu::solve_scc` DIIS (cpu_ref): also FAILS** —
  RMS=2.19e-1 after 100 iters (panicked inside `gpu_cpu_energy`).
- **H0 (non-SCC) HOMO–LUMO gap at d=1.9 = 0.148 Ha** — NOT small. The
  "0.45 mHa gap → integer-occupation flip" story was measured on the
  wrong molecule and is RETRACTED.
- **GPU + Fermi smearing kT=0.002: plateau @ rms 3.4e-6 in 37 iters**
  (unchanged — the smearing fix is real and still the right tool).

### Revised root cause

The d=1.9 instability is a **genuine SCC fixed-point limit cycle**
(charge sloshing across the proton-transfer coordinate), present in f64
on the CPU with both a simple mixer and DIIS. It is **not an f32 noise
effect, not a batching bug, and not an integer-occupation flip**. Fermi
smearing fixes it because it contracts the response map
(dq_out/dq_in) — the correct physics remedy, consistent with the
standing f32 rule: the stabilizer is physical (finite-T ensemble), not a
numerical patch.

Consequence for parity: **there is NO converged integer-occupation
reference at d=1.9** — CPU or GPU. The only honest reference is a
smeared f64 CPU SCC; `DftbCpu::solve_scc` has no temperature support
yet → parity at d=1.9 is UNVERIFIED until that exists.

### The "1.65 Ha GPU-vs-CPU discrepancy" — resolved, two stacked causes

1. Wrong molecule (above).
2. `run_dftb_scc` returns electronic E WITHOUT repulsive energy;
   `GpuDftb`/`cpu_ref` report E_el + E_rep. For GC, E_rep ≈ +2.4 Ha.
   So CPU-native −47.316 (no E_rep) vs GPU −44.92 (with E_rep) is a
   convention gap, not a solver error — consistent with the measured
   GC |dE|=1.15e-6 GPU-vs-`cpu_ref` parity. **Any cross-path comparison
   must state which convention each side reports.**

### Code-read additions to the Jacobi fix plan (from this session)

- **The inner Brent–Luk pivot loop has NO early exit** —
  `INNER_SWEEPS=20` × `JROUND_INNER=63` rounds × ~3 barriers run
  unconditionally per block pair (~11k barriers/outer sweep ≈ the
  measured ~10 ms). A per-round "all pairs below skip-tol" flag
  reduction (32 flags → 1 barrier) would collapse a nearly-diagonal
  pivot to ~1-2 rounds. This is where the 10 ms/sweep lives, and it is
  INDEPENDENT of the outer warm-start fix — needed either way.
  `PAIR_SKIP_TOL=1e-12f` is also too tight to ever trigger; the skip
  should be relative to the local diagonal scale (~1e-7·√(a_pp·a_qq)),
  which is safe because Jacobi eigenvalue error from a skipped pair is
  quadratic (a_pq²/gap).
- **Warm start needs NO kernel signature change**: keep `V=I`; instead
  pre-rotate A ← C_prevᵀ·H_scc·C_prev (2 GEMMs — same count as the
  current XᵀH_sccX since C_prev=X·Cp_prev) and post-rotate
  C_new ← C_prev·V (1 GEMM). Ã is the Ritz projection in the previous
  eigenbasis — nearly diagonal → Jacobi exits in 1-2 sweeps *if* the
  outer test is made absolute (`off < tol·‖diag‖`; the current
  `off/off0<1e-9` can never fire in f32 and is *harder* on warm input).
  Caveat to verify: C_prev S-orthonormality drift (f32 ~1e-6) shifts the
  projected eigenproblem at O(δ); `eig_rho` Rayleigh quotients already
  correct eigenvalues, and a periodic/on-stall cold Jacobi re-pins the
  basis.
- Column ORDER permutes under warm start — safe here (occupation
  bisection, weighted density and EDM are all order-free), but anything
  that assumes sorted columns would break.

### Immediate next actions (not yet done)

1. Add Fermi smearing (kT + μ bisection + fractional Mulliken + −TS) to
   `DftbCpu::solve_scc` — needed for the d=1.9 parity check AND as the
   f64 reference for GPU smearing correctness.
2. GPU smearing honesty fixes still open: −TS Mermin term in
   `energy_from_state` (forces are −∇F; reporting E without −TS makes
   the scan profile inconsistent exactly where smearing activates), and
   `eig_rho` for ALL weighted columns (smearing currently re-introduces
   the D1 ε-vs-ρ error).
3. Jacobi: absolute exit test + inner-loop early exit + warm-start
   rotation (all three independent, in that order of safety).
4. relax(): cache last eigensystem, drop the eval-for-logging
   eigensolve.
