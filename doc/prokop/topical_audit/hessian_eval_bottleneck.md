# Sparse FD-Hessian per-eval bottleneck — measured breakdown

**Question:** why does a single displaced force evaluation (one ±h FD
Hessian column) on `si_sphere_R10` (330 Si, pbc-0-3, BSR4 sparse GPU
engine) cost ~3.5–7 s when the displaced Hamiltonian/density matrix are
almost identical to the minimum's — and what should change so that a
6N-column Hessian costs seconds-to-minutes, not hours?

All numbers: RTX 3090, `--release`, `RUST_DFTB_PROF=mark`,
`sparse_vibrations`-equivalent script `rust_dftb/scripts/bench_eval_r10.rhai`
(times each of `sparse_set_coords` / `sparse_scc` / `sparse_eval(forces)`
over 4 displaced evals, atom 0, ±0.02 Å).

## 1. Where the time actually goes (measured)

Per displaced eval, R10 (n_orb=918, n_occ=459):

| phase | deg~175 (r_k=12) | deg~330 (r_k=20) | what it does |
|-------|------------------|-------------------|---------------|
| `set_coords` | 4.7 ms | 4.5 ms | CPU: BSR H0/S assembly (3.4 ms) + dense f64 γ matrix O(N²) (0.9 ms) + repulsive (0.3) + H2D upload (0.3) |
| `scc` | **3.45 s** | **4.3–7.3 s** | NS (2 iters warm) + 9–17 DIIS mix iters × full TC2 purify each |
| `forces` | 6.5 ms | 7.3 ms | CPU pair contraction D/W (5.5 ms) + downloads |

**SCC is >98% of a displaced eval.** The CPU-side geometry work the
question worried about (H0/S build, γ O(N²)) is ~5 ms — 0.15% — at
N=330. It will only matter once SCC is fixed, and it only becomes a real
bottleneck at N ≳ 2000 (γ is dense O(N²): ~55k pairs now, ~2M at 2000
atoms → ~40 ms; still fine; port to GPU later, the dense solver already
has the kernels).

Inside `scc`, profiler stage table (host ms, mark mode; `tc2.tr` is the
blocking trace read whose span *includes draining* the just-enqueued
K·S·K — i.e. it is the true per-iteration wall cost):

- deg175: `tc2.tr` 87.4% (4209 calls × 3.50 ms) — ≈55 TC2 iters per
  purify × ~15 mix iters/eval. Purify never converges (mask floor
  R_I ≈ 6e-4) — it burns all ~55 iterations against the plateau.
- deg330: `tc2.tr` 95.0% (2525 calls × 11.75 ms) — ≈41 TC2 iters per
  purify × ~12 mix iters/eval. Purify converges to R_I≈3e-5 but the
  wider mask makes each K·S·K 4.4× more expensive.

So the per-eval cost is essentially:

```text
t_eval ≈ N_mix(9–17) × N_tc2(41–55) × t_iter(3.5 ms @deg175, 11.75 ms @deg330)
```

i.e. **~450–900 SpGEMM iterations per displaced force evaluation**.

## 2. The conceptual waste (what the code does vs what physics needs)

For a ±0.02 Å displacement the converged projector K changes by ~1e-3.
Yet per eval:

1. **Purify restarts from scratch every DIIS iteration.**
   `purify_hscc{,_trs,_p}` calls `compute_k0`/`compute_p0` =
   `P0/K0 = f(Z·H_scc)` (2 SpGEMMs + Gershgorin + axpby) then a full
   ~41–55-iteration TC2/TRS4 descent — for *every* mix iteration and
   *every* displacement. The stored converged K is never used as a seed.
2. **DIIS re-converges from charge noise.** 9–17 mix iters per displaced
   eval; the purifier's own residual floor (R_I≈6e-4 at deg175, 3e-5 at
   deg330) injects charge noise that DIIS has to fight. (cube65 at
   complete mask converges in ~5–8.)
3. **NS re-runs per geometry** — already warm-started, 2 iters, ~30 ms.
   Fine.
4. **Per-TC2-iter host sync.** `tc2.tr` = one blocking read per
   iteration that drains the speculative K·S·K — unavoidable serialization
   in the single-engine design; only batching removes it.

## 3. Experiments run (both measured, code in tree, env-gated)

### 3a. Frozen-density forces — `RUST_DFTB_VIB_FROZEN=1` ✅ works

`SparseDftb::forces_frozen()` (`sparse_dftb.rs`): after `set_coords`,
rebuild `V = γ·Δq` at the new geometry with the **minimum's charges**,
rebuild `H_scc` on device (one kernel), run the normal force contract
with the stored K. No NS, no purify, no mixing.

**Speed:** whole R10 Hessian (990 columns, 1980 evals) = **23 s**
(~12 ms/eval: 5 set_coords + 7 forces). This is the seconds-not-hours
regime.

**Accuracy — measured, si10h16 (26 atoms):**
rigid modes cleaner than SCC (−0.001 vs −16 cm⁻¹); framework modes match
full-SCC within **1–9 cm⁻¹**; but **Si–H stretch block comes out ~240
cm⁻¹ (≈10%) too soft** — freezing the density deletes the
charge-redistribution response, which is exactly what stiffens the
highest modes. On R10 the frozen spectrum looks sane (dense Si–Si band
to ~500 cm⁻¹ + surface modes + a ~2050 cm⁻¹ cap cluster) but is not
reference-validated.

**Verdict:** legitimate "fast preview" / framework-mode tool; NOT
quantitative for the modes people care about (X–H stretches, anything
charge-response-sensitive). It also answers "do we need SCC per
displacement": mostly yes for accuracy, but the *SCC state* doesn't need
re-purification from scratch — see 3b.

### 3b. Warm-started TC2 purify — `RUST_DFTB_WARM_K=1` ❌ refuted

`purify_hscc_warm()` (`sparse_system.rs`): keeps the stored converged K,
refreshes only `B = Z·H_scc` (needed by the force W build), skips K0
construction. Expected ~5 TC2 iters instead of ~50.

**Measured (si10h16, complete mask):** r_I **doubles every eval**
(1.8e-7 → 2.9e-7 → 5.8e-7 → … → 7.2e-5 over 10 FIRE/eval steps),
charges wander to a non-physical self-consistent fixed point
(E_scc=23 Ha, r_scc~1e-14 while r_I=1e-3), run garbage.

**Why:** same mechanism as the §15.12 frozen-operands diagnostic — the
*masked* TC2 fixed point is **repelling**: seeding the map near the true
projector walks away from it. The cold `K0` start is load-bearing (its
eigenvalue-ordering construction lands inside the attracting basin), not
wasteful. Do not warm-start this purifier. (If warm-starting is ever
revisited, the seed needs the same guarded entry as K0 — e.g. keep the
trace/branch machinery frozen from a converged state, or use a
sign-function/Newton-polish step instead of TC2.)

## 4. What would actually make the full-SCC Hessian fast

Ordered by expected payoff at N≈300–1000, honest:

1. **Batch-parallel ±h evals** — 1980 independent force evaluations with
   identical frozen topology/plans; the real product win. Per-eval work
   is already embarrassingly parallel; N engines on queues sharing
   geometry/masks. Serial → ~hours becomes minutes.
2. **Cut TC2 iterations inside each purify.** Today every purify runs
   ~41–55 iters regardless of state. `RUST_DFTB_TC2_STOP_W=28`
   (replay-validated plateau detector, off by default) already cuts
   ~15–33% on floor-churned runs with bit-identical energies. A smarter
   residual-aware early stop for the *last* mix iters could go further.
3. **Cut DIIS mix iters for displaced evals.** 9–17 iters per ±0.02 Å
   displacement is high — the DIIS subspace persists
   (`reset_iter_only`) but each mix iter still runs a full purify.
   Options: relax `scc_tol` for Hessian columns (1e-4? needs a
   force-noise study: FD column error ∝ σ_F/h), or a cheaper inner
   update (early mix iters could purify looser — classic DFTB+ trick).
4. **Frozen-DM Hessian** (3a) as an explicit "preview" mode — 400×
   cheaper per eval, wrong stretch physics. Good for rigid-mode sanity,
   framework bands, and CI smoke tests; must be labelled non-SCC.
5. **GPU γ/H0/S assembly** — port from the dense solver's existing
   kernels when (1) lands and N≳2000; today 5 ms vs 3.5 s is noise.
6. **Coupled-perturbed SCC (analytic Hessian)** — the principled
   endpoint: solve the linear response equations for dK/dR instead of
   re-solving nonlinear SCC 6N times. Bigger change; same sparse algebra
   (products with frozen masks). Would also fix the per-iter serial
   latency differently.

## 5. Headline numbers

| run | mask | per displaced eval | full Hessian (1980 evals) |
|-----|------|--------------------|---------------------------|
| R10 SCC, deg175 | nnz_k=57 736 | ~3.46 s (scc 3.45) | ~1.9 h (measured pace) |
| R10 SCC, deg330 | nnz_k=108 090 | ~4.3–7.3 s (scc) | ~2.5–4 h |
| R10 frozen-DM | deg330 | **12 ms** | **23 s (measured)** |
| si10h16 frozen-DM | complete | ~ms | ~3 s (measured) |
| cube65 SCC (old) | complete | ~0.46 s | ~4 min (measured) |

Code state: `forces_frozen()` + `RUST_DFTB_VIB_FROZEN` and
`purify_hscc_warm()` + `RUST_DFTB_WARM_K` are in the tree, env-gated,
off by default, documented as preview / refuted respectively.

---

## Update — mode ladder implemented (same day)

Following the GPT-5.6 review plan (§"What the Hessian workflow should
look like"):

**G1 central-state snapshot/restore** — `snapshot_electronic_state()`
stores `q, K, Z, K0` engine-side; `restore_central_state()` before every
±h eval + full DIIS reset. Identical solver history per column.

**Mode B (fixed-q) — validated:** `scc_fixedq()` = warm NS (2 iters) +
ONE purify, no DIIS (`RUST_DFTB_VIB_FIXQ`).

| mode | si10h16 vs SCC | R10 per eval | R10 Hessian |
|------|----------------|--------------|-------------|
| cold SCC (D) | reference | 3.4–7.3 s | ~2–4 h |
| fixed-q cold tol=1e-5 (B) | rms 6.1, max 10.7 cm⁻¹ | ~1.0 s | ~30 min |
| **fixed-q tol=5e-5 (B)** | **rms 6.3, max 10.9 cm⁻¹** | **~0.36 s** | **~12 min** |
| fixed-q + δK0 warm (C try) | **rms 324 cm⁻¹ — WRONG** | ~0.22 s | — |
| frozen DM (A) | rms 162, max 344 cm⁻¹ | ~12 ms | 23 s |

**Mode C attempt — refuted with a sharp diagnosis.** δK0 seed
`K_conv + (K0_new − K0_center)` lands at R_I~1e-3 (300× closer than cold
K0's 0.42 — the first-order rotation IS real) but raw TC2 repels it
(diverges). McWeeny polish (`K←3KSK−2KSKSK`, contracting) converges it
in ~7 iters — **to a wrong-subspace projector**: idempotent, right
trace, but R_H=‖HKS−SKH‖/2‖HKS‖=1.4e-3 vs cold ~4e-7, and the si10h16
spectrum is off ~300 cm⁻¹. Purification polynomials can only enforce
idempotency+trace — the occupied-subspace *selection* exists only in the
H-containing K0 basin. A correct warm update must instead minimize the
commutator ‖[K,H]‖ (LNV/DMM descent) — deferred. `rh_stationarity` gate
(default 5e-4) detects this and cold-restarts.

**Measurement policy** (`§4.12.0`): `RUST_DFTB_VIB_MAXCOL` bounds
columns + per-column phase timing — the 8-col R10 measurement took
**6–14 s**, extrapolates exactly.

**Headline now:** R10 Hessian ≈ 12 min at near-SCC accuracy via
`FIXQ=1 TC2TOL=5e-5`. Next big lever: batch ±h columns (parallel
engines) and/or a correct ‖[K,H]‖-minimizing warm update.
