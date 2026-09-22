# Sparse FD-Hessian per-eval bottleneck — measured breakdown

**Update 2026-09-22.** The batching lever this note ends on was measured.
Frozen columns reached 0.27 ms/eval and stopped moving the spectrum
(the host eigensolve is ~72% of the R18 wall). DMM-lite batching
saturated at ~1.24× because the SpGEMM is bandwidth-bound. Standing
order: [`Sparse_Performance.md`](../tasts/Sparse_Nanocrystal_Vibrations/Sparse_Performance.md).
The tables below remain the record of why a cold column cost seconds.

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

### 3a. Frozen-orbital (clamped-electron) forces — `RUST_DFTB_VIB_FROZEN=1` ✅ works

`SparseDftb::forces_frozen()` (`sparse_dftb.rs`): the ENTIRE electronic
state is frozen at the central snapshot — `D=D₀`, `W=W₀=2(Z₀H₀)K₀`,
`q=q₀`. Only the explicit geometry dependence is evaluated at the
displaced position: `V = γ(R)·Δq₀` (CPU) + the pair contraction with
current `dH/dR`, `dS/dR`, `γ'(R)`, repulsion. **Zero device products,
no NS, no purify, no mixing.** (Physics: the SCF Lagrangian
differentiated with orbitals AND Lagrange multipliers frozen — the
consistent freeze preserves the `δW=(δF)K₀+F₀δK` cancellation that the
inconsistent hybrid `W̃=Z(R)H(R)K₀` breaks; measured 6.3% vs 86% column
error. See the 2026-09-16 report.)

**Speed:** ~5.5 ms/eval electronic work (was ~12 ms when it still
rebuilt H_scc/W per eval) — R10 Hessian ≈ 15–20 s. This is the
seconds-not-hours regime.

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
charge-response-sensitive). On R10 Hessian columns the consistent
freeze measures **~6.3% column error** (h-independent, h=0.02 and 0.05
alike) — the screening tier. It also answers "do we need SCC per
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
| R10 frozen-orbital | deg330 | **5.5 ms** | ~15–20 s (extrap.) |
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
| frozen-orbital (A) | rms 162, max 344 cm⁻¹ | 5.5 ms | ~15–20 s |

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

---

## Update — mode C solved: DMM commutator warm update (2026-09-16)

The "correct ‖[K,H]‖-minimizing warm update" is now implemented and
validated: `dmm_descend` in `sparse_system.rs`, driven from the
`scc_fixedq` warm path under `RUST_DFTB_VIB_DMUPD=1`. Full report:
`reports/2026-09-16_sparse_dmm_warm_density_hessian.md`.

**Step (3 SpGEMMs, Z=S⁻¹, F=Z·H, T=K·S):**
`X=F·K; Y=T·X; δK=−η(X+Xᵀ−2Y)`, η=eta_scale/(εmax−εmin), plus a planned
McWeeny retraction every `ret` steps.

**Bugs found en route:** (i) workspace Z is **S⁻¹ not S⁻¹ᐟ²** — the
first update form was an ascent direction (Tr(H·G)<0, E_band rose);
(ii) a **bsym SpGEMM plan on asymmetric X** silently computed T·Xᵀ —
fixed with generic `plan_tk_g` (error 1.4e-2 → 8.4e-7); (iii) the
`T·ZHZ` term is exactly `Xᵀ` — removed, that's why the first correct
version was *slower* than cold. Post-DMM McWeeny/TC2 polish **raises**
R_H (H-blind) — defaults off; state measured without modification.

**Measured (R10, h=0.05 Å, solve+forces per eval):**

| mode | products | ms/eval | R_H | ΔF vs cold |
|------|---------:|--------:|----:|-----------|
| frozen-orbital (clamped K₀,W₀,q₀) | 0 | 5.5 | — | 6.3% (corrected below) |
| **DMM 6/η8/ret2** | ~26 | **260–340** | 4.2e-5 | **0.30%** |
| DMM 3 +1McW | ~20 | ~175 | 8.3e-5 | 2.8% |
| cold fixq | ~50 | 330–355 | 8.0e-5 | ref |

**Honest verdict:** ~25% faster than cold, certified forces — but NOT
the 10× the task wants. Steepest descent contracts ~1.7×/step (set by
η·Δε, not the seed quality), the seed lands at R_H≈7e-4 vs the ~1e-4
force-validated gate, and retraction is mandatory at usable η
(unretracted → r_I≈2e-3 → dummy-lane force gate fires).

---

## Update — stripped tiers measured; frozen story CORRECTED (2026-09-16, later)

The GPT-5.6 rebuttal was executed. Two inversions:

**1. The stale `b_zh` was the CORRECT approximation; fixing it was the
regression.** Stale `b_zh` made `W = 2(Z₀H₀)K₀ = W₀` — the consistent
**clamped-electron (frozen-orbital) freeze**: `δq=δK=δW=0`, the SCF
Lagrangian differentiated with orbitals AND multipliers frozen. The
"corrected" hybrid `W̃=Z(R)H(R)K₀` keeps `(δF)K₀` but drops `F₀δK` —
half a cancelling response → **86% error**. Now deliberate:
`snapshot_electronic_state` stores `W₀`; `forces_frozen` is pure CPU
(compute_v + pair contraction), **0 device products → 5.5 ms/eval,
6.3% column error** (h-independent). The old 2.4% claim was
unreproducible — 6.3% is the number.

**2. ~40% of the warm-eval cost was certification inside the timed
path.** `VIB_LITE=1` / `VIB_LINEAR=1` strip all per-eval residuals
(measure/R_H/syncs — validation only now). Also: the δK0 seed is gone
(`VIB_SEED=0` — it contaminates the manifold); retractions *hurt*
(H-blind McWeeny); metric transport `K←2K−KS₁K` refuted (99% error);
per-step K symmetrization added (free).

**Final measured hierarchy (R10, h=0.02 Å, per displaced eval):**

| tier | products | ms | col err | env |
|------|---------:|---:|---------:|-----|
| clamped (K₀,W₀,q₀) | 0 | 5.5 | 6.3% | `VIB_FROZEN` |
| linear1 | 4 | 31 | 6.7% | `VIB_LINEAR` — refuted |
| DMM2-lite, central Z | 8 | 55 | 6.2% | `VIB_LITE DMM=2` |
| **1 Newton + DMM2-lite** | 11 | **64** | **3.1%** | `+NSMAX=2 NSTOL=3e-5` |
| **1 Newton + DMM4-lite** | 17 | **105** | **1.0%** | `DMM=4` |
| gated warm (ns2+DMM4) | ~22 | 175 | 1.0% | `VIB_DMUPD SEED=0` |
| cold fixed-q | ~50 | 345 | ref | `VIB_FIXQ` |

**Z accuracy is the tier discriminator:** central Z caps everything at
~6% regardless of K-work; ONE Newton update (R_Z 2e-3→1.4e-5) unlocks
1–3%. Per-eval work is now ~2× off the launch-bound product floor —
the remaining 10× is **batch-parallel ±h columns**.
