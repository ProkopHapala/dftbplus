# 2026-09-16 — Sparse DMM warm-density update for FD Hessians

**Scope:** reusing the converged central density matrix `K₀` for
finite-difference Hessian columns on `si_sphere_R10` (330 Si, 918 orbitals,
BSR4 deg≈330, RTX 3090, f32). Task: `tasts/Sparse_Nanocrystal_Vibrations/`
(G3 / "mode C"). Design discussion: `*.chat.md` (GPT-5.6 notes ~line 10605+).

## What was implemented

`scc_fixedq` warm path (`sparse_dftb.rs`) → `SparseSystemWorkspace::dmm_descend`
(`sparse_system.rs`):

```text
restore central (q,K,Z,K0) → warm NS (S⁻¹) → H_scc(q_frozen)
→ δK0 first-order seed  K ← K_c + (K0_new − K0_center)
→ DMM commutator descent (subspace rotation)
→ gate R_H ≤ 1e-4  (HARD error, no cold fallback)
→ forces
```

DMM step (collapsed 3-SpGEMM form, Z = S⁻¹, F = Z·H = `b_zh`, T = K·S):

```text
X = F·K          (plan_fk)
Y = T·X          (plan_tk_g — GENERIC plan, X is asymmetric)
δK = −η·(X + Xᵀ − 2·Y)     η = eta_scale/(εmax−εmin)
```

plus a planned McWeeny retraction every `ret` steps (all 4 products on
existing plans: Q·S rides plan_ks, U·K rides plan_tk).

## Bugs found and fixed (all measured, not assumed)

1. **Z is S⁻¹, not S⁻¹ᐟ².** The workspace Newton iteration `Z←2Z−ZSZ`
   converges to the inverse — host-f64 diagnostic: ‖ZS−I‖=6.5e-6 but
   ‖Z²S−I‖=18.5=‖S⁻¹−I‖. The original `(S⁻¹−K)HK` update was therefore an
   **ascent** direction (Tr(H·G)=−15.4, E_band rose, R_H 7e-4→0.10).
   Corrected to the generalized-overlap double commutator above —
   Tr(H·G)=+8.8e-4 ≥ 0, E_band descends monotonically.

2. **B-symmetric plan on an asymmetric operand.** `Y=T·X` ran through a
   bsym kernel that internally transposes the right operand → silently
   computed T·Xᵀ (dev-vs-f64 error 1.4e-2 while every other product was
   ~1e-6). Fixed with a generic symbolic plan `plan_tk_g` → error 8.4e-7.
   **Rule: any product whose right operand is asymmetric MUST use a
   generic plan.**

3. **The W1=T·A product was redundant.** Since SZ=I exactly,
   `KS·ZHZ = KHZ = Xᵀ` — free from `symmetrize_dev`. Removing it (plus
   the per-retract A rebuild) cut ~30% of DMM work — the reason the first
   working version was *slower* than cold.

4. **Post-DMM polish hurts stationarity.** Both McWeeny and TC2 are
   H-blind: they restore idempotency but drift the subspace (measured
   R_H 4.2e-5 → 7.5e-5 after 1 McWeeny). Post-polish now defaults off
   (`VIB_MCPOL=0`, `VIB_TC2MAX=0`); `measure_projector_state` reports
   honest r_I/Tr with 2 products and no K modification.

5. **No silent fallbacks anywhere.** R_H gate failure → `Err` with full
   context; the dummy-lane force gate (`dummy |D_ii| > 1e-6`) catches
   manifold drift — verified firing on ret=3 and unretracted runs.

## Measured results (R10, h=0.05 Å, per displaced eval = solve+forces)

| mode | products | ms/eval | R_H | ΔF vs cold fixq |
|------|---------:|--------:|----:|----------------|
| frozen (`VIB_FROZEN=1`) | 0 | **5.5** | — | 6.3% (CORRECTED — see update below; the earlier 2.4% was unreproducible) |
| warm DMM 6 steps η=8 ret=2 | ~26 | ~260–340 | 4.2–4.7e-5 | **0.30–0.35% Frobenius** |
| warm DMM 3 + 1 McW | ~20 | ~175 | 8.3e-5 | ~2.8% |
| cold fixq (reference) | ~50 | 330–355 | 8.0e-5 | — |
| full SCC | ~300 | ~1350 | — | reference-of-reference |

The 3-product DMM is **~25% faster than cold** with ΔF inside 0.4%.
Trajectory (η=8, ret=2): seed R_H=6.9e-4 → retracts 1.08e-4 → 7.4e-5
→ 4.7e-5; r_I≈3e-4, Tr(KS)=459.0001=Nocc.

## Why warm is not 10× — honest accounting

Per-eval product budget vs frozen (~10 ms): the electronic update is
~15–25 products of *mandatory* work:

- The seed lands at R_H≈7e-4, the force-validated gate is ~1e-4
  (measured: an undescended 2.9e-4 state gives ~85% force error), so
  ~7–15× residual contraction is needed.
- Steepest descent on the Grassmannian contracts ~1.7×/step at
  η≈8/Δε — the rate is set by the spectrum width, **not** by how good
  the seed is. 3–6 steps is the floor for this algorithm class.
- Some retraction is mandatory at usable η: unretracted runs drift to
  r_I≈2e-3 and trip the dummy-lane force gate (GPT-5.6's O(η²)-tangent
  claim only holds at small η).

## Open items (GPT-5.6 rebuttal, chat ~line 11600+)

The review argues the *true* Tier-1 should be ~7 products and we are
still over-solving — valid points, untested as specified:

- **Skip the δK0 seed + K0 build entirely**: seed = K_center, one
  first-order metric transport `K ← 2K − K·S₁·K` (shares T with the
  rotation) — needs center spectral bounds cached for η.
- **1 NS iter instead of 3** (warm Z first-order correction; R_Z=8.5e-5
  is ample for the direction).
- **1–2 DMM steps, no retraction**, benchmarked at the *real* h=0.02 Å
  (h=0.05 doubles the perturbation) — forces-not-R_H as the criterion.
- **Antisymmetric ±h sharing**: K(+h)−K₀ ≈ −(K(−h)−K₀) to O(h²) — one
  response solve per coordinate, not two. ~2× further.
- CG/BB on the manifold for the *certified* tier (halves the steps).

Target after these: ~5–8 products/eval → ~10× vs cold, as originally
intended.

## Configuration reference

| env | default | meaning |
|-----|---------|---------|
| `RUST_DFTB_VIB_FIXQ=1` | off | frozen-charge single-solve columns |
| `RUST_DFTB_VIB_DMUPD=1` | off | warm density update path (this work) |
| `RUST_DFTB_VIB_LINEAR=1` | off | stripped 4-product linear1 tier |
| `RUST_DFTB_VIB_LITE=1` | off | stripped DMM-lite tier (no gates/measurements) |
| `RUST_DFTB_VIB_SEED` | 1 | δK0 seed (0 = seedless, recommended) |
| `RUST_DFTB_VIB_NSMAX` | cfg | warm NS iter cap (lite default 0) |
| `RUST_DFTB_VIB_NSTOL` | cfg | warm NS residual tol |
| `RUST_DFTB_VIB_DMM` | 6 | DMM descent steps |
| `RUST_DFTB_VIB_DMM_ETA` | 8.0 | step scale η·Δε |
| `RUST_DFTB_VIB_DMM_RET` | 2 | retraction interval (0 = none — drifts) |
| `RUST_DFTB_VIB_MCPOL` | 0 | post-DMM McWeeny (degrades R_H — off) |
| `RUST_DFTB_VIB_TC2MAX` | 0 | post-DMM TC2 cap (0 = measure only) |
| `RUST_DFTB_VIB_GATES` | 1 | 0 = skip R_H certification (calibrated only) |
| `RUST_DFTB_VIB_RHGATE` | 1e-4 seeded / 5e-4 cold | stationarity gate, hard fail |
| `RUST_DFTB_VIB_DUMPCOL` | off | dump per-column ΔF vectors to dir |
| `RUST_DFTB_VIB_MAXCOL` | all | bound columns for measurement |
| `RUST_DFTB_DMM_VERIFY=1` | off | one-shot dense f64 product cross-check |

Harness: `rust_dftb/scripts/bench_vib_r10.rhai`.

---

## Update — stripped tiers + the frozen-orbital correction (same day, later)

The GPT-5.6 rebuttal was then **executed and measured** — and it
inverted the frozen-mode conclusion. Two findings:

### 1. The "stale b_zh bug" was the correct approximation, my fix was the regression

`forces_frozen` originally built `W = 2·b_zh·K` with `b_zh` stale from
the *central* `compute_k0` — i.e. `W = 2·(Z₀H₀)·K₀ = W₀`. That is the
**consistent clamped-electron (frozen-orbital) force**: the SCF
Lagrangian differentiated with BOTH the occupied projector AND its
Lagrange multipliers frozen (`δq=δK=δW=0`, only explicit geometry
dependence evaluated: `dH/dR`, `dS/dR`, `γ'(R)`, repulsion).

The "fix" — rebuilding `b_zh=Z(R)H(R)` while keeping `K₀` — produced the
inconsistent hybrid `W̃=Z(R)H(R)K₀`: it keeps `(δF)K₀` but drops
`F₀·δK`, i.e. **half of a cancelling response**. Measured column error:
**86%** vs ~6% for the consistent freeze.

Now deliberate: `snapshot_electronic_state` stores `W₀` explicitly;
`forces_frozen` = `compute_v` (γ(R)·Δq₀) + CPU pair contraction —
**zero device products per eval**, deterministic, order-independent.
**5.5 ms/eval, column error 6.3%** — h-independent (identical at h=0.02
and h=0.05). The earlier "2.4%" figure could not be reproduced under
any configuration — treat it as a stale/mismeasured claim; **6.3% is
the reliable number** (still ~63× cheaper than cold).

### 2. Stripped tiers: the certification was the hidden cost

GPT-5.6's product arithmetic (`ns0d2` should be 8 products ≈ 55 ms, not
~100 ms) exposed ~40 ms/eval of `measure_projector_state` +
`rh_stationarity` + host syncs inside the timed path. New env-gated
stripped modes skip ALL per-eval residuals: `VIB_LITE=1` (central Z +
N raw DMM steps) and `VIB_LINEAR=1` (single perturbative response on
the central metric: `X=B₁K₀−X₀=(δB)K₀`, `Y=P₀X`, `K←K₀−ηG` —
snapshotted `P₀=K₀S₀`/`X₀`, 4 products total). Lite sets `e_tot=NaN`
so `energy()` fails loudly; only `forces()` is served.

**Measured tier table (R10, h=0.02 Å, 3 cols, per displaced eval):**

| tier | products | ms | column err | env |
|------|---------:|---:|-----------:|-----|
| **clamped (K₀,W₀,q₀)** | **0** | **5.5** | 6.3% | `VIB_FROZEN=1` |
| linear1 (δB response) | 4 | 31 | 6.7% | `VIB_LINEAR=1` |
| DMM2-lite, central Z | 8 | 55 | 6.2% | `VIB_LITE=1 VIB_DMM=2` |
| **1 Newton + DMM2-lite** | 11 | **64** | **3.1%** | +`NSMAX=2 NSTOL=3e-5` |
| **1 Newton + DMM4-lite** | 17 | **105** | **1.0%** | `VIB_DMM=4` |
| gated warm (ns2+DMM4+gates) | ~22 | 175 | 1.0% | `VIB_DMUPD SEED=0` |
| cold fixed-q | ~50 | 345 | ref | `VIB_FIXQ=1` |

### 3. What the sweep taught (all measured)

- **Z accuracy is THE tier discriminator.** Central Z (R_Z≈2e-3 at the
  displaced S) caps every K-update at ~6% — the response direction
  `X=(ZH)K` inherits Z's error directly. ONE actual Newton correction
  (R_Z→1.4e-5, 3 products: check+update+check — "NS=2 iters" is 1
  update, not 2) unlocks the 1–3% tiers.
- **Retraction hurts.** Every McWeeny-retracted variant is worse than
  its unretracted twin (4.8% vs 3.0% at equal steps) — the map is
  H-blind. With the δK0 contaminant gone, ≤4 raw DMM steps stay
  on-manifold; 6 steps drift Tr(KS) over the gate (fails loudly).
- **Metric transport refuted:** `K←2K−KS₁K` alone gives **99%** column
  error — worse than frozen.
- **linear1 refuted at tested η:** 6.7% ≈ frozen's 6.3% but costs 6×
  more. A single fixed-η perturbative step is insufficient; the
  response needs ≥2 descent steps once Z is accurate.
- **Per-step K symmetrization is mandatory** (elementwise, free): the
  `Y=KHK` accumulate asymmetric truncation noise.
- **Seedless (`VIB_SEED=0`) is the default worth keeping** — the δK0
  seed is the first-order change of the spectral *initializer* (bounds
  + normalization included), not of the occupied projector.

### Final hierarchy and the real 10×

`clamped 5.5 ms/6.3% → ns2+DMM2 64 ms/3.1% → ns2+DMM4 105 ms/1.0% →
cold 345 ms`. Per-eval work is now within ~2× of the launch-bound
product floor (~7 ms/product); the remaining order-of-magnitude is
**batch-parallel ±h columns** (all 990 share topology/plans/masks).
Untested levers: ±h antisymmetric response sharing
(`K(−h)≈2K₀−K(+h)` — exact to O(h²), ~1.5× on the warm tiers),
Chebyshev/BB two-step η schedule, CG on the manifold.
