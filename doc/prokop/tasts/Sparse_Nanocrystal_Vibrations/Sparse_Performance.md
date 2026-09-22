---
type: mandate
title: Sparse GPU DFTB — warm electronic solve is the goal
tags: [gpu, sparse, bsr4, fire, lbfgs, hessian, dmm, tc2, spgemm, performance, dftb]
timestamp: 2026-09-22
status: standing order for all further sparse-solver work
---

# Sparse GPU DFTB — the warm electronic solve is the goal

Read this before any other file in
[`Sparse_Nanocrystal_Vibrations/`](.) or before touching
`rust_dftb/src/methods/sparse/` (`sparse_dftb.rs`, `sparse_system.rs`,
`sparse_hs.cl`, `sparse_bsr4_purification.cl`).

Later notes inside the report and the chat override earlier ones on
mechanism. **This section overrides them on what to time.** A session
that speeds up anything else has missed the job.

## 0. What you must focus on

The expensive object is one electronic solve on the GPU at a geometry
that is only slightly different from a geometry you already solved:
rebuild `H` and `S` for the new coordinates, and update the density
`D` from the previous density. In the code the stored projector is
`K` (`D = 2K`). Every optimizer step and every finite-difference
sample pays that solve. That is the clock.

**Primary — geometry optimization.** Each FIRE or L-BFGS step must
reuse the state it already has,

```text
H(t), S(t), D(t)   →   initial guess for   H(t+dt), S(t+dt), D(t+dt)
```

A step that throws that state away and purifies from a cold `K0` is
the failure this work exists to remove. The number to report is
milliseconds per optimizer step, the product count inside that step,
and the force (or energy) error against a cold solve of the same
geometry. A handful of steps on a small crystal. Not a 200-step
relaxation of a 1648-atom sphere.

**Secondary — a few Hessian elements, as a measurement of that same
solve.** One finite-difference entry displaces one coordinate by `±h`
and solves the electronic problem again. The initial guess is the
optimized geometry,

```text
H0, S0, D0   →   initial guess for the displaced H, S, D
```

Time a few such displacements (a few matrix elements, a few degrees
of freedom). That is enough to measure how fast the sparse electronic
solver is and how much the guess is worth. Do **not** build the full
6N-column Hessian of a large nanocrystal. The full matrix is how a
short measurement turns into an hour, and it is not required to see
whether the warm solve works.

**Out of scope — diagonalizing the vibrational Hessian.** Once the
force derivatives exist, the dense 3N×3N frequency solve is a
post-process. Its cost is negligible next to the displaced electronic
solves (or it only looks large when those solves were skipped, as in
the frozen-orbital preview). Do not tune `nalgebra`, `dsyevd`,
Lanczos-on-the-Hessian, or the spectrum wall. `core/eigh.rs` already
calls `dsyevd`. Leave it.

**Out of scope — the frozen-orbital eval as a thing to make faster.**
`VIB_FROZEN` does zero electronic products. It is a labelled preview.
It is not the solver whose reuse we are measuring. The 0.27 ms batched
frozen number and the 121× figure are real, and they are the wrong
clock.

The dense multi-system mandate is a different algorithm
([`Dense_Multi_Performance.md`](../HBond_Relaxed_Scan_GPU/Dense_Multi_Performance.md)).
Section 6 says what transfers. The 100× bar below is on this warm
electronic solve versus one CPU thread of the same recipe, not on a
full spectrum and not on a frozen column.

## 0.1 Priorities (code review 2026-09-22)

**2026-09-22 evening — the geometry step does minimize.**
`GeomStep::BoldXtrDmm` (two `η = 8` commutator steps, one McWeeny,
keep the kernel, charge mix `α = 0.2`) relaxes Si₁₉₆H₁₃₄ from a
0.1 Å/atom kick: energy −311.01 → −314.10 Ha, Si–H 1.27–1.46 Å →
1.49 Å, `Tr(KS)` stays on 459, warm step ~180 ms against a 1.9 s
cold SCC. `max|F|` saturates near 2×10⁻⁴ (best 1.80×10⁻⁴, then 40
steps with no lower value). With the per-step certificate off, the same
kick finishes in one process: 497 steps, median 35 ms (was ~175 ms),
energy −311.01 → −314.10 Ha, Si–H 1.49 Å, floor 2.2×10⁻⁴.
Setup of the masks is 6 s at 330 atoms
and 27 s at 864 atoms; that cost is once per process, and it scales
worse than the step. Full table, the profile, and the file paths:
[`Warm_Geometry_DM.md`](Warm_Geometry_DM.md) §9. The R14 descent
(864 atoms, same script) is in progress there: 278 steps, energy
−880.4 → −889.6 Ha, force still moving near 10⁻³. The paragraphs below are the
record of the recipe that failed (four unretracted commutator steps,
trace rescaling). Do not go back to that recipe. The measured
minimizer is §9.

A FIRE step has to minimize the DFTB energy and keep `Tr(KS) = Nocc`.
Cold TC2 does both. Four commutator steps do not, once they are
chained: on Si₁₉₆H₁₃₄ the raw trace climbed by ~0.6 e⁻ in 24 steps,
scaling it back moved the band energy by tenths of a hartree, and
the plotted energy rose while the force fell. DMM4 stays a one-shot
force proxy (R10, 105 ms, 1% column error at frozen `q`). It is not
the relaxation. The working minimizer is cold purification from `K0`,
with the previous charges kept. A cheap step may be kept only when
that step's own raw `|ΔTr|` stays a small fraction of an electron;
past that, discard `K` and cold-restart. Do not McWeeny-clean a
leaked `K`, and do not scale it and continue. The variants that
implement that restart — AO extrapolation, a trust-region
commutator, one McWeeny, cold TC2 on certificate failure — are
specified in [`Warm_Geometry_DM.md`](Warm_Geometry_DM.md) §7
(SiH₄, 2026-09-22: the certificate and the cold restart match the
reference; a warm accept happened only at `|ΔR| ≈ 0.01` Å). Item 1
below is the fixed four-step recipe that was wired and failed as a
relaxation. Do not extend it.

### What the geometry step actually runs

`relax` is `scc` + a loop of `fire_step` + `scc`
(`sparse_dftb.rs`). `fire_step` builds forces, moves the atoms, and
calls `set_coords`. `set_coords` rebuilds `H` and `S` on the device
and clears `z_valid`. It does **not** clear `K`. The previous density
stays in the device buffer, and `z_warm` stays true, so the next
`scc` Newton-corrects `S⁻¹` from the old inverse.

On a geometry change with a stored `K` (and not the experimental TRS
or P purifiers), `scc_inner` now takes `scc_warm_geometry` instead of
`purify_hscc`. Same-geometry re-entry and the first SCF still run
TC2. `RUST_DFTB_WARM_K` remains the refuted warm-TC2 study flag and
is not this path.

`dmm_descend` is that update. It is still what `scc_fixedq` runs
under `VIB_LITE` / `VIB_DMUPD` (frozen `q`, the finite-difference
sample). `relax` reaches it through `scc` → `scc_warm_geometry`.
A second charge-mix block on the same step was measured to leave the
manifold; the geometry step is one block, and the new Mulliken `q`
is what the next step starts from.

### Order

1. **Put the known recipe on the step.** One warm Newton correction
   of `Z`, then `refresh_b_zh` + four commutator steps (the R10
   ladder: 17 products, 105 ms, 1.0% column error at frozen `q`).
   For FIRE/L-BFGS, Mulliken the updated `K` and carry `q_out`. A
   second commutator block on the same step leaves the manifold
   (measured below). For a finite-difference element, frozen `q` is the
   right first measurement and the code path already exists
   (`VIB_LITE`, `VIB_NSMAX`). Do this before any kernel edit. The
   expected gain against today's FIRE step is the ratio of ~50 TC2
   iterations × several mixes to ~15 products, which is the factor
   of ten already seen going from cold SCC to the lite column. It
   has not been timed as an optimizer step on R10.

   **Wired 2026-09-22, measured on SiH4 only** (complete mask, debug
   build, verbose DMM prints on — the milliseconds include those
   syncs). `scc_warm_geometry`: cached `(emin, emax)` from the last
   cold K0, NS capped at 4 iterations / `R_Z=1e-4`, then exactly four
   commutator steps at η scale 8, no McWeeny, no second block.
   `q_out` is carried so the next step's `H` is not built from stale
   charges. A second block was tried and rejected: `|G|` grew and
   `Tr(KS)` ran away. After the four steps, `K` is scaled so the
   remeasured `Tr(KS)` is `Nocc` (same restore as the TC2 trace guard).
   A single step that moves the trace by more than one electron fails
   instead of being scaled. `R_H` still has to pass the stationarity gate.

   One 0.02 Å step versus a cold SCC of that same geometry:
   `|ΔE|=5.2e-3` Ha, `max|ΔF|=1.0e-3` against cold `max|F|=8.8e-2`
   (~1% of the largest component), `R_H=3.6e-4`, charge rms `3.0e-3`,
   ~0.6 ms versus ~12 ms for the cold GPU SCC. Three FIRE steps from
   a converged geometry stay at `Tr≈3.999` and `R_H<1e-4`, ~0.9 ms
   each. Several 0.02 Å-scale updates chained on the same `K` double
   the trace error; the relative gate fails loud and does not fall
   back to TC2. This is not the R10 105 ms / 1% column number.

2. **SpGEMM bandwidth, after that recipe is the one on the clock.**
   A planned product is ~2 FLOP/B: one workgroup owns an output
   block-row, and each term is a scattered 64-byte read of `B`
   (`sparse_bsr4_purification.cl`, `bsr4_spgemm_plan`). Replica
   batching of this chain already saturated at ~1.2× (§15.28). The
   untried change is to reorder plan terms so one fetched `B` block
   is reused across several accumulations. One variant, same recipe,
   a few displacements. If milliseconds per product do not move, the
   reorder missed the stream.

3. **Harness that is still on this path, and is small next to a cold
   purify.** Worth doing once a step is ~100 ms, because then these
   are a visible fraction. They are not a reason to delay item 1.
   - `set_coords` walks every H/S pair and every repulsive pair on
     the host, every geometry, before launching the GPU assembly
     (`sparse_dftb.rs`, the coincident-atom checks). The pair list
     is frozen with the topology. The check belongs at init, or in
     one device reduction.
   - `pair_forces_dev` reads five force buffers back
     (non-SCC, shift, repulsive, double-counting, total). The
     optimizer needs one 3N vector.
   - `rep_eval_dev` reads every pair repulsive energy back on every
     `set_coords`. A force step does not need that scalar until it
     prints an energy.
   - `scc_inner` parses `RUST_DFTB_TRS`, `RUST_DFTB_P_TC2`, and
     `RUST_DFTB_WARM_K` inside the mix loop.
   - `fire_step` clones the coordinate array and `set_coords` copies
     it again.
   - `finalize_scc` always runs `rh_stationarity` (extra products
     and a host reduction) even when the step is a calibrated
     recipe. One check per several steps, not every step.

4. **A shorter step length, only if item 1's forces are already in
   the 1% class.** Untried: a two-step Barzilai–Borwein / Chebyshev
   schedule on the same commutator (chat ~13790), which might cut
   four identical steepest-descent steps to two. Conjugate gradient
   on the manifold was proposed and explicitly deferred until one or
   two response steps had been shown to be not enough. Do not start
   with CG.

### Already tried — do not reopen

| idea | result |
|---|---|
| Warm-start TC2 from the stored `K` | Repelling on the masked map. Diverges. |
| `δK0` seed + McWeeny | Idempotent, wrong subspace. Spectrum off ~300 cm⁻¹. |
| Metric transport `K←2K−KS₁K` alone | 99% column error. |
| One linear response, no Newton on `Z` | Same ~6% error as frozen orbitals, at 6× the cost. |
| McWeeny retraction between DMM steps | Worse column error than the unretracted twin. |
| Float-float polish on a truncated mask | Representation floor. No gain. |
| More DMM replicas | Saturated. ~1.2× at R18. |
| Early-stop TC2 (`TC2_STOP_W=28`) | Real ~15% on a cold purify that has already hit the floor, off by default. Useful only for the **first** SCF, or a cold fallback. Not the per-step algorithm. |
| Hessian diagonalization, frozen-orbital kernel, full 6N columns | Wrong clock. §0. |

### Useful only after the one-sided step is accepted

- Antisymmetric `±h` sharing (`K(−h)≈2K₀−K(+h)`). About 2× on
  finite-difference elements. Geometry optimization has no `±h` pair.
- Fixed-iteration cold TC2 batched across many columns (manifest
  F5b). That is the from-scratch purify, parallelized. It does not
  reuse `D`.
- Column-local pair lists (F2). They shrink the force contraction.
  The contraction is not the purification. Profile a warm step
  before writing that indexing.

## 1. The goal

Performance is the primary goal. Accuracy is a constraint, and the
constraint is usefulness:

- A preview spectrum may be the clamped-electron (frozen-orbital)
  Hessian. It must be labelled as that. Framework bands can be right
  while Si–H stretches are ~10% soft.
- A quantitative spectrum must land in the class already measured for
  fixed-charge columns on si10h16: rms ~6 cm⁻¹ against full SCC, Si–H
  within ~10 cm⁻¹. Qualitative failures (inverted stretches, a
  mid-band collapsed into one blob, a wrong-subspace projector) are
  failures.
- Matching a dense f64 eigensolver to 1e-7 Ha, idempotency to 1e-6, or
  a commutator certificate on every displaced geometry is the schedule
  that made one R10 column cost seconds. Those targets are how the
  GPU path loses.

**Hard bar.** A GPU result that is not at least **100× a single CPU
thread** on the production workload is not a GPU solver anyone will
run. The f32 compromises are only justified by that speed. The
denominator is one thread of the explicit CPU reference
(`RUST_DFTB_SPARSE_CPU=1`), same recipe, same system. Device for every
number below: NVIDIA RTX 3090. Dates are the measurement dates in the
source notes.

The number that has to move is the warm electronic solve, not a
frozen column and not a frequency diagonalization:

| solve | what it is | where it stands |
|---|---|---|
| **Cold displaced SCC** | new geometry, purify from `K0`, full mix | R10 **3.5–7 s/eval**, >98% inside purification. This is the cost a warm guess has to beat. |
| **Warm density update** | previous `D` (and `S⁻¹`) carried onto the displaced geometry | R10 ladder: 1 Newton + DMM4 is **105 ms, 1.0% column error** vs cold fixed-charge. R18 central-Z DMM4 batch is **208 ms/eval** and is the ~6% tier, because that run skipped the Newton correction of `Z`. |
| **Frozen orbitals** | `D` not updated | 0.27 ms batched at R18. Skips the solve. Wrong clock. |

Replica-batching of the DMM chain saturated at ~1.2× (report §15.28).
The remaining lever on this solve is fewer products for the same
force, and a faster SpGEMM for the products that remain. Both are
measured on a few displacements or a few optimizer steps.

## 2. Where the time actually went

A displaced force evaluation used to be a cold SCC. On R10 (330 Si,
918 orbitals) that was **3.5–7 s/eval, >98% inside purification**:
9–17 DIIS iterations × 41–55 TC2 steps × one SpGEMM of 3.5–12 ms.
About 450–900 products per column. A 1980-eval Hessian was hours.
(`hessian_eval_bottleneck.md` §1.)

That number is the cost of restarting the polynomial from `K0` on
every mix iteration, then reading a 4-byte trace back to the host to
pick the branch. It is also the cost of running the polynomial after
it has stopped moving. The 1648-atom divergence (report, 2026-09-12)
was the same loop with no restoring invariant: once an eigenvalue of
`KS` left `[0,1]`, both TC2 branches double the error every step.
The standing rule from that day still holds. Stop at the measured
floor. An open-loop iteration is a design bug.

What was then removed, in the order the later notes settled:

| change | eval time | what it proved |
|---|---|---|
| Fixed-charge, one cold purify (`VIB_FIXQ`) | R10 ~345 ms; R18 ~916 ms (2026-09-17, pair physics still on the host) | Dropping DIIS is the first factor of ten. Frequencies on si10h16: **rms 5.9 cm⁻¹** vs full SCC. This is the quantitative reference tier. |
| Clamped electron (`VIB_FROZEN`) | 5.5 ms on the host pair path; **1.6 ms** after device residency; **0.27 ms** batched | Zero products. Column error **6.3%**, h-independent. Spectrum rms **157 cm⁻¹**, Si–H **−250 cm⁻¹**. Preview only. |
| GPU pair path, then residency (report §15.24–§15.26) | R18 frozen 203 ms CPU → 10.8 ms on device → **1.6 ms** once ~155 MB/eval of PCIe disappeared | The pair kernels were never the 10 ms. A frozen eval is ~0.5–1 GFLOP, ~30 µs at peak. The gap was copies. |
| Frozen replica batch (F1, §15.27) | 1.6 → **0.27 ms** at B=16, then flat through B=32 | Overhead gone. Bit-identical forces. Full R18 Hessian wall **stayed 138 s**, because the columns fell from ~16 s to ~2.7 s and the host `SymmetricEigen` of the 4944×4944 Hessian stayed ~100 s. |
| DMM-lite replica batch (F5a, §15.28) | R10 108 → 53 ms (~2×); R18 257 → **208 ms (~1.24×)**, flat by B=8 | SpGEMM is already bandwidth-saturated at B=1. Batching removed host overhead. It did not add bandwidth. |

The frozen R18 spectrum wall (138 s, of which ~100 s was the host
eigh and ~3 s was the GPU columns) is a measurement of a path that
skips the electronic solve. It is recorded in report §15.26–§15.27.
It is not a reason to touch the frequency diagonalization. When each
column actually solves the Hamiltonian, that solve dominates and the
eigh is noise — which is why a full 6N Hessian is the wrong benchmark.

**Warm-update cost, so the extrapolation stays honest.** 9888 × 208 ms ≈ **34 min** of DMM-lite
columns, and that 208 ms is `n_ns=0, n_dmm=4` (central `Z`, four
commutator steps). The September 16 ladder, same code family, says
central `Z` (R_Z ≈ 2×10⁻³ on the displaced overlap) **caps every
K-update near 6% column error**. One Newton correction of `Z`
(R_Z → 1.4×10⁻⁵) is what unlocked 3.1% (DMM2) and **1.0%** (DMM4) on
R10, at 64 ms and 105 ms scalar. The fast R18 batch number and the
1% recipe are not the same run. Shipping 208 ms as "the quantitative
Hessian" ships a frozen-class force error at two orders of magnitude
more time than the frozen eval.

## 3. Why the long purifier cannot be the column

TC2 (and TRS4, and McWeeny) is a polynomial in `K`. It can restore
idempotency and the trace. It cannot rotate the occupied subspace.
That was measured three times, and the later measurement is the one
to keep:

1. **Warm-started TC2** (`VIB_WARM_K`). The masked map
   `K′ = P_M((K·S)·K)` has **no stable fixed point** at the truncated
   exact projector. Host-f64 TC2 walks off `K_ref|M_K` on step 0, to
   the same ~10⁻³ limit cycle as f32 (report §15.12-2′). Seeding a
   converged `K` makes `R_I` double every eval until the energy is
   garbage (E = 23 Ha at `r_scc` ~ 10⁻¹⁴). The cold `K0` start is
   load-bearing because it lands in the attracting basin. Do not
   warm-start this polynomial.

2. **δK0 seed + McWeeny.** The seed is a real first-order rotation
   (`R_H` ~ 10⁻³, 300× closer than cold `K0`). McWeeny then converges
   it to an idempotent, right-trace projector in the **wrong
   subspace** (`R_H` ~ 10⁻³, si10h16 spectrum off ~300 cm⁻¹). A later
   `fixq+dmupd` spectrum was **rms 315 cm⁻¹**. Purification is
   H-blind. Post-step McWeeny raises `R_H`. Retractions on the DMM
   ladder were worse than the unretracted twin (4.8% vs 3.0% column
   error).

3. **The "f32 floor" was stored-`K` truncation.** At a complete mask,
   f32 TC2 reaches tens of µHa and preserves an injected exact `K`.
   At a production radial mask the missing tail is required by the
   map itself; an intermediate-product halo recovers 0.075% and was
   deprioritized. Float-float McWeeny (report §15.14–§15.15) buys
   ~15× residual on a full mask and **nothing on `r_k=20`**, where the
   plateau is the representation. FF32 does not belong on the Hessian
   hot path. Widening `M_K` is the accuracy knob, and it is paid in
   SpGEMM bytes.

The pattern to refuse next time: a 30- or 55-step cap, a residual
gate inside the timed column, a host read per product, and a cold
re-purify when the gate misses. That is how a 0.02 Å displacement,
whose projector moves by ~10⁻³, was turned back into a from-scratch
solve. Sparse already measured the tax: certification
(`measure_projector_state`, `rh_stationarity`, trace reads) was
**~40%** of the warm eval. The chat's production rule (line ~13866)
is the one to keep: fixed arithmetic, then the force. The certificate
is a validation benchmark, run on a sample of columns, outside the
clock.

## 4. The iterative update that actually moved the column

The update that selects the subspace is steepest descent on the
commutator, not a polynomial. `Z` in this workspace is `S⁻¹`, not
`S⁻¹ᐟ²`. The step that descends is

```text
X = (Z·H)·K
Y = (K·S)·X          generic plan — X is asymmetric
δK = −η (X + Xᵀ − 2Y)
η = eta_scale / (εmax − εmin)
```

three SpGEMMs, then an elementwise symmetrization. The first coding
of this was an ascent (Tr(H·G) = −15), because it assumed the Löwdin
square root. The second silently computed `T·Xᵀ` (error 10⁻²) by
putting an asymmetric operand on a bsym plan. Both are in
`reports/2026-09-16_sparse_dmm_warm_density_hessian.md`. Any new
product whose right operand is asymmetric uses a generic plan.

**R10 ladder, h = 0.02 Å, column error vs cold fixq** (the later
stripped-tier table; the earlier "2.4% frozen" figure does not
reproduce):

| tier | products | ms | column error | keep? |
|---|---:|---:|---:|---|
| clamped `K₀,W₀,q₀` | 0 | 5.5 (host pair; 0.27 batched after residency) | 6.3% | preview spectrum only |
| linear1, one fixed-η response | 4 | 31 | 6.7% | no — frozen's error at 6× the cost |
| metric transport `K←2K−KS₁K` | — | — | 99% | no |
| DMM2, central Z | 8 | 55 | 6.2% | no — Z error is the whole answer |
| **1 Newton + DMM2** | 11 | 64 | **3.1%** | candidate cheap quantitative |
| **1 Newton + DMM4** | 17 | 105 | **1.0%** | candidate quantitative |
| same, plus per-eval gates | ~22 | 175 | 1.0% | validation only |
| cold fixq | ~50 | 345 | reference | the chemistry anchor (rms 5.9 cm⁻¹) |

`Z` accuracy is the tier discriminator. One Newton update is one
actual correction plus residual checks, three products, and it is the
difference between a 6% column and a 1% column. DMM2 with a stale `Z`
does not become accurate by adding more commutator steps. The rate of
steepest descent itself is set by `η·Δε` (~1.7× per step), so a better
seed does not turn this into one product. CG and Barzilai–Borwein were
proposed and not measured; they are allowed only after the 1-Newton
recipe has a frequency table.

**The missing number.** Column Frobenius error is not a spectrum.
`fixq+dmupd` at rms 315 cm⁻¹ (report §15.18) is the seeded, gated
path, and it is a wrong-subspace result. The seedless 1-Newton+DMM4
recipe has a 1.0% force-column measurement on R10 and **no published
frequency table**. Until si10h16 (or R10) frequencies of that recipe
sit in the fixq class (~6 cm⁻¹ rms vs full SCC), it is a force proxy,
not a vibration method. If the frequencies come back in the 300 cm⁻¹
class, the recipe is wrong even though ΔF was 1%, and the next
algorithm discussion starts from that spectrum, not from another step
count.

## 5. The kernel is bandwidth-bound, and it is saturated

`square_regtile` in the dense solver runs at 25–30% of peak because a
batched GEMM reuses a tile. The sparse product does not. One planned
SpGEMM on the R18 DMM mask (deg ≈ 378, ~412k blocks):

- one workgroup owns one output block-row
- the left row is staged in local memory, ≤384 blocks = 24 KB, so
  **two workgroups per compute unit**
- each term is a scattered **64-byte** right-block read
- **64 FMA per 64 B ≈ 2 FLOP/B**, against a balance point near 38 FLOP/B

The kernel is bound by those scattered reads. At R18 one such product
already fills the device (1648 workgroups, about ten resident waves).
A second replica multiplies the same traffic. That is why F5a gained
1.24× and stopped at B=8, while frozen F1 gained 6×: the frozen eval
was launch-and-copy bound, and the DMM chain is 15 fat SpGEMMs.

Consequences, in performance order:

- **Reorder the plan so a fetched B block feeds more than one
  accumulation.** That is the kernel lever. A new tile size, a higher
  `B`, or a fused launch of the same traffic pattern will not create
  bandwidth.
- **Fewer products at the same error.** The 1-Newton+DMM4 count (17)
  is the current quantitative candidate. Cutting it is allowed when a
  frequency check says the shorter recipe is still in the fixq class.
  Adding steps to be safe is how the column goes back to 345 ms.
- **Narrower `M_K` only with a force and frequency re-measurement.**
  Degree is the accuracy knob and the byte knob at once. The deg-95
  "13×" was a 2 Ha error. pbc-0-3 shortens `M_HS` (~3× fewer pairs)
  and lengthens the density kernel, because the gap is smaller; R18
  SCC on pbc died at `r_k=12` (report §15.20). The two radii are
  independent, and the short table is not a free speedup.
- **Column-local pairs (manifest F2)** shrink a frozen contraction
  from ~200k pairs to ~deg_hs of the displaced atom. That contraction
  is not the electronic solve §0 times. On a warm step the SpGEMMs run
  on the full `K` mask; localizing them means a sparse response
  support, which is a different algorithm. Do F2 only after a profile
  of the warm step shows the pair kernels, not the products, on the
  clock.

Left-row chunking (`MAX_LEFT_BLOCKS=384`) is a correctness cap, not a
tuning option. R18's true product mask `M_TKS` exceeds it and is
stored on `M_K` under `TRUNC_PRODUCTS=1`. A wider mask that the user
actually needs (pbc at the gap it has, or a longer `r_k`) requires the
chunked kernel before any timing claim.

## 6. What to take from the dense solver

Dense work lives in
[`HBond_Relaxed_Scan_GPU/`](../HBond_Relaxed_Scan_GPU/). It is tiled
GEMM on many small replicas. These pieces transfer.

**Take.**

- **A short fixed recipe is the fast path.** Dense's measured geometry
  update is three products and one McWeeny. Sparse's measured analogue
  is one Newton step on `Z` plus two to four commutator steps, with
  the certificate off the clock. Both exist because a from-scratch
  polynomial on every iteration cannot hit 100×.
- **The branch and the trace decision stay off the per-product sync.**
  Dense TC2 lost a launch to a 4-byte read; sparse TC2 had the same
  stall (`tc2.tr` was 87–95% of the old column). F5b's shape is right
  when a cold purify is actually required: one fixed iteration count
  for every column of one Hessian (the gap barely moves at ±0.02 Å),
  then one device reduction and one B-scalar read, fail loud with the
  column id. A variable-convergence scheduler inside one Hessian solves
  a problem this workload does not have.
- **Stop when the observable stops moving.** Past the mask floor, more
  TC2 poisons `K`. Past 1% column error, more DMM steps are a
  frequency question, not a residual question.
- **`converged = false` stays a real result.** A column that misses
  the fixed recipe is reported. It does not expand into a 55-step
  purify inside the batch.

**Leave there.**

- Resident tiled GEMM, Jacobi occupancy, and the 0.05 ms square.
  Sparse intensity is ~2 FLOP/B. Copying the dense fusion plan onto
  SpGEMM does not change the byte count.
- McWeeny as the production retraction. On this mask it is H-blind and
  it made columns worse.
- "Three products is enough." One linear response was measured at
  frozen's error. The sparse minimum that reached 1% was ~17 products,
  and only with a corrected `Z`.
- The last part of
  [`Sparse_MultiSystem_Scheduler.chat.md`](Sparse_MultiSystem_Scheduler.chat.md),
  which drifts into dense Jacobi-vs-SP2 break-even at n=86 and n=246.
  That arithmetic belongs to the dense mandate. The earlier part of
  the same chat (compact slot ids, results indexed by column id) is
  what F1 already implemented for the uniform frozen job.

## 7. The accuracy contract

Publishable, for the quantitative tier, means the si10h16 / fixq
result: ordinary modes within a few cm⁻¹ of a full SCC Hessian of the
same model, stretches not systematically tens of percent off, rigid
modes inside the FD noise band. The cube_Si65 comparison against
DFTB+ (+5–15%, and an energy offset at the same geometry) is a
**model-parity** gap. It is not a reason to add purifier iterations,
and it is not a reason to block the Hessian schedule.

Allowed compromises, in order of preference:

- f32 for every SpGEMM. f64 for the cheap decisions: trace, branch,
  occupation, the final charge mix. A wrong branch from 10⁻⁵ noise is
  the open-loop divergence; a 10⁻⁵ error in a block is not.
- A fixed product count chosen from the ladder, re-checked when the
  mask or the SK set changes. Early columns and late columns of one
  Hessian share that count.
- Preview mode as an explicit product (`VIB_FROZEN`), with the
  spectrum labelled clamped-electron. Useful for framework bands, rigid
  modes, and smoke tests.
- A commutator or force-repeatability check on a sample of columns,
  after the run, at the same wall-clock settings. A gate that fires
  inside every eval is the 175 ms tier.
- The mask floor as a stopping point. `NumericalFloor` on a truncated
  `K` is an honest status when the tail is the limit. Iterating past
  it is not more accuracy.

Forbidden compromises: dropping the 100× bar, a host sync per SpGEMM,
allocating or building plans inside the column loop, a silent cold
restart, FF32 polish on a mask that is already representation-limited,
and declaring the 208 ms central-Z batch a 1% method.

## 8. Path forward

The order is §0.1. A step is done when a few warm solves
are faster than the cold solve of the same geometries, with the force
error written down. The system is one that finishes in seconds
(si10h16, or a handful of steps / displacements on R10). A full
Hessian and a frequency diagonalization are not the acceptance test.

1. **Time the warm guess against a cold solve, on a few geometries.**
   Two sequences, same binary, release, `RUST_DFTB_PROF=mark`:
   - Optimizer: 5–10 FIRE or L-BFGS steps. Each step starts from
     `H(t), S(t), D(t)`. Report ms/step, product count, and max|ΔF|
     against a cold SCC at the same coordinates.
   - Finite difference: 2–4 displacements of one optimized geometry,
     guess = `H0, S0, D0`. Report ms/displacement and max|ΔF| against
     cold fixed-charge. Stop there. Do not continue to 6N columns.
   The recipe to time is **1 Newton on `Z` + DMM4**, not central-Z
   `n_ns=0`, and not `VIB_FROZEN`.

2. **Keep the guess when the force says it worked.** The September 16
   ladder already says one Newton plus four commutator steps is ~1%
   column error at 105 ms on R10, against 345 ms cold fixed-charge and
   against 3.5–7 s cold SCC. If a few FD elements and a few optimizer
   steps reproduce that, that recipe is the step. If the forces come
   back in the wrong-subspace class (the 300 cm⁻¹ / 86% failures), the
   guess is wrong and a faster SpGEMM will only compute the wrong
   density faster.

3. **Then make those products cheaper.** The chain is bandwidth-bound
   (§5). One change at a time: B-block reuse in the plan order, then
   a shorter product count if step 2 still holds. Report ms per
   product on the same few displacements. A higher replica batch on
   this chain already saturated (~1.2×).

4. **Do not pick up the vibrational eigh again.** `symmetric_eigh` is
   `dsyevd`. Measured 16× versus `nalgebra` at n = 768 and n = 1024.
   That is a closed side note. It is not on this path.

Antisymmetric ±h sharing (`K(−h) ≈ 2K₀ − K(+h)`, exact to O(h²)) is a
factor of about two on finite-difference elements only. Geometry
optimization has no ±h pair. It waits until the one-sided warm step
is the recipe the forces accept.

## 9. Tests that decide

**Hard rule.** No run in this table, and no run while working this
mandate, may freeze the machine, exhaust memory, or take more than
**1 minute**. A full nanocrystal Hessian is over that cap and is also
the wrong measurement (§0). A few displacements, or a few optimizer
steps, on si10h16 or a bounded R10. The same rule is GUIDELINES.md §7.

| question | test | pass looks like |
|---|---|---|
| Does an optimizer step reuse `H,S,D`? | 5–10 FIRE or L-BFGS steps, warm vs cold SCC at the same geometry | ms/step and product count down; max\|ΔF\| in the warm-update class (~1% for 1 Newton + DMM4), not a cold purify |
| Does a displaced geometry reuse `H0,S0,D0`? | 2–4 FD displacements, `VIB_MAXCOL` of that order | same comparison against cold fixed-charge; stop after those elements |
| Is that ≥100× one CPU thread? | the same few solves vs `RUST_DFTB_SPARSE_CPU=1`, one thread | ratio on the electronic solve, not on a frozen column and not on an eigh |
| Did a kernel change break the DMM batch? | `test_sparse_dmm_batch_parity` | max\|dF\| = 0 against sequential lite |
| Did the purifier walk off the mask floor? | existing TC2 history / `PurifyStatus` | stop at the floor; a doubling trace is a failed run, not a longer budget |

Profile with `RUST_DFTB_PROF=mark` or `evt`. A host timestamp that
includes the queue drain is the wall cost; kernel-event time
(`RUST_DFTB_KTIME`) is the device cost. Print product counts, not
names like `NS=2` (that label was one update plus two checks). A green
test that widened its tolerance, or a batch path that silently runs
the scalar loop, is a failure of the change. `VIB_BATCH>1` without
the lite recipe already fails loud; keep it that way.

## 10. Document map

Later text wins inside a file. The chat is design discussion. The
report's §15.16–§15.28 and the 2026-09-16 DMM report are the
measurements that settled the Hessian ladder. The September 14
entries settled what the purifier floor actually is.

### This folder — read

| file | role |
|---|---|
| **this file** | standing order and the path |
| [`Sparse_Nanocrystal_Vibrations.report.md`](Sparse_Nanocrystal_Vibrations.report.md) **§15.16–§15.28** | the numbers: DMM ladder, what the warm guess costs, SpGEMM saturation. The eigh-wall sentences in §15.26 are the frozen path; §0 says not to chase them |
| [`../reports/2026-09-16_sparse_dmm_warm_density_hessian.md`](../../reports/2026-09-16_sparse_dmm_warm_density_hessian.md) | the commutator bugs (Z is S⁻¹, bsym on asymmetric X) and the stripped-tier table |
| [`../../topical_audit/hessian_eval_bottleneck.md`](../../topical_audit/hessian_eval_bottleneck.md) | why a cold column was 3.5–7 s, and the warm-TC2 refutation. The "next lever is batching" line at the end is done; §15.27–§15.28 are the outcome |
| report §15.12–§15.15 | the floor is stored-K truncation; FF32 helps a full mask only; one-read TC2 and the stagnation detector |
| [`Sparse_Nanocrystal_Vibrations.chat.md`](Sparse_Nanocrystal_Vibrations.chat.md) **from the Tier-1 rebuttal (~line 11600) through ~13870** | why the gates had to leave the timed path. The three stripped runs it orders were executed; the results are §15.17, and they did not match the hope that the middle tier would vanish |
| manifest **§F** (and §F.1) | residency contract, F1/F5a as built. Read after §0 so a full-Hessian or eigh task is not picked up from the open F2/F5b boxes |

### This folder — background, do not steer from these

| file | why it is not the guide |
|---|---|
| report through 2026-09-13, manifest §13–§15.11 | correctness era: Gate C/E false positives, device-NS contract, pbc parity. The physics bugs there were real and are fixed or reclassified. The work order is not the current one |
| manifest §1.4's order "correctness, then rigor, then speed" | right while forces were wrong. Applied now, it puts the certificate back inside the column |
| [`Sparse_Nanocrystal_Vibrations.review.md`](Sparse_Nanocrystal_Vibrations.review.md) | 2026-09-09. Stale relative to §15.16–§15.28 |
| [`Sparse_Nanocrystal_Vibrations.tasks.md`](Sparse_Nanocrystal_Vibrations.tasks.md) Phases A–E | phase list. A–C are done; E's "next 10× is batching" was measured and split into a finished frozen 6× and a saturated DMM 1.2× |
| [`../../topical_audit/f32_floor_sparse.md`](../../topical_audit/f32_floor_sparse.md) | 2026-09-10 classification of bugs vs floor. Still right that the early floors were mixed with stopping criteria. Superseded on mechanism by report §15.12-2′ |
| [`Sparse_MultiSystem_Scheduler.chat.md`](Sparse_MultiSystem_Scheduler.chat.md) after the slot-pool design | the ending is a dense Jacobi/SP2 argument. Do not schedule sparse columns from it |
| manifest §15.9–§15.10 "as few neighbors as possible" | the right research question, already answered well enough to stop treating it as the Hessian plan: degree follows the gap, and a short degree at 2 Ha error is not a speedup |

### Dense folder — the transferable pages

| file | take |
|---|---|
| [`Dense_Multi_Performance.md`](../HBond_Relaxed_Scan_GPU/Dense_Multi_Performance.md) | the same bar, the other roofline. Read §1 and §6 of that file with §6 of this one |
| dense notes Parts VI–VIII | evidence that a short warm recipe can beat a long polynomial **after** the predictor matches the physics. Sparse's version of that result is the DMM ladder, not a McWeeny polish |

### Tests and code named above

- `rust_dftb/src/methods/sparse/sparse_dftb.rs` — FIRE / relax, `forces_dmm_batch`, the warm column path. Frozen evals are the preview, not the clock in §0
- `rust_dftb/src/methods/sparse/sparse_system.rs` — `dmm_descend`, TC2, Newton–Schulz
- `rust_dftb/src/methods/sparse/sparse_hs.cl` — pair assembly, contract, gather
- `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl` — planned SpGEMM, TC2
- `rust_dftb/scripts/bench_vib_r10.rhai`, `bench_eval_r10.rhai` — the column clocks
- `test_sparse_frozen_batch_parity`, `test_sparse_dmm_batch_parity`, `test_sparse_pair_gpu_vs_cpu_parity`
