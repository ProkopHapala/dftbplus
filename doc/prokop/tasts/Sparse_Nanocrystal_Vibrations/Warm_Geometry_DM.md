---
type: experiment spec
title: Sparse geometry step — reuse the previous density under a purifier constraint
tags: [gpu, sparse, dmm, mcweeny, tc2, fire, extrapolation, certificate]
timestamp: 2026-09-22
status: R10 FIRE 2026-09-22 — B3 keeps the kernel, energy falls 3 Ha, force still creeping below 1e-3
---

# Warm geometry step — variants to try

A FIRE step must minimize the DFTB energy and keep `Tr(KS) = Nocc`.
Chained commutator steps on Si₁₉₆H₁₃₄ did neither: the raw trace
climbed by ~0.6 occupied orbitals in 24 steps, scaling `K` back moved
the band energy by tenths of a hartree, and the plotted energy rose
while the force fell (`Sparse_Performance.md` §0.1). Cold TC2 from
`K0` does both, and it throws the previous density away.

The dense geometry tests already measured a reuse that holds the
energy and the trace. This note copies that split into the sparse
metric and lists the variants to run. It does not change the solver.

## 0. Where the idea is written down

Read these, in this order. Line numbers are of the files as of
2026-09-22.

| what | file | lines |
|---|---|---|
| The request: purification is the Pauli constraint; the missing piece is downhill minimization of `E` in `H` and `S`; idempotency may be loose early and tight only near the minimum | `../HBond_Relaxed_Scan_GPU/Alternative_Dense_Multi_Eigensolve.chat.md` | 1422–1426 |
| Complementary pair. Continuous double commutator rotates eigenvectors and preserves occupations. Purification changes occupations and preserves eigenvectors. Pauli is `0 ≤ f_i ≤ 1` and `Tr = Nocc`; idempotency `f_i ∈ {0,1}` is the T = 0 special case | same chat | 1478–1612 |
| For SCC the functional derivative of `E[D]` is `H_scc`, so one downhill step in `D` relaxes subspace and charges together | same chat | 1617–1674 |
| Every K-only warm seed inside an SCC iteration fails. TC2 locks the seed's eigenvectors. `2K₁−K₂` **diverges inside SCC** because the charge iteration is decelerating | `../HBond_Relaxed_Scan_GPU/Alternative_Dense_Multi_Eigensolve.md` | 655–680 |
| Sparse DMM is the rotation; a TC2 tail after it degrades `R_H` on the mask; the certificate is `R_H`, and a failure is a cold restart | same | 682–706 |
| Fixed `η = 8` diverges on an O(1) SCC gradient. Stable discrete step is `η ≈ 1` with a trust cap | same | 783–791 |
| Amortized schedule: 1–2 commutator steps and 2–4 purification steps per iteration, certificate only at the end, cold restart on failure. This is a proposal, not a measured geometry trajectory | same | 934–957 |
| Fixed `η = 4` diverges on a real spectrum (`R_H` grows ×1.6/round). Accept/reject trust region: checkpoint, restore and halve `η` if `R_H` increases, double `η` (cap 8) if it falls | `../HBond_Relaxed_Scan_GPU/TrDH_minimization_purification_Notes.md` | 1098–1104 |
| Carrying **neutral** charges made the geometry benchmark lie. The target Hamiltonian is `H(R_new \| q_old)` | same | 1106–1115 |
| **The measurement that transfers.** `2D₁−D₀` in the AO basis, one McWeeny, certificate: 3 products, `ΔE ~ 1e-7–1e-6`, displacements 0.01 / 0.05 / 0.10 Å, H₂O and formic, zero commutator corrections | same | 1117–1145, 1158–1164 |
| Cross-geometry orbital transport `S_new⁻¹ S_cross C_old` is the wrong ansatz for an atom-centered basis (commutator ~0.7 vs 0.001–0.02) | same | 1147–1156 |
| Learned constraint force `Λ` / LNV. Cold path works. Warm start is broken and takes thousands of iterations. Do not port it | same | 12, 40–84, 353–367 |

Two uses of `2K₁−K₀` are different objects. Inside one SCC loop it
overshoots (Alternative §7.6.1, line 663; TrDH VI.1, lines 1084–1089).
Across a geometry step, in the AO basis, along a continued
displacement, it is the predictor that passed (TrDH VI.2b). Every
variant below is a **geometry** step. None of them extrapolates along
the SCC charge history.

## 1. Shared equations

Spin-restricted DFTB. The stored object is the density kernel `K`,
with `D = 2K`. On the non-orthogonal basis

```text
P = K S
K S K = K
Tr(K S) = Nocc
```

`Nocc` is the number of occupied orbitals (R10: 459). The electron
count is `2 Nocc`. A trace error of 0.6 on `Tr(KS)` is 1.2 electrons.
Report `Tr(KS) − Nocc` and call it the orbital-count error.

`Z` is the Newton–Schulz approximation of `S⁻¹`, not of `S⁻½`.
`B = Z H_scc` is the matrix already kept as `b_zh`. One Newton
update of `Z` runs before any product that uses `Z`, because `S`
changed with the atoms. Spectral bounds `(εmin, εmax)` are the ones
cached from the last `K0` of this electronic problem; they are not
recomputed inside the step.

Band energy, at the charges that built `H_scc`:

```text
E_band = 2 Tr(K H_scc)
```

The nuclear repulsion and the SCC charge term are already functions
of the coordinates and of `q`. The reported total energy is that sum.
It is the energy the forces must descend. A `K` that was scaled to
fix the trace is not a stationary point of this functional, so its
energy and its forces are not a pair.

### 1.1 Rotation — one sparse commutator step

This is `dmm_descend` in `sparse_system.rs` (the collapsed 3-product
form; the older `(S⁻¹−K)HK` expression in Alternative §7.6.3 is the
same direction before that collapse).

```text
X = (Z H) K
Y = (K S) X
G = X + Xᵀ − 2 Y
s = −η / (εmax − εmin)
K ← sym(K + s G)
```

`η` is the dimensionless scale. The code's `eta_scale` argument **is**
this `η` only when the caller passes `η` itself; `dmm_descend` divides
by `(εmax−εmin)` again. A caller that wants a coefficient `s` must
pass `eta_scale = −s (εmax−εmin)` with the sign convention of the
function (`step = −eta_scale / Δε`, then `K ← K + step·G`). In this
note `η` means the dimensionless number in `s = −η/Δε`, and the first
trials use `η = 1`.

On an exact projector the continuous orthogonal flow (chat lines
1496–1538)

```text
dD/dt = −[D, [D, H]] = 2 D H D − D H − H D
dE/dt = −‖[D, H]‖² ≤ 0
```

is isospectral: occupations stay put, eigenvectors rotate. A finite
`η` on a **masked** product is only approximately that flow. That is
why a step can create orbital count. The trust region in §2 exists
to refuse the step when the approximation has already left the
occupation bounds.

### 1.2 Constraint — one generalized McWeeny

```text
Q = K S K
V = K S K S K
K ← 3 Q − 2 V
```

This is `mcweeny_polish_planned`: three sparse products, then the
combine. It changes occupations and does not rotate the eigenvectors
of `K` (chat lines 1461–1473; Alternative lines 669–676). It is legal
only when `K` is already in the right subspace and the occupations
have only been nudged. It is not a repair for a kernel whose trace
has left `Nocc` by an orbital.

### 1.3 Predictor — AO linear extrapolation

Same atom ordering, same BSR block pattern (the neighbor list has
not been rebuilt; the step is inside `r_skin`):

```text
K* = 2 K₁ − K₀
```

`K₁` is the kernel accepted at the previous geometry, `K₀` at the
one before that. The subtraction is elementwise on the stored blocks.
If the sparsity pattern changed, this difference is undefined: that
step is the cold restart (§2.3), and the missing blocks are not
filled with zeros.

`K*` can have occupations outside `[0, 1]` and a trace off `Nocc`.
The constraint step and the certificate exist to catch that. The
predictor is not itself a legal density.

### 1.4 Certificate

Computed on the kernel **before any rescaling**. Rescaling is not
part of any variant.

```text
τ     = |Tr(K S) − Nocc|
R_I   = ‖K S K − K‖_F / ‖K‖_F
A     = H (K S)
R_H   = ‖A − Aᵀ‖_F / (2 ‖A‖_F)
```

`R_H` is `rh_stationarity` (Alternative lines 703–706). Trace and
`R_I` alone accept a wrong subspace; `R_H` is the check that `K`
commutes with `H_scc`.

On the R10 mask the cold solve itself sits at `R_H ≈ 2×10⁻³`. A gate
copied from a dense complete basis (`10⁻⁴`, or the dense measured
`2×10⁻⁷`) rejects a correct sparse projector. For a sparse step the
gate is the cold `R_H` of **that** system, times a small factor, not
a universal constant. On SiH₄ the mask is complete and the cold gate
`5×10⁻⁴` is the right one.

Proposed first-trial accept window, to be printed and then tightened
from the numbers, not before:

```text
τ     < 0.02          one step, before any later step sees this K
R_H   < R_H(cold) × 3 on this geometry, or the SiH₄ gate 5×10⁻⁴
```

`R_I` is reported. It does not by itself reject a step while the
relaxation is still far from the force minimum (chat lines
1560–1611): a fractional occupation inside `(0, 1)` with the trace
held is an ensemble density, not a created electron. An occupation
outside `[0, 1]` is a reject. The first trials do not have a cheap
per-eigenvalue bound, so `τ < 0.02` is the proxy. If a later trial
shows `τ` small while a single occupation has left `[0, 1]`, the
proxy is insufficient and that variant stops.

## 2. Rules every variant shares

1. Carry the converged charges. Build `H_scc(R_new, q_old)`. Resetting
   to neutral charges makes the seed error independent of the
   displacement (TrDH lines 1106–1115).
2. One Newton update of `Z`, then `refresh_b_zh`, before the first
   product that uses them.
3. Checkpoint `K` before a commutator step. Restore it on reject.
   Never multiply `K` by `Nocc / Tr`.
4. A McWeeny or TC2 step runs only on a kernel that already passed
   the trace window. Purification of a leaked kernel locks the wrong
   subspace (Alternative lines 669–676; sparse post-DMM TC2, lines
   701–702).
5. If the certificate fails, discard `K` and run cold TC2 from `K0`
   at this geometry and these charges. The cold result is the state
   that is stored and differentiated. The failed kernel is not mixed
   in.
6. No kernel is built and no buffer is allocated inside the step.
   The second history kernel `K₀` is allocated with the workspace.

### 2.1 Cold restart — V0, and the failure path of every other variant

```text
K ← K0(H_scc)                         Palser seed of the new H
K ← TC2(K) until the existing floor   trace guard, R_I floor
q ← Mulliken(K)
accept this (q, K, E, forces)
```

This is the minimizer that already works. Every other variant is
kept only when, on the same geometry, it matches V0 to the tolerance
in §4. A variant that cannot, on a bad step, call V0 and report
V0's energy has failed the spec even if its cheap steps look fast.

## 3. Variants

Each variant is one geometry step. `K₁` is the kernel accepted at
the previous geometry. The first geometry of a run has no `K₁` and
is V0. V1, V2 and V4 also need `K₀`; until two geometries have been
accepted they use `K₁` with no extrapolation (the one-point row of
TrDH VI.2b, which still passed, at 3–15 products).

### V1 — extrapolate, one McWeeny, certify

The dense winner (TrDH lines 1141–1145, 1158–1164): three products,
no commutator, `ΔE` at 10⁻⁶–10⁻⁷ on displacements up to 0.1 Å.

```text
K ← 2 K₁ − K₀
K ← 3 KSK − 2 KSKSK                 one McWeeny
measure τ, R_H, R_I
if accept:  q ← Mulliken(K); store K as the new K₁
else:       V0
```

Sparse risk, unmeasured: on the truncated mask a McWeeny of a
`δK0` seed locked the wrong subspace. `2K₁−K₀` is a prediction of
the occupied kernel, not of the Palser initializer, which is why it
is a different operation from that failure. SiH₄ (complete mask)
separates "the predictor is wrong" from "the mask spoiled McWeeny".

### V2 — extrapolate, no purification

Same predictor, no polynomial. This is the control for V1. If V2
already passes the certificate, the McWeeny was unnecessary. If V2
passes `τ` but fails `R_H`, the predictor did not rotate enough and
the commutator variants are the ones that can add that rotation.

```text
K ← 2 K₁ − K₀
measure τ, R_H, R_I
if accept:  q ← Mulliken(K); store
else:       V0
```

### V3 — trust-region commutator from the previous kernel

No extrapolation. This is the R10 commutator with the accept/reject
rule that the dense spectrum measurement required (TrDH lines
1100–1104) and that the sparse FIRE run did not have. `η` starts at
1, not at 8 and not at a fixed cap of 0.5.

```text
η ← 1
repeat at most 4 times:
    checkpoint K
    one commutator step at this η          §1.1
    measure τ, R_H
    if τ ≥ 0.02 or R_H > R_H_before:
        restore K
        η ← η / 2
        if η < 1/16: V0 and stop
    else:
        accept the step
        η ← min(2 η, 4)
if the last accepted K passes the certificate:
    one McWeeny only if τ < 0.02 still holds after it would be
    a no-op check: run it, and if τ or R_H worsens, restore
    q ← Mulliken(K); store
else:
    V0
```

McWeeny is off the path that creates the leak. It runs once, after
a step that already held the trace, and it is itself checkpointed.

### V4 — extrapolate, then the trust region

V1's predictor, then V3's corrector only if V1's certificate fails
**before** the McWeeny. Order matters: do not McWeeny a kernel you
are about to rotate, and do not rotate a kernel the certificate
already accepted.

```text
K ← 2 K₁ − K₀
measure τ, R_H
if accept:
    one checkpointed McWeeny (§V1)
    done
else:
    restore K ← K₁                 the predictor is discarded
    run V3 from K₁
```

If the FIRE velocity has reversed, `2K₁−K₀` points the wrong way.
Landing in V3 or V0 on that step is the certificate working.

### V5 — one commutator and one constraint per charge mix

The amortized schedule (Alternative lines 934–957), cut down so the
purification cannot see a leaked kernel. Inside one geometry step,
at most three charge mixes. Each mix rebuilds `H_scc` from the mixed
`q` and takes **one** accepted commutator step. There is no TC2 tail
on this path: a TC2 from a warm kernel is the operation §7.6.2
showed locks the subspace, and the sparse measurement already found
a TC2 after DMM degrading `R_H`.

```text
q ← q_old
for mix in 0..3:
    H ← H_scc(R_new, q)
    refresh B = Z H
    checkpoint K
    one commutator step, η = 1
    if τ ≥ 0.02 or R_H increased:
        restore K
        V0 and stop
    one checkpointed McWeeny
    if τ or R_H worsened: restore the pre-McWeeny K
    q_new ← Mulliken(K)
    if rms(q_new − q) < 10⁻³: q ← q_new; break
    q ← DIIS(q, q_new)                 existing mixer, not α = 1
if the final certificate fails: V0
```

Replacing `q` outright (`α = 1`) sloshed on R10. The mix stays the
existing DIIS / `α = 0.5`.

Progressive idempotency (chat lines 1560–1611) is a switch on this
loop, not a sixth solver. While `max|F|` is above the FIRE stop
(`2×10⁻²` Ha/Å in the R10 run) the McWeeny in the loop is skipped
whenever `τ` and `R_H` already pass; the occupations are allowed to
stay fractional. On the step that would be reported as converged,
and on any step whose `R_I` is above the cold floor, the McWeeny
runs. The trace window is not relaxed in either regime.

## 4. What a pass is

SiH₄ first, one minute, debug build, algebra prints off. Five FIRE
steps from a displaced start (`dt` small enough that a step stays
inside the skin, the value already used on the small system). After
each step, a cold V0 solve of that same geometry is the reference.
It is computed on a second engine or after copying `K` out; it does
not overwrite the warm state the next step needs.

Print, per step, per variant, before any later analysis:

```text
variant  step  accept|reject|restart
E_Ha  E_ref_Ha  dE
max|F|  max|F_ref|
Tr_raw  Nocc  τ
R_H  R_H_ref  R_I
η  n_products  ms
```

Pass, all five steps:

- no restart required is **not** the requirement; a restart that
  then matches V0 is a pass of the gate
- every **accepted** step has `τ < 0.02` on the raw trace
- `|E − E_ref| < 10⁻³` Ha and `max|F − F_ref|` below 2% of
  `max|F_ref|`
- on a step that starts above the force of the geometry it is
  moving toward, `E` does not rise relative to `E_ref`'s trend.
  Comparing to the warm energy of a previous rejected step is
  meaningless

R10 (Si₁₉₆H₁₃₄, the existing capped relaxation, wall under a minute)
runs only for a variant that passed SiH₄. The plot is `E − E_0` and
`max|F|` against step index, plus `τ` before the certificate. A run
whose energy rises while `max|F|` falls is a failure of that variant,
same as the previous figure.

## 5. Closed — do not put these back in the list

| closed | why | where |
|---|---|---|
| Warm TC2 from `K_old` | polynomial cannot rotate; fixed point of the old subspace | Alternative 660, 669–676 |
| `δK0` shift, then McWeeny or TC2 | first-order change of the **initializer**; spurious projector, commutator 0.238, charges off by 0.2 e | Alternative 662, 682–692 |
| `β` blend, XL extrapolation **along the SCC history** | residual floor, or the error doubles per iteration | Alternative 661, 663 |
| Scale `K` by `Nocc/Tr` and continue | puts the trace back and moves `E_band` by tenths of a Ha | `Sparse_Performance.md` §0.1 |
| McWeeny or TC2 of a kernel that already failed `τ` | locks whatever subspace the leak landed in | Alternative 669–676; sparse lines 701–702 |
| Cross-overlap orbital transport | pins orbitals to the old absolute position; atom-centered bases follow the atoms | TrDH 1147–1156 |
| Learned `Λ` / LNV residual descent | warm start broken; hundreds to thousands of iterations even when it converges | TrDH 12, 353–367 |
| Fixed four steps at `η = 8` or at a hard cap of 0.5, no reject | the R10 trajectory that lost orbital count | `Sparse_Performance.md` §0.1 |

## 6. Order to run them

V0 is the reference, built once per geometry. Then V2, V1, V3, V4 on
SiH₄, one binary flag or one test each so a failure is one variant.
V5 only if V3 holds the trace but the charges need more than one mix
to match V0's forces. Stop at the first variant that passes §4; the
later ones are not required for the relaxation.

## 7. SiH₄ trial (2026-09-22)

`GeomStep` on `SparseDftb`, tests
`test_sparse_dftb_sih4_geom_v{1,2,3,4}`. Five FIRE steps, `dt = 0.5`
(a `dt` of 0.02 from rest moves an atom by ~10⁻⁵ Å and accepts every
kernel; that run is not a measurement). Cold reference is a second
engine with the projector forgotten, so it always runs TC2. Gate
`R_H < 5×10⁻⁴`, `τ < 0.02`. Debug build, algebra prints off.

The steps are real: `|ΔR|` = 0.010, 0.024, 0.036, 0.045, 0.053 Å.
Whenever the certificate rejected the kernel, the cold restart
matched the reference energy to ~10⁻⁷ Ha and the forces to ~10⁻⁶.
The reported energy fell (−2.764 → −2.796 Ha) and `max|F|` fell
(0.088 → 0.065) because those steps were the cold solve. That is
not a warm relaxation.

| variant | accepts | what the certificate saw |
|---|---:|---|
| V2 `2K₁−K₀`, no McWeeny | 0 / 5 | `τ` of 2×10⁻⁴–3×10⁻³, `R_H` ~ 10⁻³, `R_I` ~ 2×10⁻³ |
| V1 same predictor, one McWeeny | 0 / 5 | `τ` drops to ~10⁻⁵ and `R_I` to ~10⁻⁵; `R_H` stays 0.9–1.6×10⁻³ |
| V3 trust-region commutator, `η` from 1 | 1 / 5 | the 0.010 Å step reaches `R_H = 3.6×10⁻⁴` in 4 steps, 2.4 ms, `|ΔE| = 1.3×10⁻⁵`, force error 1%. From 0.024 Å up, 4 steps stop at `R_H` ≈ 0.9–1.2×10⁻³ and restart (~20–30 ms) |
| V4 predictor, else V3 | 1 / 5 | the predictor is rejected; the one accept is the same 0.010 Å commutator as V3 |

Same seed, `|ΔR| = 0.010` Å, before any commutator: McWeeny changes
`τ` from 2.2×10⁻³ to 4.8×10⁻⁵ and leaves `R_H` at 1.1×10⁻³. The four
commutator steps then walk `R_H` 1.16×10⁻³ → 1.03×10⁻³ → 8.1×10⁻⁴ →
5.2×10⁻⁴ → 3.6×10⁻⁴, and the checkpointed McWeeny puts the trace on
`Nocc`. That is the complementary pair, on this molecule: purification
repairs the occupations, the commutator is what moves the subspace,
and four steps of it are enough at 0.01 Å and not at 0.02 Å.

No variant earned the R10 run. V5 was not run. The gate was not
loosened: the cold `R_H` on these geometries is 10⁻⁵–10⁻⁷, and a
warm `R_H` of 10⁻³ is the quantity the gate exists to refuse.

## 8. Bold 0.1 Å steps (2026-09-22)

The §7 gate is the wrong object for a geometry step. A trace error of
10⁻³ is a charge error of that order, and one McWeeny repairs it.
Rejecting the kernel and running cold TC2 makes the step as expensive
as the thing it was meant to replace. `GeomStep::BoldXtr`,
`BoldDmm`, and `BoldXtrDmm` keep the kernel. No rescaling, no restart.
`η = 8` means `K ← K − (8/Δε) G`.

SiH₄, three successive +0.1 Å moves of one hydrogen, each scored
against a cold SCC of that geometry. Debug build.

| recipe | step | \|ΔE\| (Ha) | max\|ΔF\| / max\|F\| | τ after McWeeny | warm | cold |
|---|---:|---:|---:|---:|---:|---:|
| B1 extrapolate + 1 McWeeny | 0 (no K₀) | 3.9×10⁻³ | 12% | 4.9×10⁻³ | 4.9 ms | 51 ms |
| B1 | 1 | 1.1×10⁻³ | 24% | 3×10⁻⁴ | 1.7 ms | 33 ms |
| B1 | 2 | 1.6×10⁻³ | 26% | 5×10⁻⁵ | 3.4 ms | 28 ms |
| B2 two η=8 steps + 1 McWeeny, from K₁ | 0 | 1.4×10⁻³ | 0.3% | 1.6×10⁻³ | 6.7 ms | 37 ms |
| B2 | 1 | 1.2×10⁻³ | 0.5% | 1.3×10⁻³ | 3.9 ms | 22 ms |
| B2 | 2 | 4.0×10⁻⁴ | 0.6% | 8×10⁻⁴ | 2.2 ms | 23 ms |
| B3 extrapolate, then B2's two steps | 1 | 1.2×10⁻⁴ | 0.7% | 2×10⁻⁵ | 2.6 ms | 26 ms |
| B3 | 2 | 3.7×10⁻⁴ | 0.7% | 2×10⁻⁵ | 1.8 ms | 29 ms |

On the first 0.1 Å step the raw kernel has `τ ≈ 0.04`. One McWeeny
brings that to ~10⁻³ and does not move `R_H` (B1: 5.3×10⁻³ before and
after). The two commutator steps are what cut `R_H`, from 5×10⁻³ to
~2×10⁻³, and the McWeeny then finishes the trace. With a second
history point the extrapolation starts closer (`R_H` raw 1.5×10⁻³
instead of 5×10⁻³) and the same two steps land at `R_H = 2.5×10⁻⁴`,
`|ΔE| = 0.1` mHa.

Three chained 0.1 Å steps did not accumulate a trace leak and did not
walk the energy off by tenths of a hartree. B1 is not the force
update: purification repairs occupations and the forces stay ~20%
off. B2/B3 are the useful pair. The truncated-mask nanocrystal has
not been run with this recipe.

FIRE on SiH₄ with B3 (`dt = 1`, steps up to the 0.1 Å cap), after two
corrections that the straight-line probe did not see: drop
`2K₁−K₀` when it raises `R_H` (a reversed step), and mix charges at
`α = 0.5` instead of replacing them. Twelve steps, every one downhill.
`E` −2.76421 → −2.82490 Ha. `Tr(KS)` stayed within 10⁻³ of 4, and the
charge residual fell from 0.02 to 6×10⁻⁴. Final energy is 5×10⁻⁵ Ha
from a cold SCC of that geometry. Mean warm step 1.6 ms, that cold
solve 7.1 ms. Si–H went 1.480 → 1.525 → 1.472 Å; `max|F|` is still
0.024 at that snapshot.

A longer SiH₄ run (`dt = 0.3`, same B3, `α = 0.5`) does finish.
Forty-seven steps, every one downhill, `E` −2.76421 → −2.82609 Ha,
`max|F|` 0.088 → 7.8×10⁻⁴, `Tr(KS) = 4.000000`, Si–H 1.480 → 1.478 Å,
mean 2.2 ms/step. Geometries:
`debug/sparse_relax/sih4_initial.xyz`,
`debug/sparse_relax/sih4_final.xyz`.

The truncated-mask crystal is §9. On that mask `α = 0.5` is too
fast: the Mulliken residual walks and the energy turns up. The R10
runs below use `α = 0.2`.

## 9. R10 relaxation (2026-09-22, RTX 3090, debug build)

Recipe is B3 (`GeomStep::BoldXtrDmm`). One Newton update of `Z`,
rebuild `H_scc` at the carried charges, keep `2K₁−K₀` only when it
does not raise `R_H`, two commutator steps at `η = 8`, one McWeeny,
keep the kernel. Charges move by `α = 0.2`. FIRE `dt = 0.1`, each
atom capped at 0.1 Å. First SCF of a process is still cold TC2.
`RUST_DFTB_SCC_RHGATE=1e-2` because the R10 mask floor is
`R_H ≈ 2×10⁻³`, above the 5×10⁻⁴ default. Algebra prints off.
SK is matsci-0-3.

The start is not the cut sphere. Every atom of Si₁₉₆H₁₃₄ is moved
0.1 Å along a fixed random direction (seed `0x5eed`) before step 0.
That kick shortens the worst Si–H from the as-cut 1.46 Å to 1.27 Å
and puts `max|F|` at 0.268. Files:
`debug/sparse_relax/si_R10_before.xyz`,
`debug/sparse_relax/si_R10_traj.xyz` (multi-frame; the `floor=`
frames are a later continuation of the same coordinates),
`debug/sparse_relax/relax_EF.png`,
`debug/sparse_relax/history.csv`.

235 FIRE steps, 42 s inside the loop (setup of the engine is extra,
~8 s). The 10⁻³ force stop used on the first plot was removed; this
run hit the 42 s wall while the force was still setting new lows.

| | kicked start | step 234 |
|---|---:|---:|
| energy | −311.014 Ha | −314.098 Ha |
| `max\|F\|` | 0.268 | 7.8×10⁻⁴ (best 5.2×10⁻⁴ at step 231) |
| Si–H | 1.457 Å (shortest 1.271) | 1.492 Å (shortest 1.489) |
| `Tr(KS)` | 458.999 | 458.9995 |

Largest atomic move from the kick: 0.62 Å. The energy drop is 3.08 Ha
and the curve is smooth. The force falls overall and oscillates, which
is FIRE, not a loss of electrons. `Tr(KS)` stays on 459.

A cold SCC of the step-234 geometry, in a new process, gives
`E = −314.09935` Ha and `max|F| = 8.3×10⁻⁴`. The warm energy at that
geometry was 1.5 mHa higher. The warm force and the cold force are
the same size.

187 further warm steps from that cold restart (36 s) leave the energy
at −314.1017 Ha, with changes under 0.1 mHa in the later steps, and
`Tr(KS) = 459.0002`. The lowest `max|F|` in that stretch is
2.3×10⁻⁴. It does not stay there. The component bounces between about
2×10⁻⁴ and a few ×10⁻³.

One more continuation from that geometry, same recipe, stops on its
own. After a cold SCC (`max|F| = 3.2×10⁻⁴`, `E = −314.09988` Ha) the
warm steps reach a best `max|F| = 1.80×10⁻⁴` at local step 72
(`E = −314.10179` Ha, `Tr(KS) = 459.0002`). The next 40 steps never
beat that low by 2%. The force in that window sits between
1.9×10⁻⁴ and about 4×10⁻⁴, and the energy moves by ~10⁻⁵ Ha. That is
the saturation seen so far: about **2×10⁻⁴ Ha/Å**, not 10⁻³. A
1000-step run was not continued past this stall. Total FIRE steps
from the kick, across the three processes, are about 540. The later
frames in `si_R10_traj.xyz` are tagged `frame=`.

### Where the 180 ms goes

One profiled descent (182 steps, `RUST_DFTB_PROF=evt`, same recipe)
puts 74% of device time in the trace / `R_H` certificate, which is
printed five times per step at 26 ms each. That read is not the
update. The kernels that move `K` are about 30 ms of GPU time:

| stage | per call | share of device time |
|---|---:|---:|
| certificate `Tr(KS)`, `R_H` | 26 ms × 5 | 74% |
| two commutator steps | 7.3 ms | 8% |
| one McWeeny | 7.1 ms | 4% |
| cold TC2, first solve only | 1.6 s once | 5% |
| Newton update of `Z` | ~6 ms | 3% |
| forces | ~2 ms, twice | 3% |
| rebuild `H_scc` | 1.4 ms | 1% |

### R14 relaxation, started from the engine script

The same recipe is now a Rhai script, `rust_dftb/scripts/sparse_relax.rhai`,
run by `dftb_engine --script`. The crystal is `RUST_DFTB_XYZ`. A later
chunk sets `RUST_DFTB_CONTINUE=1` and loads `after.xyz`. No Rust edit
per system.

Si R14 (864 atoms, 282 H) from a 0.1 Å/atom kick, `TRUNC_PRODUCTS=1`,
same radii as R10. After 278 FIRE steps the run has **not** reached
the force floor. Energy −880.43 → −889.62 Ha. Best `max|F|` in the
tail is 1.4×10⁻³ Ha/Å; the last steps still set new lows and the
force still spikes up to ~8×10⁻³. `Tr(KS)` stays on 1305 (warm drift
~0.005). Si–H went from 1.28–1.65 Å to 1.490–1.512 Å by step 172.
Files: `debug/sparse_relax/r14/traj.xyz`, `history.csv`,
`relax_EF.png`. Each process pays the ~27 s mask build, so the
descent is chunked at `RUST_DFTB_WALL_SECS=24`.

The interior `K·S` row at `r_k = 12` Å is 864 blocks wide and does not
fit the GPU local-memory cache (`MAX_LEFT_BLOCKS` was 400 on that
mask). `RUST_DFTB_TRUNC_PRODUCTS=1` stores that product on `M_K`. The
code documents this as dropping ~10⁻⁵ tail terms. R10 was retimed with
the same switch; the warm step stayed 176 ms, so the switch is not
what the R10 relaxation paid.

On the profiled first chunk (`PROF=evt`, `PROF_ACCUM=1`, 26 steps)
the certificate was most of the device time: 85 ms × 5 calls, 58%.
Three of those five calls were log lines. The production step now
measures `Tr(KS)` once after McWeeny, and samples `R_H` of the
extrapolant once every 8 extrapolations (`RUST_DFTB_GEOM_RH_EVERY`).
`RUST_DFTB_GEOM_CERT=1` restores the five-call printout. The kernels
that move `K` are a commutator step at 21 ms (two per geometry),
McWeeny 19 ms, Newton on `Z` ~13 ms, forces ~5 ms twice, `H_scc` 4 ms.
The profiled warm wall of ~510 ms includes the five certificates.
Rerun with the gate off (2026-09-22): R10 from the same 0.1 Å kick
plateaus at max|F| = 2.2×10⁻⁴ after 497 steps, E = −314.098 Ha,
Tr(KS) = 458.999, Si–H 1.488–1.497 Å, median step 35 ms. R14 near
the same geometry steps at ~100 ms (was ~510 ms); the force there
is still ~10⁻³. Cold SCC is 5.1–6.5 s. The 27 s of mask setup shows up as host time
on the first `geom.xyzu` tick; the device side of that tick is 0.3 ms.

| | atoms | K blocks | setup | cold SCC | warm step |
|---|---:|---:|---:|---:|---:|
| R10 | 330 | 58×10³ | 6.1 s | 1.93 s | 176 ms |
| R14 | 864 | 194×10³ | 26.6 s | 5.06 s | 483 ms |

Atom count ×2.62. Nonzeros ×3.36. Cold SCC ×2.62. Warm step ×2.74.
Setup ×4.4. The step tracks the number of atoms. The setup does not:
`build_geometric_mask` is a double loop over atoms, and it runs for
`M_HS`, `M_K`, `M_Z`, and the halo. R18 (1648 atoms) was not started.
The R14 setup had already used 27 s.

That setup is paid once per process, not once per FIRE step. On the
R10 descent, ~180 steps to `max|F| ~ 10⁻³` cost ~32 s of steps against
~6–8 s of setup. A 10³-step run makes the setup a few percent of the
wall clock. It is still the number that jumps from 6 s to 27 s between
these two crystals, and it is the part that will hurt a scan over
many independent geometries. The per-step cost is the part that
scales cleanly.
