---
type: finding
title: Bold commutator step — fast and stable, does not converge to Jacobi
tags: [gpu, dense, dmm, mcweeny, geometry, accuracy]
timestamp: 2026-09-22
status: open — one step is useful; more of the same inner step is not the fix
---

# The bold step does not converge to the Jacobi solution

One geometry step that rotates the previous density with a few
commutator steps is fast, keeps the electron count, and lands in the
right part of the energy surface. It does not become the Jacobi SCC
answer if those steps are repeated. On formic, GC, and diazaphen the
force error grows as the rotation is continued. That is a measured
fact (`test_geom_bold_nacc`, batch 256, 2026-09-22), not a tolerance
issue.

The step is still worth keeping. At batch 256–1024 it is about 8–14×
a Jacobi SCC of the same geometry step. A version that is 5× or 2×
Jacobi is still useful if the force error can be driven down on
purpose. The missing piece is a knob that buys accuracy, with a known
cost, instead of a hope that more inner steps will finish the job.

## What was claimed, and what was run

The argument that "this converges" refers to method 4 below, the
two-loop procedure. That procedure has not been run.

At one fixed Hamiltonian the continuous flow
`dK/dt = −[K,[K,H]]` decreases `Tr(KH)` and, for a gapped system
started near the ground state, ends at the occupied projector of
that H. That projector is what one Jacobi diagonalization of the
same H returns. After it, the charges are replaced by the Mulliken
charges of K, H is rebuilt, and the descent is repeated. The fixed
point of that outer loop is the DFTB SCC solution. McWeeny sits at
the end of an inner descent and repairs occupations. It does not
rotate orbitals.

The code that was timed is method 1. The N-scan is method 2: the
same rotation, the cap changed from 2 to N = 1…8, the Hamiltonian
never rebuilt. The accepted step is η = 4. Blind η = 8 raises R_H
(formic 2.3×10⁻² → 6.0×10⁻²). Method 2 has no reason to arrive at a
Jacobi SCC. The scan shows it does not.

## Pseudocode

One commutator step, shared by every method. `H'` is the Hamiltonian
in the orthogonal Löwdin basis, `K` is the orthogonal projector
(`Tr(K) = Nocc`). `Δε` is the spectral width of `H'`.

```text
T = H' K
Y = K T
G = T + Tᵀ − 2 Y
s = η / Δε                         # the geometry step lifts the ||G|| cap
K ← ½ (K + Kᵀ) − s G
```

`R_H` is the size of `[K, H']`. A step that raises it is rejected.

### 1. What runs now — measured

`GpuDftb::geom_bold_dmm`. One geometry step. Cap is 2.

```text
K, q  ← previous geometry
H'    ← Xᵀ H_scc(q) X              # q is not updated inside this block
η     ← 8
n_acc ← 0
while n_acc < 2 and η ≥ 1/16:
    K_save ← K
    one commutator step at this η
    if R_H rose:
        K ← K_save
        η ← η / 2
    else:
        n_acc ← n_acc + 1
K     ← 3 K² − 2 K³                # one McWeeny
q_new ← Mulliken(X K Xᵀ)
```

A chained relaxation then does `q ← q + 0.2 (q_new − q)`. The score
against Jacobi uses `q_new` itself (α = 0): one Jacobi diagonalization
at those Mulliken charges. That throws K away. The force error is the
error of the charge vector.

### 2. Inner-cap scan — measured, and it walks off Jacobi SCC

Same as 1, with the `2` replaced by a fixed N, N = 1…8. H is not
rebuilt between the N steps. Each row of the table below is one such
run from the same carried K.

```text
while n_acc < N and η ≥ 1/16:
    … same accept / halve …
K ← McWeeny(K)
compare Mulliken(K) to Jacobi(H') and to a Jacobi SCC
```

### 3. One rebuild of H — measured

Method 1, then the charges are replaced and method 1 runs again.
Two mixes were tried.

```text
method 1
q ← q + α (q_new − q)              # α = 1 takes Mulliken in full
                                   # α = 0.2 is the damped mix
H' ← Xᵀ H_scc(q) X
method 1
```

α = 1 helps formic and does not help GC at 0.02 Å. α = 0.2 makes the
0.02 Å forces worse. Numbers are in "One rebuild of H" below.

### 4. The loop that has a convergence argument — not run

Inner descent at fixed H until a step no longer lowers `R_H`, then
one McWeeny, then the charges move and H is rebuilt. Repeat until
the charges stop. For a gapped system started near the ground state
this ends at the SCC solution, which is what a Jacobi SCC returns.

```text
K, q ← previous geometry
repeat until |q_new − q| is small:
    H' ← Xᵀ H_scc(q) X
    η  ← 8
    repeat:
        K_save ← K
        one commutator step at this η
        if R_H rose:
            K ← K_save
            η ← η / 2
            if η < 1/16: stop
        else:
            keep K
    K     ← McWeeny(K)
    q_new ← Mulliken(X K Xᵀ)
    q     ← DIIS(q, q_new)
```

Method 2 is this loop with the inner repeat cut at N and the outer
repeat never started. That cut is why the measured forces do not
approach Jacobi.

### 5. Outer loop, inner budget left at 2 — the one to try

Method 4 with the inner repeat replaced by the cap from method 1.
Stop on a charge residual of chemical size (1e-3 or 1e-4), not on a
commutator of 1e-6. This is the accuracy knob: each extra outer pass
is one more method-1 step, about a tenth of a Jacobi SCC, and the
Hamiltonian actually changes.

```text
K, q ← previous geometry
repeat until charge RMS < tol:     # tol = 1e-3 or 1e-4
    H' ← Xᵀ H_scc(q) X
    method 1                       # two accepted steps, one McWeeny
    q ← DIIS(q, Mulliken(K))
```

Method 3 with α = 1 is the first iteration of this, without DIIS and
without a third pass. Not a measurement of the whole loop.

### 6. Same outer loop, η allowed to grow — not run, only together with 5

The trust region this code inherited doubles η (cap 8) when `R_H`
falls, and halves it when `R_H` rises. Method 1 only halves, so after
the first reject every step stays at η = 4. Put the doubling back
inside method 5. Doing it inside method 2, with H never rebuilt, is
the scan already in the table: a faster walk toward the stale
projector, and a worse SCC force.

```text
if R_H rose:
    restore K
    η ← η / 2
else:
    keep K
    η ← min(8, 2 η)
```

### 7. Coarse path, Jacobi only when the answer is read — not run

Method 1 stays the geometry step. A Jacobi SCC is used when the force
has stopped falling, or for the final energy and forces. The path is
priced at method 1 (~10× Jacobi). The published geometry is priced at
one Jacobi SCC.

```text
while the relaxation is moving:
    method 1
    move the atoms along the forces
when |F| stops falling, or at the last step:
    Jacobi SCC at this geometry
```

### 8. Sparse sibling — running, same shape as method 1

B3 on the sparse solver. Two commutator steps at η = 8 (no dense-style
halving), one McWeeny, charges damped by α = 0.2. Not an electronic
solve to the projector. It is what the R10 / R14 relaxations are
doing.

```text
K ← 2 K₁ − K₀                      # dropped if that raises R_H
two commutator steps, η = 8
K ← McWeeny(K)
q ← q + 0.2 (Mulliken(K) − q)
move the atoms
```

## Comparison to Jacobi

Method 2, `test_geom_bold_nacc`, 2026-09-22. Batch 256, identical
copies, 0.02 Å along the forces. Every accepted step was η = 4.
Trace stays on the occupied count (formic 18, GC 49, diazaphen 62,
DTH 123).

Columns against Jacobi:

- `R_H` — commutator after the N steps and one McWeeny. A Jacobi
  diagonalization of the same H has `R_H = 0`.
- `dq_H` — max |q − q_Jacobi| for one Jacobi diagonalization of this
  same stale H. Zero would mean the rotation reached that projector.
- `dq_scc`, `ΔE`, force — against a Jacobi SCC at the new geometry.
  The bold energy and forces are a Jacobi diagonalization at the bold
  Mulliken charges. Jacobi SCC is the zero of these three columns.

| system | N | R_H | dq vs Jacobi(H) | dq vs Jacobi SCC | ΔE vs Jacobi SCC | force vs Jacobi SCC |
|---|---:|---:|---:|---:|---:|---:|
| formic | 1 | 7.63e-3 | 2.24e-2 | 8.31e-3 | −0.44 meV | 8.5% |
| formic | 2 | 3.56e-3 | 2.05e-2 | 9.22e-3 | −0.57 meV | 11.4% |
| formic | 3 | 2.25e-3 | 1.64e-2 | 1.04e-2 | −1.39 meV | 22.6% |
| formic | 4 | 1.73e-3 | 1.02e-2 | 1.14e-2 | −2.94 meV | 34.3% |
| formic | 5 | 1.40e-3 | 1.24e-2 | 1.38e-2 | −3.06 meV | 34.3% |
| formic | 6 | 1.17e-3 | 7.87e-3 | 1.49e-2 | −4.80 meV | 43.7% |
| formic | 7 | 9.75e-4 | 8.99e-3 | 1.61e-2 | −4.84 meV | 43.4% |
| formic | 8 | 8.22e-4 | 7.14e-3 | 1.68e-2 | −5.74 meV | 47.5% |
| GC | 1 | 8.38e-3 | 5.41e-2 | 2.95e-2 | −3.36 meV | 4.5% |
| GC | 2 | 4.03e-3 | 5.39e-2 | 2.28e-2 | −1.77 meV | 2.4% |
| GC | 3 | 2.63e-3 | 3.72e-2 | 1.51e-2 | −3.04 meV | 3.4% |
| GC | 4 | 1.96e-3 | 3.23e-2 | 1.67e-2 | −5.98 meV | 5.4% |
| GC | 5 | 1.54e-3 | 2.50e-2 | 1.80e-2 | −10.5 meV | 8.0% |
| GC | 6 | 1.23e-3 | 2.10e-2 | 2.16e-2 | −14.6 meV | 9.9% |
| GC | 7 | 9.94e-4 | 1.67e-2 | 2.58e-2 | −19.2 meV | 11.7% |
| GC | 8 | 8.08e-4 | 1.42e-2 | 2.83e-2 | −22.7 meV | 12.8% |
| diazaphen | 1 | 8.38e-3 | 5.58e-2 | 1.20e-2 | −7.80 meV | 10.7% |
| diazaphen | 2 | 4.45e-3 | 5.25e-2 | 1.05e-2 | −2.88 meV | 4.9% |
| diazaphen | 3 | 3.06e-3 | 3.94e-2 | 7.80e-3 | −2.94 meV | 4.7% |
| diazaphen | 4 | 2.35e-3 | 3.48e-2 | 1.15e-2 | −5.84 meV | 8.9% |
| diazaphen | 5 | 1.89e-3 | 2.79e-2 | 1.84e-2 | −10.8 meV | 14.1% |
| diazaphen | 6 | 1.56e-3 | 2.39e-2 | 2.23e-2 | −16.2 meV | 18.0% |
| diazaphen | 7 | 1.30e-3 | 1.97e-2 | 2.66e-2 | −22.4 meV | 21.7% |
| diazaphen | 8 | 1.10e-3 | 1.67e-2 | 2.95e-2 | −28.3 meV | 24.7% |
| DTH | 1 | 8.74e-3 | 6.36e-3 | 3.26e-3 | −0.90 meV | 0.32% |
| DTH | 2 | 5.59e-3 | 5.28e-3 | 4.00e-3 | −1.41 meV | 0.49% |
| DTH | 3 | 4.39e-3 | 5.58e-3 | 2.83e-3 | −0.94 meV | 0.36% |
| DTH | 4 | 3.59e-3 | 4.45e-3 | 4.03e-3 | −1.32 meV | 0.41% |
| DTH | 5 | 2.97e-3 | 4.00e-3 | 4.25e-3 | −1.43 meV | 0.42% |
| DTH | 6 | 2.47e-3 | 3.46e-3 | 4.66e-3 | −1.59 meV | 0.44% |
| DTH | 7 | 2.07e-3 | 3.12e-3 | 4.80e-3 | −1.76 meV | 0.45% |
| DTH | 8 | 1.74e-3 | 2.98e-3 | 4.90e-3 | −1.91 meV | 0.46% |

Speed of method 1 and of method 3 (α = 1), same 0.02 Å protocol, against
the Jacobi-warm SCC already on the throughput figures (that Jacobi row
is a 1e-6 SCC, 16 iterations on GC and larger):

| system | batch | method 1 (sys/s) | method 3, α = 1 (sys/s) | Jacobi SCC (sys/s) | method 1 / Jacobi |
|---|---:|---:|---:|---:|---:|
| formic | 1024 | 9.0×10⁵ | 3.3×10⁵ | 8.2×10⁴ | 11× |
| GC | 256 | 8.2×10⁴ | 4.1×10⁴ | 8.7×10³ | 9× |
| diazaphen | 1024 | 2.9×10⁴ | 1.5×10⁴ | 2.1×10³ | 14× |
| DTH | 256 | 3.6×10³ | 1.8×10³ | 3.4×10² | 11× |

Two things happen at once.

R_H falls at every extra step, and `dq_H` falls with it. The
rotation is moving toward the projector of the Hamiltonian it was
given. After eight steps it has not arrived: R_H is still ~10⁻³, and
the charges are still 0.007 e (formic) to 0.014 e (GC) off that
Jacobi. Each extra step removes only about a fifth of the remaining
commutator. The inner descent is real and slow.

The SCC error moves the other way on formic, GC, and diazaphen.
Formic's force goes from 8% at one step to 48% at eight. GC goes
from 2.4% at two steps to 13%. Diazaphen from 4.9% to 25%. The
projector of the old charges is a different density from the
self-consistent density. Walking toward it walks off the SCC forces.

DTH is the case where the old charges are already close to the SCC
charges (a few millielectrons). Eight steps stay inside 2 meV and
0.5% in the force. A small response to the move hides the problem.
It does not remove it.

The cap of two accepted steps is the best force in this scan for GC
and diazaphen. It is a compromise, not a converged electronic
solution. Numbers and the throughput comparison live in
`Dense_Multi_Performance.md` next to this result.

## One rebuild of H, already measured

A second pass that takes the Mulliken charges in full (α = 1),
rebuilds H, and rotates once more:

| system | move | 1 pass | 2 passes, α = 1 |
|---|---|---|---|
| formic | 0.02 Å | −0.6 meV, 11% | −0.07 meV, 4.3% |
| formic | 0.10 Å | −0.4 meV, 3.6% | −0.05 meV, 0.9% |
| GC | 0.02 Å | −1.8 meV, 2.4% | −1.6 meV, 3.0% |
| GC | 0.10 Å | −2.1 meV, 2.2% | −0.8 meV, 1.5% |

Formic improves. GC's small move does not. Mixing only 20% of the
new charges and repeating makes the 0.02 Å forces worse (formic
11% → 26%, GC 2.4% → 7% at three passes). One rebuild is not a
general fix, and a damped rebuild is the wrong direction. Throughput
of the second pass is about half of one pass, still several times
Jacobi (GC batch 256: 8.2×10⁴ sys/s one pass, 4.1×10⁴ two passes,
8.7×10³ Jacobi).

## Sparse does the same kind of thing

The sparse recipe that was kept (B2/B3) is the same shape: previous
K, two commutator steps, one McWeeny, a damped charge update, then
the geometry moves. It is not an electronic solve to the projector.

On SiH₄ a single 0.1 Å step matches a cold SCC to about 1% in the
force. A short FIRE finishes. On the nanocrystal the relaxation is
stable and does not blow up: `Tr(KS)` stays on the occupied count,
the energy falls smoothly (R10, about 3 Ha from the kicked start),
and the atoms move to sensible bond lengths. It does not settle on
the cold-SCC geometry. R10 saturates near 2×10⁻⁴ Ha/Å with the force
still bouncing; a cold SCC at an intermediate geometry is 1.5 mHa
lower. R14 after 278 steps is still setting new force lows and still
spiking to ~8×10⁻³ Ha/Å. Record:
`../Sparse_Nanocrystal_Vibrations/Warm_Geometry_DM.md` §8–§9.

That is the behaviour to expect from this step used as an optimizer.
Stable coarse motion. A floor set by the electronic error of a
truncated rotation, not by the optimizer.

## Roadblocks

1. **The convergent loop was not the thing being timed.** A
   guarantee for a descent that is run to the commutator floor, then
   repeated at a new H, says nothing about two or eight steps at the
   old H.

2. **The inner fixed point is the wrong density for forces.** Once H
   is held fixed, a better rotation is a better projector of the
   stale charges. The N-scan is that plot: R_H down, SCC force error
   up.

3. **The inner descent is slow at the step size that stays stable.**
   η = 8 is rejected on the first try and the code never raises η
   again, so every later step is η = 4. Eight of those leave R_H at
   ~10⁻³. Reaching even the stale projector inside a useful budget
   is not demonstrated.

4. **The accept test is R_H, the quantity the flow decreases is
   Tr(KH).** A step can shrink the commutator without being a descent
   of the band energy of this H. Tr(KH) was not logged against N, so
   it is open whether the accepted steps are the flow that the
   convergence argument is about.

5. **The only outer experiment is not monotonic.** One α = 1 rebuild
   helps formic and the larger GC move. It does not help GC at
   0.02 Å. α = 0.2 repeats hurt. There is no measured DIIS loop of
   these projectors.

6. **The reported force is Jacobi at the Mulliken charges.** It is
   the exact force of that charge vector. Improving it means moving
   the charges toward the SCC charges. It does not mean a better
   force kernel on a bad K.

## What to try, in that order

The pseudocode above is the list. Speed is against the Jacobi-warm
SCC in the table (8–16 diagonalizations, charge RMS 1e-6). A result
at 2× Jacobi is still useful.

1. **Method 5.** Outer charge loop, inner cap kept at 2. Each extra
   pass rebuilds H. Three to six passes of a step that is ~10×
   cheaper than a Jacobi SCC would land near 2–5× Jacobi. Method 3
   with α = 1 is one pass of this, not the loop.
2. **Before changing the step, log Tr(KH) on method 2.** The accept
   test is R_H. The flow decreases Tr(KH). If the band energy of the
   fixed H is not falling while R_H falls, the trust test is
   accepting the wrong event. One reduction of a product that already
   exists.
3. **Method 6 only inside method 5.** Doubling η after a success is
   how the inner loop gets off η = 4. Inside method 2, with H never
   rebuilt, a larger step is a faster walk toward the stale projector,
   which the table shows makes the SCC force worse.
4. **Method 7** if method 5 is still too expensive for a long
   relaxation: coarse steps for the path, one Jacobi SCC when the
   force stalls or when the energy is read. This is what a cold SCC
   already does when it is dropped onto a warm sparse geometry.
5. **Leave method 2's cap alone.** Driving R_H from 4×10⁻³ toward
   10⁻⁶ at the old charges is the direction the force error grew.

Closed, and not a hypothesis to reopen: purification alone (McWeeny
or TC2) repairing a rotated kernel; η = 8 with no reject; scaling K
to fix the trace; extrapolating K along the SCC history; transporting
orbitals by the cross-overlap (commutator ~0.7 on an atom-centered
basis). Those are in `Warm_Geometry_DM.md` §5 and in the dense notes
cited there.
