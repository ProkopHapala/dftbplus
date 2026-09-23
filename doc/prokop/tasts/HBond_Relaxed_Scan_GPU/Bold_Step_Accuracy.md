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

Checked against DFTB+ (same matsci-0-3 files, plain SCC), 2026-09-22.
The pristine nanocrystal is bulk diamond, Si–Si = 2.352 Å. A 2-atom
DFTB+ lattice optimisation of that cell ends at Si–Si = 2.244 Å
(a = 5.183 Å). The bold R10 relaxation ends at 2.234–2.247 Å. The
shrink is the parametrization.

Si₅H₁₂ cut from that crystal (complete mask, 17 atoms): the start
matches DFTB+ (E = −11.1522 Ha, max |F| = 0.067 Ha/Å). DFTB+
conjugate-gradient finishes at Si–Si = 2.2350 Å, Si–H = 1.4836 Å,
E = −11.18835 Ha. The bold FIRE run finishes at 2.2348 Å, 1.4834 Å,
E = −11.18829 Ha, max |F| = 7×10⁻⁶. On a complete mask the truncated
step does not move the minimum.

A cold SCC at the bold R10 geometry is 1.5 mHa lower
(E = −314.0999 vs −314.0984) and its max |F| is 3.0×10⁻⁴, the same
size as the bold floor (2.2×10⁻⁴). The truncated rotation leaves an
energy offset and a force noise of a few 10⁻⁴. It is not a second,
wrong lattice.

The same check on the other tables (2026-09-23). Particles match the
silicon R10 atom count (330), not the radius: `c_equiv_R10.xyz` is
C₁₉₆H₁₃₄, C–C = 1.545 Å, C–H = 1.090 Å. DFTB+ bulk nearest neighbours,
from the same 1.545 Å (carbon) or 2.352 Å (silicon) start:

| SK | bulk NN (Å) | C₅H₁₂ / Si₅H₁₂ | 330-atom median |
|---|---:|---:|---:|
| 3ob-3-1 C | 1.559 | 1.543 | 1.567 (1.514–1.612) |
| mio-1-1 C | 1.541 | 1.526 | 1.547 (1.504–1.588) |
| matsci-0-3 C | 1.551 | 1.535 | 1.552 (1.523–1.577) |
| pbc-0-3 C | 1.542 | 1.528 | — |
| pbc-0-3 Si | 2.364 | 2.335 | run stopped |
| matsci-0-3 Si | 2.244 | 2.235 | 2.243 |

C₅H₁₂ bold FIRE matches the DFTB+ bond to 0.0004 Å on all four carbon
tables. The 330-atom carbon particles (no kick) reach max |F| ~
10⁻⁵. The median C–C stays on that table's bulk bond. The spread is
surface relaxation, not the uniform 4.6% contraction of matsci
silicon. Warm step ~120 ms, because the silicon cutoffs
(r_trunc = 5.45 Å, r_K = 12 Å) cover this smaller lattice and the
mask is nearly complete (nnz_K = 102712 vs 57736 on silicon).

H, S and the density kernel are different masks. `sparse_hs_decay`
sets r_trunc from the last grid point with |H| or |S| > 10⁻⁴.
`RUST_DFTB_R_K` sets the density kernel (default 12 Å, not scaled
from r_trunc). `RUST_DFTB_R_Z` sets the overlap inverse (default =
r_K). On the 330-atom pbc particle, H/S stayed at 5.42 Å
(nnz = 27 346) and r_Z at 12 Å (nnz = 57 736) while r_K was varied,
eight geometry steps, no kick:

| r_K (Å) | nnz_K | Tr(KS) over 8 steps |
|---:|---:|---|
| 12 | 57 736 | runs away, stop at step 7 (459.30) |
| 13 | 66 286 | peaks at 459.064, back to 459.047 |
| 14 | 75 364 | stays within 0.021 |
| 16 | 90 838 | stays within 0.005 |
| 23 (every pair) | 108 900 | stays on 459.00003 |

Deriving r_K from the short pbc table (r_K = 9.3 Å) made the leak
faster, not slower. The Hamiltonian mask was never the missing piece.

matsci-0-3 silicon is a consistent Hamiltonian and a poor description
of elemental silicon. DFTB+ and the bold solver agree: bulk
Si–Si = 2.244 Å, Si₅H₁₂ = 2.235 Å, the 330-atom particle = 2.243 Å,
from a 2.352 Å start. That minimum is real for this table and 4.6%
shorter than the diamond bond. The same 8×8×8 band structure has
VBM −4.04 eV and CBM +1.01 eV, a 5.05 eV gap. Experimental silicon
is about 1.1 eV. The file says why: Frenzel, Oliveira, Jardillier,
Heine and Seifert, TU Dresden 2004–2009, “for materials science
simulations.” The Si table is paired with O, N, C, H, Al, P, Na, Cu.
The repulsive note says it was not fitted to atomization or reaction
energies. Elemental Si bands and the diamond bond were not the target.

Si–H sets on disk, and what each file claims:

| set | Si–H | what it is for | this solver |
|---|---|---|---|
| pbc-0-3 | yes | Sieck, Paderborn 2000. File: “OK for bulk silicon and clusters.” Own caveat: gap wrong (direct) because the basis is minimal. d-onsite tweaked for the Si–O–Si angle in α-quartz. | s+p. Bulk Si–Si = 2.364 Å (exp. 2.352). Gap 1.68 eV. |
| matsci-0-3 | yes | Inorganic materials (Si with O, N, C, …), not elemental Si. | s+p. Do not use for a Si gap or a Si–Si bond one intends to believe. |
| siband-1-1 | yes | Markov et al., IEEE TED 62, 696 (2015). Bands, dielectric response, transport in oxidised Si. File: “DUMMY SPLINE: DO NOT USE FOR RELAXATION.” Basis 3s3p3d. | Not this path. BSR4 is four orbitals (s+p). |

hyb-0-2 is Si–Si only. 3ob and mio have no silicon. For an Si–H particle in this code, pbc-0-3 is the set whose own documentation and our bulk bond both say “silicon.”

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

## Coupled rounds, one commutator each — measured 2026-09-23

A round is one accepted commutator step plus one McWeeny. It is not
two matrix products, and it is not the earlier second pass (that pass
was a whole two-commutator block, repeated). Between rounds the
Hamiltonian is rebuilt. Batch 256, the same 0.02 Å step, against a
Jacobi SCC. `N = 1` is one round on the carried Hamiltonian.

Taking the new charges in full (`α = 1`) before the next round:
the second round is worse than the first on every molecule. The third
comes back. The updates overshoot.

| system | N | α = 1, ΔE | α = 1, force | α = 0.5, ΔE | α = 0.5, force |
|---|---:|---:|---:|---:|---:|
| formic | 1 | −0.44 meV | 8.5% | −0.44 meV | 8.5% |
| formic | 2 | −2.09 meV | 24.8% | −0.44 meV | 6.8% |
| formic | 3 | −0.08 meV | 4.5% | −0.008 meV | 1.3% |
| formic | 4 | −0.31 meV | 10.2% | −0.024 meV | 1.8% |
| GC | 1 | −3.4 meV | 4.5% | −3.4 meV | 4.5% |
| GC | 2 | −9.3 meV | 7.7% | −3.8 meV | 4.8% |
| GC | 3 | −1.0 meV | 2.4% | −0.90 meV | 2.2% |
| GC | 4 | −2.2 meV | 3.7% | −0.68 meV | 2.4% |
| diazaphen | 1 | −7.8 meV | 10.7% | −7.8 meV | 10.7% |
| diazaphen | 2 | −10.4 meV | 15.7% | −4.8 meV | 9.8% |
| diazaphen | 3 | −2.7 meV | 6.5% | −2.2 meV | 6.3% |
| diazaphen | 4 | −4.1 meV | 9.6% | −1.8 meV | 5.8% |
| DTH | 1 | −0.91 meV | 0.32% | −0.91 meV | 0.32% |
| DTH | 2 | −2.5 meV | 0.37% | −1.4 meV | 0.38% |
| DTH | 3 | −0.37 meV | 0.17% | −0.23 meV | 0.14% |
| DTH | 4 | −0.82 meV | 0.29% | −0.36 meV | 0.20% |

With the half-mix, round 2 beats round 1 only on formic and diazaphen,
and only by a little. Round 3 is the one that drops the error on all
four (formic 8.5% → 1.3% and −0.008 meV; DTH 0.32% → 0.14%). Round 4
does not continue that drop, except on diazaphen. One round stays the
fast step. Two rounds are not the accuracy knob. Three half-mixed
rounds are the best point this scan measured, at about three times
the cost of one round.

## Charge DIIS around one commutator — measured 2026-09-23

Each pass is one accepted commutator, one McWeeny, then the existing
charge DIIS (`α = 0.3` only on the first pass, real DIIS after that).
Batch 256, the same 0.02 Å step, against a Jacobi SCC. The force is
still a Jacobi diagonalization at the Mulliken charges of that pass.

| system | passes | rms | ΔE | force |
|---|---:|---:|---:|---:|
| formic | 1 | 1.5e-2 | −0.44 meV | 8.5% |
| formic | 6 | 1.9e-3 | −0.07 meV | 4.4% |
| GC | 1 | 1.6e-2 | −3.4 meV | 4.5% |
| GC | 6 | 1.6e-3 | −0.26 meV | 1.3% |
| diazaphen | 1 | 1.3e-2 | −7.8 meV | 10.7% |
| diazaphen | 6 | 4.5e-3 | −0.21 meV | 1.4% |
| DTH | 1 | 4.7e-3 | −0.91 meV | 0.32% |
| DTH | 6 | 8.8e-4 | −0.12 meV | 0.09% |

The residual falls, and the force is better at pass 6 than at pass 1,
but not by a factor every pass. Formic's force bounces between 4% and
10%. `R_H` stays ~10⁻³, the same size as the charge residual, so the
mixer is fitting an inexact map.

Second run: on a pass where `R_H > 0.3 ×` the charge rms, one more
commutator on that same Hamiltonian before DIIS. `comm` is the total
number of commutator steps.

| system | passes | comm | rms | ΔE | force |
|---|---:|---:|---:|---:|---:|
| formic | 1 | 2 | 1.5e-2 | −0.56 meV | 11.5% |
| formic | 4 | 6 | 2.8e-3 | −0.021 meV | 1.9% |
| formic | 5 | 8 | 6.7e-4 | −0.005 meV | 0.73% |
| formic | 6 | 10 | 8.1e-5 | +0.020 meV | 0.30% |
| GC | 2 | 3 | 1.5e-2 | −1.1 meV | 1.8% |
| GC | 3 | 4 | 1.4e-2 | −7.0 meV | 7.5% |
| GC | 5 | 8 | 1.4e-3 | −0.15 meV | 1.3% |
| GC | 6 | 9 | 2.3e-3 | −0.29 meV | 1.5% |
| diazaphen | 2 | 3 | 1.1e-2 | −1.3 meV | 2.7% |
| diazaphen | 3 | 4 | 8.4e-3 | −8.6 meV | 13.4% |
| diazaphen | 5 | 8 | 1.7e-3 | −0.35 meV | 2.2% |
| diazaphen | 6 | 10 | 1.5e-3 | −0.48 meV | 2.6% |
| DTH | 1 | 2 | 6.1e-3 | −2.8 meV | 0.67% |
| DTH | 4 | 8 | 5.5e-4 | −0.078 meV | 0.09% |
| DTH | 6 | 12 | 4.5e-4 | −0.005 meV | 0.03% |

Formic from pass 3 and DTH from pass 1 do drop by a factor of about
two to four each pass, down to 0.3% and 0.03%. GC and diazaphen jump
the wrong way on pass 3, the first pass with a real DIIS
extrapolation, and the later passes only get back to 1–3%. One
commutator plus DIIS, without the extra rotation, was better on
diazaphen at pass 6 (1.4%). The default is still one round. This is
not yet a uniform exponential approach to Jacobi.

Throughput of that six-pass recipe (`test_gpu_bold_diis_sweep`,
median of 3, powers of two through 1024). The dark-purple series on
`throughput_{formic,GC,diazaphen,DTH}.png`. Against the Jacobi-warm
row already on those figures:

| system | batch | 6-pass DIIS | one bold step | Jacobi SCC |
|---|---:|---:|---:|---:|
| formic | 1024 | 1.0×10⁵ (1.2×) | 9.0×10⁵ | 8.2×10⁴ |
| GC | 256 | 1.0×10⁴ (1.2×) | 8.2×10⁴ | 8.7×10³ |
| diazaphen | 1024 | 3.8×10³ (1.8×) | 2.9×10⁴ | 2.1×10³ |
| DTH | 256 | 4.4×10² (1.3×) | 3.6×10³ | 3.4×10² |

Six passes with about ten commutators spends the factor the single
step had. What remains is 1.2–1.8× Jacobi, not 10×.

## Carried history, and rejecting a rising residual — 2026-09-23

Both measured at batch 256 with the same second-commutator rule.
`n_filled = 10` after the initial SCC, so there was a full history
to carry. `set_coords` clears it; the carry run puts that snapshot
back.

Carrying it makes every molecule worse, and further passes keep
walking off. The stored residuals are the tail of an SCC that had
already reached 10⁻⁶, so they are not a charge response. Formic's
force goes 11% → 42% by pass 6. GC goes 2.4% → 14% (−28 meV).
Diazaphen goes 4.8% → 25%. DTH stays near 0.5% and the residual
never leaves 6×10⁻³, where a fresh history reaches 0.03%.

Rejecting a mix that raises the residual does not fire on the
GC / diazaphen spike. At that pass the residual is still falling
(GC 1.52×10⁻² → 1.45×10⁻²) while the force goes 1.8% → 7.5% and
the charge error versus Jacobi SCC goes up. The mixer is doing
what it was asked. The one-commutator map's residual is not the
distance to the SCC charges on those two molecules. A fresh DIIS
history remains the better of the two, and it is the curve already
timed.

## DIIS withheld until `R_H < 0.1 ×` charge rms — 2026-09-23

Up to three commutators on the current Hamiltonian. If the commutator
is still larger than a tenth of the charge residual, the pass takes
`α = 0.5` and does not extrapolate. Batch 256, same 0.02 Å step.

DIIS barely runs. Formic, diazaphen, and DTH never meet the gate.
GC meets it on some later passes. The trajectory is the damped mix.
There is no pass-3 spike.

| system | passes | commutators | ΔE | force |
|---|---:|---:|---:|---:|
| formic | 1 | 3 | −1.4 meV | 23% |
| formic | 5 | 15 | +0.025 meV | 0.17% |
| formic | 6 | 18 | +0.016 meV | 0.15% |
| GC | 1 | 3 | −3.0 meV | 3.4% |
| GC | 5 | 13 | −0.073 meV | 0.78% |
| GC | 6 | 16 | +0.005 meV | 0.38% |
| diazaphen | 4 | 12 | −0.018 meV | 0.20% |
| diazaphen | 5 | 15 | −0.010 meV | 0.12% |
| DTH | 3 | 9 | −0.074 meV | 0.07% |
| DTH | 4 | 12 | −0.001 meV | 0.03% |

All four end under 0.5% in the force. Formic's first pass is the
three-commutator step on the old charges (23%), and passes 5–6 bring
it back. About 12–18 commutators, against ~10 for the six-pass DIIS
recipe that was only 1.2–1.8× Jacobi and still at 1–3% on GC and
diazaphen. This is the most accurate of the geometry steps measured.
It is not the fast one.

## One-GEMM `H'` and one-GEMM charges — 2026-09-23

Per geometry, once: `C = Xᵀ S` and `H0' = Xᵀ H0 X` (three products).
Each charge update is then one product, not two:

```
T_μk = V_atom(μ) · C_kμ
M    = Xᵀ T
H'   = H0' + ½(M + Mᵀ)
```

Charges, also one product. `Y = K C`, then
`p_μ = 2 Σ_i X_μi Y_iμ` and sum onto the atom. The 2 is the
closed-shell density `D = 2 X K Xᵀ`. `X` is not assumed symmetric;
the stored `Xᵀ` is what the products use. `RUST_DFTB_BOLD_CHEAP=0`
keeps the two-product form. The bold step uses the one-product form
otherwise. The SCC loop is unchanged.

Same carried `K`, same 0.02 Å step, batch 256. Max `|Δq|` against the
two-product step, electron count, and `R_H`:

| system | `|Δq|` | Ne (two / one) | `R_H` |
|---|---:|---:|---:|
| formic | 9.5×10⁻⁷ e | 36 / 36 | 3.564×10⁻³ both |
| GC | 1.4×10⁻⁶ e | 98 / 98 | 4.027×10⁻³ both |
| diazaphen | 1.9×10⁻⁶ e | 124 / 124 | 4.448×10⁻³ both |
| DTH | 2.1×10⁻⁶ e | 246 / 246 | 5.588×10⁻³ both |

Trace and the number of accepted steps match. This is the same step.

Wall time, mean of five for one cap-2 step, and one shot of the
six-pass gate (up to three commutators, DIIS only if
`R_H < 0.1 ×` rms). The gate repeats the rebuild; that is where two
fewer products per call show up.

| system | one step, two → one | gate, two → one |
|---|---|---|
| formic | 0.39 → 0.41 ms | 7.0 → 6.8 ms (1.03×) |
| GC | 2.55 → 2.43 ms | 39 → 33 ms (1.18×) |
| diazaphen | 7.23 → 6.85 ms | 122 → 105 ms (1.16×) |
| DTH | 49 → 52 ms | 844 → 657 ms (1.29×) |

A single step is unchanged at this sample size. The trust-region
read of `R_H` is still a host sync inside every attempt, and that
sync is a large part of one call. The gate, which issues that call
many times, is 1.16–1.29× on GC, diazaphen, and DTH. DTH's gate at
batch 256 is 256 / 0.657 s ≈ 390 sys/s, next to Jacobi-warm (~340
sys/s at this batch). The fast inaccurate step stays the one
commutator round; this did not move it.

## Host reads on the bold step — 2026-09-23

Two reads were inside every geometry step. Each `read_buffer` finishes the queue.

- `R_H` of replica 0, before the step and after every attempt. That is the trust decision (keep the step, or restore `K` and halve η).
- The whole `K` of replica 0, after McWeeny, only to sum the diagonal for `Tr(K)`.

`RUST_DFTB_BOLD_DIAG=1` keeps both. The production step does not. η, the accept count, and the restore flag live in an 8-float device buffer. The host enqueues `max_acc + 1` attempts (one rejected halving, then the accepted steps — the 0.02 Å step rejects η = 8 once and then accepts at η = 4). The checkpoint copy is a queued device copy. The restore is a kernel gated by the device flag, so a rejected attempt does not need the host. One read of those 8 floats happens after the charge products are already queued. If that budget is exhausted while η is still live, the step fails and names the diagnostic switch. It does not stop short.

Same carried `K`, same cap of 2, batch 256. Charges, `R_H`, trace, and the accept count match the diagnostic loop on all four molecules (`|Δq| = 0`).

One cap-2 step, mean of five. "prod" is the device trust. "diag" is the host read every attempt.

| system | prod | diag |
|---|---:|---:|
| formic | 0.36 ms | 0.56 ms |
| GC | 3.04 ms | 3.06 ms |
| diazaphen | 9.23 ms | 7.89 ms |
| DTH | 60.4 ms | 55.7 ms |

Formic is faster. GC is the same. On diazaphen and DTH the diagnostic loop is a few milliseconds faster: the per-attempt read is one float after the GPU has already gone idle, and the device path adds the flag kernels plus a restore copy. The full-matrix read is gone from the production step either way.

## SCC energy on accepted steps — 2026-09-23

Logged, not used as the accept test. Four coupled rounds: one accepted
commutator, one McWeeny, then the Mulliken charges become the next
Hamiltonian (`α = 1`). Batch 256, the same 0.02 Å step. Repulsion is
omitted; it does not change at fixed geometry.

`E_dens = 2 Tr(K H'₀) + ½ Δqᵀ γ Δq`. The 2 is the closed-shell density
`D = 2 X K Xᵀ`, so `2 Tr(K H'₀) = Tr(D H0)`. The review wrote the same
expression without that 2 (`E_chat`). `R_H` is before the rotation and
after McWeeny, both against the Hamiltonian of that round.

| system | step | R_H | ΔE_dens | ΔE_chat |
|---|---:|---|---:|---:|
| formic | 1 | 2.3×10⁻² → 7.6×10⁻³ | −232 meV | −123 meV |
| formic | 2 | 8.1×10⁻³ → 3.3×10⁻³ | −22 meV | −22 meV |
| formic | 3 | 4.0×10⁻³ → 2.0×10⁻³ | −4.2 meV | **+4.8 meV** |
| formic | 4 | 2.4×10⁻³ → 1.3×10⁻³ | −1.3 meV | −3.8 meV |
| GC | 1 | 3.2×10⁻² → 8.4×10⁻³ | −1050 meV | −424 meV |
| GC | 2 | 8.6×10⁻³ → 3.7×10⁻³ | −78 meV | −86 meV |
| GC | 3 | 4.2×10⁻³ → 2.5×10⁻³ | −19 meV | **+40 meV** |
| GC | 4 | 2.6×10⁻³ → 1.6×10⁻³ | −8.3 meV | −24 meV |
| diazaphen | 1 | 3.2×10⁻² → 8.4×10⁻³ | −1177 meV | −591 meV |
| diazaphen | 4 | 3.0×10⁻³ → 2.1×10⁻³ | −15 meV | −9.3 meV |
| DTH | 1 | 3.5×10⁻² → 8.7×10⁻³ | −2830 meV | −1425 meV |
| DTH | 4 | 3.9×10⁻³ → 2.9×10⁻³ | −54 meV | −27 meV |

`E_dens` falls on every accepted step, on all four molecules.
`R_H` falls with it. Zero disagreements in 16 steps. The review
formula without the factor 2 rises on formic step 3 and GC step 3
while `R_H` is still falling; that sign is the missing 2, not the
trust test. Item 3 (accept on the energy, double η) is not justified.
The accept test is unchanged.

## Shadow potential — 2026-09-23

This is not in the solver. The run evaluates Niklasson's first-level
formula on one 0.02 Å step and compares it to a full SCC Jacobi.
The inverse Jacobian is a central difference, two diagonalizations
per atom. That is a measurement of the formula. A carried approximate
inverse is the version that would be cheap, and it has not been built.

Zeroth level: one Jacobi of `H(q)` at the carried charges. The charge
is not updated. Batch 1.

| system | finite-difference of the force | force vs SCC | energy vs SCC |
|---|---:|---:|---:|
| formic | 7.8% | 65% | −11 meV |
| GC | 0.16% | 15% | −40 meV |
| diazaphen | 0.21% | 25% | −30 meV |
| DTH | 0.62% | 0.57% | −4.5 meV |

First level: one Newton step on that charge residual, then one Jacobi
of the updated Hamiltonian. Same geometries.

| system | charge rms | energy vs SCC | force vs SCC |
|---|---|---|---|
| formic | 2.7×10⁻² → 2.1×10⁻⁴ | −11 meV → +0.016 meV | 65% → 0.16% |
| GC | 3.6×10⁻² → 2.8×10⁻⁴ | −40 meV → +0.029 meV | 15% → 0.15% |
| diazaphen | 2.5×10⁻² → 2.2×10⁻⁴ | −30 meV → −0.011 meV | 25% → 0.04% |
| DTH | 7.3×10⁻³ → 1.1×10⁻⁵ | −4.6 meV → +0.009 meV | 0.57% → 0.00% |

Performance, batch 256, mean of 3, the displaced geometry. "2 Jacobi"
is two full diagonalizations, the electronic cost of the zeroth level
plus the first level when the inverse Jacobian is already known.
"FD estimate" multiplies one diagonalization by `2 N_atom + 2`, which
is what the accuracy test spent building the Jacobian.

| system | SCC | 2 Jacobi | FD-Jacobian estimate |
|---|---|---|---|
| formic | 8 steps, 3.94 ms, 6.5×10⁴ sys/s | 1.12 ms, 3.5× | 12 ms |
| GC | 16 steps, 20.5 ms, 1.3×10⁴ sys/s | 5.68 ms, 3.6× | 170 ms |
| diazaphen | 16 steps, 69.4 ms, 3.7×10³ sys/s | 16.5 ms, 4.2× | 710 ms |
| DTH | 16 steps, 473 ms, 540 sys/s | 268 ms, 1.8× | 23 s |

The SCC step count is launched in chunks of 8, so 8 and 16 are the
number of steps that were enqueued. With the Jacobian known, two
diagonalizations are 1.8–4.2× a full SCC. Building the Jacobian by
finite differences is slower than SCC on every molecule except formic.

## What to try, in that order

Review of the proposals in
[`Bold_DM_Purify.chat.md`](Bold_DM_Purify.chat.md). Speed is against
the Jacobi-warm SCC in the table above. A result at 2× Jacobi is
still useful. More inner steps at a frozen `H` stay closed.

1. **Coupled bold-N.** After every accepted commutator step, rebuild
   `H'` from the new Mulliken charges and rotate against that `H'`.
   `N = 2` uses the same two rotations as now. `N = 1…8` is the plot:
   force error against Jacobi SCC should fall if the stale `H` was
   the error. McWeeny once at the end. Use the existing rebuild.
2. **Log `E_SCC` and `Tr(KH')` on those accepted steps.** Measured
   above (2026-09-23). `E_dens` and `R_H` fall together. The accept
   test stays on `R_H`.
3. **Accept on `E_SCC`, and double η after a success.** Not justified.
   The logged energy does not rise on an accepted step.
4. **Stop on charge RMS** (1e-3, then 3e-4, then 1e-4) once the
   force curve slopes the right way. Fit any `R_H ≲ c ‖Δq‖` rule from
   that plot. If the charges oscillate, wrap the existing charge DIIS
   around `q`.
5. **Persistent Broyden across geometry steps** only if item 4 needs
   fewer passes. The mixer at one geometry is the DIIS already in the
   solver.
6. **One-GEMM `q` and one-GEMM `H'`.** Measured above
   (2026-09-23). Same charges. The repeated gate is 1.16–1.29× on
   the larger molecules. A single step is not.
7. **Shadow potential last.** The zeroth level (one Jacobi at the
   carried charges) is 15–65% off the SCC force on formic, GC, and
   diazaphen. The first level, one exact Newton step on that charge
   residual and one more Jacobi, lands within 0.03 meV and 0.2% of
   SCC. The Jacobian in that test is a finite difference. A carried
   approximate inverse is the version that could be cheap.

Closed, and not a hypothesis to reopen: purification alone (McWeeny
or TC2) repairing a rotated kernel; η = 8 with no reject; scaling K
to fix the trace; extrapolating K along the SCC history; transporting
orbitals by the cross-overlap (commutator ~0.7 on an atom-centered
basis). Those are in `Warm_Geometry_DM.md` §5 and in the dense notes
cited there.
