https://chatgpt.com/share/6ab38844-3d34-83eb-aef4-cfcc5af9f74c

---

# USER

OK, we tried and tsted new method which does just 2 iterations rather than some rorrible 32-40 iterations cycle as before because we were not able to compete with jacobi solver with 32-40 iterations. New "bold" method is super fast, but seem to be not very accurate (1-4% on forces) and we do not know how to systematically imporve it

I attached detailed report, and there is link to the repo and commit:
https://github.com/ProkopHapala/dftbplus/commit/359ee76cec82baefcc309bb26c6473ce33d447d5


---

# Chat GPT 5.6 sol

I read the new report carefully and checked the linked GitHub commit. One practical note first: the linked SHA `359ee76...` is still the latest committed revision I can see and is dated September 20, while `Bold_Step_Accuracy.md` is dated September 22. I do **not** see `geom_bold_dmm` in that committed SHA, so I can review the mathematics/algorithm from the report and the underlying `dmm_update_batched` machinery, but not the exact newest wrapper code.

The report changes my interpretation quite substantially. I don't think the bold method is “slightly inaccurate DMM that needs a better step size.” It is doing something more interesting:

> **the two-step bold operation is an extremely cheap inexact SCF map.**

And that immediately explains both its impressive speed and the apparently paradoxical observation that **more DMM steps make the SCC forces worse**.

Your scan shows exactly that: increasing \(N\) steadily lowers the fixed-\(H\) commutator and generally moves closer to the projector of that fixed Hamiltonian, while the error relative to the *self-consistent* force grows badly for formic, GC, and diazaphen. 

That is not a defect of the commutator descent. It is a consequence of solving the wrong intermediate problem too accurately.

## The key distinction: there are two maps

For an input charge \(q\), define the exact SCF map

$$
F(q)=q\!\left[P(H(q))\right],
$$

where \(P(H)\) is the exact occupied projector.

SCC solves

$$
q^*=F(q^*).
$$

Your bold operation instead evaluates something like

$$
F_{\rm bold}(q,K)
=
q\!\left[B_2(K,H(q))\right],
$$

where \(B_2\) means two cheap commutator rotations plus retraction.

Now consider what the \(N\)-scan does. As \(N\rightarrow\infty\),

$$
B_N(K,H(q))
\rightarrow P(H(q)),
$$

so

$$
F_{\rm bold,N}(q,K)\rightarrow F(q).
$$

But **\(F(q)\) is not the SCC solution unless \(q=q^*\).**

So when you hold the old charge fixed and make the inner solve better, you are simply moving more accurately toward

$$
F(q_{\rm old}),
$$

which can easily lie *past* the new fixed point \(q^*\).

That is exactly what your force table is showing.

This also explains why two steps happen to be good: they are acting like an accidental damping of the SCF map.

The cap at two is not “the electronic accuracy.” It is effectively an **implicit mixing parameter**.

---

# The systematic improvement is therefore not “3, 4, 5 DMM steps”

I think that avenue is closed by your data.

The systematic accuracy knob has to be an **outer self-consistency correction**, with the Hamiltonian changing.

Your method 5 is already pointing in the correct direction. 

But I would modify it in two important ways.

## 1. Do not use simple charge mixing

You tested

$$
q\leftarrow q+\alpha(q_{\rm new}-q)
$$

with \(\alpha=1\) and \(0.2\).

The fact that \(\alpha=0.2\) gets worse is not surprising. SCC-DFTB's response is a matrix problem in atomic charge space; one scalar damping parameter is a poor inverse Jacobian.

Standard DFTB already treats the SCC problem this way: the conventional SCC procedure uses modified Broyden, i.e. an approximate inverse electronic Hessian built from previous iterations. ([PubMed Central (PMC)][1])

And this is almost free compared with what you're doing: there may be 20–100 atomic charges versus an \(86\times86\) orbital matrix.

So I would make the outer problem explicitly

$$
r(q)=F_{\rm bold}(q)-q
$$

and update

$$
\boxed{
q_{k+1}=q_k-B_k r_k
}
$$

with a Broyden/Anderson/DIIS approximation \(B_k\) to the inverse Jacobian.

Even better: **carry that Broyden information between nearby geometry steps.**

The charge-response Jacobian should vary smoothly with a 0.02–0.1 Å displacement. So the previous geometry gives you not only \(K\) and \(q\), but an approximate inverse SCC Hessian.

That information may be more useful than another four DMM rotations.

---

# 2. Treat the projector solve as an *inexact inner solve*

There is a standard numerical principle hiding here.

You do not need

$$
[H(q_k),K]=0
$$

to \(10^{-6}\) while

$$
\|r(q_k)\|\sim10^{-2}.
$$

That is massive oversolving.

But when the charge residual gets down to \(10^{-4}\), an inner projector error of \(10^{-2}\) may become the limiting error.

So I would use an adaptive condition conceptually like

$$
\boxed{
R_H
\lesssim
c\,\|r_q\|
}
$$

rather than a fixed “2 DMM steps forever” or “converge \(R_H\) completely every time.”

For example:

```text
large charge residual:
    1–2 bold steps

medium charge residual:
    2–3 bold steps

near convergence:
    enough steps that R_H is below ~0.1–0.3 × charge residual
```

The exact normalization needs empirical calibration because \(R_H\) and charge RMS have different units/scales, but the principle matters.

This gives you the accuracy ladder you were missing:

$$
\boxed{
\text{outer charge tolerance}
+
\text{inner accuracy proportional to outer residual}.
}
$$

That converges to the true SCC solution without wasting 30 fixed-\(H\) iterations at every outer point.

---

# There is an even cleaner formulation I would test before that

I think this may be the most interesting experiment.

Your current method does

```text
H(q0)
DMM
DMM
McWeeny
q1
```

Instead test

```text
H(q0)
DMM
q1 = q(K1)
H(q1)

DMM
McWeeny
q2 = q(K2)
```

That is still only **two commutator steps total**.

The difference is that the second step uses the Hamiltonian corresponding to the density after the first step.

I would call this **coupled bold-2**.

Why do I like this so much?

Because for SCC-DFTB at fixed geometry the energy is schematically

$$
E[K]
=
\operatorname{Tr}(K H'_0)
+
\frac12\Delta q[K]^T\gamma\Delta q[K]
+
E_{\rm rep},
$$

and its derivative with respect to \(K\) is the **current SCC Hamiltonian**

$$
\frac{\partial E}{\partial K}=H'[q(K)].
$$

Therefore if at every electronic step you use

$$
G(K)
=
[K,[K,H'[q(K)]]],
$$

then

$$
\boxed{
K\leftarrow K-sG
}
$$

is a genuine Riemannian gradient step of the **full self-consistent SCC energy**, not of a stale band energy.

For an exactly idempotent \(K\),

$$
\left.
\frac{dE(K-sG)}{ds}
\right|_{s=0}
=
-\|[K,H'[K]]\|_F^2
\le0.
$$

That is a fundamentally better object.

Now if you perform step 3, step 4, step 5, they really do approach the SCC stationary state.

Your current \(N\)-scan has no such property because \(H\) remains frozen. The report correctly identifies that as the central roadblock. 

---

## This would also fix the trust criterion

Right now you accept a step because \(R_H\) decreases.

But as the report notes, the quantity the continuous flow actually guarantees to decrease is the energy, not necessarily the commutator norm. 

For the coupled method I would use **actual SCC electronic energy** as the trust condition.

For a trial \(K'\):

1. compute \(q[K']\);
2. evaluate

   $$
   E_{\rm SCC}[K']
   =
   \operatorname{Tr}(K'H'_0)
   +\frac12\Delta q^T\gamma\Delta q;
   $$
3. accept if

   $$
   E_{\rm SCC}[K'] < E_{\rm SCC}[K]
   $$

   perhaps with a weak Armijo margin;
4. otherwise halve \(s\).

After several successful steps, let \(s\) grow again.

So method 6 becomes much more principled:

```text
if full SCC energy decreased:
    accept
    eta = min(eta_max, 1.5–2 eta)
else:
    reject
    eta *= 0.5
```

not “does the stale-H commutator decrease?”

---

# Very useful consequence: there need not be an inner/outer loop at all

The nested view is

```text
for q:
    solve K for H(q)
    update q
```

But since your purpose is specifically to avoid solving the inner problem, I would instead think of one sequence

$$
(K_0,q_0)
\rightarrow
(K_1,q_1)
\rightarrow
(K_2,q_2)
\rightarrow\cdots
$$

with

$$
q_k=q[K_k],
$$

$$
H_k=H[q_k],
$$

$$
K_{k+1}
=
R\left(
K_k-s_k[K_k,[K_k,H_k]]
\right).
$$

Now the charges are **never stale** relative to the electronic state.

This is direct minimization of the SCC functional on the projector manifold.

No artificial distinction between “SCF iteration” and “DMM iteration.”

For your use case that seems conceptually much cleaner.

---

# Can we afford rebuilding \(H'\) after every bold step?

I initially worried this would double the cost.

Looking at the DFTB form, I think we can make it quite cheap.

The SCC AO Hamiltonian has the form

$$
H_{\rm SCC}
=
H_0+
\frac12
\left(
D_VS+SD_V
\right),
$$

where \(D_V\) is diagonal in the AO index and contains the atomic SCC potential \(V_A\).

In the orthogonal basis,

$$
H'_{\rm SCC}
=
X^TH_0X
+
\frac12X^TD_VSX
+
\frac12X^TSD_VX.
$$

Precompute at each geometry

$$
H'_0=X^TH_0X,
$$

and

$$
B=SX.
$$

Then for each charge update calculate only

$$
M=X^T(D_V B).
$$

The second SCC term is just its transpose:

$$
X^TSD_VX=M^T.
$$

Therefore

$$
\boxed{
H'_{\rm SCC}
=
H'_0+\frac12(M+M^T).
}
$$

That means an SCC-Hamiltonian rebuild in the orthogonal basis requires:

* row scaling \(D_VB\): \(O(N^2)\);
* **one general GEMM** \(X^T(D_VB)\);
* symmetrization.

Not the two full GEMMs `Xᵀ·H·X` you currently think of.

That makes updating \(H\) after each bold step significantly more attractive.

---

# Likewise the Mulliken charges do not require building the full AO density

You have

$$
D_{\rm AO}=XKX^T.
$$

Mulliken orbital populations are the diagonal of

$$
D_{\rm AO}S
=
XKX^TS.
$$

Precompute

$$
C=X^TS.
$$

Then calculate only

$$
Y=KC.
$$

One GEMM.

The orbital population is

$$
p_\mu
=
(XY)_{\mu\mu}
=
\sum_i X_{\mu i}Y_{i\mu},
$$

which is just an \(O(N^2)\) diagonal contraction.

Then sum \(p_\mu\) by atom.

So:

$$
\boxed{
K\rightarrow q
}
$$

can also be done with **one general GEMM**, rather than explicitly constructing \(XKX^T\).

For a coupled bold step the rough product count becomes:

* 2 GEMMs: double-commutator update;
* 1 GEMM: \(K\rightarrow q\);
* 1 GEMM: \(q\rightarrow H'\);
* McWeeny: 2 products every one or two steps.

If you retract every second step:

$$
\sim 5
$$

matrix-product equivalents per coupled step.

That is not as cheap as your present bold step, but if it converges in 3–5 steps it can still be much cheaper than a 8–16-diagonalization Jacobi SCC.

And unlike the present 2-step approximation, it has a systematic convergence target.

---

# Your present speed numbers leave a lot of room

Method 1 is roughly **9–14× faster than Jacobi SCC**. 

Even method 3, which essentially doubles the work, remains several times faster.

So you absolutely do **not** need to preserve the 10× headline.

If a controlled method costs:

* 2× bold → ~5× Jacobi,
* 3× bold → ~3×,
* 5× bold → ~2×,

but takes the force error from 2–4% down to \(<0.5\%\), that is a much more valuable solver.

I think your report reaches the same practical conclusion: there is enough performance margin to buy accuracy if the extra work actually attacks the correct residual. 

---

# A second major direction: use Broyden memory, not scalar mixing

I would test this in parallel because it is cheap.

SCC-DFTB's nonlinearity lives largely in a comparatively tiny vector of atomic charges.

Existing DFTB implementations traditionally accelerate this with modified Broyden; that approximate inverse response matrix is updated from previous SCF iterations. ([PubMed Central (PMC)][1])

You can do something stronger:

> **carry the Broyden inverse between geometry steps.**

At \(R+\delta R\), initialize not just

$$
K_{\rm old},q_{\rm old},
$$

but also

$$
B_{\rm old}\approx
\left(
\frac{\partial r}{\partial q}
\right)^{-1}.
$$

Then your two-step bold evaluation gives a residual

$$
r_k=q_{\rm bold}-q_k.
$$

Do one quasi-Newton charge correction

$$
q_{k+1}=q_k-B_k r_k.
$$

Rebuild \(H\), run another two bold steps, update Broyden.

This may need only 2–3 outer passes because you're exploiting the response learned at the previous geometry.

The literature on extended-Lagrangian SCC-DFTB does precisely exploit propagated electronic/charge information to cut SCF iterations, and the conventional DFTB SCF cited there uses modified Broyden. ([PubMed Central (PMC)][1])

So I would not spend more time testing \(\alpha=0.2\). A scalar mixer is far below what the problem deserves.

---

# A third possibility is even more radical: embrace an approximate shadow potential

If your primary use is **long geometry relaxation or MD**, rather than obtaining a fully converged SCC state at every intermediate geometry, there is an established theoretical framework that is extremely close to what your bold method is accidentally doing.

Recent shadow/extended-Lagrangian BOMD work constructs a *different approximate energy functional* that can be solved in one step, and computes forces that are exactly consistent with that shadow potential. It has been developed explicitly for SCC-DFTB. ([ResearchGate][2])

The attraction is:

* instead of taking a badly converged solution of the exact SCC functional,
* you take an **exact stationary solution of a nearby shadow functional**.

Then the forces are conservative with respect to that approximate energy, rather than noisy/nonvariational forces from an incomplete SCF.

Niklasson's first-level shadow formulation even gives a hierarchy in which the potential error can improve from quadratic to quartic in the electronic residual after a Newton-like update. ([ResearchGate][2])

That is much more ambitious than the immediate fix, but conceptually it maps extremely well onto your goal:

> **one or two very cheap electronic operations per geometry step, with controlled force quality.**

Your present bold method may be thought of as a crude version of that idea without the corresponding shadow energy/force derivation.

For production relaxation this might ultimately be better than insisting every intermediate point reproduce Jacobi SCC.

---

## What I would test next, in order

I would **not** implement another 1000-line algorithm yet.

First make four very focused experiments.

1. **Coupled bold-2:** after the first accepted commutator step, immediately compute \(q(K)\), rebuild \(H'[q]\), and use that updated Hamiltonian for the second step. Same two DMM rotations as now. This is the smallest change and directly tests whether “stale H” is the dominant 1–4% force error.

2. **Coupled bold-N:** repeat the same operation 1–8 times, rebuilding \(q,H'\) after *every* step. Plot force error versus N. This is the decisive plot. Unlike the current N-scan, I predict the force error should now trend downward overall. If it does not, then my full-SCC-gradient interpretation needs revisiting.

3. Change accept/reject from

   $$
   R_H\downarrow
   $$

   to

   $$
   E_{\rm SCC}\downarrow.
   $$

   Log both. Your report correctly notes that the current trust criterion is not the quantity for which the descent theorem exists. 

4. Then compare **persistent Broyden outer correction** against the direct coupled descent. Use the same two-step bold kernel as the cheap residual evaluator. Stop at charge RMS \(10^{-3}\), \(3\times10^{-4}\), \(10^{-4}\), and measure force error and throughput.

That will give you the systematic accuracy/cost curve you're currently missing.

The most important conceptual change is:

$$
\boxed{
\text{do not spend accuracy budget solving }P(H(q_{\rm stale}))
}
$$

Spend it on

$$
\boxed{
(K,q,H)\text{ moving together toward the SCC stationary point}.
}
$$

Your current 2-step bold method looks much more promising to me after this report, not less. The force error is not evidence that the projector rotation is intrinsically inaccurate. The \(N\)-scan demonstrates that **the expensive thing you were making more accurate was the wrong intermediate fixed-\(H\) problem**.

---

# Review — what to do, in order

The diagnosis matches the scan. Two commutator steps on a frozen
Hamiltonian are a cheap inexact map `q → q(B₂(K, H(q)))`. Making the
inner rotation longer moves that map toward `q(P(H(q_old)))`, and
`P(H(q_old))` is not the SCC density. The cap of two is an accidental
damping, not an electronic tolerance. More inner steps at fixed `H`
stay closed.

What does not follow: the limit `N → ∞` was not reached. After eight
steps `R_H` is still ~10⁻³ and the charges are still 0.007–0.014 e
off a Jacobi diagonalization of that same `H`. The direction is
measured. The arrival is not.

The infinitesimal descent `dE/ds = −‖[K, H(q(K))]‖²` holds for an
idempotent `K` when `H` is the Hamiltonian of that `K`. After one
finite commutator step `K` is not idempotent (McWeeny is what puts it
back), so a trust region stays necessary. The accept test should
eventually be the SCC energy. It should not be switched before that
energy is logged on the steps we already take.

Scalar `α` is finished. `α = 0.2` made the forces worse. `α = 1` once
helped formic and did not help GC at 0.02 Å. A new scalar is not an
experiment.

## 1. Coupled bold-N — first experiment

Smallest change that tests the diagnosis. Same commutator step. After
every accepted step, form the charges, rebuild `H'`, and rotate
against that new `H'`. Total rotations `N = 1…8`, same molecules,
same 0.02 Å step, same Jacobi comparison as the frozen-`H` table.

```text
K, q ← previous geometry
repeat N times:
    H' ← Xᵀ H_scc(q) X
    one accepted commutator step     # same trust region, η starts at 8
    q  ← Mulliken(K)
K ← McWeeny(K)
compare to Jacobi SCC
```

`N = 2` is the smoke test: two rotations, as now, but the second sees
a new Hamiltonian. The plot that decides is force error, `ΔE`, and
`dq` versus Jacobi SCC against `N`. If the force error trends down,
the stale `H` was the dominant error. If it does not, this
interpretation is wrong and the later items wait.

McWeeny stays once, at the end, for the first run, so the only change
from the frozen scan is the rebuild. If the charges after a single
non-idempotent step are garbage, the follow-up is one McWeeny before
reading `q`, not a new algorithm.

Use the rebuild we already have. Do not block this plot on a fused
`H'` kernel. At batch 256 a handful of rebuilds is still well under
the Jacobi SCC.

## 2. Log the SCC energy on those steps — do not change the accept test yet

On each accepted coupled step record `R_H`, `Tr(KH')`, and

```text
E_SCC = Tr(K H'₀) + ½ Δqᵀ γ Δq
```

If `R_H` falls while `E_SCC` rises, the current trust test is
accepting the wrong event, and item 3 is justified. If they move
together, leave the test alone.

Evaluating `E_SCC` on every rejected trial would add a Mulliken to
the reject path. Log it on accepted steps only.

## 3. Accept on `E_SCC`, and let η grow after a success

Only if item 2 says `R_H` and `E_SCC` disagree.

```text
if E_SCC fell:
    keep K
    η ← min(8, 2 η)
else:
    restore K
    η ← η / 2
```

Doubling η inside a frozen-`H` scan is still the wrong experiment.
The frozen scan already showed that a longer walk toward `P(H_old)`
makes the SCC force worse.

## 4. Stop the coupled loop on the charge residual

Once item 1 has a downward force curve, stop on charge RMS rather
than on a fixed `N`. Inner effort tracks the outer residual, because
driving `R_H` to 10⁻⁶ while the charges are still 10⁻² off is the
oversolve the frozen scan already paid for.

```text
repeat:
    one coupled step (item 1)
until charge RMS < tol          # try 1e-3, then 3e-4, then 1e-4
```

`R_H ≲ c ‖Δq‖` is the same idea with an extra constant. Fit `c` from
the item-1 plot. Do not hard-code 0.1–0.3 before that plot exists.
Report force error and throughput at each tol, against the same
Jacobi SCC. A result at 2–5× Jacobi with the force under 0.5% is the
success criterion.

This is method 5 of `Bold_Step_Accuracy.md` with one rotation per
charge update. If the charges oscillate, put the existing charge DIIS
around `q`, still inside this loop. That is item 5.

## 5. Charge DIIS, then a Broyden memory carried across geometries

The nonlinearity sits in the atomic charges (10–84 numbers), not in
the orbital matrix. A scalar `α` cannot invert that response. The
GPU solver already has charge DIIS; turn that on around the coupled
step before writing a new mixer.

Persistent Broyden — keeping the approximate inverse Jacobian from
the previous geometry — is the increment after DIIS at a single
geometry. The response should change slowly for a 0.02–0.1 Å move,
so the previous geometry can hand over `B` as well as `K` and `q`.
It is an accelerator of item 4, not a replacement for the coupled
plot. If item 4 already reaches the force target in a few passes,
this stays optional.

## 6. One-GEMM charge read and one-GEMM `H'` rebuild

Implementation, after the plot says the coupled step is worth
speeding up. Both caches are per geometry: `X` and `S` do not change
when only `q` changes.

```text
H'₀ = Xᵀ H₀ X                  # once per geometry
B   = S X                      # once per geometry
M   = Xᵀ (D_V B)               # one GEMM per charge update
H'  = H'₀ + ½ (M + Mᵀ)

C = Xᵀ S                       # once per geometry
Y = K C                        # one GEMM
p_μ = Σ_i X_μi Y_iμ            # diagonal, then sum by atom
```

`D_V` is the atomic SCC potential placed on the orbitals, symmetric,
so `Xᵀ S D_V X = Mᵀ`. Today a rebuild is two GEMMs (`Xᵀ·H`, then
`(XᵀH)·X`) plus a full `X K Xᵀ` for the charges. The coupled step
then costs about two products for the commutator, one for `q`, one
for `H'`, and a McWeeny every step or two.

## 7. Shadow potential — last

For a long relaxation or for dynamics, an exact stationary point of a
nearby shadow energy can be a better target than a half-converged
point of the true SCC energy, because the forces are then consistent
with the energy that is being minimized. Niklasson's shadow /
extended-Lagrangian formulation is that construction, and it has been
written for SCC-DFTB. The bold step is not that construction: it has
no shadow energy and no force derived from one.

This changes the comparison. Jacobi SCC stops being the reference.
The tests become conservation of the shadow energy and how far that
energy sits from the SCC surface. New functional, new forces, new
acceptance. Do it after items 1–4 have either produced a force curve
against Jacobi or shown that matching Jacobi at every geometry is the
wrong price. It is not the next code change.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3160450/?utm_source=chatgpt.com "Lagrangian formulation with dissipation of Born-Oppenheimer molecular dynamics using the density-functional tight-binding method - PMC"
[2]: https://www.researchgate.net/publication/370087171_Shadow_energy_functionals_and_potentials_in_Born-Oppenheimer_molecular_dynamics?utm_source=chatgpt.com "(PDF) Shadow energy functionals and potentials in Born–Oppenheimer molecular dynamics"



---

# USER

so you implemented this 
Niklasson's linearized shadow functional
the results are in the table? It is not clear ... also I would be interested to see the performance tests ...

Can we use this Niklasson's linearized shadow functional also in sparse solver? Do you know exacly what it it how exacly it differen from normal potential? Both equations and pseudocode for evaluation ?

---

# Grokg 4.7

The first-level formula was evaluated on one 0.02 Å step. It is not installed in the dense solver or the sparse solver. The second table is that evaluation: one exact Newton step on the atomic charges, then one Jacobi diagonalization, compared with a full SCC Jacobi.

The algebraic energy is the same expression the SCC loop already uses,

\[
E = 2\,\mathrm{Tr}(K H_0) + \tfrac12 \Delta q \cdot V + E_\mathrm{rep}, \qquad V = \gamma\,\Delta q, \qquad \Delta q = q - q_0, \qquad H = H_0 + \mathrm{shift}(V).
\]

\(K\) is the ground-state projector of \(H(q)\), with \(\mathrm{Tr}(K) = N_\mathrm{occ}\).

A normal SCC step changes \(q\) until the Mulliken charge of that \(K\) equals the \(q\) that built \(H\). At that fixed point the formula is stationary, and the force is its derivative.

```text
q ← charges from the previous geometry
repeat
    H ← H0(R) + shift(V(q))
    K ← diagonalize(H)                 # sparse: purify(H)
    q_out ← Mulliken(K)
    q ← mix(q, q_out)
until |q_out − q| < tol
E ← 2 Tr(K H0) + ½ (q−q0)·V(q) + E_rep
F ← Hellmann–Feynman at this self-consistent q
```

The shadow functional keeps the same formula and stops iterating \(q\). The input \(q\) is the linearization point. One solve of \(H(q)\) is the exact ground state of that linearized functional. The Mulliken charge is computed and not written back. The force is the derivative at constant \(q\). The error against the SCC energy is quadratic in \(q_\mathrm{out}-q\).

```text
q ← carried charges                         # not updated
H ← H0(R) + shift(V(q))
K ← diagonalize(H)                          # one solve
q_out ← Mulliken(K)
E⁰ ← 2 Tr(K H0) + ½ (q−q0)·V(q) + E_rep     # uses q, not q_out
F⁰ ← Hellmann–Feynman(K, H) at constant q
```

The first level spends one Newton correction choosing a better linearization point, then does that same one-shot evaluation. The error becomes quartic in the original residual.

```text
f ← q_out − q
q¹ ← q − (∂f/∂q)⁻¹ f
H¹ ← H0(R) + shift(V(q¹))
K¹ ← diagonalize(H¹)                        # second solve
E¹ ← 2 Tr(K¹ H0) + ½ (q¹−q0)·V(q¹) + E_rep
F¹ ← Hellmann–Feynman(K¹, H¹) at constant q¹
```

In the accuracy run, \(\partial f/\partial q\) was a central difference with step \(10^{-3}\,e\): two diagonalizations per atom. That is why the residual and the energy moved as they did.

| system | charge rms | energy vs SCC | force vs SCC |
|---|---|---|---|
| formic | \(2.7\times10^{-2}\to 2.1\times10^{-4}\) | −11 meV → +0.016 meV | 65% → 0.16% |
| GC | \(3.6\times10^{-2}\to 2.8\times10^{-4}\) | −40 meV → +0.029 meV | 15% → 0.15% |
| diazaphen | \(2.5\times10^{-2}\to 2.2\times10^{-4}\) | −30 meV → −0.011 meV | 25% → 0.04% |
| DTH | \(7.3\times10^{-3}\to 1.1\times10^{-5}\) | −4.6 meV → +0.009 meV | 0.57% → 0.00% |

Batch 256, mean of three runs, at the displaced geometry. "2 Jacobi" is the electronic cost of the zeroth level plus the first level when that inverse Jacobian is already known. The last column is the cost of building it by finite differences, estimated from the measured diagonalization time.

| system | full SCC | 2 Jacobi | finite-difference Jacobian |
|---|---|---|---|
| formic | 8 steps, 3.94 ms, \(6.5\times10^{4}\) sys/s | 1.12 ms, 3.5× | 12 ms |
| GC | 16 steps, 20.5 ms, \(1.3\times10^{4}\) sys/s | 5.68 ms, 3.6× | 170 ms |
| diazaphen | 16 steps, 69.4 ms, \(3.7\times10^{3}\) sys/s | 16.5 ms, 4.2× | 710 ms |
| DTH | 16 steps, 473 ms, 540 sys/s | 268 ms, 1.8× | 23 s |

The SCC count is enqueued in chunks of 8, so 8 and 16 are steps launched, not a claim that iteration 16 was the first one under tolerance. With the Jacobian known, two diagonalizations are 1.8–4.2× a full SCC. Building the Jacobian by finite differences is slower than SCC on GC, diazaphen, and DTH.

The same charge formulas apply to the sparse solver. The only substitution is `purify(H)` in place of `diagonalize(H)`. The Jacobian stays an \(N_\mathrm{atom}\times N_\mathrm{atom}\) matrix on the charges. The sparse energy is already \(2\,\mathrm{Tr}(K H_0)+\tfrac12\Delta q\cdot V\). A shadow evaluation has to put the input \(q\), the one that built \(H\), into that double-counting term. The SCC loop currently puts the Mulliken charge there. That path is not wired up.

---

# Caveats — can this replace the SCC potential?

Plots: `debug/dense_multi/shadow_vs_jacobi_accuracy.png`, `debug/dense_multi/shadow_vs_jacobi_cost.png`.

What is preserved. The energy expression, \(H_0\), \(\gamma\), the repulsion, and the electron count. When \(q\) equals the Mulliken charge of \(K\), the shadow energy and the shadow force are the SCC energy and the SCC force. With the exact charge Jacobian, this 0.02 Å step already lands there: energy within 0.03 meV, force within 0.2% (DTH below 0.01%). The Hellmann–Feynman force is the true gradient of that energy when \(q\) is held fixed during the displacement.

What is lost if level 0 replaces SCC. Charge self-consistency. On this step that is 5–40 meV and 0.6–65% of the force. A relaxation would follow that other surface. DTH is the exception only because the carried charge was already close.

What is lost if level 1 with the exact Jacobian replaces SCC. Almost none of the energy or the force on this one step. The speed is lost instead: building that Jacobian by finite differences is slower than the Jacobi SCC on GC, diazaphen, and DTH. The purple bars in the cost plot are a different object. They assume the inverse Jacobian is already known, and then two diagonalizations are 1.8–4.2× the full SCC.

What the theorem does not give a static optimizer. Niklasson's constant-\(q\) force is conservative because \(q\) is integrated as its own dynamical variable. Rebuilding \(q^1(R)\) at every geometry and using the constant-\(q\) force drops \(\partial q^1/\partial R\). At level 1 that missing piece is small, because the energy is stationary in \(q\) to high order. At level 0 it is the charge-relaxation part of the force.

What was not measured. A cold start from \(q_0\). An occupation change. A cheap Jacobian (carried Broyden or a short Krylov). The 0.01 meV belongs to the exact Jacobian only. The step started from charges that were already an SCC solution of the previous geometry, which is the easy side of the Newton assumption.

So level 0 does not replace the SCC potential. Level 1 with the exact Jacobian replaces the energy and the force on this step and does not replace the cost. Level 1 with a free Jacobian is the only form that could replace SCC on intermediate geometries, and that Jacobian has not been timed or tested. A reported final energy still wants one converged SCC if the number has to sit on the SCC surface.