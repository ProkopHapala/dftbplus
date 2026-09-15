https://chatgpt.com/share/6aa8e403-69d8-83eb-800d-366bb27de532




---

# USER

do you think we can do excited states and carge sttes in this DFTB multi-solver framework?

I particular I would like to sample the excited state or charge state localize on one or the other molecule to study Proton-coupled electron transport ... do your think it is feasible? E.g. using CDFT?

Another thing - how difficult it is implement periodic boudnary conditions into this approach? We already have PBC in fortran version of DFTB+ so we can just copy that it should not be diffcult. Paralleization over k-points should be trivial...   























---

# Chat GPT 5.6 sol

Yes. In fact, **charge-localized diabatic states are probably one of the most natural next extensions of this multisolver**, because a Mulliken-charge CDFT constraint fits almost embarrassingly well into the SCC-DFTB Hamiltonian you already have.

And I would distinguish three things: **charged ground states**, **charge-localized diabatic states**, and **true excited adiabatic states**. They require rather different amounts of work.

## 1. CDFT / constrained DFTB looks particularly easy in your formulation

Suppose you want two diabatic states for a molecular pair,

$$
|L\rangle:\quad D^-A
$$

and

$$
|R\rangle:\quad DA^-,
$$

or similarly a hole localized on molecule 1 versus molecule 2.

Define a fragment population

$$
Q_F[P] = \sum_{A\in F} q_A[P]
$$

and enforce

$$
Q_F=Q_F^{\rm target}
$$

with a Lagrange multiplier:

$$
\mathcal L[P,\lambda]
=
E_{\rm DFTB}[P]
+
\lambda\left(Q_F[P]-Q_F^{\rm target}\right).
$$

For Mulliken charges, the resulting Hamiltonian term has essentially the form

$$
H^{\rm constr}_{\mu\nu}
=
\frac{\lambda}{2}S_{\mu\nu}
\left(w_{A(\mu)}+w_{A(\nu)}\right),
$$

where \(w_A=1\) if atom \(A\) belongs to the constrained fragment and 0 otherwise, or more general weights if desired.

But your SCC Hamiltonian is already

$$
H^{\rm SCC}_{\mu\nu}
=
H^0_{\mu\nu}
+
\frac12S_{\mu\nu}
\left(V_A+V_B\right).
$$

So computationally you just replace

$$
V_A\rightarrow V_A+\lambda w_A.
$$

That is almost absurdly convenient. **No new matrix machinery. No new eigensolver. No new density representation.**

Recent DFTB+ work actually reintroduced exactly this general constrained-ground-state formalism for Mulliken charges, writing the constrained problem as a Lagrangian in the density matrix and demonstrating diabatic charge-constrained calculations. ([PubMed Central (PMC)][1])

So I would definitely steal the exact implementation/force derivation from the Fortran DFTB+ implementation rather than reinvent it.

### The additional SCF variable is only one scalar

For a two-fragment system you may only need one independent constraint because total charge is already fixed.

You then solve simultaneously

$$
q \rightarrow q'
$$

and

$$
Q_F(\lambda)-Q_F^{target}=0.
$$

Since λ is one scalar/system, this is negligible computationally.

You could update λ with a safeguarded secant/Newton scheme. Better yet, incorporate it as another variable into your existing DIIS vector:

$$
x=(\Delta q_1,\ldots,\Delta q_{N_A},\lambda)
$$

with residual

$$
r=(q^{out}-q^{in}, Q_F-Q_F^{target}).
$$

For your batched architecture this is perfect: every replica can have a different target charge and therefore represent a different diabatic state.

---

## 2. For PCET, I think this is more immediately useful than TD-DFTB

Imagine your proton coordinate \(x_H\), and for each geometry you solve two electronic states:

$$
E_L(x_H) = E[\text{electron on molecule L}]
$$

$$
E_R(x_H) = E[\text{electron on molecule R}].
$$

Then you have precisely the **diabatic PCET surfaces** you want.

You can see directly how proton motion changes

$$
\Delta E(x_H)=E_R-E_L,
$$

where they cross, how the proton barrier changes with charge localization, etc.

That is exactly the type of problem CDFT was developed for. CDFT has long been used to construct donor- and acceptor-localized diabatic surfaces and extract Marcus/ET parameters, including applications explicitly involving PCET. ([American Chemical Society Publications][2])

And constrained DFTB itself has already been used for charge-transfer states in donor–acceptor systems, while DFTB-CI extensions have been developed to mix localized charge configurations and describe CT excited states. ([PubMed][3])

For your application I would therefore first build:

$$
\boxed{\text{SCC-DFTB + fragment charge constraint}}
$$

rather than jumping immediately to TD-DFTB.

It gives exactly the physically interpretable diabatic states needed for PCET.

---

## 3. There is an important spin complication

For simply adding/removing one electron,

$$
M \rightarrow M^+
$$

or

$$
M\rightarrow M^-,
$$

changing the total electron count is trivial.

But physically it produces an odd-electron system. So a proper calculation requires **spin-polarized DFTB**, not just taking your present closed-shell density and giving the frontier orbital occupation 0.5.

DFTB+ already supports collinear and non-collinear spin polarization. ([DFTB+][4])

Architecturally, collinear spin is not horrible:

$$
D=D^\uparrow+D^\downarrow,
\qquad
m=D^\uparrow-D^\downarrow.
$$

You need spin-resolved occupations and the DFTB spin interaction contribution.

That is more work than CDFT itself but still compatible with the same batch architecture.

And it becomes especially important for a **neutral charge-transfer state**

$$
D^+A^-.
$$

Now you effectively have two radicals. There are broken-symmetry/open-shell singlet and triplet possibilities. A charge constraint alone tells you where the charge goes; it does **not** uniquely specify the spin state.

So I would separate:

$$
\text{charge localization}
$$

from

$$
\text{spin coupling}.
$$

For initial PCET studies, a spin-polarized charge-constrained diabatic surface is already extremely useful.

---

# 4. ΔSCF excited states are also quite natural

There is another inexpensive direction before full TD-DFTB.

Instead of constraining spatial charge, constrain an occupation:

$$
\text{HOMO}\rightarrow\text{LUMO}
$$

and rerun SCC self-consistently.

DFTB+ already implements a \(\Delta\)DFTB approach for the lowest singlet excited state, using separate constrained electronic configurations and a Ziegler-type sum rule. ([American Chemical Society Publications][5])

This would fit your multisolver nicely because the expensive machinery remains exactly:

```text
H(q)
→ diagonalize
→ prescribed occupation
→ density
→ SCC
```

You merely change the occupation rule.

The complication is **state tracking**.

You cannot say forever:

```text
occupy eigenstate 43, empty eigenstate 42
```

because states cross/reorder during proton transfer or geometry relaxation.

But your warm eigenbasis is actually an advantage. You can identify the new target orbital from maximum overlap with the previous one,

$$
k_{\rm new}
=
\arg\max_k
|\langle\phi_k^{new}|\phi_{target}^{old}\rangle|^2.
$$

Even better, track a small occupied/unoccupied **subspace** rather than an individual eigenvector.

That would let you follow a locally excited state along a geometry trajectory.

---

# 5. Full TD-DFTB is feasible, but it is a separate layer

DFTB+ already has linear-response TD-DFTB/Casida. The response equation is approximately

$$
\Omega F_I=\omega_I^2F_I,
$$

where the indices are occupied→virtual transitions \(ia\). ([DFTB+ Recipes][6])

DFTB+ supports vertical/adiabatic excitations, oscillator strengths and excited-state gradients/relaxation. ([DFTB+ Recipes][7])

But this is **not simply another SCC state**.

The dimension becomes roughly

$$
N_{\rm trans}=N_{\rm occ}N_{\rm virt}.
$$

For your \(N\sim86\), perhaps

$$
40\times46\sim1800,
$$

so you definitely do **not** want to dense-diagonalize an 1800×1800 response matrix independently for hundreds of replicas.

Instead I would implement a matrix-vector operation

$$
y=\Omega x
$$

and use Davidson/Stratmann/Lanczos to obtain only the lowest few states.

The DFTB transition-charge approximation makes that relatively attractive.

But for **localized electron-transfer/PCET states**, TD-DFTB would not be my first tool anyway. Ordinary local/semi-local TD methods have well-known CT-state issues; DFTB+ supports range-separated TD-DFTB specifically for improved long-range excitation behavior. ([DFTB+ Recipes][8])

So my hierarchy would be:

| Goal                                 | Method                       | Difficulty in your solver |
| ------------------------------------ | ---------------------------- | ------------------------: |
| \(M^+\), \(M^-\) total charge        | change electron count + spin |              low–moderate |
| charge localized on molecule A/B     | **constrained SCC-DFTB**     |                   **low** |
| PCET diabatic surfaces               | **constrained SCC-DFTB**     |                   **low** |
| HOMO→LUMO/local ΔSCF state           | occupation-constrained SCC   |                  moderate |
| several interacting localized states | cDFTB-CI / nonorthogonal CI  |                  moderate |
| general optical excited states       | TD-DFTB/Casida               |               substantial |
| strong multireference excited states | REKS/etc.                    |     much more substantial |

DFTB+ itself now has TD-DFTB, ΔDFTB, REKS, nonadiabatic couplings and related excited-state machinery, so the Fortran code gives you reference implementations for almost everything in that table. ([DFTB+][4])

---

# 6. CDFT-CI could be especially interesting for your PCET idea

Once you can calculate two charge-localized states,

$$
|\Phi_L\rangle,\qquad|\Phi_R\rangle,
$$

you don't necessarily have to stop with their diabatic energies.

Build a tiny effective electronic Hamiltonian

$$
\mathbf H =
\begin{pmatrix}
E_L & H_{LR}\\
H_{LR} & E_R
\end{pmatrix}.
$$

Then the adiabatic states are

$$
E_\pm
=
\frac{E_L+E_R}{2}
\pm
\sqrt{
\left(\frac{E_L-E_R}{2}\right)^2+
H_{LR}^2
}.
$$

This is potentially exactly what you want for your broader “Kekulé-mediated correlated PCET” problem: calculate inexpensive **localized electronic configurations**, then couple them in a small state-space Hamiltonian rather than forcing one giant electronic calculation to discover all physics at once.

There are established CDFT approaches for extracting electronic couplings, and analogous constrained-DFTB/DFTB-CI approaches exist. ([PubMed][9])

That is a much more exciting extension of your multisolver than merely “calculate absorption spectra.”





