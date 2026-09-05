https://chatgpt.com/share/6a9bebc5-5838-83eb-bf8e-6b6ff3790c3b

---

# USER

Note, you mentioned that puirification is ovesold in the previous doc, but I think we may prefer such methods more for memory reasons than for performance reasons. For largers systems we are seriously memory-bound, and diagonalization of igenvectros matrix inversion densify the matrixes, but the algorithms using sparse prepresentation CSR or somthing are inherenly smaller (O(n) rather than O(n^2)) in memory so they can be perhaps done more efficiently using local memroy workgroups. But I agree this is work only for larger system far beyond one-wrogroupsise limit. It is perhaps separate route, wich we can use for solving silicon and diamond nanocrystal, graphene ribbons and such things...

plese elaborate on this and write respective OpenCL kernels, with physics/equations in code and justification of the desing (memory layout, partition to workgroups etc)

---

# GPT 5.6 sol

Yes — I think your correction is important. For the large-system branch, purification becomes attractive mainly because it lets us **avoid ever allocating the dense eigenvector matrix** and exploit electronic nearsightedness. In a localized basis, gapped systems have exponentially decaying density matrices, so after truncating beyond a localization radius the number of retained blocks per atom can approach a constant. That gives \(O(N)\) storage and, with fixed sparsity radius, asymptotically \(O(N)\) sparse algebra. This is precisely the physical basis of linear-scaling electronic-structure methods. ([APS Journals][1])

I wrote a first OpenCL implementation of this architecture:

[Download `sparse_bsr4_purification.cl`](sandbox:/mnt/data/sparse_bsr4_purification.cl)

I also syntax-checked it as OpenCL 1.2 with `clang`. Everything is `float`; there is no FP64 anywhere.

## The central architecture

I would **not use ordinary scalar CSR** for the systems you mention. For C and Si with an \(s,p\) DFTB basis, every atom naturally contributes four orbitals, so use **BSR4: block-sparse row with 4×4 atom-atom blocks**.

Conceptually:

```text
atom row i
   |
   +-- block (i,j0) : 4x4 floats
   +-- block (i,j1) : 4x4 floats
   +-- block (i,j2) : 4x4 floats
   ...
```

with

```c
row_ptr[Natom+1]
col_idx[Nblock]
values[Nblock][16]
```

I deliberately recommend storing both \(A_{ij}\) and \(A_{ji}\), even though the matrices are symmetric. That costs roughly 2× versus packed upper-triangular storage, but on GPU it buys something much more valuable: **all algorithms become gather-only**. There are no atomics, no scatter writes, and every 4×4 block is contiguous.

For a localization mask with average \(z\) atom-block neighbors,

$$
M \sim 16\,z\,N_{\rm atom}
$$

floats rather than

$$
(4N_{\rm atom})^2.
$$

So the distinction becomes enormous for nanocrystals with thousands of atoms.

The important qualification is that this is only linear scaling if \(z\) remains effectively constant as the system grows. That is well justified for sufficiently gapped systems; it becomes progressively worse as the gap closes. Diamond is nearly ideal, Si should still be good, while zero-gap graphene is intrinsically a much harder case. Finite electronic temperature restores stronger localization even in metals. ([Cambridge University Press][2])

---

# Non-orthogonal purification directly in \(S\)

This is another place where the sparse route differs fundamentally from the small dense solver.

I would **not compute \(S^{-1/2}\)**.

Let the spinless occupied density kernel be \(K\):

$$
\hat\rho =
\sum_{\mu\nu}
|\phi_\mu\rangle
K^{\mu\nu}
\langle\phi_\nu|.
$$

Because the AO basis is nonorthogonal, operator multiplication inserts \(S\). Therefore idempotency is

$$
\boxed{KSK=K}
$$

rather than \(K^2=K\), and

$$
N_{\rm occ}=\mathrm{Tr}(KS).
$$

For closed shell,

$$
N_e=2\mathrm{Tr}(KS).
$$

The generalized McWeeny transformation is then

$$
\boxed{
K'=3KSK-2KSKSK
}
$$

or, using an auxiliary sparse kernel \(L\),

$$
\boxed{
K(L)=3LSL-2LSLSL.
}
$$

That nonorthogonal density-matrix formulation is well established and, importantly for us, can be formulated without explicitly constructing \(S^{-1}\) or \(S^{-1/2}\). ([APS Journals][3])

---

# The key GPU kernel is masked sparse multiplication

All of purification ultimately reduces to sparse matrix products.

But ordinary SpGEMM has a problem: multiplying two sparse matrices creates fill-in.

We explicitly do

$$
C=P_M(AB),
$$

where \(P_M\) projects the product onto a predetermined geometric sparsity mask.

So perhaps:

$$
R_S \sim 4\text{--}6\AA,
$$

$$
R_K \sim 10\text{--}20\AA,
$$

and anything beyond \(R_K\) is simply discarded.

That is where linear memory scaling comes from.

The kernel I wrote maps:

> **one workgroup = one output atom row.**

Within that workgroup, the entire left sparse row is loaded once into local memory:

```c
__local uint  lAcol[MAX_A_ROW_BLOCKS];
__local float lAval[MAX_A_ROW_BLOCKS * 16];
```

For `MAX_A_ROW_BLOCKS=256`, this is only about 17 KiB.

Then the workgroup contains eight 16-thread teams for `WG=128`:

```text
WG 128
├── team 0 : 16 lanes -> one 4x4 C_ij block
├── team 1 : 16 lanes -> another C_ij
├── ...
└── team 7
```

Each lane owns one scalar of the output 4×4 block:

```c
r = lane >> 2;
c = lane & 3;

C_rc =
    A_r0*B_0c +
    A_r1*B_1c +
    A_r2*B_2c +
    A_r3*B_3c;
```

The cached \(A_{ik}\) row is reused for **every \(j\)** in the output row, which is where local memory gives us real benefit.

The right-hand \(B_{kj}\) block is gathered from global memory.

I intentionally used binary search in the sorted BSR rows rather than a giant precomputed multiplication plan. A multiplication plan would make lookup faster but can itself require roughly \(O(Nz^2)\) metadata. Since the motivation here is memory, I would begin with sorted rows and only introduce hash tables or multiplication plans if profiling says lookup dominates.

---

# Different masks for intermediates matter

I would not force every sparse matrix onto exactly the same radius.

For

$$
Q=KSK,
$$

compute

$$
T=KS
$$

on a somewhat **expanded mask**, because multiplication expands support:

```text
K mask:         RK
S mask:         RS

T = K*S:        ~ RK + RS
Q = T*K:        project back to RK
```

This costs more temporary memory, but it should substantially reduce truncation error.

The kernel is generic: \(A\), \(B\), and \(C\) can each have different BSR patterns.

So the host can call:

```text
bsr_spgemm( K[RK], S[RS] -> T[RT] )
bsr_spgemm( T[RT], K[RK] -> Q[RK] )
```

with

$$
R_T > R_K.
$$

This is probably one of the most important accuracy/performance parameters to benchmark.

---

# I implemented two possible solver routes

I don't yet know which will win for DFTB, so the code supports the building blocks for both.

### Route A: sparse generalized TC2

Use

$$
Q=KSK.
$$

Then

$$
K_{n+1}=
\begin{cases}
Q, & \mathrm{Tr}(KS)>N_{\rm occ}\\[3pt]
2K-Q, & \mathrm{Tr}(KS)<N_{\rm occ}.
\end{cases}
$$

This is attractive because each iteration requires essentially only

```text
T = K*S
Q = T*K

trace = Tr(T)

K = Q          or
K = 2K-Q
```

so two sparse products per iteration.

The trace is especially cheap. Because

$$
T=KS,
$$

we need only diagonal atom blocks:

$$
\mathrm{Tr}(KS)
=
\sum_A \mathrm{Tr}(T_{AA}).
$$

That is \(O(N_{\rm atom})\), not \(O(N_{\rm block})\).

I provided kernels for the partial trace, GPU reduction, TC2 update, and generalized-idempotency residual

$$
\|KSK-K\|_F.
$$

---

## But TC2 has one conceptual trap

TC2 does not magically discover the Hamiltonian eigenvectors.

It only pushes eigenvalues toward 0 or 1.

So the starting \(K_0\) must already be a spectral function of the generalized eigenproblem

$$
Hc=\varepsilon Sc.
$$

This is why simply reusing yesterday's arbitrary sparse \(K\) and applying

$$
K\rightarrow KSK
$$

does not solve the electronic ground state.

One route I implemented is a **sparse approximate inverse of \(S\)**.

Use Hotelling/Newton-Schulz:

$$
\boxed{
Z_{n+1}=2Z_n-Z_nSZ_n
}
$$

with

$$
Z\approx S^{-1}.
$$

Because \(S\) is very local and usually fairly close to identity in an AO basis, its inverse is itself strongly localized when \(S\) is well conditioned, so we can truncate \(Z\) as well.

Again, this is only sparse SpGEMM:

```text
T    = Z*S
U    = T*Z
Znew = 2Z-U
```

Then construct

$$
K_0
\simeq
\frac{
\varepsilon_{\max} Z
-
ZHZ
}{
\varepsilon_{\max}-\varepsilon_{\min}
}.
$$

If \(Z=S^{-1}\) exactly, this is simply the nonorthogonal representation of the standard linearly scaled Hamiltonian projector guess.

I included those kernels too.

---

# Route B: inverse-free LNV

This is perhaps conceptually cleaner for your long-term large-system solver.

Keep an auxiliary localized matrix \(L\), and construct

$$
K(L)=3LSL-2LSLSL.
$$

Then minimize

$$
\Omega[L]
=
2\,\mathrm{Tr}
\left[
K(L)(H-\mu S)
\right].
$$

This formulation avoids the inverse overlap altogether; that was one of the explicit motivations of the nonorthogonal LNV formulation. ([APS Journals][3])

I derived and included the matrix gradient as comments and a combination kernel.

Let

$$
F=H-\mu S.
$$

Then for symmetric \(L,S,F\),

$$
\frac12\frac{\partial\Omega}{\partial L}
=
3(SLF+FLS)
-
2(
SLSLF+
SLFLS+
FLSLS
).
$$

Every term is again composed from the same sparse multiplication kernel.

Then:

```text
G = gradient(L)

L <- L - alpha*G
```

or eventually CG/L-BFGS-like optimization.

The source includes:

```c
bsr4_lnv_gradient_combine()
bsr4_lnv_descent_step()
```

I would start with simple gradient descent or nonlinear CG only after the sparse multiplication engine is validated.

The tradeoff is clear:

| route      | advantage                                                           | disadvantage                                                   |
| ---------- | ------------------------------------------------------------------- | -------------------------------------------------------------- |
| sparse TC2 | only ~2 SpGEMMs / purification iteration                            | needs a good H-derived initial \(K\), likely sparse \(S^{-1}\) |
| LNV        | no explicit \(S^{-1}\); directly variational in nonorthogonal basis | more SpGEMMs per minimization step                             |

I would genuinely benchmark both rather than choose philosophically.

---

# Mulliken charges become extremely cheap

Once

$$
T=KS
$$

has been calculated,

$$
q_A
=
2\sum_{\mu\in A}(KS)_{\mu\mu}.
$$

For four orbitals/atom, that is simply

$$
\boxed{
q_A=2\,\mathrm{Tr}(T_{AA})
}
$$

so one thread can extract a charge from one diagonal 4×4 block.

The code contains:

```c
bsr4_mulliken_from_ks()
```

No \(C\), no dense \(D\), no \(N^3\) construction.

---

# And forces do not force us back to eigenvectors

This is another reason I now like this branch more than I did initially.

Let

$$
K=C_{\rm occ}C_{\rm occ}^T
$$

for spinless occupancy.

The energy-weighted density matrix normally written using eigenvectors is

$$
W=C_{\rm occ}\varepsilon C_{\rm occ}^T.
$$

But

$$
HC_{\rm occ}
=
SC_{\rm occ}\varepsilon,
$$

therefore

$$
\boxed{
W=KHK.
}
$$

So it is just two more sparse products.

Then the generalized-eigenproblem part of the closed-shell force has the familiar structure

$$
\frac{dE_{\rm band}}{dR}
=
2\operatorname{Tr}\left(K\frac{dH}{dR}\right)
-
2\operatorname{Tr}\left(W\frac{dS}{dR}\right),
$$

before the SCC-specific and repulsive DFTB terms.

So this architecture can in principle do MD without ever constructing the dense occupied MO matrix.

That is a fairly strong argument for the sparse-density route.

---

# Where single precision enters

Here I suspect the dominant controllable error will often become **localization/truncation**, rather than f32 rounding.

We should monitor simultaneously

$$
\|KSK-K\|,
$$

$$
|\operatorname{Tr}(KS)-N_{\rm occ}|,
$$

energy convergence versus \(R_K\),

and force convergence versus \(R_K\).

For the sparse-inverse route also monitor

$$
\|I-ZS\|.
$$

If these residuals plateau far above the f32 noise floor when increasing iteration count, the problem is probably the truncation mask rather than precision.

That gives us a nice hierarchy:

```text
increase iterations
        ↓
if no improvement:
increase RT / RK
        ↓
if still no improvement:
investigate f32 / conditioning
```

rather than immediately introducing doubles.

---

# Diamond / silicon versus graphene

This architecture looks especially well matched to the first two.

Diamond has a large gap, hence a strongly localized density kernel. Silicon's smaller gap means a longer \(R_K\), but the principle still works very well. Linear-scaling purification methods have historically been applied exactly to insulating and semiconducting systems, including C and Si. ([arXiv][4])

Graphene is the warning case. At zero temperature in an infinite gapless sheet, the density matrix is not exponentially localized in the same favorable way. A finite nanoribbon may acquire a gap depending on width/edge structure, and finite electronic temperature/smearing also improves locality, but I would not promise true practical \(O(N)\) behavior for generic large graphene from the start. ([SIAM][5])

---

# One completely separate scaling problem: SCC electrostatics

There is one major bottleneck sparse purification does **not** fix.

DFTB's \(\gamma_{AB}\) becomes Coulombic at long range:

$$
\gamma_{AB}(R)\rightarrow\frac1R.
$$

So

$$
V_A
=
\sum_B
\gamma_{AB}\Delta q_B
$$

is fundamentally long-range.

For 100 atoms, direct \(N^2\) is fine.

For 100,000 atoms, it is not.

So a genuinely large-system implementation will eventually need something like:

```text
finite nanocrystal:
    Barnes-Hut / FMM

periodic crystal:
    particle-mesh / Ewald-like solver
```

while the electronic \(H/S/K\) matrices remain block-sparse.

That actually gives a rather elegant long-term architecture:

```text
               geometry
                  │
      ┌───────────┴───────────┐
      │                       │
 short-range BSR4         long-range charge
 H0, S, repulsion          gamma electrostatics
      │                       │
      │                   FMM / mesh
      │                       │
      └───────────┬───────────┘
                  ↓
               H_SCC
                  ↓
        sparse density solver
          TC2 / LNV / FOE
                  ↓
             K, KS, KHK
                  ↓
         charge / energy / force
```

That seems to me a much more promising route to *really large* DFTB than trying to scale the dense Jacobi machinery indefinitely.

The immediate next benchmark I would do is therefore not full chemistry yet: generate synthetic 3D diamond-like BSR patterns with \(z=16,32,64,128\), random symmetric 4×4 blocks, and benchmark `bsr4_spgemm_masked()` against system size from \(10^3\) to \(10^5\) atoms. That will tell us whether the **one-WG-per-row + local-left-row cache** architecture has the bandwidth/utilization behavior we expect on the RTX 3090 before we build the full purification solver around it.

[1]: https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.71.1085?utm_source=chatgpt.com "Linear scaling electronic structure methods | Rev. Mod. Phys."
[2]: https://www.cambridge.org/core/books/abs/electronic-structure/locality-and-linear-scaling-on-methods/69CE26C604C6EEEC16330DFCE0FE9068?utm_source=chatgpt.com "Locality and linear scaling O(N) methods (Chapter 23) - Electronic Structure"
[3]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.50.17611?utm_source=chatgpt.com "Generalization of the density-matrix method to a nonorthogonal basis | Phys. Rev. B"
[4]: https://arxiv.org/abs/cond-mat/9810042?utm_source=chatgpt.com "Density matrices in O(N) electronic structure calculations: theory and applications"
[5]: https://epubs.siam.org/doi/abs/10.1137/100814019?journalCode=siread&utm_source=chatgpt.com "Decay Properties of Spectral Projectors with Applications to Electronic Structure | SIAM Review"



---

# USER

1) yes, atoms are natrally blocks and if the blosk are 4x4 it is ideall, but I mean CRS so that each atom have explicitl list of neighbors and w have index array poitn to offset where each atom and his neighbor start in the packed array.

2) yes, graphene ribons have smaller band gap, but they still ofthen have some bandgap, but also they are long which makes them less dense for geometrical reason while si and diamond crystal are 3D. The 2D and and 1D system are much sparser naturally.

3) yes, sure we should not compute any matrix inversion not even S^-(1/2) of Lowdin trasofrm.
Also we should avoid any product matrixes which naturally densify, the matrix polynominal should be computed by multiple application of sparse matrix multiplications. (this is the masked sparse multiplication you talk about right?)

And possibly we can also aporidory restric the elements of resulting matrix (assuming fast decay) so we may aproiti assume soe leements are zero and not represe tthem at all (but this should be tested)


---




# Chat GPT 5.6 sol

Yes — with those clarifications I would formulate the large-system branch much more cleanly.

## 1. Atom-block CSR is exactly the right representation

What you mean is standard CSR/CRS structurally, but each nonzero is a \(4\times4\) atom-orbital block:

```c
row_ptr[Natom + 1];      // row_ptr[i] ... row_ptr[i+1]-1
col_idx[Nblock];         // neighbor atom j
val[Nblock][16];         // 4x4 block A_ij
```

So atom \(i\) owns an explicit packed neighbor list:

```text
row_ptr[i]
   ↓
[j0][ 4x4 ]
[j1][ 4x4 ]
[j2][ 4x4 ]
...
   ↑
row_ptr[i+1]
```

This is much better than scalar CSR because the atomic block is the natural unit of both the physics and GPU arithmetic.

For \(s,p\) DFTB:

$$
b=4,
\qquad
A_{ij}\in\mathbb R^{4\times4}.
$$

If the average number of retained atom blocks per row is \(z\), one sparse matrix needs approximately

$$
N_{\rm at}\,z\,(16\cdot4+4)
$$

bytes, i.e. roughly

$$
68\,N_{\rm at}z\ {\rm bytes}.
$$

For \(10^4\) atoms and \(z=64\), that is only about **44 MB/matrix**.

The corresponding dense \(4N\times4N\) matrix is

$$
(4N)^2\,4
=
64N^2 {\rm\ bytes},
$$

or **6.4 GB for just one matrix** at \(10^4\) atoms.

That is exactly why this branch becomes fundamentally different from the dense diagonalization branch.

---

# 2. Dimensionality helps graphene enormously

Yes. There are two competing effects:

### Electronic locality

Smaller gap means longer decay length of the density kernel:

$$
|K(\mathbf r,\mathbf r')|
\sim
e^{-|\mathbf r-\mathbf r'|/\xi},
$$

with \(\xi\) generally growing as the gap shrinks.

So:

```text
diamond     large gap  → short ξ
silicon     smaller    → longer ξ
GNR         possibly small gap → longer ξ
```

But geometrical dimensionality controls how many atoms lie inside that radius.

Approximately,

$$
z(R)\propto R^d.
$$

Therefore:

$$
\begin{aligned}
{\rm 1D:}&\quad z\sim R\\
{\rm 2D:}&\quad z\sim R^2\\
{\rm 3D:}&\quad z\sim R^3.
\end{aligned}
$$

So a graphene ribbon can tolerate a substantially longer localization radius before the sparse matrix becomes expensive.

For example, very schematically:

```text
diamond, RK = 10 Å:
        neighbors ~ RK³

graphene ribbon, RK = 20 Å:
        neighbors ~ RK or RK²
```

depending on whether the cutoff is larger or smaller than the ribbon width.

So I agree: **small-gap graphene ribbons are not necessarily worse computationally than 3D Si**, despite slower electronic decay.

This is actually a very attractive use case for the sparse route.

---

# 3. Yes: “masked sparse multiplication” means exactly what you describe

Ordinary multiplication

$$
C=AB
$$

will eventually become dense even when \(A\) and \(B\) are sparse.

We instead define beforehand which elements of \(C\) are allowed to exist:

$$
\boxed{
C=P_M(AB)
}
$$

where \(M\) is the predefined sparsity mask.

So if

```text
C.row_ptr[i]
C.col_idx[]
```

contains only atoms within, say,

$$
R_K=15\ {\rm Å},
$$

the kernel **never even attempts to allocate or calculate** \(C_{ij}\) outside that neighborhood.

That is stronger than:

```text
compute dense-ish product
↓
discard small numbers
```

We instead do:

```text
know desired sparse pattern beforehand
↓
compute ONLY those blocks
```

That is exactly what I mean by masked SpGEMM.

---

# 4. This makes the matrix polynomial itself sparse

Consider generalized purification

$$
K' = 3KSK - 2KSKSK.
$$

Mathematically, every multiplication expands support.

But computationally we define

$$
P_M[X]
$$

after every operation:

$$
T=P_M(KS),
$$

$$
Q=P_M(TK),
$$

etc.

Thus the actual algorithm is something like

$$
T=P_{M_T}(KS),
$$

$$
Q=P_{M_K}(TK),
$$

$$
U=P_{M_T}(QS),
$$

$$
V=P_{M_K}(UK),
$$

$$
K'=3Q-2V.
$$

Every matrix remains \(O(Nz)\).

There is **never an \(N^2\) temporary**.

This is exactly the philosophy I think we should pursue.

---

# 5. I would actually use several predefined masks

We don't necessarily want every object to have the same cutoff.

For example:

### Physical short-range mask

For \(H\) and \(S\):

$$
M_{HS}: R<R_{\rm SK}.
$$

This is dictated by the Slater-Koster interaction.

Perhaps

```text
RHS ~ 4–6 Å
```

depending on parameterization.

### Density-kernel mask

$$
M_K:R<R_K.
$$

Maybe initially:

```text
RK = 8, 12, 16, 20 Å
```

for convergence testing.

### Intermediate-product mask

Potentially somewhat larger:

$$
M_T:R<R_T,
\qquad
R_T\gtrsim R_K.
$$

Because if we truncate

$$
KS
$$

too aggressively before multiplying by \(K\), we introduce unnecessary error.

So something like:

```text
H,S   : short
K     : medium
temp  : medium/slightly larger
```

is probably optimal.

But all remain linear-memory objects.

---

# 6. A priori geometric masks are preferable to dynamic thresholding initially

There are really two ways of declaring something zero.

### Geometric truncation

```text
if R_ij > RK:
    K_ij ≡ 0
```

This is extremely GPU-friendly.

The sparsity structure is known before the SCC solve:

```text
row_ptr[]
col_idx[]
```

never changes.

All buffers are allocated once.

No compaction, no prefix scan, no dynamic allocation, no changing load balance.

This should be our **first implementation**.

### Numerical dropping

Alternatively:

$$
\|K_{ij}\| < \epsilon
\quad\Rightarrow\quad
K_{ij}=0.
$$

This could potentially produce substantially sparser matrices.

But now sparsity changes during iteration:

```text
calculate blocks
↓
measure norm
↓
flag blocks
↓
prefix scan
↓
compact CSR
↓
rebuild row_ptr
```

That is much more complicated and has significant GPU overhead.

So I would begin with fixed geometry/graph masks, establish convergence versus \(R_K\), and add numerical dropping later if worthwhile.

---

# 7. Graph distance may actually be better than Euclidean radius in some cases

For covalent systems we can define the support by topological distance:

```text
K_ij exists if atoms i,j are within ≤ n bonds
```

rather than only spatial distance.

For perfect diamond/Si this gives extremely regular neighborhoods.

For graphene ribbons:

```text
1st shell
2nd shell
...
n-th shell
```

also gives highly predictable row lengths.

A combined rule might be even better:

$$
(i,j)\in M_K
\quad\Longleftrightarrow\quad
R_{ij}<R_K
$$

or graph distance \(<n_K\).

For MD where bonds remain mostly intact, the CSR topology could then remain unchanged for many timesteps.

That gives another major optimization: **reuse the exact CSR structure across MD steps.**

Only the block values change.

---

# 8. We can improve sparse multiplication because many matrices are symmetric

The first kernel I wrote searches for \(B_{kj}\) with a binary search in row \(k\).

But for symmetric matrices there is a better formulation.

For

$$
C_{ij}
=
\sum_k A_{ik}B_{kj}
$$

and symmetric \(B\),

$$
B_{kj}=B_{jk}^{T}.
$$

Therefore the contributing \(k\)'s are simply

$$
k\in N_A(i)\cap N_B(j).
$$

If the CSR neighbor lists are sorted, we can find that intersection with a classic two-pointer merge:

```c
ia = row_ptr_A[i];
ib = row_ptr_B[j];

while (ia < endA && ib < endB) {

    ka = A_col[ia];
    kb = B_col[ib];

    if      (ka < kb) ia++;
    else if (kb < ka) ib++;
    else {
        // same k:
        // C_ij += A_ik * transpose(B_jk)
        ia++;
        ib++;
    }
}
```

This is very attractive because it eliminates the repeated binary searches.

For symmetric products such as pieces of

$$
KSK
$$

we should exploit this wherever possible.

The intermediate

$$
T=KS
$$

is not symmetric in general, so the generic kernel is still needed, but quite a few operations can use specialized symmetric kernels.

---

# 9. We could even avoid some temporary matrices entirely

Your statement

> avoid any product matrices which naturally densify

can be pushed one step further.

Suppose we need

$$
Q=KSK.
$$

The simple implementation is:

```text
T = P(K*S)
Q = P(T*K)
```

requiring one temporary sparse matrix \(T\).

But we could calculate each desired \(Q_{ij}\) directly:

$$
Q_{ij}
=
\sum_{kl}
K_{ik}S_{kl}K_{lj}.
$$

That would avoid storing \(T\).

However it increases traversal/recomputation significantly.

So there is a classic GPU tradeoff:

```text
materialize sparse temporary:
    + less arithmetic
    + simpler
    - more global memory

fuse polynomial product:
    + less memory
    + potentially less global traffic
    - more repeated sparse traversal
    - more irregular work
```

Since your concern is specifically memory bandwidth and capacity, **fused sparse polynomial kernels are absolutely worth investigating later.**

But I would first establish the unfused sparse version because it is much easier to verify.

---

# 10. The GPU layout I now imagine

For one large system:

```text
static topology
─────────────────────────────────

HS_row_ptr
HS_col_idx

K_row_ptr
K_col_idx

T_row_ptr
T_col_idx

diag_slot[]

orbital block size = 4
```

Dynamic values:

```text
H0_val[HS_nblock][16]
S_val [HS_nblock][16]

H_val [HS_nblock][16]

K0_val[K_nblock][16]
K1_val[K_nblock][16]

T_val [T_nblock][16]

scratch_val[K/T_nblock][16]
```

Notice how **all indices can be immutable**.

The SCC loop changes only values.

That is precisely the kind of memory structure GPUs like.

---

# 11. For multiple large systems, concatenate the CSR rows

We can preserve the same multi-system architecture too.

Instead of separate allocations:

```text
system 0 CSR
system 1 CSR
system 2 CSR
```

we can concatenate:

```text
row_ptr global
col_idx global
values global

SystemDesc {
    atom_begin
    row_begin
    block_begin
}
```

or, for homogeneous nanocrystals, simply one regular stride.

Then a workgroup handles one atom row:

```c
row = get_group_id(0);
system = row_to_system[row];
```

For homogeneous copies,

```c
system = row / Natom;
atom   = row % Natom;
```

with zero lookup.

So the same idea of thousands of independent systems continues to work.

---

# 12. The key scientific test becomes convergence against the mask

I think we should explicitly treat sparsity radius as an approximation parameter just like grid spacing.

Run:

```text
RK = 6 Å
RK = 8 Å
RK = 10 Å
RK = 12 Å
RK = 16 Å
RK = 20 Å
```

and monitor:

$$
E(R_K)-E(R_{\rm ref}),
$$

$$
\mathrm{RMS}[q(R_K)-q_{\rm ref}],
$$

$$
\mathrm{RMS}[F(R_K)-F_{\rm ref}],
$$

$$
\|KSK-K\|,
$$

and memory/time.

Then we will get a very informative curve:

```text
error
│\
│ \
│  \
│   \________
│
└──────────────── RK
       ↑
       useful operating point
```

For diamond I expect this curve to become flat relatively quickly.

For silicon more slowly.

For graphene ribbons it will strongly depend on their actual gap and width.

And this lets us decide empirically whether a fixed geometric mask is sufficient or whether dynamic threshold sparsification is worth the complexity.

So yes: I now see the clean separation as **dense full-local Jacobi for small systems** and **atom-block CSR + explicitly projected sparse matrix polynomials for large localized systems**. They are not competing implementations so much as two regimes of the same DFTB engine.







---


# USER











---

# Chat GPT 5.6 sol

Yes. Below is a **self-contained pure OpenCL implementation** of the atom-block CSR route. No Python, no Rust harness, no dense matrices, no matrix inversion. I also recreated the file here:

[Download `sparse_bsr4_purification.cl`](sandbox:/mnt/data/sparse_bsr4_purification.cl)

The file is 818 lines because it contains more comments and optional LNV/diagnostic helpers. The core kernels are below so you are not dependent on the download working.

The most important representation is:

```c
row_ptr[Natom+1];   // explicit neighbor-list offsets
col_idx[Nblock];    // neighbor atom j
val[Nblock*16];     // packed 4x4 orbital block for atom pair (i,j)
```

and `col_idx` is sorted within each atom row.

### The intended purification schedule

For fixed masks `MS`, `MK`, `MT`, I would make the Rust harness enqueue this:

```text
MS : short H/S physical-neighbor mask
MK : density-kernel mask RK
MT : expanded temporary mask RT >= RK
```

For generalized McWeeny:

```text
T1[MT] = P_MT( K  * S )      bsr4_spgemm_masked_Bsym
Q [MK] = P_MK( T1 * K )      bsr4_spgemm_masked_Bsym

T2[MT] = P_MT( Q  * S )      bsr4_spgemm_masked_Bsym
V [MK] = P_MK( T2 * K )      bsr4_spgemm_masked_Bsym

Knew[MK] = 3*Q - 2*V         bsr4_mcweeny
```

Thus even though mathematically

$$
KSKSK
$$

would progressively densify, **no such dense matrix ever exists**. Each multiplication is immediately projected onto the requested block-CSR mask.

For TC2, once we have a physically valid Hamiltonian-derived starting \(K\):

```text
KS[MT] = P_MT( K  * S )      bsr4_spgemm_masked_Bsym
Q [MK] = P_MK( KS * K )      bsr4_spgemm_masked_Bsym

trace = Tr(KS)               bsr4_trace_KS_partial
                              reduce_sum_f32

Knew =
    Q                         if trace > Nocc
    2K-Q                      otherwise
```

There is no host readback required. `trace` remains a one-float GPU buffer and `bsr4_tc2()` reads it directly.

### Why I particularly like the `Bsym` kernel

The generic operation would do, for every \(A_{ik}\),

```text
search row k of B for j
```

which means many binary searches.

For symmetric \(B\),

$$
B_{kj}=B_{jk}^T,
$$

so instead we compare:

```text
neighbors(A row i)
neighbors(B row j)
```

and perform a standard sorted-list intersection:

```c
if      (ka < kb) ia++;
else if (kb < ka) ib++;
else               multiply block;
```

This should be substantially better for exactly our use case because essentially every right-hand matrix in the purification chain is either \(S\) or \(K\), both symmetric.

### One design detail I would keep rigid

I would **not dynamically rebuild CSR initially**.

Build, for example:

```text
MK(RK =  8 Å)
MK(RK = 12 Å)
MK(RK = 16 Å)
MK(RK = 20 Å)
```

and compare energies, charges and forces.

Blocks outside \(R_K\) literally do not exist:

```text
row_ptr
col_idx
values
```

contain no storage for them.

The `bsr4_drop_small()` kernel is intentionally only a diagnostic. It lets us ask whether additionally removing blocks with

$$
\|K_{ij}\|_F < \epsilon
$$

changes anything, without yet paying the complexity of GPU prefix sums, compaction and rebuilding `row_ptr`.

One next optimization I think is particularly interesting is to eliminate even the `T1 = KS` temporary by making a **direct fused `KSK` kernel** that walks the two-hop atom graph and writes only the requested `MK` blocks. That trades additional sparse traversal/arithmetic for substantially less global-memory traffic and one less \(O(Nz)\) temporary, which may be exactly the right trade on the RTX 3090 once the unfused version is verified.







---

# USER / GLM 5.2

OK, we implemented it based on your notes, there are some missing pieces 

```
What's not yet implemented (future work per the doc)
```

can you plese elaborat on those give agents detailed notes how to do it, you do not have to write code in detail (maybe just critical part if somethig is especially difficul;t), but provide explict equations and eventually pseudocode of algorithm, most imporetantly think it through carefully and critically to be sure we do it efficently and not in naive way, and we do not fogret something stupid.

\----

Continuing — running the fixed tests now.

8/9 pass. McWeeny now converges beautifully (2.6e-7). TC2 is genuinely diverging — the trace moves *away* from Nocc. This is the known TC2 limitation the chat doc explicitly warns about: "TC2 does not magically discover the Hamiltonian eigenvectors... the starting K0 must already be a spectral function of the generalized eigenproblem." My noise perturbation broke the spectral structure (eigenvalues of KS left the [0,1] basin).

The correct test: start from a **spectrally valid** K0 = α·K_exact (eigenvalues of KS are α and 0, both in [0,1]). TC2 should push α→1. Let me fix this:

All 9 tests pass. TC2 now converges cleanly (0.25 → 4e-6 in 4 steps, trace 2.4 → 2.99999). Let me clean up the unused import warning and mark complete.

9/9 pass, no warnings. Let me verify the full build is still clean.

All done. Here's a summary of what was implemented and the key findings.

## What was built

A self-contained `methods::sparse` module, independent of the `qmqm` dense/fragment solver, implementing the BSR4 sparse purification architecture from the chat doc:

**New files:**
- <ref_file file="/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/mod.rs" /> — module root
- <ref_file file="/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/bsr4.rs" /> — host BSR4 data structure + helpers (geometric/full masks, transpose map, diag map, host symmetrize, dense expand/matmul references)
- <ref_file file="/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/gpu_sparse.rs" /> — GPU wrapper (`SparseBsr4Gpu`) compiling `sparse_bsr4_purification.cl` with configurable `-D` defines, exposing both low-level buffer kernel launches and high-level `Bsr4Matrix`-in/out methods (`matmul_masked`, `matmul_masked_bsym`, `ksk`, `mcweeny_step`, `tc2_step`, `symmetrize_mat`, `mulliken`, `trace_ks`, `idempotency_err`)
- <ref_file file="/home/prokophapala/git/dftbplus/rust_dftb/tests/gpu_sparse_bsr4.rs" /> — 9 GPU tests vs dense CPU references

**Modified:**
- <ref_file file="/home/prokophapala/git/dftbplus/rust_dftb/src/methods/mod.rs" /> — registered `pub mod sparse`

## Test results: 9/9 pass

| Test | What it validates | Result |
|------|-------------------|--------|
| `spgemm_masked_vs_dense` | Generic masked SpGEMM vs dense matmul | 2.4e-7 |
| `spgemm_masked_bsym_vs_dense` | Symmetric-right SpGEMM (two-pointer intersection) | 2.4e-7 |
| `spgemm_mask_truncation` | Only masked blocks computed (10/16 blocks) | 2.4e-7 |
| `trace_ks_vs_cpu` | Tr(KS) partial reduction + recursive reduce | 2.4e-7 |
| `symmetrize_vs_host` | GPU symmetrize vs host reference + symmetry check | 0.0 |
| `mulliken_vs_cpu` | Mulliken charges q_A = 2·Tr((KS)_AA) | 1.2e-7 |
| `idempotency_exact_kernel` | K from CPU eigensolve: KSK=K, Tr(KS)=Nocc | 1.3e-7 |
| `mcweeny_convergence` | 3KSK−2KSKSK from perturbed K0 → idempotent | 2.6e-7 (5 steps) |
| `tc2_convergence` | Metric TC2 from α·K_exact → idempotent | 4.1e-6 (4 steps) |

## Key findings from testing

1. **The kernels are correct.** Both SpGEMM variants match dense CPU to f32 round-off (~2.4e-7). The Bsym two-pointer intersection is mathematically right (`B_kj = transpose(B_jk)`).

2. **TC2 has a real basin requirement.** The chat doc warns "TC2 does not magically discover the Hamiltonian eigenvectors" — I confirmed this empirically. Adding symmetric noise of 0.03 to an exact projector caused TC2 to diverge (trace → −335). The eigenvalues of KS must stay in [0,1]. Starting from α·K_exact (eigenvalues {α,0} ⊂ [0,1]) converges cleanly in 4 steps. **This means a real DFTB integration needs a Hamiltonian-derived initial K₀** (the S⁻¹ Newton-Schulz + `(εmax·Z − ZHZ)/(εmax−εmin)` route the doc describes) — reusing an arbitrary sparse K will not work.

3. **McWeeny is more robust** — converges from a noisy K₀ (0.93 → 2.6e-7 in 5 steps) because it's a fixed-point iteration that doesn't branch on trace direction.

4. **No second OpenCL context** — the module reuses the shared `GpuRuntime` for context/queue, compiling its own program with `WG`/`MAX_LEFT_BLOCKS`/`REDUCE_WG` defines.

## What's not yet implemented (future work per the doc)

- **Initial K₀ from H/S** (Newton-Schulz sparse S⁻¹ + spectral guess) — needed for real DFTB systems
- **Multi-mask support** (separate R_K, R_T, R_HS radii) — the API supports it (`k_mask`, `t_mask` params) but tests use full masks
- **Fused KSK kernel** (avoid the T=KS temporary) — the doc's suggested optimization
- **LNV gradient** kernels (`bsr4_lnv_gradient`, `bsr4_gradient_step`) — compiled but not yet wrapped/tested
- **`bsr4_drop_small`** diagnostic — compiled but not wrapped
- **SCC Hamiltonian update** (`bsr4_build_Hscc`) — compiled but not wrapped
- **Convergence vs R_K** benchmark (the doc's key scientific test: vary cutoff, monitor energy/charge/force error)






---


# Chat GPT 5.6 sol


# Sparse BSR4 DFTB: next implementation stages

The current kernels validate the sparse algebra, but do not yet constitute a real electronic solver. Implement in this order:

**P0:** multi-mask topology → sparse \(K_0(H,S)\) → real SCC integration → end-to-end dense-reference tests.
**P1:** convergence-vs-cutoff study → fused `KSK` → LNV solver.
**P2:** numerical dropping/dynamic sparsity and further performance refinements.

Do not optimize fused products before the P0 chain works.

---

## 0. Add a third convergence criterion: Hamiltonian stationarity

Current tests monitor

$$
R_I=\|KSK-K\|
$$

and

$$
R_N=|\operatorname{Tr}(KS)-N_{\rm occ}|.
$$

These are necessary but not sufficient. Any projector onto the wrong subspace can satisfy both exactly.

For the ground-state projector we also require compatibility with \(H\). An appropriate nonorthogonal commutator is

$$
\boxed{R_H=\|HKS-SKH\|.}
$$

For exact occupied eigenvectors,

$$
HC=SC\varepsilon,\qquad K=C_{\rm occ}C_{\rm occ}^T
$$

and therefore

$$
HKS=SKH.
$$

Add this as a diagnostic. It is particularly important because:

```text
McWeeny(noisy projector) -> idempotent projector
                         != necessarily projector of H
```

Rename/reinterpret the existing McWeeny test accordingly: it tests purification/idempotency, not solution of the Hamiltonian.

A real solver test must compare the final sparse \(K\) with the projector obtained from a dense generalized eigensolve of the same \(H,S\).

---

# 1. Multi-mask support

Use three principal CSR patterns:

$$
M_{HS}: H,S
$$

$$
M_K: K,\;Q,\;\text{density-like matrices}
$$

$$
M_T: KS,\;QS,\;\text{temporary expanded products}.
$$

`M_HS` is determined by the Slater-Koster interaction range. `M_K` is the chosen density-kernel localization region.

### Do not initially choose \(M_T\) merely as an arbitrary radius

A better construction is the **boolean structural product**

$$
\boxed{M_T=M_K\circ M_{HS}}
$$

meaning that block \((i,j)\) exists in \(M_T\) iff there is at least one atom \(k\) such that

$$
(i,k)\in M_K,\qquad(k,j)\in M_{HS}.
$$

This is exactly the possible support of

$$
T=KS
$$

given the already truncated \(K\) and physical \(S\).

Build this mask once on CPU:

```text
for atom i:
    set = {}
    for k in K.neighbors(i):
        for j in S.neighbors(k):
            set.insert(j)
    sort(set)
    T.neighbors(i)=set
```

This is better than blindly taking \(R_T=R_K+R_S\): it contains every structurally possible contribution but no blocks that cannot occur because of the actual atomic graph.

For moving atoms, eventually build masks with a geometric skin, analogous to an MD Verlet list, and rebuild only when displacement exceeds half the skin.

Every mask needs independently:

```text
row_ptr
col_idx
diag_block
transpose_block
```

`M_T` need not be symmetric because \(KS\) is not symmetric.

### Tests

Generate random symmetric `K` on `M_K` and `S` on `M_HS`.

Check:

$$
P_{M_T}(KS)
$$

against dense multiplication, and then

$$
P_{M_K}[(KS)K].
$$

Also deliberately omit one structurally required block from `M_T` and verify that the reference test detects the error.

---

# 2. Construct a Hamiltonian-derived \(K_0\)

This is the most important missing piece.

For TC2/SP2, we need the eigenvalues of \(K_0S\) in approximately \([0,1]\) and its eigenvectors to be those of the current generalized Hamiltonian.

The generalized problem is

$$
HC=S C\varepsilon.
$$

Let

$$
Z\approx S^{-1}.
$$

Then define

$$
\boxed{
K_0=
\frac{\varepsilon_{\max}Z-ZHZ}
{\varepsilon_{\max}-\varepsilon_{\min}}
}
$$

because, for exact \(Z=S^{-1}\),

$$
K_0S=
\frac{\varepsilon_{\max}I-ZH}
{\varepsilon_{\max}-\varepsilon_{\min}}.
$$

Therefore a generalized eigenvector with energy \(\varepsilon\) gets initial occupation eigenvalue

$$
x_0(\varepsilon)=
\frac{\varepsilon_{\max}-\varepsilon}
{\varepsilon_{\max}-\varepsilon_{\min}}
\in[0,1].
$$

Now TC2 can safely push these eigenvalues toward 0 or 1.

## 2a. Sparse inverse \(Z\)

Compute \(Z\) **once per geometry**, not once per SCC iteration.

Use sparse Newton-Schulz/Hotelling:

$$
\boxed{
Z_{n+1}=2Z_n-Z_nSZ_n.
}
$$

Use `M_K` initially also for \(Z\). We can introduce a smaller dedicated \(M_Z\) later if useful.

A safe simple initialization is

$$
Z_0=\alpha I,\qquad 0<\alpha<2/\lambda_{\max}(S).
$$

Use a conservative bound

$$
u=\|S\|_\infty=\max_\mu\sum_\nu|S_{\mu\nu}|
$$

and simply choose

$$
\boxed{\alpha=1/u.}
$$

No need to optimize this initially; convergence is more important.

Iteration:

```text
T[M_T] = Z * S
Q[M_K] = T * Z
Znew    = 2 Z - Q
symmetrize(Znew)
swap
```

Monitor

$$
R_Z=\|I-ZS\|_F/\sqrt{N_{\rm orb}}.
$$

Do not iterate forever. With fixed sparse truncation, \(R_Z\) will eventually hit a truncation floor. Stop when either the tolerance is reached or the improvement stalls, e.g.

$$
R_Z^{(n+1)}>0.9R_Z^{(n)}
$$

for several iterations.

A rapidly diverging/stagnating inverse is an important physical/numerical warning: either the overlap is poorly conditioned or the inverse-localization mask is too short.

### Important for H-passivated systems

Current BSR4 naturally covers C/Si, but graphene ribbons will often contain H.

Do not silently assume every atom has four real orbitals.

Either initially assert

```text
all atoms have exactly 4 orbitals
```

or support `norb_atom[A]`.

Padded 4×4 storage is still possible: inactive rows/columns remain zero, and the identity used in Newton-Schulz has ones only on active orbitals. Trace and residual kernels must also ignore inactive orbitals.

Do not put fake unit-overlap dummy orbitals into the physics unless deliberately designing such an embedding.

---

## 2b. Build \(K_0\) for the current Hamiltonian

For the first correct implementation:

```text
B[M_T] = Z * H
A[M_K] = B * Z      // A = Z H Z
K0      = (emax*Z - A)/(emax-emin)
symmetrize(K0)
```

Then TC2:

```text
K = K0
repeat:
    KS[M_T] = K*S
    Q[M_K]  = KS*K
    n        = Tr(KS)

    if n > Nocc: Knew=Q
    else:        Knew=2K-Q

    optional symmetrize
    monitor idempotency + trace
```

The current test should then be replaced by a real test:

```text
random physically sensible H,S
dense generalized eigensolve -> K_ref
sparse/full-mask K0 -> TC2 -> K
compare K vs K_ref
```

Use full masks initially so truncation cannot hide algorithmic errors.

---

## 2c. Spectral bounds \(\varepsilon_{\min},\varepsilon_{\max}\)

Wrong bounds are dangerous because eigenvalues of \(K_0S\) can leave \([0,1]\).

For the first implementation, use conservative bounds from

$$
B=ZH.
$$

The simplest absolutely safe estimate is

$$
R=\|B\|_\infty
$$

and

$$
\varepsilon_{\min}=-R,\qquad\varepsilon_{\max}=+R.
$$

This may be loose but is robust.

A tighter Gershgorin estimate is

$$
\varepsilon_{\min}=\min_i\left(B_{ii}-\sum_{j\ne i}|B_{ij}|\right),
$$

$$
\varepsilon_{\max}=\max_i\left(B_{ii}+\sum_{j\ne i}|B_{ij}|\right).
$$

Operate at the **orbital**, not merely atom-block, level for these row sums.

Add perhaps 5–10% padding.

Loose bounds only increase purification iterations; too-tight bounds can destroy convergence.

Do not optimize this until the end-to-end solver works.

---

# 3. Important SCC optimization once the simple \(K_0\) works

DFTB SCC has

$$
H=H^0+\frac12(SV+VS),
$$

where \(V\) is diagonal in orbital space and all orbitals on atom \(A\) share \(V_A\).

The naive \(K_0\) construction computes

$$
ZHZ
$$

with two sparse multiplications every SCC iteration.

But with exact \(Z=S^{-1}\),

$$
\begin{aligned}
ZHZ
&=ZH^0Z+\frac12(ZSVZ+ZVSZ)\\
&=ZH^0Z+\frac12(VZ+ZV).
\end{aligned}
$$

Therefore precompute once per geometry

$$
A^0=ZH^0Z,
$$

and during SCC construct

$$
\boxed{
A_{AB}=A^0_{AB}
+\frac12(V_A+V_B)Z_{AB}.
}
$$

This is only an elementwise block update, essentially the same cost as `build_Hscc`.

Then

$$
K_0=\frac{\varepsilon_{\max}Z-A}{\Delta\varepsilon}.
$$

This can remove two large sparse matrix products from every SCC iteration.

**Do not use this optimization before verifying the explicit `Z*H*Z` path.** Because \(Z\) is truncated and only approximately \(S^{-1}\), compare both paths numerically and quantify the additional error.

---

# 4. Wrap and integrate `bsr4_build_Hscc`

For atom blocks \(A,B\),

$$
\boxed{
H_{AB}=H^0_{AB}
+\frac12(V_A+V_B)S_{AB}.
}
$$

First compute

$$
\Delta q_A=q_A-q_A^0
$$

using exactly the same sign convention as the existing dense/CPU DFTB implementation. Do not infer the sign from the sparse code.

Then

$$
V_A=\sum_B\gamma_{AB}\Delta q_B.
$$

The complete SCC pipeline should become:

```text
q
-> dq
-> V = gamma*dq
-> Hscc
-> K0(Hscc,S)
-> TC2
-> KS
-> q_new = Mulliken(KS)
-> mixer
-> repeat
```

No host readback is necessary except optional diagnostics.

The first integration target should be **identical SCC energies and charges to the dense CPU solver for small systems with full masks**.

### Do not forget SCC energy double counting

The electronic solver producing the correct density is not enough. Port the total-energy expression from the existing CPU implementation exactly.

Do not derive a fresh sign convention independently.

Add parity tests for:

```text
band/electronic contribution
SCC charge contribution
repulsive contribution
total energy
```

before attempting forces.

### Long-range gamma remains separate

Sparse \(H,S,K\) does not make

$$
V_A=\sum_B\gamma_{AB}\Delta q_B
$$

short ranged: asymptotically \(\gamma_{AB}\sim1/R\).

Direct \(N^2\) is fine initially. For genuinely large systems this will eventually need its own tree/FMM or mesh treatment.

Do not complicate the sparse density solver with this yet, but keep the interface separate.

---

# 5. Fused `KSK`: do not implement a naive triple sparse loop

Baseline:

```text
T = K*S
Q = T*K
```

requires a global `T` buffer and one write + subsequent read of every block.

A naive fused implementation

$$
Q_{ij}=\sum_{kl}K_{ik}S_{kl}K_{lj}
$$

would save memory but repeat large amounts of work. Do not do that.

The useful fused implementation is **row-local materialization**.

For one atom row \(i\):

1. load/cache `K.row(i)`;
2. construct the complete sparse row \(T_i=(KS)_i\) directly into local memory using the static `M_T` row;
3. barrier;
4. compute all desired \(Q_{ij}\) from local `T_i` and symmetric `K`;
5. write only `Q_i`;
6. discard `T_i`.

Pseudocode:

```text
WG handles atom row i

local Krow[zK][16]
local Trow[zT][16]

load Krow

parallel over l in T_mask.row(i):
    T_il = sum_k K_ik S_kl
barrier

parallel over j in K_mask.row(i):
    Q_ij = sum_l T_il K_lj

write Q row
```

The second sum uses symmetry

$$
K_{lj}=K_{jl}^T
$$

and a sorted intersection between `T_col(i)` and `K_col(j)`.

This has approximately the same arithmetic as the two-kernel implementation but eliminates the global `T` buffer traffic.

### Local-memory condition

Peak local value storage is approximately

$$
64(z_K+z_T)\ {\rm bytes}
$$

plus indices.

Examples:

$$
z_K=128,\;z_T=160
\Rightarrow18.4\ {\rm KiB}
$$

for values, excellent.

For much larger 3D neighborhoods it may stop fitting.

Therefore implement degree-specialized paths:

```text
small rows  -> fused local KSK
large rows  -> existing two-kernel K*S ; T*K
```

Possible compile variants:

```text
MAX_K_ROW = 64/128/256
MAX_T_ROW = 96/192/384
```

Benchmark rather than assuming fusion is always faster. Local-memory usage can reduce occupancy.

The purpose of fusion is primarily:

```text
less global traffic
less temporary memory
```

not fewer FLOPs.

---

# 6. LNV: wrap it only after TC2 works end-to-end

LNV is valuable because, unlike pure purification of an existing \(K\), it explicitly depends on \(H\) and can rotate the occupied subspace.

Define

$$
K(L)=3LSL-2LSLSL
$$

and

$$
F=H-\mu S.
$$

Grand-potential functional:

$$
\Omega[L]=2\operatorname{Tr}[K(L)F].
$$

For symmetric \(L,S,F\),

$$
\frac12\frac{\partial\Omega}{\partial L}
=
3(SLF+FLS)
-2(SLSLF+SLFLS+FLSLS).
$$

Before making a production optimizer, first validate the existing gradient kernel by finite differences on tiny dense/full-mask systems:

$$
\frac{\Omega(L+\epsilon D)-\Omega(L-\epsilon D)}{2\epsilon}
\approx
\langle G,D\rangle.
$$

This must pass before optimization.

## Reduce the number of sparse products using symmetry

Do not independently construct all five gradient terms.

Set

$$
A=SL,\qquad P=AF=SLF.
$$

Because all three basic matrices are symmetric,

$$
FLS=P^T.
$$

Now

$$
C=AP=SLSLF,
$$

$$
C^T=FLSLS,
$$

and

$$
D=P A^T=SLFLS.
$$

Therefore

$$
\boxed{
G/2=3(P+P^T)-2(C+C^T+D).
}
$$

So the gradient needs essentially:

```text
A = S*L
P = A*F
C = A*P
D = P*A^T
G = 3(P+P^T)-2(C+C^T+D)
```

rather than separately evaluating every expression.

Use transpose maps rather than materializing transposes where possible.

## Electron number is the difficult part

A fixed-\(\mu\) LNV functional is grand-canonical. DFTB normally requires fixed electron number.

Do **not** integrate LNV into production merely by guessing \(\mu\).

For the first LNV test, use a small system where the dense eigensolver gives a known \(\mu\) inside the HOMO-LUMO gap. Verify that LNV then converges to the correct projector.

Only afterwards choose a fixed-\(N\) strategy: outer \(\mu\) adjustment, constrained minimization, or another canonical formulation.

Thus LNV is an alternative solver branch, not currently a replacement for the simpler TC2 route.

---

# 7. `drop_small`: diagnostic first, not optimization

Wrap it now only to answer the scientific question:

$$
\text{How many blocks are numerically negligible?}
$$

Use block Frobenius norm

$$
n_{ij}^2=\sum_{\mu\in i,\nu\in j}|K_{\mu\nu}|^2.
$$

But dropping must preserve symmetry. Do not independently drop \((i,j)\) and \((j,i)\).

Use the transpose map and decide once per pair, for example from

$$
n^2=\frac12(\|K_{ij}\|_F^2+\|K_{ji}\|_F^2).
$$

Never drop diagonal blocks.

Return diagnostics:

```text
number/fraction of blocks below threshold
sum norm² of dropped blocks
largest dropped norm
distribution vs distance
```

Test thresholds such as

$$
10^{-3},10^{-4},10^{-5},10^{-6}
$$

in the natural matrix units.

For now zero values but leave CSR unchanged. This gives no memory saving and little performance benefit, but tells us whether true dynamic sparsification is worth implementing.

Actual CSR compaction can come much later because it requires:

```text
mark
prefix scan
compact col/value
rebuild row_ptr
rebuild transpose/diag maps
```

and should probably happen only occasionally, not every purification iteration.

---

# 8. Convergence versus \(R_K\): do this before claiming linear-scaling accuracy

There are two distinct truncation errors:

$$
R_K:\text{ truncation of the physical density kernel}
$$

and

$$
M_T:\text{ truncation of intermediate products}.
$$

Do not mix them.

### Stage A: verify intermediate support

Fix a sufficiently large \(R_K\).

Compare:

```text
T mask = exact boolean product MK ∘ MHS
```

against progressively shorter manually chosen \(R_T\).

This tells us how much error comes merely from truncating intermediate products.

The boolean-product mask should be the reference for a given \(M_K,M_{HS}\).

### Stage B: vary \(R_K\)

For small systems where dense reference exists:

$$
R_K=4,6,8,10,12,16,\ldots
$$

until the mask becomes effectively full.

Measure:

$$
\Delta E/N,
$$

$$
\mathrm{RMS}(\Delta q_A),\qquad\max|\Delta q_A|,
$$

$$
R_I=\|KSK-K\|,
$$

$$
R_N=|\operatorname{Tr}(KS)-N_{\rm occ}|,
$$

$$
\boxed{R_H=\|HKS-SKH\|}
$$

and, when implemented,

$$
\mathrm{RMS}(\Delta F),\qquad\max|\Delta F|.
$$

Also record algorithmic quantities:

```text
nblocks(MHS), nblocks(MK), nblocks(MT)
average/max blocks per row
memory in MB
Newton-Schulz iterations
TC2 iterations
time per SpGEMM
total electronic-solve time
```

### Test geometries

Use at least three qualitatively different cases:

```text
diamond nanocrystal     // large gap, 3D
Si nanocrystal          // smaller gap, 3D
graphene nanoribbon     // smaller gap but quasi-1D/2D
```

The latter is important because geometric sparsity may compensate significantly for the longer density-matrix decay.

First test a fixed non-SCC Hamiltonian. Only then enable SCC. Otherwise SCC feedback can obscure where an error originates.

---

# 9. The real end-to-end correctness test

Before performance work, construct a system small enough for both solvers and run:

```text
same geometry
same H0,S,gamma,q0
same electron count
```

Dense path:

```text
SCC -> generalized eigensolve -> q -> mix
```

Sparse path:

```text
SCC -> Z
    -> K0(H,S)
    -> TC2
    -> q
    -> mix
```

Initially use full `M_K` so there is **zero locality approximation**.

Require agreement in:

```text
SCC charges
total energy
density kernel
band populations
SCC iteration count/order of magnitude
```

Only after this passes should geometric truncation be enabled.

That cleanly separates:

```text
solver correctness
from
locality approximation
from
f32 error
```

which will otherwise be difficult to debug.

---

# 10. Things easy to forget

### TC2 cannot use the previous SCC/MD density as its only starting point

If \(H\) changes, applying only polynomials to old \(K\) cannot rotate its eigenvectors.

For TC2/SP2:

$$
K_0=K_0(H_{\rm current})
$$

must be regenerated.

Previous \(K\) may eventually be useful in a more sophisticated predictor/corrector scheme, but not as a direct replacement for the Hamiltonian-derived initial matrix.

LNV does not have this restriction because its gradient contains \(H\).

### Forces need more than \(K\)

At zero temperature the energy-weighted density can be obtained without eigenvectors:

$$
\boxed{W=KHK.}
$$

Thus sparse forces remain possible.

Eventually the nonorthogonal electronic force contains the usual structure

$$
2\operatorname{Tr}\left(K\frac{\partial H}{\partial R}\right)
-2\operatorname{Tr}\left(W\frac{\partial S}{\partial R}\right)
$$

plus SCC and repulsive terms.

Do not design any density representation that makes construction of sparse \(W\) impossible.

### Small-gap/metallic systems

Zero-temperature projector locality becomes progressively worse as the gap closes; finite electronic temperature restores stronger locality. This is a later reason to consider finite-temperature Fermi-operator expansion for nearly metallic graphene rather than forcing zero-T TC2 indefinitely.

### BSR4 and hydrogen

Graphene ribbons frequently need H-passivated edges. Either explicitly postpone mixed orbital counts or make the padded BSR4 algebra aware of active orbital counts. Do not discover this only after the sparse solver is otherwise finished.

### Periodic systems

If the first sparse branch assumes finite clusters/Γ-point real matrices, assert that explicitly. General \(k\)-point DFTB introduces complex blocks and is a separate extension.

---

# Recommended concrete implementation sequence

1. Implement boolean multi-mask construction and tests.
2. Implement sparse Newton-Schulz \(Z\approx S^{-1}\), residual and tests.
3. Implement `ZH`, `ZHZ`, spectral bounds, Hamiltonian-derived \(K_0\).
4. Replace the current TC2 toy test by `H,S -> K0 -> TC2 -> dense projector` parity test.
5. Add Hamiltonian commutator residual \(HKS-SKH\).
6. Wrap `build_Hscc` and integrate one complete sparse SCC calculation using full masks.
7. Match dense CPU charges and energies.
8. Enable finite \(R_K\) and perform systematic convergence tests.
9. Only now benchmark/implement fused row-local `KSK`.
10. Validate LNV gradient and explore LNV as the H-dependent/warm-start solver.
11. Use `drop_small` to decide whether dynamic sparsity is worth the complexity.
