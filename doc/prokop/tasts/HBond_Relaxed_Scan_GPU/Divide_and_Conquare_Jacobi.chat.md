https://chatgpt.com/share/6aad4918-2890-83eb-9c56-80358b671f83

---

# USER

look in here how we do jacobi rotations
https://github.com/ProkopHapala/dftbplus/tree/main/rust_dftb/src/qmqm

her we have some nested solver
https://github.com/ProkopHapala/NumericalMathPlayground/blob/main/topics/LinearAlgebra/FastDirectSolvers/nested_solver.py

We are constantly figting with tradeoff - either we use too much group local shared memory (48kb limit per compute unit on nvidia) or we need many global memory reads which kill the performance. 


But what I wanted to implment and I'm not sure if it was ever implemented.

1) when you do some jacobi rotation, you can take some blokc e.g. 8x8 which you want to nullyfy, you do this and apply same trasnfrom on the remaning adjecent colum/row factors (same rotation must beapplied on whole collum/row block

this should not require so much local memory as you work just with the 8x8 tile, and then you only stream the result to the addejcent rows/vectros

2) this way we can implement divide-and-conquare, if we nulify some ofdiagonal block, the matrix effectively decompose to two indepenendt matrixes. Then we can process those cheaply. 

I don;t know if we already implemented and in which way but is seems to me this desing should be feaster than what we currectly have. 


---

# Chat GPT 5.6 sol

I checked both repositories, and the answer is quite interesting: **your idea (1) is already implemented almost exactly**, while idea (2) is the genuinely new—and mathematically trickier—part.

In `rust_dftb/src/qmqm/gpu_block_jacobi.cl`, `block_jacobi_1wg` does precisely this:

* take two blocks `bp,bq`, each of size `B`;
* gather the compound `2B × 2B` pivot into local memory `P`;
* diagonalize that little matrix locally, accumulating its transformation in local `U`;
* then stream through the rest of the matrix, applying that same `U` to the corresponding columns/rows of `A`;
* simultaneously apply `U` to the corresponding columns of the eigenvector matrix `V`.

The core is literally:

```text
P = A[{p,q},{p,q}]
P <- Uᵀ P U          // local Jacobi

A[:,{p,q}] <- A[:,{p,q}] U
A[{p,q},:] <- Uᵀ A[{p,q},:]
V[:,{p,q}] <- V[:,{p,q}] U
```

So conceptually it is exactly the architecture you were describing.

And this was already benchmarked/tuned quite extensively. The current default is actually **B=32**, so the local pivot is `64×64`; the code comments say this costs about **34 KB/WG**. The old `B=16` version costs only about **8.4 KB/WG**, but requires many more block-pair operations: for `N=246`, the comment explicitly gives **120 pairs/sweep for B=16 versus 28 for B=32**. Empirically B=32 with only **one inner Jacobi sweep per pivot** was best. The important insight from those tests was:

> fully diagonalizing each local pivot is wasteful because later block rotations disturb it again.

That observation is actually directly relevant to your divide-and-conquer idea.

## Why your point (2) does not automatically follow

Suppose

$$
A=
\begin{pmatrix}
A_{11} & B\\
B^T & A_{22}
\end{pmatrix}.
$$

If we could find an orthogonal transformation \(Q\) such that

$$
Q^T A Q=
\begin{pmatrix}
D_1&0\\
0&D_2
\end{pmatrix},
$$

then yes: from that point onward, `D1` and `D2` can be solved completely independently. And recursively doing this would be a beautiful GPU algorithm:

```text
N
├── N/2
│   ├── N/4
│   └── N/4
└── N/2
    ├── N/4
    └── N/4
```

with ever smaller local-memory problems.

But the problem is that an ordinary Jacobi transformation of one small `8×8` or `64×64` pivot **does not permanently zero a macroscopic off-diagonal block**.

Imagine three block groups `A,B,C`. You diagonalize the `(A,B)` pivot and make

$$
H_{AB}=0.
$$

Then you rotate `(A,C)`:

$$
[A,C] \rightarrow [A,C]U .
$$

That rotation changes the basis of `A`. Since `B` was expressed relative to the old `A` basis, in general this creates

$$
H_{AB}\neq0
$$

again.

That is exactly why the current algorithm revisits all block pairs sweep after sweep.

So:

> **A Jacobi-zeroed tile is not an invariant separator.**

You may freeze the split only after you have constructed an **invariant subspace**, not merely reduced one particular off-diagonal tile.

---

# But I think there is a very promising variant of your idea

I would separate two concepts.

### Current algorithm: spatial/block Jacobi

For every block pair:

```text
[p p]  [p q]
[q p]  [q q]
```

solve the small `2B` problem and propagate its transformation through all `N` rows.

That costs approximately

$$
O(N B)
$$

global-memory work per block pair.

There are approximately

$$
\frac12(N/B)^2
$$

pairs per sweep, therefore roughly

$$
O(N^3/B)
$$

streamed work per global sweep.

That is exactly where the memory traffic comes from.

For `N=246,B=32`, there are only 8 blocks → 28 pairs, which explains why B=32 wins despite consuming 34 KB local memory.

---

# Hierarchical Jacobi would be fundamentally different

Your recursive idea could instead operate on **adjacent groups of increasing size**:

```text
level 0:
[8][8] [8][8] [8][8] [8][8] ...

level 1:
[ 16 ][ 16 ] [ 16 ][ 16 ]

level 2:
[    32    ][    32    ]

level 3:
[          64          ]
```

But at each merge we need a transformation which genuinely separates the two child invariant subspaces.

That is no longer simply:

> diagonalize the interface `8×8` block.

Instead it becomes essentially a **subspace diagonalization / spectral divide-and-conquer problem**.

There are several mathematical ways to formulate that.

One particularly interesting one is to solve the block Riccati problem. For

$$
H=
\begin{pmatrix}
A&B\\
B^T&C
\end{pmatrix},
$$

look for a transformation parametrized by \(X\) that block-diagonalizes \(H\). \(X\) approximately satisfies

$$
B + AX-XC-XB^TX =0.
$$

For weakly coupled blocks, first order gives the Sylvester equation

$$
AX-XC=-B.
$$

Once \(X\) is known, you can build an orthogonal transformation that actually eliminates the inter-block coupling.

Then the split is permanent.

That really would produce true divide-and-conquer.

But it is considerably more mathematics than ordinary Jacobi.

---

# The `nested_solver.py` code is related, but not the same algorithm

Your NumericalMathPlayground code actually explored another branch of this idea.

`recursive_exact_amls()` does:

1. solve child blocks;
2. assemble their eigenvector bases into a block-diagonal transformation;
3. transform the parent:

   $$
   H' = V_\text{child}^T H V_\text{child};
   $$
4. diagonalize the parent reduced problem;
5. recurse upward.

And `ritz_correction_from_blocks()` similarly diagonalizes blocks independently and then restores inter-block coupling through

$$
H_{\rm proj}=V^THV.
$$

So you did experiment with hierarchical/local eigenspaces.

But this does **not reduce complexity if all modes are retained**. If \(V\) contains every child eigenvector, the parent problem has essentially the same total size. It is just a change of basis.

It becomes efficient only if either:

* the coupling is low rank / localized, or
* you discard irrelevant modes, or
* the transformation manages to create genuinely independent invariant subspaces.

For your dense DFTB Hamiltonian eigensolver, we need the last option if we want an **exact** replacement.

---

# There is nevertheless another optimization I would try before inventing the recursive solver

Looking carefully at your present block kernel, I think **global-memory traffic is now a more interesting target than local memory**.

The current `block_jacobi_1wg` performs, for every pivot pair, something roughly like:

```text
for every row r:
    load A[r,p:q]
    calculate A[r,p:q] U
    write A[r,p:q]
    write symmetric A[p:q,r]

    load V[r,p:q]
    calculate V[r,p:q] U
    write V[r,p:q]
```

So every pair walks effectively through the entire matrix.

For `N=246,B=32`, that's 28 whole-matrix strip transformations per sweep.

And I noticed something particularly wasteful in the current kernel:

```c
float x[PB];

for(j...)
    x[j] = gA[r*n+cj];

if(il >= 0) {
    // x isn't actually used!
    ...
} else {
    ...
}
```

The pivot rows load their global strips even though `P` already contains the result.

That's minor relative to everything else, but illustrates where the current bottleneck lives.

---

## A more important optimization: accumulate several block transformations

Suppose during a stage you generate rotations

$$
U_{01}, U_{23}, U_{45}, U_{67}
$$

between **disjoint block pairs**.

Instead of immediately streaming the entire `N×N` matrix after every pair, these can be regarded as one block-diagonal transformation

$$
Q=
\operatorname{diag}
(U_{01},U_{23},U_{45},U_{67}).
$$

Then perform

$$
A\leftarrow Q^TAQ
$$

as a tiled matrix transformation.

Now a tile of `A` can be loaded once, transformed on both sides, and written once.

This is much closer to a BLAS-3 operation.

That may be the really valuable middle ground between your current Jacobi and full divide-and-conquer.

Instead of:

```text
pivot 0-1
stream whole matrix

pivot 0-2
stream whole matrix

pivot 0-3
stream whole matrix
...
```

use a Brent-Luk-like **block pairing schedule**:

```text
round 0:
(0,7) (1,6) (2,5) (3,4)

round 1:
(0,6) (7,5) (1,4) (2,3)

...
```

All pairs in a round are disjoint.

Generate all little `U`s first.

Then apply the whole round transformation to `A` in a tiled kernel.

This changes the important unit of memory traffic from:

> one entire matrix stream **per block pair**

toward:

> one/few entire matrix streams **per block-pair round**.

For \(n_b=N/B\) blocks:

Current sequential pairs:

$$
N_\text{pair}\sim n_b(n_b-1)/2.
$$

Parallel pairing rounds:

$$
N_\text{round}\sim n_b-1.
$$

For your `N=246,B=32`, \(n_b=8\):

```text
current:    28 pair transformations
round form: 7 matrix transformation stages
```

Potentially a roughly **4× reduction in the number of global streaming phases**, before considering cache reuse.

For B=16:

```text
nb = 16
pairs  = 120
rounds = 15
```

which is an **8× structural reduction**.

Suddenly B=16 becomes much more attractive because its ~8 KB shared-memory footprint would allow much better occupancy.

This is the direction I would investigate first.

---

# And it connects beautifully to your original intuition

You were imagining:

> solve a little tile locally, then apply the transformation to adjacent vectors.

I would modify that slightly:

> **solve several independent little tiles locally, accumulate their transformations into a block-diagonal Q, then stream large matrix tiles through Q once.**

Then local memory only needs perhaps:

```text
P : 2B × 2B
U : 2B × 2B
A_tile : maybe 32×32 or 64×something
```

per workgroup.

You don't need the full matrix resident.

And global reads become much more GEMM-like and coalesced.

---

# One subtlety: updating A on both sides

Rather than your present row-owner code:

```text
A[:,I] = A[:,I] U
A[I,:] = transpose
```

I would think in matrix tiles.

For a pair of block transformations \(U_i,U_j\),

$$
A_{ij}' = U_i^T A_{ij} U_j.
$$

That is ideal GPU work.

For a block tile `Aij`:

```text
load Aij to shared/registers
Aij = Ui^T * Aij * Uj
store Aij
```

Each block tile is touched once per round.

And symmetry means only the upper/lower triangular half need actually be stored/computed if we are willing to adapt downstream accesses.

Likewise eigenvectors require only

$$
V_i' = V_i U_i,
$$

which is again a block GEMM.

This is far more bandwidth-friendly than applying every little Jacobi transform immediately.

---

# So I would rank the ideas like this

**Already implemented:** your `2B×2B local pivot + stream adjacent rows` idea. The current `block_jacobi_1wg` is exactly that.

**Worth implementing next:** change the block Jacobi scheduling from sequential block pairs to **disjoint block-pair rounds**, and apply the entire round transformation tile-wise. I think this has a very realistic chance of beating the present B=32 implementation.

**Interesting but research-level:** true recursive divide-and-conquer where a zeroed cross block is frozen permanently. This requires finding invariant subspaces / block diagonalization; ordinary Jacobi tile annihilation doesn't give you that property.

**Not likely useful by itself:** the existing `nested_solver.py` Ritz/AMLS concept with all modes retained. It rearranges the eigensolve but does not eliminate the large parent problem.

And importantly, this conclusion matches your measured code remarkably well: the repository already found that **doing a highly accurate local pivot solve is counterproductive because later pairs undo part of it**. That is almost a direct experimental demonstration of why naïvely treating a zeroed local block as a permanent divide-and-conquer boundary will not work.

The next experiment I would make is therefore quite focused: **keep the existing block-Jacobi mathematics and buffers, but prototype an 8-block Brent-Luk block schedule with `B=16` or `B=32`, store one `U` per block-pair for the round, and apply `U_iᵀ A_ij U_j` tilewise.** That lets us test the memory-traffic hypothesis without replacing the eigensolver or risking the SCC architecture.




---




# USER

what means `2B × 2B` ? 2x2 block ? or what is B? why 2B\*2B?

We must be very wise about what is worth to store in local memory and what is not. When we try to nulify some block, we may need multiple sweeps, for that it is worth to keep it in memory. But we do not need to keep in memory whole adjecent row and colum. We can recoed the initary matrix U in some separate memory block and the apply it on these columsn/rows in global memory (ebause it is much cheaper once-for-alloperation)

then we should not repeate this in many sweeps, re-reading the blocks every sweep. Once the block is nullyfied, we must make sure the further operation will never touch t (not desifying it again), that is the principle of divide and conquare. 

the memory advantage is therefore both - we do not store the adjectent row/colum blocks, and most importaintly we do not re-read the matrix every sweep.

Is this understandable? Is this viable algorth,? is this something described in litarature? Do you see some major chalanges?





---

# Chat GPT 56. sol

Yes. And first, the notation:

$$
2B\times 2B
$$

does **not** mean a `2×2` numerical matrix. In the current code, `B` is the size of one block. For example, with `B=16`, two blocks \(p,q\) give the compound pivot

$$
P=
\begin{pmatrix}
A_{pp} & A_{pq}\\
A_{qp} & A_{qq}
\end{pmatrix},
$$

where every \(A_{ij}\) is \(16\times16\). Therefore \(P\) is \(32\times32 = 2B\times2B\). With the present default `B=32`, the local pivot is \(64\times64\).

And yes: **I now understand more precisely what you mean, and the memory argument is sound.**

### What you propose locally

Suppose we want to eliminate \(A_{pq}\). Load only

$$
P =
\begin{pmatrix}
A_{pp} & A_{pq}\\
A_{qp} & A_{qq}
\end{pmatrix}
$$

into local memory.

Then do as many Jacobi sweeps as necessary **entirely locally**, accumulating the total transformation

$$
U = U_1U_2U_3\cdots
$$

also locally.

Only after convergence do:

$$
A_{:,pq}\leftarrow A_{:,pq}U,
$$

$$
A_{pq,:}\leftarrow U^TA_{pq,:},
$$

and

$$
V_{:,pq}\leftarrow V_{:,pq}U.
$$

So the large adjacent strips are read/written **once**, irrespective of whether you needed 1, 5, or 20 internal sweeps.

That part is absolutely sensible. In fact the present `block_jacobi_1wg` already has exactly this mechanism: `P` and `U` stay local while `bj_pivot_solve()` does inner sweeps, and only afterwards is `U` propagated into global `A,V`.

The current code stopped using many inner sweeps because, in ordinary cyclic block Jacobi, later block pairs spoil what you just converged. **Your proposed change is specifically to eliminate that problem by making the split permanent.**

That is the genuinely new part.

---

## The crucial mathematical issue

Consider three groups of variables:

$$
A=
\begin{pmatrix}
A_{PP} & A_{PQ} & A_{PR}\\
A_{QP} & A_{QQ} & A_{QR}\\
A_{RP} & A_{RQ} & A_{RR}
\end{pmatrix}.
$$

We locally find \(U_{PQ}\) such that

$$
A'_{PQ}=0.
$$

Wonderful.

But suppose later we rotate \(P\) together with \(R\). Since generally

$$
A_{RQ}\neq0,
$$

the \(P-R\) rotation mixes some of \(A_{RQ}\) into \(A_{PQ}\), and therefore

$$
A''_{PQ}\neq0.
$$

That is exactly why ordinary Jacobi has to keep returning to old blocks.

So your rule

> once a block is zeroed, later operations must never touch it

is mathematically the right requirement.

But it has a stronger implication:

> **we may freeze a division only when the two sides are invariant subspaces of the full matrix.**

---

# What true divide-and-conquer would look like

At some level we have

$$
A=
\begin{pmatrix}
A_L&C\\
C^T&A_R
\end{pmatrix}.
$$

We seek an orthogonal transformation \(Q\) such that

$$
Q^TAQ =
\begin{pmatrix}
\tilde A_L&0\\
0&\tilde A_R
\end{pmatrix}.
$$

Once this has happened, we're done with that cross-block **forever**.

Then every subsequent transformation has the form

$$
Q_{\rm later}=
\begin{pmatrix}
Q_L&0\\
0&Q_R
\end{pmatrix},
$$

and therefore

$$
Q_{\rm later}^T
\begin{pmatrix}
\tilde A_L&0\\
0&\tilde A_R
\end{pmatrix}
Q_{\rm later}
=
\begin{pmatrix}
Q_L^T\tilde A_LQ_L&0\\
0&Q_R^T\tilde A_RQ_R
\end{pmatrix}.
$$

The zero is preserved **exactly**.

Then recurse:

```text
                       N
                      / \
                   N/2   N/2
                  / \     / \
                N/4 N/4 N/4 N/4
                ...
```

That indeed gives the memory behaviour you're after:

* spend expensive iterations on a working subproblem while it is local/cache-friendly;
* write the accumulated transformation out once;
* never revisit the eliminated coupling;
* subproblems become progressively smaller;
* eventually entire descendants fit into shared memory.

This is a very attractive architecture for a GPU.

---

## But here is the major catch

**Zeroing one small \(B\times B\) tile is not enough to establish such a split.**

If I diagonalize

$$
\begin{pmatrix}
A_{pp}&A_{pq}\\
A_{qp}&A_{qq}
\end{pmatrix}
$$

for two `B=8` blocks, I can make \(A_{pq}=0\).

But `p` and `q` can still both be strongly coupled to all the other blocks.

They are therefore **not independent eigensystems**.

Freezing that boundary would in general prevent convergence to the exact eigenvectors.

This is, I think, the single most important thing to get right before coding anything.

### Small-pivot Jacobi

```text
      p      q      r      s
p   [xxx]  [ 0 ]  [xxx]  [xxx]
q   [ 0 ]  [xxx]  [xxx]  [xxx]
r   [xxx]  [xxx]  [xxx]  [xxx]
s   [xxx]  [xxx]  [xxx]  [xxx]
```

does **not** allow us to solve `p` and `q` separately.

For true division we need something like

```text
       LEFT             RIGHT
    +---------+       +---------+
 L  |  XXXXX  |   0   |         |
    |  XXXXX  |       |         |
    +---------+-------+---------+
 R  |    0    |       | XXXXX   |
    |         |       | XXXXX   |
    +---------+-------+---------+
```

i.e. the **whole cross-block** must vanish.

---

# And yes, this is a known class of algorithms

The closest established term is **spectral divide and conquer**.

The literature describes almost exactly the high-level principle:

> recursively compute an invariant subspace, use it to decouple the matrix into two smaller eigenproblems, and recurse.

Nakatsukasa & Higham explicitly developed spectral divide-and-conquer methods partly because **data movement/communication can dominate arithmetic**. Their algorithms are designed to achieve communication lower bounds, which is remarkably close to the motivation you are giving here. ([SIAM][1])

Earlier Bai & Demmel work computes the necessary invariant subspaces using the **matrix sign function**. ([SIAM][2])

So your broad idea is definitely not crazy or novel in the bad sense. It lands in a serious numerical-linear-algebra research direction.

However, standard spectral D&C doesn't normally construct the split by locally Jacobi-zeroing little spatial tiles. It uses operations such as spectral projectors, matrix sign/polar decomposition, QR, subspace iteration, etc.

Interestingly, the modern methods specifically trade **more floating-point work for less communication**, which sounds potentially appropriate for your GPU problem. ([SIAM][1])

There are also tile-oriented dense symmetric eigensolvers motivated by exactly the memory-access bottleneck; Haidar, Ltaief and Dongarra developed a tile divide-and-conquer approach, although that route first reduces the dense matrix to banded form rather than doing your Jacobi construction directly. ([SIAM][3])

---

# I think there are therefore two separate ideas here

### 1. Local transformation accumulation — definitely good

Your rule:

> keep only the active small matrix in shared memory, iterate there, accumulate \(U\), then apply \(U\) once to the long global-memory strips.

I completely agree with this.

For a local `2B` pivot, the expensive global traffic becomes roughly

$$
O(NB)
$$

numbers read/write once, while you can do arbitrarily many \(O(B^3)\) local arithmetic operations without touching global memory.

On modern GPUs this is exactly the trade one generally wants.

And, importantly, **we already have the machinery for this in `gpu_block_jacobi.cl`.**

---

### 2. Permanent recursive deflation — promising but mathematically harder

This is where we need a different algorithm.

The problem becomes:

> How can we cheaply find a transformation that makes an entire macroscopic cross-block zero, not merely one small tile?

That is basically an **invariant-subspace computation**.

And here your intuition about bandwidth is still important: perhaps we can devise a Jacobi-like invariant-subspace solver which keeps transformations compact/local for several iterations and only occasionally streams the big matrix.

But we must prove/test that it actually creates an invariant split.

---

# One possible Jacobi-like formulation is interesting

Suppose we decide in advance that the first \(m\) dimensions will become one eigenspace and the remaining \(n-m\) the other.

Instead of trying to diagonalize everything, minimize only

$$
F=\|A_{LR}\|_F^2.
$$

Perform **only rotations between \(L\) and \(R\)** chosen to reduce this norm.

When

$$
\|A_{LR}\|\rightarrow0,
$$

we have obtained the desired block diagonalization.

Then stop all cross-boundary rotations forever and recurse independently inside `L` and `R`.

Conceptually:

```text
spectral_split(A):
    while ||A_LR|| > tolerance:
        choose rotations mixing L <-> R
        accumulate Q
    apply/finish Q

    spectral_split(A_LL)
    spectral_split(A_RR)
```

This is much closer to what you are imagining than conventional full cyclic Jacobi.

I believe this direction is worth investigating.

But there are two serious questions.

### Which eigenvalues go left/right?

If spectra overlap, the split is not defined by the original spatial block indices. The transformation may have to mix the two halves almost completely.

A robust spectral D&C normally chooses a spectral threshold \(\sigma\):

$$
\lambda < \sigma \quad|\quad \lambda > \sigma
$$

and computes the corresponding invariant subspaces.

That gives a mathematically meaningful division.

### Near degeneracies

Suppose two eigenvalues straddle the chosen division but are almost degenerate.

Then an extremely small off-block coupling can still cause large eigenvector mixing:

$$
\theta \sim
\frac{|H_{ij}|}{|\lambda_i-\lambda_j|}.
$$

So the criterion cannot simply be

$$
\|A_{LR}\| < 10^{-6}.
$$

It has to be related to the relevant spectral gap.

For DFTB, where near-degenerate molecular orbitals are common, this could matter substantially.

---

# There is also a very nice GPU consequence

If spectral splitting works, then the local-memory problem becomes progressively easier.

For example `N=256`:

```text
level 0       256
               |
          find 128/128 split

level 1    128     128
            |       |
          64 64   64 64

level 2    64-sized matrices
           ↓
        each now fits much more comfortably
        in local/shared memory
```

At the upper levels, transformations have to stream large matrices.

But **there are very few upper-level operations**.

Further down, there are many problems, but each is small and can stay completely resident.

That is precisely the cache/shared-memory behaviour we'd like.

And total global-memory traffic has a geometric character:

$$
N^2
+2(N/2)^2
+4(N/4)^2+\cdots
$$

which gives

$$
N^2\left(1+\frac12+\frac14+\cdots\right)
<2N^2
$$

**per one-pass operation at each recursive level**.

That is dramatically different from repeated full-matrix Jacobi sweeps costing roughly

$$
N^2 \times N_{\rm sweeps}.
$$

Of course the expensive question is how many passes are actually needed to obtain each invariant split.

But **this is exactly the right performance question to investigate.**

---

## So my current assessment

Your memory principle is very good:

> **local memory should hold the object that is reused many times.**
>
> That is the active pivot/subproblem and its accumulated transformation \(U\), not the long adjacent strips that are touched only once.

I think our current `B=32`, one-inner-sweep optimum was partly forced by the *cyclic Jacobi algorithm*: spending more time locally was pointless because future pairs destroyed the work. It doesn't prove that local convergence is bad. It proves that **local convergence + cyclic re-coupling is bad**.

Your proposed D&C changes that premise.

The main conceptual obstacle is therefore **not GPU implementation**. It is finding a cheap method that turns the chosen split into a genuine invariant-subspace split.

I would **not modify the production solver yet**. The right next step is a tiny CPU/Python experiment on, say, random symmetric matrices of \(N=32\) or \(64\): implement a recursive two-way Jacobi/block-diagonalization that minimizes only the cross-block norm, freeze the split, recurse, and check whether we recover `numpy.linalg.eigh` to machine precision. If that works robustly, *then* we can design the GPU memory layout. The literature on spectral D&C gives us a very useful reference for what the mathematical split has to accomplish. ([SIAM][1])

[1]: https://epubs.siam.org/doi/10.1137/120876605?utm_source=chatgpt.com "Stable and Efficient Spectral Divide and Conquer Algorithms for the Symmetric Eigenvalue Decomposition and the SVD | SIAM Journal on Scientific Computing"
[2]: https://epubs.siam.org/doi/10.1137/S0895479896297719?utm_source=chatgpt.com "Using the Matrix Sign Function to Compute Invariant Subspaces | SIAM Journal on Matrix Analysis and Applications"
[3]: https://epubs.siam.org/doi/10.1137/110823699?utm_source=chatgpt.com "Toward a High Performance Tile Divide and Conquer Algorithm for the Dense Symmetric Eigenvalue Problem | SIAM Journal on Scientific Computing"




---

# USER

ad "
Perform **only rotations between** **`L`** **and** **`R`** chosen to reduce this norm.
When

```math
\|A_{LR}\|\rightarrow0,
```

we have obtained the desired block diagonalization.
Then stop all cross-boundary rotations forever and recurse independently inside `L` and `R`.
"

that is what I'm saying or not? how it differd from may idea.

anyway can you scket kernels for this ?

---


# Chat GPT 5.6 sol

Yes. **That is essentially your idea.** My wording made it sound more different than it is.

The only distinction I wanted to enforce was:

* during the iteration we annihilate many small \(B\times B\) pieces of the cross-coupling;
* we may **not freeze one such small tile individually**;
* once the **whole cross-block**

  $$
  A_{LR}
  $$

  between two child subspaces is sufficiently zero, we freeze that entire split and never mix \(L\) and \(R\) again.

That gives exactly your divide-and-conquer scheme.

And I think there is a quite clean GPU architecture for it.

---

## 1. The local problem

Say `B=8`.

For one pair of 8-dimensional blocks \(p\in L\), \(q\in R\), gather

$$
P=
\begin{pmatrix}
H & G\\
G^T&M
\end{pmatrix},
$$

where

$$
H,M,G\in\mathbb R^{8\times8}.
$$

So `P` is only `16×16`.

This explains my previous `2B×2B`: **the thing we want to zero is \(G\), which is \(B\times B\), but to determine the transformation we also need its two diagonal neighbors \(H,M\).**

For `B=8`:

* `P`: `16×16` = 256 floats = 1 KB
* `U`: `16×16` = 1 KB
* total roughly 2 KB shared memory.

That is tiny.

Then locally do several sweeps, but **only rotations connecting the upper and lower halves**:

```text
H | G
--+--
G'| M
```

We do not need to diagonalize `H` or `M`. We only want

```text
H' | 0
---+---
0  | M'
```

and accumulate all those rotations into `U`.

This is very close to what Saad calls a block/subspace Jacobi rotation: annihilate an entire off-diagonal block while representing the transformation by a small dense matrix. His formulation uses a Riccati correction equation; the important connection for us is exactly that the target is the whole block \(G\), not necessarily diagonalization of the compound pivot. 

---

# 2. Kernel A: solve one local \(B\times B\) cross-block

Something conceptually like:

```c
#define B   8
#define P   (2*B)
#define PLD (P+1)

kernel void make_cross_U(
    global const float* A,
    global float* Ulog,
    global const Pair* pairs,
    int n
){
    int ipair = get_group_id(0);
    int lid   = get_local_id(0);

    Pair pair = pairs[ipair];
    int p0 = pair.p0;      // block in LEFT
    int q0 = pair.q0;      // block in RIGHT

    local float X[P*PLD];
    local float U[P*PLD];

    // gather:
    //
    //     [ A_pp  A_pq ]
    // X = [             ]
    //     [ A_qp  A_qq ]
    //
    gather_2block_pivot(A, X, p0, q0, n);

    set_identity(U);

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int sweep=0; sweep<INNER_MAX; sweep++){

        // ONLY rotations crossing the B|B boundary:
        //
        // i = 0 .. B-1
        // j = B .. 2B-1

        for(int round=0; round<B; round++){

            int i = lid;
            if(i < B){
                int j = B + ((i + round) % B);

                // compute Jacobi rotation which annihilates X[i,j]
                rotation_from_2x2(
                    X[i*PLD+i],
                    X[i*PLD+j],
                    X[j*PLD+j],
                    &c[i], &s[i]
                );
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Apply all disjoint rotations of this round:
            //
            // X <- J^T X J
            // U <- U J
            //
            apply_local_round(X, U, c, s, round);

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float off = norm_cross_block(X);

        if(off < INNER_TOL)
            break;
    }

    // We need only U later.
    store_U(Ulog, ipair, U);
}
```

### This is exactly where multiple inner sweeps are cheap

If it takes

```text
1 sweep
3 sweeps
10 sweeps
```

it still hasn't reread the large matrix once.

Everything is happening in roughly **2 KB local memory**.

That is precisely the reuse local memory is meant for.

---

# 3. Do many independent \(p,q\) pairs simultaneously

At the top split:

```text
LEFT                 RIGHT

0 1 2 3 4 5 6 7  |  8 9 10 11 12 13 14 15
```

we can pair them in rounds:

```text
round 0:
0-8
1-9
2-10
...
7-15

round 1:
0-9
1-10
...
7-8

round 2:
0-10
...
```

Every block appears **once per round**.

Therefore all transformations in one round are independent and together form

$$
Q_{\rm round} =
\operatorname{diag}(U_0,U_1,\ldots).
$$

That is very useful.

We first compute all `U`s:

```text
make_cross_U
     ↓
U0 U1 U2 U3 ...
```

without modifying `A`.

Then apply the entire round simultaneously.

---

# 4. Kernel B: apply one whole round to A

This is the important global-memory kernel.

Mathematically:

$$
A' = Q_{\rm round}^{T} A Q_{\rm round}.
$$

Rather than updating one pair and streaming the matrix, then another pair and streaming it again, **each matrix tile should be read once for the whole round**.

For paired groups \(a,b\):

$$
A'_{ab}
=
U_a^T A_{ab} U_b .
$$

Kernel skeleton:

```c
kernel void apply_round_A(
    global const float* Ain,
    global float* Aout,
    global const float* Uround,
    global const Pair* pairs,
    int n
){
    int ia = get_group_id(0);
    int ib = get_group_id(1);

    // each "pair" contains 2B indices
    Pair pa = pairs[ia];
    Pair pb = pairs[ib];

    local float T[P*P];
    local float W[P*P];

    local float Ua[P*P];
    local float Ub[P*P];

    load_U(Ua, Uround, ia);
    load_U(Ub, Uround, ib);

    // gather a 2B × 2B tile from the four physical B×B blocks
    //
    // rows = {pa.p, pa.q}
    // cols = {pb.p, pb.q}
    //
    gather_pair_tile(Ain, T, pa, pb, n);

    barrier(CLK_LOCAL_MEM_FENCE);

    // W = T * Ub
    matmul_small(T, Ub, W);

    barrier(CLK_LOCAL_MEM_FENCE);

    // T = Ua^T * W
    matmul_transpose_left(Ua, W, T);

    barrier(CLK_LOCAL_MEM_FENCE);

    scatter_pair_tile(Aout, T, pa, pb, n);
}
```

This is now basically a tiny tiled GEMM.

### Local-memory usage with `B=8`

`P=16`:

```text
T       1 KB
W       1 KB
Ua      1 KB
Ub      1 KB
----------------
~4 KB / WG
```

Excellent occupancy.

Even `B=16` gives:

```text
P=32
4 matrices × 32×32×4 = 16 KB
```

which is still reasonable on 48 KB hardware.

This is why I now think **B=8 or B=16 may become preferable to the present B=32**. In the existing solver B=32 wins because we repeatedly stream strips after every pivot. This new organization changes that tradeoff.

---

# 5. Very important: don't update V during the iterations

Here I think there is another substantial saving.

To decide the Jacobi rotations we need only `A`.

`V` does **not affect convergence**.

Currently a Jacobi step effectively does

$$
A\leftarrow U^TAU
$$

and simultaneously

$$
V\leftarrow VU.
$$

There is no mathematical reason to keep updating `V`.

Instead:

```text
A:
    transformed repeatedly

U:
    transformations appended into compact log

V:
    untouched
```

Store

```text
round 0 : U0 U1 U2 ...
round 1 : U0 U1 U2 ...
round 2 : ...
...
```

Then only when the eigensolve is finished:

$$
V_{\rm final}
=
V_{\rm initial}
U_0 U_1 U_2\cdots .
$$

That's exact.

This is especially attractive because **one row of V is tiny**:

for \(N=256\),

$$
256\times4 = 1024\ {\rm bytes}.
$$

So we can load many complete rows of `V` into local memory and replay the entire transformation history there.

---

# 6. Kernel C: replay the transformations onto V once

For example keep 32 complete rows:

$$
32\times256\times4 = 32\,{\rm KB}.
$$

That fits in 48 KB shared memory.

Something like:

```c
#define NROW 32

kernel void replay_U_log_on_V(
    global float* V,
    global const float* Ulog,
    global const Op* oplog,
    int n,
    int nops
){
    int row0 = get_group_id(0) * NROW;

    local float rows[NROW * NMAX];
    local float U[P*P];

    // ------------------------------------------------
    // ONE global load of this V tile
    // ------------------------------------------------
    load_rows(V, rows, row0, n);

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int op=0; op<nops; op++){

        Op o = oplog[op];

        load_U(U, Ulog, op);

        barrier(CLK_LOCAL_MEM_FENCE);

        // only 2B elements of every row are affected
        //
        // x[{p,q}] <- x[{p,q}] U
        //
        for(each local row r){
            float x[P];

            gather_vector_pair(rows, x, r, o.p0, o.q0);

            small_vecmat(x, U);

            scatter_vector_pair(rows, x, r, o.p0, o.q0);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ------------------------------------------------
    // ONE global write of this V tile
    // ------------------------------------------------
    store_rows(V, rows, row0, n);
}
```

This one I particularly like.

Instead of:

```text
read V
write V

read V
write V

read V
write V

... every Jacobi round ...
```

we get:

```text
load V once
perform potentially hundreds of transforms locally
store V once
```

`Ulog` is much smaller and heavily reused/cacheable.

This extends the **deferred-V** idea already present in your current resident solver from “one sweep” essentially to **the entire eigensolve**.

---

# 7. Then the divide-and-conquer itself

For node

```text
[ base, size ]
```

choose midpoint:

```text
L = [base, base+size/2)
R = [base+size/2, base+size)
```

Then:

```text
while ||A_LR|| > tolerance:

    for round = 0 .. max(nblock_L,nblock_R)-1:

        make_cross_U(...)
        apply_round_A(...)

        append U's to transformation log

freeze A_LR = 0

solve(L)
solve(R)
```

Once

$$
\|A_{LR}\|
$$

is converged:

```text
               N
        ┌──────┴──────┐
       N/2           N/2
```

**no subsequent kernel may have a pair with one block in the left child and one in the right child.**

Then the zero stays zero exactly, except numerical residue.

Next:

```text
               N
        ┌──────┴──────┐
       N/2           N/2
      /   \          /   \
    N/4   N/4      N/4   N/4
```

and so on.

At sufficiently small leaf size, for example `16` or `32`, simply load the whole leaf into local memory and fully diagonalize it.

---

# 8. Kernel D: cross-block convergence

Very cheap relative to the transformation:

```c
kernel void cross_norm(
    global const float* A,
    global float* norms,
    global const Node* nodes,
    int n
){
    Node nd = nodes[get_group_id(0)];

    float s = 0.0f;

    for(int i = nd.left0 + lid; i < nd.left1; i += lsize)
        for(int j = nd.right0; j < nd.right1; j++){
            float x = A[i*n+j];
            s += x*x;
        }

    s = workgroup_sum(s);

    if(lid == 0)
        norms[node] = sqrt(s);
}
```

Run this only after a **complete cross sweep**, not after every local pair.

---

# 9. Breadth-first GPU execution is probably better than literal recursion

On CPU I'd write:

```text
solve(node)
    solve(left)
    solve(right)
```

but on GPU I wouldn't.

Use levels:

```text
level 0:
1 node × batch

level 1:
2 nodes × batch

level 2:
4 nodes × batch

level 3:
8 nodes × batch
```

So at each level there are lots of independent node problems.

For your many-system SCC setup this gives:

```text
batch × nodes_per_level × pairs_per_node
```

workgroups.

Very natural GPU parallelism.

And eventually a node may already fit fully in local memory, so recursion terminates early.

---

## One caveat I would test before coding too much

The local solver I sketched:

> repeatedly use Jacobi rotations **only across the two B-dimensional halves** until local \(G\to0\)

is the most natural implementation of your idea, and small numerical tests suggest such cross-only Jacobi sweeps can indeed drive the cross coupling down.

But I would **first implement this exact algorithm in a 100-line Python reference**, because this is where subtle convergence/affiliation issues can occur, especially near degeneracies.

Saad's 2023 block-Jacobi/subspace-rotation work is extremely relevant here: it formulates annihilation of the whole \(G\) block and derives a Riccati equation

$$
G-HY+YM-YG^TY=0
$$

whose solution defines the subspace rotation. 

So there are actually two interchangeable implementations for `make_cross_U()`:

```text
A. cross-only local Jacobi
   simplest, very GPU friendly

B. solve the small B×B Riccati problem
   mathematically more directly targeted at annihilating G
```

For `B=8`, either is dirt cheap.

### The architecture I would prototype first

```text
B = 8

make_cross_U
    16×16 P + 16×16 U in local
    multiple inner sweeps
        ↓
    emit one 16×16 U

apply_round_A
    fused Uleft^T * tile * Uright
    ~4 KB local/WG

repeat rounds until whole A_LR ≈ 0

freeze split
recurse

DO NOT TOUCH V

after complete diagonalization:

replay_U_log_on_V
    32 full V rows resident in local
    replay every U
    one global load + one global store of V
```

That is, I think, a much cleaner realization of what you were originally proposing than the current `block_jacobi_1wg`. The **two strongest bandwidth ideas are independent but complementary**: freeze converged cross-blocks so `A` progressively shrinks, and defer **all** eigenvector updates so `V` is streamed essentially once.




---

# USER

ad "32×256×4=32KB." That fits in 48 KB shared memory.

do we really need such large blocks?

the thing is we want to have more than one workgroup per compute unit! 
I want to have many workgroups per compute unit.
I want to spare the memory, thats why I wanted rather 8x8 blocks or something like that. Not store whole recangular  block, but only the square core, 16x16, 32x32 and then apply the U matrix (accumulated after all sweeps) to the adjcent recangular blocks. 



---


# Chat GPT 5.6 sol


Yes — I agree. My `32×256` local-V suggestion was exactly the wrong direction for the architecture you want. It sacrifices occupancy to save bandwidth, whereas your design goal is **tiny local state + many resident workgroups**.

The local memory should contain only the repeatedly reused **square core** and its accumulated transformation:

$$
P=
\begin{pmatrix}
A_{pp}&A_{pq}\\
A_{qp}&A_{qq}
\end{pmatrix},
\qquad U.
$$

For `B=8`, this is a `16×16` core. With padding:

* `P`: \(16\times17\times4 \approx 1.1\) kB
* `U`: another \(\approx1.1\) kB
* rotations/reduction scratch: perhaps \(<1\) kB

So only roughly **2–3 kB shared memory per workgroup**.

On a 48 kB SM, shared memory then allows many workgroups resident; actual occupancy will probably be limited by registers, warps, or NVIDIA's maximum resident-block count rather than shared memory.

For comparison:

|  B | local core | P+U approximately | 48 kB shared-memory limit |
| -: | ---------: | ----------------: | ------------------------: |
|  8 |      16×16 |           ~2.2 kB |     ~20 WGs theoretically |
| 16 |      32×32 |           ~8.4 kB |                    ~5 WGs |
| 32 |      64×64 |            ~33 kB |                      1 WG |

So **B=8 is very attractive for exactly the reason you're emphasizing.**

## The adjacent rectangles should not be stored

Yes. After the local iterations converge, suppose the affected index set is

$$
I = p\cup q,\qquad |I|=2B.
$$

We have accumulated one final `U[2B][2B]`.

Then for every external row \(r\), all we need is

$$
x=A[r,I]
$$

which is just a vector of length `2B`.

Compute

$$
x' = xU
$$

and write it back.

For `B=8`, each worker only needs **16 floats**, i.e. 64 bytes. Those should preferably live in **registers**, not shared memory.

Conceptually:

```c
// U is only 16x16, shared by the WG
__local float U[16][17];

// each work-item handles one or several external rows
for (int r = ...; r < n; r += ...) {

    float x[16];                 // registers

    // global -> registers
    for (int j=0; j<16; j++)
        x[j] = A[r*n + index[j]];

    float y[16];

    for (int j=0; j<16; j++) {
        float s = 0;
        for (int k=0; k<16; k++)
            s += x[k] * U[k][j];
        y[j] = s;
    }

    // registers -> global
    for (int j=0; j<16; j++)
        A[r*n + index[j]] = y[j];
}
```

That is the architecture.

No `N×16` rectangle in local memory.

No `32×N`.

No whole matrix.

Just:

```text
local:
    16x16 core
    16x16 U

registers/thread:
    16-value input vector
    maybe 16-value output vector

global:
    long strips streamed exactly once
```

This is much more sensible.

---

## And there are really two phases

### Phase A — expensive iterative work, but entirely local

Load once:

```text
         p       q
      +-------+-------+
 p    | App   | Apq   |
      +-------+-------+
 q    | Aqp   | Aqq   |
      +-------+-------+
```

Then:

```text
sweep
sweep
sweep
sweep
...
```

while accumulating

$$
U=J_1J_2J_3\cdots.
$$

No global access during those sweeps.

This is where we exploit the shared memory.

### Phase B — cheap one-time propagation

Once finished:

```text
              p,q
               |
               v

xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx       x <- x U
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
```

stream adjacent vectors once.

This operation has little temporal reuse, so storing the rectangle in shared memory would accomplish almost nothing.

Exactly your point.

---

# I would actually separate the kernels

I wouldn't make one giant WG do both tasks.

Have:

### Kernel 1 — `solve_pivot_U`

One WG per block pair:

```text
global -> local P
U = I

many local Jacobi sweeps

write U -> global small U buffer
```

Very small local footprint.

Potentially hundreds/thousands of simultaneous pivot WGs across systems/nodes.

Then:

### Kernel 2 — `apply_U_rows`

Many WGs apply the transformations.

Something like:

```text
dimensions:

(system/node/pair, row_chunk)
```

Each WG reads one `U`, then processes e.g. 32 or 64 matrix rows.

Its local state is still only `U`.

For `B=8`, each work-item can own one row.

```c
__local float U[16*17];

int row = row0 + get_local_id(0);

float x0,...,x15;    // registers
```

Therefore this kernel can also have many resident WGs.

---

# We probably don't even need P and U simultaneously after solving

During the actual local solve we need both because `P` changes and `U` accumulates.

So about 2.2 kB for `B=8`.

But during propagation we need **only U**:

$$
\sim1.1\text{ kB/WG}.
$$

That is nearly negligible.

---

## Applying it to both A and V

For `A`:

$$
A[:,I]\leftarrow A[:,I]U.
$$

Because `A` is symmetric, we need the corresponding left transform

$$
A[I,:]\leftarrow U^TA[I,:].
$$

But if the matrix is stored redundantly symmetric, the cleanest scheme may simply be:

1. transform each external vector `A[r,I]`,
2. write both
   `A[r,I]`
   and its symmetric mirror `A[I,r]`.

Thus only one computation.

For the internal `I×I` core, use the already converged local `P`.

So:

```text
pivot rows/cols:
    result comes directly from local P

external rows:
    load 2B elements
    x' = x U
    write x'
    optionally mirror it
```

Very little state.

For `V` it's even simpler:

$$
V[:,I]\leftarrow V[:,I]U.
$$

Again every thread needs only a `2B` vector.

So I retract the idea that we should keep a big rectangular chunk of `V` resident. **We shouldn't.**

---

# The real bandwidth saving comes from a different place

Your important optimization isn't

> cache the adjacent rectangle.

It is:

> **don't touch the adjacent rectangle during the repeated local sweeps at all.**

For example, suppose annihilating the core needs 10 sweeps.

A naïve implementation could effectively do:

```text
sweep 1: read adjacent strips
sweep 2: read adjacent strips
...
sweep 10: read adjacent strips
```

Your scheme does:

```text
load 16x16 P

10 sweeps locally

one final U

read adjacent strips ONCE
write adjacent strips ONCE
```

That is potentially a huge improvement.

And then D&C adds the second saving:

```text
after L/R split converges:
    NEVER TOUCH A_LR AGAIN
```

so later recursive work operates only on smaller matrices.

Those two savings are the heart of the algorithm.

---

## I would start with B=8

Not because `8` is necessarily globally optimal, but because it gives us exactly the design properties we want:

$$
B=8,\qquad 2B=16.
$$

The inner `16×16` problem is tiny.

A 16×16 FMA transform is easy to unroll.

`U` is ~1 kB.

`P+U` ~2 kB.

A worker's 16-element vector fits naturally in registers.

And a workgroup could perhaps be just 32 or 64 threads rather than 256–512.

Then we can experimentally compare:

```text
B = 4   -> 8x8 core
B = 8   -> 16x16 core
B = 12  -> 24x24 core
B = 16  -> 32x32 core
```

The tradeoff becomes sensible:

$$
\text{smaller B}
\Rightarrow
\text{more block-pair operations, but vastly higher occupancy}
$$

versus

$$
\text{larger B}
\Rightarrow
\text{fewer operations, but shared-memory/register pressure}.
$$

The current solver found `B=32` because it has a completely different bandwidth structure. **I would not assume that optimum carries over at all to this D&C design.**

For this architecture, `B=8` feels like the right first implementation target.





---


# USER

but running kernels is costly (huge latency) that is why I want to have loop inside one kernel. How many times we would need to run this kernel to solve matix 86x86 or 256x256 which are the actuall sizes in our examples... ?

---

# Chat GPT 5.6


Exactly. If launch latency is important, then **the whole eigensolve should be one kernel launch for the whole batch**, not one launch per Jacobi round.

For example, with 400 replicas:

```text
launch persistent_dc_jacobi
    WG 0   -> matrix 0
    WG 1   -> matrix 1
    ...
    WG 399 -> matrix 399
```

Each workgroup then does all the loops, recursion, local pivot solves, and global strip updates itself. Since one matrix is owned by one WG, we never need synchronization between workgroups for that matrix.

That is also why your tiny `B=8` local state makes sense: we already have abundant parallelism from the **number of matrices**, so we do not need multiple WGs collaborating on one matrix.

## With B=8, how much looping is inside the kernel?

Let the recursion stop when a node has size \(\le 2B=16\), because then the entire node fits in the local `16×16` core and can simply be diagonalized.

At a node of size \(N_n\), we split it approximately in half. There are

$$
m \approx \left\lceil \frac{N_n/2}{B}\right\rceil
$$

blocks on each side.

To touch every cross pair \(L_i,R_j\) once, a block-Jacobi cross sweep needs about \(m\) **rounds**, because each round contains \(m\) disjoint pairs.

For your actual sizes:

| matrix |  B | recursion                | cross-rounds for one sweep of every split |
| ------ | -: | ------------------------ | ----------------------------------------: |
| 86     |  8 | 86 → 43 → 21/22 → ~11    |                                    **20** |
| 256    |  8 | 256 → 128 → 64 → 32 → 16 |                                    **64** |

For `N=256`, it is particularly transparent:

```text
node size 256:   16 rounds
2 × 128:        2 × 8 = 16 rounds
4 × 64:         4 × 4 = 16 rounds
8 × 32:         8 × 2 = 16 rounds
--------------------------------
total                     64
```

Then there are 16 leaf matrices of `16×16`, which can be solved completely locally.

But these **64 rounds are not 64 kernel launches**. They are simply iterations of a loop inside the one workgroup:

```c
__kernel void dc_eigen(...)
{
    int sys = get_group_id(0);

    // level 0
    solve_split(0, n);

    // level 1
    solve_split(left);
    solve_split(right);

    ...

    solve_leaves();
}
```

So:

**N=86: 1 kernel launch.**
**N=256: 1 kernel launch.**

The distinction is important.

---

### Multiple cross sweeps

The 20/64 numbers assume that each split requires one complete sweep over all \(L_i,R_j\) block pairs.

In reality we might need, say, 2–5 cross sweeps before

$$
\|A_{LR}\|
$$

is sufficiently small.

Then approximately:

|   N |    2 sweeps | 3 sweeps | 5 sweeps |
| --: | ----------: | -------: | -------: |
|  86 |  ~40 rounds |      ~60 |     ~100 |
| 256 | ~128 rounds |     ~192 |     ~320 |

Again: **still one kernel launch**.

And inside each block pair, perhaps we do 2–10 Jacobi sweeps on the `16×16` local core. Those are even cheaper in terms of memory because they never touch global `A`.

So there are three nested timescales:

```text
ONE KERNEL LAUNCH

    recursive level/node

        cross sweep

            block-pair round

                local 16×16 pivot
                    inner Jacobi sweep
                    inner Jacobi sweep
                    inner Jacobi sweep
                    ...
                obtain accumulated U

            apply all U of this round to global matrix

        test ||A_LR||

    recurse
```

Only the bold global application step rereads substantial parts of `A`.

---

## We should therefore definitely not use multiple WGs per matrix initially

Because then we have a nasty problem.

OpenCL provides barriers **within a workgroup**, but generally not a global barrier across independent workgroups inside an ordinary kernel.

So if WG1 computes one `U`, WG2 another, and then all of them must see all transformations before proceeding, we would either need:

* another kernel launch, or
* complicated persistent cooperative scheduling/global atomics/spin barriers.

Both defeat the simplicity.

Instead:

```text
1 matrix = 1 WG
400 matrices = 400 WGs
```

is almost ideal for your workload.

With `B=8`, each WG uses perhaps only ~2–3 KB local memory for:

```text
P : 16×17
U : 16×17
small rotation/reduction scratch
```

so **local memory no longer prevents many WGs from residing on an SM**.

That's exactly what we want.

---

## There is one subtle point about a "round"

Suppose at the top of `N=256`, we have 16 blocks on the left and 16 on the right.

One cross round consists of:

```text
L0-R0
L1-R1
L2-R2
...
L15-R15
```

These 16 transforms act on disjoint dimensions.

We don't want:

```text
compute U0
apply U0 to whole 256 matrix

compute U1
apply U1 to whole 256 matrix

...
```

because that would reread the matrix 16 times.

Instead the single WG should do:

```text
for pair = 0..15:
    gather 16×16 P
    locally converge P
    store U[pair] into small GLOBAL scratch
```

The scratch here is cheap:

$$
16 \times 16\times16\times4
=16\,{\rm KB}
$$

for the entire top-level round of an `N=256` system.

Crucially, **that's global scratch, not local memory**.

Then:

```text
apply_round():
    stream active matrix once
    for each tile/vector:
        determine its appropriate U
        transform
```

Then reuse/overwrite the `Uround` scratch for the next round.

So local memory remains ~2 KB even though the complete round contains 16 transformations.

This matches your original principle very nicely:

```text
LOCAL memory:
    only current 16×16 core + U

GLOBAL tiny scratch:
    accumulated U for each pair of CURRENT round

GLOBAL A:
    streamed once after all U of the round are known
```

---

## B=8 versus B=16 is an interesting real tradeoff

If we terminate at `2B`, the number of round passes drops quite dramatically with larger `B`.

For one recursive cross sweep:

|     B |   local P+U | N=86 rounds | N=256 rounds |
| ----: | ----------: | ----------: | -----------: |
|     4 |     ~0.6 KB |          51 |          160 |
| **8** | **~2.2 KB** |      **20** |       **64** |
|    16 |     ~8.4 KB |           7 |           24 |
|    32 |      ~33 KB |           2 |            8 |

This is actually the benchmark I would most want.

`B=32` minimizes global passes but kills occupancy.

`B=4` gives fantastic occupancy but too many passes.

The real optimum could plausibly be **B=8 or B=16**.

I would not assume `B=8` automatically wins. `B=16` still allows roughly several WGs from the 48-KB shared-memory perspective, while reducing the recursive round count for `N=256` from 64 to 24.

So I think the first implementation should make `B` compile-time configurable and benchmark:

```text
B=8
B=16
```

before doing anything more sophisticated.

### Skeleton of the actual persistent kernel

I would aim for something structurally like:

```c
__kernel void dc_jacobi_batched(
    __global float* A,
    __global float* V,
    __global float* Uround,
    ...
){
    const int sys = get_group_id(0);

    __local float P[2*B][2*B+1];
    __local float U[2*B][2*B+1];

    for (int level=0; level<nlevels; level++) {

        int nnodes = 1 << level;

        for (int inode=0; inode<nnodes; inode++) {

            Node node = get_node(level,inode,n);

            while (true) {

                // complete cross sweep
                for (int round=0; round<nrounds(node); round++) {

                    // produce all U's for this round
                    for (int pair=0; pair<npairs(node,round); pair++) {

                        gather_core(A,P,node,round,pair);

                        set_identity(U);

                        // repeated work happens ONLY locally
                        for (int inner=0; inner<INNER_MAX; inner++) {
                            cross_jacobi_sweep(P,U);
                            if (local_cross_converged(P)) break;
                        }

                        store_U(Uround,pair,U);
                    }

                    barrier(CLK_GLOBAL_MEM_FENCE |
                            CLK_LOCAL_MEM_FENCE);

                    // ONE streaming application of the complete round
                    apply_round_to_A(A,Uround,node);

                    barrier(CLK_GLOBAL_MEM_FENCE |
                            CLK_LOCAL_MEM_FENCE);
                }

                if (cross_norm(A,node) < tol)
                    break;
            }
        }
    }

    diagonalize_leaves(...);

    // eigenvector handling, possibly deferred
}
```

One launch contains everything.

And because every matrix is private to one WG, an `N=256` matrix doing 300 internal rounds does not imply 300 host/GPU synchronizations. It is simply one relatively long-running workgroup. As soon as a WG finishes, the GPU scheduler can run another matrix WG from the batch.

**That seems much closer to the architecture you had in mind than the multi-kernel sketch I gave earlier.**





