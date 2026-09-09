https://chatgpt.com/share/6a9ed20d-1f80-83eb-9337-320bb201a5fb

---

# USER


we recently implemented multi system GPU accelrated DFTB in rust+OpenCL in this repo
https://github.com/ProkopHapala/dftbplus

Now we want to use to for screeing of Proton-Coupled hydrogen transport and also Kekule Couled Proton transform...

there is detailed manifest

plese srutinize ideas form the LLM and brains storm how to make it better. Our goal is to use this project as oportunity for polshg the method (make it more gneral, more faster, numerically stable and robust), one big limitations now s that current kernels are optimized for really small system (<64 atomic orbitals) which fit to group local memory. We should think how to make solver which is still fast and using local memory but does not have such harsh limitation 

There are notes on dependence to other repos to prepare geometries, but I do not want distract you too much rather focus on performance core on this repo, other agents are able to handle that input preparation and depenencies on other repos.

---

# Chat GPT 5.6 sol

Yes. After reading the manifest and the actual GPU kernels, I think there is a substantially better architecture than the current roadmap.

The biggest conceptual change I would make is this:

> **Do not treat “N>64” as “we now need a completely global-memory eigensolver.”**
> Treat it as a hierarchy of dense solvers that preserve local-memory computation as much as possible, and simultaneously question whether SCC needs eigenvectors at all.

The manifest correctly identifies eigensolution as the dominant current cost—roughly 50–65% of SCC iteration time—and explicitly identifies ~86–87 orbitals as the immediate blocking size.  The existing Rust code is in fact already architected around two possible directions—purification and block Jacobi—even though the production SCC path currently only uses the full-local Jacobi.

## 1. First: the 64-orbital limit is somewhat artificial

Your current Jacobi stores

$$
A_{N\times N}, \qquad V_{N\times N}
$$

simultaneously in local memory, both padded to `N×(N+1)`.

Memory is approximately

$$
M_{\rm local} \approx 2 N(N+1)4\ {\rm bytes}.
$$

So:

|   N | A+V local |
| --: | --------: |
|  48 |   18.4 kB |
|  64 |   32.5 kB |
|  80 |   50.6 kB |
|  88 |   61.2 kB |
|  96 |   72.8 kB |
| 112 |   99.0 kB |
| 128 |    129 kB |

That explains the cutoff much better than “Jacobi fundamentally stops at 64.”

The Rust specialization itself confirms that this is an implementation restriction: it generates a specialized `JN`, `JLD`, pair schedule and WG, then explicitly rejects `n>64`.

So I would **not hard-code 64 as a mathematical architecture boundary**.

Have something like:

```text
DenseSolverPolicy
    FullLocalJacobi
    SplitLocalJacobi
    BlockJacobi
    DensePurification
    HostReference
```

and select based on:

```text
N
local_mem_size
max_workgroup_size
batch
gap estimate
whether eigenvectors are actually requested
```

This immediately makes the implementation portable to GPUs with different local-memory limits.

---

# 2. There is a very promising intermediate solver between N=64 and full block Jacobi

For your immediate N≈84–90 systems, I would first investigate whether you can avoid storing **A and V simultaneously**.

That is the main source of the memory cliff.

## Variant: keep A local, V global

For N=88:

* A local ≈ 31 kB
* rotation parameters negligible
* V remains global

That suddenly fits comfortably into a 48 kB WG.

The expensive Jacobi transformation of **A** stays exactly where you want it: local memory, barriers within one WG, Brent–Luk schedule, essentially the current kernel.

Only eigenvector accumulation becomes global.

For each Brent–Luk round, all rotations are disjoint. Therefore updating

$$
V \leftarrow VJ
$$

is embarrassingly parallel by row.

Each row needs the two elements

$$
(V_{kp},V_{kq})
$$

for each pair `(p,q)`.

For one round, you're basically streaming V once:

$$
N^2\quad\text{reads+writes}.
$$

This is not great, but for N=88 the V matrix is only 31 kB. It is tiny compared with GPU cache.

### Why I think this deserves testing before multi-WG Jacobi

Your present local kernel pays an enormous synchronization/compute burden doing Jacobi rotations anyway.

The extra traffic per round for V may actually be cheaper than converting the entire algorithm into inter-workgroup tiled passes requiring kernel boundaries.

And it preserves the extremely useful property:

> **One molecular system = one workgroup.**

That is especially attractive because you have hundreds of independent systems.

So the GPU parallelism remains:

```text
GPU
 ├─ WG 0 : system 0 eigensolve
 ├─ WG 1 : system 1 eigensolve
 ├─ WG 2 : system 2 eigensolve
 ...
```

instead of spending multiple workgroups cooperatively solving one tiny 88×88 eigenproblem.

For your use case, **parallelism across systems is much more valuable than parallelism inside a single 88×88 matrix.**

I would benchmark this as the first N≈128 experiment.

---

# 3. Even better: process V in slabs

There is another intermediate architecture.

Keep:

```text
A[N×N]       local
Vslab[K×N]   local
```

and accumulate eigenvectors in slabs of rows.

Suppose N=88, K=16:

```text
A       ≈ 31 KB
V slab  ≈  5.5 KB
-----------------
          36.5 KB
```

Still fits 48 kB.

But you cannot finish A and each slab independently because the same sequence of rotations needs to be applied.

Two possibilities:

### A. Save the rotations

During Jacobi, instead of updating V, save:

```text
(c,s,p,q)
```

for every rotation.

After converging A, replay all rotations over V in tiles.

A sweep contains

$$
N(N-1)/2 \approx 3828
$$

rotations for N=88.

20 sweeps worst case ≈76k rotations.

Each rotation can be encoded compactly, perhaps `float2(c,s)` because pair identities are implied by Brent–Luk schedule.

≈600 kB per matrix worst case—not wonderful for batch=100.

But actual convergence may use substantially fewer sweeps.

### B. Apply rotations to V globally each round

Probably simpler and likely faster.

This is essentially my previous suggestion.

So I would start with **A-local / V-global**.

---

# 4. The manifest's proposed “multi-WG tiled Jacobi” is slightly too vague

The manifest currently describes “each WG handles a tile, reads row/column pairs, applies rotation, writes back.” 

The difficult part is not the local tile math.

It is the synchronization semantics:

$$
A_{k,p},A_{k,q}
$$

must see one coherent set of rotations before the next Brent–Luk round.

OpenCL gives you no global barrier between workgroups.

Therefore every Jacobi round essentially becomes one or more **kernel launches** unless you use device-side enqueue/cooperative mechanisms you probably don't want.

For N=88:

$$
87\ {\rm rounds/sweep}.
$$

Say 8 sweeps:

$$
696
$$

global synchronization phases per SCC eigensolve.

That can be disastrous if interpreted literally as 696 kernel launches.

So I would absolutely **not implement element-level multi-WG Brent–Luk that way**.

The natural scalable algorithm is instead **block Jacobi**, which your matrix layer already anticipates. The repository explicitly describes partitioning into T×T blocks, diagonalizing 2T×2T compound blocks locally, and applying rotations with tiled matrix operations.

That reduces global synchronization from roughly `N` rounds per sweep to roughly

$$
N/T
$$

block rounds.

For example N=96, T=16:

```text
6 blocks
5 Brent-Luk rounds / block sweep
```

rather than 95 element rounds.

That's a completely different synchronization cost.

---

# 5. I would therefore make the dense eigensolver hierarchy

## Solver 0 — Full-local Brent–Luk

Current implementation.

Rough range:

```text
N <= device/local-memory-dependent threshold
```

Not necessarily 64.

Fastest path.

---

## Solver 1 — A-local / V-global Brent–Luk

Probably:

```text
64 < N < ~100–120
```

depending on GPU.

This could be extremely good for exactly AT/GC/azaindole.

You retain the existing highly optimized A kernel and avoid the huge architectural jump.

I would test this **before** block Jacobi.

---

## Solver 2 — Block Jacobi

Something like:

```text
N ~ 100–300
```

Use blocks T≈16/24/32.

For each block pair `(p,q)`:

1. load compound

$$
\begin{pmatrix}
A_{pp} & A_{pq}\\
A_{qp} & A_{qq}
\end{pmatrix}
$$

of size `2T × 2T` into local memory;

2. diagonalize it using your current small local Jacobi machinery;

3. obtain block rotation \(J_{pq}\);

4. update associated block rows:

$$
[A_{kp}\ A_{kq}] \leftarrow [A_{kp}\ A_{kq}]J
$$

5. accumulate V similarly.

Your existing matrix module was already written with exactly this architecture in mind, including `brent_luk_rounds` and local small-block eigensolvers.

So interestingly:

> **Much of “Phase 5” conceptually already exists in the repository.**

What appears missing is production integration and an efficient block-update pipeline, rather than inventing the algorithm.

---

# 6. But I think the most important missed opportunity is dense purification

Here I disagree with the manifest.

The manifest effectively says:

> sparse purification doesn't help because nucleobases are dense.

That's true.

But it accidentally conflates:

```text
purification == sparse purification
```

which is not true.

You should benchmark **dense TC2 purification**.

Your matrix code already explicitly contains this pathway.

For SCC, what do you actually require at every iteration?

Mostly:

$$
P
$$

the occupied density matrix.

You do **not fundamentally need individual eigenvectors**.

The SCC loop currently does:

```text
H
 ↓
diagonalize
 ↓
C, ε
 ↓
occupation
 ↓
P
 ↓
Mulliken charges
```

Purification does directly

```text
H
 ↓
P
```

using GEMMs.

On a GPU this is potentially a much more natural operation.

---

# 7. Forces do not kill the purification idea

This is important.

For analytic nonorthogonal DFTB forces you need both the density matrix \(P\) and the energy-weighted density matrix \(W\).

Normally,

$$
P = C f C^T
$$

and

$$
W = C f\epsilon C^T.
$$

But in the orthogonal representation,

$$
P' = f(H')
$$

and because \(P'\) and \(H'\) commute,

$$
W' = P'H'.
$$

Then transform back appropriately.

So after purification, the energy-weighted density can be obtained essentially by another dense matrix multiplication.

Therefore:

> **An eigendecomposition is not intrinsically required even for analytic forces.**

This makes dense purification considerably more attractive as a long-term DFTB engine.

---

# 8. Why purification may work unusually well here

Your systems are closed-shell organic molecules.

For AT, roughly half of the states are occupied.

They should normally have a nonzero HOMO-LUMO gap.

TC2/purification convergence depends strongly on that spectral gap, but for many nucleobase geometries it may be excellent.

And GPU GEMM is exactly what hardware likes.

Your current timings already say GEMM is only ~5% while Jacobi is 50–65%. 

That makes an experiment almost obligatory:

```text
current:
  transform
  Jacobi             55%
  density build
  ...

purification:
  transform
  GEMM
  GEMM
  GEMM
  ...
```

You may discover that 5–10 GEMMs beat the Jacobi kernel.

I would not assume this. Benchmark it.

But it may change the entire architecture.

---

# 9. And purification eliminates another ugly current operation

Your present SCC route diagonalizes `H'`, then:

1. reads eigeninformation to CPU,
2. sorts it,
3. constructs occupation mask,
4. uploads mask,
5. constructs density.

The current SCC source explicitly describes this host occupation path.

That is conceptually unnecessary for the normal fixed-electron-count closed-shell DFTB case.

Even if you retain diagonalization, at minimum move this onto GPU.

For ~87 orbitals you can:

* extract eigenvalues;
* sort using bitonic/radix inside one WG;
* derive Fermi occupation;
* build density.

No CPU intervention.

It probably isn't the biggest current runtime cost, but it makes the algorithm cleaner and eventually allows **the entire SCC loop to remain inside a device-side iteration strategy**.

---

# 10. The current SCC algorithm performs more dense algebra than necessary

Current loop is roughly:

$$
H' = X H X
$$

(two GEMMs)

then eigensolve

then

$$
C = XC'
$$

(one GEMM)

then construct density.

There are transformations you can rearrange depending on what quantity you actually need.

If purification operates in orthogonal basis:

$$
P'=f(H')
$$

then AO density becomes

$$
P = XP'X
$$

which costs two GEMMs.

So naive purification has more GEMMs.

But Mulliken charge evaluation can perhaps be done without explicitly constructing every intermediate in the most expensive representation.

Given

$$
P = X P' X
$$

and Mulliken population

$$
q_A = \sum_{\mu\in A}(PS)_{\mu\mu},
$$

with

$$
X=S^{-1/2},
$$

there may be algebraic simplifications because

$$
XSX=I.
$$

I would derive the exact contraction you actually require and see whether you can compute atomic populations directly from \(P'\) and some precomputed projector matrices.

For each atom define AO projector \(M_A\).

Then Mulliken population is a trace:

$$
q_A=\mathrm{Tr}(M_A P S).
$$

Transforming projectors into the orthogonal basis once,

$$
\tilde M_A = X^{-1} M_A X
$$

or an equivalent symmetric form, may allow:

$$
q_A = \mathrm{Tr}(\tilde M_A P')
$$

without forming full AO \(P\) each SCC iteration.

This deserves a careful derivation because it could eliminate another N³ operation.

---

# 11. Bigger optimization: S does not change during SCC

You already exploit this by constructing \(X=S^{-1/2}\) once per geometry.

Good.

But geometry relaxation introduces a new opportunity:

successive geometries differ only slightly.

Therefore \(S_{k+1}\) is close to \(S_k\).

Right now every geometry step apparently intends:

```text
assemble S
diagonalize S
construct S^-1/2
```

again.

That second eigensolve becomes important once you accelerate SCC Hamiltonian diagonalization.

I would investigate warm-started inverse square root.

For example Newton–Schulz refinement:

$$
X_{n+1} =
\frac12 X_n(3I-SX_n^2)
$$

after suitable scaling.

If previous geometry gives \(X_k \approx S_{k+1}^{-1/2}\), **one or two refinement iterations may suffice**.

That turns the overlap orthogonalization from another eigensolve into several GEMMs.

This is particularly natural on GPU.

It could be a major win in batched geometry optimization.

---

# 12. Similarly: warm-start the electronic subspace, not only charges

The manifest emphasizes SCC charge warm-start, correctly. It reports ~3–4 SCC iterations expected after geometry steps versus ~13–16 cold. 

But you can also warm-start the eigensolver.

Successive SCC Hamiltonians differ only by

$$
\Delta H_{\mu\nu}
=
\frac12S_{\mu\nu}
(\Delta V_\mu+\Delta V_\nu).
$$

That is a relatively structured, modest perturbation.

Current Jacobi starts with

$$
V=I
$$

every iteration.

That's wasteful.

Use previous eigenvectors \(C'_{k-1}\):

$$
A_0 = C_{k-1}'^T H'_k C'_{k-1}
$$

which should already be nearly diagonal.

Then Jacobi only cleans residual off-diagonal components.

This costs transformations but could drastically reduce sweeps.

Even better, think of maintaining the basis throughout SCC:

```text
H'_new
transform into previous eigenbasis
few Jacobi sweeps
update eigenvectors
```

If you go from, say, 8 sweeps to 1–2, this may matter more than almost any low-level kernel optimization.

I would put **eigenvector warm-start** very high on the research roadmap.

---

# 13. This is one reason I wouldn't blindly use fixed 20 Jacobi sweeps

The current Jacobi has convergence/stagnation logic, which is good, but the roadmap mostly discusses tweaking `tol` and `MAX_SWEEPS`.

More informative diagnostics would be:

```text
offdiag_norm(sweep)
offdiag_norm(round)
number of nontrivial rotations
rotation angle distribution
sweeps per SCC iteration
```

especially with warm starts.

Then determine whether time is spent:

* doing useful rotations;
* executing almost-identity rotations;
* recomputing convergence norms;
* barriers;
* local-memory traffic.

This can guide whether Brent–Luk itself is optimal.

---

# 14. I would also rethink the Jacobi convergence metric

A single relative global Frobenius criterion

$$
\frac{\|A-\mathrm{diag}(A)\|_F}{\|A_{\rm off}^{(0)}\|_F}<\epsilon
$$

is mathematically fine but not directly tied to what SCC needs.

SCC does not need every eigenvector to machine accuracy.

What matters is primarily accuracy of the **occupied subspace** and therefore density.

Near a degeneracy *within occupied states* rotations are physically irrelevant to density.

Likewise degeneracy within unoccupied states is irrelevant.

Only occupied/unoccupied mixing matters.

That's profound.

You are spending Jacobi effort diagonalizing:

```text
occupied ↔ occupied
unoccupied ↔ unoccupied
```

although these rotations do not change

$$
P=C_{\rm occ}C_{\rm occ}^T.
$$

So a density/subspace method can avoid enormous useless work.

This is another argument for purification.

But even with eigensolvers, you could converge based on block residual separating occupied and virtual spaces rather than fully diagonalizing the matrix.

---

# 15. A very interesting alternative: occupied-subspace iteration

Because ~45 of ~87 orbitals are occupied, partial diagonalization only saves ~2×, not an order of magnitude.

Still, instead of full eigensystem:

1. retain occupied basis \(Q\);
2. iterate subspace:

   $$
   Y = g(H')Q
   $$

   where \(g\) filters low-energy states;
3. orthogonalize Q;
4. Rayleigh–Ritz only a ~45×45 matrix.

For changing SCC Hamiltonians and changing geometries, the previous occupied subspace is an excellent initial guess.

This could be extremely fast.

But I would rank it after dense TC2 because TC2 is simpler and your repository already contains most of the machinery.

---

# 16. SCC robustness: I would not do the proposed one-parameter-at-a-time sweep

The manifest suggests sweeping alpha, DIIS history, SCC tolerance, Jacobi tolerance independently. 

That is useful for debugging, but it's not the best route to a robust production solver.

SCC convergence is a nonlinear fixed-point problem:

$$
q_{n+1}=F(q_n).
$$

The important issue is conditioning of the charge-response Jacobian.

A fixed alpha=0.05 is only hiding this.

For the proton transfer surfaces, especially zwitterionic configurations, you can expect charge sloshing.

I would implement a robust hierarchy:

```text
1. charge warm-start
2. DIIS
3. trust-region / bounded DIIS
4. adaptive damping when DIIS worsens residual
5. level shift when HOMO/LUMO occupation becomes unstable
6. fallback Anderson/Broyden
```

And most importantly:

### Do not require residual monotonicity

The manifest's fail-loud invariant says:

> SCC residual monotonically decreasing.

That is actually wrong for DIIS.

DIIS commonly produces temporary increases while converging much faster overall.

So replace this invariant with things such as:

```text
finite residual
bounded charge
no sustained divergence
best residual improves over a moving window
```

The existing manifest currently lists monotonic residual decrease as an invariant.  I would remove that.

Same for:

> force norm decreasing during relaxation

FIRE does not guarantee monotonic force norms either.

That should also be diagnostic, not a correctness assertion.

---

# 17. For relaxed screening, finite-difference forces are probably the wrong first production architecture

I agree with using them as a **validation bridge**.

I disagree somewhat with making them Phase 1 of a serious scan.

For N_atom=30 you need 181 SCC solves per force.

Even if heavily batched, that multiplies the dominant eigensolver work by ~181.

The manifest's own rough estimate is ~9 seconds per force evaluation for 100 systems. 

A geometry optimization might need tens/hundreds of force evaluations.

That is going to overwhelm everything else.

So I would use finite differences for:

```text
GPU force parity tests
small number of geometries
debugging analytic derivatives
```

not as the route to the 441-point relaxed PES.

I'd actually move analytic GPU forces **before** the large relaxed scan.

---

# 18. Force GPU kernel architecture should exploit pair locality

Do not port the CPU force routine too literally.

For pair \((A,B)\), the derivative blocks are tiny:

* H: one AO block, maximum 4×4 in mio for s/p
* S: same
* pairwise repulsion
* gamma derivative

Compute all contributions **per atom pair**, locally.

Something like:

```text
WG = one molecular system

threads iterate atom pairs AB
    load pair geometry
    evaluate SK radial functions + derivatives
    construct small dH/dr, dS/dr blocks
    contract directly with P and W
    accumulate F_AB
```

For only ~30 atoms you might even use local force buffers:

```text
float3 F[32]
```

and avoid atomics by assigning each thread/warp an atom or using pair reduction.

Because systems are tiny, a **single-WG whole-system force kernel** is plausible and fits your preferred architecture beautifully.

---

# 19. Don't store dH/dR and dS/dR

This is important.

The manifest wording could lead an agent toward generating derivative matrices then doing matrix contractions.

Don't.

Forces need something like

$$
F_A
=
-\sum_{\mu\nu}
P_{\mu\nu}
\frac{\partial H_{\mu\nu}}{\partial R_A}
+
\sum_{\mu\nu}
W_{\mu\nu}
\frac{\partial S_{\mu\nu}}{\partial R_A}
+\cdots
$$

Evaluate SK derivatives and immediately contract them with P/W.

Thus:

```text
SK pair
 → derivative block
 → contract
 → discard
```

No giant derivative tensor.

That keeps global traffic low and fits the whole philosophy of the project.

---

# 20. Geometry optimization should remain mostly on GPU

The manifest proposes host FIRE steps and geometry upload every outer iteration.

For 30 atoms, the PCIe bytes are tiny, so this is not necessarily a performance catastrophe.

But it introduces synchronization and prevents future fused workflows.

FIRE itself is trivial:

```text
v += dt*f/m
P = dot(v,f)
update dt, alpha
v = ...
x += dt*v
```

For a batch of 500 systems this is a perfect GPU kernel.

Likewise constraints.

I'd keep:

```text
positions
velocities
forces
charges
active flags
```

resident.

Then the host should orchestrate relaxation at a coarse level rather than downloading coordinates every step.

---

# 21. Active masks vs compaction: use both, but at different occupancy

I would use a threshold strategy.

Let

$$
f_{\rm active}=N_{\rm active}/N_{\rm batch}.
$$

Then:

```text
f_active > ~0.6:
    keep mask

~0.2 < f_active < ~0.6:
    compact occasionally

f_active < ~0.2:
    compact immediately or merge with another pending batch
```

Compaction every iteration is unnecessary.

A prefix scan over 1000 systems is cheap, but rebuilding all batched arrays may not be.

Even simpler on CPU:

* read only active flags;
* create new index list;
* future kernels operate through indirection `system_ids[]`.

You don't need physically to rearrange all matrices.

---

# 22. Organize batches by orbital dimension buckets

Current SCC requires homogeneous `N`, `Na`, `Nocc`.

That's reasonable for maximum performance.

For general screening, don't try to remove this restriction completely.

Bucket:

```text
N 1–32
N 33–48
N 49–64
N 65–80
N 81–96
N 97–128
...
```

pad within bucket.

Same kernel specialization.

For DFTB s/p basis the orbital count varies discretely anyway.

This gives you generality without sacrificing fixed compile-time dimensions.

---

# 23. Think about SoA across systems for small matrix operations

Current matrix layout is

```text
[system][i][j]
```

which is ideal for one-WG-per-system local loading.

For larger solvers where a warp processes the same `(i,j)` across multiple systems, the better layout can become

```text
[i][j][system]
```

because adjacent lanes access adjacent systems.

I would **not globally change the layout now**.

But consider keeping two layouts conceptually:

```text
system-major     for local-Jacobi / whole-system WG
batch-major SoA  for elementwise SCC operations
```

Conversion would be expensive, so only do it if profiling demonstrates value.

For your current one-WG-per-system architecture, system-major is probably still right.

---

# 24. Another important issue: `S^{-1/2}` currently has host traffic

The current `build_inv_sqrt` implementation copies S back to host and reuploads it simply to make a working copy. That's explicitly marked as something to optimize.

That should be fixed immediately.

Use device-to-device buffer copy.

Especially during geometry relaxation.

It's a small engineering fix with zero algorithmic risk.

---

# 25. Avoid `rt.finish()` after every primitive operation

This deserves checking systematically.

If each wrapper:

```rust
kernel.enq();
rt.finish();
```

then your nominally device-resident SCC is actually globally synchronizing the command queue after every tiny kernel.

For:

```text
delta_q
gamma
H update
GEMM
GEMM
Jacobi
...
```

that can be surprisingly expensive.

For correctness you only need dependencies, which an in-order OpenCL queue already supplies.

You should normally:

```text
enqueue
enqueue
enqueue
...
finish/read only when host actually needs data
```

For profiling, use events.

This may be one of the easiest current wins.

I would audit every GPU wrapper specifically for hidden `finish()` calls.

---

# 26. Kernel construction is another easy cleanup

The SCC source itself notes kernel handles are rebuilt and that this is a performance TODO.

For this project I would create a persistent:

```rust
GpuPlan {
    dimensions,
    kernels,
    buffers,
    scratch,
    specialization,
}
```

for each bucket/template.

Then running another SCC/geometry becomes almost entirely:

```text
update data
enqueue persistent kernels
```

This is particularly useful for screening thousands of related configurations.

---

# 27. Numerical precision: don't make it simply “f32 vs f64”

I think the manifest frames this too coarsely.

Use **mixed precision by operation**.

Keep matrices and SK operations f32.

But accumulate sensitive scalars in f64 where available—or compensated f32 if not:

* energy sums;
* charge residual norms;
* DIIS dot products;
* Frobenius norms;
* trace;
* force reductions;
* small overlap eigenvalues.

The expensive N³ work remains f32.

This gives most of the numerical benefit at almost no cost.

For example:

$$
\|R\|^2=\sum_i r_i^2
$$

over even 100 entries does not need to suffer avoidable cancellation/noise.

Likewise energy differences on PES scans are much smaller than total electronic energy.

---

# 28. For screening, energy differences are more important than total energy parity

The proposed parity criterion

$$
|E_{\rm GPU}-E_{\rm CPU}|<10^{-4}\ {\rm Ha}
$$

is useful but insufficient.

1e-4 Ha ≈ 2.7 meV.

Fine.

But more relevant is preservation of:

$$
\Delta E(x)-\Delta E(x_0).
$$

Systematic offsets cancel.

I would add tests:

```text
max absolute total-energy error
RMS relative PES error
barrier error
force error
charge error
```

Especially because f32 may shift total energy slightly while the PES remains excellent.

---

# 29. Proton-transfer scans create an ideal continuation problem

Don't solve the 21×21 points independently.

The physical manifold is smooth.

For each point `(i,j)`, initialize from a converged neighbor.

For example snake ordering:

```text
row 0: left → right
row 1: right → left
row 2: left → right
```

Warm-start:

```text
charges
geometry
possibly eigenvectors/subspace
FIRE velocities perhaps reset
```

Even better:

seed each new point from whichever converged neighbor has closest `(t1,t2)`.

That can reduce both SCC and geometry iterations dramatically.

The manifest mentions strip propagation indirectly but I would make continuation a **first-class algorithmic feature**, not just an SCC rescue technique.

---

# 30. For a relaxed PES, don't necessarily optimize all 441 points from scratch

There's another screening-level strategy.

Run:

```text
11×11 coarse relaxed surface
```

then adaptively refine where:

* energy curvature is high;
* competing valleys exist;
* saddle regions appear;
* asynchronous intermediate exists.

For mechanism discovery, adaptive mesh can beat 21×21 brute force.

For a publication-quality regular grid you can always fill in later.

This doesn't alter your core solver, but dramatically improves scientific throughput.

---

# 31. Revised roadmap I would use

I would reorder the manifest substantially.

### Phase A — measurement + remove accidental overhead

Before new algorithms:

```text
A1. remove S host roundtrip
A2. remove unnecessary queue.finish()
A3. persistent kernel objects
A4. GPU eigenvalue extraction + occupation
A5. event-based kernel profiling
A6. record Jacobi sweep statistics
```

This establishes a trustworthy baseline.

---

### Phase B — N≈90 solver experiments

Implement **three competitors**, not one:

```text
B1. A-local / V-global Brent-Luk
B2. block Jacobi T=16/24/32
B3. dense TC2 purification
```

Benchmark on:

```text
N = 56
64
72
84
88
96
128
```

and batch:

```text
1
10
100
500
```

This will tell you what architecture actually wins.

This is much more scientifically valuable than immediately implementing one “recommended” N>64 path.

---

### Phase C — warm-start electronic structure

```text
C1. previous charges
C2. previous eigenspace / occupied subspace
C3. warm-start S^-1/2 across geometry steps
```

This may produce bigger speedups than kernel tuning.

---

### Phase D — analytic GPU forces

Do this before serious relaxed scans.

And implement pairwise direct contraction, not derivative tensor construction.

---

### Phase E — fully batched GPU relaxation

FIRE on device.

Masked/compacted systems.

Continuation across PES.

---

### Phase F — robust SCC

Implement adaptive DIIS/trust-region/level shift based on the troublesome scan points.

Not parameter sweeps alone.

---

# 32. My expected solver regime

If I had to bet before benchmarking:

```text
N <= ~64:
    current full-local Jacobi wins

N ~65–100:
    A-local / V-global Jacobi may win

N ~100–200:
    dense TC2 or block Jacobi

N ~200–500:
    dense purification likely increasingly attractive
    if molecular gap remains healthy

large sparse systems:
    sparse BSR purification
```

Notice that this gives you **continuous evolution** rather than one harsh boundary.

---

# 33. Most interesting research direction: eliminate eigensolve from SCC entirely

If this project is meant not merely to run proton scans but to polish your DFTB implementation into a genuinely general high-throughput engine, this is the part I would pursue most seriously.

The abstraction should eventually be:

```rust
trait ElectronicSolver {
    solve_density(H, S, state) -> {
        P,
        W,
        charges,
        energy,
        diagnostics
    }
}
```

rather than:

```rust
eigensolve(H,S) -> eigenvectors
```

Then backends can be:

```text
FullLocalJacobi
BlockJacobi
TC2
SparseTC2
SubspaceIteration
CPU/LAPACK reference
```

SCC itself should not care how \(P\) was produced.

That architecture also prepares you beautifully for your later CDFT/Kekulé/proton-electron work.

---

## The three changes I would prioritize immediately

If you give a coding agent only three things to investigate next, I would make them:

1. **Prototype A-local / V-global Jacobi for N=84–96.**
   Minimal departure from the very fast existing kernel and directly attacks the artificial local-memory limit.

2. **Benchmark dense TC2 against Jacobi on exactly your current formic/azaindole/AT Hamiltonians.**
   Don't dismiss purification because the *sparse* implementation isn't suitable. This could remove the dominant eigensolver bottleneck entirely.

3. **Audit synchronization and persistent GPU state.**
   Eliminate host copy in `build_inv_sqrt`, unnecessary `finish()` calls, kernel reconstruction, CPU occupation sorting, and then re-profile before investing heavily in more complicated algorithms.

The manifest is strong as a chemistry/project plan, but at the computational-core level I think it currently commits too early to **“multi-WG element Jacobi is the production N>64 solution.”** The existing code already contains the beginnings of a much more interesting dense solver framework, and the proton-transfer project is a good opportunity to turn that into a genuinely adaptive GPU electronic-structure backend rather than merely pushing the same Jacobi kernel from 64 to 128 orbitals. 





---

# USER

I read what you wore and generally this is great feedback exacly in direction I wanted, I agree with most but I would like to steer it a bit:



* I agree that we should not jump from local N=64 jacobi solver to something fully global-memory based, and that multi-WG jacobi may be overkill. But I would not like to make 5-6 different solvers, as we dont want to spend on it much more time and selecting the solver would be then complicated. Lets keep N=64 as special case for small system and then do something which is much more universal. I do not like the ideas of saving just A and not V as it increase the size of system just 2x, which is still to specialized. I much more like tiled jacobi but non necesarily multi WG. For simplicity it make sense to me to use still 1WG per system, but make internal loop in the kernel solving it by tiles alsways loading just the necessary tile to local-memory workspace and then saving it back to global. We should have algorthm which does not require to have the whole matrix in local shared memory, nd which still amortize the global mmory read/write cost. Consider also we are not limited to 64 threads per workgroup, we can easily have 256 or even 1024 threads per WG, but we cannot fit 1024x1024 matrix to local memory. So if we are able to partition the algorothm so we can use mre threads operating in parallel on smaller chunks of the matrix that would be ideal and that is exactly what I want from you. Figure out locla-memory-efficient algorithm within one workgorup. This apporach will work especially well when all system are same size so all WGs would have same amout of work and finish in sync.

* I totaly agree with you that synchonization over multiple kernel lunches is absolutely not acceptable. Completely agree with this
"696 global synchronization phases per SCC eigensolve.
That can be disastrous if interpreted literally as 696 kernel launches.
So I would absolutely not implement element-level multi-WG Brent–Luk that way."
... if the coding agent proposed this in the coding plan it is total bullshit, we must tell him clarly he is idiot! No way this crap!!!! that is exacly what we strive to avoid!! perfect anti-patern of our GPU optimization guidelines!!!
 
* About "dense-purification" - I'm not sire I do not like this idea too much. I never saw purification beat eigensolver for small systems, I'm very skeptical. I think it would make sense if we can save memory (like sparse matrix-vector multipication save a lot of memory so we can fit it in local shared memory). But this dense matrix multipication does not seem to save any memory so I do not see much advantage from tiled Jacobi. Yes, GEMM is much simpler algorithm than Jacobi, so it may be easilier to tile optimally, but still I think we can solver the system properly by Jacobi exacly while matrix purification is iterative method with serious compromises. At some point we probably want eigenvectros (although not necessarily every SCC iteration). All-in-all if we do not see clearly significatly more memory efficient purification than Jacobi I would prefer implement Jacobi. 

* Forces must be analysitcal not fineite differences
* yes we should not mateizalize dH/dR and dS/dR as huge matrixes, but contract them on the fly
* yes, everything should reside on GPU minimal turnaround to CPU and Bback 1
* There are many negleted lazzyliy suboptimal oprations which the coding agent did not follwed my rules about avoiding CPU/GPU copies, and finish, sync, ore even buffer/kernel allocation inside hot loop. I streaed many times this should be eliminated first, mimnimize CPU overhead, so we really are bund by GPU performance not by out rust harness.
Please write it clearly in bulletpoints [ ]/[*] checkboxes where exaclty are these problem and what should be changed, but be concise
* yes checking parity on endery differnce rather than apbsolute energyies is great idea! elaborate on it an tell coding agent what exacly to do.
* ad using one calculation as estimate for another - we can maybe use it for relaxation steps, but it break paralizaion so we cannot use it for different points on the scan beause it would be sequential then
* it may be good idea to do partially sequential scan like if we do 2D 400 configurations, we can use 40 workgroups, each with 64 threads each doing 10 confs with similar geometry alsways reusing the estimate from prevous, however, we have 10k threads on RTX3090 so we will not saturate it with 40 WGs 


---


# Chat GPT 5.6 sol

Yes — with your steering I would simplify the design substantially. I think we need only **two dense eigensolver paths**:

* existing **full-local Jacobi for N≤64**;
* one **general one-WG tiled/block Jacobi** for everything larger.

No multi-WG synchronization, no host-driven Jacobi rounds, no A-local/V-global special intermediate solver, and no purification work unless later benchmarks force us to reconsider it.

The important point I noticed while checking the actual code is that **Jacobi is not the only current N≤64 limitation**: `matmul_full_local_batched()` also explicitly rejects `N>64`. So the manifest statement that tiled Jacobi leaves everything else unchanged is incorrect.

## 1. General N>64 solver: one-WG tiled **block Jacobi**

I think the clean architecture is a cyclic **block Jacobi** with global A/V as backing storage and only a `2B×2B` pivot plus strip workspace in local memory.

Take initially:

```text
B  = 32 orbitals
2B = 64
WG = 256 or 512 threads   (benchmark these two)
```

This is particularly attractive because your already-debugged full-local Jacobi machinery knows how to diagonalize a 64×64 matrix.

For AT, N≈87 → pad to 96 → only **3 blocks of 32**.

So one block sweep has only:

$$
{3\choose2}=3
$$

pivot operations!

For general N:

```text
A[system][Npad][Npad]   global
V[system][Npad][Npad]   global

local:
    pivot[2B][2B+1]
    U[2B][2B+1]
    scratch[...]        // can alias pivot after pivot is stored
```

For B=32, just `pivot+U` costs approximately

$$
2(64)(65)(4)=33.3\ {\rm kB}.
$$

So this still fits a nominal 48 kB local-memory budget, while N itself can be 96, 200, 500, 1000... There is **no N² local-memory scaling anymore**.

### One pivot `(p,q)`

Load the principal compound block

$$
P=
\begin{pmatrix}
A_{pp}&A_{pq}\\
A_{qp}&A_{qq}
\end{pmatrix}
$$

into local memory, then use the existing local Brent–Luk solver to obtain

$$
U^T P U = D.
$$

Store the transformed principal block.

Now update the rest of A by **strips**. For an outside row tile `k`:

$$
X=[A_{kp}\;A_{kq}]
$$

is only `R × 2B`. Load it to local memory and calculate

$$
Y=XU.
$$

Then store

$$
[A'_{kp}\;A'_{kq}] = Y
$$

and simultaneously its transpose:

$$
\begin{bmatrix}
A'_{pk}\\
A'_{qk}
\end{bmatrix}=Y^T .
$$

This is important: because A is symmetric, **you do not need a separate global left-multiplication pass**. Updating the strip and writing its transpose implements both sides of

$$
A' = J^T A J.
$$

That should considerably reduce global-memory traffic.

Then do the same tiled right multiplication for eigenvectors:

$$
[V_{:p}\;V_{:q}] \leftarrow [V_{:p}\;V_{:q}]U.
$$

`U` remains in local memory throughout all these strip updates.

Only then continue to the next block pair.

The whole thing is:

```text
kernel tiled_jacobi_batched(...)
{
    sid = get_group_id(0);       // one system = one WG

    initialize V = I;

    for block_sweep {

        for each block pair (p,q) {

            // all threads cooperate
            load 2B×2B pivot A[pq,pq] -> local;

            local_Jacobi(pivot, U);

            store transformed principal block;

            barrier(GLOBAL | LOCAL);

            for row_tile k {
                load [A(k,p),A(k,q)] -> local;
                Y = X*U;
                store Y;
                store transpose(Y);       // maintain symmetry
            }

            for row_tile k {
                load [V(k,p),V(k,q)] -> local;
                Y = X*U;
                store Y;
            }

            barrier(GLOBAL | LOCAL);
        }

        compute ||offdiag(A)||;
        if converged break;
    }
}
```

**One kernel launch for the entire eigensolve of the whole batch.** Every synchronization needed by the Jacobi algorithm is a workgroup barrier because no other WG touches this system.

That is exactly the GPU architecture I would aim for.

---

## 2. We don't actually need Brent–Luk rounds between blocks

Inside the 64×64 local pivot, yes: use the existing parallel Brent–Luk implementation.

But at the *block-pair* level, because one WG owns the system and we only keep one pivot/U in local memory, I would initially just use a simple deterministic cyclic order:

```text
(0,1), (0,2), ...,
(1,2), (1,3), ...
```

Trying to process several block pairs concurrently inside one WG adds complexity and races at their cross-block intersections.

We already have massive parallelism:

* hundreds of systems;
* 256–512 threads cooperating inside each system;
* parallel Brent–Luk inside the pivot;
* parallel strip GEMMs.

Keep the outer block algorithm simple.

---

## 3. 256 vs 512 vs 1024 threads

I agree completely that the workgroup need not stay tiny.

For `2B=64`, local Jacobi has 32 independent Jacobi pairs.

With:

```text
WG=256  →  8 threads / Jacobi pair
WG=512  → 16 threads / Jacobi pair
WG=1024 → 32 threads / Jacobi pair
```

And during strip updates there are thousands of independent output elements.

So **512 is particularly interesting** for RTX 3090.

I would benchmark only:

```text
256
512
```

initially. Don't create another combinatorial tuning project.

1024 can be tried afterward if 512 clearly leaves execution resources idle. A 1024-thread WG may increase barrier/register pressure, so bigger is not automatically better.

Tile size `B` and WG size are **parameters of one solver**, not different solvers.

---

# 4. Absolutely forbid host/multi-kernel Jacobi synchronization

I would put this directly into the task specification:

* [ ] **FORBIDDEN:** do not implement one Jacobi round/pivot/sweep as one OpenCL kernel launch. No host-driven inner Jacobi loop.
* [ ] **FORBIDDEN:** do not use multiple workgroups for one matrix if correctness requires synchronization between those WGs.
* [ ] The complete N>64 eigensolve must execute inside **one kernel launch**, with **one WG owning one system** from initialization through convergence.
* [ ] Global memory is only backing storage for A/V; local memory is a reusable pivot/strip cache.
* [ ] All synchronization inside the eigensolver must be WG-local `barrier()` with the appropriate global-memory fence.

The proposed ~hundreds of global synchronization/kernel-launch phases per eigensolve would be exactly the opposite of the architecture this project is trying to achieve.

---

# 5. Dense purification: drop it from the active plan

I agree with your objection.

Dense purification does **not** solve the important memory problem:

$$
A,\;D,\;D^2,\ldots = O(N^2)
$$

global storage anyway, while replacing an exact eigensolution with an iterative polynomial procedure involving repeated O(N³) GEMMs.

There can certainly be cases where it wins, but we have no evidence it wins here, and it does not address the fundamental design target nearly as cleanly as tiled block Jacobi.

The implementation in the repo is also currently very far from a clean production GPU route: TC2 reads traces back to the host during every purification iteration.

So:

* [*] Existing purification implementation can remain as experimental/reference code.
* [ ] **Do not spend development time optimizing dense purification now.**
* [ ] Implement the general tiled/block Jacobi first.
* [ ] Reconsider purification only if later measurements reveal a concrete advantage.

---

# 6. General N>64 GEMM is also required

This is an important correction to the manifest.

The production SCC currently calls:

```rust
matmul_full_local_batched(...)
```

three times per SCC iteration, and that implementation explicitly rejects `N>64` because it also puts full matrices in local memory.

So the N>64 work is actually:

```text
general tiled Jacobi
+
general tiled GEMM
```

Fortunately GEMM is much easier because output tiles are independent. **Multiple WGs per matrix are perfectly fine for GEMM**, since they require no inter-WG synchronization.

So unlike Jacobi, there is no reason to insist on one WG/system for GEMM.

The repo already has a conventional tiled `batched_gemm`; it needs to be integrated cleanly into the shared `GpuRuntime` production SCC path instead of relying on the full-local N≤64 GEMM.

---

# 7. GPU-harness cleanup — do this BEFORE performance work

Here is the concise checklist I would give the agent.

* [ ] **`gpu_eigen.rs::jacobi_cyclic_local_batched()`** constructs a new `Kernel` every eigensolver call and immediately calls `rt.finish()`. Remove the unconditional finish; cache/reuse kernel objects for a `(N/config)` plan.
* [ ] **`gpu_eigen.rs::build_inv_sqrt()`** downloads the complete S matrix to CPU and uploads it again merely to make a copy. Replace this with a device→device copy.
* [ ] **`build_inv_sqrt()`** also performs another unconditional `rt.finish()` after its kernel. Remove it; use queue dependencies/events.
* [ ] **`gpu_matrix.rs` SCC primitives** repeatedly call `Kernel::builder()` inside the SCC hot loop (`matmul`, `gamma`, `H_scc`, Mulliken, residual, density, trace, dot...). Programs are cached but **kernel objects are not**. Build a persistent `GpuSccPlan` containing kernels and scratch buffers.
* [ ] **`gpu_scc.rs` initialization** reads `q0/init_q` GPU→CPU and immediately creates/uploads another GPU buffer. Replace with device copy/reuse.
* [ ] **Every SCC iteration currently downloads the entire `batch×N×N` H′** just to inspect its diagonal, sorts eigenvalues on CPU, then uploads an occupation mask. Move diagonal extraction + Nocc selection/mask generation entirely onto GPU.
* [ ] **CPU DIIS currently transfers `q_new` and `q_cur` to CPU every SCC iteration and uploads mixed charges again.** Ultimately implement small per-system DIIS on GPU; history is tiny (`~8×30`). At minimum stop downloading `q_cur`, because the host already knows the vector it uploaded on the previous iteration.
* [ ] Scratch matrices/buffers are currently allocated every `gpu_solve...()` call. Geometry relaxation calls this repeatedly, so allocate them once in `GpuSccPlan` with `batch_capacity` and reuse them across relaxation steps.
* [ ] Do not use `queue.finish()` for timing. Use OpenCL profiling events; otherwise asynchronous kernel enqueue time is being measured rather than GPU execution.
* [ ] **`GpuRuntime::query_capabilities()` does not actually query the GPU**: it currently returns hard-coded RTX-3090-like values (48 kB, WG=1024, 82 CUs, 24 GB). Replace this with real OpenCL capability queries. This is especially important because tile/WG selection depends directly on local memory.
* [ ] After these fixes, rerun the existing timing benchmark. **Do not optimize Jacobi based on timings contaminated by CPU synchronization/harness overhead.**

I would make completion of this checklist **Phase 0**, before N>64 work.

---

# 8. Analytic GPU forces only

Agreed. I would remove finite-difference forces from the implementation roadmap.

The production force architecture should be:

$$
(P,W,\mathbf R)
\rightarrow
F
$$

with SK derivatives generated and immediately contracted.

For every atom pair \(A,B\):

```text
evaluate SK H/S + radial derivatives
rotate to orbital blocks
generate dH and dS small pair blocks
contract immediately with P and W
discard derivative block
accumulate pair force
```

Never construct something like

```text
dH[3*Natoms][N][N]
dS[3*Natoms][N][N]
```

at all.

Finite difference should only be an **independent debugging test** on perhaps 5–10 selected Cartesian components. You already have CPU analytic forces, so there is no reason to do 6N finite differences as the normal GPU route.

---

# 9. Energy parity: test the PES, not merely the arbitrary total-energy zero

I would retain an absolute-energy test because it catches catastrophic errors, but make **relative-energy parity the scientifically important criterion**.

For every reference set of geometries \(i\):

$$
\Delta E_i^{CPU}=E_i^{CPU}-E_{\rm ref}^{CPU}
$$

$$
\Delta E_i^{GPU}=E_i^{GPU}-E_{\rm ref}^{GPU}
$$

and define

$$
\epsilon_i=
\Delta E_i^{GPU}-\Delta E_i^{CPU}.
$$

Report/test:

* [ ] `absolute_offset = E_GPU(ref)-E_CPU(ref)` — diagnostic only unless huge.
* [ ] `RMS(ε_i)` over the complete scan.
* [ ] `max |ε_i|`.
* [ ] Barrier parity:

  $$
  \delta E_b=
  (E_{TS}-E_R)_{GPU}-(E_{TS}-E_R)_{CPU}.
  $$
* [ ] For 2D PES, also compare nearest-neighbour increments along both axes:

  $$
  E_{i+1,j}-E_{i,j},\qquad E_{i,j+1}-E_{i,j}.
  $$

  This is an excellent detector of geometry-dependent GPU numerical noise.
* [ ] Keep charge parity and force parity separately.
* [ ] For the eigensolver itself also test **residual and orthogonality**, not only eigenvalues:

  $$
  \frac{\|AV-V\Lambda\|_F}{\|A\|_F},
  \qquad
  \frac{\|V^TV-I\|_F}{N}.
  $$

For initial N>64 acceptance I'd use approximately `max relative-PES error < 1e-4 Ha`; after the implementation works, tighten this based on what the N≤64 solver actually achieves. Current validated formic scan is already considerably better than that. 

This distinguishes a harmless approximately constant f32 total-energy offset from an error that actually distorts the proton-transfer barrier.

---

# 10. Warm starts without destroying scan parallelism

I agree with your correction. A full snake traversal of a 21×21 PES would be stupid for GPU screening:

```text
point 0 → point 1 → point 2 → ... → point 440
```

It sacrifices the main advantage of batching.

Warm-starting should definitely be used **inside geometry relaxation of each point**:

```text
relax step k
   ↓ charges, geometry, perhaps electronic state
relax step k+1
```

There is no lost parallelism because all scan points advance simultaneously.

For cross-scan continuation, a compromise is short independent chains.

For 400–441 geometries, instead of 40 chains of length 10, I would prefer perhaps:

```text
~100–150 independent chains
length ≈ 3–4
```

For 441 points:

```text
L=3 → 147 independent chains
L=4 → 111 independent chains
```

That is much healthier for an RTX 3090 with 82 SMs.

And this is where **WG count matters more than raw thread count**. If you have only 40 WGs, at least 42 of 82 SMs cannot receive even one of them, irrespective of whether each WG contains 64, 256, or 1024 threads.

So a sensible rule is approximately:

$$
N_{\rm independent\ chains}\gtrsim N_{\rm SM}
$$

and preferably somewhat above it, subject to actual local-memory/register occupancy.

This is another reason `B=32 + WG=512` is interesting: the tiled solver may consume enough local memory that only ~one WG/system resides per SM anyway. In that regime, ~100 independent systems can already be enough to keep the GPU broadly occupied.

So I would **not bake cross-scan continuation into the solver**. Treat it as an optional scheduling optimization with short chains, and benchmark chain lengths 1/2/4 rather than assuming 10.

---

## Revised project ordering

I would now tell the coding agent to work in this order:

```text
Phase 0
    Remove GPU↔CPU roundtrips, finish(), hot-loop kernel/buffer creation.
    Implement truthful profiling.

Phase 1
    General tiled GEMM path for N>64.

Phase 2
    One-WG tiled/block Jacobi:
        B≈32
        256/512-thread benchmark
        2B×2B local pivot
        global A/V backing
        tiled strip updates
        one kernel launch/eigensolve.

Phase 3
    Integrate N>64 S^-1/2 and complete SCC.
    GPU occupation handling.
    Relative-PES parity tests.

Phase 4
    Analytic GPU forces with on-the-fly derivative contraction.

Phase 5
    Persistent GPU batched FIRE + GPU DIIS/active masks.

Phase 6
    Proton/Kekulé screening and optional short-chain scan continuation.
```

This keeps the project focused: **one optimized tiny-system solver and one genuinely general dense tiled solver**, while using this chemistry application to eliminate the accumulated CPU/GPU plumbing compromises rather than adding more layers on top of them.







---

# USER

we implemented most if this in this commit
https://github.com/ProkopHapala/dftbplus/commit/b269ab63eee65eb854bb21184793fed4d02aeebc
please review it focus on mathematical and physical rigot as well as numerical efficienency of the kernels and minimizing of the overhead, the coding agent often do stupid things, lazzy sub-optimal harness, and physical misconception, try to indetify them and say him how to do it properly. 

Focus just on the aspect related to your sub problem (the dense multi-small-system tested on H-bonds, not the sparse solver and vibrations hessian which is topic of the other threads)


---

# Chat GPT 5.6 sol

I went through the dense/H-bond part of commit `b269ab63...` in detail. The overall direction is good, and several subtle SK/force bugs were fixed correctly, but I would **not yet regard the N>64 SCC + force pipeline as production-quality**. There are a few serious mathematical errors hidden by weakened tests, and the host harness still violates the GPU-resident architecture in several places.

The good part first: the ordered heteronuclear SK handling, correction of the one-based SK grid origin, and the missing `1/dr` in the p–s derivative were exactly the kinds of subtle errors that matter. After those fixes the reported non-SCC electronic-force parity is excellent—roughly \(10^{-6}\)–\(10^{-5}\) relative for H2/H2O/formic dimer.  The partial-tail-block concept in tiled Jacobi is also correct: N=87 really remains N=87 globally and the final 23-orbital block is treated locally rather than making a singular 96-dimensional overlap problem.

But I would give the coding agent the following corrections, in roughly this priority order:

1. **[P0 — mathematical bug] Do not zero the residual off-diagonal of a locally diagonalized pivot.** The tiled Jacobi computes a finite sequence of rotations \(U\), but then stores the compound principal block as `diag(lA)` and explicitly throws away all residual off-diagonal elements.  Mathematically, after the local rotations you have

   $$
   P'=U^T P U=D+R .
   $$

   `V` is updated using \(U\), so the global transformed matrix must store **the actual \(P'\)**, including \(R\). Replacing it by \(D\) means the invariant

   $$
   A_k = V_k^T A_0 V_k
   $$

   is no longer true. This is not merely “setting tiny numbers to zero”; with a fixed 10 local sweeps you have no proof they are tiny. Store `lA[i,j]`, not `(i==j ? lA[i,i] : 0)`. Likewise, do not blindly zero the entire final global A unless convergence was actually reached. Return eigenvalues separately if you want a clean diagonal output.

2. **[P0 — massive performance regression] Replace the serial inner pivot Jacobi by the existing Brent–Luk parallel algorithm.** The new 64×64 pivot solver does
   `for p { for q { ... } }`, thread 0 computes one rotation, and then at most 64 threads update rows. With WG=256, three quarters of the WG are idle. Worse, every scalar rotation has multiple barriers. For PB=64 this is 2016 rotations/sweep × 10 local sweeps ≈ **20,160 scalar rotations per pivot**, hence of order 60,000 barriers/pivot.

   You already solved this problem better in `gpu_eigen.cl`: 32 independent pairs per Brent–Luk round × 8 threads/pair = exactly 256 threads, with only a few barriers per round.  The tiled solver should literally reuse/adapt that algorithm inside its local `2B×2B` pivot. For a 55-dimensional tail pivot, schedule only the active 56 lanes, not all 64 dummy lanes.

   This is probably the single largest kernel-level performance error in the commit.

3. **[P0 — OpenCL correctness] The global-memory fences are missing.** The kernel initializes global V, later reads it; writes global A/V during one pivot, then the next pivot reads those values; and computes the sweep residual from global A. Yet the barriers shown are `CLK_LOCAL_MEM_FENCE` only.   NVIDIA may make this work accidentally, but that is not the OpenCL memory contract.

   At the dependency boundaries use
   `barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)`.
   You don't need a global fence after every local strip operation if the next operation does not consume those global writes, but you **do** need it before the next pivot/convergence scan and after global V initialization.

4. **[P0 — tests are hiding solver errors] Restore the stated numerical contracts.** The tiled-Jacobi test header says residual and orthogonality \(<10^{-5}\), eigenvalues \(<10^{-4}\), but the actual assertions were relaxed to \(10^{-3}\) and \(5\times10^{-3}\).  \(5\times10^{-3}\) Ha is about 0.14 eV—completely inappropriate if this machinery is intended to resolve proton-transfer barriers to meV–10 meV accuracy.

   Even worse, the N>64 SCC test permits **0.01 Ha = 0.27 eV** error in energy, eigenvalues, and 0.01 e in charges.  That test proves only that the program does not explode.

   Do not loosen a contract because the implementation misses it. Fix the solver. For the target I would restore roughly:

   $$
   \frac{\|AV-V\Lambda\|_F}{\|A\|_F}\lesssim 10^{-5},
   \qquad
   \frac{\|V^TV-I\|_F}{N}\lesssim10^{-5},
   $$

   then test the actual generalized problem as well:

   $$
   \frac{\|HC-SC\epsilon\|_F}{\|H\|_F},\qquad
   \frac{\|C^TSC-I\|_F}{N}.
   $$

   For H-bond PES parity, target relative energy errors around \(10^{-4}\) Ha or better and measure **energy differences/barriers**, not merely total energies.

5. **[P0 — physical energy] The current “SCC energy” is not the physical total DFTB energy.** Both the CPU reference and GPU SCC energy are

   $$
   E_{\rm el}= {\rm Tr}(PH^0)+\frac12\Delta q^T\gamma\Delta q,
   $$

   with no pairwise DFTB repulsive energy.  This was okay for CPU/GPU electronic parity, but it is **not okay for the proton-transfer PES**. As H moves between O/N atoms, \(E_{\rm rep}(R_{AB})\) changes and is part of the DFTB total energy:

   $$
   E_{\rm DFTB}=E_{\rm el}+E_{\rm rep}.
   $$

   The relaxed H-bond scan must report total energy including repulsive splines. Otherwise you can get quantitatively, and potentially qualitatively, wrong barriers while CPU/GPU parity appears perfect.

6. **[P0 — force physics incomplete] `gpu_forces.cl` currently computes only the non-SCC electronic component.** The kernel evaluates essentially

   $$
   2\sum_{\mu\nu}
   \left(P_{\mu\nu}\,\partial H^0_{\mu\nu}
        -W_{\mu\nu}\,\partial S_{\mu\nu}\right),
   $$

   which is only one part of the SCC-DFTB force.

   Production forces must match the already-established CPU decomposition:

   * \(P\,dH^0-W\,dS\);
   * SCC shift term proportional to \(\tfrac12(V_A+V_B)P\,dS\);
   * \(\Delta q_A\Delta q_B\,\gamma'(R_{AB})\);
   * repulsive spline derivative.

   The CPU code now has analytic H/S derivatives wired into the production force route, which is a good improvement.  Port **that exact tested formula**, including signs and Bohr/Å factors. Do not independently “rederive something similar” in OpenCL.

   Also, after SCC convergence build

   $$
   P_{\mu\nu}=2\sum_{k\in occ}C_{\mu k}C_{\nu k},
   $$

   and

   $$
   W_{\mu\nu}=2\sum_{k\in occ}\epsilon_k C_{\mu k}C_{\nu k}
   $$

   together. W is needed only for the final force evaluation, not every SCC iteration.

7. **[P0 — SCC consistency] The electronic state used for energy/forces is not exactly the state of the returned mixed charges.** In `GpuSccPlan::scc_step`, H/C/D are built from the current \(q_n\), then `q_gpu` is overwritten with the mixed \(q_{n+1}\). If the residual passes tolerance, the caller stops. `compute_energy()` then uses D from \(q_n\), but recomputes \(\Delta q,V\) from \(q_{n+1}\).

   At tight tolerance this is small, but for **analytic forces** this is precisely the kind of inconsistency that destroys energy-gradient parity.

   Proper sequence:

   $$
   q_n\to H(q_n)\to P_n\to q_{\rm out}.
   $$

   If the residual is converged, either accept the state consistently as \(q_n\), or—preferably—accept the final mixed/fixed-point q and perform **one final unmixed electronic solve**. The C/P/W/E/forces then all correspond to exactly the same charge vector.

8. **[P0/P1 — the “persistent plan” is not persistent enough.]** `GpuSccPlan` claims to own “pre-built kernel handles”, but the struct contains no `Kernel` objects. `scc_step()` simply calls the existing wrappers, each of which still does `Kernel::builder()` every invocation.   So Phase 0 was documented but not really implemented.

   Build the actual kernels once in the plan. Set changing buffer arguments and enqueue them. Likewise, `set_geometry()` currently calls `build_inv_sqrt_batched()`, which allocates new full N² buffers and finally replaces `self.x_buf`.  Preallocate:
   `S_work`, `S_vec`, `Vscaled`, `lambda`, `X`
   and overwrite them in place.

   `GpuRuntime::copy_buffer()` is an improvement over the old CPU roundtrip, but it still **allocates a new buffer**.  Add `copy_buffer_into(src,dst)` for hot-loop use.

9. **[P0/P1 — CPU is still in every SCC iteration.]** The plan extracts the eigenvalue diagonal on GPU, immediately performs a blocking read, allocates a fresh `Vec<usize>` for each system, CPU-sorts, and uploads `occ_mask`. It then reads all RMS values back every iteration.  The DIIS production path additionally reads `q_new`, allocates several `Vec<f64>` per system per iteration, does CPU DIIS, and uploads q again.

   For N≈87, occupation selection is trivial on GPU. One WG/system can bitonic-sort 128 padded `(ε,index)` entries in local memory and emit the first `Nocc` indices. You don't even have to permute C.

   DIIS is similarly tiny: history ~6–10 and ~30 charges. Let one WG own a system; parallelize residual dot products and let one lane solve the tiny 6–10-dimensional DIIS equation. The small serial solve is not the problem—the PCIe synchronization is.

   Also reduce SCC convergence on GPU to one global `max_rms` scalar if the host really must inspect every iteration.

10. **[P0 — force driver violates essentially every harness rule.]** `GpuForceDriver` creates another Context/Queue/Program; takes DM/EDM as **host slices**; allocates and uploads them; allocates force output; uploads fragments; for every species bucket uploads pair lists and SK H/S; builds a kernel; calls `queue.finish()` **after every bucket**; then downloads forces.

This must be redesigned before optimizing the force arithmetic.

There should be **one shared `GpuRuntime` and one persistent dense-batch plan** for:

$$
\text{positions}\to H/S\to SCC\to P/W\to F.
$$

Static SK tables, species, orbital offsets, pair topology, q0 etc. are uploaded once. P/W are already GPU buffers. Forces remain a GPU buffer for FIRE. No intermediate CPU copy exists.

11. **[P0 — force kernel has a batch>128 correctness bug.]** It declares
    `__local Fragment l_frags[128]`, loads only the first 128 fragments, then accesses `l_frags[p.replica]`.  Your intended workload explicitly includes batches of 200, 500, 1000. This will index garbage for replica ≥128.

Do **not** “fix” this by making the array 1024 long. A `Fragment` is only a few integers. Read `fragments[p.replica]` from global/L2, or redesign the kernel around one WG/system.

12. **[P1 — get rid of atomic force accumulation if possible.]** The current pair kernel performs six CAS-loop floating-point atomics per atom pair.  This creates contention, nondeterministic summation order, and is particularly contrary to the gather-style architecture that suits these tiny molecules.

Two much better options are:

* one WG/system, local `float3 F[Na]`, with deterministic local reduction; or
* kernel 1 writes one `float3` pair force per pair, kernel 2 gathers pair forces per atom.

For Na≈30, even recomputing each pair twice in an atom-gather kernel may be cheaper than CAS atomics and much simpler/deterministic. Benchmark rather than assuming the atomics are cheap.

13. **[P0 for relaxation — geometry is still fundamentally host-built.]** `GpuPairEntry` stores `r,l,m,n`, and `GpuBatch::from_fragments()` computes them on CPU.  The Hamiltonian driver likewise owns its own context, allocates/uploads all metadata and SK data, builds kernels, finishes, and downloads H/S.

This means a future FIRE step on GPU would still effectively do:

$$
R_{\rm GPU}\to R_{\rm CPU}\to
\text{rebuild pairs}\to\text{GPU assembly}\to
H/S_{\rm CPU}\to\text{GPU SCC}.
$$

That defeats the whole project.

For these ~30-atom systems there is no reason for a complicated dynamic neighbor list. Store static `(atom_i, atom_j, species/table, orbital offsets)` topology once. Keep positions on device and calculate

$$
\Delta R,\ r,\ \hat R
$$

in the H/S and force kernels. There are only \(30\cdot29/2=435\) pairs; simply test the SK cutoff on device.

Likewise rebuild the geometry-dependent gamma matrix on GPU once per relaxation step—or compute its entries directly from GPU positions. Do not rebuild/upload it from CPU.

14. **[P1 — the convergence norm called RMS is not RMS.]** `residual_and_mix_batched` computes

$$
\sqrt{\sum_A r_A^2}
$$

while the CPU DIIS path uses

$$
\sqrt{\frac1{N_A}\sum_A r_A^2}.
$$

```
Therefore the same SCC `tol` means different things for H2O and AT. Either divide by `sqrt(n_atoms)` in the GPU kernel or call it an L2 norm everywhere. I strongly prefer the RMS convention already used by the CPU solver.
```

15. **[P1 — GEMM tests are almost meaningless at larger N.]** `gpu_tiled_gemm.rs` uses
`tol = N² * 1e-5`; at N=128 that permits a maximum matrix-element error of ~0.164.  That's far beyond any acceptable electronic-structure error. Also the test whose description says it benchmarks full-local vs tiled contains no actual timing.

Test relative Frobenius/max errors, all transpose combinations actually used by Löwdin/S⁻¹/², and physical H/S-sized matrices. Then benchmark with OpenCL event timestamps.

There are two additional optimizations I would do after these corrections rather than before them.

First, `scale_eigenvectors_batched()` recomputes the same `rsqrt(lambda_k)` once for every matrix row, i.e. roughly N² square roots.  Precompute `rlam[k]` once per system. More importantly, **do not silently hide an invalid overlap** with

```text
rsqrt(max(lambda, 1e-7))
```

without checking it. A negative physical overlap eigenvalue is an error, not something to clip. Report \(\lambda_\min\), perhaps \(\lambda_\min/\lambda_\max\), and fail on an ill-conditioned/non-positive S. The plan currently computes `lambda_min` and throws it away.

Second, once the correctness is solid, fuse some tiny SCC kernels. For these systems,

$$
\Delta q \rightarrow V=\gamma\Delta q \rightarrow H_{\rm SCC}
$$

can naturally be one WG/system kernel: load q/q0, compute Δq and V into local memory, barrier, then all threads assemble H. Three kernel launches and intermediate global traffic become one. Similarly, W should be built only after the final converged electronic solve; it is unnecessary during ordinary SCC iterations.

### What I would tell the coding agent to do next

Do **not** add new functionality yet. First make the N>64 dense path mathematically trustworthy and actually GPU-resident:

> Fix block Jacobi similarity preservation and global memory fences; replace the inner serial pivot solver with the existing Brent–Luk parallel solver; restore strict residual/orthogonality tests; tighten N>64 SCC parity; make `GpuSccPlan` genuinely own persistent kernels and buffers; move occupation/DIIS fully onto GPU; unify Hamiltonian/SCC/force under one OpenCL runtime; remove all per-force uploads/allocations/`finish()` calls; then integrate full SCC+repulsive analytic forces and total DFTB energy.

Only after that re-profile.

One important positive conclusion is that **the analytic SK derivative mathematics itself now looks much healthier than the infrastructure around it**. The H2O/formic parity achieved after fixing orientation, grid origin, and p–s radial units is strong evidence that those derivative formulas are basically correct.  The biggest current risks are now the tiled eigensolver's mathematical shortcut, the deliberately weakened tests, incomplete SCC/repulsive force physics, and a host harness that still moves/allocates far too much data.

I also would make the next validation system **actual 7-azaindole/AT at N≈84–87**, not the artificial 72-orbital 12-water cluster. The water cluster is a useful N>64 smoke test, but it does not probe the spectrum, overlap conditioning, charge redistribution, or near-zwitterionic SCC behavior that motivated this solver in the first place.
