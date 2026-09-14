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







---

# USER

OK, we updated it, based on comments in doc/prokop/reports/2026-09-10_gpu_accuracy_physics_performance_second_review.md
and added CLI unified engine using rhai script to avoid ad hoc test.
plese review it again now, both scientific rigor, performance, and f32 numerical acuracy issue, suggest imporvemnt which does not hamper speed while overcoming f32 limits. as wel as perforamnce improvement. Focus on the sparse large systems

















---

# Chat GPT 5.6 sol

I reviewed the current dense path at `e965ae00...`, together with the September 10 report/topical audit and the new Rhai `dftb_engine` workflow. This is **substantially better than the previous version**. Most of the earlier outright correctness failures have really been fixed: the tiled Jacobi now preserves the local similarity transform, the missing global fences are present, the strict Jacobi residual tests are restored, `GpuSccPlan` genuinely owns persistent kernels/buffers, occupation and DIIS are GPU-side, the final unmixed solve makes \(q,H,C,D\) mutually consistent, W is built on GPU, and total energy/forces now include repulsion and all SCC force terms.

I would **not**, however, call the current \(3\times10^{-5}\) Ha AT error a fundamental FP32 floor yet. The evidence points to a much narrower problem: accumulated loss of orthonormality in the tiled Jacobi eigenvectors. That is exactly the sort of error that can often be repaired cheaply in FP32 without converting the O(\(N^3\)) hot path to FP64. The current implementation has actually overreacted somewhat by putting FP64 into large parts of tiled Jacobi. The audit itself finds \(||C'^TC'-I||\sim2\times10^{-6}\), while frozen occupied eigenvalue errors are only \(\sim10^{-6}\) Ha and assembly is around \(10^{-7}\).

I would give the coding agent these next instructions, in this order:

* [ ] **1. Stop calling `δ_CH` a pure eigensolver/f32 floor. Diagnose normalization separately.** Right now you compare something equivalent to \(2\sum_k c_k^T Hc_k\) against \(2\sum_k\epsilon_k\). But that equality assumes \(c_k^TSc_k=1\). Since you already measure \(C^TSC-I\neq0\), part of `δ_CH` is simply normalization error. For each occupied state compute instead

  $$
  \rho_k=\frac{c_k^T Hc_k}{c_k^TSc_k},
  \qquad
  r_k=\frac{\|Hc_k-\epsilon_kSc_k\|}{\|H\|}.
  $$

  Separately print `max|c_k^T S c_k-1|` and `max_{k≠l}|c_k^TSc_l|`. Your existing diagnostic machinery already computes the relevant generalized residuals.  This will tell us whether the current ~\(5\times10^{-5}\) Ha `δ_CH` comes mostly from column normalization, mutual nonorthogonality, or a genuine eigen-equation error.

* [ ] **2. Remove FP64 from the O(\(N^3\)) parts of tiled Jacobi.** Keeping `tau,c,s` in double is defensible as a small experiment; doing every 2×2 block update, every `lU` update and every length-64 strip dot product in double is not. The current inner pivot does \(1024\times63\times20\approx1.29\) million 2×2 block updates **per pivot**, before strip/V updates, and those are presently double arithmetic cast back to float after every store.   On a 3090 this can destroy throughput while the persistent state is still only FP32.

  Use FP32 FMA for matrix updates. For length-64 dot products, replace both FP64 accumulation and serial Kahan with 4 independent FP32 accumulators:

  ```c
  float s0=0, s1=0, s2=0, s3=0;
  for(int k=0; k<m; k+=4){
      s0 = fma(a[k  ], b[k  ], s0);
      s1 = fma(a[k+1], b[k+1], s1);
      s2 = fma(a[k+2], b[k+2], s2);
      s3 = fma(a[k+3], b[k+3], s3);
  }
  float sum = (s0+s1) + (s2+s3);
  ```

  This shortens the dependency chain, preserves FMA, improves numerical summation, and is much cheaper than either Kahan or FP64. Do the same A/B experiment for `batched_gemm`: the current Kahan implementation has already been measured not to improve `δ_CH`, while adding serial dependencies to every product.

  Benchmark exactly three Jacobi modes: pure FP32-FMA; FP64 only for scalar rotation construction \(c,s\) but FP32-FMA updates; current broad-FP64 version as an accuracy reference. Measure **event time + residual + orthogonality + AT ΔE**, not just whether tests pass.

* [ ] **3. The cheapest likely accuracy fix is occupied-column renormalization. Try this before anything complicated.** After Jacobi and occupation selection, compute

  $$
  n_k^2=C_k'^TC_k',\qquad C_k'\leftarrow C_k'/\sqrt{n_k^2}
  $$

  for occupied columns. This costs only \(O(NN_{\rm occ})\), essentially nothing compared with Jacobi. If the diagonal part of the measured \(C'^TC'-I\) dominates, this alone could eliminate a large part of `δ_CH` and may even lower the SCC charge plateau. It also preserves each eigenvector direction, so existing \(\epsilon_k\) and W construction remain conceptually valid.

  If off-diagonal orthogonality remains dominant, then do **one occupied-subspace polar/Newton correction**

  $$
  G=C_o'^TC_o',\qquad
  C_o'\leftarrow C_o'\frac{3I-G}{2}.
  $$

  Since \(G=I+E\), the new metric error is \(O(E^2)\). This is exactly the sort of algorithmic compensation FP32 is good at. Do it first only in `finalize()`; if it materially improves final E/F at small cost, then test whether using it during SCC also lowers the charge floor.

* [ ] **4. Make energy and W consistent with any occupied-subspace correction.** If you only rescale columns, old \(\epsilon_k\) are fine. If you mix occupied columns through a polar correction, do **not** blindly pair the corrected columns with the old diagonal eigenvalues. The rigorous quantity is the small occupied projected Hamiltonian

  $$
  H_o=C_o^T H_{\rm SCC} C_o .
  $$

  Then

  $$
  E_{\rm band}=2\,{\rm Tr}(H_o),
  \qquad
  W=2\,C_o H_o C_o^T .
  $$

  This is invariant to rotations inside the occupied subspace and makes P, W and energy refer to exactly the same approximate occupied projector. It also attacks the current uncomfortable situation where band-form energy is numerically better than `Tr(PH0)` but is not quite consistent with the C used for forces.

* [ ] **5. Move the Löwdin/metric repair off the CPU immediately.** This is currently a serious production anti-pattern. `repair_lowdin_x()` downloads full X and S, converts them to f64, and performs several naïve serial \(N^3\) CPU GEMMs **for every replica** before uploading X again.  For batch=100–1000 this will overwhelm the very GPU parallelism we are building.

  The repair itself is mathematically nice; keep the idea but run it on GPU:

  $$
  M=X^TSX,\qquad Q=\frac{3I-M}{2},\qquad X\leftarrow XQ.
  $$

  For \(M=I+E\),

  $$
  Q^TMQ=I-\frac34E^2+\frac14E^3,
  $$

  so one step quadratically reduces a small metric defect. This needs only three ordinary FP32 GEMMs once per geometry.

  **Do not symmetrize X afterward.** It is unnecessary. Once corrected, regard X as a general orthogonalizer satisfying \(X^TSX\simeq I\), not specifically symmetric \(S^{-1/2}\). Then use

  $$
  H'=X^T H X,
  \qquad C=XC'.
  $$

  Your tiled GEMM already supports transpose flags.

* [ ] **6. Exploit that same metric correction to avoid diagonalizing S on every relaxation step.** This is a particularly attractive optimization. First geometry: construct X by the full S eigensolve. On the next small FIRE displacement, reuse the previous X and compute

  $$
  M=X_{\rm old}^TS_{\rm new}X_{\rm old}.
  $$

  If M is close enough to I, one or two \(X\leftarrow X(3I-M)/2\) corrections should restore orthogonality. Verify that the metric residual decreases; otherwise fall back to the full S eigensolve.

  Unlike the earlier unsafe-looking `X(3I-SX²)/2` idea, this formulation is a well-defined **metric orthogonalization iteration**. It also preserves parallelism: each relaxing scan point reuses only its own previous X; different scan points remain independent.

* [ ] **7. `GpuDftb` is persistent now, but geometry preparation is still mostly CPU-driven. Fix that next.** `fill_gamma()` still loops over every replica/atom pair on CPU in f64, and coordinate/pair information is reconstructed on the host.  More seriously, the current bucket implementation allocates a capacity proportional to `batch*n_atoms²` **for every bucket**, then refills/uploads pair records containing \(r,l,m,n\) after geometry changes. For AT, batch=1000 means ~900,000 pair slots per bucket; that can become hundreds of MB across buckets and huge per-step PCIe traffic.

  Species, atom indices, block type and orbital offsets **never change during relaxation**. Build the per-template pair lists exactly once. For a homogeneous batch, a bucket only needs the pair list for **one molecule**. Launch over `(replica,pair)` and have the kernel read GPU coordinates and compute \(\Delta R,r,\hat R\) itself. Check cutoff on GPU. Then geometry change means essentially one coordinate-buffer update, not reconstructing and re-uploading pair tables.

  Likewise remove the intermediate `finish()` calls from assembly; an in-order queue already orders the kernels. Zero H/S/force buffers with device fills or initialize them in their producer kernels, not by filling host vectors and uploading them.

* [ ] **8. Replace the ill-conditioned analytic gamma formula by a tiny pretabulated species-pair spline.** The current f64 `gamma_prime_full_f32` fixes the close-Hubbard-U cancellation very accurately, but runtime FP64 exp/rational arithmetic is a poor long-term solution.  There are only a handful of H/C/N/O species pairs. At initialization, evaluate \(\gamma_{AB}(r)\) accurately in CPU f64, fit/upload a compact cubic spline, and evaluate **both γ and γ′ from the same spline** on GPU in FP32.

  This simultaneously solves three problems: it removes the catastrophic close-\(U\) cancellation, removes runtime FP64 from γ′, and lets the full G matrix be constructed on-device from coordinates instead of host `fill_gamma()`. Most importantly, γ′ becomes analytically the derivative of the **same numerical γ used in SCC energy**, improving energy-force consistency.

* [ ] **9. Keep the tiny DIIS f64 island; improve its formulation rather than optimizing it away.** The current GPU DIIS correctly fixed RMS and the one-vector DIIS bug and uses scaled f64 Gram/solve, which is exactly where FP64 makes sense.  A 10×10 solve is negligible compared with N=87 Jacobi.

  But change the final affine combination so coefficient-rounding cannot inject the large neutral population \(q_0\). Mix \(\Delta q=q-q_0\), or use an anchored form such as

  $$
  q=q_{\rm ref}+\sum_{i<n-1}c_i(q_i-q_{\rm ref}).
  $$

  Then \(\sum c_i=1\) is algebraically exact instead of relying on f32-rounded coefficients. On rank failure, first drop the oldest/most-dependent history vector and retry with \(n-1\), rather than immediately falling back to α-mixing. Also remove `printf` from the production kernel; record a per-system DIIS status/counter instead. And delete `b_mat`/`rhs` device buffers if, as the current code says, they are unused.

* [ ] **10. Reduce work in density/W and SCC orchestration.** `build_density_masked_batched` currently loops over all N orbitals and multiplies by a 0/1 occupation mask.  Have the occupation kernel output `occ_idx[Nocc]` and loop only over occupied states—49 instead of 87 for AT. Since P and W are symmetric, compute only one triangle and mirror it. Use several FP32 FMA accumulators rather than a serial sum.

  Also fuse the tiny

  $$
  q-q_0\rightarrow V=\gamma\Delta q\rightarrow H_{\rm SCC}
  $$

  stages into one one-WG-per-system kernel. The data all fit comfortably in local memory. This removes two launches and intermediate global traffic. It matters especially for N=28 where Jacobi is much cheaper.

* [ ] **11. Add per-system active masks before compaction.** The current host reads the entire RMS vector each SCC iteration and waits for the slowest system.  For a 2D proton scan, different configurations will need very different SCC iteration counts. Add one `active[sid]` flag; all kernels—including tiled GEMM workgroups—immediately return for inactive systems. No physical compaction is needed initially. Track convergence/stall per system, not only through the global max.

  Also distinguish `Converged`, `AcceptedAtNumericalPlateau`, and `Failed`. The Rhai scripts currently request `1e-6` for AT, permit the engine to declare `stalled`, and then continue evaluating it.  That is acceptable for a diagnostic script, but production chemistry must not silently conflate “did not meet requested tolerance” with “converged”.

* [ ] **12. Fix FIRE before doing any relaxed PES.** The current implementation computes

  $$
  v\leftarrow(1-\alpha)v+\alpha |F|\hat F
       =(1-\alpha)v+\alpha F,
  $$

  whereas FIRE requires

  $$
  v\leftarrow(1-\alpha)v+\alpha |v|\hat F.
  $$

  The present expression literally adds a force to a velocity. It also computes a single P, dt and α over **all replicas**, so one scan point changes the optimizer parameters of every other independent scan point.

  FIRE state must be `[batch]` for `dt,alpha,n_pos` and `[batch,natom,3]` for velocity. Once corrected, move the whole trivial FIRE step to GPU. At present every `fire_step()` downloads forces, updates host coordinates, then sends geometry through the CPU `set_coords()` path. That will be disastrous for the intended large batch. `apply_disp()` also contains its own Verlet-like \(v\,dt+\frac12Fdt^2\) displacement before the later velocity update, so choose one standard FIRE integration ordering and test it against the CPU optimizer instead of mixing schemes.

### The FP32 policy I would use

| Operation                               | Recommended precision                  |
| --------------------------------------- | -------------------------------------- |
| H/S/SCC matrices, C, P, W storage       | FP32                                   |
| Dense GEMM / Jacobi block+strip updates | FP32 FMA, 4-way accumulators           |
| Jacobi \(c,s\) construction             | benchmark FP32 vs **scalar-only FP64** |
| Column normalization / polar correction | FP32                                   |
| Löwdin metric correction                | FP32 GPU GEMMs                         |
| DIIS Gram + tiny solve                  | FP64                                   |
| γ/γ′                                    | f64-generated spline → FP32 evaluation |
| final \(\Delta q\cdot V,\;q_0\cdot V\)  | FP64 accumulation is cheap             |
| final repulsive-energy reduction        | FP64 accumulation is cheap             |
| final occupied-energy sum / total E     | FP64                                   |
| coordinates on CPU/API                  | f64 is fine; device coordinates FP32   |

The important distinction is that the **final scalar reductions contain only tens/hundreds of operations once per geometry**. Using double there costs essentially nothing relative to an N=87 eigensolve. Putting double into millions of operations inside every Jacobi pivot is the opposite tradeoff.

There is one especially promising sequence I would test before anything more elaborate:

$$
\boxed{\text{FP32-FMA Jacobi}
\rightarrow
\text{normalize occupied }C'
\rightarrow
\text{measure}}
$$

If that already cuts `δ_CH` substantially, stop there. If not:

$$
\boxed{C_o'\rightarrow C_o'(3I-C_o'^TC_o')/2}
$$

once after `finalize()`. The present data strongly suggest this is where the remaining accuracy lives. I would explicitly change the audit wording from **“f32 floor”** to **“measured floor of the current FP32 Jacobi representation”** until those two cheap corrections have been tried. The restored standalone Jacobi tests already show that FP32 can satisfy residual/orthogonality \(<10^{-5}\), so there is no evidence yet for a fundamental N≈90 precision wall.

### About the Rhai engine

I like this change. `gpu_new → gpu_scc → gpu_eval/gpu_set_coords` is the right abstraction: one persistent native object and coarse script calls, rather than a new Rust executable/test for each experiment.  The measurement scripts are also much easier to audit scientifically than dozens of ad-hoc tests.

Two corrections are important. First, **do not replace mathematical unit tests with Rhai**—Jacobi residuals, derivative parity and kernel invariants belong in Rust tests; scans, A/B experiments and production workflows belong in Rhai. Second, your current `gpu_scc_bench.rs` still benchmarks the **old `GpuDriver`/legacy SCC driver**, including the old host-style setup and 500-iteration runs, rather than `GpuDftb`.  I would delete or rewrite that benchmark now. Otherwise future agents will “optimize” based on measurements of code that production no longer uses.

I would add a native coarse `gpu_bench()` operation to `dftb_engine` which times, with OpenCL events, `set_coords`, S orthogonalization, one SCC iteration, Jacobi, GEMMs, finalize, P/W, forces and total wall time for **N=28 and N=87 at batch=1, 100, 500**. That is the benchmark that should drive all further optimization.

The largest immediate wins, in my view, are therefore **(1) remove broad FP64 Jacobi arithmetic, (2) cheap C′ normalization/orthogonality repair, (3) kill the CPU Löwdin repair, (4) stop rebuilding/uploading geometric pair records, and (5) correct/move FIRE to GPU**. Those five changes improve accuracy *and* throughput rather than trading one for the other.







---


# USER

please review the code in context of problem documented by the agents in this labbook, and suggest further path forward (focu on dense multi-system solver, let the sparse solver alone 

give clear instruction and best path forward to high speed and reasonable accuracy with f32 GPU arithmetic
https://github.com/ProkopHapala/dftbplus/commit/55ddb4c22867923810d2a950c8d9c046b8c7ab24


---


# Chat GPT 5.6 sol


I reviewed commit `55ddb4c22867923810d2a950c8d9c046b8c7ab24` against the labbook, focusing only on the dense `GpuDftb/GpuSccPlan` path. The central conclusion is stronger now:

**The FP32 accuracy problem is largely solved. The N>64 performance problem is not. Stop spending effort on generic compensation/Kahan/f64 islands and replace the tiled block-Jacobi architecture.**

The current verified chemistry is already quite good: AT/GC/AZA total-energy errors are around \(10^{-6}\) Ha, forces a few \(10^{-6}\) Ha/Å, and the formic relative PES agrees to about \(10^{-6}\) Ha. The occupied-column normalization and Rayleigh correction were the decisive accuracy fixes.   The later γ/γ′ spline and device-resident geometry work also look like the correct architecture and should be left alone. 

The performance evidence is equally decisive: an N=87 production eigensolve takes **~83 ms**, batch 1 and batch 8 cost almost identically, and one warm SCC iteration is ~74 ms all the way from batch 1 to batch 64. The kernel is not compute-bound; it is a single workgroup spending its life behind thousands of barriers. 

### Instructions I would give the coding agent

1. **First remove all redundant eigensolves. Do this before designing a new eigensolver.** `eval()` unconditionally calls `finalize()`, which performs another H′ construction + Jacobi even immediately after SCC. `fire_step()` calls `eval(true)`, and then `relax()` calls `scc()` followed by another `eval(false)` merely to print energy. The code confirms the labbook accounting.

   Change the SCC state machine. Do **not** overwrite `q_gpu` with the mixed next iterate before knowing whether another SCC iteration is needed. Introduce `q_next`. One iteration should be:

   $$
   q_n\rightarrow H(q_n)\rightarrow C,D\rightarrow q_D
   \rightarrow{\rm DIIS}(q_n,q_D)\rightarrow q_{n+1}.
   $$

   If the residual is acceptable, retain \(q_n\) and the electronic state just calculated. That state is already mutually consistent to the requested SCC tolerance. `eval()` can then reuse `H,C,D,\epsilon` and only build W/energy/forces. If another iteration is required, commit `q_next→q_gpu`.

   This removes the expensive “finalize for consistency” rather than merely hiding it. Also delete `eval(false)` from every FIRE logging step; calculate energy from the valid cached electronic state. This alone should cut roughly 2–3 eigensolves from each geometry step.

2. **Replace the N>64 block-Jacobi kernel. Do not optimize its current nested structure.** This is the most important instruction. Current N=86/87 does 3 compound block pivots per outer sweep; each pivot performs **20 inner sweeps × 63 Brent–Luk rounds**, with several workgroup barriers per round. `JACOBI_TOL=1e-9` then never terminates normally in FP32, so the solver waits for its stagnation heuristic.   The labbook correctly identified this as the ~83 ms catastrophe and also correctly notes that the relative-to-`off0` stopping criterion actively sabotages warm starts. 

   For the immediate N≈65–128 target, I recommend a **direct full-matrix parallel cyclic Jacobi**, still one kernel launch and one WG/system:

   * A and V remain in global memory.
   * Pad N to even `JN`; N=87 becomes 88, giving 44 independent Jacobi pairs per round.
   * One round has 44 disjoint rotations.
   * Each pair computes \(c,s\).
   * The WG applies \(J^TAJ\) to the whole matrix as pair×pair 2×2 blocks, exactly the algebra you already use inside the local 64×64 pivot.
   * Update eigenvector columns \(V\leftarrow VJ\).
   * Barrier; proceed to the next of 87 rounds.
   * No 64×64 “inner eigensolve”, no block-pair outer hierarchy.

   The current architecture performs of order **tens of thousands of synchronized Jacobi rounds** per eigensolve. Direct cyclic Jacobi needs roughly \(87\times N_{\rm sweep}\) rounds. Even five cold sweeps is only 435 rounds. That attacks the actual latency mechanism instead of shaving FLOPs inside it.

   Keep N≤64 exactly as it is; it is already fast. This replacement is only for the N>64 path.

3. **Make warm-starting intrinsic to the mathematical formulation.** Simply loading the previous eigenvectors into V while diagonalizing the new H′ is wrong unless A is transformed into that basis. Instead I recommend a particularly clean formulation using the **previous AO eigenbasis itself**.

   Maintain a full \(S\)-orthonormal basis \(B\), initially the previous eigenvectors:

   $$
   B^T S B\simeq I.
   $$

   For the next SCC iteration form

   $$
   A = B^T H_{\rm SCC} B.
   $$

   This needs two ordinary batched GEMMs. Because H changes only slightly between SCC iterations, A should already be almost diagonal.

   Then run direct Jacobi on A while applying every Jacobi rotation directly to B:

   $$
   A\leftarrow J^TAJ,\qquad B\leftarrow BJ.
   $$

   On exit **B is the new AO eigenvector matrix C**. There is no \(X C'\) back-transform at all.

   This is exact, not an approximation: any full basis satisfying \(B^TSB=I\) converts the generalized eigenproblem to an ordinary symmetric one. It also naturally carries eigenvectors from SCC iteration to SCC iteration. The expected regime should be **1–2 Jacobi sweeps after the first iteration**, rather than eight cold sweeps every time.

   Across FIRE geometry changes, repair the previous B against the new overlap:

   $$
   G=B^TS_{\rm new}B,\qquad
   B\leftarrow B(3I-G)/2.
   $$

   You already proved this metric-Newton idea works for X. If it fails its residual test, fall back to the existing X construction. The corrected \(X^THX\) gauge handling was an important fix and validates exactly this more general view of the orthogonalizer. 

   I would implement this warm basis only **after** the direct-global Jacobi works cold, so the two changes can be validated independently.

4. **Replace the impossible Jacobi stopping rule with one tied to the physical matrix scale.** Never again use

   $$
   \|A_{\rm off}\|/\|A_{\rm off}^{(0)}\|<10^{-9}.
   $$

   A better invariant is

   $$
   \frac{\|A_{\rm off}\|_F}{\|A\|_F}<\epsilon_{\rm eig},
   $$

   where \(\|A\|_F\) is invariant under Jacobi similarity transforms and, crucially, does not become smaller merely because the warm start is better.

   Do not guess an absurd tolerance. Start around the currently demonstrated prec1 accuracy and tune against the actual scientific diagnostic:

   $$
   \max_k\|HC_k-\rho_kSC_k\|,
   $$

   plus AT/GC/AZA energy parity and relative PES parity. The existing successful Rayleigh repair should stay.

   Likewise `PAIR_SKIP_TOL=1e-12f` is inappropriate for FP32. A rotation whose off-diagonal element is many orders below the FP32 scale cannot improve the stored matrix. Use a relative criterion such as

   $$
   |a_{pq}| < \epsilon_J(|a_{pp}|+|a_{qq}|),
   $$

   with \(\epsilon_J\) empirically around the FP32 solver floor. That will increasingly skip useless rotations as the matrix diagonalizes.

5. **Keep FP64 only where it is cheap; do not chase compensated arithmetic inside the current Jacobi.** The labbook already shows prec1 is scientifically sufficient while prec0 in the current formulation is not.  For now keep f64 only for scalar \(c,s\) construction. Everything that touches N²/N³ data stays FP32 FMA.

   Once the direct solver is fast, retry a stable all-FP32 rotation formula. The current 83→47 ms prec1→prec0 difference shows that even the scalar f64 sqrt/div sequence hurts badly on a 3090 because every round waits on it. But that is **second priority**. Removing ~100× excess synchronization comes first.

   Do not implement TwoSum/Kahan inside the old tiled Jacobi now. It would polish arithmetic inside an algorithm whose wall time is dominated by barriers.

6. **Add a real device `active[sid]` mask. This is both performance and correctness.** There is a subtle bug in the current host-side “done” machinery: `scc_mix()` stops tracking a converged replica, but subsequent GPU iterations still run Jacobi, density, Mulliken and DIIS for that replica because none of the kernels knows it is done. At the end, status is reconstructed from the *latest* RMS, so a previously converged system can in principle drift again.

   Add `active[batch]`. Major kernels should uniformly return for `active[sid]==0`, including GEMM tiles and the eigensolver. When a replica reaches either accepted convergence or an accepted numerical plateau, freeze its \(q,C,D,\rho\) state.

   Also tighten `PLATEAU_FLOOR`. The current **5×10⁻⁴** is much looser than the measured dense FP32 floor of roughly \(10^{-6}\)–\(10^{-5}\). A sensible production policy is something like

   $$
   r_{\rm plateau,max} =
   \max(3\,r_{\rm target},\,2\times10^{-5}),
   $$

   calibrated on AT/GC/AZA. A stagnant \(10^{-3}\) or \(10^{-2}\) state is a failed SCC, not an accepted plateau.

   I would also make **1e-5 the default production SCC target**, not 1e-6. Use 1e-6 for validation runs. The current physics parity is already much better than the chemistry requires while AT/GC sit near that numerical boundary. 

7. **Fix finite-temperature smearing before trusting the proton-transfer PES.** The smearing was a good numerical/physical solution to the real 0.45 mHa gap problem, but the labbook correctly identifies two current scientific errors. 

   First, with Fermi occupations the reported quantity must be the Mermin free energy:

   $$
   F_{\rm el}
   =
   E_{\rm el}
   +
   2kT\sum_k
   \left[
   f_k\ln f_k+(1-f_k)\ln(1-f_k)
   \right].
   $$

   At \(kT=0.002\) Ha, a half-occupied spatial orbital contributes roughly

   $$
   -2 kT\ln2\approx-2.77\times10^{-3}\ {\rm Ha}
   \approx-1.74\ {\rm kcal/mol},
   $$

   so this is absolutely not negligible for the proton-transfer profile.

   Second, the current code is internally inconsistent under smearing. `build_density_occ_batched` includes **all N orbitals** with Fermi weights, while `occ_rayleigh_batched` explicitly computes \(\rho_k\) only for the integer `occ_mask` and writes zero for the others.   Yet W chooses `eig_rho` whenever `occ_repair` is enabled. That makes fractionally occupied virtual states contribute correctly to D but incorrectly to W.

   Under smearing:

   $$
   D=2\sum_k f_k C_kC_k^T,
   \qquad
   W=2\sum_k f_k\rho_k C_kC_k^T,
   \qquad
   E_{\rm band}=2\sum_k f_k\rho_k.
   $$

   Normalize/Rayleigh-repair **all states with non-negligible \(f_k\)**—for N≈87, simply doing all N is fine—and add the entropy term. Host-f64 μ bisection is exactly the kind of tiny discrete decision where FP64 is appropriate; leave that alone for now.

   Once this is done, finite-temperature forces built from fractional D/W are the appropriate stationary derivative of the Mermin free energy. Label the result by kT; it is no longer a zero-temperature PES.

8. **Only after the replacement eigensolver is working, benchmark against a professional batched solver.** Current NVIDIA cuSOLVER provides `cusolverDnXsyevBatched` for batches of same-size contiguous symmetric matrices, including FP32 matrices/eigenvectors. ([NVIDIA Docs][1]) Do a standalone CUDA microbenchmark at exactly N=87 and batch 1/19/64/256 as a **performance oracle**, not as a silent production fallback.

   I would not integrate CUDA into the OpenCL engine yet; moving `cl_mem` back and forth merely to call cuSOLVER defeats the architecture. But if a redesigned OpenCL eigensolver is still taking 10–50 ms while cuSOLVER does the same work orders of magnitude faster, that is useful evidence about whether a later CUDA dense backend is justified.

I would **not implement LOBPCG/subspace iteration next**, despite the labbook mentioning it. For GC, 43-ish occupied states out of 86 is already half the spectrum, and smearing requires several frontier virtuals exactly where the gap becomes tiny. You would introduce a much more complicated convergence problem to avoid diagonalizing a matrix that is only 86×86. A sane full eigensolver at this size should be cheap. Partial eigensolvers become attractive only if the dense target grows substantially beyond ~100–200 orbitals.

### Precision policy I would freeze now

| Quantity / operation                 | Precision                                            |
| ------------------------------------ | ---------------------------------------------------- |
| H, S, C, D, W, X/B storage           | **f32**                                              |
| GEMM, Jacobi matrix/vector updates   | **f32 FMA**                                          |
| Jacobi \(c,s\)                       | **f64 temporarily**, then test stable f32            |
| occupied/all-weighted normalization  | f32                                                  |
| Rayleigh quotients                   | f32 matrix work; optional f64 final scalar reduction |
| DIIS Gram + tiny solve               | **f64**                                              |
| Fermi μ / convergence decisions      | **f64 host**                                         |
| γ/γ′                                 | current f32 spline                                   |
| final energy / entropy / scalar dots | **f64 reduction**                                    |
| coordinates API                      | f64 host → f32 device                                |

This follows the principle already written into the project: **spend precision on decisions/scalars and restoring invariants, not on bulk matrices.** The current renormalization/Rayleigh repair is a very good example of this: it improved AT energy by roughly 30× and made SCC converge faster, rather than making the matrix algebra more expensive. 

### The concrete target architecture

For N>64, I would aim for this hot SCC loop:

$$
\boxed{
q
\rightarrow H_{\rm SCC}
\rightarrow A=B^THB
\rightarrow
\text{1--2 warm direct-Jacobi sweeps}
\rightarrow C=BJ
\rightarrow D,q_D
\rightarrow DIIS
}
$$

with `B←C` only when another iteration is required. No cold eigensolve, no X transforms during ordinary SCC iterations, no redundant finalize, no work on already-converged replicas.

For a geometry step:

$$
\boxed{
R_{\rm new}
\rightarrow H_0,S,G
\rightarrow
\text{metric-repair previous }B\text{ to new }S
\rightarrow
5\text{--}10\ {\rm SCC\ iterations}
\rightarrow
E/W/F
\rightarrow FIRE
}
$$

The labbook explicitly states that **batch-parallel execution is the product**, not merely an optimization.  I would therefore set hard performance gates for the next agent: N=87 direct eigensolve **<5 ms cold as the first milestone, <1–2 ms warm**, ordinary warm SCC **<10 iterations**, no routine 100-iteration calls, and the 19-point GC relaxed scan should move from 108 seconds toward **a few seconds**, not merely 50–70 seconds. The current 108 s is almost entirely accounted for by repeatedly paying the 83 ms Jacobi tax, so this is a realistic place to expect an order-of-magnitude-plus improvement. 

One final scientific housekeeping point: the later labbook notes a 1–2 Ha CPU/GPU discrepancy in some separate diagnostic scripts, while the same-engine parity tests report ~\(10^{-6}\) Ha. The actual CPU implementation uses \(E=\mathrm{Tr}(DH_0)+\tfrac12\Delta q\cdot V\), which is equivalent to the GPU band form for the same converged state.  Therefore **do not debug that multi-Hartree discrepancy as FP32**. First require comparisons to go through the same `GpuDftb` object/geometry/species via `gpu_measure(..., true)` and `cpu_ref()`. A multi-Hartree difference is a mismatched system or energy convention until proven otherwise.

The immediate work order I would enforce is therefore: **cached SCC state → direct-global Jacobi → warm AO basis → active mask → fix smeared free energy/W → only then secondary micro-optimizations.**

[1]: https://docs.nvidia.com/cuda/cusolver/index.html?highlight=cholesky&utm_source=chatgpt.com "1. Introduction — cuSOLVER 13.4 documentation"







---

# USER

please analyse recent changes  in code primarily in
https://github.com/ProkopHapala/dftbplus/tree/main/rust_dftb/src/qmqm
https://github.com/ProkopHapala/dftbplus/blob/main/rust_dftb/src/qmqm/gpu_scc_plan.rs
https://github.com/ProkopHapala/dftbplus/blob/main/rust_dftb/src/qmqm/gpu_tiled_jacobi.cl
end report/notes in
https://github.com/ProkopHapala/dftbplus/blob/main/doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.labbook.md

suggest what are the biggest architectureal problems, optimization opportunitie, and how we can adress limitations of f32 precisions efficiently by some hybrid f64 appoach or better Kahan error compensating sum without hampering the performance. Make sure we remove all stupid fallback, allocation of buffers and compilation of kernels in hot loops and also eliminate unnecessary CPU-GPU data transfer, also that the lernels are optimally using local memory and workgroup sizes, and does not suffer by other issues hampering performance



---

# Chat GPT 5.6 sol

I reviewed the current `main` dense path, including the recent A1–A4/B1–B2/R1–R5 work in the labbook. The code is in a much better state than at `55ddb4c`: the catastrophic 83 ms tiled Jacobi has been bypassed by the direct cyclic solver, SCC has a proper `q_gpu/q_next` commit model, the AO eigenbasis is warm-started, early-converged replicas are masked correctly, W has its own state-validity mask, Mermin smearing is now physically consistent with a smeared f64 CPU reference, and FIRE is device-resident and much more defensible. The reported GC batch-19 warm SCC around 1.3 ms and GPU↔CPU energies around \(10^{-6}\) Ha show that the basic scientific architecture is now working.

The main conclusion has changed accordingly: **the dominant opportunities are no longer “invent a better eigensolver” or “use more f64.” They are eliminating synchronization points, making the generic GEMM genuinely fast, reducing unnecessary matrix repair work, and replacing nondeterministic atomic force accumulation.** The existing f32 accuracy is already considerably better than needed for screening H-bond PESs; the correct strategy is to enforce invariants and use tiny f64 reductions, not to widen bulk matrix arithmetic.

## What I would change next

1. **Remove the blocking Jacobi diagnostic read from every SCC iteration. This is now a P0 performance bug.** `eigh_solve()` calls `report_jacobi()` immediately after every N>64 H eigensolve. `report_jacobi()` downloads `jacobi_diag`, and `GpuRuntime::read_buffer()` explicitly calls `queue.finish()`. So every otherwise asynchronous SCC iteration is forcibly drained immediately after Jacobi.

   Keep the excellent `{off, off/frob, stop, sweeps}` diagnostics, but leave them on device. A tiny SCC-control kernel should inspect the Jacobi stop record together with the charge residual and update per-replica state. Only read the diagnostics when a solve ends or when a nonzero stop is actually detected. For S-Jacobi, which occurs only on exceptional cold geometry rebuilding, a blocking check is fine.

   Also change the policy for nonzero Jacobi stops. At present the code warns and still consumes the eigenvectors/eigenvalues. `stop=4` nonfinite must hard-fail that replica. `stop=2` max-sweep should fail unless the achieved residual is explicitly within an accepted f32 floor. `stop=1` on a warm solve may trigger a **GPU cold solve**, but that must be an explicit numerical recovery, not merely a warning. The direct kernel currently zeros the residual off-diagonal even on bad exits, which destroys useful diagnostic information.

2. **Move the SCC convergence controller onto the GPU. Right now you still synchronize CPU↔GPU once or more per iteration.** `scc_step_diis()` reads the entire RMS vector every iteration; the host then decides which systems are done, uploads a new `active` mask, and enqueues `q_next→q_gpu`.   The bytes transferred are tiny, but the latency/queue serialization is the problem.

   Add one persistent device-side SCC-control record per system:

   $$
   \{\text{active,status,rms,rms-history,niter}\}.
   $$

   At the end of an SCC iteration a tiny kernel should classify convergence/plateau/failure, update `active`, and commit `q_next` only for replicas which need another iteration.

   Then enqueue SCC iterations in chunks of perhaps 4 without any host read. After a chunk, read only a small global `n_active/status` record. Once all heavy kernels respect `active`, iterations after a replica converges are practically free. This can later become 8–16 queued iterations if profiling supports it.

   This requires an **active-gated SCC GEMM** as well. Right now the Jacobi/density/etc. kernels exit for inactive systems, but the GEMM tiles still run. Make a dedicated `batched_gemm_active` for the SCC path; keep the existing ungated GEMM for Löwdin/metric work.

3. **The current generic `batched_gemm` should lose Kahan summation.** This is one of the clearest remaining “accuracy fix that hurts speed without solving the real error.” Every multiply currently does Kahan:

   $$
   y=p-c,\quad t=s+y,\quad c=(t-s)-y.
   $$

   That serializes the inner product, prevents an optimal FMA dependency structure, and adds several instructions per multiply. The earlier experiments already found generic GEMM Kahan did not cure the electronic error—the successful fixes were normalization, Rayleigh quotients and metric restoration.

   Use four independent FP32 FMA accumulators:

   ```c
   s0 = fma(a0,b0,s0);
   s1 = fma(a1,b1,s1);
   s2 = fma(a2,b2,s2);
   s3 = fma(a3,b3,s3);
   ...
   sum = (s0+s1) + (s2+s3);
   ```

   This simultaneously reduces dependency-chain latency and usually improves summation accuracy over one long FP32 accumulator.

   There is an additional potentially large GEMM problem: **transposed operands are currently loaded uncoalesced**. For `trans_a=1`, consecutive loader threads effectively access `A[ac*n+ar]`, i.e. stride-N addresses. The warm SCC path performs \(B^T H\) every iteration, so this matters directly.

   For the transposed loader, change the work-item→tile mapping so `ar` varies fastest in global memory, then store the values transposed into `As`. Alternatively maintain `Bᵀ` explicitly and use the fast NN GEMM. Benchmark both. I suspect this plus removing Kahan will be a substantial fraction of the next speedup.

4. **Re-profile the direct Jacobi with pure f32 rotations now. The old “prec0 is bad” conclusion was about a substantially different solver.** The new direct solver is mathematically much cleaner: stable \(t\)-formula, absolute norm-scaled convergence, relative pair skipping, warm input and Rayleigh/normalization repair. Its current production mode still computes every \(c,s\) with f64 sqrt/div while the entire WG waits.

   Importantly, `app`, `aqq`, and `apq` are already f32. Casting them to double cannot recover information that was lost when the matrix was stored; it only prevents additional rounding in the scalar formula.

   Benchmark the **production direct kernel**, not deprecated tiled Jacobi, with `JACOBI_PREC=0` and `1` on AT/GC/AZA. Compare event time, generalized residual, \(C^TSC-I\), relative PES energy, force parity and SCC iteration count. The latest direct-kernel tests already show N=87 residuals around \(10^{-6}\) and orthogonality around \(10^{-7}\) in the current implementation.

   If pure f32 is marginal, I would prefer “f32 early sweeps + one high-precision refinement sweep” over f64 rotations throughout. But first test pure f32: I think there is a good chance the current Rayleigh/metric restoration makes it sufficient.

   Also benchmark WG = 128, 256, 512 and possibly 1024 for N≈87. With batch=19, one-WG/system fundamentally activates only 19 of 82 SMs; a larger WG may at least provide more warps and memory-level parallelism on each active SM. The direct kernel has little local-memory pressure compared with the old tiled kernel, so 512 is quite plausible. Query the **kernel-specific** workgroup limit, preferred multiple, local memory and private/register allocation rather than relying only on device defaults.

5. **Treat S-orthogonality of the entire warm basis as the invariant—not just individual column normalization.** The warm formulation

   $$
   A=B^THB
   $$

   is exact only when

   $$
   B^TSB=I.
   $$

   `snormalize_batched` repairs \(b_k^TSb_k=1\) independently, which is useful but does not repair off-diagonal \(b_i^TSb_j\).

   Jacobi rotations should preserve the metric if B starts good, so I would not pay for a full polar repair every SCC iteration. Instead monitor the full metric occasionally or when SCC/Jacobi stalls, and apply

   $$
   G=B^TSB,\qquad
   B\leftarrow B(3I-G)/2
   $$

   only when required. This is the right kind of f32 correction: **restore the invariant**, rather than increasing precision everywhere.

   The geometry-change basis repair is mathematically sensible but currently expensive: each Newton attempt performs several GEMMs and then two blocking scalar-array reads.  Move the acceptance loop completely onto the GPU.

   There is also a useful cheap bound. If

   $$
   G=I+E,\qquad Q=I-E/2,
   $$

   then

   $$
   Q^TGQ-I=-\frac34E^2+\frac14E^3.
   $$

   Hence with a submultiplicative norm \(e=\|E\|\),

   $$
   e_{\rm new}\le 0.75e^2+0.25e^3.
   $$

   If the initial metric defect is \(2.7\times10^{-3}\), this bound is roughly \(5.5\times10^{-6}\). You can therefore certify many small FIRE geometry updates after **BᵀSB + one BQ multiply**, without recomputing \(B_1^TSB_1\). Use a true matrix infinity/row-sum norm if you want to rely rigorously on this bound.

6. **Move Fermi occupation completely onto the GPU.** Smearing currently downloads all eigenvalues every SCC iteration, runs 80 f64 host bisection iterations per replica, then uploads all weights.  The algorithm is scientifically correct now—the Mermin \(-TS\) term and weighted Rayleigh values have been fixed—but this is still an unnecessary host synchronization.

   One WG/system can solve μ directly. I would use safeguarded Newton:

   $$
   N(\mu)=\sum_k f_k,\qquad
   N'(\mu)=\frac1{kT}\sum_k f_k(1-f_k).
   $$

   Parallelize the two sums over the ~87 states. Usually 5–8 iterations should suffice; retain a bracket and fall back to bisection *inside the same GPU kernel* when Newton leaves it.

   FP32 `exp` and weights are sufficient here. If you want a precision island, keep only μ and the final particle-count reduction in f64. Eight GPU iterations are preferable to 80 host iterations plus two PCIe synchronizations.

   In the smeared path you can also skip the bitonic integer occupation sort entirely: D and W use all orbitals weighted by \(f_k\), so sorting is irrelevant unless requested for diagnostics.

7. **Move final energy and residual reductions onto GPU, using f64 exactly where it is cheap.** `energy_from_state()` presently downloads `eig_diag`, `eig_rho`, `occ_mask`, \(\Delta q\cdot V\), \(q_0\cdot V\), and repulsive energy, and allocates several host vectors every call. `charge_residual()` downloads the full q and q_D vectors.

   This is an ideal hybrid-f64 target. One WG/system should calculate in f64:

   $$
   E_{\rm band}=2\sum f_k\rho_k,
   $$

   the Mermin entropy,

   $$
   \tfrac12\Delta q\cdot V,\qquad q_0\cdot V,
   $$

   and add \(E_{\rm rep}\). Only one `double E[batch]` needs to reach the host.

   Similarly make one GPU reduction return `{q_rms,q_max}`. There is no need to transfer two atom-sized vectors merely to calculate two scalars.

   This is where f64 is nearly free and actually scientifically valuable. It is much better than f64 GEMM.

8. **The FIRE loop is device-resident, but the adaptive controller still round-trips every step.** `fire_step()` downloads \(4\times batch\) floats, performs dt/α decisions on CPU in f64, uploads \(4\times batch\) controls, and then applies the step on GPU.  This is far better than downloading forces/coordinates, but at a millisecond-scale target the synchronization itself becomes relevant.

   Put `dt`, `alpha`, `n_pos`, and mode permanently on device. The reduction kernel can update them directly after P, \(\|v\|^2\), \(\|F\|^2\), maxF are known. These are exactly the tiny scalar decisions where f64 on the GPU is acceptable if you want it. Then `fire_apply` consumes the device control structure directly.

   The host only needs a compact global “all relaxed?” flag/maxF every few steps or at the end.

   Also add a **geometry-active mask**. Once one scan replica has converged/parked, do not rebuild its γ, H0/S, pair geometry or metric repair every FIRE step while other replicas continue.

9. **Fuse geometry refresh with H/S assembly and, later, with force evaluation.** Current geometry assembly first writes dynamic `r,l,m,n` into every `PairEntry`, then later assembly reads those values back from global memory.  For ≤30 atoms this intermediate representation is unnecessary.

   Store only template-static pair data:

   $$
   i,j,\text{orbital offsets},\text{SK table id}.
   $$

   `assemble_pairs` should load the two coordinates and calculate \(R,r,l,m,n\) itself. The force kernel can do exactly the same. This removes a kernel launch and global write/read of the dynamic pair structure.

   Also, onsite diagonal values are geometry independent. Initialize those once instead of launching an onsite kernel every geometry unless some future model actually changes them.

10. **The force kernels should be the next numerical+performance cleanup after GEMM.** `force_pairs` computes the non-SCC term and atomically scatters six components; `force_pairs_scc_shift` separately recomputes \(dS/dR\) and performs another six atomics.  The formula naturally combines:

    $$
    F_{ij}
    =
    2\sum_{\mu\nu}
    \left[
      D_{\mu\nu}\frac{\partial H^0_{\mu\nu}}{\partial R}
      +
      \left(\bar V D_{\mu\nu}-W_{\mu\nu}\right)
      \frac{\partial S_{\mu\nu}}{\partial R}
    \right].
    $$

    Fuse those first. One interpolation/derivative evaluation, one pair-force result.

    More importantly, get rid of atomic force accumulation. Have the pair kernel write one `float3 Fpair[p]`; a second atom-owned kernel gathers all incident pair forces using a static pair→atom adjacency list. That gives deterministic summation, avoids CAS loops and makes a tiny **f64 per-atom gather** possible. An atom has only a few tens of incident pairs, so accumulating those in double then storing f32 is vastly cheaper than doing f64 inside the SK derivative algebra.

    There is another likely performance issue in the p–p force path: each work-item declares many `float4[4]` derivative arrays for H and S. That is well over a hundred scalar floats before compiler optimization and risks register spilling/private-memory traffic. The block values themselves are often not even consumed. Rewrite the p–p routine to **contract derivatives directly into the three force components** rather than materializing six 4×4 derivative matrices.

    Only after that tune `PAIR_WG`. The current 64-thread WG reloads the SK table for every WG. A 128/256-thread WG amortizes that load better, but private-register pressure must be fixed first.

11. **Remove the remaining allocations from normal solver control.** There are still small but unnecessary allocations once the code is supposed to run in milliseconds. `reset_diis()` allocates a fresh `Vec<i32>` and uploads it to four buffers every geometry. Replace that with device `fill(0)` operations or one reset kernel.

    Production `scc_mix_inner()` also creates `hist`, `stagnant`, `done`, and even the host-f64-DIIS work arrays on every solve although production uses GPU DIIS.  Once SCC control is device-side most of these disappear. Until then, make them persistent fields and allocate the host-DIIS arrays only when `mix==2`.

    The same applies to energy and DIIS-status temporary vectors.

    Diagnostic `measure()` can allocate freely; it is not the hot path.

12. **Eliminate the remaining architectural “fallback traps.”** I do not see a silent CPU fallback in the current production `GpuDftb` dense path, which is good. The CPU DIIS route is explicitly selected by `mix==2` and batch=1. But there are still three traps worth removing.

    First, `scc()` automatically seeds a failed replica from the **nearest array-index** converged replica.  This is now state-honest, but it is still inappropriate as a generic SCC solver behavior. Array index is not a physical metric. Move this into the proton-scan driver where the continuation graph is known explicitly. Core SCC should report `Failed`; an explicit `retry_with_seed(src,dst)` workflow may then be invoked.

    Second, `gpu_eigen.rs` still presents `jacobi_batched()` as the “production entry point” but dispatches N>64 to the **deprecated tiled solver**, while the actual production `GpuSccPlan` uses direct Jacobi. Its one-shot helpers also build kernels and allocate diagnostic buffers per call.  Mark these APIs diagnostic/deprecated or make them dispatch to the new direct solver.

    Third, `GpuMatrixContext` still creates its own OpenCL context/queue and builds kernels in individual operation calls.  That is fine as legacy/test code, but it should be visibly fenced off so a future agent cannot accidentally use it in `GpuDftb`. `GpuDftb + GpuRuntime + GpuSccPlan` should be documented as the **only production dense engine**.

## Precision strategy I would freeze

| Operation                      | Production choice                                                        | Why                                            |
| ------------------------------ | ------------------------------------------------------------------------ | ---------------------------------------------- |
| H/S/C/B/D/W/X storage          | **f32**                                                                  | Current parity already ~µHa                    |
| GEMM                           | **f32 FMA, 4 accumulators**                                              | Fast; generic Kahan is counterproductive       |
| Jacobi A/V updates             | **f32 FMA**                                                              | Bulk O(N³) work                                |
| Jacobi \(c,s\)                 | Test **f32 default** now; f64 only if direct-kernel A/B proves necessary | Inputs are already f32                         |
| \(B^TSB\) restoration          | f32 GEMM + Newton/polar correction                                       | Enforce invariant rather than widen arithmetic |
| Rayleigh \(Hc,Sc\)             | f32 FMA                                                                  | Bulk matrix-vector work                        |
| Rayleigh final \(c^THc,c^TSc\) | optional **f64 scalar reduction**                                        | Cheap useful precision                         |
| Mulliken rows                  | f32, 4 accumulators                                                      | Prevent long serial accumulation               |
| DIIS                           | **f64 QR / tiny solve**                                                  | Tiny dimensions; conditioning matters          |
| μ for smearing                 | GPU f32 weights; optional f64 μ/count scalar                             | No PCIe synchronization                        |
| Total energy / entropy / dq·V  | **f64 reduction**                                                        | Essentially free                               |
| Force pair algebra             | f32                                                                      | Bulk                                           |
| Final atom force gather        | optional **f64 accumulation**                                            | Few terms/atom, deterministic                  |
| Convergence decisions          | f64 or robust f32 scalar                                                 | Precision belongs at decisions                 |

I would specifically **not use Kahan in GEMM**. If a final 30–100-term scalar contraction needs better accuracy and f64 is undesirable, Neumaier/TwoSum or float-float is reasonable there. But compensating every one of millions of matrix multiply terms is the wrong trade.

The broader pattern that has already worked extremely well in this project is:

$$
\boxed{\text{fast f32 evolution} + \text{cheap restoring projection} + \text{accurate scalar certification}}
$$

Occupied normalization, Rayleigh correction, metric Newton repair and Mermin particle-number enforcement are all examples of that.

## One numerical change I would also make to DIIS

The labbook still mentions frequent `reason=1` DIIS pivot/scale fallbacks. Rather than tolerating frequent fallback to simple α-mixing, fix the small linear algebra.

Do not form and invert a poorly conditioned Gram/KKT system if you can avoid it. With ~29 atoms and history ≤10, solve the Pulay constrained least-squares problem using **f64 pivoted QR** of the residual-history matrix itself. This avoids squaring the condition number through \(R^TR\). The operation count is only a few thousand f64 operations/system—negligible compared with the dense electronic solve. Dependent history vectors can be explicitly dropped based on the QR diagonal. Then simple mixing becomes an exceptional first-iteration behavior, not a routine numerical fallback.

## What I would profile now

The old 83 ms Jacobi monster is gone, so previous profiling priorities are obsolete. The labbook reports the post-A1–A4 warm GC batch-19 SCC at about 1.27 ms and eval+forces around 0.73 ms, but the subsequent R1–R5 work focused mainly on correctness and the labbook does not yet establish a fresh complete event-timing breakdown for the latest code.

I would now instrument exactly these stages with OpenCL event times and host wall time: projected GEMM \(B^THB\), direct Jacobi, density+Mulliken+DIIS, CPU synchronization overhead, geometry assembly, basis metric repair, W+forces, and FIRE. The important number is **GPU event sum versus host wall time**. If the GPU events sum to 0.3 ms but SCC wall time is 1.3 ms, we know immediately that the remaining problem is synchronization rather than arithmetic.

The first performance experiments I would run are therefore:

| Experiment                                                         | Expected information                         |
| ------------------------------------------------------------------ | -------------------------------------------- |
| GEMM Kahan vs 4×FMA                                                | likely direct speed win                      |
| GEMM transposed-load old vs coalesced                              | warm-basis SCC bottleneck                    |
| Direct Jacobi prec0 vs prec1                                       | whether f64 scalar island is still justified |
| Direct Jacobi WG 128/256/512                                       | occupancy/latency optimum for N≈87           |
| SCC current host-control vs 4-iteration device-control chunks      | cost of queue synchronization                |
| Basis repair current 5-GEMM verified vs 3-GEMM bounded common path | geometry/FIRE cost                           |
| Pair-force atomic vs pair-write+atom-gather                        | force speed + deterministic precision        |

I would **not start another eigensolver project yet**. The current direct warm Jacobi has moved from ~83 ms to the low-millisecond regime and passes demanding N=65–256 eigenquality tests.  There is a lot of much safer performance left in removing queue drains, fixing GEMM, and cutting geometry/force overhead.

The highest-payoff work order I would enforce on the coding agent is: **remove per-Jacobi/per-iteration host synchronization → fast/coalesced non-Kahan GEMM → direct-Jacobi f32/WG A/B → device smearing+SCC control → cheaper metric repair → GPU energy/FIRE decisions → fused atomic-free forces → deprecate legacy production-looking APIs.**

That should improve speed while actually making the f32 numerics *more* controlled, rather than trading accuracy for performance.


---

# USER


Where I agree especially

"6.Move Fermi occupation completely onto the GPU."
sure, it is totally idiotic do this on CPU! I would go further - it should be merged into some kernel (e.g. endeity matrix build from MO coefs?) because it is small amout of calculations and can be amortized, not worth the overhead of running another kernel. Kernel exacution have singinficant overhead, so we want to keep number of kernels run each SCC loop at minimum. (write that desing goal in the manifest explicitly!)


"8. The FIRE loop is device-resident, but the adaptive controller still round-trips every step."
yes, this is idiotic! thotally defeates the point!!!


10.The force kernels should be the next numerical+performance cleanup after GEMM.
"More importantly, get rid of atomic force accumulation."

- EXACTLY!!!!! I insist you must not used fucking atomic aditions!!!! they are slow as hell in OpenCL !!!! we must alsyas stronly favourize gather approach wherever posible. We must desing the force-evaluation architecutre around ather architecture atom by atom (e.g. one workgroup per atom?) Not sure what is best layout?
I hate the coding agent still push fucking stupid atomic gather when everywhre in GPU optimiazation skills is intruction to avoid it and prefer GATHER! How syould I say it more clarly and more stronly that it never violates it!!!!


"11. Remove the remaining allocations from normal solver control."
FUCK!!!!!! This is for 10x I'm stressing we must tetally remove all allocation crap from inside SCC loop!!! why that fucking idiotic coding agent keep doing this! how can I stress it  more!!! this is for 20x stilla same crap!!!

"12. Eliminate the remaining architectural “fallback traps.” the same neverending stroyy !!!! pressing 20x already the idiot still do this crap!!!! I'mg really getting crazzy!!!


---

but I disagree at some points
"2.Move the SCC convergence controller onto the GPU. Right now you still synchronize CPU↔GPU once or more per iteration" - I agree we should not do this transfer every iteration, but I would rather check for convergece e.g. every 10 iterations, rather than impleemnt these check on GPU as they are anyways ineffcinet (parralle reduction) 

ad "Move final energy and residual reductions onto GPU, using f64 exactly where it is cheap." - not sure - energy itself is perhaps not work tho do on GPU, byt what is worth is computing of the forces in GpU, I think it would make sense to merge this Energy and Force clauclation into one kernel as it reuse singificant part of the machinery (it is bacially E=Tr[H*rho] and F=dE/dR )


---


Yes. I would sharpen the design rules substantially and make them **non-negotiable invariants of the dense production path**, not “optimization suggestions.”

Your two disagreements are also mostly right. I would **not** add a dedicated GPU convergence-control kernel just for the sake of avoiding a tiny host decision. And I agree energy should not drive another kernel launch if it can be obtained almost for free while doing the force contraction.

The current code still demonstrates exactly why these rules need to be written aggressively: Fermi smearing downloads eigenvalues, does 80 host bisection iterations and uploads occupations each SCC iteration; FIRE still transfers `4*batch` statistics to the CPU and sends controls back; force accumulation still uses CAS atomics; there are still `Vec` allocations in SCC/DIIS control; and failed SCC systems still have an automatic “nearest array index” recovery policy.

Here is the section I would put directly into the manifest.

## Dense GPU production path — hard architecture invariants

These rules apply to the homogeneous batched dense DFTB path used for H-bond / proton-transfer systems, especially `GpuDftb`, `GpuSccPlan`, SCC iterations, force evaluation, and FIRE relaxation.

They are **hard requirements**, not recommendations. A coding agent MUST NOT violate them merely because a simpler implementation is easier to write.

### 1. Minimize kernel launches as a primary design objective

Kernel launch and synchronization latency is significant for matrices of only N≈30–100. The steady-state SCC loop must therefore be designed as a small number of coarse kernels.

Every additional kernel launched inside one SCC iteration requires an explicit justification.

Prefer fusing operations when:

* they consume the same data,
* they naturally belong to the same workgroup/domain,
* fusion avoids writing and rereading intermediate global buffers,
* and fusion does not introduce a global synchronization requirement inside the kernel.

Do NOT create one small kernel for every algebraic operation.

Target steady-state SCC structure should be approximately:

`q → Hscc`
→ `BᵀH`
→ `(BᵀH)B`
→ `Jacobi + eigenvalue extraction + occupations`
→ `Mulliken/residual/DIIS`

approximately 5–6 launches, not 10–20.

Any proposed additional hot-loop kernel must explain why it cannot be fused with an existing stage.

### 2. Fermi occupation MUST stay on the GPU

It is forbidden to:

* download eigenvalues each SCC iteration,
* solve μ on the CPU,
* upload occupation weights again.

For the direct Jacobi solver, the preferred design is to calculate Fermi μ and `f_k` at the end of the Jacobi kernel, because that workgroup already owns the complete eigenvalue spectrum.

The extra work is tiny compared with the eigensolve and should be amortized inside that kernel.

Use a safeguarded Newton/bisection calculation of μ satisfying

$$
\sum_k f_k = N_{\rm occ},
\qquad
f_k = [1+\exp((\epsilon_k-\mu)/kT)]^{-1}.
$$

No additional occupation kernel should exist unless profiling demonstrates that fusion is worse.

Integer occupation selection should likewise be fused into the eigensolver tail where practical.

### 3. Do not build the full density matrix during SCC unless it is actually needed

During ordinary SCC iterations the purpose of the electronic solution is primarily to obtain Mulliken populations and hence the next charge vector.

The full AO density matrix D is required for final energy/forces, but it is not intrinsically required for every SCC iteration.

Prefer computing Mulliken populations directly from orbitals:

$$
q_A =
2\sum_{\mu\in A}\sum_k
f_k C_{\mu k}(SC)_{\mu k}.
$$

A specialized post-eigensolve kernel should investigate computing:

* Mulliken populations,
* charge residual,
* DIIS history/update,
* RMS residual,

without first materializing the complete N×N density matrix.

Build D and W only once for the final converged electronic state unless measurements prove the direct-population route slower.

Rayleigh-quotient repair is likewise normally needed for the final certified E/W state, not necessarily during every SCC iteration.

### 4. NO global atomic additions in production force evaluation

Global atomic force accumulation is forbidden in the dense production path.

In particular, code of the form

`atomic_add(force[atom], pair_force)`

must not be used.

The force architecture MUST be gather-based.

Preferred first implementation:

`one workgroup = one (replica, atom)`

with a static incident-pair adjacency list.

Each atom workgroup:

1. loops/gathers all interactions involving that atom,
2. evaluates their SK/SCC/repulsive force contribution,
3. accumulates the result privately/local,
4. writes exactly one final `float3` force for that atom.

No global force atomics.

For H-bond systems with ≈20–40 atoms, WG≈32 or 64 is a natural starting point: approximately one lane per neighbour, with warp/subgroup or local reduction.

This evaluates a physical pair twice, once from each endpoint. That duplication is ACCEPTABLE if it is faster than an extra kernel launch/global intermediate buffer. The canonical pair orientation and sign should make the two evaluations deterministic.

The alternative architecture to benchmark is:

`pair-evaluation kernel → pair_force[p]`
followed by
`atom-gather kernel → F_atom`.

This evaluates each pair only once but costs one extra launch and global pair-force traffic.

Benchmark these two gather architectures. Do NOT benchmark them against an atomic implementation and then retain atomics as a fallback.

### 5. Fuse all force terms before gathering

The pair/atom force evaluator should compute in one pass

$$
2\sum_{\mu\nu}
\left[
D_{\mu\nu}\partial_R H^0_{\mu\nu}
+
(\bar V D_{\mu\nu}-W_{\mu\nu})
\partial_R S_{\mu\nu}
\right]
$$

plus SCC γ′ and repulsive contributions.

Do not evaluate `dS/dR` once for non-SCC force and again in a second SCC-shift kernel.

Do not interpolate the same SK table twice for the same physical pair merely because the force formula was split historically into separate functions.

### 6. Energy should piggyback on force work, not create another hot-loop kernel

Energy itself is not performance-critical compared with forces.

When force evaluation already evaluates the pair H/S/repulsive interactions, it may accumulate per-atom or per-pair energy contributions at essentially no extra memory traffic.

For example an atom-gather force kernel may also emit

`e_atom[replica][atom]`.

The host may sum these small arrays only when an energy value is actually requested.

Do NOT add a dedicated energy kernel to every FIRE/SCC step merely to produce logging output.

Retain the Rayleigh/band-energy formulation as the numerical reference until the pairwise `Tr[D H0]` formulation is demonstrated to achieve equal accuracy.

### 7. NO heap/buffer allocation inside SCC, force, or FIRE hot loops

After `GpuDftb::new()` / `GpuSccPlan::new()` has completed, normal execution of:

* SCC iterations,
* `eval`,
* force evaluation,
* FIRE steps,
* geometry updates,

must perform ZERO dynamic GPU-buffer allocation and ZERO program/kernel construction.

Forbidden in the hot path:

* `Buffer::builder`
* `zero_buffer`
* `buffer_from_slice`
* `Kernel::builder`
* `build_program`
* creation of another OpenCL context or queue
* dimension-dependent `vec![...]`, `Vec::new`, or equivalent scratch allocation.

All GPU buffers, host staging arrays, masks, DIIS history, force scratch, occupation scratch, energy scratch and FIRE control buffers must be persistent members of the engine/plan.

Reset persistent buffers with device fill/reset kernels. Do not allocate a host zero vector merely to clear them.

Diagnostic functions may allocate, but they must be clearly outside the production hot path.

### 8. Add automated regression guards against hot-loop allocation/compilation

Do not rely on comments or agent discipline alone.

Add instrumentation/counters for:

* GPU buffer allocations,
* OpenCL program compilations/cache misses,
* kernel construction.

A regression test should snapshot these counters after engine construction, execute several SCC + force + FIRE steps, and assert that the counters have not increased.

Where practical, add a test-only Rust allocation counter around the steady-state solver loop as well.

An optimization that reintroduces allocation or compilation into the hot loop must fail a test.

### 9. NO implicit CPU fallback or hidden algorithm fallback

The dense GPU engine either runs the requested GPU algorithm or returns a clear status/error.

It must NOT silently:

* fall back to a CPU solver,
* switch to an old eigensolver,
* rebuild a different context,
* seed from another replica,
* change mixer,
* loosen a tolerance,
* change occupation physics.

Recovery strategies must be explicit higher-level operations.

For example:

`SCC failed for replica 7`

is a valid core-solver result.

The proton-transfer scan driver may then explicitly request

`retry replica 7 using replica 6 as continuation seed`.

The generic SCC solver must NOT assume that adjacent array indices are physically adjacent configurations.

### 10. CPU↔GPU synchronization is forbidden per SCC iteration

Do not read RMS/eigenvalues/occupations/charges after every iteration.

The residual is already computed as part of SCC/DIIS work; store it on device.

Run SCC in chunks, initially e.g. 8–10 iterations, then read only a compact O(batch) convergence/status array.

The exact chunk length should be benchmarked.

Do NOT add a separate expensive GPU convergence-reduction kernel merely to avoid this host check. Piggyback residual/RMS calculation on work that already has to be done for Mulliken/DIIS.

The objective is:

many GPU iterations
→ one small host status check,

not:

GPU iteration
→ host synchronization
→ GPU iteration
→ host synchronization.

### 11. FIRE control is fully device-resident

`dt`, `alpha`, `n_positive`, convergence/park status and velocity reset mode are persistent per-replica device state.

FIRE reduction and FIRE adaptation belong on the GPU.

It is forbidden to download `{P,v²,F²,maxF}` every FIRE step, calculate adaptation on the CPU and upload controls again.

Host intervention is needed only occasionally to inspect convergence or report progress.

### 12. Prefer gather everywhere

The general GPU rule for this project is:

**OWN OUTPUT → GATHER INPUTS → WRITE ONCE.**

Prefer one workgroup/thread owning the output entity:

* atom owns force,
* atom owns Mulliken population,
* matrix tile owns its output tile,
* replica owns SCC/DIIS state.

Avoid architectures where many work-items scatter into shared global outputs.

Global atomics require explicit USER approval.

### 13. Bulk arithmetic remains FP32; precision is spent only on cheap reductions/invariants

Do not solve FP32 limitations by converting O(N²) or O(N³) matrix arithmetic to FP64.

Use:

* FP32 storage,
* FP32 FMA,
* several independent accumulators instead of a long serial sum,
* explicit normalization,
* Rayleigh quotient correction,
* S-metric/polar correction,
* physically meaningful finite-T smearing.

Use FP64 only for tiny operations where the cost is negligible:

* DIIS small linear solve,
* final scalar energy accumulation,
* optional final atom-force gather,
* chemical-potential scalar solve if profiling shows it useful,
* convergence/diagnostic scalars.

Kahan compensation is NOT the default for GEMM.

For GEMM prefer multiple independent FP32 FMA accumulators and a balanced final sum.

### 14. Production and diagnostic APIs must be impossible to confuse

There must be one clearly documented dense production route:

`GpuDftb → GpuSccPlan`.

Deprecated tiled Jacobi, legacy `GpuDriver`, standalone `GpuMatrixContext`, CPU DIIS measurement paths and one-shot helper functions must be marked diagnostic/deprecated and must not be called from production code.

A future coding agent should not be able to accidentally select an old slow path because its function name still says “production entry point”.

That wording is deliberately much stronger. In particular I would put **“global atomics require explicit USER approval”** in the manifest. That is hard to misinterpret.

On your convergence point, I agree with you more than with my previous proposal. A separate GPU convergence-control layer could easily become another collection of tiny reductions/kernels. The better architecture is:

$$
\boxed{\text{residual calculation already inside post-SCC/DIIS work}}
$$

and then simply don't look at it every iteration.

For example, run 8 iterations completely asynchronously, read `rms[batch]`, and continue another 8 if necessary. If normal warm SCC convergence is around 6–15 iterations, perhaps a chunk of 4 or 5 eventually proves optimal rather than 10; that is a benchmark question. What matters architecturally is **no host synchronization per iteration**.

There is one useful refinement: inactive-replica detection need not happen between every iteration. Even if a replica converges at iteration 3 and continues uselessly to iteration 8, five unnecessary N≈87 solves may be cheaper than five CPU/GPU synchronization barriers. That trade should be measured instead of assuming that perfect early stopping is free.

### I would go even further on Fermi occupation

Your instinct to fuse it is right, but I would fuse it into the **eigensolver**, not the density kernel.

The direct Jacobi kernel ends with one WG owning the entire spectrum of one replica. At that point it already has all \(\epsilon_k\). Before returning, that same WG can:

$$
\epsilon_k
\rightarrow \mu
\rightarrow f_k
$$

and write `eig_diag[]` and `occ_w[]`.

That removes:

* `extract_diagonal`,
* `select_occupation`,
* eigenvalue readback,
* host μ bisection,
* occupation upload.

This is an exceptionally natural fusion because there is no inter-workgroup synchronization problem.

Then the subsequent electronic-state kernel simply consumes `C` and `f_k`.

In fact, I think there is an even larger architectural improvement hiding here: **do not construct D during SCC at all.**

The current sequence effectively does

$$
C \rightarrow D \rightarrow DS \rightarrow q_A.
$$

But Mulliken populations can be calculated directly:

$$
q_A=
2\sum_{\mu\in A,k}
f_k C_{\mu k}
\sum_\nu S_{\mu\nu}C_{\nu k}.
$$

For N≈87 and only ~29 atoms, this can be a dedicated gather calculation which directly produces `q_new`. Then the same WG/system can immediately calculate residual and run the tiny DIIS solve.

So the long-term SCC could become:

$$
q
\xrightarrow{1} H_{\rm SCC}
\xrightarrow{2,3} B^THB
\xrightarrow{4} {\rm Jacobi}+\mu+f
\xrightarrow{5} q_{\rm Mulliken}+{\rm DIIS}.
$$

**Five launches per SCC iteration.**

Full D and W are then constructed once after SCC converges, because they are needed for the force.

That, to me, is a much cleaner architecture than endlessly optimizing a density matrix that SCC does not fundamentally need.

### Force layout: I would start with one WG per atom

For your systems this is unusually attractive.

If GC has ~29 atoms and batch=19:

$$
29\times19=551\ {\rm workgroups}.
$$

That fills an RTX 3090 far better than one-WG-per-system.

I would start with WG=32. Each workgroup owns exactly one atom, and approximately one lane handles one other atom/pair. For ~30 atoms this maps beautifully onto a warp.

The static initialization creates something like:

```text
atom_pair_offset[a]
atom_pair_count[a]
atom_pair_index[]
atom_pair_sign[]
```

and all dynamic geometry comes directly from coordinates.

Each lane evaluates its incident pair contribution. Then reduce `float3` within the warp/workgroup and lane 0 writes:

```text
force[replica][atom]
```

exactly once.

No atomic. Ever.

The price is evaluating each physical pair twice. I would **deliberately accept that first**, because it buys:

* one force kernel,
* hundreds of independent workgroups,
* no global pair buffer,
* no second gather launch,
* no atomics,
* deterministic accumulation.

Then benchmark against the two-kernel alternative:

$$
{\rm pair}\rightarrow F_{ij}
\quad;\quad
{\rm atom\ gather}.
$$

If the expensive 4×4 SK derivative calculation makes the 2× duplicated arithmetic dominant, the two-kernel version may win. But both architectures obey the essential rule: **pair forces are gathered, never atomically scattered**.

Also, the current force kernel materializes large arrays of 4×4 H/S derivatives inside each work-item. That is likely to create huge register pressure or spills.  For a gather kernel I would directly contract the analytic expressions with D/W while they are generated. Do not construct:

```text
dh_dx[4][4]
dh_dy[4][4]
dh_dz[4][4]
ds_dx[4][4]
...
```

when ultimately only three accumulated force numbers are needed.

That could be a very large improvement.

### Energy + force

I also agree with your correction here.

I would **not build a special GPU energy pipeline**.

When the force gather already computes for every pair:

* SK H/S,
* D and W contractions,
* SCC shift,
* γ′,
* repulsive spline,

it is nearly free to accumulate corresponding pair energy terms.

For example the same atom gather can produce:

```text
force_atom[3]
energy_atom
```

where each atom gets half of every pair energy.

Then if FIRE does not need the energy, nobody reduces `energy_atom`.

If you want logging every tenth step, read ~30×batch energy numbers and sum them on CPU—or reduce only when requested.

The only caveat is numerical: your Rayleigh/band-energy route was introduced because raw f32 `Tr(DH0)` was measurably noisier. So I would keep **Rayleigh energy as the scientific reference** and A/B test the fused force-derived energy using f64 accumulation. If it achieves µHa agreement, great: it can replace the separate path. If not, energy remains an occasional diagnostic calculation and does not contaminate the hot force architecture.

So I think the architectural principle should be:

$$
\boxed{\text{FORCES determine the hot-loop design; energy piggybacks when convenient.}}
$$

not the reverse.

And for the repeated agent violations, prose clearly isn't enough anymore. I would add **tests that fail** if a production SCC/FIRE call increases counters for GPU allocations, program compilations or kernel construction. The agent can ignore a paragraph; it cannot merge code while `test_dense_hotloop_has_zero_alloc_and_zero_compile` fails.





---

# USER

OK we added proper benchmarking and speed measurement, and I think they clearly suggest there are siginficnat overheads still, can you comment on it? Like 
scc.occ_host       42.2 ms   62   0.68 ms   HOST bisection μ + occ_w upload ← BIGGEST
seems this should not take so much time definitely.

Suggest where to focus optimization efford now:

https://github.com/ProkopHapala/dftbplus/commit/b3c16f3f15cbb98e4860d9c6a60f4da345ebbdd7

---

# SWE-2


Good question — 0.3 s wall for 45 FIRE steps is the *host-observed* time; it hides where GPU work vs. host orchestration sits. Let me check what profiling options exist on this machine, then wire in a breakdown.

`nsys`/`ncu` are installed — `ncu` can profile OpenCL kernels per-name. Let me check nsys's OpenCL trace support and look at the relax loop structure for stage timers:

`nsys` doesn't trace OpenCL on this setup; `ncu` does profile OpenCL kernels (per-kernel GPU time + launch counts) but it's heavy — good for a short run. For the host-side stage breakdown I'll add an env-gated timer with guarded `queue.finish()` at section boundaries, exactly as you suggested. Let me look at the relax loop:

Plan: (a) env-gated stage profiler in `GpuRuntime` — `RUST_DFTB_PROF=1` inserts `queue.finish()` at section boundaries only; (b) per-kernel GPU times via `ncu` on a short run — no code changes needed for that. First, the runtime internals:

`read_buffer`/`finish` are the two sync funnels — I'll count them there. Adding an env-gated stage profiler to `GpuRuntime`:

Now add ticks. In `scc_step_diis_enq` (per-stage granularity inside a step) and the chunk loop:

Now ticks in the chunk loop, `check_jacobi`, `fire_step`, and `relax`:

Now the per-kernel GPU time via `ncu`. Check what `diag_step_cost.rhai` does — probably a short workload already:

One fix first — `tick` starts `None` until `prof_reset`, so ticks outside `relax` record 0. Initialize it at construction:

Also — reports only print at `relax` end. For ad-hoc scripts like `diag_step_cost`, let me print accumulated tables at script completion in `dftb_engine` when profiling is on. Check the engine registry:

Add a `prof_report` passthrough on `GpuDftb` and call it at script end:

Clear picture at batch=19 already: **`scc.occ` (0.67 ms/iter) and `scc.hscc` (0.65 ms/iter) now dominate** — the eigensolve is third. Now the per-kernel GPU breakdown via `ncu`:

ncu didn't intercept any kernels — likely GPU perf-counter permissions (`NVreg_RestrictProfilingToAdminUsers`) or it can't hook the OpenCL runtime. Checking:

GPU perf counters are admin-locked (`RmProfilingAdminOnly: 1`) — ncu can't inject. So our own stage marks are the profiler. To get per-kernel resolution I'll place ticks between the dependent kernel enqueues inside the hot functions (each tick = guarded finish, off by default). Look at `eigh_solve`, `occupation`, `enq_dq_v_hscc`, and `eval_forces_device`:

Adding finer ticks inside `eigh_solve` (GEMM vs Jacobi split), `occupation` (select/eig-readback/host-bisect split), and `eval_forces_device` (finalize vs force kernels):

Now we have a real breakdown. `RUST_DFTB_PROF=1` gives guarded `clFinish`-delimited stage timing (off by default — zero overhead in production). Hardware counters are admin-locked on this box (`RmProfilingAdminOnly: 1`), so `ncu`/`nsys` can't inject — the built-in stage marks are the profiler. Run: `RUST_DFTB_PROF=1 dftb_engine --script ...`

## Batch=19, N=86 — per-SCC-iteration kernel stage times (56 iters)

```
stage                total      n    per-call   what it is
────────────────────────────────────────────────────────────────
scc.occ_host       42.2 ms   62   0.68 ms   HOST bisection μ + occ_w upload ← BIGGEST
scc.hscc           39.9 ms   56   0.71 ms   fused Δq→V=γΔq→H_scc kernel
scc.jacobi         29.6 ms   62   0.48 ms   direct cyclic Jacobi (warm, ~2 sweeps)
fire.force_kernels  6.4 ms    5   1.29 ms   EDM + γ' + repulsive + reduce (per FIRE step)
scc.diis            4.0 ms   56   0.07 ms   DIIS solve+mix+convergence mark
scc.rayleigh        3.1 ms   56   0.06 ms   ρ=cᵀHc−ε on occ cols
scc.density         2.5 ms   56   0.04 ms   D = 2Σ f_k c cᵀ
scc.extract_diag    2.4 ms   62   0.04 ms   eig extraction
fire.set_geometry   1.7 ms    5   0.34 ms   Löwdin repair/Jacobi(S) per step
scc.mulliken        1.6 ms   56
scc.eigh_finish     1.4 ms   56
scc.gemm_th2        1.1 ms   62   0.018 ms  W2 GEMMs are already fast
scc.gemm_th         1.1 ms   62   0.018 ms
scc.select_occ      1.1 ms   62
scc.eig_read        0.5 ms   62   0.008 ms  eig_diag readback
scc.commit          0.4 ms   56
everything else    <0.1 ms each (commit, chunkend, check_jacobi, fire.*)
```

## Answers to your questions

**What does the 0.3 s buy?** 45 FIRE steps × (8-iter SCC + finalize + forces + assemble + FIRE update) for the batch=1 relax; at batch=19 a warm SCC call is ~6.2 ms, eval+forces ~0.9 ms, fire_step ~3.5 ms.

**Kernel execution vs harness:** with PROF on, finishes=1220/reads=226 for this script (production: ~100 reads + ~15 finishes — the profiler's own finishes inflate the total, so read the table *relatively*). The genuine per-iteration syncs left are `eig_read` (0.008 ms) and `chunkend` — cheap; the expensive "harness" part is actually **host compute**: the 80-iteration × 86-eig × 19-replica f64 bisection ≈ 0.68 ms/iter — bigger than Jacobi!

**Three real optimization targets, in order:**
1. **`occ_host` 0.68 ms/iter** — host bisection is pure W5: Fermi μ + f_k on GPU (bracketed bisection is trivially parallel, ~10 tiny kernel passes) — kills both `occ_host` and `eig_read`.
2. **`hscc` 0.71 ms/iter** — the fused dq→V→H_scc kernel: γΔq matvec over atom pairs. Worth inspecting occupancy/coalescing.
3. **`jacobi` 0.48 ms/iter** — already prec1/WG512-tuned; warm path is ~2 sweeps, near its floor.

Profiler options in Rust: for CPU-side there's `criterion`/`tracing`/`pprof` (CPU sampling only — useless for GPU). For GPU-side: vendor tools (`ncu` — blocked here), or **OpenCL event profiling** — create the queue with `QueueProfiling` and attach an event per enqueue; that's the "proper" per-kernel timer, needs a wrapper at each `enq()` call site (~40 sites). The guarded-`clFinish` marks give the same per-stage numbers today with 5 lines.

Want me to proceed to **W5 (GPU Fermi)** — it's now the measured #1 cost — or first drill into the `dq_v_hscc` kernel?






---

# Chat GPT 5.6 sol


Yes — these numbers are finally useful, and they point to a fairly clear order of attack. There is also one important catch in the agent’s interpretation:

**W4 “chunked SCC with no host sync” is not actually true for the smeared production path.** `scc_step_diis_enq()` calls `occupation()`, and for `kT>0` that function does a blocking `read_buffer(eig_diag)`, 80 CPU bisection iterations, and an `occ_w` upload **on every SCC iteration**. So the 8-iteration chunk is still being broken apart by the Fermi occupation path.

That is the first thing I would fix.

### Why `occ_host = 0.68 ms/iteration` is actually plausible

For your GC benchmark it does roughly

$$
19\times86\times80 \approx 1.31\times10^5
$$

**double-precision exponentials per SCC iteration**, plus another pass to build occupations, plus a device upload. The 80-step bisection is absurdly overaccurate for values ultimately stored in f32: it is trying to determine μ to vastly more precision than the eigenvalues themselves contain.

So 0.68 ms is not evidence that `exp()` is mysteriously slow. It is evidence that the algorithm is doing a completely inappropriate amount of serial CPU work.

More importantly, the cost is worse than the 0.68 ms number suggests because it **forces synchronization between every SCC iteration**, thereby destroying the benefit of W4 chunking.

I would implement the Fermi calculation **inside the eigensolver tail**, not as another GPU kernel.

At the end of `jacobi_cyclic_global_batched`, the workgroup already owns the complete spectrum. Have it directly produce:

$$
\epsilon_k,\qquad \mu,\qquad f_k.
$$

For N≈86 the simplest implementation may actually be best: lane 0 scans the 86 diagonal values and performs ~8–12 safeguarded Newton/bisection iterations in f32, stores μ in local memory, barrier once, then all lanes write `occ_w[k]`. That is only ~1000 exponentials per system, running independently across the 19 systems. There is no need for 80 iterations or f64 `exp`.

This also means that for the smeared path you can remove **four current stages at once**:

$$
\boxed{
\text{extract\_diag}
+
\text{select\_occ}
+
\text{eig readback}
+
\text{host Fermi}
}
$$

`select_occupation_batched` is currently launched even before the code checks `kT>0`, although a smeared density subsequently uses all orbitals anyway.

I would keep the CPU f64 Fermi solver solely as the scientific reference.

---

### The second result I find very suspicious is `scc.hscc = 0.71 ms`

The arithmetic involved is tiny. For GC it is approximately:

$$
29^2\approx840
$$

FMAs for \(\gamma\Delta q\), followed by only

$$
86^2=7396
$$

Hamiltonian elements/system.

The current kernel architecture explains the poor result: it launches **one 256-thread workgroup per system**. For batch 19 that means only 19 WGs for an 82-SM RTX 3090. Worse, during the \(\Delta q\) and \(\gamma\Delta q\) parts only ~29 threads have useful work. Then those same 19 WGs serially stride over all \(N^2\) H elements.

So I would **not optimize the γ matvec arithmetic**. The architecture is wrong for the N² part.

There is a very simple low-risk fix that still keeps **one kernel launch**:

For each replica launch perhaps 4–5 workgroups rather than one. Every WG redundantly computes the tiny \(\Delta q\) and \(V=\gamma\Delta q\) into its own local memory, then each WG handles a different strided chunk of H.

For batch=19:

$$
19\times4=76
$$

workgroups, almost exactly enough to fill the 3090.

The redundant \(\gamma\Delta q\) work is microscopic:

$$
4\times29^2
$$

FMAs/system. That is a much better trade than leaving 75% of the GPU unused.

Only one WG should write the global `dq`/`V`; all WGs can calculate identical local copies for their H work. No atomics and no inter-WG synchronization.

I would first benchmark that because it is a small surgical change.

The **better eventual architecture** is to remove `H_scc` materialization entirely. Since

$$
H^{SCC}_{ij}
=
H^0_{ij}
+\frac12 S_{ij}(V_i+V_j),
$$

the first warm-basis GEMM

$$
T=B^T H^{SCC}
$$

can synthesize the H element while loading the B operand tile. Then the current sequence

$$
H_{\rm SCC}\ {\rm kernel}
\rightarrow B^TH_{\rm SCC}
$$

becomes one specialized tiled projection kernel.

That removes:

* one kernel launch,
* the full Hscc global write,
* the subsequent Hscc global read.

Given that the GEMMs themselves are already reported at only ~0.018 ms, while materializing Hscc allegedly costs ~0.7 ms, this is exactly the sort of fusion worth doing.

---

### One important profiling caveat

The new profiler is good for finding gross architectural problems, but **the GPU stage timings are not true production kernel timings**.

`prof_tick()` performs `queue.finish()` after every stage.

That destroys the asynchronous queue behavior and measures:

> launch + isolated GPU execution + driver synchronization latency

rather than normal pipelined execution.

This is visible in your own numbers. Summing the per-iteration profiler stages gives roughly 2+ ms/iteration, while the non-profiled SCC timings are substantially lower.

So I trust `occ_host` strongly, because that blocking CPU work exists in production anyway. I regard `hscc=0.71 ms` as a very strong **warning**, but I would remeasure it with proper OpenCL event profiling after fixing Fermi.

You do not need `ncu`. Create the queue with `CL_QUEUE_PROFILING_ENABLE`, attach an event to the kernel enqueue, and later read `START/END` timestamps after one normal end-of-chunk synchronization. That preserves normal scheduling.

---

### My optimization order now

| Priority | Work                                               | Reason                                                      |
| -------- | -------------------------------------------------- | ----------------------------------------------------------- |
| **P0**   | **Fuse Fermi μ + occupations into Jacobi tail**    | Largest measured cost and currently destroys SCC chunking   |
| **P1**   | **Remeasure with OpenCL events**                   | Current `clFinish` profiler exaggerates GPU-stage latency   |
| **P2**   | **Fix Hscc layout: multiple WGs/system**           | Current one-WG/system is obviously underoccupying batch=19  |
| **P2b**  | Then test **Hscc-on-the-fly inside \(B^TH\)**      | Potentially eliminates the Hscc kernel/buffer entirely      |
| **P3**   | **Replace force atomics by gather architecture**   | Already 1.29 ms/FIRE step; will become dominant after P0/P2 |
| **P4**   | Remove remaining hot allocations + hidden recovery | Not well represented in the GPU stage profiler              |
| **P5**   | Jacobi micro-optimization                          | Already down to ~0.48 ms; not the current problem           |

I would **not touch Jacobi again yet**. W3 already found that `prec1/WG512` is the best measured compromise and that pure-f32 has a real accuracy loss at larger N.

---

### There are still allocation violations in this commit

Despite the new hard rules added in the same commit, `scc_mix_inner()` still allocates on every solve:

```text
hist
stagnant
done
q_in
q_out
res
q_f32
jacobi_ran
samples
```

and later allocates DIIS diagnostic vectors. Most amusingly, the host-DIIS arrays are allocated even when production `mix=0` never uses them.

`eval()` / `eval_forces_device()` also literally do:

```rust
&vec![1i32; self.batch]
```

when checking Jacobi status.

These are probably not your current 0.7-ms bottleneck, but they should simply disappear. Make every such array a persistent field.

And the code still contains the automatic:

> failed replica → copy charges from nearest array-index converged replica → retry

inside `GpuDftb::scc()`.

That directly contradicts the “NO implicit fallbacks” rule introduced by this very commit. The solver should return `Failed`. The **scan driver**, which actually understands that replica \(i\) and \(i+1\) represent neighboring reaction coordinates, may explicitly request a continuation seed.

So the agent has improved the written rules, but the production code still violates them.

---

### After P0/P2, the force path should become the main target

The current force stage still launches separate electronic/shift kernels per bucket, then γ and repulsive kernels, and the pair kernels scatter with atomics. That 1.29 ms/FIRE step is therefore unsurprising.

For your GC geometry I still favor starting with:

$$
\boxed{\text{one WG per (replica, atom)}}
$$

so batch 19 × 29 atoms gives about **551 workgroups**, which is excellent GPU parallelism.

Each WG owns one final `float3 F_A`, gathers all interactions involving atom A, reduces them locally, and writes once. No global atomics.

WG=32 is an obvious first test: essentially one lane per possible neighbor.

The alternative remains:

$$
\text{pair evaluation}
\rightarrow F_{pair}
\rightarrow \text{atom gather}
$$

which avoids evaluating each pair twice but costs another kernel and intermediate memory. Both are legitimate. **Atomic scatter is not.**

---

One more optimization emerges from these timings: `rayleigh = 0.06 ms/iteration` is being run every SCC iteration, but it does not participate in producing the next charge—the Fermi occupation has already been chosen from `eig_diag`, and D/Mulliken have already been built. Its purpose is to make the stored final state suitable for accurate E/W.

So I would calculate Rayleigh quotients **only once on the final accepted SCC state**, not 8 times while approaching it. That removes another launch from every iteration with no change to SCC convergence.

Ultimately I think your steady-state N≈86 smeared SCC can realistically collapse toward something like:

$$
\boxed{
B^TH(q)
\rightarrow (B^TH)B
\rightarrow
\text{Jacobi + Fermi}
\rightarrow
\text{charges + DIIS/commit}
}
$$

**four coarse launches per iteration**, with no CPU contact for an entire chunk.

That is the architecture I would optimize toward now.



---

# USER

ad "Why `occ_host = 0.68 ms/iteration` is actually plausible"
I'm not saying I do not thrust the measurement, I'm saying it shows we do something stupid, I'm not even sure what the bisection should do (should explain it) why we must to do it in double precission, and why so fucking hell many iterations. What I would assume that the matrix diagonalization (Jacobi?) should take by far most of the budget, and if anything else take more (or comparable) amout of time, it is bad desing, wrong setup/parametrs, simply inefficiency. We must press on performance hard, find the sweetpot between precission accuracy, remove all overhead and unnecessary work.

analyse it again with this mindest 


---

# Chat GPT 5.6 sol

Yes — with that mindset, I would interpret the benchmark much more aggressively:

> **For N≈86, batched GPU DFTB, Jacobi should be the dominant irreducible cost. If scalar occupations, SCC Hamiltonian assembly, bookkeeping, or host orchestration are comparable to Jacobi, assume bad architecture until proven otherwise.**

Warm Jacobi is already unusually cheap because it starts nearly diagonal, so I would allow the two projection GEMMs and other real \(O(N^3)\) work to become visible. But **Fermi occupation and \(H_{\rm SCC}\) assembly have no business costing more than the eigensolver.**

The current profile is therefore very useful precisely because it exposes two glaring inefficiencies.

## 1. `occ_host = 0.68 ms`: this should essentially disappear

First, what is the bisection actually doing?

With Fermi smearing,

$$
f_k(\mu)=\frac{1}{1+\exp[(\epsilon_k-\mu)/kT]} .
$$

You cannot simply occupy the lowest `n_occ` orbitals anymore. You choose the chemical potential \(\mu\) such that the total electron number remains correct:

$$
\sum_{k=1}^{N} f_k(\mu)=N_{\rm occ}.
$$

Then

$$
D=2\sum_k f_k C_kC_k^T.
$$

So there **is** one scalar nonlinear equation per system to solve. There is generally no analytic closed form for μ, because it depends on all eigenvalues.

Bisection was presumably chosen because \(N(\mu)\) is monotonic, so it is idiot-proof and guaranteed to converge.

That part is reasonable.

What is unreasonable is virtually everything about the present implementation.

The code performs **80 bisection steps**, in f64, on the CPU, over all 86 eigenvalues, separately for 19 systems, and then uploads `occ_w` back to the GPU.

That means per SCC iteration:

$$
19\times 80\times86
\approx 131\,000
$$

double-precision exponentials.

And this is being done to determine 19 scalar chemical potentials.

That is an absurd ratio.

### 80 iterations is numerically nonsensical

Suppose the initial μ bracket is even a very wide 1 Ha.

After \(m\) bisections its width is:

$$
\Delta\mu=\frac{1}{2^m}.
$$

So approximately:

* 10 iterations → \(10^{-3}\) Ha
* 20 → \(10^{-6}\) Ha
* 24 → \(6\times10^{-8}\) Ha
* 30 → \(10^{-9}\) Ha
* 53 → around f64 mantissa resolution
* 80 → \(8\times10^{-25}\) Ha

The eigenvalues feeding this calculation are **f32**.

Therefore after roughly 20–25 bisections you are already trying to determine μ more accurately than the input spectrum contains.

And after ~53 iterations, many additional f64 bisections literally cannot even generate a distinct midpoint.

So 80 is classic “iterate harder because more precision sounds safer.” It directly violates the numerical philosophy we established.

### Nor does μ need f64 exponentials

The physical reason for finite-T occupation is precisely that occupation becomes **smooth** instead of making a discontinuous HOMO/LUMO decision.

This is therefore one of the places where f32 is particularly safe.

I would implement:

$$
N(\mu)=\sum_k f_k
$$

and

$$
N'(\mu)=\frac{1}{kT}\sum_k f_k(1-f_k)
$$

and use safeguarded Newton:

$$
\mu_{n+1}
=
\mu_n+\frac{N_{\rm occ}-N(\mu_n)}{N'(\mu_n)}.
$$

Keep a bracket around μ; if Newton leaves the bracket, take its midpoint.

Because SCC is warm-started, **also warm-start μ from the preceding SCC iteration**.

I would expect something like 1–4 Newton steps normally, perhaps max 6–8.

Not 80.

And there is another easy optimization: for

$$
x=(\epsilon-\mu)/kT,
$$

if \(x>20\), then \(f\) is effectively zero in f32; if \(x<-20\), it is effectively one.

So don't even evaluate `exp()` for deep occupied/virtual states. Only a handful of frontier states normally need an exponential.

### Precision sweet spot

I would A/B just two versions:

**Fermi-F32**

* eigenvalues f32
* μ f32
* `exp(float)`
* four FP32 accumulators for \(N\) and \(N'\)

versus

**Fermi-hybrid**

* `exp(float)`
* occupations f32
* accumulate \(N,N'\) and update μ in f64.

The second costs almost nothing because only a few scalar additions/divisions are double; crucially, **do not call double `exp()`**.

I strongly expect both to give indistinguishable PES/charge results.

### And fuse it into Jacobi

Do not make another Fermi kernel.

The Jacobi WG already owns exactly the data required: one complete spectrum of one replica.

At the end:

```text
Jacobi
   ↓
diag ε
   ↓
solve μ
   ↓
write occ_w
```

all before returning.

That also lets you delete the separate `extract_diagonal` launch for the smeared production path.

And the current code does something even sillier: it **runs integer `select_occupation_batched` before checking `kT>0`**, even though the smeared density ignores the integer occupation set and uses all orbitals.

So the current smeared path is:

```text
Jacobi
extract diagonal
integer bitonic sort
read eigenvalues to CPU
80× f64 Fermi solve
upload weights
```

when it should be:

```text
Jacobi + Fermi
```

That should be a P0 rewrite.

---

# 2. `scc.hscc = 0.71 ms` is even more suspicious

Look at how little arithmetic this operation contains.

For GC roughly:

$$
N_a\approx29,\quad N=86.
$$

Per system:

$$
\gamma\Delta q:
29^2 = 841
$$

multiply-adds, and

$$
H_{ij}^{SCC}
=
H^0_{ij}+\frac12S_{ij}(V_i+V_j)
$$

for

$$
86^2=7396
$$

matrix elements.

Across batch 19 that's only about 140,000 H elements.

The kernel reads approximately H0 + S and writes H:

$$
3\times19\times86^2\times4
\approx 1.7\ {\rm MB}.
$$

If 0.71 ms were really GPU execution time, that's roughly:

$$
2.4\ {\rm GB/s}.
$$

On an RTX 3090.

That is **ridiculous**. Even obtaining merely 10% of peak bandwidth should push this into a few tens of microseconds.

So I would assume one or more architectural problems.

## The present launch geometry is poor

The kernel is:

```text
one WG = one system
WG = 256
```

Therefore batch 19 launches only:

$$
19\ {\rm WGs}
$$

onto an 82-SM GPU.

And in the first two phases only 29 of the 256 lanes actually do useful work:

```text
dq[29]
γ·dq[29]
```

Then those same 19 WGs stride through the \(86^2\) Hamiltonian.

This is a particularly poor match to a 3090.

It gets even worse as the `active` mask shuts systems down: near convergence perhaps only 5–10 workgroups actually do anything.

### Immediate experiment

Do not split this into multiple kernels.

Instead launch perhaps:

```text
4 or 8 WGs / system
WG = 64 or 128
```

Each WG redundantly computes the tiny 29-element `dq` and `V` into its own local memory.

Then each WG owns a slice of the H matrix.

For 19 systems × 4 WGs:

$$
76\ {\rm WGs},
$$

almost exactly the 82 SMs of the 3090.

The redundant work is laughably small compared with the cost of poor occupancy.

Benchmark:

```text
groups/system = 1,2,4,8
WG            = 64,128,256
```

I would bet heavily that `1×256` is nowhere near optimum.

---

# 3. But there may be a second reason `hscc` looks so terrible: host OpenCL overhead

The profiler currently works by doing a `queue.finish()` after every stage.

And `scc.hscc` includes not only the kernel itself but also the host orchestration immediately before it.

Look at `enq_dq_v_hscc()`:

```rust
set_arg(q_gpu)
set_arg(q0)
set_arg(g)
set_arg(h0)
set_arg(s)
set_arg(orb_atom)
enq()
```

**every SCC iteration.**

But none of those buffer objects changes between SCC iterations!

Their **contents** change, but OpenCL kernel arguments hold buffer handles. You do not have to bind the same buffer handle again because its contents were updated.

So this should basically be:

```rust
unsafe { self.k_dq_v_hscc.enq()?; }
```

Nothing else.

The same pattern exists elsewhere: repeated `set_arg()` calls bind persistent C/D/S/etc. buffers in density, Mulliken, Rayleigh and occupation code.

That may be a surprisingly large part of what your stage profiler is calling `scc.hscc`.

So I would make another hard rule:

> **A steady-state SCC iteration should preferably contain only kernel enqueues. `clSetKernelArg` for invariant arguments is initialization/reconfiguration work, not iteration work.**

Prebuild separate handles where necessary:

```text
k_project_warm
k_project_cold
k_density_integer
k_density_fermi
```

instead of modifying a generic kernel object repeatedly.

Code duplication of **kernel handles** is cheap. Host API traffic every iteration is not.

---

# 4. Before optimizing more, get proper GPU timings

The current `clFinish` profiler was a very good diagnostic instrument; it revealed the problems.

But now we need to distinguish:

$$
T_{\rm stage}
=
T_{\rm host\ setup}
+
T_{\rm enqueue}
+
T_{\rm kernel}
+
T_{\rm synchronization}.
$$

Currently these are lumped together.

Enable OpenCL queue profiling and collect events.

Then report two numbers:

```text
host enqueue time
GPU event time
```

for every important kernel.

Importantly, **do not wait for the event immediately**.

Enqueue normal chunks exactly as production does; after the chunk's normal synchronization, inspect all recorded event timestamps.

Then we'll know, for example:

```text
hscc:
  host set_arg+enqueue = 0.45 ms
  GPU event            = 0.04 ms
```

or conversely:

```text
host = 0.02 ms
GPU  = 0.62 ms
```

Those imply completely different fixes.

Given the amount of arithmetic, I would consider **GPU `hscc` >0.1 ms suspicious**.

---

# 5. I would also remove Rayleigh from every SCC iteration

Current production SCC computes Rayleigh quotients every iteration.

But they are not used to generate the next q.

The order is essentially:

```text
eigenvectors
occupations
D
Mulliken q_new
Rayleigh ρ
DIIS
```

`ρ` exists to make final:

* band energy,
* energy-weighted density W,

numerically accurate.

Excellent idea.

But SCC iteration \(n\rightarrow n+1\) doesn't need it.

So do it **once after the final accepted SCC state**.

That takes:

```text
scc.rayleigh 0.06 ms/iteration
```

to approximately zero in the iterative loop and removes another launch.

The original Rayleigh correction was numerically important; I'm not suggesting removing it. I'm suggesting putting expensive correction **where its result is consumed**.

---

# 6. `commit` should not be a kernel either

The DIIS kernel now already calculates RMS and knows `rms_tol`. It even changes the active flag itself.

Therefore it knows whether the current electronic state has converged.

If converged:

```text
leave q_gpu = q_n
```

so the cached electronic state remains consistent.

If not converged:

```text
write mixed q directly into q_gpu
```

for the next iteration.

There is no fundamental reason to write:

```text
DIIS → q_next
```

and then launch:

```text
commit_q_next → q_gpu
```

as a separate kernel.

The commit model is conceptually correct. The extra kernel is not required to implement it.

So fuse commit into DIIS.

That removes:

* `q_next`,
* one global write/read,
* one kernel launch/iteration.

---

# 7. Ultimately I would remove `H_scc` from ordinary SCC iterations entirely

The current warm solve is

$$
H_{\rm SCC}
\rightarrow
B^TH_{\rm SCC}
\rightarrow
(B^TH_{\rm SCC})B.
$$

But

$$
H^{SCC}_{ij}
=
H^0_{ij}
+\frac12S_{ij}(V_A+V_B).
$$

So the first projection kernel can generate H elements **while loading its H tile**.

Thus:

```text
dq → V + Hscc
Bᵀ × Hscc
```

becomes:

```text
Bᵀ × Hscc(q)       // H synthesized on load
```

one kernel.

You avoid materializing and round-tripping an entire N² matrix.

The fused projection WG independently computes the tiny \(\gamma\Delta q\) vector and then proceeds with tiled matrix multiplication.

This is the architecture I would ultimately prefer.

And once Rayleigh is final-only, there is even less justification for keeping Hscc as a persistent per-iteration matrix.

---

# 8. Then the ideal SCC loop becomes much smaller

Current smeared production loop is roughly:

```text
1 Hscc
2 GEMM BᵀH
3 GEMM THB
4 Jacobi
5 extract diag
6 integer occupation sort      <-- useless with smearing
7 host Fermi read/compute/write
8 S-normalize
9 density
10 Mulliken
11 Rayleigh
12 DIIS
13 commit
```

That is far too much orchestration.

First realistic target:

```text
1 Hscc
2 GEMM BᵀH
3 GEMM THB
4 Jacobi + Fermi
5 S-normalize
6 density
7 Mulliken
8 DIIS + commit
```

Then second-stage architecture:

```text
1 Bᵀ Hscc(q)       # fused dq/V/H generation + first projection
2 T B
3 Jacobi + Fermi
4 S-normalize
5 charge build + DIIS + commit
```

**Five launches per SCC iteration.**

Potentially four if S-normalization can safely be done less frequently or incorporated elsewhere, but I would not compromise the known numerical benefit just to hit an arbitrary kernel count.

---

# 9. What should the performance budget look like?

I agree with your desired mental model.

For this N≈86 workload, after obvious inefficiencies are removed, I would like something roughly like:

```text
Jacobi                        60–80 %
projection matrix algebra     10–25 %
everything else combined      <20 %
```

Not:

```text
Fermi bookkeeping  > Jacobi
Hscc assembly       > Jacobi
Jacobi
...
```

Those first two are unmistakable red flags.

I would set optimization acceptance goals like:

* **Fermi occupation incremental cost:** <5% of Jacobi.
* **No CPU contact inside SCC chunks.**
* **Hscc generation:** <10–20% of Jacobi; preferably eliminated by projection fusion.
* **No repeated invariant `set_arg` calls.**
* **No allocations in SCC.**
* **≤5–6 coarse launches/iteration initially.**
* **Rayleigh once/final state, not every iteration.**
* **No commit kernel.**

Only when that is achieved would I return to Jacobi itself.

---

# 10. Jacobi may then expose the next fundamental bottleneck

The W3 result already says `prec1/WG512` is currently the sensible point: pure f32 saved ~30% but produced unacceptable eigenvalue error at N=128.

So I would leave that alone for now.

Once all the nonsense around it is gone and Jacobi genuinely consumes, say, 70% of iteration time, **then** optimize Jacobi again.

At that point the interesting issue will be structural: one WG/system means batch 19 can only occupy at most 19 of 82 SMs. WG512 makes each active SM busy, but most SMs remain unused.

Then the choices become much more substantive:

* run larger batches,
* reduce Jacobi to one warm sweep more often,
* develop a more aggressively warm/preconditioned eigensolver,
* or eventually consider multi-WG/system.

But it is premature to tackle a difficult eigensolver redesign while **131,000 CPU double exponentials** and a ~140k-element H update are each costing more than the eigensolver.

So my work order now would be:

**1. Kill CPU Fermi completely and fuse it into Jacobi.
2. Add true OpenCL event timings + host enqueue timings.
3. Prebind all invariant kernel arguments.
4. Fix Hscc occupancy immediately; then fuse Hscc into the first projection.
5. Move Rayleigh to final-state certification only.
6. Fuse DIIS + commit.
7. Then attack atomic force evaluation / FIRE.
8. Only then touch Jacobi again.**

That should push the solver toward the performance shape you expect: **diagonalization becomes the obviously dominant cost, rather than being buried under orchestration and tiny scalar work.**







---

# SWE-2

## What landed this session

| Item | Result |
|---|---|
| **Jacobi prologue fusion** | 3 n²-passes + 2 barrier chains → 1 pass + 1 packed reduce; dead off-diag zero pass removed (nothing reads `hp` off-diagonal). Warm SCC at batch 256: **2.47 → 1.81 ms (−27%)**; jacobi dev/call 2.71→1.84 ms |
| **Distinct-geometry batch 256** | 256 *different* GC geometries (H13 across d=0.8–2.4 Å): cold SCC needed a warm-start retry (rep 204←203), 88 transient DIIS fallbacks recovered, all converged; warm 8 iters @ **0.9 µs/iter/system** |
| **e_atom (I10)** | gather emits onsite+½pair band energy; `Σ e_atom` = Tr(D·H0) to **2.4e-7 Ha** (new test) |
| **md_step → real velocity-Verlet** | was full-kick (not VV); now drift with stored a(t) + ½(a+a′) kick, one eval/step |

46/46 tests pass. Scan E(r) and barrier unchanged (19.7 kcal/mol @ d=1.35).

## Detailed performance (RTX 3090, evt, N=86)

**19-pt relaxed scan:** 1.093 s, 57.5 ms/pt, 143 FIRE steps, 2068 syncs.

| stage | dev ms/call | dev% |
|---|---|---|
| scc.jacobi | 0.458 | 58.6% |
| scc.density | 0.053 | 6.0% |
| relax.eval | 0.366 | 4.8% |
| fire.set_geometry | 0.326 | 4.2% |
| scc.diis | 0.034 | 3.9% |
| fire.force_kernels | 0.301 | 3.9% |
| gemm_th+th2 | 0.036 | 4.6% |
| scc.eigh_finish | 0.022 | 2.5% |
| everything else | — | <14% |

At batch 19 jacobi is **latency-bound** (19 WGs don't fill the 3090; cost is barrier depth + launch). At batch ≥256 it's **bandwidth/reduction-bound** — that's where the fused prologue paid off. Remaining Jacobi fixed cost at nsw=0: the **Fermi tail** (≤24 Newton iters, each a fused f64 WG-reduce of ~9 barrier rounds). Next candidates: reduce over ⌈n⌉=128 lanes instead of 512, or subgroup/warp reductions.

## Accuracy analysis — the f32 floor, explained properly

The claim "energy wanders ~1e-6 Ha across tolerances" unpacks to this chain:

1. **SCC sets the charge error.** δE via charges ≈ ‖∂E/∂q‖·δq. Converging rms from 1e-5→1e-7 genuinely reduces the charge-driven energy error.
2. **But the energy is *evaluated* from f32 eigenpairs.** Even with exactly converged q, `E_band = Tr(D·H0)` is built from `c` (f32 eigenvectors, element error ~ε_f32·κ ~ 1e-7) and `D = 2Σf ccᵀ` accumulated in f32. That injects ~1e-6·|E|-scale noise **independent of how well SCC converged**.
3. **The eigensolve itself can't do better.** Jacobi stops when off-diag ≤ ~n·eps·‖A‖ ≈ 3.5e-7 relative — the rotation arithmetic in f32 can't push below that. The measured eig residual ‖HC−SCε‖ ≈ 1e-6 is this floor.
4. **So tightening tol below ~1e-5 reshuffles noise instead of reducing error** — observed E scatter across tol = {1e-5,1e-6,1e-7} was ~1e-6 Ha, i.e. eigen-noise dominates charge-convergence improvements.

In units that matter: 1e-6 Ha = **0.0006 kcal/mol** absolute, and the error that counts for scans is the *relative* error along the coordinate — biases that are geometry-smooth largely cancel in ΔE. Force parity 5e-6 Ha/B translates to relaxed-geometry error ~1e-4 Å. For the stated target this floor is already fine.

## Hybrid-precision brainstorm (not implemented)

Where the ~1e-6 Ha floor actually lives, and the cheapest fixes per source:

**A. `H' = BᵀH_sccB` formation in f32 (my bet for the dominant term).** Each element is an n=86 dot product accumulated in f32 → error ~ε·√n·‖H‖ ~ 1e-6·‖H‖, which lands *directly* on the eigenvalues. **Double-single (df32) accumulation** in just these two projection GEMMs — Dekker TwoSum via `fma` in the K-loop (~6–10 flops/term vs 1) — would push H' error to ~1e-9 while touching only ~5% of dev time. Highest leverage idea.

**B. Eigenvalue storage in Jacobi.** Diagonal updates `a_pp += t·a_pq` round at ε·|a_pp| every rotation — keep the **diagonal in df32/f64** (n extra words/replica; each rotation touches only 2 diagonal elements → trivial cost), off-diagonal stays f32. Gets ε_k toward 1e-8.

**C. One f64 Rayleigh-quotient pass at finalize.** ρ_k = cᵀHc in f64 on the n_occ occupied columns only, once per solve (not per iteration) — replaces the ε that enters the energy. Cost ~n_occ·n² f64 ≈ noticeable at 1/64 f64 rate but only once per solve; df32 variant is cheaper.

**D. Compensated density build.** `D = 2Σf ccᵀ` — same df32-trick in the k-loop. D-error ~ε·n_occ feeds both energy and forces.

**E. What's already covered / not worth it:** rotation-angle precision is *not* the limiter (PREC=1 A/B showed ΔE ≤ 2e-6 between f32/f64 rotations — R8a); Kahan-in-GEMM was already measured to not cut δ_CH; energy scalar tails and DIIS solve are already f64.

Suggested order if you want it: **A → B → C**, each as a measured A/B against the CPU reference. Expected end state ≈ 1e-7–1e-8 Ha — roughly 10× below today's floor for maybe +10–15% dev time. Whether that's worth doing depends on whether 0.0006 kcal/mol noise is acceptable for your scan physics — for the H-bond scans shown, it already is.