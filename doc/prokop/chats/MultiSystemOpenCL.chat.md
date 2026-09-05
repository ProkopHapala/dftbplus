https://chatgpt.com/share/6a9bdb9d-a818-83eb-8d81-a5fe00fa7501

---

### USER


I want to do rust+OpenCL reimplementation of DFTB+ which efficiently use GPU to efficietly solve many small quantum system (like <100 atoms each) using modern GPU with thousands of threads.

I was thinking to use synchonous apporach which we previously used for classical forcefield. Each system is one workgroup we layour nd-kernel and uniform memory buffers where hamiltonina, density matrix and molecualr obrital coefs of each system are stacked and can be indexed separately ... review the proposed apporach and what we already implemented and suggest if we can do it bettwe, what works what does not make sense... what sohouold b improved and how, we are still in a bit brainsotrming phase so do ot worry to propose fundamental architectural changes if needed. Identify major chalanges On I can see if every sub-system require different numbe of SCC-steps or MD-steps to converge, thant this approach is a bit wastefull as all the workunits will wait for the lowest.

I was thinking about alternative apporoach where we execute independnte kernels for the sub-systems but nI'm not sure how GPU scheduler is able to layout these parallel run of independnet kernels and what is the overhead, but we should probably test it how it performans wrt synchronous uniform-multi-system kernel approach.

https://github.com/ProkopHapala/dftbplus/tree/main/rust_dftb

---

### Chat GPT 5.6 sol

Yes. After reading the design note and the actual `rust_dftb` code, I think the **batched-many-small-systems idea is fundamentally the right direction**, but I would change one central abstraction:

> **Do not make “1 workgroup = 1 quantum system” the architecture. Make “system = batch index” the architecture. Each kernel is then free to use 1, several, or many workgroups per system.**

That sounds like a small distinction, but it resolves several contradictions in the present design.

Your design already notices the problem: H assembly, reductions, and local Jacobi naturally like one WG/system, whereas tiled GEMM and a serious eigensolver for larger matrices naturally require many WGs/system.  The later D7 discussion then runs into exactly this contradiction. 

## 1. The architecture I would aim for

I would organize it roughly like this:

```text
GPU
│
├── SystemClass 0: 1000 × H2O      Natom=3,  Norb=6/...
│     ├── coords[system][atom][3]
│     ├── H0[system][N][N]
│     ├── S [system][N][N]
│     ├── X [system][N][N]       S^-1/2
│     ├── G [system][Nat][Nat]    gamma matrix
│     ├── q [system][Nat]
│     ├── C/D ...
│     └── active[system]
│
├── SystemClass 1: 200 × molecule  Norb=64
│     ...
│
└── SystemClass 2: 30 × molecule   Norb=384
      ...
```

A `SystemClass` should have homogeneous matrix dimensions, ideally homogeneous topology/species as well. Then individual kernels choose their natural parallelization:

| operation                     | natural GPU mapping                     |
| ----------------------------- | --------------------------------------- |
| SK H0/S assembly              | 1 WG/system **or** `(system,pair-tile)` |
| gamma \(G_{AB}\) construction | 1 WG/system or tiled                    |
| SCC \(V=G\Delta q\)           | 1 WG/system                             |
| H-SCC update                  | matrix elements × systems               |
| GEMM                          | **many WGs/system**, one WG/output tile |
| tiny Jacobi                   | subgroup or 1 WG/system                 |
| large block Jacobi            | many WGs/system                         |
| Mulliken population           | 1–several WGs/system                    |
| residual/mixer                | 1 WG/system                             |

Your existing `batched_gemm` already follows this philosophy: system is `get_group_id(2)`, while each matrix has many tile workgroups.

So I would **embrace the currently “inconsistent” mappings rather than try to eliminate them**.

---

# 2. There is a missing third option between “host SCC” and “mega-kernel”

The design currently contrasts:

1. host-driven SCC with CPU readbacks, versus
2. a gigantic kernel doing SCC internally. 

I think the best main architecture is actually:

### Host-orchestrated, but completely device-resident

The Rust host simply enqueues:

```text
gamma_matvec
H_scc_update
orthogonalize / transform
diagonalize
Mulliken
residual
mixer
```

No charge matrices go to the CPU. No `queue.finish()`. No PCIe transfer. The commands simply form a dependency chain on one OpenCL queue.

That distinction is important: **host orchestration does not imply host computation or host-device synchronization**.

Kernel boundaries give you exactly the global synchronization that OpenCL workgroups cannot provide internally. So there is no need to contort the whole algorithm into one workgroup just to obtain barriers.

Later, if kernel-launch overhead actually becomes significant, `cl_khr_command_buffer` exists precisely to record and replay a sequence of commands, and mutable dispatch can modify arguments/sizes between executions. ([Khronos Registry][1])

I would treat the SCC mega-kernel as a **special fast path for extremely small systems**, not the main design.

---

# 3. Your concern about different SCC iteration counts is real — but slightly different from how you describe it

Suppose you really have:

```c
kernel solve_systems(...) {
    system = get_group_id(0);

    while (!converged(system)) {
        ...
    }
}
```

with one WG/system.

The workgroups **do not synchronize with each other**. Water #17 can finish after 6 SCC iterations while water #42 keeps going to 25. The completed WG frees its GPU resources. OpenCL workgroups from an NDRange can execute independently/concurrently. ([Khronos Registry][2])

So they do **not** literally all wait for the slowest system.

What happens instead is a **tail-utilization problem**:

```text
iteration

1   ████████████████████████████████████
5   ███████████████████████████████████
10  ████████████████████
15  ████████
20  ███
25  █
```

At iteration 25 only one system may remain, so most SMs are idle. The kernel itself cannot finish until that final workgroup finishes.

For thousands of systems this is often not too bad because, initially, many more WGs exist than can be resident simultaneously. Early finishers make room for queued WGs. The ugly tail appears only near the end.

But for your general `<100 atom` target the larger issue is that **the eigensolver cannot generally live in one WG anyway**.

So I would solve convergence divergence using an **active mask**:

```text
active[system] = 1

each SCC kernel:
    sid = batch_system
    if (!active[sid]) return;

...

residual kernel:
    if (rms[sid] < tol)
        active[sid] = 0;
```

Then converged systems cost essentially an early-return WG/tile on subsequent SCC iterations.

No readback is necessary.

Only if the active population becomes extremely sparse would I bother compacting `active_ids[]`.

---

# 4. The harder divergence is actually independent MD/relaxation trajectories

Here your concern becomes more serious.

Imagine 100 independent MD trajectories:

```text
trajectory A: SCC converges in 5 steps -> ready for MD step 301
trajectory B: SCC needs 30 steps
```

If the entire ensemble uses one global SCC phase, A cannot proceed to MD step 301 until B has converged.

For **NEB this is basically fine** because all images need to meet again to calculate tangents/spring forces anyway.

For a rigid scan it is also irrelevant: each point just finishes once.

For **independent MD, relaxed scans, independent geometry optimization**, I would use **microbatches**:

```text
1000 trajectories

batch 0:  0..63
batch 1: 64..127
batch 2: ...
```

Each microbatch has enough systems to saturate the GPU, but it advances independently.

Then perhaps 2–4 OpenCL queues execute different microbatches asynchronously. A pathological system stalls only its 64-system group, not all 1000.

This is where your experiment with independent kernel streams becomes very worthwhile — but I would compare **microbatch-per-queue**, not merely system-per-queue.

---

# 5. About launching one kernel per individual system

Your current `GpuMatrixContext` creates an ordinary OpenCL queue with default properties, so it is an **in-order queue**.  OpenCL explicitly provides `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`; without it, commands execute in order. Multiple queues can execute independently. ([Khronos Registry][3])

On NVIDIA hardware, work from different streams/work queues can in fact execute concurrently if resources are available. Concurrent kernels may contend for/share SM resources. ([NVIDIA Docs][4])

So I would soften one statement in the design note:

> “The hardware scheduler does NOT intelligently pack small kernels from different queues onto the same SM.”

That is too categorical.

It **can** overlap kernels and exploit otherwise unused resources. What you cannot rely upon is *how well* it will do so. Register use, local memory, number of blocks, work queues, and kernel geometry all matter.

But I still expect:

```text
one batched kernel with 1000 systems
```

to beat

```text
1000 individual kernel launches
```

quite badly for small systems.

The interesting competition is more likely:

```text
1 × batch of 1000
vs
16 × batch of ~64
vs
32 × batch of ~32 on 2-4 queues
```

rather than batch-vs-1000-single-launches.

And I would **measure launch overhead instead of baking “5–20 µs” into the design**. It depends considerably on driver, GPU, Rust wrapper, kernel creation, profiling mode, etc.

---

# 6. More important: H0 and S should not be rebuilt during SCC

This is probably the largest algorithmic architectural change I would make.

Your current GPU `assemble_pairs` calculates SK interpolation/rotation, writes S, and adds the SCC electrostatic correction using `V_a`.

But during one geometry step:

$$
H^0(\mathbf R),\quad S(\mathbf R)
$$

are fixed.

Only

$$
\Delta q_A
$$

changes during SCC.

So structure the timestep as:

### Once per geometry

$$
H^0(\mathbf R)
$$

$$
S(\mathbf R)
$$

$$
G_{AB}=\gamma_{AB}(R_{AB})
$$

and the overlap orthogonalizer

$$
X=S^{-1/2}
$$

are constructed.

### Every SCC iteration

First:

$$
V_A = \sum_B G_{AB}\Delta q_B
$$

Then

$$
H_{\mu\nu}^{SCC}
=
H^0_{\mu\nu}
+
\frac12S_{\mu\nu}(V_{A_\mu}+V_{A_\nu})
$$

which is an extremely cheap dense elementwise kernel.

Then diagonalize → populations → mix.

Your CPU implementation already has exactly this conceptual separation: `H0`/`S` live in the template and `build_h_scc()` applies the SCC shift separately.

The GPU path should copy this architecture.

This also means the current species-pair bucket H-assembly does **not need to be redesigned to support SCC synchronization at all**. Use whatever H0/S assembly mapping benchmarks fastest, then leave it behind for the entire SCC loop.

---

# 7. Precompute gamma too

Currently `onsite_and_va` evaluates `gamma_full()` including exponentials during SCC.

For fewer than 100 atoms I would simply store:

$$
G_{AB}
$$

as a dense \(N_{\rm atom}\times N_{\rm atom}\) matrix.

100 atoms means:

$$
100^2\times4\ {\rm bytes}=40\ {\rm kB/system}.
$$

Tiny compared with orbital matrices.

Then an SCC iteration uses a cheap batched matvec.

This removes a surprising amount of expensive transcendental computation from the inner SCC loop.

---

# 8. Geometry preprocessing should move onto the GPU

`GpuBatch::from_fragments()` currently creates geometry-derived `PairEntry`s containing

```text
r, l, m, n
```

on the CPU and constructs gamma-neighbor distances there as well.

That is reasonable for a first static test, but it is exactly what you don't want in MD.

For a homogeneous template, upload only static information once:

```text
pair_i[]
pair_j[]
species pair
orbital offsets
block type
```

and current

```text
coords[system][atom][3]
```

each geometry step.

The GPU calculates \(r,l,m,n\).

For <100 atoms I would initially not even build a neighbor list. At worst you have

$$
100(99)/2=4950
$$

atom pairs.

That is absolutely trivial compared with a 200–400 orbital eigensolve.

This also reveals another small mistake in D7: **4950 pairs do not imply that you require >1 WG/system**. A 256-thread WG can simply do

```c
for (int ipair=tid; ipair<nPairs; ipair+=256)
```

and each thread handles ~20 pairs.

So even H assembly can remain 1 WG/system quite far upward if that turns out to benchmark well.

---

# 9. The real size variable is orbitals, not atoms

This matters a lot.

“<100 atoms” can mean:

```text
100 H atoms       ~100 orbitals
100 C/N/O atoms   ~400 orbitals
```

with your current s/sp model.

With d shells it can be larger again.

So I would explicitly design three regimes:

| \(N_{\rm orb}\) | likely strategy                                                  |
| --------------: | ---------------------------------------------------------------- |
|         ≤ 16–32 | **one subgroup/system**, perhaps 4–8 systems per WG              |
|          ~32–64 | one WG/system, matrix fully in local memory                      |
|             >64 | tiled multi-WG GEMM + true block/parallel Jacobi or purification |

That makes the proposed `N≤64` mega-kernel a nice specialization, but not the foundation.

---

# 10. The current Jacobi implementation is not yet the eigensolver you ultimately need

There are two variants now.

`local_jacobi_blocks` loads \(A\) and \(V\) to local memory, then **thread 0 does the actual Jacobi sweeps**. That's actually not crazy for thousands of tiny matrices: parallelism is across matrices.

The “parallel” version parallelizes the row updates, but still iterates sequentially over every \(p,q\) rotation with several barriers per rotation.

For \(N=64\), that's roughly

$$
\frac{64\cdot63}{2}=2016
$$

rotations per sweep.

So you get thousands of workgroup barriers per sweep.

I think the next Jacobi should use a **true parallel cyclic/Brent–Luk schedule**:

```text
round 0:  (0,63) (1,62) (2,61) ...   all simultaneously
round 1:  another disjoint matching
...
```

i.e. \(N/2\) independent rotations per round.

Interestingly, you already have `brent_luk_rounds()` scaffolding in Rust, but the full block-Jacobi orchestration isn't there yet.

For the 100–400 orbital regime, **this eigensolver is probably the central performance project**, much more important than H assembly.

---

# 11. Purification is interesting, but the design document overstates it

The design currently calls purification roughly

$$
O(N^2 n_{\rm iter}).
$$

But your implementation computes \(D^2\), and McWeeny also \(D^3\), using dense GEMMs. Therefore it is still:

$$
O(N^3 n_{\rm iter}).
$$

Its attraction is not a better asymptotic scaling here; its attraction is that **GEMM maps extremely efficiently to the GPU** and it avoids eigenvectors.

Also, the current TC2 implementation is precisely an anti-pattern for your intended architecture:

1. GPU computes traces.
2. traces are read back to CPU,
3. Rust loops over systems,
4. launches **one `purify_tc2_one_batch` kernel per system**.

That should become simply:

```c
mode[system] = trace[system] > Ne[system];

D[system][ij] =
    mode[system] ? D2[...] : 2*D[...] - D2[...];
```

for all systems in one launch.

No readback.

I would nevertheless finish a reliable eigensolver first. For MD forces in a non-orthogonal basis you will eventually need not only population density but the information required for the overlap/Pulay contribution, so the “density only” route should be designed together with the force derivation rather than optimized in isolation.

---

# 12. One important missing GPU operation: constructing \(S^{-1/2}\)

This is currently hidden by the tests.

The generalized-eigenproblem test explicitly does:

```rust
// Step 1: CPU computes S^{-1/2}
```

and uploads \(X\) to the GPU.

For MD, NEB, or a relaxed scan, \(S(\mathbf R)\) changes with geometry, so this needs a production GPU solution.

Fortunately it is only **once per geometry**, not once per SCC iteration.

Candidates I would test are:

```text
A. diagonalize S → X = U s^-1/2 U^T
B. batched Cholesky + triangular transformations
C. Newton–Schulz inverse square root using GEMMs
```

A is the cleanest initial implementation because you need the symmetric eigensolver anyway.

C may become attractive because overlap matrices are generally reasonably close to identity and it is almost entirely GEMM.

This missing step is, to me, a bigger architectural issue than queue scheduling.

---

# 13. There is even a nice SCC optimization once \(X\) exists

This is not something I would implement first, but it is worth recording now.

Let

$$
V_{\mu\nu}=\delta_{\mu\nu}V_{A_\mu}.
$$

Then

$$
H^1=\frac12(VS+SV).
$$

With

$$
X=S^{-1/2},\qquad Y=S^{1/2}=SX,
$$

the SCC correction directly in the orthogonal basis is

$$
XH^1X
=
\frac12(XVY+YVX).
$$

Since

$$
YVX=(XVY)^T,
$$

you only need

$$
A=XVY
$$

and

$$
H'_{\rm SCC}=H'_0+\frac12(A+A^T).
$$

Because \(V\) is diagonal, multiplying by it is just a row/column scaling.

So instead of doing, **every SCC iteration**,

```text
build H_scc
X^T H_scc         GEMM
(...)*X           GEMM
```

you can precompute

$$
H'_0=XH^0X
$$

once per geometry and require only approximately **one GEMM per SCC iteration** for the electrostatic correction.

That could become quite valuable once diagonalization is fast.

---

# 14. I found several concrete correctness problems in the current OpenCL H path

I would fix these **before benchmarking architectural variants**, otherwise you could spend time optimizing a wrong kernel.

The most serious ones I noticed are:

| issue                                                                                              | consequence                                                                                                      |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `assemble_pairs` has `__local Fragment l_frags[128]` but indexes it by `p.replica`                 | **cannot support your intended thousands of replicas**; replica ≥128 accesses outside that array                 |
| `onsite_diagonal` always writes s + three p onsite entries                                         | looks wrong/out-of-bounds for H/1-orbital atoms                                                                  |
| `onsite_and_va` uses `charges` directly rather than `charges-q0`                                   | if `charges` are the Mulliken populations initialized to `q0`, neutral atoms already produce a nonzero SCC shift |
| `interp_sk_1` casts scalar table to `float4*` and indexes `tab4[base]`                             | appears to advance by \(4\,base\), not read `base..base+3`                                                       |
| `write_symmetric_4x4` casts `float*` to `float4*` but uses indices expressed in scalar-float units | appears to multiply matrix offsets by four                                                                       |
| `rotate_4x4` reuses one `float3 pp` for all three p rows                                           | does not look like the required \(u_i u_j(V_\sigma-V_\pi)+\delta_{ij}V_\pi\) outer-product structure             |

These are visible directly in the present Hamiltonian kernel.

The first three in particular are quite unambiguous to me. For the vectorized 4×4 code I would verify against CPU block-by-block before changing it, but it looks seriously suspicious.

There is also currently **no integrated GPU H/S → SCC → eigensolver driver** in `qmqm`; the matrix operations have tests, but the Hamiltonian OpenCL side does not yet have equivalent parity coverage. The existing GPU diagonalization tests are good groundwork.

---

# 15. I would also refactor the OpenCL runtime now

`GpuMatrixContext::new()` currently creates its own:

```text
Context
Queue
Program
```

and the queue is private.

Before implementing the Hamiltonian driver I would change this to:

```rust
GpuRuntime
    context
    device
    queues[]
    capabilities
    compiled program cache

HamiltonianOps
MatrixOps
SccOps
ForceOps
```

all sharing the same runtime/context.

Otherwise it is too easy to end up with H buffers created in one OpenCL context and matrix kernels in another.

Also, the current matrix methods construct a new OpenCL `Kernel` object on practically every call. For a tight SCC loop I would cache kernel objects and only update arguments.

And create a profiling-enabled queue for benchmarks; current `Queue::new(..., None)` doesn't do that.

---

# 16. Homogeneous templates let you simplify a lot

For the workload you actually care about first — scans, NEB, replicas — I would exploit sameness aggressively.

For 1000 copies of the same molecular topology, you don't need:

```text
Fragment metadata ×1000
PairEntry.replica × every pair
orbital offsets ×1000
species ×1000
```

Keep one template:

```text
Template:
    nAtoms
    nOrbs
    atomSpecies[]
    atomOrbOffset[]
    pairI[]
    pairJ[]
    pairType[]
```

and instances:

```text
coords[Nsys][Natom][3]
q[Nsys][Natom]
H0[Nsys][Norb*Norb]
...
```

Then offsets are arithmetic:

```c
atom_base = system * Natom;
mat_base  = system * Norb * Norb;
```

This removes a lot of indirect addressing and the whole `l_frags[128]` problem.

For heterogeneous work later, create several template/size classes.

---

# 17. One subtle conceptual distinction I would introduce

At the moment “fragment”, “replica”, and “system” are becoming mixed together.

I would define:

```text
System
    independent scheduling + convergence unit

Fragment
    independently diagonalized QM block inside one System
```

So:

```text
NEB:
    System = one NEB image
    usually one Fragment

QM/QM:
    System = whole mutually polarizing complex
    Fragment = individual independently-diagonalized molecular pieces

MD ensemble:
    System = one trajectory
```

This becomes important because fragments in QM/QM **cannot independently declare SCC convergence**: their charges are mutually coupled. Their parent System converges as a whole.

It will make the GPU scheduler much easier to reason about later.

---

# 18. What I would benchmark now

I would **absolutely perform the scheduling experiment you proposed**, but only after a correct minimal H/S+diagonalization pipeline exists.

Use exactly the same representative kernel chain and compare:

| mode | experiment                            |
| ---- | ------------------------------------- |
| A    | all systems in one giant batch        |
| B    | microbatches 16 / 32 / 64 / 128       |
| C    | one system/kernel, one in-order queue |
| D    | one system/kernel, 2 / 4 / 8 queues   |
| E    | microbatches on 2 / 4 queues          |

And artificially impose SCC counts such as:

```text
uniform:       all 10 iterations

mild spread:   random 5..15

bad tail:      90% need 6..10
               10% need 25..40
```

Measure both:

```text
GPU event time
host wall time
systems / second
```

because launch/driver overhead is exactly what we are investigating.

I would be surprised if C/D wins for tiny systems. But I genuinely don't know whether **A or a 2–4 queue microbatch implementation** will win once convergence has a strong heavy tail; that is worth measuring rather than deciding theoretically.

---

# Revised development direction

I would modify your staged roadmap approximately to:

**Stage 0 — correctness first.** Fix the OpenCL H/S kernels and make exhaustive CPU↔GPU parity tests: H2, C2, HC/CH, H2O, CH4, N2, etc. Test every block type separately.

**Stage 1 — homogeneous geometry batch.** Introduce a `SystemClass`/template. Coordinates stay GPU-resident. Compute H0, S, and dense gamma \(G\) on GPU for hundreds of replicas.

**Stage 2 — complete generalized eigenproblem.** Build \(S^{-1/2}\) on the GPU and get a reliable batched eigensolver. This is likely the main technical challenge.

**Stage 3 — device-resident SCC.** \(G\Delta q\) → H correction → solve → Mulliken → residual → GPU simple mixer. Active mask, no PCIe traffic inside SCC.

**Stage 4 — scheduling benchmark.** Giant batch vs microbatch vs multiple queues using synthetic convergence distributions.

**Stage 5 — forces + MD/NEB.** This will determine exactly what density/eigenvector quantities must survive the solve.

**Stage 6 — specialized fast paths.** Subgroup/WG tiny-system solver, GPU DIIS, command buffers, purification, direct orthogonal-basis SCC update, etc.

**Stage 7 — QM/QM coupling.** Two-level System/Fragment layout and batched fragment solves grouped by \(N_{\rm orb}\).

---

## The three changes I consider most important

If I compress all of this down:

**First:** abandon the requirement that one system always maps to one workgroup. Keep **system as a batch index**, and let every numerical kernel choose its own decomposition.

**Second:** separate the calculation into **geometry setup** and **SCC inner loop**. Compute H0, S, gamma, and \(S^{-1/2}\) once per geometry. The current H assembly does too much inside SCC.

**Third:** don't choose between “CPU-driven SCC” and “mega-kernel SCC.” Use a **host-enqueued but completely GPU-resident kernel pipeline**, with active masks. Then use microbatches/multiple queues only for genuinely asynchronous outer workflows such as independent MD.

With that architecture, I think your original idea scales much more naturally from H2/water all the way to ~100-atom / 300–400-orbital systems without having to redesign everything at the point where a matrix stops fitting into one workgroup.

[1]: https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_khr_command_buffer.html "https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_khr_command_buffer.html"
[2]: https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clEnqueueNDRangeKernel.html "https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clEnqueueNDRangeKernel.html"
[3]: https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_queue_properties.html "https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_queue_properties.html"
[4]: https://docs.nvidia.com/deploy/mps/architecture.html "https://docs.nvidia.com/deploy/mps/architecture.html"



---

# USER

OK, plese write me the key OpenCL kernels, make sre they are highly optimized efficiently using local memeory ashared within workgroups to amortize global memory reads and assuming some reasonable workgroups sizes (like mul;tipe of 32 for nvidia?)

 you mentioned a **true parallel cyclic/Brent–Luk schedule**:

fofucs on it. Consider that if we implement some assumptions (like that all systems are exacly same size and number of orbitals is smaller than worgroupsize (fit in on workgroup), we can simplify it, but I'm not sure if we fit the workspace whole matrix into local memory at once?

(we should consider different options there).

Now the Lawdin transform S^-(1/2) is basically again problem of Jacobi rotations and matrix multiplication so we can reuse the same kernels.

I completely agree weth you we should not to one mosnter kernel doing everything, and we should schedule the kernel exacution by CPU, but not copy data over PCI (they should reside on GPU), everything should be single precission! that is important my GPU RTX3090 is like 40x slower in double precission. Our software target cheap gaming GPUs. Lets see how this will limit the accuracy. But I'm perhaps fine if the accuracy is slightly limited. (We can think about some hybrid architexture later if we find it needed).

---


# Chat GPT 5.6 sol


Yes — with your assumptions, I think we can make a very clean fast path:

> **homogeneous batch + compile-time \(N_{\rm orb}\) + f32 + one workgroup/system + whole \(A\) and \(V\) resident in local memory during Jacobi.**

That is substantially better than the current “parallel Jacobi” kernel, because the current implementation still walks sequentially through every \((p,q)\) rotation and puts barriers around each one.

I wrote a complete OpenCL source implementing this approach and syntax-checked it as OpenCL 1.2 for \(N=32,63,64\):

[Download `dftb_small_dense.cl`](sandbox:/mnt/data/dftb_small_dense.cl)

It contains four kernels: the true parallel cyclic Jacobi eigensolver, a whole-matrix-local batched GEMM, construction of \(S^{-1/2}\), and the cheap SCC Hamiltonian update.

## 1. Full-local Jacobi is quite realistic

For \(N=64\), two float matrices require

$$
2N^2(4\,{\rm B}) = 32\ {\rm KiB}.
$$

I deliberately store them with local leading dimension `N+1` to avoid the particularly ugly bank-stride patterns of a power-of-two matrix:

$$
2N(N+1)4 = 32.5\ {\rm KiB}.
$$

With rotation parameters, reductions and sorting scratch, the kernel is around 35 KiB/workgroup.

That is a good target. OpenCL guarantees at least 32 KiB local memory, so this exact \(N=64\) specialization isn't universally guaranteed, but contemporary gaming GPUs generally have more. Ampere cc 8.6 hardware can expose up to 99 KiB/block under CUDA's opt-in mechanism, although I would **not assume this through OpenCL**; query `CL_DEVICE_LOCAL_MEM_SIZE` and `CL_KERNEL_LOCAL_MEM_SIZE`. ([docs.nvidia.com][1])

I would start with:

```text
N <= 32 : WG=128
N ~ 32-64 : WG=256
```

On NVIDIA these are 4 and 8 warps respectively. More generally, query `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` rather than hardcoding the assumption that every GPU wants 32. ([Khronos Registry][2])

Your existing Rust matrix infrastructure already specializes kernels by text substitution, so adding `N_ORB` and `WG` to that scheme fits very naturally.

---

# 2. The important Jacobi change

Instead of doing

```text
rotate (0,1)
barrier
rotate (0,2)
barrier
rotate (0,3)
...
```

we construct a Brent–Luk / round-robin matching:

```text
round 0: (63,0) (1,62) (2,61) ... (31,32)
round 1: (63,1) (2,0)  (3,62) ... 
...
```

Every orbital participates in exactly one rotation per round.

After \(N-1\) rounds, every unordered pair has been treated exactly once.

The useful observation is stronger than merely saying the rotations are independent. If

$$
G=\operatorname{diag}(G_0,G_1,\ldots),
$$

where each \(G_i\) is one independent \(2\times2\) rotation, then the whole round is

$$
A' = G^T A G.
$$

So we update \(A\) **by 2×2 pair-pair blocks**.

For pairs \(a=(p,q)\), \(b=(r,s)\),

$$
B_{ab}=G_a^T A_{ab}G_b.
$$

Every such 2×2 block is disjoint in memory. Therefore each work-item can own one block, and there are **no atomics and no races**.

That gives only **one barrier per Jacobi round**, rather than barriers for every individual rotation.

The core looks like this:

```c
for (int round = 0; round < JROUND; ++round) {

    // N/2 independent rotations
    if (lid < JPAIR) {
        ushort2 pq = jacobi_pair(round, lid);
        pairs[lid] = pq;

        int p = pq.x;
        int q = pq.y;

        rot[lid] = jacobi_rotation(
            A[p*JLD+p],
            A[q*JLD+q],
            A[p*JLD+q]
        );
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Apply A <- G^T A G blockwise
    for (int t=lid; t<JPAIR*JPAIR; t+=WG) {

        int ia = t/JPAIR;
        int ib = t-ia*JPAIR;

        if (ib < ia) continue;

        ushort2 ap = pairs[ia];
        ushort2 bp = pairs[ib];

        ...
    }

    // V <- V G
    for (int t=lid; t<JPAIR*JN; t+=WG) {
        ...
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
```

The full implementation is in the attached file.

This is, I think, the most important architectural improvement over your existing Jacobi kernel.

---

## 3. We don't even need to upload the Brent–Luk schedule

There is a particularly convenient round-robin formula.

For even dimension \(M\), fix index \(M-1\). In round \(r\):

```c
pair 0:
    (M-1, r)

pair k>0:
    ((r+k) mod (M-1),
     (r-k) mod (M-1))
```

So the kernel computes the schedule itself:

```c
inline ushort2 jacobi_pair(const int round, const int ipair) {
    if (ipair == 0)
        return (ushort2)(JN-1, round);

    const int m = JN-1;

    return (ushort2)(
        (round + ipair)     % m,
        (round + m-ipair)   % m
    );
}
```

Only \(N/2\) threads perform those integer modulo operations per round, so this is negligible.

It also gives us a very clean solution for **odd \(N\)**.

I pad

$$
N\rightarrow N+1
$$

and make the dummy state completely decoupled with a huge diagonal eigenvalue. The fixed Brent–Luk index then happens to be the dummy state. When it gets paired with a real state, the off-diagonal element is zero and the resulting Jacobi rotation is exactly identity.

So one kernel works for e.g. \(N=63\) as well as \(N=64\).

---

# 4. Matrix multiplication: for \(N\le64\), I would benchmark a surprisingly simple version

Your current tiled GEMM is fundamentally sensible: each output tile is one workgroup.

But when we have **thousands of matrices of only 32–64 orbitals**, I'd benchmark something simpler:

> one WG/system, load **both complete input matrices once** into local memory, then calculate all \(N^2\) outputs.

The kernel in the file is essentially:

```c
__kernel void matmul_full_local_batched(
    __global const float* A,
    __global const float* B,
    __global float* C
){
    int sid = get_group_id(0);
    int lid = get_local_id(0);

    __local float LA[N_ORB*(N_ORB+1)];
    __local float LB[N_ORB*(N_ORB+1)];

    // global -> local exactly once
    for(int t=lid; t<N_ORB*N_ORB; t+=WG){
        int i=t/N_ORB;
        int j=t-i*N_ORB;

        LA[i*(N_ORB+1)+j] = A[sid*N2+t];
        LB[i*(N_ORB+1)+j] = B[sid*N2+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int t=lid; t<N_ORB*N_ORB; t+=WG){
        int i=t/N_ORB;
        int j=t-i*N_ORB;

        float sum=0.0f;

        for(int k=0;k<N_ORB;k++)
            sum=fma(
                LA[i*(N_ORB+1)+k],
                LB[k*(N_ORB+1)+j],
                sum
            );

        C[sid*N2+t]=sum;
    }
}
```

For \(N=64,\ WG=256\), each thread computes 16 matrix elements.

The memory access is particularly nice: within a warp, `LA[i,k]` is mostly a broadcast and `LB[k,j]` is consecutive across `j`.

There is no reason to assume this beats tiled GEMM — **benchmark both** — but for many tiny matrices I think it has a serious chance.

For \(N>64\), I would revert to your tiled multi-WG GEMM.

---

# 5. Löwdin becomes a very clean composition

Once we have the Jacobi kernel, compute

$$
S = U\Lambda U^T
$$

using exactly the same eigensolver.

Then

$$
X=S^{-1/2}
=
U\Lambda^{-1/2}U^T.
$$

I included a kernel that loads \(U\) once into local memory and evaluates

```c
X_ij = sum_k U_ik * rsqrt(lambda_k) * U_jk;
```

for every element.

Then the generalized eigenproblem becomes:

```text
jacobi(S)            -> lambdaS, U_S

build_inv_sqrt(...)  -> X

matmul(X,H,tmp)
matmul(tmp,X,Hp)     -> H' = X H X

jacobi(Hp)           -> eps, U

matmul(X,U,C)        -> physical MO coefficients
```

Because \(X=S^{-1/2}\) is symmetric, there is no transpose complication.

And critically:

**\(X\) is calculated once per geometry, not once per SCC iteration.**

Your existing test path currently constructs \(S^{-1/2}\) on the CPU and uploads it, so this fills one of the main missing production pieces.

---

# 6. Everything stays GPU-resident

I completely agree with your constraint here.

For one geometry:

```text
coords
  ↓
H0, S, gamma
  ↓
diag(S)
  ↓
X=S^-1/2
  ↓
H0' = X H0 X
```

Then the SCC loop is merely:

```text
q
↓
V = gamma * dq
↓
H_scc
↓
orthogonal transform
↓
Jacobi
↓
Mulliken q_new
↓
mix
↓
repeat
```

Rust just enqueues these kernels into the same in-order command queue.

No `read_buffer()`.

No `queue.finish()`.

No PCIe traffic.

An in-order queue already establishes execution ordering between the kernels; kernel boundaries give you the global synchronization that workgroups themselves cannot provide. OpenCL workgroups synchronize internally through local-memory barriers, while distinct workgroups have no equivalent general synchronization within a kernel. ([Khronos Registry][3])

---

# 7. SCC Hamiltonian update is almost free

I also included:

```c
H_ij =
    H0_ij
  + S_ij * 0.5f*(V[atom(i)] + V[atom(j)]);
```

with the atom potentials first cached in local memory.

So H0 and S are each read once, while the much smaller \(V_A\) array gets reused from local memory:

```c
__local float LV[N_ATOM];

for(int a=lid; a<N_ATOM; a+=WG)
    LV[a]=V_atom[sid*N_ATOM+a];

barrier(CLK_LOCAL_MEM_FENCE);

for(int t=lid; t<N2; t+=WG){

    int i=t/N_ORB;
    int j=t-i*N_ORB;

    float vij =
        0.5f * (
            LV[orb_atom[i]]
          + LV[orb_atom[j]]
        );

    H[base+t] =
        fma(S[base+t],vij,H0[base+t]);
}
```

That is much better than embedding the SCC correction inside SK interpolation as the current `assemble_pairs` does.

---

# 8. What happens around \(N=64\)–100?

This is where there are interesting alternatives.

| representation                      |     local storage | approximate practical range under conservative 48 KiB |
| ----------------------------------- | ----------------: | ----------------------------------------------------: |
| full \(A\) + full \(V\)             |    \(8N^2\) bytes |                                                ~64–72 |
| packed symmetric \(A\) + full \(V\) |    \(6N^2\) bytes |                                                ~80–84 |
| only \(A\), no eigenvectors         |    \(4N^2\) bytes |                                                  ~100 |
| blocked/global \(A,V\)              | small local tiles |                                             arbitrary |

I think **packed symmetric \(A\)** is actually worth trying as a second version.

We don't mathematically need the lower triangle at all. The blockwise Jacobi update already knows that

$$
A_{ba}=A_{ab}^T.
$$

So we could store only the upper triangle of \(A\), while keeping \(V\) dense.

For \(N=84\):

$$
A_{\rm packed}+V
\approx 42\ {\rm KiB}.
$$

That might let the exact same one-WG Jacobi architecture reach ~80 orbitals even on a conservative 48-KiB target.

The penalty is more complicated indexing and probably somewhat worse local-memory access. I would **not implement that until the clean dense \(N\le64\) version is benchmarked**.

Beyond roughly 80 orbitals I would stop fighting local memory and use genuine **block Jacobi**:

```text
matrix -> 16x16 or 32x32 blocks

Brent-Luk schedule over BLOCKS rather than individual orbitals

for each block pair:
    load 2T x 2T compound block
    diagonalize locally
    obtain Q

kernel boundary

apply Q to block rows/columns using tiled GEMMs

next Brent-Luk round
```

That naturally extends your current `brent_luk_rounds()` infrastructure rather than replacing it.

---

# 9. Single precision: the first thing I would monitor

I agree with targeting f32 first. Double precision on cheap consumer NVIDIA GPUs is exactly the wrong optimization target for this project.

But the sensitive operation will probably **not** be ordinary H diagonalization. It will be \(S^{-1/2}\) when \(S\) becomes ill-conditioned.

Therefore I made `build_inv_sqrt_from_eig` return the lowest overlap eigenvalue per system.

I would monitor at least

$$
\lambda_{\min}(S),
$$

$$
\|U^TU-I\|,
$$

$$
\|AU-U\Lambda\|,
$$

and ultimately the generalized residual

$$
\|HC-SC\varepsilon\|.
$$

The kernel takes an explicit `lambda_floor`, but I would not silently regularize production calculations. If \(S\) develops an eigenvalue comparable to f32 numerical precision, that is useful diagnostic information.

My expectation is that ordinary chemically sensible DFTB bases will often be perfectly workable in f32, but **this has to be established empirically against your f64 CPU implementation**, particularly for energies and forces.

---

The first implementation I would now put into `rust_dftb` is exactly the attached dense-local kernel file and benchmark **`jacobi_cyclic_local_batched` versus the current `local_jacobi_blocks_parallel` at N=8,16,32,48,64 and batch=1,10,100,1000**. That experiment will immediately tell us whether this architecture is as good on the RTX 3090 as it looks theoretically.

[1]: https://docs.nvidia.com/cuda/ampere-tuning-guide/?utm_source=chatgpt.com "1. NVIDIA Ampere GPU Architecture Tuning Guide — Ampere Tuning Guide 13.3 documentation"
[2]: https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/enums.html?utm_source=chatgpt.com "enums(3)"
[3]: https://registry.khronos.org/OpenCL/specs/3.0-unified/pdf/OpenCL_API.pdf?utm_source=chatgpt.com "The OpenCL™ Specification"
