https://chatgpt.com/share/6aad3364-d644-83ed-9e32-945188f3cf88

---

# USER


We have dftb on GPU with multiple system replicas running in parallel, and we are trying to optimize it, but significant problem is that some replicas may converge faster than other so we waste idle/void threads and workgroups on them. I was thinking we should rather do something like schedulers lets say we have 400 systems we want to process we maybe start only with 100 slots, check after every 10 scc iterations, and the finished replicas replace by the yet unfinished. so it is not fully synchronouz it ti like scheduler for computing luster but much faster in rust running on single GPU 

we use jacobi rotation for eignesolver

https://github.com/ProkopHapala/dftbplus/tree/main/rust_dftb/src/qmqm

---

What I discussed with my coding agent:

wait why we have 3D kernel (col_groups, row_groups), this is like 20x20? That is completely useless, perhaps it does not affect preformance but it complicates implementation of the slots and scheduler which assing these slots to valid active systems. It make sense do following

make the kernel size smaller than total number of replicas (<400), always run the SCC for just few steps (e.g. 10 steps?) and when some systems converge, put at their place the remaining unconverged jobs. Basically like scheduler for fixed size computing cluster, but much faster using simgle GPU, it should be programed with minimal overhapd in reust, can your first thingk about the desing and write separate document. We should also consider running whole MDloop (for relaxed scan with FIRE running for each replica so each replica have outer MD loop and inner SCC loop and even inner Jacobi diagonalized loop)


---

# Chat GPT 5.6 sol

Yes — I think the **slot scheduler is the correct architectural direction**, especially once you include relaxed scans where every replica has its own outer FIRE loop and inner SCC loop.

One correction to the discussion with the coding agent: I would **not remove the 3-D GEMM launch**. The first two dimensions are genuinely useful matrix tiles. What should disappear is the assumption that `group_id(2)` *is the physical replica*. The current `batched_gemm_active` already avoids arithmetic for inactive replicas, but still launches the tile grid for the whole fixed batch.

Your existing optimization notes were already moving toward exactly this distinction: keep masks for convergence inside a chunk, but use compact IDs between chunks, with persistent physical slot storage and no matrix movement. 

The stronger version you are proposing now—**fewer physical slots than jobs, immediately refill finished slots from a job queue**—is the natural second step.

# GPU Slot Scheduler for Batched DFTB SCC, Jacobi, and FIRE Relaxation

## 1. Motivation

The present GPU DFTB implementation treats a batch of replicas as a fixed set of physical systems. Each replica owns a persistent region of all device buffers, and most kernels are launched for the complete batch. Converged replicas are masked by `active[sid]`, so they stop doing expensive arithmetic, but their workgroups are still represented in the launch geometry.

This is particularly inefficient when convergence times have a broad distribution. For example, out of 400 systems, most may converge after 20–40 SCC iterations while a small tail requires 80–100 iterations. Late in the calculation, the GPU therefore sees many workgroups which immediately return and only a few replicas performing useful computation.

The current implementation is already explicitly organized around persistent per-replica GPU state, including SCC state and device-resident FIRE state. The engine assumes a homogeneous batch: all replicas share topology/species/N/n_atoms, while geometry differs. This restriction is actually very convenient for a slot scheduler.

The proposed solution is to distinguish:

```text
JOB       = a physical problem we eventually want to solve
SLOT      = persistent GPU memory allocated for one currently executing job
WORK ID   = one slot participating in the next kernel launch
```

Thus 400 scan points might be processed by only, for example, 128 persistent GPU slots.

The correct number of slots should be measured rather than fixed in advance.

---

# 2. Do not remove the tiled 3-D GEMM organization

The current tiled GEMM uses:

```text
group_id(0) -> matrix row tile
group_id(1) -> matrix column tile
group_id(2) -> replica
```

The first two dimensions are essential.

For N=246 with a 16×16 tile, one matrix already contains roughly

```text
ceil(246/16)^2 = 16^2 = 256
```

independent tile workgroups.

We therefore should NOT replace a launch such as

```text
(col_groups, row_groups, batch)
```

with one workgroup per replica.

Instead change only the interpretation of the third coordinate.

Current:

```c
sid = get_group_id(2);

if (!active[sid]) return;

A += sid*N*N;
B += sid*N*N;
C += sid*N*N;
```

Target:

```c
int iw  = get_group_id(2);       // compact work index
int sid = work_ids[iw];          // persistent physical slot

A += sid*N*N;
B += sid*N*N;
C += sid*N*N;
```

and launch

```text
(col_groups, row_groups, n_work)
```

instead of

```text
(col_groups, row_groups, batch)
```

This preserves the efficient GEMM while completely removing empty replicas from the launch domain.

The existing `batched_gemm_active` already illustrates why the change is small: it currently obtains `ib=get_group_id(2)` and checks `active[ib]` before entering the tiled matrix work.

---

# 3. Three levels of identity

I would make the distinction explicit in the code.

```rust
type JobId  = u32;   // 0 .. total_jobs-1
type SlotId = u32;   // 0 .. n_slots-1
```

Host-side scheduler state:

```rust
struct Slot {
    job: Option<JobId>,
    phase: Phase,

    scc_total_iters: u32,
    md_steps: u32,
}

enum Phase {
    Empty,
    Assemble,
    Scc,
    Finalize,
    Force,
    Fire,
    Done,
    Failed,
}
```

Mappings:

```text
slot_job[slot] -> job
job_result[job] -> final result

scc_ids[]       -> physical slots doing SCC now
force_ids[]     -> physical slots needing forces
assemble_ids[]  -> physical slots whose geometry changed
refill_ids[]    -> newly assigned physical slots
```

Crucially:

> Matrices never move between slots.

Slot 37 always corresponds to the 37th region of

```text
H0
S
G
C
SC
q
DIIS history
coordinates
FIRE velocities
...
```

Only the meaning of slot 37 changes when one job finishes and another job is loaded into it.

---

# 4. Stage 1: compact active IDs without refill

I would implement this first because it is simple and independently testable.

Suppose physical batch size is still 400.

Initially:

```text
active_ids =
[0,1,2,...,399]
```

After an SCC chunk:

```text
active:
1110110100...
```

Rust builds:

```text
active_ids =
[0,1,2,4,5,7,...]
```

and uploads perhaps only a few hundred integers.

Then every SCC kernel launches over:

```text
n_active
```

rather than 400.

This is precisely the distinction already identified in the previous optimization notes:

```text
active mask -> stops work inside a chunk
active_ids  -> removes dead WGs between chunks
```

and the existing plan explicitly proposed compact IDs with persistent physical slots rather than moving matrices. 

This change should be made before slot refill because it exercises almost all of the kernel-addressing changes required by refill.

---

# 5. SCC should execute in chunks

Do not let one call mean:

```text
run SCC until every replica converges
```

Instead expose something like:

```rust
scc_chunk(work_ids, n_iter_chunk)
```

A reasonable initial chunk size is:

```text
8 SCC iterations
```

or perhaps 10 as originally proposed.

Conceptually:

```rust
while !queue.empty() || !active_slots.empty() {

    run_scc_chunk(&scc_ids, 8);

    read_small_status_arrays();

    for slot in scc_ids {
        if converged(slot) {
            ...
        }
    }

    compact_work_lists();
}
```

The host synchronization at this point is acceptable.

We are reading perhaps:

```text
100–300 RMS values
100–300 state integers
```

not matrices.

That is only kilobytes.

There is no justification for a complicated device-side scheduler or global atomic job queue at this stage. Rust can perform the scheduling essentially for free relative to several SCC iterations.

This also fits the existing implementation, which already has explicit per-replica SCC status and retry semantics rather than treating the batch as one indivisible convergence unit.

---

# 6. Stage 2: physical slots smaller than the job set

Once compact addressing works, change:

```text
batch = total_jobs
```

into:

```text
n_slots << total_jobs
```

Example:

```text
total_jobs = 400
n_slots    = 128
```

Startup:

```text
jobs:   0 ... 399

slots:
0   <- job 0
1   <- job 1
...
127 <- job 127

pending:
128 ... 399
```

Suppose after one SCC chunk jobs corresponding to slots 7, 19 and 48 converge.

We finalize their results and immediately do:

```text
slot 7  <- job 128
slot 19 <- job 129
slot 48 <- job 130
```

No allocation.

No moving the state belonging to the other 125 slots.

No resizing kernels.

Only overwrite/reset the three reused slots.

---

# 7. Slot refill operation

I would provide one explicit operation:

```rust
refill_slots(refills: &[(SlotId, JobId)])
```

It performs, for only those slots:

```text
coordinates <- new job geometry

reset q
reset DIIS history
reset convergence status
reset state validity
reset Jacobi diagnostics
reset FIRE state if this is a relaxation
reset velocity
reset FIRE dt/alpha/n_pos

assemble H0
assemble S
build G
build/rebuild Löwdin state as required

initialize electronic state
```

Importantly, do this through a compact:

```text
refill_ids[]
```

rather than running initialization over every slot.

The current `set_coords()` works on the full fixed batch and validates `coords.len()==batch*n_atoms`; it therefore needs a subset version for this design. Its internal operation—update coordinates and then refresh pairs/G/H0/S on device—is already the correct conceptual operation.

I would add something like:

```rust
set_coords_slots(slot_ids, coords)
assemble_slots(slot_ids)
```

rather than contorting the existing all-batch routine.

---

# 8. Choosing number of slots

I strongly recommend NOT deciding in advance that the answer is 100.

Measure:

```text
32
64
96
128
192
256
400
```

and plot:

```text
systems / second
```

for representative N.

There are competing effects.

For GEMM, surprisingly few systems may saturate the GPU because each matrix produces many tile workgroups.

For N=246:

```text
~256 WGs/system/GEMM
```

so even 32 systems produce thousands of workgroups.

But Jacobi behaves differently.

If Jacobi is approximately:

```text
1 WG / system
```

then 100 slots on an RTX 3090 with 82 SMs gives only approximately one workgroup per SM at that stage.

400 replicas provide several waves and may hide stalls substantially better.

Therefore the correct slot count is:

> the smallest slot pool for which throughput is within perhaps 95–98% of maximum throughput.

It may well be around 128–256 rather than 100.

And the best value may differ for N=86 and N=246.

---

# 9. There is no need to schedule inside Jacobi

The hierarchy is important:

```text
JOB / relaxation
    |
    +-- MD/FIRE steps
           |
           +-- SCC cycles
                  |
                  +-- Jacobi eigensolver
                         |
                         +-- Jacobi sweeps/rotations
```

I would schedule at the SCC boundary, not at individual Jacobi sweeps.

A Jacobi workgroup belonging to a system can naturally run until its local eigenproblem reaches its stopping condition.

Different workgroups are already dynamically scheduled by the GPU hardware.

Trying to interrupt Jacobi every few sweeps and reschedule it from Rust would create substantial complexity for almost no benefit.

The useful scheduling boundary is:

```text
~4–10 SCC iterations
```

because SCC convergence differs by tens of iterations between replicas.

---

# 10. Relaxed scan / FIRE requires a job state machine

This is where slots become considerably more valuable.

For a single-point calculation:

```text
LOAD
 -> SCC
 -> FINALIZE
 -> DONE
```

For relaxation:

```text
LOAD
 |
 v
ASSEMBLE
 |
 v
SCC
 |
 v
FORCES
 |
 v
FIRE STEP
 |
 +---- converged ----> DONE
 |
 +---- not converged
          |
          v
       ASSEMBLE
          |
          v
         SCC
```

Each physical slot owns an entire trajectory.

This matches the current architecture quite well because FIRE already has per-replica persistent device state including velocity, control variables and a parking mask.

A job remains in the same physical slot throughout its complete relaxation.

Therefore all of its warm electronic state is retained:

```text
q
C
SC
μ
DIIS history
```

and all FIRE state is retained:

```text
x
v
dt
alpha
n_positive
```

Only when the complete relaxation finishes is the slot reassigned.

This is substantially better than trying to migrate partially relaxed jobs between slots.

---

# 11. Allow slots to be in different phases

Eventually there is no reason all slots must be synchronized to the same outer operation.

For example:

```text
slots 0..91    SCC
slots 92..105  need force evaluation
slots 106..109 need new geometry assembly
slots 110..127 newly refilled
```

The Rust scheduler simply constructs:

```rust
scc_ids
force_ids
assemble_ids
```

and launches the appropriate kernels for each group.

Do NOT solve this by building one giant GPU kernel containing:

```text
if phase == SCC ...
if phase == FORCE ...
if phase == FIRE ...
```

Different stages need very different workgroup geometries and memory footprints.

The scheduler belongs in Rust.

The numerical work belongs in specialized GPU kernels.

---

# 12. Unified `work_ids` convention

I would make subset execution a general property of GPU kernels.

For example:

```c
__kernel void batched_gemm_slots(
    ...
    __global const int* work_ids,
    int nwork
) {
    int iw = get_group_id(2);
    if (iw >= nwork) return;

    int sid = work_ids[iw];

    ...
}
```

Likewise one-WG/system kernels:

```c
int iw  = get_group_id(0);
int sid = work_ids[iw];
```

and per-atom kernels:

```text
global dimensions = (atom work, nwork)
sid = work_ids[get_group_id(1)]
```

The important invariant is:

```text
GPU memory address = physical_slot
launch domain       = compact logical work index
```

Then the same machinery handles:

```text
active SCC slots
retry slots
force slots
refill slots
finalization slots
```

without separate concepts proliferating through every kernel.

---

# 13. Keep an active mask inside each SCC chunk

`work_ids` does not make the existing active mask obsolete.

At the beginning of a chunk:

```text
scc_ids = [3,7,9,10,...]
```

During the eight SCC iterations, slot 9 might converge after iteration 2.

We do NOT want six more iterations for it.

Therefore:

```text
work_ids -> removes systems between chunks
active[] -> disables systems during the current chunk
```

So the launch domain stays constant for the eight-iteration chunk, while kernels still inspect:

```c
active[sid]
```

and stop work for newly converged replicas.

At the chunk boundary Rust compacts the list again.

This is simple and robust.

---

# 14. Adaptive chunk size can come later

Initially:

```text
chunk = 8
```

everywhere.

Later something like:

```text
many jobs remaining:
    chunk = 8 or 12

queue nearly empty:
    chunk = 4

only tail jobs remain:
    chunk = 2
```

could reduce wasted iterations and retire slots faster.

But I would not implement this until the fixed-size scheduler works.

The exact choice is unlikely to matter nearly as much as removing hundreds of dead workgroups.

---

# 15. Warm starting newly loaded jobs

There are two levels.

### First implementation

New job gets:

```text
q = q0
normal cold electronic initialization
```

This makes scheduler correctness independent of scan ordering.

### Later optimization

For structured scans, initialize a new job from a nearby completed scan point:

```text
q_new <- q_neighbor
C_new <- C_neighbor
μ_new <- μ_neighbor
```

provided the basis/topology is identical.

But this requires copying an N² basis between physical slots.

That may still be cheap compared with SCC, but should be a separate optimization.

Do not make scheduler correctness depend on it.

---

# 16. Results must be indexed by JobId, never SlotId

This is an important source of bugs.

Device state:

```text
indexed by SlotId
```

Long-lived output:

```text
indexed by JobId
```

When a job finishes:

```rust
results[job_id] = read_or_copy_result(slot_id);
slot_job[slot_id] = None;
```

Only then can the slot be reused.

For large scans it may also be preferable to have device final-result arrays indexed by `JobId`, so a finalization kernel writes directly:

```c
result[job_id] = ...
```

using a small:

```text
slot_job_dev[slot]
```

mapping.

Then energies, convergence counters and scan output need not be shuffled on the host.

---

# 17. Failure handling

A slot should not automatically become free simply because SCC stopped.

Distinguish:

```text
Converged
Plateau
Failed
```

exactly as the current code already does.

Possible state transitions:

```text
SCC -> Converged -> Finalize
SCC -> Plateau   -> policy decision
SCC -> Failed    -> retry
```

Retries can simply form another compact list:

```text
retry_ids[]
```

The current implementation already has explicit masked retry semantics and deliberately keeps failed replica recovery separate from ordinary SCC. That concept maps naturally onto slot scheduling.

---

# 18. Proposed implementation order

## Phase A — active-ID plumbing

Keep `batch=400`.

Add:

```text
work_ids buffer
n_work
```

Convert the important SCC kernels:

```text
GEMMs
Jacobi
Hscc
population / density path
DIIS
Fermi
```

from

```text
logical replica == physical replica
```

to:

```text
sid = work_ids[logical replica]
```

Run SCC in chunks.

After every chunk build a compact active list.

This alone fixes the long convergence tail.

---

## Phase B — benchmark saturation vs physical slot count

Benchmark:

```text
S = 32,64,96,128,192,256,400
```

for at least:

```text
GC / N≈86
DTH / N≈246
```

Measure:

```text
systems/s
SCC replica-iterations/s
Jacobi device time
GEMM device time
GPU occupancy if available
```

Choose a default slot count based on throughput rather than intuition.

---

## Phase C — queue + refill

Allow:

```text
total_jobs > n_slots
```

Implement:

```text
pending JobId queue
slot_job mapping
refill_slots()
```

Initially support only fixed single-point geometries.

This gives the full cluster-scheduler behavior:

```text
pending -> running -> complete
```

on one GPU.

---

## Phase D — asynchronous FIRE trajectories

Make geometry assembly and force evaluation accept `work_ids`.

Each physical slot now executes:

```text
SCC -> forces -> FIRE -> new geometry -> SCC
```

independently.

Retire and refill a slot only after force convergence of the whole trajectory.

---

# 19. What I would specifically NOT do

Do not:

```text
copy complete matrices when compacting
```

Only compact integer slot IDs.

Do not:

```text
implement a global atomic GPU work queue
```

Rust scheduling every several SCC iterations is vastly simpler and the overhead is negligible.

Do not:

```text
fuse SCC + Jacobi + Fermi + DIIS + FIRE into a persistent megakernel
```

They have incompatible workgroup and memory requirements.

Do not:

```text
remove the row/column GEMM workgroup dimensions
```

They are useful parallelism.

Do not:

```text
hard-code 100 slots
```

Measure the saturation point.

Do not:

```text
migrate an unfinished relaxation between physical slots
```

A slot should own the whole trajectory until completion.

---

# 20. Target architecture

The final conceptual structure should look like:

```text
                  GLOBAL JOB QUEUE
              400 / 1000 / ... jobs
                         |
                         v
              +-------------------+
              |   Rust scheduler  |
              +-------------------+
                |       |       |
                |       |       |
           SCC ids   force ids  refill ids
                |       |       |
                v       v       v

        ┌───────────────────────────────┐
        │       GPU physical slots      │
        │                               │
        │ slot 0   -> job 217           │
        │ slot 1   -> job 105           │
        │ slot 2   -> job 331           │
        │ ...                           │
        │ slot S-1 -> job 181           │
        └───────────────────────────────┘

Each slot permanently owns:

    coordinates
    H0, S, G
    q / DIIS
    C / SC / eig state
    D/W when valid
    FIRE x/v/dt/alpha
    diagnostics

until that JOB finishes.
```

The GPU sees only compact lists of useful slots.

That is essentially a miniature HPC scheduler, except scheduling latency is on the order of one SCC chunk and the "compute nodes" are persistent replicas occupying regions of a single GPU's memory.

The important design principle is:

> **schedule systems, not matrix tiles; keep matrices resident, and redirect the replica index through a compact slot list.**

That gives load balancing without sacrificing the efficient tiled dense linear algebra that the DFTB solver actually needs.

The part I would emphasize to the coding agent is that there are really **two optimizations**, not one. `active_ids` is almost a mechanical change and should be implemented first; **refillable slots** then let 400/4000 jobs flow through a much smaller persistent GPU working set.

For the FIRE case, I think the cleanest final architecture is even nicer than for simple SCC: a slot represents an entire *trajectory*, so its \(q,C,\mathrm{SC}\), DIIS history and FIRE velocity remain warm while that trajectory advances. The current code already has the relevant per-replica device-resident FIRE state, so this is compatible with the direction of the implementation rather than a rewrite.

One important performance caveat: **100 slots may be too few for the Jacobi stage even if it is plenty for GEMM.** I would explicitly benchmark 64/96/128/192/256 slots before choosing the pool size. For N≈246, GEMM will saturate with very few systems because each system creates hundreds of matrix-tile WGs; one-WG-per-system Jacobi has the opposite occupancy requirement.


---




# USER

OK, but I have problem with this, what means matrix tile index row/colum? I understand our system (e.g. 86 atoms or 286 atoms) is bigger than the tile (16,32 threads) but that does not mean we should run the jacobi solver paralleized over multiple workgroups (do we?) what I wanted is simple kernel desin which use e.g. 16-64 threads and 16-64 shared memory slots to store some working tile of the system, and iterate internally over these tiles, so it should be serialized nor paralleized over tiles. This allows to efficietly use all threads and occupy the compute unit which multiple systems ( as currnetly the local memory requirement limit the numer of workgroups per compute unit.)

what you think? do you agree that the 3D krnel is stupid idea now? I do not see how it helsp and who invented it? 





---


# Chat GPT 5.6 sol

Yes — after looking directly at the three Jacobi kernels you uploaded, I agree with the architecture you have in mind. But there is an important correction:

**The current Jacobi solver is already one workgroup per system. The 3-D `(row_tile, col_tile, replica)` organization is the GEMM kernel, not Jacobi.**

Your `gpu_block_jacobi.cl` explicitly says “one-WG block Jacobi eigensolver,” and the Rust launcher uses

```rust
.global_work_size(batch * wg)
.local_work_size(wg)
```

so there is exactly one WG per system.

Likewise, `tiled_jacobi_batched` is described in the Rust code as “One workgroup per system,” with A/V in global memory and only the working pivot in local memory.

So I would separate the issue into two very different kernels.

### 1. Jacobi: your proposed architecture is right

For Jacobi I think the clean target is essentially:

```text
global workgroups:
    one WG per active system

WG:
    ~32 / 64 / maybe 128 threads

inside WG:
    for sweep
        for tile/block pair
            load small working tile from global -> local
            solve/update tile cooperatively
            write result
```

So for 400 active systems:

```text
WG 0   -> system 0
WG 1   -> system 1
WG 2   -> system 2
...
WG 399 -> system 399
```

And with the scheduler:

```c
int iw  = get_group_id(0);
int sid = active_ids[iw];
```

That's it.

There should **not** be:

```text
WG(system, block_row, block_col)
```

for Jacobi.

That would create exactly the problems you're worried about: different workgroups working on the same eigensystem, synchronization between block updates, global barriers that OpenCL doesn't provide within one kernel, and a much more complicated scheduler.

Your current block kernel actually already serializes the block pairs internally:

```c
for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
    for (int bp = 0; bp < nb; bp++) {
        for (int bq = bp + 1; bq < nb; bq++) {
            ...
        }
    }
}
```

That is exactly the conceptual organization you want.

---

## 2. But I think you are pointing at another real problem

The current **one-WG Jacobi is still using too many threads per system**.

For `gpu_block_jacobi.cl` the Rust code deliberately chooses

```text
WG = ceil(N/32)*32
```

therefore:

```text
N = 86  -> WG = 96
N = 246 -> WG = 256
```

because the implementation assumes approximately one thread per matrix row.

This is where I think your criticism is interesting.

There is no fundamental reason that an N=246 matrix must have 256 threads.

You could perfectly well have:

```text
WG = 64
```

and simply:

```c
for (int row = lid; row < n; row += 64) {
    ...
}
```

Thus each thread handles about four rows for N=246.

Or even:

```text
WG = 32
```

where each thread handles ~8 rows.

For your application, where you have **hundreds of independent matrices**, this can be a much better trade:

```text
less parallelism inside one system
            ↓
more systems resident simultaneously
            ↓
better GPU throughput across replicas
```

That is precisely the SIMD/SIMT strategy I think we should investigate.

---

## Your mental picture is therefore basically right

I would visualize one SM as something like:

```text
SM
┌───────────────────────────────────┐
│ WG: system 17                     │
│ 64 threads                        │
│ small local working tile          │
├───────────────────────────────────┤
│ WG: system 83                     │
│ 64 threads                        │
│ small local working tile          │
├───────────────────────────────────┤
│ WG: system 124                    │
│ 64 threads                        │
│ small local working tile          │
├───────────────────────────────────┤
│ WG: system 207                    │
│ 64 threads                        │
│ small local working tile          │
└───────────────────────────────────┘
```

rather than:

```text
SM
┌───────────────────────────────────┐
│ one giant WG for system 17        │
│ 256/512 threads                   │
│ lots of local scratch             │
└───────────────────────────────────┘
```

The GPU-level parallelism comes mainly from **replicas**, not from trying to parallelize one 246×246 eigensystem as aggressively as possible.

I think that principle fits your workload extremely well.

---

# There are actually two Jacobi implementations worth distinguishing

This was useful to notice in the uploaded code.

### `block_jacobi_1wg`

It stores a complete compound pivot and its transformation:

```c
__local float P[PLD * PB];
__local float U[PLD * PB];
```

For B=16:

```text
PB = 32

P : 32×33 floats ≈ 4.1 kB
U : 32×33 floats ≈ 4.1 kB
```

so already about **8.3 kB local memory**, independently of whether the WG has 64 or 256 threads.

This is important:

> Merely reducing WG 256 → WG 64 will not reduce most of the local-memory footprint of this kernel.

It reduces registers/thread pressure and the reduction arrays, but `P+U` remain.

So if local memory is what limits the number of resident systems, we need to consider **both**

```text
number of threads/system
```

and

```text
size of Jacobi working tile
```

---

### `jacobi_cyclic_global_batched`

This one is actually even closer to what you're describing.

It keeps A and V in global memory and only stores rotation information in local memory:

```c
__local jrot_t rot_c[128];
__local jrot_t rot_s[128];
__local int rot_p[128];
__local int rot_q[128];
```

Then threads stream over the matrix:

```c
for (int idx = lid; idx < n*n; idx += lsz)
```

and later over the rotation/update work:

```c
for (int it = lid; it < nblk+nvt; it += lsz)
```

That means **this kernel fundamentally does not require WG≈N**.

It could naturally be tested with:

```text
WG32
WG64
WG96
WG128
WG256
```

without changing the mathematical structure.

Ironically, the Rust wrapper currently chooses up to **WG512**, because a standalone benchmark found WG512 1.55× faster than 256 for an individual/batched test. The source says exactly that.

But that benchmark may optimize the wrong objective.

We don't care about:

> minimum latency for one eigensystem.

We care about:

> maximum eigensystems/second for 100–400 simultaneously active replicas.

Those are not necessarily the same optimum.

---

# I would therefore test something quite radical but simple

Take the **direct global Jacobi** first because its local-memory requirements are already small.

Compile versions:

```text
WG = 32
WG = 64
WG = 96
WG = 128
WG = 256
WG = 512
```

and benchmark **400 systems**, not one system.

Measure:

```text
systems / second
replica-Jacobi-solves / second
actual SCC wall time
```

My suspicion is that the optimum for batch400 could move dramatically downward from WG512.

Especially because much of the work is of the form:

```c
for (work = lid; work < total_work; work += WG)
```

so reducing WG does not change the algorithm at all. Each thread just performs more loop iterations.

---

# There is one caveat to your proposed 16–64-thread WG

I would avoid **16 threads** on NVIDIA.

A warp is 32 threads, so a 16-thread WG still occupies a warp with half the lanes disabled.

Start with:

```text
32
64
128
```

32 or 64 is closest to your philosophy.

---

# And there is another very important interaction with the scheduler

Suppose RTX 3090 has ~82 SMs.

If you choose:

```text
100 slots
64 threads/slot = 2 warps/slot
```

then you only have:

```text
100 workgroups / 82 SMs ≈ 1.2 workgroups per SM
```

or only roughly:

```text
2.4 active warps/SM
```

during Jacobi.

That is **far too little** to hide latency.

So if we shrink each system to WG64, the scheduler probably should **not have only 100 physical slots**.

You might instead want:

```text
256–512 active slots
```

or at least whatever keeps perhaps several WGs resident per SM.

This is why I would not decide the slot number independently of the Jacobi WG size.

They are coupled parameters:

$$
\boxed{
N_{\rm slots}\times N_{\rm warps/WG}
}
$$

determines the amount of independent GPU work available.

For example:

```text
WG64, 400 systems:
    800 warps total
    ≈ 9.8 warps/SM

WG128, 400 systems:
    1600 warps
    ≈ 19.5 warps/SM

WG256, 400 systems:
    3200 warps
    ≈ 39 warps/SM
```

ignoring local/register occupancy limits.

So perhaps:

```text
WG64 + 400/500 slots
```

beats

```text
WG256 + 100 slots
```

even though an individual matrix takes longer.

That's exactly the throughput benchmark we need.

---

# So what is the “3-D kernel”?

This is where I caused confusion in my previous answer.

The 3-D code is the tiled GEMM:

```text
group_id(0) = output matrix row tile
group_id(1) = output matrix column tile
group_id(2) = system
```

For example, computing

$$
C=A B
$$

with N=246 and 16×16 tiles creates:

```text
16 × 16 = 256 workgroups/system
```

because each WG independently computes one 16×16 part of the output matrix.

For **GEMM**, this is completely normal and sensible. There is no dependency between different output tiles:

```text
C00 C01 C02 ...
C10 C11 C12 ...
...
```

They can all be computed concurrently.

So I would **not call the 3-D GEMM stupid**.

It is the right organization for GEMM.

What would be stupid, in my view, is extending this idea to the Jacobi solver:

```text
(row block, col block, system)
```

and trying to have many WGs cooperate on one Jacobi eigensystem.

Your current Jacobi code does **not** do that.

---

# Who introduced it?

The current source itself says the eigensolver module was owned by `"Agent_4 (Wave 2)"`, and explicitly documents its architecture as **one workgroup per system**.

The Git history for the generic matrix-operations file is committed under your GitHub authorship, so I can't reliably attribute the original 3-D batched-GEMM implementation to a particular coding agent from Git history alone.

But importantly, whoever wrote it was not necessarily making a mistake: **3-D is appropriate for the GEMM.**

---

## So I would revise our target architecture to this

```text
                     GPU
                      |
       +--------------+--------------+
       |              |              |
    system 0       system 1       system 2
    one WG          one WG          one WG
    32–128 thr      32–128 thr      32–128 thr
       |              |              |
       v              v              v
   iterate tiles   iterate tiles   iterate tiles
   internally      internally      internally
```

For **Jacobi**:

$$
\boxed{\text{one WG = one system}}
$$

and the WG internally walks through the whole matrix.

For **GEMM**:

$$
\boxed{\text{many WGs = one system}}
$$

because independent output tiles genuinely can be calculated independently.

I think the really interesting next experiment is therefore **not rewriting the Jacobi decomposition at all**. First take `jacobi_cyclic_global_batched`, compile WG32/64/128/256/512, and benchmark batch400. That directly tests your hypothesis with almost no algorithmic change.







---


# USER

some reflection and discusion with my coding agent:

\---

# swe-2

Let me read the full document:

The doc has a solid core idea but it also overlaps with what we already measured. Let me reflect honestly, then update the roadmap.

## Reflection on `Dense_Jacobi_Eigen_Tiling_Opt.md`

**What's right:**
- Jacobi is already 1-WG-per-system — the 3D grid belongs to the GEMM and is correct there (I confirmed this in the design doc).
- The block kernel's P+U local footprint (~8–35 KB) doesn't shrink with WG — GPT correctly notes that shrinking WG256→64 won't buy co-residency for the *block* kernel.
- The **coupling insight is the real value**: `N_slots × warps/WG` sets total parallelism. Once the slot scheduler runs S≈64–128 slots, there's no WG queue to hide latency behind — resident occupancy becomes the binding constraint. This couples slot sizing to Jacobi WG size — genuinely important for the scheduler design.

**What it misses — we already measured its headline hypothesis:** the WG sweep at **batch=400** (the throughput regime, not single-matrix latency) showed runtime *monotonically decreasing* with WG — WG512 fastest at both N=86 and N=246. The "small WG → more co-resident systems → better throughput" prediction fails in the queued regime: 400 queued WGs already supply parallelism, so per-WG lane count wins. Smaller WGs just make each thread loop more.

**What's actually new and worth trying:**
1. Block kernel currently *requires* WG≥n (thread-per-row). Strided row loops (`row=lid; row<n; row+=lsz`) would enable WG<n — worth testing at n=246, especially in the small-slot-count scheduler regime where occupancy is binding.
2. Joint (S × WG) measurement for the scheduler — not independent parameters.
3. Local-mem-driven co-residency curve for block: WGs/SM vs B — we measured B16=9.7K, B24=20.2K, B32=34.8K; B32 won anyway, so serial-pivot reduction dominates, but the curve should be documented.

Let me add these to the roadmap:

I'll add the reflection + new experiments as open items in T08 (it's the eigensolver-tuning ticket):

Also worth linking the scheduler design doc from T06 since that's where it now lives:

Done — reflection recorded and roadmap updated. Summary:

## Reflection on the GPT doc

**It partially re-litigates settled questions:** the headline suggestion — small WG (32–128) for more co-resident systems — was already tested at **batch=400** and falsified: WG512 wins monotonically (34.9→25.3 ms at N=86). In the queued regime the 400-WG queue already supplies parallelism; shrinking WG only makes each thread loop more. GPT's doc doesn't know we ran this.

**Its real contributions, now on the roadmap as T08b:**

- **Block kernel WG<n via strided rows** — `block_jacobi_1wg` hard-requires WG≥n (thread-per-row). Converting to `row=lid; row<n; row+=lsz` unlocks WG∈{64,96,128} — which matters specifically in the **small-slot-count scheduler regime** (S≈64–128 slots, no WG queue) where falsified-in-queued doesn't apply.
- **Joint slots×WG co-optimization** — `N_slots × warps/WG` sets parallelism; these are coupled parameters to sweep together once the scheduler exists, not independent knobs.
- **Co-residency curve vs B** — record WGs/SM at measured local sizes (9.7/20.2/34.8 KB) so the optimum is re-derivable on other GPUs rather than re-litigated.
- Minor: `rot[128]` cap bounds direct kernel to n≤256; verify no other kernel has a hidden WG≈N assumption.

Also linked the new design doc into T06 so the ticket ↔ v2 design connection is explicit.

Want me to start Phase A of the scheduler (the `work_ids` plumbing refactor), or the block-kernel strided-row WG<n experiment first?

\---

# USER

ad "It partially re-litigates settled questions: the headline suggestion — small WG (32–128) for more co-resident systems — was already tested at batch=400 and falsified: WG512 wins monotonically (34.9→25.3 ms at N=86). In the queued regime the 400-WG queue already supplies parallelism; shrinking WG only makes each thread loop more. GPT's doc doesn't know we ran this."

but that perhaps point out so some other deep architectural problem with our whole system, we must be bottlenecked somewhere so we do not ever saturate the GPU with reall work... 

anyway what document I should give to GPT 5.6 to understand what test we already run and result concluding these other tile sises were worse ?

\---

Good question — and your instinct about a deeper bottleneck is probably right. Let me verify one thing in the kernel first, because I think I can explain *why* WG512 wins monotonically:

The kernel confirms your intuition — and I think I can name the deeper problem you're circling. Let me lay it out, then answer the doc question.

## Why WG512 wins monotonically — the real bottleneck

Look at what `jacobi_cyclic_global_batched` actually does per round: **every element of A and V is streamed through global memory**. Each of the ~85 rounds per sweep touches ~jpair² A-blocks + n×jpair V columns ≈ 30K floats ≈ 120KB of global traffic. Per sweep per system ≈ **10 MB**; batch=400 × ~5 sweeps ≈ **20 GB in 25.3 ms ≈ 790 GB/s** — that's ≈85% of an RTX 3090's DRAM peak.

**The kernel is bandwidth-bound on A/V streaming, not occupancy-bound.** Bigger WG = more outstanding memory requests per system = better memory-level parallelism — hence monotonic gain. The working set (59 KB/system × 400 = 24 MB) doesn't fit L2 (~6 MB), so every round goes to DRAM. This is exactly the "deep architectural problem" you sensed: we're not failing to saturate compute — we're failing to keep the data close. GPT's WG-shrink idea can't fix that; it reduces parallel loads.

**The fix is data residency, not WG sizing** — which is what your original intuition actually was ("16–64 shared-memory slots holding a working tile, iterate internally"). At n=86, A+V = 59 KB — **fits entirely in local memory**. A full-local Jacobi does all 85 rounds × all sweeps at local-memory bandwidth (~10–30× DRAM), touching DRAM only for load/store. That's a genuinely different kernel worth building — the real test of your idea.

Let me add this to the roadmap and write the GPT digest doc:

Now the paste-ready digest for GPT 5.6 — all the measurements in one place:

Done. To answer directly:

## Document for GPT 5.6

**`doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Measured_Facts_Jacobi_Sweeps.md`** — new paste-ready digest: kernel architecture (which kernel is which), the full WG sweep table, block-param sweep, resource queries, rejected variants, end-to-end SCC numbers, and the still-open hypotheses. Everything measured, nothing speculative.

## The deeper problem you sensed — confirmed by a roofline check

Reading the kernel made it concrete: `jacobi_cyclic_global_batched` keeps **A and V entirely in global memory** and re-streams *all* of them every one of ~85 rounds per sweep. That's ~10 MB/sweep/system → ~20 GB per batch-400 solve ≈ **790 GB/s ≈ 85% of DRAM peak**. The working set (24 MB) >> L2 (6 MB).

So we're not under-saturating the GPU — we're **bandwidth-bound on A/V streaming**. WG512 wins monotonically because more lanes = more outstanding memory requests = more memory-level parallelism; it has nothing to do with per-system latency or warp occupancy. And it means *no* WG/tile/workgroup sizing can move this — only **data residency** can.

**Your original instinct was the right frame** — "a small working tile held in shared memory, iterated internally" — GPT paraphrased it into a thread-count question, but it's really a *where-does-the-matrix-live* question. At n=86, **A+V = 59 KB fits entirely in `__local`**: a full-local Jacobi would run all sweeps at local bandwidth (~10–30× DRAM), touching global memory once for load and once for store. That, not WG sizing, is the real experiment — now the top item under T08b, along with an A-local/V-global hybrid for mid-n and strip-residency for the block kernel at n=246 (which is bandwidth-bound the same way).

Want me to implement the full-local Jacobi variant for n≤128 next?







---

# Chat GPT 5.6 sol

Yes. The new digest makes the situation much clearer, and I think your coding agent has identified the right *class* of problem. But I would modify the conclusion in two important ways.

The first is conceptual: **the GPU may actually be very well saturated already — just with memory traffic rather than useful floating-point work.** The measured WG sweep is quite strong evidence: at batch 400, performance improves monotonically all the way to WG512, from 34.9 to 25.3 ms for the warm-ish N=86 solve. So the earlier idea that WG64 would win because more systems could coexist was indeed wrong in this regime. 

The second is that I would not yet call the DRAM-bandwidth diagnosis *confirmed*. It is a very plausible source-level roofline argument, but there is an important cache/residency subtlety.

### The 10 MB/sweep estimate is actually sound

I checked the structure of `jacobi_cyclic_global_batched`, and the agent's arithmetic makes sense.

For N=86,

$$
j_{\rm pair}=43,\qquad j_{\rm round}=85.
$$

Per round the A update has

$$
43^2=1849
$$

2×2 blocks. Each active block roughly performs four loads + four stores:

$$
1849\times 8\times4
=59\,168\ {\rm bytes}.
$$

The V update has

$$
86\times43=3698
$$

pairs, each roughly two loads + two stores:

$$
3698\times4\times4
=59\,168\ {\rm bytes}.
$$

So astonishingly neatly,

$$
T_{\rm round}\approx118\,336\ {\rm B},
$$

and therefore

$$
85T_{\rm round}\approx10.1\ {\rm MB/sweep/system}.
$$

Five sweeps × 400 systems is indeed around **20 GB of logical read/write traffic per batched eigensolve**. At 25.3 ms that's about 0.8 TB/s of logical bandwidth, which is remarkably close to the RTX 3090's DRAM capability.

And the architecture description in your digest confirms why: A and V stay in global memory and are repeatedly re-streamed every Jacobi round. 

So yes: **this kernel has atrociously low arithmetic intensity.**

---

## But logical traffic ≠ DRAM traffic

This is the one part where I'd correct the agent.

Saying

> working set 24 MB > 6 MB L2, therefore every round goes to DRAM

is too simple.

The GPU is not actively processing all 400 systems simultaneously. With WG512, an Ampere CC 8.6 SM supports at most 48 resident warps and 1536 threads; depending on register usage, that means at most roughly three 512-thread WGs/SM. NVIDIA specifies 48 warps/SM and 100 KB shared memory/SM for CC 8.6. ([NVIDIA Docs][1])

Therefore the *simultaneously active* A+V working set is much smaller than 24 MB.

And there is another interesting effect:

```text
WG512 -> at most ~3 systems/SM
WG256 -> potentially ~6 systems/SM
WG128 -> potentially ~12 systems/SM
```

while one N=86 A+V pair is ~59 KB.

So larger workgroups may actually reduce **cache thrashing between systems**. WG512's advantage could therefore be some mixture of:

* more parallelism within each Jacobi round;
* fewer loop iterations/thread;
* better memory-level parallelism;
* fewer simultaneously resident systems competing for L1/L2;
* DRAM bandwidth.

That does **not** weaken the case for data residency. If anything it strengthens it. But I would call the present result:

> **strong evidence that global A/V traffic and cache locality dominate**

rather than proof that 790 GB/s physically crosses DRAM.

---

# The full-local experiment is definitely worth doing — but there is a trap

At N=86,

$$
2N^2(4{\rm B}) =
2\times86^2\times4
=59\,168\ {\rm B}
\approx57.8\ {\rm KiB}.
$$

So A+V alone indeed fit below Ampere's 99 KB maximum shared memory per block. NVIDIA specifies 100 KB/SM and 99 KB/block for CC 8.6, although allocations above 48 KB have special opt-in requirements in CUDA; you should verify what the NVIDIA OpenCL compiler allows for this kernel. ([NVIDIA Docs][2])

But A+V are not the entire local allocation.

Your measured direct kernel currently uses:

```text
no Fermi tail : 7.2 KB
with tail     : 16.4 KB
```

already. 

So roughly:

$$
L_{\rm full-local}
\approx
58 + 7
\approx65\ {\rm KiB}
$$

without the Fermi tail, and perhaps ~74 KiB with it.

That fits under 99 KB.

But:

$$
2\times65 > 100\ {\rm KiB}.
$$

Therefore **only one full-local eigensystem can reside on an SM.**

And this completely changes our earlier discussion about small WGs.

If the kernel uses ~65 KB shared memory, then changing:

```text
WG512 -> WG64
```

does **not** let eight systems coexist.

It remains:

```text
1 WG / SM
```

because shared memory is the limiting resource.

WG64 would therefore give you only two warps on the entire SM!

That would probably be terrible.

### For a full-local N=86 kernel, I would test large WGs

Something like:

```text
WG256
WG512
WG1024
```

rather than 32/64/128.

Since local memory already limits you to one system/SM, the goal becomes to populate that SM with enough warps from the **single resident system**.

That is a major change from my earlier reasoning.

---

# I actually like an intermediate experiment even more: A-local, V-global

This could be the sweet spot.

A alone costs:

$$
86^2\times4
\approx29.6\ {\rm KB}.
$$

Add the no-tail scratch:

$$
29.6+7.2\approx36.8\ {\rm KB/WG}.
$$

Now **two WGs/SM are comfortably possible**:

$$
2\times36.8\approx74\ {\rm KB}<100\ {\rm KB}.
$$

With WG512, that gives:

```text
2 systems/SM
× 16 warps/system
= 32 warps/SM
```

which is quite healthy.

At the same time, A generates approximately half of that repeated 10 MB/sweep traffic.

And A has additional reuse:

* rotation parameters read its diagonal/off-diagonal elements;
* sweep convergence reads A;
* V doesn't participate in convergence;
* after solving, you don't even need to write the whole transformed A back if only the eigenvalue diagonal is consumed.

So an **A-resident / V-streamed Jacobi** could potentially eliminate somewhat more than half of the expensive global-memory work while retaining two resident systems per SM.

Conceptually:

```text
at kernel start:
    global A -> local A       once

for all sweeps:
    rotation params from local A
    update local A
    update V in global
    offnorm from local A

at end:
    local diag(A) -> eig[]
```

There is no reason to keep writing A to global on every round.

That is a very attractive architecture.

---

## And here the previously rejected `NO_TAIL` becomes interesting again

Your test showed compiling out the Fermi tail halved local memory but changed current runtime by only ±3%. 

For the existing global kernel that means:

> no useful performance gain.

But for the **resident kernel**, it means something different:

> those 9 KB may determine whether the desired local-memory layout fits and how many WGs reside per SM.

So I'd happily separate Fermi again for the resident experiment. A tiny extra Fermi launch is insignificant if it changes the eigensolver from global-memory streaming to resident A/V.

---

# There is also an important consequence for N=246

Full A+V residency is impossible:

$$
2\times246^2\times4
\approx484\ {\rm KB}.
$$

Even A alone is ~242 KB.

So N=246 genuinely needs a block/streaming algorithm.

And here I am now **less enthusiastic about the agent's WG<n / strided-row experiment** than the agent is.

Your B32 block kernel already uses 34.8 KB local memory. 

On a 100-KB SM:

$$
2\times34.8=69.6\ {\rm KB}
$$

but

$$
3\times34.8=104.4\ {\rm KB}.
$$

So B32 is fundamentally capped at **two resident WGs/SM by shared memory**.

Changing:

```text
WG256 -> WG128
```

does not increase that from two to three.

It merely changes:

```text
2 × 8 warps = 16 warps/SM
```

into

```text
2 × 4 warps = 8 warps/SM.
```

That looks more likely to hurt than help.

And with a scheduler having only S=64–128 systems, shrinking WG cannot somehow manufacture more systems either. With S=64 there are simply only 64 workgroups globally.

So I would demote:

> block WG<n / strided rows

to a cheap control experiment, unless profiling shows registers currently restrict B32 to only one WG/SM.

The B32 result itself looks much more important: despite its 34.8 KB local footprint, reducing the number of serial pivot solves wins strongly. 

---

# What I would do next

I would now make the next experiments specifically discriminate between **DRAM/cache traffic**, **shared-memory residency**, and **scheduler occupancy**:

1. **Before changing code, sweep batch/slot count on the existing direct WG512 kernel**: S = 32, 64, 82, 96, 128, 164, 246, 400. This tells us directly how much of the impressive batch-400 throughput survives when the future scheduler only keeps ~100 systems alive.

2. **Implement A-local / V-global N=86.** Keep the same Jacobi mathematics. Load A once, never write intermediate A to global, stream only V. Prefer `NO_TAIL` initially. Benchmark WG256/512.

3. **Implement full A+V-local N=86.** First verify OpenCL allows the ~65–75 KB per-WG allocation. Since only one WG can then fit per SM, benchmark WG256/512/1024—not WG64.

4. Compare all three architectures using exactly the same warm/cold tests:
   `global/global`, `local/global`, `local/local`. Measure systems/s, sweeps, SCC wall time and accuracy.

5. **Then sweep S × WG**, but only for the winning resident architecture. The scheduler and eigensolver should be co-designed based on these measurements.

6. Leave N=246/B32 alone until this establishes the principle. N=86 is the perfect experiment because the whole electronic state is just barely small enough to expose the residency tradeoff cleanly.

The end-to-end motivation is strong because Jacobi is now reported as about **80% of device time** after density/snormalization/Mulliken were removed, while GC has a huge straggler tail—median convergence around 56 iterations but the batch running to 100. 

So there are now two almost orthogonal optimizations:

$$
\boxed{\text{scheduler} \rightarrow \text{stop wasting work on finished systems}}
$$

and

$$
\boxed{\text{resident Jacobi} \rightarrow \text{stop moving A/V through the memory hierarchy}}
$$

I would pursue both.

The most interesting correction to our previous thinking is that **your original “small local working set” intuition was right, but “small number of threads” does not necessarily follow from it**. If the local working set itself forces one WG/SM, you actually want a fairly *large* WG so that the one resident system contains enough warps to use the SM.

[1]: https://docs.nvidia.com/cuda/ampere-tuning-guide/?utm_source=chatgpt.com "1. NVIDIA Ampere GPU Architecture Tuning Guide — Ampere Tuning Guide 13.4 documentation"
[2]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html?utm_source=chatgpt.com "5.1. Compute Capabilities — CUDA Programming Guide"





---

# USER

so based on these ideas can you suggest and write optimal kernel architecture for the jacobi solver which would optimize both global memory trafic and compute-unit occupancy (i.e avoid registly and local memory spliing/trashing, so we do not limit number of WGs on the compute unit with unnecessary requirement? We are not even sure where it is worth to use shared local memory and where it is better to rely on automatic GPU cache ...


---



# Chat GPT 5.6 sol

Yes. I think there is a substantially better architecture than either the current global/global direct Jacobi or simply putting both \(A\) and \(V\) into `__local`.

The central observation is:

$$
\boxed{\text{Jacobi rotations are determined entirely by }A.}
$$

There is **no mathematical reason to stream the eigenvector matrix \(V\equiv C\) through global memory after every Brent–Luk round**. We can diagonalize \(A\), record the rotations compactly, and apply the accumulated rotations to \(C\) afterward while each row tile of \(C\) remains on-chip.

That gives us much finer control over the occupancy/memory tradeoff.

# Proposed GPU Jacobi Architecture: Resident A + Deferred Eigenvector Rotation

## 1. Design objective

The Jacobi solver should optimize three resources simultaneously:

$$
\text{global-memory traffic},
\qquad
\text{local/shared-memory per WG},
\qquad
\text{registers per thread}.
$$

Optimizing only one of these is misleading.

The current direct solver already uses one workgroup per system, but both \(A\) and \(V\) live in global memory and are re-streamed through every Jacobi round. The measured WG sweep shows that at batch=400 a larger WG monotonically wins, with WG512 giving 25.3 ms versus 34.9 ms for WG64 at \(N=86\). This strongly suggests that reducing threads does not help while the kernel retains its present memory architecture. 

After the other SCC bottlenecks were removed, Jacobi accounts for about 80% of device time, so it is now worth changing the architecture rather than merely tuning its WG. 

The design principle should be:

> **Use explicit `__local` memory only for data having strong temporal reuse inside a workgroup. Let cache handle data that is streamed or used only briefly. Never spend shared memory merely because it is faster than global memory.**

On Ampere CC 8.6, an SM has 100 KB shared memory, 64K 32-bit registers and at most 48 resident warps. The L1/texture cache and shared memory also share a unified 128 KB on-chip resource, so allocating more shared memory can reduce cache capacity. ([NVIDIA Docs][1])

This means explicit local memory has an opportunity cost.

---

# 2. Separate diagonalization from eigenvector transformation

For one SCC iteration we have an orthogonal-basis Hamiltonian

$$
A=C^THC.
$$

Jacobi generates a sequence of rotations

$$
G_1,G_2,\ldots,G_m
$$

such that

$$
A_{\rm diag}
=
G_m^T\cdots G_2^T G_1^T
A
G_1G_2\cdots G_m.
$$

The eigenvectors only need the total right transformation

$$
C_{\rm new}
=
C_{\rm old}
G_1G_2\cdots G_m.
$$

Critically, the \(G_i\) depend only on \(A\).

Therefore the current operation

```text
for every Jacobi round:
    calculate rotations from A
    update A
    update entire C
```

should become

```text
PHASE 1:
    keep A resident
    calculate rotations
    update A
    record rotations

PHASE 2:
    load a tile of C once
    apply the complete recorded rotation sequence on-chip
    store C once
```

This changes the memory structure fundamentally.

The present direct algorithm has roughly

$$
O(N^3 N_{\rm sweep})
$$

global-memory traffic because A and C are repeatedly streamed.

The proposed solver has approximately

$$
O(N^2)
$$

traffic for A,

$$
O(N^2)
$$

traffic for C,

plus the much more compact rotation log.

The arithmetic complexity remains essentially Jacobi's original complexity. We are moving arithmetic relative to memory, not changing the eigensolver.

---

# 3. Kernel A: `jacobi_localA_rotlog`

One workgroup handles one system.

```c
sid = work_ids[get_group_id(0)];
```

The input matrix \(A\) is loaded once from global memory into

```c
__local float lA[N][N+1];
```

with the extra column avoiding pathological shared-memory bank strides.

For \(N=86\),

$$
86\times87\times4
=
29\,928~{\rm B}
\approx29.2~{\rm KiB}.
$$

This is dramatically better than storing both A and C:

$$
2\times29.2\approx58.4~{\rm KiB}.
$$

The WG then performs all Jacobi sweeps entirely on `lA`.

Conceptually:

```c
load A -> lA;

for (sweep=0; sweep<MAX_SWEEPS; sweep++) {

    for (round=0; round<N-1; round++) {

        // ~N/2 independent rotations
        compute c,s from lA[p,p], lA[q,q], lA[p,q];

        store rotation parameter in rotlog;

        barrier();

        // G^T A G entirely in shared memory
        update lA;

        barrier();
    }

    compute offnorm(lA);

    if (converged/stalled)
        break;
}

write eig[k] = lA[k,k];
write nsweeps/status;
```

There is **no C/V update in this kernel**.

There is also normally no reason to write the final full A matrix back to global memory. The SCC path consumes the eigenvalues while the next projection overwrites the projected Hamiltonian anyway.

Thus the large matrix traffic is essentially:

```text
A global -> local       once
diag(A) local -> eig    once
```

---

# 4. Rotation log

The Brent–Luk pair schedule is deterministic.

Therefore we do not need to store

```text
p
q
c
s
```

for every rotation.

The transform kernel can reconstruct `(p,q)` from

```text
round
pair_index
```

using the same Brent–Luk schedule.

Initially I would store only

```c
double2(c,s)
```

for every performed rotation.

That preserves the exact rotation parameters used by the diagonalization and gives the cleanest correctness reference.

For \(N=86\):

$$
85\times43=3655
$$

rotations per sweep.

A `double2` log therefore costs

$$
3655\times16
\approx58.5~{\rm KB/sweep/system}.
$$

That is small compared with repeatedly streaming A+C through 85 rounds.

After parity is established, test storing just

```c
double t
```

where

$$
c=(1+t^2)^{-1/2},
\qquad
s=tc.
$$

This halves the log size.

Only after that should `float2(c,s)` or `float t` be considered.

The measured results already show that indiscriminately moving Jacobi rotation arithmetic to FP32 gives substantially worse eigenvalue parity, so the reference implementation should preserve the currently accepted rotation precision. 

---

# 5. Kernel B: `apply_jacobi_rotlog`

After diagonalization, applying the rotations to different **rows of C is completely independent**.

This is where using more than one WG per system becomes both safe and useful.

Launch approximately

```text
(system, row_tile)
```

rather than one WG per system.

For example,

```text
RTILE = 16
N = 86
```

gives six row-tile WGs per system.

Each WG allocates only

```c
__local float cTile[RTILE][N+1];
```

which at N=86 costs

$$
16\times87\times4
\approx5.4~{\rm KiB}.
$$

Algorithm:

```c
load 16 complete rows of C
    global -> local
    coalesced

for each sweep actually executed:
    for round=0..N-2:

        load c,s for this round into small local rotation arrays
        barrier

        for each local row
            for each independent pair (p,q)
                cp = cTile[row][p]
                cq = cTile[row][q]

                cTile[row][p] = c*cp - s*cq
                cTile[row][q] = s*cp + c*cq

        barrier

store 16 complete rows
    local -> global
    coalesced
```

The C matrix therefore crosses global memory only once in and once out for the complete eigensolve.

This kernel has excellent occupancy:

* ~5–8 KB shared memory/WG,
* almost no large private state,
* many row-tile WGs,
* hundreds of independent WGs even if the scheduler has only 64–128 active systems.

It therefore solves the scheduler-occupancy problem differently from the diagonalization kernel.

---

# 6. Why this is better than full A+C local

A straightforward full-local N=86 kernel needs roughly

$$
58-65~{\rm KiB/WG}
$$

after scratch is included.

On an RTX 3090-class CC8.6 SM with 100 KB shared memory, that means:

$$
1\ {\rm WG/SM}.
$$

A WG512 then provides only 16 resident warps on an SM whose architectural maximum is 48.

The proposed **A-only resident** kernel can plausibly fit below approximately 32 KB/WG.

That is a strategically important threshold.

For WG512:

### Three-WG target

Three resident WG512 blocks require

$$
3\times512=1536
$$

threads, exactly the CC8.6 per-SM thread limit.

Shared-memory requirement:

$$
L_{\rm WG}\lesssim100/3
\approx33~{\rm KB}.
$$

Register requirement:

$$
R_{\rm thread}
\lesssim
\frac{65536}{3\times512}
\approx42.
$$

Therefore the desired target for the A-only kernel is very concrete:

```text
WG = 512
local/WG < ~32 KB
registers/thread <= ~40
no spills
```

If we achieve that, an SM can potentially host three independent eigensystems simultaneously:

```text
3 WG × 16 warps = 48 warps/SM
```

which is full warp occupancy on CC8.6.

That is almost an ideal mapping for N≈86.

Even if registers restrict us to two WGs:

```text
2 × 16 = 32 warps/SM
```

is still much healthier than the one-WG full-A+C design.

---

# 7. Therefore remove unnecessary local scratch aggressively

The A-resident kernel should not contain the baggage of the present monolithic direct kernel.

In particular, compile Fermi completely out of this kernel.

The current experiment found that removing the Fermi tail did not accelerate the existing global-memory solver, despite reducing local memory from 16.4 KB to 7.2 KB. 

For the new architecture that same memory saving is valuable because it can change the number of resident eigensolver WGs from two to three.

Also remove WG-sized reduction arrays wherever possible.

On NVIDIA, preferably use subgroup/warp reductions:

```text
warp reduction
       ↓
one partial scalar per warp in shared memory
       ↓
final warp reduction
```

For WG512 this needs perhaps 16 partial floats rather than

```c
float reduce[512];
float reduce2[512];
double dred[512];
...
```

This can save several kilobytes of shared memory.

If subgroup operations are unavailable, use one shared reduction buffer and aggressively reuse it between operations rather than allocating separate arrays.

---

# 8. Registers must be treated as a first-class design constraint

`CL_KERNEL_PRIVATE_MEM_SIZE == 0` does not establish that register pressure is harmless.

For NVIDIA OpenCL, compile the specialized kernel with

```text
-cl-nv-verbose
```

and parse the compiler build log.

NVIDIA documents that this reports register, local and shared-memory usage. ([NVIDIA Developer Downloads][2])

For every kernel specialization record:

```text
N
WG
registers/thread
shared bytes/WG
private spill bytes
predicted max WGs/SM
measured time
```

Do not guess occupancy from shared memory alone.

For the A-local WG512 kernel the important register thresholds are approximately:

| Registers/thread | Register-limited WG/SM |
| ---------------: | ---------------------: |
|              ≤42 |                      3 |
|            43–64 |                      2 |
|              >64 |                      1 |

assuming other limits do not bind.

The kernel should therefore avoid:

```c
float x[64];
float y[64];
```

or other dynamically indexed private arrays.

Those are exactly the kinds of arrays GPU compilers may place in thread-local memory when register allocation becomes problematic; thread-local spills physically live in device memory and have global-memory-like latency. ([NVIDIA Docs][3])

Per-thread state should preferably consist of scalar indices, c/s/t values and perhaps 2–4 independent FMA accumulators.

---

# 9. Three eigensolver regimes

I would not use one kernel architecture for all N.

## Regime I — very small N: full A+C resident

Roughly:

$$
N\lesssim64
$$

although the actual crossover should be measured.

The existing full-local architecture is appropriate:

```text
one WG / system
A local
C local
rotations local
all sweeps internal
global load/store only at boundaries
```

At N=64 the A+C footprint is still small enough that multiple WGs can reside on an SM.

There is no need for a rotation log here because C itself can remain resident cheaply.

---

## Regime II — medium N: A resident + deferred C transformation

Likely around

$$
64 < N \lesssim 110\text{–}130.
$$

This should be the important GC/N86 path.

Use:

```text
Kernel A:
    one WG/system
    A local
    C absent
    rotation log global
    all Jacobi sweeps internal

Kernel B:
    several independent row-tile WGs/system
    C tile local
    rotation log read/cache
```

This is the architecture I expect to be optimal around N=86.

For N=86, A alone is ~29 KB whereas A+C is ~59 KB. This is precisely the range where storing the second matrix changes occupancy qualitatively.

---

## Regime III — large N: block Jacobi

For N≈246, even A alone is

$$
246^2\times4
\approx242~{\rm KB},
$$

so full A residency is impossible.

The current B32 block architecture is therefore conceptually appropriate. The measured sweep already found B32/IMAX1 dramatically better than the original B16/IMAX12 design, despite B32 using more local memory. 

But I would change how its global strips are processed.

---

# 10. Large-N block kernel: P+U local, strips tiled

The things that definitely belong in `__local` are:

```text
pivot matrix P
pivot transformation U
rotation parameters
small reduction scratch
```

because P and U are reused intensively during the local pivot solve.

The things that should **not** be kept there permanently are entire A/V panels.

They are too large and would reduce B32 from approximately two resident WGs/SM to one.

However, the current block kernel has another problem: each row thread uses private arrays

```c
float x[PB];
float xv[PB];
```

and individually reads its strip.

Instead replace this with a small cooperative strip tile:

```c
__local float strip[RTILE][PB+1];
```

For example:

```text
PB = 64
RTILE = 8
```

costs only

$$
8\times65\times4
\approx2~{\rm KB}.
$$

The operation becomes a tiny tiled GEMM:

```text
for row tile r0:

    cooperative coalesced load:
        A[r0:r0+RTILE, pivot_columns] -> strip

    barrier

    strip <- strip × U

    barrier

    coalesced store

repeat using SAME strip buffer for V
```

Do not allocate separate A and V strip buffers.

Reuse the same `strip[]`.

Benefits:

1. global reads/writes become coalesced;
2. no `float x[64]` private vector per thread;
3. register pressure drops drastically;
4. only ~2–4 KB extra local memory;
5. B32 can probably still retain two WGs/SM;
6. the operation maps naturally onto the same tiled-FMA strategy already working well in the GEMM kernels.

This is much preferable to storing a complete \(N\times B\) panel explicitly in shared memory.

---

# 11. The block pivot gather should also become cooperative

Current row-owned pivot gathers encourage strided warp accesses.

Instead flatten the compound pivot:

```c
for (int t = lid; t < PB*PB; t += WG) {
    i = t / PB;
    j = t % PB;

    P[i][j] = A[pivot_index(i)][pivot_index(j)];
}
```

Adjacent lanes then mostly read adjacent elements from the two contiguous B-wide regions.

The same principle applies to initialization and writeback.

The design should not depend on:

```text
one thread == one matrix row.
```

The workgroup owns the operation collectively.

This also makes a future WG<n experiment trivial without redesigning the algorithm.

---

# 12. Avoid excessive unrolling

For PB=64, a fully unrolled

```c
for (k=0; k<64; k++)
```

inside several nested loops can greatly increase live state and instruction footprint.

Use modest explicit unrolling such as four accumulators:

```c
float s0=0, s1=0, s2=0, s3=0;

for (k=0; k+4<=PB; k+=4) {
    s0 = fma(...,s0);
    s1 = fma(...,s1);
    s2 = fma(...,s2);
    s3 = fma(...,s3);
}

acc = (s0+s1)+(s2+s3);
```

This gives ILP without asking the compiler to materialize enormous unrolled bodies or keep excessive state live.

---

# 13. Cache versus explicit local memory

I would use this rule.

### Put data explicitly in `__local` when:

* it is reused many times before being discarded;
* multiple threads reuse it;
* its lifetime spans many arithmetic operations/barriers;
* explicitly caching it does not reduce WG residency below a valuable threshold.

Therefore:

```text
direct A       -> local
small-N C      -> local
block P/U      -> local
small strip    -> local
rotations      -> local during use
```

### Leave data in global/cache when:

* it is streamed once;
* it is used for one pivot and discarded;
* keeping it locally would cut resident WGs;
* access can be made coalesced;
* its reuse distance is short enough that L1/L2 may help naturally.

Therefore:

```text
large-N A/V panels -> global/cache
rotation log       -> global/L2
final eig/output   -> global
```

This matters particularly on Ampere because L1 and shared memory share the same 128-KB unified resource. More explicit shared memory does not simply give "more fast memory"; it can take capacity away from hardware-managed cache. ([NVIDIA Docs][1])

---

# 14. Optional next step: packed symmetric A

If the A-local design wins at N=86, a second version can exploit symmetry and retain only

$$
\frac{N(N+1)}2
$$

elements of A.

At N=128:

```text
full A    ≈ 64 KB
triangle  ≈ 32 KB
```

which is the difference between roughly one and potentially two or three resident WGs.

This could extend the resident-A regime substantially upward in N.

I would **not implement this first**.

A square padded A is vastly easier to validate and has cleaner bank behavior.

Only implement packed/tiled triangular A if measurements show that the shared-memory occupancy threshold around N≈110–130 is genuinely important.

---

# 15. Proposed final architecture

The complete solver should look like this:

```text
                         N
                         |
             +-----------+-----------+
             |                       |
          small N                 medium N
          ~<=64                   ~65–120
             |                       |
             v                       v

     FULL RESIDENT              RESIDENT A
     -------------              ----------
     A in local                 A in local
     C in local                 C NOT present
     one WG/system              one WG/system
     all sweeps                 record rotations
     no log                          |
                                     v
                             ROTATION APPLY
                             --------------
                             row-tiled C
                             many WGs/system
                             tiny local tile


                         large N
                         ~>120–160
                             |
                             v

                       BLOCK JACOBI
                       ------------
                       one WG/system
                       P + U local
                       A,V global
                       tiny strip tile local
                       cooperative loads
                       no private PB arrays
```

The exact crossover values are measured parameters, not architectural constants.

---

# 16. First implementation I would build

For N=86, build exactly three A/B variants:

| Variant       | A         | C                      |                      Expected local/WG |
| ------------- | --------- | ---------------------- | -------------------------------------: |
| current       | global    | global                 |                               ~7–16 KB |
| full resident | local     | local                  |                              ~60–65 KB |
| **proposed**  | **local** | **deferred row-tiled** | **~31–33 KB solver + ~6 KB transform** |

For the proposed A-resident kernel initially use:

```text
WG = 512
Fermi compiled out
double c,s rotation parameters
square padded A
subgroup reductions if supported
no private arrays
one WG/system
```

and target:

```text
<= 32–33 KB local/WG
<= 40–42 registers/thread
zero spills
```

because that gives the possibility of **three WG512 eigensystems resident per SM**.

Then the transform kernel should use roughly:

```text
WG       = 256
RTILE    = 8 or 16
local    < 8 KB
many row-tile WGs/system
```

and should apply the entire recorded rotation history while the C rows remain resident.

The benchmark should compare not just eigensolver latency but:

$$
\boxed{\text{complete SCC replica-iterations/second}}
$$

at slot counts

```text
64, 96, 128, 192, 400
```

because the final architecture must remain efficient after the slot scheduler reduces the number of simultaneously active systems.

---

# 17. Most important hypothesis

The architecture is based on one very specific hypothesis which can be falsified cleanly:

> The expensive part of the current direct Jacobi is not generation of the Jacobi rotations themselves, but repeatedly moving A and C through the memory hierarchy while applying them.

The proposed solver attacks precisely that problem while avoiding the major weakness of a naïve full-local solver:

```text
full A+C local
        ↓
too much local memory
        ↓
one WG/SM
```

Instead:

```text
A local
C deferred
        ↓
~half local footprint
        ↓
2–3 eigensolver WGs/SM

+

C transformed separately
        ↓
tiny local row tiles
        ↓
very high occupancy
```

This is the architecture I would test before spending more time tuning WG sizes in the existing global/global direct kernel.

One aspect I particularly like is that this design gives a clean answer to your uncertainty about **shared memory versus automatic cache**. We don't have to decide philosophically that one is better. We explicitly put only the *high-reuse state* in shared memory: \(A\), the block pivot \(P/U\), and tiny row/strip tiles. Everything else stays global and either gets streamed coalescently or benefits from hardware cache.

There is also a concrete diagnostic I would add before implementation: compile every specialization with NVIDIA's `-cl-nv-verbose`. Unlike `CL_KERNEL_PRIVATE_MEM_SIZE`, this should tell you actual **register count** as well as shared-memory use, which is essential because for WG512 the difference between 40 and 50 registers/thread can mean three resident WGs versus two. NVIDIA explicitly documents this OpenCL compiler option for resource reporting. ([NVIDIA Developer Downloads][2])

The **resident-A + rotation-log** idea is the part I would prioritize. It is more fundamental than another B/WG sweep because it changes the global-traffic scaling of the direct solver while deliberately keeping the per-WG shared footprint near the 3-WG/SM threshold.

[1]: https://docs.nvidia.com/cuda/ampere-tuning-guide/?utm_source=chatgpt.com "1. NVIDIA Ampere GPU Architecture Tuning Guide — Ampere Tuning Guide 13.4 documentation"
[2]: https://developer.download.nvidia.com/compute/DevZone/docs/html/OpenCL/doc/OpenCL_Programming_Guide.pdf?utm_source=chatgpt.com "NVIDIA CUDA Programming Guide"
[3]: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html?utm_source=chatgpt.com "2.3. Writing SIMT Kernels — CUDA Programming Guide"



---

# USER

the coding agent constantly suggest me to do some density matrix purification or fermi operator exapansion to replace Jacobi SCC with some lienar scaling methof. Generally I think this is bad idea for small dense system. But he keep pulling it, so I'm starting to thing maybe it is not so idiotic. What do you think? Tru is that matrix multiplication (even dense one) can be paralelized on GPU much more effciently? But intutitively I would think that diagonalization always wins form small dense system ?


---


# Chat GPT 5.6 sol


I would not dismiss it anymore — but I would separate **SP2/purification** from **Fermi-operator expansion**, because for your problem they have very different prospects.

The important point is that at \(N=86\)–246, “linear scaling” is basically irrelevant. Everything is dense and \(N^3\). The only reason a density-matrix method might beat diagonalization is:

$$
\boxed{\text{20–30 very efficient GEMMs can beat fewer but badly GPU-mapped Jacobi sweeps.}}
$$

Your measurements make that possibility real: the current Jacobi is now ~80% of the SCC device time, and its main architectural problem is repeated global A/V traffic. 

## 1. Purification is not idiotic here

Take SP2/TC2 purification in the orthogonal basis. Schematically,

$$
P_{k+1}=
\begin{cases}
P_k^2 & \text{if population too high}\\
2P_k-P_k^2 & \text{if population too low}.
\end{cases}
$$

Each iteration is essentially **one dense matrix multiplication plus a trivial elementwise kernel**.

So for \(m\) purification iterations:

$$
\mathrm{cost}\sim m \times \mathrm{GEMM}(N,N,N).
$$

Suppose \(m=20\). Arithmetic is much larger than a good dense eigensolver. But a tiled GEMM on your GPU may run at several TFLOP/s, whereas your Jacobi spends enormous time on barriers, rotations and global-memory transformations.

For N=246,

$$
2N^3 \approx 30\times10^6\ {\rm flop/GEMM/system}.
$$

For 400 replicas:

$$
\approx12\ {\rm GFLOP/GEMM}.
$$

Twenty GEMMs are ~240 GFLOP total. A GPU sustaining even 4 TFLOP/s does that arithmetic in ~60 ms.

Your current optimized B32 Jacobi is around **190 ms for the warm-ish batched eigensolve** in the parameter sweep. 

So suddenly the arithmetic says:

> **yes, SP2 could plausibly beat your Jacobi at N≈246 despite doing much more arithmetic.**

That is quite different from saying SP2 is intrinsically a better algorithm.

---

# 2. But SP2 has one enormous disadvantage: your SCC is warm

This is the strongest argument in favour of Jacobi.

Between SCC iterations,

$$
H^{(k+1)} \approx H^{(k)}+\delta H.
$$

You already possess an almost-correct eigenbasis \(C^{(k)}\). Therefore your transformed Hamiltonian

$$
C^{T}H^{(k+1)}C
$$

is nearly diagonal.

A good warm Jacobi solver should exploit this enormously: perhaps only a couple of real sweeps.

A conventional SP2 construction tends to start from the current Hamiltonian spectrum and perform its whole purification sequence again.

So conceptually:

```text
Jacobi:
    yesterday's solution + small correction
              ↓
       few rotations

SP2:
    build f(H) again
              ↓
       ~15–30 GEMMs
```

That strongly favours Jacobi once your **resident-A/deferred-C architecture** is working.

Right now SP2 can look attractive partly because your Jacobi is paying a huge implementation tax.

I would therefore not abandon Jacobi before testing the new architecture.

---

# 3. There is nevertheless a surprisingly attractive feature of density-matrix methods

You actually **don't fundamentally need eigenvectors for SCC**.

In an orthogonal basis,

$$
P = f(H)
$$

is already the density matrix.

Mulliken populations follow from transforming \(P\) back appropriately.

And the energy-weighted density matrix is simply related to

$$
W' = H'P'
$$

for the zero-temperature occupied projector, or more generally

$$
W'=H'f(H').
$$

So forces do not necessarily require explicit eigenvectors either.

That means a purification architecture could eliminate the entire chain

```text
Jacobi
eigenvectors
occupation selection
C transformations
density construction
```

and replace it approximately with

```text
orthogonal H
    ↓
SP2 -> P
    ↓
charges

final:
H P -> W
```

That's much more interesting than merely replacing the Jacobi kernel.

---

# 4. SP2 is the one I would actually test

I would **not start with McWeeny**.

McWeeny:

$$
P\leftarrow3P^2-2P^3
$$

usually needs two GEMMs per iteration.

SP2/TC2 needs essentially one:

$$
P^2
$$

followed by choosing either

$$
P\leftarrow P^2
$$

or

$$
P\leftarrow 2P-P^2.
$$

That is extremely GPU-friendly.

Also, for all 400 replicas, the branch can be made per replica from the trace.

So:

```text
P2 = batched_GEMM(P,P)

for each replica:
    if Tr(P) > Nocc:
         P = P2
    else:
         P = 2P-P2
```

That's beautifully simple.

No global synchronizations inside the GEMM. No giant local-memory eigensystem. No Jacobi rotations.

---

# 5. Fermi-operator expansion is much less attractive

Here I share your skepticism much more strongly.

You currently use finite-temperature occupations:

$$
f(E)=\frac{1}{1+\exp[\beta(E-\mu)]}.
$$

A polynomial FOE approximates

$$
P=f_\beta(H-\mu).
$$

The problem is that the required polynomial degree scales roughly with spectral width relative to thermal broadening:

$$
m\sim O(\beta\,\Delta E)
$$

up to details of approximation/tolerance.

If your electronic spectrum spans, say, \(10\!-\!20\) eV while \(kT\sim0.05\) eV, you're looking at a ratio of order

$$
200-400.
$$

A straightforward Chebyshev FOE may therefore need **hundreds of matrix recurrences**.

Every polynomial order means another matrix multiplication or matrix-H multiplication.

That is completely different from perhaps 15–30 SP2 iterations.

For N=86–246 I would expect straightforward FOE to be a poor trade.

Rational/pole expansions reduce the degree but replace GEMMs by shifted inversions/linear solves, which is even harder to justify for matrices this small.

So my ranking is:

$$
\boxed{\text{SP2 worth testing}}
$$

$$
\boxed{\text{generic FOE probably not}}
$$

---

# 6. Finite temperature is also a real problem for SP2

SP2 naturally computes an idempotent zero-temperature projector:

$$
P^2=P.
$$

Your current Fermi smearing gives

$$
0 < f_i < 1
$$

near the chemical potential.

So SP2 is not exactly equivalent to your present SCC.

That matters particularly for:

* small-gap structures;
* near-degenerate frontier states;
* metallic-ish systems;
* smooth PES/forces around orbital crossings.

You could perhaps use zero-temperature SP2 if your molecular systems normally have a decent gap. But I would **not silently replace finite-T Jacobi with SP2 and call it equivalent**.

That would be a scientific/numerical algorithm change, not merely optimization.

---

# 7. There is a hybrid that I find more interesting

Instead of using purification to obtain the final density matrix, use a polynomial filter to obtain the **occupied subspace**, then diagonalize only inside that subspace.

Something like CheFSI:

```text
warm occupied subspace Cocc
        ↓
few Chebyshev filter applications H*Cocc
        ↓
orthonormalize
        ↓
small Rayleigh-Ritz diagonalization
```

This has several appealing properties for your SCC:

1. **It is warm-start friendly.**
2. Most operations are GEMM-like.
3. You only track occupied orbitals plus a small frontier buffer.
4. You retain eigenvalues/occupations for Fermi smearing.

For DTH,

$$
N=246,\qquad N_{\rm occ}\approx123.
$$

That's unfortunately about half the full space, so the saving is not spectacular. But it could still be more GPU-friendly than complete Jacobi.

For molecules where \(N_{\rm occ}/N\sim 1/2\), I would regard CheFSI as a later experiment, not the first alternative.

---

# 8. There is an even more direct warm-density approach

If the coding agent keeps pushing matrix-function methods, I would ask it this:

> Why recompute \(P=f(H)\) from scratch each SCC iteration when we already have \(P_k\) and \(H_{k+1}=H_k+\delta H\)?

That is the real equivalent of your warm Jacobi philosophy.

One could update the density matrix perturbatively or through a small number of purification/correction steps.

For example conceptually:

$$
P_{k+1}
=
P_k+\delta P
$$

with \(\delta P\) driven by the occupied/unoccupied coupling induced by \(\delta H\).

That starts to resemble the projector-response formula your agent proposed previously.

This could eventually be excellent, but it is significantly more delicate numerically than plain SP2.

---

# 9. A very cheap experiment can settle whether SP2 deserves attention

You don't need to implement a production solver.

You already have the tiled batched GEMM.

Implement a **minimal SP2 benchmark only**:

```text
Input:
    the exact H' matrices encountered during real SCC iterations

For each matrix:
    spectral bounds
    initialize P0
    iterate SP2 until:
        ||P²-P||
        electron count
        commutator ||HP-PH||
    reach production-quality tolerance
```

Measure:

```text
number of GEMMs
wall time batch=400
P error vs Jacobi reference
energy error
Mulliken charge error
```

Do this for:

```text
GC N86:
    cold H
    warm SCC H
    near-converged H

DTH N246:
    same three cases
```

Do **not integrate it into SCC yet**.

This experiment is perhaps a day of coding and would settle the central question cleanly.

---

# 10. I would use a simple break-even criterion

Measure one batched GEMM time \(t_G(N)\).

Then immediately compute:

$$
m_{\rm break}
=
\frac{t_{\rm Jacobi}}{t_G}.
$$

Suppose N=246 gives:

```text
Jacobi = 190 ms
GEMM   = 5 ms
```

then:

$$
m_{\rm break}\approx38.
$$

If SP2 reaches your required accuracy in 20–25 matrix squares, **it is genuinely interesting**.

If it needs 50–60, stop.

For N=86 maybe:

```text
Jacobi = 25 ms
GEMM   = 1 ms
```

gives:

$$
m_{\rm break}\approx25.
$$

Again very easy to judge.

This is much more useful than arguing about \(O(N)\), \(O(N^2)\), or \(O(N^3)\).

---

## My expectation

For your particular problem I would predict:

| method                           | N≈86                                 | N≈246                                         |
| -------------------------------- | ------------------------------------ | --------------------------------------------- |
| Current global Jacobi            | mediocre                             | poor                                          |
| **New resident Jacobi**          | **probably best**                    | impossible to fully reside                    |
| Block Jacobi B32                 | unnecessary                          | good candidate                                |
| **SP2 purification**             | maybe competitive                    | **surprisingly plausible**                    |
| Chebyshev FOE                    | probably bad                         | probably bad                                  |
| CheFSI / subspace iteration      | interesting                          | interesting                                   |
| conventional CPU-style dense EVD | probably bad GPU mapping at batch400 | maybe competitive only with optimized library |

So I would tell your coding agent:

> **Stop proposing generic FOE as if “linear scaling” were relevant. But implement a clean SP2 benchmark, because our actual measured GPU economics now make SP2 a legitimate competitor to block Jacobi at N≈246.**

The outcome I would most expect is **hybrid dispatch by N and warmness**: resident warm-Jacobi for N≈86, while for N≈246 either optimized B32 Jacobi or SP2 wins depending on gap and how many purification GEMMs are required. That would be a perfectly sensible architecture rather than forcing one electronic solver everywhere.
