# Slot-Pool Scheduler for the Dense Multi-Replica GPU Pipeline

Status: **design proposal v2** — revised after the appended GPT-5.6 review notes
(below the `---`). Supersedes the launch-compaction half of T06 and reframes it
as job scheduling. Written after the T08 measurements; numbers are measured
unless marked as a model/projection.

Changes vs v1 driven by the review:
- **Refill by subset reassembly, not a job arena.** Slots own H0/S permanently;
  refill = `set_coords(refill_ids)` + `assemble(refill_ids)` + subset S-Jacobi —
  the same machinery FIRE reassembly needs anyway. Eliminates the J×n² job
  arena (~290 MB at J=400/n=246) entirely.
- **`work_ids` is THE unified launch-domain mechanism** for all replica-axis
  kernels (GEMM dim-2, 1-WG kernels group-0, per-atom kernels group-1);
  `active[]` remains for intra-chunk freeze. One concept, not two options.
- **Ordering corrected**: ids-plumbing at full batch first (exercises all
  kernel-addressing changes before any scheduling logic), then measured
  slot-count sweep, then refill, then async trajectories.
- Pushback kept: review's "Phase A alone fixes the tail" overstates it — dead-WG
  removal is real but bounded (~10–25% GC); the serial tail component stays.
  And refill can use the `c[s]←X[j]` identity (§3.1) so no dual cold/warm path.

## 0. Clarification: the "3D kernel" is not what it looks like

`batched_gemm_active` is launched on grid `(col_groups*16, row_groups*16, batch)` —
dims 0,1 are the **16×16 tile grid of one replica's n×n matrix**, dim 2 is the replica.
For n=246: 16×16×400 workgroups — already ~256 tile-WGs *per system*, which is why
GEMM saturates at far fewer replicas than the 1-WG/system Jacobi does (§4). The 2D
part stays; only dim 2 is remapped: `sid = work_ids[get_group_id(2)]`.

## 1. Where the current design actually wastes time

```
for each chunk (8 iters, one host sync):
    for each kernel:  launch batch×wg workgroups
                      dead slots: read active[sid]==0, exit in ~µs
    host reads rms[] + active[]
```

- **Dead slots do NOT consume compute.** A frozen replica's WGs exit on the flag
  before any tile work. Per-iteration device time ≈ `c0 + active(t)·c1`.
- **Waste 1 — `c0` × dead iterations.** GC b400: 100 iters, median replica done
  ~56 → ~44 iters pay full launch/schedule overhead for a handful of live slots.
  Bounded: ~10–25% of wall.
- **Waste 2 — the MD-step lockstep barrier (the big one for relaxed scans).**
  `relax()` does `scc → fire_step → scc → …` with **all** replicas stepping in
  lockstep; replica A's 20-iter SCC waits for B's 80 at *every* FIRE step. Cost
  multiplies by MD steps per point (~10–50). This is the dominant win.
- **Waste 3 — memory caps batch.** ~10 state buffers × n² ≈ 2.4 MB/slot at
  n=246 → J=400 ≈ 1 GB. Slot pool decouples J from device memory.

## 2. Three levels of identity

```rust
type JobId  = u32;   // 0..total_jobs   — a physical problem to solve
type SlotId = u32;   // 0..n_slots      — persistent GPU memory region
// work index iw = position in a compact launch list (transient, per launch)
```

Host scheduler state per slot: `{ job: Option<JobId>, phase, scc_iters, md_steps }`
with `phase ∈ {Empty, Assemble, Scc, Finalize, Force, Fire, Done, Failed}`.

Mappings:

```
slot_job[slot] -> job          (host truth; device mirror slot_job_dev)
per-phase lists:  scc_ids[] / force_ids[] / assemble_ids[] / refill_ids[] / retry_ids[]
```

**Matrices never move between slots.** Slot s always addresses the s-th region
of every state buffer; only its meaning changes on refill.

## 3. Refill — by subset reassembly, not input copies

`refill_slots(&[(SlotId, JobId)])` performs, for only those slots, via the
`refill_ids` compact domain:

```
coords[s] ← job geometry          (host upload of just those slots' coords)
assemble H0, S, G for slot s      (existing kernels on refill_ids)
S-Jacobi → X for slot s           (existing masked S-solve on refill_ids)
c[s] ← X                          (see 3.1)
q[s] ← q0, q_new[s] ← q0
diis_hist[s] zeroed               (buf_idx, n_filled, flag, reason — slot slice)
active[s]=1, jacobi_diag cleared, rms cleared
FIRE state (v, dt, α, n_pos)      (Stage D only)
```

All device-side, enqueued in-order at the chunk boundary — no extra host syncs,
no allocation. There is **no job arena**: job inputs live on the host (coords
table) and results land in `JobId`-indexed arrays at retire time.

### 3.1 Refill without a dual cold/warm path — the `c[s] ← X` identity

The plan has ONE global `b_warm` flag — a refilled slot can't be cold while
peers are warm. Dissolved by the identity: **the warm solve seeded with the
slot's own Löwdin basis is algebraically the cold solve.** Warm path does
`A = cᵀ H_scc c` then rotates `c` in place; with `c[s] := X` this is
`A = Xᵀ H_scc X` and `c ← X·V` — bit-for-bit the cold path's `X·C'`. So refill
sets `c[s] = X` (just produced by the subset S-solve) and the whole pool stays
uniformly warm — no second eigensolver branch, no per-slot warm flag.
(This is strictly better than the review's "normal cold electronic
initialization," which would have needed the dual path.)

### 3.2 Per-slot reset contract

| state | action on refill |
|---|---|
| `h_scc`, `hp`, `temp`, `sc`, `occ_w`, `eig`, `mu` | scratch — overwritten next iter |
| `coords[s]` | ← job geometry; then assemble → H0/S/G, S-Jacobi → X |
| `c[s]`, `x_buf[s]` | ← X (3.1) |
| `q[s]`, `q_new[s]` | ← q0 |
| `active[s]`, `jacobi_diag[s]`, `rms[s]` | =1 / cleared |
| `diis_hist[s]` | zeroed slot slice |
| `fire_ctl[s]`, `v_dev[s]` | per-job MD state — Stage D |

## 4. Launch domains: `work_ids` as the single convention

Every replica-axis kernel takes `__global const int* work_ids` (+ implicit
`n_work` = launch width):

```c
sid = work_ids[get_group_id(2)];   // GEMMs (dim 2 = replica)
sid = work_ids[get_group_id(0)];   // 1-WG-per-system kernels (Jacobi, hscc, …)
sid = work_ids[get_group_id(1)];   // per-atom kernels (dim 0 = atom)
```

Invariant: **memory address = physical slot; launch domain = compact work
index.** The same mechanism serves `scc_ids`, `assemble_ids`, `force_ids`,
`refill_ids`, `retry_ids`, and the identity list for full-domain phases —
no per-phase concepts proliferating through kernels.

`active[sid]` is **retained**: `work_ids` removes dead systems *between*
chunks; `active[]` disables ones that converge *inside* a chunk. The launch
width stays constant for the 8-iter chunk while the mask handles mid-chunk
freeze. Both layers coexist; neither is redundant.

## 5. Scheduling boundary = SCC chunk, never Jacobi sweeps

```
JOB/relax → MD/FIRE steps → SCC cycles → Jacobi solve → sweeps/rotations
```

The scheduler quantum is one SCC chunk (~8 iters, the existing sync point —
`read_chunk_status` already reads rms+active there, a few KB). Jacobi WGs run
to their local stop condition undisturbed; hardware already balances them.
Device-side scheduling/global-atomic queues are explicitly rejected — Rust
scheduling at chunk granularity is free relative to ~ms of device work, and
matches the manifest's no-global-atomics rule.

Adaptive chunk size (8 → 4 → 2 in the tail) is a later refinement — fixed 8
first.

## 6. Sizing S — measured, not guessed

Asymmetry that decides it: GEMM emits ~256 tile-WGs/system at n=246 (saturates
at tens of slots), while Jacobi is 1 WG/system — needs ~WGs-per-device slots.
So the knee differs per kernel; S must cover the *worst* (Jacobi).

- Measured so far (GC n=86, direct, WG=512): sys/s still rising b100→b400
  (785→996) — not saturated. DTH n=246 block WG=256/34.8 KB local ≈ 2–3 WGs/SM.
- Sweep S ∈ {32,64,96,128,192,256,400} on the existing bench for GC and DTH;
  pick the smallest S within ~95–98% of peak throughput. `RUST_DFTB_SLOT_POOL`
  override; default stays S=J until measured.

**Honest bound:** for J=S fixed-geometry scans, total work `Σ iters·c1` is
invariant — slotting wins only `c0` overhead + tail granularity (GC ~10–25%).
The big wins are the memory cap (J ≫ S) and Stage D.

## 7. Stage D — per-slot MD trajectories (the dominant win)

Each slot owns its **whole relaxation** — never migrate mid-trajectory. Its
warm electronic state (q, C, SC, μ, DIIS) and FIRE state (x, v, dt, α, n_pos)
persist across the entire job.

```
per slot:  ASSEMBLE → SCC → FORCES → FIRE ──converged──▶ retire+refill
                              ▲          │not converged
                              └──────────┘ (assemble again)
```

Slots in different phases coexist; the host builds per-phase `work_ids` lists
at the chunk boundary and launches each stage on its subset. All stages are
already masked kernels (assembly/FIRE on `park`, forces on `state_ok`, SCC on
`active`) — only the domain grouping is new. No megakernel — scheduler in
Rust, numerics in specialized kernels.

Failure semantics preserved: Converged / Plateau / Failed per job; failed jobs
form a `retry_ids` compact list (existing masked-retry maps directly).

Warm-starting a fresh job from a neighboring converged scan point (q, C, μ
copies) is a *later optimization* — scheduler correctness must never depend on
it (job order is arbitrary).

## 8. What changes where

| component | change |
|---|---|
| `gpu_scc_plan.rs` | batch→`n_slots`; `work_ids` buffer + `n_work`; per-phase domain setters; `refill_slots()`; `b_warm` stays global (all slots warm via 3.1) |
| `gpu_dftb.rs` | `set_coords`/`assemble` gain slot-subset variants; scheduler loop replaces `scc_mix_inner`'s chunk driver; job queue, `JobId`-indexed results, retirement; Stage D adds per-slot phase machine replacing `relax()`'s lockstep body |
| `.cl` kernels | `work_ids` arg on all replica-axis kernels (~12 in-loop + assemble/force/FIRE); `sid = work_ids[iw]`; `active[]` mask retained |
| tests | slot≠job indexing parity; refill lifecycle (cold-start ≡ fresh plan, DIIS reset, noncontiguous order, queue exhaustion mid-chunk, failed-job retirement, all-idle); Stage D: per-slot trajectories vs lockstep relax reference |

## 9. Risks and non-goals

- **Refill correctness is the whole risk** — stale slot state produces a
  *plausibly converged wrong answer*. The §3.2 contract must be complete; the
  guard is a per-slot parity test (refilled trajectory ≡ fresh-plan).
- **Does not remove the serial tail** — the last job still takes its iters
  alone; queueing waste is removed, not the critical path.
- **S too small** → underfilled between kernels. Empirical (§6); S=J default.
- **No mid-trajectory migration** — a job owns its slot until DONE.
- **Results are JobId-indexed, never SlotId** — slot order is arbitrary;
  `slot_job_dev` lets device kernels write results directly by job.
- Explicitly NOT doing: matrix copies on compact, GPU-side atomic job queue,
  fused SCC+Jacobi+FIRE megakernel, removing GEMM tile dims, hard-coded S,
  Jacobi-sweep-level scheduling.
- PBC plan shares the pattern — out of scope until the molecular path proves out.

## 10. Staging

1. **A — `work_ids` plumbing at batch=J.** No refill, S=J. Convert all
   replica-axis kernels to `sid = work_ids[iw]`; compact the active list at
   each chunk boundary. Pure addressing refactor — identical outputs required.
   Exercises every kernel-addressing change refill needs.
2. **B — slot-count sweep.** Measure sys/s vs S on GC+DTH; pick the knee.
3. **C — queue + refill.** `total_jobs > n_slots`; `refill_slots()` by subset
   reassembly (§3); fixed-geometry jobs first.
4. **D — async trajectories.** Per-slot phase machine; SCC→forces→FIRE→assemble
   independently per slot; retire on force convergence.




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
