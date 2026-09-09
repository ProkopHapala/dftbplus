# GPU Multi-System Parallelization — Design & Decisions

**Created:** 2026-09-05
**Revised:** 2026-09-06 (incorporates GPT 5.6 architectural review)
**Author:** prokop / Devin
**Status:** Design — Wave 1 implementation complete, Wave 2 in progress
**Related:** `OVERVIEW_Roadmap.md` §6–7, `DFTB_Hassembly_OpenCL.md`, `OpenCL_DFTB_optimization.md`, `rust_QMQM.md`, `doc/prokop/chats/MultiSystemOpenCL.chat.md`

---

## 1. Motivation & Use Cases

We want to run **many instances of the same quantum-mechanical calculation** on GPU,
exploiting the thousands of threads. Three use cases, in order of coupling:

### 1.1 Independent replicas (scan / NEB / MD ensemble)
- **NEB:** N images of the same molecule at different geometries along a reaction
  path. Each image is a **fully independent** DFTB calculation — spring forces are
  handled externally by the NEB driver on the host.
- **Rigid scan:** freeze all but one coordinate, sweep it, compute energy+forces
  at each point. Each point is independent.
- **Relaxed scan:** at each scan point, relax the remaining coordinates. Each
  point is an independent geometry optimization (many SCC + force evaluations).
- **MD ensemble / parallel tempering:** N independent MD trajectories.

**Key property:** systems do NOT interact. Simplest case, right place to start.

### 1.2 QM/QM electrostatic coupling
- Multiple **non-covalently bonded fragments** that **mutually polarize** via
  inter-fragment γ-function electrostatics.
- Fragments are diagonalized independently, but their charges couple through
  `compute_v_ext` each SCC iteration.

### 1.3 Full supersystem (single large molecule)
- One big molecule diagonalized as a whole — standard DFTB+ use case.
- Out of scope for this document; covered by the single-fragment GPU path.

### 1.4 System vs Fragment distinction (D12)

A critical conceptual distinction introduced in the GPT 5.6 review:

```
System
    independent scheduling + convergence unit

Fragment
    independently diagonalized QM block inside one System
```

Examples:
```
NEB:
    System = one NEB image
    usually one Fragment

QM/QM:
    System = whole mutually polarizing complex
    Fragment = individual independently-diagonalized molecular pieces

MD ensemble:
    System = one trajectory
```

This matters because fragments in QM/QM **cannot independently declare SCC
convergence**: their charges are mutually coupled. Their parent System converges
as a whole. The GPU scheduler reasons about Systems, not Fragments.

---

## 2. Core Architecture: Host-Orchestrated, Device-Resident

> **The central architectural principle, revised from GPT 5.6 review:**
>
> The Rust host enqueues a sequence of OpenCL kernels into one in-order command
> queue. No charge matrices, density matrices, or eigenvectors cross the PCIe
> bus during SCC. Kernel boundaries provide the global synchronization that
> OpenCL workgroups cannot provide internally.

This is the **missing third option** between the two we originally considered:

- ~~(a) Host-driven with CPU readback~~ — PCIe round-trip every SCC iteration
- ~~(b) GPU-internal mega-kernel~~ — impossible to synchronize across workgroups
- **(c) Host-orchestrated, device-resident** — host enqueues kernels, data stays on GPU ✓

### 2.1 Why (c) is correct

OpenCL workgroups synchronize internally via `barrier(CLK_LOCAL_MEM_FENCE)`.
Distinct workgroups **cannot synchronize within a kernel**. But **kernel
boundaries** are implicit global barriers — all workgroups finish before the
next kernel begins. So we get global synchronization for free by using separate
kernel launches per algorithmic step.

An in-order OpenCL queue already establishes execution ordering between kernels.
So the host simply enqueues:

```
gamma_matvec          →  V_A = G · Δq
H_scc_update          →  H = H0 + 0.5·S·(V_i + V_j)
orthogonalize         →  H' = X·H·X  (X = S^{-1/2}, precomputed)
jacobi_cyclic         →  ε, U  (eigendecomposition of H')
build_density         →  D = U·f(ε)·U^T
mulliken_charges      →  q = diag(D·S)
residual              →  RMS = ||q_new - q_old||
mixer                 →  q_mixed = mix(q_new, q_old, history)
```

No `read_buffer()`. No `queue.finish()`. No PCIe traffic. The host only reads
back the final converged energy and charges (or whatever the caller needs).

Later, if kernel-launch overhead becomes significant, `cl_khr_command_buffer`
can record and replay a sequence of commands with mutable dispatch.

### 2.2 Convergence divergence: active mask

Different systems need different SCC iteration counts. In a batched kernel,
workgroups execute independently — water #17 can finish after 6 iterations while
water #42 goes to 25. Workgroups do NOT all wait for the slowest (they're
independent). But the kernel can't finish until the last workgroup finishes.

Solution: **active mask** in global memory.

```
active[system] = 1  (initialized)

each SCC kernel:
    sid = batch_system
    if (!active[sid]) return;    // converged systems cost ~nothing

residual kernel:
    if (rms[sid] < tol)
        active[sid] = 0;
```

Converged systems early-return on subsequent SCC iterations. No readback needed.
Only if the active population becomes extremely sparse would we compact
`active_ids[]`.

### 2.3 When to use multiple queues

For **NEB and rigid scans**: all images finish independently — one queue is fine.

For **independent MD / relaxed scans**: use **microbatches** on 2–4 queues:
```
1000 trajectories
batch 0:  systems 0..63    → queue 0
batch 1:  systems 64..127  → queue 1
...
```
Each microbatch has enough systems to saturate the GPU but advances
independently. A pathological system stalls only its 64-system group.

---

## 3. The Real Size Variable: Orbitals, Not Atoms

> **"<100 atoms" is misleading.** 100 H atoms ≈ 100 orbitals. 100 C/N/O atoms
> ≈ 400 orbitals (with s/p basis). The orbital count determines everything:
> local memory fit, GEMM cost, eigensolver choice.

Three regimes by N_orb:

| N_orb | Strategy | Local memory | Workgroup mapping |
|------:|----------|-------------|-------------------|
| ≤ 16–32 | **Subgroup/system**, 4–8 systems per WG | trivial | multiple systems per WG |
| ~32–64 | **One WG/system**, A and V fully in `__local` | ~35 KiB (N=64) | 1 WG = 1 system |
| > 64 | **Tiled multi-WG GEMM + block Jacobi** | small tiles | many WGs per system |

The N≤64 regime is the primary target for scan/NEB (most organic molecules
<30 atoms with s/p basis → <120 orbitals, but many scan use cases involve
smaller fragments). The N>64 regime requires block-Jacobi and is a later
optimization.

### 3.1 Local memory budget (N≤64, f32)

For N=64, two float matrices (A + V):
```
2 × N² × 4 B = 32 KiB
```

Store with leading dimension `N+1` to avoid power-of-two bank-conflict patterns:
```
2 × N × (N+1) × 4 B = 32.5 KiB
```

With rotation parameters, reductions, and sorting scratch: ~35 KiB/workgroup.

OpenCL guarantees ≥32 KiB local memory. Contemporary gaming GPUs (Ampere) have
more. Query `CL_DEVICE_LOCAL_MEM_SIZE` and `CL_KERNEL_LOCAL_MEM_SIZE`.

Workgroup sizes:
```
N ≤ 32 :  WG = 128  (4 warps)
N ~ 32-64 : WG = 256  (8 warps)
```

Query `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` rather than hardcoding 32.

### 3.2 Packed symmetric storage (optimization for N~80)

For N slightly > 64, store only the upper triangle of A (symmetric), keep V dense:
```
A_packed + V ≈ 6N² bytes
```
For N=84: ~42 KiB. Might reach ~80 orbitals on a 48 KiB target.

Penalty: more complicated indexing, worse local-memory access. **Do not implement
until the clean dense N≤64 version is benchmarked.**

### 3.3 Block Jacobi for N > 80

```
matrix → 16×16 or 32×32 blocks
Brent-Luk schedule over BLOCKS rather than individual orbitals
for each block pair:
    load 2T × 2T compound block
    diagonalize locally
    obtain Q
kernel boundary
apply Q to block rows/columns using tiled GEMMs
next Brent-Luk round
```

This extends the existing `brent_luk_rounds()` infrastructure rather than
replacing it.

---

## 4. Key Kernels (New Architecture)

### 4.1 Brent-Luk parallel cyclic Jacobi (D8)

**Replaces** the current `local_jacobi_blocks_parallel` which walks sequentially
through every (p,q) rotation with barriers around each one.

**New approach:** Brent-Luk / round-robin matching. N/2 independent rotations
per round, one barrier per round.

```
round 0: (N-1, 0) (1, N-2) (2, N-7) ...   all simultaneously
round 1: (N-1, 1) (2, 0)  (3, N-8) ...
...
```

After N-1 rounds, every unordered pair has been treated exactly once.

The whole round is `A' = G^T A G` where `G = diag(G_0, G_1, ...)` and each `G_i`
is one independent 2×2 rotation. The update is by 2×2 pair-pair blocks:

```
B_ab = G_a^T · A_ab · G_b
```

Every 2×2 block is disjoint in memory. Each work-item owns one block. **No
atomics, no races, one barrier per round.**

For N=64: 2016 rotations / (N/2) = 63 rounds. Each round has one barrier.
Current kernel: ~2016 barriers per sweep. **~32× fewer barriers.**

#### Round-robin formula (no schedule upload needed)

For even dimension M, fix index M-1. In round r:
```c
inline ushort2 jacobi_pair(const int round, const int ipair) {
    if (ipair == 0)
        return (ushort2)(JN-1, round);
    const int m = JN-1;
    return (ushort2)(
        (round + ipair)     % m,
        (round + m - ipair) % m
    );
}
```

#### Odd N handling

Pad N → N+1 with a dummy state. The dummy has a huge diagonal eigenvalue and
zero off-diagonal elements. When paired with a real state, the Jacobi rotation
is exactly identity. One kernel works for both N=63 and N=64.

### 4.2 S^{-1/2} via Jacobi (D9)

**Currently missing** — tests hide this by computing S^{-1/2} on CPU and
uploading. For MD/NEB/relaxed scans, S changes with geometry, so this needs a
GPU solution.

Compute S = U·Λ·U^T using the same Brent-Luk Jacobi kernel. Then:
```
X = S^{-1/2} = U · Λ^{-1/2} · U^T
```

A dedicated kernel loads U once into local memory and evaluates:
```c
X_ij = sum_k U_ik * rsqrt(lambda_k) * U_jk
```

**Computed once per geometry, not once per SCC iteration.**

Candidates to benchmark later:
- (A) Jacobi eigendecomposition → U·Λ^{-1/2}·U^T  [cleanest, reuses Jacobi]
- (B) Batched Cholesky + triangular transformations
- (C) Newton-Schulz inverse square root using GEMMs [attractive when S ≈ I]

### 4.3 Full-local batched GEMM (for N ≤ 64)

For thousands of tiny matrices (N=32–64), benchmark a simple version:

```c
__kernel void matmul_full_local_batched(
    __global const float* A,
    __global const float* B,
    __global float* C
){
    int sid = get_group_id(0);    // system index
    int lid = get_local_id(0);

    __local float LA[N_ORB*(N_ORB+1)];
    __local float LB[N_ORB*(N_ORB+1)];

    // global → local exactly once
    for(int t=lid; t<N_ORB*N_ORB; t+=WG){
        int i=t/N_ORB, j=t-i*N_ORB;
        LA[i*(N_ORB+1)+j] = A[sid*N2+t];
        LB[i*(N_ORB+1)+j] = B[sid*N2+t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int t=lid; t<N_ORB*N_ORB; t+=WG){
        int i=t/N_ORB, j=t-i*N_ORB;
        float sum=0.0f;
        for(int k=0;k<N_ORB;k++)
            sum=fma(LA[i*(N_ORB+1)+k], LB[k*(N_ORB+1)+j], sum);
        C[sid*N2+t]=sum;
    }
}
```

For N=64, WG=256: each thread computes 16 elements. Memory access is nice:
within a warp, `LA[i,k]` is broadcast, `LB[k,j]` is consecutive across j.

**Benchmark this vs tiled GEMM.** For N>64, revert to tiled multi-WG GEMM.

### 4.4 SCC Hamiltonian update (cheap elementwise)

**Separate from H0/S assembly.** H0 and S are computed once per geometry.
The SCC shift is a trivial elementwise kernel:

```c
__kernel void h_scc_update(
    __global const float* H0,
    __global const float* S,
    __global const float* V_atom,   // per-atom potentials
    __global       float* H,
    __global const int*   orb_atom, // orbital → atom mapping
    int N2
){
    int sid = get_group_id(0);
    int lid = get_local_id(0);
    int base = sid * N2;

    __local float LV[N_ATOM];
    for(int a=lid; a<N_ATOM; a+=WG)
        LV[a] = V_atom[sid*N_ATOM+a];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int t=lid; t<N2; t+=WG){
        int i=t/N_ORB, j=t-i*N_ORB;
        float vij = 0.5f * (LV[orb_atom[i]] + LV[orb_atom[j]]);
        H[base+t] = fma(S[base+t], vij, H0[base+t]);
    }
}
```

This is **much better** than embedding the SCC correction inside SK interpolation
as the current `assemble_pairs` does.

### 4.5 Gamma precomputation (D15)

For <100 atoms, store G_AB as a dense N_atom × N_atom matrix:
```
100² × 4 bytes = 40 kB/system
```
Tiny compared to orbital matrices. Then SCC uses a cheap batched matvec:
```
V_A = sum_B G_AB · Δq_B
```

This removes expensive transcendental (exp) computations from the inner SCC loop.
Gamma is computed once per geometry.

### 4.6 SCC optimization with precomputed X (D16)

Once X = S^{-1/2} exists, precompute:
```
H'0 = X · H0 · X    (once per geometry)
```

Then each SCC iteration only needs:
```
V = G · Δq                      (matvec, cheap)
H'_scc = H'0 + 0.5·(X·V·Y + Y·V·X)    where Y = S^{1/2} = S·X
```

Since V is diagonal, multiplying by it is just row/column scaling. And
`Y·V·X = (X·V·Y)^T`, so only one GEMM is needed:
```
A = X · V · Y
H'_scc = H'0 + 0.5·(A + A^T)
```

**~1 GEMM per SCC iteration** instead of the current 2 GEMMs (X^T·H_scc + ...·X).

---

## 5. Design Decisions (Revised)

### D1: Workgroup-to-system mapping — DEPRECATED

**Original:** "1 workgroup = 1 replica" as the universal architecture.
**Revised:** **System = batch index.** Each kernel chooses its own
parallelization. Embrace the "inconsistent" mappings — they're correct:

| Operation | Natural GPU mapping |
|-----------|-------------------|
| SK H0/S assembly | 1 WG/system **or** (system, pair-tile) |
| Gamma G_AB construction | 1 WG/system or tiled |
| SCC V = G·Δq | 1 WG/system |
| H-SCC update | matrix elements × systems |
| GEMM (N≤64) | 1 WG/system, full-local |
| GEMM (N>64) | many WGs/system, one WG/output tile |
| Jacobi (N≤64) | 1 WG/system, full-local |
| Jacobi (N>80) | many WGs/system, block-Jacobi |
| Mulliken population | 1–several WGs/system |
| Residual/mixer | 1 WG/system |

The existing `batched_gemm` already follows this: system = `get_group_id(2)`,
many tile workgroups per matrix.

### D2: SCC loop location — RESOLVED

**Original:** host-driven with readback vs mega-kernel.
**Revised:** **Host-orchestrated, device-resident (option c).** Host enqueues
kernels into one in-order queue. No PCIe traffic during SCC. Kernel boundaries
provide global synchronization. Active mask handles convergence divergence.

Mega-kernel (option b) is a **special fast path for extremely small systems**
only, not the main design.

### D3: Diagonalization strategy — CORRECTED

**Original error:** claimed purification is O(N²·n_iter). **Corrected:** it is
O(N³·n_iter) since it uses dense GEMM. Its advantage is not better asymptotic
scaling but that **GEMM maps extremely efficiently to GPU** and avoids
eigenvectors.

**Revised recommendation:**
- **Primary:** Brent-Luk parallel cyclic Jacobi (D8) for N≤64. Yields MO
  coefficients (needed for data saving).
- **Large N:** Block Jacobi for N>80.
- **Optional:** Purification for production runs where MO coefficients aren't
  needed. Fix the current TC2 anti-pattern (readback + per-system launch →
  single batched kernel with mode[system] selection).

### D4: Memory layout — UNCHANGED

Batched contiguous: `H[system][i][j]` at `system*N*N + i*N + j`. All replicas
same N (pad if needed). For scan/NEB: zero waste.

### D5: Data to save per replica — UNCHANGED

| Quantity | Size (f32) | Save? |
|----------|-----------|-------|
| H0 | N² | yes |
| S | N² | yes |
| H_scc | N² | yes |
| C (MO coefficients) | N² | yes |
| ε (eigenvalues) | N | yes |
| D (density matrix) | N² | yes |
| q (atomic charges) | N_atoms | yes |
| E_total | 1 | yes |
| F (forces) | 3×N_atoms | yes (when available) |

**All f32 on GPU.** Save as f32 binary (or convert to f64 on host if needed).

### D6: Inter-fragment charge synchronization (QM/QM) — UNCHANGED

Separate `inter_fragment_vext` kernel reads all fragments' charges from global
memory, computes inter-fragment γ·Δq, writes v_ext to global. No host round-trip.
Defer until independent-replica path is working.

### D7: Workgroup size and occupancy — CORRECTED

**Original error:** "1 WG = 1 replica breaks at 50 atoms because 4950 pairs need
multiple WGs." **Corrected:** A 256-thread WG can stride over 4950 pairs trivially:
```c
for (int ipair=tid; ipair<nPairs; ipair+=256)
```
Each thread handles ~20 pairs. So H-assembly can remain 1 WG/system quite far
upward if that benchmarks well.

**The real constraint is local memory for the eigensolver, not pair count.**
See §3 for the three orbital regimes.

### D8: Brent-Luk parallel cyclic Jacobi — NEW

Replace sequential-rotation Jacobi with parallel cyclic schedule. N/2 independent
rotations per round, one barrier per round. Whole A and V in local memory for
N≤64. N+1 padding for odd N. N+1 leading dimension to avoid bank conflicts.

Kernel computes the Brent-Luk schedule itself via the round-robin formula — no
schedule upload needed.

### D9: S^{-1/2} on GPU — NEW

Via Jacobi eigendecomposition of S: `X = U·Λ^{-1/2}·U^T`. Computed once per
geometry. Reuses the same Brent-Luk Jacobi kernel. Returns λ_min(S) for
precision monitoring.

### D10: f32 only — NEW

**All GPU computation in single precision.** RTX 3090 is ~40× slower in f64.
Target: cheap gaming GPUs.

Monitor:
- λ_min(S) — if comparable to f32 precision (~1e-7), S is ill-conditioned
- ||U^T·U - I|| — orthogonality of eigenvectors
- ||A·U - U·Λ|| — residual of eigendecomposition
- ||H·C - S·C·ε|| — generalized residual

Do not silently regularize. If S develops a near-singular eigenvalue, report it.
Expectation: chemically sensible DFTB bases will often work in f32, but this
must be established empirically against the f64 CPU implementation.

### D11: Device-resident SCC pipeline — NEW

The full SCC loop runs on GPU with no intermediate readbacks:
```
gamma_matvec → h_scc_update → orthogonalize → jacobi →
build_density → mulliken → residual → mixer
```
Host enqueues all kernels per SCC iteration. Active mask skips converged
systems. Host reads back only final results (energy, charges) or whatever the
caller needs.

### D12: System vs Fragment distinction — NEW

See §1.4. System = independent scheduling + convergence unit. Fragment =
independently diagonalized QM block within one System. GPU scheduler reasons
about Systems. Fragments within a System are coupled and converge together.

### D13: Homogeneous template architecture — NEW

For 1000 copies of the same molecular topology, keep one template:
```
Template:
    nAtoms, nOrbs
    atomSpecies[]
    atomOrbOffset[]
    pairI[], pairJ[], pairType[]
    orb_atom[]  (orbital → atom mapping)
```

And instances:
```
coords[Nsys][Natom][3]
q[Nsys][Natom]
H0[Nsys][Norb*Norb]
...
```

Offsets are arithmetic: `atom_base = system * Natom`, `mat_base = system * Norb²`.
This removes indirect addressing and the `l_frags[128]` overflow problem.

For heterogeneous work: several template/size classes.

### D14: Runtime refactoring — NEW

Refactor OpenCL runtime before implementing SCC:
```
GpuRuntime
    context
    device
    queues[]
    capabilities (local mem size, preferred WG multiple, etc.)
    compiled program cache

HamiltonianOps   (uses GpuRuntime)
MatrixOps        (uses GpuRuntime)
SccOps           (uses GpuRuntime)
ForceOps         (uses GpuRuntime)
```

All share the same context. **Cache Kernel objects** — current code constructs a
new `ocl::Kernel` on every call. For a tight SCC loop, create once, update args.

Create a profiling-enabled queue for benchmarks.

### D15: Gamma precomputation — NEW

Store G_AB as dense N_atom × N_atom matrix per system. Compute once per geometry.
SCC inner loop uses cheap batched matvec instead of recomputing gamma_full()
(with exponentials) every iteration.

### D16: H0/S separation from SCC — NEW

**Critical architectural change.** Current `assemble_pairs` conflates SK
interpolation with SCC shift application. Separate them:

**Once per geometry:**
- Assemble H0, S (SK interpolation + rotation)
- Compute G_AB (gamma matrix)
- Compute X = S^{-1/2}
- Precompute H'0 = X · H0 · X

**Every SCC iteration:**
- V = G · Δq (matvec)
- H'_scc = H'0 + 0.5·(A + A^T) where A = X·V·Y, Y = S^{1/2} (one GEMM)
- Jacobi(H'_scc) → ε, U
- D = U · f(ε) · U^T
- q = diag(D · S)
- residual, mix

This makes the SCC inner loop extremely cheap. H0/S assembly is not in the hot
loop at all.

### D17: Geometry preprocessing on GPU — NEW

For homogeneous templates, upload only static info once (pair indices, species,
orbital offsets). Upload `coords[system][atom][3]` each geometry step. GPU
computes r, l, m, n (direction cosines) on the fly.

For <100 atoms, don't even build a neighbor list — 4950 pairs is trivial.

---

## 6. Kernel Bug Fixes (Wave 1, completed)

Agent_1 found and fixed 5 bugs in `dftb_hamiltonian.cl` (user-authorized):

1. **`onsite_diagonal` overflow:** unconditionally wrote 4 diagonal entries
   (s,p,p,p) per atom. For s-only atoms (H), wrote e_p=0 into next atom's slot
   and went out-of-bounds. **Fix:** added `n_orb_per_atom` arg, guarded writes.

2. **`assemble_pairs` SK cache OOB:** copied `n_grid * 4` elements but s-s data
   has 1 column. **Fix:** added `n_sk_cols` arg.

3. **`interp_sk_1` float4 cast:** read scalar table as `float4*`, advancing by
   4× too much. **Fix:** scalar 4-point B-spline stencil.

4. **`write_symmetric_4x4` indexing:** cast `float*` to `float4*` with scalar
   indices → 4× offset error. **Fix:** individual float writes in 4×4 loop.

5. **`rotate_4x4` formula:** used `v.yzw` (dropped py, included pad) and
   element-wise instead of outer product. **Fix:** rewrote to match CPU
   `Rotation::rotate_pp` convention.

GPT 5.6 independently identified issues 1, 3, 4, 5, and the `charges` vs `Δq`
issue (issue 2 in GPT 5.6's list, related to `onsite_and_va`). The `charges` vs
`Δq` issue is handled in the driver by passing zero charges for non-SCC H0
assembly; Agent_4 will pass real Δq for SCC.

---

## 7. Staged Implementation Plan (Revised)

### Stage 0: CPU multi-fragment validation ✅ DONE
- 2-fragment independent SCC, polarization, charge conservation verified.
- 13/13 tests pass. QM/QM interaction energy documented.

### Stage 1: GPU H-assembly runtime ✅ DONE
- `GpuDriver::gpu_assemble_batched` — compiles kernels, uploads GpuBatch,
  launches onsite + V_A + assemble_pairs, reads back H/S.
- 4/4 tests pass (smoke, H2, N2, 10×H2 multi-replica).
- H/S parity ~1e-2 (tolerance gap due to 64-point f32 B-spline resampling).
- 5 kernel bugs fixed.

> **2026-09-09:** mio H-bond no longer uses 64-point resample. Full grid +
> stopgap right pad packs into `SK_GRID_MAX=512`. AT GPU vs CPU max|dH|
> `8.6e-8`. Stage 1.5 as written is **not** the current blocker. Remaining
> interpolator work is extra-control *fitting* (not more resampling, not
> Neville). See `doc/prokop/topical_audit/sk_interpolation.md`.

### Stage 1.5: Fix SK resampling precision (BLOCKING for Stage 2)
- Increase `SK_RESAMPLE_N` from 64 to ≥256, OR
- Upload original SK table and interpolate on GPU with higher-order method.
- Target: H/S parity < 1e-4 (may not reach 1e-5 in f32, acceptable).

### Stage 2: Complete generalized eigenproblem on GPU
- Implement Brent-Luk parallel cyclic Jacobi kernel (D8).
- Implement S^{-1/2} via Jacobi (D9).
- Implement full-local batched GEMM for N≤64 (§4.3).
- Pipeline: H0/S → jacobi(S) → X → H'0 = X·H0·X → jacobi(H'0) → ε, C.
- **Test:** parity vs CPU for H2, N2, H2O, CH4 eigenvalues and MO coefficients.
- **Benchmark:** vs current `local_jacobi_blocks_parallel` at N=8,16,32,48,64
  and batch=1,10,100,1000.

### Stage 3: Device-resident SCC (D11)
- Implement gamma matvec kernel (D15).
- Implement H_scc_update kernel (§4.4).
- Implement Mulliken charges kernel.
- Implement residual + simple mixer kernel.
- Implement active mask (§2.2).
- Full SCC loop: host enqueues, no readback until convergence.
- **Test:** 10× H2O SCC parity vs CPU (energy, charges < 1e-4).
- **Save data:** `save_replica_data` — H0, S, H_scc, C, ε, D, q, E per replica.

### Stage 3.5: Runtime refactoring (D14)
- Refactor `GpuMatrixContext` + `GpuDriver` into shared `GpuRuntime`.
- Cache Kernel objects.
- Add profiling queue.

### Stage 4: Scan/NEB driver
- Rigid scan: sweep coordinate, batched SCC, energy curve.
- NEB: interpolate, batched SCC, spring forces, image update.
- Per-replica data saving.
- **Test:** H2 bond scan 20 points vs CPU.

### Stage 5: Scheduling benchmark
- Compare: (A) giant batch, (B) microbatches 16/32/64/128, (C) per-system,
  (D) multi-queue, (E) microbatches on multiple queues.
- Synthetic convergence distributions: uniform, mild spread, bad tail.
- Measure GPU event time, host wall time, systems/second.

### Stage 6: Forces + MD/NEB
- GPU force kernels (or CPU forces for now, Agent_3 done).
- Relaxed scan, NEB with real forces.
- Determine what density/eigenvector quantities must survive the solve for
  force computation.

### Stage 7: Specialized fast paths
- Subgroup/WG tiny-system solver (N≤16).
- GPU DIIS mixer.
- `cl_khr_command_buffer` for command recording.
- Purification (fix TC2 anti-pattern).
- Direct orthogonal-basis SCC update (D16, ~1 GEMM per iter).
- Packed symmetric storage for N~80.

### Stage 8: QM/QM coupling
- `inter_fragment_vext` kernel (D6).
- System/Fragment architecture (D12).
- Batched fragment solves grouped by N_orb.
- Two-water polarization parity test.

---

## 8. Open Questions

1. **D5:** Save full matrices at every scan point or only at converged points?
   Binary or text format? → **Answered:** binary f32, at every point for scans
   (disk is the bottleneck, not GPU).

2. **D10:** Is f32 precision sufficient for energies and forces? → **Must
   establish empirically** against f64 CPU. Monitor λ_min(S).

3. **NEB:** Spring-force projection on host or GPU? → **Host is fine** (NEB has
   few images ~20–50).

4. **Stage 2 benchmark:** Will full-local GEMM beat tiled GEMM for N≤64? →
   **Benchmark both.** Expect full-local to win for many tiny matrices.

5. **Stage 5 benchmark:** Giant batch vs microbatch vs multi-queue? → **Measure,
   don't decide theoretically.** Expect batched to win for uniform convergence,
   microbatches to win for heavy-tailed convergence.

---

## 9. What Changed From Original Design (Summary)

| Original | Revised | Why |
|----------|---------|-----|
| 1 WG = 1 system (universal) | System = batch index, each kernel chooses mapping | Different operations have different natural parallelizations |
| Host-driven (with readback) vs mega-kernel | Host-orchestrated, device-resident (no readback) | Kernel boundaries provide global sync; no PCIe traffic |
| Purification is O(N²) | Purification is O(N³) | Uses dense GEMM; advantage is GPU mapping, not scaling |
| 1 WG/system breaks at 50 atoms | 256-thread WG can stride over 4950 pairs | Real constraint is local memory for eigensolver, not pair count |
| H0/S + SCC in one kernel | H0/S once per geometry, SCC shift is elementwise | SCC inner loop becomes extremely cheap |
| S^{-1/2} computed on CPU | S^{-1/2} on GPU via Jacobi (D9) | Needed for MD/NEB where S changes with geometry |
| Gamma recomputed every SCC iter | Gamma precomputed as dense matrix (D15) | Removes transcendentals from inner loop |
| Sequential Jacobi rotations | Brent-Luk parallel cyclic (D8) | ~32× fewer barriers for N=64 |
| f64 on GPU considered | f32 only (D10) | RTX 3090 40× slower in f64; target gaming GPUs |
| No System/Fragment distinction | Explicit distinction (D12) | Fragments in QM/QM can't converge independently |
| Per-call Kernel construction | Cached Kernel objects (D14) | Tight SCC loop needs low launch overhead |
