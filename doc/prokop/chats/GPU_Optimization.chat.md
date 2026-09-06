https://chatgpt.com/share/6a9cae4a-723c-83ed-9d37-60e5ad4e8875

---

# USER

look we just implemented both methods - multi system GPU solver for small systems, and sparse density-matrix purification solver for large systems

Analyse deeply both the attached report and the relvant code on the git focusing mostly on performance optimization opportunities, especially focusing on the opencl kernels itself the .cl files. But maybe also architectural changes (big picture) and possible remaining overheads, redudancy and bad organization (layout, algoritm flow)


https://github.com/ProkopHapala/dftbplus/commit/b3073ef4c3ad2fb3fd0a7f03cdb672f6fd3db3b6

---

# Current-code review — 2026-09-06 (Devin)

This section supersedes the priority assumptions below; the original analysis is preserved. Inspection covered both Rust/OpenCL paths, with two bounded read-only agent audits. No GPU speedup is claimed from inspection alone.

## Confirmed blockers and corrections

- **Replica >=128 is a real assembly OOB**, not evidence of GPU saturation: `dftb_hamiltonian.cl:419–430` caches 128 fragments but indexes by unrestricted `p.replica`. Repair with a direct global metadata load, preserving ragged-fragment support; homogeneous arithmetic offsets are an optional later specialization, not necessary for the fix. Do not benchmark assembled batch441 until this is repaired and H/S parity checked around replicas 127/128.
- **Device selection matters on this machine:** `clinfo -l` reports Intel graphics as platform 0 and NVIDIA GTX 1650 (4 GB) as platform 1. `GpuRuntime::new` selects the default platform, while its capability record is hard-coded to RTX 3090. Use `OCL_DEFAULT_PLATFORM_IDX=1` here, verify the actual device name, and query real device/kernel resource limits. Previous reports cannot be retroactively assigned to a device without evidence.
- **Dense phase timings mix enqueue and execution.** `gpu_scc.rs::timed!` uses host clocks and Jacobi drains the queue. Use event START/END timestamps for kernel comparisons, separate from complete-solve wall time and compilation/setup. The old Jacobi benchmark additionally compares different algorithms/tolerances with unequal readback costs; it is not a controlled old/new kernel experiment.
- **Do NOT use the proposed Frobenius-minus-diagonal convergence shortcut in f32.** Near convergence it subtracts nearly equal numbers; the useful squared residual can be below f32 resolution. Odd N also introduces a 1e30 dummy diagonal, whose square overflows. Keep the direct off-diagonal reduction. Adaptive eigensolver tolerances are deferred, not a shortcut to performance acceptance.
- **Existing Jacobi convergence reporting is incomplete:** pair rotations stop at an absolute 1e-7 threshold but sweep convergence is relative to the initial off-norm. Some problems therefore exhaust 20 sweeps; the kernel returns a zeroed off-diagonal matrix without reporting the remaining residual or exhaustion. Future status reporting must retain the actual residual and fail on unmet accuracy. Reconstruction and `AV−VΛ` against the original matrix, not the overwritten diagonal output, are essential tests now.
- **Sparse TC2 really does five SpGEMMs on each nonconverged iteration:** `tc2_purify` computes KSK, separately recomputes KS, then `tc2_step` repeats both products. Intermediate matrices are repeatedly allocated/uploaded/read back; kernel handles and transpose maps are rebuilt. The right redesign is a persistent fixed-mask workspace with exactly two products, device-side branch/update, and reused diagnostic reductions.
- **The exposed Rust sparse route is not yet a sparse large-ribbon benchmark.** `dftb_engine.rs::rhai_run_sparse_purify` uses `build_full_mask`, requires four orbitals/atom, and derives occupations from a hard-coded species table. The CPU/Python H-passivated ribbon frontier solver is a separate path. Neither demonstrates device-resident, linear-scaling, mixed C/H GPU SCC.
- **Wide-row failure must be loud:** the sparse kernels can return without writing when a row exceeds `MAX_LEFT_BLOCKS`. A zero-initialized output is still a wrong answer, not a safe fallback. Validate all participating row degrees against the compiled capacity before launch. Validate reduction WG power-of-two/device limits too.
- **Do not simply reduce masks or remove symmetrization.** With T truncated, `P_K(P_T(KS)K)` need not be symmetric even for symmetric K/S masks. Product closure and cutoff convergence must be established; use the existing Boolean product-mask machinery and compare true dense density, electron count, idempotency and Hamiltonian commutator. Idempotency alone also accepts the wrong occupied projector. Small-gap ribbons may require long-range density support; linear scaling cannot be assumed at fixed physical accuracy.

## Prioritized work / handoff

| Priority | Work | Acceptance / scope |
|---|---|---|
| P0, straightforward handoff | Direct fragment metadata load + replica-boundary H/S regression; actual device/capability queries | Preserve heterogeneous layouts; no test-tolerance changes |
| P0, substantial sparse project | Persistent S/K/T/Q workspace, immutable structure/maps and cached kernels; two products per TC2 step | First preserve current masks/numerics; measure transfers/launches and compare every iterate to CPU. Then add validated geometric masks and C/H support as separate changes |
| **Selected here: challenging bounded kernel work** | Direct disjoint 2×2 Jacobi block transform, 4 → 2 barriers per round | Retain old compile-time variant, same schedule/tolerance/sweep cap, original-matrix residual + orthogonality + spectrum parity; event-profile on NVIDIA; molecular SCC regression |
| P1, dense architecture | Persistent SCC workspace and kernel handles; GPU eigenvalue ranking/occupation | Remove source rendering/hash lookup, allocations, N² readback and occupation upload from iteration loop; stable index tie-breaking, finite checks |
| P1, dense mathematics | Geometry-only H0′/Y, single weighted GEMM SCC transform; direct occupied-orbital Mulliken | Validate overlap conditioning, charge/energy conventions; build density once only if final outputs/forces require it |
| P1 after residency | Sparse row-degree variants; symbolic SpGEMM plan | Measure degree/intersection distributions and plan-memory size; preserve output ownership, no atomics; compare event time and total solve time |
| P2, straightforward handoff | Allocation-free CPU DIIS, right-sized small-vector WGs, atom-orbital offsets, S device-to-device copy | Independent surgical changes, one at a time; do not silently alter mixer/convergence behavior |
| P2 after profiling | Active-system lists, GPU DIIS, fused KSK | Measure benefit after transfers/allocations are removed; account for topology and local-memory footprint |

The selected Jacobi change is **not** asserted to outrank sparse residency globally. It is a suitable use of this session's expensive reasoning: a small but mathematically delicate kernel transformation with an existing molecular verification path. Sparse residency needs a coherent workspace/API change, not a quick partial wrapper rewrite.

## Kernel-level reasoning

For each round, partition A into ordered pair blocks B=(rows p,q; columns r,s). One work-item owns all four values and computes `B′ = R_a^T B R_b` entirely in registers. Blocks are disjoint, including diagonal blocks; reading/writing one's own block cannot race with another block. V updates touch independent local storage and can follow without an intervening A barrier. Only the rotation-publication barrier and round-end barrier remain. Initially keep both symmetric halves independently rather than forcing symmetry and hiding drift. Keep local footprint unchanged and avoid per-thread arrays.

The current A/V global loads are row-strided across lanes; cooperative linear loads are a later isolated optimization. Padding `JLD=JN+1` is not universally bank-conflict-free (e.g. JN=28 gives stride29, whereas other mappings have other conflicts); measure actual lane addresses. For sparse BSR4, TEAM16 means two output blocks/warp with potentially divergent intersection walks; TEAM32 can waste lanes, so neither is automatically superior. Static 17 KiB left-row caches may waste residency on small-degree ribbons. Symbolic plans trade traversal branches for index traffic and memory capacity. No register-spill or bandwidth-bound claim is justified until resource/profiler evidence exists.

## Verification checkpoint before implementation

Existing `gpu_eigenproblem` release suite on explicitly selected GTX 1650: **10/10 passed**, SK inputs present (no skips). Batched H2O max eigenvalue difference `1.0331e-6`, eigenvector difference `1.1567e-5`. Original one-shot host-wall Jacobi measurements: N32/batch100 `8.721 ms`, N64/batch100 `37.516 ms`; these include wrapper overhead and are not the controlled event-profile baseline. The implementation and controlled benchmark results follow; acceptance remains pending USER review.

## Implemented and tested — pending USER acceptance

Only the Jacobi implementation/test module was changed on the compute side (`gpu_eigen.cl`, `gpu_eigen.rs`). The old two-pass update remains under `JACOBI_BLOCK_UPDATE=0`; default `1` uses the disjoint pair-block update. No new production buffers, kernel launches, host transfers, private arrays, or per-step allocations were added. Existing wrapper allocation/synchronization problems remain explicitly deferred. Existing comments and historical code are retained; old comments claiming “one barrier” are historical, not the current count (two per round in the block path, four in the two-pass path).

### Numerical finding and root-cause refinement

The new diagnostic initially failed on the **unchanged original** N63/64 eigensolver. At N64, reconstruction error was `4.47175e-4` against an absolute `1e-4` criterion. Increasing sweeps did not help:

| Original N64, sweep cap | Off-diagonal Frobenius norm | max abs(AV−VB), full transformed B | max abs(VᵀV−I) |
|---|---:|---:|---:|
| 1 | 6.06463 | 6.52295e-5 | 6.45438e-6 |
| 4 | 1.93894e-2 | 2.16708e-4 | 2.27247e-5 |
| 8 | 3.94888e-6 | 2.16693e-4 | 2.26004e-5 |
| 20 | 3.94888e-6 | 2.16693e-4 | 2.26004e-5 |

All column norms drifted below one, trace drift reached `−6.38913e-3`, and the full-B transformation residual was already large before diagonal convergence. This isolates accumulated nonorthogonality in the rotations, rather than a need for additional sweeps or a looser stopping tolerance. Simply substituting `rsqrt` for `1/sqrt` produced identical results on this driver.

A single Newton normalization refinement is now applied **before** using each nonidentity rotation:

```c
float err = fma(-s, s, fma(-c, c, 1.0f));
c = fma(0.5f * c, err, c);
s = fma(0.5f * s, err, s);
```

This evaluates `e=1−c²−s²` with compensated products and applies the first-order inverse-norm correction `(c,s) *= 1+e/2`. It is not clamping, output renormalization, eigenvalue retuning, or an extra eigensolver iteration. It improves the rotation's defining invariant at the source. `JACOBI_NORMALIZE_ROTATION=0` retains the unrefined reference for diagnostics. Pair threshold `1e-7`, relative threshold `1e-7`, and maximum 20 sweeps are unchanged.

At N64 the refined kernel gives max spectrum error `2.90006e-5`, residual `2.75326e-5`, reconstruction `4.00673e-5`, orthogonality `2.21172e-6`. At N28 reconstruction improves from `7.18305e-5` to `6.87396e-6`. The original `1e-4` test criterion now passes, including N63 and N64; it was not loosened. The missing production residual/exhaustion status is still a separate open issue.

### Controlled NVIDIA event benchmark

GTX 1650, driver 530.41.03; real queried device local memory 49,152 B. Seven timed repetitions after warmup, alternating variant order, same inputs, fixed WG sizes, persistent buffers/programs/kernel handles. Input reset is a device copy outside the timer. No host matrix transfer occurs in the timed region. Test matrices alternate a Householder-rotated spectrum, a degenerate spectrum, and a diagonal matrix; these are **synthetic kernel measurements**, not measured formic-dimer SCC throughput. GPU clocks were not locked; interleaved ratios are more trustworthy than absolute cross-run times.

Both compared update variants use the normalization refinement, so this measures the **block-transform speedup at equal numerical accuracy**, not a faster-but-less-accurate original reference. All A/V outputs were **bitwise identical between the two update variants** for the tested fixtures. That observation is device/compiler-specific, not a cross-vendor guarantee.

| N | batch | WG | Two-pass event time (ms) | Pair-block event time (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 8 | 100 | 32 | 0.166336 | 0.139200 | 1.195× |
| 28 | 1 | 128 | 1.258048 | 0.731008 | 1.721× |
| 28 | 100 | 128 | 4.261152 | 2.647136 | 1.610× |
| 28 | 1000 | 128 | 23.236608 | 14.649440 | 1.586× |
| 32 | 100 | 128 | 3.731264 | 1.966080 | 1.898× |
| 64 | 100 | 256 | 24.629920 | 14.589248 | 1.688× |

The N28/batch100 enqueue+wait median was `4.272711 → 2.656299 ms`. Local memory is unchanged: N28 `7,236 B`, N32 `9,220 B`, N64 `34,820 B`; queried private memory reports zero for both variants. This is not a register-count or hardware spill-counter measurement. No occupancy percentage is inferred from it. No end-to-end H-bond or sparse speedup is claimed.

### Validation / reproducible commands

From `rust_dftb/`, with `OPENBLAS_NUM_THREADS=1`, `RUST_BACKTRACE=full`, `CARGO_TARGET_DIR=/home/prokophapala/.cargo-target-shared`, `OCL_DEFAULT_PLATFORM_IDX=1` (verify platform index on other machines):

```bash
cargo test --release --lib jacobi_block -- --include-ignored --nocapture --test-threads=1
cargo test --release --lib jacobi_sweep_diagnostic -- --ignored --nocapture --test-threads=1
RUST_DFTB_SK_DIR=/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1 cargo test --release --test gpu_eigenproblem --test gpu_scc -- --nocapture --test-threads=1
```

- New parity/profile tests: **2/2 pass**; N=1,2,3,7,8,14,27,28,32,63,64, including padded odd sizes and degenerate spectra. Checks use original f32 inputs evaluated by f64 CPU reference, finite values, eigenvalues, reconstruction, `AV−VΛ`, orthogonality, and update-variant equivalence.
- Existing eigensolver/S⁻¹/² suite: **10/10 pass**, real SK data loaded. Batched H2O eigenvalue difference `3.97375e-7`, eigenvector difference `1.04944e-5`.
- Existing CPU-H/S → GPU SCC suite: **3/3 pass**. Ten-geometry H2O batch max energy difference `9.54e-7 Ha`, population difference `3.34e-6`, eigenvalue difference `1.07e-6 Ha`; 24 SCC iterations. N2 energy difference `1.12e-6 Ha`, population difference zero. These tests validate SCC independently of GPU H/S assembly.
- **Remaining integration blocker:** the unchanged `hbond_gpu_scc::test_formic_dimer_single_point_scc` failed on NVIDIA at `assemble_pairs` enqueue with `CL_OUT_OF_RESOURCES`, before SCC. It was reproduced before the Jacobi implementation change; the geometry and test were not altered. Root resource cause is not yet established. The source declares ~22 KiB local storage, below the queried 48 KiB device limit, so blaming local-memory capacity alone would be unsupported. Next step: query the compiled assembly kernel's resource limits and inspect launch/input bounds; preserve this failure rather than switch devices and call it NVIDIA validation.

### Follow-up priorities after this experiment

1. Assembly resource failure and replica128 OOB: small, isolated handoff with compiled-kernel resource logging and boundary H/S parity tests.
2. Production Jacobi convergence-status reporting and consistent absolute/relative threshold semantics. Preserve direct residual checks; do not erase nonconvergence evidence on output.
3. Persistent dense/sparse workspaces and event profiling of complete loops. The current patch changes kernel execution, not the existing host overheads.
4. Sparse residency → fixed-mask reference parity → validated physical cutoff/C/H support → degree/symbolic-plan tuning. Do not collapse these into a single untestable rewrite.

---

# Chat GPT 5.6 sol

I went through the two reports and the implementation at commit `b3073ef4...`, especially `gpu_eigen.cl`, `gpu_matrix_ops.cl`, `dftb_hamiltonian.cl`, `gpu_scc.rs`, and the BSR4 sparse kernels/wrapper.

There are several substantial opportunities, but **three findings dominate everything else**:

1. The `batch=441` failure is almost certainly not mysterious GPU saturation: `assemble_pairs()` still has `__local Fragment l_frags[128]` and then indexes `l_frags[p.replica]`. System 128 onward is out-of-bounds. That matches the observed failure onset around replicas 128–135 almost absurdly well. 
2. The reported per-phase timing of the dense GPU SCC is currently **not trustworthy**. Most kernels are asynchronously enqueued, but the Jacobi wrapper calls `rt.finish()`. Consequently much of the work attributed to “Jacobi = 5.02 ms” can actually be execution of the preceding `delta_q → gamma → H_scc → GEMMs` queue. The total wall-clock throughput is valid; the 61% Jacobi pie slice is not yet a valid GPU profile. 
3. The sparse OpenCL kernels are reasonably designed, but the **Rust wrapper currently destroys the entire device-resident architecture**. A single TC2 iteration is doing approximately **five SpGEMMs instead of the mathematically necessary two**, with repeated GPU→CPU→GPU conversions and `finish()` after nearly every kernel. This is much more important than further tuning the BSR4 kernel arithmetic.

Those are where I would start.

---

# 1. Small-system batched GPU solver

Your overall architecture is now basically right:

$$
\text{many homogeneous systems}
\rightarrow
\text{one WG/system for local operations}
\rightarrow
\text{CPU orchestrates kernels}
$$

and the measured scaling up to batch=100 is encouraging: about 1,195 systems/s for the 28-orbital dimer. 

But the current SCC loop still materializes several matrices that are mathematically unnecessary.

## The batch=441 corruption is a concrete bug

In `assemble_pairs`:

```c
__local Fragment l_frags[128];

for(int i=tid; i<n_frags && i<128; i+=wg) l_frags[i]=fragments[i];
...
Fragment frag=l_frags[p.replica];
```

For replica 128:

```text
l_frags[128]    // OOB
```

This should be fixed before interpreting *any* batch>127 benchmark.

And for your homogeneous-batch use case, I would not fix it by enlarging the array. **Delete this metadata cache entirely.**

All replicas have identical topology:

$$
N_a,\quad N_{\rm orb}
$$

so:

```c
sid      = p.replica;
atom_off = sid*n_atoms;
H_base   = sid*n*n;
```

No `Fragment` lookup is necessary.

Similarly, `onsite_diagonal()` currently determines an atom's fragment by linearly scanning all fragments. For a homogeneous batch that should simply be:

```c
sid   = gid/n_atoms;
iat   = gid-sid*n_atoms;
baseH = sid*n*n;
```

The present fragment abstraction is useful for generic ragged QM/QM systems, but it is actively getting in the way of your **homogeneous high-throughput kernel path**.

Keep two layouts:

```text
HomogeneousBatch     // fast arithmetic offsets
RaggedFragments      // generic fallback
```

Don't burden the fast path with the generic metadata.

---

# 2. Fix profiling before optimizing Jacobi

The report says:

$$
t_{\rm Jacobi}=5.02{\rm ms}=61\%.
$$



But the wrappers behave approximately like:

```text
enqueue delta_q
enqueue gamma
enqueue H
enqueue GEMM
enqueue GEMM
enqueue Jacobi
finish()                 ← synchronization occurs here
```

and `timed!()` measures host time around each Rust call. `jacobi_cyclic_local_batched()` explicitly finishes the queue.

Therefore what gets measured is closer to:

```text
delta_q  : enqueue overhead
gamma    : enqueue overhead
H        : enqueue overhead
GEMMs    : enqueue overhead
Jacobi   : previous GPU work + Jacobi + synchronization
```

This probably explains the strange observation that GEMM is almost independent of batch size.

### Use OpenCL event profiling

Create the benchmarking queue with profiling enabled and retain one `Event` per enqueue.

Measure:

$$
t_k=t_{\rm END}-t_{\rm START}
$$

from OpenCL device timestamps.

Then report separately:

```text
GPU execution
host enqueue overhead
PCIe synchronization/readback
CPU DIIS
```

This may substantially revise the optimization ranking.

For production, of course, keep the asynchronous queue.

---

# 3. The Jacobi kernel does have a lot of synchronization

Even if the exact 61% is suspect, the kernel itself has obvious room.

For N=28 you have:

$$
14\ {\rm pairs/round}\times27\ {\rm rounds/sweep}.
$$

The report describes “one barrier per round,” but the actual kernel has effectively **four barriers per round**:

```text
calculate c,s
barrier

A column rotation
barrier

A row rotation
barrier

V column rotation
barrier
```

plus convergence-reduction barriers after every sweep.

At 20 sweeps:

$$
20\times27\times4=2160
$$

full-WG barriers/system.

That is a lot for a 28×28 problem.

## A much better Jacobi round formulation

Once all disjoint rotations

$$
R_0,R_1,\ldots,R_{N/2-1}
$$

are known, don't implement

$$
A\leftarrow AR
$$

followed by

$$
A\leftarrow R^TA.
$$

Partition A into 2×2 blocks corresponding to Jacobi pairs.

For pair \(a=(p,q)\) and pair \(b=(r,s)\):

$$
B=
\begin{pmatrix}
A_{pr}&A_{ps}\\
A_{qr}&A_{qs}
\end{pmatrix}.
$$

Then directly calculate

$$
\boxed{B'=R_a^TBR_b}.
$$

Every matrix element belongs to exactly one such pair-pair block.

Thus a round becomes approximately:

```text
compute all R_a
barrier

parallel over pair-pair 2×2 blocks:
    B' = Ra^T * B * Rb

parallel over V rows/pairs:
    V[:,pq] = V[:,pq] * Ra

barrier
```

So **~2 barriers/round instead of 4**.

For N=28:

$$
14(14+1)/2=105
$$

unique symmetric pair blocks. That maps very naturally to a WG of 128.

This is the Jacobi kernel rewrite I would prioritize.

---

# 4. Stop executing identity rotations

Current logic does:

```c
if(fabs(apq)<tol){
    c=1;
    s=0;
}
```

but then all threads still execute the A and V update loops.

Late sweeps therefore spend almost the same work on already-converged pairs.

Keep:

```c
active[ipair] = fabs(apq)>threshold;
```

and skip the pair-block operations where both relevant rotations are identities.

Early sweeps will still be dense, but late sweeps should become much cheaper.

---

# 5. The convergence norm can be much cheaper

Currently every sweep scans all off-diagonal elements:

$$
\sum_{i\ne j}A_{ij}^2.
$$

But orthogonal Jacobi transformations preserve the Frobenius norm:

$$
\|A\|_F^2=\text{constant}.
$$

Therefore:

$$
\boxed{
\|A_{\rm off}\|_F^2
=
\|A\|_F^2-\sum_iA_{ii}^2
}
$$

Compute \(\|A\|_F^2\) once.

After a sweep, reduce only N diagonal values, not \(N^2-N\) off-diagonal values.

And I would not necessarily check every sweep. Try every 2–4 sweeps.

---

# 6. Your Jacobi tolerance is probably far tighter than the SCC requires

SCC convergence is:

$$
\mathrm{RMS}(\Delta q)<5\times10^{-6},
$$

but Jacobi uses relative off-diagonal convergence around \(10^{-7}\). 

You probably do not need essentially fully converged eigenvectors in the first SCC iterations, when the charges themselves may still be wrong by \(10^{-2}\).

A natural adaptive strategy is:

$$
\epsilon_{\rm eig}
=
\mathrm{clamp}(c\,R_{\rm SCC},\epsilon_{\min},\epsilon_{\max}).
$$

Something like:

```text
early SCC       Jacobi ~1e-3 or 1e-4
middle          ~1e-5
final           ~1e-6
```

must of course be tested against final charge/energy parity.

The important experiment is simply record:

```text
sweeps actually required/system/iteration
```

for tolerances \(10^{-3}\dots10^{-7}\).

You currently have no data demonstrating that 20 deep sweeps are needed.

---

# 7. Remove host eigenvalue sorting completely

The current path is particularly wasteful:

```text
Jacobi leaves eigenvalues on diagonal of N×N hp
↓
read entire N² hp to CPU
↓
sort N values
↓
make occupation mask
↓
upload mask
```

The report already notices the N² readback. 

But I would go further than adding a diagonal extraction kernel.

Do the sort **inside the Jacobi workgroup while eigenvalues are still local**.

For N≤64 even an \(O(N^2)\) rank sort is tiny:

$$
r_i=
\sum_j [\epsilon_j<\epsilon_i]
$$

with index tie-breaking.

Then:

```c
occ[i] = rank[i] < nocc;
eval_sorted[rank[i]] = eig[i];
```

Optionally also reorder eigenvector columns.

Then global output is only:

```text
eigenvalues[batch*N]
occupied eigenvectors / indices
```

not the destroyed N×N `hp`.

That eliminates a synchronization point and PCIe round trip every SCC iteration.

---

# 8. The largest dense-SCC architectural improvement: remove D entirely

Your current iteration does:

```text
C' from Jacobi
C = X C'                      N×N GEMM
D = 2 Cocc Cocc^T             build N×N
q = diag(D S)                 N² work
```

But Mulliken charges do not require D.

Let

$$
X=S^{-1/2},\qquad Y=S^{1/2}.
$$

If \(U\) are eigenvectors of the orthogonalized Hamiltonian:

$$
C=XU.
$$

Also:

$$
SC=S XU=YU
$$

because

$$
SX=S^{1/2}=Y.
$$

Therefore:

$$
\boxed{
q_A=
2\sum_{\mu\in A}\sum_{k\in occ}
(XU)_{\mu k}(YU)_{\mu k}.
}
$$

So after diagonalization compute only occupied columns:

$$
C_{\rm occ}=XU_{\rm occ},
$$

$$
SC_{\rm occ}=YU_{\rm occ}.
$$

Then directly reduce populations.

This eliminates:

```text
full C[N,N]
D[N,N]
density kernel
Mulliken D*S kernel
```

from every SCC iteration.

And while diagonalizing S once, constructing \(X\) and \(Y\) costs virtually the same:

$$
X=U_S\lambda^{-1/2}U_S^T,
$$

$$
Y=U_S\lambda^{+1/2}U_S^T.
$$

For the formic dimer \(N=28,n_{\rm occ}=18\), this isn't just theoretical—it should reduce both arithmetic and global traffic noticeably.

For forces, build D and the energy-weighted density only **once after SCC convergence**.

---

# 9. You can probably remove D from the energy calculation too

The current final energy uses:

$$
E_{H0}=\operatorname{Tr}(DH^0).
$$

But

$$
\operatorname{Tr}(DH_{\rm SCC})
=
2\sum_{k\in occ}\epsilon_k.
$$

Since

$$
H_{\rm SCC}=H^0+\frac12(SV+VS),
$$

and the SCC contribution contracts to:

$$
\operatorname{Tr}(D H_1)=\sum_Aq_AV_A,
$$

then

$$
\boxed{
E_{H0}
=
2\sum_{occ}\epsilon_k-\sum_Aq_AV_A.
}
$$

Combine that with your existing double-counting term, carefully preserving your charge/sign convention.

Parity-test this before replacing the current expression, but if it matches, **D disappears completely from scans where forces are not needed**.

---

# 10. Precompute \(H'_0\): DFTB gives an unusually nice shortcut

This is probably the second-biggest dense architectural optimization.

You currently build:

$$
H=H^0+\frac12(SV+VS)
$$

and then:

$$
H'=XHX
$$

using two GEMMs every SCC iteration.

Because \(X=S^{-1/2}\), define:

$$
Y=S^{1/2}=SX=XS.
$$

Then:

$$
\boxed{
H'
=
H'_0+
\frac12(XVY+YVX),
}
$$

where

$$
H'_0=XH^0X
$$

is computed once per geometry.

Since everything is symmetric:

$$
YVX=(XVY)^T.
$$

Therefore each SCC iteration needs only:

$$
A=XVY
$$

and:

$$
H'=H'_0+\frac12(A+A^T).
$$

And \(V\) is diagonal, so multiplying by V is just column/row scaling.

So:

```text
current:
    build Hscc
    GEMM X*Hscc
    GEMM temp*X

better:
    scale X by V
    GEMM (X*V)*Y
    symmetrize + H0'
```

One GEMM instead of two, and the physical-basis `H_scc` matrix can disappear from the SCC loop.

This is a very natural optimization for DFTB specifically.

---

# 11. Small kernels are often using absurd WG sizes

Examples in `gpu_matrix_ops.cl`/wrappers:

```text
residual/mix : WG=256 for 10 atoms
H update     : WG=256
reductions   : WG=256
gamma        : WG=n_atoms, e.g. 10
```

For a ten-atom system, running 256 threads and then doing an 8-step barrier tree reduction is unnecessary.

Use kernel/problem-specific configurations, e.g.:

```text
Na <= 32   → WG 32
Na <= 64   → WG 64
larger     → 128/256
```

For NVIDIA, subgroup/warp reduction would be even better for these tiny vectors.

Also `gamma_matvec_batched()` comments that it rounds to the preferred WG multiple, but its wrapper actually uses approximately `wg=n_atoms`.

The implementation and comments have drifted.

---

# 12. `orb_atom` is unnecessary repeated metadata for homogeneous systems

For every replica you currently carry:

```text
orb_atom[batch*N]
```

even though every replica has the same species topology.

For homogeneous templates it should be:

```text
orb_atom[N]
```

shared across all systems.

Even better for s/p systems, store:

```text
atom_orb_off[Na+1]
```

and don't search orbitals by owner in the Mulliken reduction.

Current Mulliken phase 2 does approximately:

```c
for each atom A:
    for every orbital mu:
        if(orb_atom[mu]==A) sum += diag[mu];
```

That is unnecessarily \(N_aN_{\rm orb}\).

With offsets:

```c
for(mu=orb_off[A]; mu<orb_off[A+1]; mu++) sum += diag[mu];
```

---

# 13. CPU DIIS: optimize it before immediately porting it to GPU

The report measures CPU DIIS at 1.35 ms/iteration for batch100. 

Looking at `gpu_scc.rs`, each system each iteration currently creates:

```rust
Vec<f64> residual
Vec<f64> q_inout
Vec<f64> q_out
```

plus the mixer histories.

So “CPU DIIS = 1.35ms” includes a lot of allocator/conversion noise.

Before writing GPU DIIS, make CPU DIIS flat and allocation-free:

```text
q_host       [batch*Na]
qnew_host    [batch*Na]
residual     [batch*Na]
history      [batch*hist*Na]
B            [batch*(hist+1)^2]
```

all persistent.

At Na≈10 the actual DIIS arithmetic is tiny.

GPU DIIS becomes attractive if you want thousands of systems without PCIe synchronization, but I would first find out how much of the current 1.35 ms is Rust allocation.

---

# 14. Active systems should eventually use an active-ID list, not merely a flag

An `active[sid]` mask is a good first step.

But if 80% of systems converge and you still launch one WG for every system:

```c
if(!active[sid]) return;
```

you still pay scheduling overhead for empty WGs.

For highly heterogeneous SCC convergence, periodically compact:

```text
active_ids = [3,7,11,29,...]
```

and launch:

```c
sid=active_ids[get_group_id(0)];
```

You already read charge vectors for CPU DIIS, so the host can cheaply construct the active list initially.

This would also make microbatch scheduling much easier later.

---

# 15. Cache kernels and persistent SCC workspaces

Every wrapper repeatedly does:

```rust
rt.build_program(source)?
Kernel::builder()...
```

The program is cached, but the Kernel object is rebuilt.

For a nine-iteration solve with ~10 kernels/iteration that's already many kernel constructions.

More important, `gpu_solve_scc_batched_diis_warmstart()` allocates all its working buffers anew on every solve.

For a scan/MD/NEB workflow the natural object is:

```text
GpuSccWorkspace<N,Na,batch>
    X,Y,H0'
    q,dq,V
    Hp,U
    Cocc,SCocc
    DIIS buffers
    eigenvalues
    cached kernel handles
```

and then:

```text
update_geometry()
solve_scc()
update_geometry()
solve_scc()
```

No allocation in the trajectory loop.

---

# 16. `GpuRuntime` is not actually querying the GPU

This one is easy to miss.

`query_capabilities()` currently returns hard-coded:

```text
48 KB local
1024 max WG
warp 32
82 SM
24 GB
```

i.e. RTX 3090 assumptions.

That directly contradicts your target of cheap gaming GPUs.

Query actual:

```text
CL_DEVICE_LOCAL_MEM_SIZE
CL_DEVICE_MAX_WORK_GROUP_SIZE
CL_DEVICE_MAX_COMPUTE_UNITS
CL_DEVICE_GLOBAL_MEM_SIZE
```

and preferably query the preferred WG multiple per compiled kernel.

This particularly matters for your full-local \(N=64\) kernels.

---

# 17. Sparse purification: the kernel design is reasonable; the wrapper is currently catastrophic

This is the biggest performance finding on the large-system side.

Your intended TC2 iteration is:

$$
T=KS
$$

$$
Q=TK=KSK
$$

$$
n=\operatorname{Tr}(T)
$$

$$
K'=
\begin{cases}
Q & n>N_{\rm occ}\\
2K-Q&n<N_{\rm occ}.
\end{cases}
$$

That is:

> **two sparse products per TC2 iteration.**

But the current high-level Rust flow does much more.

Inside `tc2_purify()`:

```text
ksk()
    T=KS             SpGEMM #1
    Q=TK             SpGEMM #2

then separately:
    T=KS             SpGEMM #3
    trace(T)

then tc2_step():
    T=KS             SpGEMM #4
    Q=TK             SpGEMM #5
    update K
```

So **five SpGEMMs rather than two**.

This alone is a theoretical 2.5× loss before considering transfers.

And every intermediate `matmul_masked_bsym()` currently:

```text
allocates CSR GPU buffers
uploads row_ptr
uploads col_idx
uploads values
allocates output
builds Kernel
enqueue
finish
reads values GPU→CPU
constructs Bsr4Matrix
```

That completely defeats the design.

---

# 18. Sparse matrices must become GPU objects, not host objects passed to kernels

Make the structure/value distinction explicit:

```text
GpuBsrStructure
    row_ptr GPU
    col_idx GPU
    diag_block GPU
    transpose_block GPU
    nrow,nblock

GpuBsrMatrix
    structure -> shared GpuBsrStructure
    values GPU
```

Structures \(M_S,M_K,M_T\) are uploaded **once**.

Persistent sparse workspace:

```text
S,H0,H           persistent
Z0,Z1
K0,K1
T
Q
scratch
partial_reduce
trace
residual
```

Then TC2 becomes literally:

```text
spgemm(K,S,T)          // enqueue
spgemm(T,K,Q)          // enqueue
trace(T,trace)         // enqueue
tc2(K,Q,trace,Knew)    // enqueue
swap(K,Knew)
```

No CPU traffic at all.

If checking convergence every iteration:

```text
idempotency(Q,K,residual)
```

stays device-side too.

Read one scalar only when necessary.

I would perhaps check convergence every 2–4 TC2 iterations once behavior is understood.

---

# 19. Newton-Schulz should also remain completely resident

The same applies to:

$$
Z_{n+1}=2Z_n-Z_nSZ_n.
$$

Current mathematical iteration only needs:

```text
T=Z*S
Q=T*Z
Znew=2Z-Q
```

and the residual can be derived from T.

Everything else should remain GPU-side.

S and its masks never change during those iterations.

---

# 20. Sparse SpGEMM: static 256-block local arrays hurt occupancy

The current kernel always reserves:

```c
__local uint  lcol[256];
__local float lA[256*16];
```

~17 KiB/WG even if the actual row contains 20 or 30 blocks.

For ribbons and other low-dimensional systems, this is likely unnecessarily expensive.

Use degree classes:

| max row degree | local A+index |
| -------------: | ------------: |
|             32 |      ~2.1 KiB |
|             64 |      ~4.3 KiB |
|            128 |      ~8.5 KiB |
|            256 |       ~17 KiB |

Build row buckets once:

```text
rows32
rows64
rows128
rows256
```

and launch specialized kernels:

```text
MAX_LEFT_BLOCKS=32
64
128
256
```

with `row_ids[]`.

This lets the GPU keep many more small-row WGs resident.

Given your quasi-1D/2D target geometries, I think this could be a large win.

---

# 21. Precompute sparse multiplication structure

Your symmetric-right kernel currently intersects sorted neighbor lists every product:

$$
N_A(i)\cap N_B(j).
$$

That is much better than binary searching each \(k\), but the exact same structural intersection is repeated:

```text
TC2 iteration 1
TC2 iteration 2
...
TC2 iteration 50
```

while the sparsity masks don't change.

So precompute a **symbolic product plan**.

For every output block \(C_{ij}\), store:

```text
plan_ptr[block]
packed pairs:
    (offset_Aik, offset_Bjk)
```

Since your maximum row is <=256, the offsets fit in `u8` or `u16`.

For example pack:

```c
uint pair = ia | (ib<<16);
```

Then the kernel is essentially:

```c
for(t=plan_ptr[cb]; t<plan_ptr[cb+1]; t++){
    ia = plan[t]&0xffff;
    ib = plan[t]>>16;
    4x4 FMA
}
```

No:

```text
ka<kb
kb<ka
branch
pointer chasing
```

during numerical multiplication.

There is a memory cost, so benchmark it. For 1D/2D sparse systems with moderate intersection length and dozens of repeated TC2 products, I strongly suspect it will pay.

This is exactly the separation known from sparse numerical libraries:

```text
symbolic multiplication        once
numeric multiplication         many times
```

and your fixed masks make it especially attractive.

---

# 22. TEAM=16 is not obviously optimal on NVIDIA

Currently:

```text
16 lanes = one 4×4 output block
```

so one NVIDIA warp simultaneously processes two unrelated output blocks.

If their two-pointer intersections follow different branches, the two half-warps diverge.

Benchmark at least:

```text
TEAM=16    two blocks/warp
TEAM=32    one block/warp
```

TEAM32 wastes arithmetic lanes if implemented naively, but gets perfectly coherent sparse traversal.

Another possibility is 8 threads/block, each thread producing two matrix elements.

The optimal choice depends on whether you are dominated by structural traversal or 4×4 FMAs. Don't assume 16 merely because there are 16 scalars.

---

# 23. Fused KSK is worthwhile, but only after fixing residency

I still like the fused idea:

$$
Q=KSK.
$$

One WG/row can create \(T_i=(KS)_i\) in local memory and immediately use it:

```text
load K row
construct T row into local
barrier
construct Q row
```

so global T disappears.

But I would **not implement this yet**.

First get:

```text
2 resident SpGEMMs / TC2
```

rather than the current:

```text
5 SpGEMMs + repeated PCIe transfers.
```

Only then will profiling tell you whether T traffic is actually dominant.

Degree-specialized kernels are also a prerequisite, because a large \(T_i\) local row can otherwise kill occupancy.

---

# 24. Symmetrization every iteration may be unnecessary

`bsr4_symmetrize()` is sensible as a diagnostic/repair kernel.

But if:

```text
K mask is symmetric
S is symmetric
products use Bsym path correctly
```

then TC2 ought to preserve symmetry except for truncation/roundoff.

Test:

$$
\|K-K^T\|_F
$$

with symmetrization every:

```text
1, 2, 4, 8, never
```

iterations.

If asymmetry stays around f32 noise, remove it from the hot loop.

---

# 25. H-passivated ribbons expose an important mismatch

The GPU BSR4 TC2 path currently assumes four orbitals/atom and therefore pure C; the project audit explicitly says H is not supported.

But the attached large-ribbon benchmark uses **H-passivated ribbons**, with 132 H atoms in the N=1156 case. 

Before implementing variable-size block CSR, I would seriously test **padded BSR4**:

```text
C : 4 active orbitals
H : 1 active + 3 dummy slots
```

For dummy orbitals:

$$
S_{dd}=1,\qquad K_{dd}=0,
$$

all couplings zero.

Exclude dummy orbitals from Mulliken and spectral-bound calculations and explicitly keep their K entries zero.

This wastes storage for H but keeps the beautifully simple uniform 4×4 GPU kernel.

For your ribbon composition, this may still be much faster than introducing 1×1,1×4,4×1,4×4 irregular sparse products.

I would benchmark padded BSR4 before designing VBSR.

---

# 26. The second attached report is actually a third solver, not the purification solver

This distinction matters.

The sparse scaling report benchmarks:

> sparse Cholesky + implicit \(L^{-1}HL^{-T}\) + Chebyshev/Ritz **on CPU/Python**, for HOMO/LUMO only.

It explicitly says this is different from the dense GPU SCC path. 

So its 35.7 s at N=1156 tells us essentially nothing about BSR4 TC2 performance.

But it gives useful information about topology.

And I found two important issues in that report/code.

---

# 27. The claimed Cholesky scaling is actually almost quadratic

The report says the decreasing fill percentage:

$$
31.7\%\rightarrow26.1\%
$$

“confirms sub-\(O(N^2)\) fill-in.” 

But look at the actual large-N data:

$$
N:292\rightarrow580\rightarrow1156
$$

$$
nnz(L):23441\rightarrow89441\rightarrow348929.
$$

Doubling N almost quadruples `nnz(L)`:

$$
89441/23441=3.82
$$

$$
348929/89441=3.90.
$$

That is approximately:

$$
\boxed{nnz(L)\sim N^2}
$$

over the large-size part of the benchmark.

A roughly constant fill percentage of ~26% actually means quadratic storage, not linear storage.

So the interpretation in the report is backwards.

This makes the purification path more attractive, because its masked \(K,S\) representation avoids elimination fill entirely.

---

# 28. The Cholesky ordering is also suspiciously poor

The report talks about sparse/fill-reduced Cholesky, but the actual Python code calls:

```python
splu(... permc_spec='NATURAL', ...)
```

For a quasi-1D width-4 ribbon with a local H/S graph, a good ordering should normally produce something much closer to banded sparse elimination.

Yet at N=1156:

$$
nnz(S)=51548
$$

versus:

$$
nnz(L)=348929.
$$



Before concluding Cholesky necessarily has bad fill here, try a geometry/topology-aware ordering:

```text
atoms ordered along ribbon axis
reverse Cuthill-McKee
nested dissection / AMD if available
```

and apply the same symmetric permutation to H and S.

For this quasi-1D test I would expect potentially dramatic improvement.

Again, this mainly matters for the frontier eigensolver route; TC2 doesn't need Cholesky.

---

# 29. The “SpMV count” in that report is also misreported

The table gives, for example:

```text
nvec=20
degree=40
iters=20
SpMV/band=128,040
```



But:

$$
20\times40\times20=16000.
$$

The benchmark script stores:

```python
total_spmv = spmv_h + spmv_l
```

so its reported counter is for **both HOMO and LUMO bands**, not “per band.”

Additionally, `run_chebyshev_ritz_band()` appears to double-count the square-filter factor: it sets

```python
spmv_per_app = degree*2
```

and then applies the polynomial twice and adds `spmv_per_app` after *both* applications.

So the timing itself is real, but the SpMV accounting should be audited before using it for algorithmic complexity conclusions.

---

# 30. Big-picture organization

Right now you have accumulated several generations of implementation:

```text
GpuMatrixContext
GpuRuntime
legacy dense purification kernels
Brent-Luk eigensolver
new SCC kernels
BSR4 sparse runtime
Python Cholesky/Chebyshev solver
```

Some duplicate concepts and some create separate contexts.

In particular `SparseBsr4Gpu::new()` creates a new `GpuRuntime::new()`, hence a new OpenCL context, even though its comment says it uses the shared runtime.

I would converge toward:

```text
GpuRuntime
    context
    device
    queues
    event profiler
    program cache
    kernel cache

DenseBatchOps
    eigen
    GEMM
    SCC

SparseOps
    immutable CSR structures
    SpGEMM
    TC2/LNV

DftbGpuTemplate
    species topology
    SK data
    q0
    orb offsets

DenseSccWorkspace
SparseSccWorkspace
```

Everything shares the same runtime and buffers.

`GpuMatrixContext` should eventually disappear or become a thin compatibility wrapper.

---

# My revised priority order

| Priority | Change                                                     | Expected importance                                        |
| -------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| **P0**   | Fix `l_frags[128]` OOB                                     | correctness; probably fixes batch441                       |
| **P0**   | OpenCL event profiling                                     | makes all optimization decisions trustworthy               |
| **P0**   | Make sparse solver fully device-resident                   | enormous; currently fundamentally defeating GPU            |
| **P0**   | Remove duplicate TC2 products: 5 → 2 SpGEMM/iter           | potentially ~2.5× sparse solver immediately                |
| **P1**   | Rewrite Jacobi round as direct 2×2 pair-block transform    | ~2× fewer barriers + less redundant work                   |
| **P1**   | GPU-local eigenvalue rank/sort                             | removes per-iteration host synchronization                 |
| **P1**   | Eliminate D during SCC using \(XU_{\rm occ},YU_{\rm occ}\) | removes several dense N²/N³ operations and buffer          |
| **P1**   | Precompute \(H'_0\), use \(XVY\) SCC update                | 2 GEMMs → 1 GEMM/iteration                                 |
| **P1**   | Sparse row-degree kernel variants                          | substantially better occupancy for low-dimensional systems |
| **P1**   | Precompute symbolic SpGEMM plans                           | remove repeated CSR intersections                          |
| **P2**   | Allocation-free CPU DIIS / then GPU DIIS                   | relevant for batch100–1000                                 |
| **P2**   | Active-ID compaction                                       | strong with heterogeneous convergence                      |
| **P2**   | Fused local `KSK`                                          | memory-bandwidth optimization after resident baseline      |
| **P2**   | padded-BSR4 H support                                      | lets purification handle your actual passivated ribbons    |
| **P3**   | dynamic numerical sparsification                           | only after cutoff convergence tests                        |

The most useful next engineering step is actually quite narrow: **fix the replica-128 bug, add real GPU event timings, then rewrite the sparse wrapper to keep structures and values resident and benchmark a TC2 iteration that contains exactly two SpGEMMs.** After those three changes, the next profiler run will tell us whether the Jacobi block transform or sparse symbolic-plan optimization deserves attention first.
