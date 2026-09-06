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

---

# 31. Algorithmic-alternatives audit — 2026-09-06 (Devin, after Astra ran out of tokens)

Astra's run (sections 1–30 above) ended mid-audit. Two read-only subagent audits were dispatched to finish the broader algorithmic comparison (NMP sparse methods + divide-and-conquer; current masking/purification hazards). This section records what survived that audit so it is not lost. No code was changed in this phase.

## 31.1 What NumericalMathPlayground actually contains (verified, not claimed)

The NMP repo is a collection of **proof-of-concept kernels and scripts**, not a production nonorthogonal DFTB pipeline. Treat all quoted speedups as unverified for DFTB.

| File family | What it really does | What it does NOT do |
|---|---|---|
| `DensityMatrix/OrderN.py` | Dense `np.linalg.eigh` reference + Mulliken diagonal | No sparse DM, no forces, no SCC |
| `DensityMatrix/FOE.py` | Chebyshev expansion of Fermi-Dirac, stochastic trace, **diagonal-only** `rho_est`. Handles `S` via dense `S^{-1/2}` orthogonalization or dense `S\x=Hv` solves. | No off-diagonal `rho`, no full DM, no Mulliken bond populations, no forces. Dense `O(N^3)` fallbacks for `S`. |
| `DensityMatrix/GF.py` | Contour integration, **diagonal-only** `rho_accum`. Dense `np.linalg.solve` per pole per probe. | No finite-T, no full DM, no forces. |
| `OMM/OMM_ocl.py` + `cl/OMM.cl` | Local-orbital halo machinery, ELLPACK `apply_operator`/`project_orbitals`/`assemble_gradient`. CPU still does Gram matrix, Lambda mixing, S-norm. | `spmv_expand_S` in the Python reference is **dense** (`for col in range(n_basis)`). Energy is only `sum(diag(h_ij))`, not total DFTB energy. No forces. |
| `CheFSI/CheFSI.py` + `CheFSI.cl` | ELLPACK `spmm_ellpack_chebyshev` for **frontier orbitals only**. Inverse-sqrt and `eigh` on **CPU**. `n_vecs` hard-capped at 32. | No finite-T, no full DM, no charges/forces. Targets a single energy window, not the occupied manifold. |
| `Kekule_BOP/*.py` | Local Chebyshev/Lanczos probes for **per-bond off-diagonal densities** `rho_{ij}`. | Assumes **orthonormal** tight-binding (no `S`). No charges, no total energy, no forces. |
| `LinearAlgebra/FastDirectSolvers/nested_solver.py` | Geometric RCB / nested dissection, batched dense block-Jacobi on GPU (local-memory, `m<=48`), Ritz correction. | Vibration eigensolver, **not DFTB**. Diagonal mass matrix `M^{-1/2} K M^{-1/2}`. Root step is **dense `n×n` eigh** → still `O(N^3)`. All matrices converted to dense (`toarray()`). Assumes **positive semidefinite** spectrum. No `S`, no chemical potential, no occupation, no DM. |

**Verdict:** No single NMP file returns a full, approximate, nonorthogonal DFTB density matrix with charges + energy + forces. The reusable pieces are: (1) FOE's finite-T Chebyshev diagonal-charge estimator; (2) CheFSI's ELLPACK SpMM for frontier orbitals; (3) OMM's local-orbital/halo machinery; (4) Kekule_BOP's local-probe bond-density idea; (5) the AMLS recursion skeleton — but only after replacing the dense root eigh and the diagonal-mass assumption.

## 31.2 Why divide-and-conquer / nested dissection does not directly transfer

The NMP `nested_solver.py` is a vibration solver. Five concrete blockers for DFTB carbon ribbons:

1. **Overlap `S`, not diagonal mass.** `load_system_dense` builds `H = M^{-1/2} K M^{-1/2}` with a *diagonal* mass. DFTB needs a generalized eigensolver `H c = ε S c`, or `S^{-1/2} H S^{-1/2}`, or reduced generalized problems `X^T H X c = (X^T S X) c ε` inside each block and at each Ritz/AMLS level.
2. **Indefinite spectrum.** DFTB has negative and positive eigenvalues. The NMP code assumes positive semidefinite (`sqrt(clip(...,0,None))`). "Lowest `k` modes per block" is the wrong truncation for DFTB — frontier orbitals sit near `μ`, which can be anywhere. Need shift-and-invert, an energy window around `μ`, or contour-integral DM methods.
3. **Shared chemical potential / occupation.** No global `μ` or occupation logic exists in NMP. For full occupied DM you need a global Fermi level from the total/reduced DOS and electron count. For frontier eigenpairs, target `|ε−μ|<δ`.
4. **Dense root step.** `recursive_exact_amls` builds a dense `n×n` `V_local`, dense `H_reduced`, and calls `np.linalg.eigh` on it at the root. The small leaf eighs do not change asymptotics; for the reported N≈500–1700 this makes AMLS *slower* than a single `numpy.eigh`. Only **truncated Ritz** (keeping `k` modes/block) actually reduces complexity: `H_proj = V^T H V` costs `O(N m²) + O(m³)` with `m = k·n_blocks`.
5. **Dense conversions everywhere.** `toarray()`, dense `block_diag`, dense `V.T @ H @ V`. For large DFTB systems this runs out of memory long before the claimed "DOF > 5000" crossover.

The geometric RCB/ND tree builders and DOF reordering are structurally reusable if a graph partitioner is substituted. The batched dense block-Jacobi with local memory and CPU-fallback split is a useful template. But the AMLS recursion requires a sparse root solve or truncation to be viable for ribbons.

## 31.3 Astra's Cholesky-report critique (sections 27–29) — confirmed and sharpened

- **Quadratic, not sub-quadratic fill.** `nnz(L): 23441 → 89441 → 348929` as `N: 292 → 580 → 1156`. Ratios `3.82`, `3.90` ≈ `N^2`. A constant ~26% fill fraction means quadratic storage, not linear. The report's "sub-`O(N^2)` fill-in" interpretation is backwards. This makes masked purification (no elimination fill) more attractive for the large-system route.
- **Ordering is suspiciously poor.** The Python code uses `splu(... permc_spec='NATURAL', ...)`. For a quasi-1D width-4 ribbon with a local H/S graph, a geometry-aware ordering (atoms along ribbon axis, RCM, nested dissection / AMD) should produce something much closer to banded. At N=1156, `nnz(S)=51548` vs `nnz(L)=348929` is a ~6.8× blowup that a good ordering may largely remove. This matters only for the frontier-eigensolver route; TC2 purification does not need Cholesky.
- **SpMV accounting is misreported.** The table's "SpMV/band=128040" for `nvec=20, degree=40, iters=20` is actually `20×40×20=16000` per band; the script stores `total_spmv = spmv_h + spmv_l` (both bands) and `run_chebyshev_ritz_band()` appears to double-count the square-filter factor (`spmv_per_app = degree*2` added after *both* applications). Timing is real; SpMV accounting must be audited before using it for complexity conclusions.

## 31.4 Masking and approximation — physical safety requirements

From the `DensityMatrix_Idenpotency_NearestNeighbor.chat.md` (Gemini) discussion and the current code:

- **Mask of `P` must be ~2–3× larger than mask of `S`**, not equal to it. Forcing `mask(P) = mask(S)` makes "Pauli repulsion" too stiff and bond energies significantly off. The current `build_full_mask` in `dftb_engine.rs` is the opposite extreme (fully dense block structure) and destroys sparsity; `build_geometric_mask` and `build_product_mask` are the physically sensible middle ground but must be validated for closure under repeated TC2/Newton–Schulz products.
- **Mulliken charges are necessary but not sufficient** for the Pauli principle. A matrix can have perfect Mulliken charges while having eigenvalues `2.0` and `-1.0`. Acceptance must include idempotency `R_I = ||KSK−K||` and Hamiltonian commutator `R_H = ||HKS−SKH||_F`, not just `Tr(KS)=N_occ` and per-atom charges.
- **Truncation breaks idempotency.** Any `P_{ij}=0` cutoff must be followed by purification to re-enforce `PSP=P`. The question is whether the cutoff + purification cycle converges to a physically acceptable DM at the given gap. For insulators, yes (exponential decay). For small-gap/metallic ribbons, **linear scaling cannot be assumed at fixed physical accuracy** — the DM support grows as the gap shrinks.
- **Symmetry preservation under thresholding.** `bsr4_drop_small` currently zeros values without compacting CSR and treats transpose blocks independently, which can break exact symmetry. Any production dynamic sparsification must either (a) threshold the symmetric pair `(i,j)` and `(j,i)` identically, or (b) symmetrize after dropping.
- **Wide-row silent failure.** SpGEMM kernels `return` when `na > MAX_LEFT_BLOCKS`. A zero-initialized output is a *wrong answer*, not a safe fallback. Validate all participating row degrees against compiled capacity before launch; fail loudly on overflow.

## 31.5 Approximation controls and acceptance criteria (must be defined before any approximate DM work)

Controls to tune (one at a time, with a correct reference preserved):
- TC2 iteration tolerance.
- Newton–Schulz tolerance.
- Mask cutoff radius / block drop threshold.
- Periodic mask rebuilding frequency.
- Convergence check frequency (every 1/2/4 iterations).
- Approximate spectral bounds (`gershgorin` vs power-iteration `[λ_min, λ_max]`).
- Reduced subspace dimension for any Ritz/CheFSI hybrid.

Acceptance criteria (measured, not assumed):
- Energy error vs dense reference.
- Per-atom charge error.
- Force error where available.
- HOMO/LUMO or occupied-subspace error.
- `Tr(KS) − N_occ`.
- `R_I = ||KSK−K||`.
- `R_H = ||HKS−SKH||_F`.
- Symmetry preservation `||K−K^T||_F`.
- Stability across geometries and system sizes (gapped molecules, narrow-gap ribbons, near-metallic ribbons).
- Wall-clock and device-memory savings.

**Do not report an optimization as successful without numerical verification AND a measured performance result on NVIDIA.**

## 31.6 Strategic decision — what the next implementation should be

The audit converges on the same conclusion Astra reached in section 30, now backed by the algorithmic comparison:

**The largest payoff is fixing the sparse architecture first, not replacing the algorithm.**

Rationale:
- TC2 purification is already implemented and physically sound for gapped systems. The NMP alternatives (FOE, GF, OMM, CheFSI, nested dissection) are all proof-of-concept with dense fallbacks and no full nonorthogonal DM; porting any of them is a larger project than fixing the existing TC2 wrapper.
- The current wrapper does ~5 SpGEMMs/iter with host round-trips between each. A persistent device-resident workspace with exactly 2 SpGEMMs/iter is a theoretical ~2.5× speedup *before* counting transfer elimination — likely much more in practice on GTX 1650 with 4 GB.
- `build_full_mask` must be replaced with `build_geometric_mask` + `build_product_mask` (validated for closure) as a separate, parity-tested change. This is where the physical sparsity savings live.
- Degree-specialized kernels and symbolic SpGEMM plans are P1 *after* residency, because profiling a host-bound wrapper would misattribute the bottleneck.
- CheFSI / Davidson / nested-dissection are **not** the next step for the *full DM* route. They become attractive only if the user's actual target is *frontier orbitals only* (HOMO/LUMO) rather than the full occupied density — that requirement should be confirmed before investing in them.

**Recommended next implementation (single change, parity-tested):**

1. Introduce `GpuBsrStructure` (row_ptr, col_idx, diag/trans block maps on device, uploaded once) and `GpuBsrMatrix` (structure + values buffer).
2. Build a persistent `SparsePurifyWorkspace` with `S, H0, H, Z0, Z1, K0, K1, T, Q, scratch, trace, residual` all device-resident.
3. Cache kernel handles; never rebuild `Kernel::builder()` in the iteration loop.
4. Rewrite `tc2_purify` to exactly: `T=K*S; Q=T*K; trace(T); Knew=2K−Q (or Q if trace>N_occ)`. Two SpGEMMs, zero host transfers, one scalar read for the trace/branch.
5. Validate against the existing CPU reference with `R_I`, `R_H`, `Tr(KS)`, energy, charges. Preserve current masks/numerics for this first change.
6. Then, as a separate change, swap `build_full_mask` → geometric+product masks and re-validate.

This is a coherent workspace/API change, not a quick partial wrapper rewrite. It should be the next use of expensive reasoning. The degree-specialized kernels, symbolic plans, TEAM=16/32 benchmarking, fused KSK, and padded-BSR4 H-support all follow *after* residency is established and profiled.



---

# USER

OK we implemented some optimization here, and also consider some algorithmic ideas and optimization from different repo, can you review it yet again with this updated infor, think it critically, and suggest what to do noxt what make sense and what less, what should be the priories, and eventally also estimate possible spead-up

https://github.com/ProkopHapala/dftbplus/commit/fcd78958a5acd7e33df063693beb18f9c0e5cd79
https://github.com/ProkopHapala/NumericalMathPlayground/tree/main/topics/LinearAlgebra/FastDirectSolvers
https://github.com/ProkopHapala/NumericalMathPlayground/tree/main/topics/LinearScalingQM


---


# GPT 5.6 sol

Yes. With the new commit and the other repository in view, I would change the priorities somewhat. The Jacobi work was worthwhile, but I would **not continue polishing Jacobi blindly** before fixing two larger architectural issues and measuring the real SCC sweep behavior.

The most interesting new algorithmic idea I see is actually a simplification of the **large sparse density-matrix formulation**: work with the non-symmetric occupation projector

$$
P = KS
$$

rather than the symmetric density kernel \(K\). That potentially changes generalized TC2 from **two sparse matrix products per purification step to one**, and Mulliken populations become simply the diagonal of \(P\). I discuss this below because I think it deserves a prototype.

## 1. What the latest commit actually achieved

The pair-block Jacobi transformation was a good change. The controlled event benchmark gives roughly:

* N=28, batch=100: **4.261 → 2.647 ms = 1.61×**
* N=32, batch=100: **1.90×**
* N=64, batch=100: **1.69×**

at equal numerical settings, and the extra normalization refinement fixed a genuine f32 orthogonality problem: N=64 reconstruction dropped from \(4.47\times10^{-4}\) to \(4.01\times10^{-5}\).

So I would accept this implementation.

But compared with `b3073ef`, this commit changes only the Jacobi implementation/tests on the compute side. The sparse runtime, SCC dataflow, kernel caching, assembly, etc. remain essentially unchanged. That matters when deciding what to optimize next.

There is also some documentation drift now: `gpu_eigen.cl`/`.rs` still describe “one barrier per round,” whereas the new path actually has two.

---

# 2. Before anything else: fix the Jacobi stopping semantics

I think there is a potentially **very cheap large speedup still hiding here**.

You currently have:

```c
if (fabs(apq) < 1e-7)
    rotation = identity;
```

but the whole-sweep convergence condition is

$$
\frac{\|A_{\rm off}\|_F}{\|A_{\rm off}^{(0)}\|_F}<10^{-7}.
$$

These two tolerances are inconsistent.

Suppose N=64. There are about 4000 off-diagonal entries. If each remaining element is individually below \(10^{-7}\), their Frobenius norm can still be of order

$$
\sqrt{4000}\,10^{-7}\approx6\times10^{-6}.
$$

And this is almost exactly what your diagnostic sees:

$$
\|A_{\rm off}\|_F =3.95\times10^{-6}
$$

after 8 sweeps — and it **does not improve at all between sweep 8 and sweep 20**.

So the kernel can spend the last 12 sweeps doing essentially nothing because every remaining pair is below the rotation threshold, while the global convergence condition can never be satisfied.

### This should be the next Jacobi experiment

Add outputs:

```text
sweeps_used[system]
final_offnorm[system]
status[system]
```

and benchmark real H′ matrices captured from SCC iterations.

Then separate:

```text
PAIR_SKIP_TOL
GLOBAL_REL_TOL
GLOBAL_ABS_TOL
```

I would initially try something like:

$$
\mathrm{converged} =
\|A_{\rm off}\|
<
\max(
\epsilon_{\rm rel}\|A_{\rm off}^{0}\|,
\epsilon_{\rm abs})
$$

and make `PAIR_SKIP_TOL` smaller than the requested global floor.

Alternatively, explicitly detect stagnation caused by the pair threshold and return `converged_to_pair_floor`.

**Potential:** if the real SCC matrices behave like the N=64 diagnostic and become stationary after ~8 sweeps rather than 20, this alone is about:

$$
20/8=2.5\times
$$

on Jacobi, **without changing the result at all**.

That is much more attractive than another heroic kernel rewrite.

---

# 3. Then optimize identity rotations

Right now when a pair is below threshold you set

$$
c=1,\quad s=0,
$$

but the pair-block transformation still processes all \(JPAIR^2\) blocks.

Late in convergence this means lots of operations like:

```text
1*x + 0*y
```

Use an `active[pair]` flag.

For block \((a,b)\):

* both inactive → do nothing
* only `a` active → left rotation only
* only `b` active → right rotation only
* both active → full \(R_a^TBR_b\)

This becomes particularly valuable together with a descending Jacobi threshold.

A classical thresholded-Jacobi strategy would make sense:

$$
t_{\rm pair}^{(s)}
\sim
\frac{\|A_{\rm off}^{(s)}\|}{N}
$$

early, then reduce the threshold as the sweeps progress.

I would expect another **1.2–1.5× Jacobi improvement** if late sweeps currently contain many identity rotations.

---

# 4. Another good Jacobi micro-optimization: exploit symmetry

The new kernel processes all ordered pair blocks:

$$
JPAIR^2.
$$

For N=28:

$$
14^2=196.
$$

But A is symmetric. You really need only

$$
\frac{14(15)}2=105
$$

unique blocks.

Calculate \(B'_{ab}\) once and explicitly write

$$
B'_{ba}=B_{ab}^{\prime T}.
$$

That almost halves the A-transform arithmetic.

You deliberately avoided this initially so numerical asymmetry could remain visible; that was sensible for validation. Now that the independent implementation is verified, I would benchmark the symmetric version.

It may even improve numerical stability because symmetry remains exact by construction.

I would estimate roughly **1.2–1.4× further Jacobi speedup**, not 2×, because V rotation, barriers and rotation generation remain.

---

# 5. Don't forget the simple memory-load problem

The matrix loading code still effectively has:

```c
for(i = lid; i < JN; i += WG)
    for(j=0; j<JN; j++)
        ...
```

For N=28/WG128, only 28 threads load the entire matrix.

Better:

```c
for(idx=lid; idx<JN*JN; idx+=WG) {
    i=idx/JN;
    j=idx%JN;
    ...
}
```

Same for initializing V.

This gives coalesced cooperative loading and uses the whole WG.

Probably only **5–15%**, but extremely easy.

---

# 6. Do not port the FastDirectSolvers Jacobi kernel

The old `FastDirectSolvers` block Jacobi is useful historically, but your current DFTB kernel is now considerably more sophisticated.

The FastDirectSolvers implementation itself notes that its local block solver has size/padding problems and that persistent buffers and size-sorted batches are needed.

So:

> **Keep the ideas, not that kernel.**

Useful ideas from that project are:

* spatial reordering,
* nested dissection,
* size bucketing,
* persistent device buffers,
* Schur complements/domain decomposition.

Its exact AMLS eigensolver, however, is **not currently an attractive replacement for DFTB SCC**.

For the tested matrices, exact AMLS is actually slower than LAPACK through N≈1700:

|    N | dense eigh |   AMLS |
| ---: | ---------: | -----: |
|  492 |      17 ms |  20 ms |
|  810 |      41 ms |  68 ms |
| 1242 |     123 ms | 197 ms |
| 1674 |     225 ms | 452 ms |

And more fundamentally: SCC needs the **whole occupied subspace/density**, not just a handful of lowest vibrational modes. If AMLS retains all relevant modes, most of the dimensional reduction disappears.

So I would not invest in AMLS for the main SCC path now.

---

# 7. Nested dissection *is* useful — but mainly as an ordering

This is where `FastDirectSolvers` becomes interesting for DFTB.

Your previous sparse-Cholesky ribbon result had roughly

$$
nnz(L)\sim N^2
$$

even though H/S themselves are spatially local.

That is very suspicious for a width-4 quasi-1D ribbon.

The current Python Cholesky explicitly uses:

```python
permc_spec='NATURAL'
```

rather than a fill-reducing symmetric permutation.

For a long narrow ribbon, after ordering atoms along its long axis, S should essentially be a **block-banded matrix with roughly fixed bandwidth**.

Then block Cholesky has approximately:

$$
O(N w^2)
$$

work for constant ribbon width \(w\), i.e. close to linear.

So I would absolutely test:

1. geometric ordering along ribbon length,
2. reverse Cuthill–McKee,
3. simple recursive coordinate bisection,
4. nested dissection.

But apply these first as **permutations for S/H**, not as a replacement eigensolver.

This could transform the Cholesky/CheFSI frontier-orbital route dramatically.

---

# 8. The assembly failure is still P0

The newest commit explicitly says the real formic-dimer GPU H/S pipeline still dies in `assemble_pairs` with `CL_OUT_OF_RESOURCES`, and the replica>=128 OOB remains.

That blocks a trustworthy end-to-end benchmark.

And I think `assemble_pairs` itself should now be specialized.

You already launch separate species-pair buckets, and `block_type` is uniform per launch, but the kernel still contains runtime branches for:

```text
1×1
1×4
4×4
```

and statically allocates worst-case local storage.

I would produce three specialized kernels:

```text
assemble_pairs_1x1
assemble_pairs_1x4
assemble_pairs_4x4
```

or compile-time specialize `BLOCK_TYPE`.

Also replace:

```c
__local Fragment l_frags[128];
```

with direct global fragment metadata, or homogeneous arithmetic offsets.

And replace the statically worst-case SK cache:

```c
l_sk_h[512*5]
l_sk_s[512*5]
```

with actual-size local arguments if practical.

This should:

* fix the replica bug,
* reduce local memory,
* reduce register pressure,
* remove runtime branching,
* possibly fix the NVIDIA resource error.

`GpuDriver` currently uses WG=64 for this kernel.

Query:

```text
CL_KERNEL_WORK_GROUP_SIZE
CL_KERNEL_LOCAL_MEM_SIZE
CL_KERNEL_PRIVATE_MEM_SIZE
```

for each specialized variant.

I would do this **before any new major dense algorithm** because currently your actual H-bond application cannot run through the complete NVIDIA path.

---

# 9. Dense SCC: the next important optimization is not another GEMM kernel

Your current SCC does approximately:

$$
H
\rightarrow
XHX
\rightarrow
\text{Jacobi}
\rightarrow
C
\rightarrow
D
\rightarrow
q.
$$

Several of those matrices are unnecessary.

The nice DFTB identity is

$$
H =
H^0+\frac12(SV+VS),
$$

with \(V\) diagonal in the AO basis.

Let

$$
X=S^{-1/2},\qquad Y=S^{1/2}.
$$

Then

$$
H'
=
XHX
=
H_0'
+
\frac12(XVY+YVX).
$$

Now exploit the fact that

$$
V=\sum_A V_A P_A,
$$

where \(P_A\) selects orbitals belonging to atom A.

Define once per geometry

$$
\boxed{
M_A=
\frac12
\left(
X P_A Y+
Y P_A X
\right)
}
$$

and

$$
H_0'=XH^0X.
$$

Then every SCC iteration becomes simply

$$
\boxed{
H'(V)=H_0'+\sum_A V_A M_A.
}
$$

For your formic dimer:

* N=28,
* \(N_A=10\).

Per element this is ~10 FMAs.

That is roughly

$$
N_A N^2 \approx 7840
$$

FMAs/system instead of order

$$
2N^3\approx43900
$$

for the current two Löwdin GEMMs.

And you eliminate the physical `H_scc` matrix entirely.

Memory cost is not problematic for these small systems:

$$
10\times28^2\times4\approx31\text{ kB/system}.
$$

Even batch100 is only ~3 MB.

I actually prefer this over the previously suggested “one weighted GEMM”.

**Likely benefit to whole SCC:** perhaps **5–10% currently**, but it also simplifies the pipeline and becomes more valuable after Jacobi gets faster.

---

# 10. Also remove D from the SCC iterations

You only need D when you finally want forces or full density output.

With orthogonal eigenvectors U:

$$
C=XU.
$$

And:

$$
SC=SXU=YU.
$$

Therefore the Mulliken population is directly

$$
\boxed{
q_A=
2\sum_{\mu\in A}
\sum_{k\in occ}
(XU)_{\mu k}(YU)_{\mu k}.
}
$$

So use only occupied columns:

```text
Cocc  = X * Uocc
SCocc = Y * Uocc

q = rowwise dot(Cocc, SCocc), then sum per atom
```

No:

```text
full C[N,N]
D[N,N]
density build
D*S Mulliken
```

during SCC.

Construct D only after convergence if forces require it.

This probably saves another **5–10% total** on the existing profile, but again becomes relatively more important once Jacobi drops.

---

# 11. A very interesting experiment: eigensolver warm-start

Your SCC charge mixer is warm-started, but your **eigensolver is not**.

Every SCC iteration takes the new H′ and Jacobi starts from the AO-orthogonal basis again:

$$
V=I.
$$

But consecutive SCC Hamiltonians differ only moderately.

Suppose \(U_{\rm old}\) diagonalized the previous H′. Transform the new one:

$$
B=U_{\rm old}^T H'_{\rm new}U_{\rm old}.
$$

B should already be nearly diagonal.

Then Jacobi may need perhaps 1–3 sweeps instead of 5–10+.

Cost: two cheap small-matrix GEMMs.

This is exactly the sort of trade that may win on a GPU because GEMM is cheap while Jacobi barriers are expensive.

Don't implement it yet. First obtain:

```text
sweeps_per_SCC_iteration
```

from real formic-dimer/H2O runs.

If ordinary Jacobi takes:

```text
2–3 sweeps
```

forget this idea.

If it takes:

```text
8–20 sweeps
```

this may be one of the largest remaining dense improvements.

Potential eigensolver speedup in the latter case: **2–4×**.

---

# 12. Now the sparse solver: residency is still overwhelmingly first

Nothing in `fcd789...` changes the fact that sparse matrix operations repeatedly bounce through host `Bsr4Matrix` objects.

For example `matmul_masked_bsym()` still:

1. creates GPU row/col buffers,
2. uploads both matrices,
3. allocates output,
4. launches,
5. finishes,
6. reads the entire result back,
7. reconstructs a host matrix.

And TC2 still recomputes products redundantly.

So my sparse priority is unchanged:

```text
GpuBsrStructure        immutable
    row_ptr
    col_idx
    diag map
    transpose map
    product plan

GpuBsrValues
    device buffer only

SparseWorkspace
    S,H,Z,K,T,Q,...
```

Then:

```text
T = K*S
Q = T*K
trace(T)
TC2(K,Q)
swap(K,Knew)
```

No readback.

### Estimated improvement

Even just the arithmetic count goes from roughly:

$$
5\ {\rm SpGEMM/iter}
\rightarrow
2.
$$

That's 2.5×.

Removing full-matrix PCIe transfers, allocations, `finish()`, structure reuploads and kernel reconstruction should make the real improvement larger.

I would consider **3–10× over the current prototype entirely plausible**.

That estimate is much less speculative than estimating some future AMLS gain.

---

# 13. A new sparse formulation I think you should test: purify \(P=KS\), not K

This is the most interesting mathematical simplification I found while comparing the approaches.

Currently your symmetric density kernel K satisfies

$$
KSK=K.
$$

Define instead

$$
\boxed{P=KS}.
$$

Then immediately:

$$
P^2
=
KSKS
=
KS
=
P.
$$

So **P is an ordinary idempotent projector**.

Also:

$$
\boxed{\operatorname{Tr}(P)=N_{\rm occ}}.
$$

And most importantly for DFTB Mulliken populations:

$$
\boxed{
q_A=
2\sum_{\mu\in A}P_{\mu\mu}.
}
$$

No KS multiplication is needed to obtain charges — P already *is* KS.

### How do we obtain P?

The generalized eigenproblem is

$$
HC=SC\epsilon.
$$

Let

$$
Z=S^{-1}.
$$

Then

$$
A=ZH
$$

has the same eigenvalues and right eigenvectors C:

$$
AC=C\epsilon.
$$

Because

$$
C^{-1}=C^TS,
$$

the occupied spectral projector of A is exactly

$$
f(A)=C_{\rm occ}C_{\rm occ}^T S=P.
$$

Therefore you can run **ordinary SP2/TC2 purification on A**:

$$
P_0=f_0(A)
$$

then each iteration only requires

$$
P_2=P^2
$$

and

$$
P_{\rm new}=
\begin{cases}
P^2 & \operatorname{Tr}(P)>N_{\rm occ}\\
2P-P^2 & \operatorname{Tr}(P)<N_{\rm occ}.
\end{cases}
$$

### That is one SpGEMM per purification step

instead of:

```text
T=K*S
Q=T*K
```

two products.

So after sparse residency:

> metric K-TC2: **2 SpGEMM/step**
> projector P-SP2: **1 SpGEMM/step**

Potentially another **~1.5–2×**.

### The catch

A is non-symmetric.

That is mathematically okay — it is similar to the symmetric orthogonalized Hamiltonian and has a real spectrum — but sparse truncation/f32 errors can make a non-normal representation more numerically delicate.

And this formulation relies more strongly on \(Z\approx S^{-1}\) being accurate. Your current K-TC2 only uses Z to construct the initial guess; the exact S participates in every purification step.

So I would first prototype P-SP2 on CPU/dense matrices and compare every iteration against the current metric K-TC2.

Diagnostics should include:

$$
\|P^2-P\|,
$$

$$
|\operatorname{Tr}P-N_{\rm occ}|,
$$

$$
\|AP-PA\|,
$$

and the metric self-adjoint condition

$$
\|SP-P^TS\|.
$$

Finally reconstruct

$$
K=PZ
$$

and compare to dense DFTB.

This is a **research P1**, but the possible payoff is large enough that I would definitely test it.

---

# 14. It generalizes beautifully to FOE

This is where your `LinearScalingQM/DensityMatrix` work becomes directly relevant.

That repository already explores Fermi operator expansions and Green's functions as alternatives to diagonalization.

Instead of zero-temperature SP2:

$$
P=\theta(\mu I-A),
$$

at finite electronic temperature use:

$$
P=f_\beta(A-\mu I)
$$

via Chebyshev FOE.

Again:

$$
q_A=2\,\mathrm{diag}(P).
$$

This gives a nice future solver split:

| Electronic structure                   | Best candidate   |
| -------------------------------------- | ---------------- |
| decent gap, T≈0                        | **SP2/TC2 on P** |
| small gap / metallic + smearing        | **FOE on P**     |
| need frontier orbitals                 | **CheFSI**       |
| strongly localized insulator, research | OMM              |

I would **not replace TC2 with FOE for ordinary molecules**. TC2's nonlinear purification is usually much cheaper for a clear gap.

But graphene / tiny-gap systems are exactly where fixed-range density truncation and zero-T purification become difficult; FOE with physical smearing may be much more robust.

---

# 15. CheFSI: useful, but for the right job

Your separate CheFSI implementation is specifically designed for a **small frontier subspace**, with GPU ELLPACK SpMM + Gram/subspace operations.

That is a good candidate to replace the current Python/SciPy frontier eigensolver **after SCC has converged**.

It is *not* a good replacement for the SCC density solver when half the spectrum is occupied.

If:

$$
N_{\rm occ}\sim N/2,
$$

CheFSI's “small subspace” advantage disappears.

So:

> TC2/FOE → SCC density
> CheFSI → final HOMO/LUMO/wavefunctions.

Also, don't port the current `CheFSI.cl` blindly. Its SpMM kernel keeps

```c
float sums[32];
```

per thread, which can become register-heavy.

For DFTB I would use atom-block storage and tile the vector dimension.

---

# 16. ELLPACK gives one useful sparse-layout idea

The CheFSI code uses ELLPACK because fixed neighbor counts are GPU-friendly.

Your BSR4 CSR is better for 4×4 atomic blocks, but the same principle suggests:

### Block-ELL / degree-bucket layout

For regular graphene/ribbons:

```text
col[row][slot]
block[row][slot][16]
```

with fixed or bucketed slots per row.

This eliminates row-pointer chasing and gives very predictable memory access.

However, I would **not rewrite BSR4 to ELL now**.

A better immediate optimization is what your OMM project independently calls a precomputed `Atom_Inter_Map`: precompute all hot-loop interaction offsets once so numerical kernels perform only gather/FMA and no searching.

That is essentially the symbolic SpGEMM plan we discussed:

```text
output block Cij:
    [(offset Aik, offset Bkj),
     (offset Aik, offset Bkj),
     ...]
```

Once masks are fixed, there is no reason to run two-pointer intersections 40–50 times during purification.

Expected SpGEMM benefit: roughly **1.2–2×**, depending on how traversal-bound the current kernel is.

Do this **after residency**.

---

# 17. OMM is interesting scientifically, but I would not prioritize it

OMM has conceptual appeal for large localized insulating systems:

$$
\text{optimize localized occupied orbitals directly}.
$$

But your own experiments already show the difficult part: maintaining orthogonality and localization becomes harder as the number of orbitals grows. The penalty method's overlap errors grow with system size and the method needs correction/projection machinery.

There are also support recentering and support-boundary issues.

So OMM is a legitimate **alternative research branch**, but not currently an optimization of the DFTB solver.

I would first get a robust sparse density-projector solver.

---

# 18. Green's functions + nested dissection are a longer-term metallic-system route

This is where `FastDirectSolvers` and `DensityMatrix/GF` could eventually meet.

For metals/small-gap 2D systems, density-matrix locality deteriorates. Then a fixed-range \(O(N)\) TC2 representation may stop giving fixed accuracy.

A more robust route is:

$$
G(z)=(zS-H)^{-1}
$$

at several complex contour points, with density obtained by contour integration.

Nested dissection gives a good sparse factorization ordering for each shifted matrix.

For:

* quasi-1D ribbons: potentially near linear,
* 2D flakes: typically much better than dense cubic, though not true O(N),
* 3D: more expensive.

This is much more interesting than using AMLS as an eigensolver.

But implementation complexity is high, so I would keep it as a **fallback research direction if TC2/FOE locality fails**.

---

# 19. Updated priorities

My current ordering would be:

| Priority          | Work                                                                    | Why                                          |
| ----------------- | ----------------------------------------------------------------------- | -------------------------------------------- |
| **P0-A**          | Fix `assemble_pairs` OOB + NVIDIA resource failure                      | blocks real end-to-end application           |
| **P0-B**          | Jacobi sweep/residual/status output; fix pair/global tolerance mismatch | may remove many completely useless sweeps    |
| **P0-C**          | Real event-profiled full SCC using captured molecular H′ matrices       | tells us what actually dominates now         |
| **P0-D**          | Sparse persistent GPU workspace, **5→2 SpGEMM**, zero transfers         | biggest obvious sparse gain                  |
| **P1-A**          | GPU eigenvalue ranking/occupation + persistent dense kernels/workspace  | removes synchronization / allocation         |
| **P1-B**          | \(H'=H'_0+\sum_A V_A M_A\)                                              | removes SCC H construction + Löwdin GEMMs    |
| **P1-C**          | direct occupied Mulliken, no D during SCC                               | less memory/work                             |
| **P1-D research** | prototype \(P=KS\) SP2                                                  | potentially **2→1 sparse product/step**      |
| **P1-E**          | symbolic SpGEMM interaction plan + degree buckets                       | exploit fixed topology                       |
| **P2**            | eigensolver warm-start from previous U                                  | only if sweep telemetry says worthwhile      |
| **P2**            | GPU CheFSI for post-SCC HOMO/LUMO                                       | good reuse of other repo                     |
| **P2/P3**         | RCM/ND ordering for sparse Cholesky / GF                                | valuable special large-system route          |
| **P3**            | FOE fallback for small-gap/finite-T systems                             | alternative physics regime                   |
| **P3**            | OMM / exact AMLS                                                        | interesting research, not current bottleneck |

---

# 20. Rough speedup budget

These are deliberately ranges, because the old SCC phase table was not true device event profiling.

### Dense small-system path

Relative to the old `b307...` implementation:

| Change                                |                   Expected gain |
| ------------------------------------- | ------------------------------: |
| already implemented pair-block Jacobi |             **1.6–1.9× Jacobi** |
| fix useless 20-sweep tail             | potentially **1.5–2.5× Jacobi** |
| active/threshold rotations + symmetry |  additional **1.2–1.5× Jacobi** |
| persistent kernels + GPU occupation   |              **1.1–1.3× total** |
| \(H'_0+\sum V_AM_A\)                  |            **1.05–1.15× total** |
| no D during SCC                       |             **1.05–1.1× total** |

I would consider an **overall 2–3× improvement over the original dense SCC benchmark plausible** without changing the physical algorithm.

If the old reported 0.84 ms/system at batch100 were from a comparable device/run, a reasonable engineering target would therefore be around:

$$
\boxed{0.3-0.5\ {\rm ms/system}}
$$

for the same problem.

Don't treat that as a benchmark prediction until the real NVIDIA full pipeline is running.

### Sparse path

From the current prototype:

```text
host-resident + 5 SpGEMM
        ↓
device-resident + 2 SpGEMM
```

I'd expect **3–10×** wall-clock improvement.

Then, if the \(P=KS\) formulation works:

```text
2 SpGEMM / purification step
        ↓
1 SpGEMM / step
```

another **~1.5–2×**.

Symbolic interaction maps / degree-specialized kernels could plausibly add another **1.2–2×**.

So versus the present sparse prototype, a final improvement of **5–20× is not implausible**. The large uncertainty is because you still do not have a clean GPU-resident sparse timing baseline.

---

The one thing I would **not** do next is launch a large AMLS/OMM/Green-function implementation. There are still much cheaper improvements in the two algorithms you already have, and the \(P=KS\) projector idea in particular may simplify the large-system solver enough that some of those alternatives become unnecessary.

---

## §32 Implementation: GPU-Resident Sparse Workspace + Jacobi Stopping Fix (2025-09-22)

### What was implemented

Two of the highest-priority changes from the plan above are now implemented and tested on the NVIDIA GTX 1650 (platform 1).

#### P0-D: Device-resident sparse purification (`gpu_sparse.rs`)

**New types** (in `rust_dftb/src/methods/sparse/gpu_sparse.rs`):

- `GpuBsrStructure` — immutable device CSR: `row_ptr`, `col_idx`, `diag_block`, `transpose_block`. Uploaded once, shared via `Arc`.
- `GpuBsrMatrix` — `Arc<GpuBsrStructure>` + device `Buffer<f32>` values. The structure is shared; only values differ.
- `SparsePurifyWorkspace` — persistent workspace holding `s`, `k`, `knew`, `t`, `q` as `GpuBsrMatrix`, plus a 1-float `trace_buf` and reduction scratch. All allocated once at construction.

**Cached kernel handles** — `SparseBsr4Gpu` now builds all 11 kernels (`bsr4_spgemm_masked`, `bsr4_spgemm_masked_Bsym`, `bsr4_zero`, `bsr4_axpby`, `bsr4_mcweeny`, `bsr4_tc2`, `bsr4_symmetrize`, `bsr4_mulliken_KS`, `bsr4_trace_KS_partial`, `reduce_sum_f32`, `bsr4_idempotency_partial`) once at `new()` time using dummy buffers. Per-launch calls use `Kernel::set_arg` + `Kernel::cmd().global_work_size(...).enq()` — no `Kernel::builder()`, no `Buffer::builder()`, no `finish()` between kernels.

**Device-resident methods** on `SparseBsr4Gpu`:

| Method | Host transfers | Allocations | `finish()` calls |
|---|---|---|---|
| `spgemm_bsym_dev` | 0 | 0 | 0 |
| `spgemm_masked_dev` | 0 | 0 | 0 |
| `axpby_dev` | 0 | 0 | 0 |
| `zero_dev` | 0 | 0 | 0 |
| `tc2_dev` | 0 | 0 | 0 |
| `symmetrize_dev` | 0 | 0 | 0 |
| `trace_ks_dev` | 1 scalar (f32) | 1 scratch buffer | before scalar read |
| `idempotency_err_dev` | 1 scalar (f32) | 1 scratch buffer | before scalar read |
| `mulliken_dev` | `n_atom` floats | 1 output buffer | before read |

**`SparsePurifyWorkspace::tc2_step_dev`** — one TC2 iteration:

```text
T = K·S          (spgemm_bsym_dev — 0 transfers)
Q = T·K          (spgemm_bsym_dev — 0 transfers)
n = Tr(T)         (trace_ks_dev — 1 scalar read)
Knew = TC2(K,Q,n) (tc2_dev — 0 transfers)
symmetrize(Knew)  (symmetrize_dev — 0 transfers)
swap(K, Knew)     (std::mem::swap on Buffer handles)
```

**Exactly 2 SpGEMMs, 0 matrix host transfers, 1 scalar read per iteration.** This replaces the old 5-SpGEMM flow that allocated/uploaded/read back full matrices for every operation.

**`tc2_purify_dev`** — full loop with convergence checking every `check_every` iterations, divergence detection (10× best R_I), and best-iteration tracking.

#### P0-B: Jacobi stopping semantics (`gpu_eigen.cl`)

**Problem:** `PAIR_SKIP_TOL` was hardcoded equal to `JACOBI_TOL` (both 1e-7). When every remaining pair is below 1e-7, the Frobenius norm of ~4000 elements is ~6e-6, which never satisfies the global relative tolerance `off_cur/off0 < 1e-7`. The kernel then runs all 20 sweeps doing nothing.

**Fix:**

1. **Separated `PAIR_SKIP_TOL` from `JACOBI_TOL`.** New default `PAIR_SKIP_TOL = 1e-12` — small enough that the global Frobenius norm can reach `JACOBI_TOL` (worst case: `sqrt(64²) * 1e-12 = 6.4e-11 << 1e-7 * off0`), large enough to avoid denormal-float NaN issues that appeared with `PAIR_SKIP_TOL = 0`.

2. **Stagnation detection.** If the off-diagonal norm does not improve by 10% for 3 consecutive sweeps, break early. This requires multiple consecutive non-improving sweeps to avoid premature exit during transient plateaus.

3. **Updated documentation** in `gpu_eigen.rs` to reflect the actual barrier count (2 per round with block update, not 1) and the new stopping semantics.

### Test results (NVIDIA GTX 1650, platform 1)

**Sparse suite** — 15/15 passed:

```
test test_spgemm_masked_vs_dense ... ok
test test_spgemm_masked_bsym_vs_dense ... ok
test test_spgemm_mask_truncation ... ok
test test_trace_ks_vs_cpu ... ok
test test_symmetrize_vs_host ... ok
test test_mulliken_vs_cpu ... ok
test test_idempotency_exact_kernel ... ok
test test_mcweeny_convergence ... ok
test test_tc2_convergence ... ok
test test_boolean_product_mask ... ok
test test_newton_schulz_inverse ... ok
test test_k0_and_tc2_vs_dense_projector ... ok
test test_rh_distinguishes_projectors ... ok
test test_tc2_dev_resident_convergence ... ok   ← NEW
test test_tc2_dev_vs_host_parity ... ok          ← NEW
```

**Key new test: `test_tc2_dev_vs_host_parity`**

Runs 5 TC2 steps via the old host-roundtrip path and 5 via the new device-resident path, same K0.

```
TC2 host-vs-dev after 5 steps: max|K_host - K_dev| = 0e0
```

**Bitwise identical** — the device-resident path produces exactly the same result as the host-roundtrip path (same kernel, same arithmetic, just no intermediate host transfers).

**Key new test: `test_tc2_dev_resident_convergence`**

```
TC2-dev iter 0: R_I=6.115388e-2  Tr(KS)=2.400000  (Nocc=3)
TC2-dev iter 1: R_I=2.5439807e-3  Tr(KS)=2.880000  (Nocc=3)
TC2-dev iter 2: R_I=4.071165e-6  Tr(KS)=2.995200  (Nocc=3)
TC2-dev final: R_I=4.071165e-6  Tr(KS)=2.99520  iters=3
TC2-dev max|K_final - K_exact| = 1.2516975e-6
```

Converges in 3 iterations to `R_I = 4.07e-6`, `max|K - K_exact| = 1.25e-6`.

**Dense eigensolver suite** — 10/10 passed (with SK data):

```
test test_batched_jacobi ... ok   (10×H2O, max val diff = 3.97e-7, max vec diff = 9.84e-6)
test test_inv_sqrt_h2 ... ok
test test_inv_sqrt_h2o ... ok
test test_inv_sqrt_n2 ... ok
test test_jacobi_parity_ch4 ... ok
test test_jacobi_parity_h2 ... ok
test test_jacobi_parity_h2o ... ok
test test_jacobi_parity_n2 ... ok
test test_lambda_min_reporting ... ok
```

**SCC suite** — 3/3 passed:

```
test test_gpu_scc_parity_batched_h2o ... ok
test test_gpu_scc_parity_h2o ... ok
test test_gpu_scc_parity_n2 ... ok
```

### What remains

- **Event profiling** — the device-resident methods do not call `finish()` between kernels, which is correct for pipelining but means host-side timing is meaningless. OpenCL event profiling (`clGetEventProfilingInfo` with `CL_PROFILING_COMMAND_START`/`END`) should be added to measure actual kernel execution time vs. host enqueue overhead.
- **Dense assembly OOB** — `assemble_pairs` still indexes `l_frags[p.replica]` with `l_frags[128]`, causing corruption for batch > 127.
- **P=KS projector prototype** — now that the resident two-product baseline exists and is verified, the one-product P=KS formulation can be prototyped and compared.
- **Sparse masks** — the exposed sparse route still uses `build_full_mask`. Geometric/product masks should be added as a separate parity-tested change.

### Additional fixes (2025-09-22, continued)

#### Row-degree overflow — fail-loud fix

**Problem:** Both SpGEMM kernels (`bsr4_spgemm_masked`, `bsr4_spgemm_masked_Bsym`) silently `return` when a row's degree `na > MAX_LEFT_BLOCKS`, producing zero output for that row without any error. This violates the repo's "Fail Fast, Fix the Physics" rule.

**Fix:** Added `check_row_degree()` host-side validation in `SparseBsr4Gpu` that scans `row_ptr` before any SpGEMM launch and returns a descriptive `DftbError` if any row exceeds `MAX_LEFT_BLOCKS`. The check is called from:
- `spgemm_masked` (old host-roundtrip path)
- `spgemm_masked_bsym` (old host-roundtrip path)
- `GpuBsrStructure::new` (device-resident path — catches the error at structure upload time, before any kernel launch)

**Test:** `test_row_degree_overflow_fail_loud` — creates a 300-atom full mask (row degree 300 > 256) and verifies `GpuBsrStructure::new` fails with a message mentioning `MAX_LEFT_BLOCKS`.

#### Newton-Schulz device-resident rewrite

**Implemented:** `newton_schulz_inverse_dev` on `SparseBsr4Gpu` — device-resident version of the Newton-Schulz / Hotelling iteration for sparse approximate inverse. All matrices (Z, Znew, T, Q) stay on the GPU for the entire loop.

Per iteration:
- T = Z·S (spgemm_bsym_dev — 0 transfers)
- ||T||_F (frobenius_norm_dev — 1 scalar read)
- Tr(T) (trace_ks_dev — 1 scalar read)
- R_Z computed on host from the 2 scalars
- Q = T·Z (spgemm_bsym_dev — 0 transfers)
- Znew = 2·Z - Q (axpby_dev — 0 transfers)
- symmetrize(Znew) (symmetrize_dev — 0 transfers)
- swap(Z, Znew)

2 SpGEMMs, 2 scalar reads, 0 matrix host transfers per iteration.

**Test:** `test_newton_schulz_inverse_dev` — converges in 5 iterations to `R_Z = 0`, same as the old host-roundtrip path.

### Final test results (30 tests, all pass)

```
gpu_sparse_bsr4: 17 passed (15 original + 2 new TC2-dev + 1 row-degree + 1 NS-dev... actually 17 total)
gpu_eigenproblem: 10 passed
gpu_scc: 3 passed
```
