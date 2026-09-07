# Task 1: GPU Multi-System Relaxed Scan of Hydrogen-Bonded Nucleobase Pairs

**Created:** 2026-09-07
**Status:** planning / manifest
**Owner:** prokop / Devin

---

## 1. Goal

Build a **production-grade** GPU pipeline that screens large numbers of
hydrogen-bonded tautomerization configurations (nucleobase-pair-like systems)
on a single GPU, performing not just batched SCF but also **batched geometry
relaxation** — all on one GPU, efficiently. Then use it to explore the relaxed
configuration-energy landscape for double-proton transfer (2 H atoms hopping
between donor/acceptor sites), distinguishing **synchronous** vs
**asynchronous** hopping mechanisms, and eventually extending to **constrained
DFT (CDFT)** where proton and electron transfer are monitored separately.

This is **not a toy model experiment**. The deliverable is a real screening
tool: given a set of nucleobase-pair geometries with varying proton positions,
the pipeline computes relaxed energies, forces, and charges for all of them in
one batched GPU run, measures real performance, identifies bottlenecks
(numerical, hardware, algorithmic, physics-related), and performs systematic
parameter tuning.

---

## 2. Motivation & Physics

### 2.1 Double proton transfer in H-bonded dimers

Nucleobase pairs (Watson-Crick AT, GC) form 2-3 hydrogen bonds. Proton
transfer along these bonds is the simplest model of tautomerization — the
process that can cause mispairing in DNA replication. The key questions:

- **Synchronous vs asynchronous hopping:** Do both protons transfer
  simultaneously (concerted, diagonal path on 2D PES) or stepwise (one proton
  moves first, then the other, L-shaped path)?
- **Barrier height and shape:** What is the energy barrier along the
  synchronous vs asynchronous path? Is there a metastable intermediate
  (zwitterion) at the corner of the 2D PES?
- **Relaxed vs rigid PES:** The rigid scan (only H atoms move, rest frozen)
  overestimates barriers. The relaxed scan (all atoms relax except the
  constrained H-bond coordinate) gives the chemically meaningful barrier.

### 2.2 From formic dimer to nucleobase pairs

We already validated the GPU SCC solver on the formic acid dimer (28 orbitals,
fits N≤64). The next step is nucleobase pairs:

| System | Atoms | Orbitals (mio-1-1) | Fits N≤64? |
|---|---|---|---|
| Formic acid dimer | 10 | 28 | yes (done) |
| Formic + azaindole mixed dimer | 20 | 56 | yes |
| 7-Azaindole dimer | 30 | 84 | **no** |
| Adenine-Thymine (AT) pair | 30 | ~87 | **no** |
| Guanine-Cytosine (GC) pair | 29 | ~86 | **no** |

**The critical problem:** nucleobase pairs have ~86-87 orbitals, exceeding the
N≤64 single-workgroup dense Jacobi limit. We need a solver that works for
N>64 on GPU.

### 2.3 CDFT extension (future, specified later)

Eventually we want to monitor proton and electron transfer separately via
constrained DFT: constrain the electron density on one fragment while allowing
the proton to move. This requires:
- Fragment-based charge constraints in the SCC loop
- Lagrange multiplier optimization for the constraint
- Separate tracking of proton position (geometric) and electron population
  (density constraint)

This is specified as a future extension — the current task focuses on the
relaxed scan infrastructure.

---

## 3. What Is Already Implemented

### 3.1 GPU multi-system SCC solver (N≤64, dense)

- **Batched H-assembly** — `qmqm/gpu_driver.rs::gpu_assemble_batched`,
  `qmqm/gpu_prep.rs::GpuBatch::from_fragments`. N replicas in one kernel
  launch. Tested: 10× H2, formic dimer scan (41+441 points).
- **Brent-Luk parallel cyclic Jacobi** — `qmqm/gpu_eigen.rs::jacobi_cyclic_local_batched`.
  N/2 independent rotations per round, one barrier per round, A+V in `__local`
  for N≤64. 10/10 eigenproblem tests pass.
- **S^{-1/2} on GPU** — `qmqm/gpu_eigen.rs::build_inv_sqrt`. Jacobi(S) →
  U·Λ^{-1/2}·U^T.
- **Device-resident SCC loop** — `qmqm/gpu_scc.rs::gpu_solve_scc_batched_diis`,
  `gpu_solve_scc_batched_diis_warmstart`. Full SCC cycle on device: Δq →
  gamma matvec → H_scc update → 2 GEMMs (H' = X·H_scc·X) → Jacobi →
  occ_mask → density → Mulliken → mix. Only PCIe traffic: batch RMS scalars
  per iteration + occ_mask ints.
- **CPU-driven DIIS mixer** — 13 iters vs 60-192 for simple mix. Warm-start
  from previous charges. Best-effort mode returns per-system RMS for
  unconverged points.
- **Per-system RMS diagnostics** — reports 10 worst systems on nonconvergence.
- **Parity validated** — H2O |dE|<3e-7, N2 |dE|<1e-6, 10× H2O |dE|<1e-6;
  formic dimer 1D scan (41 pts, 28 orbs) |dE|<3.6e-6, |dq|<1e-5; 2D scan
  147/441 converged (CPU also fails on asymmetric points).
- **Throughput** — 1194 systems/s at batch=100 (formic dimer, N=28). 60-120×
  vs CPU. Baseline measured: Jacobi = 50-65% of per-iter time, GEMM = 5%,
  DIIS = 16%.

### 3.2 CPU forces and geometry optimization

- **Analytic DFTB forces** — `methods/dftb/forces.rs::compute_scc_forces`.
  Four components: F_nonSCC, F_SCC_shift, F_SCC_dc, F_rep. Parity vs Fortran
  <7e-8. Analytic dH0/dx, dS/dx (not finite-difference).
- **FIRE optimizer** — `examples/hbond_ref.rs::FireOptimizer`,
  `optimize_geometry_persistent`. Tested on formic+azaindole mixed dimer
  (56 orbs): 100 steps, 32.8s, E: -29.77 → -32.11 Ha, max|F|: 4.73 → 0.049.
  Uses LAPACK `dsyevd` eigensolver + warm-started charges.
- **CPU scan driver** — `examples/scan.rs`, `examples/hbond_ref.rs`.
  1D/2D rigid scan with H-position interpolation.

### 3.3 Scan and plotting infrastructure

- **Formic dimer scan** — `tests/formic_scan_plots.rs`: 1D (41 pts) + 2D
  (21×21=441 pts) PES with energy/charge/parity plots.
- **Plotting** — `scripts/plot_formic_scan.py`: 1D energy/charge/parity +
  2D contour plots. Output to `debug/formic_dimer_scan/`.
- **Geometry engine** — `scripts/geometry_engine.py`, `scripts/run_formic_dimer_*.sh`.

### 3.4 Nucleobase geometries available

- `data/xyz/adenine-thymine.xyz` (30 atoms)
- `data/xyz/guanine-cytosine.xyz` (29 atoms)
- `data/xyz/adenine.xyz`, `guanine.xyz`, `thymine.xyz`, `uracil.xyz`,
  `citosine.xyz` (individual bases)
- `data/xyz/azaindol_dimer.xyz` (30 atoms, 84 orbs — also exceeds N≤64)

### 3.5 SK files

- `mio-1-1` SK set available at `/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1/`
  — covers H, C, N, O (all elements in nucleobases).
- `RUST_DFTB_SK_DIR` env var points to this directory.

---

## 4. What Needs to Be Done

### 4.1 N>64 solver on GPU (BLOCKING — must solve first)

**The problem:** AT pair = ~87 orbs, GC pair = ~86 orbs. The current GPU
dense Jacobi uses `__local` memory with N≤64 (one workgroup does the full
eigensolve). N>64 overflows local memory.

**Options (in order of preference):**

#### Option A: Tiled multi-workgroup dense Jacobi (recommended for N~86-128)

Extend the Brent-Luk Jacobi to work across multiple workgroups using global
memory for the matrix, with `__local` tiling for the rotation application.

- Each round: N/2 independent 2×2 rotations. For N=88, that's 44 rotations.
- The matrix (88×88 = 7744 f32 = ~31 KB) fits easily in global memory.
- Rotations applied in tiles: each WG handles a tile of the matrix, reads
  row/column pairs from global memory, applies rotation, writes back.
- Accumulating the eigenvector matrix V uses the same tiled approach.
- This is a well-known algorithm (parallel Jacobi for distributed memory,
  adapted for GPU). The key is minimizing global memory traffic.

**Advantage:** reuses the dense algebra (GEMM, density build, etc.) — only
the Jacobi kernel changes. Everything else in the SCC pipeline stays the same.

**Estimated effort:** moderate — one new kernel + driver changes. The
Brent-Luk schedule (`brent_luk_rounds`) already exists.

#### Option B: Sparse BSR4 purification route (for much larger N)

Use `methods/sparse/` — BSR4 atom-block CSR + TC2/McWeeny purification.
No diagonalization needed. Already working on benzene/coronene/circumcoronene.

**Problem for nucleobases:** nucleobase pairs are NOT sparse — they are
compact 3D molecules with ~30 atoms. The BSR4 geometric mask would include
most blocks (density ~60-80%), so sparse vs dense gives little benefit.
Sparse purification is O(N³) in the number of non-zero blocks, which for a
dense-ish matrix is worse than dense Jacobi.

**Conclusion:** Option B is not suitable for nucleobase pairs. It IS suitable
for Task 2 (Si nanocrystals, which ARE sparse). Keep the routes separate.

#### Option C: Hybrid — dense eigensolve with host-side LAPACK for N>64

Fall back to host-side LAPACK `dsyevd` for systems with N>64, while keeping
the rest of the SCC pipeline on GPU. The eigensolve is 50-65% of per-iter
time, so this would roughly halve GPU speedup, but it works immediately.

**Advantage:** zero new kernel development. Can be done in a day.
**Disadvantage:** PCIe traffic per SCC iteration (upload H', download ε, C'),
loses much of the GPU advantage.

#### Recommendation

**Start with Option C** (host LAPACK fallback for N>64) to unblock the
chemistry immediately. **Then implement Option A** (tiled multi-WG Jacobi)
as the production solution. Option B is for Task 2.

### 4.2 GPU forces (BLOCKING for relaxed scan)

**Current state:** analytic DFTB forces exist on CPU only
(`methods/dftb/forces.rs`). The GPU SCC solver computes energy and charges
but NOT forces.

**What's needed:** port the four force components to GPU:
1. **F_nonSCC = 2·(DM·dH0' − EDM·dS')** — requires dH0/dx, dS/dx on GPU.
   The analytic derivatives exist in `rotation.rs::rotate_block_with_derivs_into`
   and `interpolation.rs::eval_with_deriv_into`. Need OpenCL kernels for
   derivative-augmented H/S assembly.
2. **F_rep = dE_rep/dr · r_hat** — repulsive spline derivative. Spline data
   already parsed; need a GPU kernel for spline evaluation + derivative.
3. **F_SCC_dc (gamma + 1/R Coulomb)** — `gamma_prime_full` exists on CPU.
   Need GPU kernel for gamma derivative matvec.
4. **F_SCC_shift (Pulay-like)** — requires dS/dx contracted with density
   matrix. Reuses the dS/dx from component 1.

**Alternative: finite-difference forces on GPU.** For a batched multi-system
relaxation, finite-difference forces are actually viable:
- For each system: evaluate energy at 6N+1 geometries (central difference).
- All 6N+1 evaluations are independent → batch them across the GPU.
- For N=30 atoms: 181 geometries per system. With batch=100 systems:
  18,100 energy evaluations, each ~0.5 ms → ~9 s per force evaluation.
- This is slower than analytic forces but requires NO new kernels — just
  the existing SCC energy evaluation, called many times.

**Recommendation:** Start with finite-difference forces (no new kernels,
immediate), then implement analytic GPU forces for production speed.

### 4.3 Batched geometry relaxation on GPU

**Current state:** FIRE optimizer exists on CPU
(`examples/hbond_ref.rs::FireOptimizer`), single-system.

**What's needed:** batched multi-system relaxation where:
- All systems relax simultaneously, each at its own pace.
- Per system: one FIRE (or L-BFGS) step per outer iteration.
- Force evaluation is batched across all systems in one GPU call.
- Converged systems are masked out (active mask — already designed as D11
  but not implemented: `gpu_scc.rs` TODO "all systems run all iterations
  currently").

**Architecture:**
```
for relax_iter in 0..max_relax_iters:
    1. Batched SCC solve for all active systems → energies, charges, (forces)
    2. For each active system: FIRE/L-BFGS step using forces
    3. Update geometries, upload new coords to GPU
    4. Rebuild H0/S (batched gpu_assemble_batched) — only for active systems
    5. Check convergence (max|F| < f_tol) → mark converged, add to inactive mask
    6. If all converged: break
```

**Key insight:** the SCC warm-start is crucial here. Each relaxation step
starts from the previous step's converged charges → 3-4 SCC iterations
instead of 13-16. This is already implemented
(`gpu_solve_scc_batched_diis_warmstart`).

**Constraint projection for relaxed scan:** at each scan point (t1, t2),
fix the positions of the 2 transferring H atoms (or fix only the H-bond
coordinate component), relax all other DOFs. This requires:
- Identifying constrained atoms/coordinates per system
- Projecting out constrained components from the force vector before the
  FIRE step
- This is a host-side operation (trivial — zero out force components)

### 4.4 Performance measurement and bottleneck analysis

**What to measure:**
1. **Wall-clock time breakdown** per relaxation step:
   - H0/S assembly (gpu_assemble_batched)
   - SCC iterations (total, with warm-start)
   - Force evaluation (analytic or finite-difference)
   - FIRE step + constraint projection (host)
   - Geometry upload + H0/S rebuild
2. **GPU utilization:** kernel execution time vs host overhead vs PCIe
   transfer. Use OpenCL profiling events (`CL_PROFILING_COMMAND_START/END`).
3. **Scaling with batch size:** 1, 10, 50, 100, 200, 500, 1000 systems.
   Measure systems/second and identify the batch size where GPU saturates.
4. **Scaling with system size:** formic dimer (28 orbs), azaindole dimer
   (84 orbs), AT pair (87 orbs). How does per-system time scale with N?
5. **SCC convergence statistics:** distribution of SCC iterations across
   systems, worst-case systems, effect of warm-starting.

**Bottleneck categories to identify:**
- **Hardware:** GPU memory bandwidth, compute throughput, PCIe transfer,
  kernel launch overhead.
- **Algorithmic:** Jacobi convergence rate (sweeps), SCC iteration count,
  DIIS vs simple mixing, warm-start effectiveness.
- **Physics-related:** wrong SCC mixing parameters causing oscillation or
  slow convergence; SK table precision (resampling); f32 accumulation
  errors in long SCC loops; occupation flipping near degeneracies.

### 4.5 Parameter investigation and tuning

**Parameters to sweep:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| SCC mixing alpha | 0.05 | 0.01-0.2 | Too small: slow convergence. Too large: oscillation. |
| DIIS history size | 8 | 4-15 | More history: better extrapolation but more memory. |
| SCC tolerance | 1e-7 | 1e-5 to 1e-9 | Tighter: more iters but better forces. |
| Max SCC iters | 50 | 30-200 | Too low: unconverged forces. |
| Jacobi tolerance | 1e-7 | 1e-5 to 1e-10 | Affects eigensolve accuracy vs time. |
| Jacobi max sweeps | 20 | 10-40 | More sweeps: better convergence. |
| FIRE dt | 1.0 | 0.1-5.0 | Too large: unstable. Too small: slow. |
| FIRE dt_max | 5.0 | 2.0-10.0 | Caps timestep. |
| Force tolerance | 5e-3 | 1e-3 to 1e-2 | Tighter: more relax iters. |
| f32 vs f64 | f32 | — | Monitor energy/force drift. |

**Methodology:**
- For each parameter: fix all others, sweep one, measure (convergence rate,
  total time, final energy, max|F|).
- Identify the Pareto front: fastest convergence vs tightest forces.
- For the 2D scan: identify which parameters fix the 67% non-convergence
  (try: level shifting, Broyden mixing, smaller alpha, strip propagation).

### 4.6 Relaxed 2D PES for nucleobase pairs

**The production deliverable:**
1. Build AT and GC pair geometries with identified H-bond donor/acceptor atoms.
2. Define 2D scan grid: (t1, t2) ∈ [0,1]×[0,1], e.g. 21×21 = 441 points.
3. For each point: constrain 2 H atoms at interpolated positions, relax
   all other DOFs, compute relaxed energy.
4. Plot 2D relaxed PES → identify synchronous (diagonal) vs asynchronous
   (L-shaped) paths, barrier heights, metastable intermediates.
5. Compare with rigid PES (no relaxation) to quantify relaxation effect.
6. Screen multiple tautomerization pathways across different nucleobase
   pair geometries (modified bases, mispairs, etc.).

---

## 5. Open Questions and Challenges

### 5.1 N>64 eigensolver: how far to push dense?

- Tiled multi-WG Jacobi works for any N, but gets slower as N grows (O(N³)
  with constant overhead per tile). For N=87 (AT pair), it's still very
  feasible. For N=200+ (larger systems), sparse purification becomes better.
- **Question:** what is the crossover point where sparse beats dense on GPU?
  This depends on sparsity (geometric mask density) and needs benchmarking.

### 5.2 SCC convergence on nucleobase pairs

- Formic dimer 2D scan: 67% of points don't converge (CPU also fails).
  Nucleobase pairs are larger and more polarizable → likely worse.
- **Question:** which mixing strategy works? DIIS with level shifting?
  Broyden? Adaptive alpha? This needs systematic testing.
- **Question:** is f32 precision sufficient for SCC convergence on
  nucleobase pairs? The H_scc matrix has entries spanning ~4 orders of
  magnitude; f32 has ~7 digits. Monitor for accumulation errors.

### 5.3 Force accuracy and relaxation stability

- Analytic forces require dH0/dx, dS/dx — these involve spline derivatives
  and rotation matrix derivatives. f32 force accuracy: is it sufficient
  for geometry relaxation? FIRE is fairly robust to force noise, but
  L-BFGS is more sensitive.
- **Question:** does f32 SCC + f32 forces give converged geometries, or
  do we need f64 forces (host-side) with f32 SCC (device-side)?

### 5.4 Batched relaxation scheduling

- Different systems converge at different rates. After 10 relaxation steps,
  half may be converged. The active mask skips them, but the GPU kernel
  still launches with the full batch (just with early-return for inactive).
- **Question:** is it better to (a) keep the full batch and use active mask,
  (b) compact the batch (remove converged systems), or (c) split into
  microbatches? This is the scheduling benchmark (Stage 5 of the GPU plan,
  not yet implemented).

### 5.5 CDFT (future)

- How to implement charge constraints in the batched GPU SCC loop?
- Lagrange multiplier optimization: host-side or device-side?
- How does CDFT interact with geometry relaxation?
- **Deferred** — specified later by user.

---

## 6. Contracts and Tests

### 6.1 Contract: N>64 GPU SCC parity

**Test:** `tests/gpu_scc_n64plus.rs::test_gpu_scc_at_pair` (new)
- System: adenine-thymine pair (30 atoms, ~87 orbs)
- Run: GPU SCC (Option C: host LAPACK fallback, then Option A: tiled Jacobi)
- Reference: CPU SCC (`HamiltonianBuilder::build_scc`) with LAPACK
- Tolerances (f32):
  - Energy: |dE| < 1e-4 Ha
  - Charges: max|dq| < 1e-3 e
  - Eigenvalues: max|dε| < 1e-4 Ha
- **Pass criterion:** parity within tolerance AND no NaN/Inf/convergence
  failure. If convergence fails, test reports the failure loudly (no
  silent skip).

### 6.2 Contract: GPU force parity (when implemented)

**Test:** `tests/gpu_forces.rs::test_gpu_forces_formic_dimer` (new)
- System: formic acid dimer (28 orbs)
- Run: GPU analytic forces (or finite-difference forces)
- Reference: CPU analytic forces (`compute_scc_forces`)
- Tolerances (f32):
  - max|dF| < 1e-3 Ha/Å per atom
  - Energy: |dE| < 1e-5 Ha
- **Diagnostic output:** per-atom force residual, worst contributor, sign
  of deviation. Not just pass/fail.

### 6.3 Contract: batched relaxation convergence

**Test:** `tests/gpu_relax.rs::test_batched_relax_formic_dimer` (new)
- Systems: 10× formic acid dimer, same starting geometry
- Run: batched GPU relaxation (FIRE, 50 steps max, f_tol=5e-3)
- Reference: CPU relaxation (`optimize_geometry_persistent`)
- Tolerances:
  - Final energy: |dE| < 1e-3 Ha (relaxed geometries may differ slightly)
  - Final max|F|: < 1e-2 Ha/Å (all systems converged)
  - All 10 systems converge within 50 steps
- **Diagnostic:** per-system convergence history (energy, max|F|, SCC iters
  per step). Plot: `debug/hbond_relax/batched_relax_convergence.png`

### 6.4 Contract: relaxed 2D PES — nucleobase pair

**Test:** `tests/gpu_relaxed_pes.rs::test_at_pair_relaxed_2d` (new, long-running)
- System: adenine-thymine pair (~87 orbs)
- Grid: 11×11 = 121 points (coarse first, then 21×21 for production)
- Run: batched GPU relaxed scan (constrain 2 H atoms, relax rest)
- Reference: CPU relaxed scan on a subset (e.g. diagonal t1=t2, 11 points)
- Tolerances:
  - Relaxed energy: |dE| < 1e-3 Ha per point (vs CPU reference)
  - All points converge (or report non-converged points loudly)
- **Diagnostic output:**
  - 2D PES contour plot: `debug/nucleobase_scan/at_relaxed_2d_pes.png`
  - Synchronous 1D cut (diagonal): `debug/nucleobase_scan/at_sync_1d.png`
  - Asynchronous 1D cuts (rows/columns): `debug/nucleobase_scan/at_async_1d.png`
  - Barrier height: report E_TS - E_reactant along synchronous path
  - Intermediate detection: report if (0,1) or (1,0) corners are local minima

### 6.5 Contract: performance benchmark

**Test:** `tests/gpu_perf.rs::bench_batched_relaxation` (new, `--ignored`)
- Systems: formic dimer (28 orbs), AT pair (87 orbs)
- Batch sizes: 1, 10, 50, 100, 200, 500, 1000
- Measure:
  - Wall time per relaxation step (H0/S, SCC, forces, FIRE)
  - SCC iterations per step (with warm-start)
  - Total relaxation time to convergence
  - GPU kernel time breakdown (OpenCL profiling events)
  - Systems/second throughput
- **Output:** `debug/gpu_perf/batched_relax_benchmark.tsv` + plot
  `debug/gpu_perf/batched_relax_scaling.png`
- **Must report:** bottleneck kernel (highest % of time), GPU occupancy,
  PCIe transfer overhead.

### 6.6 Contract: parameter sweep

**Test:** `tests/gpu_param_sweep.rs::sweep_scc_params` (new, `--ignored`)
- System: AT pair, single geometry (near TS)
- Sweep: alpha ∈ {0.01, 0.05, 0.1, 0.2}, DIIS history ∈ {4, 8, 12, 15},
  SCC tol ∈ {1e-5, 1e-7, 1e-9}, Jacobi tol ∈ {1e-5, 1e-7, 1e-10}
- Measure: SCC iterations to convergence, total time, final residual
- **Output:** `debug/gpu_perf/param_sweep.tsv` + heatmap
  `debug/gpu_perf/param_sweep_heatmap.png`
- **Goal:** identify optimal parameter set for nucleobase pairs.

### 6.7 Monitoring: fail-loud invariants

Throughout all tests, assert:
- All energies finite (no NaN/Inf)
- All charges finite and |q_i| < 10 (unphysical if larger)
- SCC residual monotonically decreasing (or report oscillation)
- No silent fallback to CPU (if GPU path fails, crash with context)
- Force norm decreasing during relaxation (or report divergence)

---

## 7. Implementation Plan (phased)

### Phase 0: Unblock N>64 (1-2 days)
- Implement Option C: host LAPACK fallback for N>64 in `gpu_scc.rs`
- Add `gpu_solve_scc_batched_diis_hybrid` that uses GPU Jacobi for N≤64
  and host LAPACK for N>64
- Test: AT pair SCC parity (Contract 6.1)

### Phase 1: Batched relaxation with finite-difference forces (3-5 days)
- Implement finite-difference force evaluation using existing GPU SCC
  energy kernel (batched: 6N+1 energy evals per system)
- Implement batched FIRE optimizer with active mask
- Implement constraint projection (fix H atoms)
- Test: batched relaxation convergence (Contract 6.3)
- Benchmark: performance with finite-difference forces (Contract 6.5)

### Phase 2: Relaxed 2D PES for nucleobase pairs (2-3 days)
- Build AT and GC pair scan geometries (identify H-bond atoms)
- Run coarse 11×11 relaxed 2D scan
- Plot PES, identify synchronous/asynchronous paths
- Test: Contract 6.4

### Phase 3: Parameter tuning (2-3 days)
- Run parameter sweep (Contract 6.6)
- Identify optimal parameters for nucleobase pairs
- Fix 2D scan non-convergence (try level shifting, Broyden, etc.)
- Re-run full 21×21 scan with tuned parameters

### Phase 4: Analytic GPU forces (5-7 days, can parallelize)
- Port dH0/dx, dS/dx to OpenCL kernels
- Port gamma' and repulsive spline' to OpenCL
- Implement GPU analytic force kernel
- Test: force parity (Contract 6.2)
- Benchmark: analytic vs finite-difference forces

### Phase 5: Tiled multi-WG Jacobi (5-7 days, can parallelize)
- Implement tiled Brent-Luk Jacobi for N>64 using global memory
- Replace host LAPACK fallback
- Test: AT pair SCC parity with tiled Jacobi
- Benchmark: tiled Jacobi vs host LAPACK vs N≤64 local Jacobi

### Phase 6: Production screening (ongoing)
- Screen multiple nucleobase pair geometries (AT, GC, mispairs, modified bases)
- Full 21×21 relaxed 2D PES for each
- Compare barriers across systems
- Report: tautomerization propensity across nucleobase pairs

### Phase 7: CDFT (future, specified later)
- Charge constraints in SCC loop
- Separate proton/electron tracking
- Constrained relaxation

---

## 8. Geometry Generation — External Repos (do NOT duplicate here)

**Policy:** We do NOT want to pollute the dftbplus repo with geometry-building
machinery that already exists in other repos. Instead, generate geometries
**in the external repos** and export `.xyz` / `.mol2` files into
`data/xyz/` here. This section documents where the tools live and how to use
them.

### 8.1 SPAMMM — H-bonded system builder (ASCII art + Kekule)

**Repo:** `/home/prokop/git/SPAMMM`
**Codemap:** `SPAMMM/CODEMAP.md` (entry point)
**Topical audit:** `SPAMMM/doc/Topics/ReactionCoordinateScan.md`

SPAMMM has the most developed and well-tested H-bonded system builder. The
core idea: draw the molecular topology as **ASCII art**, where `:` marks
H-bond donor-acceptor pairs. The builder generates 3D coordinates, assigns
bond orders via Kekule solver, caps with H atoms, and resolves H-bond pairs.

**Key files:**

| File | Role |
|---|---|
| `spammm/topology/ascii_art_heterocycle.py` | ASCII art parser → `AtomicSystem` with 3D coords, H-bond `:` marks, Kekule bond orders. Contains `ASCII_EXAMPLES` dict with pre-built nucleobases: `uracil`, `cytosin`, `guanin`, `purin`, `7azaindol`, `karbazol`, etc. |
| `spammm/topology/hbond_utils.py` | `HbondRecord`, `find_hbonds_graph`, `controls_to_fractions` — H-bond discovery on molecular graphs for RC scans |
| `spammm/quantum/hbond_scan.py` | Rigid DFTB proton-transfer scan: `make_hbond_transfer_path`, `run_hbond_transfer_scan` — slides H along donor→acceptor axis |
| `spammm/quantum/coordinate_scan.py` | Multi-control RC scan: `build_control_grid`, `build_frame`, `run_rigid_dftb_scan`, `run_pm_neb` (relax + interp + Mulliken SP) |
| `spammm/topology/scan_dataset.py` | `ScanDataset` `.npz` I/O for trajectory data (geometry, charges, controls) |
| `spammm/topology/KekulePure.py` | Kekule pi-bond order solver |
| `spammm/topology/MoleculeEditorBackend.py` | Molecular editor: graph ↔ dense arrays, hex grid, export |
| `spammm/quantum/DFTB_utils.py` | `run_dftb_sp`, `run_dftb_relax`, `parse_mulliken_charges` — DFTB+ interface |

**Pre-built ASCII examples** (in `ascii_art_heterocycle.py::ASCII_EXAMPLES`):
- `uracil`, `cytosin`, `guanin`, `purin`, `7azaindol`, `karbazol`
- `NTCDA`, `NTCDI` (naphthalene diimide dimers with `:` H-bonds)
- `naphthalene`, `perylene`, `biphenylene`, `phenanthrene` (PAHs)

**Pre-built geometries** (in `SPAMMM/data/`):
- `data/xyz/adenine.xyz`, `guanine.xyz`, `thymine.xyz`, `uracil.xyz`,
  `azaindol.xyz`, `azaindol_dimer.xyz`, `azaindol_isodimer.xyz`
- `data/mol/adenine-uracil.mol2`, `adenine-uracil-iso.mol2`,
  `azaindol_dimer.mol2`, `formic_acid.mol2`, `benzoicacid_dimer.mol2`,
  `benzoicamid_dimer.mol2`

**Workflow to generate nucleobase pair scan geometries:**

1. **Build the base pair from ASCII art** (or load existing `.mol2`):
   ```python
   from spammm.topology.ascii_art_heterocycle import parse_ascii_art, ASCII_EXAMPLES, resolve_hbond_pairs
   atoms = parse_ascii_art(ASCII_EXAMPLES['guanin'], hbond_length=1.6)
   atoms.neighs()
   resolve_hbond_pairs(atoms)  # → atoms.hbonds_ascii = [(h_idx, acceptor_idx), ...]
   ```

2. **Or build a dimer** with `:` H-bond marks (ASCII dimer format):
   ```
   O o O          ← donor/acceptor atoms, `o` = lowercase = sp3
    | |           ← bond row: | = vertical dimer bond
   | | |          ← `:` pairs become H-bonds
    | |
   O o O
   ```

3. **Build the 2D scan grid**:
   ```python
   from spammm.quantum.coordinate_scan import build_control_grid, build_frame
   from spammm.topology.hbond_utils import HbondRecord, default_mapping, controls_to_fractions
   # For 2 H-bonds, 2 controls (asynchronous): m=2
   controls = build_control_grid([(0, 1), (0, 1)], dx=0.05)  # 21×21 grid
   mapping = default_mapping(n_hbonds=2, m=2)  # each H-bond has its own control
   frames = [build_frame(atoms.apos, hbonds, ctrl, mapping) for ctrl in controls]
   ```

4. **Export frames as XYZ** for the dftbplus Rust solver:
   ```python
   from spammm.atomicUtils import saveXYZ
   for i, frame in enumerate(frames):
       saveXYZ(f'data/xyz/scan_frame_{i:04d}.xyz', atoms.enames, frame)
   ```

5. **Or use the rigid DFTB scan directly** (SPAMMM calls DFTB+ Fortran):
   ```python
   from spammm.quantum.coordinate_scan import run_rigid_dftb_scan
   result = run_rigid_dftb_scan(atoms, hbonds, ranges=[(0,1),(0,1)], dx=0.05, work_dir='debug/scan/')
   ```

**For this task:** use SPAMMM to generate the 2D scan frame geometries as
`.xyz` files, then feed them to the Rust GPU solver in batched mode. The
SPAMMM DFTB+ interface (`DFTB_utils.py`) can also compute CPU reference
energies for parity.

### 8.2 Existing geometries in dftbplus

These are already in the dftbplus repo and can be used directly:

| File | Atoms | Orbs | Notes |
|---|---|---|---|
| `data/xyz/formic_dimer.xyz` | 10 | 28 | Formic acid dimer, validated |
| `data/xyz/adenine-thymine.xyz` | 30 | ~87 | AT Watson-Crick pair |
| `data/xyz/guanine-cytosine.xyz` | 29 | ~86 | GC Watson-Crick pair |
| `data/xyz/azaindol_dimer.xyz` | 30 | 84 | 7-azaindole dimer |
| `data/xyz/azaindol_isodimer.xyz` | 30 | 84 | iso-azaindole dimer |
| `data/xyz/adenine.xyz` | 15 | 45 | adenine monomer |
| `data/xyz/guanine.xyz` | 16 | 49 | guanine monomer |
| `data/xyz/thymine.xyz` | 15 | 42 | thymine monomer |
| `data/xyz/uracil.xyz` | 12 | 34 | uracil monomer |
| `data/xyz/citosine.xyz` | 13 | 37 | cytosine monomer |

**Note:** the nucleobase pair `.xyz` files in dftbplus may not have the
transferring H atoms in the right positions for a proton-transfer scan. Use
SPAMMM's `coordinate_scan.build_frame` to generate the displaced geometries
from the base pair, or use SPAMMM's ASCII art builder to construct the
dimer with explicit `:` H-bond marks.

### 8.3 What NOT to build in dftbplus

- **No ASCII art parser** — use SPAMMM's `ascii_art_heterocycle.py`
- **No Kekule solver** — use SPAMMM's `KekulePure.py`
- **No H-bond discovery** — use SPAMMM's `hbond_utils.py`
- **No DFTB+ Python interface** — use SPAMMM's `DFTB_utils.py` for reference
- **No ScanDataset I/O** — use SPAMMM's `scan_dataset.py` if needed

The dftbplus Rust crate should only **consume** the generated `.xyz` files.
All geometry building stays in SPAMMM.

---

## 9. File Ownership

| File | Status | Owner |
|---|---|---|
| `rust_dftb/src/qmqm/gpu_scc.rs` | exists, extend for N>64 hybrid | this task |
| `rust_dftb/src/qmqm/gpu_forces.rs` | new (analytic GPU forces) | this task |
| `rust_dftb/src/qmqm/gpu_relax.rs` | new (batched FIRE/L-BFGS) | this task |
| `rust_dftb/src/qmqm/gpu_eigen.rs` | exists, extend for tiled Jacobi | this task |
| `rust_dftb/tests/gpu_scc_n64plus.rs` | new | this task |
| `rust_dftb/tests/gpu_forces.rs` | new | this task |
| `rust_dftb/tests/gpu_relax.rs` | new | this task |
| `rust_dftb/tests/gpu_relaxed_pes.rs` | new | this task |
| `rust_dftb/tests/gpu_perf.rs` | new | this task |
| `rust_dftb/tests/gpu_param_sweep.rs` | new | this task |
| `rust_dftb/examples/nucleobase_scan.rs` | new (production CLI) | this task |
| `scripts/plot_relaxed_pes.py` | new | this task |
| `data/xyz/adenine-thymine.xyz` | exists | — |
| `data/xyz/guanine-cytosine.xyz` | exists | — |
| `data/xyz/scan_frame_*.xyz` | generated by SPAMMM, imported | external |

---

## 10. Related Documents

- `doc/prokop/tasts/GPU_MultiSystem/hbond_switching.md` — prior H-bond scan
  task (formic dimer, azaindole dimer). This task supersedes the "relaxed
  scan" section (Task 6, previously deferred).
- `doc/prokop/DFTB_Reimplementation_Progress/GPU_MultiSystem_Design.md` —
  GPU architecture design (D1-D17 decisions).
- `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` — master
  status checklist. Sections 6, 7 track GPU SCC and multi-system.
- `doc/prokop/reports/2025-09-06_gpu_scc_benchmarks.md` — baseline
  performance measurements.
- `doc/prokop/tasts/GPU_MultiSystem/task_master.md` — original 5-agent
  task plan (Wave 1 + Wave 2, all completed).
