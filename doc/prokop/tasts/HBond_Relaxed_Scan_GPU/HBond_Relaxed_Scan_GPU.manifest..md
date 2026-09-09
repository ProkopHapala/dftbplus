# Task 1: GPU Multi-System Relaxed Scan of Hydrogen-Bonded Nucleobase Pairs

**Created:** 2026-09-07
**Status:** dense H-bond kernel — interpolator stopgap + analytic forces measured. AT/GC GPU SCC rms plateaus `~1e-5` — **hypothesis: f32 floor, not a broken mixer** (see §3.0.1). Sparse is a **separate** agent — do not edit `rust_dftb/src/methods/sparse/`.
**Owner:** prokop / Devin
**Interpolator spec:** `doc/prokop/topical_audit/sk_interpolation.md` (what was done vs what must be done next)

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

### 1.1 Performance mandate — fastest GPU DFTB in the world

This project aspires to be the **fastest GPU-accelerated DFTB+ implementation
in the world**, not a toy for one afternoon. Every design decision is judged
against three ordered criteria:

1. **Physical correctness** — the math must be right, the model must be
   self-consistent, the forces must be the exact gradient of the energy.
2. **Scientific rigor** — tests must prove what they claim; tolerances must
   reflect the science (meV barriers, not 0.27 eV); no test is green by
   loosening its contract.
3. **Maximum performance** — eliminate every avoidable overhead: no
   allocation in hot loops, no host roundtrip in an iteration, no kernel
   rebuild per step, no `finish()` for bookkeeping, no dense fallback hidden
   in a sparse path, no atomic where a deterministic reduction suffices, no
   CPU sort of 87 numbers that one WG can do in local memory.

**KISS, but not at the cost of speed.** Simple code is preferred when it is
*also* fast. When "simple" means "slow" — e.g. a CPU bridge through an
allocation-heavy dense function, a serial pivot where a parallel Brent–Luk
exists, a per-bucket `finish()`, a CAS-loop atomic where a one-WG local
reduction is deterministic and cheaper — the answer is **no**. We write the
fast version. The code may be longer; that is acceptable.

**The recurring failure mode this section exists to prevent:** a coding agent
takes a shortcut that is locally simpler, the test passes because the
shortcut is correct in isolation, and the result is a 2–10× performance
regression or a hidden correctness bug that only surfaces under real
workloads (batch=500, N=87, 6N Hessian displacements). The GPT 5.6 review
(section 11) catalogs exactly this pattern. Every item there is a case where
"simple and easy" was chosen over "fast and correct", and the cost was paid
in either physics or throughput.

**Concrete implications for this task:**

- One shared `GpuRuntime` and one persistent plan own positions → H/S →
  SCC → P/W → forces. No second context, no per-bucket uploads, no force
  driver that allocates and downloads.
- P and W are already GPU buffers after SCC. Forces stay on GPU for FIRE.
  No CPU copy between SCC and force evaluation.
- Occupation selection, DIIS, and convergence reduction happen on device.
  A 4-byte scalar read per iteration is a synchronization, and
  synchronization dominates a many-small-system workload.
- The eigensolver, GEMM, and force kernels are built once and reused. Buffer
  capacity is preallocated; changing arguments is `set_arg`, not
  `Buffer::builder()`.
- Analytic derivatives are the production route. Finite differences of H/S
  are test/reference only.
- Tests prove what they claim. A test that says "residual < 1e-5" must
  assert `< 1e-5`, not `< 1e-3` because the solver misses. Fix the solver.

This mandate is the lens through which the GPT 5.6 review (section 11) and
all subsequent implementation should be read.

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

### 3.0 SK interpolation end BCs (2026-09-09) — read before touching the spline

CPU f64 is the reference; GPU f32 must match it. Dense H-bond only.

**Bug that was fixed:** production `eval_bspline_*` delegated `[last_grid, last_grid+1 Bohr]` to 8-point Neville + `poly5_to_zero`. On H–H the last table value is ~1e-5 Ha; at 10.39 Bohr (AT pair) CPU returned Hss **−0.4 Ha**. GPU clamped to ~0. That was the AT/GC `max|dH|~0.4` “assembly” failure. Do **not** restore Neville for Fortran-tail parity. DFTB+ uses the same 1 Bohr fudge; we are replacing that tail because it is unphysical.

**What the code does now (stopgap):**

- Production path: C² cubic B-spline, analytic `V'` from the same controls (`interpolation.rs`, `spline_resample.rs::bspline3_eval_v_d1_d2`). GPU uses the same 4-point stencil (`cubic_weights` / `cubic_weights_d1`).
- **Left end:** phantom control `c_{-1} = 2c_0 − c_1` at evaluation. SK at small r is large — do not pad zeros on the left. This is the *right kind* of BC (extra control implementing a condition).
- **Right end (blunt):** append `N_PAD_END=4` **function samples of exact 0**, then refit the whole tridiagonal (`fit_bspline_controls_zero_end`). That interpolates zeros in the pad and killed the explosion. Extra knots are **hardcoded 0**, not solved for. Cutoff is `last_grid + 4·dr` (~0.08 Bohr), not +1 Bohr.

**What must be done next (do not code as “set pad to zero” again):**

Cubic B-splines need a few extra control points **before and after** the tabulated domain. Those points must be **fitted**:

1. General fitter: original samples + extra conditions (value / `V'` / `V''` continuity at the last interior knot; `V→0`, `V'→0` at cutoff).
2. Unknowns = extra controls left of sample 0 and right of the last sample (2–4 per side).
3. Solve so the interpolant on the **valid domain** stays accurate (pad must not pollute the interior polynomial).
4. Left: generalize the phantom, never force `V=0`. GPU’s stored dummy 0 at `r=0` is also blunt — should become a fitted left control or a pure index shift.

The phantom formula is the pattern to generalize. The zero-sample pad is only a temporary way to stop the tail explosion.

**Measured (mio-1-1, RTX 3090, `tests/gpu_hbond_physics.rs`):** H–H at 10.00 Bohr Hss `2e-22`; at 10.39 Bohr exact 0. AT/GC/H2O GPU vs CPU max|dH| `8.6e-8` / `7.4e-8` / `4.4e-8`. CPU analytic F vs FD of energy (h=1e-3 Å) rel `1.05e-5`. GPU four force kernels vs CPU rel `~3e-5`.

#### 3.0.1 AT/GC GPU SCC rms `~1e-5` — f32 floor hypothesis (not a proven bug)

Observation: GPU SCC on AT/GC (N=87/86) does not reach charge-rms `<1e-6` in 500 steps; it plateaus `~1e-5`. CPU f64 on the **same** H/S goes to `~1e-9` in ~20 steps. H/S GPU vs CPU already matches (`max|dH|~1e-7`), so this is **not** the interpolator.

**Hypothesis (do not treat as a mixer/Jacobi failure until checked):** this is the f32 dynamical range.

- f32 relative rounding is `~1e-7` (machine ε) to `~1e-8` after a well-conditioned chain. Take `~1e-8` as the optimistic relative floor.
- Hamiltonian / potential entries of order `~100` are normal (e.g. ~100 eV electron–nuclei / onsite-scale `H_ij` near `r=0`; equivalently a few Hartree). Absolute noise is then `100 × 1e-7 … 1e-8` **`= 1e-5 … 1e-6`**.
- Charge rms `~1e-5` is therefore **consistent with f32**, not evidence that DIIS or tiled Jacobi is stuck. CPU f64 (ε `~1e-16`) can honestly go to `1e-9`; GPU f32 cannot.

Do **not** chase rms `<1e-6` on f32 AT/GC by loosening the mixer or the test. First check the scale: print `max|H|`, `max|S|`, onsite, energy, and whether rms is a plateau vs a slow drift. If energy vs CPU is already at f32-noise and charges oscillate at `1e-5`, the production contract for N~90 f32 SCC is rms `~1e-5`, not `1e-6`. H2O (N=6, smaller `|H|` span) already reaches `~1e-7` — that does not contradict a larger-system floor.

### 3.1 GPU multi-system SCC solver (N≤64, dense)

- **Batched H/S assembly** — `qmqm/gpu_driver.rs::gpu_assemble_batched`,
  `qmqm/gpu_prep.rs::GpuBatch::from_fragments`. Tested on replicated small
  systems and formic-dimer scans.
- **Brent–Luk parallel cyclic Jacobi** —
  `qmqm/gpu_eigen.rs::jacobi_cyclic_local_batched`. One WG per system,
  A+V stored in `__local`; hard-coded production limit N≤64.
- **S⁻¹/²** — `qmqm/gpu_eigen.rs::build_inv_sqrt`. Jacobi(S) followed by
  `U·Λ⁻¹/²·Uᵀ`.
- **Full-local GEMM** — `qmqm/gpu_matrix.rs::matmul_full_local_batched`.
  This is a second independent N≤64 limitation: both operands are loaded
  fully into local memory.
- **GPU SCC arithmetic** — Δq → gamma matvec → H_scc update →
  Löwdin transform → Jacobi → density → Mulliken is implemented on GPU.
- **Important current plumbing limitation:** the production DIIS path is
  *not yet fully device-resident*. Per SCC iteration it currently:
  1. reads the full `batch×N×N` diagonalized H′ buffer to the host merely
     to extract/sort eigenvalues and build occupations;
  2. uploads the occupation mask;
  3. reads `q_new` and `q_cur` to the host for CPU DIIS;
  4. uploads mixed charges again.
  These transfers and host synchronizations must be removed before treating
  current timing as a clean GPU-compute baseline.
- **Kernel/buffer lifecycle limitation:** programs are cached, but many
  `Kernel::builder()` calls still occur in frequently executed wrappers,
  and SCC scratch buffers are allocated per solve rather than once per
  persistent relaxation plan.
- **Parity already validated** on small systems and the formic-dimer scan.
- **Measured baseline** (before the cleanup above): ~1194 systems/s at
  batch=100 for the 28-orbital formic dimer; Jacobi is the dominant kernel
  (~50–65% of per-iteration time in the existing measurements).

This inventory is deliberately literal: do not describe a CPU/GPU transfer
or allocation as “negligible” merely because the byte count is small. In a
many-small-system workload, synchronization and launch/allocation overhead
can dominate the arithmetic.

---

### 3.2 CPU forces and geometry optimization

- **Validated CPU SCC forces** —
  `methods/dftb/forces.rs::compute_scc_forces`, with four components:
  `F_nonSCC + F_SCC_shift + F_SCC_dc + F_rep`. Existing parity against the
  Fortran reference is excellent.
- **But the current validated force path is not fully analytic in H/S
  derivatives.** `non_scc_electronic_force()` and `scc_shift_force()` call
  `pair_block_derivative()`, which forms central finite differences of the
  pair H/S block.
- **Analytic SK-block derivatives do already exist** in
  `rotation.rs::rotate_block_with_derivs_into` together with radial spline
  derivatives. They are promising, but they are not yet the source used by
  the validated `compute_scc_forces()` path. Before porting them to OpenCL,
  wire them into a CPU force path and prove parity against both the current
  finite-difference derivative implementation and the Fortran force.
- **FIRE optimizer** — `examples/hbond_ref.rs::FireOptimizer`,
  `optimize_geometry_persistent`; currently CPU/single-system.
- **CPU scan driver** — rigid and relaxed reference infrastructure exists.

This distinction matters: the GPU task is not merely “port the already
validated analytic force kernel”. It is “validate the analytic derivative
formulas first, then port exactly the validated formulas and conventions”.

---

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

### 4.1 Remove accidental CPU/GPU overhead first (BLOCKING Phase 0)

The chemistry workload consists of many tiny dense systems. Therefore kernel
launches, blocking reads, queue finishes, kernel construction, and repeated
buffer allocation are architectural costs, not cosmetic cleanup.

#### Mandatory checklist

- [ ] Remove unconditional `rt.finish()` from computational wrappers such as
      `jacobi_cyclic_local_batched()` and `build_inv_sqrt()`. An in-order queue
      already preserves command order; synchronize only when the host genuinely
      needs a result.
- [ ] Replace the S GPU→CPU→GPU copy in `build_inv_sqrt()` by a device-to-device
      copy.
- [ ] Create a persistent `GpuSccPlan` (or equivalent) which owns scratch buffers
      and reusable kernel objects for a fixed `(N, Na, batch_capacity, config)`.
- [ ] Do not allocate SCC scratch matrices in every relaxation step.
- [ ] Move eigenvalue extraction / occupation construction to the GPU. Do not
      download an entire N×N matrix just to read N diagonal numbers.
- [ ] Remove CPU DIIS from the hot SCC iteration. A per-system DIIS history is
      tiny; a small local solve on GPU is appropriate. Until GPU DIIS exists,
      at minimum avoid downloading data already mirrored on the host.
- [ ] Replace hard-coded RTX-3090-like capability values in
      `GpuRuntime::query_capabilities()` with actual OpenCL queries.
- [ ] Use profiling events for GPU timing. Do not insert `queue.finish()` merely
      to make wall-clock timers look synchronous.
- [ ] Re-run the existing formic-dimer benchmarks after cleanup and save the new
      baseline before optimizing the eigensolver.

**Hard rule:** no buffer allocation, kernel construction, host roundtrip, or
global queue synchronization inside the simulation/SCC hot loop unless there
is a measured and documented reason.

### 4.2 Dense eigensolver architecture: exactly two paths

We intentionally keep the solver family small:

1. **N≤64:** existing full-local Brent–Luk Jacobi.
2. **N>64:** one-WG tiled/block Jacobi with global A/V backing and O(B²)
   local workspace.

No production host LAPACK fallback, no multi-WG cooperative Jacobi requiring
global barriers, no A-local/V-global special-case solver, and no dense
purification work in this task.

#### Why one WG per system

The target workload supplies many independent systems. The useful top-level
parallelism is therefore across systems. Inside each system, 256–512 threads
can cooperate on a local pivot and strip updates while all inter-step
synchronization remains a cheap WG-local barrier.

**Forbidden anti-patterns**

- [ ] **FORBIDDEN:** one Jacobi round, pivot, or sweep per kernel launch.
- [ ] **FORBIDDEN:** multiple WGs cooperate on one Jacobi matrix if correctness
      requires synchronization between them.
- [ ] **FORBIDDEN:** host-side LAPACK in every SCC iteration as a production
      N>64 path.
- [ ] **FORBIDDEN:** any host-driven inner eigensolver loop.

The complete eigensolve of one batch must be one OpenCL kernel launch, with
one WG owning each system until that system converges.

### 4.3 One-WG tiled/block Jacobi

Partition the physical N×N matrix into blocks of capacity B. For a block pair
`(p,q)`, load only the compound principal block and a strip workspace into
local memory.

Initial candidates:

- B = 24 or 32;
- WG = 256 or 512.

Do **not** assume B=32 fits just because
`2·64·65·sizeof(float) ≈ 33 kB`. The actual kernel also needs rotation
parameters, reductions, strip scratch, and compiler-accounted local storage.
Query both `CL_DEVICE_LOCAL_MEM_SIZE` and the built kernel's
`CL_KERNEL_LOCAL_MEM_SIZE`.

#### Critical tail-block rule: do not globally pad the physical problem

For N=87 and B=32 there are block sizes `32, 32, 23`. Do **not** turn the
physical H or S into a 96×96 matrix with ordinary zero padding. In particular,
zero-padding S introduces artificial zero overlap eigenvalues and makes
S⁻¹/² mathematically singular.

Instead:

- keep global matrices at the real physical dimension N×N;
- let the final block have `B_last = N - floor(N/B)·B`;
- for a local fixed-capacity `2B×2B` pivot, mark unused local lanes as dummy
  states with zero coupling to real lanes and a harmless sentinel diagonal;
- exclude dummy lanes from convergence norms, eigenvalue output, occupation,
  and all physical matrix stores.

The existing N≤64 Jacobi already uses this idea for an odd dummy state
(huge diagonal, zero off-diagonal). Generalize that concept only inside the
local pivot; do not create physical dummy orbitals.

#### Pivot transformation

For active block sizes `Bp`, `Bq`, let `m = Bp+Bq ≤ 2B` and form

\[
P =
\begin{pmatrix}
A_{pp} & A_{pq}\\
A_{qp} & A_{qq}
\end{pmatrix}.
\]

The whole WG diagonalizes this local symmetric matrix:

\[
U^T P U = D.
\]

Use the already-tested local Brent–Luk rotation machinery where practical.

Then, for each outside row tile `k`, load

\[
X = [A_{kp}\;A_{kq}]
\]

into local strip storage and compute

\[
Y = XU.
\]

Store Y into the two affected column strips and store `Yᵀ` into the matching
row strips. Because A is maintained explicitly symmetric, this applies both
sides of the orthogonal similarity transform without a separate global
left-multiplication pass.

Eigenvectors are accumulated as

\[
[V_{:p}\;V_{:q}] \leftarrow [V_{:p}\;V_{:q}]U.
\]

`U` remains in local memory while all A and V strips for that pivot are
processed.

Conceptual kernel:

```c
kernel tiled_jacobi_batched(A, V, N, ...) {
    sid = get_group_id(0);        // exactly one system
    initialize physical V = I;

    for (sweep = 0; sweep < max_sweeps; ++sweep) {
        for each cyclic block pair (p,q) {
            load physical compound pivot into local storage;
            fill only unused local lanes with dummy sentinel states;
            barrier(local);

            local_jacobi(pivot, U, active_m);

            store transformed physical principal block;
            barrier(local | global);

            for each outside physical row tile k {
                load [A_kp | A_kq];
                Y = X * U;
                store Y and its symmetric transpose;
            }

            for each physical V row tile k {
                load [V_kp | V_kq];
                Y = X * U;
                store Y;
            }

            barrier(local | global);
        }

        compute offdiag norm over physical N×N only;
        if (converged) break;
    }
}
```

At the outer block level use a simple deterministic cyclic pair order first.
Do not add Brent–Luk scheduling between blocks until profiling demonstrates
a need; parallel block-pair execution within one WG complicates shared strip
updates and is not required for N≈87.

#### Scope

This algorithm has no N² local-memory limit, so it is *functionally* general
for dense matrices larger than 64. The performance target of this project is
roughly N≈65–160, especially N≈84–90 nucleobase dimers. Do not claim that one
WG per matrix is automatically efficient for N=500–1000; that is a future
scaling question.

### 4.4 General tiled GEMM is independently required

`matmul_full_local_batched()` also rejects N>64, so replacing Jacobi alone
does not unblock nucleobases.

Integrate the existing conventional tiled GEMM into the shared `GpuRuntime`
production path:

- output tiles may use multiple WGs per matrix because GEMM has no
  inter-WG synchronization dependency;
- keep the existing full-local GEMM for N≤64 only if it remains faster after
  clean profiling;
- the production SCC API should choose only between small full-local GEMM and
  the general tiled GEMM, without host intervention.

N>64 S⁻¹/² reconstruction also needs the general tiled algebra; the existing
`build_inv_sqrt_from_eig` loads the complete eigenvector matrix into local
memory and therefore is not itself a general N>64 implementation.

### 4.5 Occupation and density handling must remain on device

After Jacobi, the physical diagonal contains N eigenvalues and V contains the
corresponding (not necessarily sorted) eigenvectors.

For a fixed closed-shell electron count:

1. extract the N diagonal values on GPU;
2. select/sort the `N_occ` lowest eigenvalue indices in local memory;
3. build an occupation/index list without permuting the full eigenvector
   matrix unless permutation is actually beneficial;
4. build the AO density directly on GPU.

For analytic forces, build **both** ordinary and energy-weighted density in
the same occupied-state contraction after back-transforming the eigenvectors:

\[
P_{\mu\nu}=2\sum_{k\in occ} C_{\mu k}C_{\nu k},
\]

\[
W_{\mu\nu}=2\sum_{k\in occ} \epsilon_k C_{\mu k}C_{\nu k}.
\]

Do not add an unnecessary dense `W'=P'H'` GEMM when eigenvalues/eigenvectors
are already available. Reuse the same C loads and occupation list to produce
P and W together.

### 4.6 Warm starts: separate required from experimental

**Required**

- Charge warm-start between successive SCC solves of the *same relaxing
  geometry trajectory*. This is already implemented and is high value.

**Optional after the clean N>64 baseline**

- Eigenbasis warm-start between SCC iterations:
  `A0 = C_prev'^T H'_new C_prev'`, followed by a few Jacobi sweeps and
  `C_new' = C_prev' U`. Benchmark total time, because the basis transform
  costs two GEMMs.
- Eigenbasis warm-start between relaxation steps.

**Deferred unless profiling proves worthwhile**

- Newton–Schulz or other iterative S⁻¹/² updates between geometry steps.
  The simple scalar-looking formula
  `X <- 0.5 X (3I - S X²)` is not automatically a safe symmetric
  inverse-square-root iteration for two non-commuting matrices from different
  geometries. It needs scaling, residual checks, and a numerically justified
  iteration. For the first production implementation simply recompute S⁻¹/²
  with the general eigensolver.

Do not create sequential dependencies between independent scan points merely
to warm-start them. Short continuation chains may be benchmarked later as a
scheduler optimization, but the default scan is fully batched.

### 4.7 Analytic GPU forces (BLOCKING for relaxed scan)

Production forces must be analytic. Finite differences are validation only.

#### Step A: validate the analytic CPU derivative path first

The currently validated CPU force routine uses central finite differences for
H/S pair-block derivatives, even though analytic helpers exist.

Before writing OpenCL:

- [ ] compare `rotate_block_with_derivs_into` against central finite differences
      for H and S blocks over random distances, orientations, and H/C/N/O pair
      types;
- [ ] pay special attention to s–p / p–s transpose/sign conventions and the
      `(py,pz,px)` orbital ordering;
- [ ] verify radial derivative units (SK distance in Bohr, molecular coordinates
      in Å);
- [ ] wire analytic derivatives into a CPU force path;
- [ ] recover the existing CPU-vs-Fortran total-force parity.

Only then port those exact validated formulas to OpenCL.

#### Step B: GPU pairwise direct contraction

Never materialize

```text
dH[3*Natoms][N][N]
dS[3*Natoms][N][N]
```

Instead, for each unique atom pair `(A,B)`:

1. load pair geometry and species;
2. evaluate SK radial values and radial derivatives;
3. form the small rotated H/S derivative blocks in registers/local scratch;
4. immediately contract those blocks with the corresponding P and W blocks;
5. evaluate SCC gamma derivative and repulsive derivative;
6. accumulate equal-and-opposite pair forces;
7. discard the derivative block.

For mio s/p atoms the orbital pair block is at most 4×4, so this is naturally
register/local-memory sized.

Port the **exact tested CPU sign and unit conventions** for all four force
components rather than re-deriving signs ad hoc in the OpenCL kernel:

- non-SCC electronic term;
- SCC shift / overlap term;
- SCC gamma double-counting term;
- repulsive spline term.

A one-WG-per-system force kernel is a plausible starting point for ~30 atoms.
Avoid atomics if possible by local force accumulation/reduction.

Finite-difference validation is limited to a handful of selected Cartesian
components and derivative-block tests; never use 6N+1 SCC evaluations as the
production force path.

### 4.8 Device-resident constrained relaxation

Positions, velocities, charges, forces, convergence flags, and optimizer state
remain on GPU across relaxation steps. FIRE itself is cheap enough to run on
device.

#### Critical physics point: constrain reaction coordinates, not all proton XYZ

The chemically meaningful relaxed scan should constrain one scalar transfer
coordinate per transferring proton while allowing the remaining degrees of
freedom to relax.

A robust coordinate is for example

\[
\xi = r_{D-H} - r_{A-H}
\]

(or an equivalent normalized transfer fraction). For two proton transfers
there are two scalar constraints `g1(R)=0`, `g2(R)=0`.

Do **not** simply freeze all three Cartesian coordinates of each H unless a
specific diagnostic scan intentionally requests that much stronger
constraint. Freezing H in absolute space is especially wrong when donor and
acceptor atoms themselves relax.

Because `ξ` depends on D, H, and A positions, the constraint gradient acts on
all three atoms. Therefore “zero the H force component” is not a complete
constraint algorithm.

For each system:

1. compute analytic physical force F;
2. compute constraint Jacobian G;
3. project force (and FIRE velocity) into the tangent space of the constraint
   manifold, preferably mass-weighted;
4. perform the FIRE step;
5. re-project/correct positions back onto `g(R)=0` (SHAKE-like correction);
6. verify constraint residual explicitly.

With only two scalar constraints, the Lagrange-multiplier solve is at most
2×2 per system and is trivial on GPU.

Convergence of a constrained relaxation is based on:

- projected/tangential force norm;
- constraint residual `|g|`;
- finite energy/coordinates.

The raw total force need not vanish because the constraint carries a reaction
force.

Correct outer order:

```text
for relax_iter:
    assemble H0/S for active systems
    solve SCC -> E, q, C, eps, P, W
    analytic force kernel -> F
    compute/project constraints -> F_tangent, v_tangent
    FIRE update on GPU
    constraint position correction
    update active flags from projected force + constraint residual
```

### 4.9 SCC robustness: keep the production hierarchy small

The existing DIIS already gives a large speedup. Do not respond to difficult
proton-transfer points by implementing five unrelated mixers immediately.

Implement first:

1. charge warm-start;
2. GPU DIIS;
3. a safeguard: if a DIIS extrapolation is non-finite or worsens the residual
   catastrophically, reject it and take a damped-mixing step;
4. adaptive damping based on recent residual behavior.

Only if diagnostics show HOMO/LUMO occupation flipping should level shifting
be added. Broyden/Anderson is a later fallback, not a Phase-0 requirement.

Do not require residual monotonicity. DIIS can increase the residual
temporarily. Fail on non-finite values, runaway charges, persistent divergence,
or exhausted iteration limits with explicit status.

### 4.10 Precision policy: measure-driven, not blanket f64

Keep expensive O(N³) matrix algebra in f32.

For reductions and small solves:

- first use numerically sensible FP32 reduction trees/pairwise accumulation;
- use f64 selectively where parity demonstrates a real need (for example the
  tiny DIIS linear solve or final diagnostic reductions);
- do not assume FP64 is “free” on an RTX 3090-class consumer GPU — FP64
  throughput is much weaker than FP32;
- compensated FP32 is another option for sensitive sums.

The quantities that determine whether precision is adequate are relative PES
errors, force parity, SCC convergence, eigensolver residual, and
orthogonality—not an abstract preference for f32 or f64.

### 4.11 Performance measurements

After Phase 0 cleanup, profile with OpenCL events and report both GPU and host
wall time.

Measure:

- H/S assembly;
- S⁻¹/²;
- Löwdin GEMMs;
- Jacobi time and sweeps;
- occupation + P/W build;
- SCC mixing;
- analytic force kernel;
- constraint + FIRE kernels;
- total relaxation-step wall time;
- PCIe bytes and blocking synchronizations;
- systems/s versus batch size.

Benchmark batch sizes `1, 10, 50, 100, 200, 500, 1000` where memory permits,
and system sizes covering formic dimer, azaindole, AT/GC, plus synthetic
dimensions around the block boundaries (`N=64,65,87,96,97,128`).

### 4.12 Relaxed 2D PES and scan scheduling

Production output remains a relaxed 2D proton-transfer PES for AT/GC and later
other H-bond/Kekulé-coupled systems.

Default scheduling keeps all scan points independent and advances them in
parallel. Warm-start each point across its *own relaxation steps*.

Optional later optimization: split a large scan into short continuation chains
of length ~2–4, but keep the number of independent chains at least comparable
to the number of GPU SMs. Benchmark this; do not make a 400-point scan into a
single or a few long sequential chains.

A coarse 11×11 surface may be followed by a second batched refinement in
interesting regions. This does not require sequential dependence between
individual points.

---

## 5. Open Questions and Challenges

### 5.1 Block-Jacobi tile/workgroup choice

Start with a very small benchmark matrix:

- B ∈ {24, 32};
- WG ∈ {256, 512};
- N ∈ {65, 87, 96, 97, 128};
- representative batch sizes.

B=32 is attractive because the pivot capacity is 64, but it is valid only if
the *actual built kernel* fits local memory with all scratch included.

Do not spend days tuning. Pick one general N>64 configuration after the first
measured comparison unless a clear size-dependent crossover appears.

### 5.2 Partial final block correctness

This deserves an explicit test because it is easy to get subtly wrong.

For N not divisible by B:

- no physical zero-padding of S;
- dummy pivot lanes never enter physical eigenvectors;
- residual/orthogonality are computed on the real N-dimensional problem;
- occupation ignores dummy lanes;
- N=65, 87, and 97 must be first-class test dimensions.

### 5.3 Block Jacobi convergence

Record per system:

- physical off-diagonal Frobenius norm after each block sweep;
- number of sweeps;
- number of useful/nontrivial local rotations;
- final residual and orthogonality.

Test both synthetic symmetric matrices and actual DFTB H′/S matrices.
Include clustered/near-degenerate spectra, because a chemically realistic
matrix is often easier than an adversarial eigensolver test.

### 5.4 Eigenbasis warm-start

This is promising but optional. The relevant benchmark is total SCC iteration
time, not Jacobi sweep count alone. Two extra GEMMs are worthwhile only if the
Jacobi saving is larger.

### 5.5 SCC convergence near proton-transfer/zwitterionic configurations

Use the smallest robust hierarchy in §4.9 first. Diagnose whether failures are
true fixed-point instability, occupation flipping, insufficient precision, or
simply bad initial charges before adding more algorithms.

### 5.6 Analytic-force derivative correctness

The analytic H/S derivative helper must first reproduce the existing
finite-difference derivative path and the validated Fortran total force.
Check:

- random bond orientation;
- short/long interpolation ranges;
- homo/heteronuclear pairs;
- s–s, s–p, p–s, p–p blocks;
- unit conversion;
- Newton's third law.

### 5.7 Constrained relaxation

Compare full Cartesian freezing with the intended scalar reaction-coordinate
constraint on a few reference points. Verify that the scalar constraint allows
physically sensible transverse proton and donor/acceptor relaxation.

### 5.8 CDFT

Deferred. The future fragment charge constraint should reuse the device-side
SCC/constraint infrastructure, but it must not complicate the present
eigensolver/force task.

---

## 6. Contracts and Tests

### 6.1 Contract: tiled Jacobi mathematical correctness

Test standalone symmetric eigensolves before integrating SCC.

Dimensions:

`N = {63, 64, 65, 87, 88, 96, 97, 128}`.

Matrices:

- random symmetric with controlled spectrum;
- clustered/near-degenerate spectrum;
- actual DFTB orthogonalized H′;
- actual overlap S where relevant.

Reference: CPU LAPACK/nalgebra double precision.

Required diagnostics:

\[
r = \frac{\|AV-V\Lambda\|_F}{\|A\|_F},
\qquad
o = \frac{\|V^TV-I\|_F}{N}.
\]

Initial f32 targets:

- `r < 1e-5`;
- `o < 1e-5`;
- eigenvalue error compatible with SCC energy target;
- no dummy/tail-block state in physical output;
- no NaN/Inf.

Do not silently pass because the diagonal looks plausible.

### 6.2 Contract: N>64 GPU SCC parity on identical geometries

System: AT/GC/azaindole geometries with N>64.

Reference: CPU SCC on exactly the same coordinates.

Compare:

- total energy absolute error;
- Mulliken charges;
- eigenvalues;
- SCC residual/convergence status.

Starting tolerances:

- `|dE| < 1e-4 Ha`;
- `max|dq| < 1e-3 e`;
- `max|dε| < 1e-4 Ha`.

These are starting acceptance limits, not permission to stop improving if the
N≤64 path demonstrates much tighter attainable parity.

### 6.3 Contract: relative-energy/PES parity on fixed geometries

For a set of *identical* scan geometries evaluated by CPU and GPU:

\[
\Delta E_i^{CPU}=E_i^{CPU}-E_{ref}^{CPU},
\]

\[
\Delta E_i^{GPU}=E_i^{GPU}-E_{ref}^{GPU},
\]

\[
\epsilon_i=\Delta E_i^{GPU}-\Delta E_i^{CPU}.
\]

Report:

- absolute energy offset at the reference geometry;
- RMS and maximum `|ε_i|`;
- nearest-neighbor energy-increment error along both scan axes;
- barrier-height difference using the same path/grid definition.

Relative energy is the primary chemistry metric because a constant offset
does not distort a PES. Nevertheless, a large absolute error is still a bug
signal; do not use “offset cancellation” to excuse uncontrolled errors.

### 6.4 Contract: analytic derivative and GPU force parity

Tests proceed in layers.

**A. CPU analytic SK derivative validation**

Compare `rotate_block_with_derivs_into` / `eval_with_deriv_into` to central
finite differences of the **same** B-spline V. Tight f64 agreement is
expected in the valid domain. Measured: H–H analytic dHss/dr vs FD max rel
`8.6e-10`; synthetic table `1.5e-8`. Do not compare to Neville-tail values
past last grid — that tail is the bug we removed.

**B. CPU analytic total force**

CPU `compute_scc_forces` vs FD of `E_el+E_rep`. Measured H2O (h=1e-3 Å):
rel `1.05e-5`. At h=1e-2 Å even f64 F-vs-FD is `~1e-3` (O(h²) truncation) —
do not treat that stencil as a missing-term diagnosis.

**C. GPU analytic total force**

GPU f32 vs CPU f64 on the same SCC density. Measured H2O four kernels + total:
rel `~3e-5`. Energy-surface check at h=1e-2 Å is GPU FD vs CPU FD (rel
`2.2e-4`), not GPU F vs GPU FD (truncation-limited at `~1e-3` even on CPU).

Reference the validated CPU analytic force on exactly the same SCC solution
and geometry.

Starting GPU target:

- `max|dF| < 1e-3 Ha/Å`, with per-component diagnostics.

Also test:

- total-force sum / Newton's third law;
- translation invariance;
- finite values;
- 5–10 independent GPU finite-difference energy derivatives as a debugging
  cross-check, not a production method.

### 6.5 Contract: constrained batched relaxation

For each system report:

- projected/tangential max force;
- reaction-coordinate constraint residual;
- energy;
- SCC iterations;
- FIRE step count;
- explicit convergence state.

A constrained system is converged when both the projected force and constraint
residual satisfy tolerance. The raw force may contain a nonzero constraint
reaction component.

Compare GPU and CPU/reference relaxation by final constrained relative energy
and force/constraint residual. Do not require identical Cartesian coordinates
when multiple nearby minima exist.

### 6.6 Contract: relaxed 2D PES

First compute a coarse 11×11 surface, later 21×21.

For low-level numerical parity, evaluate CPU and GPU on the **same saved
geometries** (§6.3). Separately compare relaxed surfaces:

- relative energy landscape;
- synchronous/asynchronous minimum-energy paths;
- barrier heights;
- intermediate minima;
- projected-force and constraint convergence at every point.

This separation prevents optimizer trajectory differences from being mistaken
for electronic-structure errors.

### 6.7 Performance contracts

Use profiling events, not forced queue synchronization.

Record:

- kernel GPU time;
- host wall time;
- number/bytes of host-device transfers;
- number of blocking synchronization points;
- Jacobi sweeps;
- SCC iterations;
- systems/s.

Benchmarks must include both the cleaned N≤64 baseline and N>64 systems.

### 6.8 Fail-loud invariants

Always assert/report:

- all energies, charges, forces, matrix diagnostics finite;
- no silent CPU fallback;
- no physical occupation of dummy pivot lanes;
- overlap eigenvalues for the physical S remain positive above a documented
  threshold;
- constraint residual finite and bounded;
- explicit SCC/geometry convergence status;
- exhausted iteration limit is failure/best-effort status, never “converged”.

Do **not** require monotonic SCC residual or monotonic FIRE energy/force.

---

## 7. Implementation Plan

### Phase 0 — Clean the GPU harness and establish a truthful baseline

- [ ] Remove unnecessary `finish()` calls.
- [ ] Device-to-device copy for S working buffer.
- [ ] Persistent `GpuSccPlan` with reusable scratch buffers/kernels.
- [ ] Real OpenCL device capability queries.
- [ ] GPU eigenvalue extraction/occupation handling.
- [ ] GPU DIIS or equivalent removal of the per-iteration CPU charge roundtrip.
- [ ] Event-based profiling.
- [ ] Re-run formic-dimer N≤64 benchmarks and store the clean baseline.

**Do not start Jacobi tuning before this phase is complete.**

### Phase 1 — General tiled GEMM

- [ ] Integrate tiled GEMM into the shared runtime.
- [ ] Verify N=65/87/128 parity.
- [ ] Use general GEMM for N>64 Löwdin transform, back-transform, and S⁻¹/²
      reconstruction.
- [ ] Retain full-local GEMM for N≤64 only if measured faster.

### Phase 2 — One-WG tiled/block Jacobi

- [ ] Implement physical N×N global backing with partial final block.
- [ ] No global N→multiple-of-B zero padding.
- [ ] Local `2B×2B` compound pivot with dummy lanes only inside local memory.
- [ ] Local Brent–Luk pivot diagonalization.
- [ ] Tiled A strip update + symmetric transpose store.
- [ ] Tiled V strip update.
- [ ] Physical off-diagonal convergence reduction inside the same kernel.
- [ ] One kernel launch per batched eigensolve; one WG per system.
- [ ] Benchmark B={24,32}, WG={256,512}, then choose one default.
- [ ] Pass standalone residual/orthogonality tests before SCC integration.

### Phase 3 — Complete N>64 SCC

- [ ] Use tiled Jacobi for both H′ and overlap eigensolves.
- [ ] General N>64 S⁻¹/² reconstruction.
- [ ] Device occupation/index selection.
- [ ] Build P and W on GPU (W required later by forces).
- [~] AT/GC/azaindole SCC: H/S matches; GPU charge-rms plateaus `~1e-5`. **Hypothesis: f32 floor** (manifest §3.0.1), not a proven mixer bug. Do not chase `<1e-6` until `max|H|` scale is checked.
- [ ] Fixed-geometry relative PES parity tests.

Optional only after the baseline works:

- [ ] benchmark eigenbasis warm-start; keep only if total SCC time improves.

### Phase 4 — Make the force path genuinely analytic, then port it

- [~] CPU analytic H/S derivatives vs FD of the same B-spline — pass (`gpu_hbond_physics.rs`, H–H rel `8.6e-10`). Remaining: replace blunt zero-sample pad with fitted extra controls (`sk_interpolation.md`).
- [~] CPU total analytic forces vs FD of energy — H2O rel `1.05e-5`. Fortran force parity (`parity_forces.rs`) is older; re-check after the extra-control fitter, not by restoring Neville.
- [~] Port to OpenCL — H2O four components vs CPU rel `~3e-5`. `vload2` fixed the 1×4 NVIDIA crash. Gamma' and repulsive' are in the same test.
- [~] Pairwise on-the-fly P/W contraction — H2O full-chain GPU P then forces rel `3.4e-5`.
- [~] GPU-vs-CPU force parity + Newton — H2O pass. AT/GC forces not yet, because GPU SCC rms plateaus `~1e-5` (f32-floor hypothesis, §3.0.1).

### Phase 5 — Device-resident constrained FIRE relaxation

- [ ] GPU FIRE state and update.
- [ ] Scalar proton-transfer reaction-coordinate constraints.
- [ ] Constraint Jacobian, projected force/velocity, and position correction.
- [ ] Convergence on projected force + constraint residual.
- [ ] Active mask first; add compaction only if profiling shows it is worthwhile.
- [ ] Robust GPU DIIS safeguard/adaptive damping for difficult points.

### Phase 6 — Production proton/Kekulé screening

- [ ] Generate/import AT and GC scan geometries from SPAMMM.
- [ ] Coarse 11×11 relaxed 2D surfaces.
- [ ] Verify convergence and relative-energy parity.
- [ ] Full 21×21 surfaces where scientifically useful.
- [ ] Identify synchronous/asynchronous paths and intermediates.
- [ ] Extend to additional H-bond/Kekulé-coupled systems.

### Optional optimization phase — only if profiling justifies it

- eigenbasis warm-start;
- sophisticated S⁻¹/² reuse/refinement;
- active-batch compaction;
- short cross-scan continuation chains;
- additional SCC mixers.

These are not prerequisites for the first production result.

### Future — CDFT

- fragment charge constraints;
- Lagrange multiplier optimization;
- separate proton/electron transfer coordinates;
- constrained relaxation.

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
| `rust_dftb/src/qmqm/gpu_scc.rs` | exists, cleanup + extend for N>64 | this task |
| `rust_dftb/src/qmqm/gpu_eigen.rs` | exists, cleanup + add tiled block Jacobi | this task |
| `rust_dftb/src/qmqm/gpu_matrix.rs` | exists, cleanup + integrate tiled GEMM | this task |
| `rust_dftb/src/qmqm/gpu_runtime.rs` | exists, real capability queries + persistent plan | this task |
| `rust_dftb/src/qmqm/gpu_scc_plan.rs` | new (persistent kernels + scratch buffers) | this task |
| `rust_dftb/src/qmqm/gpu_tiled_jacobi.cl` | new (one-WG tiled block Jacobi kernel) | this task |
| `rust_dftb/src/methods/dftb/interpolation.rs` | production B-spline + stopgap right pad; extra-control fitter not done | this task (dense) |
| `rust_dftb/src/methods/dftb/spline_resample.rs` | `fit_bspline_controls_zero_end`, `bspline3_eval_v_d1_d2` | this task (dense) |
| `rust_dftb/src/methods/dftb/forces.rs` | CPU `compute_scc_forces`; analytic SK V' via B-spline | this task |
| `rust_dftb/tests/gpu_hbond_physics.rs` | honest dense H-bond physics tests (H/S, tail, forces, energy-gradient) | this task |
| `rust_dftb/src/methods/dftb/rotation.rs` | analytic H/S block derivatives exist; validate against finite differences | this task |
| `rust_dftb/src/qmqm/gpu_forces.rs` | new (analytic GPU forces, pairwise contraction) | this task |
| `rust_dftb/src/qmqm/gpu_forces.cl` | new (force kernels) | this task |
| `rust_dftb/src/qmqm/gpu_relax.rs` | new (GPU FIRE, device-resident) | this task |
| `rust_dftb/src/qmqm/gpu_relax.cl` | new (FIRE + constraint projection kernel) | this task |
| `rust_dftb/src/qmqm/gpu_occupation.rs` | new (device-side occupation handling) | this task |
| `rust_dftb/tests/gpu_scc_n64plus.rs` | new | this task |
| `rust_dftb/tests/gpu_forces.rs` | new | this task |
| `rust_dftb/tests/gpu_relax.rs` | new | this task |
| `rust_dftb/tests/gpu_relaxed_pes.rs` | new | this task |
| `rust_dftb/tests/gpu_perf.rs` | new | this task |
| `rust_dftb/tests/gpu_tiled_jacobi_bench.rs` | new | this task |
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
- `doc/prokop/topical_audit/sk_interpolation.md` — **SSOT for SK spline BCs.**
  What was done (blunt zero samples + left phantom) vs what to do (fitted extra
  controls). Do not re-Neville.
- `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.report.md` —
  Phase 4 hand-off; 1×4 crash is **stale** (fixed by `vload2`). Interpolator
  stopgap is in the 2026-09-09 addendum.

---

## 11. GPT 5.6 Review — Required Corrections (commit b269ab63)

Review of commit `b269ab63` by GPT 5.6 (see `/doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.chat.md` lines 2093–2321).
Priority order as given. All items currently **unfixed** unless marked `[*]`.

**These items are the concrete manifestation of the performance mandate
(§1.1).** Each one is a case where "simple and easy" was chosen over "fast and
correct", and the cost was paid in physics (R1, R4, R5, R6, R7), throughput
(R2, R8, R9, R10, R13), or both (R3, R11, R12). The fix for every item is the
fast-and-correct version, not a slower workaround. Read each checkbox as:
"the production path must do this the high-performance way, not the easy way."

### P0 — Mathematical correctness

- [*] **R1. Tiled Jacobi: do not zero residual off-diagonal of pivot.**
      `gpu_tiled_jacobi.cl:226` stores `(i==j ? lA[i,i] : 0)` after local
      diagonalization, throwing away residual `R` in `P' = U^T P U = D + R`.
      This breaks the invariant `A_k = V_k^T A_0 V_k`. Store the actual `lA[i,j]`,
      not just the diagonal. Return eigenvalues separately if a clean diagonal
      is needed for output.
      **FIXED:** Store actual `lA[i*PLD+j]` for all active (i,j). Final zeroing
      of global A off-diagonal only happens after convergence is established.
- [*] **R2. Tiled Jacobi: replace serial inner pivot with Brent–Luk parallel.**
      The inner pivot solver does `for p { for q { ... } }` with thread 0
      computing one rotation at a time — 75% of WG=256 idle, ~20k scalar
      rotations/pivot, ~60k barriers/pivot. Reuse the existing Brent–Luk
      algorithm from `gpu_eigen.cl` (32 independent pairs/round × 8
      threads/pair = 256). Schedule only active lanes for tail blocks.
      **FIXED:** Adapted Brent–Luk from `gpu_eigen.cl`: JPAIR_INNER=32 pairs,
      PPG_INNER=8 threads/pair, block update of JPAIR² 2×2 blocks. All 256
      threads active. Rotation params and block update computed in f64 for
      precision (f32 storage). Residual improved 30×, eigenvalue parity 50×.
- [*] **R3. Tiled Jacobi: add `CLK_GLOBAL_MEM_FENCE` at dependency boundaries.**
      All barriers were `CLK_LOCAL_MEM_FENCE` only. Global A/V writes during
      one pivot are read by the next pivot and by the convergence scan.
      Use `barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)` before
      next-pivot reads and after global V initialization.
      **FIXED:** Added `CLK_GLOBAL_MEM_FENCE` after V init, after principal
      block store (before strip reads), and after all strip/V updates (before
      next pivot and convergence scan).
- [*] **R4. Restore strict numerical contracts in tests.**
      `gpu_tiled_jacobi.rs`: assertions relaxed to 1e-3 / 5e-3 (header says
      1e-5). `gpu_scc.rs` N>64 test: 1e-2 Ha = 0.27 eV — proves only that the
      program doesn't explode. Restore: residual < 1e-5, orthogonality < 1e-5,
      eigenvalue parity < 1e-4. N>64 SCC: |dE| < 1e-4, |dq| < 1e-3, |dε| < 1e-4.
      Fix the solver, don't loosen the contract.
      **FIXED:** Tiled Jacobi tests now assert residual < 1e-5, orthogonality
      < 1e-5, eigenvalue parity < 1e-4. All pass with margin:
      residual ~1e-6, orthogonality ~2e-7, eigenvalue parity ~6e-5 (worst N=128).
      N>64 SCC tolerances remain pending (R4b).
- [~] **R5. SCC energy must include repulsive spline.**
      `gpu_scc_plan.rs::compute_energy` returns only `Tr(D·H0) + 0.5·Δq·V`.
      The DFTB total energy is `E_DFTB = E_el + E_rep`. Without `E_rep` the
      proton-transfer PES is quantitatively/qualitatively wrong while CPU/GPU
      parity looks perfect. Add repulsive spline evaluation to the GPU energy.
      **PARTIALLY FIXED:** Added `repulsive_energy_batched` GPU kernel and
      `GpuSccPlan::set_repulsive_splines` method. The kernel evaluates the
      full spline (exponential head + cubic intervals + polynomial tail) per
      atom pair. `compute_energy` now adds E_rep when spline data is set.
      The kernel is built and the Rust harness is wired, but no test yet
      feeds actual spline data. Remaining: write a test that uploads real
      SK-file spline data and verifies E_rep parity vs CPU.
- [~] **R6. Force kernel: port all four force components, not just non-SCC.**
      `gpu_forces.cl` computes only `P·dH0 - W·dS`. Production forces must
      match the CPU decomposition: (a) non-SCC electronic, (b) SCC shift
      `0.5·(V_A+V_B)·P·dS`, (c) gamma derivative `Δq_A·Δq_B·γ'(R)`, (d)
      repulsive spline derivative. Port the exact tested CPU formulas — do
      not rederive in OpenCL.
      **PARTIALLY FIXED:** All four force kernels now implemented in
      `gpu_forces.cl`:
      (a) `force_pairs` — non-SCC electronic (existing, unchanged).
      (b) `force_pairs_scc_shift` — SCC shift force, reuses dS/dR
          infrastructure, takes `v_shift` per atom. Rust driver:
          `gpu_scc_shift_force_batched`.
      (c) `force_gamma_deriv_batched` — gamma derivative force with full
          `gamma_prime_full_f32` (same-U and different-U paths matching
          CPU `gamma_prime_full`). Rust driver:
          `gpu_gamma_deriv_force_batched`.
      (d) `force_repulsive_batched` — repulsive spline derivative (exp head
          + cubic intervals + polynomial tail), same spline layout as
          `repulsive_energy_batched`. Rust driver:
          `gpu_repulsive_force_batched`.
      All formulas ported from CPU `forces.rs` with exact constants
      (TAU_FACTOR=3.2, SAME_U_C0/C1/C2, ANG2BOHR). Existing 5 force tests
      still pass (component a only). Remaining: write parity tests for
      components (b), (c), (d) vs CPU reference.
- [~] **R7. SCC consistency: energy/forces must use the same charge state.**
      `scc_step` builds H/C/D from `q_n`, then overwrites `q_gpu` with mixed
      `q_{n+1}`. If residual passes, `compute_energy` uses D from `q_n` but
      recomputes Δq/V from `q_{n+1}`. For analytic forces this inconsistency
      destroys energy-gradient parity. Fix: either accept state as `q_n`
      consistently, or do one final unmixed electronic solve after convergence
      so C/P/W/E/forces all correspond to the same q.
      **PARTIALLY FIXED:** Added `GpuSccPlan::finalize()` — does one unmixed
      electronic solve (steps 1-8: Δq→V→H_scc→H'→Jacobi→occ→C→D) with the
      current `q_gpu`, so D, C, H_scc, V, Δq all correspond to the same
      charge state. `compute_energy` now calls `finalize` first, then
      computes `Tr(D·H0) + 0.5·Δq·V` from the finalized state. Signature
      changed to include `s_buf`, `orb_atom_buf`, `n_occ`. Remaining: force
      path must also call `finalize` before computing forces.

### P0/P1 — Harness and architecture

- [*] **R8. `GpuSccPlan` must genuinely own persistent kernels and buffers.**
      The struct claims pre-built kernels but contains no `Kernel` objects.
      `scc_step()` still calls wrappers that do `Kernel::builder()` per call.
      `set_geometry()` allocates new N² buffers. Build kernels once, store
      them, swap buffer args via `set_arg`. Preallocate `S_work`, `S_vec`,
      `Vscaled`, `lambda`, `X` and overwrite in place.
      **FIXED:** `GpuSccPlan` now owns 13 pre-built `Kernel` objects:
      `k_delta_q`, `k_gamma`, `k_h_scc`, `k_matmul_xh`, `k_matmul_tx`,
      `k_matmul_xc`, `k_jacobi`, `k_extract_diag`, `k_density`, `k_mulliken`,
      `k_residual_mix`, `k_frobenius_trace`, `k_dot`. All built once in `new()`
      with the plan's own scratch buffers as initial args. `scc_step()` and
      `compute_energy()` use `Kernel::set_arg(idx, buf)` + `enq()` — no
      `Kernel::builder()` in the hot loop. Matmul kernels dispatch to
      full-local (N≤64) or tiled (N>64) at construction; buffer arg indices
      tracked via `matmul_buf_base`. All 5 SCC tests, 3 tiled Jacobi tests,
      and 5 force tests pass.
- [~] **R9. Move occupation selection and DIIS fully onto GPU.**
      Eigenvalue extraction does blocking read, CPU sort, upload occ_mask
      every iteration. DIIS reads `q_new`, allocates Vecs, CPU DIIS, uploads q.
      For N≈87, one WG/system can bitonic-sort 128 padded (ε,index) pairs in
      local memory. DIIS: one WG/system, parallel residual dots, one lane
      solves the 6–10-dim DIIS equation. Reduce SCC convergence to one global
      `max_rms` scalar if host must inspect.
      **PARTIALLY FIXED:** `GpuSccPlan` now uses `select_occupation_batched`
      GPU kernel — bitonic sort of (eigenvalue, index) pairs in local memory
      + occupation marking, all on device. No CPU read/sort/upload roundtrip
      in the SCC hot loop. `OCC_MAX_N` specialized per plan. Remaining: GPU
      DIIS (currently simple mixing on GPU; DIIS path in `gpu_scc.rs` still
      uses CPU DIIS with host roundtrip).
- [~] **R10. Force driver: eliminate separate context, per-bucket uploads, `finish()`.**
      `GpuForceDriver` creates its own Context/Queue/Program, takes DM/EDM as
      host slices, uploads them, uploads fragments per bucket, builds kernel
      per bucket, calls `finish()` per bucket, downloads forces. Must share
      one `GpuRuntime` with SCC. P/W are already GPU buffers — no CPU copy.
      Forces stay on GPU for FIRE.
      **PARTIALLY FIXED:** `GpuForceDriver` now uses the shared `GpuRuntime`
      (no separate Context/Queue/Program). Uses `rt.build_program()` for
      program caching. Eliminated per-bucket `finish()` calls — in-order
      queue preserves command order. Only one `finish()` at the end before
      reading forces. Fragments/DM/EDM uploaded once per call (not per-bucket).
      Added `gpu_force_batched_dev` for GPU-resident DM/EDM (no host roundtrip).
      Remaining: SK tables and pair data still uploaded per-bucket (small,
      batch structure may change); pre-built kernel per-bucket still uses
      `Kernel::builder()` (program is cached, but kernel object is rebuilt).
- [ ] **R11. Force kernel: fix `__local Fragment l_frags[128]` batch>128 bug.**
      Declares 128 fragments, loads only first 128, accesses `l_frags[p.replica]`.
      Intended workload includes batches of 200/500/1000. Do not make array
      1024 long — read `fragments[p.replica]` from global/L2, or redesign
      around one WG/system.
- [ ] **R12. Force kernel: replace atomic accumulation with deterministic reduction.**
      Six CAS-loop float atomics per pair create contention and
      nondeterministic summation order. Better: (a) one WG/system with local
      `float3 F[Na]` and deterministic reduction, or (b) kernel 1 writes
      per-pair `float3`, kernel 2 gathers per-atom. Benchmark rather than
      assume atomics are cheap.
- [ ] **R13. Geometry must be device-resident for relaxation.**
      `GpuPairEntry` stores `r,l,m,n` computed on CPU. A future FIRE step
      would do `R_GPU → R_CPU → rebuild pairs → GPU assembly → H/S_CPU → GPU SCC`.
      Store static `(atom_i, atom_j, species, orb offsets)` topology once.
      Compute `ΔR, r, R̂` in the H/S and force kernels from device positions.
      Rebuild gamma on GPU per relaxation step. Only ~435 pairs for 30 atoms.

### P1 — Numerical and testing quality

- [ ] **R14. Fix RMS norm: divide by `sqrt(n_atoms)`.**
      `residual_and_mix_batched` computes `sqrt(Σ r_A²)` (L2 norm), but CPU
      DIIS uses `sqrt(Σ r_A² / N_A)` (RMS). Same `tol` means different things
      for H2O and AT. Divide by `sqrt(n_atoms)` in the GPU kernel or call
      it L2 everywhere. Prefer RMS to match CPU.
- [ ] **R15. GEMM tests: use meaningful tolerances, add timing.**
      `gpu_tiled_gemm.rs` uses `tol = N² * 1e-5`; at N=128 permits max element
      error ~0.164. Test relative Frobenius/max errors, all transpose
      combinations used by Löwdin/S⁻¹/², physical H/S-sized matrices. Add
      OpenCL event-timestamp benchmarks.

### Additional optimizations (after correctness)

- [ ] **R16. `scale_eigenvectors_batched`: precompute `rsqrt(λ_k)` once per system.**
      Currently recomputes `rsqrt(lambda_k)` for every matrix row (~N² square
      roots). Precompute `rlam[k]` once.
- [ ] **R17. Do not hide invalid overlap with `rsqrt(max(λ, 1e-7))`.**
      `LAMBDA_FLOOR = 1e-7` silently clips negative/zero overlap eigenvalues.
      A negative physical S eigenvalue is an error. Report `λ_min`,
      `λ_min/λ_max`, and fail on ill-conditioned/non-positive S. The plan
      already computes `lambda_min` but throws it away.
- [ ] **R18. Fuse `Δq → V=γΔq → H_SCC` into one WG/system kernel.**
      Three kernel launches + intermediate global traffic → one kernel:
      load q/q0, compute Δq and V into local memory, barrier, assemble H.
- [ ] **R19. Build W only after final converged electronic solve.**
      W is unnecessary during ordinary SCC iterations. Build it once after
      convergence alongside the final P.
- [ ] **R20. Next validation system: real 7-azaindole/AT at N≈84–87, not 12×H2O.**
      The 72-orbital water cluster is a useful N>64 smoke test but does not
      probe spectrum, overlap conditioning, charge redistribution, or
      near-zwitterionic SCC behavior that motivated this solver.

### Summary of positive findings (GPT 5.6)

- Analytic SK derivative mathematics looks healthy after orientation, grid
  origin, and p–s radial unit fixes. H2O/formic parity ~1e-6–1e-5 is strong
  evidence the derivative formulas are basically correct.
- Ordered heteronuclear SK handling, one-based grid origin correction, and
  `1/dr` in p–s derivative were exactly the right subtle fixes.
- Partial-tail-block concept in tiled Jacobi is correct: N=87 stays N=87
  globally, final 23-orbital block treated locally, no singular 96-dim S.
