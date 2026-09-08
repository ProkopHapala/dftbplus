# Task 2: Sparse GPU DFTB Forces and Vibrations of Si/Diamond Nanocrystals

**Created:** 2026-09-07  
**Revision:** v3 — scientific/numerical review of agent-produced v2  
**Status:** implementation manifest / source of truth  
**Owner:** prokop / coding agent

> **Purpose of this revision.** The v2 manifest absorbed most of the previous
> critique, but it still showed a recurring failure mode of a coding agent:
> copying a locally correct statement and turning it into a universal rule.
> This v3 therefore distinguishes **mathematical invariants**, **safe baseline
> choices**, and **performance hypotheses that must be benchmarked**.
>
> Important corrections relative to v2 include:
> - the cubic B-spline example contained an actual code bug (`B2` missed the
>   `*t` term); this revision adds algebraic basis tests so such a transcription
>   error cannot survive;
> - `R_HS < R_K < R_Z` is **not a theorem**. `R_K` and `R_Z` are independent
>   convergence parameters controlled by different physics/numerics;
> - `|F_sparse32-F_dense64|` is **total force error**, not automatically a
>   stochastic/noise floor. Smooth bias and non-smooth solver noise must be
>   measured separately;
> - direct SCC `gamma*dq` is intentionally `O(N_atom^2)` unless we later add a
>   long-range accelerator. Therefore the honest claim is sparse/near-linear
>   **electronic matrix algebra**, not automatically linear total SCC;
> - the current device TC2 branch already consumes `Tr(KS)` on the GPU. The
>   optimization target is to eliminate the unnecessary **host scalar read / queue
>   synchronization** from the hot loop, not to move a branch that is already
>   device-side;
> - the SKF repulsive spline and the electronic H/S interpolation are different
>   objects. Preserve supplied repulsive polynomial coefficients; do not refit
>   them merely to force everything through the same B-spline code;
> - rigid-mode diagnostics must distinguish raw Cartesian and mass-weighted
>   coordinates, and linear systems have five rather than six independent rigid
>   modes;
> - uniform nonsingular BSR4 padding for H is promoted from a throw-away
>   prototype to the **baseline production representation**. Variable-size
>   blocks are only justified if profiling proves padding materially expensive.

---

## 1. Goal and scope

Build a **genuinely sparse, GPU-resident DFTB SCC + force path** for insulating
Si/diamond nanocrystals, then use it to optimize structures and compute
finite-difference force Hessians for roughly 300–1000 atoms.

Target accuracy is pragmatic rather than formal:

- ordinary vibrational frequencies: approximately **5%** is acceptable;
- low-frequency modes: compare with an **absolute** tolerance, not percent;
- tiny negative frequencies associated with imperfect rigid modes are allowed;
- significant localized imaginary modes must be diagnosed, not silently
  projected away;
- expensive sparse matrix algebra should remain f32 unless a measured numerical
  problem justifies a more expensive accumulation scheme.

### 1.1 What “linear scaling” means here

Do not use the phrase carelessly.

The target is near-linear scaling of the **short-range sparse electronic matrix
part** at fixed physical accuracy:

- H/S assembly: `O(N)` at fixed neighbor count;
- sparse approximate inverse / purification: target `O(N)` if the required mask
  radii and iteration counts saturate with system size;
- short-range analytic force contractions: `O(N)`.

But the complete workflow is not mathematically linear:

- direct SCC Coulomb/gamma matvec is currently `O(N_atom^2)`;
- a complete `3N x 3N` Hessian contains `O(N^2)` numbers and needs `6N` force
  evaluations with a central difference;
- therefore a full Hessian is `O(N^2)` only if each force call is `O(N)`, and is
  asymptotically worse if an `O(N^2)` SCC term dominates;
- final dense diagonalization is formally `O(N^3)` in the Hessian dimension,
  although `3000 x 3000` is still acceptable for the present N <= 1000 target.

**Report these pieces separately.** Do not fit one exponent to the entire
pipeline and call it “linear DFTB”.

### 1.2 Decisive numerical question

The central question is not simply “is f32 accurate enough?” but:

> **Is the map R -> F(R) smooth, deterministic enough, and consistent with the
> same approximate energy/model used for optimization?**

A smooth f32 force can give a useful Hessian. A discontinuous f64 force caused
by mask changes, spline-knot derivative jumps, inconsistent SCC convergence, or
nested finite differences cannot.

### 1.3 Non-negotiable contracts

Production sparse SCC/force code must satisfy all of the following:

1. **No diagonalization in the sparse production path.** Diagonalization is
   allowed only in dense/reference tests and the final nuclear Hessian solve.
2. **No numerical finite difference inside the force.** H/S radial and angular
   derivatives are analytic. The only production finite difference is
   `force -> nuclear Hessian`.
3. **No dense orbital matrix hidden in the sparse route.** A dense `Norb x Norb`
   allocation, matmul, inverse, or eigenproblem is a hard failure.
4. **No matrix host round-trip in iterative SCC/purification.** Small diagnostic
   scalars may be read only at controlled intervals.
5. **Frozen sparse topology during the final optimization/Hessian.** Numerical
   values may become zero; CSR structure must not jitter with geometry.
6. **Every approximation has its own convergence parameter and diagnostic.** In
   particular `R_K`, `R_Z`, SCC tolerance, arithmetic mode, and Hessian `h` are
   not to be collapsed into one vague “precision” setting.

---

## 2. Physics and numerical invariants

### 2.1 Density-kernel locality

For an insulating, properly passivated nanocrystal the density matrix should
decay approximately exponentially in real space. This is why H-passivated Si
is a favorable target and why bare dangling-bond clusters are a poor long-term
proxy: surface states can shrink the gap and lengthen the density-matrix decay.

However, **do not hard-code an ordering such as `R_HS < R_K < R_Z`.**

Define independent supports:

- `M_HS`: physical H/S derivative support plus any frozen geometric skin;
- `M_K`: density-kernel support;
- `M_Z`: approximate inverse-overlap support;
- `M_T`: exact Boolean product support needed by a particular truncated
  multiplication, e.g. `M_T = M_K o M_HS` for `K*S`;
- `M_val`: larger validation support used only to measure leakage that the
  production mask suppresses.

`R_K` is mainly controlled by electronic locality/gap. `R_Z` is controlled by
the conditioning and locality of `S^{-1}`. Either may need to be the larger one.
Measure them independently.

For a production mask `M_K`, distinguish:

```text
R_in   = || P_MK( K S K - K ) ||
R_leak = || P_(Mval \ MK)( K S K ) ||
```

`R_in` can be tiny merely because the mask projects away the missing product.
`R_leak` is therefore essential.

Also monitor a stationarity/commutator residual using one convention
consistently, for example

```text
R_H = || H K S - S K H ||
```

on a sufficiently wide validation support. A truncated matrix can be nearly
idempotent yet not represent the occupied subspace of the current Hamiltonian.

### 2.2 Density and energy-weighted density without diagonalization

Use the following convention explicitly throughout the code and tests.

For closed-shell DFTB, let

```text
K = C_occ C_occ^T
C_occ^T S C_occ = I
Tr(K S) = N_occ
```

where `N_occ = N_e/2`.

Then

```text
D = 2 K
Tr(D S) = N_e
```

and, from `H C_occ = S C_occ eps_occ`,

```text
K H K = C_occ eps_occ C_occ^T
W = 2 K H_scc K
```

where `W` is the energy-weighted density matrix entering the overlap/Pulay term.
This removes the need to recover occupied eigenvectors/eigenvalues after
purification.

The identity is exact for the converged projector of the matrix algebra being
solved. In the truncated sparse calculation, its physical quality must still be
judged using `R_in`, `R_leak`, `R_H`, trace, force parity, and radius sweeps.
Do not write “exact” without that qualification.

Only `W_ij` where `dS_ij/dR != 0` is needed by the force, so the final EDM output
can be projected directly to `M_HS`.

### 2.3 Hessian convention

Use

```text
H_{j beta, i alpha} = - dF_{j beta} / dR_{i alpha}
```

and the production three-point central formula

```text
H[:,a] = -(F(R+h_a e_a) - F(R-h_a e_a)) / (2 h_a)
```

The optional five-point **Hessian** reference is

```text
H[:,a] ~= [ F(+2h) - 8 F(+h) + 8 F(-h) - F(-2h) ] / (12 h)
```

(the sign is easy to get wrong because the standard five-point formula gives
`dF/dx`, whereas the Hessian is `-dF/dx`).

Do not chase `h -> 0` in f32. Start the practical sweep at

```text
h = 0.01, 0.02, 0.05, 0.10 Angstrom
```

and add smaller values only as diagnostics. A likely production value for Si is
around 0.05 Angstrom, while H may benefit from 0.02–0.03 Angstrom, but the
plateau test decides.

### 2.4 “Force error” is not the same as “force noise”

Do not define

```text
sigma_F = |F_sparse32 - F_dense64|
```

and then automatically insert it into `sigma_F/h`. That difference contains
smooth systematic approximation error as well as non-smooth numerical error.

Separate at least:

- **bias:** `deltaF_bias = F_sparse - F_dense` at the same geometry;
- **repeatability/history sensitivity:** same geometry solved from different
  valid SCC starts / warm-start histories;
- **convergence sensitivity:** force change when SCC/TC2/inverse tolerances are
  tightened;
- **arithmetic sensitivity:** fast-f32 vs compensated-f32 vs selective-f64;
- **FD truncation:** change with `h` and with three- vs five-point formulas.

Only a non-smooth/repeatability component generically gives a Hessian error
roughly proportional to `1/h`. A smooth force bias differentiates as a smooth
bias and need not blow up in this way.

### 2.5 Rigid modes: raw vs mass-weighted

Keep these objects distinct:

```text
H_raw   : unsymmetrized Cartesian Hessian from FD forces
H_sym   : (H_raw + H_raw^T)/2
H_mw    : M^-1/2 H_sym M^-1/2
H_phys  : rigid-mode-projected mass-weighted Hessian used for spectrum
```

For **raw Cartesian diagnostics**, translations are uniform atomic
displacements and rotations are

```text
dR_i = axis x (R_i - R_COM)
```

without `sqrt(m)` factors.

For projection in **mass-weighted coordinates**, use

```text
q_trans,i = sqrt(m_i) * axis
q_rot,i   = sqrt(m_i) * [axis x (R_i - R_COM)]
```

orthonormalize them, and drop numerically dependent vectors. A nonlinear free
cluster has six rigid modes; a linear molecule has only five. Do not hard-code
six for the Si2 reference test.

Projection is for the final spectrum. Always inspect `H_raw`/`H_sym` first.

---

## 3. Current repository state and blocking gaps

This section is a snapshot to orient implementation. **Verify against current
code before editing; comments in old helper paths may be stale.**

### 3.1 Sparse GPU algebra

Already present:

- BSR4 CSR with symmetric blocks stored in both directions;
- device-resident `GpuBsrStructure`, `GpuBsrMatrix`, and purification workspace;
- masked SpGEMM and symmetric-right SpGEMM;
- Newton-Schulz approximate inverse;
- metric TC2 purification;
- direct identity residual `||ZS-I||` that avoids the earlier cancellation bug;
- sparse Mulliken charge extraction;
- persistent buffers and cached kernel handles in the newer device path.

Still blocking honest sparse performance:

- some convenience/reference helpers densify matrices;
- the production entry path has historically started from full/dense masks and
  assumes four orbitals per atom;
- geometric mask construction is currently all-pairs on CPU;
- product-mask construction does unnecessary search/deduplication work;
- symbolic row intersections are repeated inside every SpGEMM;
- some old/high-level wrappers still allocate kernels/buffers or call `finish()`
  frequently — the device-resident route must be the only production route;
- current TC2 still performs hot-loop scalar host reads for bookkeeping even
  though the **branch itself can consume the device `trace_buf` directly**;
- convergence checks can trigger extra KSK products after the update.

### 3.2 Force path

The dense force code has the correct high-level DFTB decomposition and excellent
reference parity, and analytic Slater–Koster rotation-derivative machinery
exists. However, the fetched top-level force implementation still contains the
DFTB+-style tiny central finite difference of H/S pair blocks.

Treat that as a **reference/test helper only**. Production sparse force must not
reach it.

The strongest regression test is architectural, not grep-only:

```text
production_sparse_force()
    -> analytic_pair_block_with_derivs()
    -> spline V,V' + analytic angular derivatives

reference_pair_fd()    // test module only, not callable from production feature
```

Then compare analytic derivatives against f64 finite differences in tests.

### 3.3 Electronic and repulsive interpolation are separate

For electronic H/S tables:

- `spline_resample.rs` already fits a natural cubic spline in f64 and converts
  it to cubic B-spline control points;
- the GPU Hamiltonian already uses a four-point cubic B-spline stencil;
- therefore most of the C2 solution already exists and should become canonical.

For the **repulsive SKF spline**, preserve the coefficients supplied by the SK
file: cubic polynomial intervals plus the supplied final degree-5 polynomial
segment. Evaluate those coefficients and derivatives analytically. Test their
continuity, but do **not** refit the repulsive potential into the electronic
B-spline representation merely for aesthetic uniformity.

FireCore's quintic B-spline basis is a useful future experiment for a global
quintic representation, but it is **not the same thing** as the six-coefficient
polynomial cutoff segment in an SKF repulsive spline.

---

## 4. Required implementation architecture

### 4.1 P0 — sparse-path firewall and whole-program profiling

P0 is an **audit/instrumentation step**, not a reason to postpone the numerical
force work for days of micro-optimization.

Production sparse SCC/force must fail loudly if it performs any of these:

```text
Norb x Norb dense allocation
orbital dense matmul / inverse / eigenproblem
to_dense() in production SCC/force
CPU sparse matmul in an iteration
matrix K/H/S host transfer in an iteration
CSR/product-plan rebuild inside SCC
numerical finite difference of H/S inside force
silent fallback from sparse to dense
```

Allowed and explicitly reported:

- dense **atomic** `gamma[Natom,Natom]` for N ~ 1000;
- dense reference calculations in tests;
- dense nuclear Hessian and final CPU eigensolve;
- one-time CPU symbolic preprocessing;
- infrequent scalar diagnostics.

Instrument OpenCL events without inserting a `finish()` after every kernel.
Report host synchronization separately because a 4-byte read can cost far more
than its bandwidth suggests.

Suggested statistics:

```rust
struct SparsePerfStats {
    n_atom: usize,
    n_orb_physical: usize,
    n_orb_padded: usize,

    nnz_hs: usize,
    nnz_k: usize,
    nnz_z: usize,
    plan_terms: usize,
    plan_bytes: usize,
    gpu_bytes_peak: usize,

    kernel_launches: usize,
    host_syncs: usize,
    host_read_bytes: usize,

    t_mask_plan: f64,
    t_hs: f64,
    t_gamma_build: f64,
    t_gamma_mv: f64,
    t_inverse: f64,
    t_kinit: f64,
    t_tc2: f64,
    t_scc: f64,
    t_force: f64,
    t_host_sync: f64,
}
```

### 4.2 P1 — one canonical C2 electronic SK spline

The canonical electronic H/S radial function must be at least C2 over the
physically sampled range and cutoff transition.

The representation can be B-spline controls or per-interval polynomial/Hermite
coefficients. What matters is that all coefficients come from **one globally
C2 spline**, not independently estimated endpoint slopes.

Because the code already has the machinery, use f64 CPU preprocessing -> f32
cubic B-spline controls as the baseline.

For uniform grid samples, the interior cardinal cubic controls obey

```text
c[i-1] + 4 c[i] + c[i+1] = 6 f[i]
```

and can be obtained by a tridiagonal f64 solve.

#### Correct GPU evaluator

```c
// Return (V, dV/dr, d2V/dr2) for one uniform cubic B-spline interval.
inline float3 bspline3_v_d1_d2(
    float c0, float c1, float c2, float c3,
    float t, float inv_dr
){
    const float t2 = t*t;
    const float t3 = t2*t;
    const float u  = 1.0f - t;

    const float b0 = (u*u*u)                         * (1.0f/6.0f);
    const float b1 = (3.0f*t3 - 6.0f*t2 + 4.0f)    * (1.0f/6.0f);
    const float b2 = (-3.0f*t3 + 3.0f*t2 + 3.0f*t
                      + 1.0f)                        * (1.0f/6.0f);
    const float b3 = t3                              * (1.0f/6.0f);

    const float d0 = -0.5f*u*u;
    const float d1 =  1.5f*t2 - 2.0f*t;
    const float d2 = -1.5f*t2 + t + 0.5f;
    const float d3 =  0.5f*t2;

    const float dd0 =  1.0f - t;
    const float dd1 =  3.0f*t - 2.0f;
    const float dd2 = -3.0f*t + 1.0f;
    const float dd3 =  t;

    const float v  = fma(c0,b0,fma(c1,b1,fma(c2,b2,c3*b3)));
    const float d  = fma(c0,d0,fma(c1,d1,fma(c2,d2,c3*d3))) * inv_dr;
    const float dd = fma(c0,dd0,fma(c1,dd1,fma(c2,dd2,c3*dd3)))
                   * inv_dr * inv_dr;
    return (float3)(v,d,dd);
}
```

The v2 draft had `+3.0f+1.0f` in `b2`; the comment had the right formula but the
code did not. Add tests that make this class of mistake impossible:

```text
sum_k B_k(t)   = 1
sum_k B'_k(t)  = 0
sum_k B''_k(t) = 0
```

for many random `t`, plus exact/near-exact checks at `t=0,1`.

#### What to compare

Do **two separate tests**:

1. f64 canonical spline vs a high-resolution/reference interpolation of the
   original SK table: model/interpolation error;
2. f32 GPU evaluator vs the **same canonical f64 spline**: arithmetic/GPU error.

A raw table alone does not uniquely define `V'` or `V''`; do not report
“derivative error vs raw data” without specifying the reference interpolant.

#### Cutoff

Do not hard-code “last 0.5–1.0 Angstrom” as a universal join interval.
Preserve the parameter-set semantics and existing intended cutoff, and construct
or retain a transition that is verified numerically to satisfy the desired
continuity. If a quintic transition is generated, constrain
`V,V',V''` at the join and `V=V'=V''=0` at cutoff.

#### GPU table access

Benchmark, do not decree:

```text
A. direct/read-only-cache global loads
B. whole small species-pair table staged in local memory per WG
C. vector-packed/aligned table layout
```

Local staging only makes sense when a workgroup reuses the same species-pair
table enough times to amortize the cooperative load and barrier. If workgroups
mix pair types or do little reuse, hardware read-only cache may win.

### 4.3 P2 — fully analytic electronic force

The production path is:

```text
r, unit vector
    -> C2 radial V(r), V'(r)
    -> analytic derivatives of direction cosines / SK rotation
    -> dH0/dR, dS/dR
    -> density/EDM contractions
```

Never:

```text
H(R+delta) - H(R-delta)
```

inside production force.

Validation order:

1. one pair: analytic `dH,dS` vs tiny **f64 test-only** finite difference;
2. full system: analytic f64 force vs finite difference of total f64 energy;
3. compare to DFTB+ where conventions match;
4. compare GPU f32 analytic force to the same canonical f64 Rust model.

This sequence localizes errors. A full force mismatch should not be debugged by
jumping directly to a nanocrystal Hessian.

### 4.4 P3 — diagonalization-free sparse force: D and W

Use

```text
D = 2 K
W = 2 K H_scc K
```

with the convention in §2.2.

Do not build a dense EDM. The force needs `W` only on `M_HS`:

```text
M_TW = boolean_product(M_K, M_HS)   // exact for the truncated operands
T    = K * H_scc                    // on M_TW
W    = 2 * project_MHS(T * K)       // final short-range support
symmetrize(W)
```

The actual product plan should be generated from the operand/output CSR
structures, not inferred from a simplistic radius formula.

Parity gate: at the same geometry and same Hamiltonian, compare

```text
D_sparse = 2K               vs dense occupied-orbital D
W_sparse = 2K H K           vs dense eigenvalue/eigenvector W
F_sparse                     vs dense analytic force
```

before any Hessian work.

### 4.5 Locality masks: measured design, not hard-coded hierarchy

Perform staged/2D sweeps over `R_K` and `R_Z`.

A practical approach is:

```text
1. make Z generous; sweep R_K
2. fix a converged R_K; sweep R_Z
3. verify a few crossed combinations around the chosen point
```

Record:

- energy error;
- charge error;
- force error;
- `Tr(KS)-Nocc`;
- `R_in`, `R_leak`, `R_H`;
- iteration counts;
- wall time and memory;
- gap/locality diagnostic where a reliable gap estimate is available.

The result we care about is a **plateau of physical observables vs radius**, not
one magic cutoff.

### 4.6 P4 — symbolic SpGEMM plans

The current symmetric-right kernel repeats a sorted-row intersection every time
it computes the same structural product. With frozen masks, precompute those
intersections once.

For each output block `C_ij`, store the exact contributing numerical blocks:

```text
plan_ptr[cb] .. plan_ptr[cb+1]
    -> (local A_ik block index, B_jk block index) pairs
```

For a symmetric right operand, `B_kj = B_jk^T`.

The hot GPU loop then performs only loads + 4x4 FMAs. A compact plan may pack a
small local-A index with a global-B block index, but the packing format is an
implementation detail and must fail loudly on overflow.

**This is a performance hypothesis, not a law.** Record:

```text
number of plan terms
bytes in plan
average terms/output block
SpGEMM time before/after
occupancy/local-memory use
```

If a wide mask makes the plan consume unreasonable memory, keep the
intersection kernel for that case.

#### Degree buckets

Bucket rows by left degree and compile/use appropriate local-memory capacities,
e.g. 32/64/128/256. The exact boundaries are benchmark parameters.

Also benchmark the current 16-lane/4x4-block team against a 4-lane team where
each lane accumulates four output scalars. The latter may reduce redundant B
loads/register traffic on some GPUs; do not assume which wins.

### 4.7 P5 — TC2 hot loop with no unnecessary Q buffer or host sync

Be precise about the current code: the TC2 **update kernel already reads the
one-float `trace_buf` on device**. The expensive mistake is reading that scalar
to the host every iteration for bookkeeping, which forces synchronization.

Target iteration:

```text
A. T = K*S
   + produce trace partials

   device reduction -> trace_buf

B. q = (T*K)_element                         // do not store full Q unless needed
   if trace_buf > Nocc: knew = q
   else                 knew = 2*k - q

   on diagnostic iterations only:
       err = q - k                            // idempotency of OLD K
       accumulate err^2 partials

   symmetrize knew
```

Normal iteration:

```text
no matrix readback
no trace readback
no queue finish for host
swap K <-> Knew
```

Every `check_every` iterations:

```text
reduce idempotency partials on device
read a tiny diagnostic packet (trace, residual, flags)
if residual(old K) < tol:
    return OLD K without swapping to the unnecessary next iterate
else:
    swap and continue
```

This avoids the current pattern of updating K and then performing an additional
KSK solely to know the residual of the new iterate.

Keep a slow/debug mode that records every iteration; it must not be the
production default.

### 4.8 P6 — SCC gamma: cache by geometry, then optimize the reduction

`gamma_ij(R)` depends on geometry/species, not on the SCC charge vector. Do not
recompute `gamma_full(r_ij)` inside every SCC matvec.

For N <= 1000, the simplest robust baseline is a dense f32 atomic gamma matrix:

```text
Gamma[Natom,Natom]  ~ 4 MB for N=1000
```

Per geometry:

```text
build Gamma once
```

Per SCC iteration:

```text
V = Gamma * dq
```

For a Hessian displacement of atom `i`, start from the center matrix and update
only row/column `i` because only distances involving `i` changed.

A one-work-item-per-atom serial loop over all `j` exposes only ~1000 threads and
is not the preferred GPU mapping. Baseline GPU matvec:

```c
// conceptual mapping: one work-group per output atom i
// each lane sums j = lane, lane+WG, ...
// then local pairwise reduction
```

Precision variants to benchmark:

```text
GAMMA_FAST       f32 FMA lane sums + pairwise WG reduction
GAMMA_KAHAN      compensated lane sums + pairwise WG reduction
GAMMA_FP64_TAIL  f32 partials + tiny device-f64 final reduction, if supported
```

Compile compensated kernels without FP reassociation / fast-relaxed-math that
would invalidate Kahan/TwoSum assumptions.

The SCC energy can reuse the converged potential vector according to the code's
established sign convention, rather than recomputing another dense double sum.
Gamma-force terms are evaluated once for the converged SCC state, not once per
SCC iteration.

**Scaling honesty:** this matvec remains `O(N_atom^2)`. It is probably cheap at
N~1000, but measure it. Only if it becomes dominant should we consider
Coulomb/FMM/tree/mesh acceleration.

### 4.9 Precision policy

Bulk f32 remains the baseline:

- H/S/K/Z/W values;
- SpGEMM;
- B-spline evaluation;
- SK rotations;
- Newton-Schulz / TC2 element updates;
- short-range force contractions.

Use more careful accumulation only where measurement justifies it.

Recommended hierarchy:

| Quantity | Baseline | Optional robustness variant |
|---|---|---|
| 4x4 sparse product sum | f32 FMA | compensated experiment only |
| positive norm/residual | pairwise f32 reduction | device-f64 tail |
| `Tr(KS)` | pairwise f32 reduction | compensated / device-f64 tail |
| `Gamma*dq` | pairwise f32 WG reduction | Kahan per lane / f64 tail |
| SCC/total energy scalar | f32 partials | compensated or f64 final |
| atomic force accumulation | f32 / pairwise | compensated if parity demands |
| coordinates / displacement definition | f64 host | — |
| Hessian assembly | f64 CPU | — |
| mass weighting / eigensolve | f64 CPU | — |

Do not move a reduction tail to the **host** inside the hot TC2/SCC loop merely
to get fp64 arithmetic; the synchronization can cost more than the saved
roundoff. Prefer device pairwise/compensated reduction, and use host f64 for
infrequent diagnostics/final assembly.

The earlier identity-residual cancellation bug is the design lesson: first
choose a numerically stable expression; only then add precision if needed.

### 4.10 H passivation: nonsingular padded BSR4 baseline

H has one physical 1s orbital while Si/C use four `sp3` basis functions. Padding
H with three **zero-overlap** dummy orbitals makes S singular and is forbidden.

Baseline uniform BSR4 embedding:

```text
H active orbital: normal physical H/S matrix elements
H dummy orbitals:
    S_dd = 1
    H_dd = E_dummy
    all active-dummy and interatomic dummy couplings = 0 exactly
```

Keep the physical electron count / `Nocc` unchanged. Choose `E_dummy` with a
safety margin above the occupied physical spectrum but **not absurdly high**,
because it enlarges the spectral interval and may slow purification.

Do not silently “ignore” dummy dimensions in all residuals. Instead assert that
they behave as the decoupled unoccupied subspace we intended:

```text
Tr(KS)                    -> physical Nocc
max dummy occupation      < tolerance
total dummy occupation    < tolerance
active Mulliken electron sum has correct physical count
```

If dummy occupation is appreciable, fail and fix bounds/`E_dummy`/purification.

This padded representation is allowed to remain the production solution. A
variable-block 4x4/4x1/1x4/1x1 sparse format is a **future optimization only if
profiling shows the padding cost is material**. Keeping one simple, heavily
optimized BSR4 algebra may be faster overall than adding structural complexity.

### 4.11 Frozen topology and skin

Use two optimization stages.

**Coarse stage:** masks may be rebuilt, but only at explicit checkpoints.
Never change CSR support silently inside one optimizer line search / FIRE step.

**Final stage + entire Hessian:** topology is frozen.

Build structural H/S support with

```text
R_struct = R_physical_cutoff + skin
```

Pairs in the skin exist structurally but their physical H/S value is zero once
they lie beyond the actual SK cutoff. This keeps the numerical potential
smooth while allowing geometry motion.

Freeze:

```text
M_HS, M_K, M_Z
all product masks
symbolic SpGEMM plans
transpose/diagonal maps
row buckets
```

Track displacement relative to the mask-reference geometry. Because two atoms
can move in opposite directions, pair-distance change can exceed one atom's
`dmax`. Choose the skin conservatively to cover the final optimizer motion and
`+-h_max` Hessian displacements. If the skin is exhausted, **abort/rebuild
between stages**, never silently mutate topology during the Hessian.

### 4.12 Hessian-displacement reuse

For the converged central geometry `R0`, cache:

```text
frozen sparse structures + product plans
H0_center, S_center
Gamma_center
q_center
Z_center
safe spectral-bound policy
```

For each coordinate `a=(i,alpha)` and sign, start independently from the center:

```text
R = R0
R[i,alpha] += sign*h_i

1. H0 <- H0_center ; S <- S_center
   patch only physical H/S blocks involving atom i

2. Gamma <- Gamma_center
   patch only row/column i

3. Z <- Z_center
   Newton-Schulz-correct Z against the NEW S

4. q <- q_center
   SCC loop:
       V     = Gamma * dq
       Hscc  = H0 + 0.5*S*(V_i+V_j)

       bounds = CURRENT safe bounds, or a cached conservative envelope that
                has been proven valid and is asserted at runtime

       Kinit = current-H projector initializer(Hscc,S,Z,bounds)
       K     = TC2(Kinit)
       qnew  = Mulliken(K,S)
       q     = mix(q,qnew)
       converge

5. D = 2K
   W = 2 K Hscc K projected to M_HS

6. F = analytic_sparse_force(...)
```

Do not blindly reuse old `K`: a polynomial purification of an old projector
cannot rotate it into the new occupied subspace. A future variational/LNV
continuation step may exploit old K, but only after the baseline is correct.

Do not blindly reuse `bounds0` either. Geometry/SCC shifts perturb the spectrum.
Recompute cheap **sparse** safe bounds, or use a deliberately conservative
cached envelope with verification. Never restore a dense bounds helper.

Always initialize `+h` and `-h` from the same central q/Z state. Do not make the
second side inherit the first side's solver history.

Optional later experiments:

- cache pair derivative blocks at the center and patch those touching the moved
  atom;
- process a small number of independent Hessian displacements concurrently if
  profiling shows one N~300–1000 system does not saturate the GPU.

Neither is required before the baseline is correct.

### 4.13 Hessian inspection and negative-mode diagnosis

Always preserve `H_raw`.

Diagnostics before projection:

```text
eta_asym = ||H_raw-H_raw^T||F / ||H_raw||F
translational sum-rule violation
raw rigid-displacement responses
||H_ij||F versus pair distance
```

Then form `H_sym`, mass-weight, construct the correct rank-5/rank-6 rigid basis,
and project a separate copy/operator.

A significant negative mode is **not automatically a numerical bug**. For every
suspect localized mode, displace the optimized geometry a small amount along
`+-mode` and evaluate energy/forces:

- if energy decreases in one/both directions, the structure is genuinely not a
  minimum -> reoptimize / allow reconstruction;
- if the energy scan is locally convex while the Hessian reports a strong
  negative curvature, investigate force noise, SCC convergence, `h`, masks,
  and interpolation.

This prevents us from “fixing” a real surface reconstruction by projection.

---

## 5. Validation gates

The gates are ordered to answer one scientific question at a time. A gate may
use slow/reference code; production performance work must not obscure a failed
numerical gate.

### Gate A — canonical spline and derivative consistency

Use representative Si-Si, Si-H, C-C, C-H channels.

Require:

- basis identities (`sum B=1`, derivative sums zero);
- f64 canonical spline reproduces the chosen reference interpolation to the
  intended tolerance;
- f32 GPU matches the same f64 canonical spline in `V,V',V''`;
- left/right `V,V',V''` are continuous at every knot to numerical tolerance;
- electronic cutoff transition is C2;
- supplied repulsive spline/polynomial continuity is checked separately.

### Gate B — analytic-force and EDM parity

On small systems already covered by dense tests:

1. pair analytic derivatives vs f64 FD test helper;
2. full analytic f64 force vs f64 total-energy finite difference;
3. compare to DFTB+ where possible;
4. sparse `D=2K`, `W=2KHK` vs dense D/W at the same geometry;
5. sparse f32 analytic force vs dense f64 analytic force.

Hessian implementation is blocked until this passes.

### Gate C — independent locality sweep

Use insulating systems first. Sweep `R_K` and `R_Z` independently/staged.
Record accuracy, residual leakage, stationarity, iterations, memory and timing.

Do **not** use structural `nnz` growth as a metric — the mask fixes structural
`nnz` by construction.

A useful output is a table/heatmap of

```text
(R_K,R_Z) -> force error, energy error, R_leak, R_H, time, memory
```

rather than separate one-dimensional plots that hide coupling.

### Gate D — realistic Si/H basis

Use the nonsingular padded-H BSR4 embedding on a small H-passivated Si cluster.
Require:

- correct electron count;
- negligible dummy occupation;
- dense/sparse energy/charge/force parity;
- locality sweep is at least as well behaved as expected from the passivated
  gap compared with a bare cluster of similar size.

Variable blocks are **not** required to pass Gate D.

### Gate E — determinism, arithmetic sensitivity, and Hessian h plateau

On a fixed near-equilibrium geometry, separate:

**A. same-geometry repeatability/history tests**

```text
cold SCC start
central warm start
several perturbed valid q starts
fast vs tighter SCC/TC2/Z tolerances
fast-f32 vs compensated variants
```

Measure force spread/history sensitivity.

**B. sparse-vs-dense bias**

Measure `F_sparse-F_dense` as a separate quantity.

**C. h sweep**

Start with

```text
0.01, 0.02, 0.05, 0.10 Angstrom
```

and optionally extend downward for diagnosis. Compare raw Hessian asymmetry,
Hessian error vs dense f64 at the same geometry, and 3-point vs 5-point
references. Choose a broad stable plateau, not the smallest h.

### Gate F — geometry optimization at the method's own minimum

Optimize a small H-passivated Si system with the sparse model:

```text
coarse stage: FIRE, masks can rebuild at explicit checkpoints
final stage: frozen masks, FIRE or L-BFGS as appropriate
```

Tie the stopping target to measured convergence/repeatability behavior rather
than a hard-coded f32 folklore number.

For small systems, compute the full Hessian if cheap. For larger systems, a few
finite-difference Hessian-vector probes orthogonal to rigid motion can cheaply
check for obvious unstable directions before a full Hessian.

### Gate G — same-geometry Hessian parity

At **identical frozen coordinates**, compute:

- sparse f32 Hessian;
- dense f64 Rust Hessian using the same canonical DFTB model;
- optional DFTB+ reference.

Compare:

```text
max absolute element error
||DeltaH||F / ||H||F
eta_asym
raw rigid-mode leakage
frequencies
mode overlap / MAC
subspace overlap for near-degenerate groups
```

Use relative frequency error for ordinary modes and absolute error for very low
modes. Do not hard-code one final threshold before Gate E establishes the
actual numerical floor.

### Gate H — spectra at each method's own minimum

Now optimize each method independently and compare spectra at each model's own
minimum.

For any significant imaginary sparse mode absent from the dense reference,
report:

```text
mode localization / atoms involved
raw vs projected eigenvalue
force norm at minimum
SCC/TC2/Z residuals
R_K,R_Z,R_leak,R_H
h
precision mode
energy scan along +-mode
```

### Gate I — scaling and whole-program profile

Use approximately

```text
N ~ 60, 150, 300, 600, 1000, 1600
```

where available.

Fit/report separate scaling for:

- H/S sparse assembly;
- Z / K initialization / TC2 sparse core;
- `Gamma*dq`;
- complete SCC iteration;
- analytic force;
- full Hessian construction;
- final Hessian eigensolve.

Do not assert `alpha<1.5` for **total SCC** while direct gamma remains `O(N^2)`.
The meaningful sparse-core target is near `alpha~1` once launch overhead and
small-N effects are excluded.

### Gate J — production N~300, then 800–1000

Make N~300 the first real success criterion. Only after its optimization and
full Hessian are routine should we move to 800–1000 atoms.

Every production result records:

```text
git commit
SK-set hash
spline representation + node count
physical/padded orbital count
R_HS / skin / R_K / R_Z / validation mask
SCC, Z, TC2 convergence settings
precision mode
optimizer settings
Hessian h by species
GPU/device/driver
complete timing breakdown + host sync count
```

Performance numbers such as “Hessian < 1 h” are **targets to observe and improve,
not correctness gates invented before profiling**.

### Fail-loud invariants

- no NaN/Inf in energies, forces, matrix diagnostics, or Hessian;
- no silent dense fallback;
- no production FD of H/S;
- no hidden dense orbital matrix in sparse SCC/force;
- no matrix host roundtrip in an iteration;
- frozen topology really remains frozen during final optimization/Hessian;
- dummy H orbitals stay unoccupied;
- validation-mask leakage and stationarity residual are recorded;
- every convergence failure reports mask radii, iterations, residuals, and
  precision mode.

---

## 6. Implementation order for the coding agent

The previous v2 said “force smoothness first” but then made the spline depend on
a broad performance P0 and placed locality after kernel optimization. This order
is corrected.

| Order | Step | Purpose / exit condition |
|---|---|---|
| **P0** | Sparse-path firewall + event timing | No hidden dense/host-roundtrip path can masquerade as sparse; no major tuning yet. |
| **P1** | Canonical C2 electronic spline | Gate A passes; repulsive spline kept semantically separate. |
| **P2** | Wire analytic H/S derivatives end-to-end | Analytic f64 force passes energy-FD and DFTB+ checks. |
| **P3** | Sparse force with `D=2K`, `W=2KHK` | Gate B passes with no diagonalization. |
| **C** | Locality sweep `R_K x R_Z` | Choose measured production masks / validation mask. |
| **D** | Nonsingular padded Si/H basis | Realistic passivated small NC passes parity/locality; dummy occupation negligible. |
| **P4** | Structural/performance sparse work | Remove dense bounds, add cell-list masks, ProductPlan + degree/team benchmarks. |
| **P5** | TC2 hot-loop cleanup | Device branch, no full Q buffer if fusion wins, no per-iteration host scalar sync, no extra KSK just for convergence. |
| **P6** | Gamma cache + precision variants | Measure fast/pairwise/Kahan/f64-tail; choose simplest adequate default. |
| **E** | Determinism + h sweep | Separate bias from noise and choose stable h plateau. |
| **P7/F** | Coarse then frozen-mask optimization | Small Si/H reaches its own stationary minimum robustly. |
| **P8/G** | Hessian engine + same-geometry parity | `H_raw/H_sym/H_mw/H_phys` pipeline correct; Gate G passes. |
| **H** | Own-minimum spectrum validation | Significant negative modes diagnosed physically. |
| **I** | Scaling/profile | Whole-program bottlenecks known; sparse core scaling demonstrated honestly. |
| **J** | N~300 then N~1000 production | Reproducible useful spectrum and timing report. |

Do not attach day estimates to these steps in the source-of-truth manifest. The
point is dependency order and exit criteria, not optimistic scheduling.

---

## 7. Kernel / harness ownership

Keep numerical responsibilities separated enough that an optimization to one
kernel cannot silently change the physics in another.

| Area | Recommended ownership |
|---|---|
| sparse BSR4 SpGEMM / reductions / TC2 / Newton-Schulz | `methods/sparse/sparse_bsr4_purification.cl` + `gpu_sparse.rs` |
| sparse structures, safe sparse bounds, masks | `methods/sparse/bsr4.rs`, `masks.rs` |
| symbolic product plans / degree buckets | `methods/sparse/symbolic_plan.rs` |
| electronic SK spline preprocessing | `methods/dftb/spline_resample.rs` |
| electronic SK GPU interpolation + H/S assembly | existing `dftb_hamiltonian.cl` or a shared SK interpolation include used by H/S + force kernels |
| analytic SK angular derivatives | `methods/dftb/rotation.rs` + GPU equivalent/shared helper |
| SCC gamma build/patch/matvec + Hscc update | new/appropriate sparse SCC kernel/module, **not** the BSR4 purification file |
| analytic sparse force contractions | `methods/sparse/sparse_forces.rs` + dedicated OpenCL kernel if GPU contraction is separated |
| dense/reference force and FD validation helper | `methods/dftb/forces.rs`, FD helper test-only |
| Hessian assembly / raw diagnostics | `core/hessian.rs` |
| mass weighting / rigid projection / frequencies | `core/phonon.rs` |

In particular, do **not** put `V''` spline evaluation and gamma/Kahan kernels into
`sparse_bsr4_purification.cl` simply because they are GPU kernels. That file
should remain sparse linear algebra.



---

## 8. Geometry Generation — External Repos (do NOT duplicate here)

**Policy:** We do NOT want to pollute the dftbplus repo with nanocrystal
geometry-building machinery that already exists in FireCore. Instead, generate
geometries **in FireCore** and export `.xyz` / `.mol2` / `.npz` files into
`data/xyz/` here. This section documents where the tools live and how to use
them. **There is no diamond-cubic builder to write in Rust** — the only thing
needed here is the `.xyz` loader (`rust_dftb/src/io.rs::parse_xyz`, exists).

### 8.1 FireCore — Si/diamond nanocrystal generator (JS + Python)

**Repo:** `/home/prokop/git/FireCore`
**Codemap:** `FireCore/CODEMAP.md` (entry point)
**Topical audit:** `FireCore/doc/topical_audit/Nanocrystal_Vibrations.md`
**Working hub:** `FireCore/tests/tSiNCs/README.md` + `AGENTS.md`

FireCore has a very rich nanocrystal generator supporting spherical cuts,
Miller-plane facets, Wulff shapes, H-passivation, bridge defects, silyl
passivation, and ensemble batch generation. Both JavaScript (feature-complete)
and Python (spherical cuts) CLIs are available.

**Key files:**

| File | Role |
|---|---|
| `web/molgui_webgpu/Nanocrystals.js` | **Core JS library**: CIF → cuts → prune → H-cap → bridges / `silyl100Prob` / `silyl111Prob` / fuse. Wulff shapes, Miller planes, defect operators. |
| `web/molgui_webgpu/EditableMolecule.js` | Molecular graph with editing ops |
| `web/molgui_webgpu/CrystalUtils.js` | Crystal symmetry, primitive cell ops |
| `web/common_js/npzIO.js` | NPZ I/O (crystal arrays, topology) |
| `web/common_js/nanocrystalSvg.js` | SVG export, ring detection viz |
| `tests/tSiNCs/nanocrystals.mjs` | **Unified CLI**: `generate`, `ensemble`, `topology`, `audit`, `nonbond`, `rings` |
| `tests/tSiNCs/gen_nanocrystals.py` | **Python CLI**: spherical cuts native; Miller planes delegate to Node |
| `pyBall/nanocrystal_gen.py` | **Python sphere-cut builder**: `build_spherical_nanoparticle`, `save_xyz`, `find_cap_hh_pairs`. Parity target for JS. |
| `pyBall/nanocrystal_pipeline.py` | **NPZ pipeline CLI**: `relax` → `hessian` → `spectrum` → `accumulate` (stages 01–05) |
| `pyBall/FTIR.py` | Vibrational spectra post-processing: `build_hessian_from_linear_topology`, rigid-mode projection, mass matrix |
| `tests/tSiNCs/crosscheck_nanocrystal_generators.py` | JS vs Python generator parity verification |
| `tests/tSiNCs/chem_atlas.json` | Atlas config for batch ensemble generation |

**Crystal primitive cells** (in `cpp/common_resources/crystals/`):
`Si_primitive.xyz/.cif`, `Si-sym.cif`, `Si_conventional.xyz`,
`diamond_primitive.xyz/.cif`, `C_diamond_sym.cif`, `diamond_conventional.xyz`.

**Pre-built fixtures** (in `tests/tSiNCs/fixtures/`): `si_1nm_passivation/`
(nine-crystal NPZ pipeline gallery, stages 01–05), `npz_viewer/`,
`vibration_benchmarks/`.

**Workflow to generate Si nanocrystal geometries:**

1. **Spherical cut (Python, simplest):**
   ```bash
   cd /home/prokop/git/FireCore
   python3 tests/tSiNCs/gen_nanocrystals.py \
       --cutMode sphere --element Si --sphere-r 10.0 --sphere-nrep 5 \
       --caps H --outDir tests/tSiNCs/OUT_nanocrystals_py
   ```
2. **Miller-plane facets (JS, feature-complete):**
   ```bash
   node tests/tSiNCs/nanocrystals.mjs generate \
       --cif cpp/common_resources/crystals/Si-sym.cif \
       --cutMode planes --planeTemplates a111 \
       --nx-range 3,3 --ny-range 3,3 --nz-range 3,3 \
       --caps H --outDir tests/tSiNCs/OUT_nanocrystals
   ```
3. **Wulff shape:** `nanocrystals.mjs generate --wulffShape octahedron ...`
4. **Batch ensemble:** `nanocrystals.mjs ensemble --atlas chem_atlas.json ...`
5. **Copy `.xyz` into dftbplus `data/xyz/`.**
6. **Verify generator parity:** `crosscheck_nanocrystal_generators.py`.

For diamond (C): same tools, `--element C` + `C_diamond_sym.cif`.

**Sizing** (achievable with FireCore generators):

| Shape | Radius (Å) | N_Si | N_H | N_total | N_orbs (Si=4, H=1) |
|---|---|---|---|---|---|
| Small | 5 | ~30 | ~30 | ~60 | ~150 |
| Medium | 10 | ~200 | ~100 | ~300 | ~900 |
| Large | 15 | ~600 | ~200 | ~800 | ~2600 |
| X-large | 20 | ~1200 | ~400 | ~1600 | ~5200 |

### 8.2 FireCore — Hessian and vibration pipeline (reference)

FireCore has a complete Hessian + vibration pipeline (classical force fields,
not DFTB). Useful as a **reference**:

| File | Role |
|---|---|
| `pyBall/nanocrystal_pipeline.py` | NPZ pipeline: `relax` (MMFF) → `hessian` → `spectrum` → `accumulate` |
| `pyBall/FTIR.py` | `build_hessian_from_linear_topology`, rigid-mode projection, `vibration_spectrum_from_modes` |
| `pyBall/MMFF.py` | `getHessian3Nx3N(inds, dx)` — 3N×3N Hessian via central FD in C++ |
| `spammm/dynamics/Vibrations.py` | (SPAMMM) Normal-mode analysis: DFTB/UFF/SPFF Hessian, rigid-mode projection, mode analysis |

**Key lesson** (`doc/Topics/FTIR_Nanocrystals/Hessian_at_own_minimum.md`):
> Harmonic spectrum = Hessian at that method's **own minimum**. Relax with the
> **same** potential until f_max < f_conv, then Hessian. DFTB geometry + MMFF
> Hessian is FFfit only, never a spectrum.

This is why Gate F→H insists the Hessian be computed at the **DFTB-optimized**
geometry of the *same* (sparse/f32) model, not a geometry from another method.

### 8.3 SPAMMM — Vibrational analysis (reference)

**Repo:** `/home/prokop/git/SPAMMM` · `SPAMMM/doc/Topics/Vibrations.md`

| File | Role |
|---|---|
| `spammm/dynamics/Vibrations.py` | `run_vibrations(mol, backend=...)` — Hessian assembly, rigid-mode projection, mode analysis |
| `spammm/dynamics/VibrationPlot.py` | Top-view mode plots |
| `spammm/quantum/DFTB_utils.py` | `write_dftb_input_hessian`, `read_hessian`, `hessian_hartree_bohr_to_eV_angstrom` |

### 8.4 What NOT to build in dftbplus

- **No diamond cubic / Wulff / Miller-plane builder** — use FireCore.
- **No H-passivation logic** — FireCore handles Si-H, C-H, silyl, bridges.
- **No MMFF Hessian pipeline** — FireCore's `nanocrystal_pipeline.py` is
  classical; our task uses the DFTB Hessian.
- **No rigid-mode projection machinery from scratch** — reference SPAMMM's
  `Vibrations.py` / FireCore's `FTIR.py` for the algorithm; implement in Rust.
- **No NPZ I/O** — `.xyz` is the interchange format.

---

---

## 9. File ownership / expected edits

This is a working map, not permission to create every file before it is needed.
Prefer the smallest integration that preserves the separation in §7.

| File | Expected change |
|---|---|
| `rust_dftb/src/methods/sparse/gpu_sparse.rs` | production-resident orchestration, event/perf stats, no hot-loop host scalar read, fused/planned paths |
| `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl` | ProductPlan SpGEMM variants, TC2 fusion/reductions; **linear algebra only** |
| `rust_dftb/src/methods/sparse/bsr4.rs` | sparse-safe bounds/helpers; keep dense helpers clearly reference-only |
| `rust_dftb/src/methods/sparse/masks.rs` | independent H/S, K, Z, validation masks; cell-list construction; frozen-skin metadata |
| `rust_dftb/src/methods/sparse/symbolic_plan.rs` | optional precomputed numerical product plan + stats |
| `rust_dftb/src/methods/dftb/spline_resample.rs` | canonical f64 -> cubic B-spline preprocessing and continuity tests |
| existing SK/Hamiltonian OpenCL source | corrected analytic `V,V',V''`, table-access benchmarks, mixed 4x4 physical block assembly |
| `rust_dftb/src/methods/dftb/rotation.rs` | ensure analytic derivatives are the production route |
| `rust_dftb/src/methods/dftb/forces.rs` | dense/reference force; isolate pair FD as test/reference only |
| `rust_dftb/src/methods/sparse/sparse_scc.rs` | SCC loop, gamma cache/patch/matvec, q/Z warm-start policy |
| `rust_dftb/src/methods/sparse/sparse_forces.rs` | sparse analytic force using `D=2K`, `W=2KHK` |
| `rust_dftb/src/core/hessian.rs` | three-/five-point force Hessian, displacement reuse, `H_raw` diagnostics |
| `rust_dftb/src/core/phonon.rs` | mass weighting, rank-aware rigid projection, f64 eigensolve |

Suggested tests:

```text
sparse_spline_parity.rs        Gate A
analytic_force_parity.rs       Gate B (pair + full force)
sparse_edm_force_parity.rs     Gate B (D/W + sparse force)
sparse_locality_sweep.rs       Gate C
sih_padded_basis.rs            Gate D
force_determinism_h_sweep.rs   Gate E
nanocrystal_optimize.rs        Gate F
hessian_parity.rs              Gate G
nanocrystal_vib.rs             Gate H
sparse_scaling.rs              Gate I
nanocrystal_perf.rs            Gate J
```

Do not split tests merely to satisfy this list; combine them when that makes the
workflow easier to run and understand.

---

## 10. SK-file policy

- Use one SK parameterization consistently within each validation chain.
- `siband-1-1` is the natural Si/Si-H starting point if its force/phonon behavior
  passes the small references.
- `matsci-0-3` is a natural diamond/C-H and mixed-material candidate.
- `pbc-0-3` is another comparison point where appropriate.
- Do not declare one set “best” from its name. Benchmark equilibrium geometry,
  force parity and small-system vibrational behavior.
- Do not hard-code 128–256 resampled nodes as a requirement. Sweep enough node
  counts to show `V,V',V''` convergence; interpolation cost is expected to be
  small relative to purification, so choose the smallest count already on a
  clean plateau rather than minimizing bytes at all costs.
- Record SK-set hash and spline preprocessing settings in every production run.



---

## 11. Related Documents

- `doc/prokop/tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.chat.md`
  — the critique this revision answers.
- `doc/prokop/tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md`
  — superseded original draft (kept for reference).
- `doc/prokop/tasts/GPU_Sparse_Resident_Integration/task.md` — prior task that
  integrated the device-resident sparse purification path.
- `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` — §7.5 tracks
  sparse BSR4 purification status.
- `doc/prokop/topical_audit/sparse_tc2_purification.md` — TC2 audit.
- `doc/prokop/topical_audit/davidson_eigensolver.md` — Davidson (frontier
  orbitals / gap diagnostics).
- `test/app/phonons/Si/` — DFTB+ Fortran Si₂ phonon reference.
- `doc/prokop/AGENTS/guidelines/efficiency.md` — efficiency rules (no
  allocation in hot loops, three-tier data lifetime, verify library internals
  before claiming).

---

---

## 12. Key Differences from Task 1 (H-Bond Relaxed Scan)

| Aspect | Task 1 (H-bond scan) | Task 2 (Nanocrystal vib) |
|---|---|---|
| System type | Nucleobase pairs (compact, dense) | Si/diamond nanocrystals (extended, sparse) |
| System size | ~30 atoms, ~87 orbs | 300–1000 atoms, ~1500–4000 orbs |
| Number of systems | Many (batched, 100–1000) | One (single system) |
| GPU strategy | Batched dense SCC, N>64 extension | Sparse BSR4 purification, masked SpGEMM |
| Key bottleneck | N>64 eigensolver, GPU forces | Smooth analytic force, measured `R_K/R_Z` locality, frozen topology, whole-program scaling |
| Forces | Analytic (port to GPU) | Analytic sparse via `D=2K, W=2KHK`; FD only for force→Hessian |
| Precision | f32 GPU, dense | f32 bulk + benchmarked compensated/f64 reductions; bias and history sensitivity measured separately |
| Sparsity question | n/a (dense) | Truncation error vs radius — **not** densification (nnz fixed by mask) |
| Post-processing | Relaxed 2D PES, barrier analysis | Hessian, vibrational frequencies |
| Accuracy concern | SCC convergence on polar systems | Imaginary modes from force noise / spline discontinuity / mask jitter |
| SK set | mio-1-1 (H,C,N,O) | siband-1-1 (Si) or matsci-0-3 (C,Si) |
| Geometry source | SPAMMM (ASCII art + coordinate_scan) | FireCore (Nanocrystals.js + gen_nanocrystals.py) |
