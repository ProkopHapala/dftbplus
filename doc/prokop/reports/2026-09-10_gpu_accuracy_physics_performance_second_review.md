# Second review: GPU accuracy floors, physical validity, and performance

Date: 2026-09-10. Status: review and proposals, not implemented or GPU-verified in this pass.

## Executive conclusion

Keep matrix storage and the main dense/sparse products in f32. **The available evidence does not justify either a wholesale f64 conversion or the claim that the observed errors are unavoidable f32 limits.** Several distinct effects are currently called an “f32 floor”: output quantization, inaccurate arithmetic, incomplete electronic convergence, fragile DIIS, and finite-difference truncation. They require different remedies.

The best provisional precision allocation is:

1. Preserve final energy components and totals in f64; do not round an extensive total to f32 before subtracting nearby geometries.
2. Make the small DIIS solve numerically reliable, preferably with f64 arithmetic, scaled residual Gram matrices, and explicit rank/constraint checks. This is a much cheaper place for f64 than matrix multiplication.
3. Keep f32 matrix storage and FMA-based products. Benchmark multiple accumulators/pairwise accumulation before compensated dots; use compensation only where frozen-input diagnostics show accumulation dominates.
4. Retain the existing f64 gamma-derivative implementation as the correctness reference. A stable reformulation and cached species-pair coefficients are better eventual speed targets than merely summing its two ill-conditioned terms more accurately.
5. Measure metric/eigen/projector defects independently. Consider occasional higher-accuracy overlap or occupied-subspace repair only if these measurements identify the limiting error.
6. Remove repeated geometry-independent work and host dense W construction. Those costs can outweigh the arithmetic improvements under discussion.

**Most important corrections to the preliminary interpretation:** the dense eigenvalue comparisons use different SCC states; the current trace diagnostic reads a repurposed buffer; the tiled Jacobi implementation already performs substantial hot-loop f64 work; and the sparse Hessian results have not separated solver stopping error from arithmetic error.

## 1. Scope and evidence

This report inspects the working tree, including uncommitted changes, against HEAD `c2c45aecade12a4e010f42099fa05c291245b5e7`. Other agents were changing the tree during inspection, particularly `gpu_dftb.rs`. Locations below identify symbols and approximate observed lines, not an immutable release. No simulation source, tests, build configuration, or Git index was changed by this review.

I read the current kernels, orchestration, tests, and existing dense/sparse numerical audits. I independently recomputed statistics from the saved Gate F/G CSVs. **I did not run a fresh GPU test suite or benchmark.** Recorded RTX 3090 results below are attributed to existing reports, not presented as new measurements. Source inspection can establish contract defects, but cannot determine the fastest precision policy on the target GPU.

Primary existing numerical accounts:

- [Dense numerical audit](../topical_audit/f32_floor_dense_hbond.md).
- [Sparse numerical audit](../topical_audit/f32_floor_sparse.md).
- [Earlier force-resolution report](2026-09-08_gpu_force_resolution.md).

The scientific/testing guidance informed the distinction between kernel parity and physical validation; performance guidance informed the separation of arithmetic cost from allocation, transfers, and repeated work. This remains the requested Markdown document, not a new application or implementation.

## 2. Dense and sparse paths: common physics, different numerical engines

Use `q` for positive electron populations, `q0` for reference populations, `dq=q-q0`, and `V=G dq`. For the closed-shell model reviewed here:

\[
H=H_0+\tfrac12 S\odot(V_A+V_B),\qquad
E[D]=\operatorname{Tr}(DH_0)+\tfrac12 dq[D]^T Gdq[D]+E_{rep}.
\]

| Layer | Dense multi-system path | Sparse path | Shared issue |
|---|---|---|---|
| Electronic representation | `C`, eigenvalues, `D=2 Cocc Coccᵀ` | Spinless `K`, `D=2K` | Correct occupation, metric normalization, SCC state |
| Overlap treatment | `X=S⁻¹/²`, then `X H X` | `Z≈S⁻¹`, generalized TC2 | Conditioning of S and rounded products |
| Electronic solver | Full-local/tiled Jacobi | NS initialization and TC2 purification | Genuine residual versus a stopping heuristic |
| Energy-weighted density | `W=2 Cocc εocc Coccᵀ` | `W=2 K H K` | Equivalent for the appropriate converged spectral projector, not arbitrary approximate K |
| Forces | GPU non-SCC, shift, gamma derivative, repulsive kernels | Physics gates currently use CPU contractions from GPU K | Same four physical terms; different coverage |
| Lifetime owner | `GpuDftb` + `GpuSccPlan`, still incomplete in important places | `SparseSystemWorkspace` exists, but `run_scc` is one-shot purification | Persistent buffers alone do not make a complete SCC/force engine |

Do not unify the algebra prematurely. Share physical conventions, geometry/static parameter data, residual definitions, and diagnostic fixtures. Keep eigensolver-specific and purification-specific numerical policies separate.

## 3. Findings that obstruct a trustworthy accuracy-floor diagnosis

### 3.1 Dense DIIS is a plausible accuracy limiter, not a cleared suspect

Evidence: [gpu_matrix_ops.cl](../../../rust_dftb/src/qmqm/gpu_matrix_ops.cl), `diis_step_batched`, approximately lines 1245–1420; [gpu_scc_plan.rs](../../../rust_dftb/src/qmqm/gpu_scc_plan.rs), `scc_step_diis` and `reset_diis`.

The kernel forms the residual Gram matrix with serial f32 dot products and solves the augmented system in f32. Near convergence the Gram entries can be around 1e-12 while the constraint row/column contains ones. More seriously:

- Elimination skips pivots below the absolute threshold `1e-14`.
- Back substitution substitutes zero for a small diagonal without reporting a deficient solve.
- Acceptance checks coefficient finiteness and magnitude, but not `sum(c)=1` or the linear-system residual.
- The history count is incremented before the `n==0` branch. That initial simple-mixing branch is normally unreachable; a one-entry DIIS history produces the unmixed output rather than the advertised alpha-damped first step.
- The quantity written to `rms` is `sqrt(sum(res²))`, an L2 norm, **not RMS**. CPU/sparse SCC uses division by atom count. Identical numeric thresholds therefore do not mean identical convergence requirements.
- The host maximum reduction uses floating-point `max`, which can discard NaNs when paired with finite values. A maximum is not a finiteness check.

The final DIIS sum combines absolute atomic populations. If coefficients sum to `1+δ`, the neutral population offset contributes an artificial `δ q0`. This is avoidable amplification, not a fundamental need for f64 density matrices.

**Recommended first experiment:** hold the electronic map fixed and compare the current mixer against a scaled, rank-aware small f64 solve. Validate affine constraint, solve residual, total electron count, and every finite value. Use an anchored affine combination or mix excess charges. Construct excess populations accurately before rounding: subtracting q0 from an already-rounded f32 population cannot recover discarded bits.

Numerical rank loss is expected when residual histories become dependent. Handle it as an explicit, diagnosed mixer operation; do not silently invent coefficients or reinterpret failure as convergence. Merely widening arithmetic without addressing rank/scaling is insufficient.

The existing audit reports that f64 GEMM worsened SCC convergence. This does **not** demonstrate that accurate GEMM is intrinsically harmful. It changes the nonlinear iteration, exposing mixer sensitivity. Compare fixed-H products/eigensolutions before judging the end-to-end result.

### 3.2 Finalizing the Hamiltonian is not proving self-consistency

Evidence: `GpuSccPlan::finalize`, approximately lines 749–818; `compute_energy`, approximately lines 678–744.

`finalize` rebuilds V and H from `q_gpu`, solves the electronic problem, and constructs D. It does not recompute the final Mulliken charges or validate their residual against `q_gpu`. Thus all outputs derive from one input charge vector, but the density can still imply another charge vector.

There is a useful exact diagnostic separating this issue from arithmetic. Let:

\[
q_{in}=q_{gpu},\quad q_D=\mathrm{Mulliken}(D,S),\quad r=q_D-q_{in},
\]
\[
\delta_{eig}=E_{band}-\operatorname{Tr}(D H[q_{in}]).
\]

For symmetric G, ignoring additional rounding in the diagnostic itself:

\[
E_{bandform}-[\operatorname{Tr}(D H_0)+\tfrac12 dq_{in}^TV]
=\delta_{eig}+r^TV,
\]

and, relative to the canonical density-evaluated electronic energy,

\[
E_{bandform}-E[D]=\delta_{eig}-\tfrac12 r^TGr.
\]

Here `Ebandform=Eband−½dq_in·V−q0·V`; omit the identical repulsive term on both sides. These identities follow directly from the implemented Hamiltonian/population convention.

Consequences:

- Switching to the band formula is not inherently cheating or a different converged physical model.
- A smaller energy error after switching does not prove that the remaining discrepancy is exclusively eigenvalue error.
- A good total energy can coexist with a materially imperfect charge/force state.
- Diagnostics should print `r`, `δeig`, both energy forms, and their predicted difference from the **same arrays**.

The sparse force gate has a related finite-SCC issue: `run_sparse_scc` purifies H built from input charges, evaluates energy and shifts using output charges, and retains the input H for `W=2KHK`. At finite residual this is not one fully stationary state. The CPU reference `MultiSystemSolver::solve_scc` also returns with density/output charges from the latest solve and shifts from its input. Therefore shared finite-tolerance discrepancies must be isolated before blaming GPU arithmetic.

### 3.3 Current dense diagnostics cannot establish the claimed eigen floor

Evidence: [gpu_hbond_physics.rs](../../../rust_dftb/tests/gpu_hbond_physics.rs), `run_full_chain_scc`, approximately lines 1037–1145.

`compute_energy` now stores `q0·V` in `plan.tr`. The test still reads that buffer as `e_h0_kernel`, labels it GPU `E_h0`, and compares it against a host Frobenius trace. That diagnostic is stale. Earlier trace measurements may have been valid with the earlier implementation; the current printout does not reproduce them.

The same test compares CPU and GPU eigenvalues after separately converging SCC. Even with identical H0/S, different charges produce different V and H. A nearly uniform potential displacement shifts eigenvalues systematically; band and double-counting terms can then largely cancel. The reported occupied-eigenvalue difference around 2.2e-3 Ha, largely cancelled by `q0·V`, is precisely why fixed-H evidence is necessary.

**Required isolation:** diagonalize the identical rounded H/S/V on CPU and GPU; separately compare the rounded problem against the original f64 problem. Measure `HC−SCε`, `CᵀSC−I`, occupied-subspace/projector differences, and the metric condition number. Individual eigenvectors are not an appropriate comparator inside degenerate eigenspaces.

The duplicated C/D reconstruction block in this diagnostic also adds unnecessary readbacks and computation. This is secondary to fixing what the numbers mean.

### 3.4 Jacobi can stop without exposing failure, and overlap clamping hides invalid inputs

Evidence: [gpu_tiled_jacobi.cl](../../../rust_dftb/src/qmqm/gpu_tiled_jacobi.cl), outer convergence and final cleanup near lines 398–420; [gpu_eigen.cl](../../../rust_dftb/src/qmqm/gpu_eigen.cl), inverse-square-root construction near lines 416 and 477.

The tiled kernel exits on a relative off-diagonal criterion, a three-sweep stagnation heuristic, or iteration exhaustion. It then zeros the remaining off-diagonal entries unconditionally. No accompanying convergence status/residual is consumed by the SCC plan. A diagonal output is therefore not evidence that the original matrix was diagonalized accurately.

The inverse-square-root path evaluates `rsqrt(max(λ,1e-7))`. The plan has a `lambda_min` buffer, but `set_geometry` does not validate it. Negative or nearly singular overlap eigenvalues can be silently replaced by the floor.

This is not solved by Kahan. Preserve/report the measured residual and termination reason, and validate overlap admissibility and conditioning. If removing near-linear dependencies is ever desired, that is an explicit basis/model operation with an error contract, not an invisible clamp.

### 3.5 Sparse convergence checks need stronger physical contracts

Evidence: [gpu_sparse.rs](../../../rust_dftb/src/methods/sparse/gpu_sparse.rs), `tc2_purify`, `SparsePurifyWorkspace::tc2_purify_dev`; [sparse_system.rs](../../../rust_dftb/src/methods/sparse/sparse_system.rs), `tc2_purify`.

The general TC2 routines accept `||KSK−K|| < tol` without requiring the requested trace simultaneously. An exactly idempotent wrong-rank projector—including K=0—can satisfy that condition. The `purify_h` wrapper adds a trace check, but that does not repair the public lower-level contracts.

A physical acceptance gate needs occupation/trace, generalized idempotency, finite values, and Hamiltonian compatibility, e.g. `HKS−SKH`. Even those together do not prove ground-state occupation: a projector onto the wrong eigenstates can satisfy them. Retain valid spectral bounds/initialization and reference checks of occupied energy/subspace.

For a truncated sparse mask, distinguish the residual projected onto stored blocks from the full-operator residual. Validate on a sufficient product/verification mask. Sparse locality error and roundoff are independent axes; compensation cannot reconstruct deliberately discarded matrix entries.

Other relevant details:

- The implemented bounds use **ZH**, appropriate to the generalized eigenproblem when Z approximates S⁻¹. The audit's algorithm diagram says ZHZ; that is misleading. ZHZ belongs in the K0 construction, not as interchangeable spectral bounds.
- `bsr4_build_Hscc` shifts all block lanes, including dummy diagonal slots with Sdd=1. The host `apply_shift_padded` intentionally shifts physical slots only. Before using the device SCC builder in production, test this convention with nonzero shifts and check dummy occupation/order.
- `SparseSystemWorkspace::run_scc` still performs Z, K0, TC2, Mulliken once—no self-consistent charge loop. Its name and existence must not substitute for SCC validation.

### 3.6 Device Newton–Schulz: investigate the residual, but do not inherit an unproven root cause

The recorded mismatch is `R_Z≈1.9e-5` with `max|Z−S⁻¹|≈2.18e-3`. The current source already computes the direct residual `||ZS−I||`, avoiding the old cancellation-prone expression `||ZS||²−2Tr(ZS)+N`. I did not reproduce the historical mismatch on the current tree.

The right diagnostic is to download the returned Z and the exact S used by the kernel, recompute `ZS−I` in f64, and compare each stage: product values, diagonal flags, partial reductions, scalar result, and returned buffer identity. The structure constructor currently builds diagonal flags and validates row capacity; do not assume those are missing.

For the full operator:

\[
Z-S^{-1}=(ZS-I)S^{-1},\quad
\|Z-S^{-1}\|_F\le\sqrt N R_Z\|S^{-1}\|_2.
\]

Thus inverse error need not equal residual, but its amplification is bounded by conditioning. The existing 12-orbital test has diagonal one and off-diagonals bounded by 0.075; its Gershgorin lower bound is at least 0.175, so the crude amplification bound is about 19.8. The historical ratio over 100 merits investigation if reproduced. A universal `error < 20 R_Z` is not valid for arbitrary overlaps.

`dftb_engine.rs` still calls the device NS path. Until that exact path is reverified, keep its status explicitly unresolved. Do not label the current direct-residual implementation proven broken solely from a historical log, or solve an indexing/state bug by increasing precision.

## 4. What the saved tests establish—and what they do not

Independent recalculation of the two 15×15 unsymmetrized Hessian CSVs:

| Quantity | Recomputed value |
|---|---:|
| Sparse `||H||F` | 3.0832438180 Ha/Å² |
| Dense `||H||F` | 3.0858191941 Ha/Å² |
| Sparse `||H−Hᵀ||F/||H||F` | 3.8703889e-4 |
| Dense same asymmetry | 3.8858238e-4 |
| Relative sparse/dense Hessian difference | 1.0951149e-3 |
| Maximum element difference | 1.16538435e-3 Ha/Å², zero-based (3,3) |

Sources: [sparse Hessian](../../../debug/sparse_review/gate_g_h_sparse.csv), [dense Hessian](../../../debug/sparse_review/gate_g_h_dense.csv).

The [Gate F trajectory](../../../debug/sparse_review/gate_f_sih4.csv) contains 68 rows, ending at step 67 with E=−2.826057140448 Ha, mean Si–H=1.47727947 Å and force norm 4.1796e-4 Ha/Å. It contains one small upward energy step (~9.16e-7 Ha). Standard FIRE is not a monotonic line-search algorithm, so this alone is not a defect; do not describe the saved trajectory as strictly monotonic.

The new unsymmetrized Hessian calculation, full electronic-plus-repulsive energy, explicit valence fixture, and fail-loud GPU harness are meaningful improvements over tautological symmetry tests and incomplete energy models. Nevertheless:

- The physics path uses NS tolerance 1e-5, TC2 idempotency tolerance 1e-4, and SCC tolerance 1e-5. One observed error at those stopping settings is not a precision-floor measurement.
- Similar dense/sparse asymmetry at **one h=0.01 Å** is evidence of a common finite-difference/convergence contribution, not proof that compensation or tighter convergence cannot help.
- Gate G accepts 5% relative Hessian error or 1e-3 absolute error. It does not enforce the observed ~0.11% performance as a regression limit. Many reported spectral quantities are diagnostics, not assertions.
- G3.4 accepts an absolute alternative to the original relative energy-gradient target. This is disclosed, which is good, but the original target remains unestablished until an h/tolerance sweep explains the discrepancy.
- AT/GC force-component tests use CPU-fed D/W. They validate GPU force contraction, not the complete GPU SCC-to-force path. Sparse G3/F/G use full masks on tiny SiH4 and CPU force contractions, not a production masked sparse force pipeline.
- The new `gpu_dftb` integration test is smoke coverage: requested SCC tolerance is 1e-6 but accepted residual is 1e-4; energy bounds are broad; force finiteness is checked after a NaN-discarding maximum. It does not demonstrate MD correctness or allocation-free operation.
- The sparse runner combines several integration binaries without `--no-fail-fast`; an earlier failing binary can prevent later gates from executing. A red overall run does not establish that every requested diagnostic ran. Its explicit `PIPESTATUS[0]` return also ignores a possible `tee` failure.

**Rigid modes:** translation invariance implies exact translational Hessian null directions at any geometry, not just at a minimum. Rotation tangents satisfy a relation involving the gradient and become null at a stationary point. Finite force therefore does not explain every kind of rigid-mode leakage. The saved Hessians give `||Ht||` around 3e-4 Ha/Å² for normalized translations on both paths; study its h/convergence dependence. Retain raw H before symmetrizing/projecting for spectra. Compare degenerate mode subspaces rather than individual eigenvectors.

**Fixture correction:** the G3 geometry labelled tetrahedral actually has H–Si–H angles 109.47° (three pairs), 100.67°, 65.96°, and 141.06°. It is a valid distorted molecule, but not a tetrahedral equilibrium fixture. Do not infer tetrahedral degeneracies or zero forces from that label.

## 5. A precision budget that fits slow-f64 gaming GPUs

### 5.1 There is no single f32 floor

Separate five contributions:

1. Input/representation error: rounding geometry, H/S, coefficients, charges, or final outputs.
2. Arithmetic error: dot products, cancellation, repeated matrix updates.
3. Solver error: incomplete eigensolve, purification, inverse, or SCC convergence.
4. Method error: interpolation/tail choices, sparse truncation, occupation/model restrictions.
5. Observable estimation error: finite-difference truncation and amplified energy/force noise.

For a smooth central energy difference, a useful model is `C h² + δE/h`; for a force-difference Hessian it is `C' h² + δF/h`. The constants and error correlation must be measured. Larger h reduces noise amplification but increases truncation; smaller h is not automatically more accurate. Subtraction in f64 helps only if the inputs retain the relevant precision.

For binary32, unit roundoff is `u=2⁻²⁴≈5.96e-8`. A well-conditioned short calculation can do much better than the current observed errors, while a poorly conditioned one can do far worse. These are not universal absolute tolerances.

### 5.2 Highest-value f64 use: final scalar energies and the small mixer solve

`GpuSccPlan::compute_energy` already sums occupied eigenvalues in host f64, but then casts the result to f32 and adds repulsion in f32. `GpuDftb::energy` returns `Vec<f32>`.

At |E| between 64 and 128 Ha, a binary32 ULP is 7.6294e-6 Ha. One ULP of difference between two rounded energies divided by `2h` at h=1e-3 Å is 3.8147e-3 Ha/Å. This is an illustrative output-quantization scale, not a measurement of AT's force error. Above 128 Ha the ULP doubles.

**Recommendation:** f64 final scalar/component accumulation and f64 output, or a rigorously defined hi/lo scalar representation if device capability demands it. Avoid premature f32 rounding of individual cancelling components too. This is negligible data volume compared with dense matrices and prevents a hard, avoidable output bottleneck. It does not repair incorrect D/C/K.

Likewise, f64 for a DIIS system with at most about eleven rows is a small arithmetic island. Prefer keeping it on device if a host solve adds a synchronization point. Measure total time to convergence, not just the tiny solve duration.

### 5.3 Kahan is useful—but only at the correct layer

Kahan/Neumaier compensation addresses accumulation error, including cancellation in summation. It cannot recover errors already made while evaluating the summands, missing sparse entries, or a wrong electronic state. The preliminary statement that Kahan is essentially for same-sign sums is too restrictive. For dot products, error-free product/sum transformations provide a more relevant extension than compensating only the final scalar. [Ogita, Rump and Oishi, Accurate Sum and Dot Product](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).

Candidate order for dense GEMM and sparse SpGEMM:

1. Explicit f32 FMA and several independent accumulators, then a balanced combination.
2. Compensated combination of tile/block partial sums where the cross-block accumulation dominates.
3. Full compensated dots only in sensitive kernels/rows identified by diagnostics.
4. f64 accumulation as a reference and selective production option, not the global default.

FMA rounds a product-plus-sum once. It reduces rounding but does not eliminate the serial accumulation chain or conditioning of the answer. Explicit `fma(a,b,-p)` can recover the residual of a rounded product p under the usual range/rounding assumptions; combine it with a proper compensated sum if product error matters. Preserve required evaluation order and do not enable reassociation that algebraically removes compensation. [NVIDIA floating-point guide](https://docs.nvidia.com/cuda/archive/11.3.1/floating-point/index.html).

For the sparse 4×4 products, the four-term block contraction is short; accumulation over many contributing blocks is the more natural first compensation target. Record row contribution counts and cancellation indicators. Kahan in an already f64 host trace of f32 K is unlikely to improve the existing sparse force/Hessian result.

Compensation adds dependent operations and registers; full dot compensation can hurt occupancy or spill. No speedup or fixed overhead percentage is established here. Benchmark the implementation, not FLOP counts alone.

### 5.4 Metric/eigensolver refinement: conditional, not blanket promotion

First determine whether the dominant defect is in X, transformed H, the eigensolve, or the occupied reconstruction. Measure in higher precision from unchanged rounded inputs:

- `||XᵀSX−I||` and minimum/maximum overlap eigenvalues.
- `||HC−SCε||`, with scale-normalized residuals as well as absolute maxima.
- `||CoccᵀSCocc−I||`, charge conservation and projector idempotency.
- Frozen-input energy/force sensitivity to each replacement.

If overlap construction dominates, it is reused throughout an SCC solve: extra accuracy **once per geometry** can be amortized. Candidate approaches include a more accurate overlap factorization/reconstruction, or one measured metric correction. For example, if `M=X0ᵀ S X0=I+E` with small E, `X1=X0(I−E/2)` reduces the metric defect to second order in exact arithmetic. This is a proposal, not a validated implementation. X1 need not be symmetric: the transformed Hamiltonian must then use **X1ᵀ H X1**, not blindly reuse XHX.

For the final occupied state, metric orthonormalization and a small projected eigensolve can repair some defects. Rayleigh quotients evaluated accurately can improve reported eigenvalues, but D/W/occupations must remain mutually consistent. These operations are not free and do not repair the wrong occupied subspace. Gauge their cost against one full Jacobi solve and use only after identifying the error source.

For sparse NS, accurate residual evaluation followed by a correction is useful only when the matrix/mask permits it. Storing the corrected Z back in one f32 value per element still imposes a representation limit. A truncated inverse additionally has a locality floor. Do not promise f64-quality operators from f32 storage in ill-conditioned cases.

### 5.5 Gamma/gamma-prime: reformulate cancellation before widespread f64

`gpu_forces.cl::gamma_prime_full_f32` now evaluates internally in f64. Its unequal-U expression subtracts large terms to obtain a small derivative; the prior audit reports ~3% error for an N–H two-atom f32 calculation and substantial improvement after promotion. That is a concrete justification for the existing f64 reference.

Kahan on the two final terms cannot restore errors already introduced by powers, denominators, and exponentials. Promising low-cost alternatives are:

- Precompute species-pair U-dependent coefficients outside geometry/force loops.
- Derive a stable symmetric expansion near equal U; separately handle the small-r cancellation regime.
- Where justified by bounds, use a controlled long-distance asymptotic form.
- If using tabulation, derive value and derivative from one smooth representation and validate both errors and boundary continuity.

Do not just enlarge the equal-U threshold until tests pass: that changes the evaluated model without a controlled truncation error. Near-coincident distinct atoms should produce an explicit invalid-geometry diagnostic, not disappear through a force-kernel `continue`.

The GPU gamma **value** implementation in `dftb_hamiltonian.cl` remains a separate f32 path. Dense SCC tests and the current `GpuDftb` upload host-computed G, so those tests do not certify on-device gamma values. Test gamma and its derivative together before changing the dataflow.

### 5.6 Coordinates and force accumulation

Dense pair H/S preparation uses relative displacements derived on the host, while gamma/repulsive kernels also consume absolute coordinates rounded to f32 and then subtract them. Translating a molecule far from the origin can therefore degrade one channel differently from another. A cheap candidate is per-replica recentering in host f64 before conversion, consistently across relevant channels. Test translation/rotation invariance over coordinate scales, not just near the origin.

The force kernels scatter through f32 atomic additions. Summation order can vary and large opposing force components can leave noisy small totals. Options are per-pair contributions followed by deterministic atom-owned gather, or atom-owned evaluation. Benchmark extra traffic versus atomic contention.

Do not implement “atomic Kahan” as independent atomics to sum and correction: that is not a coherent compensated update. For the electronic terms, combining non-SCC and shift contractions can also reuse derivatives and reduce separate atomic additions; validate the combined expression and all component diagnostics first.

## 6. Major performance bottlenecks, ranked provisionally

These are source-established costs and profiling priorities, not measured percentage allocations.

### 6.1 Tiled Jacobi already spends f64 in the expensive part

`gpu_tiled_jacobi.cl` uses f64 not only for rotation angles but also for compound-tile 2×2 updates, accumulated local rotation updates, and strip products updating both A and eigenvectors. Describing this as essentially f32 with a tiny scalar island understates the cost.

With default parameters, one pivot runs 20 inner sweeps × 63 rounds × three barriers = **3,780 inner barriers**, before surrounding load/update synchronization. An N=87 outer sweep visits three block pairs, yielding 11,340 such barriers. Inner work includes the padded 64-wide pivot. There is no inner convergence exit.

The declared local arrays total approximately **43,264 bytes per workgroup** with the defaults. Actual residency depends on device/resource limits and compiler output; one cannot infer occupancy from workgroup size alone.

First profile kernel time, outer sweeps, local/private memory, spills and batch utilization. Then benchmark adaptive inner work and f32/FMA or compensated strip updates against the current f64 reference. Do not lower sweeps without independent residual checks. This may offer accuracy **and** throughput improvement if excessive updates merely accumulate drift after useful convergence.

### 6.2 Sparse physics gates redo geometry-independent work inside SCC

`scc.rs::run_sparse_scc` calls `purify_h` each iteration. `purify_h` reconstructs full masks/matrices and recomputes S⁻¹ although S is fixed during that geometry's SCC. The host-roundtrip NS/TC2 APIs repeatedly allocate buffers, upload/download matrices, and build some kernels. Host TC2 also discards the KS product returned by `ksk` and recomputes it for the trace.

`repulsive_energy` is evaluated inside SCC although it depends only on geometry and species. Gamma geometry work is also repeated. Separate lifetimes: immutable SK/species/plans; per-geometry H0/S/G/Erep and inverse/factorization; per-iteration H/K/D/q. Reuse existing device-workspace products and reductions after validating their numerical contracts.

This is the largest structural obstacle to calling the current sparse physics harness an efficient GPU solver. It should be resolved before extrapolating tiny SiH4 timings to nanocrystals.

### 6.3 Both paths currently bring W construction back to the CPU

`GpuDftb::forces` downloads C/eigenvalues/occupations, constructs W with scalar host f32 loops, and uploads W. This is O(batch × N² × Nocc) work plus matrix traffic. The precision differs from tests that construct W in host f64.

Sparse `dw_from_k_padded` converts K/H to host f64 dense arrays and performs two cubic padded products, then unpads. For H-rich systems the four-orbital padding is especially wasteful. Existing sparse force workspace machinery is not what G3/F/G validates.

Prefer dense joint D/W construction from shared C loads, with a controlled accumulation policy; prefer the corresponding sparse products for K-based W. Share the physical interface, not the inappropriate dense implementation.

### 6.4 Dense geometry changes still rebuild static host data

`GpuDftb::set_coords` reconstructs fragments/templates and calls `GpuBatch::from_fragments`. That calls `pack_sk_tables`, including control fitting and allocations, although SK tables are static. `gamma_matrix` allocates a fresh dense host buffer; pair staging uploads full capacity; several explicit `finish()` calls serialize assembly.

The runtime and many device kernels/buffers are persistent—real progress—but this does not satisfy the no-allocation/no-static-rebuild hot-loop contract. Separate geometry-only packing from static preparation, retain host staging buffers, and profile actual live transfer lengths. Existing ordered species-pair keys must remain part of cache validity, not only block type/count.

### 6.5 Repeated finalization and batch-wide convergence waste work

`energy()` calls `finalize`; `forces()` calls it again. A caller requesting both at unchanged geometry/charges can redo the full electronic solve. Cache a validated final state with explicit geometry/charge revisions; invalidate it on changes.

Every SCC iteration runs every system and reads a convergence scalar back to the host. Heterogeneous batches pay for the slowest member. After reliable residuals exist, benchmark per-system active flags and less frequent diagnostic reads. Do not “freeze” a system using a pre-mix or stale residual.

### 6.6 Pair indexing, expensive gamma formulas, and force scatter

Gamma and repulsive force kernels decode each linear pair index by scanning atom rows. That adds O(n_atoms) indexing work to each pair—O(n_atoms³) indexing over all pairs for an otherwise quadratic pair calculation. Precomputed indices or direct row ownership remove this without a precision tradeoff.

Gamma's long-range contribution cannot simply be truncated like short-range repulsion. Any hierarchy/approximation needs its own physical error budget. Meanwhile cached coefficients, fewer duplicate distance evaluations, and an appropriate gather strategy are straightforward profiling candidates.

## 7. Remaining integration/physics issues outside precision tuning

These should not be hidden behind “f32 noise”:

- **Geometry/history invalidation:** `GpuDftb::set_coords` calls `set_geometry` but neither resets DIIS history. History reuse across different electronic maps is currently unqualified. Reusing charges is sensible; reusing residual histories needs a deliberate policy and validation.
- **Convergence contract:** `GpuDftb::scc` returns `Ok` with `stalled=true`; `relax` continues without enforcing that flag. Distinguish converged, diagnosed stagnation, exhausted, and invalid states. A caller-visible flag is insufficient if production consumers ignore it.
- **FIRE formula:** `fire_step` uses `alpha × |F_atom| × Fhat_atom`, which is effectively `alpha F_atom`, instead of velocity-norm-based mixing. It also shares power, dt and alpha across all replicas. This is not the stated standard independent-replica FIRE behavior. Batch coupling is algorithmic, not physical force coupling.
- **MD label:** `md_step` applies the old-force velocity increment, with no new-force second half-kick. A later SCC call does not supply that missing velocity update. Its displacement clipping and unit masses are also not physical velocity-Verlet MD. The smoke test proves only that one step stays finite.
- **Stale result metadata:** `relax` returns the initial SCC residual rather than the final one, and prints force and energy values from different points in the update sequence.
- **Occupation contract:** dense `n_occ` is rounded from half the reference electron total. Unsupported odd-electron/charged/spin cases need explicit rejection or an occupation model, not rounding into a different system.
- **Parameter fallback:** `per_atom_u` supplies 0.4 for missing onsite data. Unexpected missing parameters should fail with species/context.
- **Brief reader note:** the observed matsci Si=0/H≈0.4919 `q0` values are caused by taking trailing numeric fields from the onsite line, not evidence that standard SK occupations are non-valence data. Explicit sparse fixture valences bypass the parser. Treat this as a separate shared-input bug; it is not a GPU precision issue.
- **Interpolation method:** production value/derivative evaluation now uses the B-spline path. The appended-zero-sample tail is a boundary/model choice, not f32 error. The fitter's equations still interpolate original samples; the concern is between-sample/tail behavior and derivative boundary conditions, not automatically corruption of every tabulated value. Agree on a physical cutoff/continuity contract before changing the extra-control fit.

## 8. Focused experiment plan for the next implementation pass

Do these in order, retaining current baselines and reporting failures rather than relaxing criteria.

### A. Establish trustworthy frozen-input diagnostics

Use representative N=6, 28, 64/65, 87 and larger dense matrices; use sparse full-mask and genuinely truncated cases with varying row degree and gap. Freeze exact GPU-rounded inputs and compare against f64 evaluation of **those inputs**.

Record overlap spectrum, H/S symmetry, direct eigen/inverse/projector residuals, occupation, electron count, final unmixed SCC residual, and all energy components. For forces, feed identical D/W/V into both contractions before comparing self-consistent paths. Include nonfinite and wrong-rank negative controls.

### B. Isolate mixer, representation, and arithmetic changes

Compare current DIIS against scaled/rank-aware f64 small-system DIIS, with an unchanged electronic kernel. Then compare f32 FMA/multi-accumulator/compensated/f64 accumulation for one kernel at a time. Preserve returned energy in f64 for all diagnostic finite differences.

Measure whether improved arithmetic changes residuals of the frozen problem, SCC iteration count, final physical error, or only printed precision. A lower scalar residual without a better independently reconstructed operator is not success.

### C. Separate finite differences from convergence

Sweep at least h, h/2, h/4 over a useful initial interval, independently tightening SCC and electronic tolerances while possible. Look for second-order truncation followed by a noise plateau. If a target is unattainable, show the curve and the limiting stage; do not assert impossibility from one stencil.

Measure energy **differences** along relevant H-bond/geometry scans and their force consistency. A systematic absolute bias might cancel, but that cancellation is currently unestablished. Test translation, rotation, atom permutation, and batch composition invariance. For vibrations compare at a common geometry for implementation parity and at each method's own stationary geometry for physical spectra.

### D. Benchmark the actual solver lifetime

Use release builds, `OPENBLAS_NUM_THREADS=1`, and a verified hardware GPU. Record device, driver, build flags, source revision, matrix dimensions, mask density and batch size. Separate compilation/setup, geometry assembly, overlap preparation, SCC, finalization, D/W, and force evaluation.

Use device event timings and synchronized end-to-end time. Count allocations/object builds, launches, host synchronizations and transferred bytes on the executed path—not dummy statistics. Benchmark batch=1 and representative many-system batches. The optimization objective is **wall time to an accepted physical accuracy**, not f32 kernel FLOP/s or a green smoke test.

## 9. Recommended decision

**Do not choose all-f64, and do not accept the present “do not chase below this floor” statements as established numerical limits.**

The first implementation package should combine trustworthy final-state/residual contracts with two cheap precision changes: f64 scalar energy output and a reliable small DIIS solve. Keep the main matrix representation f32. Next, profile the existing broad-f64 tiled Jacobi path and repeated host work, then selectively introduce compensated block accumulation or amortized metric repair where frozen-input evidence warrants it. Retain f64 gamma-prime until a stable, physically consistent replacement is verified.

This approach addresses avoidable errors without paying the user's approximately 40× f64-throughput penalty across the dominant matrix workload. The exact speed/accuracy winner remains to be measured; the current evidence supports this ordering, not a universal optimal kernel.

---

## Addendum (2026-09-10, after Package 1–2 GPU runs — not part of the inspect)

The inspect above did **not** run GPU. These numbers are NVIDIA RTX 3090 `--release`, `dftb_engine` + `scripts/test_gpu_dftb_{molecules,measure}.rhai`.

**Already implemented from §3.1 / §9 before this addendum:** f64 energy totals; DIIS RMS + n<2 α-mix + f64 GE + fail-loud fallback; GPU DIIS hist `min(10,n_atoms)` (H2O hist=3, 6–8 iters). Those §3.1 defects are **closed** on the production kernel. AT/GC still stall ~25 iters (rms ~1.3e-6) — mixer is honest, not “broken then silent.”

**Package 2 refutes the old F3 story** (`max|δε|=2.5e-5` from SCC-then-compare). Frozen rounded `H_scc`/`S`: AT `max|δε_occ|=1.1e-6`; `|dE|~3e-5` tracks `δ_CH=E_band−2ΣCᵀHC` (~5e-5). Density `δ_D~1e-7`. Forces vs CPU ~1e-6 (H2O) / ~5e-6 (AT).

**Löwdin Newton** (`repair_lowdin_x`, skip if `e1≥e0`): AT `||XᵀSX−I||` 2.9e-6→2.0e-7; leftover metric is Jacobi `||C'ᵀC'−I||~2e-6`. Formic z-scan `|ΔE_gpu−ΔE_cpu|` 1.54e-5→4.6e-6. **Kept.**

**f32 Kahan in `batched_gemm`** (not f64 GEMM): H2O/formic bit-identical to Newton-only. AT `δ_CH` 4.71e-5→5.03e-5 (not the target). `|dE|` 3.36e-5→2.79e-5 is cancellation. No DIIS blowup. **Kept as cheap; leftover is `C'`, not GEMM.**

**Bench** (`gpu_scc_bench.rs`, legacy `gpu_solve_scc_batched_diis_warmstart`, **not** `GpuDftb`): formic N=28 batch=1/100 = 0.96 / 1.48 ms per SCC iter (gemm 0.18 / 0.27). AT N=87 “389 / 788 ms” is **host occ_sort+upload** (388 / 786 ms); jacobi+gemm stay ~0.2 ms. Do not quote AT wall as GPU.

**Still open:** AT stall; Jacobi residual/stop reason; occupied-subspace if `δ_CH` must drop; origin recenter; FIRE `|v|`; interpolator fitter. Do not require AT `|dE|<1e-5`. Do not re-enable f64 GEMM.

SSOT: `../topical_audit/f32_floor_dense_hbond.md` §3.1; manifest §0.6; roadmap §6.4.

## Appendix: primary source map and provenance

Read these symbols first:

- Dense state/energy: [gpu_scc_plan.rs](../../../rust_dftb/src/qmqm/gpu_scc_plan.rs), `scc_step_diis`, `finalize`, `compute_energy`, `set_geometry`.
- Dense arithmetic: [gpu_matrix_ops.cl](../../../rust_dftb/src/qmqm/gpu_matrix_ops.cl), `diis_step_batched`, density/Mulliken/dot kernels; [gpu_tiled_jacobi.cl](../../../rust_dftb/src/qmqm/gpu_tiled_jacobi.cl); [gpu_eigen.cl](../../../rust_dftb/src/qmqm/gpu_eigen.cl).
- Dense lifetime/forces: [gpu_dftb.rs](../../../rust_dftb/src/qmqm/gpu_dftb.rs), `set_coords`, `forces`, `scc`, `fire_step`, `md_step`; [gpu_prep.rs](../../../rust_dftb/src/qmqm/gpu_prep.rs), `from_fragments`, `pack_sk_tables`; [gpu_forces.cl](../../../rust_dftb/src/qmqm/gpu_forces.cl), force and gamma kernels.
- Sparse arithmetic: [gpu_sparse.rs](../../../rust_dftb/src/methods/sparse/gpu_sparse.rs), NS, TC2, reduction and workspace routines; [sparse_bsr4_purification.cl](../../../rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl), SpGEMM, direct residual, TC2 and Hscc kernels.
- Sparse physics/lifetime: [scc.rs](../../../rust_dftb/src/methods/sparse/scc.rs), `purify_h`, `run_sparse_scc`; [sparse_forces.rs](../../../rust_dftb/src/methods/sparse/sparse_forces.rs), `dw_from_k_padded`; [sparse_system.rs](../../../rust_dftb/src/methods/sparse/sparse_system.rs), `run_scc`.
- Tests: [gpu_hbond_physics.rs](../../../rust_dftb/tests/gpu_hbond_physics.rs), [gpu_sparse_bsr4.rs](../../../rust_dftb/tests/gpu_sparse_bsr4.rs), [gate_g3_energy.rs](../../../rust_dftb/tests/gate_g3_energy.rs), [gate_g_hessian.rs](../../../rust_dftb/tests/gate_g_hessian.rs), [gpu_dftb.rs](../../../rust_dftb/tests/gpu_dftb.rs).

Selected SHA-256 fingerprints at the final source inspection, for identifying whether a finding predates another agent's change:

```text
gpu_scc_plan.rs       d09792b45901decd38d12bd717c2b8ea84b36f6e5fbc32230ec669837422c576
gpu_matrix_ops.cl     7cf3902f709c1b8a0799db18ce006b5ca21313c1ebf361e2d14f915fc9930fdd
gpu_dftb.rs           9e7d19776ca963e7182e3b6c4b1e7409ba23c306e5706540d5e0c6a1d5766cd1
gpu_tiled_jacobi.cl   b5b61a03339edf5a9b3fa2f0f76bfd8233460c693b0c17c273098d9bf68768d0
gpu_sparse.rs        7d9ba1774a295ca86b9e3971ffc87cf2e0d8c2c8b8a2851482cb221907689a2b
gate_g_h_sparse.csv  041e2a428524f9ef49c99d76db22009b1abf4a439986a1ff0a41497d15d68972
gate_g_h_dense.csv   75c1c6c0275c0b2741a8336d1b0fbad47803892f5c9a26ea7206956d2d14654d
```

No finding is marked resolved by this report. Fresh GPU evidence and user confirmation remain necessary for implementation status changes.
