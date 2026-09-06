# GPU optimization architecture review

**Date:** 2026-09-06  
**Scope:** read-only review at HEAD `1b368776`; no code changes or GPU benchmarks  
**Sources:** `GPU_Optimization.chat.md`, current Rust/OpenCL sparse path, and the
NumericalMathPlayground linear-scaling notes

This review separates confirmed current behavior from experiments that still
need measurement. The largest opportunity is architectural: the production
large-system entrypoint still uses the old dense and host-roundtrip path even
though a device-resident prototype exists.

## Current large-system path and confirmed bottlenecks

`rhai_run_sparse_purify` first obtains a stored dense SCC or non-SCC result, creates a full
atom-block mask, and converts dense `H` and `S` to BSR4 at
[`rust_dftb/src/bin/dftb_engine.rs:275`](/home/prokophapala/git/dftbplus/rust_dftb/src/bin/dftb_engine.rs:275).
It then calls the old `newton_schulz_inverse` and `tc2_purify` methods at
[`dftb_engine.rs:294`](/home/prokophapala/git/dftbplus/rust_dftb/src/bin/dftb_engine.rs:294)
and [`dftb_engine.rs:315`](/home/prokophapala/git/dftbplus/rust_dftb/src/bin/dftb_engine.rs:315).
The resident `SparsePurifyWorkspace` is only used by tests. Consequently the
actual route retains dense SCC diagonalization, dense matrix storage, full
`N_atom²` block storage, host/device copies, and a final dense expansion at
[`dftb_engine.rs:343`](/home/prokophapala/git/dftbplus/rust_dftb/src/bin/dftb_engine.rs:343).
It is therefore not yet an end-to-end linear-scaling solver.

The old TC2 loop also duplicates products. At
[`gpu_sparse.rs:1058`](/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/gpu_sparse.rs:1058)
it forms `KSK`, then recomputes `KS` for the trace at line 1065, and
`tc2_step` recomputes both products again at lines 753 and 757. This is five
SpGEMMs per iteration before counting uploads, downloads, allocations, and
`finish()` calls. The resident prototype reduces the mathematical iteration to
two products, but that improvement is not connected to production.

The resident code still has hot-loop allocations. `trace_ks_dev` allocates a
partial buffer at [`gpu_sparse.rs:1491`](/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/gpu_sparse.rs:1491),
and `reduce_to_one_dev` allocates each recursive output at line 1520. Each
TC2 step allocates a host vector and reads the trace back before writing it to
the device at lines 1771–1773. This is a synchronization point and defeats the
claim that all reduction state is persistent. A device reduction result can be
consumed directly by the branch kernel on the same ordered queue; convergence
diagnostics should be less frequent or remain device-side.

The resident convergence path recomputes `KSK` for every requested residual
check at [`gpu_sparse.rs:1829`](/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/gpu_sparse.rs:1829),
adding two products. The existing `Q` and `K` before the update already give a
valid residual for the current state; reorganizing the loop to test before
the update, or using the next iteration's two products, can remove this
duplicate diagnostic work without weakening the stopping criterion. More
seriously, divergence handling returns the current
`K` while reporting the best residual and iteration at lines 1857–1861. This
must become explicit nonconvergence with a matching matrix and residual, or a
device-side backup of the best matrix. Returning a state that does not match
the reported diagnostic is unsafe for physical results. A backup may support
diagnostics or a deliberate caller-selected fallback, but failure to reach the
requested tolerance must remain an explicit nonconvergence status, not an
alternate successful `Ok` result.

The Newton–Schulz residual has a numerical cancellation failure. Lines
1610–1614 evaluate `||T||² - 2 Tr(T) + N` in `f32`. For `T=(1+10^-4)I` and
`N=16`, direct `f32` arithmetic gives `||T||=4.0004`, `Tr(T)=16.0016`, a
computed squared residual of zero, while the true normalized `||I-T||` is
`10^-4`. Residual convergence must use a direct reduction of `I-T`, a more
stable reformulation, or higher-precision accumulation. The current reported
`R_Z=0` is not sufficient evidence of inverse accuracy.

The kernel reserves local storage for `MAX_LEFT_BLOCKS=256` in every row. Rows
above the compiled limit return from OpenCL; the host check now fails early,
which is correct, but the default full mask cannot support large systems. A
degree-bucketed family (for example 32/64/128/256) is a later optimization
after residency and physical masks are working. The current fixed 256-row
allocation also wastes local memory for narrow ribbons.

## Sparse representation and physics limits

`build_full_mask` at [`bsr4.rs:177`](/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/bsr4.rs:177)
destroys sparsity. Even the intended geometric mask at lines 156–174 is an
all-pairs `O(N²)` construction and must eventually use a cell list or retained
neighbor topology. `build_product_mask` at lines 302–321 uses
`Vec::contains` inside nested neighbor loops; a marker array or sorted merge
would make symbolic construction proportional to the generated product.
More important than this one-time cost is that the same sorted-list
intersections are repeated on every SpGEMM. A precomputed symbolic plan with
packed operand offsets is a credible experiment once the resident baseline is
measured; its cost should be counted in candidate block contributions and
stored plan bytes, not inferred from final `nnz` alone.

The supposedly one-time spectral setup can also densify. `gershgorin_bounds`
calls `b.to_dense()` at [`bsr4.rs:353`](/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/bsr4.rs:353),
and `inf_norm` does the same at line 379. Each allocates an orbital-level
`N²` array and then scans it. A sparse row/block reduction should replace these
before claiming large-system scaling.

The BSR4 path assumes four orbitals per atom. The production check at
[`dftb_engine.rs:265`](/home/prokophapala/git/dftbplus/rust_dftb/src/bin/dftb_engine.rs:265)
rejects hydrogen-containing systems, despite passivated ribbons being a target.
Padded BSR4 (one active H orbital and three dummy slots) is a reasonable first
mixed-basis experiment; a variable-block CSR is more general but introduces
irregular kernels. Any padded formulation must exclude dummy orbitals from
electron counts, bounds, Mulliken output, and residuals.

TC2 truncation needs physical acceptance criteria. Idempotency and
`Tr(KS)=N_occ` do not prove that the occupied subspace is correct. Purification
is a polynomial in the initial metric projector and does not repair a wrong
occupied eigenspace. The initial `K0` uses an approximate, truncated `Z≈S⁻¹`
at [`gpu_sparse.rs:904`](/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/gpu_sparse.rs:904);
its eigenvectors can already be wrong. Every cutoff experiment therefore needs
energy, charge, `R_I`, Hamiltonian commutator `R_H=||HKS-SKH||`, symmetry, and
force parity where forces are enabled. Small-gap ribbons may require growing
support or finite-temperature occupations; fixed-radius linear scaling is not
guaranteed at zero temperature.

The sparse route currently has no scalable SCC loop. It consumes an already
converged dense SCC Hamiltonian. Long-range DFTB gamma interactions are dense
at atom level, so a real large-system path needs an FMM, H²/treecode, or
periodic FFT/Ewald-style charge-potential evaluator. Sparse H/S products alone
cannot make SCC linear scaling.

The Davidson implementation is also dense despite its sparse placement:
[`davidson.rs:117`](/home/prokophapala/git/dftbplus/rust_dftb/src/methods/sparse/davidson.rs:117)
uses dense `DMatrix` products and allocates expanded subspaces each iteration.
For frontier-only output, BSR4 SpMM plus Chebyshev filtering, or Davidson with
a real sparse preconditioner, is a separate route from full density
purification.

## Recommended architecture experiments

For independent tiny systems, keep one persistent GPU workspace per topology
and batch systems through block-local SCC kernels. Fuse local SCC work only to
the resource limits of the device; query local memory, registers, and preferred
workgroup sizes instead of assuming one block size. Remove unnecessary density
construction by forming occupied coefficients and direct occupied Mulliken
contractions, and precompute the geometry-invariant transformed `H0` where the
mathematics permits it. Preserve eigen/subspace guesses, mixer history, and
response state across neighboring geometries. Validate every shortcut with
actual residuals and parity, not timing alone.

For small gapped systems requiring only charges and energy, a dense orthogonal
projector SP2 experiment could replace full eigensystems with regular batched
GEMMs. The existing `gpu_matrix.rs:339` purification building block has
host-side per-batch control; its comment that trace is exactly preserved is
not generally true for TC2. Measure it against warm-started Jacobi including
overlap setup, back-transform, and charge costs. Repurifying a previous
projector after `H` changes has the same invariant-subspace trap as sparse
TC2, so the new Hamiltonian must enter a mathematically valid initial
projector. The dense path also reads an `N²` matrix merely to obtain a diagonal
at `gpu_scc.rs:514`, and the GPU eigensolver starts each solve from `V=I` at
`gpu_eigen.cl:149`; both are confirmed dataflow targets for review.

The useful accounting identity is

```text
total work = iterations × (eigensolve + other SCC work) + setup/transfer overhead.
```

If an experiment reduced iterations from 24 to 6 while per-iteration work was
unchanged, the iteration component would fall by 4×. That is a conditional
example, not a speed prediction; convergence and overhead must be measured.

For large gapped systems, a strong experiment is graph core-plus-halo local
matrix-function problems batched on the GPU. Reuse the small-system kernels,
but derive halos from the density/metric support, maintain a common global
chemical potential and electron count, and let only core rows own output. The
halo must be enlarged until energy, charge, and force results converge. A local
inverse of `S` is not the corresponding submatrix of the global inverse, so
boundary treatment requires an explicit derivation and parity tests. This is
the domain-decomposition direction discussed in [arXiv:1603.00937](https://arxiv.org/abs/1603.00937);
it is an experiment, not a guaranteed tenfold gain, and it cannot omit global
SCC electrostatics.

For long fixed-width, small-gap ribbons, pole expansion with block-banded
selected inversion is another substantial architectural candidate. Factor
`H-(mu+z)S` and compute only density entries needed by H/S and force support,
without constructing `S⁻¹` or full `K`. Forces generally require both the
density matrix and the energy-weighted density on the needed derivative
support, so selected entries for only one matrix are insufficient. At fixed
width this can scale as `O(L b³)` for fixed pole count, but width, complex
poles, chemical-potential search, and numerical factorization costs matter.
Reuse symbolic ordering, not numeric factors, as geometry changes. See the
[PEXSI introduction](https://pexsi.readthedocs.io/en/latest/introduction.html).

Before treating `P=KS` as an automatic 2× improvement, compare actual support,
block-triple counts, and kernel costs. `P` is nonsymmetric, so it cannot use
the existing symmetric-right kernel unchanged; a pretranspose strategy would
add its own storage and work. Its product mask may also be substantially wider
than K. First compare dense and sparse support/error
against metric TC2. Mask radius cannot be chosen as a universal multiple of
the S radius; it must converge against gap, temperature, dimensionality, and
the requested observables. Also test SP2
scale-and-fold spectral bounds and block-product screening with a stated error
budget; relevant references are [Rubensson and Niklasson on SP2 purification](https://arxiv.org/abs/1302.7292)
and [Artemov and Rubensson on sparse matrix-function screening](https://arxiv.org/abs/2005.10680).

For MD specifically, upstream XL-BOMD may remove many inner SCC iterations:
`src/dftbp/md/xlbomd.F90` and `extlagrangian.F90` are the implementation points.
Force correction, energy drift, and time-step convergence are mandatory, and
XL-BOMD is not a replacement for converged static SCF. The workflow is
described in the [DFTB+ XL-BOMD recipe](https://dftbplus-recipes.readthedocs.io/en/stable/moleculardynamics/xlbomd.html).

The cost-aware order is: integrate and measure the resident sparse path;
replace full masks and dense bounds; make reductions persistent and diagnostics
numerically stable; add mixed-basis and long-range SCC support; then compare
symbolic plans, P-SP2, core-halo methods, and selected inversion. No speedup
claim should be made until wall time, device events, memory, and physical
parity are measured on representative gapped and small-gap systems.

The NumericalMathPlayground material is useful as design input, not as a
production solver. `DensityMatrix/GF.py:68-91` performs dense
`np.linalg.solve` per probe and pole; factor reuse or selected inversion would
be required for scale. `FastDirectSolvers/nested_solver.py:1214-1252` retains a
full-size dense root `eigh`, while `CheFSI/CheFSI.py:103-110` currently invokes
its kernel with the Chebyshev flag disabled and duplicates an input argument.
These are prototype limitations, not reasons to reject the underlying
algorithms, and should remain explicit when selecting an implementation.
