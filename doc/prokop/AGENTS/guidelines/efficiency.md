# Efficiency Guidelines — DFTB+ Rust Reimplementation

## Why this document exists

**Performance is not an afterthought here. It is the primary motivation for
this entire project.** DFTB+ already works — the physics is solved, the
Fortran code is correct, and it produces the right numbers. We are not
changing the physics. We are reimplementing it in Rust + OpenCL to make it
**faster**: batched many-small-system computation on consumer GPUs, for scans,
NEB, MD ensembles, and QM/QM fragment coupling that are impractically slow in
the original.

Therefore the whole architecture must revolve around efficiency. It is a
primary concern in the design, ground up — not something to bolt on after the
code "works." Every architectural decision (data layout, memory lifetime,
algorithm choice, GPU/CPU split, batching strategy) is made with performance
as a first-class constraint. Code that is correct but does unnecessary work is
a design failure, not a "premature optimization."

**The only thing more important than speed is scientific rigor and
debuggability.** We can sacrifice accuracy in a controlled way and we do
(e.g. single precision f32 on GPU), but we must be sure that fundamentally
everything is physically sound and that the accuracy compromises are
well-understood and documented. A fast wrong answer is useless. A slow wrong
answer is also useless but at least won't be mistaken for success. The
priority order is:

1. **Physical correctness** — the math must be right, conservation laws
   respected, parity with the Fortran reference established.
2. **Debuggability** — when something is wrong (and it will be), we must be
   able to locate the bug without re-running. Gated prints, informative
   errors, parity tests, trajectory dumps.
3. **Performance** — once correct and debuggable, make it fast. But design
   for performance from the start; don't build an architecture that is
   fundamentally wasteful and then try to optimize it later.

These three are not in tension when the architecture is right. A clean
three-tier data lifetime (static / per-geometry / per-iteration) is both
faster *and* more debuggable than rebuilding everything every step, because
the state is explicit and inspectable rather than implicit and ephemeral.

---

## The rules

Concrete rules derived from a code review (see
`doc/prokop/chats/CPU_Optimization.chat.md`) that found 18 categories of
performance and correctness problems in the CPU DFTB pipeline. These rules are
specific to this repo but generalize to any numerical computing project.

**Every rule here was violated.** The violations are documented with concrete
examples so the pattern is recognizable.

---

## Rule 1: Three-tier data lifetime

**Separate data by how often it changes. Never rebuild data from a slower tier
inside a faster tier.**

| Tier | Changes | Examples | When to build |
|------|---------|----------|---------------|
| **Static** | Never (per system) | SK tables, species mapping, orbital offsets, q0, repulsive spline params, pair-table LUT | Once, at load time |
| **Per-geometry** | Every geometry step | H0, S, Cholesky L, gamma matrix G_AB, gamma' derivatives, neighbor list, H0' = L⁻¹H⁰L⁻ᵀ | Once per geometry update |
| **Per-SCC-iteration** | Every SCC iteration | H_scc, eigenvalues, eigenvectors, charges, V = G·Δq | Inside SCC loop only |

**Violation**: `HamiltonianBuilder::build_scc()` rebuilds the entire solver from
scratch every geometry step — new `HamiltonianBuilder`, new
`FragmentTemplate` (which deep-clones `SkData` including HashMaps and
interpolation tables), new `GammaTable`, new `FragmentNeighborList`, new
`DiisMixer`, new `MultiSystemSolver`. Static data is cloned and topology is
rebuilt inside the geometry loop.

**Fix**: Persistent `DftbCpu` struct that owns all static data once. Geometry
updates only rebuild per-geometry data. SCC only touches per-iteration data.

```rust
struct DftbCpu {
    // static — built once
    ctx: SystemContextOwned,
    gamma_params: GammaTable,
    repulsive: RepulsiveTables,
    q0: Vec<f64>,
    norb: usize,
    // per-geometry — rebuilt on update_geometry()
    h0: Vec<f64>, s: Vec<f64>, l: Vec<f64>,
    gamma: Vec<f64>, gamma_prime: Vec<f64>,
    h0_prime: Vec<f64>,  // L⁻¹H⁰L⁻ᵀ, precomputed
    // per-SCC-iteration — workspace, reused
    q: Vec<f64>, v: Vec<f64>, h_prime: Vec<f64>,
    eig: Vec<f64>, c: Vec<f64>, sc: Vec<f64>,
    work: Vec<f64>, iwork: Vec<i32>,  // LAPACK workspace
    mixer: DiisMixer,
}
```

---

## Rule 2: No allocation in hot loops

**The hot loop (SCC iteration, force evaluation, geometry step) must not
allocate. All `Vec`, `String`, `HashMap`, `Box` creation must happen during
initialization or geometry update — never inside `solve_scc`, `forces`, or
per-iteration code.**

This is already in `AGENTS.md` under Performance, but it was violated in:
- `build_all_h_scc()`: `let offsets: Vec<usize> = ...` every SCC iteration
- `DiisMixer::mix()`: `q_in.to_vec()`, `residual.to_vec()` every DIIS step
- `repulsive_force()`: fresh `HashMap` every force call
- `FragmentTemplate::new()`: `sk.clone()` deep-clones HashMaps + tables

**Fix**: Precompute offsets in the solver struct. Use fixed ring buffers for
DIIS (`q_hist[max_hist * natom]`, not `VecDeque<Vec<_>>`). Parse repulsive
splines into flat arrays at load time.

---

## Rule 3: No strings, HashMaps, or clones in hot paths

**Parameter lookup in the hot path must use integer indices into flat arrays.
Strings, HashMaps, and `.clone()` are for initialization, not computation.**

**Violation**: `repulsive_force()` does:
```rust
let key = (species_names[i].clone(), species_names[j].clone());
let spline = repulsive_map.get(&key).unwrap().clone();
```
This creates two `String` allocations, a HashMap lookup with string hashing, and
clones a `RepulsiveSpline` containing `Vec`s — **per atom pair, per force call**.

**Fix**: At load time, build `repulsive: Vec<Option<RepulsiveSpline>>` indexed
by `pair_type = species_i * n_species + species_j`. Hot path becomes:
```rust
let spline = &repulsive[pair_type];  // array index, no string, no clone
```

---

## Rule 4: Don't recompute what was already computed

**If a quantity was already computed in a previous step or function, reuse it.
Never compute the same thing twice.**

**Violations**:
- `compute_scc_forces()` **re-diagonalizes** H_scc to get eigenvectors that the
  final SCC iteration already produced. Pure duplicated electronic solve.
- `non_scc_electronic_force()` and `scc_shift_force()` both call
  `pair_block_derivative()` for the same pair/direction — **6 SK block
  evaluations × 2 = 12** instead of 6.
- `build_scc()` computes `Tr(D·H0)` by building the full N×N matrix product
  (`(&density * &frag.template.h0).trace()`) — O(N³) for a quantity that is
  O(N²) via Frobenius contraction `Σ D_μν H⁰_μν`.
- Every SCC iteration recomputes `gamma(r)` distances and exponentials even
  though geometry doesn't change during SCC.

**Fix**: Pass eigenvalues/eigenvectors/D/W from SCC directly to forces. Fuse
derivative evaluations. Use elementwise trace. Precompute gamma matrix once
per geometry.

---

## Rule 5: Use the right algorithm — verify, don't assume

**Before stating what algorithm a library uses or why it's slow, read the
source. Before choosing an algorithm, understand its complexity class and
when it applies.**

**Violation**: I claimed `nalgebra::SymmetricEigen` uses the Jacobi algorithm
with "N² rotations per sweep, 10-20 sweeps, inner-loop allocations, sequential
ordering." **This was completely fabricated.** nalgebra 0.33 uses Householder
tridiagonalization (`SymmetricTridiagonal::new`) followed by implicit shifted
QR with Givens rotations and Wilkinson shift. I never read the source. The
entire `eigensolver_performance.md` explanation is built on a false premise.

The empirical observation (20ms → 0.7ms with LAPACK) was correct, but the
explanation was wrong. The real reason nalgebra is slower than LAPACK is that
LAPACK uses optimized BLAS-3 operations and architecture-specific tuning,
while nalgebra is a pure-Rust implementation without BLAS acceleration.

**Fix**: When a library is slow, read its source before explaining why. When
choosing a replacement, use the right routine for the whole problem:
`dpotrf` (Cholesky) + `dsygst` (generalized→standard transform) + `dsyevd`
(eigensolve) + `dtrsm` (back-transform), not just `dsyevd` with nalgebra
triangular solves in between.

---

## Rule 6: Benchmark in release mode

**Never compare timing of unoptimized Rust (`opt-level=0`) against optimized
native libraries. Always benchmark with `--release` or a profiling profile.**

**Violation**: All reported timings used `cargo run --example hbond_ref`
without `--release`. The `[profile.dev]` section sets `debug = 1` but does NOT
set `opt-level`, so Cargo defaults to `opt-level=0`. This means:
- Rust/nalgebra/SK/force loops: compiled at `-O0`
- LAPACK/OpenBLAS `dsyevd`: optimized native library

The comparison was misleading. The 20ms "nalgebra eigensolve" and 750ms/step
numbers are all contaminated by debug compilation.

**Fix**: Always use:
```bash
OPENBLAS_NUM_THREADS=1 cargo run --release --example hbond_ref -- ...
```
For small matrices, `OPENBLAS_NUM_THREADS=1` avoids multithreaded BLAS overhead.
For profiling, use a dedicated profile:
```toml
[profile.perf]
inherits = "release"
debug = 1
debug-assertions = true
overflow-checks = true
lto = false
```

---

## Rule 7: Exploit mathematical structure

**Don't compute more than the math requires. If a formula has structure
(symmetry, factorization, sparsity), exploit it.**

**Violations and fixes**:

1. **Trace of product**: `Tr(D·H⁰)` builds full N×N product for a trace.
   Fix: `Σ_μν D_μν H⁰_μν` — O(N²) Frobenius contraction.

2. **Mulliken charges**: Computes `(D·S)_μμ` via triple loop. But after
   Cholesky diagonalization, `C = L⁻ᵀY`, so `SC = LLᵀL⁻ᵀY = LY`.
   Fix: `SC = L·Y` (one `dtrmm`), then `p_μ = 2Σ_k C_μk (SC)_μk`.

3. **SCC Hamiltonian transform**: Every SCC iteration does full
   `L⁻¹H_sccL⁻ᵀ` (two triangular solves). But `H_scc = H⁰ + ½(SV+VS)` where
   V is diagonal, and `H⁰' = L⁻¹H⁰L⁻ᵀ` is geometry-invariant.
   Fix: Precompute `H⁰'` once. Per SCC iteration:
   `X = L⁻¹VL` (one triangular solve), `H' = H⁰' + ½(X+Xᵀ)`.
   Halves the transform work.

4. **Full density matrix during SCC**: D is only needed for forces, not for
   charges or energy. Fix: Build D and W once after convergence using
   `dsyrk` and `dgemm`.

---

## Rule 8: Precompute polynomial coefficients, don't Neville at runtime

**For repeated interpolation on a fixed grid, precompute polynomial
coefficients at load time. Runtime becomes Horner's method (~7 FMAs), not
Neville recursion (nested loops + stack matrices).**

**Violation**: `eval_eqgrid_new_into()` does 8-point Neville interpolation at
runtime, with `xa[8]` and `yb[20][8]` stack arrays and nested loops. This is
called for every shell pair (ss, sp, ps, pp) of every atom pair — ~4× repeated
interpolation per pair.

**Fix**: At SK-loading time, for each interval `[i, i+1)` and each channel,
compute degree-7 polynomial coefficients `a_0..a_7` in the local variable
`t = (R - R_i) / ΔR`. Runtime:
```rust
let t = (r - r_start) / dr;
let v  = horner(coeffs, t);       // 7 FMAs
let dv = horner_deriv(coeffs, t); // 7 FMAs, derivative for forces
```
No Neville recursion. No stack matrices. Derivative comes for free.

---

## Rule 9: Evaluate all channels at once

**When a function evaluates multiple related quantities (e.g., all shell
integrals for a species pair), evaluate them in one call, not one per
channel.**

**Violation**: `rotate_diatomic_block_into()` calls
`eval_shell_integrals_into()` separately for each shell pair (ss, sp, ps, pp).
Each call does a full `h.eval_into(r)` + `s.eval_into(r)` interpolation. For
sp atoms, the same interpolation is repeated ~4× per pair.

**Fix**: One `eval_all(r, Hsk[], Ssk[])` call per species table/pair, then
rotate all shell pairs from the cached results.

---

## Rule 10: Analytic derivatives, not finite differences

**When analytic derivatives are available and simple (e.g., sp basis sets),
use them. Reserve finite differences for parity testing only.**

**Violation**: Force calculation does 6 complete SK block evaluations per pair
(x±, y±, z±) via finite differences. For an sp basis, analytic derivatives are
closed-form:

```
∂R/∂R_a = u_a
∂u_i/∂R_a = (δ_ia - u_i·u_a) / R

H_ss = V_ssσ(R)     →  ∂_a H_ss = V'_ssσ · u_a
H_sp_i = u_i·V_spσ  →  ∂_a H_sp_i = [(δ_ia - u_i·u_a)/R]·V_spσ + u_i·V'_spσ·u_a
H_p_ip_j = V_π·δ_ij + (V_σ-V_π)·u_i·u_j  →  (closed form, see chat)
```

**Fix**: One pair evaluation returns `H[16], S[16], dHdx[16], dHdy[16],
dHdz[16], dSdx[16], dSdy[16], dSdz[16]` with **one radial interpolation**.
Keep finite difference only as a parity test.

---

## Rule 11: Check units

**Verify unit consistency across all code paths, especially when sharing
parameters (like cutoffs) between functions.**

**Violation**: Hamiltonian construction converts coords to Bohr before
neighbor search (correct — SK cutoff is in Bohr). But the force code passes
coords in Å to `NeighborBuilder` with the same Bohr cutoff. A ~10 Bohr cutoff
is interpreted as ~10 Å (~1.89× too large), including vastly too many pairs
in the expensive force calculation.

**Fix**: Build one neighbor/pair list per geometry and share it between H/S
construction and forces. Store pair data (i, j, R, u, table index) once.

---

## Rule 12: Warm-start iterative solvers

**When iterating on a sequence of related problems (geometry steps, scan
points), use the previous solution as the initial guess. Reset only what
must be reset (e.g., DIIS history), not the entire state.**

**Violation**: Every geometry step starts SCC from q=0 (neutral charges),
requiring 16-17 iterations. With warm start from the previous converged q,
this would be 3-4 iterations.

**Fix**: Persistent state (Rule 1) solves this naturally. Keep `q` from
previous geometry. Reset DIIS history (the history is invalid for a new
geometry) but keep the charges. Optionally use a predictor: `q_guess = 2q_n
- q_{n-1}`.

---

## General principles (abstracted from the above)

### G1: Before optimizing, eliminate unnecessary work

The biggest speedups come not from making existing work faster, but from
**removing work that shouldn't exist**. Re-diagonalizing in forces, rebuilding
the solver every step, recomputing gamma every iteration — these are not
"slow operations to optimize," they are **unnecessary operations to delete**.

### G2: Review code for efficiency before measuring

Don't wait for profiling to find architectural problems. Read the code and
ask: "What does this do? Does it need to exist? Was this already computed
somewhere else? Is there a cheaper way to get the same result?" Profiling
finds hot spots; code review finds unnecessary work.

### G3: Verify claims before stating them

Before writing "nalgebra uses Jacobi" in a document, read the nalgebra source.
Before writing "this takes 20ms because of X," verify X is actually happening.
Unverified claims propagate into wrong decisions and erode trust.

### G4: Separate parameter data from geometry data from iteration data

This is the three-tier lifetime rule (Rule 1) generalized. Parameter data
(SK tables, repulsive splines, Hubbard U) is read once. Geometry data (H0, S,
gamma, Cholesky) is computed once per geometry. Iteration data (charges,
eigenvectors, H_scc) changes every iteration. Never mix the tiers.

### G5: The right data structure for the hot path is a flat array indexed by
integers

Strings, HashMaps, Vecs-of-Vecs, and clones are for initialization and
configuration. The hot path uses `&[f64]`, `&mut [f64]`, and integer indices.
Parameter lookup is `table[pair_type]`, not `map.get(&(s1.clone(),
s2.clone()))`.

### G6: Fuse computations that share intermediate results

If two force terms both need `dS/dx`, compute `dS/dx` once and use it for
both. If all shell channels share one interpolation, interpolate once and
rotate all channels. If the trace of a product is needed, don't compute the
product.

---

## Reference: the 18 violations

See `doc/prokop/chats/CPU_Optimization.chat.md` for the full review. Summary:

| # | Problem | Rule |
|---|---------|------|
| 0 | Benchmarked in debug mode (-O0) | Rule 6 |
| 1 | Fabricated Jacobi explanation (nalgebra uses Householder+QR) | Rule 5 |
| 2 | Rebuilds entire solver every geometry step | Rule 1 |
| 3 | Gamma recomputed every SCC iteration | Rule 1, 4 |
| 4 | No warm start — q=0 every step | Rule 12 |
| 5 | Only dsyevd replaced, not the whole transform | Rule 5 |
| 6 | Full H_scc transform every SCC iteration (H0' is geometry-invariant) | Rule 7 |
| 7 | Mulliken charges don't exploit Cholesky (SC = LY) | Rule 7 |
| 8 | Forces re-diagonalize the converged Hamiltonian | Rule 4 |
| 9 | H/S derivatives computed twice (non_scc + scc_shift) | Rule 4, G6 |
| 10 | Finite differences instead of analytic sp derivatives | Rule 10 |
| 11 | SK interpolation repeated per shell pair | Rule 9 |
| 12 | Neville interpolation instead of precomputed polynomials | Rule 8 |
| 13 | Force neighbor list unit bug (Å vs Bohr) | Rule 11 |
| 14 | String/HashMap/clone in repulsive force hot path | Rule 3 |
| 15 | Gamma derivative not cached with geometry | Rule 1, 4 |
| 16 | O(N³) matrix product for O(N²) trace | Rule 7 |
| 17 | D/W not built once after convergence | Rule 4, 7 |
| 18 | Vec allocations inside SCC loop despite "zero-alloc" claim | Rule 2 |

---

## Cross-references

- Review chat: `doc/prokop/chats/CPU_Optimization.chat.md`
- Performance audit (partially wrong — see Rule 5): `doc/prokop/topical_audit/eigensolver_performance.md`
- Session report: `doc/prokop/reports/2025-09-05_hbond_optimization_lapack.md`
- AGENTS.md: `/home/prokophapala/git/dftbplus/AGENTS.md` (§Performance)
- Skills: `~/.config/devin/skills/perf/SKILL.md`, `~/.config/devin/skills/debug/SKILL.md`
