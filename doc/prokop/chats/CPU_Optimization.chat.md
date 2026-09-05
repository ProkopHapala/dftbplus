# USER

OK we implemented the things in last commit. But I realize our even CPU implementation is horribly stupid and inefficient and we should correct it. plese analyse it and tell what exacly we should do 

from my conversation with coding LLM


USER

OK, 
1) write report what we were doing and where, how to run it what we impleemnted, what were the chalanges

I'm still disatistief, it still take everything exteremely long, we must have horendously bad bottleneckls , horrendously bad desing of the whole algorithem, write separate docvument about that, we must optimize it 

mention the problem with nalgebra Jacobi. I'm still not really trusting what you say Jacobi should not be so slow, n^2 jacobi rotations should already make diagonal matrix (I think maybe I miss something) I wo not understand why we have this huge number of operaions this is absurd!


then update readme.md files in relevant folders, codemap, topical audit, we should integrate this into the whole do system and crossreference us skill doc-audit for that

I have to go now, we will optimize it when I renturn as the first thing ...


GML 5.2

## On the Jacobi question

You were right to be skeptical. The document explains it honestly: Jacobi **should** take \~3-5ms for N=56 (N² rotations × \~15 sweeps × O(N) per rotation ≈ 3.5M ops). The 20ms is **5× worse than the algorithm predicts** because nalgebra's implementation does heap allocations in the inner loop, doesn't vectorize rotation application, and uses sequential ordering. LAPACK `dsyevd` is faster not because Jacobi is bad, but because it uses a completely different algorithm (Householder + D&C) that does \~50× fewer operations.

When you return, the optimization plan is ready in `eigensolver_performance.md` — SCC warm start is priority 1 (4× speedup, \~8s for 100 steps).


https://github.com/ProkopHapala/dftbplus
also maybe check last commit
https://github.com/ProkopHapala/dftbplus/commit/e79f9932cc9e3838f2fb902c4f8fc0b97a62b91f


---

# Chat GPT 5.6 sol

Yes. After looking at the actual CPU path, I think the problem is substantially worse — but also easier to fix — than the performance document suggests.

The main conclusion is:

> **Do not spend time micro-optimizing nalgebra. The whole CPU execution pipeline is doing large amounts of work that should not exist at all.**

And there is one immediate benchmarking mistake that should be fixed before measuring anything else.

## 0. First: all your reported Rust timings appear to be debug-build timings

The documented command is:

```bash
cargo run --example hbond_ref -- ...
```

not `cargo run --release`. 

Your `[profile.dev]` changes debug information but does **not** set `opt-level`, so Cargo defaults to `opt-level=0`.

This makes the comparison particularly misleading:

```text
Rust/nalgebra/SK/force loops    compiled -O0
LAPACK/OpenBLAS dsyevd         optimized native library
```

So first rerun:

```bash
OPENBLAS_NUM_THREADS=1 cargo run --release --example hbond_ref -- ...
```

`OPENBLAS_NUM_THREADS=1` matters for 56×56 matrices; multithreaded BLAS overhead can be larger than the computation. For CPU multi-system calculations, parallelize **systems externally** and keep BLAS single-threaded.

I would actually make a profiling profile:

```toml
[profile.perf]
inherits = "release"
debug = 1
debug-assertions = true
overflow-checks = true
lto = false
```

and benchmark using `--profile perf`.

Do this **before interpreting any absolute timing numbers**.

---

# 1. The Jacobi analysis in the current document is wrong

This is worth correcting because your intuition was reacting to an incorrect explanation.

`nalgebra::SymmetricEigen` 0.33.2 **does not use Jacobi**.

Its actual source does:

```text
symmetric matrix
    ↓
SymmetricTridiagonal::new()
    ↓
tridiagonal matrix
    ↓
implicit shifted QR
    ↓
Givens rotations + Wilkinson shift
```

The nalgebra 0.33.2 source explicitly constructs `SymmetricTridiagonal`, then iterates Givens rotations with a Wilkinson shift.

So this section of the performance document—

> N² Jacobi rotations per sweep, 10–20 sweeps, inner-loop allocations, sequential Jacobi ordering

—is describing an algorithm that nalgebra isn't actually running. 

Also, for classical Jacobi your original statement

> N² rotations should already make it diagonal

isn't quite true. A Jacobi rotation annihilates \(A_{pq}\), but later rotations involving \(p\) or \(q\) generally recreate nonzero \(A_{pq}\). Therefore one \(N(N-1)/2\)-rotation sweep does not diagonalize a generic matrix exactly; multiple sweeps are required.

But that discussion is irrelevant to this CPU implementation because nalgebra isn't using Jacobi anyway.

The useful empirical observation remains:

$$
20.3\ {\rm ms}\rightarrow0.7\ {\rm ms}
$$

with LAPACK. 

So keep LAPACK and forget about CPU Jacobi.

There is also a repository/report mismatch: commit `e79f993...` still contains `SymmetricEigen` in `fragment.rs`, and its `Cargo.toml` has no LAPACK dependency; the attached report evidently describes newer local changes after that commit.

---

# 2. The biggest architectural problem: every geometry step constructs an entire electronic solver from scratch

`HamiltonianBuilder::build_scc()` currently does approximately:

```text
geometry step
    ↓
FragmentTemplate::new()
    ↓
new HamiltonianBuilder
build H0/S
build SystemContext
extract q0
    ↓
new Fragment
    ↓
new GammaTable
    ↓
new FragmentNeighborList
    ↓
new DIIS mixer
    ↓
new MultiSystemSolver
    ↓
start q from neutral
    ↓
SCC
    ↓
clone matrices into SccResult
```

That's the wrong lifetime model.

Worse, `FragmentTemplate::new()` does:

```rust
let builder = HamiltonianBuilder::new(sk.clone());
```

and `SkData` is a deep `Clone` containing hash maps and entire interpolation tables.

This should disappear completely.

### Replace `build_scc()` with a persistent CPU state

I would make something conceptually this simple:

```rust
struct DftbCpu {
    // topology/static
    ctx: SystemContextOwned,
    gamma_params: GammaTable,
    repulsive: RepulsiveTables,
    q0: Vec<f64>,
    norb: usize,

    // current geometry
    neigh: NeighborList,
    h0: Vec<f64>,
    s: Vec<f64>,
    l: Vec<f64>,          // Cholesky(S)
    gamma: Vec<f64>,      // Nat × Nat, current geometry

    // SCC workspace
    q: Vec<f64>,
    qout: Vec<f64>,
    v: Vec<f64>,
    h: Vec<f64>,
    eig: Vec<f64>,
    c: Vec<f64>,
    sc: Vec<f64>,
    mixer: DiisMixer,

    // LAPACK workspace
    work: Vec<f64>,
    iwork: Vec<i32>,
}
```

Then:

```text
DftbCpu::new(species, SK)       once

for geometry:
    update_geometry(coords)     once
    solve_scc()                 3-5 iterations
    forces()                    once
```

This is the same separation we designed for GPU:

> **geometry-dependent setup outside SCC; charge-dependent work inside SCC.**

---

# 3. Precompute the entire gamma matrix once per geometry

This is an obvious algorithmic waste in the current solver.

Every SCC iteration currently recomputes distances, square roots, and `gamma(r)` evaluations inside `compute_intra_shifts()`.

But geometry does not move during SCC.

Compute once:

$$
G_{AB}=\gamma_{AB}(R_{AB})
$$

including

$$
G_{AA}=U_A.
$$

Then every SCC iteration is simply

$$
\boxed{V=G\,\Delta q}.
$$

For \(N_{\rm atom}=20\), that's a 20×20 dense matrix. It is effectively free.

```text
update_geometry():
    for A<=B:
        G[A,B] = gamma(R_AB)
        G[B,A] = G[A,B]

solve_scc():
    dq = q-q0
    V  = G*dq
```

Do the same for multiple fragments. Don't split `v_intra` and `v_ext` computationally unless it is physically useful elsewhere; mathematically it is one gamma matrix over all participating atoms.

For future giant systems this becomes FMM/mesh, but for the small CPU solver a dense matrix is exactly right.

---

# 4. Warm-start charges; stop destroying them every geometry step

This is probably your largest multiplicative speedup.

The report already observes ~16 SCC iterations from neutral and expects ~3–4 from the previous geometry. 

But don't bolt a `build_scc_warm(...)` function onto the current architecture.

Persistent state solves it naturally:

```text
geometry 0:
    q = q0
    SCC → q*

geometry 1:
    q starts from previous q*
    reset DIIS history
    SCC → new q*

geometry 2:
    same
```

I would initially **keep previous charges but reset DIIS history** after geometry movement. Later test retaining DIIS history or using a predictor:

$$
q^{guess}_{n+1}=2q_n-q_{n-1}.
$$

But previous converged \(q\) alone should already be excellent.

---

# 5. The generalized eigensolver should use LAPACK for the whole transformation, not only `dsyevd`

The current algorithm is mathematically fine:

$$
S=LL^T
$$

$$
H'=L^{-1}HL^{-T}
$$

$$
H'Y=Y\varepsilon
$$

$$
C=L^{-T}Y.
$$

But the current transform/back-transform uses nalgebra triangular solves, which the report measures as 7.6 ms/SCC iteration. 

Don't merely replace them with hand calls to `dtrtrs`.

Use the LAPACK routine designed for this transformation:

```text
update_geometry:
    Swork = S
    dpotrf(Swork)                 // once/geometry

each SCC:
    A = Hscc
    dsygst(itype=1, A, Swork)     // A = L^-1 H L^-T
    dsyevr(...) or dsyevd(...)
    dtrsm(... trans=L^T ...)      // C = L^-T Y
```

`dsygst` knows that the matrix is symmetric and performs the generalized→standard transform efficiently with BLAS-3 operations.

### Compute only occupied eigenvectors if useful

You have \(N=56\), but only approximately \(N_{\rm occ}\) eigenvectors contribute.

Benchmark:

```text
dsyevd   // all eigenpairs
vs
dsyevr(range=I, il=1, iu=Nocc+1)
```

The extra `+1` gives the LUMO for HOMO/LUMO/fermi information.

For N=56, `dsyevd` may still win because it is simple and highly optimized. Do not assume partial solve is faster; benchmark.

### LAPACK already sorts the eigenvalues

If your local LAPACK conversion retained this code:

```rust
idx.sort_by(...)
select_columns(&idx)
```

delete it.

`DSYEVD/DSYEVR` return eigenvalues in ascending order.

---

# 6. We can do better than transforming the whole SCC Hamiltonian every iteration

This is the first optimization I'd do **after** the clean LAPACK implementation works.

You have

$$
H=H^0+\frac12(SV+VS),
$$

where \(V\) is diagonal in AO space:

$$
V_{\mu\mu}=V_{A(\mu)}.
$$

With

$$
S=LL^T,
$$

precompute once per geometry

$$
H_0'=L^{-1}H^0L^{-T}.
$$

Then:

$$
\begin{aligned}
L^{-1}HL^{-T}
&=H_0'
+\frac12\left(
L^{-1}SVL^{-T}
+L^{-1}VSL^{-T}
\right)\\
&=H_0'
+\frac12\left(
L^TVL^{-T}
+L^{-1}VL
\right).
\end{aligned}
$$

Define

$$
X=L^{-1}VL.
$$

Because \(V=V^T\),

$$
L^TVL^{-T}=X^T.
$$

Therefore:

$$
\boxed{H'=H_0'+\frac12(X+X^T)}
$$

with only

```text
B = V*L              // row scaling, O(N²)
X = solve(L,B)       // one triangular solve
H' = H0' + sym(X)
```

instead of building `Hscc` and doing the full generalized transform every SCC iteration.

That should roughly halve the transformation work.

It also means the SCC loop need not build physical-basis `H_scc` at all unless somebody explicitly requests it.

---

# 7. Mulliken populations should exploit the Cholesky transformation

Current `compute_charges()` has the triple loop:

$$
(SC)_{\mu k}=\sum_\nu S_{\mu\nu}C_{\nu k}
$$

inside the orbital/occupied-state loops.

But after diagonalization we already have

$$
C=L^{-T}Y.
$$

Therefore

$$
SC=LL^TL^{-T}Y=LY.
$$

So calculate:

```text
C  = L^-T Y     // dtrsm
SC = L Y        // dtrmm
```

and Mulliken population is just

$$
p_\mu=2\sum_{k\in occ}C_{\mu k}(SC)_{\mu k}.
$$

Then:

$$
q_A=\sum_{\mu\in A}p_\mu.
$$

That's extremely cheap and cache-friendly.

No full density matrix is needed during SCC.

---

# 8. The force implementation currently contains several enormous wastes

This is probably the worst code path in the whole current CPU implementation.

## 8a. It diagonalizes the converged Hamiltonian AGAIN

`compute_scc_forces()` says explicitly:

> Re-diagonalize H_scc to get consistent eigenvectors for EDM.

And it calls its own `diagonalize_h_scc()`, which in commit `e79f...` even uses the slow nalgebra `SymmetricEigen` again.

This is completely unnecessary.

The final SCC iteration already has:

```text
eigenvalues
eigenvectors
```

Keep them.

After SCC convergence, construct once:

$$
D=2C_{\rm occ}C_{\rm occ}^T
$$

and energy-weighted density

$$
W=2C_{\rm occ}\varepsilon_{\rm occ}C_{\rm occ}^T.
$$

Pass `D,W` directly to forces.

Never diagonalize inside `forces()`.

---

# 9. You calculate the same H/S finite-difference derivatives twice

`non_scc_electronic_force()` calculates `pair_block_derivative()` for every pair and every Cartesian direction.

Then `scc_shift_force()` does exactly another `pair_block_derivative()` for the same pair/direction just to get \(dS\).

Fuse them.

The two current contributions are:

$$
F^{0}_{ij,\alpha}
=
2\sum_{\mu\nu}
\left[
D_{\mu\nu}
\frac{\partial H^0_{\mu\nu}}{\partial R_\alpha}
-
W_{\mu\nu}
\frac{\partial S_{\mu\nu}}{\partial R_\alpha}
\right]
$$

and

$$
F^{SCCshift}_{ij,\alpha}
=
2\sum_{\mu\nu}
D_{\mu\nu}
\frac{V_i+V_j}{2}
\frac{\partial S_{\mu\nu}}{\partial R_\alpha}.
$$

So simply compute:

$$
\boxed{
F^{el}_{ij,\alpha}
=
2\sum_{\mu\nu}
\left[
D_{\mu\nu}\frac{\partial H^0_{\mu\nu}}{\partial R_\alpha}
+
\left(\frac{V_i+V_j}{2}D_{\mu\nu}-W_{\mu\nu}\right)
\frac{\partial S_{\mu\nu}}{\partial R_\alpha}
\right].
}
$$

One derivative evaluation supplies both terms.

Immediately ~2× less derivative work.

---

# 10. But finite-differencing the SK blocks is itself unnecessary for your sp basis

Currently for each pair:

```text
x+: rebuild H/S block
x-: rebuild H/S block
y+: rebuild H/S block
y-: rebuild H/S block
z+: rebuild H/S block
z-: rebuild H/S block
```

= **6 complete SK block evaluations**.

And until the previous fusion, you were doing that twice → 12.

For an \(s,p\) basis, analytic derivatives are extremely easy.

Let

$$
\mathbf u=\mathbf R/R,\qquad u_i=R_i/R.
$$

Then

$$
\frac{\partial R}{\partial R_a}=u_a,
$$

and

$$
\boxed{
\frac{\partial u_i}{\partial R_a}
=
\frac{\delta_{ia}-u_i u_a}{R}.
}
$$

For `ss`:

$$
H_{ss}=V_{ss\sigma}(R)
$$

so

$$
\partial_a H_{ss}=V'_{ss\sigma}u_a.
$$

For `sp`:

$$
H_{sp_i}=u_iV_{sp\sigma},
$$

hence

$$
\partial_a H_{sp_i}
=
\frac{\delta_{ia}-u_i u_a}{R}V_{sp\sigma}
+
u_iV'_{sp\sigma}u_a.
$$

For `pp`:

$$
H_{p_ip_j}=V_\pi\delta_{ij}+(V_\sigma-V_\pi)u_i u_j.
$$

Let

$$
\Delta V=V_\sigma-V_\pi.
$$

Then

$$
\partial_a H_{ij}
=
V_\pi'u_a\delta_{ij}
+\Delta V'u_a u_i u_j
+\frac{\Delta V}{R}
[(\delta_{ia}-u_i u_a)u_j+u_i(\delta_{ja}-u_j u_a)].
$$

Same equations for \(S\).

So one pair evaluation should return:

```text
H[16], S[16]
dHdx[16], dHdy[16], dHdz[16]
dSdx[16], dSdy[16], dSdz[16]
```

with **one radial interpolation**.

That should demolish the current force cost.

Keep finite difference only as a parity/debug test for the analytic derivative.

---

# 11. The SK interpolation is also doing the same expensive interpolation repeatedly

This is another big one.

`SkTableSp::eval_shell_integrals_into()` calls:

```rust
self.h.eval_into(...)
self.s.eval_into(...)
```

and evaluates all available channels, then extracts only the requested shell channel.

`rotate_diatomic_block_into()` then calls this separately for every shell pair.

For `sp` atoms:

```text
ss
sp
ps
pp
```

so essentially the same interpolation is repeated ~4 times/pair.

And the interpolation itself is an 8-point Neville algorithm with rather large stack matrices and nested loops.

### Fix API first

Instead of:

```text
eval_shell(ss)
eval_shell(sp)
eval_shell(ps)
eval_shell(pp)
```

do:

```text
eval_all(r, Hsk[], Ssk[])
rotate_complete_block(Hsk,Ssk,u)
```

One H interpolation + one S interpolation per species table/pair.

For heteronuclear pairs where forward/reverse tables differ, evaluate each relevant table once.

---

# 12. Then replace Neville interpolation with precomputed interval polynomials

This can preserve the exact current interpolation mathematically.

For an 8-point uniform interpolation stencil, each channel is simply a degree-7 polynomial in

$$
t=(R-R_0)/\Delta R.
$$

At SK-loading time, transform the 8 grid values into polynomial coefficients:

$$
V(t)=a_0+a_1t+\cdots+a_7t^7.
$$

Store the coefficients per interval/channel.

Runtime becomes:

```text
interval = floor(r/dr)
V  = horner(coeff, t)
V' = horner_derivative(coeff, t)/dr
```

~7 FMAs for the value and derivative.

No Neville recursion. No 20×8 `cc/dd` scratch arrays. No repeated stencil copying.

And the derivative needed by forces comes essentially for free.

This is probably a very substantial CPU improvement and maps beautifully to GPU later.

---

# 13. There is a unit bug in the force neighbor list

Hamiltonian construction does:

```text
coords Å → coords_bohr
NeighborBuilder(cutoff_from_SK)
```

which is consistent because the SK cutoff is in Bohr.

But the force code does:

```rust
NeighborBuilder { cutoff }.build(coords)
```

where `coords` are still in Å.

`NeighborBuilder` simply compares coordinate distance to `cutoff` without units/conversion.

Therefore a ~10 Bohr SK cutoff is being interpreted as ~10 Å rather than ~5.29 Å.

That can include vastly too many pair blocks in the expensive finite-difference force calculation.

Don't “fix the conversion” locally.

Better:

> build one geometry neighbor/pair list and share it between H/S construction and force calculation.

Each pair should cache:

```text
i,j
R
u = Rvec/R
species-pair table index
```

No independent neighbor lists.

---

# 14. Repulsive force is doing string/hash/file work in the hot path

Current `repulsive_force()` creates a fresh `HashMap<(String,String),...>` every force call.

Inside the atom-pair loop it:

```rust
let key = (species_names[i].clone(), species_names[j].clone());
```

and even when the spline is cached:

```rust
s.clone()
```

which clones `Option<RepulsiveSpline>`; `RepulsiveSpline` contains `Vec`s.

This is awful for a force kernel.

Repulsive splines are **parameter data**, not geometry data.

Parse them once when loading `SkData`.

Then:

```text
pair_type = species_i*n_species + species_j
repulsive[pair_type] -> spline reference/index
```

At runtime:

```text
(r, pair_type) -> Erep(r), dErep/dr
```

No strings. No HashMap. No files. No cloning.

And if you implement precomputed polynomial interpolation, repulsive evaluation can use the same general mechanism.

---

# 15. Gamma derivative should also be cached with geometry

While building

$$
G_{AB}=\gamma(R_{AB}),
$$

also calculate

$$
g'_{AB}=\frac{d\gamma}{dR}.
$$

Store perhaps:

```text
gamma[Nat*Nat]
gamma_force_pair[p] = gamma'(R)/R
```

Then final SCC double-counting force:

$$
\mathbf F_A^{\gamma}
=
-\sum_{B>A}
\Delta q_A\Delta q_B
\gamma'(R_{AB})\hat{\mathbf R}_{AB}.
$$

No exponentials or gamma derivatives during force evaluation.

---

# 16. Do not compute a full matrix product just to get `Tr(D H0)`

At the end of `build_scc()`:

```rust
let e_h0 = (&density * &frag.template.h0).trace();
```

builds an entire \(N\times N\) matrix product solely to extract its trace.

Use the Frobenius contraction:

$$
\boxed{
E_{H0}=\operatorname{Tr}(DH^0)
=\sum_{\mu\nu}D_{\mu\nu}H^0_{\mu\nu}
}
$$

for symmetric real matrices.

That's \(O(N^2)\), not \(O(N^3)\).

Even better, once validated, use the occupied eigenvalue relation and avoid constructing \(D\) until forces require it.

---

# 17. Build D and W once, after SCC convergence

During SCC:

```text
only need:
eigenvalues
Cocc
SCocc
Mulliken q
```

After SCC converges:

$$
D=2C_{\rm occ}C_{\rm occ}^T
$$

$$
W=2C_{\rm occ}\varepsilon C_{\rm occ}^T.
$$

Use BLAS.

For \(D\):

```text
DSYRK
```

For \(W\):

```text
CE[:,k] = 2*eps[k]*C[:,k]
W = CE*C^T
```

via `DGEMM`.

Then force contraction reads `D` and `W`.

This also fixes the current bizarre situation where `SccResult` stores `D` but **not the eigenvectors/EDM**, causing force code to diagonalize again.

I'd simply store `W` in the final result/workspace.

---

# 18. `MultiSystemSolver` is not allocation-free despite claiming to be

For example:

```rust
let offsets: Vec<usize> = ...
```

is constructed inside `build_all_h_scc()` every SCC iteration.

And `DiisMixer::mix()` does:

```rust
let q_in_copy = q_inout.to_vec();
let res_copy  = residual.to_vec();
```

each DIIS iteration.

For your 20-atom example these aren't large runtime costs, but they indicate the wrong architecture.

Precompute fragment offsets once.

For DIIS use fixed ring buffers:

```text
q_hist[max_hist * natom]
r_hist[max_hist * natom]
head
count
```

No `VecDeque<Vec<_>>`.

The DIIS Gram matrix can also be updated incrementally instead of recomputing all \(m^2\) dot products, although at `m<=10` this is very low priority.

More importantly, the current Gaussian elimination handles a nearly singular pivot by `continue`, which does not really constitute a valid failure/fallback strategy.  That may contribute to the observed DIIS bouncing/floor. I would return failure on a singular system and fall back to simple mixing, or use tiny LAPACK `dgesv` with regularization.

---

# The CPU pipeline I would actually implement

This is the architecture I would give the coding agent:

```text
INITIALIZATION — once
────────────────────────────────
load SK
build species codes / orbital offsets
build pair-table LUT
parse repulsive tables
q0, Ne
allocate all matrices/workspaces
allocate LAPACK workspaces


EACH GEOMETRY — once
────────────────────────────────
build/update atom pair list
for each SK pair:
    interpolate H,S ONCE
    build H0,S
    ideally also analytic dH/dx,dS/dx

build gamma_AB and gamma'_AB
evaluate/cache repulsive force coefficients

Cholesky S:
    L = chol(S)

precompute:
    H0' = L^-1 H0 L^-T


SCC — typically ~3–5 iterations
────────────────────────────────
dq = q-q0
V  = Gamma*dq

Vorb[μ] = V[atom(μ)]

B  = Vorb * L                // row-scale L
X  = L^-1 B
H' = H0' + 0.5*(X+X^T)

diagonalize H':
    Y, eps

C  = L^-T Yocc
SC = L Yocc

qout_A = 2 Σ_{μ∈A,k} C[μ,k]*SC[μ,k]

residual
mix q

repeat


AFTER CONVERGENCE — once
────────────────────────────────
D = 2 Cocc Cocc^T
W = 2 Cocc eps Cocc^T

energy

for each pair ONCE:
    use cached/analytic dH,dS
    electronic force
    SCC-shift force simultaneously
    gamma force
    repulsive force
```

That is the correct CPU reference architecture, in my view.

---

## What I would ask the coding agent to do first

| priority | change                                                | reason                                              |
| -------- | ----------------------------------------------------- | --------------------------------------------------- |
| **0**    | rerun all timing in `--release`, OpenBLAS 1 thread    | current measurements are contaminated by `-O0`      |
| **1**    | persistent `DftbCpu` state + warm-start `q`           | removes rebuilding + ~4× fewer SCC iterations       |
| **2**    | stop re-diagonalizing in `compute_scc_forces`         | pure duplicated electronic solve                    |
| **3**    | precompute `Gamma_AB` once/geometry                   | removes sqrt/exp from SCC                           |
| **4**    | replace nalgebra transforms by `dpotrf/dsygst/dtrsm`  | remove remaining 7.6 ms reported transform overhead |
| **5**    | fuse force H/S derivatives; evaluate once             | ~2× force derivative reduction immediately          |
| **6**    | fix force neighbor-list units                         | correctness + potentially many fewer pairs          |
| **7**    | evaluate all SK channels once/pair                    | removes repeated Neville interpolation              |
| **8**    | analytic sp derivatives + polynomial SK interpolation | should collapse the force/H-build cost              |
| **9**    | cache repulsive tables, no Strings/HashMaps/clones    | remove absurd hot-path parameter handling           |
| **10**   | then optimize DIIS/workspace allocations              | cleanup, probably minor runtime                     |

The single thing I would **not** do now is spend time designing another CPU eigensolver. LAPACK solved that problem. The architecture around the eigensolver is currently wasting much more work.

And I would definitely rewrite the `eigensolver_performance.md` explanation: its measured timings are useful, but the explanation of nalgebra as a bad Jacobi implementation is factually wrong, and the much bigger discovery is that those measurements appear to have been made with Rust at `opt-level=0`. 
