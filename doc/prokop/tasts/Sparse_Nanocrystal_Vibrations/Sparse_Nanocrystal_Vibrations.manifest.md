# Task 2: Sparse GPU DFTB Forces and Vibrations of Si/Diamond Nanocrystals

**Created:** 2026-09-07  
**Revision:** v4 — absorbs second GPT-5.6 review (commit `e965ae0`, §14)  
**Last wrap (2026-09-11):** read **§0** first, then **§14** — second review:
three remaining blockers, the N4 contract bug, revised order §14.7. Floor vs bug map:
`doc/prokop/topical_audit/f32_floor_sparse.md`. Interpolator:
`doc/prokop/topical_audit/sk_interpolation.md`. Dense H-bond is a **separate**
agent — do not edit `qmqm/gpu_forces.cl`.
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

## 0. State for the next LLM (2026-09-10) — read this, then the rest

Job of the current sparse-physics agent: **map the algorithm**, separate
**bugs / physical misconceptions / method stopgaps / arithmetic floor**, keep
tests **truthful** (not fake-green, not demanding f32 miracles), and **document
the floor** so a later dedicated pass can do Kahan / hybrid f64 islands /
error-compensated SpGEMM **without guessing**. That compensation pass is not
one-shot and must be benchmarked; it needs this map first.

NVIDIA RTX 3090, `RUST_DFTB_SK_DIR=.../matsci-0-3`. Nothing below is “done”
until USER confirms.

### 0.1 What was resolved (do not re-open as if unknown)

| Item | Outcome |
|------|---------|
| Energy was `Tr(K H0)` without E_rep / without SCC | **Fixed in physics gates.** Canonical `E = 2 Tr(K H0) + ½ Δq·V + E_rep`. Gate F old path collapsed Si–H 1.60→**0.93 Å**; new FIRE → **1.477 Å**. |
| SK file `q0` used as SiH4 valence | **Misconception.** matsci Si=0, H≈0.49 (sum ≈ 2 e⁻). Physical `[4,1,1,1,1]`. First G3.2 \|dE\|=3.71 Ha was this, not TC2. |
| Sparse K from dense `build_scc` Hamiltonian | **Fixed in G3.2/F/G.** `run_sparse_scc` is a real mix loop (host NS + TC2). `SparseSystemWorkspace::run_scc` is still purify-once — do not confuse them. |
| Analytic F | **G3.3.** `D=2K`, `W=2KHK`, CPU `compute_forces_from_dw`. vs dense max\|dF\|=3.7e-6. `qmqm/gpu_forces.cl` **read-only**. |
| Gate G Hessian | **Unsymmetrized FD of F**, h=0.01 Å, same Gate F coords. η_asym from H_raw (not Gate E triangle-copy). vs dense \|\|ΔH\|\|_F/\|\|H\|\|_F=**1.10e-3**, ordinary \|Δν\|≤4 cm⁻¹. |
| C² B-spline **evaluation** in the force path | **Fixed.** `eval_with_deriv_into` is B-spline. GPT-5.6 “Hermite still production” is stale. Remaining interpolator work is the **fitter** (§0.3), not switching evaluators again. |
| Neville `poly5_to_zero` tail −0.4 Ha | **Fixed** (shared with dense). Do not restore for Fortran-tail parity. |
| Host NS identity residual cancellation | Host path uses `\|I−T\|_F` directly. Device NS N4 is **explained** — contract bug, next row. |
| N4 "f32 NS failure" mystery | **Contract bug, not numerics.** `identity_residual_scalar_dev` returns ‖I−T‖² (no `sqrt`, gpu_sparse.rs:2020); NS divides only by √N (:2322). Reported 1.9e-5 was a *squared* norm → real ‖I−T‖ ~1e-2, consistent with max\|Z−S⁻¹\|=2.18e-3. Fix = §14 R5. Also `build_identity_dev`+`scale_dev` leave stale off-diagonal Z on geometry ≥2 (R6). |

G3.1 \|dE_h0\|=1.92e-7; G3.2 \|dE_el\|=1.13e-7, max\|dq\|=2.07e-5. Gate F
E_tot=−2.826057, \|F\|=4.18e-4. Tests: `gate_g3_energy.rs`, `gate_f_geom_opt.rs`,
`gate_g_hessian.rs`.

### 0.2 Open problems (four different kinds — do not mix them)

**A. Bugs** — fix the code; compensation will not help.

| ID | What | Action |
|----|------|--------|
| N4 / B1 | `newton_schulz_inverse_dev`: R_Z≈1.9e-5 but max\|Z−S⁻¹\|≈2.18e-3 | **Cause identified (§14.0):** missing `sqrt` in `identity_residual_scalar_dev` + stale off-diagonal Z after `build_identity_dev`. Keep `test_newton_schulz_inverse_dev` **red** until fixed *and* cross-checked vs host f64 residual of identical T. Then delete the per-iteration T download in `compute_z`. Leftover `run_sparse_purify` has `_dev` **commented** — do not restore. |
| D1 | `SparseDftb` holds dense `h0_phys`/`s_phys` f64[N²] + 4 padded f32[(4N)²]; `set_coords` densifies then re-sparsifies | **BLOCKER** (~512 MB host at N=1000). Direct BSR pair assembly, §14 R1. |
| D2 | `forces()` → `dw_from_k_padded`: two dense f64 triple-loop matmuls O((4N)³) | **BLOCKER**. Route through `SparseDWWorkspace` plans → W on M_HS, §14 R2. |
| D3 | `finalize_scc` stores `q_fin` with `V`,`H_scc` of `q_new` — not one stationary state | **BLOCKER** for forces/Hessian at finite tol, §14 R3. |
| D4 | One structural mask for H/S, K, Z; skin defeated by rebuild-and-demand-equality in `set_coords` | Independent R_HS/R_K/R_Z + Verlet skin, §14 R4/R15. |
| B2 | `SparseSystemWorkspace::run_scc` is purify-once | Real SCC is `SparseDftb::scc`. Rename or leave as one-shot purify. Do not call `run_scc` from jobs. |
| B4 | Gates C and E still in tree as false positives | Do not copy. Do not treat green-there as locality/Hessian proof. |

**B. Interpolator extra-control fitter** — method, not f32. Plain language §0.3.
Spec: `sk_interpolation.md`.

**C. f32 / mixer / FD floor** — mapped in `f32_floor_sparse.md`. Do not chase:

- G3.4 rel \< 1e-3 at h=1e-3 Å (energy FD is noisier than the force; G3.3 is 3.7e-6);
- Gate G rigid modes to 0 cm⁻¹ (`n_unstable≤6` was a cheat);
- SiH4 charge rms \< 1e-8.

Measured floor worth compensating later: Gate G Hessian \|\|ΔH\|\|_F/\|\|H\|\|_F
≈ **0.11%** vs dense f64; η_asym ≈ **3.9e-4** on **both** sparse and dense
(so it is the 3-point F stencil + SCC residual, not a unique sparse bug).
Kahan on host `Tr(K H0)` will not move that Hessian. Device SpGEMM / NS
reductions might — **after** N4 is a real inverse.

**D. Production sparse run loop is coded, not a nanocrystal yet.** `SparseDftb`
+ `dftb_engine` `sparse_*` + `scripts/test_sparse_dftb_sih4.rhai`. Lab numbers
§0.7. Remaining — now classified **blockers** by the second review (§14):
dense host H0/S + padded arrays *inside* `SparseDftb`, dense O(N³) force,
non-stationary SCC finalization, single mask, γ+Hscc rebuild+upload per iter,
K download+densify per iter. Do not publish timings from `cargo test`.
Geometric mask (\(n>64\)) not exercised.

### 0.3 What “interpolator fitter” means

Slater–Koster files are **numbers on a 1D grid** (typically `dr=0.02` Bohr).
We interpolate with a **cubic B-spline**. A cubic B-spline needs extra
**control points** off each end of that table. They are not extra SK
measurements; they are the degrees of freedom that implement boundary
conditions (how the curve starts; how it dies at cutoff).

- **Left (already the right kind of thing):** phantom `c_{-1}=2c_0−c_1` at
  evaluation. Small-r SK values are large — never zero-pad the left.
- **Right (current stopgap, blunt):** glue on four **function values
  hardcoded to 0**, then refit (`fit_bspline_controls_zero_end`). Stopped the
  −0.4 Ha explosion. Those zeros **leak** into the last real samples; cutoff
  is only `~0.08` Bohr past the last grid.
- **GPU dummy 0 at r=0:** index padding, not a fitted left control.

**The fitter** is the missing linear solve: extra controls are **unknowns**.
Solve so (1) the interpolant on the **tabulated** region still matches the SK
file, (2) `V` and `V'` go smoothly to 0 at a chosen cutoff. Do **not**
implement this as “pad with more zeros.” Do not restore Neville.

Independent of the sparse f32 Hessian floor. Interior H/S already ~1e-7.
**Shared with the dense H-bond task** — one fitter, both consumers.

### 0.4 Production pipeline — we do **not** have it, and we **must**

A normal DFT/DFTB code does:

```
INIT (once)          load SK, compile kernels, preallocate all buffers
PER GEOMETRY         update coords → neighbors → H0, S, γ
SCC (warm-started)   iterate until Δq plateaus
FORCES               D, W already available → F
MD / FIRE / Hessian  move atoms → go to PER GEOMETRY
```

**CPU already follows this:** `rust_dftb/src/methods/dftb/dftb_cpu.rs` (`DftbCpu`)
+ FIRE in `examples/hbond_ref.rs`.

**Sparse GPU now has the owner.** `SparseDftb` (`sparse_dftb.rs`) is the GPU
analogue of `DftbCpu`. User path is **one binary** `dftb_engine` + a `.rhai`
script (`sparse_new` / `sparse_scc` / `sparse_eval` / `sparse_fire_step` /
`sparse_md_step` / `sparse_relax`). Docs: `doc/prokop/userguide/sparse_dftb.md`.
Do **not** add `src/bin` / `examples/` / `tests/*.rs` per molecule.

| Piece | Role now |
|-------|----------|
| `SparseDftb` | Production solver. INIT once; `set_coords` / `scc` / `forces` / FIRE / MD. |
| `dftb_engine` `sparse_*` | Script API onto that object. |
| `run_sparse_scc` / `eval_sparse_energy_forces` | Leftover allocating path. Do not grow. Gates should call `SparseDftb`. |
| `run_sparse_purify` | Leftover: GPU purify of a **dense CPU** SCC. Not `SparseDftb`. |
| `cargo test --test sparse_dftb` | SiH4 smoke. Same physics as `test_sparse_dftb_sih4.rhai`. **Not a benchmark.** |

```
SparseDftb::new(sk, sk_dir, species, coords)  // INIT
  .set_coords(R)                               // values only; frozen mask
  .scc()                                       // workspace NS + device TC2
  .forces() / .fire_step() / .md_step() / .relax()
```

Host `‖I−T‖` of downloaded T is the production Z check until B1 is a real
inverse. No `Kernel::builder` / `Program::build` / `Buffer::builder` in `scc` /
`fire_step`. Remaining host work: CPU H0/S, CPU γ + Hscc upload per mix iter,
CPU `compute_forces_from_dw`. Benchmark in `--release` on NVIDIA, long-lived
process, warm-up discarded. Never quote `cargo test` wall time as GPU DFTB.

### 0.5 Honest tests vs impossible tests

Truthful: print the numbers; assert physical invariants (Si–H window, Newton
ΣF, η_asym from H_raw, valence q0).

Impossible / wrong: G3.4 rel 1e-3 at h=1e-3; η_asym identically 0; device NS
green while max\|Z−S⁻¹\| is 2e-3; SK q0 as SiH4; timings of test setup.

Keep N4 **red**. Keep G3.3 tight. Gate G 0.11% Hessian vs dense is the
compensation target, not a reason to loosen G3.3.

### 0.6 Pointers

| Doc | Role |
|-----|------|
| `doc/prokop/topical_audit/f32_floor_sparse.md` | Algorithm map, ID table B/P/M/F, pipeline, handoff |
| `doc/prokop/topical_audit/sk_interpolation.md` | Extra-control fitter spec |
| `doc/prokop/topical_audit/f32_floor_dense_hbond.md` | Dense cousin (do not mix solvers) |
| `OVERVIEW_Roadmap.md` §7.6 | Status checkboxes (investigating until USER confirms) |
| `doc/prokop/userguide/sparse_dftb.md` | User CLI (same binary as dense `gpu_*`) |
| This file §0.7 | Lab notebook: measured CLI / SparseDftb numbers |

### 0.7 Lab notebook — 2026-09-10 (SparseDftb CLI)

Recorded so we can return. **Not** “USER confirmed done.” Distorted SiH₄
(1.48 Å construction, not tetrahedral). SK: matsci-0-3. Device: **NVIDIA
GeForce RTX 3090**. Command (from `rust_dftb/`):

```
RUST_DFTB_SPARSE_ALGEBRA_VERBOSE=0
cargo run --release --bin dftb_engine -- \
  --script scripts/test_sparse_dftb_sih4.rhai \
  --sk-dir $RUST_DFTB_SK_DIR
```

| Quantity | Value |
|----------|-------|
| Device | NVIDIA GeForce RTX 3090 (local_mem=49152, wg=1024, CUs=82) |
| n_atom / n_orbs / nocc / mask | 5 / 8 / 4 / full (25 BSR blocks) |
| valence q0 | `[4, 1, 1, 1, 1]` (not SK parser) |
| NS (once / geom) | 7 iters, **R_Z = 6.265×10⁻⁸** (host ‖I−T‖ of downloaded T) |
| SCC 1 | **E = −2.76420524 Ha**, rms=9.23×10⁻⁶, **17** mix iters, Tr(KS)=4.000002, R_I=3.88×10⁻⁶, r_scc=1.97×10⁻⁵, max\|F\|=8.83×10⁻² Ha/Å |
| Mulliken q | 3.947, 1.025, 1.074, 0.981, 0.973 |
| SCC 2 (reuse, same R) | E = −2.76420569 Ha, **4** mix iters, \|ΔE\|≈4.5×10⁻⁷ Ha |
| 1 FIRE + SCC | E = −2.76442147 Ha, 8 mix iters (NS again, R_Z=5.2×10⁻⁸) |
| 1 MD (dt=0.05) + SCC | E = −2.76464354 Ha, 8 mix iters |

Earlier **allocating** gates (`scc.rs` / `eval_sparse_energy_forces`), same
device/SK, different start (Gate F from 1.60 Å) and different FIRE:

| Gate | Recorded (2026-09-10, still on `scc.rs` until switched to `SparseDftb`) |
|------|---------|
| G3.1 | \|dE_h0\|=1.92×10⁻⁷ vs dense non-SCC |
| G3.2 | \|dE_el\|=1.13×10⁻⁷, max\|dq\|=2.07×10⁻⁵, r_scc≈2.0×10⁻⁵ |
| G3.3 | max\|dF\| vs dense 3.7×10⁻⁶ |
| G3.4 | \|F_ana−F_fd\| 1.8×10⁻⁵, rel 1.3×10⁻³ at h=1e-3 Å (not an f32 law) |
| Gate F | FIRE Si–H mean **1.477 Å**, E_tot=−2.826057, \|F\|=4.18×10⁻⁴ |
| Gate G | same frozen coords; ‖ΔH‖_F/‖H‖_F=**1.10×10⁻³**, η_asym≈3.9×10⁻⁴ (H_raw); ordinary \|Δν\|≤4 cm⁻¹ |

**CLI / engine work this session (do not re-invent):**

- Same binary as dense: `dftb_engine` gained `sparse_*` (no second `src/bin`).
- Script: `rust_dftb/scripts/test_sparse_dftb_sih4.rhai`.
- User guide: `doc/prokop/userguide/sparse_dftb.md` (sibling of `dftb_engine.md`).
- Geometry `Element` table: F, Si, P, S, Cl + `valence_electrons()` (Si=4, H=1).
- `SparseDftb::energy()` fails if `n_scc==0` (was returning 0.0).
- Leftover `run_sparse_purify` is **not** this solver.

Next: nanocrystal `.rhai` with geometric mask (\(n>64\)). Device NS `_dev` stays red.
G3.1 still uses `energy_non_scc` (non-SCC diagnostic). `run_sparse_purify` leftover.

**Same day, after unifying G3.2–G / F onto `SparseDftb` (NVIDIA 3090):**

| Gate | SparseDftb result |
|------|-------------------|
| G3.2 | \|dE_el\|=3.47×10⁻⁷, max\|dq\|=1.87×10⁻⁵, r_scc=1.97×10⁻⁵, E=−2.76420524 (matches CLI) |
| G3.3 | max\|dF\|=5.30×10⁻⁶ vs dense, rel=6.0×10⁻⁵ |
| G3.4 | \|F_ana−F_fd\|=6.0×10⁻⁵, **rel=4.4×10⁻³** at h=1e-3 Å (passes on abs\<1e-4; rel is mixer/FD, not an f32 law) |
| Gate F | 72 FIRE steps, E=**−2.826054**, \|F\|=9.32×10⁻⁴, Si–H 1.477 Å. Coords in `gate_g_hessian.rs::gate_f_sih4_coords`. |
| Gate G | η_asym sp=3.25×10⁻⁴ / dn=3.88×10⁻⁴; ‖ΔH‖_F/‖H‖_F=**1.16×10⁻³**; ordinary stretch Δν ≈ −5 cm⁻¹ (2314 vs 2319) |

CLI smoke (1.48 Å distorted, not Gate F min) and Gate F (relaxed) are **different geometries**. Do not mix the two energies (−2.764 vs −2.826).

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

### 1.4 Performance mandate — fastest GPU DFTB in the world

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
   CPU sort of structural indices that one WG can do in local memory.

**KISS, but not at the cost of speed.** Simple code is preferred when it is
*also* fast. When "simple" means "slow" — e.g. a CPU bridge through an
allocation-heavy dense force function, a serial spectral-bound computation
where a GPU row-sum reduction exists, a per-iteration `finish()` for a 4-byte
trace read, a CAS-loop atomic where a one-WG local reduction is deterministic
and cheaper — the answer is **no**. We write the fast version. The code may be
longer; that is acceptable.

**The recurring failure mode this section exists to prevent:** a coding agent
takes a shortcut that is locally simpler, the test passes because the
shortcut is correct in isolation, and the result is a 2–10× performance
regression or a hidden correctness bug that only surfaces under real
workloads (N=300–1000, 6N Hessian displacements, 200 FIRE steps). The GPT-5.6
review (section 13) catalogs exactly this pattern. Every item there is a case
where "simple and easy" was chosen over "fast and correct", and the cost was
paid in either physics or throughput.

**Concrete implications for the sparse solver:**

- One persistent `SparseSystemWorkspace` owns positions → H/S → SCC → D/W →
  forces. No second context, no per-displacement uploads, no force driver that
  allocates and downloads. All frozen patterns, GPU structures, symbolic plans,
  and scratch matrices live in the workspace for the lifetime of the topology.
- D and W are GPU buffers after SCC/purification. Forces stay on GPU for FIRE.
  No CPU copy between SCC and force evaluation. Do not materialize D=2K as a
  separate matrix — pass K + spin_factor=2 to the force contraction.
- TC2 convergence, Mulliken charge extraction, and spectral-bound reductions
  happen on device. A 4-byte scalar read per iteration is a synchronization,
  and synchronization dominates when 6N Hessian displacements each run dozens
  of SCC/TC2 iterations.
- The SpGEMM, purification, Newton-Schulz, and force kernels are built once and
  reused. Buffer capacity is preallocated; changing arguments is `set_arg`,
  not `Buffer::builder()`. Symbolic plans are built once at frozen topology
  and reused across all SCC iterations, force calls, and Hessian displacements.
- Analytic derivatives are the production route. Finite differences of H/S are
  test/reference only. The only production finite difference is force → Hessian.
- The sparse SCC loop is self-consistent: K from sparse purification of a
  sparse Hscc built from sparse Mulliken charges — no dense `SccResult`
  anywhere. The force must be the gradient of the same sparse energy whose
  Hessian we diagonalize.
- Tests prove what they claim. A test that says "residual < 1e-5" must assert
  `< 1e-5`, not `< 1e-3` because the solver misses. Fix the solver. A test that
  claims "locality plateau" must test a real insulating system at growing N,
  not a 5-atom toy where R=7Å covers everything.

This mandate is the lens through which the GPT-5.6 review (section 13) and all
subsequent implementation should be read. See `Sparse_Nanocrystal_Vibrations.tasks.md`
for the phased breakdown that enforces it.

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

**2026-09-11 update:** the second GPT-5.6 review (§14) inserts a corrective
pass **before** P4/E-series kernel tuning: NS contract bugs → dense-storage
removal + independent masks/skin → sparse energy (no per-iteration K
densification) → stationary SCC finalization + R_H → sparse W/force →
per-geometry precomputation → normalized tolerances and selective precision.
The detailed order is §14.7 / tasks.md **Phase F**.

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
- `doc/prokop/topical_audit/f32_floor_sparse.md` — **bugs vs floor vs missing
  pipeline** (2026-09-10). Read with manifest §0.
- `doc/prokop/topical_audit/sk_interpolation.md` — extra-control B-spline fitter.
- `doc/prokop/topical_audit/f32_floor_dense_hbond.md` — dense H-bond cousin.

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

---

## 13. GPT-5.6 code review — corrections and action checklist

**Source:** GPT-5.6 review of commit `b269ab6`, transcribed in
`Sparse_Nanocrystal_Vibrations.chat.md` from line 2064 onward.

**Summary:** The commit moved the project in the right direction (analytic SK
angular derivatives, D/W formulation, symbolic-plan concept), but several
"completed" items are **not actually wired into the production numerical
path**. The largest gains are not another OpenCL micro-optimization — they are
removing whole unnecessary matrix products, eliminating
synchronization/allocation from the harness, and ensuring the force is
genuinely the derivative of the same smooth self-consistent sparse model whose
Hessian we intend to diagonalize.

**Checkbox convention:**
- `[*]` — done correctly and verified
- `[ ]` — not done / needs fixing / needs rework
- `[~]` — partially done (implemented but not wired into production, or test
  passes but does not test what it claims)

### 13.1 Critical blockers (must fix before phonons)

- [~] **#1 — C² spline not canonical in production force path.** **FIXED:**
  `EqGridTable` now stores canonical C² B-spline control points (computed
  via tridiagonal solve at construction). `eval_into` and
  `eval_with_deriv_into` now use the B-spline path (`eval_bspline_into`
  and `eval_bspline_with_deriv_into`). The tail derivative uses analytic
  B-spline V' and V'' at the last grid point (no finite differences).
  Hermite path retained as `eval_hermite_into` / `eval_hermite_with_deriv_into`
  for reference. Verified: `test_bspline_reproduces_grid_values` (exact at
  grid points), `test_bspline_deriv_c2_continuity` (V and V' continuous at
  knots), `test_bspline_derivative_smoothness` (smooth derivatives),
  `test_bspline_vs_hermite_values` (close to Hermite). The production force
  path (`rotation.rs` → `SkTableSp::eval_shell_integrals_and_derivs_into`
  → `EqGridTable::eval_with_deriv_into`) now uses the canonical C² B-spline.
  *(Manifest §4.2, §3.3)*

- [ ] **#2 — No sparse SCC solver; sparse purification attached to dense SCC.**
  `rhai_run_sparse_purify_geom()` retrieves an already-converged dense
  `SccResult`, converts dense `h_scc`/`s` to BSR4, then purifies. The sparse K
  is computed for a Hamiltonian whose SCC charges came from the **dense**
  solution. For Hessian variational consistency, the production path must be:
  `sparse H0/S → q init → γ·Δq → sparse Hscc → sparse K0+TC2 → Mulliken q → mix
  → repeat`, with no dense `SccResult` anywhere in the loop. *(Manifest §4.12)*

- [~] **#3 — TC2 does ~4 SpGEMMs/iter instead of 2.** **FIXED (A2):**
  `tc2_purify_dev` now computes `R_I=||Q-K||` **before** the update, using
  the same Q from the 2 SpGEMMs. No extra KSK for convergence check. No
  per-iteration host sync — trace and residual are read together only on
  diagnostic iterations. Normal iteration: 2 SpGEMMs, 0 host reads, 0
  queue finish. Verified: `test_tc2_dev_resident_convergence`,
  `test_tc2_dev_vs_host_parity`, `test_tc2_nonconvergence_is_error` all
  pass. *(Manifest §4.7)*

- [~] **#4 — P4 symbolic plans not integrated into TC2/NS/K0/W.** **PARTIALLY
  FIXED (A3):** `SparsePurifyWorkspace::new` now builds and uploads symbolic
  plans for `K*S` and `T*K` at construction. `tc2_products_dev` uses
  `spgemm_plan_bsym_dev` when a plan is available, falls back to
  `spgemm_bsym_dev` otherwise. `SparseDWWorkspace` also builds plans for
  `K*Hscc` and `T*K`. Remaining: `Z*S`, `T_ZS*Z`, `Z*H`, `T_ZH*Z` in the
  Newton-Schulz path. Verified: all 22 `gpu_sparse_bsr4` tests pass, plus
  `test_dw_workspace_vs_oneshot` with zero diff. *(Manifest §4.6)*

- [~] **#8 — `build_dw_sparse()` is an allocation machine.** **FIXED:**
  New `SparseDWWorkspace` pre-allocates all structures, buffers, and
  symbolic plans once at construction. `build_dw_into` enqueues only
  kernels — no allocation, no host transfer. Verified:
  `test_dw_workspace_vs_oneshot` shows zero diff vs one-shot API, and
  workspace is reusable across calls. Legacy `build_dw_sparse` kept for
  backward compatibility. *(Manifest §4.4, AGENTS.md hot-loop rule)*

- [ ] **#13 — Gate E is a false positive.** (a) "Sparse force" is FD of sparse
  **energy**, not analytic sparse D/W force. (b) Hessian h-sweep uses dense f64
  non-SCC path only. (c) Hessian is symmetrized by construction
  (`hess[i][j]=h_ij; hess[j][i]=h_ij`) so measured "asymmetry" is identically
  zero. (d) `h_ref=0.05` is in the candidate list → that entry has exactly
  zero diff, guaranteeing a "plateau." (e) `band_energy()` sums occupied
  eigenvalues without the closed-shell factor 2. **Fix:** wait for analytic
  sparse force (#7), then redo Gate E per §5 Gate E and §13.2 item 9. *(Manifest
  §5 Gate E)*

- [ ] **#10 — Gate C should be marked NOT passed.** (a) 5 atoms on a 1.5Å line
  — at R=7Å the entire 6Å system is inside the mask, so the "plateau"
  approaches dense/full support. (b) `h[i,i]+=2.0` shifts **every eigenvalue by
  exactly 2** and leaves all level spacings (including HOMO-LUMO gap) unchanged
  — it does **not** create a gap. **Fix:** use real passivated Si/H systems
  (Si29H36, Si~80H..., Si~150H...) and test whether a **fixed** R_K gives
  stable errors as N grows. *(Manifest §4.5, §5 Gate C)*

### 13.2 Required rework (correctness/architecture)

- [~] **#5 — Planned SpGEMM kernel contains dead code.** **PARTIALLY FIXED:**
  Removed `A_col`, `C_col` arguments and `lcol[MAX_LEFT_BLOCKS]` local
  array from `bsr4_spgemm_plan_Bsym` kernel. Saves ~256 uint of local
  memory and eliminates useless global loads. Updated Rust kernel
  construction and `spgemm_plan_bsym_dev` to match. Verified: all 24
  `gpu_sparse_bsr4` + `spgemm_plan` tests pass. Remaining: pack
  `plan_a_idx` (8 bits) + `plan_b_idx` (24 bits) into one `uint` to halve
  plan-term traffic. *(Manifest §4.6)*

- [ ] **#6 — 16-lane mapping may reload B 4× unnecessarily.** Current: 16
  lanes = 16 scalar `C_rc`; each lane does 4 global B loads per contribution →
  ~64 B loads. Alternative: 16 lanes = `(m,c)`; each lane reads one `B_mc` and
  accumulates 4 output rows. **Benchmark, don't assume** — NVIDIA caches may
  already handle it. Also: P4 perf test uses 8-atom chain / 10 launches — use
  realistic N=300/600/1000 with degree distributions at R_K~5-10Å, 100-1000
  repeated products. *(Manifest §4.6)*

- [ ] **#7 — Degree buckets not implemented.** `MAX_LEFT_BLOCKS=256` reserved
  for every workgroup regardless of row degree. Implement buckets: ≤32, ≤64,
  ≤128, ≤256. For Si density at ~7Å, expect 64/128 to dominate. *(Manifest
  §4.6)*

- [ ] **#9 — W symmetrization should be diagnostic, not error-hiding.** For
  symmetric H and K, `KHK` is symmetric. If the intermediate mask is the exact
  symbolic product support, the first multiplication should not introduce
  structural truncation. Measure `η_W = ||W-W^T||/||W||`. If ~f32 roundoff,
  don't launch a full symmetrization pass. For force contraction, cheaply use
  `(W_ij + W_ji^T)/2` on the pair. Same question for K symmetrization after
  every TC2 iteration — benchmark every-iter vs diagnostic-iter vs
  threshold-gated. *(Manifest §4.4)*

- [ ] **#11 — `R_leak` formula is mathematically wrong.** Current:
  `R_leak ≈ ||Q_val|| - ||Q_K||` (difference of norms, not norm of omitted
  residual). Correct: `R_leak = sqrt(||Q_val||² - ||Q_inside||²)`. Better:
  compute `Q=KSK` once on validation support, tag each output block
  inside/outside `M_K`, accumulate `sum_{outside M_K} |Q_ij|²` via GPU
  reduction, then `R_leak = sqrt(sum)`. No approximation. *(Manifest §2.1)*

- [ ] **#12 — R_K and R_Z mixed in locality experiment.** Test does
  `z = z_on_mz.project_to_mask(&m_k)` **before** constructing K0, destroying
  R_Z/R_K independence. Correct: `K0 = P_MK[(ε_max·Z - ZHZ)/(ε_max-ε_min)]`
  using full Z on M_Z in `ZHZ`; only the **final result** is projected to M_K.
  When R_Z > R_K, projecting first throws away the extra inverse-overlap
  accuracy. *(Manifest §4.5)*

- [ ] **#16 — Do not bridge to dense force function for production.**
  `non_scc_electronic_force()` allocates multiple `Vec<f64>` per pair (DM
  block, EDM block, H, S, dH/dx,y,z, dS/dx,y,z). Fine as CPU reference. Do
  **not** build a bridge by repeatedly extracting BSR blocks into those Vecs.
  Production: dedicated GPU sparse force path — kernel 1: one pair, evaluate
  SK V,V' + analytic angular derivatives, read K_ij/W_ij, output
  `float4 pair_force[p]`; kernel 2: one atom, gather incident pair_forces, sum
  into F_atom. Precompute pair→BSR block index, transpose index, physical
  orbital counts, species-pair table id. No CSR binary searches in force loop.
  *(Manifest §4.3, §7)*

- [ ] **#17 — E_dummy should not control TC2 condition number.** Since dummy
  orbitals are exactly decoupled (`S_dummy=I`, `Z_dummy=I`, `K_dummy=0` is a TC2
  invariant): exclude dummy orbitals from the physical spectral-bound
  calculation, explicitly init `K0 dummy rows/cols = 0`. Then E_dummy only
  makes the padded generalized matrix nonsingular. Also: ensure SCC H update
  does not turn `H_dummy=E_dummy` into `H_dummy=E_dummy+V_H` (because
  `S_dd=1`). Keep dummy diagonal explicitly fixed. Use `active_orbital_mask`
  (4 bits/atom). *(Manifest §4.10)*

- [~] **#18 — Newton-Schulz not fully device-resident.** **PARTIALLY FIXED:**
  `newton_schulz_inverse_dev()` now computes `||S||∞` on the device via
  `inf_norm_dev()` (row_abs_sum + reduce_max kernels), builds identity on
  the device via `build_identity_dev()` (using diag_block map), and scales
  on the device via `scale_dev()`. No S download, no row_ptr/col_idx
  download, no host identity build. Remaining: Z is still downloaded at
  convergence (needed for K0 construction); for Hessian, `Z = Z0 + few NS
  corrections against new S` without download/reupload is not yet
  implemented. Verified: `test_newton_schulz_inverse_dev` passes. *(Manifest
  §4.12)*

- [~] **#19 — K0/spectral_bounds use allocation-heavy wrappers.** **FIXED:**
  New `spectral_bounds_dev` computes B = Z·H on device, then Gershgorin
  bounds on device (new `bsr4_gershgorin_partial` + `reduce_min_f32`
  kernels), reads only 2 scalars (emin, emax) to host. New `build_k0_dev`
  computes B = Z·H and A = B·Z on device, then K₀ = (emax·Z - A)/Δε via
  `axpby_dev`, symmetrizes on device. No matrix download, no host
  roundtrip. Verified: `test_k0_dev_vs_host` shows zero diff vs host path,
  and TC2 from device K0 converges (R_I=1.6e-7, Tr=3.000000). *(Manifest
  §4.12)*

- [ ] **#20 — Performance audit currently lies.** `let largest_dense = 0` does
  not check anything. The engine has dense `SccResult`, dense H/S, dense→BSR
  conversion, `K.to_dense()`, while audit prints "largest dense alloc 0 bytes."
  Fix: real counters at GPU wrapper/runtime level — buffer allocation count,
  current/peak GPU bytes, kernel launch count, blocking read count/bytes,
  queue finish count, structure/plan upload bytes. Make stage timings
  explicitly exclusive or hierarchical, never sum overlapping timers. The
  `sparse_firewall` catches `Bsr4Matrix::to_dense()` but not a dense
  `DMatrix` created elsewhere — type-level separation of the production sparse
  API is more robust. *(Manifest §4.1)*

- [~] **#21 — Mask construction is O(N²).** **PARTIALLY FIXED:**
  `build_product_mask()` now uses a stamping array (`marks[j] != stamp`)
  instead of `neighbors.contains(&j)`, reducing per-row dedup from O(n²)
  to O(candidates). `build_geometric_mask()` still uses all-pairs distance
  loop — cell list not yet implemented (setup only, not hot loop).
  Verified: `test_boolean_product_mask` passes. *(Manifest §4.5)*

- [ ] **#22 — GPU B-spline cutoff behavior needs boundary tests.**
  `cubic_interp_params()` clamps `i = clamp(i, 1, n_grid-3)`, can extrapolate
  with `t<0` or `t>1` unless caller rejects out-of-range distances. Test real
  Si-Si and Si-H SKF tables at: each knot ±ε, first physical sample, SK bond
  distances, start of cutoff tail, cutoff−ε, cutoff, cutoff+ε — for V, V',
  V''. Verify SKF physical grid origin convention is preserved in the
  canonical sparse H/S path. *(Manifest §4.2)*

### 13.3 What was done correctly (keep)

- [*] **#15 — Analytic angular derivatives are mathematically correct.** The
  ss/sp/pp differentiation follows `∂r/∂R_a = u_a`,
  `∂u_i/∂R_a = (δ_ia - u_i·u_a)/r`, with the pp decomposition
  `V_π·δ_ij + (V_σ-V_π)·u_i·u_j`. Unit handling is consistent (SK radial
  derivative per Bohr, angular 1/r with r in Bohr, final ×ANG2BOHR). **Keep
  this machinery.** The correction is to feed it V,V' from the canonical C²
  spline instead of `EqGridTable` Hermite. *(Manifest §4.3)*

### 13.4 Ordered implementation plan (from GPT-5.6)

The following order is from the GPT-5.6 review's "What I would tell the coding
agent to do next" section. Items 1-9 are required before secondary kernel
tuning (item 10).

- [ ] **Step 1 — Unmark P1/P2/Gate C/Gate E as completed.** P2 angular
  differentiation is implemented, but P1 is not canonical in the sparse force
  call graph; Gate C and E do not yet test their advertised properties. Update
  the report and roadmap accordingly.

- [~] **Step 2 — Unify SK interpolation.** **FIXED:** `EqGridTable` now stores
  canonical C² B-spline control points. `eval_into` and `eval_with_deriv_into`
  use the B-spline path with analytic derivatives. Tail derivative uses
  B-spline V' and V'' (no finite differences). Hermite/Neville retained as
  reference only. *(= issue #1)*

- [ ] **Step 3 — Build genuinely sparse SCC workspace.** Not a sparse
  postprocessor of `SccResult`. H0/S, Hscc, K, q and force must form one
  self-consistent sparse calculation. *(= issue #2)*

- [ ] **Step 4 — Fix TC2 immediately.** Residual before update, no recomputed
  KSK, no trace host read each iteration. Reduces normal iteration from ~4
  SpGEMMs to 2. *(= issue #3)*

- [ ] **Step 5 — Integrate symbolic plans into TC2/NS/K0/W.** Rather than
  leaving P4 as a standalone test. *(= issue #4)*

- [~] **Step 6 — Make one persistent `SparseSystemWorkspace`.** **IMPLEMENTED:**
  New `sparse_system.rs` module with `SparseSystemWorkspace` struct owning all
  GPU structures, buffers, and symbolic plans for the lifetime of a frozen
  topology. Pipeline: Z≈S⁻¹ (device-resident NS) → spectral_bounds (device
  Gershgorin) → K0 (device) → TC2 (device) → Mulliken charges. All SpGEMMs
  use symbolic plans when available. No allocation in hot loops. Verified:
  `test_sparse_system_scc_pipeline` (full pipeline converges, R_I=9.5e-6,
  Tr=3.000010) and `test_sparse_system_reuse` (workspace reusable, both runs
  converge to same state). *(= issues #8, #18, #19)*

- [ ] **Step 7 — Implement sparse analytic force directly on BSR blocks.**
  Preferably pair-force + atom-gather, not through the allocation-heavy dense
  CPU force API. *(= issue #16)*

- [ ] **Step 8 — Redo Gate C on real passivated Si/H systems.** Compute exact
  outside-mask leakage, test whether fixed R_K/R_Z remains accurate as N grows.
  *(= issues #10, #11, #12)*

- [ ] **Step 9 — Redo Gate E only after Step 7.** Using actual SCC analytic
  sparse forces and unsymmetrized force-difference Hessians. *(= issue #13)*

- [ ] **Step 10 — Secondary kernel tuning (only after 1-9 pass).** Packed
  plans, degree buckets, alternative lane mapping, selective compensated
  reductions. *(= issues #5, #6, #7, #9)*

### 13.5 Status corrections to propagate

When Step 1 is executed, the following status corrections must be propagated
to `Sparse_Nanocrystal_Vibrations.report.md`,
`doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md`, and
`doc/prokop/topical_audit/sparse_nanocrystal_vibrations.md`:

| Item | Was | Should be |
|---|---|---|
| P1 | "completed" | `[x]` canonical C² B-spline in production force path |
| P2 | "completed" | `[x]` angular + radial derivatives from canonical C² B-spline |
| Gate C | "passed" | `[ ]` false positive — 5-atom toy system, +2I doesn't create gap |
| Gate E | "passed" | `[ ]` false positive — FD-of-energy forces, symmetrized Hessian tautology |
| P4 | "completed" | `[~]` plan infrastructure built, not integrated into TC2/NS/K0/W |
| P3 | "completed" | `[~]` D/W formula correct, but allocation-heavy and attached to dense SCC |

---

## 14. GPT-5.6 second review (2026-09-11, commit `e965ae0`) — verified findings

**Source:** `Sparse_Nanocrystal_Vibrations.chat.md` lines 3276–4022. Reviews the
state *after* the §13 fixes. **Every claim below was re-verified against the
code before adoption** — file:line references are current, not the review's.

**Headline:** the architecture is finally a persistent solver (one
`SparseDftb`, precomputed plans for KS/KSK, canonical C² B-spline for values
*and* derivatives, Rhai CLI). But it is **not yet a large-system sparse
solver**: three blockers (dense host storage, dense O(N³) force, non-stationary
SCC state), four HIGH items, three MEDIUM, and one solved mystery (N4 = missing
`sqrt`, not an f32 floor). No reason to abandon the f32 BSR4 + TC2 design —
the remaining limits are contracts, dense leftovers, and convergence
definitions, not f32 itself.

### 14.0 The N4 mystery is solved — contract bug, not f32 floor

`identity_residual_to_dev` computes ‖A−I‖²_F; `identity_residual_scalar_dev`
(`gpu_sparse.rs:2006-2021`) returns the scalar **without `sqrt`**;
`newton_schulz_inverse_dev` (`:2322`) divides only by √N_orb. The reported
"R_Z≈1.9e-5" was therefore ‖I−T‖²/√N — matching the independently measured
max|Z−S⁻¹|≈2.18e-3. A 100–200× "numerical failure" was a norm-vs-norm²
contract bug.

**Generalized rule:** every diagnostic scalar has a *contract* (which norm,
which normalization). A small printed residual is meaningless until the
residual's *definition* is cross-checked: device reduction vs host-f64
recomputation of the identical matrix, agreeing to ordinary f32 reduction
accuracy. Apply this to every new reduction kernel.

### 14.1 BLOCKERS — fix before any kernel tuning

- [ ] **R1 — Dense host storage inside `SparseDftb`.** Verified:
  `h0_phys`/`s_phys` f64[N²], `h0_pad`/`s_pad`/`h_scc_pad`/`k_pad` f32[(4N)²]
  (`sparse_dftb.rs:128-133`); `set_coords` builds dense `DMatrix` via
  `build_non_scc`, copies N², pads, then extracts BSR (`:273-288`). At N=1000
  Si (N_orb=4000) persistent host arrays alone ≈ 512 MB — linear scaling dies
  on the host before GPU sparse algebra matters.
  **Action:** direct sparse H/S assembly into BSR values: iterate the frozen
  physical pair list, evaluate SK(r)+rotation, write the 4×4/4×1/1×1 block
  into BSR block b(i,j); write onsite diagonal blocks; dummy diag S=1,
  H=E_dummy. CPU first is fine. No `DMatrix`, no `h0_phys`/`h0_pad`, no
  `fill_bsr_values_from_dense` in production. Hessian bonus: displacing atom
  a then updates only blocks involving a — O(n_neigh), not a rebuild.
  (Absorbs tasks.md B2.)

- [ ] **R2 — Sparse force is a dense O((4N)³) host computation.** Verified:
  `forces()` → `sparse_analytic_forces` → `dw_from_k_padded`
  (`sparse_forces.rs:259-294`): two naive triple-loop f64 matmuls —
  ~1.3×10¹¹ FMA at N=1000, dominating everything.
  **Action (staged):** `SparseDWWorkspace` already has persistent T/W buffers
  and plans for K·Hscc and T·K. Stage 1: GPU T=K·Hscc, W=P_{M_HS}(T·K);
  download only W[M_HS] and K[M_HS]; keep the tested CPU pair contraction.
  Never materialize D=2K (spin factor inside the contraction). Stage 2: GPU
  pair-force + atom-gather kernels (tasks C1/C2). Fuse W→force later only
  after the simple version is validated.

- [ ] **R3 — SCC finalization is not one stationary state.** Verified:
  `finalize_scc` (`sparse_dftb.rs:373-398`) builds V[q_new], H[q_new],
  purifies → q_fin, then stores `self.q = q_fin` while `self.v` and
  `h_scc_pad` still belong to q_new; `store_energy(q_fin)` evaluates
  dq_fin·V[q_new]; the force gets q_fin with H[q_new]. Identical only at
  exact convergence — geometry optimization and Hessians live at finite
  tolerance.
  **Action:** explicit state contract q_in → H[q_in] → K → q_out; iterate the
  finalization until R_SCC = rms(q_out − q_in) is below the final tolerance;
  store both q_in and q_out for diagnostics. H, K, W, V, energy and force
  must provably belong to one identified state. W=2KHK has the EDM meaning
  only when K is the projector of *that* H — TC2 idempotency alone does not
  guarantee it.

### 14.2 HIGH

- [ ] **R4 — One structural mask for H/S, K and Z.** Verified: single `mask`
  (`sparse_dftb.rs:196`) used for h_bsr/s_bsr/hscc_bsr *and* passed as k_mask
  into the workspace (`:211`); Z lives on M_K. Destroys the intended
  independent R_K/R_Z locality control — the whole point of Gate C.
  **Action:** independent `R_HS` / `R_K` / `R_Z` masks and product plans.
  Do **not** project Z to M_K before ZHZ — use all of M_Z in ZHZ, project only
  the resulting K0 to M_K (old issue #12 restated).

- [ ] **R5 — NS squared-norm contract bug** (= §14.0). Rename to
  `identity_residual_sq` or add `.sqrt()`; re-validate device NS; then delete
  the per-iteration full-T download in `SparseSystemWorkspace::compute_z`
  (`sparse_system.rs:359`) — it exists only as the N4 workaround and reads an
  entire sparse matrix every NS iteration. Leave one scalar diagnostic read.

- [ ] **R6 — Persistent-Z "identity reset" leaves stale off-diagonals.**
  Verified: `bsr4_build_identity_dev` (`.cl:1431-1443`) writes only diagonal
  blocks; `scale_dev` then scales **all** blocks → on geometry ≥2 the initial
  Z is neither αI nor a valid warm start.
  **Action — choose deliberately:** (a) cold start: one kernel writes *every*
  structural entry (physical diag → 1, everything else → 0); or (b) warm
  start (better for Hessians): retain central Z and NS-correct it against the
  new S — one step gives ≈ Z − Z·δS·Z, the first-order inverse correction,
  so ±h displacements should need only a few iterations. Use the *same
  central Z* for +h and −h independently (no history asymmetry).

- [ ] **R7 — K downloaded + densified every SCC iteration just for energy.**
  Verified: `k_to_dense_into` inside the loop (`sparse_dftb.rs:337`) +
  `trace_ab` over the full padded N² (`:418`). Major sync + N² CPU overhead.
  **Action:** E_H0 = 2·Tr(K·H0) and H0 exists only on M_HS — precompute an
  HS-block → K-transpose-block map, GPU f32-FMA block dot products → one f32
  partial per atom → host f64 sum of ~1000 partials (~4 kB transfer, no GPU
  fp64 needed). Make per-iteration energy a configurable diagnostic;
  production needs only the final energy.

### 14.3 MEDIUM

- [ ] **R8 — Spare products per SCC iteration.** Verified: (a)
  `mulliken_charges` recomputes K·S though `t_ks` already holds it after TC2
  accepts (`sparse_system.rs:253-256`) — reuse it; (b) `compute_k0_from_hscc`
  computes ZH twice — `spectral_bounds_dev` (`gpu_sparse.rs:2209`) then
  `build_k0_dev` (`:2244`) — compute ZH once, feed both Gershgorin and ZH·Z;
  (c) `plan_zh`/`plan_bz` are built (`sparse_system.rs:181-182`) but unused —
  route both products through plans; Z·H *is* Bsym-suitable (Bsym requires
  the **right operand** to be symmetric, not the result).

- [ ] **R9 — γ recomputed O(N²) every SCC iteration.** Verified:
  `compute_intra_shifts` in the loop (`sparse_dftb.rs:326`, again in
  `finalize_scc:375`) recomputes every R_ij and γ(R_ij) incl. exponentials
  although geometry is fixed (`qmqm/shifts.rs:38-55`).
  **Action:** precompute dense f64 G[N×N] once per geometry (8 MB at N=1000);
  per iteration V = G·Δq in CPU f64 (~10⁶ FMA — cheap, no GPU sync, no f32
  cancellation). Hessian: patch only the moved atom's row/column, O(N).
  GPU matvec only if profiling later demands it.

- [ ] **R10 — "Persistent" helpers still allocate GPU buffers in-loop.**
  Verified: `mulliken_dev` allocs qbuf (`gpu_sparse.rs:1982`);
  `gershgorin_bounds_dev` allocs emin/emax partials (`:2125-2126`);
  `identity_residual_scalar_dev` allocs 4 buffers (`:2013-2016`);
  `reduce_min_dev`/`reduce_max_scalar_dev`/`reduce_to_one_dev` allocate per
  recursion level (`:1968`, `:2150`, `:2176`); `frobenius_norm_dev` allocs
  nblock·16 zeros (`:1999`); `trace_ks_dev` allocs (`:1939`).
  **Action:** move all into `SparseSystemWorkspace` (q_atom, gersh min/max,
  reduction scratch, two-scalar diagnostic buffer). Enforce with an
  allocation counter: zero `Buffer::builder` between `scc()` entry and exit.

### 14.4 Physics/testing-quality items

- [ ] **R11 — Final Hamiltonian-compatibility residual R_H, cheaply.**
  Production leaves `r_h = NaN` (`sparse_dftb.rs:218,346,389`). After final
  SCC convergence only: SKH = (HKS)ᵀ by symmetry and T=KS already exists at
  TC2 end → compute A = H·T once, measure ‖A−Aᵀ‖_F/(2‖A‖_F+ε). One extra
  SpGEMM at finalization; catches a beautifully idempotent projector onto the
  wrong subspace.

- [ ] **R12 — Normalize TC2/SCC convergence.** Raw ‖KSK−K‖_F < 1e-4 is
  size-dependent: at N=1000 (~8×10⁵ stored scalars) it demands ~10⁻⁷ RMS per
  scalar — f32 machine precision; a reasonable SiH4 tolerance becomes absurd
  purely because N grew. Converge on r_I = ‖KSK−K‖_F/max(‖K‖_F,ε) (or at
  least /√N_occ) and r_N = |Tr(KS)−N_occ|/N_occ. Keep raw values for
  debugging. Prevents mislabeling normal f32 saturation as failure and saves
  useless late iterations. (NS already does the right thing via /√N.)

- [ ] **R13 — Hscc built on GPU from N atom potentials; dummy-exact.**
  `bsr4_build_Hscc` exists (`.cl:755`) but (a) is unused — `SparseDftb` runs
  CPU `apply_shift_padded_into` + uploads all Hscc blocks per iteration
  (`sparse_dftb.rs:327-329`) — and (b) adds dV to all 16 lanes, turning
  H_dummy=E_dummy into E_dummy+V because S_dd=1 (old #17, still open).
  **Action:** upload only V[N] per iteration; kernel writes
  H = H0 + ½(Vi+Vj)·S on active lanes and leaves the dummy diagonal at
  E_dummy exactly (active-orbital mask per §4.10).

- [ ] **R14 — Mulliken must ignore dummy lanes; tighten charge check.**
  Verified: `bsr4_mulliken_KS` sums all four diagonal lanes (`.cl:870-879`);
  charge acceptance is |Σq−N_e| < **0.5 e⁻** (`sparse_dftb.rs:408`) — half an
  electron cannot catch implementation errors.
  **Action:** pass n_orb/4-bit active mask (Si lanes 0,5,10,15; H lane 0);
  report dummy occupation as a separate diagnostic. Replace the 0.5 check
  with the internal identity Σ_A q_A ≈ 2·Tr(KS) (same K,S → tight agreement)
  and |2·Tr(KS)−N_e| vs the TC2 rank tolerance.

- [ ] **R15 — Verlet-skin semantics, not rebuild-and-demand-equality.**
  Verified: `set_coords` rebuilds the O(N²) geometric mask every geometry and
  errors on any difference (`sparse_dftb.rs:263-271`) — the skin is paid for
  but never used, and mask construction is still all-pairs.
  **Action:** build structural support at R_phys+R_skin once, store build
  coords, rebuild only if 2·max_i|ΔR_i| > R_skin; inside the structural mask
  H/S are simply zero beyond the physical cutoff. For Hessians choose the
  skin to cover all ±h → never rebuild; removes mask jitter by construction.

- [ ] **R16 — Per-call CPU force setup + repulsive reparse + Bohr/Å wart.**
  Verified: `compute_forces_from_dw` rebuilds `SystemContext`, `GammaTable`,
  the neighbor list, and re-parses every repulsive spline per call
  (`forces.rs:1303-1345`); `repulsive_energy` re-parses SK files every
  `set_coords` (`sparse_dftb.rs:291` → `forces.rs:1067`). For a 6N-force
  Hessian this is filesystem/string parsing in the hot loop.
  **Action:** `SparseDftb::new` owns SystemContext, repulsive tables
  (`parse_all_repulsive` once), species-pair indices, the physical pair list,
  gamma species coefficients — never reconstructed.
  **Units wart:** SK `cutoff()` is Bohr but is passed raw to
  `NeighborBuilder` over Å coords (`forces.rs:1316-1317`) → neighbor radius
  ~1.89× too large. SK eval returns zero for the excess pairs so this is
  mostly wasted work, not an energy error — still fix.

### 14.5 f32 policy — measure, do not blanket-f64

Confirmed direction: pure f32+FMA for K/S/H/Z/W storage, all SpGEMMs, SK GPU
eval, Hscc, pair contractions. Selective upgrades only:

- **compensated f32** for long signed reductions (gamma matvec, force gather,
  trace) — cheap because these kernels are tiny vs TC2;
- **host/tiny-device f64** for scalar energies, SCC mixing/DIIS, Hessian
  assembly, frequencies, final diagnostics;
- **multi-accumulator SpGEMM** — 2–4 independent partial sums over successive
  plan terms; shortens the FMA dependency chain (speed) *and* reduces serial
  roundoff depth (accuracy). Benchmark this **before** any Kahan-in-SpGEMM;
- combine trace + R_I² into one device packet → **one** blocking read;
  benchmark check_every = 2–3 (TC2 is expensive; balance sync saved vs
  overshoot);
- f32 trace is **not** the current limiter: 1 ULP ≈ 1.2×10⁻⁴ e⁻ at
  N_occ~2000 vs the 0.05 e⁻ acceptance width. Compensate the trace only if
  branch jitter is demonstrated;
- no compensated SpGEMM everywhere unless a frozen-input experiment proves
  real cancellation.

### 14.6 Interpolator + benchmark scope

- The right-tail zero-sample refit remains a documented stopgap; before final
  vibration work replace with the constrained extra-control f64 fitter of
  `sk_interpolation.md` / §0.3 — V=V'=V''=0 at the chosen cutoff, tabulated
  region preserved, zero runtime cost on GPU. (Review §19 = existing plan.)
- The SiH4 CLI proves persistence, **not** sparsity (n≤64 → full mask,
  `sparse_dftb.rs:45,195`). The benchmark is the same CLI on real
  Si~100/300/600/1000 H-passivated systems printing stage timings, nnz and
  row-degree distributions, allocation counts, host-transfer bytes, TC2/SCC
  iteration counts, and *normalized* residuals.

### 14.7 Revised implementation order (supersedes §6 where they conflict)

| # | Action | tasks.md |
|---|--------|----------|
| 1 | Fix NS contract bugs (sqrt + stale identity); revalidate device NS; delete per-iteration T download; use NS plans | F1 |
| 2 | Remove dense matrices from `SparseDftb`: direct BSR H/S assembly; separate M_HS/M_K/M_Z; Verlet skin | F2 (absorbs B2) |
| 3 | Eliminate `k_to_dense_into` from SCC: masked sparse energy; reuse final t_ks for Mulliken; single ZH via plan_zh/plan_bz; kill in-loop buffer allocs | F3 |
| 4 | Stationary SCC finalization (consistent q_in/q_out) + final normalized R_H | F4 |
| 5 | Sparse force: `SparseDWWorkspace` T/W on GPU → download W[M_HS],K[M_HS] → CPU pair contraction; then GPU pair-force + gather | F5 (= staged C1–C3) |
| 6 | Per-geometry/static precomputation: gamma matrix, repulsive tables, SystemContext, pair list; fix Bohr/Å neighbor cutoff | F6 |
| 7 | f32 arithmetic pass: normalized tolerances, GPU Hscc/Mulliken dummy-exact, multi-accumulator SpGEMM, compensated reductions, packed diagnostic read | F7 |
| 8 | Kernel micro-opts: packed plan indices, degree buckets, fused W→force, lane mapping | Phase E (E3–E6) |

Then gates D1–D5 and scaling E7–E8 as before.

### 14.8 Development discipline — why items marked FIXED still had bugs

The first review's "FIXED" items still contained contract bugs (N4 residual
semantics, stale-Z reset, unused plans). For this task the following are
**exit criteria, not aspirations**:

- A reduction/diagnostic kernel is not done until its scalar is cross-checked
  against a host-f64 recomputation of identical inputs (would have caught N4).
- "Persistent" is not done until an allocation counter shows zero
  `Buffer::builder`/`Kernel::builder` between `scc()` entry and exit
  (would have caught R10).
- "Sparse" is not done until counters show zero dense Norb×Norb allocations
  and zero matrix host transfers inside SCC/force — the §4.1 firewall must
  *count*, not `let largest_dense = 0`.
- Convergence criteria live on normalized residuals (R12); raw Frobenius over
  N² data is a diagnostic, never a tolerance.
- State claims in reports/tests must name the state: which q, which H, which
  K produced this energy/force (would have caught R3).
- Nothing is marked `[*]`/FIXED without the test output shown to USER. A red
  test documenting real physics beats a green test asserting the wrong norm.
