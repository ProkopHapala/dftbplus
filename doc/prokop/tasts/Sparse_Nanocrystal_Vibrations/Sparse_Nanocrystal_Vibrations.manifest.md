# Task 2: Sparse GPU DFTB Forces and Vibrations of Si/Diamond Nanocrystals

**Created:** 2026-09-07  
**Revision:** v4 — absorbs second GPT-5.6 review (commit `e965ae0`, §14)  
**Last wrap (2026-09-11):** read **§0** first, then **§14** — second review:
three remaining blockers, the N4 contract bug, revised order §14.7. Floor vs bug map:
`doc/prokop/topical_audit/f32_floor_sparse.md`. Interpolator:
`doc/prokop/topical_audit/sk_interpolation.md`. Dense H-bond is a **separate**
agent — do not edit `qmqm/gpu_forces.cl`.
**Current sparse work order: §15.** Source review of P/TRS, W, screening, and GPU limits; supersedes conflicting completion/stability claims in prior reports. Review notes only; no solver changes or fresh benchmarks.

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

#### 4.12.0 Measurement policy — NEVER run a full Hessian to measure it

A displaced-eval experiment must answer "per-eval cost + iteration
counts", not produce a Hessian. Rules:

- **HARD BOUND — `timeout 30` on EVERY engine invocation, no
  exceptions.** Not for "production" runs, not for "it's almost done",
  not for final artifacts. I violated this once with `timeout 1500` on
  a "production" Hessian — exactly the bug this rule prevents. If a run
  cannot finish in 30 s, bound its *work* (fewer columns/steps), never
  extend the timeout.
- **`RUST_DFTB_VIB_MAXCOL=n`** — bounds the FD columns (e.g. 4–8 columns
  = 8–16 evals); per-column phase times (`rst+geom`, `solve+f`) and the
  purify/SCC iteration counts are always printed, so ~8 columns
  extrapolate to 6N exactly. A bounded run exits before the eigensolve.
- **A full Hessian is a user-launched artifact only** (e.g. an overnight
  batch the USER starts) — never an agent-launched unbounded loop.
- Same discipline for relax/scan scripts: bound the step count or time
  before launching.

#### 4.12.1 Measured status (2026-09-15) — per-eval bottleneck analysis

Micro-benchmark (`scripts/bench_eval_r10.rhai`, R10 = 330 Si, ±0.02 Å on
atom 0, `RUST_DFTB_PROF=mark`; full detail in
`doc/prokop/topical_audit/hessian_eval_bottleneck.md`):

- [*] Per-phase breakdown measured: `set_coords` ≈ **5 ms** (CPU H0/S
  assembly 3.4 + dense f64 γ O(N²) 0.9 + repulsive 0.3 + uploads),
  `scc` ≈ **3.4–7.3 s**, `forces` ≈ **7 ms**. SCC is >98% of an eval.
- [*] Structure of the waste measured: ~9–17 DIIS mix iters × ~41–55 TC2
  iters ≈ 450–900 SpGEMM iterations per displaced eval. Every mix iter
  rebuilds `K0/P0 = f(Z·H_scc)` from scratch (`compute_k0_from_hscc` /
  `compute_p0_from_hscc`) — the converged projector is never reused.
- [*] `tc2.tr` = 87–95% of host time is the one blocking read per TC2
  iteration draining the queued K·S·K — per-iteration serialization.
- [*] **Mode A (frozen DM) implemented** — `forces_frozen()` +
  `RUST_DFTB_VIB_FROZEN`: R10 Hessian = **23 s** (12 ms/eval, 1980
  evals). Accuracy on si10h16: framework modes ±1–9 cm⁻¹ of full SCC,
  but Si–H stretches **~240 cm⁻¹ soft** (charge response stiffens the
  top modes). Preview/framework tool, not quantitative. *(2026-09-16:
  re-implemented as the explicit clamped-electron freeze — `W₀`
  snapshotted, ZERO device products → 5.5 ms/eval, ~6.3% column error
  on R10; see §4.12.1 mode-C update / report §15.17.)*
- [*] **Warm-seeded TC2 implemented and REFUTED** —
  `purify_hscc_warm()` + `RUST_DFTB_WARM_K`: seeding stored converged K
  walks away from the fixed point (r_I doubles every eval, nonsense
  self-consistent charge state reached). Consistent with this section's
  warning: pure polynomial purification cannot rotate the occupied
  subspace. A Hamiltonian-aware update is required.
- [*] **Central-state snapshot/restore DONE** —
  `snapshot_electronic_state` / `restore_central_state` on `SparseDftb`
  (stores q, K, Z, K0; resets DIIS fully per column). Every ±h column
  now starts from the identical central state — implemented in
  `rhai_sparse_vibrations`.
- [*] **Mode B (fixed-q) DONE and validated** — `scc_fixedq()`: NS warm
  (2 iters) + ONE cold purify, no DIIS. si10h16 vs full SCC:
  **rms 6.1 cm⁻¹ / max 10.7** (frozen-DM was 162/344). R10 per eval:
  ~0.36–1.0 s (one purify).
- [*] **Mode C attempt #1 REFUTED with diagnosis** — δK0 seed
  `K_conv + (K0_new − K0_center)` (`k_seed_shift` + `RUST_DFTB_VIB_DMUPD`)
  lands at R_I~1e-3 (300× closer than cold K0) but is *repelled* by TC2
  (diverges 1.35e-3→2.2e-2). Adding McWeeny polish (`mcweeny_polish`,
  contracting 3KSK−2KSKSK) makes it converge in ~7 iters — **but to a
  wrong-subspace projector**: idempotent yet R_H=1.4e-3 vs cold ~4e-7
  (si10h16 spectrum off ~300 cm⁻¹). Conclusion: purification can only
  enforce idempotency + trace, never the occupied-subspace *selection* —
  that information comes only from the H-containing K0 basin. A correct
  warm update must minimize ‖[K,H]‖ (LNV/DMM commutator descent), not
  polish idempotency. R_H gate (`rh_stationarity`, default 5e-4) now
  validates warm results and cold-restarts on failure.
- [*] **Mode C SOLVED via DMM commutator descent (2026-09-16)** —
  `dmm_descend` (`sparse_system.rs`), `δK=−η(X+Xᵀ−2Y)` with `X=(Z·H)·K`,
  `Y=(K·S)·X`, Z=S⁻¹: 3 SpGEMMs/step + planned McWeeny retraction every
  `ret` steps. Defaults `VIB_DMM=6, VIB_DMM_ETA=8, VIB_DMM_RET=2,
  VIB_MCPOL=0, VIB_TC2MAX=0` (post-polish *raises* R_H — off). R10,
  h=0.05 Å: R_H 4.2e-5 (vs cold 8e-5), **ΔF 0.30% vs cold fixq**,
  ~260–340 ms/eval ≈ 25% faster than cold. Bugs fixed: Z=S⁻¹ root cause
  of the ascent direction; bsym plan on asymmetric X → generic
  `plan_tk_g`; redundant `T·ZHZ=Xᵀ` removed. Report:
  `reports/2026-09-16_sparse_dmm_warm_density_hessian.md`.
- [*] **Stripped tiers + frozen-orbital correction (2026-09-16, same
  report)** — the "stale b_zh" was the CORRECT clamped-electron
  approximation (`W=W₀`, not `Z(R)H(R)K₀` — the hybrid keeps half a
  cancelling response → 86% err). Now explicit `W₀` snapshot;
  `forces_frozen` = 0 device products, **5.5 ms @ 6.3% column error**.
  New `VIB_LITE`/`VIB_LINEAR` strip all per-eval residual gates (~40%
  of warm-eval cost was certification). Measured ladder (h=0.02):
  clamped 5.5 ms/6.3% → **1 Newton + DMM2-lite 64 ms/3.1%** →
  **+DMM4 105 ms/1.0%** → cold 345 ms. Z accuracy is the discriminator
  (central Z caps at ~6%; 1 Newton update → R_Z 1.4e-5 unlocks <6%).
  Refuted by measurement: linear1 (6.7% ≈ frozen at 6× cost), metric
  transport (99%), retractions in lite mode (H-blind, hurt). Remaining
  10× lever: batch-parallel ±h columns.
- [*] **Floor-stop wins measured** — `RUST_DFTB_VIB_TC2TOL=5e-5` (above
  the deg330 floor 4.1e-5): purify converges at **25 iters** instead of
  plateau-churning to 80. Same Hessian accuracy (si10h16 rms 6.3 vs
  6.1 cm⁻¹). R10 fixq eval: **~355 ms** → full Hessian ≈ 12 min.
- [*] **`RUST_DFTB_VIB_MAXCOL` + per-column phase timing** — bounded
  measurement (8 cols ≈ 6–14 s) replaces full-Hessian measurement runs.
- [ ] **SCC ≤ ~3–5 mix iters for displaced evals** — investigate why
  DIIS needs 9–17 at h=0.02 Å (floor noise vs genuine sloshing; measure
  on complete-mask reference).
- [ ] **Purify stops at its measured floor** — never iterate to
  `tc2_max` on a plateau (W=28 detector exists, env-gated; wire it into
  the Hessian path default).
- [ ] **Per-TC2-iter host sync elimination** — device-side branch or
  fused multi-step; only after iteration counts are cut.
- [ ] **GPU H0/S + γ assembly** (B2/B3 below) — deferred until SCC is
  cheap; ~5 ms/eval at N=330, matters at N≳2000.
- [ ] **Batch ±h columns** — independent engines sharing frozen
  topology/plans; the product-level fix for 6N serial evals.
- [ ] **Validate the mode ladder A<B<C<D on cube65** — force and
  frequency error vs cold-SCC reference before any mode becomes default.
- [ ] Performance target (GPT-5.6 review): frozen ~5–20 ms, fixed-q
  ~10–50 ms, warm-SCC ~20–100 ms per eval at N=330.

#### 4.12.2 TC2 numerical-floor study (2026-09-15) — measured + GPT-5.6 directives

Controlled experiment (`scripts/tc2_conv_study.rhai` +
`plot_tc2_convergence.py`, `RUST_DFTB_TC2_HIST` CSV): ONE cold
`purify_now` (K0 rebuilt from a fixed near-converged H_scc) on R10
(330 Si, deg_k=deg_z=330 — full atom support, so mask truncation is NOT
the floor here). `debug/tc2_conv_r10.png`.

**Measured floor ladder** (tol=1e-9 → runs until a stop identifier cuts):

| variant | min R_I | iters | note |
|---|---|---|---|
| ACC4 (default) | 1.78e-5 | 150 (max) | period-2 sawtooth at floor |
| ACC8 | 1.75e-5 | 119 | same sawtooth |
| ACC16 (new) | 1.32e-5 | 81 | same sawtooth; ~2× faster/iter (6.2 vs 12.4 ms — ILP) |
| Kahan f32 | 9.7e-6 | 39 | floor barely moves; trace runaway → abort |
| McWeeny endgame | 5.4e-6 | 150 | smooth monotone — but GLACIAL |
| Kahan+McW | 5.4e-6 | 150 | identical (see directive 3 — expected, not evidence) |

**GPT-5.6 analysis (chat.md L7875+) — reflection:**

- The sawtooth is **intrinsic to trace-correcting TC2 at finite
  precision**: near the fixed point the two branches apply
  δ→2δ / ε→ε² vs δ→δ² / ε→2ε — a finite error η cycles as
  η→2η→η. No accumulator scheme can cure it; ACC4→16→Kahan only
  shifted η (1.8e-5→1.3e-5→9.7e-6). **ACC studies closed**; keep ACC16
  only if its ~2×/iter speed reproduces.
- **Correct production form:** TC2 to ~R_I 1e-3–1e-4 (rank+trace
  locked), then McWeeny `K' = 3KSK − 2KSKSK` for the endgame — contracts
  BOTH subspaces (ε'≈3ε², δ'≈3δ²), no branch saddle. TC2 must never
  enter the 1e-5 sawtooth in production.
- **The measured McWeeny descent is ANOMALOUS**: quadratic convergence
  should do 1e-5→~1e-10 in ~1 step; observed ~5e-6 after >100 iters
  means each step INJECTS ~1e-6–1e-5 error. Prime suspect: the extra
  products (Q·S, U·K) run on the *masked* bsym kernel — the least
  accurate multiply in the solver, no ACC/Kahan. Kahan+McW identical is
  therefore EXPECTED, not evidence Kahan is irrelevant.
- **"Residual measurement is fine" was too strong**: the f64 host finish
  is exact, but Q=KSK is f32-produced — measured R_I conflates state
  error with product error. One-shot CPU-f64 residual of saved best-K is
  still owed.
- **Dummy-orbital clue**: a prior Hessian run reported
  `dummy |D_ii| sum = 2.4e-5` — same scale as the floor. Decompose R_I
  into physical vs dummy lanes; if the floor lives in dummy lanes this
  is a padding-contamination BUG, not precision.
- McWeeny trace drift ~1e-6 is also consistent with inaccurate products;
  do NOT rescale K per McW step (α-scale destroys idempotency by
  O(|α−1|) and can set a new floor).
- **Production endgame may be P-space McWeeny**: with P=KS locked,
  P' = 3P² − 2P³ is only 2 SpGEMMs/iter (plan_pp exists), then K=PZ
  once for forces. Debug K-McWeeny first (directly comparable), then
  switch if it works.
- **Disagreement logged**: "needs f64 state" is NOT established. Find
  the per-iteration error injection before paying for f64.

**Ordered action list (GPT-5.6) — results 2026-09-15:**

- [x] Stop ACC studies — floor is map-intrinsic (ACC16 kept for speed only).
- [x] **F64-DIAG** (`sparse_ri_f64`, dense f64 ‖KSK−K‖/‖K‖ of the
  device K): K-TC2 measured 1.78e-5 → **true 4.1e-5** (measurement
  UNDRESTIMATES ~2.3×, not over — GPT's Case B: the state is at the
  floor, not the residual kernel). K+McWeeny measured 5.35e-6 → true
  1.42e-5 (best state). Trace exact (459.000025).
- [x] **DUMMY-DECOMP**: dummy-lane contribution **exactly 0.0** — the
  floor is 100% physical lanes; no padding contamination (the 2.4e-5
  dummy-occupancy abort was a different bug).
- [x] **MCW-PLAN**: McWeeny products routed through plan_ks/plan_tk
  (also fixed: U=Q·S was truncated onto M_K, now on M_TKS — trace drift
  4.4e-7→7e-8 fixed). Floor unchanged: 5.35e-6; +Kahan identical →
  product accuracy is NOT the limiter either.
- [x] **MCW-SHORT**: the per-iter history IS the plot — McWeeny descent
  is smooth but asymptotic (1e-5→5.4e-6 over 120 iters, still
  descending), not quadratic.
- [x] **P-space purifiers measured**: P-TC2 floors at R_I(P)=5.6e-6 and
  TRS4 at 6.0e-6 — BUT the recovered K=PZ is 7–8e-5 (Z recovery +
  truncation loses it). P-McWeeny actually DRIFTS UP (1.2e-5→1e-4 over
  100 iters — P is non-normal; the symmetric-map assumption fails).
- [ ] **F64-ENDGAME**: see the mechanism below before deciding.
- [ ] **P-MCW**: P-space McWeeny REJECTED (non-normal iterate drifts).

**Mechanism — CORRECTED (GPT-5.6, chat.md L8397+):** my earlier claim
"the floor is occ↔unocc rotation error that only H-aware descent fixes"
was WRONG. A rotated projector still satisfies P²=P exactly — the
idempotency-defect linearization (PE+EP−E) cancels E_ou at first order.
Subspace rotation produces NO first-order R_I; **stationarity error and
idempotency error are different problems** (R_H vs R_I — report BOTH).
The corrected reading: TRS4 arriving at the same ~1e-5 wall by a very
different trajectory, plus f64-measured 1.4e-5 on the stored state, plus
zero dummy contribution, plus insensitivity to product accuracy — all
point to a genuine **f32 representation floor** for this matrix.

**Ordered action list, round 2 (GPT-5.6):**

- [x] **F64-MCW** (`sparse_mcw_f64`, `matmul_f64_mt` host f64 dense):
  **quadratic collapse confirmed** — stored best K (true R_I=1.42e-5)
  → 4.0e-9 → 2.2e-15 → machine eps in 3 steps. **f32 precision IS the
  cold-purifier floor**; the map is healthy, no algebra bug.
- [x] **A/B storage-vs-arithmetic**: variant A (f32 storage, f64
  products) stalls at **1.3e-8** — f32 storage quantization. Variant B
  (all-f64) → 8e-16. Read: the ~1e-5 f32 floor is accumulated PRODUCT
  ARITHMETIC noise (one f64 step collapses it); f32 STORAGE alone
  supports ~1e-8. **Production polish = f32-state + 1–2 high-precision
  products → ~1e-8, ~1000× below the floor.** All-f64 only needed below
  ~1e-8.
- [x] **R_H reported alongside R_I**: mcw_f64 prints both per step
  (CSV col 3). On THIS run R_H=9.2e-2 — but H_scc came from only 4 SCC
  mixers (rms 5e-2), so the subspace leg is unmeasured; redo on a
  converged H_scc before warm-DM work.
- [ ] **TRS4 as fast-approach solver**: reaches the wall in fewer iters;
  compare by SpGEMM count/wall time, not iters (2 products vs 1 per
  iter). Candidate production form: TRS4 approach → precision endgame.
- [ ] **Force-noise-vs-polish study**: measure max|ΔF| on the SAME
  geometry for K polished to R_I ~ 3e-5 / 1e-5 / 1e-6 / 1e-8 — the
  Hessian needs δF/h, not small R_I per se. The 1e-5 floor may already
  be sufficient; the hour-long Hessian was mostly REFUSING TO STOP at
  the floor.
- [x] **FF32-POLISH (chosen design, chat L9436+) — IMPLEMENTED +
  VALIDATED 2026-09-17**: emulated float-float (`float2(hi,lo)` ≈
  40–48 bit) McWeeny polish, all FP32/FMA. Results (R10): products
  verified 7e-15–9.7e-14 vs masked-f64; r_k=40 Å → R_I 2.7e-7→
  **1.8e-8** (f32-storage fixed point ~1.4e-8); r_k=20 Å → ~3e-5
  (M_K storage tail 5.3e-6 dominates — arithmetic CANNOT beat mask).
  Cost ~44 ms/step ≈ **3 f32 iters** (spec target was ≤8). In-loop
  trigger `RUST_DFTB_TC2_FF=1` + `FF_SWITCH`/`FF_STEPS` (terminal —
  returning to f32 re-pollutes). Bug found: fused product+combine
  kernel lost compensation to OpenCL compiler reassociation of the
  TwoSum chain (1.7e-7); split path + explicit-fma combine is bitwise
  = host emulation. Details: report §15.14.
- [ ] LNV/commutator (‖[K,H]‖) descent remains REQUIRED — but for the
  warm-DM subspace-rotation problem (G3), not for this floor.

**FF32-POLISH spec (GPT-5.6, chat L9520–9776):**

- TwoSum/TwoProd via f32 FMA (`p=a*b; pe=fma(a,b,-p)`); NO
  -cl-fast-relaxed-math / reassociation on these kernels (current
  program has no fast flags — keep it that way).
- Only `f32×f32→ff` and `ff×f32→ff` products needed (right operand is
  always f32 K or S); never `ff×ff`.
- Dataflow per polish step, reusing existing plans:
  `T_ff=K·S (plan_ks)`, `Q_ff=T_ff·K (plan_tk)`, `U_ff=Q_ff·S (plan_ks)`,
  fused final `Knew=f32(3Q_ff−2(U_ff·K)) (plan_tk)` — V never stored.
- Only TWO new persistent buffers: `ff_t_lo` on M_TKS, `ff_q_lo` on
  M_K — the hi parts reuse `t_ks.values`/`q.values`.
- Cheaper accumulate allowed: TwoProd + TwoSum per term into
  (hi,lo_running-error) — renormalize only at output; optimize after
  correctness (target ≈3–8× a plain f32 product, ≈8 f32-iter-equiv per
  polish — NOT 50).
- Integration: f32 TC2/TRS4 to a switch threshold (test 1e-2…1e-5),
  then ONE FF-McWeeny step; maybe a second. Stop f32 floor-dancing.
- Acceptance gate: R10 f32 input R_I~1e-4..1e-5 → after one FF-McW
  the CPU-f64 `sparse_ri_f64` must read ≲1e-7 (ideally ~1e-8). Then
  benchmark polish wall-time vs f32 iter cost.

**Production purification policy — CONSOLIDATED (2026-09-15; GPT-5.6
review chat L10089–10604 + Devin analysis). IMPLEMENTED + BENCHMARKED
2026-09-17 (report §15.15): Phase A/B in `tc2_purify`, `PolishedFF`
status, `cfg.tc2_hiacc`, specialized ff kernels; measured FF step
27 ms vs f32-ACC16 iter 6.8 ms (≈4×); rk40 polish 1.1e-7 → 2.8e-8
(f64-verified) in 2 steps, +30–40% purify wall.**

Two separate phases — FF is a *terminal* phase, never a branch inside
the TC2 loop (each in-loop FF iteration currently wastes one full f32
diagnostic iteration T=KS,Q=KSK,tr,res ≈14.5 ms before the next FF
step):

```text
PHASE A — f32 TC2, hard budget ~30 iters
  exit Converged     : R_I < tol (1e-5 fast / 1e-6 default) && trace ok
  floor anticipation : trace_locked && best_ri < ~1e-3 && stalled
                       (best improved <5% over ~5 checks — the ri<1e-3
                       gate makes a short window safe; mid-descent
                       stalls live at R_I >> 1e-2)
                       → restore k_best → Phase B gate / NumericalFloor
  budget exhausted   : if NOT (trace_locked && best_ri < ~1e-3)
                       → FAIL LOUD / robust path — not an arithmetic-
                       floor problem, do NOT polish garbage.
PHASE B — terminal FF32 McWeeny (accurate mode only)
  restore best valid K. Per step: FF-McW + cheap R_I re-measure
  (Q=KSK already in buffer → one reduce ≈0.2 ms) + k_best snapshot.
  early exit: R_I < target OR step gain < ~2×;  hard cap 5
  (measured: 2.7e-7 → 4e-8 → 2e-8 — 1 step normal, 2 strict; after
  step 2 it is polishing the f32-storage floor ~1.3e-8).
  Then recompute T=K·S once, measure trace once → return PolishedFF.
```

Production-hygiene deltas vs the study code:

- Resolve the purification policy ONCE into a struct
  (mode/tc2_budget/ff_switch/ff_steps/tol) — no `std::env::var` in the
  hot loop.
- New `PurifyStatus::PolishedFF` (≠ NumericalFloor — a successful
  polish is not a failed convergence); post-FF R_I must be MEASURED on
  the returned state (current code returns the stale pre-FF `last_r_i`).
- FF entry threshold is chosen by COST, not aesthetics: calibrate
  R_switch ∈ {1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5} — TC2 to switch +
  exactly ONE FF step → true R_I; take the EARLIEST switch landing
  under target (1 FF step ≈ 3 f32 iters, so never buy the last f32
  decade an FF step delivers anyway). Initial default 1e-3.
- Floor accounting: `R_obs ≈ max(R_arith, R_mask)`. FF sets R_arith→0
  and thereby makes R_mask cleanly measurable. Optional M_K-tail probe
  (one product, once per geometry/mask) predicts the mask floor
  a-priori → `tol_eff = max(tol, ~5·tail)` or skip FF when it cannot
  help — this is the "anticipate" half of the contract.

FF-kernel optimization order — **corrected baseline**: the 44 ms/step
benchmark ran with ACC16 OFF; best f32 iter is ~6–7 ms (ACC16), not
14.5 ms → the honest ratio is ~6×/step, ~3×/product, not 3×/step.
Both baselines must use the best f32 kernel.

1. Split the generic kernel: `f32×f32→ff` with NO `lA_lo` tile and no
   `a_has_lo` branch — the generic kernel statically allocates
   2×21 KiB local = ~42 KiB → 1 WG/SM vs 2 for f32 (occupancy is the
   suspected source of the +50% over f32 on the same plan).
2. hi-local / lo-global A/B for the `ff×f32` kernel: A_lo reuse is
   ~330× per row so L2 should absorb it; frees 21 KiB → 2 WG/SM on the
   three expensive products. Measure, don't theorize.
3. 2-way alternating ff accumulators per m (break the serial TwoSum
   chain — the FF analogue of ACC16; watch register pressure, don't
   go 4-way).
4. **V = T·Q reformulation → 3 products, not 4**: V = KSKSK is
   symmetric and T·Q = KS·KSK = V exactly; plan_tk reusable verbatim.
   REQUIRES streaming B_lo (right operand is f32-only today; dropping
   Q_lo injects ~1e-7 and defeats the purpose). The kernel is
   latency- not bandwidth-bound, so the extra B stream may be nearly
   free. NOTE: this falsifies the "2× hard floor" claim — that floor
   assumed 4 products; 3 products → ~1.5–2× TC2-iter per step.
   **MEASURED 2026-09-17: a WASH on R10** (27.4 vs 27.8 ms/step) —
   the dropped product (Q·S on plan_ks ≈1.7 ms) is the cheap one,
   while V=T·Q pays double-width B streaming on the expensive plan.
   Kept as `RUST_DFTB_TC2_FF_VTQ=1` option; may win at higher
   deg_S/deg_K contrast.
5. FF-TC2 experiment (2 products → ~1.5×/step): the branch-saddle
   limit cycle was f32-noise-driven; with ~1e-13 products plain TC2
   may collapse to the f32-storage floor directly. Numerically risky —
   McWeeny stays the proven fallback.
6. Fused V+combine retry — LAST and smallest win; verify bitwise vs
   host emulation (compiler reassociation already broke it once).

Guideline: deterministic stop + never re-entering f32 matter more than
squeezing the step cost; the endgame runs once per converged SCC.

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

## 15. Sparse-only review and coding-agent work order — supersedes recent completion claims

**Scope:** source review of the current working tree, following the user's sparse-only clarification. Instructions, not implementation. No new benchmark or numerical validation was run for this review. Historical timings/results below are observations from the preceding session, not fresh measurements. Dense `qmqm/*`, shared dense kernels, and the H-bond task belong to the parallel agent: do not edit them for these tickets. All tickets remain open pending implementation, diagnostics, and user acceptance.

**Recommendation:** keep BSR4 and device-resident sparse products. First repair state/diagnostic contracts and establish a trustworthy P-TC2 baseline; stop wasting products at a measured plateau; add a bounded-memory SpGEMM route; then use error-budgeted screening that actually removes multiplication terms. Do not start with more f64, larger radii everywhere, or another speculative purification polynomial. Optimize time to an accepted energy/force, not time to a misleading `Ok`.

### 15.1 Corrections to the preceding report

1. **W shortcut is conditional, not an exact statement about production buffers.** `2S⁻¹HK = 2KHK` for the exact occupied generalized projector. But `sparse_system.rs::new` sets `m_t_zs = m_z`; `compute_p0_impl` / `compute_k0_impl` store `b_zh = project_MZ(ZH)`. The force shortcut removes the KH product, NOT all intermediate truncation. Z, K, and stationarity are approximate. `test_w_zhk_vs_khk_parity` performs dense CPU algebra, not `SparseDWWorkspace` GPU parity; its K is returned through f32 before conversion to f64, so its ~5e-8 difference is not simply f64 rounding. Default-on was premature without genuine sparse-mask force validation. Preserve both routes; request/obtain approval before changing defaults.
2. **The current “TRS4” has no demonstrated restoring invariant.** For `f(x)=a x²+(1−a)x⁴`, `a=3` gives `f(0.9)=1.1178` and `f'(1)=−2`; `a=−1` gives `f(0.1)=−0.0098`. Clamping a to [−1,3] neither preserves [0,1] nor guarantees convergence. Clamping also breaks exact trace reset. Positive `Tr(P²)−Tr(P⁴)` does NOT exclude eigenvalues >1: eigenvalues {1.01,0.5} give a positive denominator (~0.167). Negative eigenvalues can escape this even-power diagnostic too. A next-step complement is not guaranteed to repair the spectrum. The attribution to a published Niklasson algorithm needs an actual matching reference; treat this as an experimental quartic map until then.
3. **The quartic implementation has an intermediate projection:** `Q=project_MP(P²)` then `R=project_MP(Q²)` is not `project_MP(P⁴)`. An exact scalar-polynomial argument does not establish stability for that operation. NS has the analogous `project_MZ(ZS)` intermediate. P-TC2 avoids one such extra product inside a purification iteration, but does not remove all truncation errors in initialization, inverse overlap, recovery, or forces.
4. **Trace-coefficient conditioning gets worse near a projector:** `Tr(Q)−Tr(R)→0`, not better. f64 subtraction of already-rounded f32 traces cannot recover lost matrix information. The source's “well-conditioned near convergence” statement is wrong.
5. **Reported floors are not comparable yet.** P/TRS divides `||Q−P||F` by the norm of INITIAL P0. That norm depends on spectral bounds, system, and padding; it is not a common physical normalization. `P=KS` is generally non-symmetric/non-normal; even an exact projector need not have `||P||F=sqrt(Nocc)`. Report raw norm and an explicitly defined physical normalization; do not silently relabel old numbers.
6. **~61 neighbors is an optimistic compression estimate, not a validated operating point.** The histogram drops independently per row, without symmetric-pair decisions, mandatory support, re-solving SCC, or force/energy tests. Its `b=0.03` is 3% relative ROW NORM, at most 0.09% squared norm mass per row—not “3% dropped mass.” Likewise 1% norm is 0.01% squared mass. Neither implies a 1% force error. The ~140-neighbor suggestion is a candidate to measure, not the honest optimum already established.
7. **“All green / same under TRS” overstates evidence.** The prior default suite passed, but TRS G3.4 failed (~7.3e-4 Ha/Å discrepancy). SCC history and energy rounding are hypotheses, not an established root cause. The 1648-atom histogram took ~48.9 s and commonly spent 80 purification iterations per SCC iteration; that is not the required throughput. A smaller idempotency residual alone does not certify energies, forces, or the occupied ground-state subspace.

### 15.2 S1 — Fix state and reduction contracts before tuning (highest priority)

**Owner/files:** numerical-state agent; `sparse_system.rs`, `sparse_dftb.rs`, existing sparse tests.

- **Confirmed stale-product bug:** in `tc2_purify_p`, Q=P² is formed before the trace guard rescales P. The guard then measures `Q−alpha P` and updates using stale Q. Correct the operation ordering or consistently update/recompute Q for the scaled state. Test the guard branch explicitly with controlled inputs; check the actual post-operation trace rather than returning Nocc by assignment. The K-TC2 path forms Q after its scaling and must remain an independent comparison.
- `purify_hscc_p/trs` returns the P trace/residual after `recover_k_from_p` has projected and symmetrized K; SCC labels that trace `Tr(KS)` but obtains charges from reconstructed KS. Return separately named P diagnostics and actual recovered-K diagnostics. Enforce `sum(q_out)=2Tr(KS)` to measured reduction tolerance; the present 0.5-electron charge check is not an adequate consistency test.
- Make accepted-state validity explicit. `set_coords` and `set_q` currently leave `last.n_scc` usable; failed SCC can leave an intermediate `last` plus subsequently changed input q. `energy/forces` only check `last.n_scc`. Invalidate accepted energy/forces on input changes and failed solves, or retain a complete explicitly identified prior snapshot. Never combine old K/H/W with new coordinates, q, or V.
- Preserve provenance `{geometry, topology, H input, K/P, Z, q_in, q_out, status}`. W reuse must check the matching H/Z/K state. On final acceptance validate physical trace, dummy lanes, idempotency, and stationarity together. Distinguish `Converged`, certified `NumericalFloor`, iteration exhaustion, and failure; exhaustion alone is not proof of a floor. Floors require explicit run-policy permission, not permission for Hessians by default.
- Correct the frozen-topology skin check: `set_coords` currently measures the maximum Cartesian component, whereas the stated skin/2 condition needs the maximum Euclidean atom displacement (or another proven pair-displacement bound). Test a diagonal displacement; do not enlarge radius to conceal missed neighbors. Electronic active-mask validity is a separate condition from H/S geometric skin validity.

**Exit evidence:** guard-on CPU/GPU operation parity; cold/warm same-state trace parity; calls after geometry/charge mutation and after failed SCC rejected or explicitly restored; diagonal-motion skin test. No tolerance relaxation to make these pass.

**Status (implemented, tests green — awaiting user acceptance):**

- `tc2_purify_p` reordered: trace measured → guard may rescale P → `Q=P²` formed from the post-rescale P → residual `||Q−P||` and the branch/update all act on the same state. Post-rescale trace is re-measured (reduction), not assigned Nocc.
- `tc2_purify` (K path) likewise re-forms `T=K·S` and re-measures `Tr(KS)` after a guard rescale.
- `guard_fires` counter on the workspace; tests seed a controlled λ>1 leak (scale converged P/K by 1.0001 → dev_rel=1e-4 grows ×2/iter on the squaring branch) and assert the guard fires and the accepted trace is within `tc2_trace_tol`.
- `recovered_k_diagnostics` computes `R_I(K)=‖KSK−K‖/‖K‖` and `Tr(KS)` on the post-recovery K; `purify_hscc_p/trs` now return the recovered-K diagnostics (the state that feeds charges and the energy), not the P iterate's. Cross-checked vs host-f64 dense recomputation of the downloaded K (`host_k_diagnostics`).
- `set_coords` uses the Euclidean per-atom displacement vs `r_skin/2` and invalidates `last`; `set_q` invalidates `last`; `scc` invalidates `last` before propagating a failure (a failed solve leaves no usable energy/forces).
- `mulliken_checked` additionally enforces `|Σq − 2·Tr(KS)| ≤ 1e-6·n_atom + 1e-5` (worst-case lane-order f32 bound) in place of the 0.5-electron check for consistency — the absolute electron-count check is retained.
- `SparseDWWorkspace::new`/`build_dw_into` test call sites updated for the `(tzs_struct, b_zh, w_zk)` signature — legacy `2KHK` route kept as the comparison in `test_dw_workspace_vs_oneshot`.
- NS residual contract confirmed: device reduction accumulates the SQUARED Frobenius sum; host applies `sqrt` and `/√N_orb`; `test_ns_device_residual_contract` and `test_compute_z_second_geometry` match host-f64 recompute to f32 tolerance.
- Cold `Z` rebuild writes a full structural identity (all blocks incl. off-diagonal) before NS — no stale off-diagonal state.

**New tests:** `sparse_system.rs::{test_p_tc2_guard_and_recovery_contract, test_k_tc2_guard_fires}`, `gate_g3_energy.rs::{test_s1_state_invalidation_contract, test_s1_skin_euclidean_diagonal}`.

**Run record (unfiltered):** `cargo test -p rust_dftb --lib sparse::` → 10/10; `--test gate_g3_energy` → 7/7 (1 ignored diagnostic); `--test gpu_dftb` → 2/2; `--test gate_f_geom_opt` → 1/1; `--test gate_g_hessian` → 1/1; `--test gpu_sparse_bsr4` → 23/23 (1 ignored); `--test sparse_dftb` → 1/1; `gate_e_determinism` remains intentionally red (analytic sparse force of E_el+E_rep is an S3 deliverable, not an S1 regression).

**Not done in S1 (deferred per work order):** `trace_kh0_dev` still reduces the band energy to one f32 scalar — selective-precision item for S6, not a state-contract bug. Masked-residual honesty (expanded-support audit of R_H / R_Z) is S2.

### 15.3 S2 — Establish an error budget and a short, trustworthy purifier

**Use repaired P-TC2 as the comparison baseline**, legacy K-TC2 retained. Keep the current quartic opt-in and uncertified. Do not count “10 TRS iterations vs 20 TC2” as a 2× improvement: TRS uses two SpGEMMs and three separate trace readbacks per iteration; P-TC2 uses one product. Count all products, reductions, copies, and SCC iterations.

- Reuse the existing endgame/stagnation machinery, but do not copy its labels blindly: P/TRS currently detects 10× growth or exhausts the cap, not an ordinary plateau. Evaluate raw residual history, trace history, 2-cycles, and improvement relative to measured f32/truncation error. Stop on sustained endgame stagnation and return the best jointly admissible snapshot. Report when residual is still decreasing at the cap: that is exhaustion, not a measured floor. Re-evaluate all diagnostics on the restored state.
- Avoid an almost-zero quartic trace denominator rather than iterating/clamping through it. Compare its magnitude to the measured error in the trace difference, not the fixed 1e-9 alone. A repaired or replacement method must specify the admissible spectrum/metric, occupation ordering, perturbation bounds, and terminal behavior. Trace normalization by itself does not enforce spectral admissibility or ground-state occupation.
- Separate error axes on identical inputs: dense f64 reference at test scale; dense/full-support f32 arithmetic; the same sparse masks with host f64 product diagnostics; then f32 sparse. Vary R_HS, R_Z, R_K/M_P independently. This distinguishes arithmetic, support, solver, and SCC errors without calling all of them “noise.” The production path stays sparse.
- Include the nonorthogonal metric: exact P=KS satisfies `PᵀS=SP`, is similar to a symmetric projector, and has the correct occupied subspace. General sparse projection need not preserve that relation. Measure its defect on diagnostic fixtures. `R_I`, trace, and stationarity can all vanish for a wrong occupied subset; compare energy/frontier occupation against a reference at test scale.
- Treat masked residuals honestly. Current NS residual sees `project_MZ(ZS)`; current R_H uses already-masked KS and an M_K output. Audit omitted contributions on an expanded diagnostic support or by sparse operator applications, without a dense production matrix. A small masked residual is not an upper bound for the full residual.
- Gershgorin discs are valid for a non-symmetric matrix too. They bound the matrix actually supplied, not the untruncated ZH nor automatically a real [0,1] spectrum after nonsymmetric truncation. Account for omitted row sums/operator perturbations; 10% padding is empirical, not a proof. Wider bounds also cost purification iterations: measure that tradeoff rather than making them arbitrarily huge.
- Warm-start q and certified Z. **Do not simply purify yesterday's P after H changes:** a polynomial in P preserves its invariant subspace and leaves an exact old projector fixed. Any density predictor must incorporate the new H and verify stationarity/occupation. Defer new predictor machinery until the baseline is trustworthy.

**Speed target:** eliminate repeated 80-step floor-chasing before chasing small kernel gains. Choose early-SCC electronic accuracy from measured induced charge error relative to the current SCC residual; tighten to the final observable budget on acceptance. Never hide an inaccurate final solve behind a loose early tolerance.

### 15.4 S3 — Make W and forces earn acceptance on genuinely sparse systems

**Owner/files:** sparse force-validation agent; `sparse_forces.rs`, existing G3/Si65 harness, sparse scripts. Do not edit dense force kernels.

- Compare BOTH GPU W constructions for exactly the same K,H,Z. At manageable size compare to the exact occupied EDM and to products of the same approximate operands with untruncated intermediates. This separates algebraic equivalence from numerical approximation and implementation errors.
- Test full-mask SiH4/Si65, then genuinely incomplete independent K/Z masks, then a representative larger crystal. Use actual analytic dS contractions, not only random symmetric probes. Compare W before symmetrization as well: symmetrization can hide a consistency defect.
- Test own-energy gradients over a displacement-size range with identical frozen masks and symmetric initialization/history. Repeat cold and warm ±h solves; keep all failed G3.4 results visible. Compare finite differences of analytic forces when energy quantization limits energy FD. Demonstrate absolute and relative force error and repeatability, not just matching maximum force magnitude.
- Error decomposition for the shortcut: with `Z=S⁻¹+E_Z` and `B=ZH−E_B`, `BK−KHK = (S⁻¹HK−KHK)+E_ZHK−E_BK`. The first term measures non-projector/nonstationary K effects; the others expose inverse and intermediate-support errors. Estimate each before choosing which mask to enlarge.
- If the sparse state is not sufficiently stationary, ordinary Hellmann–Feynman/Pulay contractions are not automatically gradients of its approximate energy. Meet the stationarity/gradient budget or derive the appropriate variational/response treatment. Swapping W formulas is not a general cure.

**Exit evidence:** actual sparse GPU W/dS parity and energy-gradient plateau. Until then retain the shortcut as an experiment, not a justification for Hessians.

### 15.5 S4 — Remove the local-memory degree ceiling without destroying sparsity

**Owner/files:** sparse kernel agent; `gpu_sparse.rs`, `sparse_bsr4_purification.cl`, `bsr4.rs`. Do not change algebra/masks while benchmarking this ticket.

The planned kernels cache a full left row: 16 f32 values/block = **64 bytes per left block**. Their 64/128/256/512 specializations need 4/8/16/32 KiB for lA alone; the intersection variants additionally cache 4 bytes/block of columns. Default WG=128 comprises eight 16-lane output-block teams. The current fixed-512 kernel reserves the large cache even for short rows. `MAX_LEFT` is an implementation limit, not a physical locality criterion.

1. **Implement an exact global-gather planned route first as the overflow/reference route.** Read A using `A_row[i]+plan_a_idx[t]` instead of caching the entire row. Distribute work by output block or small output-block tiles with unique output ownership. This needs no row-sized local array and works above degree 512. Retain deterministic term ordering and f32 accumulators. It may also win below 512 when hardware cache captures reuse; benchmark it.
2. **Specialize row-cached kernels into degree buckets 64/128/256/512**, dispatch via prebuilt row-ID lists. Compile/cache once. Query actual kernel local/private memory, WG limits, and device limits. An OpenCL per-workgroup local-memory limit is not necessarily the total shared-memory capacity per SM; do not infer occupancy solely from the 48 KiB device value.
3. Bucket/split by **work**, too: sum of plan lengths per row/output tile predicts imbalance better than degree alone. Split exceptionally long rows across output tiles without atomics. Only split a single long dot across workgroups if profiling justifies the extra partial buffer and deterministic reduction. A chunked-row local cache is an alternative if the global route loses; partition plans by chunk so each term is visited once, not rescanned for every tile.
4. Preserve the existing four f32 FMA accumulators; they already break much of the dependency chain. Benchmark WG=64/128/256 and 16-lane versus 32-lane cooperative output blocks. A larger WG or `float4` spelling is not automatically faster. Track spills and actual bytes/term, not theoretical arithmetic alone.
5. Audit ALL degree-dependent kernels. `bsr4_build_Hscc` also silently returns when a row exceeds MAX_LEFT; `build_hscc_dev` does not make the SpGEMM left-degree check. Make assembly row-length independent or explicitly reject before launch. No skipped row may leave stale/zero output advertised as a valid H.

**Exit evidence:** planned products versus same-operands reference at degree boundaries (including 513+), odd/tail output counts, non-symmetric P, and poisoned initial outputs. Report time at equal error and identical plan terms, not with a changed radius.

### 15.6 S5 — Error-budgeted screening: remove contributions, not just store zeros

**Do not implement “take the largest 64 blocks.”** Minimize work subject to an observable error budget. The budget decides degree; 64 is a target, not a correctness condition.

- Separate three objects: a geometric candidate/support envelope; a frozen active storage mask; and a multiplication contribution schedule. Zeroing small blocks in the current CSR leaves row loads, plan traffic, and FMAs intact. Count retained plan terms as well as stored nnz. The existing `bsr4_drop_small` is not sufficient for speed by itself.
- First validate screening on a compact test-scale wide-mask state. Reuse its already-computed K/P/Z rather than repeatedly running the ~49-second diagnostic. Histograms are exploratory; the accepted screened run must actually be re-solved and checked. A full wide-mask SCC prepass is acceptable only if amortized over enough subsequent steps/displacements; include its cost in end-to-end timing.
- K,Z and P need separate criteria; K's histogram does not determine P²/ZH/ZS product errors. For symmetric K/Z, drop an off-diagonal pair only if BOTH row budgets permit; otherwise retain both. Preserve diagonal/physical/dummy contracts. Symmetric union of independently retained rows is a safe starting point for the norm budget but retains more than the reported ~61 neighbors. P is numerically non-symmetric: never assume a symmetric-right product for it.
- For a stored-row Frobenius budget use `sum_dropped ||A_ij||F² <= epsilon_i²`. This bounds the matrix perturbation only. For multiplication, bound omitted contributions using `b_ij = sum_skipped_k ||A_ik||F ||B_kj||F`, so `||Delta C_ij||F <= b_ij` before roundoff. For a row use `sqrt(sum_j b_ij²) <= epsilon_C,i` (or a documented more conservative bound). If dropping an intermediate T before TB, propagate its effect, e.g. `||Delta T B||F <= ||Delta T||F ||B||2`; a small T norm alone is insufficient.
- Such bounds are conservative and are NOT direct force/energy bounds. Validate energy per atom AND force max/RMS, charge, trace, stationarity, iteration count, and FD smoothness across budget values. Count screening overhead; computing norms/sorting a huge plan every iteration can erase the gain.
- Freeze active masks and contribution schedules over each Hessian/finite-difference family, certified for a declared geometry/state neighborhood. Include mandatory/union support from representative perturbations, then check budget drift. If the certification fails, fail/reconfigure explicitly outside the hot loop and regenerate affected results; do not independently change screening at +h and −h.
- Approximate masks are part of the numerical method: fewer blocks can increase purification/SCC iterations or change the accepted solution. Select on complete accepted solve time, not nnz alone.

### 15.7 S6 — Cheap structural wins and selective precision

These can follow the baseline contract fixes and be measured independently of screening:

- **Pack diagnostics into one readback per decision point.** TRS currently performs three `trace_ks_f64` readbacks plus a residual read. Write per-atom trace components and residual partials into one persistent packet; host f64 finishes reductions and branch decisions. Near a small trace denominator, accumulate diagonal differences in f64 from appropriately exposed components rather than subtracting separately rounded atom totals. This reduces avoidable cancellation, not the underlying f32 product error. Do not move decisions blindly back to a single f32 scalar.
- **Fix energy reduction precision first, not all matrix arithmetic.** `trace_kh0_dev` returns one f32 extensive band-energy sum; casting it to f64 in `store_energy` cannot recover precision. At total energies of order 1800 Ha a single f32 number has spacing of order 1e-4 Ha. Return workgroup/atom partials and accumulate in host f64, or use a measured compensated scheme. Validate finite differences on identical K before blaming SCC. Use f64 for small DIIS/decision systems, not broad GPU SpGEMM on a consumer GPU.
- **Do not compute a full KS just to obtain charges.** After recovering K, diagonal `(KS)_ii` can be contracted directly over matched K/S blocks in O(nnz intersection), keeping charges consistent with K. Retain full KS only when the chosen purifier/residual actually consumes it. Do not switch to diagonal P without certifying the recovery mismatch.
- **Remove unused work while preserving A/B routes.** `build_dw_into` builds D=2K, but production `forces()` downloads K and forms 2K in the CPU contraction. Avoid this D materialization on that path. Fuse factor 2 into W output if profitable. Do not materialize unused legacy scratch/plans for every configured solver; choose persistent workspaces at initialization, never allocate them mid-SCC. Share identical structural plans (e.g. ZS and ZH use the same mask triple), not just their builder code.
- **Compress plans only after recording plan memory/traffic.** Current pair indices are two u32 arrays (8 bytes/term), plus output pointers. Left row-local indices can fit u16 under a checked degree limit; packing 9-bit left + 23-bit global-right into one u32 is possible only for degree <=512 and right-block count <2^23. Keep a general wider format and explicit capacity checks; never silently truncate indices when widening masks. Measure decode cost and reuse before committing to a format.
- **Remove host allocation/readback from the repeated path.** `mulliken_charges` makes two Vec copies; `mulliken_checked` another; `store_energy` clones q/v every SCC iteration. Force evaluation rebuilds SystemContext, allocates scratch, and downloads all K plus W. Use persistent staging and in-place outputs. First gather only force-required K blocks; then implement sparse pair force + per-atom gather on GPU, preserving a CPU parity route and avoiding edits to the dense agent's kernels. R_H needs reduced norm partials, not an entire `a_ht` download.
- Keep cached gamma as an explicit O(N_atom²) component. Profile it, then move the tiled matvec/force accumulation to a sparse-owned GPU path if material. Do not claim linear total scaling or cut the Coulomb tail to meet a neighbor target. FMM/tree acceleration is a separate later project, not the first fix for repeated sparse matrix products.

### 15.8 S7 — Throughput measurement, batch layout, and stop/go gates

**Reuse `SparsePerfStats` and the existing Rhai harness; do not invent another benchmark executable.** Instrument production entry points, not only one-shot helper tests. Count real allocations, launches, syncs, matrix bytes, plan bytes/terms, accepted/rejected states, and total SpGEMMs. Avoid double-counting inclusive `t_scc` alongside its children in the existing audit's “misc” subtraction.

- Measure initialization, one cold SCC, one small geometry update plus warm SCC, force-only evaluation, and a short repeated geometry sequence separately. Small fixtures first; reserve 1648 atoms for periodic acceptance. Full unfiltered foreground output; no piped grep/tail, and preserve the test process's actual exit status. Earlier piped commands could report exit 0 after a test failure.
- Record queried device/kernel resources and release build/configuration. Kernel event timing and wall timing answer different questions; report both. Run the matrix {K-TC2, repaired P-TC2, experimental quartic} against {intersection, planned row-cache, planned global, buckets} only in staged one-factor experiments, not one huge combinatorial sweep.
- Sparse batching is over independent replicas AND sparse rows/output tiles, not one workgroup for the entire nanocrystal. Share topology/plans for identical frozen structures, keep values and convergence state per replica, and test singleton versus mixed-convergence batch parity. Reuse one runtime; do not instantiate an OpenCL program/queue per replica. A 1648-atom single system already supplies many row workgroups: batching helps amortization/throughput but is not automatically its primary bottleneck.
- Preflight memory before selecting batch size: matrix storage is 64 bytes per BSR4 block per live buffer per replica; current code allocates both K and P families and multiple duplicate plans. Plan terms may dominate matrix storage. Avoid launching a batch that fits individual matrices but not the complete workspace.
- For Hessians use shared frozen topology and a reproducible base electronic state; schedule +h/−h pairs consistently. History-dependent masks, DIIS histories, or plateau snapshot choices need an explicit repeatability check, not post-hoc Hessian symmetrization.

**Dispatch order:** S1 contracts → S2 baseline/floor diagnosis and S3 W/force certification → S4 degree-independent exact kernel → S5 bounded screening. S6 low-risk allocation/reduction improvements may be interleaved after S1, one measured change at a time. S7 instrumentation starts with S1; batch execution follows reliable singleton states. No new solver polynomial, blanket f64, fixed-64 truncation, or Hessian production until these acceptance gates are satisfied.

**What the next agent should return:** the failing diagnostic first, its identified cause, one surgical change, unfiltered verification, and before/after accepted-solve time at equal accuracy. Keep experimental implementations for comparison. A source review is not permission to mark scientific results validated.

### 15.9 Third review work order — sparsity contract, kernel sizing, selective precision (GPT-5.6, 2026-09-13, chat line 4806+)

**User directive restated:** the whole point of the sparse solver is *as few neighbors as possible* — that is the dominant term for speed. Matrices must not densify under multiplication: every product is truncated onto a fixed target mask **by design, not as an afterthought**. This section refines S2/S4/S5/S6 into explicit checkable items; it does not replace the S1→S7 dispatch order (S1 done pending USER acceptance; the items below start after it).

**Production algebra rule (fixed):**

> Fixed input masks + fixed output mask + direct projected multiplication `C = P_M(AB)`. No "wide halo → measure → drop → compress → rebuild plan" in production. A halo on the *final* result is pure waste (extra C_ij's do not improve in-mask C_ij's); a halo is only justified for an *intermediate* feeding another product — and only where experiments prove a narrow intermediate is inaccurate (primarily `T=ZS` inside Newton–Schulz; see SC7). Threshold/halo calculations are **calibration/diagnostic tools to choose the fixed masks**, not the routine representation.

#### SC — Sparsity contract (hard requirements; extends S4/S5)

- [*] **SC1. Explicit degree budgets per mask at engine construction.** **DONE 2026-09-13:** `deg_hs/deg_k/deg_z` measured at `SparseDftb::with_config` and printed (`deg_*=measured/budget` in the init line); `cfg.max_deg_{hs,k,z}` (`Option<u32>`, env `RUST_DFTB_MAX_DEG_{HS,K,Z}`, defaults 512/128/256) — exceeding a budget is a hard `Err` naming mask/degree/budget. Derived masks (M_P, M_TKS, M_TZS, M_HT) are clones of the three base masks, so their degrees are covered.
- [*] **SC2. Compile `MAX_LEFT_BLOCKS` from the measured mask budget, not the default 512.** **DONE 2026-09-13:** `SparseDftb` compiles `max_left_blocks = ceil16(max(deg_hs, deg_k, deg_z))` — SiH4 now gets 16 (~1 KiB/WG vs 32 KiB before). Frozen topology ⇒ degree cannot grow. Direct `SparseBsr4Gpu::new` callers (tests) still default 512 — intentional, no mask context there. No degree buckets (YAGNI).
- [*] **SC3. Plan kernel mandatory in production — remove the silent intersection fallback.** **DONE 2026-09-13:** plan-build/upload failure is a hard `Err` in `SparseSystemWorkspace`, `SparseDWWorkspace` (sparse_forces), and `Tc2Workspace`. `None` → intersection remains reachable ONLY via the explicit `RUST_DFTB_SPARSE_PLANS=0` diagnostic toggle (kept for tests/A-B — not a fallback). New `plan_ht`: the R_H stationarity product `H_scc·(K·S)` now uses the generic planned gather kernel too — no production product runs the intersection kernel by default.
- [*] **SC4. Add the missing degree guard to `build_hscc_dev`.** **DONE 2026-09-13:** host-side `max_deg > max_left_blocks → Err` before enqueue — the kernel's `if(nb > MAX_LEFT_BLOCKS) return;` stale-row bail can no longer fire silently.
- [~] **SC5. Physical defaults must not be permissive.** **PARTIAL 2026-09-13:** running with ALL radii defaulted (r_trunc/r_k/r_z = None → full SK radius + skin) now prints a loud `WARNING ... permissive wide-mask regime` at init — the degree budgets remain the hard contract. Radii stay script-explicit rather than code-defaulted: the MS3 sweep showed the right value is system-dependent (deg 212 ≈ r_k 10 Å is the usable floor on Si spheres; see MS3), so a single hard-coded default would be wrong either way. Fully enforcing "no sparsity budget → no start" (hard-Err instead of warning) deferred — the warning + budget check is the current contract.
- [ ] **SC6. Shrink H/S aggressively AND shrink the 1 Å skin.** Si–Si coupling is ~1e-4 Ha by ~7 Å; the 10.6 Å table end is not physics. Frozen-topology Hessians use ±0.02 Å displacements → required skin is far below 1 Å (skin counts toward structural degree and plan work even where values are zero). Geometry optimization may rebuild masks at explicit checkpoints instead of carrying a large skin forever.
- [*] **SC7. Fixed halo only where algebra requires an intermediate.** **DONE 2026-09-13:** `SparseSystemWorkspace::new` takes an explicit `tzs_mask`; `SparseDftbConfig.r_zs_halo_ang` (default 0 → `M_TZS = M_Z`, legacy behaviour) builds `M_TZS = geometric(r_z + halo)`. It is degree-measured and budget-checked under the M_Z ceiling (`deg_tzs` in the init line) and feeds `MAX_LEFT_BLOCKS` sizing. `plan_zs`/`plan_zh` write on `M_TZS`; `plan_tz`/`plan_bz` read `M_TZS` as the left operand and truncate results to `M_Z`/`M_K` — result projection is preserved, no unrestricted product support anywhere.
- [ ] **SC8. Codified warning — do NOT assume K is more local than H.** DM decay length is set by the electronic gap and basis and can exceed the Hamiltonian's range (SP2 literature reports DMs several times denser than H). K/P mask sizes are a *measured* approximation: smallest degree meeting the energy/force/frequency error target, not an assumption about decay.
- [x] **SC8b. Cost coupling — DM support is NOT set by H/S, but product cost IS.** (USER 2026-09-14, confirmed by §15.12-A measurements.) The final density's required support follows its own decay length (R10: deg~275 for ~1 mHa, ~330/complete for 23 μHa) — far beyond deg_hs~60. Every iterative product's work scales with the *operand* degrees: `K·S` gathers deg_S blocks per output, `K·S·K` another deg_K. **REFUTED sub-claim (§15.12-2′):** the intermediate-halo lever (b) is dead — measured intermediate truncation loss is 0.075%; the bias lives in K's own truncation destabilizing the masked map, which no intermediate support fixes. H/S narrow (a) remains valid for product cost.

#### PR — Selective precision (extends S6; order matters)

- [~] **PR1. First precision experiment: 8-way f32 partial accumulators** in the planned SpGEMM — two independent accumulators per orbital channel (alternating terms, pairwise combine). ~No extra FLOPs, halves the serial summation chain. **IMPLEMENTED 2026-09-13 as a compile-time variant:** `-DACC8` on both plan kernels (two quads alternating over plan terms + odd-term tail), selected by `SparseBsr4Config.spgemm_acc8` (env `RUST_DFTB_SPGEMM_ACC8=1`, default OFF). Default stays 4-acc: the reassociation shifts K by ~ulp noise, which the tight SiH4 FD-vs-analytic force gate (G3.4, |ΔF|~1e-4) detects and fails on — physics is unchanged, the gate is at the noise edge. Still needed: a wall-clock A/B benchmark at deg~200-400 before any promotion.
- [ ] **PR2. Compensated f32 (Kahan) as an endgame mode, not a permanent tax.** Optional compensated planned-SpGEMM used only after `R_I ≲ 1e-3` / at plateau, and for the once-per-call products `K = P·Z` and `W = (ZH)·K`. Benchmark 8-accumulator vs 4-Kahan on the RTX 3090 — whichever wins is data, not assumption.
- [*] **PR3. Reduction tails in host f64.** **DONE 2026-09-13:** `reduce_partials_f64` shrinks the device tree to ≤`REDUCE_TAIL`(128) partials, downloads, sums in f64. Applied to `idempotency_to_f64`, `identity_residual_to_f64` (NS residual), `frob_sq_to_f64` (‖K‖/‖P‖ norms), `trace_hk_to_f64` — every decision scalar now has an f64 tail. The old all-GPU-to-scalar `*_to_dev` paths remain for tests.
- [*] **PR4. Band energy `2·Tr(K·H0)`: f64 tail + stop computing it every SCC iteration.** **DONE 2026-09-13:** `trace_kh0_dev` returns f64 (host-f64 tail). `store_energy` runs only on the converged SCC iteration or in verbose mode — the reduction + host sync are gone from non-final iterations.
- [ ] **PR5. No GPU f64 in matrix products — ever.** f64 only for scalar decisions and reduction tails (already the case: traces/branch logic/DIIS solve/Hessian eigensolve).

#### AL — Purifier/algebra direction (extends S2)

- [~] **AL1. Promote P-TC2 to the main performance candidate.** **PARTIAL 2026-09-13:** selection is now explicit — `SparseDftb::set_purifier("k"|"p"|"trs")` + rhai `sparse_purifier(name, mode)`. **Fix found by measurement:** `K=P·Z` recovery inherited the ZS−I residual → Tr(KS) drifted ~1.3e-4 relative at 864 atoms and tripped the SCC Tr-gate → hard fail. `recover_k_from_p` now restores the charge-conservation invariant (build T=K·S once, rescale K and T by Nocc/Tr — same pattern as the TC2 trace guard) and leaves t_ks fresh. Measured on R14: P-TC2 ~1.5–3× faster wall per SCC and reaches SCC convergence everywhere K-TC2 does (plus r_k=8 where K-TC2 limit-cycles), but converges to its own fixed point (−4.1 mHa vs K-TC2 at deg 386 — no dense reference yet to say which is right). **Not promoted to default** — needs the wide/dense reference comparison of AL1's validation requirement.
- [ ] **AL2. Keep `W = 2(ZH)K` as the production force path.** Do not return to `KH → KHK` intermediates except in parity tests. Principle: reformulate the algebra to produce the narrow matrix directly rather than truncate a wide intermediate.
- [ ] **AL3. TRS4 stays opt-in.** It costs two SpGEMMs/iter (P² and P⁴) — only worthwhile if its stability reduces *total* work (products + iterations + SCC steps), not merely iteration count.

#### MS — Mask design & decisive measurement (extends S5)

- [ ] **MS1. Wide-reference calibration runs (occasional, not production).** For each product/mask report `ε_M = ‖C_wide − P_M C_wide‖_F / ‖C_wide‖_F`; choose the fixed mask that meets the error target.
- [ ] **MS2. Magnitude-aware mask design.** From one wider P/K calculation, score blocks `s_ij = ‖P_ij‖_F` and keep the strongest *symmetric* neighbor graph under the degree cap — can beat a spherical cutoff near surfaces/defects while keeping a fully static GPU layout. **Correction to the earlier coding-agent claim:** dropped *numerical* mass is NOT free at plan build — structure knows contribution *counts* only. Error estimation needs either a wider reference product or runtime bounds `‖A_ik B_kj‖ ≤ ‖A_ik‖_F·‖B_kj‖_F` (submatrix-product screening, Rubensson et al.).
- [*] **MS3. The decisive matrix.** **MEASURED 2026-09-13** on `si_sphere_R14` (864 atoms — interior degrees saturate like R18; cube_si65 is too small, deg saturates at n_atom=65 for every radius tried). Script `rust_dftb/scripts/sparse_degree_sweep_big.rhai`, r_trunc=8 Å, tc2_tol=1e-5, ns_tol=1e-4, budget 512:

| r_k=r_z (Å) | deg_k | mode | E (Ha) | ΔE vs ref (mHa, total / per atom) | SCC | t_scc (ms) |
|---|---|---|---|---|---|---|
| 12 | 386 | k | −882.51862 | 0 | 13 it | 7073 |
| 12 | 386 | p | −882.52271 | −4.1 / −0.005 | 12 it | 8129 |
| 10 | 212 | k | −882.38115 | +137.5 / +0.16 | 14 it | 3088 |
| 10 | 212 | p | −882.41197 | +106.7 / +0.12 | 12 it | 1890 |
| 8  | 108 | k | **did not converge** — limit-cycle rms 4.5e-4↔2.2e-3 (R_I floor ~2e-3 charge noise > DIIS floor) | — | 60 it nc | 3890 |
| 8  | 108 | p | −881.83529 | +683 / +0.79 | 12 it | 579 |
| 6  | 52  | k | −880.66655 | +1852 / +2.14 | 12 it | 218 |
| 6  | 52  | p | −880.43556 | +2083 / +2.41 | 12 it | 181 |

**Answers the question directly: ~deg 200 is the floor for Si, not 64–128.** deg 212 costs 0.16 mHa/atom (usable for energies, marginal for the 1e-5 Ha/atom phonon criterion); deg ≤108 is unusable — either non-convergent or converged to a state wrong by ~1–2 mHa/atom. The speedup is real (~30× from deg 386→52) but the K-decay length on Si — not the kernel — bounds the sparsity. Convergence is not a proxy for accuracy: r_k=6 converges "fine" to a 1.85 Ha error. For vibrational work the production contract should keep deg_k ≈ 256+ budget with r_k ≈ 10–12 Å on Si spheres.
- [ ] **MS4. Sparsity is also the precision fix.** Sequential f32 accumulation error scales ~`n·u` — cutting 300→64 contributions lowers the residual floor *before* any compensated arithmetic. Reduce degree first, then decide whether Kahan is still needed.

**Suggested order within this section:** SC4 (one-line hole) → SC3 (fail-loud plans) → SC1/SC2 (degree budgets + kernel sizing) → SC5/SC6 (explicit tighter defaults) → MS1–MS3 (measure what the masks cost) → AL1 (P-TC2 validation/default) → PR1–PR4 (precision experiments, each benchmarked). SC7/SC8 and MS2/MS4 are standing constraints on all of the above.

### 15.10 Fourth review work order — locality rescue plan (GPT-5.6, 2026-09-13, chat line 5400+)

**Context.** MS3 measured: blind radial truncation of the matsci density kernel cannot go below deg ~200 on Si. That mixes three things — physical locality of the occupied subspace, the nonorthogonal AO basis, and the crude "everything inside a sphere" mask. This section attacks all three, in increasing implementation cost. **User priority: L1 (pbc basis) FIRST — by far the easiest — then the rest.**

**Basis-set lever — the parameter set IS the locality knob.**

- [*] **L1. Rerun the degree sweep with `pbc-0-3` instead of matsci.** **MEASURED 2026-09-13 — this is the win.** pbc Si: confinement r₀=3.3 a₀, all tables end at 10.4 a₀ ≈ 5.5 Å. On R14 with r_trunc=5.3 Å: **full pbc mask (r_k=0 → r_full=6.5 Å incl. skin, deg_k=deg_z=95, deg_hs=54) converges cleanly — 19 SCC iters, 537 ms, E=−845.11517 Ha** vs matsci's deg-386/7073 ms reference → **13× faster at 4× fewer neighbors, deg 95 already under the 128 budget.** Truncating the pbc mask below its own table end FAILS: r_k ∈ {3.0–6.2} Å (deg 6–52) all diverge — K-TC2 blows up (`R_I=inf` at iter ~50–60, the truncated Z breaks the Gershgorin bounds → K0 outside the polynomial's contractive range) and P-TC2 plateaus at r_I~1e-2. Also **P-TC2 fails on pbc even at the full mask** (r_I floor ~1.5e-2 — needs investigation; K-TC2 is the working purifier there). Interpretation: the density matrix in the confined pbc basis is genuinely short-ranged — the basis solves most of the locality problem. The "nothing below ~5.5 Å" is what *blind radial truncation* showed — NOT a proven floor: L3/L4/L5 (P-locality, top-k masks, LNV) remain open ways to push degree well below 95. **USER DIRECTIVE: strongly prefer pbc-0-3 over matsci-0-3 for the sparse solver — fewest neighbors is the top performance priority; matsci is reference/parity only.** **Open:** (a) pbc physics unvalidated — needs a Fortran DFTB+ E/F parity run on the same geometry; (b) P-TC2-on-pbc failure (works on matsci, floors at r_I~1.5e-2 on pbc even at full mask).
- [ ] **L2. If pbc shows confinement → locality, consider a sparsity-aware Si/H refit later** (confinement radius is a legitimate parameterization DOF; DSKO-style). NOT now. `siband` is rejected: needs d-orbitals (9-orbital blocks = worse for BSR4) and has no usable repulsive potential.

**Representation lever — maybe P is already local and only K=P·Z is not.**

- [*] **L3. Measure P-locality directly, before K recovery.** `E_band = 2Tr(P·ZH)` — A=ZH is already built for P0, so `2Tr(P·A)` needs no globally-truncated K. Likewise q_A = 2Tr P_AA. Take the deg-386 reference, then purify P at deg 52/108/212 and compare E_P. If P is accurate at deg ~108 but K=PZ is not, the *hot SCC loop* can run at deg 64–128 and only short-range K/W blocks get built afterward for forces — a major win. The orthogonalization tail of S⁻¹ may make K intrinsically longer-ranged than P. **MEASURED 2026-09-13 — P is NOT more local than K.** `band_energy_from_p` (restrict b_zh→M_P + masked trace) + `sparse_eval_p` rhai: on matsci R14, E_P drift vs deg-356 reference: deg 212 → +144 mHa, deg 108 → +205 mHa, deg 52 → +4.65 Ha — same magnitude as the recovered-K path (or worse). The K=PZ recovery is NOT the limiter; the projector itself carries the ~10–12 Å tail in this basis.
- [*] **L4. Magnitude-aware (top-k) masks — MS2 made concrete.** From the wide reference, compute b_ij = ‖P_ij‖_F per block; build masks keeping the top-32/48/64/96/128 blocks per row (symmetrized). Rerun the same calculation. Case A (top-96 works): the sphere was a terrible sparsifier — production masks can be predictor-based (‖P_ij‖, ‖H_ij‖, bond connectivity), frozen once for Hessians. Case B (top-128 fails): the projector genuinely isn't sparse in the AO basis → stop optimizing kernels, change representation (L6). **This experiment is more informative than another radius sweep.** **MEASURED 2026-09-13** (`tests/sparse_topk.rs`, ignored GPU test; `build_topk_mask` in bsr4.rs; `mask_kz` config injection): top-k on K norms — top32/deg60: non-converged; top48/deg94: +384 mHa; top64/deg114: +206 mHa; top96/deg174: +83 mHa; top128/deg236: rms 1.3e-5 ≈ tol, +36 mHa. **Verdict: mostly Case B** — magnitude masks DO converge where same-degree geometric masks limit-cycle (top64/114 vs geo deg-108) and give ~2× better error-per-neighbor (deg 174→83 mHa vs geo 212→137 mHa), but even optimal selection at deg ~100–230 leaves 80–400 mHa — nowhere near production accuracy. On matsci the projector truly isn't compressible to 64–128 neighbors → representation change (L1 pbc: deg 95 works) is the fix, not mask cleverness.

**Algorithmic lever — if truncation still fails.**

- [ ] **L5. Variational localized-DM refinement (LNV).** Instead of truncating the exact projector, minimize the energy over the *allowed* sparse elements of an auxiliary L on the fixed mask: K(L) = 3LSL − 2LSLSL (nonorthogonal LNV needs S, never S⁻¹ — the variational DM stays localized). Concretely: P-TC2 → fast initial localized state, then 5–15 variational refinement steps on the same deg-64/96/128 mask — in-mask elements optimally compensate the missing tail. Classic LNV was tested on 512-atom Si with a finite localization range. This has a real chance of making 6–8 Å useful where hard truncation cannot.
- [ ] **L6. If even LNV-on-mask fails: localized bond/support orbitals** (CONQUEST/ONETEP philosophy — a few bond shells, not a 12 Å AO sphere). Large rewrite; last resort. NOTE: on-site s,p→sp³ rotation does NOT help BSR4 (unitary within the 4-orbital block preserves ‖P_ij‖_F).
- [ ] **L7. Smooth-step damping of P tails** (w = 1−(6x⁵−15x⁴+10x³), C² at both ends) — useful for Hessian smoothness and as the LNV localization window, but at the same outer cutoff it removes MORE amplitude than a hard cut; **it will not recover the 8 Å error by itself**. Only interesting combined with L5.
- [ ] **L8. Finite electronic temperature** shortens DM decay (gap × kT controls the exponent). Changes the formal problem (Mermin free energy, entropy in forces) — ranked below L1/L3/L4.

**Do NOT do now:** Kahan/more GPU micro-optimization — MS3 shows truncation error dominates f32 arithmetic by orders of magnitude.

**Order:** L1 (pbc sweep — trivial, do first per user) → L3 (P-vs-K locality diagnostic on matsci) → L4 (top-k masks — the decisive compressibility test) → L5 (LNV refinement) → L6 (orbital representation, last resort). L7 rides inside L5. L8 only if small-gap surface states prove to be the locality killer.

### 15.10b — GPT-5.6 reframing + open parity problem (chat line 5855+, 2026-09-13 evening)

**PARITY ROOT-CAUSED (2026-09-14):** the 2 Ha gap is the **purification floor**, not an engine bug. Decomposition: Tr(K·H0) is the only off term (+1.975 Ha; E_scc/E_rep/Δq all match DFTB+ to ~mHa). Mask sweep: dE ∝ R_I² — deg95→+1972mHa, deg168→+634mHa, deg386→+151mHa. DFTB+ non-SCC E_H0=−848.727 vs Rust single-purification ~3 Ha off → purifier on H0 alone, no SCC-state dependence. **The pbc K-DM is NOT more local than matsci** — the deg-95 "13×" run was just a dirty purifier. pbc's real win: deg_hs=56 vs 356 (cheap H_scc/traces), NOT a narrower loop matrix. Dense-CPU cross-check still running.

**BUG FIXED — `cut_bohr` sized "full" masks off unrelated SKF tables** (pbc-0-3's F-O 620pt table set r_full=7.59 Å for Si+H whose true end is 5.5 Å). Now filtered to system species → pbc r_full=6.54 Å (deg~56). This resolved the "deg 52→95 jump" (T3) — it was never a physical shell. **Side effect: the default pbc mask (deg~56) no longer converges SCC** — the old deg-95 run was accidentally wide. Explicit r_k/r_z now required for pbc.

**TC2 blowup is deterministic, not random** — identical divergence (iter 62, R_I=0.78) whenever tc2_tol is tighter than the plateau; pbc purification is marginal.

**GPT-5.6 reframing — "95 neighbors is the floor" is NOT proven.** The sweep chopped K, Z and all intermediates by the SAME radius; that proves only that this combined truncated algebra fails. The 6.5 Å "full" mask includes a 1 Å structural skin — Z legitimately extends past the 5.5 Å H/S range. Objective ≠ "every matrix ≤64" — it is **"the matrix multiplied 10–30× per SCC iter (P/K) has degree ~64; Z built once/geometry may stay wide".**

**Tomorrow's plan (priority order):**
- [x] **T0a. pbc parity gap → purification floor.** dE ∝ R_I² across mask widths; only Tr(K·H0) off; charges match DFTB+. Still pending: dense-CPU cross-check to confirm E→−847.09 at wide mask.
- [x] **T0b. "Nondeterministic" TC2 blowup = deterministic** — tighter tc2_tol pushes past the R_I floor into runaway. pbc TC2 is marginal; needs the trace guard / plateau logic to be robust.
- [x] **T1. Wide Z does NOT rescue narrow K** (r_z=9/deg-168 fixed, r_k swept): deg_k 21–52 diverge; 56 dirty +2975 mHa; up through deg-386 +151 mHa. K itself needs the width.
- [x] **T2. P-TC2 works on pbc at deg-168** (14 iters, E_P=−899.21 vs DFTB+ band −900.01; E_K identical to K-TC2 −846.455). Cross-validates both representations.
- [x] **T3. deg 52→95 jump SOLVED — cut_bohr bug** (unrelated F-O table inflated r_full to 7.59 Å; the 95-mask was really 7.6 Å). No physical shell. Per-species stats at honest 6.5 Å: avg 42, max 55 (Si rows avg 45.5/max 55, H rows avg 34.4/max 41).
- [x] **T4. Top-k oracle on pbc done** — ~4× better than radial per degree but heavy-tailed: deg-113→+345 mHa, deg-235→+34 mHa vs deg-386 ref. Not deg-64-accurate alone.
- [x] **f32 floor measured: ~52 μHa/atom, linear in N** (R10 complete-mask deg-330: +17.3 mHa; R14 deg-630: +46 mHa). The purifier's f32 SpGEMM roundoff — NOT mask, NOT basis — sets the accuracy ceiling at ~50–150 mHa for 300–3000 atoms.

**REVISED priority (post root-cause):** the mask story is understood — radial fails, top-k helps ~4×, but everything saturates at the f32 floor. **Cheap levers all falsified** (R10/pbc complete mask): ACC8 accumulation → R_I identical; guard-off → identical; tight bounds (E_DUMMY 2.0→0.8) → *worse*; host-f64 McWeeny polish → diverges (the fixed point is subspace-defective, not noisy). The ~5e-4 R_I floor is intrinsic f32 TC2 iteration dynamics (~250× above naive f32 storage noise).
- [ ] **A1. Rethink: compensated *summation* likely insufficient** — the floor is in the iterate dynamics, not sum order. Candidate real fixes: (a) f64 or f32-pair K storage inside the TC2 loop only (mask stays, values get extra mantissa); (b) a different local iteration with a stable f32 fixed point (steepest-descent/LNV-style update that *minimizes* rather than iterates to a polynomial fixed point).
- [ ] **A2. LNV variational refinement on the fixed mask** — the strongest combined lever: minimizes E on the mask so in-mask elements compensate the tail AND replaces the unstable polynomial fixed point with a descent method.
- [ ] **A3. Decision point after A1/A2:** if f32 floor is beaten, re-run the deg sweep — the true DM locality curve may sit below the current measurements.
- [ ] **A4. Fix dense-CPU reference path for pbc** — `run_dftb_scc` mixer diverges on R10/pbc and `run_dftb_nonscc` counts 1 electron/atom (wrong q0). Needed for cross-checks at scale.
- [ ] **T5. Wendland C² taper on K** (w=(1−x)⁴(1+4x), compactly supported PSD → W∘K stays PSD by Schur) — as a stability/localization aid feeding LNV, not a cure.

### 15.11 — Briefing for the next review LLM (2026-09-14, post-parity root-cause)

**Superseded where corrected by §15.12 and the report's appended Speed–accuracy source review.** Preserve measurements; “intrinsic f32 floor confirmed,” “f64 McWeeny,” and “overhead solved” are not established conclusions.

**Read first:** report `2026-09-14` section (parity root-cause + falsified fixes). All numbers below are measured, not hypothesized.

#### What is already solved (do not re-litigate)

- **Harness overhead:** fully GPU-resident workspace; persistent buffers/kernel handles; 2 SpGEMMs per TC2 iter (from 5); zero matrix host transfers per iter; 1 scalar read per iter; plans precomputed once per geometry (no per-iter intersection); band energy evaluated only at convergence.
- **Local memory / workgroups:** MAX_LEFT_BLOCKS compiled from measured mask degree; BSR4 4×4 blocks gathered to `__local`; plan-driven gather (no atomics, write-once outputs); degree budgets SC1–SC8 hard-fail on overflow.
- **f32 where it works:** device f64 reduction tails for O(N) scalar decisions (Tr(KS), R_I, R_Z, band energy); endgame trace guard with lock conditions (fixed the 1648-atom runaway); plateau-restore best-K.
- **Correctness:** SiH4 parity 1e-7 (matsci) / 34 μHa (pbc); energy decomposition exposes E_band/E_scc/E_rep separately; cut_bohr now filters to species actually present.
- **Diagnostics:** top-k mask oracle (`tests/sparse_topk.rs`), host-f64 K readback + McWeeny (`tests/sparse_f64check.rs`), mask-degree statistics, `RUST_DFTB_EMIN/EMAX` bounds override.

#### Open problems — specific questions

1. **f32 TC2 floor mechanism.** Complete-mask (deg = n_atom, zero truncation) R_I ≈ 5e-4 — ~250× above f32 storage noise. ACC8 (8 accumulators) doesn't move it; disabling the trace guard doesn't move it; f64 McWeeny iterates *diverge* from the converged K (R_I grows 7.5e-4→7e-3) — the fixed point is defective in the *subspace*, not just non-idempotent. What mechanism produces a biased fixed point rather than a noise ball? Branch-alternation limit cycle? Quantization of the polynomial step? Is this a known failure mode of TC2/SP2 in f32?
2. **Best accuracy-per-flop fix.** Options: (a) f32-pair (double-single) storage of K inside the purifier only; (b) f64 accumulation in the SpGEMM inner loop with f32 storage (≈ACC8 didn't help → summation order isn't it); (c) a *descent* method (LNV/minimization) whose fixed point is variational, not polynomial; (d) monotonic purification (no branch flips — e.g., canonical purification or sign-iteration). Which addresses a *subspace* defect, not just idempotency?
3. **Variational masking.** On a truncated mask the purified K is not the best in-mask projector — a variational minimum on the same support could be strictly better in exact arithmetic. Quantify: how much of the 2 Ha at deg-95 is "mask too small" vs "wrong in-mask point"? Does LNV recover the missing-tail energy to first order?
4. **Relative-energy error cancellation.** The product is energy *differences* along geometry scans with a frozen mask. If the purifier defect is smooth in geometry, ΔE error ≪ absolute error. Is there a way to make the defect deterministic-in-geometry (same fixed point bias) so differences cancel? Any literature on systematic vs random purification error?
5. **Mask for production.** Radial is dead; top-k ‖K_ij‖ gives ~4×/degree but still heavy-tailed (deg-113 → +345 mHa). Better proxy for the production mask: ‖P_ij‖ (bond order, AO basis — asymmetric but physical), H/S-derived connectivity, or iteration history? Should the mask be chosen by *energy* sensitivity (∂E/∂mask-entry ∝ |H_ij·P_ji|) rather than |P| or |K| alone?
6. **Dummy-lane spectral design.** BSR4 pads H to 4 lanes/atom with E_DUMMY=2.0, which IS the Gershgorin emax (physical top +0.24). Tight bounds need dummies outside the map, but outside-map lanes get occupied. Worth pinning dummy K-lanes to 0 each iteration + excluding them from bounds — or irrelevant since bounds tightening didn't help?
7. **Performance model.** If accuracy ultimately needs deg~150–250 for the loop matrix, at what N does planned-SpGEMM sparse beat dense diagonalization on a 3090 — and is there a degree/workgroup assignment that keeps 82 CUs saturated at deg~200 (local-mem 49 KiB → ~deg-380 cap with current kernel)? Is the purifier or the eigensolver the long pole at production sizes?

### 15.12 — Speed–accuracy work order (2026-09-14; source review only)

**Authority/status:** supersedes conflicting causal claims and next-step ordering above. Read the appended report review for evidence and answers to §15.11. All tickets below are **open/unverified**; no code or new numerical experiments were produced by this review. Preserve existing uncommitted work; sparse agents must not edit the parallel dense/H-bond work.

**Objective:** minimize time to an accepted observable. Modest measured energy/force bias is allowed; divergence, history-dependent force jumps and unsupported convergence labels are not accuracy compromises. Keep f32 bulk, gather ownership, persistent buffers/plans and explicit failure. pbc remains the preferred candidate; compare solver accuracy within the same parameterization and assess physical suitability separately from timing.

#### A — Correct the diagnostic baseline first (numerical agent)

**Files:** `sparse_system.rs`, its existing NS tests, `tests/sparse_f64check.rs`, existing diagnostic helpers. Do not rewrite the solver or launch a broad sweep.

- [x] **A1: NS normalization.** DONE 2026-09-14: `compute_z` divided by `n_orb` instead of `√n_orb` (36.3× understatement at R10; reported 3.75e-5 was really 1.36e-3). Fixed; `test_ns_device_residual_contract` now asserts reported vs independently recomputed f64. Post-fix true `||ZS−I||/√N = 7.6e-6`.
- [x] **A2: polish diagnostic.** DONE: true `3KSK−2KSKSK`, f64 throughout. On truncated mask oscillates ~7.5e-4 — post-hoc polish cannot repair, but cause is masked-fixed-point bias (below), not f32.
- [x] **A3: frozen-input separation.** DONE (`sparse_f64check.rs::frozen_input_dense_ref`): f64 Cholesky+dsyevd of the engine's own converged H_scc+S → `2Tr(P·H0) = −298.807141` vs DFTB+ −298.807668 (0.5 mHa). **All sparse inputs correct; error lives solely in purified K.**

**A-exit result (2026-09-14):** the "f32 floor" is retracted. At ~complete mask (deg~330) the f32 purifier reaches R_I 3.3e-5 and **23 μHa** vs the frozen reference, and *preserves* an injected exact K. At deg-175 (53% pairs) products `K·S`/`KSK` truncated to M_K bias the fixed point → +15.7 mHa. NS fix alone recovered ~2.5 mHa. **Mechanism (§15.12-2′, 2026-09-14, REVISED):** NOT intermediate-product truncation — the frozen-operands f64 diagnostic shows `‖P_M(P_M(KS)K)−P_M((KS)K)‖` is only 0.075% while the masked map `K′=P_M((K·S)·K)` has **no stable fixed point at K_ref|M_K** (host-f64 TC2 walks off it immediately, same ~1e-3 limit cycle as f32). The dominant error is **stored-K truncation destabilizing the map itself** — widening the KS intermediate halo would recover only 0.075%. Accuracy knob = stored-K degree (its own decay length); alternatives: top-k masks or LNV.

#### B — Stop early when justified; accept the actual returned state (state agent, after A)

- [ ] Distinguish reached tolerance, evidenced stagnation, exhaustion and failure. `stagnant=false`, 10× growth restoration and cap-based `NumericalFloor` are not an ordinary plateau detector. Use branch-aware histories and observed loss of convergence order; validate conditioning versus stagnation for the actual nonorthogonal/masked polynomial sequence. Do not revive “N non-improving checks” during normal TC2 conditioning. [Stopping-criterion reference](https://arxiv.org/abs/1507.02087).
- [ ] Stop at the first state satisfying the requested calibrated budget or an evidenced floor. Calibrate early-SCC electronic accuracy against its induced charge error relative to current SCC residual; tighten for final acceptance. Reset incompatible DIIS history when the map/accuracy changes. An inaccurate final solve must not inherit acceptance from an earlier cheap state.
- [ ] Revalidate restored/recovered state provenance, trace, dummy occupancy, idempotency and stationarity; classify P status against recovered K. A printed R_H without an acceptance check is only a diagnostic. Trace rescaling does not guarantee a valid spectrum or ground-state occupancy.
- [ ] Make floor permission explicit in run policy. A certified floor may eventually serve vibrations if it passes C; neither a status name nor `accept_numerical_floor=false` establishes quality. Until calibration use strict floor rejection for vibrational acceptance. Do not silently loosen thresholds or switch algorithms. Document existing automatic NS cold retry/SCC mixer rescue and expose recovery as an explicit configured driver policy when modifying that contract.

**Exit:** bounded cold/warm histories, no exhaustion mislabeled as a floor, matching diagnostics for returned states and unchanged/improved force repeatability. Count saved products and wall time, not only iterations.

#### C — Calibrate derivatives and mode stability (physics agent, after A)

Reuse Gates E–H and existing Rhai/force/Hessian machinery. Start with a small full Hessian and selected R10/R14 columns or Hessian-vector probes. Probes can reject a poor policy cheaply but cannot certify stability in unprobed directions.

| Quantity | Required evidence |
|---|---|
| Energy | Separate absolute bias from changes of that bias along scans; no system-independent 1 mHa total target for vibrations. |
| Force | Same-geometry reference bias and own-energy gradient consistency, including W/Pulay and taper derivatives. |
| Repeatability | Max/RMS force spread from cold, central-warm, perturbed-charge and reversed displacement-order solves, separate from bias. |
| Hessian | Existing h sweep: 0.01, 0.02, 0.05, 0.10 Å; raw asymmetry, rigid-mode leakage and sensitivity to electronic accuracy. |
| Frequencies | Relative errors for ordinary modes; absolute errors and uncertainty for soft modes. Derive limits from the application and measured floors. |
| Robustness/time | Every sampled failure/retry plus total time to accepted output; no cherry-picked successes. |

For a central-force Hessian, `δH[:,a]=−(δF(+h)−δF(−h))/(2h)`. If each nonsmooth force error has norm ≤ε_F, column error is ≤ε_F/h; independent random component errors give RMS σ_F/(√2 h). Smooth force bias must be differentiated separately. Balance nonsmooth error against O(h²) FD truncation using the measured plateau.

For `D=M^(−1/2)HM^(−1/2)`, a bound `||δD||2≤ε_D` gives eigenvalue uncertainty ≤ε_D. Internal eigenvalues exceeding that uncertainty have a defensible positive sign. Variation across h/tolerances gives an empirical uncertainty estimate, not automatically a rigorous bound. For ordinary modes `δω/ω≈δλ/(2λ)`; soft modes need special attention.

**No artificial positivity:** relax at the accepted method's own stationary minimum. Investigate significant negative internal modes using h variation, tighter solves and an energy scan along the mode. Preserve genuine instabilities. Report raw results before symmetry/rigid-motion projection; never clip eigenvalues, shift the Hessian diagonal or project away an internal instability. Tiny rigid-mode signs compatible with uncertainty are not evidence of a physical saddle.

Freeze support/plans/model settings and calibrated final accuracy policy across the stencil. Restore the same central seed for each sign; do not chain `+h→−h`. Compare default `W=2(ZH)K`, legacy `2KHK` and reference EDM contractions on **identical approximate operands**; neither shortcut identity validates arbitrary nonstationary/truncated K. Check cutoff smoothness wherever the stencil crosses a taper boundary.

**Output:** a small Pareto table of wall time versus relative-energy, force and frequency errors. Select explicit screening/relaxation/vibration policies from evidence, not guessed defaults. A coarse relaxation must be re-relaxed and checked under the final vibration policy.

#### D — Select one justified numerical improvement after A–C

- [ ] **P-TC2:** compare after inverse certification, including recovery/trace restoration/diagnostics and total SCC/force time at equal quality.
- [ ] **Compensated f32:** only if identical-input diagnostics implicate dot accumulation; benchmark an opt-in endgame variant against 4-accumulator/ACC8. Check compiler reassociation and register cost. Keep PR5: no GPU f64 matrix arithmetic. Host-f64 small reference experiments remain diagnostics. Late compensation cannot undo bad upstream Z/K0. Double-single storage requires separate evidence that storage rounding dominates.
- [ ] **LNV:** only if H-dependent subspace error or poor masked stationarity remains limiting. Inventory/reuse `bsr4_lnv_gradient`; define L support, actual K(L), every projection, electron-count constraint and safeguarded descent. Differentiate the **implemented masked functional**, verify its electronic gradient by finite differences, then derive/check nuclear forces. The untruncated combination kernel is not proof for arbitrary masked products. Count line-search products; lower energy outside admissible occupation/charge constraints is not improvement. No guaranteed recovery fraction or “first-order tail repair.” [Nonorthogonal LNV reference](https://www.physics.rutgers.edu/~dhv/pubs/local_copy/rw_dms.pdf).
- [ ] **Masks:** separate K/P and Z injection (current top-k couples them). Keep radial control; compare symmetric magnitude selection and omitted-product bounds at equal executed work. Independently audit the KS intermediate: the ZS halo does not fix `T=KS` projected onto M_K. Use expanded diagnostic support before designing a bounded production contribution schedule. Freeze selection across ±h and include oracle/setup cost.
- [ ] **Dummies:** later test structural zero dummy density throughout initialization/update/recovery, retaining nonsingular dummy S. Only then exclude dummy spectrum from bounds; fuse enforcement into existing writes and measure products saved. Do not modify physical occupations.

Do not develop every branch speculatively. Change one structural feature, remeasure C, and retain only a demonstrated accuracy/time benefit. Electronic smearing changes the target energy/free-energy and force convention; it is deferred rather than introduced as a free numerical accuracy knob.

#### E — Remove measured throughput costs (performance agent, after an accepted scalar policy)

- [x] Profile release builds on the actual NVIDIA GPU: **DONE 2026-09-14 (+ §15.12-3/4 follow-ups).** `SparseBsr4Gpu::prof_*` passthroughs + ticks at every stage boundary; `RUST_DFTB_PROF=evt|mark`. Raw data `debug/prof_sparse/*.txt`. **Findings:** `tc2.ksk` = 73–83% of device time (cost ∝ nnz_out × deg_operand); `scc.k0` ≈ 5–8%. **§15.12-3 DONE:** common-path one-read TC2 — speculative Q+resid enqueued behind trace partials, one blocking read, guard path recomputes (~1.8% of iters); energies bit-identical all three workloads. **§15.12-4 DONE:** `RUST_DFTB_KTIME=1` true kernel START/END — markers undercounted ksk ~16% (12.33ms real vs 10.34ms elapsed on deg330); products = 86–88% of wall. **§15.12-2 DONE:** `RUST_DFTB_TC2_HIST` replay-validated stagnation detector `RUST_DFTB_TC2_STOP_W=28` (env-gated; deg330 −15% wall bit-identical, zero false fires R14/SiH4; W=8 diverged live on R14 — rejected). Remaining: batch decisions; init/gamma/W share on bigger systems.
- [ ] Preallocate SCC charge staging; reduce stationarity to device partials instead of downloading O(nnz). Move sparse force contractions to pair ownership/atom gather when their measured share justifies it. Respect the dense agent's ownership; no global atomics or per-step kernel/buffer builds.
- [ ] For product p count `T_p=Σ_output_blocks number_of_retained_k_terms`; about `128·T_p` FLOPs. Model full evaluation as geometry + NS + all SCC initialization/products/reductions/mixing + W/forces. Measure bytes, metadata and launches too. Degree alone is not a cost model.
- [ ] Cache size is `64·MAX_LEFT_BLOCKS` bytes/WG. A wide Z/halo can inflate resource reservation in narrow-K products. Consider per-product specialization/buckets or §15.5's exact global-gather route only with profiling evidence. Query compiled local/private memory and residency limits; do not infer a degree-380 cap from 49 KiB.
- [ ] Benchmark a small matched size series with physical orbital counts and equal accepted E/F quality. Separate setup from repeated evaluation/Hessian throughput. Report dense crossover only after matching parameterization, accuracy, hardware and outputs; the broken dense SCC harness cannot supply a trusted comparison.
- [ ] Batch independent ±h states sharing topology/plans, with independent q/Z/K/scratch/status. Choose batch size from measured memory/occupancy; output must not depend on batch order. Include failures/retries and amortize setup over the real workload.

**Agent handoff:** source changes, unfiltered diagnostics under `debug/`, accepted/rejected settings with actual values, wall-time impact and remaining uncertainty. Update this report and the roadmap when implementation status changes. No completion claim without user acceptance.

#### F — Device residency contract + batched column launches (2026-09-18)

Measured motivation (R18 frozen, RTX 3090): the GPU pair path evaluates
~0.5–1 GFLOP of pair work per displaced eval — **~30 µs at peak** — yet
took ~11 ms. The gap is not kernel throughput, it is per-eval PCIe
traffic and host work. **Residency is a hard contract, not an
optimization:** no buffer that is derivable on device may cross PCIe
inside a force/Hessian eval. Residual per-eval host traffic is capped at
O(n_atom) scalars/forces plus explicitly justified diagnostics.

Residency leaks found (fixed 2026-09-18, see report §15.25):

- [x] `restore_central_state` uploaded `s.k` (M_K ≈ 26 MB) + `s.z` (26 MB)
      host→device **every column** — data that had been downloaded to
      host at snapshot time. → device-resident `GpuCentralState`
      (`k/z/k0/w0` value-buffer copies taken once at snapshot) +
      `copy_f32` device→device restore (~100 µs).
- [x] `forces_frozen` uploaded `s.k` + `s.w0` (~52 MB) every eval →
      `hs_contract` now consumes the snapshot buffers directly (zero
      copies — the kernels only read them).
- [x] `set_coords` GPU path read back `h0`/`s` value buffers (~25 MB)
      every eval "for diagnostics" → mirrors now lazy:
      `h_bsr()/s_bsr()` refresh on demand; hot path skips entirely.
- [x] `snapshot_electronic_state` restored converged K via
      `inject_k_values` host upload → `copy_f32` from the device copy.

Remaining per-eval transfers (accepted, O(n_atom) or O(n_rep)): xyzu
upload 4N f32, dq upload N, v/kdummy readback N, pe_rep readback n_rep,
5×3N force readback. Coincident-atom guards stay host-side (O(n_pairs)
compare, fail-loud; the same loops also feed the pair list audit).

Next steps (in order):

- [ ] **F1 — batched column launches.** The point of residency: launch
      `hs_contract` once per *color class* of independent displacements
      (atoms ≥ r_full apart can be displaced simultaneously without pair
      overlap). Batched xyzu/dq/out buffers, one upload + one readback
      per class → 8–32× more work in flight per launch; the only regime
      where the 3090's throughput is actually engaged.
- [ ] **F2 — column-local pair subranges.** Each column's H/S blocks and
      force contributions touch only the ~deg_hs pairs of the displaced
      atoms. Precompute per-atom pair sublists at init; pass subrange
      offsets into the same kernels. Turns the all-pairs eval into
      O(color_class·deg) work.
- [ ] **F3 — fuse the frozen-eval launch chain.** Current eval =
      hs_diag + hs_assemble + gamma_matvec + gamma_force + hs_contract +
      rep_eval + force_gather ≈ 7 launches + ~6 syncs. Fuse the
      dq-independent stages and merge readbacks into one mapping.
- [ ] **F4 — pe_rep reduction on device** (n_rep readback → 1 float) and
      skip `compute_v` host v in the pure-force path (v_buf already
      device-resident; host v only needed for energies).
- [ ] Re-benchmark R18 frozen after F1–F3; update §15.22 projection
      table. Success metric: frozen eval ≲2 ms at N=1648 and Hessian
      column time dominated by eigensolve, not forces.

##### F.1 — Batched frozen-eval design study (2026-09-18, pre-implementation)

Ground-truth facts verified in code before designing:

- **`hs_contract` never reads `h0`/`s`.** It recomputes SK splines,
  rotation, and analytic derivs from `xyzu` + `sk_*` tables per pair and
  contracts directly against `k_vals`/`w_vals`/`v_atom`
  (sparse_hs.cl:265–367). `assemble_hs_dev` — `geom.hs` ≈ 0.374 ms/eval,
  ~23 % of the frozen eval — is **dead work in the frozen path**. The
  only consumers of assembled H/S are the SCC/purify stages, which
  frozen columns never run.
- **Pair topology is static.** `hs_pairs` is built once at construction
  (`hs_pairs_from_mask`); the Verlet-skin contract guarantees no pair
  enters/leaves the cutoff within a ±h displacement. The per-geometry
  O(n_pairs) host coincident-atom guard (~0.2–0.5 ms of the unprofiled
  floor) is re-validation of a static list — hoistable to init or to a
  device-side min-r² reduction while keeping the fail-loud semantics.
- **The central electronic state is shared read-only.** Frozen mode
  evaluates every displaced geometry against the *same* `k0`/`w0`/`dq₀`.
  Per-replica state is therefore only geometry + outputs — no per-replica
  K/W copies, no replica SCC machinery. This is what makes the batch
  trivially parallel: replicas are independent gather problems over
  shared inputs.
- **Frozen force eval consumes:** `xyzu` (geom), `dq₀` (shared),
  `v_atom` = γ·dq₀ at the displaced geometry (needed by the SCC force
  term inside `hs_contract`, arg `v_atom`), `k0`/`w0` (shared),
  repulsive pairs. Emits per-atom forces.
- **Frozen force eval does NOT need:** `hs_diag`, `hs_assemble`,
  `hs_kdummy`, `s_inf` reduce, H_scc build, K/Z restore (the live
  buffers are untouched — contract reads `central_dev` directly; the
  current `restore_central_state` call in the vib loop is hygiene for
  the SCC modes, dead work in frozen mode worth auditing).

Options considered:

- **A. One mega-kernel tiled over systems — rejected.** The eval is
  already a *sequence* of kernels with different work shapes (matvec,
  pair-contract, n-body force, gather). Fusing systems inside each
  kernel adds nothing over giving each kernel a replica axis — and
  couples failure/timing of unrelated stages. Not needed.
- **B. B independent workspaces, pipelined submissions — rejected.**
  Zero kernel changes, but the per-eval host orchestration, syncs, and
  guard loops stay serial; the ~0.7 ms floor is amortized only if the
  queue never drains, which the current synchronous code structure
  prevents. Also multiplies full workspaces (h0/s ≈ 26 MB/replica) for
  buffers the frozen path doesn't even need.
- **C. Replica axis on the existing 5 kernels — CHOSEN.** Extend each
  frozen-path kernel's NDRange with `get_global_id(1) = b` (or flat
  `p + b·npairs`); index per-replica buffers at `b·stride`; shared
  inputs (`pairs`, `sk_*`, `dq₀`, `k0`, `w0`) read-only, unchanged.
  Same math, same order, deterministic = bit-identical to sequential.
  No atomics anywhere (each work-item still owns one output slot).
- **D. Column-local ΔF evaluation — composes with C, phase F2.**
  Displacing atom m changes force contributions only on pairs touching
  m (~deg_hs ≈ 121 at R18) plus the γ′ row (j,m) for all j (O(N), since
  F_j gets a changed term γ′_{jm}·dq_j·dq_m). Atoms outside the touched
  set keep central-geometry forces: the eval can return
  `F⁰ + ΔF` with ΔF computed on the touched set only — ~200k→~121
  pair-contract items + one O(N) γ′ row per column. Per-atom pair
  sublists already exist (`pair_gather_adj`); a batched local launch
  iterates a concatenated touched-pair list with per-replica offsets.
  ΔV similarly updates only via the (j,m) γ term — an O(N) row vs the
  full O(N²) matvec.

Memory budget at R18 (per replica): `xyzu` 4N·4 B = 26 KB,
`v_atom` N·4 B = 6.6 KB, `pf` 2·n_pairs·16 B ≈ 6.4 MB (dominant),
`pf_rep`+`pe_rep` ≈ n_rep·16 B ≈ 0.5 MB, `f_out` ~4N·4 B = 26 KB ⇒
≈ **6.9 MB/replica** → B=8 ≈ 55 MB, B=16 ≈ 110 MB, B=32 ≈ 220 MB on a
25 GB device. With F2 column-local, `pf` shrinks to the concatenated
touched list (~deg·B) and replicas cost ~0.2 MB each — B limited by
orchestration, not memory.

Expected per-column cost at R18: F0 alone ~1.6→~1.0 ms (drop dead
assemble + host guards); +F1 batch B≈8–16 → ~0.5–0.7 ms (floor
amortized, kernels already near occupancy); +F2 → ~0.05–0.15 ms
(launch-bound; needs bigger batches to feed the GPU). Caveat: the dense
4944×4944 eigensolve (~100 s of the 138 s wall) then dominates — see
report §15.26; batching matters most for N≥3k columns and for any future
non-frozen (fixq) mode, where the T06 compact-domain pattern from the
qmqm batched-SCC workstream is the model (replicas there DO need
per-slot convergence state — deferred, not needed for frozen).

Invariants (same as every GPU path here): gather-only, write-once
outputs, zero atomics; persistent buffers allocated at init for
`B_max` (env `RUST_DFTB_VIB_BATCH`, default 8 — B=1 must reduce to
today's sequential semantics exactly); batch result must equal
sequential column-by-column **bitwise** (same kernel math per replica);
CPU reference path untouched; missing `central_dev` still fails loud.

Phased plan + gates:

- [ ] **F0 — frozen-path dead-work removal (no kernel changes).** Skip
      `assemble_hs_dev`/mirror marking when the caller will only do
      frozen forces (add a `set_coords_light`/mode arg — do NOT weaken
      `set_coords`'s SCC contract); hoist the coincident-atom guard to a
      device min-r² reduction over the static pair lists (fail-loud
      unchanged); skip the `compute_v` host readback in the force-only
      path. Gate: parity tests + R18 eval ≲1.0 ms.
- [ ] **F1 — replica axis + `forces_frozen_batch`.** 2D NDRange on
      gamma_matvec/gamma_force/hs_contract/rep_eval/force_gather;
      batched `xyzu`/`v_atom`/`pf`/`pf_rep`/`f_out`; one upload + one
      readback per batch; `B=1` ≡ sequential. Gate: batch-vs-sequential
      force columns bitwise equal; R10/R18 bounded bench.
- [ ] **F2 — column-local subranges.** Concatenated touched-pair list +
      per-replica offsets from `pair_gather_adj`; O(N) γ′ row kernel and
      ΔV row update; ΔF added to stored central F⁰ on device; readback
      touched atoms only. Gate: identical Hessian columns vs F1 output
      on a bounded column set.
- [ ] **F3 — fuse + resync.** Merge launches sharing `xyzu[b]` reads;
      single pe_rep device reduction. Re-benchmark; update report.
