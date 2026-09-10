---
type: TopicalAudit
title: Sparse GPU DFTB — bugs vs physical misconceptions vs f32 arithmetic floor
tags: [topic, gpu, f32, dftb, sparse, bsr4, numerical-floor, opencl, nanocrystal]
timestamp: 2026-09-10
---

# Sparse GPU DFTB — bugs vs floor vs missing pipeline

**SSOT for:** what is a **bug**, what is a **physical misconception**, what is a
**method stopgap**, and what is the **measured f32 / mixer / FD floor** on the
sparse Si/H path. Compensation (Kahan, hybrid f32/f64 islands, compensated
SpGEMM) is a **later dedicated effort** — this file is the map that effort
starts from.

**Do not chase G3.4 `rel < 1e-3` at `h=1e-3` Å as if it were an f32 law, or Kahan on device NS.**
Those mix FD, SCC stopping, and possible bugs. **Do not** mark a test green by hiding a real bug.
The 2026-09-10 second review (`doc/prokop/reports/2026-09-10_gpu_accuracy_physics_performance_second_review.md`)
says the previous “do not chase below this floor” table is **not** an established numerical limit.

Dense H-bond floor map (different solver): `f32_floor_dense_hbond.md`.
Interpolator method spec (shared CPU/GPU SK tables): `sk_interpolation.md`.
Task manifest (read **§0** first):
`tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md`.

Device: NVIDIA RTX 3090. SK: matsci-0-3. Tests:
`gate_g3_energy.rs`, `gate_f_geom_opt.rs`, `gate_g_hessian.rs`.

---

## 1. Algorithm map (sparse DFTB, one nanocrystal)

Dense CPU f64 eigen-SCC is the **reference**. Sparse production algebra is
**TC2 purification of K** (no orbital diagonalization in the loop). GPU is
f32. Host Newton–Schulz is the Z path that physics gates actually use.

```
geometry R (Å)
  │
  ├─ SK tables ── C² B-spline V(r), V'(r) ──► H0, S     [CPU assembly today]
  ├─ Hubbard U, γ(R) ────────────────────────► V = γ·Δq [CPU today]
  └─ repulsive Spline (SKF coeffs, not B-spline) ► E_rep, F_rep
  │
  pad physical orbs → BSR4 (H dummy: S_dd=1, H_dd=E_dummy=2.0)
  │
  SCC (repeat until rms(Δq) plateaus)     [scc.rs::run_sparse_scc]
  │  Hscc = H0 + ½ S ⊙ (V_a + V_b)        [physical slots only]
  │  Z  ≈ S⁻¹   Newton–Schulz once per geometry (S fixed)
  │  K0 from Gershgorin of **ZH** (not ZHZ; ZHZ is only inside K0)
  │  TC2 → K,  KSK=K **and** Tr(KS)=N_occ
  │  q  = Mulliken(K, S)
  │  mix q
  │  on converge: rebuild H[q_out], purify once more (K,H,q one state)
  │
  E = 2 Tr(K H0) + ½ Δq·V + E_rep
  D = 2K,  W = 2 K Hscc K                 [unpad to physical]
  F = F_nonSCC(D,W) + F_shift(D,V) + F_γ'(Δq) + F_rep   [CPU compute_forces_from_dw]
  Hessian = FD of F, unsymmetrized: H[:,a] = −(F(+h)−F(−h))/(2h)
```

| Stage | What actually runs in G3/F/G | Intended production | File |
|-------|------------------------------|---------------------|------|
| SK V, V' | CPU f64 B-spline | same controls packed to GPU later | `interpolation.rs` |
| H0/S | `HamiltonianBuilder::build_non_scc` every geom | GPU pair assembly, persistent | `hamiltonian.rs` |
| Z ≈ S⁻¹ | **host-roundtrip** `newton_schulz_inverse` in G3; **`SparseDftb` workspace intersection NS** (host `\|I−T\|`) | `newton_schulz_inverse_dev` — **do not use** | `gpu_sparse.rs`, `sparse_system.rs` |
| K / TC2 | host-roundtrip in G3; device TC2 in `SparseDftb` | same | `gpu_sparse.rs` |
| γ, Hscc, mix | CPU in G3; CPU γ + upload Hscc in `SparseDftb` | GPU Hscc kernel later | `scc.rs`, `sparse_dftb.rs` |
| E, F | `eval_sparse_energy_forces` in G3/F/G | `SparseDftb::scc` / `forces` (CPU F still) | `sparse_dftb.rs` |
| FIRE / Hessian | tests own the loop in F/G | `SparseDftb::fire_step` / `relax` | `sparse_dftb.rs` — **not USER-confirmed** |

`SparseSystemWorkspace` (`sparse_system.rs`) was written as that owner. Its
`run_scc` is **one-shot purify** (Z, K0, TC2, Mulliken). No γ, no Hscc update,
no mix, no loop. **G3/F/G do not call it.** `dftb_engine.rs` leftover purify used `newton_schulz_inverse_dev`
(N4-wrong). **Commented 2026-09-10.** Production jobs are `sparse_*` →
`SparseDftb` (workspace intersection NS). Do not restore `_dev`.

---

## 2. Classification (read this first)

| ID | Symptom (measured NVIDIA 3090) | Class | Why | What to do |
|----|-------------------------------|-------|-----|------------|
| **B1** | Device NS `newton_schulz_inverse_dev`: kernel `R_Z=1.87e-5` vs f64 `\|ZS−I\|_F/√N=2.32e-3` on returned Z | **BUG** | Residual scalar does not describe returned Z. **Intersection NS + host `\|I−T\|` of downloaded T** (workspace `compute_z`, 2026-09-10 RTX 3090) reaches `R_Z≈6e-8`. Do not use `_dev` or the plan kernel for Z. | Keep `test_newton_schulz_inverse_dev` **red**. Production `SparseDftb` uses workspace intersection NS. |
| **B2** | `SparseSystemWorkspace::run_scc` named SCC, body is purify-once | **BUG / incomplete API** | Compiler: `h_scc`, `plan_zh`, `plan_bz` unread. | Production SCC is `SparseDftb::scc`. |
| **B3** | `dftb_engine` sparse purify used `newton_schulz_inverse_dev` | **BUG (switched 2026-09-10)** | Production binary on unresolved Z. | Now **host** NS; device call commented until B1 reverified. |
| **B4** | Gate C “locality plateau”, Gate E “η_asym=0”, Gate F old 0.93 Å | **TEST LIES** | Gate E copied the triangle. Gate F optimized `Tr(K H0)` without E_rep. | Do not copy. |
| **B5** | TC2 accepted `\|KSK−K\|<tol` without occupation | **CONTRACT** (second review §3.5) | K=0 is idempotent. | Host+device TC2 now require `\|Tr(KS)−N_occ\|≤5e-2` to accept. |
| **B6** | SCC returned K from H[q_in] but energy/V from q_out | **FINITE-SCC mismatch** §3.2 | Same class as dense `finalize`. | `run_sparse_scc` now rebuilds H[q_out] and purifies once more; prints `r_scc`, `R_H`. |
| **P1** | matsci SK `q0`: Si=0, H≈0.49 | **parser bug**, not “SK has no valence” | Trailing numeric fields on the onsite line (second review §7). Explicit `[4,1,1,1,1]` bypasses it. Shared with dense — **do not change the parser in a sparse-only pass** (other agent on dense). | Keep valence fixture. |
| **P2** | Interpolator extra-control fitter missing | **method stopgap**, not f32 | See `sk_interpolation.md`. | Do not pad more zeros. |
| **M1** | Host NS allocated every **SCC iter** | **harness** | S is fixed at a geometry. | **Z computed once** in `run_sparse_scc` (2026-09-10). Remaining: buffers still allocated inside each TC2/NS call. |
| **M2** | No `SparseDftb` lifetime | **coded 2026-09-10, not USER-confirmed** | `sparse_dftb.rs` + CLI `sparse_*` on `dftb_engine` (`scripts/test_sparse_dftb_sih4.rhai`, `userguide/sparse_dftb.md`). NVIDIA SiH4: NS `R_Z=6e-8`, SCC E=−2.764, reuse SCC 17→4 iters. Remaining: CPU H0/S, CPU F, Hscc upload per mix iter. | Drive jobs as scripts, not new binaries. Do not time `cargo test`. Gate F/G still call `eval_sparse_energy_forces`. |
| **F\*** | Gate G 0.11%, G3.4 rel 4e-3, … | **not an established f32 floor** | Second review §4: those numbers mix solver stopping (NS 1e-5, TC2 1e-4, SCC 1e-5), FD truncation, and arithmetic. Need h-sweep + tighter tolerances **frozen-input** before calling them floors. | Print `r_scc`/`R_H`; do not loosen G3.3. Do not claim “do not chase below F5”. |

---

## 3. Production pipeline — we do **not** have it

A normal DFT/DFTB code:

```
INIT (once)        load SK, compile OpenCL, preallocate every buffer and kernel
PER GEOMETRY       upload R → neighbors → H0,S,γ  (no Program::build)
SCC (warm-start)   iterate; only set_arg + enqueue
FORCES             D,W already on device → F
MD / FIRE / Hessian displacements → PER GEOMETRY
```

**CPU already follows this:** `methods/dftb/dftb_cpu.rs` (`DftbCpu`).

**Sparse GPU does not.** Pieces exist; nothing owns the lifetime.

| Piece | What it is | What happens in G3/F/G |
|-------|------------|------------------------|
| `SparseBsr4Gpu::new` | Compile BSR4 program, cache `Kernel` handles | **Once per test process.** Good. Not once per MD step. |
| `GpuRuntime` | OpenCL context | New runtime inside `SparseBsr4Gpu::new`. Tests do not rebuild *inside* FIRE, but **every `cargo test` function** constructs a new one. |
| `run_sparse_scc` / `purify_h` | Physics SCC + host NS/TC2 | **Allocates GPU buffers every NS iteration** (`buf_f32`, `zero_f32`, download Z). Rebuilds BSR4 from dense H0/S every geometry. |
| `eval_sparse_energy_forces` | H0/S + SCC + CPU F | Correct physics wrapper. **Not** a persistent solver. Gate F/G call it 10² times. |
| `SparseSystemWorkspace` | Intended persistent owner | Unused by physics gates. `run_scc` ≠ SCC. |
| `cargo test` | Honest **physics** checks | Compiles the crate, compiles OpenCL, builds CPU SK, then runs. **Not a benchmark. Not the product.** |

**`SparseDftb` exists** (`sparse_dftb.rs`, 2026-09-10). Drive it with **`dftb_engine`**
`sparse_*` + `scripts/test_sparse_dftb_sih4.rhai` (`userguide/sparse_dftb.md`),
not a new binary. NVIDIA SiH4: NS `R_Z=6e-8`, SCC E=−2.764 Ha, reuse SCC 17→4
iters, FIRE/MD on the same object. Not USER-confirmed. Remaining per-geometry:
CPU `build_non_scc`; per SCC iter: CPU γ + Hscc upload; per force: download K +
CPU `compute_forces_from_dw`. No `Program::build` / `Kernel::builder` /
`Buffer::builder` in `scc`/`fire_step`.

```
SparseDftb::new(sk, sk_dir, species, coords)  // INIT
  .set_coords(R)                              // values only; frozen mask
  .scc()                                      // workspace NS + device TC2
  .forces() / .fire_step() / .md_step() / .relax()
```

**Do not time `cargo test` or `eval_sparse_energy_forces` and call it GPU DFTB.**
Gate F/G still use the allocating `scc.rs` path until they are switched.

---

## 4. What “interpolator fitter” is (plain language)

Slater–Koster files are **numbers on a 1D grid** (`dr` typically 0.02 Bohr).
We interpolate with a **cubic B-spline**. A cubic B-spline does not live only
on those samples: it needs a few extra **control points** off each end of the
table so the curve knows how to start and how to go to zero. Those extra
points are **not** extra physics measurements; they are degrees of freedom
for boundary conditions.

- **Left end (already the right kind of thing):** phantom control
  `c_{-1}=2c_0−c_1` at evaluation (`V''=0` at the first knot). SK values at
  small `r` are large — never force them to 0.
- **Right end (stopgap, blunt):** append `N_PAD_END=4` **function samples
  hardcoded to 0**, then refit the tridiagonal
  (`fit_bspline_controls_zero_end`). That killed the Neville explosion
  (H–H at 10.4 Bohr was **−0.4 Ha**). It is **not** a BC solve: the zeros
  leak into the last real samples, and cutoff is only `~4·dr` past the last
  grid (DFTB+ used a 1 Bohr polynomial fudge; we are not copying that).
- **GPU dummy 0 at `r=0`:** index padding, not a fitted left control.

**The fitter** = linear (least-squares) solve: extra controls are **unknowns**.
Constraints: (1) interpolant on the **tabulated** region still matches the SK
file; (2) `V` and `V'` → 0 at a chosen cutoff. **Do not** implement as “pad
more zeros.” **Do not** restore Neville / `poly5_to_zero`.

This is **independent of f32**. Interior SiH4 / H2O H/S already match at
~1e-7. Shared with the dense H-bond task. Full spec: `sk_interpolation.md`.

Production **evaluation** is already B-spline (`eval_into` /
`eval_with_deriv_into`). GPT-5.6 “Hermite still in the force path” is
**stale** — `SkTableSp::eval_shell_integrals_and_derivs_into` calls the
B-spline. Hermite remains as an unused reference path.

---

## 5. Honest test contract (do not fake green, do not demand the impossible)

| Test | Must prove | Must not demand |
|------|-----------|-----------------|
| G3.1 | E = E_el + E_rep; same Spline both sides | \|dE_h0\| ≪ 1e-7 |
| G3.2 | sparse SCC (no eig in the loop), valence q0 | max\|dq\| < 1e-8 |
| G3.3 | F from D=2K, W=2KHK vs dense F | — 3.7e-6 is already below F3 |
| G3.4 | own F ≈ own −dE/dR | rel 1e-3 at h=1e-3 Å |
| Gate F | FIRE on E_tot + analytic F; Si–H in 1.40–1.55 Å | \|F\| at 1e-8 |
| Gate G | unsymmetrized FD-of-F Hessian vs dense; print η_asym, \|\|ΔH\|\|_F | η_asym = 0; n_unstable ≤ 6; 0 cm⁻¹ rigid |
| Gate E/C old | — | Do not treat as physics |

`cargo test` remaining **red** is required for B1 (device NS). Keep it red.

---

## 6. Measured numbers (2026-09-10, RTX 3090, matsci-0-3, SiH4)

**G3** frozen 1.48 Å **distorted** SiH4 (not tetrahedral). Numbers below were **before** the 2026-09-10 SCC finalize / Z-reuse; re-measure on NVIDIA after that change.

**Do not treat F1–F6 as established f32 limits** (second review §4, §9). Next sparse precision work: frozen-input NS/TC2 residuals; h ∈ {0.01, 0.005, 0.0025} Hessian sweep with tighter SCC; translation `||Ht||`. Kahan on host `Tr(K H0)` of f32 K will not move the Hessian. SpGEMM block-accumulation compensation only after frozen-input evidence.

**Gate F** FIRE from 1.60 Å: 67 steps, mean Si–H **1.477 Å**, \|F\|=4.18e-4,
E_tot=−2.826057. Old Tr(KH0) path: 0.93 Å (unphysical).

**Gate G** at that geometry, h=0.01 Å, 30+30 force evals:

| | sparse | dense |
|--|--------|-------|
| η_asym | 3.87e-4 | 3.89e-4 |
| \|\|H\|\|_F | 3.083 | 3.086 |
| max\|ΔH\| | 1.17e-3 | — |
| \|\|ΔH\|\|_F/\|\|H\|\|_F | 1.10e-3 | — |
| ν (cm⁻¹) | 856, 856, 857, 990, 991, 2315, 2344, 2345, 2346 | within 4 cm⁻¹ |

CSV: `debug/sparse_review/gate_g_h_sparse.csv`, `gate_g_h_dense.csv`.

---

## 7. Handoff — this agent vs compensation LLM vs pipeline LLM

**This sparse-physics agent:** implement review §3.2/3.5/6.2 contracts; keep N4 diagnostic red; do not start Kahan; do not touch `qmqm/gpu_forces.cl` / `gpu_dftb.rs` (dense agent).

| Layer | Status | Next |
|-------|--------|------|
| SCC Z reuse + finalize + TC2 occupation | **coded 2026-09-10**, GPU-verify G3 | this agent |
| Device NS N4 | Diagnostic prints `\|ZS−I\|` f64; still must fail if inverse is wrong | algebra, not Kahan |
| `dftb_engine` NS | Host path | do not restore `_dev` without B1 |
| SK q0 parser | Shared; other agent on dense | valence fixture until shared fix |
| Extra-control fitter | Stopgap | `sk_interpolation.md` |
| `SparseDftb` lifetime | **coded** + CLI `sparse_*`; NVIDIA SiH4 §0.7 (E=−2.764, reuse 17→4). Not USER-confirmed. | Gates onto this object; remaining CPU H0/S and CPU F |
| Compensation / Kahan SpGEMM | **Not started.** Review §5.3: 4×4 block is short; accumulate over contributing blocks first. | after frozen-input evidence |
| Hessian h-sweep | Gate G still one h=0.01 | experiment C |

**Do not still chase:** G3.4 rel 1e-3 at h=1e-3; Gate G rigid 0 cm⁻¹;
AT-style rms `<1e-6` on SiH4; restoring Neville; calling `SparseSystemWorkspace::run_scc` production SCC; reporting PoCL as GPU.
