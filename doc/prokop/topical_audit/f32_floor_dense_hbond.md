---
type: TopicalAudit
title: Dense H-bond GPU pipeline — bugs vs f32 arithmetic floor
tags: [topic, gpu, f32, dftb, hbond, numerical-floor, opencl]
timestamp: 2026-09-10
---

# Dense H-bond GPU pipeline — bugs vs f32 floor

**SSOT for:** what is a bug / physical misconception, what is a method stopgap,
and what is the **measured** dense GPU floor after Package 2 (frozen-H) +
Löwdin Newton + f32 GEMM Kahan (2026-09-10, NVIDIA 3090 `--release`).
Compensation is no longer “unstarted”: metric repair is in; Kahan-on-GEMM was
measured and does **not** remove AT `δ_CH`. Remaining limiter is Jacobi `C'`
(`||C'ᵀC'−I||`), not occupied-ε vs host GEVP and not `Tr(DH)`.

**Do not** chase charge-rms `<1e-6` or AT `|dE|<1e-5` on f32 N~90. Those
numbers are below the measured eigen floor. **Do not** mark a test green by
hiding a real bug. Sparse (`rust_dftb/src/methods/sparse/`) is a different
agent — out of scope.

Tests: `rust_dftb/tests/gpu_hbond_physics.rs`. SK: mio-1-1. GPU: NVIDIA RTX 3090.

Related: `sk_interpolation.md` (B-spline BCs), `gpu_scc_pipeline.md` (loop
plumbing), H-bond task `tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.manifest..md`.

---

## 1. Algorithm map (dense DFTB, one system)

CPU f64 is the **reference**. GPU is f32 except where a **f64 island** is
already documented below.

```
geometry (Å)
  │
  ├─ SK tables ── B-spline V(r), V'(r) ──► H0, S          [assembly]
  ├─ Hubbard U, γ(R) ────────────────────► G (n_atoms²)   [host f64→f32 upload today]
  └─ repulsive splines ──────────────────► E_rep, F_rep
  │
  SCC (repeat until Δq plateaus)
  │  Δq = q − q0
  │  V  = G · Δq
  │  H  = H0 + ½ S ⊙ (V_a(μ)+V_a(ν))
  │  X  = S^{-1/2}          (once per geometry)
  │  H' = X H X
  │  Jacobi(H') → ε, C'
  │  C  = X C'
  │  D  = 2 Σ_{k occ} C_k C_kᵀ
  │  q  = Mulliken(D, S)
  │  mix q (DIIS)
  │
  E = 2 Σ_{occ} ε − ½ Δq·V − q0·V     [equivalent to Tr(D H0)+½ Δq·V]
  F = F_nonSCC(D,W) + F_shift(D,V) + F_γ'(Δq) + F_rep
```

| Stage | CPU f64 | GPU f32 | File |
|-------|---------|---------|------|
| SK B-spline V, V' | `interpolation.rs` | `dftb_hamiltonian.cl`, `gpu_forces.cl` | same controls packed in `gpu_prep.rs` |
| H0/S assembly | `hamiltonian.rs` | `assemble_pairs` | pair buckets 1×1 / 1×4 / 4×4 |
| γ(R) **value** | `gamma.rs::gamma_full` | `dftb_hamiltonian.cl::gamma_full` (on-device); SCC test **uploads host f64→f32** | ill-conditioned when U≈U' |
| γ'(R) force | `forces.rs::gamma_prime_full` | `gpu_forces.cl::gamma_prime_full_f32` | **f64 island** (2026-09-10) |
| S⁻¹/², H'=XHX, C=XC' | LAPACK | tiled Jacobi + `batched_gemm` | N>64: `gpu_tiled_jacobi.cl` |
| Density / Mulliken | `hamiltonian.rs` | `build_density_masked`, `mulliken_charges` | |
| Energy | `Tr(D H0)+½Δq·V` | **band form** `2Σε−½ΔqV−q0V` in `GpuSccPlan::compute_energy` | see §3.1 |
| Forces | `forces.rs` | `gpu_forces.cl` four kernels | CPU-fed P/W in AT/GC tests |

N: H2O=6, formic dimer=28, AT=87, GC=86. N≤64 = full-local Jacobi; N>64 = tiled.

---

## 2. Classification (read this first)

| ID | Symptom (measured) | Class | Why | Status |
|----|-------------------|-------|-----|--------|
| B1 | H–H at 10.4 Bohr Hss = **−0.4 Ha** (CPU) vs ~0 (GPU) | **BUG / unphysical interpolant** | Neville + `poly5_to_zero` on a ~1e-5 tail | **Fixed** — B-spline; do not restore Neville for Fortran-tail parity. Remaining: blunt zero-sample pad (`sk_interpolation.md`), not f32. |
| B2 | NVIDIA crash `CL_OUT_OF_RESOURCES` on 1×4 H2O/formic | **BUG** | `__local float2*` misaligned; UB | **Fixed** (`vload2`) |
| B3 | Replica cap `l_frags[128]` | **BUG** | silent truncation | **Fixed** |
| B4 | AT/GC `max\|dH\|≈0.4 Ha` | **not GPU** | was B1 on CPU SK tail, worst pair H–H past last grid | **Fixed** with B1. In-range AT `max\|dH\|=8.6e-8` |
| B5 | AT/GC gamma-force rel **1%**, H2O **2.6e-5** | **ill-conditioned f32 formula** | `Ua≠Ub` S' is two O(10³) terms that cancel; mio N–H ΔU≈0.011. OpenCL f32 3% off on covalent N–H (2-atom test). H-bond H···O at 1.92 Å was already exact. | **Workaround in:** `gamma_prime_full_f32` evaluates in **f64**, returns f32. AT/GC gamma-force rel **4e-7**. On-device **γ value** (`dftb_hamiltonian.cl::gamma_full`) still f32 — unused in the AT SCC test (host G). |
| F1 | AT GPU SCC rms plateaus **~7e-6** (CPU f64 on same H/S: 1e-9 / 20 iter) | **f32 floor** | `max\|H0\|=0.88 Ha` (not ~100). Occupation 49/49. Gap 0.124 Ha both. Mixer is not stuck. | **Do not chase `<1e-6`.** Test: rms `<1e-4` + plateau, not `<1e-6`. |
| F2 | AT `Tr(D·H0)` GPU vs CPU **7.1e-5** | **f32 projector, not the trace** | Host f64 `Tr` of the **same** GPU D matches the kernel to **1e-6**. f64 D from GPU C does **not** help (`max\|ΔD\|=6.8e-6`). Error is in **C/ε**. | Energy no longer uses `Tr(D H0)` — see F3. Kahan on the Frobenius sum would buy ~1e-6, **not** 7e-5. |
| F3 | AT `|dE|` band vs CPU **~2.8e-5 Ha (~0.8 meV)** | **`δ_CH`, not occupied-ε 2.5e-5** | Package 2 frozen-H: `max\|δε_occ\|~1.1e-6` (~20× smaller than `|dE|`). Production energy is band form; `δ_eig ≈ δ_CH = E_band − 2Σ CᵀHC` (~5e-5). After Löwdin Newton, `\|\|XᵀSX−I\|\|~2e-7` and `\|\|CᵀSC−I\|\|~2e-6` matches `\|\|C'ᵀC'−I\|\|` (Jacobi). Density `δ_D~1e-7`. The old `max\|δε\|=2.5e-5` was SCC-then-compare, **not** the frozen residual. H2O N=6: `|dE|~5e-7`. | **Honest contract:** N=6 keep `1e-5`; N~90 require `<1e-4` (bug line). **Not** `<1e-5`. |
| F4 | AT `½Δq·V` GPU vs CPU **2.9e-6**; `|dq|=1.2e-5` | **consistent with F1** | Charges and SCC energy term are at the rms floor. | OK |
| F5 | GPU finite-difference forces at **h=1e-3 Å** | **f32 energy noise** | Do not assert GPU F vs GPU FD at this stencil. h=1e-2 is O(h²)~1e-3 even on CPU. | Print-only at 1e-3; F vs CPU F is the contract |
| X1 | Right-end SK pad = hardcoded zeros | **METHOD STOPGAP** | Not f32. Extra controls must be *solved*. | Open — `sk_interpolation.md` |
| P1 | `Kernel::builder` / host DIIS / eig readback in `compute_energy` | **PERF / plumbing** | Not a precision floor. | Later |

**Already-applied f64 islands** (pattern for compensation work): Jacobi strip updates in `gpu_tiled_jacobi.cl`; `gamma_prime_full_f32`; occupied-ε sum in `compute_energy` (host f64).

**Tried and reverted:** `batched_gemm` inner product in **f64**. AT `|dE|` went **2.6e-5 → 3.7e-5**, DIIS **stopped plateauing**. Do not re-enable.

**Tried and kept:** f32 Kahan across K in `batched_gemm` (2026-09-10). No DIIS blowup. Does not cut AT `δ_CH`. See §3.1.

---

## 3. Measured floors (RTX 3090, mio-1-1, 2026-09-10)

All GPU vs CPU f64 on the **same** physics (after B1/B5).

| Quantity | H2O (N=6) | AT (N=87) | What it means |
|----------|-----------|-----------|----------------|
| `max\|H0\|` | ~1 Ha | **0.88 Ha** | f32 ulp ~1e-7 × scale; **not** 100×1e-7 |
| `max\|dH\|` assembly | 4e-8 | 9e-8 … 3e-7 | Assembly is **not** the floor |
| SCC charge rms | 1e-7 (converges) | **plateau 7e-6** | F1 |
| `|dq|` vs CPU | 4e-7 | 1.2e-5 | F4 |
| `|dE|` `Tr(D H0)` path | — | **7.1e-5** | F2 — discarded |
| `|dE|` band path (production) | **~5e-7** | **~2.8e-5** | F3 — tracks `δ_CH`, not frozen `δε` |
| frozen `max\|δε_occ\|` vs host GEVP | **1.5e-7** | **1.1e-6** | **not** the `|dE|` floor (was mis-cited as 2.5e-5 from SCC-then-compare) |
| `δ_CH` (`E_band−2ΣCᵀHC`) | **3e-8** | **~5e-5** | F3 source after Package 2 |
| `max\|D_gpu−D_cpu\|` | — | **6.7e-6** | follows C |
| Force TOTAL rel vs CPU | 2e-5 | **4.6e-5** | after B5; CPU-fed P/W |
| Gamma-force rel | 3e-8 | **4e-7** | after B5 |
| CPU F vs FD of E (h=1e-3 Å) | rel 1.05e-5 | — | analytic F is the gradient |

Conversion: `1e-5 Ha ≈ 0.27 meV`, `2.6e-5 Ha ≈ 0.71 meV`, `1e-4 Ha ≈ 2.7 meV`.

Tautomer / H-bond **barriers** are typically tens of meV. Absolute E vs CPU at ~0.8 meV is a **floor**, not a screening-killer, **if** relative PES cancels. Formic dimer z-scan (Package 2 + Löwdin Newton): `|ΔE_gpu−ΔE_cpu|` up to **4.6e-6 Ha** (was 1.5e-5 before Newton). Absolute bias does **not** fully cancel.

### 3.1 Package 2 → Löwdin Newton → f32 GEMM Kahan (2026-09-10, RTX 3090 `--release`)

Script: `rust_dftb/scripts/test_gpu_dftb_measure.rhai`. Same GPU-rounded `H_scc`/`S` for frozen-H. Production object `GpuDftb`.

| | H2O N=6 | AT N=87 | formic N=28 |
|--|---------|---------|-------------|
| **Package 2** (DIIS hist `min(10,n_atoms)`) |  |  |  |
| `\|\|CᵀSC−I\|\|_max` | 3.5e-7 | **5.04e-6** | — |
| frozen `max\|δε_occ\|` | 2.1e-7 | 1.35e-6 | — |
| `δ_CH` / `δ_D` | 1.8e-7 / 3e-8 | **4.56e-5 / 5.9e-7** | — |
| `|dE_GPU−CPU|` | 4.0e-7 | 3.25e-5 | z-scan `|ΔΔE|` up to **1.54e-5** |
| GPU DIIS | hist=3, **6–8 it** | hist=10, **stall 25** rms~1.7e-6 | fallbacks n=8–10 |
| **+ Löwdin Newton** `X←X(I−E/2)` if `e1<e0` |  |  |  |
| `\|\|XᵀSX−I\|\|` before→after write | 3.0e-7 → 2.0e-8 | 2.9e-6 → 2.0e-7 | 1.5e-6 → 1.0e-7 |
| after SCC `\|\|XᵀSX−I\|\|` / `\|\|C'ᵀC'−I\|\|` | 9.9e-8 / 2.4e-7 | **1.9e-7 / 1.9e-6** | — |
| `\|\|CᵀSC−I\|\|` | 2.4e-7 | **2.09e-6** (2.4×) | — |
| `δ_CH` / `|dE|` | 2.8e-8 / 5.4e-7 | 4.71e-5 / 3.36e-5 | `|ΔΔE|` max **4.6e-6** (~3×) |
| `max\|F_gpu−F_cpu\|` | 1.38e-6 | 5.24e-6 | — |
| **+ f32 Kahan in `batched_gemm`** (across K tiles; **not** f64 GEMM) |  |  |  |
| H2O / formic | **bit-identical** to Newton-only | — | **bit-identical** |
| AT `δ_CH` / `|dE|` | — | **5.03e-5 / 2.79e-5** | — |
| AT `\|\|C'ᵀC'−I\|\|` / `δε_occ` | — | 2.28e-6 / 1.07e-6 | — |
| AT `max\|F\|` diff / mixer | — | 4.44e-6 / rms 1.30e-6 stall 25 | — |

**Keep Newton:** `X` is no longer the metric limiter; formic relative PES improved; H2O not worse; skip writeback if `e1≥e0`.

**Keep f32 Kahan:** no DIIS blowup (the f64-GEMM failure mode). Does **not** cut `δ_CH`. AT `|dE|` vs CPU moved 3.36→2.79e-5 by cancellation, not by fixing band vs `CᵀHC`. Next lever is Jacobi `C'`, not more GEMM summation.

`gpu_scc_bench.rs` (`--release`, NVIDIA, `RUST_DFTB_TIMING=1`), **legacy** `gpu_solve_scc_batched_diis_warmstart` (not `GpuDftb`):

| | batch=1 | batch=100 |
|--|---------|-----------|
| formic N=28 per-iter | **0.96 ms** (jacobi 0.10, gemm 0.18) | **1.48 ms** (jacobi 0.14, gemm 0.27) |
| AT N=87 per-iter | **389 ms** of which occ_sort+upload **388 ms**; jacobi 0.10, gemm 0.12 | **788 ms** / occ_sort **786 ms**; jacobi 0.11, gemm 0.12 |

AT wall time in that bench is **host occupation sort**, not Kahan. Do not quote 389 ms as GPU SCC. Production path is `GpuDftb`.

---

## 4. Why `Tr(D H0)` lost and the band form did not cheat

CPU: `E = Tr(D H0) + ½ Δq·V`.

Algebra: `Tr(D H_scc) = 2 Σ_{occ} ε` and `H_scc = H0 + ΔH(V)`, `Tr(D ΔH) = q·V`, `q = Δq+q0`, so

```
E = 2 Σ_{occ} ε − ½ Δq·V − q0·V
```

CPU identity: **5e-10** on AT. This is the same physics, not a retune.

On GPU, `Tr(D H0)` sums ~N² ~ 7500 products of a noisy D. Host f64 of that D already differs from CPU by 7e-5 **because D differs**, not because the reduction is Kahan-hungry (kernel vs host f64 of GPU D: **1e-6**).

Band form uses ε (Jacobi diagonal, partly f64 rotations). It removes the extra projector noise. Remaining AT `|dE|~3e-5` is **`δ_CH`**: occupied `ε` vs the same `C` on `H` (`CᵀHC ≠ ε`), not frozen `δε_occ` (~1e-6) and not Kahan on `Tr(DH)` (`δ_D~1e-7`). After repairing `X`, that leftover matches `||C'ᵀC'−I||`.

---

## 5. Test contract (truthful, not fake-green, not impossible)

File: `tests/gpu_hbond_physics.rs`.

| Gate | Assert (bug if red) | Print-only / floor | Must not do |
|------|---------------------|--------------------|-------------|
| G0 device/SK | NVIDIA + mio files | — | skip |
| G1 H/S | `max\|dH\|, max\|dS\| < 1e-6` | measured ~1e-7 | loosen to 1e-2 |
| G3.1 E_rep | `\|dE\| < 1e-5` | — | — |
| G3.2 forces | rel `< 1e-4` all four + total; Newton `<1e-6` | AT/GC TOTAL ~4e-5 | restore B5 f32 γ' |
| G3.3 energy-grad | CPU F vs FD h=1e-3 rel `<1e-3`; GPU F vs CPU F | GPU FD at h=1e-3 | assert GPU F vs GPU FD at 1e-3 |
| G3.4 SCC H2O | `|dE|<1e-5`, `|dq|<1e-4`, `rms<1e-6` | measured 3e-7 | — |
| G3.4 SCC AT/GC | `|dE|<1e-4` (**regression line**, measured `|dE|~3e-5`), `|dq|<1e-4`, `rms<1e-4` + plateau | **FLOOR line:** `|dE|`, `δ_CH`, `||C'ᵀC'−I||`, rms | require `|dE|<1e-5` or `rms<1e-6` |

`1e-4` Ha on N~90 is the **divergence / regression** line, not a claim of f64 parity. A jump to 1e-3 is a bug. Sitting at `~3e-5` is `δ_CH` / Jacobi `C'`, not a license to require `|dE|<1e-5`.

---

## 6. Handoff — what compensation already did, what is left

Measured 2026-09-10 on NVIDIA 3090 `--release`. Do not re-run f64 `batched_gemm`. Do not chase AT `|dE|<1e-5`.

### 6.1 Where compensation actually mattered

| Target | Result | Trap |
|--------|--------|------|
| Occupied ε vs host GEVP N>64 | Frozen `max\|δε_occ\|=1.1e-6` — **not** F3 | Old `max\|δε\|=2.5e-5` was SCC-then-compare |
| `X=S^{-1/2}` Löwdin Newton once/geometry | **Kept.** AT `\|\|XᵀSX−I\|\|` 2.9e-6→2.0e-7; `\|\|CᵀSC−I\|\|` 5e-6→2e-6; formic `|ΔΔE|` 1.5e-5→4.6e-6. `δ_CH` unchanged | Skip writeback if `e1≥e0` |
| `H'=XHX` / `C=XC'` f32 Kahan GEMM | **Kept, weak.** H2O/formic bit-identical. AT `δ_CH` 4.7e-5→5.0e-5 (not better). `|dE|` 3.36→2.79e-5 is cancellation | **f64 `batched_gemm` reverted** — DIIS no plateau |
| `D=2C_occ C_occᵀ` | `δ_D~1e-7`, `\|\|D−2CCᵀ\|\|_F~1e-6` | Don't start here |
| `Tr(D H0)` Frobenius | Production energy **does not use this** | Dead end for F3 |
| Jacobi `C'` / `CᵀHC≠ε` | **Open.** After Newton, `\|\|C'ᵀC'−I\|\|~2e-6` matches leftover metric. Surface residual + stop reason still `[ ]` | More sweeps already stall |

Kahan helps **long same-sign accumulators**. It is the wrong tool for **catastrophic cancellation** (B5) and was the wrong first tool for F3 (`δ_CH` is `C'` vs `ε`, not a 87-term GEMM sum).

### 6.2 Hybrid architecture

Keep hot Jacobi/GEMM in f32. Newton on `X` is a host f64 island once per geometry (amortized). Next: Jacobi residual / occupied-subspace repair only if `\|\|HC−SCε\|\|` on frozen H still lags `C'`.

### 6.3 Relative vs absolute

Measured: formic z-scan `|ΔE_gpu−ΔE_cpu|` max **4.6e-6 Ha** after Newton (was 1.5e-5). Absolute AT `|dE|~3e-5` does **not** imply the same relative error. Origin recenter still `[ ]`.

---

## 7. Open method issues (not arithmetic)

- Extra-control B-spline **fitter** — extra spline controls off the ends of the SK table must be *solved for* (match the table; `V,V'→0` at cutoff). Today we glue on zeros instead. Not an f32 problem. `sk_interpolation.md`.
- **`GpuDftb` exists** (`qmqm/gpu_dftb.rs`). W on GPU (same density kernel as D). `set_coords` refills pairs in place. Drive production and benches through it. FIRE |F| vs CPU on this path still open. Manifest §0.4.

---

## 8. Dense H-bond: what is done vs what to hand off

| Layer | Status | Next |
|-------|--------|------|
| Known **bugs** (Neville tail, 1×4 `vload2`, replica cap, f32 γ' on N–H) | Fixed. Do not regress. | — |
| Honest tests (`gpu_hbond_physics.rs`) | Two-tier energy; SCC cap 100 + stall at 25. | Keep red only for real bugs |
| Package 2 frozen-H / mixer A/B / formic ΔE | Measured. F3 is `δ_CH`. | — |
| Löwdin Newton on `X` | **In.** Skip if no improvement. | — |
| f32 Kahan `batched_gemm` | **In.** Does not cut `δ_CH`. | Do not re-enable f64 GEMM |
| Jacobi `C'` / AT SCC stall | Open. | Residual + stop reason; mixer |
| Interpolator extra-control **fitter** | Stopgap (zero samples). | Separate method task |
| Analytic **forces** | Kernels exist; `GpuDftb` F vs CPU ~1e-6. | FIRE formula still wrong (`|v|`) |
| Relative PES | Formic z-scan measured. Origin recenter `[ ]`. | — |

Do **not** still chase: AT `|dE|<1e-5`, rms `<1e-6`, GPU F vs GPU FD at h=1e-3 Å, restoring Neville, re-enabling f64 `batched_gemm`.

---

## 9. Analytic forces — implemented, not a robust production path

Physics of the four terms is real (not FD of energy on the GPU):

| Term | Kernel | Fed with | vs CPU (after γ' f64 island) |
|------|--------|----------|------------------------------|
| non-SCC electronic | `force_pairs` (`gpu_forces.cl`) | P and W | H2O TOTAL rel `~2e-5`; AT/GC `~4e-5` |
| SCC shift | `force_shift_*` | P, V | in that total |
| γ' Pulay/charge | `force_gamma_deriv_batched` | Δq | AT/GC gamma rel `4e-7` (was 1% in f32) |
| repulsive | `force_rep_*` | splines | H2O E_rep `|dE|<1e-5` |

CPU analytic F vs FD of E (h=1e-3 Å, H2O): rel `1.05e-5`. Newton 3rd-law checks pass.

**Not robust yet — do not treat four-component tests as FIRE-ready:**

1. **AT/GC tests CPU-feed P and W.** They prove the *force kernels*, not “GPU SCC density → GPU force.” H2O full-chain uses GPU P then **host-built W**. There is no device W kernel (`W_μν = 2 Σ_k ε_k C_μk C_νk`).
2. **`GpuForceDriver` is not a persistent plan.** Every call: `Kernel::builder`, new force buffer, re-upload fragments + SK + pairs *per bucket*. Contrast `GpuSccPlan` (kernels/buffers once). A relaxation step would rebuild this every FIRE step.
3. **Interpolator stopgap** (hardcoded right-end zeros) can still pollute long-range pair forces. Interior H2O/AT numbers look fine; do not claim cutoff-pair derivatives are final until the extra-control fitter exists.
4. **No geometry loop.** No device FIRE, no constraints, no `H0/S` rebuild policy when atoms move (must reassemble + new `X=S^{-1/2}` per geometry; SCC can warm-start charges).
5. **γ value on device** is still f32 (`dftb_hamiltonian.cl::gamma_full`). Forces use γ' in f64. If G is ever rebuilt on GPU, the same U≈U' cancellation returns for **charges**, hence for shift/γ' inputs.

Contract for “forces are done”: one geometry-change step that (a) GPU-assembles H/S, (b) GPU SCC, (c) GPU P **and** W, (d) four GPU force kernels, (e) matches CPU F at the existing `FORCE_REL=1e-4`, **without** host P/W. That does not exist.

---

## 10. Why even H2O/AT “feel slow” — harness bottlenecks (not the RTX 3090)

H2O full-chain in `cargo test` is ~0.5 s wall, mostly **startup**. AT is slow because **tiled Jacobi runs every SCC iter**, plus the test process is an unoptimized host. Ranked so the next agent does not “optimize the wrong thing.”

### 10.1 Do not confuse test-process cost with kernel cost

| Cause | Where | Effect | What to do |
|-------|-------|--------|------------|
| **`cargo test` = `profile.dev`, opt-level 0** | `rust_dftb/Cargo.toml` | CPU SK load, B-spline fit, CPU f64 SCC reference, host diagnostics are debug-slow | Time **`--release`**. Never quote test-profile ms as GPU time |
| **New `GpuRuntime` + OpenCL compile per test** | `require_nvidia()` in `gpu_hbond_physics.rs` | `build_program` for Hamiltonian, Jacobi (N-specialized), tiled Jacobi, GEMM, forces. Cache is per-runtime; tests throw it away | One process-lifetime runtime. Cache already exists (`gpu_runtime.rs::program_cache`) but dies with the struct |
| **Test rebuilds the universe** | `run_full_chain_scc` | CPU `build_scc` from scratch, GPU assemble, `GpuSccPlan::new`, then **two** full N² readbacks of D and C for FLOOR prints | Keep FLOOR prints; don’t add a third. Don’t use this function as a bench |
| **`compute_energy` → `finalize`** | `gpu_scc_plan.rs` | Extra full electronic solve (Jacobi again) so D,Δq,V match mixed q | Correct (R7). Count it as +1 SCC iter in timings |
| **SCC used to run 400–500 iters** | tests, now capped | Plateau detector never fired | `SCC_MAX_ITER=100`, stall at 25. Do not put 500 back |

### 10.2 Real GPU / driver costs (production path)

`GpuSccPlan::scc_step_diis` itself is the **right** shape: persistent kernels, `set_arg` + enqueue, RMS-only readback.

| Cause | Where | Effect | What to do |
|-------|-------|--------|------------|
| **Tiled Jacobi every SCC iter, `TILED_MAX_SWEEPS=100`** | `gpu_scc_plan.rs`, `gpu_tiled_jacobi.cl` | Dominant for N=87. Kernel can stall after 3 non-improving sweeps, but this is still the O(N³) work | Bench sweeps-to-stall vs residual. Warm-start eigenvectors (optional, after floor). **This is also where F3 lives** — precision and cost are the same kernel |
| **batch=1 in H-bond tests** | `GpuSccPlan::new(..., batch=1)` | GPU occupancy for N=6 or even N=87 is poor. The design is many-small-systems | Bench formic `batch=100` (`gpu_scc_bench.rs`, `--release`). Do not tune Jacobi for batch=1 H2O |
| **`GpuForceDriver` / `GpuDriver` rebuild kernels + upload SK per call** | `gpu_forces.rs`, `gpu_driver.rs` | Fine for a one-shot test; fatal inside FIRE | Persistent force plan, same pattern as `GpuSccPlan` |
| **Legacy `gpu_scc.rs`** | still used by older tests | Comment admits `Kernel::builder` every call + host occ-mask | Prefer `GpuSccPlan`. Don’t “fix” f32 in the legacy driver |
| **`set_arg` every kernel every iter** | `scc_step_diis` | Cheap vs Jacobi | Ignore until Jacobi is timed |
| **No active mask** | plan runs all systems every iter | Only matters at large batch | Later |

### 10.3 How to time (so compensation does not hamper throughput)

1. `cargo test --release --test gpu_scc_bench -- --ignored --nocapture` (or equivalent) on NVIDIA. `OPENBLAS_NUM_THREADS=1`.
2. Split: OpenCL compile (once) / `GpuSccPlan::new` / first SCC iter (Jacobi cold) / later iters / `finalize`.
3. Quote **ms/SCC-iter at batch=1 and batch=100**, N=28 and N=87. H2O N=6 is not a GPU benchmark.
4. Any f64 island or Kahan goes in this table **before** it is declared acceptable.

---

## Related

- `/doc/prokop/topical_audit/sk_interpolation.md`
- `/doc/prokop/topical_audit/gpu_scc_pipeline.md`
- `/doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.manifest..md` §3.0.1
- `/doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` §6.4
- Tests: `rust_dftb/tests/gpu_hbond_physics.rs`
