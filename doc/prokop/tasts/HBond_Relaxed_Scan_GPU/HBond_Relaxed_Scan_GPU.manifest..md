# Task 1: GPU Multi-System Relaxed Scan of Hydrogen-Bonded Nucleobase Pairs

**Created:** 2026-09-07
**Last wrap (2026-09-11):** read **§0** first, then **§0.7 + §12** — new GPT 5.6 dense-path review of `e965ae00` is the current work order. Sparse is a **separate** agent — do not edit `rust_dftb/src/methods/sparse/`.
**Owner:** prokop / Devin
**Interpolator spec:** `doc/prokop/topical_audit/sk_interpolation.md`
**f32 floor + harness map:** `doc/prokop/topical_audit/f32_floor_dense_hbond.md`

---

## 0. State for the next LLM (2026-09-10) — read this, then the rest

### 0.1 What was resolved (do not re-open)

| Item | Outcome |
|------|---------|
| AT/GC `max\|dH\|~0.4 Ha` | **CPU SK tail bug**, not GPU. Neville + `poly5_to_zero` exploded on H–H at 10.39 Bohr. Production is B-spline. **Do not restore Neville** for Fortran-tail parity. |
| NVIDIA 1×4 crash | `__local float2*` alignment; fixed with `vload2`. |
| Replica cap | `l_frags[128]` silent truncation; fixed. |
| AT/GC gamma-force 1% off | Ill-conditioned f32 `Ua≠Ub` (mio N–H). Kernel `gamma_prime_full_f32` now **f64 island**. |
| AT SCC “mixer broken” | Mixer **does** stall (~25 it, rms~1.3e-6). Frozen-H Jacobi is **not** the `|dE|` floor: AT `max|δε_occ|~1.1e-6` vs `|dE|~2.8e-5`. AT `|dE|` tracks `δ_CH=E_band−2ΣCᵀHC` (~5e-5), not `δε_occ`. |
| Fake-green energy tests | Two-tier: H2O `|dE|<1e-5` (measured 3e-7). AT/GC `|dE|<1e-4` is a **regression line**, not f64 parity. SCC cap **100** (stall at 25). |
| f64 `batched_gemm` acc | Tried, **reverted** (worse energy, DIIS no plateau). |

### 0.2 Open problems (three different kinds — do not mix them)

**A. Accuracy (not one “f32 floor”)** — Package 1 `[*]`, Package 2 measure `[*]`, GPU DIIS hist cap `[*]` (H2O), Löwdin Newton `[*]`, f32 GEMM Kahan `[*]` (does not cut `δ_CH`). AT `|dE|` is `δ_CH` / Jacobi `C'`. **Next is §12 D1–D4** (2026-09-11 review): decompose `δ_CH` into normalization vs orthogonality vs true eigen-residual, repair occupied C′ in f32 — do **not** accept a “floor” before that measurement. Then AT stall, origin recenter. **Master checklist:** `OVERVIEW_Roadmap.md` §6.4.

**B. Interpolator fitter** — method, not f32. Plain language in §0.3. Spec: `sk_interpolation.md`.

**C. Production GPU run loop — `GpuDftb`.** Drive **this** object. Do not add throwaway `GpuRuntime::new` tests.

CPU analogue: `DftbCpu`. GPU: `rust_dftb/src/qmqm/gpu_dftb.rs`.
User/test entry: **`dftb_engine --script foo.rhai`** (see §0.5). Do not add a new `cargo test` / `src/bin` / `examples/` target per molecule.

```
GpuDftb::new(sk, sk_dir, species, coords, batch)  // INIT once
  .set_coords(...)                                 // pairs in place → H0/S, G, X
  .scc()
  .eval(want_forces)                               // one finalize; forces optional
  .fire_step() / .md_step()
```

**Status on this object (2026-09-10):**

1. **W on GPU, same kernel as D.** Done. `s_k=1` → D, `s_k=ε_k` → W. No C/ε host roundtrip.
2. **`set_coords` in-place pairs.** Done. No `GpuBatch::from_fragments` on the geometry step.
3. **`eval(want_forces)`.** One electronic finalize. Energy always; forces only if asked. Do not call `energy()` then `forces()` (that double-solves). Thin wrappers remain for old call sites.
4. **Neighbor list + G on CPU is OK.** Once per geometry, not per SCC. Same O(n_atoms²) distances — a later optional **one GPU kernel** can write pairs `(r,l,m,n)` and G together. Not a must-fix; do not rebuild SK when doing it.
5. **Homogeneous batch only.** Replicas = copies of **one** template. AT and GC are two `GpuDftb` objects (or two script steps), not one mixed launch. Different replica counts = `batch` on that template.
6. Exercise AT, GC, azaindole, H2O-replicas through the **Rhai script**, not a new Rust test file. `gpu_hbond_physics.rs` is leftover bisect.

Topology / pair overflow → fail loud, rebuild `new`. FIRE/MD cap 0.1 Å.

### 0.3 What “interpolator fitter” means

Slater–Koster files are **numbers on a 1D grid** (typically `dr=0.02` Bohr). We interpolate with a **cubic B-spline**. A cubic B-spline does not live only on those samples: it needs a few extra **control points** off each end of the table so the curve knows how to start and how to die.

- **Left end (already the right kind of thing):** a phantom control `c_{-1}=2c_0−c_1` at evaluation. SK values at small `r` are large — never force them to zero.
- **Right end (current stopgap, blunt):** glue on four **function values hardcoded to 0**, then refit. That stopped the −0.4 Ha explosion. It is **not** a boundary-condition solve: those zeros leak into the last real samples, and cutoff is only `~0.08` Bohr past the last grid (DFTB+ used a 1 Bohr polynomial fudge; we are not copying that fudge).
- **GPU dummy 0 at `r=0`:** same kind of cheat (index padding), not a fitted left control.

**The fitter** is the missing linear solve: extra controls are **unknowns**. Solve so (1) the interpolant on the **tabulated** region still matches the SK file, (2) `V` and `V'` go smoothly to 0 at a chosen cutoff. Do **not** implement this as “pad with more zeros.” Do not restore Neville.

This is independent of the f32 eigen floor. Interior H2O/AT H/S already match at `~1e-7`.

### 0.4 Production pipeline — we do **not** have it, and we **must**

A normal DFT/DFTB code does:

```
INIT (once)          load SK, compile kernels, preallocate all buffers
PER GEOMETRY         update coords → neighbors → H0,S,G,X=S^{-1/2}
SCC (warm-started)   iterate until Δq plateaus
FORCES               P,W already on device → F
MD / FIRE            move atoms → go to PER GEOMETRY
```

**CPU already follows this:** `rust_dftb/src/methods/dftb/dftb_cpu.rs` (`DftbCpu`) + FIRE in `examples/hbond_ref.rs`.

**GPU owner of that lifetime:** `rust_dftb/src/qmqm/gpu_dftb.rs` (`GpuDftb`).
**How a human (and a test) runs it:** `dftb_engine --script rust_dftb/scripts/<case>.rhai --sk-dir …` (§0.5).

| Piece | Role now |
|-------|----------|
| `GpuDftb` | **Use this.** `new` / `set_coords` / `scc` / **`eval(want_forces)`** / `fire_step` / `relax`. |
| `GpuSccPlan` | Inner SCC, owned by `GpuDftb`. Do not construct one per test. |
| `dftb_engine` + `.rhai` | **Product CLI.** New scenarios are scripts. |
| `GpuDriver` / `GpuForceDriver` / `gpu_scc.rs` | Legacy one-shot. Do not use for production or benches. |
| `gpu_hbond_physics.rs` | Physics diagnostics. Still throwaway runtimes — do not grow. |

Gaps inside `GpuDftb` (fix here, do not bypass):

| Gap | Status |
|-----|--------|
| W on host from C,ε | **Done.** Same density kernel, `s_k=ε_k`. |
| `set_coords` → `GpuBatch::from_fragments` | **Done.** In-place `refill_pairs`. |
| Separate `energy()` + `forces()` double finalize | **`eval(want_forces)`** — one solve. |
| Neighbor + G on CPU | **Accepted** (per-geometry). Optional later: one GPU kernel, same distance loop. |
| `orb_atom` not tiled over batch | **Fixed.** Kernels index `[batch][N]`; one copy made replica>0 read garbage (H2O batch=4 \|dE\|~0.20 Ha). |
| AT / GC / azaindole / replica counts on this object | Via `scripts/test_gpu_dftb_molecules.rhai`, not a new `tests/*.rs`. |
| `gpu_hbond_physics.rs` throwaway runtimes | Diagnostics only. |
| FIRE \|F\| vs CPU on this object | Open. |

**Production and all timing** (`--release`, NVIDIA) go through `GpuDftb` driven by `dftb_engine`.

### 0.5 HARD RULE — one engine, scripts as tests (stop new binaries)

The user-facing program is **`dftb_engine`**. A calculation is an input script (`.rhai` now; `.ron` later if we want data-only input). That is also how we test. A test that cannot be a script means the CLI is missing a function — **add the function to `dftb_engine`, do not add a binary.**

**Do not** create any of these for a new molecule, scan, FIRE run, or replica count:

- `rust_dftb/tests/<name>.rs` (each file is a separate cargo test crate / compile target)
- `rust_dftb/src/bin/<name>.rs`
- `rust_dftb/examples/<name>.rs`

**Do** this instead:

```
cargo run --release --bin dftb_engine -- --script rust_dftb/scripts/<case>.rhai --sk-dir $RUST_DFTB_SK_DIR
```

Existing cargo tests (`parity_*.rs`, `gpu_hbond_physics.rs`, `gpu_dftb.rs` H2O smoke) stay as physics bisects / compile checks. **Do not grow them.** Do not add `gpu_dftb_at.rs`. New AT/GC/azaindole/replica work is a `.rhai` file.

`GpuDftb` is homogeneous: one species template, `batch` identical replicas in one launch. Different molecules = sequential `gpu_new` in the same script (two engines), not AT+GC packed together.

### 0.6 Astra review (2026-09-10) — implement this order

Source: `doc/prokop/reports/2026-09-10_gpu_accuracy_physics_performance_second_review.md`.
**Dense only here.** Do not edit `rust_dftb/src/methods/sparse/`. Keep H/S/D/C/W in **f32**. Drive via `dftb_engine` + `.rhai`, not a new cargo test. f64 `batched_gemm` stays reverted. f32 Kahan on GEMM is in; it did not remove `δ_CH`.

The review is right on the diagnosis: several different errors were lumped into “f32 floor” (mixer, incomplete SCC, stale diagnostics, output rounding, Jacobi stop). It is **stale** on pipeline items we already closed after the inspect (W on GPU, in-place pairs, `eval(want_forces)`, `orb_atom` tiled over batch).

**Do not** convert matrices to f64. **Do** keep γ' in f64 until a stable rewrite exists.

#### Already done (do not re-open)

- [*] W on GPU, same density kernel as D (`s_k = 1` or `ε_k`)
- [*] `set_coords` in-place pair refill (no `GpuBatch::from_fragments` / SK re-pack)
- [*] `eval(want_forces)` — one finalize, not `energy()` then `forces()`
- [*] `orb_atom` length `[batch][N]` (replica>0 was OOB)
- [*] Neighbor list + G on CPU accepted (per-geometry)
- [*] AT/GC/azaindole + replica counts via `scripts/test_gpu_dftb_molecules.rhai`
- [*] γ' force kernel is an f64 island (`gamma_prime_full_f32`)
- [ ] f64 `batched_gemm` — tried, **reverted**; do not re-enable without frozen-H evidence

#### Package 1 — contracts + cheap f64 (do first)

Honest numbers, then the two cheap precision islands. Matrices stay f32.

NVIDIA RTX 3090 `--release`, `scripts/test_gpu_dftb_molecules.rhai`, 2026-09-10. Mixer RMS is now `√(Σres²/n_atoms)` (was L2). Do not retune physics to match old L2.

| case | mixer rms | iters | stall | E (f64 Ha) | q_rms (finalize) | q_max | max\|F\| |
|------|-----------|-------|-------|------------|------------------|-------|----------|
| H2O×1 | 8.881e-7 | 19 | no | −4.076143626124 | 1.177e-6 | 1.907e-6 | 6.85e-2 |
| H2O×4 | 8.881e-7 | 19 | no | bit-identical replicas | 1.177e-6 | 1.907e-6 | 6.85e-2 |
| AT×1/×2 | 1.680e-6 | 25 | yes | −44.626055076718 | 9.435e-7 | 1.907e-6 | 8.49e-2 |
| GC | 1.436e-6 | 25 | yes | −44.921077474952 | 1.572e-6 | 5.245e-6 | 7.64e-2 |
| azaindole | 6.203e-7 | 12 | no | −38.090388506651 | 6.445e-7 | 1.431e-6 | 2.83e-2 |

H2O DIIS: GPU hist is now `min(10,n_atoms)` (H2O=3 → 6 SCC iters, one leftover pivot fallback). Cap-history on the kernel is done; AT stall is not (hist already 10).

- [*] **f64 energy out** — `eval` / `energy_from_state` keep the total in f64 (band sum already f64; do not cast to f32 before adding E_rep or returning). Print Ha in f64 from `dftb_engine`.
- [*] **Finalize proves charges** — after D, Mulliken → `q_D`. Print `max|q_D−q_in|` and true RMS (`/√n_atoms` vs current L2). Do not call the state “converged” from mixer rms alone.
- [*] **Stale `E_h0` diagnostic** — `plan.tr` is `q0·V`, not `Tr(D·H0)`. Do not grow `gpu_hbond_physics.rs`. New prints live in `gpu_eval`.
- [*] **DIIS, same kernel** — (1) first two history points: α-mix, not 1-vector DIIS. (2) residual **RMS** = `√(Σres²/n_atoms)` so `tol` matches CPU. (3) Gram + tiny GE in **f64**, scale Gram before the 1-constraint. (4) small pivot / non-finite / `|Σc−1|>tol` → explicit simple-mix + print, **never** silent `c_i=0`. (5) host max(rms) fails on NaN.
- [*] **`set_coords` resets DIIS** — charges may warm-start; residual history must not.
- [*] **Overlap λ_min** — after `X=S^{-1/2}`, fail loud if λ≤0 or λ too small (today `rsqrt(max(λ,1e-7))` hides it).
- [*] **No `U=0.4` fallback** — missing onsite Hubbard fails with species.
- [*] **Occupation** — reject odd/charged systems; do not round `Σq0/2` into a different molecule.

#### Package 2 — mixer experiment, then arithmetic (only after Package 1 prints exist)

Measured 2026-09-10 NVIDIA 3090 `--release`, `scripts/test_gpu_dftb_measure.rhai`.

- [*] Frozen-H: GPU-rounded `H_scc`/`S`, host Löwdin+GEVP vs GPU Jacobi. H2O `max|δε_occ|=2.09e-7` `||HC−SCε||=2.78e-7`; AT `max|δε_occ|=1.35e-6` `||HC−SCε||_occ=1.02e-6`. Not “SCC then compare ε”.
- [*] Mixer A/B on unchanged kernels (H2O): GPU DIIS **hist=3, 6 it** `|dE_CPU|=4.0e-7`; GPU simple 28 it; host f64 DIIS 17 it. AT GPU DIIS still stalls 25 it (hist=10).
- [*] H2O **forces vs CPU** on `GpuDftb` (full SCC→F): `max|F_gpu−F_cpu|=1.53e-6` (`max|F|=6.85e-2`).
- [*] Relative `ΔE` formic dimer z-scan: `|ΔE_gpu−ΔE_cpu|` up to **1.54e-5 Ha** — absolute bias does not reliably cancel.
- [*] AT `δ_eig=4.62e-5` splits as `δ_CH=4.56e-5` + `δ_D=5.9e-7` (`||D−2CCᵀ||_F=1.2e-6`). Not `Tr(DH)` Kahan. `rms(q_D−q_cpu)=6.1e-6`.
- [*] **Löwdin Newton** once per geometry (`gpu_scc_plan.rs::repair_lowdin_x`): `M=XᵀSX=I+E` → `X←X(I−E/2)`, skip if `e1≥e0`. AT `||XᵀSX−I||` 2.9e-6→2.0e-7; `||CᵀSC−I||` 5.0e-6→2.1e-6. Formic `|ΔΔE|` 1.54e-5→**4.6e-6**. `δ_CH` **not** reduced (leftover is `||C'ᵀC'−I||~2e-6`).
- [*] **f32 Kahan** in `batched_gemm` (compensation across K; not f64 GEMM). H2O/formic **bit-identical** to Newton-only. AT `δ_CH` 4.71e-5→5.03e-5 (not better); `|dE|` 3.36e-5→2.79e-5 (cancellation); still stall 25. No DIIS blowup.

Numbers after Newton+Kahan (NVIDIA 3090 `--release`): H2O `|dE|=5.4e-7` 8 it; AT `|dE|=2.79e-5` `δ_CH=5.03e-5` `max|δε_occ|=1.07e-6` `max|F_diff|=4.4e-6`. SSOT table: `f32_floor_dense_hbond.md` §3.1.

#### Package 3 — engine honesty (not “f32”)

- [ ] Jacobi: surface residual / stop reason; do not treat a zeroed off-diagonal as a solved eigenproblem.
- [ ] FIRE: Bitzek mix uses `|v|`, not `|F|`; per-replica `dt/α` (today all replicas share one FIRE state).
- [ ] `md_step` is not velocity-Verlet (missing half-kick). Rename or fix; do not document as MD.
- [ ] `relax` returns **final** rms, not the first SCC’s; do not ignore `stalled`.
- [ ] Optional: fuse neighbor+G in one GPU kernel (same O(n²) distances).
- [ ] Profile tiled Jacobi f64 inner sweeps before touching them (already a large f64 island).

#### Sparse (other agent — listed so we do not “fix f32” there by accident)

- [ ] `run_scc` is not an SCC loop; `purify_h` rebuilds Z every iter
- [ ] Host dense W from K; TC2 idempotency without trace
- [ ] NS `||ZS−I||` reverification
- [ ] Gate G / Hessian: stopping error vs arithmetic; fixture is not tetrahedral

#### Explicitly out of this pass

Interpolator extra-control **fitter** (`sk_interpolation.md`). On-device γ value rewrite. All-f64 matrices. New `tests/*.rs`.


### 0.7 GPT 5.6 dense-path review (2026-09-11, `e965ae00`) — current work order

Source: `HBond_Relaxed_Scan_GPU.chat.md` lines 2355–2555. Full item list + FP32 policy in **§12**.

**Verdict:** the earlier P0 correctness failures are genuinely fixed (similarity-preserving tiled Jacobi, global fences, strict Jacobi tests, persistent `GpuSccPlan`, GPU occupation+DIIS, finalize consistency, W on GPU, repulsive E+F). Two meta-findings drive the next pass:

1. **The AT `~3e-5` Ha error is NOT a proven f32 floor.** Evidence points to accumulated loss of eigenvector orthonormality in tiled Jacobi (`||C'ᵀC'−I||~2e-6` while frozen `δε_occ~1e-6` and assembly `~1e-7`). Repairable cheaply in f32 — do not accept "f32 floor" wording until §12 D1–D4 are measured.
2. **The code overreacted the other way:** broad FP64 inside the O(N³) Jacobi path (rotation params, 2×2 block updates, strip dots — ~1.29M double block-updates per pivot) and a CPU f64 serial-GEMM `repair_lowdin_x` per replica. On a 3090 that is a throughput disaster. Accuracy must come from better f32 algorithms, not f64 hot loops.

**Order (do not reorder — each step is cheap and may make the next unnecessary):**

| # | Work | §12 |
|---|------|-----|
| 1 | Decompose `δ_CH`: per-column `cᵀSc`, Rayleigh quotients, residuals — normalization vs orthogonality vs true eigen error | D1 |
| 2 | Remove broad FP64 from Jacobi updates → FP32 FMA + 4-accumulator dots; benchmark 3 modes (events + residual + AT ΔE); A/B `batched_gemm` Kahan the same way | D2 |
| 3 | Occupied-column renormalization `C'←C'/‖C'‖`, then optionally one polar step `C_o'←C_o'(3I−G)/2` — first in `finalize()` only | D3 |
| 4 | Keep E/W consistent with any occupied-subspace mixing: `H_o=C_oᵀH_sccC_o`, `E_band=2Tr(H_o)`, `W=2C_oH_oC_oᵀ` | D4 |
| 5 | `repair_lowdin_x` off CPU → same metric repair as 3 GPU GEMMs; then reuse X across FIRE steps instead of re-diagonalizing S | D5, D6 |
| 6 | Device-resident geometry: per-template pair lists once; kernels compute `ΔR,r,R̂` from GPU coords; kill pair rebuild+upload + assembly `finish()` calls | D7 |
| 7 | Pretabulated γ/γ′ species-pair spline (f64 fit at init → f32 eval, γ and γ′ from the *same* spline) | D8 |
| 8 | Fix FIRE physics first (`α‖v‖F̂` not `α‖F‖F̂`; per-replica `dt/α/n_pos`; one standard ordering) — then move on GPU | D12 |
| 9 | Engine honesty: per-system `active` mask; `Converged`/`AcceptedAtNumericalPlateau`/`Failed`; DIIS anchored Δq-mix, no `printf`; rewrite `gpu_scc_bench.rs` (benchmarks legacy `GpuDriver`); add native `gpu_bench()` | D9–D11 |

**Working discipline (the failure mode this task keeps hitting):**

- **Diagnose before declaring a floor.** `δ_CH` was labeled "f32 floor" for weeks; the review shows it is probably unnormalized/nonorthogonal C′ — an algorithmic bug, not a precision wall. Every "floor"/"limit" claim must be backed by a measured decomposition, else it is a hypothesis.
- **Do not buy accuracy with f64 in hot loops.** FP64 belongs to scalar islands (energy sums, DIIS Gram, γ-spline *fitting*). Inside O(N³) Jacobi/GEMM the currency is algorithm quality: renormalization, polar correction, multi-accumulator FMA. See the precision table in §12.
- **A/B with real measurements.** Any arithmetic change reports OpenCL event time + residual/orthogonality + AT ΔE — not just "tests pass". Benchmark only the production path (`GpuDftb` via `dftb_engine`), never legacy drivers.
- **Never loosen a test to green; never let "stalled" read as "converged".** A red test localizes broken physics. If the solver misses its contract, fix the solver — or document the measured reason and keep the contract visible.
- **Patience over shortcuts.** Do the cheap correct thing first (D1–D3 are ~hours of work and may remove the entire "floor"), verify on real molecules (AT/GC/azaindole, not 12×H2O), and record numbers in this manifest — not impressions.


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

#### 3.0.1 AT/GC GPU SCC rms `~1e-5` — measured floor of the *current* FP32 Jacobi representation (not a proven f32 floor)

> **2026-09-11 (§12 D1–D4):** the `δ_CH` label is a hypothesis, not a verdict.
> Evidence points to `||C'ᵀC'−I||~2e-6` orthonormality loss in tiled Jacobi —
> repairable in f32. Decompose `δ_CH` into normalization/orthogonality/true
> eigen-residual **before** accepting any floor claim.

Observation: GPU SCC on AT/GC (N=87/86) plateaus charge-rms `~7e-6`. CPU f64 on the **same** H/S goes to `~1e-9` in ~20 steps. H/S GPU vs CPU matches (`max|dH|~3e-7`). Occupation is correct (49/49), HOMO–LUMO gap `0.124` Ha both sides.

**Measured scale (AT, mio-1-1):** `max|H0|=0.88` Ha (not ~100). f32 ε×|H| `~1e-7`. Charge rms `~7e-6` is consistent with that.

**Energy split (AT), not a mixer bug:**
- `½Δq·V` GPU vs CPU `2.9e-6` (fine).
- `Tr(D·H0)` was `7.1e-5` off — `max|D_gpu−D_cpu|=6.7e-6`. Host f64 `Tr` of the same D recovers the GPU kernel to `1e-6` (reduction is not the 7e-5). f64-D from GPU C does not help: the error is in **C / ε**, not the density contraction.
- Band identity `E = 2Σ_{occ}ε − ½Δq·V − q0·V` holds on CPU to `5e-10`. `GpuSccPlan::compute_energy` now uses it. AT `|dE|` drops `6.8e-5 → 2.6e-5`. H2O stays `3e-7`.
- Remaining `|dE|~3e-5` Ha is **`δ_CH`** (`E_band−2ΣCᵀHC`), not frozen `max|δε_occ|~1e-6` and not `max|δε|=2.5e-5` from SCC-then-compare. See §0 Package 2 and `f32_floor_dense_hbond.md` §3.1. f64 acc in `batched_gemm` was tried and **reverted**. f32 Kahan on GEMM is in and does **not** cut `δ_CH`.

**Accepted as the N~90 `δ_CH` floor until Jacobi `C'` is repaired.** Honest test contract (`gpu_hbond_physics.rs` G3.4): H2O `|dE|<1e-5` (measured ~5e-7); AT/GC `|dE|<1e-4` is a **regression / bug line**, not f64 parity. Print `FLOOR`. Do **not** chase rms `<1e-6`. Do not require AT `|dE|<1e-5`. Kahan on `Tr(D·H0)` will **not** move F3. f32 GEMM Kahan did **not** move `δ_CH`. See `f32_floor_dense_hbond.md` §3.1.

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
- [~] AT/GC/azaindole SCC: see **§0 Package 2** (not this stale occupied-ε line). Frozen `max|δε_occ|~1e-6`; `|dE|` is `δ_CH`. G3.4 AT/GC asserts `|dE|<1e-4` (regression), not `<1e-5`. SSOT `f32_floor_dense_hbond.md` §3.1.
- [ ] Fixed-geometry relative PES parity tests.

Optional only after the baseline works:

- [ ] benchmark eigenbasis warm-start; keep only if total SCC time improves.

### Phase 4 — Make the force path genuinely analytic, then port it

- [~] CPU analytic H/S derivatives vs FD of the same B-spline — pass (`gpu_hbond_physics.rs`, H–H rel `8.6e-10`). Remaining: replace blunt zero-sample pad with fitted extra controls (`sk_interpolation.md`).
- [~] CPU total analytic forces vs FD of energy — H2O rel `1.05e-5`. Fortran force parity (`parity_forces.rs`) is older; re-check after the extra-control fitter, not by restoring Neville.
- [~] Port to OpenCL — H2O/AT/GC four components vs CPU (CPU-fed P/W). H2O TOTAL rel `2.1e-5`. AT/GC gamma' was 1% off: OpenCL f32 `Ua≠Ub` S' cancels two O(10³) terms (mio N–H ΔU≈0.011); kernel now evaluates γ' in f64 (`gpu_forces.cl::gamma_prime_full_f32`). After that: AT/GC gamma rel `4e-7`, TOTAL rel `~4e-5`. Same f32 VALUE cancellation still lives in `dftb_hamiltonian.cl::gamma_full` (on-device γ); the AT SCC test uploads host f64→f32 γ so that is not the |dE| path.
- [~] Pairwise on-the-fly P/W contraction — H2O full-chain GPU P then forces rel `3.4e-5`.
- [~] GPU-vs-CPU force parity + Newton — H2O/AT/GC four-component tests pass (`test_force_four_components_*`). AT/GC **GPU SCC** plateaus rms `~7e-6`, `|dE|~2.6e-5` (eigen floor, §3.0.1 / `f32_floor_dense_hbond.md`). Four-component forces (CPU-fed P/W) are not full-chain FIRE-ready.

### Phase 5 — Production GPU lifetime + FIRE (must-build)

**The run loop is `GpuDftb`.** CPU already has `DftbCpu`. GPU: `qmqm/gpu_dftb.rs`. See §0.4.

- [x] **`GpuDftb`** — one persistent object. Drive via `dftb_engine` + `.rhai` (§0.5). `tests/gpu_dftb.rs` is H2O smoke only — do not add more cargo tests.
- [x] **W on device** — `build_density_masked_batched` per-MO scale (`1` → D, `ε_k` → W). Same kernel.
- [x] **`set_coords` in-place pairs** — no `GpuBatch::from_fragments`. Fill `pair_staging`. Fail loud if `n_pairs > pair_cap` or a new (block, species) slot.
- [x] **`eval(want_forces)`** — one finalize; energy always, forces optional. `energy()`/`forces()` are wrappers.
- [ ] **Optional:** fuse neighbor list + G into one GPU kernel (same O(n²) distances). CPU is fine until profiled.
- [~] AT / GC / azaindole / replica counts — `scripts/test_gpu_dftb_molecules.rhai` (not a new `tests/*.rs`).
- [~] Four force kernels + repulsive already on `GpuDftb`. After W-on-GPU: F from device P,W.
- [~] GPU FIRE / MD — exists; 0.1 Å cap. Not done until \|F\| vs CPU on this object.
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

> **Status reconciled against code 2026-09-11** (new review of `e965ae00`, §12).
> R1–R11 verified in code; remaining open items are superseded/refined by §12
> (D2 covers R16-class accumulation, D7 covers R13, D10 covers R18, D11 covers
> R4b/N>64-SCC tolerance and convergence semantics).

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
      N>64 SCC tolerances remain pending (R4b — still `1e-2` at
      `gpu_scc.rs:513`; tighten under §12 D11 once D1–D4 land).
- [*] **R5. SCC energy must include repulsive spline.**
      `gpu_scc_plan.rs::compute_energy` returns only `Tr(D·H0) + 0.5·Δq·V`.
      The DFTB total energy is `E_DFTB = E_el + E_rep`. Without `E_rep` the
      proton-transfer PES is quantitatively/qualitatively wrong while CPU/GPU
      parity looks perfect. Add repulsive spline evaluation to the GPU energy.
      **FIXED:** `repulsive_energy_batched` kernel (exp head + cubic intervals
      + polynomial tail) + `GpuSccPlan::set_repulsive_splines`, wired into
      `GpuDftb::new` (real SK spline data uploaded at construction).
      `compute_energy`/`energy_from_state` add E_rep. Tests:
      `test_e_rep_kernel_h2o`, `test_e_rep_in_scc_energy_h2o`
      (`gpu_hbond_physics.rs`) — GPU E_rep vs CPU spline and E_tot=E_el+E_rep.
- [*] **R6. Force kernel: port all four force components, not just non-SCC.**
      `gpu_forces.cl` computes only `P·dH0 - W·dS`. Production forces must
      match the CPU decomposition: (a) non-SCC electronic, (b) SCC shift
      `0.5·(V_A+V_B)·P·dS`, (c) gamma derivative `Δq_A·Δq_B·γ'(R)`, (d)
      repulsive spline derivative. Port the exact tested CPU formulas — do
      not rederive in OpenCL.
      **FIXED:** All four kernels implemented and wired into `GpuDftb`
      production `forces_from_state` (`k_force`, `k_shift` per bucket +
      `k_gamma_f`, `k_rep_force`). γ′ uses f64 island `gamma_prime_full_f32`
      (mio close-U cancellation). Parity tests:
      `test_force_four_components_{h2o,at,gc}`, repulsive-force parity,
      `test_gpu_energy_gradient_h2o`. Note: γ′ f64 island is a stopgap —
      §12 D8 replaces it with a pretabulated species-pair spline.
- [*] **R7. SCC consistency: energy/forces must use the same charge state.**
      `scc_step` builds H/C/D from `q_n`, then overwrites `q_gpu` with mixed
      `q_{n+1}`. If residual passes, `compute_energy` uses D from `q_n` but
      recomputes Δq/V from `q_{n+1}`. For analytic forces this inconsistency
      destroys energy-gradient parity. Fix: either accept state as `q_n`
      consistently, or do one final unmixed electronic solve after convergence
      so C/P/W/E/forces all correspond to the same q.
      **FIXED:** `GpuSccPlan::finalize()` does one unmixed electronic solve at
      current `q_gpu`; `GpuDftb::eval` = `finalize` → `energy_from_state` →
      `forces_from_state`, so E and F share one stationary charge state
      (`gpu_dftb.rs:583`). `charge_residual()` prints `q_rms`/`q_max` of
      `q_D−q_in` for diagnostics.

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
- [*] **R9. Move occupation selection and DIIS fully onto GPU.**
      Eigenvalue extraction does blocking read, CPU sort, upload occ_mask
      every iteration. DIIS reads `q_new`, allocates Vecs, CPU DIIS, uploads q.
      For N≈87, one WG/system can bitonic-sort 128 padded (ε,index) pairs in
      local memory. DIIS: one WG/system, parallel residual dots, one lane
      solves the 6–10-dim DIIS equation. Reduce SCC convergence to one global
      `max_rms` scalar if host must inspect.
      **FIXED:** `select_occupation_batched` (GPU bitonic sort + occ marking)
      and `diis_step_batched` (one WG/system, f64 Gram + tiny solve on device,
      α-mix warmup, RMS residual) both in `GpuSccPlan` — production `scc_mix
      mix=0` is GPU DIIS. Only per-iteration host contact is one `[batch]`
      rms scalar read. Refinements tracked in §12 D9/D11 (anchored Δq-mix,
      remove `printf`, per-system status, unused `b_mat`/`rhs` buffers).
- [*] **R10. Force driver: eliminate separate context, per-bucket uploads, `finish()`.**
      `GpuForceDriver` creates its own Context/Queue/Program, takes DM/EDM as
      host slices, uploads them, uploads fragments per bucket, builds kernel
      per bucket, calls `finish()` per bucket, downloads forces. Must share
      one `GpuRuntime` with SCC. P/W are already GPU buffers — no CPU copy.
      Forces stay on GPU for FIRE.
      **FIXED (production path):** `GpuDftb` owns all force kernels
      (`k_gamma_f`, `k_rep_force`, per-bucket `k_force`/`k_shift`) built once
      in `new()`; `forces_from_state` runs them on the already-resident D/W —
      no context, no per-call upload, no per-bucket `finish()`. The legacy
      `GpuForceDriver`/`gpu_forces.rs` wrappers still have builders+finish
      but are non-production (§0.4). Forces are still downloaded for the
      CPU FIRE — device-resident FIRE is §12 D12.
- [*] **R11. Force kernel: fix `__local Fragment l_frags[128]` batch>128 bug.**
      Declares 128 fragments, loads only first 128, accesses `l_frags[p.replica]`.
      Intended workload includes batches of 200/500/1000. Do not make array
      1024 long — read `fragments[p.replica]` from global/L2, or redesign
      around one WG/system.
      **FIXED:** `l_frags` removed from `gpu_forces.cl`; kernels read
      `fragments[...]` from global memory. Batch-200 replica test exists
      (`test_hs_assembly_batch200_h2_replica_cap`).
- [ ] **R12. Force kernel: replace atomic accumulation with deterministic reduction.**
      Six CAS-loop float atomics per pair create contention and
      nondeterministic summation order. Better: (a) one WG/system with local
      `float3 F[Na]` and deterministic reduction, or (b) kernel 1 writes
      per-pair `float3`, kernel 2 gathers per-atom. Benchmark rather than
      assume atomics are cheap.
      **STILL OPEN (verified 2026-09-11):** `atomic_add_f32` CAS loops remain in
      all four force kernels (`gpu_forces.cl`). Do during the D7/D12
      device-resident-geometry rework, not before.
- [ ] **R13. Geometry must be device-resident for relaxation.**
      `GpuPairEntry` stores `r,l,m,n` computed on CPU. A future FIRE step
      would do `R_GPU → R_CPU → rebuild pairs → GPU assembly → H/S_CPU → GPU SCC`.
      Store static `(atom_i, atom_j, species, orb offsets)` topology once.
      Compute `ΔR, r, R̂` in the H/S and force kernels from device positions.
      Rebuild gamma on GPU per relaxation step. Only ~435 pairs for 30 atoms.
      **STILL OPEN — superseded by §12 D7** (per-template pair lists, on-device
      `r,l,m,n` from GPU coords, on-device γ via D8 spline, no `finish()` in
      assembly, device buffer fills instead of host zero+upload).

### P1 — Numerical and testing quality

- [*] **R14. Fix RMS norm: divide by `sqrt(n_atoms)`.**
      `residual_and_mix_batched` computes `sqrt(Σ r_A²)` (L2 norm), but CPU
      DIIS uses `sqrt(Σ r_A² / N_A)` (RMS). Same `tol` means different things
      for H2O and AT. Divide by `sqrt(n_atoms)` in the GPU kernel or call
      it L2 everywhere. Prefer RMS to match CPU.
      **FIXED:** `rms = sqrt(Σr²/n_atoms)` in `residual_and_mix_batched` and
      `diis_step_batched` (`gpu_matrix_ops.cl`). Do not retune physics to
      match old L2 numbers.
- [ ] **R15. GEMM tests: use meaningful tolerances, add timing.**
      `gpu_tiled_gemm.rs` uses `tol = N² * 1e-5`; at N=128 permits max element
      error ~0.164. Test relative Frobenius/max errors, all transpose
      combinations used by Löwdin/S⁻¹/², physical H/S-sized matrices. Add
      OpenCL event-timestamp benchmarks.
      **STILL OPEN (verified 2026-09-11):** `tol = n²·1e-5` unchanged in
      `gpu_tiled_gemm.rs:95`. Fix while doing §12 D2 GEMM A/B.

### Additional optimizations (after correctness)

- [ ] **R16. `scale_eigenvectors_batched`: precompute `rsqrt(λ_k)` once per system.**
      Currently recomputes `rsqrt(lambda_k)` for every matrix row (~N² square
      roots). Precompute `rlam[k]` once.
      **STILL OPEN (verified 2026-09-11):** per-element `rsqrt(fmax(lam,…))`
      at `gpu_eigen.cl:477` (N>64 path). Cheap fix alongside §12 D2.
- [*] **R17. Do not hide invalid overlap with `rsqrt(max(λ, 1e-7))`.**
      `LAMBDA_FLOOR = 1e-7` silently clips negative/zero overlap eigenvalues.
      A negative physical S eigenvalue is an error. Report `λ_min`,
      `λ_min/λ_max`, and fail on ill-conditioned/non-positive S. The plan
      already computes `lambda_min` but throws it away.
      **FIXED:** `check_overlap_lambda()` fails loud on `λ_min≤1e-6` after every
      S eigensolve (`GpuSccPlan::new` + `set_geometry`). The in-kernel
      `fmax(lam, LAMBDA_FLOOR)` is now guarded by the host check.
- [ ] **R18. Fuse `Δq → V=γΔq → H_SCC` into one WG/system kernel.**
      Three kernel launches + intermediate global traffic → one kernel:
      load q/q0, compute Δq and V into local memory, barrier, assemble H.
      **STILL OPEN — folded into §12 D10** (same fusion, plus γ from the D8
      spline on device).
- [*] **R19. Build W only after final converged electronic solve.**
      W is unnecessary during ordinary SCC iterations. Build it once after
      convergence alongside the final P.
      **FIXED:** `build_edm` is called only from `forces_from_state`
      (post-finalize, same density kernel with `s_k=ε_k`).
- [*] **R20. Next validation system: real 7-azaindole/AT at N≈84–87, not 12×H2O.**
      The 72-orbital water cluster is a useful N>64 smoke test but does not
      probe spectrum, overlap conditioning, charge redistribution, or
      near-zwitterionic SCC behavior that motivated this solver.
      **DONE:** `scripts/test_gpu_dftb_molecules.rhai` drives AT/GC/azaindole
      through `GpuDftb`; full-chain assemble→SCC→forces tests exist for
      H2O/AT/GC (`gpu_hbond_physics.rs`). The 12×H2O N>64 SCC test
      (`gpu_scc.rs:457`) still asserts only `1e-2` — tighten under §12 D11.

### Summary of positive findings (GPT 5.6)

- Analytic SK derivative mathematics looks healthy after orientation, grid
  origin, and p–s radial unit fixes. H2O/formic parity ~1e-6–1e-5 is strong
  evidence the derivative formulas are basically correct.
- Ordered heteronuclear SK handling, one-based grid origin correction, and
  `1/dr` in p–s derivative were exactly the right subtle fixes.
- Partial-tail-block concept in tiled Jacobi is correct: N=87 stays N=87
  globally, final 23-orbital block treated locally, no singular 96-dim S.

---

## 12. GPT 5.6 Dense-Path Review (commit `e965ae00`, 2026-09-11) — Current Work Order

Review text: `HBond_Relaxed_Scan_GPU.chat.md` lines 2355–2555.
**This section is the active work order** — it supersedes the open items of
§11 (those still open are folded in below). Priority order as given by the
reviewer; each step is designed to possibly make the next one unnecessary —
measure after each, do not batch-implement blindly.

**Central finding:** AT `|dE|~3e-5` is **not a proven f32 floor**. Frozen
`δε_occ~1e-6` and assembly `~1e-7` vs `||C'ᵀC'−I||~2e-6` point at
**loss of eigenvector orthonormality in tiled Jacobi** — a repairable f32
algorithmic issue. Meanwhile the code overreacted: broad FP64 sits inside the
O(N³) Jacobi path and a CPU f64 serial-GEMM `repair_lowdin_x` runs per
replica. Both must go.

### 12.A The δ_CH decomposition — diagnose first (D1–D4)

- [*] **D1. Decompose `δ_CH` before touching arithmetic.**
      `E_band=2Σε_k` vs `2Σc_kᵀHc_k` equality assumes `cᵀSc=1`. Print for
      occupied states:
      `ρ_k = c_kᵀHc_k / c_kᵀSc_k`, `r_k = ‖Hc_k−ε_kSc_k‖/‖H‖`,
      `max|c_kᵀSc_k−1|`, `max_{k≠l}|c_kᵀSc_l|`.
      Reuse existing `GpuDftb::measure` diagnostics (frozen-H machinery already
      computes generalized residuals). This decides whether the ~5e-5 Ha comes
      from column norms, mutual nonorthogonality, or the eigen-equation.
      **DONE + MEASURED (2026-09-11):** `decompose_occ` in `gpu_dftb.rs`
      prints `δ_CH = δ_norm + δ_res` exactly. AT: `δ_norm=2.25e-5` +
      `δ_res=2.78e-5` — **both** defects present ~equally. `max|cᵀSc−1|=1.97e-6`,
      `max_offdiag=7.97e-7`, `max|ε−ρ|=1.07e-6` (same-sign biased sum, not
      random noise). Confirmed: vector quality, not an f32 floor.
- [*] **D3+D4. Occupied-column renormalization + ρ weights — DONE, floor collapsed.**
      `occ_normalize_batched` (C′ columns, one WG/(sys,col)) +
      `occ_rayleigh_batched` (ρ_k=cᵀH_scc c_k/cᵀSc_k → `eig_rho`) in
      `gpu_matrix_ops.cl`; wired into `finalize()` and (after A/B) into
      `scc_step`/`scc_step_diis` via `occ_repair`/`occ_repair_scc` flags;
      `gpu_occ_repair(name,mode)` rhai toggle. `energy_from_state` and
      `build_edm` use ρ_k (variationally correct weight) not drifted ε_k.
      **MEASURED AT:** `|dE_GPU−CPU|` **2.79e-5 → 1.04e-6 Ha** (27×);
      `max|cᵀSc−1|` 1.97e-6 → 2.7e-7. In-SCC renorm: rms 1.30e-6 **stalled@25**
      → 8.9e-7 **converged@13**. H2O `|dE|` 5.4e-7 → 6.6e-8; formic
      `|ΔΔE|` 4.6e-6 → ≤8e-7. The “f32 floor” was eigenvector orthonormality.
      Residual `δ_res=2.8e-5` (ε vs ρ drift) is now *absorbed* — energy uses
      ρ of the actual vectors. Remaining gap is projector error
      (`‖P−P_round‖~6e-6`) — polar step deferred unless needed.
- [x] **D2. Remove broad FP64 from the O(N³) tiled-Jacobi path.**
      `gpu_tiled_jacobi.cl` currently does rotation params, all 2×2 block
      updates, `lU` updates and length-64 strip dots in `double`
      (~1.29M block-updates per pivot). On a 3090 this destroys throughput
      while persistent state is f32.
      Replace with **FP32 FMA + 4 independent accumulators** for dots:
      `s0..s3` over `k+=4`, `sum=(s0+s1)+(s2+s3)` — shorter dependency chain,
      better summation, cheaper than Kahan or f64.
      **Benchmark exactly 3 modes:** (a) pure FP32-FMA, (b) FP64 only for
      scalar rotation construction `c,s` + FP32-FMA updates, (c) current
      broad-FP64 as accuracy reference. Report event time + residual +
      orthogonality + AT ΔE — not just test pass/fail.
      Same A/B for `batched_gemm`: current Kahan measured not to improve
      `δ_CH` while adding serial dependency to every product.

      **DONE 2026-09-11** — `JACOBI_PREC` knob (0/1/2), `jacobi_prec_bench`
      measured (RTX 3090, N=87, batch=1): prec0 43 ms resid 4.8e-5 orth 7.9e-6
      (too coarse — would reintroduce δ_CH~5e-5), prec1 82 ms resid 2.1e-6
      orth 2.2e-7, prec2 243 ms resid 1.2e-6 orth 1.4e-7 → **prec1 default**
      (~3× prec2, equal end-to-end accuracy). Verified: AT |dE|=1.43e-6,
      17 iters no stall; AZA |dE|=8.0e-7, 12 iters. **Caveat:** GC plateaus at
      rms=1.6e-6 under prec1 (tol 1e-6) where prec2 converges at 9.9e-7 —
      reported as `plateau` status (D11), energy still |dE|=1.15e-6.
- [ ] **D3. Occupied-column renormalization — cheapest likely fix, do first.**
      After Jacobi + occupation: `n_k²=C_k'ᵀC_k'`, `C_k'←C_k'/√n_k²` —
      O(N·N_occ), preserves directions so ε_k/W stay valid.
      If off-diagonal nonorthogonality dominates instead, do **one polar step**
      `G=C_o'ᵀC_o'`, `C_o'←C_o'(3I−G)/2` (metric error → O(E²)).
      Apply first in `finalize()` only; if it materially improves E/F, then
      test whether per-SCC use lowers the charge plateau.
- [ ] **D4. Energy and W must be consistent with the corrected occupied subspace.**
      Column rescale: old ε_k fine. Polar mixing of occupied columns: do NOT
      pair corrected columns with old diagonal ε. Use the small occupied
      projected Hamiltonian `H_o=C_oᵀH_sccC_o`, then
      `E_band=2Tr(H_o)`, `W=2C_oH_oC_oᵀ` — invariant under occupied-subspace
      rotations; P, W, E then refer to the same approximate projector.
      Also resolves today's inconsistency: band-form energy is numerically
      better than `Tr(PH0)` but uses a different effective state than forces.

**Measurement sequence (stop early if the floor collapses):**
`FP32-FMA Jacobi → normalize occupied C′ → measure`. If `δ_CH` remains, add
the single polar step after `finalize()`. Rename the audit concept from
“f32 floor” to **“measured floor of the current FP32 Jacobi representation”**
until these are done (§3.0.1 updated accordingly).

### 12.B Remove the CPU islands (D5–D7)

- [x] **D5. `repair_lowdin_x` off CPU, onto GPU — immediately.** DONE —
      5 f32 batched GEMMs + metric + strict-improvement accept; only 2·batch
      floats to host. AT 5.2e-6→3.6e-7, GC 5.7e-6→3.6e-7, AZA 3.1e-6→3.6e-7,
      formic ~1.5e-6→2.4e-7. CPU serial-GEMM path retained as dead/reference.
      Current code downloads full X and S, converts to f64, runs several
      serial N³ CPU GEMMs **per replica**, uploads X — fatal for
      batch=100–1000. Keep the math, change the location: three ordinary
      f32 GEMMs once per geometry —
      `M=XᵀSX`, `Q=(3I−M)/2`, `X←XQ` (with `M=I+E`: `QᵀMQ=I−¾E²+¼E³`).
      **Do not symmetrize X afterward** — X only needs `XᵀSX≈I`, not
      `X=S^{-1/2}`. Then `H'=XᵀHX`, `C=XC'` (tiled GEMM already has
      transpose flags).
- [x] **D6. Reuse X across relaxation steps instead of re-diagonalizing S.**
      DONE — `x_warm`/`x_reuse`: subsequent `set_geometry` Newton-polishes old
      X (≤3 steps, tol 1e-5 on max‖XᵀSX−I‖) before falling back to full
      Jacobi(S) + λ_min check. Per-geometry cost: ~6 GEMMs+2 reads per step
      vs 82 ms Jacobi at N=87. Fallback also covers non-SPD S (Newton can't
      converge) — the λ_min check still fires there.
      **Critical fix found by A/B:** reused X is S^{-1/2}·U, NOT symmetric —
      the old `H'=X·H·X` silently relied on Xᵀ=X. Fixed by maintaining
      `x_t=Xᵀ` (`transpose_batched`) and binding it as the A operand of
      `k_matmul_xh` — correct for any gauge. Scan parity verified
      |ΔE_gpu−ΔE_cpu| ≤ 1.0e-6 reuse ON/OFF. Lesson in labbook.
- [x] **D7. Device-resident geometry preparation in `GpuDftb`.** DONE —
      pair lists are frozen at `new` (ALL i<j pairs per (block_type, s_i,s_j)
      bucket, template-static fields only, exact sizing — no pair_cap, no
      cutoff membership). Per `set_coords`: 2 coord uploads →
      `refresh_pair_geom` (in-place r,l,m,n per bucket) →
      `build_gamma_batched` (G on device via the D8 spline) →
      `onsite_diagonal` → `assemble_pairs` per bucket. No finish(), no
      host fills/uploads for G/H0/S/V_asm — every off-diagonal element is
      written by a pair block (full i<j coverage), diag by onsite/persistent
      S=I. Bug caught by GC: `cubic_interp_params` clamps the stencil index
      but t keeps growing → far pairs must be zero-written/skipped
      (`r ≥ (n_grid−1)·dr` guard in assemble_pairs/force_pairs/
      force_pairs_scc_shift — mandatory since H0/S are no longer
      pre-zeroed). Verified: GC |dE|=6.4e-7 F=3.3e-6; AZA |dE|=9.3e-7
      F=5.9e-6; H2O/sp3/formic parity ≤5e-7; n64 batch ok. (Subsumes R13;
      R12 deterministic force reduction still open.)

### 12.C Physics-consistency upgrades (D8–D9)

- [x] **D8. Replace analytic γ/γ′ with pretabulated species-pair splines.**
      DONE — `methods/dftb/gamma_spline.rs`. SPAMMM convention: natural
      cubic stored as float4 per knot (T, T″_spl, T′, T‴_spl) with
      T(r)=1−r·γ(r); two value-splines (never a derivative of noisy f32
      knots — measured: differentiating the T-spline amplifies knot noise
      1/dr, γ′ err grows with nk). Curvatures from a Thomas tridiagonal
      solve in f64 at build. nk=256, dr≈0.157 bohr, r_max=40 bohr:
      physical-range max|Δγ|=5.1e-7, max|Δγ′|=1.7e-6; table 64KB total
      (4KB/pair — `__local`-sized). Wired into `build_gamma_batched` (D7)
      and `force_gamma_deriv_batched` — f64 `gamma_prime_full_f32` island
      gone; energy and force now evaluate ONE numerical γ.
- [x] **D9. Keep the DIIS f64 island; fix its formulation, don't shrink it.**
      DONE — anchored Δq-mix (q = q_latest + Σ c_i(q_i − q_latest), Σc=1 exact),
      f64 Gram/solve kept, drop-oldest retry before α-mix fallback, `printf`
      removed → per-system flag/reason device buffers reported by Rust
      (`DIIS fallbacks: N total, last reason=…`), `b_mat`/`rhs` deleted.
      Measured: AT 17 iters no stall, GC |dE|=1.15e-6, formic 9–12 iters/pt
      with occasional pivot-fallback (reason=1) reported explicitly.

### 12.D Throughput + engine honesty (D10–D12)

- [x] **D10. Density/W + SCC-stage fusion.**
      DONE — `occ_idx[Nocc]` from `select_occupation_batched`; density/EDM
      loop only occupied (49 vs 87 for AT), symmetric triangle + mirror,
      4-accumulator FMA + local-cached weights. `fused_dq_v_hscc_batched`
      replaces delta_q + gamma_matvec + h_scc_update (3 launches → 1, Δq/V
      in local, still written to global for the energy dot). Verified
      bit-consistent: AT |dE|=1.43e-6, GC 1.15e-6, AZA 8.0e-7, formic scan
      |ΔE_gpu−ΔE_cpu| ≤ 8.4e-7 — identical to pre-fusion output.
      (Subsumes R18.)
- [~] **D11. Per-system convergence state; never conflate stall with converge.**
      PARTIAL — per-system `statuses: Vec<SccStatus>` (Converged / Plateau /
      Failed) reported by `scc_mix`, `gpu_scc_status(name)` Rhai getter,
      plateau-vs-cap-reach distinguished (plateau detector → Plateau;
      max_iter while still descending or non-finite → Failed). N>64 test
      tightened 1e-2 → 1e-4 (measured |dE|=4.4e-5 |dq|=9.1e-6 |d_eig|=1.1e-5
      on the legacy path). **Remaining:** device `active[sid]` early-outs in
      per-iteration kernels — deferred until batch≫1 scans make the gain
      real (batch=1 is a no-op).
- [x] **D12. Fix FIRE physics — CPU-half done; device-resident FIRE open.**
      Fixed in `GpuDftb::fire_step` AND the CPU `FireOptimizer` (same bug):
      mixing is now Bitzek `v←(1−α)v+α·F̂·‖v‖` with GLOBAL per-replica norms
      (was per-atom α‖F_i‖F̂_i = adding a force to a velocity); dt/α/n_pos
      are per-replica `Vec`s (was one shared state across the batch);
      added the missing vmax=2.0 cap; ordering standardized to
      P→adapt→mix→v+=F·dt→x+=v·dt. Verified: GC relax max|F| 7.6e-2→8.1e-4
      in 45 steps, E monotone −44.9210→−44.9278. New Rhai:
      `gpu_fire_step`, `gpu_relax`, `gpu_bench` (wall-clock ms/scc).
      **Open:** FIRE on device (today downloads forces per step — ~100KB
      at batch=1000, acceptable for now); `md_step` still not
      velocity-Verlet; `relax` stalled-reporting still open.

### 12.E FP32 policy (reviewer's table — adopt verbatim)

| Operation                                | Precision                              |
| ---------------------------------------- | -------------------------------------- |
| H/S/SCC matrices, C, P, W storage        | FP32                                   |
| Dense GEMM / Jacobi block+strip updates  | FP32 FMA, 4-way accumulators           |
| Jacobi `c,s` construction                | benchmark FP32 vs scalar-only FP64     |
| Column normalization / polar correction  | FP32                                   |
| Löwdin metric correction                 | FP32 GPU GEMMs                         |
| DIIS Gram + tiny solve                   | FP64                                   |
| γ/γ′                                     | f64-fitted spline → FP32 evaluation    |
| final `Δq·V`, `q0·V`                     | FP64 accumulation (cheap)              |
| final repulsive-energy reduction         | FP64 accumulation (cheap)              |
| final occupied-energy sum / total E      | FP64                                   |
| coords on CPU/API                        | f64 fine; device coords FP32           |

Rationale: final scalar reductions are tens/hundreds of ops once per
geometry — f64 there costs nothing. f64 inside millions of Jacobi-pivot ops
is the opposite tradeoff.

### 12.F Rhai engine + benchmarking rules (from the same review)

- Rhai is for **workflows/A-B experiments/production runs**; Jacobi
  residuals, derivative parity, kernel invariants stay **Rust unit tests**.
  Do not replace mathematical unit tests with scripts.
- ~~`gpu_scc_bench.rs` still benchmarks the legacy `GpuDriver`~~ —
  REWRITTEN to the production `GpuDftb` path (wall ms/scc at batch
  1/8/32; formic N=28 → ~0.41 ms/iter, launch-bound, batch nearly free).
- `gpu_bench(name, n_runs, max_iter, rms_tol)` added to `dftb_engine` —
  wall-clock per SCC call on the production engine. **Still open:**
  OpenCL-event breakdown (set_coords / orthonormalize / per-iter stages /
  forces) at N=28 and N=87 × batch — that drives further optimization.

### 12.G Five immediate wins (reviewer's summary)

1. Remove broad FP64 Jacobi arithmetic (D2).
2. Cheap C′ normalization/orthogonality repair (D3, D4).
3. Kill the CPU Löwdin repair (D5, D6).
4. Stop rebuilding/uploading geometric pair records (D7).
5. Correct FIRE and move it to GPU (D12).

Each improves accuracy *and* throughput — none is a tradeoff.
