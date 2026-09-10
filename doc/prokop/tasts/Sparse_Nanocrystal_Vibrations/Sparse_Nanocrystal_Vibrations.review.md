---
type: review
title: "Sparse_Nanocrystal_Vibrations — implementation audit + review protocol"
tags: [sparse, gpu, review, verification, tc2, newton-schulz, hessian, nanocrystal]
timestamp: 2026-09-09
status: audit complete / all findings unverified-by-user
scope: sparse BSR4 solver only (dense multi-system H-bond pipeline is a separate task)
---

# Sparse_Nanocrystal_Vibrations — what is actually implemented, and how to review it

This document has two halves:

- **Part A — Audit.** What the working tree actually contains, measured on the
  real GPU today, versus what `.report.md`, `.tasks.md`, and the GPT-5.6 review
  in `.chat.md` (from line 2064) claim. Emphasis: **unphysical shortcuts that
  make tests green while the physics is wrong.**
- **Part B — Review protocol.** The gates, commands, thresholds and
  "what counts as proof" rules by which this task should be reviewed from here
  on.

**Scope.** Sparse BSR4 purification / SCC / forces / Hessians for Si/H
nanocrystals. The dense multi-system H-bond GPU solver
(`qmqm/`, `hbond_gpu_scc`, tiled Jacobi) is a **separate** task and is not
re-reviewed here except where a shared file (`interpolation.rs`) is on the
sparse call graph.

## How this audit was produced

| | |
|---|---|
| Repo state | working tree on top of `b269ab63` |
| Device | `NVIDIA GeForce RTX 3090`, OpenCL platform 0 (`NVIDIA CUDA`); platform 1 is PoCL CPU and was **not** used |
| SK set | `RUST_DFTB_SK_DIR=/home/prokop/SIMULATIONS/dftbplus/slakos/matsci-0-3` |
| Build | `cargo test --no-run --tests` — compiles (interpolation.rs was briefly mid-edit during the session; current tree compiles) |
| Suites run | `gpu_sparse_bsr4` (23), `gpu_bspline_eval`, `spgemm_plan`, `locality_sweep`, lib `sparse_system` + `sparse_forces`, `sih_padded_basis`, `gate_e_determinism`, `gate_f_geom_opt` |
| Mode | debug profile, `--test-threads=1 --nocapture`, unfiltered output |

Everything below marked *measured* is from a run today on the 3090. Everything
marked *read* is from the source. Nothing here is confirmed as fixed — per
`AGENTS.md`, status stays "investigating" until the USER confirms.

A sandboxed `clinfo` on this machine shows **only PoCL**. Tests must be run
outside the sandbox or they silently skip / hide NVIDIA-only bugs. Same class
of harness lie as HBond review B2.

---

# Part A — Audit

## A.0 Three blockers, in priority order

### B1 — There is still no sparse DFTB calculation

GPT-5.6 said this. The working tree has **not** closed it.

What exists is a sparse **purifier attached to a frozen Hamiltonian**:

```text
CPU HamiltonianBuilder (dense H0/S, often after dense SCC)
       ↓
pad to BSR4 (full mask on every physics gate)
       ↓
Newton–Schulz Z ≈ S⁻¹
       ↓
K0 from spectral bounds of that frozen H
       ↓
TC2 → K
       ↓
Tr(K·H0)  called "energy"
```

Evidence:

- Production leftover `rhai_run_sparse_purify` / `_geom` still starts from a
  stored dense `SccResult`. **Do not use for new jobs.** Production sparse is
  `sparse_new` / `sparse_scc` / `sparse_eval` (`SparseDftb`). Lab numbers:
  manifest §0.7.
- `SparseSystemWorkspace::run_scc` (`sparse_system.rs:454`) is named SCC and
  documents `γ·Δq → Hscc → repeat`. The body is `compute_z; compute_k0;
  tc2_purify; mulliken_charges`. **No gamma, no Hscc update, no charge mix,
  no loop.** Compiler warning: fields `h_scc`, `plan_zh`, `plan_bz` are never
  read.
- Gates C, D, E, F all call `gpu.newton_schulz_inverse` + `gpu.tc2_purify`
  (host-roundtrip wrappers). **None of them construct or use
  `SparseSystemWorkspace`.**
- Gate D explicitly does `builder.build_scc(...)` then purifies that
  *dense-converged* H. Sparse K is not a self-consistent charge state.

For phonons this is fatal: the force/Hessian of a K that was purified on
someone else's Hamiltonian is **not** the derivative of a sparse energy, and
the geometry "minimum" of `Tr(K H0)` without SCC or E_rep is not a stationary
point of DFTB. Gate F measured that in the open (A.2).

**Why this is blocker #1:** every later gate (E, F, G, H) is defined on
"the sparse model". Today that model does not exist as a closed loop.

### B2 — The harness converts physics failures into green tests

Same disease as the HBond review, different symptoms. Catalog of
**measured** cheats:

| Cheat | Where | What it does |
|---|---|---|
| `try_gpu()` swallows `Err` **and** `catch_unwind` panics as skip | every sparse test file | GPU faults become `ok` |
| `tc2_purify_dev` `Err` → `eprintln!; return;` | `gpu_sparse_bsr4.rs:950–952` | a failed purification **passes** |
| `SparsePurifyWorkspace::new` `Err` → return | same test | constructor failure passes |
| Gate C `FAILED:` rows do not fail the test | `locality_sweep.rs:396, 421` | R_K=2..5 and R_Z=2..5 all failed TC2/NS today; test still `ok` |
| `converged_rk.unwrap_or(5.0)` | `locality_sweep.rs:405, 430` | if nothing converged, invent R=5 and continue |
| `R_H` / `R_leak` `unwrap_or(NAN)` then no assert on NaN | `locality_sweep.rs:287, 331` | silent NaN leakage metric |
| Gate E `rel_bias < 0.1 \|\| max_bias < 1e-3` | `gate_e_determinism.rs:359` | **10% force error** is a pass |
| Gate E Hessian filled symmetrically, then "max\|asym\|" asserted | `gate_e_determinism.rs:445` + `:393` | asymmetry is **identically zero** |
| `h_ref = 0.05` is inside the candidate list | `:377, :368` | that row has `diff = 0` by construction; plateau cannot be empty |
| Gate F `n_unstable = count(eig < -1e-2)`, allow `<= 6` | `gate_f_geom_opt.rs:407–413` | a real negative mode of −1.48e-3 is not "unstable"; six large imaginaries would still pass |
| Gate F `f_tol = 1e-2` Ha/Å | `:311` | ~0.27 eV/Å stop; tied to a bogus Gate E "bias floor" |
| `SparsePerfStats` prints `largest_dense = 0` as a literal | `gpu_sparse.rs:105` | the audit **cannot** report a dense alloc |
| perf-stats test constructs dummy timings, **no assertion** | `gpu_sparse_bsr4.rs:1209–1225` | "if it prints, the struct works" |
| workspace reuse: Run 2 `Err` → return | `sparse_system.rs:617–620` | second SCC failure passes |
| `Tr(KS)` asserts at `1e-2` everywhere | Gates C/D, TC2 tests, workspace | measured errors are 1e-6; a 10 000× dead projector still passes |
| K₀ vs K_ref asserted `< 1.0` | `gpu_sparse_bsr4.rs:808` | any K₀ in the same ballpark of magnitude |

Missing SK data is a skip, not a fail (`sih_padded_basis.rs:234`,
`gate_e_determinism.rs:258`, `gate_f_geom_opt.rs:276`).

This is worse than loose tolerances. Several tests are **structurally
incapable of going red** for the physics they advertise.

### B3 — The validated slice is algebra on toys; the physics gates are tautologies

*Measured* today, the kernels that **actually work**:

| Kernel / path | System | Result |
|---|---|---|
| masked SpGEMM vs dense | 4 atoms, full/geo mask | max\|dC\| = 2.4e-7 |
| SpGEMM plan vs intersection | 8-atom chain, 10 launches | exact match, 1.07× (not a nanocrystal benchmark) |
| McWeeny from perturbed projector | 3 atoms | ‖KSK−K‖ → 2.6e-7 |
| TC2 from `0.8·K_exact` | 3 atoms | 4 iters, R_I = 4.1e-6, ‖K−K_ref‖ = 1.3e-6 |
| GPU vs CPU B-spline stencil | synthetic 64 nodes, 200 pts | ΔV = 8e-8, ΔV' = 5e-7, ΔV'' = 6e-6 |
| D=2K / W=2KHK vs dense f64 | 4 atoms, random K,H | D exact, W max err 2.6e-5 |
| SiH4 padded dummy occupation | full mask, dense-SCC H | dummy occ = 0, \|dE\| = 8.6e-8 |

That is real linear algebra. It is **not** sparse DFTB, not locality, not a
force, not a Hessian, not a spectrum.

The physics gates (C, E, F) all passed today. Their numbers are in A.2.
**Every one of them is a false positive** of a different kind. The organising
principle is the same as the HBond review: *the green checkmarks do not mean
what the test name says.*

---

## A.1 GPT-5.6 / tasks.md status: claimed vs verified

Legend: **✓** verified · **~** partial · **✗** not done · **!** claim stronger
than the code · **F** false-positive test (green, physics wrong).

| # | Item | Tasks / report | Audit verdict |
|---|---|---|---|
| P0 | sparse firewall + perf stats | [*] | **~ !** Firewall is an *opt-in* feature (`sparse_firewall`), off in tests. `to_dense()` is called from production `dftb_engine.rs:380,540` and from every physics gate. `largest_dense = 0` is a hardcoded print, not a measurement. Perf-stats test does not run the solver. |
| P1 | canonical C² B-spline | [~] | **~** Working-tree `EqGridTable::eval_into` / `eval_with_deriv_into` now call the B-spline (shared file; dense agent also touching it). GPU stencil parity is good on **synthetic** points. **Not Gate A:** no real Si–Si / Si–H SKF at knots, cutoff, or bond windows; GPU `cubic_interp_params` still `clamp`s (`dftb_hamiltonian.cl:194`); sparse H/S is still CPU `HamiltonianBuilder`, not a GPU BSR assembler. |
| P2 | analytic SK + angular derivs | [~] | **~** Angular machinery exists in `rotation.rs` / `forces.rs` (dense-agent scope). Sparse production force in Gates E/F is **FD of Tr(K H0)**, not those analytic derivatives. |
| P3 | D=2K, W=2KHK | [~] | **~** Algebra parity holds (A.2). Workspace exists. Not attached to a self-consistent sparse K, not used for any force. Still allocates D. |
| P4 | symbolic SpGEMM plans | [~] | **~** Plan kernel matches intersection on 8 atoms. `SparseSystemWorkspace` builds plans then **does not use** `plan_zh`/`plan_bz` for K0. Gates C–F never call `spgemm_plan_bsym_dev`. Host TC2 still uses the intersection wrappers. |
| A2 | TC2: 2 SpGEMMs, residual before update, no host sync | [ ] | **~** `tc2_purify_dev` / workspace TC2 do residual-before-update. Host `tc2_purify` (what Gates C–F call) still rebuilds KS/KSK. Workspace still `read_f32`s trace **every** diagnostic iter (`check_every=1` in the tests). |
| B1 workspace | persistent GPU state, zero alloc in loop | [ ] | **~** Struct exists. Unused by any gate. `run_scc` is not SCC. |
| B2–B7 | GPU H0/S, gamma, Hscc, K0, SCC loop, Mulliken | [ ] | **✗** No GPU sparse H/S assembly. No gamma. No Hscc. Mulliken exists as a kernel and is used after a one-shot purification. |
| C1–C3 | GPU pair-force + atom-gather | [ ] | **✗** Gates E/F use nested FD of energy. |
| Gate C | locality R_K × R_Z | [ ] false positive | **F** confirmed today. See A.2 / N1. |
| Gate D | padded Si/H | [*] | **~** Dummy isolation **works** (dummy occ = 0, energy parity 9e-8). But: full mask, dense-SCC H, hardcoded valence, E_dummy inside Gershgorin emax, asserts at 1e-2 / 5e-2 / 1e-1. Not a sparse production path. |
| Gate E | determinism + h plateau | [ ] false positive | **F** confirmed today. See A.2 / N2. |
| Gate F | geom opt at own minimum | [ ] broken | **F** confirmed today: **Si–H collapsed to 0.93 Å** and the test passed. See A.2 / N3. |
| #10 +2I gap | | | **✗** still in Gate C and in `gpu_sparse_bsr4` / `sparse_system` random-H tests. |
| #11 R_leak | | | **✗** still `‖Q_val‖ − ‖Q_K‖`. Measured 0.0 at the "plateau". |
| #12 Z projected before K0 | | | **✗** `locality_sweep.rs:252` still `z_on_mz.project_to_mask(&m_k)` before `build_k0`. |
| #17 dummy vs spectral bounds | | | **✗** SiH4 emax = 2.31 Ha, E_DUMMY = 2.0. Dummy not excluded from bounds. |
| #18/#19 NS/K0 host roundtrip | | | **~** `_dev` paths exist. Gates still use host wrappers. Device NS vs dense S⁻¹ is **200× worse** than host NS (A.2). |
| #20 perf audit lies | | | **✗** confirmed; dummy print + hardcoded 0. |
| #21 O(N²) mask | | | **✗** `build_geometric_mask` is still all-pairs. |

---

## A.2 Measured numbers (RTX 3090, today)

### Algebra (full mask, random / synthetic, 3–8 atoms)

Tiled-unrelated sparse kernels, f32:

| Test | Asserted | Measured |
|---|---|---|
| SpGEMM vs dense | (implicit small) | max\|dC\| = 2.4e-7 |
| Plan vs intersection | — | 0 (exact), 8 atoms, 1.07× |
| McWeeny | — | R_I = 2.6e-7 |
| TC2 from 0.8 K_exact | R_I < 1e-3, \|Tr−Nocc\| < 1e-2 | R_I = 4.1e-6, Tr err ~ 8e-6, ‖ΔK‖ = 1.3e-6 |
| TC2 from K0 (random H + **+2I**) | R_I < 1e-3, ‖ΔK‖ < 5e-2 | R_I = 2.2e-7, ‖ΔK‖ = 7.7e-7 after **30 oscillating iters** (Tr 6.36 → 1.14 → 4.39 → … → 3.00) |
| Host NS vs dense S⁻¹ | err < 1e-2 | max\|dZ\| = **1.1e-5**, R_Z = 1.2e-5 |
| Device NS vs dense S⁻¹ | err < 1e-2 | max\|dZ\| = **2.18e-3**, R_Z = 1.9e-5 |
| D/W vs dense f64 | — | D 0, W 2.6e-5 |
| GPU B-spline stencil | — | ΔV 8e-8 / ΔV' 5e-7 / ΔV'' 6e-6 (synthetic; not SKF) |

**Device Newton–Schulz residual does not match the actual inverse error.**
Host: R_Z ≈ max\|Z−S⁻¹\|. Device: R_Z is 100× *smaller* than max\|Z−S⁻¹\|.
The reported R_Z cannot be trusted as a convergence certificate on the `_dev`
path. The 1e-2 assertion still passes with 4.6× margin on a 2e-3 inverse error.

`SparseSystemWorkspace::run_scc` (3 atoms, random H+2I, **not SCC**):
40 TC2 iters, R_I = 9.5e-6, Tr(KS) = 3.000010. Prints `Trace(K²) ≈ 2.711
(should be ≈ 3)` — the wrong invariant for a non-orthogonal metric — and
**does not assert it**. Reuse run: R_I = 9.5e-6 then 1.4e-7 on the same
geometry (not bit-reproducible); asserted `< 1e-4` so it passes.

### Gate C — locality sweep (5 atoms on a 1.5 Å line, random H **+ 2I**)

System length ≈ 6 Å. Masks at R ≥ 7 Å cover **the whole molecule**.

Phase 1 (R_Z = 10):

| R_K | Result today |
|---|---|
| 2, 3, 4 | TC2 exhausted 80 iters, R_I ≈ 9e-3–2e-2 |
| 5 | TC2 **diverged** at iter 65 (best R_I = 7.4e-4) |
| 7, 10 | E_err = 5.2e-6, q_err = 5.4e-6, \|Tr−Nocc\| = 3.6e-6, R_leak = **0**, R_H = 3.2e-7 |

Phase 2 (R_K = 7): R_Z = 2,3,4,5 Newton–Schulz **stalled**. R_Z = 7,10 same
numbers as above.

The test printed `FAILED:` on 8 of 12 cells, then `Gate C: PASS`.
`R_leak = 0` is the difference-of-norms formula on a validation mask that
already contains M_K.

TC2 at the "plateau" still starts at Tr(KS) = **10.5** for Nocc = 3 and
oscillates through 0.90 before landing. That is not a healthy purifier on an
"insulator"; it is TC2 fighting a random matrix whose gap was not created
by `+2I` (adding 2I shifts every eigenvalue equally).

### Gate D — SiH4 padded basis (the one mostly-real physics test)

*Measured:*

| Quantity | Asserted | Measured |
|---|---|---|
| \|Tr(KS) − 4\| | < 1e-2 | **6.7e-6** |
| max dummy occ | < 1e-2 | **0** |
| active Mulliken electrons | < 1e-1 of 8 | **8.000013** |
| \|dE\| vs dense projector on padded H | < 1e-2 | **8.6e-8** |
| max \|dq\| | < 1e-2 | **3.1e-6** |
| max \|ΔK\| | < 5e-2 | **2.3e-6** |
| spectral bounds | — | emin = −1.36, **emax = 2.31** (E_DUMMY = 2.0) |
| TC2 | — | 20 iters, Tr oscillates 6.23 → 3.56 → 5.51 → … → 4.000 |

Dummy isolation works. Assertions are 1 000–100 000× looser than the
measurement. The Hamiltonian came from **dense SCC**; the mask is **full**;
valence counts are **hardcoded** because matsci-0-3 q0 parsing is broken.

### Gate E — "force noise" and Hessian h plateau

*Measured:*

- Energy repeatability: spread = 0 (same bits in, same bits out — not a force test).
- TC2 tol 1e-3 vs 1e-4: ΔE = **7.7e-5 Ha**.
- Sparse vs dense "force": max\|F\| = 0.120, max\|bias\| = **1.56e-3**,
  rel_bias = **1.3e-2**. Pass criterion is `rel < 0.1 || abs < 1e-3`.
- Both "forces" are 3-point FD of a **spinless band energy** (no factor 2,
  no E_rep, no SCC, no analytic D/W contraction).
- Hessian: dense f64 **energy** stencil, **symmetrized by filling
  `hess[j][i] = h_ij`**. max\|asym\| = **0.0000** at every h.
- `‖H(0.05) − H_ref‖ = 0` because h_ref **is** 0.05.
- Plateau = {0.01, 0.02, 0.05, 0.10} — every candidate. Chosen h = 0.02.

This test cannot detect force noise, Hessian asymmetry, or an h floor.
GPT-5.6 #13 is still an exact description of the file.

### Gate F — FIRE "optimization" (the money plot, without a plot)

*Measured, 563 s on the 3090:*

| | Start (1.60 Å, tetrahedral) | "Converged" step 123 |
|---|---|---|
| E = Tr(K H0) | −1.444 | **−1.671** |
| \|F\| | — | 9.95e-3  (`f_tol = 1e-2`) |
| Si–H | 1.60 Å | **0.931 / 0.921 / 0.926 / 0.933 Å** |

Physical Si–H in silane is ~1.48 Å. The optimizer **collapsed the bonds by
0.55 Å** because the energy being minimized has **no repulsive term**.
Electronic-only `Tr(K H0)` wants atoms closer. That is textbook
wrong-potential optimization.

The Hessian at this geometry is again the **dense f64 band-energy** Hessian,
not the sparse force Jacobian:

- 1 negative eigenvalue (−1.48e-3), 4 near-zero, 10 positive
- "Unstable" defined as eig < **−1e-2** → count = 0 → PASS
- allow `n_unstable <= 6`

`mask rebuild checkpoint` every 20 steps is a **print statement**. The code
always uses `build_full_mask`. 123 FIRE steps × 31 FD energy evals ≈ 3800
sparse NS+TC2 pipelines on 5 atoms — the architectural crime GPT-5.6 #16
warned about, now measured.

**A green Gate F today means: we can collapse SiH4 to 0.93 Å and not notice.**

---

## A.3 New findings (beyond GPT-5.6, from today's run)

**N1 — Gate C failure-to-pass conversion is complete.** Not only is the
system too small and `+2I` not a gap: 8/12 sweep cells hard-failed TC2 or
Newton–Schulz and the test function still returned `ok`. A locality sweep
whose truncated cells are allowed to fail without failing the test is a
progress printer, not a gate.

**N2 — Gate E's 10% force contract is incompatible with 5% frequencies.**
Hessian error scales as σ_F / h. With σ_F / \|F\| = 1.3% *admitted*, and
the assertion allowing 10%, you cannot claim ~5% ordinary-mode accuracy.
And that 1.3% is not even sparse-truncation noise: it is FD-of-two-different-
pipelines (dense eig vs sparse TC2) of a half-electron band energy.

**N3 — Gate F is the existence proof that a green test can be unphysical.**
Collapsed 0.93 Å Si–H, no E_rep, electronic-only energy, stop at 1e-2 Ha/Å,
Hessian of a different method, imaginary-mode threshold set so the actual
negative eigenvalue does not count. This is the behaviour `AGENTS.md`
forbids: *making the test green at the cost of butchering the physics.*

**N4 — Device NS inverse is not the host NS inverse.** 2.18e-3 vs 1.1e-5
against dense S⁻¹, with a residual that pretends 1e-5. This will pollute
every `_dev` K0/TC2 once the gates actually switch to the workspace.

**N5 — TC2 from a real K0 (even SiH4, even full mask) oscillates for ~20
iters** with Tr(KS) swinging by O(Nocc). The easy test (`K0 = 0.8 K_exact`)
converges in 4 iters and is what `test_tc2_convergence` uses. The hard test
exists (`test_k0_and_tc2_vs_dense_projector`) and is good — but its ‖ΔK‖
assert is 5e-2 against a 8e-7 measurement, so it will not catch a 1000×
regression. For phonons, a purifier that takes 20 noisy iters per geometry
and is restarted from scratch on every FD displacement is both a correctness
and a performance bug (no warm-start, Gate F's 9.4 min for 5 atoms).

**N6 — `run_scc` is a naming lie.** Callers and the tasks.md "Phase B done
when workspace runs SCC" will treat this as the loop. It is one purification
of H0. `h_scc` is allocated and never written.

**N7 — Plans are built and then ignored for the products that matter.**
`plan_zh` / `plan_bz` unused; Gates still on intersection `tc2_purify`.
P4 remains "a kernel that can match", not "TC2 is faster".

**N8 — GPU B-spline Gate A is an isolated-evaluator test.** It compiles a
mini-program with `bspline3_v_d1_d2`, not the production Hamiltonian path,
and never opens an SKF file. Knot ± ε, cutoff, Si–Si / Si–H bond windows
from GPT-5.6 #22 are still absent.

**N9 — Shared `interpolation.rs` is in flux.** During this audit the file
did not compile (orphaned Hermite-tail statements after a partial B-spline
swap), then compiled. Production CPU `eval_into` now claims C² B-spline
with trailing zero knots (good direction). Do not mark P1/A1 done until
Gate A on **real SKF** exists, CPU and GPU use the same controls, and the
sparse assembler actually calls that path. The dense H-bond agent also
edits this file — coordinate, do not fight.

---

# Part B — Review protocol

The organising principle is the same as the HBond review: **the problem is
not "are any numbers good" — several algebraic numbers are good. It is "do
the green checkmarks mean anything".** Today they do not, for three
independent reasons (B1/B2/B3). Do not review performance or nanocrystal
spectra until the harness cannot lie and the energy being differentiated
is a self-consistent sparse DFTB energy including E_rep.

Dense multi-system H-bonds stay out of this protocol.

## Gate 0 — Make the harness incapable of reporting a false pass

| ID | Requirement | Accept when |
|---|---|---|
| G0.1 | Missing SK data is a hard failure. Resolve `RUST_DFTB_SK_DIR` to a repo default (matsci-0-3 for Si/H) or `panic!` with searched paths. | `cargo test --test sih_padded_basis` with the variable unset **fails**. |
| G0.2 | Device banner once per suite: name, vendor, local-mem size. Non-NVIDIA aborts unless `RUST_DFTB_ALLOW_CPU_CL=1`. | Forcing PoCL (`OCL_DEFAULT_PLATFORM_IDX=1`) fails clearly. Sandboxed `clinfo` is not an excuse to skip. |
| G0.3 | No OpenCL `Err` or panic is mapped to skip. Absence of *any* device may skip; a device that errors must fail. | Grep shows no `Skipping Gate` / `Skipping GPU sparse` reachable from a `CL_*` status. Delete `catch_unwind` → skip. |
| G0.4 | `Err` from TC2/NS/workspace construction is a test failure, never `return`. | `gpu_sparse_bsr4.rs:950`, `sparse_system.rs:617` become `panic!` / `unwrap`. |
| G0.5 | Gate C: a `FAILED` cell is either expected (then recorded as a *measured* large error, not a solver crash) or it fails the test. Solver non-convergence at small R is data; `unwrap_or(5.0)` is not. | No `unwrap_or` default radius. Plateau selection only from rows that actually returned metrics. |
| G0.6 | One entry point: `scripts/run_sparse_review.sh` — NVIDIA, matsci-0-3, `--test-threads=1 --nocapture`, `tee` to `debug/sparse_review/<date>/`, non-zero exit on failure **or skip**. | Exit code is a valid L0 gate. |

## Gate 1 — Stop testing the wrong invariants

These are the unphysical shortcuts that must be removed **before** any
"fix the solver" work, or the next agent will retune around them.

| ID | Shortcut to delete | Replace with |
|---|---|---|
| G1.1 | `h[i,i] += 2.0` "to make a gap" | A real passivated Si/H (start: SiH4 already in Gate D; then Si29H36). Gap = HOMO–LUMO of the physical spectrum, printed. |
| G1.2 | `R_leak ≈ ‖Q_val‖ − ‖Q_K‖` | Exact outside-mask ‖P_{Mval\MK}(KSK)‖_F on device (GPT-5.6 #11). |
| G1.3 | `z.project_to_mask(&m_k)` before K0 | Full Z on M_Z in ZHZ; project only the final K0 to M_K. |
| G1.4 | FD of `Tr(K H0)` called a force | Analytic sparse force from the **same** K, same H, same E_rep. Until that exists, Gate E/F stay **red / ignored**, not green. |
| G1.5 | Symmetrized Hessian + h_ref in the candidate list | Unsymmetrized `H[:,a] = −(F(R+h e_a)−F(R−h e_a))/(2h)`. h_ref **not** in the sweep. Report ‖H−Hᵀ‖/‖H‖. |
| G1.6 | `rel_bias < 0.1`, `f_tol = 1e-2`, `eig < -1e-2` is unstable, `n_unstable <= 6` | Thresholds from measurement with ~3× margin, named, and justified. A collapsed geometry (Si–H < 1.2 Å on silane) is a **hard fail**, independent of \|F\|. |
| G1.7 | Energy without E_rep used for opt / PES / Hessian | E_tot = E_el + E_rep. No FIRE until E_rep parity exists. Gate F's 0.93 Å result is the demonstration. |
| G1.8 | `Trace(K²) ≈ Nocc` | `Tr(KS) = Nocc` and `‖KSK−K‖_F`. Never Tr(K²) for a non-orthogonal metric. |
| G1.9 | `largest_dense = 0` literal; dummy `SparsePerfStats` test | Real counters. A test that only constructs a struct is not P0. |
| G1.10 | Easy TC2 (`K0 = α K_exact`) as the only tight test | Keep it as a unit test. The **regression** test is K0-from-H on SiH4 / Si29H36 with asserts at measured accuracy (G2). |

## Gate 2 — Tighten every contract to measured accuracy

Red tests here are diagnostics. Do not loosen.

| Test | Now | Proposed | Measured today |
|---|---|---|---|
| TC2 from α K_exact, R_I | 1e-3 | **1e-5** | 4e-6 |
| TC2 from α K_exact, \|Tr−Nocc\| | 1e-2 | **1e-5** | 8e-6 |
| K0→TC2 vs dense projector, ‖ΔK‖_max | 5e-2 | **1e-5** | 7.7e-7 |
| Host NS vs S⁻¹, max\|dZ\| | 1e-2 | **1e-4** | 1.1e-5 |
| Device NS vs S⁻¹, max\|dZ\| | 1e-2 | **keep 1e-2 until N4 is understood, then 1e-4** | **2.18e-3** (this one should stay **red** until the 200× discrepancy is diagnosed) |
| Gate D \|dE\| | 1e-2 | **1e-6** | 8.6e-8 |
| Gate D \|dq\| / dummy occ | 1e-2 | **1e-5** / **1e-8** | 3e-6 / 0 |
| Gate D electrons | 1e-1 | **1e-4** | 1.3e-5 |
| D/W W-err | (none tight) | **1e-4** | 2.6e-5 |
| Si–H after any "opt" | (none) | **1.40–1.55 Å** | 0.93 Å (must fail) |

Plus: device NS must report a residual that matches max\|Z−S⁻¹‖ within a
small factor, or fail loud with both numbers.

## Gate 3 — Close the sparse physics seam (this is what makes Gates C–H reviewable)

Order matters. Do not start FIRE before G3.4.

**G3.1 — One canonical energy.** For a frozen geometry, sparse E_el + E_rep
versus dense f64 DFTB on SiH4. Same SK, same C² spline, same cutoff.
Until this exists, no energy in this pipeline is a DFTB energy.

**G3.2 — Self-consistent sparse SCC.** The loop in tasks.md B6, actually
executed: sparse H0/S (CPU first is acceptable if it is the *sparse* H0/S
on M_HS, not a dense `SccResult`) → γ·Δq → Hscc (dummy-safe) → K0/TC2 →
Mulliken → mix. No dense eigenproblem in the loop. Converged q and E
versus dense SCC on SiH4, then a small Si cluster.

**G3.3 — Analytic force of *that* energy.** D=2K, W=2KHK from the
converged sparse K, contracted with analytic dH/dR, dS/dR from the same
C² spline, plus F_rep and SCC γ' terms. Newton's third law, ΣF ≈ 0,
translation invariance. FD of energy is a **check**, not the production
force. Relative analytic-vs-FD < 1e-3 on SiH4 at equilibrium.

Implementation: `sparse_forces.rs::sparse_analytic_forces` builds D/W
from K and calls CPU `forces.rs::compute_forces_from_dw` (same four
components as dense). The H-bond GPU kernels (`qmqm/gpu_forces.cl`) are
a **separate in-progress path** — treat them read-only from this work.
Do **not** merge sparse BSR4 forces with the H-bond GPU kernel until
**both** codepaths are developed and tested; the end goal is one shared
pair-force kernel, but joining now would mix two unfinished bug lists.

**G3.4 — Energy-gradient consistency.** Central-difference dE/dR of the
pipeline's own E_tot vs its own analytic F. This is the only test that
catches "E and F evaluated at different charge states" and "forgot E_rep".
Test: `tests/gate_g3_energy.rs::test_g3_3_analytic_force_and_g3_4_energy_gradient`.

Do not begin Gate F/G/H before G3.4 is green **and** Si–H is near 1.48 Å.

## Gate 4 — Redo locality and dummy spectral physics

**G4.1 Gate C** on SiH4 (full physical H/S, truncated K/Z) then Si29H36.
Independent R_K, R_Z. Exact R_leak. Fixed R_K as N grows. No `+2I`.
2D table of (R_K, R_Z) → E, q, F, Tr, R_in, R_leak, R_H, iters, time.

**G4.2 Dummy orbitals.** Exclude dummy from spectral bounds and from
Tr(KS)/Mulliken. K0 dummy rows = 0. Hscc must not add V_H to E_dummy.
Assert dummy occ < 1e-8 after SCC, not just after one-shot purification
of a dense H.

**G4.3 Diagnose N4 and N5** before using `_dev` NS/TC2 in production:
why device Z disagrees with S⁻¹ by 2e-3; why K0-TC2 oscillates 20 iters
on SiH4. Fix the solver, not the scenario.

## Gate 5 — Performance (only after Gates 0–4)

Prerequisites: profiling queue, real `SparsePerfStats` counters, NVIDIA
banner, no PoCL timings.

Then, and only then: SpGEMM count per TC2 iter (must be 2), host scalar
reads per iter (must be 0 on non-diagnostic), plan vs intersection on
N ≈ 300 at R_K ~ 7 Å, workspace reuse across Hessian displacements
(zero `Buffer::builder` in the displacement loop). Gate I/J from the
manifest stay after correctness.

P4's 8-atom 1.07× is not a performance result.

## Gate 6 — L2 human review artifacts

Plots to `debug/sparse_review/`, not into `doc/prokop/tasts/`.

1. Gate C heatmap (R_K, R_Z) → log energy error and R_leak, on SiH4 and
   Si29H36. Makes the 5-atom "plateau = full molecule" lie obvious.
2. TC2 traces Tr(KS) and R_I vs iter for SiH4 K0 (the oscillating one)
   vs α K_exact (the easy one).
3. SiH4 FIRE: E, \|F\|, mean Si–H vs step, **with and without E_rep**,
   on the same axes. Today's run without E_rep is the "before" curve
   (collapses to 0.93 Å). That single figure justifies G1.7 / G3.1.
4. Unsymmetrized Hessian η_asym vs h, sparse analytic F vs dense analytic F.
5. Device vs host ‖Z−S⁻¹‖ and R_Z vs iter (N4).

---

## What counts as proof

Same rules as the HBond review, applied to this solver:

1. **"FIXED" requires the exact command and unfiltered NVIDIA output**,
   with the assertion at the *target* tolerance, and with the relevant
   test actually executing (no skip, no `return` on `Err`).
2. A physics gate that passed by tautology (symmetrized Hessian, h_ref
   in the list, full-mask "locality", collapsed geometry with a loose
   eig threshold) is **not** a pass. Mark it `[F]` until rewritten.
3. Partial work is marked `[~]` with the remaining item named.
   `SparseSystemWorkspace` existing is not "sparse SCC exists".
4. **A red test stays red until the physics is understood.** Gate F at
   0.93 Å must stay a failure of the *model/test*, not an invitation to
   raise `f_tol` or `n_unstable`.
5. Never mark fixed without USER confirmation.

## Suggested order of work

1. **Gate 0** — harness honesty. Small. Everything downstream depends on it.
2. **G1.1–G1.10** — delete the shortcuts. Several are one-line test edits
   that turn false greens into real reds (Gate C `FAILED` → fail; Gate F
   Si–H window; device NS assert; drop `+2I`; drop Hessian symmetrization).
   **Do this before implementing more kernels.** Otherwise the next agent
   will make the new kernels pass the old cheats.
3. **G2** — tighten algebra contracts to measured accuracy. Device NS
   stays red (N4).
4. **G3.1–G3.2** — E_rep + real sparse SCC loop. Unblocks every energy claim.
5. **G4.3** — N4/N5 (device NS, TC2 oscillation) so the workspace path is
   numerically trustworthy.
6. **G3.3–G3.4** — analytic force of the same energy. Prerequisite for
   any Hessian.
7. **G4.1–G4.2** — real locality + dummy spectral hygiene.
8. **Then** rewrite Gates E, F, G, H against the real force.
9. **Gate 5** performance, with P4 actually wired into TC2/NS/K0/W.

Do **not** spend time on degree buckets, packed plans, or lane-mapping
benchmarks while Gate F can pass a 0.93 Å silane.

## Related documents

- `Sparse_Nanocrystal_Vibrations.manifest.md` — source of truth; §13
  checklist needs reconciling with Part A.1.
- `Sparse_Nanocrystal_Vibrations.report.md` — P0–P4 / C–E "completed"
  marks are still too strong; Gate F section is superseded by A.2 (the
  FD-of-energy problem is no longer hypothetical: it collapsed SiH4).
- `Sparse_Nanocrystal_Vibrations.tasks.md` — Phase B `run_scc` does not
  match the code. Phase C must not start from the current Gate E/F force.
- `Sparse_Nanocrystal_Vibrations.chat.md` from line 2064 — GPT-5.6 review;
  items #1–#22 remain the right technical list. This audit adds
  measurements and the Gate 0/1 "stop lying" layer.
- Dense H-bond review (separate scope):
  `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.review.md`
- `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md`
