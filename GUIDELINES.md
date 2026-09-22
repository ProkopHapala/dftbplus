# dftbplus — Detailed Rules & Architecture

The full rulebook referenced by `AGENTS.md`. `AGENTS.md` carries the one-line statements; this file carries the rationale, the "how", and the edge cases. Read the relevant section *before* editing code in that area — each section exists because its rule has been violated repeatedly and expensively.

Sections:

- **§1** Core Principles (expanded)
- **§2** Tests Are Diagnostics, Not the Goal
- **§3** f32 Is The Architecture, Not An Inconvenience
- **§4** Dense GPU Hot Path — Non-Negotiable Invariants
- **§5** Surgical Edits & Checkpointing
- **§6** Reusable Architecture
- **§7** Testing & Validation (levels, debug prints, messages)
- **§8** Performance
- **§9** Style
- **§10** Repo Navigation & Folder Policy

---

## §1 Core Principles

- **KISS** — simplest solution that works; one-liner > ten-liner.
- **AHA** — avoid hasty abstractions and boilerplate.
- **YAGNI** — surgical edits; touch only what's needed; no unrelated cleanup; comment out, don't delete; ask if ambiguous.
- **DRY** — inventory existing code before writing new; generalize rather than duplicate.
- **SoC** — separate compute from presentation. `rust_dftb/src/{core,methods,qmqm}` are pure compute; `src/bin/` holds executables; plotting/scripts stay in `scripts/` or `debug/`.
- **SSOT** — one authoritative source of truth. Fortran `src/dftbp/` is the parity reference; `rust_dftb/` is the implementation; `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` is the status tracker.
- **TDD** — define verification before coding; parity vs Fortran reference/analytical/physical invariants; run tests after every change.
- **Fail Fast, Fix the Physics** — **silent fallbacks are strictly prohibited.** Anything unexpected (NaN, Inf, out-of-range value, missing file, shape mismatch, failed convergence, violated invariant) must **fail loud and early** with a full stack trace — panics > silent `Ok`. Never mask a bug by clamping divergent values, returning a default, retuning until `Ok`, or broad error-swallowing. **Rust:** no `.unwrap_or(default)`/`.ok()` that drops an `Err` you didn't diagnose; no `let _ = result;`; propagate with `?` or `panic!`/`assert!`/`expect("context")` instead. **Python:** no bare `except:` or `except Exception:` that passes/logs-and-continues; catch only the specific error you handle, else let it raise. If a solver fails, **fix the solver, not the scenario**.
- Compact code, unlimited line length; short names for math/physics symbols (`E_tot`, `T_ij`, `m_i`, `F_ij`).

---

## §2 Tests Are Diagnostics, Not the Goal (most-violated rule)

**Passing tests is NOT the endgoal.** A green test suite says nothing about physical correctness. Tests are a tool to **locate where the program is wrong**, in order to **understand and fix the physics**. The goal is root cause, not green-at-any-cost.

- **Do NOT make tests green at all cost.** A red test is a diagnostic: it tells you *where the physics is wrong*. Keep it red if necessary to document broken physics — a known problem is better than a masked one.
- **Never cheat a test green** by violating physics, suppressing errors, loosening tolerances, or picking cautious parameters.
- When a test fails, the response is to **investigate the physics**, not to silence the test. If the test itself is wrong, fix the test — but say so explicitly and justify it.

---

## §3 f32 Is The Architecture, Not An Inconvenience (second-most-violated rule)

GPU kernels are **single precision**. Agents repeatedly write iterative schemes that are only stable in exact arithmetic, then "fix" the resulting blowup by iterating harder or tightening tolerances. That is backwards and has cost days of debugging. Mandatory discipline:

- **No open-loop iterations.** Any iterative scheme (purification/TC2/SP2, Newton–Schulz, DIIS, FIRE, Jacobi) must have an **enforced restoring invariant** — trace normalization, renormalization, projection back into the admissible set. "The math says the spectrum stays in [0,1]" is not enforcement: f32 noise (~1e-5 relative per SpGEMM) accumulates and violates the precondition, after which divergent polynomial maps amplify it exponentially. State each solver's invariant and *how it is enforced*.
- **Tolerances must be set to the MEASURED floor, then the loop must STOP.** Iterating against a noise floor is a bug, not diligence. Report the floor honestly (a plateau is a legitimate answer); hard-fail only on genuine blowup. Never silently continue past a floor and never tighten a tolerance below the measured floor to "look better".
- **Discrete decisions in f64, on the host.** Branch selects, occupancy/trace counts, convergence tests, bracket/bisection logic: if the decision quantity costs O(N) (not O(N²)) to reduce, compute it in f64. A wrong branch caused by 1e-5 noise is catastrophic; a 1e-5 error inside a matrix element is harmless. Budget precision where decisions are made, not uniformly.
- **ANY f64 inside a GPU kernel is a policy violation unless deeply justified** (user-stated 2026-09-18). On consumer GPUs f64 arithmetic runs at ~1/32–1/64 rate and f64 `exp`/`sqrt` are software-emulated; f64 scratch also doubles local-memory/register pressure and halves occupancy. Same violation class as CPU↔GPU transfers in the loop and kernel/buffer (re)builds in the loop. If numerical accuracy is claimed as justification, the justification must analyze whether a hybrid suffices first — double-single (2×f32) emulation, compensated/rearranged summation, f64 only for the scalar decision with f32 bulk arithmetic. Audit duty: every f64 site must be listed (see `Measured_Facts_Jacobi_Sweeps.md` §4b) with its justification or its removal plan; "it's only a small kernel" is not a justification — indirect damage (register pressure, occupancy, instruction mix) must be checked.
- **Truncating an INTERMEDIATE is a different error class than truncating a RESULT.** Dropping blocks of an intermediate deletes real contributions to in-mask outputs (`C_ij = Σ_k A_ik B_kj`). Use magnitude/τ-based dropping with an error bound, never an ad-hoc radius, for intermediates.
- **Approximate inputs invalidate rigorous bounds.** Gershgorin/norm bounds computed from a truncated or non-symmetric operand are no longer bounds. If a solver's stability depends on a bound, it must be computed from something that still guarantees it, or be made generous by a measured margin.
- **Every new numerical scheme must document:** stability invariant · enforcement mechanism · measured f32 floor · behaviour at the floor. Missing any of these = the change is unfinished.

---

## §4 Dense GPU Hot Path — Non-Negotiable Invariants (USER-stated 20×; violations are defects, not style)

Applies to `GpuDftb`/`GpuSccPlan` and every kernel in the SCC / force / FIRE / geometry-update hot path. Full rationale + checklist: `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.manifest..md` **§14**. An agent MUST NOT violate these because a simpler implementation is easier to write.

- **NO global atomics — EVER.** `atomic_add(force[atom], …)` and friends are forbidden in production kernels; global atomics require explicit USER approval. The architecture is **OWN OUTPUT → GATHER INPUTS → WRITE ONCE**: an atom workgroup owns its force and Mulliken population (one WG per (replica, atom) over a static incident-pair list, or a pair-write kernel + atom-gather kernel — benchmark the two gather layouts, never against an atomic "fallback"), a matrix tile owns its tile, a replica owns its SCC state. Scatter-based accumulation is rejected by design, not merely discouraged.
- **ZERO allocation / compilation in hot loops.** After engine construction, SCC iterations, `eval`, force evaluation, FIRE steps and geometry updates perform ZERO `Buffer`/`Kernel`/`Program`/context/queue creation and ZERO dimension-dependent `Vec` allocation — on host AND device. All scratch (masks, DIIS history, force/occupation/energy scratch, FIRE control) is a persistent field; clearing = device fill kernel, never a fresh zero-vector upload. A regression counter test must make any reintroduced hot-loop allocation fail CI.
- **NO implicit fallbacks.** No silent CPU solver, no switching to a deprecated eigensolver/algorithm, no automatic nearest-index replica re-seed, no mixer switch, no tolerance loosening, no warn-and-continue onto a slower path. `Failed` is a valid solver result; recovery (e.g. `retry_with_seed(src,dst)`) is an explicit higher-level driver call — array index is not a physical metric.
- **Minimize kernel launches per iteration.** Launch overhead dominates at N≈30–100. Target ≈5–6 launches per SCC iteration (`q→H_scc → BᵀH → (BᵀH)B → Jacobi+occupations → Mulliken/DIIS`); fuse anything sharing inputs/workgroup; every additional hot-loop kernel needs written justification. Fermi μ/f_k are fused into the Jacobi tail (that WG owns the spectrum) — never download eigs → host bisection → upload occupations.
- **NO host synchronization per iteration.** Run SCC in ~8–10-iteration async chunks, then read one compact `rms[batch]`/status record; same for FIRE steps. Do NOT add a dedicated GPU convergence-reduction kernel — piggyback the residual on Mulliken/DIIS work already done and simply don't read it every iteration. FIRE `dt/α/n_pos/mode` are persistent device state updated by the existing reduction kernel — never download 4·batch stats + upload controls per step.
- **Energy piggybacks on forces.** `E=Tr[H·ρ]` and `F=dE/dR` share pair machinery — the fused gather kernel emits `e_atom[replica][atom]`; the host sums it only when a value is requested. No dedicated energy kernel in the hot loop.
- **f32 bulk, f64 only for cheap scalar decisions** (DIIS QR, energy/reduction tails, convergence scalars). **NO Kahan in GEMM** — 4 independent f32 FMA accumulators + balanced sum.

---

## §5 Surgical Edits & Checkpointing

- **Minimum intervention:** write only what the task needs.
- **Strict checkpointing:** after every significant step, summarize what changed, what was verified, what remains.
- **Preservation:** back up before major module changes; comment out (`//`/`#`) deprecated/experimental code instead of deleting; mark unfinished with `// TODO` / `// DEBUG`.
- **Never mark "fixed"/"done" without explicit USER confirmation.** A code change is not proof. You must: (1) run a test demonstrating the fix, (2) show the result, (3) wait for confirmation. When in doubt, leave status as "investigating"/"unverified".

---

## §6 Reusable Architecture

- **Inventory first** — review reference sources (Fortran `src/dftbp/`, `Import_other_Repos.md`, `CODEMAP.md`) before writing anything new.
- **Composability over bloat** — build integrated systems, not isolated scripts; refactor into shared-crate functions.
- **Generalize over duplicate** — if a function almost fits, generalize it; if generalization risks backward compatibility, stop and report for approval.
- Separate executables (`rust_dftb/src/bin/`), compute modules (`rust_dftb/src/{core,methods,qmqm}/`), and helper scripts (`scripts/`). Test scripts are thin wrappers calling shared-crate functions; consolidate related scripts into one with CLI routing.

---

## §7 Testing & Validation

- **HARD RULE — no oversized runs.** Never start a test, benchmark, Hessian, SCC, eigensolve, or script that can freeze the machine, exhaust host or GPU memory, or take more than **1 minute** of wall time. This is a hard stop, not a preference. In particular: no full nanocrystal Hessians at R10 or larger, no unbounded column loops, no `B` large enough to allocate multi-GB replica slabs, no matrix whose cubic diagonalization is already known to be minutes (the 4944×4944 host eigh is ~100 s — do not run it). Bound the case first (SiH4 or a few dozen atoms, a synthetic matrix with `n` chosen so the run finishes in seconds, `VIB_MAXCOL` of a handful of columns). If a run is approaching the cap, kill it and shrink the case. A number that required a multi-minute job is not a measurement. Compile time is not an excuse to then run the huge case.
- **Numerical sanity:** place checks ensuring values are finite, in-range, not unexpected zeros.
- **Diagnostic tests, not pass/fail:** print actual numbers (per-atom residuals, per-element energies, worst contributor, sign of deviation). Assert on physical invariants (energy conservation, sign convention, monotonicity, symmetry). On failure, output should locate the bug without re-running. `assert_eq!` is a smoke check, not a scientific test.
- **Parity before coding:** define correctness checks vs Fortran reference (`src/dftbp/`), analytical solutions, conservation laws, symmetry, physical limits. Parity tests live in `rust_dftb/tests/parity_*.rs`.
- **Foreground execution:** run tests synchronously with full output — never background, `| tail`, `| head`, `| grep`, or `&`.
- **Three review levels:** L0 `cargo test` (automated regression) · L1 agent reads `.out`/`.log` artifacts unfiltered · L2 human reviews `.png`/`.svg` plots in `debug/`.
- **Refactoring discipline:** before refactoring, run each old test and show results to USER; delete old files only after explicit approval; never delete plots.
- **Visual review:** use shared plotting utilities, not ad-hoc code.
- **Images in chat:** use `<ref_file file="/abs/path/to/image.png" />` — the only format that renders as a clickable image. Markdown `![]()`/`[]()` do NOT render. Save PNGs to `debug/`.
- **Long-running scripts MUST print unbuffered progress** (`eprintln!`/`println!` with flush, or `PYTHONUNBUFFERED=1`) — print starts, accepted steps with energy decrease, and finish. Never run silently for minutes.
- **Debug prints are gated, not deleted.** Use verbosity-gated logging (Rust: `log` crate macros `error!`/`warn!`/`info!`/`debug!`/`trace!` filtered via `RUST_LOG`; or `eprintln!` behind a `const VERBOSE: bool`/`--verbose` flag) so output is controlled by level, not by removing lines. **Do NOT remove debug prints until the program is functioning correctly.** If output is too noisy, **lower the debug level** (e.g. `RUST_LOG=warn`) — do not delete the print statements. Silent code is undebuggable; gated prints let you re-raise verbosity the moment something breaks again.
- **Informative messages, not "it broke".** Every error, panic, assertion, and debug print must carry enough context to locate the bug without re-running: **where** (function/module/file:line — Rust's `panic!`/`expect` and `#[track_caller]` give this), **what** happened (the violated invariant / unexpected state, in plain words), and the **values of all relevant variables** (inputs, indices, shapes, energies, residuals — print the numbers, not just names). Bad: `"failed"`, `"NaN error"`, `unwrap()`. Good: `expect(&format!("bond {i}-{j} stretched: |r|={r:.3} > cutoff {c:.3}"))`, `panic!("energy non-finite at step {step}: E={E}, max|F|={fmax}");`.

---

## §8 Performance

- **Efficiency is a primary design goal, not an afterthought.** This project exists to make DFTB faster — the physics is already solved in DFTB+, we are reimplementing it for speed (batched many-small-system GPU computation). The whole architecture must revolve around performance from the ground up. Code that is correct but does unnecessary work is a design failure. Priority order: (1) physical correctness, (2) debuggability, (3) performance — but design for performance from the start, don't bolt it on later. Accuracy compromises (e.g. f32 on GPU) must be controlled, understood, and documented. See `doc/prokop/AGENTS/guidelines/efficiency.md` for the full rationale and 12 concrete rules.
- **Rust is the engine** — all simulation logic in Rust. Flat arrays, cache-aware, preallocate; prefer `&[T]`/`&mut [T]` over `Vec<T>` in hot paths; SoA/data-oriented layouts; be explicit about `f32` vs `f64`.
- **OpenCL is the accelerator** — GPU must match CPU within tolerance. Prefer **NVIDIA GPU**; never report PoCL/CPU timings as GPU timings. GPU is single precision f32 by default, packed float4 arrays preferred, workgroup size ~32 preferred.
- **HARD RULE — never build or allocate in a simulation hot loop.** OpenCL `Context`, `Program`, `Queue`, every `Kernel`, every device `Buffer`, and reusable host staging arrays must be created/preallocated during harness initialization or an explicit topology/size reconfiguration step, then retained persistently. A per-step/per-iteration `Kernel::builder().build()`, `Program::build()`, `Buffer::builder().build()`, `Vec` allocation, or equivalent OpenCL object creation is a performance bug even if tests pass. Hot steps may only update changing scalar kernel arguments, enqueue already-built kernels, and reuse existing buffers. If dimensions/topology change, rebuild once outside timing/SCC loops and fail loudly on stale capacity; never silently allocate from `step`, `eval`, `scc_iter`, `run_*`, or force-evaluation methods.
- **GPU kernels:** design for memory latency; **gather only — global atomics are banned in production** (see §4); minimize branching/sync; maximize shared/local memory; avoid host-device transfers; **fuse secondary checks into existing kernels** — keep launches per iteration minimal. See `rust_dftb/src/qmqm/*.cl` and `rust_dftb/src/methods/*/*.cl`.
- **Build footprint** — `rust_dftb/Cargo.toml` profiles use `debug = 1` + `strip = "debuginfo"` (dev & release): drops debug sections (~16× smaller debug binaries), `.eh_frame` survives so backtraces keep function names + panic `file:line`. Release adds `lto = "thin"`, `codegen-units = 1`, `incremental = true`.
- **EFFICIENCY GUIDELINES — read and follow `doc/prokop/AGENTS/guidelines/efficiency.md`.** 12 concrete rules + 6 general principles derived from an 18-point code review of the CPU DFTB pipeline. Covers: three-tier data lifetime (static / per-geometry / per-SCC-iteration), no allocation in hot loops, no strings/HashMaps/clones in hot paths, don't recompute existing results, verify library internals before claiming (don't fabricate explanations), benchmark in release mode (`--release`, `OPENBLAS_NUM_THREADS=1`), exploit mathematical structure (Cholesky factorization, Frobenius trace, precomputed H0'), precompute polynomial coefficients instead of Neville at runtime, evaluate all channels at once, analytic derivatives over finite differences, check units (Bohr vs Å), warm-start iterative solvers.

---

## §9 Style

- **No micro-abstractions** — no 1-line stubs/wrappers; inline if simple.
- **Clean interfaces** — group related state into structs; use builder/default named args to avoid long call strings.
- **Compact layout** — long lines, minimal blank lines; no wrapping that disrupts readability.
- **Naming & comments** — short math/physics symbol names; comments for intent/rationale/derivations only, placed inline behind the code line.
- **Rust:** gated debug logging (see §7 Debug prints); `&[f32]`/`&mut [f32]` in hot paths; `bytemuck` for zero-copy OpenCL casts; `///` rustdoc (not `/* */`).
- **OpenCL:** kernels in `.cl` next to their Rust driver module; CPU reference authoritative.
- **Python:** support scripts/utilities only; NumPy for array glue; `plt.show()` only in CLI/main, never in libs.
- **Parity work:** when porting from Fortran `src/dftbp/` or reference repos (SPAMMM, FireCore, learn_Rust), cite the reference file+function in a comment (e.g. `// ported from src/dftbp/dftb/hamiltonian.F90:build_H0`).

---

## §10 Repo Navigation & Folder Policy — see `CODEMAP.md`

`CODEMAP.md` is the repo navigation router (big-picture map: Fortran reference, Rust crate layout, docs, tasks, scripts, debug). Read it first when you need to find something.

Hard rules:
- **`debug/`** — all debug artifacts (PNGs, scratch CSVs, SCC dumps, one-off plots/scripts) go here as `debug/<topic>/`. **Never commit anything under `debug/`.** Not gitignored (kept navigable); enforced by convention. Stage with `git add -A -- . ':!debug/'` and review `git status` before committing.
- **`scripts/`** — reusable kept scripts (Python/Bash); outputs go to `debug/`, never here.
- **`doc/prokop/tasts/<task>/`** — task specs only (Markdown). No `scripts/`/`artifacts/` subfolders. Specs may *reference* `debug/...` paths but must not *contain* artifacts.
- **`tests/`** — reference data + inputs only. Debug dumps → `debug/scc/`.
- **`doc/prokop/DFTB_Reimplementation_Progress/`** — design notes & status roadmap. Update `OVERVIEW_Roadmap.md` when implementation status changes.
- Before every commit: confirm nothing under `debug/` is staged, no large/regenerable files (`.png`, `.csv`, `.xyz`, `.log`) staged unless intended.
