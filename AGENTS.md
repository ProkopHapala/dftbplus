# dftbplus — Agent Rules of Conduct

DFTB+ fork: upstream Fortran reference (`src/dftbp/`) + from-scratch Rust reimplementation (`rust_dftb/`) of semi-empirical LCAO solvers (DFTB, xTB) and a multi-system QM/QM fragment solver with OpenCL GPU offload. Python utilities (`pyBall/`, `tools/`) wrap/extend both. Numerical correctness, physical consistency, debuggability, and performance are paramount.

**Languages:** Rust (reimplementation, GPU orchestration) + OpenCL (GPU kernels) + Fortran (upstream reference for parity) + Python (utilities/glue, never a hot path).

## Core Principles

- **KISS** — simplest solution that works; one-liner > ten-liner.
- **AHA** — avoid hasty abstractions and boilerplate.
- **YAGNI** — surgical edits; touch only what's needed; no unrelated cleanup; comment out, don't delete; ask if ambiguous.
- **DRY** — inventory existing code before writing new; generalize rather than duplicate.
- **SoC** — separate compute from presentation. `rust_dftb/src/{core,methods,qmqm}` are pure compute; `src/bin/` holds executables; plotting/scripts stay in `scripts/` or `debug/`.
- **SSOT** — one authoritative source of truth. Fortran `src/dftbp/` is the parity reference; `rust_dftb/` is the implementation; `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` is the status tracker.
- **TDD** — define verification before coding; parity vs Fortran reference/analytical/physical invariants; run tests after every change.
- **Fail Fast, Fix the Physics** — **silent fallbacks are strictly prohibited.** Anything unexpected (NaN, Inf, out-of-range value, missing file, shape mismatch, failed convergence, violated invariant) must **fail loud and early** with a full stack trace — panics > silent `Ok`. Never mask a bug by clamping divergent values, returning a default, retuning until `Ok`, or broad error-swallowing. **Rust:** no `.unwrap_or(default)`/`.ok()` that drops an `Err` you didn't diagnose; no `let _ = result;`; propagate with `?` or `panic!`/`assert!`/`expect("context")` instead. **Python:** no bare `except:` or `except Exception:` that passes/logs-and-continues; catch only the specific error you handle, else let it raise. If a solver fails, **fix the solver, not the scenario**.
- Compact code, unlimited line length; short names for math/physics symbols (`E_tot`, `T_ij`, `m_i`, `F_ij`).

## ⚑ Tests Are Diagnostics, Not the Goal (most-violated rule)

**Passing tests is NOT the endgoal.** A green test suite says nothing about physical correctness. Tests are a tool to **locate where the program is wrong**, in order to **understand and fix the physics**. The goal is root cause, not green-at-any-cost.

- **Do NOT make tests green at all cost.** A red test is a diagnostic: it tells you *where the physics is wrong*. Keep it red if necessary to document broken physics — a known problem is better than a masked one.
- **Never cheat a test green** by violating physics, suppressing errors, loosening tolerances, or picking cautious parameters.
- When a test fails, the response is to **investigate the physics**, not to silence the test. If the test itself is wrong, fix the test — but say so explicitly and justify it.

## Never Do This

- **NEVER use `rm`, `sed -i`, `cat >`, `echo >>`, heredocs, or shell redirects to delete/modify files.** Use the Devin `edit`/`write`/`read` tools so changes appear in the diff viewer. If an edit tool fails, do smaller targeted edits — never fall back to shell.
- **Cascade bash tool: ALWAYS use Background=false** (never Background=true); never use `exit` in commands — otherwise it is stuck and user must kill it.
- Never delete/rearrange existing code, or make unrelated aesthetic edits, without explicit permission.
- Never apply quick-fixes that hide root causes (hard-coded outputs, clamping to hide divergence).
- Never reinvent existing functionality — inventory first (`CODEMAP.md`, `Import_other_Repos.md`, Fortran `src/dftbp/`, reference repos: SPAMMM, FireCore, tblite).
- Never copy-paste between modules — extract to a shared function in `rust_dftb/src/core/`.
- Never cheat a test green (see ⚑ Tests Are Diagnostics above).
- **Ask, don't Guess** — when unsure, ask the user.

## Surgical Edits & Checkpointing

- **Minimum intervention:** write only what the task needs.
- **Strict checkpointing:** after every significant step, summarize what changed, what was verified, what remains.
- **Preservation:** back up before major module changes; comment out (`//`/`#`) deprecated/experimental code instead of deleting; mark unfinished with `// TODO` / `// DEBUG`.
- **Never mark "fixed"/"done" without explicit USER confirmation.** A code change is not proof. You must: (1) run a test demonstrating the fix, (2) show the result, (3) wait for confirmation. When in doubt, leave status as "investigating"/"unverified".

## Reusable Architecture

- **Inventory first** — review reference sources (Fortran `src/dftbp/`, `Import_other_Repos.md`, `CODEMAP.md`) before writing anything new.
- **Composability over bloat** — build integrated systems, not isolated scripts; refactor into shared-crate functions.
- **Generalize over duplicate** — if a function almost fits, generalize it; if generalization risks backward compatibility, stop and report for approval.
- Separate executables (`rust_dftb/src/bin/`), compute modules (`rust_dftb/src/{core,methods,qmqm}/`), and helper scripts (`scripts/`). Test scripts are thin wrappers calling shared-crate functions; consolidate related scripts into one with CLI routing.

## Testing & Validation

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

## Performance

- **Rust is the engine** — all simulation logic in Rust. Flat arrays, cache-aware, preallocate; prefer `&[T]`/`&mut [T]` over `Vec<T>` in hot paths; SoA/data-oriented layouts; be explicit about `f32` vs `f64`.
- **OpenCL is the accelerator** — GPU must match CPU within tolerance. Prefer **NVIDIA GPU**; never report PoCL/CPU timings as GPU timings. GPU is single precision f32 by default, packed float4 arrays preferred, workgroup size ~32 preferred.
- **HARD RULE — never build or allocate in a simulation hot loop.** OpenCL `Context`, `Program`, `Queue`, every `Kernel`, every device `Buffer`, and reusable host staging arrays must be created/preallocated during harness initialization or an explicit topology/size reconfiguration step, then retained persistently. A per-step/per-iteration `Kernel::builder().build()`, `Program::build()`, `Buffer::builder().build()`, `Vec` allocation, or equivalent OpenCL object creation is a performance bug even if tests pass. Hot steps may only update changing scalar kernel arguments, enqueue already-built kernels, and reuse existing buffers. If dimensions/topology change, rebuild once outside timing/SCC loops and fail loudly on stale capacity; never silently allocate from `step`, `eval`, `scc_iter`, `run_*`, or force-evaluation methods.
- **GPU kernels:** design for memory latency; gather > scatter; minimize branching/atomics/sync; maximize shared/local memory; avoid host-device transfers; **fuse secondary checks into existing kernels**. See `rust_dftb/src/qmqm/*.cl` and `rust_dftb/src/methods/*/*.cl`.
- **Build footprint** — `rust_dftb/Cargo.toml` profiles use `debug = 1` + `strip = "debuginfo"` (dev & release): drops debug sections (~16× smaller debug binaries), `.eh_frame` survives so backtraces keep function names + panic `file:line`. Release adds `lto = "thin"`, `codegen-units = 1`, `incremental = true`.

## Style

- **No micro-abstractions** — no 1-line stubs/wrappers; inline if simple.
- **Clean interfaces** — group related state into structs; use builder/default named args to avoid long call strings.
- **Compact layout** — long lines, minimal blank lines; no wrapping that disrupts readability.
- **Naming & comments** — short math/physics symbol names; comments for intent/rationale/derivations only, placed inline behind the code line.
- **Rust:** gated debug logging (see Testing & Validation §Debug prints); `&[f32]`/`&mut [f32]` in hot paths; `bytemuck` for zero-copy OpenCL casts; `///` rustdoc (not `/* */`).
- **OpenCL:** kernels in `.cl` next to their Rust driver module; CPU reference authoritative.
- **Python:** support scripts/utilities only; NumPy for array glue; `plt.show()` only in CLI/main, never in libs.
- **Parity work:** when porting from Fortran `src/dftbp/` or reference repos (SPAMMM, FireCore, learn_Rust), cite the reference file+function in a comment (e.g. `// ported from src/dftbp/dftb/hamiltonian.F90:build_H0`).

## Repo Navigation & Folder Policy — see `CODEMAP.md`

`CODEMAP.md` is the repo navigation router (big-picture map: Fortran reference, Rust crate layout, docs, tasks, scripts, debug). Read it first when you need to find something.

Hard rules:
- **`debug/`** — all debug artifacts (PNGs, scratch CSVs, SCC dumps, one-off plots/scripts) go here as `debug/<topic>/`. **Never commit anything under `debug/`.** Not gitignored (kept navigable); enforced by convention. Stage with `git add -A -- . ':!debug/'` and review `git status` before committing.
- **`scripts/`** — reusable kept scripts (Python/Bash); outputs go to `debug/`, never here.
- **`doc/prokop/tasts/<task>/`** — task specs only (Markdown). No `scripts/`/`artifacts/` subfolders. Specs may *reference* `debug/...` paths but must not *contain* artifacts.
- **`tests/`** — reference data + inputs only. Debug dumps → `debug/scc/`.
- **`doc/prokop/DFTB_Reimplementation_Progress/`** — design notes & status roadmap. Update `OVERVIEW_Roadmap.md` when implementation status changes.
- Before every commit: confirm nothing under `debug/` is staged, no large/regenerable files (`.png`, `.csv`, `.xyz`, `.log`) staged unless intended.
