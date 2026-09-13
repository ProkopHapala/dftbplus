# dftbplus — Agent Rules of Conduct

DFTB+ fork: upstream Fortran reference (`src/dftbp/`) + from-scratch Rust reimplementation (`rust_dftb/`) of semi-empirical LCAO solvers (DFTB, xTB) and a multi-system QM/QM fragment solver with OpenCL GPU offload. Python utilities (`pyBall/`, `tools/`) wrap/extend both. Numerical correctness, physical consistency, debuggability, and performance are paramount.

**Languages:** Rust (reimplementation, GPU orchestration) + OpenCL (GPU kernels) + Fortran (upstream reference for parity) + Python (utilities/glue, never a hot path).

**This file states the general rules. The concrete details live in [`GUIDELINES.md`](GUIDELINES.md) — each topic below names its section (§N); read it before working in that area.** Violating these rules has repeatedly cost days of debugging.

## Core Principles — details: GUIDELINES.md §1

- **KISS · AHA · YAGNI · DRY** — simplest working solution; no hasty abstractions; surgical edits only; inventory before writing, generalize rather than duplicate.
- **SoC · SSOT** — compute in `rust_dftb/src/{core,methods,qmqm}`, executables in `src/bin/`, scripts in `scripts/`; Fortran `src/dftbp/` is the parity reference, `OVERVIEW_Roadmap.md` the status tracker.
- **TDD** — define verification before coding; run tests after every change.
- **Fail Fast, Fix the Physics** — silent fallbacks strictly prohibited; fail loud with full context (values, indices, file:line); fix the solver, not the scenario. No undiagnosed `.unwrap_or`/`.ok()`, no `let _ =`, no broad catch-and-continue.
- Compact code, unlimited line length, short math symbol names.

## ⚑ Most-violated rules — the one-line version is not the whole rule; read the section

- **Tests are diagnostics, not the goal** (§2) — never make a test green by weakening physics, tolerances, or suppressing errors; a red test locates broken physics.
- **f32 is the architecture, not an inconvenience** (§3) — every iterative scheme needs an *enforced* restoring invariant; tolerances set to the MEASURED floor, then STOP; f64 only for cheap O(N) scalar decisions, never to widen bulk matrix arithmetic.
- **Dense GPU hot path invariants** (§4, manifest §14) — non-negotiable: **NO global atomics** — gather inputs, never scatter outputs (own output → gather inputs → write once); **ZERO allocation/kernel/program builds inside solver loops** — everything persistent from construction; **NO implicit fallbacks** (`Failed` is a valid result; recovery is an explicit driver call); **keep kernel launches and host syncs to a minimum** — fuse work sharing the same inputs, and don't synchronize with the host inside iterations.

## Never Do This — details: §2/§5/§7

- NEVER use `rm`, `sed -i`, `cat >`, `echo >>`, heredocs, or shell redirects to delete/modify files — use the `edit`/`write`/`read` tools.
- **Cascade bash tool: ALWAYS use Background=false**; never use `exit` in commands.
- Never delete/rearrange existing code or make unrelated aesthetic edits without explicit permission; comment out, don't delete.
- Never apply quick-fixes that hide root causes (hard-coded outputs, clamping divergence).
- Never reinvent existing functionality — inventory first (`CODEMAP.md`, Fortran `src/dftbp/`, reference repos: SPAMMM, FireCore, tblite).
- Never copy-paste between modules — extract a shared function in `rust_dftb/src/core/`.
- **Ask, don't Guess** — when unsure, ask the user.

## Working practices — details: §5/§7

- **Checkpointing:** after every significant step summarize what changed, what was verified, what remains. **Never mark "fixed"/"done" without explicit USER confirmation** — a code change is not proof.
- **Tests:** foreground execution, full unfiltered output (no `| tail`/`| head`/backgrounding); diagnostic prints with actual numbers; parity vs Fortran/analytical/invariants; levels L0 `cargo test` · L1 read `.out`/`.log` unfiltered · L2 human reviews `debug/*.png`.
- **Debug prints are gated, not deleted** (`log` macros/`RUST_LOG` or a `VERBOSE` const). Long runs print unbuffered progress. Every error/panic carries where + what + all relevant values.
- **Performance:** correctness → debuggability → performance, but designed for speed from the start; see `doc/prokop/AGENTS/guidelines/efficiency.md` (12 rules: three-tier data lifetime, no hot-loop allocation, verify library internals, benchmark `--release`, exploit structure, analytic derivatives, check units, warm starts).

## Repo policy — details: §10 (`CODEMAP.md` is the router)

- `debug/` — all debug artifacts; **never commit** it. `scripts/` — kept reusable scripts only. `doc/prokop/tasts/<task>/` — specs only, no artifacts. `tests/` — reference data only. Update `OVERVIEW_Roadmap.md` when status changes.
