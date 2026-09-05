---
type: SessionReport
title: H-Bond Optimization — FIRE Trajectory, SCC Convergence Analysis, LAPACK Eigensolver
tags: [hbond, optimization, fire, scc, lapack, performance]
timestamp: 2025-09-05
---

# H-Bond Optimization — FIRE Trajectory, SCC Convergence Analysis, LAPACK Eigensolver

## Context

Continuing work on the formic-acid/7-azaindole hydrogen-bond switching benchmark
(`doc/prokop/tasts/GPU_MultiSystem/hbond_switching.md`). The goal is to obtain
optimized reactant and product geometries, then interpolate a scan between them.

The immediate problem: geometry optimization was taking impractically long
(75+ seconds for 100 FIRE steps on a 20-atom / 56-orbital system), and the user
demanded diagnosis of the bottlenecks before running anything longer.

## What was done

### 1. SCC convergence analysis

Added verbose per-iteration output to `MultiSystemSolver::solve_scc`
(`rust_dftb/src/qmqm/solver.rs`), gated by `RUST_DFTB_SCC_VERBOSE=1`.

Ran a single SCC with `--max-iter 500 --tol 1e-12` and observed the full
convergence trajectory:

```
iter  0: RMS=3.2e-1   (initial q=0)
iter  5: RMS=6.2e-2   (linear, ~5x/iter)
iter 10: RMS=8.8e-5   (DIIS kicks in, superlinear)
iter 14: RMS=2.3e-7
iter 16: RMS=1.0e-8
iter 17: RMS=1.7e-9   ← best
iter 18: RMS=2.5e-9   ← BOUNCES BACK
iter 19-500: oscillates 1e-9 to 4e-8, never below ~5e-10
```

**Finding**: the DIIS mixer saturates at RMS ~1e-8. It hits ~1e-9 occasionally
but bounces back. This is the numerical floor of the mixer + diagonalization
precision for this system. Requesting `tol < 1e-7` is wasted effort — the solver
runs hundreds of iterations with no improvement.

**Corrected settings**: `--tol 1e-7 --max-iter 50`. SCC converges in ~14-17
iterations, well above the noise floor.

### 2. FIRE optimizer tuning

The original FIRE parameters were too aggressive (dt=0.1, dt_max=dt×20, vmax=1.0)
causing the timestep to collapse to ~0.009 after a few P<0 events. The optimizer
was crawling with sub-0.01 timesteps.

Tuned to: dt₀=1.0, dt_max=dt×5, vmax=2.0, f_dec=0.7 (was 0.5), n_min=10 (was 5).

Added a **max displacement cap** of 0.1 Å per atom per step. Without this, the
first step (max|F|=4.73) moved atoms too far and SCC diverged on the next
geometry.

### 3. Trajectory and history output

Modified `optimize_geometry` in `rust_dftb/examples/hbond_ref.rs` to save:
- Every FIRE step to a multi-frame XYZ trajectory (`reactant_traj.xyz`)
- Energy/force/dt history to CSV (`reactant_hist.csv`)

Plotting script: `debug/hbond_switching/plot_trajectory.py`
Output plot: `debug/hbond_switching/reactant_opt_trajectory.png`

### 4. Bottleneck diagnosis: nalgebra Jacobi eigensolver

Added timing instrumentation gated by `RUST_DFTB_TIMING=1` to:
- `build_scc` in `hamiltonian.rs` (template/frag/gamma/neigh/solver_init/scc breakdown)
- `diagonalize` in `fragment.rs` (chol/transform/eigen/back breakdown)

**Measured per SCC iteration (N=56):**
```
chol=0.9ms  transform=4.7ms  eigen=20.3ms  back=2.9ms  total=28.8ms
```

The eigensolve (`nalgebra::SymmetricEigen::new`) dominated at 20ms.
See `doc/prokop/topical_audit/eigensolver_performance.md` for the full analysis.

### 5. LAPACK eigensolver replacement

Added `lapack = "0.20"` and `openblas-src = { version = "0.10", features = ["system"] }`
to `rust_dftb/Cargo.toml`. Created `rust_dftb/build.rs` to link system OpenBLAS.

Replaced `SymmetricEigen::new(h_prime)` with LAPACK `dsyevd` (divide-and-conquer)
in `Fragment::diagonalize` (`rust_dftb/src/qmqm/fragment.rs`).

nalgebra DMatrix is column-major (Fortran order), so the raw data is passed
directly to LAPACK — no copy needed.

**Result:**
```
eigen: 20.3ms → 0.7ms  (29x faster)
total diag: 28.8ms → 8.5ms  (3.4x)
SCC loop: 500ms → 210ms  (2.4x)
FIRE step: 750ms → 320ms  (2.3x)
100 steps: 75s → 33s  (2.3x)
```

Energy and trajectory are identical to the nalgebra version — LAPACK gives the
same results, just faster.

### 6. Reactant optimization result

100 FIRE steps, 32.8 seconds:
- E: -29.77 → -32.11 Hartree (ΔE = -2.34)
- max|F|: 4.73 → 0.049 (did NOT reach f_tol=5e-3, ended ~10x above)
- Major geometry rearrangement at iter 43-55 (E drops 1 Hartree)
- Trajectory: `debug/hbond_switching/reactant_traj.xyz` (100 frames)
- Final geometry: `debug/hbond_switching/reactant_opt.xyz`

## How to run

```bash
# From rust_dftb/ directory:
RUST_DFTB_SK_DIR=/home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1 \
CARGO_TARGET_DIR=/home/prokophapala/.cargo-target-shared \
cargo run --example hbond_ref -- \
  --xyz ../data/xyz/formic_azaindole_dimer.xyz \
  --mode optimize \
  --h1 14 --donor1 6 --acceptor1 17 \
  --h2 19 --donor2 18 --acceptor2 1 \
  --out ../debug/hbond_switching/reactant_opt.xyz \
  --data-dir ../debug/hbond_switching \
  --opt-max-iter 100 --opt-f-tol 5e-3 \
  --max-iter 50 --tol 1e-7

# With timing breakdown:
RUST_DFTB_TIMING=1  # shows per-SCC-iteration timing
RUST_DFTB_SCC_VERBOSE=1  # shows per-SCC-iteration RMS

# Plot trajectory after:
cd /home/prokophapala/git/dftbplus
python3 debug/hbond_switching/plot_trajectory.py
```

## Challenges

1. **Absurd SCC tolerances** — previous runs used `tol=1e-10` or `1e-12`, far
   below the DIIS mixer's numerical floor of ~1e-8. The solver ran 500 iterations
   with no convergence. Fixed by using `tol=1e-7`.

2. **FIRE timestep collapse** — the optimizer's adaptive dt kept shrinking to
   ~0.009 because of aggressive f_dec=0.5 and no displacement cap. Fixed by
   tuning parameters and adding a 0.1 Å max displacement per atom.

3. **First-step blow-up** — with dt=1.0 and max|F|=4.73, the first step moved
   atoms into a geometry where SCC diverged (RMS=0.64 after 50 iterations).
   Fixed by the displacement cap.

4. **nalgebra Jacobi eigensolver** — 20ms for a 56×56 matrix, dominating the
   SCC loop. Replaced with LAPACK dsyevd (0.7ms). See
   `doc/prokop/topical_audit/eigensolver_performance.md`.

5. **No SCC warm start** — every FIRE step starts SCC from q=0, requiring 16-17
   iterations. With warm start from previous charges, this would be 3-4
   iterations. **Not yet implemented** — see optimization plan in
   `doc/prokop/topical_audit/eigensolver_performance.md`.

## Remaining bottlenecks

After the LAPACK fix, per FIRE step (320ms):
- Template build (H0, S): 8ms — fine
- SCC loop (16 iter × 13ms): 210ms — still dominated by nalgebra triangular solves
  (transform=4.8ms + back=2.8ms per SCC iteration)
- Forces: 100ms — repulsive spline derivative evaluation

Next optimization targets (see `eigensolver_performance.md`):
1. SCC warm start → 4x fewer iterations → ~80ms/step
2. Replace nalgebra triangular solves with LAPACK (dtrtrs) → ~5ms/step saved
3. Cache SystemContext/GammaTable across FIRE steps → 8ms/step saved

## Files modified

- `rust_dftb/Cargo.toml` — added lapack + openblas-src
- `rust_dftb/build.rs` — new, links system OpenBLAS
- `rust_dftb/src/qmqm/fragment.rs` — LAPACK dsyevd in diagonalize(), timing instrumentation
- `rust_dftb/src/qmqm/solver.rs` — verbose SCC output (RUST_DFTB_SCC_VERBOSE)
- `rust_dftb/src/methods/dftb/hamiltonian.rs` — timing in build_scc (RUST_DFTB_TIMING)
- `rust_dftb/examples/hbond_ref.rs` — FIRE tuning, trajectory/history output, displacement cap

## Cross-references

- Task spec: `doc/prokop/tasts/GPU_MultiSystem/hbond_switching.md`
- Performance analysis: `doc/prokop/topical_audit/eigensolver_performance.md`
- Trajectory plot: `debug/hbond_switching/reactant_opt_trajectory.png`
- Roadmap: `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md`
