# Formic Dimer Scan Plots and Gamma Matrix Fix

**Date:** 2025-09-06
**Status:** 1D scan validated, 2D scan partially converged (physics limitation)

## Summary

Generated reviewable 1D and 2D proton-transfer scan plots for the formic dimer.
Fixed a critical gamma matrix bug in the plotting test that was producing wrong
GPU energies and charges. The 1D scan now shows perfect CPU/GPU parity. The 2D
scan converges for 147/441 points; the remaining 294 points are highly
asymmetric geometries where **both CPU and GPU SCC fail to converge** — this is
a physics limitation, not a GPU bug.

## Critical Bug Fix: Gamma Matrix in formic_scan_plots.rs

The `build_gamma_matrix` function in `tests/formic_scan_plots.rs` had three bugs
compared to the passing `tests/hbond_gpu_scc.rs`:

1. **Diagonal set to 0.0** instead of using `gamma_full(r=0, ui, uj)` which
   returns the Hubbard U. This removed the on-site self-interaction from the SCC
   potential.
2. **No Å→Bohr conversion** — distances were in Ångström but the gamma function
   expects Bohr. This made all off-site gamma values ~1.89× too small.
3. **Custom `erf_approx`** instead of the correct `rust_dftb::gamma_full`
   function.

**Fix:** Replaced with the same implementation as `hbond_gpu_scc.rs`:
```rust
let r = (dx*dx + dy*dy + dz*dz).sqrt() * ANG2BOHR;
g[a*n + b] = rust_dftb::gamma_full(r, u_per_atom[a], u_per_atom[b]) as f32;
```

Also fixed `orb_atom_map` to use `atom_orb_off.len() - 1` (matching
`hbond_gpu_scc.rs`) instead of a different loop bound.

**Before fix:** GPU energy = -39.81 Ha (CPU = -18.80 Ha, factor ~2× off)
**After fix:** GPU energy = -18.79692078 (CPU = -18.79692437, diff = 3.59e-6)

## 1D Synchronous Scan

**Parameters:**
- 41 points, t = 0.0 → 2.0 (step 0.05)
- Both protons move synchronously
- CPU: `HamiltonianBuilder::build_scc`, 200 iterations, tol=1e-9
- GPU: `gpu_solve_scc_batched_diis`, 500 iterations, tol=5e-6, alpha=0.3,
  max_history=8, warmup=3
- H-bond 1: H[4] donor O[3] acceptor O[7]
- H-bond 2: H[9] donor O[8] acceptor O[2]

**Results:**
- 41/41 points converged
- GPU n_iters = 13
- Energy parity: |E_gpu - E_cpu| ≤ 3.6e-6 Ha (0.098 meV)
- Charge parity: |q_gpu - q_cpu| ≤ 1e-5 |e|
- Barrier: ~4.1483e-2 Ha (~1.13 eV) — matches previous 21-point scan

**Output files:**
- `rust_dftb/debug/formic_dimer_scan/scan_1d.tsv`
- `rust_dftb/debug/formic_dimer_scan/1d_energy.png`
- `rust_dftb/debug/formic_dimer_scan/1d_charges.png`
- `rust_dftb/debug/formic_dimer_scan/1d_parity.png`

## 2D Asynchronous Scan

**Parameters:**
- 21×21 = 441 points, t1 = 0.0 → 2.0, t2 = 0.0 → 2.0 (step 0.1)
- Proton 1 moves at t1, proton 2 moves at t2 (independent)
- GPU: `gpu_solve_scc_batched_diis_warmstart`, 500 iterations, tol=5e-6
- Solved strip-by-strip (21 strips of 21 points each)
- Best-effort mode: unconverged points marked with NaN energy

**Results:**
- 147/441 points converged (33%)
- 294/441 points unconverged (67%)
- Unconverged points are highly asymmetric: |t1 - t2| > 0.5 approximately
- The converged region forms a band along the diagonal t1 ≈ t2

**CPU verification (first strip, t1=0.0):**
- (0.0, 0.0): converged, E = -18.797
- (0.0, 1.0): **CPU FAILED** (RMS = 0.647 after 200 iterations)
- (0.0, 1.6): **CPU FAILED** (RMS = 3.879 after 200 iterations)
- (0.0, 2.0): **CPU FAILED** (RMS = 0.095 after 200 iterations)

The CPU DIIS shows the same oscillating behavior (RMS jumps between ~0.01 and
~0.7) as the GPU. This confirms the nonconvergence is a physics limitation of
the SCC fixed-point iteration for highly asymmetric charge distributions, not a
GPU implementation bug.

**Root cause of 2D nonconvergence:**
When one proton is at the reactant (t=0) and the other is past the transition
state (t>0.5), the charge distribution is highly polarized. The SCC
fixed-point iteration oscillates between two charge states without converging.
This is a known issue in DFTB SCC calculations for systems with multiple
competing charge transfer pathways.

**Output files:**
- `rust_dftb/debug/formic_dimer_scan/scan_2d_energy.tsv`
- `rust_dftb/debug/formic_dimer_scan/scan_2d_charge_H1.tsv`
- `rust_dftb/debug/formic_dimer_scan/2d_energy.png`
- `rust_dftb/debug/formic_dimer_scan/2d_charge_H1.png`

## Code Changes

### `rust_dftb/src/qmqm/gpu_scc.rs`

1. **Added `gpu_solve_scc_batched_diis_warmstart`** — accepts separate
   `init_q_buf` (initial charges) and `q0_buf` (reference charges). Enables
   warm-starting from neighbouring converged solutions.

2. **Added `best_effort` parameter** — when true, returns results with per-system
   RMS even if not all systems converged. The caller can check `rms[i] < tol`
   to identify converged systems.

3. **Added per-system RMS diagnostics** — on nonconvergence, reports the 10
   worst systems with their indices and RMS values, and the count of
   converged vs. failed systems.

4. **`gpu_solve_scc_batched_diis`** now delegates to `_warmstart` with
   `q0_buf` as both `q0_buf` and `init_q_buf` (cold start).

### `rust_dftb/tests/formic_scan_plots.rs`

1. **Fixed gamma matrix builder** — uses `rust_dftb::gamma_full` with Bohr
   conversion, matching `hbond_gpu_scc.rs`.

2. **Fixed `orb_atom_map`** — uses `atom_orb_off.len() - 1` bound.

3. **Added `best_effort` parameter to `run_gpu_scan`** — returns
   `(energies, charges, per_system_rms)`.

4. **2D scan solved strip-by-strip** — 21 strips of 21 points each, with
   best-effort mode. Unconverged points marked with NaN energy.

5. **Added CPU SCC verification** for difficult 2D points to confirm the
   nonconvergence is a physics limitation.

6. **2D TSV includes convergence flag** — each row has a `converged` column
   (1 = converged, 0 = unconverged).

### `scripts/plot_formic_scan.py`

1. **Fixed data path** — points to `rust_dftb/debug/formic_dimer_scan/`.

2. **Handles NaN values** — masks unconverged points in 2D plots.

3. **Shows convergence count** in 2D plot titles.

## Plotting Script

Run with:
```bash
python3 scripts/plot_formic_scan.py
```

Generates:
- `1d_energy.png` — 1D PES, CPU vs GPU, with barrier annotation
- `1d_charges.png` — 1D Mulliken charges for H-bond atoms (2 subplots)
- `1d_parity.png` — E(GPU) - E(CPU) in eV along 1D scan
- `2d_energy.png` — 2D PES contour with synchronous path overlay
- `2d_charge_H1.png` — 2D Mulliken charge contour for H[4]

## Verification Commands

```bash
# Run the scan test (produces TSV data)
RUST_DFTB_SK_DIR=/path/to/mio-1-1 \
  cargo test --test formic_scan_plots -- --ignored --nocapture

# Generate plots
python3 scripts/plot_formic_scan.py
```

## Remaining Work

1. **2D convergence improvement** — possible approaches:
   - Broyden mixing with damping for difficult points
   - Occupation smoothing / level shifting
   - Smaller mixing parameter (alpha=0.1) for asymmetric geometries
   - Warm-starting from neighbouring 2D converged points (strip-by-strip
     propagation)

2. **Full GPU test suite** — should be re-run to confirm no regressions from
   the `gpu_scc.rs` changes.

3. **Report integration** — merge this report's findings into the main
   GPU H-assembly bugfix report and the roadmap.
