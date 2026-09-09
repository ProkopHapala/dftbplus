---
type: TopicalAudit
title: SK Interpolation
tags: [topic, cross-language, interpolation, dftb, bspline, boundary-conditions]
timestamp: 2026-09-09
---

# SK Interpolation

## Summary

Slater–Koster tables live on a uniform 1D grid (`dr` typically 0.02 Bohr,
~500 points). Production evaluation is a **C² cubic B-spline** on both CPU
(f64, reference) and GPU (f32). Analytic `dV/dr` is the derivative of that
same spline — not a finite difference, not a second interpolant.

The old 8-point Neville + `poly5_to_zero` tail is **not** the production
path. On a nearly-flat SK tail (~1e-5 Ha at last H–H grid point) that
polynomial exploded to **−0.4 Ha** at 10.39 Bohr. GPU already clamped to ~0.
AT/GC “H assembly failed” was that CPU garbage, not a GPU assembly bug.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Fortran | `src/dftbp/dftb/slakoeqgrid.F90` | reference | 8-point Neville + `poly5ToZero` over `distFudge=1` Bohr. Physical reference for *interior* SK; the tail polynomial is **not** a physics we copy. |
| Rust CPU f64 | `rust_dftb/src/methods/dftb/interpolation.rs` | active (stopgap BC) | Production: `eval_into` / `eval_with_deriv_into` → B-spline. Hermite and Neville kept as unused reference paths. |
| Rust fit | `rust_dftb/src/methods/dftb/spline_resample.rs` | active | `function_to_bspline_control_points`, `fit_bspline_controls_zero_end`, `bspline3_eval_v_d1_d2`. |
| GPU f32 | `dftb_hamiltonian.cl`, `gpu_forces.cl`, packed in `qmqm/gpu_prep.rs` | active (same stopgap) | Same 4-point stencil + analytic `cubic_weights_d1`. Dense H-bond tables: original grid + r=0 dummy + `N_PAD_END` (fits `SK_GRID_MAX=512` for mio). |

## What was done (2026-09-09) — stopgap, not the design

Two different end treatments. Do not confuse them.

**Left end (the better *kind* of thing):** at evaluation, a **phantom control**
`c_{-1} = 2 c_0 − c_1` implements the natural-spline condition `V''=0` at the
first tabulated knot. SK values at small `r` are large — we do **not** pad
zeros on the left.

**Right end (blunt stopgap):** append `N_PAD_END=4` **function samples of
exact 0** after the last SK point, then refit the whole tridiagonal
(`fit_bspline_controls_zero_end`). That interpolates zeros in the pad and
killed the Neville explosion. It is **not** an optimal extra-control fit:

- extra knots are hardcoded 0, not solved for;
- the global tridiagonal couples those zeros back into the last original
  samples;
- cutoff shrinks from DFTB+ `last_grid + 1 Bohr` to `last_grid + 4·dr`
  (~0.08 Bohr). Distant pairs (AT H–H at 10.39 Bohr) are now exact 0, which
  is physical for a ~1e-5 table, but the *method* is blunt.

GPU packing prepends a dummy 0 at `r=0` for 0-based indexing. That dummy is
also blunt, not a fitted left control.

Neville / `poly5_to_zero` / `DIST_FUDGE` remain in the file, unused in
production. Do not restore them for “Fortran tail parity”.

## What should be done — extra controls as a general BC fitter

Cubic B-splines need a few extra control points **before and after** the
tabulated domain to implement boundary conditions. Those points must be
**computed**, not bluntly zeroed.

Need one general fitter (CPU f64, then pack the same controls to GPU f32):

1. **Inputs:** original samples on the valid grid; optional extra conditions
   (values and/or derivatives at selected knots).
2. **Unknowns:** a small number of extra controls left of sample 0 and right
   of the last sample (2–4 per side is enough for a 4-point stencil).
3. **Solve** (linear / least-squares) so that:
   - the interpolant on the **valid domain** reproduces the original samples
     (or the unconstrained spline) accurately — extra points must not pollute
     the interior polynomial;
   - `V`, `V'` (and preferably `V''`) are continuous at the first/last
     tabulated knot;
   - at cutoff: `V → 0`, `V' → 0` (optionally `V'' → 0`);
   - left end: **not** `V=0`. Phantom `c_{-1}=2c_0−c_1` is one valid
     condition; fitted left controls are the generalization.
4. Cutoff length is then a consequence of how far the extra right controls
   go, not a 1 Bohr polynomial fudge.

The phantom-knot formula is the pattern to generalize. The zero-sample pad
is only a temporary way to stop the tail explosion.

## Parity Status (measured 2026-09-09, NVIDIA RTX 3090, mio-1-1)

CPU f64 is the reference. GPU f32 must match it. Tests:
`cargo test --lib interpolation` and
`RUST_DFTB_SK_DIR=.../mio-1-1 cargo test --test gpu_hbond_physics`.

| Check | Result |
|-------|--------|
| B-spline reproduces grid values | pass (`<1e-10`) |
| Analytic V' vs FD of same V (synthetic) | max rel `1.5e-8` |
| Real H–H SK: analytic dHss/dr vs FD | max rel `8.6e-10` |
| H–H at 10.00 Bohr (in zero-pad) | Hss `~2e-22` (was `−0.4`) |
| H–H at 10.39 Bohr (AT pair, past cutoff) | exact 0 |
| AT / GC / H2O H/S GPU vs CPU | max\|dH\| `8.6e-8` / `7.4e-8` / `4.4e-8` |
| H2O CPU analytic F vs FD of E (h=1e-3 Å) | rel `1.05e-5` |
| H2O four GPU force kernels vs CPU | rel `~3e-5` |

Do not chase Fortran Neville-tail values past last grid. That tail is the
bug we replaced.

## Open Issues

- [ ] **Replace blunt zero-sample pad with the general extra-control fitter**
  described above. Shared CPU/GPU controls. This is the next interpolator
  work — do not re-Neville, do not restore `DIST_FUDGE` poly5.
- [ ] GPU dummy at `r=0` should become a fitted left control (or stay a
  pure index shift with the phantom applied in the kernel), not a stored 0
  that the stencil can eat at small `r`.
- [ ] All-channel evaluation reuse (Rule 9 in `efficiency.md`):
  `rotate_diatomic_block_into` still interpolates per shell pair.
- AT/GC GPU SCC charge-rms plateaus `~1e-5` with matching H/S — **not this
  interpolator.** Hypothesis: f32 floor (`100 × 1e-8` → `1e-6`), see H-bond
  manifest §3.0.1. Do not chase `<1e-6` as a mixer bug until the Hamiltonian
  scale is checked.

## Related

- `/doc/prokop/tasts/HBond_Relaxed_Scan_GPU/HBond_Relaxed_Scan_GPU.manifest..md`
- `/doc/prokop/AGENTS/guidelines/efficiency.md` (Rule 8)
- `/rust_dftb/tests/gpu_hbond_physics.rs`
- Fortran: `src/dftbp/math/interpolation.F90`, `src/dftbp/dftb/slakoeqgrid.F90`
