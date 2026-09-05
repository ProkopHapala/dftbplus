---
type: TopicalAudit
title: SK Interpolation
tags: [topic, cross-language, interpolation, dftb]
---

# SK Interpolation

## Summary

Slater–Koster integral tables are stored on a uniform 1D grid (spacing `dr`,
typically 0.02 Bohr, ~500 points). At runtime, the SK integrals V(r) must be
evaluated at arbitrary interatomic distances r. The interpolation method
determines accuracy, performance, and whether analytic derivatives are
available for the force path.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Fortran | `src/dftbp/dftb/slakoeqgrid.F90` | active | 8-point Neville (`polyInterUniform`), `poly5ToZero` tail. Reference implementation. |
| Rust (CPU) | `rust_dftb/src/methods/dftb/interpolation.rs` | active | Cubic Hermite spline (`eval_hermite_into`), precomputed derivatives. Neville retained as `eval_neville_into` for parity + tail. |
| Rust (GPU) | `rust_dftb/src/methods/dftb/dftb_hamiltonian.cl` | active | Cubic B-spline (`cubic_interp_params`, `interp_sk_*_indexed`), host-resampled to ≤256 points. |

## Parity Status

| Pair | Tolerance | Test | Notes |
|------|-----------|------|-------|
| Rust Hermite vs Rust Neville (interior) | <1e-10 | `examples/test_hermite.rs` (temporary) | Exact match in interior [dr, last_grid_r] |
| Rust Hermite vs Rust Neville (tail) | <1.2e-7 | same | Tail delegates to Neville `poly5_to_zero` |
| Rust Neville vs Fortran Neville | — | `tests/parity_non_scc.rs` | Historical parity, 8-point Neville ported directly |
| Rust Hermite energy vs Neville energy | 7 sig figs | `examples/hbond_ref.rs` | -3.2112994345e1 vs -3.2112994161e1 (100 opt iters) |
| Rust Hermite forces vs Neville finite-diff | exact | `examples/hbond_ref.rs` | max\|F\|=4.7305, rms\|F\|=1.2098 (identical) |

## Methods Compared

### Neville 8-point polynomial (Fortran, Rust fallback)
- Degree-7 polynomial through 8 grid points
- O(n²) per evaluation (~64 FMAs per channel)
- No precomputation — every eval re-derives the polynomial
- Derivatives require 3 separate evaluations (r, r±dr) for finite differences
- Tail: `poly5_to_zero` with derivatives computed via 8-point polynomial finite differences

### Cubic Hermite spline (Rust CPU, current)
- Degree-3 polynomial per interval, C¹ continuous
- Precomputes f'(x_i) at each grid point at load time (4th-order central differences)
- O(1) per evaluation (4 FMAs per channel)
- Analytic derivative dV/dr in same call (`eval_hermite_with_deriv_into`) — no finite differences
- Tail [last_grid_r, r_max]: delegates to Neville `poly5_to_zero` (the tail derivative
  is sensitive to the finite-difference order; 2nd-order backward difference was
  insufficient — caused 0.5 Hartree energy error)

### Cubic B-spline (Rust GPU)
- Precomputed B-spline coefficients, host-resampled to ≤256 grid points
- GPU-friendly (uniform grid, local stencil)
- Used in `dftb_hamiltonian.cl` for GPU H0/S assembly

## Open Issues

- [ ] All-channel evaluation reuse (Rule 9 in `efficiency.md`): `rotate_diatomic_block_into`
  still calls `eval_shell_integrals_into` separately for ss, sp, ps, pp. Could
  evaluate all channels once per pair and reuse.
- [ ] Consider quintic Hermite (C², degree 5) if higher smoothness needed — would
  require 2nd derivatives in precomputation but only 2 extra FMAs per eval.
- [ ] GPU B-spline and CPU Hermite are different algorithms — cross-parity not
  formally verified (both match Fortran independently).
