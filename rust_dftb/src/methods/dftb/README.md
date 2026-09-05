# rust_dftb/src/methods/dftb/

DFTB Hamiltonian assembly, SK table interpolation, SCC, and forces — the core
of the DFTB method reimplementation.

- **sk_data.rs** — SK file I/O and shell integral extraction. `SkData` holds
  onsite parameters + pair tables. `eval_shell_integrals_into` evaluates the
  spline at distance r into stack buffers (zero allocation).
  `eval_shell_integrals_and_derivs_into` returns V(r) and dV/dr in one Hermite
  eval (no finite differences).
- **interpolation.rs** — Cubic Hermite spline on uniform grid.
  `EqGridTable::new` precomputes per-grid-point derivatives (4th-order central
  differences) at load time. `eval_into` is O(n_integ) — 4 FMAs per channel vs
  Neville's O(n²)=64. `eval_with_deriv_into` returns value + analytic derivative
  in one call. Tail region [last_grid_r, r_max] delegates to Neville
  `poly5_to_zero` for exact DFTB+ parity (the tail derivative is sensitive to
  the finite-difference order). Neville code retained as `eval_neville_into`
  for parity verification.
- **rotation.rs** — direction-cosines rotation of diatomic SK integrals into the
  molecular frame. `rotate_diatomic_block_into` writes H and S blocks in-place.
  `rotate_block_with_derivs_into` also returns dH/dR_a, dS/dR_a for all 3
  Cartesian directions using closed-form analytic formulas (ss, sp, ps, pp).
- **hamiltonian.rs** — `HamiltonianBuilder` assembles H0 and S. `build_non_scc`
  for the full matrix, `build_non_scc_sp_only` fast path for sp-only basis sets.
  `build_scc` runs the standalone SCC loop (builds template, gamma, solver,
  converges, extracts `SccResult`). Timing instrumentation via
  `RUST_DFTB_TIMING=1`.
- **gamma.rs** — `GammaTable::from_sk_data` (auto-extracts Hubbard U from onsite
  SK parameters). Exponential gamma model matching DFTB+.
- **forces.rs** — SCC forces: analytic SK derivatives (dH/dR, dS/dR via
  `rotate_block_with_derivs_into`), repulsive spline, gamma derivative.
  Finite-difference force path replaced by analytic derivatives (P7/P8/P12).
- **spline_resample.rs** — SK spline resampling for compression.
- **dftb_hamiltonian.cl** — OpenCL kernel for GPU H0/S assembly.
