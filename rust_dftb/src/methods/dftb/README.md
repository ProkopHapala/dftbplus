# rust_dftb/src/methods/dftb/

DFTB Hamiltonian assembly, SK table interpolation, SCC, and forces — the core
of the DFTB method reimplementation.

- **sk_data.rs** — SK file I/O and shell integral extraction. `SkData` holds
  onsite parameters + pair tables. `eval_shell_integrals_into` / `_and_derivs_into`
  evaluate the production B-spline (V and analytic dV/dr) into stack buffers.
- **interpolation.rs** — Production: C² cubic B-spline, analytic V' from the
  same controls. Left end: phantom `c_{-1}=2c_0−c_1`. Right end (stopgap):
  `N_PAD_END` blunt zero *samples*, then refit — kills the old Neville
  `poly5_to_zero` explosion (H–H 10.4 Bohr → −0.4 Ha). **Next (not done):**
  general extra-control fitter that *solves* for pad points so the valid-domain
  polynomial is preserved and V,V'→0 at cutoff; see
  `doc/prokop/topical_audit/sk_interpolation.md`. Hermite/Neville kept as
  unused reference paths.
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
