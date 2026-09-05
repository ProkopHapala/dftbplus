# rust_dftb/src/methods/dftb/

DFTB Hamiltonian assembly, SK table interpolation, SCC, and forces — the core
of the DFTB method reimplementation.

- **sk_data.rs** — SK file I/O and shell integral extraction. `SkData` holds
  onsite parameters + pair tables. `eval_shell_integrals_into` evaluates the
  spline at distance r into stack buffers (zero allocation).
- **interpolation.rs** — Neville 8-point interpolation on uniform grid.
  `EqGridTable::eval_into` matches DFTB+'s `poly_inter_uniform` with hard cutoff
  at rMax and poly5ToZero tail extrapolation.
- **rotation.rs** — direction-cosines rotation of diatomic SK integrals into the
  molecular frame. `rotate_diatomic_block_into` writes H and S blocks in-place.
- **hamiltonian.rs** — `HamiltonianBuilder` assembles H0 and S. `build_non_scc`
  for the full matrix, `build_non_scc_sp_only` fast path for sp-only basis sets.
  `build_scc` runs the standalone SCC loop (builds template, gamma, solver,
  converges, extracts `SccResult`). Timing instrumentation via
  `RUST_DFTB_TIMING=1`.
- **gamma.rs** — `GammaTable::from_sk_data` (auto-extracts Hubbard U from onsite
  SK parameters). Exponential gamma model matching DFTB+.
- **forces.rs** — SCC forces (finite-difference dH0/dx, dS/dx, repulsive spline,
  gamma derivative). **Not yet fully implemented** — see roadmap §1.3.
- **spline_resample.rs** — SK spline resampling for compression.
- **dftb_hamiltonian.cl** — OpenCL kernel for GPU H0/S assembly.
