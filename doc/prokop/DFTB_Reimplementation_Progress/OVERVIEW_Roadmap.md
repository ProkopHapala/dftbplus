# rust_dftb — Master Roadmap & Status Checklist

**Last updated:** 2026-09-09
**Maintained by:** prokop / Devin
**Purpose:** Single source of truth for what is done (`[*]`) and what is not (`[ ]`)
across the whole `rust_dftb` reimplementation (DFTB, xTB, QM/QM multi-system, OpenCL GPU).

Legend: `[*]` done & verified · `[~]` implemented but unverified / partial · `[ ]` not done

> **Active parallel task plan:** `doc/prokop/tasts/GPU_MultiSystem/task_master.md`
> — 5 agents, 2 waves. **Wave 1: ACCEPTED** (Agent_1 GPU driver, Agent_2 CPU multi-frag,
> Agent_3 forces — all verified by coordinator). **Wave 2: ready to dispatch** (Agent_4 GPU
> SCC, Agent_5 scan/NEB — independent, launch simultaneously).
>
> **Architecture revised (2026-09-06):** see `GPU_MultiSystem_Design.md` — host-orchestrated
> device-resident SCC (no PCIe traffic), Brent-Luk parallel cyclic Jacobi, S^{-1/2} on GPU,
> f32 only, System vs Fragment distinction, homogeneous templates, H0/S separated from SCC
> inner loop. Decisions D1–D7 revised; D8–D17 added.

Each bullet names the **file / function / test** that implements or verifies it
(format: `file.rs::fn` or `tests/x.rs::test_fn`). `[ ]` items name the **target**
file/function where the work should land.

---

## 1. DFTB — CPU Hamiltonian & SCC

### 1.1 Non-SCC H0 / S assembly
- [*] SK file I/O & shell integral extraction — `methods/dftb/sk_data.rs::load_sk_folder`, `::eval_shell_integrals_into`, `::onsite`
- [*] Cubic B-spline interpolation on uniform grid (production; replaces Neville) — `methods/dftb/interpolation.rs::EqGridTable::eval_into`, `::eval_with_deriv_into`. Analytic V' from the same controls (`spline_resample.rs::bspline3_eval_v_d1_d2`). Left: phantom `c_{-1}=2c_0−c_1`. Right (stopgap): `N_PAD_END=4` blunt zero *samples* then refit (`fit_bspline_controls_zero_end`) — kills unphysical Neville `poly5_to_zero` (H–H 10.4 Bohr was −0.4 Ha). Hermite/Neville kept unused.
- [ ] **Extra-control BC fitter** (do not blunt-zero pad) — extra controls before/after the table must be *solved* so the valid-domain interpolant stays accurate and V,V'→0 at cutoff. See `doc/prokop/topical_audit/sk_interpolation.md`.
- [*] Neville 8-point retained as unused parity reference — `interpolation.rs::eval_eqgrid_new_into`, `EqGridTable::eval_neville_into` (not production)
- [*] `r_max` hard cutoff — now `(n_data + N_PAD_END)*dr`, not `n_grid*dr + DIST_FUDGE`. Fortran 1 Bohr fudge is intentionally not copied.
- [*] Diatomic rotation matrices, direction cosines — `methods/dftb/rotation.rs::Rotation::rotate_diatomic_block_into`, `DirectionCosines::from_vec`
- [*] Generic `build_non_scc()` (s,p,d) — `methods/dftb/hamiltonian.rs::HamiltonianBuilder::build_non_scc`, `fill_pairs`, `fill_onsite`
- [*] Hand-unrolled `build_non_scc_sp_only()` fast path — `methods/dftb/hamiltonian.rs::build_non_scc_sp_only`, `fill_pairs_sp_only`, `fill_onsite_sp_only`
- [*] Parity vs Fortran DFTB+ to `<1e-8` (H0, S) — `tests/parity_non_scc.rs::parity_h0_methane_example`, `parity_sp_only_vs_generic`; `tests/parity_universal.rs::parity_universal_from_env` (13 mols via `tests/run_parity.py`)

### 1.2 SCC (self-consistent charges)
- [*] `GammaTable::from_sk_data()` (auto-extract Hubbard U) — `methods/dftb/gamma.rs::GammaTable::from_sk_data`; `qmqm/gamma.rs::GammaTable::from_sk_data`
- [*] `HamiltonianBuilder::build_scc()` standalone API — `methods/dftb/hamiltonian.rs::HamiltonianBuilder::build_scc` (returns `SccResult`)
- [*] Full SCC loop: H_scc → diag → Mulliken → mixer → converge — `qmqm/solver.rs::MultiSystemSolver::solve_scc`; per-frag `qmqm/fragment.rs::Fragment::diagonalize`, `::compute_charges`, `::build_h_scc`
- [*] DIIS mixer with 5-iter simple-mixing warmup — `qmqm/mixer.rs::DiisMixer::mix`, `SimpleMixer::mix`
- [*] Cholesky factorization caching across SCC iters — `qmqm/fragment.rs::Fragment::diagonalize` (uses `cholesky_l` field)
- [*] Diagonal-only Mulliken charges (O(N²·n_occ)) — `qmqm/fragment.rs::Fragment::compute_charges`
- [*] SCC total energy matching DFTB+ "Total Electronic energy" — `methods/dftb/hamiltonian.rs::SccResult` (energy field), computed in `build_scc`
- [*] Parity on 13 molecules (H2O … PTCDA, 3–38 atoms) to `<1e-6` — `tests/parity_scc.rs::scc_convergence_from_xyz` (driven by `tests/run_scc_full.py`)
- [*] LAPACK `dsyevd` eigensolver (replaces nalgebra Jacobi, 29× faster) — `qmqm/fragment.rs::Fragment::diagonalize`; see `doc/prokop/topical_audit/eigensolver_performance.md`
- [ ] **SCC warm start** (reuse previous charges → 4× fewer iterations) — target: `methods/dftb/hamiltonian.rs::build_scc_warm`; see `eigensolver_performance.md` §Optimization plan
- [ ] **LAPACK triangular solves** (replace nalgebra `solve_lower_triangular` with `dtrtrs`) — target: `qmqm/fragment.rs::diagonalize`; see `eigensolver_performance.md`

### 1.3 Forces  ← `Forces_Implementation_Notes.md`
- [*] Density matrix DM exposed in `SccResult` — `methods/dftb/hamiltonian.rs::SccResult`
- [*] Energy-weighted density matrix EDM — `methods/dftb/hamiltonian.rs::build_scc`
- [*] Analytic dH0/dx, dS/dx — `rotation.rs::rotate_block_with_derivs_into` + production `interpolation.rs::eval_with_deriv_into` (B-spline analytic V', not Hermite). H2O CPU F vs FD of energy rel `1.05e-5` (`tests/gpu_hbond_physics.rs::test_cpu_energy_gradient_h2o`).
- [*] Repulsive spline parsing from SK file — `methods/dftb/sk_data.rs::read_skf_all`
- [*] F_nonSCC = 2·(DM·dH0' − EDM·dS') — `methods/dftb/forces.rs::non_scc_forces`
- [*] F_rep = dE_rep/dr · r_hat — `methods/dftb/forces.rs::repulsive_forces`
- [*] `gamma_prime_full(r, U1, U2)` — `methods/dftb/gamma.rs`
- [*] F_SCC_dc (gamma + 1/R Coulomb) — `methods/dftb/forces.rs::scc_dc_forces`
- [*] F_SCC_shift (Pulay-like) — `methods/dftb/forces.rs::scc_shift_forces`
- [*] `tests/parity_forces.rs` + `tests/run_forces.py` — force parity vs Fortran
- [*] Non-SCC force parity vs Fortran — `tests/parity_forces.rs::non_scc_forces_from_xyz`
- [*] Full SCC force parity vs Fortran — `tests/parity_forces.rs::scc_forces_from_xyz`
- [*] Analytic force parity vs finite-difference baseline: max|F|=4.7305 exact match, energy 7 sig figs — `examples/hbond_ref.rs`

---

## 2. xTB (GFN1 / GFN2)  ← `xTB_reimplementation.md`

### 2.1 GFN1
- [*] Slater-type orbital → Gaussian expansion → Cartesian integrals — `methods/xtb/basis.rs::build_element_cgtos`, `methods/xtb/integrals.rs::overlap_cgto`, `build_overlap_matrix`
- [*] Non-SCC H0 / S — `methods/xtb/hamiltonian.rs::build_h0_s`, `compute_cn_gfn1`
- [*] Coulomb model (Klopman-Ohno `effective_coulomb`) — `methods/xtb/coulomb.rs::build_coulomb_matrix`, `build_shell_hubbard_matrix`
- [*] Mulliken shell charges — `methods/xtb/mulliken.rs::shell_charges`, `atomic_charges`, `reference_shell_occupations`
- [*] SCF potential (`add_vao_to_h1`) — `methods/xtb/scf.rs::build_scc_hamiltonian`, `build_scc_hamiltonian_with_thirdorder`
- [*] SCC loop (simple linear mixer) — `methods/xtb/scf.rs::run_scf`, `solve_gevp`
- [*] Parity vs `tblite` for H2, N2, HCOOH — `tests/xtb_parity.rs::test_h2_gfn1_parity`, `test_n2_gfn1_parity`, `test_hcooh_gfn1_parity`, `test_h2_scc_parity`, `test_n2_scc_parity`, `test_hcooh_scc_parity` (via `tests/tblite_helper.c`)
- [ ] Fermi smearing (kt > 0) — target: `methods/xtb/scf.rs::run_scf` (add Fermi occupation update)
- [ ] Repulsion energy — target: `methods/xtb/repulsion.rs` (new, mirror `xtb/repulsion.f90`)
- [ ] Halogen bonding — target: `methods/xtb/halogen.rs` (new)
- [ ] Dispersion (D3/D4 — explicitly deferred) — target: `methods/xtb/dispersion.rs` (new)
- [ ] Full element coverage (only H–O hardcoded) — target: `methods/xtb/params.rs`, `methods/xtb/basis.rs::build_element_cgtos`

### 2.2 GFN2  ← `INTEGRAL_DEBUG_NOTES.md`
- [*] Shell-resolved Coulomb — `methods/xtb/coulomb.rs::build_coulomb_matrix_gfn2`, `build_shell_hubbard_matrix_gfn2`
- [*] Multipole integrals (dipole/quadrupole) — `methods/xtb/multipole_integrals.rs::build_cgto_basis`; `methods/xtb/scf.rs::build_multipole_interaction_matrices_0d`, `compute_multipole_potentials`, `add_multipole_to_h1`
- [*] CN-dependent damping — `methods/xtb/hamiltonian.rs::compute_cn_gfn2`; `methods/xtb/scf.rs::compute_coordination_numbers`, `compute_multipole_radii`
- [~] Parity: H2 PASS; N2 eigenvalues 0.36% off; HCOOH ~0.03% off — `tests/xtb_parity.rs::test_h2_gfn2_parity`, `test_n2_gfn2_parity`, `test_hcooh_gfn2_parity`, `test_h2_gfn2_scc_hamiltonian_parity`, `test_n2_gfn2_scc_parity`, `test_hcooh_gfn2_scc_parity`, `test_n_dipole_integrals_parity`, `test_single_dipole_integral_h2`
- [ ] Resolve N2 H1 residual (1.1e-4 at (2,2)) — target: `methods/xtb/coulomb.rs::build_coulomb_matrix_gfn2` / `scf.rs::build_scc_hamiltonian_with_thirdorder_gfn2`
- [ ] Resolve HCOOH H1 residual (9.1e-4) — target: same as above
- [ ] Suspected: gamma-matrix / third-order / integral cutoff — debug in `methods/xtb/coulomb.rs::thirdorder_potential_gfn2`, `multipole_integrals.rs`

---

## 3. QM/QM Multi-System Solver (CPU)  ← `rust_QMQM.md`

### 3.1 Core infrastructure
- [*] `Fragment` / `FragmentTemplate` encapsulation — `qmqm/fragment.rs::Fragment`, `FragmentTemplate::new`, `Fragment::from_template`
- [*] `MultiSystemSolver` with flattened global charge vector — `qmqm/solver.rs::MultiSystemSolver::new`
- [*] Zero-allocation hot SCC loop — `qmqm/solver.rs::solve_scc` (pre-allocated `charges`, `v_ext`, `q_out`, `residual`)
- [*] `FragmentNeighborList` (cell-list, O(N) centroid neighbor finding) — `qmqm/neighbor.rs::FragmentNeighborList::build`, `::of_frag`
- [*] `compute_v_ext()` — inter-fragment gamma polarization — `qmqm/solver.rs::MultiSystemSolver::compute_v_ext`
- [*] `build_all_h_scc()` — intra-fragment shifts + H_scc — `qmqm/solver.rs::build_all_h_scc`; `qmqm/shifts.rs::compute_intra_shifts`, `compute_intra_shift_atom`; `qmqm/fragment.rs::Fragment::build_h_scc`
- [*] `gather_charges` / `scatter_charges` — `qmqm/solver.rs::gather_charges`, `scatter_charges`
- [*] Simple + DIIS mixers — `qmqm/mixer.rs::SimpleMixer`, `DiisMixer`

### 3.2 Multi-fragment validation  ⚠️ CRITICAL GAP
- [*] Single-fragment parity (frag == full-system) for H2, N2, HCOOH (non-SCC) — `tests/qmqm_integration.rs::fragment_h2_matches_full_system_non_scc`, `fragment_n2_matches_full_system_non_scc`, `fragment_hcooh_matches_full_system_non_scc`
- [*] Single-fragment fixed-charge SCC parity — `tests/qmqm_integration.rs::h2_fixed_charge_scc_parity`, `n2_fixed_charge_scc_parity`, `hcooh_fixed_charge_scc_parity`
- [*] Single-fragment diagonalization / neutral charges — `tests/qmqm_integration.rs::fragment_h2_diagonalization`, `fragment_h2_fixed_neutral_charges`
- [*] Gamma self-consistency — `tests/qmqm_integration.rs::gamma_self_consistency`
- [ ] **2+ fragment SCC convergence test** — every test uses `vec![frag]` (1 fragment); target: `tests/qmqm_integration.rs::two_fragment_scc_parity` (new)
- [ ] **Inter-fragment `compute_v_ext` exercised & verified** — `qmqm/solver.rs::compute_v_ext` is currently a dead path in tests; target: `tests/qmqm_integration.rs::two_fragment_polarization`
- [ ] Polarization convergence: two polar fragments mutually polarizing — target: `tests/qmqm_integration.rs::two_water_polarization`
- [ ] Charge conservation across fragments (sum dQ = 0) — target: `tests/qmqm_integration.rs::charge_conservation_multi_frag`
- [ ] Scaling test: 100s–1000s of identical fragments (templating) — target: `tests/qmqm_integration.rs::many_fragment_scaling`; uses `FragmentTemplate` cloning
- [ ] Energy decomposition: intra vs inter fragment — target: `qmqm/solver.rs` (expose energy split); `tests/qmqm_integration.rs::energy_decomposition`

---

## 4. OpenCL GPU — Diagonalization  ← `GPU_Diagonalization_Test_Plan.md`

### 4.1 Kernels (`gpu_matrix_ops.cl`) & API (`gpu_matrix.rs`)
- [*] `batched_gemm` (tiled, row-major, batched in 3rd dim) — `qmqm/gpu_matrix_ops.cl::batched_gemm`; `qmqm/gpu_matrix.rs::GpuMatrixContext::batched_gemm`
- [*] `local_jacobi_blocks` (serial cyclic Jacobi, single thread) — `qmqm/gpu_matrix_ops.cl::local_jacobi_blocks`; `GpuMatrixContext::local_jacobi_blocks`
- [*] `local_jacobi_blocks_parallel` (row-parallel rotations) — `qmqm/gpu_matrix_ops.cl::local_jacobi_blocks_parallel`; `GpuMatrixContext::local_jacobi_blocks_parallel`
- [*] `scale_density_guess`, `purify_mcweeny`, `purify_tc2_*` — `qmqm/gpu_matrix_ops.cl::scale_density_guess`, `purify_mcweeny`, `purify_tc2_step`, `purify_tc2_one_batch`; `GpuMatrixContext::scale_density_guess`, `purify_palser_manolopoulos`, `purify_trace_correcting`
- [*] `trace_reduce`, `idempotency_reduce` — `qmqm/gpu_matrix_ops.cl::trace_reduce`, `idempotency_reduce`; `GpuMatrixContext::trace`, `idempotency_error`
- [*] `GpuMatrixContext` (context, buffer mgmt, kernel compile) — `qmqm/gpu_matrix.rs::GpuMatrixContext::new`, `buffer_from_slice`, `zero_buffer`, `read_buffer`; `MatrixKernelConfig::render_source`
- [*] `lowdin_transform` (H' = X^T·H·X, two GEMMs) — `GpuMatrixContext::lowdin_transform`
- [*] Brent-Luk block-Jacobi schedule — `qmqm/gpu_matrix.rs::brent_luk_rounds`

### 4.2 Tests (`tests/gpu_diagonalization.rs`) — 8/8 PASS
- [*] GEMM correctness (4×4) — `tests/gpu_diagonalization.rs::test_gpu_gemm_correctness`
- [*] Jacobi synthetic 8×8 (serial + parallel) — `tests/gpu_diagonalization.rs::test_gpu_jacobi_synthetic`
- [*] Diagonalize S from H2 (2×2), N2 (8×8) → S^{-1/2} — `tests/gpu_diagonalization.rs::test_gpu_jacobi_overlap_h2`, `test_gpu_jacobi_overlap_n2`
- [*] Löwdin transform on GPU (N2) — `tests/gpu_diagonalization.rs::test_gpu_lowodin_transform`
- [*] Full HC=SCε pipeline (H2, N2) — `tests/gpu_diagonalization.rs::test_gpu_full_diagonalization_h2`, `test_gpu_full_diagonalization_n2` (helper `gpu_diagonalize`)
- [*] Batched Jacobi (3× 4×4 in one launch) — `tests/gpu_diagonalization.rs::test_gpu_batched_jacobi`

### 4.3 Current Jacobi optimization (2026-09-06, awaiting USER acceptance)
- [~] `gpu_eigen.cl`: disjoint 2×2 pair-block update, 4 → 2 barriers/round; two-pass reference retained with `JACOBI_BLOCK_UPDATE=0`. GTX 1650 event benchmark at equal accuracy: N28/batch100 4.261 → 2.647 ms (1.61×); no end-to-end speedup claimed.
- [~] Rotation normalization refinement addresses measured f32 orthogonality drift without changing tolerances; N64 reconstruction error 4.47e-4 → 4.01e-5. New `gpu_eigen.rs` parity/profile tests pass through N64; existing `gpu_eigenproblem` 10/10 and `gpu_scc` 3/3 pass. Numerical derivation, benchmark protocol and handoff: `../chats/GPU_Optimization.chat.md`.
- [ ] Remaining: production Jacobi residual/exhaustion reporting; NVIDIA `assemble_pairs` resource failure (reproduced before optimization), replica>=128 metadata OOB, persistent kernel/workspace lifetimes. H-bond full-pipeline acceptance is blocked by assembly, not established by the passing CPU-H/S SCC tests.

### 4.4 Gaps
- [ ] HCOOH test (N=14, mixed species) — target: `tests/gpu_diagonalization.rs::test_gpu_full_diagonalization_hcooh`
- [ ] Block-Jacobi driver for N>64 (only subproblem kernel exists) — target: `qmqm/gpu_matrix.rs` (new driver using `brent_luk_rounds` + `batched_gemm` to apply rotations)
- [ ] Purification path tested vs CPU density matrix — target: `tests/gpu_diagonalization.rs::test_gpu_purification_*` (uses `purify_palser_manolopoulos` / `purify_trace_correcting`)
- [ ] GPU-side S^{-1/2} reconstruction (currently host-side `compute_s_inv_half`) — target: `qmqm/gpu_matrix.rs` (batched GEMM V·Λ^{-1/2}·V^T)
- [ ] Batched op with different N per fragment (padding) — target: `qmqm/gpu_matrix.rs` (pad + mask)
- [ ] Parallel Jacobi on 16×16, 32×32 — target: `tests/gpu_diagonalization.rs::test_gpu_jacobi_*_16`, `_32`
- [ ] f64 precision path (currently f32 only) — target: `qmqm/gpu_matrix_ops.cl` (templated f64 kernels, requires `cl_khr_fp64`)

---

## 5. OpenCL GPU — Hamiltonian Assembly  ← `DFTB_Hassembly_OpenCL.md`, `OpenCL_DFTB_optimization.md`

### 5.1 Design & kernels
- [*] Design document — `doc/prokop/DFTB_Reimplementation_Progress/DFTB_Hassembly_OpenCL.md`
- [*] `dftb_hamiltonian.cl` kernels — `methods/dftb/dftb_hamiltonian.cl::assemble_pairs`, `onsite_and_va`, `onsite_diagonal`
- [*] Pair-bucket sorted launch (by species-pair + block type 1×1/1×4/4×4) — `qmqm/gpu_prep.rs::build_pair_buckets`, `GpuPairBucket`
- [*] `__local` SK table caching — `methods/dftb/dftb_hamiltonian.cl::assemble_pairs` (loads `GpuSkTable` into `__local`)
- [*] Cubic B-spline interpolation (GPU, same controls as CPU) — `dftb_hamiltonian.cl` / `gpu_forces.cl` `cubic_interp_params` + analytic `cubic_weights_d1`. Host pack: `gpu_prep.rs` uses `fit_bspline_controls_zero_end` (same stopgap as CPU). Mio tables: original grid + r=0 dummy + 4 pad ≤ `SK_GRID_MAX=512` (no 64-point resample on mio).
- [*] Heterogeneous fragment support (prefix-sum offsets) — `qmqm/gpu_prep.rs::GpuFragment` (atom_off, h_base), `GpuBatch::from_fragments`
- [*] f32 throughout — all kernels in `dftb_hamiltonian.cl` + `gpu_matrix_ops.cl`

### 5.2 Host prep
- [*] `gpu_prep.rs` — pack fragments/pairs/SK/gamma into flat arrays — `qmqm/gpu_prep.rs::GpuBatch::from_fragments`, `build_global_species`, `pack_sk_tables`, `build_gamma_neigh`
- [*] `GpuFragment`, `GpuPairEntry` structs (match OpenCL layout) — `qmqm/gpu_prep.rs::GpuFragment`, `GpuPairEntry`, `GpuPairBucket`, `GpuSkTable`, `GpuGammaNeigh`
- [~] SK table packing — `gpu_prep.rs::pack_sk_tables`. Mio: full grid + pad (no resample). Resample-to-`SK_RESAMPLE_N` only if `n_grid+1+N_PAD_END > SK_GRID_MAX`. Old “H/S parity 1e-2 from 64-point resample” is **not** the current mio H-bond path (AT max\|dH\| `8.6e-8`).
- [*] Pair bucketing by species-pair + block type — `qmqm/gpu_prep.rs::build_pair_buckets`, `determine_block_type`, `extract_shell_old_or_new`, `n_orb_from_ang`
- [*] `gpu_prep` wired into `qmqm/mod.rs` — `qmqm/mod.rs:17 pub mod gpu_prep;`

### 5.3 Runtime  ✅ DONE (Wave 1, Agent_1)
- [*] OpenCL driver module (device init, buffer upload, kernel enqueue) — `qmqm/gpu_driver.rs::GpuDriver::new`, `::gpu_assemble_batched`
- [*] Compile `dftb_hamiltonian.cl` at runtime — `qmqm/gpu_driver.rs::GpuDriver::new` (Program::build)
- [*] Smoke test: build batch → upload → `assemble_pairs` → read back H/S — `tests/gpu_hamiltonian.rs::test_gpu_assemble_pairs_smoke`
- [*] H/S parity vs CPU (dense H-bond, mio, padded B-spline) — `tests/gpu_hbond_physics.rs`: H2/H2O/formic/AT/GC max\|dH\| `~1e-8`–`8.6e-8`. Older `gpu_hamiltonian.rs` 1e-2 numbers were the 64-point resample gap.
- [*] Multi-replica batched H assembly test — `tests/gpu_hamiltonian.rs::test_gpu_multi_replica` (10× H2); `gpu_hbond_physics.rs` batch=200 H2 (replica cap)
- [ ] Performance benchmark vs CPU — target: `tests/gpu_hamiltonian.rs::bench_gpu_assemble` (or `examples/bench_gpu.rs`)
- [~] 5 kernel bug fixes in `dftb_hamiltonian.cl` (onsite overflow, SK cache OOB, float4 cast, write_symmetric_4x4, rotate_4x4); later: `vload2` for 1×4 (NVIDIA alignment)
- [ ] **Fix SK resampling precision** — only for tables that still hit the resample branch (`n_gpu > SK_GRID_MAX`). Mio H-bond does not.

### 5.4 Known optimization opportunities (from `OpenCL_DFTB_optimization.md`)
- [ ] Replace `if (du<0) du=-du` with `fabs()`; pre-sort pairs by same-U — `methods/dftb/dftb_hamiltonian.cl::gamma_full` (L78-91); host sort in `gpu_prep.rs::build_pair_buckets`
- [ ] Remove dead range-check branches in `cubic_interp_params` (host guarantees validity) — `methods/dftb/dftb_hamiltonian.cl::cubic_interp_params` (L111, L117-118); use `clamp()`
- [ ] Remove dead `if (base < 0)` in `interp_sk_*_indexed` — `methods/dftb/dftb_hamiltonian.cl::interp_sk_*_indexed` (L128, L137, L152)
- [ ] Precompute onsite diagonal on host (skip uncoalesced diagonal writes) — `methods/dftb/dftb_hamiltonian.cl::onsite_and_va` (L305-308); or separate kernel
- [ ] Pre-load `fragments[]` into `__local` memory — `methods/dftb/dftb_hamiltonian.cl::assemble_pairs` (L373)
- [ ] Ensure pair sort groups by replica then atom_i for coalesced V_a reads — `qmqm/gpu_prep.rs::build_pair_buckets` (sort key)

---

## 6. OpenCL GPU — Full SCC Cycle  ❌ NOT STARTED

> **Design (revised 2026-09-06):** see `GPU_MultiSystem_Design.md` decisions D2 (host-orchestrated
> device-resident), D8 (Brent-Luk Jacobi), D9 (S^{-1/2} on GPU), D11 (device-resident SCC),
> D15 (gamma precompute), D16 (H0/S separated from SCC inner loop).
>
> **Architecture:** H0/S/Gamma/X computed once per geometry. SCC inner loop is a sequence of
> cheap kernels enqueued by host, no PCIe traffic. Active mask skips converged systems.

### 6.1 Generalized eigenproblem on GPU (Stage 2)
- [ ] **Brent-Luk parallel cyclic Jacobi kernel** (D8) — N/2 independent rotations per round, one barrier per round, A+V in `__local` for N≤64 — target: `qmqm/gpu_matrix_ops.cl::jacobi_cyclic_local_batched` (new; replace `local_jacobi_blocks_parallel`)
- [ ] **S^{-1/2} via Jacobi** (D9) — jacobi(S) → U·Λ^{-1/2}·U^T, returns λ_min(S) for precision monitoring — target: `qmqm/gpu_matrix_ops.cl::build_inv_sqrt_from_eig` (new)
- [ ] **Full-local batched GEMM** for N≤64 (§4.3) — load both matrices into `__local` once — target: `qmqm/gpu_matrix_ops.cl::matmul_full_local_batched` (new); benchmark vs `batched_gemm`
- [ ] **N+1 leading dimension** to avoid power-of-two bank conflicts — target: `qmqm/gpu_matrix_ops.cl` (JLD = N+1)
- [ ] **Odd N padding** (N→N+1 with dummy state) for Brent-Luk — target: `qmqm/gpu_matrix_ops.cl`
- [ ] Eigenproblem pipeline: H0/S → jacobi(S) → X → H'0 = X·H0·X → jacobi(H'0) → ε, C — target: `qmqm/gpu_driver.rs::gpu_solve_gevp_batched` (new)
- [ ] Parity vs CPU for H2, N2, H2O, CH4 eigenvalues + MO coefficients — target: `tests/gpu_eigenproblem.rs::test_*` (new)
- [ ] Benchmark: Brent-Luk vs `local_jacobi_blocks_parallel` at N=8,16,32,48,64, batch=1,10,100,1000 — target: `tests/gpu_eigenproblem.rs::bench_jacobi_*` (new)

### 6.2 Device-resident SCC (Stage 3, D11)
- [*] **Gamma precompute kernel** (D15) — dense N_atom×N_atom matrix per system, once per geometry — host-built, uploaded; `tests/gpu_scc_kernels.rs::build_gamma_matrix`
- [*] **Gamma matvec kernel** — V_A = G·Δq, batched — `qmqm/gpu_matrix_ops.cl::gamma_matvec_batched`; parity <2.3e-9
- [*] **H_scc_update kernel** (§4.4) — H = H0 + 0.5·S·(V_i+V_j), elementwise — `qmqm/gpu_matrix_ops.cl::h_scc_update_batched`; parity <3e-8
- [*] **Mulliken charges kernel** — q = diag(D·S), batched — `qmqm/gpu_matrix_ops.cl::mulliken_charges_batched`; parity <4.8e-7
- [*] **Residual + simple mixer kernel** — RMS = ||q_new - q_old||, q_mixed = α·q_new + (1-α)·q_old — `qmqm/gpu_matrix_ops.cl::residual_and_mix_batched`; parity <1.5e-8
- [*] **Density build kernel** — D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T with occupation mask — `qmqm/gpu_matrix_ops.cl::build_density_masked_batched` (new, Wave 3)
- [*] **Frobenius trace + dot kernels** — for energy computation Tr(D·H0) and Σ Δq·V — `qmqm/gpu_matrix_ops.cl::frobenius_trace_batched`, `dot_batched` (new, Wave 3)
- [ ] **Active mask** (§2.2) — active[system] flag, converged systems early-return — target: `qmqm/gpu_scc.rs` (TODO: all systems run all iterations currently)
- [*] **SCC loop driver** — host enqueues kernel sequence per iteration, minimal readback (RMS + occ_mask only) — `qmqm/gpu_scc.rs::gpu_solve_scc_batched` (new, Wave 3)
- [*] **CPU-driven DIIS mixer** — `gpu_solve_scc_batched_diis`, `gpu_solve_scc_batched_diis_warmstart` — DIIS on CPU driving GPU charge vectors. 13 iters vs 60-192 for simple mix.
- [*] **Warm-start + best-effort mode** — `gpu_solve_scc_batched_diis_warmstart` accepts separate init_q; `best_effort` flag returns per-system RMS for unconverged points
- [*] **Per-system RMS diagnostics** — reports 10 worst systems on nonconvergence
- [ ] **SCC optimization** (D16) — precompute H'0 = X·H0·X once, then ~1 GEMM per SCC iter — target: `qmqm/gpu_scc.rs` (optional optimization, after basic SCC works)
- [*] End-to-end GPU SCC parity vs CPU SCC — `tests/gpu_scc.rs`: H2O |dE|<3e-7, N2 |dE|<1e-6, 10× H2O |dE|<1e-6; `tests/hbond_gpu_scc.rs`: formic dimer 1D scan (21 pts, 28 orbs) |dE|<2e-5, |dq|<6e-6; `tests/formic_scan_plots.rs`: 1D (41 pts) |dE|<3.6e-6, 2D (21×21) 147/441 converged (CPU also fails on asymmetric pts)
- [ ] Per-replica data save (H0, S, H_scc, C, ε, D, q, E as f32 binary) — target: `qmqm/gpu_scc.rs::save_replica_data`
- [*] **Scan plotting** — `scripts/plot_formic_scan.py` + `tests/formic_scan_plots.rs` — 1D energy/charge/parity + 2D contour plots under `debug/formic_dimer_scan/`

### 6.3 Runtime refactoring (Stage 3.5, D14)
- [ ] **Shared GpuRuntime** — context, device, queues, capabilities, program cache — target: `qmqm/gpu_runtime.rs` (new; refactor from `GpuMatrixContext` + `GpuDriver`)
- [ ] **Cache Kernel objects** — create once, update args per call (current code creates new `ocl::Kernel` every call) — target: `qmqm/gpu_runtime.rs`
- [ ] **Profiling-enabled queue** for benchmarks — target: `qmqm/gpu_runtime.rs`

### 6.4 f32 precision monitoring (D10)
- [ ] Monitor λ_min(S) per system — if ~1e-7, S ill-conditioned — target: `qmqm/gpu_driver.rs` (return from `build_inv_sqrt_from_eig`)
- [ ] Monitor ||U^T·U - I|| (orthogonality) — target: `tests/gpu_eigenproblem.rs` (diagnostic)
- [ ] Monitor ||H·C - S·C·ε|| (generalized residual) — target: `tests/gpu_eigenproblem.rs` (diagnostic)
- [ ] Establish f32 vs f64 accuracy empirically for energies and forces — target: `tests/gpu_scc.rs::test_f32_f64_comparison`

---

## 7. OpenCL GPU — Multi-System Parallelization  ⚠️ IN PROGRESS

> **Full design doc:** `GPU_MultiSystem_Design.md` (revised 2026-09-06, decisions D1–D17).
> **Architecture:** System = batch index, each kernel chooses its own parallelization.
> Host-orchestrated, device-resident SCC. Active mask for convergence divergence.
> f32 only. Homogeneous templates for scan/NEB.

### 7.1 Design decisions — RESOLVED (see `GPU_MultiSystem_Design.md` §9)
- [*] **D1:** ~~1 WG = 1 system~~ → System = batch index, each kernel chooses mapping ✓
- [*] **D2:** ~~host-driven vs mega-kernel~~ → Host-orchestrated, device-resident (no readback) ✓
- [*] **D3:** ~~purification O(N²)~~ → Corrected: O(N³); Jacobi primary, purification optional ✓
- [*] **D4:** Batched contiguous for homogeneous; pad for heterogeneous ✓
- [*] **D5:** Binary f32, save at every scan point ✓
- [*] **D6:** `inter_fragment_vext` kernel, defer until independent path works ✓
- [*] **D7:** ~~breaks at 50 atoms~~ → Corrected: 256-thread WG strides over 4950 pairs; real constraint is local memory for eigensolver ✓
- [*] **D8:** Brent-Luk parallel cyclic Jacobi — NEW ✓ (designed, not implemented)
- [*] **D9:** S^{-1/2} on GPU via Jacobi — NEW ✓ (designed, not implemented)
- [*] **D10:** f32 only, monitor precision — NEW ✓ (policy set)
- [*] **D11:** Device-resident SCC pipeline — NEW ✓ (designed, not implemented)
- [*] **D12:** System vs Fragment distinction — NEW ✓ (defined)
- [*] **D13:** Homogeneous template architecture — NEW ✓ (designed, not implemented)
- [*] **D14:** Runtime refactoring (shared GpuRuntime, cached kernels) — NEW ✓ (designed, not implemented)
- [*] **D15:** Gamma precompute as dense matrix — NEW ✓ (designed, not implemented)
- [*] **D16:** H0/S separated from SCC inner loop — NEW ✓ (designed, not implemented)
- [*] **D17:** Geometry preprocessing on GPU — NEW ✓ (designed, not implemented)

### 7.2 Independent replicas (scan / NEB / MD ensemble) — no inter-system coupling
- [*] Batched H-assembly: N replicas in one kernel launch — `qmqm/gpu_driver.rs::gpu_assemble_batched`; `tests/gpu_hamiltonian.rs::test_gpu_multi_replica` (10× H2)
- [ ] Batched diagonalization across replicas (Brent-Luk Jacobi, D8) — target: `qmqm/gpu_driver.rs::gpu_solve_gevp_batched`; test `tests/gpu_eigenproblem.rs::test_batched_diag_multi_replica`
- [ ] Batched independent SCC (device-resident, D11) — target: `qmqm/gpu_driver.rs::gpu_solve_scc_batched`; test `tests/gpu_scc.rs::test_gpu_scc_independent_h2o`
- [ ] Per-replica data save (H0, S, H_scc, C, ε, D, q, E as f32 binary) — target: `qmqm/gpu_driver.rs::save_replica_data`
- [ ] Rigid scan driver — target: `examples/scan.rs` (sweep coordinate → batched SCC → energy curve)
- [ ] NEB driver (spring forces + image update on host) — target: `examples/neb.rs`
- [ ] Scan test: H2 bond scan 0.5–3.0 Å, 20 points vs CPU — target: `tests/scan.rs::test_h2_bond_scan`
- [ ] Scaling benchmark: 100, 1000 replicas — target: `tests/gpu_multi.rs::bench_independent_replicas`

### 7.3 Scheduling benchmark (Stage 5)
- [ ] Compare: (A) giant batch, (B) microbatches 16/32/64/128, (C) per-system, (D) multi-queue, (E) microbatches on multiple queues — target: `tests/gpu_sched.rs` (new)
- [ ] Synthetic convergence distributions: uniform, mild spread, bad tail — target: `tests/gpu_sched.rs`
- [ ] Measure GPU event time, host wall time, systems/second — target: `tests/gpu_sched.rs`

### 7.4 QM/QM electrostatic coupling — inter-fragment polarization (Stage 8)
- [ ] `inter_fragment_vext` OpenCL kernel (D6) — target: `qmqm/gpu_matrix_ops.cl::inter_fragment_vext` (new)
- [ ] System/Fragment architecture (D12) — target: `qmqm/gpu_driver.rs` (System = convergence unit, Fragment = diagonalized block)
- [ ] GPU SCC with inter-fragment coupling — target: `qmqm/gpu_driver.rs::gpu_solve_scc_qmqm`
- [ ] Two-water polarization test on GPU — target: `tests/gpu_qmqm.rs::test_two_water_polarization` (compare vs CPU `MultiSystemSolver::solve_scc`)

### 7.5 Heterogeneous QM/QM (mixed fragment sizes, Stage 8)
- [ ] Grouped batched: group fragments by template (D13), one kernel per group — target: `qmqm/gpu_prep.rs::GpuBatch::grouped_from_fragments`
- [ ] Multi-command-queue overlap for inter-group parallelism — target: `qmqm/gpu_driver.rs`
- [ ] Scaling benchmark: 1000s of fragments — target: `tests/gpu_multi.rs::bench_1000_fragments`

---

## 7.5 Sparse BSR4 Purification + Frontier Orbitals  ← `topical_audit/sparse_tc2_purification.md`, `topical_audit/davidson_eigensolver.md`

### 7.5.1 BSR4 sparse matrices
- [*] BSR4 layout (4×4 atom blocks, CSR, symmetric storage) — `methods/sparse/bsr4.rs::Bsr4Matrix`
- [*] Geometric + full masks for sparsity — `methods/sparse/bsr4.rs::build_geometric_mask`, `build_full_mask`
- [*] Dense ↔ BSR4 conversion — `methods/sparse/bsr4.rs::from_dense`

### 7.5.2 GPU TC2 purification
- [*] Newton-Schulz `Z ≈ S⁻¹` — `methods/sparse/gpu_sparse.rs` (OpenCL)
- [*] Spectral bound estimation — `methods/sparse/gpu_sparse.rs`
- [*] TC2 purification loop with idempotency residual `R_I` — `methods/sparse/gpu_sparse.rs`, `sparse_bsr4_purification.cl`
- [*] Sparse Mulliken charges from `KS` diagonal blocks — `methods/sparse/gpu_sparse.rs::mulliken`
- [*] Convergence history export — `SparseResult.history`, Rhai `save_convergence`
- [*] Parity vs dense + DFTB+ on benzene/coronene/circumcoronene (max\|Δq\| < 6.5e-5 e) — `scripts/compare_rust_vs_dftbplus.py`
- [*] **Device-resident sparse workspace** (2025-09-22) — `GpuBsrStructure`, `GpuBsrMatrix`, `SparsePurifyWorkspace` with persistent device buffers, cached kernel handles, exactly 2 SpGEMMs per TC2 iteration (down from 5), 0 matrix host transfers per iteration, 1 scalar read for trace/branch. Bitwise-identical parity vs old host-roundtrip path verified. — `methods/sparse/gpu_sparse.rs`, `tests/gpu_sparse_bsr4.rs::test_tc2_dev_vs_host_parity`, `test_tc2_dev_resident_convergence`
- [*] **Jacobi stopping fix** (2025-09-22) — separated `PAIR_SKIP_TOL` (1e-12) from `JACOBI_TOL` (1e-7), added 3-sweep stagnation detection. Fixes the "20 sweeps but 12 are useless" problem. All 10 eigen + 3 SCC tests still pass. — `qmqm/gpu_eigen.cl`, `qmqm/gpu_eigen.rs`

### 7.5.3 Davidson partial eigensolver
- [*] Generalized Davidson `H C = S C ε` with S-orthonormalization — `methods/sparse/davidson.rs::davidson_generalized`
- [*] Diagonal preconditioner with regularization for near-degenerate states — `methods/sparse/davidson.rs`
- [*] Subspace restart when `m > max_subspace` — `methods/sparse/davidson.rs`
- [*] Unit test vs dense `SymmetricEigen` — `methods/sparse/davidson.rs::tests::test_davidson_vs_dense_small`
- [*] Rhai exposure `davidson_homo_lumo(name, n_target)` — `bin/dftb_engine.rs`
- [*] Validated on benzene (3 iters, match to 1e-10 Ha)
- [~] **Coronene/circumcoronene: does not converge** — diagonal preconditioner insufficient for dense near-degenerate frontier manifolds. Needs SSOR/ILU or shift-invert. See `topical_audit/davidson_eigensolver.md`

### 7.5.4 DFTB+ Fortran parity harness
- [*] `scripts/run_dftbplus_ref.py` — generates HSD, runs DFTB+, parses `detailed.out` + `band.out` (eV→Ha)
- [*] `scripts/plot_charges_homo_lumo.py` — spatial charge map + HOMO-LUMO levels
- [*] `scripts/compare_rust_vs_dftbplus.py` — numerical parity report
- [*] End-to-end on benzene/coronene/circumcoronene: dense charges match to machine precision, eigenvalues to ~2e-6 Ha

### 7.5.6 Wavefunction projection (real-space grid)
- [*] `SccResult.eigenvectors` field added (norb × norb MO coefficient matrix)
- [*] Rhai `save_eigenvectors(name, path)` — exports geometry + eigenvectors + eigenvalues to TSV
- [*] Rhai `save_hs_matrix(name, path)` — exports H_scc + S matrices for sparse iterative eigensolving
- [*] `scripts/plot_wavefunctions.py` — projects MOs onto 2D grid via pyBall OpenCL `GridProjector` + STO basis
- [*] `scripts/sparse_homo_lumo.py` — Chebyshev filter + Rayleigh-Ritz iterative eigensolver (from NumericalMathPlayground) for sparse HOMO/LUMO
- [*] `scripts/compare_homo_lumo_3way.py` — 3-way comparison: dense vs sparse(Cheb+Ritz) vs Fortran
- [*] Validated on benzene/coronene/circumcoronene (eigenvalues + wavefunction contour plots)
- [ ] Numerical parity vs DFTB+ waveplot cube files (visual only so far)
- See `topical_audit/wavefunction_projection.md`

### 7.5.5 Open issues
- [ ] Davidson: stronger preconditioner for degenerate systems (SSOR/ILU or CheFSI)
- [ ] Davidson: wire to sparse BSR4 matvec operator (avoid densification)
- [ ] BSR4: variable block size or dense fallback for H atoms
- [ ] Fermi temperature alignment between Rust and DFTB+ (currently ~2e-6 Ha eigenvalue diff)

---

## 7.6 Sparse Nanocrystal Vibrations (Si/H)  ← `tasts/Sparse_Nanocrystal_Vibrations/`

> **Manifest:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.manifest.md`
> **Report:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.report.md`
> **Master task:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.tasks.md`
> **Topical audit:** `topical_audit/sparse_nanocrystal_vibrations.md`
> **GPT-5.6 review:** `tasts/Sparse_Nanocrystal_Vibrations/Sparse_Nanocrystal_Vibrations.chat.md` line 2064+ (22 issues + 10-step plan)

Goal: sparse GPU DFTB for vibrational calculations on 300–1000 atom Si/H
nanocrystals. f32 GPU arithmetic, analytic forces, finite-difference Hessians,
~5% frequency accuracy, no pathological imaginary modes. Everything
GPU-resident — no CPU bridges, no hybrid paths.

### 7.6.1 Status (revised after GPT-5.6 review, 2026-09-09)

GPT-5.6 reviewed commit `b269ab6` and found several "completed" items are not
actually wired into the production numerical path. See manifest §13 for the
full 22-issue checklist and `tasks.md` for the phased master task breakdown.

- [*] **P0** — Sparse perf stats + `sparse_firewall` feature — correct
- [~] **P1** — C² B-spline evaluator built but **not canonical** in production force path; `SkTableSp` still uses C¹ Hermite + numerical FD tail derivative
- [~] **P2** — Angular derivatives correct; but radial V,V' still from Hermite, not C²
- [~] **P3** — D/W formula correct; but allocation-heavy and attached to dense SCC (not self-consistent sparse)
- [~] **P4** — Plan infrastructure built; **not integrated** into TC2/NS/K0/W
- [ ] **Gate C** — False positive — 5-atom toy system, +2I doesn't create gap
- [*] **Gate D** — Nonsingular padded Si/H basis — correct
- [ ] **Gate E** — False positive — FD-of-energy forces, symmetrized Hessian tautology

### 7.6.2 Current plan (Phases A–E, see `tasks.md`)

- **Phase A** — Foundation: canonical C² spline (A1), TC2 fix 2 SpGEMMs/iter (A2), plan integration (A3)
- **Phase B** — Sparse SCC solver, GPU-resident, self-consistent — **current focus**
  - B1: `SparseSystemWorkspace`, B2: GPU H0/S assembly, B3: GPU gamma, B4: GPU Hscc, B5: GPU K0+bounds, B6: SCC loop, B7: Mulliken
- **Phase C** — GPU-resident sparse analytic forces (pair-force + atom-gather kernel, after dense agent finishes)
- **Phase D** — Correctness gates: redo C (D1), E (D2), then F (D3), G (D4), H (D5)
- **Phase E** — Performance: real counters (E1), cell-list masks (E2), degree buckets (E3), packed plans (E4), lane benchmark (E5), symmetrization policy (E6), scaling Gate I (E7), production Gate J (E8)

---

## 8. I/O, Tooling & Test Infrastructure

### 8.1 I/O
- [*] `io.rs` — XYZ parsing, species/coords, output formats — `io.rs::parse_xyz`, `parse_species`, `parse_coords`, `parse_f64_list`, `load_sk_for_species`, `DftbOutput`, `OutputFormat`; matrix compare `max_abs_diff`, `compare_matrices`, `compare_vecs`, `permute_sp_per_atom`

### 8.2 Test drivers
- [*] `tests/run_parity.py` — XYZ → Fortran → Rust non-SCC comparison
- [*] `tests/run_scc_full.py` — XYZ → Fortran SCC → Rust SCC comparison
- [*] `tests/tblite_helper.c` + binary — xTB reference extraction (linked against `tblite`)
- [*] `tests/test_utils.py` — shared Python helpers

### 8.3 Gaps
- [ ] CI / automated test runner (GitHub Actions or justfile) — target: `.github/workflows/test.yml` or `justfile`
- [ ] Single-command "build helper + run all tests" Makefile/justfile — target: `Makefile` or `justfile` (build `tblite_helper`, set `RUST_DFTB_SK_DIR`, run `cargo test`)
- [*] `tests/run_forces.py` (forces driver) — `tests/run_forces.py::main` (parse "Total Forces" from `detailed.out`, 4-column format)
- [ ] GPU H-assembly test driver — target: `tests/run_gpu_parity.py` (new) or extend `run_parity.py`

---

## 9. Build & Repo Health

- [*] `cargo build` succeeds (lib, 71 warnings — mostly snake_case naming) — `rust_dftb/Cargo.toml`
- [*] `Cargo.toml` profiles tuned (debug=1, strip debuginfo) per boc footprint plan — `rust_dftb/Cargo.toml [profile.dev]`, `[profile.release]`
- [*] `ocl = "0.19"` dependency present — `rust_dftb/Cargo.toml`
- [ ] Reduce warning count (17 auto-fixable via `cargo fix`) — target: `cargo fix --lib -p rust_dftb`
- [ ] Document required env vars (`RUST_DFTB_SK_DIR`, `RUST_DFTB_REF_H`, `RUST_DFTB_REF_S`, `RUST_DFTB_SCC_XYZ`, `RUST_DFTB_FORCES_XYZ`) in one place — target: `rust_dftb/README.md` or `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` §8

---

## 10. Quick Status at a Glance

| Area | Done | Verdict |
|------|------|---------|
| DFTB CPU non-SCC | ✅ | Complete, parity verified — `tests/parity_non_scc.rs`, `tests/parity_universal.rs` |
| DFTB CPU SCC | ✅ | Complete, 13-molecule parity — `tests/parity_scc.rs` |
| DFTB CPU forces | ✅ | Complete, H2O non-SCC+SCC parity < 7e-8 — `methods/dftb/forces.rs`, `tests/parity_forces.rs` |
| xTB GFN1 | ✅ | Complete, parity on 3 mols — `tests/xtb_parity.rs::test_*_gfn1_*` |
| xTB GFN2 | ⚠️ | ~0.1% residual on N2/HCOOH — `tests/xtb_parity.rs::test_*_gfn2_*` |
| QM/QM CPU solver | ✅ | 2-fragment validated, 13/13 tests pass — `tests/qmqm_integration.rs` |
| GPU diagonalization | ✅ | 8/8 tests pass — `tests/gpu_diagonalization.rs` (pre-existing N2 failures flagged) |
| GPU H-assembly | ⚠️ | 4/4 tests pass (H2, N2); **s-p rotation sign bug** for H-O/H-C pairs (formic dimer |dH0|=0.66) — `qmqm/gpu_driver.rs`, `tests/gpu_hamiltonian.rs` |
| GPU generalized eigensolver | ✅ | 10/10 tests pass — Brent-Luk Jacobi + S^{-1/2} — `qmqm/gpu_eigen.rs`, `tests/gpu_eigenproblem.rs` |
| GPU SCC component kernels | ✅ | 10/10 tests pass — GEMM, gamma, H_scc, Mulliken, mixer — `qmqm/gpu_matrix.rs`, `tests/gpu_scc_kernels.rs` |
| GPU SCC cycle | ✅ | Device-resident SCC loop with DIIS — H2O/N2/10×H2O parity <1e-6; formic dimer 1D scan (41 pts, 28 orbs) |dE|<3.6e-6, |dq|<1e-5; 2D scan 147/441 converged (CPU also fails) — `qmqm/gpu_scc.rs`, `tests/gpu_scc.rs`, `tests/hbond_gpu_scc.rs`, `tests/formic_scan_plots.rs` |
| GPU multi-system (independent) | ✅ | Batched SCC for N≤64, homogeneous templates, warm-start, best-effort — `qmqm/gpu_scc.rs::gpu_solve_scc_batched_diis_warmstart` |
| GPU multi-system (QM/QM coupling) | ❌ | Not started — Stage 8 of revised plan |
| Scan / NEB driver | ✅ | CPU scan: 2/2 tests pass; GPU scan: 1D+2D formic dimer with plots — `examples/scan.rs`, `tests/scan.rs`, `tests/formic_scan_plots.rs`, `scripts/plot_formic_scan.py` |
| Sparse TC2 purification | ✅ | Benzene/coronene/circumcoronene parity < 6.5e-5 e — `methods/sparse/gpu_sparse.rs` |
| Sparse nanocrystal vibrations | ⚠️ | GPT-5.6 review: P1/P2/Gate C/Gate E unmarked (not wired into production); Phase B (sparse SCC) in progress — `tasts/Sparse_Nanocrystal_Vibrations/` |
| Davidson partial eigensolver | ⚠️ | Benzene OK; coronene/circumcoronene do not converge (diagonal preconditioner) — `methods/sparse/davidson.rs` |
| DFTB+ parity harness | ✅ | `scripts/run_dftbplus_ref.py` + `compare_rust_vs_dftbplus.py` |
| Test infra / CI | ⚠️ | Drivers exist, no CI |

**Overall "multi-system DFTB in OpenCL": ~75% complete.**
CPU foundation complete (non-SCC + SCC + forces + QM/QM 2-frag). GPU H-assembly
runtime working. Missing: Brent-Luk Jacobi, S^{-1/2}, device-resident SCC, scan/NEB.
Architecture revised per GPT 5.6 review: host-orchestrated device-resident, f32,
homogeneous templates, H0/S separated from SCC inner loop.
Full plan: `GPU_MultiSystem_Design.md`.

---

## 11. Recommended Next Steps (priority order)

> **Architecture revised (2026-09-06):** host-orchestrated device-resident SCC,
> Brent-Luk Jacobi, S^{-1/2} on GPU, f32 only, homogeneous templates.
> See `GPU_MultiSystem_Design.md` for the full revised staged plan (D1–D17).

1. **SK extra-control BC fitter** — replace blunt zero-sample pad. Extra
   controls before/after the table must be *solved* (valid-domain accuracy +
   V,V'→0 at cutoff), not hardcoded zeros. Do not restore Neville/`DIST_FUDGE`.
   — `doc/prokop/topical_audit/sk_interpolation.md`, `interpolation.rs`,
   `spline_resample.rs`
   (Old “64-point resample H/S ~1e-2” is **not** the current mio H-bond path;
   AT max|dH| is `8.6e-8`. Resample branch only if `n_grid+1+N_PAD_END > SK_GRID_MAX`.)
2. ~~**Brent-Luk parallel cyclic Jacobi kernel** (D8)~~ — ✅ DONE — `qmqm/gpu_eigen.rs::jacobi_cyclic_local_batched`
3. ~~**S^{-1/2} on GPU** (D9)~~ — ✅ DONE — `qmqm/gpu_eigen.rs::build_inv_sqrt`
4. ~~**Full-local batched GEMM** for N≤64~~ — ✅ DONE — `qmqm/gpu_matrix.rs::matmul_full_local_batched`
5. ~~**Device-resident SCC** (D11)~~ — ✅ DONE — `qmqm/gpu_scc.rs::gpu_solve_scc_batched_diis`
6. ~~**Dispatch Wave 2**~~ — ✅ DONE — Agent_4 (GPU SCC) + Agent_5 (scan)
7. **Runtime refactoring** (D14) — shared GpuRuntime, cached kernels.
   — `qmqm/gpu_runtime.rs`
8. **Scheduling benchmark** (Stage 5) — giant batch vs microbatch vs multi-queue.
   — `tests/gpu_sched.rs`
9. **GPU QM/QM inter-fragment coupling** (D6, D12) — `inter_fragment_vext` kernel.
   — Stage 8 of `GPU_MultiSystem_Design.md`
10. **CI + justfile** — stop relying on manual env-var setup.
11. **GFN2 residual debug** — deep, uncertain; defer unless xTB needed.
12. **2D scan convergence** — 67% of 2D points don't converge (CPU also fails).
    Try: Broyden mixing, level shifting, smaller alpha, strip propagation.
    — `qmqm/gpu_scc.rs`, `tests/formic_scan_plots.rs`
13. **Performance optimization** — cache Kernel objects, active mask, diagonal-extract
    kernel. See `doc/prokop/reports/2025-09-06_gpu_scc_benchmarks.md`.
    **Baseline measured:** Jacobi = 50-65% of per-iter time, GEMM = 5%, DIIS = 16%.
    Throughput: 1194 systems/s at batch=100 (formic dimer, N=28). 60-120× vs CPU.
