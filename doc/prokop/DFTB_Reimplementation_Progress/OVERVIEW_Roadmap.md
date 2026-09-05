# rust_dftb — Master Roadmap & Status Checklist

**Last updated:** 2025-09-05
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
- [*] Neville 8-point interpolation on uniform grid — `methods/dftb/interpolation.rs::EqGridTable::eval_into`, `poly_inter_uniform_into`
- [*] `r_max` hard cutoff matching Fortran `slakoeqgrid.F90` (bug fixed) — `methods/dftb/interpolation.rs::eval_eqgrid_new_into`
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
- [ ] Density matrix DM exposed in `SccResult` — target: `methods/dftb/hamiltonian.rs::SccResult` (add `density` field)
- [ ] Energy-weighted density matrix EDM — target: `methods/dftb/hamiltonian.rs::build_scc` (compute `2·C_occ·diag(eps)·C_occ^T`)
- [ ] Finite-difference dH0/dx, dS/dx (delta = eps^0.25) — target: `methods/dftb/forces.rs` (new), reuses `rotation.rs::rotate_diatomic_block_into` + `interpolation.rs::eval_into`
- [ ] Repulsive spline parsing from SK file — target: `methods/dftb/sk_data.rs::read_skf_all` / new `parse_spline_section`
- [ ] F_nonSCC = 2·(DM·dH0' − EDM·dS') — target: `methods/dftb/forces.rs::non_scc_forces`
- [ ] F_rep = dE_rep/dr · r_hat — target: `methods/dftb/forces.rs::repulsive_forces` (spline eval from `splinerep.F90` analog)
- [ ] `gamma_prime_full(r, U1, U2)` — target: `methods/dftb/gamma.rs` (mirror `shortgammafuncs.F90::expGammaPrime`)
- [ ] F_SCC_dc (gamma + 1/R Coulomb) — target: `methods/dftb/forces.rs::scc_dc_forces`
- [ ] F_SCC_shift (Pulay-like, needs block-resolved shifts) — target: `methods/dftb/forces.rs::scc_shift_forces`; needs block shifts from `qmqm/shifts.rs::compute_intra_shifts`
- [ ] `tests/parity_forces.rs` + `tests/run_forces.py` — target: new test + driver (parse "Total Forces" from `detailed.out`)
- [ ] Non-SCC force parity vs Fortran — target: `tests/parity_forces.rs::non_scc_forces_from_xyz`
- [ ] Full SCC force parity vs Fortran — target: `tests/parity_forces.rs::scc_forces_from_xyz`

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

### 4.3 Gaps
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
- [*] Cubic B-spline interpolation (GPU-friendly, replaces Neville) — `methods/dftb/dftb_hamiltonian.cl::cubic_interp_params`, `interp_sk_*_indexed`; host resample `methods/dftb/spline_resample.rs::resample_bspline`, `resample_sk_column`
- [*] Heterogeneous fragment support (prefix-sum offsets) — `qmqm/gpu_prep.rs::GpuFragment` (atom_off, h_base), `GpuBatch::from_fragments`
- [*] f32 throughout — all kernels in `dftb_hamiltonian.cl` + `gpu_matrix_ops.cl`

### 5.2 Host prep
- [*] `gpu_prep.rs` — pack fragments/pairs/SK/gamma into flat arrays — `qmqm/gpu_prep.rs::GpuBatch::from_fragments`, `build_global_species`, `pack_sk_tables`, `build_gamma_neigh`
- [*] `GpuFragment`, `GpuPairEntry` structs (match OpenCL layout) — `qmqm/gpu_prep.rs::GpuFragment`, `GpuPairEntry`, `GpuPairBucket`, `GpuSkTable`, `GpuGammaNeigh`
- [*] SK table resampling to ≤256 grid points — `methods/dftb/spline_resample.rs::resample_bspline`, `resample_sk_column`, `cubic_spline_d2_uniform`; const `gpu_prep.rs::SK_GRID_MAX`
- [*] Pair bucketing by species-pair + block type — `qmqm/gpu_prep.rs::build_pair_buckets`, `determine_block_type`, `extract_shell_old_or_new`, `n_orb_from_ang`
- [*] `gpu_prep` wired into `qmqm/mod.rs` — `qmqm/mod.rs:17 pub mod gpu_prep;`

### 5.3 Runtime  ✅ DONE (Wave 1, Agent_1)
- [*] OpenCL driver module (device init, buffer upload, kernel enqueue) — `qmqm/gpu_driver.rs::GpuDriver::new`, `::gpu_assemble_batched`
- [*] Compile `dftb_hamiltonian.cl` at runtime — `qmqm/gpu_driver.rs::GpuDriver::new` (Program::build)
- [*] Smoke test: build batch → upload → `assemble_pairs` → read back H/S — `tests/gpu_hamiltonian.rs::test_gpu_assemble_pairs_smoke`
- [*] H/S parity vs CPU reference — `tests/gpu_hamiltonian.rs::test_gpu_hs_parity_h2`, `_n2` (max|dH| ~3e-3..8e-3, tol 1e-2; contract 1e-5 NOT met — SK resampling precision gap)
- [*] Multi-replica batched H assembly test — `tests/gpu_hamiltonian.rs::test_gpu_multi_replica` (10× H2)
- [ ] Performance benchmark vs CPU — target: `tests/gpu_hamiltonian.rs::bench_gpu_assemble` (or `examples/bench_gpu.rs`)
- [~] 5 kernel bug fixes in `dftb_hamiltonian.cl` (onsite overflow, SK cache OOB, float4 cast, write_symmetric_4x4, rotate_4x4) — user-authorized scope deviation
- [ ] **Fix SK resampling precision** (BLOCKING for Stage 2) — increase `SK_RESAMPLE_N` from 64 to ≥256, or interpolate on GPU — target: `qmqm/gpu_prep.rs::SK_RESAMPLE_N`, `methods/dftb/spline_resample.rs`

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
- [ ] **Gamma precompute kernel** (D15) — dense N_atom×N_atom matrix per system, once per geometry — target: `qmqm/gpu_matrix_ops.cl::build_gamma_matrix` (new)
- [ ] **Gamma matvec kernel** — V_A = G·Δq, batched — target: `qmqm/gpu_matrix_ops.cl::gamma_matvec_batched` (new)
- [ ] **H_scc_update kernel** (§4.4) — H = H0 + 0.5·S·(V_i+V_j), elementwise — target: `qmqm/gpu_matrix_ops.cl::h_scc_update` (new)
- [ ] **Mulliken charges kernel** — q = diag(D·S), batched — target: `qmqm/gpu_matrix_ops.cl::mulliken_charges` (new)
- [ ] **Residual + simple mixer kernel** — RMS = ||q_new - q_old||, q_mixed = α·q_new + (1-α)·q_old — target: `qmqm/gpu_matrix_ops.cl::residual_and_mix` (new)
- [ ] **Active mask** (§2.2) — active[system] flag, converged systems early-return — target: `qmqm/gpu_driver.rs` (active mask buffer + kernel early-return)
- [ ] **SCC loop driver** — host enqueues kernel sequence per iteration, no readback until convergence — target: `qmqm/gpu_driver.rs::gpu_solve_scc_batched` (new)
- [ ] **SCC optimization** (D16) — precompute H'0 = X·H0·X once, then ~1 GEMM per SCC iter — target: `qmqm/gpu_driver.rs` (optional optimization, after basic SCC works)
- [ ] End-to-end GPU SCC parity vs CPU SCC — target: `tests/gpu_scc.rs::test_gpu_scc_parity_h2`, `_n2`, `_h2o` (compare vs `HamiltonianBuilder::build_scc`)
- [ ] Per-replica data save (H0, S, H_scc, C, ε, D, q, E as f32 binary) — target: `qmqm/gpu_driver.rs::save_replica_data`

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
| GPU H-assembly | ✅ | 4/4 tests pass, 5 kernel bugs fixed — `qmqm/gpu_driver.rs`, `tests/gpu_hamiltonian.rs` |
| GPU SCC cycle | ❌ | Not started — revised design in `GPU_MultiSystem_Design.md` D8–D17 |
| GPU multi-system (independent) | ❌ | Not started — Stage 2–3 of revised plan |
| GPU multi-system (QM/QM coupling) | ❌ | Not started — Stage 8 of revised plan |
| Scan / NEB driver | ❌ | Not started (Agent_5 ready to dispatch, CPU backend) |
| Sparse TC2 purification | ✅ | Benzene/coronene/circumcoronene parity < 6.5e-5 e — `methods/sparse/gpu_sparse.rs` |
| Davidson partial eigensolver | ⚠️ | Benzene OK; coronene/circumcoronene do not converge (diagonal preconditioner) — `methods/sparse/davidson.rs` |
| DFTB+ parity harness | ✅ | `scripts/run_dftbplus_ref.py` + `compare_rust_vs_dftbplus.py` |
| Test infra / CI | ⚠️ | Drivers exist, no CI |

**Overall "multi-system DFTB in OpenCL": ~45% complete.**
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

1. **Fix SK resampling precision** (BLOCKING) — increase `SK_RESAMPLE_N` from 64
   to ≥256, or interpolate on GPU. Current H/S parity ~1e-2, need ~1e-4 for SCC.
   — `qmqm/gpu_prep.rs::SK_RESAMPLE_N`, `methods/dftb/spline_resample.rs`
2. **Brent-Luk parallel cyclic Jacobi kernel** (D8) — N/2 independent rotations
   per round, one barrier per round, A+V in `__local` for N≤64. This is the
   central technical challenge. — `qmqm/gpu_matrix_ops.cl::jacobi_cyclic_local_batched`
3. **S^{-1/2} on GPU** (D9) — via Jacobi eigendecomposition of S. Computed once
   per geometry. — `qmqm/gpu_matrix_ops.cl::build_inv_sqrt_from_eig`
4. **Full-local batched GEMM** for N≤64 — benchmark vs tiled GEMM.
   — `qmqm/gpu_matrix_ops.cl::matmul_full_local_batched`
5. **Device-resident SCC** (D11) — gamma matvec + H_scc_update + Mulliken +
   residual + mixer + active mask. Host enqueues, no readback.
   — `qmqm/gpu_driver.rs::gpu_solve_scc_batched`
6. **Dispatch Wave 2** — Agent_4 (GPU SCC) + Agent_5 (scan/NEB, CPU backend).
   Independent, launch simultaneously. — `doc/prokop/tasts/GPU_MultiSystem/`
7. **Runtime refactoring** (D14) — shared GpuRuntime, cached kernels.
   — `qmqm/gpu_runtime.rs`
8. **Scheduling benchmark** (Stage 5) — giant batch vs microbatch vs multi-queue.
   — `tests/gpu_sched.rs`
9. **GPU QM/QM inter-fragment coupling** (D6, D12) — `inter_fragment_vext` kernel.
   — Stage 8 of `GPU_MultiSystem_Design.md`
10. **CI + justfile** — stop relying on manual env-var setup.
11. **GFN2 residual debug** — deep, uncertain; defer unless xTB needed.
