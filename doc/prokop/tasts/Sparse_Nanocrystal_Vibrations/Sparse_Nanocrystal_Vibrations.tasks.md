# Sparse Nanocrystal Vibrations — Master Task Breakdown

**Created:** 2026-09-09
**Source of truth:** `Sparse_Nanocrystal_Vibrations.manifest.md` (§13 checklist)
**GPT-5.6 review:** `Sparse_Nanocrystal_Vibrations.chat.md` from line 2064
**Status report:** `Sparse_Nanocrystal_Vibrations.report.md`

---

## Objective

Build the fastest GPU-accelerated sparse DFTB solver for vibrational calculations
on Si/diamond nanocrystals (300–1000 atoms). Everything GPU-resident — no CPU
bridges, no hybrid paths, no shortcuts that hamper performance.

## Design principles

**See manifest §1.4 for the full performance mandate.** Summary:

1. **GPU-resident by default.** All production matrix algebra, SCC, and force
   contraction happens on GPU. CPU is for setup, reference tests, and final
   Hessian eigensolve only.
2. **No allocation in hot loops.** All buffers, kernels, structures, plans
   created once during setup; reused across SCC iterations, force calls, and
   Hessian displacements.
3. **No dense matrices in production.** No `DMatrix`, no `Norb×Norb` allocation,
   no dense `SccResult` in the sparse path.
4. **No host round-trips in iterations.** Small diagnostic scalars read only
   at controlled intervals. Matrix data stays on GPU.
5. **Analytic derivatives everywhere.** No FD of H/S in production. FD only for
   force→Hessian.
6. **Self-consistent sparse model.** K from sparse SCC, not from dense SCC.
   The force must be the derivative of the same sparse energy whose Hessian we
   diagonalize.
7. **Fail loudly.** No silent fallbacks, no clamping, no default values.

## Parallel work streams

- **This agent (sparse):** Phases A, B, D, E (sparse SCC, TC2, workspace,
  correctness gates, performance)
- **Other agent (dense forces):** Analytic force contraction for the dense
  solver — will be reused as the mathematical reference for the GPU sparse
  force kernel (Phase C)
- **Integration point:** Phase C (GPU sparse force kernel) depends on both
  Phase B (sparse SCC → D/W) and the dense agent's analytic force work

---

## Phase A — Foundation: canonical interpolation + TC2 fix

These are prerequisites for everything else. They fix the most critical
numerical issues identified by GPT-5.6.

### A1 — Unify SK interpolation: one canonical C² B-spline

**GPT-5.6 issue:** #1 (critical blocker)
**Manifest ref:** §4.2, §3.3
**Depends on:** nothing
**Blocks:** B4 (sparse Hscc), C1 (GPU force kernel), all force/Hessian work

**Problem:** `SkTableSp` stores `EqGridTable` (C¹ Hermite with numerical FD tail
derivative). The C² B-spline evaluator from P1 exists but is not used in
production. The production force path differentiates a C¹ function with a
numerical tail derivative, not a C² function.

**Spec:**
- Create one canonical `SkRadialTable` struct storing C² cubic B-spline control
  points for H and S, per shell pair, per species pair.
- f64 CPU preprocessing → f32 B-spline controls (existing `spline_resample.rs`).
- Both CPU H/S evaluation and CPU analytic derivative must use the same
  coefficients.
- Both GPU H/S evaluation and GPU analytic derivative must use the same
  coefficients.
- Remove `EqGridTable::eval_with_deriv_into()` from the production SK force
  call graph. Keep Hermite/Neville as reference-only (test feature).
- Delete the numerical FD tail derivative (`dr_step=1e-4`).
- Preserve the repulsive SKF spline coefficients separately (do not refit into
  B-spline).

**Acceptance criteria:**
- [ ] `SkRadialTable` is the single canonical SK radial representation.
- [ ] CPU H/S, CPU dH/dR dS/dR, GPU H/S, GPU dH/dR dS/dR all evaluate the same
      C² B-spline controls.
- [ ] No `EqGridTable::eval_with_deriv_into()` in production code path.
- [ ] No numerical FD tail derivative in production.
- [ ] Gate A passes: basis identities, f64 canonical vs reference, f32 GPU vs
      f64 canonical, knot continuity, C² cutoff transition.
- [ ] Boundary tests at knots ±ε, first physical sample, SK bond distances,
      cutoff tail start, cutoff±ε for V, V', V'' (issue #22).

**Files:**
- `rust_dftb/src/methods/dftb/spline_resample.rs` — canonical preprocessing
- `rust_dftb/src/methods/dftb/sk_table.rs` (or equivalent) — `SkRadialTable`
- `rust_dftb/src/methods/dftb/dftb_hamiltonian.cl` — GPU evaluator
- `rust_dftb/src/methods/dftb/rotation.rs` — ensure derivatives use C² V,V'
- `rust_dftb/tests/sparse_spline_parity.rs` — Gate A

---

### A2 — Fix TC2 hot loop: 2 SpGEMMs/iter, no host sync

**GPT-5.6 issue:** #3 (critical blocker)
**Manifest ref:** §4.7
**Depends on:** nothing
**Blocks:** B6 (sparse SCC loop), all purification-dependent work

**Problem:** TC2 does ~4 SpGEMMs per iteration: computes T=KS and Q=T*K, updates
K, then recomputes KS+KSK for the new K to check convergence. Also reads
`Tr(KS)` to host every iteration even though the branch kernel already
consumes `trace_buf` on device.

**Spec:**
- Compute residual `R_I = ||Q - K||` **before** updating K, using the Q already
  computed (`Q = T*K = KSK`).
- On diagnostic iterations only: reduce idempotency partials on device, read a
  tiny diagnostic packet (trace, residual, flags) to host.
- Normal iteration: 2 SpGEMMs, 0 host sync, swap K↔Knew.
- Do not read `Tr(KS)` to host every iteration. The TC2 branch kernel already
  consumes `trace_buf` on device.
- If residual(old K) < tol: return OLD K without swapping to the unnecessary
  next iterate.
- Keep a slow/debug mode that records every iteration; must not be production
  default.

**Acceptance criteria:**
- [ ] Normal TC2 iteration: exactly 2 SpGEMMs, 0 host scalar reads, 0 queue
      finish for host.
- [ ] Diagnostic iteration: 2 SpGEMMs + 2 cheap reductions + 1 combined scalar
      read.
- [ ] Convergence uses residual before update, not after.
- [ ] No `Tr(KS)` host read in normal iterations.
- [ ] Parity: same convergence result as old implementation.
- [ ] Test verifies SpGEMM count per iteration.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl` — TC2 kernels
- `rust_dftb/src/methods/sparse/gpu_sparse.rs` — `tc2_purify_dev`, `tc2_step_dev`

---

### A3 — Integrate symbolic plans into TC2/NS/K0/W

**GPT-5.6 issue:** #4 (critical blocker)
**Manifest ref:** §4.6
**Depends on:** A2 (TC2 fix)
**Blocks:** B6 (sparse SCC loop)

**Problem:** P4 symbolic plan infrastructure exists (`spgemm_plan_bsym_dev()`)
but TC2 still calls `spgemm_bsym_dev()` and `build_dw_sparse()` uses the generic
runtime-search path. The plan is not used in production.

**Spec:**
- Build plans for all recurring products at frozen topology:
  - `Z * S → T_ZS`, `T_ZS * Z → Znew` (Newton-Schulz)
  - `Z * H → T_ZH`, `T_ZH * Z → K0` (K0 initialization)
  - `K * S → T_KS`, `T_KS * K → Q/Knew` (TC2)
  - `K * Hscc → T_KH`, `T_KH * K → W` (energy-weighted density)
- Note: `Z * H` **can** use the Bsym kernel — H is symmetric; the product C
  need not be symmetric, only the right operand B must be.
- Plans built once during setup, reused across all iterations/displacements.
- Fallback to intersection kernel only if plan memory exceeds threshold.

**Acceptance criteria:**
- [ ] TC2 uses `spgemm_plan_bsym_dev()` for K*S and T_KS*K.
- [ ] Newton-Schulz uses planned products for Z*S and T_ZS*Z.
- [ ] K0 initialization uses planned products for Z*H and T_ZH*Z.
- [ ] build_dw_sparse uses planned products for K*Hscc and T_KH*K.
- [ ] Parity: planned products give same results as intersection kernel.
- [ ] Plan memory reported; fallback threshold documented.

**Files:**
- `rust_dftb/src/methods/sparse/gpu_sparse.rs` — wire plans into all products
- `rust_dftb/src/methods/sparse/bsr4.rs` — plan builders for all product types
- `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl` — planned kernel

---

## Phase B — Sparse SCC solver (GPU-resident)

This is the core of the project. The goal is a fully self-consistent sparse
SCC calculation with no dense `SccResult` anywhere in the loop.

### B1 — SparseSystemWorkspace: persistent GPU state

**GPT-5.6 issues:** #8, #18, #19
**Manifest ref:** §4.12, §4.4
**Depends on:** A3 (plans integrated)
**Blocks:** B2-B7, C1, all production work

**Problem:** Current code allocates buffers, uploads structures, and round-trips
matrices through host in every SCC iteration / force call / Hessian
displacement. Newton-Schulz downloads S and CSR arrays to host. K0 and
spectral_bounds materialize host BSR matrices.

**Spec:**
- One persistent `SparseSystemWorkspace` struct containing:
  - All frozen CSR patterns (`BsrPattern`: row_ptr, col_idx on host + GPU)
  - All GPU matrices (S, H0, Hscc, Z, K, T, W, scratch)
  - All symbolic plans (`SpgemmPlanGpu` for each recurring product)
  - All GPU kernel handles (compiled once, retained)
  - Gamma matrix (dense f32 atomic, GPU-resident)
  - Scratch buffers for reductions, diagnostics
  - `active_orbital_mask` (4 bits/atom for dummy orbital handling)
  - Spectral bound envelope (emin, emax scalars, GPU-resident or cached)
- Created once during setup; reused across SCC iterations, force calls, Hessian
  displacements.
- No `Buffer::builder()`, `GpuBsrStructure::new`, product-mask construction, or
  matrix host roundtrip in any hot loop.
- `BsrPattern` kept on host alongside GPU structure — never download CSR
  arrays merely to reconstruct something known.
- Z stays on GPU after Newton-Schulz convergence. For Hessian: `Z = Z0 + few
  NS corrections against new S`, no download/reupload.

**Acceptance criteria:**
- [ ] `SparseSystemWorkspace` struct defined and used.
- [ ] Zero buffer allocations in SCC iteration / force call / Hessian
      displacement.
- [ ] Zero matrix host roundtrips in SCC iteration.
- [ ] Z stays on GPU after convergence.
- [ ] BsrPattern available on host without GPU download.
- [ ] Test: run 2 SCC iterations, verify no new allocations between them.

**Files:**
- `rust_dftb/src/methods/sparse/workspace.rs` (new) — `SparseSystemWorkspace`
- `rust_dftb/src/methods/sparse/gpu_sparse.rs` — use workspace
- `rust_dftb/src/methods/sparse/bsr4.rs` — `BsrPattern` host-side
- `rust_dftb/src/methods/sparse/mod.rs` — re-exports

---

### B2 — Sparse H0/S assembly (GPU-resident)

**GPT-5.6 issues:** #1 (canonical spline), #17 (dummy isolation)
**Manifest ref:** §4.2, §4.10
**Depends on:** A1 (canonical spline), B1 (workspace)
**Blocks:** B4 (Hscc update), B6 (SCC loop)

**Spec:**
- GPU kernel assembles H0 and S directly into BSR4 device buffers from:
  - Atomic positions (host, uploaded once per geometry)
  - Canonical C² B-spline controls (host, uploaded once during setup)
  - Species-pair table IDs (precomputed)
  - `active_orbital_mask` (precomputed)
- Padded BSR4: every atom gets 4×4 block. H dummy orbitals: S_dd=1, H_dd=E_dummy,
  all active-dummy and interatomic dummy couplings = 0 exactly.
- Dummy diagonal explicitly fixed: SCC update must not turn H_dummy=E_dummy
  into H_dummy=E_dummy+V_H (because S_dd=1). Use `active_orbital_mask` to
  exclude dummy orbitals from SCC H update.
- For Hessian displacement of atom i: patch only physical H/S blocks involving
  atom i, not full rebuild.

**Acceptance criteria:**
- [ ] H0/S assembled on GPU into BSR4 device buffers.
- [ ] Dummy orbitals correctly isolated (S_dd=1, H_dd=E_dummy, couplings=0).
- [ ] SCC H update does not modify dummy diagonal.
- [ ] Hessian displacement patches only blocks involving moved atom.
- [ ] Parity: GPU H0/S matches dense reference at same geometry.
- [ ] No dense matrix allocation in assembly.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_hs_assembly.rs` (new)
- `rust_dftb/src/methods/sparse/sparse_hs_assembly.cl` (new) — GPU kernel
- `rust_dftb/src/methods/sparse/workspace.rs` — H0/S buffers

---

### B3 — Sparse gamma build/patch/matvec (GPU-resident)

**GPT-5.6 issues:** (manifest §4.8, P6)
**Manifest ref:** §4.8
**Depends on:** B1 (workspace)
**Blocks:** B4 (Hscc update), B6 (SCC loop)

**Spec:**
- Dense f32 atomic gamma matrix `Gamma[Natom,Natom]` (~4MB for N=1000),
  GPU-resident.
- Build once per geometry on GPU.
- Per SCC iteration: `V = Gamma * dq` via GPU matvec.
  - Mapping: one work-group per output atom i, each lane sums j=lane,
    lane+WG, ..., then local pairwise reduction.
- Precision variants to benchmark:
  - `GAMMA_FAST`: f32 FMA lane sums + pairwise WG reduction
  - `GAMMA_KAHAN`: compensated lane sums + pairwise WG reduction
  - `GAMMA_FP64_TAIL`: f32 partials + tiny device-f64 final reduction
- For Hessian displacement of atom i: patch only row/column i of Gamma.
- SCC energy reuses converged potential vector (established sign convention).
- Gamma-force terms evaluated once for converged SCC state.

**Acceptance criteria:**
- [ ] Gamma built on GPU, resident in workspace.
- [ ] `V = Gamma * dq` matvec on GPU.
- [ ] Hessian displacement patches only row/column i.
- [ ] Scaling: measure fast/Kahan/f64-tail, choose simplest adequate default.
- [ ] No host roundtrip in matvec.
- [ ] Parity: GPU gamma matvec matches CPU f64 reference.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_scc.rs` (new) — gamma build/patch/matvec
- `rust_dftb/src/methods/sparse/sparse_scc.cl` (new) — GPU kernels
- `rust_dftb/src/methods/sparse/workspace.rs` — Gamma buffer

---

### B4 — Sparse Hscc update (GPU-resident, dummy-safe)

**GPT-5.6 issues:** #17 (dummy isolation)
**Manifest ref:** §4.10
**Depends on:** A1 (canonical spline), B2 (H0/S), B3 (gamma)
**Blocks:** B6 (SCC loop)

**Spec:**
- `Hscc = H0 + 0.5 * S * (V_i + V_j)` computed on GPU, in-place on BSR4
  device buffer.
- `active_orbital_mask` ensures dummy diagonal stays fixed at E_dummy.
  SCC potential V is not applied to dummy orbitals.
- For Hessian displacement: patch H0 blocks involving atom i, then reapply
  SCC shift to those blocks only.

**Acceptance criteria:**
- [ ] Hscc updated on GPU.
- [ ] Dummy diagonal remains E_dummy after SCC update.
- [ ] Parity: GPU Hscc matches dense reference.
- [ ] No host roundtrip in Hscc update.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_scc.rs` — Hscc update
- `rust_dftb/src/methods/sparse/sparse_scc.cl` — GPU kernel

---

### B5 — Sparse K0 initialization + spectral bounds (GPU-resident)

**GPT-5.6 issues:** #17 (dummy exclusion), #19 (allocation-heavy)
**Manifest ref:** §4.12
**Depends on:** A3 (plans), B1 (workspace), B4 (Hscc)
**Blocks:** B6 (SCC loop)

**Spec:**
- Spectral bounds computed on GPU:
  - Persistent `T_ZH = Z * H` GPU buffer (planned product).
  - Row-sum bounds via GPU reduction → only `emin, emax` (two scalars).
  - Exclude dummy orbitals from physical spectral-bound calculation.
  - For Hessian/SCC: recompute cheap conservative bounds device-side, or maintain
    a safely padded envelope with runtime assertion.
- K0 initialization:
  ```
  K0 = P_MK[(ε_max·Z - ZHZ) / (ε_max - ε_min)]
  ```
  using full Z on M_Z in ZHZ (issue #12 — do not project Z to M_K before ZHZ).
  Only the final result is projected to M_K.
- Dummy rows/cols of K0 explicitly initialized to 0.

**Acceptance criteria:**
- [ ] Spectral bounds computed on GPU, only 2 scalars read to host.
- [ ] Dummy orbitals excluded from spectral bounds.
- [ ] K0 uses full Z on M_Z in ZHZ, projects only final result to M_K.
- [ ] K0 dummy rows/cols = 0.
- [ ] No host BSR materialization.
- [ ] Parity: K0 matches dense reference.

**Files:**
- `rust_dftb/src/methods/sparse/gpu_sparse.rs` — `build_k0_dev`, `spectral_bounds_dev`
- `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl` — bounds reduction kernel

---

### B6 — Sparse SCC loop (self-consistent, GPU-resident)

**GPT-5.6 issues:** #2 (no sparse SCC), #3 (TC2 fix)
**Manifest ref:** §4.12
**Depends on:** A2 (TC2 fix), A3 (plans), B1-B5
**Blocks:** C1 (force), D1-D5 (gates)

**Spec:**
The production SCC loop, entirely GPU-resident:
```
for scc_iter:
    V     = Gamma * dq                    # GPU matvec (B3)
    Hscc  = H0 + 0.5*S*(V_i + V_j)       # GPU update (B4), dummy-safe
    bounds = spectral_bounds(Hscc, S)     # GPU reduction (B5)
    Kinit = k0_initializer(Hscc, S, Z, bounds)  # GPU (B5)
    K     = TC2(Kinit)                     # GPU purification (A2, A3)
    qnew  = Mulliken(K, S)                 # GPU charge extraction
    q     = mix(q, qnew)
    converge?
```
- No dense `SccResult` anywhere in this loop.
- No matrix host roundtrip.
- No buffer allocation.
- Diagnostic scalars (energy, dq residual) read only at controlled intervals.
- Warm-start: for Hessian displacement, start from central q/Z state.

**Acceptance criteria:**
- [ ] Full SCC loop runs entirely on GPU.
- [ ] No `SccResult` or dense matrix in the loop.
- [ ] No buffer allocation in the loop.
- [ ] No matrix host roundtrip in the loop.
- [ ] Converges to same q/energy as dense reference (within tolerance).
- [ ] Diagnostic packet read only at controlled intervals.
- [ ] Test: verify zero allocations between SCC iterations.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_scc.rs` — SCC loop driver
- `rust_dftb/src/methods/sparse/gpu_sparse.rs` — orchestration
- `rust_dftb/src/methods/sparse/sparse_mulliken.cl` (new) — GPU Mulliken

---

### B7 — Sparse Mulliken charge extraction (GPU-resident)

**Depends on:** B6 (SCC loop)
**Blocks:** B6 (SCC loop needs q)

**Spec:**
- `q_A = 2 * sum_{μ∈A} (K*S)[μ,μ]` computed on GPU.
- Device reduction per atom → charge vector on GPU.
- No host roundtrip during SCC; charge read to host only at convergence or
  diagnostic intervals.

**Acceptance criteria:**
- [ ] Mulliken charges computed on GPU.
- [ ] Parity: matches dense reference.
- [ ] No host roundtrip during SCC iteration.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_mulliken.cl` (new)
- `rust_dftb/src/methods/sparse/gpu_sparse.rs` — launch

---

## Phase C — GPU-resident sparse analytic forces

**Depends on:** B6 (sparse SCC → D/W), dense agent's analytic force work
**GPT-5.6 issue:** #16
**Manifest ref:** §4.3, §4.4, §7

This phase integrates after the dense agent finishes analytic forces. The
mathematical formulas are the reference; the implementation is a dedicated GPU
kernel, not a CPU bridge.

### C1 — GPU pair-force kernel

**Spec:**
- Kernel 1 (one directed/undirected physical pair per work-item):
  - Evaluate SK V(r), V'(r) from canonical C² B-spline (A1).
  - Analytic angular derivatives (existing rotation machinery, issue #15).
  - Read K_ij and W_ij from BSR4 device buffers (skip dummy orbitals using
    `active_orbital_mask`).
  - Contract: `F_pair[a] = 2 * ANG2BOHR * Σ(D·dH/dR - W·dS/dR)`.
  - Output `float4 pair_force[3]` (x,y,z + padding) to a pair-force buffer.
- Precomputed per pair: BSR block index in M_HS, transpose index, physical
  orbital counts (ni, nj), species-pair table ID.
- No CSR binary searches in the force loop.

### C2 — GPU atom-gather kernel

**Spec:**
- Kernel 2 (one atom per work-group):
  - Gather incident pair_forces from all pairs involving this atom.
  - Sum into `F_atom[3]`.
  - Deterministic (no atomics, fixed order).
- Include SCC double-counting force and repulsive force.

### C3 — D/W from sparse purification (no D allocation)

**GPT-5.6 issue:** #8
**Spec:**
- Do not create D = 2K as a separate matrix.
- Pass K + spin_factor=2 to the force contraction.
- Fuse 2.0 into the W product or let the force contraction know W carries
  the spin factor.
- W = 2*K*Hscc*K computed via planned SpGEMM (A3), projected to M_HS.
- Measure η_W = ||W-W^T||/||W|| as diagnostic (issue #9). If ~f32 roundoff,
  do not symmetrize. For force contraction, use (W_ij + W_ji^T)/2 on the pair.

**Acceptance criteria (Phase C):**
- [ ] GPU pair-force kernel produces correct forces.
- [ ] GPU atom-gather kernel produces correct forces.
- [ ] No D matrix allocation.
- [ ] No CPU bridge / no `non_scc_electronic_force()` in production.
- [ ] No CSR binary searches in force loop.
- [ ] Parity: GPU sparse force matches dense f64 analytic force.
- [ ] Newton's third-law residual checked.
- [ ] Force finiteness and repeatability checked.
- [ ] No dummy-orbital contribution to force.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_forces.rs` — driver
- `rust_dftb/src/methods/sparse/sparse_forces.cl` (new) — GPU force kernels
- `rust_dftb/src/methods/sparse/workspace.rs` — pair-force buffer

---

## Phase D — Correctness gates

### D1 — Redo Gate C: real Si/H locality sweep

**GPT-5.6 issues:** #10, #11, #12
**Manifest ref:** §4.5, §5 Gate C
**Depends on:** B6 (sparse SCC)

**Spec:**
- Use real passivated Si/H systems: Si29H36, Si~80H..., Si~150H...
- Sweep R_K and R_Z independently/staged.
- Use full Z on M_Z in ZHZ; project only final K0 to M_K (issue #12).
- Compute exact outside-mask leakage:
  ```
  Q = KSK on validation support
  R_leak = sqrt(sum_{outside M_K} |Q_ij|²)
  ```
  via GPU reduction (issue #11). No approximation.
- Test whether a **fixed** R_K gives stable errors as N grows (the actual
  linear-scaling question).
- Do not use `+2I` to create a gap — it shifts all eigenvalues by 2 and
  doesn't change level spacing (issue #10).
- Record: energy error, charge error, force error, Tr(KS)-Nocc, R_in, R_leak,
  R_H, iteration counts, wall time, memory.

**Acceptance criteria:**
- [ ] Real Si/H systems used (not 5-atom toy chain).
- [ ] R_leak computed exactly (not norm difference).
- [ ] R_K and R_Z tested independently (Z not projected before ZHZ).
- [ ] Stability of fixed R_K as N grows measured.
- [ ] 2D heatmap of (R_K, R_Z) → errors/residuals/timing.

**Files:**
- `rust_dftb/tests/locality_sweep.rs` — rewrite

---

### D2 — Redo Gate E: determinism + Hessian h plateau

**GPT-5.6 issue:** #13
**Manifest ref:** §5 Gate E
**Depends on:** C1-C3 (analytic sparse force)

**Spec:**
- Use actual SCC analytic sparse forces (not FD of energy).
- A. Same-geometry repeatability: cold q start, central warm start, perturbed q
  starts, fast vs tight SCC/TC2/Z tolerances, fast-f32 vs compensated variants.
  Measure force spread.
- B. Sparse-vs-dense bias: `F_sparse - F_dense` as separate quantity.
- C. h sweep: 0.01, 0.02, 0.05, 0.10 Å. Use **unsymmetrized** Hessian:
  ```
  H[:,a] = -(F(R+h*e_a) - F(R-h*e_a)) / (2h)
  ```
  Measure `||H-H^T||/||H||` on raw matrix. Compare to dense f64 analytic-force
  Hessian + optional 5-point reference. Do not include h_ref in candidate list.

**Acceptance criteria:**
- [ ] Forces from analytic sparse D/W, not FD of energy.
- [ ] Hessian not symmetrized by construction.
- [ ] h_ref not in candidate list.
- [ ] Raw Hessian asymmetry measured.
- [ ] Force bias separated from force noise.
- [ ] Stable h plateau chosen.

**Files:**
- `rust_dftb/tests/gate_e_determinism.rs` — rewrite

---

### D3 — Gate F: geometry optimization at method's own minimum

**Manifest ref:** §5 Gate F
**Depends on:** D2 (Gate E)

**Spec:**
- Optimize small H-passivated Si with sparse model using analytic forces.
- Coarse stage: FIRE, masks can rebuild at explicit checkpoints.
- Final stage: frozen masks, FIRE or L-BFGS.
- Stopping target tied to measured convergence/repeatability (from Gate E).
- One sparse SCC pipeline run per geometry (not per displacement).
- For small systems: compute full Hessian if cheap.

**Acceptance criteria:**
- [ ] FIRE uses analytic sparse forces (not FD).
- [ ] One pipeline run per geometry step.
- [ ] Converges to stationary minimum.
- [ ] Force tolerance based on measured repeatability floor.

**Files:**
- `rust_dftb/tests/gate_f_geom_opt.rs` — rewrite

---

### D4 — Gate G: same-geometry Hessian parity

**Manifest ref:** §5 Gate G
**Depends on:** D3 (Gate F)

**Spec:**
- At identical frozen coordinates: sparse f32 Hessian vs dense f64 Hessian.
- Compare: max abs element error, ||ΔH||_F/||H||_F, η_asym, rigid-mode leakage,
  frequencies, MAC, subspace overlap for near-degenerate groups.
- Relative frequency error for ordinary modes, absolute error for low modes.

**Files:**
- `rust_dftb/tests/hessian_parity.rs` (new)

---

### D5 — Gate H: spectra at each method's own minimum

**Manifest ref:** §5 Gate H
**Depends on:** D4 (Gate G)

**Spec:**
- Optimize each method independently, compare spectra at each model's own
  minimum.
- For any significant imaginary sparse mode absent from dense reference:
  - Mode localization / atoms involved
  - Raw vs projected eigenvalue
  - Force norm at minimum
  - SCC/TC2/Z residuals, R_K, R_Z, R_leak, R_H, h, precision mode
  - Energy scan along ±mode

**Files:**
- `rust_dftb/tests/nanocrystal_vib.rs` (new)

---

## Phase E — Performance (after correctness)

### E1 — Real performance counters

**GPT-5.6 issue:** #20
**Depends on:** B1 (workspace)

**Spec:**
- Real counters at GPU wrapper/runtime level:
  - Buffer allocation count, current/peak GPU bytes
  - Kernel launch count, blocking read count/bytes
  - Queue finish count, structure/plan upload bytes
- Stage timings explicitly exclusive or hierarchical, never sum overlapping.
- Type-level separation of production sparse API from dense reference.

**Files:**
- `rust_dftb/src/methods/sparse/gpu_sparse.rs` — `SparsePerfStats` real counters

---

### E2 — Cell-list mask construction

**GPT-5.6 issue:** #21
**Depends on:** nothing

**Spec:**
- `build_geometric_mask()`: cell list instead of all-pairs.
- `build_product_mask()`: stamping array (`marks[j] != stamp`) instead of
  `neighbors.contains(&j)`.
- Setup only, so not urgent, but easy.

**Files:**
- `rust_dftb/src/methods/sparse/masks.rs`

---

### E3 — Degree buckets

**GPT-5.6 issue:** #7
**Depends on:** A3 (plans)

**Spec:**
- Bucket rows by left degree: ≤32, ≤64, ≤128, ≤256.
- Compile/use appropriate local-memory capacities per bucket.
- For Si density at ~7Å, expect 64/128 to dominate.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl`
- `rust_dftb/src/methods/sparse/bsr4.rs`

---

### E4 — Packed plans + dead code cleanup

**GPT-5.6 issue:** #5
**Depends on:** A3 (plans)

**Spec:**
- Delete `A_col`, `C_col`, `lcol[MAX_LEFT_BLOCKS]` from planned kernel (dead
  code).
- Pack `plan_a_idx` (8 bits) + `plan_b_idx` (24 bits) into one `uint`.
- Fail loudly on overflow.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl`
- `rust_dftb/src/methods/sparse/bsr4.rs`

---

### E5 — Lane mapping benchmark

**GPT-5.6 issue:** #6
**Depends on:** A3 (plans)

**Spec:**
- Benchmark current 16-lane (16 scalar C_rc) vs alternative (m,c) mapping.
- Use realistic matrices: N=300, 600, 1000, degree distributions at R_K~5-10Å,
  100-1000 repeated products.
- Profile actual kernel events, not launch overhead.

**Files:**
- `rust_dftb/src/methods/sparse/sparse_bsr4_purification.cl`
- `rust_dftb/tests/spgemm_plan.rs` — realistic benchmark

---

### E6 — W/K symmetrization policy

**GPT-5.6 issue:** #9
**Depends on:** C3 (D/W)

**Spec:**
- Measure η_W = ||W-W^T||/||W||. If ~f32 roundoff, don't symmetrize.
- For force: use (W_ij + W_ji^T)/2 on the pair.
- Benchmark K symmetrization: every-iter vs diagnostic-iter vs threshold-gated.

**Files:**
- `rust_dftb/src/methods/sparse/gpu_sparse.rs`

---

### E7 — Gate I: scaling and whole-program profile

**Manifest ref:** §5 Gate I
**Depends on:** D5 (Gate H), E1-E6

**Spec:**
- N ~ 60, 150, 300, 600, 1000, 1600.
- Fit/report separate scaling for: H/S assembly, Z/K init/TC2, Gamma*dq,
  complete SCC iteration, analytic force, full Hessian, final eigensolve.
- Do not assert α<1.5 for total SCC while direct gamma remains O(N²).

**Files:**
- `rust_dftb/tests/sparse_scaling.rs` (new)

---

### E8 — Gate J: production N~300, then 800–1000

**Manifest ref:** §5 Gate J
**Depends on:** E7 (Gate I)

**Spec:**
- N~300 first real success criterion.
- Every production result records: git commit, SK-set hash, spline
  representation, physical/padded orbital count, R_HS/skin/R_K/R_Z/validation
  mask, SCC/Z/TC2 settings, precision mode, optimizer settings, Hessian h,
  GPU/device/driver, complete timing breakdown + host sync count.

**Files:**
- `rust_dftb/tests/nanocrystal_perf.rs` (new)

---

## Dependency graph

```
A1 (canonical spline) ──┬──> B2 (H0/S assembly) ──> B4 (Hscc) ──> B6 (SCC loop)
                        └──> C1 (GPU force)
A2 (TC2 fix) ────────────────> B6 (SCC loop)
A3 (plans integrated) ──┬──> B5 (K0 init)
                          ├──> B6 (SCC loop)
                          └──> C3 (D/W)
B1 (workspace) ─────┬──> B2, B3, B5, B6, C1
                     └──> E1 (perf counters)
B3 (gamma) ──────────────> B4 (Hscc) ──> B6 (SCC loop)
B6 (SCC loop) ───────────> C1 (force) ──> D1 (Gate C)
                                           ├──> D2 (Gate E)
                                           ├──> D3 (Gate F)
                                           ├──> D4 (Gate G)
                                           └──> D5 (Gate H) ──> E7 (Gate I) ──> E8 (Gate J)
E2-E6 (perf tuning) ───────────────────────────────────────────> after D5
```

## Current focus

**Phase B (sparse SCC)** is the current focus, since the dense agent handles
analytic forces in parallel. Within Phase B, the order is:

1. B1 — SparseSystemWorkspace (foundation for everything)
2. A1 — Canonical C² spline (can proceed in parallel with B1)
3. A2 — TC2 fix (can proceed in parallel with B1)
4. A3 — Plan integration (depends on A2)
5. B2 — Sparse H0/S assembly (depends on A1, B1)
6. B3 — Sparse gamma (depends on B1)
7. B4 — Sparse Hscc (depends on A1, B2, B3)
8. B5 — Sparse K0 + bounds (depends on A3, B1, B4)
9. B6 — Sparse SCC loop (depends on A2, A3, B1-B5)
10. B7 — Sparse Mulliken (depends on B6)

Phase C (GPU forces) starts after B6 is done and the dense agent's analytic
force work is available for reference.
