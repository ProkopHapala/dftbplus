# Sparse Nanocrystal Vibrations — Master Task Breakdown

**Created:** 2026-09-09
**Source of truth:** `Sparse_Nanocrystal_Vibrations.manifest.md` (§13 + §14 checklists)
**GPT-5.6 reviews:** `Sparse_Nanocrystal_Vibrations.chat.md` line 2064 (review 1, commit `b269ab6`) and line 3276 (review 2, commit `e965ae0`)
**Status report:** `Sparse_Nanocrystal_Vibrations.report.md`

> **2026-09-11 — second review.** Review 2 found three remaining blockers and
> the root cause of the N4 mystery (a missing `sqrt` contract bug, not an f32
> floor). **Phase F below is now the active work** — it precedes all remaining
> Phase-E tuning. See manifest §14 for the verified findings (file:line) and
> §14.8 for the stricter "done" criteria this phase enforces.

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
**Status (2026-09-12):** prototype driver `sparse_vibrations` (FD Hessian →
mass-weighted eigen → freqs) exists in `dftb_engine`; first end-to-end
Si10H16 spectrum produced (see report 2026-09-12). Gate NOT satisfied —
no dense-f64 Hessian parity comparison yet.

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

## Phase F — Second-review corrections (GPT-5.6 review 2, commit `e965ae0`)

**Manifest ref:** §14 (each item verified against code with file:line).
**Priority: before all remaining Phase-E tuning.** These are correctness
contracts and architecture, not optimizations. Ordering follows review §"What
I would implement next" = manifest §14.7.

### Phase F status — 2026-09-11 (measured, see report labbook)

| Task | Status | Evidence |
|------|--------|----------|
| F1 | **DONE** | `compute_z` device-resident, full-write identity, plan_zs/plan_tz, warm-Z + cold fallback; 4 `sparse_system` tests pass |
| F2 | **DONE** | direct BSR H/S assembly, independent M_HS/M_K/M_Z + plans, Verlet skin guard fails loud; `test_f2_direct_bsr_hs_parity` max\|dH\|,\|dS\|<5e-7 vs dense |
| F3 | **DONE** | `trace_kh0_dev` masked energy (no dense K/trace); `n_buf_allocs` counter; `test_f3_no_device_allocs_in_scc` = 0 growth across scc/forces/geom2 |
| F4 | **DONE** | `finalize_scc` consistent (q_in,K,H_scc,V) + `rh_stationarity` one-SpGEMM R_H; G3.2 passes |
| F5 | **DONE (stage 1)** | `SparseDWWorkspace` device T/W + `sparse_forces_bsr` pair contraction; G3.3 D/W parity + G3.4 E-Fd agreement |
| F6 | **DONE** | repulsive splines + γ matrix + pair lists cached in `new()`; `SystemContext` per-call (can't store — `&'a SkData` borrow) |
| F7 | **PARTIAL** | normalized R_I=‖KSK−K‖/‖K‖ (R12) + masked trace/Mulliken/Hscc + dummy-lane checks done; packed Mulliken read done (2N buffer, 1 read); multi-acc SpGEMM done (1.06× on 8-atom toy — latency-bound); remaining: packed residual reads |
| F8 | **PENDING** | = Phase E items, after D-gates revalidated |

Known designed-red tests (diagnostic, not regressions): `gate_e_determinism`
E-B (its own panic says it stays red until the analytic sparse force exists —
it now does, test upgrade is a Gate-D item), `locality_sweep` Gate C
(5-atom toy is not a locality test — needs real gapped Si/H cluster),
`spline_resample` sin test (pre-existing, untouched file).

### F1 — Newton–Schulz contract bugs (manifest R5+R6, solves N4)

**Problem:** `identity_residual_scalar_dev` returns ‖I−T‖² (missing `sqrt`) —
the "R_Z=1.9e-5 vs max|Z−S⁻¹|=2.18e-3" gap was a squared norm, not an f32
failure. Separately, `bsr4_build_identity_dev` writes only diagonal blocks and
`scale_dev` then scales **all** blocks → geometry ≥2 starts from stale
off-diagonals (neither αI nor a valid warm start).

**Spec:**
- Make the contract explicit: rename to `identity_residual_sq` or add `.sqrt()`.
- Cold start: one kernel writes every structural entry (physical diag→1,
  else 0); OR warm start: keep previous/central Z and NS-correct against new
  S (first-order correction Z−Z·δS·Z, few iters for ±h). Same central Z for
  +h and −h independently — no history asymmetry.
- After fix: remove the per-iteration full-T download in
  `SparseSystemWorkspace::compute_z`; validate device reduction vs host f64
  reduction of the identical T; route Z·S / T·Z through `plan_zs`/`plan_tz`.

**Acceptance criteria:**
- [ ] `test_newton_schulz_inverse_dev` green with the **correct** norm, and a
      cross-check test (device residual vs host f64 ‖I−T‖ of the same
      downloaded T) agrees to f32 reduction accuracy.
- [ ] One scalar diagnostic read per NS iteration (no matrix transfer).
- [ ] Two-consecutive-geometry regression test: second geometry's Z is
      correct (no stale off-diagonals / valid warm start).

**Files:** `gpu_sparse.rs`, `sparse_system.rs`, `sparse_bsr4_purification.cl`, `bsr4.rs`

---

### F2 — Remove dense storage from `SparseDftb`; independent masks; real skin (manifest R1, R4, R15)

**Problem:** `SparseDftb` still holds `h0_phys`/`s_phys` f64[N²] + four padded
f32[(4N)²] arrays (~512 MB at N=1000); `set_coords` builds dense `DMatrix`
then re-sparsifies. One mask serves H/S, K and Z. The skin is paid for but
`set_coords` rebuilds the O(N²) mask every geometry and demands exact
equality — so it is never used.

**Spec:**
- Direct sparse H/S assembly into BSR values over the frozen physical pair
  list (SK eval + rotation per pair → block b(i,j); onsite diag; dummy S=1,
  H=E_dummy). CPU first. No `DMatrix`, no `*_pad`, no
  `fill_bsr_values_from_dense` in production.
- Independent `R_HS`/`R_K`/`R_Z` masks + their product plans. ZHZ uses full
  M_Z; only K0 is projected to M_K.
- Verlet skin: structural support at R_phys+R_skin built once, build coords
  stored, rebuild only when 2·max|ΔR_i| > R_skin; in-support pairs beyond the
  physical cutoff carry exactly-zero H/S. Hessian: skin covers all ±h → zero
  rebuilds, zero mask jitter.
- Cell list for `build_geometric_mask` (E2) can land here since the mask
  builder is being touched anyway.

**Acceptance criteria:**
- [ ] Zero dense Norb×Norb host allocation inside `SparseDftb` (firewall
      counter proves it).
- [ ] R_K/R_Z sweepable independently (Gate-C prerequisite).
- [ ] `set_coords` with |ΔR| < skin/2 does not rebuild the mask; exhausted
      skin aborts loudly, never silently mutates topology.
- [ ] Hessian-displacement H/S update touches only blocks of the moved atom.

**Files:** `sparse_dftb.rs`, `bsr4.rs`, `masks.rs`/`bsr4.rs` mask builders,
new direct-assembly routine (extend `sparse_hs_assembly` if present, else
minimal new module per §9)

---

### F3 — Sparse energy; kill per-iteration K densification and spare products (manifest R7, R8, R10)

**Problem:** `k_to_dense_into` + `trace_ab` over the padded N² run every SCC
iteration; `mulliken_charges` recomputes K·S although `t_ks` holds it;
`compute_k0_from_hscc` computes ZH twice; `plan_zh`/`plan_bz` are built but
unused; `mulliken_dev`/`gershgorin_bounds_dev`/`identity_residual_scalar_dev`/
reduction helpers allocate GPU buffers per call.

**Spec:**
- E_H0 = 2·Tr(K·H0) over M_HS blocks only: precomputed HS-block→K-block map,
  f32 FMA block dot products on GPU → one partial per atom → host f64 sum
  (~4 kB transfer). Per-iteration energy becomes a configurable diagnostic.
- Reuse final `t_ks` for Mulliken — delete the extra K·S.
- Single ZH product feeding both Gershgorin bounds and ZH·Z; both through
  `plan_zh`/`plan_bz`.
- Move q_atom buffer, gershgorin partials, reduction scratch, and a packed
  diagnostic buffer into `SparseSystemWorkspace`.

**Acceptance criteria:**
- [ ] No `k_pad`, no `trace_ab` over padded N², no matrix download inside SCC.
- [ ] ≥2 fewer SpGEMMs per SCC iteration (measured by launch counter).
- [ ] Allocation counter: zero `Buffer::builder` between `scc()` entry/exit.
- [ ] E_H0 parity vs dense `2·Tr(K·H0)` at the same K.

**Files:** `sparse_dftb.rs`, `sparse_system.rs`, `gpu_sparse.rs`

---

### F4 — Stationary SCC finalization + final R_H (manifest R3, R11)

**Problem:** `finalize_scc` stores `q_fin` but keeps `V`/`H_scc` of `q_new` —
energy, force, and stored state are not one stationary electronic state at
finite tolerance. `r_h` is NaN in production.

**Spec:**
- Explicit contract: q_in → H[q_in] → K → q_out. Iterate the finalization
  until R_SCC = rms(q_out−q_in) < final force tolerance; store q_in and q_out
  for diagnostics; V, H_scc, K, W, F all belong to the identified state.
- After final convergence only: A = H·(KS) with the existing T=KS →
  R_H = ‖A−Aᵀ‖_F/(2‖A‖_F+ε). One extra SpGEMM at finalization, none in-loop.

**Acceptance criteria:**
- [ ] Energy/force/diagnostics demonstrably use one consistent state (test:
      force evaluated with stored q and stored H_scc agree to the SCC
      residual, not mixed-state).
- [ ] `r_h` populated at every converged SCC; reported in SparseDftbScc.

**Files:** `sparse_dftb.rs`, `sparse_system.rs`, `gpu_sparse.rs`

---

### F5 — Sparse force via `SparseDWWorkspace`, then GPU contraction (manifest R2)

**Problem:** `sparse_analytic_forces` → `dw_from_k_padded` runs two dense f64
triple-loop matmuls O((4N)³) — catastrophic at N≥300.

**Spec:**
- Stage 1 (immediate): GPU T = K·Hscc (planned), W = P_{M_HS}(T·K) via
  `SparseDWWorkspace`; download only W[M_HS] and K[M_HS]; existing tested CPU
  pair contraction. No D=2K materialization (factor 2 inside contraction);
  η_W = ‖W−Wᵀ‖/‖W‖ measured as diagnostic before any symmetrize pass.
- Stage 2 (after Stage-1 parity): GPU pair-force + atom-gather kernels
  (C1/C2), no atomics, precomputed pair→block maps.

**Acceptance criteria:**
- [ ] No O(N³) host work anywhere in `forces()`.
- [ ] Stage-1 force parity vs dense f64 at same geometry/state (Gate B
      standard) before Stage 2 starts.
- [ ] Downloaded bytes per force call = O(nnz(M_HS)), not O(N²).

**Files:** `sparse_forces.rs`, `sparse_dftb.rs`, `gpu_sparse.rs`,
`sparse_bsr4_purification.cl` (stage 2)

---

### F6 — Per-geometry / static precomputation (manifest R9, R16)

**Problem:** `compute_intra_shifts` recomputes every R_ij and γ(R_ij) (incl.
exponentials) every SCC iteration; `compute_forces_from_dw` rebuilds
SystemContext/GammaTable/neighbors and re-parses repulsive splines per call;
`repulsive_energy` re-parses SK files every `set_coords`. Also SK `cutoff()`
(Bohr) is passed raw to `NeighborBuilder` on Å coords → ~1.89× oversized
neighbor radius.

**Spec:**
- Dense f64 G[N²] built once per geometry (8 MB @ N=1000); V = G·Δq in CPU
  f64 per iteration (~10⁶ FMA — cheaper than the GPU sync it replaces);
  Hessian patches only the moved atom's row/column, O(N).
- `SparseDftb::new` owns SystemContext, repulsive tables (one
  `parse_all_repulsive`), species-pair indices, physical pair list, gamma
  coefficients — never reconstructed.
- Convert SK cutoff Bohr→Å before `NeighborBuilder` (and audit the same
  pattern wherever SK cutoffs feed distance comparisons).

**Acceptance criteria:**
- [ ] Zero SK file I/O inside SCC / force / Hessian-displacement loops.
- [ ] γ evaluated O(N²) once per geometry, O(N) per Hessian displacement.
- [ ] Neighbor count consistent with the physical cutoff (log nnz/pair count).

**Files:** `sparse_dftb.rs`, `scc.rs`/`shifts.rs` call sites, `forces.rs`

---

### F7 — f32 tolerance + selective-precision pass (manifest R12–R14, §14.5)

**Depends on:** F1–F6. **Problem:** raw ‖KSK−K‖_F < 1e-4 demands ~10⁻⁷
RMS/scalar at N=1000 (f32 machine epsilon) — mislabels normal f32 saturation
as failure and wastes iterations. Hscc/Mulliken apply to dummy lanes.

**Spec:**
- Normalized convergence: r_I = ‖KSK−K‖/max(‖K‖,ε), r_N = |Tr(KS)−Nocc|/Nocc;
  raw norms remain printed diagnostics.
- Hscc on GPU: upload only V[N] per iteration; `bsr4_build_Hscc` respects the
  active-orbital mask so the dummy diagonal stays E_dummy exactly.
- Mulliken with per-atom orbital mask (Si lanes 0,5,10,15; H lane 0); dummy
  occupation reported separately; charge check = Σq_A vs 2·Tr(KS) (tight),
  not |Σq−N_e|<0.5.
- One packed diagnostic read (trace + R_I² + flags); benchmark check_every
  2–3.
- Multi-accumulator SpGEMM (2–4 partials over successive plan terms)
  benchmarked **before** any Kahan-in-SpGEMM; compensated f32 only for the
  long signed reductions (gamma matvec, force gather, trace).

**Acceptance criteria:**
- [ ] Convergence no longer demands ~10⁻⁷ RMS/scalar at N=1000.
- [ ] Dummy occupation exactly zero-structure (reported, asserted).
- [ ] Multi-accumulator SpGEMM benchmark recorded (speed + residual depth).

**Files:** `gpu_sparse.rs`, `sparse_system.rs`, `sparse_bsr4_purification.cl`,
`sparse_dftb.rs`

---

### F8 — Kernel micro-opts last (manifest §14.5/§18)

Packed plan indices (8+24 bit), degree buckets, fused W→force, lane-mapping
benchmark — the existing Phase-E items (E3–E6). Only after F1–F7 are verified
by USER.

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

── 2026-09-11: Phase F (second-review corrections) gates everything below ──
F1 (NS contracts) ──> F2 (de-densify + masks/skin) ──> F3 (sparse energy, no spare SpGEMMs)
F3 ──> F4 (stationary SCC + R_H) ──> F5 (sparse W/force) ──> D-gates
F2 ──> F5 ; F6 (per-geometry precompute) ──> F7 (f32 tolerances/precision) ──> F8 = E3–E6
E2-E6 (perf tuning) ───────────────────────────────────────────> after D5
```

## Current focus

**Phase F (second-review corrections)** is the current focus — it fixes the
physics/architecture contracts that the partially-complete A/B phases left
behind. Order (manifest §14.7):

1. F1 — NS contract bugs (missing sqrt + stale identity) → revalidate
2. F2 — remove dense storage from `SparseDftb`, independent R_HS/R_K/R_Z, Verlet skin
3. F3 — sparse energy, no per-iteration K densification, no spare SpGEMMs, no in-loop allocs
4. F4 — stationary SCC finalization + final R_H
5. F5 — sparse force via `SparseDWWorkspace` (staged → GPU kernels)
6. F6 — per-geometry precomputation (gamma, repulsive, context) + Bohr/Å fix
7. F7 — normalized tolerances + selective precision
8. F8 — kernel micro-opts (= Phase E items, last)

Only then resume gates D1–D5 → E7–E8. Phase-B items not yet absorbed by F
(B2's GPU assembly, B3's GPU gamma option) remain as *later* GPU-side upgrades
— F2/F6 make the CPU versions correct and cheap first.

Phase C (GPU forces) starts after F5 stage-1 parity is demonstrated and the
dense agent's analytic force work is available for reference.
