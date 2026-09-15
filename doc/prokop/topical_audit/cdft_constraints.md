---
type: TopicalAudit
title: CDFT Fragment-Charge Constraints
tags: [topic, cdft, gpu, opencl, batched, pcet, cross-language]
---

# CDFT — fragment-charge constraints on the dense GPU solver

## Summary

Constrained DFT (CDFT): fragment Mulliken-charge constraints
`Q_F = Σ_{A∈F} Δq_A = Q_F^target` via Lagrange multipliers λ_F, used for
charge-localized diabatic states (PCET, electron-transfer surfaces).
Implemented as a pure add-on to the dense `GpuDftb` solver: the constraint
is an on-site shift `V_A → V_A + λ_F·w_A`, i.e. one kernel launch per
h_scc rebuild plus a host-side f64 λ outer loop. Verified end-to-end on
the pyridone-dimer 2D proton-transfer scan (3 surfaces: neutral,
Q₁=−1 e, Q₁=+1 e).

Spec: `doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Dense_Multi_CDFT.spec.md`
Design discussion: `…/Dense_Multi_CDFT.chat.md`
User guide: `doc/prokop/userguide/cdft.md`
Test: `rust_dftb/tests/gpu_cdft.rs`

## The key structural fact

SCC-DFTB builds `H_scc[μν] = H0 + ½·S[μν]·(V_A + V_B)`. A charge constraint
is an on-site potential shift `V_A → V_A + λ_F·w_A`, i.e.

    H_scc[μν] += ½·λ_F·S[μν]·(w_μ + w_ν),   w_μ = 1 iff atom(μ) ∈ F.

No new eigensolver / mixer / density machinery. Consequences:

- `e_band = Σ f·ε` picks up `+λ_F·Q_gross(F)` — the shift couples to the
  GROSS Mulliken population `Σ_{a∈F}(q0_a+Δq_a)`, **not** the excess Δq.
  First implementation subtracted only `λ·Q_F` and leaked `λ·Q0_F`
  (~0.38 Ha × ~35 e = ~13 Ha on pyridone — CT state appeared 12 Ha BELOW
  the neutral state, caught by the variational sanity check).
  `cdft_energies()` subtracts `λ·(Q_F + Q0_F)` — **the λ·Q0_F term is
  the easy-to-miss term** (constant per fragment but large).
- The force path derives W from h_scc eigenpairs → the constraint force
  `−λ_F·dQ_F/dR` enters automatically through `dS/dR`. Forces are the
  constrained-surface gradient at fixed λ; the `(Q−Q_t)·dλ/dR` remainder
  vanishes at convergence.
- λ is per-replica per-fragment → one batch = a diabatic-state ladder.

## Implementations

| Language | Location | Status | Notes |
| --- | --- | --- | --- |
| OpenCL | `rust_dftb/src/qmqm/gpu_cdft.cl` (`cdft_hscc_shift_batched`) | active | one WG/system, local-cached λw, gated on `active` |
| Rust | `rust_dftb/src/qmqm/gpu_cdft.rs` (`GpuCdft`) | active | frag map, λ buffer, targets, `q0_frag`, damped-secant/bracket state |
| Rust | `rust_dftb/src/qmqm/gpu_scc_plan.rs::enq_dq_v_hscc` | active | `plan.cdft: Option<GpuCdft>`; `None` = zero cost |
| Rust | `rust_dftb/src/qmqm/gpu_dftb.rs` (`set_cdft`, `cdft_scc`, `cdft_set_lam`, `cdft_qfrag`, `cdft_energies`, `clear_cdft`) | active | outer-λ loop: continuation ramp + damped secant + basin reset + best-λ restore |
| Rhai | `rust_dftb/src/bin/dftb_engine.rs` (`gpu_cdft*`) | active | see `userguide/cdft.md` |
| Rust test | `rust_dftb/tests/gpu_cdft.rs` | active | H2O {O} target + 8-replica ladder |
| Rhai example | `rust_dftb/scripts/scan2d_pyridone_cdft.rhai` + `plot_scan2d_cdft.py` | active | 3-surface PCET demo → `debug/pyridone_2d_scan_cdft.png` |
| Fortran | `src/dftbp/` | unported | DFTB+ constrained ground state exists upstream — parity reference, not checked yet |

The λ outer loop and Q_F reduction run on the host in f64
(O(batch·n_atoms) scalars, off the GPU hot path).

## Parity status

- **Self-consistency**: constrained E ≥ E₀ asserted in `tests/gpu_cdft.rs`
  (caught the Q_gross accounting bug); `clear_cdft` restores E₀ to 3e-9.
- **vs Fortran DFTB+**: not checked. DFTB+ has constrained ground state
  with Mulliken constraints — the formalism (Lagrangian in the density
  matrix, on-site shift) matches; use it for a real parity check when
  needed.
- **CPU Rust path** (`methods/dftb/`): no CDFT — the GPU layer is ahead
  here.

## Verified numbers (H2O {O}, mio-1-1, 2026-09-15, post-fix)

- target −0.20 e: Q_O = −0.200016, λ = 0.334 Ha, constrained E
  = E0 + 0.131 Ha (penalty direction correct); `clear_cdft` restores
  E₀ to 3e-9.
- batch=8 ladder 0.00–0.35 e: all converge, q_err ≤ 1e-4.

## Real-system result: pyridone-dimer 2D PCET scan (2026-09-16)

`scripts/scan2d_pyridone_cdft.rhai` — the same 20×20 N–H…O proton grid as
`scan2d_pyridone.rhai`, run 3× (neutral, Q₁=−1 e, Q₁=+1 e):

- 385/400 and 371/400 replicas converge to ±1 e within 40 outer iters
  (~5 s per CT map, batch=400); unconverged replicas get the
  closest-achievable state via best-λ restore.
- Oxidizing M1 collapses the double well into a single minimum at
  (d1≈1.78, d2≈1.02): the proton transfers to the oxidized monomer's O —
  the expected PCET coupling.
- A few grid points showed E_CT − E_neutral < 0 → the neutral SCC there
  is a metastable basin, not the global minimum — the constrained search
  found a deeper self-consistent state. Flag, don't hide.

### Convergence machinery actually needed on a real dimer

A bare secant is not enough — the SCC has multiple self-consistent
basins, so Q(λ) is piecewise-smooth with basin-switching jumps (~1–2 e
in closed shell). What works, in order of impact:

1. **Best-λ restore at exit** — guarantees every replica reports its
   closest-achievable state; makes "lucky" sampling deterministic.
2. **Damped secant** with per-(b,f) step halving on err sign flip.
3. **Target continuation** — ramp natural→target over 8 outer iters so
   the state tracks the neutral-connected basin.
4. **Basin reset on stall** — q→q0, λ→0 after 3 stalled iters.
   Bisection inside a λ-bracket was tried and *removed*: it locks at the
   basin boundary and never escapes (194/400 vs 385/400 with secant).

## Open issues / not implemented (spec §"follow-ups")

- ~5% of the pyridone grid cannot reach a ±1 e Mulliken target (basin
  switching, closed-shell ~2 e charge steps) — replicas get the
  closest-achievable state via best-λ restore.
- Spin-polarized DFTB — required for odd-electron D⁺A⁻ states (charge
  constraint localizes charge, not spin).
- ΔSCF occupation constraints (the `occ_w`/`occ_idx` machinery already
  exists; needs state tracking across geometries).
- CDFT-CI: coupling H_LR between diabatic states (needs inter-state
  overlaps — two constrained solutions on one geometry).
- TD-DFTB/Casida — separate response layer, not an SCC state.
- PBC/k-point CDFT — orthogonal; lands with `gpu_pbc_plan.rs`.
