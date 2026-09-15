# Dense_Multi_CDFT — task spec

Constrained-DFT (CDFT) augmentation of the dense batched `GpuDftb`
multi-solver, for charge-localized diabatic states / PCET surfaces.
Runs **in parallel** with the PBC work (`gpu_pbc_plan.rs`) — this spec
touches only `gpu_scc_plan.rs` (one hook line), `gpu_dftb.rs` (additive
methods), `mod.rs`, `dftb_engine.rs` (new Rhai fns), plus a NEW module
file + NEW kernel file. No changes to any existing kernel or solve path
semantics.

## Physics

Constrained SCC-DFTB: for fragment F with Mulliken excess charge
Q_F = Σ_{A∈F} Δq_A, minimize E_DFTB + λ_F·(Q_F − Q_F^target).
Because the SCC Hamiltonian is H_scc[μν] = H0 + ½·S[μν]·(V_A + V_B),
the constraint enters as a pure on-site shift V_A → V_A + λ_F(A).
Equivalently H_scc += ½·λ_F·S·(w_μ + w_ν) applied AFTER the fused
dq→V→H_scc build, before the eigensolve.

Consequences that make this cheap and clean:

- No new eigensolver, mixer, or density machinery.
- Reported band energy Σf·ε automatically contains +λ·Q_F, so
  E_DFTB(constrained) = E_eval − Σ_F λ_F·Q_F. Expose both.
- Existing force path uses h_scc-derived eigenvalues/W → the
  −λ·dQ_F/dR constraint force rides along automatically through
  dS/dR. Forces on the constrained surface are correct at fixed λ;
  the extra (Q−Q_t)·dλ/dR term vanishes at convergence.
- Per-replica targets → one batch = many diabatic states in one launch.

## Modules (implementation order)

- [x] **M0 spec** — this document.
- [x] **M1 kernel** `src/qmqm/gpu_cdft.cl` —
      `cdft_hscc_shift_batched`: one WG/system, caches λw[A] in local,
      then `H[idx] += 0.5·S[idx]·(λw[oa[i]]+λw[oa[j]])` strided over
      N². Gates on `active[]`. Gather-in / own-write; no atomics.
- [x] **M2 state** `src/qmqm/gpu_cdft.rs` — `GpuCdft`: frag map
      [n_atoms], λ buffer [batch·nfrag], targets, per-(b,f) secant
      history. Kernel built once at attach; λ upload between solves.
- [x] **M3 hook** `gpu_scc_plan.rs` — `pub cdft: Option<GpuCdft>`
      field (None → zero cost); `enq_dq_v_hscc` enqueues the shift
      right after the fused build so EVERY h_scc rebuild (SCC iters,
      finalize, eval) carries the constraint. ~4 lines.
- [x] **M4 driver** `gpu_dftb.rs` — `set_cdft(frag, targets)`,
      `clear_cdft()`, `cdft_scc(max_outer, scc_iter, rms_tol, q_tol)`
      outer-λ loop (converge SCC at fixed λ → read dq → per-(b,f)
      err → damped secant update → upload λ; plus target-continuation
      ramp, stall basin-reset, and best-λ restore — see audit),
      `cdft_qfrag()`, `cdft_energies()` (E_eval − Σλ·(Q_F+Q0_F) —
      shift couples to GROSS population), `cdft_set_lam` (manual λ for
      Q(λ) response scans).
- [x] **M5 bindings** `dftb_engine.rs` — `gpu_cdft(name, frag,
      targets)`, `gpu_cdft_clear(name)`, `gpu_cdft_scc(name,
      max_outer, scc_iter, rms_tol, q_tol)`, `gpu_cdft_qf(name,b,f)`,
      `gpu_cdft_lam(name,b,f)`, `gpu_cdft_set_lam(name,b,f,lam)`,
      `gpu_cdft_energies(name)`.
- [x] **M6 test** `tests/gpu_cdft.rs` — H2O monomer: constrain
      +0.3 e on O, check Q_F within tol, E ≥ unconstrained, λ=0 path
      identical to plain scc.

## Explicitly NOT in this module (follow-ups, see chat doc)

- Spin-polarized DFTB (needed for odd-electron / D+A− states).
- ΔSCF occupation constraints (occ_w machinery already exists).
- CDFT-CI coupling between diabatic states (needs state overlaps).
- TD-DFTB/Casida (separate response layer).
- PBC k-point version — orthogonal, lands with gpu_pbc_plan.
