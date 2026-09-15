# Dense_Multi_PBC — implementation plan

GPU PBC (complex k-point) DFTB+ path. Companion docs:
`Dense_Multi_PBC.chat.md` (prototype kernel), `Dense_Multi_PBC.arch_notes.md`
(real-path architecture + what the complex path mirrors),
`Dense_Multi_PBC.ewald_notes.md` (Fortran electrostatics spec).

## Status

- [x] Complex Hermitian Jacobi (`gpu_hermitian_jacobi.cl`) — validated
- [x] Complex matrix ops (`gpu_zmatrix_ops.cl`) — validated
- [x] `GpuPbcPlan` — persistent complex SCC plan — smoke-tested end-to-end
- [x] **P1** `pbc_cell.rs` — host cell math (recVecs, volume, R/G lattice
      lists, α/cutoff autotuning, host f64 Ewald reference) — DONE
- [x] **P2** `gpu_pbc.cl` — `ewald_invr_batched`, `gamma_pbc_batched`,
      `assemble_pairs_img`, `kpoint_phase_sum_batched` — DONE
- [x] **P3** `gpu_pbc.rs` — `GpuPbc` driver: owns lattice + pair lists +
      all buffers/kernels; `set_geometry` → assemble → fold → γ → SCC — DONE
- [x] **P4** `tests/gpu_pbc.rs` — 7/7 pass:
      erfc vs NIST values; α/maxR/maxG autotuning; cubic self-term
      (−2.837297/L); NaCl Madelung (−1.747565/r0, diff <2e-5);
      GPU↔host f64 Ewald parity; big-box H₂ Γ parity (γ_pbc[i,j] =
      γ_func + C_box with C_box ≈ −2.837297/L; H0(Γ)/S(Γ) ≡ molecular);
      H(−k)=H(k)* + Hermiticity + SCC smoke + charge conservation on
      4-H chain at k=±0.21.
- [x] Fortran finite-precision parity — `tests/pbc_fortran/` +
      `tests/gpu_pbc_fortran.rs`: periodic C-O chain (real Δq transfer,
      4 explicit k-points incl. BZ edge). Measured: Mulliken 1.7e-4 e,
      ε 1.9e-5 Ha, E_band 7e-5 Ha, E_elec 1.8e-6 Ha. Regenerate via
      `run_reference.sh` (documented in that folder's README).
- [x] First physics application — 2-D proton-transfer scan on the
      QX/HQ PBC chain: `pbc_*` rhai bindings in `dftb_engine`,
      `repulsive_energy_pbc` in `methods/dftb/forces.rs`,
      `scripts/scan2d_qxhq_pbc.rhai` (20×20 batch, nk=4, 18 iters,
      ~2.9 s). Verified: d1↔d2 symmetry 9e-6 Ha; donor/transferred
      endpoints degenerate to 0.007 kcal/mol; stepwise-wins mechanism
      (intermediate +8.7, synchronous ~40 kcal/mol). Cell built by
      `scripts/make_qxhq_chain.py` (ascii-art + herringbone tilt).
      Audit `topical_audit/gpu_pbc_hbond_scans.md`, guide
      `userguide/hbond_pbc_scans.md`.

### Implementation notes (things that bit)

- **Real-space Ewald must filter by PAIR distance** |r_ij+R| < maxR
  (neighbor-list convention), NOT by |R| < maxR — the origin-centered
  ball misses images on the far side of the cell (~2.6e-4 error on the
  NaCl Madelung energy). Host ref filters per-pair over a superset
  R-list (maxR + max pair displacement); GPU kernel uses the CSR
  image-pair list which is already per-pair.
- **`GpuPbcPlan::new` runs the S^{-1/2} pipeline** — the driver must run
  one full assembly chain (fill+img+fold+ewald+γ) BEFORE constructing
  the plan, or λ_min certification sees S=0.
- **erfc**: Taylor series for x<3.5, continued fraction for x≥3.5 — the
  CF converges too slowly at small x (3e-5 error at x=0.5 after 60
  terms).
- **γ_pbc ≠ molecular γ in a big box**: every element carries the Ewald
  background shift C_box ≈ −2.837297/L; γ_pbc[i,j] − γ_pbc[i',j'] does
  converge to the molecular difference.
- **`enumerate_sk_pairs` bucket metadata**: `sk_table_idx` must be set
  on the first OUT-PAIR (`outs.is_empty()` checked before pushing), not
  on `slots.is_empty()` — the slot list is already non-empty after the
  first pair's pushes, so a heteronuclear bucket (C-O) whose first pair
  has slots was left at `usize::MAX` → index-out-of-bounds. Homonuclear
  tests passed only because a slotless diagonal pair initialized the
  bucket first. Caught by the C-O Fortran parity test.
- **HSD comments are `#`, not `!`** — `!` lines become free text and the
  parser fails with "Node already contains free text".

## Design decisions (settled)

1. **SK evaluation is k-independent.** Per geometry: evaluate each
   (atom-pair, R-image) block ONCE into a per-slot block buffer
   (`assemble_pairs_img` — same SK spline machinery as `assemble_pairs`,
   output to per-slot blocks instead of dense H). Then
   `kpoint_phase_sum_batched` folds: per unordered pair (i≤j), per k:
   `H(k)[j,i] = Σ_R e^{ik·R}·B_R`, `H(k)[i,j] = conj`. One work-item per
   (pair, k, element) — gather inputs, write owned outputs, no atomics.
   - i==j pairs: enumerate the FULL symmetric R set (both signs), acc
     starts at onsite+I (R=0 self-block), write [A,A] once.
   - i<j pairs: R enumerates images of j near i; write [j,i]=acc,
     [i,j]=acc^H.
   - S(k) same path; diagonal gets I.
   - Blocks stay pure H0/S — the SCC shift is NOT folded in here;
     `zhscc_batched` already does `H_scc = H0 + ½S(V_A+V_B)` per iteration.

2. **γ_pbc = Ewald(1/R) + short-range correction** (per
   `ewald_notes.md`): `ewald_invr_batched` computes invRMat per replica
   (real-space sum over per-pair CSR R-list + reciprocal sum over
   precomputed half-space G-list + self/const terms); `gamma_pbc_batched`
   subtracts Σ_short `expGamma` (= 1/r − γ_func, analytic in-kernel).
   `zdq_v_batched` (V = γ·Δq) unchanged.

3. **All lists host-enumerated once** at `GpuPbc::new` (with margin):
   SK image pairs, short-γ image pairs, Ewald per-pair R-lists, G-list.
   Geometry updates only re-evaluate distances — the pair/R SET is fixed
   (fail-loud check if an atom moves enough to need re-enumeration).

4. **`GpuPbcPlan` unchanged** — it already consumes s_buf/h0_buf/g_buf;
   the driver fills them.

5. Units: atomic units throughout (Bohr, Hartree) — matches both the
   Fortran internals and the existing Rust path (buf_coords_bohr).

## Test ladder (L0 → L2)

1. `pbc_cell`: α/cutoff tuning sanity, R/G list counts, half-space check.
2. `ewald_invr` vs host f64 reference (same formulas — consistency) AND
   **NaCl Madelung**: primitive FCC 2-atom cell, E_coh = −α_M/a,
   α_M = 1.747565 — a real analytic target.
3. Big-box molecule at Γ: γ_pbc ≈ γ_func molecular (→1/r + O(1/V));
   H0(Γ),S(Γ) ≡ molecular H0/S exactly (only R=0 pairs).
4. Hermiticity: H(−k)=H(k)*, ε(−k)=ε(k) on a low-symmetry cell.
5. End-to-end SCC on a real small cell (needs RUST_DFTB_SK_DIR, gated);
   band energies vs Fortran `dftb+` binary output (gated, _build exists).
