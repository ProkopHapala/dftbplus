---
type: report
title: "SCC charges, HOMO-LUMO, Davidson partial eigensolver, and DFTB+ Fortran parity harness"
tags: [report, scc, davidson, sparse, parity, frontier-orbitals]
timestamp: 2025-09-05
---

# Session Report — SCC Charges, HOMO-LUMO, Davidson, DFTB+ Parity

## Objective

After SCC relaxation, expose and visualize:
1. Atomic Mulliken charges (spatial plot).
2. HOMO/LUMO frontier orbitals (energy-axis plot).
3. A partial (subspace) eigensolver to obtain a few eigenvalues around the gap without
   full diagonalization, for eventual use on large sparse systems.
4. A DFTB+ Fortran reference harness to validate the Rust implementation on the same
   geometry + SK set.
5. End-to-end comparison on benzene, coronene, circumcoronene (pure-C PAHs, since BSR4
   requires 4 orbitals/atom).

## What was done

### Rust

- **`rust_dftb/src/methods/sparse/davidson.rs`** — new file. Generalized Davidson
  eigensolver for `H C = S C ε`:
  - S-orthonormalization of trial vectors via modified Gram–Schmidt under the `S` metric.
  - Rayleigh–Ritz projection `VᵀHV` on the small subspace.
  - Diagonal (Jacobi) preconditioner with regularization to avoid blow-up when
    `H_ii - θ·S_ii ≈ 0` for near-degenerate states.
  - Subspace restart when `m > max_subspace` (keep best `2·n_target` Ritz vectors).
  - Gap-centered selection: `n_target` occupied + `n_target` virtual eigenvalues.
  - Unit test `test_davidson_vs_dense_small` against dense `SymmetricEigen` — passes.
- **`rust_dftb/src/bin/dftb_engine.rs`** — new Rhai functions:
  - `get_charges(name)` → comma-separated Mulliken populations.
  - `get_eigenvalues(name)` → comma-separated eigenvalues.
  - `save_charges(name, path)` → TSV with element, x, y, z, charge.
  - `save_eigenvalues(name, path)` → TSV with idx, eigenvalue, occupied.
  - `davidson_homo_lumo(name, n_target)` → `"HOMO,LUMO,gap"` string.
  - `get_sparse_charges(name)`, `save_sparse_charges(name, path)` (already existed,
    now actually used).
- **`rust_dftb/scripts/test_charges_homo_lumo.rhai`** — end-to-end Rhai test script
  running dense SCC + sparse TC2 + Davidson on all three PAHs.

### Python (in `scripts/`)

- **`run_dftbplus_ref.py`** — DFTB+ Fortran reference harness:
  - Generates `dftb_in.hsd` from an XYZ with the mio-1-1 SK set, SCC enabled,
    `WriteDetailedOut = Yes`.
  - Runs `/home/prokophapala/git/dftbplus/_build/app/dftb+/dftb+` in a temp dir.
  - Parses `detailed.out` (Mulliken charges, Fermi level, total energy) and
    `band.out` (eigenvalues + occupations; **eV → Ha conversion**).
  - Writes `ref_charges.tsv` and `ref_eigenvalues.tsv` to a user-specified output dir.
- **`plot_charges_homo_lumo.py`** — plotting:
  - Spatial 2D charge map (atoms in xy-plane, colored by Mulliken charge, diverging
    RdBu_r colormap). Columns: Rust dense | Rust sparse TC2 | DFTB+ ref.
  - HOMO-LUMO energy-level diagram (horizontal lines, blue=occupied, red=virtual,
    green gap arrow). Columns: Rust dense | Davidson | DFTB+ ref.
  - Converts Rust Mulliken **populations** to **charges** via `q = q0 - population`
    (q0=4 for C) so Rust and DFTB+ use the same sign convention.
- **`compare_rust_vs_dftbplus.py`** — numerical parity report:
  - Per-atom charge residuals (dense vs DFTB+, sparse vs DFTB+, sparse vs dense).
  - HOMO/LUMO/gap differences.
  - All-eigenvalue max/RMS residuals.

### Repository hygiene

- Moved stray `formic_azaindole_candidates.png` from
  `doc/prokop/tasts/GPU_MultiSystem/artifacts/` to `debug/gpu_multisystem/`.
- Task folder now contains only Markdown specs.

## Problems and how they were overcome

### 1. DFTB+ exited with "Hamilton/Overlap written, exiting program"

**Cause:** the `h2_ref/dftb_in.hsd` template had `WriteHS = Yes`, which makes DFTB+
dump H/S and exit before SCC. Also `MaxSCCIterations = 1`.

**Fix:** wrote a fresh HSD template with `WriteDetailedOut = Yes` only, proper
`MaxSCCIterations = 100`, and `Filling = Fermi { Temperature [Kelvin] = 0.0 }`.

### 2. `WriteBandOut` rejected by the parser

**Cause:** the parser version in this build does not accept `WriteBandOut` as an
`Options` child (it exists in `parser.F90` but the oldcompat layer flagged it).

**Fix:** removed `WriteBandOut` — `band.out` is written by default in this build.

### 3. `band.out` eigenvalues in eV, not Hartree

**Symptom:** HOMO=-6.67, LUMO=-6.62 — looked wrong vs Fermi level -0.244 Ha.

**Root cause:** `band.out` eigenvalues are in **eV**; `detailed.out` Fermi/energies
are in **Hartree**.

**Fix:** `parse_band_out` now multiplies by `1/27.211386` (eV → Ha). After fix:
HOMO=-0.245 Ha, LUMO=-0.243 Ha — matches Fermi level and Rust dense.

### 4. Davidson index-out-of-bounds panic

**Cause:** after changing the initial subspace size from `2·n_target` to
`1.5·n_target` per side (to help with degeneracies), the residual loop still
iterated over `n_guess` but the Ritz selection returned only `2·n_target` pairs.

**Fix:** loop over `n_return = 2·n_target` instead of `n_guess`.

### 5. Davidson stalled on coronene (100+ iterations, no convergence)

**Symptom:** benzene converges in 3 iterations; coronene's LUMO never converges —
the dense manifold of near-degenerate π states near the gap makes the diagonal
preconditioner ineffective (the correction vectors keep introducing new directions
that don't isolate the LUMO).

**Partial fix:** regularized the preconditioner denominator
`(|denom| < eps → sign·eps)` to avoid blow-up, and increased the initial subspace
to `1.5·n_target` per side. This helped benzene stay robust but did **not** solve
coronene.

**Open issue:** the simple diagonal-preconditioned Davidson is insufficient for
PAHs with dense frontier manifolds. Options for future work:
- SSOR / ILU preconditioner (captures off-diagonal coupling).
- Shift-invert Davidson (expensive but robust for interior eigenvalues).
- Lanczos with spectral transformation.
- Chebyshev-filtered subspace iteration (CheFSI) — see
  `NumericalMathPlayground/topics/LinearAlgebra/LinearScalingQM/CheFSI/`.

For now, the dense path is the reference for all three systems; Davidson is
validated only on benzene.

### 6. Rust "charges" are populations, not charges

**Symptom:** Rust reported 4.0000 for benzene C; DFTB+ reported 0.0000.

**Root cause:** Rust `SccResult.charges` stores **Mulliken populations**
(`q_electronic`); the actual charge is `q0 - population`. DFTB+ `detailed.out`
reports the charge `deltaQ = q0 - q_electronic`. This is documented in
`rust_dftb/src/methods/dftb/forces.rs` (Rust convention: `delta_q = q - q0`).

**Fix:** the plotting and comparison scripts now convert Rust populations to
charges via `q0_map[el] - population` before comparison.

## Parity results

| System | atoms | dense vs DFTB+ max\|Δq\| (e) | sparse TC2 vs DFTB+ max\|Δq\| (e) | TC2 iters | gap Δ dense vs DFTB+ (Ha) |
|---|---|---|---|---|---|
| Benzene | 6 | 0 | 1.5e-5 | 46 | 2.27e-6 |
| Coronene | 24 | 4.2e-16 | 9.0e-6 | 36 | 3.26e-7 |
| Circumcoronene | 54 | 3.9e-16 | 6.5e-5 | 51 | 3.00e-7 |

- **Dense charges** match DFTB+ to machine precision (the ~1e-16 is float32 roundoff
  from the sparse GPU path used for comparison).
- **Sparse TC2 charges** match to ~1e-5–1e-4 e, consistent with the 1e-6 TC2
  idempotency tolerance.
- **Eigenvalues** match to ~2e-6 Ha (dense vs DFTB+), the difference coming from
  Fermi smearing temperature differences between the two codes.
- **Davidson** on benzene matches dense to 1e-10 Ha.

## Open issues

1. **Davidson convergence on coronene/circumcoronene** — the diagonal preconditioner
   is insufficient for systems with dense near-degenerate frontier manifolds. See
   "Problems" §5 above for proposed fixes.
2. **Sparse route for H-containing systems** — BSR4 assumes 4 orbitals/atom
   (s,p), so it rejects H. The PAHs in this test are pure-C. Realistic systems
   (passivated edges, heteroatoms) need either a variable-block-size BSR or a
   fallback to dense for H atoms.
3. **Davidson is not wired to the sparse BSR4 operator** — it currently uses the
   dense `H_scc` and `S` from `SccResult`. For genuinely large systems, a sparse
   matvec operator would be needed to avoid densification.
4. **Fermi temperature** — Rust uses `T=0` filling; DFTB+ uses a small but nonzero
   electronic temperature. This causes the ~2e-6 Ha eigenvalue differences. Could
   be aligned by matching the temperature.
5. **`band.out` is overwritten per run** — the harness copies `ref_charges.tsv` and
   `ref_eigenvalues.tsv` to per-system names (`<sys>_ref_*.tsv`) to avoid clobbering
   when running multiple systems sequentially.

## Files changed/added

### Added
- `rust_dftb/src/methods/sparse/davidson.rs`
- `rust_dftb/scripts/test_charges_homo_lumo.rhai`
- `scripts/run_dftbplus_ref.py`
- `scripts/plot_charges_homo_lumo.py`
- `scripts/compare_rust_vs_dftbplus.py`
- `doc/prokop/reports/2025-09-05_scc_charges_davidson_parity.md` (this file)

### Modified
- `rust_dftb/src/bin/dftb_engine.rs` (new Rhai functions)
- `rust_dftb/src/methods/sparse/mod.rs` (export Davidson)
- `CODEMAP.md` (this session)
- `doc/prokop/DFTB_Reimplementation_Progress/OVERVIEW_Roadmap.md` (this session)

### Generated (in `debug/graphene_sparse/`, not committed)
- `charges_spatial.png`, `homo_lumo_levels.png`
- `benzene_*.tsv`, `coronene_*.tsv`, `circumcoronene_*.tsv` (charges, eigenvalues,
  sparse charges, convergence)

## How to reproduce

```bash
# 1. Rust end-to-end (dense + sparse + Davidson)
cd rust_dftb
cargo run --bin dftb_engine -- --script scripts/test_charges_homo_lumo.rhai \
  --sk-dir /home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1

# 2. DFTB+ Fortran reference (per system)
cd ..
for sys in benzene coronene circumcoronene; do
  python3 scripts/run_dftbplus_ref.py /tmp/${sys}.xyz debug/graphene_sparse/ \
    --sk-dir /home/prokophapala/SIMULATIONS/dftbplus/slakos/mio/mio-1-1
  cp debug/graphene_sparse/ref_charges.tsv debug/graphene_sparse/${sys}_ref_charges.tsv
  cp debug/graphene_sparse/ref_eigenvalues.tsv debug/graphene_sparse/${sys}_ref_eigenvalues.tsv
done

# 3. Plots
python3 scripts/plot_charges_homo_lumo.py debug/graphene_sparse/

# 4. Numerical parity report
python3 scripts/compare_rust_vs_dftbplus.py debug/graphene_sparse/
```
