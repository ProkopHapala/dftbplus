---
type: TopicalAudit
title: Periodic H-bond / proton-transfer scans (GpuPbc)
tags: [topic, gpu, pbc, hbond, proton-transfer, scan, opencl, ewald]
---

# Periodic H-bond / proton-transfer scans (GpuPbc)

## Summary

2-D proton-transfer scans on a **periodic** hydrogen-bond wire, run the
same way as the molecular scans (`doc/prokop/userguide/hbond_2d_scans.md`)
but through the complex-k-point `GpuPbc` engine instead of `GpuDftb`.
`n_rep` replicas of a periodic cell share one Ewald/SK image table; one
batched SCC solves all 400 geometries of the (d1, d2) grid. The first
application is the quinoxaline / dihydroquinoxaline (QX/HQ) alternating
chain — two N–H···N junctions per cell, one of which **crosses the cell
boundary** (donor in cell i, acceptor in cell i+1).

## Geometry: ASCII-art chain builder

`rust_dftb/scripts/make_qxhq_chain.py` builds the cell via SPAMMM's
`ascii_art_heterocycle` (dimer format):

```
   N C
  | | |
   N C
   :
 C n
| | |
 C n
```

- Each block is a 2-ring aza-naphthalene; the hetero atoms sit at the
  top/bottom **vertices** of the pyrazine ring → para-diazine edge, so
  the N···H–N bonds run along the chain axis (y).
- `:` marks the intracell N···H–N at `hbond_length=2.9 Å`; the second
  junction (bottom donor → next cell's top acceptor) is implicit in the
  periodic tiling — the geometry only, no bond bookkeeping.
- Lowercase `n` gets a capping H placed along the missing-bond direction
  (vertex outward → straight onto the acceptor axis, r(N–H)=1.01 Å).
- **Herringbone tilt** (`TILT_DEG=25`): each molecule is rotated about
  the vertical line through its two hetero vertices by ±25° (opposite
  signs per molecule). On-axis atoms (N, donor H) are invariant; the
  interface C–H···H–C contacts open from 2.00 Å → 2.59 Å. Without the
  tilt the planar stack has tight H···H contacts.

Outputs `data/xyz/qxhq_chain_cell.xyz` (+ `.gen` for Fortran dftb+),
`debug/qxhq_chain.png` (3-cell review plot, top + side view).
Cell: 34 atoms (16+18), L_y = 11.480 Å, transverse vacuum 16.4 × 15 Å.

## Engine bindings (rhai)

`dftb_engine` exposes `GpuPbc` in parallel to `GpuDftb`
(`src/bin/dftb_engine.rs`):

| Function | Purpose |
|----------|---------|
| `pbc_new(name, sk_dir, lat9, kpts_flat, kw, batch)` | build engine: lattice (row-major Å), fractional k-points + weights (Σ=1), `batch` replicas share the cell |
| `pbc_set_coords(name, xyz_flat)` | per-replica geometries → assemble H0(k)/S(k) blocks, Bloch fold, Ewald γ, Löwdin prep |
| `pbc_scc(name, max_iter, tol)` | batched DIIS SCC (α=0.3); resets to neutral charges each call — no cross-call warm start yet |
| `pbc_smearing(name, kT)` | Fermi smearing (Ha) |
| `pbc_eval(name)` | finalize + energy per replica = band + SCC + **E_rep** |
| `pbc_energy_i(name, i)` | per-replica energy readback |

`compute_energy` is electronic only; the repulsive term is added
host-side by `repulsive_energy_pbc` (`methods/dftb/forces.rs` — spline
sum over ±1 image cells, each pair counted once: R=0 → i<j, R≠0 →
lex-positive half-space; fail-loud on sub-`MIN_NEIGH_DIST` contacts and
missing splines).

## Scan script

`rust_dftb/scripts/scan2d_qxhq_pbc.rhai` mirrors `scan2d_dzp.rhai`:
junctions `J1 = N11–H27···N8` (intracell, axis +y) and
`J2 = N19–H33···N0` (cross-boundary, axis −y to the N0 image at y−L).
d1,d2 ∈ [1.0, 1.9] Å, r(N···N)=2.9 Å, 20×20 grid, nk=4 along y
(0, ±¼, ½ — the Fortran reference k-set). Plot:
`scripts/plot_scan2d_pbc.py <log>` → `debug/qxhq_pbc_Emap.png`.

## Measured results (2026-10, RTX 3090)

- N=94 orbitals/cell, batch=400, nk=4: cold SCC 18 iters → rms 1e-5,
  ~2.9 s; all replicas converged, 0 uncertified.
- **Junction symmetry**: max |E(d1,d2) − E(d2,d1)| = 9e-6 Ha
  (0.006 kcal/mol) — translation-equivalent junctions reproduce to ~cal.
- **Degenerate endpoints**: E(1.0,1.0) = −42.20557 vs E(1.9,1.9) =
  −42.20556 Ha → 0.007 kcal/mol. Concerted double transfer swaps every
  Q↔DHQ, so the transferred "tautomer" is the same chain shifted —
  exact degeneracy that a finite dimer cannot show.
- **Stepwise wins**: single-transfer intermediate at (1.9,1.0) is
  +8.7 kcal/mol; synchronous (diagonal) barrier ≈ 40 kcal/mol.

## Parity status

| Check | Reference | Tolerance / measured |
|-------|-----------|----------------------|
| Engine parity (C–O chain) | Fortran `dftb+`, `tests/pbc_fortran/` | Mulliken 1.7e-4 e, ε 1.9e-5 Ha, E_elec 1.8e-6 Ha — `tests/gpu_pbc_fortran.rs` |
| γ_pbc / Ewald | host f64 Ewald, NaCl Madelung −1.747565/r0 | diff < 2e-5 — `tests/gpu_pbc.rs` |
| Map symmetries | translation invariance of the chain | 9e-6 Ha (d1↔d2) |

## Open issues

- `GpuPbc::scc` always restarts from neutral q0 — a `scc_warm` flag
  would enable sequential-scan warm starts (not needed for the batched
  grid).
- Cell is fixed at `pbc_new` (image tables / Ewald cutoffs depend on
  it) → a lattice-parameter (N···N compression) scan needs one engine
  rebuild per L — cheap for 34 atoms, but not yet wrapped.
- E_rep is host-side per replica (34 atoms trivial); a PBC repulsive
  GPU kernel would matter only for large cells / forces.
- Fortran parity reference for THIS cell not yet generated (`.gen`
  written for that purpose).
