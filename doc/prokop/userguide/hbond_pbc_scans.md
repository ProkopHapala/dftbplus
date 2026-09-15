# Periodic 2-D H-bond scans (`pbc_*`)

Same recipe as [hbond_2d_scans.md](hbond_2d_scans.md) — move two donor
protons on a frozen scaffold, batched SCC, E(d1,d2) map — but the
system is a **periodic cell** solved by `GpuPbc` (complex k-points,
Ewald γ). Read the non-PBC guide first; this page only covers what
changes.

## Geometry: build the cell from ASCII art

`rust_dftb/scripts/make_qxhq_chain.py` builds a quinoxaline /
dihydroquinoxaline (QX/HQ) alternating chain via SPAMMM's
`ascii_art_heterocycle` parser:

```
   N C
  | | |
   N C
   :
 C n
| | |
 C n
```

- `:` marks the intracell N···H–N (default 2.9 Å). The second junction
  — bottom donor `n` → next cell's top acceptor `N` — comes for free
  from the periodic tiling; nothing is drawn for it.
- Lowercase `n` = donor (gets a capping H along the missing-bond
  direction, i.e. straight onto the N···N axis); uppercase `N` =
  acceptor (pyridinic, no H).
- `TILT_DEG = 25` rotates each molecule ±25° about the line through its
  two hetero vertices (herringbone). On-axis atoms (N, donor H) don't
  move; planar stacking would leave C–H···H–C contacts at ~2.0 Å —
  after the tilt they are ~2.6 Å.
- Outputs: `data/xyz/qxhq_chain_cell.xyz` (+ `.gen` for the Fortran
  reference) and `debug/qxhq_chain.png` — a 3-cell review plot with a
  side view showing the tilt. Always look at it before scanning.

## Rhai API (mirrors `gpu_*`)

| call | notes |
|------|-------|
| `pbc_new(name, SK_DIR, lat9, kpts, kw, batch)` | `lat9` = 3 lattice vectors row-major in Å; `kpts` = flat fractional [kx,ky,kz,…]; `kw` = weights summing to 1. `batch` replicas all share this cell. |
| `pbc_set_coords(name, xyz_flat)` | per-replica geometries (`batch·n_atoms·3`); rebuilds H0(k)/S(k), Ewald γ, Löwdin prep — pair/image tables stay fixed |
| `pbc_scc(name, max_iter, tol)` | one batched DIIS SCC for all replicas (α=0.3); starts from neutral charges |
| `pbc_smearing(name, kT)` | Fermi smearing in Ha (use ~0.002) |
| `pbc_eval(name)` | finalize; per-replica E = band + SCC + repulsive (E_rep summed over image cells on the host) |
| `pbc_energy_i(name, i)` | E of replica i |

## The scan script — what differs from the non-PBC one

`rust_dftb/scripts/scan2d_qxhq_pbc.rhai`:

- **Boundary-crossing junction**: J1 = N11–H27···N8 is intracell, but
  J2's acceptor is atom 0 of the **next** cell. The axis is computed
  against the image: `a2 = xyz[J2A] − (0,Ly,0) − xyz[J2D]`. The proton
  is allowed to slide past the cell edge — positions are unwrapped,
  minimum-image handles it.
- **k-points**: `nk=4` along the chain axis (0, ±¼, ½), the same set as
  the Fortran parity test. The transverse directions are vacuum.
- Everything else — grid loop, replica packing, `ENERGY MAP` printing —
  is identical to `scan2d_dzp.rhai`.

```bash
cargo run --release --bin dftb_engine -- --script scripts/scan2d_qxhq_pbc.rhai
python3 scripts/plot_scan2d_pbc.py <logfile>     # -> debug/qxhq_pbc_Emap.png
```

## What the map shows (and why PBC is the point)

- **Exact mirror symmetry** |E(d1,d2)−E(d2,d1)| ≈ 0.006 kcal/mol — the
  junctions are related by translation, not by an imposed constraint.
- **Degenerate endpoints** (~0.007 kcal/mol): moving *both* protons
  converts every Q→DHQ and every DHQ→Q — the product is the same
  infinite chain, shifted. A finite dimer can never show this.
- **Stepwise vs synchronous**: the single-transfer intermediate
  (charge-separated pair) is only ~9 kcal/mol up, while the diagonal
  (both protons mid-flight) barrier is ~40 kcal/mol — on a proton wire
  the defects move independently.

## Caveats

- The cell (and therefore Ewald/SK image tables) is fixed at `pbc_new`;
  scanning the N···N distance itself needs one engine rebuild per L.
- `pbc_scc` always cold-starts from neutral charges — no warm-start
  between calls yet.
- `pbc_eval` adds E_rep on the host (34-atom cell → trivial); there is
  no PBC repulsive/force kernel yet, so no `pbc_relax`.

Details, measurements, and parity status:
[`../topical_audit/gpu_pbc_hbond_scans.md`](../topical_audit/gpu_pbc_hbond_scans.md).
