# Bloch slices and near-EF density (k-resolved)

Real-space pictures of DFTB wavefunctions for a **periodic** cell: one Bloch state, or a Tersoff–Hamann sum of |\ψ|² over states near the Fermi level. The coefficients come from `DFTBcore` (`C(k)`, complex). The grid evaluation is one OpenCL work-item per point.

This is not the older Γ-only projector. That path lives in `pyBall/OCL/cl/Grid.cl` and `tests/grid/test_waveplot_dftbcore.py`, packs orbitals as Fireball `[px, py, pz, s]`, and has no k-phase. Do not mix the two.

Matrix export itself is [dftbcore_python.md](dftbcore_python.md). This page is only the step from `C(k)` to a picture.

## What is plotted

One state `s` = one pair (molecular orbital, k-point):

```
ψ_s(r) = Σ_{μ,n} C_{sμ} φ_μ(r − τ_μ − R_n) exp(−i 2π k_s · n)
```

`n` is an integer cell index, `k_s` is fractional (the same numbers `get_kpoints()` returns), `φ` is the STO from the official `wfc.*.hsd` for that Slater–Koster set. The phase sign is the one waveplot actually plots: Fortran `dot_product` conjugates a stored `exp(+ikr)` down to `exp(−ikr)`. The kernel writes that factor explicitly. Do not pre-multiply `C`.

A density map is **not** a sum of the complex waves and **not** a density-matrix contraction:

```
ρ(r) = Σ_s w_s |ψ_s(r)|²
```

Because each term is already |\ψ|², the `e^{ikR}` color is gone and `ρ` is periodic with the primitive cell. Nodal planes of one Bloch state do not survive the sum. To see the phase, plot one state.

For a spin-unpolarized energy window the scripts use `w_s = 2 w_k` (both spins, k-point weight). The window selects the states. They are not multiplied again by the Fermi occupation.

## Files

| Piece | Path | Role |
|---|---|---|
| Kernel | `pyBall/OCL/cl/DFTBplusGrid.cl` `project_bloch_points` | One point per work-item. Home-cell atoms, image list, phase inside the kernel. |
| Host call | `pyBall/OCL/DFTBplusGridProjector.py` `project_bloch_points` | Packs `C` as float2, uploads, returns `ρ` and optionally `ψ`. |
| Basis | `pyBall/OCL/DFTBplusParser.py` `parse_basis_hsd_ang` | `wfc.*.hsd` is in Bohr; this converts exponents and cutoffs to Å. |
| Coefficients | `pyBall/DFTBcore.py` `get_eigvecs_cplx` | `C[iks, mo, orb]`, `E` in Hartree. Columns of the Fortran matrix are MOs. |
| Plots | `pyBall/plotUtils.py` | `plot_2d_array`, `plot_complex_hsv`, `plot_bands_and_maps`, `plot_ldos_row`. |
| Graphene driver | `tests/grid/test_graphene_bloch2d.py` | 2-atom sheet, 3ob-3-1. CPU/GPU check, then the pictures. |
| Ribbon driver | `tests/grid/test_ribbon_bloch.py` | Relaxed vacuum ribbons from the SPAMMM enum, mio-1-1. |
| Basis tables | `tests/grid/dftb_ptcda/wfc.3ob-3-1.hsd`, `wfc.mio-1-1.hsd` | Official multi-zeta STO files. The SK set of the Hamiltonian and this file must be the same one. |

Scratch (gitignored `tests/grid/work/`): `graphene_bloch/`, `ribbon_bloch/<tag>/`. DFTB+ writes `detailed.out` there because both drivers `chdir` into the scratch dir before `init`. Pictures go to `debug/graphene_bloch/` and `debug/ribbon_bloch/`.

`*.png` is gitignored globally. These two folders are excepted at the bottom of `.gitignore`, otherwise the editor will not open the plots.

## Run the graphene sheet

Needs `libdftbcore.so` (see [dftbcore_python.md](dftbcore_python.md)) and a 3ob-3-1 Slater–Koster tree. The script looks at `$DFTB_SK_PATH`, then `~/SIMULATIONS/dftbplus/slakos/`, and appends `3ob-3-1/`.

```
cd tests/grid
python test_graphene_bloch2d.py
```

About two seconds. It runs a 6×6 Γ-centered mesh for the coefficients, checks a few points against an independent float32 spline sum, then a 24-point Γ–M–K–Γ path for the bands.

| File | What it is |
|---|---|
| `debug/graphene_bloch/gamma_pistar_phase.png` | Γ π* (the empty π, +6.39 eV). Hue is the phase. The two sublattices differ by π, so the node between them is visible. Real, no cell-to-cell drift. |
| `debug/graphene_bloch/gamma_pi_abs2.png` | Occupied Γ π, |\ψ|². Same sign on both sublattices, so no in-plane node. |
| `debug/graphene_bloch/K_phase_large.png` | One state at K = (2/3, 1/3). Hue advances 120° per lattice step and repeats every three cells. |
| `debug/graphene_bloch/K_abs2_large.png` | Same state, |\ψ|². Identical in every cell. |
| `debug/graphene_bloch/K_phase.png` | The same K state on a patch too small to see the three-cell repeat. Prefer the `_large` file. |
| `debug/graphene_bloch/fermi_ldos_large.png` | Σ 2 w_k |\ψ|² for the four states in EF±1.2 eV on that mesh (the Dirac pair at K and at K'). Periodic. No hue winding. |
| `debug/graphene_bloch/bands_windows.png` | Bands Γ–M–K–Γ plus one |\ψ|² map per energy window. Each map has its own color scale; the number in the title is that map's maximum. |

The slice is `z = 1` Å. A `p_z` lobe is zero in the nuclear plane, so `z = 0` is a blank plot even when the state is pure π.

## Run the ribbons

Inputs are the already relaxed vacuum ribbons, not rebuilt here:

```
/home/prokop/git/SPAMMM/debug/ribbon_mio/enumv_{C,N,O}_r{1..5}/{state}/
    geom.out.gen     relaxed geometry (gen, Cartesian Å, origin at 0)
    dftb_in.hsd      Slater–Koster Prefix and the SupercellFolding mesh are copied from this file
```

`{state}` is the protonation label from `spammm/topology/ribbon_pbc.py` `junction_state_strings`: `0H`, `1H-p`, `1H-d`, `2H-ss-adj`, `2H-ss-sep`, `2H-os-adj`, `2H-os-sep`. Generation of those cells is `tests/topology/testplot_ribbon.py` in the SPAMMM repo (`--vac --chem {C,N,O} --widths 1,2,3,4,5 --ncells 4 --enum --sk mio-1-1`). This driver does not relax anything. It reads `geom.out.gen` and does one single-point SCC.

```
cd tests/grid
python test_ribbon_bloch.py
```

The default set is every protonation state of r1, plus the bare edge `0H` at r2–r5, for C, N and O. On this machine that is a few tens of seconds. Edit `CASES`, `STATES`, `WIDTHS`, `HALF_WIN`, `Z_ABOVE`, `DX` at the top of the script to change it. `RIBBON_ROOT` is the absolute path above.

What it writes:

| File | What it is |
|---|---|
| `debug/ribbon_bloch/enumv_{C,N,O}_r1_ldos.png` | One panel per protonation state. Σ 2 w_k |\ψ|², states with \|E−EF\| < 0.5 eV, one cell. Each panel is scaled to its own maximum. |
| `debug/ribbon_bloch/enumv_{C,N,O}_0H_widths.png` | Bare edge, widths r1–r5, same window. |
| `debug/ribbon_bloch/enumv_{chem}_r{1,5}_0H_phase.png` | The single state closest to EF, three cells along the ribbon, hue = arg(ψ). |

The k-mesh in those inputs is an 8×1×1 Monkhorst–Pack grid with a half-step shift. DFTB+ folds k with −k, so `get_kpoints()` returns 4 points and the weights are 1/4. That is the mesh the relaxation used, not a coarser one.

If nothing falls inside ±0.5 eV (a real gap), the script takes the highest occupied and lowest empty orbital **at every k** and says so on stdout (`frontier pair at each k`). The panel title is the actual energy span of whatever was summed. A span of several eV means that panel is the frontier pair, not a narrow STM window. `enumv_C_r1` `2H-os-adj` is that case.

`1H-p` and `1H-d` are mirrors. Their total energies match (C −30.944143 Ha, N −33.075399 Ha, O −56.864447 Ha on this run). If a future edit breaks that, the Hamiltonian changed, not the plot scale.

The ribbon script refuses a cell whose `a1` is not along x or whose `a2` has an x component. The self-junction stacks (`enumsj_*`, periodic in y, often tilted) are outside this driver.

## Systematic STM-window maps + unfolded bands

`tests/grid/test_ribbon_stm_sys.py` runs the full enum — chem {C,N,O} × r1–r5 × all 7 states — on a denser mesh (`NKX=48` supercell, half-shifted; 24 irreducible k), and for each (chem,width) writes three figures to `debug/ribbon_stm_sys/`:

- `{tag}_occ_ldos.png` / `{tag}_unocc_ldos.png` — Σ 2w_k|ψ|² over EF−0.5..EF and EF..EF+0.5 eV (≈ STM at ∓0.5 V), 7 states in a row, green `+` marks switched sites. Empty window → frontier state per k (panel title shows the real span).
- `{tag}_bands.png` — per state: unfolded spectral weight on the primitive BZ (red, size ∝ W, mirrored to the full BZ by time reversal), black lines = pristine x1 bands (ideal geometry, dense mesh), grey = relaxed-0H unfold (W>0.5). Row 0H shows red≡grey as a consistency check. Shaded bands mark the STM windows.

The primitive reference uses the ideal `build_zigzag_ribbon` at ncells=1: cutting a subcell out of the relaxed supercell works for C but the relaxed N primitive hits a `dgesv` singularity in SCC at T<3000 K (flat edge band at EF → singular mixer Jacobian). Ideal-vs-relaxed band offset (~0.3 eV on edge bands) is real and visible in the plots.

Per-case caches (`work/ribbon_stm_sys/*.npz`) make `--plot-only` replotting free; `--chem/--widths/--states/--nkx/--win/--dx` select subsets. Unfold uses SPAMMM `subcell_group_indices`+`unfold_spectral_weights` on the ideal builder geometry (`strict=False` so the defect-only orbitals — extra H's — are excluded from primitive projection).

## Reading a hue plot

`plot_complex_hsv`: hue = `(arg ψ + π) / 2π`, brightness = |ψ| / max|ψ|. A node is black because the argument is undefined there, not because the wavefunction jumped. A π phase difference is the opposite hue (red against cyan).

On a large enough patch the hue of one state must repeat after a whole number of cells set by `k`. At graphene K, `k · a1 = 2π · 2/3`, so the repeat is three cells. A patch of one or two cells looks like a different color in every cell and a seam on the boundary. That seam is the node of `u_k` or the truncated repeat, not a discontinuity: the step of ψ across the primitive-cell edge is smaller than the step over the same distance inside the cell, and shifting by `m` lattice vectors reproduces `exp(−i 2π k_x m) ψ` to ~1e-7.

Atom dots are element-colored (`scatter_atoms`): C black, H white, N blue, O red, white edge so they stay visible on both the dark and the bright parts of the map.

## Calling it yourself

```python
import os, numpy as np
from pyBall.DFTBcore import DFTBcore
from pyBall.OCL.DFTBplusGridProjector import DFTBplusGridProjector
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang

os.chdir(scratch)                  # DFTB+ writes detailed.out, charges.bin, … here
dftb = DFTBcore()
dftb.init('dftb_in.hsd')          # init turns eigenvector storage on
energy = dftb.run_scf()
C, E = dftb.get_eigvecs_cplx()    # C[ik, mo, orb] complex128, E Hartree
kpts, wk = dftb.get_kpoints()     # fractional
ef_ev = ...                       # "Fermi level:" line in detailed.out, in eV
dftb.finalize()

species = parse_basis_hsd_ang('basis.hsd')   # Basis { Resolution = 0.1  <<+ "wfc.….hsd" }
basis = {'species': species}
name_to_i = {sp['name']: i for i, sp in enumerate(species)}
ispec = np.array([name_to_i[s] for s in symbols], np.int32)   # index into the wfc list

proj = DFTBplusGridProjector(verbosity=0)
proj.load_basis_dftb(basis)
atoms = proj.prepare_atoms_dftb(pos_A, ispec, basis)          # home cell only

# images: integer n, and the same shift in Å. Cover the STO cutoff (~3–4 Å).
cell_cart, cell_n = ...
rho, psi = proj.project_bloch_points(
    points_A, atoms, cell_cart, cell_n,
    coeffs,          # (nstate, norb) complex, one row = one (mo, k)
    k_frac,          # (nstate, 3) fractional
    weights,         # (nstate,)  e.g. 2*wk for an unpolarized window
    write_psi=True)  # psi is (nstate, npoints) complex64; rho is (npoints,) float32
```

`basis.hsd` is only the include plus `Resolution` (Bohr). `parse_basis_hsd_ang` follows `<<+`.

Build the plane with `meshgrid(..., indexing='xy')` and reshape `ρ` to `(ny, nx)`. `imshow(..., origin='lower')` then wants that array **without** a transpose: row 0 is the bottom of the extent. `plot_2d_array` and `plot_ldos_row` follow that convention. `plot_bands_and_maps` draws the bands and one map per window, each map scaled to its own max so a narrow window is not washed out by a wide one.

`norb` of `coeffs` must equal the orbital count implied by `atoms` (`i0orb[-1] + norb[-1]`). The kernel raises if any atom has more than 4 orbitals.

## Conventions that silently make a wrong picture

- **Project `C`, not `C*`.** `H(k) C = E S(k) C` holds for the unconjugated columns (~1e-15). At a general k, `C*` has a residual of order 1. Time reversal is `C(−k) = (per-MO phase) × C(k)*`, which is why |\ψ(−k)|² = |\ψ(k)|² even though the coefficient matrices are not conjugates of each other. `tests/dftb/test_dftbcore_kpoints.py` uses `.conj().T`; that is only valid on a mesh whose k-points are all Γ or a zone-boundary TRIM point. Do not copy it onto a general k.
- **Orbital order is DFTB+: `s`, `py`, `pz`, `px`.** Hydrogen is `s` only. `Grid.cl` uses the Fireball order `[px, py, pz, s]`. Feeding one packing to the other kernel scrambles every p lobe.
- **Species index is the index in the parsed `wfc` list**, not the column index in the gen file. A gen that lists `C H O` and a wfc file that lists `C, H, N, O` are different numberings.
- **The wfc file must belong to the SK set.** mio-1-1 and 3ob-3-1 do not share radial functions. A picture from the wrong table is a different basis than the Hamiltonian that produced `C`.
- **`z = 0` hides π states.** Put the plane ~1 Å off the nuclei.
- **Image list too short.** The STO cutoff is a few Å. If a grid point sits within that distance of an atom in a neighboring cell and that cell is not in `cell_n`, the wave is missing a piece. |\ψ|² will then show a fake dip at the boundary.
- **Real getters on a k-point run return zeros.** `get_eigvecs_dense()` is the cluster / Γ path. A periodic run prints that the real eigenvectors are unavailable; that is expected. Use `get_eigvecs_cplx`. Storage is turned on inside `dftbcore_init`. `enable_hamiltonian_storage(False)` switches it off again, and then `get_eigvecs_cplx` fails.
- **Complex data exist on the serial dense diagonalization only.** A ScaLAPACK build does not fill this store.

## What was checked

Graphene, 3ob-3-1, against the same sum written in float32 on the CPU (same spline): max |\Δψ| ~ 1e-8 for the Γ π state and for a sum of occupied π states.

At K, for shifts of 1, 2 and 3 lattice vectors along `a1`:

- max |ψ(r + m a1) − exp(−i 2π k_x m) ψ(r)| is 2e-8, 6e-8, 2e-7
- max ||ψ|²(r + m a1) − |ψ|²(r)| is ~1e-8
- the step of ψ across the cell edge is smaller than the neighboring interior steps

Ribbon total energies of each mirror pair agree to the digits printed above. That checks the Hamiltonian, not the projector. The projector on the ribbons is the same kernel.

## Not in this path

- Self-junction ribbons (`enumsj_*` under `debug/ribbon/` and `debug/ribbon_mio/`). The cell is not the orthogonal vacuum cell this driver assumes.
- `project_orbital_periodic` in `DFTBplusGridProjector.py`. It is a stub and does not apply a k-phase. Call `project_bloch_points`.
- The SPAMMM STM helpers (`spammm/SPM/AFM_utils.py` `compute_stm`, `compute_stm_fgr`). Those are Γ-only, and `compute_stm` can replace the STO tail by an exponential. `get_density_from_dftb_dense` does not write k-resolved eigenvectors. The maps here are the STO Bloch sum.
- d orbitals. The kernel stops at s+p.
