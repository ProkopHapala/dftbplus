# DFTBcore — Python access to DFTB+ matrices (incl. periodic k-points)

`DFTBcore` is a thin ctypes wrapper over `libdftbcore.so` (Fortran, `app/dftbcore/libdftbcore.F90`). It runs a normal DFTB+ input file and hands you the internal matrices as NumPy arrays — Hamiltonian, overlap, density matrix, MO coefficients, eigenvalues — including **complex matrices at arbitrary k-points** for periodic systems.

Python side: `pyBall/DFTBcore.py`. Runnable examples: `tests/dftb/test_dftbcore*.py`, `tests/dftb/plot_*.py`.

---

## 1. Build & load

Requires the shared-library build (`-DBUILD_SHARED_LIBS=ON`). The wrapper searches `_build/app/dftbcore/libdftbcore.so`, the CWD, then `~/opt/dftbplus/lib/`. When the lib is found under `_build/`, the sibling `libdftbplus.so` is preloaded so a stale installed copy cannot shadow it via rpath.

```python
import sys; sys.path.insert(0, '/path/to/dftbplus/pyBall')
from DFTBcore import DFTBcore

dftb = DFTBcore()                      # or DFTBcore(libpath='.../libdftbcore.so')
dftb.init('input.hsd')                 # parses the hsd file you pass
dftb.enable_matrix_collection(dm=True, h=True, s=True)   # BEFORE run_scf
E = dftb.run_scf()                     # Hartree; runs the full driver (incl. geo opt)
dftb.finalize()
```

`init` writes `dftb_pin.hsd`, `detailed.out`, `results.tag`, `charges.bin` etc. into the **current directory** — run inside a scratch dir (`os.chdir(wd)`).

## 2. Cluster / Γ-only (real path)

```python
n   = dftb.get_basis_size()
H   = dftb.get_h_dense()      # (n,n) float64
S   = dftb.get_s_dense()
P   = dftb.get_dm_dense()
C,E = dftb.get_eigvecs_dense()   # C[orb,mo], E[mo]
```

For a run with non-Γ k-points these return **zeros** — a warning is printed; use the complex getters below.

## 3. Periodic with k-points (complex path)

```python
norb, nks, nkpts, nspin = dftb.get_cplx_dims()
kpts, w = dftb.get_kpoints()          # kpts (nk,3) fractional recip. coords, w (nk,)
H = dftb.get_h_cplx()                 # (nks, norb, norb) complex128
S = dftb.get_s_cplx()
P = dftb.get_dm_cplx()
C, E = dftb.get_eigvecs_cplx()        # C (nks,norb,norb) [iks,mo,orb], E (nks,norb) [Ha]
```

Conventions:

- **Slot index** `iks = ik + ispin*nkpts` (k runs fastest). Unpolarized: `iks == ik`.
- **Spin**: for collinear spin-polarized runs `nks = nspin*nkpts` and `get_dm_cplx()`
  returns the **per-spin-channel** DM — slots `0..nk-1` are spin-up, `nk..2nk-1`
  spin-down. Use `get_dm_cplx_total()` (sums channels) for the physical density;
  using only the first `nk` slots (or the real-path `get_dm_dense()` on older
  builds, which kept only the *last* channel — often the empty one for a fully
  polarized system) silently drops density.
- `P(k)` is the per-k-point DM **without k-weights** — apply `w_k` yourself when integrating.
- Matrices are Hermitian; `H(k) C(k) = E S(k) C(k)` holds to ~1e-15.
- The packing convention (from `unpackHS_cmplx_kpts`) is
  `A(k)[j,i] = Σ_R e^{+ik·R} A_{j(R),i(0)}` — `i` = orbital in home cell, `j` = orbital in cell shifted by `R`. Inverting requires `e^{-ik·R}`.
- **Project `C`, not `C*`.** The eigenvector equation holds for the unconjugated columns. At a general k, `C*` is not an eigenvector. Real-space pictures of these states: [bloch_slice.md](bloch_slice.md).

Cheap sanity checks (used in `test_dftbcore_kpoints.py`):

```python
assert np.abs(H - H.conj().transpose(0,2,1)).max() < 1e-12      # Hermitian
np.einsum('kij,kjl,kml->kim', C.conj(), S, C)                    # ≈ I per k
sum_k w_k Tr(S(k) P(k)) = N_e                                  # electron count
```

## 4. Real-space bond orders (the point of all this)

Inverse-Fourier-transform to cell-pair blocks, then a Mulliken-type bond order:

```python
def fourier_to_realspace(Ak, kpts, w, R_cells):
    kx = kpts[:,0]
    return {R: np.einsum('k,kij->ij', w*np.exp(-2j*np.pi*kx*R), Ak)
            for R in R_cells}                       # 1-D chain; generalize k·R for 2/3-D

P_R = fourier_to_realspace(P, kpts, w, [-1,0,1])
S_R = fourier_to_realspace(S, kpts, w, [-1,0,1])

def bond_order(R, atomA_orbs, atomB_orbs):          # B in cell R, A in cell 0
    i, j = np.ix_(atomB_orbs, atomA_orbs)
    return np.sum((P_R[R][i,j] * S_R[R][i,j]).real)
```

`BO_AB(R) = Σ Re[P_νμ(R) S_νμ(R)]` — Σ over orbitals of the two atoms. The same physical bond must appear from both sides (`R` and `-R` swapped indices) — use it as a Hermiticity cross-check.

## 5. Worked example 1 — Peierls distortion of a carbon chain

`test_dftbcore_peierls.py`: 2-atom cell, a=2.60 Å, scan `d_intra` (C1–C2 within cell) so `d_inter = a − d_intra` alternates. nk=8 suffices for the |R|≤1 blocks.

| d_intra | gap [eV] | BO_intra | BO_inter |
|---|---|---|---|
| 1.30 | 0.00 | 0.62 | 0.62 |
| 1.20 | 2.82 | 0.80 | 0.47 |
| 1.10 | 5.92 | 0.90 | 0.41 |

Equal bonds → BO equal and gap closes (metallic); dimerization opens the gap **at the zone edge X** (4-fold π degeneracy at kx=0.5 splits) — textbook SSH. `plot_peierls.py` renders geometry+bonds plus folded bands.

## 6. Worked example 2 — poly(p-phenylene), constrained relaxation

`test_dftbcore_ppp.py` + `plot_ppp.py`: C₆H₄ per cell. The two para (linking) carbons are moved apart along x and **fixed**; the rest relaxes inside the same run — the driver executes inside `run_scf()`:

```
Driver = GeometryOptimisation {
  Optimiser = LBFGS { Memory = 20 }
  MovedAtoms = 2 3 5 6 7 8 9 10     # everything except fixed atoms 1,4
  Convergence { GradElem = 1e-4 }
  MaxSteps = 100
  OutputPrefix = "geo_end"          # relaxed geometry -> geo_end.gen
}
```

Stretching the para-para axis at fixed cell drives aromatic→quinoid: the inter-ring bond order rises 0.40→0.62 (double-bond character), para-adjacent ring bonds weaken, and the band gap is non-monotonic (2.29 → 0.52 → 1.00 eV) as the two electronic structures cross.

## 7. Pitfalls (all learned the hard way)

- **gen `'S'` vs `'F'`**: `'S'` = periodic **Cartesian Å**, `'F'` = fractional. Writing fractional coords under `'S'` silently produces a wrong (e.g. dimerized) geometry — this once masqueraded as a broken Hamiltonian. Check `dftb_pin.hsd` and the actual bond lengths if results look insane.
- **Atoms outside [0,a)**: DFTB+ folds them into the home cell; their couplings then land in `P(R=±1)` instead of `P(R=0)`. Keep coordinates inside the cell, or search all R channels for each bond.
- **DM weights**: `get_dm_cplx()` excludes `w_k`; integrate with weights.
- **Open-shell systems**: per-spin DM slots must be summed (`get_dm_cplx_total()`);
  regression: `tests/dftb/test_dftbcore_spin.py` (H atom cluster + H chain).
- **BZ plotting**: when recentering `kx∈[0,1)` to `[-0.5,0.5)`, the physical X point exists only once. Duplicate `kx=0.5` explicitly at both plot edges or the gap closure is not visible (`kx_centered` in `plot_ppp.py` does this + asserts the two edge spectra are identical).
- **Stale installed lib**: if `~/opt/dftbplus` shadows `_build` symbols, pass `libpath=` under `_build/` — the wrapper preloads the matching `libdftbplus.so`.

## 8. Limitations

- Complex stores hooked in the **serial** dense path only (no ScaLAPACK/BLACS); Pauli spin (`nSpin=4`) not covered.
- `get_h_dense()` etc. intentionally return zeros for k-point runs rather than a wrong real matrix.
- Bond order is Mulliken-like (basis-set dependent) — good for trends/SSH-style alternation, not an absolute measure.
