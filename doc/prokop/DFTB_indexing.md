# Fortran Sparse Indexing vs. Rust Reimplementation

## 1. The Sparse Data Structure (`iPair` / `iSparseStart`)

Fortran DFTB+ does **not** store the full dense Hamiltonian. Instead, it stores a 1D "primitive" sparse vector. The indexing array that manages this is called `iPair` (or `iSparseStart` in some modules).

### How `iPair` is built

From `getSparseDescriptor` in `@/home/prokophapala/git/dftbplus/src/dftbp/dftb/sparse2dense.F90:3616-3662`:

```fortran
ind = 0
do iAt1 = 1, nAtom
  nOrb1 = orb%nOrbAtom(iAt1)
  do iNeigh1 = 0, nNeighbour(iAt1)
    iPair(iNeigh1, iAt1) = ind
    nOrb2 = orb%nOrbAtom(img2CentCell(iNeighbour(iNeigh1, iAt1)))
    ind = ind + nOrb1 * nOrb2
  end do
end do
sparseSize = ind
```

### In plain English

- **Loop over every atom** `iAt1` in the central cell.
- **Loop over every neighbor** of that atom, including the atom itself (`iNeigh1 = 0` is the **onsite** block).
- For each `(atom, neighbor)` pair, `iPair(iNeigh1, iAt1)` stores the **0-based byte offset** into the 1D sparse array where that atomic block starts.
- Each block occupies exactly `nOrb1 * nOrb2` consecutive doubles in memory.

### The onsite block (neighbor 0)

For `iNeigh1 = 0`, `iNeighbour(0, iAt1) = iAt1`, so `nOrb2 = nOrb1`. The onsite block has size `nOrb1 * nOrb1`.

Inside `buildH0` (`@/home/prokophapala/git/dftbplus/src/dftbp/dftb/nonscc.F90:109-116`):

```fortran
ind = iPair(0, iAt1) + 1
do iOrb1 = 1, orb%nOrbAtom(iAt1)
  ham(ind) = selfegy(...)
  ind = ind + orb%nOrbAtom(iAt1) + 1
end do
```

This stores **only the diagonal elements** of the onsite block, striding by `nOrb+1`. Because Fortran arrays are **1-based**, the positions are:

| relative index | meaning |
|---|---|
| `+1` | orbital 1, orbital 1 |
| `+1 + (nOrb+1)` | orbital 2, orbital 2 |
| `+1 + 2*(nOrb+1)` | orbital 3, orbital 3 |

This works because the full `nOrb × nOrb` block is stored in **column-major** order, and the diagonal of a column-major square matrix is exactly spaced by `nOrb+1`. All off-diagonal onsite elements remain zero (initialized earlier).

### The offsite blocks (neighbors 1..n)

Inside `buildDiatomicBlocks` (`@/home/prokophapala/git/dftbplus/src/dftbp/dftb/nonscc.F90:406-420`):

```fortran
do iNeigh1 = 1, nNeighbourSK(iAt1)
  iAt2 = iNeighbours(iNeigh1, iAt1)
  ...
  ind = iPair(iNeigh1, iAt1)
  ...
  out(ind + 1 : ind + nOrb2 * nOrb1) = reshape(tmp(1:nOrb2, 1:nOrb1), [nOrb2 * nOrb1])
end do
```

The block is stored as a flat column-major vector of length `nOrb2 * nOrb1`. Later, `unpackHS_real` reconstructs it with:

```fortran
square(jj:jj+nOrb2-1, ii:ii+nOrb1-1) = &
    square(jj:jj+nOrb2-1, ii:ii+nOrb1-1) + &
    reshape(orig(iOrig:iOrig+nOrb1*nOrb2-1), [nOrb2, nOrb1])
```

**Key convention:** the block stored at `iPair(iNeigh, iAt1)` represents the **lower-left** submatrix of the dense Hamiltonian:
- **rows** = orbitals of the neighbor atom `iAt2`
- **cols** = orbitals of the central atom `iAt1`

When unpacking, it is placed at `square(jj:jj+nOrb2-1, ii:ii+nOrb1-1)`. Only the lower triangle is filled; the upper triangle is reconstructed by hermitian symmetry elsewhere.

### Comparison to Rust

The Rust code in `@/home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:52-53` builds **dense** matrices directly:

```rust
let mut h0 = DMatrix::<f64>::zeros(n_orb, n_orb);
let mut s = DMatrix::<f64>::identity(n_orb, n_orb);
```

This is fine for small molecules, but it skips the entire sparse machinery. There is no `iPair` equivalent. The orbital offsets are computed cumulatively:

```rust
let mut i_orb_atom = vec![0usize; n_atom + 1];
for i in 0..n_atom {
    let n = self.sk.n_orb_species(&species[i])?;
    i_orb_atom[i + 1] = i_orb_atom[i] + n;
}
```

This is the Rust equivalent of Fortran's `iAtomStart`.

## 2. Orbital Ordering (Tesseral / Real Spherical Harmonics)

From `@/home/prokophapala/git/dftbplus/src/dftbp/type/orbitals.F90:59-64`:

```fortran
character(lenOrbitalNames), parameter :: orbitalNames(-3:3,0:3) = reshape([&
    ...
    & '         ','         ','y        ','z        ','x        ','         ','         ',&
    & '         ','xy       ','yz       ','z2       ','xz       ','x2-y2    ','         ',&
    ...
    &], [7,4])
```

### In plain English

For each angular momentum `l`, the orbitals are ordered by **magnetic quantum number m = -l, ..., +l** (tesseral / real spherical harmonics):

| shell | order | names |
|---|---|---|
| `l=0` (s) | 1 orbital | `s` |
| `l=1` (p) | 3 orbitals | `py` (m=-1), `pz` (m=0), `px` (m=+1) |
| `l=2` (d) | 5 orbitals | `xy`, `yz`, `z2`, `xz`, `x2-y2` |
| `l=3` (f) | 7 orbitals | `y(3x²-y²)`, `x²+y²+z²`, `yz²`, `z³`, `xz²`, `z(x²-y²)`, `x(x²-3y²)` |

**Important:** the p-order is **`py, pz, px`**, not `px, py, pz`.

### Per-atom orbital layout

From `rotateH0` (`@/home/prokophapala/git/dftbplus/src/dftbp/dftb/sk.F90:72-138`):

```fortran
iCol = 1
do iSh1 = 1, orb%nShell(iSp1)
  ang1 = orb%angShell(iSh1, iSp1)
  nOrb1 = 2 * ang1 + 1
  iRow = 1
  do iSh2 = 1, orb%nShell(iSp2)
    ang2 = orb%angShell(iSh2, iSp2)
    nOrb2 = 2 * ang2 + 1
    ...
    iRow = iRow + nOrb2
  end do
  iCol = iCol + nOrb1
end do
```

For a species with shells `[s, p, d]`:
- orbitals 1..1 = s
- orbitals 2..4 = p (py, pz, px)
- orbitals 5..9 = d (xy, yz, z2, xz, x2-y2)

### Comparison to Rust

In `@/home/prokophapala/git/dftbplus/rust_dftb/src/hamiltonian.rs:67-78`, the Rust code places onsite energies shell by shell:

```rust
for &l in ang {
    let n = (2 * l + 1) as usize;
    for k in 0..n {
        h0[(base + off + k, base + off + k)] = e;
    }
    off += n;
}
```

And in `@/home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:107-139`, the shell loops are:

```rust
let mut i_col = 0;
for &ang1 in ang1_list {
    let mut i_row = 0;
    for &ang2 in ang2_list {
        ...
        i_row += n_orb2_sh;
    }
    i_col += n_orb1_sh;
}
```

This matches Fortran exactly: outer loop over atom-i shells, inner loop over atom-j shells.

## 3. Neighbor List Conventions

From `updateNeighbourList` (`@/home/prokophapala/git/dftbplus/src/dftbp/dftb/periodic.F90:446-698`):

### Key rules

1. **`iNeighbour(0, iAt1) = iAt1`** — every atom is its own 0-th neighbor.
2. **Neighbors are sorted by distance** (ascending).
3. **`nNeighbourSK(iAt1)`** counts only the **offsite** neighbors; the total number of entries for atom `iAt1` is `nNeighbourSK(iAt1) + 1` (including the self-entry at index 0).
4. In non-symmetric mode (DFTB+ default), the neighbor list is **lower-triangular only**: atom `iAt1` only lists neighbors `iAt2 <= iAt1` (or periodic images thereof). The upper triangle is reconstructed by symmetry during unpacking.

### Comparison to Rust

In `@/home/prokophapala/git/dftbplus/rust_dftb/src/neighbor.rs:23-41`:

```rust
for i in 0..n {
    for j in (i + 1)..n {
        ...
        if r <= self.cutoff {
            pairs.push(NeighborPair { i, j, r, vec_ij: v });
        }
    }
}
```

The Rust code uses a **full pairwise list** (`i < j`), not the sparse lower-triangular convention. This is fine for a dense-matrix builder because it writes both `H[i,j]` and `H[j,i]` explicitly.

**Discrepancy:** The Rust neighbor list does **not** include the self-pair (`i==j`). The onsite block is handled separately in `fill_onsite`. This is functionally equivalent, but the data structure is different.

## 4. Slater-Koster Table Lookup (`getFullTable`)

This is where many subtle bugs hide.

### The two SK file arrays

In Fortran (`@/home/prokophapala/git/dftbplus/src/dftbp/dftbplus/parser.F90:3955-4026`):

```fortran
subroutine getFullTable(skHam, skOver, skData12, skData21, angShells1, angShells2)
```

- **`skData12`** — SK files for the pair read as **species1–species2**.
- **`skData21`** — SK files for the pair read as **species2–species1**.
- Both are needed because old-format `.skf` files store integrals with the convention that the **lower angular momentum** species comes "first" in the file's internal column ordering.

### The `skMap` array

```fortran
integer, parameter :: skMap(0:maxL, 0:maxL, 0:maxL) = reshape((/&
    &20, 0,  0,  0,  19,  0,  0,  0,  18,  0,  0,  0,  17,  0,  0,  0,&
    & 0, 0,  0,  0,  15, 16,  0,  0,  13, 14,  0,  0,  11, 12,  0,  0,&
    & 0, 0,  0,  0,   0,  0,  0,  0,   8,  9, 10,  0,   5,  6,  7,  0,&
    & 0, 0,  0,  0,   0,  0,  0,  0,   0,  0,  0,  0,   1,  2,  3,  4/),&
    &(/maxL + 1, maxL + 1, maxL + 1/))
```

`skMap(mm, lMax, lMin)` converts `(mm, lMax, lMin)` to a **column index** in the old-format `.skf` grid table.

### The lookup logic

```fortran
if (l1 <= l2) then
  pHam => skData12(iSK2,iSK1)%skHam
  lMin = l1
  lMax = l2
else
  pHam => skData21(iSK1,iSK2)%skHam
  lMin = l2
  lMax = l1
end if
do mm = 0, lMin
  skHam(:,ind) = pHam(:,skMap(mm,lMax,lMin))
  ind = ind + 1
end do
```

### In plain English

1. For a shell pair with angular momenta `l1` (species A) and `l2` (species B):
   - If `l1 <= l2`, use the `skData12` file (A–B) directly.
   - If `l1 > l2`, use the **reversed** file `skData21` (B–A).
2. Set `lMin = min(l1,l2)`, `lMax = max(l1,l2)`.
3. For `mm = 0, 1, ..., lMin`, extract column `skMap(mm, lMax, lMin)` from the chosen file.
4. These `lMin+1` columns become the `lMin+1` SK integrals (σ, π, δ, ...) for this shell pair.

**Why this matters:** Old SK files have a fixed column layout. For example, a C–C sp file has columns arranged assuming `l_s <= l_p`. If you need a p–s block (where `l1 > l2`), you must read the **transposed** file and handle the parity sign separately in `rotateH0`.

### Comparison to Rust

In `@/home/prokophapala/git/dftbplus/rust_dftb/src/sk_data.rs:171-198`:

```rust
let (lookup_sp1, lookup_sp2) = if ang1 <= ang2 {
    (sp1, sp2)
} else {
    (sp2, sp1)
};
let tab = self.get_pair(lookup_sp1, lookup_sp2).ok_or_else(|| ...)?;
let h_all = tab.h.eval(r)?;
let s_all = tab.s.eval(r)?;
...
let h_shell = if is_extended {
    extract_shell_integrals_new(&h_all, ang1, ang2)
} else {
    extract_shell_integrals_old(&h_all, ang1, ang2)
};
```

The Rust code correctly swaps `(sp1, sp2)` when `ang1 > ang2`, matching Fortran's `skData21` logic. The `extract_shell_integrals_*` functions must then apply the same `skMap` indexing to pick the right columns from the interpolated 1D array.

**Discrepancy watch:** The extended-format (new) vs old-format column indices in `sk_data.rs` were already a source of bugs (see Testing_and_Parity_Codemap.md fixes 1 and 2). The old format stores 20 columns in a specific order; the extended format stores up to 20 columns with a different layout. If `extract_shell_integrals_old/new` do not exactly mirror `skMap`, the integrals will be wrong.

---

## 5. Block Assembly in `rotateH0`

From `@/home/prokophapala/git/dftbplus/src/dftbp/dftb/sk.F90:128-133`:

```fortran
if (ang1 <= ang2) then
  hh(iRow:iRow+nOrb2-1,iCol:iCol+nOrb1-1) = tmpH(1:nOrb2,1:nOrb1)
else
  hh(iRow:iRow+nOrb2-1,iCol:iCol+nOrb1-1) = (-1.0_dp)**(ang1+ang2) &
      &* transpose(tmpH(1:nOrb1,1:nOrb2))
end if
```

### In plain English

- `tmpH` is the raw rotated sub-block returned by `ss`, `sp`, `pp`, etc.
- If `ang1 <= ang2`, place `tmpH` directly.
- If `ang1 > ang2`, **transpose** `tmpH` and multiply by `(-1)^(ang1+ang2)`.

**Why the transpose?** The rotation routines like `sp`, `sd`, etc., are written assuming the **first** angular momentum is the **column** (atom i) and the **second** is the **row** (atom j), with `ang1 <= ang2`. When `ang1 > ang2`, the routine is still called with the original arguments (e.g., `sp(tmpH, ll, mm, nn, pSK)` for p-s), but the result needs to be transposed to fit into the global block where columns = atom i and rows = atom j.

**Why the sign?** Swapping the two atoms in a diatomic interaction introduces a parity factor `(-1)^(l1+l2)` because the real spherical harmonics pick up a sign under inversion (depending on `l`).

### The rotation subroutines

For `pp` (`@/home/prokophapala/git/dftbplus/src/dftbp/dftb/sk.F90:254-285`):

```fortran
hh(1,1) = (1.0_dp-nn**2-ll**2)*sk(1)+(nn**2+ll**2)*sk(2)   ! py-py (sigma/pi)
hh(2,1) = nn*mm*sk(1)-nn*mm*sk(2)                         ! pz-py
hh(3,1) = ll*mm*sk(1)-ll*mm*sk(2)                         ! px-py
...
```

`sk(1)` = σ integral, `sk(2)` = π integral. The matrix is symmetric and stored in full.

### Comparison to Rust

In `@/home/prokophapala/git/dftbplus/rust_dftb/src/rotation.rs:119-133`:

```rust
if ang1 <= ang2 {
    for a in 0..n_orb2_sh {
        for b in 0..n_orb1_sh {
            h_blk[(i_row + a, i_col + b)] = h_sub[(a, b)];
        }
    }
} else {
    let sign = if (ang1 + ang2) % 2 == 0 { 1.0 } else { -1.0 };
    for a in 0..n_orb2_sh {
        for b in 0..n_orb1_sh {
            h_blk[(i_row + a, i_col + b)] = sign * h_sub[(b, a)];
        }
    }
}
```

This matches Fortran **exactly** (after the bug fix noted in the codemap).

## 6. Summary of Discrepancies (Rust vs. Fortran)

Based on the code and the Testing_and_Parity_Codemap, here are the key differences and past fixes:

| # | Location | Issue | Fortran Reference |
|---|---|---|---|
| 1 | `sk_data.rs` | Old-format `skMap` column indices misread | `skMap` in `parser.F90:3984-3990` |
| 2 | `sk_data.rs` | Extended-format column indices misread | New format has different column order |
| 3 | `rotation.rs` | Missing `(-1)^(ang1+ang2)` sign for `ang1 > ang2` | `rotateH0` in `sk.F90:131` |
| 4 | `hamiltonian.rs` | Block transpose convention was wrong | `buildDiatomicBlocks` stores `tmp(1:nOrb2, 1:nOrb1)` |
| 5 | `hamiltonian.rs` | Hardcoded `n_orb = 4 * n_atom` | Should use `orb%nOrbAtom` per atom |
| 6 | `sk_data.rs` | Reversed SK file not used when `ang1 > ang2` | `getFullTable` swaps `skData12` ↔ `skData21` |

### What the Rust code does correctly

- Dense matrix assembly with per-atom orbital offsets (`i_orb_atom`).
- Shell-by-shell iteration in the correct order (s, p, d, f).
- Tesseral orbital ordering (py, pz, px) inside p-shells.
- Direction cosine conventions (`l, m, n` = `x, y, z` components of unit vector).
- Symmetrization of H and S (`H[j,i] = H[i,j]`).

### What the Rust code omits (by design)

- **Sparse storage**: Rust builds dense `DMatrix` directly. This is fine for small systems.
- **Periodic boundary conditions**: No `img2CentCell`, `iCellVec`, or k-point phases.
- **MPI chunking**: `distributeRangeInChunks` and `assembleChunks` are not needed in Rust.
- **Neighbor sorting by distance**: Rust just uses `i < j` ordering.

## 7. Quick Checklist for Future Parity Debugging

If H0 or S mismatch appears again, check in this order:

1. **SK column indices** — Are `extract_shell_integrals_old` / `extract_shell_integrals_new` picking the right columns from the interpolated 20-element array?
2. **Species swap** — When `ang1 > ang2`, does Rust load the reversed SK pair and apply the parity sign?
3. **Orbital offsets** — Does `i_orb_atom` match Fortran's `iAtomStart` exactly?
4. **Block shape** — Is the rotated block placed as `rows = nOrb_j`, `cols = nOrb_i`?
5. **Onsite stride** — For sparse comparisons, remember Fortran onsite diagonals are strided by `nOrb+1`, not contiguous.
6. **p-ordering** — `py, pz, px` not `px, py, pz`.

If you need, I can now create a side-by-side code snippet document or add this to your existing [Testing_and_Parity_Codemap.md](cci:7://file:///home/prokophapala/git/dftbplus/doc/prokop/Testing_and_Parity_Codemap.md:0:0-0:0).