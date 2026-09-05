//! Host-side BSR4 (block-CSR, 4×4 atom blocks) data structures and helpers.
//!
//! CSR layout (matches `sparse_bsr4_purification.cl`):
//!
//! ```text
//! row_ptr[n_atom+1]   row_ptr[i] .. row_ptr[i+1]-1  = blocks of atom i
//! col_idx[nblock]     neighbor atom j of block b
//! values[nblock*16]   block b, element (r,c) at values[16*b + 4*r + c]
//! ```
//!
//! For symmetric matrices we store **both** (i,j) and (j,i) blocks so all GPU
//! kernels are gather-only (no scatter, no atomics).

use crate::core::error::{DftbError, Result};

pub const BS: usize = 4;
pub const BS2: usize = 16;

/// A BSR4 sparse matrix: atom-block CSR with 4×4 blocks.
///
/// `values.len()` must equal `col_idx.len() * BS2`.
/// `row_ptr.len()` must equal `n_atom + 1`.
/// Each row's `col_idx` slice must be **sorted ascending** (the GPU kernels
/// rely on binary search and two-pointer intersection).
#[derive(Debug, Clone)]
pub struct Bsr4Matrix {
    pub n_atom: usize,
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub values: Vec<f32>,
}

impl Bsr4Matrix {
    /// Number of stored blocks.
    pub fn nblock(&self) -> usize {
        self.col_idx.len()
    }

    /// Build an empty (all-zero values) matrix from a CSR structure.
    pub fn from_structure(n_atom: usize, row_ptr: Vec<u32>, col_idx: Vec<u32>) -> Result<Self> {
        if row_ptr.len() != n_atom + 1 {
            return Err(DftbError::InvalidInput(format!(
                "row_ptr len {} != n_atom+1 {}",
                row_ptr.len(),
                n_atom + 1
            )));
        }
        let nblock = col_idx.len();
        // Verify sorted rows.
        for i in 0..n_atom {
            let (a, b) = (row_ptr[i] as usize, row_ptr[i + 1] as usize);
            if a > b {
                return Err(DftbError::InvalidInput(format!("row {i}: row_ptr not monotonic")));
            }
            for k in a + 1..b {
                if col_idx[k] <= col_idx[k - 1] {
                    return Err(DftbError::InvalidInput(format!(
                        "row {i}: col_idx not strictly ascending at block {k}"
                    )));
                }
            }
        }
        Ok(Self {
            n_atom,
            row_ptr,
            col_idx,
            values: vec![0.0; nblock * BS2],
        })
    }

    /// Build a matrix from structure + values.
    pub fn from_parts(
        n_atom: usize,
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        values: Vec<f32>,
    ) -> Result<Self> {
        if values.len() != col_idx.len() * BS2 {
            return Err(DftbError::InvalidInput(format!(
                "values len {} != col_idx len {} * {BS2}",
                values.len(),
                col_idx.len()
            )));
        }
        let mut m = Self::from_structure(n_atom, row_ptr, col_idx)?;
        m.values = values;
        Ok(m)
    }

    /// Read block (i,j) if present, as a 4×4 row-major array.
    pub fn block(&self, i: usize, j: usize) -> Option<[f32; BS2]> {
        let b = self.find(i, j)?;
        let mut out = [0.0f32; BS2];
        out.copy_from_slice(&self.values[b * BS2..(b + 1) * BS2]);
        Some(out)
    }

    /// Binary-search block (i,j); returns the block index or None.
    pub fn find(&self, i: usize, j: usize) -> Option<usize> {
        let lo = self.row_ptr[i] as usize;
        let hi = self.row_ptr[i + 1] as usize;
        let mut a = lo;
        let mut b = hi;
        while a < b {
            let mid = (a + b) / 2;
            match self.col_idx[mid].cmp(&(j as u32)) {
                std::cmp::Ordering::Less => a = mid + 1,
                _ => b = mid,
            }
        }
        if a < hi && self.col_idx[a] == j as u32 {
            Some(a)
        } else {
            None
        }
    }

    /// Expand to a dense `(4*n_atom) × (4*n_atom)` row-major f32 matrix.
    /// Missing blocks are zero. Useful for CPU reference checks.
    pub fn to_dense(&self) -> Vec<f32> {
        let n = self.n_atom * BS;
        let mut d = vec![0.0f32; n * n];
        for i in 0..self.n_atom {
            let (a, b) = (self.row_ptr[i] as usize, self.row_ptr[i + 1] as usize);
            for blk in a..b {
                let j = self.col_idx[blk] as usize;
                let v = &self.values[blk * BS2..(blk + 1) * BS2];
                for r in 0..BS {
                    for c in 0..BS {
                        d[(i * BS + r) * n + (j * BS + c)] = v[r * BS + c];
                    }
                }
            }
        }
        d
    }

    /// Fill block (i,j) from a 4×4 row-major slice. Block must already exist
    /// in the CSR structure.
    pub fn set_block(&mut self, i: usize, j: usize, v: &[f32; BS2]) -> Result<()> {
        let b = self.find(i, j).ok_or_else(|| {
            DftbError::InvalidInput(format!("set_block: ({i},{j}) not in mask"))
        })?;
        self.values[b * BS2..(b + 1) * BS2].copy_from_slice(v);
        Ok(())
    }
}

/// A CSR sparsity mask: just the structure without values.
pub type Bsr4Mask = (Vec<u32>, Vec<u32>); // (row_ptr, col_idx)

/// Build a geometric BSR4 mask: block (i,j) exists iff `dist(i,j) <= cutoff`
/// (always including the diagonal i==j). Returns sorted CSR structure.
///
/// `pos` is in any consistent length unit (typically Å); `cutoff` in the same.
pub fn build_geometric_mask(pos: &[[f64; 3]], cutoff: f64) -> (Vec<u32>, Vec<u32>) {
    let n = pos.len();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let c2 = cutoff * cutoff;
    row_ptr.push(0u32);
    for i in 0..n {
        for j in 0..n {
            let dx = pos[i][0] - pos[j][0];
            let dy = pos[i][1] - pos[j][1];
            let dz = pos[i][2] - pos[j][2];
            if dx * dx + dy * dy + dz * dz <= c2 {
                col_idx.push(j as u32);
            }
        }
        // col_idx for this row is already ascending because j loops 0..n.
        row_ptr.push(col_idx.len() as u32);
    }
    (row_ptr, col_idx)
}

/// Build a full (dense) BSR4 mask: every block (i,j) exists.
pub fn build_full_mask(n_atom: usize) -> (Vec<u32>, Vec<u32>) {
    let row_ptr: Vec<u32> = (0..=n_atom).map(|i| (i * n_atom) as u32).collect();
    let col_idx: Vec<u32> = (0..n_atom).flat_map(|i| (0..n_atom).map(move |j| j as u32)).collect();
    (row_ptr, col_idx)
}

/// For each block b=(i,j), find the block index of the transpose (j,i).
/// Returns `INVALID = 0xffffffff` if the transpose is not in the mask
/// (should not happen for symmetric masks, but is checked on the GPU side
/// via the `b > bt` ownership guard).
pub fn transpose_block_map(m: &Bsr4Matrix) -> Vec<u32> {
    let n = m.nblock();
    let mut out = vec![0xffffffffu32; n];
    for i in 0..m.n_atom {
        let (a, b) = (m.row_ptr[i] as usize, m.row_ptr[i + 1] as usize);
        for blk in a..b {
            let j = m.col_idx[blk] as usize;
            if let Some(t) = m.find(j, i) {
                out[blk] = t as u32;
            }
        }
    }
    out
}

/// For each atom i, the block index of the diagonal block (i,i).
/// The diagonal block must be present for trace / Mulliken kernels.
pub fn diag_block_map(m: &Bsr4Matrix) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(m.n_atom);
    for i in 0..m.n_atom {
        let b = m.find(i, i).ok_or_else(|| {
            DftbError::InvalidInput(format!("diag block ({i},{i}) missing from mask"))
        })?;
        out.push(b as u32);
    }
    Ok(out)
}

/// Symmetrize a BSR4 matrix on the host: A_ij <- 0.5*(A_ij + A_ji^T).
/// Mirror of the GPU `bsr4_symmetrize` kernel, used as a CPU reference.
pub fn symmetrize_host(m: &mut Bsr4Matrix) -> Result<()> {
    let transpose = transpose_block_map(m);
    let nblock = m.nblock();
    // Work on a copy of values to avoid ordering issues.
    let src = m.values.clone();
    for b in 0..nblock {
        let bt = transpose[b];
        if bt == 0xffffffff {
            continue;
        }
        let bt = bt as usize;
        if b > bt {
            continue; // only the lower-index side owns the pair
        }
        let i_block = &src[b * BS2..(b + 1) * BS2];
        let j_block = &src[bt * BS2..(bt + 1) * BS2];
        if b == bt {
            // diagonal: X <- (X + X^T)/2
            for r in 0..BS {
                for c in r + 1..BS {
                    let v = 0.5 * (i_block[r * BS + c] + i_block[c * BS + r]);
                    m.values[b * BS2 + r * BS + c] = v;
                    m.values[b * BS2 + c * BS + r] = v;
                }
            }
        } else {
            // off-diagonal pair:
            // X_ij[r,c] <- 0.5*(X_ij[r,c] + X_ji[c,r])
            // X_ji[c,r] <- same value  (i.e. X_ji = X_ij^T)
            for r in 0..BS {
                for c in 0..BS {
                    let v = 0.5 * (i_block[r * BS + c] + j_block[c * BS + r]);
                    m.values[b * BS2 + r * BS + c] = v;
                    m.values[bt * BS2 + c * BS + r] = v;
                }
            }
        }
    }
    Ok(())
}

/// CPU reference: dense 4N×4N matrix product C = A·B (row-major f32).
pub fn dense_matmul(n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// Frobenius norm of the difference of two dense matrices.
pub fn dense_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0f32, f32::max)
}

// ============================================================================
// Boolean structural product mask:  M_T = M_K ∘ M_HS
//
// Block (i,j) exists in M_T iff there is at least one atom k such that
// (i,k) ∈ M_K and (k,j) ∈ M_HS.  This is exactly the possible support of
// T = K·S given the already-truncated K and physical S.
//
// M_T is NOT symmetric (KS is not symmetric in general).
// ============================================================================

pub fn build_product_mask(
    n_atom: usize,
    k_mask: &(Vec<u32>, Vec<u32>),
    s_mask: &(Vec<u32>, Vec<u32>),
) -> (Vec<u32>, Vec<u32>) {
    let mut row_ptr = Vec::with_capacity(n_atom + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0u32);
    for i in 0..n_atom {
        let mut neighbors: Vec<u32> = Vec::new();
        // For each k in K.neighbors(i):
        let k0 = k_mask.0[i] as usize;
        let k1 = k_mask.0[i + 1] as usize;
        for blk_k in k0..k1 {
            let k = k_mask.1[blk_k] as usize;
            // For each j in S.neighbors(k):
            let s0 = s_mask.0[k] as usize;
            let s1 = s_mask.0[k + 1] as usize;
            for blk_s in s0..s1 {
                let j = s_mask.1[blk_s];
                if !neighbors.contains(&j) {
                    neighbors.push(j);
                }
            }
        }
        neighbors.sort();
        // Deduplicate (already sorted, just remove consecutive dups).
        neighbors.dedup();
        col_idx.extend(neighbors);
        row_ptr.push(col_idx.len() as u32);
    }
    (row_ptr, col_idx)
}

/// Build a Bsr4Matrix representing the identity (4×4 identity on each
/// diagonal block, zero elsewhere) on the given mask. The mask must include
/// all diagonal blocks (i,i).
pub fn build_identity(n_atom: usize, mask: &(Vec<u32>, Vec<u32>)) -> Result<Bsr4Matrix> {
    let mut m = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone())?;
    for i in 0..n_atom {
        let v = [
            1.0f32, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        m.set_block(i, i, &v)?;
    }
    Ok(m)
}

/// Gershgorin spectral bounds of a BSR4 matrix B at the **orbital** level.
///
/// Returns (emin, emax) where:
///   emin = min_i ( B_ii - sum_{j≠i} |B_ij| )
///   emax = max_i ( B_ii + sum_{j≠i} |B_ij| )
///
/// with i,j running over individual orbitals (not atom blocks).
/// B must include all diagonal blocks.
pub fn gershgorin_bounds(b: &Bsr4Matrix) -> Result<(f32, f32)> {
    let n_atom = b.n_atom;
    let n_orb = n_atom * BS;
    // Build dense to compute orbital-level row sums. For the sizes we expect
    // (masks are sparse but bounds need full row sums), this is acceptable
    // as a one-time-per-geometry computation.
    let dense = b.to_dense();
    let mut emin = f32::INFINITY;
    let mut emax = f32::NEG_INFINITY;
    for mu in 0..n_orb {
        let diag = dense[mu * n_orb + mu];
        let mut offdiag_sum = 0.0f32;
        for nu in 0..n_orb {
            if nu != mu {
                offdiag_sum += dense[mu * n_orb + nu].abs();
            }
        }
        emin = emin.min(diag - offdiag_sum);
        emax = emax.max(diag + offdiag_sum);
    }
    Ok((emin, emax))
}

/// Infinity norm of a BSR4 matrix (max orbital-level absolute row sum).
pub fn inf_norm(b: &Bsr4Matrix) -> f32 {
    let n_orb = b.n_atom * BS;
    let dense = b.to_dense();
    let mut max_row = 0.0f32;
    for mu in 0..n_orb {
        let row_sum: f32 = (0..n_orb).map(|nu| dense[mu * n_orb + nu].abs()).sum();
        max_row = max_row.max(row_sum);
    }
    max_row
}

/// Frobenius norm of a dense matrix.
pub fn dense_frobenius(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Trace of a dense n×n matrix.
pub fn dense_trace(a: &[f32], n: usize) -> f32 {
    (0..n).map(|i| a[i * n + i]).sum()
}
