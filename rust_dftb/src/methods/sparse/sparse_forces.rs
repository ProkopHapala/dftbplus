//! Sparse analytic force support: `D = 2K` and `W = 2 K H_scc K` (P3, manifest v3 §4.4).
//!
//! Builds the density and energy-weighted density matrices needed by the
//! sparse force path **without diagonalization**. The density kernel `K`
//! comes from TC2 purification; `H_scc` is the SCC Hamiltonian on the H/S
//! mask. The products are masked SpGEMM operations on the GPU.
//!
//! ## Algorithm (manifest v3 §4.4)
//!
//! ```text
//! M_TW = boolean_product(M_K, M_HS)   // exact for the truncated operands
//! T    = K * H_scc                    // on M_TW
//! W    = 2 * project_MHS(T * K)       // final short-range support
//! symmetrize(W)
//! D    = 2 * K                        // same mask as K
//! ```
//!
//! `W` is built only on `M_HS` (the H/S support mask), never as a dense
//! matrix. This is the diagonalization-free route: no eigenvalues, no
//! eigenvectors, no dense EDM.
//!
//! ## Parity gate (manifest v3 §4.4)
//!
//! At the same geometry and same Hamiltonian, compare:
//!   - `D_sparse = 2K`  vs dense occupied-orbital D
//!   - `W_sparse = 2KHK` vs dense eigenvalue/eigenvector W
//!   - `F_sparse` vs dense analytic force
//!
//! Hessian implementation is blocked until this passes.

use crate::core::error::Result;
use crate::methods::sparse::bsr4::{build_product_mask, Bsr4Matrix, BS, BS2};
use crate::methods::sparse::gpu_sparse::{GpuBsrMatrix, GpuBsrStructure, SparseBsr4Gpu};
use std::sync::Arc;

/// Output of `build_dw_sparse`: device-resident `D` and `W` matrices.
///
/// `D` lives on `M_K` (the density-kernel mask), `W` lives on `M_HS`
/// (the H/S support mask). Both are symmetric.
pub struct SparseDW {
    /// Density matrix `D = 2K` on `M_K`.
    pub d: GpuBsrMatrix,
    /// Energy-weighted density matrix `W = 2 K H_scc K` on `M_HS`.
    pub w: GpuBsrMatrix,
}

/// Build `D = 2K` and `W = 2 K H_scc K` on the GPU using masked SpGEMM.
///
/// Inputs:
/// - `k`: density kernel (purified) on mask `M_K`
/// - `h_scc`: SCC Hamiltonian on mask `M_HS` (symmetric)
///
/// The boolean product mask `M_TW = M_K ∩ M_HS` is computed on the host
/// (cheap, one-time per topology). All matrix products are device-resident
/// masked SpGEMM — no dense allocation, no host roundtrip.
///
/// Returns `D` on `M_K` and `W` on `M_HS`, both symmetric.
pub fn build_dw_sparse(
    gpu: &SparseBsr4Gpu,
    k: &GpuBsrMatrix,
    h_scc: &GpuBsrMatrix,
    m_k: &(Vec<u32>, Vec<u32>),
    m_hs: &(Vec<u32>, Vec<u32>),
) -> Result<SparseDW> {
    let n_atom = k.struct_.n_atom;
    assert_eq!(h_scc.struct_.n_atom, n_atom, "K and H_scc must have same n_atom");

    // 1. M_TW = boolean_product(M_K, M_HS) — host-side, one-time per topology.
    let m_tw = build_product_mask(n_atom, m_k, m_hs);

    // 2. Allocate device structures for T (on M_TW) and W (on M_HS).
    let tw_struct = Arc::new(GpuBsrStructure::new(gpu, n_atom, &m_tw)?);
    let hs_struct = h_scc.struct_.clone(); // W reuses M_HS structure
    let t = GpuBsrMatrix::zero(gpu, &tw_struct)?;
    let w = GpuBsrMatrix::zero(gpu, &hs_struct)?;

    // 3. T = K * H_scc on M_TW (masked SpGEMM, device-resident).
    gpu.spgemm_masked_dev(k, h_scc, &t)?;

    // 4. W = T * K on M_HS (masked SpGEMM, device-resident).
    //    Then scale by 2: W = 2 * project_MHS(T * K).
    gpu.spgemm_masked_dev(&t, k, &w)?;
    let nblock_w = w.struct_.nblock;
    gpu.axpby_dev(nblock_w, 2.0, &w.values, 0.0, &w.values, &w.values)?;

    // 5. symmetrize(W) — W must be symmetric for the force contraction.
    gpu.symmetrize_dev(nblock_w, w.struct_.transpose_block(), &w.values)?;

    // 6. D = 2 * K (scale K values by 2, same mask M_K).
    let d = GpuBsrMatrix::zero(gpu, &k.struct_)?;
    let nblock_k = k.struct_.nblock;
    gpu.axpby_dev(nblock_k, 2.0, &k.values, 0.0, &k.values, &d.values)?;

    Ok(SparseDW { d, w })
}

/// Host-side reference: build `D = 2K` and `W = 2 K H_scc K` using dense
/// arithmetic for parity validation. This is **test/reference only** —
/// production uses `build_dw_sparse` (device-resident, masked).
///
/// Returns `(D_dense, W_dense)` as `(n_orb × n_orb)` row-major f64 matrices
/// where `n_orb = n_atom * 4` (BSR4 padded).
#[cfg(test)]
pub fn build_dw_dense_reference(
    k: &Bsr4Matrix,
    h_scc: &Bsr4Matrix,
) -> (Vec<f64>, Vec<f64>, usize) {
    let n_atom = k.n_atom;
    let n_orb = n_atom * BS;
    // f32 dense → f64 dense
    let k_dense_f32 = k.to_dense();
    let h_dense_f32 = h_scc.to_dense();
    let k_dense: Vec<f64> = k_dense_f32.iter().map(|x| *x as f64).collect();
    let h_dense: Vec<f64> = h_dense_f32.iter().map(|x| *x as f64).collect();

    // D = 2 * K (dense)
    let d_dense: Vec<f64> = k_dense.iter().map(|x| 2.0 * x).collect();

    // W = 2 * K * H_scc * K (dense triple product)
    let t_dense = dense_matmul_f64(n_orb, &k_dense, &h_dense);
    let mut w_dense = dense_matmul_f64(n_orb, &t_dense, &k_dense);
    for v in w_dense.iter_mut() { *v *= 2.0; }
    // Symmetrize W
    for i in 0..n_orb {
        for j in (i + 1)..n_orb {
            let avg = 0.5 * (w_dense[i * n_orb + j] + w_dense[j * n_orb + i]);
            w_dense[i * n_orb + j] = avg;
            w_dense[j * n_orb + i] = avg;
        }
    }
    (d_dense, w_dense, n_orb)
}

/// Dense f64 matrix multiply: C = A · B (row-major, n×n).
#[cfg(test)]
fn dense_matmul_f64(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Max absolute difference between two f32 BSR4 matrices on the same mask.
#[cfg(test)]
pub fn bsr4_max_abs_diff(a: &Bsr4Matrix, b: &Bsr4Matrix) -> f32 {
    assert_eq!(a.nblock(), b.nblock(), "matrix block count mismatch");
    assert_eq!(a.row_ptr, b.row_ptr, "row_ptr mismatch");
    assert_eq!(a.col_idx, b.col_idx, "col_idx mismatch");
    let mut max = 0.0f32;
    for i in 0..a.values.len() {
        max = max.max((a.values[i] - b.values[i]).abs());
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::sparse::bsr4::build_geometric_mask;
    use crate::methods::sparse::gpu_sparse::SparseBsr4Config;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Try to create a GPU; return None if no OpenCL device.
    fn try_gpu() -> Option<SparseBsr4Gpu> {
        match catch_unwind(AssertUnwindSafe(|| {
            SparseBsr4Gpu::new(SparseBsr4Config::default())
        })) {
            Ok(Ok(gpu)) => Some(gpu),
            Ok(Err(e)) => {
                eprintln!("Skipping P3 test: no OpenCL device ({e})");
                None
            }
            Err(_) => {
                eprintln!("Skipping P3 test: OpenCL panic");
                None
            }
        }
    }

    /// Build a random symmetric Bsr4Matrix on a given mask with values in [-1, 1].
    fn random_symmetric_bsr4(
        n_atom: usize,
        mask: &(Vec<u32>, Vec<u32>),
        seed: u64,
    ) -> Bsr4Matrix {
        use std::collections::HashMap;
        let mut m = Bsr4Matrix::from_structure(
            n_atom, mask.0.clone(), mask.1.clone(),
        ).unwrap();
        // Simple LCG for reproducibility
        let mut state = seed;
        let next = |state: &mut u64| {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*state >> 33) as f64 / (1u64 << 33) as f64) * 2.0 - 1.0
        };
        // Fill diagonal blocks
        let mut block_map: HashMap<(usize, usize), [f32; BS2]> = HashMap::new();
        for i in 0..n_atom {
            let (start, end) = (mask.0[i] as usize, mask.0[i + 1] as usize);
            for blk in start..end {
                let j = mask.1[blk] as usize;
                if i == j {
                    let mut v = [0.0f32; BS2];
                    for k in 0..BS2 {
                        v[k] = next(&mut state) as f32;
                    }
                    // Make diagonal blocks symmetric
                    for r in 0..BS {
                        for c in (r + 1)..BS {
                            let avg = 0.5 * (v[r * BS + c] + v[c * BS + r]);
                            v[r * BS + c] = avg;
                            v[c * BS + r] = avg;
                        }
                    }
                    block_map.insert((i, j), v);
                }
            }
        }
        // Fill off-diagonal blocks (symmetric pairs)
        for i in 0..n_atom {
            let (start, end) = (mask.0[i] as usize, mask.0[i + 1] as usize);
            for blk in start..end {
                let j = mask.1[blk] as usize;
                if i < j {
                    let mut v = [0.0f32; BS2];
                    for k in 0..BS2 {
                        v[k] = next(&mut state) as f32;
                    }
                    block_map.insert((i, j), v);
                    // Transpose for (j, i)
                    let mut vt = [0.0f32; BS2];
                    for r in 0..BS {
                        for c in 0..BS {
                            vt[c * BS + r] = v[r * BS + c];
                        }
                    }
                    block_map.insert((j, i), vt);
                }
            }
        }
        // Write blocks
        for ((i, j), v) in &block_map {
            m.set_block(*i, *j, v).unwrap();
        }
        m
    }

    /// Test: D = 2K and W = 2KHK match dense reference.
    #[test]
    fn test_build_dw_sparse_vs_dense() {
        let Some(gpu) = try_gpu() else { return; };

        // Small system: 4 atoms, geometric mask with cutoff 3.0 Å
        let n_atom = 4;
        let pos: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let m_hs = build_geometric_mask(&pos, 3.0);
        // K mask = H/S mask (for this test; in production R_K may differ)
        let m_k = m_hs.clone();

        // Build random symmetric K and H_scc on M_K and M_HS
        let k_host = random_symmetric_bsr4(n_atom, &m_k, 42);
        let h_scc_host = random_symmetric_bsr4(n_atom, &m_hs, 123);

        // Dense reference (f64)
        let (d_dense, w_dense, n_orb) = build_dw_dense_reference(&k_host, &h_scc_host);

        // Sparse GPU path
        let k_dev = GpuBsrMatrix::from_host(&gpu, &k_host).unwrap();
        let h_scc_dev = GpuBsrMatrix::from_host(&gpu, &h_scc_host).unwrap();
        let dw = build_dw_sparse(&gpu, &k_dev, &h_scc_dev, &m_k, &m_hs).unwrap();

        // Read back D and W (blocking reads synchronize the queue)
        let d_sparse = dw.d.to_host(&gpu).unwrap();
        let w_sparse = dw.w.to_host(&gpu).unwrap();

        // Compare D: sparse (f32) vs dense (f64)
        let d_sparse_dense = d_sparse.to_dense();
        let mut max_err_d = 0.0f64;
        for i in 0..n_orb * n_orb {
            let err = (d_sparse_dense[i] as f64 - d_dense[i]).abs();
            max_err_d = max_err_d.max(err);
        }

        // Compare W: sparse (f32) vs dense (f64)
        let w_sparse_dense = w_sparse.to_dense();
        let mut max_err_w = 0.0f64;
        for i in 0..n_orb * n_orb {
            let err = (w_sparse_dense[i] as f64 - w_dense[i]).abs();
            max_err_w = max_err_w.max(err);
        }

        eprintln!("P3 parity: D max|err| = {max_err_d:.3e}, W max|err| = {max_err_w:.3e}");

        // f32 SpGEMM vs f64 dense: expect ~1e-4 relative error
        let d_scale = d_dense.iter().map(|x| x.abs()).fold(0.0f64, f64::max).max(1e-10);
        let w_scale = w_dense.iter().map(|x| x.abs()).fold(0.0f64, f64::max).max(1e-10);
        assert!(max_err_d / d_scale < 1e-4, "D error too large: {max_err_d:.3e} / {d_scale:.3e}");
        assert!(max_err_w / w_scale < 1e-4, "W error too large: {max_err_w:.3e} / {w_scale:.3e}");
    }

    /// Test: D and W are symmetric.
    #[test]
    fn test_dw_symmetry() {
        let Some(gpu) = try_gpu() else { return; };

        let n_atom = 3;
        let pos: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
        ];
        let m_hs = build_geometric_mask(&pos, 3.0);
        let m_k = m_hs.clone();

        let k_host = random_symmetric_bsr4(n_atom, &m_k, 99);
        let h_scc_host = random_symmetric_bsr4(n_atom, &m_hs, 88);

        let k_dev = GpuBsrMatrix::from_host(&gpu, &k_host).unwrap();
        let h_scc_dev = GpuBsrMatrix::from_host(&gpu, &h_scc_host).unwrap();
        let dw = build_dw_sparse(&gpu, &k_dev, &h_scc_dev, &m_k, &m_hs).unwrap();

        let d_host = dw.d.to_host(&gpu).unwrap();
        let w_host = dw.w.to_host(&gpu).unwrap();

        // Check symmetry: D[i,j] == D[j,i] for each block
        let check_sym = |m: &Bsr4Matrix, label: &str| {
            for i in 0..n_atom {
                let (s, e) = (m.row_ptr[i] as usize, m.row_ptr[i + 1] as usize);
                for blk in s..e {
                    let j = m.col_idx[blk] as usize;
                    if i == j { continue; }
                    let blk_ji = m.find(j, i).expect("symmetric block missing") as usize;
                    let v_ij = &m.values[blk * BS2..(blk + 1) * BS2];
                    let v_ji = &m.values[blk_ji * BS2..(blk_ji + 1) * BS2];
                    let mut max_diff = 0.0f32;
                    for r in 0..BS {
                        for c in 0..BS {
                            max_diff = max_diff.max((v_ij[r * BS + c] - v_ji[c * BS + r]).abs());
                        }
                    }
                    assert!(max_diff < 1e-4, "{label}: asymmetry at ({i},{j}): {max_diff:.3e}");
                }
            }
        };
        check_sym(&d_host, "D");
        check_sym(&w_host, "W");
    }
}
