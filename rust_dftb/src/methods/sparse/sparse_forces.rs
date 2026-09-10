//! Sparse analytic force support: `D = 2K` and `W = 2 K H_scc K` (P3, manifest v3 §4.4).
//!
//! Builds the density and energy-weighted density matrices needed by the
//! sparse force path **without diagonalization**. The density kernel `K`
//! comes from TC2 purification; `H_scc` is the SCC Hamiltonian on the H/S
//! mask. The products are masked SpGEMM operations on the GPU.
//!
//! G3.3 contracts those `D`/`W` with the **same CPU force components** as
//! dense DFTB (`methods/dftb/forces.rs`: `dH0`/`dS` Pulay, SCC shift, γ′,
//! F_rep). That CPU module is the formula SSOT for this path.
//!
//! ## Two force codepaths — do not merge yet
//!
//! The dense multi-system H-bond GPU kernels (`qmqm/gpu_forces.cl`,
//! `qmqm/gpu_forces.rs`) implement the same pair-force physics for the
//! fragment QM/QM solver. That work is **in progress and not fully verified**.
//! Treat it as **read-only** from this sparse nanocrystal path.
//!
//! Eventual goal: one GPU pair-force + atom-gather kernel (shared SK
//! interpolation, `D`/`W` layout) used by both H-bond batches and sparse
//! BSR4. **Wait until both codepaths are developed and well tested** so the
//! caveats of each (f32 vs f64, padded dummy orbitals, fragment vs BSR
//! indexing, SCC split) are known. Merging earlier would mix two unfinished
//! bugs.
//!
//! ## Algorithm (manifest v3 §4.4)
//!
//! ```text
//! M_TW = boolean_product(M_K, M_HS)   // exact for the truncated operands
//! T    = K * H_scc                    // on M_TW
//! W    = 2 * project_MHS(T * K)       // final short-range support
//! symmetrize(W)
//! D    = 2 * K                        // same mask as K
//! F    = 2 (D:dH0 − W:dS) + F_shift + F_γ′ + F_rep
//! ```
//!
//! `W` is built only on `M_HS` (the H/S support mask), never as a dense
//! matrix. This is the diagonalization-free route: no eigenvalues, no
//! eigenvectors, no dense EDM from an eigenproblem.
//!
//! ## Parity gate (manifest v3 §4.4 / review G3.3–G3.4)
//!
//! At the same geometry and same Hamiltonian, compare:
//!   - `D_sparse = 2K`  vs dense occupied-orbital D
//!   - `W_sparse = 2KHK` vs dense eigenvalue/eigenvector W
//!   - `F_sparse` vs dense analytic force
//!   - own `E_tot` finite-difference vs own analytic F (G3.4)
//!
//! Hessian implementation is blocked until this passes.

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::forces::{compute_forces_from_dw, Forces};
use crate::methods::dftb::sk_data::SkData;
use crate::methods::sparse::bsr4::{build_product_mask, build_spgemm_plan_bsym, Bsr4Matrix, SpgemmPlan, BS, BS2};
use crate::methods::sparse::gpu_sparse::{GpuBsrMatrix, GpuBsrStructure, SparseBsr4Gpu, SpgemmPlanGpu};
use nalgebra::DMatrix;
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

/// Persistent workspace for `D = 2K` and `W = 2 K H_scc K` construction.
///
/// GPT-5.6 issue #8 / manifest §1.4: all structures, buffers, and symbolic
/// plans are built once at construction and reused across all calls.
/// No allocation in the hot loop — `build_dw_into` only enqueues kernels
/// and swaps `set_arg`s.
///
/// The workspace owns:
/// - `tw_struct`: GPU structure for T = K·H_scc (on M_TW = M_K ∘ M_HS)
/// - `t`: scratch GPU matrix on M_TW
/// - `w`: GPU matrix on M_HS (the output W)
/// - `d`: GPU matrix on M_K (the output D)
/// - `plan_kh`: symbolic plan for T = K·H_scc (A=K, B=H_scc sym, C=T)
/// - `plan_tk`: symbolic plan for W = T·K (A=T, B=K sym, C=W on M_HS)
pub struct SparseDWWorkspace {
    /// GPU structure for T = K·H_scc on M_TW.
    tw_struct: Arc<GpuBsrStructure>,
    /// Scratch T = K·H_scc (on M_TW).
    t: GpuBsrMatrix,
    /// Output W = 2·T·K (on M_HS, reuses h_scc structure).
    w: GpuBsrMatrix,
    /// Output D = 2·K (on M_K, reuses k structure).
    d: GpuBsrMatrix,
    /// P4: symbolic plan for T = K·H_scc.
    plan_kh: Option<SpgemmPlanGpu>,
    /// P4: symbolic plan for W = T·K.
    plan_tk: Option<SpgemmPlanGpu>,
}

impl SparseDWWorkspace {
    /// Build a persistent workspace for D/W construction on the given masks.
    ///
    /// `k_struct` is the K mask structure (M_K), `h_struct` is the H/S mask
    /// structure (M_HS). Both must have the same `n_atom`.
    ///
    /// All GPU structures and buffers are allocated once here. The symbolic
    /// plans for the two SpGEMMs are built and uploaded. If plan building
    /// fails (e.g., row degree overflow), falls back to the intersection
    /// kernel.
    pub fn new(
        gpu: &SparseBsr4Gpu,
        k_struct: &Arc<GpuBsrStructure>,
        h_struct: &Arc<GpuBsrStructure>,
    ) -> Result<Self> {
        let n_atom = k_struct.n_atom;
        assert_eq!(h_struct.n_atom, n_atom, "K and H_scc must have same n_atom");

        // M_TW = boolean_product(M_K, M_HS) — host-side, one-time.
        let k_mask = (k_struct.row_ptr_host(gpu)?, k_struct.col_idx_host(gpu)?);
        let hs_mask = (h_struct.row_ptr_host(gpu)?, h_struct.col_idx_host(gpu)?);
        let m_tw = build_product_mask(n_atom, &k_mask, &hs_mask);

        // Allocate device structures and buffers once.
        let tw_struct = Arc::new(GpuBsrStructure::new(gpu, n_atom, &m_tw)?);
        let t = GpuBsrMatrix::zero(gpu, &tw_struct)?;
        let w = GpuBsrMatrix::zero(gpu, h_struct)?;
        let d = GpuBsrMatrix::zero(gpu, k_struct)?;

        // P4: Build symbolic plans for the two SpGEMMs.
        // Plan for T = K·H_scc: A=K (k_mask), B=H_scc (hs_mask, sym), C=T (m_tw).
        // Plan for W = T·K: A=T (m_tw), B=K (k_mask, sym), C=W (hs_mask).
        let plan_kh = {
            let k_dummy = Bsr4Matrix::from_structure(n_atom, k_mask.0.clone(), k_mask.1.clone())?;
            let h_dummy = Bsr4Matrix::from_structure(n_atom, hs_mask.0.clone(), hs_mask.1.clone())?;
            match build_spgemm_plan_bsym(&k_dummy, &h_dummy, &m_tw) {
                Ok(plan) => Some(gpu.upload_plan(&plan)?),
                Err(e) => {
                    eprintln!("P4: plan_kh build failed, falling back to intersection: {e}");
                    None
                }
            }
        };
        let plan_tk = {
            let t_dummy = Bsr4Matrix::from_structure(n_atom, m_tw.0.clone(), m_tw.1.clone())?;
            let k_dummy = Bsr4Matrix::from_structure(n_atom, k_mask.0.clone(), k_mask.1.clone())?;
            match build_spgemm_plan_bsym(&t_dummy, &k_dummy, &hs_mask) {
                Ok(plan) => Some(gpu.upload_plan(&plan)?),
                Err(e) => {
                    eprintln!("P4: plan_tk build failed, falling back to intersection: {e}");
                    None
                }
            }
        };

        Ok(Self { tw_struct, t, w, d, plan_kh, plan_tk })
    }

    /// Build `D = 2K` and `W = 2 K H_scc K` into the workspace's persistent
    /// buffers. No allocation, no host transfer — only kernel launches.
    ///
    /// Returns references to the device-resident D and W. The caller must
    /// not hold these references across another `build_dw_into` call
    /// (they point into the same workspace buffers).
    pub fn build_dw_into(
        &mut self,
        gpu: &SparseBsr4Gpu,
        k: &GpuBsrMatrix,
        h_scc: &GpuBsrMatrix,
    ) -> Result<(&GpuBsrMatrix, &GpuBsrMatrix)> {
        // 1. T = K * H_scc on M_TW (SpGEMM, device-resident).
        match &self.plan_kh {
            Some(plan) => gpu.spgemm_plan_bsym_dev(k, h_scc, plan, &self.t)?,
            None => gpu.spgemm_masked_dev(k, h_scc, &self.t)?,
        }

        // 2. W = T * K on M_HS (SpGEMM, device-resident).
        match &self.plan_tk {
            Some(plan) => gpu.spgemm_plan_bsym_dev(&self.t, k, plan, &self.w)?,
            None => gpu.spgemm_masked_dev(&self.t, k, &self.w)?,
        }

        // 3. Scale W by 2: W = 2 * project_MHS(T * K).
        let nblock_w = self.w.struct_.nblock;
        gpu.axpby_dev(nblock_w, 2.0, &self.w.values, 0.0, &self.w.values, &self.w.values)?;

        // 4. symmetrize(W).
        gpu.symmetrize_dev(nblock_w, self.w.struct_.transpose_block(), &self.w.values)?;

        // 5. D = 2 * K (scale K values by 2, same mask M_K).
        let nblock_k = self.d.struct_.nblock;
        gpu.axpby_dev(nblock_k, 2.0, &k.values, 0.0, &k.values, &self.d.values)?;

        Ok((&self.d, &self.w))
    }

    /// Access the D matrix (last computed).
    pub fn d(&self) -> &GpuBsrMatrix { &self.d }
    /// Access the W matrix (last computed).
    pub fn w(&self) -> &GpuBsrMatrix { &self.w }
}

/// Build `D = 2K` and `W = 2 K H_scc K` on the GPU using masked SpGEMM.
///
/// **Legacy one-shot API.** Allocates structures and buffers per call.
/// Production code should use `SparseDWWorkspace::build_dw_into` instead,
/// which pre-allocates once and reuses across all calls (GPT-5.6 #8).
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

/// Host `D = 2K`, `W = 2 K H_scc K` on the padded BSR4 dense layout.
///
/// G3.3 uses this (not the device NS inverse). Device SpGEMM `build_dw_sparse`
/// is the production-scale path once N4 is fixed.
pub fn dw_from_k_padded(k_pad: &[f32], h_scc_pad: &[f32], n_atom: usize) -> (Vec<f64>, Vec<f64>) {
    let n = n_atom * BS;
    assert_eq!(k_pad.len(), n * n, "dw_from_k_padded: K len {} != (n_atom*4)² {}", k_pad.len(), n * n);
    assert_eq!(h_scc_pad.len(), n * n, "dw_from_k_padded: H len {} != (n_atom*4)² {}", h_scc_pad.len(), n * n);
    let k: Vec<f64> = k_pad.iter().map(|&x| x as f64).collect();
    let h: Vec<f64> = h_scc_pad.iter().map(|&x| x as f64).collect();
    let d: Vec<f64> = k.iter().map(|x| 2.0 * x).collect();
    let mut t = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..n {
                s += k[i * n + p] * h[p * n + j];
            }
            t[i * n + j] = s;
        }
    }
    let mut w = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..n {
                s += t[i * n + p] * k[p * n + j];
            }
            w[i * n + j] = 2.0 * s;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (w[i * n + j] + w[j * n + i]);
            w[i * n + j] = avg;
            w[j * n + i] = avg;
        }
    }
    (d, w)
}

/// Copy physical orbital blocks out of a padded BSR4 dense matrix.
pub fn unpad_to_physical(padded: &[f64], atom_n_orb: &[u8]) -> DMatrix<f64> {
    let n_atom = atom_n_orb.len();
    let n_pad = n_atom * BS;
    let n_phys: usize = atom_n_orb.iter().map(|&n| n as usize).sum();
    assert_eq!(padded.len(), n_pad * n_pad, "unpad_to_physical: len {} != n_pad² {}", padded.len(), n_pad * n_pad);
    let mut phys_off = vec![0usize; n_atom];
    let mut acc = 0usize;
    for a in 0..n_atom {
        phys_off[a] = acc;
        acc += atom_n_orb[a] as usize;
    }
    let mut m = DMatrix::<f64>::zeros(n_phys, n_phys);
    for a in 0..n_atom {
        let na = atom_n_orb[a] as usize;
        for b in 0..n_atom {
            let nb = atom_n_orb[b] as usize;
            for i in 0..na {
                for j in 0..nb {
                    let pi = a * BS + i;
                    let pj = b * BS + j;
                    m[(phys_off[a] + i, phys_off[b] + j)] = padded[pi * n_pad + pj];
                }
            }
        }
    }
    m
}

/// Analytic force of the sparse SCC energy: `D = 2K`, `W = 2 K H_scc K`.
///
/// Contracts with CPU `compute_forces_from_dw` (read-only use of the dense
/// force formulas). Does not call `qmqm/gpu_forces`.
pub fn sparse_analytic_forces(
    sk: &SkData,
    species: &[String],
    coords: &[[f64; 3]],
    atom_n_orb: &[u8],
    k_pad: &[f32],
    h_scc_pad: &[f32],
    q: &[f64],
    q0: &[f64],
    sk_dir: &str,
) -> Result<Forces> {
    let n_atom = atom_n_orb.len();
    if k_pad.is_empty() || h_scc_pad.is_empty() {
        return Err(DftbError::InvalidInput(
            "sparse_analytic_forces: empty K or H_scc — run_sparse_scc must store k_pad/h_scc_pad".into(),
        ));
    }
    let n_pad = n_atom * BS;
    let (d_pad, w_pad) = dw_from_k_padded(k_pad, h_scc_pad, n_atom);
    let mut dummy_d = 0.0f64;
    for (a, &nphys) in atom_n_orb.iter().enumerate() {
        for d in (nphys as usize)..BS {
            let idx = a * BS + d;
            dummy_d += d_pad[idx * n_pad + idx].abs();
        }
    }
    if dummy_d > 1e-6 {
        return Err(DftbError::InvalidInput(format!(
            "sparse_analytic_forces: dummy |D_ii| sum={dummy_d:.3e} > 1e-6 (dummy orbitals occupied)"
        )));
    }
    let dm = unpad_to_physical(&d_pad, atom_n_orb);
    let edm = unpad_to_physical(&w_pad, atom_n_orb);
    let forces = compute_forces_from_dw(sk, species, coords, &dm, &edm, q, q0, sk_dir)?;
    let mut sum = [0.0f64; 3];
    for f in &forces.forces {
        sum[0] += f[0];
        sum[1] += f[1];
        sum[2] += f[2];
    }
    let newton = sum.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if newton > 1e-5 {
        panic!(
            "sparse_analytic_forces: Newton's third law |ΣF|={newton:.3e} (sum={sum:?}) n_atom={n_atom}"
        );
    }
    Ok(forces)
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
    use crate::methods::sparse::harness::require_sparse_gpu;

    fn try_gpu() -> Option<SparseBsr4Gpu> {
        require_sparse_gpu()
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

    /// Test: SparseDWWorkspace produces the same D and W as the one-shot
    /// build_dw_sparse, verifying that the persistent workspace + symbolic
    /// plans give identical results to the per-call allocation path.
    #[test]
    fn test_dw_workspace_vs_oneshot() {
        let Some(gpu) = try_gpu() else { return; };

        let n_atom = 4;
        let pos: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let m_hs = build_geometric_mask(&pos, 3.0);
        let m_k = m_hs.clone();

        let k_host = random_symmetric_bsr4(n_atom, &m_k, 42);
        let h_scc_host = random_symmetric_bsr4(n_atom, &m_hs, 123);

        // One-shot path
        let k_dev = GpuBsrMatrix::from_host(&gpu, &k_host).unwrap();
        let h_scc_dev = GpuBsrMatrix::from_host(&gpu, &h_scc_host).unwrap();
        let dw_oneshot = build_dw_sparse(&gpu, &k_dev, &h_scc_dev, &m_k, &m_hs).unwrap();
        let d_oneshot = dw_oneshot.d.to_host(&gpu).unwrap();
        let w_oneshot = dw_oneshot.w.to_host(&gpu).unwrap();

        // Persistent workspace path
        let mut ws = SparseDWWorkspace::new(&gpu, &k_dev.struct_, &h_scc_dev.struct_).unwrap();
        let (d_ws_ref, w_ws_ref) = ws.build_dw_into(&gpu, &k_dev, &h_scc_dev).unwrap();
        let d_ws = d_ws_ref.to_host(&gpu).unwrap();
        let w_ws = w_ws_ref.to_host(&gpu).unwrap();

        // Compare: should be identical (same kernels, same order, just
        // different buffer ownership).
        let d_diff = bsr4_max_abs_diff(&d_oneshot, &d_ws);
        let w_diff = bsr4_max_abs_diff(&w_oneshot, &w_ws);
        eprintln!("DW workspace vs oneshot: D max|diff|={d_diff:e}, W max|diff|={w_diff:e}");
        assert!(d_diff < 1e-6, "D mismatch: {d_diff:e}");
        assert!(w_diff < 1e-6, "W mismatch: {w_diff:e}");

        // Verify workspace is reusable: second call should give same result.
        let (d2_ref, w2_ref) = ws.build_dw_into(&gpu, &k_dev, &h_scc_dev).unwrap();
        let d2 = d2_ref.to_host(&gpu).unwrap();
        let w2 = w2_ref.to_host(&gpu).unwrap();
        let d2_diff = bsr4_max_abs_diff(&d_oneshot, &d2);
        let w2_diff = bsr4_max_abs_diff(&w_oneshot, &w2);
        eprintln!("DW workspace reuse: D max|diff|={d2_diff:e}, W max|diff|={w2_diff:e}");
        assert!(d2_diff < 1e-6, "D reuse mismatch: {d2_diff:e}");
        assert!(w2_diff < 1e-6, "W reuse mismatch: {w2_diff:e}");
    }
}
