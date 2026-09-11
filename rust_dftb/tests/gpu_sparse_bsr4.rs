//! GPU tests for the BSR4 sparse purification kernels
//! (`methods::sparse`), independent of the `qmqm` dense/fragment solver.
//!
//! These tests validate the OpenCL kernels in `sparse_bsr4_purification.cl`
//! against dense CPU references (nalgebra):
//!
//!   1. `bsr4_spgemm_masked`        — generic masked SpGEMM vs dense matmul
//!   2. `bsr4_spgemm_masked_Bsym`   — symmetric-right SpGEMM vs dense matmul
//!   3. masked truncation           — only masked blocks are computed
//!   4. `bsr4_trace_KS_partial`     — Tr(KS) vs CPU
//!   5. `bsr4_symmetrize`           — vs host symmetrize reference
//!   6. `bsr4_mulliken_KS`          — Mulliken charges vs CPU
//!   7. `bsr4_idempotency_partial`  — ||KSK-K||_F for an exact idempotent K
//!   8. TC2 / McWeeny purification  — convergence from a perturbed K0
//!
//! OpenCL Err is a test failure. Skip only if the machine has zero platforms.

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::methods::sparse::bsr4::{
    build_full_mask, build_geometric_mask, build_identity, build_product_mask,
    dense_matmul, dense_max_abs_diff, dense_frobenius, diag_block_map,
    gershgorin_bounds, inf_norm, symmetrize_host, Bsr4Matrix, BS, BS2,
};
use rust_dftb::methods::sparse::gpu_sparse::{SparseBsr4Config, SparseBsr4Gpu, SparsePurifyWorkspace};
use rust_dftb::methods::sparse::gpu_sparse;
use rust_dftb::methods::sparse::harness::require_sparse_gpu;

/// Deterministic LCG for reproducible random-ish data.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift64
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        // map to [-1, 1]
        ((x as i64 as f64) / (i64::MAX as f64)) as f32
    }
}

fn try_gpu() -> Option<SparseBsr4Gpu> {
    require_sparse_gpu()
}

/// Build a Bsr4Matrix from a dense (4N)x(4N) row-major f32 matrix using the
/// given mask. Blocks outside the mask are dropped.
fn bsr4_from_dense(n_atom: usize, dense: &[f32], mask: &(Vec<u32>, Vec<u32>)) -> Bsr4Matrix {
    let n = n_atom * BS;
    let mut m = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
    for i in 0..n_atom {
        let (a, b) = (mask.0[i] as usize, mask.0[i + 1] as usize);
        for blk in a..b {
            let j = mask.1[blk] as usize;
            let mut v = [0.0f32; BS2];
            for r in 0..BS {
                for c in 0..BS {
                    v[r * BS + c] = dense[(i * BS + r) * n + (j * BS + c)];
                }
            }
            m.set_block(i, j, &v).unwrap();
        }
    }
    m
}

/// Make a dense symmetric matrix from a full BSR4 mask + random blocks.
fn random_symmetric_dense(n_atom: usize, rng: &mut Rng, scale: f32) -> Vec<f32> {
    let n = n_atom * BS;
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            let v = scale * rng.next();
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    d
}

/// Build a symmetric positive-definite S: S = I + small symmetric offdiag.
fn make_overlap_dense(n_atom: usize, rng: &mut Rng, offdiag: f32) -> Vec<f32> {
    let n = n_atom * BS;
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        d[i * n + i] = 1.0;
    }
    for i in 0..n {
        for j in i + 1..n {
            let v = offdiag * rng.next() * 0.5;
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    d
}

/// CPU generalized eigensolve H c = S c eps. Returns spinless density kernel
/// K = sum_{occ} c_i c_i^T (row-major f32) and the occupied count.
fn cpu_density_kernel(h: &[f32], s: &[f32], n: usize, nocc: usize) -> Vec<f32> {
    let hf = row_major_to_dmatrix_f64(h, n);
    let sf = row_major_to_dmatrix_f64(s, n);
    // S^{-1/2} via eigendecomposition.
    let se = SymmetricEigen::new(sf.clone());
    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = 1.0 / se.eigenvalues[i].max(1e-12).sqrt();
    }
    let s_inv_sqrt = &se.eigenvectors * &d * se.eigenvectors.transpose();
    // H_orth = S^{-1/2} H S^{-1/2}
    let h_orth = &s_inv_sqrt * &hf * &s_inv_sqrt;
    let he = SymmetricEigen::new(h_orth);
    // Sort ascending.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| he.eigenvalues[i].partial_cmp(&he.eigenvalues[j]).unwrap());
    // C = S^{-1/2} * V_orth (columns reordered).
    let v_sorted = he.eigenvectors.select_columns(&idx);
    let c = &s_inv_sqrt * &v_sorted;
    // K = sum_{i in occ} c[:,i] c[:,i]^T  (spinless)
    let mut k = DMatrix::<f64>::zeros(n, n);
    for i in 0..nocc {
        let col = c.column(i);
        k += &col * col.transpose();
    }
    dmatrix_to_row_major_f32(&k)
}

fn row_major_to_dmatrix_f64(data: &[f32], n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = data[i * n + j] as f64;
        }
    }
    m
}

fn dmatrix_to_row_major_f32(m: &DMatrix<f64>) -> Vec<f32> {
    let n = m.nrows();
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = m[(i, j)] as f32;
        }
    }
    out
}

fn dense_trace(a: &[f32], n: usize) -> f32 {
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Independent host reference for the two values returned by resident TC2.
/// Keeping this in nalgebra avoids validating the GPU trace and residual with
/// the same sparse helper implementation that produced them.
fn reference_trace_and_idempotency(k: &[f32], s: &[f32], n: usize) -> (f32, f32) {
    let km = row_major_to_dmatrix_f64(k, n);
    let sm = row_major_to_dmatrix_f64(s, n);
    let trace = (&km * &sm).trace() as f32;
    let residual = ((&km * &sm * &km) - &km).norm() as f32;
    (trace, residual)
}

// (dense_frobenius is now imported from bsr4.rs)

// =====================================================================
// 1. Generic masked SpGEMM vs dense
// =====================================================================

#[test]
fn test_spgemm_masked_vs_dense() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 4;
    let n = n_atom * BS;
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let a_dense = random_symmetric_dense(n_atom, &mut rng, 1.0);
    // B not necessarily symmetric for the generic kernel.
    let mut b_dense = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            b_dense[i * n + j] = 0.5 * rng.next();
        }
    }
    let mask = build_full_mask(n_atom);
    let a = bsr4_from_dense(n_atom, &a_dense, &mask);
    let b = bsr4_from_dense(n_atom, &b_dense, &mask);

    let c = gpu.matmul_masked(&a, &b, &mask).unwrap();
    let c_ref = dense_matmul(n, &a_dense, &b_dense);
    let err = gpu_sparse::compare_to_dense(&c, &c_ref);
    println!("spgemm_masked (n_atom={n_atom}): max|dC| = {err:e}");
    assert!(err < 1e-4, "generic SpGEMM parity failed: {err:e}");
}

// =====================================================================
// 2. Symmetric-right SpGEMM vs dense
// =====================================================================

#[test]
fn test_spgemm_masked_bsym_vs_dense() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 4;
    let n = n_atom * BS;
    let mut rng = Rng(0xa1b2_c3d4_e5f6_0718);
    let a_dense = random_symmetric_dense(n_atom, &mut rng, 1.0);
    let b_dense = random_symmetric_dense(n_atom, &mut rng, 0.7); // symmetric B
    let mask = build_full_mask(n_atom);
    let a = bsr4_from_dense(n_atom, &a_dense, &mask);
    let b = bsr4_from_dense(n_atom, &b_dense, &mask);

    let c = gpu.matmul_masked_bsym(&a, &b, &mask).unwrap();
    let c_ref = dense_matmul(n, &a_dense, &b_dense);
    let err = gpu_sparse::compare_to_dense(&c, &c_ref);
    println!("spgemm_masked_Bsym (n_atom={n_atom}): max|dC| = {err:e}");
    assert!(err < 1e-4, "symmetric-right SpGEMM parity failed: {err:e}");
}

// =====================================================================
// 3. Masked truncation: only masked blocks computed
// =====================================================================

#[test]
fn test_spgemm_mask_truncation() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 4;
    let n = n_atom * BS;
    // Place atoms on a line with spacing 1.5; cutoff 2.0 -> only nearest
    // neighbors + self are in the mask.
    let pos: Vec<[f64; 3]> = (0..n_atom)
        .map(|i| [1.5 * i as f64, 0.0, 0.0])
        .collect();
    let mask = build_geometric_mask(&pos, 2.0);
    let mut rng = Rng(0x55aa_55aa_55aa_55aa);
    let a_dense = random_symmetric_dense(n_atom, &mut rng, 1.0);
    let b_dense = random_symmetric_dense(n_atom, &mut rng, 0.8);
    let a = bsr4_from_dense(n_atom, &a_dense, &mask);
    let b = bsr4_from_dense(n_atom, &b_dense, &mask);

    let c = gpu.matmul_masked_bsym(&a, &b, &mask).unwrap();
    // Reference: dense product of the MASK-PROJECTED A and B (blocks outside
    // the mask are zero in the BSR4 representation), then projected onto the
    // output mask. Using the raw full dense matrices would include k that the
    // GPU kernel never sees.
    let a_proj = a.to_dense();
    let b_proj = b.to_dense();
    let c_full = dense_matmul(n, &a_proj, &b_proj);
    let c_ref = bsr4_from_dense(n_atom, &c_full, &mask);
    let err = gpu_sparse::compare_to_dense(&c, &c_ref.to_dense());
    // Count blocks in mask.
    let nblock = mask.1.len();
    println!(
        "spgemm truncation (n_atom={n_atom}, nblock={nblock}/{}): max|dC| = {err:e}",
        n_atom * n_atom
    );
    assert!(nblock < n_atom * n_atom, "mask should be sparse");
    assert!(err < 1e-4, "masked truncation parity failed: {err:e}");
}

// =====================================================================
// 4. Tr(KS) vs CPU
// =====================================================================

#[test]
fn test_trace_ks_vs_cpu() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 4;
    let n = n_atom * BS;
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let k_dense = random_symmetric_dense(n_atom, &mut rng, 1.0);
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.3);
    let mask = build_full_mask(n_atom);
    let k = bsr4_from_dense(n_atom, &k_dense, &mask);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // T = K·S (symmetric right).
    let t = gpu.matmul_masked_bsym(&k, &s, &mask).unwrap();
    let diag = diag_block_map(&t).unwrap();
    let diag_buf = gpu.buf_u32(&diag).unwrap();
    let t_buf = gpu.buf_f32(&t.values).unwrap();
    let n_orb_buf = gpu.buf_u32(&vec![4u32; n_atom]).unwrap();
    let tr_gpu = gpu.trace_ks(n_atom, &diag_buf, &t_buf, &n_orb_buf).unwrap();

    let t_ref = dense_matmul(n, &k_dense, &s_dense);
    let tr_cpu = dense_trace(&t_ref, n);
    let err = (tr_gpu - tr_cpu).abs();
    println!("trace_KS: GPU={tr_gpu:.6} CPU={tr_cpu:.6} |d|={err:e}");
    assert!(err < 1e-3, "Tr(KS) parity failed: {err:e}");
}

// =====================================================================
// 5. Symmetrize vs host reference
// =====================================================================

#[test]
fn test_symmetrize_vs_host() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let mut rng = Rng(0xfeed_face_dead_beef);
    let mask = build_full_mask(n_atom);
    // Asymmetric dense matrix.
    let n = n_atom * BS;
    let mut a_dense = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            a_dense[i * n + j] = 0.3 * rng.next();
        }
    }
    let a = bsr4_from_dense(n_atom, &a_dense, &mask);

    // GPU symmetrize.
    let a_gpu_sym = gpu.symmetrize_mat(&a).unwrap();
    // Host symmetrize on a copy.
    let mut a_host = a.clone();
    symmetrize_host(&mut a_host).unwrap();
    let err = dense_max_abs_diff(&a_gpu_sym.to_dense(), &a_host.to_dense());
    println!("symmetrize: max|dA| = {err:e}");
    assert!(err < 1e-5, "symmetrize parity failed: {err:e}");

    // Verify symmetry of the result: A == A^T.
    let d = a_gpu_sym.to_dense();
    let mut sym_err = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            sym_err = sym_err.max((d[i * n + j] - d[j * n + i]).abs());
        }
    }
    assert!(sym_err < 1e-5, "result not symmetric: {sym_err:e}");
}

// =====================================================================
// 6. Mulliken charges vs CPU
// =====================================================================

#[test]
fn test_mulliken_vs_cpu() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 4;
    let n = n_atom * BS;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let k_dense = random_symmetric_dense(n_atom, &mut rng, 1.0);
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.3);
    let mask = build_full_mask(n_atom);
    let k = bsr4_from_dense(n_atom, &k_dense, &mask);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    let ks = gpu.matmul_masked_bsym(&k, &s, &mask).unwrap();
    let (q_gpu, _q_dum) = gpu.mulliken(&ks, &vec![4u8; n_atom]).unwrap();

    // CPU: q_A = 2 * sum_{mu in A} (KS)[mu,mu]
    let ks_ref = dense_matmul(n, &k_dense, &s_dense);
    let mut q_cpu = vec![0.0f32; n_atom];
    for a in 0..n_atom {
        let mut tr = 0.0f32;
        for r in 0..BS {
            let mu = a * BS + r;
            tr += ks_ref[mu * n + mu];
        }
        q_cpu[a] = 2.0 * tr;
    }
    let err = q_gpu
        .iter()
        .zip(q_cpu.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!("mulliken: q_gpu={q_gpu:?} q_cpu={q_cpu:?} max|dq|={err:e}");
    assert!(err < 1e-4, "Mulliken parity failed: {err:e}");
}

// =====================================================================
// 7. Idempotency of an exact density kernel
//    K from CPU generalized eigensolve -> KSK should equal K (full mask,
//    no truncation error). Validates ksk + idempotency kernels.
// =====================================================================

#[test]
fn test_idempotency_exact_kernel() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let nocc = 3; // spinless occupied orbitals
    let mut rng = Rng(0xc0de_1234_5678_abcd);
    // H with a clear gap: diagonal dominant.
    let mut h_dense = random_symmetric_dense(n_atom, &mut rng, 0.4);
    for i in 0..n {
        h_dense[i * n + i] += 2.0;
    }
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.25);
    let k_dense = cpu_density_kernel(&h_dense, &s_dense, n, nocc);

    let mask = build_full_mask(n_atom);
    let k = bsr4_from_dense(n_atom, &k_dense, &mask);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    let (_t, q) = gpu.ksk(&k, &s, &mask, &mask).unwrap();
    // ||KSK - K||_F
    let q_buf = gpu.buf_f32(&q.values).unwrap();
    let k_buf = gpu.buf_f32(&k.values).unwrap();
    let idem_gpu = gpu.idempotency_err(k.nblock(), &q_buf, &k_buf).unwrap();

    // CPU reference.
    let ks = dense_matmul(n, &k_dense, &s_dense);
    let ksk = dense_matmul(n, &ks, &k_dense);
    let mut diff = vec![0.0f32; n * n];
    for i in 0..n * n {
        diff[i] = ksk[i] - k_dense[i];
    }
    let idem_cpu = dense_frobenius(&diff);
    println!("idempotency exact K: ||KSK-K||_F GPU={idem_gpu:e} CPU={idem_cpu:e}");
    assert!(idem_gpu < 5e-4, "exact K not idempotent on GPU: {idem_gpu:e}");
    assert!((idem_gpu - idem_cpu).abs() < 5e-4, "idempotency mismatch GPU vs CPU");

    // Also check Tr(KS) == Nocc.
    let t = gpu.matmul_masked_bsym(&k, &s, &mask).unwrap();
    let diag = diag_block_map(&t).unwrap();
    let diag_buf = gpu.buf_u32(&diag).unwrap();
    let t_buf = gpu.buf_f32(&t.values).unwrap();
    let n_orb_buf = gpu.buf_u32(&vec![4u32; n_atom]).unwrap();
    let tr = gpu.trace_ks(n_atom, &diag_buf, &t_buf, &n_orb_buf).unwrap();
    println!("Tr(KS) = {tr:.6} (expected Nocc = {nocc})");
    assert!((tr - nocc as f32).abs() < 1e-3, "Tr(KS) != Nocc: {tr} vs {nocc}");
}

// =====================================================================
// 8. McWeeny purification convergence from a perturbed K0
//    Start from K0 = exact K + noise (non-idempotent), run McWeeny steps,
//    verify ||KSK-K||_F decreases monotonically.
// =====================================================================

#[test]
fn test_mcweeny_convergence() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let nocc = 3;
    let mut rng = Rng(0x51de_0bad_f00d_1234);
    let mut h_dense = random_symmetric_dense(n_atom, &mut rng, 0.4);
    for i in 0..n {
        h_dense[i * n + i] += 2.0;
    }
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.25);
    let k_exact_dense = cpu_density_kernel(&h_dense, &s_dense, n, nocc);
    let mask = build_full_mask(n_atom);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // Perturb: K0 = K_exact + 0.05 * symmetric noise, then re-symmetrize.
    // McWeeny is quadratically convergent once inside the basin; a small
    // perturbation keeps K0 safely in the basin.
    let noise = random_symmetric_dense(n_atom, &mut rng, 0.05);
    let mut k0_dense = vec![0.0f32; n * n];
    for i in 0..n * n {
        k0_dense[i] = k_exact_dense[i] + noise[i];
    }
    let mut k = bsr4_from_dense(n_atom, &k0_dense, &mask);
    k = gpu.symmetrize_mat(&k).unwrap();

    let mut prev_idem = f32::INFINITY;
    for step in 0..20 {
        let (_t, q) = gpu.ksk(&k, &s, &mask, &mask).unwrap();
        let q_buf = gpu.buf_f32(&q.values).unwrap();
        let k_buf = gpu.buf_f32(&k.values).unwrap();
        let idem = gpu.idempotency_err(k.nblock(), &q_buf, &k_buf).unwrap();
        println!("McWeeny step {step}: ||KSK-K||_F = {idem:e}");
        if step > 0 {
            assert!(
                idem <= prev_idem * 1.02 + 1e-7,
                "McWeeny not decreasing at step {step}: {idem:e} vs {prev_idem:e}"
            );
        }
        prev_idem = idem;
        if idem < 1e-6 {
            break;
        }
        k = gpu.mcweeny_step(&k, &s, &mask, &mask).unwrap();
        k = gpu.symmetrize_mat(&k).unwrap();
    }
    println!("McWeeny final ||KSK-K||_F = {prev_idem:e}");
    assert!(prev_idem < 1e-4, "McWeeny did not converge: {prev_idem:e}");
}

// =====================================================================
// 9. TC2 purification keeps trace at Nocc and converges
// =====================================================================

#[test]
fn test_tc2_convergence() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let nocc: f32 = 3.0;
    let mut rng = Rng(0x7e57_c0de_face_cafe);
    let mut h_dense = random_symmetric_dense(n_atom, &mut rng, 0.4);
    for i in 0..n {
        h_dense[i * n + i] += 2.0;
    }
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.25);
    let k_exact_dense = cpu_density_kernel(&h_dense, &s_dense, n, nocc as usize);
    let mask = build_full_mask(n_atom);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // diag_block for the T mask (same as K mask here, full).
    let t_dummy = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
    let diag = diag_block_map(&t_dummy).unwrap();
    let diag_buf = gpu.buf_u32(&diag).unwrap();

    // TC2 requires K0 to already be a spectral function of the generalized
    // eigenproblem (the chat doc warns: "TC2 does not magically discover the
    // Hamiltonian eigenvectors"). A spectrally valid perturbation is to scale
    // the exact projector: K0 = alpha * K_exact, whose KS-eigenvalues are
    // {alpha, 0} subset [0,1]. TC2 should push alpha -> 1.
    let alpha = 0.8f32;
    let mut k0_dense = vec![0.0f32; n * n];
    for i in 0..n * n {
        k0_dense[i] = alpha * k_exact_dense[i];
    }
    let mut k = bsr4_from_dense(n_atom, &k0_dense, &mask);
    k = gpu.symmetrize_mat(&k).unwrap();

    let mut prev_idem = f32::INFINITY;
    let mut last_tr = f32::NAN;
    for step in 0..30 {
        let (_t, q) = gpu.ksk(&k, &s, &mask, &mask).unwrap();
        let q_buf = gpu.buf_f32(&q.values).unwrap();
        let k_buf = gpu.buf_f32(&k.values).unwrap();
        let idem = gpu.idempotency_err(k.nblock(), &q_buf, &k_buf).unwrap();
        // trace of KS via T = K·S
        let t = gpu.matmul_masked_bsym(&k, &s, &mask).unwrap();
        let t_buf = gpu.buf_f32(&t.values).unwrap();
        let n_orb_buf = gpu.buf_u32(&vec![4u32; n_atom]).unwrap();
        let tr = gpu.trace_ks(n_atom, &diag_buf, &t_buf, &n_orb_buf).unwrap();
        println!("TC2 step {step}: ||KSK-K||_F={idem:e}  Tr(KS)={tr:.5} (Nocc={nocc})");
        // Trace must stay bounded and in [0, 2*Nocc].
        assert!(tr >= -1.0 && tr <= 2.0 * nocc, "TC2 trace out of bounds at step {step}: {tr}");
        prev_idem = idem;
        last_tr = tr;
        if idem < 1e-5 {
            break;
        }
        let (knew, _n) = gpu.tc2_step(&k, &s, nocc, &mask, &mask, &diag_buf, &n_orb_buf).unwrap();
        k = gpu.symmetrize_mat(&knew).unwrap();
    }
    println!("TC2 final ||KSK-K||_F = {prev_idem:e}  Tr(KS)={last_tr:.5}");
    assert!(prev_idem < 1e-5, "TC2 did not converge: {prev_idem:e} (measured ~4e-6; G2)");
    assert!((last_tr - nocc).abs() < 1e-5, "TC2 trace != Nocc: {last_tr} vs {nocc} (G2)");
}

// =====================================================================
// 10. Boolean product mask M_T = M_K ∘ M_HS
//     Verify it contains exactly the structurally possible support of K·S,
//     and that omitting a required block is detectable.
// =====================================================================

#[test]
fn test_boolean_product_mask() {
    let n_atom = 5;
    // Place atoms on a line, spacing 1.5.
    let pos: Vec<[f64; 3]> = (0..n_atom)
        .map(|i| [1.5 * i as f64, 0.0, 0.0])
        .collect();
    // M_HS: cutoff 2.0 -> self + nearest neighbors.
    let s_mask = build_geometric_mask(&pos, 2.0);
    // M_K: cutoff 3.5 -> self + 2 nearest neighbors.
    let k_mask = build_geometric_mask(&pos, 3.5);
    let t_mask = build_product_mask(n_atom, &k_mask, &s_mask);

    // Reference: dense boolean product computed on host from the dense
    // adjacency of K and S.
    let k_adj = adjacency_dense(n_atom, &k_mask);
    let s_adj = adjacency_dense(n_atom, &s_mask);
    let mut t_ref = vec![false; n_atom * n_atom];
    for i in 0..n_atom {
        for k in 0..n_atom {
            if !k_adj[i * n_atom + k] {
                continue;
            }
            for j in 0..n_atom {
                if s_adj[k * n_atom + j] {
                    t_ref[i * n_atom + j] = true;
                }
            }
        }
    }
    // Compare t_mask against t_ref.
    let t_adj = adjacency_dense(n_atom, &t_mask);
    let mut mismatches = 0;
    for i in 0..n_atom {
        for j in 0..n_atom {
            if t_adj[i * n_atom + j] != t_ref[i * n_atom + j] {
                mismatches += 1;
                println!("  mismatch ({i},{j}): mask={} ref={}", t_adj[i * n_atom + j], t_ref[i * n_atom + j]);
            }
        }
    }
    println!(
        "boolean product mask: |M_K|={} |M_HS|={} |M_T|={} mismatches={mismatches}",
        k_mask.1.len(),
        s_mask.1.len(),
        t_mask.1.len()
    );
    assert_eq!(mismatches, 0, "boolean product mask incorrect");

    // Omission detection: drop one structurally-required block from a copy
    // of t_mask and verify the SpGEMM result differs from the full-mask
    // reference.
    let Some(gpu) = try_gpu() else { return };
    let n = n_atom * BS;
    let mut rng = Rng(0xbad_cafe_dead_beef);
    let k_dense = random_symmetric_dense(n_atom, &mut rng, 1.0);
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.3);
    let k = bsr4_from_dense(n_atom, &k_dense, &k_mask);
    let s = bsr4_from_dense(n_atom, &s_dense, &s_mask);

    // Full T mask result.
    let t_full = gpu.matmul_masked_bsym(&k, &s, &t_mask).unwrap();
    // Truncated T mask: remove the last block of row 0 (if it's not the only
    // block). This drops a structurally possible (0, j) contribution.
    let mut t_trunc_row = t_mask.0.clone();
    let mut t_trunc_col = t_mask.1.clone();
    let b0 = t_mask.0[0] as usize;
    let b1 = t_mask.0[1] as usize;
    if b1 - b0 > 1 {
        // Remove the last block of row 0 by shifting col_idx and row_ptr.
        let removed = t_trunc_col.remove(b1 - 1);
        for r in 1..=n_atom {
            t_trunc_row[r] -= 1;
        }
        println!("  omitted block (0,{removed}) from M_T");
        let t_trunc = (t_trunc_row, t_trunc_col);
        let t_trunc_res = gpu.matmul_masked_bsym(&k, &s, &t_trunc).unwrap();
        // Each output block is computed independently, so blocks present in
        // both masks are identical. The omission is detectable only in the
        // MISSING block: the full result has a nonzero block at (0, removed)
        // that the truncated result lacks entirely.
        let full_block = t_full.block(0, removed as usize).unwrap_or([0.0; BS2]);
        let full_norm = dense_frobenius(&full_block);
        println!("  omitted block (0,{removed}) full norm = {full_norm:e}");
        // The truncated result should NOT have this block.
        assert!(t_trunc_res.find(0, removed as usize).is_none(),
            "truncated mask still contains omitted block");
        // And the full result should have a nonzero block there (otherwise
        // the omission is physically irrelevant for this data).
        assert!(full_norm > 1e-6,
            "omitted block is zero in full result — test data doesn't exercise the omission (full_norm={full_norm:e})");
        let max_diff = full_norm; // the "diff" is the entire missing block
        println!("  omission detected: missing block norm = {max_diff:e}");
        assert!(max_diff > 1e-6, "omission not detected (max_diff={max_diff:e})");
    }
}

fn adjacency_dense(n_atom: usize, mask: &(Vec<u32>, Vec<u32>)) -> Vec<bool> {
    let mut adj = vec![false; n_atom * n_atom];
    for i in 0..n_atom {
        let (a, b) = (mask.0[i] as usize, mask.0[i + 1] as usize);
        for blk in a..b {
            adj[i * n_atom + mask.1[blk] as usize] = true;
        }
    }
    adj
}

// =====================================================================
// 11. Newton-Schulz sparse inverse Z ≈ S⁻¹
//     Verify ||I - ZS||_F / sqrt(N) decreases and Z matches dense S⁻¹.
// =====================================================================

#[test]
fn test_newton_schulz_inverse() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let mut rng = Rng(0x5a5a_5a5a_5a5a_5a5a);
    // Well-conditioned S: I + small offdiag.
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.15);
    let mask = build_full_mask(n_atom);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    let (z, rz, iters) = gpu
        .newton_schulz_inverse(&s, &mask, &mask, 30, 1e-4, 3)
        .unwrap();
    println!("Newton-Schulz: {iters} iters, R_Z = {rz:e}");

    // CPU reference: dense S⁻¹ via eigendecomposition.
    let s_f64 = row_major_to_dmatrix_f64(&s_dense, n);
    let se = SymmetricEigen::new(s_f64.clone());
    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = 1.0 / se.eigenvalues[i].max(1e-12);
    }
    let s_inv = &se.eigenvectors * &d * se.eigenvectors.transpose();
    let s_inv_dense = dmatrix_to_row_major_f32(&s_inv);

    let z_dense = z.to_dense();
    let err = dense_max_abs_diff(&z_dense, &s_inv_dense);
    println!("  ||Z - S⁻¹||_max = {err:e}");
    assert!(rz < 1e-3, "Newton-Schulz did not converge: R_Z = {rz:e}");
    assert!(err < 1e-4, "Z != S⁻¹: max|dZ| = {err:e} (host NS measured 1.1e-5; G2)");
}

// =====================================================================
// 11b. Device-resident Newton-Schulz (P0-D: GPU residency)
//      Same test as above but using the device-resident path.
// =====================================================================

#[test]
fn test_newton_schulz_inverse_dev() {
    use rust_dftb::methods::sparse::gpu_sparse::{GpuBsrMatrix, GpuBsrStructure};
    use std::sync::Arc;

    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let mut rng = Rng(0x5a5a_5a5a_5a5a_5a5a);
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.15);
    let mask = build_full_mask(n_atom);
    let s_host = bsr4_from_dense(n_atom, &s_dense, &mask);

    // Build device-resident structures.
    let k_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &mask).unwrap());
    let t_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &mask).unwrap());
    let s_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &(s_host.row_ptr.clone(), s_host.col_idx.clone())).unwrap());
    let s = GpuBsrMatrix { struct_: s_struct, values: gpu.buf_f32(&s_host.values).unwrap() };

    let (z, rz, iters) = gpu
        .newton_schulz_inverse_dev(&s, &k_struct, &t_struct, 30, 1e-4, 3)
        .unwrap();
    println!("Newton-Schulz-dev: {iters} iters, R_Z = {rz:e}");

    // CPU reference: dense S⁻¹.
    let s_f64 = row_major_to_dmatrix_f64(&s_dense, n);
    let se = SymmetricEigen::new(s_f64.clone());
    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n { d[(i, i)] = 1.0 / se.eigenvalues[i].max(1e-12); }
    let s_inv = &se.eigenvectors * &d * se.eigenvectors.transpose();
    let s_inv_dense = dmatrix_to_row_major_f32(&s_inv);

    let z_dense = z.to_dense();
    let err = dense_max_abs_diff(&z_dense, &s_inv_dense);
    // Frozen-input residual of the *same* S the kernel used, in f64 (second review §3.6).
    let mut zs_m_i = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += z_dense[i * n + k] as f64 * s_dense[k * n + j] as f64;
            }
            let t = if i == j { acc - 1.0 } else { acc };
            zs_m_i += t * t;
        }
    }
    let rz_f64 = zs_m_i.sqrt() / (n as f64).sqrt();
    println!("  ||Z_dev - S⁻¹||_max = {err:e}  R_Z_kernel={rz:e}  ||ZS−I||_F/√N (f64)={rz_f64:e}");
    assert!(rz < 1e-3, "Newton-Schulz-dev did not converge: R_Z = {rz:e}");
    // Residual must track the actual inverse error. Do not treat a historical
    // 100× gap as proven on this tree until this print is inspected. Keep red
    // if inverse error is large.
    assert!(
        err <= 20.0 * rz || err < 1e-4,
        "NS-dev residual lies: R_Z={rz:e} but max|Z-S⁻¹|={err:e}  ||ZS−I||_F/√N={rz_f64:e} (N4)"
    );
    assert!(err < 1e-4, "Z_dev != S⁻¹: max|dZ| = {err:e} R_Z={rz:e} ||ZS−I||_F/√N={rz_f64:e} (keep red until N4)");
}

// =====================================================================
// 12. Hamiltonian-derived K₀ and real parity test
//     H,S -> Z -> K₀ -> TC2 -> K, compare against dense projector K_ref.
//     Also check R_H = ||HKS - SKH||_F -> 0.
// =====================================================================

#[test]
fn test_k0_and_tc2_vs_dense_projector() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let nocc = 3;
    let mut rng = Rng(0x1234_abcd_5678_ef90);
    // H with a clear gap.
    let mut h_dense = random_symmetric_dense(n_atom, &mut rng, 0.4);
    for i in 0..n {
        h_dense[i * n + i] += 2.0;
    }
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.2);

    // Dense reference projector.
    let k_ref_dense = cpu_density_kernel(&h_dense, &s_dense, n, nocc);

    let mask = build_full_mask(n_atom);
    let h = bsr4_from_dense(n_atom, &h_dense, &mask);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // 1. Z ≈ S⁻¹
    println!("Computing Z ≈ S⁻¹ ...");
    let (z, rz, z_iters) = gpu
        .newton_schulz_inverse(&s, &mask, &mask, 30, 1e-4, 3)
        .unwrap();
    println!("  Z: {z_iters} iters, R_Z = {rz:e}");
    assert!(rz < 1e-3, "Z did not converge for K₀ test");

    // 2. Spectral bounds of B = Z·H (Gershgorin + 10% padding).
    let (emin, emax) = gpu.spectral_bounds(&h, &z, &mask, 0.1).unwrap();
    println!("  spectral bounds: emin={emin:.4} emax={emax:.4}");
    assert!(emax > emin, "degenerate spectral bounds");

    // 3. K₀ = (emax·Z - ZHZ) / Δε
    let k0 = gpu.build_k0(&h, &s, &z, &mask, &mask, emin, emax).unwrap();
    let k0_dense = k0.to_dense();
    let k0_err = dense_max_abs_diff(&k0_dense, &k_ref_dense);
    println!("  ||K₀ - K_ref||_max = {k0_err:e} (before purification)");
    // K₀ is not the projector; this is a sanity bound, not the regression test.
    // The regression is K_final vs K_ref below (G1.10 / G2).
    assert!(k0_err < 1.0, "K₀ wildly off from K_ref: {k0_err:e}");

    // 4. TC2 purification from K₀.
    let nocc_f = nocc as f32;
    println!("TC2 purification from K₀ ...");
    let (k_final, r_i, tr, iters, _history) = gpu
        .tc2_purify(&k0, &s, nocc_f, &mask, &mask, &vec![4u8; n_atom], 40, 1e-5)
        .unwrap();
    println!("  TC2: {iters} iters, R_I={r_i:e}, Tr(KS)={tr:.6}");

    // 5. Compare final K against dense reference projector.
    let k_final_dense = k_final.to_dense();
    let k_err = dense_max_abs_diff(&k_final_dense, &k_ref_dense);
    println!("  ||K_final - K_ref||_max = {k_err:e}");
    assert!(r_i < 1e-5, "TC2 did not converge: R_I = {r_i:e} (G2)");
    assert!((tr - nocc_f).abs() < 1e-5, "Tr(KS) != Nocc: {tr} vs {nocc_f} (G2)");
    // The final K should match the dense projector much better than K₀.
    assert!(k_err < k0_err, "TC2 did not improve over K₀: {k_err:e} vs {k0_err:e}");
    // With full mask + well-conditioned S, expect good agreement.
    assert!(k_err < 1e-5, "K_final != K_ref: {k_err:e} (measured 7.7e-7; G2)");

    // 6. Hamiltonian commutator R_H = ||HKS - SKH||_F.
    let r_h = gpu.hamiltonian_residual(&h, &k_final, &s, &mask).unwrap();
    // Normalize by sqrt(N_orb) for scale.
    let r_h_norm = r_h / (n as f32).sqrt();
    println!("  R_H = ||HKS-SKH||_F = {r_h:e}  (normalized {r_h_norm:e})");
    // For the exact projector R_H = 0. With f32 + Newton-Schulz truncation,
    // expect small but nonzero. It should be much smaller than the raw K
    // magnitude.
    assert!(r_h_norm < 1e-2, "R_H too large: {r_h_norm:e}");

    // 7. Sanity: R_H for the dense reference projector should also be ~0
    //    (validates the diagnostic itself).
    let k_ref_mat = bsr4_from_dense(n_atom, &k_ref_dense, &mask);
    let r_h_ref = gpu.hamiltonian_residual(&h, &k_ref_mat, &s, &mask).unwrap();
    let r_h_ref_norm = r_h_ref / (n as f32).sqrt();
    println!("  R_H(K_ref) = {r_h_ref:e}  (normalized {r_h_ref_norm:e})");
    assert!(r_h_ref_norm < 1e-4, "R_H of exact projector too large: {r_h_ref_norm:e}");
}

// =====================================================================
// 13. R_H distinguishes eigenvector-aligned from non-aligned projectors
//     R_H = ||HKS-SKH||_F = 0 for ANY projector onto eigenvectors of the
//     generalized problem (occupied or not), because Hc_i = Sc_i ε_i for
//     every eigenvector. R_H is large only for K NOT aligned with the
//     eigenvector basis. This validates R_H as a "is K in the eigenspace?"
//     diagnostic, which together with Tr(KS)=Nocc ensures correctness.
// =====================================================================

#[test]
fn test_rh_distinguishes_projectors() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let nocc = 3;
    let mut rng = Rng(0xc0de_face_1234_5678);
    let mut h_dense = random_symmetric_dense(n_atom, &mut rng, 0.4);
    for i in 0..n {
        h_dense[i * n + i] += 2.0;
    }
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.2);
    let mask = build_full_mask(n_atom);
    let h = bsr4_from_dense(n_atom, &h_dense, &mask);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // Correct projector: lowest nocc eigenvectors.
    let k_correct_dense = cpu_density_kernel(&h_dense, &s_dense, n, nocc);
    let k_correct = bsr4_from_dense(n_atom, &k_correct_dense, &mask);

    // Non-eigenvector-aligned "projector": a random symmetric matrix with
    // trace(KS) ≈ Nocc. This is NOT idempotent and NOT eigenvector-aligned,
    // so R_H should be large.
    let k_random_dense = random_symmetric_dense(n_atom, &mut rng, 0.5);
    // Scale to roughly match trace. (Not critical — R_H is about alignment,
    // not trace.)
    let k_random = bsr4_from_dense(n_atom, &k_random_dense, &mask);

    // R_I and R_H for the correct projector.
    let (_, q_c) = gpu.ksk(&k_correct, &s, &mask, &mask).unwrap();
    let q_c_buf = gpu.buf_f32(&q_c.values).unwrap();
    let k_c_buf = gpu.buf_f32(&k_correct.values).unwrap();
    let r_i_c = gpu.idempotency_err(k_correct.nblock(), &q_c_buf, &k_c_buf).unwrap();
    let r_h_c = gpu.hamiltonian_residual(&h, &k_correct, &s, &mask).unwrap();

    // R_H for the random (non-aligned) matrix.
    let r_h_r = gpu.hamiltonian_residual(&h, &k_random, &s, &mask).unwrap();

    let sqrt_n = (n as f32).sqrt();
    println!(
        "correct: R_I={r_i_c:e} R_H={:.4e}  random: R_H={:.4e}",
        r_h_c / sqrt_n,
        r_h_r / sqrt_n
    );
    // Correct projector: R_H ≈ 0 (eigenvector-aligned).
    assert!(r_h_c / sqrt_n < 1e-3, "R_H of correct projector too large: {:.4e}", r_h_c / sqrt_n);
    // Random matrix: R_H should be much larger (not eigenvector-aligned).
    assert!(r_h_r / sqrt_n > 1e-2,
        "R_H of random matrix too small (does not distinguish!): {:.4e}",
        r_h_r / sqrt_n);
    // And the ratio should be large.
    let ratio = r_h_r / r_h_c.max(1e-30);
    println!("  R_H ratio (random/correct) = {ratio:.1e}");
    assert!(ratio > 100.0, "R_H does not distinguish: ratio={ratio:.1e}");
}

// =====================================================================
// 14. Device-resident TC2 purification (P0-D: GPU residency)
//     Verify that the new SparsePurifyWorkspace produces the same result
//     as the old host-roundtrip path, and converges to the correct density.
// =====================================================================

#[test]
fn test_tc2_dev_resident_convergence() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let nocc: f32 = 3.0;
    let mut rng = Rng(0x7e57_c0de_face_cafe);
    let mut h_dense = random_symmetric_dense(n_atom, &mut rng, 0.4);
    for i in 0..n { h_dense[i * n + i] += 2.0; }
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.25);
    let k_exact_dense = cpu_density_kernel(&h_dense, &s_dense, n, nocc as usize);
    let mask = build_full_mask(n_atom);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // K0 = alpha * K_exact (spectrally valid perturbation, same as test_tc2_convergence).
    let alpha = 0.8f32;
    let mut k0_dense = vec![0.0f32; n * n];
    for i in 0..n * n { k0_dense[i] = alpha * k_exact_dense[i]; }
    let mut k0 = bsr4_from_dense(n_atom, &k0_dense, &mask);
    k0 = gpu.symmetrize_mat(&k0).unwrap();

    // Build the device-resident workspace.
    // T mask = K mask (full) for this small test.
    let t_mask = mask.clone();
    let all4 = vec![4u8; n_atom];
    let mut ws = SparsePurifyWorkspace::new(gpu, &k0, &s, &mask, &t_mask, &all4, nocc)
        .unwrap_or_else(|e| panic!("SparsePurifyWorkspace::new failed (no skip): {e}"));

    // Run device-resident TC2 purification.
    let (k_final, r_i, tr, iters, _history) = ws.tc2_purify_dev(30, 1e-5, 1)
        .unwrap_or_else(|e| panic!("tc2_purify_dev failed (no skip): {e}"));

    println!("TC2-dev final: R_I={r_i:e}  Tr(KS)={tr:.5}  iters={iters}");

    // Same acceptance criteria as test_tc2_convergence.
    assert!(r_i < 1e-5, "TC2-dev did not converge: R_I={r_i:e} (G2)");
    assert!((tr - nocc).abs() < 1e-5, "TC2-dev trace != Nocc: {tr} vs {nocc} (G2)");

    // Compare the device-resident result to the exact density kernel.
    let k_final_dense = k_final.to_dense();
    let max_diff = dense_max_abs_diff(&k_final_dense, &k_exact_dense);
    println!("  TC2-dev max|K_final - K_exact| = {max_diff:e}");
    assert!(max_diff < 1e-5, "TC2-dev result too far from exact: {max_diff:e} (G2)");

    // The returned diagnostics must describe the returned K, not the K from
    // the preceding iteration.  This catches the old tc2_step_dev contract,
    // which reported Tr(K_old S) after swapping in K_new.
    let (trace_ref, residual_ref) = reference_trace_and_idempotency(&k_final_dense, &s_dense, n);
    println!(
        "  returned-state reference: Tr(KS)={trace_ref:.8} R_I={residual_ref:e}; API: Tr={tr:.8} R_I={r_i:e}"
    );
    assert!((tr - trace_ref).abs() < 5e-4, "returned Tr(KS) is not for returned K: API={tr}, reference={trace_ref}");
    assert!((r_i - residual_ref).abs() < 5e-5, "returned R_I is not for returned K: API={r_i:e}, reference={residual_ref:e}");
}

// =====================================================================
// 15. Non-convergence must be an explicit error.
//     A caller cannot safely consume a returned matrix when the requested
//     tolerance was not reached (especially inside an SCC loop).
// =====================================================================

#[test]
fn test_tc2_nonconvergence_is_error() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 1;
    let mask = build_full_mask(n_atom);
    let n = n_atom * BS;
    let mut k0_dense = vec![0.0f32; n * n];
    k0_dense[0] = 0.8;
    k0_dense[n + 1] = 0.8;
    let k0 = bsr4_from_dense(n_atom, &k0_dense, &mask);
    let s = build_identity(n_atom, &mask).unwrap();
    let result = gpu.tc2_purify(&k0, &s, 2.0, &mask, &mask, &vec![4u8; n_atom], 1, 1e-12);
    match result {
        Err(e) => {
            let msg = format!("{e}");
            println!("TC2 non-convergence error (expected): {msg}");
            assert!(msg.to_ascii_lowercase().contains("converg"), "error lacks convergence context: {msg}");
        }
        Ok((_, r_i, _, iters, _)) => panic!(
            "TC2 returned success after {iters} iteration(s) without reaching tol: R_I={r_i:e}"
        ),
    }
}

#[test]
fn test_newton_schulz_near_identity_nonconvergence_is_not_cancelled() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 1;
    let mask = build_full_mask(n_atom);
    let n = n_atom * BS;
    let mut s_dense = vec![0.0f32; n * n];
    s_dense[0] = 1.0001;
    s_dense[n + 1] = 0.9999;
    s_dense[2 * n + 2] = 1.0;
    s_dense[3 * n + 3] = 1.0;
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // After one Newton-Schulz update the true residual is small but nonzero.
    // The old ||T||² - 2Tr(T) + N formula can round it to zero in f32 and
    // falsely report convergence, so use a tolerance below that residual.
    let result = gpu.newton_schulz_inverse(&s, &mask, &mask, 1, 1e-9, 1);
    match result {
        Err(e) => {
            let msg = format!("{e}");
            println!("Newton-Schulz near-identity error (expected): {msg}");
            assert!(msg.to_ascii_lowercase().contains("converg"), "error lacks convergence context: {msg}");
        }
        Ok((_, r_z, _,)) => panic!(
            "Newton-Schulz falsely converged after cancellation: R_Z={r_z:e}"
        ),
    }
}

// =====================================================================
// 16. Device-resident TC2 parity vs old host-roundtrip TC2
//     Run both paths with the same K0 and verify they produce the same K
//     after the same number of iterations.
// =====================================================================

#[test]
fn test_tc2_dev_vs_host_parity() {
    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let nocc: f32 = 3.0;
    let mut rng = Rng(0xface_b00c_1234_5678);
    let mut h_dense = random_symmetric_dense(n_atom, &mut rng, 0.4);
    for i in 0..n { h_dense[i * n + i] += 2.0; }
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.25);
    let k_exact_dense = cpu_density_kernel(&h_dense, &s_dense, n, nocc as usize);
    let mask = build_full_mask(n_atom);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // K0 = 0.8 * K_exact
    let alpha = 0.8f32;
    let mut k0_dense = vec![0.0f32; n * n];
    for i in 0..n * n { k0_dense[i] = alpha * k_exact_dense[i]; }
    let mut k0 = bsr4_from_dense(n_atom, &k0_dense, &mask);
    k0 = gpu.symmetrize_mat(&k0).unwrap();

    // --- Old host-roundtrip path: run 5 TC2 steps ---
    let diag_dummy = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
    let diag = diag_block_map(&diag_dummy).unwrap();
    let diag_buf = gpu.buf_u32(&diag).unwrap();
    let mut k_host = k0.clone();
    let n_orb_buf = gpu.buf_u32(&vec![4u32; n_atom]).unwrap();
    for step in 0..5 {
        let (knew, _n) = gpu.tc2_step(&k_host, &s, nocc, &mask, &mask, &diag_buf, &n_orb_buf).unwrap();
        k_host = gpu.symmetrize_mat(&knew).unwrap();
    }

    // --- New device-resident path: run 5 TC2 steps ---
    let t_mask = mask.clone();
    let gpu2 = SparseBsr4Gpu::new(SparseBsr4Config::default()).unwrap();
    let all4 = vec![4u8; n_atom];
    let mut ws = SparsePurifyWorkspace::new(gpu2, &k0, &s, &mask, &t_mask, &all4, nocc).unwrap();
    for step in 0..5 {
        let _tr = ws.tc2_step_dev().unwrap();
    }
    let k_dev = ws.k_to_host().unwrap();

    // Compare: both should produce nearly identical K after 5 steps.
    let k_host_dense = k_host.to_dense();
    let k_dev_dense = k_dev.to_dense();
    let max_diff = dense_max_abs_diff(&k_host_dense, &k_dev_dense);
    println!("TC2 host-vs-dev after 5 steps: max|K_host - K_dev| = {max_diff:e}");
    // f32 roundoff + different reduction order → allow 1e-4.
    assert!(max_diff < 1e-4, "TC2 dev vs host mismatch: {max_diff:e}");
}

// =====================================================================
// 17. Row-degree overflow check (fail-loud, not silent zero output)
//     Verify that GpuBsrStructure::new rejects masks with rows exceeding
//     MAX_LEFT_BLOCKS, instead of silently producing incorrect results.
// =====================================================================

#[test]
fn test_row_degree_overflow_fail_loud() {
    let Some(gpu) = try_gpu() else { return };
    // Create a mask with a row that has > MAX_LEFT_BLOCKS (default 256) blocks.
    // With n_atom = 300 and full mask, row 0 has 300 blocks > 256.
    let n_atom = 300;
    let mask = build_full_mask(n_atom);
    // GpuBsrStructure::new should fail with a descriptive error.
    let result = rust_dftb::methods::sparse::gpu_sparse::GpuBsrStructure::new(&gpu, n_atom, &mask);
    assert!(result.is_err(), "GpuBsrStructure::new should fail for row degree > MAX_LEFT_BLOCKS");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => "unexpected success".to_string(),
    };
    println!("Row-degree overflow error (expected): {err_msg}");
    assert!(
        err_msg.contains("MAX_LEFT_BLOCKS") || err_msg.contains("max_left_blocks"),
        "Error should mention MAX_LEFT_BLOCKS: {err_msg}"
    );
}

// =====================================================================
// 18. P0 regression: sparse gershgorin_bounds and inf_norm must not
//     densify. Verify they produce the same result as the old dense
//     path, computed independently here from to_dense + manual scan.
//     Also verify they work on geometric (sparse) masks, not just full.
// =====================================================================

#[test]
fn test_sparse_gershgorin_no_densify() {
    let n_atom = 8;
    let mut rng = Rng(0x1234_5678_9abc_def0);
    // Build a random sparse matrix with a geometric mask.
    let pos: Vec<[f64; 3]> = (0..n_atom).map(|i| {
        [(i as f64 * 1.5) % 6.0, (i as f64 * 0.7) % 4.0, (i as f64 * 2.3) % 5.0]
    }).collect();
    let mask = build_geometric_mask(&pos, 4.0);
    let nblock = mask.1.len();
    assert!(nblock < n_atom * n_atom, "mask should be sparse");

    // Fill blocks with random values.
    let mut bsr = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
    for v in &mut bsr.values {
        *v = rng.next();
    }

    // Sparse gershgorin (new path — no to_dense).
    let (emin_sparse, emax_sparse) = gershgorin_bounds(&bsr).unwrap();

    // Dense reference: expand and compute manually.
    let dense = bsr.to_dense();
    let n_orb = n_atom * BS;
    let mut emin_dense = f32::INFINITY;
    let mut emax_dense = f32::NEG_INFINITY;
    for mu in 0..n_orb {
        let diag = dense[mu * n_orb + mu];
        let mut offdiag = 0.0f32;
        for nu in 0..n_orb {
            if nu != mu { offdiag += dense[mu * n_orb + nu].abs(); }
        }
        emin_dense = emin_dense.min(diag - offdiag);
        emax_dense = emax_dense.max(diag + offdiag);
    }

    let tol = 1e-5f32;
    assert!((emin_sparse - emin_dense).abs() < tol,
        "gershgorin emin mismatch: sparse={emin_sparse:e} dense={emin_dense:e}");
    assert!((emax_sparse - emax_dense).abs() < tol,
        "gershgorin emax mismatch: sparse={emax_sparse:e} dense={emax_dense:e}");
    println!("gershgorin (sparse mask, {nblock} blocks): emin={emin_sparse:.4} emax={emax_sparse:.4} — matches dense");
}

#[test]
fn test_sparse_inf_norm_no_densify() {
    let n_atom = 8;
    let mut rng = Rng(0xaabb_ccdd_eeff_0011);
    let pos: Vec<[f64; 3]> = (0..n_atom).map(|i| {
        [(i as f64 * 1.3) % 5.0, (i as f64 * 1.9) % 3.0, (i as f64 * 0.5) % 7.0]
    }).collect();
    let mask = build_geometric_mask(&pos, 3.5);
    let nblock = mask.1.len();
    assert!(nblock < n_atom * n_atom, "mask should be sparse");

    let mut bsr = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
    for v in &mut bsr.values {
        *v = rng.next();
    }

    // Sparse inf_norm (new path — no to_dense).
    let norm_sparse = inf_norm(&bsr);

    // Dense reference.
    let dense = bsr.to_dense();
    let n_orb = n_atom * BS;
    let mut norm_dense = 0.0f32;
    for mu in 0..n_orb {
        let row_sum: f32 = (0..n_orb).map(|nu| dense[mu * n_orb + nu].abs()).sum();
        norm_dense = norm_dense.max(row_sum);
    }

    let tol = 1e-5f32;
    assert!((norm_sparse - norm_dense).abs() < tol,
        "inf_norm mismatch: sparse={norm_sparse:e} dense={norm_dense:e}");
    println!("inf_norm (sparse mask, {nblock} blocks): {norm_sparse:.6} — matches dense");
}

// =====================================================================
// 19. P0 regression: SparsePerfStats struct exists and audit prints.
// =====================================================================

#[test]
#[ignore = "G1.9: dummy SparsePerfStats construction is not a performance contract. Instrument real counters on an NS+TC2 run."]
fn test_sparse_perf_stats_audit() {
    let stats = rust_dftb::methods::sparse::SparsePerfStats {
        n_atom: 100,
        nnz_hs: 1200,
        nnz_k: 1200,
        nnz_z: 1200,
        t_hs: 0.001,
        t_ns: 0.05,
        t_tc2: 0.2,
        t_gamma: 0.01,
        t_force: 0.0,
        t_total: 0.3,
        ..Default::default()
    };
    stats.print_audit();
    panic!("G1.9: SparsePerfStats dummy construction is not P0. Instrument real counters \
        (kernel_launches, host_syncs, largest_dense) on an actual NS+TC2 run and assert them.");
}

// =====================================================================
// 20. GPT-5.6 #19: Device-resident K0/spectral_bounds.
//     Verify that device-side spectral_bounds_dev and build_k0_dev produce
//     the same results as the host-roundtrip path.
// =====================================================================

#[test]
fn test_k0_dev_vs_host() {
    use rust_dftb::methods::sparse::gpu_sparse::{GpuBsrMatrix, GpuBsrStructure};
    use std::sync::Arc;

    let Some(gpu) = try_gpu() else { return };
    let n_atom = 3;
    let n = n_atom * BS;
    let nocc = 3;
    let mut rng = Rng(0x5a5a_c0de_1234_5678);
    let mut h_dense = random_symmetric_dense(n_atom, &mut rng, 0.4);
    for i in 0..n { h_dense[i * n + i] += 2.0; }
    let s_dense = make_overlap_dense(n_atom, &mut rng, 0.2);
    let mask = build_full_mask(n_atom);
    let h = bsr4_from_dense(n_atom, &h_dense, &mask);
    let s = bsr4_from_dense(n_atom, &s_dense, &mask);

    // 1. Z ≈ S⁻¹ via device-resident NS.
    let s_dev = GpuBsrMatrix::from_host(&gpu, &s).unwrap();
    let k_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &mask).unwrap());
    let t_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &mask).unwrap());
    let (z_host, rz, z_iters) = gpu
        .newton_schulz_inverse_dev(&s_dev, &k_struct, &t_struct, 30, 1e-4, 3)
        .unwrap();
    println!("Z (dev): {z_iters} iters, R_Z = {rz:e}");
    assert!(rz < 1e-3, "Z did not converge: R_Z={rz:e}");

    // 2. Host spectral bounds (reference).
    let (emin_host, emax_host) = gpu.spectral_bounds(&h, &z_host, &mask, 0.1).unwrap();
    println!("spectral bounds (host): emin={emin_host:.4} emax={emax_host:.4}");

    // 3. Device spectral bounds.
    let h_dev = GpuBsrMatrix::from_host(&gpu, &h).unwrap();
    let z_dev = GpuBsrMatrix::from_host(&gpu, &z_host).unwrap();
    let b_dev = GpuBsrMatrix::zero(&gpu, &t_struct).unwrap();
    let (emin_dev, emax_dev) = gpu.spectral_bounds_dev(&z_dev, &h_dev, &b_dev, 0.1).unwrap();
    println!("spectral bounds (dev):  emin={emin_dev:.4} emax={emax_dev:.4}");

    // Should match to f32 roundoff (same kernels, same order).
    let emin_err = (emin_host - emin_dev).abs();
    let emax_err = (emax_host - emax_dev).abs();
    println!("spectral bounds diff: emin_err={emin_err:e} emax_err={emax_err:e}");
    assert!(emin_err < 1e-5, "emin mismatch: host={emin_host:e} dev={emin_dev:e}");
    assert!(emax_err < 1e-5, "emax mismatch: host={emax_host:e} dev={emax_dev:e}");

    // 4. Device K0 construction.
    let a_dev = GpuBsrMatrix::zero(&gpu, &k_struct).unwrap();
    let k0_dev = GpuBsrMatrix::zero(&gpu, &k_struct).unwrap();
    gpu.build_k0_dev(&z_dev, &h_dev, &b_dev, &a_dev, &k0_dev, emin_dev, emax_dev)
        .unwrap();
    let k0_dev_host = k0_dev.to_host(&gpu).unwrap();

    // 5. Host K0 construction (reference).
    let k0_host = gpu.build_k0(&h, &s, &z_host, &mask, &mask, emin_host, emax_host).unwrap();

    // 6. Compare.
    let k0_diff = dense_max_abs_diff(&k0_host.to_dense(), &k0_dev_host.to_dense());
    println!("K0 dev vs host: max|diff| = {k0_diff:e}");
    assert!(k0_diff < 1e-5, "K0 mismatch: {k0_diff:e}");

    // 7. Verify K0 leads to correct TC2 convergence.
    let t_mask = mask.clone();
    let all4 = vec![4u8; n_atom];
    let mut ws = SparsePurifyWorkspace::new(gpu, &k0_dev_host, &s, &mask, &t_mask, &all4, nocc as f32).unwrap();
    let (k_final, r_i, tr, iters, _) = ws.tc2_purify_dev(40, 1e-5, 1).unwrap();
    println!("TC2 from dev K0: {iters} iters, R_I={r_i:e}, Tr(KS)={tr:.6}");
    assert!(r_i < 1e-5, "TC2 did not converge: R_I={r_i:e} (G2)");
    assert!((tr - nocc as f32).abs() < 1e-5, "Tr(KS) mismatch: {tr} vs {nocc} (G2)");
}
