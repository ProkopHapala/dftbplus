//! GPU diagonalization tests: compare OpenCL Jacobi + GEMM against CPU (nalgebra).
//!
//! Tests:
//!   1. GEMM correctness (synthetic small matrix)
//!   2. Jacobi — synthetic symmetric matrix (serial + parallel variants)
//!   3. Jacobi — overlap matrix S from real molecules (H2, N2)
//!   4. Löwdin transform H' = X^T·H·X on GPU
//!   5. Full generalized eigenvalue problem HC = SCε (H2, N2)
//!   6. Batched diagonalization (multiple fragments in one launch)
//!
//! Environment:
//!   RUST_DFTB_SK_DIR — directory with .skf files (for molecule-based tests)
//!
//! Tests skip gracefully if no OpenCL device is available.

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::qmqm::gpu_matrix::{GpuMatrixContext, MatrixKernelConfig, Transpose};
use rust_dftb::qmqm::{Fragment, FragmentTemplate};
use rust_dftb::{load_sk_for_species, max_abs_diff};

// ==================================================================
// Helper functions
// ==================================================================

/// Convert nalgebra DMatrix<f64> (column-major) to row-major Vec<f32>.
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

/// Convert row-major Vec<f32> to nalgebra DMatrix<f64>.
fn row_major_f32_to_dmatrix(data: &[f32], n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = data[i * n + j] as f64;
        }
    }
    m
}

/// CPU reference: matrix multiply C = A · B (both n×n, row-major f64).
fn cpu_matmul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a * b
}

/// CPU reference: symmetric eigendecomposition via nalgebra.
/// Returns (eigenvalues sorted ascending, eigenvectors as columns matching sorted order).
fn cpu_symeig(a: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let se = SymmetricEigen::new(a.clone());
    let n = se.eigenvalues.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| se.eigenvalues[i].partial_cmp(&se.eigenvalues[j]).unwrap());
    let eigvals: Vec<f64> = idx.iter().map(|&i| se.eigenvalues[i]).collect();
    let eigvecs = se.eigenvectors.select_columns(&idx);
    (eigvals, eigvecs)
}

/// Compute S^{-1/2} = V · Λ^{-1/2} · V^T from eigenvalues and eigenvectors of S.
fn compute_s_inv_half(eigvals: &[f64], eigvecs: &DMatrix<f64>) -> DMatrix<f64> {
    let n = eigvals.len();
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = 1.0 / eigvals[i].sqrt().max(1e-12);
    }
    eigvecs * d * eigvecs.transpose()
}

/// Sort GPU eigenvalues ascending and reorder eigenvectors accordingly.
/// Returns (sorted_eigvals, sorted_eigvecs as DMatrix).
fn sort_gpu_eig(eigvals: &[f32], eigvecs: &[f32], n: usize) -> (Vec<f64>, DMatrix<f64>) {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| eigvals[i].partial_cmp(&eigvals[j]).unwrap());
    let sorted_vals: Vec<f64> = idx.iter().map(|&i| eigvals[i] as f64).collect();
    // eigvecs is row-major: eigvecs[b * n * n + row * n + col]
    // For single block (b=0): eigvecs[row * n + col] = V[row, col]
    let mut sorted_vecs = DMatrix::zeros(n, n);
    for (col_out, &src_col) in idx.iter().enumerate() {
        for row in 0..n {
            sorted_vecs[(row, col_out)] = eigvecs[row * n + src_col] as f64;
        }
    }
    (sorted_vals, sorted_vecs)
}

/// Fix eigenvector sign ambiguity: for each column, if dot(v_cpu, v_gpu) < 0, flip v_gpu.
fn align_eigenvector_signs(v_cpu: &DMatrix<f64>, v_gpu: &mut DMatrix<f64>) {
    let n = v_cpu.nrows();
    for col in 0..v_cpu.ncols() {
        let mut dot = 0.0;
        for row in 0..n {
            dot += v_cpu[(row, col)] * v_gpu[(row, col)];
        }
        if dot < 0.0 {
            for row in 0..n {
                v_gpu[(row, col)] = -v_gpu[(row, col)];
            }
        }
    }
}

/// Try to create a GPU context; return None if no OpenCL device available.
fn try_gpu_ctx() -> Option<GpuMatrixContext> {
    let config = MatrixKernelConfig::nvidia_default();
    match GpuMatrixContext::new(config) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

// ==================================================================
// Test 1: GEMM correctness
// ==================================================================

#[test]
fn test_gpu_gemm_correctness() {
    let Some(ctx) = try_gpu_ctx() else { return; };

    // 4×4 matrices with known values
    let a = DMatrix::from_row_slice(4, 4, &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]);
    let b = DMatrix::from_row_slice(4, 4, &[
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0,
    ]);

    let c_cpu = cpu_matmul(&a, &b);

    let a_f32 = dmatrix_to_row_major_f32(&a);
    let b_f32 = dmatrix_to_row_major_f32(&b);

    let buf_a = ctx.buffer_from_slice(&a_f32).unwrap();
    let buf_b = ctx.buffer_from_slice(&b_f32).unwrap();
    let buf_c = ctx.zero_buffer(4 * 4).unwrap();

    ctx.batched_gemm(4, 1, Transpose::No, Transpose::No, 1.0, 0.0, &buf_a, &buf_b, &buf_c).unwrap();

    let mut c_gpu = vec![0.0f32; 16];
    ctx.read_buffer(&buf_c, &mut c_gpu).unwrap();
    let c_gpu_mat = row_major_f32_to_dmatrix(&c_gpu, 4);

    let diff = max_abs_diff(&c_cpu, &c_gpu_mat);
    assert!(diff < 1e-4, "GEMM mismatch: diff = {diff:e}\nCPU:\n{c_cpu}\nGPU:\n{c_gpu_mat}");
}

// ==================================================================
// Test 2: Jacobi — synthetic symmetric matrix
// ==================================================================

#[test]
fn test_gpu_jacobi_synthetic() {
    let Some(ctx) = try_gpu_ctx() else { return; };

    // 8×8 symmetric matrix with known structure
    let n = 8;
    let a = DMatrix::from_row_slice(n, n, &[
        4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -3.0,
    ]);

    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(&a);

    let a_f32 = dmatrix_to_row_major_f32(&a);
    let buf_a = ctx.buffer_from_slice(&a_f32).unwrap();
    let buf_eigvals = ctx.zero_buffer(n).unwrap();
    let buf_eigvecs = ctx.zero_buffer(n * n).unwrap();

    // Test serial Jacobi
    ctx.local_jacobi_blocks(n, 1, &buf_a, &buf_eigvals, &buf_eigvecs, 100, 1e-6).unwrap();

    let mut eigvals_gpu = vec![0.0f32; n];
    let mut eigvecs_gpu = vec![0.0f32; n * n];
    ctx.read_buffer(&buf_eigvals, &mut eigvals_gpu).unwrap();
    ctx.read_buffer(&buf_eigvecs, &mut eigvecs_gpu).unwrap();

    let (eigvals_gpu_sorted, mut eigvecs_gpu_sorted) = sort_gpu_eig(&eigvals_gpu, &eigvecs_gpu, n);
    align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu_sorted);

    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu_sorted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(val_diff < 1e-4, "Jacobi eigenvalue mismatch (serial): max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu_sorted:?}");

    let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu_sorted);
    assert!(vec_diff < 1e-3, "Jacobi eigenvector mismatch (serial): diff = {vec_diff:e}");

    // Test parallel Jacobi
    let buf_eigvals2 = ctx.zero_buffer(n).unwrap();
    let buf_eigvecs2 = ctx.zero_buffer(n * n).unwrap();
    ctx.local_jacobi_blocks_parallel(n, 1, &buf_a, &buf_eigvals2, &buf_eigvecs2, 100, 1e-6).unwrap();

    ctx.read_buffer(&buf_eigvals2, &mut eigvals_gpu).unwrap();
    ctx.read_buffer(&buf_eigvecs2, &mut eigvecs_gpu).unwrap();

    let (eigvals_gpu_sorted, mut eigvecs_gpu_sorted) = sort_gpu_eig(&eigvals_gpu, &eigvecs_gpu, n);
    align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu_sorted);

    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu_sorted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(val_diff < 1e-4, "Jacobi eigenvalue mismatch (parallel): max diff = {val_diff:e}");

    let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu_sorted);
    assert!(vec_diff < 1e-3, "Jacobi eigenvector mismatch (parallel): diff = {vec_diff:e}");
}

// ==================================================================
// Test 3: Jacobi — diagonalize overlap matrix S from real molecules
// ==================================================================

#[test]
fn test_gpu_jacobi_overlap_h2() {
    let Some(ctx) = try_gpu_ctx() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();

    let s = &template.s;
    let n = template.n_orbs;
    assert_eq!(n, 2);

    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(s);

    let s_f32 = dmatrix_to_row_major_f32(s);
    let buf_s = ctx.buffer_from_slice(&s_f32).unwrap();
    let buf_eigvals = ctx.zero_buffer(n).unwrap();
    let buf_eigvecs = ctx.zero_buffer(n * n).unwrap();

    ctx.local_jacobi_blocks(n, 1, &buf_s, &buf_eigvals, &buf_eigvecs, 100, 1e-6).unwrap();

    let mut eigvals_gpu = vec![0.0f32; n];
    let mut eigvecs_gpu = vec![0.0f32; n * n];
    ctx.read_buffer(&buf_eigvals, &mut eigvals_gpu).unwrap();
    ctx.read_buffer(&buf_eigvecs, &mut eigvecs_gpu).unwrap();

    let (eigvals_gpu_sorted, mut eigvecs_gpu_sorted) = sort_gpu_eig(&eigvals_gpu, &eigvecs_gpu, n);
    align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu_sorted);

    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu_sorted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(val_diff < 1e-4, "S eigenvalue mismatch (H2): max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu_sorted:?}");

    let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu_sorted);
    assert!(vec_diff < 1e-3, "S eigenvector mismatch (H2): diff = {vec_diff:e}");

    // Compute S^{-1/2} from both CPU and GPU results, compare
    let s_inv_half_cpu = compute_s_inv_half(&eigvals_cpu, &eigvecs_cpu);
    let s_inv_half_gpu = compute_s_inv_half(&eigvals_gpu_sorted, &eigvecs_gpu_sorted);
    let sih_diff = max_abs_diff(&s_inv_half_cpu, &s_inv_half_gpu);
    assert!(sih_diff < 1e-3, "S^(-1/2) mismatch (H2): diff = {sih_diff:e}");
}

#[test]
fn test_gpu_jacobi_overlap_n2() {
    let Some(ctx) = try_gpu_ctx() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["N".to_string(), "N".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();

    let s = &template.s;
    let n = template.n_orbs;
    assert_eq!(n, 8);

    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(s);

    let s_f32 = dmatrix_to_row_major_f32(s);
    let buf_s = ctx.buffer_from_slice(&s_f32).unwrap();
    let buf_eigvals = ctx.zero_buffer(n).unwrap();
    let buf_eigvecs = ctx.zero_buffer(n * n).unwrap();

    ctx.local_jacobi_blocks(n, 1, &buf_s, &buf_eigvals, &buf_eigvecs, 100, 1e-6).unwrap();

    let mut eigvals_gpu = vec![0.0f32; n];
    let mut eigvecs_gpu = vec![0.0f32; n * n];
    ctx.read_buffer(&buf_eigvals, &mut eigvals_gpu).unwrap();
    ctx.read_buffer(&buf_eigvecs, &mut eigvecs_gpu).unwrap();

    let (eigvals_gpu_sorted, mut eigvecs_gpu_sorted) = sort_gpu_eig(&eigvals_gpu, &eigvecs_gpu, n);
    align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu_sorted);

    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu_sorted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(val_diff < 1e-4, "S eigenvalue mismatch (N2): max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu_sorted:?}");

    let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu_sorted);
    assert!(vec_diff < 1e-3, "S eigenvector mismatch (N2): diff = {vec_diff:e}");

    // S^{-1/2} comparison
    let s_inv_half_cpu = compute_s_inv_half(&eigvals_cpu, &eigvecs_cpu);
    let s_inv_half_gpu = compute_s_inv_half(&eigvals_gpu_sorted, &eigvecs_gpu_sorted);
    let sih_diff = max_abs_diff(&s_inv_half_cpu, &s_inv_half_gpu);
    assert!(sih_diff < 1e-3, "S^(-1/2) mismatch (N2): diff = {sih_diff:e}");
}

// ==================================================================
// Test 4: Löwdin transform H' = X^T · H · X on GPU
// ==================================================================

#[test]
fn test_gpu_lowdin_transform() {
    let Some(ctx) = try_gpu_ctx() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["N".to_string(), "N".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();

    let n = template.n_orbs;
    let h0 = &template.h0;
    let s = &template.s;

    // CPU: compute X = S^{-1/2} and H' = X^T · H · X
    let (eigvals_s, eigvecs_s) = cpu_symeig(s);
    let x = compute_s_inv_half(&eigvals_s, &eigvecs_s);
    let h_prime_cpu = x.transpose() * h0 * &x;

    // GPU: upload X and H0, compute H' = X^T · H · X via two GEMMs
    let x_f32 = dmatrix_to_row_major_f32(&x);
    let h0_f32 = dmatrix_to_row_major_f32(h0);

    let buf_x = ctx.buffer_from_slice(&x_f32).unwrap();
    let buf_h = ctx.buffer_from_slice(&h0_f32).unwrap();
    let buf_scratch = ctx.zero_buffer(n * n).unwrap();
    let buf_out = ctx.zero_buffer(n * n).unwrap();

    ctx.lowdin_transform(n, 1, &buf_x, &buf_h, &buf_scratch, &buf_out).unwrap();

    let mut h_prime_gpu = vec![0.0f32; n * n];
    ctx.read_buffer(&buf_out, &mut h_prime_gpu).unwrap();
    let h_prime_gpu_mat = row_major_f32_to_dmatrix(&h_prime_gpu, n);

    let diff = max_abs_diff(&h_prime_cpu, &h_prime_gpu_mat);
    assert!(diff < 1e-3, "Löwdin transform mismatch (N2): diff = {diff:e}\nCPU:\n{h_prime_cpu}\nGPU:\n{h_prime_gpu_mat}");
}

// ==================================================================
// Test 5: Full generalized eigenvalue problem HC = SCε
// ==================================================================

/// Helper: full GPU diagonalization pipeline for a single molecule.
/// Returns (eigenvalues, eigenvectors) sorted ascending.
fn gpu_diagonalize(
    ctx: &GpuMatrixContext,
    h0: &DMatrix<f64>,
    s: &DMatrix<f64>,
    n: usize,
) -> (Vec<f64>, DMatrix<f64>) {
    // Step 1: CPU computes S^{-1/2} (from eigendecomposition)
    let (eigvals_s, eigvecs_s) = cpu_symeig(s);
    let x = compute_s_inv_half(&eigvals_s, &eigvecs_s);

    // Step 2: GPU Löwdin transform H' = X^T · H · X
    let x_f32 = dmatrix_to_row_major_f32(&x);
    let h0_f32 = dmatrix_to_row_major_f32(h0);

    let buf_x = ctx.buffer_from_slice(&x_f32).unwrap();
    let buf_h = ctx.buffer_from_slice(&h0_f32).unwrap();
    let buf_scratch = ctx.zero_buffer(n * n).unwrap();
    let buf_hp = ctx.zero_buffer(n * n).unwrap();

    ctx.lowdin_transform(n, 1, &buf_x, &buf_h, &buf_scratch, &buf_hp).unwrap();

    // Step 3: GPU Jacobi diagonalization of H'
    let buf_eigvals = ctx.zero_buffer(n).unwrap();
    let buf_eigvecs = ctx.zero_buffer(n * n).unwrap();
    ctx.local_jacobi_blocks(n, 1, &buf_hp, &buf_eigvals, &buf_eigvecs, 100, 1e-6).unwrap();

    let mut eigvals_hp = vec![0.0f32; n];
    let mut eigvecs_hp = vec![0.0f32; n * n];
    ctx.read_buffer(&buf_eigvals, &mut eigvals_hp).unwrap();
    ctx.read_buffer(&buf_eigvecs, &mut eigvecs_hp).unwrap();

    let (eigvals_sorted, eigvecs_sorted) = sort_gpu_eig(&eigvals_hp, &eigvecs_hp, n);

    // Step 4: Back-transform C = X · C'
    let eigvecs_sorted_f32: Vec<f32> = {
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                v[i * n + j] = eigvecs_sorted[(i, j)] as f32;
            }
        }
        v
    };
    let buf_cp = ctx.buffer_from_slice(&eigvecs_sorted_f32).unwrap();
    let buf_c = ctx.zero_buffer(n * n).unwrap();
    // C = X · C'  (no transpose)
    ctx.batched_gemm(n, 1, Transpose::No, Transpose::No, 1.0, 0.0, &buf_x, &buf_cp, &buf_c).unwrap();

    let mut c_gpu = vec![0.0f32; n * n];
    ctx.read_buffer(&buf_c, &mut c_gpu).unwrap();
    let c_mat = row_major_f32_to_dmatrix(&c_gpu, n);

    (eigvals_sorted, c_mat)
}

#[test]
fn test_gpu_full_diagonalization_h2() {
    let Some(ctx) = try_gpu_ctx() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();
    let mut frag = Fragment::from_template(template, coords);

    // Build non-SCC Hamiltonian (neutral, zero shifts)
    frag.build_h_scc();

    // CPU reference
    frag.diagonalize().unwrap();
    let eigvals_cpu: Vec<f64> = frag.eigenvalues.iter().cloned().collect();
    let eigvecs_cpu = frag.eigenvectors.clone();

    // GPU pipeline
    let (eigvals_gpu, mut eigvecs_gpu) = gpu_diagonalize(&ctx, &frag.template.h0, &frag.template.s, frag.template.n_orbs);
    align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu);

    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(val_diff < 1e-4, "Full diag eigenvalue mismatch (H2): max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu:?}");

    let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu);
    assert!(vec_diff < 1e-3, "Full diag eigenvector mismatch (H2): diff = {vec_diff:e}");
}

#[test]
fn test_gpu_full_diagonalization_n2() {
    let Some(ctx) = try_gpu_ctx() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["N".to_string(), "N".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();
    let mut frag = Fragment::from_template(template, coords);

    frag.build_h_scc();
    frag.diagonalize().unwrap();
    let eigvals_cpu: Vec<f64> = frag.eigenvalues.iter().cloned().collect();
    let eigvecs_cpu = frag.eigenvectors.clone();

    let (eigvals_gpu, mut eigvecs_gpu) = gpu_diagonalize(&ctx, &frag.template.h0, &frag.template.s, frag.template.n_orbs);
    align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu);

    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(val_diff < 1e-4, "Full diag eigenvalue mismatch (N2): max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu:?}");

    let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu);
    assert!(vec_diff < 1e-3, "Full diag eigenvector mismatch (N2): diff = {vec_diff:e}");
}

// ==================================================================
// Test 6: Batched diagonalization (multiple fragments in one launch)
// ==================================================================

#[test]
fn test_gpu_batched_jacobi() {
    let Some(ctx) = try_gpu_ctx() else { return; };

    // 3 copies of the same 4×4 symmetric matrix, with different scaling
    let n = 4;
    let batch = 3;
    let a_base = DMatrix::from_row_slice(n, n, &[
        4.0, 1.0, 0.0, 0.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
    ]);

    // Scale factors: 1.0, 2.0, 0.5
    let scales = [1.0_f64, 2.0, 0.5];

    // Build batched flat array: [batch][n*n] row-major
    let mut batched_a = vec![0.0f32; batch * n * n];
    for b in 0..batch {
        let a_scaled = &a_base * scales[b];
        let a_f32 = dmatrix_to_row_major_f32(&a_scaled);
        batched_a[b * n * n..(b + 1) * n * n].copy_from_slice(&a_f32);
    }

    let buf_a = ctx.buffer_from_slice(&batched_a).unwrap();
    let buf_eigvals = ctx.zero_buffer(batch * n).unwrap();
    let buf_eigvecs = ctx.zero_buffer(batch * n * n).unwrap();

    ctx.local_jacobi_blocks(n, batch, &buf_a, &buf_eigvals, &buf_eigvecs, 100, 1e-6).unwrap();

    let mut eigvals_gpu = vec![0.0f32; batch * n];
    let mut eigvecs_gpu = vec![0.0f32; batch * n * n];
    ctx.read_buffer(&buf_eigvals, &mut eigvals_gpu).unwrap();
    ctx.read_buffer(&buf_eigvecs, &mut eigvecs_gpu).unwrap();

    for b in 0..batch {
        let a_scaled = &a_base * scales[b];
        let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(&a_scaled);

        // Extract batch b eigenvalues and eigenvectors
        let vals_b: Vec<f32> = eigvals_gpu[b * n..(b + 1) * n].to_vec();
        let vecs_b: &[f32] = &eigvecs_gpu[b * n * n..(b + 1) * n * n];
        let (eigvals_gpu_sorted, mut eigvecs_gpu_sorted) = sort_gpu_eig(&vals_b, vecs_b, n);
        align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu_sorted);

        let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu_sorted.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(val_diff < 1e-4, "Batched Jacobi eigenvalue mismatch (batch {b}): max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu_sorted:?}");

        let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu_sorted);
        assert!(vec_diff < 1e-3, "Batched Jacobi eigenvector mismatch (batch {b}): diff = {vec_diff:e}");
    }
}
