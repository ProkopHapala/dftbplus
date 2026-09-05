//! GPU eigensolver tests: Brent-Luk parallel cyclic Jacobi + S^{-1/2}.
//!
//! Agent_4 (Wave 2) test file. Tests:
//!   1. Jacobi parity vs CPU (nalgebra) for H2, N2, H2O, CH4
//!   2. S^{-1/2} parity vs CPU for H2, N2, H2O
//!   3. λ_min reporting
//!   4. Batched Jacobi (10× H2O at different geometries)
//!   5. Benchmark vs existing local_jacobi_blocks_parallel
//!
//! Environment:
//!   RUST_DFTB_SK_DIR — directory with .skf files (mio-1-1 set)
//!
//! Tests skip gracefully if no OpenCL device or no SK dir.

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::qmqm::gpu_eigen::{build_inv_sqrt, jacobi_cyclic_local_batched};
use rust_dftb::qmqm::gpu_matrix::{GpuMatrixContext, MatrixKernelConfig};
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use rust_dftb::qmqm::FragmentTemplate;
use rust_dftb::{load_sk_for_species, max_abs_diff};
use std::time::Instant;

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

/// Extract eigenvalues from the diagonal of a row-major f32 buffer (one batch).
fn extract_eigvals_from_diagonal(a: &[f32], n: usize) -> Vec<f32> {
    (0..n).map(|i| a[i * n + i]).collect()
}

/// Sort GPU eigenvalues ascending and reorder eigenvectors accordingly.
fn sort_gpu_eig(eigvals: &[f32], eigvecs: &[f32], n: usize) -> (Vec<f64>, DMatrix<f64>) {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| eigvals[i].partial_cmp(&eigvals[j]).unwrap());
    let sorted_vals: Vec<f64> = idx.iter().map(|&i| eigvals[i] as f64).collect();
    let mut sorted_vecs = DMatrix::zeros(n, n);
    for (col_out, &src_col) in idx.iter().enumerate() {
        for row in 0..n {
            sorted_vecs[(row, col_out)] = eigvecs[row * n + src_col] as f64;
        }
    }
    (sorted_vals, sorted_vecs)
}

/// Fix eigenvector sign ambiguity: for each column, if dot(v_cpu, v_gpu) < 0, flip.
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

/// Degeneracy-robust eigendecomposition parity check.
///
/// Checks:
/// 1. Eigenvalues match CPU (within `val_tol`).
/// 2. Eigenvectors are orthonormal: V^T·V = I (within `vec_tol`).
/// 3. Reconstruction: A = V·diag(λ)·V^T (within `val_tol`).
/// 4. For non-degenerate eigenvalues, eigenvectors match CPU (sign-insensitive,
///    within `vec_tol`). Degenerate eigenvalues are skipped (any orthonormal
///    basis of the degenerate subspace is valid).
fn check_eig_parity(
    a_orig: &DMatrix<f64>,
    eigvals_cpu: &[f64],
    eigvecs_cpu: &DMatrix<f64>,
    eigvals_gpu: &[f64],
    eigvecs_gpu: &DMatrix<f64>,
    val_tol: f64,
    vec_tol: f64,
) -> Result<(), String> {
    let n = a_orig.nrows();

    // 1. Eigenvalue comparison
    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    if val_diff > val_tol {
        return Err(format!(
            "Eigenvalue mismatch: max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu:?}"
        ));
    }

    // 2. Orthogonality: V_gpu^T · V_gpu = I
    let vt_v = eigvecs_gpu.transpose() * eigvecs_gpu;
    let identity = DMatrix::identity(n, n);
    let ortho_diff = max_abs_diff(&vt_v, &identity);
    if ortho_diff > vec_tol {
        return Err(format!("Eigenvector orthogonality failed: diff = {ortho_diff:e}"));
    }

    // 3. Reconstruction: A = V_gpu · diag(λ_gpu) · V_gpu^T
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = eigvals_gpu[i];
    }
    let a_recon = eigvecs_gpu * d * eigvecs_gpu.transpose();
    let recon_diff = max_abs_diff(a_orig, &a_recon);
    if recon_diff > val_tol {
        return Err(format!("Reconstruction mismatch: diff = {recon_diff:e}"));
    }

    // 4. Direct eigenvector comparison for non-degenerate eigenvalues only.
    // Eigenvalues within 1e-4 of each other are considered degenerate.
    let degenerate_threshold = 1e-4f64;
    let mut max_vec_diff = 0.0f64;
    for i in 0..n {
        let mut is_degenerate = false;
        for j in 0..n {
            if i != j && (eigvals_cpu[i] - eigvals_cpu[j]).abs() < degenerate_threshold {
                is_degenerate = true;
                break;
            }
        }
        if is_degenerate {
            continue;
        }
        // Sign-insensitive comparison
        let mut dot = 0.0;
        for row in 0..n {
            dot += eigvecs_cpu[(row, i)] * eigvecs_gpu[(row, i)];
        }
        let diff = if dot >= 0.0 {
            (0..n).map(|row| (eigvecs_cpu[(row, i)] - eigvecs_gpu[(row, i)]).abs()).fold(0.0, f64::max)
        } else {
            (0..n).map(|row| (eigvecs_cpu[(row, i)] + eigvecs_gpu[(row, i)]).abs()).fold(0.0, f64::max)
        };
        max_vec_diff = max_vec_diff.max(diff);
    }
    if max_vec_diff > vec_tol {
        return Err(format!(
            "Eigenvector mismatch (non-degenerate): max diff = {max_vec_diff:e}"
        ));
    }

    Ok(())
}

/// Try to create a GpuRuntime; return None if no OpenCL device available.
fn try_rt() -> Option<GpuRuntime> {
    match GpuRuntime::new() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

/// Build a FragmentTemplate for a molecule, or None if SK dir missing.
fn try_template(species: Vec<String>, coords: Vec<[f64; 3]>) -> Option<FragmentTemplate> {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return None;
    };
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    Some(FragmentTemplate::new(&sk, species, coords).unwrap())
}

/// Run GPU Jacobi on a single matrix and return (sorted eigvals, sorted eigvecs).
fn gpu_jacobi_single(
    rt: &mut GpuRuntime,
    mat_f32: &[f32],
    n: usize,
) -> (Vec<f64>, DMatrix<f64>) {
    let a_buf = rt.buffer_from_slice(mat_f32).unwrap();
    let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
    jacobi_cyclic_local_batched(rt, &a_buf, &v_buf, n, 1).unwrap();

    let mut a_host = vec![0.0f32; n * n];
    let mut v_host = vec![0.0f32; n * n];
    rt.read_buffer(&a_buf, &mut a_host).unwrap();
    rt.read_buffer(&v_buf, &mut v_host).unwrap();

    let eigvals = extract_eigvals_from_diagonal(&a_host, n);
    sort_gpu_eig(&eigvals, &v_host, n)
}

/// Simple deterministic PRNG (LCG) for benchmark matrices — no rand dependency.
fn make_symmetric(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = if i == j {
                ((state % 900) as f32) / 100.0 + 1.0 // diagonal in [1, 10)
            } else {
                ((state % 200) as f32) / 100.0 - 1.0 // off-diagonal in [-1, 1)
            };
            a[i * n + j] = val;
            a[j * n + i] = val;
        }
    }
    a
}

// ==================================================================
// Test 1: Jacobi parity vs CPU
// ==================================================================

#[test]
fn test_jacobi_parity_h2() {
    let Some(mut rt) = try_rt() else { return; };
    let Some(template) = try_template(
        vec!["H".to_string(), "H".to_string()],
        vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
    ) else { return; };

    let n = template.n_orbs;
    assert_eq!(n, 2);
    let h0 = &template.h0;
    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(h0);

    let h0_f32 = dmatrix_to_row_major_f32(h0);
    let (eigvals_gpu, mut eigvecs_gpu) = gpu_jacobi_single(&mut rt, &h0_f32, n);
    align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu);

    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(val_diff < 1e-4,
        "Jacobi eigenvalue mismatch (H2): max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu:?}");

    let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu);
    assert!(vec_diff < 1e-3,
        "Jacobi eigenvector mismatch (H2): diff = {vec_diff:e}");
}

#[test]
fn test_jacobi_parity_n2() {
    let Some(mut rt) = try_rt() else { return; };
    let Some(template) = try_template(
        vec!["N".to_string(), "N".to_string()],
        vec![[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]],
    ) else { return; };

    let n = template.n_orbs;
    assert_eq!(n, 8);
    let h0 = &template.h0;
    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(h0);

    let h0_f32 = dmatrix_to_row_major_f32(h0);
    let (eigvals_gpu, eigvecs_gpu) = gpu_jacobi_single(&mut rt, &h0_f32, n);

    // N2 has degenerate eigenvalues (π orbitals) — use degeneracy-robust check.
    if let Err(e) = check_eig_parity(h0, &eigvals_cpu, &eigvecs_cpu,
                                     &eigvals_gpu, &eigvecs_gpu, 1e-4, 1e-3) {
        panic!("Jacobi parity (N2): {e}");
    }
}

#[test]
fn test_jacobi_parity_h2o() {
    let Some(mut rt) = try_rt() else { return; };
    let Some(template) = try_template(
        vec!["O".to_string(), "H".to_string(), "H".to_string()],
        vec![
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [0.2393, 0.9267, 0.0],
        ],
    ) else { return; };

    let n = template.n_orbs;
    assert_eq!(n, 6);
    let h0 = &template.h0;
    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(h0);

    let h0_f32 = dmatrix_to_row_major_f32(h0);
    let (eigvals_gpu, mut eigvecs_gpu) = gpu_jacobi_single(&mut rt, &h0_f32, n);
    align_eigenvector_signs(&eigvecs_cpu, &mut eigvecs_gpu);

    let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_gpu.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(val_diff < 1e-4,
        "Jacobi eigenvalue mismatch (H2O): max diff = {val_diff:e}\nCPU: {eigvals_cpu:?}\nGPU: {eigvals_gpu:?}");

    let vec_diff = max_abs_diff(&eigvecs_cpu, &eigvecs_gpu);
    assert!(vec_diff < 1e-3,
        "Jacobi eigenvector mismatch (H2O): diff = {vec_diff:e}");
}

#[test]
fn test_jacobi_parity_ch4() {
    let Some(mut rt) = try_rt() else { return; };
    let Some(template) = try_template(
        vec!["C".to_string(), "H".to_string(), "H".to_string(), "H".to_string(), "H".to_string()],
        vec![
            [0.0, 0.0, 0.0],
            [0.6294, 0.6294, 0.6294],
            [-0.6294, -0.6294, 0.6294],
            [-0.6294, 0.6294, -0.6294],
            [0.6294, -0.6294, -0.6294],
        ],
    ) else { return; };

    let n = template.n_orbs;
    assert_eq!(n, 8);
    let h0 = &template.h0;
    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(h0);

    let h0_f32 = dmatrix_to_row_major_f32(h0);
    let (eigvals_gpu, eigvecs_gpu) = gpu_jacobi_single(&mut rt, &h0_f32, n);

    // CH4 (Td symmetry) has triply-degenerate eigenvalues — use degeneracy-robust check.
    if let Err(e) = check_eig_parity(h0, &eigvals_cpu, &eigvecs_cpu,
                                     &eigvals_gpu, &eigvecs_gpu, 1e-4, 1e-3) {
        panic!("Jacobi parity (CH4): {e}");
    }
}

// ==================================================================
// Test 2: S^{-1/2} parity vs CPU
// ==================================================================

#[test]
fn test_inv_sqrt_h2() {
    let Some(mut rt) = try_rt() else { return; };
    let Some(template) = try_template(
        vec!["H".to_string(), "H".to_string()],
        vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
    ) else { return; };

    let n = template.n_orbs;
    let s = &template.s;
    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(s);
    let x_cpu = compute_s_inv_half(&eigvals_cpu, &eigvecs_cpu);

    let s_f32 = dmatrix_to_row_major_f32(s);
    let s_buf = rt.buffer_from_slice(&s_f32).unwrap();
    let (x_buf, lambda_min_buf) = build_inv_sqrt(&mut rt, &s_buf, n, 1).unwrap();

    let mut x_host = vec![0.0f32; n * n];
    let mut lambda_min = vec![0.0f32; 1];
    rt.read_buffer(&x_buf, &mut x_host).unwrap();
    rt.read_buffer(&lambda_min_buf, &mut lambda_min).unwrap();
    let x_gpu = row_major_f32_to_dmatrix(&x_host, n);

    let diff = max_abs_diff(&x_cpu, &x_gpu);
    assert!(diff < 1e-3,
        "S^(-1/2) mismatch (H2): diff = {diff:e}\nCPU:\n{x_cpu}\nGPU:\n{x_gpu}");

    // λ_min should be the smallest eigenvalue of S
    let lambda_min_cpu = eigvals_cpu[0];
    let lm_gpu = lambda_min[0] as f64;
    assert!((lm_gpu - lambda_min_cpu).abs() < 1e-3,
        "lambda_min mismatch (H2): GPU={lm_gpu:e}, CPU={lambda_min_cpu:e}");
}

#[test]
fn test_inv_sqrt_n2() {
    let Some(mut rt) = try_rt() else { return; };
    let Some(template) = try_template(
        vec!["N".to_string(), "N".to_string()],
        vec![[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]],
    ) else { return; };

    let n = template.n_orbs;
    let s = &template.s;
    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(s);
    let x_cpu = compute_s_inv_half(&eigvals_cpu, &eigvecs_cpu);

    let s_f32 = dmatrix_to_row_major_f32(s);
    let s_buf = rt.buffer_from_slice(&s_f32).unwrap();
    let (x_buf, lambda_min_buf) = build_inv_sqrt(&mut rt, &s_buf, n, 1).unwrap();

    let mut x_host = vec![0.0f32; n * n];
    let mut lambda_min = vec![0.0f32; 1];
    rt.read_buffer(&x_buf, &mut x_host).unwrap();
    rt.read_buffer(&lambda_min_buf, &mut lambda_min).unwrap();
    let x_gpu = row_major_f32_to_dmatrix(&x_host, n);

    let diff = max_abs_diff(&x_cpu, &x_gpu);
    assert!(diff < 1e-3,
        "S^(-1/2) mismatch (N2): diff = {diff:e}");

    let lambda_min_cpu = eigvals_cpu[0];
    let lm_gpu = lambda_min[0] as f64;
    assert!((lm_gpu - lambda_min_cpu).abs() < 1e-3,
        "lambda_min mismatch (N2): GPU={lm_gpu:e}, CPU={lambda_min_cpu:e}");
}

#[test]
fn test_inv_sqrt_h2o() {
    let Some(mut rt) = try_rt() else { return; };
    let Some(template) = try_template(
        vec!["O".to_string(), "H".to_string(), "H".to_string()],
        vec![
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [0.2393, 0.9267, 0.0],
        ],
    ) else { return; };

    let n = template.n_orbs;
    let s = &template.s;
    let (eigvals_cpu, eigvecs_cpu) = cpu_symeig(s);
    let x_cpu = compute_s_inv_half(&eigvals_cpu, &eigvecs_cpu);

    let s_f32 = dmatrix_to_row_major_f32(s);
    let s_buf = rt.buffer_from_slice(&s_f32).unwrap();
    let (x_buf, lambda_min_buf) = build_inv_sqrt(&mut rt, &s_buf, n, 1).unwrap();

    let mut x_host = vec![0.0f32; n * n];
    let mut lambda_min = vec![0.0f32; 1];
    rt.read_buffer(&x_buf, &mut x_host).unwrap();
    rt.read_buffer(&lambda_min_buf, &mut lambda_min).unwrap();
    let x_gpu = row_major_f32_to_dmatrix(&x_host, n);

    let diff = max_abs_diff(&x_cpu, &x_gpu);
    assert!(diff < 1e-3,
        "S^(-1/2) mismatch (H2O): diff = {diff:e}");

    let lambda_min_cpu = eigvals_cpu[0];
    let lm_gpu = lambda_min[0] as f64;
    assert!((lm_gpu - lambda_min_cpu).abs() < 1e-3,
        "lambda_min mismatch (H2O): GPU={lm_gpu:e}, CPU={lambda_min_cpu:e}");
}

// ==================================================================
// Test 3: λ_min reporting
// ==================================================================

#[test]
fn test_lambda_min_reporting() {
    let Some(mut rt) = try_rt() else { return; };

    // Use a known 4×4 symmetric matrix with known eigenvalues
    let n = 4;
    let a = DMatrix::from_row_slice(n, n, &[
        4.0, 1.0, 0.0, 0.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
    ]);
    let (eigvals_cpu, _) = cpu_symeig(&a);
    let lambda_min_cpu = eigvals_cpu[0];

    let a_f32 = dmatrix_to_row_major_f32(&a);
    let a_buf = rt.buffer_from_slice(&a_f32).unwrap();
    let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
    jacobi_cyclic_local_batched(&mut rt, &a_buf, &v_buf, n, 1).unwrap();

    // Now use build_inv_sqrt to get lambda_min
    let s_buf = rt.buffer_from_slice(&a_f32).unwrap();
    let (_x_buf, lambda_min_buf) = build_inv_sqrt(&mut rt, &s_buf, n, 1).unwrap();
    let mut lambda_min = vec![0.0f32; 1];
    rt.read_buffer(&lambda_min_buf, &mut lambda_min).unwrap();

    assert!(lambda_min[0] > 0.0, "lambda_min should be positive, got {}", lambda_min[0]);
    let lm_gpu = lambda_min[0] as f64;
    assert!((lm_gpu - lambda_min_cpu).abs() < 1e-3,
        "lambda_min mismatch: GPU={lm_gpu:e}, CPU={lambda_min_cpu:e}");
    eprintln!("test_lambda_min_reporting: GPU={lm_gpu:e}, CPU={lambda_min_cpu:e}");
}

// ==================================================================
// Test 4: Batched Jacobi (10× H2O at different geometries)
// ==================================================================

#[test]
fn test_batched_jacobi() {
    let Some(mut rt) = try_rt() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // Build 10 H2O molecules at different O-H bond lengths
    let batch = 10;
    let mut templates = Vec::new();
    for i in 0..batch {
        let r_oh = 0.85 + 0.05 * i as f64; // 0.85 to 1.30 Å
        let coords = vec![
            [0.0, 0.0, 0.0],
            [r_oh, 0.0, 0.0],
            [r_oh * 0.25, r_oh * 0.968, 0.0],
        ];
        let template = FragmentTemplate::new(&sk, species.clone(), coords).unwrap();
        templates.push(template);
    }

    let n = templates[0].n_orbs;
    assert_eq!(n, 6);

    // Stack all H0 matrices into one buffer
    let mut batched_h0 = vec![0.0f32; batch * n * n];
    for (b, template) in templates.iter().enumerate() {
        let h0_f32 = dmatrix_to_row_major_f32(&template.h0);
        batched_h0[b * n * n..(b + 1) * n * n].copy_from_slice(&h0_f32);
    }

    // CPU reference for each
    let cpu_refs: Vec<(Vec<f64>, DMatrix<f64>)> = templates.iter()
        .map(|t| cpu_symeig(&t.h0))
        .collect();

    // GPU batched diagonalization
    let a_buf = rt.buffer_from_slice(&batched_h0).unwrap();
    let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    jacobi_cyclic_local_batched(&mut rt, &a_buf, &v_buf, n, batch).unwrap();

    let mut a_host = vec![0.0f32; batch * n * n];
    let mut v_host = vec![0.0f32; batch * n * n];
    rt.read_buffer(&a_buf, &mut a_host).unwrap();
    rt.read_buffer(&v_buf, &mut v_host).unwrap();

    // Compare each batch element
    let mut max_val_diff = 0.0f64;
    let mut max_vec_diff = 0.0f64;
    for b in 0..batch {
        let eigvals = extract_eigvals_from_diagonal(&a_host[b * n * n..], n);
        let (eigvals_sorted, mut eigvecs_sorted) = sort_gpu_eig(
            &eigvals, &v_host[b * n * n..], n);
        let (eigvals_cpu, eigvecs_cpu) = &cpu_refs[b];
        align_eigenvector_signs(eigvecs_cpu, &mut eigvecs_sorted);

        let val_diff: f64 = eigvals_cpu.iter().zip(eigvals_sorted.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let vec_diff = max_abs_diff(eigvecs_cpu, &eigvecs_sorted);
        max_val_diff = max_val_diff.max(val_diff);
        max_vec_diff = max_vec_diff.max(vec_diff);
    }

    assert!(max_val_diff < 1e-4,
        "Batched Jacobi eigenvalue mismatch: max diff = {max_val_diff:e}");
    assert!(max_vec_diff < 1e-3,
        "Batched Jacobi eigenvector mismatch: max diff = {max_vec_diff:e}");
    eprintln!("test_batched_jacobi: {batch}×H2O, max val diff = {max_val_diff:e}, max vec diff = {max_vec_diff:e}");
}

// ==================================================================
// Benchmark: Brent-Luk Jacobi vs existing local_jacobi_blocks_parallel
// ==================================================================

#[test]
fn bench_jacobi_vs_old() {
    let Some(mut rt) = try_rt() else { return; };

    let config = MatrixKernelConfig::nvidia_default();
    let ctx = match GpuMatrixContext::new(config) {
        Ok(c) => c,
        Err(e) => { eprintln!("Skipping benchmark: no GpuMatrixContext ({e})"); return; }
    };

    eprintln!("\n=== Benchmark: Brent-Luk Jacobi (Agent_4) vs local_jacobi_blocks_parallel (Agent_1) ===");
    eprintln!("{:>6} {:>8} {:>16} {:>16} {:>10}", "N", "batch", "Brent-Luk (µs)", "Old Jacobi (µs)", "speedup");

    let configs: &[(usize, usize)] = &[
        (8, 1), (8, 10), (8, 100), (8, 1000),
        (16, 1), (16, 10), (16, 100), (16, 1000),
        (32, 1), (32, 10), (32, 100), (32, 1000),
        (48, 1), (48, 10), (48, 100), (48, 1000),
        (64, 1), (64, 10), (64, 100), (64, 1000),
    ];

    for &(n, batch) in configs {
        // Generate batched symmetric matrices
        let mut data = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let mut m = make_symmetric(n, 0xDEAD_BEEF + b as u64 * 777 + n as u64);
            data.append(&mut m);
        }

        // --- Brent-Luk (Agent_4) ---
        let a_buf = rt.buffer_from_slice(&data).unwrap();
        let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
        // Warmup (compile kernel)
        let _ = jacobi_cyclic_local_batched(&mut rt, &a_buf, &v_buf, n, batch);
        rt.finish().unwrap();

        // Re-upload (a_buf was modified in place)
        let a_buf = rt.buffer_from_slice(&data).unwrap();
        rt.finish().unwrap();
        let t0 = Instant::now();
        jacobi_cyclic_local_batched(&mut rt, &a_buf, &v_buf, n, batch).unwrap();
        rt.finish().unwrap();
        let t_brent_luk = t0.elapsed();

        // --- Old Jacobi (Agent_1) ---
        let blocks_buf = ctx.buffer_from_slice(&data).unwrap();
        let eigvals_buf = ctx.zero_buffer(n * batch).unwrap();
        let eigvecs_buf = ctx.zero_buffer(n * n * batch).unwrap();
        // Warmup
        let _ = ctx.local_jacobi_blocks_parallel(n, batch, &blocks_buf, &eigvals_buf, &eigvecs_buf, 100, 1e-6);
        let mut dummy = vec![0.0f32; n * batch];
        ctx.read_buffer(&eigvals_buf, &mut dummy).unwrap();

        let t0 = Instant::now();
        ctx.local_jacobi_blocks_parallel(n, batch, &blocks_buf, &eigvals_buf, &eigvecs_buf, 100, 1e-6).unwrap();
        ctx.read_buffer(&eigvals_buf, &mut dummy).unwrap();
        let t_old = t0.elapsed();

        let bl_us = t_brent_luk.as_secs_f64() * 1e6;
        let old_us = t_old.as_secs_f64() * 1e6;
        let per_sys_bl = bl_us / batch as f64;
        let per_sys_old = old_us / batch as f64;
        let speedup = old_us / bl_us.max(1e-9);

        eprintln!("{:>6} {:>8} {:>12.1} ({:>5.2}/sys) {:>12.1} ({:>5.2}/sys) {:>9.2}x",
            n, batch, bl_us, per_sys_bl, old_us, per_sys_old, speedup);
    }
    eprintln!("=== Benchmark complete ===\n");
}
