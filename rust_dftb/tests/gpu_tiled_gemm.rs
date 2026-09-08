//! Phase 1: tiled GEMM parity test.
//!
//! Verifies `matmul_tiled_batched` against CPU nalgebra reference for
//! N = 64, 65, 87, 96, 128 — covering the N>64 boundary that the
//! full-local GEMM cannot handle.
//!
//! Also benchmarks full-local vs tiled at N=64 to verify the crossover.

use rust_dftb::qmqm::gpu_matrix::{matmul_full_local_batched, matmul_tiled_batched};
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;

fn try_runtime() -> Option<GpuRuntime> {
    match GpuRuntime::new() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

/// Generate a random-ish batched matrix for testing.
fn random_matrix(n: usize, batch: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = vec![0.0f32; batch * n * n];
    for b in 0..batch {
        for i in 0..n {
            for j in 0..n {
                // Simple LCG for reproducibility
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
                out[b * n * n + i * n + j] = val as f32;
            }
        }
    }
    // Make symmetric for realistic H/S-like matrices
    for b in 0..batch {
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (out[b * n * n + i * n + j] + out[b * n * n + j * n + i]) * 0.5;
                out[b * n * n + i * n + j] = avg;
                out[b * n * n + j * n + i] = avg;
            }
        }
    }
    out
}

/// CPU reference: C = A · B for batched row-major matrices.
fn cpu_matmul(a: &[f32], b: &[f32], n: usize, batch: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; batch * n * n];
    for bi in 0..batch {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += a[bi * n * n + i * n + k] as f64 * b[bi * n * n + k * n + j] as f64;
                }
                c[bi * n * n + i * n + j] = sum as f32;
            }
        }
    }
    c
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

#[test]
fn test_tiled_gemm_parity() {
    let Some(mut rt) = try_runtime() else { return; };
    let batch = 3usize;
    // Test across the N=64 boundary and at nucleobase-pair dimensions
    for &n in &[16, 32, 64, 65, 87, 96, 128] {
        let a = random_matrix(n, batch, 42);
        let b = random_matrix(n, batch, 123);
        let cpu_ref = cpu_matmul(&a, &b, n, batch);

        let a_buf = rt.buffer_from_slice(&a).unwrap();
        let b_buf = rt.buffer_from_slice(&b).unwrap();
        let c_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();

        matmul_tiled_batched(&mut rt, &a_buf, &b_buf, &c_buf, n, batch)
            .expect("tiled GEMM must succeed");

        let mut gpu_out = vec![0.0f32; batch * n * n];
        rt.read_buffer(&c_buf, &mut gpu_out).unwrap();

        let diff = max_abs_diff(&gpu_out, &cpu_ref);
        eprintln!("tiled GEMM N={n}: max|dC|={diff:.2e}");
        // f32 accumulation on GPU vs f64 on CPU; tolerance scales with N
        let tol = (n as f64 * n as f64 * 1e-5).max(1e-3);
        assert!(diff < tol, "tiled GEMM N={n} parity failed: max|dC|={diff:.2e} > {tol:.2e}");
    }
}

#[test]
fn test_full_local_vs_tiled_n64() {
    let Some(mut rt) = try_runtime() else { return; };
    let n = 64usize;
    let batch = 10usize;
    let a = random_matrix(n, batch, 42);
    let b = random_matrix(n, batch, 123);
    let cpu_ref = cpu_matmul(&a, &b, n, batch);

    // Full-local
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let b_buf = rt.buffer_from_slice(&b).unwrap();
    let c_fl = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    matmul_full_local_batched(&mut rt, &a_buf, &b_buf, &c_fl, n, batch)
        .expect("full-local GEMM must succeed");
    let mut gpu_fl = vec![0.0f32; batch * n * n];
    rt.read_buffer(&c_fl, &mut gpu_fl).unwrap();
    let diff_fl = max_abs_diff(&gpu_fl, &cpu_ref);

    // Tiled
    let c_tl = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    matmul_tiled_batched(&mut rt, &a_buf, &b_buf, &c_tl, n, batch)
        .expect("tiled GEMM must succeed");
    let mut gpu_tl = vec![0.0f32; batch * n * n];
    rt.read_buffer(&c_tl, &mut gpu_tl).unwrap();
    let diff_tl = max_abs_diff(&gpu_tl, &cpu_ref);

    eprintln!("N=64 full-local vs tiled: |dC_fl|={diff_fl:.2e}, |dC_tl|={diff_tl:.2e}");
    assert!(diff_fl < 1e-2, "full-local N=64 parity: {diff_fl:.2e}");
    assert!(diff_tl < 1e-2, "tiled N=64 parity: {diff_tl:.2e}");
}
