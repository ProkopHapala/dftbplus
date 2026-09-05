//! GPU Hamiltonian/overlap assembly tests (Agent_1, Wave 1).
//!
//! Verifies that `GpuDriver::gpu_assemble_batched` produces H0/S matrices
//! matching the CPU reference `HamiltonianBuilder::build_non_scc` to within
//! the frozen tolerance (1e-5 max abs, f32 GPU vs f64 CPU).
//!
//! Tests:
//!   1. `test_gpu_assemble_pairs_smoke`   — 1× H2, kernels launch, shape 2×2
//!   2. `test_gpu_hs_parity_h2`           — H2 parity vs CPU
//!   3. `test_gpu_hs_parity_n2`           — N2 parity vs CPU (8×8)
//!   4. `test_gpu_multi_replica`          — 10× H2 at varying bond lengths
//!
//! Environment:
//!   RUST_DFTB_SK_DIR — directory with mio-1-1 .skf files
//!
//! Tests skip gracefully if no OpenCL device or no SK dir (same pattern as
//! `tests/gpu_diagonalization.rs::try_gpu_ctx`).

use nalgebra::DMatrix;
use rust_dftb::qmqm::gpu_driver::GpuDriver;
use rust_dftb::qmqm::gpu_prep::GpuBatch;
use rust_dftb::qmqm::{Fragment, FragmentTemplate, GammaTable};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder};

/// Try to create a GPU driver; return None if no OpenCL device available.
fn try_gpu() -> Option<GpuDriver> {
    match GpuDriver::new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

/// Build a single `Fragment` (one replica) for the given species/coords.
fn make_fragment(sk: &rust_dftb::SkData, species: &[String], coords: &[[f64; 3]]) -> Fragment {
    let template = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
    Fragment::from_template(template, coords.to_vec())
}

/// Extract replica `r` (n×n) from a flat row-major batch buffer.
fn extract_replica(flat: &[f32], r: usize, n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = flat[r * n * n + i * n + j] as f64;
        }
    }
    m
}

/// Max abs diff between two same-shaped matrices.
fn max_abs_diff(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

// ==================================================================
// Test 1: smoke — kernels launch, output has expected shape
// ==================================================================

#[test]
fn test_gpu_assemble_pairs_smoke() {
    let Some(_driver) = try_gpu() else { return; };
    let driver = _driver;
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();

    let frag = make_fragment(&sk, &species, &coords);
    let batch = GpuBatch::from_fragments(&[frag], &sk, &gamma).unwrap();

    let (h, s) = driver.gpu_assemble_batched(&batch).unwrap();

    // H2: 2 orbitals, 1 replica → 2*2 = 4 elements
    assert_eq!(h.len(), 4, "H buffer length wrong: {}", h.len());
    assert_eq!(s.len(), 4, "S buffer length wrong: {}", s.len());

    // S diagonal should be 1.0 (onsite overlap)
    assert!((s[0] - 1.0).abs() < 1e-5, "S[0,0] != 1.0: {}", s[0]);
    assert!((s[3] - 1.0).abs() < 1e-5, "S[1,1] != 1.0: {}", s[3]);
    // H diagonal should be e_s(H) from SK file (nonzero, finite)
    assert!(h[0].abs() > 1e-6, "H[0,0] suspiciously small: {}", h[0]);
    assert!(h[3].abs() > 1e-6, "H[1,1] suspiciously small: {}", h[3]);
    // Off-diagonal should be symmetric
    assert!((h[1] - h[2]).abs() < 1e-6, "H not symmetric: {} vs {}", h[1], h[2]);
    assert!((s[1] - s[2]).abs() < 1e-6, "S not symmetric: {} vs {}", s[1], s[2]);

    eprintln!("smoke OK: H2 H = {:?}", h);
    eprintln!("smoke OK: H2 S = {:?}", s);
}

// ==================================================================
// Test 2: H2 parity vs CPU
// ==================================================================

#[test]
fn test_gpu_hs_parity_h2() {
    let Some(_driver) = try_gpu() else { return; };
    let driver = _driver;
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();

    // CPU reference
    let builder = HamiltonianBuilder::new(sk.clone());
    let ham_cpu = builder.build_non_scc(&species, &coords).unwrap();
    let n = ham_cpu.h0.nrows();
    assert_eq!(n, 2);

    // GPU
    let frag = make_fragment(&sk, &species, &coords);
    let batch = GpuBatch::from_fragments(&[frag], &sk, &gamma).unwrap();
    let (h_flat, s_flat) = driver.gpu_assemble_batched(&batch).unwrap();

    let h_gpu = extract_replica(&h_flat, 0, n);
    let s_gpu = extract_replica(&s_flat, 0, n);

    let dh = max_abs_diff(&ham_cpu.h0, &h_gpu);
    let ds = max_abs_diff(&ham_cpu.s, &s_gpu);

    eprintln!("H2 parity: max|dH| = {dh:e}, max|dS| = {ds:e}");
    // Tolerance 1e-5: GPU f32 + original grid (499 pts) with B-spline control
    // point conversion. Off-by-one grid convention fixed 2026-09-06.
    // Achieved: ~1e-8 for H2 (s-only).
    assert!(dh < 1e-5, "H2 H parity failed: max|dH| = {dh:e}\nCPU:\n{}\nGPU:\n{}", ham_cpu.h0, h_gpu);
    assert!(ds < 1e-5, "H2 S parity failed: max|dS| = {ds:e}\nCPU:\n{}\nGPU:\n{}", ham_cpu.s, s_gpu);
}

// ==================================================================
// Test 3: N2 parity vs CPU (8×8, block_type 2 = sp-sp)
// ==================================================================

#[test]
fn test_gpu_hs_parity_n2() {
    let Some(_driver) = try_gpu() else { return; };
    let driver = _driver;
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    let species = vec!["N".to_string(), "N".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();

    // CPU reference
    let builder = HamiltonianBuilder::new(sk.clone());
    let ham_cpu = builder.build_non_scc(&species, &coords).unwrap();
    let n = ham_cpu.h0.nrows();
    assert_eq!(n, 8);

    // GPU
    let frag = make_fragment(&sk, &species, &coords);
    let batch = GpuBatch::from_fragments(&[frag], &sk, &gamma).unwrap();
    let (h_flat, s_flat) = driver.gpu_assemble_batched(&batch).unwrap();

    let h_gpu = extract_replica(&h_flat, 0, n);
    let s_gpu = extract_replica(&s_flat, 0, n);

    let dh = max_abs_diff(&ham_cpu.h0, &h_gpu);
    let ds = max_abs_diff(&ham_cpu.s, &s_gpu);

    eprintln!("N2 parity: max|dH| = {dh:e}, max|dS| = {ds:e}");
    // Tolerance 1e-5: GPU f32 + original grid with B-spline control points.
    // Achieved: ~1e-7 for N2 (sp-sp, 4-channel).
    assert!(dh < 1e-5, "N2 H parity failed: max|dH| = {dh:e}\nCPU:\n{}\nGPU:\n{}", ham_cpu.h0, h_gpu);
    assert!(ds < 1e-5, "N2 S parity failed: max|dS| = {ds:e}\nCPU:\n{}\nGPU:\n{}", ham_cpu.s, s_gpu);
}

// ==================================================================
// Test 4: 10× H2 at different bond lengths, one batched launch
// ==================================================================

#[test]
fn test_gpu_multi_replica() {
    let Some(_driver) = try_gpu() else { return; };
    let driver = _driver;
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    let species = vec!["H".to_string(), "H".to_string()];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();

    // 10 replicas at bond lengths 0.6 .. 1.5 Å
    let bond_lengths: Vec<f64> = (0..10).map(|i| 0.6 + 0.1 * i as f64).collect();
    let n = 2;

    // Build CPU references and GPU fragments
    let builder = HamiltonianBuilder::new(sk.clone());
    let mut cpu_h = Vec::with_capacity(10);
    let mut cpu_s = Vec::with_capacity(10);
    let mut frags = Vec::with_capacity(10);
    for &bl in &bond_lengths {
        let coords = vec![[0.0, 0.0, 0.0], [bl, 0.0, 0.0]];
        let ham = builder.build_non_scc(&species, &coords).unwrap();
        cpu_h.push(ham.h0);
        cpu_s.push(ham.s);
        frags.push(make_fragment(&sk, &species, &coords));
    }

    let batch = GpuBatch::from_fragments(&frags, &sk, &gamma).unwrap();
    let (h_flat, s_flat) = driver.gpu_assemble_batched(&batch).unwrap();

    let mut worst_dh = 0.0f64;
    let mut worst_ds = 0.0f64;
    let mut worst_idx = 0usize;
    for (i, bl) in bond_lengths.iter().enumerate() {
        let h_gpu = extract_replica(&h_flat, i, n);
        let s_gpu = extract_replica(&s_flat, i, n);
        let dh = max_abs_diff(&cpu_h[i], &h_gpu);
        let ds = max_abs_diff(&cpu_s[i], &s_gpu);
        eprintln!("replica {i} (bl={bl:.2}): max|dH|={dh:e}, max|dS|={ds:e}");
        if dh > worst_dh || ds > worst_ds {
            worst_dh = worst_dh.max(dh);
            worst_ds = worst_ds.max(ds);
            worst_idx = i;
        }
    }

    eprintln!("multi-replica worst: replica {worst_idx}, max|dH|={worst_dh:e}, max|dS|={worst_ds:e}");
    // Tolerance 1e-5: GPU f32 + original grid with B-spline control points.
    assert!(worst_dh < 1e-5, "multi-replica H parity failed: max|dH| = {worst_dh:e}");
    assert!(worst_ds < 1e-5, "multi-replica S parity failed: max|dS| = {worst_ds:e}");
}
