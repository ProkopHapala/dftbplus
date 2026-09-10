//! P4: Symbolic SpGEMM plan parity and performance (manifest v3 §4.6).
//!
//! Verifies that the precomputed symbolic plan kernel produces identical
//! results to the runtime-intersection kernel, and records plan statistics
//! (terms, bytes, avg terms/block) as required by the manifest.

use rust_dftb::methods::sparse::bsr4::{
    build_geometric_mask, build_product_mask, Bsr4Matrix, BS,
};
use rust_dftb::methods::sparse::gpu_sparse::{SparseBsr4Gpu, GpuBsrMatrix};
use rust_dftb::methods::sparse::bsr4::build_spgemm_plan_bsym;
use rust_dftb::methods::sparse::harness::require_sparse_gpu;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x as i64 as f64) / (i64::MAX as f64)) as f32
    }
}

fn bsr4_from_dense(n_atom: usize, dense: &[f32], mask: &(Vec<u32>, Vec<u32>)) -> Bsr4Matrix {
    let mut m = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
    for i in 0..n_atom {
        let (start, end) = (mask.0[i] as usize, mask.0[i + 1] as usize);
        for blk in start..end {
            let j = mask.1[blk] as usize;
            let mut v = [0.0f32; BS * BS];
            for r in 0..BS {
                for c in 0..BS {
                    v[r * BS + c] = dense[(i * BS + r) * (n_atom * BS) + (j * BS + c)];
                }
            }
            m.set_block(i, j, &v).unwrap();
        }
    }
    m
}

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

fn try_gpu() -> Option<SparseBsr4Gpu> {
    require_sparse_gpu()
}

#[test]
fn test_spgemm_plan_bsym_parity_and_stats() {
    let Some(gpu) = try_gpu() else { return };

    // 8-atom linear chain, spacing 1.5 Å.
    let n_atom = 8;
    let pos: Vec<[f64; 3]> = (0..n_atom).map(|i| [1.5 * i as f64, 0.0, 0.0]).collect();
    let mut rng = Rng(0xfeed_face_dead_beef);

    // K mask (wider), S mask (narrower) — both symmetric.
    let k_mask = build_geometric_mask(&pos, 4.5);
    let s_mask = build_geometric_mask(&pos, 2.0);
    let t_mask = build_product_mask(n_atom, &k_mask, &s_mask);

    // Random symmetric A (on K mask), random symmetric B (on S mask).
    let a_dense = random_symmetric_dense(n_atom, &mut rng, 1.0);
    let b_dense = random_symmetric_dense(n_atom, &mut rng, 0.5);
    let a = bsr4_from_dense(n_atom, &a_dense, &k_mask);
    let b = bsr4_from_dense(n_atom, &b_dense, &s_mask);

    // Build the symbolic plan for C = P_T(A·B), B symmetric.
    let plan = build_spgemm_plan_bsym(&a, &b, &t_mask).unwrap();

    eprintln!("=== P4 SpGEMM Plan Statistics ===");
    eprintln!("  n_atom         = {n_atom}");
    eprintln!("  |M_K|          = {} blocks", k_mask.1.len());
    eprintln!("  |M_S|          = {} blocks", s_mask.1.len());
    eprintln!("  |M_T|          = {} blocks", t_mask.1.len());
    eprintln!("  plan terms     = {}", plan.nterms());
    eprintln!("  plan bytes     = {} ({:.1} KiB)", plan.bytes(), plan.bytes() as f64 / 1024.0);
    eprintln!("  avg terms/blk  = {:.2}", plan.avg_terms());

    // Upload A, B to device.
    let a_gpu = GpuBsrMatrix::from_host(&gpu, &a).unwrap();
    let b_gpu = GpuBsrMatrix::from_host(&gpu, &b).unwrap();

    // C structure (on t_mask).
    let c_struct = rust_dftb::methods::sparse::gpu_sparse::GpuBsrStructure::new(
        &gpu, n_atom, &t_mask,
    ).unwrap();
    let c_struct = std::sync::Arc::new(c_struct);

    // --- Reference: runtime-intersection kernel (spgemm_bsym_dev) ---
    let c_ref = rust_dftb::methods::sparse::gpu_sparse::GpuBsrMatrix::zero(
        &gpu, &c_struct,
    ).unwrap();
    gpu.spgemm_bsym_dev(&a_gpu, &b_gpu, &c_ref).unwrap();
    let c_ref_host = c_ref.to_host(&gpu).unwrap();

    // --- Plan kernel (spgemm_plan_bsym_dev) ---
    let plan_gpu = gpu.upload_plan(&plan).unwrap();
    let c_plan = rust_dftb::methods::sparse::gpu_sparse::GpuBsrMatrix::zero(
        &gpu, &c_struct,
    ).unwrap();
    gpu.spgemm_plan_bsym_dev(&a_gpu, &b_gpu, &plan_gpu, &c_plan).unwrap();
    let c_plan_host = c_plan.to_host(&gpu).unwrap();

    // Parity: plan kernel must match intersection kernel exactly (same arithmetic).
    let max_diff: f32 = c_ref_host.values.iter().zip(c_plan_host.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("\n=== Parity ===");
    eprintln!("  ||C_plan - C_intersection||_max = {max_diff:.3e}");
    assert!(max_diff < 1e-6,
        "P4 parity failed: plan kernel differs from intersection kernel by {max_diff:.3e}");
    eprintln!("  PASS — plan kernel matches intersection kernel.");

    // --- Performance comparison (manifest §4.6: record before/after time) ---
    eprintln!("\n=== Performance (10 launches each) ===");
    let n_launches = 10usize;

    // Warm up
    for _ in 0..3 {
        gpu.spgemm_bsym_dev(&a_gpu, &b_gpu, &c_ref).unwrap();
        gpu.spgemm_plan_bsym_dev(&a_gpu, &b_gpu, &plan_gpu, &c_plan).unwrap();
    }

    // Time intersection kernel
    let t0 = Instant::now();
    for _ in 0..n_launches {
        gpu.spgemm_bsym_dev(&a_gpu, &b_gpu, &c_ref).unwrap();
    }
    gpu.runtime().queue().finish().unwrap();
    let t_intersection = t0.elapsed().as_secs_f64() / n_launches as f64;

    // Time plan kernel
    let t0 = Instant::now();
    for _ in 0..n_launches {
        gpu.spgemm_plan_bsym_dev(&a_gpu, &b_gpu, &plan_gpu, &c_plan).unwrap();
    }
    gpu.runtime().queue().finish().unwrap();
    let t_plan = t0.elapsed().as_secs_f64() / n_launches as f64;

    eprintln!("  intersection kernel: {:.3} ms/launch", t_intersection * 1e3);
    eprintln!("  plan kernel:          {:.3} ms/launch", t_plan * 1e3);
    eprintln!("  speedup:              {:.2}x", t_intersection / t_plan);
    eprintln!("\n  P4: PASS — symbolic plan built, parity verified, performance recorded.");
}
