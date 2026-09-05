//! Minimal H-O pair test: isolates s-p block_type=1 GPU H-assembly bug.
//!
//! H-O is a heteronuclear s-p pair (block_type=1) never tested by
//! gpu_hamiltonian.rs (which only tests H2 s-s and N2 sp-sp).

use rust_dftb::qmqm::gpu_driver::GpuDriver;
use rust_dftb::qmqm::gpu_prep::GpuBatch;
use rust_dftb::qmqm::{Fragment, FragmentTemplate, GammaTable};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder};

fn try_gpu() -> Option<GpuDriver> {
    match GpuDriver::new() {
        Ok(d) => Some(d),
        Err(e) => { eprintln!("Skipping: no OpenCL ({e})"); None }
    }
}

fn make_fragment(sk: &rust_dftb::SkData, species: &[String], coords: &[[f64; 3]]) -> Fragment {
    let tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
    Fragment::from_template(tmpl, coords.to_vec())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| ((*x as f64) - (*y as f64)).abs()).fold(0.0f64, f64::max)
}

#[test]
fn test_gpu_hs_parity_ho() {
    let Some(driver) = try_gpu() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    // H-O at 1.0 Å (typical O-H bond length)
    let species = vec!["H".to_string(), "O".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();

    // CPU reference
    let builder = HamiltonianBuilder::new(sk.clone());
    let ham_cpu = builder.build_non_scc(&species, &coords).unwrap();
    let n = ham_cpu.h0.nrows();
    eprintln!("H-O: N_orbs={n}");

    // GPU
    let frag = make_fragment(&sk, &species, &coords);
    let batch = GpuBatch::from_fragments(&[frag], &sk, &gamma).unwrap();
    let (h_flat, s_flat) = driver.gpu_assemble_batched(&batch).unwrap();

    // Compare
    let h_cpu_flat: Vec<f32> = (0..n*n).map(|idx| {
        let i = idx / n; let j = idx - i*n;
        ham_cpu.h0[(i,j)] as f32
    }).collect();
    let s_cpu_flat: Vec<f32> = (0..n*n).map(|idx| {
        let i = idx / n; let j = idx - i*n;
        ham_cpu.s[(i,j)] as f32
    }).collect();

    let dh = max_abs_diff(&h_flat, &h_cpu_flat);
    let ds = max_abs_diff(&s_flat, &s_cpu_flat);
    eprintln!("H-O parity: max|dH|={dh:.2e}, max|dS|={ds:.2e}");

    // Print full matrices for comparison
    eprintln!("GPU H:");
    for i in 0..n {
        let row: Vec<f32> = (0..n).map(|j| h_flat[i*n+j]).collect();
        eprintln!("  [{i}] {row:?}");
    }
    eprintln!("CPU H:");
    for i in 0..n {
        let row: Vec<f64> = (0..n).map(|j| ham_cpu.h0[(i,j)]).collect();
        eprintln!("  [{i}] {row:?}");
    }
    eprintln!("GPU S:");
    for i in 0..n {
        let row: Vec<f32> = (0..n).map(|j| s_flat[i*n+j]).collect();
        eprintln!("  [{i}] {row:?}");
    }
    eprintln!("CPU S:");
    for i in 0..n {
        let row: Vec<f64> = (0..n).map(|j| ham_cpu.s[(i,j)]).collect();
        eprintln!("  [{i}] {row:?}");
    }

    assert!(dh < 1e-4, "H-O H parity failed: max|dH|={dh:.2e}");
    assert!(ds < 1e-4, "H-O S parity failed: max|dS|={ds:.2e}");
}

#[test]
fn test_gpu_hs_parity_h2o() {
    let Some(driver) = try_gpu() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();

    let builder = HamiltonianBuilder::new(sk.clone());
    let ham_cpu = builder.build_non_scc(&species, &coords).unwrap();
    let n = ham_cpu.h0.nrows();

    let frag = make_fragment(&sk, &species, &coords);
    let batch = GpuBatch::from_fragments(&[frag], &sk, &gamma).unwrap();
    let (h_flat, s_flat) = driver.gpu_assemble_batched(&batch).unwrap();

    let h_cpu_flat: Vec<f32> = (0..n*n).map(|idx| {
        let i = idx / n; let j = idx - i*n;
        ham_cpu.h0[(i,j)] as f32
    }).collect();
    let s_cpu_flat: Vec<f32> = (0..n*n).map(|idx| {
        let i = idx / n; let j = idx - i*n;
        ham_cpu.s[(i,j)] as f32
    }).collect();

    let dh = max_abs_diff(&h_flat, &h_cpu_flat);
    let ds = max_abs_diff(&s_flat, &s_cpu_flat);
    eprintln!("H2O parity: max|dH|={dh:.2e}, max|dS|={ds:.2e}");

    // Find worst elements
    for i in 0..n {
        for j in 0..n {
            let d = (h_flat[i*n+j] as f64 - ham_cpu.h0[(i,j)] as f64).abs();
            if d > 1e-4 {
                eprintln!("  H[{i},{j}] = GPU {:+.6} vs CPU {:+.6} |d|={d:.2e}",
                    h_flat[i*n+j], ham_cpu.h0[(i,j)]);
            }
        }
    }

    assert!(dh < 1e-4, "H2O H parity failed: max|dH|={dh:.2e}");
    assert!(ds < 1e-4, "H2O S parity failed: max|dS|={ds:.2e}");
}
