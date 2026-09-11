//! GPU SCC performance benchmarks — production `GpuDftb` path (D7/D8/D9).
//!
//! Wall-clock per full `scc()` call (reset_q0 + all SCC iterations +
//! final readback) at various batch sizes, using the formic dimer
//! (28 orbitals, 10 atoms) — the legacy `GpuDriver` benchmark was retired
//! since it measured the non-production free-function path.
//!
//! Run with:
//! ```bash
//! RUST_DFTB_SK_DIR=/path/to/mio-1-1 \
//! cargo test --release --test gpu_scc_bench -- --ignored --nocapture
//! ```

use rust_dftb::io::parse_xyz;
use rust_dftb::load_sk_for_species;
use rust_dftb::qmqm::gpu_dftb::GpuDftb;

const N_RUNS: usize = 3;

fn bench_dftb(
    sk: &rust_dftb::SkData,
    sk_dir: &str,
    species: &[String],
    coords: &[[f64; 3]],
    batch: usize,
    label: &str,
) {
    let mut all_coords = Vec::with_capacity(batch * coords.len());
    for _ in 0..batch { all_coords.extend_from_slice(coords); }

    let mut eng = GpuDftb::new(sk.clone(), sk_dir, species.to_vec(), all_coords, batch)
        .expect("GpuDftb::new failed");

    let mut total_ms = 0.0f64;
    let mut last_iters = 0usize;
    for r in 0..N_RUNS {
        eng.reset_q0().expect("reset_q0 failed");
        let t0 = std::time::Instant::now();
        let scc = eng.scc(500, 5e-6).expect("scc failed");
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        last_iters = scc.n_iters;
        total_ms += dt;
        eprintln!("  run {r}: {dt:.2} ms ({} iters, rms={:.3e})", scc.n_iters, scc.rms);
    }
    let avg = total_ms / N_RUNS as f64;
    let per_iter = avg / last_iters.max(1) as f64;
    eprintln!("=== {label} batch={batch}: avg={avg:.2} ms/scc ({last_iters} iters, {per_iter:.3} ms/iter, {:.1} systems/s) ===",
        batch as f64 / (avg / 1e3));
}

#[test]
#[ignore]
fn test_gpu_scc_benchmark() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    let candidates = [
        "data/xyz/formic_dimer.xyz".to_string(),
        format!("{}/data/xyz/formic_dimer.xyz", env!("CARGO_MANIFEST_DIR")),
        format!("{}/../data/xyz/formic_dimer.xyz", env!("CARGO_MANIFEST_DIR")),
    ];
    let mut xyz = None;
    for path in &candidates {
        if let Ok(x) = parse_xyz(path) { xyz = Some(x); break; }
    }
    let xyz = match xyz {
        Some(x) => x,
        None => { eprintln!("Skipping: cannot load formic_dimer.xyz"); return; }
    };

    let sk = load_sk_for_species(&sk_dir, &xyz.species).unwrap();
    eprintln!("Formic dimer: N_atoms={}", xyz.species.len());

    eprintln!("\n=== GPU SCC benchmark (production GpuDftb) ===");
    for &batch in &[1usize, 8, 32] {
        bench_dftb(&sk, &sk_dir, &xyz.species, &xyz.coords, batch,
            &format!("formic_dimer"));
    }
}
