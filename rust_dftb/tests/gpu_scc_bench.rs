//! GPU SCC performance benchmarks — timing breakdown by phase and batch size.
//!
//! Measures wall-clock time for each GPU SCC phase (S^{-1/2}, GEMM, Jacobi,
//! density, Mulliken, DIIS mixing) at various batch sizes using the formic
//! dimer geometry (28 orbitals, 10 atoms).
//!
//! Run with:
//! ```bash
//! RUST_DFTB_SK_DIR=/path/to/mio-1-1 \
//! RUST_DFTB_TIMING=1 \
//! cargo test --test gpu_scc_bench -- --ignored --nocapture
//! ```

use rust_dftb::io::parse_xyz;
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use rust_dftb::qmqm::gpu_scc::{gpu_solve_scc_batched_diis_warmstart, GpuSccTiming};
use rust_dftb::qmqm::{Fragment, FragmentTemplate, GammaTable};
use rust_dftb::qmqm::gpu_driver::GpuDriver;
use rust_dftb::qmqm::gpu_prep::GpuBatch;
use rust_dftb::{load_sk_for_species, HamiltonianBuilder};

const ANG2BOHR: f64 = 1.889_726_133;

fn per_atom_u(sk: &rust_dftb::SkData, species: &[String]) -> Vec<f64> {
    let mut unique: Vec<String> = Vec::new();
    for sp in species { if !unique.contains(sp) { unique.push(sp.clone()); } }
    let u_unique: Vec<f64> = unique.iter()
        .map(|sp| sk.onsite(sp).map(|p| p.u_hubbard).unwrap_or(0.4))
        .collect();
    species.iter().map(|sp| {
        let idx = unique.iter().position(|s| s == sp).unwrap();
        u_unique[idx]
    }).collect()
}

fn build_gamma_matrix(coords: &[[f64; 3]], u_per_atom: &[f64]) -> Vec<f32> {
    let n = coords.len();
    let mut g = vec![0.0f32; n * n];
    for a in 0..n {
        for b in 0..n {
            let dx = coords[a][0] - coords[b][0];
            let dy = coords[a][1] - coords[b][1];
            let dz = coords[a][2] - coords[b][2];
            let r = (dx*dx + dy*dy + dz*dz).sqrt() * ANG2BOHR;
            g[a*n + b] = rust_dftb::gamma_full(r, u_per_atom[a], u_per_atom[b]) as f32;
        }
    }
    g
}

fn orb_atom_map(atom_orb_off: &[u16], n_orbs: usize) -> Vec<i32> {
    let mut map = vec![0i32; n_orbs];
    for a in 0..atom_orb_off.len() - 1 {
        for mu in atom_orb_off[a] as usize..atom_orb_off[a+1] as usize {
            map[mu] = a as i32;
        }
    }
    map
}

/// Run one benchmark at a given batch size. Returns timing struct.
fn bench_batch(
    rt: &mut GpuRuntime,
    sk: &rust_dftb::SkData,
    gamma_table: &GammaTable,
    species: &[String],
    coords: &[[f64; 3]],
    n_atoms: usize,
    n_orbs: usize,
    n_occ: usize,
    batch: usize,
    label: &str,
) -> (GpuSccTiming, usize, f32) {
    // Build batched geometries (all identical — benchmarking compute, not physics)
    let geoms: Vec<Vec<[f64; 3]>> = (0..batch).map(|_| coords.to_vec()).collect();

    let frags: Vec<Fragment> = geoms.iter()
        .map(|c| {
            let tmpl = FragmentTemplate::new(sk, species.to_vec(), c.to_vec()).unwrap();
            Fragment::from_template(tmpl, c.to_vec())
        })
        .collect();
    let driver = GpuDriver::new().expect("GpuDriver init failed");
    let gpu_batch = GpuBatch::from_fragments(&frags, sk, gamma_table).unwrap();
    let (all_h0, all_s) = driver.gpu_assemble_batched(&gpu_batch).unwrap();

    let u_per_atom = per_atom_u(sk, species);
    let mut all_g = Vec::with_capacity(batch * n_atoms * n_atoms);
    let mut all_q0 = Vec::with_capacity(batch * n_atoms);
    let mut all_oa = Vec::with_capacity(batch * n_orbs);
    for c in &geoms {
        all_g.extend(build_gamma_matrix(c, &u_per_atom));
        let tmpl = FragmentTemplate::new(sk, species.to_vec(), c.to_vec()).unwrap();
        all_q0.extend(tmpl.q0.iter().map(|&q| q as f32));
        all_oa.extend(orb_atom_map(&tmpl.atom_orb_off, n_orbs));
    }

    let h0_buf = rt.buffer_from_slice(&all_h0).unwrap();
    let s_buf = rt.buffer_from_slice(&all_s).unwrap();
    let g_buf = rt.buffer_from_slice(&all_g).unwrap();
    let q0_buf = rt.buffer_from_slice(&all_q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&all_oa).unwrap();

    eprintln!("\n=== Benchmark: {label} (batch={batch}) ===");

    // Warm-up run (first call compiles kernels — don't time it)
    // Use best_effort=true to avoid panicking on large-batch convergence issues
    let _warm = gpu_solve_scc_batched_diis_warmstart(
        rt, &h0_buf, &s_buf, &g_buf, &q0_buf, &q0_buf, &oa_buf,
        n_orbs, n_atoms, n_occ, batch,
        500, 5e-6, 0.3, 8, 3, true,
    ).expect("warm-up SCC failed");

    // Timed run
    let result = gpu_solve_scc_batched_diis_warmstart(
        rt, &h0_buf, &s_buf, &g_buf, &q0_buf, &q0_buf, &oa_buf,
        n_orbs, n_atoms, n_occ, batch,
        500, 5e-6, 0.3, 8, 3, true,
    ).expect("timed SCC failed");

    let tm = result.timing.expect("timing not populated (set RUST_DFTB_TIMING=1)");
    let max_e = result.energies.iter().fold(0.0f32, |m, &e| m.max(e.abs()));
    let n_conv = result.rms.iter().filter(|&&r| r < 5e-6).count();
    if n_conv < batch {
        eprintln!("  WARNING: {n_conv}/{batch} systems converged (identical geometries — investigate GPU race condition at large batch)");
    }
    (tm, result.n_iters, max_e)
}

#[test]
#[ignore]
fn test_gpu_scc_benchmark() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    if std::env::var("RUST_DFTB_TIMING").is_err() {
        eprintln!("WARNING: RUST_DFTB_TIMING not set — timing will not be populated.");
        eprintln!("  Run with: RUST_DFTB_TIMING=1 cargo test --test gpu_scc_bench -- --ignored --nocapture");
    }

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

    let coords = xyz.coords.clone();
    let species = xyz.species.clone();
    let n_atoms = species.len();
    assert_eq!(n_atoms, 10);

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();

    // Get n_orbs and n_occ from CPU reference
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc = builder.build_scc(&species, &coords, 200, 1e-9).unwrap();
    let n_orbs = scc.h0.nrows();
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    eprintln!("Formic dimer: N_orbs={n_orbs}, N_occ={n_occ}, N_atoms={n_atoms}");
    eprintln!("CPU reference energy: {:.8}", scc.energy);

    let mut rt = GpuRuntime::new().expect("GpuRuntime init failed");

    // Benchmark at various batch sizes
    let batch_sizes = [1, 10, 41, 100, 441];
    let mut results: Vec<(usize, GpuSccTiming, usize, f32)> = Vec::new();

    for &batch in &batch_sizes {
        let (tm, n_iters, max_e) = bench_batch(
            &mut rt, &sk, &gamma, &species, &coords,
            n_atoms, n_orbs, n_occ, batch,
            &format!("formic_dimer N={n_orbs}"),
        );
        results.push((batch, tm, n_iters, max_e));
    }

    // Summary table
    eprintln!("\n=== BENCHMARK SUMMARY (formic dimer, N={}, N_occ={}) ===", n_orbs, n_occ);
    eprintln!("  {:>6}  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "batch", "iters", "total_s", "per_iter_ms", "jacobi_ms", "gemm_ms", "diis_ms", "systems/s");

    for (batch, tm, n_iters, _max_e) in &results {
        let per_iter = (tm.t_delta_q + tm.t_gamma_matvec + tm.t_h_scc_update
            + tm.t_gemm + tm.t_jacobi + tm.t_occ_sort + tm.t_back_gemm
            + tm.t_density + tm.t_mulliken + tm.t_diis_mix) / *n_iters as f64;
        let jacobi_per = tm.t_jacobi / *n_iters as f64 * 1e3;
        let gemm_per = tm.t_gemm / *n_iters as f64 * 1e3;
        let diis_per = tm.t_diis_mix / *n_iters as f64 * 1e3;
        let sys_per_sec = *batch as f64 / tm.t_total;
        eprintln!("  {:>6}  {:>6}  {:>10.4}  {:>10.3}  {:>10.3}  {:>10.3}  {:>10.3}  {:>10.1}",
            batch, n_iters, tm.t_total, per_iter * 1e3, jacobi_per, gemm_per, diis_per, sys_per_sec);
    }

    // Phase breakdown for largest batch
    if let Some((batch, tm, n_iters, _)) = results.last() {
        eprintln!("\n=== PHASE BREAKDOWN (batch={}, {} iters) ===", batch, n_iters);
        tm.print("formic_dimer", n_orbs, *batch, *n_iters);
    }

    // Save results to TSV
    let out_dir = format!("{}/debug/gpu_scc_bench", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(&out_dir).ok();
    let tsv_path = format!("{out_dir}/bench_results.tsv");
    let mut tsv = String::from("# GPU SCC benchmark results\n# batch  iters  total_s  per_iter_ms  jacobi_ms  gemm_ms  diis_ms  occ_sort_ms  density_ms  mulliken_ms  delta_q_ms  gamma_ms  h_scc_ms  back_gemm_ms  energy_ms  readback_ms  inv_sqrt_ms  alloc_ms  systems_per_s\n");
    for (batch, tm, n_iters, _) in &results {
        let per_iter = (tm.t_delta_q + tm.t_gamma_matvec + tm.t_h_scc_update
            + tm.t_gemm + tm.t_jacobi + tm.t_occ_sort + tm.t_back_gemm
            + tm.t_density + tm.t_mulliken + tm.t_diis_mix) / *n_iters as f64;
        let sys_per_sec = *batch as f64 / tm.t_total;
        tsv.push_str(&format!(
            "{batch}  {n_iters}  {:.6}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.1}\n",
            tm.t_total, per_iter * 1e3,
            tm.t_jacobi / *n_iters as f64 * 1e3,
            tm.t_gemm / *n_iters as f64 * 1e3,
            tm.t_diis_mix / *n_iters as f64 * 1e3,
            tm.t_occ_sort / *n_iters as f64 * 1e3,
            tm.t_density / *n_iters as f64 * 1e3,
            tm.t_mulliken / *n_iters as f64 * 1e3,
            tm.t_delta_q / *n_iters as f64 * 1e3,
            tm.t_gamma_matvec / *n_iters as f64 * 1e3,
            tm.t_h_scc_update / *n_iters as f64 * 1e3,
            tm.t_back_gemm / *n_iters as f64 * 1e3,
            tm.t_energy * 1e3,
            tm.t_readback * 1e3,
            tm.t_inv_sqrt * 1e3,
            tm.t_alloc * 1e3,
            sys_per_sec,
        ));
    }
    std::fs::write(&tsv_path, tsv).unwrap();
    eprintln!("\nBenchmark results saved to {tsv_path}");
}
