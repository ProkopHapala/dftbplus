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
use rust_dftb::methods::dftb::dftb_cpu::DftbCpu;
use rust_dftb::methods::dftb::forces::{parse_all_repulsive, repulsive_energy};
use rust_dftb::qmqm::gpu_dftb::GpuDftb;
use rust_dftb::qmqm::gpu_dftb::SccStatus;

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

// ==================================================================
// Multi-system 20x20-scan saturation benchmark (batch=400)
// ==================================================================
//
// Reproduces the production scan workload: one `GpuDftb` holding a whole
// 20x20 grid of displaced geometries (two H-bond junction coordinates per
// system, same convention as scripts/scan2d_*.rhai and
// test_gpu_dftb_gc_ptscan_pes_forces_vs_cpu). Heterogeneous replicas →
// per-replica convergence spread, the realistic production case.
// kT = 0.002 (production scan setting — mid-transfer gap ~0.5 mHa).
//
// Measured per system: scc() wall time (3 runs), eval(true) once, and a
// single-point CPU f64 (DftbCpu) time for the speedup estimate
// (batch CPU cost ≈ 400 × single-point).
//
// Run:
//   RUST_DFTB_SK_DIR=... cargo test --release --test gpu_scc_bench \
//       test_gpu_scc_scan400 -- --ignored --nocapture --test-threads=1

/// One junction coordinate: `pos[moved] = pos[donor] + u·d` where
/// u = normalize(pos[acc] − pos[donor]). acceptor==moved stretches the
/// donor–moved bond (used for systems without an H-bond).
type Junction = (usize, usize, usize);

struct ScanSystem {
    name: &'static str,
    file: &'static str,
    j1: Junction,
    j2: Junction,
}

const SCAN_SYSTEMS: &[ScanSystem] = &[
    // H2O: both O–H bond stretches.
    ScanSystem { name: "H2O",   file: "H2O.xyz",                      j1: (1, 0, 1),  j2: (2, 0, 2) },
    // formic dimer: two O–H···O protons (hbond_gpu_scc.rs indices).
    ScanSystem { name: "formic", file: "formic_dimer.xyz",            j1: (4, 3, 7),  j2: (9, 8, 2) },
    // 7-azaindole dimer: J1 N6–H21···N10, J2 N15–H27···N1 (scan2d_azaindol.rhai).
    ScanSystem { name: "azaindol", file: "azaindol_dimer.xyz",        j1: (21, 6, 10), j2: (27, 15, 1) },
    // GC: J1 N8–H13···N20 (PT scan), J2 N10–H15···O22 (2nd H-bond).
    ScanSystem { name: "GC",    file: "guanine-cytosine.xyz",         j1: (13, 8, 20), j2: (15, 10, 22) },
    // AT: J1 T-N3–H26···A-N1, J2 A-N10–H9···T-O21 (Watson-Crick).
    ScanSystem { name: "AT",    file: "adenine-thymine.xyz",          j1: (26, 19, 6), j2: (9, 10, 21) },
    // diazaphenalene dimer: N9–H20···N23, N30–H41···N2.
    ScanSystem { name: "diazaphen", file: "diazaphenalene_dimer.xyz", j1: (20, 9, 23), j2: (41, 30, 2) },
    // DiTetraceno-helicene: no H-bond — rim C–H stretches (throughput proxy).
    ScanSystem { name: "DTH",   file: "DiTetraceno_helicene_1a.xyz",  j1: (75, 33, 75), j2: (83, 79, 83) },
];

const NP: usize = 20;          // 20x20 grid → batch 400
const SCAN_D0: f64 = 1.0;      // Å
const SCAN_DSTEP: f64 = 0.05;  // Å

fn xyz_file(name: &str) -> rust_dftb::io::XyzMolecule {
    for p in [
        format!("data/xyz/{name}"),
        format!("{}/data/xyz/{name}", env!("CARGO_MANIFEST_DIR")),
        format!("{}/../data/xyz/{name}", env!("CARGO_MANIFEST_DIR")),
    ] {
        if let Ok(x) = parse_xyz(&p) { return x; }
    }
    panic!("cannot load {name}");
}

fn scan_geoms(base: &[[f64; 3]], j1: Junction, j2: Junction) -> Vec<[f64; 3]> {
    let mut all = Vec::with_capacity(NP * NP * base.len());
    for i1 in 0..NP {
        for i2 in 0..NP {
            let mut g = base.to_vec();
            for &(moved, donor, acc) in &[j1, j2] {
                let d = if (moved, donor, acc) == j1 { SCAN_D0 + i1 as f64 * SCAN_DSTEP }
                        else { SCAN_D0 + i2 as f64 * SCAN_DSTEP };
                let mut u = [0.0f64; 3];
                for c in 0..3 { u[c] = base[acc][c] - base[donor][c]; }
                let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                for c in 0..3 { g[moved][c] = base[donor][c] + u[c] / n * d; }
            }
            all.extend_from_slice(&g);
        }
    }
    all
}

/// Time one CPU f64 point (SCC + repulsive + forces) — the sequential
/// baseline the GPU batch replaces.
fn cpu_point_ms(sk: &rust_dftb::SkData, sk_dir: &str, species: &[String], coords: &[[f64; 3]]) -> f64 {
    let mut unique: Vec<String> = Vec::new();
    for s in species { if !unique.contains(s) { unique.push(s.clone()); } }
    let repulsive = parse_all_repulsive(sk_dir, &unique, unique.len()).unwrap();
    let mut cpu = DftbCpu::new(sk.clone(), species.to_vec()).unwrap();
    cpu.update_geometry(coords).unwrap();
    cpu.set_smearing(0.002);
    let t0 = std::time::Instant::now();
    cpu.reset_charges();
    cpu.solve_scc(200, 1e-8).unwrap();
    let scc = cpu.build_result();
    let _e = scc.energy + repulsive_energy(sk_dir, species, coords).unwrap();
    let _f = cpu.compute_forces(&scc, &repulsive).unwrap();
    t0.elapsed().as_secs_f64() * 1e3
}

#[test]
#[ignore]
fn test_gpu_scc_scan400_benchmark() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    eprintln!("=== 20x20 scan benchmark (batch={}): production GpuDftb, kT=0.002, tol=1e-6 ===", NP * NP);
    eprintln!("{:>11} {:>5} {:>5} {:>7} | {:>9} {:>5} {:>8} {:>9} | {:>8} | {:>9} {:>8}",
        "system", "atoms", "orbs", "batch", "scc_ms", "iters", "ms/iter", "sys/s", "evalF_ms", "cpu_1pt", "speedup");
    for sys in SCAN_SYSTEMS {
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let base = xyz.coords.clone();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let geoms = scan_geoms(&base, sys.j1, sys.j2);
        let n_atoms = sp.len();

        // CPU single-point baseline (base geometry).
        let cpu_ms = cpu_point_ms(&sk, &sk_dir, &sp, &base);

        for &batch in &[1usize, 100, NP * NP] {
            let coords = &geoms[..batch * n_atoms];
            let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), coords.to_vec(), batch)
                .unwrap_or_else(|e| panic!("GpuDftb::new {} batch={batch}: {e}", sys.name));
            eng.set_smearing(0.002);
            let mut total = 0.0f64;
            let mut iters = 0usize;
            let mut n_failed = 0usize;
            for _ in 0..N_RUNS {
                eng.reset_q0().unwrap();
                let t0 = std::time::Instant::now();
                let s = eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("scc {} batch={batch}: {e}", sys.name));
                total += t0.elapsed().as_secs_f64() * 1e3;
                iters = s.n_iters;
                // Failed replicas are data, not a crash — a stalled replica
                // costs its 100 iters, which is the production cost anyway.
                n_failed = s.statuses.iter().filter(|st| **st == SccStatus::Failed).count();
            }
            let t0 = std::time::Instant::now();
            eng.eval(true).unwrap();
            let eval_ms = t0.elapsed().as_secs_f64() * 1e3;
            let scc_ms = total / N_RUNS as f64;
            let speedup = cpu_ms * batch as f64 / scc_ms;
            let n_orbs = eng.n();
            eprintln!("{:>11} {:>5} {:>5} {:>7} | {:>9.2} {:>5} {:>8.3} {:>9.1} | {:>8.2} | {:>9.2} {:>7.1}x  failed={n_failed}",
                sys.name, n_atoms, n_orbs, batch, scc_ms, iters,
                scc_ms / iters.max(1) as f64, batch as f64 / (scc_ms / 1e3), eval_ms,
                cpu_ms, speedup);
        }
    }
}
