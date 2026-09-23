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

extern "C" {
    fn openblas_set_num_threads(n: i32);
    fn openblas_get_num_threads() -> i32;
    fn openblas_get_num_procs() -> i32;
    fn openblas_get_parallel() -> i32;
    fn openblas_get_config() -> *const libc_char;
}
// Avoid a libc dependency for one C string.
#[allow(non_camel_case_types)]
type libc_char = i8;

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
    for _ in 0..batch {
        all_coords.extend_from_slice(coords);
    }

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
        eprintln!(
            "  run {r}: {dt:.2} ms ({} iters, rms={:.3e})",
            scc.n_iters, scc.rms
        );
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
        format!(
            "{}/../data/xyz/formic_dimer.xyz",
            env!("CARGO_MANIFEST_DIR")
        ),
    ];
    let mut xyz = None;
    for path in &candidates {
        if let Ok(x) = parse_xyz(path) {
            xyz = Some(x);
            break;
        }
    }
    let xyz = match xyz {
        Some(x) => x,
        None => {
            eprintln!("Skipping: cannot load formic_dimer.xyz");
            return;
        }
    };

    let sk = load_sk_for_species(&sk_dir, &xyz.species).unwrap();
    eprintln!("Formic dimer: N_atoms={}", xyz.species.len());

    eprintln!("\n=== GPU SCC benchmark (production GpuDftb) ===");
    for &batch in &[1usize, 8, 32] {
        bench_dftb(
            &sk,
            &sk_dir,
            &xyz.species,
            &xyz.coords,
            batch,
            &format!("formic_dimer"),
        );
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
    ScanSystem {
        name: "H2O",
        file: "H2O.xyz",
        j1: (1, 0, 1),
        j2: (2, 0, 2),
    },
    // formic dimer: two O–H···O protons (hbond_gpu_scc.rs indices).
    ScanSystem {
        name: "formic",
        file: "formic_dimer.xyz",
        j1: (4, 3, 7),
        j2: (9, 8, 2),
    },
    // 7-azaindole dimer: J1 N6–H21···N10, J2 N15–H27···N1 (scan2d_azaindol.rhai).
    ScanSystem {
        name: "azaindol",
        file: "azaindol_dimer.xyz",
        j1: (21, 6, 10),
        j2: (27, 15, 1),
    },
    // GC: J1 N8–H13···N20 (PT scan), J2 N10–H15···O22 (2nd H-bond).
    ScanSystem {
        name: "GC",
        file: "guanine-cytosine.xyz",
        j1: (13, 8, 20),
        j2: (15, 10, 22),
    },
    // AT: J1 T-N3–H26···A-N1, J2 A-N10–H9···T-O21 (Watson-Crick).
    ScanSystem {
        name: "AT",
        file: "adenine-thymine.xyz",
        j1: (26, 19, 6),
        j2: (9, 10, 21),
    },
    // diazaphenalene dimer: N9–H20···N23, N30–H41···N2.
    ScanSystem {
        name: "diazaphen",
        file: "diazaphenalene_dimer.xyz",
        j1: (20, 9, 23),
        j2: (41, 30, 2),
    },
    // DiTetraceno-helicene: no H-bond — rim C–H stretches (throughput proxy).
    ScanSystem {
        name: "DTH",
        file: "DiTetraceno_helicene_1a.xyz",
        j1: (75, 33, 75),
        j2: (83, 79, 83),
    },
];

const SCAN_D0: f64 = 1.0; // Å
const SCAN_DSTEP: f64 = 0.05; // Å

fn xyz_file(name: &str) -> rust_dftb::io::XyzMolecule {
    for p in [
        format!("data/xyz/{name}"),
        format!("{}/data/xyz/{name}", env!("CARGO_MANIFEST_DIR")),
        format!("{}/../data/xyz/{name}", env!("CARGO_MANIFEST_DIR")),
    ] {
        if let Ok(x) = parse_xyz(&p) {
            return x;
        }
    }
    panic!("cannot load {name}");
}

fn scan_geoms(base: &[[f64; 3]], j1: Junction, j2: Junction, np: usize) -> Vec<[f64; 3]> {
    let mut all = Vec::with_capacity(np * np * base.len());
    for i1 in 0..np {
        for i2 in 0..np {
            let mut g = base.to_vec();
            for &(moved, donor, acc) in &[j1, j2] {
                // Same physical window as the original 20×20 grid (d ∈ [1.0, 1.95] Å).
            // A larger np only densifies that window — stretching d past ~2 Å
            // drives overlap eigenvalues negative on these junctions.
            let span = 19.0 * SCAN_DSTEP;
            let step = if np <= 1 { 0.0 } else { span / (np - 1) as f64 };
            let d = if (moved, donor, acc) == j1 {
                    SCAN_D0 + i1 as f64 * step
                } else {
                    SCAN_D0 + i2 as f64 * step
                };
                let mut u = [0.0f64; 3];
                for c in 0..3 {
                    u[c] = base[acc][c] - base[donor][c];
                }
                let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                for c in 0..3 {
                    g[moved][c] = base[donor][c] + u[c] / n * d;
                }
            }
            all.extend_from_slice(&g);
        }
    }
    all
}

/// Time one CPU f64 point (SCC + repulsive + forces) — the sequential
/// baseline the GPU batch replaces.
/// Equilibrium single point at the same kT as the GPU run.
/// Returns (wall ms, electronic+repulsive energy, Mulliken charges).
fn cpu_point(
    sk: &rust_dftb::SkData,
    sk_dir: &str,
    species: &[String],
    coords: &[[f64; 3]],
    kt: f64,
) -> (f64, f64, Vec<f64>) {
    let mut unique: Vec<String> = Vec::new();
    for s in species {
        if !unique.contains(s) {
            unique.push(s.clone());
        }
    }
    let repulsive = parse_all_repulsive(sk_dir, &unique, unique.len()).unwrap();
    let mut cpu = DftbCpu::new(sk.clone(), species.to_vec()).unwrap();
    cpu.update_geometry(coords).unwrap();
    cpu.set_smearing(kt);
    let t0 = std::time::Instant::now();
    cpu.reset_charges();
    cpu.solve_scc(200, 1e-8).unwrap();
    let scc = cpu.build_result();
    let e = scc.energy + repulsive_energy(sk_dir, species, coords).unwrap();
    let _f = cpu.compute_forces(&scc, &repulsive).unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!("  CPU ref: E={e:.6} Ha  kT={kt}  {ms:.1} ms  q={:?}", scc.charges);
    (ms, e, scc.charges)
}

/// `nthreads` operating-system threads, each with its own `DftbCpu`.
/// `blas_threads` is process-global and must be set only while no BLAS
/// call is in flight — OpenBLAS's pthread build has one pool, and
/// calling the setter from inside a worker restarts that pool.
fn cpu_batch_ms(
    sk: &rust_dftb::SkData,
    species: &[String],
    coords: &[[f64; 3]],
    batch: usize,
    kt: f64,
    nthreads: usize,
    blas_threads: i32,
) -> f64 {
    let nthreads = nthreads.max(1).min(batch);
    let coords = coords.to_vec();
    unsafe { openblas_set_num_threads(blas_threads) };
    let blas_now = unsafe { openblas_get_num_threads() };
    let t0 = std::time::Instant::now();
    std::thread::scope(|scope| {
        for t in 0..nthreads {
            let sk = sk.clone();
            let species = species.to_vec();
            let coords = coords.clone();
            scope.spawn(move || {
                let mut cpu = DftbCpu::new(sk, species).unwrap_or_else(|e| {
                    panic!("DftbCpu::new thread {t}: {e}")
                });
                cpu.set_smearing(kt);
                for s in (t..batch).step_by(nthreads) {
                    cpu.update_geometry(&coords).unwrap_or_else(|e| {
                        panic!("update_geometry thread {t} sys {s}: {e}")
                    });
                    cpu.reset_charges();
                    cpu.solve_scc(200, 1e-8).unwrap_or_else(|e| {
                        panic!("solve_scc thread {t} sys {s}: {e}")
                    });
                }
            });
        }
    });
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "  CPU os={nthreads} blas={blas_now}: {batch}× equilibrium  {ms:.1} ms  {:.1} sys/s",
        batch as f64 / (ms / 1e3)
    );
    ms
}

#[test]
#[ignore]
fn test_gpu_scc_scan400_benchmark() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    unsafe {
        let cfg = openblas_get_config();
        let cfg = std::ffi::CStr::from_ptr(cfg as *const std::ffi::c_char);
        eprintln!(
            "OpenBLAS config={} parallel={} (0=serial,1=OpenMP,2=pthread) procs={}",
            cfg.to_string_lossy(),
            openblas_get_parallel(),
            openblas_get_num_procs()
        );
    }
    eprintln!("=== scan benchmark: production GpuDftb, tol=1e-6 ===");
    // §16.D: optional subset filter for A/B runs (comma list of names).
    let only: Option<Vec<String>> = std::env::var("RUST_DFTB_BENCH_SYSTEMS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    eprintln!(
        "{:>11} {:>5} {:>5} {:>7} | {:>9} {:>5} {:>8} {:>9} | {:>8} | {:>9} {:>8}",
        "system",
        "atoms",
        "orbs",
        "batch",
        "scc_ms",
        "iters",
        "ms/iter",
        "sys/s",
        "evalF_ms",
        "cpu_1pt",
        "speedup"
    );
    for sys in SCAN_SYSTEMS {
        if let Some(o) = &only {
            if !o.iter().any(|x| x == sys.name) {
                continue;
            }
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let base = xyz.coords.clone();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let kt: f32 = std::env::var("RUST_DFTB_BENCH_KT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.002);
        let batches: Vec<usize> = std::env::var("RUST_DFTB_BENCH_BATCHES")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect()
            })
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| vec![1, 100, 400]);
        let n_runs: usize = std::env::var("RUST_DFTB_BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(N_RUNS);
        let np = batches.iter().copied().max().unwrap_or(400).max(400);
        let np = (np as f64).sqrt().ceil() as usize;
        let geoms = scan_geoms(&base, sys.j1, sys.j2, np);
        let n_atoms = sp.len();

        // CPU single-point baseline — equilibrium geometry, same kT.
        let (cpu_ms, cpu_e, cpu_q) = cpu_point(&sk, &sk_dir, &sp, &base, kt as f64);
        {
            let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), base.clone(), 1)
                .unwrap_or_else(|e| panic!("GpuDftb::new {} accuracy: {e}", sys.name));
            eng.set_smearing(kt as f32);
            let s = eng
                .scc(100, 1e-6)
                .unwrap_or_else(|e| panic!("scc {} accuracy: {e}", sys.name));
            let q = eng.plan.read_charges(&eng.rt).unwrap();
            let mut dq = 0.0f64;
            for (a, &qc) in cpu_q.iter().enumerate() {
                dq = dq.max((q[a] as f64 - qc).abs());
            }
            eprintln!(
                "  ACCURACY {} equilibrium kT={kt}: max|Δq|={dq:.3e} e   SCC rms={:.3e} iters={} status={:?}   CPU E={cpu_e:.6} Ha  (Mulliken of the SCC density, before finalize)",
                sys.name, s.rms, s.n_iters, s.statuses.first()
            );
        }

        // Eight copies: one per physical core. Three layouts —
        //   os=1 blas=1   one core, the denominator
        //   os=1 blas=8   one diagonalization, OpenBLAS spreads it
        //   os=8 blas=1   eight independent diagonalizations
        let cpu_n = 8usize;
        let one_ms = cpu_batch_ms(&sk, &sp, &base, cpu_n, kt as f64, 1, 1) / cpu_n as f64;
        let wide = cpu_batch_ms(&sk, &sp, &base, cpu_n, kt as f64, 1, 8) / cpu_n as f64;
        let indep = cpu_batch_ms(&sk, &sp, &base, cpu_n, kt as f64, 8, 1) / cpu_n as f64;
        eprintln!(
            "  CPU layouts {}  1×blas1 {:.1} ms/sys | 1×blas8 {:.1} ms/sys ({:.1}×) | 8×blas1 {:.1} ms/sys ({:.1}×)",
            sys.name,
            one_ms,
            wide,
            one_ms / wide,
            indep,
            one_ms / indep
        );
        let _ = cpu_ms;

        for &batch in &batches {
            if batch * n_atoms > geoms.len() {
                eprintln!("  skip {batch}: grid has {} geometries", geoms.len() / n_atoms);
                continue;
            }
            let coords = &geoms[..batch * n_atoms];
            let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), coords.to_vec(), batch)
                .unwrap_or_else(|e| panic!("GpuDftb::new {} batch={batch}: {e}", sys.name));
            eng.set_smearing(kt as f32);
            let mut total = 0.0f64;
            let mut iters = 0usize;
            let mut n_failed = 0usize;
            for _ in 0..n_runs {
                eng.reset_q0().unwrap();
                let t0 = std::time::Instant::now();
                let s = eng
                    .scc(100, 1e-6)
                    .unwrap_or_else(|e| panic!("scc {} batch={batch}: {e}", sys.name));
                total += t0.elapsed().as_secs_f64() * 1e3;
                iters = s.n_iters;
                // Failed replicas are data, not a crash — a stalled replica
                // costs its 100 iters, which is the production cost anyway.
                n_failed = s
                    .statuses
                    .iter()
                    .filter(|st| **st == SccStatus::Failed)
                    .count();
            }
            if std::env::var("RUST_DFTB_PROF").is_ok() {
                eng.prof_report(&format!("batch={batch}"));
            }
            let t0 = std::time::Instant::now();
            let ev = eng.eval(true).unwrap();
            let eval_ms = t0.elapsed().as_secs_f64() * 1e3;
            if std::env::var("RUST_DFTB_BENCH_Q").is_ok() {
                let mut q = vec![0.0f32; n_atoms];
                eng.rt.read_buffer(&eng.plan.q_gpu, &mut q).unwrap();
                eprintln!("  q0(replica0): {q:?}");
            }
            let scc_ms = total / n_runs as f64;
            let speedup = cpu_ms * batch as f64 / scc_ms;
            let n_orbs = eng.n();
            eprintln!("{:>11} {:>5} {:>5} {:>7} | {:>9.2} {:>5} {:>8.3} {:>9.1} | {:>8.2} | {:>9.2} {:>7.1}x  failed={n_failed} E0={:.6}",
                sys.name, n_atoms, n_orbs, batch, scc_ms, iters,
                scc_ms / iters.max(1) as f64, batch as f64 / (scc_ms / 1e3), eval_ms,
                cpu_ms, speedup, ev.energy[0]);
        }
    }
}

/// One small step along the forces of the optimized geometry.
/// rms(|δr|) = `rms_ang` Å. Direction is the Cartesian force.
fn step_along_forces(coords: &[[f64; 3]], forces: &[[f64; 3]], rms_ang: f64) -> Vec<[f64; 3]> {
    assert_eq!(coords.len(), forces.len());
    let na = coords.len() as f64;
    let mut n2 = 0.0f64;
    for f in forces {
        n2 += f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
    }
    let n = n2.sqrt().max(1e-30);
    let scale = rms_ang * na.sqrt() / n;
    coords
        .iter()
        .zip(forces)
        .map(|(r, f)| [r[0] + scale * f[0], r[1] + scale * f[1], r[2] + scale * f[2]])
        .collect()
}

fn replicate(coords: &[[f64; 3]], batch: usize) -> Vec<[f64; 3]> {
    let mut all = Vec::with_capacity(batch * coords.len());
    for _ in 0..batch {
        all.extend_from_slice(coords);
    }
    all
}

/// Eight independent `dsyevd`s, one BLAS thread each.
/// `seed` Some = warm charges from the G0 solution; None = atomic q0.
/// Returns (wall ms, iters on thread 0's last system).
fn cpu_geom_step_ms(
    sk: &rust_dftb::SkData,
    species: &[String],
    g1: &[[f64; 3]],
    batch: usize,
    kt: f64,
    nthreads: usize,
    seed: Option<&[f64]>,
) -> (f64, usize) {
    let nthreads = nthreads.max(1).min(batch);
    let g1 = g1.to_vec();
    let seed = seed.map(|q| q.to_vec());
    unsafe { openblas_set_num_threads(1) };
    let iters = std::sync::Mutex::new(0usize);
    let t0 = std::time::Instant::now();
    std::thread::scope(|scope| {
        for t in 0..nthreads {
            let sk = sk.clone();
            let species = species.to_vec();
            let g1 = g1.clone();
            let seed = seed.clone();
            let iters = &iters;
            scope.spawn(move || {
                let mut cpu = DftbCpu::new(sk, species).unwrap_or_else(|e| {
                    panic!("DftbCpu::new thread {t}: {e}")
                });
                cpu.set_smearing(kt);
                let mut last = 0usize;
                for s in (t..batch).step_by(nthreads) {
                    cpu.update_geometry(&g1).unwrap_or_else(|e| {
                        panic!("update_geometry thread {t} sys {s}: {e}")
                    });
                    if let Some(q) = &seed {
                        cpu.set_charges(q);
                    } else {
                        cpu.reset_charges();
                    }
                    cpu.solve_scc(200, 1e-8).unwrap_or_else(|e| {
                        panic!("solve_scc thread {t} sys {s}: {e}")
                    });
                    last = cpu.n_scc_iter;
                }
                if t == 0 {
                    *iters.lock().unwrap() = last;
                }
            });
        }
    });
    let n_iter = *iters.lock().unwrap();
    (t0.elapsed().as_secs_f64() * 1e3, n_iter)
}

struct WarmRow {
    sys: String,
    n: usize,
    method: &'static str,
    ms: f64,
    iters: usize,
    failed: usize,
    dq: f64,
}

fn q_delta(eng: &mut GpuDftb, q_ref: &[f64], n_atoms: usize) -> f64 {
    let mut q = vec![0.0f32; n_atoms];
    eng.rt.read_buffer(&eng.plan.q_gpu, &mut q).unwrap();
    let mut m = 0.0f64;
    for a in 0..n_atoms {
        m = m.max((q[a] as f64 - q_ref[a]).abs());
    }
    m
}

fn gpu_scc_timed(eng: &mut GpuDftb, rms_tol: f32) -> (f64, usize, usize) {
    let t0 = std::time::Instant::now();
    let s = eng.scc(100, rms_tol).unwrap_or_else(|e| panic!("scc: {e}"));
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let failed = s.statuses.iter().filter(|st| **st == SccStatus::Failed).count();
    (ms, s.n_iters, failed)
}

/// Geometry-step comparison. Converge at the xyz geometry, take one
/// 0.02 Å-rms step along the forces, time the next SCC. Every method
/// sees the same G1. Warm rows carry that method's own converged
/// charges (and, for purify, K). Cold rows start from atomic charges.
/// CPU is 8 OS threads × 1 BLAS thread. Not a random matrix.
#[test]
#[ignore]
fn test_gpu_warm_step_benchmark() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe {
        let cfg = openblas_get_config();
        let cfg = std::ffi::CStr::from_ptr(cfg as *const std::ffi::c_char);
        eprintln!(
            "OpenBLAS config={} parallel={} procs={}",
            cfg.to_string_lossy(),
            openblas_get_parallel(),
            openblas_get_num_procs()
        );
    }
    let batch: usize = std::env::var("RUST_DFTB_WARM_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let kt = 0.0;
    let only: Option<Vec<String>> = std::env::var("RUST_DFTB_BENCH_SYSTEMS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let want = ["formic", "GC", "diazaphen", "DTH"];
    let mut rows: Vec<WarmRow> = Vec::new();
    eprintln!("=== warm geometry step: batch={batch} kT=0 rms=0.02 Å ===");
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if let Some(o) = &only {
            if !o.iter().any(|x| x == sys.name) {
                continue;
            }
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) {
                unique.push(s.clone());
            }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.set_smearing(kt);
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        let scc = cpu.build_result();
        let q_g0 = scc.charges.clone();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let mut d2 = 0.0f64;
        for a in 0..n_atoms {
            for c in 0..3 {
                let d = g1[a][c] - g0[a][c];
                d2 += d * d;
            }
        }
        eprintln!(
            "-- {} atoms={n_atoms} step_rms={:.4} Å  G0 iters={}",
            sys.name,
            (d2 / n_atoms as f64).sqrt(),
            cpu.n_scc_iter
        );
        let mut cpu_w = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu_w.update_geometry(&g1).unwrap();
        cpu_w.set_smearing(kt);
        cpu_w.set_charges(&q_g0);
        cpu_w.solve_scc(200, 1e-8).unwrap_or_else(|e| panic!("CPU warm q {}: {e}", sys.name));
        let q_ref = cpu_w.charges.clone();

        let (ms, iters) = cpu_geom_step_ms(&sk, &sp, &g1, batch, kt, 8, None);
        rows.push(WarmRow { sys: sys.name.into(), n: 0, method: "cpu 8×1 cold", ms, iters, failed: 0, dq: 0.0 });
        let (ms, iters) = cpu_geom_step_ms(&sk, &sp, &g1, batch, kt, 8, Some(&q_g0));
        rows.push(WarmRow { sys: sys.name.into(), n: 0, method: "cpu 8×1 warm", ms, iters, failed: 0, dq: 0.0 });

        unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER") };
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g1, batch), batch)
            .unwrap_or_else(|e| panic!("jacobi cold {}: {e}", sys.name));
        eng.set_smearing(kt as f32);
        let (ms, iters, failed) = gpu_scc_timed(&mut eng, 1e-6);
        let dq = q_delta(&mut eng, &q_ref, n_atoms);
        let n_orbs = eng.n();
        rows.push(WarmRow { sys: sys.name.into(), n: n_orbs, method: "gpu jacobi cold", ms, iters, failed, dq });
        drop(eng);

        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("jacobi warm {}: {e}", sys.name));
        eng.set_smearing(kt as f32);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi G0 {}: {e}", sys.name));
        let t0 = std::time::Instant::now();
        eng.set_coords(&replicate(&g1, batch)).unwrap();
        let geom_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (ms, iters, failed) = gpu_scc_timed(&mut eng, 1e-3);
        eprintln!("  jacobi warm geom={geom_ms:.1} ms  scc={ms:.1} ms");
        let dq = q_delta(&mut eng, &q_ref, n_atoms);
        rows.push(WarmRow { sys: sys.name.into(), n: n_orbs, method: "gpu jacobi warm", ms, iters, failed, dq });
        drop(eng);

        unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify") };
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("purify {}: {e}", sys.name));
        eng.set_smearing(kt as f32);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("purify G0 {}: {e}", sys.name));
        let t0 = std::time::Instant::now();
        eng.set_coords_keep_k(&replicate(&g1, batch)).unwrap();
        let geom_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (ms, iters, failed) = gpu_scc_timed(&mut eng, 1e-6);
        eprintln!("  purify warm geom={geom_ms:.1} ms  scc={ms:.1} ms");
        let dq = q_delta(&mut eng, &q_ref, n_atoms);
        rows.push(WarmRow { sys: sys.name.into(), n: n_orbs, method: "gpu purify warm", ms, iters, failed, dq });
        eng.reset_q0().unwrap();
        let (ms, iters, failed) = gpu_scc_timed(&mut eng, 1e-6);
        let dq = q_delta(&mut eng, &q_ref, n_atoms);
        rows.push(WarmRow { sys: sys.name.into(), n: n_orbs, method: "gpu purify cold", ms, iters, failed, dq });
        drop(eng);
        for r in rows.iter_mut().rev() {
            if r.sys != sys.name {
                break;
            }
            if r.n == 0 {
                r.n = n_orbs;
            }
        }
    }
    eprintln!("\n| system | n | method | wall ms | iters | sys/s | failed | max\\|Δq\\| vs CPU warm |");
    eprintln!("|---|---|---|---|---|---|---|---|");
    for r in &rows {
        let rate = batch as f64 / (r.ms / 1e3);
        eprintln!(
            "| {} | {} | {} | {:.1} | {} | {:.1} | {} | {:.3e} |",
            r.sys, r.n, r.method, r.ms, r.iters, rate, r.failed, r.dq
        );
    }
}

fn csv_throughput(system: &str, n: usize, batch: usize, method: &str, ms: f64, iters: usize, failed: usize, dq: f64) {
    if std::env::var_os("RUST_DFTB_SWEEP_NOCSV").is_some() {
        eprintln!("  {system} n={n} batch={batch}  {method}  {ms:.1} ms  {iters} iters  failed={failed}  dq={dq:.3e}");
        return;
    }
    let dir = "/home/prokop/git/dftbplus/debug/dense_multi";
    std::fs::create_dir_all(dir).unwrap();
    let path = format!("{dir}/throughput.csv");
    let new = !std::path::Path::new(&path).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
    if new {
        use std::io::Write;
        writeln!(f, "system,n,batch,method,wall_ms,iters,sys_per_s,failed,dq").unwrap();
    }
    use std::io::Write;
    let rate = batch as f64 / (ms / 1e3);
    writeln!(f, "{system},{n},{batch},{method},{ms:.3},{iters},{rate:.2},{failed},{dq:.6e}").unwrap();
    eprintln!("  {system} n={n} batch={batch}  {method}  {ms:.1} ms  {iters} iters  {rate:.1} sys/s  failed={failed}  dq={dq:.3e}");
}

fn time_masked(eng: &mut GpuDftb, b: usize, bmax: usize, cold: bool, rms_tol: f32) -> (f64, usize, usize) {
    if cold {
        eng.plan.set_basis_warm(false);
        eng.reset_q0().unwrap();
    }
    eng.rt.finish().unwrap();
    let mask: Vec<bool> = (0..bmax).map(|i| i < b).collect();
    let t0 = std::time::Instant::now();
    let s = eng.scc_masked(100, rms_tol, &mask).unwrap_or_else(|e| panic!("scc_masked b={b}: {e}"));
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let failed = s.statuses.iter().take(b).filter(|st| **st == SccStatus::Failed).count();
    (ms, s.n_iters, failed)
}

/// Throughput vs batch. One engine at Bmax; smaller batches are the first
/// B replicas with the launch domain shrunk to B. The density guess is
/// restored before every warm point, so each batch starts from the same
/// G0 solution at the perturbed geometry. CPU "8 threads" is 8 OS threads
/// × 1 BLAS thread. CPU "1 thread" is a single dsyevd.
#[test]
#[ignore]
fn test_gpu_throughput_sweep() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let bmax: usize = std::env::var("RUST_DFTB_SWEEP_BMAX").ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let batches: Vec<usize> = std::env::var("RUST_DFTB_SWEEP_BATCHES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&b| b >= 1 && b <= bmax).collect())
        .unwrap_or_else(|| vec![8, 16, 32, 64, 128, 256, 400].into_iter().filter(|&b| b <= bmax).collect());
    let only = std::env::var("RUST_DFTB_BENCH_SYSTEMS").unwrap_or_else(|_| "formic".into());
    let sys = SCAN_SYSTEMS.iter().find(|s| s.name == only).unwrap_or_else(|| panic!("unknown system {only}"));
    let kt = 0.0f64;
    unsafe {
        openblas_set_num_threads(1);
        std::env::set_var("RUST_DFTB_SCC_QUIET", "1");
    }
    let xyz = xyz_file(sys.file);
    let sp = xyz.species.clone();
    let g0 = xyz.coords.clone();
    let n_atoms = g0.len();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let mut unique: Vec<String> = Vec::new();
    for s in &sp {
        if !unique.contains(s) { unique.push(s.clone()); }
    }
    let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
    let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
    cpu.update_geometry(&g0).unwrap();
    cpu.set_smearing(kt);
    cpu.reset_charges();
    cpu.solve_scc(200, 1e-8).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
    let scc = cpu.build_result();
    let q_g0 = scc.charges.clone();
    let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
    let g1 = step_along_forces(&g0, &forces.forces, 0.02);
    let mut cpu_w = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
    cpu_w.update_geometry(&g1).unwrap();
    cpu_w.set_smearing(kt);
    cpu_w.set_charges(&q_g0);
    cpu_w.solve_scc(200, 1e-8).unwrap();
    let q_ref = cpu_w.charges.clone();
    eprintln!("=== throughput sweep {} bmax={bmax} batches={batches:?} ===", sys.name);

    let (ms, iters) = cpu_geom_step_ms(&sk, &sp, &g1, 8, kt, 1, Some(&q_g0));
    csv_throughput(sys.name, 0, 8, "cpu 1 thread, warm", ms, iters, 0, 0.0);
    if ms < 800.0 {
        let (ms2, iters2) = cpu_geom_step_ms(&sk, &sp, &g1, 32, kt, 1, Some(&q_g0));
        csv_throughput(sys.name, 0, 32, "cpu 1 thread, warm", ms2, iters2, 0, 0.0);
    }
    let cpu_max: usize = std::env::var("RUST_DFTB_SWEEP_CPU_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let mut last_cpu = 0.0f64;
    for &b in &batches {
        if b < 8 || b > cpu_max { continue; }
        if last_cpu > 2000.0 { break; }
        let (ms, iters) = cpu_geom_step_ms(&sk, &sp, &g1, b, kt, 8, Some(&q_g0));
        csv_throughput(sys.name, 0, b, "cpu 8 threads, warm", ms, iters, 0, 0.0);
        let (ms_c, iters_c) = cpu_geom_step_ms(&sk, &sp, &g1, b, kt, 8, None);
        csv_throughput(sys.name, 0, b, "cpu 8 threads, cold", ms_c, iters_c, 0, 0.0);
        last_cpu = ms.max(ms_c);
    }

    if std::env::var("RUST_DFTB_SWEEP_SKIP_GPU").is_ok() { return; }
    let started = std::time::Instant::now();
    for (solver, label_w, label_c) in [("jacobi", "gpu jacobi warm", "gpu jacobi cold"), ("purify", "gpu purify warm", "gpu purify cold")] {
        if solver == "jacobi" {
            unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER") };
        } else {
            unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify") };
        }
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, bmax), bmax)
            .unwrap_or_else(|e| panic!("{solver} new: {e}"));
        eng.set_smearing(kt as f32);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("{solver} G0: {e}"));
        eng.set_coords_keep_k(&replicate(&g1, bmax)).unwrap();
        let guess = eng.plan.capture_density(&eng.rt).unwrap();
        let n_orbs = eng.n();
        for &b in &batches {
            if started.elapsed().as_secs() > 45 {
                eprintln!("sweep budget: stop before {solver} batch={b}");
                break;
            }
            eng.plan.restore_density(&eng.rt, &guess).unwrap();
            eng.plan.reset_diis(&eng.rt).unwrap();
            let prof = std::env::var_os("RUST_DFTB_SWEEP_PROF").is_some();
            let ktime = std::env::var("RUST_DFTB_KTIME").ok().as_deref() == Some("1");
            if prof { eng.prof_reset(); }
            if ktime { eng.plan.ktime_begin(); }
            // Jacobi warm 1e-3: |ΔE| ≤ 0.07 meV vs 1e-6 on formic/GC/diazaphen/DTH
            // (test_scc_rms_stop). Purify stays at 1e-6 — 1e-3 fails DTH.
            // Published throughput curves are still the 1e-6 grid.
            let warm_tol = if solver == "jacobi" { 1e-3 } else { 1e-6 };
            let (ms, iters, failed) = time_masked(&mut eng, b, bmax, false, warm_tol);
            if ktime { eng.plan.ktime_report(&format!("{label_w} wall={ms:.1} ms iters={iters}")); }
            if prof { eng.prof_report(&format!("{label_w} iters={iters}")); }
            let dq = q_delta(&mut eng, &q_ref, n_atoms);
            csv_throughput(sys.name, n_orbs, b, label_w, ms, iters, failed, dq);
            if prof { eng.prof_reset(); }
            if ktime { eng.plan.ktime_begin(); }
            let (ms, iters, failed) = time_masked(&mut eng, b, bmax, true, 1e-6);
            if ktime { eng.plan.ktime_report(&format!("{label_c} wall={ms:.1} ms iters={iters}")); }
            if prof { eng.prof_report(&format!("{label_c} iters={iters}")); }
            let dq = q_delta(&mut eng, &q_ref, n_atoms);
            csv_throughput(sys.name, n_orbs, b, label_c, ms, iters, failed, dq);
        }
        drop(eng);
    }
}

/// Same 0.02 Å force step as the throughput sweep, batch 8 (every copy is
/// the same molecule, so the iteration count matches the large-batch trace).
/// G0 is always converged to 1e-6. Only the geometry step uses the looser
/// charge RMS. Energy is one finalize after the solve, not a kernel inside
/// the SCC loop. ΔE is versus that method's own 1e-6 step.
#[test]
#[ignore]
fn test_scc_rms_stop() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    let kt = 0.0f64;
    let batch = 8usize;
    let tols = [1e-6f32, 1e-3, 3e-3];
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== SCC stop: batch={batch} kT=0 step=0.02 Å  tols={tols:?} ===");
    eprintln!("method system tol iters rms dq_vs_cpu dE_meV_vs_1e-6 failed");
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) {
                unique.push(s.clone());
            }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.set_smearing(kt);
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        let scc = cpu.build_result();
        let q_g0 = scc.charges.clone();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let mut cpu_w = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu_w.update_geometry(&g1).unwrap();
        cpu_w.set_smearing(kt);
        cpu_w.set_charges(&q_g0);
        cpu_w.solve_scc(200, 1e-8).unwrap_or_else(|e| panic!("CPU warm {}: {e}", sys.name));
        let q_ref = cpu_w.charges.clone();
        let e_cpu = cpu_w.build_result().energy;
        eprintln!("-- {} E_cpu={e_cpu:.8} Ha", sys.name);

        for (solver, keep_k) in [("jacobi", false), ("purify", true)] {
            if solver == "jacobi" {
                unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER") };
            } else {
                unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify") };
            }
            let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
                .unwrap_or_else(|e| panic!("{solver} new {}: {e}", sys.name));
            eng.set_smearing(kt as f32);
            eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("{solver} G0 {}: {e}", sys.name));
            if keep_k {
                eng.set_coords_keep_k(&replicate(&g1, batch)).unwrap();
            } else {
                eng.set_coords(&replicate(&g1, batch)).unwrap();
            }
            let guess = eng.plan.capture_density(&eng.rt).unwrap();
            let mut e_tight = 0.0f64;
            for (i, &tol) in tols.iter().enumerate() {
                eng.plan.restore_density(&eng.rt, &guess).unwrap();
                eng.plan.reset_diis(&eng.rt).unwrap();
                let s = eng.scc(100, tol).unwrap_or_else(|e| panic!("{solver} tol={tol}: {e}"));
                let dq = q_delta(&mut eng, &q_ref, n_atoms);
                let failed = s.statuses.iter().filter(|st| **st == SccStatus::Failed).count();
                let ev = eng.energy().unwrap_or_else(|e| panic!("{solver} energy: {e}"));
                let e = ev[0];
                if i == 0 {
                    e_tight = e;
                }
                let de_mev = (e - e_tight) * 27211.386;
                eprintln!(
                    "  {solver:7} tol={tol:.0e} iters={} rms={:.3e} dq={dq:.3e} dE={de_mev:+.4} meV failed={failed}",
                    s.n_iters, s.rms
                );
            }
            drop(eng);
        }
    }
}

/// Sparse B2 on the dense solver: up to two accepted commutator steps
/// (η starts at 8, halved when R_H rises), one McWeeny, no SCC loop.
/// G0 is a real purify solve. Energy and forces are a Jacobi finalize
/// at the bold step's own Mulliken charges.
#[test]
#[ignore]
fn test_geom_bold_dmm() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify"); }
    let want = ["formic", "GC"];
    eprintln!("=== geom bold: 2×DMM η=8 + 1 McWeeny, then stop ===");
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) {
                unique.push(s.clone());
            }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let moves: Vec<(&str, Vec<[f64; 3]>)> = vec![
            ("0.02Å-force", step_along_forces(&g0, &forces.forces, 0.02)),
            ("0.10Å-atom0", {
                let mut g = g0.clone();
                g[0][0] += 0.1;
                g
            }),
        ];
        for (tag, g1) in moves {
            let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, 1), 1)
                .unwrap_or_else(|e| panic!("new {}: {e}", sys.name));
            eng.set_smearing(0.0);
            eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0 scc {}: {e}", sys.name));
            eng.set_coords_keep_k(&g1).unwrap();
            let t0 = std::time::Instant::now();
            let (rh0, rh, tr, eta, n_acc) = eng.geom_bold_dmm().unwrap_or_else(|e| panic!("bold {tag}: {e}"));
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let q_new = eng.read_q_new().unwrap();
            let (e_cpu, f_cpu, q_cpu) = eng.cpu_ref().unwrap_or_else(|e| panic!("cpu_ref: {e}"));
            let mut dq = 0.0f64;
            for a in 0..n_atoms {
                dq = dq.max((q_new[a] as f64 - q_cpu[a]).abs());
            }
            eng.stage_mulliken().unwrap();
            let ev = eng.eval(true).unwrap_or_else(|e| panic!("eval: {e}"));
            let de_mev = (ev.energy[0] - e_cpu) * 27211.386;
            let f = ev.forces.as_ref().unwrap();
            let mut df = 0.0f64;
            let mut fmax = 0.0f64;
            for a in 0..n_atoms {
                for c in 0..3 {
                    let d = f[3 * a + c] as f64 - f_cpu[a][c];
                    df = df.max(d.abs());
                    fmax = fmax.max(f_cpu[a][c].abs());
                }
            }
            let tau = (tr as f64 - eng.n_occ() as f64).abs();
            eprintln!(
                "  {sys} {tag}  acc={n_acc} η={eta:.3}  R_H {rh0:.3e} → {rh:.3e}  Tr={tr:.4} τ={tau:.3e}  max|Δq|={dq:.3e}  dE={de_mev:+.3} meV  max|dF|={df:.3e}  max|F|={fmax:.3e}  {ms:.1} ms",
                sys = sys.name
            );
            assert!(tau < 0.05, "{} {tag} trace left Nocc: τ={tau:.4e}", sys.name);
            assert!(dq < 0.05, "{} {tag} Mulliken left the CPU charges: max|Δq|={dq:.3e}", sys.name);
        }
    }
}

/// One extra geometry step rebuilds H at the mixed charges and rotates
/// again. α is the mix between steps. The last step is scored at its
/// own Mulliken charges, same as the single-step test.
#[test]
#[ignore]
fn test_geom_bold_passes() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify"); }
    let recipes: &[(&str, usize, f32)] = &[
        ("1×  α=0.2", 1, 0.2),
        ("2×  α=0.2", 2, 0.2),
        ("3×  α=0.2", 3, 0.2),
        ("2×  α=1", 2, 1.0),
    ];
    eprintln!("=== bold passes: extra H rebuild + rotation, formic and GC ===");
    for sys in SCAN_SYSTEMS {
        if sys.name != "formic" && sys.name != "GC" {
            continue;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let moves = [
            ("0.02Å", step_along_forces(&g0, &forces.forces, 0.02)),
            ("0.10Å", { let mut g = g0.clone(); g[0][0] += 0.1; g }),
        ];
        for (tag, g1) in moves {
            let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, 1), 1)
                .unwrap_or_else(|e| panic!("new {}: {e}", sys.name));
            eng.set_smearing(0.0);
            eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
            eng.set_coords_keep_k(&g1).unwrap();
            let guess = eng.plan.capture_density(&eng.rt).unwrap();
            let (e_cpu, f_cpu, q_cpu) = eng.cpu_ref().unwrap();
            for &(label, npass, alpha) in recipes {
                eng.plan.restore_density(&eng.rt, &guess).unwrap();
                let t0 = std::time::Instant::now();
                let mut rh0 = 0.0f32;
                let mut rh = 0.0f32;
                let mut tr = 0.0f32;
                let mut n_acc = 0usize;
                for p in 0..npass {
                    let a = if p + 1 == npass { 0.0 } else { alpha };
                    let rep = eng.geom_bold_on_domain(a).unwrap_or_else(|e| panic!("{label}: {e}"));
                    if p == 0 { rh0 = rep.0; }
                    rh = rep.1;
                    tr = rep.2;
                    n_acc += rep.4;
                }
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let q_new = eng.read_q_new().unwrap();
                let mut dq = 0.0f64;
                for a in 0..n_atoms {
                    dq = dq.max((q_new[a] as f64 - q_cpu[a]).abs());
                }
                eng.stage_mulliken().unwrap();
                let ev = eng.eval(true).unwrap();
                let de_mev = (ev.energy[0] - e_cpu) * 27211.386;
                let f = ev.forces.as_ref().unwrap();
                let mut df = 0.0f64;
                let mut fmax = 0.0f64;
                for a in 0..n_atoms {
                    for c in 0..3 {
                        df = df.max((f[3 * a + c] as f64 - f_cpu[a][c]).abs());
                        fmax = fmax.max(f_cpu[a][c].abs());
                    }
                }
                let pct = 100.0 * df / fmax.max(1e-12);
                eprintln!(
                    "  {} {tag} {label}  dmm={n_acc}  R_H {rh0:.3e}→{rh:.3e}  Tr={tr:.4}  dq={dq:.3e}  dE={de_mev:+.3} meV  dF={pct:.2}%  {ms:.2} ms",
                    sys.name
                );
                let _ = (tr, n_acc);
            }
        }
    }
}

/// Throughput of the bold geometry step only. Jacobi and CPU rows already
/// live in throughput.csv. One purify G0, then the step at each batch.
#[test]
#[ignore]
fn test_gpu_bold_sweep() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let bmax: usize = std::env::var("RUST_DFTB_SWEEP_BMAX").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let batches: Vec<usize> = std::env::var("RUST_DFTB_SWEEP_BATCHES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&b| b >= 1 && b <= bmax).collect())
        .unwrap_or_else(|| vec![8, 16, 32, 64, 128, 256, 512, 1024].into_iter().filter(|&b| b <= bmax).collect());
    let only = std::env::var("RUST_DFTB_BENCH_SYSTEMS").unwrap_or_else(|_| "formic".into());
    // 1× is the step as measured. 2× takes the Mulliken charges (α = 1)
    // and rotates once more. α = 0.2 on the second pass made the 0.02 Å
    // forces worse, so that mix is not on the graph.
    let sys = SCAN_SYSTEMS.iter().find(|s| s.name == only).unwrap_or_else(|| panic!("unknown system {only}"));
    unsafe {
        openblas_set_num_threads(1);
        std::env::set_var("RUST_DFTB_SCC_QUIET", "1");
        std::env::set_var("RUST_DFTB_EIGSOLVER", "purify");
    }
    let xyz = xyz_file(sys.file);
    let sp = xyz.species.clone();
    let g0 = xyz.coords.clone();
    let n_atoms = g0.len();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let mut unique: Vec<String> = Vec::new();
    for s in &sp {
        if !unique.contains(s) { unique.push(s.clone()); }
    }
    let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
    let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
    cpu.update_geometry(&g0).unwrap();
    cpu.reset_charges();
    cpu.solve_scc(200, 1e-8).unwrap();
    let scc = cpu.build_result();
    let q_g0 = scc.charges.clone();
    let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
    let g1 = step_along_forces(&g0, &forces.forces, 0.02);
    let mut cpu_w = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
    cpu_w.update_geometry(&g1).unwrap();
    cpu_w.set_charges(&q_g0);
    cpu_w.solve_scc(200, 1e-8).unwrap();
    let q_ref = cpu_w.charges.clone();
    let recipes: &[(&str, usize, f32)] = &[("gpu bold", 1, 0.2), ("gpu bold x2", 2, 1.0)];
    eprintln!("=== bold sweep {} bmax={bmax} batches={batches:?} ===", sys.name);
    let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, bmax), bmax)
        .unwrap_or_else(|e| panic!("new: {e}"));
    eng.set_smearing(0.0);
    let t_g0 = std::time::Instant::now();
    eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0: {e}"));
    eprintln!("  G0 {:.1} ms", t_g0.elapsed().as_secs_f64() * 1e3);
    eng.set_coords_keep_k(&replicate(&g1, bmax)).unwrap();
    let guess = eng.plan.capture_density(&eng.rt).unwrap();
    let n_orbs = eng.n();
    let started = std::time::Instant::now();
    for &b in &batches {
        if started.elapsed().as_secs() > 40 {
            eprintln!("bold sweep budget: stop before batch={b}");
            break;
        }
        let ids: Vec<i32> = (0..b as i32).collect();
        for &(label, npass, alpha) in recipes {
            // Sub-millisecond steps jitter. Five repeats, median wall.
            let mut samples = Vec::with_capacity(5);
            let mut q_new = Vec::new();
            for _ in 0..5 {
                eng.plan.restore_density(&eng.rt, &guess).unwrap();
                eng.plan.set_work_domain(&eng.rt, &ids).unwrap();
                eng.rt.finish().unwrap();
                let t0 = std::time::Instant::now();
                for p in 0..npass {
                    let a = if p + 1 == npass { 0.0 } else { alpha };
                    eng.geom_bold_on_domain(a).unwrap_or_else(|e| panic!("{label} b={b}: {e}"));
                }
                eng.rt.finish().unwrap();
                samples.push(t0.elapsed().as_secs_f64() * 1e3);
                q_new = eng.read_q_new().unwrap();
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ms = samples[2];
            let mut dq = 0.0f64;
            for a in 0..n_atoms {
                dq = dq.max((q_new[a] as f64 - q_ref[a]).abs());
            }
            eng.plan.restore_work_domain(&eng.rt).unwrap();
            csv_throughput(sys.name, n_orbs, b, label, ms, npass, 0, dq);
        }
    }
}

/// Throughput of two Jacobi diagonalizations, the electronic cost of
/// Niklasson's first-level shadow once the charge Jacobian is already known.
/// Same batches and masked engine as the Jacobi sweep. Not a finite-difference
/// Jacobian, and not the bold commutator.
#[test]
#[ignore]
fn test_gpu_shadow_sweep() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let bmax: usize = std::env::var("RUST_DFTB_SWEEP_BMAX").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let batches: Vec<usize> = std::env::var("RUST_DFTB_SWEEP_BATCHES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&b| b >= 1 && b <= bmax).collect())
        .unwrap_or_else(|| vec![8, 16, 32, 64, 128, 256, 400, 512, 1024].into_iter().filter(|&b| b <= bmax).collect());
    let only = std::env::var("RUST_DFTB_BENCH_SYSTEMS").unwrap_or_else(|_| "formic".into());
    let sys = SCAN_SYSTEMS.iter().find(|s| s.name == only).unwrap_or_else(|| panic!("unknown system {only}"));
    unsafe {
        openblas_set_num_threads(1);
        std::env::set_var("RUST_DFTB_SCC_QUIET", "1");
        std::env::remove_var("RUST_DFTB_EIGSOLVER");
    }
    let xyz = xyz_file(sys.file);
    let sp = xyz.species.clone();
    let g0 = xyz.coords.clone();
    let n_atoms = g0.len();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let mut unique: Vec<String> = Vec::new();
    for s in &sp {
        if !unique.contains(s) { unique.push(s.clone()); }
    }
    let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
    let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
    cpu.update_geometry(&g0).unwrap();
    cpu.reset_charges();
    cpu.solve_scc(200, 1e-8).unwrap();
    let scc = cpu.build_result();
    let q_g0 = scc.charges.clone();
    let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
    let g1 = step_along_forces(&g0, &forces.forces, 0.02);
    let mut cpu_w = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
    cpu_w.update_geometry(&g1).unwrap();
    cpu_w.set_charges(&q_g0);
    cpu_w.solve_scc(200, 1e-8).unwrap();
    let q_ref = cpu_w.charges.clone();
    eprintln!("=== shadow sweep {} bmax={bmax} batches={batches:?} (2 Jacobi, Jacobian known) ===", sys.name);
    let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, bmax), bmax)
        .unwrap_or_else(|e| panic!("new: {e}"));
    eng.set_smearing(0.0);
    eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0: {e}"));
    eng.set_coords_keep_k(&replicate(&g1, bmax)).unwrap();
    let mut q_carried = vec![0.0f32; bmax * n_atoms];
    eng.rt.read_buffer(&eng.plan.q_gpu, &mut q_carried).unwrap();
    let guess = eng.plan.capture_density(&eng.rt).unwrap();
    let n_orbs = eng.n();
    let started = std::time::Instant::now();
    for &b in &batches {
        if started.elapsed().as_secs() > 40 {
            eprintln!("shadow sweep budget: stop before batch={b}");
            break;
        }
        let mask: Vec<bool> = (0..bmax).map(|i| i < b).collect();
        let mut samples = Vec::with_capacity(5);
        let mut n_it = 0usize;
        for _ in 0..5 {
            eng.plan.restore_density(&eng.rt, &guess).unwrap();
            eng.plan.reset_diis(&eng.rt).unwrap();
            eng.set_charges(&q_carried).unwrap();
            eng.rt.finish().unwrap();
            let t0 = std::time::Instant::now();
            let s = eng.scc_masked(2, 1e-12, &mask).unwrap_or_else(|e| panic!("shadow b={b}: {e}"));
            eng.rt.finish().unwrap();
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
            n_it = s.n_iters;
        }
        samples.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let ms = samples[2];
        let dq = q_delta(&mut eng, &q_ref, n_atoms);
        csv_throughput(sys.name, n_orbs, b, "gpu shadow (2 Jacobi)", ms, n_it, 0, dq);
    }
}

/// Throughput of six DIIS passes of one commutator, with a second
/// commutator on the same H when R_H > 0.3 × the charge rms. Jacobi rows
/// are already in throughput.csv. One system per process
/// (`RUST_DFTB_BENCH_SYSTEMS`).
#[test]
#[ignore]
fn test_gpu_bold_diis_sweep() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let bmax: usize = std::env::var("RUST_DFTB_SWEEP_BMAX").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let batches: Vec<usize> = std::env::var("RUST_DFTB_SWEEP_BATCHES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&b| b >= 1 && b <= bmax).collect())
        .unwrap_or_else(|| vec![8, 16, 32, 64, 128, 256, 512, 1024].into_iter().filter(|&b| b <= bmax).collect());
    let only = std::env::var("RUST_DFTB_BENCH_SYSTEMS").unwrap_or_else(|_| "formic".into());
    let sys = SCAN_SYSTEMS.iter().find(|s| s.name == only).unwrap_or_else(|| panic!("unknown system {only}"));
    let passes = 6usize;
    unsafe {
        openblas_set_num_threads(1);
        std::env::set_var("RUST_DFTB_SCC_QUIET", "1");
        std::env::set_var("RUST_DFTB_EIGSOLVER", "purify");
    }
    let xyz = xyz_file(sys.file);
    let sp = xyz.species.clone();
    let g0 = xyz.coords.clone();
    let n_atoms = g0.len();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let mut unique: Vec<String> = Vec::new();
    for s in &sp {
        if !unique.contains(s) { unique.push(s.clone()); }
    }
    let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
    let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
    cpu.update_geometry(&g0).unwrap();
    cpu.reset_charges();
    cpu.solve_scc(200, 1e-8).unwrap();
    let scc = cpu.build_result();
    let q_g0 = scc.charges.clone();
    let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
    let g1 = step_along_forces(&g0, &forces.forces, 0.02);
    let mut cpu_w = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
    cpu_w.update_geometry(&g1).unwrap();
    cpu_w.set_charges(&q_g0);
    cpu_w.solve_scc(200, 1e-8).unwrap();
    let q_ref = cpu_w.charges.clone();
    eprintln!("=== bold+DIIS sweep {} bmax={bmax} passes={passes} batches={batches:?} ===", sys.name);
    let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, bmax), bmax)
        .unwrap_or_else(|e| panic!("new: {e}"));
    eng.set_smearing(0.0);
    let t_g0 = std::time::Instant::now();
    eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0: {e}"));
    eprintln!("  G0 {:.1} ms", t_g0.elapsed().as_secs_f64() * 1e3);
    eng.set_coords_keep_k(&replicate(&g1, bmax)).unwrap();
    let guess = eng.plan.capture_density(&eng.rt).unwrap();
    let n_orbs = eng.n();
    let started = std::time::Instant::now();
    for &b in &batches {
        if started.elapsed().as_secs() > 35 {
            eprintln!("diis sweep budget: stop before batch={b}");
            break;
        }
        let ids: Vec<i32> = (0..b as i32).collect();
        let mut samples = Vec::with_capacity(3);
        let mut q_new = Vec::new();
        let mut comm = 0usize;
        let mut q_in = vec![0.0f32; n_atoms];
        let mut q_out = vec![0.0f32; n_atoms];
        for _ in 0..3 {
            eng.plan.restore_density(&eng.rt, &guess).unwrap();
            eng.plan.set_work_domain(&eng.rt, &ids).unwrap();
            eng.plan.reset_diis(&eng.rt).unwrap();
            eng.plan.activate_all(&eng.rt).unwrap();
            eng.rt.finish().unwrap();
            let t0 = std::time::Instant::now();
            let mut n_comm = 0usize;
            for _p in 0..passes {
                let (_rh0, mut rh, _tr, _eta, n_acc) = eng.plan.geom_bold_dmm_capped(&mut eng.rt, 1)
                    .unwrap_or_else(|e| panic!("{} b={b}: {e}", sys.name));
                n_comm += 1;
                if n_acc < 1 {
                    break;
                }
                eng.rt.read_buffer(&eng.plan.q_gpu, &mut q_in).unwrap();
                eng.rt.read_buffer(&eng.plan.q_new, &mut q_out).unwrap();
                let mut acc = 0.0f64;
                for a in 0..n_atoms {
                    let d = q_out[a] as f64 - q_in[a] as f64;
                    acc += d * d;
                }
                let rms_q = (acc / n_atoms as f64).sqrt() as f32;
                if rh > 0.3 * rms_q.max(1e-8) {
                    let (_a, rh2, _b2, _c, n2) = eng.plan.geom_bold_dmm_capped(&mut eng.rt, 1)
                        .unwrap_or_else(|e| panic!("{} extra b={b}: {e}", sys.name));
                    rh = rh2;
                    n_comm += 1;
                    if n2 < 1 {
                        break;
                    }
                }
                eng.plan.diis_on_qnew(&mut eng.rt).unwrap_or_else(|e| panic!("diis b={b}: {e}"));
                let _ = rh;
            }
            eng.rt.finish().unwrap();
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
            comm = n_comm;
            q_new = eng.read_q_new().unwrap();
        }
        samples.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let ms = samples[1];
        let mut dq = 0.0f64;
        for a in 0..n_atoms {
            dq = dq.max((q_new[a] as f64 - q_ref[a]).abs());
        }
        let rate = b as f64 / (ms * 1e-3);
        eprintln!("  {} b={b}  {ms:.2} ms  {rate:.0} sys/s  comm={comm}  dq={dq:.3e}", sys.name);
        eng.plan.restore_work_domain(&eng.rt).unwrap();
        csv_throughput(sys.name, n_orbs, b, "gpu bold diis", ms, passes, 0, dq);
    }
}

/// Fixed cap on accepted commutator steps, one Hamiltonian (the charges
/// carried from the previous geometry). Batch 256. Each N is the same
/// rotation stopped after N accepted steps, then one McWeeny.
/// `dq_H` is against one Jacobi diagonalization of that same H.
/// `dq_scc`, energy and forces are against a Jacobi SCC at the new geometry.
#[test]
#[ignore]
fn test_geom_bold_nacc() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    let batch = 256usize;
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== bold N=1..8 at batch {batch}, one H, then one McWeeny ===");
    eprintln!("system N acc R_H Tr dq_H dq_scc dE_meV dF%");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 45 {
            eprintln!("nacc budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let g1b = replicate(&g1, batch);

        unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER") };
        let mut jac = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("jacobi {}: {e}", sys.name));
        jac.set_smearing(0.0);
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi G0 {}: {e}", sys.name));
        jac.set_coords(&g1b).unwrap();
        jac.eval(false).unwrap_or_else(|e| panic!("jacobi same-H {}: {e}", sys.name));
        let q_h = jac.read_q_new().unwrap();
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi SCC {}: {e}", sys.name));
        let ev_j = jac.eval(true).unwrap_or_else(|e| panic!("jacobi eval {}: {e}", sys.name));
        let q_scc = jac.read_q_new().unwrap();
        let e_scc = ev_j.energy[0];
        let f_scc = ev_j.forces.unwrap();
        drop(jac);

        unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify") };
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("purify {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("purify G0 {}: {e}", sys.name));
        eng.set_coords_keep_k(&g1b).unwrap();
        let guess = eng.plan.capture_density(&eng.rt).unwrap();
        for n in 1..=8 {
            eng.plan.restore_density(&eng.rt, &guess).unwrap();
            let (rh0, rh, tr, eta, n_acc) = eng.geom_bold_capped(n).unwrap_or_else(|e| panic!("N={n}: {e}"));
            let q_new = eng.read_q_new().unwrap();
            let mut dq_h = 0.0f64;
            let mut dq_scc = 0.0f64;
            for a in 0..n_atoms {
                dq_h = dq_h.max((q_new[a] as f64 - q_h[a] as f64).abs());
                dq_scc = dq_scc.max((q_new[a] as f64 - q_scc[a] as f64).abs());
            }
            eng.stage_mulliken().unwrap();
            let ev = eng.eval(true).unwrap_or_else(|e| panic!("eval N={n}: {e}"));
            let de_mev = (ev.energy[0] - e_scc) * 27211.386;
            let f = ev.forces.as_ref().unwrap();
            let mut df = 0.0f64;
            let mut fmax = 0.0f64;
            for i in 0..3 * n_atoms {
                df = df.max((f[i] as f64 - f_scc[i] as f64).abs());
                fmax = fmax.max(f_scc[i].abs() as f64);
            }
            let pct = 100.0 * df / fmax.max(1e-12);
            eprintln!(
                "  {}  N={n} acc={n_acc} η={eta:.3}  R_H {rh0:.3e}→{rh:.3e}  Tr={tr:.4}  dq_H={dq_h:.3e}  dq_scc={dq_scc:.3e}  dE={de_mev:+.3} meV  dF={pct:.2}%",
                sys.name
            );
            if n_acc < n {
                eprintln!("  {}  trust region stopped at {n_acc} accepted steps", sys.name);
                break;
            }
        }
    }
}

/// One accepted commutator and one McWeeny, then the charges replace q
/// and the next round rebuilds H. N is the number of such rounds.
/// N = 1 is a single round on the carried Hamiltonian. A second round
/// is kept only when it is closer to the Jacobi SCC than the first.
#[test]
#[ignore]
fn test_geom_bold_coupled() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    let batch = 256usize;
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== coupled bold, batch {batch}: ONE commutator + McWeeny per round; between rounds q ← q + 0.5 (q_new − q) ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 45 {
            eprintln!("coupled budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let g1b = replicate(&g1, batch);

        unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER") };
        let mut jac = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("jacobi {}: {e}", sys.name));
        jac.set_smearing(0.0);
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi G0 {}: {e}", sys.name));
        jac.set_coords(&g1b).unwrap();
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi SCC {}: {e}", sys.name));
        let ev_j = jac.eval(true).unwrap_or_else(|e| panic!("jacobi eval {}: {e}", sys.name));
        let q_scc = jac.read_q_new().unwrap();
        let e_scc = ev_j.energy[0];
        let f_scc = ev_j.forces.unwrap();
        drop(jac);

        unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify") };
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("purify {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("purify G0 {}: {e}", sys.name));
        eng.set_coords_keep_k(&g1b).unwrap();
        let guess = eng.plan.capture_density(&eng.rt).unwrap();
        for n in 1..=4 {
            eng.plan.restore_density(&eng.rt, &guess).unwrap();
            let mut rh = 0.0f32;
            let mut tr = 0.0f32;
            let mut eta = 0.0f32;
            let mut stopped = false;
            for round in 1..=n {
                let (_rh0, rh1, tr1, eta1, n_acc) =
                    eng.geom_bold_capped(1).unwrap_or_else(|e| panic!("{} round {round}: {e}", sys.name));
                rh = rh1;
                tr = tr1;
                eta = eta1;
                if n_acc < 1 {
                    stopped = true;
                    break;
                }
                if round < n {
                    eng.mix_mulliken(0.5).unwrap();
                }
            }
            let q_new = eng.read_q_new().unwrap();
            let mut dq_scc = 0.0f64;
            for a in 0..n_atoms {
                dq_scc = dq_scc.max((q_new[a] as f64 - q_scc[a] as f64).abs());
            }
            eng.stage_mulliken().unwrap();
            let ev = eng.eval(true).unwrap_or_else(|e| panic!("eval N={n}: {e}"));
            let de_mev = (ev.energy[0] - e_scc) * 27211.386;
            let f = ev.forces.as_ref().unwrap();
            let mut df = 0.0f64;
            let mut fmax = 0.0f64;
            for i in 0..3 * n_atoms {
                df = df.max((f[i] as f64 - f_scc[i] as f64).abs());
                fmax = fmax.max(f_scc[i].abs() as f64);
            }
            let pct = 100.0 * df / fmax.max(1e-12);
            eprintln!(
                "  {}  rounds={n}  η={eta:.3}  R_H={rh:.3e}  Tr={tr:.4}  dq_scc={dq_scc:.3e}  dE={de_mev:+.3} meV  dF={pct:.2}%",
                sys.name
            );
            if stopped {
                eprintln!("  {}  a round accepted nothing", sys.name);
                break;
            }
        }
    }
}

/// One accepted commutator + McWeeny, then the existing charge DIIS.
/// The first pass falls back to α = 0.3; from the second pass the mixer
/// is DIIS. Batch 256, against a Jacobi SCC.
#[test]
#[ignore]
fn test_geom_bold_diis() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    let batch = 256usize;
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== bold + charge DIIS, batch {batch}: one commutator + McWeeny, then DIIS; a second commutator on the same H when R_H > 0.3·rms ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 45 {
            eprintln!("diis budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let g1b = replicate(&g1, batch);

        unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER") };
        let mut jac = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("jacobi {}: {e}", sys.name));
        jac.set_smearing(0.0);
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi G0 {}: {e}", sys.name));
        jac.set_coords(&g1b).unwrap();
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi SCC {}: {e}", sys.name));
        let ev_j = jac.eval(true).unwrap_or_else(|e| panic!("jacobi eval {}: {e}", sys.name));
        let q_scc = jac.read_q_new().unwrap();
        let e_scc = ev_j.energy[0];
        let f_scc = ev_j.forces.unwrap();
        drop(jac);

        unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify") };
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("purify {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("purify G0 {}: {e}", sys.name));
        eng.set_coords_keep_k(&g1b).unwrap();
        let guess = eng.plan.capture_density(&eng.rt).unwrap();
        for n in 1..=6 {
            eng.plan.restore_density(&eng.rt, &guess).unwrap();
            eng.plan.reset_diis(&eng.rt).unwrap();
            let mut rh = 0.0f32;
            let mut rms = 0.0f32;
            let mut n_comm = 0usize;
            let mut stopped = false;
            for _pass in 1..=n {
                let (_rh0, rh1, _tr, _eta, n_acc) =
                    eng.geom_bold_capped(1).unwrap_or_else(|e| panic!("{} pass: {e}", sys.name));
                rh = rh1;
                n_comm += 1;
                if n_acc < 1 {
                    stopped = true;
                    break;
                }
                let q_out = eng.read_q_new().unwrap();
                let mut q_in = vec![0.0f32; n_atoms];
                eng.rt.read_buffer(&eng.plan.q_gpu, &mut q_in).unwrap();
                let mut acc = 0.0f64;
                for a in 0..n_atoms {
                    let d = q_out[a] as f64 - q_in[a] as f64;
                    acc += d * d;
                }
                let rms_q = (acc / n_atoms as f64).sqrt() as f32;
                if rh > 0.3 * rms_q.max(1e-8) {
                    let (_a, rh2, _b, _c, n2) =
                        eng.geom_bold_capped(1).unwrap_or_else(|e| panic!("{} extra: {e}", sys.name));
                    rh = rh2;
                    n_comm += 1;
                    if n2 < 1 {
                        stopped = true;
                        break;
                    }
                }
                rms = eng.plan.diis_on_qnew(&mut eng.rt).unwrap_or_else(|e| panic!("diis: {e}"));
            }
            let q_new = eng.read_q_new().unwrap();
            let mut dq_scc = 0.0f64;
            for a in 0..n_atoms {
                dq_scc = dq_scc.max((q_new[a] as f64 - q_scc[a] as f64).abs());
            }
            eng.stage_mulliken().unwrap();
            let ev = eng.eval(true).unwrap_or_else(|e| panic!("eval N={n}: {e}"));
            let de_mev = (ev.energy[0] - e_scc) * 27211.386;
            let f = ev.forces.as_ref().unwrap();
            let mut df = 0.0f64;
            let mut fmax = 0.0f64;
            for i in 0..3 * n_atoms {
                df = df.max((f[i] as f64 - f_scc[i] as f64).abs());
                fmax = fmax.max(f_scc[i].abs() as f64);
            }
            let pct = 100.0 * df / fmax.max(1e-12);
            eprintln!(
                "  {}  passes={n}  comm={n_comm}  rms={rms:.3e}  R_H={rh:.3e}  dq_scc={dq_scc:.3e}  dE={de_mev:+.3} meV  dF={pct:.2}%",
                sys.name
            );
            if stopped {
                eprintln!("  {}  a pass accepted nothing", sys.name);
                break;
            }
        }
    }
}

/// Two attempts to get the force down in fewer than six passes.
/// `carry`: reuse the G0 DIIS history (the response from the previous
/// geometry). `reject`: if a mix raises the residual, put the charges
/// back, clear the history, and take α = 0.5 instead.
#[test]
#[ignore]
fn test_geom_bold_diis_carry() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    let batch = 256usize;
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== carry G0 DIIS history, and reject a mix that raises rms. batch {batch} ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 45 {
            eprintln!("carry budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let g1b = replicate(&g1, batch);

        unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER") };
        let mut jac = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("jacobi {}: {e}", sys.name));
        jac.set_smearing(0.0);
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi G0 {}: {e}", sys.name));
        jac.set_coords(&g1b).unwrap();
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi SCC {}: {e}", sys.name));
        let ev_j = jac.eval(true).unwrap_or_else(|e| panic!("jacobi eval {}: {e}", sys.name));
        let q_scc = jac.read_q_new().unwrap();
        let e_scc = ev_j.energy[0];
        let f_scc = ev_j.forces.unwrap();
        drop(jac);

        unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify") };
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("purify {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("purify G0 {}: {e}", sys.name));
        let mut nf = vec![0i32; 1];
        eng.rt.read_buffer(&eng.plan.diis_n_filled, &mut nf).unwrap();
        let diis_g0 = eng.plan.capture_diis(&eng.rt).unwrap();
        eprintln!("  {}  G0 DIIS n_filled={}", sys.name, nf[0]);
        eng.set_coords_keep_k(&g1b).unwrap();
        let guess = eng.plan.capture_density(&eng.rt).unwrap();
        let mut q_save = vec![0.0f32; batch * n_atoms];
        for &reject in &[false, true] {
            let tag = if reject { "reject" } else { "carry" };
            for n in 1..=6 {
                eng.plan.restore_density(&eng.rt, &guess).unwrap();
                if reject {
                    eng.plan.reset_diis(&eng.rt).unwrap();
                } else {
                    eng.plan.restore_diis(&eng.rt, &diis_g0).unwrap();
                }
                let mut rh = 0.0f32;
                let mut rms = 0.0f32;
                let mut n_comm = 0usize;
                let mut n_rej = 0usize;
                let mut prev_rms = f32::MAX;
                let mut stopped = false;
                let mut q_in = vec![0.0f32; n_atoms];
                let mut q_out = vec![0.0f32; n_atoms];
                for _pass in 1..=n {
                    let (_rh0, rh1, _tr, _eta, n_acc) =
                        eng.geom_bold_capped(1).unwrap_or_else(|e| panic!("{} {tag}: {e}", sys.name));
                    rh = rh1;
                    n_comm += 1;
                    if n_acc < 1 {
                        stopped = true;
                        break;
                    }
                    eng.rt.read_buffer(&eng.plan.q_gpu, &mut q_in).unwrap();
                    eng.rt.read_buffer(&eng.plan.q_new, &mut q_out).unwrap();
                    let mut acc = 0.0f64;
                    for a in 0..n_atoms {
                        let d = q_out[a] as f64 - q_in[a] as f64;
                        acc += d * d;
                    }
                    let mut rms_q = (acc / n_atoms as f64).sqrt() as f32;
                    if rh > 0.3 * rms_q.max(1e-8) {
                        let (_a, rh2, _b, _c, n2) = eng.geom_bold_capped(1)
                            .unwrap_or_else(|e| panic!("{} {tag} extra: {e}", sys.name));
                        rh = rh2;
                        n_comm += 1;
                        if n2 < 1 {
                            stopped = true;
                            break;
                        }
                        eng.rt.read_buffer(&eng.plan.q_new, &mut q_out).unwrap();
                        acc = 0.0;
                        for a in 0..n_atoms {
                            let d = q_out[a] as f64 - q_in[a] as f64;
                            acc += d * d;
                        }
                        rms_q = (acc / n_atoms as f64).sqrt() as f32;
                    }
                    if reject {
                        eng.rt.read_buffer(&eng.plan.q_gpu, &mut q_save).unwrap();
                    }
                    rms = eng.plan.diis_on_qnew(&mut eng.rt).unwrap_or_else(|e| panic!("diis: {e}"));
                    if reject && rms > prev_rms {
                        eng.rt.write_buffer(&eng.plan.q_gpu, &q_save).unwrap();
                        eng.plan.reset_diis(&eng.rt).unwrap();
                        eng.mix_mulliken(0.5).unwrap();
                        n_rej += 1;
                        prev_rms = 0.5 * rms_q.max(rms);
                    } else {
                        prev_rms = rms;
                    }
                }
                let q_new = eng.read_q_new().unwrap();
                let mut dq_scc = 0.0f64;
                for a in 0..n_atoms {
                    dq_scc = dq_scc.max((q_new[a] as f64 - q_scc[a] as f64).abs());
                }
                eng.stage_mulliken().unwrap();
                let ev = eng.eval(true).unwrap_or_else(|e| panic!("eval: {e}"));
                let de_mev = (ev.energy[0] - e_scc) * 27211.386;
                let f = ev.forces.as_ref().unwrap();
                let mut df = 0.0f64;
                let mut fmax = 0.0f64;
                for i in 0..3 * n_atoms {
                    df = df.max((f[i] as f64 - f_scc[i] as f64).abs());
                    fmax = fmax.max(f_scc[i].abs() as f64);
                }
                let pct = 100.0 * df / fmax.max(1e-12);
                eprintln!(
                    "  {}  {tag}  passes={n}  comm={n_comm}  rej={n_rej}  rms={rms:.3e}  R_H={rh:.3e}  dq={dq_scc:.3e}  dE={de_mev:+.3} meV  dF={pct:.2}%",
                    sys.name
                );
                if stopped {
                    break;
                }
            }
        }
    }
}

/// DIIS only when the commutator is well below the charge residual
/// (`R_H < 0.1 × rms`). Otherwise α = 0.5, so an inexact map is not
/// extrapolated. Up to three commutators on the current H per pass.
#[test]
#[ignore]
fn test_geom_bold_diis_gate() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    let batch = 256usize;
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== DIIS only if R_H < 0.1·rms, else α=0.5. Up to 3 commutators per pass. batch {batch} ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 45 {
            eprintln!("gate budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let g1b = replicate(&g1, batch);

        unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER") };
        let mut jac = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("jacobi {}: {e}", sys.name));
        jac.set_smearing(0.0);
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi G0 {}: {e}", sys.name));
        jac.set_coords(&g1b).unwrap();
        jac.scc(100, 1e-6).unwrap_or_else(|e| panic!("jacobi SCC {}: {e}", sys.name));
        let ev_j = jac.eval(true).unwrap_or_else(|e| panic!("jacobi eval {}: {e}", sys.name));
        let q_scc = jac.read_q_new().unwrap();
        let e_scc = ev_j.energy[0];
        let f_scc = ev_j.forces.unwrap();
        drop(jac);

        unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify") };
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("purify {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("purify G0 {}: {e}", sys.name));
        eng.set_coords_keep_k(&g1b).unwrap();
        let guess = eng.plan.capture_density(&eng.rt).unwrap();
        let mut q_in = vec![0.0f32; n_atoms];
        let mut q_out = vec![0.0f32; n_atoms];
        for n in 1..=6 {
            eng.plan.restore_density(&eng.rt, &guess).unwrap();
            eng.plan.reset_diis(&eng.rt).unwrap();
            let mut rh = 0.0f32;
            let mut rms = 0.0f32;
            let mut n_comm = 0usize;
            let mut n_diis = 0usize;
            let mut ratio = 0.0f32;
            let mut stopped = false;
            for _pass in 1..=n {
                let mut rms_q = 0.0f32;
                for _k in 0..3 {
                    let (_a, rh1, _b, _c, n_acc) =
                        eng.geom_bold_capped(1).unwrap_or_else(|e| panic!("{}: {e}", sys.name));
                    rh = rh1;
                    n_comm += 1;
                    if n_acc < 1 {
                        stopped = true;
                        break;
                    }
                    eng.rt.read_buffer(&eng.plan.q_gpu, &mut q_in).unwrap();
                    eng.rt.read_buffer(&eng.plan.q_new, &mut q_out).unwrap();
                    let mut acc = 0.0f64;
                    for a in 0..n_atoms {
                        let d = q_out[a] as f64 - q_in[a] as f64;
                        acc += d * d;
                    }
                    rms_q = (acc / n_atoms as f64).sqrt() as f32;
                    ratio = rh / rms_q.max(1e-8);
                    if ratio < 0.1 {
                        break;
                    }
                }
                if stopped {
                    break;
                }
                if ratio < 0.1 {
                    rms = eng.plan.diis_on_qnew(&mut eng.rt).unwrap_or_else(|e| panic!("diis: {e}"));
                    n_diis += 1;
                } else {
                    eng.mix_mulliken(0.5).unwrap();
                    rms = 0.5 * rms_q;
                }
            }
            let q_new = eng.read_q_new().unwrap();
            let mut dq_scc = 0.0f64;
            for a in 0..n_atoms {
                dq_scc = dq_scc.max((q_new[a] as f64 - q_scc[a] as f64).abs());
            }
            eng.stage_mulliken().unwrap();
            let ev = eng.eval(true).unwrap_or_else(|e| panic!("eval: {e}"));
            let de_mev = (ev.energy[0] - e_scc) * 27211.386;
            let f = ev.forces.as_ref().unwrap();
            let mut df = 0.0f64;
            let mut fmax = 0.0f64;
            for i in 0..3 * n_atoms {
                df = df.max((f[i] as f64 - f_scc[i] as f64).abs());
                fmax = fmax.max(f_scc[i].abs() as f64);
            }
            let pct = 100.0 * df / fmax.max(1e-12);
            eprintln!(
                "  {}  passes={n}  comm={n_comm}  diis={n_diis}  ratio={ratio:.2}  rms={rms:.3e}  R_H={rh:.3e}  dq={dq_scc:.3e}  dE={de_mev:+.3} meV  dF={pct:.2}%",
                sys.name
            );
            if stopped {
                break;
            }
        }
    }
}

/// One-GEMM H′ (H0' + ½(M+Mᵀ), M = Xᵀ D Cᵀ) and one-GEMM Mulliken
/// (Y = K·C, p_μ = 2 Σ_i X_μi Y_iμ) against the two-GEMM bold step.
/// Same carried K, same 0.02 Å step, batch 256. A mismatch is a bug.
#[test]
#[ignore]
fn test_geom_bold_cheap() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    unsafe { std::env::set_var("RUST_DFTB_BOLD_DIAG", "0"); }
    let names = std::env::var("RUST_DFTB_BENCH_SYSTEMS").unwrap_or_else(|_| "formic,GC".into());
    let want: Vec<&str> = names.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    let batch = 256usize;
    eprintln!("=== one-GEMM H'/q vs two-GEMM, batch {batch}, systems {names} ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.iter().any(|w| *w == sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 50 {
            eprintln!("cheap budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let g1b = replicate(&g1, batch);

        unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify"); }
        unsafe { std::env::set_var("RUST_DFTB_BOLD_CHEAP", "1"); }
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("purify {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        eng.set_coords_keep_k(&g1b).unwrap();
        let guess = eng.plan.capture_density(&eng.rt).unwrap();

        unsafe { std::env::set_var("RUST_DFTB_BOLD_CHEAP", "0"); }
        eng.plan.restore_density(&eng.rt, &guess).unwrap();
        let (_a0, rh0, tr0, _e0, n0) = eng.geom_bold_capped(2).unwrap_or_else(|e| panic!("two-GEMM {}: {e}", sys.name));
        let q0 = eng.read_q_new().unwrap();
        let ne0: f32 = q0.iter().take(n_atoms).sum();

        unsafe { std::env::set_var("RUST_DFTB_BOLD_CHEAP", "1"); }
        eng.plan.restore_density(&eng.rt, &guess).unwrap();
        let (_a1, rh1, tr1, _e1, n1) = eng.geom_bold_capped(2).unwrap_or_else(|e| panic!("one-GEMM {}: {e}", sys.name));
        let q1 = eng.read_q_new().unwrap();
        let ne1: f32 = q1.iter().take(n_atoms).sum();
        let mut dq = 0.0f32;
        for a in 0..n_atoms {
            dq = dq.max((q1[a] - q0[a]).abs());
        }
        eprintln!(
            "  {}  parity  |Δq|={dq:.3e} e  Ne {ne0:.4}/{ne1:.4}  R_H {rh0:.3e}/{rh1:.3e}  Tr {tr0:.4}/{tr1:.4}  acc {n0}/{n1}",
            sys.name
        );
        assert!(dq < 1e-3, "{} one-GEMM charges differ by {dq:.3e} e", sys.name);
        assert!((ne1 - ne0).abs() < 1e-2, "{} electron count {ne0} vs {ne1}", sys.name);

        unsafe { std::env::set_var("RUST_DFTB_BOLD_CHEAP", "1"); }
        unsafe { std::env::set_var("RUST_DFTB_BOLD_DIAG", "1"); }
        eng.plan.restore_density(&eng.rt, &guess).unwrap();
        let (_ad, rhd, trd, _ed, nd) = eng.geom_bold_capped(2).unwrap_or_else(|e| panic!("diag {}: {e}", sys.name));
        let qd = eng.read_q_new().unwrap();
        unsafe { std::env::set_var("RUST_DFTB_BOLD_DIAG", "0"); }
        eng.plan.restore_density(&eng.rt, &guess).unwrap();
        let (_ap, rhp, trp, _ep, np) = eng.geom_bold_capped(2).unwrap_or_else(|e| panic!("prod {}: {e}", sys.name));
        let qp = eng.read_q_new().unwrap();
        let mut dq_sync = 0.0f32;
        for a in 0..n_atoms {
            dq_sync = dq_sync.max((qp[a] - qd[a]).abs());
        }
        eprintln!(
            "  {}  diag vs prod  |Δq|={dq_sync:.3e} e  R_H {rhd:.6e}/{rhp:.6e}  Tr {trd:.4}/{trp:.4}  acc {nd}/{np}",
            sys.name
        );
        assert!(dq_sync < 1e-4, "{} device trust |Δq|={dq_sync:.3e}", sys.name);
        assert_eq!(nd, np, "{} accepted steps diag {nd} prod {np}", sys.name);
        assert!((rhd - rhp).abs() < 1e-4, "{} R_H {rhd} vs {rhp}", sys.name);
        assert!((trd - trp).abs() < 1e-3, "{} Tr {trd} vs {trp}", sys.name);

        let time_bold = |eng: &mut GpuDftb, cheap: &str, diag: &str| -> f64 {
            unsafe { std::env::set_var("RUST_DFTB_BOLD_CHEAP", cheap); }
            unsafe { std::env::set_var("RUST_DFTB_BOLD_DIAG", diag); }
            let mut secs = 0.0f64;
            for _ in 0..5 {
                eng.plan.restore_density(&eng.rt, &guess).unwrap();
                let t0 = std::time::Instant::now();
                eng.geom_bold_capped(2).unwrap();
                secs += t0.elapsed().as_secs_f64();
            }
            secs / 5.0
        };
        let t_two = time_bold(&mut eng, "0", "0");
        let t_one = time_bold(&mut eng, "1", "0");
        let t_diag = time_bold(&mut eng, "1", "1");
        eprintln!(
            "  {}  one bold cap2  two-GEMM {:.2} ms  one-GEMM {:.2} ms  diag {:.2} ms  (prod/diag {:.2}×)",
            sys.name, t_two * 1e3, t_one * 1e3, t_diag * 1e3, t_diag / t_one.max(1e-9)
        );

        let time_gate = |eng: &mut GpuDftb, cheap: &str| -> f64 {
            unsafe { std::env::set_var("RUST_DFTB_BOLD_CHEAP", cheap); }
            unsafe { std::env::set_var("RUST_DFTB_BOLD_DIAG", "0"); }
            eng.plan.restore_density(&eng.rt, &guess).unwrap();
            eng.plan.reset_diis(&eng.rt).unwrap();
            let mut q_in = vec![0.0f32; n_atoms];
            let mut q_out = vec![0.0f32; n_atoms];
            let t0 = std::time::Instant::now();
            for _pass in 0..6 {
                let mut ratio = 1.0f32;
                let mut rms_q = 0.0f32;
                let mut stopped = false;
                for _k in 0..3 {
                    let (_a, rh1, _b, _c, n_acc) = eng.geom_bold_capped(1).unwrap();
                    if n_acc < 1 {
                        stopped = true;
                        break;
                    }
                    eng.rt.read_buffer(&eng.plan.q_gpu, &mut q_in).unwrap();
                    eng.rt.read_buffer(&eng.plan.q_new, &mut q_out).unwrap();
                    let mut acc = 0.0f64;
                    for a in 0..n_atoms {
                        let d = q_out[a] as f64 - q_in[a] as f64;
                        acc += d * d;
                    }
                    rms_q = (acc / n_atoms as f64).sqrt() as f32;
                    ratio = rh1 / rms_q.max(1e-8);
                    if ratio < 0.1 {
                        break;
                    }
                }
                if stopped {
                    break;
                }
                if ratio < 0.1 {
                    eng.plan.diis_on_qnew(&mut eng.rt).unwrap();
                } else {
                    eng.mix_mulliken(0.5).unwrap();
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let g_two = time_gate(&mut eng, "0");
        let g_one = time_gate(&mut eng, "1");
        eprintln!(
            "  {}  gate 6-pass  two-GEMM {:.1} ms  one-GEMM {:.1} ms  ({:.2}×)",
            sys.name, g_two * 1e3, g_one * 1e3, g_two / g_one.max(1e-9)
        );
    }
    unsafe { std::env::set_var("RUST_DFTB_BOLD_CHEAP", "0"); }
}

/// Log E_SCC on accepted coupled steps. The trust test is not changed.
/// E_dens = 2 Tr(K H'₀) + ½ ΔqᵀγΔq. E_chat drops the 2.
/// A step where R_H falls and E_dens rises is marked DISAGREE.
#[test]
#[ignore]
fn test_geom_bold_energy() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    unsafe { std::env::set_var("RUST_DFTB_BOLD_DIAG", "0"); }
    let batch = 256usize;
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== E_SCC on accepted coupled steps (α=1). Trust test unchanged. batch {batch} ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 45 {
            eprintln!("energy budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc = cpu.build_result();
        let forces = cpu.compute_forces(&scc, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &forces.forces, 0.02);
        let g1b = replicate(&g1, batch);

        unsafe { std::env::set_var("RUST_DFTB_EIGSOLVER", "purify"); }
        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("purify {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        eng.set_coords_keep_k(&g1b).unwrap();
        let (e0, chat0, _) = eng.bold_scc_energy(false).unwrap_or_else(|e| panic!("E0 {}: {e}", sys.name));
        eprintln!("  {}  step 0  E_dens={e0:.8} Ha  E_chat={chat0:.8} Ha", sys.name);
        let mut e_prev = e0;
        let mut chat_prev = chat0;
        let mut n_dis = 0usize;
        for step in 1..=4 {
            let (rh0, rh, _tr, eta, n_acc) = eng.geom_bold_capped(1).unwrap_or_else(|e| panic!("{} step {step}: {e}", sys.name));
            let (e, chat, tr_h) = eng.bold_scc_energy(true).unwrap_or_else(|e| panic!("E {}: {e}", sys.name));
            let de = (e - e_prev) * 27211.386;
            let dc = (chat - chat_prev) * 27211.386;
            let fell = rh < rh0;
            let rose = e > e_prev;
            let flag = if n_acc >= 1 && fell && rose { "DISAGREE" } else { "together" };
            if flag == "DISAGREE" {
                n_dis += 1;
            }
            eprintln!(
                "  {}  step {step}  acc={n_acc}  η={eta:.3}  R_H {rh0:.3e} → {rh:.3e}  ΔE_dens={de:+.4} meV  ΔE_chat={dc:+.4} meV  Tr(KH')={tr_h:.6}  {flag}",
                sys.name
            );
            e_prev = e;
            chat_prev = chat;
            if n_acc < 1 {
                break;
            }
            eng.stage_mulliken().unwrap();
        }
        eprintln!("  {}  disagreements {n_dis}/4", sys.name);
    }
}

/// Frozen-charge shadow: one Jacobi of H(R, q_carried), no charge update.
/// The analytic force is compared to a finite difference of that energy,
/// and to a full SCC Jacobi at the same geometry.
#[test]
#[ignore]
fn test_geom_shadow_fd() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER"); }
    // Force buffer is dE/dR with R in Å (kernel multiplies by ANG2BOHR).
    let deltas = [1.0e-3f64, 2.0e-4f64];
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== frozen-q shadow: one Jacobi vs FD (Ha/Å) and vs SCC. batch 1 ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 45 {
            eprintln!("shadow budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc0 = cpu.build_result();
        let f0 = cpu.compute_forces(&scc0, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &f0.forces, 0.02);

        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), g0.clone(), 1)
            .unwrap_or_else(|e| panic!("shadow {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        eng.set_coords(&g1).unwrap();
        let ev = eng.eval(true).unwrap_or_else(|e| panic!("shadow eval {}: {e}", sys.name));
        let e_sh = ev.energy[0];
        let f_sh = ev.forces.unwrap();
        if f_sh.len() != 3 * n_atoms {
            panic!("{} force len {} != {}", sys.name, f_sh.len(), 3 * n_atoms);
        }
        let mut imax = 0usize;
        let mut fmax = 0.0f64;
        for (i, &f) in f_sh.iter().enumerate() {
            if !f.is_finite() {
                panic!("{} F_shadow[{i}]={f} non-finite", sys.name);
            }
            if (f as f64).abs() > fmax {
                fmax = (f as f64).abs();
                imax = i;
            }
        }
        let atom = imax / 3;
        let comp = imax % 3;
        let f_an = f_sh[imax] as f64;
        let mut rel_line = String::new();
        for delta in deltas {
            let mut e_side = [0.0f64; 2];
            for (k, sign) in [1.0f64, -1.0f64].into_iter().enumerate() {
                let mut g = g1.clone();
                g[atom][comp] += sign * delta;
                eng.set_coords(&g).unwrap();
                let evs = eng.eval(false).unwrap_or_else(|e| panic!("FD {}: {e}", sys.name));
                e_side[k] = evs.energy[0];
                if !e_side[k].is_finite() {
                    panic!("{} E_shadow({sign})={} non-finite", sys.name, e_side[k]);
                }
            }
            let f_num = -(e_side[0] - e_side[1]) / (2.0 * delta);
            let rel = (f_num - f_an).abs() / f_an.abs().max(1e-12);
            rel_line.push_str(&format!("  δ={delta:.1e} F_num={f_num:.6e} |Δ|/|F|={rel:.3e}"));
        }

        eng.set_coords(&g1).unwrap();
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("SCC {}: {e}", sys.name));
        let evj = eng.eval(true).unwrap_or_else(|e| panic!("SCC eval {}: {e}", sys.name));
        let e_scc = evj.energy[0];
        let f_scc = evj.forces.unwrap();
        let mut df = 0.0f64;
        let mut fscc_max = 0.0f64;
        for i in 0..3 * n_atoms {
            df = df.max((f_sh[i] as f64 - f_scc[i] as f64).abs());
            fscc_max = fscc_max.max((f_scc[i] as f64).abs());
        }
        let pct = 100.0 * df / fscc_max.max(1e-12);
        let de = (e_sh - e_scc) * 27211.386;
        let axis = ["x", "y", "z"][comp];
        eprintln!(
            "  {}  atom {atom} {axis}  F_shadow={f_an:.6e}{rel_line}   vs SCC  dF={pct:.2}%  ΔE={de:+.3} meV",
            sys.name
        );
    }
}

/// Partial-pivot solve of A x = b. A is row-major n×n.
fn solve_ge(a: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut m = a.to_vec();
    let mut b = rhs.to_vec();
    for k in 0..n {
        let mut piv = k;
        let mut best = m[k * n + k].abs();
        for i in (k + 1)..n {
            let v = m[i * n + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-14 {
            panic!("shadow Newton: pivot {best:.3e} at column {k}/{n}");
        }
        if piv != k {
            for j in 0..n {
                m.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        let diag = m[k * n + k];
        for i in (k + 1)..n {
            let f = m[i * n + k] / diag;
            for j in (k + 1)..n {
                m[i * n + j] -= f * m[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= m[i * n + j] * x[j];
        }
        x[i] = s / m[i * n + i];
        if !x[i].is_finite() {
            panic!("shadow Newton: x[{i}] non-finite");
        }
    }
    x
}

/// 1st-level shadow: one Newton step on the charge residual of a single
/// diagonalization, then one diagonalization of H at the updated charges.
/// The Jacobian is a central difference, a measurement, not the production step.
#[test]
#[ignore]
fn test_geom_shadow_newton() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER"); }
    let eps = 1.0e-3f64;
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== 1st-level shadow: Newton on q, then one Jacobi. ε={eps:.1e} e. batch 1 ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 50 {
            eprintln!("newton budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc0 = cpu.build_result();
        let f0cpu = cpu.compute_forces(&scc0, &repulsive).unwrap();
        let g1 = step_along_forces(&g0, &f0cpu.forces, 0.02);

        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), g0.clone(), 1)
            .unwrap_or_else(|e| panic!("newton {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        let mut q0 = vec![0.0f32; n_atoms];
        eng.rt.read_buffer(&eng.plan.q_gpu, &mut q0).unwrap();
        eng.set_coords(&g1).unwrap();

        let residual = |eng: &mut GpuDftb, q: &[f32]| -> (Vec<f64>, f64, Vec<f32>) {
            eng.set_charges(q).unwrap_or_else(|e| panic!("set_charges {}: {e}", sys.name));
            let ev = eng.eval(true).unwrap_or_else(|e| panic!("eval {}: {e}", sys.name));
            let qn = eng.read_q_new().unwrap();
            let mut f = vec![0.0f64; n_atoms];
            let mut s2 = 0.0f64;
            for a in 0..n_atoms {
                f[a] = qn[a] as f64 - q[a] as f64;
                if !f[a].is_finite() {
                    panic!("{} residual[{a}] non-finite", sys.name);
                }
                s2 += f[a] * f[a];
            }
            (f, (s2 / n_atoms as f64).sqrt(), ev.forces.unwrap())
        };

        let (f_res, rms0, f_sh0) = residual(&mut eng, &q0);
        let e0 = {
            eng.set_charges(&q0).unwrap();
            eng.eval(false).unwrap().energy[0]
        };

        let mut jac = vec![0.0f64; n_atoms * n_atoms];
        let mut qp = q0.clone();
        let mut qm = q0.clone();
        for a in 0..n_atoms {
            qp.copy_from_slice(&q0);
            qm.copy_from_slice(&q0);
            qp[a] = (q0[a] as f64 + eps) as f32;
            qm[a] = (q0[a] as f64 - eps) as f32;
            eng.set_charges(&qp).unwrap();
            eng.eval(false).unwrap();
            let q_plus = eng.read_q_new().unwrap();
            eng.set_charges(&qm).unwrap();
            eng.eval(false).unwrap();
            let q_minus = eng.read_q_new().unwrap();
            for b in 0..n_atoms {
                let fp = q_plus[b] as f64 - qp[b] as f64;
                let fm = q_minus[b] as f64 - qm[b] as f64;
                jac[b * n_atoms + a] = (fp - fm) / (2.0 * eps);
            }
        }
        let x = solve_ge(&jac, &f_res);
        let mut q1 = vec![0.0f32; n_atoms];
        let mut sum_x = 0.0f64;
        for a in 0..n_atoms {
            sum_x += x[a];
            let v = q0[a] as f64 - x[a];
            if !v.is_finite() {
                panic!("{} q1[{a}] non-finite", sys.name);
            }
            q1[a] = v as f32;
        }
        let (_f1, rms1, f_sh1) = residual(&mut eng, &q1);
        let e1 = {
            eng.set_charges(&q1).unwrap();
            eng.eval(false).unwrap().energy[0]
        };

        eng.set_charges(&q0).unwrap();
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("SCC {}: {e}", sys.name));
        let evj = eng.eval(true).unwrap_or_else(|e| panic!("SCC eval {}: {e}", sys.name));
        let e_scc = evj.energy[0];
        let f_scc = evj.forces.unwrap();
        let force_pct = |f: &[f32]| {
            let mut df = 0.0f64;
            let mut mx = 0.0f64;
            for i in 0..3 * n_atoms {
                df = df.max((f[i] as f64 - f_scc[i] as f64).abs());
                mx = mx.max((f_scc[i] as f64).abs());
            }
            100.0 * df / mx.max(1e-12)
        };
        eprintln!(
            "  {}  rms f {rms0:.3e} → {rms1:.3e}  Σx={sum_x:.3e}  ΔE0={:+.3} meV  ΔE1={:+.3} meV  dF0={:.2}%  dF1={:.2}%",
            sys.name,
            (e0 - e_scc) * 27211.386,
            (e1 - e_scc) * 27211.386,
            force_pct(&f_sh0),
            force_pct(&f_sh1)
        );
    }
}

/// Wall time of a full SCC against two Jacobi diagonalizations.
/// Two diagonalizations are the electronic cost of the 0th-level plus the
/// 1st-level shadow once the inverse charge Jacobian is already known.
/// The finite-difference Jacobian used in the accuracy test is not timed;
/// its cost is estimated from the measured diagonalization time.
#[test]
#[ignore]
fn test_geom_shadow_perf() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    unsafe { std::env::set_var("RUST_DFTB_SCC_QUIET", "1"); }
    unsafe { std::env::remove_var("RUST_DFTB_EIGSOLVER"); }
    let batch = 256usize;
    let reps = 3usize;
    let want = ["formic", "GC", "diazaphen", "DTH"];
    eprintln!("=== shadow cost, batch {batch}: full SCC vs 2 Jacobi (Jacobian assumed known) ===");
    let started = std::time::Instant::now();
    for sys in SCAN_SYSTEMS {
        if !want.contains(&sys.name) {
            continue;
        }
        if started.elapsed().as_secs() > 45 {
            eprintln!("perf budget: skip {}", sys.name);
            break;
        }
        let xyz = xyz_file(sys.file);
        let sp = xyz.species.clone();
        let g0 = xyz.coords.clone();
        let n_atoms = g0.len();
        let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
        let mut unique: Vec<String> = Vec::new();
        for s in &sp {
            if !unique.contains(s) { unique.push(s.clone()); }
        }
        let repulsive = parse_all_repulsive(&sk_dir, &unique, unique.len()).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), sp.clone()).unwrap();
        cpu.update_geometry(&g0).unwrap();
        cpu.reset_charges();
        cpu.solve_scc(200, 1e-8).unwrap();
        let scc0 = cpu.build_result();
        let f0 = cpu.compute_forces(&scc0, &repulsive).unwrap();
        let g1 = replicate(&step_along_forces(&g0, &f0.forces, 0.02), batch);

        let mut eng = GpuDftb::new(sk.clone(), &sk_dir, sp.clone(), replicate(&g0, batch), batch)
            .unwrap_or_else(|e| panic!("perf {}: {e}", sys.name));
        eng.set_smearing(0.0);
        eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("G0 {}: {e}", sys.name));
        let mut q = vec![0.0f32; batch * n_atoms];
        eng.rt.read_buffer(&eng.plan.q_gpu, &mut q).unwrap();
        eng.set_coords(&g1).unwrap();

        let mut scc_ms = 0.0f64;
        let mut n_it = 0usize;
        for rep in 0..reps {
            eng.set_charges(&q).unwrap();
            let t = std::time::Instant::now();
            let s = eng.scc(100, 1e-6).unwrap_or_else(|e| panic!("SCC {}: {e}", sys.name));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if rep == 0 {
                n_it = s.n_iters;
            }
            scc_ms += ms;
        }
        scc_ms /= reps as f64;

        eng.set_charges(&q).unwrap();
        eng.eval(false).unwrap();
        let mut two_ms = 0.0f64;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            eng.set_charges(&q).unwrap();
            eng.eval(false).unwrap();
            eng.set_charges(&q).unwrap();
            eng.eval(false).unwrap();
            two_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        two_ms /= reps as f64;
        let one_ms = two_ms / 2.0;
        let fd_ms = one_ms * (2.0 * n_atoms as f64 + 2.0);
        eprintln!(
            "  {}  atoms={n_atoms}  SCC {n_it} iters  {scc_ms:.2} ms  {:.0} sys/s    2 Jacobi {two_ms:.2} ms  {:.0} sys/s  ({:.2}×)    FD-Jacobian estimate {fd_ms:.0} ms",
            sys.name,
            batch as f64 / (scc_ms / 1e3),
            batch as f64 / (two_ms / 1e3),
            scc_ms / two_ms
        );
    }
}