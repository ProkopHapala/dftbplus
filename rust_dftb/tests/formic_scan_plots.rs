//! Formic dimer 1D + 2D proton-transfer scan: GPU SCC with DIIS.
//! Saves energy + Mulliken charges to debug/formic_dimer_scan/ for plotting.
//!
//! Run: RUST_DFTB_SK_DIR=/path/to/mio-1-1 cargo test --test formic_scan_plots -- --nocapture --ignored

use rust_dftb::qmqm::gpu_driver::GpuDriver;
use rust_dftb::qmqm::gpu_prep::GpuBatch;
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use rust_dftb::qmqm::gpu_scc::{gpu_solve_scc_batched_diis, gpu_solve_scc_batched_diis_warmstart};
use rust_dftb::qmqm::{Fragment, FragmentTemplate, GammaTable};
use rust_dftb::{load_sk_for_species, parse_xyz, HamiltonianBuilder};
use std::fs;
use std::io::Write;

const ANG2BOHR: f64 = 1.889_726_133;

fn per_atom_u(sk: &rust_dftb::SkData, species: &[String]) -> Vec<f64> {
    species.iter().map(|sp| sk.onsite(sp).unwrap().u_hubbard).collect()
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

fn interp_h_pos(h_reactant: &[f64; 3], donor: &[f64; 3], acceptor: &[f64; 3], t: f64) -> [f64; 3] {
    let mid = [(donor[0]+acceptor[0])*0.5, (donor[1]+acceptor[1])*0.5, (donor[2]+acceptor[2])*0.5];
    let h_product = [2.0*mid[0]-h_reactant[0], 2.0*mid[1]-h_reactant[1], 2.0*mid[2]-h_reactant[2]];
    [
        h_reactant[0] + t*(h_product[0]-h_reactant[0]),
        h_reactant[1] + t*(h_product[1]-h_reactant[1]),
        h_reactant[2] + t*(h_product[2]-h_reactant[2]),
    ]
}

/// 1D synchronous scan: both protons move at same parameter t.
fn make_scan_geom_1d(
    base: &[[f64; 3]],
    h1: usize, d1: usize, a1: usize,
    h2: usize, d2: usize, a2: usize,
    t: f64,
) -> Vec<[f64; 3]> {
    let mut c = base.to_vec();
    c[h1] = interp_h_pos(&base[h1], &base[d1], &base[a1], t);
    c[h2] = interp_h_pos(&base[h2], &base[d2], &base[a2], t);
    c
}

/// 2D asynchronous scan: proton 1 moves at t1, proton 2 moves at t2.
fn make_scan_geom_2d(
    base: &[[f64; 3]],
    h1: usize, d1: usize, a1: usize,
    h2: usize, d2: usize, a2: usize,
    t1: f64, t2: f64,
) -> Vec<[f64; 3]> {
    let mut c = base.to_vec();
    c[h1] = interp_h_pos(&base[h1], &base[d1], &base[a1], t1);
    c[h2] = interp_h_pos(&base[h2], &base[d2], &base[a2], t2);
    c
}

/// Run GPU SCC for a batch of geometries. If `init_charges` is provided,
/// warm-start each system from the given charges instead of q0.
/// Returns (energies, charges, per_system_rms).
fn run_gpu_scan(
    sk: &rust_dftb::SkData,
    gamma_table: &GammaTable,
    species: &[String],
    geometries: &[Vec<[f64; 3]>],
    n_atoms: usize,
    n_orbs: usize,
    n_occ: usize,
    label: &str,
    tol: f32,
    init_charges: Option<&[f32]>,  // [batch * n_atoms] warm-start charges, or None for cold start
    best_effort: bool,             // if true, return results even if not all converge
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_points = geometries.len();
    eprintln!("  [GPU] {label}: {n_points} points, building fragments...");

    let frags: Vec<Fragment> = geometries.iter()
        .map(|coords| {
            let tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
            Fragment::from_template(tmpl, coords.to_vec())
        })
        .collect();
    let driver = GpuDriver::new().expect("GpuDriver init failed");
    let batch = GpuBatch::from_fragments(&frags, sk, gamma_table).unwrap();
    let (all_h0, all_s) = driver.gpu_assemble_batched(&batch).unwrap();
    eprintln!("  [GPU] H0/S assembled. Running SCC with DIIS...");

    let u_per_atom = per_atom_u(sk, species);
    let mut all_g = Vec::with_capacity(n_points * n_atoms * n_atoms);
    let mut all_q0 = Vec::with_capacity(n_points * n_atoms);
    let mut all_oa = Vec::with_capacity(n_points * n_orbs);
    for coords in geometries {
        all_g.extend(build_gamma_matrix(coords, &u_per_atom));
        let tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
        all_q0.extend(tmpl.q0.iter().map(|&q| q as f32));
        all_oa.extend(orb_atom_map(&tmpl.atom_orb_off, n_orbs));
    }

    let mut rt = GpuRuntime::new().expect("GpuRuntime init failed");
    let h0_buf = rt.buffer_from_slice(&all_h0).unwrap();
    let s_buf = rt.buffer_from_slice(&all_s).unwrap();
    let g_buf = rt.buffer_from_slice(&all_g).unwrap();
    let q0_buf = rt.buffer_from_slice(&all_q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&all_oa).unwrap();

    let gpu = if let Some(init) = init_charges {
        eprintln!("  [GPU] Warm-starting from neighbouring converged charges");
        assert_eq!(init.len(), n_points * n_atoms,
            "init_charges length {} != expected {} (n_points * n_atoms)",
            init.len(), n_points * n_atoms);
        let init_buf = rt.buffer_from_slice(init).unwrap();
        gpu_solve_scc_batched_diis_warmstart(
            &mut rt, &h0_buf, &s_buf, &g_buf, &q0_buf, &init_buf, &oa_buf,
            n_orbs, n_atoms, n_occ, n_points,
            500, tol, 0.3, 8, 3, best_effort,
        ).expect("GPU SCC (warm-start) failed unexpectedly")
    } else {
        // Cold start: init_q = q0
        gpu_solve_scc_batched_diis_warmstart(
            &mut rt, &h0_buf, &s_buf, &g_buf, &q0_buf, &q0_buf, &oa_buf,
            n_orbs, n_atoms, n_occ, n_points,
            500, tol, 0.3, 8, 3, best_effort,
        ).expect("GPU SCC failed unexpectedly")
    };

    let n_conv = gpu.rms.iter().filter(|&&r| r < tol).count();
    eprintln!("  [GPU] SCC done: n_iters={}, {n_conv}/{n_points} converged, max|E|={:.6}",
        gpu.n_iters, gpu.energies.iter().fold(0.0f32, |m, &e| m.max(e.abs())));
    (gpu.energies, gpu.charges, gpu.rms)
}

fn save_1d_data(
    path: &str, t_values: &[f64],
    cpu_energies: &[f64], gpu_energies: &[f32],
    cpu_charges: &[f64], gpu_charges: &[f32],
    n_atoms: usize, h1: usize, h2: usize, d1: usize, a1: usize, d2: usize, a2: usize,
) {
    let mut f = fs::File::create(path).unwrap();
    writeln!(f, "# Formic dimer 1D synchronous proton-transfer scan").unwrap();
    writeln!(f, "# t  E_cpu  E_gpu  q_cpu[H1]  q_gpu[H1]  q_cpu[D1]  q_gpu[D1]  q_cpu[A1]  q_gpu[A1]  q_cpu[H2]  q_gpu[H2]  q_cpu[D2]  q_gpu[D2]  q_cpu[A2]  q_gpu[A2]").unwrap();
    for i in 0..t_values.len() {
        write!(f, "{:.4}  {:.8}  {:.8}", t_values[i], cpu_energies[i], gpu_energies[i]).unwrap();
        for &atom in &[h1, d1, a1, h2, d2, a2] {
            write!(f, "  {:.6}  {:.6}", cpu_charges[i*n_atoms + atom], gpu_charges[i*n_atoms + atom]).unwrap();
        }
        writeln!(f).unwrap();
    }
    eprintln!("  Saved 1D data to {path}");
}

fn save_2d_data(
    path: &str, n: usize,
    t_values: &[f64], energies: &[f32], charges: &[f32],
    n_atoms: usize, atom_idx: usize,
) {
    let mut f = fs::File::create(path).unwrap();
    writeln!(f, "# Formic dimer 2D asynchronous proton-transfer scan").unwrap();
    writeln!(f, "# n_t1={n} n_t2={n}").unwrap();
    writeln!(f, "# i  j  t1  t2  E  q[atom={atom_idx}]  converged").unwrap();
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let conv = if energies[idx].is_nan() { 0 } else { 1 };
            let e_str = if energies[idx].is_nan() { "NaN".to_string() } else { format!("{:.8}", energies[idx]) };
            let q_str = if energies[idx].is_nan() { "NaN".to_string() } else { format!("{:.6}", charges[idx * n_atoms + atom_idx]) };
            writeln!(f, "{i}  {j}  {:.4}  {:.4}  {e_str}  {q_str}  {conv}",
                t_values[i], t_values[j]).unwrap();
        }
    }
    eprintln!("  Saved 2D data to {path}");
}

#[test]
#[ignore = "scan + plot generator: run with --ignored --nocapture"]
fn test_formic_dimer_scan_plots() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    let out_dir = format!("{}/debug/formic_dimer_scan", env!("CARGO_MANIFEST_DIR"));
    fs::create_dir_all(&out_dir).unwrap();

    // Load geometry
    let xyz_path = format!("{}/../data/xyz/formic_dimer.xyz", env!("CARGO_MANIFEST_DIR"));
    let xyz = parse_xyz(&xyz_path).unwrap();
    let base_coords = xyz.coords.clone();
    let species = xyz.species.clone();
    let n_atoms = species.len();

    // H-bond atoms (0-based)
    let h1 = 4; let d1 = 3; let a1 = 7;  // H-O(3)...O(7)
    let h2 = 9; let d2 = 8; let a2 = 2;  // H-O(8)...O(2)

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();

    // Get n_orbs, n_occ from a single CPU calculation
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc0 = builder.build_scc(&species, &base_coords, 200, 1e-9).unwrap();
    let n_orbs = scc0.h0.nrows();
    let n_occ = (scc0.q0.iter().sum::<f64>() / 2.0).round() as usize;
    eprintln!("Formic dimer: N_orbs={n_orbs}, N_occ={n_occ}, N_atoms={n_atoms}");
    eprintln!("H-bond 1: H[{h1}] donor O[{d1}] acceptor O[{a1}]");
    eprintln!("H-bond 2: H[{h2}] donor O[{d2}] acceptor O[{a2}]");

    // ===================== 1D SCAN =====================
    eprintln!("\n=== 1D synchronous scan (41 points) ===");
    let n_1d = 41usize;
    let t_1d: Vec<f64> = (0..n_1d).map(|i| 2.0 * i as f64 / (n_1d - 1) as f64).collect();
    let geoms_1d: Vec<Vec<[f64; 3]>> = t_1d.iter()
        .map(|&t| make_scan_geom_1d(&base_coords, h1, d1, a1, h2, d2, a2, t))
        .collect();

    // CPU reference
    eprintln!("  [CPU] Running SCC for {} points...", n_1d);
    let mut cpu_e_1d = Vec::with_capacity(n_1d);
    let mut cpu_q_1d = Vec::with_capacity(n_1d * n_atoms);
    for (i, coords) in geoms_1d.iter().enumerate() {
        let scc = builder.build_scc(&species, coords, 200, 1e-9)
            .unwrap_or_else(|e| panic!("CPU SCC failed at 1D point {i}: {e}"));
        cpu_e_1d.push(scc.energy);
        cpu_q_1d.extend(scc.charges.iter().map(|&q| q as f32 as f64));
        if i % 10 == 0 {
            eprintln!("    [CPU] 1D t={:.2}: E={:.8}", t_1d[i], scc.energy);
        }
    }

    // GPU
    let (gpu_e_1d, gpu_q_1d, _) = run_gpu_scan(
        &sk, &gamma, &species, &geoms_1d, n_atoms, n_orbs, n_occ, "1D scan", 5e-6, None, false,
    );

    // Save
    let path_1d = format!("{out_dir}/scan_1d.tsv");
    save_1d_data(&path_1d, &t_1d, &cpu_e_1d, &gpu_e_1d, &cpu_q_1d, &gpu_q_1d,
        n_atoms, h1, h2, d1, a1, d2, a2);

    // ===================== 2D SCAN =====================
    eprintln!("\n=== 2D asynchronous scan (21×21 = 441 points) ===");
    let n_2d = 21usize;
    let t_2d: Vec<f64> = (0..n_2d).map(|i| 2.0 * i as f64 / (n_2d - 1) as f64).collect();

    // Solve 2D scan strip by strip (21 strips of 21 points each).
    // Each strip uses the previous strip's converged charges as warm-start.
    // The first strip uses the 1D converged charges at t1 as warm-start.
    let n_1d = t_1d.len();
    let dt_1d = 2.0 / (n_1d - 1) as f64;
    let mut gpu_e_2d = vec![0.0f32; n_2d * n_2d];
    let mut gpu_q_2d = vec![0.0f32; n_2d * n_2d * n_atoms];

    for i in 0..n_2d {
        let t1 = t_2d[i];
        let strip_geoms: Vec<Vec<[f64; 3]>> = (0..n_2d)
            .map(|j| make_scan_geom_2d(&base_coords, h1, d1, a1, h2, d2, a2, t1, t_2d[j]))
            .collect();

        // CPU reference for first strip to verify convergence is possible
        if i == 0 {
            eprintln!("  [CPU] Checking SCC for first strip (t1=0.0)...");
            for j in [0usize, 10, 16, 20] {
                let scc = builder.build_scc(&species, &strip_geoms[j], 200, 1e-9);
                match scc {
                    Ok(s) => eprintln!("    [CPU] 2D t1=0.0 t2={:.1}: E={:.8}", t_2d[j], s.energy),
                    Err(e) => eprintln!("    [CPU] 2D t1=0.0 t2={:.1}: FAILED: {e}", t_2d[j]),
                }
            }
        }

        // Best-effort: some asymmetric 2D points don't converge (CPU also fails).
        // Mark unconverged points with NaN energy.
        let (strip_e, strip_q, strip_rms) = run_gpu_scan(
            &sk, &gamma, &species, &strip_geoms, n_atoms, n_orbs, n_occ,
            &format!("2D strip i={i} t1={t1:.2}"), 5e-6, None, true,
        );

        // Store results in the 2D grid (NaN for unconverged)
        for j in 0..n_2d {
            let idx = i * n_2d + j;
            if strip_rms[j] < 5e-6 {
                gpu_e_2d[idx] = strip_e[j];
                gpu_q_2d[idx * n_atoms..(idx + 1) * n_atoms]
                    .copy_from_slice(&strip_q[j * n_atoms..(j + 1) * n_atoms]);
            } else {
                gpu_e_2d[idx] = f32::NAN;  // mark as unconverged
                eprintln!("  [GPU] 2D point ({t1:.1},{:.1}) UNCONVERGED: RMS={:.2e}", t_2d[j], strip_rms[j]);
            }
        }
        eprintln!("  [GPU] 2D strip i={i} done");
    }

    // Save 2D energy
    let path_2d_e = format!("{out_dir}/scan_2d_energy.tsv");
    save_2d_data(&path_2d_e, n_2d, &t_2d, &gpu_e_2d, &gpu_q_2d, n_atoms, h1);

    // Save 2D charge on H1
    let path_2d_q = format!("{out_dir}/scan_2d_charge_H1.tsv");
    save_2d_data(&path_2d_q, n_2d, &t_2d, &gpu_e_2d, &gpu_q_2d, n_atoms, h1);

    // Convergence summary
    let n_conv_2d = gpu_e_2d.iter().filter(|e| !e.is_nan()).count();
    let n_total_2d = n_2d * n_2d;
    eprintln!("\n=== 2D convergence: {n_conv_2d}/{n_total_2d} points converged ===");
    eprintln!("  (CPU also fails on highly asymmetric points — see diagnostics above)");

    eprintln!("\n=== Done. Data saved to {out_dir}/ ===");
    eprintln!("  1D: {path_1d}");
    eprintln!("  2D: {path_2d_e}, {path_2d_q}");
    eprintln!("Run: python3 scripts/plot_formic_scan.py");
}
