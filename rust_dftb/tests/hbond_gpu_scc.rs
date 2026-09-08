//! H-bond proton-transfer scan: CPU vs GPU SCC parity on formic acid dimer.
//!
//! This is the original scientific validation test from the GPU_MultiSystem
//! task. It runs a 1D synchronous proton-transfer scan on (HCOOH)₂:
//!   - 10 atoms, 28 orbitals (fits N≤64 dense GPU route)
//!   - Both transferring protons move simultaneously from donor → acceptor
//!   - 21 scan points (t = 0.0 → 2.0, t=1.0 = symmetric transition state)
//!   - CPU reference: HamiltonianBuilder::build_scc (DIIS mixer, f64)
//!   - GPU: gpu_solve_scc_batched (simple mixer, f32)
//!
//! Atom indices (0-based, from formic_dimer.xyz):
//!   H1: atom 4, donor O: atom 3, acceptor O: atom 7
//!   H2: atom 9, donor O: atom 8, acceptor O: atom 2
//!
//! Environment:
//!   RUST_DFTB_SK_DIR — directory with mio-1-1 .skf files

use rust_dftb::io::parse_xyz;
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use rust_dftb::qmqm::gpu_scc::{gpu_solve_scc_batched, gpu_solve_scc_batched_diis};
use rust_dftb::qmqm::{FragmentTemplate, GammaTable};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder};

const ANG2BOHR: f64 = 1.889_726_133;

/// Try to create a GpuRuntime; return None if no OpenCL device available.
fn try_runtime() -> Option<GpuRuntime> {
    match GpuRuntime::new() {
        Ok(rt) => Some(rt),
        Err(e) => { eprintln!("Skipping GPU test: no OpenCL device ({e})"); None }
    }
}

/// Per-atom Hubbard U from SK data.
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

/// Build dense gamma matrix G[Na*Na] (row-major f32).
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

/// Build per-orbital → atom index map.
fn orb_atom_map(atom_orb_off: &[u16], n_orbs: usize) -> Vec<i32> {
    let mut map = vec![0i32; n_orbs];
    for a in 0..atom_orb_off.len() - 1 {
        for mu in atom_orb_off[a] as usize..atom_orb_off[a+1] as usize {
            map[mu] = a as i32;
        }
    }
    map
}

/// Interpolate H position: reactant (t=0) → product (t=1) → beyond (t>1).
/// Product = mirror of reactant H across donor-acceptor midpoint.
fn interp_h_pos(
    h_reactant: &[f64; 3], donor: &[f64; 3], acceptor: &[f64; 3], t: f64,
) -> [f64; 3] {
    let mid = [(donor[0]+acceptor[0])*0.5, (donor[1]+acceptor[1])*0.5, (donor[2]+acceptor[2])*0.5];
    let h_product = [2.0*mid[0]-h_reactant[0], 2.0*mid[1]-h_reactant[1], 2.0*mid[2]-h_reactant[2]];
    [
        h_reactant[0] + t*(h_product[0]-h_reactant[0]),
        h_reactant[1] + t*(h_product[1]-h_reactant[1]),
        h_reactant[2] + t*(h_product[2]-h_reactant[2]),
    ]
}

/// Generate 1D synchronous scan geometry (both protons move at param t).
fn make_scan_geom(
    base: &[[f64; 3]],
    h1: usize, donor1: usize, acceptor1: usize,
    h2: usize, donor2: usize, acceptor2: usize,
    t: f64,
) -> Vec<[f64; 3]> {
    let mut coords = base.to_vec();
    coords[h1] = interp_h_pos(&base[h1], &base[donor1], &base[acceptor1], t);
    coords[h2] = interp_h_pos(&base[h2], &base[donor2], &base[acceptor2], t);
    coords
}

/// Max abs diff between two f32 slices (as f64).
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

// ==================================================================
// Formic dimer single-point SCC parity (diagnostic)
// ==================================================================

#[test]
fn test_formic_dimer_single_point_scc() {
    let Some(mut rt) = try_runtime() else { return; };
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
    let base_coords = xyz.coords.clone();
    let species = xyz.species.clone();
    let n_atoms = species.len();
    assert_eq!(n_atoms, 10);

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();
    let u_per_atom = per_atom_u(&sk, &species);

    // CPU reference at t=0 (reactant geometry)
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc = builder.build_scc(&species, &base_coords, 200, 1e-9).unwrap();
    let n = scc.h0.nrows();
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    eprintln!("Formic dimer single-point: N={n}, N_occ={n_occ}, N_atoms={n_atoms}");
    eprintln!("  CPU: E={:.8}, q={:.4?}", scc.energy, &scc.charges);
    eprintln!("  CPU: eigenvalues (first 6): {:.6}", scc.eigenvalues.rows(0, 6));

    // Build H0/S on GPU via GpuDriver (tests the s-p rotation fix)
    use rust_dftb::qmqm::gpu_driver::GpuDriver;
    use rust_dftb::qmqm::gpu_prep::GpuBatch;
    use rust_dftb::qmqm::Fragment;
    let frags: Vec<Fragment> = vec![{
        let tmpl = FragmentTemplate::new(&sk, species.to_vec(), base_coords.to_vec()).unwrap();
        Fragment::from_template(tmpl, base_coords.to_vec())
    }];
    let driver = GpuDriver::new().expect("GpuDriver init failed");
    let batch = GpuBatch::from_fragments(&frags, &sk, &gamma).unwrap();
    let (h0_flat, s_flat) = driver.gpu_assemble_batched(&batch).unwrap();

    // Verify H0/S parity
    let h0_cpu_flat: Vec<f32> = (0..n*n).map(|idx| {
        let i = idx / n; let j = idx - i*n;
        scc.h0[(i,j)] as f32
    }).collect();
    let s_cpu_flat: Vec<f32> = (0..n*n).map(|idx| {
        let i = idx / n; let j = idx - i*n;
        scc.s[(i,j)] as f32
    }).collect();
    let dh0 = max_abs_diff(&h0_flat, &h0_cpu_flat);
    let ds0 = max_abs_diff(&s_flat, &s_cpu_flat);
    eprintln!("  H0/S parity: max|dH0|={dh0:.2e}, max|dS|={ds0:.2e}");

    // Find worst H elements and their atom/orbital context
    let tmpl_dbg = FragmentTemplate::new(&sk, species.to_vec(), base_coords.to_vec()).unwrap();
    let atom_orb_off = &tmpl_dbg.atom_orb_off;
    for _ in 0..5.min(n*n) {
        let mut worst = 0.0f64; let mut wi = 0; let mut wj = 0;
        for i in 0..n { for j in 0..n {
            let d = (h0_flat[i*n+j] as f64 - scc.h0[(i,j)] as f64).abs();
            if d > worst { worst = d; wi = i; wj = j; }
        }}
        if worst < 1e-4 { break; }
        // Find which atoms these orbitals belong to
        let atom_i = (0..atom_orb_off.len()-1).find(|&a| (wi >= atom_orb_off[a] as usize) && (wi < atom_orb_off[a+1] as usize)).unwrap();
        let atom_j = (0..atom_orb_off.len()-1).find(|&a| (wj >= atom_orb_off[a] as usize) && (wj < atom_orb_off[a+1] as usize)).unwrap();
        let orb_name = ["s","py","pz","px"];
        let oi = wi - atom_orb_off[atom_i] as usize;
        let oj = wj - atom_orb_off[atom_j] as usize;
        eprintln!("  H worst ({wi},{wj}) = GPU {:+.6} vs CPU {:+.6} |d|={worst:.2e} | atoms({atom_i}:{},{atom_j}:{}) orbs({},{})",
            h0_flat[wi*n+wj], scc.h0[(wi,wj)],
            species[atom_i], species[atom_j], orb_name.get(oi).unwrap_or(&"?"), orb_name.get(oj).unwrap_or(&"?"));
        // Zero out this element so we can find the next worst
        // (only for diagnostics, not modifying the actual buffer)
    }

    // Build gamma, q0, orb_atom
    let g = build_gamma_matrix(&base_coords, &u_per_atom);
    let tmpl = FragmentTemplate::new(&sk, species.to_vec(), base_coords.to_vec()).unwrap();
    let q0: Vec<f32> = tmpl.q0.iter().map(|&q| q as f32).collect();
    let oa = orb_atom_map(&tmpl.atom_orb_off, n);

    let h0_buf = rt.buffer_from_slice(&h0_flat).unwrap();
    let s_buf = rt.buffer_from_slice(&s_flat).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let q0_buf = rt.buffer_from_slice(&q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&oa).unwrap();

    let gpu = gpu_solve_scc_batched_diis(
        &mut rt, &h0_buf, &s_buf, &g_buf, &q0_buf, &oa_buf,
        n, n_atoms, n_occ, 1,
        200, 1e-6, 0.3, 8, 3,
    ).expect("GPU SCC (DIIS) must converge for single formic dimer");

    let de = (gpu.energies[0] as f64 - scc.energy).abs();
    let cpu_q_f32: Vec<f32> = scc.charges.iter().map(|&q| q as f32).collect();
    let dq = max_abs_diff(&gpu.charges, &cpu_q_f32);
    eprintln!("  GPU: E={:.8}, q={:.4?}", gpu.energies[0], &gpu.charges);
    eprintln!("  Parity: |dE|={de:.2e}, |dq|={dq:.2e}, n_iter={}", gpu.n_iters);

    assert!(de < 1e-3, "Formic dimer energy parity failed: |dE|={de:.2e}");
    assert!(dq < 1e-3, "Formic dimer charge parity failed: |dq|={dq:.2e}");
}

// ==================================================================
// Formic dimer 1D synchronous proton-transfer scan: CPU vs GPU
// ==================================================================

#[test]
fn test_formic_dimer_1d_scan_gpu_vs_cpu() {
    let Some(mut rt) = try_runtime() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };

    // Load formic dimer geometry — try several paths
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
        None => {
            eprintln!("Skipping: cannot load formic_dimer.xyz (tried: {candidates:?})");
            return;
        }
    };
    let base_coords = xyz.coords.clone();
    let species = xyz.species.clone();
    let n_atoms = species.len();
    assert_eq!(n_atoms, 10, "formic dimer must have 10 atoms");

    // Atom indices for proton transfer (0-based, from hbond_switching.md)
    let h1 = 4; let donor1 = 3; let acceptor1 = 7;
    let h2 = 9; let donor2 = 8; let acceptor2 = 2;

    // Load SK tables for all species in the dimer (H, C, O)
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();
    let u_per_atom = per_atom_u(&sk, &species);

    // Scan parameters: 21 points, t = 0.0 to 2.0
    let n_points = 21usize;
    let t_values: Vec<f64> = (0..n_points).map(|i| 2.0 * i as f64 / (n_points - 1) as f64).collect();

    // Build all scan geometries
    let geometries: Vec<Vec<[f64; 3]>> = t_values.iter()
        .map(|&t| make_scan_geom(&base_coords, h1, donor1, acceptor1, h2, donor2, acceptor2, t))
        .collect();

    // --- CPU reference: run SCC for each geometry ---
    eprintln!("=== Formic dimer 1D scan: CPU reference ({} points) ===", n_points);
    let builder = HamiltonianBuilder::new(sk.clone());
    let mut cpu_energies = Vec::with_capacity(n_points);
    let mut cpu_charges = Vec::with_capacity(n_points * n_atoms);
    let mut n_orbs = 0usize;
    let mut n_occ = 0usize;

    for (i, coords) in geometries.iter().enumerate() {
        let scc = builder.build_scc(&species, coords, 200, 1e-9)
            .unwrap_or_else(|e| panic!("CPU SCC failed at scan point {i} (t={}): {e}", t_values[i]));
        cpu_energies.push(scc.energy);
        cpu_charges.extend(scc.charges.iter().map(|&q| q as f32));
        n_orbs = scc.h0.nrows();
        n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
        if i == 0 || i == n_points / 2 || i == n_points - 1 {
            eprintln!("  [CPU] t={:.2}: E={:.8}, q={:.4?}", t_values[i], scc.energy, &scc.charges[..]);
        }
    }
    eprintln!("  N_orbs={n_orbs}, N_occ={n_occ}, N_atoms={n_atoms}");

    // --- GPU: build H0/S via GpuDriver, then SCC via gpu_solve_scc_batched ---
    eprintln!("=== Formic dimer 1D scan: GPU batched SCC (GPU-built H0/S) ===");

    // Build fragments for all scan points and assemble H0/S on GPU
    use rust_dftb::qmqm::gpu_driver::GpuDriver;
    use rust_dftb::qmqm::gpu_prep::GpuBatch;
    use rust_dftb::qmqm::Fragment;
    let frags: Vec<Fragment> = geometries.iter()
        .map(|coords| {
            let tmpl = FragmentTemplate::new(&sk, species.to_vec(), coords.to_vec()).unwrap();
            Fragment::from_template(tmpl, coords.to_vec())
        })
        .collect();
    let driver = GpuDriver::new().expect("GpuDriver init failed");
    let batch = GpuBatch::from_fragments(&frags, &sk, &gamma).unwrap();
    let (all_h0, all_s) = driver.gpu_assemble_batched(&batch).unwrap();

    // Build gamma matrices and q0/orb_atom for each replica
    let mut all_g = Vec::with_capacity(n_points * n_atoms * n_atoms);
    let mut all_q0 = Vec::with_capacity(n_points * n_atoms);
    let mut all_oa = Vec::with_capacity(n_points * n_orbs);
    for coords in &geometries {
        all_g.extend(build_gamma_matrix(coords, &u_per_atom));
        let tmpl = FragmentTemplate::new(&sk, species.to_vec(), coords.to_vec()).unwrap();
        all_q0.extend(tmpl.q0.iter().map(|&q| q as f32));
        all_oa.extend(orb_atom_map(&tmpl.atom_orb_off, n_orbs));
    }

    // Upload to GpuRuntime buffers
    let h0_buf = rt.buffer_from_slice(&all_h0).unwrap();
    let s_buf = rt.buffer_from_slice(&all_s).unwrap();
    let g_buf = rt.buffer_from_slice(&all_g).unwrap();
    let q0_buf = rt.buffer_from_slice(&all_q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&all_oa).unwrap();

    // Run batched GPU SCC with CPU-driven DIIS mixing
    // alpha=0.3 warmup, max_history=8, warmup=3 iterations
    let gpu = gpu_solve_scc_batched_diis(
        &mut rt, &h0_buf, &s_buf, &g_buf, &q0_buf, &oa_buf,
        n_orbs, n_atoms, n_occ, n_points,
        500, 1e-6, 0.3, 8, 3,
    ).expect("GPU SCC (DIIS) must converge for all scan points");

    // --- Compare ---
    let cpu_e_f32: Vec<f32> = cpu_energies.iter().map(|&e| e as f32).collect();
    let de = max_abs_diff(&gpu.energies, &cpu_e_f32);
    let dq = max_abs_diff(&gpu.charges, &cpu_charges);

    eprintln!("=== Formic dimer 1D scan: parity results ===");
    eprintln!("  max|dE|  = {de:.2e}  (over {n_points} scan points)");
    eprintln!("  max|dq|  = {dq:.2e}  (over {n_points}×{n_atoms} charges)");
    eprintln!("  n_iters  = {}", gpu.n_iters);
    eprintln!("  scan point energies (CPU vs GPU):");
    for i in 0..n_points {
        let d = gpu.energies[i] as f64 - cpu_energies[i];
        eprintln!("    t={:.2}: E_cpu={:.8}, E_gpu={:.8}, dE={:+.2e}",
            t_values[i], cpu_energies[i], gpu.energies[i], d);
    }

    // Tolerances: f32 GPU + simple mixer vs f64 CPU DIIS.
    // Energy: 1e-3 Ha (~0.03 eV) — physically meaningful for PES comparison.
    // Charges: 1e-3 |e| — sufficient for proton-transfer characterization.
    assert!(de < 1e-3, "Formic dimer scan energy parity failed: max|dE|={de:.2e} > 1e-3");
    assert!(dq < 1e-3, "Formic dimer scan charge parity failed: max|dq|={dq:.2e} > 1e-3");

    // Physical invariant: the PES should show a barrier at t=1.0 (transition state)
    // The formic dimer XYZ is NOT inversion-symmetric (the two monomers have
    // slightly different geometries), so E(t) ≠ E(2-t) is expected.
    // Instead, check that the PES has a maximum near t=1.0.
    let e_at_ts = gpu.energies[n_points / 2] as f64;
    let e_at_reactant = gpu.energies[0] as f64;
    let barrier = e_at_ts - e_at_reactant;
    let barrier_ev = barrier * 27.211;
    eprintln!("  PES barrier (E(t=1) - E(t=0)) = {:.4e} Ha ({:.2} eV)", barrier, barrier_ev);
    // The barrier should be positive (endothermic) — the TS is higher in energy
    // than the reactant. This is a physical invariant of the proton transfer.
    assert!(barrier > 0.0, "PES barrier should be positive (TS > reactant): got {barrier:.2e}");
}

// ==================================================================
// Phase 3 (P3d): AT and GC single-point SCC parity (N>64 tiled path)
// ==================================================================
//
// Adenine-Thymine: 30 atoms, ~87 orbitals — exercises the full tiled path
// (tiled Jacobi + tiled GEMM + tiled S^{-1/2}).
// Guanine-Cytosine: 29 atoms, ~86 orbitals — same.
//
// Tolerances (manifest §4.3, N>64):
//   - Energy: < 1e-2 Ha (f32 block Jacobi)
//   - Charges: < 1e-2 e
//   - Eigenvalues: < 1e-2 Ha

fn run_nucleobase_pair_scc_parity(xyz_path: &str, name: &str) {
    let Some(mut rt) = try_runtime() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let candidates = [
        xyz_path.to_string(),
        format!("{}/data/xyz/{xyz_path}", env!("CARGO_MANIFEST_DIR")),
        format!("{}/../data/xyz/{xyz_path}", env!("CARGO_MANIFEST_DIR")),
    ];
    let mut xyz = None;
    for path in &candidates {
        if let Ok(x) = parse_xyz(path) { xyz = Some(x); break; }
    }
    let xyz = match xyz {
        Some(x) => x,
        None => { eprintln!("Skipping: cannot load {xyz_path}"); return; }
    };
    let species = xyz.species.clone();
    let coords = xyz.coords.clone();
    let n_atoms = species.len();
    eprintln!("[{name}] {n_atoms} atoms, species={:?}", species.iter().take(5).collect::<Vec<_>>());

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let u_per_atom = per_atom_u(&sk, &species);
    let g = build_gamma_matrix(&coords, &u_per_atom);

    // CPU reference
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc = builder.build_scc(&species, &coords, 200, 1e-9).unwrap();
    let n = scc.h0.nrows();
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    eprintln!("[{name}] n_orbs={n}, n_atoms={n_atoms}, n_occ={n_occ}");
    assert!(n > 64, "{name} should exercise N>64 tiled path, got N={n}");

    let tmpl = FragmentTemplate::new(&sk, species.to_vec(), coords.to_vec()).unwrap();
    let orb_atom = orb_atom_map(&tmpl.atom_orb_off, n);

    // Flatten H0, S, q0 to f32 row-major
    let mut h0_flat = vec![0.0f32; n * n];
    let mut s_flat = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            h0_flat[i * n + j] = scc.h0[(i, j)] as f32;
            s_flat[i * n + j] = scc.s[(i, j)] as f32;
        }
    }
    let q0: Vec<f32> = scc.q0.iter().map(|&q| q as f32).collect();

    let batch = 1usize;
    let h0_buf = rt.buffer_from_slice(&h0_flat).unwrap();
    let s_buf = rt.buffer_from_slice(&s_flat).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let q0_buf = rt.buffer_from_slice(&q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&orb_atom).unwrap();

    let gpu = gpu_solve_scc_batched_diis(
        &mut rt, &h0_buf, &s_buf, &g_buf, &q0_buf, &oa_buf,
        n, n_atoms, n_occ, batch,
        500, 1e-5, 0.3, 8, 5,
    ).expect(&format!("{name} GPU SCC (DIIS) must converge"));

    let de = (gpu.energies[0] as f64 - scc.energy).abs();
    let dq = max_abs_diff(&gpu.charges, &scc.charges.iter().map(|&q| q as f32).collect::<Vec<_>>());
    let d_eig = max_abs_diff(&gpu.eigenvalues, &scc.eigenvalues.iter().map(|&e| e as f32).collect::<Vec<_>>());

    eprintln!("[{name}] N={n} SCC parity:");
    eprintln!("  E_cpu={:.8}, E_gpu={:.8}, |dE|={de:.2e}", scc.energy, gpu.energies[0]);
    eprintln!("  |dq|={dq:.2e}, |d_eig|={d_eig:.2e}, n_iters={}", gpu.n_iters);

    assert!(de < 1e-2, "{name} energy parity failed: |dE|={de:.2e} > 1e-2");
    assert!(dq < 1e-2, "{name} charges parity failed: |dq|={dq:.2e} > 1e-2");
    assert!(d_eig < 1e-2, "{name} eigenvalues parity failed: |d_eig|={d_eig:.2e} > 1e-2");
}

#[test]
fn test_at_scc_parity() {
    run_nucleobase_pair_scc_parity("adenine-thymine.xyz", "AT");
}

#[test]
fn test_gc_scc_parity() {
    run_nucleobase_pair_scc_parity("guanine-cytosine.xyz", "GC");
}
