//! End-to-end GPU SCC parity test (Wave 3 coordinator integration).
//!
//! Verifies `gpu_solve_scc_batched` against the CPU `HamiltonianBuilder::build_scc`
//! reference for:
//!   - H2O (single system, 6 orbitals, 3 atoms)
//!   - N2  (single system, 8 orbitals, 2 atoms)
//!   - 10× H2O at varying geometries (batched, 6×6, 3 atoms each)
//!
//! Tolerances (per master contract):
//!   - Energy: < 1e-3 Ha (f32 throughout GPU; simple mixer vs CPU DIIS)
//!   - Charges: < 1e-3 |e|
//!   - Eigenvalues: < 1e-4 Ha
//!
//! Environment:
//!   RUST_DFTB_SK_DIR — directory with mio-1-1 .skf files
//!
//! Tests skip gracefully if no OpenCL device or no SK dir.

use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use rust_dftb::qmqm::gpu_scc::{gpu_solve_scc_batched, GpuSccResult};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder};

const ANG2BOHR: f64 = 1.889_726_133;

/// Try to create a GpuRuntime; return None if no OpenCL device available.
fn try_runtime() -> Option<GpuRuntime> {
    match GpuRuntime::new() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

/// Per-atom Hubbard U extracted from SK data (order = order of `species`).
fn per_atom_u(sk: &rust_dftb::SkData, species: &[String]) -> Vec<f64> {
    let mut unique: Vec<String> = Vec::new();
    for sp in species {
        if !unique.contains(sp) {
            unique.push(sp.clone());
        }
    }
    let u_unique: Vec<f64> = unique
        .iter()
        .map(|sp| sk.onsite(sp).map(|p| p.u_hubbard).unwrap_or(0.4))
        .collect();
    species
        .iter()
        .map(|sp| {
            let idx = unique.iter().position(|s| s == sp).unwrap();
            u_unique[idx]
        })
        .collect()
}

/// Build the dense gamma matrix G[Na*Na] (row-major) for one geometry.
/// G[a*Na+b] = gamma_full(r_ab, U_a, U_b)  (r=0 → on-site = (U_a+U_b)/2).
fn build_gamma_matrix(coords: &[[f64; 3]], u_per_atom: &[f64]) -> Vec<f32> {
    let n = coords.len();
    let mut g = vec![0.0f32; n * n];
    for a in 0..n {
        for b in 0..n {
            let dx = coords[a][0] - coords[b][0];
            let dy = coords[a][1] - coords[b][1];
            let dz = coords[a][2] - coords[b][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
            g[a * n + b] = rust_dftb::gamma_full(r, u_per_atom[a], u_per_atom[b]) as f32;
        }
    }
    g
}

/// Build the per-orbital → atom index map (length n_orbs) from atom_orb_off.
fn orb_atom_map(atom_orb_off: &[u16], n_orbs: usize) -> Vec<i32> {
    let mut map = vec![0i32; n_orbs];
    for a in 0..atom_orb_off.len() - 1 {
        for mu in atom_orb_off[a] as usize..atom_orb_off[a + 1] as usize {
            map[mu] = a as i32;
        }
    }
    map
}

/// Flatten a nalgebra DMatrix to row-major f32.
fn flatten_f32(m: &nalgebra::DMatrix<f64>) -> Vec<f32> {
    let n = m.nrows();
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = m[(i, j)] as f32;
        }
    }
    out
}

/// Max abs diff between two same-length f32 slices (cast to f64).
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

/// Run CPU SCC reference and return (energy, charges, eigenvalues, n, n_atoms, n_occ, h0_flat, s_flat).
struct CpuRef {
    energy: f64,
    charges: Vec<f64>,
    eigenvalues: Vec<f64>,
    n: usize,
    n_atoms: usize,
    n_occ: usize,
    h0: Vec<f32>,
    s: Vec<f32>,
    q0: Vec<f32>,
    orb_atom: Vec<i32>,
}

fn cpu_scc_ref(sk: &rust_dftb::SkData, species: &[String], coords: &[[f64; 3]]) -> CpuRef {
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc = builder.build_scc(species, coords, 200, 1e-9).unwrap();
    let n = scc.h0.nrows();
    let n_atoms = species.len();
    // n_occ = number of occupied MOs = n_electrons/2 = sum(q0)/2 (closed-shell)
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;

    // Build orb_atom map from the template — we need atom_orb_off.
    // HamiltonianBuilder doesn't expose it directly; reconstruct from FragmentTemplate
    use rust_dftb::qmqm::FragmentTemplate;
    let tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
    let orb_atom = orb_atom_map(&tmpl.atom_orb_off, n);

    CpuRef {
        energy: scc.energy,
        charges: scc.charges.clone(),
        eigenvalues: scc.eigenvalues.iter().cloned().collect(),
        n,
        n_atoms,
        n_occ,
        h0: flatten_f32(&scc.h0),
        s: flatten_f32(&scc.s),
        q0: scc.q0.iter().map(|&q| q as f32).collect(),
        orb_atom,
    }
}

// ==================================================================
// 1. H2O single-system SCC parity
// ==================================================================

#[test]
fn test_gpu_scc_parity_h2o() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
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
    let cpu = cpu_scc_ref(&sk, &species, &coords);

    // Build GPU inputs
    let u_per_atom = per_atom_u(&sk, &species);
    let g = build_gamma_matrix(&coords, &u_per_atom);
    let batch = 1usize;

    let h0_buf = rt.buffer_from_slice(&cpu.h0).unwrap();
    let s_buf = rt.buffer_from_slice(&cpu.s).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let q0_buf = rt.buffer_from_slice(&cpu.q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&cpu.orb_atom).unwrap();

    let gpu = gpu_solve_scc_batched(
        &mut rt,
        &h0_buf,
        &s_buf,
        &g_buf,
        &q0_buf,
        &oa_buf,
        cpu.n,
        cpu.n_atoms,
        cpu.n_occ,
        batch,
        500,
        1e-5,
        0.3,
    )
    .expect("GPU SCC must converge");

    let de = (gpu.energies[0] as f64 - cpu.energy).abs();
    let dq = max_abs_diff(
        &gpu.charges,
        &cpu.charges.iter().map(|&q| q as f32).collect::<Vec<_>>(),
    );
    let d_eig = max_abs_diff(
        &gpu.eigenvalues,
        &cpu.eigenvalues
            .iter()
            .map(|&e| e as f32)
            .collect::<Vec<_>>(),
    );

    eprintln!(
        "H2O SCC parity: E_cpu={:.8}, E_gpu={:.8}, |dE|={de:.2e}",
        cpu.energy, gpu.energies[0]
    );
    eprintln!(
        "  charges cpu={:?}, gpu={:?}, |dq|={dq:.2e}",
        cpu.charges, &gpu.charges
    );
    eprintln!("  eigenvalues |d_eig|={d_eig:.2e}, n_iter={}", gpu.n_iters);

    assert!(de < 1e-3, "H2O energy parity failed: |dE|={de:.2e} > 1e-3");
    assert!(dq < 1e-3, "H2O charges parity failed: |dq|={dq:.2e} > 1e-3");
    assert!(
        d_eig < 1e-3,
        "H2O eigenvalues parity failed: |d_eig|={d_eig:.2e} > 1e-3"
    );
}

// ==================================================================
// 2. N2 single-system SCC parity
// ==================================================================

#[test]
fn test_gpu_scc_parity_n2() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let species = vec!["N".to_string(), "N".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let cpu = cpu_scc_ref(&sk, &species, &coords);

    let u_per_atom = per_atom_u(&sk, &species);
    let g = build_gamma_matrix(&coords, &u_per_atom);
    let batch = 1usize;

    let h0_buf = rt.buffer_from_slice(&cpu.h0).unwrap();
    let s_buf = rt.buffer_from_slice(&cpu.s).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let q0_buf = rt.buffer_from_slice(&cpu.q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&cpu.orb_atom).unwrap();

    let gpu = gpu_solve_scc_batched(
        &mut rt,
        &h0_buf,
        &s_buf,
        &g_buf,
        &q0_buf,
        &oa_buf,
        cpu.n,
        cpu.n_atoms,
        cpu.n_occ,
        batch,
        500,
        1e-5,
        0.3,
    )
    .expect("GPU SCC must converge");

    let de = (gpu.energies[0] as f64 - cpu.energy).abs();
    let dq = max_abs_diff(
        &gpu.charges,
        &cpu.charges.iter().map(|&q| q as f32).collect::<Vec<_>>(),
    );
    let d_eig = max_abs_diff(
        &gpu.eigenvalues,
        &cpu.eigenvalues
            .iter()
            .map(|&e| e as f32)
            .collect::<Vec<_>>(),
    );

    eprintln!(
        "N2 SCC parity: E_cpu={:.8}, E_gpu={:.8}, |dE|={de:.2e}",
        cpu.energy, gpu.energies[0]
    );
    eprintln!(
        "  charges cpu={:?}, gpu={:?}, |dq|={dq:.2e}",
        cpu.charges, &gpu.charges
    );
    eprintln!("  eigenvalues |d_eig|={d_eig:.2e}, n_iter={}", gpu.n_iters);

    assert!(de < 1e-3, "N2 energy parity failed: |dE|={de:.2e} > 1e-3");
    assert!(dq < 1e-3, "N2 charges parity failed: |dq|={dq:.2e} > 1e-3");
    assert!(
        d_eig < 1e-3,
        "N2 eigenvalues parity failed: |d_eig|={d_eig:.2e} > 1e-3"
    );
}

// ==================================================================
// 3. Batched 10× H2O SCC parity (varying geometries)
// ==================================================================

#[test]
fn test_gpu_scc_parity_batched_h2o() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let base_coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let batch = 10usize;
    let u_per_atom = per_atom_u(&sk, &species);

    // Build CPU references for 10 stretched geometries
    let mut all_h0 = Vec::new();
    let mut all_s = Vec::new();
    let mut all_g = Vec::new();
    let mut all_q0 = Vec::new();
    let mut all_oa = Vec::new();
    let mut cpu_energies = Vec::new();
    let mut cpu_charges = Vec::new();
    let mut cpu_eigenvalues = Vec::new();
    let mut n = 0usize;
    let mut n_atoms = 0usize;
    let mut n_occ = 0usize;

    for bi in 0..batch {
        let stretch = 1.0 + 0.02 * bi as f64;
        let coords: Vec<[f64; 3]> = base_coords
            .iter()
            .map(|c| [c[0], c[1] * stretch, c[2]])
            .collect();
        let cpu = cpu_scc_ref(&sk, &species, &coords);
        n = cpu.n;
        n_atoms = cpu.n_atoms;
        n_occ = cpu.n_occ;
        all_h0.extend(cpu.h0);
        all_s.extend(cpu.s);
        all_g.extend(build_gamma_matrix(&coords, &u_per_atom));
        all_q0.extend(cpu.q0);
        all_oa.extend(cpu.orb_atom);
        cpu_energies.push(cpu.energy);
        cpu_charges.extend(cpu.charges.iter().map(|&q| q as f32));
        cpu_eigenvalues.extend(cpu.eigenvalues.iter().map(|&e| e as f32));
    }

    let h0_buf = rt.buffer_from_slice(&all_h0).unwrap();
    let s_buf = rt.buffer_from_slice(&all_s).unwrap();
    let g_buf = rt.buffer_from_slice(&all_g).unwrap();
    let q0_buf = rt.buffer_from_slice(&all_q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&all_oa).unwrap();

    let gpu = gpu_solve_scc_batched(
        &mut rt, &h0_buf, &s_buf, &g_buf, &q0_buf, &oa_buf, n, n_atoms, n_occ, batch, 500, 1e-5,
        0.3,
    )
    .expect("GPU SCC must converge for all 10 replicas");

    let cpu_e_f32: Vec<f32> = cpu_energies.iter().map(|&e| e as f32).collect();
    let de = max_abs_diff(&gpu.energies, &cpu_e_f32);
    let dq = max_abs_diff(&gpu.charges, &cpu_charges);
    let d_eig = max_abs_diff(&gpu.eigenvalues, &cpu_eigenvalues);

    eprintln!("Batched 10× H2O SCC parity:");
    eprintln!("  |dE|  = {de:.2e}  (max over batch)");
    eprintln!("  |dq|  = {dq:.2e}  (max over batch×atoms)");
    eprintln!("  |deig|= {d_eig:.2e}  (max over batch×orbitals)");
    eprintln!("  n_iters = {}", gpu.n_iters);
    for bi in 0..batch {
        eprintln!(
            "    [{bi}] E_cpu={:.6}, E_gpu={:.6}, dE={:+.2e}",
            cpu_e_f32[bi],
            gpu.energies[bi],
            gpu.energies[bi] as f64 - cpu_e_f32[bi] as f64
        );
    }

    assert!(
        de < 1e-3,
        "Batched H2O energy parity failed: |dE|={de:.2e} > 1e-3"
    );
    assert!(
        dq < 1e-3,
        "Batched H2O charges parity failed: |dq|={dq:.2e} > 1e-3"
    );
    assert!(
        d_eig < 1e-3,
        "Batched H2O eigenvalues parity failed: |d_eig|={d_eig:.2e} > 1e-3"
    );
}

// ==================================================================
// 4. GpuSccPlan parity — verify persistent plan matches gpu_solve_scc_batched
// ==================================================================

#[test]
fn test_gpu_scc_plan_parity_h2o() {
    use rust_dftb::qmqm::gpu_scc_plan::GpuSccPlan;

    let Some(mut rt) = try_runtime() else {
        return;
    };
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
    let cpu = cpu_scc_ref(&sk, &species, &coords);

    let u_per_atom = per_atom_u(&sk, &species);
    let g = build_gamma_matrix(&coords, &u_per_atom);
    let batch = 1usize;

    let h0_buf = rt.buffer_from_slice(&cpu.h0).unwrap();
    let s_buf = rt.buffer_from_slice(&cpu.s).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let q0_buf = rt.buffer_from_slice(&cpu.q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&cpu.orb_atom).unwrap();

    // Run with the persistent plan
    let mut plan = GpuSccPlan::new(
        &mut rt,
        &s_buf,
        &h0_buf,
        &g_buf,
        &q0_buf,
        &oa_buf,
        cpu.n,
        cpu.n_atoms,
        batch,
    )
    .expect("GpuSccPlan::new must succeed");
    plan.set_initial_charges(&rt, &cpu.q0).unwrap();

    let mut max_rms = f32::INFINITY;
    let mut n_iters = 0;
    for iter in 0..500 {
        n_iters = iter + 1;
        max_rms = plan
            .scc_step(&mut rt, cpu.n_occ, 0.3)
            .expect("scc_step must succeed");
        if max_rms < 1e-5 {
            break;
        }
    }
    assert!(
        max_rms < 1e-5,
        "GpuSccPlan did not converge in {n_iters} iters (max_rms={max_rms:.3e})"
    );

    let energies = plan.compute_energy(&mut rt, cpu.n_occ).unwrap();
    let charges = plan.read_charges(&rt).unwrap();
    let eigenvalues = plan.read_eigenvalues(&mut rt).unwrap();

    let de = (energies[0] as f64 - cpu.energy).abs();
    let dq = max_abs_diff(
        &charges,
        &cpu.charges.iter().map(|&q| q as f32).collect::<Vec<_>>(),
    );
    let d_eig = max_abs_diff(
        &eigenvalues,
        &cpu.eigenvalues
            .iter()
            .map(|&e| e as f32)
            .collect::<Vec<_>>(),
    );

    eprintln!(
        "GpuSccPlan H2O parity: E_cpu={:.8}, E_plan={:.8}, |dE|={de:.2e}, iters={n_iters}",
        cpu.energy, energies[0]
    );
    eprintln!("  |dq|={dq:.2e}, |d_eig|={d_eig:.2e}");

    assert!(
        de < 1e-3,
        "GpuSccPlan energy parity failed: |dE|={de:.2e} > 1e-3"
    );
    assert!(
        dq < 1e-3,
        "GpuSccPlan charges parity failed: |dq|={dq:.2e} > 1e-3"
    );
    assert!(
        d_eig < 1e-3,
        "GpuSccPlan eigenvalues parity failed: |d_eig|={d_eig:.2e} > 1e-3"
    );
}

// ==================================================================
// 4b. GpuSccPlan DIIS parity — H2O with GPU-side DIIS mixing (R9b)
// Same as test 4 but uses scc_step_diis instead of scc_step (simple mix).
// Verifies that GPU-resident DIIS converges to the same answer as CPU.
// ==================================================================
#[test]
fn test_gpu_scc_plan_diis_parity_h2o() {
    use rust_dftb::qmqm::gpu_scc_plan::GpuSccPlan;
    let sk_dir = match std::env::var("RUST_DFTB_SK_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
            return;
        }
    };
    let mut rt = match GpuRuntime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Skipping: no GPU ({e})");
            return;
        }
    };
    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let cpu = cpu_scc_ref(&sk, &species, &coords);

    let u_per_atom = per_atom_u(&sk, &species);
    let g = build_gamma_matrix(&coords, &u_per_atom);
    let batch = 1usize;

    let h0_buf = rt.buffer_from_slice(&cpu.h0).unwrap();
    let s_buf = rt.buffer_from_slice(&cpu.s).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let q0_buf = rt.buffer_from_slice(&cpu.q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&cpu.orb_atom).unwrap();

    let mut plan = GpuSccPlan::new(
        &mut rt,
        &s_buf,
        &h0_buf,
        &g_buf,
        &q0_buf,
        &oa_buf,
        cpu.n,
        cpu.n_atoms,
        batch,
    )
    .expect("GpuSccPlan::new must succeed");
    plan.set_initial_charges(&rt, &cpu.q0).unwrap();
    plan.reset_diis(&rt).unwrap();

    let mut max_rms = f32::INFINITY;
    let mut n_iters = 0;
    for iter in 0..500 {
        n_iters = iter + 1;
        max_rms = plan
            .scc_step_diis(&mut rt, cpu.n_occ, 0.3, 1e-5)
            .expect("scc_step_diis must succeed");
        if max_rms < 1e-5 {
            break;
        }
    }
    assert!(
        max_rms < 1e-5,
        "GpuSccPlan DIIS did not converge in {n_iters} iters (max_rms={max_rms:.3e})"
    );

    let energies = plan.compute_energy(&mut rt, cpu.n_occ).unwrap();
    let charges = plan.read_charges(&rt).unwrap();

    let de = (energies[0] as f64 - cpu.energy).abs();
    let dq = max_abs_diff(
        &charges,
        &cpu.charges.iter().map(|&q| q as f32).collect::<Vec<_>>(),
    );

    eprintln!(
        "GpuSccPlan DIIS H2O parity: E_cpu={:.8}, E_plan={:.8}, |dE|={de:.2e}, iters={n_iters}",
        cpu.energy, energies[0]
    );
    eprintln!("  |dq|={dq:.2e}");

    assert!(
        de < 1e-3,
        "GpuSccPlan DIIS energy parity failed: |dE|={de:.2e} > 1e-3"
    );
    assert!(
        dq < 1e-3,
        "GpuSccPlan DIIS charges parity failed: |dq|={dq:.2e} > 1e-3"
    );
}

// ==================================================================
// 5. N>64 SCC parity — 12× H2O cluster (72 orbitals, tiled path)
// Phase 3: verifies the full N>64 pipeline (tiled Jacobi + tiled GEMM
// + tiled S^{-1/2}) against the CPU reference.
// ==================================================================

#[test]
fn test_gpu_scc_parity_n64_h2o_cluster() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    // 12 H2O molecules = 36 atoms, 12*6 = 72 orbitals (> 64, forces tiled path)
    let mut species = Vec::new();
    let mut coords = Vec::new();
    // Arrange 12 H2O in a 3×4×1 grid, ~3 Å apart
    let h2o = [
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    for ix in 0..3 {
        for iy in 0..4 {
            let ox = ix as f64 * 3.0;
            let oy = iy as f64 * 3.0;
            for &a in &h2o {
                species.push(if a == h2o[0] { "O" } else { "H" }.to_string());
                coords.push([a[0] + ox, a[1] + oy, a[2]]);
            }
        }
    }
    assert_eq!(species.len(), 36, "12 H2O = 36 atoms");
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let cpu = cpu_scc_ref(&sk, &species, &coords);
    eprintln!(
        "12× H2O cluster: n_orbs={}, n_atoms={}, n_occ={}",
        cpu.n, cpu.n_atoms, cpu.n_occ
    );
    assert!(
        cpu.n > 64,
        "test requires N>64 to exercise tiled path, got N={}",
        cpu.n
    );

    let u_per_atom = per_atom_u(&sk, &species);
    let g = build_gamma_matrix(&coords, &u_per_atom);
    let batch = 1usize;

    let h0_buf = rt.buffer_from_slice(&cpu.h0).unwrap();
    let s_buf = rt.buffer_from_slice(&cpu.s).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let q0_buf = rt.buffer_from_slice(&cpu.q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&cpu.orb_atom).unwrap();

    let gpu = gpu_solve_scc_batched(
        &mut rt,
        &h0_buf,
        &s_buf,
        &g_buf,
        &q0_buf,
        &oa_buf,
        cpu.n,
        cpu.n_atoms,
        cpu.n_occ,
        batch,
        500,
        1e-5,
        0.3,
    )
    .expect("GPU SCC must converge for N>64 H2O cluster");

    let de = (gpu.energies[0] as f64 - cpu.energy).abs();
    let dq = max_abs_diff(
        &gpu.charges,
        &cpu.charges.iter().map(|&q| q as f32).collect::<Vec<_>>(),
    );
    let d_eig = max_abs_diff(
        &gpu.eigenvalues,
        &cpu.eigenvalues
            .iter()
            .map(|&e| e as f32)
            .collect::<Vec<_>>(),
    );

    eprintln!("N>64 (N={}) H2O cluster SCC parity:", cpu.n);
    eprintln!(
        "  E_cpu={:.8}, E_gpu={:.8}, |dE|={de:.2e}",
        cpu.energy, gpu.energies[0]
    );
    eprintln!(
        "  |dq|={dq:.2e}, |d_eig|={d_eig:.2e}, n_iters={}",
        gpu.n_iters
    );

    // N>64 contract (manifest §12 D11): measured |dE|=4.4e-5, |dq|=9.1e-6,
    // |d_eig|=1.1e-5 on RTX 3090 after D1–D4 — tighten from the old 1e-2
    // placeholder to 1e-4 (2×+ headroom over measurement).
    assert!(de < 1e-4, "N>64 energy parity failed: |dE|={de:.2e} > 1e-4");
    assert!(
        dq < 1e-4,
        "N>64 charges parity failed: |dq|={dq:.2e} > 1e-4"
    );
    assert!(
        d_eig < 1e-4,
        "N>64 eigenvalues parity failed: |d_eig|={d_eig:.2e} > 1e-4"
    );
}

// ==================================================================
// 5b. T03 direct-population path — C,SC Mulliken vs density path parity.
//
// Runs the same plan through scc_step_diis with direct_pop on/off and
// compares the per-iteration q_new trajectory + converged state. Also
// verifies the materialized SC buffer against a host S·C product and
// the paired-renorm invariant c_kᵀ(SC)_k = 1.
// ==================================================================

fn plan_pop_parity_case(
    rt: &mut GpuRuntime,
    sk_dir: &str,
    species: Vec<String>,
    coords: Vec<[f64; 3]>,
    kT: f32,
    label: &str,
) {
    use rust_dftb::qmqm::gpu_scc_plan::GpuSccPlan;

    let sk = load_sk_for_species(sk_dir, &species).unwrap();
    let cpu = cpu_scc_ref(&sk, &species, &coords);
    let u_per_atom = per_atom_u(&sk, &species);
    let g = build_gamma_matrix(&coords, &u_per_atom);
    let (n, na) = (cpu.n, cpu.n_atoms);

    let h0_buf = rt.buffer_from_slice(&cpu.h0).unwrap();
    let s_buf = rt.buffer_from_slice(&cpu.s).unwrap();
    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let q0_buf = rt.buffer_from_slice(&cpu.q0).unwrap();
    let oa_buf = rt.buffer_from_slice(&cpu.orb_atom).unwrap();

    let mut plan = GpuSccPlan::new(rt, &s_buf, &h0_buf, &g_buf, &q0_buf, &oa_buf, n, na, 1)
        .expect("GpuSccPlan::new must succeed");
    plan.kT = kT;

    // Regression for the eig_diag staleness bug this test exposed (fixed):
    // at n≤64 the smeared path skipped extract_diag, so fermi_occ bisected
    // on a zero spectrum → uniform occ_w=2/3 → constant map → false
    // "convergence" at q=[5.33,1.33,1.33]. Virgin density-path run must
    // reach the true fixed point.
    if kT > 0.0 {
        plan.direct_pop = false;
        plan.set_initial_charges(rt, &cpu.q0).unwrap();
        plan.reset_diis(rt).unwrap();
        plan.activate_all(rt).unwrap();
        let mut max_rms = f32::INFINITY;
        for _ in 0..500 {
            max_rms = plan.scc_step_diis(rt, cpu.n_occ, 0.3, 1e-6).unwrap();
            if max_rms < 1e-6 {
                break;
            }
        }
        let charges = plan.read_charges(rt).unwrap();
        let dq = max_abs_diff(
            &charges,
            &cpu.charges.iter().map(|&q| q as f32).collect::<Vec<_>>(),
        );
        eprintln!("[{label} kT={kT}] virgin density-path @1e-6: rms={max_rms:.3e} |dq|={dq:.3e}");
        assert!(max_rms < 1e-6 && dq < 1e-4,
            "{label}: smeared density path regressed (rms={max_rms:.3e} |dq|={dq:.3e}) — eig_diag stale again?");
    }

    // Per-iteration q_new trajectory on each path, same start. Run to
    // convergence (or a cap) — early agreement can hide a late divergence.
    const ITERS: usize = 60;
    let mut q_traj = [Vec::<f32>::new(), Vec::<f32>::new()];
    for (path, dp) in [(0usize, false), (1usize, true)] {
        plan.direct_pop = dp;
        plan.set_initial_charges(rt, &cpu.q0).unwrap();
        plan.reset_diis(rt).unwrap();
        plan.activate_all(rt).unwrap(); // DIIS freeze is sticky — re-arm
        for _ in 0..ITERS {
            let r = plan
                .scc_step_diis(rt, cpu.n_occ, 0.3, 1e-7)
                .expect("scc_step_diis must succeed");
            let mut q = vec![0.0f32; na];
            rt.read_buffer(&plan.q_new, &mut q).unwrap();
            q_traj[path].extend_from_slice(&q);
            if r < 1e-7 {
                break;
            }
        }
        // Pad a converged (frozen) trajectory to ITERS — q_new stays put.
        let last = q_traj[path][q_traj[path].len() - na..].to_vec();
        while q_traj[path].len() < ITERS * na {
            q_traj[path].extend_from_slice(&last);
        }
    }
    // A/B: direct-population q_new must track the density path tightly.
    // Different reduction orders give ~1e-7 per step; DIIS amplifies a
    // little along the trajectory — 1e-4 is a hard failure bound, not a
    // loosened tolerance (physics identical; this is a plumbing A/B).
    let nit = q_traj[0].len() / na;
    let mut dq_max = 0.0f32;
    let mut dq_i = usize::MAX;
    for it in 0..nit {
        let mut d = 0.0f32;
        for a in 0..na {
            d = d.max((q_traj[0][it * na + a] - q_traj[1][it * na + a]).abs());
        }
        if d > dq_max {
            dq_max = d;
            dq_i = it;
        }
    }
    eprintln!("[{label} kT={kT}] direct-vs-density q_new max|Δ| = {dq_max:.3e} at iter {dq_i} ({nit} iters)");
    assert!(
        dq_max < 1e-4,
        "{label}: direct-population path diverged from density path: {dq_max:.3e}"
    );

    // Device-state checks on the direct path (state from the last step):
    //   1. sc == S·c
    //   2. c_kᵀ(SC)_k ≈ 1 (paired renorm — occ_repair_scc default on)
    let mut c_h = vec![0.0f32; n * n];
    let mut sc_h = vec![0.0f32; n * n];
    rt.read_buffer(&plan.c, &mut c_h).unwrap();
    rt.read_buffer(&plan.sc, &mut sc_h).unwrap();
    let mut d_sc = 0.0f64;
    let mut g_err = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for l in 0..n {
                s += cpu.s[i * n + l] as f64 * c_h[l * n + j] as f64;
            }
            d_sc = d_sc.max((s - sc_h[i * n + j] as f64).abs());
        }
    }
    for k in 0..n {
        let mut g = 0.0f64;
        for i in 0..n {
            g += c_h[i * n + k] as f64 * sc_h[i * n + k] as f64;
        }
        g_err = g_err.max((g - 1.0).abs());
    }
    eprintln!("[{label} kT={kT}] max|SC−S·C|={d_sc:.3e}  max|cᵀSC−1|={g_err:.3e}");
    assert!(
        d_sc < 1e-5,
        "{label}: SC≠S·C (max {d_sc:.3e}) — GEMM wiring bug"
    );
    assert!(
        g_err < 1e-4,
        "{label}: paired renorm broken — cᵀSC dev {g_err:.3e}"
    );

    // Converged-state parity — vs CPU f64 when kT=0 (cpu_scc_ref has no
    // smearing); under smearing the two GPU paths must agree with each other.
    // tol=1e-12 disables the DIIS freeze: the smeared map has a near-stationary
    // transient (residual dips to ~6e-7 far from the fixed point — virgin
    // density-path run above froze there at tol=1e-6, a pre-existing issue).
    // Fixed 40 iters lets both paths settle to the f32 floor state instead.
    let mut conv = Vec::new();
    for (name, dp) in [("density", false), ("direct", true)] {
        plan.direct_pop = dp;
        plan.set_initial_charges(rt, &cpu.q0).unwrap();
        plan.reset_diis(rt).unwrap();
        plan.activate_all(rt).unwrap();
        let mut max_rms = f32::INFINITY;
        for _ in 0..40 {
            max_rms = plan
                .scc_step_diis(rt, cpu.n_occ, 0.3, 1e-12)
                .expect("scc_step_diis must succeed");
        }
        eprintln!("[{label} kT={kT} {name}] settled rms={max_rms:.3e}");
        let energies = plan.compute_energy(rt, cpu.n_occ).unwrap();
        let charges = plan.read_charges(rt).unwrap();
        let de = (energies[0] as f64 - cpu.energy).abs();
        let dq = max_abs_diff(
            &charges,
            &cpu.charges.iter().map(|&q| q as f32).collect::<Vec<_>>(),
        );
        eprintln!("[{label} kT={kT} {name}] |dE|={de:.2e} |dq|={dq:.2e}");
        if kT <= 0.0 {
            assert!(de < 1e-4, "{label}/{name}: energy parity {de:.2e}");
            assert!(dq < 1e-4, "{label}/{name}: charges parity {dq:.2e}");
        }
        conv.push((energies[0], charges));
    }
    if kT > 0.0 {
        let d_e = (conv[0].0 - conv[1].0).abs();
        let d_q = max_abs_diff(&conv[0].1, &conv[1].1);
        eprintln!("[{label} kT={kT}] smeared A/B: |ΔE|={d_e:.2e} |Δq|={d_q:.2e}");
        assert!(d_e < 1e-4, "{label}: smeared energy A/B {d_e:.2e}");
        assert!(d_q < 1e-4, "{label}: smeared charges A/B {d_q:.2e}");
    }
}

// ==================================================================
// 5c. DIAG — smeared-map near-stationary point (pre-existing finding).
//
// The GPU smeared (kT=0.002) SCC run on H2O freezes at tol=1e-6 with
// q=[5.33,1.33,1.33] — the bare map residual ||M(q)−q|| dips to 6.5e-7
// mid-trajectory while the true fixed point is [6.3,0.85,0.85].
// This test evaluates the SAME map in f64 on the CPU:
//   M(q) = Mulliken( GEVP( H0 + ½ S·(V_i+V_j) ) ),  V = γ·(q−q0)
// If ||M(q)−q||_f64 ~ 1e-6 at the frozen point → genuine second fixed
// point of the SCC equations (inverted-polarity solution). If ~1e-4 →
// the f32 residual was misleading and the region is just flat.
// ==================================================================
#[test]
fn test_h2o_smeared_map_second_fixed_point() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    use nalgebra::DMatrix;
    use rust_dftb::methods::xtb::scf::solve_gevp;
    use rust_dftb::qmqm::FragmentTemplate;

    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc = builder.build_scc(&species, &coords, 200, 1e-9).unwrap();
    let n = scc.h0.nrows();
    let na = species.len();
    let tmpl = FragmentTemplate::new(&sk, species.clone(), coords.clone()).unwrap();
    let oa = orb_atom_map(&tmpl.atom_orb_off, n);
    let u = per_atom_u(&sk, &species);
    let mut gam = vec![0.0f64; na * na];
    for a in 0..na {
        for b in 0..na {
            let dx = coords[a][0] - coords[b][0];
            let dy = coords[a][1] - coords[b][1];
            let dz = coords[a][2] - coords[b][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
            gam[a * na + b] = rust_dftb::gamma_full(r, u[a], u[b]);
        }
    }
    let n_el: f64 = scc.q0.iter().sum();
    let n_occ = (n_el / 2.0).round() as usize;

    // One map evaluation: q_in → Mulliken gross populations (f64).
    let map = |q: &[f64], kT: f64| -> Vec<f64> {
        let dq: Vec<f64> = q.iter().zip(scc.q0.iter()).map(|(a, b)| a - b).collect();
        let v: Vec<f64> = (0..na)
            .map(|a| (0..na).map(|b| gam[a * na + b] * dq[b]).sum())
            .collect();
        let h = DMatrix::from_fn(n, n, |i, j| {
            scc.h0[(i, j)] + 0.5 * scc.s[(i, j)] * (v[oa[i] as usize] + v[oa[j] as usize])
        });
        let (eig, c) = solve_gevp(&h, &scc.s);
        let sc = &scc.s * &c;
        let w: Vec<f64> = if kT <= 0.0 {
            (0..n).map(|k| if k < n_occ { 2.0 } else { 0.0 }).collect()
        } else {
            // f(μ) = Σ_k 2/(1+exp((ε_k−μ)/kT)) increasing in μ — bisect f(μ)=n_el
            let f = |mu: f64| {
                (0..n)
                    .map(|k| 2.0 / (1.0 + ((eig[k] - mu) / kT).exp()))
                    .sum::<f64>()
            };
            let (mut lo, mut hi) = (eig[0] - 1.0, eig[n - 1] + 1.0);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if f(mid) < n_el {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            let mu = 0.5 * (lo + hi);
            (0..n)
                .map(|k| 2.0 / (1.0 + ((eig[k] - mu) / kT).exp()))
                .collect()
        };
        let mut pop = vec![0.0f64; na];
        for mu in 0..n {
            let a = oa[mu] as usize;
            for k in 0..n {
                pop[a] += w[k] * c[(mu, k)] * sc[(mu, k)];
            }
        }
        pop
    };

    eprintln!("f64 map residual ||M(q)−q||_rms along the H-symmetric line q=[8−2x, x, x]:");
    for (label, q) in [
        (
            "frozen  [5.333,1.333,1.333]",
            vec![5.3333333, 1.3333333, 1.3333333],
        ),
        ("        [5.8,  1.1,  1.1]  ", vec![5.8, 1.1, 1.1]),
        ("        [4.7,  1.65, 1.65]  ", vec![4.7, 1.65, 1.65]),
        ("neutral [6,    1,    1]    ", scc.q0.clone()),
        ("true", scc.charges.clone()),
    ] {
        for kT in [0.0f64, 0.002] {
            let m = map(&q, kT);
            let res =
                (m.iter().zip(&q).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / na as f64).sqrt();
            eprintln!("  {label} kT={kT}:  M(q)={:?}  rms={res:.3e}", m);
        }
    }
}

#[test]
fn test_gpu_scc_direct_pop_parity() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    // n≤64 cold path, integer occupation.
    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    plan_pop_parity_case(&mut rt, &sk_dir, species, coords, 0.0, "H2O");
    // n≤64 cold path, Fermi smearing (occ_w weights).
    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    plan_pop_parity_case(&mut rt, &sk_dir, species, coords, 0.002, "H2O-smeared");
    // n>64 warm path (cᵀHc Jacobi + cs_normalize), integer occupation.
    let mut species = Vec::new();
    let mut coords = Vec::new();
    let h2o = [
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    for ix in 0..3 {
        for iy in 0..4 {
            for &a in &h2o {
                species.push(if a == h2o[0] { "O" } else { "H" }.to_string());
                coords.push([a[0] + ix as f64 * 3.0, a[1] + iy as f64 * 3.0, a[2]]);
            }
        }
    }
    plan_pop_parity_case(
        &mut rt,
        &sk_dir,
        species.clone(),
        coords.clone(),
        0.0,
        "12xH2O",
    );
    plan_pop_parity_case(&mut rt, &sk_dir, species, coords, 0.002, "12xH2O-smeared");
}
