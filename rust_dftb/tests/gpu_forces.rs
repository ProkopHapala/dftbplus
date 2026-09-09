//! GPU analytic force parity test (Phase 4d).
//!
//! Compares GPU `force_pairs` kernel output against CPU
//! `non_scc_electronic_force` for synthetic H2 and sp3 systems, then
//! against real SK data (H2O, formic dimer) if RUST_DFTB_SK_DIR is set.
//!
//! The CPU reference is the validated analytic force (rel_err ~1e-10 vs
//! finite differences of total energy). The GPU should match within f32
//! tolerance (~1e-3 relative).

use rust_dftb::qmqm::gpu_forces::GpuForceDriver;
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use rust_dftb::qmqm::gpu_prep::GpuBatch;
use rust_dftb::qmqm::{Fragment, FragmentTemplate, GammaTable};
use rust_dftb::{
    HamiltonianBuilder, NeighborBuilder, SkData, SkTableSp, SpeciesOrbitals,
    AtomicParamsSp, SystemContext,
};
use rust_dftb::methods::dftb::forces::non_scc_electronic_force;
use rust_dftb::methods::dftb::interpolation::EqGridTable;
use nalgebra::{Cholesky, DMatrix, DVector, SymmetricEigen};

use std::collections::HashMap;

fn try_gpu() -> Option<(GpuRuntime, GpuForceDriver)> {
    match GpuRuntime::new() {
        Ok(mut rt) => match GpuForceDriver::new(&mut rt) {
            Ok(d) => Some((rt, d)),
            Err(e) => { eprintln!("Skipping: force driver failed ({e})"); None }
        },
        Err(e) => { eprintln!("Skipping: no OpenCL ({e})"); None }
    }
}

/// Diagonalize H0/S and build DM/EDM (replicates forces.rs private helpers).
fn diagonalize_and_build_dm_edm(
    ham: &rust_dftb::Hamiltonian,
    n_electrons: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = ham.h0.nrows();
    let cholesky = Cholesky::new(ham.s.clone()).unwrap();
    let l = cholesky.l();
    let m = l.solve_lower_triangular(&ham.h0).unwrap();
    let n_mat = l.solve_lower_triangular(&m.transpose()).unwrap();
    let h_prime = n_mat.transpose();
    let se = SymmetricEigen::new(h_prime);
    let eigs = se.eigenvalues;
    let c_prime = se.eigenvectors;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eigs[a].partial_cmp(&eigs[b]).unwrap());
    let sorted_eigs: Vec<f64> = idx.iter().map(|&i| eigs[i]).collect();
    let sorted_c_prime = c_prime.select_columns(&idx);
    let c = l.tr_solve_lower_triangular(&sorted_c_prime).unwrap();
    let n_occ = (n_electrons / 2.0).round() as usize;
    let c_occ = c.columns(0, n_occ).into_owned();
    let eps_occ: Vec<f64> = sorted_eigs.iter().take(n_occ).copied().collect();
    let dm = &c_occ * c_occ.transpose() * 2.0;
    let mut edm = DMatrix::<f64>::zeros(n, n);
    for k in 0..n_occ {
        let col = c_occ.column(k);
        let scaled = col * (2.0 * eps_occ[k]);
        edm += &scaled * col.transpose();
    }
    (dm, edm)
}

/// Flatten DM/EDM to GPU layout: [replica][i][j] row-major, f32.
fn flatten_dm_edm(dm: &DMatrix<f64>, edm: &DMatrix<f64>) -> (Vec<f32>, Vec<f32>) {
    let n = dm.nrows();
    let dm_flat: Vec<f32> = (0..n*n).map(|idx| {
        let i = idx / n; let j = idx - i*n;
        dm[(i, j)] as f32
    }).collect();
    let edm_flat: Vec<f32> = (0..n*n).map(|idx| {
        let i = idx / n; let j = idx - i*n;
        edm[(i, j)] as f32
    }).collect();
    (dm_flat, edm_flat)
}

/// Build a synthetic SkData for H (1s orbital) with smooth exponential SK tables.
fn make_h2_sk_data() -> SkData {
    let dr = 0.1;
    let n_grid = 100;
    let r_max = dr * n_grid as f64;
    let hh_values: Vec<Vec<f64>> = (0..n_grid)
        .map(|i| {
            let r = i as f64 * dr;
            let tail = if r > r_max - 1.0 { let t = (r_max - r) / 1.0; t * t } else { 1.0 };
            let v = -0.3 * (-1.0 * r).exp() * tail;
            let mut row = vec![0.0f64; 20];
            row[19] = v;
            row
        })
        .collect();
    let hh_h = EqGridTable::new(dr, hh_values.clone());
    let hh_s = EqGridTable::new(dr, hh_values);
    let hh_table = SkTableSp { sp1: "H".to_string(), sp2: "H".to_string(), h: hh_h, s: hh_s };
    let mut pairs = HashMap::new();
    pairs.insert(("H".to_string(), "H".to_string()), hh_table);
    let mut onsite = HashMap::new();
    onsite.insert("H".to_string(), AtomicParamsSp { e_s: -0.4, e_p: 0.0, q0: 1.0, u_hubbard: 0.5 });
    let mut orbital_info = HashMap::new();
    orbital_info.insert("H".to_string(), SpeciesOrbitals::from_ang_momenta(&[0]));
    SkData { onsite, pairs, orbital_info }
}

/// Synthetic 4-orbital (sp) species for a multi-atom test.
fn make_c_like_sk_data() -> SkData {
    let dr = 0.1;
    let n_grid = 100;
    let r_max = dr * n_grid as f64;
    let make_values = |decay: f64, amp: f64| -> Vec<Vec<f64>> {
        (0..n_grid)
            .map(|i| {
                let r = i as f64 * dr;
                let tail = if r > r_max - 1.0 { let t = (r_max - r) / 1.0; t * t } else { 1.0 };
                let base = amp * (-decay * r).exp() * tail;
                let mut row = vec![0.0f64; 20];
                row[19] = base; row[18] = 0.8 * base; row[14] = 0.6 * base; row[15] = 0.3 * base;
                row
            })
            .collect()
    };
    let x_h = EqGridTable::new(dr, make_values(1.0, -0.3));
    let x_s = EqGridTable::new(dr, make_values(1.0, 0.2));
    let xx_table = SkTableSp { sp1: "X".to_string(), sp2: "X".to_string(), h: x_h, s: x_s };
    let mut pairs = HashMap::new();
    pairs.insert(("X".to_string(), "X".to_string()), xx_table);
    let mut onsite = HashMap::new();
    onsite.insert("X".to_string(), AtomicParamsSp { e_s: -0.5, e_p: -0.1, q0: 4.0, u_hubbard: 0.5 });
    let mut orbital_info = HashMap::new();
    orbital_info.insert("X".to_string(), SpeciesOrbitals::from_ang_momenta(&[0, 1]));
    SkData { onsite, pairs, orbital_info }
}

fn make_fragment(sk: &SkData, species: &[String], coords: &[[f64; 3]]) -> Fragment {
    let tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
    Fragment::from_template(tmpl, coords.to_vec())
}

/// Run CPU non-SCC electronic force and GPU force, compare.
fn run_parity(
    rt: &GpuRuntime,
    driver: &GpuForceDriver,
    sk: &SkData,
    species: &[String],
    coords: &[[f64; 3]],
    n_electrons: f64,
    label: &str,
) {
    let n_atoms = species.len();
    let builder = HamiltonianBuilder::new(sk.clone());
    let ham = builder.build_non_scc(species, coords).unwrap();
    let n = ham.h0.nrows();
    let ctx = SystemContext::from_sk_data(&builder.sk, species).unwrap();
    let cutoff = builder.sk.pairs.values().map(|t| t.cutoff()).fold(0.0_f64, f64::max);
    let neigh = NeighborBuilder { cutoff }.build(coords).unwrap();

    // CPU reference
    let (dm, edm) = diagonalize_and_build_dm_edm(&ham, n_electrons);
    let mut cpu_forces = vec![[0.0f64; 3]; n_atoms];
    non_scc_electronic_force(&ctx, &neigh, coords, &dm, &edm, &mut cpu_forces).unwrap();

    // GPU
    let gamma = GammaTable::from_sk_data(sk, species).unwrap();
    let frag = make_fragment(sk, species, coords);
    let batch = GpuBatch::from_fragments(&[frag], sk, &gamma).unwrap();
    let (dm_flat, edm_flat) = flatten_dm_edm(&dm, &edm);
    eprintln!("{label}: total_atoms={} total_h={} n_frags={} pair_buckets={}",
        batch.total_atoms, batch.total_h_elements, batch.n_frags, batch.pair_buckets.len());
    for (bi, bucket) in batch.pair_buckets.iter().enumerate() {
        let skt = &batch.sk_tables[bucket.sk_table_idx];
        eprintln!("  bucket {bi}: block_type={} n_pairs={} n_grid={} n_sk_cols={} dr={}",
            bucket.block_type, bucket.n_pairs, skt.n_grid, skt.n_sk_cols, skt.dr);
        for p in &bucket.pairs {
            eprintln!("    pair: replica={} atom_i={} atom_j={} orb_i={} orb_j={} r={:.6} l={:.6} m={:.6} n={:.6}",
                p.replica, p.atom_i, p.atom_j, p.orb_i, p.orb_j, p.r, p.l, p.m, p.n);
        }
    }
    for f in &batch.fragments {
        eprintln!("  frag: n_atoms={} n_orbs={} atom_off={}", f.n_atoms, f.n_orbs, f.atom_off);
    }
    let gpu_forces = driver.gpu_force_batched(rt, &batch, &dm_flat, &edm_flat).unwrap();

    // Compare
    let mut max_err = 0.0f64;
    let mut max_force = 0.0f64;
    for i in 0..n_atoms {
        for d in 0..3 {
            let cpu = cpu_forces[i][d];
            let gpu = gpu_forces[3*i + d] as f64;
            let err = (cpu - gpu).abs();
            max_err = max_err.max(err);
            max_force = max_force.max(cpu.abs());
            eprintln!("{label} atom {i} dir {d}: cpu={cpu:.6e} gpu={gpu:.6e} err={err:.3e}");
        }
    }
    let rel_err = if max_force > 1e-10 { max_err / max_force } else { max_err };
    eprintln!("{label}: N_orbs={n} max|F|={max_force:.3e} max|err|={max_err:.3e} rel_err={rel_err:.3e}");

    // f32 tolerance: GPU is single-precision, B-spline interpolation + atomic
    // accumulation. Expect ~1e-3 relative for well-conditioned systems.
    assert!(rel_err < 5e-3,
        "{label}: GPU-CPU force parity failed: rel_err={rel_err:.3e} (max|F|={max_force:.3e}, max|err|={max_err:.3e})");

    // Newton's third law: sum of forces ≈ 0
    let mut sum = [0.0f64; 3];
    for i in 0..n_atoms {
        for d in 0..3 { sum[d] += gpu_forces[3*i + d] as f64; }
    }
    let sum_mag = sum[0].abs().max(sum[1].abs()).max(sum[2].abs());
    eprintln!("{label}: GPU force sum = {sum:?}, |sum|={sum_mag:.3e}");
    assert!(sum_mag < 1e-6, "{label}: Newton 3rd law violated: |sum|={sum_mag:.3e}");
}

#[test]
fn test_gpu_force_parity_h2() {
    let Some((rt, driver)) = try_gpu() else { return; };
    let sk = make_h2_sk_data();
    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    run_parity(&rt, &driver,&sk, &species, &coords, 2.0, "H2");
}

#[test]
fn test_gpu_force_parity_h2_tilted() {
    let Some((rt, driver)) = try_gpu() else { return; };
    let sk = make_h2_sk_data();
    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [0.5, 0.6, 0.7]];
    run_parity(&rt, &driver,&sk, &species, &coords, 2.0, "H2_tilted");
}

#[test]
fn test_gpu_force_parity_sp3() {
    let Some((rt, driver)) = try_gpu() else { return; };
    let sk = make_c_like_sk_data();
    let species = vec!["X".to_string(), "X".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.3, 0.4, 0.2]];
    run_parity(&rt, &driver,&sk, &species, &coords, 8.0, "sp3");
}

#[test]
fn test_gpu_force_parity_h2o() {
    let Some((rt, driver)) = try_gpu() else { return; };
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
    let sk = rust_dftb::load_sk_for_species(&sk_dir, &species).unwrap();
    let n_electrons: f64 = 6.0 + 1.0 + 1.0; // O:6, H:1, H:1
    run_parity(&rt, &driver,&sk, &species, &coords, n_electrons, "H2O");
}

#[test]
fn test_gpu_force_parity_formic_dimer() {
    let Some((rt, driver)) = try_gpu() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let xyz_path = "/home/prokop/git/dftbplus/data/xyz/formic_dimer.xyz";
    let Ok(xyz) = rust_dftb::parse_xyz(xyz_path) else {
        eprintln!("Skipping: formic_dimer.xyz not found");
        return;
    };
    let species: Vec<String> = xyz.species.iter().map(|s| s.clone()).collect();
    let coords: Vec<[f64; 3]> = xyz.coords.iter().map(|c| [c[0], c[1], c[2]]).collect();
    let sk = rust_dftb::load_sk_for_species(&sk_dir, &species).unwrap();
    // Formic dimer HCOOH·HCOOH: C2O2H4 → 2*(4+6+4+1) = ... let's count valence e-
    // C:4 each, O:6 each, H:1 each. 2C + 4O + 4H = 2*4 + 4*6 + 4*1 = 8+24+4 = 36
    let n_electrons = 36.0;
    run_parity(&rt, &driver,&sk, &species, &coords, n_electrons, "formic_dimer");
}
