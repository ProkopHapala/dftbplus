//! G3.1 / G3.2: canonical sparse DFTB energy (E_el + E_rep) and sparse SCC
//! versus dense f64 DFTB on SiH4.
//!
//! These tests exist to expose missing physics. They do not skip. A red
//! result is a diagnostic.
//!
//!   G3.1  frozen SiH4: 2 Tr(K H0) + E_rep  vs dense non-SCC Tr(D H0) + E_rep
//!   G3.2  sparse SCC loop (no dense eigenproblem) vs dense eig SCC + E_rep
//!         Both sides use valence q0 = [4,1,1,1,1]. matsci-0-3 SK q0 is Si=0,
//!         H≈0.49 (sum ≈ 2 e⁻) — that is not a SiH4 reference.
//!   G3.3  analytic F of that energy: D=2K, W=2KHK, CPU dH/dS/γ′/F_rep
//!         (does not touch qmqm/gpu_forces.cl — H-bond GPU path is read-only)
//!   G3.4  own E_tot central-difference vs own analytic F (Si x, h=1e-3 Å)

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::methods::dftb::forces::{compute_forces_from_dw, repulsive_energy, Forces};
use rust_dftb::methods::dftb::gamma::GammaTable;
use rust_dftb::methods::sparse::harness::{require_sih_sk_dir, require_sparse_gpu};
use rust_dftb::methods::sparse::scc::energy_non_scc;
use rust_dftb::methods::sparse::sparse_forces::{dw_from_k_padded, unpad_to_physical};
use rust_dftb::methods::sparse::SparseDftb;
use rust_dftb::qmqm::{Fragment, FragmentNeighborList, FragmentTemplate, MultiSystemSolver, SimpleMixer};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder, SkData};

fn sih4_geom() -> (Vec<String>, Vec<[f64; 3]>, Vec<u8>, f64, usize) {
    // Distorted SiH4: one H along +x, others from a 109.47° construction that
    // does **not** yield tetrahedral angles (second review §4: 109.47° ×3,
    // plus 100.67°, 65.96°, 141.06°). Valid molecule, not an equilibrium fixture.
    let species = vec!["Si".into(), "H".into(), "H".into(), "H".into(), "H".into()];
    let bond = 1.48f64;
    let theta = 109.47f64 * std::f64::consts::PI / 180.0;
    let (c, s) = (theta.cos(), theta.sin());
    let coords = vec![
        [0.0, 0.0, 0.0],
        [bond, 0.0, 0.0],
        [bond * c, bond * s, 0.0],
        [bond * c, bond * s * c, bond * s * s],
        [bond * c, -bond * s * c, -bond * s * s],
    ];
    let atom_n_orb = vec![4u8, 1, 1, 1, 1];
    (species, coords, atom_n_orb, 8.0, 4)
}

/// Physical valence electrons for SiH4 (matsci SK `q0` is not this).
fn sih4_valence_q0() -> Vec<f64> {
    vec![4.0, 1.0, 1.0, 1.0, 1.0]
}

struct DenseSccRef {
    e_h0: f64,
    e_scc: f64,
    e_el: f64,
    q: Vec<f64>,
    n_iter: usize,
    q0_sk: Vec<f64>,
    atom_species: Vec<u8>,
    density: DMatrix<f64>,
    edm: DMatrix<f64>,
}

/// Dense f64 SCC with an explicit q0 (linear mix). Same energy as `build_scc`:
/// Tr(D H0) + ½ Δq·V, D = 2 C_occ C_occᵀ.
fn dense_scc_valence(
    sk: &SkData,
    species: &[String],
    coords: &[[f64; 3]],
    q0: &[f64],
    mix: f64,
    tol: f64,
    max_iter: usize,
) -> DenseSccRef {
    let mut tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec())
        .unwrap_or_else(|e| panic!("G3.2 dense template: {e}"));
    let q0_sk = tmpl.q0.clone();
    let atom_species = tmpl.atom_species.clone();
    tmpl.q0 = q0.to_vec();
    let frag = Fragment::from_template(tmpl, coords.to_vec());
    let n_elec: f64 = q0.iter().sum();
    assert!(
        (frag.n_electrons - n_elec).abs() < 1e-12,
        "G3.2 dense n_electrons={} != sum(q0)={n_elec}",
        frag.n_electrons
    );
    let gamma = GammaTable::from_sk_data(sk, species).unwrap_or_else(|e| panic!("G3.2 dense gamma: {e}"));
    let n = coords.len() as f64;
    let centroid = coords.iter().fold([0.0; 3], |acc, c| [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]]);
    let frag_neighbors = FragmentNeighborList::build(&vec![[centroid[0] / n, centroid[1] / n, centroid[2] / n]], 10.0);
    let mixer = SimpleMixer::new(mix);
    let mut solver = MultiSystemSolver::new(vec![frag], frag_neighbors, gamma, mixer);
    solver.solve_scc(max_iter, tol).unwrap_or_else(|e| panic!("G3.2 dense valence SCC failed: {e}"));
    let frag = &solver.fragments[0];
    let n_occ = (frag.n_electrons / 2.0).round() as usize;
    let c_occ = frag.eigenvectors.columns(0, n_occ);
    let density = &c_occ * c_occ.transpose() * 2.0;
    let mut edm = DMatrix::<f64>::zeros(frag.template.n_orbs, frag.template.n_orbs);
    for k in 0..n_occ {
        let c = frag.eigenvectors.column(k);
        let eps = frag.eigenvalues[k];
        edm += (2.0 * eps) * (&c * c.transpose());
    }
    let e_h0 = (&density * &frag.template.h0).trace();
    let delta_q: Vec<f64> = frag.charges.iter().zip(frag.template.q0.iter()).map(|(q, q0)| q - q0).collect();
    let e_scc: f64 = 0.5 * delta_q.iter().zip(frag.shift.iter()).map(|(dq, s)| dq * s).sum::<f64>();
    DenseSccRef {
        e_h0, e_scc, e_el: e_h0 + e_scc,
        q: frag.charges.clone(),
        n_iter: solver.n_scc_iter,
        q0_sk,
        atom_species,
        density, edm,
    }
}

fn flatten(m: &DMatrix<f64>) -> Vec<f64> {
    let n = m.nrows();
    (0..n * n).map(|i| m[(i / n, i % n)]).collect()
}

/// Closed-shell E_h0 = Tr(D H0), D = 2 C_occ C_occ^T from H0 c = S c ε.
fn dense_e_h0_non_scc(h0: &DMatrix<f64>, s: &DMatrix<f64>, n_occ: usize) -> f64 {
    let n = h0.nrows();
    let se = SymmetricEigen::new(s.clone());
    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = 1.0 / se.eigenvalues[i].max(1e-12).sqrt();
    }
    let s_inv_sqrt = &se.eigenvectors * &d * se.eigenvectors.transpose();
    let h_orth = &s_inv_sqrt * h0 * &s_inv_sqrt;
    let he = SymmetricEigen::new(h_orth);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| he.eigenvalues[a].partial_cmp(&he.eigenvalues[b]).unwrap());
    let mut dens = DMatrix::<f64>::zeros(n, n);
    for &k in idx.iter().take(n_occ) {
        let c = he.eigenvectors.column(k);
        let co = &s_inv_sqrt * &c;
        dens += &co * co.transpose();
    }
    dens *= 2.0;
    (&dens * h0).trace()
}

#[test]
fn test_g3_1_sih4_energy_el_plus_rep() {
    let Some(gpu) = require_sparse_gpu() else { return };
    let sk_dir = require_sih_sk_dir();
    let (species, coords, atom_n_orb, _n_elec, n_occ) = sih4_geom();
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let builder = HamiltonianBuilder::new(sk);
    let ham = builder.build_non_scc(&species, &coords).unwrap();
    let e_h0_dense = dense_e_h0_non_scc(&ham.h0, &ham.s, n_occ);
    let e_rep = repulsive_energy(&sk_dir, &species, &coords).unwrap();
    eprintln!("=== G3.1 SiH4 frozen (Si–H = 1.48 Å; not tetrahedral — H–Si–H angles mixed) ===");
    eprintln!("  dense non-SCC: E_h0={e_h0_dense:.10}  E_rep={e_rep:.10}  E_tot={:.10}", e_h0_dense + e_rep);
    assert!(e_rep.abs() > 1e-4, "G3.1: E_rep={e_rep:.3e} is ~0 — Spline missing or not evaluated");
    assert!(e_h0_dense.is_finite() && e_rep.is_finite(), "G3.1 dense energy non-finite");

    let h0 = flatten(&ham.h0);
    let s = flatten(&ham.s);
    let sparse = energy_non_scc(&gpu, &h0, &s, &atom_n_orb, n_occ as f32, &sk_dir, &species, &coords)
        .unwrap_or_else(|e| panic!("G3.1 sparse energy failed: {e}"));
    eprintln!(
        "  sparse non-SCC: E_h0={:.10}  E_rep={:.10}  E_tot={:.10}  Tr(KS)={:.6}  R_I={:.3e}",
        sparse.e_h0, sparse.e_rep, sparse.e_tot, sparse.tr_ks, sparse.r_i
    );
    let d_el = (sparse.e_h0 - e_h0_dense).abs();
    let d_rep = (sparse.e_rep - e_rep).abs();
    eprintln!("  |dE_h0|={d_el:.3e}  |dE_rep|={d_rep:.3e}");
    assert!((sparse.tr_ks - n_occ as f32).abs() < 1e-4, "G3.1 Tr(KS)={} != Nocc={n_occ}", sparse.tr_ks);
    assert!(d_rep < 1e-12, "G3.1 E_rep must be the same function on both sides: |d|={d_rep:.3e}");
    // f32 purify vs f64 eig. Gate D |dE| on Tr(K H0) was 8.6e-8 → ×2 ≈ 2e-7.
    assert!(d_el < 1e-5, "G3.1 |E_h0_sparse - E_h0_dense|={d_el:.3e} (target 1e-5)");
    eprintln!("  G3.1: E_tot = E_el + E_rep is a DFTB energy on this frozen geometry.");
}

#[test]
fn test_g3_2_sih4_sparse_scc() {
    let Some(_gpu) = require_sparse_gpu() else { return };
    let sk_dir = require_sih_sk_dir();
    let (species, coords, _atom_n_orb, n_elec, n_occ) = sih4_geom();
    let q0 = sih4_valence_q0();
    assert!((q0.iter().sum::<f64>() - n_elec).abs() < 1e-12);
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let e_rep = repulsive_energy(&sk_dir, &species, &coords).unwrap();
    // Previous G3.2 compared against HamiltonianBuilder::build_scc, which uses
    // SK-parsed q0 (Si=0, H≈0.49, n_occ=1). That is not SiH4. Kept as a print:
    //   dense SK-q0: E_el≈-0.899  q0=[0, 0.49, …]  sum(q)≈2
    let dense = dense_scc_valence(&sk, &species, &coords, &q0, 0.5, 1e-5, 80);
    let e_tot_dense = dense.e_el + e_rep;
    eprintln!("=== G3.2 SiH4 sparse SCC vs dense SCC (valence q0) ===");
    eprintln!("  SK-parsed q0={:?}  sum={:.4}  (ignored; matsci Si q0 is 0)", dense.q0_sk, dense.q0_sk.iter().sum::<f64>());
    eprintln!(
        "  dense: E_h0={:.10}  E_scc={:.10}  E_el={:.10}  E_rep={e_rep:.10}  E_tot={e_tot_dense:.10}  n_iter={}  q0={:?}  q={:?}",
        dense.e_h0, dense.e_scc, dense.e_el, dense.n_iter, q0, dense.q
    );
    assert!(e_rep.abs() > 1e-4, "G3.2: E_rep={e_rep:.3e} ~0");
    let qsum_dense: f64 = dense.q.iter().sum();
    assert!((qsum_dense - n_elec).abs() < 1e-4, "G3.2 dense sum(q)={qsum_dense} != N_elec={n_elec}");

    let mut eng = SparseDftb::new(sk.clone(), &sk_dir, species.clone(), coords.clone())
        .unwrap_or_else(|e| panic!("G3.2 SparseDftb::new: {e}"));
    let scc = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("G3.2 sparse SCC failed: {e}"));
    let sparse = eng.last_energy();
    eprintln!(
        "  sparse: E_el={:.10}  E_h0={:.10}  E_scc={:.10}  E_rep={:.10}  E_tot={:.10}  n_scc={}  rms={:.3e}  q={:?}",
        sparse.e_el, sparse.e_h0, sparse.e_scc, sparse.e_rep, sparse.e_tot, scc.n_iters, scc.rms, sparse.q
    );
    let d_el = (sparse.e_el - dense.e_el).abs();
    let d_h0 = (sparse.e_h0 - dense.e_h0).abs();
    let d_scc = (sparse.e_scc - dense.e_scc).abs();
    let d_tot = (sparse.e_tot - e_tot_dense).abs();
    let mut max_dq = 0.0f64;
    for a in 0..species.len() {
        max_dq = max_dq.max((sparse.q[a] - dense.q[a]).abs());
    }
    eprintln!("  |dE_h0|={d_h0:.3e}  |dE_scc|={d_scc:.3e}  |dE_el|={d_el:.3e}  |dE_tot|={d_tot:.3e}  max|dq|={max_dq:.3e}  N_elec={n_elec}  r_scc={:.3e}  R_H={:.3e}", sparse.r_scc, sparse.r_h);
    assert!((sparse.tr_ks - n_occ as f32).abs() < 1e-4, "G3.2 Tr(KS)={}", sparse.tr_ks);
    assert!(d_el < 1e-4, "G3.2 |E_el_sparse - E_el_dense|={d_el:.3e}");
    assert!(d_tot < 1e-4, "G3.2 |E_tot_sparse - E_tot_dense|={d_tot:.3e}");
    assert!(max_dq < 1e-3, "G3.2 max|q_sparse - q_dense|={max_dq:.3e}");
}

fn print_forces(label: &str, f: &Forces) {
    eprintln!("  {label} components (Ha/Å):");
    for i in 0..f.forces.len() {
        eprintln!(
            "    atom {i}: F=[{:.6e}, {:.6e}, {:.6e}]  nonSCC=[{:.6e}, {:.6e}, {:.6e}]  shift=[{:.6e}, {:.6e}, {:.6e}]  dc=[{:.6e}, {:.6e}, {:.6e}]  rep=[{:.6e}, {:.6e}, {:.6e}]",
            f.forces[i][0], f.forces[i][1], f.forces[i][2],
            f.non_scc[i][0], f.non_scc[i][1], f.non_scc[i][2],
            f.scc_shift[i][0], f.scc_shift[i][1], f.scc_shift[i][2],
            f.scc_dc[i][0], f.scc_dc[i][1], f.scc_dc[i][2],
            f.repulsive[i][0], f.repulsive[i][1], f.repulsive[i][2],
        );
    }
}

fn max_abs_mat(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    let mut m = 0.0f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            m = m.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    m
}

fn sparse_eng(sk: &SkData, sk_dir: &str, species: &[String], coords: Vec<[f64; 3]>) -> SparseDftb {
    SparseDftb::new(sk.clone(), sk_dir, species.to_vec(), coords)
        .unwrap_or_else(|e| panic!("SparseDftb::new: {e}"))
}

#[test]
fn test_g3_3_analytic_force_and_g3_4_energy_gradient() {
    let Some(_gpu) = require_sparse_gpu() else { return };
    let sk_dir = require_sih_sk_dir();
    let (species, coords, atom_n_orb, n_elec, n_occ) = sih4_geom();
    let q0 = sih4_valence_q0();
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let dense = dense_scc_valence(&sk, &species, &coords, &q0, 0.5, 1e-5, 80);
    let mut eng = sparse_eng(&sk, &sk_dir, &species, coords.clone());
    eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("G3.3 sparse SCC: {e}"));
    let sparse = eng.last_energy().clone();
    eprintln!("=== G3.3 SiH4 analytic force (D=2K, W=2KHK)  [SparseDftb] ===");
    eprintln!("  sparse E_tot={:.10}  dense E_el={:.10}  N_elec={n_elec}", sparse.e_tot, dense.e_el);

    let (d_pad, w_pad) = dw_from_k_padded(eng.k_pad(), eng.h_scc_pad(), species.len());
    let d_sp = unpad_to_physical(&d_pad, &atom_n_orb);
    let w_sp = unpad_to_physical(&w_pad, &atom_n_orb);
    let d_err = max_abs_mat(&d_sp, &dense.density);
    let w_err = max_abs_mat(&w_sp, &dense.edm);
    eprintln!("  max|D_sparse-D_dense|={d_err:.3e}  max|W_sparse-W_dense|={w_err:.3e}");

    let f_sp = eng.forces().unwrap_or_else(|e| panic!("G3.3 sparse analytic force failed: {e}"));
    let f_dn = compute_forces_from_dw(&sk, &species, &coords, &dense.density, &dense.edm, &dense.q, &q0, &sk_dir)
        .unwrap_or_else(|e| panic!("G3.3 dense analytic force failed: {e}"));
    print_forces("sparse", &f_sp);
    print_forces("dense ", &f_dn);

    let mut max_df = 0.0f64;
    let mut max_f = 0.0f64;
    let mut worst = (0usize, 0usize);
    for i in 0..species.len() {
        for c in 0..3 {
            let df = (f_sp.forces[i][c] - f_dn.forces[i][c]).abs();
            max_f = max_f.max(f_sp.forces[i][c].abs()).max(f_dn.forces[i][c].abs());
            if df > max_df {
                max_df = df;
                worst = (i, c);
            }
        }
    }
    let rel_f = max_df / max_f.max(1e-8);
    eprintln!("  max|dF|={max_df:.3e}  max|F|={max_f:.3e}  rel={rel_f:.3e}  worst=atom {} dir {}", worst.0, worst.1);
    assert!(max_df < 1e-4 || rel_f < 1e-3, "G3.3 |F_sparse-F_dense| max={max_df:.3e} rel={rel_f:.3e} (target 1e-4 abs or 1e-3 rel)");

    let h = 1e-3f64;
    let mut xyz_p = coords.clone();
    let mut xyz_m = coords.clone();
    xyz_p[0][0] += h;
    xyz_m[0][0] -= h;
    eprintln!("=== G3.4 energy-gradient (Si x, h={h} Å)  [SparseDftb, q reset to q0 each side] ===");
    eng.set_q(&q0).unwrap();
    eng.set_coords(&xyz_p).unwrap_or_else(|e| panic!("G3.4 +h set_coords: {e}"));
    eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("G3.4 +h SCC: {e}"));
    let e_p = eng.energy().unwrap();
    eng.set_q(&q0).unwrap();
    eng.set_coords(&xyz_m).unwrap_or_else(|e| panic!("G3.4 -h set_coords: {e}"));
    eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("G3.4 -h SCC: {e}"));
    let e_m = eng.energy().unwrap();
    let f_fd = -(e_p - e_m) / (2.0 * h);
    let f_ana = f_sp.forces[0][0];
    let abs_eg = (f_ana - f_fd).abs();
    let rel_eg = abs_eg / f_ana.abs().max(f_fd.abs()).max(1e-4);
    eprintln!(
        "  E(+h)={e_p:.10}  E(-h)={e_m:.10}  F_fd={f_fd:.6e}  F_ana={f_ana:.6e}  |d|={abs_eg:.3e}  rel={rel_eg:.3e}  n_occ={n_occ}"
    );
    assert!(
        abs_eg < 1e-4 || rel_eg < 1e-3,
        "G3.4 |F_ana - F_fd|={abs_eg:.3e} rel={rel_eg:.3e} (target 1e-4 abs or 1e-3 rel). F_ana={f_ana:.6e} F_fd={f_fd:.6e}"
    );
}
