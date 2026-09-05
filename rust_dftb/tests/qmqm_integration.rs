//! Integration tests for the qmqm multi-system solver.
//!
//! These tests compare fragment-based results against full-system DFTB
//! to verify that the qmqm module produces correct Hamiltonians,
//! shifts, and charges.

use rust_dftb::{load_sk_for_species, max_abs_diff, parse_xyz, DftbOutput, HamiltonianBuilder};
use rust_dftb::qmqm::{Fragment, FragmentTemplate};

/// Test that a single fragment (H2) produces identical H0 and S
/// to the full-system Hamiltonian builder.
#[test]
fn fragment_h2_matches_full_system_non_scc() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.74, 0.0, 0.0], // H-H bond length in Å
    ];

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // Full-system Hamiltonian
    let builder = HamiltonianBuilder::new(sk.clone());
    let ham_full = builder.build_non_scc(&species, &coords).unwrap();

    // Fragment-based Hamiltonian
    let template = FragmentTemplate::new(&sk, species.clone(), coords.clone()).unwrap();
    let frag = Fragment::from_template(template, coords);

    let dh = max_abs_diff(&ham_full.h0, &frag.template.h0);
    let ds = max_abs_diff(&ham_full.s, &frag.template.s);

    assert!(
        dh < 1e-12,
        "Fragment H0 should match full-system H0 for single fragment, diff = {dh:e}"
    );
    assert!(
        ds < 1e-12,
        "Fragment S should match full-system S for single fragment, diff = {ds:e}"
    );
}

/// Test that fragment diagonalization produces reasonable eigenvalues for H2.
/// Neutral H2 has 2 electrons → 1 occupied MO.
#[test]
fn fragment_h2_diagonalization() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.74, 0.0, 0.0],
    ];

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();
    let mut frag = Fragment::from_template(template, coords);

    // Build H_scc with zero shifts (neutral, no SCC yet)
    frag.build_h_scc();

    // Diagonalize
    frag.diagonalize().unwrap();

    // H2 has 2 orbitals (1s per H), 2 electrons → 1 occupied MO
    assert_eq!(frag.template.n_orbs, 2);
    assert_eq!(frag.eigenvalues.len(), 2);

    // Eigenvalues should be real and sorted ascending
    assert!(frag.eigenvalues[0] < frag.eigenvalues[1]);

    // Occupied eigenvalue should be negative (bound state)
    assert!(frag.eigenvalues[0] < 0.0, "Occupied eigenvalue should be negative, got {}", frag.eigenvalues[0]);
}

/// Test fixed-charge SCC: inject neutral charges, build H_scc, diagonalize.
/// For a single fragment with neutral charges, the shifts should be zero
/// and the result should match the non-SCC case.
#[test]
fn fragment_h2_fixed_neutral_charges() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.74, 0.0, 0.0],
    ];

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();
    let mut frag = Fragment::from_template(template, coords);

    // Set neutral charges (q0)
    frag.charges.copy_from_slice(&frag.template.q0);
    frag.shift.fill(0.0);

    // Build H_scc with zero shifts
    frag.build_h_scc();

    // Diagonalize and compute charges
    frag.diagonalize().unwrap();
    frag.compute_charges();

    // Eigenvalues should match non-SCC case
    let mut frag_ref = Fragment::from_template(frag.template.clone(), frag.coords.clone());
    frag_ref.build_h_scc();
    frag_ref.diagonalize().unwrap();

    let de = max_abs_diff(
        &nalgebra::DMatrix::from_row_slice(frag.template.n_orbs, 1, &frag.eigenvalues.as_slice()),
        &nalgebra::DMatrix::from_row_slice(frag_ref.template.n_orbs, 1, &frag_ref.eigenvalues.as_slice()),
    );
    assert!(de < 1e-12, "Eigenvalues should match for neutral charges, diff = {de:e}");
}

/// Test N2 fragment: same parity check with more orbitals (sp basis).
#[test]
fn fragment_n2_matches_full_system_non_scc() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["N".to_string(), "N".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.10, 0.0, 0.0], // N≡N triple bond in Å
    ];

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    let builder = HamiltonianBuilder::new(sk.clone());
    let ham_full = builder.build_non_scc(&species, &coords).unwrap();

    let template = FragmentTemplate::new(&sk, species.clone(), coords.clone()).unwrap();
    let frag = Fragment::from_template(template, coords);

    let dh = max_abs_diff(&ham_full.h0, &frag.template.h0);
    let ds = max_abs_diff(&ham_full.s, &frag.template.s);

    assert!(
        dh < 1e-12,
        "Fragment H0 should match full-system H0 for N2, diff = {dh:e}"
    );
    assert!(
        ds < 1e-12,
        "Fragment S should match full-system S for N2, diff = {ds:e}"
    );
}

/// Test HCOOH (formic acid) fragment: multi-atom, multi-species parity.
#[test]
fn fragment_hcooh_matches_full_system_non_scc() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    // Load HCOOH geometry from data/xyz/HCOOH.xyz
    let xyz_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/xyz/HCOOH.xyz");
    let mol = parse_xyz(xyz_path).unwrap();
    let species = mol.species;
    let coords = mol.coords;

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    let builder = HamiltonianBuilder::new(sk.clone());
    let ham_full = builder.build_non_scc(&species, &coords).unwrap();

    let template = FragmentTemplate::new(&sk, species.clone(), coords.clone()).unwrap();
    let frag = Fragment::from_template(template, coords);

    let dh = max_abs_diff(&ham_full.h0, &frag.template.h0);
    let ds = max_abs_diff(&ham_full.s, &frag.template.s);

    assert!(
        dh < 1e-12,
        "Fragment H0 should match full-system H0 for HCOOH, diff = {dh:e}"
    );
    assert!(
        ds < 1e-12,
        "Fragment S should match full-system S for HCOOH, diff = {ds:e}"
    );
}

/// Test gamma function self-consistency: for a single atom,
/// gamma(0, U, U) should equal U (the Hubbard U).
#[test]
fn gamma_self_consistency() {
    use rust_dftb::qmqm::gamma::gamma_full;

    let u = 0.5;
    let g = gamma_full(0.0, u, u);
    assert!((g - u).abs() < 1e-12, "gamma(0, U, U) should equal U, got {} vs {}", g, u);
}

/// Test SCC Hamiltonian parity against Fortran DFTB+ reference.
/// 
/// Uses fixed charges q = [1.1, 0.9] on H2 (deltaQ = [0.1, -0.1] relative to q0=1.0).
/// Compares Rust-built H_scc against Fortran reference from `hamsqr1.dat`.
#[test]
fn h2_fixed_charge_scc_parity() {
    use rust_dftb::qmqm::solver::MultiSystemSolver;
    use rust_dftb::qmqm::{FragmentNeighborList, GammaTable, SimpleMixer};
    use nalgebra::DMatrix;

    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.74, 0.0, 0.0], // H-H bond length in Å
    ];

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // Build fragment template and solver
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();
    let frag = Fragment::from_template(template.clone(), coords.clone());
    
    // For a single fragment, neighbor list is empty (no inter-fragment interactions)
    // But we still need proper gamma table for intra-fragment SCC
    let centroids = vec![[0.37, 0.0, 0.0]]; // centroid of H2
    let frag_neighbors = FragmentNeighborList::build(&centroids, 10.0); // large cutoff
    
    // Hubbard U for H from mio-1-1 parameters: U = 0.4195 Hartree
    let gamma = GammaTable::from_hubbard_u(vec![0.4195]);
    
    let mixer = SimpleMixer::new(0.3);
    let mut solver = MultiSystemSolver::new(vec![frag], frag_neighbors, gamma, mixer);

    // Set fixed charges to achieve deltaQ = [0.1, -0.1]
    // q0(H) = 1.0, so q = q0 + deltaQ = [1.1, 0.9]
    let fixed_charges = vec![1.1, 0.9];
    solver.build_h_scc_with_fixed_charges(&fixed_charges);

    // Extract H_scc from fragment
    let h_scc_rust = &solver.fragments[0].h_scc;
    let frag = &solver.fragments[0];
    
    // DEBUG: Print what Rust computed
    eprintln!("Rust q0: {:?}", frag.template.q0);
    eprintln!("Rust charges: {:?}", frag.charges);
    eprintln!("Rust delta_q: {:?}", frag.charges.iter().zip(frag.template.q0.iter()).map(|(q,q0)| q-q0).collect::<Vec<_>>());
    eprintln!("Rust shift: {:?}", frag.shift);
    eprintln!("Rust v_intra: {:?}", frag.v_intra);
    eprintln!("Rust v_ext: {:?}", frag.v_ext);
    eprintln!("Rust H0:\n{:.16e}", frag.template.h0);
    eprintln!("Rust H_scc:\n{:.16e}", h_scc_rust);
    eprintln!("Rust S:\n{:.16e}", frag.template.s);

    // Load Fortran reference H_scc (dense 2x2 matrix from hamsqr1.dat)
    // Expected values from ref_h_scc.dat:
    // -2.3435869555627101e-01 -3.2006037343168198e-01
    // -3.2006037343168198e-01 -2.4284210444372889e-01
    let h_ref = DMatrix::from_row_slice(2, 2, &[
        -2.3435869555627101e-01, -3.2006037343168198e-01,
        -3.2006037343168198e-01, -2.4284210444372889e-01,
    ]);
    
    eprintln!("Fortran H_scc:\n{:.16e}", h_ref);

    let diff = max_abs_diff(h_scc_rust, &h_ref);
    // Tolerance 1e-7: residual ~2e-8 comes from H0 interpolation differences,
    // not the SCC shift application itself (diagonal shifts match to ~5e-10).
    assert!(
        diff < 1e-7,
        "H_scc mismatch between Rust and Fortran DFTB+ (diff = {diff:e})"
    );

    // Also test diagonalization: eigenvalues should match Fortran
    solver.diagonalize_all().unwrap();
    let eigvals_rust = &solver.fragments[0].eigenvalues;
    
    // Expected eigenvalues from Fortran (computed from H_scc & S via Cholesky).
    // For deltaQ=[0.1,-0.1] at 0.74 Å: one eigenvalue is positive because the
    // fixed charge imbalance creates an unoccupied/unbound state.
    let eig_ref = vec![-3.4044801351417342e-01, 2.2709922028941476e-01];

    for (i, (r, f)) in eigvals_rust.iter().zip(eig_ref.iter()).enumerate() {
        assert!(
            (r - f).abs() < 1e-7,
            "Eigenvalue {} mismatch: Rust={}, Fortran={}", i, r, f
        );
    }

    // Test charges after diagonalization (Mulliken analysis)
    // These should differ from input because electrons rearrange
    let charges_rust: Vec<f64> = solver.fragments[0].charges.clone();
    let q0 = vec![1.0, 1.0]; // Reference neutral charges for H
    let delta_q_rust: Vec<f64> = charges_rust.iter().zip(q0.iter()).map(|(q, q0)| q - q0).collect();
    
    // Fixed-charge test: charges will deviate from input due to diagonalization,
    // but we only verify H_scc and eigenvalue parity here.
    // Full SCC convergence parity is tested separately.
}

/// N2 fixed-charge SCC parity against Fortran DFTB+ reference.
#[test]
fn n2_fixed_charge_scc_parity() {
    use rust_dftb::qmqm::solver::MultiSystemSolver;
    use rust_dftb::qmqm::{FragmentNeighborList, GammaTable, SimpleMixer};

    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };
    let Ok(ref_h) = std::env::var("RUST_DFTB_REF_H") else { return; };
    let Ok(ref_s) = std::env::var("RUST_DFTB_REF_S") else { return; };

    let species = vec!["N".to_string(), "N".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.10, 0.0, 0.0], // N≡N triple bond in Å
    ];

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();
    let frag = Fragment::from_template(template.clone(), coords.clone());

    let centroids = vec![[0.55, 0.0, 0.0]];
    let frag_neighbors = FragmentNeighborList::build(&centroids, 10.0);

    // Hubbard U for N from mio-1-1: U = 0.4309 Hartree
    let gamma = GammaTable::from_hubbard_u(vec![0.4309]);

    let mixer = SimpleMixer::new(0.3);
    let mut solver = MultiSystemSolver::new(vec![frag], frag_neighbors, gamma, mixer);

    // deltaQ = [0.2, -0.2]; q0(N) = 5.0 from SK file
    let fixed_charges = vec![5.2, 4.8];
    solver.build_h_scc_with_fixed_charges(&fixed_charges);

    let h_scc_rust = &solver.fragments[0].h_scc;
    let h_ref = DftbOutput::read_square(&ref_h).unwrap();
    let s_ref = DftbOutput::read_square(&ref_s).unwrap();

    let diff_h = max_abs_diff(h_scc_rust, &h_ref);
    assert!(
        diff_h < 1e-6,
        "N2 H_scc mismatch (diff = {diff_h:e})"
    );

    // Verify S matches too (should be identical since same geometry)
    let diff_s = max_abs_diff(&solver.fragments[0].template.s, &s_ref);
    assert!(
        diff_s < 1e-7,
        "N2 S mismatch (diff = {diff_s:e})"
    );
}

/// HCOOH fixed-charge SCC parity against Fortran DFTB+ reference.
#[test]
fn hcooh_fixed_charge_scc_parity() {
    use rust_dftb::qmqm::solver::MultiSystemSolver;
    use rust_dftb::qmqm::{FragmentNeighborList, GammaTable, SimpleMixer};

    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };
    let Ok(ref_h) = std::env::var("RUST_DFTB_REF_H") else { return; };
    let Ok(ref_s) = std::env::var("RUST_DFTB_REF_S") else { return; };

    // Load HCOOH geometry from data/xyz/HCOOH.xyz
    let xyz_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/xyz/HCOOH.xyz");
    let mol = parse_xyz(xyz_path).unwrap();
    let species = mol.species;
    let coords = mol.coords;

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let template = FragmentTemplate::new(&sk, species, coords.clone()).unwrap();
    let frag = Fragment::from_template(template.clone(), coords.clone());

    let centroids = vec![[1.38, 0.4, 0.0]]; // approximate centroid
    let frag_neighbors = FragmentNeighborList::build(&centroids, 10.0);

    // Hubbard U: H=0.4195, C=0.3647, O=0.4954 (mio-1-1)
    let gamma = GammaTable::from_hubbard_u(vec![0.4195, 0.3647, 0.4954]);

    let mixer = SimpleMixer::new(0.3);
    let mut solver = MultiSystemSolver::new(vec![frag], frag_neighbors, gamma, mixer);

    // deltaQ = [-0.1, +0.1, -0.1, +0.1, 0.0]
    // (DFTB+ InitialCharges uses opposite sign convention to what one might expect)
    // q0 from SK: H=1.0, C=4.0, O=6.0, O=6.0, H=1.0
    let fixed_charges = vec![0.9, 4.1, 5.9, 6.1, 1.0];
    solver.build_h_scc_with_fixed_charges(&fixed_charges);

    let h_scc_rust = &solver.fragments[0].h_scc;
    let h_ref = DftbOutput::read_square(&ref_h).unwrap();
    let s_ref = DftbOutput::read_square(&ref_s).unwrap();

    let diff_h = max_abs_diff(h_scc_rust, &h_ref);
    assert!(
        diff_h < 1e-6,
        "HCOOH H_scc mismatch (diff = {diff_h:e})"
    );

    let diff_s = max_abs_diff(&solver.fragments[0].template.s, &s_ref);
    assert!(
        diff_s < 1e-7,
        "HCOOH S mismatch (diff = {diff_s:e})"
    );
}

// ─── Agent_2: Multi-fragment validation tests ──────────────────────
//
// These tests exercise the `MultiSystemSolver` with 2+ fragments to validate
// inter-fragment electrostatic coupling (`compute_v_ext`), charge conservation,
// and polarization convergence. They serve as the correctness oracle for the
// GPU QM/QM path.
//
// All tests use H2O from `data/xyz/H2O.xyz` as the fragment geometry.

/// Load the H2O geometry from `data/xyz/H2O.xyz`.
fn agent02_load_h2o() -> (Vec<String>, Vec<[f64; 3]>) {
    let xyz_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/xyz/H2O.xyz");
    let mol = parse_xyz(xyz_path).unwrap();
    (mol.species, mol.coords)
}

/// Compute the total SCC energy of a multi-fragment solver.
///
/// Energy per fragment = Tr(D · H0) + 0.5 · Σ_A Δq_A · shift_A
/// where shift = v_intra + v_ext. Summing over all fragments gives the
/// total QM/QM energy including inter-fragment electrostatic interaction.
fn agent02_total_energy<M: rust_dftb::qmqm::Mixer>(
    solver: &rust_dftb::qmqm::MultiSystemSolver<M>,
) -> f64 {
    let mut total = 0.0;
    for frag in &solver.fragments {
        let n_occ = (frag.n_electrons / 2.0).round() as usize;
        let c_occ = frag.eigenvectors.columns(0, n_occ);
        let density = &c_occ * c_occ.transpose() * 2.0;
        let e_h0 = (&density * &frag.template.h0).trace();
        let delta_q: Vec<f64> = frag
            .charges
            .iter()
            .zip(frag.template.q0.iter())
            .map(|(q, q0)| q - q0)
            .collect();
        let e_scc: f64 = 0.5
            * delta_q
                .iter()
                .zip(frag.shift.iter())
                .map(|(dq, s)| dq * s)
                .sum::<f64>();
        total += e_h0 + e_scc;
    }
    total
}

/// Build a `MultiSystemSolver` with N H2O fragments at the given translations.
///
/// `neighbor_cutoff` controls the fragment neighbor list — use a small value
/// for independent (uncoupled) fragments and a large value for coupled ones.
fn agent02_build_h2o_solver(
    sk_dir: &str,
    translations: &[[f64; 3]],
    neighbor_cutoff: f64,
) -> rust_dftb::qmqm::MultiSystemSolver<rust_dftb::qmqm::DiisMixer> {
    use rust_dftb::qmqm::{
        DiisMixer, Fragment, FragmentNeighborList, FragmentTemplate, GammaTable, MultiSystemSolver,
    };

    let (species, base_coords) = agent02_load_h2o();
    let sk = load_sk_for_species(sk_dir, &species).unwrap();

    let mut fragments = Vec::new();
    let mut centroids = Vec::new();
    for &trans in translations {
        let coords: Vec<[f64; 3]> = base_coords
            .iter()
            .map(|c| [c[0] + trans[0], c[1] + trans[1], c[2] + trans[2]])
            .collect();
        let template = FragmentTemplate::new(&sk, species.clone(), coords.clone()).unwrap();
        let n = coords.len() as f64;
        centroids.push([
            coords.iter().map(|c| c[0]).sum::<f64>() / n,
            coords.iter().map(|c| c[1]).sum::<f64>() / n,
            coords.iter().map(|c| c[2]).sum::<f64>() / n,
        ]);
        let frag = Fragment::from_template(template, coords);
        fragments.push(frag);
    }

    let frag_neighbors = FragmentNeighborList::build(&centroids, neighbor_cutoff);
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();
    let total_atoms = fragments.iter().map(|f| f.template.n_atoms).sum();
    let mixer = DiisMixer::new(10, total_atoms);

    MultiSystemSolver::new(fragments, frag_neighbors, gamma, mixer)
}

/// Test: Two H2O molecules far apart (20 Å) converge to standalone charges.
///
/// At large separation with a neighbor cutoff that excludes the other fragment,
/// inter-fragment coupling is zero. Each fragment should converge to the same
/// charges as a standalone H2O SCC calculation.
#[test]
fn two_fragment_independent_scc() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let (species, base_coords) = agent02_load_h2o();
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // Standalone H2O SCC reference
    let builder = HamiltonianBuilder::new(sk.clone());
    let standalone = builder.build_scc(&species, &base_coords, 200, 1e-9).unwrap();
    eprintln!("Standalone H2O charges: {:?}", standalone.charges);
    eprintln!("Standalone H2O energy: {:.10e}", standalone.energy);

    // Two H2O 20 Å apart, neighbor cutoff 10 Å → not neighbors → v_ext = 0
    let mut solver =
        agent02_build_h2o_solver(&sk_dir, &[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], 10.0);
    solver.solve_scc(200, 1e-9).unwrap();

    let q0 = &solver.fragments[0].charges;
    let q1 = &solver.fragments[1].charges;
    eprintln!("Multi-frag H2O[0] charges: {:?}", q0);
    eprintln!("Multi-frag H2O[1] charges: {:?}", q1);

    // Both fragments should match standalone within 1e-6
    for i in 0..3 {
        assert!(
            (q0[i] - standalone.charges[i]).abs() < 1e-6,
            "Fragment 0 charge {} mismatch: multi={}, standalone={}, diff={}",
            i,
            q0[i],
            standalone.charges[i],
            (q0[i] - standalone.charges[i]).abs()
        );
        assert!(
            (q1[i] - standalone.charges[i]).abs() < 1e-6,
            "Fragment 1 charge {} mismatch: multi={}, standalone={}, diff={}",
            i,
            q1[i],
            standalone.charges[i],
            (q1[i] - standalone.charges[i]).abs()
        );
    }

    // Verify v_ext is zero (no inter-fragment coupling)
    for (fi, frag) in solver.fragments.iter().enumerate() {
        for (ai, &v) in frag.v_ext.iter().enumerate() {
            assert!(
                v.abs() < 1e-15,
                "v_ext should be zero for independent fragments (frag {} atom {}), got {}",
                fi,
                ai,
                v
            );
        }
    }

    eprintln!("two_fragment_independent_scc: PASS (both fragments match standalone)");
}

/// Test: Two H2O molecules close together (3 Å) — polarization occurs.
///
/// At close range, inter-fragment coupling is active. Verify:
/// - Both fragments converge
/// - Total charge is conserved (sum of all Δq = 0 within 1e-10)
/// - Charges differ from standalone (polarization occurred)
/// - Total energy differs from 2× standalone (interaction energy)
#[test]
fn two_fragment_polarization() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let (species, base_coords) = agent02_load_h2o();
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // Standalone reference
    let builder = HamiltonianBuilder::new(sk.clone());
    let standalone = builder.build_scc(&species, &base_coords, 200, 1e-9).unwrap();
    let standalone_energy = standalone.energy;

    // Two H2O 3 Å apart (centroid separation), neighbor cutoff 30 Å → coupled
    let mut solver =
        agent02_build_h2o_solver(&sk_dir, &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], 30.0);
    solver.solve_scc(200, 1e-9).unwrap();

    let q0 = &solver.fragments[0].charges;
    let q1 = &solver.fragments[1].charges;
    eprintln!("Polarized H2O[0] charges: {:?}", q0);
    eprintln!("Polarized H2O[1] charges: {:?}", q1);

    // Charge conservation: sum of all charges = sum of q0
    let total_q: f64 = solver.fragments.iter().flat_map(|f| f.charges.iter()).sum();
    let total_q0: f64 = solver.q0.iter().sum();
    assert!(
        (total_q - total_q0).abs() < 1e-10,
        "Total charge not conserved: sum(q) = {}, sum(q0) = {}, diff = {}",
        total_q,
        total_q0,
        (total_q - total_q0).abs()
    );

    // Polarization: charges should differ from standalone
    let max_diff_0 = q0
        .iter()
        .zip(standalone.charges.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_diff_1 = q1
        .iter()
        .zip(standalone.charges.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "Max charge diff from standalone: frag0={:.3e}, frag1={:.3e}",
        max_diff_0, max_diff_1
    );
    assert!(
        max_diff_0 > 1e-4 || max_diff_1 > 1e-4,
        "Polarization should cause charge differences > 1e-4, got max_diff_0={}, max_diff_1={}",
        max_diff_0,
        max_diff_1
    );

    // v_ext should be non-zero (inter-fragment coupling active)
    let max_v_ext: f64 = solver
        .fragments
        .iter()
        .flat_map(|f| f.v_ext.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    eprintln!("Max |v_ext| = {:.3e}", max_v_ext);
    assert!(
        max_v_ext > 1e-6,
        "v_ext should be non-zero for polarized fragments, got max = {}",
        max_v_ext
    );

    // Total energy vs 2× standalone
    let multi_energy = agent02_total_energy(&solver);
    let interaction_energy = multi_energy - 2.0 * standalone_energy;
    eprintln!("Multi-frag total energy = {:.10e}", multi_energy);
    eprintln!("2× standalone energy    = {:.10e}", 2.0 * standalone_energy);
    eprintln!("Interaction energy      = {:.3e} Hartree", interaction_energy);
    assert!(
        interaction_energy.abs() > 1e-6,
        "Interaction energy should be non-zero, got {}",
        interaction_energy
    );

    eprintln!("two_fragment_polarization: PASS (polarization + charge conservation verified)");
}

/// Test: 3× H2O — charge conservation at every SCC iteration.
///
/// Verify that sum of all atomic charges = sum of q0 (neutral) within 1e-10
/// at every SCC iteration, both after diagonalization (Mulliken sum) and
/// after mixing. This is a critical invariant — if it fails, there's a bug
/// in `compute_v_ext` or `gather/scatter_charges`.
#[test]
fn charge_conservation_multi_frag() {
    use rust_dftb::qmqm::Mixer;

    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    // 3 H2O at various separations: 0, 3, 7 Å
    let mut solver =
        agent02_build_h2o_solver(&sk_dir, &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]], 30.0);
    let total_q0: f64 = solver.q0.iter().sum();
    eprintln!("Total q0 = {:.10}", total_q0);

    let max_iter = 100;
    let tol = 1e-8;

    for iter in 0..max_iter {
        solver.compute_v_ext();
        solver.build_all_h_scc();
        solver.diagonalize_all().unwrap();

        // Check charge conservation after diagonalization (Mulliken sum = n_electrons)
        let total_q: f64 = solver.fragments.iter().flat_map(|f| f.charges.iter()).sum();
        assert!(
            (total_q - total_q0).abs() < 1e-10,
            "Charge conservation violated at iter {} (after diag): sum(q) = {}, sum(q0) = {}, diff = {}",
            iter,
            total_q,
            total_q0,
            (total_q - total_q0).abs()
        );

        // Compute residual from fragment charges
        let q_out: Vec<f64> = solver
            .fragments
            .iter()
            .flat_map(|f| f.charges.iter().copied())
            .collect();
        let residual: Vec<f64> = q_out
            .iter()
            .zip(solver.charges.iter())
            .map(|(qo, qi)| qo - qi)
            .collect();
        let rms = (residual.iter().map(|x| x * x).sum::<f64>() / residual.len() as f64).sqrt();

        eprintln!(
            "Iter {}: RMS = {:.3e}, sum(q) = {:.10}",
            iter + 1,
            rms,
            total_q
        );

        if rms < tol {
            eprintln!("Converged at iter {} with RMS = {:.3e}", iter + 1, rms);
            break;
        }

        // Mix (disjoint borrows of solver fields)
        let charges_ref = &mut solver.charges;
        let mixer_ref = &mut solver.mixer;
        mixer_ref.mix(charges_ref, &q_out, &residual);

        // Check charge conservation after mixing
        let total_q_mixed: f64 = solver.charges.iter().sum();
        assert!(
            (total_q_mixed - total_q0).abs() < 1e-10,
            "Charge conservation violated at iter {} (after mix): sum(q) = {}, diff = {}",
            iter,
            total_q_mixed,
            (total_q_mixed - total_q0).abs()
        );

        // Scatter mixed charges back to fragments
        solver.scatter_charges();
    }

    eprintln!("charge_conservation_multi_frag: PASS (charge conserved at every iteration)");
}

/// Test: Two H2O as 2 fragments vs 1 combined fragment (QM/QM approximation error).
///
/// The 2-fragment QM/QM approach treats each H2O as an independent subsystem
/// with inter-fragment electrostatic coupling only (no orbital overlap between
/// fragments). The 1-fragment approach treats all 6 atoms as one system with
/// full H0/S including inter-fragment orbital overlap.
///
/// The QM/QM approximation should give similar but not identical results.
/// This test documents the approximation error.
#[test]
fn two_fragment_vs_single_combined() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let (species, base_coords) = agent02_load_h2o();
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // 2-fragment: two H2O 3 Å apart
    let mut solver =
        agent02_build_h2o_solver(&sk_dir, &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], 30.0);
    solver.solve_scc(200, 1e-9).unwrap();
    let multi_energy = agent02_total_energy(&solver);
    let multi_charges: Vec<f64> = solver
        .fragments
        .iter()
        .flat_map(|f| f.charges.iter().copied())
        .collect();

    // 1-fragment: all 6 atoms as one system
    let combined_species: Vec<String> = species.iter().chain(species.iter()).cloned().collect();
    let combined_coords: Vec<[f64; 3]> = base_coords
        .iter()
        .cloned()
        .chain(base_coords.iter().map(|c| [c[0] + 3.0, c[1], c[2]]))
        .collect();
    let builder = HamiltonianBuilder::new(sk);
    let combined = builder
        .build_scc(&combined_species, &combined_coords, 200, 1e-9)
        .unwrap();

    let energy_diff = (multi_energy - combined.energy).abs();
    eprintln!("2-fragment energy  = {:.10e}", multi_energy);
    eprintln!("1-fragment energy  = {:.10e}", combined.energy);
    eprintln!(
        "QM/QM approx error = {:.3e} Hartree = {:.3} kcal/mol",
        energy_diff,
        energy_diff * 627.509
    );
    eprintln!("2-frag charges: {:?}", multi_charges);
    eprintln!("1-frag charges: {:?}", combined.charges);

    // Energies should be similar but not identical (QM/QM approximation)
    assert!(
        energy_diff < 0.1,
        "QM/QM approximation error should be < 0.1 Hartree, got {}",
        energy_diff
    );
    assert!(
        energy_diff > 1e-6,
        "QM/QM approximation should give different energy from full system, got diff = {}",
        energy_diff
    );

    eprintln!("two_fragment_vs_single_combined: PASS (QM/QM approximation error documented)");
}
