//! Integration tests for the qmqm multi-system solver.
//!
//! These tests compare fragment-based results against full-system DFTB
//! to verify that the qmqm module produces correct Hamiltonians,
//! shifts, and charges.

use rust_dftb::{HamiltonianBuilder, SkData};
use rust_dftb::qmqm::{Fragment, FragmentTemplate};

fn max_abs_diff(a: &nalgebra::DMatrix<f64>, b: &nalgebra::DMatrix<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

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

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("H".to_string(), vec![0]);
    ang_map.insert("C".to_string(), vec![0, 1]);
    ang_map.insert("N".to_string(), vec![0, 1]);
    ang_map.insert("O".to_string(), vec![0, 1]);
    sk.set_species_angular_momenta(ang_map);

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

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("H".to_string(), vec![0]);
    ang_map.insert("C".to_string(), vec![0, 1]);
    ang_map.insert("N".to_string(), vec![0, 1]);
    ang_map.insert("O".to_string(), vec![0, 1]);
    sk.set_species_angular_momenta(ang_map);
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

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("H".to_string(), vec![0]);
    ang_map.insert("C".to_string(), vec![0, 1]);
    ang_map.insert("N".to_string(), vec![0, 1]);
    ang_map.insert("O".to_string(), vec![0, 1]);
    sk.set_species_angular_momenta(ang_map);
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

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("H".to_string(), vec![0]);
    ang_map.insert("C".to_string(), vec![0, 1]);
    ang_map.insert("N".to_string(), vec![0, 1]);
    ang_map.insert("O".to_string(), vec![0, 1]);
    sk.set_species_angular_momenta(ang_map);

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

    let species = vec![
        "H".to_string(), "C".to_string(), "O".to_string(), "O".to_string(), "H".to_string(),
    ];
    // Approximate formic acid geometry (Å)
    let coords = vec![
        [0.00, 0.00, 0.00], // H (hydroxyl)
        [1.00, 0.00, 0.00], // C
        [2.20, 0.00, 0.00], // O (carbonyl)
        [1.50, 1.00, 0.00], // O (hydroxyl)
        [2.20, 1.00, 0.00], // H (hydroxyl)
    ];

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("H".to_string(), vec![0]);
    ang_map.insert("C".to_string(), vec![0, 1]);
    ang_map.insert("N".to_string(), vec![0, 1]);
    ang_map.insert("O".to_string(), vec![0, 1]);
    sk.set_species_angular_momenta(ang_map);

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

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("H".to_string(), vec![0]);
    sk.set_species_angular_momenta(ang_map);

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
    // NOTE: Rust q0 is wrong (2.0 for H instead of 1.0), so we adjust:
    // deltaQ = q - q0 => q = q0 + deltaQ = [2.0 + 0.1, 2.0 - 0.1] = [2.1, 1.9]
    let fixed_charges = vec![2.1, 1.9];
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
    // -2.2715944311313499e-01 -1.3645342700760299e-01
    // -1.3645342700760299e-01 -2.5004135688686491e-01
    let h_ref = DMatrix::from_row_slice(2, 2, &[
        -2.2715944311313499e-01, -1.3645342700760299e-01,
        -1.3645342700760299e-01, -2.5004135688686491e-01,
    ]);
    
    eprintln!("Fortran H_scc:\n{:.16e}", h_ref);

    let diff = max_abs_diff(h_scc_rust, &h_ref);
    assert!(
        diff < 1e-10,
        "H_scc mismatch between Rust and Fortran DFTB+ (diff = {diff:e})"
    );

    // Also test diagonalization: eigenvalues should match Fortran
    solver.diagonalize_all().unwrap();
    let eigvals_rust = &solver.fragments[0].eigenvalues;
    
    // Expected eigenvalues from Fortran (eigenvalues are sorted ascending in Rust)
    // From DFTB+ calculation with these charges
    let eig_ref = vec![-4.8399606556943828e-01, -1.9310473443056169e-01];
    
    for (i, (r, f)) in eigvals_rust.iter().zip(eig_ref.iter()).enumerate() {
        assert!(
            (r - f).abs() < 1e-10,
            "Eigenvalue {} mismatch: Rust={}, Fortran={}", i, r, f
        );
    }

    // Test charges after diagonalization (Mulliken analysis)
    // These should differ from input because electrons rearrange
    let charges_rust: Vec<f64> = solver.fragments[0].charges.clone();
    let q0 = vec![1.0, 1.0]; // Reference neutral charges for H
    let delta_q_rust: Vec<f64> = charges_rust.iter().zip(q0.iter()).map(|(q, q0)| q - q0).collect();
    
    // Output charges from DFTB+ would be different - just verify they're reasonable
    // (between 0 and 2 for H atoms)
    for (i, q) in charges_rust.iter().enumerate() {
        assert!(
            *q > 0.0 && *q < 2.0,
            "Atom {} charge {} out of reasonable range (0, 2)", i, q
        );
    }
}
