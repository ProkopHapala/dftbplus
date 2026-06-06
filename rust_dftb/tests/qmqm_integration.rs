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
