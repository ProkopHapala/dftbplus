//! Full SCC convergence parity tests.
//!
//! These tests run the Rust self-consistent charge loop to convergence and
//! compare the resulting charges and eigenvalues against Fortran DFTB+
//! references.
//!
//! Geometry is loaded from .xyz files so that the same file can be used
//! for both Fortran DFTB+ reference generation and Rust testing.
//!
//! Driven by environment variables:
//! - RUST_DFTB_SK_DIR   – directory with .skf files (mio set)
//! - RUST_DFTB_SCC_XYZ  – path to .xyz file with molecule geometry
//! - RUST_DFTB_SCC_REF_CHARGES – optional: comma-separated converged charges from Fortran
//! - RUST_DFTB_SCC_REF_EIGS    – optional: comma-separated converged eigenvalues from Fortran
//! - RUST_DFTB_SCC_TOL         – tolerance (default 1e-6)
//!
//! To generate Fortran reference:
//!   1. Convert .xyz to .gen (or use run_parity.py which does this)
//!   2. Run DFTB+ with SCC=Yes, SCCTolerance=1e-10, MaxSCCIterations=200
//!   3. Read converged charges from detailed.out
//!   4. Read eigenvalues from eigenvec.out or band.out

use rust_dftb::{
    load_sk_for_species, max_abs_diff, max_abs_diff_vec, parse_f64_list, parse_xyz,
    DftbOutput, HamiltonianBuilder, SccResult,
};

/// Run SCC on an XYZ file and return the result.
/// Skips test if RUST_DFTB_SK_DIR or RUST_DFTB_SCC_XYZ not set.
fn run_scc_from_env(label: &str) -> Option<(SccResult, Vec<String>)> {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping {label}: RUST_DFTB_SK_DIR not set");
        return None;
    };
    let Ok(xyz_path) = std::env::var("RUST_DFTB_SCC_XYZ") else {
        eprintln!("Skipping {label}: RUST_DFTB_SCC_XYZ not set");
        return None;
    };

    let mol = parse_xyz(&xyz_path).unwrap();
    let species = mol.species;
    let coords = mol.coords;
    eprintln!("[{label}] Loaded {} atoms from {xyz_path}", species.len());

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    let builder = HamiltonianBuilder::new(sk);
    let result = builder.build_scc(&species, &coords, 1000, 1e-8).unwrap();

    eprintln!("[{label}] SCC converged in {} iterations", result.n_iter);
    eprintln!("  charges: {:?}", result.charges);
    eprintln!("  q0:      {:?}", result.q0);
    eprintln!("  deltaQ:  {:?}", result.charges.iter().zip(result.q0.iter()).map(|(q, q0)| q - q0).collect::<Vec<_>>());
    eprintln!("  eigenvalues: {:?}", result.eigenvalues.as_slice());
    eprintln!("  energy: {:.10}", result.energy);
    eprintln!("  density trace: {:.10}", result.density.trace());

    Some((result, species))
}

/// Compare charges and eigenvalues against Fortran reference if provided.
fn check_fortran_parity(label: &str, result: &SccResult) {
    let tol: f64 = std::env::var("RUST_DFTB_SCC_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-6);

    if let Ok(ref_charges) = std::env::var("RUST_DFTB_SCC_REF_CHARGES") {
        let ref_q = parse_f64_list(&ref_charges);
        assert_eq!(ref_q.len(), result.charges.len(),
            "{label}: charge count mismatch (ref={}, rust={})", ref_q.len(), result.charges.len());
        let diff = max_abs_diff_vec(&result.charges, &ref_q);
        eprintln!("  Charge diff vs Fortran: {diff:e}");
        assert!(diff < tol, "{label} SCC charges mismatch: diff = {diff:e} (tol = {tol:e})");
    }

    if let Ok(ref_eigs) = std::env::var("RUST_DFTB_SCC_REF_EIGS") {
        let ref_e = parse_f64_list(&ref_eigs);
        assert_eq!(ref_e.len(), result.eigenvalues.len(),
            "{label}: eigenvalue count mismatch (ref={}, rust={})", ref_e.len(), result.eigenvalues.len());
        let diff = max_abs_diff_vec(result.eigenvalues.as_slice(), &ref_e);
        eprintln!("  Eigenvalue diff vs Fortran: {diff:e}");
        assert!(diff < tol * 10.0, "{label} SCC eigenvalues mismatch: diff = {diff:e} (tol = {:.3e})", tol * 10.0);
    }

    // Compare total energy
    if let Ok(ref_energy_s) = std::env::var("RUST_DFTB_SCC_REF_ENERGY") {
        let ref_energy: f64 = ref_energy_s.parse().unwrap();
        let diff = (result.energy - ref_energy).abs();
        eprintln!("  Energy: rust={:.10}, fortran={:.10}, diff={diff:e}", result.energy, ref_energy);
        assert!(diff < tol * 100.0, "{label} SCC energy mismatch: diff = {diff:e} (tol = {:.3e})", tol * 100.0);
    }

    // Compare H_scc matrix
    if let Ok(ref_h_scc_path) = std::env::var("RUST_DFTB_SCC_REF_H_SCC") {
        let h_ref = DftbOutput::read_square(&ref_h_scc_path).unwrap();
        let diff = max_abs_diff(&result.h_scc, &h_ref);
        eprintln!("  H_scc diff vs Fortran: {diff:e}");
        assert!(diff < tol, "{label} H_scc mismatch: diff = {diff:e} (tol = {tol:e})");
    }

    // Compare S matrix
    if let Ok(ref_s_path) = std::env::var("RUST_DFTB_SCC_REF_S") {
        let s_ref = DftbOutput::read_square(&ref_s_path).unwrap();
        let diff = max_abs_diff(&result.s, &s_ref);
        eprintln!("  S diff vs Fortran: {diff:e}");
        assert!(diff < tol, "{label} S mismatch: diff = {diff:e} (tol = {tol:e})");
    }
}

/// SCC convergence test for a molecule loaded from XYZ.
///
/// Set env vars:
///   RUST_DFTB_SK_DIR=/path/to/mio-1-1
///   RUST_DFTB_SCC_XYZ=/path/to/molecule.xyz
///   RUST_DFTB_SCC_REF_CHARGES=... (optional, for Fortran parity)
///   RUST_DFTB_SCC_REF_EIGS=... (optional)
///
/// Example XYZ file for H2:
/// ```text
/// 2
/// H2
/// H  0.0  0.0  0.0
/// H  0.74 0.0  0.0
/// ```
#[test]
fn scc_convergence_from_xyz() {
    let Some((result, species)) = run_scc_from_env("scc_from_xyz") else { return; };

    // Basic sanity checks:
    // 1. Charge conservation: sum(charges) ≈ sum(q0) = total electrons
    let total_q: f64 = result.charges.iter().sum();
    let total_q0: f64 = result.q0.iter().sum();
    eprintln!("  total charge: {total_q:.12} (expected {total_q0:.12})");
    assert!((total_q - total_q0).abs() < 1e-10, "Charge not conserved");

    // 2. For homonuclear diatomics, check zero charge transfer
    if species.len() == 2 && species[0] == species[1] {
        let dq: Vec<f64> = result.charges.iter().zip(result.q0.iter()).map(|(q, q0)| q - q0).collect();
        let max_dq = dq.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        eprintln!("  homonuclear check: max |deltaQ| = {max_dq:e}");
        assert!(max_dq < 1e-8, "Homonuclear diatomic should have zero charge transfer, max |deltaQ| = {max_dq:e}");
    }

    // 3. Fortran parity if reference provided
    check_fortran_parity("scc_from_xyz", &result);
}
