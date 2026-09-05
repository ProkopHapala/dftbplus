//! DFTB force parity tests (Agent_3, Wave 1).
//!
//! These tests compute DFTB forces with the Rust implementation and compare
//! them against Fortran DFTB+ reference forces parsed from `detailed.out`.
//!
//! Driven by environment variables:
//! - `RUST_DFTB_SK_DIR`     – directory with mio `.skf` files
//! - `RUST_DFTB_FORCES_XYZ` – path to the molecule `.xyz` file
//! - `RUST_DFTB_FORCES_REF` – path to a reference forces file produced by
//!   `run_forces.py`. Format: one line per atom with 3 whitespace-separated
//!   floats (fx, fy, fz) in Hartree/Bohr, matching DFTB+ `detailed.out`.
//! - `RUST_DFTB_FORCES_SCC` – if set to "1" or "yes", run the SCC force test;
//!   otherwise run the non-SCC force test.
//! - `RUST_DFTB_FORCES_TOL` – tolerance (default 1e-5).
//!
//! See `tests/run_forces.py` for the end-to-end driver that generates the
//! reference file and invokes this test.

use rust_dftb::{
    load_sk_for_species,
    parse_f64_list,
    parse_xyz,
    HamiltonianBuilder,
};
use rust_dftb::methods::dftb::forces::{
    compute_non_scc_forces,
    compute_scc_forces,
    forces_hartree_ang_to_bohr,
};

/// Load molecule + reference forces from env vars.
fn load_env() -> Option<(Vec<String>, Vec<[f64; 3]>, Vec<[f64; 3]>, f64, bool)> {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return None;
    };
    let Ok(xyz_path) = std::env::var("RUST_DFTB_FORCES_XYZ") else {
        eprintln!("Skipping: RUST_DFTB_FORCES_XYZ not set");
        return None;
    };
    let Ok(ref_path) = std::env::var("RUST_DFTB_FORCES_REF") else {
        eprintln!("Skipping: RUST_DFTB_FORCES_REF not set");
        return None;
    };

    let mol = parse_xyz(&xyz_path).expect("failed to parse XYZ");
    let species = mol.species;
    let coords = mol.coords;

    // Parse reference forces file: one atom per line, 3 floats (Hartree/Bohr).
    let ref_text = std::fs::read_to_string(&ref_path)
        .unwrap_or_else(|e| panic!("failed to read ref forces {ref_path}: {e}"));
    let mut ref_forces: Vec<[f64; 3]> = Vec::new();
    for line in ref_text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let vals = parse_f64_list(line);
        assert_eq!(vals.len(), 3,
            "ref force line must have 3 values, got {}: '{line}'", vals.len());
        ref_forces.push([vals[0], vals[1], vals[2]]);
    }
    assert_eq!(ref_forces.len(), species.len(),
        "ref force count {} != atom count {}", ref_forces.len(), species.len());

    let tol: f64 = std::env::var("RUST_DFTB_FORCES_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-5);

    let scc_flag = std::env::var("RUST_DFTB_FORCES_SCC")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("yes"))
        .unwrap_or(false);

    // Stash sk_dir into the env so forces.rs can read it (redundant but safe).
    std::env::set_var("RUST_DFTB_SK_DIR", &sk_dir);

    Some((species, coords, ref_forces, tol, scc_flag))
}

/// Compute the maximum absolute component-wise difference between two
/// force arrays, and print a per-atom breakdown.
fn compare_forces(rust: &[[f64; 3]], ref_f: &[[f64; 3]], label: &str) -> f64 {
    assert_eq!(rust.len(), ref_f.len());
    let mut max_err = 0.0f64;
    let mut max_atom = 0;
    let mut max_comp = 0;
    for (i, (r, f)) in rust.iter().zip(ref_f.iter()).enumerate() {
        for c in 0..3 {
            let d = (r[c] - f[c]).abs();
            if d > max_err {
                max_err = d;
                max_atom = i;
                max_comp = c;
            }
        }
    }
    eprintln!("[{label}] max |ΔF| = {max_err:.6e} at atom {max_atom} comp {max_comp}");
    for (i, (r, f)) in rust.iter().zip(ref_f.iter()).enumerate() {
        let dx = (r[0] - f[0]).abs();
        let dy = (r[1] - f[1]).abs();
        let dz = (r[2] - f[2]).abs();
        eprintln!(
            "  atom {i}: rust = [{:+.10e} {:+.10e} {:+.10e}]  ref = [{:+.10e} {:+.10e} {:+.10e}]  |Δ| = [{:.3e} {:.3e} {:.3e}]",
            r[0], r[1], r[2], f[0], f[1], f[2], dx, dy, dz,
        );
    }
    max_err
}

/// Count total valence electrons from the SK onsite q0 values.
fn count_electrons(sk_dir: &str, species: &[String]) -> f64 {
    let sk = load_sk_for_species(sk_dir, species).expect("failed to load SK");
    let builder = HamiltonianBuilder::new(sk);
    let ctx = rust_dftb::methods::dftb::hamiltonian::SystemContext::from_sk_data(
        &builder.sk, species,
    ).expect("failed to build ctx");
    // q0 per atom = sum of valence electrons from onsite params.
    let mut total = 0.0f64;
    for i in 0..species.len() {
        let si = ctx.atom_species[i] as usize;
        let p = ctx.species_onsite[si];
        total += p.q0;
    }
    total
}

/// Non-SCC force parity test.
#[test]
fn non_scc_forces_from_xyz() {
    let Some((species, coords, ref_forces, tol, scc_flag)) = load_env() else {
        return;
    };
    if scc_flag {
        eprintln!("Skipping non_scc_forces_from_xyz: RUST_DFTB_FORCES_SCC is set");
        return;
    }

    let sk_dir = std::env::var("RUST_DFTB_SK_DIR").unwrap();
    let sk = load_sk_for_species(&sk_dir, &species).expect("failed to load SK");
    let builder = HamiltonianBuilder::new(sk);

    let n_electrons = count_electrons(&sk_dir, &species);
    eprintln!("[non_scc] n_atoms = {}, n_electrons = {}", species.len(), n_electrons);

    let forces = compute_non_scc_forces(&builder, &species, &coords, n_electrons)
        .expect("non-SCC force computation failed");

    // Convert Rust forces (Hartree/Å) to Hartree/Bohr for comparison.
    let rust_bohr = forces_hartree_ang_to_bohr(&forces.forces);

    eprintln!("[non_scc] component breakdown (Hartree/Å):");
    for i in 0..species.len() {
        eprintln!(
            "  atom {i} {:>2}: non_scc = [{:+.10e} {:+.10e} {:+.10e}]  rep = [{:+.10e} {:+.10e} {:+.10e}]",
            species[i],
            forces.non_scc[i][0], forces.non_scc[i][1], forces.non_scc[i][2],
            forces.repulsive[i][0], forces.repulsive[i][1], forces.repulsive[i][2],
        );
    }

    let max_err = compare_forces(&rust_bohr, &ref_forces, "non_scc");
    assert!(max_err < tol,
        "non-SCC force mismatch: max |ΔF| = {max_err:.6e} >= tol {tol:.6e}");
}

/// SCC force parity test.
#[test]
fn scc_forces_from_xyz() {
    let Some((species, coords, ref_forces, tol, scc_flag)) = load_env() else {
        return;
    };
    if !scc_flag {
        eprintln!("Skipping scc_forces_from_xyz: RUST_DFTB_FORCES_SCC not set");
        return;
    }

    let sk_dir = std::env::var("RUST_DFTB_SK_DIR").unwrap();
    let sk = load_sk_for_species(&sk_dir, &species).expect("failed to load SK");
    let builder = HamiltonianBuilder::new(sk);

    eprintln!("[scc] n_atoms = {}", species.len());

    // Run SCC to convergence.
    let scc = builder.build_scc(&species, &coords, 1000, 1e-10)
        .expect("SCC did not converge");
    eprintln!("[scc] converged in {} iterations, energy = {:.10}", scc.n_iter, scc.energy);
    eprintln!("[scc] charges = {:?}", scc.charges);
    eprintln!("[scc] q0      = {:?}", scc.q0);

    let forces = compute_scc_forces(&builder, &species, &coords, &scc)
        .expect("SCC force computation failed");

    let rust_bohr = forces_hartree_ang_to_bohr(&forces.forces);

    eprintln!("[scc] component breakdown (Hartree/Å):");
    for i in 0..species.len() {
        eprintln!(
            "  atom {i} {:>2}: non_scc = [{:+.10e} {:+.10e} {:+.10e}]  shift = [{:+.10e} {:+.10e} {:+.10e}]  dc = [{:+.10e} {:+.10e} {:+.10e}]  rep = [{:+.10e} {:+.10e} {:+.10e}]",
            species[i],
            forces.non_scc[i][0], forces.non_scc[i][1], forces.non_scc[i][2],
            forces.scc_shift[i][0], forces.scc_shift[i][1], forces.scc_shift[i][2],
            forces.scc_dc[i][0], forces.scc_dc[i][1], forces.scc_dc[i][2],
            forces.repulsive[i][0], forces.repulsive[i][1], forces.repulsive[i][2],
        );
    }

    let max_err = compare_forces(&rust_bohr, &ref_forces, "scc");
    assert!(max_err < tol,
        "SCC force mismatch: max |ΔF| = {max_err:.6e} >= tol {tol:.6e}");
}
