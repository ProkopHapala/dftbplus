//! Parity tests: Rust xTB implementation vs tblite C API reference.

use std::process::{Command, Stdio};
use std::io::Write;
use rust_dftb::compare_matrices;
use rust_dftb::compare_vecs;

const TBLITE_HELPER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tblite_helper");

/// Run tblite C helper on a molecule and return parsed JSON.
/// method: 1 = GFN1, 2 = GFN2
/// Returns None if tblite_helper is not available
fn run_tblite(nat: usize, charge: i32, uhf: i32, method: i32, atoms: &[(usize, [f64; 3])]) -> Option<serde_json::Value> {
    let mut child = Command::new(TBLITE_HELPER)
        .arg(format!("{}", nat))
        .arg(format!("{}", charge))
        .arg(format!("{}", uhf))
        .arg(format!("{}", method))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    if child.is_err() {
        return None;
    }

    let mut child = child.unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        for (_i, (z, pos)) in atoms.iter().enumerate() {
            writeln!(stdin, "{} {} {} {}", z, pos[0], pos[1], pos[2]).unwrap();
        }
    }

    let output = child.wait_with_output().expect("Failed to read tblite_helper output");
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Find the JSON part (skip any SCF output lines)
    let json_start = stdout.find('{').expect("No JSON found in output");
    let json_str = &stdout[json_start..];
    serde_json::from_str(json_str).ok()
}

/// Parse a flat array from JSON into a DMatrix.
fn json_to_dmatrix(json: &serde_json::Value, key: &str, n: usize) -> nalgebra::DMatrix<f64> {
    let arr = json[key].as_array().unwrap();
    let mut mat = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            // Fortran exports arrays in column-major order:
            // arr[k] corresponds to Fortran element (i+1, j+1) where k = j*n + i
            mat[(i, j)] = arr[j * n + i].as_f64().unwrap();
        }
    }
    mat
}

fn json_to_vec(json: &serde_json::Value, key: &str) -> Vec<f64> {
    json[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

#[test]
fn test_h2_gfn1_parity() {
    // H2 at equilibrium: 0.74 Å
    let atoms = vec![
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.74]),
    ];

    let ref_data = run_tblite(2, 0, 0, 1, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_h2_gfn1_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let nao = ref_data["nao"].as_i64().unwrap() as usize;
    println!("H2 nao = {}", nao);

    let h_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao);

    // Build with our Rust code (coordinates in Bohr)
    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74 * aatoau],
    ];
    let elem_idx = vec![0usize, 0]; // H = index 0
    let (h_rust, s_rust, _shell_elem, _shell_idx) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s(&coords, &elem_idx);

    // Print matrices for debugging
    println!("\nReference H0:");
    for i in 0..nao {
        for j in 0..nao {
            print!("{:14.8} ", h_ref[(i, j)]);
        }
        println!();
    }
    println!("\nRust H0:");
    for i in 0..nao {
        for j in 0..nao {
            print!("{:14.8} ", h_rust[(i, j)]);
        }
        println!();
    }
    println!("\nReference S:");
    for i in 0..nao {
        for j in 0..nao {
            print!("{:14.8} ", s_ref[(i, j)]);
        }
        println!();
    }
    println!("\nRust S:");
    for i in 0..nao {
        for j in 0..nao {
            print!("{:14.8} ", s_rust[(i, j)]);
        }
        println!();
    }

    compare_matrices(&h_rust, &h_ref, "H2 H0", 1e-4);
    compare_matrices(&s_rust, &s_ref, "H2 S", 1e-4);
}

#[test]
fn test_n2_gfn1_parity() {
    // N2 at equilibrium: 1.10 Å
    let atoms = vec![
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, 1.10]),
    ];

    let ref_data = run_tblite(2, 0, 0, 1, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_n2_gfn1_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let nao = ref_data["nao"].as_i64().unwrap() as usize;
    println!("N2 nao = {}", nao);

    let h_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao);

    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.10 * aatoau],
    ];
    let elem_idx = vec![6usize, 6]; // N = index 6
    let (h_rust, s_rust, _shell_elem, _shell_idx) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s(&coords, &elem_idx);

    compare_matrices(&h_rust, &h_ref, "N2 H0", 1e-4);
    compare_matrices(&s_rust, &s_ref, "N2 S", 1e-4);
}

#[test]
fn test_hcooh_gfn1_parity() {
    // Formic acid (HCOOH) approximate geometry in Angstrom
    let atoms = vec![
        (1, [-1.55,  1.10, 0.0]), // H (hydroxyl)
        (8, [-0.66,  1.10, 0.0]), // O (hydroxyl)
        (6, [ 0.00,  0.00, 0.0]), // C
        (8, [ 1.20,  0.00, 0.0]), // O (carbonyl)
        (1, [-0.36, -0.95, 0.0]), // H (on C)
    ];

    let ref_data = run_tblite(5, 0, 0, 1, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_hcooh_gfn1_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let nao = ref_data["nao"].as_i64().unwrap() as usize;
    println!("HCOOH nao = {}", nao);

    let h_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao);

    let aatoau = 1.889726133;
    let coords: Vec<[f64; 3]> = atoms.iter().map(|(_, p)| {
        [p[0] * aatoau, p[1] * aatoau, p[2] * aatoau]
    }).collect();
    let elem_idx = vec![0usize, 7, 5, 7, 0]; // H=0, O=7, C=5
    let (h_rust, s_rust, _shell_elem, _shell_idx) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s(&coords, &elem_idx);

    compare_matrices(&h_rust, &h_ref, "HCOOH H0", 1e-4);
    compare_matrices(&s_rust, &s_ref, "HCOOH S", 1e-4);
}

#[test]
fn test_h2_scc_parity() {
    // H2 at equilibrium: 0.74 Å
    let atoms = vec![
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.74]),
    ];

    let ref_data = run_tblite(2, 0, 0, 1, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_h2_scc_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let nat = ref_data["nat"].as_i64().unwrap() as usize;
    let nao = ref_data["nao"].as_i64().unwrap() as usize;

    let q_ref = json_to_vec(&ref_data, "charges");
    let emo_ref = json_to_vec(&ref_data, "eigenvalues");

    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74 * aatoau],
    ];
    let elem_idx = vec![0usize, 0]; // H = index 0

    // Run SCC
    let n_electrons = 2; // H2 has 2 electrons
    let (_density, qsh, emo) = rust_dftb::methods::xtb::scf::run_scf(
        &coords, &elem_idx, n_electrons, 100, 1e-6
    );

    // Convert shell charges to atomic charges
    let nshell_per_atom = vec![2, 2]; // H has 2 shells
    let q_rust = rust_dftb::methods::xtb::mulliken::atomic_charges(&qsh, &nshell_per_atom);

    println!("Reference charges: {:?}", q_ref);
    println!("Rust charges: {:?}", q_rust.data.as_vec());
    println!("Reference eigenvalues: {:?}", emo_ref);
    println!("Rust eigenvalues: {:?}", emo.data.as_vec());

    compare_vecs(q_rust.data.as_vec(), &q_ref, "H2 charges", 1e-3);
    compare_vecs(emo.data.as_vec(), &emo_ref, "H2 eigenvalues", 1e-3);
}

#[test]
fn test_n2_scc_parity() {
    // N2 at equilibrium: 1.10 Å
    let atoms = vec![
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, 1.10]),
    ];

    let ref_data = run_tblite(2, 0, 0, 1, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_n2_scc_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let q_ref = json_to_vec(&ref_data, "charges");
    let emo_ref = json_to_vec(&ref_data, "eigenvalues");

    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.10 * aatoau],
    ];
    let elem_idx = vec![6usize, 6]; // N = index 6
    let n_electrons = 10; // N2 has 10 electrons

    // Compare Rust-computed vs tblite-extracted multipole integrals
    let (dint_rust, qint_rust) = rust_dftb::methods::xtb::multipole_integrals::build_multipole_integrals_gfn2(&coords, &elem_idx);
    let dint_ref = json_to_vec(&ref_data, "dipole_integrals");
    let qint_ref = json_to_vec(&ref_data, "quadrupole_integrals");
    println!("Dipole integrals comparison:");
    let mut max_dint_err = 0.0f64;
    for i in 0..dint_rust.len() {
        let err = (dint_rust[i] - dint_ref[i]).abs();
        if err > max_dint_err { max_dint_err = err; }
        if err > 1e-6 && i < 30 {
            println!("  dint[{}]: rust={:.10e} ref={:.10e} err={:.10e}", i, dint_rust[i], dint_ref[i], err);
        }
    }
    println!("  max dipole integral error = {:.6e}", max_dint_err);
    println!("Quadrupole integrals comparison:");
    let mut max_qint_err = 0.0f64;
    for i in 0..qint_rust.len() {
        let err = (qint_rust[i] - qint_ref[i]).abs();
        if err > max_qint_err { max_qint_err = err; }
        if err > 1e-6 && i < 30 {
            println!("  qint[{}]: rust={:.10e} ref={:.10e} err={:.10e}", i, qint_rust[i], qint_ref[i], err);
        }
    }
    println!("  max quadrupole integral error = {:.6e}", max_qint_err);

    let (_density, qsh, emo) = rust_dftb::methods::xtb::scf::run_scf_gfn2(
        &coords, &elem_idx, n_electrons, 100, 1e-6
    );

    let nshell_per_atom: Vec<usize> = elem_idx.iter().map(|&z| rust_dftb::methods::xtb::params_gfn2::nshell[z]).collect();
    let q_rust = rust_dftb::methods::xtb::mulliken::atomic_charges(&qsh, &nshell_per_atom);

    // Also get H0 eigenvalues for diagnosis
    {
        let aatoau2 = 1.889726133;
        let coords2 = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.10 * aatoau2]];
        let elem_idx2 = vec![6usize, 6];
        let (h0, s0, _, _) = rust_dftb::methods::xtb::hamiltonian::build_h0_s(&coords2, &elem_idx2);
        let l2 = s0.clone().cholesky().unwrap().l();
        let l_inv2 = l2.clone().try_inverse().unwrap();
        let lt_inv2 = l_inv2.transpose();
        let h_prime2 = &l_inv2 * &h0 * &lt_inv2;
        let eigen2 = h_prime2.symmetric_eigen();
        let mut emo_h0: Vec<f64> = eigen2.eigenvalues.iter().cloned().collect();
        emo_h0.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("N2 H0 eigenvalues: {:?}", emo_h0);
    }
    println!("N2 charges ref: {:?}", q_ref);
    println!("N2 charges rust: {:?}", q_rust.data.as_vec());
    println!("N2 eigenvalues ref (SCC tblite): {:?}", emo_ref);
    println!("N2 eigenvalues rust (SCC): {:?}", emo.data.as_vec());

    compare_vecs(q_rust.data.as_vec(), &q_ref, "N2 charges", 1e-3);
    compare_vecs(emo.data.as_vec(), &emo_ref, "N2 eigenvalues", 1e-3);
}

#[test]
fn test_hcooh_scc_parity() {
    // Formic acid (HCOOH) approximate geometry in Angstrom
    let atoms = vec![
        (1, [-1.55,  1.10, 0.0]), // H (hydroxyl)
        (8, [-0.66,  1.10, 0.0]), // O (hydroxyl)
        (6, [ 0.00,  0.00, 0.0]), // C
        (8, [ 1.20,  0.00, 0.0]), // O (carbonyl)
        (1, [-0.36, -0.95, 0.0]), // H (on C)
    ];

    let ref_data = run_tblite(5, 0, 0, 1, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_hcooh_scc_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let q_ref = json_to_vec(&ref_data, "charges");
    let emo_ref = json_to_vec(&ref_data, "eigenvalues");

    let aatoau = 1.889726133;
    let coords: Vec<[f64; 3]> = atoms.iter().map(|(_, p)| {
        [p[0] * aatoau, p[1] * aatoau, p[2] * aatoau]
    }).collect();
    let elem_idx = vec![0usize, 7, 5, 7, 0]; // H=0, O=7, C=5
    let n_electrons = 18; // HCOOH has 18 electrons

    let (_density, qsh, emo) = rust_dftb::methods::xtb::scf::run_scf_gfn2(
        &coords, &elem_idx, n_electrons, 100, 1e-6
    );

    let nshell_per_atom: Vec<usize> = elem_idx.iter().map(|&z| rust_dftb::methods::xtb::params_gfn2::nshell[z]).collect();
    let q_rust = rust_dftb::methods::xtb::mulliken::atomic_charges(&qsh, &nshell_per_atom);

    // --- Direct Hamiltonian comparison for fixed charges ---
    let nao = 16;
    let p_tblite = json_to_dmatrix(&ref_data, "density", nao);
    let s_tblite = json_to_dmatrix(&ref_data, "overlap", nao);
    let h0_tblite = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let c_tblite = json_to_dmatrix(&ref_data, "coefficients", nao).transpose();
    let ao2sh = vec![0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9];
    let mut ang_per_shell = Vec::new();
    for &z in &elem_idx {
        let nsh = rust_dftb::methods::xtb::params_gfn2::nshell[z];
        for ish in 0..nsh {
            ang_per_shell.push(rust_dftb::methods::xtb::params_gfn2::ang_shell[z][ish]);
        }
    }
    let n0sh = rust_dftb::methods::xtb::mulliken::reference_shell_occupations_gfn2(&nshell_per_atom, &elem_idx, &ang_per_shell);
    let qsh_from_tblite = rust_dftb::methods::xtb::mulliken::shell_charges(&p_tblite, &s_tblite, &ao2sh, &n0sh);

    // Reconstruct tblite SCC Hamiltonian: H1 = S * C * E * C^T * S
    let mut e_diag = nalgebra::DMatrix::zeros(nao, nao);
    for i in 0..nao {
        e_diag[(i, i)] = emo_ref[i];
    }
    let h1_tblite = &s_tblite * &c_tblite * &e_diag * c_tblite.transpose() * &s_tblite;

    // Build our H_scc with same qsh
    let (h0_rust, s_rust, _, _) = rust_dftb::methods::xtb::hamiltonian::build_h0_s(&coords, &elem_idx);
    let ang_per_shell = vec![0, 0, 0, 1, 0, 1, 0, 1, 0, 0];
    let gamma = rust_dftb::methods::xtb::coulomb::build_coulomb_matrix(&coords, &nshell_per_atom, &elem_idx, &ang_per_shell);
    let h1_rust = rust_dftb::methods::xtb::scf::build_scc_hamiltonian_with_thirdorder(
        &h0_rust, &s_rust, &gamma, &qsh_from_tblite, &nshell_per_atom, &elem_idx, &ao2sh
    );

    println!("HCOOH qsh from tblite density: {:?}", qsh_from_tblite.data.as_vec());
    println!("HCOOH charges ref: {:?}", q_ref);
    println!("HCOOH charges rust: {:?}", q_rust.data.as_vec());
    println!("HCOOH eigenvalues ref: {:?}", emo_ref);
    println!("HCOOH eigenvalues rust: {:?}", emo.data.as_vec());

    // Verify eigenvector orthonormality and reconstruction
    let csc = c_tblite.transpose() * &s_tblite * &c_tblite;
    let mut csc_max_err = 0.0f64;
    for i in 0..nao {
        for j in 0..nao {
            let expected = if i == j { 1.0 } else { 0.0 };
            let err = (csc[(i, j)] - expected).abs();
            if err > csc_max_err { csc_max_err = err; }
        }
    }
    println!("C^T * S * C max deviation from I: {:.6e}", csc_max_err);

    // Verify H1 * c_0 = E_0 * S * c_0
    let c0 = c_tblite.column(0);
    let lhs = &h1_tblite * c0;
    let rhs = emo_ref[0] * &s_tblite * c0;
    let gevp_err = (&lhs - rhs).norm();
    println!("||H1*c0 - E0*S*c0||: {:.6e}", gevp_err);

    // Extract vao from reconstructed tblite H1: vao[i] = H0[i,i] - H1[i,i]
    let mut vao_tblite = vec![0.0f64; nao];
    for iao in 0..nao {
        vao_tblite[iao] = h0_tblite[(iao, iao)] - h1_tblite[(iao, iao)];
    }
    // Map to shells (average vao per shell)
    let mut vsh_tblite_extracted = vec![0.0f64; 10];
    let mut vsh_count = vec![0usize; 10];
    for iao in 0..nao {
        let ish = ao2sh[iao];
        vsh_tblite_extracted[ish] += vao_tblite[iao];
        vsh_count[ish] += 1;
    }
    for ish in 0..10 {
        if vsh_count[ish] > 0 {
            vsh_tblite_extracted[ish] /= vsh_count[ish] as f64;
        }
    }
    println!("vsh extracted from tblite H1: {:?}", vsh_tblite_extracted);

    // Compute our vsh for comparison
    let vsh_rust = &gamma * &qsh_from_tblite;
    let v3_rust = rust_dftb::methods::xtb::coulomb::thirdorder_potential(&qsh_from_tblite, &nshell_per_atom, &elem_idx);
    println!("vsh rust (gamma*qsh):          {:?}", vsh_rust.data.as_vec());
    println!("v3 rust (thirdorder):          {:?}", v3_rust.data.as_vec());
    println!("vao total rust per AO:         {:?}", (0..nao).map(|iao| vsh_rust[ao2sh[iao]] + v3_rust[ao2sh[iao]]).collect::<Vec<_>>());

    // Print diagonal elements for first few AOs
    println!("AO | ao2sh | H0_rust | H1_rust | H1_tblite | diff");
    for iao in 0..nao.min(8) {
        println!("{:2} | {:5} | {:9.6} | {:9.6} | {:9.6} | {:9.6}",
            iao, ao2sh[iao], h0_rust[(iao,iao)], h1_rust[(iao,iao)], h1_tblite[(iao,iao)],
            h1_rust[(iao,iao)] - h1_tblite[(iao,iao)]);
    }

    // Compare H1 matrices directly
    compare_matrices(&h1_rust, &h1_tblite, "HCOOH H1", 1e-4);
    compare_matrices(&h0_rust, &h0_tblite, "HCOOH H0", 1e-4);
    compare_matrices(&s_rust, &s_tblite, "HCOOH S", 1e-4);

    compare_vecs(q_rust.data.as_vec(), &q_ref, "HCOOH charges", 1e-3);
    compare_vecs(emo.data.as_vec(), &emo_ref, "HCOOH eigenvalues", 1e-3);
}

// ---------------------------------------------------------------------------
// GFN2 parity tests
// ---------------------------------------------------------------------------

#[test]
fn test_h2_gfn2_parity() {
    let atoms = vec![
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.74]),
    ];
    let ref_data = run_tblite(2, 0, 0, 2, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_h2_gfn2_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let nao = ref_data["nao"].as_i64().unwrap() as usize;
    println!("H2 GFN2 nao = {}", nao);

    let h_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao);

    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74 * aatoau],
    ];
    let elem_idx = vec![0usize, 0];
    let (h_rust, s_rust, _, _) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s_gfn2(&coords, &elem_idx);

    compare_matrices(&h_rust, &h_ref, "H2 GFN2 H0", 1e-4);
    compare_matrices(&s_rust, &s_ref, "H2 GFN2 S", 1e-4);
}

#[test]
fn test_n2_gfn2_parity() {
    let atoms = vec![
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, 1.10]),
    ];
    let ref_data = run_tblite(2, 0, 0, 2, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_n2_gfn2_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let nao = ref_data["nao"].as_i64().unwrap() as usize;
    println!("N2 GFN2 nao = {}", nao);

    let h_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao);

    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.10 * aatoau],
    ];
    let elem_idx = vec![6usize, 6];
    let (h_rust, s_rust, _, _) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s_gfn2(&coords, &elem_idx);

    compare_matrices(&h_rust, &h_ref, "N2 GFN2 H0", 1e-4);
    compare_matrices(&s_rust, &s_ref, "N2 GFN2 S", 1e-4);
}

#[test]
fn test_hcooh_gfn2_parity() {
    let atoms = vec![
        (1, [-1.55,  1.10, 0.0]),
        (8, [-0.66,  1.10, 0.0]),
        (6, [ 0.00,  0.00, 0.0]),
        (8, [ 1.20,  0.00, 0.0]),
        (1, [-0.36, -0.95, 0.0]),
    ];
    let ref_data = run_tblite(5, 0, 0, 2, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_hcooh_gfn2_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let nao = ref_data["nao"].as_i64().unwrap() as usize;
    println!("HCOOH GFN2 nao = {}", nao);

    let h_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao);

    let aatoau = 1.889726133;
    let coords: Vec<[f64; 3]> = atoms.iter().map(|(_, p)| {
        [p[0] * aatoau, p[1] * aatoau, p[2] * aatoau]
    }).collect();
    let elem_idx = vec![0usize, 7, 5, 7, 0];
    let (h_rust, s_rust, _, _) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s_gfn2(&coords, &elem_idx);

    compare_matrices(&h_rust, &h_ref, "HCOOH GFN2 H0", 1e-4);
    compare_matrices(&s_rust, &s_ref, "HCOOH GFN2 S", 1e-4);
}

#[test]
fn test_h2_gfn2_scc_hamiltonian_parity() {
    // Test SCC Hamiltonian with fixed shell charges from tblite
    // Using method=3 for GFN2 without D4 dispersion
    let atoms = vec![
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.74]),
    ];
    let ref_data = run_tblite(2, 0, 0, 3, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_h2_gfn2_scc_hamiltonian_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();

    // Get reference data
    let nao = ref_data["nao"].as_i64().unwrap() as usize;
    let nsh = ref_data["nsh"].as_i64().unwrap() as usize;
    let h0_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let h_scc_ref = json_to_dmatrix(&ref_data, "effective_hamiltonian", nao);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao);
    let p_ref = json_to_dmatrix(&ref_data, "density", nao);
    let qsh_ref_vec = json_to_vec(&ref_data, "shell_charges");
    let emo_ref = json_to_vec(&ref_data, "eigenvalues");

    // Parse multipole data from tblite
    let nat = ref_data["nat"].as_i64().unwrap() as usize;
    let dipole_ints_vec = json_to_vec(&ref_data, "dipole_integrals");
    let quadrupole_ints_vec = json_to_vec(&ref_data, "quadrupole_integrals");
    let dipole_pot_vec = json_to_vec(&ref_data, "dipole_potential");
    let quadrupole_pot_vec = json_to_vec(&ref_data, "quadrupole_potential");
    let charge_pot_vec = json_to_vec(&ref_data, "charge_potential");

    // Build Rust H0, S, and gamma for GFN2
    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74 * aatoau],
    ];
    let elem_idx = vec![0usize, 0];

    let (h0_rust, s_rust, _, _) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s_gfn2(&coords, &elem_idx);

    // Verify H0 and S match first
    println!("H2 GFN2 H0 comparison:");
    compare_matrices(&h0_rust, &h0_ref, "H2 GFN2 H0", 1e-4);
    println!("H2 GFN2 S comparison:");
    compare_matrices(&s_rust, &s_ref, "H2 GFN2 S", 1e-4);

    // Build nshell_per_atom and ang_per_shell from params_gfn2 (like SCF code does)
    let nshell_per_atom: Vec<usize> = elem_idx.iter().map(|&z| {
        rust_dftb::methods::xtb::params_gfn2::nshell[z]
    }).collect();

    let mut ang_per_shell = Vec::new();
    for &z in &elem_idx {
        let nsh = rust_dftb::methods::xtb::params_gfn2::nshell[z];
        for ish in 0..nsh {
            ang_per_shell.push(rust_dftb::methods::xtb::params_gfn2::ang_shell[z][ish]);
        }
    }

    // Build Coulomb matrix for GFN2
    let gamma = rust_dftb::methods::xtb::coulomb::build_coulomb_matrix_gfn2(
        &coords, &nshell_per_atom, &elem_idx, &ang_per_shell
    );

    println!("Gamma matrix shape: {}x{}", gamma.nrows(), gamma.ncols());
    println!("Gamma matrix:\n{}", gamma);
    println!("nshell_per_atom: {:?}", nshell_per_atom);
    println!("ang_per_shell: {:?}", ang_per_shell);
    println!("qsh_ref_vec len: {}, values: {:?}", qsh_ref_vec.len(), qsh_ref_vec);

    // Compute vsh = gamma * qsh manually to verify
    let mut vsh_manual = vec![0.0; qsh_ref_vec.len()];
    for i in 0..qsh_ref_vec.len() {
        for j in 0..qsh_ref_vec.len() {
            vsh_manual[i] += gamma[(i, j)] * qsh_ref_vec[j];
        }
    }
    println!("vsh_manual (should be ~0 for qsh~0): {:?}", vsh_manual);

    // Build SCC Hamiltonian using reference shell charges
    let qsh_ref = nalgebra::DVector::from_vec(qsh_ref_vec.clone());

    // Build ao2sh mapping
    let mut ao2sh = Vec::new();
    let mut ish = 0;
    for (iat, &nsh_at) in nshell_per_atom.iter().enumerate() {
        for _ in 0..nsh_at {
            let ang = ang_per_shell[ish];
            let nao_sh = 2 * ang + 1;
            for _ in 0..nao_sh {
                ao2sh.push(ish);
            }
            ish += 1;
        }
    }

    let h_scc_rust = rust_dftb::methods::xtb::scf::build_scc_hamiltonian_with_thirdorder_gfn2(
        &h0_rust, &s_rust, &gamma, &qsh_ref, &nshell_per_atom, &elem_idx, &ang_per_shell, &ao2sh
    );

    // Build ao2at mapping for multipole terms
    let mut ao2at = Vec::new();
    for (iat, &nsh_at) in nshell_per_atom.iter().enumerate() {
        for _ in 0..nsh_at {
            let ang = ang_per_shell[ao2sh[ao2at.len()]];
            let nao_sh = 2 * ang + 1;
            for _ in 0..nao_sh {
                ao2at.push(iat);
            }
        }
    }

    // Add multipole contribution to H using library function
    let mut h_scc_with_mp = h_scc_rust.clone();
    rust_dftb::methods::xtb::scf::add_multipole_to_h1(
        &mut h_scc_with_mp,
        &s_rust,
        &dipole_ints_vec,
        &quadrupole_ints_vec,
        &charge_pot_vec,
        &dipole_pot_vec,
        &quadrupole_pot_vec,
        &ao2at,
    );

    println!("H0 matrix:\n{}", h0_rust);
    println!("tblite effective Hamiltonian (H_scc):\n{}", h_scc_ref);
    println!("Rust SCC Hamiltonian matrix (charge only):\n{}", h_scc_rust);
    println!("Rust SCC Hamiltonian (with multipole):\n{}", h_scc_with_mp);
    println!("Difference (tblite H_scc - Rust H_scc with mp):\n{}", &h_scc_ref - &h_scc_with_mp);

    // Compute atomic multipoles from density matrix for sanity check
    let dpat_rust = rust_dftb::methods::xtb::mulliken::atomic_multipoles(
        &p_ref, &dipole_ints_vec, &ao2at, 3
    );
    let qpat_rust = rust_dftb::methods::xtb::mulliken::atomic_multipoles(
        &p_ref, &quadrupole_ints_vec, &ao2at, 6
    );
    println!("Rust atomic dipole moments:\n{}", dpat_rust);
    println!("Rust atomic quadrupole moments:\n{}", qpat_rust);
    // Total molecular dipole should be ~0 for H2
    let mut total_dipole = vec![0.0f64; 3];
    for iat in 0..nat {
        for cmp in 0..3 {
            total_dipole[cmp] += dpat_rust[(cmp, iat)];
        }
    }
    println!("Total molecular dipole: {:?}", total_dipole);

    // Direct comparison of effective Hamiltonian matrices
    println!("H2 GFN2 effective Hamiltonian comparison (with multipole):");
    compare_matrices(&h_scc_with_mp, &h_scc_ref, "H2 GFN2 SCC Hamiltonian with multipole", 1e-4);

    // First, check what H0 eigenvalues should be (standard eigenvalue problem)
    let h0_eigen = h0_rust.clone().symmetric_eigen();
    let mut emo_h0: Vec<f64> = h0_eigen.eigenvalues.iter().copied().collect();
    emo_h0.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("H2 GFN2 H0 eigenvalues (standard): {:?}", emo_h0);

    // The reference eigenvalues are from generalized eigenvalue problem H0 * c = E * S * c
    // Let's compute those using the cholesky method
    let chol = s_rust.clone().cholesky().unwrap();
    let l = chol.l();
    let l_inv = l.clone().try_inverse().unwrap();
    let l_t_inv = l_inv.transpose();
    let h0_prime = &l_inv * &h0_rust * &l_t_inv;
    let h0_gen_eigen = h0_prime.symmetric_eigen();
    let mut emo_h0_gen: Vec<f64> = h0_gen_eigen.eigenvalues.iter().copied().collect();
    emo_h0_gen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("H2 GFN2 H0 eigenvalues (generalized): {:?}", emo_h0_gen);

    // Diagonalize SCC Hamiltonian and compare eigenvalues with reference
    let chol_scc = s_rust.clone().cholesky().unwrap();
    let l_scc = chol_scc.l();
    let l_inv_scc = l_scc.clone().try_inverse().unwrap();
    let l_t_inv_scc = l_inv_scc.transpose();
    let h_scc_prime = &l_inv_scc * &h_scc_with_mp * &l_t_inv_scc;
    let eigen = h_scc_prime.symmetric_eigen();
    let mut emo_rust: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    emo_rust.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("H2 GFN2 SCC Hamiltonian eigenvalues ref: {:?}", emo_ref);
    println!("H2 GFN2 SCC Hamiltonian eigenvalues rust: {:?}", emo_rust);

    // For small charges, SCC eigenvalues should be close to H0 eigenvalues
    println!("Difference from H0 (ref):  [{}, {}]",
             emo_ref[0] - emo_h0_gen[0], emo_ref[1] - emo_h0_gen[1]);
    println!("Difference from H0 (rust): [{}, {}]",
             emo_rust[0] - emo_h0_gen[0], emo_rust[1] - emo_h0_gen[1]);

    compare_vecs(&emo_rust, &emo_ref, "H2 GFN2 SCC eigenvalues", 1e-3);
}

#[test]
fn test_n2_gfn2_scc_parity() {
    let atoms = vec![
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, 1.10]),
    ];
    let ref_data = run_tblite(2, 0, 0, 2, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_n2_gfn2_scc_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let q_ref = json_to_vec(&ref_data, "charges");
    let emo_ref = json_to_vec(&ref_data, "eigenvalues");
    let nao_ref = ref_data["nao"].as_i64().unwrap() as usize;
    let h0_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao_ref);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao_ref);
    let h_scc_ref = json_to_dmatrix(&ref_data, "effective_hamiltonian", nao_ref);

    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.10 * aatoau],
    ];
    let elem_idx = vec![6usize, 6];
    let n_electrons = 10;
    let nat = coords.len();

    let (density, qsh, emo) = rust_dftb::methods::xtb::scf::run_scf_gfn2(
        &coords, &elem_idx, n_electrons, 100, 1e-6
    );

    let nshell_per_atom = vec![2, 2]; // GFN2: N has 2 shells
    let q_rust = rust_dftb::methods::xtb::mulliken::atomic_charges(&qsh, &nshell_per_atom);

    println!("N2 GFN2 charges ref: {:?}", q_ref);
    println!("N2 GFN2 charges rust: {:?}", q_rust.data.as_vec());
    println!("N2 GFN2 eigenvalues ref: {:?}", emo_ref);
    println!("N2 GFN2 eigenvalues rust: {:?}", emo.data.as_vec());

    // Also compare H0, S, and effective Hamiltonian matrices
    let (h0_rust, s_rust, _, _) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s_gfn2(&coords, &elem_idx);
    println!("N2 GFN2 H0 comparison:");
    compare_matrices(&h0_rust, &h0_ref, "N2 GFN2 H0", 1e-4);
    println!("N2 GFN2 S comparison:");
    compare_matrices(&s_rust, &s_ref, "N2 GFN2 S", 1e-4);

    // Build effective Hamiltonian with reference charges for direct comparison
    let nshell_per_atom = vec![2, 2];
    let mut ang_per_shell = Vec::new();
    for &z in &elem_idx {
        let nsh = rust_dftb::methods::xtb::params_gfn2::nshell[z];
        for ish in 0..nsh {
            ang_per_shell.push(rust_dftb::methods::xtb::params_gfn2::ang_shell[z][ish]);
        }
    }
    let mut ao2sh = Vec::new();
    let mut ish = 0;
    for (iat, &nsh_at) in nshell_per_atom.iter().enumerate() {
        for _ in 0..nsh_at {
            let ang = ang_per_shell[ish];
            let nao_sh = 2 * ang + 1;
            for _ in 0..nao_sh {
                ao2sh.push(ish);
            }
            ish += 1;
        }
    }
    let qsh_ref = nalgebra::DVector::from_vec(qsh.data.as_vec().clone());
    let gamma = rust_dftb::methods::xtb::coulomb::build_coulomb_matrix_gfn2(
        &coords, &nshell_per_atom, &elem_idx, &ang_per_shell
    );
    let mut h_scc_rust = rust_dftb::methods::xtb::scf::build_scc_hamiltonian_with_thirdorder_gfn2(
        &h0_rust, &s_rust, &gamma, &qsh, &nshell_per_atom, &elem_idx, &ang_per_shell, &ao2sh
    );

    // Add multipole terms using Rust converged density for fair comparison
    let mut ao2at = Vec::new();
    let mut ish = 0;
    for (iat, &nsh_at) in nshell_per_atom.iter().enumerate() {
        for _ in 0..nsh_at {
            let ang = ang_per_shell[ish];
            let nao_sh = 2 * ang + 1;
            for _ in 0..nao_sh {
                ao2at.push(iat);
            }
            ish += 1;
        }
    }
    let (dipole_ints, quadrupole_ints) = rust_dftb::methods::xtb::multipole_integrals::build_multipole_integrals_gfn2(&coords, &elem_idx);
    let cn = rust_dftb::methods::xtb::scf::compute_coordination_numbers(&coords, &elem_idx);
    let mrad = rust_dftb::methods::xtb::scf::compute_multipole_radii(&cn, &elem_idx);
    let (amat_sd, amat_dd, amat_sq) = rust_dftb::methods::xtb::scf::build_multipole_interaction_matrices_0d(&coords, &mrad);
    let dpat_mat = rust_dftb::methods::xtb::mulliken::atomic_multipoles(&density, &dipole_ints, &ao2at, 3);
    let qpat_mat = rust_dftb::methods::xtb::mulliken::atomic_multipoles(&density, &quadrupole_ints, &ao2at, 6);
    let mut dpat = vec![0.0f64; 3 * nat];
    let mut qpat = vec![0.0f64; 6 * nat];
    for iat in 0..nat {
        for cmp in 0..3 { dpat[cmp + 3 * iat] = dpat_mat[(cmp, iat)]; }
        for cmp in 0..6 { qpat[cmp + 6 * iat] = qpat_mat[(cmp, iat)]; }
    }
    let qat_vec = rust_dftb::methods::xtb::mulliken::atomic_charges(&qsh_ref, &nshell_per_atom);
    let mut qat = vec![0.0f64; nat];
    for iat in 0..nat { qat[iat] = qat_vec[iat]; }
    let (vat_computed, vdp, vqp) = rust_dftb::methods::xtb::scf::compute_multipole_potentials(
        &qat, &dpat, &qpat, &amat_sd, &amat_dd, &amat_sq, &elem_idx
    );
    // Use reference vat (includes D4 dispersion charge-dependent term missing from vat_computed)
    let vat_ref_raw = json_to_vec(&ref_data, "charge_potential");
    let vat: Vec<f64> = (0..nat).map(|i| vat_ref_raw[i]).collect();
    println!("N2 D4 dispersion contribution to vat[0]: {:.9e}", vat[0] - vat_computed[0]);
    rust_dftb::methods::xtb::scf::add_multipole_to_h1(&mut h_scc_rust, &s_rust, &dipole_ints, &quadrupole_ints, &vat, &vdp, &vqp, &ao2at);

    // Diagnostic: solve GEVP with reference H and S to check solver accuracy
    let (emo_from_ref, _) = rust_dftb::methods::xtb::scf::solve_gevp(&h_scc_ref, &s_ref);
    let mut max_err_solver = 0.0f64;
    for i in 0..emo_from_ref.len() {
        let err = (emo_from_ref[i] - emo_ref[i]).abs();
        if err > max_err_solver { max_err_solver = err; }
    }
    println!("N2 GFN2 eigenvalue solver error from ref H: max_err = {:.6e}", max_err_solver);

    // Diagnostic: print shell charges
    println!("N2 GFN2 shell charges ref: {:?}", qsh_ref.data.as_vec());
    println!("N2 GFN2 shell charges rust: {:?}", qsh.data.as_vec());

    // Diagnostic: compare charge-only Hamiltonian with H0
    let h_charge_only = rust_dftb::methods::xtb::scf::build_scc_hamiltonian_with_thirdorder_gfn2(
        &h0_rust, &s_rust, &gamma, &qsh_ref, &nshell_per_atom, &elem_idx, &ang_per_shell, &ao2sh
    );
    let mut max_h0_diff = 0.0f64;
    for i in 0..h0_rust.nrows() {
        for j in 0..h0_rust.ncols() {
            let diff = (h_charge_only[(i,j)] - h0_rust[(i,j)]).abs();
            if diff > max_h0_diff { max_h0_diff = diff; }
        }
    }
    println!("N2 GFN2 charge-only H - H0 max diff: {:.6e}", max_h0_diff);

    // Diagnostic: compare multipole potentials directly (before compare_matrices panics)
    let vat_ref = json_to_vec(&ref_data, "charge_potential");
    let vdp_ref = json_to_vec(&ref_data, "dipole_potential");
    let vqp_ref = json_to_vec(&ref_data, "quadrupole_potential");
    println!("N2 GFN2 charge potential ref: {:?}", vat_ref);
    println!("N2 GFN2 dipole potential ref: {:?}", vdp_ref);
    println!("N2 GFN2 quadrupole potential ref: {:?}", vqp_ref);

    // Compute Rust potentials for comparison
    let (dipole_ints, quadrupole_ints) = rust_dftb::methods::xtb::multipole_integrals::build_multipole_integrals_gfn2(&coords, &elem_idx);
    let dpat_mat = rust_dftb::methods::xtb::mulliken::atomic_multipoles(&density, &dipole_ints, &ao2at, 3);
    let qpat_mat = rust_dftb::methods::xtb::mulliken::atomic_multipoles(&density, &quadrupole_ints, &ao2at, 6);
    let mut dpat = vec![0.0f64; 3 * nat];
    let mut qpat = vec![0.0f64; 6 * nat];
    for iat in 0..nat {
        for cmp in 0..3 { dpat[cmp + 3 * iat] = dpat_mat[(cmp, iat)]; }
        for cmp in 0..6 { qpat[cmp + 6 * iat] = qpat_mat[(cmp, iat)]; }
    }
    let qat_vec = rust_dftb::methods::xtb::mulliken::atomic_charges(&qsh, &nshell_per_atom);
    let mut qat = vec![0.0f64; nat];
    for iat in 0..nat { qat[iat] = qat_vec[iat]; }
    let cn = rust_dftb::methods::xtb::scf::compute_coordination_numbers(&coords, &elem_idx);
    let mrad = rust_dftb::methods::xtb::scf::compute_multipole_radii(&cn, &elem_idx);
    println!("N2 mrad: {:?}", mrad);
    println!("N2 cn:   {:?}", cn);
    let (amat_sd, amat_dd, amat_sq) = rust_dftb::methods::xtb::scf::build_multipole_interaction_matrices_0d(&coords, &mrad);
    // Diagnostic: print specific amat_sq and amat_sd elements
    println!("N2 amat_sd[z,0,0] = {:.9e}", amat_sd[2 + 3*0 + 3*nat*0]);
    println!("N2 amat_sd[z,1,0] = {:.9e}", amat_sd[2 + 3*1 + 3*nat*0]);
    println!("N2 amat_sd[z,0,1] = {:.9e}", amat_sd[2 + 3*0 + 3*nat*1]);
    println!("N2 amat_sd[x,0,1] = {:.9e}", amat_sd[0 + 3*0 + 3*nat*1]);
    println!("N2 amat_sd[y,0,1] = {:.9e}", amat_sd[1 + 3*0 + 3*nat*1]);
    println!("N2 amat_sd[x,1,0] = {:.9e}", amat_sd[0 + 3*1 + 3*nat*0]);
    println!("N2 amat_sd[y,1,0] = {:.9e}", amat_sd[1 + 3*1 + 3*nat*0]);
    println!("N2 amat_sq[zz,0,0] = {:.9e}", amat_sq[5 + 6*0 + 6*nat*0]);
    println!("N2 amat_sq[zz,1,0] = {:.9e}", amat_sq[5 + 6*1 + 6*nat*0]);
    println!("N2 amat_sq[zz,0,1] = {:.9e}", amat_sq[5 + 6*0 + 6*nat*1]);
    println!("N2 amat_sq[zz,1,1] = {:.9e}", amat_sq[5 + 6*1 + 6*nat*1]);
    // Compute reference dpat/qpat from reference density
    let density_ref = json_to_dmatrix(&ref_data, "density", nao_ref);
    let dpat_ref_mat = rust_dftb::methods::xtb::mulliken::atomic_multipoles(&density_ref, &dipole_ints, &ao2at, 3);
    let qpat_ref_mat = rust_dftb::methods::xtb::mulliken::atomic_multipoles(&density_ref, &quadrupole_ints, &ao2at, 6);
    let mut dpat_ref = vec![0.0f64; 3 * nat];
    let mut qpat_ref = vec![0.0f64; 6 * nat];
    for iat in 0..nat {
        for cmp in 0..3 { dpat_ref[cmp + 3 * iat] = dpat_ref_mat[(cmp, iat)]; }
        for cmp in 0..6 { qpat_ref[cmp + 6 * iat] = qpat_ref_mat[(cmp, iat)]; }
    }

    // Compare dipole/quadrupole integrals directly
    let dint_ref = json_to_vec(&ref_data, "dipole_integrals");
    let qint_ref = json_to_vec(&ref_data, "quadrupole_integrals");
    println!("N2 dipole int [z,0,4] rust={:.9e} ref={:.9e}", dipole_ints[2 + 3*0 + 3*8*4], dint_ref[2 + 3*0 + 3*8*4]);
    println!("N2 dipole int [z,4,0] rust={:.9e} ref={:.9e}", dipole_ints[2 + 3*4 + 3*8*0], dint_ref[2 + 3*4 + 3*8*0]);
    println!("N2 quadrupole int [zz,0,4] rust={:.9e} ref={:.9e}", quadrupole_ints[5 + 6*0 + 6*8*4], qint_ref[5 + 6*0 + 6*8*4]);

    println!("N2 GFN2 dpat rust: {:?}", dpat);
    println!("N2 GFN2 dpat ref:  {:?}", dpat_ref);
    println!("N2 GFN2 qpat rust: {:?}", qpat);
    println!("N2 GFN2 qpat ref:  {:?}", qpat_ref);

    // Manual computation of vat_sd from reference dpat
    let mut vat_sd_manual = vec![0.0f64; nat];
    for iat in 0..nat {
        for jat in 0..nat {
            for cmp in 0..3 {
                let sd = amat_sd[cmp + 3 * jat + 3 * nat * iat];
                vat_sd_manual[iat] += sd * dpat_ref[cmp + 3 * jat];
            }
        }
    }
    println!("N2 manual vat_sd from ref dpat: {:?}", vat_sd_manual);

    // Manual computation of vat_sq from reference qpat
    let mut vat_sq_manual = vec![0.0f64; nat];
    for iat in 0..nat {
        for jat in 0..nat {
            for cmp in 0..6 {
                let sq = amat_sq[cmp + 6 * jat + 6 * nat * iat];
                vat_sq_manual[iat] += sq * qpat_ref[cmp + 6 * jat];
            }
        }
    }
    println!("N2 manual vat_sq from ref qpat: {:?}", vat_sq_manual);
    println!("N2 manual vat total (sd+sq): {:?}", vec![vat_sd_manual[0] + vat_sq_manual[0], vat_sd_manual[1] + vat_sq_manual[1]]);

    // Compute Rust potentials using REFERENCE multipole moments for fair comparison
    let qat_ref_vec = json_to_vec(&ref_data, "charges");
    let mut qat_ref = vec![0.0f64; nat];
    for iat in 0..nat { qat_ref[iat] = qat_ref_vec[iat]; }
    let (vat_rust_ref_mom, vdp_rust_ref_mom, vqp_rust_ref_mom) = rust_dftb::methods::xtb::scf::compute_multipole_potentials(
        &qat_ref, &dpat_ref, &qpat_ref, &amat_sd, &amat_dd, &amat_sq, &elem_idx
    );
    println!("N2 GFN2 charge potential rust(ref_mom): {:?}", vat_rust_ref_mom);
    println!("N2 GFN2 dipole potential rust(ref_mom):  {:?}", vdp_rust_ref_mom);
    println!("N2 GFN2 quadrupole potential rust(ref_mom):{:?}", vqp_rust_ref_mom);

    // Also compute with Rust moments for comparison
    let (vat_rust, vdp_rust, vqp_rust) = rust_dftb::methods::xtb::scf::compute_multipole_potentials(
        &qat, &dpat, &qpat, &amat_sd, &amat_dd, &amat_sq, &elem_idx
    );
    println!("N2 GFN2 charge potential rust: {:?}", vat_rust);
    println!("N2 GFN2 dipole potential rust: {:?}", vdp_rust);
    println!("N2 GFN2 quadrupole potential rust: {:?}", vqp_rust);

    println!("N2 GFN2 effective Hamiltonian comparison:");
    compare_matrices(&h_scc_rust, &h_scc_ref, "N2 GFN2 effective Hamiltonian", 1e-4);

    compare_vecs(q_rust.data.as_vec(), &q_ref, "N2 GFN2 charges", 1e-3);
    compare_vecs(emo.data.as_vec(), &emo_ref, "N2 GFN2 eigenvalues", 1e-3);
}

#[test]
fn test_n_dipole_integrals_parity() {
    // Test dipole integrals for a single N atom (has s and p orbitals)
    // Place two N atoms at 2.0 Bohr to get non-zero off-diagonal p-p integrals
    let bond_length_bohr = 2.0;
    let bond_length_ang = bond_length_bohr / 1.889726133;
    let atoms_bohr = vec![
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, bond_length_bohr]),
    ];
    let atoms_angstrom = vec![
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, bond_length_ang]),
    ];

    let ref_data = run_tblite(2, 0, 0, 2, &atoms_angstrom);
    if ref_data.is_none() {
        println!("Skipping test_n_dipole_integrals_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();

    let dint_ref = ref_data["dipole_integrals"].as_array().unwrap();
    let qint_ref = ref_data["quadrupole_integrals"].as_array().unwrap();
    let smat_ref = ref_data["overlap"].as_array().unwrap();
    let nao = ref_data["nao"].as_i64().unwrap() as usize;
    println!("tblite nao for N2 = {}", nao);

    println!("N2 dipole integrals ref (3 x 5 x 5):");
    for cmp in 0..3 {
        println!("  Component {}: {:?}", cmp, &dint_ref[cmp*nao*nao..(cmp+1)*nao*nao]);
    }

    println!("N2 quadrupole integrals ref (6 x 5 x 5):");
    for cmp in 0..6 {
        println!("  Component {}: {:?}", cmp, &qint_ref[cmp*nao*nao..(cmp+1)*nao*nao]);
    }

    println!("N2 overlap ref (5 x 5): {:?}", &smat_ref[0..nao*nao]);

    // Compute Rust integrals
    let coords: Vec<[f64; 3]> = atoms_bohr.iter().map(|(_, p)| {
        [p[0], p[1], p[2]]
    }).collect();
    let elem_idx = vec![6usize, 6]; // N

    let (dipole_rust, quadrupole_rust) = rust_dftb::methods::xtb::multipole_integrals::build_multipole_integrals_gfn2(
        &coords, &elem_idx
    );
    let overlap_rust = rust_dftb::methods::xtb::multipole_integrals::build_overlap_gfn2(&coords, &elem_idx);

    println!("N2 dipole integrals rust (3 x 5 x 5):");
    for cmp in 0..3 {
        println!("  Component {}: {:?}", cmp, &dipole_rust[cmp*nao*nao..(cmp+1)*nao*nao]);
    }

    println!("N2 quadrupole integrals rust (6 x 5 x 5):");
    for cmp in 0..6 {
        println!("  Component {}: {:?}", cmp, &quadrupole_rust[cmp*nao*nao..(cmp+1)*nao*nao]);
    }

    println!("N2 overlap rust (5 x 5): {:?}", &overlap_rust[0..nao*nao]);

    // Compare element-wise
    let mut max_dipole_error = 0.0;
    for cmp in 0..3 {
        for i in 0..nao {
            for j in 0..nao {
                let idx = cmp + 3 * (j + nao * i);
                let ref_val = dint_ref[idx].as_f64().unwrap();
                let rust_val = dipole_rust[idx];
                let error = (ref_val - rust_val).abs();
                if error > max_dipole_error {
                    max_dipole_error = error;
                }
                if error > 1e-5 {
                    println!("  DIPOLE MISMATCH: dipole[{},{}][{}] = ref: {:.10e}, rust: {:.10e}, err: {:.2e}",
                             i, j, cmp, ref_val, rust_val, error);
                }
            }
        }
    }

    let mut max_quad_error = 0.0;
    for cmp in 0..6 {
        for i in 0..nao {
            for j in 0..nao {
                let idx = cmp + 6 * (j + nao * i);
                let ref_val = qint_ref[idx].as_f64().unwrap();
                let rust_val = quadrupole_rust[idx];
                let error = (ref_val - rust_val).abs();
                if error > max_quad_error {
                    max_quad_error = error;
                }
                if error > 1e-5 {
                    println!("  QUAD MISMATCH: quad[{},{}][{}] = ref: {:.10e}, rust: {:.10e}, err: {:.2e}",
                             i, j, cmp, ref_val, rust_val, error);
                }
            }
        }
    }

    let mut max_overlap_error = 0.0;
    for i in 0..nao {
        for j in 0..nao {
            let idx = j + nao * i;
            let ref_val = smat_ref[idx].as_f64().unwrap();
            let rust_val = overlap_rust[idx];
            let error = (ref_val - rust_val).abs();
            if error > max_overlap_error {
                max_overlap_error = error;
            }
            if error > 1e-5 {
                println!("  OVERLAP MISMATCH: overlap[{},{}] = ref: {:.10e}, rust: {:.10e}, err: {:.2e}",
                         i, j, ref_val, rust_val, error);
            }
        }
    }

    println!("Max dipole integral error: {:.2e}", max_dipole_error);
    println!("Max quadrupole integral error: {:.2e}", max_quad_error);
    println!("Max overlap error: {:.2e}", max_overlap_error);

    assert!(max_dipole_error < 1e-5, "Dipole integral error too large: {:.2e}", max_dipole_error);
    assert!(max_quad_error < 1e-5, "Quadrupole integral error too large: {:.2e}", max_quad_error);
    assert!(max_overlap_error < 1e-5, "Overlap error too large: {:.2e}", max_overlap_error);
}

#[test]
fn test_hcooh_gfn2_scc_parity() {
    let atoms = vec![
        (1, [-1.55,  1.10, 0.0]),
        (8, [-0.66,  1.10, 0.0]),
        (6, [ 0.00,  0.00, 0.0]),
        (8, [ 1.20,  0.00, 0.0]),
        (1, [-0.36, -0.95, 0.0]),
    ];
    let ref_data = run_tblite(5, 0, 0, 2, &atoms);
    if ref_data.is_none() {
        println!("Skipping test_hcooh_gfn2_scc_parity: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();
    let q_ref = json_to_vec(&ref_data, "charges");
    let emo_ref = json_to_vec(&ref_data, "eigenvalues");
    let nao_ref = ref_data["nao"].as_i64().unwrap() as usize;
    let h0_ref = json_to_dmatrix(&ref_data, "hamiltonian", nao_ref);
    let s_ref = json_to_dmatrix(&ref_data, "overlap", nao_ref);
    let h_scc_ref = json_to_dmatrix(&ref_data, "effective_hamiltonian", nao_ref);
    let qsh_ref_vec = json_to_vec(&ref_data, "shell_charges");

    let aatoau = 1.889726133;
    let coords: Vec<[f64; 3]> = atoms.iter().map(|(_, p)| {
        [p[0] * aatoau, p[1] * aatoau, p[2] * aatoau]
    }).collect();
    let elem_idx = vec![0usize, 7, 5, 7, 0];
    let n_electrons = 18;
    let nat = coords.len();

    let (_density, qsh, emo) = rust_dftb::methods::xtb::scf::run_scf_gfn2(
        &coords, &elem_idx, n_electrons, 100, 1e-6
    );

    let nshell_per_atom = vec![1, 2, 2, 2, 1]; // GFN2: H=1, O=2, C=2, O=2, H=1
    let q_rust = rust_dftb::methods::xtb::mulliken::atomic_charges(&qsh, &nshell_per_atom);

    println!("HCOOH GFN2 charges ref: {:?}", q_ref);
    println!("HCOOH GFN2 charges rust: {:?}", q_rust.data.as_vec());
    println!("HCOOH GFN2 eigenvalues ref: {:?}", emo_ref);
    println!("HCOOH GFN2 eigenvalues rust: {:?}", emo.data.as_vec());

    // FIXED-CHARGE HAMILTONIAN COMPARISON
    // Build Rust H0 and S
    let (h0_rust, s_rust, _, _) =
        rust_dftb::methods::xtb::hamiltonian::build_h0_s_gfn2(&coords, &elem_idx);
    println!("HCOOH GFN2 H0 comparison:");
    compare_matrices(&h0_rust, &h0_ref, "HCOOH GFN2 H0", 1e-4);
    println!("HCOOH GFN2 S comparison:");
    compare_matrices(&s_rust, &s_ref, "HCOOH GFN2 S", 1e-4);

    // Build Rust effective Hamiltonian with REFERENCE shell charges
    let mut ang_per_shell = Vec::new();
    for &z in &elem_idx {
        let nsh = rust_dftb::methods::xtb::params_gfn2::nshell[z];
        for ish in 0..nsh {
            ang_per_shell.push(rust_dftb::methods::xtb::params_gfn2::ang_shell[z][ish]);
        }
    }
    let mut ao2sh = Vec::new();
    let mut ish = 0;
    for (iat, &nsh_at) in nshell_per_atom.iter().enumerate() {
        for _ in 0..nsh_at {
            let ang = ang_per_shell[ish];
            let nao_sh = 2 * ang + 1;
            for _ in 0..nao_sh {
                ao2sh.push(ish);
            }
            ish += 1;
        }
    }
    let qsh_ref = nalgebra::DVector::from_vec(qsh_ref_vec.clone());
    let gamma = rust_dftb::methods::xtb::coulomb::build_coulomb_matrix_gfn2(
        &coords, &nshell_per_atom, &elem_idx, &ang_per_shell
    );

    // Diagnostic: print specific gamma elements for comparison
    println!("HCOOH gamma[0,0] = {:.9e}", gamma[(0,0)]);
    println!("HCOOH gamma[0,1] = {:.9e}", gamma[(0,1)]);
    println!("HCOOH gamma[1,0] = {:.9e}", gamma[(1,0)]);

    // Diagnostic: compute shell-resolved potential vsh = gamma * qsh + third_order
    let nshell = qsh_ref.len();
    let mut vsh_rust = nalgebra::DVector::zeros(nshell);
    for ish in 0..nshell {
        for jsh in 0..nshell {
            vsh_rust[ish] += gamma[(ish, jsh)] * qsh_ref[jsh];
        }
    }
    let v3 = rust_dftb::methods::xtb::coulomb::thirdorder_potential_gfn2(
        &qsh_ref, &nshell_per_atom, &elem_idx, &ang_per_shell
    );
    for ish in 0..nshell {
        vsh_rust[ish] += v3[ish];
    }

    // Compare with reference vsh if available
    if let Some(vsh_ref_val) = ref_data.get("vsh") {
        let vsh_ref_vec = vsh_ref_val.as_array().unwrap();
        let vsh_ref = nalgebra::DVector::from_vec(vsh_ref_vec.iter().map(|x| x.as_f64().unwrap()).collect());
        println!("HCOOH vsh comparison:");
        compare_vecs(vsh_rust.data.as_vec(), vsh_ref.data.as_vec(), "HCOOH vsh", 1e-3);
    }

    let h_charge_rust = rust_dftb::methods::xtb::scf::build_scc_hamiltonian_with_thirdorder_gfn2(
        &h0_rust, &s_rust, &gamma, &qsh_ref, &nshell_per_atom, &elem_idx, &ang_per_shell, &ao2sh
    );

    // Diagnostic: charge-only difference (reference effective - reference H0 vs rust charge - rust H0)
    let mut max_charge_diff = 0.0f64;
    for i in 0..h0_rust.nrows() {
        for j in 0..h0_rust.ncols() {
            let diff = (h_charge_rust[(i,j)] - h0_rust[(i,j)] - (h_scc_ref[(i,j)] - h0_ref[(i,j)])).abs();
            if diff > max_charge_diff { max_charge_diff = diff; }
        }
    }
    println!("HCOOH charge-only term max diff: {:.6e}", max_charge_diff);

    let mut h_scc_rust = h_charge_rust.clone();

    // Add multipole terms for fair comparison with tblite (which includes them in effective Hamiltonian)
    let mut ao2at = Vec::new();
    let mut ish = 0;
    for (iat, &nsh_at) in nshell_per_atom.iter().enumerate() {
        for _ in 0..nsh_at {
            let ang = ang_per_shell[ish];
            let nao_sh = 2 * ang + 1;
            for _ in 0..nao_sh {
                ao2at.push(iat);
            }
            ish += 1;
        }
    }
    let (dipole_ints, quadrupole_ints) = rust_dftb::methods::xtb::multipole_integrals::build_multipole_integrals_gfn2(&coords, &elem_idx);
    let cn = rust_dftb::methods::xtb::scf::compute_coordination_numbers(&coords, &elem_idx);
    let mrad = rust_dftb::methods::xtb::scf::compute_multipole_radii(&cn, &elem_idx);
    let (amat_sd, amat_dd, amat_sq) = rust_dftb::methods::xtb::scf::build_multipole_interaction_matrices_0d(&coords, &mrad);
    // Use reference density from tblite for multipole terms
    let density_ref = json_to_dmatrix(&ref_data, "density", nao_ref);
    let dpat_mat = rust_dftb::methods::xtb::mulliken::atomic_multipoles(&density_ref, &dipole_ints, &ao2at, 3);
    let qpat_mat = rust_dftb::methods::xtb::mulliken::atomic_multipoles(&density_ref, &quadrupole_ints, &ao2at, 6);
    let mut dpat = vec![0.0f64; 3 * nat];
    let mut qpat = vec![0.0f64; 6 * nat];
    for iat in 0..nat {
        for cmp in 0..3 { dpat[cmp + 3 * iat] = dpat_mat[(cmp, iat)]; }
        for cmp in 0..6 { qpat[cmp + 6 * iat] = qpat_mat[(cmp, iat)]; }
    }
    let qat_ref_vec = json_to_vec(&ref_data, "charges");
    let mut qat = vec![0.0f64; nat];
    for iat in 0..nat { qat[iat] = qat_ref_vec[iat]; }
    let (vat_computed, vdp, vqp) = rust_dftb::methods::xtb::scf::compute_multipole_potentials(
        &qat, &dpat, &qpat, &amat_sd, &amat_dd, &amat_sq, &elem_idx
    );
    // Use reference vat (includes D4 dispersion charge-dependent term)
    let vat_ref_raw = json_to_vec(&ref_data, "charge_potential");
    let vat: Vec<f64> = (0..nat).map(|i| vat_ref_raw[i]).collect();
    println!("HCOOH D4 dispersion contribution to vat[0]: {:.9e}", vat[0] - vat_computed[0]);
    rust_dftb::methods::xtb::scf::add_multipole_to_h1(&mut h_scc_rust, &s_rust, &dipole_ints, &quadrupole_ints, &vat, &vdp, &vqp, &ao2at);

    println!("HCOOH GFN2 effective Hamiltonian comparison (fixed charges+density):");
    // Print specific element (9, 12) before compare_matrices panics
    println!("HCOOH H_scc ref[9,12] = {:.16e}, rust[9,12] = {:.16e}", h_scc_ref[(9, 12)], h_scc_rust[(9, 12)]);
    compare_matrices(&h_scc_rust, &h_scc_ref, "HCOOH GFN2 effective Hamiltonian", 1e-4);

    println!("HCOOH shell charges ref: {:?}", qsh_ref_vec);
    compare_vecs(q_rust.data.as_vec(), &q_ref, "HCOOH GFN2 charges", 1e-3);
    compare_vecs(emo.data.as_vec(), &emo_ref, "HCOOH GFN2 eigenvalues", 1e-3);
}

#[test]
fn test_single_dipole_integral_h2() {
    // Test single dipole integral for H2 s-s shell pair
    // H2 at 0.74 Å bond length (1.4 Bohr)
    let bond_length_bohr = 1.4; // Bohr
    let bond_length_ang = bond_length_bohr / 1.889726133; // Angstrom (C helper expects Angstrom)
    let atoms_bohr = vec![
        (1, [0.0, 0.0, 0.0]),
        (1, [bond_length_bohr, 0.0, 0.0]),
    ];
    let atoms_angstrom = vec![
        (1, [0.0, 0.0, 0.0]),
        (1, [bond_length_ang, 0.0, 0.0]),
    ];

    let ref_data = run_tblite(2, 0, 0, 2, &atoms_angstrom);
    if ref_data.is_none() {
        println!("Skipping test_single_dipole_integral_h2: tblite_helper not available");
        return;
    }
    let ref_data = ref_data.unwrap();

    // Get full dipole integrals from tblite (3 x nao x nao)
    let dint_ref = ref_data["dipole_integrals"].as_array().unwrap();
    let nao = 2; // H2 has 2 AOs (1s on each H)

    println!("Tblite dipole integrals (3 x 2 x 2):");
    for cmp in 0..3 {
        println!("  Component {}: {:?}", cmp, &dint_ref[cmp*nao*nao..(cmp+1)*nao*nao]);
    }

    // Also get overlap from tblite for comparison
    let smat_ref = ref_data["overlap"].as_array().unwrap();
    println!("Tblite overlap (2 x 2): {:?}", &smat_ref[0..nao*nao]);

    // Compute Rust dipole integrals (coords in Bohr)
    let coords: Vec<[f64; 3]> = atoms_bohr.iter().map(|(_, p)| {
        [p[0], p[1], p[2]]
    }).collect();
    let elem_idx = vec![0usize, 0]; // Both H

    // Debug: print CGTO parameters
    let cgtos = rust_dftb::methods::xtb::multipole_integrals::build_cgto_basis(&elem_idx);
    println!("CGTO basis for H2:");
    for (i, cgto) in cgtos.iter().enumerate() {
        println!("  Shell {}: ang={}, nprim={}", i, cgto.ang, cgto.nprim);
        for j in 0..cgto.nprim {
            println!("    Prim {}: alpha={:.10e}, coeff={:.10e}", j, cgto.alpha[j], cgto.coeff[j]);
        }
    }

    let (dipole_rust, _quadrupole_rust) = rust_dftb::methods::xtb::multipole_integrals::build_multipole_integrals_gfn2(
        &coords, &elem_idx
    );

    // Build overlap from Rust for comparison
    let overlap_rust = rust_dftb::methods::xtb::multipole_integrals::build_overlap_gfn2(&coords, &elem_idx);
    println!("Rust overlap (2 x 2): {:?}", &overlap_rust[0..nao*nao]);

    println!("Rust dipole integrals (3 x 2 x 2):");
    for cmp in 0..3 {
        println!("  Component {}: {:?}", cmp, &dipole_rust[cmp*nao*nao..(cmp+1)*nao*nao]);
    }

    // Compare element-wise
    let mut max_error = 0.0;
    for cmp in 0..3 {
        for i in 0..nao {
            for j in 0..nao {
                let idx = cmp + 3 * (j + nao * i); // Fortran column-major
                let ref_val = dint_ref[idx].as_f64().unwrap();
                let rust_val = dipole_rust[idx];
                let error = (ref_val - rust_val).abs();
                if error > max_error {
                    max_error = error;
                }
                println!("  dipole[{},{}][{}] = ref: {:.10e}, rust: {:.10e}, err: {:.2e}",
                         i, j, cmp, ref_val, rust_val, error);
            }
        }
    }

    println!("Max dipole integral error: {:.2e}", max_error);
    assert!(max_error < 1e-6, "Dipole integral error too large: {:.2e}", max_error);
}
