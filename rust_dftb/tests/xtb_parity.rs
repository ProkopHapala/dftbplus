//! Parity tests: Rust xTB implementation vs tblite C API reference.

use std::process::{Command, Stdio};
use std::io::Write;

const TBLITE_HELPER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tblite_helper");

/// Run tblite C helper on a molecule and return parsed JSON.
fn run_tblite(nat: usize, charge: i32, uhf: i32, atoms: &[(usize, [f64; 3])]) -> serde_json::Value {
    let mut child = Command::new(TBLITE_HELPER)
        .arg(format!("{}", nat))
        .arg(format!("{}", charge))
        .arg(format!("{}", uhf))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn tblite_helper");

    {
        let stdin = child.stdin.as_mut().unwrap();
        for (_i, (z, pos)) in atoms.iter().enumerate() {
            writeln!(stdin, "{} {} {} {}", z, pos[0], pos[1], pos[2]).unwrap();
        }
    }

    let output = child.wait_with_output().expect("Failed to read tblite_helper output");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("tblite_helper failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Find the JSON part (skip any SCF output lines)
    let json_start = stdout.find('{').expect("No JSON found in output");
    let json_str = &stdout[json_start..];
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Parse a flat array from JSON into a DMatrix.
fn json_to_dmatrix(json: &serde_json::Value, key: &str, n: usize) -> nalgebra::DMatrix<f64> {
    let arr = json[key].as_array().unwrap();
    let mut mat = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            mat[(i, j)] = arr[i * n + j].as_f64().unwrap();
        }
    }
    mat
}

fn json_to_vec(json: &serde_json::Value, key: &str) -> Vec<f64> {
    json[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

/// Compare two matrices element-wise and report max error.
fn compare_matrices(a: &nalgebra::DMatrix<f64>, b: &nalgebra::DMatrix<f64>, name: &str) {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    let mut max_err = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut max_idx = (0, 0);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = (a[(i, j)] - b[(i, j)]).abs();
            let ref_val = b[(i, j)].abs().max(1e-10);
            let rel = diff / ref_val;
            if diff > max_err {
                max_err = diff;
                max_rel = rel;
                max_idx = (i, j);
            }
        }
    }
    println!("{name}: max_abs_err = {max_err:.6e}, max_rel_err = {max_rel:.6e} at {:?}", max_idx);
    assert!(max_err < 1e-4, "{name} mismatch too large: max_err={max_err:.6e}");
}

fn compare_vecs(a: &[f64], b: &[f64], name: &str) {
    assert_eq!(a.len(), b.len());
    let mut max_err = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        let ref_val = b[i].abs().max(1e-10);
        let rel = diff / ref_val;
        if diff > max_err {
            max_err = diff;
            max_rel = rel;
        }
    }
    println!("{name}: max_abs_err = {max_err:.6e}, max_rel_err = {max_rel:.6e}");
    assert!(max_err < 1e-3, "{name} mismatch too large: max_err={max_err:.6e}");
}

#[test]
fn test_h2_gfn1_parity() {
    // H2 at equilibrium: 0.74 Å
    let atoms = vec![
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.74]),
    ];

    let ref_data = run_tblite(2, 0, 0, &atoms);
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

    compare_matrices(&h_rust, &h_ref, "H2 H0");
    compare_matrices(&s_rust, &s_ref, "H2 S");
}

#[test]
fn test_n2_gfn1_parity() {
    // N2 at equilibrium: 1.10 Å
    let atoms = vec![
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, 1.10]),
    ];

    let ref_data = run_tblite(2, 0, 0, &atoms);
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

    compare_matrices(&h_rust, &h_ref, "N2 H0");
    compare_matrices(&s_rust, &s_ref, "N2 S");
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

    let ref_data = run_tblite(5, 0, 0, &atoms);
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

    compare_matrices(&h_rust, &h_ref, "HCOOH H0");
    compare_matrices(&s_rust, &s_ref, "HCOOH S");
}

#[test]
fn test_h2_scc_parity() {
    // H2 at equilibrium: 0.74 Å
    let atoms = vec![
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.74]),
    ];

    let ref_data = run_tblite(2, 0, 0, &atoms);
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

    compare_vecs(q_rust.data.as_vec(), &q_ref, "H2 charges");
    compare_vecs(emo.data.as_vec(), &emo_ref, "H2 eigenvalues");
}

#[test]
fn test_n2_scc_parity() {
    // N2 at equilibrium: 1.10 Å
    let atoms = vec![
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, 1.10]),
    ];

    let ref_data = run_tblite(2, 0, 0, &atoms);
    let q_ref = json_to_vec(&ref_data, "charges");
    let emo_ref = json_to_vec(&ref_data, "eigenvalues");

    let aatoau = 1.889726133;
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.10 * aatoau],
    ];
    let elem_idx = vec![6usize, 6]; // N = index 6
    let n_electrons = 10; // N2 has 10 electrons

    let (_density, qsh, emo) = rust_dftb::methods::xtb::scf::run_scf(
        &coords, &elem_idx, n_electrons, 100, 1e-6
    );

    let nshell_per_atom = vec![2, 2]; // N has 2 shells
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

    compare_vecs(q_rust.data.as_vec(), &q_ref, "N2 charges");
    compare_vecs(emo.data.as_vec(), &emo_ref, "N2 eigenvalues");
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

    let ref_data = run_tblite(5, 0, 0, &atoms);
    let q_ref = json_to_vec(&ref_data, "charges");
    let emo_ref = json_to_vec(&ref_data, "eigenvalues");

    let aatoau = 1.889726133;
    let coords: Vec<[f64; 3]> = atoms.iter().map(|(_, p)| {
        [p[0] * aatoau, p[1] * aatoau, p[2] * aatoau]
    }).collect();
    let elem_idx = vec![0usize, 7, 5, 7, 0]; // H=0, O=7, C=5
    let n_electrons = 18; // HCOOH has 18 electrons

    let (_density, qsh, emo) = rust_dftb::methods::xtb::scf::run_scf(
        &coords, &elem_idx, n_electrons, 100, 1e-6
    );

    let nshell_per_atom = vec![2, 2, 2, 2, 2]; // All have 2 shells
    let q_rust = rust_dftb::methods::xtb::mulliken::atomic_charges(&qsh, &nshell_per_atom);

    // --- Direct Hamiltonian comparison for fixed charges ---
    let nao = 16;
    let p_tblite = json_to_dmatrix(&ref_data, "density", nao);
    let s_tblite = json_to_dmatrix(&ref_data, "overlap", nao);
    let h0_tblite = json_to_dmatrix(&ref_data, "hamiltonian", nao);
    let c_tblite = json_to_dmatrix(&ref_data, "coefficients", nao).transpose();
    let ao2sh = vec![0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9];
    let n0sh = rust_dftb::methods::xtb::mulliken::reference_shell_occupations(&nshell_per_atom, &elem_idx, &vec![0,0,0,1,0,1,0,1,0,0]);
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
    compare_matrices(&h1_rust, &h1_tblite, "HCOOH H1");
    compare_matrices(&h0_rust, &h0_tblite, "HCOOH H0");
    compare_matrices(&s_rust, &s_tblite, "HCOOH S");

    compare_vecs(q_rust.data.as_vec(), &q_ref, "HCOOH charges");
    compare_vecs(emo.data.as_vec(), &emo_ref, "HCOOH eigenvalues");
}
