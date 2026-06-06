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
