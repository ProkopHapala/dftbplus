use rust_dftb::{DftbOutput, HamiltonianBuilder, SkData};

fn max_abs_diff(a: &nalgebra::DMatrix<f64>, b: &nalgebra::DMatrix<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn parity_h0_methane_example() {
    // This test is designed to be run locally after generating DFTB+ reference files.
    // Provide these env vars:
    // - RUST_DFTB_SK_DIR: directory with .skf files (mio set)
    // - RUST_DFTB_REF_H: path to DFTB+ hamsqr1.dat generated with SCC=No
    // - RUST_DFTB_REF_S: path to DFTB+ oversqr.dat generated with SCC=No

    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };
    let Ok(ref_h) = std::env::var("RUST_DFTB_REF_H") else { return; };
    let Ok(ref_s) = std::env::var("RUST_DFTB_REF_S") else { return; };

    // Methane geometry in Angstrom (match your DFTB+ input!).
    // Note: DFTB+ uses atomic units internally but input/output here is Hartree.
    let species = vec![
        "C".to_string(),
        "H".to_string(),
        "H".to_string(),
        "H".to_string(),
        "H".to_string(),
    ];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.629118, 0.629118, 0.629118],
        [-0.629118, -0.629118, 0.629118],
        [-0.629118, 0.629118, -0.629118],
        [0.629118, -0.629118, -0.629118],
    ];

    let sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let builder = HamiltonianBuilder::new(sk);
    let ham = builder.build_non_scc(&species, &coords).unwrap();

    let h_ref = DftbOutput::read_square(&ref_h).unwrap();
    let s_ref = DftbOutput::read_square(&ref_s).unwrap();

    let dh = max_abs_diff(&ham.h0, &h_ref);
    let ds = max_abs_diff(&ham.s, &s_ref);

    // Start with a relaxed tolerance; tighten once units + geometry conventions are verified.
    assert!(dh < 1e-8, "H0 mismatch max diff = {dh:e}");
    assert!(ds < 1e-8, "S mismatch max diff = {ds:e}");
}

/// Verify that build_non_scc_sp_only produces identical results to the generic
/// build_non_scc for an sp-only system (pure carbon with mio-1-1 SK set).
#[test]
fn parity_sp_only_vs_generic() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec![
        "C".to_string(),
        "C".to_string(),
        "C".to_string(),
        "C".to_string(),
    ];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("C".to_string(), vec![0, 1]);
    sk.set_species_angular_momenta(ang_map);
    let builder = HamiltonianBuilder::new(sk);

    let ham_generic = builder.build_non_scc(&species, &coords).unwrap();
    let ham_sp_only = builder.build_non_scc_sp_only(&species, &coords).unwrap();

    let dh = max_abs_diff(&ham_generic.h0, &ham_sp_only.h0);
    let ds = max_abs_diff(&ham_generic.s, &ham_sp_only.s);

    assert!(dh < 1e-14, "sp-only H0 diverges from generic: max diff = {dh:e}");
    assert!(ds < 1e-14, "sp-only S diverges from generic: max diff = {ds:e}");
}

/// Parity + performance: sweep 1000 small displacements and rebuild H/S each step.
/// Uses a pure carbon system so both generic and sp-only paths are valid.
#[test]
fn benchmark_parity_sp_only() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };

    let species = vec![
        "C".to_string(),
        "C".to_string(),
        "C".to_string(),
        "C".to_string(),
    ];
    let base = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("C".to_string(), vec![0, 1]);
    sk.set_species_angular_momenta(ang_map);

    let builder = HamiltonianBuilder::new(sk);

    const N_STEPS: usize = 1000;
    let mut t_generic = std::time::Duration::ZERO;
    let mut t_sp_only = std::time::Duration::ZERO;

    for step in 0..N_STEPS {
        let disp = (step as f64) * 1e-4;
        let coords: Vec<[f64; 3]> = base
            .iter()
            .map(|c| [c[0] + disp, c[1] + disp, c[2] + disp])
            .collect();

        let t0 = std::time::Instant::now();
        let ham_g = builder.build_non_scc(&species, &coords).unwrap();
        t_generic += t0.elapsed();

        let t1 = std::time::Instant::now();
        let ham_s = builder.build_non_scc_sp_only(&species, &coords).unwrap();
        t_sp_only += t1.elapsed();

        let dh = max_abs_diff(&ham_g.h0, &ham_s.h0);
        let ds = max_abs_diff(&ham_g.s, &ham_s.s);
        assert!(dh < 1e-14, "step {step}: H0 diff = {dh:e}");
        assert!(ds < 1e-14, "step {step}: S diff = {ds:e}");
    }

    let ratio = t_generic.as_secs_f64() / t_sp_only.as_secs_f64();
    eprintln!("\n=== benchmark_parity_sp_only ===");
    eprintln!("  generic : {:?} ({:.3} us/step)", t_generic, t_generic.as_secs_f64() * 1e6 / N_STEPS as f64);
    eprintln!("  sp-only : {:?} ({:.3} us/step)", t_sp_only, t_sp_only.as_secs_f64() * 1e6 / N_STEPS as f64);
    eprintln!("  speedup : {:.2}x", ratio);
}
