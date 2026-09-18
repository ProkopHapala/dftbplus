//! Finite-precision parity test: Rust GPU PBC path vs the Fortran DFTB+
//! reference implementation.
//!
//! Reference data lives in repo-root `tests/pbc_fortran/` (outside the crate):
//!   - `dftb_in.hsd`   — periodic C-O chain, 4 explicit fractional
//!                       k-points, SCC, mio-1-1
//!   - `reference.txt` — extracted charges/eigenvalues/energies
//!                       (regenerate with `./run_reference.sh`)
//!
//! What is compared (finite-precision, NOT binary parity):
//!   - Mulliken populations:  q_rust vs q0 − q_net_fortran, tol 1e-3 e
//!   - Eigenvalues per k:     converged SCC levels, tol 2e-3 Ha
//!                            (band.out itself is printed at ~4e-6 Ha)
//!   - Electronic energy:     e_band − ½Δq·v − q0·v  vs
//!                            "Total Electronic energy", tol 5e-3 Ha
//!
//! Gated: needs an OpenCL device + RUST_DFTB_SK_DIR (default mio-1-1).

use rust_dftb::qmqm::gpu_pbc::GpuPbc;
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;

const REF_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/pbc_fortran/reference.txt"
);

/// Flat `key = v1 v2 ...` file parser.
fn parse_reference(path: &str) -> std::collections::HashMap<String, Vec<f64>> {
    let mut out = std::collections::HashMap::new();
    for ln in std::fs::read_to_string(path).unwrap().lines() {
        let ln = ln.trim();
        if ln.is_empty() || ln.starts_with('#') {
            continue;
        }
        let (k, v) = ln
            .split_once('=')
            .unwrap_or_else(|| panic!("bad line: {ln}"));
        let vals: Vec<f64> = v.split_whitespace().map(|t| t.parse().unwrap()).collect();
        out.insert(k.trim().to_string(), vals);
    }
    out
}

fn sk_dir() -> Option<String> {
    let dir = std::env::var("RUST_DFTB_SK_DIR")
        .unwrap_or_else(|_| "/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1".to_string());
    if std::path::Path::new(&dir).is_dir() {
        Some(dir)
    } else {
        None
    }
}

/// Geometry must mirror repo-root tests/pbc_fortran/dftb_in.hsd exactly.
/// Returns (species, coords Å, lattice Å, k_frac, k_weights).
fn cchain_input() -> (
    Vec<String>,
    Vec<[f64; 3]>,
    [[f64; 3]; 3],
    Vec<[f64; 3]>,
    Vec<f32>,
) {
    let species = vec!["C".to_string(), "O".to_string()];
    let coords = vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]];
    let lat = [[3.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]];
    let k_frac = vec![
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [-0.25, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ];
    let kw = vec![0.25f32; 4];
    (species, coords, lat, k_frac, kw)
}

#[test]
fn gpu_pbc_fortran_parity() {
    let Some(dir) = sk_dir() else {
        eprintln!("SK dir missing — skipping");
        return;
    };
    if GpuRuntime::new().is_err() {
        eprintln!("No OpenCL device — skipping");
        return;
    }
    let reference = match std::path::Path::new(REF_PATH).exists() {
        true => parse_reference(REF_PATH),
        false => panic!("reference.txt missing — run tests/pbc_fortran/run_reference.sh"),
    };
    let n_at = reference["n_atoms"][0] as usize;
    let nk = reference["n_kpoints"][0] as usize;
    let q_net_ref = &reference["charges_net"];
    let eig_ref = &reference["eigenvalues_ha"];
    let e_elec_ref = reference["energy_electronic"][0];
    let e_band_ref = reference["energy_band"][0];
    assert_eq!(n_at, 2);
    assert_eq!(nk, 4);

    let (species, coords, lat, k_frac, kw) = cchain_input();
    let sk = rust_dftb::load_sk_for_species(&dir, &species).unwrap();
    let mut eng =
        GpuPbc::new(sk, species.clone(), coords.clone(), lat, &k_frac, &kw, None).unwrap();

    let (ok, hist) = eng.scc(0.3, 1e-7, 40).unwrap();
    eprintln!(
        "scc rms hist: {:?}",
        hist.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>()
    );
    assert!(ok.iter().all(|&x| x), "jacobi cert failed");
    assert!(
        *hist.last().unwrap() < 1e-6,
        "SCC did not converge: rms={}",
        hist.last().unwrap()
    );

    // --- Mulliken populations: Fortran "gross charge" is q0 − pop ---
    let q0: Vec<f32> = species
        .iter()
        .map(|s| match s.as_str() {
            "C" => 4.0,
            "O" => 6.0,
            _ => panic!("q0 for {s}?"),
        })
        .collect();
    let q = eng.plan.read_charges(&eng.rt).unwrap();
    let mut dq_max = 0.0f64;
    for a in 0..n_at {
        let pop_ref = q0[a] as f64 - q_net_ref[a];
        let d = (q[a] as f64 - pop_ref).abs();
        eprintln!(
            "atom {a}: pop rust={:.6} fortran={pop_ref:.6} (|Δ|={d:.2e})",
            q[a]
        );
        dq_max = dq_max.max(d);
    }
    assert!(dq_max < 1e-3, "Mulliken parity failed: max|Δ|={dq_max:e}");

    // --- energies first: compute_energy finalize()s the state so the
    // eigenvalues below correspond to the committed converged charges ---
    let n_occ = eng.n_occ();
    let e = eng.plan.compute_energy(&mut eng.rt, n_occ).unwrap();
    let e_band_rust = eng.plan.e_scal_host[0];

    // --- eigenvalues per k (band.out order == k_frac order) ---
    let eigs = eng.plan.read_eigenvalues(&mut eng.rt).unwrap();
    let n = eig_ref.len() / nk;
    assert_eq!(eigs.len(), nk * n, "eig count mismatch");
    let mut de_max = 0.0f64;
    let mut worst = (0, 0, 0.0f64, 0.0f64);
    for k in 0..nk {
        for b in 0..n {
            let d = (eigs[k * n + b] as f64 - eig_ref[k * n + b]).abs();
            if d > de_max {
                de_max = d;
                worst = (k, b, eigs[k * n + b] as f64, eig_ref[k * n + b]);
            }
        }
    }
    eprintln!(
        "eig parity: max|Δε|={de_max:.3e} Ha  worst k={} band={} rust={:.6} ref={:.6}",
        worst.0, worst.1, worst.2, worst.3
    );
    assert!(de_max < 2e-3, "eigenvalue parity failed: {de_max:e} Ha");

    // --- energies: e_band and assembled electronic energy ---
    eprintln!(
        "e_band: rust={:.8} fortran={:.8} (Δ={:.2e})",
        e_band_rust,
        e_band_ref,
        e_band_rust - e_band_ref
    );
    eprintln!(
        "e_elec: rust={:.8} fortran={:.8} (Δ={:.2e})",
        e[0],
        e_elec_ref,
        e[0] - e_elec_ref
    );
    assert!(
        (e_band_rust - e_band_ref).abs() < 5e-3,
        "band energy off: {}",
        e_band_rust - e_band_ref
    );
    assert!(
        (e[0] - e_elec_ref).abs() < 5e-3,
        "electronic energy off: {}",
        e[0] - e_elec_ref
    );
}
