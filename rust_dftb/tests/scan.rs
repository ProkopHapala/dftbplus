//! Scan/NEB driver tests (Agent_5, Wave 2).
//!
//! These tests exercise the rigid coordinate scan driver logic against the
//! existing CPU `HamiltonianBuilder::build_scc()` backend. They verify:
//!  - `test_h2_bond_scan`: H2 bond scan 0.5–3.0 Å, 20 points, energy curve is
//!    smooth and physically reasonable (minimum near 0.74 Å, dissociation at
//!    large R).
//!  - `test_scan_saves_data`: per-replica data files are written and readable.
//!
//! Driven by `RUST_DFTB_SK_DIR` env var. Skipped gracefully if not set.

use rust_dftb::methods::dftb::forces::{parse_repulsive_spline, RepulsiveSpline};
use rust_dftb::{load_sk_for_species, DftbOutput, HamiltonianBuilder};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_SK_DIR: &str =
    "/home/prokophapala/git_SW/dftbplus/external/slakos/origin/mio-1-1";

const ANG2BOHR: f64 = 1.889_726_133;
const MIN_NEIGH_DIST: f64 = 1.0e-2;

/// Sum of DFTB repulsive pair energies (mirror of scan.rs helper).
fn repulsive_energy(sk_dir: &str, species: &[String], coords: &[[f64; 3]]) -> f64 {
    let n = coords.len();
    let mut cache: HashMap<(String, String), Option<RepulsiveSpline>> = HashMap::new();
    let mut e_rep = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let key = (species[i].clone(), species[j].clone());
            let spline = if let Some(s) = cache.get(&key) {
                s.clone()
            } else {
                let p1 = format!("{}/{}-{}.skf", sk_dir, key.0, key.1);
                let p2 = format!("{}/{}-{}.skf", sk_dir, key.1, key.0);
                let s = if Path::new(&p1).exists() {
                    parse_repulsive_spline(&p1).ok().flatten()
                } else if Path::new(&p2).exists() {
                    parse_repulsive_spline(&p2).ok().flatten()
                } else {
                    None
                };
                cache.insert(key.clone(), s.clone());
                s
            };
            let Some(spline) = spline else { continue };
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < MIN_NEIGH_DIST * MIN_NEIGH_DIST {
                continue;
            }
            let r_bohr = r2.sqrt() * ANG2BOHR;
            let (e, _de) = spline.eval(r_bohr);
            e_rep += e;
        }
    }
    e_rep
}

/// Skip the test gracefully if no SK dir is available.
fn sk_dir_or_skip() -> Option<String> {
    let dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| DEFAULT_SK_DIR.to_string());
    if !Path::new(&dir).exists() {
        eprintln!("[scan_test] SK dir {dir} does not exist — skipping");
        return None;
    }
    // Quick check that H-H.skf is present (needed for H2 tests).
    if !Path::new(&dir).join("H-H.skf").exists() {
        eprintln!("[scan_test] H-H.skf missing in {dir} — skipping");
        return None;
    }
    Some(dir)
}

/// Set the bond length between atoms `i` and `j` to `r` Å (mirror of scan.rs).
fn set_bond_length(coords: &mut [[f64; 3]], i: usize, j: usize, r: f64) {
    let dx = coords[j][0] - coords[i][0];
    let dy = coords[j][1] - coords[i][1];
    let dz = coords[j][2] - coords[i][2];
    let cur = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!(cur > 1e-12, "atoms {i} and {j} are coincident");
    let scale = r / cur;
    coords[j][0] = coords[i][0] + dx * scale;
    coords[j][1] = coords[i][1] + dy * scale;
    coords[j][2] = coords[i][2] + dz * scale;
}

/// Run an H2 bond scan and return (bond_lengths, total energies).
/// Total energy = electronic SCC energy + repulsive pair energy.
fn h2_bond_scan(sk_dir: &str, from: f64, to: f64, n: usize) -> Vec<(f64, f64)> {
    let species = vec!["H".to_string(), "H".to_string()];
    let sk = load_sk_for_species(sk_dir, &species).expect("failed to load SK");
    let builder = HamiltonianBuilder::new(sk);

    let base = [[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]]; // arbitrary initial bond, will be reset
    let mut curve = Vec::with_capacity(n);
    for p in 0..n {
        let t = if n > 1 { p as f64 / (n - 1) as f64 } else { 0.0 };
        let r = from + (to - from) * t;
        let mut coords = base.to_vec();
        set_bond_length(&mut coords, 0, 1, r);
        let scc = builder
            .build_scc(&species, &coords, 1000, 1e-10)
            .unwrap_or_else(|e| panic!("SCC failed at r={r}: {e}"));
        let e_rep = repulsive_energy(sk_dir, &species, &coords);
        let e_total = scc.energy + e_rep;
        eprintln!(
            "[scan_test] r={r:.4} Å  E_elec={:.10}  E_rep={:.10}  E_total={:.10}  n_iter={}",
            scc.energy, e_rep, e_total, scc.n_iter
        );
        curve.push((r, e_total));
    }
    curve
}

#[test]
fn test_h2_bond_scan() {
    let Some(sk_dir) = sk_dir_or_skip() else { return };

    let curve = h2_bond_scan(&sk_dir, 0.5, 3.0, 20);
    assert_eq!(curve.len(), 20, "expected 20 scan points");

    // 1. Energies must be finite (no NaN/Inf).
    for (r, e) in &curve {
        assert!(e.is_finite(), "non-finite energy at r={r}: {e}");
    }

    // 2. Minimum should be near the H2 equilibrium bond length (~0.74 Å for
    //    mio-1-1, tolerance ±0.15 Å).
    let (r_min, _e_min) = curve
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    eprintln!("[scan_test] minimum at r={r_min:.4} Å");
    assert!(
        (r_min - 0.74).abs() < 0.15,
        "H2 minimum at {r_min:.4} Å, expected near 0.74 Å"
    );

    // 3. Dissociation: energy at large R (3.0 Å) should be higher than the
    //    minimum (bond is stretched).
    let e_min = curve.iter().map(|(_, e)| *e).fold(f64::INFINITY, f64::min);
    let e_dissoc = curve.last().unwrap().1;
    assert!(
        e_dissoc > e_min,
        "dissociated energy {e_dissoc} should exceed minimum {e_min}"
    );

    // 4. Smoothness: no point should jump by more than 0.5 Hartree from its
    //    neighbor (H2 bond scan is smooth on this scale).
    for w in curve.windows(2) {
        let de = (w[0].1 - w[1].1).abs();
        assert!(de < 0.5, "energy jump {de:.4} between r={:.4} and r={:.4} too large", w[0].0, w[1].0);
    }

    // 5. Monotonic increase away from the minimum toward dissociation.
    //    Check the tail (from r_min onward) is non-decreasing within tolerance.
    let min_idx = curve
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    for w in curve[min_idx..].windows(2) {
        // Allow tiny numerical wiggles (1e-4 Ha).
        assert!(
            w[1].1 >= w[0].1 - 1e-4,
            "energy decreased after minimum: r={:.4} e={:.8} -> r={:.4} e={:.8}",
            w[0].0, w[0].1, w[1].0, w[1].1
        );
    }

    eprintln!("[scan_test] test_h2_bond_scan PASSED");
}

#[test]
fn test_scan_saves_data() {
    let Some(sk_dir) = sk_dir_or_skip() else { return };

    let species = vec!["H".to_string(), "H".to_string()];
    let sk = load_sk_for_species(&sk_dir, &species).expect("failed to load SK");
    let builder = HamiltonianBuilder::new(sk);

    let tmp = std::env::temp_dir().join("rust_dftb_scan_test_data");
    std::fs::create_dir_all(&tmp).unwrap();

    // Run a 3-point scan and save per-replica data (mirrors scan.rs logic).
    let rs = [0.7_f64, 0.74, 0.8];
    for (p, &r) in rs.iter().enumerate() {
        let mut coords = vec![[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
        set_bond_length(&mut coords, 0, 1, r);
        let scc = builder
            .build_scc(&species, &coords, 1000, 1e-10)
            .unwrap_or_else(|e| panic!("SCC failed at r={r}: {e}"));

        let rep_dir = tmp.join(format!("rep_{p:02}"));
        std::fs::create_dir_all(&rep_dir).unwrap();

        // geometry.xyz
        let mut f = File::create(rep_dir.join("geometry.xyz")).unwrap();
        writeln!(f, "2").unwrap();
        writeln!(f, "scan point {p} r={r}").unwrap();
        for (sp, c) in species.iter().zip(coords.iter()) {
            writeln!(f, "{sp} {:.10} {:.10} {:.10}", c[0], c[1], c[2]).unwrap();
        }
        // matrices
        DftbOutput::write_square(rep_dir.join("h0.dat").to_str().unwrap(), &scc.h0).unwrap();
        DftbOutput::write_square(rep_dir.join("h_scc.dat").to_str().unwrap(), &scc.h_scc).unwrap();
        DftbOutput::write_square(rep_dir.join("s.dat").to_str().unwrap(), &scc.s).unwrap();
        DftbOutput::write_square(rep_dir.join("density.dat").to_str().unwrap(), &scc.density).unwrap();
        // eigenvalues
        let mut f = File::create(rep_dir.join("eigenvalues.txt")).unwrap();
        writeln!(f, "# orbital eigenvalues (Hartree)").unwrap();
        for (k, e) in scc.eigenvalues.iter().enumerate() {
            writeln!(f, "{k} {:.16e}", e).unwrap();
        }
        // charges
        let mut f = File::create(rep_dir.join("charges.txt")).unwrap();
        writeln!(f, "# atom Mulliken_charge reference_q0").unwrap();
        for (a, (q, q0)) in scc.charges.iter().zip(scc.q0.iter()).enumerate() {
            writeln!(f, "{a} {:.16e} {:.16e}", q, q0).unwrap();
        }
        // energy (electronic, repulsive, total, n_iter)
        let e_rep = repulsive_energy(&sk_dir, &species, &coords);
        let e_total = scc.energy + e_rep;
        let mut f = File::create(rep_dir.join("energy.txt")).unwrap();
        writeln!(f, "# electronic_scc_energy repulsive_energy total_energy n_iter").unwrap();
        writeln!(f, "{:.16e} {:.16e} {:.16e} {}", scc.energy, e_rep, e_total, scc.n_iter).unwrap();
    }

    // Verify files exist and are readable.
    for p in 0..rs.len() {
        let rep_dir = tmp.join(format!("rep_{p:02}"));
        for name in ["geometry.xyz", "h0.dat", "h_scc.dat", "s.dat", "density.dat", "eigenvalues.txt", "charges.txt", "energy.txt"] {
            let path = rep_dir.join(name);
            assert!(path.exists(), "missing {name} for rep_{p:02}");
            let content = std::fs::read_to_string(&path).unwrap();
            assert!(!content.is_empty(), "{name} for rep_{p:02} is empty");
        }
        // Re-read h0.dat (write_square format: "n n" header then n rows of n floats)
        // and verify it parses back to a 2x2 matrix.
        let h0_txt = std::fs::read_to_string(rep_dir.join("h0.dat")).unwrap();
        let mut lines = h0_txt.lines();
        let header = lines.next().unwrap();
        let dims: Vec<usize> = header.split_whitespace().map(|x| x.parse().unwrap()).collect();
        assert_eq!(dims, vec![2, 2], "H0 header should be '2 2'");
        let mut n_rows = 0;
        let mut n_vals = 0;
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            n_rows += 1;
            n_vals += line.split_whitespace().filter(|x| x.parse::<f64>().is_ok()).count();
        }
        assert_eq!(n_rows, 2, "H0 should have 2 data rows");
        assert_eq!(n_vals, 4, "H0 should have 4 values total");
        // Re-read energy.txt and check the total energy value parses & is finite.
        let energy_txt = std::fs::read_to_string(rep_dir.join("energy.txt")).unwrap();
        let val_line = energy_txt.lines().nth(1).unwrap();
        let toks: Vec<f64> = val_line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        assert!(toks.len() >= 3, "energy.txt data line should have >=3 floats");
        assert!(toks[2].is_finite(), "re-read total energy not finite: {}", toks[2]);
    }

    // Cleanup.
    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("[scan_test] test_scan_saves_data PASSED");
}
