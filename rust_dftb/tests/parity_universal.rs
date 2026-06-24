//! Universal parity test for arbitrary molecules from XYZ files.
//!
//! Driven by environment variables:
//! - RUST_DFTB_SK_DIR        – directory with .skf files
//! - RUST_DFTB_SPECIES        – comma-separated species names
//! - RUST_DFTB_COORDS         – comma-separated coordinates (x1,y1,z1,x2,y2,z2,...)
//! - RUST_DFTB_REF_H          – path to Fortran hamsqr1.dat (non-SCC H0)
//! - RUST_DFTB_REF_S          – path to Fortran oversqr.dat
//! - RUST_DFTB_REF_H_SCC      – optional: path to Fortran H_scc (SCC with fixed charges)
//! - RUST_DFTB_DELTA_Q        – optional: comma-separated deltaQ values for SCC
//! - RUST_DFTB_TOLERANCE      – optional: tolerance (default 1e-7 for H0, 1e-6 for SCC)

use rust_dftb::{load_sk_for_species, max_abs_diff, parse_coords, parse_f64_list, parse_species, DftbOutput, HamiltonianBuilder};
use rust_dftb::qmqm::{
    Fragment, FragmentNeighborList, FragmentTemplate, GammaTable, SimpleMixer,
    solver::MultiSystemSolver,
};

#[test]
fn parity_universal_from_env() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };
    let Ok(ref_h) = std::env::var("RUST_DFTB_REF_H") else { return; };
    let Ok(ref_s) = std::env::var("RUST_DFTB_REF_S") else { return; };
    let Ok(species_s) = std::env::var("RUST_DFTB_SPECIES") else { return; };
    let Ok(coords_s) = std::env::var("RUST_DFTB_COORDS") else { return; };

    let tol: f64 = std::env::var("RUST_DFTB_TOLERANCE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-7);

    let species = parse_species(&species_s);
    let coords = parse_coords(&coords_s);
    assert_eq!(species.len(), coords.len(), "species/coords length mismatch");

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // ---- Non-SCC H0 parity ----
    let builder = HamiltonianBuilder::new(sk.clone());
    let ham = builder.build_non_scc(&species, &coords).unwrap();

    let h_ref = DftbOutput::read_square(&ref_h).unwrap();
    let s_ref = DftbOutput::read_square(&ref_s).unwrap();

    let dh = max_abs_diff(&ham.h0, &h_ref);
    let ds = max_abs_diff(&ham.s, &s_ref);

    assert!(dh < tol, "H0 mismatch max diff = {dh:e}");
    assert!(ds < tol, "S mismatch max diff = {ds:e}");

    // ---- Optional SCC parity ----
    if let (Ok(ref_h_scc), Ok(delta_q_s)) = (
        std::env::var("RUST_DFTB_REF_H_SCC"),
        std::env::var("RUST_DFTB_DELTA_Q"),
    ) {
        let delta_q = parse_f64_list(&delta_q_s);
        assert_eq!(
            delta_q.len(),
            species.len(),
            "delta_q length must match number of atoms"
        );

        let template = FragmentTemplate::new(&sk, species.clone(), coords.clone()).unwrap();
        let frag = Fragment::from_template(template.clone(), coords.clone());

        // Single-fragment: centroid + empty neighbor list
        let centroid = coords.iter().fold([0.0, 0.0, 0.0], |mut acc, c| {
            acc[0] += c[0];
            acc[1] += c[1];
            acc[2] += c[2];
            acc
        });
        let n = coords.len() as f64;
        let centroid = [centroid[0] / n, centroid[1] / n, centroid[2] / n];
        let frag_neighbors = FragmentNeighborList::build(&vec![centroid], 10.0);

        // Build gamma table from env-provided Hubbard U values.
        // RUST_DFTB_HUBBARD_U = comma-separated U values (one per unique species)
        // RUST_DFTB_SPECIES_U   = comma-separated species names in same order
        let hubbard_u: Vec<f64> = std::env::var("RUST_DFTB_HUBBARD_U")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| {
                // Fallback: hardcoded values for common elements
                let mut u_map = std::collections::HashMap::new();
                u_map.insert("H", 0.4195);
                u_map.insert("C", 0.3647);
                u_map.insert("N", 0.4309);
                u_map.insert("O", 0.4954);
                u_map.insert("F", 0.4500);
                u_map.insert("S", 0.3200);
                u_map.insert("P", 0.3500);
                let unique: Vec<String> = species.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
                unique.iter().map(|sp| *u_map.get(sp.as_str()).unwrap_or(&0.4)).collect()
            });
        let gamma = GammaTable::from_hubbard_u(hubbard_u);

        let mixer = SimpleMixer::new(0.3);
        let mut solver = MultiSystemSolver::new(vec![frag], frag_neighbors, gamma, mixer);

        // fixed_charges = q0 + delta_q
        let fixed_charges: Vec<f64> = solver.fragments[0]
            .template
            .q0
            .iter()
            .zip(delta_q.iter())
            .map(|(q0, dq)| q0 + dq)
            .collect();
        solver.build_h_scc_with_fixed_charges(&fixed_charges);

        let h_scc_rust = &solver.fragments[0].h_scc;
        let h_scc_ref = DftbOutput::read_square(&ref_h_scc).unwrap();

        let tol_scc: f64 = std::env::var("RUST_DFTB_TOLERANCE_SCC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-6);

        let diff_scc = max_abs_diff(h_scc_rust, &h_scc_ref);
        assert!(
            diff_scc < tol_scc,
            "H_scc mismatch max diff = {diff_scc:e}"
        );
    }
}
