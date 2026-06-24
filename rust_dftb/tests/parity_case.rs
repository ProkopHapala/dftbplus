use rust_dftb::{
    load_sk_for_species, max_abs_diff, parse_coords, parse_species, permute_sp_per_atom,
    DftbOutput, HamiltonianBuilder,
};

#[test]
fn parity_non_scc_case_from_env() {
    // Generic parity test driven by environment variables.
    // This is intended for sweeps over distances / rotations.
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else { return; };
    let Ok(ref_h) = std::env::var("RUST_DFTB_REF_H") else { return; };
    let Ok(ref_s) = std::env::var("RUST_DFTB_REF_S") else { return; };
    let Ok(species_s) = std::env::var("RUST_DFTB_SPECIES") else { return; };
    let Ok(coords_s) = std::env::var("RUST_DFTB_COORDS") else { return; };

    let species = parse_species(&species_s);
    let coords = parse_coords(&coords_s);
    assert_eq!(species.len(), coords.len());

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let builder = HamiltonianBuilder::new(sk);
    let ham = builder.build_non_scc(&species, &coords).unwrap();

    let h_ref = DftbOutput::read_square(&ref_h).unwrap();
    let s_ref = DftbOutput::read_square(&ref_s).unwrap();

    // For s-only systems (no p-orbitals), no permutation needed
    // For sp-only systems, try all p-orbital orderings
    if ham.h0.nrows() % 4 == 0 {
        let perms = [
            ([0, 1, 2], "p:1-2-3"),
            ([2, 0, 1], "p:3-1-2"),
            ([1, 2, 0], "p:2-3-1"),
            ([1, 0, 2], "p:2-1-3"),
            ([2, 1, 0], "p:3-2-1"),
            ([0, 2, 1], "p:1-3-2"),
        ];

        let mut best = (f64::INFINITY, f64::INFINITY, "");
        for (perm, name) in perms {
            let h = permute_sp_per_atom(&ham.h0, perm);
            let s = permute_sp_per_atom(&ham.s, perm);
            let dh = max_abs_diff(&h, &h_ref);
            let ds = max_abs_diff(&s, &s_ref);
            if dh + ds < best.0 + best.1 {
                best = (dh, ds, name);
            }
        }
        assert!(best.0 < 1e-7, "H0 mismatch best(max diff) = {:e} for {}", best.0, best.2);
        assert!(best.1 < 1e-7, "S mismatch best(max diff) = {:e} for {}", best.1, best.2);
    } else {
        // s-only: direct comparison
        let dh = max_abs_diff(&ham.h0, &h_ref);
        let ds = max_abs_diff(&ham.s, &s_ref);
        assert!(dh < 1e-7, "H0 mismatch (s-only) = {:e}", dh);
        assert!(ds < 1e-7, "S mismatch (s-only) = {:e}", ds);
    }
}
