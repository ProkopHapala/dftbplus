use rust_dftb::load_sk_for_species;
#[test]
#[ignore] // diagnostic: prints per-pair SK cutoffs (the cut_bohr investigation, 2026-09-14)
fn print_sk_cutoffs() {
    for dir in ["/home/prokop/SIMULATIONS/dftbplus/slakos/pbc-0-3",
                "/home/prokop/SIMULATIONS/dftbplus/slakos/matsci-0-3"] {
        let species = vec!["Si".to_string(), "H".to_string()];
        let sk = load_sk_for_species(dir, &species).unwrap();
        eprintln!("== {dir}");
        for ((a, b), t) in &sk.pairs {
            eprintln!("   {a}-{b}: r_max = {:.4} bohr = {:.4} A  n_grid={}", t.cutoff(), t.cutoff()/0.529177, t.h.n_grid());
        }
    }
}
