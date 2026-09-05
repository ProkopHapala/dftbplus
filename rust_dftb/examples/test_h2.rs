use rust_dftb::{HamiltonianBuilder, SkData};

fn main() {
    let sk_dir = "/home/prokophapala/git_SW/dftbplus/external/slakos/origin/mio-1-1";
    let species = vec!["H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.74, 0.0, 0.0],
    ];

    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("H".to_string(), vec![0]);
    ang_map.insert("C".to_string(), vec![0, 1]);
    ang_map.insert("N".to_string(), vec![0, 1]);
    ang_map.insert("O".to_string(), vec![0, 1]);
    sk.set_species_angular_momenta(ang_map);

    let builder = HamiltonianBuilder::new(sk);
    let ham = builder.build_non_scc(&species, &coords).unwrap();

    println!("Rust H0:\n{:.16e}", ham.h0);
    println!("Rust S:\n{:.16e}", ham.s);
}
