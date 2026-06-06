use rust_dftb::{DftbOutput, HamiltonianBuilder, SkData};
use std::env;

fn main() {
    let sk_dir = env::var("RUST_DFTB_SK_DIR").unwrap();
    let ref_h = env::var("RUST_DFTB_REF_H").unwrap();
    let ref_s = env::var("RUST_DFTB_REF_S").unwrap();

    let mol = env::var("RUST_DFTB_MOL").unwrap_or("H2".to_string());
    let (species, coords): (Vec<String>, Vec<[f64; 3]>) = match mol.as_str() {
        "N2" => (vec!["N".to_string(), "N".to_string()], vec![[-1.05, 0.0, 0.0], [1.05, 0.0, 0.0]]),
        "HCOOH" => (
            vec!["C".to_string(), "O".to_string(), "O".to_string(), "H".to_string(), "H".to_string()],
            vec![
                [ 0.000,  0.000,  0.000],   // C
                [ 2.300,  0.000,  0.000],   // O (carbonyl)
                [-1.120,  2.250,  0.000],   // O (hydroxyl)
                [-2.100,  0.000,  0.000],   // H (formyl)
                [-2.500,  2.200,  0.000],   // H (hydroxyl)
            ]
        ),
        "HCONH2" => (
            vec!["C".to_string(), "O".to_string(), "N".to_string(), "H".to_string(), "H".to_string(), "H".to_string()],
            vec![
                [ 0.000,  0.000,  0.000],   // C
                [ 2.300,  0.000,  0.000],   // O
                [-1.300,  2.300,  0.000],   // N
                [-2.100,  0.000,  0.000],   // H (formyl)
                [-2.200,  2.300,  1.800],   // H (amine 1)
                [-0.800,  3.200,  0.000],   // H (amine 2)
            ]
        ),
        "HCOOH_rot" => (
            vec!["C".to_string(), "O".to_string(), "O".to_string(), "H".to_string(), "H".to_string()],
            vec![
                [ 0.0000,  0.0000,  0.0000],
                [ 1.4085,  1.6263, -0.8132],
                [-2.0637,  0.7990,  1.1915],
                [-1.2860, -1.4849,  0.7425],
                [-2.8782, -0.2121,  1.6617],
            ]
        ),
        _ => (vec!["H".to_string(), "H".to_string()], vec![[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]]),
    };

    println!("species.len() = {}", species.len());
    for (i, s) in species.iter().enumerate() {
        println!("  {}: {}", i, s);
    }

    let mut sk = SkData::load_sk_folder(&sk_dir, ".skf", "-").unwrap();

    // Set angular momenta per species for mio-1-1: H=s-only, C/O/N=sp
    let mut ang_map = std::collections::HashMap::new();
    for sp in ["H", "C", "O", "N"] {
        let shells = if sp == "H" { vec![0] } else { vec![0, 1] };
        ang_map.insert(sp.to_string(), shells);
    }
    sk.set_species_angular_momenta(ang_map);

    // Debug: print the H-H table header and a few grid points
    if let Some(table) = sk.pairs.get(&("H".to_string(), "H".to_string())) {
        println!("H-H table: n_grid_h={} n_grid_s={} dr={}", table.h.values.len(), table.s.values.len(), table.h.dr);
        println!("H-H onsite: {:?}", sk.onsite.get("H"));
        for i in 0..5.min(table.h.values.len()) {
            println!("  grid[{}] H={:?} S={:?}", i, &table.h.values[i], &table.s.values[i]);
        }
    } else {
        println!("H-H table NOT FOUND");
        for key in sk.pairs.keys() {
            println!("  key: {:?}", key);
        }
    }

    let builder = HamiltonianBuilder::new(sk);
    // inspect neighbor list manually
    let cutoff = builder.sk.pairs.values().map(|t| t.cutoff()).fold(0.0_f64, f64::max);
    println!("cutoff = {}", cutoff);
    let neigh = rust_dftb::neighbor::NeighborBuilder { cutoff }.build(&coords).unwrap();
    println!("neigh pairs = {:?}", neigh.pairs);

    // direct eval of H-H at 1.5
    if let Some(tab) = builder.sk.pairs.get(&("H".to_string(), "H".to_string())) {
        let h_sk = tab.h.eval(1.5).unwrap();
        let s_sk = tab.s.eval(1.5).unwrap();
        println!("h_sk(1.5) = {:?}", h_sk);
        println!("s_sk(1.5) = {:?}", s_sk);
    }

    let ham = builder.build_non_scc(&species, &coords).unwrap();

    println!("\n=== Rust H0 ===");
    for i in 0..ham.h0.nrows() {
        for j in 0..ham.h0.ncols() {
            print!("{:15.8e} ", ham.h0[(i, j)]);
        }
        println!();
    }
    println!("\n=== Rust S ===");
    for i in 0..ham.s.nrows() {
        for j in 0..ham.s.ncols() {
            print!("{:15.8e} ", ham.s[(i, j)]);
        }
        println!();
    }

    let h_ref = DftbOutput::read_square(&ref_h).unwrap();
    let s_ref = DftbOutput::read_square(&ref_s).unwrap();

    println!("\n=== DFTB+ H0 (ref) ===");
    for i in 0..h_ref.nrows() {
        for j in 0..h_ref.ncols() {
            print!("{:15.8e} ", h_ref[(i, j)]);
        }
        println!();
    }
    println!("\n=== DFTB+ S (ref) ===");
    for i in 0..s_ref.nrows() {
        for j in 0..s_ref.ncols() {
            print!("{:15.8e} ", s_ref[(i, j)]);
        }
        println!();
    }

    let mut max_dh = 0.0_f64;
    let mut max_ds = 0.0_f64;
    for i in 0..ham.h0.nrows() {
        for j in 0..ham.h0.ncols() {
            max_dh = max_dh.max((ham.h0[(i, j)] - h_ref[(i, j)]).abs());
            max_ds = max_ds.max((ham.s[(i, j)] - s_ref[(i, j)]).abs());
        }
    }
    println!("\nMax diff H0 = {:e}", max_dh);
    println!("Max diff S  = {:e}", max_ds);
    println!("\n=== Diff S (Rust - ref) ===");
    for i in 0..ham.s.nrows() {
        for j in 0..ham.s.ncols() {
            print!("{:12.5e} ", ham.s[(i, j)] - s_ref[(i, j)]);
        }
        println!();
    }
}
