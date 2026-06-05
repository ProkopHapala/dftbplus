use rust_dftb::{DftbOutput, HamiltonianBuilder, SkData};

fn max_abs_diff(a: &nalgebra::DMatrix<f64>, b: &nalgebra::DMatrix<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn permute_sp_per_atom(mat: &nalgebra::DMatrix<f64>, perm_p: [usize; 3]) -> nalgebra::DMatrix<f64> {
    // Per atom basis assumed size 4: [s, p1, p2, p3].
    // perm_p maps (px,py,pz) desired order? Here we implement generic permutation of the 3 p slots.
    let n = mat.nrows();
    assert_eq!(n, mat.ncols());
    assert_eq!(n % 4, 0);
    let n_at = n / 4;

    let mut idx = Vec::with_capacity(n);
    for a in 0..n_at {
        let base = 4 * a;
        idx.push(base);
        // p slots are base+1..base+3
        let p = [base + 1, base + 2, base + 3];
        idx.push(p[perm_p[0]]);
        idx.push(p[perm_p[1]]);
        idx.push(p[perm_p[2]]);
    }

    let mut out = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = mat[(idx[i], idx[j])];
        }
    }
    out
}

fn parse_species(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

fn parse_coords(s: &str) -> Vec<[f64; 3]> {
    let vals: Vec<f64> = s
        .split(|c| c == ',' || c == ' ' || c == ';' || c == '\n' || c == '\t' || c == '[' || c == ']' || c == '(' || c == ')')
        .filter(|x| !x.is_empty())
        .filter_map(|x| x.parse::<f64>().ok())
        .collect();
    assert!(vals.len() % 3 == 0);
    vals.chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

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

    let sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let builder = HamiltonianBuilder::new(sk);
    let ham = builder.build_non_scc(&species, &coords).unwrap();

    let h_ref = DftbOutput::read_square(&ref_h).unwrap();
    let s_ref = DftbOutput::read_square(&ref_s).unwrap();

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

    assert!(best.0 < 1e-8, "H0 mismatch best(max diff) = {:e} for {}", best.0, best.2);
    assert!(best.1 < 1e-8, "S mismatch best(max diff) = {:e} for {}", best.1, best.2);
}
