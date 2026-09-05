use rust_dftb::SkData;

fn main() {
    let sk_dir = "/home/prokophapala/git_SW/dftbplus/external/slakos/origin/mio-1-1";
    let mut sk = SkData::load_sk_folder(sk_dir, ".skf", "-").unwrap();
    let mut ang_map = std::collections::HashMap::new();
    ang_map.insert("H".to_string(), vec![0]);
    sk.set_species_angular_momenta(ang_map);

    let tab = sk.get_pair("H", "H").unwrap();
    
    // Check grid parameters
    println!("dr: {}, n_grid: {}, n_integ: {}", tab.h.dr, tab.h.n_grid(), tab.h.n_integ());
    
    // Compare at different distances
    for &r in &[0.74_f64, 1.3984_f64] {
        let h_all = tab.h.eval(r).unwrap();
        let s_all = tab.s.eval(r).unwrap();
        println!("r={}: h[9]={:.10e}, s[9]={:.10e}", r, h_all[9], s_all[9]);
    }
}
