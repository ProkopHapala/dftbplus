//! Production SparseDftb loop: init once, reuse kernels/buffers across SCC and FIRE.
//!
//! Not a throwaway per-test OpenCL context. NVIDIA required.
//!
//!   RUST_DFTB_SK_DIR=... cargo test --test sparse_dftb -- --nocapture --test-threads=1

use rust_dftb::load_sk_for_species;
use rust_dftb::methods::sparse::harness::require_sih_sk_dir;
use rust_dftb::methods::sparse::SparseDftb;

fn sih4() -> (Vec<String>, Vec<[f64; 3]>) {
    let species = vec!["Si".into(), "H".into(), "H".into(), "H".into(), "H".into()];
    let bond = 1.48f64;
    let theta = 109.47f64 * std::f64::consts::PI / 180.0;
    let (c, s) = (theta.cos(), theta.sin());
    let coords = vec![
        [0.0, 0.0, 0.0],
        [bond, 0.0, 0.0],
        [bond * c, bond * s, 0.0],
        [bond * c, bond * s * c, bond * s * s],
        [bond * c, -bond * s * c, -bond * s * s],
    ];
    (species, coords)
}

fn mean_sih(coords: &[[f64; 3]]) -> f64 {
    let si = coords[0];
    let mut s = 0.0;
    for h in &coords[1..] {
        let dx = h[0] - si[0]; let dy = h[1] - si[1]; let dz = h[2] - si[2];
        s += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    s / 4.0
}

#[test]
fn test_sparse_dftb_sih4_reuse_scc_and_fire() {
    let dir = require_sih_sk_dir();
    let (sp, xyz) = sih4();
    let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("load SK: {e}"));
    let mut eng = SparseDftb::new(sk, &dir, sp, xyz).unwrap_or_else(|e| panic!("SparseDftb::new: {e}"));
    let s1 = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC1: {e}"));
    let e1 = eng.energy().unwrap();
    eprintln!("SiH4 SparseDftb SCC1: E={e1:.8} rms={:.3e} iters={} Tr(KS)={:.6} R_I={:.3e}", s1.rms, s1.n_iters, s1.tr_ks, s1.r_i);
    assert!(e1.is_finite(), "E non-finite: {e1}");
    assert!(s1.rms < 1e-4, "SiH4 SCC diverged: rms={:.3e}", s1.rms);
    assert!((s1.tr_ks - 4.0).abs() < 0.05, "Tr(KS)={} != 4", s1.tr_ks);

    let s2 = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC2 reuse: {e}"));
    let e2 = eng.energy().unwrap();
    eprintln!("SiH4 SparseDftb SCC2 (reuse): E={e2:.8} rms={:.3e} iters={}", s2.rms, s2.n_iters);
    assert!((e2 - e1).abs() < 1e-5, "reuse SCC |dE|={:.3e} — pipeline is not persistent/warm", (e2 - e1).abs());

    let f = eng.forces().unwrap_or_else(|e| panic!("forces: {e}"));
    let mut max_f = 0.0f64;
    for fi in &f.forces {
        for &c in fi { max_f = max_f.max(c.abs()); }
    }
    eprintln!("SiH4 SparseDftb |F|_max={max_f:.4e}");
    assert!(max_f.is_finite(), "forces non-finite");

    let r0 = mean_sih(eng.coords());
    let mf = eng.fire_step(1e-2).unwrap_or_else(|e| panic!("FIRE: {e}"));
    let s3 = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC after FIRE: {e}"));
    let e3 = eng.energy().unwrap();
    let r1 = mean_sih(eng.coords());
    eprintln!("SiH4 SparseDftb after 1 FIRE: E={e3:.8} max|F|={mf:.4e} rms={:.3e} <SiH> {r0:.4}→{r1:.4} Å", s3.rms);
    assert!(e3.is_finite() && e3.abs() < 20.0, "FIRE step exploded: E={e3}");
    assert!((e3 - e1).abs() < 1.0, "FIRE moved energy by {:.3} Ha (E0={e1:.8} E={e3:.8})", (e3 - e1).abs());
    assert!(r1 > 1.2 && r1 < 2.0, "FIRE Si–H mean {r1:.3} Å left physical window");

    let mf_md = eng.md_step(0.05).unwrap_or_else(|e| panic!("MD: {e}"));
    let s4 = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC after MD: {e}"));
    let e4 = eng.energy().unwrap();
    eprintln!("SiH4 SparseDftb after 1 MD: E={e4:.8} max|F|={mf_md:.4e} rms={:.3e}", s4.rms);
    assert!(e4.is_finite() && e4.abs() < 20.0, "MD step exploded: E={e4}");
}
