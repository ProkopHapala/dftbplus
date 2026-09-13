//! L4 (manifest §15.10): magnitude-aware top-k mask experiment — the
//! decisive compressibility test. From a wide-mask converged K on
//! si_sphere_R14, keep the top-{32,48,64,96,128} ‖K_ij‖_F blocks per row
//! (symmetrized) as M_K=M_Z, rerun SCC, and compare E/max|F|/iters vs the
//! reference. Case A: top-k works → the geometric sphere was a terrible
//! sparsifier and a 64–128-neighbor representation exists. Case B: even
//! top-128 fails → the projector genuinely isn't sparse in the AO basis.
//!
//! GPU-heavy → `#[ignore]`; run:
//!   cargo test --release --test sparse_topk -- --ignored --nocapture

use rust_dftb::io::parse_xyz;
use rust_dftb::load_sk_for_species;
use rust_dftb::methods::sparse::bsr4::build_topk_mask;
use rust_dftb::methods::sparse::harness::require_sih_sk_dir;
use rust_dftb::methods::sparse::sparse_dftb::{SparseDftb, SparseDftbConfig};

fn max_deg(m: &(Vec<u32>, Vec<u32>), n: usize) -> u32 {
    (0..n).map(|i| m.0[i + 1] - m.0[i]).max().unwrap_or(0)
}

#[test]
#[ignore]
fn topk_mask_sweep_r14() {
    let sk_dir = require_sih_sk_dir();
    let mol = parse_xyz("../debug/nanocrystals/si_sphere_R14.xyz")
        .unwrap_or_else(|e| panic!("parse_xyz R14: {e}"));
    let n = mol.species.len();
    let sk = load_sk_for_species(&sk_dir, &mol.species)
        .unwrap_or_else(|e| panic!("load_sk_for_species: {e}"));

    // Wide reference: full K/Z radius, r_trunc=8, K-TC2.
    let mut cfg = SparseDftbConfig {
        r_trunc_ang: Some(8.0), taper_w_ang: 1.0,
        max_deg_hs: Some(512), max_deg_k: Some(512), max_deg_z: Some(512),
        tc2_tol: 1e-5, ns_tol: 1e-4, purifier_p: Some(false),
        ..Default::default()
    };
    let mut eng = SparseDftb::with_config(sk.clone(), &sk_dir, mol.species.clone(), mol.coords.clone(), cfg.clone())
        .unwrap_or_else(|e| panic!("ref SparseDftb: {e}"));
    let scc = eng.scc(60, 1e-5).unwrap_or_else(|e| panic!("ref scc: {e}"));
    let e_ref = eng.energy().unwrap();
    let k_ref = eng.k_bsr().unwrap_or_else(|e| panic!("k_bsr: {e}"));
    let deg_ref = (0..n).map(|i| k_ref.row_ptr[i + 1] - k_ref.row_ptr[i]).max().unwrap_or(0);
    eprintln!("REF r_k=full deg={deg_ref} E={e_ref:.8} iters={} rms={:.2e}", scc.n_iters, scc.rms);

    for &k in &[32usize, 48, 64, 96, 128] {
        let mask = build_topk_mask(&k_ref, k);
        let deg = max_deg(&mask, n);
        let mut c2 = cfg.clone();
        c2.mask_kz = Some(mask);
        match SparseDftb::with_config(sk.clone(), &sk_dir, mol.species.clone(), mol.coords.clone(), c2) {
            Err(e) => eprintln!("top{k}: engine build failed: {e}"),
            Ok(mut e2) => match e2.scc(60, 1e-5) {
                Err(e) => eprintln!("top{k} (deg {deg}): *** SCC FAILED: {e}"),
                Ok(s) => {
                    let e = e2.energy().unwrap();
                    let f = e2.forces().unwrap_or_else(|e| panic!("top{k} forces: {e}"));
                    let fmax = f.forces.iter().flatten().fold(0.0f64, |a, &x| a.max(x.abs()));
                    eprintln!("top{k} (deg {deg}): E={:.8} dE={:+.2} mHa max|F|={:.4} iters={} rms={:.2e}",
                        e, (e - e_ref) * 1e3, fmax, s.n_iters, s.rms);
                }
            },
        }
    }
}
