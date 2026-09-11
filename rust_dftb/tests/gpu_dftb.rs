//! Production GpuDftb loop: init once, reuse kernels/buffers across SCC and FIRE.
//!
//! Not a throwaway per-test OpenCL context. NVIDIA required.
//!
//!   RUST_DFTB_SK_DIR=... cargo test --test gpu_dftb -- --nocapture --test-threads=1

use rust_dftb::load_sk_for_species;
use rust_dftb::qmqm::GpuDftb;
use std::path::Path;

const DEFAULT_SK: &str = "/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1";

fn sk_dir() -> String {
    let dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| DEFAULT_SK.to_string());
    assert!(Path::new(&dir).is_dir(), "SK directory missing: {dir}");
    assert!(Path::new(&dir).join("H-H.skf").is_file(), "H-H.skf missing in {dir}");
    dir
}

fn h2o() -> (Vec<String>, Vec<[f64; 3]>) {
    (vec!["O".into(), "H".into(), "H".into()],
     vec![[0.0, 0.0, 0.0], [-0.7580632005, 0.6358101311, 0.0], [0.7580632005, 0.6358101311, 0.0]])
}

#[test]
fn test_gpu_dftb_h2o_reuse_scc_and_fire() {
    let dir = sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng = GpuDftb::new(sk, &dir, sp, xyz, 1).unwrap_or_else(|e| panic!("GpuDftb::new: {e}"));
    eprintln!("[gpu] caps={}", eng.rt.caps().name);
    let s1 = eng.scc(100, 1e-6).unwrap();
    let ev1 = eng.eval(true).unwrap_or_else(|e| panic!("eval: {e}"));
    let e1 = ev1.energy[0];
    let f = ev1.forces.as_ref().expect("eval(true) returns forces");
    eprintln!("H2O GpuDftb SCC1: E={e1:.8} rms={:.3e} iters={} stalled={}", s1.rms, s1.n_iters, s1.stalled);
    assert!(e1.is_finite(), "E non-finite: {e1}");
    assert!(s1.rms < 1e-4, "H2O SCC diverged: rms={:.3e}", s1.rms);
    // Same object, same coords — must not rebuild runtime. Energy must match.
    let s2 = eng.scc(100, 1e-6).unwrap();
    let e2 = eng.eval(false).unwrap().energy[0];
    eprintln!("H2O GpuDftb SCC2 (reuse): E={e2:.8} rms={:.3e} iters={}", s2.rms, s2.n_iters);
    assert!((e2 - e1).abs() < 1e-5, "reuse SCC |dE|={:.3e} — pipeline is not persistent/warm", (e2 - e1).abs());
    let mut max_f = 0.0f32;
    for &x in f.iter() { max_f = max_f.max(x.abs()); }
    eprintln!("H2O GpuDftb |F|_max={max_f:.4e}");
    assert!(max_f.is_finite(), "forces non-finite");
    // One FIRE step on the same engine (set_coords + scc, no compile). 0.1 Å cap.
    let mf = eng.fire_step(1e-2).unwrap();
    let s3 = eng.scc(100, 1e-6).unwrap();
    let e3 = eng.eval(false).unwrap().energy[0];
    eprintln!("H2O GpuDftb after 1 FIRE: E={e3:.8} max|F|={mf:.4e} rms={:.3e}", s3.rms);
    assert!(e3.is_finite() && e3.abs() < 20.0, "FIRE step exploded: E={e3}");
    assert!((e3 - e1).abs() < 1.0, "FIRE moved energy by {:.3} Ha (E0={e1:.8} E={e3:.8}) — forces/repulsive likely wrong", (e3 - e1).abs());
    // MD step reuses the same kernels/buffers.
    let mf_md = eng.md_step(0.1).unwrap();
    let s4 = eng.scc(100, 1e-6).unwrap();
    let e4 = eng.eval(false).unwrap().energy[0];
    eprintln!("H2O GpuDftb after 1 MD: E={e4:.8} max|F|={mf_md:.4e} rms={:.3e}", s4.rms);
    assert!(e4.is_finite() && e4.abs() < 20.0, "MD step exploded: E={e4}");
}
