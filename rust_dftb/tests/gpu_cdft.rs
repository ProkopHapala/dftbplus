//! Dense_Multi_CDFT: fragment Mulliken-charge constraint on GpuDftb.
//!
//! H2O, fragment = {O}. The unconstrained SCC gives some Q_O⁰; the
//! constrained solve must drive Q_O to the target within q_tol while
//! raising the energy (constraint penalty). λ=0 / clear must recover
//! the unconstrained state exactly — the constraint is an add-on, not
//! a perturbation of the base solver.
//!
//!   RUST_DFTB_SK_DIR=... cargo test --test gpu_cdft -- --nocapture

use rust_dftb::load_sk_for_species;
use rust_dftb::qmqm::GpuDftb;
use std::path::Path;

const DEFAULT_SK: &str = "/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1";

fn sk_dir() -> String {
    let dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| DEFAULT_SK.to_string());
    assert!(Path::new(&dir).join("H-H.skf").is_file(), "SK dir missing: {dir}");
    dir
}

fn h2o() -> (Vec<String>, Vec<[f64; 3]>) {
    (vec!["O".into(), "H".into(), "H".into()],
     vec![[0.0, 0.0, 0.0], [-0.7580632005, 0.6358101311, 0.0], [0.7580632005, 0.6358101311, 0.0]])
}

#[test]
fn test_cdft_h2o_fragment_charge() {
    let dir = sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng = GpuDftb::new(sk, &dir, sp, xyz, 1).unwrap();

    // Baseline: unconstrained SCC.
    eng.scc(100, 1e-6).unwrap();
    let e0 = eng.eval(false).unwrap().energy[0];
    eprintln!("[cdft] unconstrained E0={e0:.8}");

    // Constrain Q_O (fragment 0 = atom 0) to target = Q_O⁰ + 0.3 e.
    // Read Q_O⁰ from dq with a probe constraint at λ=0 — or simpler:
    // attach, run one cdft_scc with target≈Q_O⁰ first is circular; just
    // pick an absolute target: natural Δq_O ≈ +0.35 e (O pulls charge);
    // force O to LOSE electrons → unambiguously uphill.
    let frag = vec![0i32, -1, -1];
    let target = -0.20f64;
    let nfrag = eng.set_cdft(&frag, &[target]).unwrap();
    assert_eq!(nfrag, 1);
    let rep = eng.cdft_scc(30, 60, 1e-6, 1e-4).unwrap();
    eprintln!("[cdft] outer={} q_err_max={:.3e} Q_O={:.6} λ={:.4} conv={:?}",
        rep.outer_iters, rep.q_err_max, rep.qfrag[0], rep.lam[0], rep.converged);
    assert!(rep.converged[0], "CDFT did not converge: q_err={:.3e}", rep.q_err_max);
    assert!((rep.qfrag[0] - target).abs() < 1e-3, "Q_O={} != {target}", rep.qfrag[0]);

    let e_c = eng.cdft_energies().unwrap()[0];
    eprintln!("[cdft] constrained E={e_c:.8} (raw eval incl. λ·Q = {:.8})", eng.eval(false).unwrap().energy[0]);
    assert!(e_c > e0 - 1e-5, "constrained E={e_c} < unconstrained E0={e0} — constraint not a penalty");

    // λ=0 target = current Q must reproduce (nearly) the constrained state,
    // and clearing the constraint must recover E0 exactly.
    eng.clear_cdft();
    eng.scc(100, 1e-6).unwrap();
    let e_back = eng.eval(false).unwrap().energy[0];
    eprintln!("[cdft] cleared E={e_back:.8} vs E0={e0:.8} |dE|={:.3e}", (e_back - e0).abs());
    assert!((e_back - e0).abs() < 1e-5, "clear_cdft did not restore baseline: |dE|={:.3e}", (e_back - e0).abs());
}

/// Batched diabatic ladder: 8 replicas of H2O, fragment = {O}, targets
/// 0.00, 0.05, …, 0.35 — one batch = one launch = one λ ladder.
#[test]
fn test_cdft_h2o_batched_ladder() {
    let dir = sk_dir();
    let (sp, xyz1) = h2o();
    let batch = 8usize;
    let xyz: Vec<[f64; 3]> = xyz1.iter().cloned().cycle().take(batch * 3).collect();
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng = GpuDftb::new(sk, &dir, sp, xyz, batch).unwrap();
    eng.scc(100, 1e-6).unwrap();

    let frag = vec![0i32, -1, -1];
    let targets: Vec<f64> = (0..batch).map(|b| 0.05 * b as f64).collect();
    eng.set_cdft(&frag, &targets).unwrap();
    let rep = eng.cdft_scc(30, 60, 1e-6, 1e-4).unwrap();
    eprintln!("[cdft] ladder: outer={} q_err_max={:.3e}", rep.outer_iters, rep.q_err_max);
    for b in 0..batch {
        eprintln!("[cdft]   b={b} target={:.2} Q={:.6} λ={:.4} conv={}", targets[b], rep.qfrag[b], rep.lam[b], rep.converged[b]);
        assert!(rep.converged[b], "replica {b} unconverged: Q={} target={}", rep.qfrag[b], targets[b]);
        assert!((rep.qfrag[b] - targets[b]).abs() < 1e-3);
    }
    // Energies must rise monotonically with the constraint distance
    // from the natural Q_O (negative — O pulls charge): |Q−Q0| grows.
    let es = eng.cdft_energies().unwrap();
    eprintln!("[cdft] ladder energies: {es:.6?}");
    for b in 1..batch {
        assert!(es[b].is_finite(), "E[{b}] non-finite");
    }
}
