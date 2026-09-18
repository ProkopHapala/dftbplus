//! Production SparseDftb loop: init once, reuse kernels/buffers across SCC and FIRE.
//!
//! Not a throwaway per-test OpenCL context. NVIDIA required.
//!
//!   RUST_DFTB_SK_DIR=... cargo test --test sparse_dftb -- --nocapture --test-threads=1

use rust_dftb::load_sk_for_species;
use rust_dftb::methods::sparse::harness::require_sih_sk_dir;
use rust_dftb::methods::sparse::{SparseDftb, SparseDftbConfig};

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
        let dx = h[0] - si[0];
        let dy = h[1] - si[1];
        let dz = h[2] - si[2];
        s += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    s / 4.0
}

#[test]
fn test_sparse_dftb_sih4_reuse_scc_and_fire() {
    let dir = require_sih_sk_dir();
    let (sp, xyz) = sih4();
    let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("load SK: {e}"));
    let mut eng =
        SparseDftb::new(sk, &dir, sp, xyz).unwrap_or_else(|e| panic!("SparseDftb::new: {e}"));
    let s1 = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC1: {e}"));
    let e1 = eng.energy().unwrap();
    eprintln!(
        "SiH4 SparseDftb SCC1: E={e1:.8} rms={:.3e} iters={} Tr(KS)={:.6} R_I={:.3e}",
        s1.rms, s1.n_iters, s1.tr_ks, s1.r_i
    );
    assert!(e1.is_finite(), "E non-finite: {e1}");
    assert!(s1.rms < 1e-4, "SiH4 SCC diverged: rms={:.3e}", s1.rms);
    assert!((s1.tr_ks - 4.0).abs() < 0.05, "Tr(KS)={} != 4", s1.tr_ks);

    let s2 = eng
        .scc(80, 1e-5)
        .unwrap_or_else(|e| panic!("SCC2 reuse: {e}"));
    let e2 = eng.energy().unwrap();
    eprintln!(
        "SiH4 SparseDftb SCC2 (reuse): E={e2:.8} rms={:.3e} iters={}",
        s2.rms, s2.n_iters
    );
    assert!(
        (e2 - e1).abs() < 1e-5,
        "reuse SCC |dE|={:.3e} — pipeline is not persistent/warm",
        (e2 - e1).abs()
    );

    let f = eng.forces().unwrap_or_else(|e| panic!("forces: {e}"));
    let mut max_f = 0.0f64;
    for fi in &f.forces {
        for &c in fi {
            max_f = max_f.max(c.abs());
        }
    }
    eprintln!("SiH4 SparseDftb |F|_max={max_f:.4e}");
    assert!(max_f.is_finite(), "forces non-finite");

    let r0 = mean_sih(eng.coords());
    let mf = eng.fire_step(1e-2).unwrap_or_else(|e| panic!("FIRE: {e}"));
    let s3 = eng
        .scc(80, 1e-5)
        .unwrap_or_else(|e| panic!("SCC after FIRE: {e}"));
    let e3 = eng.energy().unwrap();
    let r1 = mean_sih(eng.coords());
    eprintln!("SiH4 SparseDftb after 1 FIRE: E={e3:.8} max|F|={mf:.4e} rms={:.3e} <SiH> {r0:.4}→{r1:.4} Å", s3.rms);
    assert!(
        e3.is_finite() && e3.abs() < 20.0,
        "FIRE step exploded: E={e3}"
    );
    assert!(
        (e3 - e1).abs() < 1.0,
        "FIRE moved energy by {:.3} Ha (E0={e1:.8} E={e3:.8})",
        (e3 - e1).abs()
    );
    assert!(
        r1 > 1.2 && r1 < 2.0,
        "FIRE Si–H mean {r1:.3} Å left physical window"
    );

    let mf_md = eng.md_step(0.05).unwrap_or_else(|e| panic!("MD: {e}"));
    let s4 = eng
        .scc(80, 1e-5)
        .unwrap_or_else(|e| panic!("SCC after MD: {e}"));
    let e4 = eng.energy().unwrap();
    eprintln!(
        "SiH4 SparseDftb after 1 MD: E={e4:.8} max|F|={mf_md:.4e} rms={:.3e}",
        s4.rms
    );
    assert!(
        e4.is_finite() && e4.abs() < 20.0,
        "MD step exploded: E={e4}"
    );
}

// ============================================================================
// R17: GPU pair-physics kernels (sparse_hs.cl) vs the explicit CPU reference.
// Two engines on the same geometry — cfg.cpu_pair selects the backend, it is
// NOT a fallback. Compares H/S BSR values, E_rep (inside e_tot), and every
// force component at the same converged state.
// ============================================================================

fn max_diff3(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(x, y)| x.iter().zip(y.iter()).map(|(u, v)| (u - v).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn test_sparse_pair_gpu_vs_cpu_parity() {
    let dir = require_sih_sk_dir();
    let (sp, xyz) = sih4();
    let mk = |cpu: bool| {
        let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("load SK: {e}"));
        let cfg = SparseDftbConfig {
            cpu_pair: Some(cpu),
            ..Default::default()
        };
        SparseDftb::with_config(sk, &dir, sp.clone(), xyz.clone(), cfg)
            .unwrap_or_else(|e| panic!("SparseDftb(cpu_pair={cpu}): {e}"))
    };
    let mut cpu = mk(true);
    let mut gpu = mk(false);

    // H/S assembly parity — f32 kernel vs f64→f32 host reference.
    let dh = {
        let (ha, hb) = (cpu.h_bsr().unwrap(), gpu.h_bsr().unwrap());
        assert_eq!(ha.values.len(), hb.values.len(), "H0 nblock mismatch");
        ha.values
            .iter()
            .zip(hb.values.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max)
    };
    let ds = {
        let (sa, sb) = (cpu.s_bsr().unwrap(), gpu.s_bsr().unwrap());
        sa.values
            .iter()
            .zip(sb.values.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max)
    };
    eprintln!("[pair parity] set_coords: max|dH0|={dh:.3e}  max|dS|={ds:.3e}");
    assert!(dh < 2e-4, "H0 block parity {dh:.3e}"); // f32 eval vs f64→f32 cast
    assert!(ds < 2e-5, "S block parity {ds:.3e}");

    // Converged state on both engines (same GPU solver; pair path differs).
    let sc = cpu.scc(80, 1e-5).unwrap_or_else(|e| panic!("CPU scc: {e}"));
    let sg = gpu.scc(80, 1e-5).unwrap_or_else(|e| panic!("GPU scc: {e}"));
    assert!(
        sc.rms < 1e-4 && sg.rms < 1e-4,
        "SCC did not converge: {sc:?} {sg:?}"
    );
    let de = (cpu.energy().unwrap() - gpu.energy().unwrap()).abs();
    eprintln!(
        "[pair parity] E_cpu={:.8} E_gpu={:.8} |dE|={de:.3e}",
        cpu.energy().unwrap(),
        gpu.energy().unwrap()
    );
    assert!(
        de < 5e-4,
        "energy parity |dE|={de:.3e} (E_rep path included)"
    );

    let fc = cpu.forces().unwrap_or_else(|e| panic!("CPU forces: {e}"));
    let fg = gpu.forces().unwrap_or_else(|e| panic!("GPU forces: {e}"));
    for (name, a, b, tol) in [
        ("non_scc", &fc.non_scc, &fg.non_scc, 5e-3),
        ("scc_shift", &fc.scc_shift, &fg.scc_shift, 5e-3),
        ("repulsive", &fc.repulsive, &fg.repulsive, 5e-4),
        ("scc_dc", &fc.scc_dc, &fg.scc_dc, 5e-4),
        ("total", &fc.forces, &fg.forces, 5e-3),
    ] {
        let d = max_diff3(a, b);
        eprintln!("[pair parity] forces {name}: max|dF|={d:.3e} (tol {tol:.0e})");
        assert!(
            d < tol,
            "GPU-vs-CPU {name} force parity {d:.3e} > {tol:.0e}"
        );
    }

    // Displaced geometry — different pair distances exercise the taper and
    // a different B-spline stencil region.
    let mut xyz2 = xyz.clone();
    xyz2[1][0] += 0.07;
    xyz2[3][1] -= 0.05;
    cpu.set_coords(&xyz2)
        .unwrap_or_else(|e| panic!("CPU set_coords: {e}"));
    gpu.set_coords(&xyz2)
        .unwrap_or_else(|e| panic!("GPU set_coords: {e}"));
    let dh2 = {
        let (ha, hb) = (cpu.h_bsr().unwrap(), gpu.h_bsr().unwrap());
        ha.values
            .iter()
            .zip(hb.values.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max)
    };
    let ds2 = {
        let (sa, sb) = (cpu.s_bsr().unwrap(), gpu.s_bsr().unwrap());
        sa.values
            .iter()
            .zip(sb.values.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max)
    };
    eprintln!("[pair parity] displaced: max|dH0|={dh2:.3e}  max|dS|={ds2:.3e}");
    assert!(
        dh2 < 2e-4 && ds2 < 2e-5,
        "displaced H/S parity {dh2:.3e} {ds2:.3e}"
    );
    cpu.scc(80, 1e-5)
        .unwrap_or_else(|e| panic!("CPU scc2: {e}"));
    gpu.scc(80, 1e-5)
        .unwrap_or_else(|e| panic!("GPU scc2: {e}"));
    let fc2 = cpu.forces().unwrap_or_else(|e| panic!("CPU forces2: {e}"));
    let fg2 = gpu.forces().unwrap_or_else(|e| panic!("GPU forces2: {e}"));
    let d2 = max_diff3(&fc2.forces, &fg2.forces);
    eprintln!("[pair parity] displaced forces total: max|dF|={d2:.3e}");
    assert!(d2 < 5e-3, "displaced force parity {d2:.3e}");
}

// ============================================================================
// F1 (manifest §F.1): batched frozen-orbital evals — one launch per stage
// over `evals` replica slots (kernel dim1 = JobId) vs the scalar
// restore→set_coords→forces_frozen reference. The batched path shares the
// central K0/W0/dq0 read-only and must be BIT-IDENTICAL to the scalar
// chain (same per-work-item math, no atomics, no reordering).
// ============================================================================

#[test]
fn test_sparse_frozen_batch_parity() {
    let dir = require_sih_sk_dir();
    let (sp, xyz) = sih4();
    let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("load SK: {e}"));
    let mut eng =
        SparseDftb::new(sk, &dir, sp, xyz).unwrap_or_else(|e| panic!("SparseDftb::new: {e}"));
    eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC: {e}"));
    eng.snapshot_electronic_state()
        .unwrap_or_else(|e| panic!("snapshot: {e}"));
    if eng.cpu_pair() {
        panic!("test requires the GPU pair path — RUST_DFTB_SPARSE_CPU is set");
    }
    let h = 0.01f64;
    let x0 = eng.coords().to_vec();
    let n = eng.n_atom();
    // All (atom, axis, ±h) evals — 30 one-shot jobs, JobId = eval index.
    let evals: Vec<(usize, usize, f64)> = (0..n)
        .flat_map(|a| (0..3).flat_map(move |ax| [(a, ax, 1.0), (a, ax, -1.0)]))
        .collect();
    // Scalar reference — the driver's sequential frozen chain.
    let mut work = x0.clone();
    let mut f_ref: Vec<Vec<[f64; 3]>> = Vec::with_capacity(evals.len());
    for &(a, ax, sign) in &evals {
        eng.restore_central_state()
            .unwrap_or_else(|e| panic!("restore: {e}"));
        work[a][ax] = x0[a][ax] + sign * h;
        eng.set_coords(&work)
            .unwrap_or_else(|e| panic!("set_coords: {e}"));
        work[a][ax] = x0[a][ax];
        f_ref.push(
            eng.forces_frozen()
                .unwrap_or_else(|e| panic!("forces_frozen: {e}"))
                .forces,
        );
    }
    // Batched — all 30 evals in ONE call (cap grown to 30). NOTE: the
    // scalar loop left eng.coords at the LAST displaced geometry — the
    // explicit x0 arg is what makes the batched base unambiguous.
    let f_b = eng
        .forces_frozen_batch(&x0, &evals, h)
        .unwrap_or_else(|e| panic!("forces_frozen_batch: {e}"));
    assert_eq!(f_b.len(), evals.len(), "batch result count mismatch");
    let mut dmax = 0.0f64;
    for (e, (fr, fb)) in f_ref.iter().zip(f_b.iter()).enumerate() {
        let d = max_diff3(fr, &fb.forces);
        dmax = dmax.max(d);
        assert_eq!(
            d, 0.0,
            "eval {e}: batched vs scalar max|dF|={d:.3e} — replica slot \
             math must be bit-identical to the scalar chain"
        );
    }
    eprintln!(
        "[frozen batch] {} evals: max|dF|={dmax:.3e} (bit-identical)",
        evals.len()
    );
    // Partial reuse — a smaller second batch must not corrupt cap state.
    let sub: Vec<(usize, usize, f64)> = evals[..7].to_vec();
    let f_s = eng
        .forces_frozen_batch(&x0, &sub, h)
        .unwrap_or_else(|e| panic!("forces_frozen_batch subset: {e}"));
    assert_eq!(f_s.len(), 7);
    for (e, (fr, fb)) in f_ref.iter().take(7).zip(f_s.iter()).enumerate() {
        let d = max_diff3(fr, &fb.forces);
        assert_eq!(d, 0.0, "subset eval {e}: max|dF|={d:.3e}");
    }
    eprintln!("[frozen batch] subset reuse (7 evals): bit-identical");
}
