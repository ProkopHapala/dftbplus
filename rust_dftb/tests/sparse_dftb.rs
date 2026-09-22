//! Production SparseDftb loop: init once, reuse kernels/buffers across SCC and FIRE.
//!
//! Not a throwaway per-test OpenCL context. NVIDIA required.
//!
//!   RUST_DFTB_SK_DIR=... cargo test --test sparse_dftb -- --nocapture --test-threads=1

use rust_dftb::load_sk_for_species;
use rust_dftb::methods::sparse::harness::require_sih_sk_dir;
use rust_dftb::methods::sparse::{GeomOutcome, GeomStep, SparseDftb, SparseDftbConfig};

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

// ============================================================================
// F5a (manifest §F.1): batched fixed-iteration DMM-lite evals — each
// replica runs the same predetermined recipe (assemble→V→H_scc→B=Z·H→
// n_dmm DMM steps→W→contract) on its own matrix slabs. Bit-parity target
// vs the scalar lite chain (restore→set_coords→scc_fixedq[lite]→forces):
// identical kernel sequence per replica, deterministic row reductions.
// ============================================================================

#[test]
fn test_sparse_dmm_batch_parity() {
    // Scalar lite recipe envs (read inside scc_fixedq): lite + dmupd
    // selects the fixed-cost branch; DMM/ETA match the batch args.
    std::env::set_var("RUST_DFTB_VIB_LITE", "1");
    std::env::set_var("RUST_DFTB_VIB_DMUPD", "1");
    std::env::set_var("RUST_DFTB_VIB_DMM", "4");
    std::env::set_var("RUST_DFTB_VIB_DMM_ETA", "8.0");
    std::env::set_var("RUST_DFTB_VIB_NSMAX", "0");
    std::env::remove_var("RUST_DFTB_VIB_LINEAR");
    std::env::remove_var("RUST_DFTB_VIB_SEED");
    std::env::remove_var("RUST_DFTB_VIB_METRIC");
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
    let evals: Vec<(usize, usize, f64)> = (0..n)
        .flat_map(|a| (0..3).flat_map(move |ax| [(a, ax, 1.0), (a, ax, -1.0)]))
        .collect();
    // Scalar lite reference — restore central state, displace, fixedq
    // (lite → refresh_b_zh + 4 DMM steps), forces.
    let mut work = x0.clone();
    let mut f_ref: Vec<Vec<[f64; 3]>> = Vec::with_capacity(evals.len());
    for &(a, ax, sign) in &evals {
        eng.restore_central_state()
            .unwrap_or_else(|e| panic!("restore: {e}"));
        work[a][ax] = x0[a][ax] + sign * h;
        eng.set_coords(&work)
            .unwrap_or_else(|e| panic!("set_coords: {e}"));
        work[a][ax] = x0[a][ax];
        eng.scc_fixedq()
            .unwrap_or_else(|e| panic!("scc_fixedq lite: {e}"));
        f_ref.push(
            eng.forces()
                .unwrap_or_else(|e| panic!("forces: {e}"))
                .forces,
        );
    }
    // Batched — same recipe: n_ns=0 (shared central Z), 4 DMM steps,
    // eta=8.0. One call, JobId-indexed outputs.
    let f_b = eng
        .forces_dmm_batch(&x0, &evals, h, 0, 4, 8.0)
        .unwrap_or_else(|e| panic!("forces_dmm_batch: {e}"));
    assert_eq!(f_b.len(), evals.len(), "batch result count mismatch");
    let mut dmax = 0.0f64;
    for (e, (fr, fb)) in f_ref.iter().zip(f_b.iter()).enumerate() {
        let d = max_diff3(fr, &fb.forces);
        dmax = dmax.max(d);
        assert_eq!(
            d, 0.0,
            "eval {e}: dmm-batch vs scalar lite max|dF|={d:.3e} — replica \
             slot math must be bit-identical to the scalar chain"
        );
    }
    eprintln!(
        "[dmm batch] {} evals: max|dF|={dmax:.3e} (bit-identical)",
        evals.len()
    );
    // Subset reuse — a smaller second batch must not corrupt cap state.
    let sub: Vec<(usize, usize, f64)> = evals[..7].to_vec();
    let f_s = eng
        .forces_dmm_batch(&x0, &sub, h, 0, 4, 8.0)
        .unwrap_or_else(|e| panic!("forces_dmm_batch subset: {e}"));
    assert_eq!(f_s.len(), 7);
    for (e, (fr, fb)) in f_ref.iter().take(7).zip(f_s.iter()).enumerate() {
        let d = max_diff3(fr, &fb.forces);
        assert_eq!(d, 0.0, "subset eval {e}: max|dF|={d:.3e}");
    }
    eprintln!("[dmm batch] subset reuse (7 evals): bit-identical");
}

/// One 0.02 Å geometry step must reuse the stored projector (short DMM),
/// and the resulting energy and forces must match a cold SCC at the same
/// geometry. SiH4 is a complete mask, so this is a wiring check, not the
/// R10 truncation floor.
#[test]
fn test_sparse_dftb_sih4_warm_step_vs_cold() {
    let dir = require_sih_sk_dir();
    let (sp, xyz) = sih4();
    let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("load SK: {e}"));
    let mut warm =
        SparseDftb::new(sk, &dir, sp.clone(), xyz.clone()).unwrap_or_else(|e| panic!("warm new: {e}"));
    let t0 = std::time::Instant::now();
    let s0 = warm.scc(80, 1e-5).unwrap_or_else(|e| panic!("cold SCC: {e}"));
    eprintln!(
        "SiH4 cold SCC: iters={} rms={:.3e} Tr={:.6} {:.1} ms",
        s0.n_iters,
        s0.rms,
        s0.tr_ks,
        t0.elapsed().as_secs_f64() * 1e3
    );

    let mut xyz2 = xyz.clone();
    xyz2[1][0] += 0.02;
    let t1 = std::time::Instant::now();
    warm.set_coords(&xyz2).unwrap_or_else(|e| panic!("set_coords: {e}"));
    let sw = warm
        .scc(80, 1e-5)
        .unwrap_or_else(|e| panic!("warm SCC: {e}"));
    let ms_w = t1.elapsed().as_secs_f64() * 1e3;
    let ew = warm.energy().unwrap();
    let fw = warm.forces().unwrap_or_else(|e| panic!("warm forces: {e}"));
    eprintln!(
        "SiH4 warm step: mixes={} rms={:.3e} Tr={:.6} R_H={:.3e} E={ew:.8} {ms_w:.1} ms",
        sw.n_iters, sw.rms, sw.tr_ks, sw.r_h
    );
    assert!(sw.n_iters <= 3, "warm step expanded: mixes={}", sw.n_iters);
    assert!((sw.tr_ks - 4.0).abs() < 0.05, "Tr(KS)={}", sw.tr_ks);
    assert!(sw.r_h.is_finite() && sw.r_h < 5e-4, "R_H={:.3e}", sw.r_h);

    let sk2 = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("load SK2: {e}"));
    let mut cold =
        SparseDftb::new(sk2, &dir, sp, xyz2).unwrap_or_else(|e| panic!("cold new: {e}"));
    let t2 = std::time::Instant::now();
    let sc = cold.scc(80, 1e-5).unwrap_or_else(|e| panic!("cold displaced SCC: {e}"));
    let ec = cold.energy().unwrap();
    let fc = cold.forces().unwrap_or_else(|e| panic!("cold forces: {e}"));
    eprintln!(
        "SiH4 cold displaced: iters={} rms={:.3e} E={ec:.8} {:.1} ms",
        sc.n_iters,
        sc.rms,
        t2.elapsed().as_secs_f64() * 1e3
    );
    let de = (ew - ec).abs();
    let mut df = 0.0f64;
    let mut fmax = 0.0f64;
    for (a, b) in fw.forces.iter().zip(fc.forces.iter()) {
        for c in 0..3 {
            df = df.max((a[c] - b[c]).abs());
            fmax = fmax.max(b[c].abs());
        }
    }
    eprintln!("SiH4 warm vs cold: |dE|={de:.3e} Ha  max|dF|={df:.3e}  cold max|F|={fmax:.3e}");
    // One 4-step block on a complete mask. Measured 2026-09-22: |dE|≈5e-3 Ha,
    // max|dF|≈1e-3. These bounds catch a wrong subspace, not the f32 floor.
    assert!(de < 1e-2, "|dE|={de:.3e} Ha");
    assert!(df < 2e-3, "max|dF|={df:.3e}");

    // FIRE from a converged state. Steps are much smaller than the 0.02 Å
    // probe; each one must stay a single DMM block with a stable trace.
    for k in 0..3 {
        let t = std::time::Instant::now();
        let mf = cold.fire_step(1e-8).unwrap_or_else(|e| panic!("FIRE {k}: {e}"));
        let s = cold.scc(80, 1e-5).unwrap_or_else(|e| panic!("FIRE SCC {k}: {e}"));
        eprintln!(
            "SiH4 FIRE {k}: max|F|={mf:.3e} mixes={} rms={:.3e} Tr={:.6} R_H={:.3e} {:.1} ms",
            s.n_iters,
            s.rms,
            s.tr_ks,
            s.r_h,
            t.elapsed().as_secs_f64() * 1e3
        );
        assert!(s.n_iters <= 3, "FIRE {k} expanded: mixes={}", s.n_iters);
        assert!((s.tr_ks - 4.0).abs() < 2e-3, "FIRE {k} Tr={}", s.tr_ks);
        assert!(s.r_h < 5e-4, "FIRE {k} R_H={:.3e}", s.r_h);
    }
}

/// Five FIRE steps of each Warm_Geometry_DM variant against a cold SCC
/// of the same geometry. A restart that matches the cold solve passes.
/// An accepted step that misses the energy or the force does not.
fn run_sih4_geom_variant(step: GeomStep, name: &str) -> (usize, usize) {
    let dir = require_sih_sk_dir();
    let (sp, xyz) = sih4();
    let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("{name} SK: {e}"));
    let mut warm =
        SparseDftb::new(sk, &dir, sp.clone(), xyz.clone()).unwrap_or_else(|e| panic!("{name} warm: {e}"));
    warm.set_geom_step(step);
    // dt = 0.02 from rest moves an atom by ~2e-5 Å (½ F dt²), which
    // leaves the old kernel inside the certificate and tests nothing.
    // 0.5 Å gives a first step of ~0.01 Å, the dense geometry-step scale.
    warm.set_fire_dt(0.5);
    warm.scc(80, 1e-5).unwrap_or_else(|e| panic!("{name} warm init: {e}"));
    let sk2 = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("{name} SK2: {e}"));
    let mut cold =
        SparseDftb::new(sk2, &dir, sp, xyz).unwrap_or_else(|e| panic!("{name} cold: {e}"));
    cold.scc(80, 1e-5).unwrap_or_else(|e| panic!("{name} cold init: {e}"));
    let mut n_acc = 0usize;
    let mut n_restart = 0usize;
    for k in 0..5 {
        let t = std::time::Instant::now();
        let xyz_before = warm.coords().to_vec();
        let mf = warm.fire_step(1e-8).unwrap_or_else(|e| panic!("{name} FIRE {k}: {e}"));
        let mut d_r = 0.0f64;
        for (a, b) in xyz_before.iter().zip(warm.coords().iter()) {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            d_r = d_r.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        let xyz_k = warm.coords().to_vec();
        let sw = warm.scc(80, 1e-5).unwrap_or_else(|e| panic!("{name} warm scc {k}: {e}"));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let outcome = warm.geom_outcome();
        match outcome {
            GeomOutcome::Accept => n_acc += 1,
            GeomOutcome::Restart => n_restart += 1,
            GeomOutcome::Cold => {}
        }
        let ew = warm.energy().unwrap_or_else(|e| panic!("{name} Ew {k}: {e}"));
        let fw = warm.forces().unwrap_or_else(|e| panic!("{name} Fw {k}: {e}"));
        cold.set_coords(&xyz_k).unwrap_or_else(|e| panic!("{name} cold coords {k}: {e}"));
        cold.forget_projector();
        let sc = cold.scc(80, 1e-5).unwrap_or_else(|e| panic!("{name} cold scc {k}: {e}"));
        let ec = cold.energy().unwrap_or_else(|e| panic!("{name} Ec {k}: {e}"));
        let fc = cold.forces().unwrap_or_else(|e| panic!("{name} Fc {k}: {e}"));
        let de = (ew - ec).abs();
        let mut df = 0.0f64;
        let mut fmax = 0.0f64;
        for (a, b) in fw.forces.iter().zip(fc.forces.iter()) {
            for c in 0..3 {
                df = df.max((a[c] - b[c]).abs());
                fmax = fmax.max(b[c].abs());
            }
        }
        let how = match outcome {
            GeomOutcome::Accept => "accept",
            GeomOutcome::Restart => "restart",
            GeomOutcome::Cold => "cold",
        };
        eprintln!(
            "{name}  step {k}  {how}  |dR|={d_r:.4}  E={ew:.8}  Eref={ec:.8}  dE={de:.3e}  max|F|={mf:.3e}  max|dF|={df:.3e}  Fref={fmax:.3e}  Tr={:.6}  R_H={:.3e}  {ms:.1} ms",
            sw.tr_ks, sw.r_h
        );
        let tau = (sw.tr_ks as f64 - 4.0).abs();
        assert!(tau < 0.02, "{name} step {k} τ={tau:.4e}");
        assert!(de < 1e-3, "{name} step {k} |dE|={de:.3e} Ha");
        let f_tol = (0.02 * fmax).max(5e-4);
        assert!(df < f_tol, "{name} step {k} max|dF|={df:.3e} tol={f_tol:.3e} Fref={fmax:.3e}");
        let _ = sc;
    }
    eprintln!("{name}  summary  accept={n_acc}  restart={n_restart}  of 5");
    (n_acc, n_restart)
}

fn report_geom(name: &str, n_acc: usize, n_restart: usize) {
    if n_acc >= 3 {
        eprintln!("{name}  USEFUL  {n_acc} accepted steps");
    } else {
        eprintln!("{name}  NOT USEFUL  accept={n_acc} restart={n_restart}");
    }
}

#[test]
fn test_sparse_dftb_sih4_geom_v2() {
    let (a, r) = run_sih4_geom_variant(GeomStep::Xtr, "V2");
    report_geom("V2", a, r);
}

#[test]
fn test_sparse_dftb_sih4_geom_v1() {
    let (a, r) = run_sih4_geom_variant(GeomStep::XtrMcWeeny, "V1");
    report_geom("V1", a, r);
}

#[test]
fn test_sparse_dftb_sih4_geom_v3() {
    let (a, r) = run_sih4_geom_variant(GeomStep::TrustDmm, "V3");
    report_geom("V3", a, r);
}

#[test]
fn test_sparse_dftb_sih4_geom_v4() {
    let (a, r) = run_sih4_geom_variant(GeomStep::XtrThenTrust, "V4");
    report_geom("V4", a, r);
}

/// Three successive 0.1 Å moves of one hydrogen. The warm step is kept
/// even when R_H is above the cold gate. It has to be faster than a cold
/// SCC of the same geometry, and the trace after one McWeeny has to stay
/// within 0.05 of Nocc.
fn run_sih4_bold(step: GeomStep, name: &str) {
    let dir = require_sih_sk_dir();
    let (sp, xyz0) = sih4();
    let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("{name} SK: {e}"));
    let mut warm = SparseDftb::new(sk, &dir, sp.clone(), xyz0.clone())
        .unwrap_or_else(|e| panic!("{name} warm: {e}"));
    warm.set_geom_step(step);
    warm.scc(80, 1e-5).unwrap_or_else(|e| panic!("{name} warm init: {e}"));
    let sk2 = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("{name} SK2: {e}"));
    let mut cold = SparseDftb::new(sk2, &dir, sp, xyz0.clone())
        .unwrap_or_else(|e| panic!("{name} cold: {e}"));
    cold.scc(80, 1e-5).unwrap_or_else(|e| panic!("{name} cold init: {e}"));
    let mut xyz = xyz0;
    for k in 0..3 {
        xyz[1][0] += 0.1;
        let t_w = std::time::Instant::now();
        warm.set_coords(&xyz).unwrap_or_else(|e| panic!("{name} set {k}: {e}"));
        let sw = warm.scc(80, 1e-5).unwrap_or_else(|e| panic!("{name} warm {k}: {e}"));
        let ms_w = t_w.elapsed().as_secs_f64() * 1e3;
        let ew = warm.energy().unwrap_or_else(|e| panic!("{name} Ew {k}: {e}"));
        let fw = warm.forces().unwrap_or_else(|e| panic!("{name} Fw {k}: {e}"));
        let t_c = std::time::Instant::now();
        cold.set_coords(&xyz).unwrap_or_else(|e| panic!("{name} cold set {k}: {e}"));
        cold.forget_projector();
        cold.scc(80, 1e-5).unwrap_or_else(|e| panic!("{name} cold {k}: {e}"));
        let ms_c = t_c.elapsed().as_secs_f64() * 1e3;
        let ec = cold.energy().unwrap_or_else(|e| panic!("{name} Ec {k}: {e}"));
        let fc = cold.forces().unwrap_or_else(|e| panic!("{name} Fc {k}: {e}"));
        let de = (ew - ec).abs();
        let mut df = 0.0f64;
        let mut fmax = 0.0f64;
        for (a, b) in fw.forces.iter().zip(fc.forces.iter()) {
            for c in 0..3 {
                df = df.max((a[c] - b[c]).abs());
                fmax = fmax.max(b[c].abs());
            }
        }
        let tau = (sw.tr_ks as f64 - 4.0).abs();
        eprintln!(
            "{name}  step {k}  |dR|=0.10  E={ew:.6}  Eref={ec:.6}  dE={de:.3e}  max|dF|={df:.3e}  Fref={fmax:.3e}  τ={tau:.4e}  R_H={:.3e}  warm {ms_w:.1} ms  cold {ms_c:.1} ms",
            sw.r_h
        );
        assert!(tau < 0.05, "{name} step {k} trace not repaired: τ={tau:.4e}");
        assert!(de.is_finite() && df.is_finite(), "{name} step {k} non-finite");
        assert!(ms_w < ms_c, "{name} step {k} warm {ms_w:.1} ms not faster than cold {ms_c:.1} ms");
    }
}

#[test]
fn test_sparse_dftb_sih4_bold_xtr() {
    run_sih4_bold(GeomStep::BoldXtr, "B1");
}

#[test]
fn test_sparse_dftb_sih4_bold_dmm() {
    run_sih4_bold(GeomStep::BoldDmm, "B2");
}

#[test]
fn test_sparse_dftb_sih4_bold_xtr_dmm() {
    run_sih4_bold(GeomStep::BoldXtrDmm, "B3");
}

/// FIRE with the kept kernel (extrapolate, two η=8 commutator steps, one
/// McWeeny). Energy must fall, Tr(KS) must stay on Nocc, and a step must
/// be cheaper than a cold SCC.
#[test]
fn test_sparse_dftb_sih4_bold_fire() {
    let dir = require_sih_sk_dir();
    let (sp, xyz) = sih4();
    let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("SK: {e}"));
    let mut eng = SparseDftb::new(sk, &dir, sp.clone(), xyz.clone()).unwrap_or_else(|e| panic!("new: {e}"));
    eng.set_geom_step(GeomStep::BoldXtrDmm);
    // First step from rest is ½|F|dt². dt=1 reaches ~0.04 Å and later
    // steps hit the 0.1 Å cap as the velocity builds.
    eng.set_fire_dt(1.0);
    let s0 = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("init SCC: {e}"));
    let e0 = eng.energy().unwrap();
    eprintln!(
        "FIRE start  E={e0:.8}  Tr={:.6}  R_H={:.3e}  <SiH>={:.4}",
        s0.tr_ks, s0.r_h, mean_sih(eng.coords())
    );
    let mut e_prev = e0;
    let mut e_min = e0;
    let mut ms_sum = 0.0;
    let n_fire = 12;
    for k in 0..n_fire {
        let xyz_before = eng.coords().to_vec();
        let t = std::time::Instant::now();
        let mf = eng.fire_step(1e-6).unwrap_or_else(|e| panic!("FIRE {k}: {e}"));
        let mut d_r = 0.0f64;
        for (a, b) in xyz_before.iter().zip(eng.coords().iter()) {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            d_r = d_r.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        let s = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC {k}: {e}"));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        ms_sum += ms;
        let e = eng.energy().unwrap_or_else(|err| panic!("E {k}: {err}"));
        let tau = (s.tr_ks as f64 - 4.0).abs();
        eprintln!(
            "FIRE {k:2}  |dR|={d_r:.4}  E={e:.8}  dE={:+.3e}  max|F|={mf:.4e}  Tr={:.6}  τ={tau:.3e}  R_H={:.3e}  <SiH>={:.4}  {ms:.1} ms  {:?}",
            e - e_prev,
            s.tr_ks,
            s.r_h,
            mean_sih(eng.coords()),
            eng.geom_outcome()
        );
        assert!(tau < 0.05, "FIRE {k} charge left Nocc: τ={tau:.4e}");
        assert!(e.is_finite() && mf.is_finite(), "FIRE {k} non-finite");
        assert!(
            e < e_min + 0.02,
            "FIRE {k} energy jumped: E={e:.6} min={e_min:.6}"
        );
        e_min = e_min.min(e);
        e_prev = e;
    }
    let e_end = eng.energy().unwrap();
    eprintln!(
        "FIRE done  E {e0:.8} → {e_end:.8}  dE={:.4e}  mean step {:.1} ms",
        e_end - e0,
        ms_sum / n_fire as f64
    );
    assert!(e_end < e0, "energy did not fall: {e0:.8} → {e_end:.8}");

    let xyz_end = eng.coords().to_vec();
    let sk2 = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("SK2: {e}"));
    let mut cold = SparseDftb::new(sk2, &dir, sp, xyz_end).unwrap_or_else(|e| panic!("cold: {e}"));
    let t_c = std::time::Instant::now();
    cold.scc(80, 1e-5).unwrap_or_else(|e| panic!("cold final: {e}"));
    let ms_c = t_c.elapsed().as_secs_f64() * 1e3;
    let e_c = cold.energy().unwrap();
    let de_cold = (e_end - e_c).abs();
    eprintln!(
        "FIRE vs cold final  dE={de_cold:.3e}  cold {ms_c:.1} ms  warm mean {:.1} ms",
        ms_sum / n_fire as f64
    );
    assert!(
        de_cold < 5e-3,
        "final energy off the cold solve by {de_cold:.3e} Ha"
    );
}

fn write_xyz(path: &std::path::Path, species: &[String], coords: &[[f64; 3]], comment: &str) {
    let mut s = format!("{}\n{comment}\n", species.len());
    for (sp, c) in species.iter().zip(coords.iter()) {
        s.push_str(&format!("{sp:2} {:14.8} {:14.8} {:14.8}\n", c[0], c[1], c[2]));
    }
    std::fs::write(path, s).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// FIRE until max|F| < 1e-3 Ha/Å. Writes the input and the relaxed geometry.
#[test]
fn test_sparse_dftb_sih4_bold_fire_minimum() {
    let dir = require_sih_sk_dir();
    let (sp, xyz) = sih4();
    let out = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../debug/sparse_relax");
    std::fs::create_dir_all(&out).unwrap();
    write_xyz(&out.join("sih4_initial.xyz"), &sp, &xyz, "SiH4 initial");
    let sk = load_sk_for_species(&dir, &sp).unwrap_or_else(|e| panic!("SK: {e}"));
    let mut eng = SparseDftb::new(sk, &dir, sp.clone(), xyz).unwrap_or_else(|e| panic!("new: {e}"));
    eng.set_geom_step(GeomStep::BoldXtrDmm);
    eng.set_fire_dt(0.3);
    let s0 = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("init: {e}"));
    let e0 = eng.energy().unwrap();
    let f_tol = 1e-3;
    let mut max_f = {
        let f = eng.forces().unwrap();
        f.forces.iter().flatten().fold(0.0f64, |m, c| m.max(c.abs()))
    };
    eprintln!(
        "MIN start  E={e0:.8}  max|F|={max_f:.4e}  Tr={:.6}  <SiH>={:.4}",
        s0.tr_ks,
        mean_sih(eng.coords())
    );
    let mut e_prev = e0;
    let mut n = 0usize;
    let mut ms_sum = 0.0;
    for k in 0..60 {
        if max_f < f_tol {
            break;
        }
        n = k + 1;
        let t = std::time::Instant::now();
        eng.fire_step(f_tol).unwrap_or_else(|e| panic!("FIRE {k}: {e}"));
        let s = eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("SCC {k}: {e}"));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        ms_sum += ms;
        let e = eng.energy().unwrap();
        let f = eng.forces().unwrap();
        max_f = f.forces.iter().flatten().fold(0.0f64, |m, c| m.max(c.abs()));
        let tau = (s.tr_ks as f64 - 4.0).abs();
        eprintln!(
            "MIN {k:2}  E={e:.8}  dE={:+.3e}  max|F|={max_f:.4e}  Tr={:.6}  τ={tau:.3e}  <SiH>={:.4}  {ms:.1} ms",
            e - e_prev,
            s.tr_ks,
            mean_sih(eng.coords())
        );
        assert!(tau < 0.05, "MIN {k} τ={tau:.4e}");
        assert!(e.is_finite(), "MIN {k} non-finite energy");
        e_prev = e;
    }
    let e_end = eng.energy().unwrap();
    write_xyz(
        &out.join("sih4_final.xyz"),
        &sp,
        eng.coords(),
        &format!("SiH4 FIRE  E={e_end:.8}  max|F|={max_f:.4e}  steps={n}"),
    );
    eprintln!(
        "MIN done  steps={n}  E {e0:.8} → {e_end:.8}  max|F|={max_f:.4e}  <SiH>={:.4}  mean {:.1} ms  xyz {}",
        mean_sih(eng.coords()),
        if n > 0 { ms_sum / n as f64 } else { 0.0 },
        out.display()
    );
    assert!(n > 0 && n < 60, "did not reach max|F|<{f_tol:.1e} in {n} steps (F={max_f:.3e})");
    assert!(max_f < f_tol, "max|F|={max_f:.3e}");
    assert!(e_end < e0, "energy rose: {e0:.8} → {e_end:.8}");
}
