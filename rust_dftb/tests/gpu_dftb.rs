//! Production GpuDftb loop: init once, reuse kernels/buffers across SCC and FIRE.
//!
//! Not a throwaway per-test OpenCL context. NVIDIA required.
//!
//!   RUST_DFTB_SK_DIR=... cargo test --test gpu_dftb -- --nocapture --test-threads=1

use rust_dftb::load_sk_for_species;
use rust_dftb::qmqm::{GpuDftb, SccStatus};
use std::path::Path;

const DEFAULT_SK: &str = "/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1";

fn sk_dir() -> String {
    let dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| DEFAULT_SK.to_string());
    assert!(Path::new(&dir).is_dir(), "SK directory missing: {dir}");
    assert!(
        Path::new(&dir).join("H-H.skf").is_file(),
        "H-H.skf missing in {dir}"
    );
    dir
}

fn h2o() -> (Vec<String>, Vec<[f64; 3]>) {
    (
        vec!["O".into(), "H".into(), "H".into()],
        vec![
            [0.0, 0.0, 0.0],
            [-0.7580632005, 0.6358101311, 0.0],
            [0.7580632005, 0.6358101311, 0.0],
        ],
    )
}

#[test]
fn test_gpu_dftb_h2o_reuse_scc_and_fire() {
    let dir = sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng =
        GpuDftb::new(sk, &dir, sp, xyz, 1).unwrap_or_else(|e| panic!("GpuDftb::new: {e}"));
    eprintln!("[gpu] caps={}", eng.rt.caps().name);
    let s1 = eng.scc(100, 1e-6).unwrap();
    let ev1 = eng.eval(true).unwrap_or_else(|e| panic!("eval: {e}"));
    let e1 = ev1.energy[0];
    let f = ev1.forces.as_ref().expect("eval(true) returns forces");
    eprintln!(
        "H2O GpuDftb SCC1: E={e1:.8} rms={:.3e} iters={} stalled={}",
        s1.rms, s1.n_iters, s1.stalled
    );
    assert!(e1.is_finite(), "E non-finite: {e1}");
    assert!(s1.rms < 1e-4, "H2O SCC diverged: rms={:.3e}", s1.rms);
    // Same object, same coords — must not rebuild runtime. Energy must match.
    let s2 = eng.scc(100, 1e-6).unwrap();
    let e2 = eng.eval(false).unwrap().energy[0];
    eprintln!(
        "H2O GpuDftb SCC2 (reuse): E={e2:.8} rms={:.3e} iters={}",
        s2.rms, s2.n_iters
    );
    assert!(
        (e2 - e1).abs() < 1e-5,
        "reuse SCC |dE|={:.3e} — pipeline is not persistent/warm",
        (e2 - e1).abs()
    );
    let mut max_f = 0.0f32;
    for &x in f.iter() {
        max_f = max_f.max(x.abs());
    }
    eprintln!("H2O GpuDftb |F|_max={max_f:.4e}");
    assert!(max_f.is_finite(), "forces non-finite");
    // One FIRE step on the same engine (set_coords + scc, no compile). 0.1 Å cap.
    let mf = eng.fire_step(1e-2).unwrap();
    let s3 = eng.scc(100, 1e-6).unwrap();
    let e3 = eng.eval(false).unwrap().energy[0];
    eprintln!(
        "H2O GpuDftb after 1 FIRE: E={e3:.8} max|F|={mf:.4e} rms={:.3e}",
        s3.rms
    );
    assert!(
        e3.is_finite() && e3.abs() < 20.0,
        "FIRE step exploded: E={e3}"
    );
    assert!(
        (e3 - e1).abs() < 1.0,
        "FIRE moved energy by {:.3} Ha (E0={e1:.8} E={e3:.8}) — forces/repulsive likely wrong",
        (e3 - e1).abs()
    );
    // MD step reuses the same kernels/buffers.
    let mf_md = eng.md_step(0.1).unwrap();
    let s4 = eng.scc(100, 1e-6).unwrap();
    let e4 = eng.eval(false).unwrap().energy[0];
    eprintln!(
        "H2O GpuDftb after 1 MD: E={e4:.8} max|F|={mf_md:.4e} rms={:.3e}",
        s4.rms
    );
    assert!(
        e4.is_finite() && e4.abs() < 20.0,
        "MD step exploded: E={e4}"
    );
}

/// R1 regression: replicas that converge BEFORE the last SCC iteration must
/// still get a fresh energy-weighted density W for the force eval.
///
/// Bug: `build_edm` shared the `k_density` kernel whose work gate is the SCC
/// `active` mask. At `scc_mix` exit the device mask holds the LAST
/// iteration's active set, so early-converged replicas had active=0 → W not
/// rebuilt → stale (previous geometry) or zero (first eval) → the Pulay
/// `−W·dS` force term was missing/wrong. Single-replica and uniform batches
/// (all replicas converge in the same final iteration) evade the bug, which
/// is why only heterogeneous scan batches bifurcated.
///
/// Check: per-replica forces from a heterogeneous batch must equal solo
/// (batch=1) forces at the same geometry. With the bug the first-eval W is
/// all-zero for early finishers → O(1) Ha/Å force error. A second `scc`
/// call on the same state converges all replicas in one iteration and would
/// mask the bug — so forces must be read right after the FIRST scc.
/// R3: one-step constant-force algebra — kick–drift on a fresh engine
/// (v=0 → P=0 → mode 2 v-reset, dt 1.0→0.7) must give Δx = 0.49·F per
/// component. The old kernel produced Δx = 1.5·F·dt² = 0.735·F (double
/// acceleration) — clearly distinguishable.
#[test]
fn test_gpu_dftb_fire_one_step_algebra() {
    let dir = sk_dir();
    let (sp, mut xyz) = h2o();
    xyz[1][0] += 0.05; // small displacement → small nonzero F
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng = GpuDftb::new(sk, &dir, sp, xyz, 1).unwrap();
    eng.scc(100, 1e-6).unwrap();
    let ev = eng.eval(true).unwrap();
    let f = ev.forces.as_ref().unwrap().clone();
    let c0: Vec<[f64; 3]> = eng.coords().to_vec();
    let mf = eng.fire_step(0.0).unwrap(); // f_tol=0 → never park, always step
    eng.sync_coords_to_host().unwrap();
    let mut worst = 0.0f64;
    for a in 0..3 {
        for c in 0..3 {
            let pred = 0.49 * f[3 * a + c] as f64; // v = F·dt; Δx = v·dt = F·dt²
            let act = eng.coords()[a][c] - c0[a][c];
            worst = worst.max((pred - act).abs());
        }
    }
    eprintln!("[R3] one-step max|Δx_pred−Δx_act| = {worst:.3e} Å (max|F|={mf:.3e})");
    assert!(worst < 1e-5, "FIRE one-step algebra mismatch {worst:.3e} — kick–drift violated (double acceleration would give 0.735·F)");
}

/// I10: e_atom from the force gather must reconstruct the band energy —
/// Σ_a e_atom = Tr(D·H0) = e_band (the Rayleigh scalar in e_scal[4b]).
#[test]
fn test_gpu_dftb_e_atom_band_energy() {
    let dir = sk_dir();
    let (sp, xyz) = h2o();
    let mut all = Vec::new();
    for _ in 0..4 {
        all.extend(xyz.iter().cloned());
    }
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng = GpuDftb::new(sk, &dir, sp, all, 4).unwrap();
    eng.scc(100, 1e-6).unwrap();
    eng.eval(true).unwrap();
    let ea = eng.read_e_atom().unwrap();
    for b in 0..4 {
        let e_pair: f64 = (0..3).map(|a| ea[b * 3 + a] as f64).sum();
        // e_scal e_band is Rayleigh Tr(D·H_scc) = Tr(D·H0) + Δq·V + q0·V —
        // the pairwise e_atom covers only the H0 part.
        let e_band = eng.plan.e_scal_host[4 * b]
            - eng.plan.e_scal_host[4 * b + 2]
            - eng.plan.e_scal_host[4 * b + 3];
        let d = (e_pair - e_band).abs();
        eprintln!("[I10] replica {b}: Σe_atom={e_pair:.9}  Tr(D·H0)={e_band:.9}  |Δ|={d:.3e}");
        assert!(
            d < 1e-4,
            "e_atom sum {e_pair:.9} != Tr(D·H0) {e_band:.9} (|Δ|={d:.3e} Ha)"
        );
    }
}

/// R4: constraint + frozen-DOF contracts on the device FIRE path.
/// Covers: invalid-input rejection, initial-coordinate projection onto the
/// constraint surface, distance held over FIRE steps, frozen endpoint and
/// frozen atom stationarity, both-frozen rejection both directions.
#[test]
fn test_gpu_dftb_constraint_and_frozen() {
    let dir = sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let dist = |e: &GpuDftb, i: usize, j: usize| {
        let c = e.coords();
        let d = [c[j][0] - c[i][0], c[j][1] - c[i][1], c[j][2] - c[i][2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };

    // --- invalid inputs must fail loud (before any state mutation) ---
    {
        let mut eng = GpuDftb::new(sk.clone(), &dir, sp.clone(), xyz.clone(), 1).unwrap();
        assert!(eng.set_constraint(1, 1, &[1.5]).is_err(), "i==j must fail");
        assert!(
            eng.set_constraint(1, 7, &[1.5]).is_err(),
            "out-of-range must fail"
        );
        assert!(
            eng.set_constraint(1, 2, &[]).is_err(),
            "targets.len != batch must fail"
        );
        assert!(
            eng.set_constraint(1, 2, &[0.0]).is_err(),
            "zero target must fail"
        );
        assert!(
            eng.set_constraint(1, 2, &[-1.0]).is_err(),
            "negative target must fail"
        );
        assert!(
            eng.set_constraint(1, 2, &[f64::NAN]).is_err(),
            "NaN target must fail"
        );
        eng.set_frozen_atoms(&[1, 2]).unwrap();
        assert!(
            eng.set_constraint(1, 2, &[1.5]).is_err(),
            "both-frozen must fail at set_constraint"
        );
    }
    // freezing both endpoints AFTER the constraint exists must also fail
    {
        let mut eng = GpuDftb::new(sk.clone(), &dir, sp.clone(), xyz.clone(), 1).unwrap();
        eng.set_constraint(1, 2, &[1.6]).unwrap();
        assert!(
            eng.set_frozen_atoms(&[1, 2]).is_err(),
            "freezing both endpoints post-hoc must fail"
        );
    }

    // --- projection + one-frozen-endpoint + stationarity ---
    let mut eng = GpuDftb::new(sk, &dir, sp, xyz, 1).unwrap();
    eng.scc(100, 1e-6).unwrap();
    let d0 = dist(&eng, 1, 2);
    eprintln!("[R4] initial H-H dist = {d0:.6} Å");
    eng.set_constraint(1, 2, &[1.6]).unwrap(); // target differs → must project
    let d1 = dist(&eng, 1, 2);
    eprintln!("[R4] after set_constraint: dist = {d1:.6} (target 1.6)");
    assert!(
        (d1 - 1.6).abs() < 1e-5,
        "initial projection failed: dist={d1}"
    );

    // Freeze endpoint 1 and a third atom (0). Only atom 2 may move, and the
    // constraint must still be satisfied by its mobility-weighted projection.
    eng.set_frozen_atoms(&[0, 1]).unwrap();
    let c_frozen: Vec<[f64; 3]> = vec![eng.coords()[0], eng.coords()[1]];
    for step in 0..5 {
        let mf = eng.fire_step(0.0).unwrap(); // f_tol=0 → never park, always step
        eng.scc(100, 1e-6).unwrap();
        eng.sync_coords_to_host().unwrap();
        let d = dist(&eng, 1, 2);
        eprintln!("[R4] step {step}: dist={d:.6} max|F|={mf:.3e}");
        assert!(
            (d - 1.6).abs() < 1e-4,
            "constraint violated at step {step}: dist={d}"
        );
        // Frozen atoms: device coords are f32 — the host f64 mirror picks up
        // ~1e-8 readback quantization on the first sync. Physical motion is
        // what we check, at a tolerance above f32 eps (~6e-8·|x|).
        for k in 0..2 {
            let dx: f64 = (0..3)
                .map(|c| (eng.coords()[k][c] - c_frozen[k][c]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(dx < 1e-6, "frozen atom {k} moved {dx:.3e} Å at step {step}");
        }
    }
    eprintln!("[R4] constraint + frozen contracts OK");
}

/// R2: masked SCC retry must re-solve ONLY the flagged replicas — a
/// converged replica's device state (C/D/q) must be untouched by a retry
/// targeting another replica, and exhausted SCC must report honest
/// Failed + not move atoms on uncertified state.
#[test]
fn test_gpu_dftb_masked_retry_isolation() {
    let dir = sk_dir();
    let (sp, xyz0) = h2o();
    let mut xyz1 = xyz0.clone();
    xyz1[2][0] += 0.3; // stretched O–H → different replica
    let mut all = xyz0.clone();
    all.extend(xyz1.iter().cloned());
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng = GpuDftb::new(sk, &dir, sp, all, 2).unwrap();
    let s1 = eng.scc(100, 1e-6).unwrap();
    let e1 = eng.eval(true).unwrap();
    eprintln!(
        "[R2] run-1: E={:?} iters={} statuses={:?}",
        e1.energy, s1.n_iters, s1.statuses
    );

    // Masked retry on replica 1 only — replica 0 must be bitwise untouched.
    let s2 = eng.scc_masked(100, 1e-6, &[false, true]).unwrap();
    let e2 = eng.eval(true).unwrap();
    eprintln!(
        "[R2] masked retry rep1: E={:?} iters={} statuses={:?}",
        e2.energy, s2.n_iters, s2.statuses
    );
    assert_eq!(
        e2.energy[0], e1.energy[0],
        "replica 0 energy changed under masked retry — its state was touched"
    );
    assert!(
        (e2.energy[1] - e1.energy[1]).abs() < 1e-5,
        "replica 1 re-solve landed elsewhere: dE={:.3e}",
        e2.energy[1] - e1.energy[1]
    );

    // Exhausted SCC (max_iter=1, impossible tol) → all Failed, honestly.
    let s3 = eng.scc(1, 1e-12).unwrap();
    eprintln!(
        "[R2] exhausted: statuses={:?} stalled={} rms={:.3e}",
        s3.statuses, s3.stalled, s3.rms
    );
    assert!(s3.stalled, "exhausted SCC must report stalled");
    assert!(
        s3.statuses.iter().all(|s| *s == SccStatus::Failed),
        "exhausted SCC must report Failed: {:?}",
        s3.statuses
    );
    // Parked on uncertified state: fire_step must not move atoms.
    eng.sync_coords_to_host().unwrap();
    let c0: Vec<[f64; 3]> = eng.coords().to_vec();
    let _ = eng.fire_step(1e-2).unwrap();
    eng.sync_coords_to_host().unwrap();
    for (b, (a, b0)) in eng.coords().iter().zip(c0.iter()).enumerate() {
        for c in 0..3 {
            assert!(
                (a[c] - b0[c]).abs() < 1e-6,
                "replica {b} atom moved on uncertified SCC state"
            );
        }
    }
    eprintln!("[R2] masked retry isolation + honest exhaustion OK");
}

/// R5: jacobi_cyclic_global_batched keeps jn/2 pair rotations in __local
/// arrays of 128 → n ≤ 256. Check the boundary is accepted and beyond is
/// rejected at plan construction (not a silent local-memory overrun).
#[test]
fn test_gpu_dftb_jacobi_capacity_guard() {
    let dir = sk_dir();
    // C gives 4 orbitals/4e, N gives 4 orbitals/5e, H gives 1/1:
    // 64C → n_orb=256 (jpair=128, at capacity → accept);
    // 62C+1N+1H → n_orb=253, n_el=254 (odd n, even e — pad path, accept);
    // 65C → n_orb=260 (jpair=130 → reject).
    let cases: [(usize, usize, usize, bool); 3] =
        [(64, 0, 0, true), (62, 1, 1, true), (65, 0, 0, false)];
    for (nc, nn, nh, ok) in cases {
        let mut sp = vec!["C".to_string(); nc];
        sp.extend(std::iter::repeat("N".to_string()).take(nn));
        sp.extend(std::iter::repeat("H".to_string()).take(nh));
        let n_orb = 4 * nc + 4 * nn + nh;
        let coords: Vec<[f64; 3]> = (0..nc + nn + nh)
            .map(|i| [1.4 * i as f64, 0.3 * (i % 7) as f64, 0.0])
            .collect();
        let sk = load_sk_for_species(&dir, &sp).unwrap();
        match GpuDftb::new(sk, &dir, sp, coords, 1) {
            Ok(_) if !ok => panic!(
                "n={n_orb} (jpair={}) accepted — R5 capacity guard missing",
                (n_orb + 1) / 2
            ),
            Err(e) if ok => panic!("n={n_orb} wrongly rejected: {e}"),
            Ok(_) => eprintln!("[R5] n={n_orb} accepted (jpair≤128)"),
            Err(e) => {
                eprintln!("[R5] n={n_orb} correctly rejected: {e}");
                assert!(
                    e.to_string().contains("capacity"),
                    "rejection should name the capacity limit: {e}"
                );
            }
        }
    }
}

#[test]
fn test_gpu_dftb_edm_fresh_mixed_convergence() {
    check_edm_fresh(0.0); // integer occupation
    check_edm_fresh(0.002); // Fermi smearing — exercises use_w on k_edm
}

fn check_edm_fresh(kt: f32) {
    let dir = sk_dir();
    let (sp, xyz) = h2o();
    // Heterogeneous batch: same species, staggered O–H stretches →
    // different SCC iteration counts per replica.
    let stretch = [0.0f64, 0.15, 0.35, 0.60, -0.20, 0.45];
    let nb = stretch.len();
    let mut coords = Vec::with_capacity(nb * xyz.len());
    for &d in &stretch {
        let mut g = xyz.clone();
        g[1][0] += d; // stretch O–H1 bond
        g[2][1] += 0.5 * d; // bend H2 a little too
        coords.extend_from_slice(&g);
    }
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng = GpuDftb::new(sk, &dir, sp.clone(), coords.clone(), nb)
        .unwrap_or_else(|e| panic!("GpuDftb::new batch: {e}"));
    eng.set_smearing(kt);
    let s = eng
        .scc(100, 1e-6)
        .unwrap_or_else(|e| panic!("batch scc: {e}"));
    eprintln!(
        "[R1 kT={kt}] batch scc iters={} rms={:.3e} statuses={:?}",
        s.n_iters, s.rms, s.statuses
    );
    let ev = eng.eval(true).unwrap_or_else(|e| panic!("batch eval: {e}"));
    let f_batch = ev.forces.as_ref().expect("eval(true) returns forces");

    // Solo reference: same geometry, batch=1.
    let mut max_df = 0.0f64;
    let mut worst = usize::MAX;
    for (b, &d) in stretch.iter().enumerate() {
        let mut g = xyz.clone();
        g[1][0] += d;
        g[2][1] += 0.5 * d;
        let sk1 = load_sk_for_species(&dir, &sp).unwrap();
        let mut e1 = GpuDftb::new(sk1, &dir, sp.clone(), g, 1)
            .unwrap_or_else(|e| panic!("GpuDftb::new solo {b}: {e}"));
        e1.set_smearing(kt);
        let s1 = e1
            .scc(100, 1e-6)
            .unwrap_or_else(|e| panic!("solo scc {b}: {e}"));
        let ev1 = e1
            .eval(true)
            .unwrap_or_else(|e| panic!("solo eval {b}: {e}"));
        let f1 = ev1.forces.as_ref().expect("solo eval(true) returns forces");
        let na = xyz.len();
        let mut mb = 0.0f64;
        for i in 0..3 * na {
            let df = (f_batch[b * 3 * na + i] as f64 - f1[i] as f64).abs();
            mb = mb.max(df);
        }
        let fb_max = (0..3 * na).fold(0.0f64, |a, i| a.max(f_batch[b * 3 * na + i].abs() as f64));
        eprintln!("[R1 kT={kt}] replica {b}: solo iters={} rms={:.2e} max|F|={fb_max:.4e} max|ΔF|={mb:.3e}", s1.n_iters, s1.rms);
        if mb > max_df {
            max_df = mb;
            worst = b;
        }
    }
    eprintln!("[R1 kT={kt}] worst replica {worst}: max|ΔF|={max_df:.3e} Ha/Å");
    assert!(max_df < 1e-3, "kT={kt}: batched vs solo force mismatch {max_df:.3e} Ha/Å at replica {worst} — stale/zero W in early-converged replicas (R1)");
}

/// W10/I3 regression: zero device-buffer allocations inside solver loops.
/// The runtime counts every buffer_from_slice/zero_buffer/copy_buffer call.
/// After GpuDftb::new (construction allocs are legal), scc → eval →
/// fire_step → scc → eval must add exactly ZERO to the counter — a nonzero
/// delta means a device alloc crept into a hot loop.
#[test]
fn test_gpu_dftb_no_loop_allocs() {
    use std::sync::atomic::Ordering::Relaxed;
    let dir = sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&dir, &sp).unwrap();
    let mut eng = GpuDftb::new(sk, &dir, sp, xyz, 1).unwrap();
    eng.set_smearing(0.002); // smeared production path — exercises occ tail
    let a0 = eng.rt.alloc_count.load(Relaxed);
    eng.scc(50, 1e-6).unwrap();
    eng.eval(true).unwrap();
    eng.fire_step(0.0).unwrap();
    eng.scc(50, 1e-6).unwrap();
    eng.eval(true).unwrap();
    let a1 = eng.rt.alloc_count.load(Relaxed);
    eprintln!(
        "[W10] alloc_count {a0} → {a1} (delta {} across scc+eval+fire+scc+eval)",
        a1 - a0
    );
    assert_eq!(
        a1,
        a0,
        "device buffer allocations inside solver loops: +{}",
        a1 - a0
    );
}
