//! Standalone validation of the complex Hermitian Jacobi eigensolver +
//! zgemm S^{-1/2} pipeline (PBC path, experimental).
//!
//! Checks (per Dense_Multi_PBC.chat.md):
//!   - Residual:     ||A·V - V·Λ||_F / ||A||_F          (< 1e-4, f32 path)
//!   - Unitarity:    ||V†·V - I||_F / N                 (< 1e-4)
//!   - Eigenvalue parity vs LAPACK zheevd (Weyl-bounded by the residual)
//!   - Real-symmetric reduction (imag=0) parity
//!   - 2x2 pure-phase analytic case
//!   - Bloch invariant: H(-k) = conj(H(k)) → identical spectrum
//!   - Warm start (init_v=1) on a perturbed matrix
//!   - S^{-1/2} pipeline: X = V·rsqrt(λ)·V† satisfies ‖X·S·X − I‖ ~ f32
//!   - Zero alloc growth inside a repeated-solve loop (alloc_count delta)
//!
//! diag record per flat system: {off, off/‖A‖_F, stop, sweeps};
//! stop: 0 converged · 1 stall · 2 max sweeps · 3 n>256 · 4 non-finite.

use ocl::prm::Float2;
use rust_dftb::qmqm::gpu_hermitian::{
    hermitian_jacobi_batched, hermitian_jacobi_simple, zgemm_batched, zscale_eigenvectors_batched,
    ZOP_H, ZOP_N,
};
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;

fn try_runtime() -> Option<GpuRuntime> {
    match GpuRuntime::new() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

// ---- host-side complex helpers ([f64;2]) ----

fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn cconj(a: [f64; 2]) -> [f64; 2] {
    [a[0], -a[1]]
}

/// Seeded RNG (SplitMix-ish, same style as the real-path tests).
fn next_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
}

/// Random complex Hermitian n×n, row-major [re,im] pairs as Float2.
fn random_hermitian(n: usize, seed: u64) -> Vec<Float2> {
    let mut st = seed;
    let mut a = vec![Float2::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in i..n {
            let re = next_f64(&mut st);
            let im = if i == j { 0.0 } else { next_f64(&mut st) };
            a[i * n + j] = Float2::new(re as f32, im as f32);
            a[j * n + i] = Float2::new(re as f32, -im as f32);
        }
    }
    a
}

/// LAPACK zheevd reference (f64). `a` is row-major [re,im] f32 pairs;
/// LAPACK sees it column-major = Aᵀ = conj(A) — same real eigenvalues.
fn cpu_heevd(a: &[Float2], n: usize) -> Vec<f64> {
    let mut am: Vec<lapack::c64> = a
        .iter()
        .map(|z| lapack::c64::new(z[0] as f64, z[1] as f64))
        .collect();
    let mut w = vec![0.0f64; n];
    let mut info = 0i32;
    // workspace queries
    let mut wk = [lapack::c64::new(0.0, 0.0)];
    let mut rwk = [0.0f64];
    let mut iwk = [0i32];
    unsafe {
        lapack::zheevd(
            b'V', b'L', n as i32, &mut am, n as i32, &mut w, &mut wk, -1, &mut rwk, -1, &mut iwk,
            -1, &mut info,
        );
    }
    assert_eq!(info, 0, "zheevd workspace query failed info={info}");
    let lw = wk[0].re as usize;
    let lrw = rwk[0] as usize;
    let liw = iwk[0] as usize;
    let mut work = vec![lapack::c64::new(0.0, 0.0); lw];
    let mut rwork = vec![0.0f64; lrw];
    let mut iwork = vec![0i32; liw];
    unsafe {
        lapack::zheevd(
            b'V', b'L', n as i32, &mut am, n as i32, &mut w, &mut work, lw as i32, &mut rwork,
            lrw as i32, &mut iwork, liw as i32, &mut info,
        );
    }
    assert_eq!(info, 0, "zheevd failed info={info}");
    w
}

/// ‖A·V − V·Λ‖_F / ‖A‖_F — complex, f64 accumulation.
fn residual_c(a: &[Float2], v: &[Float2], eigs: &[f32], n: usize) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = [0.0f64; 2];
            for k in 0..n {
                let av = [a[i * n + k][0] as f64, a[i * n + k][1] as f64];
                let vv = [v[k * n + j][0] as f64, v[k * n + j][1] as f64];
                let p = cmul(av, vv);
                s[0] += p[0];
                s[1] += p[1];
            }
            let vl = [
                v[i * n + j][0] as f64 * eigs[j] as f64,
                v[i * n + j][1] as f64 * eigs[j] as f64,
            ];
            num += (s[0] - vl[0]).powi(2) + (s[1] - vl[1]).powi(2);
        }
    }
    for z in a {
        den += (z[0] as f64).powi(2) + (z[1] as f64).powi(2);
    }
    num.sqrt() / den.sqrt().max(1e-30)
}

/// ‖V†·V − I‖_F / N.
fn unitarity(v: &[Float2], n: usize) -> f64 {
    let mut num = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = [0.0f64; 2];
            for k in 0..n {
                let vi = cconj([v[k * n + i][0] as f64, v[k * n + i][1] as f64]);
                let vj = [v[k * n + j][0] as f64, v[k * n + j][1] as f64];
                let p = cmul(vi, vj);
                s[0] += p[0];
                s[1] += p[1];
            }
            let d = if i == j { 1.0 } else { 0.0 };
            num += (s[0] - d).powi(2) + s[1].powi(2);
        }
    }
    num.sqrt() / n as f64
}

/// ‖A − A†‖_F — Hermiticity defect of the diagonalized output.
fn hermiticity_defect(a: &[Float2], n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let dr = a[i * n + j][0] as f64 - a[j * n + i][0] as f64;
            let di = a[i * n + j][1] as f64 + a[j * n + i][1] as f64;
            s += dr * dr + di * di;
        }
    }
    s.sqrt()
}

/// Run the batched solve and return (a_out, v_out, diag).
fn solve(
    rt: &mut GpuRuntime,
    a_in: &[Float2],
    n: usize,
    batch: usize,
    init_v: Option<&[Float2]>,
) -> (Vec<Float2>, Vec<Float2>, Vec<f32>) {
    let a_buf = rt.buffer_from_slice(a_in).unwrap();
    let v_buf = match init_v {
        Some(v) => rt.buffer_from_slice(v).unwrap(),
        None => rt.zero_buffer::<Float2>(batch * n * n).unwrap(),
    };
    let active = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    let diag = rt.zero_buffer::<f32>(4 * batch).unwrap();
    hermitian_jacobi_batched(
        rt,
        &a_buf,
        &v_buf,
        n,
        batch,
        1,
        &active,
        &diag,
        if init_v.is_some() { 1 } else { 0 },
        true,
    )
    .unwrap();
    let mut a_out = vec![Float2::new(0.0, 0.0); batch * n * n];
    let mut v_out = vec![Float2::new(0.0, 0.0); batch * n * n];
    let mut d = vec![0.0f32; 4 * batch];
    rt.read_buffer(&a_buf, &mut a_out).unwrap();
    rt.read_buffer(&v_buf, &mut v_out).unwrap();
    rt.read_buffer(&diag, &mut d).unwrap();
    (a_out, v_out, d)
}

fn eigvals(a: &[Float2], b: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| a[b * n * n + i * n + i][0]).collect()
}

#[test]
fn test_hermitian_jacobi_parity() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    for &n in &[2usize, 8, 33, 64, 65, 87, 96, 128] {
        let batch = 2usize;
        let mut all_a = Vec::new();
        for b in 0..batch {
            all_a.extend(random_hermitian(n, 42 + b as u64));
        }
        let a_orig = all_a.clone();
        let (ga, gv, diag) = solve(&mut rt, &all_a, n, batch, None);
        for b in 0..batch {
            let a0 = &a_orig[b * n * n..(b + 1) * n * n];
            let v = &gv[b * n * n..(b + 1) * n * n];
            let e = eigvals(&ga, b, n);
            let res = residual_c(a0, v, &e, n);
            let uni = unitarity(v, n);
            let herm = hermiticity_defect(&ga[b * n * n..(b + 1) * n * n], n);
            let stop = diag[4 * b + 2] as i32;
            eprintln!("herm N={n} b={b}: stop={stop} sweeps={} res={res:.3e} uni={uni:.3e} herm_defect={herm:.3e}", diag[4 * b + 3]);
            assert_eq!(
                stop, 0,
                "herm Jacobi N={n} b={b} stop={stop} (1=stall 2=maxsweeps 3=cap 4=nonfinite)"
            );
            assert!(res < 1e-4, "herm Jacobi N={n} b={b} residual {res:.3e}");
            assert!(uni < 1e-4, "herm Jacobi N={n} b={b} unitarity {uni:.3e}");
            assert!(
                herm < 1e-4,
                "herm Jacobi N={n} b={b} Hermiticity defect {herm:.3e}"
            );
            // eigenvalue parity vs LAPACK — Weyl: sorted-eig deviation ≤ res·‖A‖_F
            let ce = cpu_heevd(a0, n);
            let mut gs = e.clone();
            gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mut par = 0.0f64;
            for i in 0..n {
                par = par.max((gs[i] as f64 - ce[i]).abs());
            }
            let af: f64 = a0
                .iter()
                .map(|z| (z[0] as f64).powi(2) + (z[1] as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let weyl = 2.0 * res * af + 1e-4;
            eprintln!("  eig parity max|dλ|={par:.3e} (weyl={weyl:.3e})");
            assert!(
                par < weyl,
                "herm Jacobi N={n} b={b} parity {par:.3e} exceeds Weyl bound {weyl:.3e}"
            );
        }
    }
}

/// Real symmetric matrix stored as complex (imag = 0) — the complex kernel
/// must reduce to the real one in accuracy.
#[test]
fn test_hermitian_real_reduction() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    for &n in &[33usize, 87, 128] {
        let mut st = 7u64;
        let mut a = vec![Float2::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in i..n {
                let v = next_f64(&mut st) as f32;
                a[i * n + j] = Float2::new(v, 0.0);
                a[j * n + i] = Float2::new(v, 0.0);
            }
        }
        let a_orig = a.clone();
        let (ga, gv, diag) = solve(&mut rt, &a, n, 1, None);
        let e = eigvals(&ga, 0, n);
        let res = residual_c(&a_orig, &gv, &e, n);
        let uni = unitarity(&gv, n);
        assert_eq!(diag[2] as i32, 0, "real-reduction N={n} stop={}", diag[2]);
        assert!(
            res < 1e-4 && uni < 1e-4,
            "real-reduction N={n} res={res:.3e} uni={uni:.3e}"
        );
        let ce = cpu_heevd(&a_orig, n);
        let mut gs = e.clone();
        gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let par: f64 = (0..n)
            .map(|i| (gs[i] as f64 - ce[i]).abs())
            .fold(0.0, f64::max);
        eprintln!("real-reduction N={n}: res={res:.3e} uni={uni:.3e} parity={par:.3e}");
        assert!(par < 1e-3, "real-reduction N={n} parity {par:.3e}");
    }
}

/// 2×2 pure-phase case from the chat:
///   [1, 0.2 e^{i1.3}; 0.2 e^{-i1.3}, 2] → λ = 1.5 ± sqrt(0.25 + 0.04)
#[test]
fn test_hermitian_phase_2x2() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let n = 2usize;
    let phi = 1.3f32;
    let a = vec![
        Float2::new(1.0, 0.0),
        Float2::new(0.2 * phi.cos(), 0.2 * phi.sin()),
        Float2::new(0.2 * phi.cos(), -0.2 * phi.sin()),
        Float2::new(2.0, 0.0),
    ];
    let a_orig = a.clone();
    let (ga, gv, diag) = solve(&mut rt, &a, n, 1, None);
    let e = eigvals(&ga, 0, n);
    let expect_lo = 1.5f64 - (0.25f64 + 0.04).sqrt();
    let expect_hi = 1.5f64 + (0.25f64 + 0.04).sqrt();
    let mut gs = e.clone();
    gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let res = residual_c(&a_orig, &gv, &e, n);
    let uni = unitarity(&gv, n);
    eprintln!(
        "phase 2x2: λ=({:.6},{:.6}) expected ({:.6},{:.6}) res={res:.3e} uni={uni:.3e} stop={}",
        gs[0], gs[1], expect_lo, expect_hi, diag[2]
    );
    assert_eq!(diag[2] as i32, 0);
    assert!(
        (gs[0] as f64 - expect_lo).abs() < 1e-6 && (gs[1] as f64 - expect_hi).abs() < 1e-6,
        "phase 2x2 eigenvalues off: ({},{})",
        gs[0],
        gs[1]
    );
    assert!(res < 1e-6 && uni < 1e-6);
}

/// Bloch invariant: for a 1-cell chain with hopping T and onsite D (both
/// REAL — then H(-k) = conj(H(k)) exactly),
/// H(k) = D + T·e^{ik} + Tᵀ·e^{-ik} is Hermitian and the spectra at ±k
/// must be identical (time-reversal). Batch runs both k-points.
#[test]
fn test_hermitian_bloch_invariant() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let n = 24usize;
    let mut st = 99u64;
    // real symmetric D, real T (arbitrary, not symmetric)
    let mut d = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = next_f64(&mut st);
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    let mut t = vec![0.0f64; n * n];
    for z in t.iter_mut() {
        *z = 0.3 * next_f64(&mut st);
    }
    let k0 = 0.37f64;
    let mut all_a = Vec::new();
    for &k in &[k0, -k0] {
        for i in 0..n {
            for j in 0..n {
                // D_ij + T_ij e^{ik} + T_ji e^{-ik}
                let (c, s) = (k.cos(), k.sin());
                let re = d[i * n + j] + t[i * n + j] * c + t[j * n + i] * c;
                let im = t[i * n + j] * s - t[j * n + i] * s;
                all_a.push(Float2::new(re as f32, im as f32));
            }
        }
    }
    let (ga, _gv, diag) = solve(&mut rt, &all_a, n, 2, None);
    let mut e0 = eigvals(&ga, 0, n);
    let mut e1 = eigvals(&ga, 1, n);
    e0.sort_by(|a, b| a.partial_cmp(b).unwrap());
    e1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let par: f64 = (0..n)
        .map(|i| (e0[i] as f64 - e1[i] as f64).abs())
        .fold(0.0, f64::max);
    eprintln!(
        "bloch ±k parity: max|dλ|={par:.3e} stops=({},{})",
        diag[2], diag[6]
    );
    assert_eq!(diag[2] as i32, 0);
    assert_eq!(diag[6] as i32, 0);
    assert!(par < 1e-4, "ε(-k) vs ε(k) parity {par:.3e}");
}

/// Warm start: cold-solve A → eigenbasis C; perturb A' = A + δH; project
/// W = C†A'C via zgemm (the plan's warm path does exactly this with
/// C†H_scc·C); solve W with init_v=1 so Jacobi rotates C in place —
/// V_out = C·R are then eigenvectors of A' (A'·C·R = C·W·R = C·R·Λ).
/// Must converge (stop=0) in fewer sweeps than the cold solve of A'.
#[test]
fn test_hermitian_warm_start() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let n = 48usize;
    let a = random_hermitian(n, 17);
    let (_ga, gv, _d0) = solve(&mut rt, &a, n, 1, None);
    // small Hermitian perturbation δ = 0.01·random_hermitian
    let dh = random_hermitian(n, 23);
    let ap: Vec<Float2> = a
        .iter()
        .zip(dh.iter())
        .map(|(x, d)| Float2::new(x[0] + 0.01 * d[0], x[1] + 0.01 * d[1]))
        .collect();

    // cold solve of A' (reference sweep count + reference eigenvalues)
    let (ga_cold, _v_cold, d_cold) = solve(&mut rt, &ap, n, 1, None);

    // project W = gv†·A'·gv (two zgemm launches — mirrors the plan's
    // warm eigh: temp = C†H, hp = temp·C)
    let gv_buf = rt.buffer_from_slice(&gv).unwrap();
    let ap_buf = rt.buffer_from_slice(&ap).unwrap();
    let t_buf = rt.zero_buffer::<Float2>(n * n).unwrap();
    let w_buf = rt.zero_buffer::<Float2>(n * n).unwrap();
    zgemm_batched(&mut rt, &gv_buf, &ap_buf, &t_buf, n, 1, ZOP_H, ZOP_N).unwrap(); // T = C†·A'
    zgemm_batched(&mut rt, &t_buf, &gv_buf, &w_buf, n, 1, ZOP_N, ZOP_N).unwrap(); // W = T·C

    // warm solve of W with V = C (init_v=1 → rotate in place)
    let active = rt.buffer_from_slice(&vec![1i32; 1]).unwrap();
    let diag = rt.zero_buffer::<f32>(4).unwrap();
    hermitian_jacobi_batched(&mut rt, &w_buf, &gv_buf, n, 1, 1, &active, &diag, 1, true).unwrap();
    let mut ga_warm = vec![Float2::new(0.0, 0.0); n * n];
    let mut v_warm = vec![Float2::new(0.0, 0.0); n * n];
    let mut d_warm = vec![0.0f32; 4];
    rt.read_buffer(&w_buf, &mut ga_warm).unwrap();
    rt.read_buffer(&gv_buf, &mut v_warm).unwrap();
    rt.read_buffer(&diag, &mut d_warm).unwrap();

    let e_cold = eigvals(&ga_cold, 0, n);
    let e_warm = eigvals(&ga_warm, 0, n);
    let res = residual_c(&ap, &v_warm, &e_warm, n);
    let uni = unitarity(&v_warm, n);
    let sw_cold = d_cold[3];
    let sw_warm = d_warm[3];
    eprintln!("warm start N={n}: cold sweeps={sw_cold} warm sweeps={sw_warm} res={res:.3e} uni={uni:.3e} stops=({},{})", d_cold[2], d_warm[2]);
    assert_eq!(d_warm[2] as i32, 0, "warm solve stop={}", d_warm[2]);
    assert!(
        res < 1e-4 && uni < 1e-4,
        "warm residual {res:.3e} uni {uni:.3e}"
    );
    let mut gc = e_cold.clone();
    gc.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut gw = e_warm.clone();
    gw.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let par: f64 = (0..n)
        .map(|i| (gw[i] as f64 - gc[i] as f64).abs())
        .fold(0.0, f64::max);
    assert!(par < 1e-4, "warm-vs-cold eig parity {par:.3e}");
    assert!(
        sw_warm < sw_cold,
        "warm ({sw_warm}) should need fewer sweeps than cold ({sw_cold})"
    );
}

/// S^{-1/2} pipeline: S SPD → Hermitian Jacobi → V·rsqrt(λ) → X = Vs·V†,
/// verify ‖X·S·X − I‖_F ≈ f32 floor. Exercises zgemm op_b=H.
#[test]
fn test_zinvsqrt_pipeline() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let n = 33usize;
    // Strictly diagonally dominant SPD: S = I + α·H with α small enough
    // that Gershgorin guarantees λ_min ≥ 1 − α·max_row_sum|H| > 0.
    // α = 0.02, |h_ij| ≤ 1 → row off-sum ≤ 0.02·(n−1) = 0.64 → λ_min ≥ 0.36.
    let h = random_hermitian(n, 31);
    let s: Vec<Float2> = h
        .iter()
        .enumerate()
        .map(|(idx, z)| {
            let i = idx / n;
            Float2::new(
                if i == idx % n {
                    1.0 + 0.02 * z[0]
                } else {
                    0.02 * z[0]
                },
                0.02 * z[1],
            )
        })
        .collect();

    let s_buf = rt.buffer_from_slice(&s).unwrap();
    let sw = rt.buffer_from_slice(&s).unwrap(); // s_work
    let sv = rt.zero_buffer::<Float2>(n * n).unwrap(); // eigenvectors
    let svs = rt.zero_buffer::<Float2>(n * n).unwrap(); // V·rsqrt(λ)
    let x_buf = rt.zero_buffer::<Float2>(n * n).unwrap();
    let lmin = rt.zero_buffer::<f32>(1).unwrap();
    let t_buf = rt.zero_buffer::<Float2>(n * n).unwrap();
    let m_buf = rt.zero_buffer::<Float2>(n * n).unwrap();

    let _d = hermitian_jacobi_simple(&mut rt, &sw, &sv, n, 1, true).unwrap();
    zscale_eigenvectors_batched(&mut rt, &sw, &sv, &svs, &lmin, n, 1, 1).unwrap();
    zgemm_batched(&mut rt, &svs, &sv, &x_buf, n, 1, ZOP_N, ZOP_H).unwrap(); // X = Vs·V†
    zgemm_batched(&mut rt, &x_buf, &s_buf, &t_buf, n, 1, ZOP_N, ZOP_N).unwrap(); // T = X·S
    zgemm_batched(&mut rt, &t_buf, &x_buf, &m_buf, n, 1, ZOP_N, ZOP_N).unwrap(); // M = T·X = XSX

    let mut m = vec![Float2::new(0.0, 0.0); n * n];
    let mut lm = [0.0f32];
    rt.read_buffer(&m_buf, &mut m).unwrap();
    rt.read_buffer(&lmin, &mut lm).unwrap();
    let mut dev = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let d = if i == j { 1.0 } else { 0.0 };
            dev += (m[i * n + j][0] as f64 - d).powi(2) + (m[i * n + j][1] as f64).powi(2);
        }
    }
    dev = dev.sqrt();
    eprintln!("zinvsqrt N={n}: λ_min={:.4} ‖XSX−I‖={dev:.3e}", lm[0]);
    assert!(lm[0] > 1e-6, "λ_min {} too small", lm[0]);
    assert!(
        dev < 1e-4,
        "X·S·X − I residual {dev:.3e} — f32 path expects ~1e-5"
    );
}

/// zgemm parity vs CPU f64 for all op combos (N,T,H × N,T,H).
/// The conjugate-transpose sign convention is the bug-prone part — pin it.
#[test]
fn test_zgemm_all_ops() {
    use rust_dftb::qmqm::gpu_hermitian::ZOP_T;
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let n = 17usize;
    let mk = |seed: u64| -> Vec<Float2> {
        let mut st = seed;
        (0..n * n)
            .map(|_| Float2::new(next_f64(&mut st) as f32, next_f64(&mut st) as f32))
            .collect()
    };
    let a = mk(101);
    let b = mk(202);
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let b_buf = rt.buffer_from_slice(&b).unwrap();
    let c_buf = rt.zero_buffer::<Float2>(n * n).unwrap();
    let mut ch = vec![Float2::new(0.0, 0.0); n * n];
    for op_a in [ZOP_N, ZOP_T, ZOP_H] {
        for op_b in [ZOP_N, ZOP_T, ZOP_H] {
            zgemm_batched(&mut rt, &a_buf, &b_buf, &c_buf, n, 1, op_a, op_b).unwrap();
            rt.read_buffer(&c_buf, &mut ch).unwrap();
            // CPU reference: C = op(A)·op(B), f64
            let get = |m: &[Float2], op: i32, i: usize, j: usize| -> [f64; 2] {
                let (re, im) = if op == 0 {
                    (m[i * n + j][0], m[i * n + j][1])
                } else {
                    (m[j * n + i][0], m[j * n + i][1])
                };
                if op == 2 {
                    [re as f64, -(im as f64)]
                } else {
                    [re as f64, im as f64]
                }
            };
            let mut err = 0.0f64;
            let mut nrm = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let mut s = [0.0f64; 2];
                    for k in 0..n {
                        let p = cmul(get(&a, op_a, i, k), get(&b, op_b, k, j));
                        s[0] += p[0];
                        s[1] += p[1];
                    }
                    err += (ch[i * n + j][0] as f64 - s[0]).powi(2)
                        + (ch[i * n + j][1] as f64 - s[1]).powi(2);
                    nrm += s[0] * s[0] + s[1] * s[1];
                }
            }
            let rel = err.sqrt() / nrm.sqrt().max(1e-30);
            eprintln!("zgemm op_a={op_a} op_b={op_b}: rel_err={rel:.3e}");
            assert!(
                rel < 1e-5,
                "zgemm op_a={op_a} op_b={op_b} rel_err {rel:.3e}"
            );
        }
    }
}

/// GpuPbcPlan end-to-end smoke test: n=4 (2 atoms × 2 orbitals),
/// nk=2 k-points, n_rep=2 replicas. Runs the full SCC pipeline
/// (zdq_v → zhscc → zgemm × 2 → Hermitian Jacobi → kpoint_occ →
/// zsc_mulliken → qreduce → DIIS) for several iterations and checks
/// the residual converges, charges conserve per replica, and energy
/// is finite. Exercises every bound kernel arg — a binding bug here
/// fails loudly.
#[test]
fn test_gpu_pbc_plan_scc_smoke() {
    use rust_dftb::qmqm::gpu_pbc_plan::GpuPbcPlan;
    let Some(mut rt) = try_runtime() else {
        return;
    };

    let n = 4usize;
    let n_atoms = 2usize;
    let n_rep = 2usize;
    let nk = 2usize;
    let n_sys = n_rep * nk;
    let kw = vec![0.5f32, 0.5];

    // S(k): diagonally dominant Hermitian per flat system
    // H0(k): random Hermitian with −2 onsite shift (valence-ish spectrum)
    let mut s_host = vec![Float2::new(0.0, 0.0); n_sys * n * n];
    let mut h0_host = vec![Float2::new(0.0, 0.0); n_sys * n * n];
    for sid in 0..n_sys {
        let hs = random_hermitian(n, 700 + sid as u64);
        let hh = random_hermitian(n, 900 + sid as u64);
        for i in 0..n {
            for j in 0..n {
                let idx = sid * n * n + i * n + j;
                s_host[idx] = Float2::new(
                    if i == j {
                        1.0 + 0.02 * hs[idx % (n * n)][0]
                    } else {
                        0.02 * hs[idx % (n * n)][0]
                    },
                    0.02 * hs[idx % (n * n)][1],
                );
                h0_host[idx] = Float2::new(
                    if i == j {
                        -2.0 + 0.3 * hh[idx % (n * n)][0]
                    } else {
                        0.3 * hh[idx % (n * n)][0]
                    },
                    0.3 * hh[idx % (n * n)][1],
                );
            }
        }
    }
    // γ (real, per replica): γ_AA=0.4, γ_AB=0.2 — DFTB-magnitude feedback
    let mut g_host = vec![0.0f32; n_rep * n_atoms * n_atoms];
    for r in 0..n_rep {
        g_host[r * 4 + 0] = 0.4;
        g_host[r * 4 + 1] = 0.2;
        g_host[r * 4 + 2] = 0.2;
        g_host[r * 4 + 3] = 0.4;
    }
    // q0: neutral reference (2 electrons per atom → 1 band-occupation each)
    let q0_host = vec![2.0f32, 2.0, 2.0, 2.0]; // [n_rep][n_atoms]
    let mut oa_host = vec![0i32; n_rep * n];
    for r in 0..n_rep {
        oa_host[r * n..r * n + n].copy_from_slice(&[0, 0, 1, 1]);
    }

    let s_buf = rt.buffer_from_slice(&s_host).unwrap();
    let h0_buf = rt.buffer_from_slice(&h0_host).unwrap();
    let g_buf = rt.buffer_from_slice(&g_host).unwrap();
    let q0_buf = rt.buffer_from_slice(&q0_host).unwrap();
    let oa_buf = rt.buffer_from_slice(&oa_host).unwrap();

    let mut plan = GpuPbcPlan::new(
        &mut rt, &s_buf, &h0_buf, &g_buf, &q0_buf, &oa_buf, n, n_atoms, n_rep, nk, &kw,
    )
    .unwrap();
    plan.set_initial_charges(&rt, &q0_host).unwrap();
    plan.reset_diis(&rt).unwrap();

    // n_occ = 2 doubly-occupied band-equivalents per cell (n_elec/2)
    let mut rms_hist = Vec::new();
    for _ in 0..12 {
        let r = plan.scc_step_diis(&mut rt, 2, 0.3, 1e-7).unwrap();
        rms_hist.push(r);
    }
    eprintln!(
        "pbc smoke rms: {:?}",
        rms_hist
            .iter()
            .map(|x| format!("{x:.2e}"))
            .collect::<Vec<_>>()
    );
    let q = plan.read_charges(&rt).unwrap();
    let e = plan.compute_energy(&mut rt, 2).unwrap();
    let mu = {
        let mut m = vec![0.0f32; n_rep];
        rt.read_buffer(&plan.mu_out, &mut m).unwrap();
        m
    };
    eprintln!("pbc smoke: q={q:?} E={e:?} mu={mu:?}");

    // charge conservation: Σ_atoms q = 4.0 per replica (2 atoms × 2 e−)
    for r in 0..n_rep {
        let qsum: f32 = q[r * n_atoms..(r + 1) * n_atoms].iter().sum();
        assert!(
            (qsum - 4.0).abs() < 1e-3,
            "rep {r} charge not conserved: Σq={qsum}"
        );
        assert!(e[r].is_finite(), "rep {r} energy non-finite: {}", e[r]);
        assert!(mu[r].is_finite());
    }
    // convergence: rms should be decreasing and end small
    let r0 = rms_hist[0];
    let rl = *rms_hist.last().unwrap();
    assert!(rl < r0, "rms did not decrease: {r0:.3e} → {rl:.3e}");
    assert!(rl < 1e-4, "rms failed to converge: {rl:.3e}");

    // Jacobi certification across the (rep,k) grid
    let ok = plan.check_jacobi(&rt, &vec![1i32; n_rep]).unwrap();
    assert!(ok.iter().all(|&x| x), "jacobi certification failed: {ok:?}");
}

/// No-alloc contract: a repeated-solve loop must not grow the runtime's
/// allocation counter (the plan-level guarantee tested at kernel level).
#[test]
fn test_hermitian_jacobi_no_alloc_in_loop() {
    use std::sync::atomic::Ordering;
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let n = 48usize;
    let batch = 3usize;
    let mut all_a = Vec::new();
    for b in 0..batch {
        all_a.extend(random_hermitian(n, 60 + b as u64));
    }
    let a_buf = rt.buffer_from_slice(&all_a).unwrap();
    let v_buf = rt.zero_buffer::<Float2>(batch * n * n).unwrap();
    let active = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    let diag = rt.zero_buffer::<f32>(4 * batch).unwrap();
    let a0 = rt.alloc_count.load(Ordering::Relaxed);
    for _ in 0..10 {
        // reset A on device (copy is a transfer, not an alloc), re-solve
        rt.write_buffer(&a_buf, &all_a).unwrap();
        hermitian_jacobi_batched(
            &mut rt, &a_buf, &v_buf, n, batch, 1, &active, &diag, 0, true,
        )
        .unwrap();
    }
    rt.finish().unwrap();
    let a1 = rt.alloc_count.load(Ordering::Relaxed);
    assert_eq!(
        a0,
        a1,
        "alloc_count grew by {} inside the solve loop — violates the preallocation contract",
        a1 - a0
    );
}
