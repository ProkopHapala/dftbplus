//! Phase 2: tiled block Jacobi eigensolver parity test.
//!
//! Verifies `tiled_jacobi_batched` against CPU nalgebra symmetric eigendecomposition
//! for N = 65, 87, 96, 97, 128 — covering the N>64 boundary.
//!
//! Checks:
//!   - Residual: ||A·V - V·Λ||_F / ||A||_F
//!   - Orthogonality: ||V^T·V - I||_F / N
//!   - Eigenvalue parity vs CPU reference
//!
//! Tolerances (manifest §4.3):
//!   - Residual < 1e-5
//!   - Orthogonality < 1e-5
//!   - Eigenvalue parity < 1e-4 Ha

use rust_dftb::qmqm::gpu_eigen::{
    block_jacobi_batched, direct_jacobi_batched, jacobi_batched, resident_jacobi_batched,
    tiled_jacobi_batched,
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

/// Generate a random symmetric matrix.
fn random_symmetric(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32;
            a[i * n + j] = val;
            a[j * n + i] = val;
        }
    }
    a
}

/// CPU reference: symmetric eigendecomposition via nalgebra.
fn cpu_eig(a: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut m = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = a[i * n + j] as f64;
        }
    }
    let sym = nalgebra::SymmetricEigen::new(m);
    let mut eigs = vec![0.0f32; n];
    for i in 0..n {
        eigs[i] = sym.eigenvalues[i] as f32;
    }
    let mut vecs = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            vecs[i * n + j] = sym.eigenvectors[(i, j)] as f32;
        }
    }
    (eigs, vecs)
}

/// Compute ||A·V - V·Λ||_F / ||A||_F (residual).
fn residual(a: &[f32], v: &[f32], eigs: &[f32], n: usize) -> f64 {
    let mut av = vec![0.0f64; n * n];
    let mut vl = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += a[i * n + k] as f64 * v[k * n + j] as f64;
            }
            av[i * n + j] = s;
        }
    }
    for i in 0..n {
        for j in 0..n {
            vl[i * n + j] = v[i * n + j] as f64 * eigs[j] as f64;
        }
    }
    let mut num = 0.0f64;
    for i in 0..n * n {
        num += (av[i] - vl[i]).powi(2);
    }
    let mut den = 0.0f64;
    for i in 0..n * n {
        den += a[i] as f64 * a[i] as f64;
    }
    (num.sqrt()) / den.sqrt().max(1e-30)
}

/// Compute ||V^T·V - I||_F / N (orthogonality).
fn orthogonality(v: &[f32], n: usize) -> f64 {
    let mut vtv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += v[k * n + i] as f64 * v[k * n + j] as f64;
            }
            vtv[i * n + j] = s;
        }
    }
    let mut num = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let d = vtv[i * n + j] - if i == j { 1.0 } else { 0.0 };
            num += d * d;
        }
    }
    num.sqrt() / n as f64
}

#[test]
fn test_tiled_jacobi_residual_orthogonality() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let batch = 2usize;
    for &n in &[65, 87, 96, 97, 128] {
        let mut all_a = Vec::new();
        let mut all_v = Vec::new();
        for b in 0..batch {
            let a = random_symmetric(n, 42 + b as u64);
            all_a.extend(a);
            all_v.extend(vec![0.0f32; n * n]);
        }
        // Save original A for residual computation
        let all_a_orig = all_a.clone();
        let a_buf = rt.buffer_from_slice(&all_a).unwrap();
        let v_buf = rt.buffer_from_slice(&all_v).unwrap();

        tiled_jacobi_batched(&mut rt, &a_buf, &v_buf, n, batch).expect("tiled Jacobi must succeed");

        let mut gpu_a = vec![0.0f32; batch * n * n];
        let mut gpu_v = vec![0.0f32; batch * n * n];
        rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
        rt.read_buffer(&v_buf, &mut gpu_v).unwrap();

        for b in 0..batch {
            let a_orig = &all_a_orig[b * n * n..(b + 1) * n * n]; // ORIGINAL A for residual
            let v_slice = &gpu_v[b * n * n..(b + 1) * n * n];
            let mut eigs = vec![0.0f32; n];
            for i in 0..n {
                eigs[i] = gpu_a[b * n * n + i * n + i];
            }
            let res = residual(a_orig, v_slice, &eigs, n);
            let orth = orthogonality(v_slice, n);
            eprintln!("tiled Jacobi N={n} batch {b}: residual={res:.2e}, orthogonality={orth:.2e}");
            assert!(
                res < 1e-5,
                "tiled Jacobi N={n} residual {res:.2e} too large (target 1e-5)"
            );
            assert!(
                orth < 1e-5,
                "tiled Jacobi N={n} orthogonality {orth:.2e} too large (target 1e-5)"
            );
        }
    }
}

#[test]
fn test_tiled_jacobi_eigenvalue_parity() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let batch = 1usize;
    for &n in &[65, 87, 96, 97, 128] {
        let a = random_symmetric(n, 42);
        let (cpu_eigs, _cpu_vecs) = cpu_eig(&a, n);

        let a_buf = rt.buffer_from_slice(&a).unwrap();
        let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();

        tiled_jacobi_batched(&mut rt, &a_buf, &v_buf, n, batch).expect("tiled Jacobi must succeed");

        let mut gpu_a = vec![0.0f32; batch * n * n];
        rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
        let mut gpu_eigs = vec![0.0f32; n];
        for i in 0..n {
            gpu_eigs[i] = gpu_a[i * n + i];
        }

        // Sort both for comparison
        let mut cpu_sorted = cpu_eigs.clone();
        cpu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut gpu_sorted = gpu_eigs.clone();
        gpu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut max_diff = 0.0f64;
        for i in 0..n {
            let d = (gpu_sorted[i] as f64 - cpu_sorted[i] as f64).abs();
            max_diff = max_diff.max(d);
        }
        eprintln!("tiled Jacobi N={n}: eigenvalue parity max|dλ|={max_diff:.2e}");
        // GPT 5.6 target: eigenvalue parity < 1e-4 Ha
        assert!(
            max_diff < 1e-4,
            "tiled Jacobi N={n} eigenvalue parity {max_diff:.2e} too large (target 1e-4)"
        );
    }
}

/// §12 D2: 3-mode precision benchmark for the tiled Jacobi.
/// Reports OpenCL event time, worst residual/orthogonality/eigenvalue parity
/// per mode. Run:
///   cargo test --release --test gpu_tiled_jacobi jacobi_prec_bench -- --ignored --nocapture
#[test]
#[ignore]
fn jacobi_prec_bench() {
    use ocl::{enums::ProfilingInfo, flags, Event, Kernel, Queue};
    use rust_dftb::qmqm::gpu_eigen::render_tiled_source;

    let Some(mut rt) = try_runtime() else {
        return;
    };
    let queue = Queue::new(
        rt.context(),
        *rt.device(),
        Some(flags::CommandQueueProperties::new().profiling()),
    )
    .expect("profiling queue for Jacobi prec bench");
    let repeats = 5usize;
    for &n in &[87usize, 128] {
        for &batch in &[1usize, 8] {
            let mut input = vec![0.0f32; batch * n * n];
            for b in 0..batch {
                let a = random_symmetric(n, 42 + b as u64);
                input[b * n * n..(b + 1) * n * n].copy_from_slice(&a);
            }
            let orig = input.clone();
            let src = rt.buffer_from_slice(&input).unwrap();
            let a_buf = rt.zero_buffer::<f32>(input.len()).unwrap();
            let v_buf = rt.zero_buffer::<f32>(input.len()).unwrap();
            let wids = rt
                .buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())
                .unwrap(); // T06 identity
            let mut ah = vec![0.0f32; input.len()];
            let mut vh = vec![0.0f32; input.len()];
            for prec in 0..3u32 {
                let source = render_tiled_source(32, 256, prec);
                let program = rt
                    .build_program(&source)
                    .expect("build Jacobi prec variant");
                let kernel = Kernel::builder()
                    .program(&program)
                    .name("tiled_jacobi_batched")
                    .queue(queue.clone())
                    .global_work_size(batch * 256)
                    .local_work_size(256)
                    .arg(&a_buf)
                    .arg(&v_buf)
                    .arg(n as i32)
                    .arg(batch as i32)
                    .arg(&wids)
                    .build()
                    .expect("build Jacobi prec kernel");
                let mut times = Vec::new();
                for r in 0..=repeats {
                    src.cmd()
                        .queue(&queue)
                        .copy(&a_buf, None, None)
                        .enq()
                        .expect("reset Jacobi input");
                    queue.finish().expect("finish input reset");
                    let mut ev = Event::empty();
                    unsafe {
                        kernel
                            .cmd()
                            .enew(&mut ev)
                            .enq()
                            .expect("enqueue Jacobi prec");
                    }
                    ev.wait_for().expect("wait Jacobi event");
                    let t0 = ev
                        .profiling_info(ProfilingInfo::Start)
                        .unwrap()
                        .time()
                        .unwrap();
                    let t1 = ev
                        .profiling_info(ProfilingInfo::End)
                        .unwrap()
                        .time()
                        .unwrap();
                    assert!(t1 > t0, "invalid Jacobi event timestamps N={n} prec={prec}");
                    if r > 0 {
                        times.push((t1 - t0) as f64 * 1e-3);
                    }
                }
                times.sort_by(f64::total_cmp);
                a_buf
                    .cmd()
                    .queue(&queue)
                    .read(&mut ah)
                    .enq()
                    .expect("read A");
                v_buf
                    .cmd()
                    .queue(&queue)
                    .read(&mut vh)
                    .enq()
                    .expect("read V");
                queue.finish().expect("finish reads");
                assert!(
                    ah.iter().chain(vh.iter()).all(|x| x.is_finite()),
                    "non-finite Jacobi output N={n} batch={batch} prec={prec}"
                );
                let (mut wres, mut worth, mut wpar) = (0.0f64, 0.0f64, 0.0f64);
                for b in 0..batch {
                    let a0 = &orig[b * n * n..(b + 1) * n * n];
                    let v = &vh[b * n * n..(b + 1) * n * n];
                    let mut eigs = vec![0.0f32; n];
                    for i in 0..n {
                        eigs[i] = ah[b * n * n + i * n + i];
                    }
                    wres = wres.max(residual(a0, v, &eigs, n));
                    worth = worth.max(orthogonality(v, n));
                    let (ce, _) = cpu_eig(a0, n);
                    let mut gs = eigs.clone();
                    gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    let mut cs = ce.clone();
                    cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    for i in 0..n {
                        wpar = wpar.max((gs[i] as f64 - cs[i] as f64).abs());
                    }
                }
                eprintln!("JACOBI_PREC N={n} batch={batch} prec={prec}: median_event_us={:.1} residual={:.3e} orth={:.3e} eig_parity={:.3e}",
                    times[times.len() / 2], wres, worth, wpar);
            }
        }
    }
}

/// W3 (manifest §14): A/B on the PRODUCTION direct kernel
/// `jacobi_cyclic_global_batched` — prec ∈ {0,1} × WG ∈ {128,256,512},
/// event-timed. The deprecated `jacobi_prec_bench` measures the tiled path;
/// this is the kernel GpuSccPlan actually runs. Run:
///   cargo test --release --test gpu_tiled_jacobi direct_jacobi_bench -- --ignored --nocapture
#[test]
#[ignore]
fn direct_jacobi_bench() {
    use ocl::{enums::ProfilingInfo, flags, Event, Kernel, Queue};
    use rust_dftb::qmqm::gpu_eigen::render_tiled_source;

    let Some(mut rt) = try_runtime() else {
        return;
    };
    let queue = Queue::new(
        rt.context(),
        *rt.device(),
        Some(flags::CommandQueueProperties::new().profiling()),
    )
    .expect("profiling queue for direct Jacobi bench");
    let repeats = 5usize;
    for &n in &[87usize, 128] {
        for &batch in &[1usize, 8, 19] {
            let mut input = vec![0.0f32; batch * n * n];
            for b in 0..batch {
                let a = random_symmetric(n, 42 + b as u64);
                input[b * n * n..(b + 1) * n * n].copy_from_slice(&a);
            }
            let orig = input.clone();
            let src = rt.buffer_from_slice(&input).unwrap();
            let a_buf = rt.zero_buffer::<f32>(input.len()).unwrap();
            let v_buf = rt.zero_buffer::<f32>(input.len()).unwrap();
            let ones = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
            let wids = rt
                .buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())
                .unwrap(); // T06 identity
            let diag_buf = rt.zero_buffer::<f32>(4 * batch).unwrap();
            let mut ah = vec![0.0f32; input.len()];
            let mut vh = vec![0.0f32; input.len()];
            let mut dh = vec![0.0f32; 4 * batch];
            for prec in [0u32, 1] {
                for &wg in &[128usize, 256, 512] {
                    let source = render_tiled_source(32, wg, prec);
                    let program = match rt.build_program(&source) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("direct prec={prec} wg={wg}: build failed {e}");
                            continue;
                        }
                    };
                    // R5 tail args — disabled (fermi_tail=0); bound dummies.
                    let occ_w = rt.zero_buffer::<f32>(batch * n).expect("occ_w bench buf");
                    let mu = rt.zero_buffer::<f32>(batch).expect("mu bench buf");
                    let kernel = match Kernel::builder()
                        .program(&program)
                        .name("jacobi_cyclic_global_batched")
                        .queue(queue.clone())
                        .global_work_size(batch * wg)
                        .local_work_size(wg)
                        .arg(&a_buf)
                        .arg(&v_buf)
                        .arg(n as i32)
                        .arg(batch as i32)
                        .arg(0i32)
                        .arg(&ones)
                        .arg(&diag_buf)
                        .arg(0i32)
                        .arg(0i32)
                        .arg(0.0f32)
                        .arg(&occ_w)
                        .arg(&mu)
                        .arg(&wids)
                        .build()
                    {
                        Ok(k) => k,
                        Err(e) => {
                            eprintln!(
                                "direct prec={prec} wg={wg}: kernel build failed {e} (WG limit?)"
                            );
                            continue;
                        }
                    };
                    let mut times = Vec::new();
                    for r in 0..=repeats {
                        src.cmd()
                            .queue(&queue)
                            .copy(&a_buf, None, None)
                            .enq()
                            .expect("reset Jacobi input");
                        queue.finish().expect("finish input reset");
                        let mut ev = Event::empty();
                        unsafe {
                            kernel
                                .cmd()
                                .enew(&mut ev)
                                .enq()
                                .expect("enqueue direct Jacobi");
                        }
                        ev.wait_for().expect("wait direct Jacobi event");
                        let t0 = ev
                            .profiling_info(ProfilingInfo::Start)
                            .unwrap()
                            .time()
                            .unwrap();
                        let t1 = ev
                            .profiling_info(ProfilingInfo::End)
                            .unwrap()
                            .time()
                            .unwrap();
                        assert!(
                            t1 > t0,
                            "invalid event timestamps N={n} prec={prec} wg={wg}"
                        );
                        if r > 0 {
                            times.push((t1 - t0) as f64 * 1e-3);
                        }
                    }
                    times.sort_by(f64::total_cmp);
                    a_buf
                        .cmd()
                        .queue(&queue)
                        .read(&mut ah)
                        .enq()
                        .expect("read A");
                    v_buf
                        .cmd()
                        .queue(&queue)
                        .read(&mut vh)
                        .enq()
                        .expect("read V");
                    diag_buf
                        .cmd()
                        .queue(&queue)
                        .read(&mut dh)
                        .enq()
                        .expect("read diag");
                    queue.finish().expect("finish reads");
                    assert!(
                        ah.iter().chain(vh.iter()).all(|x| x.is_finite()),
                        "non-finite output N={n} batch={batch} prec={prec} wg={wg}"
                    );
                    let (mut wres, mut worth, mut wpar, mut nstop) = (0.0f64, 0.0f64, 0.0f64, 0);
                    for b in 0..batch {
                        let a0 = &orig[b * n * n..(b + 1) * n * n];
                        let v = &vh[b * n * n..(b + 1) * n * n];
                        let mut eigs = vec![0.0f32; n];
                        for i in 0..n {
                            eigs[i] = ah[b * n * n + i * n + i];
                        }
                        wres = wres.max(residual(a0, v, &eigs, n));
                        worth = worth.max(orthogonality(v, n));
                        let (ce, _) = cpu_eig(a0, n);
                        let mut gs = eigs.clone();
                        gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                        let mut cs = ce.clone();
                        cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                        for i in 0..n {
                            wpar = wpar.max((gs[i] as f64 - cs[i] as f64).abs());
                        }
                        if dh[4 * b + 2] as i32 != 0 {
                            nstop += 1;
                        }
                    }
                    eprintln!("DIRECT N={n} batch={batch} prec={prec} wg={wg}: median_us={:.1} res={:.3e} orth={:.3e} eig_par={:.3e} bad_stop={nstop}",
                        times[times.len() / 2], wres, worth, wpar);
                }
            }
        }
    }
}

/// A = Q·diag(eigs)·Qᵀ with Q from a random symmetric matrix's CPU eigvec.
fn from_spectrum(n: usize, eigs: &[f64], seed: u64) -> Vec<f32> {
    let r = random_symmetric(n, seed);
    let (_, q) = cpu_eig(&r, n);
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += q[i * n + k] as f64 * eigs[k] * q[j * n + k] as f64;
            }
            a[i * n + j] = s;
        }
    }
    a.iter().map(|&x| x as f32).collect()
}

/// R5: exercise the PRODUCTION direct kernel `jacobi_cyclic_global_batched`
/// (what GpuSccPlan actually runs for N>64 — not the deprecated tiled path)
/// on an eigen-quality matrix: boundary/odd/capacity Ns, clustered, repeated,
/// indefinite, exact zero-block and all-zero spectra. Checks residual
/// ‖AV−VΛ‖_F/‖A‖_F vs the ORIGINAL matrix, orthogonality, eigenvalue parity
/// vs CPU f64, and the per-system stop reason (0 = converged).
#[test]
fn test_direct_jacobi_eigen_quality() {
    let Some(mut rt) = try_runtime() else {
        return;
    };

    // --- boundary/capacity sweep on random symmetric ---
    for &n in &[65usize, 87, 96, 97, 128, 255, 256] {
        let a_orig = random_symmetric(n, 42);
        let a_buf = rt.buffer_from_slice(&a_orig).unwrap();
        let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
        let diag = direct_jacobi_batched(&mut rt, &a_buf, &v_buf, n, 1, 1).unwrap();
        let mut gpu_a = vec![0.0f32; n * n];
        let mut gpu_v = vec![0.0f32; n * n];
        rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
        rt.read_buffer(&v_buf, &mut gpu_v).unwrap();
        let mut eigs = vec![0.0f32; n];
        for i in 0..n {
            eigs[i] = gpu_a[i * n + i];
        }
        let res = residual(&a_orig, &gpu_v, &eigs, n);
        let orth = orthogonality(&gpu_v, n);
        let (ce, _) = cpu_eig(&a_orig, n);
        let mut gs = eigs.clone();
        gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut cs = ce.clone();
        cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let par: f64 = (0..n)
            .map(|i| (gs[i] as f64 - cs[i] as f64).abs())
            .fold(0.0, f64::max);
        // Weyl bound: sorted-eig deviation ≤ ‖A − VΛVᵀ‖_F = res·‖A‖_F.
        // A fixed absolute tol is wrong here — the parity floor scales with
        // ‖A‖_F (~N·σ for random); the check self-tightens if res improves.
        let af: f64 = a_orig
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let weyl = 2.0 * res * af + 1e-4;
        eprintln!("[R5] direct N={n}: stop={} sweeps={} res={res:.3e} orth={orth:.3e} eig_par={par:.3e} (weyl={weyl:.3e})", diag[2], diag[3]);
        assert_eq!(
            diag[2] as i32, 0,
            "direct Jacobi N={n} stop={} (1=stall 2=maxsweeps 3=cap 4=nonfinite)",
            diag[2]
        );
        assert!(res < 1e-4, "direct Jacobi N={n} residual {res:.3e}");
        assert!(orth < 1e-4, "direct Jacobi N={n} orth {orth:.3e}");
        assert!(par < weyl, "direct Jacobi N={n} eig parity {par:.3e} exceeds Weyl bound {weyl:.3e} — not explainable by the measured residual");
    }

    // --- adversarial spectra at N=87, one batch ---
    let n = 87usize;
    // b0 clustered: 5 spread + rest in a 1e-6-wide cluster at 1.0
    let mut e = vec![1.0f64; n];
    for k in 5..n {
        e[k] = 1.0 + 1e-6 * (k % 7) as f64;
    }
    e[0] = -1.5;
    e[1] = -0.3;
    e[2] = 0.1;
    e[3] = 0.5;
    e[4] = 2.0;
    let clustered = from_spectrum(n, &e, 7);
    // b1 repeated: two identical-eigenvalue clusters
    let e: Vec<f64> = (0..n).map(|k| if k < n / 2 { 0.5 } else { 1.5 }).collect();
    let repeated = from_spectrum(n, &e, 11);
    // b2 exact zero 8×8 subblock: last 8 rows/cols are literal zeros while
    // the leading block is dense — the all-zero pivot (a_pp=a_qq=a_pq=0)
    // must be skipped, not turned into a 0/0 rotation.
    let mut zero_blk = random_symmetric(n, 13);
    for i in (n - 8)..n {
        for j in 0..n {
            zero_blk[i * n + j] = 0.0;
            zero_blk[j * n + i] = 0.0;
        }
    }
    // b3 all-zero matrix: every pivot is the pathological exact-zero case
    let all_zero = vec![0.0f32; n * n];

    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("clustered", clustered),
        ("repeated", repeated),
        ("zero_blk", zero_blk),
        ("all_zero", all_zero),
    ];
    let batch = cases.len();
    let mut a_flat = Vec::new();
    for (_, a) in &cases {
        a_flat.extend_from_slice(a);
    }
    let orig = a_flat.clone();
    let a_buf = rt.buffer_from_slice(&a_flat).unwrap();
    let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    let diag = direct_jacobi_batched(&mut rt, &a_buf, &v_buf, n, batch, 1).unwrap();
    let mut gpu_a = vec![0.0f32; batch * n * n];
    let mut gpu_v = vec![0.0f32; batch * n * n];
    rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
    rt.read_buffer(&v_buf, &mut gpu_v).unwrap();
    for (b, (name, _)) in cases.iter().enumerate() {
        let a0 = &orig[b * n * n..(b + 1) * n * n];
        let v = &gpu_v[b * n * n..(b + 1) * n * n];
        assert!(
            v.iter().all(|x| x.is_finite()),
            "direct Jacobi {name}: non-finite eigenvectors"
        );
        let mut eigs = vec![0.0f32; n];
        for i in 0..n {
            eigs[i] = gpu_a[b * n * n + i * n + i];
        }
        assert!(
            eigs.iter().all(|x| x.is_finite()),
            "direct Jacobi {name}: non-finite eigenvalues"
        );
        let res = residual(a0, v, &eigs, n);
        let orth = orthogonality(v, n);
        let stop = diag[4 * b + 2] as i32;
        eprintln!(
            "[R5] direct N=87 {name}: stop={stop} sweeps={} res={res:.3e} orth={orth:.3e}",
            diag[4 * b + 3]
        );
        assert_eq!(stop, 0, "direct Jacobi {name}: stop={stop}");
        assert!(res < 1e-4, "direct Jacobi {name} residual {res:.3e}");
        assert!(orth < 1e-4, "direct Jacobi {name} orth {orth:.3e}");
    }
}

#[test]
fn test_jacobi_batched_dispatcher() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    // Verify the dispatcher routes correctly: N=64 → full-local, N=65 → tiled
    for &n in &[64, 65] {
        let a = random_symmetric(n, 42);
        let a_orig = a.clone();
        let a_buf = rt.buffer_from_slice(&a).unwrap();
        let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
        jacobi_batched(&mut rt, &a_buf, &v_buf, n, 1)
            .expect("jacobi_batched dispatcher must succeed");
        let mut gpu_a = vec![0.0f32; n * n];
        rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
        let mut gpu_v = vec![0.0f32; n * n];
        rt.read_buffer(&v_buf, &mut gpu_v).unwrap();
        let mut eigs = vec![0.0f32; n];
        for i in 0..n {
            eigs[i] = gpu_a[i * n + i];
        }
        let res = residual(&a_orig, &gpu_v, &eigs, n);
        eprintln!("jacobi_batched N={n}: residual={res:.2e}");
        assert!(
            res < 1e-5,
            "jacobi_batched N={n} residual {res:.2e} too large (target 1e-5)"
        );
    }
}

/// §16.D: `block_jacobi_1wg` — the new one-WG/one-thread-per-row block
/// Jacobi eigensolver (pivot+U in local, A/V streamed). Covers ALL n:
/// the single-block path (n≤32), the block-pair path (n>32), boundary
/// sizes, adversarial spectra, and the warm-path probe exit (exactly
/// diagonal input → 0 sweeps). Same quality gates as the direct kernel.
#[test]
fn test_block_jacobi_eigen_quality() {
    let Some(mut rt) = try_runtime() else {
        return;
    };

    for &n in &[
        6usize, 16, 31, 32, 33, 48, 64, 65, 86, 87, 96, 128, 246, 256,
    ] {
        let a_orig = random_symmetric(n, 42);
        let a_buf = rt.buffer_from_slice(&a_orig).unwrap();
        let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
        let diag = block_jacobi_batched(&mut rt, &a_buf, &v_buf, n, 1).unwrap();
        let mut gpu_a = vec![0.0f32; n * n];
        let mut gpu_v = vec![0.0f32; n * n];
        rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
        rt.read_buffer(&v_buf, &mut gpu_v).unwrap();
        let mut eigs = vec![0.0f32; n];
        for i in 0..n {
            eigs[i] = gpu_a[i * n + i];
        }
        let res = residual(&a_orig, &gpu_v, &eigs, n);
        let orth = orthogonality(&gpu_v, n);
        let (ce, _) = cpu_eig(&a_orig, n);
        let mut gs = eigs.clone();
        gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut cs = ce.clone();
        cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let par: f64 = (0..n)
            .map(|i| (gs[i] as f64 - cs[i] as f64).abs())
            .fold(0.0, f64::max);
        let af: f64 = a_orig
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let weyl = 2.0 * res * af + 1e-4;
        eprintln!("[block] N={n}: stop={} sweeps={} res={res:.3e} orth={orth:.3e} eig_par={par:.3e} (weyl={weyl:.3e})",
            diag[2], diag[3]);
        assert_eq!(
            diag[2] as i32, 0,
            "block Jacobi N={n} stop={} (1=stall 2=maxsweeps 4=nonfinite)",
            diag[2]
        );
        assert!(res < 1e-4, "block Jacobi N={n} residual {res:.3e}");
        assert!(orth < 1e-4, "block Jacobi N={n} orth {orth:.3e}");
        assert!(
            par < weyl,
            "block Jacobi N={n} eig parity {par:.3e} exceeds Weyl bound {weyl:.3e}"
        );
    }

    // --- adversarial spectra at N=87, one batch (same cases as direct) ---
    let n = 87usize;
    let mut e = vec![1.0f64; n];
    for k in 5..n {
        e[k] = 1.0 + 1e-6 * (k % 7) as f64;
    }
    e[0] = -1.5;
    e[1] = -0.3;
    e[2] = 0.1;
    e[3] = 0.5;
    e[4] = 2.0;
    let clustered = from_spectrum(n, &e, 7);
    let e: Vec<f64> = (0..n).map(|k| if k < n / 2 { 0.5 } else { 1.5 }).collect();
    let repeated = from_spectrum(n, &e, 11);
    let mut zero_blk = random_symmetric(n, 13);
    for i in (n - 8)..n {
        for j in 0..n {
            zero_blk[i * n + j] = 0.0;
            zero_blk[j * n + i] = 0.0;
        }
    }
    let all_zero = vec![0.0f32; n * n];
    // b4 exactly-diagonal: probe must exit with 0 sweeps (warm-path cost).
    let mut diag_only = vec![0.0f32; n * n];
    for i in 0..n {
        diag_only[i * n + i] = (i as f32) * 0.1 - 4.0;
    }

    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("clustered", clustered),
        ("repeated", repeated),
        ("zero_blk", zero_blk),
        ("all_zero", all_zero),
        ("diag_only", diag_only),
    ];
    let batch = cases.len();
    let mut a_flat = Vec::new();
    for (_, a) in &cases {
        a_flat.extend_from_slice(a);
    }
    let orig = a_flat.clone();
    let a_buf = rt.buffer_from_slice(&a_flat).unwrap();
    let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    let diag = block_jacobi_batched(&mut rt, &a_buf, &v_buf, n, batch).unwrap();
    let mut gpu_a = vec![0.0f32; batch * n * n];
    let mut gpu_v = vec![0.0f32; batch * n * n];
    rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
    rt.read_buffer(&v_buf, &mut gpu_v).unwrap();
    for (b, (name, _)) in cases.iter().enumerate() {
        let a0 = &orig[b * n * n..(b + 1) * n * n];
        let v = &gpu_v[b * n * n..(b + 1) * n * n];
        assert!(
            v.iter().all(|x| x.is_finite()),
            "block Jacobi {name}: non-finite eigenvectors"
        );
        let mut eigs = vec![0.0f32; n];
        for i in 0..n {
            eigs[i] = gpu_a[b * n * n + i * n + i];
        }
        assert!(
            eigs.iter().all(|x| x.is_finite()),
            "block Jacobi {name}: non-finite eigenvalues"
        );
        let res = residual(a0, v, &eigs, n);
        let orth = orthogonality(v, n);
        let stop = diag[4 * b + 2] as i32;
        let nsw = diag[4 * b + 3];
        eprintln!("[block] N=87 {name}: stop={stop} sweeps={nsw} res={res:.3e} orth={orth:.3e}");
        assert_eq!(stop, 0, "block Jacobi {name}: stop={stop}");
        assert!(res < 1e-4, "block Jacobi {name} residual {res:.3e}");
        assert!(orth < 1e-4, "block Jacobi {name} orth {orth:.3e}");
        if *name == "diag_only" || *name == "all_zero" {
            assert_eq!(
                nsw, 0.0,
                "block Jacobi {name}: probe should exit with 0 sweeps, got {nsw}"
            );
        }
    }
}

/// T01 (Dense_Multi tasks): deterministic lane-coverage regression for the
/// block-Jacobi WG reductions. The `o = lsz>>1` halving drops lanes for
/// non-power-of-two WGs — at WG96 lanes ≡2 mod 3 never reach reduce[0]
/// (96→64 effective). A 0.01 impulse on rows (2,5) — both dropped — makes
/// the broken probe report off=0 and exit with 0 sweeps, leaving the
/// rotation unapplied (true residual ≈ √2·0.01/√86 ≈ 1.5e-3).
#[test]
fn test_block_jacobi_wg96_impulse_reduction() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let n = 86usize;
    let eps = 0.01f32;
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    a[2 * n + 5] = eps;
    a[5 * n + 2] = eps;
    let a_orig = a.clone();
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
    let diag = block_jacobi_batched(&mut rt, &a_buf, &v_buf, n, 1).unwrap();
    let mut gpu_a = vec![0.0f32; n * n];
    let mut gpu_v = vec![0.0f32; n * n];
    rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
    rt.read_buffer(&v_buf, &mut gpu_v).unwrap();
    let mut eigs = vec![0.0f32; n];
    for i in 0..n {
        eigs[i] = gpu_a[i * n + i];
    }
    let res = residual(&a_orig, &gpu_v, &eigs, n);
    let true_off = (2.0f64).sqrt() * eps as f64;
    eprintln!("[T01] N=86 impulse(2,5): stop={} sweeps={} reported_off={:.3e} res={res:.3e} (true initial off={true_off:.3e})",
        diag[2], diag[3], diag[0]);
    assert!(diag[3] >= 1.0,
        "block Jacobi probe missed the impulse (reported off={:.3e}, true {true_off:.3e}) — a WG96 reduction dropped the rows", diag[0]);
    assert!(
        res < 1e-4,
        "impulse residual {res:.3e} — the rotation was never applied"
    );
    assert_eq!(
        diag[2] as i32, 0,
        "impulse solve should converge, stop={}",
        diag[2]
    );
}

/// T01: the probe's row reduction must count EVERY lane, incl. the
/// non-power-of-two WG tail — not just a known-dropped residue class.
/// One small impulse on every super/sub-diagonal element; the total is
/// kept below off_exit so the kernel exits at the probe and reports the
/// measured norm in diag[0] — a direct count of which rows contributed.
#[test]
fn test_block_jacobi_probe_counts_all_rows() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let eps = 5.0e-7f32;
    for &n in &[
        33usize, 48, 64, 86, 87, 96, 128, 160, 161, 192, 224, 225, 246, 256,
    ] {
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        for i in 0..n - 1 {
            a[i * n + i + 1] = eps;
            a[(i + 1) * n + i] = eps;
        }
        let off_true = eps as f64 * (2.0 * (n - 1) as f64).sqrt();
        let a_buf = rt.buffer_from_slice(&a).unwrap();
        let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
        let diag = block_jacobi_batched(&mut rt, &a_buf, &v_buf, n, 1).unwrap();
        let off_rep = diag[0] as f64;
        eprintln!(
            "[T01] probe N={n}: reported off={off_rep:.4e} true={off_true:.4e} stop={} sweeps={}",
            diag[2], diag[3]
        );
        assert_eq!(
            diag[3], 0.0,
            "N={n}: expected probe exit (0 sweeps), got {}",
            diag[3]
        );
        assert_eq!(
            diag[2] as i32, 0,
            "N={n}: expected converged probe exit, stop={}",
            diag[2]
        );
        let rel = (off_rep - off_true).abs() / off_true;
        assert!(rel < 2.0e-2,
            "N={n}: probe off-norm {off_rep:.4e} vs true {off_true:.4e} (rel err {rel:.3e}) — reduction dropped lanes");
    }
}

/// T01: the n≤PB single-block path must report honest status — an
/// exhausted INNER_MAX with a finite residual above tolerance is a
/// NON-converged solve (stop≠0), not silent success. Renders the
/// template with INNER_MAX=1 so the cap is guaranteed hit on a dense
/// random pivot.
#[test]
fn test_block_jacobi_inner_cap_reports_failure() {
    use ocl::Kernel;
    use rust_dftb::qmqm::gpu_eigen::{block_jacobi_wg, render_block_source};
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let n = 24usize; // ≤ PB=32 → single-block path
    let a_orig = random_symmetric(n, 42);
    let a_buf = rt.buffer_from_slice(&a_orig).unwrap();
    let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
    let ones = rt.buffer_from_slice(&vec![1i32; 1]).unwrap();
    let wids = rt.buffer_from_slice(&[0i32]).unwrap(); // T06 identity
    let diag_buf = rt.zero_buffer::<f32>(4).unwrap();
    let eig_buf = rt.zero_buffer::<f32>(n).unwrap();
    let wg = block_jacobi_wg(n);
    let src = render_block_source(16, wg).replace("#define INNER_MAX 12", "#define INNER_MAX 1");
    assert!(
        src.contains("#define INNER_MAX 1"),
        "INNER_MAX specialization failed"
    );
    let program = rt.build_program(&src).unwrap();
    let kernel = Kernel::builder()
        .program(&program)
        .name("block_jacobi_1wg")
        .queue(rt.queue().clone())
        .global_work_size(wg)
        .local_work_size(wg)
        .arg(&a_buf)
        .arg(&v_buf)
        .arg(n as i32)
        .arg(1i32)
        .arg(0i32)
        .arg(&ones)
        .arg(&diag_buf)
        .arg(&eig_buf)
        .arg(&wids)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
    let mut d = vec![0.0f32; 4];
    rt.read_buffer(&diag_buf, &mut d).unwrap();
    eprintln!(
        "[T01] INNER_MAX=1 N={n}: stop={} off={:.3e} rel={:.3e}",
        d[2], d[0], d[1]
    );
    assert_ne!(d[2] as i32, 0,
        "n≤PB path claimed success with INNER_MAX=1 — finite residual above tolerance must report stop≠0");

    // The first-failure latch held stop=2 — a second launch in the same
    // window must NOT overwrite it. Verify, then clear (new window) and
    // check the nonfinite input is reported, not silently certified (4).
    let mut bad = a_orig.clone();
    bad[3 * n + 7] = f32::NAN;
    a_buf.write(&bad).enq().unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
    rt.read_buffer(&diag_buf, &mut d).unwrap();
    eprintln!(
        "[T01] latch check N={n}: stop={} (first failure preserved)",
        d[2]
    );
    assert_eq!(
        d[2] as i32, 2,
        "first failure must survive a later launch, got {}",
        d[2]
    );
    diag_buf.cmd().fill(0.0f32, None).enq().unwrap(); // new solve window
    unsafe {
        kernel.enq().unwrap();
    }
    rt.read_buffer(&diag_buf, &mut d).unwrap();
    eprintln!("[T01] NaN input N={n}: stop={}", d[2]);
    assert_eq!(d[2] as i32, 4, "NaN input must report stop=4, got {}", d[2]);
}

/// §16.D A/B: block_jacobi_1wg vs jacobi_cyclic_global_batched at
/// production batch sizes. Programs built once; each rep re-uploads the
/// ORIGINAL matrices then enqueues + finish (upload included in both
/// paths symmetrically). Three regimes: cold (random), one (just above
/// tolerance), warm (0 sweeps — the common SCC iteration).
/// `cargo test --release --test gpu_tiled_jacobi -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_block_vs_direct_jacobi_bench() {
    use ocl::Kernel;
    use rust_dftb::qmqm::gpu_eigen::{block_jacobi_wg, render_block_source};
    use std::time::Instant;
    let Some(mut rt) = try_runtime() else {
        return;
    };

    // direct kernel program (prec=0, WG=512 — production settings)
    let wg_d = 512usize.min(rt.caps().max_work_group_size);
    let src_d = rust_dftb::qmqm::gpu_eigen::render_tiled_source(32, wg_d, 0);
    let prog_d = rt.build_program(&src_d).unwrap();

    // Kernel resource footprint → theoretical WGs/SM (occupancy ceiling).
    let print_res = |rt: &GpuRuntime, k: &Kernel, tag: &str| {
        use ocl::enums::KernelWorkGroupInfo::*;
        let g = |i| {
            k.wg_info(*rt.device(), i)
                .map(|r| format!("{r:?}"))
                .unwrap_or_else(|_| "?".into())
        };
        eprintln!(
            "[res] {tag}: local={} priv={} maxwg={}",
            g(LocalMemSize),
            g(PrivateMemSize),
            g(WorkGroupSize)
        );
    };

    let mk = |n: usize, batch: usize, mode: &str, seed: u64| -> Vec<f32> {
        let mut a = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let mut m = random_symmetric(n, seed + b as u64);
            match mode {
                "warm" => {
                    for i in 0..n {
                        for j in 0..n {
                            if i != j {
                                m[i * n + j] *= 1e-9;
                            }
                        }
                    }
                }
                "one" => {
                    for i in 0..n {
                        for j in 0..n {
                            if i != j {
                                m[i * n + j] *= 3e-3;
                            }
                        }
                    }
                }
                _ => {}
            }
            a.extend_from_slice(&m);
        }
        a
    };

    eprintln!("[bench] n batch mode | direct ms (sweeps) | block ms (sweeps) | ratio");
    for &n in &[86usize, 246] {
        let wg_b = block_jacobi_wg(n);
        let src_b = render_block_source(16, wg_b);
        let prog_b = rt.build_program(&src_b).unwrap();
        for &batch in &[400usize, 80, 20] {
            for mode in ["warm", "one", "cold"] {
                let a = mk(n, batch, mode, 42);
                let a_buf = rt.buffer_from_slice(&a).unwrap();
                let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
                let ones = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
                let wids = rt
                    .buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())
                    .unwrap(); // T06 identity
                let diag_d = rt.zero_buffer::<f32>(4 * batch).unwrap();
                let occ_w = rt.zero_buffer::<f32>(batch * n).unwrap();
                let mu = rt.zero_buffer::<f32>(batch).unwrap();
                let k_d = Kernel::builder()
                    .program(&prog_d)
                    .name("jacobi_cyclic_global_batched")
                    .queue(rt.queue().clone())
                    .global_work_size(batch * wg_d)
                    .local_work_size(wg_d)
                    .arg(&a_buf)
                    .arg(&v_buf)
                    .arg(n as i32)
                    .arg(batch as i32)
                    .arg(0i32)
                    .arg(&ones)
                    .arg(&diag_d)
                    .arg(0i32)
                    .arg(0i32)
                    .arg(0.0f32)
                    .arg(&occ_w)
                    .arg(&mu)
                    .arg(&wids)
                    .build()
                    .unwrap();
                let diag_b = rt.zero_buffer::<f32>(4 * batch).unwrap();
                let eig_b = rt.zero_buffer::<f32>(batch * n).unwrap();
                let k_b = Kernel::builder()
                    .program(&prog_b)
                    .name("block_jacobi_1wg")
                    .queue(rt.queue().clone())
                    .global_work_size(batch * wg_b)
                    .local_work_size(wg_b)
                    .arg(&a_buf)
                    .arg(&v_buf)
                    .arg(n as i32)
                    .arg(batch as i32)
                    .arg(0i32)
                    .arg(&ones)
                    .arg(&diag_b)
                    .arg(&eig_b)
                    .arg(&wids)
                    .build()
                    .unwrap();
                if batch == 400 && mode == "warm" {
                    print_res(&rt, &k_d, &format!("direct n={n} wg={wg_d}"));
                    print_res(&rt, &k_b, &format!("block  n={n} wg={wg_b}"));
                }

                let run = |rt: &mut GpuRuntime,
                           k: &Kernel,
                           a_buf: &ocl::Buffer<f32>,
                           a: &[f32]|
                 -> std::time::Duration {
                    rt.write_buffer(a_buf, a).unwrap();
                    let t0 = Instant::now();
                    unsafe {
                        k.enq().unwrap();
                    }
                    rt.finish().unwrap();
                    t0.elapsed()
                };
                // warm-up once, then 3 timed reps
                run(&mut rt, &k_d, &a_buf, &a);
                let td: f64 = (0..3)
                    .map(|_| run(&mut rt, &k_d, &a_buf, &a).as_secs_f64())
                    .sum::<f64>()
                    / 3.0;
                run(&mut rt, &k_b, &a_buf, &a);
                let tb: f64 = (0..3)
                    .map(|_| run(&mut rt, &k_b, &a_buf, &a).as_secs_f64())
                    .sum::<f64>()
                    / 3.0;

                let mut d4 = vec![0.0f32; 4 * batch];
                rt.read_buffer(&diag_d, &mut d4).unwrap();
                let (sw_d, bad_d) = (0..batch).fold((0.0f32, 0usize), |(s, c), b| {
                    (s + d4[4 * b + 3], c + (d4[4 * b + 2] != 0.0) as usize)
                });
                rt.read_buffer(&diag_b, &mut d4).unwrap();
                let (sw_b, bad_b) = (0..batch).fold((0.0f32, 0usize), |(s, c), b| {
                    (s + d4[4 * b + 3], c + (d4[4 * b + 2] != 0.0) as usize)
                });
                eprintln!("[bench] n={n} batch={batch} {mode}: direct {:.2} ms (sw {:.1}, bad {bad_d}) | block {:.2} ms (sw {:.1}, bad {bad_b}) | {:.2}×",
                    td * 1e3, sw_d / batch as f32, tb * 1e3, sw_b / batch as f32, td / tb);
            }
        }
    }
}

/// T08 sweep: block-Jacobi B × INNER_TOL × INNER_MAX on identical inputs.
/// Equal-work AND equal-accuracy: every config must converge (stop=0) under
/// the same global off/‖A‖_F<1e-6 exit; residual/orth/eig-parity are
/// spot-checked on the first systems so no config "wins" by under-converging.
/// Modes: "one" (just above tolerance — the typical mid-SCC iteration),
/// "cold" (random — first-iteration regime). Warm (probe exit) is skipped:
/// solver params do not affect it.
///   cargo test --release --test gpu_tiled_jacobi block_jacobi_param_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn block_jacobi_param_sweep() {
    use ocl::Kernel;
    use rust_dftb::qmqm::gpu_eigen::{block_jacobi_wg, render_block_source_cfg};
    use std::time::Instant;
    let Some(mut rt) = try_runtime() else {
        return;
    };

    let mk = |n: usize, batch: usize, mode: &str, seed: u64| -> Vec<f32> {
        let mut a = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let mut m = random_symmetric(n, seed + b as u64);
            if mode == "one" {
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            m[i * n + j] *= 3e-3;
                        }
                    }
                }
            }
            a.extend_from_slice(&m);
        }
        a
    };

    // (B, INNER_MAX, INNER_TOL): stage 1 = B×ITOL at IMAX=12;
    // stage 2 = IMAX sweep at ITOL=1e-6.
    let cfgs: &[(usize, usize, f32)] = &[
        (16, 12, 1e-7),
        (16, 12, 1e-6),
        (16, 12, 1e-5),
        (24, 12, 1e-7),
        (24, 12, 1e-6),
        (24, 12, 1e-5),
        (32, 12, 1e-7),
        (32, 12, 1e-6),
        (32, 12, 1e-5),
        (16, 4, 1e-6),
        (16, 2, 1e-6),
        (24, 4, 1e-6),
        (24, 2, 1e-6),
        (24, 1, 1e-6),
        (32, 4, 1e-6),
        (32, 2, 1e-6),
        (32, 1, 1e-6),
    ];

    for &n in &[246usize, 86] {
        let wg = block_jacobi_wg(n);
        for mode in ["one", "cold"] {
            let batch = 400usize;
            let a = mk(n, batch, mode, 42);
            let a_buf = rt.buffer_from_slice(&a).unwrap();
            let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
            let ones = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
            let wids = rt
                .buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())
                .unwrap(); // T06 identity
            let diag = rt.zero_buffer::<f32>(4 * batch).unwrap();
            let eig = rt.zero_buffer::<f32>(batch * n).unwrap();
            for &(b, imax, itol) in cfgs {
                let src = render_block_source_cfg(b, wg, imax, itol);
                assert!(
                    src.contains(&format!("#define INNER_MAX {imax}")),
                    "INNER_MAX replace no-op"
                );
                assert!(src.contains(&format!("#define B {b}\n")), "B replace no-op");
                let program = match rt.build_program(&src) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!(
                            "[sweep] n={n} {mode} B={b} IMAX={imax} ITOL={itol:e}: BUILD FAIL {e}"
                        );
                        continue;
                    }
                };
                let k = match Kernel::builder()
                    .program(&program)
                    .name("block_jacobi_1wg")
                    .queue(rt.queue().clone())
                    .global_work_size(batch * wg)
                    .local_work_size(wg)
                    .arg(&a_buf)
                    .arg(&v_buf)
                    .arg(n as i32)
                    .arg(batch as i32)
                    .arg(0i32)
                    .arg(&ones)
                    .arg(&diag)
                    .arg(&eig)
                    .arg(&wids)
                    .build()
                {
                    Ok(k) => k,
                    Err(e) => {
                        eprintln!(
                            "[sweep] n={n} {mode} B={b} IMAX={imax} ITOL={itol:e}: KERNEL FAIL {e}"
                        );
                        continue;
                    }
                };
                let run = |rt: &mut GpuRuntime| -> std::time::Duration {
                    rt.write_buffer(&a_buf, &a).unwrap();
                    let t0 = Instant::now();
                    unsafe {
                        k.enq().unwrap();
                    }
                    rt.finish().unwrap();
                    t0.elapsed()
                };
                run(&mut rt);
                let t: f64 = (0..3).map(|_| run(&mut rt).as_secs_f64()).sum::<f64>() / 3.0;
                let mut d4 = vec![0.0f32; 4 * batch];
                rt.read_buffer(&diag, &mut d4).unwrap();
                let (sw, bad, off) = (0..batch).fold((0.0f32, 0usize, 0.0f32), |(s, c, m), b| {
                    (
                        s + d4[4 * b + 3],
                        c + (d4[4 * b + 2] != 0.0) as usize,
                        m.max(d4[4 * b + 1]),
                    )
                });
                // equal-accuracy spot check: residual/orth/eig-parity on 2 systems
                let mut ah = vec![0.0f32; n * n];
                let mut vh = vec![0.0f32; n * n];
                let mut wres = 0.0f64;
                let mut worth = 0.0f64;
                let mut wpar = 0.0f64;
                for sb in 0..2 {
                    rt.queue().finish().unwrap();
                    let mut tmp = vec![0.0f32; batch * n * n];
                    rt.read_buffer(&a_buf, &mut tmp).unwrap();
                    ah.copy_from_slice(&tmp[sb * n * n..(sb + 1) * n * n]);
                    rt.read_buffer(&v_buf, &mut tmp).unwrap();
                    vh.copy_from_slice(&tmp[sb * n * n..(sb + 1) * n * n]);
                    let a0 = &a[sb * n * n..(sb + 1) * n * n];
                    let mut eigs = vec![0.0f32; n];
                    for i in 0..n {
                        eigs[i] = ah[i * n + i];
                    }
                    wres = wres.max(residual(a0, &vh, &eigs, n));
                    worth = worth.max(orthogonality(&vh, n));
                    let (ce, _) = cpu_eig(a0, n);
                    let mut gs = eigs.clone();
                    gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    let mut cs = ce.clone();
                    cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    for i in 0..n {
                        wpar = wpar.max((gs[i] as f64 - cs[i] as f64).abs());
                    }
                }
                eprintln!("[sweep] n={n} {mode} B={b} IMAX={imax} ITOL={itol:e}: {:.2}ms sw={:.1} bad={bad} off={:.2e} res={:.2e} orth={:.2e} par={:.2e}",
                    t * 1e3, sw / batch as f32, off, wres, worth, wpar);
            }
        }
    }
}

/// T08 #7: kernel resource footprint for block_jacobi_1wg at each B —
/// ncu cannot profile OpenCL kernels, so CL_KERNEL_*_MEM_SIZE queries are
/// the available truth for the private-array spilling hypothesis (x[PB]
/// + xv[PB] per thread, dynamically indexed). Also reports for the direct
/// kernel with and without the Fermi tail compiled in.
///   cargo test --release --test gpu_tiled_jacobi kernel_resources -- --ignored --nocapture
#[test]
#[ignore]
fn kernel_resources() {
    use ocl::enums::KernelWorkGroupInfo::*;
    use ocl::Kernel;
    use rust_dftb::qmqm::gpu_eigen::{
        block_jacobi_wg, render_block_source_cfg, render_tiled_source_cfg,
    };
    let Some(mut rt) = try_runtime() else {
        return;
    };

    let dev = *rt.device();
    let print = |k: &Kernel, tag: &str| {
        let g = |i| {
            k.wg_info(dev, i)
                .map(|r| format!("{r:?}"))
                .unwrap_or_else(|_| "?".into())
        };
        eprintln!(
            "[res] {tag}: local={}B priv={}B maxwg={}",
            g(LocalMemSize),
            g(PrivateMemSize),
            g(WorkGroupSize)
        );
    };
    let n = 246usize;
    let a = rt.zero_buffer::<f32>(n * n).unwrap();
    let v = rt.zero_buffer::<f32>(n * n).unwrap();
    let ones = rt.buffer_from_slice(&[1i32]).unwrap();
    let wids = rt.buffer_from_slice(&[0i32]).unwrap(); // T06 identity
    let diag = rt.zero_buffer::<f32>(4).unwrap();
    let eig = rt.zero_buffer::<f32>(n).unwrap();
    let occ_w = rt.zero_buffer::<f32>(n).unwrap();
    let mu = rt.zero_buffer::<f32>(1).unwrap();
    for &b in &[16usize, 24, 32] {
        let wg = block_jacobi_wg(n);
        for &imax in &[12usize, 1] {
            let src = render_block_source_cfg(b, wg, imax, 1e-6);
            let program = rt.build_program(&src).unwrap();
            let k = Kernel::builder()
                .program(&program)
                .name("block_jacobi_1wg")
                .queue(rt.queue().clone())
                .global_work_size(wg)
                .local_work_size(wg)
                .arg(&a)
                .arg(&v)
                .arg(n as i32)
                .arg(1i32)
                .arg(0i32)
                .arg(&ones)
                .arg(&diag)
                .arg(&eig)
                .arg(&wids)
                .build()
                .unwrap();
            print(&k, &format!("block  B={b} IMAX={imax} wg={wg}"));
        }
    }
    for &wg in &[96usize, 256, 512] {
        for &no_tail in &[false, true] {
            let src = render_tiled_source_cfg(32, wg, 1, no_tail);
            let program = rt.build_program(&src).unwrap();
            let k = Kernel::builder()
                .program(&program)
                .name("jacobi_cyclic_global_batched")
                .queue(rt.queue().clone())
                .global_work_size(wg)
                .local_work_size(wg)
                .arg(&a)
                .arg(&v)
                .arg(n as i32)
                .arg(1i32)
                .arg(0i32)
                .arg(&ones)
                .arg(&diag)
                .arg(0i32)
                .arg(0i32)
                .arg(0.0f32)
                .arg(&occ_w)
                .arg(&mu)
                .arg(&wids)
                .build()
                .unwrap();
            print(&k, &format!("direct wg={wg} notail={no_tail}"));
        }
    }
    // Packed-lA resident kernel: static __local + dynamic arg_local must sum
    // under ~24 KB for two workgroups to co-reside on a 48 KB SM.
    let n86 = 86usize;
    let rotlog = rt.zero_buffer::<f32>(n86 * n86).unwrap();
    for &wg in &[256usize, 512] {
        for &no_tail in &[false, true] {
            let src = render_tiled_source_cfg(32, wg, 0, no_tail);
            let program = rt.build_program(&src).unwrap();
            let k = Kernel::builder()
                .program(&program)
                .name("jacobi_resident_batched")
                .queue(rt.queue().clone())
                .global_work_size(wg)
                .local_work_size(wg)
                .arg(&a)
                .arg(&v)
                .arg(n86 as i32)
                .arg(1i32)
                .arg(0i32)
                .arg(&ones)
                .arg(&diag)
                .arg(0i32)
                .arg(0i32)
                .arg(0.0f32)
                .arg(&occ_w)
                .arg(&mu)
                .arg(&rotlog)
                .arg_local::<f32>(n86 * (n86 + 1) / 2)
                .arg_local::<f32>(1)
                .arg(&wids)
                .build()
                .unwrap();
            let dyn_b = n86 * (n86 + 1) / 2 * 4 + 4;
            print(&k, &format!("res-defV wg={wg} notail={no_tail} (+{dyn_b}B arg_local)"));
        }
    }
}

/// T08 #5: direct-kernel `jacobi_cyclic_global_batched` sweep at N=86 —
/// WG × JACOBI_PREC × tail-compiled-out. The kernel now reduces with
/// fold-then-halve (arbitrary lsz), so non-PoT WGs (96/160/192) are legal;
/// every config is accuracy-gated (stop=0, residual/orth/parity vs CPU).
/// `no_tail` compiles the Fermi scratch out — valid only when the caller
/// uses standalone fermi_occ; here fermi_tail=0 is passed regardless.
///   cargo test --release --test gpu_tiled_jacobi direct_jacobi_wg_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn direct_jacobi_wg_sweep() {
    use ocl::Kernel;
    use rust_dftb::qmqm::gpu_eigen::render_tiled_source_cfg;
    use std::time::Instant;
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let wg_max = rt.caps().max_work_group_size;

    let mk = |n: usize, batch: usize, mode: &str, seed: u64| -> Vec<f32> {
        let mut a = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let mut m = random_symmetric(n, seed + b as u64);
            if mode == "one" {
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            m[i * n + j] *= 3e-3;
                        }
                    }
                }
            }
            a.extend_from_slice(&m);
        }
        a
    };

    for &n in &[86usize, 246] {
        for mode in ["one", "cold"] {
            let batch = 400usize;
            let a = mk(n, batch, mode, 42);
            let a_buf = rt.buffer_from_slice(&a).unwrap();
            let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
            let ones = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
            let wids = rt
                .buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())
                .unwrap(); // T06 identity
            let diag = rt.zero_buffer::<f32>(4 * batch).unwrap();
            let occ_w = rt.zero_buffer::<f32>(batch * n).unwrap();
            let mu = rt.zero_buffer::<f32>(batch).unwrap();
            for &wg in &[64usize, 96, 128, 160, 192, 256, 384, 512] {
                if wg > wg_max {
                    continue;
                }
                for &prec in &[0u32, 1] {
                    for &no_tail in &[false, true] {
                        let src = render_tiled_source_cfg(32, wg, prec, no_tail);
                        let program = match rt.build_program(&src) {
                            Ok(p) => p,
                            Err(e) => {
                                eprintln!("[wsweep] n={n} {mode} WG={wg} prec={prec} notail={no_tail}: BUILD FAIL {e}");
                                continue;
                            }
                        };
                        let k = match Kernel::builder()
                            .program(&program)
                            .name("jacobi_cyclic_global_batched")
                            .queue(rt.queue().clone())
                            .global_work_size(batch * wg)
                            .local_work_size(wg)
                            .arg(&a_buf)
                            .arg(&v_buf)
                            .arg(n as i32)
                            .arg(batch as i32)
                            .arg(0i32)
                            .arg(&ones)
                            .arg(&diag)
                            .arg(0i32)
                            .arg(0i32)
                            .arg(0.0f32)
                            .arg(&occ_w)
                            .arg(&mu)
                            .arg(&wids)
                            .build()
                        {
                            Ok(k) => k,
                            Err(e) => {
                                eprintln!("[wsweep] n={n} {mode} WG={wg} prec={prec} notail={no_tail}: KERNEL FAIL {e}");
                                continue;
                            }
                        };
                        let run = |rt: &mut GpuRuntime| -> std::time::Duration {
                            rt.write_buffer(&a_buf, &a).unwrap();
                            let t0 = Instant::now();
                            unsafe {
                                k.enq().unwrap();
                            }
                            rt.finish().unwrap();
                            t0.elapsed()
                        };
                        run(&mut rt);
                        let t: f64 = (0..3).map(|_| run(&mut rt).as_secs_f64()).sum::<f64>() / 3.0;
                        let mut d4 = vec![0.0f32; 4 * batch];
                        rt.read_buffer(&diag, &mut d4).unwrap();
                        let (sw, bad, off) =
                            (0..batch).fold((0.0f32, 0usize, 0.0f32), |(s, c, m), b| {
                                (
                                    s + d4[4 * b + 3],
                                    c + (d4[4 * b + 2] != 0.0) as usize,
                                    m.max(d4[4 * b + 1]),
                                )
                            });
                        // equal-accuracy spot check on 2 systems
                        let mut tmp_a = vec![0.0f32; batch * n * n];
                        let mut tmp_v = vec![0.0f32; batch * n * n];
                        rt.read_buffer(&a_buf, &mut tmp_a).unwrap();
                        rt.read_buffer(&v_buf, &mut tmp_v).unwrap();
                        let (mut wres, mut worth, mut wpar) = (0.0f64, 0.0f64, 0.0f64);
                        for sb in 0..2 {
                            let ah = &tmp_a[sb * n * n..(sb + 1) * n * n];
                            let vh = &tmp_v[sb * n * n..(sb + 1) * n * n];
                            let a0 = &a[sb * n * n..(sb + 1) * n * n];
                            let mut eigs = vec![0.0f32; n];
                            for i in 0..n {
                                eigs[i] = ah[i * n + i];
                            }
                            wres = wres.max(residual(a0, vh, &eigs, n));
                            worth = worth.max(orthogonality(vh, n));
                            let (ce, _) = cpu_eig(a0, n);
                            let mut gs = eigs.clone();
                            gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                            let mut cs = ce.clone();
                            cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                            for i in 0..n {
                                wpar = wpar.max((gs[i] as f64 - cs[i] as f64).abs());
                            }
                        }
                        eprintln!("[wsweep] n={n} {mode} WG={wg} prec={prec} notail={no_tail}: {:.2}ms sw={:.1} bad={bad} off={:.2e} res={:.2e} orth={:.2e} par={:.2e}",
                            t * 1e3, sw / batch as f32, off, wres, worth, wpar);
                    }
                }
            }
        }
    }
}

/// T08b: `jacobi_resident_batched` parity — A-local + deferred-V (RESIDENT_V=0)
/// and full-local A+V (RESIDENT_V=1) must reproduce the direct kernel's
/// eigenquality: same diag contract (stop=0), residual/orth/eig-parity vs CPU
/// within the manifest tolerances. N=86 (GC class) and N=128 (largest n where
/// 2×n(n+1)×4 ≈ 66 KB still fits a 100 KB-local device for RESIDENT_V).
#[test]
fn test_resident_jacobi_parity() {
    let Some(mut rt) = try_runtime() else {
        return;
    };
    for &n in &[86usize, 128] {
        let batch = 4usize;
        let mut a = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            a.extend_from_slice(&random_symmetric(n, 900 + b as u64));
        }
        for &resident_v in &[false, true] {
            let a_buf = rt.buffer_from_slice(&a).unwrap();
            let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
            let d = match resident_jacobi_batched(&mut rt, &a_buf, &v_buf, n, batch, 1, resident_v)
            {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[res-par] n={n} rv={resident_v}: SKIP {e}");
                    continue;
                }
            };
            let mut tmp_a = vec![0.0f32; batch * n * n];
            let mut tmp_v = vec![0.0f32; batch * n * n];
            rt.read_buffer(&a_buf, &mut tmp_a).unwrap();
            rt.read_buffer(&v_buf, &mut tmp_v).unwrap();
            for sb in 0..batch {
                assert_eq!(
                    d[4 * sb + 2],
                    0.0,
                    "n={n} rv={resident_v} sys{sb}: stop={} off={:.3e} sw={}",
                    d[4 * sb + 2],
                    d[4 * sb],
                    d[4 * sb + 3]
                );
                let ah = &tmp_a[sb * n * n..(sb + 1) * n * n];
                let vh = &tmp_v[sb * n * n..(sb + 1) * n * n];
                let a0 = &a[sb * n * n..(sb + 1) * n * n];
                let mut eigs = vec![0.0f32; n];
                for i in 0..n {
                    eigs[i] = ah[i * n + i];
                }
                let res = residual(a0, vh, &eigs, n);
                let orth = orthogonality(vh, n);
                let (ce, _) = cpu_eig(a0, n);
                let mut gs = eigs.clone();
                gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let mut cs = ce.clone();
                cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let par = (0..n).fold(0.0f64, |m, i| m.max((gs[i] as f64 - cs[i] as f64).abs()));
                eprintln!("[res-par] n={n} rv={resident_v} sys{sb}: res={res:.3e} orth={orth:.3e} par={par:.3e} sw={}", d[4*sb+3]);
                assert!(res < 1e-5, "residual {res:.3e}");
                assert!(orth < 1e-5, "orthogonality {orth:.3e}");
                assert!(par < 1e-4, "eigenvalue parity {par:.3e}");
            }
        }
    }
}

/// T08b sweep: direct(global A/V) vs resident A+V-local vs resident A-local+
/// deferred-V, × WG{256,512} × tail{in,out}, batch=400, equal inputs,
/// accuracy-gated. The hypothesis under test: the direct kernel is bound by
/// A/V global streaming (~10 MB/sweep/system ≈ DRAM roofline); residency
/// should win if the roofline argument is right.
///   cargo test --release --test gpu_tiled_jacobi resident_jacobi_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn resident_jacobi_sweep() {
    use ocl::Kernel;
    use rust_dftb::qmqm::gpu_eigen::{render_resident_source, render_tiled_source_cfg};
    use std::time::Instant;
    let Some(mut rt) = try_runtime() else {
        return;
    };
    let wg_max = rt.caps().max_work_group_size;
    let local_cap = rt.caps().local_mem_size;
    eprintln!(
        "[rsweep] device: max_wg={wg_max} local_mem={} KB",
        local_cap / 1024
    );

    let mk = |n: usize, batch: usize, mode: &str, seed: u64| -> Vec<f32> {
        let mut a = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let mut m = random_symmetric(n, seed + b as u64);
            if mode == "one" {
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            m[i * n + j] *= 3e-3;
                        }
                    }
                }
            }
            a.extend_from_slice(&m);
        }
        a
    };

    for &n in &[86usize, 128] {
        for mode in ["one", "cold"] {
            let batch = 400usize;
            let a = mk(n, batch, mode, 42);
            let a_buf = rt.buffer_from_slice(&a).unwrap();
            let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
            let ones = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
            let wids = rt
                .buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())
                .unwrap(); // T06 identity
            let diag = rt.zero_buffer::<f32>(4 * batch).unwrap();
            let occ_w = rt.zero_buffer::<f32>(batch * n).unwrap();
            let mu = rt.zero_buffer::<f32>(batch).unwrap();
            let jn = if n & 1 == 1 { n + 1 } else { n };
            // jlog2_t entry: prec=1 → double2 (4 f32); prec=0 → float2 (2 f32).
            // This sweep renders resident kernels at prec=1 → 4 f32s/entry.
            let rotlog = rt
                .zero_buffer::<f32>(batch * (jn - 1) * (jn / 2) * 4)
                .unwrap();
            // (tag, kind): direct | resident deferred-V | resident full-local
            for &(tag, rv) in &[("direct", -1i32), ("res-defV", 0), ("res-AV", 1)] {
                for &wg in &[256usize, 512] {
                    if wg > wg_max {
                        continue;
                    }
                    for &no_tail in &[false, true] {
                        let tag = format!("{tag} WG={wg} notail={no_tail}");
                        // packed lA (triangle) + square lV when resident_v
                        let la = n * (n + 1) / 2 * 4 + if rv == 1 { n * (n + 1) * 4 } else { 0 };
                        // +16 KB: kernel's own static __local scratch
                        // (reduce/dred/le/rot arrays) — same headroom as
                        // RESIDENT_SCRATCH_HEADROOM in gpu_eigen.rs.
                        if rv >= 0 && la as u64 + 16 * 1024 > local_cap {
                            eprintln!("[rsweep] n={n} {mode} {tag}: SKIP local {la}B over cap");
                            continue;
                        }
                        let (src, kname) = if rv < 0 {
                            (
                                render_tiled_source_cfg(32, wg, 1, no_tail),
                                "jacobi_cyclic_global_batched",
                            )
                        } else {
                            (
                                render_resident_source(wg, 1, rv == 1, no_tail),
                                "jacobi_resident_batched",
                            )
                        };
                        let program = match rt.build_program(&src) {
                            Ok(p) => p,
                            Err(e) => {
                                eprintln!("[rsweep] n={n} {mode} {tag}: BUILD FAIL {e}");
                                continue;
                            }
                        };
                        let mut kb = Kernel::builder();
                        kb.program(&program)
                            .name(kname)
                            .queue(rt.queue().clone())
                            .global_work_size(batch * wg)
                            .local_work_size(wg)
                            .arg(&a_buf)
                            .arg(&v_buf)
                            .arg(n as i32)
                            .arg(batch as i32)
                            .arg(0i32)
                            .arg(&ones)
                            .arg(&diag)
                            .arg(0i32)
                            .arg(0i32)
                            .arg(0.0f32)
                            .arg(&occ_w)
                            .arg(&mu);
                        if rv >= 0 {
                            kb.arg(&rotlog)
                                .arg_local::<f32>(n * (n + 1) / 2)
                                .arg_local::<f32>(if rv == 1 { n * (n + 1) } else { 1 });
                        }
                        kb.arg(&wids); // T06: last arg in both kernel signatures
                        let k = match kb.build() {
                            Ok(k) => k,
                            Err(e) => {
                                eprintln!("[rsweep] n={n} {mode} {tag}: KERNEL FAIL {e}");
                                continue;
                            }
                        };
                        let run = |rt: &mut GpuRuntime| -> std::time::Duration {
                            rt.write_buffer(&a_buf, &a).unwrap();
                            let t0 = Instant::now();
                            unsafe {
                                k.enq().unwrap();
                            }
                            rt.finish().unwrap();
                            t0.elapsed()
                        };
                        run(&mut rt);
                        let t: f64 = (0..3).map(|_| run(&mut rt).as_secs_f64()).sum::<f64>() / 3.0;
                        let mut d4 = vec![0.0f32; 4 * batch];
                        rt.read_buffer(&diag, &mut d4).unwrap();
                        let (sw, bad, off) =
                            (0..batch).fold((0.0f32, 0usize, 0.0f32), |(s, c, m), b| {
                                (
                                    s + d4[4 * b + 3],
                                    c + (d4[4 * b + 2] != 0.0) as usize,
                                    m.max(d4[4 * b + 1]),
                                )
                            });
                        let mut tmp_a = vec![0.0f32; batch * n * n];
                        let mut tmp_v = vec![0.0f32; batch * n * n];
                        rt.read_buffer(&a_buf, &mut tmp_a).unwrap();
                        rt.read_buffer(&v_buf, &mut tmp_v).unwrap();
                        let (mut wres, mut worth, mut wpar) = (0.0f64, 0.0f64, 0.0f64);
                        for sb in 0..2 {
                            let ah = &tmp_a[sb * n * n..(sb + 1) * n * n];
                            let vh = &tmp_v[sb * n * n..(sb + 1) * n * n];
                            let a0 = &a[sb * n * n..(sb + 1) * n * n];
                            let mut eigs = vec![0.0f32; n];
                            for i in 0..n {
                                eigs[i] = ah[i * n + i];
                            }
                            wres = wres.max(residual(a0, vh, &eigs, n));
                            worth = worth.max(orthogonality(vh, n));
                            let (ce, _) = cpu_eig(a0, n);
                            let mut gs = eigs.clone();
                            gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                            let mut cs = ce.clone();
                            cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                            for i in 0..n {
                                wpar = wpar.max((gs[i] as f64 - cs[i] as f64).abs());
                            }
                        }
                        eprintln!("[rsweep] n={n} {mode} {tag}: {:.2}ms sw={:.1} bad={bad} off={:.2e} res={:.2e} orth={:.2e} par={:.2e}",
                            t * 1e3, sw / batch as f32, off, wres, worth, wpar);
                    }
                }
            }
        }
    }
}
