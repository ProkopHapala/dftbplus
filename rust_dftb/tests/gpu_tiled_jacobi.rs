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

use rust_dftb::qmqm::gpu_eigen::{direct_jacobi_batched, jacobi_batched, tiled_jacobi_batched};
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
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    for i in 0..n { eigs[i] = sym.eigenvalues[i] as f32; }
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
            for k in 0..n { s += a[i*n+k] as f64 * v[k*n+j] as f64; }
            av[i*n+j] = s;
        }
    }
    for i in 0..n {
        for j in 0..n {
            vl[i*n+j] = v[i*n+j] as f64 * eigs[j] as f64;
        }
    }
    let mut num = 0.0f64;
    for i in 0..n*n { num += (av[i] - vl[i]).powi(2); }
    let mut den = 0.0f64;
    for i in 0..n*n { den += a[i] as f64 * a[i] as f64; }
    (num.sqrt()) / den.sqrt().max(1e-30)
}

/// Compute ||V^T·V - I||_F / N (orthogonality).
fn orthogonality(v: &[f32], n: usize) -> f64 {
    let mut vtv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n { s += v[k*n+i] as f64 * v[k*n+j] as f64; }
            vtv[i*n+j] = s;
        }
    }
    let mut num = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let d = vtv[i*n+j] - if i == j { 1.0 } else { 0.0 };
            num += d * d;
        }
    }
    num.sqrt() / n as f64
}

#[test]
fn test_tiled_jacobi_residual_orthogonality() {
    let Some(mut rt) = try_runtime() else { return; };
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

        tiled_jacobi_batched(&mut rt, &a_buf, &v_buf, n, batch)
            .expect("tiled Jacobi must succeed");

        let mut gpu_a = vec![0.0f32; batch * n * n];
        let mut gpu_v = vec![0.0f32; batch * n * n];
        rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
        rt.read_buffer(&v_buf, &mut gpu_v).unwrap();

        for b in 0..batch {
            let a_orig = &all_a_orig[b*n*n..(b+1)*n*n];  // ORIGINAL A for residual
            let v_slice = &gpu_v[b*n*n..(b+1)*n*n];
            let mut eigs = vec![0.0f32; n];
            for i in 0..n { eigs[i] = gpu_a[b*n*n + i*n+i]; }
            let res = residual(a_orig, v_slice, &eigs, n);
            let orth = orthogonality(v_slice, n);
            eprintln!("tiled Jacobi N={n} batch {b}: residual={res:.2e}, orthogonality={orth:.2e}");
            assert!(res < 1e-5, "tiled Jacobi N={n} residual {res:.2e} too large (target 1e-5)");
            assert!(orth < 1e-5, "tiled Jacobi N={n} orthogonality {orth:.2e} too large (target 1e-5)");
        }
    }
}

#[test]
fn test_tiled_jacobi_eigenvalue_parity() {
    let Some(mut rt) = try_runtime() else { return; };
    let batch = 1usize;
    for &n in &[65, 87, 96, 97, 128] {
        let a = random_symmetric(n, 42);
        let (cpu_eigs, _cpu_vecs) = cpu_eig(&a, n);

        let a_buf = rt.buffer_from_slice(&a).unwrap();
        let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();

        tiled_jacobi_batched(&mut rt, &a_buf, &v_buf, n, batch)
            .expect("tiled Jacobi must succeed");

        let mut gpu_a = vec![0.0f32; batch * n * n];
        rt.read_buffer(&a_buf, &mut gpu_a).unwrap();
        let mut gpu_eigs = vec![0.0f32; n];
        for i in 0..n { gpu_eigs[i] = gpu_a[i*n+i]; }

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
        assert!(max_diff < 1e-4, "tiled Jacobi N={n} eigenvalue parity {max_diff:.2e} too large (target 1e-4)");
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

    let Some(mut rt) = try_runtime() else { return; };
    let queue = Queue::new(rt.context(), *rt.device(), Some(flags::CommandQueueProperties::new().profiling()))
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
            let mut ah = vec![0.0f32; input.len()];
            let mut vh = vec![0.0f32; input.len()];
            for prec in 0..3u32 {
                let source = render_tiled_source(32, 256, prec);
                let program = rt.build_program(&source).expect("build Jacobi prec variant");
                let kernel = Kernel::builder()
                    .program(&program).name("tiled_jacobi_batched").queue(queue.clone())
                    .global_work_size(batch * 256).local_work_size(256)
                    .arg(&a_buf).arg(&v_buf).arg(n as i32).arg(batch as i32)
                    .build().expect("build Jacobi prec kernel");
                let mut times = Vec::new();
                for r in 0..=repeats {
                    src.cmd().queue(&queue).copy(&a_buf, None, None).enq().expect("reset Jacobi input");
                    queue.finish().expect("finish input reset");
                    let mut ev = Event::empty();
                    unsafe { kernel.cmd().enew(&mut ev).enq().expect("enqueue Jacobi prec"); }
                    ev.wait_for().expect("wait Jacobi event");
                    let t0 = ev.profiling_info(ProfilingInfo::Start).unwrap().time().unwrap();
                    let t1 = ev.profiling_info(ProfilingInfo::End).unwrap().time().unwrap();
                    assert!(t1 > t0, "invalid Jacobi event timestamps N={n} prec={prec}");
                    if r > 0 { times.push((t1 - t0) as f64 * 1e-3); }
                }
                times.sort_by(f64::total_cmp);
                a_buf.cmd().queue(&queue).read(&mut ah).enq().expect("read A");
                v_buf.cmd().queue(&queue).read(&mut vh).enq().expect("read V");
                queue.finish().expect("finish reads");
                assert!(ah.iter().chain(vh.iter()).all(|x| x.is_finite()), "non-finite Jacobi output N={n} batch={batch} prec={prec}");
                let (mut wres, mut worth, mut wpar) = (0.0f64, 0.0f64, 0.0f64);
                for b in 0..batch {
                    let a0 = &orig[b * n * n..(b + 1) * n * n];
                    let v = &vh[b * n * n..(b + 1) * n * n];
                    let mut eigs = vec![0.0f32; n];
                    for i in 0..n { eigs[i] = ah[b * n * n + i * n + i]; }
                    wres = wres.max(residual(a0, v, &eigs, n));
                    worth = worth.max(orthogonality(v, n));
                    let (ce, _) = cpu_eig(a0, n);
                    let mut gs = eigs.clone();
                    gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    let mut cs = ce.clone();
                    cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    for i in 0..n { wpar = wpar.max((gs[i] as f64 - cs[i] as f64).abs()); }
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

    let Some(mut rt) = try_runtime() else { return; };
    let queue = Queue::new(rt.context(), *rt.device(), Some(flags::CommandQueueProperties::new().profiling()))
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
            let diag_buf = rt.zero_buffer::<f32>(4 * batch).unwrap();
            let mut ah = vec![0.0f32; input.len()];
            let mut vh = vec![0.0f32; input.len()];
            let mut dh = vec![0.0f32; 4 * batch];
            for prec in [0u32, 1] {
                for &wg in &[128usize, 256, 512] {
                    let source = render_tiled_source(32, wg, prec);
                    let program = match rt.build_program(&source) {
                        Ok(p) => p,
                        Err(e) => { eprintln!("direct prec={prec} wg={wg}: build failed {e}"); continue; }
                    };
                    let kernel = match Kernel::builder()
                        .program(&program).name("jacobi_cyclic_global_batched").queue(queue.clone())
                        .global_work_size(batch * wg).local_work_size(wg)
                        .arg(&a_buf).arg(&v_buf).arg(n as i32).arg(batch as i32).arg(0i32)
                        .arg(&ones).arg(&diag_buf)
                        .build()
                    {
                        Ok(k) => k,
                        Err(e) => { eprintln!("direct prec={prec} wg={wg}: kernel build failed {e} (WG limit?)"); continue; }
                    };
                    let mut times = Vec::new();
                    for r in 0..=repeats {
                        src.cmd().queue(&queue).copy(&a_buf, None, None).enq().expect("reset Jacobi input");
                        queue.finish().expect("finish input reset");
                        let mut ev = Event::empty();
                        unsafe { kernel.cmd().enew(&mut ev).enq().expect("enqueue direct Jacobi"); }
                        ev.wait_for().expect("wait direct Jacobi event");
                        let t0 = ev.profiling_info(ProfilingInfo::Start).unwrap().time().unwrap();
                        let t1 = ev.profiling_info(ProfilingInfo::End).unwrap().time().unwrap();
                        assert!(t1 > t0, "invalid event timestamps N={n} prec={prec} wg={wg}");
                        if r > 0 { times.push((t1 - t0) as f64 * 1e-3); }
                    }
                    times.sort_by(f64::total_cmp);
                    a_buf.cmd().queue(&queue).read(&mut ah).enq().expect("read A");
                    v_buf.cmd().queue(&queue).read(&mut vh).enq().expect("read V");
                    diag_buf.cmd().queue(&queue).read(&mut dh).enq().expect("read diag");
                    queue.finish().expect("finish reads");
                    assert!(ah.iter().chain(vh.iter()).all(|x| x.is_finite()),
                        "non-finite output N={n} batch={batch} prec={prec} wg={wg}");
                    let (mut wres, mut worth, mut wpar, mut nstop) = (0.0f64, 0.0f64, 0.0f64, 0);
                    for b in 0..batch {
                        let a0 = &orig[b * n * n..(b + 1) * n * n];
                        let v = &vh[b * n * n..(b + 1) * n * n];
                        let mut eigs = vec![0.0f32; n];
                        for i in 0..n { eigs[i] = ah[b * n * n + i * n + i]; }
                        wres = wres.max(residual(a0, v, &eigs, n));
                        worth = worth.max(orthogonality(v, n));
                        let (ce, _) = cpu_eig(a0, n);
                        let mut gs = eigs.clone(); gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                        let mut cs = ce.clone();   cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                        for i in 0..n { wpar = wpar.max((gs[i] as f64 - cs[i] as f64).abs()); }
                        if dh[4 * b + 2] as i32 != 0 { nstop += 1; }
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
            for k in 0..n { s += q[i * n + k] as f64 * eigs[k] * q[j * n + k] as f64; }
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
    let Some(mut rt) = try_runtime() else { return; };

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
        for i in 0..n { eigs[i] = gpu_a[i * n + i]; }
        let res = residual(&a_orig, &gpu_v, &eigs, n);
        let orth = orthogonality(&gpu_v, n);
        let (ce, _) = cpu_eig(&a_orig, n);
        let mut gs = eigs.clone(); gs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut cs = ce.clone();   cs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let par: f64 = (0..n).map(|i| (gs[i] as f64 - cs[i] as f64).abs()).fold(0.0, f64::max);
        // Weyl bound: sorted-eig deviation ≤ ‖A − VΛVᵀ‖_F = res·‖A‖_F.
        // A fixed absolute tol is wrong here — the parity floor scales with
        // ‖A‖_F (~N·σ for random); the check self-tightens if res improves.
        let af: f64 = a_orig.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let weyl = 2.0 * res * af + 1e-4;
        eprintln!("[R5] direct N={n}: stop={} sweeps={} res={res:.3e} orth={orth:.3e} eig_par={par:.3e} (weyl={weyl:.3e})", diag[2], diag[3]);
        assert_eq!(diag[2] as i32, 0, "direct Jacobi N={n} stop={} (1=stall 2=maxsweeps 3=cap 4=nonfinite)", diag[2]);
        assert!(res < 1e-4, "direct Jacobi N={n} residual {res:.3e}");
        assert!(orth < 1e-4, "direct Jacobi N={n} orth {orth:.3e}");
        assert!(par < weyl, "direct Jacobi N={n} eig parity {par:.3e} exceeds Weyl bound {weyl:.3e} — not explainable by the measured residual");
    }

    // --- adversarial spectra at N=87, one batch ---
    let n = 87usize;
    // b0 clustered: 5 spread + rest in a 1e-6-wide cluster at 1.0
    let mut e = vec![1.0f64; n];
    for k in 5..n { e[k] = 1.0 + 1e-6 * (k % 7) as f64; }
    e[0] = -1.5; e[1] = -0.3; e[2] = 0.1; e[3] = 0.5; e[4] = 2.0;
    let clustered = from_spectrum(n, &e, 7);
    // b1 repeated: two identical-eigenvalue clusters
    let e: Vec<f64> = (0..n).map(|k| if k < n / 2 { 0.5 } else { 1.5 }).collect();
    let repeated = from_spectrum(n, &e, 11);
    // b2 exact zero 8×8 subblock: last 8 rows/cols are literal zeros while
    // the leading block is dense — the all-zero pivot (a_pp=a_qq=a_pq=0)
    // must be skipped, not turned into a 0/0 rotation.
    let mut zero_blk = random_symmetric(n, 13);
    for i in (n - 8)..n { for j in 0..n { zero_blk[i * n + j] = 0.0; zero_blk[j * n + i] = 0.0; } }
    // b3 all-zero matrix: every pivot is the pathological exact-zero case
    let all_zero = vec![0.0f32; n * n];

    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("clustered", clustered), ("repeated", repeated),
        ("zero_blk", zero_blk), ("all_zero", all_zero),
    ];
    let batch = cases.len();
    let mut a_flat = Vec::new();
    for (_, a) in &cases { a_flat.extend_from_slice(a); }
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
        assert!(v.iter().all(|x| x.is_finite()), "direct Jacobi {name}: non-finite eigenvectors");
        let mut eigs = vec![0.0f32; n];
        for i in 0..n { eigs[i] = gpu_a[b * n * n + i * n + i]; }
        assert!(eigs.iter().all(|x| x.is_finite()), "direct Jacobi {name}: non-finite eigenvalues");
        let res = residual(a0, v, &eigs, n);
        let orth = orthogonality(v, n);
        let stop = diag[4 * b + 2] as i32;
        eprintln!("[R5] direct N=87 {name}: stop={stop} sweeps={} res={res:.3e} orth={orth:.3e}", diag[4 * b + 3]);
        assert_eq!(stop, 0, "direct Jacobi {name}: stop={stop}");
        assert!(res < 1e-4, "direct Jacobi {name} residual {res:.3e}");
        assert!(orth < 1e-4, "direct Jacobi {name} orth {orth:.3e}");
    }
}

#[test]
fn test_jacobi_batched_dispatcher() {
    let Some(mut rt) = try_runtime() else { return; };
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
        for i in 0..n { eigs[i] = gpu_a[i*n+i]; }
        let res = residual(&a_orig, &gpu_v, &eigs, n);
        eprintln!("jacobi_batched N={n}: residual={res:.2e}");
        assert!(res < 1e-5, "jacobi_batched N={n} residual {res:.2e} too large (target 1e-5)");
    }
}
