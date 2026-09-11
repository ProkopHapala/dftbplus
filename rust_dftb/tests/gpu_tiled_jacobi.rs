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

use rust_dftb::qmqm::gpu_eigen::{jacobi_batched, jacobi_cyclic_local_batched, tiled_jacobi_batched};
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
