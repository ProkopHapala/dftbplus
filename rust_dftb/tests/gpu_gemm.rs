//! Standalone batched-GEMM variant benchmark (manifest §17.5).
//!
//! C[s] = A[s]·B[s] over [batch][n²] device buffers — the kernel
//! interior that FOE/DM-purification is built from. Correctness vs a
//! CPU f64 reference; timing across kernel structures and tile sizes.
//!
//! Run: cargo test --release --test gpu_gemm -- --nocapture
//!      cargo test --release --test gpu_gemm gemm_bench -- --ignored --nocapture

use rust_dftb::qmqm::gpu_gemm::{gemm_kernel, GemmVariant};
use rust_dftb::qmqm::gpu_matrix::{GpuMatrixContext, MatrixKernelConfig, Transpose};
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;

fn rand_mat(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut a = vec![0.0f32; n * n];
    for v in a.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32;
    }
    a
}

/// Symmetric matrix — required for the sq (B≡A) variants, which compute
/// A·Aᵀ = A² only when A is symmetric (the purify D² case).
fn rand_sym(n: usize, seed: u64) -> Vec<f32> {
    let mut a = rand_mat(n, seed);
    for i in 0..n {
        for j in i + 1..n {
            a[j * n + i] = a[i * n + j];
        }
    }
    a
}

fn cpu_mm(a: &[f32], b: &[f32], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let av = a[i * n + k] as f64;
            if av == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += av * b[k * n + j] as f64;
            }
        }
    }
    c
}

fn variants() -> Vec<GemmVariant> {
    vec![
        // coverage-optimal: 11×11 thr × 8×8 regs = 88×88 tile covers n=86
        // in ONE pass — minimum traffic (A,B,C read/written once = 35MB)
        GemmVariant::RegTile { tx: 11, ty: 11, rtx: 8, rty: 8, tk: 16, split_m: 1, split_n: 1, sq: false },
        // square variant (purify D² = D·D, D symmetric): B ≡ A, As_t
        // serves both fragments — halves staging + B traffic
        GemmVariant::RegTile { tx: 11, ty: 11, rtx: 8, rty: 8, tk: 16, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 11, ty: 11, rtx: 8, rty: 8, tk: 32, split_m: 1, split_n: 1, sq: true },
        // sq + deep staging: tk=44 (15.5 KB) and tk=88 (whole A ≈ 30 KB,
        // one barrier total)
        GemmVariant::RegTile { tx: 11, ty: 11, rtx: 8, rty: 8, tk: 44, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 11, ty: 11, rtx: 8, rty: 8, tk: 88, split_m: 1, split_n: 1, sq: true },
        // sq + more threads / fewer regs: r4x8 on 22×11 (242 thr, 32
        // accums), r8x4 on 11×22, r4x4 on 22×22 (484 thr, 16 accums)
        GemmVariant::RegTile { tx: 22, ty: 11, rtx: 4, rty: 8, tk: 16, split_m: 1, split_n: 1, sq: true },
        // winner tk sweep
        GemmVariant::RegTile { tx: 22, ty: 11, rtx: 4, rty: 8, tk: 8, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 22, ty: 11, rtx: 4, rty: 8, tk: 22, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 22, ty: 11, rtx: 4, rty: 8, tk: 32, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 22, ty: 11, rtx: 4, rty: 8, tk: 44, split_m: 1, split_n: 1, sq: true },
        // extreme thread counts at 16/8 accums
        GemmVariant::RegTile { tx: 44, ty: 11, rtx: 2, rty: 8, tk: 16, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 44, ty: 22, rtx: 2, rty: 4, tk: 16, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 22, ty: 44, rtx: 4, rty: 2, tk: 16, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 11, ty: 22, rtx: 8, rty: 4, tk: 16, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 22, ty: 22, rtx: 4, rty: 4, tk: 16, split_m: 1, split_n: 1, sq: true },
        GemmVariant::RegTile { tx: 11, ty: 11, rtx: 8, rty: 8, tk: 32, split_m: 1, split_n: 1, sq: false },
        // same tile, 32 accums (rty=4 → WM=44, 2 m-passes, B×2 traffic)
        GemmVariant::RegTile { tx: 11, ty: 11, rtx: 8, rty: 4, tk: 16, split_m: 1, split_n: 1, sq: false },
        // 88×48 tiles on 66 thr, col-split → 2 WGs/system
        GemmVariant::RegTile { tx: 6, ty: 11, rtx: 8, rty: 8, tk: 16, split_m: 1, split_n: 2, sq: false },
        // 88×32 tiles on 44 thr, col-split → 3 WGs
        GemmVariant::RegTile { tx: 4, ty: 11, rtx: 8, rty: 8, tk: 16, split_m: 1, split_n: 3, sq: false },
        // 32×88 tiles on 44 thr, row-split → 3 WGs
        GemmVariant::RegTile { tx: 11, ty: 4, rtx: 8, rty: 8, tk: 16, split_m: 3, split_n: 1, sq: false },
        // 48×48 tiles on 36 thr, 2×2 split → 4 WGs (80% coverage)
        GemmVariant::RegTile { tx: 6, ty: 6, rtx: 8, rty: 8, tk: 16, split_m: 2, split_n: 2, sq: false },
        // previous best
        GemmVariant::RegTile { tx: 8, ty: 8, rtx: 8, rty: 8, tk: 16, split_m: 2, split_n: 2, sq: false },
        // 32 accums (4×8): less register pressure → higher occupancy
        GemmVariant::RegTile { tx: 8, ty: 16, rtx: 8, rty: 4, tk: 16, split_m: 2, split_n: 2, sq: false },
        GemmVariant::RegTile { tx: 16, ty: 8, rtx: 4, rty: 8, tk: 16, split_m: 2, split_n: 2, sq: false },
        // 16 accums (4×4) 256 thr — max threads, low regs
        GemmVariant::RegTile { tx: 16, ty: 16, rtx: 4, rty: 4, tk: 16, split_m: 2, split_n: 2, sq: false },
        // small WG tile 32×32 on 64 thr (4×4 regs), heavy split
        GemmVariant::RegTile { tx: 8, ty: 8, rtx: 4, rty: 4, tk: 16, split_m: 3, split_n: 3, sq: false },
        GemmVariant::RegTile { tx: 8, ty: 8, rtx: 4, rty: 4, tk: 16, split_m: 3, split_n: 2, sq: false },
        // 32×64 tile, 4×8 regs on 128 thr
        GemmVariant::RegTile { tx: 8, ty: 8, rtx: 8, rty: 4, tk: 16, split_m: 3, split_n: 2, sq: false },
        // 1-WG-per-system reference at the best single-WG shape
        GemmVariant::RegTile { tx: 8, ty: 8, rtx: 8, rty: 8, tk: 16, split_m: 1, split_n: 1, sq: false },
        GemmVariant::RegTile { tx: 8, ty: 16, rtx: 8, rty: 4, tk: 16, split_m: 1, split_n: 1, sq: false },
        // classic 1-elem floor
        GemmVariant::RegTile { tx: 16, ty: 16, rtx: 1, rty: 1, tk: 16, split_m: 1, split_n: 1, sq: false },
        // ---- floor + early regtile baselines (the progression story) ----
        // first regtile winner: 8×16 thr × 8×8 regs = 64×128 cover, 128 thr
        GemmVariant::RegTile { tx: 8, ty: 16, rtx: 8, rty: 8, tk: 16, split_m: 1, split_n: 1, sq: false },
        // 4×4 micro-tile on 16×16 thr (256 thr, 16 accums)
        GemmVariant::RegTile { tx: 16, ty: 16, rtx: 4, rty: 4, tk: 16, split_m: 1, split_n: 1, sq: false },
        // one-thread-per-element global dot — the absolute floor
        GemmVariant::OneElem { wg: 256 },
        GemmVariant::OneElem { wg: 512 },
        // whole A resident in local (29.6 KB at n=86) + B k-tiles
        GemmVariant::FullA { wg: 64, tk: 8 },
        GemmVariant::FullA { wg: 128, tk: 8 },
    ]
}

#[test]
fn test_gemm_variants_parity() {
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[gemm] no GPU: {e}");
            return;
        }
    };
    let n = 86usize;
    let batch = 8usize;

    let mut a = Vec::with_capacity(batch * n * n);
    let mut b = Vec::with_capacity(batch * n * n);
    for s in 0..batch {
        a.extend_from_slice(&rand_sym(n, 7000 + s as u64)); // symmetric — sq needs it
        b.extend_from_slice(&rand_mat(n, 9000 + s as u64));
    }
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let b_buf = rt.buffer_from_slice(&b).unwrap();
    let c_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    let mut c = vec![0.0f32; batch * n * n];

    let mut n_bad = 0;
    for v in variants() {
        // sq variants: B ≡ A (symmetric A² path) — pass A as the B arg
        // and check against A², not A·B.
        let sq = matches!(v, GemmVariant::RegTile { sq: true, .. });
        let b_arg = if sq { &a_buf } else { &b_buf };
        let k = match gemm_kernel(&mut rt, &v, n, batch, &a_buf, b_arg, &c_buf) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[gemm] {} build/launch FAILED: {e}", v.label());
                n_bad += 1;
                continue;
            }
        };
        unsafe { k.enq().unwrap() };
        rt.read_buffer(&c_buf, &mut c).unwrap();
        let mut worst = 0.0f64;
        for s in 0..batch {
            let bs = if sq { &a } else { &b };
            let cref = cpu_mm(&a[s * n * n..(s + 1) * n * n], &bs[s * n * n..(s + 1) * n * n], n);
            for i in 0..n * n {
                let d = (c[s * n * n + i] as f64 - cref[i]).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
        eprintln!("[gemm] {:24} max|C−Cref|={worst:.3e}", v.label());
        if !(worst < 1e-3) {
            n_bad += 1;
        }
    }
    assert_eq!(n_bad, 0, "{n_bad} GEMM variants failed parity");
}

#[test]
#[ignore]
fn gemm_bench() {
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[gemm] no GPU: {e}");
            return;
        }
    };
    let n: usize = std::env::var("GEMM_N").ok().and_then(|s| s.parse().ok()).unwrap_or(86);
    let batch: usize = std::env::var("GEMM_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let reps: usize = std::env::var("GEMM_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let flops = 2.0 * (n * n * n) as f64 * batch as f64;

    let mut a = Vec::with_capacity(batch * n * n);
    let mut b = Vec::with_capacity(batch * n * n);
    for s in 0..batch {
        a.extend_from_slice(&rand_mat(n, 7000 + s as u64));
        b.extend_from_slice(&rand_mat(n, 9000 + s as u64));
    }
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let b_buf = rt.buffer_from_slice(&b).unwrap();
    let c_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();

    eprintln!("[gemm-bench] n={n} batch={batch} flops/call={:.2} MFLOP", flops / 1e6);
    for v in variants() {
        let k = match gemm_kernel(&mut rt, &v, n, batch, &a_buf, &b_buf, &c_buf) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[gemm-bench] {:24} FAILED: {e}", v.label());
                continue;
            }
        };
        unsafe { k.enq().unwrap() };
        rt.finish().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            unsafe { k.enq().unwrap() };
        }
        rt.finish().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let tf = flops / (ms * 1e-3) / 1e12;
        eprintln!("[gemm-bench] {:24} {ms:7.3} ms  {tf:6.2} TFLOPS  ({:.1}% of ~36T)", v.label(), tf / 36.0 * 100.0);
    }

    // baseline: production batched_gemm (tile-per-WG) across tile configs.
    // GpuMatrixContext owns a separate OpenCL context — buffers must be
    // allocated through it, and its queue is flushed via read_buffer.
    let mut c_flush = vec![0.0f32; batch * n * n];
    for (tm, tn, tk) in [(8usize, 8usize, 16usize), (16, 16, 16), (16, 16, 32), (32, 32, 8)] {
        let cfg = MatrixKernelConfig {
            tile_m: tm,
            tile_n: tn,
            tile_k: tk,
            ..MatrixKernelConfig::nvidia_default()
        };
        let ctx = match GpuMatrixContext::new(cfg) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[gemm-bench] batched_gemm {tm}x{tn}x{tk} ctx FAILED: {e}");
                continue;
            }
        };
        let a2 = ctx.buffer_from_slice(&a).unwrap();
        let b2 = ctx.buffer_from_slice(&b).unwrap();
        let c2 = ctx.zero_buffer(batch * n * n).unwrap();
        if ctx
            .batched_gemm(n, batch, Transpose::No, Transpose::No, 1.0, 0.0, &a2, &b2, &c2)
            .is_err()
        {
            eprintln!("[gemm-bench] batched_gemm {tm}x{tn}x{tk} launch FAILED");
            continue;
        }
        ctx.read_buffer(&c2, &mut c_flush).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ctx.batched_gemm(n, batch, Transpose::No, Transpose::No, 1.0, 0.0, &a2, &b2, &c2)
                .unwrap();
        }
        ctx.read_buffer(&c2, &mut c_flush).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let tf = flops / (ms * 1e-3) / 1e12;
        eprintln!("[gemm-bench] batched{tm}x{tn}/tk{tk}      {ms:7.3} ms  {tf:6.2} TFLOPS  ({:.1}% of ~36T)", tf / 36.0 * 100.0);
    }
}
