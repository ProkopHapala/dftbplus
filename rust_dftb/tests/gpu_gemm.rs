//! Standalone batched-GEMM variant benchmark (manifest §17.5).
//!
//! C[s] = A[s]·B[s] over [batch][n²] device buffers — the kernel
//! interior that FOE/DM-purification is built from. Correctness vs a
//! CPU f64 reference; timing across kernel structures and tile sizes.
//!
//! Run: cargo test --release --test gpu_gemm -- --nocapture
//!      cargo test --release --test gpu_gemm gemm_bench -- --ignored --nocapture

use rust_dftb::qmqm::gpu_gemm::{gemm_kernel, sq_iter_kernel, GemmVariant};
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

/// CPU f64 reference for gemm_sq_iter: niter × (D ← scl·D²).
fn cpu_sq_iter(a: &[f32], n: usize, niter: usize, scl: f64) -> Vec<f64> {
    let mut d: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let mut t = vec![0.0f64; n * n];
    for _ in 0..niter {
        t.fill(0.0);
        for i in 0..n {
            for k in 0..n {
                let av = d[i * n + k];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    t[i * n + j] += av * d[k * n + j];
                }
            }
        }
        for v in t.iter_mut() {
            *v *= scl;
        }
        std::mem::swap(&mut d, &mut t);
    }
    d
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
    let rt = |tx, ty, rtx, rty, tk, split_m, split_n, sq, tri, nfix| GemmVariant::RegTile {
        tx, ty, rtx, rty, tk, split_m, split_n, sq, tri, nfix,
    };
    vec![
        // coverage-optimal: 11×11 thr × 8×8 regs = 88×88 tile covers n=86
        // in ONE pass — minimum traffic (A,B,C read/written once = 35MB)
        rt(11, 11, 8, 8, 16, 1, 1, false, false, 0),
        // square variant (purify D² = D·D, D symmetric): B ≡ A, As_t
        // serves both fragments — halves staging + B traffic
        rt(11, 11, 8, 8, 16, 1, 1, true, false, 0),
        rt(11, 11, 8, 8, 32, 1, 1, true, false, 0),
        // sq + deep staging: tk=44 (15.5 KB) and tk=88 (whole A ≈ 30 KB,
        // one barrier total)
        rt(11, 11, 8, 8, 44, 1, 1, true, false, 0),
        rt(11, 11, 8, 8, 88, 1, 1, true, false, 0),
        // sq + more threads / fewer regs: r4x8 on 22×11 (242 thr, 32
        // accums), r8x4 on 11×22, r4x4 on 22×22 (484 thr, 16 accums)
        rt(22, 11, 4, 8, 16, 1, 1, true, false, 0),
        // winner tk sweep
        rt(22, 11, 4, 8, 8, 1, 1, true, false, 0),
        rt(22, 11, 4, 8, 22, 1, 1, true, false, 0),
        rt(22, 11, 4, 8, 32, 1, 1, true, false, 0),
        rt(22, 11, 4, 8, 44, 1, 1, true, false, 0),
        // extreme thread counts at 16/8 accums
        rt(44, 11, 2, 8, 16, 1, 1, true, false, 0),
        rt(44, 22, 2, 4, 16, 1, 1, true, false, 0),
        rt(22, 44, 4, 2, 16, 1, 1, true, false, 0),
        rt(11, 22, 8, 4, 16, 1, 1, true, false, 0),
        rt(22, 22, 4, 4, 16, 1, 1, true, false, 0),
        // ---- accumulator-count sweep around the 32-acc sweet spot ----
        // (register-occupancy lever — chat §4)
        rt(15, 15, 6, 6, 16, 1, 1, true, false, 0), // r6x6, 36 acc, 225 thr
        rt(22, 15, 4, 6, 16, 1, 1, true, false, 0), // r6x4, 24 acc, 330 thr
        rt(29, 11, 3, 8, 16, 1, 1, true, false, 0), // r8x3, 24 acc, 319 thr
        rt(11, 18, 8, 5, 16, 1, 1, true, false, 0), // r5x8, 40 acc, 198 thr (WN=88≤WM=90)
        rt(13, 13, 7, 7, 16, 1, 1, true, false, 0), // r7x7, 49 acc, 169 thr
        rt(22, 10, 4, 9, 16, 1, 1, true, false, 0), // r9x4, 36 acc, 220 thr
        // ---- triangular symmetric square (chat §2): only j≥i tiles ----
        // controls: masked large tiles — per-thread FMA count unchanged,
        // predicts ~no gain
        rt(22, 11, 4, 8, 16, 1, 1, true, true, 0),
        rt(11, 11, 8, 8, 16, 1, 1, true, true, 0),
        // real test: 4×4 tiles → ~253 active tiles ≈ half per-thread FMAs
        rt(22, 22, 4, 4, 16, 1, 1, true, true, 0),
        rt(44, 11, 2, 8, 16, 1, 1, true, true, 0), // 8×2 tiles, 8 acc
        rt(11, 44, 8, 2, 16, 1, 1, true, true, 0), // 2×8 tiles, 16 acc
        // ---- fixed-N specialization (chat §3): n=86 compile-time ----
        rt(22, 11, 4, 8, 16, 1, 1, true, false, 86),
        rt(22, 11, 4, 8, 22, 1, 1, true, false, 86),
        rt(11, 11, 8, 8, 16, 1, 1, true, false, 86),
        rt(22, 22, 4, 4, 16, 1, 1, true, true, 86),
        rt(11, 11, 8, 8, 32, 1, 1, true, false, 0),
        // same tile, 32 accums (rty=4 → WM=44, 2 m-passes, B×2 traffic)
        rt(11, 11, 8, 4, 16, 1, 1, false, false, 0),
        // 88×48 tiles on 66 thr, col-split → 2 WGs/system
        rt(6, 11, 8, 8, 16, 1, 2, false, false, 0),
        // 88×32 tiles on 44 thr, col-split → 3 WGs
        rt(4, 11, 8, 8, 16, 1, 3, false, false, 0),
        // 32×88 tiles on 44 thr, row-split → 3 WGs
        rt(11, 4, 8, 8, 16, 3, 1, false, false, 0),
        // 48×48 tiles on 36 thr, 2×2 split → 4 WGs (80% coverage)
        rt(6, 6, 8, 8, 16, 2, 2, false, false, 0),
        // previous best
        rt(8, 8, 8, 8, 16, 2, 2, false, false, 0),
        // 32 accums (4×8): less register pressure → higher occupancy
        rt(8, 16, 8, 4, 16, 2, 2, false, false, 0),
        rt(16, 8, 4, 8, 16, 2, 2, false, false, 0),
        // 16 accums (4×4) 256 thr — max threads, low regs
        rt(16, 16, 4, 4, 16, 2, 2, false, false, 0),
        // small WG tile 32×32 on 64 thr (4×4 regs), heavy split
        rt(8, 8, 4, 4, 16, 3, 3, false, false, 0),
        rt(8, 8, 4, 4, 16, 3, 2, false, false, 0),
        // 32×64 tile, 4×8 regs on 128 thr
        rt(8, 8, 8, 4, 16, 3, 2, false, false, 0),
        // 1-WG-per-system reference at the best single-WG shape
        rt(8, 8, 8, 8, 16, 1, 1, false, false, 0),
        rt(8, 16, 8, 4, 16, 1, 1, false, false, 0),
        // classic 1-elem floor
        rt(16, 16, 1, 1, 16, 1, 1, false, false, 0),
        // ---- floor + early regtile baselines (the progression story) ----
        // first regtile winner: 8×16 thr × 8×8 regs = 64×128 cover, 128 thr
        rt(8, 16, 8, 8, 16, 1, 1, false, false, 0),
        // 4×4 micro-tile on 16×16 thr (256 thr, 16 accums)
        rt(16, 16, 4, 4, 16, 1, 1, false, false, 0),
        // one-thread-per-element global dot — the absolute floor
        GemmVariant::OneElem { wg: 256 },
        GemmVariant::OneElem { wg: 512 },
        // whole A resident in local (29.6 KB at n=86) + B k-tiles
        GemmVariant::FullA { wg: 64, tk: 8 },
        GemmVariant::FullA { wg: 128, tk: 8 },
    ]
}

/// SqIter variants — iterated D ← scl·D² in ONE launch. Prices the two
/// costs folded into the fused tc2 step (0.141 vs 0.082 ms isolated):
/// residency (`loc` = D in __local all iterations, `glob` = ping-pong)
/// and the per-iter WG reduce (red 0 none / 1 two fold-halve = current
/// tc2_step / 2 merged float2 / 3 merged 2-level). Plus the round-2
/// microkernel experiments: `vec` (explicit floatN accumulators), `ku`
/// (k-loop unroll), `tri` (compact SYRK — 66 upper 8×8 tiles split
/// across 8/rtx threads; tx·ty = 132 for rtx=4, 264 for rtx=2).
fn sqiter_variants(niter: usize) -> Vec<GemmVariant> {
    let si = |tx, ty, rtx, rty, resident, red| GemmVariant::SqIter {
        tx, ty, rtx, rty, tk: 16, niter, resident, red, tri: false, ku: 1, vec: false,
    };
    let sqit = |tx, ty, rtx, rty, resident, red, tri, ku, vec| GemmVariant::SqIter {
        tx, ty, rtx, rty, tk: 16, niter, resident, red, tri, ku, vec,
    };
    vec![
        si(22, 11, 4, 8, true, 0),  // local-resident, no reduce
        si(22, 11, 4, 8, true, 1),  // + 2 fold-halve reduces (current)
        si(22, 11, 4, 8, true, 2),  // + merged float2 reduce
        si(22, 11, 4, 8, true, 3),  // + merged 2-level reduce
        si(22, 11, 4, 8, false, 0), // global ping-pong, no reduce
        si(22, 11, 4, 8, false, 1), // global ping-pong + reduces
        si(11, 11, 8, 8, true, 0),  // resident, 121 thr / 64 acc — best
        si(11, 11, 8, 8, true, 1),  // resident r8x8 + fold-halve reduces
        si(11, 11, 8, 8, true, 2),  // resident r8x8 + merged reduce
        si(11, 22, 8, 4, true, 0),  // resident r4x8, 242 thr / 32 acc
        si(11, 22, 8, 4, true, 2),  // resident r4x8 + merged reduce
        si(22, 22, 4, 4, true, 0),  // resident, 484 thr / 16 acc
        // GEMM_VEC — explicit float8/float4 accumulators, both shapes
        sqit(11, 11, 8, 8, true, 0, false, 1, true),
        sqit(11, 11, 8, 8, true, 2, false, 1, true),
        sqit(22, 11, 4, 8, true, 0, false, 1, true),
        sqit(11, 22, 8, 4, true, 0, false, 1, true),
        // GEMM_KU — k-loop unroll ×2/×4 on the r8x8 winner (scalar + vec)
        sqit(11, 11, 8, 8, true, 0, false, 2, false),
        sqit(11, 11, 8, 8, true, 0, false, 4, false),
        sqit(11, 11, 8, 8, true, 0, false, 2, true),
        sqit(11, 11, 8, 8, true, 0, false, 4, true),
        // compact SYRK — 66 upper 8×8 tiles × 2 thr (132) or ×4 (264)
        sqit(12, 11, 4, 8, true, 0, true, 1, false),
        sqit(12, 11, 4, 8, true, 2, true, 1, false),
        sqit(12, 11, 4, 8, true, 0, true, 4, false),
        sqit(24, 11, 2, 8, true, 0, true, 1, false),
        sqit(24, 11, 2, 8, true, 2, true, 1, false),
        sqit(24, 11, 2, 8, true, 0, true, 4, false),
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
        // sq/tri variants: B ≡ A (symmetric A² path) — pass A as the B
        // arg and check against A², not A·B.
        let sq = matches!(
            v,
            GemmVariant::RegTile { sq: true, .. } | GemmVariant::RegTile { tri: true, .. }
        );
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

    // ---- gemm_sq_iter: D ← scl·D², niter rounds in one launch ----
    let niter = 4usize;
    let scl = 1.0f32 / n as f32;
    let errs_buf = rt.zero_buffer::<f32>(batch).unwrap();
    let trs_buf = rt.zero_buffer::<f32>(batch).unwrap();
    for v in sqiter_variants(niter) {
        let lbl = v.label();
        let red = matches!(v, GemmVariant::SqIter { red: r, .. } if r > 0);
        let k = match sq_iter_kernel(
            &mut rt, &v, scl, n, batch, &a_buf, &b_buf, &c_buf, &errs_buf, &trs_buf,
        ) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[gemm] {lbl:24} build FAILED: {e}");
                n_bad += 1;
                continue;
            }
        };
        unsafe { k.enq().unwrap() };
        rt.read_buffer(&c_buf, &mut c).unwrap();
        let mut worst = 0.0f64;
        let mut worst_tr = 0.0f64;
        let mut trs_h = vec![0.0f32; batch];
        if red {
            rt.read_buffer(&trs_buf, &mut trs_h).unwrap();
        }
        for s in 0..batch {
            let dref = cpu_sq_iter(&a[s * n * n..(s + 1) * n * n], n, niter, scl as f64);
            let mut tr_ref = 0.0f64;
            for i in 0..n {
                tr_ref += dref[i * n + i];
            }
            if red {
                let d = (trs_h[s] as f64 - tr_ref).abs();
                if d > worst_tr {
                    worst_tr = d;
                }
            }
            for i in 0..n * n {
                let d = (c[s * n * n + i] as f64 - dref[i]).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
        eprintln!("[gemm] {lbl:24} max|D−Dref|={worst:.3e} max|Tr−Trref|={worst_tr:.3e} (niter={niter})");
        if !(worst < 1e-3) || (red && !(worst_tr < 1e-2)) {
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

    // ---- iterated symmetric square: niter rounds per launch ----
    // ms/iter is the honest comparison vs the fused tc2 step (0.141 ms);
    // TFLOPS-eq counts 2n³·niter·batch useful FLOPs.
    let niter: usize = std::env::var("GEMM_ITER").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let scl = 1.0f32 / n as f32;
    let errs_buf = rt.zero_buffer::<f32>(batch).unwrap();
    let trs_buf = rt.zero_buffer::<f32>(batch).unwrap();
    for v in sqiter_variants(niter) {
        let lbl = v.label();
        let k = match sq_iter_kernel(
            &mut rt, &v, scl, n, batch, &a_buf, &b_buf, &c_buf, &errs_buf, &trs_buf,
        ) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[gemm-bench] {lbl:26} FAILED: {e}");
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
        let tf = flops * niter as f64 / (ms * 1e-3) / 1e12;
        eprintln!(
            "[gemm-bench] {lbl:26} {ms:7.3} ms/call  {ms_it:7.4} ms/iter  {tf:6.2} TFLOPS-eq",
            ms_it = ms / niter as f64
        );
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
