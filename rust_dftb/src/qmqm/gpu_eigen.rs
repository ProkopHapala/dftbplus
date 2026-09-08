//! GPU eigensolver module: Brent-Luk parallel cyclic Jacobi + S^{-1/2}.
//!
//! Agent_4 (Wave 2) owned module. Provides:
//! - `jacobi_cyclic_local_batched` — batched symmetric eigensolver for N<=64
//! - `build_inv_sqrt` — S^{-1/2} via Jacobi eigendecomposition
//!
//! # Architecture
//!
//! One workgroup per system. The full N×N matrix and eigenvector matrix
//! reside in `__local` memory (padded to JN×JLD where JN is the next even
//! number >= N and JLD = JN+1 to avoid bank conflicts). The Brent-Luk
//! parallel cyclic schedule pairs JN/2 independent element pairs per round;
//! each pair is handled by PPG work-items that split the JN rows. With the
//! block-update path (JACOBI_BLOCK_UPDATE=1, default), two barriers per
//! round: one after rotation publication, one after the A+V update. With the
//! legacy two-pass path (JACOBI_BLOCK_UPDATE=0), four barriers per round.
//! Up to MAX_SWEEPS sweeps with relative off-diagonal norm convergence check
//! and stagnation detection (breaks if the off-norm does not improve by 10%
//! between sweeps). Pair skip threshold PAIR_SKIP_TOL (default 0) is separate
//! from the global convergence tolerance JACOBI_TOL (default 1e-7).
//!
//! # Specialization
//!
//! The OpenCL source is text-substituted at runtime for the given N:
//! JN, JLD, JPAIR, JROUND, WG, PPG are baked in as `#define` constants.
//! This avoids dynamic indexing overhead in the inner loop and lets the
//! compiler unroll and optimize the fixed-size local memory operations.

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_matrix::{matmul_tiled_batched, matmul_tiled_batched_params};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::{Buffer, Kernel};

const GPU_EIGEN_TEMPLATE: &str = include_str!("gpu_eigen.cl");

/// Maximum Jacobi sweeps before forced exit.
const MAX_SWEEPS: usize = 20;
/// Work-items per Jacobi pair (fixed; WG is derived from this).
const PPG: usize = 8;

/// Compute specialization parameters for a given (unpadded) dimension N.
///
/// Returns `(jn, jld, jpair, jround, wg)`:
/// - `jn` — working dimension (N if even, N+1 if odd; always even)
/// - `jld` — leading dimension = JN + 1 (avoids power-of-2 bank conflicts)
/// - `jpair` — pairs per round = JN / 2
/// - `jround` — rounds per sweep = JN - 1
/// - `wg` — workgroup size (power of 2, >= 32, <= 1024)
fn spec_params(n: usize) -> (usize, usize, usize, usize, usize) {
    if n == 0 {
        return (0, 1, 0, 0, 32);
    }
    let jn = if n % 2 == 0 { n } else { n + 1 };
    let jld = jn + 1;
    let jpair = jn / 2;
    let jround = jn - 1;
    let active = jpair * PPG;
    let wg = active.next_power_of_two().max(32).min(1024);
    (jn, jld, jpair, jround, wg)
}

/// Render the OpenCL template with specialization constants for the given N.
fn render_source(n: usize) -> String {
    let (jn, jld, jpair, jround, wg) = spec_params(n);
    GPU_EIGEN_TEMPLATE
        .replace("#define JN 8", &format!("#define JN {}", jn))
        .replace("#define JLD 9", &format!("#define JLD {}", jld))
        .replace("#define JPAIR 4", &format!("#define JPAIR {}", jpair))
        .replace("#define JROUND 7", &format!("#define JROUND {}", jround))
        .replace("#define WG 32", &format!("#define WG {}", wg))
        .replace("#define PPG 8", &format!("#define PPG {}", PPG))
        .replace("#define MAX_SWEEPS 20", &format!("#define MAX_SWEEPS {}", MAX_SWEEPS))
}

/// Batched Brent-Luk parallel cyclic Jacobi eigensolver.
///
/// Diagonalizes `batch` symmetric N×N matrices stored in `a_buf` (row-major,
/// `[batch][N*N]`). On return, `a_buf` holds eigenvalues on the diagonal
/// (off-diagonal elements are zeroed) and `v_buf` holds the eigenvector
/// matrix (columns are eigenvectors).
///
/// One workgroup per system. Full-local for N≤64. Padded leading dimension
/// JLD = JN+1 to avoid bank conflicts. N/2 independent rotations per round
/// (Brent-Luk parallel cyclic ordering), one barrier per round.
///
/// # Arguments
/// - `rt` — shared OpenCL runtime (mutable because `build_program` uses a cache)
/// - `a_buf` — `[batch][N*N]` symmetric matrices (in/out: eigenvalues on diagonal)
/// - `v_buf` — `[batch][N*N]` buffer (out: eigenvectors; content on input is ignored)
/// - `n` — matrix dimension (N ≤ 64)
/// - `batch` — number of matrices
pub fn jacobi_cyclic_local_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
) -> Result<()> {
    if n == 0 || batch == 0 {
        return Ok(());
    }
    if n > 64 {
        return Err(DftbError::InvalidInput(format!(
            "jacobi_cyclic_local_batched: n={} exceeds maximum supported N=64",
            n
        )));
    }
    let (_, _, _, _, wg) = spec_params(n);
    let source = render_source(n);
    let program = rt.build_program(&source)?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("jacobi_cyclic_local_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(a_buf)
        .arg(v_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;
    unsafe {
        kernel.enq().map_err(map_ocl_err)?;
    }
    // No rt.finish() — in-order queue preserves command order. The caller
    // must synchronize only when the host genuinely needs the result (e.g.
    // via read_buffer, which finishes the queue before returning).
    Ok(())
}

// ------------------------------------------------------------------
// Phase 2: Tiled block Jacobi for N > 64
// ------------------------------------------------------------------

const GPU_TILED_JACOBI_TEMPLATE: &str = include_str!("gpu_tiled_jacobi.cl");

/// Maximum sweeps for the tiled block Jacobi (N>64).
/// Block Jacobi converges slower than full-local element Jacobi because
/// block pairs are processed sequentially, not in parallel.
const TILED_MAX_SWEEPS: usize = 50;

/// Render the tiled Jacobi OpenCL template with block-size specialization.
fn render_tiled_source(b: usize, wg: usize) -> String {
    let pb = 2 * b;
    let pld = pb + 1;
    GPU_TILED_JACOBI_TEMPLATE
        .replace("#define B 32", &format!("#define B {}", b))
        .replace("#define PB 64", &format!("#define PB {}", pb))
        .replace("#define PLD 65", &format!("#define PLD {}", pld))
        .replace("#define WG 256", &format!("#define WG {}", wg))
        .replace("#define STRIP_R 32", &format!("#define STRIP_R {}", b))
        .replace("#define MAX_SWEEPS 50", &format!("#define MAX_SWEEPS {}", TILED_MAX_SWEEPS))
}

/// Tiled block Jacobi eigensolver for N > 64.
///
/// Diagonalizes `batch` symmetric N×N matrices. One workgroup per system.
/// A and V reside in global memory; only the 2B×2B compound pivot and strip
/// workspace live in local memory. Block size B=32 (default), WG=256.
///
/// Uses the existing `jacobi_cyclic_local_batched` for N ≤ 64; this function
/// is the N > 64 path. Use `jacobi_batched` (below) for automatic dispatch.
///
/// # Arguments
/// - `rt` — shared OpenCL runtime
/// - `a_buf` — `[batch][N*N]` symmetric matrices (in/out: eigenvalues on diagonal)
/// - `v_buf` — `[batch][N*N]` buffer (out: eigenvectors)
/// - `n` — matrix dimension (N > 64)
/// - `batch` — number of matrices
pub fn tiled_jacobi_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
) -> Result<()> {
    if n == 0 || batch == 0 {
        return Ok(());
    }
    if n <= 64 {
        return Err(DftbError::InvalidInput(format!(
            "tiled_jacobi_batched: n={n} <= 64, use jacobi_cyclic_local_batched instead"
        )));
    }
    let b = 32usize;  // block size
    let wg = 256usize; // workgroup size
    let source = render_tiled_source(b, wg);
    let program = rt.build_program(&source)?;
    let kernel = Kernel::builder()
        .program(&program)
        .name("tiled_jacobi_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(a_buf)
        .arg(v_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;
    unsafe {
        kernel.enq().map_err(map_ocl_err)?;
    }
    Ok(())
}

/// Dispatch eigensolver: full-local Jacobi for N≤64, tiled block Jacobi for N>64.
/// This is the production entry point — callers should use this instead of
/// `jacobi_cyclic_local_batched` directly.
pub fn jacobi_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
) -> Result<()> {
    if n <= 64 {
        jacobi_cyclic_local_batched(rt, a_buf, v_buf, n, batch)
    } else {
        tiled_jacobi_batched(rt, a_buf, v_buf, n, batch)
    }
}

/// Compute S^{-1/2} via Jacobi eigendecomposition.
///
/// Returns `(X_buf, lambda_min_buf)` where `X = U · diag(rsqrt(λ)) · U^T`
/// and `lambda_min` is the smallest eigenvalue per system for precision
/// monitoring. The input `s_buf` is not modified.
///
/// # Arguments
/// - `rt` — shared OpenCL runtime
/// - `s_buf` — `[batch][N*N]` overlap matrices (not modified)
/// - `n` — matrix dimension (N ≤ 64)
/// - `batch` — number of matrices
///
/// # Returns
/// - `X_buf` — `[batch][N*N]` S^{-1/2} matrices
/// - `lambda_min_buf` — `[batch]` smallest eigenvalue per system
pub fn build_inv_sqrt(
    rt: &mut GpuRuntime,
    s_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
) -> Result<(Buffer<f32>, Buffer<f32>)> {
    if n == 0 || batch == 0 {
        let x = rt.zero_buffer::<f32>(n * n * batch)?;
        let lm = rt.zero_buffer::<f32>(batch)?;
        return Ok((x, lm));
    }
    if n > 64 {
        return Err(DftbError::InvalidInput(format!(
            "build_inv_sqrt: n={} exceeds maximum supported N=64",
            n
        )));
    }

    // Step 1: copy S to a working buffer (Jacobi modifies in place).
    // Device-to-device copy — no host roundtrip (Phase 0b cleanup).
    let a_work = rt.copy_buffer(s_buf, n * n * batch)?;

    // Step 2: allocate V buffer (kernel initializes to identity).
    let v_buf = rt.zero_buffer::<f32>(n * n * batch)?;

    // Step 3: diagonalize S → a_work has eigenvalues on diagonal, v_buf has eigenvectors.
    jacobi_cyclic_local_batched(rt, &a_work, &v_buf, n, batch)?;

    // Step 4: allocate output buffers.
    let x_buf = rt.zero_buffer::<f32>(n * n * batch)?;
    let lambda_min_buf = rt.zero_buffer::<f32>(batch)?;

    // Step 5: launch build_inv_sqrt_from_eig kernel.
    let (_, _, _, _, wg) = spec_params(n);
    let source = render_source(n);
    let program = rt.build_program(&source)?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("build_inv_sqrt_from_eig")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(&a_work)
        .arg(&v_buf)
        .arg(&x_buf)
        .arg(&lambda_min_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;
    unsafe {
        kernel.enq().map_err(map_ocl_err)?;
    }
    // No rt.finish() — caller synchronizes via read_buffer when needed.

    Ok((x_buf, lambda_min_buf))
}

// ------------------------------------------------------------------
// Phase 3: N>64 S^{-1/2} via tiled GEMM
// ------------------------------------------------------------------

/// Build S^{-1/2} for N>64 using tiled GEMM.
///
/// X = V · diag(rsqrt(λ)) · V^T
///
/// Steps:
/// 1. Diagonalize S via `jacobi_batched` (tiled for N>64)
/// 2. Scale V → V_scaled = V · diag(rsqrt(λ)) via `scale_eigenvectors_batched`
/// 3. X = V_scaled · V^T via `matmul_tiled_batched`
///
/// All operations use global memory only — no N² local memory limit.
fn build_inv_sqrt_tiled(
    rt: &mut GpuRuntime,
    s_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
) -> Result<(Buffer<f32>, Buffer<f32>)> {
    if n == 0 || batch == 0 {
        let x = rt.zero_buffer::<f32>(n * n * batch)?;
        let lm = rt.zero_buffer::<f32>(batch)?;
        return Ok((x, lm));
    }

    // Step 1: copy S to working buffer and diagonalize.
    let a_work = rt.copy_buffer(s_buf, n * n * batch)?;
    let v_buf = rt.zero_buffer::<f32>(n * n * batch)?;
    jacobi_batched(rt, &a_work, &v_buf, n, batch)?;

    // Step 2: scale V → V_scaled = V · rsqrt(λ), get λ_min.
    let v_scaled = rt.zero_buffer::<f32>(n * n * batch)?;
    let lambda_min_buf = rt.zero_buffer::<f32>(batch)?;
    // Use a simple 1D launch: one workgroup per system, 256 threads.
    let wg = 256usize;
    let source = render_source(64);  // reuse the template (LAMBDA_FLOOR define)
    let program = rt.build_program(&source)?;
    let kernel = Kernel::builder()
        .program(&program)
        .name("scale_eigenvectors_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(&a_work)
        .arg(&v_buf)
        .arg(&v_scaled)
        .arg(&lambda_min_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;
    unsafe { kernel.enq().map_err(map_ocl_err)?; }

    // Step 3: X = V_scaled · V^T (tiled GEMM, transpose B).
    let x_buf = rt.zero_buffer::<f32>(n * n * batch)?;
    // C = V_scaled · V^T → trans_a=false, trans_b=true
    // Use matmul_tiled_batched_params for the transpose.
    matmul_tiled_batched_params(
        rt, &v_scaled, &v_buf, &x_buf, n, batch,
        false, true, 1.0, 0.0,
    )?;

    Ok((x_buf, lambda_min_buf))
}

/// Dispatch S^{-1/2} construction: full-local for N≤64, tiled for N>64.
/// This is the production entry point.
pub fn build_inv_sqrt_batched(
    rt: &mut GpuRuntime,
    s_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
) -> Result<(Buffer<f32>, Buffer<f32>)> {
    if n <= 64 {
        build_inv_sqrt(rt, s_buf, n, batch)
    } else {
        build_inv_sqrt_tiled(rt, s_buf, n, batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector, SymmetricEigen};
    use ocl::{enums::{DeviceInfo, KernelWorkGroupInfo, ProfilingInfo}, flags, Event, Queue};
    use std::time::Instant;

    fn fixture(n: usize, kind: usize) -> DMatrix<f64> {
        let v = DVector::from_fn(n, |i, _| ((i + 1) as f64).sin());
        let q = DMatrix::identity(n, n) - (&v * v.transpose()) * (2.0 / v.norm_squared());
        let d = DMatrix::from_diagonal(&DVector::from_fn(n, |i, _| if kind == 1 { (i / 2) as f64 * 0.2 - 1.0 } else { i as f64 * 0.2 - 1.0 }));
        let a = if kind == 2 { d } else { &q * d * q.transpose() };
        DMatrix::from_fn(n, n, |i, j| a[(i.min(j), i.max(j))] as f32 as f64)
    }

    fn check_variants(configs: &[(usize, usize)], repeats: usize) {
        assert!(repeats % 2 == 1, "Jacobi comparison needs odd repeats so the final readback is the old variant; repeats={repeats}");
        let mut rt = GpuRuntime::new().expect("Jacobi verification requires a working OpenCL device");
        let name = rt.device().name().expect("query Jacobi verification device name");
        let vendor = rt.device().vendor().expect("query Jacobi verification device vendor");
        assert!(vendor.contains("NVIDIA"), "Jacobi performance verification requires NVIDIA, selected {vendor}: {name}; select OCL_DEFAULT_PLATFORM_IDX from clinfo -l");
        eprintln!("Jacobi verification: {name}, local={:?}, max_wg={:?}, repeats={repeats}", rt.device().info(DeviceInfo::LocalMemSize).expect("query device local memory"), rt.device().info(DeviceInfo::MaxWorkGroupSize).expect("query device WG limit"));
        let queue = Queue::new(rt.context(), *rt.device(), Some(flags::CommandQueueProperties::new().profiling())).expect("create Jacobi profiling queue");
        for &(n, batch) in configs {
            let (_, _, _, _, wg) = spec_params(n);
            let fixtures: Vec<DMatrix<f64>> = (0..3).map(|kind| fixture(n, kind)).collect();
            let references: Vec<Vec<f64>> = fixtures.iter().map(|a| {
                let mut vals = SymmetricEigen::new(a.clone()).eigenvalues.as_slice().to_vec();
                vals.sort_by(f64::total_cmp);
                vals
            }).collect();
            let mut input = vec![0.0f32; batch * n * n];
            for b in 0..batch { for i in 0..n { for j in 0..n { input[b*n*n+i*n+j] = fixtures[b % 3][(i, j)] as f32; } } }
            let src = rt.buffer_from_slice(&input).expect("upload Jacobi fixtures");
            let a = rt.zero_buffer::<f32>(input.len()).expect("allocate Jacobi A");
            let v = rt.zero_buffer::<f32>(input.len()).expect("allocate Jacobi V");
            let mut ah = vec![0.0f32; input.len()];
            let mut vh = vec![0.0f32; input.len()];
            let mut block_a = vec![0.0f32; input.len()];
            let mut block_v = vec![0.0f32; input.len()];
            let mut accuracy_ok = true;
            let kernels: Vec<Kernel> = (0..2).map(|mode| {
                let source = format!("#define JACOBI_BLOCK_UPDATE {mode}\n{}", render_source(n));
                let program = rt.build_program(&source).expect("compile Jacobi comparison variant");
                let kernel = Kernel::builder().program(&program).name("jacobi_cyclic_local_batched").queue(queue.clone())
                    .global_work_size(batch * wg).local_work_size(wg).arg(&a).arg(&v).arg(n as i32).arg(batch as i32)
                    .build().expect("build persistent Jacobi comparison kernel");
                eprintln!("  N={n} batch={batch} mode={mode} WG={wg} local={:?} private={:?} kernel_max_wg={:?}", kernel.wg_info(*rt.device(), KernelWorkGroupInfo::LocalMemSize).expect("query kernel local memory"), kernel.wg_info(*rt.device(), KernelWorkGroupInfo::PrivateMemSize).expect("query kernel private memory"), kernel.wg_info(*rt.device(), KernelWorkGroupInfo::WorkGroupSize).expect("query kernel WG limit"));
                kernel
            }).collect();
            rt.finish().expect("finish fixture uploads before using profiling queue");
            let mut times = [vec![0.0f64; repeats], vec![0.0f64; repeats]];
            let mut walls = [vec![0.0f64; repeats], vec![0.0f64; repeats]];
            for sample in 0..=repeats {
                for turn in 0..2 {
                    let mode = (sample + turn) % 2;
                    src.cmd().queue(&queue).copy(&a, None, None).enq().expect("reset Jacobi input by device copy");
                    queue.finish().expect("finish Jacobi input reset outside timing");
                    let mut event = Event::empty();
                    let start = Instant::now();
                    unsafe { kernels[mode].cmd().enew(&mut event).enq().expect("enqueue Jacobi comparison"); }
                    event.wait_for().expect("wait for Jacobi comparison event");
                    let wall = start.elapsed().as_secs_f64() * 1e6;
                    let t0 = event.profiling_info(ProfilingInfo::Start).expect("Jacobi event START").time().expect("START timestamp");
                    let t1 = event.profiling_info(ProfilingInfo::End).expect("Jacobi event END").time().expect("END timestamp");
                    assert!(t1 > t0, "invalid Jacobi timestamps: N={n} batch={batch} mode={mode} start={t0} end={t1}");
                    if sample > 0 { times[mode][sample-1] = (t1 - t0) as f64 * 1e-3; walls[mode][sample-1] = wall; }
                    if sample != repeats { continue; }
                    a.cmd().queue(&queue).read(&mut ah).enq().expect("read Jacobi A for parity");
                    v.cmd().queue(&queue).read(&mut vh).enq().expect("read Jacobi V for parity");
                    assert!(ah.iter().chain(&vh).all(|x| x.is_finite()), "non-finite Jacobi output N={n} batch={batch} mode={mode}");
                    let mut worst = [0.0f64; 4];
                    for b in 0..batch {
                        let vals = DVector::from_fn(n, |i, _| ah[b*n*n+i*n+i] as f64);
                        let eig = DMatrix::from_fn(n, n, |i, j| vh[b*n*n+i*n+j] as f64);
                        let d = DMatrix::from_diagonal(&vals);
                        let orig = &fixtures[b % 3];
                        let mut sorted = vals.as_slice().to_vec();
                        sorted.sort_by(f64::total_cmp);
                        let errors = [
                            sorted.iter().zip(&references[b % 3]).map(|(a, b)| (a-b).abs()).fold(0.0, f64::max),
                            (orig * &eig - &eig * &d).amax(),
                            (&eig * &d * eig.transpose() - orig).amax(),
                            (eig.transpose() * &eig - DMatrix::identity(n, n)).amax(),
                        ];
                        if batch <= 3 { eprintln!("    N={n} mode={mode} system={b} spectrum/residual/reconstruction/orthogonality={errors:?}"); }
                        for i in 0..4 { worst[i] = worst[i].max(errors[i]); }
                        if !errors.iter().all(|e| e.is_finite() && *e < 1e-4) {
                            eprintln!("Jacobi accuracy violation N={n} batch={batch} mode={mode} system={b}, errors={errors:?}, tolerance=1e-4");
                            accuracy_ok = false;
                        }
                    }
                    if mode == 1 { block_a.copy_from_slice(&ah); block_v.copy_from_slice(&vh); }
                    eprintln!("  N={n} batch={batch} mode={mode} worst_errors={worst:?}");
                }
            }
            let differences = ah.iter().chain(&vh).zip(block_a.iter().chain(&block_v)).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            eprintln!("  N={n} batch={batch} old/block bitwise-different outputs={differences}");
            assert_eq!(differences, 0, "Jacobi block transform differs from original on NVIDIA: N={n} batch={batch}");
            for mode in 0..2 { times[mode].sort_by(f64::total_cmp); walls[mode].sort_by(f64::total_cmp); }
            eprintln!("Jacobi N={n} batch={batch}: event_us old={:.3} block={:.3} speedup={:.3}; enqueue+wait_us old={:.3} block={:.3}", times[0][repeats/2], times[1][repeats/2], times[0][repeats/2]/times[1][repeats/2], walls[0][repeats/2], walls[1][repeats/2]);
            assert!(accuracy_ok, "Jacobi absolute accuracy failed N={n} batch={batch}; per-system old/block diagnostics above, tolerance=1e-4 (unchanged)");
        }
        eprintln!("Jacobi verification finished");
    }

    #[test]
    #[ignore]
    fn jacobi_sweep_diagnostic() {
        let mut rt = GpuRuntime::new().expect("Jacobi sweep diagnostic requires OpenCL");
        eprintln!("Jacobi sweep diagnostic on {}", rt.device().name().expect("query device name"));
        let n = 64;
        let orig = fixture(n, 0);
        let input: Vec<f32> = (0..n*n).map(|ij| orig[(ij/n, ij%n)] as f32).collect();
        let a = rt.buffer_from_slice(&input).expect("upload Jacobi diagnostic matrix");
        let v = rt.zero_buffer::<f32>(input.len()).expect("allocate Jacobi diagnostic eigenvectors");
        let mut ah = vec![0.0f32; input.len()];
        let mut vh = vec![0.0f32; input.len()];
        let (_, _, _, _, wg) = spec_params(n);
        for (rsqrt_mode, sweeps) in (0..3).flat_map(|mode| [1, 2, 3, 4, 5, 8, 12, 20].map(|sweeps| (mode, sweeps))) {
            let mut source = format!("#define JACOBI_NORMALIZE_ROTATION {}\n#define MAX_SWEEPS {sweeps}\n{}", usize::from(rsqrt_mode == 2), render_source(n))
                .replace("gA[i] = (r == c) ? lA[r * JLD + c] : 0.0f;", "gA[i] = lA[r * JLD + c];");
            if rsqrt_mode == 1 { source = source.replace("float c = 1.0f / sqrt(1.0f + t * t);", "float c = rsqrt(1.0f + t * t);"); }
            let program = rt.build_program(&source).expect("compile Jacobi sweep diagnostic");
            let kernel = Kernel::builder().program(&program).name("jacobi_cyclic_local_batched").queue(rt.queue().clone())
                .global_work_size(wg).local_work_size(wg).arg(&a).arg(&v).arg(n as i32).arg(1i32)
                .build().expect("build Jacobi sweep diagnostic");
            a.write(&input).enq().expect("reset Jacobi diagnostic input");
            unsafe { kernel.enq().expect("enqueue Jacobi sweep diagnostic"); }
            rt.read_buffer(&a, &mut ah).expect("read full transformed Jacobi matrix");
            rt.read_buffer(&v, &mut vh).expect("read diagnostic eigenvectors");
            assert!(ah.iter().chain(&vh).all(|x| x.is_finite()), "Jacobi diagnostic nonfinite at sweeps={sweeps}");
            let b = DMatrix::from_fn(n, n, |i, j| ah[i*n+j] as f64);
            let eig = DMatrix::from_fn(n, n, |i, j| vh[i*n+j] as f64);
            let d = DMatrix::from_diagonal(&b.diagonal());
            let gram = eig.transpose() * &eig;
            eprintln!("rsqrt_mode={rsqrt_mode} sweeps={sweeps}: offnorm={:.9e} AV-VD={:.9e} AV-VB={:.9e} reconstruction={:.9e} full_reconstruction={:.9e} orthogonality={:.9e} trace_drift={:.9e} frobenius_drift={:.9e} column_norm2_min={:.9e} max={:.9e}",
                (&b-&d).norm(), (&orig*&eig-&eig*&d).amax(), (&orig*&eig-&eig*&b).amax(), (&eig*&d*eig.transpose()-&orig).amax(), (&eig*&b*eig.transpose()-&orig).amax(),
                (&gram-DMatrix::identity(n,n)).amax(), b.trace()-orig.trace(), b.norm()-orig.norm(), gram.diagonal().min(), gram.diagonal().max());
        }
    }

    #[test]
    fn jacobi_block_parity() {
        check_variants(&[(1, 3), (2, 3), (3, 3), (7, 3), (8, 3), (14, 3), (27, 3), (28, 3), (32, 3), (63, 3), (64, 3)], 1);
    }

    #[test]
    #[ignore]
    fn jacobi_block_benchmark() {
        check_variants(&[(8, 100), (28, 1), (28, 100), (28, 1000), (32, 100), (64, 100)], 7);
    }
}
