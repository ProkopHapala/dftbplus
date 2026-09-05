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
//! each pair is handled by PPG work-items that split the JN rows. One
//! barrier per round. Up to MAX_SWEEPS sweeps with relative off-diagonal
//! norm convergence check.
//!
//! # Specialization
//!
//! The OpenCL source is text-substituted at runtime for the given N:
//! JN, JLD, JPAIR, JROUND, WG, PPG are baked in as `#define` constants.
//! This avoids dynamic indexing overhead in the inner loop and lets the
//! compiler unroll and optimize the fixed-size local memory operations.

use crate::core::error::{DftbError, Result};
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
    rt.finish()
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
    // We read S back to host and re-upload — simple and correct. The
    // coordinator can optimize this to a device-to-device copy in Wave 3.
    let mut s_host = vec![0.0f32; n * n * batch];
    rt.read_buffer(s_buf, &mut s_host)?;
    let a_work = rt.buffer_from_slice(&s_host)?;

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
    rt.finish()?;

    Ok((x_buf, lambda_min_buf))
}
