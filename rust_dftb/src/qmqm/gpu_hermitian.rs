//! Complex Hermitian eigensolver + complex GEMM wrappers for the PBC path.
//!
//! The complex-valued sibling of `gpu_eigen.rs` / `gpu_matrix.rs` for
//! arbitrary-k PBC: H(k), S(k) are Hermitian `float2` matrices, one
//! workgroup per flat (replica, k) system — `sid = rep*nk + kpt`.
//!
//! # Architecture (mirrors the real path — see Dense_Multi_PBC notes)
//!
//! - `jacobi_hermitian_cyclic_global_batched` — direct Brent–Luk parallel
//!   cyclic Jacobi on the FULL matrix in global memory, unitary 2×2
//!   rotations J = [[c, s·u], [−s·u*, c]] (u = A_pq/|A_pq|); same ~3
//!   barriers/round, relative pivot skip, per-sweep Hermiticity restore
//!   (the f32 restoring invariant), and diag[4] stop contract as the real
//!   `jacobi_cyclic_global_batched`.
//! - `zgemm_*_batched` — tiled complex GEMM with op ∈ {N, T, H}; the
//!   conjugate is applied at the local-tile load so the inner product is
//!   a uniform complex FMA chain (4 independent accumulators).
//! - Occupation under PBC couples all k-points of a replica through μ:
//!   `kpoint_occ_batched` solves Σ_k w_k Σ_b f(ε_bk−μ) = n_occ per replica
//!   and writes occ_w = w_k·f — the real kernel's in-Jacobi Fermi tail
//!   cannot exist here (cross-workgroup k-reduction).
//!
//! Buffers are `Buffer<Float2>` (`__global float2*`), row-major
//! `[sid][n*n]`. Charge-side vectors stay real (`Buffer<f32>`,
//! `[rep*n_atoms]`) — the real DIIS/mixer kernels are reused unchanged.
//!
//! Capacity: n ≤ 256 (128 __local rotation slots), same as the real path.

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_eigen::jacobi_sweeps;
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::prm::Float2;
use ocl::{Buffer, Kernel};

/// Hermitian Jacobi kernel source (template — text-substituted per config).
pub const HERM_JACOBI_TEMPLATE: &str = include_str!("gpu_hermitian_jacobi.cl");
/// Complex matrix-ops kernel source (zgemm, hscc, occ, mulliken, ...).
pub const ZMATRIX_TEMPLATE: &str = include_str!("gpu_zmatrix_ops.cl");

/// ZGEMM operand ops.
pub const ZOP_N: i32 = 0;
pub const ZOP_T: i32 = 1;
pub const ZOP_H: i32 = 2;

/// Render the Hermitian Jacobi template.
/// `wg` — workgroup size (512 measured fastest on RTX 3090 for the real
/// kernel; clamped by the caller to the device limit).
/// `rot_fp64` — HJ_ROT_FP64: f64 scalar c,s construction (A/B knob; the
/// f32 variant trades rotation exactness for throughput — self-correcting
/// under Jacobi iteration, same argument as the real kernel's prec=0).
pub fn render_hermitian_source(wg: usize, rot_fp64: bool) -> String {
    HERM_JACOBI_TEMPLATE
        .replace("#define WG 512", &format!("#define WG {wg}"))
        .replace(
            "#define MAX_CSWEEPS 40",
            &format!("#define MAX_CSWEEPS {}", jacobi_sweeps(40)),
        )
        .replace(
            "#define HJ_ROT_FP64 1",
            &format!("#define HJ_ROT_FP64 {}", rot_fp64 as i32),
        )
        .replace(
            "#ifndef JACOBI_OFF_TOL\n#define JACOBI_OFF_TOL 1.0e-6f\n#endif",
            &format!(
                "#define JACOBI_OFF_TOL {:.3e}f",
                std::env::var("RUST_DFTB_JACOBI_OFF_TOL")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(1.0e-6)
            ),
        )
}

/// Render the complex matrix-ops template (tile sizes are defaults for now).
pub fn render_zmatrix_source() -> String {
    ZMATRIX_TEMPLATE.to_string()
}

/// Batched complex Hermitian Jacobi — the general entry point.
///
/// - `a_buf` — `[n_sys][n*n]` Float2 Hermitian in/out (eigenvalues on Re diag)
/// - `v_buf` — `[n_sys][n*n]` Float2 in/out (eigenvectors; kept when init_v=1)
/// - `n` — matrix dimension (n ≤ 256)
/// - `n_sys` — flat (replica,k) system count
/// - `nk` — k-points per replica (replica mask stride; 1 = per-system)
/// - `active` — per-replica gate `[n_rep]` (`n_sys/nk` entries)
/// - `diag` — `[n_sys*4]` stop record {off, off/‖A‖_F, stop, sweeps}
/// - `init_v` — 0 cold (V←I), 1 warm (V rotated in place)
pub fn hermitian_jacobi_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<Float2>,
    v_buf: &Buffer<Float2>,
    n: usize,
    n_sys: usize,
    nk: usize,
    active: &Buffer<i32>,
    diag: &Buffer<f32>,
    init_v: i32,
    rot_fp64: bool,
) -> Result<()> {
    if n == 0 || n_sys == 0 {
        return Ok(());
    }
    if n > 256 {
        return Err(DftbError::InvalidInput(format!(
            "hermitian_jacobi_batched: n={n} exceeds capacity 256 (jn/2 pair rotations fit __local rot[128])"
        )));
    }
    if n_sys % nk != 0 {
        return Err(DftbError::InvalidInput(format!(
            "hermitian_jacobi_batched: n_sys={n_sys} not divisible by nk={nk}"
        )));
    }
    let wg = rt.caps().max_work_group_size.min(512).max(64);
    let source = render_hermitian_source(wg, rot_fp64);
    let program = rt.build_program(&source)?;
    let kernel = Kernel::builder()
        .program(&program)
        .name("jacobi_hermitian_cyclic_global_batched")
        .queue(rt.queue().clone())
        .global_work_size(n_sys * wg)
        .local_work_size(wg)
        .arg(a_buf)
        .arg(v_buf)
        .arg(n as i32)
        .arg(n_sys as i32)
        .arg(init_v)
        .arg(nk as i32)
        .arg(active)
        .arg(diag)
        .build()
        .map_err(map_ocl_err)?;
    unsafe { kernel.enq().map_err(map_ocl_err)?; }
    // No rt.finish() — in-order queue; caller syncs at genuine readbacks.
    Ok(())
}

/// Standalone test/bench helper: all-active solve, returns the per-system
/// diag record `[n_sys][4]` = {off, off/‖A‖_F, stop, sweeps}.
pub fn hermitian_jacobi_simple(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<Float2>,
    v_buf: &Buffer<Float2>,
    n: usize,
    n_sys: usize,
    rot_fp64: bool,
) -> Result<Vec<f32>> {
    let active = rt.buffer_from_slice(&vec![1i32; n_sys])?;
    let diag = rt.zero_buffer::<f32>(4 * n_sys)?;
    hermitian_jacobi_batched(rt, a_buf, v_buf, n, n_sys, 1, &active, &diag, 0, rot_fp64)?;
    let mut d = vec![0.0f32; 4 * n_sys];
    rt.read_buffer(&diag, &mut d)?;
    Ok(d)
}

/// Standalone V·rsqrt(λ) scale helper (tests + S^{-1/2} driver).
/// `a_buf` holds the Jacobi-diagonalized Hermitian matrix (eigenvalues on
/// Re(diag)); `v_buf` its eigenvectors. Writes `vs_buf` = V·diag(rsqrt λ)
/// and `lambda_min` per system.
pub fn zscale_eigenvectors_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<Float2>,
    v_buf: &Buffer<Float2>,
    vs_buf: &Buffer<Float2>,
    lambda_min: &Buffer<f32>,
    n: usize,
    n_sys: usize,
    nk: usize,
) -> Result<()> {
    let program = rt.build_program(&render_zmatrix_source())?;
    let active = rt.buffer_from_slice(&vec![1i32; n_sys / nk])?;
    let wg = 256usize;
    let kernel = Kernel::builder()
        .program(&program)
        .name("zscale_eigenvectors_batched")
        .queue(rt.queue().clone())
        .global_work_size(n_sys * wg)
        .local_work_size(wg)
        .arg(n as i32)
        .arg(n_sys as i32)
        .arg(nk as i32)
        .arg(a_buf)
        .arg(v_buf)
        .arg(vs_buf)
        .arg(lambda_min)
        .arg(&active)
        .build()
        .map_err(map_ocl_err)?;
    unsafe { kernel.enq().map_err(map_ocl_err)?; }
    Ok(())
}

/// Standalone complex GEMM helper (tests): C_b = op(A_b)·op(B_b).
/// op: 0 = N, 1 = T, 2 = H (conjugate transpose).
pub fn zgemm_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<Float2>,
    b_buf: &Buffer<Float2>,
    c_buf: &Buffer<Float2>,
    n: usize,
    batch: usize,
    op_a: i32,
    op_b: i32,
) -> Result<()> {
    if n == 0 || batch == 0 {
        return Ok(());
    }
    let program = rt.build_program(&render_zmatrix_source())?;
    let rg = (n + ZTILE_M - 1) / ZTILE_M;
    let cg = (n + ZTILE_N - 1) / ZTILE_N;
    let gws = ocl::SpatialDims::Three(cg * ZTILE_N, rg * ZTILE_M, batch);
    let lws = ocl::SpatialDims::Two(ZTILE_N, ZTILE_M);
    let kernel = Kernel::builder()
        .program(&program)
        .name("zgemm_batched")
        .queue(rt.queue().clone())
        .global_work_size(gws)
        .local_work_size(lws)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(op_a)
        .arg(op_b)
        .arg(1.0f32)
        .arg(0.0f32)
        .arg(a_buf)
        .arg(b_buf)
        .arg(c_buf)
        .arg_local::<Float2>(ZTILE_M * ZTILE_K)
        .arg_local::<Float2>(ZTILE_N * (ZTILE_K + 1))
        .build()
        .map_err(map_ocl_err)?;
    unsafe { kernel.enq().map_err(map_ocl_err)?; }
    Ok(())
}

/// Tile geometry shared with `gpu_pbc_plan` (mirrors real TILE_M/N/K).
pub const ZTILE_M: usize = 16;
pub const ZTILE_N: usize = 16;
pub const ZTILE_K: usize = 32;
