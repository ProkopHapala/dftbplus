//! GPU B-spline evaluator parity test (P1, manifest v3 §4.2 Gate A).
//!
//! Verifies that the f32 GPU `bspline3_v_d1_d2` evaluator matches the f64
//! CPU canonical spline in V, V', V'' at many points.
//!
//! This is the "f32 GPU matches the same f64 canonical spline" test from
//! Gate A. The CPU reference is `bspline3_eval_at` in `spline_resample.rs`,
//! which has been verified against the natural cubic spline through the same
//! grid points to near-machine precision.
//!
//! GPU B-spline evaluator parity test (P1, manifest v3 §4.2 Gate A).
//!
//! Verifies that the f32 GPU `bspline3_v_d1_d2` evaluator matches the f64
//! CPU canonical spline in V, V', V'' at many points.
//!
//! This is the "f32 GPU matches the same f64 canonical spline" test from
//! Gate A. The CPU reference is `bspline3_eval_at` in `spline_resample.rs`,
//! which has been verified against the natural cubic spline through the same
//! grid points to near-machine precision.
//!
//! Skip only if the ICD exposes zero OpenCL platforms. A device that
//! errors, or a non-NVIDIA device without `RUST_DFTB_ALLOW_CPU_CL=1`, is a
//! failure.

use ocl::{builders::ProgramBuilder, flags, Buffer, Kernel, Program};
use rust_dftb::methods::dftb::spline_resample::{
    bspline3_eval_at, resample_bspline,
};
use rust_dftb::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use rust_dftb::methods::sparse::harness::require_nvidia_runtime;

/// Minimal OpenCL source: just the bspline3_v_d1_d2 function + a test kernel
/// that evaluates it at N points and writes (V, dV, d²V) to an output buffer.
const BSPLINE_TEST_SRC: &str = r#"
inline float3 bspline3_v_d1_d2(
    float c0, float c1, float c2, float c3,
    float t, float inv_dr
) {
    float t2 = t * t;
    float t3 = t2 * t;
    float u  = 1.0f - t;

    float b0 = u * u * u * 0.16666667f;
    float b1 = (3.0f * t3 - 6.0f * t2 + 4.0f) * 0.16666667f;
    float b2 = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) * 0.16666667f;
    float b3 = t3 * 0.16666667f;

    float d0 = -0.5f * u * u;
    float d1 =  1.5f * t2 - 2.0f * t;
    float d2 = -1.5f * t2 + t + 0.5f;
    float d3 =  0.5f * t2;

    float dd0 =  1.0f - t;
    float dd1 =  3.0f * t - 2.0f;
    float dd2 = -3.0f * t + 1.0f;
    float dd3 =  t;

    float v  = fma(c0, b0, fma(c1, b1, fma(c2, b2, c3 * b3)));
    float dv = (fma(c0, d0, fma(c1, d1, fma(c2, d2, c3 * d3)))) * inv_dr;
    float ddv = (fma(c0, dd0, fma(c1, dd1, fma(c2, dd2, c3 * dd3)))) * inv_dr * inv_dr;
    return (float3)(v, dv, ddv);
}

// Test kernel: evaluate bspline3 at N points.
// controls: f32 control points (n_ctrl floats)
// points:   f32 query positions (r values, N floats)
// output:   float3 per point (V, dV/dr, d²V/dr²), N*3 floats
__kernel void test_bspline_eval(
    const int    n_ctrl,
    __global const float* controls,
    const float  dr,
    const int    n_points,
    __global const float* points,
    __global float* output
) {
    int gid = get_global_id(0);
    if (gid >= n_points) return;

    float r = points[gid];
    float inv_dr = 1.0f / dr;
    float u = r / dr;
    int i = (int)u;
    // Clamp to valid range [0, n_ctrl-2]
    if (i < 0) i = 0;
    if (i > n_ctrl - 2) i = n_ctrl - 2;
    float t = u - (float)i;

    // Phantom boundary control points (natural spline: d²=0 at endpoints)
    float c0 = (i == 0)              ? (2.0f * controls[0] - controls[1])              : controls[i - 1];
    float c1 = controls[i];
    float c2 = controls[i + 1];
    float c3 = (i + 2 >= n_ctrl)     ? (2.0f * controls[n_ctrl-1] - controls[n_ctrl-2]) : controls[i + 2];

    float3 result = bspline3_v_d1_d2(c0, c1, c2, c3, t, inv_dr);
    output[gid * 3 + 0] = result.s0;
    output[gid * 3 + 1] = result.s1;
    output[gid * 3 + 2] = result.s2;
}
"#;

fn try_gpu() -> Option<GpuRuntime> {
    require_nvidia_runtime()
}

#[test]
fn test_gpu_bspline3_v_d1_d2_parity() {
    let Some(mut rt) = try_gpu() else { return; };

    // Build a canonical B-spline from SK-like data (exponential decay)
    let h_orig = 0.01_f64;
    let y_orig: Vec<f64> = (0..500).map(|i| (-i as f64 * h_orig * 2.0).exp()).collect();
    let (ctrl_f32, dr_f32) = resample_bspline(&y_orig, h_orig, 64);
    let ctrl_f64: Vec<f64> = ctrl_f32.iter().map(|&v| v as f64).collect();
    let dr = dr_f32 as f64;

    // Query points: interior of the spline range
    let r_max = (ctrl_f64.len() - 1) as f64 * dr;
    let n_points = 200usize;
    let points: Vec<f32> = (0..n_points)
        .map(|k| (k as f64 * r_max / (n_points as f64 - 1.0)) as f32)
        .collect();

    // Compile the test kernel
    let device = rt.device().clone();
    let context = rt.context().clone();
    let mut builder = ProgramBuilder::new();
    builder.devices(device);
    builder.src(BSPLINE_TEST_SRC);
    let program = builder.build(&context).map_err(map_ocl_err).unwrap();

    // Allocate buffers
    let queue = rt.queue().clone();
    let ctrl_buf = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(ctrl_f32.len())
        .copy_host_slice(&ctrl_f32)
        .build().map_err(map_ocl_err).unwrap();
    let points_buf = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(n_points)
        .copy_host_slice(&points)
        .build().map_err(map_ocl_err).unwrap();
    let out_buf = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_WRITE_ONLY)
        .len(n_points * 3)
        .build().map_err(map_ocl_err).unwrap();

    // Build and launch the kernel
    let kernel = Kernel::builder()
        .program(&program)
        .name("test_bspline_eval")
        .queue(queue.clone())
        .arg(&(ctrl_f32.len() as i32))
        .arg(&ctrl_buf)
        .arg(&dr_f32)
        .arg(&(n_points as i32))
        .arg(&points_buf)
        .arg(&out_buf)
        .global_work_size(n_points)
        .build().map_err(map_ocl_err).unwrap();

    unsafe { kernel.enq().map_err(map_ocl_err).unwrap(); }

    // Read results
    let mut gpu_out = vec![0.0f32; n_points * 3];
    rt.read_buffer(&out_buf, &mut gpu_out).unwrap();

    // Compare against CPU f64 reference
    let mut max_err_v = 0.0_f64;
    let mut max_err_dv = 0.0_f64;
    let mut max_err_ddv = 0.0_f64;
    for k in 0..n_points {
        let r = points[k] as f64;
        let (v_ref, dv_ref, ddv_ref) = bspline3_eval_at(&ctrl_f64, dr, r);
        let v_gpu = gpu_out[k * 3] as f64;
        let dv_gpu = gpu_out[k * 3 + 1] as f64;
        let ddv_gpu = gpu_out[k * 3 + 2] as f64;

        // Skip points outside the valid range (both CPU and GPU return 0)
        if r < 0.0 || r > r_max { continue; }

        max_err_v = max_err_v.max((v_gpu - v_ref).abs());
        max_err_dv = max_err_dv.max((dv_gpu - dv_ref).abs());
        max_err_ddv = max_err_ddv.max((ddv_gpu - ddv_ref).abs());
    }

    eprintln!("GPU bspline3 vs CPU f64 (64 nodes, 200 points):");
    eprintln!("  max|ΔV|   = {max_err_v:.3e}");
    eprintln!("  max|ΔV'|  = {max_err_dv:.3e}");
    eprintln!("  max|ΔV''| = {max_err_ddv:.3e}");

    // f32 GPU vs f64 CPU: expect ~1e-6 for V (f32 roundoff), slightly larger
    // for derivatives due to amplification by inv_dr and inv_dr².
    assert!(max_err_v < 1e-5, "V error too large: {max_err_v:.3e}");
    assert!(max_err_dv < 1e-4, "V' error too large: {max_err_dv:.3e}");
    assert!(max_err_ddv < 1e-2, "V'' error too large: {max_err_ddv:.3e}");
}
