//! Dense batched density-matrix purification (TC2) — FOE path.
//!
//! One workgroup per system; designed for the many-small-system regime
//! (batch≈400, n≈86) where the Jacobi eigensolver is round-latency-bound
//! (~5 % of peak). All iteration work is batched n³ GEMM — see
//! doc/prokop/tasts/HBond_Relaxed_Scan_GPU/Alternative_Dense_Multi_Eigensolve.md
//! §6 for the design. Semantics mirror the sparse metric-TC2 path
//! (`gpu_sparse.rs::tc2_step`): branch `Tr(D) > Nocc → D←D² else 2D−D²`,
//! fail-loud acceptance requires BOTH ‖D²−D‖_F < tol AND |Tr(D)−Nocc| ≤
//! trace tolerance (a converged wrong-rank projector is not accepted).
//!
//! This module is a separate code path — it does not replace or modify
//! the Jacobi eigensolver; plan-level dispatch is opt-in.

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::{Buffer, Kernel};

const GPU_PURIFY_TEMPLATE: &str = include_str!("gpu_purify.cl");

/// Default workgroup size for the purify kernels.
const PURIFY_WG_DEFAULT: usize = 256;
/// Convergence check granularity — one tiny host readback per CHUNK
/// iterations (not per iteration: zero syncs inside the chunk).
const PURIFY_CHUNK: usize = 4;

/// Render the purify source: `wg` = workgroup size, `tile` = PURIFY_TILE
/// (0 → barrier-free row·row GEMM; 16/32 → cooperative local tiles).
pub fn render_purify_source(wg: usize, tile: usize) -> String {
    GPU_PURIFY_TEMPLATE
        .replace("#define PURIFY_WG 256", &format!("#define PURIFY_WG {wg}"))
        .replace("#define PURIFY_TILE 0", &format!("#define PURIFY_TILE {tile}"))
}

/// Per-system result of a batched TC2 purification.
pub struct PurifyDiag {
    /// ‖D²−D‖_F at the last launched iteration.
    pub errs: Vec<f32>,
    /// Tr(D) after the last launched iteration.
    pub traces: Vec<f32>,
    /// Iterations actually launched (≤ max_iter).
    pub iters: usize,
    /// Which internal buffer holds the final D (always copied to `d_out`
    /// before return — informational only).
    pub converged: bool,
}

/// Batched TC2 purification: H̃ (orthogonal basis, [batch][n²]) → density
/// D ([batch][n²], Tr(D)=nocc, idempotent). Cold start via Gershgorin
/// Palser guess. Returns the density in `d_out` plus diagnostics.
/// `nocc` = per-system occupied-orbital count (Tr(D) target, NOT
/// electron count — closed shell ⇒ nocc = Ne/2).
pub fn purify_tc2_batched(
    rt: &mut GpuRuntime,
    h_buf: &Buffer<f32>,
    d_out: &mut Vec<f32>,
    n: usize,
    batch: usize,
    nocc: &[f32],
    max_iter: usize,
    tol: f32,
) -> Result<PurifyDiag> {
    if nocc.len() != batch {
        return Err(DftbError::InvalidInput(format!(
            "purify_tc2_batched: nocc len {} != batch {batch}",
            nocc.len()
        )));
    }
    if batch == 0 {
        return Ok(PurifyDiag { errs: vec![], traces: vec![], iters: 0, converged: true });
    }
    // Tuning knobs (same convention as RUST_DFTB_JACOBI_*): workgroup size
    // and GEMM interior (0 = barrier-free row·row; 16/32 = local tiles).
    let wg: usize = std::env::var("RUST_DFTB_PURIFY_WG")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(PURIFY_WG_DEFAULT);
    let tile: usize = std::env::var("RUST_DFTB_PURIFY_TILE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let source = render_purify_source(wg, tile);
    let program = rt.build_program(&source)?;
    let d_a = rt.zero_buffer::<f32>(batch * n * n)?;
    let d_b = rt.zero_buffer::<f32>(batch * n * n)?;
    let nocc_b = rt.buffer_from_slice(nocc)?;
    let traces = rt.zero_buffer::<f32>(batch)?;
    let errs = rt.zero_buffer::<f32>(batch)?;
    let done = rt.zero_buffer::<i32>(batch)?;

    let k_init = Kernel::builder()
        .program(&program)
        .name("tc2_init_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(h_buf)
        .arg(&d_a)
        .arg(&traces)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;
    let k_step = Kernel::builder()
        .program(&program)
        .name("tc2_step_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(&d_a)
        .arg(&d_b)
        .arg(&nocc_b)
        .arg(&traces)
        .arg(&errs)
        .arg(&done)
        .arg(tol)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;

    unsafe { k_init.enq().map_err(map_ocl_err)?; }

    if std::env::var_os("PURIFY_DEBUG_INIT").is_some() {
        let mut d0 = vec![0.0f32; batch * n * n];
        rt.read_buffer(&d_a, &mut d0)?;
        for s in 0..batch {
            let sl = &d0[s * n * n..(s + 1) * n * n];
            let mx = sl.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
            let mut tr = 0.0f64;
            for i in 0..n { tr += sl[i * n + i] as f64; }
            if !(mx > 0.0 && mx < 10.0) || !tr.is_finite() {
                eprintln!("[purify-init] sys{s}: max|D0|={mx:.3e} Tr={tr:.4}");
            }
        }
        let mut tr_h = vec![0.0f32; batch];
        rt.read_buffer(&traces, &mut tr_h)?;
        for s in 0..batch {
            if !(tr_h[s] > 0.0 && tr_h[s] < n as f32) {
                eprintln!("[purify-init] sys{s}: traces={:.4e}", tr_h[s]);
            }
        }
    }

    let mut iters = 0usize;
    let mut converged = false;
    let mut err_h = vec![0.0f32; batch];
    let mut tr_h = vec![0.0f32; batch];
    // din/dout ping-pong: step reads arg0 (d_a), writes arg1 (d_b); swap each iter.
    let mut swapped = false; // false → din=d_a dout=d_b; true → reversed
    while iters < max_iter && !converged {
        for _ in 0..PURIFY_CHUNK.min(max_iter - iters) {
            if swapped {
                k_step.set_arg(0u32, &d_b).map_err(map_ocl_err)?;
                k_step.set_arg(1u32, &d_a).map_err(map_ocl_err)?;
            } else {
                k_step.set_arg(0u32, &d_a).map_err(map_ocl_err)?;
                k_step.set_arg(1u32, &d_b).map_err(map_ocl_err)?;
            }
            unsafe { k_step.enq().map_err(map_ocl_err)?; }
            swapped = !swapped;
            iters += 1;
        }
        rt.read_buffer(&errs, &mut err_h)?;
        rt.read_buffer(&traces, &mut tr_h)?;
        if std::env::var_os("PURIFY_DEBUG").is_some() {
            let (wi, we) = err_h
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, e)| (i, *e))
                .unwrap_or((0, 0.0));
            eprintln!(
                "[purify-dbg] iters={iters} sys0: err={:.4e} tr={:.6} | worst sys{wi}: err={we:.4e} tr={:.4e}",
                err_h[0], tr_h[0], tr_h[wi]
            );
        }
        converged = (0..batch).all(|i| {
            err_h[i] < tol && (tr_h[i] - nocc[i]).abs() <= trace_tol(nocc[i])
        });
    }

    // Final D lives in the buffer last WRITTEN. `swapped` holds the NEXT
    // launch direction: swapped==true → last launch wrote arg1=d_b.
    let final_buf = if swapped { &d_b } else { &d_a };
    rt.read_buffer(final_buf, d_out)?;
    Ok(PurifyDiag { errs: err_h, traces: tr_h, iters, converged })
}

/// Trace tolerance — same formula as the sparse TC2 path
/// (`gpu_sparse.rs::tc2_trace_tol`): scale with Nocc, absolute floor.
fn trace_tol(nocc: f32) -> f32 {
    (2e-5 * nocc).max(1e-4)
}
