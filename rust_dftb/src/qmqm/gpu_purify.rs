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

/// GEMM interior selector for the fused TC2 step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PurifyGemm {
    /// row·row global dots — barrier-free, ~0.9 TFLOPS at n=86/b400.
    RowRow,
    /// cooperative local tiles (PURIFY_TILE) — BROKEN, kept for reference.
    Tiled(usize),
    /// register-tiled symmetric-square (measured best: 6.7 TFLOPS).
    /// Requires the WG tile (ty·RTY)×(tx·RTX) to cover n×n — shape is
    /// derived from `n` by `regtile_shape`.
    RegTileSq,
}

/// Register-tile shape for `RegTileSq`: fixed micro-tile RTY×RTX = 8×4
/// (measured optimum at n=86 — 32 accums, 242 threads); the thread grid
/// is derived so the WG tile covers n×n. Returns (tx, ty, rtx, rty, tk)
/// or None when infeasible (tx·ty > 1024 → needs a bigger micro-tile or
/// the split path — fail loud, pick RowRow).
fn regtile_shape(n: usize) -> Option<(usize, usize, usize, usize, usize)> {
    let (rtx, rty) = (4usize, 8usize);
    let tx = n.div_ceil(rtx);
    let ty = n.div_ceil(rty);
    if tx * ty > 1024 {
        return None;
    }
    // k-staging depth: 44 ≈ n/2 measured marginally fastest at n=86 but
    // costs 4× the local memory of tk=8 for ~equal perf → prefer tk=8;
    // local = tk·WM·4 B must stay small for WG residency.
    let tk = if 8 * ty * rty * 4 <= 48 * 1024 { 8 } else { 4 };
    Some((tx, ty, rtx, rty, tk))
}

/// Resolved (wg, tile, gemm, shape) for the purify kernels at a given n —
/// shared by the standalone bench (`purify_tc2_batched`) and the SCC
/// plan's persistent kernels so both compile the identical source.
/// `gemm` is the PURIFY_GEMM define (0 row·row / tiled, 2 regtile-sq).
/// Env: RUST_DFTB_PURIFY_{GEMM,WG,TILE,TK} — see purify_tc2_batched docs.
pub fn purify_shape_config(
    n: usize,
) -> Result<(usize, usize, usize, Option<(usize, usize, usize, usize, usize)>)> {
    let gemm_sel: usize = std::env::var("RUST_DFTB_PURIFY_GEMM")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let wg_env: usize = std::env::var("RUST_DFTB_PURIFY_WG")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(PURIFY_WG_DEFAULT);
    let (wg, tile, gemm, shape) = match gemm_sel {
        2 => {
            let mut sh = regtile_shape(n).ok_or_else(|| {
                DftbError::InvalidInput(format!(
                    "purify regtile GEMM infeasible for n={n} (thread grid > 1024) — \
                     set RUST_DFTB_PURIFY_GEMM=0 for the row·row path"
                ))
            })?;
            if let Some(tk) = std::env::var("RUST_DFTB_PURIFY_TK")
                .ok().and_then(|s| s.parse::<usize>().ok())
            {
                sh.4 = tk;
            }
            (sh.0 * sh.1, 0usize, 2usize, Some(sh))
        }
        1 => {
            let tile: usize = std::env::var("RUST_DFTB_PURIFY_TILE")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
            (wg_env, tile, 0usize, None)
        }
        _ => (wg_env, 0usize, 0usize, None),
    };
    Ok((wg, tile, gemm, shape))
}

/// Render the purify source with the given workgroup size, GEMM
/// interior, and (for RegTileSq) register-tile shape.
pub fn render_purify_source(
    wg: usize,
    tile: usize,
    gemm: usize,
    shape: Option<(usize, usize, usize, usize, usize)>,
) -> String {
    let mut src = GPU_PURIFY_TEMPLATE
        .replace("#define PURIFY_WG 256", &format!("#define PURIFY_WG {wg}"))
        .replace("#define PURIFY_TILE 0", &format!("#define PURIFY_TILE {tile}"))
        .replace("#define PURIFY_GEMM 0", &format!("#define PURIFY_GEMM {gemm}"));
    if let Some((tx, ty, rtx, rty, tk)) = shape {
        src = src
            .replace("#define PURIFY_TX 22", &format!("#define PURIFY_TX {tx}"))
            .replace("#define PURIFY_TY 11", &format!("#define PURIFY_TY {ty}"))
            .replace("#define PURIFY_RTX 4", &format!("#define PURIFY_RTX {rtx}"))
            .replace("#define PURIFY_RTY 8", &format!("#define PURIFY_RTY {rty}"))
            .replace("#define PURIFY_TK 8", &format!("#define PURIFY_TK {tk}"));
    }
    src
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
    // Tuning knobs (same convention as RUST_DFTB_JACOBI_*):
    //   RUST_DFTB_PURIFY_GEMM  2 = register-tiled symmetric-square
    //     (default; measured 6.7 TFLOPS vs 0.9 row·row at n=86/b400);
    //     1 = cooperative local tiles (BROKEN — kept for reference);
    //     0 = row·row reference path.
    //   RUST_DFTB_PURIFY_WG    workgroup size for GEMM 0/1 (GEMM 2
    //     derives wg = tx·ty from the register-tile shape).
    //   RUST_DFTB_PURIFY_TILE  local tile size for GEMM 1 only.
    //   RUST_DFTB_PURIFY_TK    k-staging depth for GEMM 2 (default 8).
    let (wg, tile, gemm, shape) = purify_shape_config(n)?;
    let source = render_purify_source(wg, tile, gemm, shape);
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
