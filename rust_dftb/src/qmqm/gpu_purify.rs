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

// ======================================================================
// relax+purify — energy-minimizer formulation (design: chat doc §17)
//
// TC2 purification is only a *retraction* here; the actual solver is a
// constrained descent on E = Tr(D·H). One fused kernel per step:
//
//   F  = (H−Λ) − a·I − b·D        (commuting modes projected out)
//   X  = D − α·F                  (α = η/Δε — Gershgorin span)
//   D' = q∓(q±(X))                1–2 resident symmetric squares
//   Λ += β(D'−X)/α                — LEARNED Λ DISABLED BY DEFAULT (β=0):
//                                  deposits measure only the normal part
//                                  of F; Λ's tangent component is invisible
//                                  to the update → phantom force →
//                                  degenerate fixed manifold (any clean
//                                  projector admits a self-consistent Λ).
//                                  Stateless F = H−aI−bD has the correct
//                                  fixed point T_D(H)=0 by construction.
//
// Convergence gate: the kernel-side `done` flag (step/corr/trace) does
// NOT test physical stationarity — a spurious projector passes it.
// At apparent convergence the driver launches gemm_nn(H·D→T) +
// comm_gate(‖T−Tᵀ‖/2‖T‖) and refuses convergence until it passes.
// ======================================================================

/// Per-system result of a batched relax+purify solve.
pub struct RelaxDiag {
    /// ‖X−D‖_F of the last launched step — the projected residual force
    /// times α (→ 0 at the fixed point).
    pub steps: Vec<f32>,
    /// Retraction work ‖D'−X‖_F of the last launched step.
    pub corrs: Vec<f32>,
    /// Tr(D) after the last launched step.
    pub traces: Vec<f32>,
    /// Idempotency defect of the last pre-combination square.
    pub errs: Vec<f32>,
    /// Stationarity certificate ‖HD−DH‖_F/(2‖HD‖_F) at the last
    /// comm_gate launch (empty if convergence never looked apparent).
    pub comms: Vec<f32>,
    /// Steps actually launched (≤ max_iter).
    pub iters: usize,
    /// All systems reported done AND passed the comm_gate certificate.
    pub converged: bool,
}

/// Batched relax+purify solve: H̃ (orthogonal basis) → density D.
///
/// `d_io`/`lam_io` are persistent state: `warm=false` cold-starts D from
/// the Palser–Gershgorin guess and Λ from 0; `warm=true` reuses both
/// (after a converged solve Λ ≈ H, so a nearby H' starts from
/// F = H'−Λ ≈ ΔH — the occupied-subspace rotation that bare TC2 cannot
/// express). Kernels:
///   `spec_span_batched`  — once per solve: Δε → α = η/Δε
///   `relax_step_batched` — per step, fully in-place on D and Λ
///   (`tc2_init_batched`, optional `tc2_step_batched` preconditioning)
///
/// Knobs (RUST_DFTB_RELAX_*): NPUR (1|2 retractions, default 2), ETA
/// (α scale, default 1.0), BETA (Λ rate, default 0.5), CHUNK (host
/// check granularity, default 4), PRE_TC2 (TC2 preconditioning steps on
/// the cold path, default 0), DEBUG (per-chunk diag dump).
pub fn relax_purify_batched(
    rt: &mut GpuRuntime,
    h_buf: &Buffer<f32>,
    d_io: &Buffer<f32>,
    lam_io: &Buffer<f32>,
    n: usize,
    batch: usize,
    nocc: &[f32],
    warm: bool,
    max_iter: usize,
    tol: f32,
) -> Result<RelaxDiag> {
    if nocc.len() != batch {
        return Err(DftbError::InvalidInput(format!(
            "relax_purify_batched: nocc len {} != batch {batch}",
            nocc.len()
        )));
    }
    if batch == 0 {
        return Ok(RelaxDiag {
            steps: vec![],
            corrs: vec![],
            traces: vec![],
            errs: vec![],
            comms: vec![],
            iters: 0,
            converged: true,
        });
    }
    let npur: usize = std::env::var("RUST_DFTB_RELAX_NPUR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    if !(npur == 1 || npur == 2) {
        return Err(DftbError::InvalidInput(format!(
            "RUST_DFTB_RELAX_NPUR={npur}: expected 1|2"
        )));
    }
    let eta: f32 = std::env::var("RUST_DFTB_RELAX_ETA")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    // β = learned-Λ rate. Default 0 (stateless F = H−aI−bD): the
    // deposit channel only measures the NORMAL part of F, so carried /
    // frame-dragged Λ tangent content is a phantom force that shifts
    // the fixed point (measured: comm≈0.3 biased warm solves, eig(ΔH)
    // drift with Λ≈H_old). β>0 kept for reference experiments only.
    let beta: f32 = std::env::var("RUST_DFTB_RELAX_BETA")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let chunk: usize = std::env::var("RUST_DFTB_RELAX_CHUNK")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(PURIFY_CHUNK);
    let pre_tc2: usize = std::env::var("RUST_DFTB_RELAX_PRE_TC2")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let dbg = std::env::var_os("RUST_DFTB_RELAX_DEBUG").is_some();

    // The relax kernel is self-contained regtile-sq (the fusion needs
    // the As_t slab structure); the shape comes from the same
    // regtile_shape() as the TC2 path.
    let sh = regtile_shape(n).ok_or_else(|| {
        DftbError::InvalidInput(format!(
            "relax regtile shape infeasible for n={n} (thread grid > 1024)"
        ))
    })?;
    let wg = sh.0 * sh.1;
    let source = render_purify_source(wg, 0, 2, Some(sh))
        .replace("#define PURIFY_NPUR 2", &format!("#define PURIFY_NPUR {npur}"));
    let program = rt.build_program(&source)?;

    let nocc_b = rt.buffer_from_slice(nocc)?;
    let spans = rt.zero_buffer::<f32>(batch)?;
    let diag = rt.zero_buffer::<f32>(batch * 4)?;
    let done = rt.zero_buffer::<i32>(batch)?;
    // Stationarity certificate: T = H·D (naive general GEMM — once per
    // apparent convergence only) + comm_gate ‖T−Tᵀ‖/2‖T‖. The kernel
    // `done` flags alone accept spurious fixed points (Λ can zero the
    // force at a wrong projector; with β=0 a stalled state is less
    // likely but the certificate is cheap insurance — one launch each).
    let t_buf = rt.zero_buffer::<f32>(batch * n * n)?;
    let rh_buf = rt.zero_buffer::<f32>(batch)?;

    let k_span = Kernel::builder()
        .program(&program)
        .name("spec_span_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(h_buf)
        .arg(&spans)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;
    let k_relax = Kernel::builder()
        .program(&program)
        .name("relax_step_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(d_io)
        .arg(h_buf)
        .arg(lam_io)
        .arg(&nocc_b)
        .arg(&spans)
        .arg(&diag)
        .arg(&done)
        .arg(eta)
        .arg(beta)
        .arg(tol)
        .arg(0.0f32) // ldecay — updated per chunk below
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;
    let k_gemm = Kernel::builder()
        .program(&program)
        .name("gemm_nn_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(h_buf)
        .arg(d_io)
        .arg(&t_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;
    let k_comm = Kernel::builder()
        .program(&program)
        .name("comm_gate_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(&t_buf)
        .arg(&rh_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .build()
        .map_err(map_ocl_err)?;

    if !warm {
        // Cold: Palser–Gershgorin seed D0 = (λmax I − H)/span; Λ starts
        // at 0 (nothing learned yet). Optional TC2 preconditioning pulls
        // the seed toward the projector manifold before relaxing.
        let trc = rt.zero_buffer::<f32>(batch)?;
        let k_init = Kernel::builder()
            .program(&program)
            .name("tc2_init_batched")
            .queue(rt.queue().clone())
            .global_work_size(batch * wg)
            .local_work_size(wg)
            .arg(h_buf)
            .arg(d_io)
            .arg(&trc)
            .arg(n as i32)
            .arg(batch as i32)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { k_init.enq().map_err(map_ocl_err)?; }
        lam_io.cmd().fill(0.0f32, None).enq().map_err(map_ocl_err)?;
        if pre_tc2 > 0 {
            let d_b = rt.zero_buffer::<f32>(batch * n * n)?;
            let errs_s = rt.zero_buffer::<f32>(batch)?;
            let done_s = rt.zero_buffer::<i32>(batch)?;
            let k_step = Kernel::builder()
                .program(&program)
                .name("tc2_step_batched")
                .queue(rt.queue().clone())
                .global_work_size(batch * wg)
                .local_work_size(wg)
                .arg(d_io)
                .arg(&d_b)
                .arg(&nocc_b)
                .arg(&trc)
                .arg(&errs_s)
                .arg(&done_s)
                .arg(tol)
                .arg(n as i32)
                .arg(batch as i32)
                .build()
                .map_err(map_ocl_err)?;
            let mut swapped = false;
            for _ in 0..pre_tc2 {
                if swapped {
                    k_step.set_arg(0u32, &d_b).map_err(map_ocl_err)?;
                    k_step.set_arg(1u32, d_io).map_err(map_ocl_err)?;
                } else {
                    k_step.set_arg(0u32, d_io).map_err(map_ocl_err)?;
                    k_step.set_arg(1u32, &d_b).map_err(map_ocl_err)?;
                }
                unsafe { k_step.enq().map_err(map_ocl_err)?; }
                swapped = !swapped;
            }
            if swapped {
                // last write went to scratch — fold back into d_io
                rt.copy_into(&d_b, d_io, batch * n * n)?;
            }
        }
    } else {
        // Warm solve: what to do with the learned Λ from the previous
        // solve (RUST_DFTB_RELAX_WARM_LAM, default "zero"):
        //   "zero"  — Λ=0: F0 = H, plain D-warm-start. CORRECT fixed
        //             point by construction (no stale tangent content).
        //   "keep"  — Λ = H₁−aI−b·D₁ as learned: converges fast but the
        //             b·D_old term is a fixed tangent matrix w.r.t. the
        //             rotating D → biased fixed point (comm ≈ b‖ΔD‖).
        //   "canon" — fold the aI+bD gauge into Λ once (Λ≈H₁, F0≈ΔH):
        //             increments are blockdiag-only for a clean D, so
        //             Λ cannot shed the tangent content — drifts.
        // Measured (n=86, Δ=0.02, batch 16): keep → comm≈0.31 wrong
        // projector; canon → comm≈7 slow drift; zero → correct.
        match std::env::var("RUST_DFTB_RELAX_WARM_LAM")
            .unwrap_or_else(|_| "zero".into())
            .as_str()
        {
            "keep" => {}
            "canon" => {
                let k_canon = Kernel::builder()
                    .program(&program)
                    .name("relax_canon_batched")
                    .queue(rt.queue().clone())
                    .global_work_size(batch * wg)
                    .local_work_size(wg)
                    .arg(d_io)
                    .arg(h_buf)
                    .arg(lam_io)
                    .arg(n as i32)
                    .arg(batch as i32)
                    .build()
                    .map_err(map_ocl_err)?;
                unsafe { k_canon.enq().map_err(map_ocl_err)?; }
            }
            "zero" => {
                lam_io.cmd().fill(0.0f32, None).enq().map_err(map_ocl_err)?;
            }
            other => {
                return Err(DftbError::InvalidInput(format!(
                    "RUST_DFTB_RELAX_WARM_LAM={other}: expected zero|keep|canon"
                )));
            }
        }
    }

    unsafe { k_span.enq().map_err(map_ocl_err)?; }
    done.cmd().fill(0i32, None).enq().map_err(map_ocl_err)?;

    // Deposit gate: Λ increments β(Y−X)/α deposit block-diagonal
    // content in the *current* D frame. While the residual is large D
    // is still rotating fast, and those deposits become permanent stale
    // *tangent* bias (increments can only ever add blockdiag content —
    // measured: warm fixed points commuted with H−Λ_tan, not H, bias
    // ≈ Λ_tan ≈ 0.3). Gate β by the residual: β_eff = β·min(1, θ/step)
    // — no deposits during the violent approach, full Λ learning once
    // the frame is nearly final → unbiased fixed point.
    // RUST_DFTB_RELAX_BETA_GATE = θ/tol (default 100); LDECAY_* kept
    // as an alternative knob (default off — it leaves a residual floor).
    let beta_gate: f32 = std::env::var("RUST_DFTB_RELAX_BETA_GATE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let decay_k: f32 = std::env::var("RUST_DFTB_RELAX_LDECAY_K")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let decay_cap: f32 = std::env::var("RUST_DFTB_RELAX_LDECAY_CAP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let theta = beta_gate * tol;
    let mut ldecay = 0.0f32;
    let mut beta_eff = beta;
    // Relative comm_gate threshold: ‖HD−DH‖/(2‖HD‖) — fp32 floor is
    // ~1e-6 at n=86; a biased fixed point sits at ~0.02. Default 1e-3.
    let comm_tol: f32 = std::env::var("RUST_DFTB_RELAX_COMM_TOL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);

    let mut iters = 0usize;
    let mut converged = false;
    let mut done_h = vec![0i32; batch];
    let mut diag_h = vec![0.0f32; batch * 4];
    let mut rh_h = vec![0.0f32; batch];
    while iters < max_iter && !converged {
        k_relax.set_arg(8u32, beta_eff).map_err(map_ocl_err)?;
        if ldecay != 0.0f32 {
            k_relax.set_arg(10u32, ldecay).map_err(map_ocl_err)?;
        }
        for _ in 0..chunk.min(max_iter - iters) {
            unsafe { k_relax.enq().map_err(map_ocl_err)?; }
            iters += 1;
        }
        rt.read_buffer(&done, &mut done_h)?;
        rt.read_buffer(&diag, &mut diag_h)?;
        let stepmax = (0..batch).map(|s| diag_h[4 * s]).fold(0.0f32, f32::max);
        if beta_gate > 0.0 {
            beta_eff = beta * (theta / stepmax.max(theta)).min(1.0).max(0.0);
        }
        ldecay = (decay_k * stepmax).min(decay_cap);
        if dbg {
            let s0 = &diag_h[0..4];
            eprintln!(
                "[relax-dbg] it={iters} sys0: step={:.3e} corr={:.3e} tr={:.4} er={:.3e} | worst step={stepmax:.3e} β_eff={beta_eff:.3} ldecay={ldecay:.3} done={}/{}",
                s0[0], s0[1], s0[2], s0[3],
                done_h.iter().filter(|&&d| d != 0).count(),
                batch
            );
        }
        if done_h.iter().all(|&d| d != 0) {
            // Apparent convergence — certify physical stationarity
            // ‖HD−DH‖/2‖HD‖ (the kernel done flag accepts spurious
            // projectors; measured biased fixed point comm≈0.3).
            // Failures are unfrozen and the loop continues → fail loud
            // at max_iter if a spurious fixed point is truly sticky.
            unsafe {
                k_gemm.enq().map_err(map_ocl_err)?;
                k_comm.enq().map_err(map_ocl_err)?;
            }
            rt.read_buffer(&rh_buf, &mut rh_h)?;
            let mut all_ok = true;
            for s in 0..batch {
                if rh_h[s] > comm_tol {
                    done_h[s] = 0;
                    all_ok = false;
                }
            }
            if all_ok {
                converged = true;
            } else {
                done.write(done_h.as_slice()).enq().map_err(map_ocl_err)?;
                if dbg {
                    eprintln!(
                        "[relax-dbg] it={iters} comm_gate reject: max_rh={:.3e} (tol={comm_tol:.1e}) — unfrozen, continuing",
                        rh_h.iter().cloned().fold(0.0f32, f32::max)
                    );
                }
            }
        }
    }

    Ok(RelaxDiag {
        steps: (0..batch).map(|s| diag_h[4 * s]).collect(),
        corrs: (0..batch).map(|s| diag_h[4 * s + 1]).collect(),
        traces: (0..batch).map(|s| diag_h[4 * s + 2]).collect(),
        errs: (0..batch).map(|s| diag_h[4 * s + 3]).collect(),
        comms: rh_h,
        iters,
        converged,
    })
}
