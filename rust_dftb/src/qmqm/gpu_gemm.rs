//! Batched dense GEMM variants for the FOE/purify path — standalone
//! microbenchmark machinery (`gpu_gemm.cl`). The question being
//! answered: which kernel structure gives the highest TFLOPS for
//! C[s] = A[s]·B[s] on many small matrices (n≈86, batch≈400), one
//! workgroup per system, small `__local` staging so multiple WGs stay
//! resident per CU.
//!
//! The tile-per-WG baseline lives in `gpu_matrix.rs::batched_gemm`
//! (16×16 output tile per WG over a 3-D grid); it is benchmarked
//! alongside these variants in `tests/gpu_gemm.rs`.
//!
//! This module is bench infrastructure — production integration comes
//! after the variant sweep (manifest §17.5).

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::{Buffer, Kernel};

const GPU_GEMM_TEMPLATE: &str = include_str!("gpu_gemm.cl");

/// Which kernel structure to launch.
#[derive(Clone, Copy, Debug)]
pub enum GemmVariant {
    /// One thread per output element, global row·col dot. The floor.
    OneElem { wg: usize },
    /// Register-tiled schoolbook: `tx`×`ty` thread grid, `rty`×`rtx`
    /// register micro-tile per thread, `tk`-deep local k-staging.
    /// (8,8,8,8,16,_,_) = user-spec 64-thr/8×8-tile; (16,16,1,1,16,_,_)
    /// = classic 1-elem-per-thread tiled. `split_m`×`split_n` workgroups
    /// per system, each taking interleaved output tiles — the occupancy
    /// lever: 400 systems × small WGs starve the SM thread slots.
    RegTile { tx: usize, ty: usize, rtx: usize, rty: usize, tk: usize, split_m: usize, split_n: usize, sq: bool },
    /// Whole A resident in `__local` (n²·4 B), B staged in `tk` rows.
    /// `wg` threads stride the output with register accumulators.
    FullA { wg: usize, tk: usize },
}

impl GemmVariant {
    pub fn label(&self) -> String {
        match *self {
            GemmVariant::OneElem { wg } => format!("1elem/wg{wg}"),
            GemmVariant::RegTile { tx, ty, rtx, rty, tk, split_m, split_n, sq } => {
                format!("reg{tx}x{ty}/r{rty}x{rtx}/tk{tk}/s{split_m}x{split_n}{}", if sq { "/sq" } else { "" })
            }
            GemmVariant::FullA { wg, tk } => format!("fulla/wg{wg}/tk{tk}"),
        }
    }

    /// Local-work-size (1-D equivalent total threads).
    pub fn wg_size(&self) -> usize {
        match *self {
            GemmVariant::OneElem { wg } | GemmVariant::FullA { wg, .. } => wg,
            GemmVariant::RegTile { tx, ty, .. } => tx * ty,
        }
    }
}

/// Render the kernel source with this variant's compile-time params.
fn render_source(v: &GemmVariant) -> String {
    let (tx, ty, rtx, rty, tk, sm, sn, sq, ftk, fmax) = match *v {
        GemmVariant::RegTile { tx, ty, rtx, rty, tk, split_m, split_n, sq } => {
            (tx, ty, rtx, rty, tk, split_m, split_n, sq as usize, 8, 116)
        }
        GemmVariant::FullA { wg, tk } => {
            // FULLA_MAXELEM must cover ceil(n²/wg) — computed by caller's n;
            // 116 covers n=86@wg64 / n=128@wg144. Conservative default.
            (8, 8, 8, 8, 16, 1, 1, 0, tk, 128usize.max((86 * 86) / wg + 2))
        }
        GemmVariant::OneElem { .. } => (8, 8, 8, 8, 16, 1, 1, 0, 8, 116),
    };
    GPU_GEMM_TEMPLATE
        .replace("#define GEMM_TX 8", &format!("#define GEMM_TX {tx}"))
        .replace("#define GEMM_TY 8", &format!("#define GEMM_TY {ty}"))
        .replace("#define GEMM_RTX 8", &format!("#define GEMM_RTX {rtx}"))
        .replace("#define GEMM_RTY 8", &format!("#define GEMM_RTY {rty}"))
        .replace("#define GEMM_TK 16", &format!("#define GEMM_TK {tk}"))
        .replace("#define GEMM_SPLIT_M 1", &format!("#define GEMM_SPLIT_M {sm}"))
        .replace("#define GEMM_SPLIT_N 1", &format!("#define GEMM_SPLIT_N {sn}"))
        .replace("#define GEMM_SQ 0", &format!("#define GEMM_SQ {sq}"))
        .replace("#define FULLA_TK 8", &format!("#define FULLA_TK {ftk}"))
        .replace("#define FULLA_MAXELEM 116", &format!("#define FULLA_MAXELEM {fmax}"))
}

/// Build a ready-to-enqueue kernel for `variant`. Buffers stay bound;
/// repeated `enq()` re-runs the same GEMM (bench loop).
pub fn gemm_kernel(
    rt: &mut GpuRuntime,
    variant: &GemmVariant,
    n: usize,
    batch: usize,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &Buffer<f32>,
) -> Result<Kernel> {
    let wg = variant.wg_size();
    if wg == 0 || wg > 1024 {
        return Err(DftbError::InvalidInput(format!(
            "gemm variant {}: wg={wg} out of range",
            variant.label()
        )));
    }
    let split = match *variant {
        GemmVariant::RegTile { split_m, split_n, .. } => split_m * split_n,
        _ => 1,
    };
    let source = render_source(variant);
    let program = rt.build_program(&source)?;
    let mut kb = Kernel::builder();
    kb.program(&program)
        .queue(rt.queue().clone())
        .global_work_size(batch * split * wg)
        .local_work_size(wg)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(a)
        .arg(b)
        .arg(c);
    match *variant {
        GemmVariant::OneElem { .. } => {
            kb.name("gemm_1elem");
        }
        GemmVariant::RegTile { tx, ty, rtx, rty, tk, sq, .. } => {
            let wm = ty * rty;
            let wn = tx * rtx;
            if sq && (wm < n || wn < n) {
                return Err(DftbError::InvalidInput(format!(
                    "gemm sq variant needs a full-coverage tile (WM={wm}, WN={wn} ≥ n={n}) — \
                     the As_t reuse trick only holds for a single m0=n0=0 tile"
                )));
            }
            kb.name("gemm_regtile")
                .arg_local::<f32>(tk * wm) // As_t[kk][row]
                .arg_local::<f32>(if sq { 1 } else { tk * wn }); // Bs unused in sq
        }
        GemmVariant::FullA { tk, .. } => {
            kb.name("gemm_fulla")
                .arg_local::<f32>(n * n) // whole A
                .arg_local::<f32>(tk * n); // B k-tile
        }
    }
    kb.build().map_err(map_ocl_err)
}
