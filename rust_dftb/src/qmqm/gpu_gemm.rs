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
    /// `tri` (implies `sq`): compute only upper-triangle tiles and
    /// mirror-write — needs a full-coverage tile like `sq`. `nfix` > 0
    /// compiles a fixed matrix size (unrolled loops, constant masks).
    RegTile { tx: usize, ty: usize, rtx: usize, rty: usize, tk: usize, split_m: usize, split_n: usize, sq: bool, tri: bool, nfix: usize },
    /// Whole A resident in `__local` (n²·4 B), B staged in `tk` rows.
    /// `wg` threads stride the output with register accumulators.
    FullA { wg: usize, tk: usize },
    /// Iterated symmetric square D ← scl·D², `niter` rounds in ONE
    /// launch. `resident`: D in __local (LN²·4B, no staging/global
    /// traffic) vs global B↔C ping-pong. `red`: 0 none, 1 two fold-
    /// halve reduces, 2 merged float2, 3 merged 2-level — prices the
    /// fused-step reduce overhead. Requires tx·rtx == ty·rty ≥ n.
    /// `tri` → gemm_sq_iter_tri: compact SYRK, resident only — the 66
    /// upper 8×8 block-tiles split across 8/rtx threads each, so
    /// tx·ty must be 66·8/rtx (rtx∈{2,4}, rty=8, LN=88). `ku`: k-loop
    /// `#pragma unroll` factor (1 = off). `vec`: explicit floatN
    /// accumulators (rtx∈{2,4,8}) instead of scalar arrays.
    SqIter { tx: usize, ty: usize, rtx: usize, rty: usize, tk: usize, niter: usize, resident: bool, red: usize, tri: bool, ku: usize, vec: bool },
}

impl GemmVariant {
    pub fn label(&self) -> String {
        match *self {
            GemmVariant::OneElem { wg } => format!("1elem/wg{wg}"),
            GemmVariant::RegTile { tx, ty, rtx, rty, tk, split_m, split_n, sq, tri, nfix } => {
                format!(
                    "reg{tx}x{ty}/r{rty}x{rtx}/tk{tk}/s{split_m}x{split_n}{}{}{}",
                    if sq { "/sq" } else { "" },
                    if tri { "/tri" } else { "" },
                    if nfix > 0 { format!("/nf{nfix}") } else { String::new() },
                )
            }
            GemmVariant::FullA { wg, tk } => format!("fulla/wg{wg}/tk{tk}"),
            GemmVariant::SqIter { tx, ty, rtx, rty, tk, niter, resident, red, tri, ku, vec } => {
                format!(
                    "sqit{tx}x{ty}/r{rty}x{rtx}/tk{tk}/{}/n{niter}/red{red}{}{}",
                    if tri { "tri" } else if resident { "loc" } else { "glob" },
                    if ku > 1 { format!("/ku{ku}") } else { String::new() },
                    if vec { "/vec" } else { "" },
                )
            }
        }
    }

    /// Local-work-size (1-D equivalent total threads).
    pub fn wg_size(&self) -> usize {
        match *self {
            GemmVariant::OneElem { wg } | GemmVariant::FullA { wg, .. } => wg,
            GemmVariant::RegTile { tx, ty, .. } | GemmVariant::SqIter { tx, ty, .. } => tx * ty,
        }
    }
}

/// Render the kernel source with this variant's compile-time params.
fn render_source(v: &GemmVariant) -> String {
    let (tx, ty, rtx, rty, tk, sm, sn, sq, tri, nfix, res, red, ku, vec, ftk, fmax) = match *v {
        GemmVariant::RegTile { tx, ty, rtx, rty, tk, split_m, split_n, sq, tri, nfix } => {
            (tx, ty, rtx, rty, tk, split_m, split_n, (sq || tri) as usize, tri as usize, nfix, 0, 0, 1, 0, 8, 116)
        }
        GemmVariant::SqIter { tx, ty, rtx, rty, tk, resident, red, ku, vec, .. } => {
            (tx, ty, rtx, rty, tk, 1, 1, 1, 0, 0, resident as usize, red, ku, vec as usize, 8, 116)
        }
        GemmVariant::FullA { wg, tk } => {
            // FULLA_MAXELEM must cover ceil(n²/wg) — computed by caller's n;
            // 116 covers n=86@wg64 / n=128@wg144. Conservative default.
            (8, 8, 8, 8, 16, 1, 1, 0, 0, 0, 0, 0, 1, 0, tk, 128usize.max((86 * 86) / wg + 2))
        }
        GemmVariant::OneElem { .. } => (8, 8, 8, 8, 16, 1, 1, 0, 0, 0, 0, 0, 1, 0, 8, 116),
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
        .replace("#define GEMM_TRI 0", &format!("#define GEMM_TRI {tri}"))
        .replace("#define GEMM_NFIX 0", &format!("#define GEMM_NFIX {nfix}"))
        .replace("#define GEMM_ITER_RESIDENT 0", &format!("#define GEMM_ITER_RESIDENT {res}"))
        .replace("#define GEMM_ITER_RED 0", &format!("#define GEMM_ITER_RED {red}"))
        .replace("#define GEMM_KU 1", &format!("#define GEMM_KU {ku}"))
        .replace("#define GEMM_VEC 0", &format!("#define GEMM_VEC {vec}"))
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
        GemmVariant::RegTile { tx, ty, rtx, rty, tk, sq, tri, nfix, split_m, split_n } => {
            let wm = ty * rty;
            let wn = tx * rtx;
            if (sq || tri) && (wm < n || wn < n) {
                return Err(DftbError::InvalidInput(format!(
                    "gemm sq/tri variant needs a full-coverage tile (WM={wm}, WN={wn} ≥ n={n}) — \
                     the As_t reuse trick only holds for a single m0=n0=0 tile"
                )));
            }
            if (sq || tri) && wn > wm {
                return Err(DftbError::InvalidInput(format!(
                    "gemm sq/tri variant needs WN ≤ WM (WN={wn} > WM={wm}) — \
                     the B fragment is read from As_t[kk][0..WN), which is only WM wide"
                )));
            }
            if tri && (split_m != 1 || split_n != 1) {
                return Err(DftbError::InvalidInput(format!(
                    "gemm tri variant needs split 1x1 (got {split_m}x{split_n})"
                )));
            }
            if nfix > 0 && nfix != n {
                return Err(DftbError::InvalidInput(format!(
                    "gemm nfix={nfix} but launched at n={n}"
                )));
            }
            kb.name("gemm_regtile")
                .arg_local::<f32>(tk * wm) // As_t[kk][row]
                .arg_local::<f32>(if sq || tri { 1 } else { tk * wn }); // Bs unused in sq/tri
        }
        GemmVariant::FullA { tk, .. } => {
            kb.name("gemm_fulla")
                .arg_local::<f32>(n * n) // whole A
                .arg_local::<f32>(tk * n); // B k-tile
        }
        GemmVariant::SqIter { .. } => {
            return Err(DftbError::InvalidInput(
                "SqIter uses sq_iter_kernel() (extra niter/scl/errs/trs args)".into(),
            ));
        }
    }
    kb.build().map_err(map_ocl_err)
}

/// Build the iterated symmetric-square kernel (`gemm_sq_iter`,
/// GemmVariant::SqIter): D ← scl·D² for `niter` rounds in one launch.
/// `b`/`c` are the global ping-pong pair (result lands in `c`); `errs`,
/// `trs` receive the last iteration's ‖T−D‖/Tr when red>0 (pass any
/// batch-sized buffer otherwise — unused).
pub fn sq_iter_kernel(
    rt: &mut GpuRuntime,
    v: &GemmVariant,
    scl: f32,
    n: usize,
    batch: usize,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &Buffer<f32>,
    errs: &Buffer<f32>,
    trs: &Buffer<f32>,
) -> Result<Kernel> {
    let &GemmVariant::SqIter { tx, ty, rtx, rty, tk: _, niter, resident: _, red: _, tri, ku: _, vec } = v else {
        return Err(DftbError::InvalidInput("sq_iter_kernel needs a SqIter variant".into()));
    };
    let wg = tx * ty;
    let ln = ty * rty;
    if wg == 0 || wg > 1024 {
        return Err(DftbError::InvalidInput(format!("sq_iter wg={wg} out of range")));
    }
    if vec && !(rtx == 2 || rtx == 4 || rtx == 8) {
        return Err(DftbError::InvalidInput(format!("sq_iter vec needs rtx∈{{2,4,8}} (got {rtx})")));
    }
    if tri {
        // compact SYRK: 66 upper 8×8 tiles × (8/rtx) threads, rty=8 rows.
        let want = 66 * (8 / rtx);
        if rty != 8 || !(rtx == 2 || rtx == 4) || wg != want || ln < n {
            return Err(DftbError::InvalidInput(format!(
                "sq_iter tri needs rty=8, rtx∈{{2,4}}, wg={want} (got {wg}), LN={ln}≥n={n}"
            )));
        }
    } else if tx * rtx != ln || ln < n {
        // sq staging indexes As_t/Ls by BOTH fragments — requires a square
        // full-coverage tile (WN == WM == LN ≥ n).
        return Err(DftbError::InvalidInput(format!(
            "sq_iter needs square full-coverage tile (LN=TY·RTY={ln}, TX·RTX={}, n={n})",
            tx * rtx
        )));
    }
    let program = rt.build_program(&render_source(v))?;
    Kernel::builder()
        .program(&program)
        .queue(rt.queue().clone())
        .name(if tri { "gemm_sq_iter_tri" } else { "gemm_sq_iter" })
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(niter as i32)
        .arg(scl)
        .arg(a)
        .arg(b)
        .arg(c)
        .arg(errs)
        .arg(trs)
        .build()
        .map_err(map_ocl_err)
}
