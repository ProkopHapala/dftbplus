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
        .replace(
            "#define MAX_SWEEPS 20",
            &format!("#define MAX_SWEEPS {}", MAX_SWEEPS),
        )
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

    let wids = rt.buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())?; // T06 identity
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
        .arg(&wids)
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
const TILED_MAX_SWEEPS: usize = 100;

/// Rotlog capacity in sweeps for `jacobi_resident_batched` deferred-V —
/// a solve exceeding it flush-replays mid-solve and keeps going. Chosen
/// so the batch-400 n=86 log region (~47 MB) stays L2-sized: measured
/// cold-solve regression at 8 (≈200 KB/sys → log reads miss L2 during
/// the end replay). Must match the kernel-side JACOBI_LOG_SWEEPS
/// injected by render_resident_source.
pub const RESIDENT_LOG_SWEEPS: usize = 4;

/// `RUST_DFTB_JACOBI_SWEEPS` env override (diagnostic), else `default`.
/// Shared so the production direct kernel (MAX_CSWEEPS) and the deprecated
/// tiled path (MAX_SWEEPS) honor the same knob — R5.
pub fn jacobi_sweeps(default: usize) -> usize {
    std::env::var("RUST_DFTB_JACOBI_SWEEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Render the tiled Jacobi OpenCL template with block-size specialization.
/// `prec` = JACOBI_PREC (0=FP32-FMA, 1=FP64 c/s only, 2=broad FP64 reference).
/// Public for the event-timed 3-mode benchmark in tests/gpu_tiled_jacobi.rs.
pub fn render_tiled_source(b: usize, wg: usize, prec: u32) -> String {
    render_tiled_source_cfg(b, wg, prec, false)
}

/// Full render: `no_tail` compiles the Fermi tail (and its ~9 KB/WG local
/// scratch) out via JACOBI_NO_TAIL — valid only when the caller uses the
/// standalone fermi_occ path; the kernel marks stop=5 if the tail is
/// requested anyway (fail-loud host misconfig check).
pub fn render_tiled_source_cfg(b: usize, wg: usize, prec: u32, no_tail: bool) -> String {
    let pb = 2 * b;
    let pld = pb + 1;
    let src = GPU_TILED_JACOBI_TEMPLATE
        .replace("#define B 32", &format!("#define B {}", b))
        .replace("#define PB 64", &format!("#define PB {}", pb))
        .replace("#define PLD 65", &format!("#define PLD {}", pld))
        .replace("#define WG 256", &format!("#define WG {}", wg))
        .replace("#define STRIP_R 32", &format!("#define STRIP_R {}", b))
        .replace("#define MAX_SWEEPS 50", &format!("#define MAX_SWEEPS {}", jacobi_sweeps(TILED_MAX_SWEEPS)))
        .replace("#define MAX_CSWEEPS 40", &format!("#define MAX_CSWEEPS {}", jacobi_sweeps(40)))
        .replace("#define JACOBI_LOG_SWEEPS 8", &format!("#define JACOBI_LOG_SWEEPS {}", RESIDENT_LOG_SWEEPS))
        .replace("#define JACOBI_PREC 2", &format!("#define JACOBI_PREC {}", prec))
        // R8b: env-tunable off-norm exit threshold for sweep/tolerance sweeps.
        .replace("#ifndef JACOBI_OFF_TOL\n#define JACOBI_OFF_TOL 1.0e-6f    // off/‖A‖_F exit threshold\n#endif",
                 &format!("#define JACOBI_OFF_TOL {:.3e}f",
                     std::env::var("RUST_DFTB_JACOBI_OFF_TOL").ok()
                         .and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0e-6)));
    if no_tail {
        format!("#define JACOBI_NO_TAIL 1\n{src}")
    } else {
        src
    }
}

/// Direct-kernel workgroup size. `RUST_DFTB_JACOBI_WG` overrides for sweeps
/// (fail-loud on unparseable/over-limit); default min(512, device max).
/// Arbitrary sizes are valid — the kernel's reductions handle non-PoT lsz
/// (jac_sum_* fold-then-halve), so WG≈N (96/128/160) is legal now.
pub fn direct_jacobi_wg(max_wg: usize) -> usize {
    let default = max_wg.min(512).max(64);
    let wg = std::env::var("RUST_DFTB_JACOBI_WG")
        .ok()
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("RUST_DFTB_JACOBI_WG={v}: not a usize"))
        })
        .unwrap_or(default);
    assert!(
        wg >= 32 && wg <= max_wg,
        "RUST_DFTB_JACOBI_WG={wg}: outside device range 32..={max_wg}"
    );
    wg
}

/// Resident-Jacobi render (T08b): `jacobi_resident_batched` — same template
/// file, same helpers and diag/tail contract as the direct kernel.
/// `resident_v` selects RESIDENT_V=1 (A+V both __local — ~2n(n+1)×4B/WG,
/// 1 WG/SM at n≈86) vs RESIDENT_V=0 (A __local + deferred V: rotations
/// logged to a global rotlog and applied once per sweep — ~n(n+1)×4B/WG,
/// 2–3 WGs/SM). `no_tail` compiles the Fermi tail out (its ~9 KB scratch
/// decides whether 3 WGs/SM fit at WG512).
pub fn render_resident_source(wg: usize, prec: u32, resident_v: bool, no_tail: bool) -> String {
    let src = render_tiled_source_cfg(32, wg, prec, no_tail);
    if resident_v {
        format!("#define RESIDENT_V 1\n{src}")
    } else {
        src
    }
}

/// Standalone resident-Jacobi benchmark/diagnostic — same contract as
/// `direct_jacobi_batched` (all active, init_v=0, returns diag
/// [batch][4] = {off, off/‖A‖_F, stop, sweeps}). Allocates the rotlog
/// scratch itself; the production plan keeps it persistent.
pub fn resident_jacobi_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
    prec: u32,
    resident_v: bool,
) -> Result<Vec<f32>> {
    if n <= 64 || n > 256 {
        return Err(DftbError::InvalidInput(format!(
            "resident_jacobi_batched: n={n} out of range 65..=256 (n≤64 → jacobi_cyclic_local_batched)"
        )));
    }
    if batch == 0 {
        return Ok(Vec::new());
    }
    let la = n * (n + 1) / 2 * 4; // packed symmetric lA
    let lv = if resident_v { n * (n + 1) * 4 } else { 4 };
    let local_need = (la + lv) as u64;
    let local_cap = rt.caps().local_mem_size;
    if local_need > local_cap {
        return Err(DftbError::InvalidInput(format!(
            "resident_jacobi_batched: n={n} resident_v={resident_v} needs {local_need} B __local > device local_mem_size {local_cap} B"
        )));
    }
    let wg = direct_jacobi_wg(rt.caps().max_work_group_size);
    let source = render_resident_source(wg, prec, resident_v, false);
    let program = rt.build_program(&source)?;
    let ones = rt.buffer_from_slice(&vec![1i32; batch])?;
    let diag = rt.zero_buffer::<f32>(4 * batch)?;
    let jn = if n & 1 == 1 { n + 1 } else { n };
    // jlog2_t entry: double2 (4 f32) at prec>=1, float2 (2 f32) at prec=0;
    // the log holds RESIDENT_LOG_SWEEPS sweeps — V replays once at solve end.
    let log_len = if resident_v {
        1
    } else {
        batch * RESIDENT_LOG_SWEEPS * (jn - 1) * (jn / 2) * if prec >= 1 { 4 } else { 2 }
    };
    let rotlog = rt.zero_buffer::<f32>(log_len)?;
    let occ_w = rt.zero_buffer::<f32>(batch * n)?;
    let mu = rt.zero_buffer::<f32>(batch)?;
    let wids = rt.buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())?; // T06 identity
    let kernel = Kernel::builder()
        .program(&program)
        .name("jacobi_resident_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(a_buf)
        .arg(v_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(0i32)
        .arg(&ones)
        .arg(&diag)
        .arg(0i32)
        .arg(0i32)
        .arg(0.0f32)
        .arg(&occ_w)
        .arg(&mu)
        .arg(&rotlog)
        .arg_local::<f32>(n * (n + 1) / 2)
        .arg_local::<f32>(if resident_v { n * (n + 1) } else { 1 })
        .arg(&wids)
        .build()
        .map_err(map_ocl_err)?;
    unsafe {
        kernel.enq().map_err(map_ocl_err)?;
    }
    let mut d = vec![0.0f32; 4 * batch];
    rt.read_buffer(&diag, &mut d)?;
    Ok(d)
}

/// Production direct cyclic Jacobi for 64 < N ≤ 256 — the same
/// `jacobi_cyclic_global_batched` kernel GpuSccPlan runs (global-memory A/V,
/// round-robin schedule, one WG per system). All systems are active.
/// Returns the per-system diagnostic record `[batch][4]` =
/// {off-norm, off/‖A‖_F, stop, sweeps}; stop: 0 converged · 1 stagnation ·
/// 2 MAX_CSWEEPS · 3 n>capacity · 4 non-finite. `prec` = JACOBI_PREC.
pub fn direct_jacobi_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
    prec: u32,
) -> Result<Vec<f32>> {
    if n <= 64 || n > 256 {
        return Err(DftbError::InvalidInput(format!(
            "direct_jacobi_batched: n={n} out of range 65..=256 (n≤64 → jacobi_cyclic_local_batched; n>256 exceeds __local rot[128] capacity)"
        )));
    }
    if batch == 0 {
        return Ok(Vec::new());
    }
    // W3: WG=512 ~1.55× faster than 256 on RTX 3090 (direct_jacobi_bench);
    // T08: arbitrary WG is now safe (fold-then-halve reductions) — env
    // override via RUST_DFTB_JACOBI_WG for the WG≈N sweep.
    let wg = direct_jacobi_wg(rt.caps().max_work_group_size);
    let source = render_tiled_source(32, wg, prec);
    let program = rt.build_program(&source)?;
    let ones = rt.buffer_from_slice(&vec![1i32; batch])?;
    let diag = rt.zero_buffer::<f32>(4 * batch)?;
    // R5 tail args — disabled (fermi_tail=0); occ_w/mu are bound dummies.
    let occ_w = rt.zero_buffer::<f32>(batch * n)?;
    let mu = rt.zero_buffer::<f32>(batch)?;
    let wids = rt.buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())?; // T06 identity
    let kernel = Kernel::builder()
        .program(&program)
        .name("jacobi_cyclic_global_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(a_buf)
        .arg(v_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(0i32)
        .arg(&ones)
        .arg(&diag)
        .arg(0i32)
        .arg(0i32)
        .arg(0.0f32)
        .arg(&occ_w)
        .arg(&mu)
        .arg(&wids)
        .build()
        .map_err(map_ocl_err)?;
    unsafe {
        kernel.enq().map_err(map_ocl_err)?;
    }
    let mut d = vec![0.0f32; 4 * batch];
    rt.read_buffer(&diag, &mut d)?;
    Ok(d)
}

// ------------------------------------------------------------------
// Block Jacobi (manifest §16.D) — one WG/system, one thread per row,
// pivot+U in local memory only. New architecture alongside the direct
// kernel — the old path stays default; this is selected by the caller.
// ------------------------------------------------------------------

const GPU_BLOCK_JACOBI_TEMPLATE: &str = include_str!("gpu_block_jacobi.cl");

/// Workgroup for `block_jacobi_1wg`: one thread per row, rounded to a
/// warp — ceil(n/32)·32 (n=86→96, n=246→256). The kernel assumes WG ≥ n.
pub fn block_jacobi_wg(n: usize) -> usize {
    ((n.max(1) + 31) / 32 * 32).max(32)
}

/// Render the block-Jacobi template. `b` = block size (pivot is 2B×2B in
/// local memory: B=16 → ~8.4 KB/WG).
pub fn render_block_source(b: usize, wg: usize) -> String {
    render_block_source_cfg(b, wg, 12, 1.0e-7)
}

/// Full render with inner-pivot tuning: `inner_max` = Brent–Luk sweep cap
/// per pivot, `inner_tol` = pivot off-norm exit relative to pivot ‖·‖_F.
pub fn render_block_source_cfg(b: usize, wg: usize, inner_max: usize, inner_tol: f32) -> String {
    let pb = 2 * b;
    GPU_BLOCK_JACOBI_TEMPLATE
        .replace("#define B 16", &format!("#define B {b}"))
        .replace("#define PB 32", &format!("#define PB {pb}"))
        .replace("#define PLD 33", &format!("#define PLD {}", pb + 1))
        .replace("#define WG 96", &format!("#define WG {wg}"))
        .replace(
            "#define MAX_SWEEPS 40",
            &format!("#define MAX_SWEEPS {}", jacobi_sweeps(40)),
        )
        .replace(
            "#define INNER_MAX 12",
            &format!("#define INNER_MAX {inner_max}"),
        )
        .replace(
            "#define INNER_TOL 1.0e-7f",
            &format!("#define INNER_TOL {inner_tol:e}f"),
        )
}

/// Block-Jacobi tuning knobs — env overrides for parameter sweeps:
///   RUST_DFTB_BJ_B     block size B (pivot is 2B×2B in local; default 32)
///   RUST_DFTB_BJ_IMAX  inner Brent–Luk sweep cap per pivot (default 1)
///   RUST_DFTB_BJ_ITOL  inner pivot off-norm exit, relative (default 1e-6)
/// Fail-loud on unparseable values; B limited to 8..=32 (B=32 → PB=64,
/// ~34 KB local per WG — larger B exceeds typical device local memory).
/// Defaults are the measured T08 optimum (block_jacobi_param_sweep, batch
/// 400, equal accuracy, bad=0): IMAX=1 is the dominant lever — converging
/// each pivot to 1e-7 was waste since outer sweeps revisit them — and
/// B=32 minimises serial pivots/sweep (n=246: 120→28). B16/IMAX12/ITOL1e-7
/// = the pre-T08 baseline; envs restore it for A/B.
pub fn block_jacobi_cfg() -> (usize, usize, f32) {
    let b = std::env::var("RUST_DFTB_BJ_B")
        .ok()
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("RUST_DFTB_BJ_B={v}: not a usize"))
        })
        .unwrap_or(32);
    let imax = std::env::var("RUST_DFTB_BJ_IMAX")
        .ok()
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("RUST_DFTB_BJ_IMAX={v}: not a usize"))
        })
        .unwrap_or(1);
    let itol = std::env::var("RUST_DFTB_BJ_ITOL")
        .ok()
        .map(|v| {
            v.parse::<f32>()
                .unwrap_or_else(|_| panic!("RUST_DFTB_BJ_ITOL={v}: not an f32"))
        })
        .unwrap_or(1.0e-6);
    assert!(
        (8..=32).contains(&b),
        "RUST_DFTB_BJ_B={b}: B must be in 8..=32 (PB=2B local pivot)"
    );
    assert!(imax >= 1, "RUST_DFTB_BJ_IMAX={imax}: must be >= 1");
    assert!(
        itol.is_finite() && itol > 0.0 && itol < 1.0,
        "RUST_DFTB_BJ_ITOL={itol}: must be in (0,1)"
    );
    (b, imax, itol)
}

/// T08b n>64 eigensolver kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EigKind {
    /// jacobi_cyclic_global_batched — A/V streamed through global per round.
    Direct,
    /// jacobi_resident_batched RESIDENT_V=0 — A in __local, rotations logged
    /// and applied to global V once per sweep. ~2.3–2.7× direct at N=86
    /// (resident_jacobi_sweep, batch=400, identical res/orth/parity).
    ResidentDefV,
    /// jacobi_resident_batched RESIDENT_V=1 — A+V both __local. Largest
    /// footprint (~2n(n+1)×4B/WG); on 48 KB-local devices only n≤~74 fits.
    ResidentAV,
    /// block_jacobi_1wg — pivot+U local, strips global. For n>128.
    Block,
}

/// Headroom reserved for the resident kernel's static __local scratch
/// (rot arrays ~1.5 KB + reduce/red2 ≤4 KB + Fermi tail ~9 KB at WG512)
/// when deciding whether lA (=n(n+1)·4B dynamic local) fits the device.
const RESIDENT_SCRATCH_HEADROOM: u64 = 16 * 1024;

/// Whether A can be __local-resident on this device: dynamic lA + static
/// scratch must fit local_mem_size. Fails loud via assert when an explicit
/// RUST_DFTB_EIGSOLVER=resident* selection doesn't fit; in auto mode it
/// simply picks a non-resident kind (a capacity-based algorithm choice at
/// plan build, not a runtime fallback).
fn resident_fits(n: usize, resident_v: bool, local_cap: u64) -> bool {
    // lA is packed-symmetric (n(n+1)/2 floats); lV stays square.
    let la = (n * (n + 1) / 2 * 4) as u64;
    let lv = if resident_v {
        (n * (n + 1) * 4) as u64
    } else {
        0
    };
    la + lv + RESIDENT_SCRATCH_HEADROOM <= local_cap
}

/// T08/T08b measured solver dispatch: `RUST_DFTB_EIGSOLVER` ∈
/// {auto,direct,block,resident,resident_av} (unset/auto → size+local rule).
/// Measured at batch=400: N=86 — res-defV 11.4/20.0 ms vs direct 21.9/45.7 ms
/// (one/cold, ~2.2×; resident_jacobi_sweep); end-to-end GC 2.19 vs 4.01
/// ms/iter (1.84×). N=246 — block 71.4 ms/iter (resident can't fit: A alone
/// is 242 KB). Auto: n≤128 → resident when lA+16KB ≤ local_mem, else direct;
/// n>128 → block. Fail-loud on unknown values / unfit explicit selection.
pub fn eigsolver_kind(n: usize, local_cap: u64) -> EigKind {
    match std::env::var("RUST_DFTB_EIGSOLVER").ok().as_deref() {
        None | Some("auto") => {
            if n > 128 {
                EigKind::Block
            } else if resident_fits(n, false, local_cap) {
                EigKind::ResidentDefV
            } else {
                EigKind::Direct
            }
        }
        Some("block") => EigKind::Block,
        Some("direct") => EigKind::Direct,
        Some("resident") => {
            assert!(resident_fits(n, false, local_cap),
                "RUST_DFTB_EIGSOLVER=resident: n={n} needs {} B __local + scratch > device local_mem_size {local_cap} B",
                n * (n + 1) / 2 * 4);
            EigKind::ResidentDefV
        }
        Some("resident_av") => {
            assert!(resident_fits(n, true, local_cap),
                "RUST_DFTB_EIGSOLVER=resident_av: n={n} needs {} B __local + scratch > device local_mem_size {local_cap} B",
                n * (n + 1) / 2 * 4 + n * (n + 1) * 4);
            EigKind::ResidentAV
        }
        Some(v) => {
            panic!("RUST_DFTB_EIGSOLVER={v}: expected auto|direct|block|resident|resident_av")
        }
    }
}

/// Legacy predicate kept for callers that only distinguish block-vs-not.
/// Equivalent to `eigsolver_kind(n, u64::MAX) == Block` — the block solver
/// is only ever selected for n>128, where residency cannot fit anyway.
pub fn use_block_jacobi(n: usize) -> bool {
    eigsolver_kind(n, u64::MAX) == EigKind::Block
}

/// Bound-handle builder for `jacobi_resident_batched` — same prebound-handle
/// pattern and tail-arg indices (7..11) as the direct kernel, so the plan's
/// `bind_solve_params`/`check_jacobi` contracts carry over unchanged.
/// `rotlog` is the deferred-V scratch ([batch][jround·jpair] jlog2_t —
/// double2 at prec>=1, float2 at prec=0); pass any valid buffer when
/// `resident_v` (unused by the kernel then).
pub fn build_resident_jacobi_kernel(
    rt: &mut GpuRuntime,
    n: usize,
    batch: usize,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    act_buf: &Buffer<i32>,
    prec: u32,
    diag: &Buffer<f32>,
    init_v: i32,
    occ_w: &Buffer<f32>,
    mu: &Buffer<f32>,
    rotlog: &Buffer<f32>,
    resident_v: bool,
    no_tail: bool,
    work_ids: &Buffer<i32>,
) -> Result<Kernel> {
    if n <= 64 || n > 256 {
        return Err(DftbError::InvalidInput(format!(
            "jacobi_resident_batched: n={n} out of range 65..=256"
        )));
    }
    if !resident_fits(n, resident_v, rt.caps().local_mem_size) {
        return Err(DftbError::InvalidInput(format!(
            "jacobi_resident_batched: n={n} resident_v={resident_v} needs {} B __local + scratch > device local_mem_size {} B",
            n * (n + 1) / 2 * 4 + if resident_v { n * (n + 1) * 4 } else { 0 }, rt.caps().local_mem_size
        )));
    }
    let wg = direct_jacobi_wg(rt.caps().max_work_group_size);
    let source = render_resident_source(wg, prec, resident_v, no_tail);
    let program = rt.build_program(&source)?;
    Kernel::builder()
        .program(&program)
        .name("jacobi_resident_batched")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(a_buf)
        .arg(v_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(init_v)
        .arg(act_buf)
        .arg(diag)
        .arg(0i32)
        .arg(0i32)
        .arg(0.0f32)
        .arg(occ_w)
        .arg(mu)
        .arg(rotlog)
        .arg_local::<f32>(n * (n + 1) / 2)
        .arg_local::<f32>(if resident_v { n * (n + 1) } else { 1 })
        .arg(work_ids)
        .build()
        .map_err(map_ocl_err)
}

/// Bound-handle builder for `block_jacobi_1wg` — same prebound-handle
/// pattern as `build_jacobi_kernel` but the signature has no Fermi tail;
/// instead `eig` (= the plan's eig_diag buffer) receives the eigenvalues,
/// replacing the extract_diag launch. Occupation then runs the existing
/// standalone `fermi_occ_batched`.
pub fn build_block_jacobi_kernel(
    rt: &mut GpuRuntime,
    n: usize,
    batch: usize,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    act_buf: &Buffer<i32>,
    diag: &Buffer<f32>,
    eig: &Buffer<f32>,
    init_v: i32,
    work_ids: &Buffer<i32>,
) -> Result<Kernel> {
    if n == 0 || n > 256 {
        return Err(DftbError::InvalidInput(format!(
            "block_jacobi_1wg: n={n} out of range 1..=256"
        )));
    }
    let wg = block_jacobi_wg(n);
    if wg > rt.caps().max_work_group_size {
        return Err(DftbError::InvalidInput(format!(
            "block_jacobi_1wg: n={n} needs WG={wg} > device max_work_group_size {}",
            rt.caps().max_work_group_size
        )));
    }
    let (b, imax, itol) = block_jacobi_cfg();
    // n ≤ PB = the whole matrix IS the pivot: the inner solve must fully
    // converge (it IS the solve — no outer revisit rescues it). Keep a real
    // cap there; approximate pivots (IMAX=1) are only valid when n > PB.
    let imax = if n <= 2 * b { imax.max(12) } else { imax };
    let source = render_block_source_cfg(b, wg, imax, itol);
    let program = rt.build_program(&source)?;
    Kernel::builder()
        .program(&program)
        .name("block_jacobi_1wg")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(a_buf)
        .arg(v_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(init_v)
        .arg(act_buf)
        .arg(diag)
        .arg(eig)
        .arg(work_ids)
        .build()
        .map_err(map_ocl_err)
}

/// Standalone block-Jacobi benchmark/diagnostic — same contract as
/// `direct_jacobi_batched` (all active, init_v=0, returns diag
/// [batch][4] = {off, off/‖A‖_F, stop, sweeps}).
pub fn block_jacobi_batched(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
) -> Result<Vec<f32>> {
    if n == 0 || batch == 0 {
        return Ok(Vec::new());
    }
    if n > 256 {
        return Err(DftbError::InvalidInput(format!(
            "block_jacobi_batched: n={n} exceeds capacity 256"
        )));
    }
    let wg = block_jacobi_wg(n);
    let source = render_block_source(16, wg);
    let program = rt.build_program(&source)?;
    let ones = rt.buffer_from_slice(&vec![1i32; batch])?;
    let diag = rt.zero_buffer::<f32>(4 * batch)?;
    let eig = rt.zero_buffer::<f32>(batch * n)?;
    let wids = rt.buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())?; // T06 identity
    let kernel = Kernel::builder()
        .program(&program)
        .name("block_jacobi_1wg")
        .queue(rt.queue().clone())
        .global_work_size(batch * wg)
        .local_work_size(wg)
        .arg(a_buf)
        .arg(v_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(0i32)
        .arg(&ones)
        .arg(&diag)
        .arg(&eig)
        .arg(&wids)
        .build()
        .map_err(map_ocl_err)?;
    unsafe {
        kernel.enq().map_err(map_ocl_err)?;
    }
    let mut d = vec![0.0f32; 4 * batch];
    rt.read_buffer(&diag, &mut d)?;
    Ok(d)
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
    tiled_jacobi_batched_prec(rt, a_buf, v_buf, n, batch, 2)
}

/// `prec` = JACOBI_PREC: 0 pure FP32-FMA, 1 FP64 rotation params only,
/// 2 broad FP64 (accuracy reference).
pub fn tiled_jacobi_batched_prec(
    rt: &mut GpuRuntime,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
    prec: u32,
) -> Result<()> {
    if n == 0 || batch == 0 {
        return Ok(());
    }
    if n <= 64 {
        return Err(DftbError::InvalidInput(format!(
            "tiled_jacobi_batched: n={n} <= 64, use jacobi_cyclic_local_batched instead"
        )));
    }
    let b = 32usize; // block size
    let wg = 256usize; // workgroup size
    let source = render_tiled_source(b, wg, prec);
    let program = rt.build_program(&source)?;
    let wids = rt.buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())?; // T06 identity
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
        .arg(&wids)
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

    let wids = rt.buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())?; // T06 identity
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
        .arg(&wids)
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
    let source = render_source(64); // reuse the template (LAMBDA_FLOOR define)
    let program = rt.build_program(&source)?;
    let wids = rt.buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())?; // T06 identity
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
        .arg(&wids)
        .build()
        .map_err(map_ocl_err)?;
    unsafe {
        kernel.enq().map_err(map_ocl_err)?;
    }

    // Step 3: X = V_scaled · V^T (tiled GEMM, transpose B).
    let x_buf = rt.zero_buffer::<f32>(n * n * batch)?;
    // C = V_scaled · V^T → trans_a=false, trans_b=true
    // Use matmul_tiled_batched_params for the transpose.
    matmul_tiled_batched_params(
        rt, &v_scaled, &v_buf, &x_buf, n, batch, false, true, 1.0, 0.0,
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
    use ocl::{
        enums::{DeviceInfo, KernelWorkGroupInfo, ProfilingInfo},
        flags, Event, Queue,
    };
    use std::time::Instant;

    fn fixture(n: usize, kind: usize) -> DMatrix<f64> {
        let v = DVector::from_fn(n, |i, _| ((i + 1) as f64).sin());
        let q = DMatrix::identity(n, n) - (&v * v.transpose()) * (2.0 / v.norm_squared());
        let d = DMatrix::from_diagonal(&DVector::from_fn(n, |i, _| {
            if kind == 1 {
                (i / 2) as f64 * 0.2 - 1.0
            } else {
                i as f64 * 0.2 - 1.0
            }
        }));
        let a = if kind == 2 { d } else { &q * d * q.transpose() };
        DMatrix::from_fn(n, n, |i, j| a[(i.min(j), i.max(j))] as f32 as f64)
    }

    fn check_variants(configs: &[(usize, usize)], repeats: usize) {
        assert!(repeats % 2 == 1, "Jacobi comparison needs odd repeats so the final readback is the old variant; repeats={repeats}");
        let mut rt =
            GpuRuntime::new().expect("Jacobi verification requires a working OpenCL device");
        let name = rt
            .device()
            .name()
            .expect("query Jacobi verification device name");
        let vendor = rt
            .device()
            .vendor()
            .expect("query Jacobi verification device vendor");
        assert!(vendor.contains("NVIDIA"), "Jacobi performance verification requires NVIDIA, selected {vendor}: {name}; select OCL_DEFAULT_PLATFORM_IDX from clinfo -l");
        eprintln!(
            "Jacobi verification: {name}, local={:?}, max_wg={:?}, repeats={repeats}",
            rt.device()
                .info(DeviceInfo::LocalMemSize)
                .expect("query device local memory"),
            rt.device()
                .info(DeviceInfo::MaxWorkGroupSize)
                .expect("query device WG limit")
        );
        let queue = Queue::new(
            rt.context(),
            *rt.device(),
            Some(flags::CommandQueueProperties::new().profiling()),
        )
        .expect("create Jacobi profiling queue");
        for &(n, batch) in configs {
            let (_, _, _, _, wg) = spec_params(n);
            let fixtures: Vec<DMatrix<f64>> = (0..3).map(|kind| fixture(n, kind)).collect();
            let references: Vec<Vec<f64>> = fixtures
                .iter()
                .map(|a| {
                    let mut vals = SymmetricEigen::new(a.clone())
                        .eigenvalues
                        .as_slice()
                        .to_vec();
                    vals.sort_by(f64::total_cmp);
                    vals
                })
                .collect();
            let mut input = vec![0.0f32; batch * n * n];
            for b in 0..batch {
                for i in 0..n {
                    for j in 0..n {
                        input[b * n * n + i * n + j] = fixtures[b % 3][(i, j)] as f32;
                    }
                }
            }
            let src = rt
                .buffer_from_slice(&input)
                .expect("upload Jacobi fixtures");
            let a = rt
                .zero_buffer::<f32>(input.len())
                .expect("allocate Jacobi A");
            let v = rt
                .zero_buffer::<f32>(input.len())
                .expect("allocate Jacobi V");
            let mut ah = vec![0.0f32; input.len()];
            let mut vh = vec![0.0f32; input.len()];
            let mut block_a = vec![0.0f32; input.len()];
            let mut block_v = vec![0.0f32; input.len()];
            let mut accuracy_ok = true;
            let wids = rt
                .buffer_from_slice(&(0..batch as i32).collect::<Vec<_>>())
                .expect("T06 identity work_ids");
            let kernels: Vec<Kernel> = (0..2).map(|mode| {
                let source = format!("#define JACOBI_BLOCK_UPDATE {mode}\n{}", render_source(n));
                let program = rt.build_program(&source).expect("compile Jacobi comparison variant");
                let kernel = Kernel::builder().program(&program).name("jacobi_cyclic_local_batched").queue(queue.clone())
                    .global_work_size(batch * wg).local_work_size(wg).arg(&a).arg(&v).arg(n as i32).arg(batch as i32).arg(&wids)
                    .build().expect("build persistent Jacobi comparison kernel");
                eprintln!("  N={n} batch={batch} mode={mode} WG={wg} local={:?} private={:?} kernel_max_wg={:?}", kernel.wg_info(*rt.device(), KernelWorkGroupInfo::LocalMemSize).expect("query kernel local memory"), kernel.wg_info(*rt.device(), KernelWorkGroupInfo::PrivateMemSize).expect("query kernel private memory"), kernel.wg_info(*rt.device(), KernelWorkGroupInfo::WorkGroupSize).expect("query kernel WG limit"));
                kernel
            }).collect();
            rt.finish()
                .expect("finish fixture uploads before using profiling queue");
            let mut times = [vec![0.0f64; repeats], vec![0.0f64; repeats]];
            let mut walls = [vec![0.0f64; repeats], vec![0.0f64; repeats]];
            for sample in 0..=repeats {
                for turn in 0..2 {
                    let mode = (sample + turn) % 2;
                    src.cmd()
                        .queue(&queue)
                        .copy(&a, None, None)
                        .enq()
                        .expect("reset Jacobi input by device copy");
                    queue
                        .finish()
                        .expect("finish Jacobi input reset outside timing");
                    let mut event = Event::empty();
                    let start = Instant::now();
                    unsafe {
                        kernels[mode]
                            .cmd()
                            .enew(&mut event)
                            .enq()
                            .expect("enqueue Jacobi comparison");
                    }
                    event.wait_for().expect("wait for Jacobi comparison event");
                    let wall = start.elapsed().as_secs_f64() * 1e6;
                    let t0 = event
                        .profiling_info(ProfilingInfo::Start)
                        .expect("Jacobi event START")
                        .time()
                        .expect("START timestamp");
                    let t1 = event
                        .profiling_info(ProfilingInfo::End)
                        .expect("Jacobi event END")
                        .time()
                        .expect("END timestamp");
                    assert!(t1 > t0, "invalid Jacobi timestamps: N={n} batch={batch} mode={mode} start={t0} end={t1}");
                    if sample > 0 {
                        times[mode][sample - 1] = (t1 - t0) as f64 * 1e-3;
                        walls[mode][sample - 1] = wall;
                    }
                    if sample != repeats {
                        continue;
                    }
                    a.cmd()
                        .queue(&queue)
                        .read(&mut ah)
                        .enq()
                        .expect("read Jacobi A for parity");
                    v.cmd()
                        .queue(&queue)
                        .read(&mut vh)
                        .enq()
                        .expect("read Jacobi V for parity");
                    assert!(
                        ah.iter().chain(&vh).all(|x| x.is_finite()),
                        "non-finite Jacobi output N={n} batch={batch} mode={mode}"
                    );
                    let mut worst = [0.0f64; 4];
                    for b in 0..batch {
                        let vals = DVector::from_fn(n, |i, _| ah[b * n * n + i * n + i] as f64);
                        let eig = DMatrix::from_fn(n, n, |i, j| vh[b * n * n + i * n + j] as f64);
                        let d = DMatrix::from_diagonal(&vals);
                        let orig = &fixtures[b % 3];
                        let mut sorted = vals.as_slice().to_vec();
                        sorted.sort_by(f64::total_cmp);
                        let errors = [
                            sorted
                                .iter()
                                .zip(&references[b % 3])
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0, f64::max),
                            (orig * &eig - &eig * &d).amax(),
                            (&eig * &d * eig.transpose() - orig).amax(),
                            (eig.transpose() * &eig - DMatrix::identity(n, n)).amax(),
                        ];
                        if batch <= 3 {
                            eprintln!("    N={n} mode={mode} system={b} spectrum/residual/reconstruction/orthogonality={errors:?}");
                        }
                        for i in 0..4 {
                            worst[i] = worst[i].max(errors[i]);
                        }
                        if !errors.iter().all(|e| e.is_finite() && *e < 1e-4) {
                            eprintln!("Jacobi accuracy violation N={n} batch={batch} mode={mode} system={b}, errors={errors:?}, tolerance=1e-4");
                            accuracy_ok = false;
                        }
                    }
                    if mode == 1 {
                        block_a.copy_from_slice(&ah);
                        block_v.copy_from_slice(&vh);
                    }
                    eprintln!("  N={n} batch={batch} mode={mode} worst_errors={worst:?}");
                }
            }
            let differences = ah
                .iter()
                .chain(&vh)
                .zip(block_a.iter().chain(&block_v))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!("  N={n} batch={batch} old/block bitwise-different outputs={differences}");
            assert_eq!(
                differences, 0,
                "Jacobi block transform differs from original on NVIDIA: N={n} batch={batch}"
            );
            for mode in 0..2 {
                times[mode].sort_by(f64::total_cmp);
                walls[mode].sort_by(f64::total_cmp);
            }
            eprintln!("Jacobi N={n} batch={batch}: event_us old={:.3} block={:.3} speedup={:.3}; enqueue+wait_us old={:.3} block={:.3}", times[0][repeats/2], times[1][repeats/2], times[0][repeats/2]/times[1][repeats/2], walls[0][repeats/2], walls[1][repeats/2]);
            assert!(accuracy_ok, "Jacobi absolute accuracy failed N={n} batch={batch}; per-system old/block diagnostics above, tolerance=1e-4 (unchanged)");
        }
        eprintln!("Jacobi verification finished");
    }

    #[test]
    #[ignore]
    fn jacobi_sweep_diagnostic() {
        let mut rt = GpuRuntime::new().expect("Jacobi sweep diagnostic requires OpenCL");
        eprintln!(
            "Jacobi sweep diagnostic on {}",
            rt.device().name().expect("query device name")
        );
        let n = 64;
        let orig = fixture(n, 0);
        let input: Vec<f32> = (0..n * n).map(|ij| orig[(ij / n, ij % n)] as f32).collect();
        let a = rt
            .buffer_from_slice(&input)
            .expect("upload Jacobi diagnostic matrix");
        let v = rt
            .zero_buffer::<f32>(input.len())
            .expect("allocate Jacobi diagnostic eigenvectors");
        let mut ah = vec![0.0f32; input.len()];
        let mut vh = vec![0.0f32; input.len()];
        let (_, _, _, _, wg) = spec_params(n);
        for (rsqrt_mode, sweeps) in
            (0..3).flat_map(|mode| [1, 2, 3, 4, 5, 8, 12, 20].map(|sweeps| (mode, sweeps)))
        {
            let mut source = format!(
                "#define JACOBI_NORMALIZE_ROTATION {}\n#define MAX_SWEEPS {sweeps}\n{}",
                usize::from(rsqrt_mode == 2),
                render_source(n)
            )
            .replace(
                "gA[i] = (r == c) ? lA[r * JLD + c] : 0.0f;",
                "gA[i] = lA[r * JLD + c];",
            );
            if rsqrt_mode == 1 {
                source = source.replace(
                    "float c = 1.0f / sqrt(1.0f + t * t);",
                    "float c = rsqrt(1.0f + t * t);",
                );
            }
            let program = rt
                .build_program(&source)
                .expect("compile Jacobi sweep diagnostic");
            let wids = rt
                .buffer_from_slice(&[0i32])
                .expect("T06 identity work_ids");
            let kernel = Kernel::builder()
                .program(&program)
                .name("jacobi_cyclic_local_batched")
                .queue(rt.queue().clone())
                .global_work_size(wg)
                .local_work_size(wg)
                .arg(&a)
                .arg(&v)
                .arg(n as i32)
                .arg(1i32)
                .arg(&wids)
                .build()
                .expect("build Jacobi sweep diagnostic");
            a.write(&input)
                .enq()
                .expect("reset Jacobi diagnostic input");
            unsafe {
                kernel.enq().expect("enqueue Jacobi sweep diagnostic");
            }
            rt.read_buffer(&a, &mut ah)
                .expect("read full transformed Jacobi matrix");
            rt.read_buffer(&v, &mut vh)
                .expect("read diagnostic eigenvectors");
            assert!(
                ah.iter().chain(&vh).all(|x| x.is_finite()),
                "Jacobi diagnostic nonfinite at sweeps={sweeps}"
            );
            let b = DMatrix::from_fn(n, n, |i, j| ah[i * n + j] as f64);
            let eig = DMatrix::from_fn(n, n, |i, j| vh[i * n + j] as f64);
            let d = DMatrix::from_diagonal(&b.diagonal());
            let gram = eig.transpose() * &eig;
            eprintln!("rsqrt_mode={rsqrt_mode} sweeps={sweeps}: offnorm={:.9e} AV-VD={:.9e} AV-VB={:.9e} reconstruction={:.9e} full_reconstruction={:.9e} orthogonality={:.9e} trace_drift={:.9e} frobenius_drift={:.9e} column_norm2_min={:.9e} max={:.9e}",
                (&b-&d).norm(), (&orig*&eig-&eig*&d).amax(), (&orig*&eig-&eig*&b).amax(), (&eig*&d*eig.transpose()-&orig).amax(), (&eig*&b*eig.transpose()-&orig).amax(),
                (&gram-DMatrix::identity(n,n)).amax(), b.trace()-orig.trace(), b.norm()-orig.norm(), gram.diagonal().min(), gram.diagonal().max());
        }
    }

    #[test]
    fn jacobi_block_parity() {
        check_variants(
            &[
                (1, 3),
                (2, 3),
                (3, 3),
                (7, 3),
                (8, 3),
                (14, 3),
                (27, 3),
                (28, 3),
                (32, 3),
                (63, 3),
                (64, 3),
            ],
            1,
        );
    }

    #[test]
    #[ignore]
    fn jacobi_block_benchmark() {
        check_variants(
            &[
                (8, 100),
                (28, 1),
                (28, 100),
                (28, 1000),
                (32, 100),
                (64, 100),
            ],
            7,
        );
    }
}
