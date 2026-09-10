//! GPU wrapper for the BSR4 sparse purification kernels
//! (`sparse_bsr4_purification.cl`).
//!
//! This module is self-contained: it owns its compiled `Program` (built from
//! the shared `GpuRuntime`'s context/device so no second OpenCL context is
//! created) and exposes both low-level buffer-level kernel launches and
//! high-level `Bsr4Matrix`-in / `Bsr4Matrix`-out convenience methods.
//!
//! It does **not** depend on any `qmqm` solver logic — only on the shared
//! `GpuRuntime` OpenCL runtime.

use crate::core::error::{DftbError, Result};
use crate::methods::sparse::bsr4::{
    dense_max_abs_diff, Bsr4Matrix, SpgemmPlan, BS, BS2,
};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::{builders::ProgramBuilder, flags, Buffer, Kernel, Program};
use std::sync::Arc;

const BSR4_KERNEL_SOURCE: &str = include_str!("sparse_bsr4_purification.cl");

/// Occupation contract for a claimed TC2 projector (second review §3.5).
/// `||KSK−K||` alone accepts a wrong-rank projector, including K=0.
pub const TC2_TRACE_TOL: f32 = 5e-2;

/// Performance statistics for a sparse SCC/purification run.
///
/// Every performance run should end with `print_execution_audit()` to make
/// hidden host costs visible and prevent the recurring failure of optimizing
/// the kernel while ignoring the harness bottleneck (P0, manifest v3 §4.1).
///
/// Fields follow the manifest v3 `SparsePerfStats` struct specification.
/// The key distinction from the old struct:
/// - `n_orb_physical` vs `n_orb_padded` — tracks the H-padding overhead;
/// - `plan_terms` / `plan_bytes` — symbolic SpGEMM plan statistics (P4);
/// - `host_syncs` — counts queue synchronizations, not just reads (a 4-byte
///   read can cost far more than its bandwidth suggests);
/// - `t_mask_plan` — geometric mask + symbolic plan construction time;
/// - `t_kinit` — K₀ initialization (Z·H·Z + spectral bounds);
/// - `t_scc` — complete SCC loop (gamma + Hscc + TC2 + charge mixing);
/// - `t_host_sync` — time spent in host synchronization (finish/read).
#[derive(Debug, Clone, Default)]
pub struct SparsePerfStats {
    // ── Problem size ──
    pub n_atom: usize,
    pub n_orb_physical: usize, // sum of physical orbitals (1 for H, 4 for Si/C)
    pub n_orb_padded: usize,   // n_atom * 4 (BSR4 padded)

    // ── Sparsity ──
    pub nnz_hs: usize,    // H/S block count (physical SK mask + skin)
    pub nnz_k: usize,     // K/density-kernel block count
    pub nnz_z: usize,     // Z/S⁻¹ block count
    pub plan_terms: usize, // symbolic SpGEMM contribution count (P4)
    pub plan_bytes: usize, // bytes in the symbolic plan (P4)
    pub gpu_bytes_peak: usize, // peak GPU memory allocated

    // ── Host/device traffic ──
    pub kernel_launches: usize,
    pub host_syncs: usize,      // queue finish + blocking read count
    pub host_read_bytes: usize, // total bytes read back to host
    /// Largest dense host allocation observed in this run (bytes). 0 means
    /// uninstrumented, not a proof that no dense alloc happened (review G1.9).
    pub largest_dense: usize,

    // ── Stage timings (seconds) ──
    pub t_mask_plan: f64,  // geometric mask + symbolic plan construction
    pub t_hs: f64,         // H/S assembly
    pub t_gamma_build: f64, // gamma matrix construction (per geometry)
    pub t_gamma_mv: f64,    // gamma matvec (per SCC iteration)
    pub t_inverse: f64,    // Newton-Schulz approximate inverse
    pub t_kinit: f64,      // K₀ initialization (Z·H·Z + spectral bounds)
    pub t_tc2: f64,        // TC2 purification
    pub t_scc: f64,        // complete SCC loop (gamma + Hscc + TC2 + mixing)
    pub t_force: f64,      // force evaluation
    pub t_host_sync: f64,  // time in host synchronization (finish + read)
    pub t_total: f64,      // total wall time

    // ── Legacy aliases (for backward compatibility with existing callers) ──
    pub gpu_bytes: usize,   // alias for gpu_bytes_peak
    pub n_kernel_launch: usize, // alias for kernel_launches
    pub n_host_read: usize,     // alias for host_syncs
    pub t_ns: f64,              // alias for t_inverse
    pub t_gamma: f64,           // alias for t_gamma_mv
}

impl SparsePerfStats {
    /// Print the execution audit to stderr (fail-loud: makes hidden costs visible).
    pub fn print_audit(&self) {
        eprintln!("=== SPARSE EXECUTION AUDIT (P0, manifest v3 §4.1) ===");
        eprintln!("N atoms                 {}", self.n_atom);
        eprintln!("N orbs (physical)       {}", self.n_orb_physical);
        eprintln!("N orbs (padded BSR4)    {}", self.n_orb_padded);
        let pad_overhead = if self.n_orb_physical > 0 {
            100.0 * (self.n_orb_padded as f64 / self.n_orb_physical as f64 - 1.0)
        } else { 0.0 };
        eprintln!("H-padding overhead      {pad_overhead:.1}%");
        eprintln!("H/S blocks (nnz_hs)     {}", self.nnz_hs);
        eprintln!("K blocks (nnz_k)        {}", self.nnz_k);
        eprintln!("Z blocks (nnz_z)        {}", self.nnz_z);
        if self.plan_terms > 0 {
            eprintln!("Plan terms              {}", self.plan_terms);
            eprintln!("Plan bytes              {}", self.plan_bytes);
            let avg = self.plan_terms as f64 / self.nnz_k.max(1) as f64;
            eprintln!("Avg terms/output block  {avg:.1}");
        }
        eprintln!("GPU bytes peak          {}", self.gpu_bytes_peak);
        eprintln!("kernel launches         {}", self.kernel_launches);
        eprintln!("host syncs              {}", self.host_syncs);
        eprintln!("host read bytes         {}", self.host_read_bytes);
        // P0: print the measured counter, not a hardcoded 0 (review G1.9).
        if self.largest_dense == 0 {
            eprintln!("largest dense alloc     0 bytes (uninstrumented — not a P0 proof)");
        } else {
            eprintln!("largest dense alloc     {} bytes (must be 0 in production sparse path)", self.largest_dense);
        }

        eprintln!();
        eprintln!("--- Stage timings (seconds) ---");
        eprintln!("mask/plan   {:.6}", self.t_mask_plan);
        eprintln!("H/S assembly {:.6}", self.t_hs);
        eprintln!("gamma build  {:.6}", self.t_gamma_build);
        eprintln!("gamma matvec {:.6}", self.t_gamma_mv);
        eprintln!("inverse (NS) {:.6}", self.t_inverse);
        eprintln!("K init       {:.6}", self.t_kinit);
        eprintln!("TC2          {:.6}", self.t_tc2);
        eprintln!("SCC total    {:.6}", self.t_scc);
        eprintln!("force        {:.6}", self.t_force);
        eprintln!("host sync    {:.6}", self.t_host_sync);

        let accounted = self.t_mask_plan + self.t_hs + self.t_gamma_build
            + self.t_gamma_mv + self.t_inverse + self.t_kinit
            + self.t_tc2 + self.t_scc + self.t_force + self.t_host_sync;
        let t_misc = self.t_total - accounted;
        eprintln!("misc/other  {:.6}", t_misc);
        eprintln!("TOTAL       {:.6}", self.t_total);

        // Warn if host synchronization is a large fraction of total time.
        if self.t_host_sync > 0.3 * self.t_total && self.t_total > 0.01 {
            eprintln!("WARNING: host sync is {:.0}% of total — a 4-byte read can cost more than its bandwidth",
                100.0 * self.t_host_sync / self.t_total);
        }
        if t_misc > 0.6 * self.t_total && self.t_total > 0.01 {
            eprintln!("WARNING: misc/other is {:.0}% of total — kernel speedup may be irrelevant",
                100.0 * t_misc / self.t_total);
        }
        // P0 firewall check: host syncs should be minimal in the hot loop.
        // A typical TC2 run with check_every=5 should have ~max_iter/5 syncs.
        if self.host_syncs > 100 && self.t_total > 0.01 {
            eprintln!("WARNING: {} host syncs — consider reducing check frequency or fusing diagnostics",
                self.host_syncs);
        }
    }
}

/// Tunable build-time parameters for the BSR4 kernels. These are passed to
/// the OpenCL compiler as `-D` defines; the `.cl` file guards each with
/// `#ifndef` so unspecified values keep their defaults.
#[derive(Debug, Clone)]
pub struct SparseBsr4Config {
    /// Workgroup size for the SpGEMM kernels. Must be a multiple of 16
    /// (one 16-thread team per output 4×4 block). Default 128.
    pub wg: i32,
    /// Max blocks in a left row that fits in the local-memory cache.
    /// Local mem ≈ `MAX_LEFT_BLOCKS * (16*4 + 4)` bytes. Default 256 (~17 KiB).
    pub max_left_blocks: i32,
    /// Workgroup size for reduction kernels. Default 256.
    pub reduce_wg: i32,
}

impl Default for SparseBsr4Config {
    fn default() -> Self {
        Self {
            wg: 128,
            max_left_blocks: 256,
            reduce_wg: 256,
        }
    }
}

/// Host NS/TC2 iteration prints. Default on (G3 diagnostics). Set
/// `RUST_DFTB_SPARSE_ALGEBRA_VERBOSE=0` to silence inside FIRE loops.
pub(crate) fn algebra_verbose() -> bool {
    match std::env::var("RUST_DFTB_SPARSE_ALGEBRA_VERBOSE") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        _ => true,
    }
}

/// Compiled BSR4 sparse kernel set bound to an OpenCL device.
pub struct SparseBsr4Gpu {
    rt: GpuRuntime,
    program: Program,
    config: SparseBsr4Config,
    // Cached kernel handles — built once at init, reused for every launch.
    // This eliminates the per-call Kernel::builder() overhead.
    k_spgemm_masked: Kernel,
    k_spgemm_bsym: Kernel,
    k_zero: Kernel,
    k_axpby: Kernel,
    k_mcweeny: Kernel,
    k_tc2: Kernel,
    k_symmetrize: Kernel,
    k_mulliken_ks: Kernel,
    k_trace_partial: Kernel,
    k_reduce: Kernel,
    k_identity_residual: Kernel,
    k_idempotency: Kernel,
    // P4: symbolic SpGEMM plan kernel (Bsym variant)
    k_spgemm_plan_bsym: Kernel,
    // GPT-5.6 #9: device-side inf_norm, identity, scale for NS Z0 init
    k_row_abs_sum: Kernel,
    k_reduce_max: Kernel,
    k_build_identity: Kernel,
    k_scale: Kernel,
    // GPT-5.6 #19: device-side Gershgorin spectral bounds
    k_gershgorin_partial: Kernel,
    k_reduce_min: Kernel,
}

impl SparseBsr4Gpu {
    /// Initialize OpenCL and compile the BSR4 kernels with the given config.
    /// All kernel handles are built once and cached for the lifetime of this
    /// `SparseBsr4Gpu`. Per-launch calls use `set_arg` to swap buffers, never
    /// `Kernel::builder()`.
    pub fn new(config: SparseBsr4Config) -> Result<Self> {
        if config.wg % 16 != 0 || config.wg <= 0 {
            return Err(DftbError::InvalidInput(format!(
                "wg must be a positive multiple of 16, got {}",
                config.wg
            )));
        }
        let mut rt = GpuRuntime::new()?;
        super::harness::check_sparse_device(&rt)?;
        let device = rt.device().clone();
        let context = rt.context().clone();

        let mut builder = ProgramBuilder::new();
        builder.devices(device);
        builder.src(BSR4_KERNEL_SOURCE);
        builder.cmplr_def("WG", config.wg);
        builder.cmplr_def("MAX_LEFT_BLOCKS", config.max_left_blocks);
        builder.cmplr_def("REDUCE_WG", config.reduce_wg);
        let program = builder.build(&context).map_err(map_ocl_err)?;

        // Touch the program cache so the runtime is aware (no-op effectively,
        // but keeps the runtime's cache consistent if later reused).
        let _ = rt.build_program(BSR4_KERNEL_SOURCE);
        let _ = &mut rt; // silence unused_mut if any

        // Build dummy buffers for kernel construction. Kernels require all
        // args to be present at build time; we swap them with set_arg per
        // launch. These are 1-element buffers kept alive for the lifetime of
        // this struct.
        let queue = rt.queue().clone();
        let dummy_u32 = Buffer::<u32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(1)
            .fill_val(0u32)
            .build()
            .map_err(map_ocl_err)?;
        let dummy_f32 = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(1)
            .fill_val(0.0f32)
            .build()
            .map_err(map_ocl_err)?;

        let wg_usize = config.wg as usize;
        let reduce_wg_usize = config.reduce_wg as usize;
        let gws_spgemm = wg_usize; // will be scaled by nrow per launch
        let gws_elem = BS2; // per-block elementwise kernels
        let gws_reduce = reduce_wg_usize;

        // Helper: build a kernel with dummy args matching the kernel signature.
        // Each kernel's arg count and types must match the .cl definition.
        let build_k = |name: &str, n_scalar_u32: usize, n_buf_u32: usize, n_buf_f32: usize, n_scalar_f32: usize| -> Result<Kernel> {
            let mut b = Kernel::builder();
            b.program(&program).name(name).queue(queue.clone());
            for _ in 0..n_scalar_u32 { b.arg(0u32); }
            for _ in 0..n_buf_u32    { b.arg(&dummy_u32); }
            for _ in 0..n_buf_f32    { b.arg(&dummy_f32); }
            for _ in 0..n_scalar_f32 { b.arg(0.0f32); }
            b.build().map_err(map_ocl_err)
        };

        // bsr4_spgemm_masked: nrow, A_row,A_col,A, B_row,B_col,B, C_row,C_col,C
        //   1 u32 scalar, 3 u32 bufs, 3 f32 bufs (A,B,C), 0 f32 scalar
        // Actually: A,B,C are f32 value bufs; A_row,A_col,B_row,B_col,C_row,C_col are u32 bufs
        //   = 1 scalar_u32, 6 buf_u32, 3 buf_f32
        let k_spgemm_masked = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_spgemm_masked").queue(queue.clone());
            b.arg(0u32); // nrow
            b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_f32); // A_row, A_col, A
            b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_f32); // B_row, B_col, B
            b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_f32); // C_row, C_col, C
            b.build().map_err(map_ocl_err)?
        };
        let k_spgemm_bsym = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_spgemm_masked_Bsym").queue(queue.clone());
            b.arg(0u32);
            b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_f32);
            b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_f32);
            b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_zero: nblock, A   -> 1 u32 scalar, 1 f32 buf
        let k_zero = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_zero").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_axpby: nblock, alpha, A, beta, B, C  -> 1 u32, 2 f32 buf (A,B are f32, C is f32), 2 f32 scalar
        //   args: nblock(u32), alpha(f32), A(f32 buf), beta(f32), B(f32 buf), C(f32 buf)
        let k_axpby = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_axpby").queue(queue.clone());
            b.arg(0u32); b.arg(0.0f32); b.arg(&dummy_f32); b.arg(0.0f32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_mcweeny: nblock, Q, V, Knew  -> 1 u32, 3 f32 buf
        let k_mcweeny = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_mcweeny").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_f32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_tc2: nblock, K, Q_KSK, trace_KS, Nocc, Knew -> 1 u32, 3 f32 buf, 1 f32 scalar
        let k_tc2 = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_tc2").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_f32); b.arg(&dummy_f32); b.arg(&dummy_f32); b.arg(0.0f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_symmetrize: nblock, transpose_block, A -> 1 u32, 1 u32 buf, 1 f32 buf
        let k_symmetrize = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_symmetrize").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_u32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_mulliken_KS: nrow, diag_block, ks, q -> 1 u32, 2 u32 buf (diag_block is u32), 2 f32 buf (ks, q)
        //   Actually: diag_block is u32 buf, ks is f32 buf, q is f32 buf
        let k_mulliken_ks = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_mulliken_KS").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_u32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_trace_KS_partial: nrow, diag_block, ks, partial -> 1 u32, 1 u32 buf, 2 f32 buf
        let k_trace_partial = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_trace_KS_partial").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_u32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // reduce_sum_f32: n, input, output -> 1 u32, 2 f32 buf
        let k_reduce = {
            let mut b = Kernel::builder();
            b.program(&program).name("reduce_sum_f32").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_identity_residual_partial: nblock, diag_flag, A, partial
        let k_identity_residual = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_identity_residual_partial").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_u32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_idempotency_partial: nblock, q_ksk, k, partial -> 1 u32, 3 f32 buf
        let k_idempotency = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_idempotency_partial").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_f32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // P4: bsr4_spgemm_plan_Bsym: nrow, A_row, A, B, plan_ptr, plan_a_idx,
        //     plan_b_idx, C_row, C  (GPT-5.6 #5: removed A_col, C_col — dead code)
        //   -> 1 u32 scalar, 5 u32 bufs, 3 f32 bufs
        let k_spgemm_plan_bsym = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_spgemm_plan_Bsym").queue(queue.clone());
            b.arg(0u32); // nrow
            b.arg(&dummy_u32); b.arg(&dummy_f32); // A_row, A
            b.arg(&dummy_f32); // B
            b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_u32); // plan_ptr, plan_a_idx, plan_b_idx
            b.arg(&dummy_u32); b.arg(&dummy_f32); // C_row, C
            b.build().map_err(map_ocl_err)?
        };
        // GPT-5.6 #9: device-side inf_norm, identity, scale
        // bsr4_row_abs_sum: n_atom, row_ptr, col_idx, values, row_sums
        //   -> 1 u32 scalar, 2 u32 bufs, 2 f32 bufs
        let k_row_abs_sum = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_row_abs_sum").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // reduce_max_f32: n, x, out -> 1 u32 scalar, 2 f32 bufs
        let k_reduce_max = {
            let mut b = Kernel::builder();
            b.program(&program).name("reduce_max_f32").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_build_identity_dev: n_atom, diag_block, values
        //   -> 1 u32 scalar, 1 u32 buf, 1 f32 buf
        let k_build_identity = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_build_identity_dev").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_u32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // bsr4_scale_dev: nblock, alpha, values -> 1 u32 scalar, 1 f32 scalar, 1 f32 buf
        let k_scale = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_scale_dev").queue(queue.clone());
            b.arg(0u32); b.arg(0.0f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };
        // GPT-5.6 #19: Gershgorin bounds + min reduction
        // bsr4_gershgorin_partial: n_atom, row_ptr, col_idx, values, diag_flag,
        //   emin_partial, emax_partial -> 1 u32 scalar, 4 u32 bufs, 3 f32 bufs
        let k_gershgorin_partial = {
            let mut b = Kernel::builder();
            b.program(&program).name("bsr4_gershgorin_partial").queue(queue.clone());
            b.arg(0u32); // n_atom
            b.arg(&dummy_u32); b.arg(&dummy_u32); b.arg(&dummy_f32); // row_ptr, col_idx, values
            b.arg(&dummy_u32); // diag_flag
            b.arg(&dummy_f32); b.arg(&dummy_f32); // emin_partial, emax_partial
            b.build().map_err(map_ocl_err)?
        };
        // reduce_min_f32: n, x, out -> 1 u32 scalar, 2 f32 bufs
        let k_reduce_min = {
            let mut b = Kernel::builder();
            b.program(&program).name("reduce_min_f32").queue(queue.clone());
            b.arg(0u32); b.arg(&dummy_f32); b.arg(&dummy_f32);
            b.build().map_err(map_ocl_err)?
        };

        let _ = gws_spgemm; let _ = gws_elem; let _ = gws_reduce; let _ = build_k;

        Ok(Self {
            rt,
            program,
            config,
            k_spgemm_masked, k_spgemm_bsym, k_zero, k_axpby, k_mcweeny,
            k_tc2, k_symmetrize, k_mulliken_ks, k_trace_partial, k_reduce,
            k_identity_residual,
            k_idempotency,
            k_spgemm_plan_bsym,
            k_row_abs_sum, k_reduce_max, k_build_identity, k_scale,
            k_gershgorin_partial, k_reduce_min,
        })
    }

    /// Borrow the underlying runtime (for buffer allocation helpers).
    pub fn runtime(&self) -> &GpuRuntime {
        &self.rt
    }

    pub fn config(&self) -> &SparseBsr4Config {
        &self.config
    }

    /// Validate that no row in `row_ptr` exceeds `MAX_LEFT_BLOCKS`.
    /// The GPU SpGEMM kernels silently `return` (producing zero output) when
    /// a row's degree exceeds the local-memory cache size. This host-side
    /// check makes the failure loud and early, per AGENTS.md "Fail Fast".
    ///
    /// Call this before any SpGEMM launch with the left operand's `row_ptr`.
    fn check_row_degree(&self, row_ptr: &[u32], n_atom: usize) -> Result<()> {
        let max = self.config.max_left_blocks as u32;
        for i in 0..n_atom {
            let na = row_ptr[i + 1] - row_ptr[i];
            if na > max {
                return Err(DftbError::InvalidInput(format!(
                    "bsr4_spgemm: row {i} has {na} blocks > MAX_LEFT_BLOCKS={max}. \
                     The GPU kernel cannot cache this row in local memory. \
                     Options: (1) increase SparseBsr4Config.max_left_blocks, \
                     (2) use a sparser mask, (3) implement degree-bucketed kernels."
                )));
            }
        }
        Ok(())
    }

    /// Device-side variant: read `row_ptr` from the GPU buffer and check.
    /// Used by device-resident methods where `row_ptr` is not on the host.
    fn check_row_degree_dev(&self, row_ptr_buf: &Buffer<u32>, n_atom: usize) -> Result<()> {
        let mut row_ptr = vec![0u32; n_atom + 1];
        self.read_u32(row_ptr_buf, &mut row_ptr)?;
        self.check_row_degree(&row_ptr, n_atom)
    }

    // ------------------------------------------------------------------
    // Buffer helpers
    // ------------------------------------------------------------------

    /// Allocate a `u32` GPU buffer from a host slice.
    pub fn buf_u32(&self, data: &[u32]) -> Result<Buffer<u32>> {
        Buffer::<u32>::builder()
            .queue(self.rt.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .map_err(map_ocl_err)
    }

    /// Allocate an `f32` GPU buffer from a host slice.
    pub fn buf_f32(&self, data: &[f32]) -> Result<Buffer<f32>> {
        Buffer::<f32>::builder()
            .queue(self.rt.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .map_err(map_ocl_err)
    }

    /// Allocate a zero-filled `f32` buffer.
    pub fn zero_f32(&self, len: usize) -> Result<Buffer<f32>> {
        Buffer::<f32>::builder()
            .queue(self.rt.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .fill_val(0.0f32)
            .build()
            .map_err(map_ocl_err)
    }

    /// Read an `f32` buffer back to host (blocking).
    pub fn read_f32(&self, buf: &Buffer<f32>, out: &mut [f32]) -> Result<()> {
        self.rt.read_buffer(buf, out)
    }

    /// Read a `u32` buffer back to host (blocking).
    pub fn read_u32(&self, buf: &Buffer<u32>, out: &mut [u32]) -> Result<()> {
        self.rt.read_buffer(buf, out)
    }

    fn queue(&self) -> &ocl::Queue {
        self.rt.queue()
    }

    // ------------------------------------------------------------------
    // Low-level kernel launches
    // ------------------------------------------------------------------

    /// Generic masked block-sparse product `C = P_M(A·B)`.
    /// `C` must already be allocated with the right size; it is overwritten.
    pub fn spgemm_masked(
        &self,
        nrow: usize,
        a: &Bsr4Matrix,
        b: &Bsr4Matrix,
        c_row: &Buffer<u32>,
        c_col: &Buffer<u32>,
        c: &Buffer<f32>,
    ) -> Result<()> {
        self.check_row_degree(&a.row_ptr, nrow)?;
        let a_row = self.buf_u32(&a.row_ptr)?;
        let a_col = self.buf_u32(&a.col_idx)?;
        let a_val = self.buf_f32(&a.values)?;
        let b_row = self.buf_u32(&b.row_ptr)?;
        let b_col = self.buf_u32(&b.col_idx)?;
        let b_val = self.buf_f32(&b.values)?;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_spgemm_masked")
            .queue(self.queue().clone())
            .global_work_size(nrow * self.config.wg as usize)
            .local_work_size(self.config.wg as usize)
            .arg(nrow as u32)
            .arg(&a_row)
            .arg(&a_col)
            .arg(&a_val)
            .arg(&b_row)
            .arg(&b_col)
            .arg(&b_val)
            .arg(c_row)
            .arg(c_col)
            .arg(c)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()
    }

    /// Masked product with symmetric right operand `B` (two-pointer
    /// intersection, no binary search). `C = P_M(A·B)` with `B = B^T`.
    pub fn spgemm_masked_bsym(
        &self,
        nrow: usize,
        a: &Bsr4Matrix,
        b: &Bsr4Matrix,
        c_row: &Buffer<u32>,
        c_col: &Buffer<u32>,
        c: &Buffer<f32>,
    ) -> Result<()> {
        self.check_row_degree(&a.row_ptr, nrow)?;
        let a_row = self.buf_u32(&a.row_ptr)?;
        let a_col = self.buf_u32(&a.col_idx)?;
        let a_val = self.buf_f32(&a.values)?;
        let b_row = self.buf_u32(&b.row_ptr)?;
        let b_col = self.buf_u32(&b.col_idx)?;
        let b_val = self.buf_f32(&b.values)?;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_spgemm_masked_Bsym")
            .queue(self.queue().clone())
            .global_work_size(nrow * self.config.wg as usize)
            .local_work_size(self.config.wg as usize)
            .arg(nrow as u32)
            .arg(&a_row)
            .arg(&a_col)
            .arg(&a_val)
            .arg(&b_row)
            .arg(&b_col)
            .arg(&b_val)
            .arg(c_row)
            .arg(c_col)
            .arg(c)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()
    }

    /// `C = alpha*A + beta*B` (identical CSR structure for A, B, C).
    pub fn axpby(
        &self,
        nblock: usize,
        alpha: f32,
        a: &Buffer<f32>,
        beta: f32,
        b: &Buffer<f32>,
        c: &Buffer<f32>,
    ) -> Result<()> {
        let total = nblock * BS2;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_axpby")
            .queue(self.queue().clone())
            .global_work_size(total)
            .arg(nblock as u32)
            .arg(alpha)
            .arg(a)
            .arg(beta)
            .arg(b)
            .arg(c)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()
    }

    /// Zero a value buffer.
    pub fn zero(&self, nblock: usize, a: &Buffer<f32>) -> Result<()> {
        let total = nblock * BS2;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_zero")
            .queue(self.queue().clone())
            .global_work_size(total)
            .arg(nblock as u32)
            .arg(a)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()
    }

    /// Generalized McWeeny combination: `Knew = 3*Q - 2*V`.
    pub fn mcweeny(
        &self,
        nblock: usize,
        q: &Buffer<f32>,
        v: &Buffer<f32>,
        knew: &Buffer<f32>,
    ) -> Result<()> {
        let total = nblock * BS2;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_mcweeny")
            .queue(self.queue().clone())
            .global_work_size(total)
            .arg(nblock as u32)
            .arg(q)
            .arg(v)
            .arg(knew)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()
    }

    /// Metric TC2 update. `trace_ks` is a single-float device buffer holding
    /// `Tr(KS)`; the kernel reads it uniformly.
    pub fn tc2(
        &self,
        nblock: usize,
        k: &Buffer<f32>,
        q: &Buffer<f32>,
        trace_ks: &Buffer<f32>,
        nocc: f32,
        knew: &Buffer<f32>,
    ) -> Result<()> {
        let total = nblock * BS2;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_tc2")
            .queue(self.queue().clone())
            .global_work_size(total)
            .arg(nblock as u32)
            .arg(k)
            .arg(q)
            .arg(trace_ks)
            .arg(nocc)
            .arg(knew)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()
    }

    /// Symmetrize a BSR4 matrix in place using a precomputed transpose map.
    pub fn symmetrize(
        &self,
        nblock: usize,
        transpose_block: &Buffer<u32>,
        a: &Buffer<f32>,
    ) -> Result<()> {
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_symmetrize")
            .queue(self.queue().clone())
            .global_work_size(nblock)
            .arg(nblock as u32)
            .arg(transpose_block)
            .arg(a)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()
    }

    /// Mulliken charges from `KS`: `q_A = 2 * Tr((KS)_AA)`.
    pub fn mulliken_ks(
        &self,
        nrow: usize,
        diag_block: &Buffer<u32>,
        ks: &Buffer<f32>,
        q: &Buffer<f32>,
    ) -> Result<()> {
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_mulliken_KS")
            .queue(self.queue().clone())
            .global_work_size(nrow)
            .arg(nrow as u32)
            .arg(diag_block)
            .arg(ks)
            .arg(q)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()
    }

    /// `Tr(KS) = sum_A Tr((KS)_AA)` — partial reduction. Recursively reduce
    /// `partial` until a single float remains; returns that float on host.
    pub fn trace_ks(
        &self,
        nrow: usize,
        diag_block: &Buffer<u32>,
        ks: &Buffer<f32>,
    ) -> Result<f32> {
        let reduce_wg = self.config.reduce_wg as usize;
        // Stage 1: per-atom diagonal trace + first reduction.
        let n_groups = div_ceil(nrow, reduce_wg);
        let partial = self.zero_f32(n_groups)?;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_trace_KS_partial")
            .queue(self.queue().clone())
            .global_work_size(n_groups * reduce_wg)
            .local_work_size(reduce_wg)
            .arg(nrow as u32)
            .arg(diag_block)
            .arg(ks)
            .arg(&partial)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()?;
        self.reduce_to_one(n_groups, &partial)
    }

    /// `||KSK - K||_F^2` — partial reduction over all block scalars.
    pub fn idempotency_err(
        &self,
        nblock: usize,
        q_ksk: &Buffer<f32>,
        k: &Buffer<f32>,
    ) -> Result<f32> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n = nblock * BS2;
        let n_groups = div_ceil(n, reduce_wg);
        let partial = self.zero_f32(n_groups)?;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("bsr4_idempotency_partial")
            .queue(self.queue().clone())
            .global_work_size(n_groups * reduce_wg)
            .local_work_size(reduce_wg)
            .arg(nblock as u32)
            .arg(q_ksk)
            .arg(k)
            .arg(&partial)
            .build()
            .map_err(map_ocl_err)?;
        unsafe {
            kernel.enq().map_err(map_ocl_err)?;
        }
        self.rt.finish()?;
        let s2 = self.reduce_to_one(n_groups, &partial)?;
        Ok(s2.sqrt())
    }

    /// Recursive `reduce_sum_f32` until one float remains.
    fn reduce_to_one(&self, mut n: usize, input: &Buffer<f32>) -> Result<f32> {
        let reduce_wg = self.config.reduce_wg as usize;
        let mut current = input.clone();
        while n > 1 {
            let n_groups = div_ceil(n, reduce_wg);
            let out = self.zero_f32(n_groups)?;
            let kernel = Kernel::builder()
                .program(&self.program)
                .name("reduce_sum_f32")
                .queue(self.queue().clone())
                .global_work_size(n_groups * reduce_wg)
                .local_work_size(reduce_wg)
                .arg(n as u32)
                .arg(&current)
                .arg(&out)
                .build()
                .map_err(map_ocl_err)?;
            unsafe {
                kernel.enq().map_err(map_ocl_err)?;
            }
            self.rt.finish()?;
            current = out;
            n = n_groups;
        }
        let mut host = [0.0f32; 1];
        self.read_f32(&current, &mut host)?;
        Ok(host[0])
    }

    // ------------------------------------------------------------------
    // High-level Bsr4Matrix convenience methods
    // ------------------------------------------------------------------

    /// Run generic masked SpGEMM and return the result as a `Bsr4Matrix`
    /// with the given output mask `(c_row, c_col)`.
    pub fn matmul_masked(
        &self,
        a: &Bsr4Matrix,
        b: &Bsr4Matrix,
        c_mask: &(Vec<u32>, Vec<u32>),
    ) -> Result<Bsr4Matrix> {
        let nrow = a.n_atom;
        let nblock = c_mask.1.len();
        let c_row = self.buf_u32(&c_mask.0)?;
        let c_col = self.buf_u32(&c_mask.1)?;
        let c = self.zero_f32(nblock * BS2)?;
        self.spgemm_masked(nrow, a, b, &c_row, &c_col, &c)?;
        let mut values = vec![0.0f32; nblock * BS2];
        self.read_f32(&c, &mut values)?;
        Bsr4Matrix::from_parts(nrow, c_mask.0.clone(), c_mask.1.clone(), values)
    }

    /// Run symmetric-right masked SpGEMM and return the result as a
    /// `Bsr4Matrix`. `b` must be symmetric.
    pub fn matmul_masked_bsym(
        &self,
        a: &Bsr4Matrix,
        b: &Bsr4Matrix,
        c_mask: &(Vec<u32>, Vec<u32>),
    ) -> Result<Bsr4Matrix> {
        let nrow = a.n_atom;
        let nblock = c_mask.1.len();
        let c_row = self.buf_u32(&c_mask.0)?;
        let c_col = self.buf_u32(&c_mask.1)?;
        let c = self.zero_f32(nblock * BS2)?;
        self.spgemm_masked_bsym(nrow, a, b, &c_row, &c_col, &c)?;
        let mut values = vec![0.0f32; nblock * BS2];
        self.read_f32(&c, &mut values)?;
        Bsr4Matrix::from_parts(nrow, c_mask.0.clone(), c_mask.1.clone(), values)
    }

    /// Compute `Q = K·S·K` via two masked products:
    ///   `T = P_MT(K·S)`  (on the T mask)
    ///   `Q = P_MK(T·K)`  (on the K mask)
    /// Returns `(T, Q)`. Both `S` and `K` are symmetric, so both products use
    /// the symmetric-right kernel.
    pub fn ksk(
        &self,
        k: &Bsr4Matrix,
        s: &Bsr4Matrix,
        k_mask: &(Vec<u32>, Vec<u32>),
        t_mask: &(Vec<u32>, Vec<u32>),
    ) -> Result<(Bsr4Matrix, Bsr4Matrix)> {
        let t = self.matmul_masked_bsym(k, s, t_mask)?;
        let q = self.matmul_masked_bsym(&t, k, k_mask)?;
        Ok((t, q))
    }

    /// One generalized McWeeny purification step:
    ///   `Q = K·S·K`, `V = Q·S·K`, `K' = 3Q − 2V`.
    /// `V = K·S·K·S·K` is computed as `(Q·S)·K` using the T mask for the
    /// intermediate `Q·S` and the K mask for the final product.
    pub fn mcweeny_step(
        &self,
        k: &Bsr4Matrix,
        s: &Bsr4Matrix,
        k_mask: &(Vec<u32>, Vec<u32>),
        t_mask: &(Vec<u32>, Vec<u32>),
    ) -> Result<Bsr4Matrix> {
        let (_t_ks, q) = self.ksk(k, s, k_mask, t_mask)?;
        let us = self.matmul_masked_bsym(&q, s, t_mask)?; // U = Q·S
        let v = self.matmul_masked_bsym(&us, k, k_mask)?; // V = U·K = KSKSK
        let nblock = k.nblock();
        let q_buf = self.buf_f32(&q.values)?;
        let v_buf = self.buf_f32(&v.values)?;
        let knew_buf = self.zero_f32(nblock * BS2)?;
        self.mcweeny(nblock, &q_buf, &v_buf, &knew_buf)?;
        let mut values = vec![0.0f32; nblock * BS2];
        self.read_f32(&knew_buf, &mut values)?;
        Bsr4Matrix::from_parts(k.n_atom, k.row_ptr.clone(), k.col_idx.clone(), values)
    }

    /// One metric TC2 purification step:
    ///   `Q = K·S·K`, `n = Tr(KS)`, then `K' = Q` if `n > Nocc` else `2K − Q`.
    /// Returns `(K', n)` where `n` is the current `Tr(KS)`.
    pub fn tc2_step(
        &self,
        k: &Bsr4Matrix,
        s: &Bsr4Matrix,
        nocc: f32,
        k_mask: &(Vec<u32>, Vec<u32>),
        t_mask: &(Vec<u32>, Vec<u32>),
        diag_block: &Buffer<u32>,
    ) -> Result<(Bsr4Matrix, f32)> {
        // T = K·S (on T mask); trace uses T's diagonal blocks.
        let t = self.matmul_masked_bsym(k, s, t_mask)?;
        // Tr(KS) = sum_A Tr(T_AA). diag_block indexes into T's structure.
        let n = self.trace_ks(k.n_atom, diag_block, &self.buf_f32(&t.values)?)?;
        // Q = T·K (on K mask).
        let q = self.matmul_masked_bsym(&t, k, k_mask)?;
        let nblock = k.nblock();
        let k_buf = self.buf_f32(&k.values)?;
        let q_buf = self.buf_f32(&q.values)?;
        // trace_ks for the TC2 decision: store n in a 1-float device buffer.
        let trace_buf = self.buf_f32(&[n])?;
        let knew_buf = self.zero_f32(nblock * BS2)?;
        self.tc2(nblock, &k_buf, &q_buf, &trace_buf, nocc, &knew_buf)?;
        let mut values = vec![0.0f32; nblock * BS2];
        self.read_f32(&knew_buf, &mut values)?;
        Bsr4Matrix::from_parts(k.n_atom, k.row_ptr.clone(), k.col_idx.clone(), values).map(|m| (m, n))
    }

    /// Symmetrize a `Bsr4Matrix` on the GPU and return the result.
    pub fn symmetrize_mat(&self, m: &Bsr4Matrix) -> Result<Bsr4Matrix> {
        let transpose = crate::methods::sparse::bsr4::transpose_block_map(m);
        let tbuf = self.buf_u32(&transpose)?;
        let vals = self.buf_f32(&m.values)?;
        self.symmetrize(m.nblock(), &tbuf, &vals)?;
        let mut out = vec![0.0f32; m.values.len()];
        self.read_f32(&vals, &mut out)?;
        Bsr4Matrix::from_parts(m.n_atom, m.row_ptr.clone(), m.col_idx.clone(), out)
    }

    /// Mulliken charges `q_A = 2·Tr((KS)_AA)` from a `KS` matrix.
    pub fn mulliken(&self, ks: &Bsr4Matrix) -> Result<Vec<f32>> {
        let diag = crate::methods::sparse::bsr4::diag_block_map(ks)?;
        let diag_buf = self.buf_u32(&diag)?;
        let ks_buf = self.buf_f32(&ks.values)?;
        let qbuf = self.zero_f32(ks.n_atom)?;
        self.mulliken_ks(ks.n_atom, &diag_buf, &ks_buf, &qbuf)?;
        let mut q = vec![0.0f32; ks.n_atom];
        self.read_f32(&qbuf, &mut q)?;
        Ok(q)
    }

    // ==================================================================
    // P0 high-level solver pieces (per SparseLargeSystemOpenCL.chat.md)
    // ==================================================================

    /// Frobenius norm of a Bsr4Matrix via GPU reduction.
    /// Uses `bsr4_idempotency_partial` with a zero K buffer (so it computes
    /// sum Q[i]^2 = ||Q||_F^2), then takes sqrt on host.
    pub fn frobenius_norm(&self, m: &Bsr4Matrix) -> Result<f32> {
        let nblock = m.nblock();
        let q_buf = self.buf_f32(&m.values)?;
        let zero = self.zero_f32(m.values.len())?;
        let norm_sq = self.idempotency_err(nblock, &q_buf, &zero)?;
        Ok(norm_sq)
    }

    /// Direct identity residual `||A - I||_F` via the
    /// `bsr4_identity_residual_partial` kernel.  This avoids the
    /// catastrophic cancellation in `||A||² - 2·Tr(A) + N` when A ≈ I.
    pub fn identity_residual(&self, a: &Bsr4Matrix) -> Result<f32> {
        let nblock = a.nblock();
        let a_buf = self.buf_f32(&a.values)?;
        // Build diag_flag: 1 for diagonal blocks, 0 otherwise.
        let diag = crate::methods::sparse::bsr4::diag_block_map(a)?;
        let mut diag_flag = vec![0u32; nblock];
        for &b in &diag { diag_flag[b as usize] = 1; }
        let diag_flag_buf = self.buf_u32(&diag_flag)?;
        // Partial reduction + recursive reduction.
        let reduce_wg = self.config.reduce_wg as usize;
        let n_groups = div_ceil(nblock * BS2, reduce_wg);
        let partial = self.zero_f32(n_groups)?;
        self.identity_residual_partial_dev(nblock, &diag_flag_buf, &a_buf, &partial)?;
        let s2 = self.reduce_to_one_dev(n_groups, &partial)?;
        Ok(s2.sqrt())
    }

    /// Newton-Schulz / Hotelling iteration for sparse approximate inverse:
    ///
    ///   Z_{n+1} = 2 Z_n - Z_n S Z_n
    ///
    /// with Z₀ = α·I, α = 1/||S||_∞.
    ///
    /// Uses M_K for Z storage and M_T = M_K ∘ M_HS for the intermediate
    /// T = Z·S.  Monitors R_Z = ||I - ZS||_F / sqrt(N_orb).
    ///
    /// Stops when R_Z < tol or when improvement stalls (R_Z^{n+1} > 0.9 R_Z^n
    /// for `stall` consecutive iterations).
    ///
    /// Returns (Z, final_R_Z, iterations).
    pub fn newton_schulz_inverse(
        &self,
        s: &Bsr4Matrix,
        k_mask: &(Vec<u32>, Vec<u32>),
        t_mask: &(Vec<u32>, Vec<u32>),
        max_iter: usize,
        tol: f32,
        stall: usize,
    ) -> Result<(Bsr4Matrix, f32, usize)> {
        let n_atom = s.n_atom;
        let n_orb = (n_atom * BS) as f32;

        // Z₀ = α·I on M_K, α = 1/||S||_∞.
        let s_inf = crate::methods::sparse::bsr4::inf_norm(s);
        let alpha = 1.0 / s_inf.max(1e-30);
        let mut z = crate::methods::sparse::bsr4::build_identity(n_atom, k_mask)?;
        for v in z.values.iter_mut() {
            *v *= alpha;
        }

        let mut prev_rz = f32::INFINITY;
        let mut stall_count = 0;
        let mut iter_done = 0;

        for iter in 0..max_iter {
            iter_done = iter + 1;
            // T = Z·S  (on M_T, S symmetric → Bsym)
            let t = self.matmul_masked_bsym(&z, s, t_mask)?;

            // R_Z = ||I - T||_F / sqrt(N_orb) — direct identity residual,
            // avoids catastrophic cancellation in ||T||² - 2·Tr(T) + N when
            // T ≈ I (which falsely rounds to zero in f32).
            let rz = self.identity_residual(&t)? / n_orb.sqrt();

            if algebra_verbose() {
                println!("  Newton-Schulz iter {iter}: R_Z = {rz:e}");
            }

            if rz < tol {
                return Ok((z, rz, iter_done));
            }
            if iter > 0 && rz > 0.9 * prev_rz {
                stall_count += 1;
                if stall_count >= stall {
                    println!("  Newton-Schulz stalled after {iter_done} iters, R_Z = {rz:e}");
                    return Err(DftbError::InvalidInput(format!(
                        "Newton-Schulz did not converge: stalled after {iter_done} iters, R_Z={rz:e} (tol={tol:e})"
                    )));
                }
            } else {
                stall_count = 0;
            }
            prev_rz = rz;

            // Q = T·Z  (on M_K, Z symmetric → Bsym)
            let q = self.matmul_masked_bsym(&t, &z, k_mask)?;

            // Z_new = 2*Z - Q  (axpby: alpha=2, beta=-1)
            let nblock = z.nblock();
            let z_buf = self.buf_f32(&z.values)?;
            let q_buf = self.buf_f32(&q.values)?;
            let znew_buf = self.zero_f32(nblock * BS2)?;
            self.axpby(nblock, 2.0, &z_buf, -1.0, &q_buf, &znew_buf)?;
            let mut znew_vals = vec![0.0f32; nblock * BS2];
            self.read_f32(&znew_buf, &mut znew_vals)?;
            z = Bsr4Matrix::from_parts(n_atom, k_mask.0.clone(), k_mask.1.clone(), znew_vals)?;
            z = self.symmetrize_mat(&z)?;
        }

        // Final residual — direct identity residual, no cancellation.
        let t = self.matmul_masked_bsym(&z, s, t_mask)?;
        let rz = self.identity_residual(&t)? / n_orb.sqrt();
        Err(DftbError::InvalidInput(format!(
            "Newton-Schulz did not converge: exhausted {max_iter} iters, final R_Z={rz:e} (tol={tol:e})"
        )))
    }

    /// Build the Hamiltonian-derived initial density kernel:
    ///
    ///   K₀ = (εmax·Z - Z·H·Z) / (εmax - εmin)
    ///
    /// where Z ≈ S⁻¹.  Uses:
    ///   B = Z·H   (on M_T, generic SpGEMM — Z symmetric, H symmetric, but
    ///              Z·H is NOT symmetric)
    ///   A = B·Z   (on M_K, Bsym — Z is symmetric right operand)
    ///   K₀ = (emax·Z - A) / Δε   (elementwise axpby)
    ///
    /// Returns K₀ on M_K.
    pub fn build_k0(
        &self,
        h: &Bsr4Matrix,
        s: &Bsr4Matrix,
        z: &Bsr4Matrix,
        k_mask: &(Vec<u32>, Vec<u32>),
        t_mask: &(Vec<u32>, Vec<u32>),
        emin: f32,
        emax: f32,
    ) -> Result<Bsr4Matrix> {
        let delta = (emax - emin).max(1e-12);
        let alpha_k = emax / delta;
        let beta_k = -1.0 / delta;

        // B = Z·H  (on M_T, generic — Z·H not symmetric)
        let b = self.matmul_masked(z, h, t_mask)?;

        // A = B·Z  (on M_K, Bsym — Z symmetric)
        let a = self.matmul_masked_bsym(&b, z, k_mask)?;

        // K₀ = alpha_k * Z - beta_k_abs * A  (axpby with beta = beta_k)
        let nblock = z.nblock();
        let z_buf = self.buf_f32(&z.values)?;
        let a_buf = self.buf_f32(&a.values)?;
        let k0_buf = self.zero_f32(nblock * BS2)?;
        self.axpby(nblock, alpha_k, &z_buf, beta_k, &a_buf, &k0_buf)?;
        let mut k0_vals = vec![0.0f32; nblock * BS2];
        self.read_f32(&k0_buf, &mut k0_vals)?;
        let mut k0 =
            Bsr4Matrix::from_parts(z.n_atom, k_mask.0.clone(), k_mask.1.clone(), k0_vals)?;
        k0 = self.symmetrize_mat(&k0)?;
        Ok(k0)
    }

    /// Compute spectral bounds (emin, emax) of B = Z·H via Gershgorin at the
    /// orbital level, with `padding` fractional widening.
    ///
    /// Returns bounds suitable for K₀ construction. Conservative is safe;
    /// too-tight bounds can push K₀S eigenvalues outside [0,1].
    pub fn spectral_bounds(
        &self,
        h: &Bsr4Matrix,
        z: &Bsr4Matrix,
        t_mask: &(Vec<u32>, Vec<u32>),
        padding: f32,
    ) -> Result<(f32, f32)> {
        let b = self.matmul_masked(z, h, t_mask)?;
        let (mut emin, mut emax) = crate::methods::sparse::bsr4::gershgorin_bounds(&b)?;
        let span = (emax - emin).abs() * padding;
        emin -= span;
        emax += span;
        Ok((emin, emax))
    }

    /// Hamiltonian commutator residual:
    ///
    ///   R_H = ||H·K·S - S·K·H||_F
    ///
    /// For the exact ground-state projector, HKS = SKH, so R_H → 0.
    /// This catches the case where K is idempotent but projects onto the
    /// wrong subspace.
    ///
    /// Uses full-mask-compatible logic: all products on the same mask.
    /// For multi-mask, all products use `mask` (which must be large enough
    /// to contain the support of all intermediates — use full mask for now).
    pub fn hamiltonian_residual(
        &self,
        h: &Bsr4Matrix,
        k: &Bsr4Matrix,
        s: &Bsr4Matrix,
        mask: &(Vec<u32>, Vec<u32>),
    ) -> Result<f32> {
        // KS = K·S  (Bsym, S symmetric)
        let ks = self.matmul_masked_bsym(k, s, mask)?;
        // HKS = H·KS  (generic — KS not symmetric)
        let hks = self.matmul_masked(h, &ks, mask)?;

        // SK = S·K  (Bsym, K symmetric)
        let sk = self.matmul_masked_bsym(s, k, mask)?;
        // SKH = SK·H  (Bsym, H symmetric)
        let skh = self.matmul_masked_bsym(&sk, h, mask)?;

        // R_H = ||HKS - SKH||_F
        // Use axpby to compute diff = HKS - SKH, then frobenius_norm.
        let nblock = hks.nblock();
        let hks_buf = self.buf_f32(&hks.values)?;
        let skh_buf = self.buf_f32(&skh.values)?;
        let diff_buf = self.zero_f32(nblock * BS2)?;
        self.axpby(nblock, 1.0, &hks_buf, -1.0, &skh_buf, &diff_buf)?;
        let mut diff_vals = vec![0.0f32; nblock * BS2];
        self.read_f32(&diff_buf, &mut diff_vals)?;
        let diff = Bsr4Matrix::from_parts(
            hks.n_atom,
            hks.row_ptr.clone(),
            hks.col_idx.clone(),
            diff_vals,
        )?;
        self.frobenius_norm(&diff)
    }

    /// Full TC2 purification loop from a starting K₀:
    ///
    ///   repeat:
    ///     KS = K·S, Q = KS·K, n = Tr(KS)
    ///     K = Q if n > Nocc else 2K-Q
    ///     symmetrize, monitor
    ///
    /// Stops when ||KSK-K||_F < tol or max_iter reached.
    /// Returns (K_final, final_R_I, final_Tr, iterations).
    pub fn tc2_purify(
        &self,
        k0: &Bsr4Matrix,
        s: &Bsr4Matrix,
        nocc: f32,
        k_mask: &(Vec<u32>, Vec<u32>),
        t_mask: &(Vec<u32>, Vec<u32>),
        max_iter: usize,
        tol: f32,
    ) -> Result<(Bsr4Matrix, f32, f32, usize, Vec<(usize, f32, f32)>)> {
        let diag_dummy =
            crate::methods::sparse::bsr4::Bsr4Matrix::from_structure(
                k0.n_atom,
                t_mask.0.clone(),
                t_mask.1.clone(),
            )?;
        let diag = crate::methods::sparse::bsr4::diag_block_map(&diag_dummy)?;
        let diag_buf = self.buf_u32(&diag)?;

        let mut k = k0.clone();
        let mut r_i = f32::INFINITY;
        let mut tr = 0.0f32;
        let mut history: Vec<(usize, f32, f32)> = Vec::new();

        // Track best iteration to guard against TC2 divergence.
        // TC2 can converge to a minimum R_I then diverge (especially in f32
        // when the tolerance is below the achievable precision).
        let mut best_k = k.clone();
        let mut best_r_i = f32::INFINITY;
        let mut best_tr = 0.0f32;
        let mut best_iter = 0usize;

        for iter in 0..max_iter {
            // Q = K·S·K  (on K mask)
            let (_t, q) = self.ksk(&k, s, k_mask, t_mask)?;
            // R_I = ||KSK - K||_F
            let q_buf = self.buf_f32(&q.values)?;
            let k_buf = self.buf_f32(&k.values)?;
            r_i = self.idempotency_err(k.nblock(), &q_buf, &k_buf)?;

            // Tr(KS) via T = K·S
            let t = self.matmul_masked_bsym(&k, s, t_mask)?;
            let t_buf = self.buf_f32(&t.values)?;
            tr = self.trace_ks(k.n_atom, &diag_buf, &t_buf)?;

            if algebra_verbose() {
                println!("  TC2 iter {iter}: R_I={r_i:e}  Tr(KS)={tr:.6}  (Nocc={nocc})");
            }
            history.push((iter, r_i, tr));

            // Track best (minimum R_I) iteration
            if r_i < best_r_i {
                best_r_i = r_i;
                best_k = k.clone();
                best_tr = tr;
                best_iter = iter;
            }

            if r_i < tol {
                if (tr - nocc).abs() <= TC2_TRACE_TOL {
                    return Ok((k, r_i, tr, iter + 1, history));
                }
                if algebra_verbose() {
                    println!(
                        "  TC2 R_I={r_i:e} < tol but Tr(KS)={tr:.6} != Nocc={nocc} (wrong-rank projector not accepted)"
                    );
                }
            }

            // Divergence detection: if R_I has grown by >10x from the best,
            // TC2 is diverging. Return the best K found.
            if r_i > best_r_i * 10.0 && best_r_i < f32::INFINITY {
                println!(
                    "  TC2 diverging at iter {iter}: R_I={r_i:e} > 10×best={best_r_i:e}, returning best (iter {best_iter})"
                );
                return Err(DftbError::InvalidInput(format!(
                    "TC2 purification did not converge: diverged at iter {iter}, best R_I={best_r_i:e} at iter {best_iter} (tol={tol:e}, max_iter={max_iter})"
                )));
            }

            let (knew, _) = self.tc2_step(&k, s, nocc, k_mask, t_mask, &diag_buf)?;
            k = self.symmetrize_mat(&knew)?;
        }

        // Exhausted iterations without converging.
        if best_r_i < r_i {
            println!(
                "  TC2 exhausted {max_iter} iters, best R_I={best_r_i:e} at iter {best_iter}"
            );
            Err(DftbError::InvalidInput(format!(
                "TC2 purification did not converge: exhausted {max_iter} iters, best R_I={best_r_i:e} at iter {best_iter} (tol={tol:e})"
            )))
        } else {
            Err(DftbError::InvalidInput(format!(
                "TC2 purification did not converge: exhausted {max_iter} iters, final R_I={r_i:e} (tol={tol:e})"
            )))
        }
    }
}

/// Compare a GPU-produced `Bsr4Matrix` against a dense reference (already
/// expanded) and return the max absolute elementwise difference.
///
/// **P0 firewall (manifest v3 §4.1):** This calls `to_dense()` and is
/// **reference/test-only**. When the `sparse_firewall` feature is enabled,
/// this function panics.
pub fn compare_to_dense(gpu: &Bsr4Matrix, dense_ref: &[f32]) -> f32 {
    #[cfg(feature = "sparse_firewall")]
    {
        panic!("P0 SPARSE FIREWALL: compare_to_dense() called from production sparse path. \
                This calls to_dense() which allocates an O(Norb²) dense matrix. \
                Disable the `sparse_firewall` feature for test/reference use.");
    }
    #[cfg(not(feature = "sparse_firewall"))]
    {
        let dense_gpu = gpu.to_dense();
        dense_max_abs_diff(&dense_gpu, dense_ref)
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// Re-export BS/BS2 for convenience.
pub use crate::methods::sparse::bsr4::{BS as PUB_BS, BS2 as PUB_BS2};

// ============================================================================
// Device-resident BSR4 structures (P0-D: GPU residency)
//
// These structures keep CSR structure and values on the GPU for the entire
// purification loop, eliminating the per-operation host→GPU→host round-trips
// that dominate the current wrapper. See GPU_Optimization.chat.md §17–§19,
// §31.6.
// ============================================================================

/// Immutable BSR4 CSR structure on the GPU: row pointers, column indices,
/// diagonal block map, and transpose block map. Uploaded once and shared
/// by all matrices with the same sparsity pattern.
pub struct GpuBsrStructure {
    pub n_atom: usize,
    pub nblock: usize,
    row_ptr: Buffer<u32>,
    col_idx: Buffer<u32>,
    diag_block: Buffer<u32>,
    diag_flag: Buffer<u32>,
    transpose_block: Buffer<u32>,
}

impl GpuBsrStructure {
    /// Public accessor for the transpose-block map (needed by `symmetrize_dev`).
    pub fn transpose_block(&self) -> &Buffer<u32> { &self.transpose_block }
    /// Public accessor for the diagonal-block map (needed by `trace_ks_*`).
    pub fn diag_block(&self) -> &Buffer<u32> { &self.diag_block }
    /// Public accessor for the row pointer buffer.
    pub fn row_ptr(&self) -> &Buffer<u32> { &self.row_ptr }
    /// Public accessor for the column index buffer.
    pub fn col_idx(&self) -> &Buffer<u32> { &self.col_idx }

    /// Build a device-resident structure from a host CSR mask `(row_ptr,
    /// col_idx)`. Computes `diag_block_map` and `transpose_block_map` on the
    /// host, then uploads all four arrays once.
    pub fn new(gpu: &SparseBsr4Gpu, n_atom: usize, mask: &(Vec<u32>, Vec<u32>)) -> Result<Self> {
        // Fail-loud check: verify no row exceeds MAX_LEFT_BLOCKS before
        // uploading. The GPU kernel silently returns (producing zero output)
        // for rows that overflow the local-memory cache.
        gpu.check_row_degree(&mask.0, n_atom)?;
        let nblock = mask.1.len();
        let row_ptr = gpu.buf_u32(&mask.0)?;
        let col_idx = gpu.buf_u32(&mask.1)?;
        // Compute diag and transpose maps on host (they are structural and
        // never change during purification).
        let dummy = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone())?;
        let diag = crate::methods::sparse::bsr4::diag_block_map(&dummy)?;
        let transpose = crate::methods::sparse::bsr4::transpose_block_map(&dummy);
        let mut diag_flag = vec![0u32; nblock];
        for &b in &diag {
            diag_flag[b as usize] = 1;
        }
        let diag_block = gpu.buf_u32(&diag)?;
        let diag_flag = gpu.buf_u32(&diag_flag)?;
        let transpose_block = gpu.buf_u32(&transpose)?;
        Ok(Self { n_atom, nblock, row_ptr, col_idx, diag_block, diag_flag, transpose_block })
    }

    pub fn n_atom(&self) -> usize { self.n_atom }
    pub fn nblock(&self) -> usize { self.nblock }

    /// Read `row_ptr` back to host (blocking). Used for one-time setup.
    pub fn row_ptr_host(&self, gpu: &SparseBsr4Gpu) -> Result<Vec<u32>> {
        let mut out = vec![0u32; self.n_atom + 1];
        gpu.read_u32(&self.row_ptr, &mut out)?;
        Ok(out)
    }

    /// Read `col_idx` back to host (blocking). Used for one-time setup.
    pub fn col_idx_host(&self, gpu: &SparseBsr4Gpu) -> Result<Vec<u32>> {
        let mut out = vec![0u32; self.nblock];
        gpu.read_u32(&self.col_idx, &mut out)?;
        Ok(out)
    }
}

/// A device-resident BSR4 matrix: a reference to an immutable `GpuBsrStructure`
/// plus a device values buffer. The structure is shared (via `Arc`); only the
/// values differ between matrices with the same mask.
pub struct GpuBsrMatrix {
    pub struct_: Arc<GpuBsrStructure>,
    pub values: Buffer<f32>,
}

/// Device-resident symbolic SpGEMM plan (P4, manifest v3 §4.6).
/// Uploaded once for a frozen mask triple and reused across launches.
pub struct SpgemmPlanGpu {
    pub plan_ptr: Buffer<u32>,
    pub plan_a_idx: Buffer<u32>,
    pub plan_b_idx: Buffer<u32>,
}

impl GpuBsrMatrix {
    /// Create a device-resident matrix from a host `Bsr4Matrix`, uploading
    /// values once. The structure is built (or reused) from the matrix's CSR
    /// mask.
    pub fn from_host(gpu: &SparseBsr4Gpu, m: &Bsr4Matrix) -> Result<Self> {
        let structure = GpuBsrStructure::new(gpu, m.n_atom, &(m.row_ptr.clone(), m.col_idx.clone()))?;
        let values = gpu.buf_f32(&m.values)?;
        Ok(Self { struct_: Arc::new(structure), values })
    }

    /// Create a zero-filled device-resident matrix on a given structure.
    pub fn zero(gpu: &SparseBsr4Gpu, struct_: &Arc<GpuBsrStructure>) -> Result<Self> {
        let values = gpu.zero_f32(struct_.nblock * BS2)?;
        Ok(Self { struct_: struct_.clone(), values })
    }

    /// Upload values from a host slice into an existing device buffer
    /// (overwrites). The slice length must match `nblock * BS2`.
    pub fn upload_values(&self, gpu: &SparseBsr4Gpu, host: &[f32]) -> Result<()> {
        if host.len() != self.struct_.nblock * BS2 {
            return Err(DftbError::InvalidInput(format!(
                "upload_values: len {} != nblock {} * BS2 {}",
                host.len(), self.struct_.nblock, BS2
            )));
        }
        gpu.write_f32(&self.values, host)
    }

    /// Read values back to host (blocking). Use sparingly — only for
    /// diagnostics or final output.
    pub fn read_values(&self, gpu: &SparseBsr4Gpu) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.struct_.nblock * BS2];
        gpu.read_f32(&self.values, &mut out)?;
        Ok(out)
    }

    /// Convert back to a host `Bsr4Matrix` (blocking readback).
    pub fn to_host(&self, gpu: &SparseBsr4Gpu) -> Result<Bsr4Matrix> {
        let values = self.read_values(gpu)?;
        // Reconstruct row_ptr and col_idx from the structure. We need to read
        // them back from the device.
        let mut row_ptr = vec![0u32; self.struct_.n_atom + 1];
        gpu.read_u32(&self.struct_.row_ptr, &mut row_ptr)?;
        let mut col_idx = vec![0u32; self.struct_.nblock];
        gpu.read_u32(&self.struct_.col_idx, &mut col_idx)?;
        Bsr4Matrix::from_parts(self.struct_.n_atom, row_ptr, col_idx, values)
    }
}

// ============================================================================
// Device-resident kernel launches (no host transfer, no allocation, no
// Kernel::builder() per call). These use the cached kernel handles and
// `set_arg` to swap buffers.
// ============================================================================

impl SparseBsr4Gpu {
    /// Write an f32 buffer (non-blocking unless the queue is flushed).
    pub fn write_f32(&self, buf: &Buffer<f32>, data: &[f32]) -> Result<()> {
        buf.write(data).enq().map_err(map_ocl_err)
    }

    /// Device-resident masked SpGEMM with symmetric right operand.
    /// `C = P_M(A·B)` where `B = B^T`. No host transfer, no allocation,
    /// no `finish()`. The caller is responsible for ensuring `C` is zeroed
    /// if needed (the kernel overwrites, not accumulates).
    ///
    /// **Does not call finish()** — the caller controls synchronization.
    pub fn spgemm_bsym_dev(
        &self,
        a: &GpuBsrMatrix,
        b: &GpuBsrMatrix,
        c: &GpuBsrMatrix,
    ) -> Result<()> {
        let nrow = a.struct_.n_atom;
        let gws = nrow * self.config.wg as usize;
        let k = &self.k_spgemm_bsym;
        k.set_arg(0, nrow as u32).map_err(map_ocl_err)?;
        k.set_arg(1, &a.struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(2, &a.struct_.col_idx).map_err(map_ocl_err)?;
        k.set_arg(3, &a.values).map_err(map_ocl_err)?;
        k.set_arg(4, &b.struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(5, &b.struct_.col_idx).map_err(map_ocl_err)?;
        k.set_arg(6, &b.values).map_err(map_ocl_err)?;
        k.set_arg(7, &c.struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(8, &c.struct_.col_idx).map_err(map_ocl_err)?;
        k.set_arg(9, &c.values).map_err(map_ocl_err)?;
        // Use a default global work size; the actual size is set via
        // cmd_buffer. ocl Kernel::enq uses the builder's GWS.
        // We need to re-set the global work size per launch.
        // Unfortunately ocl Kernel doesn't support changing GWS after build.
        // We use the raw enqueue path.
        unsafe {
            // ocl Kernel enq uses the GWS set at build time. To change it,
            // we use the underlying clEnqueueNDRangeKernel via ocl's
            // internal API. The simplest approach: use Kernel::cmd() which
            // returns a builder that allows setting GWS per launch.
            k.cmd()
                .global_work_size(gws)
                .local_work_size(self.config.wg as usize)
                .enq()
                .map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident generic masked SpGEMM (non-symmetric right operand).
    /// `C = P_M(A·B)`. No host transfer.
    pub fn spgemm_masked_dev(
        &self,
        a: &GpuBsrMatrix,
        b: &GpuBsrMatrix,
        c: &GpuBsrMatrix,
    ) -> Result<()> {
        let nrow = a.struct_.n_atom;
        let gws = nrow * self.config.wg as usize;
        let k = &self.k_spgemm_masked;
        k.set_arg(0, nrow as u32).map_err(map_ocl_err)?;
        k.set_arg(1, &a.struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(2, &a.struct_.col_idx).map_err(map_ocl_err)?;
        k.set_arg(3, &a.values).map_err(map_ocl_err)?;
        k.set_arg(4, &b.struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(5, &b.struct_.col_idx).map_err(map_ocl_err)?;
        k.set_arg(6, &b.values).map_err(map_ocl_err)?;
        k.set_arg(7, &c.struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(8, &c.struct_.col_idx).map_err(map_ocl_err)?;
        k.set_arg(9, &c.values).map_err(map_ocl_err)?;
        unsafe {
            k.cmd()
                .global_work_size(gws)
                .local_work_size(self.config.wg as usize)
                .enq()
                .map_err(map_ocl_err)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // P4: Symbolic SpGEMM plan (manifest v3 §4.6)
    // -----------------------------------------------------------------

    /// Upload a symbolic SpGEMM plan to the device. The plan is built once
    /// for a frozen mask triple (A, B, C) and reused across all SpGEMM
    /// launches with that mask. See `bsr4::build_spgemm_plan_bsym`.
    pub fn upload_plan(&self, plan: &SpgemmPlan) -> Result<SpgemmPlanGpu> {
        let plan_ptr = self.rt.buffer_from_slice(&plan.plan_ptr)?;
        let plan_a_idx = self.rt.buffer_from_slice(&plan.plan_a_idx)?;
        let plan_b_idx = self.rt.buffer_from_slice(&plan.plan_b_idx)?;
        Ok(SpgemmPlanGpu { plan_ptr, plan_a_idx, plan_b_idx })
    }

    /// Device-resident symbolic-plan SpGEMM (Bsym variant).
    /// `C = P_M(A·B)` where B is symmetric, using a precomputed plan.
    /// No host transfer, no intersection at runtime — only loads + 4×4 FMAs.
    pub fn spgemm_plan_bsym_dev(
        &self,
        a: &GpuBsrMatrix,
        b: &GpuBsrMatrix,
        plan: &SpgemmPlanGpu,
        c: &GpuBsrMatrix,
    ) -> Result<()> {
        let nrow = a.struct_.n_atom;
        let gws = nrow * self.config.wg as usize;
        let k = &self.k_spgemm_plan_bsym;
        k.set_arg(0, nrow as u32).map_err(map_ocl_err)?;
        k.set_arg(1, &a.struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(2, &a.values).map_err(map_ocl_err)?;
        k.set_arg(3, &b.values).map_err(map_ocl_err)?;
        k.set_arg(4, &plan.plan_ptr).map_err(map_ocl_err)?;
        k.set_arg(5, &plan.plan_a_idx).map_err(map_ocl_err)?;
        k.set_arg(6, &plan.plan_b_idx).map_err(map_ocl_err)?;
        k.set_arg(7, &c.struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(8, &c.values).map_err(map_ocl_err)?;
        unsafe {
            k.cmd()
                .global_work_size(gws)
                .local_work_size(self.config.wg as usize)
                .enq()
                .map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident `C = alpha*A + beta*B` (elementwise, same structure).
    /// No host transfer, no `finish()`.
    pub fn axpby_dev(
        &self,
        nblock: usize,
        alpha: f32,
        a: &Buffer<f32>,
        beta: f32,
        b: &Buffer<f32>,
        c: &Buffer<f32>,
    ) -> Result<()> {
        let total = nblock * BS2;
        let k = &self.k_axpby;
        k.set_arg(0, nblock as u32).map_err(map_ocl_err)?;
        k.set_arg(1, alpha).map_err(map_ocl_err)?;
        k.set_arg(2, a).map_err(map_ocl_err)?;
        k.set_arg(3, beta).map_err(map_ocl_err)?;
        k.set_arg(4, b).map_err(map_ocl_err)?;
        k.set_arg(5, c).map_err(map_ocl_err)?;
        unsafe {
            k.cmd().global_work_size(total).enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident zero of a values buffer. No `finish()`.
    pub fn zero_dev(&self, nblock: usize, a: &Buffer<f32>) -> Result<()> {
        let total = nblock * BS2;
        let k = &self.k_zero;
        k.set_arg(0, nblock as u32).map_err(map_ocl_err)?;
        k.set_arg(1, a).map_err(map_ocl_err)?;
        unsafe {
            k.cmd().global_work_size(total).enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident TC2 update: reads `trace_ks` (1-float buffer) and
    /// writes `Knew`. No host transfer, no `finish()`.
    pub fn tc2_dev(
        &self,
        nblock: usize,
        k: &Buffer<f32>,
        q: &Buffer<f32>,
        trace_ks: &Buffer<f32>,
        nocc: f32,
        knew: &Buffer<f32>,
    ) -> Result<()> {
        let total = nblock * BS2;
        let kern = &self.k_tc2;
        kern.set_arg(0, nblock as u32).map_err(map_ocl_err)?;
        kern.set_arg(1, k).map_err(map_ocl_err)?;
        kern.set_arg(2, q).map_err(map_ocl_err)?;
        kern.set_arg(3, trace_ks).map_err(map_ocl_err)?;
        kern.set_arg(4, nocc).map_err(map_ocl_err)?;
        kern.set_arg(5, knew).map_err(map_ocl_err)?;
        unsafe {
            kern.cmd().global_work_size(total).enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident symmetrize in place. No `finish()`.
    pub fn symmetrize_dev(
        &self,
        nblock: usize,
        transpose_block: &Buffer<u32>,
        a: &Buffer<f32>,
    ) -> Result<()> {
        let k = &self.k_symmetrize;
        k.set_arg(0, nblock as u32).map_err(map_ocl_err)?;
        k.set_arg(1, transpose_block).map_err(map_ocl_err)?;
        k.set_arg(2, a).map_err(map_ocl_err)?;
        unsafe {
            k.cmd().global_work_size(nblock).enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident `Tr(KS)` partial reduction. Writes partial sums to
    /// `partial` buffer. No `finish()`.
    pub fn trace_ks_partial_dev(
        &self,
        nrow: usize,
        diag_block: &Buffer<u32>,
        ks: &Buffer<f32>,
        partial: &Buffer<f32>,
    ) -> Result<()> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n_groups = div_ceil(nrow, reduce_wg);
        let k = &self.k_trace_partial;
        k.set_arg(0, nrow as u32).map_err(map_ocl_err)?;
        k.set_arg(1, diag_block).map_err(map_ocl_err)?;
        k.set_arg(2, ks).map_err(map_ocl_err)?;
        k.set_arg(3, partial).map_err(map_ocl_err)?;
        unsafe {
            k.cmd()
                .global_work_size(n_groups * reduce_wg)
                .local_work_size(reduce_wg)
                .enq()
                .map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident `reduce_sum_f32`. No `finish()`.
    pub fn reduce_dev(
        &self,
        n: usize,
        input: &Buffer<f32>,
        output: &Buffer<f32>,
    ) -> Result<()> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n_groups = div_ceil(n, reduce_wg);
        let k = &self.k_reduce;
        k.set_arg(0, n as u32).map_err(map_ocl_err)?;
        k.set_arg(1, input).map_err(map_ocl_err)?;
        k.set_arg(2, output).map_err(map_ocl_err)?;
        unsafe {
            k.cmd()
                .global_work_size(n_groups * reduce_wg)
                .local_work_size(reduce_wg)
                .enq()
                .map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident idempotency partial reduction. No `finish()`.
    pub fn idempotency_partial_dev(
        &self,
        nblock: usize,
        q_ksk: &Buffer<f32>,
        k: &Buffer<f32>,
        partial: &Buffer<f32>,
    ) -> Result<()> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n = nblock * BS2;
        let n_groups = div_ceil(n, reduce_wg);
        let kern = &self.k_idempotency;
        kern.set_arg(0, nblock as u32).map_err(map_ocl_err)?;
        kern.set_arg(1, q_ksk).map_err(map_ocl_err)?;
        kern.set_arg(2, k).map_err(map_ocl_err)?;
        kern.set_arg(3, partial).map_err(map_ocl_err)?;
        unsafe {
            kern.cmd()
                .global_work_size(n_groups * reduce_wg)
                .local_work_size(reduce_wg)
                .enq()
                .map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Device-resident direct identity residual partial reduction. No
    /// cancellation through a trace or matrix norm is used.
    pub fn identity_residual_partial_dev(
        &self,
        nblock: usize,
        diag_flag: &Buffer<u32>,
        a: &Buffer<f32>,
        partial: &Buffer<f32>,
    ) -> Result<()> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n = nblock * BS2;
        let n_groups = div_ceil(n, reduce_wg);
        let kern = &self.k_identity_residual;
        kern.set_arg(0, nblock as u32).map_err(map_ocl_err)?;
        kern.set_arg(1, diag_flag).map_err(map_ocl_err)?;
        kern.set_arg(2, a).map_err(map_ocl_err)?;
        kern.set_arg(3, partial).map_err(map_ocl_err)?;
        unsafe {
            kern.cmd()
                .global_work_size(n_groups * reduce_wg)
                .local_work_size(reduce_wg)
                .enq()
                .map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Reduce a partial vector into a preallocated one-float device buffer.
    /// scratch_a and scratch_b must not alias input or output.
    fn reduce_to_output_dev(
        &self,
        mut n: usize,
        input: &Buffer<f32>,
        scratch_a: &Buffer<f32>,
        scratch_b: &Buffer<f32>,
        output: &Buffer<f32>,
    ) -> Result<()> {
        if n == 0 {
            return Err(DftbError::InvalidInput("reduction input is empty".into()));
        }
        let reduce_wg = self.config.reduce_wg as usize;
        let mut current = input;
        let mut use_a = true;
        loop {
            let n_groups = div_ceil(n, reduce_wg);
            let out = if n_groups == 1 {
                output
            } else if use_a {
                scratch_a
            } else {
                scratch_b
            };
            self.reduce_dev(n, current, out)?;
            if n_groups == 1 {
                return Ok(());
            }
            current = out;
            use_a = !use_a;
            n = n_groups;
        }
    }

    /// Enqueue Tr(KS) into a preallocated one-float device buffer.
    pub fn trace_ks_to_dev(
        &self,
        struct_: &GpuBsrStructure,
        ks: &Buffer<f32>,
        partial: &Buffer<f32>,
        scratch_a: &Buffer<f32>,
        scratch_b: &Buffer<f32>,
        output: &Buffer<f32>,
    ) -> Result<()> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n_groups = div_ceil(struct_.n_atom, reduce_wg);
        self.trace_ks_partial_dev(struct_.n_atom, &struct_.diag_block, ks, partial)?;
        self.reduce_to_output_dev(n_groups, partial, scratch_a, scratch_b, output)
    }

    /// Enqueue ||KSK-K||² into a preallocated one-float device buffer.
    pub fn idempotency_to_dev(
        &self,
        nblock: usize,
        q_ksk: &Buffer<f32>,
        k: &Buffer<f32>,
        partial: &Buffer<f32>,
        scratch_a: &Buffer<f32>,
        scratch_b: &Buffer<f32>,
        output: &Buffer<f32>,
    ) -> Result<()> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n_groups = div_ceil(nblock * BS2, reduce_wg);
        self.idempotency_partial_dev(nblock, q_ksk, k, partial)?;
        self.reduce_to_output_dev(n_groups, partial, scratch_a, scratch_b, output)
    }

    /// Enqueue direct ||A-I||² reduction into a device scalar.
    pub fn identity_residual_to_dev(
        &self,
        struct_: &GpuBsrStructure,
        a: &Buffer<f32>,
        partial: &Buffer<f32>,
        scratch_a: &Buffer<f32>,
        scratch_b: &Buffer<f32>,
        output: &Buffer<f32>,
    ) -> Result<()> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n_groups = div_ceil(struct_.nblock * BS2, reduce_wg);
        self.identity_residual_partial_dev(struct_.nblock, &struct_.diag_flag, a, partial)?;
        self.reduce_to_output_dev(n_groups, partial, scratch_a, scratch_b, output)
    }

    /// Device-resident full `Tr(KS)` reduction: enqueues partial + recursive
    /// reduction, returns one f32 to host. This is the **only** host transfer
    /// in the hot loop — a single scalar.
    pub fn trace_ks_dev(&self, struct_: &GpuBsrStructure, ks: &Buffer<f32>) -> Result<f32> {
        let nrow = struct_.n_atom;
        let reduce_wg = self.config.reduce_wg as usize;
        let n_groups = div_ceil(nrow, reduce_wg);
        let partial = self.zero_f32(n_groups)?;
        self.trace_ks_partial_dev(nrow, &struct_.diag_block, ks, &partial)?;
        self.reduce_to_one_dev(n_groups, &partial)
    }

    /// Device-resident `||KSK - K||_F`: enqueues partial + recursive
    /// reduction, returns one f32 to host.
    pub fn idempotency_err_dev(
        &self,
        nblock: usize,
        q_ksk: &Buffer<f32>,
        k: &Buffer<f32>,
    ) -> Result<f32> {
        let reduce_wg = self.config.reduce_wg as usize;
        let n = nblock * BS2;
        let n_groups = div_ceil(n, reduce_wg);
        let partial = self.zero_f32(n_groups)?;
        self.idempotency_partial_dev(nblock, q_ksk, k, &partial)?;
        let s2 = self.reduce_to_one_dev(n_groups, &partial)?;
        Ok(s2.sqrt())
    }

    /// Recursive reduction on device until one float remains. Reads that one
    /// float to host (the only host transfer).
    fn reduce_to_one_dev(&self, mut n: usize, input: &Buffer<f32>) -> Result<f32> {
        let reduce_wg = self.config.reduce_wg as usize;
        let mut current = input.clone();
        while n > 1 {
            let n_groups = div_ceil(n, reduce_wg);
            let out = self.zero_f32(n_groups)?;
            self.reduce_dev(n, &current, &out)?;
            current = out;
            n = n_groups;
        }
        let mut host = [0.0f32; 1];
        self.read_f32(&current, &mut host)?;
        Ok(host[0])
    }

    /// Device-resident Mulliken charges: `q_A = 2*Tr((KS)_AA)`.
    /// Reads `n_atom` floats to host.
    pub fn mulliken_dev(&self, struct_: &GpuBsrStructure, ks: &Buffer<f32>) -> Result<Vec<f32>> {
        let nrow = struct_.n_atom;
        let qbuf = self.zero_f32(nrow)?;
        let k = &self.k_mulliken_ks;
        k.set_arg(0, nrow as u32).map_err(map_ocl_err)?;
        k.set_arg(1, &struct_.diag_block).map_err(map_ocl_err)?;
        k.set_arg(2, ks).map_err(map_ocl_err)?;
        k.set_arg(3, &qbuf).map_err(map_ocl_err)?;
        unsafe {
            k.cmd().global_work_size(nrow).enq().map_err(map_ocl_err)?;
        }
        let mut q = vec![0.0f32; nrow];
        self.read_f32(&qbuf, &mut q)?;
        Ok(q)
    }

    /// Device-resident Frobenius norm of a values buffer.
    /// Uses `idempotency_err_dev` with a zero K buffer. Reads 1 scalar.
    pub fn frobenius_norm_dev(&self, nblock: usize, vals: &Buffer<f32>) -> Result<f32> {
        let zero = self.zero_f32(nblock * BS2)?;
        self.idempotency_err_dev(nblock, vals, &zero)
    }

    /// Device-resident direct identity residual `||A-I||_F` via the
    /// `bsr4_identity_residual_partial` kernel.  Avoids the catastrophic
    /// cancellation in `||A||² - 2·Tr(A) + N` when A ≈ I. Reads 1 scalar.
    pub fn identity_residual_scalar_dev(
        &self,
        struct_: &GpuBsrStructure,
        a: &Buffer<f32>,
    ) -> Result<f32> {
        let reduce_wg = self.config.reduce_wg as usize;
        let reduce_len = div_ceil(struct_.nblock * BS2, reduce_wg);
        let partial = self.zero_f32(reduce_len)?;
        let scratch_a = self.zero_f32(reduce_len)?;
        let scratch_b = self.zero_f32(reduce_len)?;
        let output = self.zero_f32(1)?;
        self.identity_residual_to_dev(struct_, a, &partial, &scratch_a, &scratch_b, &output)?;
        let mut host = [0.0f32; 1];
        self.read_f32(&output, &mut host)?;
        Ok(host[0])
    }

    // ------------------------------------------------------------------
    // GPT-5.6 #9: Device-side inf_norm, identity, scale.
    // These avoid downloading S and row_ptr/col_idx for NS Z0 init.
    // ------------------------------------------------------------------

    /// Compute ||A||_inf on the device (max absolute orbital-level row sum).
    /// Reads 1 scalar to host. Uses two kernels: row_abs_sum + reduce_max.
    pub fn inf_norm_dev(
        &self,
        struct_: &GpuBsrStructure,
        a: &Buffer<f32>,
    ) -> Result<f32> {
        let n_atom = struct_.n_atom;
        let n_orb = n_atom * BS;
        // Step 1: per-orbital absolute row sums.
        let row_sums = self.zero_f32(n_orb)?;
        let k = &self.k_row_abs_sum;
        k.set_arg(0, n_atom as u32).map_err(map_ocl_err)?;
        k.set_arg(1, &struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(2, &struct_.col_idx).map_err(map_ocl_err)?;
        k.set_arg(3, a).map_err(map_ocl_err)?;
        k.set_arg(4, &row_sums).map_err(map_ocl_err)?;
        unsafe {
            k.cmd().global_work_size(n_orb).enq().map_err(map_ocl_err)?;
        }
        // Step 2: max-reduce the row sums to a single scalar.
        let reduce_wg = self.config.reduce_wg as usize;
        let mut n = n_orb;
        let mut current = row_sums;
        loop {
            let n_groups = div_ceil(n, reduce_wg);
            let out = self.zero_f32(n_groups)?;
            let kr = &self.k_reduce_max;
            kr.set_arg(0, n as u32).map_err(map_ocl_err)?;
            kr.set_arg(1, &current).map_err(map_ocl_err)?;
            kr.set_arg(2, &out).map_err(map_ocl_err)?;
            unsafe {
                kr.cmd()
                    .global_work_size(n_groups * reduce_wg)
                    .local_work_size(reduce_wg)
                    .enq()
                    .map_err(map_ocl_err)?;
            }
            current = out;
            n = n_groups;
            if n <= 1 { break; }
        }
        let mut host = [0.0f32; 1];
        self.read_f32(&current, &mut host)?;
        Ok(host[0])
    }

    /// Build identity matrix on the device into `values` (on the given
    /// structure). Uses `diag_block` map — no host download of row_ptr/col_idx.
    pub fn build_identity_dev(
        &self,
        struct_: &GpuBsrStructure,
        values: &Buffer<f32>,
    ) -> Result<()> {
        let n_atom = struct_.n_atom;
        let k = &self.k_build_identity;
        k.set_arg(0, n_atom as u32).map_err(map_ocl_err)?;
        k.set_arg(1, &struct_.diag_block).map_err(map_ocl_err)?;
        k.set_arg(2, values).map_err(map_ocl_err)?;
        unsafe {
            k.cmd().global_work_size(n_atom * BS2).enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Scale all elements of a BSR4 values buffer by `alpha` on the device.
    pub fn scale_dev(
        &self,
        nblock: usize,
        alpha: f32,
        values: &Buffer<f32>,
    ) -> Result<()> {
        let total = nblock * BS2;
        let k = &self.k_scale;
        k.set_arg(0, nblock as u32).map_err(map_ocl_err)?;
        k.set_arg(1, alpha).map_err(map_ocl_err)?;
        k.set_arg(2, values).map_err(map_ocl_err)?;
        unsafe {
            k.cmd().global_work_size(total).enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // GPT-5.6 #19: Device-side Gershgorin spectral bounds.
    // ------------------------------------------------------------------

    /// Compute Gershgorin spectral bounds (emin, emax) of a BSR4 matrix
    /// on the device. Reads 2 scalars to host. No matrix download.
    pub fn gershgorin_bounds_dev(
        &self,
        struct_: &GpuBsrStructure,
        values: &Buffer<f32>,
    ) -> Result<(f32, f32)> {
        let n_atom = struct_.n_atom;
        let n_orb = n_atom * BS;
        // Step 1: per-orbital Gershgorin bounds.
        let emin_partial = self.zero_f32(n_orb)?;
        let emax_partial = self.zero_f32(n_orb)?;
        let k = &self.k_gershgorin_partial;
        k.set_arg(0, n_atom as u32).map_err(map_ocl_err)?;
        k.set_arg(1, &struct_.row_ptr).map_err(map_ocl_err)?;
        k.set_arg(2, &struct_.col_idx).map_err(map_ocl_err)?;
        k.set_arg(3, values).map_err(map_ocl_err)?;
        k.set_arg(4, &struct_.diag_flag).map_err(map_ocl_err)?;
        k.set_arg(5, &emin_partial).map_err(map_ocl_err)?;
        k.set_arg(6, &emax_partial).map_err(map_ocl_err)?;
        unsafe {
            k.cmd().global_work_size(n_orb).enq().map_err(map_ocl_err)?;
        }
        // Step 2: reduce emin (min) and emax (max) to single scalars.
        let emin = self.reduce_min_dev(n_orb, &emin_partial)?;
        let emax = self.reduce_max_scalar_dev(n_orb, &emax_partial)?;
        Ok((emin, emax))
    }

    /// Device-side min reduction to a single scalar. Reads 1 float to host.
    fn reduce_min_dev(&self, mut n: usize, input: &Buffer<f32>) -> Result<f32> {
        let reduce_wg = self.config.reduce_wg as usize;
        let mut current = input.clone();
        while n > 1 {
            let n_groups = div_ceil(n, reduce_wg);
            let out = self.zero_f32(n_groups)?;
            let k = &self.k_reduce_min;
            k.set_arg(0, n as u32).map_err(map_ocl_err)?;
            k.set_arg(1, &current).map_err(map_ocl_err)?;
            k.set_arg(2, &out).map_err(map_ocl_err)?;
            unsafe {
                k.cmd()
                    .global_work_size(n_groups * reduce_wg)
                    .local_work_size(reduce_wg)
                    .enq()
                    .map_err(map_ocl_err)?;
            }
            current = out;
            n = n_groups;
        }
        let mut host = [0.0f32; 1];
        self.read_f32(&current, &mut host)?;
        Ok(host[0])
    }

    /// Device-side max reduction to a single scalar. Reads 1 float to host.
    fn reduce_max_scalar_dev(&self, mut n: usize, input: &Buffer<f32>) -> Result<f32> {
        let reduce_wg = self.config.reduce_wg as usize;
        let mut current = input.clone();
        while n > 1 {
            let n_groups = div_ceil(n, reduce_wg);
            let out = self.zero_f32(n_groups)?;
            let k = &self.k_reduce_max;
            k.set_arg(0, n as u32).map_err(map_ocl_err)?;
            k.set_arg(1, &current).map_err(map_ocl_err)?;
            k.set_arg(2, &out).map_err(map_ocl_err)?;
            unsafe {
                k.cmd()
                    .global_work_size(n_groups * reduce_wg)
                    .local_work_size(reduce_wg)
                    .enq()
                    .map_err(map_ocl_err)?;
            }
            current = out;
            n = n_groups;
        }
        let mut host = [0.0f32; 1];
        self.read_f32(&current, &mut host)?;
        Ok(host[0])
    }

    /// Device-resident spectral bounds of B = Z·H with fractional padding.
    /// Computes B on device, then Gershgorin bounds on device, reads only
    /// 2 scalars (emin, emax) to host. No matrix download.
    ///
    /// `b` must be a preallocated scratch buffer on `t_struct` for B = Z·H.
    pub fn spectral_bounds_dev(
        &self,
        z: &GpuBsrMatrix,
        h: &GpuBsrMatrix,
        b: &GpuBsrMatrix,
        padding: f32,
    ) -> Result<(f32, f32)> {
        // B = Z·H on device (generic SpGEMM — Z·H not symmetric).
        self.spgemm_masked_dev(z, h, b)?;
        let (mut emin, mut emax) = self.gershgorin_bounds_dev(&b.struct_, &b.values)?;
        let span = (emax - emin).abs() * padding;
        emin -= span;
        emax += span;
        Ok((emin, emax))
    }

    /// Device-resident K0 construction:
    ///
    ///   K₀ = (εmax·Z - Z·H·Z) / (εmax - εmin)
    ///
    /// All products on device, no host roundtrip. Uses preallocated
    /// scratch buffers:
    /// - `b`: scratch for B = Z·H (on t_struct)
    /// - `a`: scratch for A = B·Z (on k_struct)
    /// - `k0`: output K0 (on k_struct)
    ///
    /// Returns nothing — K0 is written into `k0.values`. Caller should
    /// symmetrize if needed.
    pub fn build_k0_dev(
        &self,
        z: &GpuBsrMatrix,
        h: &GpuBsrMatrix,
        b: &GpuBsrMatrix,
        a: &GpuBsrMatrix,
        k0: &GpuBsrMatrix,
        emin: f32,
        emax: f32,
    ) -> Result<()> {
        let delta = (emax - emin).max(1e-12);
        let alpha_k = emax / delta;
        let beta_k = -1.0 / delta;

        // B = Z·H on device (generic SpGEMM — Z·H not symmetric).
        self.spgemm_masked_dev(z, h, b)?;

        // A = B·Z on device (Bsym — Z symmetric right operand).
        self.spgemm_bsym_dev(b, z, a)?;

        // K₀ = alpha_k * Z + beta_k * A  (axpby)
        let nblock = z.struct_.nblock;
        self.axpby_dev(nblock, alpha_k, &z.values, beta_k, &a.values, &k0.values)?;

        // Symmetrize K0.
        self.symmetrize_dev(nblock, &k0.struct_.transpose_block, &k0.values)?;

        Ok(())
    }

    /// Device-resident Newton-Schulz / Hotelling iteration for sparse
    /// approximate inverse:
    ///
    ///   Z_{n+1} = 2 Z_n - Z_n S Z_n
    ///
    /// with Z₀ = α·I, α = 1/||S||_∞.
    ///
    /// All matrices stay on the GPU for the entire loop. Per iteration:
    ///   T = Z·S          (spgemm_bsym_dev — 0 transfers)
    ///   ||I-T||_F        (identity_residual_scalar_dev — 1 scalar read)
    ///   R_Z = ||I-T||_F / sqrt(N_orb)  (host)
    ///   Q = T·Z          (spgemm_bsym_dev — 0 transfers)
    ///   Znew = 2·Z - Q   (axpby_dev — 0 transfers)
    ///   symmetrize(Znew) (symmetrize_dev — 0 transfers)
    ///   swap(Z, Znew)
    ///
    /// 2 SpGEMMs, 1 scalar read per iteration. No matrix host transfers.
    /// Returns (Z_host, final_R_Z, iterations).
    pub fn newton_schulz_inverse_dev(
        &self,
        s: &GpuBsrMatrix,
        k_struct: &Arc<GpuBsrStructure>,
        t_struct: &Arc<GpuBsrStructure>,
        max_iter: usize,
        tol: f32,
        stall: usize,
    ) -> Result<(Bsr4Matrix, f32, usize)> {
        let n_atom = s.struct_.n_atom;
        let n_orb = (n_atom * BS) as f32;
        let nblock = k_struct.nblock;

        // GPT-5.6 #9: Z₀ = α·I on M_K, α = 1/||S||_∞.
        // ||S||_∞ and identity construction happen entirely on the device —
        // no S download, no row_ptr/col_idx download, no host identity build.
        let s_inf = self.inf_norm_dev(&s.struct_, &s.values)?;
        if !s_inf.is_finite() || s_inf < 1e-30 {
            return Err(DftbError::InvalidInput(format!(
                "Newton-Schulz Z0 init: ||S||_inf = {s_inf:e} is non-finite or near-zero"
            )));
        }
        let alpha = 1.0 / s_inf;

        // Allocate persistent device buffers: Z, Znew, T, Q.
        let mut z = GpuBsrMatrix::zero(self, k_struct)?;
        let mut znew = GpuBsrMatrix::zero(self, k_struct)?;
        let t = GpuBsrMatrix::zero(self, t_struct)?;
        let q = GpuBsrMatrix::zero(self, k_struct)?;

        // Build identity on device, then scale by alpha.
        self.build_identity_dev(k_struct, &z.values)?;
        self.scale_dev(nblock, alpha, &z.values)?;

        let mut prev_rz = f32::INFINITY;
        let mut stall_count = 0;
        let mut iter_done = 0;

        for iter in 0..max_iter {
            iter_done = iter + 1;
            // T = Z·S (Bsym: S symmetric)
            self.spgemm_bsym_dev(&z, s, &t)?;

            // R_Z = ||I - T||_F / sqrt(N_orb) — direct identity residual,
            // avoids catastrophic cancellation in ||T||² - 2·Tr(T) + N.
            let rz = self.identity_residual_scalar_dev(t_struct, &t.values)? / n_orb.sqrt();

            println!(
                "  Newton-Schulz-dev iter {iter}: R_Z = {rz:e}"
            );

            if rz < tol {
                let z_out = z.to_host(self)?;
                return Ok((z_out, rz, iter_done));
            }
            if iter > 0 && rz > 0.9 * prev_rz {
                stall_count += 1;
                if stall_count >= stall {
                    println!("  Newton-Schulz-dev stalled after {iter_done} iters, R_Z = {rz:e}");
                    let z_out = z.to_host(self)?;
                    return Err(DftbError::InvalidInput(format!(
                        "Newton-Schulz did not converge: stalled after {iter_done} iters, R_Z={rz:e} (tol={tol:e})"
                    )));
                }
            } else {
                stall_count = 0;
            }
            prev_rz = rz;

            // Q = T·Z (Bsym: Z symmetric)
            self.spgemm_bsym_dev(&t, &z, &q)?;

            // Znew = 2*Z - Q
            self.axpby_dev(nblock, 2.0, &z.values, -1.0, &q.values, &znew.values)?;

            // Symmetrize Znew in place.
            self.symmetrize_dev(nblock, &k_struct.transpose_block, &znew.values)?;

            // Swap Z and Znew.
            std::mem::swap(&mut z.values, &mut znew.values);
        }

        // Final residual — direct identity residual, no cancellation.
        self.spgemm_bsym_dev(&z, s, &t)?;
        let rz = self.identity_residual_scalar_dev(t_struct, &t.values)? / n_orb.sqrt();
        let z_out = z.to_host(self)?;
        Err(DftbError::InvalidInput(format!(
            "Newton-Schulz did not converge: exhausted {max_iter} iters, final R_Z={rz:e} (tol={tol:e})"
        )))
    }
}

// ============================================================================
// Persistent sparse purification workspace (P0-D)
//
// All matrices stay on the GPU for the entire purification loop. The only
// host transfer per iteration is a single scalar `Tr(KS)` for the TC2 branch
// decision and optionally `R_I` for convergence checking.
// ============================================================================

/// Persistent device-resident workspace for TC2 / Newton–Schulz purification.
///
/// All buffers are allocated once at construction and reused for every
/// iteration. The structures (`k_struct`, `t_struct`) are shared `Arc`s
/// uploaded once. Kernel handles are borrowed from `SparseBsr4Gpu`.
///
/// See GPU_Optimization.chat.md §17–§19, §31.6 for the design rationale.
pub struct SparsePurifyWorkspace {
    /// Shared GPU runtime + cached kernels.
    gpu: SparseBsr4Gpu,
    /// Immutable structure for K, Q, Knew (all share the K mask).
    k_struct: Arc<GpuBsrStructure>,
    /// Immutable structure for T = K·S (the product mask, possibly different
    /// from K mask).
    t_struct: Arc<GpuBsrStructure>,
    /// S values (constant during purification).
    s: GpuBsrMatrix,
    /// K current values (on k_struct).
    k: GpuBsrMatrix,
    /// Knew output (on k_struct).
    knew: GpuBsrMatrix,
    /// T = K·S intermediate (on t_struct).
    t: GpuBsrMatrix,
    /// Q = T·K = KSK (on k_struct).
    q: GpuBsrMatrix,
    /// 1-float buffer for Tr(KS) — written by reduction, read to host for
    /// the TC2 branch decision.
    trace_buf: Buffer<f32>,
    /// 1-float device residual buffer.
    residual_buf: Buffer<f32>,
    /// Scratch buffer for reductions (reused across iterations).
    reduce_partial: Buffer<f32>,
    reduce_a: Buffer<f32>,
    reduce_b: Buffer<f32>,
    /// Number of occupied orbitals.
    nocc: f32,
    /// P4: Precomputed symbolic plans for the two recurring SpGEMMs.
    /// Built once at construction, reused across all TC2 iterations.
    /// plan_ks: T = K·S (A=K on k_mask, B=S on s_mask, C=T on t_mask)
    /// plan_tk: Q = T·K (A=T on t_mask, B=K on k_mask, C=Q on k_mask)
    plan_ks: Option<SpgemmPlanGpu>,
    plan_tk: Option<SpgemmPlanGpu>,
}

impl SparsePurifyWorkspace {
    /// Build a workspace from initial host matrices. `k0` and `s` must already
    /// be on their respective masks (`k_mask` for K, `s_mask` for S). The
    /// `t_mask` is the product mask `M_T = M_K ∘ M_HS`.
    ///
    /// **All structures and values are uploaded once here and never
    /// re-uploaded during purification.**
    ///
    /// P4: Symbolic SpGEMM plans for the two recurring products (K·S and
    /// T·K) are built here and reused across all TC2 iterations. If plan
    /// building fails (e.g., row degree overflow), the workspace falls
    /// back to the runtime intersection kernel.
    pub fn new(
        gpu: SparseBsr4Gpu,
        k0: &Bsr4Matrix,
        s: &Bsr4Matrix,
        k_mask: &(Vec<u32>, Vec<u32>),
        t_mask: &(Vec<u32>, Vec<u32>),
        nocc: f32,
    ) -> Result<Self> {
        let k_struct = Arc::new(GpuBsrStructure::new(&gpu, k0.n_atom, k_mask)?);
        let t_struct = Arc::new(GpuBsrStructure::new(&gpu, k0.n_atom, t_mask)?);

        // S lives on its own structure (which may differ from K). For TC2,
        // S is the right operand in T=K·S and must have its own structure.
        // However, the SpGEMM kernel reads B's row_ptr/col_idx from B's
        // structure, so S needs its own GpuBsrStructure.
        let s_struct = Arc::new(GpuBsrStructure::new(&gpu, s.n_atom, &(s.row_ptr.clone(), s.col_idx.clone()))?);
        let s_mat = GpuBsrMatrix { struct_: s_struct, values: gpu.buf_f32(&s.values)? };

        let k = GpuBsrMatrix { struct_: k_struct.clone(), values: gpu.buf_f32(&k0.values)? };
        let knew = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let t = GpuBsrMatrix::zero(&gpu, &t_struct)?;
        let q = GpuBsrMatrix::zero(&gpu, &k_struct)?;

        let trace_buf = gpu.zero_f32(1)?;
        let residual_buf = gpu.zero_f32(1)?;
        let nrow = k0.n_atom;
        let reduce_wg = gpu.config().reduce_wg as usize;
        let reduce_len = div_ceil(
            nrow.max(k_struct.nblock * BS2),
            reduce_wg,
        );
        let reduce_partial = gpu.zero_f32(reduce_len)?;
        let reduce_a = gpu.zero_f32(reduce_len)?;
        let reduce_b = gpu.zero_f32(reduce_len)?;

        // P4: Build symbolic plans for the two recurring SpGEMMs.
        // Plan for T = K·S: A=K (k_mask), B=S (s_mask, sym), C=T (t_mask).
        // Plan for Q = T·K: A=T (t_mask), B=K (k_mask, sym), C=Q (k_mask).
        let plan_ks = {
            let k_dummy = Bsr4Matrix::from_structure(k0.n_atom, k_mask.0.clone(), k_mask.1.clone())?;
            let s_dummy = Bsr4Matrix::from_structure(s.n_atom, s.row_ptr.clone(), s.col_idx.clone())?;
            match crate::methods::sparse::bsr4::build_spgemm_plan_bsym(&k_dummy, &s_dummy, t_mask) {
                Ok(plan) => Some(gpu.upload_plan(&plan)?),
                Err(e) => {
                    eprintln!("P4: plan_ks build failed, falling back to intersection kernel: {e}");
                    None
                }
            }
        };
        let plan_tk = {
            let t_dummy = Bsr4Matrix::from_structure(k0.n_atom, t_mask.0.clone(), t_mask.1.clone())?;
            let k_dummy = Bsr4Matrix::from_structure(k0.n_atom, k_mask.0.clone(), k_mask.1.clone())?;
            match crate::methods::sparse::bsr4::build_spgemm_plan_bsym(&t_dummy, &k_dummy, k_mask) {
                Ok(plan) => Some(gpu.upload_plan(&plan)?),
                Err(e) => {
                    eprintln!("P4: plan_tk build failed, falling back to intersection kernel: {e}");
                    None
                }
            }
        };

        Ok(Self {
            gpu, k_struct, t_struct, s: s_mat, k, knew, t, q,
            trace_buf, residual_buf, reduce_partial, reduce_a, reduce_b, nocc,
            plan_ks, plan_tk,
        })
    }

    /// Run one TC2 purification step on the device:
    ///
    /// ```text
    /// T = K·S          (SpGEMM #1, Bsym)
    /// Q = T·K          (SpGEMM #2, Bsym)
    /// n = Tr(T)        (reduction → 1 scalar to host)
    /// Knew = Q if n > Nocc else 2K−Q
    /// swap(K, Knew)
    /// ```
    ///
    /// **Exactly 2 SpGEMMs, zero matrix host transfers, 1 scalar read.**
    /// No `Kernel::builder()`, no `Buffer::builder()`, no `finish()` between
    /// kernels (only before the scalar read).
    ///
    /// Returns `Tr(KS)` for this iteration.
    fn tc2_products_dev(&mut self, with_residual: bool) -> Result<()> {
        // P4: Use symbolic plan if available; else fall back to intersection kernel.
        match &self.plan_ks {
            Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.k, &self.s, plan, &self.t)?,
            None => self.gpu.spgemm_bsym_dev(&self.k, &self.s, &self.t)?,
        }
        match &self.plan_tk {
            Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.t, &self.k, plan, &self.q)?,
            None => self.gpu.spgemm_bsym_dev(&self.t, &self.k, &self.q)?,
        }
        self.gpu.trace_ks_to_dev(
            &self.t_struct,
            &self.t.values,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &self.trace_buf,
        )?;
        if with_residual {
            self.gpu.idempotency_to_dev(
                self.k.struct_.nblock,
                &self.q.values,
                &self.k.values,
                &self.reduce_partial,
                &self.reduce_a,
                &self.reduce_b,
                &self.residual_buf,
            )?;
        }
        Ok(())
    }

    fn tc2_update_dev(&mut self) -> Result<()> {
        let nblock = self.k.struct_.nblock;
        self.gpu.tc2_dev(
            nblock,
            &self.k.values,
            &self.q.values,
            &self.trace_buf,
            self.nocc,
            &self.knew.values,
        )?;
        self.gpu.symmetrize_dev(
            nblock,
            &self.k_struct.transpose_block,
            &self.knew.values,
        )?;
        std::mem::swap(&mut self.k.values, &mut self.knew.values);
        Ok(())
    }

    pub fn tc2_step_dev(&mut self) -> Result<f32> {
        self.tc2_products_dev(false)?;
        self.tc2_update_dev()?;
        let mut tr = [0.0f32; 1];
        self.gpu.read_f32(&self.trace_buf, &mut tr)?;
        if !tr[0].is_finite() {
            return Err(DftbError::InvalidInput(format!(
                "TC2 trace is non-finite after update: Tr(KS)={}",
                tr[0]
            )));
        }
        Ok(tr[0])
    }

    /// Compute `R_I = ||KSK - K||_F` on the device. Q already contains KSK
    /// from the last `tc2_step_dev`. This reads 1 scalar to host.
    ///
    /// If `recompute_q` is true, recompute Q = T·K first (needed if called
    /// standalone without a preceding step).
    pub fn idempotency_err_dev(&self, recompute_q: bool) -> Result<f32> {
        let nblock = self.k.struct_.nblock;
        if recompute_q {
            // T = K·S, Q = T·K
            self.gpu.spgemm_bsym_dev(&self.k, &self.s, &self.t)?;
            self.gpu.spgemm_bsym_dev(&self.t, &self.k, &self.q)?;
        }
        self.gpu.idempotency_err_dev(nblock, &self.q.values, &self.k.values)
    }

    /// Compute `Tr(KS)` for the current K using the workspace's persistent
    /// buffers.  Requires `self.t` to already contain K·S (e.g. after
    /// `tc2_products_dev` or `idempotency_err_dev(true)`).  Reads 1 scalar.
    fn trace_ks_current_dev(&self) -> Result<f32> {
        self.gpu.trace_ks_to_dev(
            &self.t_struct,
            &self.t.values,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &self.trace_buf,
        )?;
        let mut tr = [0.0f32; 1];
        self.gpu.read_f32(&self.trace_buf, &mut tr)?;
        Ok(tr[0])
    }

    /// Run the full TC2 purification loop on the device. Returns
    /// `(K_final_host, final_R_I, final_Tr, iterations, history)`.
    ///
    /// **A2 fix (GPT-5.6 issue #3, manifest §4.7):** Residual is computed
    /// BEFORE the update, using the Q already computed from the 2 SpGEMMs.
    /// No extra KSK for convergence check. No per-iteration host sync.
    ///
    /// Normal iteration: 2 SpGEMMs, 0 host reads, 0 queue finish.
    /// Diagnostic iteration: 2 SpGEMMs + 2 cheap reductions + 1 combined
    /// scalar read (trace + residual).
    ///
    /// If residual < tol on a diagnostic iteration, returns OLD K without
    /// swapping to the unnecessary next iterate.
    pub fn tc2_purify_dev(
        &mut self,
        max_iter: usize,
        tol: f32,
        check_every: usize,
    ) -> Result<(Bsr4Matrix, f32, f32, usize, Vec<(usize, f32, f32)>)> {
        let mut history: Vec<(usize, f32, f32)> = Vec::new();
        let mut best_r_i = f32::INFINITY;
        let mut best_iter = 0usize;
        let mut last_tr = 0.0f32;
        let mut last_r_i = f32::INFINITY;

        for iter in 0..max_iter {
            // Compute T=K·S, Q=T·K=KSK, trace=Tr(T) on device.
            // On diagnostic iterations, also compute R_I=||Q-K|| on device.
            // Both use the SAME Q from the SAME 2 SpGEMMs — no extra products.
            let do_check = (iter % check_every == 0) || (iter == max_iter - 1);
            self.tc2_products_dev(do_check)?;

            if do_check {
                // Read 2 scalars: trace and residual (the only host sync).
                let mut tr = [0.0f32; 1];
                self.gpu.read_f32(&self.trace_buf, &mut tr)?;
                if !tr[0].is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "TC2 trace non-finite at iter {iter}: Tr(KS)={}", tr[0]
                    )));
                }
                let mut ri_sq = [0.0f32; 1];
                self.gpu.read_f32(&self.residual_buf, &mut ri_sq)?;
                let ri = ri_sq[0].sqrt();
                if !ri.is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "TC2 residual non-finite at iter {iter}: R_I={}", ri
                    )));
                }
                last_tr = tr[0];
                last_r_i = ri;
                println!("  TC2-dev iter {iter}: R_I={ri:e}  Tr(KS)={:.6}  (Nocc={})", tr[0], self.nocc);
                history.push((iter, ri, tr[0]));
                if ri < best_r_i {
                    best_r_i = ri;
                    best_iter = iter;
                }

                // Convergence: return OLD K (before update) — it is already
                // good enough. No swap needed.
                if ri < tol {
                    if (tr[0] - self.nocc).abs() <= TC2_TRACE_TOL {
                        let k_host = self.k.to_host(&self.gpu)?;
                        return Ok((k_host, ri, tr[0], iter + 1, history));
                    }
                    println!(
                        "  TC2-dev R_I={ri:e} < tol but Tr(KS)={:.6} != Nocc={} (wrong-rank projector not accepted)",
                        tr[0], self.nocc
                    );
                }

                // Divergence detection.
                if ri > best_r_i * 10.0 && best_r_i < f32::INFINITY {
                    println!(
                        "  TC2-dev diverging at iter {iter}: R_I={ri:e} > 10×best={best_r_i:e}"
                    );
                    let k_host = self.k.to_host(&self.gpu)?;
                    return Err(DftbError::InvalidInput(format!(
                        "TC2 purification did not converge: diverged at iter {iter}, best R_I={best_r_i:e} at iter {best_iter} (tol={tol:e}, max_iter={max_iter})"
                    )));
                }
            }

            // Update K: Knew = TC2_branch(K, Q, trace, Nocc), symmetrize, swap.
            // trace_buf is already on device — no host read needed.
            self.tc2_update_dev()?;
        }

        // Exhausted max_iter without convergence.
        let k_host = self.k.to_host(&self.gpu)?;
        let final_r_i = if last_r_i.is_nan() || last_r_i.is_infinite() {
            // Never checked — compute now.
            self.idempotency_err_dev(true)?
        } else {
            last_r_i
        };
        let final_tr = if last_r_i.is_nan() || last_r_i.is_infinite() {
            self.gpu.spgemm_bsym_dev(&self.k, &self.s, &self.t)?;
            self.trace_ks_current_dev()?
        } else {
            last_tr
        };
        if best_r_i < final_r_i {
            println!(
                "  TC2-dev exhausted {} iters, best R_I={best_r_i:e} at iter {best_iter}",
                max_iter
            );
            Err(DftbError::InvalidInput(format!(
                "TC2 purification did not converge: exhausted {max_iter} iters, best R_I={best_r_i:e} at iter {best_iter} (tol={tol:e})"
            )))
        } else {
            Err(DftbError::InvalidInput(format!(
                "TC2 purification did not converge: exhausted {max_iter} iters, final R_I={final_r_i:e} (tol={tol:e})"
            )))
        }
    }

    /// Borrow the underlying GPU.
    pub fn gpu(&self) -> &SparseBsr4Gpu { &self.gpu }

    /// Read the current K back to host (blocking).
    pub fn k_to_host(&self) -> Result<Bsr4Matrix> {
        self.k.to_host(&self.gpu)
    }

    /// Read current Mulliken charges `q_A = 2*Tr((KS)_AA)` from the device.
    /// Computes T=K·S first, then reads `n_atom` floats.
    pub fn mulliken_dev(&mut self) -> Result<Vec<f32>> {
        // T = K·S
        self.gpu.spgemm_bsym_dev(&self.k, &self.s, &self.t)?;
        self.gpu.mulliken_dev(&self.t_struct, &self.t.values)
    }
}
