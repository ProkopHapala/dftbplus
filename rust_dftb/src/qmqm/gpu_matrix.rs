//! GPU matrix operations harness for the DFTB SCF cycle.
//!
//! This module provides a Rust API over OpenCL kernels for the core
//! linear algebra operations needed in the self-consistent charge (SCC)
//! cycle on GPU. It is designed for batched processing of many small
//! fragment matrices (N = 100–1000) in parallel.
//!
//! # Architecture
//!
//! The generalized eigenvalue problem H·C = S·C·ε is solved via one
//! of two strategies:
//!
//! - **Purification path** (density matrix only, no orbitals):
//!   1. Löwdin transform: H' = X^T·H·X where X = S^{-1/2} (two GEMMs)
//!   2. Initial guess: D = I − (H' − λ_min·I)/(λ_max − λ_min)
//!   3. Iterate: D ← 3D²−2D³ (McWeeny) or D²/2D−D² (TC2) until idempotent
//!   4. Convergence check: ||D²−D||_F < tol
//!
//! - **Direct diagonalization path** (yields orbitals):
//!   1. Löwdin transform (same as above)
//!   2. Block Jacobi on H': partition into T×T blocks, diagonalize
//!      2T×2T compound blocks in local memory, apply rotations via GEMM
//!   3. Back-transform: C = X·C'
//!
//! # Kernel template
//!
//! The OpenCL source (`gpu_matrix_ops.cl`) uses `#define` constants
//! for tile sizes and workgroup dimensions. `MatrixKernelConfig::
//! render_source()` performs text substitution to specialize the
//! kernels for a given GPU architecture and problem size, avoiding
//! the overhead of runtime parameters in inner loops.
//!
//! # Kernels (via GpuMatrixContext methods)
//!
//! - `batched_gemm` — tiled batched C = α·A·B + β·C
//! - `lowdin_transform` — H' = X^T·H·X (two GEMMs)
//! - `scale_density_guess` — initial D from Gershgorin-bounded H
//! - `purify_palser_manolopoulos` — McWeeny iteration D ← 3D²−2D³
//! - `purify_trace_correcting` — TC2 with per-batch trace control
//! - `trace` — per-batch Tr(A)
//! - `idempotency_error` — per-batch ||D²−D||_F
//! - `local_jacobi_blocks` — serial local Jacobi for small blocks
//! - `local_jacobi_blocks_parallel` — row-parallel local Jacobi
//! - `brent_luk_rounds` — block-pair schedule for block Jacobi

use crate::core::error::{DftbError, Result};
use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue, SpatialDims};

const MATRIX_KERNEL_TEMPLATE: &str = include_str!("gpu_matrix_ops.cl");

/// Tunable parameters for GPU matrix kernels.
///
/// These values are baked into the OpenCL source at compile time via
/// text substitution. Different GPU architectures and problem sizes
/// benefit from different tile sizes and workgroup dimensions.
#[derive(Debug, Clone)]
pub struct MatrixKernelConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub reduce_wg: usize,
    pub jacobi_max_m: usize,
}

impl MatrixKernelConfig {
    /// Default configuration tuned for NVIDIA GPUs (16×16 tiles, 256-thread reductions).
    pub fn nvidia_default() -> Self {
        Self {
            tile_m: 16,
            tile_n: 16,
            tile_k: 32,
            reduce_wg: 256,
            jacobi_max_m: 64,
        }
    }

    /// Configuration for smaller matrices (N < 256): smaller tiles reduce local memory pressure.
    pub fn small_matrix() -> Self {
        Self {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            reduce_wg: 128,
            jacobi_max_m: 32,
        }
    }

    /// Render the OpenCL template with this configuration's parameters substituted in.
    pub fn render_source(&self) -> String {
        MATRIX_KERNEL_TEMPLATE
            .replace("#define TILE_M 16", &format!("#define TILE_M {}", self.tile_m))
            .replace("#define TILE_N 16", &format!("#define TILE_N {}", self.tile_n))
            .replace("#define TILE_K 32", &format!("#define TILE_K {}", self.tile_k))
            .replace("#define WG_REDUCE 256", &format!("#define WG_REDUCE {}", self.reduce_wg))
            .replace("#define JACOBI_MAX_M 64", &format!("#define JACOBI_MAX_M {}", self.jacobi_max_m))
    }
}

/// Whether to transpose a matrix operand in GEMM.
#[derive(Debug, Clone, Copy)]
pub enum Transpose {
    No,
    Yes,
}

impl Transpose {
    fn as_i32(self) -> i32 {
        match self {
            Self::No => 0,
            Self::Yes => 1,
        }
    }
}

/// One round of block Jacobi: a set of non-overlapping block pairs to rotate simultaneously.
#[derive(Debug, Clone)]
pub struct BlockJacobiRound {
    pub pairs: Vec<(usize, usize)>,
}

/// Generate the Brent–Luk round schedule for block Jacobi diagonalization.
///
/// In each round, the N blocks are paired so that no block appears in
/// more than one pair. Over N−1 rounds, every pair of blocks is visited
/// exactly once. This is the standard parallel ordering for cyclic
/// Jacobi on a block-partitioned matrix, ensuring that all rotations
/// within a round are independent and can be applied in parallel.
///
/// For odd N, a dummy slot is added so the algorithm works on an even
/// number of slots; the dummy pairs are simply skipped.
pub fn brent_luk_rounds(n_blocks: usize) -> Vec<BlockJacobiRound> {
    if n_blocks < 2 {
        return Vec::new();
    }
    let mut idx: Vec<Option<usize>> = (0..n_blocks).map(Some).collect();
    if n_blocks % 2 == 1 {
        idx.push(None);
    }
    let m = idx.len();
    let mut rounds = Vec::with_capacity(m - 1);
    for _ in 0..m - 1 {
        let mut pairs = Vec::with_capacity(m / 2);
        for i in 0..m / 2 {
            if let (Some(a), Some(b)) = (idx[i], idx[m - 1 - i]) {
                pairs.push((a.min(b), a.max(b)));
            }
        }
        rounds.push(BlockJacobiRound { pairs });
        let last = idx.pop().unwrap();
        idx.insert(1, last);
    }
    rounds
}

/// OpenCL context and compiled kernel set for batched matrix operations.
///
/// Holds the OpenCL context, command queue, and compiled program.
/// All kernel launch methods operate on `Buffer<f32>` inputs that the
/// caller allocates via `buffer_from_slice` or `zero_buffer`.
pub struct GpuMatrixContext {
    pub config: MatrixKernelConfig,
    context: Context,
    queue: Queue,
    program: Program,
}

impl GpuMatrixContext {
    /// Create a new GPU context: selects the first available device,
    /// compiles the kernel template with the given configuration.
    pub fn new(config: MatrixKernelConfig) -> Result<Self> {
        let platform = Platform::default();
        let device = Device::first(platform).map_err(map_ocl_err)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .map_err(map_ocl_err)?;
        let queue = Queue::new(&context, device.clone(), None).map_err(map_ocl_err)?;
        let source = config.render_source();
        let program = Program::builder()
            .devices(device)
            .src(source)
            .build(&context)
            .map_err(map_ocl_err)?;
        Ok(Self { config, context, queue, program })
    }

    /// Allocate a GPU buffer initialized from a host slice.
    pub fn buffer_from_slice(&self, data: &[f32]) -> Result<Buffer<f32>> {
        Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .map_err(map_ocl_err)
    }

    /// Allocate a zero-filled GPU buffer of the given length.
    pub fn zero_buffer(&self, len: usize) -> Result<Buffer<f32>> {
        Buffer::<f32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .fill_val(0.0f32)
            .build()
            .map_err(map_ocl_err)
    }

    /// Copy a GPU buffer back to host memory (blocking).
    pub fn read_buffer(&self, buf: &Buffer<f32>, out: &mut [f32]) -> Result<()> {
        buf.read(out).enq().map_err(map_ocl_err)?;
        self.queue.finish().map_err(map_ocl_err)
    }

    /// Batched tiled matrix multiply: C_b = α·op(A_b)·op(B_b) + β·C_b.
    ///
    /// This is the fundamental building block for all matrix operations
    /// in the SCF cycle. Each batch element is an N×N matrix; the third
    /// NDRange dimension indexes the batch. Tiles of A and B are loaded
    /// into local memory for high arithmetic intensity.
    pub fn batched_gemm(
        &self,
        n: usize,
        batch: usize,
        trans_a: Transpose,
        trans_b: Transpose,
        alpha: f32,
        beta: f32,
        a: &Buffer<f32>,
        b: &Buffer<f32>,
        c: &Buffer<f32>,
    ) -> Result<()> {
        let row_groups = div_ceil(n, self.config.tile_m);
        let col_groups = div_ceil(n, self.config.tile_n);
        let global = SpatialDims::Three(
            row_groups * self.config.tile_n,
            col_groups * self.config.tile_m,
            batch,
        );
        let local = SpatialDims::Two(self.config.tile_n, self.config.tile_m);
        let a_local = self.config.tile_m * self.config.tile_k;
        let b_local = self.config.tile_k * self.config.tile_n;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("batched_gemm")
            .queue(self.queue.clone())
            .global_work_size(global)
            .local_work_size(local)
            .arg(n as i32)
            .arg(batch as i32)
            .arg(trans_a.as_i32())
            .arg(trans_b.as_i32())
            .arg(alpha)
            .arg(beta)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg_local::<f32>(a_local)
            .arg_local::<f32>(b_local)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { kernel.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Löwdin orthogonalization transform: H' = X^T·H·X.
    ///
    /// Converts the generalized eigenvalue problem H·C = S·C·ε into a
    /// standard one H'·C' = C'·ε, where X = S^{-1/2}. Implemented as
    /// two batched GEMMs: scratch = X^T·H, then out = scratch·X.
    /// The result H' is symmetric and can be diagonalized directly.
    pub fn lowdin_transform(
        &self,
        n: usize,
        batch: usize,
        x: &Buffer<f32>,
        h: &Buffer<f32>,
        scratch: &Buffer<f32>,
        out: &Buffer<f32>,
    ) -> Result<()> {
        self.batched_gemm(n, batch, Transpose::Yes, Transpose::No, 1.0, 0.0, x, h, scratch)?;
        self.batched_gemm(n, batch, Transpose::No, Transpose::No, 1.0, 0.0, scratch, x, out)
    }

    /// Compute the initial density matrix guess for purification.
    ///
    /// Given H and its Gershgorin bounds [λ_min, λ_max] per batch,
    /// produces D = I − (H − λ_min·I)/(λ_max − λ_min), mapping the
    /// spectrum into [0,1] so purification can converge.
    pub fn scale_density_guess(
        &self,
        n: usize,
        batch: usize,
        h: &Buffer<f32>,
        d: &Buffer<f32>,
        bounds: &Buffer<f32>,
    ) -> Result<()> {
        let total = n * n * batch;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("scale_density_guess")
            .queue(self.queue.clone())
            .global_work_size(total)
            .arg(n as i32)
            .arg(batch as i32)
            .arg(h)
            .arg(d)
            .arg(bounds)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { kernel.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// McWeeny purification: iterate D ← 3D²−2D³ until idempotent.
    ///
    /// Each iteration requires two GEMMs (D² = D·D, D³ = D²·D) and
    /// one elementwise update. Converges quadratically when all
    /// eigenvalues of D are in [0,1]. Preserves the trace of the
    /// initial guess. Does NOT yield orbitals — only the density matrix.
    pub fn purify_palser_manolopoulos(
        &self,
        n: usize,
        batch: usize,
        d: &Buffer<f32>,
        d2: &Buffer<f32>,
        d3: &Buffer<f32>,
        n_iter: usize,
    ) -> Result<()> {
        for _ in 0..n_iter {
            self.batched_gemm(n, batch, Transpose::No, Transpose::No, 1.0, 0.0, d, d, d2)?;
            self.batched_gemm(n, batch, Transpose::No, Transpose::No, 1.0, 0.0, d2, d, d3)?;
            self.purify_mcweeny(n, batch, d2, d3, d)?;
        }
        Ok(())
    }

    /// Trace-correcting (TC2) purification with per-batch electron count control.
    ///
    /// At each step, computes D² = D·D, then checks Tr(D) against the
    /// target electron count for each fragment. If Tr(D) is too large,
    /// applies D ← D²; if too small, applies D ← 2D − D². This keeps
    /// Tr(D) exactly at the electron count at every step, unlike McWeeny.
    /// Requires a host readback of traces per iteration (one float per batch).
    pub fn purify_trace_correcting(
        &self,
        n: usize,
        batch: usize,
        d: &Buffer<f32>,
        d2: &Buffer<f32>,
        traces: &Buffer<f32>,
        electron_counts: &[f32],
        n_iter: usize,
    ) -> Result<()> {
        if electron_counts.len() != batch {
            return Err(DftbError::InvalidInput(format!(
                "electron_counts len {} != batch {}",
                electron_counts.len(), batch
            )));
        }
        let mut trace_host = vec![0.0f32; batch];
        for _ in 0..n_iter {
            self.batched_gemm(n, batch, Transpose::No, Transpose::No, 1.0, 0.0, d, d, d2)?;
            self.trace(n, batch, d, traces)?;
            self.read_buffer(traces, &mut trace_host)?;
            for ib in 0..batch {
                let mode = if trace_host[ib] > electron_counts[ib] { 0 } else { 1 };
                self.purify_tc2_single_batch(n, ib, mode, d, d2, d)?;
            }
        }
        Ok(())
    }

    /// Compute the trace of each batch element: Tr(A_b) = Σ_i A_b[i,i].
    ///
    /// One workgroup per batch element with tree reduction in local memory.
    /// Result is one f32 per batch written to `traces`.
    pub fn trace(&self, n: usize, batch: usize, a: &Buffer<f32>, traces: &Buffer<f32>) -> Result<()> {
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("trace_reduce")
            .queue(self.queue.clone())
            .global_work_size(batch * self.config.reduce_wg)
            .local_work_size(self.config.reduce_wg)
            .arg(n as i32)
            .arg(batch as i32)
            .arg(a)
            .arg(traces)
            .arg_local::<f32>(self.config.reduce_wg)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { kernel.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Compute the idempotency error ||D²−D||_F for each batch element.
    ///
    /// This is the convergence criterion for purification: when the
    /// Frobenius norm of D²−D is below tolerance, D is idempotent and
    /// the density matrix is converged.
    pub fn idempotency_error(
        &self,
        n: usize,
        batch: usize,
        d: &Buffer<f32>,
        d2: &Buffer<f32>,
        errs: &Buffer<f32>,
    ) -> Result<()> {
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("idempotency_reduce")
            .queue(self.queue.clone())
            .global_work_size(batch * self.config.reduce_wg)
            .local_work_size(self.config.reduce_wg)
            .arg(n as i32)
            .arg(batch as i32)
            .arg(d)
            .arg(d2)
            .arg(errs)
            .arg_local::<f32>(self.config.reduce_wg)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { kernel.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Diagonalize small symmetric matrices via classical cyclic Jacobi.
    ///
    /// One workgroup per block; the entire m×m matrix is loaded into
    /// local memory and a single thread performs the Jacobi sweeps.
    /// Suitable for the 2T×2T compound blocks in block Jacobi, or for
    /// direct diagonalization of small fragment Hamiltonians (m ≤ 64).
    /// Returns eigenvalues and eigenvectors.
    pub fn local_jacobi_blocks(
        &self,
        m: usize,
        n_blocks: usize,
        blocks: &Buffer<f32>,
        eigvals: &Buffer<f32>,
        eigvecs: &Buffer<f32>,
        max_sweeps: usize,
        tol: f32,
    ) -> Result<()> {
        if m > self.config.jacobi_max_m {
            return Err(DftbError::InvalidInput(format!(
                "local_jacobi_blocks: m={} exceeds jacobi_max_m={}",
                m, self.config.jacobi_max_m
            )));
        }
        let local_elems = m * m;
        let wg = m.next_power_of_two().min(256).max(1);
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("local_jacobi_blocks")
            .queue(self.queue.clone())
            .global_work_size(n_blocks * wg)
            .local_work_size(wg)
            .arg(m as i32)
            .arg(max_sweeps as i32)
            .arg(tol)
            .arg(blocks)
            .arg(eigvals)
            .arg(eigvecs)
            .arg_local::<f32>(local_elems)
            .arg_local::<f32>(local_elems)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { kernel.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Diagonalize small symmetric matrices via row-parallel Jacobi.
    ///
    /// Same purpose as `local_jacobi_blocks` but all workgroup threads
    /// collaborate on applying each Givens rotation: thread 0 computes
    /// the angle, broadcasts it, then each thread updates its row.
    /// Faster than the serial variant when m is large enough to saturate
    /// the workgroup. Based on the block_jacobi_padded pattern from
    /// nested_solver.py.
    pub fn local_jacobi_blocks_parallel(
        &self,
        m: usize,
        n_blocks: usize,
        blocks: &Buffer<f32>,
        eigvals: &Buffer<f32>,
        eigvecs: &Buffer<f32>,
        max_sweeps: usize,
        tol: f32,
    ) -> Result<()> {
        if m > self.config.jacobi_max_m {
            return Err(DftbError::InvalidInput(format!(
                "local_jacobi_blocks_parallel: m={} exceeds jacobi_max_m={}",
                m, self.config.jacobi_max_m
            )));
        }
        let local_elems = m * m;
        let wg = m.next_power_of_two().min(256).max(2);
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("local_jacobi_blocks_parallel")
            .queue(self.queue.clone())
            .global_work_size(n_blocks * wg)
            .local_work_size(wg)
            .arg(m as i32)
            .arg(max_sweeps as i32)
            .arg(tol)
            .arg(blocks)
            .arg(eigvals)
            .arg(eigvecs)
            .arg_local::<f32>(local_elems)
            .arg_local::<f32>(local_elems)
            .arg_local::<f32>(wg)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { kernel.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Internal: launch the McWeeny elementwise kernel D ← 3D²−2D³.
    fn purify_mcweeny(
        &self,
        n: usize,
        batch: usize,
        d2: &Buffer<f32>,
        d3: &Buffer<f32>,
        d: &Buffer<f32>,
    ) -> Result<()> {
        let total = n * n * batch;
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("purify_mcweeny")
            .queue(self.queue.clone())
            .global_work_size(total)
            .arg(n as i32)
            .arg(batch as i32)
            .arg(d2)
            .arg(d3)
            .arg(d)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { kernel.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Internal: launch TC2 step for a single batch element.
    fn purify_tc2_single_batch(
        &self,
        n: usize,
        batch_index: usize,
        mode: i32,
        d: &Buffer<f32>,
        d2: &Buffer<f32>,
        out: &Buffer<f32>,
    ) -> Result<()> {
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("purify_tc2_one_batch")
            .queue(self.queue.clone())
            .global_work_size(n * n)
            .arg(n as i32)
            .arg(batch_index as i32)
            .arg(mode)
            .arg(d)
            .arg(d2)
            .arg(out)
            .build()
            .map_err(map_ocl_err)?;
        unsafe { kernel.enq().map_err(map_ocl_err)?; }
        Ok(())
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn map_ocl_err(err: ocl::Error) -> DftbError {
    DftbError::InvalidInput(format!("OpenCL error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_luk_rounds_cover_all_pairs_even() {
        let rounds = brent_luk_rounds(4);
        assert_eq!(rounds.len(), 3);
        let mut pairs = rounds.into_iter().flat_map(|r| r.pairs).collect::<Vec<_>>();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn brent_luk_rounds_cover_all_pairs_odd() {
        let rounds = brent_luk_rounds(5);
        assert_eq!(rounds.len(), 5);
        let mut pairs = rounds.into_iter().flat_map(|r| r.pairs).collect::<Vec<_>>();
        pairs.sort_unstable();
        assert_eq!(pairs.len(), 10);
        assert_eq!(pairs[0], (0, 1));
        assert_eq!(pairs[9], (3, 4));
    }

    #[test]
    fn template_replaces_parameters() {
        let cfg = MatrixKernelConfig { tile_m: 8, tile_n: 16, tile_k: 64, reduce_wg: 128, jacobi_max_m: 32 };
        let src = cfg.render_source();
        assert!(src.contains("#define TILE_M 8"));
        assert!(src.contains("#define TILE_N 16"));
        assert!(src.contains("#define TILE_K 64"));
        assert!(src.contains("#define WG_REDUCE 128"));
        assert!(src.contains("#define JACOBI_MAX_M 32"));
    }
}
