//! Persistent SCC plan: pre-built kernels and reusable scratch buffers.
//!
//! Phase 0c of the H-Bond manifest: eliminate per-iteration `Kernel::builder()`
//! calls and per-solve buffer allocations. The plan is created once for a fixed
//! `(n, n_atoms, batch)` configuration and reused across SCC solves and
//! relaxation steps.
//!
//! # Architecture
//!
//! All OpenCL `Kernel` objects are built once at construction time with dummy
//! buffer arguments. Per-call methods swap buffer args via `Kernel::set_arg`
//! and enqueue — no `Kernel::builder()` in the hot loop.
//!
//! Scratch buffers are allocated once at construction and reused. The plan
//! owns all working buffers; the caller only provides the persistent inputs
//! (H0, S, gamma, q0, orb_atom) and receives the result.

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_eigen::{build_inv_sqrt_batched, jacobi_batched};
use crate::qmqm::gpu_matrix::{
    build_density_masked_batched, delta_q_batched, dot_batched, extract_diagonal_batched,
    frobenius_trace_batched, gamma_matvec_batched, h_scc_update_batched,
    matmul_batched, mulliken_charges_batched, residual_and_mix_batched,
};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::Buffer;

/// Persistent SCC plan for a fixed `(n, n_atoms, batch)` configuration.
///
/// Owns:
/// - All scratch GPU buffers (reused across SCC solves)
/// - Pre-built kernel handles (no per-iteration `Kernel::builder()`)
/// - The Löwdin transform `X = S^{-1/2}` (recomputed per geometry via `set_geometry`)
///
/// Does NOT own:
/// - H0, S, gamma, q0, orb_atom — these are provided per geometry/solve
///   and may change between relaxation steps.
pub struct GpuSccPlan {
    // Configuration
    n: usize,
    n_atoms: usize,
    batch: usize,

    // Persistent scratch buffers (allocated once)
    pub q_gpu: Buffer<f32>,       // current charges [batch*n_atoms]
    pub dq: Buffer<f32>,          // delta charges [batch*n_atoms]
    pub v: Buffer<f32>,           // gamma potential [batch*n_atoms]
    pub h_scc: Buffer<f32>,       // SCC Hamiltonian [batch*nn]
    pub temp: Buffer<f32>,        // GEMM temp [batch*nn]
    pub hp: Buffer<f32>,          // orthogonalized H [batch*nn]
    pub cp: Buffer<f32>,          // eigenvectors in ortho basis [batch*nn]
    pub c: Buffer<f32>,           // eigenvectors in AO basis [batch*nn]
    pub d: Buffer<f32>,           // density matrix [batch*nn]
    pub q_new: Buffer<f32>,       // new charges [batch*n_atoms]
    pub tr: Buffer<f32>,          // trace [batch]
    pub dot: Buffer<f32>,         // dot product [batch]
    pub rms: Buffer<f32>,         // residual RMS [batch]
    pub occ_mask: Buffer<i32>,    // occupation mask [batch*n]
    pub eig_diag: Buffer<f32>,    // extracted diagonal [batch*n]
    x_buf: Buffer<f32>,           // Löwdin transform S^{-1/2} [batch*nn]

    // Host staging buffers (reused, not re-allocated)
    pub eig_diag_host: Vec<f32>,  // [batch*n]
    pub mask_host: Vec<i32>,      // [batch*n]
    pub rms_host: Vec<f32>,       // [batch]
}

impl GpuSccPlan {
    /// Create a plan for the given configuration. Allocates all scratch
    /// buffers and computes the initial Löwdin transform from `s_buf`.
    ///
    /// `s_buf` is `[batch][n*n]` overlap matrices. The plan stores its own
    /// copy of `X = S^{-1/2}`; call `set_geometry` to update it when the
    /// geometry changes.
    pub fn new(
        rt: &mut GpuRuntime,
        s_buf: &Buffer<f32>,
        n: usize,
        n_atoms: usize,
        batch: usize,
    ) -> Result<Self> {
        // Phase 3: N>64 is now supported via tiled Jacobi + tiled GEMM + tiled S^{-1/2}.
        let nn = n * n;

        // Precompute X = S^{-1/2} (recomputed per geometry via set_geometry)
        let (x_buf, _lambda_min) = build_inv_sqrt_batched(rt, s_buf, n, batch)?;

        // Allocate all scratch buffers once
        let q_gpu = rt.zero_buffer::<f32>(batch * n_atoms)?;
        let dq = rt.zero_buffer::<f32>(batch * n_atoms)?;
        let v = rt.zero_buffer::<f32>(batch * n_atoms)?;
        let h_scc = rt.zero_buffer::<f32>(batch * nn)?;
        let temp = rt.zero_buffer::<f32>(batch * nn)?;
        let hp = rt.zero_buffer::<f32>(batch * nn)?;
        let cp = rt.zero_buffer::<f32>(batch * nn)?;
        let c = rt.zero_buffer::<f32>(batch * nn)?;
        let d = rt.zero_buffer::<f32>(batch * nn)?;
        let q_new = rt.zero_buffer::<f32>(batch * n_atoms)?;
        let tr = rt.zero_buffer::<f32>(batch)?;
        let dot = rt.zero_buffer::<f32>(batch)?;
        let rms = rt.zero_buffer::<f32>(batch)?;
        let occ_mask = rt.zero_buffer::<i32>(batch * n)?;
        let eig_diag = rt.zero_buffer::<f32>(batch * n)?;

        Ok(Self {
            n, n_atoms, batch,
            q_gpu, dq, v, h_scc, temp, hp, cp, c, d, q_new, tr, dot, rms, occ_mask, eig_diag,
            x_buf,
            eig_diag_host: vec![0.0; batch * n],
            mask_host: vec![0; batch * n],
            rms_host: vec![0.0; batch],
        })
    }

    /// Update the Löwdin transform for a new geometry. Call this when the
    /// overlap matrix changes (e.g. between relaxation steps).
    pub fn set_geometry(&mut self, rt: &mut GpuRuntime, s_buf: &Buffer<f32>) -> Result<()> {
        let (x_buf, _lambda_min) = build_inv_sqrt_batched(rt, s_buf, self.n, self.batch)?;
        self.x_buf = x_buf;
        Ok(())
    }

    /// Set the initial charges for a new SCC solve (e.g. warm-start from
    /// a previous relaxation step).
    pub fn set_initial_charges(&mut self, rt: &GpuRuntime, init_q: &[f32]) -> Result<()> {
        if init_q.len() != self.batch * self.n_atoms {
            return Err(DftbError::InvalidInput(format!(
                "set_initial_charges: len {} != batch*n_atoms {}*{} = {}",
                init_q.len(), self.batch, self.n_atoms, self.batch * self.n_atoms
            )));
        }
        self.q_gpu.write(init_q).enq().map_err(map_ocl_err)?;
        Ok(())
    }

    /// Run one SCC iteration step (steps 1-9 of the SCC loop).
    /// Returns the max RMS residual across all systems.
    ///
    /// All buffer args are from the plan's own scratch buffers; the caller
    /// only provides the persistent inputs (H0, S, gamma, q0, orb_atom).
    pub fn scc_step(
        &mut self,
        rt: &mut GpuRuntime,
        h0_buf: &Buffer<f32>,
        s_buf: &Buffer<f32>,
        g_buf: &Buffer<f32>,
        q0_buf: &Buffer<f32>,
        orb_atom_buf: &Buffer<i32>,
        n_occ: usize,
        alpha: f32,
    ) -> Result<f32> {
        let n = self.n;
        let n_atoms = self.n_atoms;
        let batch = self.batch;

        // 1. Δq = q − q0
        delta_q_batched(rt, &self.q_gpu, q0_buf, &self.dq, n_atoms, batch)?;

        // 2. V = G · Δq
        gamma_matvec_batched(rt, g_buf, &self.dq, &self.v, n_atoms, batch)?;

        // 3. H_scc = H0 + 0.5·S·(V_i + V_j)
        h_scc_update_batched(rt, h0_buf, s_buf, &self.v, &self.h_scc, orb_atom_buf, n, n_atoms, batch)?;

        // 4. H' = X · H_scc · X (2 GEMMs)
        matmul_batched(rt, &self.x_buf, &self.h_scc, &self.temp, n, batch)?;
        matmul_batched(rt, &self.temp, &self.x_buf, &self.hp, n, batch)?;

        // 5. Jacobi(H') → eigenvalues on diag(hp), eigenvectors in cp
        jacobi_batched(rt, &self.hp, &self.cp, n, batch)?;

        // 6. Extract diag on GPU → sort → occ_mask → upload
        extract_diagonal_batched(rt, &self.hp, &self.eig_diag, n, batch)?;
        rt.read_buffer(&self.eig_diag, &mut self.eig_diag_host)?;
        for bi in 0..batch {
            let mut idxs: Vec<usize> = (0..n).collect();
            idxs.sort_unstable_by(|&a, &b| {
                self.eig_diag_host[bi * n + a]
                    .partial_cmp(&self.eig_diag_host[bi * n + b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (rank, &k) in idxs.iter().enumerate() {
                self.mask_host[bi * n + k] = if rank < n_occ { 1 } else { 0 };
            }
        }
        self.occ_mask.write(&self.mask_host).enq().map_err(map_ocl_err)?;

        // 7. C = X · C'
        matmul_batched(rt, &self.x_buf, &self.cp, &self.c, n, batch)?;

        // 8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T
        build_density_masked_batched(rt, &self.c, &self.occ_mask, &self.d, n, batch)?;

        // 9. q_new = Mulliken(D, S)
        mulliken_charges_batched(rt, &self.d, s_buf, &self.q_new, orb_atom_buf, n, n_atoms, batch)?;

        // 10. residual + mix
        residual_and_mix_batched(
            rt, &self.q_new, &self.q_gpu, &self.q_gpu, &self.rms, alpha, n_atoms, batch,
        )?;

        // Read RMS and return max
        rt.read_buffer(&self.rms, &mut self.rms_host)?;
        let max_rms = self.rms_host.iter().fold(0.0f32, |m, &r| m.max(r));
        Ok(max_rms)
    }

    /// Compute the SCC energy after convergence: E = Tr(D·H0) + 0.5·ΣΔq·V.
    /// Returns energies per system.
    pub fn compute_energy(
        &mut self,
        rt: &mut GpuRuntime,
        h0_buf: &Buffer<f32>,
        g_buf: &Buffer<f32>,
        q0_buf: &Buffer<f32>,
    ) -> Result<Vec<f32>> {
        let n = self.n;
        let n_atoms = self.n_atoms;
        let batch = self.batch;

        // E_h0 = Tr(D · H0)
        frobenius_trace_batched(rt, &self.d, h0_buf, &self.tr, n, batch)?;

        // Recompute Δq and V for E_scc
        delta_q_batched(rt, &self.q_gpu, q0_buf, &self.dq, n_atoms, batch)?;
        gamma_matvec_batched(rt, g_buf, &self.dq, &self.v, n_atoms, batch)?;
        dot_batched(rt, &self.dq, &self.v, &self.dot, n_atoms, batch)?;

        let mut e_h0 = vec![0.0f32; batch];
        let mut e_scc = vec![0.0f32; batch];
        rt.read_buffer(&self.tr, &mut e_h0)?;
        rt.read_buffer(&self.dot, &mut e_scc)?;

        Ok((0..batch).map(|i| e_h0[i] + 0.5 * e_scc[i]).collect())
    }

    /// Read back final charges from the plan.
    pub fn read_charges(&self, rt: &GpuRuntime) -> Result<Vec<f32>> {
        let mut charges = vec![0.0f32; self.batch * self.n_atoms];
        rt.read_buffer(&self.q_gpu, &mut charges)?;
        Ok(charges)
    }

    /// Read back eigenvalues (sorted ascending per system).
    pub fn read_eigenvalues(&mut self, rt: &mut GpuRuntime) -> Result<Vec<f32>> {
        let n = self.n;
        let batch = self.batch;
        extract_diagonal_batched(rt, &self.hp, &self.eig_diag, n, batch)?;
        rt.read_buffer(&self.eig_diag, &mut self.eig_diag_host)?;
        let mut eigenvalues = vec![0.0f32; batch * n];
        for bi in 0..batch {
            let mut eigs: Vec<f32> = self.eig_diag_host[bi * n..(bi + 1) * n].to_vec();
            eigs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            eigenvalues[bi * n..(bi + 1) * n].copy_from_slice(&eigs);
        }
        Ok(eigenvalues)
    }

    /// Get the plan configuration.
    pub fn config(&self) -> (usize, usize, usize) {
        (self.n, self.n_atoms, self.batch)
    }
}
