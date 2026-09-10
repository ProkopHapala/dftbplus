//! Persistent GPU-resident workspace for the full sparse DFTB SCC pipeline.
//!
//! **Caveat (2026-09-10):** this struct is the *intended* owner (manifest §0.4).
//! `run_scc` is still **one-shot purify** (Z, K0, TC2, Mulliken) — no γ, no
//! Hscc update, no charge mix. Physics gates G3/F/G use `scc.rs` instead.
//! Device NS inside this workspace is N4-wrong; do not treat a green workspace
//! unit test as a DFTB SCC. See `doc/prokop/topical_audit/f32_floor_sparse.md`.
//!
//! GPT-5.6 #2/#8, manifest §1.4: one persistent workspace owns all GPU
//! structures, buffers, and symbolic plans for the lifetime of a frozen
//! topology. No allocation in the SCC/force/Hessian hot loops.
//!
//! Pipeline:
//! ```text
//! H0/S → Z≈S⁻¹ → spectral_bounds → K0 → TC2 → Mulliken q → γ·Δq → Hscc → repeat
//! ```
//!
//! All matrices stay on the GPU. The only host transfers are:
//! - 2 scalars (emin, emax) for spectral bounds (once per SCC outer iteration)
//! - 1 scalar (Tr(KS)) + 1 scalar (R_I) per TC2 diagnostic iteration
//! - n_atom floats (Mulliken charges) per SCC iteration
//!
//! The workspace is constructed once for a frozen topology (masks, plans,
//! structures) and reused across all geometry displacements. Only values
//! (H0, S, positions) change between calls.

use crate::core::error::{DftbError, Result};
use crate::methods::sparse::bsr4::{
    build_geometric_mask, build_product_mask, build_spgemm_plan_bsym, inf_norm, Bsr4Matrix, BS, BS2,
};
use crate::methods::sparse::gpu_sparse::{
    GpuBsrMatrix, GpuBsrStructure, SparseBsr4Gpu, SparseBsr4Config, SpgemmPlanGpu, TC2_TRACE_TOL,
};
use ocl::{flags, Buffer};
use std::sync::Arc;

/// Persistent GPU-resident workspace for the full sparse DFTB SCC pipeline.
///
/// Owns all GPU structures, buffers, and symbolic plans for the lifetime of
/// a frozen topology. Built once via `SparseSystemWorkspace::new`, reused
/// across all SCC iterations, force evaluations, and Hessian displacements.
///
/// **No allocation in hot loops** — all buffers are preallocated at
/// construction. Only kernel arguments change between calls.
pub struct SparseSystemWorkspace {
    /// GPU runtime + cached kernels.
    gpu: SparseBsr4Gpu,

    // ── Frozen patterns (host-side, for reference) ──
    n_atom: usize,
    m_hs: (Vec<u32>, Vec<u32>),
    m_k: (Vec<u32>, Vec<u32>),
    m_t: (Vec<u32>, Vec<u32>),

    // ── GPU structures (immutable, shared via Arc) ──
    hs_struct: Arc<GpuBsrStructure>,
    k_struct: Arc<GpuBsrStructure>,
    t_struct: Arc<GpuBsrStructure>, // T = K·S product mask

    // ── Persistent GPU matrices ──
    // H/S (change with geometry, uploaded per SCC outer iteration)
    h0: GpuBsrMatrix,     // H0 on M_HS
    s: GpuBsrMatrix,      // S on M_HS
    h_scc: GpuBsrMatrix,  // H_scc on M_HS

    // Z ≈ S⁻¹ (changes with S, recomputed per geometry)
    z: GpuBsrMatrix,      // Z on M_K
    znew: GpuBsrMatrix,   // Znew scratch on M_K

    // K (density kernel, changes each SCC iteration)
    k: GpuBsrMatrix,      // K on M_K
    knew: GpuBsrMatrix,   // Knew scratch on M_K
    t_ks: GpuBsrMatrix,   // T = K·S on M_T
    q: GpuBsrMatrix,      // Q = T·K = KSK on M_K

    // K0 construction scratch
    b_zh: GpuBsrMatrix,   // B = Z·H on M_T (scratch)
    a_zhz: GpuBsrMatrix,  // A = B·Z on M_K (scratch)

    // NS scratch (for Z·S)
    t_zs: GpuBsrMatrix,   // T = Z·S on M_T (scratch)

    // ── Symbolic plans (built once, reused forever) ──
    plan_ks: Option<SpgemmPlanGpu>,  // K·S
    plan_tk: Option<SpgemmPlanGpu>,  // T·K (for TC2 Q = T·K)
    plan_zs: Option<SpgemmPlanGpu>,  // Z·S (for NS)
    plan_tz: Option<SpgemmPlanGpu>,  // T·Z (for NS Q = T·Z)
    plan_zh: Option<SpgemmPlanGpu>,  // Z·H (for K0 B = Z·H)
    plan_bz: Option<SpgemmPlanGpu>,  // B·Z (for K0 A = B·Z)

    // ── Reduction scratch ──
    trace_buf: Buffer<f32>,
    residual_buf: Buffer<f32>,
    reduce_partial: Buffer<f32>,
    reduce_a: Buffer<f32>,
    reduce_b: Buffer<f32>,

    // Host scratch for honest NS residual (downloaded T, not the lying kernel scalar).
    t_host: Vec<f32>,
    k_host: Vec<f32>,
    s_inf: f32,

    // ── System parameters ──
    nocc: f32,
}

impl SparseSystemWorkspace {
    /// Build a persistent workspace for a frozen topology.
    ///
    /// `h0` and `s` are the initial H0 and S matrices (their values will be
    /// overwritten per geometry, but their CSR structure defines M_HS).
    /// `k_mask` defines M_K (the density-kernel support). `nocc` is the
    /// number of occupied orbitals.
    ///
    /// All GPU structures, buffers, and symbolic plans are allocated and
    /// uploaded once here. No allocation happens in subsequent SCC/force
    /// calls.
    pub fn new(
        gpu: SparseBsr4Gpu,
        h0: &Bsr4Matrix,
        s: &Bsr4Matrix,
        k_mask: &(Vec<u32>, Vec<u32>),
        nocc: f32,
    ) -> Result<Self> {
        let n_atom = h0.n_atom;
        assert_eq!(s.n_atom, n_atom, "H0 and S must have same n_atom");

        // M_HS = H/S mask (from h0 structure).
        let m_hs = (h0.row_ptr.clone(), h0.col_idx.clone());
        // M_K = K mask (provided).
        let m_k = k_mask.clone();
        // M_T = product mask M_K ∘ M_HS (for T = K·S).
        let m_t = build_product_mask(n_atom, &m_k, &m_hs);

        // Build GPU structures (immutable, shared).
        let hs_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_hs)?);
        let k_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_k)?);
        let t_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_t)?);

        // Upload initial H0 and S values.
        let h0_mat = GpuBsrMatrix { struct_: hs_struct.clone(), values: gpu.buf_f32(&h0.values)? };
        let s_mat = GpuBsrMatrix { struct_: hs_struct.clone(), values: gpu.buf_f32(&s.values)? };
        let h_scc = GpuBsrMatrix::zero(&gpu, &hs_struct)?;

        // Z, K, and scratch matrices.
        let z = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let znew = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let k = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let knew = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let t_ks = GpuBsrMatrix::zero(&gpu, &t_struct)?;
        let q = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let b_zh = GpuBsrMatrix::zero(&gpu, &t_struct)?;
        let a_zhz = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let t_zs = GpuBsrMatrix::zero(&gpu, &t_struct)?;

        // Build symbolic plans for all recurring SpGEMMs.
        let build_plan = |a: &Bsr4Matrix, b: &Bsr4Matrix, c_mask: &(Vec<u32>, Vec<u32>), label: &str| -> Option<SpgemmPlanGpu> {
            match build_spgemm_plan_bsym(a, b, c_mask) {
                Ok(plan) => match gpu.upload_plan(&plan) {
                    Ok(gpu_plan) => Some(gpu_plan),
                    Err(e) => {
                        eprintln!("P4: {label} plan upload failed, falling back to intersection: {e}");
                        None
                    }
                },
                Err(e) => {
                    eprintln!("P4: {label} plan build failed, falling back to intersection: {e}");
                    None
                }
            }
        };

        let k_dummy = Bsr4Matrix::from_structure(n_atom, m_k.0.clone(), m_k.1.clone())?;
        let hs_dummy = Bsr4Matrix::from_structure(n_atom, m_hs.0.clone(), m_hs.1.clone())?;
        let t_dummy = Bsr4Matrix::from_structure(n_atom, m_t.0.clone(), m_t.1.clone())?;

        let plan_ks = build_plan(&k_dummy, &hs_dummy, &m_t, "plan_ks");
        let plan_tk = build_plan(&t_dummy, &k_dummy, &m_k, "plan_tk");
        let plan_zs = build_plan(&k_dummy, &hs_dummy, &m_t, "plan_zs");
        let plan_tz = build_plan(&t_dummy, &k_dummy, &m_k, "plan_tz");
        let plan_zh = build_plan(&k_dummy, &hs_dummy, &m_t, "plan_zh");
        let plan_bz = build_plan(&t_dummy, &k_dummy, &m_k, "plan_bz");

        // Reduction scratch buffers.
        let trace_buf = gpu.zero_f32(1)?;
        let residual_buf = gpu.zero_f32(1)?;
        let reduce_wg = gpu.config().reduce_wg as usize;
        let reduce_len = (n_atom.max(k_struct.nblock * BS2) + reduce_wg - 1) / reduce_wg;
        let reduce_len = reduce_len.max(1);
        let reduce_partial = gpu.zero_f32(reduce_len)?;
        let reduce_a = gpu.zero_f32(reduce_len)?;
        let reduce_b = gpu.zero_f32(reduce_len)?;
        let t_host = vec![0.0f32; t_struct.nblock * BS2];
        let k_host = vec![0.0f32; k_struct.nblock * BS2];
        let s_inf = inf_norm(s); // may be 0 at construction if S is a zero placeholder

        Ok(Self {
            gpu,
            n_atom,
            m_hs,
            m_k,
            m_t,
            hs_struct,
            k_struct,
            t_struct,
            h0: h0_mat,
            s: s_mat,
            h_scc,
            z,
            znew,
            k,
            knew,
            t_ks,
            q,
            b_zh,
            a_zhz,
            t_zs,
            plan_ks,
            plan_tk,
            plan_zs,
            plan_tz,
            plan_zh,
            plan_bz,
            trace_buf,
            residual_buf,
            reduce_partial,
            reduce_a,
            reduce_b,
            t_host,
            k_host,
            s_inf,
            nocc,
        })
    }

    // ── Accessors ──

    pub fn n_atom(&self) -> usize { self.n_atom }
    pub fn nocc(&self) -> f32 { self.nocc }
    pub fn gpu(&self) -> &SparseBsr4Gpu { &self.gpu }
    pub fn m_hs(&self) -> &(Vec<u32>, Vec<u32>) { &self.m_hs }
    pub fn m_k(&self) -> &(Vec<u32>, Vec<u32>) { &self.m_k }
    pub fn nblock_k(&self) -> usize { self.k_struct.nblock }
    pub fn nblock_hs(&self) -> usize { self.hs_struct.nblock }

    /// Read current K back to host (blocking). Use only for diagnostics.
    pub fn k_to_host(&self) -> Result<Bsr4Matrix> {
        self.k.to_host(&self.gpu)
    }

    /// Read current Mulliken charges q_A = 2*Tr((KS)_AA) from device.
    /// Computes T=K·S first, then reads n_atom floats.
    pub fn mulliken_charges(&mut self) -> Result<Vec<f32>> {
        self.spgemm_ks();
        self.gpu.mulliken_dev(&self.t_struct, &self.t_ks.values)
    }

    // ── Upload methods (per geometry) ──

    /// Upload new H0 values into the persistent H0 buffer.
    /// Structure must match (same M_HS).
    pub fn upload_h0(&mut self, h0: &Bsr4Matrix) -> Result<()> {
        if h0.values.len() != self.h0.struct_.nblock * BS2 {
            return Err(DftbError::InvalidInput(format!(
                "upload_h0: values len {} != expected {}",
                h0.values.len(), self.h0.struct_.nblock * BS2
            )));
        }
        self.h0.upload_values(&self.gpu, &h0.values)
    }

    /// Upload new S values into the persistent S buffer.
    /// Structure must match (same M_HS).
    pub fn upload_s(&mut self, s: &Bsr4Matrix) -> Result<()> {
        if s.values.len() != self.s.struct_.nblock * BS2 {
            return Err(DftbError::InvalidInput(format!(
                "upload_s: values len {} != expected {}",
                s.values.len(), self.s.struct_.nblock * BS2
            )));
        }
        self.s.upload_values(&self.gpu, &s.values)?;
        self.s_inf = inf_norm(s);
        if !self.s_inf.is_finite() || self.s_inf < 1e-30 {
            return Err(DftbError::InvalidInput(format!(
                "upload_s: ||S||_inf={:e} non-finite or near-zero", self.s_inf
            )));
        }
        Ok(())
    }

    /// Upload H_scc values (same M_HS as H0).
    pub fn upload_h_scc(&mut self, h: &Bsr4Matrix) -> Result<()> {
        if h.values.len() != self.h_scc.struct_.nblock * BS2 {
            return Err(DftbError::InvalidInput(format!(
                "upload_h_scc: values len {} != expected {}",
                h.values.len(), self.h_scc.struct_.nblock * BS2
            )));
        }
        self.h_scc.upload_values(&self.gpu, &h.values)
    }

    /// Upload H_scc from a packed BSR value slice (same length as M_HS).
    pub fn upload_h_scc_values(&mut self, values: &[f32]) -> Result<()> {
        self.h_scc.upload_values(&self.gpu, values)
    }

    /// Upload H0 from a packed BSR value slice.
    pub fn upload_h0_values(&mut self, values: &[f32]) -> Result<()> {
        self.h0.upload_values(&self.gpu, values)
    }

    /// Upload S from a packed BSR value slice. `s_inf` must be set by the caller (host inf_norm of the same values).
    pub fn upload_s_values(&mut self, values: &[f32], s_inf: f32) -> Result<()> {
        if !s_inf.is_finite() || s_inf < 1e-30 {
            return Err(DftbError::InvalidInput(format!("upload_s_values: ||S||_inf={s_inf:e}")));
        }
        self.s_inf = s_inf;
        self.s.upload_values(&self.gpu, values)
    }

    // ── SpGEMM helpers (use plans when available) ──

    /// T = K·S using plan or intersection kernel.
    fn spgemm_ks(&mut self) {
        match &self.plan_ks {
            Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.k, &self.s, plan, &self.t_ks),
            None => self.gpu.spgemm_bsym_dev(&self.k, &self.s, &self.t_ks),
        }.expect("spgemm_ks failed");
    }

    /// Q = T·K using plan or intersection kernel.
    fn spgemm_tk(&mut self) {
        match &self.plan_tk {
            Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.t_ks, &self.k, plan, &self.q),
            None => self.gpu.spgemm_bsym_dev(&self.t_ks, &self.k, &self.q),
        }.expect("spgemm_tk failed");
    }

    // ── Newton-Schulz inverse (device-resident) ──

    /// Compute Z ≈ S⁻¹ via Newton-Schulz on persistent GPU buffers.
    ///
    /// Products use the **intersection** `spgemm_bsym_dev` kernel (same family as
    /// G3 host NS). Residual is `||I−T||_F/√N` of the **downloaded** T — not
    /// `identity_residual_scalar_dev`, which lies about the returned Z (N4).
    /// No `Buffer::builder` / `Kernel::builder` in this loop.
    pub fn compute_z(&mut self, max_iter: usize, tol: f32, stall: usize) -> Result<(f32, usize)> {
        let n_orb = (self.n_atom * BS) as f32;
        let nblock = self.k_struct.nblock;
        let alpha = 1.0 / self.s_inf;
        self.gpu.build_identity_dev(&self.k_struct, &self.z.values)?;
        self.gpu.scale_dev(nblock, alpha, &self.z.values)?;

        let mut prev_rz = f32::INFINITY;
        let mut stall_count = 0;
        for iter in 0..max_iter {
            // T = Z·S  — intersection kernel, not the P4 plan (N4 diagnostic).
            self.gpu.spgemm_bsym_dev(&self.z, &self.s, &self.t_zs)?;
            self.gpu.read_f32(&self.t_zs.values, &mut self.t_host)?;
            let rz = host_identity_rz(&self.t_host, &self.m_t.0, &self.m_t.1, self.n_atom) / n_orb.sqrt();
            if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                eprintln!("  Newton-Schulz (workspace) iter {iter}: R_Z = {rz:e}  (host ||I−T|| of downloaded T)");
            }
            if !rz.is_finite() {
                return Err(DftbError::InvalidInput(format!("compute_z: non-finite R_Z={rz:e} at iter {iter}")));
            }
            if rz < tol {
                return Ok((rz, iter + 1));
            }
            if iter > 0 && rz > 0.9 * prev_rz {
                stall_count += 1;
                if stall_count >= stall {
                    return Err(DftbError::InvalidInput(format!(
                        "compute_z: stalled after {} iters, R_Z={rz:e} (tol={tol:e})", iter + 1
                    )));
                }
            } else {
                stall_count = 0;
            }
            prev_rz = rz;
            self.gpu.spgemm_bsym_dev(&self.t_zs, &self.z, &self.q)?;
            self.gpu.axpby_dev(nblock, 2.0, &self.z.values, -1.0, &self.q.values, &self.znew.values)?;
            self.gpu.symmetrize_dev(nblock, &self.k_struct.transpose_block(), &self.znew.values)?;
            std::mem::swap(&mut self.z.values, &mut self.znew.values);
        }
        self.gpu.spgemm_bsym_dev(&self.z, &self.s, &self.t_zs)?;
        self.gpu.read_f32(&self.t_zs.values, &mut self.t_host)?;
        let rz = host_identity_rz(&self.t_host, &self.m_t.0, &self.m_t.1, self.n_atom) / n_orb.sqrt();
        Err(DftbError::InvalidInput(format!(
            "compute_z: exhausted {max_iter} iters, final R_Z={rz:e} (tol={tol:e})"
        )))
    }

    // ── K0 construction (device-resident) ──

    /// Compute spectral bounds of B = Z·H on device, then K₀ on device.
    /// Returns (emin, emax) — the padded spectral bounds.
    pub fn compute_k0(&mut self, padding: f32) -> Result<(f32, f32)> {
        // Spectral bounds: B = Z·H, Gershgorin on device.
        let (emin, emax) = self.gpu.spectral_bounds_dev(
            &self.z, &self.h0, &self.b_zh, padding,
        )?;

        // K₀ = (emax·Z - Z·H·Z) / Δε on device.
        self.gpu.build_k0_dev(
            &self.z, &self.h0, &self.b_zh, &self.a_zhz, &self.k,
            emin, emax,
        )?;

        Ok((emin, emax))
    }

    /// K₀ from the current **H_scc** (not H0). Call after `upload_h_scc`.
    pub fn compute_k0_from_hscc(&mut self, padding: f32) -> Result<(f32, f32)> {
        let (emin, emax) = self.gpu.spectral_bounds_dev(&self.z, &self.h_scc, &self.b_zh, padding)?;
        if !emin.is_finite() || !emax.is_finite() || emax <= emin {
            return Err(DftbError::InvalidInput(format!(
                "compute_k0_from_hscc: bad ZH Gershgorin emin={emin} emax={emax}"
            )));
        }
        self.gpu.build_k0_dev(&self.z, &self.h_scc, &self.b_zh, &self.a_zhz, &self.k, emin, emax)?;
        Ok((emin, emax))
    }

    // ── TC2 purification (device-resident) ──

    /// Run TC2 purification on the current K. Returns (final_R_I, final_Tr, iterations).
    /// K is updated in place on the device.
    pub fn tc2_purify(&mut self, max_iter: usize, tol: f32, check_every: usize) -> Result<(f32, f32, usize)> {
        let mut best_r_i = f32::INFINITY;
        let mut last_tr = 0.0f32;
        let mut last_r_i = f32::INFINITY;

        for iter in 0..max_iter {
            let do_check = (iter % check_every == 0) || (iter == max_iter - 1);

            // T = K·S, Q = T·K, trace = Tr(T), optionally R_I = ||Q-K||.
            match &self.plan_ks {
                Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.k, &self.s, plan, &self.t_ks)?,
                None => self.gpu.spgemm_bsym_dev(&self.k, &self.s, &self.t_ks)?,
            }
            match &self.plan_tk {
                Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.t_ks, &self.k, plan, &self.q)?,
                None => self.gpu.spgemm_bsym_dev(&self.t_ks, &self.k, &self.q)?,
            }
            self.gpu.trace_ks_to_dev(
                &self.t_struct, &self.t_ks.values,
                &self.reduce_partial, &self.reduce_a, &self.reduce_b, &self.trace_buf,
            )?;
            if do_check {
                self.gpu.idempotency_to_dev(
                    self.k.struct_.nblock, &self.q.values, &self.k.values,
                    &self.reduce_partial, &self.reduce_a, &self.reduce_b, &self.residual_buf,
                )?;
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
                if ri < best_r_i { best_r_i = ri; }

                if ri < tol {
                    if (tr[0] - self.nocc).abs() <= TC2_TRACE_TOL {
                        return Ok((ri, tr[0], iter + 1));
                    }
                    if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                        eprintln!("  TC2 R_I={ri:e} < tol but Tr(KS)={} != Nocc={} (wrong-rank projector not accepted)", tr[0], self.nocc);
                    }
                }
                if ri > best_r_i * 10.0 && best_r_i < f32::INFINITY {
                    return Err(DftbError::InvalidInput(format!(
                        "TC2 diverged at iter {iter}, R_I={ri:e}, best={best_r_i:e}"
                    )));
                }
            }

            // Update K: Knew = TC2_branch(K, Q, trace, Nocc), symmetrize, swap.
            let nblock = self.k.struct_.nblock;
            self.gpu.tc2_dev(
                nblock, &self.k.values, &self.q.values, &self.trace_buf,
                self.nocc, &self.knew.values,
            )?;
            self.gpu.symmetrize_dev(nblock, &self.k_struct.transpose_block(), &self.knew.values)?;
            std::mem::swap(&mut self.k.values, &mut self.knew.values);
        }

        Err(DftbError::InvalidInput(format!(
            "TC2 exhausted {max_iter} iters, final R_I={last_r_i:e} Tr(KS)={last_tr} (tol={tol:e} Nocc={})",
            self.nocc
        )))
    }

    /// K0 + TC2 of the current device H_scc using the current Z. No NS.
    pub fn purify_hscc(&mut self, tc2_max: usize, tc2_tol: f32) -> Result<(f32, f32, usize)> {
        let (emin, emax) = self.compute_k0_from_hscc(0.1)?;
        if crate::methods::sparse::gpu_sparse::algebra_verbose() {
            eprintln!("  bounds (ZH Gershgorin, H_scc): emin={emin:.4} emax={emax:.4}");
        }
        self.tc2_purify(tc2_max, tc2_tol, 1)
    }

    /// Download K values into persistent `k_host` (no extra alloc).
    pub fn k_values_host(&mut self) -> Result<&[f32]> {
        self.gpu.read_f32(&self.k.values, &mut self.k_host)?;
        Ok(&self.k_host)
    }

    /// Read K into a caller dense pad buffer (full mask: BSR order is row-major blocks).
    pub fn k_to_dense_into(&mut self, out: &mut [f32]) -> Result<()> {
        self.gpu.read_f32(&self.k.values, &mut self.k_host)?;
        crate::methods::sparse::bsr4::bsr_values_to_dense(self.n_atom, &self.m_k.0, &self.m_k.1, &self.k_host, out);
        Ok(())
    }

    // ── Full SCC pipeline ──

    /// Run the complete sparse SCC pipeline:
    ///
    /// 1. Z ≈ S⁻¹ (Newton-Schulz)
    /// 2. Spectral bounds + K0 construction
    /// 3. TC2 purification → K
    /// 4. Mulliken charges
    ///
    /// Returns (Mulliken charges, final_R_I, final_Tr, tc2_iters).
    pub fn run_scc(
        &mut self,
        ns_max_iter: usize,
        ns_tol: f32,
        tc2_max_iter: usize,
        tc2_tol: f32,
        tc2_check_every: usize,
    ) -> Result<(Vec<f32>, f32, f32, usize)> {
        // 1. Z ≈ S⁻¹
        let (_, _) = self.compute_z(ns_max_iter, ns_tol, 3)?;

        // 2. K0
        let _ = self.compute_k0(0.1)?;

        // 3. TC2
        let (r_i, tr, iters) = self.tc2_purify(tc2_max_iter, tc2_tol, tc2_check_every)?;

        // 4. Mulliken charges
        let charges = self.mulliken_charges()?;

        Ok((charges, r_i, tr, iters))
    }
}

/// `||T − I||_F` from downloaded BSR values (f64 accum). Diagonal blocks subtract I.
fn host_identity_rz(t: &[f32], row_ptr: &[u32], col_idx: &[u32], n_atom: usize) -> f32 {
    assert_eq!(t.len(), col_idx.len() * BS2);
    let mut s2 = 0.0f64;
    for i in 0..n_atom {
        let (a, b) = (row_ptr[i] as usize, row_ptr[i + 1] as usize);
        for blk in a..b {
            let j = col_idx[blk] as usize;
            let v = &t[blk * BS2..(blk + 1) * BS2];
            for r in 0..BS {
                for c in 0..BS {
                    let mut x = v[r * BS + c] as f64;
                    if i == j && r == c { x -= 1.0; }
                    s2 += x * x;
                }
            }
        }
    }
    s2.sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::sparse::bsr4::build_full_mask;
    use crate::methods::sparse::harness::require_sparse_gpu;

    fn try_gpu() -> Option<SparseBsr4Gpu> {
        require_sparse_gpu()
    }

    /// Simple LCG for reproducible random matrices.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as f64 / (1u64 << 33) as f64
        }
    }

    fn random_symmetric_bsr4(n_atom: usize, mask: &(Vec<u32>, Vec<u32>), seed: u64) -> Bsr4Matrix {
        let mut m = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
        let mut rng = Rng(seed);
        for i in 0..n_atom {
            let (s, e) = (mask.0[i] as usize, mask.0[i + 1] as usize);
            for blk in s..e {
                let j = mask.1[blk] as usize;
                if i <= j {
                    let mut v = [0.0f32; BS2];
                    for k in 0..BS2 { v[k] = (rng.next() * 2.0 - 1.0) as f32; }
                    if i == j {
                        for r in 0..BS { for c in (r+1)..BS {
                            let avg = 0.5 * (v[r*BS+c] + v[c*BS+r]);
                            v[r*BS+c] = avg; v[c*BS+r] = avg;
                        }}
                        // Add diagonal shift for gap
                        for r in 0..BS { v[r*BS+r] += 2.0; }
                    } else {
                        let mut vt = [0.0f32; BS2];
                        for r in 0..BS { for c in 0..BS { vt[c*BS+r] = v[r*BS+c]; }}
                        m.set_block(j, i, &vt).unwrap();
                    }
                    m.set_block(i, j, &v).unwrap();
                }
            }
        }
        m
    }

    fn make_overlap_bsr4(n_atom: usize, mask: &(Vec<u32>, Vec<u32>), seed: u64) -> Bsr4Matrix {
        let mut m = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
        let mut rng = Rng(seed);
        for i in 0..n_atom {
            let (s, e) = (mask.0[i] as usize, mask.0[i + 1] as usize);
            for blk in s..e {
                let j = mask.1[blk] as usize;
                if i <= j {
                    let mut v = [0.0f32; BS2];
                    if i == j {
                        // Diagonal: identity-like
                        v[0] = 1.0; v[5] = 1.0; v[10] = 1.0; v[15] = 1.0;
                    } else {
                        // Off-diagonal: small random
                        for k in 0..BS2 { v[k] = (rng.next() * 0.2) as f32; }
                        for r in 0..BS { for c in (r+1)..BS {
                            let avg = 0.5 * (v[r*BS+c] + v[c*BS+r]);
                            v[r*BS+c] = avg; v[c*BS+r] = avg;
                        }}
                        let mut vt = [0.0f32; BS2];
                        for r in 0..BS { for c in 0..BS { vt[c*BS+r] = v[r*BS+c]; }}
                        m.set_block(j, i, &vt).unwrap();
                    }
                    m.set_block(i, j, &v).unwrap();
                }
            }
        }
        m
    }

    /// Test: full SCC pipeline (Z → K0 → TC2 → Mulliken) on a small system.
    #[test]
    fn test_sparse_system_scc_pipeline() {
        let Some(gpu) = try_gpu() else { return };
        let n_atom = 3;
        let nocc = 3.0f32;
        let mask = build_full_mask(n_atom);
        let h0 = random_symmetric_bsr4(n_atom, &mask, 42);
        let s = make_overlap_bsr4(n_atom, &mask, 99);

        let mut ws = SparseSystemWorkspace::new(gpu, &h0, &s, &mask, nocc).unwrap();

        // Run full SCC pipeline.
        let (charges, r_i, tr, iters) = ws.run_scc(30, 1e-4, 40, 1e-5, 1).unwrap();
        println!("SCC: {iters} TC2 iters, R_I={r_i:e}, Tr(KS)={tr:.6}");
        println!("Mulliken charges: {:?}", &charges);

        // Verify convergence.
        assert!(r_i < 1e-5, "TC2 did not converge: R_I={r_i:e} (G2)");
        assert!((tr - nocc).abs() < 1e-5, "Tr(KS) mismatch: {tr} vs {nocc} (G2)");
        assert_eq!(charges.len(), n_atom);

        // G1.8: Tr(KS)=Nocc, never Tr(K²) for a non-orthogonal metric.
        let k_host = ws.k_to_host().unwrap();
        let k_dense = k_host.to_dense();
        let s_dense = s.to_dense();
        let n = n_atom * BS;
        let mut tr_ks = 0.0f32;
        for i in 0..n {
            let mut ks_ii = 0.0f32;
            for j in 0..n {
                ks_ii += k_dense[i * n + j] * s_dense[j * n + i];
            }
            tr_ks += ks_ii;
        }
        println!("Tr(KS) = {tr_ks:.4} (Nocc={nocc})");
        assert!((tr_ks - nocc).abs() < 1e-5, "host Tr(KS)={tr_ks} != Nocc={nocc} (G1.8)");
    }

    /// Test: workspace is reusable across multiple SCC runs.
    #[test]
    fn test_sparse_system_reuse() {
        let Some(gpu) = try_gpu() else { return };
        let n_atom = 3;
        let nocc = 3.0f32;
        let mask = build_full_mask(n_atom);
        let h0 = random_symmetric_bsr4(n_atom, &mask, 42);
        let s = make_overlap_bsr4(n_atom, &mask, 99);

        let mut ws = SparseSystemWorkspace::new(gpu, &h0, &s, &mask, nocc).unwrap();

        // First run.
        let (_, r_i1, tr1, _) = ws.run_scc(30, 1e-4, 60, 1e-5, 1).unwrap();
        println!("Run 1: R_I={r_i1:e}, Tr={tr1:.6}");

        // Second run with same geometry (should give same result).
        let (_, r_i2, tr2, _) = ws.run_scc(30, 1e-4, 60, 1e-5, 1)
            .unwrap_or_else(|e| panic!("Run 2 failed (no skip): {e}"));
        println!("Run 2: R_I={r_i2:e}, Tr={tr2:.6}");

        // Both runs should converge to the same state.
        assert!((tr1 - tr2).abs() < 1e-4, "Tr mismatch: {tr1} vs {tr2}");
        assert!((r_i1 - r_i2).abs() < 1e-4, "R_I mismatch: {r_i1} vs {r_i2}");
    }
}
