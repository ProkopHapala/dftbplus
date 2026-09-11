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
    build_product_mask, build_spgemm_plan_bsym, inf_norm, Bsr4Matrix, BS, BS2,
};
use crate::methods::sparse::gpu_sparse::{
    GpuBsrMatrix, GpuBsrStructure, SparseBsr4Gpu, SpgemmPlanGpu, TC2_TRACE_TOL,
    TC2_TRACE_GUARD_REL, TC2_TRACE_LOCK_REL, TC2_LOCK_RI, TC2_STAGNANT_MAX,
};
use ocl::Buffer;
use std::sync::Arc;

/// Persistent GPU-resident workspace for the full sparse DFTB SCC pipeline.
///
/// Owns all GPU structures, buffers, and symbolic plans for the lifetime of
/// a frozen topology. Built once via `SparseSystemWorkspace::new`, reused
/// across all SCC iterations, force evaluations, and Hessian displacements.
///
/// **Independent masks (second review R3):** `M_HS` (H/S support) ⊂ support
/// of S; `M_K` (density kernel), `M_Z` (inverse overlap) are *separate*
/// geometric masks. Product masks `M_TKS = M_K∘M_HS`, `M_TZS = M_Z∘M_HS`,
/// `M_HT = M_HS∘M_TKS` (R_H only) are exact intermediates.
///
/// **No allocation in hot loops** — all buffers are preallocated at
/// construction. Only kernel arguments change between calls.
pub struct SparseSystemWorkspace {
    /// GPU runtime + cached kernels.
    gpu: SparseBsr4Gpu,

    // ── Frozen patterns (host-side, for reference / tests) ──
    n_atom: usize,
    m_hs: (Vec<u32>, Vec<u32>),
    m_k: (Vec<u32>, Vec<u32>),
    #[allow(dead_code)] m_z: (Vec<u32>, Vec<u32>),
    #[allow(dead_code)] m_t_ks: (Vec<u32>, Vec<u32>),  // T = K·S support
    #[allow(dead_code)] m_t_zs: (Vec<u32>, Vec<u32>),  // T = Z·S and B = Z·H support
    m_ht: (Vec<u32>, Vec<u32>),    // A = H·(KS) support (R_H, final only)
    ht_transpose: Vec<u32>,        // host map: ht block (i,j) → block (j,i)

    // ── GPU structures (immutable, shared via Arc) ──
    hs_struct: Arc<GpuBsrStructure>,
    k_struct: Arc<GpuBsrStructure>,
    z_struct: Arc<GpuBsrStructure>,
    t_ks_struct: Arc<GpuBsrStructure>,
    t_zs_struct: Arc<GpuBsrStructure>,
    #[allow(dead_code)] ht_struct: Arc<GpuBsrStructure>,

    // ── Persistent GPU matrices ──
    // H/S (change with geometry, uploaded per set_coords)
    h0: GpuBsrMatrix,     // H0 on M_HS
    s: GpuBsrMatrix,      // S on M_HS
    h_scc: GpuBsrMatrix,  // H_scc on M_HS (built on device from V — R13)

    // Z ≈ S⁻¹ on M_Z (recomputed/corrected per geometry)
    z: GpuBsrMatrix,      // Z on M_Z
    znew: GpuBsrMatrix,   // Znew scratch on M_Z
    qz: GpuBsrMatrix,     // Q = T·Z = ZSZ on M_Z (NS)

    // K (density kernel, changes each SCC iteration)
    k: GpuBsrMatrix,      // K on M_K
    knew: GpuBsrMatrix,   // Knew scratch on M_K
    k_best: Buffer<f32>,  // best-K snapshot for TC2 plateau recovery
    q: GpuBsrMatrix,      // Q = T·K = KSK on M_K (TC2)
    a_zhz: GpuBsrMatrix,  // A = (Z·H)·Z restricted to M_K (K0)
    z_on_k: GpuBsrMatrix, // Z restricted to M_K (K0 axpby — R8b)

    // Intermediates
    t_ks: GpuBsrMatrix,   // T = K·S on M_TKS
    t_zs: GpuBsrMatrix,   // T = Z·S on M_TZS (NS)
    b_zh: GpuBsrMatrix,   // B = Z·H on M_TZS (K0)
    a_ht: GpuBsrMatrix,   // A = H_scc·(KS) on M_HT (R_H, once per SCC)

    // ── Symbolic plans (built once, reused forever) ──
    plan_ks: Option<SpgemmPlanGpu>,  // K·S → M_TKS
    plan_tk: Option<SpgemmPlanGpu>,  // T·K → M_K (TC2 Q = KSK)
    plan_zs: Option<SpgemmPlanGpu>,  // Z·S → M_TZS (NS)
    plan_tz: Option<SpgemmPlanGpu>,  // T·Z → M_Z (NS Q = ZSZ)
    plan_zh: Option<SpgemmPlanGpu>,  // Z·H → M_TZS (K0 B = Z·H)
    plan_bz: Option<SpgemmPlanGpu>,  // B·Z → M_K (K0 A = ZHZ)

    // ── Atom-level buffers ──
    v_buf: Buffer<f32>,      // atom potentials V[N] for device Hscc (R13)
    n_orb_buf: Buffer<u32>,  // physical orbitals per atom (R14)
    /// Packed Mulliken out [2·N]: q at [0..N), q_dum at [N..2N) — one
    /// device write, one host read per iteration (F7).
    qpack_buf: Buffer<f32>,
    qpack_host: Vec<f32>,

    // ── Cross-mask maps (device) ──
    hs_to_kt: Buffer<i32>,   // hs block (i,j) → k block (j,i) or -1 (R7)
    k_to_z: Buffer<i32>,     // k block (i,j) → z block (i,j) or -1 (R8b)

    // ── Reduction / diagnostic scratch ──
    trace_buf: Buffer<f32>,
    residual_buf: Buffer<f32>,
    ksq_buf: Buffer<f32>,
    emin_buf: Buffer<f32>,
    emax_buf: Buffer<f32>,
    gersh_emin: Buffer<f32>,  // per-orbital partials for Gershgorin
    gersh_emax: Buffer<f32>,
    reduce_partial: Buffer<f32>,
    reduce_a: Buffer<f32>,
    reduce_b: Buffer<f32>,

    // Host scratch for K download (`k_values_host`, force/energy diagnostics).
    k_host: Vec<f32>,
    a_ht_host: Vec<f32>,      // R_H download scratch (once per SCC)
    s_inf: f32,

    // ── System parameters ──
    nocc: f32,
    /// True when `t_ks` holds K·S for the *current* K (set at TC2 convergence
    /// — the converged K is the one that produced t_ks). Mulliken reuses it
    /// instead of a spare SpGEMM (R8a).
    t_ks_valid: bool,
}

impl SparseSystemWorkspace {
    /// Build a persistent workspace for a frozen topology.
    ///
    /// `h0` and `s` are the initial H0 and S matrices (their values will be
    /// overwritten per geometry, but their CSR structure defines M_HS).
    /// `k_mask`/`z_mask` define M_K (density kernel) and M_Z (inverse
    /// overlap) — independent geometric supports (second review R3).
    /// `n_orb` is the physical orbital count per atom (1 or 4 — masks the
    /// padded BSR4 lanes in Mulliken/trace/Hscc, R14). `nocc` is the number
    /// of occupied orbitals.
    ///
    /// All GPU structures, buffers, and symbolic plans are allocated and
    /// uploaded once here. No allocation happens in subsequent SCC/force
    /// calls.
    pub fn new(
        gpu: SparseBsr4Gpu,
        h0: &Bsr4Matrix,
        s: &Bsr4Matrix,
        k_mask: &(Vec<u32>, Vec<u32>),
        z_mask: &(Vec<u32>, Vec<u32>),
        n_orb: &[u8],
        nocc: f32,
    ) -> Result<Self> {
        let n_atom = h0.n_atom;
        assert_eq!(s.n_atom, n_atom, "H0 and S must have same n_atom");
        if n_orb.len() != n_atom {
            return Err(DftbError::InvalidInput(format!(
                "SparseSystemWorkspace::new: n_orb len {} != n_atom {n_atom}", n_orb.len()
            )));
        }
        for (i, &n) in n_orb.iter().enumerate() {
            if n == 0 || n > BS as u8 {
                return Err(DftbError::InvalidInput(format!(
                    "SparseSystemWorkspace::new: n_orb[{i}]={n} — BSR4 path needs 1..=4"
                )));
            }
        }

        // Independent masks (R3). M_HS = H/S support; M_K, M_Z separate.
        let m_hs = (h0.row_ptr.clone(), h0.col_idx.clone());
        let m_k = k_mask.clone();
        let m_z = z_mask.clone();
        // Multiply-truncate (standard SP2 idiom): intermediates live on the
        // PRESCRIBED masks, not the symbolic product — product(M_K,M_HS)
        // reaches r_k+r_hs (~19 Å → ~500 blocks/row, exceeds the SpGEMM
        // local-mem cap). T=K·S on M_K: Tr(KS) and Mulliken need only the
        // diagonal blocks (always in-mask), and the dropped off-diagonal
        // terms are bounded by the same K-decay that justifies M_K itself.
        let m_t_ks = m_k.clone();
        let m_t_zs = m_z.clone();
        // A = H·(KS) for the stationarity residual R_H (R11) — evaluated on
        // M_K, consistent with R_I (M_K) and R_Z (M_Z): the residual is
        // masked like the algebra it diagnoses. (product(M_HS,M_K) is NOT
        // symmetric when r_hs≠r_k and reaches ~r_hs+r_k ≈ 20 Å.)
        let m_ht = m_k.clone();

        // Build GPU structures (immutable, shared).
        let hs_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_hs)?);
        let k_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_k)?);
        let z_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_z)?);
        let t_ks_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_t_ks)?);
        let t_zs_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_t_zs)?);
        let ht_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_ht)?);

        // Upload initial H0 and S values.
        let h0_mat = GpuBsrMatrix { struct_: hs_struct.clone(), values: gpu.buf_f32(&h0.values)? };
        let s_mat = GpuBsrMatrix { struct_: hs_struct.clone(), values: gpu.buf_f32(&s.values)? };
        let h_scc = GpuBsrMatrix::zero(&gpu, &hs_struct)?;

        // Z on M_Z; K and K0 intermediates on M_K.
        let z = GpuBsrMatrix::zero(&gpu, &z_struct)?;
        let znew = GpuBsrMatrix::zero(&gpu, &z_struct)?;
        let qz = GpuBsrMatrix::zero(&gpu, &z_struct)?;
        let k = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let knew = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let k_best = gpu.zero_f32(k_struct.nblock * BS2)?;   // best-K snapshot for TC2 plateau
        let q = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let a_zhz = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let z_on_k = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let t_ks = GpuBsrMatrix::zero(&gpu, &t_ks_struct)?;
        let t_zs = GpuBsrMatrix::zero(&gpu, &t_zs_struct)?;
        let b_zh = GpuBsrMatrix::zero(&gpu, &t_zs_struct)?;
        let a_ht = GpuBsrMatrix::zero(&gpu, &ht_struct)?;

        // Cross-mask maps.
        let hs_to_kt_host = block_map_transpose(&m_hs, &m_k, n_atom);
        let k_to_z_host = block_map_same(&m_k, &m_z, n_atom);
        let ht_transpose = block_map_transpose(&m_ht, &m_ht, n_atom)
            .iter().map(|&x| x as u32).collect::<Vec<u32>>();
        for (b, &t) in ht_transpose.iter().enumerate() {
            if t == u32::MAX {
                return Err(DftbError::InvalidInput(format!(
                    "M_HT not symmetric at block {b} — product masks must be symmetric"
                )));
            }
        }
        let hs_to_kt = gpu.buf_i32(&hs_to_kt_host)?;
        let k_to_z = gpu.buf_i32(&k_to_z_host)?;

        // Atom-level buffers.
        let v_buf = gpu.zero_f32(n_atom)?;
        let n_orb_u32: Vec<u32> = n_orb.iter().map(|&n| n as u32).collect();
        let n_orb_buf = gpu.buf_u32(&n_orb_u32)?;
        let qpack_buf = gpu.zero_f32(2 * n_atom)?;
        let qpack_host = vec![0.0f32; 2 * n_atom];

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
        let z_dummy = Bsr4Matrix::from_structure(n_atom, m_z.0.clone(), m_z.1.clone())?;
        let t_ks_dummy = Bsr4Matrix::from_structure(n_atom, m_t_ks.0.clone(), m_t_ks.1.clone())?;
        let t_zs_dummy = Bsr4Matrix::from_structure(n_atom, m_t_zs.0.clone(), m_t_zs.1.clone())?;

        let plan_ks = build_plan(&k_dummy, &hs_dummy, &m_t_ks, "plan_ks");
        let plan_tk = build_plan(&t_ks_dummy, &k_dummy, &m_k, "plan_tk");
        let plan_zs = build_plan(&z_dummy, &hs_dummy, &m_t_zs, "plan_zs");
        let plan_tz = build_plan(&t_zs_dummy, &z_dummy, &m_z, "plan_tz");
        let plan_zh = build_plan(&z_dummy, &hs_dummy, &m_t_zs, "plan_zh");
        let plan_bz = build_plan(&t_zs_dummy, &z_dummy, &m_k, "plan_bz");

        // Reduction scratch buffers.
        let trace_buf = gpu.zero_f32(1)?;
        let residual_buf = gpu.zero_f32(1)?;
        let ksq_buf = gpu.zero_f32(1)?;
        let emin_buf = gpu.zero_f32(1)?;
        let emax_buf = gpu.zero_f32(1)?;
        let gersh_emin = gpu.zero_f32(n_atom * BS)?;
        let gersh_emax = gpu.zero_f32(n_atom * BS)?;
        let reduce_wg = gpu.config().reduce_wg as usize;
        let max_nblock = k_struct.nblock
            .max(t_ks_struct.nblock)
            .max(t_zs_struct.nblock)
            .max(hs_struct.nblock);
        let reduce_len = (n_atom.max(max_nblock * BS2) + reduce_wg - 1) / reduce_wg;
        let reduce_len = reduce_len.max(1);
        let reduce_partial = gpu.zero_f32(reduce_len)?;
        let reduce_a = gpu.zero_f32(reduce_len)?;
        let reduce_b = gpu.zero_f32(reduce_len)?;
        let k_host = vec![0.0f32; k_struct.nblock * BS2];
        let a_ht_host = vec![0.0f32; ht_struct.nblock * BS2];
        let s_inf = inf_norm(s); // may be 0 at construction if S is a zero placeholder

        Ok(Self {
            gpu,
            n_atom,
            m_hs,
            m_k,
            m_z,
            m_t_ks,
            m_t_zs,
            m_ht,
            ht_transpose,
            hs_struct,
            k_struct,
            z_struct,
            t_ks_struct,
            t_zs_struct,
            ht_struct,
            h0: h0_mat,
            s: s_mat,
            h_scc,
            z,
            znew,
            qz,
            k,
            knew,
            k_best,
            q,
            a_zhz,
            z_on_k,
            t_ks,
            t_zs,
            b_zh,
            a_ht,
            plan_ks,
            plan_tk,
            plan_zs,
            plan_tz,
            plan_zh,
            plan_bz,
            v_buf,
            n_orb_buf,
            qpack_buf,
            qpack_host,
            hs_to_kt,
            k_to_z,
            trace_buf,
            residual_buf,
            ksq_buf,
            emin_buf,
            emax_buf,
            gersh_emin,
            gersh_emax,
            reduce_partial,
            reduce_a,
            reduce_b,
            k_host,
            a_ht_host,
            s_inf,
            nocc,
            t_ks_valid: false,
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

    /// Current device K matrix (for `SparseDWWorkspace::build_dw_into`).
    pub fn k(&self) -> &GpuBsrMatrix { &self.k }
    /// Current device H_scc (for `SparseDWWorkspace::build_dw_into`).
    pub fn h_scc(&self) -> &GpuBsrMatrix { &self.h_scc }
    /// Shared K structure (M_K) — for `SparseDWWorkspace::new`.
    pub fn k_struct(&self) -> &Arc<GpuBsrStructure> { &self.k_struct }
    /// Shared H/S structure (M_HS) — for `SparseDWWorkspace::new`.
    pub fn hs_struct(&self) -> &Arc<GpuBsrStructure> { &self.hs_struct }
    /// Host map: hs block (i,j) → k block (j,i) or −1 (force contraction).


    /// Build H_scc on the device from atom potentials (R13):
    /// upload `v` (n_atom f32) then one `bsr4_build_Hscc` launch —
    /// `H = H0 + ½S(V_i+V_j)` on physical lanes only. No H_scc host transfer.
    pub fn build_hscc_from_v(&mut self, v: &[f32]) -> Result<()> {
        if v.len() != self.n_atom {
            return Err(DftbError::InvalidInput(format!(
                "build_hscc_from_v: v len {} != n_atom {}", v.len(), self.n_atom
            )));
        }
        for (i, &x) in v.iter().enumerate() {
            if !x.is_finite() {
                return Err(DftbError::InvalidInput(format!("build_hscc_from_v: v[{i}]={x}")));
            }
        }
        self.v_buf.write(v).enq().map_err(crate::qmqm::gpu_runtime::map_ocl_err)?;
        self.gpu.build_hscc_dev(
            &self.hs_struct, &self.h0.values, &self.s.values,
            &self.v_buf, &self.n_orb_buf, &self.h_scc.values,
        )
    }

    /// Sparse masked band energy `Tr(K·H0)` on device (R7) — one reduction
    /// + one scalar read, no `k_to_dense`/`trace_ab`. Caller multiplies by 2
    /// (spin) and adds E_scc/E_rep.
    pub fn trace_kh0_dev(&mut self) -> Result<f32> {
        self.gpu.trace_hk_to_dev(
            self.hs_struct.nblock, &self.h0.values, &self.k.values, &self.hs_to_kt,
            &self.reduce_partial, &self.reduce_a, &self.reduce_b, &self.residual_buf,
        )?;
        let mut tr = [0.0f32; 1];
        self.gpu.read_f32(&self.residual_buf, &mut tr)?;
        Ok(tr[0])
    }

    /// Hamiltonian stationarity residual (R11):
    ///   A = H_scc·(K·S);  R_H = ‖A − Aᵀ‖_F / (2‖A‖_F + ε)
    /// because (HKS)ᵀ = SKH for symmetric S,K,H. One extra SpGEMM on M_HT
    /// (T not symmetric → intersection kernel), one O(nnz) download, host
    /// f64 norms. Once per SCC finalization — not in the inner loop.
    pub fn rh_stationarity(&mut self) -> Result<f64> {
        if !self.t_ks_valid {
            self.spgemm_ks();
            self.t_ks_valid = true;
        }
        self.gpu.spgemm_masked_dev(&self.h_scc, &self.t_ks, &self.a_ht)?;
        self.gpu.read_f32(&self.a_ht.values, &mut self.a_ht_host)?;
        // ‖A − Aᵀ‖ and ‖A‖ on host f64 over M_HT.
        let rp = &self.m_ht.0;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..self.n_atom {
            for p in (rp[i] as usize)..(rp[i + 1] as usize) {
                let t = self.ht_transpose[p] as usize;
                for l in 0..BS2 {
                    let a = self.a_ht_host[p * BS2 + l] as f64;
                    let at = self.a_ht_host[t * BS2 + (l % BS) * BS + l / BS] as f64;
                    num += (a - at) * (a - at);
                    den += a * a;
                }
            }
        }
        Ok(num.sqrt() / (2.0 * den.sqrt() + 1e-30))
    }

    /// Read current Mulliken charges q_A = 2·Tr_phys((KS)_AA) from device.
    /// Reuses `t_ks` when it holds K·S for the current K (R8a — set by a
    /// converged `tc2_purify`); otherwise recomputes T=K·S once.
    /// Returns `(q_phys, q_dum)` — `q_dum` is the dummy-lane occupation and
    /// should be ~0 (R14 leak diagnostic).
    pub fn mulliken_charges(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        if !self.t_ks_valid {
            self.spgemm_ks();
            self.t_ks_valid = true;
        }
        self.gpu.mulliken_to_dev(
            &self.t_ks_struct, &self.t_ks.values, &self.n_orb_buf, &self.qpack_buf,
        )?;
        self.gpu.read_f32(&self.qpack_buf, &mut self.qpack_host)?;
        let (q, qd) = self.qpack_host.split_at(self.n_atom);
        Ok((q.to_vec(), qd.to_vec()))
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

    // ── Newton-Schulz inverse (device-resident) ──

    /// Compute Z ≈ S⁻¹ via Newton-Schulz on persistent GPU buffers.
    ///
    /// Products use the precomputed symbolic plans `plan_zs`/`plan_tz`
    /// (intersection kernel fallback if plan build failed). Residual is
    /// `||I−T||_F/√N` measured **on the device** — `identity_residual_to_dev`
    /// writes the *squared* norm into `residual_buf`, one scalar read per
    /// iteration (the N4 bug was a missing sqrt in the old scalar path, plus
    /// stale off-diagonal Z after a diagonal-only identity write).
    ///
    /// `warm`: if true, keep the persistent Z from the previous geometry and
    /// NS-correct it against the new S (one step gives ≈ Z − Z·δS·Z, the
    /// first-order inverse correction — few iterations for small steps).
    /// If a warm attempt stalls, diverges, or hits non-finite residual, we
    /// restart cold from αI once; cold failure is a loud Err.
    /// No `Buffer::builder` / `Kernel::builder` / matrix transfer in the loop.
    pub fn compute_z(&mut self, max_iter: usize, tol: f32, stall: usize, warm: bool) -> Result<(f32, usize)> {
        let n_orb = (self.n_atom * BS) as f32;
        let nblock = self.z_struct.nblock;
        let alpha = 1.0 / self.s_inf;
        let n_attempts = if warm { 2 } else { 1 };
        let mut last_err = String::new();

        for attempt in 0..n_attempts {
            let cold = attempt == n_attempts - 1 || !warm;
            if cold {
                // Full-write identity: every structural entry (R6).
                self.gpu.build_identity_dev(&self.z_struct, &self.z.values)?;
                self.gpu.scale_dev(nblock, alpha, &self.z.values)?;
            }
            let mut prev_rz = f32::INFINITY;
            let mut stall_count = 0;
            let mut rz = f32::INFINITY;
            let mut restart = false;
            for iter in 0..max_iter {
                // T = Z·S (planned; S symmetric)
                match &self.plan_zs {
                    Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.z, &self.s, plan, &self.t_zs)?,
                    None => self.gpu.spgemm_bsym_dev(&self.z, &self.s, &self.t_zs)?,
                }
                // ||I−T||² → device scalar → one host read; sqrt on host.
                self.gpu.identity_residual_to_dev(
                    &self.t_zs_struct, &self.t_zs.values,
                    &self.reduce_partial, &self.reduce_a, &self.reduce_b, &self.residual_buf,
                )?;
                let mut r2 = [0.0f32; 1];
                self.gpu.read_f32(&self.residual_buf, &mut r2)?;
                rz = r2[0].sqrt() / n_orb.sqrt();
                if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                    eprintln!("  Newton-Schulz (workspace) iter {iter}: R_Z = {rz:e}  (device ||I−T||_F/√N)");
                }
                if !rz.is_finite() {
                    last_err = format!("compute_z: non-finite R_Z={rz:e} at iter {iter}");
                    restart = true;
                    break;
                }
                if rz < tol {
                    return Ok((rz, iter + 1));
                }
                if iter > 0 && rz > 0.9 * prev_rz {
                    stall_count += 1;
                    if stall_count >= stall {
                        last_err = format!("compute_z: stalled after {} iters, R_Z={rz:e} (tol={tol:e})", iter + 1);
                        restart = true;
                        break;
                    }
                } else {
                    stall_count = 0;
                }
                prev_rz = rz;
                // Q = T·Z (planned; Z symmetric right operand) → M_Z
                match &self.plan_tz {
                    Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.t_zs, &self.z, plan, &self.qz)?,
                    None => self.gpu.spgemm_bsym_dev(&self.t_zs, &self.z, &self.qz)?,
                }
                self.gpu.axpby_dev(nblock, 2.0, &self.z.values, -1.0, &self.qz.values, &self.znew.values)?;
                self.gpu.symmetrize_dev(nblock, &self.z_struct.transpose_block(), &self.znew.values)?;
                std::mem::swap(&mut self.z.values, &mut self.znew.values);
            }
            if restart && !cold {
                eprintln!("  compute_z: warm start failed ({last_err}) — restarting cold from αI");
                continue;
            }
            if !restart && rz.is_finite() && rz >= tol {
                last_err = format!("compute_z: exhausted {max_iter} iters, final R_Z={rz:e} (tol={tol:e})");
            }
            return Err(DftbError::InvalidInput(last_err));
        }
        Err(DftbError::InvalidInput(format!(
            "compute_z: all attempts failed, last: {last_err}"
        )))
    }

    // ── K0 construction (device-resident) ──

    /// K0 for a given device-resident H on M_HS (H0 or H_scc).
    ///
    /// Single `B = Z·H` product (R8b): Gershgorin bounds and the
    /// `A = B·Z` term share it. `Z` is then projected onto `M_K` via the
    /// restrict map for the axpby — the product `ZHZ` uses the *full* M_Z
    /// support before projection.
    fn compute_k0_impl(&mut self, h: &GpuBsrMatrix, padding: f32) -> Result<(f32, f32)> {
        // B = Z·H on M_TZS (planned; H symmetric right operand).
        match &self.plan_zh {
            Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.z, h, plan, &self.b_zh)?,
            None => self.gpu.spgemm_bsym_dev(&self.z, h, &self.b_zh)?,
        }
        // Gershgorin bounds of B — persistent buffers, 2 scalar reads.
        self.gpu.gershgorin_to_dev(
            &self.b_zh.struct_, &self.b_zh.values,
            &self.gersh_emin, &self.gersh_emax,
            &self.reduce_a, &self.reduce_b, &self.emin_buf, &self.emax_buf,
        )?;
        let mut lo = [0.0f32; 1];
        let mut hi = [0.0f32; 1];
        self.gpu.read_f32(&self.emin_buf, &mut lo)?;
        self.gpu.read_f32(&self.emax_buf, &mut hi)?;
        let (mut emin, mut emax) = (lo[0], hi[0]);
        let span = (emax - emin).abs() * padding;
        emin -= span;
        emax += span;
        if !emin.is_finite() || !emax.is_finite() || emax <= emin {
            return Err(DftbError::InvalidInput(format!(
                "compute_k0: bad ZH Gershgorin emin={emin} emax={emax}"
            )));
        }

        // A = B·Z on M_K (planned; Z symmetric right operand).
        match &self.plan_bz {
            Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.b_zh, &self.z, plan, &self.a_zhz)?,
            None => self.gpu.spgemm_bsym_dev(&self.b_zh, &self.z, &self.a_zhz)?,
        }

        // Z|M_K via restrict map, then K0 = (emax·Z − A)/Δε on M_K.
        self.gpu.restrict_dev(self.k_struct.nblock, &self.k_to_z, &self.z.values, &self.z_on_k.values)?;
        let delta = (emax - emin).max(1e-12);
        self.gpu.axpby_dev(
            self.k_struct.nblock, emax / delta, &self.z_on_k.values,
            -1.0 / delta, &self.a_zhz.values, &self.k.values,
        )?;
        self.gpu.symmetrize_dev(self.k_struct.nblock, &self.k_struct.transpose_block(), &self.k.values)?;
        self.t_ks_valid = false;
        Ok((emin, emax))
    }

    /// K₀ from H0. Returns (emin, emax) — the padded spectral bounds.
    pub fn compute_k0(&mut self, padding: f32) -> Result<(f32, f32)> {
        let h0 = GpuBsrMatrix { struct_: self.h0.struct_.clone(), values: self.h0.values.clone() };
        self.compute_k0_impl(&h0, padding)
    }

    /// K₀ from the current **H_scc** (not H0). Call after `build_hscc_dev`.
    pub fn compute_k0_from_hscc(&mut self, padding: f32) -> Result<(f32, f32)> {
        let h = GpuBsrMatrix { struct_: self.h_scc.struct_.clone(), values: self.h_scc.values.clone() };
        self.compute_k0_impl(&h, padding)
    }

    // ── TC2 purification (device-resident) ──

    /// Run TC2 purification on the current K. Returns (final r_I, Tr, iters).
    /// K is updated in place on the device.
    ///
    /// `tol` is the **relative** idempotency tolerance (R12):
    /// `r_I = ‖KSK−K‖_F / max(‖K‖_F, ε)`. Raw ‖KSK−K‖_F saturates at the
    /// f32 SpGEMM floor (~1e-5·‖K‖) — a raw absolute threshold mislabels
    /// normal f32 saturation as non-convergence.
    /// `Tr(KS)` counts physical lanes only (masked trace, R14).
    pub fn tc2_purify(&mut self, max_iter: usize, tol: f32, check_every: usize) -> Result<(f32, f32, usize)> {
        // Normalization scale: ‖K‖_F of the incoming K (K0 or previous
        // iterate — drifts little during purification).
        self.gpu.frob_sq_to_dev(
            self.k_struct.nblock * BS2, &self.k.values,
            &self.reduce_partial, &self.reduce_a, &self.reduce_b, &self.ksq_buf,
        )?;
        let mut ksq = [0.0f32; 1];
        self.gpu.read_f32(&self.ksq_buf, &mut ksq)?;
        let k_norm = ksq[0].sqrt().max(1e-30);

        let mut best_r_i = f32::INFINITY;
        let mut best_tr = 0.0f32;
        let mut best_snapshotted = false;
        let mut last_tr = 0.0f32;
        let mut last_r_i = f32::INFINITY;
        let mut n_stagnant = 0usize;
        let mut trace_locked = false;
        self.t_ks_valid = false;

        for iter in 0..max_iter {
            let do_check = (iter % check_every == 0) || (iter == max_iter - 1);

            // T = K·S, Q = T·K, trace = Tr(T), optionally R_I = ||Q-K||.
            match &self.plan_ks {
                Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.k, &self.s, plan, &self.t_ks)?,
                None => self.gpu.spgemm_bsym_dev(&self.k, &self.s, &self.t_ks)?,
            }
            self.gpu.trace_ks_to_dev(
                &self.t_ks_struct, &self.t_ks.values, &self.n_orb_buf,
                &self.reduce_partial, &self.reduce_a, &self.reduce_b, &self.trace_buf,
            )?;
            // ── ENFORCED RESTORING INVARIANT: Tr(KS) ≈ Nocc (AGENTS.md f32 rule) ──
            // TC2's polynomials λ→λ² and λ→2λ−λ² are UNCONDITIONALLY divergent
            // outside [0,1]. Under f32 + masked SpGEMM the spectrum leaks out
            // (observed on 1648 atoms: Tr drifted 2627→2655 with the excess
            // DOUBLING each iteration — the exact λ=1+δ ⇒ λ²=1+2δ signature).
            // Tr(αKS)=α·Tr(KS) is exactly linear, so rescaling K pins the trace
            // and caps that growth. Triggered only outside the f32 jitter band
            // (~1e-5 relative) so healthy iterations are untouched.
            let mut tr_now = [0.0f32; 1];
            self.gpu.read_f32(&self.trace_buf, &mut tr_now)?;
            if !tr_now[0].is_finite() {
                return Err(DftbError::InvalidInput(format!(
                    "TC2 trace non-finite at iter {iter}: Tr(KS)={}", tr_now[0]
                )));
            }
            let dev_rel = ((tr_now[0] - self.nocc) as f64 / self.nocc.max(1.0) as f64).abs();
            // The guard applies ONLY in the endgame. During the transient the
            // trace legitimately swings far from Nocc (driving it there is
            // exactly what the TC2 branch does — Tr started at 2841 vs
            // Nocc=2627 here), so rescaling then would destroy the iteration.
            // Once TC2 has locked onto Nocc, any later drift is f32 noise /
            // spectral leakage and MUST be corrected.
            // Lock requires BOTH a near-Nocc trace and a small idempotency
            // residual (previous check): the TC2 transient *crosses* Nocc on
            // its way (observed Tr=2628.19 at iter 2 while R_I was still
            // O(1)), so the trace alone must not arm the guard.
            if dev_rel < TC2_TRACE_LOCK_REL && last_r_i < TC2_LOCK_RI { trace_locked = true; }
            let guard = trace_locked && dev_rel > TC2_TRACE_GUARD_REL && tr_now[0] > 0.0;
            if guard {
                let alpha = (self.nocc as f64 / tr_now[0] as f64) as f32;
                self.gpu.scale_dev(self.k.struct_.nblock, alpha, &self.k.values)?;
                self.gpu.scale_dev(self.t_ks.struct_.nblock, alpha, &self.t_ks.values)?;
                // Trace is now exactly Nocc, so the sign-based branch is
                // undefined; bias it to the CONTRACTING branch (Knew=Q maps
                // λ→λ², shrinking in [0,1]) by writing Nocc·(1+ε).
                self.gpu.write_f32(&self.trace_buf, &[self.nocc * (1.0 + 1e-6)])?;
                if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                    eprintln!(
                        "    TC2 trace guard @iter {iter}: Tr={:.4} (dev_rel={dev_rel:.2e}) → rescaled K by {alpha:.8}",
                        tr_now[0]
                    );
                }
            }
            match &self.plan_tk {
                Some(plan) => self.gpu.spgemm_plan_bsym_dev(&self.t_ks, &self.k, plan, &self.q)?,
                None => self.gpu.spgemm_bsym_dev(&self.t_ks, &self.k, &self.q)?,
            }
            if do_check {
                self.gpu.idempotency_to_dev(
                    self.k.struct_.nblock, &self.q.values, &self.k.values,
                    &self.reduce_partial, &self.reduce_a, &self.reduce_b, &self.residual_buf,
                )?;
                // Trace of the state that produced this Q (post-guard).
                let tr = [if guard { self.nocc } else { tr_now[0] }];
                let mut ri_sq = [0.0f32; 1];
                self.gpu.read_f32(&self.residual_buf, &mut ri_sq)?;
                let ri = ri_sq[0].sqrt() / k_norm;
                if !ri.is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "TC2 residual non-finite at iter {iter}: R_I={}", ri
                    )));
                }
                last_tr = tr[0];
                last_r_i = ri;
                // JOINT snapshot criterion: min R_I among iterates whose trace
                // is ALSO valid. Selecting on R_I alone once captured an
                // iterate that was already mid-runaway (best R_I at the same
                // iter where Tr had drifted 0.7 e⁻) — the snapshot was then
                // rejected by the trace gate and the run hard-failed.
                if ri < best_r_i && (tr[0] - self.nocc).abs() <= TC2_TRACE_TOL {
                    best_r_i = ri;
                    best_tr = tr[0];
                    self.gpu.copy_f32(&self.k.values, &self.k_best, self.k.struct_.nblock * BS2)?;
                    best_snapshotted = true;
                    n_stagnant = 0;
                } else {
                    n_stagnant += 1;
                }
                if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                    eprintln!("    TC2 iter {iter:3}  R_I={ri:.4e}  Tr(KS)={:.6}", tr[0]);
                }

                if ri < tol {
                    if (tr[0] - self.nocc).abs() <= TC2_TRACE_TOL {
                        // Converged: K was not updated after this T — t_ks
                        // still holds K·S for the returned K (R8a reuse).
                        self.t_ks_valid = true;
                        return Ok((ri, tr[0], iter + 1));
                    }
                    if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                        eprintln!("  TC2 R_I={ri:e} < tol but Tr(KS)={} != Nocc={} (wrong-rank projector not accepted)", tr[0], self.nocc);
                    }
                }
                // NOTE (tried and REVERTED): a "stagnation break" that exits
                // after N non-improving checks. It fired during the NORMAL
                // TC2 descent (non-improving checks are routine while the
                // trace branch swings), cutting SiH4 off at R_I=1.2e-4 and
                // breaking gate_f at geometry step 17. The existing
                // ri > 10·best detector already catches the limit cycle, and
                // the trace guard above prevents the runaway that made the
                // limit cycle fatal. Do not re-add without a criterion that
                // can distinguish "descending slowly" from "at the floor".
                let stagnant = false;
                let _ = n_stagnant;
                if ri > best_r_i * 10.0 && best_r_i < f32::INFINITY || stagnant {
                    // Truncated-support TC2 has a floor set by the mask and
                    // the f32 kernels; above it the iteration limit-cycles.
                    // If the best iterate was already near a projector
                    // (Tr(KS)≈Nocc, R_I small), return the BEST-K snapshot —
                    // that is the honest achievable answer. Only a genuine
                    // blowup (best residual still large) is a hard error.
                    if best_snapshotted && (best_tr - self.nocc).abs() <= TC2_TRACE_TOL && best_r_i < 1e-2 {
                        eprintln!(
                            "  TC2 plateau at iter {iter} ({}): restoring best K (R_I={best_r_i:e}, Tr(KS)={best_tr}) — this IS the f32/mask floor",
                            if stagnant { "stagnant" } else { "oscillating" }
                        );
                        self.gpu.copy_f32(&self.k_best, &mut self.k.values, self.k.struct_.nblock * BS2)?;
                        self.spgemm_ks();            // T consistent with restored K
                        self.t_ks_valid = true;
                        return Ok((best_r_i, best_tr, iter + 1));
                    }
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

        if best_snapshotted && (best_tr - self.nocc).abs() <= TC2_TRACE_TOL && best_r_i < 1e-2 {
            eprintln!(
                "  TC2 exhausted {max_iter} iters — restoring best K at floor (R_I={best_r_i:e} < tol={tol:e} not reached, Tr(KS)={best_tr})"
            );
            self.gpu.copy_f32(&self.k_best, &mut self.k.values, self.k.struct_.nblock * BS2)?;
            self.spgemm_ks();
            self.t_ks_valid = true;
            return Ok((best_r_i, best_tr, max_iter));
        }
        Err(DftbError::InvalidInput(format!(
            "TC2 exhausted {max_iter} iters, final r_I={last_r_i:e} Tr(KS)={last_tr} (rel. tol={tol:e} Nocc={})",
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
    /// Returns (q_phys, q_dum, r_I, Tr, tc2_iters) — `r_I` is the *relative*
    /// idempotency residual `‖KSK−K‖/‖K‖` (R12).
    pub fn run_scc(
        &mut self,
        ns_max_iter: usize,
        ns_tol: f32,
        tc2_max_iter: usize,
        tc2_tol: f32,
        tc2_check_every: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, f32, f32, usize)> {
        // 1. Z ≈ S⁻¹ (cold start — one-shot purify API has no previous Z)
        let (_, _) = self.compute_z(ns_max_iter, ns_tol, 3, false)?;

        // 2. K0
        let _ = self.compute_k0(0.1)?;

        // 3. TC2
        let (r_i, tr, iters) = self.tc2_purify(tc2_max_iter, tc2_tol, tc2_check_every)?;

        // 4. Mulliken charges (reuses t_ks — converged TC2 leaves T = K·S)
        let (charges, q_dum) = self.mulliken_charges()?;

        Ok((charges, q_dum, r_i, tr, iters))
    }
}

/// For each block (i,j) in mask `a`, the index of block (j,i) in mask `b`,
/// or −1 when (j,i) is outside `b`. Used for `Tr(K·H)` (needs K_ji for
/// H_ij) and the R_H antisymmetry check.
fn block_map_transpose(
    a: &(Vec<u32>, Vec<u32>),
    b: &(Vec<u32>, Vec<u32>),
    n_atom: usize,
) -> Vec<i32> {
    let mut map = vec![-1i32; a.1.len()];
    for i in 0..n_atom {
        for pa in (a.0[i] as usize)..(a.0[i + 1] as usize) {
            let j = a.1[pa] as usize;
            // find block (j,i) in b's row j
            for pb in (b.0[j] as usize)..(b.0[j + 1] as usize) {
                if b.1[pb] as usize == i {
                    map[pa] = pb as i32;
                    break;
                }
            }
        }
    }
    map
}

/// For each block (i,j) in mask `a`, the index of block (i,j) in mask `b`,
/// or −1 when absent. Used to restrict Z (on M_Z) onto M_K.
fn block_map_same(
    a: &(Vec<u32>, Vec<u32>),
    b: &(Vec<u32>, Vec<u32>),
    n_atom: usize,
) -> Vec<i32> {
    let mut map = vec![-1i32; a.1.len()];
    for i in 0..n_atom {
        for pa in (a.0[i] as usize)..(a.0[i + 1] as usize) {
            let j = a.1[pa] as usize;
            for pb in (b.0[i] as usize)..(b.0[i + 1] as usize) {
                if b.1[pb] as usize == j {
                    map[pa] = pb as i32;
                    break;
                }
            }
        }
    }
    map
}

/// `||T − I||_F` from downloaded BSR values (f64 accum). Diagonal blocks subtract I.
/// Host-side contract check for the device residual (second review §14.0:
/// every diagnostic scalar must be cross-checked against a host-f64
/// recomputation of identical inputs). Used by tests; production NS reads the
/// device scalar only.
#[allow(dead_code)]
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

        let all4 = vec![4u8; n_atom];
        let mut ws = SparseSystemWorkspace::new(gpu, &h0, &s, &mask, &mask, &all4, nocc).unwrap();

        // Run full SCC pipeline. r_I is the RELATIVE residual ‖KSK−K‖/‖K‖ (R12).
        let (charges, q_dum, r_i, tr, iters) = ws.run_scc(30, 1e-4, 40, 1e-5, 1).unwrap();
        println!("SCC: {iters} TC2 iters, r_I={r_i:e} (relative), Tr(KS)={tr:.6}");
        println!("Mulliken charges: {:?}", &charges);
        let qd_max: f32 = q_dum.iter().map(|x| x.abs()).fold(0.0, f32::max);
        println!("Dummy occupation: {qd_max:e} (should be ~0, R14)");

        // Verify convergence — relative idempotency residual.
        assert!(r_i < 1e-5, "TC2 did not converge: r_I={r_i:e} (G2)");
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

    /// F1 contract test (second review §14.0): the device identity residual
    /// must agree with a host-f64 recomputation of the *identical* T. This is
    /// the cross-check that would have caught the N4 missing-sqrt bug.
    #[test]
    fn test_ns_device_residual_contract() {
        let Some(gpu) = try_gpu() else { return };
        let n_atom = 3;
        let nocc = 3.0f32;
        let mask = build_full_mask(n_atom);
        let h0 = random_symmetric_bsr4(n_atom, &mask, 42);
        let s = make_overlap_bsr4(n_atom, &mask, 99);
        let all4 = vec![4u8; n_atom];
        let mut ws = SparseSystemWorkspace::new(gpu, &h0, &s, &mask, &mask, &all4, nocc).unwrap();
        let (rz_reported, iters) = ws.compute_z(30, 1e-4, 3, false).unwrap();

        // Recompute T = Z·S and measure the same residual two ways.
        match &ws.plan_zs {
            Some(plan) => ws.gpu.spgemm_plan_bsym_dev(&ws.z, &ws.s, plan, &ws.t_zs).unwrap(),
            None => ws.gpu.spgemm_bsym_dev(&ws.z, &ws.s, &ws.t_zs).unwrap(),
        }
        ws.gpu.identity_residual_to_dev(
            &ws.t_zs_struct, &ws.t_zs.values,
            &ws.reduce_partial, &ws.reduce_a, &ws.reduce_b, &ws.residual_buf,
        ).unwrap();
        let mut r2 = [0.0f32; 1];
        ws.gpu.read_f32(&ws.residual_buf, &mut r2).unwrap();
        let n_orb = (n_atom * BS) as f32;
        let rz_dev = r2[0].sqrt() / n_orb.sqrt();
        let mut t_host = vec![0.0f32; ws.t_zs_struct.nblock * BS2];
        ws.gpu.read_f32(&ws.t_zs.values, &mut t_host).unwrap();
        let rz_host = host_identity_rz(&t_host, &ws.m_t_zs.0, &ws.m_t_zs.1, n_atom) / n_orb.sqrt();
        println!(
            "F1 contract: compute_z R_Z={rz_reported:e} ({iters} iters)  \
             dev={rz_dev:e}  host_f64={rz_host:e}"
        );
        // Device f32 reduction vs host f64: agree to f32 reduction accuracy.
        let scale = rz_host.max(1e-6);
        assert!(
            (rz_dev - rz_host).abs() < 0.1 * scale + 1e-7,
            "device residual {rz_dev:e} disagrees with host {rz_host:e} — contract bug (N4 class)"
        );
        assert!(rz_host < 1e-3, "Z is not a converged inverse: host R_Z={rz_host:e}");
    }

    /// F1/R6 regression: a second geometry must not inherit stale
    /// off-diagonal Z. Cold start writes every structural entry; warm start
    /// NS-corrects the previous Z against the new S.
    #[test]
    fn test_compute_z_second_geometry() {
        let Some(gpu) = try_gpu() else { return };
        let n_atom = 3;
        let nocc = 3.0f32;
        let mask = build_full_mask(n_atom);
        let h0 = random_symmetric_bsr4(n_atom, &mask, 42);
        let s1 = make_overlap_bsr4(n_atom, &mask, 99);
        let all4 = vec![4u8; n_atom];
        let mut ws = SparseSystemWorkspace::new(gpu, &h0, &s1, &mask, &mask, &all4, nocc).unwrap();

        let (rz1, it1) = ws.compute_z(30, 1e-4, 3, false).unwrap();
        println!("geom1 cold: {it1} iters R_Z={rz1:e}");

        // Perturbed geometry: small uniform scaling of S (stand-in for a
        // small displacement; keeps SPD and stays on the same mask).
        let mut s2 = s1.clone();
        for v in s2.values.iter_mut() { *v *= 1.02; }
        ws.upload_s(&s2).unwrap();

        // Warm start from previous Z — must converge, and the Z buffer still
        // holds old off-diagonals internally before correction.
        let (rz2, it2) = ws.compute_z(30, 1e-4, 3, true).unwrap();
        println!("geom2 warm: {it2} iters R_Z={rz2:e}");

        // Verify Z·S2 ≈ I on host (independent of device residual).
        ws.gpu.spgemm_bsym_dev(&ws.z, &ws.s, &ws.t_zs).unwrap();
        let mut t = vec![0.0f32; ws.t_zs_struct.nblock * BS2];
        ws.gpu.read_f32(&ws.t_zs.values, &mut t).unwrap();
        let n_orb = (n_atom * BS) as f32;
        let rz_host_warm = host_identity_rz(&t, &ws.m_t_zs.0, &ws.m_t_zs.1, n_atom) / n_orb.sqrt();
        println!("geom2 warm host R_Z={rz_host_warm:e}");
        assert!(rz_host_warm < 1e-3, "warm-start Z wrong vs S2: R_Z={rz_host_warm:e}");

        // Cold restart on the *dirty* Z buffer: must zero stale off-diagonals.
        let (rz3, it3) = ws.compute_z(30, 1e-4, 3, false).unwrap();
        ws.gpu.spgemm_bsym_dev(&ws.z, &ws.s, &ws.t_zs).unwrap();
        ws.gpu.read_f32(&ws.t_zs.values, &mut t).unwrap();
        let rz_host_cold = host_identity_rz(&t, &ws.m_t_zs.0, &ws.m_t_zs.1, n_atom) / n_orb.sqrt();
        println!("geom2 cold-on-dirty: {it3} iters R_Z={rz3:e} host={rz_host_cold:e}");
        assert!(rz_host_cold < 1e-3, "cold restart left stale Z: R_Z={rz_host_cold:e} (R6)");
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

        let all4 = vec![4u8; n_atom];
        let mut ws = SparseSystemWorkspace::new(gpu, &h0, &s, &mask, &mask, &all4, nocc).unwrap();

        // First run.
        let (_, _, r_i1, tr1, _) = ws.run_scc(30, 1e-4, 60, 1e-5, 1).unwrap();
        println!("Run 1: r_I={r_i1:e}, Tr={tr1:.6}");

        // Second run with same geometry (should give same result).
        let (_, _, r_i2, tr2, _) = ws.run_scc(30, 1e-4, 60, 1e-5, 1)
            .unwrap_or_else(|e| panic!("Run 2 failed (no skip): {e}"));
        println!("Run 2: R_I={r_i2:e}, Tr={tr2:.6}");

        // Both runs should converge to the same state.
        assert!((tr1 - tr2).abs() < 1e-4, "Tr mismatch: {tr1} vs {tr2}");
        assert!((r_i1 - r_i2).abs() < 1e-4, "R_I mismatch: {r_i1} vs {r_i2}");
    }
}
