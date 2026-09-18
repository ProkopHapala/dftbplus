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
    GpuBsrMatrix, GpuBsrStructure, PurifyStatus, SparseBsr4Gpu, SpgemmPlanGpu, TC2_LOCK_RI,
    TC2_TRACE_GUARD_REL, TC2_TRACE_LOCK_REL,
};
use crate::methods::sparse::sparse_forces::{pair_gather_adj, HsPair, SkGpuPack};
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
    #[allow(dead_code)]
    m_z: (Vec<u32>, Vec<u32>),
    #[allow(dead_code)]
    m_t_ks: (Vec<u32>, Vec<u32>), // T = K·S support
    #[allow(dead_code)]
    m_t_zs: (Vec<u32>, Vec<u32>), // T = Z·S and B = Z·H support
    m_ht: (Vec<u32>, Vec<u32>), // A = H·(KS) support (R_H, final only)
    ht_transpose: Vec<u32>,     // host map: ht block (i,j) → block (j,i)

    // ── GPU structures (immutable, shared via Arc) ──
    hs_struct: Arc<GpuBsrStructure>,
    k_struct: Arc<GpuBsrStructure>,
    z_struct: Arc<GpuBsrStructure>,
    t_ks_struct: Arc<GpuBsrStructure>,
    t_zs_struct: Arc<GpuBsrStructure>,
    #[allow(dead_code)]
    ht_struct: Arc<GpuBsrStructure>,

    // ── Persistent GPU matrices ──
    // H/S (change with geometry, uploaded per set_coords)
    h0: GpuBsrMatrix,    // H0 on M_HS
    s: GpuBsrMatrix,     // S on M_HS
    h_scc: GpuBsrMatrix, // H_scc on M_HS (built on device from V — R13)

    // Z ≈ S⁻¹ on M_Z (recomputed/corrected per geometry)
    z: GpuBsrMatrix,    // Z on M_Z
    znew: GpuBsrMatrix, // Znew scratch on M_Z
    qz: GpuBsrMatrix,   // Q = T·Z = ZSZ on M_Z (NS)

    // K (density kernel, changes each SCC iteration)
    k: GpuBsrMatrix,      // K on M_K
    knew: GpuBsrMatrix,   // Knew scratch on M_K
    k_best: Buffer<f32>,  // best-K snapshot for TC2 plateau recovery
    q: GpuBsrMatrix,      // Q = T·K = KSK on M_K (TC2)
    a_zhz: GpuBsrMatrix,  // A = (Z·H)·Z restricted to M_K (K0)
    z_on_k: GpuBsrMatrix, // Z restricted to M_K (K0 axpby — R8b)

    // Intermediates
    t_ks: GpuBsrMatrix, // T = K·S on M_TKS
    t_zs: GpuBsrMatrix, // T = Z·S on M_TZS (NS)
    b_zh: GpuBsrMatrix, // B = Z·H on M_TZS (K0)
    a_ht: GpuBsrMatrix, // A = H_scc·(KS) on M_HT (R_H, once per SCC)

    // ── Symbolic plans (built once, reused forever) ──
    plan_ks: Option<SpgemmPlanGpu>,   // K·S → M_TKS
    plan_tk: Option<SpgemmPlanGpu>,   // T·K → M_K (TC2 Q = KSK)
    plan_zs: Option<SpgemmPlanGpu>,   // Z·S → M_TZS (NS)
    plan_tz: Option<SpgemmPlanGpu>,   // T·Z → M_Z (NS Q = ZSZ)
    plan_zh: Option<SpgemmPlanGpu>,   // Z·H → M_TZS (K0 B = Z·H)
    plan_bz: Option<SpgemmPlanGpu>,   // B·Z → M_K (K0 A = ZHZ)
    plan_ht: Option<SpgemmPlanGpu>,   // H·T → M_HT (R11 stationarity residual, generic plan)
    plan_fk: Option<SpgemmPlanGpu>,   // F·K = (Z·H)·K → M_K (DMM double-commutator descent, G3)
    plan_tk_g: Option<SpgemmPlanGpu>, // T·W2 → M_K, GENERIC plan — W2=F·K is asymmetric; the bsym kernel would silently compute T·W2ᵀ (measured: 1.4e-2 error vs 1e-6 for symmetric operands)

    // ── Atom-level buffers ──
    v_buf: Buffer<f32>,     // atom potentials V[N] for device Hscc (R13)
    xyzu_buf: Buffer<f32>,  // packed (x,y,z Å, u Ha) per atom — γ kernels
    dq_buf: Buffer<f32>,    // Δq[N] input to the γ kernels
    gf_buf: Buffer<f32>,    // γ′ force out [4·N] (xyz used, w pad)
    n_orb_buf: Buffer<u32>, // physical orbitals per atom (R14)
    n_orb_host: Vec<u8>,    // host copy — debug verification masks (dmm_verify)
    /// Packed Mulliken out [2·N]: q at [0..N), q_dum at [N..2N) — one
    /// device write, one host read per iteration (F7).
    qpack_buf: Buffer<f32>,
    qpack_host: Vec<f32>,

    // ── Cross-mask maps (device) ──
    hs_to_kt: Buffer<i32>, // hs block (i,j) → k block (j,i) or -1 (R7)
    k_to_z: Buffer<i32>,   // k block (i,j) → z block (i,j) or -1 (R8b)

    // ── Reduction / diagnostic scratch ──
    trace_atom: Buffer<f32>,   // per-atom Tr(T_ii) partials → host f64 sum
    trace_atom_host: Vec<f32>, // persistent staging (no per-iter alloc)
    residual_buf: Buffer<f32>,
    ksq_buf: Buffer<f32>,
    emin_buf: Buffer<f32>,
    emax_buf: Buffer<f32>,
    gersh_emin: Buffer<f32>, // per-orbital partials for Gershgorin
    gersh_emax: Buffer<f32>,
    reduce_partial: Buffer<f32>,
    reduce_a: Buffer<f32>,
    reduce_b: Buffer<f32>,
    /// Host staging for the f64 reduction tail (PR3, §15.9): ≤128 partials
    /// downloaded once, summed in f64. Persistent — no per-iter alloc.
    reduce_tail_host: Vec<f32>,

    /// Kernel-event timing (RUST_DFTB_KTIME=1): true START/END events for
    /// the two TC2 SpGEMMs — unlike marker-elapsed, this is kernel exec.
    ktime_on: bool,
    ev_ks: Vec<ocl::Event>,
    ev_ksk: Vec<ocl::Event>,

    // Host scratch for K download (`k_values_host`, force/energy diagnostics).
    k_host: Vec<f32>,
    a_ht_host: Vec<f32>, // R_H download scratch (once per SCC)
    s_inf: f32,

    // ── System parameters ──
    nocc: f32,
    /// True when `t_ks` holds K·S for the *current* K (set at TC2 convergence
    /// — the converged K is the one that produced t_ks). Mulliken reuses it
    /// instead of a spare SpGEMM (R8a).
    t_ks_valid: bool,
    /// Number of endgame trace-guard rescales applied across all purify
    /// calls (S1 observability — a test can seed a λ>1 leak and assert the
    /// guard actually fired).
    guard_fires: usize,

    // ── P = K·S purifier (GPT-5.6 item 4, EXPERIMENTAL) ──
    // P is the density-side projector in the non-orthogonal metric:
    // P² = (KS)² = KSKS = KS = P iff KSK = K, Tr(P) = Nocc, and
    // q_A = 2·Tr(P_AA) directly. ONE generic planned P² SpGEMM per
    // iteration — no intermediate product to truncate. P is NOT symmetric
    // (no symmetrize step); K = P·Z is recovered after convergence for the
    // energy/force path. Mask: M_P = M_K for the prototype.
    m_p: (Vec<u32>, Vec<u32>),
    p_struct: Arc<GpuBsrStructure>,
    p: GpuBsrMatrix,
    q_p2: GpuBsrMatrix,             // P² product buffer (also P0-build scratch)
    r_p4: GpuBsrMatrix,             // P⁴=Q² buffer for TRS4 (item 5)
    pnew: GpuBsrMatrix,             // TC2 update (also P0-build scratch)
    p_best: Buffer<f32>,            // plateau snapshot
    plan_pp: Option<SpgemmPlanGpu>, // generic P·P on M_P
    plan_pz: Option<SpgemmPlanGpu>, // K = P·Z on M_K (Z symmetric → Bsym)
    p_to_tzs: Buffer<i32>,          // restrict map M_P-block → M_TZS-block
    p_t: Buffer<i32>,               // L3: M_P block (i,j) → M_P block (j,i) — for Tr(P·ZH)
    /// True when `p` holds the converged P = K·S (P-purifier path).
    p_valid: bool,

    // ── FF32-POLISH (manifest §4.12.2 r2): float-float (hi+lo) McWeeny
    // polish — f32 FMA only, ~46-bit effective, no native fp64. The hi
    // parts reuse `t_ks.values`/`q.values`; only the LO parts are new
    // persistent buffers.
    ff_t_lo: Buffer<f32>, // lo part of T_ff/U_ff on M_TKS
    ff_q_lo: Buffer<f32>, // lo part of Q_ff on M_K
    ff_v_lo: Buffer<f32>, // lo part of V_ff on M_K (hi reuses a_zhz)

    /// Production hi-accuracy switch (manifest §4.12.2 consolidated):
    /// when set, tc2_purify runs the terminal FF32 Phase B after the
    /// f32 phase. Env `RUST_DFTB_TC2_FF=1` still force-enables for
    /// studies.
    tc2_hiacc: bool,

    /// R17: persistent pair-physics state (sparse_hs.cl) — `Some` when the
    /// GPU pair path is active, `None` under the explicit CPU reference
    /// switch (`SparseDftbConfig::cpu_pair` / `RUST_DFTB_SPARSE_CPU`).
    pair: Option<GpuPairState>,
}

/// float4 staging → [f64;3] per atom (pair-force readback).
fn pull3(src: &[f32], dst: &mut [[f64; 3]]) {
    for (a, d) in dst.iter_mut().enumerate() {
        d[0] = src[4 * a] as f64;
        d[1] = src[4 * a + 1] as f64;
        d[2] = src[4 * a + 2] as f64;
    }
}

/// R17: device-resident pair data for `sparse_hs.cl` — packed once at
/// `init_pair_data` on the frozen topology; all buffers persistent.
/// No global atomics anywhere: pair kernels write per-pair records,
/// `force_gather` owns one atom each.
pub struct GpuPairState {
    n_pairs: usize,
    n_rep: usize,
    n_sp: usize,
    max_ctrl: usize,
    rep_max_int: usize,
    /// (r0_ang, w_ang, enabled, 0) — cosine taper on H/S blocks.
    taper: [f32; 4],
    pairs: Buffer<i32>,   // int4 {i, j, b_ij, b_ji} per M_HS pair
    pairs_k: Buffer<i32>, // M_K block of (i,j) or −1
    rpairs: Buffer<i32>,  // int2 {i, j} repulsive-range pairs
    fp_ptr: Buffer<u32>,  // per-atom CSR over pairs, (p<<1)|is_j
    fp_list: Buffer<i32>,
    rp_ptr: Buffer<u32>, // same over rpairs
    rp_list: Buffer<i32>,
    species: Buffer<u32>,  // species code per atom
    onsite: Buffer<f32>,   // float4 onsite energies per atom
    sk_meta: Buffer<i32>,  // int4 per species pair {n_ctrl, n_integ, present, 0}
    sk_parm: Buffer<f32>,  // float4 per species pair {dr, r_max} (Bohr)
    sk_ctrl: Buffer<f32>,  // nsp² · 8ch · max_ctrl controls
    rep_off: Buffer<i32>,  // record offsets per species pair (−1 none)
    rep_data: Buffer<f32>, // packed repulsive records
    pf: Buffer<f32>,       // 2·n_pairs float4 — per-pair (non_scc, shift)
    pf_rep: Buffer<f32>,   // n_rep float4 — (F_i xyz, E_pair)
    pe_rep: Buffer<f32>,   // n_rep — pair energies for the host sum
    f_nscc: Buffer<f32>,   // 4·n per-atom component outputs
    f_shift: Buffer<f32>,
    f_rep: Buffer<f32>,
    f_dc: Buffer<f32>,
    f_tot: Buffer<f32>,
    kdummy: Buffer<f32>,   // n per-atom |2K| dummy-lane partials
    row_sums: Buffer<f32>, // 4·n — s_inf row-abs-sum scratch
    sinf_out: Buffer<f32>, // 1 — s_inf reduce output
    // Host staging for the readback (no per-call alloc).
    f_host: Vec<f32>,  // 5·4·n packed [nscc, shift, rep, dc, tot]
    pe_host: Vec<f32>, // n_rep
    kd_host: Vec<f32>, // n
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
        // SC7 (manifest §15.9): optional modest fixed halo for the NS
        // intermediate T=Z·S — `Some(m)` uses it for M_TZS, `None` clones
        // M_Z (legacy). The halo improves in-mask T values (dropped terms
        // near the M_Z boundary feed in-mask products); never the full
        // product support.
        tzs_mask: Option<&(Vec<u32>, Vec<u32>)>,
        n_orb: &[u8],
        nocc: f32,
    ) -> Result<Self> {
        let n_atom = h0.n_atom;
        assert_eq!(s.n_atom, n_atom, "H0 and S must have same n_atom");
        if n_orb.len() != n_atom {
            return Err(DftbError::InvalidInput(format!(
                "SparseSystemWorkspace::new: n_orb len {} != n_atom {n_atom}",
                n_orb.len()
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
        // M_TKS = supp(M_K ∘ M_HS) — the TRUE symbolic product support.
        // FF32-POLISH measurement (§4.12.2 r3, masked-vs-dense f64 split)
        // proved the old M_TKS=M_K shortcut dropped real K·S tail terms
        // worth ~7e-6/product → THE 1.4e-5 R_I floor was mask truncation,
        // not f32 arithmetic (the dense-f64 decisive test could not see
        // it). Cost: left-operand degree grows toward r_k+r_hs reach —
        // check_left_degree fails loud if it exceeds MAX_LEFT_BLOCKS
        // (then the plan kernel needs left-row chunking — future work).
        // RUST_DFTB_TRUNC_PRODUCTS=1 keeps the legacy M_K store — required
        // on crystals larger than ~2(r_k+r_hs) across, where interior
        // M_TKS rows are dense and cannot fit the local-mem row cache.
        let trunc = crate::methods::sparse::gpu_sparse::trunc_products();
        let m_t_ks = if trunc {
            m_k.clone()
        } else {
            build_product_mask(n_atom, &m_k, &m_hs)
        };
        // Fail EARLY, not mid-SCC: T is a left operand in T·K/T·Q/T·Z, so
        // its row degree must fit the compiled MAX_LEFT_BLOCKS row cache.
        let deg_tks = (0..n_atom)
            .map(|i| m_t_ks.0[i + 1] - m_t_ks.0[i])
            .max()
            .unwrap_or(0);
        let cap = gpu.config().max_left_blocks as u32;
        if deg_tks > cap {
            return Err(DftbError::InvalidInput(format!(
                "SparseSystemWorkspace::new: M_TKS max row degree {deg_tks} > MAX_LEFT_BLOCKS={cap} \
                 (product support spans r_k+r_hs — dense interior rows on large crystals). \
                 Set RUST_DFTB_TRUNC_PRODUCTS=1 to store T on M_K (drops ~1e-5 tail terms), \
                 shrink r_k_ang/r_trunc_ang, or implement left-row chunking."
            )));
        }
        eprintln!(
            "[sparse ws] M_TKS nnz={} deg={} trunc_products={trunc}",
            m_t_ks.1.len(),
            deg_tks
        );
        let m_t_zs = tzs_mask.cloned().unwrap_or_else(|| m_z.clone());
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
        let h0_mat = GpuBsrMatrix {
            struct_: hs_struct.clone(),
            values: gpu.buf_f32(&h0.values)?,
        };
        let s_mat = GpuBsrMatrix {
            struct_: hs_struct.clone(),
            values: gpu.buf_f32(&s.values)?,
        };
        let h_scc = GpuBsrMatrix::zero(&gpu, &hs_struct)?;

        // Z on M_Z; K and K0 intermediates on M_K.
        let z = GpuBsrMatrix::zero(&gpu, &z_struct)?;
        let znew = GpuBsrMatrix::zero(&gpu, &z_struct)?;
        let qz = GpuBsrMatrix::zero(&gpu, &z_struct)?;
        let k = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let knew = GpuBsrMatrix::zero(&gpu, &k_struct)?;
        let k_best = gpu.zero_f32(k_struct.nblock * BS2)?; // best-K snapshot for TC2 plateau
                                                           // FF32-POLISH: lo parts of the float-float intermediates.
        let ff_t_lo = gpu.zero_f32(t_ks_struct.nblock * BS2)?;
        let ff_q_lo = gpu.zero_f32(k_struct.nblock * BS2)?;
        let ff_v_lo = gpu.zero_f32(k_struct.nblock * BS2)?;
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
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<u32>>();
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
        let xyzu_buf = gpu.zero_f32(4 * n_atom)?;
        let dq_buf = gpu.zero_f32(n_atom)?;
        let gf_buf = gpu.zero_f32(4 * n_atom)?;
        let n_orb_u32: Vec<u32> = n_orb.iter().map(|&n| n as u32).collect();
        let n_orb_buf = gpu.buf_u32(&n_orb_u32)?;
        let qpack_buf = gpu.zero_f32(2 * n_atom)?;
        let qpack_host = vec![0.0f32; 2 * n_atom];

        // Build symbolic plans for all recurring SpGEMMs.
        // SC3 (manifest §15.9): plan kernels are MANDATORY in production —
        // a build/upload failure is a hard error, never a silent switch to
        // the many-times-slower intersection kernel. `None` is reachable
        // only via the explicit diagnostic toggle RUST_DFTB_SPARSE_PLANS=0.
        let plans_on = crate::methods::sparse::gpu_sparse::sparse_plans_enabled();
        let build_plan = |a: &Bsr4Matrix,
                          b: &Bsr4Matrix,
                          c_mask: &(Vec<u32>, Vec<u32>),
                          label: &str|
         -> Result<Option<SpgemmPlanGpu>> {
            if !plans_on {
                return Ok(None);
            } // RUST_DFTB_SPARSE_PLANS=0 diagnostic toggle
            let plan = build_spgemm_plan_bsym(a, b, c_mask)
                .map_err(|e| DftbError::InvalidInput(format!("{label} plan build failed (plans are mandatory in production — fix the mask or set RUST_DFTB_SPARSE_PLANS=0 for diagnostic intersection mode): {e}")))?;
            gpu.upload_plan(&plan)
                .map(Some)
                .map_err(|e| DftbError::InvalidInput(format!("{label} plan upload failed: {e}")))
        };

        let k_dummy = Bsr4Matrix::from_structure(n_atom, m_k.0.clone(), m_k.1.clone())?;
        let hs_dummy = Bsr4Matrix::from_structure(n_atom, m_hs.0.clone(), m_hs.1.clone())?;
        let z_dummy = Bsr4Matrix::from_structure(n_atom, m_z.0.clone(), m_z.1.clone())?;
        let t_ks_dummy = Bsr4Matrix::from_structure(n_atom, m_t_ks.0.clone(), m_t_ks.1.clone())?;
        let t_zs_dummy = Bsr4Matrix::from_structure(n_atom, m_t_zs.0.clone(), m_t_zs.1.clone())?;

        let plan_ks = build_plan(&k_dummy, &hs_dummy, &m_t_ks, "plan_ks")?;
        let plan_tk = build_plan(&t_ks_dummy, &k_dummy, &m_k, "plan_tk")?;
        let plan_zs = build_plan(&z_dummy, &hs_dummy, &m_t_zs, "plan_zs")?;
        let plan_tz = build_plan(&t_zs_dummy, &z_dummy, &m_z, "plan_tz")?;
        let plan_zh = build_plan(&z_dummy, &hs_dummy, &m_t_zs, "plan_zh")?;
        let plan_bz = build_plan(&t_zs_dummy, &z_dummy, &m_k, "plan_bz")?;
        // plan_ht: A = H_scc·(K·S) on M_HT for the R_H stationarity
        // residual (R11). T is non-symmetric → generic plan kernel.
        let plan_ht = if plans_on {
            let plan =
                crate::methods::sparse::bsr4::build_spgemm_plan(&hs_dummy, &t_ks_dummy, &m_ht)
                    .map_err(|e| {
                        DftbError::InvalidInput(format!(
                            "plan_ht build failed (plans are mandatory in production): {e}"
                        ))
                    })?;
            Some(
                gpu.upload_plan(&plan)
                    .map_err(|e| DftbError::InvalidInput(format!("plan_ht upload failed: {e}")))?,
            )
        } else {
            None
        };
        // plan_fk: F·K = (Z·H)·K → M_K — DMM double-commutator descent
        // (G3). NOTE: `z` is S⁻¹ (Newton iter Z←2Z−ZSZ), NOT S^{-1/2} —
        // F = Z·H = S⁻¹H is the generalized-eigenvalue matrix and the
        // projector is P = K·S, so the tangent gradient of E = 2Tr(FP)
        // is the double commutator [P,[P,F]], S-selfadjointness making
        // δE = −2η·Σμ² ≤ 0. B-symmetric kernel (K symmetric right op).
        let plan_fk = build_plan(&t_zs_dummy, &k_dummy, &m_k, "plan_fk")?;
        // plan_tk_g: generic (non-bsym) twin of plan_tk for products
        // whose right operand is asymmetric — DMM W3 = T·(F·K).
        let plan_tk_g = if plans_on {
            let plan = crate::methods::sparse::bsr4::build_spgemm_plan(&t_ks_dummy, &k_dummy, &m_k)
                .map_err(|e| DftbError::InvalidInput(format!("plan_tk_g build failed: {e}")))?;
            Some(
                gpu.upload_plan(&plan).map_err(|e| {
                    DftbError::InvalidInput(format!("plan_tk_g upload failed: {e}"))
                })?,
            )
        } else {
            None
        };

        // ── P = K·S purifier buffers (GPT-5.6 item 4) ──
        // M_P = M_K for the prototype (same sparsity class as the legacy
        // T=KS mask; the error-budgeted widening is item 7).
        let m_p = m_k.clone();
        let p_struct = Arc::new(GpuBsrStructure::new(&gpu, n_atom, &m_p)?);
        let p = GpuBsrMatrix::zero(&gpu, &p_struct)?;
        let q_p2 = GpuBsrMatrix::zero(&gpu, &p_struct)?;
        let r_p4 = GpuBsrMatrix::zero(&gpu, &p_struct)?;
        let pnew = GpuBsrMatrix::zero(&gpu, &p_struct)?;
        let p_best = gpu.zero_f32(p_struct.nblock * BS2)?;
        let p_to_tzs_host = block_map_same(&m_p, &m_t_zs, n_atom);
        let p_to_tzs = gpu.buf_i32(&p_to_tzs_host)?;
        let p_t_host = block_map_transpose(&m_p, &m_p, n_atom);
        if p_t_host.iter().any(|&x| x < 0) {
            return Err(DftbError::InvalidInput(
                "M_P not symmetric — product masks must be symmetric".into(),
            ));
        }
        let p_t = gpu.buf_i32(&p_t_host)?;
        let p_dummy = Bsr4Matrix::from_structure(n_atom, m_p.0.clone(), m_p.1.clone())?;
        let plan_pp = if plans_on {
            let plan = crate::methods::sparse::bsr4::build_spgemm_plan(&p_dummy, &p_dummy, &m_p)
                .map_err(|e| {
                    DftbError::InvalidInput(format!(
                        "plan_pp build failed (plans are mandatory in production): {e}"
                    ))
                })?;
            Some(
                gpu.upload_plan(&plan)
                    .map_err(|e| DftbError::InvalidInput(format!("plan_pp upload failed: {e}")))?,
            )
        } else {
            None
        };
        let plan_pz = build_plan(&p_dummy, &z_dummy, &m_k, "plan_pz")?;

        // Reduction scratch buffers.
        let trace_atom = gpu.zero_f32(n_atom)?;
        let trace_atom_host = vec![0.0f32; n_atom];
        let residual_buf = gpu.zero_f32(1)?;
        let ksq_buf = gpu.zero_f32(1)?;
        let emin_buf = gpu.zero_f32(1)?;
        let emax_buf = gpu.zero_f32(1)?;
        let gersh_emin = gpu.zero_f32(n_atom * BS)?;
        let gersh_emax = gpu.zero_f32(n_atom * BS)?;
        let reduce_wg = gpu.config().reduce_wg as usize;
        let max_nblock = k_struct
            .nblock
            .max(t_ks_struct.nblock)
            .max(t_zs_struct.nblock)
            .max(hs_struct.nblock);
        let reduce_len = (n_atom.max(max_nblock * BS2) + reduce_wg - 1) / reduce_wg;
        let reduce_len = reduce_len.max(1);
        let reduce_partial = gpu.zero_f32(reduce_len)?;
        let reduce_a = gpu.zero_f32(reduce_len)?;
        let reduce_b = gpu.zero_f32(reduce_len)?;
        let reduce_tail_host =
            vec![0.0f32; crate::methods::sparse::gpu_sparse::SparseBsr4Gpu::REDUCE_TAIL];
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
            xyzu_buf,
            dq_buf,
            gf_buf,
            n_orb_buf,
            n_orb_host: n_orb.to_vec(),
            qpack_buf,
            qpack_host,
            hs_to_kt,
            k_to_z,
            trace_atom,
            trace_atom_host,
            residual_buf,
            ksq_buf,
            reduce_tail_host,
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
            ktime_on: std::env::var("RUST_DFTB_KTIME")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false),
            ev_ks: Vec::new(),
            ev_ksk: Vec::new(),
            guard_fires: 0,
            m_p,
            p_struct,
            p,
            q_p2,
            r_p4,
            pnew,
            p_best,
            plan_pp,
            plan_pz,
            plan_ht,
            plan_fk,
            plan_tk_g,
            p_to_tzs,
            p_t,
            p_valid: false,
            ff_t_lo,
            ff_q_lo,
            ff_v_lo,
            tc2_hiacc: false,
            pair: None,
        })
    }

    /// Enable/disable the terminal FF32 polish phase in `tc2_purify`
    /// (production hi-accuracy mode). `SparseDftb` drives this from
    /// `SparseDftbConfig::tc2_hiacc`.
    pub fn set_tc2_hiacc(&mut self, on: bool) {
        self.tc2_hiacc = on;
    }

    // ── Accessors ──

    pub fn n_atom(&self) -> usize {
        self.n_atom
    }
    pub fn nocc(&self) -> f32 {
        self.nocc
    }
    pub fn gpu(&self) -> &SparseBsr4Gpu {
        &self.gpu
    }
    pub fn m_hs(&self) -> &(Vec<u32>, Vec<u32>) {
        &self.m_hs
    }
    pub fn m_k(&self) -> &(Vec<u32>, Vec<u32>) {
        &self.m_k
    }
    pub fn nblock_k(&self) -> usize {
        self.k_struct.nblock
    }
    pub fn nblock_hs(&self) -> usize {
        self.hs_struct.nblock
    }

    /// Read current K back to host (blocking). Use only for diagnostics.
    pub fn k_to_host(&self) -> Result<Bsr4Matrix> {
        self.k.to_host(&self.gpu)
    }

    /// Current device K matrix (for `SparseDWWorkspace::build_dw_into`).
    pub fn k(&self) -> &GpuBsrMatrix {
        &self.k
    }
    /// Current device Z ≈ S⁻¹ (diagnostics).
    pub fn z(&self) -> &GpuBsrMatrix {
        &self.z
    }
    /// Current device H_scc (for `SparseDWWorkspace::build_dw_into`).
    pub fn h_scc(&self) -> &GpuBsrMatrix {
        &self.h_scc
    }
    /// Current device B = Z·H_scc on M_TZS (for the `W=2(ZH)K` force path).
    /// Fresh from the last `compute_k0_impl`/`compute_p0` — the same H_scc
    /// the forces are taken at.
    pub fn b_zh(&self) -> &GpuBsrMatrix {
        &self.b_zh
    }
    /// Shared K structure (M_K) — for `SparseDWWorkspace::new`.
    pub fn k_struct(&self) -> &Arc<GpuBsrStructure> {
        &self.k_struct
    }
    /// Shared H/S structure (M_HS) — for `SparseDWWorkspace::new`.
    pub fn hs_struct(&self) -> &Arc<GpuBsrStructure> {
        &self.hs_struct
    }
    /// Current device H0 (M_HS) — host diagnostics read back via
    /// `read_f32(&ws.h0().values, ..)` on the GPU pair path.
    pub fn h0(&self) -> &GpuBsrMatrix {
        &self.h0
    }
    /// Current device S (M_HS).
    pub fn s(&self) -> &GpuBsrMatrix {
        &self.s
    }
    /// Shared TZS structure (M_TZS, ZH/ZS products) — for `SparseDWWorkspace::new`.
    pub fn t_zs_struct(&self) -> &Arc<GpuBsrStructure> {
        &self.t_zs_struct
    }
    /// Host map: hs block (i,j) → k block (j,i) or −1 (force contraction).

    /// Build H_scc on the device from atom potentials (R13):
    /// upload `v` (n_atom f32) then one `bsr4_build_Hscc` launch —
    /// `H = H0 + ½S(V_i+V_j)` on physical lanes only. No H_scc host transfer.
    pub fn build_hscc_from_v(&mut self, v: &[f32]) -> Result<()> {
        if v.len() != self.n_atom {
            return Err(DftbError::InvalidInput(format!(
                "build_hscc_from_v: v len {} != n_atom {}",
                v.len(),
                self.n_atom
            )));
        }
        for (i, &x) in v.iter().enumerate() {
            if !x.is_finite() {
                return Err(DftbError::InvalidInput(format!(
                    "build_hscc_from_v: v[{i}]={x}"
                )));
            }
        }
        self.v_buf
            .write(v)
            .enq()
            .map_err(crate::qmqm::gpu_runtime::map_ocl_err)?;
        self.gpu.build_hscc_dev(
            &self.hs_struct,
            &self.h0.values,
            &self.s.values,
            &self.v_buf,
            &self.n_orb_buf,
            &self.h_scc.values,
        )
    }

    /// Upload packed atom data for the γ/γ′ n-body kernels: `xyzu` =
    /// 4·n_atom f32 (x,y,z Å, w = Hubbard u Ha). Once per geometry.
    pub fn gamma_upload_geometry(&mut self, xyzu: &[f32]) -> Result<()> {
        if xyzu.len() != 4 * self.n_atom {
            return Err(DftbError::InvalidInput(format!(
                "gamma_upload_geometry: len {} != 4·n_atom {}",
                xyzu.len(),
                self.n_atom
            )));
        }
        self.gpu.write_f32(&self.xyzu_buf, xyzu)
    }

    /// V = γ·Δq on device into `v_buf`, then read back to `v_out`
    /// (host needs V for E_scc and the force v_shift). `dq`/`v_out` = n_atom.
    pub fn gamma_v(&mut self, dq: &[f32], v_out: &mut [f32]) -> Result<()> {
        if dq.len() != self.n_atom || v_out.len() != self.n_atom {
            return Err(DftbError::InvalidInput(format!(
                "gamma_v: dq {} / v_out {} != n_atom {}",
                dq.len(),
                v_out.len(),
                self.n_atom
            )));
        }
        self.gpu.write_f32(&self.dq_buf, dq)?;
        self.gpu
            .gamma_v_dev(self.n_atom, &self.xyzu_buf, &self.dq_buf, &self.v_buf)?;
        self.gpu.read_f32(&self.v_buf, v_out)
    }

    /// SCC double-counting force on device into `gf_buf`, read back as
    /// 4·n_atom f32 (xyz used). Replaces the CPU `scc_double_counting_force`.
    pub fn gamma_forces(&mut self, dq: &[f32], f_out: &mut [f32]) -> Result<()> {
        if dq.len() != self.n_atom || f_out.len() != 4 * self.n_atom {
            return Err(DftbError::InvalidInput(format!(
                "gamma_forces: dq {} / f_out {} != n_atom {}",
                dq.len(),
                f_out.len(),
                self.n_atom
            )));
        }
        self.gpu.write_f32(&self.dq_buf, dq)?;
        self.gpu
            .gamma_f_dev(self.n_atom, &self.xyzu_buf, &self.dq_buf, &self.gf_buf)?;
        self.gpu.read_f32(&self.gf_buf, f_out)
    }

    /// Same launch as `gamma_forces` without the readback — `gf_buf` is
    /// consumed directly by `force_gather` on the GPU pair path.
    pub fn gamma_forces_dev(&mut self, dq: &[f32]) -> Result<()> {
        if dq.len() != self.n_atom {
            return Err(DftbError::InvalidInput(format!(
                "gamma_forces_dev: dq {} != n_atom {}",
                dq.len(),
                self.n_atom
            )));
        }
        self.gpu.write_f32(&self.dq_buf, dq)?;
        self.gpu
            .gamma_f_dev(self.n_atom, &self.xyzu_buf, &self.dq_buf, &self.gf_buf)
    }

    // ── R17 GPU pair physics ───────────────────────────────────────────

    /// Upload all frozen-topology pair data for the `sparse_hs.cl`
    /// kernels. Called once after construction when the GPU pair path is
    /// active (`SparseDftbConfig::cpu_pair == false`). Under the explicit
    /// CPU reference switch this is never called and `pair` stays `None`.
    pub fn init_pair_data(
        &mut self,
        hs_pairs: &[HsPair],
        rep_pairs: &[(u32, u32)],
        onsite_orb: &[[f64; 4]],
        species_code: &[u8],
        sk: &SkGpuPack,
        rep: &(Vec<i32>, Vec<f32>, usize),
        taper: Option<(f64, f64)>,
    ) -> Result<()> {
        let n = self.n_atom;
        let np = hs_pairs.len();
        let nr = rep_pairs.len();
        let mut pairs = vec![0i32; 4 * np];
        let mut pairs_k = vec![0i32; np];
        for (p, hp) in hs_pairs.iter().enumerate() {
            pairs[4 * p] = hp.i as i32;
            pairs[4 * p + 1] = hp.j as i32;
            pairs[4 * p + 2] = hp.b_ij;
            pairs[4 * p + 3] = hp.b_ji;
            pairs_k[p] = hp.b_ij_k;
        }
        let mut rp = vec![0i32; 2 * nr.max(1)];
        for (p, &(i, j)) in rep_pairs.iter().enumerate() {
            rp[2 * p] = i as i32;
            rp[2 * p + 1] = j as i32;
        }
        let (fp_ptr, fp_list) = pair_gather_adj(hs_pairs.iter().map(|p| (p.i, p.j)), n);
        let (rp_ptr, rp_list) = pair_gather_adj(rep_pairs.iter().copied(), n);
        let onsite: Vec<f32> = onsite_orb
            .iter()
            .flat_map(|o| o.iter().map(|&e| e as f32))
            .collect();
        let species: Vec<u32> = species_code.iter().map(|&s| s as u32).collect();
        let taper_f = match taper {
            Some((r0, w)) => [r0 as f32, w as f32, 1.0, 0.0],
            None => [0.0; 4],
        };
        let gpu = &self.gpu;
        self.pair = Some(GpuPairState {
            n_pairs: np,
            n_rep: nr,
            n_sp: sk.n_species,
            max_ctrl: sk.max_ctrl,
            rep_max_int: rep.2,
            taper: taper_f,
            pairs: gpu.buf_i32(&pairs)?,
            pairs_k: gpu.buf_i32(&pairs_k)?,
            rpairs: gpu.buf_i32(&rp)?,
            fp_ptr: gpu.buf_u32(&fp_ptr)?,
            fp_list: gpu.buf_i32(&fp_list)?,
            rp_ptr: gpu.buf_u32(&rp_ptr)?,
            rp_list: gpu.buf_i32(&rp_list)?,
            species: gpu.buf_u32(&species)?,
            onsite: gpu.buf_f32(&onsite)?,
            sk_meta: gpu.buf_i32(&sk.meta)?,
            sk_parm: gpu.buf_f32(&sk.parm)?,
            sk_ctrl: gpu.buf_f32(&sk.ctrl)?,
            rep_off: gpu.buf_i32(&rep.0)?,
            rep_data: gpu.buf_f32(&rep.1)?,
            pf: gpu.zero_f32(8 * np.max(1))?,
            pf_rep: gpu.zero_f32(4 * nr.max(1))?,
            pe_rep: gpu.zero_f32(nr.max(1))?,
            f_nscc: gpu.zero_f32(4 * n)?,
            f_shift: gpu.zero_f32(4 * n)?,
            f_rep: gpu.zero_f32(4 * n)?,
            f_dc: gpu.zero_f32(4 * n)?,
            f_tot: gpu.zero_f32(4 * n)?,
            kdummy: gpu.zero_f32(n)?,
            row_sums: gpu.zero_f32(4 * n)?,
            sinf_out: gpu.zero_f32(1)?,
            f_host: vec![0.0f32; 4 * n],
            pe_host: vec![0.0f32; nr.max(1)],
            kd_host: vec![0.0f32; n],
        });
        Ok(())
    }

    /// True when the GPU pair path is armed (init_pair_data ran).
    pub fn pair_gpu(&self) -> bool {
        self.pair.is_some()
    }

    /// Device H0/S assembly (R17): zero → `hs_diag` → `hs_assemble`, then
    /// `s_inf` from the device row-abs-sum + max-reduce (persistent
    /// buffers — no alloc). Replaces host `assemble_hs_bsr` + uploads.
    /// Fails loudly when pair data is missing — never a silent CPU path.
    pub fn assemble_hs_dev(&mut self) -> Result<()> {
        if self.pair.is_none() {
            return Err(DftbError::InvalidInput(
                "assemble_hs_dev: GPU pair data not initialized (init_pair_data)".into(),
            ));
        }
        let nb = self.hs_struct.nblock;
        self.gpu.zero_dev(nb, &self.h0.values)?;
        self.gpu.zero_dev(nb, &self.s.values)?;
        {
            let st = self.pair.as_ref().unwrap();
            self.gpu.hs_diag_dev(
                self.n_atom,
                self.hs_struct.diag_block(),
                &self.n_orb_buf,
                &st.onsite,
                &self.h0.values,
                &self.s.values,
            )?;
            self.gpu.hs_assemble_dev(
                st.n_pairs,
                &st.pairs,
                &self.xyzu_buf,
                &st.species,
                &self.n_orb_buf,
                st.n_sp as u32,
                &st.sk_meta,
                &st.sk_parm,
                &st.sk_ctrl,
                st.max_ctrl as u32,
                st.taper,
                &self.h0.values,
                &self.s.values,
            )?;
            self.gpu.inf_norm_into_dev(
                &self.hs_struct,
                &self.s.values,
                &st.row_sums,
                &self.reduce_a,
                &self.reduce_b,
                &st.sinf_out,
            )?;
        }
        let st = self.pair.as_mut().unwrap();
        self.gpu.read_f32(&st.sinf_out, &mut st.kd_host[..1])?;
        let s_inf = st.kd_host[0];
        if !s_inf.is_finite() || s_inf < 1e-30 {
            return Err(DftbError::InvalidInput(format!(
                "assemble_hs_dev: ||S||_inf = {s_inf:e} — degenerate S (coincident atoms?)"
            )));
        }
        self.s_inf = s_inf;
        Ok(())
    }

    /// `rep_eval` on the frozen `rpairs` list → `pf_rep` (forces stay
    /// device-resident for `force_gather`) + host f64 sum → `e_rep`.
    pub fn rep_eval_dev(&mut self) -> Result<f64> {
        let st = self.pair.as_mut().ok_or_else(|| {
            DftbError::InvalidInput("rep_eval_dev: GPU pair data not initialized".into())
        })?;
        self.gpu.rep_eval_dev(
            st.n_rep,
            &st.rpairs,
            &self.xyzu_buf,
            &st.species,
            st.n_sp as u32,
            &st.rep_off,
            st.rep_max_int as u32,
            &st.rep_data,
            &st.pf_rep,
            &st.pe_rep,
        )?;
        self.gpu.read_f32(&st.pe_rep, &mut st.pe_host[..st.n_rep])?;
        Ok(st.pe_host[..st.n_rep].iter().map(|&e| e as f64).sum())
    }

    /// Full pair force path on device: `hs_kdummy` guard → `hs_contract`
    /// → `force_gather` → readback → `Forces`. `k`/`w` are the device
    /// K (M_K) and W (M_HS) value buffers at the current H_scc.
    pub fn pair_forces_dev(
        &mut self,
        k: &Buffer<f32>,
        w: &Buffer<f32>,
    ) -> Result<crate::methods::dftb::forces::Forces> {
        let n = self.n_atom;
        if self.pair.is_none() {
            return Err(DftbError::InvalidInput(
                "pair_forces_dev: GPU pair data not initialized".into(),
            ));
        }
        // OCC-GUARD-DUMMY: same 1e-6 gate as sparse_forces_bsr.
        {
            let st = self.pair.as_mut().unwrap();
            self.gpu.hs_kdummy_dev(
                n,
                self.k_struct.diag_block(),
                k,
                &self.n_orb_buf,
                &st.kdummy,
            )?;
            self.gpu.read_f32(&st.kdummy, &mut st.kd_host)?;
            let occ_dummy: f64 = st.kd_host[..n].iter().map(|&x| (x as f64).abs()).sum();
            if occ_dummy > 1e-6 {
                return Err(DftbError::InvalidInput(format!(
                    "Padded (dummy) orbitals carry occupation Σ|2K|={occ_dummy:.3e} — \
                     BSR4 padding leaks into the physics"
                )));
            }
        }
        {
            let st = self.pair.as_ref().unwrap();
            self.gpu.hs_contract_dev(
                st.n_pairs,
                &st.pairs,
                &st.pairs_k,
                &self.xyzu_buf,
                &st.species,
                &self.n_orb_buf,
                st.n_sp as u32,
                &st.sk_meta,
                &st.sk_parm,
                &st.sk_ctrl,
                st.max_ctrl as u32,
                st.taper,
                k,
                w,
                &self.v_buf,
                &st.pf,
            )?;
            self.gpu.force_gather_dev(
                n,
                &st.fp_ptr,
                &st.fp_list,
                &st.pf,
                &st.rp_ptr,
                &st.rp_list,
                &st.pf_rep,
                &self.gf_buf,
                &st.f_nscc,
                &st.f_shift,
                &st.f_rep,
                &st.f_dc,
                &st.f_tot,
            )?;
        }
        // Readback — 5 component buffers → Forces (f32→f64 on host).
        let mut out = crate::methods::dftb::forces::Forces::zeros(n);
        {
            let st = self.pair.as_mut().unwrap();
            self.gpu.read_f32(&st.f_nscc, &mut st.f_host)?;
            pull3(&st.f_host, &mut out.non_scc);
            self.gpu.read_f32(&st.f_shift, &mut st.f_host)?;
            pull3(&st.f_host, &mut out.scc_shift);
            self.gpu.read_f32(&st.f_rep, &mut st.f_host)?;
            pull3(&st.f_host, &mut out.repulsive);
            self.gpu.read_f32(&st.f_dc, &mut st.f_host)?;
            pull3(&st.f_host, &mut out.scc_dc);
            self.gpu.read_f32(&st.f_tot, &mut st.f_host)?;
            pull3(&st.f_host, &mut out.forces);
        }
        crate::methods::dftb::forces::check_finite(&out.forces, "forces (GPU pair path)");
        crate::methods::dftb::forces::check_finite(&out.non_scc, "non-SCC forces (GPU pair path)");
        crate::methods::dftb::forces::check_finite(
            &out.scc_shift,
            "SCC shift forces (GPU pair path)",
        );
        crate::methods::dftb::forces::check_finite(
            &out.repulsive,
            "repulsive forces (GPU pair path)",
        );
        // f32 gather form: Newton holds to the f32 floor, not exactly —
        // the CPU reference (tol 1e-6) keeps exact symmetry for tests.
        crate::methods::dftb::forces::check_newton(&out.forces, "forces (GPU pair path)", 1e-4);
        crate::methods::dftb::forces::check_newton(
            &out.non_scc,
            "non-SCC forces (GPU pair path)",
            1e-4,
        );
        crate::methods::dftb::forces::check_newton(
            &out.repulsive,
            "repulsive forces (GPU pair path)",
            1e-4,
        );
        Ok(out)
    }

    /// Sparse masked band energy `Tr(K·H0)` on device (R7) — f32 partials
    /// + host-f64 tail (PR3/PR4, §15.9): this is an extensive sum with
    /// cancellation, so the final reduction is done in f64, not collapsed
    /// to one f32 scalar. No `k_to_dense`/`trace_ab`. Caller multiplies by
    /// 2 (spin) and adds E_scc/E_rep.
    pub fn trace_kh0_dev(&mut self) -> Result<f64> {
        self.gpu.trace_hk_to_f64(
            self.hs_struct.nblock,
            &self.h0.values,
            &self.k.values,
            &self.hs_to_kt,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )
    }

    /// L3 (manifest §15.10): band energy WITHOUT K recovery —
    /// `E_band = 2·Tr(P·ZH)` using `b_zh` = Z·H_scc still held from the
    /// P0 build. Restricts ZH onto M_P and takes the masked trace (same
    /// kernel as Tr(K·H0), map = M_P transpose). Diagnostic: if this is
    /// accurate at low degree while the recovered-K path is not, the
    /// K=PZ recovery — not P locality — is what sets the degree floor.
    /// Requires a P purifier run (`p_valid`); Errs otherwise.
    pub fn band_energy_from_p(&mut self) -> Result<f64> {
        if !self.p_valid {
            return Err(DftbError::InvalidInput(
                "band_energy_from_p: no P state — engine ran K-TC2, not purifier p/trs".into(),
            ));
        }
        // ZH restricted M_TZS→M_P into q_p2 (free scratch after purify).
        self.gpu.restrict_dev(
            self.p_struct.nblock,
            &self.p_to_tzs,
            &self.b_zh.values,
            &self.q_p2.values,
        )?;
        let tr = self.gpu.trace_hk_to_f64(
            self.p_struct.nblock,
            &self.p.values,
            &self.q_p2.values,
            &self.p_t,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )?;
        Ok(2.0 * tr)
    }

    /// Hamiltonian stationarity residual (R11):
    ///   A = H_scc·(K·S);  R_H = ‖A − Aᵀ‖_F / (2‖A‖_F + ε)
    /// because (HKS)ᵀ = SKH for symmetric S,K,H. One extra SpGEMM on M_HT
    /// (T not symmetric → intersection kernel), one O(nnz) download, host
    /// f64 norms. Once per SCC finalization — not in the inner loop.
    pub fn rh_stationarity(&mut self) -> Result<f64> {
        if !self.t_ks_valid {
            self.spgemm_ks()?;
            self.t_ks_valid = true;
        }
        // A = H_scc·T on M_HT — generic planned SpGEMM (T non-symmetric).
        // None only under the explicit RUST_DFTB_SPARSE_PLANS=0 diagnostic.
        match &self.plan_ht {
            Some(plan) => self
                .gpu
                .spgemm_plan_dev(&self.h_scc, &self.t_ks, plan, &self.a_ht)?,
            None => self
                .gpu
                .spgemm_masked_dev(&self.h_scc, &self.t_ks, &self.a_ht)?,
        }
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
        // Prefer t_ks = K·S of the CURRENT K: the reported (K,q) state must
        // be mutually consistent or the SCC energy sits off-stationarity
        // (FD-vs-analytic force parity breaks ~5e-4 on SiH4 G3.4 when q
        // came from P while the energy used the recovered K=PZ).
        if self.t_ks_valid {
            self.gpu.mulliken_to_dev(
                &self.t_ks_struct,
                &self.t_ks.values,
                &self.n_orb_buf,
                &self.qpack_buf,
            )?;
        } else if self.p_valid {
            // P-purifier fast path: P IS K·S — Mulliken reads P's diagonal
            // directly (only when no recovered K exists yet).
            self.gpu.mulliken_to_dev(
                &self.p_struct,
                &self.p.values,
                &self.n_orb_buf,
                &self.qpack_buf,
            )?;
        } else {
            self.spgemm_ks()?;
            self.t_ks_valid = true;
            self.gpu.mulliken_to_dev(
                &self.t_ks_struct,
                &self.t_ks.values,
                &self.n_orb_buf,
                &self.qpack_buf,
            )?;
        }
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
                h0.values.len(),
                self.h0.struct_.nblock * BS2
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
                s.values.len(),
                self.s.struct_.nblock * BS2
            )));
        }
        self.s.upload_values(&self.gpu, &s.values)?;
        self.s_inf = inf_norm(s);
        if !self.s_inf.is_finite() || self.s_inf < 1e-30 {
            return Err(DftbError::InvalidInput(format!(
                "upload_s: ||S||_inf={:e} non-finite or near-zero",
                self.s_inf
            )));
        }
        Ok(())
    }

    /// Upload H_scc values (same M_HS as H0).
    pub fn upload_h_scc(&mut self, h: &Bsr4Matrix) -> Result<()> {
        if h.values.len() != self.h_scc.struct_.nblock * BS2 {
            return Err(DftbError::InvalidInput(format!(
                "upload_h_scc: values len {} != expected {}",
                h.values.len(),
                self.h_scc.struct_.nblock * BS2
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
            return Err(DftbError::InvalidInput(format!(
                "upload_s_values: ||S||_inf={s_inf:e}"
            )));
        }
        self.s_inf = s_inf;
        self.s.upload_values(&self.gpu, values)
    }

    // ── SpGEMM helpers (use plans when available) ──

    /// T = K·S using plan or intersection kernel.
    fn spgemm_ks(&mut self) -> Result<()> {
        match &self.plan_ks {
            Some(plan) if self.ktime_on => {
                let mut ev = ocl::Event::empty();
                self.gpu
                    .spgemm_plan_bsym_dev_ev(&self.k, &self.s, plan, &self.t_ks, &mut ev)?;
                self.ev_ks.push(ev);
            }
            Some(plan) => self
                .gpu
                .spgemm_plan_bsym_dev(&self.k, &self.s, plan, &self.t_ks)?,
            None => self.gpu.spgemm_bsym_dev(&self.k, &self.s, &self.t_ks)?,
        }
        Ok(())
    }

    /// Q = T·K (the dominant TC2 product); same plan/fallback policy.
    fn spgemm_ksk(&mut self) -> Result<()> {
        match &self.plan_tk {
            Some(plan) if self.ktime_on => {
                let mut ev = ocl::Event::empty();
                self.gpu
                    .spgemm_plan_bsym_dev_ev(&self.t_ks, &self.k, plan, &self.q, &mut ev)?;
                self.ev_ksk.push(ev);
            }
            Some(plan) => self
                .gpu
                .spgemm_plan_bsym_dev(&self.t_ks, &self.k, plan, &self.q)?,
            None => self.gpu.spgemm_bsym_dev(&self.t_ks, &self.k, &self.q)?,
        }
        Ok(())
    }

    /// Sum kernel-event START→END for the recorded TC2 product events and
    /// report TRUE kernel execution time (not marker-elapsed spans).
    /// Drains both buffers. Env-gated by RUST_DFTB_KTIME.
    pub fn kern_time_report(&mut self) {
        if !self.ktime_on {
            return;
        }
        use ocl::enums::{ProfilingInfo, ProfilingInfoResult};
        let mut drain = |evs: &mut Vec<ocl::Event>, name: &'static str| {
            let n = evs.len();
            let mut ms = 0.0f64;
            for ev in evs.drain(..) {
                let _ = ev.wait_for();
                if let (Ok(ProfilingInfoResult::Start(s)), Ok(ProfilingInfoResult::End(e))) = (
                    ev.profiling_info(ProfilingInfo::Start),
                    ev.profiling_info(ProfilingInfo::End),
                ) {
                    ms += (e - s) as f64 * 1e-6;
                }
            }
            if n > 0 {
                eprintln!(
                    "[ktime] {name}: {n} launches, {ms:.1} ms kernel-exec, {:.3} ms/launch",
                    ms / n as f64
                );
            }
        };
        drain(&mut self.ev_ks, "tc2.ks");
        drain(&mut self.ev_ksk, "tc2.ksk");
    }

    // ── Newton-Schulz inverse (device-resident) ──

    /// Compute Z ≈ S⁻¹ via Newton-Schulz on persistent GPU buffers.
    ///
    /// Products use the precomputed symbolic plans `plan_zs`/`plan_tz`
    /// (mandatory — plan-build failure is a construction error, SC3;
    /// intersection kernel only under the explicit SPARSE_PLANS=0 toggle).
    /// Residual is `||I−T||_F/√N` from device f32 partials + host-f64 tail
    /// (PR3 — `identity_residual_to_f64`), one ≤128-float read per
    /// iteration (the N4 bug was a missing sqrt in the old scalar path,
    /// plus stale off-diagonal Z after a diagonal-only identity write).
    ///
    /// `warm`: if true, keep the persistent Z from the previous geometry and
    /// NS-correct it against the new S (one step gives ≈ Z − Z·δS·Z, the
    /// first-order inverse correction — few iterations for small steps).
    /// If a warm attempt stalls, diverges, or hits non-finite residual, we
    /// restart cold from αI once; cold failure is a loud Err.
    /// No `Buffer::builder` / `Kernel::builder` / matrix transfer in the loop.
    pub fn compute_z(
        &mut self,
        max_iter: usize,
        tol: f32,
        stall: usize,
        warm: bool,
    ) -> Result<(f32, usize)> {
        let n_orb = (self.n_atom * BS) as f32;
        let nblock = self.z_struct.nblock;
        let alpha = 1.0 / self.s_inf;
        let n_attempts = if warm { 2 } else { 1 };
        let mut last_err = String::new();

        for attempt in 0..n_attempts {
            let cold = attempt == n_attempts - 1 || !warm;
            if cold {
                // Full-write identity: every structural entry (R6).
                self.gpu
                    .build_identity_dev(&self.z_struct, &self.z.values)?;
                self.gpu.scale_dev(nblock, alpha, &self.z.values)?;
            }
            let mut prev_rz = f32::INFINITY;
            let mut stall_count = 0;
            let mut rz = f32::INFINITY;
            let mut restart = false;
            self.gpu.prof_tick("ns.z0");
            for iter in 0..max_iter {
                // T = Z·S (planned; S symmetric)
                match &self.plan_zs {
                    Some(plan) => self
                        .gpu
                        .spgemm_plan_bsym_dev(&self.z, &self.s, plan, &self.t_zs)?,
                    None => self.gpu.spgemm_bsym_dev(&self.z, &self.s, &self.t_zs)?,
                }
                self.gpu.prof_tick("ns.zs");
                // ||I−T||² → f32 partials + host-f64 tail (PR3) — the
                // residual drives the NS accept decision; sqrt + normalize
                // in f64 on host.
                let r2 = self.gpu.identity_residual_to_f64(
                    &self.t_zs_struct,
                    &self.t_zs.values,
                    &self.reduce_partial,
                    &self.reduce_a,
                    &self.reduce_b,
                    &mut self.reduce_tail_host,
                )?;
                self.gpu.prof_tick("ns.rz");
                // Contract: R_Z = ||I−ZS||_F / √N_orb(padded). Was erroneously
                // /n_orb — understated the true residual by √N (36× on R10,
                // 59× on R14) so reported 3.7e-5 was really ~1.4e-3.
                let n_dim = (n_orb as f64).sqrt();
                rz = (r2.sqrt() / n_dim) as f32;
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
                        last_err = format!(
                            "compute_z: stalled after {} iters, R_Z={rz:e} (tol={tol:e})",
                            iter + 1
                        );
                        restart = true;
                        break;
                    }
                } else {
                    stall_count = 0;
                }
                prev_rz = rz;
                // Q = T·Z (planned; Z symmetric right operand) → M_Z
                match &self.plan_tz {
                    Some(plan) => self
                        .gpu
                        .spgemm_plan_bsym_dev(&self.t_zs, &self.z, plan, &self.qz)?,
                    None => self.gpu.spgemm_bsym_dev(&self.t_zs, &self.z, &self.qz)?,
                }
                self.gpu.prof_tick("ns.tz");
                self.gpu.axpby_dev(
                    nblock,
                    2.0,
                    &self.z.values,
                    -1.0,
                    &self.qz.values,
                    &self.znew.values,
                )?;
                self.gpu.symmetrize_dev(
                    nblock,
                    &self.z_struct.transpose_block(),
                    &self.znew.values,
                )?;
                std::mem::swap(&mut self.z.values, &mut self.znew.values);
                self.gpu.prof_tick("ns.upd");
            }
            if restart && !cold {
                eprintln!("  compute_z: warm start failed ({last_err}) — restarting cold from αI");
                continue;
            }
            if !restart && rz.is_finite() && rz >= tol {
                last_err = format!(
                    "compute_z: exhausted {max_iter} iters, final R_Z={rz:e} (tol={tol:e})"
                );
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
            Some(plan) => self
                .gpu
                .spgemm_plan_bsym_dev(&self.z, h, plan, &self.b_zh)?,
            None => self.gpu.spgemm_bsym_dev(&self.z, h, &self.b_zh)?,
        }
        // Gershgorin bounds of B — persistent buffers, 2 scalar reads.
        self.gpu.gershgorin_to_dev(
            &self.b_zh.struct_,
            &self.b_zh.values,
            &self.gersh_emin,
            &self.gersh_emax,
            &self.reduce_a,
            &self.reduce_b,
            &self.emin_buf,
            &self.emax_buf,
        )?;
        let mut lo = [0.0f32; 1];
        let mut hi = [0.0f32; 1];
        self.gpu.read_f32(&self.emin_buf, &mut lo)?;
        self.gpu.read_f32(&self.emax_buf, &mut hi)?;
        let (mut emin, mut emax) = (lo[0], hi[0]);
        // DIAGNOSTIC ONLY: RUST_DFTB_EMIN/EMAX override the Gershgorin bounds
        // (e.g. with true eigenvalues) to test whether loose bounds cause the
        // f32 TC2 floor. Never set in production.
        if let (Ok(a), Ok(b)) = (
            std::env::var("RUST_DFTB_EMIN"),
            std::env::var("RUST_DFTB_EMAX"),
        ) {
            emin = a.parse::<f32>().unwrap_or(emin);
            emax = b.parse::<f32>().unwrap_or(emax);
            eprintln!("  [bounds OVERRIDE] emin={emin} emax={emax}");
        }
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
            Some(plan) => self
                .gpu
                .spgemm_plan_bsym_dev(&self.b_zh, &self.z, plan, &self.a_zhz)?,
            None => self.gpu.spgemm_bsym_dev(&self.b_zh, &self.z, &self.a_zhz)?,
        }

        // Z|M_K via restrict map, then K0 = (emax·Z − A)/Δε on M_K.
        self.gpu.restrict_dev(
            self.k_struct.nblock,
            &self.k_to_z,
            &self.z.values,
            &self.z_on_k.values,
        )?;
        let delta = (emax - emin).max(1e-12);
        self.gpu.axpby_dev(
            self.k_struct.nblock,
            emax / delta,
            &self.z_on_k.values,
            -1.0 / delta,
            &self.a_zhz.values,
            &self.k.values,
        )?;
        self.gpu.symmetrize_dev(
            self.k_struct.nblock,
            &self.k_struct.transpose_block(),
            &self.k.values,
        )?;
        self.t_ks_valid = false;
        Ok((emin, emax))
    }

    /// K₀ from H0. Returns (emin, emax) — the padded spectral bounds.
    pub fn compute_k0(&mut self, padding: f32) -> Result<(f32, f32)> {
        let h0 = GpuBsrMatrix {
            struct_: self.h0.struct_.clone(),
            values: self.h0.values.clone(),
        };
        self.compute_k0_impl(&h0, padding)
    }

    /// K₀ from the current **H_scc** (not H0). Call after `build_hscc_dev`.
    pub fn compute_k0_from_hscc(&mut self, padding: f32) -> Result<(f32, f32)> {
        let h = GpuBsrMatrix {
            struct_: self.h_scc.struct_.clone(),
            values: self.h_scc.values.clone(),
        };
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
    /// `Tr(KS)` counts physical lanes only (masked trace, R14) and is
    /// reduced in **f64 on the host** from per-atom partials (GPT-5.6
    /// item 2) — the trace feeds a discrete branch decision where an
    /// f32-comparison flip is catastrophic. Returns (r_I, Tr[f64], iters).
    pub fn tc2_purify(
        &mut self,
        max_iter: usize,
        tol: f32,
        check_every: usize,
    ) -> Result<(PurifyStatus, f32, f64, usize)> {
        // Normalization scale: ‖K‖_F of the incoming K (K0 or previous
        // iterate — drifts little during purification).
        let ksq = self.gpu.frob_sq_to_f64(
            self.k_struct.nblock * BS2,
            &self.k.values,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )?;
        let k_norm = (ksq.sqrt() as f32).max(1e-30);
        self.gpu.prof_tick("tc2.knorm");

        let nocc64 = self.nocc as f64;
        let tol_tr = crate::methods::sparse::gpu_sparse::tc2_trace_tol(nocc64);
        let mut best_r_i = f32::INFINITY;
        let mut best_tr = 0.0f64;
        let mut best_snapshotted = false;
        let mut last_tr = 0.0f64;
        let mut last_r_i = f32::INFINITY;
        let mut n_stagnant = 0usize;
        let mut iter_best = 0usize; // iter of last absolute-best snapshot (restore target)
        let _ = iter_best;
        // §15.12-B replay-validated (hist_{r10_rk12,r10_rk20,r14_rk12,sih4}):
        // stagnation = the trace-gated absolute best improved <5% over the
        // last W=28 checked iters. Shorter windows fire on real mid-descent
        // stalls (R14 has genuine ~20-iter stalls before a 44% breakthrough —
        // live W=8 diverged). W=28: +16% iters saved on deg330, zero quality
        // loss and zero false-fires on all four saved histories.
        let mut best_hist: std::collections::VecDeque<(usize, f32)> =
            std::collections::VecDeque::new();
        let mut trace_locked = false;
        let mut dev_prev = f64::MAX; // previous |Tr−Nocc|/Nocc — growth detector
        let mut dev_run = 0usize; // consecutive exponentially-growing deviations
        self.t_ks_valid = false;
        // §15.12-B replay: per-iter history for offline stopping-rule
        // evaluation. Env-gated; CSV: iter,branch,tr,dev_rel,guard,r_i,
        // best_r_i,snapshotted + '# call/end' markers. Never in hot path
        // unless explicitly enabled.
        let hist_path = std::env::var("RUST_DFTB_TC2_HIST")
            .ok()
            .filter(|p| !p.is_empty());
        if let Some(p) = &hist_path {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
            {
                let _ = writeln!(
                    f,
                    "# call ts_ms={} max_iter={max_iter} tol={tol:e} nocc={nocc64}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0)
                );
            }
        }
        let t_call = std::time::Instant::now(); // for the t_ms column
        let hist_rec = |f: &mut std::fs::File,
                        iter: usize,
                        branch: u32,
                        tr: f64,
                        dev_rel: f64,
                        guard: bool,
                        ri: f32,
                        best: f32,
                        snap: bool| {
            let _ = std::io::Write::write_fmt(
                f,
                format_args!(
                    "{iter},{branch},{tr:.6e},{dev_rel:.3e},{},{ri:.6e},{best:.6e},{},{:.3}\n",
                    guard as u8,
                    snap as u8,
                    t_call.elapsed().as_secs_f64() * 1e3
                ),
            );
        };
        let hist_end = |reason: &str| {
            if let Some(p) = &hist_path {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                {
                    let _ = writeln!(f, "# end {reason}");
                }
            }
        };

        // ── Production purification policy (manifest §4.12.2
        // consolidated, 2026-09-15) — resolved ONCE; no env reads inside
        // the loop. Phase A = f32 TC2 with a hard budget; Phase B =
        // terminal FF32 McWeeny polish (accurate mode only). ──
        let env_flag = |n: &str| {
            std::env::var(n)
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        };
        let env_usz = |n: &str, d: usize| {
            std::env::var(n)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let env_f32 = |n: &str, d: f32| {
            std::env::var(n)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        // Caller's max_iter IS the f32 budget — the production "30" lives
        // in SparseDftbConfig::tc2_max; env only lowers it (study scripts
        // pass big max_iter deliberately to observe the floor).
        let budget = max_iter.min(env_usz("RUST_DFTB_TC2_BUDGET", usize::MAX));
        let ff_on = self.tc2_hiacc || env_flag("RUST_DFTB_TC2_FF");
        let ff_switch = env_f32("RUST_DFTB_TC2_FF_SWITCH", 1e-3);
        let ff_steps = env_usz("RUST_DFTB_TC2_FF_STEPS", 5);
        // Below this the polish is only refining f32-storage quantization
        // (measured fixed point ~1.3e-8 on R10); default a decade above.
        let ff_target = env_f32("RUST_DFTB_TC2_FF_TARGET", 1e-7);
        // 3-product V=T·Q path (ff×ff kernel) instead of 4-product U,V.
        let ff_vtq = env_flag("RUST_DFTB_TC2_FF_VTQ");
        let mcw_end = env_flag("RUST_DFTB_TC2_MCW");
        let stop_w = env_usz("RUST_DFTB_TC2_STOP_W", 0);
        // Floor-stop window in CHECKED iters — only armed below
        // ff_switch, where genuine mid-descent stalls cannot live.
        const FLOOR_W: usize = 5;
        let mut ff_entry: Option<f32> = None; // Some(entry_ri) → Phase B
        let mut n_iter = 0usize;

        for iter in 0..budget {
            n_iter = iter + 1;
            let do_check = (iter % check_every == 0) || (iter == budget - 1);

            // T = K·S, Q = T·K, trace = Tr(T), optionally R_I = ||Q-K||.
            self.spgemm_ks()?;
            self.gpu.prof_tick("tc2.ks");
            // §15.12-3 ONE-READ PATH: enqueue the per-atom trace partials,
            // then Q=T·K and the residual partials SPECULATIVELY — all on
            // the in-order queue, before the single blocking trace read.
            // The branch decision does NOT gate Q (both TC2 branches use
            // it); the only pre-KSK dependency is the trace guard, which
            // fires on ~2-6% of iters — those recompute Q + resid after the
            // rescale (corrected state, identical semantics to the old
            // read-then-compute order).
            self.gpu.trace_atom_dev(
                self.t_ks_struct.n_atom,
                &self.t_ks_struct.diag_block(),
                &self.t_ks.values,
                &self.n_orb_buf,
                &self.trace_atom,
            )?;
            self.spgemm_ksk()?;
            self.gpu.prof_tick("tc2.ksk");
            if do_check {
                self.gpu.idempotency_partial_dev(
                    self.k.struct_.nblock,
                    &self.q.values,
                    &self.k.values,
                    &self.reduce_partial,
                )?;
            }
            // The ONE blocking read — the queue drains T→tr→Q→resid while
            // the host sleeps once instead of twice.
            self.gpu
                .read_f32(&self.trace_atom, &mut self.trace_atom_host)?;
            self.gpu.prof_tick("tc2.tr");
            let tr_now: f64 = self.trace_atom_host.iter().map(|&x| x as f64).sum();
            if !tr_now.is_finite() {
                return Err(DftbError::InvalidInput(format!(
                    "TC2 trace non-finite at iter {iter}: Tr(KS)={tr_now}"
                )));
            }
            let dev_rel = ((tr_now - nocc64) / nocc64.max(1.0)).abs();
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
            if dev_rel < TC2_TRACE_LOCK_REL && last_r_i < TC2_LOCK_RI {
                trace_locked = true;
            }
            // Runaway signature (1648-atom sphere): a leaked eigenvalue
            // λ=1+δ grows as λ²≈1+2δ under the squaring branch — the trace
            // excess DOUBLES each iteration. Fire the guard only on
            // confirmed growth (≥2 consecutive ~1.5×+ expansions), not on
            // healthy endgame jitter: on SiH4 the deviation plateaus ~2e-4
            // and plain TC2 alternation removes it, while a premature
            // rescale injects ~1e-4 state error that broke the G3.4
            // force/FD parity by 6×.
            if trace_locked && dev_rel > TC2_TRACE_GUARD_REL && dev_rel > 1.5 * dev_prev {
                dev_run += 1;
            } else {
                dev_run = 0;
            }
            dev_prev = dev_rel;
            let guard = crate::methods::sparse::gpu_sparse::tc2_guard_enabled()
                && dev_run >= 2
                && tr_now > 0.0;
            // tr_eff = trace of the state that produces Q this iteration.
            // After a rescale Tr(α·KS)=α·Tr is exactly linear — but we
            // RE-MEASURE the rescaled T instead of returning Nocc by
            // assignment (S1: the reported trace must be a measurement of
            // the state that continues, not a construction claim).
            let tr_eff = if guard {
                let alpha = (nocc64 / tr_now) as f32;
                self.gpu
                    .scale_dev(self.k.struct_.nblock, alpha, &self.k.values)?;
                self.gpu
                    .scale_dev(self.t_ks.struct_.nblock, alpha, &self.t_ks.values)?;
                self.guard_fires += 1;
                if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                    eprintln!(
                        "    TC2 trace guard @iter {iter}: Tr={tr_now:.6} (dev_rel={dev_rel:.2e}) → rescaled K by {alpha:.8}"
                    );
                }
                // The speculative Q/resid used the PRE-rescale T — stale.
                // Recompute from the corrected state (~2-6% of iters; keeps
                // the old read-then-compute semantics exactly).
                self.spgemm_ksk()?;
                if do_check {
                    self.gpu.idempotency_partial_dev(
                        self.k.struct_.nblock,
                        &self.q.values,
                        &self.k.values,
                        &self.reduce_partial,
                    )?;
                }
                let tr2: f64 = self.gpu.trace_ks_f64(
                    &self.t_ks_struct,
                    &self.t_ks.values,
                    &self.n_orb_buf,
                    &self.trace_atom,
                    &mut self.trace_atom_host,
                )?;
                if !tr2.is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "TC2 post-rescale trace non-finite at iter {iter}: Tr(KS)={tr2}"
                    )));
                }
                tr2
            } else {
                tr_now
            };
            // Host f64 branch decision. Post-rescale Tr==Nocc is degenerate:
            // choose SQUARING (branch=1) so the next measured trace dips
            // below Nocc and the following iteration takes the complement —
            // preserving the natural TC2 alternation. Taking the complement
            // here instead ratchets: complement pushes all λ<1 up (trace
            // rises) → guard rescales → complement again — observed on SiH4
            // as a limit cycle with the excess doubling each iteration and
            // the R_I floor degrading to 7e-4.
            let branch: u32 = if guard { 1 } else { (tr_now > nocc64) as u32 };
            let mut ri_now = f32::NAN; // filled only on check iters
            if do_check {
                // Residual partials were enqueued behind Q (or recomputed
                // on the guard path); only the reduce+read remains — the
                // queue is already drained so this returns immediately.
                let ri_sq = self.gpu.idempotency_finish_f64(
                    self.k.struct_.nblock,
                    &self.reduce_partial,
                    &self.reduce_a,
                    &self.reduce_b,
                    &mut self.reduce_tail_host,
                )?;
                self.gpu.prof_tick("tc2.res");
                let tr = tr_eff;
                let ri = (ri_sq.sqrt() as f32) / k_norm;
                if !ri.is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "TC2 residual non-finite at iter {iter}: R_I={}",
                        ri
                    )));
                }
                last_tr = tr;
                last_r_i = ri;
                ri_now = ri;
                // JOINT snapshot criterion: min R_I among iterates whose trace
                // is ALSO valid. Selecting on R_I alone once captured an
                // iterate that was already mid-runaway (best R_I at the same
                // iter where Tr had drifted 0.7 e⁻) — the snapshot was then
                // rejected by the trace gate and the run hard-failed.
                if ri < best_r_i && (tr - nocc64).abs() <= tol_tr {
                    best_r_i = ri;
                    best_tr = tr;
                    iter_best = iter;
                    self.gpu
                        .copy_f32(&self.k.values, &self.k_best, self.k.struct_.nblock * BS2)?;
                    best_snapshotted = true;
                    n_stagnant = 0;
                } else {
                    n_stagnant += 1;
                }
                best_hist.push_back((iter, best_r_i));
                if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                    eprintln!("    TC2 iter {iter:3}  R_I={ri:.4e}  Tr(KS)={tr:.6}");
                }

                if ri < tol {
                    if (tr - nocc64).abs() <= tol_tr {
                        if ff_on {
                            // Phase B gate: converged at f32 accuracy —
                            // polish it in the terminal FF phase.
                            ff_entry = Some(ri);
                            break;
                        }
                        // Converged: K was not updated after this T — t_ks
                        // still holds K·S for the returned K (R8a reuse).
                        self.t_ks_valid = true;
                        hist_end("converged");
                        return Ok((PurifyStatus::Converged, ri, tr, iter + 1));
                    }
                    if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                        eprintln!("  TC2 R_I={ri:e} < tol but Tr(KS)={tr} != Nocc={nocc64} (wrong-rank projector not accepted)");
                    }
                }
                // Production floor-stop (manifest §4.12.2 consolidated):
                // trace-locked AND already inside the projector basin
                // (best < ff_switch) AND the trace-gated best improved <5%
                // over the last FLOOR_W checks → this IS the f32/mask
                // floor; continuing is floor-dancing. The ri<ff_switch
                // gate is what makes the short window safe — genuine
                // mid-descent stalls (R14: ~20 iters at R_I~1e-2..1e-1)
                // cannot satisfy it.
                if trace_locked
                    && best_snapshotted
                    && (best_tr - nocc64).abs() <= tol_tr
                    && best_r_i < ff_switch
                    && best_hist.len() > FLOOR_W
                    && best_r_i > best_hist[best_hist.len() - 1 - FLOOR_W].1 * 0.95
                {
                    eprintln!(
                        "  TC2 floor-stop @iter {iter} (best improved <5% over last {FLOOR_W} checks at R_I={best_r_i:e} < switch={ff_switch:e}): restore best K (Tr={best_tr})",
                    );
                    self.gpu.copy_f32(
                        &self.k_best,
                        &mut self.k.values,
                        self.k.struct_.nblock * BS2,
                    )?;
                    self.spgemm_ks()?;
                    self.t_ks_valid = true;
                    if ff_on {
                        ff_entry = Some(best_r_i);
                        break;
                    }
                    hist_end(&format!("floor_stop iter={iter}"));
                    return Ok((PurifyStatus::NumericalFloor, best_r_i, best_tr, iter + 1));
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
                    if crate::methods::sparse::gpu_sparse::tc2_plateau_enabled()
                        && best_snapshotted
                        && (best_tr - nocc64).abs() <= tol_tr
                        && best_r_i < 1e-2
                    {
                        eprintln!(
                            "  TC2 plateau at iter {iter} ({}): restoring best K (R_I={best_r_i:e}, Tr(KS)={best_tr}) — this IS the f32/mask floor",
                            if stagnant { "stagnant" } else { "oscillating" }
                        );
                        self.gpu.copy_f32(
                            &self.k_best,
                            &mut self.k.values,
                            self.k.struct_.nblock * BS2,
                        )?;
                        self.spgemm_ks()?; // T consistent with restored K
                        self.t_ks_valid = true;
                        if ff_on && best_r_i < ff_switch {
                            ff_entry = Some(best_r_i);
                            break;
                        }
                        hist_end(&format!("plateau_restore iter={iter}"));
                        return Ok((PurifyStatus::NumericalFloor, best_r_i, best_tr, iter + 1));
                    }
                    hist_end(&format!("diverged iter={iter}"));
                    return Err(DftbError::InvalidInput(format!(
                        "TC2 diverged at iter {iter}, R_I={ri:e}, best={best_r_i:e}"
                    )));
                }
                // §15.12-B candidate (env-gated; 0/absent = off): stop when
                // the trace-gated best improved <5% over the last W checked
                // iters (endgame: dev_rel < 5e-4, valid snapshot). Same
                // restore path + NumericalFloor status as the 10× detector.
                // (stop_w resolved once in the policy block above.)
                if stop_w > 0 && best_hist.len() > stop_w {
                    let (i_prev, b_prev) = best_hist[best_hist.len() - 1 - stop_w];
                    let _ = i_prev;
                    if best_r_i > b_prev * 0.95
                        && dev_rel < 5.0e-4
                        && best_snapshotted
                        && (best_tr - nocc64).abs() <= tol_tr
                        && best_r_i < 1e-2
                    {
                        eprintln!(
                        "  TC2 stagnation stop @iter {iter} (best improved <5% over last {stop_w} iters: {b_prev:e}→{best_r_i:e}): restore best K (Tr={best_tr})",
                    );
                        self.gpu.copy_f32(
                            &self.k_best,
                            &mut self.k.values,
                            self.k.struct_.nblock * BS2,
                        )?;
                        self.spgemm_ks()?;
                        self.t_ks_valid = true;
                        if ff_on && best_r_i < ff_switch {
                            ff_entry = Some(best_r_i);
                            break;
                        }
                        hist_end(&format!("windowed_best iter={iter}"));
                        return Ok((PurifyStatus::NumericalFloor, best_r_i, best_tr, iter + 1));
                    }
                }
            }

            // Update K: Knew = TC2_branch(K, Q, branch), symmetrize, swap.
            // `branch` is the host f64 decision (GPT-5.6 item 2).
            let nblock = self.k.struct_.nblock;
            // McWeeny endgame (env-gated STUDY path): once trace-locked AND
            // below the TC2 floor zone, switch to the CONTRACTING McWeeny
            // map K ← 3Q − 2·(Q·S·K). Kept for A/B — the production
            // high-accuracy endgame is the terminal FF32 Phase B below.
            let use_mcw = mcw_end && trace_locked && last_r_i < 1e-3;
            if use_mcw {
                // MCW-PLAN (§4.12.2): U = Q·S is the SAME structural product
                // as K·S → plan_ks; V = U·K same as T·K → plan_tk. This
                // routes McWeeny through the ACC/Kahan-capable planned
                // kernels instead of the single-accumulator masked bsym —
                // and writes U on the correct M_TKS structure (the masked
                // path truncated it to M_K).
                match &self.plan_ks {
                    Some(plan) => self
                        .gpu
                        .spgemm_plan_bsym_dev(&self.q, &self.s, plan, &self.t_ks)?,
                    None => self.gpu.spgemm_bsym_dev(&self.q, &self.s, &self.t_ks)?,
                }
                match &self.plan_tk {
                    Some(plan) => {
                        self.gpu
                            .spgemm_plan_bsym_dev(&self.t_ks, &self.k, plan, &self.a_zhz)?
                    }
                    None => self.gpu.spgemm_bsym_dev(&self.t_ks, &self.k, &self.a_zhz)?,
                }
                self.gpu.mcweeny(
                    nblock,
                    &self.q.values,
                    &self.a_zhz.values,
                    &self.knew.values,
                )?;
            } else {
                self.gpu.tc2_dev(
                    nblock,
                    &self.k.values,
                    &self.q.values,
                    branch,
                    &self.knew.values,
                )?;
            }
            self.gpu
                .symmetrize_dev(nblock, &self.k_struct.transpose_block(), &self.knew.values)?;
            std::mem::swap(&mut self.k.values, &mut self.knew.values);
            self.gpu.prof_tick("tc2.upd");
            let branch = if use_mcw { 9 } else { branch }; // hist: 9=McWeeny (FF is Phase B, branch=8 rows written there)
            if let Some(p) = &hist_path {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                {
                    hist_rec(
                        &mut f,
                        iter,
                        branch,
                        tr_eff,
                        dev_rel,
                        guard,
                        ri_now,
                        best_r_i,
                        best_snapshotted,
                    );
                }
            }
        }

        // ── end of Phase A ──
        // Budget exhausted: restore the best valid iterate. Whether that
        // is a usable floor state or a hard failure depends on the
        // projector-basin gate — a budget that ends with an unlocked
        // trace or a large residual is NOT an arithmetic-floor problem.
        if ff_entry.is_none() {
            if crate::methods::sparse::gpu_sparse::tc2_plateau_enabled()
                && best_snapshotted
                && (best_tr - nocc64).abs() <= tol_tr
                && best_r_i < 1e-2
            {
                eprintln!(
                    "  TC2 exhausted {budget} iters — restoring best K at floor (R_I={best_r_i:e} < tol={tol:e} not reached, Tr(KS)={best_tr})"
                );
                self.gpu.copy_f32(
                    &self.k_best,
                    &mut self.k.values,
                    self.k.struct_.nblock * BS2,
                )?;
                self.spgemm_ks()?;
                self.t_ks_valid = true;
                if ff_on && best_r_i < ff_switch {
                    ff_entry = Some(best_r_i);
                } else {
                    hist_end("exhausted_restore");
                    return Ok((PurifyStatus::NumericalFloor, best_r_i, best_tr, budget));
                }
            } else {
                hist_end("exhausted");
                return Err(DftbError::InvalidInput(format!(
                    "TC2 exhausted {budget} iters, final r_I={last_r_i:e} Tr(KS)={last_tr} (rel. tol={tol:e} Nocc={})",
                    self.nocc
                )));
            }
        }

        // ── Phase B — terminal FF32 McWeeny polish ──
        // The loop measures the CURRENT K's residual via its first two
        // products (Q_ff = KSK is needed anyway), so the early exit never
        // spends the U,V products on a state that is already good — and
        // the returned R_I is measured on the RETURNED state.
        let entry_ri = ff_entry.unwrap();
        let nblock = self.k_struct.nblock;
        // k_best := entry state — the Phase-B best tracker is consistent
        // with what is actually stored (Phase A may have left an earlier
        // iterate in k_best).
        self.gpu
            .copy_f32(&self.k.values, &self.k_best, nblock * BS2)?;
        let mut ri = entry_ri;
        let mut prev_ri = f32::INFINITY;
        let mut best_ff = entry_ri;
        let mut nsteps = 0usize;
        loop {
            self.ff_prod_tq()?; // T_ff, Q_ff = products of current K
            self.gpu.idempotency_partial_dev(
                nblock,
                &self.q.values,
                &self.k.values,
                &self.reduce_partial,
            )?;
            let ri_sq = self.gpu.idempotency_finish_f64(
                nblock,
                &self.reduce_partial,
                &self.reduce_a,
                &self.reduce_b,
                &mut self.reduce_tail_host,
            )?;
            let ri_now = (ri_sq.sqrt() as f32) / k_norm;
            if ri_now < best_ff {
                best_ff = ri_now;
                self.gpu
                    .copy_f32(&self.k.values, &self.k_best, nblock * BS2)?;
            }
            let stalled = nsteps > 0 && ri_now > prev_ri * 0.7;
            if let Some(p) = &hist_path {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                {
                    hist_rec(
                        &mut f,
                        n_iter + nsteps,
                        8,
                        f64::NAN,
                        0.0,
                        false,
                        ri_now,
                        best_ff,
                        true,
                    );
                }
            }
            if ri_now < ff_target || stalled || nsteps >= ff_steps {
                ri = ri_now;
                break;
            }
            prev_ri = ri_now;
            self.ff_prod_uv(ff_vtq)?;
            nsteps += 1;
        }
        if best_ff < ri {
            // a later step regressed — return the best
            self.gpu
                .copy_f32(&self.k_best, &mut self.k.values, nblock * BS2)?;
            ri = best_ff;
        }
        self.spgemm_ks()?; // T consistent with returned K
        let tr_ret = self.gpu.trace_ks_f64(
            &self.t_ks_struct,
            &self.t_ks.values,
            &self.n_orb_buf,
            &self.trace_atom,
            &mut self.trace_atom_host,
        )?;
        self.t_ks_valid = true;
        eprintln!(
            "  TC2 Phase-B FF32 polish: {nsteps} McWeeny step(s) after {n_iter} f32 iters — R_I={ri:e} (entry {entry_ri:e}), Tr(KS)={tr_ret}"
        );
        hist_end(&format!("ff_polish steps={nsteps}"));
        Ok((PurifyStatus::PolishedFF, ri, tr_ret, n_iter + nsteps))
    }

    // ── P = K·S purifier (GPT-5.6 item 4) ──

    /// P0 = (emax·I − ZH)/Δ on M_P — the non-orthogonal analogue of K0.
    /// Exact identity: K0·S = (emax·I − ZH)·Z·S/Δ = (emax·I − ZH)/Δ.
    /// Shares the Z·H product + Gershgorin bounds with `compute_k0_impl`.
    fn compute_p0_impl(&mut self, h: &GpuBsrMatrix, padding: f32) -> Result<(f32, f32)> {
        // B = Z·H on M_TZS (planned; H symmetric right operand).
        match &self.plan_zh {
            Some(plan) => self
                .gpu
                .spgemm_plan_bsym_dev(&self.z, h, plan, &self.b_zh)?,
            None => self.gpu.spgemm_bsym_dev(&self.z, h, &self.b_zh)?,
        }
        self.gpu.gershgorin_to_dev(
            &self.b_zh.struct_,
            &self.b_zh.values,
            &self.gersh_emin,
            &self.gersh_emax,
            &self.reduce_a,
            &self.reduce_b,
            &self.emin_buf,
            &self.emax_buf,
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
                "compute_p0: bad ZH Gershgorin emin={emin} emax={emax}"
            )));
        }

        // P0 = (emax·I − B)|_{M_P}/Δ : identity into pnew (scratch),
        // B restricted M_TZS→M_P into q_p2 (scratch), axpby → p.
        self.gpu
            .build_identity_dev(&self.p_struct, &self.pnew.values)?;
        self.gpu.restrict_dev(
            self.p_struct.nblock,
            &self.p_to_tzs,
            &self.b_zh.values,
            &self.q_p2.values,
        )?;
        let delta = (emax - emin).max(1e-12);
        self.gpu.axpby_dev(
            self.p_struct.nblock,
            emax / delta,
            &self.pnew.values,
            -1.0 / delta,
            &self.q_p2.values,
            &self.p.values,
        )?;
        self.p_valid = false;
        self.t_ks_valid = false;
        Ok((emin, emax))
    }

    /// P0 from the current **H_scc** (not H0). Call after `build_hscc_dev`.
    pub fn compute_p0_from_hscc(&mut self, padding: f32) -> Result<(f32, f32)> {
        let h = GpuBsrMatrix {
            struct_: self.h_scc.struct_.clone(),
            values: self.h_scc.values.clone(),
        };
        self.compute_p0_impl(&h, padding)
    }

    /// P=KS TC2 purification (GPT-5.6 item 4, EXPERIMENTAL).
    ///
    /// The iterate is the density-side projector P = K·S:
    ///   P² = (KS)² = KSKS = K·S = P   iff  KSK = K,
    ///   Tr(P) = Tr(KS) = Nocc,  q_A = 2·Tr(P_AA) directly.
    ///
    /// ONE generic planned P² SpGEMM per iteration — unlike K-TC2 there is
    /// no truncated intermediate product feeding a second product; the only
    /// truncation is the iterate's own M_P result-mask (the same class of
    /// approximation as truncating K to M_K). P is NOT symmetric → no
    /// symmetrize step. Same f64-trace/guard/plateau machinery as
    /// `tc2_purify` — the restoring invariant is identical (Tr(P)≈Nocc).
    pub fn tc2_purify_p(
        &mut self,
        max_iter: usize,
        tol: f32,
        check_every: usize,
    ) -> Result<(PurifyStatus, f32, f64, usize)> {
        let p_sq = self.gpu.frob_sq_to_f64(
            self.p_struct.nblock * BS2,
            &self.p.values,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )?;
        let p_norm = (p_sq.sqrt() as f32).max(1e-30);

        let nocc64 = self.nocc as f64;
        let tol_tr = crate::methods::sparse::gpu_sparse::tc2_trace_tol(nocc64);
        let mut best_r_i = f32::INFINITY;
        let mut best_tr = 0.0f64;
        let mut best_snapshotted = false;
        let mut last_tr = 0.0f64;
        let mut last_r_i = f32::INFINITY;
        let mut trace_locked = false;
        let mut dev_prev = f64::MAX;
        let mut dev_run = 0usize;
        self.p_valid = false;
        self.t_ks_valid = false;
        // §study: per-iter history (same CSV schema as tc2_purify).
        let hist_path = std::env::var("RUST_DFTB_TC2_HIST")
            .ok()
            .filter(|p| !p.is_empty());
        if let Some(p) = &hist_path {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
            {
                let _ = writeln!(
                    f,
                    "# call ts_ms={} max_iter={max_iter} tol={tol:e} nocc={nocc64} space=P",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0)
                );
            }
        }
        let t_call = std::time::Instant::now(); // for the t_ms column
        let hist_rec = |f: &mut std::fs::File,
                        iter: usize,
                        branch: u32,
                        tr: f64,
                        dev_rel: f64,
                        guard: bool,
                        ri: f32,
                        best: f32,
                        snap: bool| {
            let _ = std::io::Write::write_fmt(
                f,
                format_args!(
                    "{iter},{branch},{tr:.6e},{dev_rel:.3e},{},{ri:.6e},{best:.6e},{},{:.3}\n",
                    guard as u8,
                    snap as u8,
                    t_call.elapsed().as_secs_f64() * 1e3
                ),
            );
        };
        let hist_end = |reason: &str| {
            if let Some(p) = &hist_path {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                {
                    let _ = writeln!(f, "# end {reason}");
                }
            }
        };

        for iter in 0..max_iter {
            let do_check = (iter % check_every == 0) || (iter == max_iter - 1);

            // Tr(P) = Tr(KS) on the CURRENT iterate — per-atom partials,
            // host f64 (item 2). Measured BEFORE the product so the guard
            // rescale below acts on the state whose residual is evaluated.
            let tr_now: f64 = self.gpu.trace_ks_f64(
                &self.p_struct,
                &self.p.values,
                &self.n_orb_buf,
                &self.trace_atom,
                &mut self.trace_atom_host,
            )?;
            if !tr_now.is_finite() {
                return Err(DftbError::InvalidInput(format!(
                    "P-TC2 trace non-finite at iter {iter}: Tr(P)={tr_now}"
                )));
            }
            let dev_rel = ((tr_now - nocc64) / nocc64.max(1.0)).abs();
            if dev_rel < TC2_TRACE_LOCK_REL && last_r_i < TC2_LOCK_RI {
                trace_locked = true;
            }
            if trace_locked && dev_rel > TC2_TRACE_GUARD_REL && dev_rel > 1.5 * dev_prev {
                dev_run += 1;
            } else {
                dev_run = 0;
            }
            dev_prev = dev_rel;
            let guard = crate::methods::sparse::gpu_sparse::tc2_guard_enabled()
                && dev_run >= 2
                && tr_now > 0.0;
            // S1 stale-product fix: rescale FIRST, then form Q = P² of the
            // rescaled state. The old order measured ||P_old² − α·P_old||
            // and updated with a stale Q whenever the guard fired.
            // tr_eff is RE-MEASURED after the rescale — never Nocc by
            // assignment.
            let tr_eff = if guard {
                let alpha = (nocc64 / tr_now) as f32;
                self.gpu
                    .scale_dev(self.p.struct_.nblock, alpha, &self.p.values)?;
                self.guard_fires += 1;
                if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                    eprintln!(
                        "    P-TC2 trace guard @iter {iter}: Tr={tr_now:.6} (dev_rel={dev_rel:.2e}) → rescaled P by {alpha:.8}"
                    );
                }
                let tr2: f64 = self.gpu.trace_ks_f64(
                    &self.p_struct,
                    &self.p.values,
                    &self.n_orb_buf,
                    &self.trace_atom,
                    &mut self.trace_atom_host,
                )?;
                if !tr2.is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "P-TC2 post-rescale trace non-finite at iter {iter}: Tr(P)={tr2}"
                    )));
                }
                tr2
            } else {
                tr_now
            };
            // Post-rescale branch=1 (squaring) — see tc2_purify for why.
            let branch: u32 = if guard { 1 } else { (tr_now > nocc64) as u32 };

            // Q = P·P on M_P — the ONLY product per iteration, formed from
            // the (possibly rescaled) iterate.
            match &self.plan_pp {
                Some(plan) => self
                    .gpu
                    .spgemm_plan_dev(&self.p, &self.p, plan, &self.q_p2)?,
                None => self.gpu.spgemm_masked_dev(&self.p, &self.p, &self.q_p2)?,
            }

            let mut ri_now = f32::NAN; // filled only on check iters
            if do_check {
                let ri_sq = self.gpu.idempotency_to_f64(
                    self.p_struct.nblock,
                    &self.q_p2.values,
                    &self.p.values,
                    &self.reduce_partial,
                    &self.reduce_a,
                    &self.reduce_b,
                    &mut self.reduce_tail_host,
                )?;
                let tr = tr_eff;
                let ri = (ri_sq.sqrt() as f32) / p_norm;
                if !ri.is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "P-TC2 residual non-finite at iter {iter}: R_I={ri}"
                    )));
                }
                last_tr = tr;
                last_r_i = ri;
                ri_now = ri;
                if ri < best_r_i && (tr - nocc64).abs() <= tol_tr {
                    best_r_i = ri;
                    best_tr = tr;
                    self.gpu
                        .copy_f32(&self.p.values, &self.p_best, self.p_struct.nblock * BS2)?;
                    best_snapshotted = true;
                }
                if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                    eprintln!("    P-TC2 iter {iter:3}  R_I={ri:.4e}  Tr(P)={tr:.6}");
                }

                if ri < tol {
                    if (tr - nocc64).abs() <= tol_tr {
                        self.p_valid = true;
                        hist_end("converged");
                        return Ok((PurifyStatus::Converged, ri, tr, iter + 1));
                    }
                    if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                        eprintln!("  P-TC2 R_I={ri:e} < tol but Tr(P)={tr} != Nocc={nocc64} (wrong-rank projector not accepted)");
                    }
                }
                if ri > best_r_i * 10.0 && best_r_i < f32::INFINITY {
                    if crate::methods::sparse::gpu_sparse::tc2_plateau_enabled()
                        && best_snapshotted
                        && (best_tr - nocc64).abs() <= tol_tr
                        && best_r_i < 1e-2
                    {
                        eprintln!(
                            "  P-TC2 plateau at iter {iter}: restoring best P (R_I={best_r_i:e}, Tr(P)={best_tr}) — f32/mask floor"
                        );
                        self.gpu.copy_f32(
                            &self.p_best,
                            &mut self.p.values,
                            self.p_struct.nblock * BS2,
                        )?;
                        self.p_valid = true;
                        hist_end(&format!("plateau_restore iter={iter}"));
                        return Ok((PurifyStatus::NumericalFloor, best_r_i, best_tr, iter + 1));
                    }
                    hist_end(&format!("diverged iter={iter}"));
                    return Err(DftbError::InvalidInput(format!(
                        "P-TC2 diverged at iter {iter}, R_I={ri:e}, best={best_r_i:e}"
                    )));
                }
            }

            // Update P: Pnew = TC2_branch(P, P², branch). NO symmetrize —
            // P = KS is genuinely non-symmetric.
            let nblock = self.p_struct.nblock;
            // P-McWeeny endgame (§4.12.2, env-gated): once trace-locked and
            // below the sawtooth zone, P' = 3P² − 2P³ — the contracting map.
            // r_p4 = Q·P = P³ via the SAME plan_pp (Q and P share M_P).
            let mcw_end = std::env::var("RUST_DFTB_TC2_MCW")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let use_mcw = mcw_end && trace_locked && last_r_i < 1e-3;
            if use_mcw {
                match &self.plan_pp {
                    Some(plan) => self
                        .gpu
                        .spgemm_plan_dev(&self.q_p2, &self.p, plan, &self.r_p4)?,
                    None => self
                        .gpu
                        .spgemm_masked_dev(&self.q_p2, &self.p, &self.r_p4)?,
                }
                self.gpu.mcweeny(
                    nblock,
                    &self.q_p2.values,
                    &self.r_p4.values,
                    &self.pnew.values,
                )?;
            } else {
                self.gpu.tc2_dev(
                    nblock,
                    &self.p.values,
                    &self.q_p2.values,
                    branch,
                    &self.pnew.values,
                )?;
            }
            std::mem::swap(&mut self.p.values, &mut self.pnew.values);
            let branch = if use_mcw { 9 } else { branch }; // hist: 9 = McWeeny
            if let Some(p) = &hist_path {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                {
                    hist_rec(
                        &mut f,
                        iter,
                        branch,
                        tr_eff,
                        dev_rel,
                        guard,
                        ri_now,
                        best_r_i,
                        best_snapshotted,
                    );
                }
            }
        }

        if crate::methods::sparse::gpu_sparse::tc2_plateau_enabled()
            && best_snapshotted
            && (best_tr - nocc64).abs() <= tol_tr
            && best_r_i < 1e-2
        {
            eprintln!(
                "  P-TC2 exhausted {max_iter} iters — restoring best P at floor (R_I={best_r_i:e} < tol={tol:e} not reached, Tr(P)={best_tr})"
            );
            self.gpu
                .copy_f32(&self.p_best, &mut self.p.values, self.p_struct.nblock * BS2)?;
            self.p_valid = true;
            hist_end("exhausted_restore");
            return Ok((PurifyStatus::NumericalFloor, best_r_i, best_tr, max_iter));
        }
        hist_end("exhausted_fail");
        Err(DftbError::InvalidInput(format!(
            "P-TC2 exhausted {max_iter} iters, final r_I={last_r_i:e} Tr(P)={last_tr} (rel. tol={tol:e} Nocc={})",
            self.nocc
        )))
    }

    /// TRS4 purification on P (GPT-5.6 item 5 — Niklasson et al., trace-
    /// resetting purification). The rigorous restoring mechanism:
    ///
    ///   Q = P²,  R = Q²;   P_new = a·Q + b·R
    ///   a + b = 1            (preserves idempotent fixed points)
    ///   a·Tr(Q) + b·Tr(R) = Nocc   (trace reset EXACTLY each iteration)
    ///
    /// ⇒ a = (Nocc − Tr(R)) / (Tr(Q) − Tr(R)),  b = 1 − a.
    /// Tr(Q) ≥ Tr(R) while eigenvalues ∈ [0,1] (x²≥x⁴), so the solve is
    /// well-conditioned near convergence. Two planned SpGEMMs per iter
    /// (P² and P⁴ — both reuse plan_pp since Q lives on M_P) + three
    /// O(N_atom) diagonal readbacks, all decisions in host f64.
    ///
    /// Fallback (fail-loud, documented): if the solve is degenerate
    /// (Tr(Q)≈Tr(R) → P already idempotent) or `a` leaves the stability
    /// window (−1 ≤ a ≤ 3 for f(x)=ax²+(1−a)x⁴ on [0,1]), the iteration
    /// takes a TC2 branch step instead. No guard needed — the trace is
    /// reset by construction every iteration.
    pub fn trs_purify_p(
        &mut self,
        max_iter: usize,
        tol: f32,
        check_every: usize,
    ) -> Result<(PurifyStatus, f32, f64, usize)> {
        let p_sq = self.gpu.frob_sq_to_f64(
            self.p_struct.nblock * BS2,
            &self.p.values,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )?;
        let p_norm = (p_sq.sqrt() as f32).max(1e-30);

        let nocc64 = self.nocc as f64;
        let tol_tr = crate::methods::sparse::gpu_sparse::tc2_trace_tol(nocc64);
        let mut best_r_i = f32::INFINITY;
        let mut best_tr = 0.0f64;
        let mut best_snapshotted = false;
        let mut last_tr = 0.0f64;
        let mut last_r_i = f32::INFINITY;
        let nblock = self.p_struct.nblock;
        self.p_valid = false;
        self.t_ks_valid = false;
        // Same per-iter history as tc2_purify (study overlay).
        let hist_path = std::env::var("RUST_DFTB_TC2_HIST")
            .ok()
            .filter(|p| !p.is_empty());
        if let Some(p) = &hist_path {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
            {
                let _ = writeln!(
                    f,
                    "# call ts_ms={} max_iter={max_iter} tol={tol:e} nocc={nocc64} space=TRS4",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0)
                );
            }
        }
        let t_call = std::time::Instant::now(); // for the t_ms column
        let hist_rec = |f: &mut std::fs::File,
                        iter: usize,
                        branch: u32,
                        tr: f64,
                        dev_rel: f64,
                        guard: bool,
                        ri: f32,
                        best: f32,
                        snap: bool| {
            let _ = std::io::Write::write_fmt(
                f,
                format_args!(
                    "{iter},{branch},{tr:.6e},{dev_rel:.3e},{},{ri:.6e},{best:.6e},{},{:.3}\n",
                    guard as u8,
                    snap as u8,
                    t_call.elapsed().as_secs_f64() * 1e3
                ),
            );
        };
        let hist_end = |reason: &str| {
            if let Some(p) = &hist_path {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                {
                    let _ = writeln!(f, "# end {reason}");
                }
            }
        };

        for iter in 0..max_iter {
            let do_check = (iter % check_every == 0) || (iter == max_iter - 1);

            // Q = P², R = Q² = P⁴ — same M_P structure, same plan.
            match &self.plan_pp {
                Some(plan) => {
                    self.gpu
                        .spgemm_plan_dev(&self.p, &self.p, plan, &self.q_p2)?;
                    self.gpu
                        .spgemm_plan_dev(&self.q_p2, &self.q_p2, plan, &self.r_p4)?;
                }
                None => {
                    self.gpu.spgemm_masked_dev(&self.p, &self.p, &self.q_p2)?;
                    self.gpu
                        .spgemm_masked_dev(&self.q_p2, &self.q_p2, &self.r_p4)?;
                }
            }
            // Three scalar decisions in host f64 (item 2 machinery).
            let tr_p: f64 = self.gpu.trace_ks_f64(
                &self.p_struct,
                &self.p.values,
                &self.n_orb_buf,
                &self.trace_atom,
                &mut self.trace_atom_host,
            )?;
            let tr_q: f64 = self.gpu.trace_ks_f64(
                &self.q_p2.struct_,
                &self.q_p2.values,
                &self.n_orb_buf,
                &self.trace_atom,
                &mut self.trace_atom_host,
            )?;
            let tr_r: f64 = self.gpu.trace_ks_f64(
                &self.r_p4.struct_,
                &self.r_p4.values,
                &self.n_orb_buf,
                &self.trace_atom,
                &mut self.trace_atom_host,
            )?;
            if !tr_p.is_finite() || !tr_q.is_finite() || !tr_r.is_finite() {
                return Err(DftbError::InvalidInput(format!(
                    "TRS4 trace non-finite at iter {iter}: Tr(P)={tr_p} Tr(Q)={tr_q} Tr(R)={tr_r}"
                )));
            }

            // Trace-resetting degree-4 update: P_new = a·Q + (1−a)·R with
            // a = (Nocc − Tr R)/(Tr Q − Tr R) so Tr(P_new) = Nocc exactly.
            // a∈[1,2] is the provably-monotone band (a=2 ⇔ 2x²−x⁴);
            // a∈(2,3] overshoots slightly (max f = a²/4(a−1) ≤ 9/8 ≈ 1.13)
            // but is REQUIRED to raise the trace when the spectrum sits
            // mid-range (Tr(Q) ≪ Nocc — clamping to a≤2 stalls at a
            // wrong-rank fixed point, measured at r_k=12). The overshoot
            // self-corrects: eigenvalues pushed above 1 make the NEXT
            // den = Tr(Q)−Tr(R) ≤ 0, which selects the complement map
            // 2x−x² — it pulls x>1 back below 1 (a squaring step there
            // would amplify them — measured divergence, first prototype).
            let den = tr_q - tr_r;
            enum TrsUpd {
                Coef(f64),
                Complement,
            }
            let upd = if den > 1e-9 {
                TrsUpd::Coef(((nocc64 - tr_r) / den).clamp(-1.0, 3.0))
            } else {
                TrsUpd::Complement
            };

            let mut ri_now = f32::NAN; // filled only on check iters
            if do_check {
                let ri_sq = self.gpu.idempotency_to_f64(
                    nblock,
                    &self.q_p2.values,
                    &self.p.values,
                    &self.reduce_partial,
                    &self.reduce_a,
                    &self.reduce_b,
                    &mut self.reduce_tail_host,
                )?;
                let ri = (ri_sq.sqrt() as f32) / p_norm;
                if !ri.is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "TRS4 residual non-finite at iter {iter}: R_I={ri}"
                    )));
                }
                // Trace AFTER the update = Nocc by construction (TRS) or
                // the branch result (fallback) — report tr_p honestly.
                let tr = tr_p;
                last_tr = tr;
                last_r_i = ri;
                ri_now = ri;
                if ri < best_r_i && (tr - nocc64).abs() <= tol_tr {
                    best_r_i = ri;
                    best_tr = tr;
                    self.gpu
                        .copy_f32(&self.p.values, &self.p_best, nblock * BS2)?;
                    best_snapshotted = true;
                }
                if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                    let a_s = match upd {
                        TrsUpd::Coef(a) => format!("{a:.4}"),
                        TrsUpd::Complement => "compl".to_string(),
                    };
                    eprintln!(
                        "    TRS4 iter {iter:3}  R_I={ri:.4e}  Tr(P)={tr_p:.6}  Tr(Q)={tr_q:.6}  a={a_s}"
                    );
                }

                if ri < tol {
                    if (tr - nocc64).abs() <= tol_tr {
                        self.p_valid = true;
                        hist_end("converged");
                        return Ok((PurifyStatus::Converged, ri, tr, iter + 1));
                    }
                    if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                        eprintln!("  TRS4 R_I={ri:e} < tol but Tr(P)={tr} != Nocc={nocc64} (wrong-rank projector not accepted)");
                    }
                }
                if ri > best_r_i * 10.0 && best_r_i < f32::INFINITY {
                    if crate::methods::sparse::gpu_sparse::tc2_plateau_enabled()
                        && best_snapshotted
                        && (best_tr - nocc64).abs() <= tol_tr
                        && best_r_i < 1e-2
                    {
                        eprintln!(
                            "  TRS4 plateau at iter {iter}: restoring best P (R_I={best_r_i:e}, Tr(P)={best_tr}) — f32/mask floor"
                        );
                        self.gpu
                            .copy_f32(&self.p_best, &mut self.p.values, nblock * BS2)?;
                        self.p_valid = true;
                        hist_end(&format!("plateau_restore iter={iter}"));
                        return Ok((PurifyStatus::NumericalFloor, best_r_i, best_tr, iter + 1));
                    }
                    hist_end(&format!("diverged iter={iter}"));
                    return Err(DftbError::InvalidInput(format!(
                        "TRS4 diverged at iter {iter}, R_I={ri:e}, best={best_r_i:e}"
                    )));
                }
            }

            // Update P: a·Q + (1−a)·R, or the complement map 2P−Q when
            // the denominator is degenerate/contaminated (eigs > 1).
            match upd {
                TrsUpd::Coef(a) => {
                    self.gpu.axpby_dev(
                        nblock,
                        a as f32,
                        &self.q_p2.values,
                        (1.0 - a) as f32,
                        &self.r_p4.values,
                        &self.pnew.values,
                    )?;
                }
                TrsUpd::Complement => {
                    if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                        eprintln!("    TRS4 iter {iter}: den={den:.4} ≤ 0 — complement step 2P−Q (eigs>1 cleanup)");
                    }
                    self.gpu.axpby_dev(
                        nblock,
                        2.0,
                        &self.p.values,
                        -1.0,
                        &self.q_p2.values,
                        &self.pnew.values,
                    )?;
                }
            }
            std::mem::swap(&mut self.p.values, &mut self.pnew.values);
            if let Some(p) = &hist_path {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                {
                    let dev_rel = ((tr_p - nocc64) / nocc64.max(1.0)).abs();
                    hist_rec(
                        &mut f,
                        iter,
                        7,
                        tr_p,
                        dev_rel,
                        false,
                        ri_now,
                        best_r_i,
                        best_snapshotted,
                    );
                }
            }
        }

        if crate::methods::sparse::gpu_sparse::tc2_plateau_enabled()
            && best_snapshotted
            && (best_tr - nocc64).abs() <= tol_tr
            && best_r_i < 1e-2
        {
            eprintln!(
                "  TRS4 exhausted {max_iter} iters — restoring best P at floor (R_I={best_r_i:e} < tol={tol:e} not reached, Tr(P)={best_tr})"
            );
            self.gpu
                .copy_f32(&self.p_best, &mut self.p.values, nblock * BS2)?;
            self.p_valid = true;
            hist_end("exhausted_restore");
            return Ok((PurifyStatus::NumericalFloor, best_r_i, best_tr, max_iter));
        }
        hist_end("exhausted_fail");
        Err(DftbError::InvalidInput(format!(
            "TRS4 exhausted {max_iter} iters, final r_I={last_r_i:e} Tr(P)={last_tr} (rel. tol={tol:e} Nocc={})",
            self.nocc
        )))
    }

    /// P0 + TRS4 on the current device H_scc; K = P·Z recovered after.
    pub fn purify_hscc_trs(
        &mut self,
        tc2_max: usize,
        tc2_tol: f32,
    ) -> Result<(PurifyStatus, f32, f64, usize)> {
        let (emin, emax) = self.compute_p0_from_hscc(0.1)?;
        self.gpu.prof_tick("scc.k0");
        if crate::methods::sparse::gpu_sparse::algebra_verbose() {
            eprintln!("  bounds (ZH Gershgorin, H_scc): emin={emin:.4} emax={emax:.4}  [TRS4]");
        }
        let out = self.trs_purify_p(tc2_max, tc2_tol, 1)?;
        self.gpu.prof_tick("scc.tc2");
        self.recover_k_from_p()?; // leaves fresh t_ks (t_ks_valid)
        self.p_valid = true;
        let (r_i_k, tr_ks) = self.recovered_k_diagnostics()?;
        Ok((out.0, r_i_k, tr_ks, out.3))
    }

    /// Recover K = P·Z on M_K (P non-symmetric left, Z symmetric right →
    /// the Bsym plan applies). K is symmetrized once — P·Z is symmetric to
    /// within the P- and Z-residuals. Needed by the energy/force path.
    ///
    /// Charge-conservation restore: K=P·Z inherits the ZS−I residual, so
    /// Tr(KS) drifts off Nocc by O(R_Z) — at 864 atoms / wide masks that
    /// was 6× the SCC trace gate (measured: Tr=1304.835 vs Nocc=1305).
    /// Same invariant enforcement as the TC2 trace guard: build T=K·S once,
    /// rescale K ← K·(Nocc/Tr) and T ← T·(Nocc/Tr) (T is linear in K).
    /// Leaves `t_ks` FRESH and valid — callers must not rebuild it.
    fn recover_k_from_p(&mut self) -> Result<()> {
        match &self.plan_pz {
            Some(plan) => self
                .gpu
                .spgemm_plan_bsym_dev(&self.p, &self.z, plan, &self.k)?,
            None => self.gpu.spgemm_bsym_dev(&self.p, &self.z, &self.k)?,
        }
        self.gpu.symmetrize_dev(
            self.k_struct.nblock,
            &self.k_struct.transpose_block(),
            &self.k.values,
        )?;
        self.spgemm_ks()?;
        let tr = self.gpu.trace_ks_f64(
            &self.t_ks_struct,
            &self.t_ks.values,
            &self.n_orb_buf,
            &self.trace_atom,
            &mut self.trace_atom_host,
        )?;
        if !tr.is_finite() || tr.abs() < 1e-30 {
            return Err(DftbError::InvalidInput(format!(
                "recover_k_from_p: Tr(KS)={tr} non-finite/zero — cannot restore charge"
            )));
        }
        let scale = (self.nocc as f64 / tr) as f32;
        if !scale.is_finite() {
            return Err(DftbError::InvalidInput(format!(
                "recover_k_from_p: scale=Nocc/Tr={scale} non-finite (Tr={tr})"
            )));
        }
        if (scale - 1.0).abs() > 1e-7 {
            if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                eprintln!("  K-recovery trace restore: Tr(KS)={tr:.6} → rescale K,T by {scale:.8}");
            }
            self.gpu
                .scale_dev(self.k_struct.nblock, scale, &self.k.values)?;
            self.gpu
                .scale_dev(self.t_ks_struct.nblock, scale, &self.t_ks.values)?;
        }
        self.t_ks_valid = true;
        Ok(())
    }

    /// P0 + P-TC2 of the current device H_scc using the current Z. No NS.
    /// K = P·Z is recovered afterwards so the downstream energy/force
    /// path is unchanged. Returns (status, r_I, Tr[KS] as f64, iters) —
    /// r_I/Tr are measured on the RECOVERED K (S1: the reported state is
    /// the state that feeds charges and the energy, not the P iterate).
    pub fn purify_hscc_p(
        &mut self,
        tc2_max: usize,
        tc2_tol: f32,
    ) -> Result<(PurifyStatus, f32, f64, usize)> {
        let (emin, emax) = self.compute_p0_from_hscc(0.1)?;
        self.gpu.prof_tick("scc.k0");
        if crate::methods::sparse::gpu_sparse::algebra_verbose() {
            eprintln!("  bounds (ZH Gershgorin, H_scc): emin={emin:.4} emax={emax:.4}");
        }
        let out = self.tc2_purify_p(tc2_max, tc2_tol, 1)?;
        self.gpu.prof_tick("scc.tc2");
        // Consistent finalization: the reported (K,q) state must come from
        // the SAME matrix or the SCC energy is evaluated off-stationarity —
        // q from P while K=PZ leaves an O(ZS−I) inconsistency that leaks
        // into FD-vs-analytic force parity (measured ~5e-4 on SiH4 G3.4).
        // recover_k_from_p builds T=K·S once (needed for the trace restore)
        // and leaves it fresh — Mulliken then reads t_ks.
        self.recover_k_from_p()?; // leaves fresh t_ks (t_ks_valid)
        self.p_valid = true;
        let (r_i_k, tr_ks) = self.recovered_k_diagnostics()?;
        Ok((out.0, r_i_k, tr_ks, out.3))
    }

    /// Diagnostics of the recovered K state (S1): returns
    /// (‖KSK−K‖_F/‖K‖_F, Tr(KS)) measured on `t_ks` = K·S of the CURRENT K
    /// — i.e. the same matrix that produces Mulliken charges and the band
    /// energy, not the P iterate whose own residual/trace the P-purifier
    /// reports internally. Requires `t_ks_valid` (fresh `spgemm_ks`).
    /// Costs one T·K product + three scalar reductions per call — once per
    /// SCC iteration, after purification.
    fn recovered_k_diagnostics(&mut self) -> Result<(f32, f64)> {
        debug_assert!(self.t_ks_valid, "recovered_k_diagnostics: t_ks stale");
        let tr: f64 = self.gpu.trace_ks_f64(
            &self.t_ks_struct,
            &self.t_ks.values,
            &self.n_orb_buf,
            &self.trace_atom,
            &mut self.trace_atom_host,
        )?;
        let ksq = self.gpu.frob_sq_to_f64(
            self.k_struct.nblock * BS2,
            &self.k.values,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )?;
        // Q = (K·S)·K = KSK on M_K — reuses the tc2_purify Q buffer/plan.
        match &self.plan_tk {
            Some(plan) => self
                .gpu
                .spgemm_plan_bsym_dev(&self.t_ks, &self.k, plan, &self.q)?,
            None => self.gpu.spgemm_bsym_dev(&self.t_ks, &self.k, &self.q)?,
        }
        let ri_sq = self.gpu.idempotency_to_f64(
            self.k_struct.nblock,
            &self.q.values,
            &self.k.values,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )?;
        if !tr.is_finite() || !ri_sq.is_finite() || !ksq.is_finite() {
            return Err(DftbError::InvalidInput(format!(
                "recovered-K diagnostics non-finite: Tr(KS)={tr} ||KSK−K||²={ri_sq} ||K||²={ksq}"
            )));
        }
        let r_i = (ri_sq.sqrt() / ksq.sqrt().max(1e-30)) as f32;
        if crate::methods::sparse::gpu_sparse::algebra_verbose() {
            eprintln!("  recovered-K: R_I(K)={r_i:.4e}  Tr(KS)={tr:.6}  (state feeding q/E)");
        }
        Ok((r_i, tr))
    }

    /// K0 + TC2 of the current device H_scc using the current Z. No NS.
    /// Returns (status, r_I, Tr[KS] as f64, tc2_iters) — see `PurifyStatus`.
    pub fn purify_hscc(
        &mut self,
        tc2_max: usize,
        tc2_tol: f32,
    ) -> Result<(PurifyStatus, f32, f64, usize)> {
        let (emin, emax) = self.compute_k0_from_hscc(0.1)?;
        self.gpu.prof_tick("scc.k0");
        if crate::methods::sparse::gpu_sparse::algebra_verbose() {
            eprintln!("  bounds (ZH Gershgorin, H_scc): emin={emin:.4} emax={emax:.4}");
        }
        let r = self.tc2_purify(tc2_max, tc2_tol, 1);
        self.gpu.prof_tick("scc.tc2");
        r
    }

    /// Warm-start TC2 (env `RUST_DFTB_WARM_K`): keep the stored K — the
    /// converged projector of the previous mix iter / previous ±h
    /// geometry — and tighten it for the new H_scc. Skips the K0 build
    /// (B=Z·H Gershgorin + ZHZ + axpby) EXCEPT that B=Z·H_scc must be
    /// refreshed: `forces()` consumes `b_zh` for the W build. For a
    /// 0.02 Å FD displacement K changes by ~1e-3 — TC2 should converge
    /// in ~O(5) iters instead of ~50. NO cold fallback in the caller:
    /// a warm-seed Err propagates (silent re-solve hid wrong-subspace
    /// states — measured R_H=1.4e-2, E_tot off 0.13 Ha).
    pub fn purify_hscc_warm(
        &mut self,
        tc2_max: usize,
        tc2_tol: f32,
    ) -> Result<(PurifyStatus, f32, f64, usize)> {
        self.refresh_b_zh()?;
        self.gpu.prof_tick("scc.k0");
        let r = self.tc2_purify(tc2_max, tc2_tol, 1);
        self.gpu.prof_tick("scc.tc2");
        r
    }

    /// Refresh `b_zh = Z·H_scc` on M_TZS — one SpGEMM (plan_zh).
    /// Mandatory after `build_hscc_from_v` whenever the caller skips the
    /// K0 build: `forces()`'s W=2(ZH)K path and `dmm_descend`'s X=(ZH)K
    /// both consume b_zh, so a stale buffer silently injects the
    /// PREVIOUS geometry's ZH (frozen-mode bug, GPT-5.6 chat §1).
    pub fn refresh_b_zh(&mut self) -> Result<()> {
        match &self.plan_zh {
            Some(plan) => self
                .gpu
                .spgemm_plan_bsym_dev(&self.z, &self.h_scc, plan, &self.b_zh)?,
            None => self.gpu.spgemm_bsym_dev(&self.z, &self.h_scc, &self.b_zh)?,
        }
        Ok(())
    }

    /// First-order metric transport (GPT-5.6 tier-1 seed): the overlap
    /// changed S₀→S₁ under the displacement; the projector's first-order
    /// correction is  K ← 2K − K·S₁·K  (= K − K·δS·K + O(δS²)). Two
    /// planned products: T=K·S (plan_ks), Q=T·K (plan_tk); K ← 2K − Q.
    pub fn metric_transport(&mut self) -> Result<()> {
        if self.plan_ks.is_none() || self.plan_tk.is_none() {
            return Err(DftbError::InvalidInput(
                "metric_transport: plan_ks/plan_tk missing".into(),
            ));
        }
        self.gpu.spgemm_plan_bsym_dev(
            &self.k,
            &self.s,
            self.plan_ks.as_ref().unwrap(),
            &self.t_ks,
        )?;
        self.gpu.spgemm_plan_bsym_dev(
            &self.t_ks,
            &self.k,
            self.plan_tk.as_ref().unwrap(),
            &self.q,
        )?;
        let nb = self.k_struct.nblock;
        self.gpu.axpby_dev(
            nb,
            2.0,
            &self.k.values,
            -1.0,
            &self.q.values,
            &self.k.values,
        )?;
        self.t_ks_valid = false;
        Ok(())
    }

    /// DIAGNOSTIC (frozen-input experiments): upload host K values on the
    /// existing M_K structure; invalidate cached T=K·S so the next purifier
    /// call re-forms it.
    pub fn inject_k_values(&mut self, vals: &[f32]) -> Result<()> {
        self.k.upload_values(&self.gpu, vals)?;
        self.t_ks_valid = false;
        Ok(())
    }

    /// Invalidate the cached T=K·S after an external K/Z upload
    /// (electronic-state restore).
    pub fn invalidate_ks(&mut self) {
        self.t_ks_valid = false;
    }

    /// δK0 warm seed (Phase G3): `k` must currently hold K0_new =
    /// (emax·Z−ZHZ)/Δ for the displaced geometry (call
    /// `compute_k0_from_hscc` first — it also refreshes `b_zh` for
    /// forces). This transforms k in place:
    ///   K_seed = K_conv_center + (K0_new − K0_center)
    /// The shift is the first-order occupied-subspace rotation that a
    /// bare TC2-on-old-K cannot express (polynomials can't rotate
    /// eigenvectors — RUST_DFTB_WARM_K refuted it). TC2 then polishes
    /// from inside the attracting basin.
    pub fn k_seed_shift(&mut self, k0_center: &[f32], k_conv: &[f32]) -> Result<()> {
        self.a_zhz.upload_values(&self.gpu, k0_center)?;
        self.z_on_k.upload_values(&self.gpu, k_conv)?;
        let nb = self.k_struct.nblock;
        // elementwise — in-place out==x is safe
        self.gpu.axpby_dev(
            nb,
            1.0,
            &self.k.values,
            -1.0,
            &self.a_zhz.values,
            &self.k.values,
        )?;
        self.gpu.axpby_dev(
            nb,
            1.0,
            &self.k.values,
            1.0,
            &self.z_on_k.values,
            &self.k.values,
        )?;
        self.gpu
            .symmetrize_dev(nb, &self.k_struct.transpose_block(), &self.k.values)?;
        self.t_ks_valid = false;
        Ok(())
    }

    /// McWeeny polish (Phase G3): `K ← 3KSK − 2KSKSK`, n_iters times.
    /// Unlike TC2 the McWeeny map is *contracting* toward idempotency
    /// with no branch discontinuity — the right polish after the δK0
    /// warm seed, which TC2 repels (masked-map saddle). 3 SpGEMMs per
    /// iter; trace is NOT conserved (drift is fixed by the TC2
    /// floor-walk that follows).
    pub fn mcweeny_polish(&mut self, n_iters: usize) -> Result<()> {
        for _ in 0..n_iters {
            self.spgemm_ks()?; // t_ks = K·S
            self.spgemm_ksk()?; // q = K·S·K
            self.gpu.spgemm_bsym_dev(&self.q, &self.s, &self.z_on_k)?; // u = Q·S
            self.gpu
                .spgemm_bsym_dev(&self.z_on_k, &self.k, &self.a_zhz)?; // v = U·K = KSKSK
            self.gpu.mcweeny(
                self.k_struct.nblock,
                &self.q.values,
                &self.a_zhz.values,
                &self.k.values,
            )?;
        }
        self.gpu.symmetrize_dev(
            self.k_struct.nblock,
            &self.k_struct.transpose_block(),
            &self.k.values,
        )?;
        self.t_ks_valid = false;
        Ok(())
    }

    /// Fully-planned McWeeny (DMM retraction path): K ← 3KSK − 2KSKSK,
    /// **3 products**: T=K·S (plan_ks), Q=T·K (plan_tk), V=T·Q (plan_tk —
    /// V=(KS)(KSK)=KSKSK, same identity as the FF VTQ path). Buffers:
    /// t_ks=T, q=Q, a_zhz=V. Clobbers t_ks/q/a_zhz — callers holding
    /// A=a_zhz must rebuild.
    pub fn mcweeny_polish_planned(&mut self, n_iters: usize) -> Result<()> {
        if self.plan_ks.is_none() || self.plan_tk.is_none() {
            return Err(DftbError::InvalidInput(
                "mcweeny_polish_planned: plan_ks/plan_tk missing".into(),
            ));
        }
        for _ in 0..n_iters {
            self.gpu.spgemm_plan_bsym_dev(
                &self.k,
                &self.s,
                self.plan_ks.as_ref().unwrap(),
                &self.t_ks,
            )?; // T = K·S
            self.gpu.spgemm_plan_bsym_dev(
                &self.t_ks,
                &self.k,
                self.plan_tk.as_ref().unwrap(),
                &self.q,
            )?; // Q = T·K = KSK
            self.gpu.spgemm_plan_bsym_dev(
                &self.t_ks,
                &self.q,
                self.plan_tk.as_ref().unwrap(),
                &self.a_zhz,
            )?; // V = T·Q = KSKSK
            self.gpu.mcweeny(
                self.k_struct.nblock,
                &self.q.values,
                &self.a_zhz.values,
                &self.k.values,
            )?;
        }
        self.gpu.symmetrize_dev(
            self.k_struct.nblock,
            &self.k_struct.transpose_block(),
            &self.k.values,
        )?;
        self.t_ks_valid = false;
        Ok(())
    }

    /// Measure-only (r_I, Tr(KS)) of the current K — gate inputs for
    /// the post-DMM no-TC2 path (the TC2 map itself degrades R_H, so
    /// skipping it still needs the honest residuals). Two planned
    /// products (T=K·S, Q=T·K) + two reductions; K untouched.
    pub fn measure_projector_state(&mut self) -> Result<(f32, f64)> {
        if self.plan_ks.is_none() || self.plan_tk.is_none() {
            return Err(DftbError::InvalidInput(
                "measure_projector_state: plan_ks/plan_tk missing".into(),
            ));
        }
        self.gpu.spgemm_plan_bsym_dev(
            &self.k,
            &self.s,
            self.plan_ks.as_ref().unwrap(),
            &self.t_ks,
        )?;
        self.t_ks_valid = true; // t_ks = K·S for the CURRENT K — rh_stationarity reuses it
        self.gpu.spgemm_plan_bsym_dev(
            &self.t_ks,
            &self.k,
            self.plan_tk.as_ref().unwrap(),
            &self.q,
        )?;
        let tr = self.gpu.trace_ks_f64(
            &self.t_ks_struct,
            &self.t_ks.values,
            &self.n_orb_buf,
            &self.trace_atom,
            &mut self.trace_atom_host,
        )?;
        self.gpu.idempotency_partial_dev(
            self.k_struct.nblock,
            &self.q.values,
            &self.k.values,
            &self.reduce_partial,
        )?;
        let ri_sq = self.gpu.idempotency_finish_f64(
            self.k_struct.nblock,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )?;
        let ksq = self.gpu.frob_sq_to_f64(
            self.k_struct.nblock * BS2,
            &self.k.values,
            &self.reduce_partial,
            &self.reduce_a,
            &self.reduce_b,
            &mut self.reduce_tail_host,
        )?;
        Ok(((ri_sq.sqrt() as f32) / (ksq.sqrt() as f32).max(1e-30), tr))
    }

    /// DMM/LNV commutator energy-descent (Phase G3 — the missing
    /// subspace-rotation mechanism), CORRECTED generalized-overlap form.
    /// `z` here is S⁻¹ (Newton Z←2Z−ZSZ), NOT S^{-1/2} — measured:
    /// ‖ZSZ−I‖=‖Z²S−I‖=‖S⁻¹−I‖=18.5 on R10. The projector is P = K·S
    /// and F = Z·H = S⁻¹H is the generalized-eigenvalue matrix
    /// (b_zh — still held from compute_k0). E = 2Tr(F·P); the tangent
    /// gradient is the double commutator
    ///   δP = −η·[P,[P,F]] = −η(PF + FP − 2PFP)
    /// and S-selfadjointness of P,F (SP, SF symmetric) makes [P,F]
    /// S-antisymmetric → pure-imaginary spectrum → δE = −2η·Σμ² ≤ 0
    /// (verified dense-f64; the earlier (S⁻¹−K)HK form used Z²=S^{-1/2}²
    /// wrongly and was an ASCENT direction: Tr(H·G)=−15.4).
    /// Back in K (δK = δP·Z, using PZ = KS·S⁻¹ = K and FZ = ZHZ = A):
    ///   δK = −η·(T·A + F·K − 2·T·W2),  T=K·S, W2 = F·K.
    /// GPT-5.6 collapsed form (3 SpGEMMs): since SZ=I (Z=S⁻¹),
    ///   T·A = KHZ = (ZHK)ᵀ = Xᵀ — free from symmetrize — and
    ///   T·W2 = KHK = Y, so  G = X + Xᵀ − 2Y  with X = (ZH)·K,
    ///   Y = (KS)·X. The old 4-product form computed Xᵀ via a whole
    ///   extra product (T·A) plus an A=ZHZ rebuild after every
    ///   retraction — ~30% wasted work.
    /// Per step: T=KS (plan_ks), X=F·K (plan_fk), Y=T·X (plan_tk_g —
    /// X is asymmetric → generic plan, not bsym), then
    /// knew = 2·sym(X) − 2·Y. A=ZHZ no longer needed at all.
    /// Vanishes exactly at a stationary projector and descends E —
    /// the H-information polynomial purifiers lack. η = eta_scale/Δε.
    /// `retract_every`: every N steps one McWeeny retraction pulls K
    /// back onto the manifold.
    pub fn dmm_descend(
        &mut self,
        emin: f32,
        emax: f32,
        n_steps: usize,
        eta_scale: f32,
        retract_every: usize,
    ) -> Result<()> {
        if self.plan_ks.is_none() || self.plan_fk.is_none() || self.plan_tk_g.is_none() {
            return Err(DftbError::InvalidInput(
                "dmm_descend: plan_ks/plan_fk/plan_tk_g missing (RUST_DFTB_SPARSE_PLANS=0?)".into(),
            ));
        }
        let nb = self.k_struct.nblock;
        let delta = (emax - emin).max(1e-6);
        let step = -eta_scale / delta;
        let verbose = crate::methods::sparse::gpu_sparse::algebra_verbose();
        for istep in 0..n_steps {
            // T = K·S on M_TKS
            self.gpu.spgemm_plan_bsym_dev(
                &self.k,
                &self.s,
                self.plan_ks.as_ref().unwrap(),
                &self.t_ks,
            )?;
            // X = F·K = (Z·H)·K on M_K → z_on_k
            self.gpu.spgemm_plan_bsym_dev(
                &self.b_zh,
                &self.k,
                self.plan_fk.as_ref().unwrap(),
                &self.z_on_k,
            )?;
            // Y = T·X = KS·F·K on M_K → q — GENERIC plan: X is
            // asymmetric, the bsym kernel would compute T·Xᵀ.
            self.gpu.spgemm_plan_dev(
                &self.t_ks,
                &self.z_on_k,
                self.plan_tk_g.as_ref().unwrap(),
                &self.q,
            )?;
            if istep == 0
                && std::env::var("RUST_DFTB_DMM_VERIFY")
                    .map(|v| v == "1")
                    .unwrap_or(false)
            {
                self.dmm_verify_host()?; // one-shot dense f64 cross-check of X/Y
            }
            // knew = X + Xᵀ − 2Y = [P,[P,F]]·Z
            self.gpu.axpby_dev(
                nb,
                1.0,
                &self.z_on_k.values,
                0.0,
                &self.knew.values,
                &self.knew.values,
            )?; // knew = X
            self.gpu
                .symmetrize_dev(nb, &self.k_struct.transpose_block(), &self.knew.values)?; // knew = (X+Xᵀ)/2
            self.gpu.axpby_dev(
                nb,
                2.0,
                &self.knew.values,
                -2.0,
                &self.q.values,
                &self.knew.values,
            )?; // knew = X+Xᵀ−2Y
            self.gpu.axpby_dev(
                nb,
                1.0,
                &self.k.values,
                step,
                &self.knew.values,
                &self.k.values,
            )?;
            // Y=KHK is symmetric analytically — the asymmetry in Y is
            // pure truncation noise; without this K drifts asymmetric
            // and R_H inflates (measured: 1.9e-4→6.4e-4 over 4 steps,
            // recovered to 2.3e-4 by the final symmetrize).
            self.gpu
                .symmetrize_dev(nb, &self.k_struct.transpose_block(), &self.k.values)?;
            if verbose {
                let g2 = self.gpu.frob_sq_to_f64(
                    nb * BS2,
                    &self.knew.values,
                    &self.reduce_partial,
                    &self.reduce_a,
                    &self.reduce_b,
                    &mut self.reduce_tail_host,
                )?;
                let e_band = 2.0
                    * self.gpu.trace_hk_to_f64(
                        self.hs_struct.nblock,
                        &self.h_scc.values,
                        &self.k.values,
                        &self.hs_to_kt,
                        &self.reduce_partial,
                        &self.reduce_a,
                        &self.reduce_b,
                        &mut self.reduce_tail_host,
                    )?;
                eprintln!(
                    "    [dmm] step {istep}: |G|_F={:.3e}  E_band={e_band:.6}",
                    g2.sqrt()
                );
            }
            // Retract on schedule INCLUDING the last step — post-DMM
            // polish is gone, so the terminal state is what forces see
            // (measured: skipping it leaves r_I~2e-3 → dummy lanes
            // occupied → loud force-gate failure).
            if retract_every > 0 && (istep + 1) % retract_every == 0 {
                self.mcweeny_polish_planned(1)?;
                if verbose {
                    let rh = self.rh_stationarity()?;
                    eprintln!("    [dmm] retract after step {istep}: R_H={rh:.3e}");
                }
            }
        }
        self.t_ks_valid = false;
        self.gpu.prof_tick("dmm");
        Ok(())
    }

    /// Snapshot central P₀ = K·S (M_TKS) and X₀ = (Z·H)·K (M_K) for the
    /// linear1 tier — call with the converged central K on device and
    /// b_zh = Z·H_scc fresh at the central geometry. Two products, once
    /// per Hessian.
    pub fn snapshot_p0_x0(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.plan_ks.is_none() || self.plan_fk.is_none() {
            return Err(DftbError::InvalidInput(
                "snapshot_p0_x0: plan_ks/plan_fk missing".into(),
            ));
        }
        self.gpu.spgemm_plan_bsym_dev(
            &self.k,
            &self.s,
            self.plan_ks.as_ref().unwrap(),
            &self.t_ks,
        )?;
        let mut p0 = vec![0.0f32; self.t_ks_struct.nblock * BS2];
        self.gpu.read_f32(&self.t_ks.values, &mut p0)?;
        self.gpu.spgemm_plan_bsym_dev(
            &self.b_zh,
            &self.k,
            self.plan_fk.as_ref().unwrap(),
            &self.z_on_k,
        )?;
        let mut x0 = vec![0.0f32; self.k_struct.nblock * BS2];
        self.gpu.read_f32(&self.z_on_k.values, &mut x0)?;
        Ok((p0, x0))
    }

    /// GPT-5.6 "linear1": ONE perturbative H-response step on the
    /// CENTRAL metric — 3 products/eval (the 4th is the force W built
    /// by forces()). Caller has restored K=K₀, kept Z=Z₀ and refreshed
    /// b_zh = Z₀·H₁.
    ///   X = B₁·K₀ − X₀ = (δB)·K₀     (plan_fk, X₀ uploaded to `knew`)
    ///   Y = P₀·X    (P₀ = K₀S₀ central, uploaded to `t_ks`; plan_tk_g —
    ///              X is asymmetric, bsym would compute P₀·Xᵀ)
    ///   K ← K₀ − η·(X + Xᵀ − 2Y)
    /// Working on δB directly avoids the X+Xᵀ≈2Y cancellation of the
    /// full nonlinear gradient at a nearly-stationary point. No NS, no
    /// retraction, no residual gates — calibrated production tier.
    pub fn linear_response(&mut self, eta: f32, x0: &[f32], p0: &[f32]) -> Result<()> {
        if self.plan_fk.is_none() || self.plan_tk_g.is_none() {
            return Err(DftbError::InvalidInput(
                "linear_response: plan_fk/plan_tk_g missing (RUST_DFTB_SPARSE_PLANS=0?)".into(),
            ));
        }
        let nb = self.k_struct.nblock;
        self.gpu.spgemm_plan_bsym_dev(
            &self.b_zh,
            &self.k,
            self.plan_fk.as_ref().unwrap(),
            &self.z_on_k,
        )?; // X₁ = B₁·K₀
        self.knew.upload_values(&self.gpu, x0)?;
        self.gpu.axpby_dev(
            nb,
            1.0,
            &self.z_on_k.values,
            -1.0,
            &self.knew.values,
            &self.z_on_k.values,
        )?; // X = X₁ − X₀
        self.t_ks.upload_values(&self.gpu, p0)?;
        self.gpu.spgemm_plan_dev(
            &self.t_ks,
            &self.z_on_k,
            self.plan_tk_g.as_ref().unwrap(),
            &self.q,
        )?; // Y = P₀·X
        self.gpu.axpby_dev(
            nb,
            1.0,
            &self.z_on_k.values,
            0.0,
            &self.knew.values,
            &self.knew.values,
        )?; // knew = X
        self.gpu
            .symmetrize_dev(nb, &self.k_struct.transpose_block(), &self.knew.values)?; // (X+Xᵀ)/2
        self.gpu.axpby_dev(
            nb,
            2.0,
            &self.knew.values,
            -2.0,
            &self.q.values,
            &self.knew.values,
        )?; // G = X+Xᵀ−2Y
        self.gpu.axpby_dev(
            nb,
            1.0,
            &self.k.values,
            -eta,
            &self.knew.values,
            &self.k.values,
        )?; // K ← K₀ − ηG
        self.gpu
            .symmetrize_dev(nb, &self.k_struct.transpose_block(), &self.k.values)?;
        self.t_ks_valid = false;
        Ok(())
    }

    /// One-shot dense f64 cross-check of the DMM products (debug only,
    /// RUST_DFTB_DMM_VERIFY=1, called while z_on_k=X=(ZH)K, q=Y=(KS)X,
    /// t_ks=T=KS — before the X+Xᵀ−2Y combine). Expands K,S,H,Z,F to
    /// dense f64, rebuilds each product masked exactly like the device
    /// plans, and reports per-product deviations + the descent sign
    /// Tr(H·G). ~O(n³) once at step 0.
    fn dmm_verify_host(&mut self) -> Result<()> {
        let na = self.n_atom;
        let n = na * BS;
        let mut kd = vec![0.0f32; n * n];
        let mut sd = vec![0.0f32; n * n];
        let mut hd = vec![0.0f32; n * n];
        let mut zd = vec![0.0f32; n * n];
        let mut fd = vec![0.0f32; n * n]; // F = Z·H (b_zh on M_TZS)
        let mut td = vec![0.0f32; n * n]; // T = K·S (t_ks on M_TKS)
        let mut xd = vec![0.0f32; n * n]; // X = F·K (z_on_k on M_K)
        let mut yd = vec![0.0f32; n * n]; // Y = T·X (q on M_K)
        let mut vk = vec![0.0f32; self.k_struct.nblock * BS2];
        let mut vs = vec![0.0f32; self.hs_struct.nblock * BS2];
        let mut vh = vec![0.0f32; self.hs_struct.nblock * BS2];
        let mut vz = vec![0.0f32; self.z_struct.nblock * BS2];
        let mut vf = vec![0.0f32; self.t_zs_struct.nblock * BS2];
        let mut vt = vec![0.0f32; self.t_ks_struct.nblock * BS2];
        let mut vx = vec![0.0f32; self.k_struct.nblock * BS2];
        let mut vy = vec![0.0f32; self.k_struct.nblock * BS2];
        self.gpu.read_f32(&self.k.values, &mut vk)?;
        self.gpu.read_f32(&self.s.values, &mut vs)?;
        self.gpu.read_f32(&self.h_scc.values, &mut vh)?;
        self.gpu.read_f32(&self.z.values, &mut vz)?;
        self.gpu.read_f32(&self.b_zh.values, &mut vf)?;
        self.gpu.read_f32(&self.t_ks.values, &mut vt)?;
        self.gpu.read_f32(&self.z_on_k.values, &mut vx)?;
        self.gpu.read_f32(&self.q.values, &mut vy)?;
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            na,
            &self.m_k.0,
            &self.m_k.1,
            &vk,
            &mut kd,
        );
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            na,
            &self.m_hs.0,
            &self.m_hs.1,
            &vs,
            &mut sd,
        );
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            na,
            &self.m_hs.0,
            &self.m_hs.1,
            &vh,
            &mut hd,
        );
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            na,
            &self.m_z.0,
            &self.m_z.1,
            &vz,
            &mut zd,
        );
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            na,
            &self.m_t_zs.0,
            &self.m_t_zs.1,
            &vf,
            &mut fd,
        );
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            na,
            &self.m_t_ks.0,
            &self.m_t_ks.1,
            &vt,
            &mut td,
        );
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            na,
            &self.m_k.0,
            &self.m_k.1,
            &vx,
            &mut xd,
        );
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            na,
            &self.m_k.0,
            &self.m_k.1,
            &vy,
            &mut yd,
        );
        let f64 = |m: &[f32]| -> Vec<f64> { m.iter().map(|&x| x as f64).collect() };
        let (k64, s64, h64, z64, f64_) = (f64(&kd), f64(&sd), f64(&hd), f64(&zd), f64(&fd));
        let (tdev, xdev, ydev) = (f64(&td), f64(&xd), f64(&yd));
        let mm = |a: &[f64], b: &[f64]| -> Vec<f64> {
            let mut c = vec![0.0f64; n * n];
            for i in 0..n {
                for l in 0..n {
                    let ail = a[i * n + l];
                    if ail != 0.0 {
                        for j in 0..n {
                            c[i * n + j] += ail * b[l * n + j];
                        }
                    }
                }
            }
            c
        };
        // Physical-lane mask: padded BSR4 lanes (dummy orbitals) must not
        // contaminate the checks.
        let mut phys = vec![false; n];
        for a in 0..na {
            for l in 0..self.n_orb_host[a] as usize {
                phys[a * BS + l] = true;
            }
        }
        // symmetry of downloaded K (pad lanes ignored)
        let mut k_asym = 0.0f64;
        for i in 0..n {
            if !phys[i] {
                continue;
            }
            for j in 0..n {
                if !phys[j] {
                    continue;
                }
                let d = k64[i * n + j] - k64[j * n + i];
                k_asym += d * d;
            }
        }
        // Z identity check: Z should be S⁻¹ → ‖Z·S − I‖ ≈ 0
        let zs = mm(&z64, &s64);
        let mut e_zs = 0.0;
        for i in 0..n {
            if !phys[i] {
                continue;
            }
            for j in 0..n {
                if !phys[j] {
                    continue;
                }
                let d = zs[i * n + j] - if i == j { 1.0 } else { 0.0 };
                e_zs += d * d;
            }
        }
        // Mask application helper: zero dense entries outside a CSR mask.
        let apply_mask = |m: &[f64], mask: &(Vec<u32>, Vec<u32>)| -> Vec<f64> {
            let mut out = vec![0.0f64; n * n];
            for i in 0..na {
                for p in (mask.0[i] as usize)..(mask.0[i + 1] as usize) {
                    let j = mask.1[p] as usize;
                    for r in 0..BS {
                        for c in 0..BS {
                            out[(i * BS + r) * n + j * BS + c] = m[(i * BS + r) * n + j * BS + c];
                        }
                    }
                }
            }
            out
        };
        let fro_p = |m: &[f64]| -> f64 {
            let mut s = 0.0;
            for i in 0..n {
                if !phys[i] {
                    continue;
                }
                for j in 0..n {
                    if !phys[j] {
                        continue;
                    }
                    s += m[i * n + j] * m[i * n + j];
                }
            }
            s.sqrt()
        };
        let diff = |a: &[f64], b: &[f64]| -> f64 {
            let mut s = 0.0;
            for i in 0..n {
                if !phys[i] {
                    continue;
                }
                for j in 0..n {
                    if !phys[j] {
                        continue;
                    }
                    let d = a[i * n + j] - b[i * n + j];
                    s += d * d;
                }
            }
            s.sqrt()
        };
        // Per-product host references (masked exactly like the device):
        let ks_host = apply_mask(&mm(&k64, &s64), &self.m_t_ks);
        let x_host = apply_mask(&mm(&f64_, &k64), &self.m_k);
        let y_host = apply_mask(&mm(&ks_host, &x_host), &self.m_k);
        eprintln!("    [dmm-verify] per-product ‖dev−host_f64masked‖:  T=KS {:.4e}  X=(ZH)K {:.4e}  Y=(KS)X {:.4e}",
            diff(&tdev, &ks_host), diff(&xdev, &x_host), diff(&ydev, &y_host));
        // Full combined G = X + Xᵀ − 2Y, host masked chain:
        let g_host = {
            let mut g = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    g[i * n + j] = x_host[i * n + j] + x_host[j * n + i] - 2.0 * y_host[i * n + j];
                }
            }
            g
        };
        let g_dev = {
            let mut g = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    g[i * n + j] = xdev[i * n + j] + xdev[j * n + i] - 2.0 * ydev[i * n + j];
                }
            }
            g
        };
        let tr_hg = |g: &[f64]| -> f64 {
            let hg = mm(&h64, g);
            (0..n).filter(|&i| phys[i]).map(|i| hg[i * n + i]).sum()
        };
        eprintln!("    [dmm-verify] ‖ZS−I‖_phys={:.3e}  ‖G_host‖={:.4e}  ‖G_dev−G_host‖={:.4e}  Tr(H·G_host)={:+.6e}  Tr(H·G_dev)={:+.6e}   (Tr must be ≥0 for descent)",
            e_zs.sqrt(), fro_p(&g_host), diff(&g_dev, &g_host), tr_hg(&g_host), tr_hg(&g_dev));
        Ok(())
    }

    /// FF32-POLISH (manifest §4.12.2 r2): ONE float-float McWeeny step
    ///   K' = 3KSK − 2KSKSK
    /// with ALL intermediates carried as (hi,lo) float-float — f32 FMA
    /// only, ~46-bit effective arithmetic, no native fp64. Four planned
    /// products on the existing plans, V fused into the final combine
    /// (never stored). Decisive-test expectation: R_I ~1e-5 → ~1e-8.
    /// Hi parts reuse `t_ks.values`/`q.values`; lo parts are the
    /// persistent `ff_t_lo`/`ff_q_lo`. Requires `plan_ks`/`plan_tk`
    /// (plans are mandatory in production — SC3).
    /// FF32-POLISH step, first half: T_ff = K·S then Q_ff = T_ff·K.
    /// Leaves Q_ff = KSK of the CURRENT K in (q, ff_q_lo) — the residual
    /// ‖Q−K‖ is therefore measurable here before the expensive U,V
    /// products are spent (Phase-B early exit).
    fn ff_prod_tq(&mut self) -> Result<()> {
        let plan_ks = self.plan_ks.as_ref().ok_or_else(|| {
            DftbError::InvalidInput(
                "ff_prod_tq: plan_ks missing (RUST_DFTB_SPARSE_PLANS=0?)".into(),
            )
        })?;
        let plan_tk = self
            .plan_tk
            .as_ref()
            .ok_or_else(|| DftbError::InvalidInput("ff_prod_tq: plan_tk missing".into()))?;
        // T_ff = K32·S32            (f32×f32 → ff; dedicated no-lo kernel)
        self.gpu
            .spgemm_plan_bsym_ff0_dev(&self.k, &self.s, plan_ks, &self.t_ks, &self.ff_t_lo)?;
        self.gpu.prof_tick("ff.ks");
        // Q_ff = T_ff·K32           (ff×f32 → ff)
        self.gpu.spgemm_plan_bsym_ff_dev(
            &self.t_ks,
            &self.ff_t_lo,
            true,
            &self.k,
            plan_tk,
            &self.q,
            &self.ff_q_lo,
        )?;
        self.gpu.prof_tick("ff.tk");
        Ok(())
    }

    /// FF32-POLISH step, second half + update. Two product paths:
    ///   4-product (default):  U_ff = Q_ff·S, V_ff = U_ff·K
    ///   3-product (FF_VTQ=1): V_ff = T_ff·Q_ff — V=KSKSK is symmetric and
    ///     T·Q = KS·KSK = V exactly; needs the ff×ff kernel so Q's lo part
    ///     is not dropped. plan_tk serves verbatim (same structures).
    /// Requires `ff_prod_tq` to have left T_ff in (t_ks, ff_t_lo) — note
    /// the 4-product path OVERWRITES t_ks with U_ff (same structure).
    fn ff_prod_uv(&mut self, vtq: bool) -> Result<()> {
        let plan_ks = self
            .plan_ks
            .as_ref()
            .ok_or_else(|| DftbError::InvalidInput("ff_prod_uv: plan_ks missing".into()))?;
        let plan_tk = self
            .plan_tk
            .as_ref()
            .ok_or_else(|| DftbError::InvalidInput("ff_prod_uv: plan_tk missing".into()))?;
        if vtq {
            // V_ff = T_ff·Q_ff (ff×ff → ff) — 3 products total
            self.gpu.spgemm_plan_bsym_ffb_dev(
                &self.t_ks,
                &self.ff_t_lo,
                &self.q,
                &self.ff_q_lo,
                plan_tk,
                &self.a_zhz,
                &self.ff_v_lo,
            )?;
            self.gpu.prof_tick("ff.uk");
        } else {
            // U_ff = Q_ff·S32 → reuse T buffers (ff×f32 → ff)
            self.gpu.spgemm_plan_bsym_ff_dev(
                &self.q,
                &self.ff_q_lo,
                true,
                &self.s,
                plan_ks,
                &self.t_ks,
                &self.ff_t_lo,
            )?;
            self.gpu.prof_tick("ff.qs");
            // V_ff = U_ff·K32 → a_zhz (hi) + ff_v_lo — via the VERIFIED
            // product kernel (the fused product+combine variant lost its
            // compensation under compiler scheduling: 1.7e-7 vs 9.7e-14)
            self.gpu.spgemm_plan_bsym_ff_dev(
                &self.t_ks,
                &self.ff_t_lo,
                true,
                &self.k,
                plan_tk,
                &self.a_zhz,
                &self.ff_v_lo,
            )?;
            self.gpu.prof_tick("ff.uk");
        }
        // K' = round_f32(3Q_ff − 2V_ff) — elementwise combine
        self.gpu.mcw_ff_combine_dev(
            self.k_struct.nblock,
            &self.q.values,
            &self.ff_q_lo,
            &self.a_zhz.values,
            &self.ff_v_lo,
            &self.knew.values,
        )?;
        self.gpu.symmetrize_dev(
            self.k_struct.nblock,
            &self.k_struct.transpose_block(),
            &self.knew.values,
        )?;
        self.gpu.prof_tick("ff.comb");
        std::mem::swap(&mut self.k.values, &mut self.knew.values);
        self.t_ks_valid = false;
        Ok(())
    }

    pub fn mcweeny_ff_step(&mut self) -> Result<()> {
        let vtq = std::env::var("RUST_DFTB_TC2_FF_VTQ")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        self.ff_prod_tq()?;
        self.ff_prod_uv(vtq)
    }

    /// FF32-POLISH unit check: run T_ff = K·S then Q_ff = T_ff·K via the
    /// ff plan kernels, comparing (hi+lo) against host-f64 dense products
    /// on each output mask. Returns (err1_hi, err1_ff, err2_hi, err2_ff).
    /// Diagnostic only.
    pub fn ff_test_ks(&mut self) -> Result<(f64, f64, f64, f64)> {
        let plan_ks = self
            .plan_ks
            .as_ref()
            .ok_or_else(|| DftbError::InvalidInput("ff_test_ks: no plan_ks".into()))?;
        let plan_tk = self
            .plan_tk
            .as_ref()
            .ok_or_else(|| DftbError::InvalidInput("ff_test_ks: no plan_tk".into()))?;
        // product 1: T_ff = K32·S32  (f32×f32)
        self.gpu.spgemm_plan_bsym_ff_dev(
            &self.k,
            &self.ff_t_lo,
            false,
            &self.s,
            plan_ks,
            &self.t_ks,
            &self.ff_t_lo,
        )?;
        // snapshot T_ff before product 3 reuses the buffers
        let nt = self.t_ks_struct.nblock * BS2;
        let mut t_hi_snap = vec![0.0f32; nt];
        let mut t_lo_snap = vec![0.0f32; nt];
        self.gpu.read_f32(&self.t_ks.values, &mut t_hi_snap)?;
        self.gpu.read_f32(&self.ff_t_lo, &mut t_lo_snap)?;
        // product 2: Q_ff = T_ff·K32 (ff×f32)
        self.gpu.spgemm_plan_bsym_ff_dev(
            &self.t_ks,
            &self.ff_t_lo,
            true,
            &self.k,
            plan_tk,
            &self.q,
            &self.ff_q_lo,
        )?;
        // A/B: same product with a_has_lo=0 — if outputs are identical,
        // the A_lo path is dead in the kernel.
        let nq = self.k_struct.nblock * BS2;
        let mut q_hi_a = vec![0.0f32; nq];
        let mut q_lo_a = vec![0.0f32; nq];
        self.gpu.read_f32(&self.q.values, &mut q_hi_a)?;
        self.gpu.read_f32(&self.ff_q_lo, &mut q_lo_a)?;
        self.gpu.spgemm_plan_bsym_ff_dev(
            &self.t_ks,
            &self.ff_t_lo,
            false,
            &self.k,
            plan_tk,
            &self.q,
            &self.ff_q_lo,
        )?;
        let mut q_hi_b = vec![0.0f32; nq];
        let mut q_lo_b = vec![0.0f32; nq];
        self.gpu.read_f32(&self.q.values, &mut q_hi_b)?;
        self.gpu.read_f32(&self.ff_q_lo, &mut q_lo_b)?;
        let d_hi: f64 = q_hi_a
            .iter()
            .zip(&q_hi_b)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let d_lo: f64 = q_lo_a
            .iter()
            .zip(&q_lo_b)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        eprintln!("[ff_test] A_lo on/off diff: ‖ΔQ_hi‖={d_hi:.3e}  ‖ΔQ_lo‖={d_lo:.3e}");
        // restore the a_has_lo=1 result
        self.gpu.spgemm_plan_bsym_ff_dev(
            &self.t_ks,
            &self.ff_t_lo,
            true,
            &self.k,
            plan_tk,
            &self.q,
            &self.ff_q_lo,
        )?;
        // product 3: U_ff = Q_ff·S (ff×f32 → ff on M_TKS)
        self.gpu.spgemm_plan_bsym_ff_dev(
            &self.q,
            &self.ff_q_lo,
            true,
            &self.s,
            plan_ks,
            &self.t_ks,
            &self.ff_t_lo,
        )?;
        // product 4 (production split path): V_ff = U_ff·K on M_K, then
        // elementwise combine K' = round_f32(3Q_ff − 2V_ff).
        self.gpu.spgemm_plan_bsym_ff_dev(
            &self.t_ks,
            &self.ff_t_lo,
            true,
            &self.k,
            plan_tk,
            &self.a_zhz,
            &self.ff_v_lo,
        )?;
        let knew_test = GpuBsrMatrix::zero(&self.gpu, &self.k_struct)?;
        self.gpu.mcw_ff_combine_dev(
            self.k_struct.nblock,
            &self.q.values,
            &self.ff_q_lo,
            &self.a_zhz.values,
            &self.ff_v_lo,
            &knew_test.values,
        )?;

        // host f64 references
        let n4 = self.n_atom * BS;
        let mut kd = vec![0.0f32; n4 * n4];
        self.k_to_dense_into(&mut kd)?;
        let s_host = self.s.to_host(&self.gpu)?;
        let mut sd = vec![0.0f32; n4 * n4];
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            self.n_atom,
            &s_host.row_ptr,
            &s_host.col_idx,
            &s_host.values,
            &mut sd,
        );
        let k64: Vec<f64> = kd.iter().map(|&x| x as f64).collect();
        let s64: Vec<f64> = sd.iter().map(|&x| x as f64).collect();
        let tref = crate::methods::sparse::bsr4::matmul_f64_mt(n4, &k64, &s64);
        let qref = crate::methods::sparse::bsr4::matmul_f64_mt(n4, &tref, &k64);
        let uref = crate::methods::sparse::bsr4::matmul_f64_mt(n4, &qref, &s64);
        let vref = crate::methods::sparse::bsr4::matmul_f64_mt(n4, &uref, &k64);
        let kref: Vec<f64> = (0..n4 * n4)
            .map(|i| 3.0 * qref[i] - 2.0 * vref[i])
            .collect();

        let check = |gpu: &crate::methods::sparse::gpu_sparse::SparseBsr4Gpu,
                     st: &std::sync::Arc<crate::methods::sparse::gpu_sparse::GpuBsrStructure>,
                     hi: &[f32],
                     lo: &[f32],
                     reff: &[f64]|
         -> Result<(f64, f64)> {
            let rp = st.row_ptr_host(gpu)?;
            let ci = st.col_idx_host(gpu)?;
            let mut e_hi = 0.0f64;
            let mut e_ff = 0.0f64;
            let mut den = 0.0f64;
            for i in 0..self.n_atom {
                for b in rp[i] as usize..rp[i + 1] as usize {
                    let j = ci[b] as usize;
                    for l in 0..BS2 {
                        let r = i * BS + l / BS;
                        let c = j * BS + l % BS;
                        let t = reff[r * n4 + c];
                        let h = hi[b * BS2 + l] as f64;
                        let f = h + lo[b * BS2 + l] as f64;
                        e_hi += (h - t) * (h - t);
                        e_ff += (f - t) * (f - t);
                        den += t * t;
                    }
                }
            }
            Ok((
                e_hi.sqrt() / den.sqrt().max(1e-30),
                e_ff.sqrt() / den.sqrt().max(1e-30),
            ))
        };
        let (e1h, e1f) = check(&self.gpu, &self.t_ks_struct, &t_hi_snap, &t_lo_snap, &tref)?;
        // V_ff via the verified kernel (product 4)
        let mut v_hi = vec![0.0f32; self.k_struct.nblock * BS2];
        let mut v_lo = vec![0.0f32; self.k_struct.nblock * BS2];
        self.gpu.read_f32(&self.a_zhz.values, &mut v_hi)?;
        self.gpu.read_f32(&self.ff_v_lo, &mut v_lo)?;
        let (e4h, e4f) = check(&self.gpu, &self.k_struct, &v_hi, &v_lo, &vref)?;
        eprintln!("[ff_test] V=UK err(hi)={e4h:.3e} err(ff)={e4f:.3e}");
        // NOTE: t_ks now holds U_ff (product 3 overwrote it); check U_ff.
        let mut u_hi = vec![0.0f32; nt];
        let mut u_lo = vec![0.0f32; nt];
        self.gpu.read_f32(&self.t_ks.values, &mut u_hi)?;
        self.gpu.read_f32(&self.ff_t_lo, &mut u_lo)?;
        let (e3h, e3f) = check(&self.gpu, &self.t_ks_struct, &u_hi, &u_lo, &uref)?;
        eprintln!("[ff_test] U=QS err(hi)={e3h:.3e} err(ff)={e3f:.3e}");
        // f32-emulated combine (mirror of bsr4_mcw_ff_combine — used for
        // host-side replay comparisons below)
        let ff_combine = |qh: f32, ql: f32, vh: f32, vl: f32| -> f32 {
            3.0f32.mul_add(qh, (-2.0f32).mul_add(vh, 3.0f32.mul_add(ql, -2.0f32 * vl)))
        };
        // final K' vs dense-f64 McWeeny reference (masked on M_K)
        let mut e_mx_at = (0usize, 0usize, 0usize);
        let den_k;
        let kd2;
        {
            let mut kd2v = vec![0.0f32; self.k_struct.nblock * BS2];
            self.gpu.read_f32(&knew_test.values, &mut kd2v)?;
            let mut e = 0.0f64;
            let mut den = 0.0f64;
            let mut e_mx = 0.0f64;
            let mut n_big = 0usize;
            for i in 0..self.n_atom {
                for b in self.m_k.0[i] as usize..self.m_k.0[i + 1] as usize {
                    let j = self.m_k.1[b] as usize;
                    for l in 0..BS2 {
                        let r = i * BS + l / BS;
                        let c = j * BS + l % BS;
                        let d = kd2v[b * BS2 + l] as f64 - kref[r * n4 + c];
                        e += d * d;
                        den += kref[r * n4 + c] * kref[r * n4 + c];
                        if d.abs() > e_mx {
                            e_mx = d.abs();
                            e_mx_at = (i, j, l);
                        }
                        if d.abs() > 1e-7 {
                            n_big += 1;
                        }
                    }
                }
            }
            eprintln!("[ff_test] K' vs f64-McW ref: {:.3e}  max|d|={e_mx:.3e} at {e_mx_at:?}  n(>1e-7)={n_big}", e.sqrt() / den.sqrt().max(1e-30));
            den_k = den;
            kd2 = kd2v;
        }
        // NOTE: q/lo buffers still hold Q_ff (untouched by product 3/4)
        let mut q_hiv = vec![0.0f32; self.k_struct.nblock * BS2];
        let mut q_lov = vec![0.0f32; self.k_struct.nblock * BS2];
        self.gpu.read_f32(&self.q.values, &mut q_hiv)?;
        self.gpu.read_f32(&self.ff_q_lo, &mut q_lov)?;
        let (e2h, e2f) = check(&self.gpu, &self.k_struct, &q_hiv, &q_lov, &qref)?;

        // Host-side replay of the fused final from the GPU's OWN
        // (Q_ff, U_ff) buffers: V_h = U_ff·K in f64, K'_h = 3Q_ff−2V_h.
        // If K'_h ≈ kref but GPU K' differs → bug inside the final kernel.
        {
            let mut ud = vec![0.0f64; n4 * n4];
            let trp = self.t_ks_struct.row_ptr_host(&self.gpu)?;
            let tci = self.t_ks_struct.col_idx_host(&self.gpu)?;
            for i in 0..self.n_atom {
                for b in trp[i] as usize..trp[i + 1] as usize {
                    let j = tci[b] as usize;
                    for l in 0..BS2 {
                        ud[(i * BS + l / BS) * n4 + j * BS + l % BS] =
                            u_hi[b * BS2 + l] as f64 + u_lo[b * BS2 + l] as f64;
                    }
                }
            }
            let mut qd = vec![0.0f64; n4 * n4];
            for i in 0..self.n_atom {
                for b in self.m_k.0[i] as usize..self.m_k.0[i + 1] as usize {
                    let j = self.m_k.1[b] as usize;
                    for l in 0..BS2 {
                        qd[(i * BS + l / BS) * n4 + j * BS + l % BS] =
                            q_hiv[b * BS2 + l] as f64 + q_lov[b * BS2 + l] as f64;
                    }
                }
            }
            let vh = crate::methods::sparse::bsr4::matmul_f64_mt(n4, &ud, &k64);
            let knh: Vec<f64> = (0..n4 * n4).map(|i| 3.0 * qd[i] - 2.0 * vh[i]).collect();
            let mut kd3 = vec![0.0f32; self.k_struct.nblock * BS2];
            self.gpu.read_f32(&knew_test.values, &mut kd3)?;
            let (mut e_ref, mut e_dev, mut den) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..self.n_atom {
                for b in self.m_k.0[i] as usize..self.m_k.0[i + 1] as usize {
                    let j = self.m_k.1[b] as usize;
                    for l in 0..BS2 {
                        let r = i * BS + l / BS;
                        let c = j * BS + l % BS;
                        e_ref += (knh[r * n4 + c] - kref[r * n4 + c]).powi(2);
                        e_dev += (kd3[b * BS2 + l] as f64 - knh[r * n4 + c]).powi(2);
                        den += kref[r * n4 + c] * kref[r * n4 + c];
                    }
                }
            }
            eprintln!(
                "[ff_test] host-replay K'_h vs f64 ref: {:.3e} | GPU K' vs K'_h: {:.3e}",
                e_ref.sqrt() / den.sqrt().max(1e-30),
                e_dev.sqrt() / den.sqrt().max(1e-30)
            );
            // f32-emulated combine on the same (Q_ff,V_ff) buffers —
            // if it matches the GPU kernel, the residual is the scheme's
            // own precision, not a kernel bug.
            let mut e_kern = 0.0f64;
            let mut e_hionly = 0.0f64;
            let mut e_ffv = 0.0f64;
            for i in 0..self.n_atom {
                for b in self.m_k.0[i] as usize..self.m_k.0[i + 1] as usize {
                    let j = self.m_k.1[b] as usize;
                    for l in 0..BS2 {
                        let ix = b * BS2 + l;
                        let rc = (i * BS + l / BS) * n4 + j * BS + l % BS;
                        let emu = ff_combine(q_hiv[ix], q_lov[ix], v_hi[ix], v_lo[ix]) as f64;
                        e_kern += (emu - kd3[ix] as f64).powi(2);
                        e_ffv += (emu - kref[rc]).powi(2);
                        let hi = 3.0f32 * q_hiv[ix] - 2.0f32 * v_hi[ix];
                        e_hionly += (hi as f64 - kref[rc]).powi(2);
                    }
                }
            }
            eprintln!(
                "[ff_test] host-f32-emulated combine vs GPU K': {:.3e}",
                e_kern.sqrt() / den.sqrt().max(1e-30)
            );
            eprintln!(
                "[ff_test] emulated-combine vs kref: {:.3e} | hi-only 3Q−2V vs kref: {:.3e}",
                e_ffv.sqrt() / den.sqrt().max(1e-30),
                e_hionly.sqrt() / den.sqrt().max(1e-30)
            );
            // worst element dump
            {
                let (i, j, l) = e_mx_at;
                let b0 = self.m_k.0[i] as usize;
                let b1 = self.m_k.0[i + 1] as usize;
                let mut bix = None;
                for b in b0..b1 {
                    if self.m_k.1[b] as usize == j {
                        bix = Some(b);
                        break;
                    }
                }
                let ix = bix.unwrap() * BS2 + l;
                let kn = ff_combine(q_hiv[ix], q_lov[ix], v_hi[ix], v_lo[ix]);
                eprintln!("[ff_test] worst el ({i},{j},l{l}): Qhi={:.9e} Qlo={:.3e} Vhi={:.9e} Vlo={:.3e} GPU={:.9e} emu={:.9e} ref={:.9e}",
                    q_hiv[ix], q_lov[ix], v_hi[ix], v_lo[ix], kd3[ix], kn,
                    kref[(i * BS + l / BS) * n4 + j * BS + l % BS]);
            }
            // Theoretical floor: pure f32 rounding of the exact update.
            let mut e_rnd = 0.0f64;
            for i in 0..self.n_atom {
                for b in self.m_k.0[i] as usize..self.m_k.0[i + 1] as usize {
                    let j = self.m_k.1[b] as usize;
                    for l in 0..BS2 {
                        let x = kref[(i * BS + l / BS) * n4 + j * BS + l % BS];
                        e_rnd += (x - x as f32 as f64).powi(2);
                    }
                }
            }
            eprintln!(
                "[ff_test] pure f32 rounding floor: {:.3e}",
                e_rnd.sqrt() / den.sqrt().max(1e-30)
            );
        }

        // Masked-f64 reference: Q through the plan terms only — separates
        // mask TRUNCATION error from arithmetic error. M_TKS = M_K clone,
        // but true supp(K·S) reaches ~r_hs beyond M_K — the plan drops
        // those left-operand terms.
        let tks_dummy =
            Bsr4Matrix::from_structure(self.n_atom, self.m_t_ks.0.clone(), self.m_t_ks.1.clone())?;
        let k_dummy =
            Bsr4Matrix::from_structure(self.n_atom, self.m_k.0.clone(), self.m_k.1.clone())?;
        let plan_h = build_spgemm_plan_bsym(&tks_dummy, &k_dummy, &self.m_k)?;
        let mut q_mask = vec![0.0f64; n4 * n4];
        for i in 0..self.n_atom {
            let a0 = self.m_t_ks.0[i] as usize;
            let c0 = self.m_k.0[i] as usize;
            let c1 = self.m_k.0[i + 1] as usize;
            for cb in c0..c1 {
                let j = self.m_k.1[cb] as usize;
                for t in plan_h.plan_ptr[cb] as usize..plan_h.plan_ptr[cb + 1] as usize {
                    let ka = self.m_t_ks.1[a0 + plan_h.plan_a_idx[t] as usize] as usize;
                    let kb = plan_h.plan_b_idx[t] as usize;
                    let kj = self.m_k.1[kb] as usize;
                    for r in 0..BS {
                        for m in 0..BS {
                            for c in 0..BS {
                                q_mask[(i * BS + r) * n4 + j * BS + c] += tref
                                    [(i * BS + r) * n4 + ka * BS + m]
                                    * k64[(kj * BS + m) * n4 + j * BS + c];
                            }
                        }
                    }
                }
            }
        }
        let (e2m, e_tr): (f64, f64) = {
            let mut em = 0.0f64;
            let mut et = 0.0f64;
            let mut den = 0.0f64;
            // GPU Q_ff vs masked ref, and masked vs dense ref
            let mut hi = vec![0.0f32; self.k_struct.nblock * BS2];
            let mut lo = vec![0.0f32; self.k_struct.nblock * BS2];
            self.gpu.read_f32(&self.q.values, &mut hi)?;
            self.gpu.read_f32(&self.ff_q_lo, &mut lo)?;
            for i in 0..self.n_atom {
                for b in self.m_k.0[i] as usize..self.m_k.0[i + 1] as usize {
                    let j = self.m_k.1[b] as usize;
                    for l in 0..BS2 {
                        let r = i * BS + l / BS;
                        let c = j * BS + l % BS;
                        let f = hi[b * BS2 + l] as f64 + lo[b * BS2 + l] as f64;
                        em += (f - q_mask[r * n4 + c]) * (f - q_mask[r * n4 + c]);
                        et += (q_mask[r * n4 + c] - qref[r * n4 + c])
                            * (q_mask[r * n4 + c] - qref[r * n4 + c]);
                        den += qref[r * n4 + c] * qref[r * n4 + c];
                    }
                }
            }
            (
                em.sqrt() / den.sqrt().max(1e-30),
                et.sqrt() / den.sqrt().max(1e-30),
            )
        };
        eprintln!("[ff_test] Q_ff vs masked-f64: {e2m:.3e} | mask truncation (masked vs dense f64): {e_tr:.3e}");
        Ok((e1h, e1f, e2h, e2f))
    }

    /// Debug probe (FF32-POLISH): Frobenius norms of the float-float LO
    /// buffers on host — expected ~eps·‖hi‖ ~1e-7× values if the
    /// TwoProd/TwoSum compensation is live; ~0 means the compiler ate it.
    pub fn ff_lo_norms(&mut self) -> Result<(f64, f64)> {
        let nq = self.k_struct.nblock * BS2;
        let nt = self.t_ks_struct.nblock * BS2;
        let mut vq = vec![0.0f32; nq];
        let mut vt = vec![0.0f32; nt];
        self.gpu.read_f32(&self.ff_q_lo, &mut vq)?;
        self.gpu.read_f32(&self.ff_t_lo, &mut vt)?;
        let sq = |v: &[f32]| {
            v.iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt()
        };
        Ok((sq(&vq), sq(&vt)))
    }

    /// Download K values into persistent `k_host` (no extra alloc).
    pub fn k_values_host(&mut self) -> Result<&[f32]> {
        self.gpu.read_f32(&self.k.values, &mut self.k_host)?;
        Ok(&self.k_host)
    }

    /// Read K into a caller dense pad buffer (full mask: BSR order is row-major blocks).
    pub fn k_to_dense_into(&mut self, out: &mut [f32]) -> Result<()> {
        self.gpu.read_f32(&self.k.values, &mut self.k_host)?;
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            self.n_atom,
            &self.m_k.0,
            &self.m_k.1,
            &self.k_host,
            out,
        );
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
    /// Returns (q_phys, q_dum, status, r_I, Tr[f64], tc2_iters) — `r_I` is
    /// the *relative* idempotency residual `‖KSK−K‖/‖K‖` (R12); `status`
    /// is `PurifyStatus` (NumericalFloor ⇒ energies NOT validated).
    pub fn run_scc(
        &mut self,
        ns_max_iter: usize,
        ns_tol: f32,
        tc2_max_iter: usize,
        tc2_tol: f32,
        tc2_check_every: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, PurifyStatus, f32, f64, usize)> {
        // 1. Z ≈ S⁻¹ (cold start — one-shot purify API has no previous Z)
        let (_, _) = self.compute_z(ns_max_iter, ns_tol, 3, false)?;

        // 2. K0
        let _ = self.compute_k0(0.1)?;

        // 3. TC2
        let (status, r_i, tr, iters) = self.tc2_purify(tc2_max_iter, tc2_tol, tc2_check_every)?;

        // 4. Mulliken charges (reuses t_ks — converged TC2 leaves T = K·S)
        let (charges, q_dum) = self.mulliken_charges()?;

        Ok((charges, q_dum, status, r_i, tr, iters))
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
fn block_map_same(a: &(Vec<u32>, Vec<u32>), b: &(Vec<u32>, Vec<u32>), n_atom: usize) -> Vec<i32> {
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
                    if i == j && r == c {
                        x -= 1.0;
                    }
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
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
                    for k in 0..BS2 {
                        v[k] = (rng.next() * 2.0 - 1.0) as f32;
                    }
                    if i == j {
                        for r in 0..BS {
                            for c in (r + 1)..BS {
                                let avg = 0.5 * (v[r * BS + c] + v[c * BS + r]);
                                v[r * BS + c] = avg;
                                v[c * BS + r] = avg;
                            }
                        }
                        // Add diagonal shift for gap
                        for r in 0..BS {
                            v[r * BS + r] += 2.0;
                        }
                    } else {
                        let mut vt = [0.0f32; BS2];
                        for r in 0..BS {
                            for c in 0..BS {
                                vt[c * BS + r] = v[r * BS + c];
                            }
                        }
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
                        v[0] = 1.0;
                        v[5] = 1.0;
                        v[10] = 1.0;
                        v[15] = 1.0;
                    } else {
                        // Off-diagonal: small random
                        for k in 0..BS2 {
                            v[k] = (rng.next() * 0.2) as f32;
                        }
                        for r in 0..BS {
                            for c in (r + 1)..BS {
                                let avg = 0.5 * (v[r * BS + c] + v[c * BS + r]);
                                v[r * BS + c] = avg;
                                v[c * BS + r] = avg;
                            }
                        }
                        let mut vt = [0.0f32; BS2];
                        for r in 0..BS {
                            for c in 0..BS {
                                vt[c * BS + r] = v[r * BS + c];
                            }
                        }
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
        let mut ws =
            SparseSystemWorkspace::new(gpu, &h0, &s, &mask, &mask, None, &all4, nocc).unwrap();

        // Run full SCC pipeline. r_I is the RELATIVE residual ‖KSK−K‖/‖K‖ (R12).
        let (charges, q_dum, status, r_i, tr, iters) = ws.run_scc(30, 1e-4, 40, 1e-5, 1).unwrap();
        println!("SCC: {iters} TC2 iters, r_I={r_i:e} (relative), Tr(KS)={tr:.6}");
        println!("Mulliken charges: {:?}", &charges);
        let qd_max: f32 = q_dum.iter().map(|x| x.abs()).fold(0.0, f32::max);
        println!("Dummy occupation: {qd_max:e} (should be ~0, R14)");

        // Verify convergence — relative idempotency residual.
        assert_eq!(
            status,
            crate::methods::sparse::gpu_sparse::PurifyStatus::Converged
        );
        assert!(r_i < 1e-5, "TC2 did not converge: r_I={r_i:e} (G2)");
        assert!(
            (tr - nocc as f64).abs() < 1e-5,
            "Tr(KS) mismatch: {tr} vs {nocc} (G2)"
        );
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
        assert!(
            (tr_ks - nocc).abs() < 1e-5,
            "host Tr(KS)={tr_ks} != Nocc={nocc} (G1.8)"
        );
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
        let mut ws =
            SparseSystemWorkspace::new(gpu, &h0, &s, &mask, &mask, None, &all4, nocc).unwrap();
        let (rz_reported, iters) = ws.compute_z(30, 1e-4, 3, false).unwrap();

        // Recompute T = Z·S and measure the same residual two ways.
        match &ws.plan_zs {
            Some(plan) => ws
                .gpu
                .spgemm_plan_bsym_dev(&ws.z, &ws.s, plan, &ws.t_zs)
                .unwrap(),
            None => ws.gpu.spgemm_bsym_dev(&ws.z, &ws.s, &ws.t_zs).unwrap(),
        }
        ws.gpu
            .identity_residual_to_dev(
                &ws.t_zs_struct,
                &ws.t_zs.values,
                &ws.reduce_partial,
                &ws.reduce_a,
                &ws.reduce_b,
                &ws.residual_buf,
            )
            .unwrap();
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
        // The production-reported residual must agree with the independent
        // recomputation — this is what the normalization bug escaped.
        assert!(
            (rz_reported - rz_host).abs() < 0.1 * scale + 1e-7,
            "compute_z reported R_Z={rz_reported:e} but true residual is {rz_host:e}"
        );
        assert!(
            rz_host < 1e-3,
            "Z is not a converged inverse: host R_Z={rz_host:e}"
        );
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
        let mut ws =
            SparseSystemWorkspace::new(gpu, &h0, &s1, &mask, &mask, None, &all4, nocc).unwrap();

        let (rz1, it1) = ws.compute_z(30, 1e-4, 3, false).unwrap();
        println!("geom1 cold: {it1} iters R_Z={rz1:e}");

        // Perturbed geometry: small uniform scaling of S (stand-in for a
        // small displacement; keeps SPD and stays on the same mask).
        let mut s2 = s1.clone();
        for v in s2.values.iter_mut() {
            *v *= 1.02;
        }
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
        assert!(
            rz_host_warm < 1e-3,
            "warm-start Z wrong vs S2: R_Z={rz_host_warm:e}"
        );

        // Cold restart on the *dirty* Z buffer: must zero stale off-diagonals.
        let (rz3, it3) = ws.compute_z(30, 1e-4, 3, false).unwrap();
        ws.gpu.spgemm_bsym_dev(&ws.z, &ws.s, &ws.t_zs).unwrap();
        ws.gpu.read_f32(&ws.t_zs.values, &mut t).unwrap();
        let rz_host_cold = host_identity_rz(&t, &ws.m_t_zs.0, &ws.m_t_zs.1, n_atom) / n_orb.sqrt();
        println!("geom2 cold-on-dirty: {it3} iters R_Z={rz3:e} host={rz_host_cold:e}");
        assert!(
            rz_host_cold < 1e-3,
            "cold restart left stale Z: R_Z={rz_host_cold:e} (R6)"
        );
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
        let mut ws =
            SparseSystemWorkspace::new(gpu, &h0, &s, &mask, &mask, None, &all4, nocc).unwrap();

        // First run.
        let (_, _, _, r_i1, tr1, _) = ws.run_scc(30, 1e-4, 60, 1e-5, 1).unwrap();
        println!("Run 1: r_I={r_i1:e}, Tr={tr1:.6}");

        // Second run with same geometry (should give same result).
        let (_, _, _, r_i2, tr2, _) = ws
            .run_scc(30, 1e-4, 60, 1e-5, 1)
            .unwrap_or_else(|e| panic!("Run 2 failed (no skip): {e}"));
        println!("Run 2: R_I={r_i2:e}, Tr={tr2:.6}");

        // Both runs should converge to the same state.
        assert!((tr1 - tr2).abs() < 1e-4, "Tr mismatch: {tr1} vs {tr2}");
        assert!((r_i1 - r_i2).abs() < 1e-4, "R_I mismatch: {r_i1} vs {r_i2}");
    }

    /// Host-f64 dense reference of the recovered-K state: returns
    /// (Tr(KS), ‖KSK−K‖_F/‖K‖_F) computed from the downloaded K and the
    /// host S — the S1 cross-check that the device-reported diagnostics
    /// describe the same matrix that feeds charges and the energy.
    fn host_k_diagnostics(k_dense: &[f32], s_dense: &[f32], n: usize) -> (f64, f64) {
        let mut ks = vec![0.0f64; n * n];
        let mut ksk = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut a = 0.0f64;
                for l in 0..n {
                    a += k_dense[i * n + l] as f64 * s_dense[l * n + j] as f64;
                }
                ks[i * n + j] = a;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let mut a = 0.0f64;
                for l in 0..n {
                    a += ks[i * n + l] * k_dense[l * n + j] as f64;
                }
                ksk[i * n + j] = a;
            }
        }
        let mut tr = 0.0f64;
        let mut num = 0.0f64;
        let mut kn = 0.0f64;
        for i in 0..n {
            tr += ks[i * n + i];
        }
        for i in 0..n {
            for j in 0..n {
                let d = ksk[i * n + j] - k_dense[i * n + j] as f64;
                num += d * d;
                kn += (k_dense[i * n + j] as f64) * (k_dense[i * n + j] as f64);
            }
        }
        (tr, num.sqrt() / kn.sqrt())
    }

    /// S1 contract: the P-TC2 trace guard must act on the state whose
    /// residual is evaluated (no stale P²), and `purify_hscc_p`'s returned
    /// (R_I, Tr) must describe the RECOVERED K — checked against a host-f64
    /// recomputation of the identical matrix.
    #[test]
    fn test_p_tc2_guard_and_recovery_contract() {
        let Some(gpu) = try_gpu() else { return };
        let n_atom = 3;
        let nocc = 3.0f32;
        let mask = build_full_mask(n_atom);
        let h0 = random_symmetric_bsr4(n_atom, &mask, 42);
        let s = make_overlap_bsr4(n_atom, &mask, 99);
        let all4 = vec![4u8; n_atom];
        let mut ws =
            SparseSystemWorkspace::new(gpu, &h0, &s, &mask, &mask, None, &all4, nocc).unwrap();
        ws.upload_h_scc(&h0).unwrap();
        let (rz, _) = ws.compute_z(30, 1e-4, 3, false).unwrap();
        println!("NS inverse: R_Z={rz:e}");

        // Baseline P-TC2 convergence.
        ws.compute_p0_from_hscc(0.1).unwrap();
        let (st, r_i, tr, iters) = ws.tc2_purify_p(60, 1e-5, 1).unwrap();
        println!("P-TC2 cold: {iters} iters status={st:?} R_I={r_i:e} Tr(P)={tr:.6}");
        assert_eq!(st, PurifyStatus::Converged);

        // Seed a leaked eigenvalue: scaling the converged P by 1+1e-4 gives
        // eigenvalues {0, 1.0001}; the squaring branch amplifies the leak
        // ~2×/iter — the exact runaway signature the guard exists for.
        ws.gpu
            .scale_dev(ws.p.struct_.nblock, 1.0001, &ws.p.values)
            .unwrap();
        let g0 = ws.guard_fires;
        let (st2, r_i2, tr2, it2) = ws.tc2_purify_p(60, 1e-5, 1).unwrap();
        let fires = ws.guard_fires - g0;
        println!("P-TC2 seeded leak: {it2} iters status={st2:?} R_I={r_i2:e} Tr(P)={tr2:.6} guard_fires={fires}");
        assert!(
            fires > 0,
            "trace guard never fired on a seeded λ=1.0001 leak"
        );
        let tol_tr = crate::methods::sparse::gpu_sparse::tc2_trace_tol(nocc as f64);
        assert!(
            (tr2 - nocc as f64).abs() <= tol_tr,
            "accepted state with Tr(P)={tr2} != Nocc={nocc} (guard returned an invalid trace)"
        );

        // Recovery contract: returned diagnostics must describe the
        // recovered K — recompute Tr(KS) and ‖KSK−K‖/‖K‖ in host f64.
        ws.recover_k_from_p().unwrap();
        ws.spgemm_ks().unwrap();
        ws.t_ks_valid = true;
        let (r_i_k, tr_ks) = ws.recovered_k_diagnostics().unwrap();
        let k_host = ws.k_to_host().unwrap();
        let k_dense = k_host.to_dense();
        let s_dense = s.to_dense();
        let (tr_host, r_i_host) = host_k_diagnostics(&k_dense, &s_dense, n_atom * BS);
        println!(
            "recovered K: dev R_I={r_i_k:e} Tr={tr_ks:.6} | host R_I={r_i_host:e} Tr={tr_host:.6}"
        );
        assert!(
            (tr_ks - tr_host).abs() < 1e-4,
            "Tr(KS) mismatch dev={tr_ks} host={tr_host}"
        );
        assert!(
            (r_i_k as f64 - r_i_host).abs() < 1e-4 + 0.5 * r_i_host,
            "R_I(K) mismatch dev={r_i_k:e} host={r_i_host:e} — reported residual is not the returned state's"
        );
    }

    /// S1 contract: same seeded-leak exercise on the K-TC2 path — the guard
    /// rescales K and T=K·S consistently and the reported post-rescale
    /// trace is a measurement, not Nocc by assignment.
    #[test]
    fn test_k_tc2_guard_fires() {
        let Some(gpu) = try_gpu() else { return };
        let n_atom = 3;
        let nocc = 3.0f32;
        let mask = build_full_mask(n_atom);
        let h0 = random_symmetric_bsr4(n_atom, &mask, 42);
        let s = make_overlap_bsr4(n_atom, &mask, 99);
        let all4 = vec![4u8; n_atom];
        let mut ws =
            SparseSystemWorkspace::new(gpu, &h0, &s, &mask, &mask, None, &all4, nocc).unwrap();

        let (_, _, st, r_i, tr, _) = ws.run_scc(30, 1e-4, 60, 1e-5, 1).unwrap();
        println!("K-TC2 baseline: status={st:?} R_I={r_i:e} Tr={tr:.6}");
        assert_eq!(st, PurifyStatus::Converged);

        // Same λ>1 seed on K: Tr(KS) scales linearly with K.
        ws.gpu
            .scale_dev(ws.k.struct_.nblock, 1.0001, &ws.k.values)
            .unwrap();
        let g0 = ws.guard_fires;
        let (st2, r_i2, tr2, it2) = ws.tc2_purify(60, 1e-5, 1).unwrap();
        let fires = ws.guard_fires - g0;
        println!("K-TC2 seeded leak: {it2} iters status={st2:?} R_I={r_i2:e} Tr(KS)={tr2:.6} guard_fires={fires}");
        assert!(
            fires > 0,
            "K-TC2 trace guard never fired on a seeded λ=1.0001 leak"
        );
        let tol_tr = crate::methods::sparse::gpu_sparse::tc2_trace_tol(nocc as f64);
        assert!(
            (tr2 - nocc as f64).abs() <= tol_tr,
            "accepted state with Tr(KS)={tr2} != Nocc={nocc}"
        );
    }
}
