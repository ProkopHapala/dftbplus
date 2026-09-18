//! Persistent complex/PBC SCC plan — the complex-valued sibling of
//! `GpuSccPlan` for periodic boundary conditions at arbitrary k.
//!
//! # Geometry
//!
//! - `n` — orbitals per unit cell; `n_atoms` — atoms per cell
//! - `n_rep` — replicas (geometries/constraints in the multi-system batch)
//! - `nk` — k-points per replica; flat system index `sid = rep*nk + kpt`
//! - `n_sys = n_rep*nk` — all complex matrix buffers are `[n_sys][n*n]`
//!   `float2` row-major; all charge vectors are REAL `[n_rep][n_atoms]`.
//!
//! # Invariants (identical to GpuSccPlan — the no-overhead contract)
//!
//! - ALL buffers allocated at construction; `rt.alloc_count` must not grow
//!   inside `scc_step_diis_enq`.
//! - ALL kernels built at construction with io buffers bound once; the only
//!   `set_arg` calls are per-solve scalars (`bind_solve_params`,
//!   `bind_mix_params`) — zero per-iteration arg traffic.
//! - Chunked enqueue-only iterations: `scc_step_diis_enq` does NO host
//!   readback; `read_chunk_status` syncs once per chunk.
//! - One replica mask `active_r[n_rep]`; flat-system kernels index it as
//!   `active[sid/nk]` — the device-side DIIS convergence clear stops all
//!   k-point workgroups of a done replica with zero extra launches.
//!
//! # Per-iteration launch sequence
//!
//! 1. `zdq_v_batched`        — dq = q−q0, V = γ·dq          (rep grid, real)
//! 2. `zhscc_batched`        — H_scc = H0 + ½·S·(V_A+V_B)   (flat, complex)
//! 3. warm: `zgemm` C†·H→temp, temp·C→hp — projected near-diagonal problem
//!    cold: `zgemm` X†·H→temp, temp·X→hp  (X = S^{-1/2} from set_geometry)
//! 4. `jacobi_hermitian_cyclic_global_batched` — ε on Re(diag hp);
//!    cold: V=I→cp; warm: rotates c in place (init_v=1)
//! 5. `zextract_diagonal_batched` — eig_diag = Re(diag hp)  (flat)
//! 6. `kpoint_occ_batched`   — shared-μ solve over all nk bands:
//!    occ_w = w_k·f, mu, e_band, mts                        (rep grid)
//! 7. cold: `zgemm` X·cp→c (marks warm for next iter);
//!    warm: `zsnormalize_batched` — S(k)-metric column renorm (f32 repair)
//! 8. `zgemm` S·C→sc + `zsc_mulliken_batched` — per-(rep,k) charges
//!    qk_μ = 2·Re Σ_t w_t C_μt conj((SC)_μt); D never materialized
//! 9. `kpoint_qreduce_batched` — q_new = Σ_k qk             (rep grid)
//! 10. `diis_step_batched`   — REAL kernel reused with batch=n_rep:
//!    Δq-space DIIS, device-side rms + convergence clear, in-place commit
//!
//! ≈10 launches/iteration (real path: 9) — the extra launch is the
//! unavoidable k-reduction for the shared μ.
//!
//! # Deliberate v1 limits (fail loud, not silent)
//!
//! - Cross-geometry warm-basis repair (complex metric residual + Newton)
//!   is NOT implemented — `set_geometry` always rebuilds X by a full
//!   Hermitian Jacobi on S(k) and drops the warm basis (b_warm=false).
//!   In-solve warm start (the dominant win) is unaffected.
//! - No occ_mask/occ_idx dual paths: occupation is uniform —
//!   kT=0 → bisection on the weighted band count (integer occ), kT>0 →
//!   Fermi smearing; both write occ_w = w_k·f.
//! - `zbuild_density_batched` exists for the future EDM/forces W-build
//!   but is unused on the charge path.
//!
//! Prototype: Dense_Multi_PBC.chat.md; architecture notes:
//! Dense_Multi_PBC.arch_notes.md (same task directory).

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_hermitian::{
    render_hermitian_source, render_zmatrix_source, ZOP_H, ZOP_N, ZTILE_K, ZTILE_M, ZTILE_N,
};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::prm::Float2;
use ocl::{Buffer, Kernel};

const MATRIX_KERNEL_TEMPLATE: &str = include_str!("gpu_matrix_ops.cl");

fn max_finite_f32(xs: &[f32], ctx: &str) -> Result<f32> {
    let mut m = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if !x.is_finite() {
            return Err(DftbError::InvalidInput(format!(
                "{ctx}[{i}]={x} non-finite"
            )));
        }
        if x > m {
            m = x;
        }
    }
    if !m.is_finite() {
        return Err(DftbError::InvalidInput(format!(
            "{ctx}: no finite values (len={})",
            xs.len()
        )));
    }
    Ok(m)
}

/// Persistent complex-PBC SCC plan for fixed (n, n_atoms, n_rep, nk).
///
/// The caller owns the persistent inputs (h0, s — complex `[n_sys][n*n]`;
/// gamma, q0 — real `[n_rep]`-indexed; orb_atom `[n_rep][n]`) whose cl_mem
/// handles must stay alive for the plan's lifetime; `set_geometry` rewrites
/// their contents in place.
pub struct GpuPbcPlan {
    n: usize,
    n_atoms: usize,
    n_rep: usize,
    nk: usize,
    n_sys: usize, // n_rep * nk — flat (replica,k) count

    // ---- complex [n_sys·nn] scratch ----
    h_scc: Buffer<Float2>,      // SCC Hamiltonian H(k)
    temp: Buffer<Float2>,       // zgemm temp
    hp: Buffer<Float2>,         // transformed/projected H — eigenvalues on Re(diag)
    cp: Buffer<Float2>,         // eigenvectors, orthonormal basis (cold path)
    pub c: Buffer<Float2>,      // eigenvectors, AO basis (warm basis when b_warm)
    sc: Buffer<Float2>,         // S·C scratch for the Mulliken contraction
    x_buf: Buffer<Float2>,      // X = S^{-1/2} (Löwdin), per (rep,k)
    s_work: Buffer<Float2>,     // Jacobi A workspace for S (destroyed)
    s_v: Buffer<Float2>,        // Jacobi eigenvectors of S
    s_v_scaled: Buffer<Float2>, // s_v·rsqrt(λ)

    // ---- real per-replica charge state ----
    pub q_gpu: Buffer<f32>, // [n_rep*n_atoms] current charges (committed)
    dq: Buffer<f32>,        // [n_rep*n_atoms] Δq
    v: Buffer<f32>,         // [n_rep*n_atoms] γ·Δq
    pub q_new: Buffer<f32>, // [n_rep*n_atoms] Mulliken output charges
    q_next: Buffer<f32>,    // [n_rep*n_atoms] simple-mix buffer (mix path)
    qk: Buffer<f32>,        // [n_sys*n_atoms] per-(rep,k) partial charges

    // ---- real eigensolver state ----
    eig_diag: Buffer<f32>,     // [n_sys*n] Re(diag hp) eigenvalues
    occ_w: Buffer<f32>,        // [n_sys*n] occ weights = w_k·f_bk
    pub mu_out: Buffer<f32>,   // [n_rep] shared chemical potential
    lambda_min: Buffer<f32>,   // [n_sys] λ_min(S(k))
    jacobi_diag: Buffer<f32>,  // [n_sys*4] {off, off/‖A‖_F, stop, sweeps}
    pub active_r: Buffer<i32>, // [n_rep] THE mask (flat kernels read [sid/nk])
    ones_r: Buffer<i32>,       // [n_rep] all-ones — S-solve must never gate
    rms: Buffer<f32>,          // [n_rep] residual RMS
    kw: Buffer<f32>,           // [nk] k-point weights (Σ_k w_k = 1)
    e_scal: Buffer<f64>,       // [n_rep*4] {e_band, mts, dq·v, q0·v}

    // ---- DIIS (per replica — real kernels from gpu_matrix_ops.cl) ----
    pub diis_q_hist: Buffer<f32>,
    pub diis_r_hist: Buffer<f32>,
    pub diis_buf_idx: Buffer<i32>,
    pub diis_n_filled: Buffer<i32>,
    pub diis_coeffs: Buffer<f32>,
    pub diis_flag: Buffer<i32>,
    pub diis_reason: Buffer<i32>,
    // pub diis_work: Buffer<f64>,   // REMOVED 2026-09-18: W13 f64 QR columns moved to
    //                              // __local inside diis_step_batched.
    pub diis_max_hist: usize,

    // ---- host mirrors ----
    pub rms_host: Vec<f32>,    // [n_rep]
    pub active_host: Vec<i32>, // [n_rep]
    pub e_scal_host: Vec<f64>, // [n_rep*4]
    eig_diag_host: Vec<f32>,   // [n_sys*n]
    lambda_min_host: Vec<f32>, // [n_sys]
    jacobi_diag_h: Vec<f32>,   // [n_sys*4]

    // ---- prebuilt kernels (bound once at construction) ----
    k_zdq_v: Kernel,        // zdq_v_batched
    k_zhscc: Kernel,        // zhscc_batched
    k_zxh: Kernel,          // zgemm: X†·H_scc → temp   (cold)
    k_ztx: Kernel,          // zgemm: temp·X → hp        (cold)
    k_zxc: Kernel,          // zgemm: X·cp → c           (cold back-transform)
    k_zxh_warm: Kernel,     // zgemm: C†·H_scc → temp    (warm)
    k_ztx_warm: Kernel,     // zgemm: temp·C → hp        (warm)
    k_zsc: Kernel,          // zgemm: S·c → sc           (Mulliken contraction)
    k_zjacobi: Kernel,      // Hermitian Jacobi, cold (init_v=0, V→cp)
    k_zjacobi_warm: Kernel, // Hermitian Jacobi, warm (init_v=1, V=c in place)
    k_zextract: Kernel,     // zextract_diagonal_batched
    k_kocc: Kernel,         // kpoint_occ_batched (shared-μ + occ_w + e_band)
    k_zmull: Kernel,        // zsc_mulliken_batched
    k_qreduce: Kernel,      // kpoint_qreduce_batched
    k_zsnorm: Kernel,       // zsnormalize_batched (warm-basis S-metric renorm)
    k_s_jacobi: Kernel,     // Hermitian Jacobi on s_work (S eigensolve)
    k_s_scale: Kernel,      // zscale_eigenvectors_batched
    k_s_xgemm: Kernel,      // zgemm: s_v_scaled·s_v† → x_buf
    k_diis: Kernel,         // diis_step_batched (real, batch=n_rep)
    k_residual_mix: Kernel, // residual_and_mix_batched (real, batch=n_rep)
    k_commit: Kernel,       // commit_q_batched (real, batch=n_rep)
    k_energy_tail: Kernel,  // zkpoint_energy_tail_batched

    // ---- per-solve scalars ----
    n_occ: f32,  // occupied-band equivalents per cell
    pub kT: f32, // Fermi smearing kT (Hartree); 0 → integer occ
    /// f64 scalar rotation construction in the Hermitian Jacobi
    /// (HJ_ROT_FP64). Default on — it is off the throughput path; the f32
    /// variant exists for A/B measurement like the real kernel's prec=0.
    pub rot_fp64: bool,
    /// Warm AO basis: `c` holds the previous solve's S(k)-orthonormal
    /// eigenvectors. In-solve warm start only — set_geometry drops it.
    b_warm: bool,
}

impl GpuPbcPlan {
    /// Build the plan: allocates every buffer, builds every kernel, computes
    /// the initial X = S(k)^{-1/2}. `kw` are the k-point weights (must sum to
    /// 1). `g_buf` is the REAL periodic γ matrix `[n_rep][n_atoms²]`.
    pub fn new(
        rt: &mut GpuRuntime,
        s_buf: &Buffer<Float2>,
        h0_buf: &Buffer<Float2>,
        g_buf: &Buffer<f32>,
        q0_buf: &Buffer<f32>,
        orb_atom_buf: &Buffer<i32>,
        n: usize,
        n_atoms: usize,
        n_rep: usize,
        nk: usize,
        kw: &[f32],
    ) -> Result<Self> {
        if n == 0 || n_rep == 0 || nk == 0 {
            return Err(DftbError::InvalidInput(format!(
                "GpuPbcPlan::new: n={n} n_rep={n_rep} nk={nk} must be nonzero"
            )));
        }
        if n > 256 {
            return Err(DftbError::InvalidInput(format!(
                "GpuPbcPlan::new: n={n} exceeds Hermitian-Jacobi capacity 256 (jn/2 pair rotations fit __local rot[128])"
            )));
        }
        if kw.len() != nk {
            return Err(DftbError::InvalidInput(format!(
                "GpuPbcPlan::new: kw len {} != nk {nk}",
                kw.len()
            )));
        }
        let wsum: f64 = kw.iter().map(|&w| w as f64).sum();
        if (wsum - 1.0).abs() > 1e-5 {
            return Err(DftbError::InvalidInput(format!(
                "GpuPbcPlan::new: k-point weights sum to {wsum}, expected 1"
            )));
        }
        let n_sys = n_rep * nk;
        let nn = n * n;

        // ---- complex buffers ----
        let h_scc = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let temp = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let hp = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let cp = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let c = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let sc = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let x_buf = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let s_work = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let s_v = rt.zero_buffer::<Float2>(n_sys * nn)?;
        let s_v_scaled = rt.zero_buffer::<Float2>(n_sys * nn)?;

        // ---- real charge state ----
        let q_gpu = rt.zero_buffer::<f32>(n_rep * n_atoms)?;
        let dq = rt.zero_buffer::<f32>(n_rep * n_atoms)?;
        let v = rt.zero_buffer::<f32>(n_rep * n_atoms)?;
        let q_new = rt.zero_buffer::<f32>(n_rep * n_atoms)?;
        let q_next = rt.zero_buffer::<f32>(n_rep * n_atoms)?;
        let qk = rt.zero_buffer::<f32>(n_sys * n_atoms)?;

        let eig_diag = rt.zero_buffer::<f32>(n_sys * n)?;
        let occ_w = rt.zero_buffer::<f32>(n_sys * n)?;
        let mu_out = rt.zero_buffer::<f32>(n_rep)?;
        let lambda_min = rt.zero_buffer::<f32>(n_sys)?;
        let jacobi_diag = rt.zero_buffer::<f32>(n_sys * 4)?;
        let active_r = rt.buffer_from_slice(&vec![1i32; n_rep])?;
        let ones_r = rt.buffer_from_slice(&vec![1i32; n_rep])?;
        // T06 Phase A: shared gpu_matrix_ops kernels now take a launch-domain
        // work_ids — the PBC plan's domain is n_rep replicas (identity).
        let wids = rt.buffer_from_slice(&(0..n_rep as i32).collect::<Vec<_>>())?;
        let rms = rt.zero_buffer::<f32>(n_rep)?;
        let kw_buf = rt.buffer_from_slice(kw)?;
        let e_scal = rt.zero_buffer::<f64>(n_rep * 4)?;

        let diis_max_hist = n_atoms.min(10).max(1);
        let diis_q_hist = rt.zero_buffer::<f32>(n_rep * diis_max_hist * n_atoms)?;
        let diis_r_hist = rt.zero_buffer::<f32>(n_rep * diis_max_hist * n_atoms)?;
        let diis_buf_idx = rt.zero_buffer::<i32>(n_rep)?;
        let diis_n_filled = rt.zero_buffer::<i32>(n_rep)?;
        let diis_coeffs = rt.zero_buffer::<f32>(n_rep * diis_max_hist)?;
        let diis_flag = rt.zero_buffer::<i32>(n_rep)?;
        let diis_reason = rt.zero_buffer::<i32>(n_rep)?;
        // W13 f64 QR working columns are __local inside diis_step_batched —
        // no global scratch buffer needed.
        // let diis_work = rt.zero_buffer::<f64>(n_rep * diis_max_hist * n_atoms)?;

        // ---- programs (once; program cache hashes the source) ----
        let rot_fp64 = std::env::var("RUST_DFTB_HJACOBI_FP64")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|p| p != 0)
            .unwrap_or(true);
        let wg_jacobi = rt.caps().max_work_group_size.min(512).max(64);
        let hsrc = render_hermitian_source(wg_jacobi, rot_fp64);
        let h_prog = rt.build_program(&hsrc)?;
        let z_prog = rt.build_program(&render_zmatrix_source())?;
        let mat_prog = rt.build_program(MATRIX_KERNEL_TEMPLATE)?; // real kernels (DIIS etc.)
                                                                  // DIIS_MAX_HIST specialization (same as GpuSccPlan — private arrays
                                                                  // in the kernel are compile-time sized).
        let diis_src = MATRIX_KERNEL_TEMPLATE.replace(
            "#ifndef DIIS_MAX_HIST\n#define DIIS_MAX_HIST 10\n#endif",
            &format!("#define DIIS_MAX_HIST {diis_max_hist}"),
        );
        let diis_prog = rt.build_program(&diis_src)?;

        // ============ kernel handles — io buffers bound once ============

        // zdq_v_batched: [0]n_atoms [1]n_rep [2]q [3]q0 [4]G [5]dq [6]V [7]ldq [8]active
        let wg_rep = 256usize;
        let k_zdq_v = Kernel::builder()
            .program(&z_prog)
            .name("zdq_v_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * wg_rep)
            .local_work_size(wg_rep)
            .arg(n_atoms as i32)
            .arg(n_rep as i32)
            .arg(&q_gpu)
            .arg(q0_buf)
            .arg(g_buf)
            .arg(&dq)
            .arg(&v)
            .arg_local::<f32>(n_atoms)
            .arg(&active_r)
            .build()
            .map_err(map_ocl_err)?;

        // zhscc_batched: [0]n [1]n_atoms [2]batch [3]nk [4]H0 [5]S [6]V [7]orb_atom [8]H [9]active
        let k_zhscc = Kernel::builder()
            .program(&z_prog)
            .name("zhscc_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_sys * wg_rep)
            .local_work_size(wg_rep)
            .arg(n as i32)
            .arg(n_atoms as i32)
            .arg(n_sys as i32)
            .arg(nk as i32)
            .arg(h0_buf)
            .arg(s_buf)
            .arg(&v)
            .arg(orb_atom_buf)
            .arg(&h_scc)
            .arg(&active_r)
            .build()
            .map_err(map_ocl_err)?;

        // zgemm_active_batched: [0]n [1]batch [2]op_a [3]op_b [4]alpha [5]beta
        //   [6]A [7]B [8]C [9]As [10]Bs [11]nk [12]active
        let rg = (n + ZTILE_M - 1) / ZTILE_M;
        let cg = (n + ZTILE_N - 1) / ZTILE_N;
        let gws = ocl::SpatialDims::Three(cg * ZTILE_N, rg * ZTILE_M, n_sys);
        let lws = ocl::SpatialDims::Two(ZTILE_N, ZTILE_M);
        let mk_zgemm = |prog: &ocl::Program,
                        op_a: i32,
                        op_b: i32,
                        a: &Buffer<Float2>,
                        b: &Buffer<Float2>,
                        cbuf: &Buffer<Float2>,
                        act: &Buffer<i32>|
         -> Result<Kernel> {
            Kernel::builder()
                .program(prog)
                .name("zgemm_active_batched")
                .queue(rt.queue().clone())
                .global_work_size(gws.clone())
                .local_work_size(lws.clone())
                .arg(n as i32)
                .arg(n_sys as i32)
                .arg(op_a)
                .arg(op_b)
                .arg(1.0f32)
                .arg(0.0f32)
                .arg(a)
                .arg(b)
                .arg(cbuf)
                .arg_local::<Float2>(ZTILE_M * ZTILE_K)
                .arg_local::<Float2>(ZTILE_N * (ZTILE_K + 1))
                .arg(nk as i32)
                .arg(act)
                .build()
                .map_err(map_ocl_err)
        };
        let k_zxh = mk_zgemm(&z_prog, ZOP_H, ZOP_N, &x_buf, &h_scc, &temp, &active_r)?; // temp = X†·H_scc
        let k_ztx = mk_zgemm(&z_prog, ZOP_N, ZOP_N, &temp, &x_buf, &hp, &active_r)?; // hp = temp·X
        let k_zxc = mk_zgemm(&z_prog, ZOP_N, ZOP_N, &x_buf, &cp, &c, &active_r)?; // c = X·cp
        let k_zxh_warm = mk_zgemm(&z_prog, ZOP_H, ZOP_N, &c, &h_scc, &temp, &active_r)?; // temp = C†·H_scc
        let k_ztx_warm = mk_zgemm(&z_prog, ZOP_N, ZOP_N, &temp, &c, &hp, &active_r)?; // hp = temp·C
        let k_zsc = mk_zgemm(&z_prog, ZOP_N, ZOP_N, s_buf, &c, &sc, &active_r)?; // sc = S·c
        let k_s_xgemm = mk_zgemm(&z_prog, ZOP_N, ZOP_H, &s_v_scaled, &s_v, &x_buf, &ones_r)?; // X = Vs·V†

        // Hermitian Jacobi: [0]A [1]V [2]n [3]batch [4]init_v [5]nk [6]active [7]diag
        let mk_zjacobi = |a: &Buffer<Float2>,
                          v: &Buffer<Float2>,
                          init_v: i32,
                          act: &Buffer<i32>|
         -> Result<Kernel> {
            Kernel::builder()
                .program(&h_prog)
                .name("jacobi_hermitian_cyclic_global_batched")
                .queue(rt.queue().clone())
                .global_work_size(n_sys * wg_jacobi)
                .local_work_size(wg_jacobi)
                .arg(a)
                .arg(v)
                .arg(n as i32)
                .arg(n_sys as i32)
                .arg(init_v)
                .arg(nk as i32)
                .arg(act)
                .arg(&jacobi_diag)
                .build()
                .map_err(map_ocl_err)
        };
        let k_zjacobi = mk_zjacobi(&hp, &cp, 0, &active_r)?;
        let k_zjacobi_warm = mk_zjacobi(&hp, &c, 1, &active_r)?;
        let k_s_jacobi = mk_zjacobi(&s_work, &s_v, 0, &ones_r)?;

        // zextract_diagonal_batched: [0]n [1]batch [2]nk [3]a [4]diag [5]active — flat grid
        let total_diag = n * n_sys;
        let gws_diag = ((total_diag + 63) / 64) * 64;
        let k_zextract = Kernel::builder()
            .program(&z_prog)
            .name("zextract_diagonal_batched")
            .queue(rt.queue().clone())
            .global_work_size(gws_diag)
            .local_work_size(64)
            .arg(n as i32)
            .arg(n_sys as i32)
            .arg(nk as i32)
            .arg(&hp)
            .arg(&eig_diag)
            .arg(&active_r)
            .build()
            .map_err(map_ocl_err)?;

        // kpoint_occ_batched: [0]n [1]nk [2]n_rep [3]n_occ [4]kT [5]eig [6]kw
        //   [7]occ_w [8]mu [9]e_scal [10]le [11]red [12]red2 [13]lohi [14]active
        let wg_occ = 256usize;
        let k_kocc = Kernel::builder()
            .program(&z_prog)
            .name("kpoint_occ_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * wg_occ)
            .local_work_size(wg_occ)
            .arg(n as i32)
            .arg(nk as i32)
            .arg(n_rep as i32)
            .arg(0.0f32)
            .arg(0.0f32) // n_occ, kT — bind_solve_params
            .arg(&eig_diag)
            .arg(&kw_buf)
            .arg(&occ_w)
            .arg(&mu_out)
            .arg(&e_scal)
            .arg_local::<f32>(nk * n)
            .arg_local::<f64>(wg_occ)
            .arg_local::<f64>(wg_occ)
            .arg_local::<f64>(4)
            .arg(&active_r)
            .build()
            .map_err(map_ocl_err)?;

        // zsc_mulliken_batched: [0]n [1]n_atoms [2]batch [3]nk [4]C [5]SC
        //   [6]occ_w [7]orb_atom [8]qk [9]diag [10]active
        let wg_mull = n.max(n_atoms).min(256).max(1);
        let k_zmull = Kernel::builder()
            .program(&z_prog)
            .name("zsc_mulliken_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_sys * wg_mull)
            .local_work_size(wg_mull)
            .arg(n as i32)
            .arg(n_atoms as i32)
            .arg(n_sys as i32)
            .arg(nk as i32)
            .arg(&c)
            .arg(&sc)
            .arg(&occ_w)
            .arg(orb_atom_buf)
            .arg(&qk)
            .arg_local::<f32>(n)
            .arg(&active_r)
            .build()
            .map_err(map_ocl_err)?;

        // kpoint_qreduce_batched: [0]n_atoms [1]nk [2]n_rep [3]qk [4]q_new [5]active
        let k_qreduce = Kernel::builder()
            .program(&z_prog)
            .name("kpoint_qreduce_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * wg_rep)
            .local_work_size(wg_rep)
            .arg(n_atoms as i32)
            .arg(nk as i32)
            .arg(n_rep as i32)
            .arg(&qk)
            .arg(&q_new)
            .arg(&active_r)
            .build()
            .map_err(map_ocl_err)?;

        // zsnormalize_batched: [0]n [1]batch [2]nk [3]C [4]S [5]loc [6]active — per (sid,col)
        let wg_sn = 128usize;
        let k_zsnorm = Kernel::builder()
            .program(&z_prog)
            .name("zsnormalize_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_sys * n * wg_sn)
            .local_work_size(wg_sn)
            .arg(n as i32)
            .arg(n_sys as i32)
            .arg(nk as i32)
            .arg(&c)
            .arg(s_buf)
            .arg_local::<Float2>(n + wg_sn) // t_s[n] float2 + red[wg] floats (8B slots cover it)
            .arg(&active_r)
            .build()
            .map_err(map_ocl_err)?;

        // zscale_eigenvectors_batched: [0]n [1]batch [2]nk [3]A [4]V [5]Vs [6]lmin [7]active
        let k_s_scale = Kernel::builder()
            .program(&z_prog)
            .name("zscale_eigenvectors_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_sys * wg_rep)
            .local_work_size(wg_rep)
            .arg(n as i32)
            .arg(n_sys as i32)
            .arg(nk as i32)
            .arg(&s_work)
            .arg(&s_v)
            .arg(&s_v_scaled)
            .arg(&lambda_min)
            .arg(&ones_r)
            .build()
            .map_err(map_ocl_err)?;

        // ---- real DIIS/mix kernels reused with batch = n_rep ----
        // residual_and_mix_batched: [0]n_atoms [1]batch [2]alpha [3]q_new
        //   [4]q_old [5]q_mixed [6]rms [7]active [8]scratch
        let k_residual_mix = Kernel::builder()
            .program(&mat_prog)
            .name("residual_and_mix_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * wg_rep)
            .local_work_size(wg_rep)
            .arg(n_atoms as i32)
            .arg(n_rep as i32)
            .arg(0.3f32)
            .arg(&q_new)
            .arg(&q_gpu)
            .arg(&q_next)
            .arg(&rms)
            .arg(&active_r)
            .arg_local::<f32>(wg_rep)
            .arg(&wids)
            .build()
            .map_err(map_ocl_err)?;

        // commit_q_batched: [0]n_atoms [1]batch [2]q_next [3]q [4]active
        let k_commit = Kernel::builder()
            .program(&mat_prog)
            .name("commit_q_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * wg_rep)
            .local_work_size(wg_rep)
            .arg(n_atoms as i32)
            .arg(n_rep as i32)
            .arg(&q_next)
            .arg(&q_gpu)
            .arg(&active_r)
            .arg(&wids)
            .build()
            .map_err(map_ocl_err)?;

        // diis_step_batched: [0]n_atoms [1]batch [2]alpha [3]q_new [4]q_old [5]q0
        //   [6]q_next(→q_gpu in place) [7]dq_hist [8]r_hist [9]buf_idx [10]n_filled
        //   [11]coeffs [12]flag [13]reason [14]rms [15]active [16]rms_tol
        //   [17]scratch [18]lW(__local f64 QR columns) [19]work_ids
        let wg_diis = 256usize;
        let k_diis = Kernel::builder()
            .program(&diis_prog)
            .name("diis_step_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * wg_diis)
            .local_work_size(wg_diis)
            .arg(n_atoms as i32)
            .arg(n_rep as i32)
            .arg(0.3f32)
            .arg(&q_new)
            .arg(&q_gpu)
            .arg(q0_buf)
            .arg(&q_gpu)
            .arg(&diis_q_hist)
            .arg(&diis_r_hist)
            .arg(&diis_buf_idx)
            .arg(&diis_n_filled)
            .arg(&diis_coeffs)
            .arg(&diis_flag)
            .arg(&diis_reason)
            .arg(&rms)
            .arg(&active_r)
            .arg(0.0f32) // rms_tol — bind_mix_params
            .arg_local::<f32>(wg_diis)
            .arg_local::<f64>(diis_max_hist * n_atoms) // [18] lW — __local QR columns
            .arg(&wids)
            .build()
            .map_err(map_ocl_err)?;

        // zkpoint_energy_tail_batched: [0]n_atoms [1]n_rep [2]dq [3]q0 [4]v
        //   [5]e_scal [6]red [7]red2 [8]active
        let k_energy_tail = Kernel::builder()
            .program(&z_prog)
            .name("zkpoint_energy_tail_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * wg_rep)
            .local_work_size(wg_rep)
            .arg(n_atoms as i32)
            .arg(n_rep as i32)
            .arg(&dq)
            .arg(q0_buf)
            .arg(&v)
            .arg(&e_scal)
            .arg_local::<f64>(wg_rep)
            .arg_local::<f64>(wg_rep)
            .arg(&active_r)
            .build()
            .map_err(map_ocl_err)?;

        let mut plan = Self {
            n,
            n_atoms,
            n_rep,
            nk,
            n_sys,
            h_scc,
            temp,
            hp,
            cp,
            c,
            sc,
            x_buf,
            s_work,
            s_v,
            s_v_scaled,
            q_gpu,
            dq,
            v,
            q_new,
            q_next,
            qk,
            eig_diag,
            occ_w,
            mu_out,
            lambda_min,
            jacobi_diag,
            active_r,
            ones_r,
            rms,
            kw: kw_buf,
            e_scal,
            diis_q_hist,
            diis_r_hist,
            diis_buf_idx,
            diis_n_filled,
            diis_coeffs,
            diis_flag,
            diis_reason,
            diis_max_hist,
            rms_host: vec![0.0; n_rep],
            active_host: vec![1; n_rep],
            e_scal_host: vec![0.0; n_rep * 4],
            eig_diag_host: vec![0.0; n_sys * n],
            lambda_min_host: vec![0.0; n_sys],
            jacobi_diag_h: vec![0.0; n_sys * 4],
            k_zdq_v,
            k_zhscc,
            k_zxh,
            k_ztx,
            k_zxc,
            k_zxh_warm,
            k_ztx_warm,
            k_zsc,
            k_zjacobi,
            k_zjacobi_warm,
            k_zextract,
            k_kocc,
            k_zmull,
            k_qreduce,
            k_zsnorm,
            k_s_jacobi,
            k_s_scale,
            k_s_xgemm,
            k_diis,
            k_residual_mix,
            k_commit,
            k_energy_tail,
            n_occ: 0.0,
            kT: 0.0,
            rot_fp64,
            b_warm: false,
        };
        plan.set_geometry(rt, s_buf).map_err(|e| {
            DftbError::InvalidInput(format!("GpuPbcPlan::new initial X=S^{{-1/2}}: {e}"))
        })?;
        Ok(plan)
    }

    // ==================================================================
    // Per-solve scalar binding (never per-iteration)
    // ==================================================================

    /// Bind the per-solve scalars — n_occ (occupied-band equivalents per
    /// cell, i.e. N_e/2 for closed shell) and the current `self.kT`.
    /// Called once per solve entry, never per iteration.
    pub fn bind_solve_params(&mut self, n_occ: usize) -> Result<()> {
        let n_occ_f = n_occ as f32;
        self.k_kocc.set_arg(3u32, n_occ_f).map_err(map_ocl_err)?;
        self.k_kocc.set_arg(4u32, self.kT).map_err(map_ocl_err)?;
        self.n_occ = n_occ_f;
        Ok(())
    }

    /// Bind the mixer scalars — alpha + device convergence tolerance.
    pub fn bind_mix_params(&mut self, alpha: f32, rms_tol: f32) -> Result<()> {
        self.k_diis.set_arg(2u32, alpha).map_err(map_ocl_err)?;
        self.k_diis.set_arg(16u32, rms_tol).map_err(map_ocl_err)?;
        self.k_residual_mix
            .set_arg(2u32, alpha)
            .map_err(map_ocl_err)?;
        Ok(())
    }

    // ==================================================================
    // set_geometry — S(k) eigensolve → X = S^{-1/2}
    // ==================================================================

    /// Update the Löwdin transform for a new geometry: copy S→s_work,
    /// Hermitian Jacobi on all n_sys k-points, V·rsqrt(λ), X = Vs·V†.
    /// Full rebuild every call — v1 has no complex Newton reuse (the warm
    /// AO basis is also dropped: it is not S(k_new)-orthonormal).
    /// Fails loud on λ_min ≤ 1e-6 or a Jacobi stop ≠ 0.
    pub fn set_geometry(&mut self, rt: &mut GpuRuntime, s_buf: &Buffer<Float2>) -> Result<()> {
        self.b_warm = false; // warm basis is not S_new-orthonormal — cold next solve
        rt.copy_into(s_buf, &self.s_work, self.n_sys * self.n * self.n)
            .map_err(|e| DftbError::InvalidInput(format!("S→s_work copy for S^{{-1/2}}: {e}")))?;
        unsafe {
            self.k_s_jacobi.enq().map_err(map_ocl_err)?;
            self.k_s_scale.enq().map_err(map_ocl_err)?;
            self.k_s_xgemm.enq().map_err(map_ocl_err)?;
        }
        // Certification — once per geometry, not in a hot loop.
        rt.read_buffer(&self.lambda_min, &mut self.lambda_min_host)?;
        for (i, &l) in self.lambda_min_host.iter().enumerate() {
            if !l.is_finite() || l <= 1e-6 {
                return Err(DftbError::InvalidInput(format!(
                    "S^{{-1/2}}: overlap λ_min[{i}]={l} (non-finite or ≤1e-6) — fail loud, no rsqrt-clamp"
                )));
            }
        }
        self.check_jacobi(rt, &vec![1i32; self.n_rep])?;
        Ok(())
    }

    // ==================================================================
    // SCC step — enqueue-only body
    // ==================================================================

    /// Eigenproblem solve: h_scc → rotated hp (ε on Re diag), eigenvectors
    /// in c (warm, rotated in place) or cp (cold, ortho basis).
    fn eigh_solve(&mut self) -> Result<()> {
        if self.b_warm {
            unsafe {
                self.k_zxh_warm.enq().map_err(map_ocl_err)?; // temp = C†·H_scc
                self.k_ztx_warm.enq().map_err(map_ocl_err)?; // hp = temp·C
                self.k_zjacobi_warm.enq().map_err(map_ocl_err)?; // rotate c in place
            }
        } else {
            unsafe {
                self.k_zxh.enq().map_err(map_ocl_err)?; // temp = X†·H_scc
                self.k_ztx.enq().map_err(map_ocl_err)?; // hp = temp·X
                self.k_zjacobi.enq().map_err(map_ocl_err)?; // V=I → cp
            }
        }
        unsafe {
            self.k_zextract.enq().map_err(map_ocl_err)?;
        } // eig_diag = Re(diag hp)
        Ok(())
    }

    /// Post-Jacobi eigenvector production: cold path back-transforms c = X·cp
    /// and marks the basis warm; warm path S(k)-metric-renorms the in-place
    /// rotated c (f32 drift repair, same role as real snormalize).
    fn eigh_finish(&mut self) -> Result<()> {
        if self.b_warm {
            unsafe {
                self.k_zsnorm.enq().map_err(map_ocl_err)?;
            }
        } else {
            unsafe {
                self.k_zxc.enq().map_err(map_ocl_err)?;
            }
            self.b_warm = true;
        }
        Ok(())
    }

    /// Enqueue-only SCC iteration — NO host readback (chunked W4 contract).
    /// `diis_step_batched` clears `active_r[rep]` itself on rms < tol or
    /// nonfinite; all flat-system kernels see it via active[sid/nk].
    /// Requires `bind_solve_params` + `bind_mix_params` to have run.
    pub fn scc_step_diis_enq(&mut self, rt: &mut GpuRuntime) -> Result<()> {
        unsafe {
            self.k_zdq_v.enq().map_err(map_ocl_err)?;
        } // dq, V (rep grid)
        rt.prof_tick("pbc.dq_v");
        unsafe {
            self.k_zhscc.enq().map_err(map_ocl_err)?;
        } // H_scc (flat)
        rt.prof_tick("pbc.hscc");

        self.eigh_solve()?; // Jacobi (flat)
        rt.prof_tick("pbc.jacobi");

        unsafe {
            self.k_kocc.enq().map_err(map_ocl_err)?;
        } // shared-μ occ (rep grid)
        rt.prof_tick("pbc.kocc");

        self.eigh_finish()?; // warm renorm / cold X·cp
        rt.prof_tick("pbc.eigh_finish");

        unsafe {
            self.k_zsc.enq().map_err(map_ocl_err)?;
        } // sc = S·c
        unsafe {
            self.k_zmull.enq().map_err(map_ocl_err)?;
        } // qk (flat)
        unsafe {
            self.k_qreduce.enq().map_err(map_ocl_err)?;
        } // q_new (rep)
        rt.prof_tick("pbc.mulliken");

        unsafe {
            self.k_diis.enq().map_err(map_ocl_err)?;
        } // DIIS + commit in place
        rt.prof_tick("pbc.diis");
        Ok(())
    }

    /// Same pipeline ending in simple mixing (residual_and_mix + commit) —
    /// the non-DIIS reference path.
    pub fn scc_step_enq(&mut self, rt: &mut GpuRuntime) -> Result<()> {
        unsafe {
            self.k_zdq_v.enq().map_err(map_ocl_err)?;
        }
        unsafe {
            self.k_zhscc.enq().map_err(map_ocl_err)?;
        }
        self.eigh_solve()?;
        unsafe {
            self.k_kocc.enq().map_err(map_ocl_err)?;
        }
        self.eigh_finish()?;
        unsafe {
            self.k_zsc.enq().map_err(map_ocl_err)?;
        }
        unsafe {
            self.k_zmull.enq().map_err(map_ocl_err)?;
        }
        unsafe {
            self.k_qreduce.enq().map_err(map_ocl_err)?;
        }
        unsafe {
            self.k_residual_mix.enq().map_err(map_ocl_err)?;
        }
        unsafe {
            self.k_commit.enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Synchronous single step (reference/debug path): DIIS step + one
    /// rms readback. The chunked hot loop uses `scc_step_diis_enq`.
    pub fn scc_step_diis(
        &mut self,
        rt: &mut GpuRuntime,
        n_occ: usize,
        alpha: f32,
        rms_tol: f32,
    ) -> Result<f32> {
        self.bind_solve_params(n_occ)?;
        self.bind_mix_params(alpha, rms_tol)?;
        self.scc_step_diis_enq(rt)?;
        rt.read_buffer(&self.rms, &mut self.rms_host)?;
        max_finite_f32(&self.rms_host, "DIIS rms")
    }

    /// Chunk-end sync — reads `rms` + `active_r` into the host mirrors.
    /// One `queue.finish()` covers both reads.
    pub fn read_chunk_status(&mut self, rt: &GpuRuntime) -> Result<()> {
        rt.read_buffer(&self.rms, &mut self.rms_host)?;
        rt.read_buffer(&self.active_r, &mut self.active_host)?;
        max_finite_f32(&self.rms_host, "DIIS rms (chunk end)")?;
        Ok(())
    }

    /// Upload `active_host` to the device replica mask.
    pub fn set_active(&mut self, rt: &GpuRuntime) -> Result<()> {
        self.active_r
            .write(&self.active_host)
            .enq()
            .map_err(map_ocl_err)?;
        Ok(())
    }

    /// Mark every replica active — required before finalize/eval since the
    /// SCC loop's device-side clear may have left zeros.
    pub fn activate_all(&mut self, rt: &GpuRuntime) -> Result<()> {
        for f in self.active_host.iter_mut() {
            *f = 1;
        }
        self.set_active(rt)
    }

    /// Upload initial charges for a new solve (e.g. warm-start q).
    pub fn set_initial_charges(&mut self, rt: &GpuRuntime, init_q: &[f32]) -> Result<()> {
        if init_q.len() != self.n_rep * self.n_atoms {
            return Err(DftbError::InvalidInput(format!(
                "set_initial_charges: len {} != n_rep*n_atoms {}*{}",
                init_q.len(),
                self.n_rep,
                self.n_atoms
            )));
        }
        self.q_gpu.write(init_q).enq().map_err(map_ocl_err)?;
        Ok(())
    }

    /// Reset DIIS history (geometry change / fresh solve).
    pub fn reset_diis(&mut self, rt: &GpuRuntime) -> Result<()> {
        let zeros = vec![0i32; self.n_rep];
        self.diis_buf_idx.write(&zeros).enq().map_err(map_ocl_err)?;
        self.diis_n_filled
            .write(&zeros)
            .enq()
            .map_err(map_ocl_err)?;
        self.diis_flag.write(&zeros).enq().map_err(map_ocl_err)?;
        self.diis_reason.write(&zeros).enq().map_err(map_ocl_err)?;
        Ok(())
    }

    /// Read DIIS fallback counters — (fallback_count, last_reason) per replica.
    pub fn diis_status(
        &mut self,
        rt: &GpuRuntime,
        flag: &mut [i32],
        reason: &mut [i32],
    ) -> Result<()> {
        if flag.len() != self.n_rep || reason.len() != self.n_rep {
            return Err(DftbError::InvalidInput(format!(
                "diis_status: host len {}/{} != n_rep {}",
                flag.len(),
                reason.len(),
                self.n_rep
            )));
        }
        rt.read_buffer(&self.diis_flag, flag)?;
        rt.read_buffer(&self.diis_reason, reason)?;
        Ok(())
    }

    // ==================================================================
    // Finalize / energy
    // ==================================================================

    /// Unmixed electronic solve at the current q_gpu — D-equivalent state,
    /// occ_w, eigenvalues all consistent with the committed charges.
    /// Runs for EVERY replica (activate_all first).
    pub fn finalize(&mut self, rt: &mut GpuRuntime, n_occ: usize) -> Result<()> {
        self.activate_all(rt)?;
        self.bind_solve_params(n_occ)?;
        unsafe {
            self.k_zdq_v.enq().map_err(map_ocl_err)?;
        }
        unsafe {
            self.k_zhscc.enq().map_err(map_ocl_err)?;
        }
        self.eigh_solve()?;
        unsafe {
            self.k_kocc.enq().map_err(map_ocl_err)?;
        }
        self.eigh_finish()?;
        unsafe {
            self.k_zsc.enq().map_err(map_ocl_err)?;
        }
        unsafe {
            self.k_zmull.enq().map_err(map_ocl_err)?;
        }
        unsafe {
            self.k_qreduce.enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Band + SCC energy from the finalized state:
    ///   E = e_band + 2kT·mts − ½·Δq·V − q0·V   (per replica, f64)
    /// `kpoint_occ` already wrote e_band+mts; the tail kernel adds the two
    /// real charge dots — one launch, one e_scal readback.
    /// Does NOT re-solve; caller must `finalize` (or compute_energy) first.
    pub fn energy_from_state(&mut self, rt: &mut GpuRuntime) -> Result<Vec<f64>> {
        self.activate_all(rt)?; // energy is defined for frozen replicas too
        unsafe {
            self.k_energy_tail.enq().map_err(map_ocl_err)?;
        }
        rt.read_buffer(&self.e_scal, &mut self.e_scal_host)?;
        let kt = self.kT as f64;
        let mut e = vec![0.0f64; self.n_rep];
        for r in 0..self.n_rep {
            let (e_band, mts, dqv, q0v) = (
                self.e_scal_host[4 * r],
                self.e_scal_host[4 * r + 1],
                self.e_scal_host[4 * r + 2],
                self.e_scal_host[4 * r + 3],
            );
            e[r] = e_band + if self.kT > 0.0 { 2.0 * kt * mts } else { 0.0 } - 0.5 * dqv - q0v;
            if !e[r].is_finite() {
                return Err(DftbError::InvalidInput(format!(
                    "energy_from_state: E[{r}] non-finite band={e_band} mts={mts} dq·V={dqv} q0·V={q0v}"
                )));
            }
        }
        Ok(e)
    }

    /// Finalize + energy in one call (same contract as real compute_energy).
    pub fn compute_energy(&mut self, rt: &mut GpuRuntime, n_occ: usize) -> Result<Vec<f64>> {
        self.finalize(rt, n_occ)?;
        self.energy_from_state(rt)
    }

    // ==================================================================
    // Certification
    // ==================================================================

    /// Deferred Jacobi certification — read the diag buffer ONCE at solve
    /// end (never per iteration). `ran[r] != 0` marks replicas that launched
    /// at least one H-Jacobi this solve. Aggregates the flat (rep,k) records
    /// per replica: any non-zero stop marks the replica; stop=3 → Err
    /// (capacity guard bug), stop=4 or rel>1e-4 → not certified.
    /// Returns per-REPLICA `ok` flags.
    pub fn check_jacobi(&mut self, rt: &GpuRuntime, ran: &[i32]) -> Result<Vec<bool>> {
        let mut ok = vec![true; self.n_rep];
        rt.read_buffer(&self.jacobi_diag, &mut self.jacobi_diag_h)?;
        for r in 0..self.n_rep {
            if ran.get(r).copied().unwrap_or(0) == 0 {
                continue;
            }
            for k in 0..self.nk {
                let sid = r * self.nk + k;
                let stop = self.jacobi_diag_h[4 * sid + 2] as i32;
                if stop == 0 {
                    continue;
                }
                let (off, rel, nsw) = (
                    self.jacobi_diag_h[4 * sid],
                    self.jacobi_diag_h[4 * sid + 1],
                    self.jacobi_diag_h[4 * sid + 3],
                );
                eprintln!("[GpuPbcPlan] Jacobi rep {r} k {k}: stop={stop} (1=stall 2=maxsweeps 3=n>256 4=nonfinite) off={off:.3e} rel={rel:.3e} sweeps={nsw:.0}");
                if stop == 3 {
                    return Err(DftbError::InvalidInput(format!(
                        "Jacobi rep {r} k {k}: n>256 reached the kernel — host capacity guard failed (bug, not data)"
                    )));
                }
                if stop == 4 || rel > 1.0e-4 {
                    ok[r] = false;
                }
            }
        }
        Ok(ok)
    }

    /// Read back final committed charges `[n_rep*n_atoms]`.
    pub fn read_charges(&self, rt: &GpuRuntime) -> Result<Vec<f32>> {
        let mut q = vec![0.0f32; self.n_rep * self.n_atoms];
        rt.read_buffer(&self.q_gpu, &mut q)?;
        Ok(q)
    }

    /// Read back eigenvalues `[n_sys*n]` (sorted ascending per flat system).
    pub fn read_eigenvalues(&mut self, rt: &mut GpuRuntime) -> Result<Vec<f32>> {
        unsafe {
            self.k_zextract.enq().map_err(map_ocl_err)?;
        }
        rt.read_buffer(&self.eig_diag, &mut self.eig_diag_host)?;
        let mut out = vec![0.0f32; self.n_sys * self.n];
        for s in 0..self.n_sys {
            let mut e: Vec<f32> = self.eig_diag_host[s * self.n..(s + 1) * self.n].to_vec();
            e.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[s * self.n..(s + 1) * self.n].copy_from_slice(&e);
        }
        Ok(out)
    }

    /// (n, n_atoms, n_rep, nk)
    pub fn config(&self) -> (usize, usize, usize, usize) {
        (self.n, self.n_atoms, self.n_rep, self.nk)
    }

    /// Current warm-basis flag (diagnostics).
    pub fn is_warm(&self) -> bool {
        self.b_warm
    }
}
