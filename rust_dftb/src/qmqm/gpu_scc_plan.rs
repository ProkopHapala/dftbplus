//! Persistent SCC plan: pre-built kernels and reusable scratch buffers.
//!
//! Phase 0c of the H-Bond manifest: eliminate per-iteration `Kernel::builder()`
//! calls and per-solve buffer allocations. The plan is created once for a fixed
//! `(n, n_atoms, batch)` configuration and reused across SCC solves and
//! relaxation steps.
//!
//! # Architecture
//!
//! All OpenCL `Kernel` objects are built once at construction time with the
//! plan's own scratch buffers as initial arguments. Per-call methods swap
//! buffer args via `Kernel::set_arg` and enqueue — no `Kernel::builder()` in
//! the hot loop.
//!
//! Scratch buffers are allocated once at construction and reused. The plan
//! owns all working buffers; the caller only provides the persistent inputs
//! (H0, S, gamma, q0, orb_atom) and receives the result.

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::{Buffer, Kernel, Program};

// ---- Matrix kernel source (same as gpu_matrix.rs) ----
const MATRIX_KERNEL_TEMPLATE: &str = include_str!("gpu_matrix_ops.cl");

// ---- Eigen kernel source and helpers (same as gpu_eigen.rs) ----
const GPU_EIGEN_TEMPLATE: &str = include_str!("gpu_eigen.cl");
const GPU_TILED_JACOBI_TEMPLATE: &str = include_str!("gpu_tiled_jacobi.cl");
const EIGEN_MAX_SWEEPS: usize = 20;
const EIGEN_PPG: usize = 8;
const TILED_MAX_SWEEPS: usize = 100;

fn max_finite_f32(xs: &[f32], ctx: &str) -> Result<f32> {
    let mut m = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if !x.is_finite() {
            return Err(DftbError::InvalidInput(format!("{ctx}[{i}]={x} non-finite")));
        }
        if x > m { m = x; }
    }
    if !m.is_finite() {
        return Err(DftbError::InvalidInput(format!("{ctx}: no finite values (len={})", xs.len())));
    }
    Ok(m)
}

fn eigen_spec_params(n: usize) -> (usize, usize, usize, usize, usize) {
    if n == 0 { return (0, 1, 0, 0, 32); }
    let jn = if n % 2 == 0 { n } else { n + 1 };
    let jld = jn + 1;
    let jpair = jn / 2;
    let jround = jn - 1;
    let active = jpair * EIGEN_PPG;
    let wg = active.next_power_of_two().max(32).min(1024);
    (jn, jld, jpair, jround, wg)
}

fn eigen_render_source(n: usize) -> String {
    let (jn, jld, jpair, jround, wg) = eigen_spec_params(n);
    GPU_EIGEN_TEMPLATE
        .replace("#define JN 8", &format!("#define JN {}", jn))
        .replace("#define JLD 9", &format!("#define JLD {}", jld))
        .replace("#define JPAIR 4", &format!("#define JPAIR {}", jpair))
        .replace("#define JROUND 7", &format!("#define JROUND {}", jround))
        .replace("#define WG 32", &format!("#define WG {}", wg))
        .replace("#define PPG 8", &format!("#define PPG {}", EIGEN_PPG))
        .replace("#define MAX_SWEEPS 20", &format!("#define MAX_SWEEPS {}", EIGEN_MAX_SWEEPS))
}

fn tiled_render_source(b: usize, wg: usize) -> String {
    let pb = 2 * b;
    let pld = pb + 1;
    GPU_TILED_JACOBI_TEMPLATE
        .replace("#define B 32", &format!("#define B {}", b))
        .replace("#define PB 64", &format!("#define PB {}", pb))
        .replace("#define PLD 65", &format!("#define PLD {}", pld))
        .replace("#define WG 256", &format!("#define WG {}", wg))
        .replace("#define STRIP_R 32", &format!("#define STRIP_R {}", b))
        .replace("#define MAX_SWEEPS 50", &format!("#define MAX_SWEEPS {}", TILED_MAX_SWEEPS))
}

fn full_local_wg(n: usize) -> usize {
    let n2 = n * n;
    if n2 <= 256 { n2.max(1) } else { 256 }
}

fn render_source_full_local(n: usize, wg: usize) -> String {
    MATRIX_KERNEL_TEMPLATE
        .replace("#define FL_NORB 64", &format!("#define FL_NORB {}", n))
        .replace("#define FL_WG 256", &format!("#define FL_WG {}", wg))
}

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
    pub x_buf: Buffer<f32>,       // Löwdin transform S^{-1/2} [batch*nn]
    s_work: Buffer<f32>,          // Jacobi A workspace for S (copy of S; destroyed)
    s_v: Buffer<f32>,             // Jacobi V for S
    s_v_scaled: Buffer<f32>,      // V·rsqrt(λ) (N>64 path; unused N≤64)
    lambda_min: Buffer<f32>,      // [batch] λ_min(S)

    // R9b: GPU-side DIIS history buffers
    pub diis_q_hist: Buffer<f32>,     // [batch*max_hist*n_atoms] q_in ring buffer
    pub diis_r_hist: Buffer<f32>,     // [batch*max_hist*n_atoms] residual ring buffer
    pub diis_buf_idx: Buffer<i32>,    // [batch] ring buffer write position
    pub diis_n_filled: Buffer<i32>,   // [batch] number of valid entries
    pub diis_b_mat: Buffer<f32>,      // [batch*(max_hist+1)*(max_hist+1)] B matrix
    pub diis_rhs: Buffer<f32>,        // [batch*(max_hist+1)] RHS
    pub diis_coeffs: Buffer<f32>,     // [batch*max_hist] coefficients
    pub diis_max_hist: usize,          // max history length (typically 10)

    // Host staging buffers (reused, not re-allocated)
    pub eig_diag_host: Vec<f32>,  // [batch*n]
    pub mask_host: Vec<i32>,      // [batch*n]
    pub rms_host: Vec<f32>,       // [batch]
    lambda_min_host: Vec<f32>,    // [batch] overlap λ_min
    scratch_x: Vec<f32>,          // [batch*nn] Löwdin Newton (set_geometry only)
    scratch_s: Vec<f32>,          // [batch*nn]
    work_a: Vec<f64>,             // [n*n] f64 island for XᵀSX
    work_b: Vec<f64>,
    work_c: Vec<f64>,

    // Pre-built kernels (R8: no Kernel::builder() in hot loops)
    // SCC step kernels:
    k_delta_q: Kernel,        // delta_q_batched
    k_gamma: Kernel,          // gamma_matvec_batched
    k_h_scc: Kernel,          // h_scc_update_batched
    k_matmul_xh: Kernel,      // X · H_scc → temp
    k_matmul_tx: Kernel,      // temp · X → hp
    k_matmul_xc: Kernel,      // X · cp → c
    k_jacobi: Kernel,         // jacobi (full-local or tiled)
    k_extract_diag: Kernel,   // extract_diagonal_batched
    k_select_occ: Kernel,     // select_occupation_batched (R9: GPU-side occ selection)
    k_density: Kernel,         // build_density_masked_batched
    k_mulliken: Kernel,        // mulliken_charges_batched
    k_residual_mix: Kernel,    // residual_and_mix_batched (simple mixing fallback)
    k_diis: Kernel,            // diis_step_batched (R9b: GPU-side DIIS)
    // Energy kernels:
    k_frobenius_trace: Kernel, // frobenius_trace_batched
    k_dot: Kernel,             // dot_batched

    // Matmul arg layout: for N≤64, buffer args are at [2,3,4]; for N>64, at [6,7,8].
    matmul_buf_base: u32,

    // n_occ for the occupation kernel (set once per solve, not per iteration)
    n_occ: usize,

    // S^{-1/2} kernels — built once; set_geometry only copies S and enqueues.
    k_s_jacobi: Kernel,
    k_s_invsqrt: Option<Kernel>,      // N≤64: build_inv_sqrt_from_eig
    k_s_scale: Option<Kernel>,        // N>64: scale_eigenvectors_batched
    k_s_xgemm: Option<Kernel>,        // N>64: X = V_scaled · V^T

    // R5: Repulsive spline energy (optional — set via set_repulsive_splines)
    k_rep_energy: Option<Kernel>,         // repulsive_energy_batched
    rep_coords: Option<Buffer<f32>>,        // [batch*n_atoms*3] coordinates (Bohr)
    rep_species_idx: Option<Buffer<i32>>,    // [batch*n_atoms] species index per atom
    rep_spline_offsets: Option<Buffer<i32>>, // [n_species*n_species] offset into spline_data
    rep_spline_data: Option<Buffer<f32>>,    // flat buffer with all spline coefficients
    rep_e_rep: Option<Buffer<f32>>,           // [batch] repulsive energy output
    rep_n_species: usize,
}

impl GpuSccPlan {
    /// Create a plan for the given configuration. Allocates all scratch
    /// buffers, builds all kernels, and computes the initial Löwdin transform.
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
        let nn = n * n;

        // Persistent S^{-1/2} workspace (filled by set_geometry; no alloc there)
        let s_work = rt.zero_buffer::<f32>(batch * nn)?;
        let s_v = rt.zero_buffer::<f32>(batch * nn)?;
        let s_v_scaled = rt.zero_buffer::<f32>(batch * nn)?;
        let x_buf = rt.zero_buffer::<f32>(batch * nn)?;
        let lambda_min = rt.zero_buffer::<f32>(batch)?;

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

        // R9b: DIIS history. Cap at n_atoms — more vectors than the charge space is rank-deficient
        // (H2O: 10 hist in 3-atom q → pivot fallback). AT n_atoms=30 still uses 10.
        let diis_max_hist = n_atoms.min(10).max(1);
        let diis_q_hist = rt.zero_buffer::<f32>(batch * diis_max_hist * n_atoms)?;
        let diis_r_hist = rt.zero_buffer::<f32>(batch * diis_max_hist * n_atoms)?;
        let diis_buf_idx = rt.zero_buffer::<i32>(batch)?;
        let diis_n_filled = rt.zero_buffer::<i32>(batch)?;
        let diis_b_mat = rt.zero_buffer::<f32>(batch * (diis_max_hist + 1) * (diis_max_hist + 1))?;
        let diis_rhs = rt.zero_buffer::<f32>(batch * (diis_max_hist + 1))?;
        let diis_coeffs = rt.zero_buffer::<f32>(batch * diis_max_hist)?;
        eprintln!("[GpuSccPlan] DIIS hist={diis_max_hist} (min(10,n_atoms={n_atoms})) N={n} batch={batch}");

        // ---- Build all kernels once ----
        // Matrix ops program (shared by most kernels)
        let mat_prog = rt.build_program(MATRIX_KERNEL_TEMPLATE)?;

        // 1. delta_q_batched: args [0]=n_atoms, [1]=batch, [2]=q, [3]=q0, [4]=dq
        let wg_dq = n_atoms.min(256).max(1);
        let k_delta_q = Kernel::builder()
            .program(&mat_prog).name("delta_q_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_dq).local_work_size(wg_dq)
            .arg(n_atoms as i32).arg(batch as i32)
            .arg(&q_gpu).arg(&q_gpu).arg(&dq)  // dummy q/q0, set per-call
            .build().map_err(map_ocl_err)?;

        // 2. gamma_matvec_batched: args [0]=n_atoms, [1]=batch, [2]=g, [3]=dq, [4]=v
        let wg_gamma = n_atoms.min(rt.caps().max_work_group_size).max(1);
        let k_gamma = Kernel::builder()
            .program(&mat_prog).name("gamma_matvec_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_gamma).local_work_size(wg_gamma)
            .arg(n_atoms as i32).arg(batch as i32)
            .arg(&q_gpu).arg(&dq).arg(&v)  // dummy g, set per-call
            .build().map_err(map_ocl_err)?;

        // 3. h_scc_update_batched: args [0]=n, [1]=n_atoms, [2]=batch, [3]=h0, [4]=s, [5]=v, [6]=orb_atom, [7]=h, [8]=local(n_atoms)
        let wg_hsc = 256usize;
        let k_h_scc = Kernel::builder()
            .program(&mat_prog).name("h_scc_update_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_hsc).local_work_size(wg_hsc)
            .arg(n as i32).arg(n_atoms as i32).arg(batch as i32)
            .arg(&h_scc).arg(&h_scc).arg(&v).arg(&occ_mask).arg(&h_scc)  // dummy h0/s/orb_atom, set per-call
            .arg_local::<f32>(n_atoms)
            .build().map_err(map_ocl_err)?;

        // 4-6. Three matmul kernels (X·H_scc→temp, temp·X→hp, X·cp→c)
        let (matmul_buf_base, k_matmul_xh, k_matmul_tx, k_matmul_xc) =
            build_matmul_kernels(rt, &mat_prog, n, batch, &x_buf, &h_scc, &temp, &hp, &cp, &c)?;

        // 7. Jacobi eigensolver
        let k_jacobi = build_jacobi_kernel(rt, n, batch, &hp, &cp)?;

        // 8. extract_diagonal_batched: args [0]=n, [1]=batch, [2]=a, [3]=diag
        let total_diag = n * batch;
        let gws_diag = ((total_diag + 63) / 64) * 64;
        let k_extract_diag = Kernel::builder()
            .program(&mat_prog).name("extract_diagonal_batched").queue(rt.queue().clone())
            .global_work_size(gws_diag).local_work_size(64)
            .arg(n as i32).arg(batch as i32)
            .arg(&hp).arg(&eig_diag)
            .build().map_err(map_ocl_err)?;

        // 8b. select_occupation_batched: GPU-side bitonic sort + occupation marking (R9)
        //     args [0]=n, [1]=n_occ, [2]=batch, [3]=eig_diag, [4]=occ_mask
        //     OCC_MAX_N = next power of 2 ≥ n, specialized via text substitution.
        let occ_max_n = n.next_power_of_two().max(2);
        let occ_source = MATRIX_KERNEL_TEMPLATE
            .replace("#define OCC_MAX_N 128", &format!("#define OCC_MAX_N {}", occ_max_n));
        let occ_prog = rt.build_program(&occ_source)?;
        let occ_wg = occ_max_n.min(256).max(1);
        let k_select_occ = Kernel::builder()
            .program(&occ_prog).name("select_occupation_batched").queue(rt.queue().clone())
            .global_work_size(batch * occ_wg).local_work_size(occ_wg)
            .arg(n as i32).arg(0i32).arg(batch as i32)  // n_occ=0 dummy, set per-solve
            .arg(&eig_diag).arg(&occ_mask)
            .build().map_err(map_ocl_err)?;

        // 9. build_density_masked_batched: args [0]=n [1]=batch [2]=c [3]=occ_mask [4]=out [5]=use_eig [6]=eig
        //    use_eig=0 → D (s_k=1); use_eig=1 → W (s_k=ε_k). Same kernel.
        let wg_den = (n * n).min(256).max(1);
        let k_density = Kernel::builder()
            .program(&mat_prog).name("build_density_masked_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_den).local_work_size(wg_den)
            .arg(n as i32).arg(batch as i32)
            .arg(&c).arg(&occ_mask).arg(&d).arg(0i32).arg(&eig_diag)
            .build().map_err(map_ocl_err)?;

        // 10. mulliken_charges_batched: args [0]=n, [1]=n_atoms, [2]=batch, [3]=d, [4]=s, [5]=orb_atom, [6]=q, [7]=local(n)
        let wg_mull = n.max(n_atoms).min(256).max(1);
        let k_mulliken = Kernel::builder()
            .program(&mat_prog).name("mulliken_charges_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_mull).local_work_size(wg_mull)
            .arg(n as i32).arg(n_atoms as i32).arg(batch as i32)
            .arg(&d).arg(&h_scc).arg(&occ_mask).arg(&q_new)  // dummy s/orb_atom, set per-call
            .arg_local::<f32>(n)
            .build().map_err(map_ocl_err)?;

        // 11. residual_and_mix_batched: args [0]=n_atoms, [1]=batch, [2]=alpha, [3]=q_new, [4]=q_old, [5]=q_mixed, [6]=rms, [7]=local(wg)
        let wg_res = 256usize;
        let k_residual_mix = Kernel::builder()
            .program(&mat_prog).name("residual_and_mix_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_res).local_work_size(wg_res)
            .arg(n_atoms as i32).arg(batch as i32).arg(0.3f32)  // alpha, set per-call
            .arg(&q_new).arg(&q_gpu).arg(&q_gpu).arg(&rms)
            .arg_local::<f32>(wg_res)
            .build().map_err(map_ocl_err)?;

        // 11b. diis_step_batched: specialize DIIS_MAX_HIST (private Bd[NP1²] is compile-time).
        // Local scratch is scratch[lid] for lid < lsz — must be ≥ workgroup, not n_atoms.
        let wg_diis = 256usize;
        let diis_src = MATRIX_KERNEL_TEMPLATE.replace(
            "#ifndef DIIS_MAX_HIST\n#define DIIS_MAX_HIST 10\n#endif",
            &format!("#define DIIS_MAX_HIST {diis_max_hist}"),
        );
        if !diis_src.contains(&format!("#define DIIS_MAX_HIST {diis_max_hist}")) {
            return Err(DftbError::InvalidInput(format!("DIIS_MAX_HIST specialize failed hist={diis_max_hist}")));
        }
        if diis_max_hist != 10 && diis_src.contains("#define DIIS_MAX_HIST 10") {
            return Err(DftbError::InvalidInput("DIIS_MAX_HIST specialize left default 10".into()));
        }
        let diis_prog = rt.build_program(&diis_src)?;
        let k_diis = Kernel::builder()
            .program(&diis_prog).name("diis_step_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_diis).local_work_size(wg_diis)
            .arg(n_atoms as i32).arg(batch as i32).arg(0.3f32)
            .arg(&q_new).arg(&q_gpu)
            .arg(&diis_q_hist).arg(&diis_r_hist)
            .arg(&diis_buf_idx).arg(&diis_n_filled)
            .arg(&diis_b_mat).arg(&diis_rhs).arg(&diis_coeffs)
            .arg(&rms)
            .arg_local::<f32>(wg_diis)
            .build().map_err(map_ocl_err)?;

        // 12. frobenius_trace_batched: args [0]=n, [1]=batch, [2]=a, [3]=b, [4]=tr, [5]=local(wg)
        let wg_frob = 256usize;
        let k_frobenius_trace = Kernel::builder()
            .program(&mat_prog).name("frobenius_trace_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_frob).local_work_size(wg_frob)
            .arg(n as i32).arg(batch as i32)
            .arg(&d).arg(&h_scc).arg(&tr)  // dummy b, set per-call
            .arg_local::<f32>(wg_frob)
            .build().map_err(map_ocl_err)?;

        // 13. dot_batched: args [0]=n, [1]=batch, [2]=x, [3]=y, [4]=dot, [5]=local(wg)
        let wg_dot = 256usize;
        let k_dot = Kernel::builder()
            .program(&mat_prog).name("dot_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_dot).local_work_size(wg_dot)
            .arg(n_atoms as i32).arg(batch as i32)  // n_atoms for dot (charge vector length)
            .arg(&dq).arg(&v).arg(&dot)
            .arg_local::<f32>(wg_dot)
            .build().map_err(map_ocl_err)?;

        let k_s_jacobi = build_jacobi_kernel(rt, n, batch, &s_work, &s_v)?;
        let (k_s_invsqrt, k_s_scale, k_s_xgemm) = build_sinv_kernels(
            rt, &mat_prog, n, batch, &s_work, &s_v, &s_v_scaled, &x_buf, &lambda_min,
        )?;
        enqueue_sinv(rt, s_buf, &s_work, batch * nn, &k_s_jacobi, k_s_invsqrt.as_ref(), k_s_scale.as_ref(), k_s_xgemm.as_ref())?;
        let mut lambda_min_host = vec![0.0f32; batch];
        rt.read_buffer(&lambda_min, &mut lambda_min_host)?;
        check_overlap_lambda(&lambda_min_host)?;
        let mut scratch_x = vec![0.0f32; batch * nn];
        let mut scratch_s = vec![0.0f32; batch * nn];
        let mut work_a = vec![0.0f64; nn];
        let mut work_b = vec![0.0f64; nn];
        let mut work_c = vec![0.0f64; nn];
        repair_lowdin_x(rt, &x_buf, s_buf, n, batch, &mut scratch_x, &mut scratch_s, &mut work_a, &mut work_b, &mut work_c)?;

        Ok(Self {
            n, n_atoms, batch,
            q_gpu, dq, v, h_scc, temp, hp, cp, c, d, q_new, tr, dot, rms, occ_mask, eig_diag,
            x_buf, s_work, s_v, s_v_scaled, lambda_min,
            diis_q_hist, diis_r_hist, diis_buf_idx, diis_n_filled,
            diis_b_mat, diis_rhs, diis_coeffs, diis_max_hist,
            eig_diag_host: vec![0.0; batch * n],
            mask_host: vec![0; batch * n],
            rms_host: vec![0.0; batch],
            lambda_min_host,
            scratch_x, scratch_s, work_a, work_b, work_c,
            k_delta_q, k_gamma, k_h_scc,
            k_matmul_xh, k_matmul_tx, k_matmul_xc,
            k_jacobi, k_extract_diag, k_select_occ, k_density, k_mulliken, k_residual_mix,
            k_diis, k_frobenius_trace, k_dot,
            matmul_buf_base,
            n_occ: 0,
            k_s_jacobi, k_s_invsqrt, k_s_scale, k_s_xgemm,
            k_rep_energy: None,
            rep_coords: None,
            rep_species_idx: None,
            rep_spline_offsets: None,
            rep_spline_data: None,
            rep_e_rep: None,
            rep_n_species: 0,
        })
    }

    /// Update the Löwdin transform for a new geometry. Call this when the
    /// overlap matrix changes (e.g. between relaxation steps).
    /// Copies S into a workspace and overwrites `x_buf` in place — no Buffer/Kernel alloc.
    pub fn set_geometry(&mut self, rt: &mut GpuRuntime, s_buf: &Buffer<f32>) -> Result<()> {
        enqueue_sinv(
            rt, s_buf, &self.s_work, self.batch * self.n * self.n,
            &self.k_s_jacobi, self.k_s_invsqrt.as_ref(), self.k_s_scale.as_ref(), self.k_s_xgemm.as_ref(),
        )?;
        rt.read_buffer(&self.lambda_min, &mut self.lambda_min_host)?;
        check_overlap_lambda(&self.lambda_min_host)?;
        repair_lowdin_x(
            rt, &self.x_buf, s_buf, self.n, self.batch,
            &mut self.scratch_x, &mut self.scratch_s, &mut self.work_a, &mut self.work_b, &mut self.work_c,
        )
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

    /// Set the repulsive spline data for repulsive energy evaluation (R5).
    ///
    /// Arguments:
    /// - `coords` — `[batch*n_atoms*3]` atom coordinates in Bohr
    /// - `species_idx` — `[batch*n_atoms]` species index per atom (0-based)
    /// - `spline_offsets` — `[n_species*n_species]` offset into spline_data
    ///   (-1 = no spline for that pair)
    /// - `spline_data` — flat buffer with all spline coefficients
    /// - `n_species` — number of unique species
    /// - `max_intervals` — max number of spline intervals across all pairs
    ///
    /// Spline data layout per pair (at offset spline_offsets[p]):
    ///   [0]   n_intervals (i32 as f32 bit pattern)
    ///   [1]   cutoff (f32)
    ///   [2-4] exp_coeffs (a, b, c) (3 f32)
    ///   [5..5+max_intervals]  x_start (max_intervals f32)
    ///   [5+max_intervals..5+max_intervals+(max_intervals-1)*4]  sp_coeffs
    ///   [5+max_intervals+(max_intervals-1)*4..5+max_intervals+(max_intervals-1)*4+6]  sp_last_coeffs
    pub fn set_repulsive_splines(
        &mut self,
        rt: &mut GpuRuntime,
        coords: &[f32],
        species_idx: &[i32],
        spline_offsets: &[i32],
        spline_data: &[f32],
        n_species: usize,
        max_intervals: usize,
    ) -> Result<()> {
        let batch = self.batch;
        let n_atoms = self.n_atoms;

        // Upload buffers
        let rep_coords = rt.buffer_from_slice(coords)?;
        let rep_species_idx = rt.buffer_from_slice(species_idx)?;
        let rep_spline_offsets = rt.buffer_from_slice(spline_offsets)?;
        let rep_spline_data = rt.buffer_from_slice(spline_data)?;
        let rep_e_rep = rt.zero_buffer::<f32>(batch)?;

        // Build kernel with REP_MAX_INTERVALS specialization
        let source = MATRIX_KERNEL_TEMPLATE
            .replace("#define REP_MAX_INTERVALS 30", &format!("#define REP_MAX_INTERVALS {}", max_intervals));
        let prog = rt.build_program(&source)?;
        let wg = 256usize;
        let k_rep_energy = Kernel::builder()
            .program(&prog).name("repulsive_energy_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(n_atoms as i32).arg(batch as i32)
            .arg(&rep_coords).arg(&rep_species_idx)
            .arg(&rep_spline_offsets).arg(n_species as i32)
            .arg(&rep_spline_data).arg(&rep_e_rep)
            .build().map_err(map_ocl_err)?;

        self.k_rep_energy = Some(k_rep_energy);
        self.rep_coords = Some(rep_coords);
        self.rep_species_idx = Some(rep_species_idx);
        self.rep_spline_offsets = Some(rep_spline_offsets);
        self.rep_spline_data = Some(rep_spline_data);
        self.rep_e_rep = Some(rep_e_rep);
        self.rep_n_species = n_species;
        Ok(())
    }

    /// Update coordinates for repulsive energy evaluation (call when
    /// geometry changes, before compute_energy).
    pub fn set_repulsive_coords(&mut self, rt: &GpuRuntime, coords: &[f32]) -> Result<()> {
        if let Some(ref buf) = self.rep_coords {
            if coords.len() != self.batch * self.n_atoms * 3 {
                return Err(DftbError::InvalidInput(format!(
                    "set_repulsive_coords: len {} != batch*n_atoms*3 {}*{}*3 = {}",
                    coords.len(), self.batch, self.n_atoms, self.batch * self.n_atoms * 3
                )));
            }
            buf.write(coords).enq().map_err(map_ocl_err)?;
        }
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
        let b = self.matmul_buf_base;

        // 1. Δq = q − q0
        self.k_delta_q.set_arg(2u32, &self.q_gpu).map_err(map_ocl_err)?;
        self.k_delta_q.set_arg(3u32, q0_buf).map_err(map_ocl_err)?;
        self.k_delta_q.set_arg(4u32, &self.dq).map_err(map_ocl_err)?;
        unsafe { self.k_delta_q.enq().map_err(map_ocl_err)?; }

        // 2. V = G · Δq
        self.k_gamma.set_arg(2u32, g_buf).map_err(map_ocl_err)?;
        self.k_gamma.set_arg(3u32, &self.dq).map_err(map_ocl_err)?;
        self.k_gamma.set_arg(4u32, &self.v).map_err(map_ocl_err)?;
        unsafe { self.k_gamma.enq().map_err(map_ocl_err)?; }

        // 3. H_scc = H0 + 0.5·S·(V_i + V_j)
        self.k_h_scc.set_arg(3u32, h0_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(5u32, &self.v).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(6u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(7u32, &self.h_scc).map_err(map_ocl_err)?;
        unsafe { self.k_h_scc.enq().map_err(map_ocl_err)?; }

        // 4. H' = X · H_scc · X (2 GEMMs)
        set_matmul_args(&self.k_matmul_xh, b, &self.x_buf, &self.h_scc, &self.temp)?;
        unsafe { self.k_matmul_xh.enq().map_err(map_ocl_err)?; }
        set_matmul_args(&self.k_matmul_tx, b, &self.temp, &self.x_buf, &self.hp)?;
        unsafe { self.k_matmul_tx.enq().map_err(map_ocl_err)?; }

        // 5. Jacobi(H') → eigenvalues on diag(hp), eigenvectors in cp
        self.k_jacobi.set_arg(0u32, &self.hp).map_err(map_ocl_err)?;
        self.k_jacobi.set_arg(1u32, &self.cp).map_err(map_ocl_err)?;
        unsafe { self.k_jacobi.enq().map_err(map_ocl_err)?; }

        // 6. Extract diag on GPU → GPU sort + occ_mask (R9: no host roundtrip)
        self.k_extract_diag.set_arg(2u32, &self.hp).map_err(map_ocl_err)?;
        self.k_extract_diag.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        unsafe { self.k_extract_diag.enq().map_err(map_ocl_err)?; }
        // GPU-side bitonic sort + occupation marking
        self.k_select_occ.set_arg(1u32, n_occ as i32).map_err(map_ocl_err)?;
        self.k_select_occ.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        self.k_select_occ.set_arg(4u32, &self.occ_mask).map_err(map_ocl_err)?;
        unsafe { self.k_select_occ.enq().map_err(map_ocl_err)?; }

        // 7. C = X · C'
        set_matmul_args(&self.k_matmul_xc, b, &self.x_buf, &self.cp, &self.c)?;
        unsafe { self.k_matmul_xc.enq().map_err(map_ocl_err)?; }

        // 8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T
        self.k_density.set_arg(2u32, &self.c).map_err(map_ocl_err)?;
        self.k_density.set_arg(3u32, &self.occ_mask).map_err(map_ocl_err)?;
        self.k_density.set_arg(4u32, &self.d).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, 0i32).map_err(map_ocl_err)?;
        unsafe { self.k_density.enq().map_err(map_ocl_err)?; }

        // 9. q_new = Mulliken(D, S)
        self.k_mulliken.set_arg(3u32, &self.d).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(5u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(6u32, &self.q_new).map_err(map_ocl_err)?;
        unsafe { self.k_mulliken.enq().map_err(map_ocl_err)?; }

        // 10. residual + mix
        self.k_residual_mix.set_arg(2u32, alpha).map_err(map_ocl_err)?;
        self.k_residual_mix.set_arg(3u32, &self.q_new).map_err(map_ocl_err)?;
        self.k_residual_mix.set_arg(4u32, &self.q_gpu).map_err(map_ocl_err)?;
        self.k_residual_mix.set_arg(5u32, &self.q_gpu).map_err(map_ocl_err)?;
        self.k_residual_mix.set_arg(6u32, &self.rms).map_err(map_ocl_err)?;
        unsafe { self.k_residual_mix.enq().map_err(map_ocl_err)?; }

        // Read RMS and return max
        rt.read_buffer(&self.rms, &mut self.rms_host)?;
        max_finite_f32(&self.rms_host, "SCC rms")
    }

    /// One SCC step with GPU DIIS. Returns max residual RMS (√(Σres²/n_atoms)).
    ///
    /// This is the same as `scc_step` but uses `diis_step_batched` instead
    /// of simple mixing. The DIIS kernel maintains a per-system ring buffer
    /// of q_in and residual history, builds the B matrix, solves the small
    /// DIIS linear system, and mixes on the GPU. Only the RMS scalar is
    /// read back to the host for convergence inspection.
    pub fn scc_step_diis(
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
        let b = self.matmul_buf_base;

        // Steps 1-9: same as scc_step (Δq → V → H_scc → H' → Jacobi → occ → C → D → q_new)
        self.k_delta_q.set_arg(2u32, &self.q_gpu).map_err(map_ocl_err)?;
        self.k_delta_q.set_arg(3u32, q0_buf).map_err(map_ocl_err)?;
        self.k_delta_q.set_arg(4u32, &self.dq).map_err(map_ocl_err)?;
        unsafe { self.k_delta_q.enq().map_err(map_ocl_err)?; }

        self.k_gamma.set_arg(2u32, g_buf).map_err(map_ocl_err)?;
        self.k_gamma.set_arg(3u32, &self.dq).map_err(map_ocl_err)?;
        self.k_gamma.set_arg(4u32, &self.v).map_err(map_ocl_err)?;
        unsafe { self.k_gamma.enq().map_err(map_ocl_err)?; }

        self.k_h_scc.set_arg(3u32, h0_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(5u32, &self.v).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(6u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(7u32, &self.h_scc).map_err(map_ocl_err)?;
        unsafe { self.k_h_scc.enq().map_err(map_ocl_err)?; }

        set_matmul_args(&self.k_matmul_xh, b, &self.x_buf, &self.h_scc, &self.temp)?;
        unsafe { self.k_matmul_xh.enq().map_err(map_ocl_err)?; }
        set_matmul_args(&self.k_matmul_tx, b, &self.temp, &self.x_buf, &self.hp)?;
        unsafe { self.k_matmul_tx.enq().map_err(map_ocl_err)?; }

        self.k_jacobi.set_arg(0u32, &self.hp).map_err(map_ocl_err)?;
        self.k_jacobi.set_arg(1u32, &self.cp).map_err(map_ocl_err)?;
        unsafe { self.k_jacobi.enq().map_err(map_ocl_err)?; }

        self.k_extract_diag.set_arg(2u32, &self.hp).map_err(map_ocl_err)?;
        self.k_extract_diag.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        unsafe { self.k_extract_diag.enq().map_err(map_ocl_err)?; }
        self.k_select_occ.set_arg(1u32, n_occ as i32).map_err(map_ocl_err)?;
        self.k_select_occ.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        self.k_select_occ.set_arg(4u32, &self.occ_mask).map_err(map_ocl_err)?;
        unsafe { self.k_select_occ.enq().map_err(map_ocl_err)?; }

        set_matmul_args(&self.k_matmul_xc, b, &self.x_buf, &self.cp, &self.c)?;
        unsafe { self.k_matmul_xc.enq().map_err(map_ocl_err)?; }

        self.k_density.set_arg(2u32, &self.c).map_err(map_ocl_err)?;
        self.k_density.set_arg(3u32, &self.occ_mask).map_err(map_ocl_err)?;
        self.k_density.set_arg(4u32, &self.d).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, 0i32).map_err(map_ocl_err)?;
        unsafe { self.k_density.enq().map_err(map_ocl_err)?; }

        self.k_mulliken.set_arg(3u32, &self.d).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(5u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(6u32, &self.q_new).map_err(map_ocl_err)?;
        unsafe { self.k_mulliken.enq().map_err(map_ocl_err)?; }

        // 10. GPU-side DIIS mixing (R9b)
        self.k_diis.set_arg(2u32, alpha).map_err(map_ocl_err)?;
        self.k_diis.set_arg(3u32, &self.q_new).map_err(map_ocl_err)?;
        self.k_diis.set_arg(4u32, &self.q_gpu).map_err(map_ocl_err)?;
        // History buffers and workspace already bound at construction
        unsafe { self.k_diis.enq().map_err(map_ocl_err)?; }

        // Read RMS and return max (only scalar readback)
        rt.read_buffer(&self.rms, &mut self.rms_host)?;
        max_finite_f32(&self.rms_host, "DIIS rms")
    }

    /// Reset DIIS history (call when geometry changes or for warm start).
    pub fn reset_diis(&mut self, rt: &GpuRuntime) -> Result<()> {
        let batch = self.batch;
        // Zero out buf_idx and n_filled
        let zeros_i = vec![0i32; batch];
        self.diis_buf_idx.write(&zeros_i).enq().map_err(map_ocl_err)?;
        self.diis_n_filled.write(&zeros_i).enq().map_err(map_ocl_err)?;
        Ok(())
    }
    /// Returns energies per system.
    ///
    /// R7: Calls `finalize` first to ensure D, Δq, V all correspond to the
    /// current q_gpu. Without this, D is from q_n (last scc_step) but Δq/V
    /// would be from q_{n+1} (mixed), creating an energy-gradient inconsistency.
    pub fn compute_energy(
        &mut self,
        rt: &mut GpuRuntime,
        h0_buf: &Buffer<f32>,
        s_buf: &Buffer<f32>,
        g_buf: &Buffer<f32>,
        q0_buf: &Buffer<f32>,
        orb_atom_buf: &Buffer<i32>,
        n_occ: usize,
    ) -> Result<Vec<f64>> {
        // R7: Finalize — re-solve electronics with current q_gpu so D, Δq, V
        // all correspond to the same charge state.
        self.finalize(rt, h0_buf, s_buf, g_buf, q0_buf, orb_atom_buf, n_occ)?;
        self.energy_from_state(rt, q0_buf)
    }

    /// Band + repulsive energy from the current finalized state. Does **not** re-solve.
    /// Caller must have called `finalize` (or `eval`).
    pub fn energy_from_state(&mut self, rt: &mut GpuRuntime, q0_buf: &Buffer<f32>) -> Result<Vec<f64>> {
        // E = Tr(D H0) + ½ Δq·V.  On the CPU this is identical to
        //   2 Σ_{k occ} ε_k − ½ Δq·V − q0·V
        // (AT identity holds to 5e-10).  Tr(D H0) in f32 for N=87 was 7e-5 Ha
        // off CPU because D is a noisy f32 projector; occupied ε are better.
        // Remaining AT |dE|~2.6e-5 is the eigen floor (C/ε), not this sum.
        // SSOT: doc/prokop/topical_audit/f32_floor_dense_hbond.md
        let batch = self.batch;
        let n = self.n;
        self.k_dot.set_arg(2u32, &self.dq).map_err(map_ocl_err)?;
        self.k_dot.set_arg(3u32, &self.v).map_err(map_ocl_err)?;
        self.k_dot.set_arg(4u32, &self.dot).map_err(map_ocl_err)?;
        unsafe { self.k_dot.enq().map_err(map_ocl_err)?; }
        self.k_dot.set_arg(2u32, q0_buf).map_err(map_ocl_err)?;
        self.k_dot.set_arg(3u32, &self.v).map_err(map_ocl_err)?;
        self.k_dot.set_arg(4u32, &self.tr).map_err(map_ocl_err)?;
        unsafe { self.k_dot.enq().map_err(map_ocl_err)?; }

        rt.read_buffer(&self.eig_diag, &mut self.eig_diag_host)?;
        rt.read_buffer(&self.occ_mask, &mut self.mask_host)?;
        let mut dqv = vec![0.0f32; batch];
        let mut q0v = vec![0.0f32; batch];
        rt.read_buffer(&self.dot, &mut dqv)?;
        rt.read_buffer(&self.tr, &mut q0v)?;

        let mut e = vec![0.0f64; batch];
        for bi in 0..batch {
            let mut e_band = 0.0f64;
            let base = bi * n;
            for k in 0..n {
                if self.mask_host[base + k] != 0 {
                    e_band += 2.0 * self.eig_diag_host[base + k] as f64;
                }
            }
            e[bi] = e_band - 0.5 * dqv[bi] as f64 - q0v[bi] as f64;
            if !e[bi].is_finite() {
                return Err(DftbError::InvalidInput(format!("energy_from_state: E[{bi}]={} band={e_band} Δq·V={} q0·V={}", e[bi], dqv[bi], q0v[bi])));
            }
        }

        // R5: Add repulsive energy if spline data is set
        if let Some(ref k) = self.k_rep_energy {
            let rep_buf = self.rep_e_rep.as_ref().unwrap();
            unsafe { k.enq().map_err(map_ocl_err)?; }
            let mut e_rep = vec![0.0f32; batch];
            rt.read_buffer(rep_buf, &mut e_rep)?;
            for i in 0..batch {
                if !e_rep[i].is_finite() {
                    return Err(DftbError::InvalidInput(format!("E_rep[{i}]={} non-finite", e_rep[i])));
                }
                e[i] += e_rep[i] as f64;
            }
        }

        Ok(e)
    }

    /// Finalize the electronic state: do one unmixed electronic solve with
    /// the current q_gpu so D, C, H_scc, V, Δq all correspond to q_gpu.
    ///
    /// R7: After `scc_step`, q_gpu is the mixed q_{n+1} but D/C/H_scc/V are
    /// from q_n. This method re-solves steps 1-8 of the SCC loop without
    /// mixing, so all quantities are consistent with the final charges.
    /// Call this before `compute_energy` or before computing forces.
    pub fn finalize(
        &mut self,
        rt: &mut GpuRuntime,
        h0_buf: &Buffer<f32>,
        s_buf: &Buffer<f32>,
        g_buf: &Buffer<f32>,
        q0_buf: &Buffer<f32>,
        orb_atom_buf: &Buffer<i32>,
        n_occ: usize,
    ) -> Result<()> {
        let n = self.n;
        let n_atoms = self.n_atoms;
        let batch = self.batch;
        let b = self.matmul_buf_base;
        let _ = (n, n_atoms, batch);  // used in set_arg calls below

        // 1. Δq = q_gpu − q0 (using current/final q_gpu)
        self.k_delta_q.set_arg(2u32, &self.q_gpu).map_err(map_ocl_err)?;
        self.k_delta_q.set_arg(3u32, q0_buf).map_err(map_ocl_err)?;
        self.k_delta_q.set_arg(4u32, &self.dq).map_err(map_ocl_err)?;
        unsafe { self.k_delta_q.enq().map_err(map_ocl_err)?; }

        // 2. V = G · Δq
        self.k_gamma.set_arg(2u32, g_buf).map_err(map_ocl_err)?;
        self.k_gamma.set_arg(3u32, &self.dq).map_err(map_ocl_err)?;
        self.k_gamma.set_arg(4u32, &self.v).map_err(map_ocl_err)?;
        unsafe { self.k_gamma.enq().map_err(map_ocl_err)?; }

        // 3. H_scc = H0 + 0.5·S·(V_i + V_j)
        self.k_h_scc.set_arg(3u32, h0_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(5u32, &self.v).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(6u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_h_scc.set_arg(7u32, &self.h_scc).map_err(map_ocl_err)?;
        unsafe { self.k_h_scc.enq().map_err(map_ocl_err)?; }

        // 4. H' = X · H_scc · X (2 GEMMs)
        set_matmul_args(&self.k_matmul_xh, b, &self.x_buf, &self.h_scc, &self.temp)?;
        unsafe { self.k_matmul_xh.enq().map_err(map_ocl_err)?; }
        set_matmul_args(&self.k_matmul_tx, b, &self.temp, &self.x_buf, &self.hp)?;
        unsafe { self.k_matmul_tx.enq().map_err(map_ocl_err)?; }

        // 5. Jacobi(H') → eigenvalues on diag(hp), eigenvectors in cp
        self.k_jacobi.set_arg(0u32, &self.hp).map_err(map_ocl_err)?;
        self.k_jacobi.set_arg(1u32, &self.cp).map_err(map_ocl_err)?;
        unsafe { self.k_jacobi.enq().map_err(map_ocl_err)?; }

        // 6. Extract diag → GPU sort + occ_mask
        self.k_extract_diag.set_arg(2u32, &self.hp).map_err(map_ocl_err)?;
        self.k_extract_diag.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        unsafe { self.k_extract_diag.enq().map_err(map_ocl_err)?; }
        self.k_select_occ.set_arg(1u32, n_occ as i32).map_err(map_ocl_err)?;
        self.k_select_occ.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        self.k_select_occ.set_arg(4u32, &self.occ_mask).map_err(map_ocl_err)?;
        unsafe { self.k_select_occ.enq().map_err(map_ocl_err)?; }

        // 7. C = X · C'
        set_matmul_args(&self.k_matmul_xc, b, &self.x_buf, &self.cp, &self.c)?;
        unsafe { self.k_matmul_xc.enq().map_err(map_ocl_err)?; }

        // 8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T
        self.k_density.set_arg(2u32, &self.c).map_err(map_ocl_err)?;
        self.k_density.set_arg(3u32, &self.occ_mask).map_err(map_ocl_err)?;
        self.k_density.set_arg(4u32, &self.d).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, 0i32).map_err(map_ocl_err)?;
        unsafe { self.k_density.enq().map_err(map_ocl_err)?; }

        // q_D from this D (q_gpu is still q_in). Do not mix.
        self.k_mulliken.set_arg(3u32, &self.d).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(5u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(6u32, &self.q_new).map_err(map_ocl_err)?;
        unsafe { self.k_mulliken.enq().map_err(map_ocl_err)?; }

        Ok(())
    }

    /// W = 2 Σ_{k occ} ε_k C_k C_kᵀ into `out`. Same kernel as D (`use_eig=1`). Restores D args after.
    pub fn build_edm(&mut self, out: &Buffer<f32>) -> Result<()> {
        self.k_density.set_arg(2u32, &self.c).map_err(map_ocl_err)?;
        self.k_density.set_arg(3u32, &self.occ_mask).map_err(map_ocl_err)?;
        self.k_density.set_arg(4u32, out).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, 1i32).map_err(map_ocl_err)?;
        self.k_density.set_arg(6u32, &self.eig_diag).map_err(map_ocl_err)?;
        let enq = unsafe { self.k_density.enq().map_err(map_ocl_err) };
        self.k_density.set_arg(4u32, &self.d).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, 0i32).map_err(map_ocl_err)?;
        enq
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
        self.k_extract_diag.set_arg(2u32, &self.hp).map_err(map_ocl_err)?;
        self.k_extract_diag.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        unsafe { self.k_extract_diag.enq().map_err(map_ocl_err)?; }
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

// ---- Helper functions for building persistent kernels ----

/// Set matmul buffer arguments at the correct indices.
/// For N≤64 (full-local): a=[2], b=[3], c=[4].
/// For N>64 (tiled):      a=[6], b=[7], c=[8].
fn set_matmul_args(k: &Kernel, base: u32, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<()> {
    k.set_arg(base, a).map_err(map_ocl_err)?;
    k.set_arg(base + 1, b).map_err(map_ocl_err)?;
    k.set_arg(base + 2, c).map_err(map_ocl_err)?;
    Ok(())
}

/// Build the three matmul kernels needed by the SCC step.
/// Returns (buf_base, k_xh, k_tx, k_xc) where buf_base is the index of the first buffer arg.
fn build_matmul_kernels(
    rt: &mut GpuRuntime,
    mat_prog: &Program,
    n: usize,
    batch: usize,
    x_buf: &Buffer<f32>,
    h_scc: &Buffer<f32>,
    temp: &Buffer<f32>,
    hp: &Buffer<f32>,
    cp: &Buffer<f32>,
    c: &Buffer<f32>,
) -> Result<(u32, Kernel, Kernel, Kernel)> {
    if n <= 64 {
        // Full-local matmul: args [0]=n, [1]=batch, [2]=a, [3]=b, [4]=c
        let wg = full_local_wg(n);
        let source = render_source_full_local(n, wg);
        let prog = rt.build_program(&source)?;
        let k_xh = Kernel::builder()
            .program(&prog).name("matmul_full_local_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(n as i32).arg(batch as i32)
            .arg(x_buf).arg(h_scc).arg(temp)
            .build().map_err(map_ocl_err)?;
        let k_tx = Kernel::builder()
            .program(&prog).name("matmul_full_local_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(n as i32).arg(batch as i32)
            .arg(temp).arg(x_buf).arg(hp)
            .build().map_err(map_ocl_err)?;
        let k_xc = Kernel::builder()
            .program(&prog).name("matmul_full_local_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(n as i32).arg(batch as i32)
            .arg(x_buf).arg(cp).arg(c)
            .build().map_err(map_ocl_err)?;
        Ok((2, k_xh, k_tx, k_xc))
    } else {
        // Tiled GEMM: args [0]=n, [1]=batch, [2]=trans_a, [3]=trans_b, [4]=alpha, [5]=beta, [6]=a, [7]=b, [8]=c, [9]=local_a, [10]=local_b
        const TILE_M: usize = 16;
        const TILE_N: usize = 16;
        const TILE_K: usize = 32;
        let row_groups = (n + TILE_M - 1) / TILE_M;
        let col_groups = (n + TILE_N - 1) / TILE_N;
        let gws = ocl::SpatialDims::Three(col_groups * TILE_N, row_groups * TILE_M, batch);
        let lws = ocl::SpatialDims::Two(TILE_N, TILE_M);
        let a_local = TILE_M * TILE_K;
        let b_local = TILE_K * TILE_N;
        let k_xh = Kernel::builder()
            .program(mat_prog).name("batched_gemm").queue(rt.queue().clone())
            .global_work_size(gws.clone()).local_work_size(lws.clone())
            .arg(n as i32).arg(batch as i32)
            .arg(0i32).arg(0i32).arg(1.0f32).arg(0.0f32)  // no transpose, α=1, β=0
            .arg(x_buf).arg(h_scc).arg(temp)
            .arg_local::<f32>(a_local).arg_local::<f32>(b_local)
            .build().map_err(map_ocl_err)?;
        let k_tx = Kernel::builder()
            .program(mat_prog).name("batched_gemm").queue(rt.queue().clone())
            .global_work_size(gws.clone()).local_work_size(lws.clone())
            .arg(n as i32).arg(batch as i32)
            .arg(0i32).arg(0i32).arg(1.0f32).arg(0.0f32)
            .arg(temp).arg(x_buf).arg(hp)
            .arg_local::<f32>(a_local).arg_local::<f32>(b_local)
            .build().map_err(map_ocl_err)?;
        let k_xc = Kernel::builder()
            .program(mat_prog).name("batched_gemm").queue(rt.queue().clone())
            .global_work_size(gws).local_work_size(lws)
            .arg(n as i32).arg(batch as i32)
            .arg(0i32).arg(0i32).arg(1.0f32).arg(0.0f32)
            .arg(x_buf).arg(cp).arg(c)
            .arg_local::<f32>(a_local).arg_local::<f32>(b_local)
            .build().map_err(map_ocl_err)?;
        Ok((6, k_xh, k_tx, k_xc))
    }
}

/// Build the Jacobi eigensolver kernel (full-local for N≤64, tiled for N>64).
fn build_jacobi_kernel(
    rt: &mut GpuRuntime,
    n: usize,
    batch: usize,
    a_buf: &Buffer<f32>,
    v_buf: &Buffer<f32>,
) -> Result<Kernel> {
    if n <= 64 {
        let (_, _, _, _, wg) = eigen_spec_params(n);
        let source = eigen_render_source(n);
        let program = rt.build_program(&source)?;
        Kernel::builder()
            .program(&program).name("jacobi_cyclic_local_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(a_buf).arg(v_buf).arg(n as i32).arg(batch as i32)
            .build().map_err(map_ocl_err)
    } else {
        let b = 32usize;
        let wg = 256usize;
        let source = tiled_render_source(b, wg);
        let program = rt.build_program(&source)?;
        Kernel::builder()
            .program(&program).name("tiled_jacobi_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(a_buf).arg(v_buf).arg(n as i32).arg(batch as i32)
            .build().map_err(map_ocl_err)
    }
}

/// Persistent S^{-1/2} kernels. Full-local `build_inv_sqrt_from_eig` for N≤64;
/// tiled scale + GEMM for N>64. Buffers are bound once; `enqueue_sinv` only copies S.
fn build_sinv_kernels(
    rt: &mut GpuRuntime,
    mat_prog: &Program,
    n: usize,
    batch: usize,
    s_work: &Buffer<f32>,
    s_v: &Buffer<f32>,
    s_v_scaled: &Buffer<f32>,
    x_buf: &Buffer<f32>,
    lambda_min: &Buffer<f32>,
) -> Result<(Option<Kernel>, Option<Kernel>, Option<Kernel>)> {
    if n <= 64 {
        let (_, _, _, _, wg) = eigen_spec_params(n);
        let source = eigen_render_source(n.max(1));
        let program = rt.build_program(&source)?;
        let k = Kernel::builder()
            .program(&program).name("build_inv_sqrt_from_eig").queue(rt.queue().clone())
            .global_work_size(batch.max(1) * wg).local_work_size(wg)
            .arg(s_work).arg(s_v).arg(x_buf).arg(lambda_min)
            .arg(n as i32).arg(batch as i32)
            .build().map_err(map_ocl_err)?;
        Ok((Some(k), None, None))
    } else {
        let wg = 256usize;
        let source = eigen_render_source(64); // LAMBDA_FLOOR only; n is a kernel arg
        let program = rt.build_program(&source)?;
        let k_scale = Kernel::builder()
            .program(&program).name("scale_eigenvectors_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(s_work).arg(s_v).arg(s_v_scaled).arg(lambda_min)
            .arg(n as i32).arg(batch as i32)
            .build().map_err(map_ocl_err)?;
        const TILE_M: usize = 16;
        const TILE_N: usize = 16;
        const TILE_K: usize = 32;
        let row_groups = (n + TILE_M - 1) / TILE_M;
        let col_groups = (n + TILE_N - 1) / TILE_N;
        let gws = ocl::SpatialDims::Three(col_groups * TILE_N, row_groups * TILE_M, batch);
        let lws = ocl::SpatialDims::Two(TILE_N, TILE_M);
        let k_gemm = Kernel::builder()
            .program(mat_prog).name("batched_gemm").queue(rt.queue().clone())
            .global_work_size(gws).local_work_size(lws)
            .arg(n as i32).arg(batch as i32)
            .arg(0i32).arg(1i32).arg(1.0f32).arg(0.0f32) // X = V_scaled · V^T
            .arg(s_v_scaled).arg(s_v).arg(x_buf)
            .arg_local::<f32>(TILE_M * TILE_K).arg_local::<f32>(TILE_K * TILE_N)
            .build().map_err(map_ocl_err)?;
        Ok((None, Some(k_scale), Some(k_gemm)))
    }
}

fn enqueue_sinv(
    rt: &GpuRuntime,
    s_buf: &Buffer<f32>,
    s_work: &Buffer<f32>,
    n_elem: usize,
    k_jacobi: &Kernel,
    k_invsqrt: Option<&Kernel>,
    k_scale: Option<&Kernel>,
    k_xgemm: Option<&Kernel>,
) -> Result<()> {
    rt.copy_into(s_buf, s_work, n_elem)
        .map_err(|e| DftbError::InvalidInput(format!("S→s_work copy for S^{{-1/2}}: {e}")))?;
    unsafe { k_jacobi.enq().map_err(|e| DftbError::InvalidInput(format!("S Jacobi for S^{{-1/2}}: {e}")))?; }
    if let Some(k) = k_invsqrt {
        unsafe { k.enq().map_err(|e| DftbError::InvalidInput(format!("build_inv_sqrt_from_eig: {e}")))?; }
    } else {
            let k_s = k_scale.ok_or_else(|| DftbError::InvalidInput("S^{-1/2} tiled path missing scale kernel".into()))?;
        let k_g = k_xgemm.ok_or_else(|| DftbError::InvalidInput("S^{-1/2} tiled path missing GEMM kernel".into()))?;
        unsafe { k_s.enq().map_err(|e| DftbError::InvalidInput(format!("scale_eigenvectors for S^{{-1/2}}: {e}")))?; }
        unsafe { k_g.enq().map_err(|e| DftbError::InvalidInput(format!("X=V_scaled·V^T for S^{{-1/2}}: {e}")))?; }
    }
    Ok(())
}

fn check_overlap_lambda(ls: &[f32]) -> Result<()> {
    for (i, &l) in ls.iter().enumerate() {
        if !l.is_finite() || l <= 1e-6 {
            return Err(DftbError::InvalidInput(format!(
                "S^{{-1/2}}: overlap λ_min[{i}]={l} (non-finite or ≤1e-6). Kernel would rsqrt-clamp — fail loud instead."
            )));
        }
    }
    Ok(())
}

/// One Newton step on X: M=XᵀSX=I+E → X ← X(I−E/2), then symmetrize.
/// Host f64, once per geometry. Skips writeback if it does not reduce max|E|.
fn repair_lowdin_x(
    rt: &GpuRuntime,
    x_buf: &Buffer<f32>,
    s_buf: &Buffer<f32>,
    n: usize,
    batch: usize,
    scratch_x: &mut [f32],
    scratch_s: &mut [f32],
    work_a: &mut [f64],
    work_b: &mut [f64],
    work_c: &mut [f64],
) -> Result<()> {
    let nn = n * n;
    if scratch_x.len() != batch * nn || scratch_s.len() != batch * nn {
        return Err(DftbError::InvalidInput(format!("repair_lowdin_x: scratch {}/{} != batch*n² {batch}*{nn}", scratch_x.len(), scratch_s.len())));
    }
    if work_a.len() != nn || work_b.len() != nn || work_c.len() != nn {
        return Err(DftbError::InvalidInput(format!("repair_lowdin_x: work len {} {} {} != n² {nn}", work_a.len(), work_b.len(), work_c.len())));
    }
    rt.read_buffer(x_buf, scratch_x)?;
    rt.read_buffer(s_buf, scratch_s)?;
    let mut e0_max = 0.0f64;
    let mut e1_max = 0.0f64;
    let mut wrote = false;
    for b in 0..batch {
        let x = &mut scratch_x[b * nn..(b + 1) * nn];
        let s = &scratch_s[b * nn..(b + 1) * nn];
        for i in 0..nn {
            work_a[i] = x[i] as f64;
            work_b[i] = s[i] as f64;
        }
        // C = Xᵀ S
        gemm_at_b(work_a, work_b, work_c, n);
        // M = C X  → work_b
        gemm_nn(work_c, work_a, work_b, n);
        let e0 = max_metric_err(work_b, n);
        if !e0.is_finite() {
            return Err(DftbError::InvalidInput(format!("repair_lowdin_x: max|XᵀSX−I| replica {b} = {e0} non-finite")));
        }
        e0_max = e0_max.max(e0);
        if e0 < 1e-8 { e1_max = e1_max.max(e0); continue; }
        // E = M − I in work_b; XE in work_c; X1 = X − ½ XE in work_a (overwrite copy of X)
        for i in 0..n { work_b[i * n + i] -= 1.0; }
        gemm_nn(work_a, work_b, work_c, n);
        for i in 0..nn {
            let v = work_a[i] - 0.5 * work_c[i];
            if !v.is_finite() {
                return Err(DftbError::InvalidInput(format!("repair_lowdin_x: X1[{i}]={v} replica {b} non-finite")));
            }
            work_a[i] = v;
        }
        for i in 0..n {
            for j in 0..i {
                let a = 0.5 * (work_a[i * n + j] + work_a[j * n + i]);
                work_a[i * n + j] = a;
                work_a[j * n + i] = a;
            }
        }
        // M1 = X1ᵀ S X1
        for i in 0..nn { work_b[i] = s[i] as f64; }
        gemm_at_b(work_a, work_b, work_c, n);
        gemm_nn(work_c, work_a, work_b, n);
        let e1 = max_metric_err(work_b, n);
        if !e1.is_finite() {
            return Err(DftbError::InvalidInput(format!("repair_lowdin_x: after Newton max|E| replica {b} = {e1} non-finite")));
        }
        if e1 < e0 {
            for i in 0..nn { x[i] = work_a[i] as f32; }
            wrote = true;
            e1_max = e1_max.max(e1);
        } else {
            e1_max = e1_max.max(e0);
            eprintln!("[GpuSccPlan] Löwdin Newton skip replica {b}: max|E| {e0:.3e} → {e1:.3e} (not improved)");
        }
    }
    if wrote { rt.write_buffer(x_buf, scratch_x)?; }
    eprintln!("[GpuSccPlan] Löwdin Newton max||XᵀSX−I|| {e0_max:.3e} → {e1_max:.3e} wrote={wrote}");
    Ok(())
}

fn gemm_nn(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n { s += a[i * n + k] * b[k * n + j]; }
            c[i * n + j] = s;
        }
    }
}

fn gemm_at_b(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    // C = Aᵀ B, A,B row-major
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n { s += a[k * n + i] * b[k * n + j]; }
            c[i * n + j] = s;
        }
    }
}

fn max_metric_err(m: &[f64], n: usize) -> f64 {
    let mut mx = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let t = if i == j { 1.0 } else { 0.0 };
            mx = mx.max((m[i * n + j] - t).abs());
        }
    }
    mx
}
