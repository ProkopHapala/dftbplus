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

/// §12 D2: tiled-Jacobi arithmetic precision. 0 = pure FP32-FMA,
/// 1 = FP64 scalar rotation construction only, 2 = broad FP64 (reference).
/// Set before `GpuSccPlan::new` (construction-time knob for A/B runs).
/// Default 1 — measured (RTX 3090, tests/gpu_tiled_jacobi.rs jacobi_prec_bench,
/// N=87): prec0 residual 4.8e-5 (too coarse), prec1 2.1e-6 at 82 ms,
/// prec2 1.2e-6 at 243 ms → prec1 ≈ prec2 accuracy at ~3× speed.
pub static JACOBI_PREC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

pub fn set_jacobi_prec(p: u32) {
    assert!(p <= 2, "JACOBI_PREC must be 0,1,2 — got {p}");
    JACOBI_PREC.store(p, std::sync::atomic::Ordering::Relaxed);
}

fn tiled_render_source(b: usize, wg: usize, prec: u32) -> String {
    let pb = 2 * b;
    let pld = pb + 1;
    GPU_TILED_JACOBI_TEMPLATE
        .replace("#define B 32", &format!("#define B {}", b))
        .replace("#define PB 64", &format!("#define PB {}", pb))
        .replace("#define PLD 65", &format!("#define PLD {}", pld))
        .replace("#define WG 256", &format!("#define WG {}", wg))
        .replace("#define STRIP_R 32", &format!("#define STRIP_R {}", b))
        .replace("#define MAX_SWEEPS 50", &format!("#define MAX_SWEEPS {}", tiled_max_sweeps()))
        .replace("#define JACOBI_PREC 2", &format!("#define JACOBI_PREC {}", prec))
}

/// Jacobi outer-sweep cap, `RUST_DFTB_JACOBI_SWEEPS` override (diagnostic).
/// Measured GC N=86 batch=19, one warm SCC call: cap=5 → 56 ms / rms 4.4e-6,
/// cap=8 → 80 ms / rms 4.1e-6, cap=16/100 → 80 ms (identical) — i.e. the
/// stagnation detector fires at ~8 sweeps, ~10 ms PER SWEEP. Capping below 5
/// is counterproductive: cap=3 → SCC needs 60 iters instead of 10.
fn tiled_max_sweeps() -> usize {
    std::env::var("RUST_DFTB_JACOBI_SWEEPS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(TILED_MAX_SWEEPS)
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
    pub q_next: Buffer<f32>,      // mixed next iterate [batch*n_atoms] (commit model)
    pub active: Buffer<i32>,      // [batch] 1 = still iterating, 0 = frozen/done
    ones: Buffer<i32>,            // [batch] all-ones — for kernels that must never gate (S Jacobi)
    pub tr: Buffer<f32>,          // trace [batch]
    pub dot: Buffer<f32>,         // dot product [batch]
    pub rms: Buffer<f32>,         // residual RMS [batch]
    pub occ_mask: Buffer<i32>,    // occupation mask [batch*n]
    pub occ_idx: Buffer<i32>,     // sorted occupied column indices [batch*n] (D10)
    pub occ_w: Buffer<f32>,       // per-orbital Fermi weights f_k [batch*n] (smearing)
    pub eig_diag: Buffer<f32>,    // extracted diagonal [batch*n]
    pub eig_rho: Buffer<f32>,     // occupied Rayleigh quotients ρ_k [batch*n] (§12 D3/D4)
    pub x_buf: Buffer<f32>,       // Löwdin transform S^{-1/2} [batch*nn]
    x_t: Buffer<f32>,             // Xᵀ — H' = Xᵀ·H·X must not rely on X symmetric (D6)
    s_work: Buffer<f32>,          // Jacobi A workspace for S (copy of S; destroyed)
    s_v: Buffer<f32>,             // Jacobi V for S
    s_v_scaled: Buffer<f32>,      // V·rsqrt(λ) (N>64 path; unused N≤64)
    lambda_min: Buffer<f32>,      // [batch] λ_min(S)

    // R9b: GPU-side DIIS history buffers
    pub diis_q_hist: Buffer<f32>,     // [batch*max_hist*n_atoms] Δq_in ring buffer (D9: Δq-space)
    pub diis_r_hist: Buffer<f32>,     // [batch*max_hist*n_atoms] residual ring buffer
    pub diis_buf_idx: Buffer<i32>,    // [batch] ring buffer write position
    pub diis_n_filled: Buffer<i32>,   // [batch] number of valid entries
    pub diis_coeffs: Buffer<f32>,     // [batch*max_hist] coefficients
    pub diis_flag: Buffer<i32>,       // [batch] fallback counter (D9: no printf)
    pub diis_reason: Buffer<i32>,     // [batch] last fallback reason
    pub diis_max_hist: usize,          // max history length (typically 10)

    // Host staging buffers (reused, not re-allocated)
    pub eig_diag_host: Vec<f32>,  // [batch*n]
    pub eig_rho_host: Vec<f32>,   // [batch*n] occupied Rayleigh quotients
    pub mask_host: Vec<i32>,      // [batch*n]
    pub rms_host: Vec<f32>,       // [batch]
    pub active_host: Vec<i32>,    // [batch] staging for `active` mask
    lambda_min_host: Vec<f32>,    // [batch] overlap λ_min
    lowdin_e0_host: Vec<f32>,     // [batch] ‖XᵀSX−I‖∞ before/after repair
    lowdin_e1_host: Vec<f32>,

    // Pre-built kernels (R8: no Kernel::builder() in hot loops)
    // SCC step kernels:
    k_dq_v_hscc: Kernel,      // fused_dq_v_hscc_batched (D10/R18)
    k_matmul_xh: Kernel,      // Xᵀ · H_scc → temp (via x_t; D6)
    k_matmul_tx: Kernel,      // temp · X → hp
    k_matmul_xc: Kernel,      // X · cp → c
    k_transpose: Kernel,      // transpose_batched — x_t ← x_bufᵀ (D6)
    k_jacobi: Kernel,         // jacobi (full-local or tiled)
    k_extract_diag: Kernel,   // extract_diagonal_batched
    k_select_occ: Kernel,     // select_occupation_batched (R9: GPU-side occ selection)
    k_density: Kernel,         // build_density_masked_batched
    k_occ_renorm: Kernel,      // occ_normalize_batched (§12 D3: repair C′ orthonormality)
    k_occ_snorm: Kernel,       // snormalize_batched (S-metric col renorm for warm AO basis)
    k_occ_rayleigh: Kernel,    // occ_rayleigh_batched (§12 D4: ρ_k for E_band and W)
    k_mulliken: Kernel,        // mulliken_charges_batched
    k_residual_mix: Kernel,    // residual_and_mix_batched (simple mixing fallback)
    k_diis: Kernel,            // diis_step_batched (R9b: GPU-side DIIS)
    k_commit: Kernel,          // commit_q_batched (q ← q_next for active replicas)
    // Energy kernels:
    k_frobenius_trace: Kernel, // frobenius_trace_batched
    k_dot: Kernel,             // dot_batched

    // Matmul arg layout: for N≤64, buffer args are at [2,3,4]; for N>64, at [6,7,8].
    matmul_buf_base: u32,

    // n_occ for the occupation kernel (set once per solve, not per iteration)
    n_occ: usize,
    /// Fermi smearing kT (Hartree). 0 → integer occupation (default).
    /// >0 → per-orbital weights occ_w = f(ε,μ,kT), μ per replica via host
    /// bisection on eig_diag (tiny readback, ~KB/iter).
    pub kT: f32,
    occ_w_host: Vec<f32>,   // [batch*n] weights, reused as scratch + energy readback

    /// §12 D3/D4: renormalize occupied C′ columns in `finalize` and use
    /// Rayleigh quotients ρ_k (not the drifted Jacobi diagonal ε_k) for
    /// E_band and W. Disable only for A/B measurement.
    pub occ_repair: bool,
    /// §12 D3: also renormalize occupied C′ inside each SCC step (density
    /// path). Measured: AT rms 1.30e-6 stalled@25 → 8.9e-7 converged@13.
    /// Default on; disable only for A/B measurement.
    pub occ_repair_scc: bool,
    /// §12 D6: true once x_buf holds a certified X — subsequent set_geometry
    /// calls Newton-polish the old X before considering a Jacobi rebuild.
    x_warm: bool,
    /// §12 D6 A/B: disable to force a full Jacobi(S) rebuild each geometry.
    pub x_reuse: bool,
    /// Warm AO basis: `c` holds the previous solve's eigenvectors
    /// (S-orthonormal). When true and n>64 the eigensolve projects
    /// A = cᵀH_scc c and Jacobi rotates c in place — no X transform,
    /// no X·C′ back-GEMM, ~1-2 sweeps instead of a cold solve.
    b_warm: bool,

    // S^{-1/2} kernels — built once; set_geometry only copies S and enqueues.
    k_s_jacobi: Kernel,
    k_s_invsqrt: Option<Kernel>,      // N≤64: build_inv_sqrt_from_eig
    k_s_scale: Option<Kernel>,        // N>64: scale_eigenvectors_batched
    k_s_xgemm: Option<Kernel>,        // N>64: X = V_scaled · V^T

    // §12 D5: GPU Löwdin repair (set_geometry only — once per geometry, not
    // a hot loop). Replaces the CPU f64 serial-GEMM `repair_lowdin_x`:
    //   M = XᵀSX (2 GEMMs), Q = (3I−M)/2, X1 = X·Q, M1 = X1ᵀSX1 (2 GEMMs),
    //   per-system accept where ‖M1−I‖ < ‖M−I‖. Only [batch] scalars read back.
    lowdin_t: Buffer<f32>,            // [batch*nn] T1 then X1
    lowdin_m: Buffer<f32>,            // [batch*nn] M then M1
    lowdin_q: Buffer<f32>,            // [batch*nn] Q then T2
    lowdin_e0: Buffer<f32>,           // [batch]
    lowdin_e1: Buffer<f32>,           // [batch]
    k_lowdin_gemm: Kernel,            // batched_gemm (args re-bound per call)
    k_lowdin_qf: Kernel,              // lowdin_q_from_m_batched
    k_metric: Kernel,                 // metric_residual_batched
    k_lowdin_acc: Kernel,             // lowdin_accept_batched
    k_lowdin_acc_c: Kernel,           // lowdin_accept_batched bound to `c` (B-repair)

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
        let x_t = rt.zero_buffer::<f32>(batch * nn)?;
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
        let q_next = rt.zero_buffer::<f32>(batch * n_atoms)?;
        // 1 = still iterating, 0 = frozen/done. Starts all-active; the SCC
        // loop writes per-replica flags each iteration, and every path
        // outside the loop (finalize/eval) calls activate_all() first —
        // the mask is an SCC-loop construct, default = all-ones.
        let active = rt.buffer_from_slice(&vec![1i32; batch])?;
        let ones = rt.buffer_from_slice(&vec![1i32; batch])?;
        let tr = rt.zero_buffer::<f32>(batch)?;
        let dot = rt.zero_buffer::<f32>(batch)?;
        let rms = rt.zero_buffer::<f32>(batch)?;
        let occ_mask = rt.zero_buffer::<i32>(batch * n)?;
        let occ_idx = rt.zero_buffer::<i32>(batch * n)?;
        let occ_w = rt.zero_buffer::<f32>(batch * n)?;
        let eig_diag = rt.zero_buffer::<f32>(batch * n)?;
        let eig_rho = rt.zero_buffer::<f32>(batch * n)?;

        // R9b: DIIS history. Cap at n_atoms — more vectors than the charge space is rank-deficient
        // (H2O: 10 hist in 3-atom q → pivot fallback). AT n_atoms=30 still uses 10.
        let diis_max_hist = n_atoms.min(10).max(1);
        let diis_q_hist = rt.zero_buffer::<f32>(batch * diis_max_hist * n_atoms)?;
        let diis_r_hist = rt.zero_buffer::<f32>(batch * diis_max_hist * n_atoms)?;
        let diis_buf_idx = rt.zero_buffer::<i32>(batch)?;
        let diis_n_filled = rt.zero_buffer::<i32>(batch)?;
        let diis_coeffs = rt.zero_buffer::<f32>(batch * diis_max_hist)?;
        let diis_flag = rt.zero_buffer::<i32>(batch)?;
        let diis_reason = rt.zero_buffer::<i32>(batch)?;
        eprintln!("[GpuSccPlan] DIIS hist={diis_max_hist} (min(10,n_atoms={n_atoms})) N={n} batch={batch}");

        // ---- Build all kernels once ----
        // Matrix ops program (shared by most kernels)
        let mat_prog = rt.build_program(MATRIX_KERNEL_TEMPLATE)?;

        // 1-3 fused: fused_dq_v_hscc_batched (D10/R18) — one WG/system launch
        // for Δq → V=γΔq → H_scc = H0+½S(V_A+V_B). Δq/V in local, also written
        // to global for the energy dot. args [0]=n [1]=n_atoms [2]=batch
        // [3]=q [4]=q0 [5]=g [6]=h0 [7]=s [8]=orb_atom [9]=dq [10]=v [11]=h_scc
        // [12]=local ldq [13]=local lv
        let wg_fuse = 256usize;
        let k_dq_v_hscc = Kernel::builder()
            .program(&mat_prog).name("fused_dq_v_hscc_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_fuse).local_work_size(wg_fuse)
            .arg(n as i32).arg(n_atoms as i32).arg(batch as i32)
            .arg(&q_gpu).arg(&q_gpu).arg(&v)              // q/q0/g placeholders, set per-call
            .arg(&h_scc).arg(&h_scc).arg(&occ_mask)      // h0/s/orb_atom placeholders
            .arg(&dq).arg(&v).arg(&h_scc)
            .arg_local::<f32>(n_atoms).arg_local::<f32>(n_atoms)
            .arg(&active)                                // [14] active mask
            .build().map_err(map_ocl_err)?;

        // 4-6. Three matmul kernels (Xᵀ·H_scc→temp via x_t, temp·X→hp, X·cp→c).
        // D6: A-operand is x_t, NOT x_buf — Newton-reused X is S^{-1/2}·U
        // (non-symmetric gauge); X·H·X would be the wrong eigenproblem.
        let (matmul_buf_base, k_matmul_xh, k_matmul_tx, k_matmul_xc) =
            build_matmul_kernels(rt, &mat_prog, n, batch, &x_t, &x_buf, &h_scc, &temp, &hp, &cp, &c)?;

        // transpose_batched: args [0]=n [1]=batch [2]=a [3]=at — binds
        // x_buf→x_t once; enqueued at the end of every set_geometry.
        let wg_tr = 256usize;
        let k_transpose = Kernel::builder()
            .program(&mat_prog).name("transpose_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_tr).local_work_size(wg_tr)
            .arg(n as i32).arg(batch as i32)
            .arg(&x_buf).arg(&x_t)
            .build().map_err(map_ocl_err)?;

        // 7. Jacobi eigensolver — gated on `active` so done replicas' WGs exit
        let jacobi_prec = JACOBI_PREC.load(std::sync::atomic::Ordering::Relaxed);
        let k_jacobi = build_jacobi_kernel(rt, n, batch, &hp, &cp, &active, jacobi_prec)?;

        // 8. extract_diagonal_batched: args [0]=n, [1]=batch, [2]=a, [3]=diag
        let total_diag = n * batch;
        let gws_diag = ((total_diag + 63) / 64) * 64;
        let k_extract_diag = Kernel::builder()
            .program(&mat_prog).name("extract_diagonal_batched").queue(rt.queue().clone())
            .global_work_size(gws_diag).local_work_size(64)
            .arg(n as i32).arg(batch as i32)
            .arg(&hp).arg(&eig_diag).arg(&active)
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
            .arg(&eig_diag).arg(&occ_mask).arg(&occ_idx).arg(&active)
            .build().map_err(map_ocl_err)?;

        // 9. build_density_occ_batched (D10): args [0]=n [1]=batch [2]=n_occ
        //    [3]=c [4]=occ_idx [5]=out [6]=use_eig [7]=eig [8]=use_w
        //    [9]=occ_w [10]=lw [11]=loi
        //    use_eig=0 → D (s_k=1); use_eig=1 → W (s_k=ρ_k). Same kernel.
        //    use_w=1 → ×occ_w[k] Fermi weights (smearing, n_occ=n).
        //    Occupied-index list + lower triangle + 4-acc FP32 FMA.
        let wg_den = (n * n).min(256).max(1);
        let k_density = Kernel::builder()
            .program(&mat_prog).name("build_density_occ_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_den).local_work_size(wg_den)
            .arg(n as i32).arg(batch as i32).arg(0i32)  // n_occ=0 dummy, set per-call
            .arg(&c).arg(&occ_idx).arg(&d).arg(0i32).arg(&eig_diag)
            .arg(0i32).arg(&occ_w)
            .arg_local::<f32>(n).arg_local::<i32>(n)
            .arg(&active)                                // [12] active mask
            .build().map_err(map_ocl_err)?;

        // 9b. occ_normalize_batched: args [0]=n [1]=batch [2]=occ_mask [3]=cp [4]=local(wg)
        //     One WG per (system, column); unoccupied early-exit.
        let wg_on = 128usize;
        let k_occ_renorm = Kernel::builder()
            .program(&mat_prog).name("occ_normalize_batched").queue(rt.queue().clone())
            .global_work_size(batch * n * wg_on).local_work_size(wg_on)
            .arg(n as i32).arg(batch as i32)
            .arg(&occ_mask).arg(&cp)
            .arg_local::<f32>(wg_on)
            .arg(&active)                                // [5] active mask
            .build().map_err(map_ocl_err)?;

        // 9b2. snormalize_batched: args [0]=n [1]=batch [2]=c [3]=s [4]=local(n+wg)
        //      S-metric column renorm for the warm AO basis — all columns.
        let wg_sn = 128usize;
        let k_occ_snorm = Kernel::builder()
            .program(&mat_prog).name("snormalize_batched").queue(rt.queue().clone())
            .global_work_size(batch * n * wg_sn).local_work_size(wg_sn)
            .arg(n as i32).arg(batch as i32)
            .arg(&c).arg(&h_scc)  // arg3 = s_buf, set per-call
            .arg_local::<f32>(n + wg_sn)
            .arg(&active)                                // [5] active mask
            .build().map_err(map_ocl_err)?;

        // 9c. occ_rayleigh_batched: args [0]=n [1]=batch [2]=occ_mask [3]=c [4]=h_scc [5]=s [6]=rho [7]=local(2n+2·wg)
        //     One WG per (system, column); ρ_k = cᵀH_sc/cᵀSc.
        let wg_or = 128usize;
        let k_occ_rayleigh = Kernel::builder()
            .program(&mat_prog).name("occ_rayleigh_batched").queue(rt.queue().clone())
            .global_work_size(batch * n * wg_or).local_work_size(wg_or)
            .arg(n as i32).arg(batch as i32)
            .arg(&occ_mask).arg(&c).arg(&h_scc).arg(&h_scc).arg(&eig_rho)  // arg5=s set per-call
            .arg_local::<f32>(2 * n + 2 * wg_or)
            .arg(&active)                                // [8] active mask
            .arg(&occ_w).arg(0i32)                       // [9]=occ_w [10]=use_w set per-call
            .build().map_err(map_ocl_err)?;

        // 10. mulliken_charges_batched: args [0]=n, [1]=n_atoms, [2]=batch, [3]=d, [4]=s, [5]=orb_atom, [6]=q, [7]=local(n)
        let wg_mull = n.max(n_atoms).min(256).max(1);
        let k_mulliken = Kernel::builder()
            .program(&mat_prog).name("mulliken_charges_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_mull).local_work_size(wg_mull)
            .arg(n as i32).arg(n_atoms as i32).arg(batch as i32)
            .arg(&d).arg(&h_scc).arg(&occ_mask).arg(&q_new)  // dummy s/orb_atom, set per-call
            .arg_local::<f32>(n)
            .arg(&active)                                // [8] active mask
            .build().map_err(map_ocl_err)?;

        // 11. residual_and_mix_batched: args [0]=n_atoms, [1]=batch, [2]=alpha, [3]=q_new,
        //     [4]=q_old, [5]=q_mixed(→q_next), [6]=rms, [7]=active, [8]=local(wg)
        let wg_res = 256usize;
        let k_residual_mix = Kernel::builder()
            .program(&mat_prog).name("residual_and_mix_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_res).local_work_size(wg_res)
            .arg(n_atoms as i32).arg(batch as i32).arg(0.3f32)  // alpha, set per-call
            .arg(&q_new).arg(&q_gpu).arg(&q_next).arg(&rms).arg(&active)
            .arg_local::<f32>(wg_res)
            .build().map_err(map_ocl_err)?;

        // 11a. commit_q_batched: args [0]=n_atoms [1]=batch [2]=q_next [3]=q [4]=active
        let k_commit = Kernel::builder()
            .program(&mat_prog).name("commit_q_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_res).local_work_size(wg_res)
            .arg(n_atoms as i32).arg(batch as i32)
            .arg(&q_next).arg(&q_gpu).arg(&active)
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
        // args: [0]n_atoms [1]batch [2]alpha [3]q_new [4]q_old [5]q0
        //       [6]q_next [7]dq_hist [8]r_hist [9]buf_idx [10]n_filled
        //       [11]coeffs [12]diis_flag [13]diis_reason [14]rms [15]active
        //       [16]scratch(local)
        let k_diis = Kernel::builder()
            .program(&diis_prog).name("diis_step_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg_diis).local_work_size(wg_diis)
            .arg(n_atoms as i32).arg(batch as i32).arg(0.3f32)
            .arg(&q_new).arg(&q_gpu).arg(&q_gpu).arg(&q_next)  // q0 set per-call
            .arg(&diis_q_hist).arg(&diis_r_hist)
            .arg(&diis_buf_idx).arg(&diis_n_filled)
            .arg(&diis_coeffs)
            .arg(&diis_flag).arg(&diis_reason)
            .arg(&rms).arg(&active)
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

        let k_s_jacobi = build_jacobi_kernel(rt, n, batch, &s_work, &s_v, &ones, jacobi_prec)?;
        let (k_s_invsqrt, k_s_scale, k_s_xgemm) = build_sinv_kernels(
            rt, &mat_prog, n, batch, &s_work, &s_v, &s_v_scaled, &x_buf, &lambda_min,
        )?;

        // §12 D5: GPU Löwdin repair buffers + kernels (set_geometry only).
        // batched_gemm handles trans + n<tile bounds, so it serves all n.
        let lowdin_t = rt.zero_buffer::<f32>(batch * nn)?;
        let lowdin_m = rt.zero_buffer::<f32>(batch * nn)?;
        let lowdin_q = rt.zero_buffer::<f32>(batch * nn)?;
        let lowdin_e0 = rt.zero_buffer::<f32>(batch)?;
        let lowdin_e1 = rt.zero_buffer::<f32>(batch)?;
        const LG_M: usize = 16;
        const LG_N: usize = 16;
        const LG_K: usize = 32;
        let lg_gws = ocl::SpatialDims::Three(
            ((n + LG_N - 1) / LG_N) * LG_N, ((n + LG_M - 1) / LG_M) * LG_M, batch,
        );
        let lg_lws = ocl::SpatialDims::Two(LG_N, LG_M);
        let k_lowdin_gemm = Kernel::builder()
            .program(&mat_prog).name("batched_gemm").queue(rt.queue().clone())
            .global_work_size(lg_gws).local_work_size(lg_lws)
            .arg(n as i32).arg(batch as i32)
            .arg(0i32).arg(0i32).arg(1.0f32).arg(0.0f32)
            .arg(&lowdin_t).arg(&lowdin_t).arg(&lowdin_m)  // placeholder, re-bound per call
            .arg_local::<f32>(LG_M * LG_K).arg_local::<f32>(LG_K * LG_N)
            .build().map_err(map_ocl_err)?;
        let k_lowdin_qf = Kernel::builder()
            .program(&mat_prog).name("lowdin_q_from_m_batched").queue(rt.queue().clone())
            .global_work_size(batch * 256).local_work_size(256)
            .arg(n as i32).arg(batch as i32)
            .arg(&lowdin_m).arg(&lowdin_q)
            .build().map_err(map_ocl_err)?;
        let k_metric = Kernel::builder()
            .program(&mat_prog).name("metric_residual_batched").queue(rt.queue().clone())
            .global_work_size(batch * 256).local_work_size(256)
            .arg(n as i32).arg(batch as i32)
            .arg(&lowdin_m).arg(&lowdin_e0)  // res re-bound per call
            .arg_local::<f32>(256)
            .build().map_err(map_ocl_err)?;
        let k_lowdin_acc = Kernel::builder()
            .program(&mat_prog).name("lowdin_accept_batched").queue(rt.queue().clone())
            .global_work_size(batch * 256).local_work_size(256)
            .arg(n as i32).arg(batch as i32)
            .arg(&lowdin_e0).arg(&lowdin_e1)
            .arg(&lowdin_t).arg(&x_buf)
            .build().map_err(map_ocl_err)?;
        // Same accept kernel, output bound to `c` — the warm AO basis
        // metric-repair (G=BᵀS_newB, B←B·(3I−G)/2) across geometry changes.
        let k_lowdin_acc_c = Kernel::builder()
            .program(&mat_prog).name("lowdin_accept_batched").queue(rt.queue().clone())
            .global_work_size(batch * 256).local_work_size(256)
            .arg(n as i32).arg(batch as i32)
            .arg(&lowdin_e0).arg(&lowdin_e1)
            .arg(&lowdin_t).arg(&c)
            .build().map_err(map_ocl_err)?;

        let mut plan = Self {
            n, n_atoms, batch,
            q_gpu, dq, v, h_scc, temp, hp, cp, c, d, q_new, q_next, active, ones, tr, dot, rms, occ_mask, occ_idx, occ_w, eig_diag, eig_rho,
            x_buf, x_t, s_work, s_v, s_v_scaled, lambda_min,
            diis_q_hist, diis_r_hist, diis_buf_idx, diis_n_filled,
            diis_coeffs, diis_flag, diis_reason, diis_max_hist,
            eig_diag_host: vec![0.0; batch * n],
            eig_rho_host: vec![0.0; batch * n],
            mask_host: vec![0; batch * n],
            rms_host: vec![0.0; batch],
            active_host: vec![1; batch],
            lambda_min_host: vec![0.0; batch],
            lowdin_e0_host: vec![0.0; batch],
            lowdin_e1_host: vec![0.0; batch],
            k_dq_v_hscc,
            k_matmul_xh, k_matmul_tx, k_matmul_xc, k_transpose,
            k_jacobi, k_extract_diag, k_select_occ, k_density, k_occ_renorm, k_occ_snorm, k_occ_rayleigh, k_mulliken, k_residual_mix,
            k_diis, k_commit, k_frobenius_trace, k_dot,
            matmul_buf_base,
            n_occ: 0,
            kT: 0.0,
            occ_w_host: vec![0.0; batch * n],
            occ_repair: true,
            occ_repair_scc: true,
            x_warm: false,
            x_reuse: true,
            b_warm: false,
            k_s_jacobi, k_s_invsqrt, k_s_scale, k_s_xgemm,
            lowdin_t, lowdin_m, lowdin_q, lowdin_e0, lowdin_e1,
            k_lowdin_gemm, k_lowdin_qf, k_metric, k_lowdin_acc, k_lowdin_acc_c,
            k_rep_energy: None,
            rep_coords: None,
            rep_species_idx: None,
            rep_spline_offsets: None,
            rep_spline_data: None,
            rep_e_rep: None,
            rep_n_species: 0,
        };
        plan.set_geometry(rt, s_buf)
            .map_err(|e| DftbError::InvalidInput(format!("GpuSccPlan::new initial S^{{-1/2}}+repair: {e}")))?;
        Ok(plan)
    }

    /// D10/R18: one fused launch for Δq → V=γΔq → H_scc. Replaces the
    /// delta_q + gamma_matvec + h_scc_update sequence (3 launches, and the
    /// dq/v global round-trip between them).
    fn enq_dq_v_hscc(
        &mut self,
        h0_buf: &Buffer<f32>,
        s_buf: &Buffer<f32>,
        g_buf: &Buffer<f32>,
        q0_buf: &Buffer<f32>,
        orb_atom_buf: &Buffer<i32>,
    ) -> Result<()> {
        self.k_dq_v_hscc.set_arg(3u32, &self.q_gpu).map_err(map_ocl_err)?;
        self.k_dq_v_hscc.set_arg(4u32, q0_buf).map_err(map_ocl_err)?;
        self.k_dq_v_hscc.set_arg(5u32, g_buf).map_err(map_ocl_err)?;
        self.k_dq_v_hscc.set_arg(6u32, h0_buf).map_err(map_ocl_err)?;
        self.k_dq_v_hscc.set_arg(7u32, s_buf).map_err(map_ocl_err)?;
        self.k_dq_v_hscc.set_arg(8u32, orb_atom_buf).map_err(map_ocl_err)?;
        unsafe { self.k_dq_v_hscc.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Eigenproblem solve: h_scc → rotated matrix in `hp` (eigenvalues on
    /// its diagonal), eigenvectors in `cp` (cold) or `c` (warm).
    ///
    /// Warm path (b_warm, n>64): A = cᵀ·H_scc·c — c is the previous
    /// eigenbasis, S-orthonormal, so the projected problem is already
    /// near-diagonal and Jacobi exits in ~1-2 sweeps. Jacobi rotates c
    /// in place (init_v=1): on exit c IS the new eigenvector matrix —
    /// no X·C′ back-transform GEMM at all.
    /// Cold path: A = Xᵀ·H_scc·X, Jacobi with V=I into cp.
    fn eigh_solve(&mut self, rt: &GpuRuntime) -> Result<()> {
        let b = self.matmul_buf_base;
        if self.b_warm && self.n > 64 {
            set_lowdin_gemm(&self.k_matmul_xh, 1, 0, &self.c, &self.h_scc, &self.temp)?;   // temp = cᵀ·H_scc
            unsafe { self.k_matmul_xh.enq().map_err(map_ocl_err)?; }
            set_matmul_args(&self.k_matmul_tx, b, &self.temp, &self.c, &self.hp)?;        // hp = temp·c
            unsafe { self.k_matmul_tx.enq().map_err(map_ocl_err)?; }
            self.k_jacobi.set_arg(0u32, &self.hp).map_err(map_ocl_err)?;
            self.k_jacobi.set_arg(1u32, &self.c).map_err(map_ocl_err)?;
            self.k_jacobi.set_arg(4u32, 1i32).map_err(map_ocl_err)?;
            unsafe { self.k_jacobi.enq().map_err(map_ocl_err)?; }
        } else {
            if self.n > 64 {
                // tiled batched_gemm — trans args exist; restore trans_a=0
                // (a previous warm iteration may have left it at 1)
                set_lowdin_gemm(&self.k_matmul_xh, 0, 0, &self.x_t, &self.h_scc, &self.temp)?;
            } else {
                // matmul_full_local_batched has no trans args (x_t pretransposed)
                set_matmul_args(&self.k_matmul_xh, b, &self.x_t, &self.h_scc, &self.temp)?;
            }
            unsafe { self.k_matmul_xh.enq().map_err(map_ocl_err)?; }
            set_matmul_args(&self.k_matmul_tx, b, &self.temp, &self.x_buf, &self.hp)?;
            unsafe { self.k_matmul_tx.enq().map_err(map_ocl_err)?; }
            self.k_jacobi.set_arg(0u32, &self.hp).map_err(map_ocl_err)?;
            self.k_jacobi.set_arg(1u32, &self.cp).map_err(map_ocl_err)?;
            if self.n > 64 { self.k_jacobi.set_arg(4u32, 0i32).map_err(map_ocl_err)?; }
            unsafe { self.k_jacobi.enq().map_err(map_ocl_err)?; }
        }
        self.k_extract_diag.set_arg(2u32, &self.hp).map_err(map_ocl_err)?;
        self.k_extract_diag.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        unsafe { self.k_extract_diag.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Eigenvector production after `eigh_solve`+`occupation`. Warm: c is
    /// already the eigenbasis — S-metric column renorm arrests f32 drift.
    /// Cold: renorm occupied C′ columns then c = X·C′; marks b_warm so the
    /// next solve at the same geometry can take the warm path.
    fn eigh_finish(&mut self, rt: &GpuRuntime, s_buf: &Buffer<f32>, renorm: bool) -> Result<()> {
        let b = self.matmul_buf_base;
        if self.b_warm && self.n > 64 {
            if renorm {
                self.k_occ_snorm.set_arg(3u32, s_buf).map_err(map_ocl_err)?;
                unsafe { self.k_occ_snorm.enq().map_err(map_ocl_err)?; }
            }
        } else {
            if renorm {
                unsafe { self.k_occ_renorm.enq().map_err(map_ocl_err)?; }
            }
            set_matmul_args(&self.k_matmul_xc, b, &self.x_buf, &self.cp, &self.c)?;
            unsafe { self.k_matmul_xc.enq().map_err(map_ocl_err)?; }
            self.b_warm = self.n > 64;
        }
        Ok(())
    }

    /// Metric-repair the warm AO basis B (=c) against a new overlap S:
    ///   G = BᵀSB, B ← B·(3I−G)/2  — same Newton step as the Löwdin repair,
    /// looped up to LOWDIN_REUSE_MAX times (quadratic convergence: a
    /// metric defect ~0.3 needs all 3 steps to reach 1e-5).
    /// Returns max ‖BᵀSB−I‖∞ after the accepted steps.
    fn repair_basis_c(&mut self, rt: &GpuRuntime, s_buf: &Buffer<f32>) -> Result<f32> {
        let mut e1m = 0.0f32;
        for _ in 0..Self::LOWDIN_REUSE_MAX {
            unsafe {
                set_lowdin_gemm(&self.k_lowdin_gemm, 1, 0, &self.c, s_buf, &self.lowdin_t)?;            // T = BᵀS
                self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
                set_lowdin_gemm(&self.k_lowdin_gemm, 0, 0, &self.lowdin_t, &self.c, &self.lowdin_m)?;   // G = T·B
                self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
                self.k_metric.set_arg(3u32, &self.lowdin_e0).map_err(map_ocl_err)?;
                self.k_metric.enq().map_err(map_ocl_err)?;                                            // e0 = max|G−I|
                self.k_lowdin_qf.enq().map_err(map_ocl_err)?;                                         // Q = (3I−G)/2
                set_lowdin_gemm(&self.k_lowdin_gemm, 0, 0, &self.c, &self.lowdin_q, &self.lowdin_t)?;   // B1 = B·Q
                self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
                set_lowdin_gemm(&self.k_lowdin_gemm, 1, 0, &self.lowdin_t, s_buf, &self.lowdin_q)?;     // T2 = B1ᵀS
                self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
                set_lowdin_gemm(&self.k_lowdin_gemm, 0, 0, &self.lowdin_q, &self.lowdin_t, &self.lowdin_m)?; // G1 = T2·B1
                self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
                self.k_metric.set_arg(3u32, &self.lowdin_e1).map_err(map_ocl_err)?;
                self.k_metric.enq().map_err(map_ocl_err)?;                                            // e1 = max|G1−I|
                self.k_lowdin_acc_c.enq().map_err(map_ocl_err)?;                                      // B ← B1 if e1<e0
            }
            rt.read_buffer(&self.lowdin_e0, &mut self.lowdin_e0_host)?;
            rt.read_buffer(&self.lowdin_e1, &mut self.lowdin_e1_host)?;
            e1m = 0.0;
            for b in 0..self.batch {
                let (e0, e1) = (self.lowdin_e0_host[b], self.lowdin_e1_host[b]);
                if !e0.is_finite() || !e1.is_finite() {
                    return Err(DftbError::InvalidInput(format!("warm-basis repair replica {b}: e0={e0} e1={e1} non-finite")));
                }
                // If the accept kernel rejected the step (e1≥e0), c is
                // unchanged and the effective residual is e0, not e1.
                e1m = e1m.max(if e1 < e0 { e1 } else { e0 });
            }
            if e1m < Self::LOWDIN_REUSE_TOL { break; }
        }
        Ok(e1m)
    }

    /// Update the Löwdin transform for a new geometry. Call this when the
    /// overlap matrix changes (e.g. between relaxation steps).
    /// Copies S into a workspace and overwrites `x_buf` in place — no Buffer/Kernel alloc.
    /// §12 D6: accept XᵀSX−I ∞-norm below this after Newton reuse — the same
    /// level the post-Jacobi polish reaches (≈3e-7), well under the 1e-5 that
    /// would measurably perturb eigenpairs.
    const LOWDIN_REUSE_TOL: f32 = 1e-5;
    /// Max Newton steps before falling back to full Jacobi(S). Quadratic
    /// convergence: 1 step ≈ (δS)², so 3 steps cover a decade of overlap drift.
    const LOWDIN_REUSE_MAX: usize = 3;

    pub fn set_geometry(&mut self, rt: &mut GpuRuntime, s_buf: &Buffer<f32>) -> Result<()> {
        // Lazy-X (2026-09-13): repair the warm AO basis B (=c) FIRST — when it
        // certifies (the common FIRE-step case), the whole Löwdin X path
        // (≤3 Newton iters of GEMMs+readbacks, possibly a cold Jacobi(S),
        // plus the Xᵀ transpose) is dead work because the warm eigensolve
        // never touches X. X is repaired/rebuilt ONLY on the cold path —
        // i.e. when B repair fails or was never warm. This is not a silent
        // fallback: the cold path is the certified primary route; the warm
        // basis is an accelerator that must pass the same metric check.
        let mut b_ok = false;
        if self.b_warm {
            let e = self.repair_basis_c(rt, s_buf)?;
            if e < Self::LOWDIN_REUSE_TOL {
                eprintln!("[GpuSccPlan] warm basis metric-repaired ‖BᵀSB−I‖∞={e:.2e}");
                b_ok = true;
            } else {
                eprintln!("[GpuSccPlan] warm basis repair failed ‖BᵀSB−I‖∞={e:.2e} — next solve cold");
                self.b_warm = false;
            }
        }
        if !b_ok {
            // D6: warm X from the previous geometry — Newton-polish it against
            // the new S (M = XᵀSX is already ≈I for small moves). Rebuild X
            // from a full Jacobi(S) only when the metric refuses to certify.
            let mut reused = false;
            if self.x_warm && self.x_reuse {
                for _ in 0..Self::LOWDIN_REUSE_MAX {
                    let e1 = self.repair_lowdin_gpu(rt, s_buf)?;
                    if e1 < Self::LOWDIN_REUSE_TOL { reused = true; break; }
                }
                if !reused {
                    eprintln!("[GpuSccPlan] Löwdin X reuse exceeded tolerance after {} Newton steps — full Jacobi rebuild",
                        Self::LOWDIN_REUSE_MAX);
                }
            }
            if !reused {
                enqueue_sinv(
                    rt, s_buf, &self.s_work, self.batch * self.n * self.n,
                    &self.k_s_jacobi, self.k_s_invsqrt.as_ref(), self.k_s_scale.as_ref(), self.k_s_xgemm.as_ref(),
                )?;
                rt.read_buffer(&self.lambda_min, &mut self.lambda_min_host)?;
                check_overlap_lambda(&self.lambda_min_host)?;
                self.repair_lowdin_gpu(rt, s_buf)?;
                self.x_warm = true;
            }
            // D6: refresh x_t = Xᵀ — H' = Xᵀ·H·X must not rely on X symmetric.
            unsafe { self.k_transpose.enq().map_err(map_ocl_err)?; }
        }
        Ok(())
    }

    /// §12 D5: Löwdin X repair fully on GPU — replaces the CPU f64 serial-GEMM
    /// `repair_lowdin_x`. Five f32 `batched_gemm` calls:
    ///   T1 = Xᵀ·S, M = T1·X, X1 = X·(3I−M)/2, T2 = X1ᵀ·S, M1 = T2·X1.
    /// Accept per system where ‖M1−I‖∞ < ‖M−I‖∞; only 2·batch floats read back.
    /// The kernel accepts only on strict improvement, so an already-converged
    /// or NaN M1 leaves X untouched (fail-safe, no silent write).
    /// Returns max‖XᵀSX−I‖∞ after the accepted step (or e0 if rejected).
    fn repair_lowdin_gpu(&mut self, rt: &GpuRuntime, s_buf: &Buffer<f32>) -> Result<f32> {
        unsafe {
            set_lowdin_gemm(&self.k_lowdin_gemm, 1, 0, &self.x_buf, s_buf, &self.lowdin_t)?;  // T1 = XᵀS
            self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
            set_lowdin_gemm(&self.k_lowdin_gemm, 0, 0, &self.lowdin_t, &self.x_buf, &self.lowdin_m)?;  // M = T1·X
            self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
            self.k_metric.set_arg(3u32, &self.lowdin_e0).map_err(map_ocl_err)?;
            self.k_metric.enq().map_err(map_ocl_err)?;                                       // e0 = max|M−I|
            self.k_lowdin_qf.enq().map_err(map_ocl_err)?;                                    // Q = (3I−M)/2
            set_lowdin_gemm(&self.k_lowdin_gemm, 0, 0, &self.x_buf, &self.lowdin_q, &self.lowdin_t)?;  // X1 = X·Q
            self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
            set_lowdin_gemm(&self.k_lowdin_gemm, 1, 0, &self.lowdin_t, s_buf, &self.lowdin_q)?;        // T2 = X1ᵀS
            self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
            set_lowdin_gemm(&self.k_lowdin_gemm, 0, 0, &self.lowdin_q, &self.lowdin_t, &self.lowdin_m)?; // M1 = T2·X1
            self.k_lowdin_gemm.enq().map_err(map_ocl_err)?;
            self.k_metric.set_arg(3u32, &self.lowdin_e1).map_err(map_ocl_err)?;
            self.k_metric.enq().map_err(map_ocl_err)?;                                       // e1 = max|M1−I|
            self.k_lowdin_acc.enq().map_err(map_ocl_err)?;                                   // X ← X1 if e1<e0
        }
        rt.read_buffer(&self.lowdin_e0, &mut self.lowdin_e0_host)?;
        rt.read_buffer(&self.lowdin_e1, &mut self.lowdin_e1_host)?;
        let mut e0m = 0.0f32;
        let mut e1m = 0.0f32;
        let mut nskip = 0usize;
        for b in 0..self.batch {
            let (e0, e1) = (self.lowdin_e0_host[b], self.lowdin_e1_host[b]);
            if !e0.is_finite() || !e1.is_finite() {
                return Err(DftbError::InvalidInput(format!(
                    "Löwdin Newton replica {b}: residual non-finite e0={e0} e1={e1}"
                )));
            }
            e0m = e0m.max(e0);
            e1m = e1m.max(e1);
            if !(e1 < e0) { nskip += 1; }
        }
        eprintln!("[GpuSccPlan] Löwdin Newton max||XᵀSX−I|| {e0m:.3e} → {e1m:.3e} wrote={} skipped={nskip}", nskip < self.batch);
        Ok(if nskip < self.batch { e1m } else { e0m })
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

    /// Bind the repulsive-energy kernel's coord arg to a persistent buffer
    /// (e.g. `buf_coords_bohr`) — removes the per-geometry rep_coords upload.
    pub fn bind_rep_energy_coords(&mut self, coords: &Buffer<f32>) -> Result<()> {
        if let Some(ref mut k) = self.k_rep_energy {
            k.set_arg(2u32, coords).map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Update coordinates for repulsive energy evaluation (call when
    /// geometry changes, before compute_energy). DEPRECATED when the coord
    /// arg is bound to a persistent buffer via `bind_rep_energy_coords`.
    #[allow(dead_code)]
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
        // 1-3. Δq → V=γΔq → H_scc = H0+½S(V_A+V_B) — one fused launch (D10/R18)
        self.enq_dq_v_hscc(h0_buf, s_buf, g_buf, q0_buf, orb_atom_buf)?;

        // 4-6. Eigenproblem: warm A=cᵀH_scc c + in-place Jacobi, or cold
        //      A=XᵀH_sccX + Jacobi(V=I) → cp. Eigenvalues on diag(hp).
        self.eigh_solve(rt)?;
        let n_den = self.occupation(rt, n_occ)?;

        // 6b-7. Column renorm (S-metric on c warm / plain on cp cold) and
        //       cold-path c = X·cp.
        self.eigh_finish(rt, s_buf, self.occ_repair_scc)?;

        // 8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T  (D10: occ_idx + triangle)
        self.k_density.set_arg(2u32, n_den).map_err(map_ocl_err)?;
        self.k_density.set_arg(3u32, &self.c).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, &self.d).map_err(map_ocl_err)?;
        self.k_density.set_arg(6u32, 0i32).map_err(map_ocl_err)?;
        unsafe { self.k_density.enq().map_err(map_ocl_err)?; }

        // 9. q_new = Mulliken(D, S)
        self.k_mulliken.set_arg(3u32, &self.d).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(5u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(6u32, &self.q_new).map_err(map_ocl_err)?;
        unsafe { self.k_mulliken.enq().map_err(map_ocl_err)?; }

        // §12 D4 + B1: ρ on the just-solved state so the A1 cached-state
        // energy is correct without a second eigensolve. Smeared runs need
        // ρ on every weighted column (use_w).
        if self.occ_repair || self.kT > 0.0 {
            self.k_occ_rayleigh.set_arg(5u32, s_buf).map_err(map_ocl_err)?;
            self.k_occ_rayleigh.set_arg(10u32, (self.kT > 0.0) as i32).map_err(map_ocl_err)?;
            unsafe { self.k_occ_rayleigh.enq().map_err(map_ocl_err)?; }
        }

        // 10. residual + mix → q_next (host commits to q_gpu only for active replicas)
        self.k_residual_mix.set_arg(2u32, alpha).map_err(map_ocl_err)?;
        self.k_residual_mix.set_arg(3u32, &self.q_new).map_err(map_ocl_err)?;
        self.k_residual_mix.set_arg(4u32, &self.q_gpu).map_err(map_ocl_err)?;
        self.k_residual_mix.set_arg(5u32, &self.q_next).map_err(map_ocl_err)?;
        self.k_residual_mix.set_arg(6u32, &self.rms).map_err(map_ocl_err)?;
        unsafe { self.k_residual_mix.enq().map_err(map_ocl_err)?; }

        // Read RMS and return max
        rt.read_buffer(&self.rms, &mut self.rms_host)?;
        max_finite_f32(&self.rms_host, "SCC rms")
    }

    /// Occupation selection after `extract_diag`: integer sort by default,
    /// or Fermi smearing (kT>0): per-replica chemical potential via host
    /// bisection on eig_diag (Σ_k f_k = n_occ), weights uploaded to occ_w.
    /// Returns the density-kernel loop bound (n_occ, or n when smeared —
    /// all orbitals enter with their Fermi weight).
    fn occupation(&mut self, rt: &mut GpuRuntime, n_occ: usize) -> Result<i32> {
        self.k_select_occ.set_arg(1u32, n_occ as i32).map_err(map_ocl_err)?;
        self.k_select_occ.set_arg(3u32, &self.eig_diag).map_err(map_ocl_err)?;
        self.k_select_occ.set_arg(4u32, &self.occ_mask).map_err(map_ocl_err)?;
        unsafe { self.k_select_occ.enq().map_err(map_ocl_err)?; }
        if self.kT <= 0.0 {
            self.k_density.set_arg(8u32, 0i32).map_err(map_ocl_err)?;
            return Ok(n_occ as i32);
        }
        // smearing: host bisection on eig_diag — Σ_k 1/(1+exp((ε−μ)/kT)) = n_occ
        rt.read_buffer(&self.eig_diag, &mut self.eig_diag_host)?;
        let n = self.n;
        let kt = self.kT as f64;
        for b in 0..self.batch {
            let e = &self.eig_diag_host[b * n..(b + 1) * n];
            let mut lo = e.iter().fold(f64::INFINITY, |a, &v| a.min(v as f64)) - 32.0 * kt;
            let mut hi = e.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64)) + 32.0 * kt;
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                let s: f64 = e.iter().map(|&ek| 1.0 / (1.0 + (((ek as f64) - mid) / kt).exp())).sum();
                // s(μ) is increasing in μ: too many electrons → μ too high
                if s > n_occ as f64 { hi = mid } else { lo = mid }
            }
            let mu = 0.5 * (lo + hi);
            if !mu.is_finite() {
                return Err(DftbError::InvalidInput(format!("Fermi smearing: μ non-finite replica {b} (kT={})", self.kT)));
            }
            for k in 0..n {
                let x = ((e[k] as f64) - mu) / kt;
                self.occ_w_host[b * n + k] = (1.0 / (1.0 + x.exp().min(1e300))) as f32;
            }
        }
        self.occ_w.write(&self.occ_w_host).enq().map_err(map_ocl_err)?;
        self.k_density.set_arg(8u32, 1i32).map_err(map_ocl_err)?;
        Ok(n as i32)
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
        // Steps 1-9: same as scc_step (Δq → V → H_scc → eigh → occ → C → D → q_new)
        self.enq_dq_v_hscc(h0_buf, s_buf, g_buf, q0_buf, orb_atom_buf)?;

        self.eigh_solve(rt)?;
        let n_den = self.occupation(rt, n_occ)?;
        self.eigh_finish(rt, s_buf, self.occ_repair_scc)?;

        self.k_density.set_arg(2u32, n_den).map_err(map_ocl_err)?;
        self.k_density.set_arg(3u32, &self.c).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, &self.d).map_err(map_ocl_err)?;
        self.k_density.set_arg(6u32, 0i32).map_err(map_ocl_err)?;
        unsafe { self.k_density.enq().map_err(map_ocl_err)?; }

        self.k_mulliken.set_arg(3u32, &self.d).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(5u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(6u32, &self.q_new).map_err(map_ocl_err)?;
        unsafe { self.k_mulliken.enq().map_err(map_ocl_err)?; }

        // §12 D4 + B1: ρ on the just-solved state (see scc_step).
        if self.occ_repair || self.kT > 0.0 {
            self.k_occ_rayleigh.set_arg(5u32, s_buf).map_err(map_ocl_err)?;
            self.k_occ_rayleigh.set_arg(10u32, (self.kT > 0.0) as i32).map_err(map_ocl_err)?;
            unsafe { self.k_occ_rayleigh.enq().map_err(map_ocl_err)?; }
        }

        // 10. GPU-side DIIS mixing (R9b + D9: Δq-anchored, drop-oldest retry)
        self.k_diis.set_arg(2u32, alpha).map_err(map_ocl_err)?;
        self.k_diis.set_arg(3u32, &self.q_new).map_err(map_ocl_err)?;
        self.k_diis.set_arg(4u32, &self.q_gpu).map_err(map_ocl_err)?;
        self.k_diis.set_arg(5u32, q0_buf).map_err(map_ocl_err)?;
        // History buffers and workspace already bound at construction
        unsafe { self.k_diis.enq().map_err(map_ocl_err)?; }

        // Read RMS and return max (only scalar readback)
        rt.read_buffer(&self.rms, &mut self.rms_host)?;
        max_finite_f32(&self.rms_host, "DIIS rms")
    }

    /// Upload `active_host` to the `active` device mask. 1 = still iterating,
    /// 0 = done/frozen (mixer + commit early-out on it).
    pub fn set_active(&mut self, rt: &GpuRuntime) -> Result<()> {
        self.active.write(&self.active_host).enq().map_err(map_ocl_err)?;
        Ok(())
    }

    /// Mark every replica active on the device mask. All SCC-loop kernels
    /// gate on `active`; paths outside the SCC loop (finalize/eval/forces/
    /// measure) must run for every replica — call this first.
    pub fn activate_all(&mut self, rt: &GpuRuntime) -> Result<()> {
        for f in self.active_host.iter_mut() { *f = 1; }
        self.set_active(rt)
    }

    /// Commit q_next → q_gpu for active replicas only. Done replicas keep the
    /// just-solved input charge so the device state (C,D,H_scc,Δq,V) stays
    /// consistent with q_gpu after the SCC loop exits.
    pub fn commit_q_next(&mut self, rt: &GpuRuntime) -> Result<()> {
        unsafe { self.k_commit.enq().map_err(map_ocl_err)?; }
        Ok(())
    }

    /// Reset DIIS history (call when geometry changes or for warm start).
    pub fn reset_diis(&mut self, rt: &GpuRuntime) -> Result<()> {
        let batch = self.batch;
        // Zero out buf_idx, n_filled, and the fallback status counters
        let zeros_i = vec![0i32; batch];
        self.diis_buf_idx.write(&zeros_i).enq().map_err(map_ocl_err)?;
        self.diis_n_filled.write(&zeros_i).enq().map_err(map_ocl_err)?;
        self.diis_flag.write(&zeros_i).enq().map_err(map_ocl_err)?;
        self.diis_reason.write(&zeros_i).enq().map_err(map_ocl_err)?;
        Ok(())
    }

    /// Read DIIS fallback counters set by the kernel (replaces printf, D9).
    /// Returns per-system (fallback_count, last_reason). Reason codes:
    /// 1 = pivot/scale, 2 = non-finite coefficient, 3 = |Σc−1| too large.
    pub fn diis_status(&mut self, rt: &GpuRuntime, flag_host: &mut [i32], reason_host: &mut [i32]) -> Result<()> {
        if flag_host.len() != self.batch || reason_host.len() != self.batch {
            return Err(DftbError::InvalidInput(format!(
                "diis_status: host len {}/{} != batch {}", flag_host.len(), reason_host.len(), self.batch
            )));
        }
        rt.read_buffer(&self.diis_flag, flag_host)?;
        rt.read_buffer(&self.diis_reason, reason_host)?;
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
        // §12 D4: with occ_repair the weight is the Rayleigh quotient ρ_k of
        // the renormalized vectors (eig_rho), not the drifted Jacobi diagonal.
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
        rt.read_buffer(&self.eig_rho, &mut self.eig_rho_host)?;
        rt.read_buffer(&self.occ_mask, &mut self.mask_host)?;
        let mut dqv = vec![0.0f32; batch];
        let mut q0v = vec![0.0f32; batch];
        rt.read_buffer(&self.dot, &mut dqv)?;
        rt.read_buffer(&self.tr, &mut q0v)?;

        let mut e = vec![0.0f64; batch];
        let kt = self.kT as f64;
        for bi in 0..batch {
            let mut e_band = 0.0f64;
            let mut mts = 0.0f64;   // −T·S Mermin term = 2kT·Σ[f ln f + (1−f)ln(1−f)] ≤ 0
            let base = bi * n;
            if self.kT > 0.0 {
                // Fermi smearing: F = E_band − TS. Weights f_k and Rayleigh
                // quotients ρ_k cover ALL orbitals (fractional frontier too).
                for k in 0..n {
                    let f = self.occ_w_host[base + k] as f64;
                    let rho = self.eig_rho_host[base + k] as f64;
                    e_band += 2.0 * f * rho;
                    let g = 1.0 - f;
                    if f > 1e-300 { mts += f * f.ln(); }
                    if g > 1e-300 { mts += g * g.ln(); }
                }
                e[bi] = e_band + 2.0 * kt * mts - 0.5 * dqv[bi] as f64 - q0v[bi] as f64;
            } else {
                for k in 0..n {
                    if self.mask_host[base + k] != 0 {
                        e_band += 2.0 * (if self.occ_repair { self.eig_rho_host[base + k] } else { self.eig_diag_host[base + k] }) as f64;
                    }
                }
                e[bi] = e_band - 0.5 * dqv[bi] as f64 - q0v[bi] as f64;
            }
            if !e[bi].is_finite() {
                return Err(DftbError::InvalidInput(format!("energy_from_state: E[{bi}]={} band={e_band} −TS={} Δq·V={} q0·V={}", e[bi], 2.0 * kt * mts, dqv[bi], q0v[bi])));
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
        // Finalize runs OUTSIDE the SCC loop — all replicas must be solved
        // at their q_gpu, including ones that were frozen at loop exit.
        self.activate_all(rt)?;

        // 1-3. Δq → V=γΔq → H_scc — one fused launch (D10/R18)
        self.enq_dq_v_hscc(h0_buf, s_buf, g_buf, q0_buf, orb_atom_buf)?;

        // 4-6. Eigenproblem (warm cᵀHc or cold XᵀHX) → eigenvalues on
        //      diag(hp), eigenvectors in c (warm, in place) or cp (cold).
        self.eigh_solve(rt)?;
        let n_den = self.occupation(rt, n_occ)?;

        // 6b-7. §12 D3: column renorm — S-metric on c (warm) or plain on
        //       C′ (cold, then c = X·C′). Repairs the Jacobi normality loss.
        self.eigh_finish(rt, s_buf, self.occ_repair)?;

        // 7b. §12 D4: ρ_k = cᵀH_scc c_k / cᵀSc_k on the renormalized AO
        //     eigenvectors — replaces the drifted Jacobi diagonal ε_k as the
        //     weight in E_band and W. Under Fermi smearing ρ is needed for
        //     ALL fractionally-weighted orbitals, not only the integer-occ set.
        if self.occ_repair || self.kT > 0.0 {
            self.k_occ_rayleigh.set_arg(5u32, s_buf).map_err(map_ocl_err)?;
            self.k_occ_rayleigh.set_arg(10u32, (self.kT > 0.0) as i32).map_err(map_ocl_err)?;
            unsafe { self.k_occ_rayleigh.enq().map_err(map_ocl_err)?; }
        }

        // 8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T  (D10: occ_idx + triangle)
        self.k_density.set_arg(2u32, n_den).map_err(map_ocl_err)?;
        self.k_density.set_arg(3u32, &self.c).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, &self.d).map_err(map_ocl_err)?;
        self.k_density.set_arg(6u32, 0i32).map_err(map_ocl_err)?;
        unsafe { self.k_density.enq().map_err(map_ocl_err)?; }

        // q_D from this D (q_gpu is still q_in). Do not mix.
        self.k_mulliken.set_arg(3u32, &self.d).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(4u32, s_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(5u32, orb_atom_buf).map_err(map_ocl_err)?;
        self.k_mulliken.set_arg(6u32, &self.q_new).map_err(map_ocl_err)?;
        unsafe { self.k_mulliken.enq().map_err(map_ocl_err)?; }

        Ok(())
    }

    /// W = 2 Σ_{k occ} w_k C_k C_kᵀ into `out`. Same kernel as D (`use_eig=1`).
    /// Weight is ρ_k (Rayleigh quotient of the stored vectors, §12 D4) when
    /// `occ_repair` is on, else the raw Jacobi diagonal ε_k. Restores D args after.
    pub fn build_edm(&mut self, out: &Buffer<f32>, n_occ: usize) -> Result<()> {
        let n_den = if self.kT > 0.0 { self.n as i32 } else { n_occ as i32 };
        self.k_density.set_arg(2u32, n_den).map_err(map_ocl_err)?;
        self.k_density.set_arg(3u32, &self.c).map_err(map_ocl_err)?;
        self.k_density.set_arg(5u32, out).map_err(map_ocl_err)?;
        self.k_density.set_arg(6u32, 1i32).map_err(map_ocl_err)?;
        let w = if self.occ_repair || self.kT > 0.0 { &self.eig_rho } else { &self.eig_diag };
        self.k_density.set_arg(7u32, w).map_err(map_ocl_err)?;
        let enq = unsafe { self.k_density.enq().map_err(map_ocl_err) };
        self.k_density.set_arg(5u32, &self.d).map_err(map_ocl_err)?;
        self.k_density.set_arg(6u32, 0i32).map_err(map_ocl_err)?;
        self.k_density.set_arg(7u32, &self.eig_diag).map_err(map_ocl_err)?;
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
    x_t_buf: &Buffer<f32>,
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
            .arg(x_t_buf).arg(h_scc).arg(temp)
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
            .arg(x_t_buf).arg(h_scc).arg(temp)
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
    act_buf: &Buffer<i32>,
    prec: u32,
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
        let source = tiled_render_source(b, wg, prec);
        let program = rt.build_program(&source)?;
        // Direct cyclic Jacobi (replaces the ~16k-barrier tiled path).
        // args: [0]=A [1]=V [2]=n [3]=batch [4]=init_v (0=V←I cold, 1=warm)
        // [5]=active mask — done replicas' WGs exit immediately
        Kernel::builder()
            .program(&program).name("jacobi_cyclic_global_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(a_buf).arg(v_buf).arg(n as i32).arg(batch as i32).arg(0i32)
            .arg(act_buf)
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

/// Re-bind a `batched_gemm` handle for the Löwdin repair GEMM sequence.
/// Buffers/scalars at indices: [2]=trans_a [3]=trans_b [6]=A [7]=B [8]=C.
fn set_lowdin_gemm(k: &Kernel, ta: i32, tb: i32, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<()> {
    k.set_arg(2u32, ta).map_err(map_ocl_err)?;
    k.set_arg(3u32, tb).map_err(map_ocl_err)?;
    k.set_arg(6u32, a).map_err(map_ocl_err)?;
    k.set_arg(7u32, b).map_err(map_ocl_err)?;
    k.set_arg(8u32, c).map_err(map_ocl_err)?;
    Ok(())
}

/// One Newton step on X: M=XᵀSX=I+E → X ← X(I−E/2), then symmetrize.
/// Host f64, once per geometry. Skips writeback if it does not reduce max|E|.
///
/// CPU reference for `GpuSccPlan::repair_lowdin_gpu` (§12 D5) — retained for
/// parity checks; not called in production (O(batch·N³) serial f64 on host).
#[allow(dead_code)]
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

#[allow(dead_code)]
fn gemm_nn(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n { s += a[i * n + k] * b[k * n + j]; }
            c[i * n + j] = s;
        }
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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
