//! Device-resident SCC loop driver (Wave 3 coordinator integration).
//!
//! Wires Agent_4's Jacobi + S^{-1/2} and Agent_6's GEMM + SCC kernels
//! into a complete device-resident SCC loop. The only PCIe traffic
//! during SCC iterations is reading `batch` RMS residuals per iteration
//! (negligible) and uploading `batch*N` occupation mask ints (negligible).
//!
//! # Algorithm (per SCC iteration, all on device)
//! 1. Δq = q − q0
//! 2. V = G · Δq  (gamma matvec)
//! 3. H_scc = H0 + 0.5·S·(V_i + V_j)  (elementwise)
//! 4. H' = X·H_scc·X  (2 GEMMs, X = S^{-1/2} precomputed once)
//! 5. Jacobi(H') → ε (diag), C' (eigenvectors)  (Brent-Luk)
//! 6. Read diag(H') → sort on host → build occ_mask → upload
//! 7. C = X·C'  (back-transform, 1 GEMM)
//! 8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T  (masked density)
//! 9. q_new = Mulliken(D, S)
//! 10. residual + simple mix → q_next
//!
//! After convergence: E = Tr(D·H0) + 0.5·Σ Δq·V
//!
//! # Limitations (first implementation)
//! - Simple mixing only (no DIIS). Converges slower than CPU DIIS path.
//! - Kernel objects are rebuilt each call (program cache hit, but Kernel
//!   handle creation per iteration is a known performance TODO).
//! - Occupation mask is built on host each iteration (N*batch floats read
//!   back + N*batch ints uploaded — negligible for N≤64, batch≤1000).

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_eigen::{build_inv_sqrt_batched, jacobi_batched};
use crate::qmqm::gpu_matrix::{
    build_density_masked_batched, delta_q_batched, dot_batched, extract_diagonal_batched,
    frobenius_trace_batched, gamma_matvec_batched, h_scc_update_batched,
    matmul_batched, mulliken_charges_batched, residual_and_mix_batched,
};
use crate::qmqm::gpu_runtime::GpuRuntime;
use ocl::Buffer;

/// Result of a batched GPU SCC solve.
#[derive(Debug, Clone)]
pub struct GpuSccResult {
    /// Total SCC energy per system (electronic only, Hartree).
    pub energies: Vec<f32>,
    /// Final Mulliken charges (electron population) per atom, `[batch*Na]`.
    pub charges: Vec<f32>,
    /// Eigenvalues per orbital, `[batch*N]` (ascending order within each system).
    pub eigenvalues: Vec<f32>,
    /// Number of SCC iterations performed.
    pub n_iters: usize,
    /// Final RMS residual per system, `[batch]`.
    pub rms: Vec<f32>,
    /// Optional timing breakdown (populated when `RUST_DFTB_TIMING=1`).
    pub timing: Option<GpuSccTiming>,
}

/// Per-phase wall-clock timing for one SCC solve (seconds).
///
/// Populated when the `RUST_DFTB_TIMING` environment variable is set.
/// All times are cumulative across all SCC iterations.
#[derive(Debug, Clone, Default)]
pub struct GpuSccTiming {
    /// S^{-1/2} precomputation (Jacobi + reconstruction, once per geometry).
    pub t_inv_sqrt: f64,
    /// Buffer allocation (once, outside SCC loop).
    pub t_alloc: f64,
    /// Per-iteration: Δq = q − q0 (cumulative across all iters).
    pub t_delta_q: f64,
    /// Per-iteration: V = G · Δq (cumulative).
    pub t_gamma_matvec: f64,
    /// Per-iteration: H_scc = H0 + 0.5·S·(V_i+V_j) (cumulative).
    pub t_h_scc_update: f64,
    /// Per-iteration: H' = X·H_scc·X, 2 GEMMs (cumulative).
    pub t_gemm: f64,
    /// Per-iteration: Jacobi eigensolver (cumulative).
    pub t_jacobi: f64,
    /// Per-iteration: read diag(H') + host sort + occ_mask upload (cumulative).
    pub t_occ_sort: f64,
    /// Per-iteration: C = X·C' back-transform GEMM (cumulative).
    pub t_back_gemm: f64,
    /// Per-iteration: D = 2·Σ occ C[:,k]C[:,k]^T (cumulative).
    pub t_density: f64,
    /// Per-iteration: q_new = Mulliken(D, S) (cumulative).
    pub t_mulliken: f64,
    /// Per-iteration: CPU DIIS mixing + readback + upload (cumulative).
    pub t_diis_mix: f64,
    /// Post-convergence: energy computation (Tr(D·H0) + 0.5·ΣΔq·V).
    pub t_energy: f64,
    /// Post-convergence: readback of energies, charges, eigenvalues.
    pub t_readback: f64,
    /// Total wall-clock time for the entire `gpu_solve_scc_batched_diis_warmstart` call.
    pub t_total: f64,
}

impl GpuSccTiming {
    /// Print a formatted timing table to stderr.
    pub fn print(&self, label: &str, n: usize, batch: usize, n_iters: usize) {
        eprintln!("  [timing] {label} (N={n}, batch={batch}, iters={n_iters}):");
        eprintln!("    S^{{-1/2}} precompute : {:.4} s", self.t_inv_sqrt);
        eprintln!("    alloc buffers       : {:.4} s", self.t_alloc);
        eprintln!("    --- per-iter (cumulative, {} iters) ---", n_iters);
        eprintln!("    delta_q             : {:.4} s  ({:.4} ms/iter)", self.t_delta_q, self.t_delta_q / n_iters as f64 * 1e3);
        eprintln!("    gamma_matvec        : {:.4} s  ({:.4} ms/iter)", self.t_gamma_matvec, self.t_gamma_matvec / n_iters as f64 * 1e3);
        eprintln!("    h_scc_update        : {:.4} s  ({:.4} ms/iter)", self.t_h_scc_update, self.t_h_scc_update / n_iters as f64 * 1e3);
        eprintln!("    gemm (2×)           : {:.4} s  ({:.4} ms/iter)", self.t_gemm, self.t_gemm / n_iters as f64 * 1e3);
        eprintln!("    jacobi              : {:.4} s  ({:.4} ms/iter)", self.t_jacobi, self.t_jacobi / n_iters as f64 * 1e3);
        eprintln!("    occ_sort+upload     : {:.4} s  ({:.4} ms/iter)", self.t_occ_sort, self.t_occ_sort / n_iters as f64 * 1e3);
        eprintln!("    back_gemm           : {:.4} s  ({:.4} ms/iter)", self.t_back_gemm, self.t_back_gemm / n_iters as f64 * 1e3);
        eprintln!("    density             : {:.4} s  ({:.4} ms/iter)", self.t_density, self.t_density / n_iters as f64 * 1e3);
        eprintln!("    mulliken            : {:.4} s  ({:.4} ms/iter)", self.t_mulliken, self.t_mulliken / n_iters as f64 * 1e3);
        eprintln!("    diis_mix (CPU)      : {:.4} s  ({:.4} ms/iter)", self.t_diis_mix, self.t_diis_mix / n_iters as f64 * 1e3);
        eprintln!("    --- post-convergence ---");
        eprintln!("    energy              : {:.4} s", self.t_energy);
        eprintln!("    readback            : {:.4} s", self.t_readback);
        eprintln!("    TOTAL               : {:.4} s", self.t_total);
        let per_iter = (self.t_delta_q + self.t_gamma_matvec + self.t_h_scc_update
            + self.t_gemm + self.t_jacobi + self.t_occ_sort + self.t_back_gemm
            + self.t_density + self.t_mulliken + self.t_diis_mix) / n_iters as f64;
        eprintln!("    per-iter total      : {:.4} ms", per_iter * 1e3);
        let jacobi_pct = if per_iter > 0.0 { self.t_jacobi / n_iters as f64 / per_iter * 100.0 } else { 0.0 };
        eprintln!("    jacobi fraction     : {:.1}%", jacobi_pct);
    }
}

fn timing_enabled() -> bool {
    std::env::var("RUST_DFTB_TIMING").is_ok()
}

macro_rules! timed {
    ($tm:expr, $field:ident, $body:expr) => {{
        let _t0 = std::time::Instant::now();
        let _r = $body;
        if let Some(tm) = $tm.as_mut() {
            tm.$field += _t0.elapsed().as_secs_f64();
        }
        _r
    }};
}

/// Solve the SCC fixed-point for a batch of homogeneous systems on GPU.
///
/// All systems must have the same N (orbital count), Na (atom count),
/// and n_occ (occupied MOs). Heterogeneous batches require grouping by
/// template first.
///
/// # Arguments
/// - `h0_buf`, `s_buf` — `[batch][N*N]` Hamiltonian/overlap (from gpu_assemble_batched)
/// - `g_buf` — `[batch][Na*Na]` dense gamma matrix (host-built, uploaded once)
/// - `q0_buf` — `[batch][Na]` reference (neutral valence) charges
/// - `orb_atom_buf` — `[batch][N]` orbital→atom index mapping
/// - `n` — orbital dimension (N ≤ 64)
/// - `n_atoms` — atoms per system
/// - `n_occ` — occupied MOs (homogeneous batch)
/// - `batch` — number of systems
/// - `max_iter`, `tol`, `alpha` — SCC convergence params (alpha = mixing fraction)
pub fn gpu_solve_scc_batched(
    rt: &mut GpuRuntime,
    h0_buf: &Buffer<f32>,
    s_buf: &Buffer<f32>,
    g_buf: &Buffer<f32>,
    q0_buf: &Buffer<f32>,
    orb_atom_buf: &Buffer<i32>,
    n: usize,
    n_atoms: usize,
    n_occ: usize,
    batch: usize,
    max_iter: usize,
    tol: f32,
    alpha: f32,
) -> Result<GpuSccResult> {
    // Phase 3: N>64 is now supported via tiled Jacobi + tiled GEMM + tiled S^{-1/2}.
    if n_occ > n {
        return Err(DftbError::InvalidInput(format!(
            "gpu_solve_scc_batched: n_occ={n_occ} > n={n}"
        )));
    }

    // --- Precompute X = S^{-1/2} (once per geometry) ---
    let (x_buf, _lambda_min) = build_inv_sqrt_batched(rt, s_buf, n, batch)?;

    // --- Allocate working buffers (once, outside SCC loop) ---
    let nn = n * n;
    // Start from neutral charges: q = q0
    let mut q0_host = vec![0.0f32; batch * n_atoms];
    rt.read_buffer(q0_buf, &mut q0_host)?;
    let q_a = rt.buffer_from_slice(&q0_host)?;
    let q_b = rt.zero_buffer::<f32>(batch * n_atoms)?;
    let q_new = rt.zero_buffer::<f32>(batch * n_atoms)?;
    let dq = rt.zero_buffer::<f32>(batch * n_atoms)?;
    let v = rt.zero_buffer::<f32>(batch * n_atoms)?;
    let h_scc = rt.zero_buffer::<f32>(batch * nn)?;
    let temp = rt.zero_buffer::<f32>(batch * nn)?;
    let hp = rt.zero_buffer::<f32>(batch * nn)?;
    let cp = rt.zero_buffer::<f32>(batch * nn)?;
    let c = rt.zero_buffer::<f32>(batch * nn)?;
    let d = rt.zero_buffer::<f32>(batch * nn)?;
    let rms = rt.zero_buffer::<f32>(batch)?;
    let tr = rt.zero_buffer::<f32>(batch)?;
    let dot = rt.zero_buffer::<f32>(batch)?;
    let occ_mask = rt.zero_buffer::<i32>(batch * n)?;
    let eig_diag_buf = rt.zero_buffer::<f32>(batch * n)?; // Phase 0d: diagonal extraction

    // Double-buffered charges: q_bufs[q_cur] = current, q_bufs[1-q_cur] = next mixed
    let q_bufs = [q_a, q_b];
    let mut q_cur = 0usize;

    // Host staging for eigenvalue extraction + occupation mask.
    // Phase 0d: extract only the diagonal (N*batch floats) instead of
    // reading the full N²*batch matrix.
    let mut eig_diag = vec![0.0f32; batch * n];
    let mut mask_host = vec![0i32; batch * n];

    // --- SCC loop ---
    let mut n_iters = 0;
    let mut converged = false;

    let verbose = std::env::var("RUST_DFTB_SCC_VERBOSE").is_ok();

    for iter in 0..max_iter {
        n_iters = iter + 1;

        // 1. Δq = q − q0
        delta_q_batched(rt, &q_bufs[q_cur], q0_buf, &dq, n_atoms, batch)?;

        // 2. V = G · Δq  (per-atom shift)
        gamma_matvec_batched(rt, g_buf, &dq, &v, n_atoms, batch)?;

        // 3. H_scc = H0 + 0.5·S·(V_i + V_j)
        h_scc_update_batched(rt, h0_buf, s_buf, &v, &h_scc, orb_atom_buf, n, n_atoms, batch)?;

        // 4. H' = X · H_scc · X  (2 GEMMs)
        matmul_batched(rt, &x_buf, &h_scc, &temp, n, batch)?;
        matmul_batched(rt, &temp, &x_buf, &hp, n, batch)?;

        // 5. Jacobi(H') → eigenvalues on diag(hp), eigenvectors in cp
        jacobi_batched(rt, &hp, &cp, n, batch)?;

        if verbose && iter < 5 {
            // Debug: read back charges and eigenvalues for first system
            let mut q_dbg = vec![0.0f32; n_atoms];
            rt.read_buffer(&q_bufs[q_cur], &mut q_dbg)?;
            let mut dq_dbg = vec![0.0f32; n_atoms];
            rt.read_buffer(&dq, &mut dq_dbg)?;
            let mut v_dbg = vec![0.0f32; n_atoms];
            rt.read_buffer(&v, &mut v_dbg)?;
            let mut eig_dbg = vec![0.0f32; n * n];
            rt.read_buffer(&hp, &mut eig_dbg)?;
            let diags: Vec<f32> = (0..n).map(|i| eig_dbg[i * n + i]).collect();
            eprintln!("    [gpu_scc] iter {iter}: q0={:?}", &q_dbg[..n_atoms.min(4)]);
            eprintln!("    [gpu_scc] iter {iter}: dq={:?}", &dq_dbg[..n_atoms.min(4)]);
            eprintln!("    [gpu_scc] iter {iter}: V ={:?}", &v_dbg[..n_atoms.min(4)]);
            eprintln!("    [gpu_scc] iter {iter}: eig(diag)={:?}", &diags[..n.min(6)]);
        }

        // 6. Extract diag(H') on GPU → read N*batch floats → sort → occ_mask → upload
        // Phase 0d: uses extract_diagonal_batched kernel instead of reading
        // the full N²*batch matrix. Reduces readback by factor of N.
        extract_diagonal_batched(rt, &hp, &eig_diag_buf, n, batch)?;
        rt.read_buffer(&eig_diag_buf, &mut eig_diag)?;
        for bi in 0..batch {
            let mut idxs: Vec<usize> = (0..n).collect();
            idxs.sort_unstable_by(|&a, &b| {
                eig_diag[bi * n + a]
                    .partial_cmp(&eig_diag[bi * n + b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (rank, &k) in idxs.iter().enumerate() {
                mask_host[bi * n + k] = if rank < n_occ { 1 } else { 0 };
            }
        }
        occ_mask
            .write(&mask_host)
            .enq()
            .map_err(crate::qmqm::gpu_runtime::map_ocl_err)?;

        // 7. C = X · C'
        matmul_batched(rt, &x_buf, &cp, &c, n, batch)?;

        // 8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T
        build_density_masked_batched(rt, &c, &occ_mask, &d, n, batch)?;

        // 9. q_new = Mulliken(D, S)
        mulliken_charges_batched(rt, &d, s_buf, &q_new, orb_atom_buf, n, n_atoms, batch)?;

        // 10. residual + mix
        residual_and_mix_batched(
            rt, &q_new, &q_bufs[q_cur], &q_bufs[1 - q_cur], &rms, alpha, n_atoms, batch,
        )?;

        // Check convergence (read batch floats — negligible)
        let mut rms_host = vec![0.0f32; batch];
        rt.read_buffer(&rms, &mut rms_host)?;
        let max_rms = rms_host.iter().fold(0.0f32, |m, &r| m.max(r));

        if max_rms < tol {
            converged = true;
            q_cur = 1 - q_cur; // switch to mixed charges
            break;
        }

        q_cur = 1 - q_cur; // switch to mixed charges for next iteration
    }

    if !converged {
        let mut rms_host = vec![0.0f32; batch];
        rt.read_buffer(&rms, &mut rms_host)?;
        let max_rms = rms_host.iter().fold(0.0f32, |m, &r| m.max(r));
        return Err(DftbError::SccNotConverged(format!(
            "GPU SCC did not converge in {max_iter} iterations (max RMS = {max_rms:.3e})"
        )));
    }

    // --- Energy computation (after convergence) ---
    // E_h0 = Tr(D · H0)
    frobenius_trace_batched(rt, &d, h0_buf, &tr, n, batch)?;

    // Recompute Δq and V at convergence for E_scc
    delta_q_batched(rt, &q_bufs[q_cur], q0_buf, &dq, n_atoms, batch)?;
    gamma_matvec_batched(rt, g_buf, &dq, &v, n_atoms, batch)?;
    // E_scc = 0.5 · Σ Δq · V
    dot_batched(rt, &dq, &v, &dot, n_atoms, batch)?;

    // --- Read back results ---
    let mut e_h0 = vec![0.0f32; batch];
    let mut e_scc = vec![0.0f32; batch];
    rt.read_buffer(&tr, &mut e_h0)?;
    rt.read_buffer(&dot, &mut e_scc)?;
    let energies: Vec<f32> = (0..batch).map(|i| e_h0[i] + 0.5 * e_scc[i]).collect();

    let mut charges = vec![0.0f32; batch * n_atoms];
    rt.read_buffer(&q_bufs[q_cur], &mut charges)?;

    // Eigenvalues: extract diagonal on GPU and sort ascending (Phase 0d)
    extract_diagonal_batched(rt, &hp, &eig_diag_buf, n, batch)?;
    rt.read_buffer(&eig_diag_buf, &mut eig_diag)?;
    let mut eigenvalues = vec![0.0f32; batch * n];
    for bi in 0..batch {
        let mut eigs: Vec<f32> = eig_diag[bi * n..(bi + 1) * n].to_vec();
        eigs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues[bi * n..(bi + 1) * n].copy_from_slice(&eigs);
    }

    let mut rms_final = vec![0.0f32; batch];
    rt.read_buffer(&rms, &mut rms_final)?;

    Ok(GpuSccResult {
        energies,
        charges,
        eigenvalues,
        n_iters,
        rms: rms_final,
        timing: None,
    })
}

/// Solve SCC with CPU-driven DIIS mixing.
///
/// Same GPU kernel pipeline as `gpu_solve_scc_batched`, but replaces the
/// simple mixer with an Anderson/DIIS mixer running on the CPU. Each iteration:
/// - GPU computes q_new (Mulliken charges) — device-resident
/// - Host reads back q_new and q_cur (batch*n_atoms floats — negligible transfer)
/// - Host runs per-system DIIS mixing using `DiisMixer`
/// - Host uploads mixed charges back to GPU
///
/// All heavy computation (gamma matvec, H_scc update, GEMM, Jacobi, density,
/// Mulliken) remains on the GPU. Only the small charge vectors (n_atoms per
/// system) cross the PCIe bus.
///
/// # Arguments
/// - `max_history` — DIIS history length (typically 6-10)
/// - `warmup` — number of simple-mixing iterations before DIIS kicks in
/// - `alpha` — simple mixing fraction for warmup/fallback
pub fn gpu_solve_scc_batched_diis(
    rt: &mut GpuRuntime,
    h0_buf: &Buffer<f32>,
    s_buf: &Buffer<f32>,
    g_buf: &Buffer<f32>,
    q0_buf: &Buffer<f32>,
    orb_atom_buf: &Buffer<i32>,
    n: usize,
    n_atoms: usize,
    n_occ: usize,
    batch: usize,
    max_iter: usize,
    tol: f32,
    alpha: f32,
    max_history: usize,
    warmup: usize,
) -> Result<GpuSccResult> {
    // Cold-start variant: initial charges = q0 (neutral valence)
    gpu_solve_scc_batched_diis_warmstart(
        rt, h0_buf, s_buf, g_buf, q0_buf, q0_buf, orb_atom_buf,
        n, n_atoms, n_occ, batch, max_iter, tol, alpha, max_history, warmup, false,
    )
}

/// Same as `gpu_solve_scc_batched_diis` but accepts separate initial charges
/// (`init_q_buf`) distinct from the reference charges (`q0_buf`). This enables
/// warm-starting from a neighbouring converged solution — essential for
/// asynchronous 2D scans where some geometries are too far from neutral to
/// converge from q0.
///
/// - `q0_buf` — reference (neutral valence) charges, used for Δq = q − q0 throughout
/// - `init_q_buf` — initial charges for the SCC iteration (may be warm-started)
/// - `best_effort` — if true, return results even if not all systems converged
///   (per-system RMS in `GpuSccResult.rms` indicates which converged)
pub fn gpu_solve_scc_batched_diis_warmstart(
    rt: &mut GpuRuntime,
    h0_buf: &Buffer<f32>,
    s_buf: &Buffer<f32>,
    g_buf: &Buffer<f32>,
    q0_buf: &Buffer<f32>,
    init_q_buf: &Buffer<f32>,
    orb_atom_buf: &Buffer<i32>,
    n: usize,
    n_atoms: usize,
    n_occ: usize,
    batch: usize,
    max_iter: usize,
    tol: f32,
    alpha: f32,
    max_history: usize,
    warmup: usize,
    best_effort: bool,
) -> Result<GpuSccResult> {
    use crate::qmqm::mixer::{DiisMixer, Mixer};

    // Phase 3: N>64 is now supported via tiled Jacobi + tiled GEMM + tiled S^{-1/2}.

    let do_timing = timing_enabled();
    let mut tm: Option<GpuSccTiming> = if do_timing { Some(GpuSccTiming::default()) } else { None };
    let t_total_start = std::time::Instant::now();

    // --- Precompute X = S^{-1/2} (once) ---
    let (x_buf, _) = timed!(tm, t_inv_sqrt, build_inv_sqrt_batched(rt, s_buf, n, batch)?);

    // --- Allocate working buffers (once) ---
    let nn = n * n;
    let (q0_host, init_q_host, q_gpu, dq, v, h_scc, temp, hp, cp, c, d, q_new, tr, dot, occ_mask, eig_diag_buf) = timed!(tm, t_alloc, {
    let q0_host = {
        let mut tmp = vec![0.0f32; batch * n_atoms];
        rt.read_buffer(q0_buf, &mut tmp)?;
        tmp
    };
    // Start from warm-started initial charges (not necessarily q0)
    let init_q_host = {
        let mut tmp = vec![0.0f32; batch * n_atoms];
        rt.read_buffer(init_q_buf, &mut tmp)?;
        tmp
    };
    let q_gpu = rt.buffer_from_slice(&init_q_host)?;  // current charges (GPU)
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
    let occ_mask = rt.zero_buffer::<i32>(batch * n)?;
    let eig_diag_buf = rt.zero_buffer::<f32>(batch * n)?; // Phase 0d
    (q0_host, init_q_host, q_gpu, dq, v, h_scc, temp, hp, cp, c, d, q_new, tr, dot, occ_mask, eig_diag_buf)
    });

    // Host staging buffers
    // Phase 0d: extract only the diagonal (N*batch floats) instead of
    // reading the full N²*batch matrix.
    let mut eig_diag = vec![0.0f32; batch * n];
    let mut mask_host = vec![0i32; batch * n];
    let mut q_new_host = vec![0.0f32; batch * n_atoms];
    let mut q_cur_host = vec![0.0f32; batch * n_atoms];
    // Phase 0e: q_mixed_host starts as init_q_host — it's the host mirror
    // of q_gpu and is updated each iteration after the DIIS mix + upload.
    let mut q_mixed_host = init_q_host.clone();
    let mut rms_per_sys = vec![0.0f32; batch]; // per-system RMS for diagnostics

    // Per-system DIIS mixers
    let mut mixers: Vec<DiisMixer> = (0..batch)
        .map(|_| {
            let mut m = DiisMixer::new(max_history, n_atoms);
            m.warmup = warmup;
            m.alpha = alpha as f64;
            m
        })
        .collect();

    let verbose = std::env::var("RUST_DFTB_SCC_VERBOSE").is_ok();
    let mut n_iters = 0;
    let mut converged = false;
    let mut max_rms_final = 0.0f32;

    for iter in 0..max_iter {
        n_iters = iter + 1;

        // 1-2. Δq = q − q0, V = G·Δq
        timed!(tm, t_delta_q, delta_q_batched(rt, &q_gpu, q0_buf, &dq, n_atoms, batch)?);
        timed!(tm, t_gamma_matvec, gamma_matvec_batched(rt, g_buf, &dq, &v, n_atoms, batch)?);

        // 3. H_scc = H0 + 0.5·S·(V_i + V_j)
        timed!(tm, t_h_scc_update, h_scc_update_batched(rt, h0_buf, s_buf, &v, &h_scc, orb_atom_buf, n, n_atoms, batch)?);

        // 4. H' = X · H_scc · X
        timed!(tm, t_gemm, {
            matmul_batched(rt, &x_buf, &h_scc, &temp, n, batch)?;
            matmul_batched(rt, &temp, &x_buf, &hp, n, batch)?;
        });

        // 5. Jacobi(H') → eigenvalues/eigenvectors
        timed!(tm, t_jacobi, jacobi_batched(rt, &hp, &cp, n, batch)?);

        // 6. Extract diag on GPU → read N*batch floats → sort → occ_mask → upload
        // Phase 0d: uses extract_diagonal_batched kernel instead of reading
        // the full N²*batch matrix. Reduces readback by factor of N.
        timed!(tm, t_occ_sort, {
        extract_diagonal_batched(rt, &hp, &eig_diag_buf, n, batch)?;
        rt.read_buffer(&eig_diag_buf, &mut eig_diag)?;
        for bi in 0..batch {
            let mut idxs: Vec<usize> = (0..n).collect();
            idxs.sort_unstable_by(|&a, &b| {
                eig_diag[bi * n + a]
                    .partial_cmp(&eig_diag[bi * n + b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (rank, &k) in idxs.iter().enumerate() {
                mask_host[bi * n + k] = if rank < n_occ { 1 } else { 0 };
            }
        }
        occ_mask.write(&mask_host).enq().map_err(crate::qmqm::gpu_runtime::map_ocl_err)?;
        });

        // 7. C = X · C'
        timed!(tm, t_back_gemm, matmul_batched(rt, &x_buf, &cp, &c, n, batch)?);

        // 8. D = 2·Σ_{k∈occ} C[:,k]·C[:,k]^T
        timed!(tm, t_density, build_density_masked_batched(rt, &c, &occ_mask, &d, n, batch)?);

        // 9. q_new = Mulliken(D, S) — device-resident
        timed!(tm, t_mulliken, mulliken_charges_batched(rt, &d, s_buf, &q_new, orb_atom_buf, n, n_atoms, batch)?);

        // 10. CPU-driven DIIS mixing
        // Phase 0e: eliminated redundant q_gpu read — q_mixed_host from the
        // previous iteration is already the current q_gpu content (we just
        // uploaded it). Only q_new (Mulliken output) needs to be read.
        let max_rms = timed!(tm, t_diis_mix, {
        rt.read_buffer(&q_new, &mut q_new_host)?;
        // q_cur_host = q_mixed_host from previous iter (or init_q_host on iter 0)
        q_cur_host.copy_from_slice(&q_mixed_host);

        let mut max_rms = 0.0f32;
        rms_per_sys.fill(0.0f32);
        for bi in 0..batch {
            let q_new_slice = &q_new_host[bi * n_atoms..(bi + 1) * n_atoms];
            let q_cur_slice = &q_cur_host[bi * n_atoms..(bi + 1) * n_atoms];
            let q_mixed_slice = &mut q_mixed_host[bi * n_atoms..(bi + 1) * n_atoms];

            // residual = q_new - q_cur
            let residual: Vec<f64> = (0..n_atoms)
                .map(|i| (q_new_slice[i] as f64) - (q_cur_slice[i] as f64))
                .collect();
            let rms = residual.iter().map(|r| r * r).sum::<f64>().sqrt() / (n_atoms as f64).sqrt();
            rms_per_sys[bi] = rms as f32;
            max_rms = max_rms.max(rms as f32);

            // DIIS mix: convert to f64, mix, convert back to f32
            let mut q_inout: Vec<f64> = q_cur_slice.iter().map(|&q| q as f64).collect();
            let q_out: Vec<f64> = q_new_slice.iter().map(|&q| q as f64).collect();
            mixers[bi].mix(&mut q_inout, &q_out, &residual);
            for i in 0..n_atoms {
                q_mixed_slice[i] = q_inout[i] as f32;
            }
        }

        // Upload mixed charges to GPU
        q_gpu.write(&q_mixed_host).enq().map_err(crate::qmqm::gpu_runtime::map_ocl_err)?;
        max_rms
        }); // end timed!(t_diis_mix)

        if verbose && (iter < 5 || iter % 10 == 0) {
            eprintln!("    [gpu_scc_diis] iter {iter}: max_rms={max_rms:.3e}");
        }

        max_rms_final = max_rms;
        if max_rms < tol {
            converged = true;
            break;
        }
    }

    if !converged {
        // Report worst systems for diagnosis
        let mut worst: Vec<(usize, f32)> = (0..batch).map(|i| (i, rms_per_sys[i])).collect();
        worst.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_report = batch.min(10);
        let mut diag = String::new();
        for k in 0..n_report {
            diag.push_str(&format!("\n  sys[{}] RMS={:.3e}", worst[k].0, worst[k].1));
        }
        let n_conv = rms_per_sys.iter().filter(|&&r| r < tol).count();
        let n_fail = batch - n_conv;
        eprintln!("    [gpu_scc_diis] WARNING: {n_conv}/{batch} systems converged, {n_fail} failed.{diag}");
        if !best_effort {
            return Err(DftbError::SccNotConverged(format!(
                "GPU SCC (DIIS) did not converge in {max_iter} iterations (max RMS = {max_rms_final:.3e}).{diag}"
            )));
        }
        // best_effort: fall through to energy computation with unconverged charges
    }

    // --- Energy computation ---
    timed!(tm, t_energy, {
        frobenius_trace_batched(rt, &d, h0_buf, &tr, n, batch)?;
        delta_q_batched(rt, &q_gpu, q0_buf, &dq, n_atoms, batch)?;
        gamma_matvec_batched(rt, g_buf, &dq, &v, n_atoms, batch)?;
        dot_batched(rt, &dq, &v, &dot, n_atoms, batch)?;
    });

    let (energies, charges, eigenvalues) = timed!(tm, t_readback, {
        let mut e_h0 = vec![0.0f32; batch];
        let mut e_scc = vec![0.0f32; batch];
        rt.read_buffer(&tr, &mut e_h0)?;
        rt.read_buffer(&dot, &mut e_scc)?;
        let energies: Vec<f32> = (0..batch).map(|i| e_h0[i] + 0.5 * e_scc[i]).collect();

        let mut charges = vec![0.0f32; batch * n_atoms];
        rt.read_buffer(&q_gpu, &mut charges)?;

        // Phase 0d: extract diagonal on GPU instead of reading full N²*batch
        extract_diagonal_batched(rt, &hp, &eig_diag_buf, n, batch)?;
        rt.read_buffer(&eig_diag_buf, &mut eig_diag)?;
        let mut eigenvalues = vec![0.0f32; batch * n];
        for bi in 0..batch {
            let mut eigs: Vec<f32> = eig_diag[bi * n..(bi + 1) * n].to_vec();
            eigs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            eigenvalues[bi * n..(bi + 1) * n].copy_from_slice(&eigs);
        }
        (energies, charges, eigenvalues)
    });

    if let Some(tm) = tm.as_mut() {
        tm.t_total = t_total_start.elapsed().as_secs_f64();
        tm.print("gpu_solve_scc_batched_diis_warmstart", n, batch, n_iters);
    }

    Ok(GpuSccResult {
        energies,
        charges,
        eigenvalues,
        n_iters,
        rms: vec![max_rms_final; batch],
        timing: tm,
    })
}
