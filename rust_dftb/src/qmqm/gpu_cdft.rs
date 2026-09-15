//! Constrained-DFT (CDFT) layer for the dense batched solver
//! (`Dense_Multi_CDFT` spec): fragment Mulliken-charge constraints
//!     Q_F = Σ_{A∈F} Δq_A  =  Q_F^target
//! enforced by a per-fragment Lagrange multiplier λ_F that enters the
//! SCC Hamiltonian as an on-site shift  V_A → V_A + λ_F·w_A, i.e.
//!     H_scc[μν] += ½·λ_F·S[μν]·(w_μ + w_ν).
//!
//! The shift is injected by `cdft_hscc_shift_batched` right after the
//! fused dq→V→H_scc build inside `GpuSccPlan::enq_dq_v_hscc`, so every
//! Hamiltonian rebuild (SCC iterations, finalize, eval) carries the
//! constraint — while the unshifted V buffer keeps the energy dot
//! ½Δq·V on the pure DFTB part. Reported band energies then contain
//! +Σ_F λ_F·Q_F; `GpuDftb::cdft_energies` subtracts it to recover the
//! constrained-state DFTB energy. Forces come out correct at fixed λ
//! (the −λ·dQ/dR term rides along through the existing dS/dR path).
//!
//! All state is per-replica: `lam`/`target` are [batch*nfrag] so one
//! batch can carry a whole diabatic-state ladder in a single launch.
//! The λ outer loop and the fragment-charge reductions run on the host
//! in f64 — both are O(batch·n_atoms) scalars, off the GPU hot path.

use crate::core::error::{DftbError, Result};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use ocl::{Buffer, Kernel};

const CDFT_KERNEL_SOURCE: &str = include_str!("gpu_cdft.cl");

/// Per-replica fragment-charge constraint state. Owned by `GpuSccPlan`
/// (`plan.cdft`); `None` means the solver pays literally zero cost.
pub struct GpuCdft {
    pub nfrag: usize,
    /// Host copy of the atom→fragment map (template-level, -1 = none).
    pub frag: Vec<i32>,
    /// [batch*nfrag] Lagrange multipliers on device.
    buf_lam: Buffer<f32>,
    /// Host λ state [batch*nfrag] — f64 for the secant updates.
    pub lam: Vec<f64>,
    /// Secant history: previous λ and previous Q-err per (b,f).
    lam_prev: Vec<f64>,
    err_prev: Vec<f64>,
    /// Per-(b,f) adaptive step cap — shrinks when err flips sign
    /// (root bracketed → the secant is overshooting a steep Q(λ)).
    step: Vec<f64>,
    /// Per-(b,f) λ bracket: err>0 (too many electrons on F) means λ is a
    /// LOWER bound for λ*; err<0 an upper bound. NaN = unbounded side.
    br_lo: Vec<f64>,
    br_hi: Vec<f64>,
    /// [batch*nfrag] target fragment excess charge (Mulliken Δq sum).
    pub target: Vec<f64>,
    /// [nfrag] fragment REFERENCE (neutral) population Σ_{A∈F} q0_A —
    /// the h_scc shift contributes λ·Q_gross to e_band, not λ·Δq, so
    /// cdft_energies must subtract λ·(Q_F + q0_frag).
    pub q0_frag: Vec<f64>,
    k_shift: Kernel,
}

/// Result of one `GpuDftb::cdft_scc` outer-λ loop.
pub struct CdftReport {
    pub outer_iters: usize,
    /// max |Q_F − Q_F^target| over all (b,f) at exit.
    pub q_err_max: f64,
    /// Per-replica converged flag (SCC ok AND all fragment errs ≤ q_tol).
    pub converged: Vec<bool>,
    /// Fragment excess charges at exit, [batch*nfrag], f64.
    pub qfrag: Vec<f64>,
    /// Final multipliers [batch*nfrag].
    pub lam: Vec<f64>,
}

impl GpuCdft {
    /// Attach a constraint set to a plan. `frag` is [n_atoms] (template
    /// level: −1 = unconstrained, else fragment id 0..nfrag−1).
    /// `targets` is [batch*nfrag] — target excess charge Q_F in e.
    pub fn new(
        rt: &mut GpuRuntime,
        n: usize,
        n_atoms: usize,
        batch: usize,
        frag: &[i32],
        targets: &[f64],
        s_buf: &Buffer<f32>,
        orb_atom: &Buffer<i32>,
        h_scc: &Buffer<f32>,
        active: &Buffer<i32>,
    ) -> Result<Self> {
        if frag.len() != n_atoms {
            return Err(DftbError::InvalidInput(format!(
                "GpuCdft: frag.len()={} != n_atoms={n_atoms}", frag.len()
            )));
        }
        let nfrag = frag.iter().copied().max().unwrap_or(-1) + 1;
        if nfrag <= 0 {
            return Err(DftbError::InvalidInput("GpuCdft: no atom assigned to any fragment".into()));
        }
        let nfrag = nfrag as usize;
        if targets.len() != batch * nfrag {
            return Err(DftbError::InvalidInput(format!(
                "GpuCdft: targets.len()={} != batch*nfrag {batch}*{nfrag}", targets.len()
            )));
        }
        for (a, &f) in frag.iter().enumerate() {
            if f >= nfrag as i32 || f < -1 {
                return Err(DftbError::InvalidInput(format!(
                    "GpuCdft: frag[{a}]={f} outside -1..{}", nfrag as i32 - 1
                )));
            }
        }
        if let Some(i) = targets.iter().position(|t| !t.is_finite()) {
            return Err(DftbError::InvalidInput(format!("GpuCdft: targets[{i}]={} non-finite", targets[i])));
        }
        let buf_frag = rt.buffer_from_slice(frag)?;
        let buf_lam = rt.zero_buffer::<f32>(batch * nfrag)?;
        let prog = rt.build_program(CDFT_KERNEL_SOURCE)?;
        let wg = 256usize;
        let k_shift = Kernel::builder()
            .program(&prog).name("cdft_hscc_shift_batched").queue(rt.queue().clone())
            .global_work_size(batch * wg).local_work_size(wg)
            .arg(n as i32).arg(n_atoms as i32).arg(batch as i32).arg(nfrag as i32)
            .arg(s_buf).arg(orb_atom).arg(&buf_frag).arg(&buf_lam).arg(h_scc)
            .arg_local::<f32>(n_atoms)
            .arg(active)
            .build().map_err(map_ocl_err)?;
        Ok(Self {
            nfrag,
            frag: frag.to_vec(),
            buf_lam,
            lam: vec![0.0; batch * nfrag],
            lam_prev: vec![0.0; batch * nfrag],
            err_prev: vec![f64::NAN; batch * nfrag],
            step: vec![f64::INFINITY; batch * nfrag],
            br_lo: vec![f64::NAN; batch * nfrag],
            br_hi: vec![f64::NAN; batch * nfrag],
            target: targets.to_vec(),
            q0_frag: vec![0.0; nfrag],   // filled by set_cdft (needs q0)
            k_shift,
        })
    }

    /// Enqueue the h_scc shift on the kernel's bound queue. Called from
    /// `enq_dq_v_hscc` after the fused build — one extra launch per
    /// h_scc rebuild, only while a constraint set is attached.
    pub fn enq_shift(&self) -> Result<()> {
        unsafe { self.k_shift.enq().map_err(map_ocl_err) }
    }

    /// Clear the bracket/secant history for element i — needed during
    /// target continuation, where the effective target moves every outer
    /// iteration and a stale bracket bisects toward the wrong root.
    pub fn clear_search(&mut self, i: usize) {
        self.br_lo[i] = f64::NAN;
        self.br_hi[i] = f64::NAN;
        self.err_prev[i] = f64::NAN;
    }

    /// Diagnostic accessors for the driver stall report.
    pub fn debug_br(&self, i: usize) -> (f64, f64) { (self.br_lo[i], self.br_hi[i]) }
    pub fn lam_prev(&self, i: usize) -> f64 { self.lam_prev[i] }
    pub fn err_prev(&self, i: usize) -> f64 { self.err_prev[i] }

    /// Upload host `lam` to the device buffer (between SCC solves).
    pub fn upload_lam(&self, rt: &GpuRuntime) -> Result<()> {
        let v: Vec<f32> = self.lam.iter().map(|&x| x as f32).collect();
        rt.write_buffer(&self.buf_lam, &v)
    }

    /// Fragment excess charges Q_F[b][f] = Σ_{a∈f} dq[b][a] from the
    /// device dq buffer — host f64 reduction (O(batch·n_atoms), once
    /// per outer iteration, off the GPU hot path).
    pub fn qfrag_from_dq(&self, dq: &[f32], batch: usize, n_atoms: usize) -> Vec<f64> {
        let mut qf = vec![0.0f64; batch * self.nfrag];
        for b in 0..batch {
            for (a, &f) in self.frag.iter().enumerate() {
                if f >= 0 {
                    qf[b * self.nfrag + f as usize] += dq[b * n_atoms + a] as f64;
                }
            }
        }
        qf
    }

    /// Secant/damped λ update for one (b,f) element.
    /// err = Q_F − Q_F^target > 0 means too many electrons on F →
    /// raise λ to push them out. κ0 = u_mean/n_f Ha per e on the first
    /// step; afterwards a safeguarded secant on err(λ).
    /// Safeguarded damped-secant λ update for one (b,f).
    /// err = Q_F − Q_F^target > 0 means too many electrons on F → λ must
    /// grow (dQ/dλ < 0). Q(λ) is piecewise-smooth but the SCC switches
    /// metastable basins at level crossings, so the response is NOT
    /// globally monotone — bisection locks at the crossing and never
    /// escapes; empirical coverage is best with a plain capped secant
    /// that keeps sampling (the driver restores best-λ at the end).
    /// The bracket is kept for diagnostics only.
    pub fn update_lam(&mut self, b: usize, f: usize, err: f64, kappa0: f64, step_cap: f64) {
        let i = b * self.nfrag + f;
        if !self.step[i].is_finite() { self.step[i] = step_cap; }
        if err > 0.0 {
            if self.br_lo[i].is_nan() || self.lam[i] > self.br_lo[i] { self.br_lo[i] = self.lam[i]; }
        } else {
            if self.br_hi[i].is_nan() || self.lam[i] < self.br_hi[i] { self.br_hi[i] = self.lam[i]; }
        }
        let prev_err = self.err_prev[i];
        if prev_err.is_finite() && err * prev_err < 0.0 { self.step[i] *= 0.5; }
        let mut d = if prev_err.is_finite() && (err - prev_err).abs() > 1e-8 {
            // secant: λ_new = λ − err·(λ − λ_prev)/(err − err_prev)
            -err * (self.lam[i] - self.lam_prev[i]) / (err - prev_err)
        } else {
            kappa0 * err
        };
        if !d.is_finite() { d = kappa0 * err; }
        d = d.clamp(-self.step[i], self.step[i]);
        self.lam_prev[i] = self.lam[i];
        self.err_prev[i] = err;
        self.lam[i] += d;
    }
}
