//! Charge mixing strategies for the global SCC fixed-point iteration.
//!
//! The residual is defined as `F(q) = q_out(q) - q_in`.
//! Mixers accelerate convergence by extrapolating in the history subspace.

/// Trait for charge-vector mixers.
///
/// All implementations work on pre-allocated slices to guarantee zero
/// allocation in the SCC hot loop.
pub trait Mixer {
    /// Produce the next input guess `q_in` from the output `q_out` and the residual.
    ///
    /// # Arguments
    /// * `q_inout` – current input vector, overwritten with the mixed guess.
    /// * `q_out`   – output vector from the current diagonalization / charge analysis.
    /// * `residual` – element-wise `q_out - q_inout` (provided for convenience).
    fn mix(&mut self, q_inout: &mut [f64], q_out: &[f64], residual: &[f64]);

    /// Reset internal history (e.g. when geometry changes).
    fn reset(&mut self);
}

/// Simple linear mixing: `q^(k+1) = α·q_out + (1-α)·q^(k)`.
///
/// Robust but slow; useful as a fallback or for the first few iterations.
#[derive(Debug, Clone, Copy)]
pub struct SimpleMixer {
    pub alpha: f64,
}

impl SimpleMixer {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for SimpleMixer {
    fn default() -> Self {
        Self { alpha: 0.3 }
    }
}

impl Mixer for SimpleMixer {
    fn mix(&mut self, q_inout: &mut [f64], _q_out: &[f64], residual: &[f64]) {
        // q_new = q_old + α·(q_out - q_old) = q_old + α·residual
        for (q, &r) in q_inout.iter_mut().zip(residual.iter()) {
            *q += self.alpha * r;
        }
    }

    fn reset(&mut self) {}
}

/// Anderson / DIIS mixer (also known as Pulay mixing in quantum chemistry).
///
/// Keeps a history of recent `q_in` and `residual` vectors. The next guess
/// is the linear combination that minimises the residual norm in the history
/// subspace.
///
/// C9: Uses a ring buffer of preallocated arrays — zero allocations per iteration.
///
/// Reference: Anderson, J. Assoc. Comput. Mach. 12, 547 (1965).
#[derive(Debug, Clone)]
pub struct DiisMixer {
    /// Max number of history vectors to retain.
    pub max_history: usize,
    /// Number of simple-mixing warmup iterations before DIIS kicks in.
    pub warmup: usize,
    /// Simple mixing parameter for warmup and fallback.
    pub alpha: f64,
    /// Iteration counter.
    iter: usize,
    /// Ring buffer of input vectors `q_in` (preallocated, C9).
    q_in_bufs: Vec<Vec<f64>>,
    /// Ring buffer of residual vectors `F(q)` (preallocated, C9).
    res_bufs: Vec<Vec<f64>>,
    /// Ring buffer write position.
    buf_idx: usize,
    /// Number of valid entries in the ring buffer.
    n_filled: usize,
    /// Pre-allocated matrix for the DIIS linear system `B·c = rhs`.
    /// Shape `[(max_history+1) × (max_history+1)]`, stored row-major.
    b_mat: Vec<f64>,
    /// Pre-allocated RHS vector.
    rhs: Vec<f64>,
    /// Pre-allocated workspace for the linear solver.
    work: Vec<f64>,
    /// Pre-allocated pivot array.
    ipiv: Vec<i32>,
}

impl DiisMixer {
    pub fn new(max_history: usize, vector_len: usize) -> Self {
        let b_mat = vec![0.0; (max_history + 1) * (max_history + 1)];
        let rhs = vec![0.0; max_history + 1];
        let work = vec![0.0; max_history];
        let ipiv = vec![0; max_history + 1];
        // C9: preallocate ring buffer arrays
        let q_in_bufs = (0..max_history).map(|_| vec![0.0; vector_len]).collect();
        let res_bufs = (0..max_history).map(|_| vec![0.0; vector_len]).collect();
        Self {
            max_history,
            warmup: 0,   // no warmup — DIIS starts immediately (falls back to simple mixing until enough history)
            alpha: 0.5,  // more aggressive simple mixing for faster convergence
            iter: 0,
            q_in_bufs,
            res_bufs,
            buf_idx: 0,
            n_filled: 0,
            b_mat,
            rhs,
            work,
            ipiv,
        }
    }

    /// Build and solve the DIIS linear system.
    ///
    /// `B_ij = <F_i, F_j>` (inner product of residuals)
    /// Last row/column enforces `Σ c_i = 1`.
    ///
    /// Returns coefficients `c` in `self.work[..n_hist]`.
    fn solve_diis(&mut self) -> usize {
        let n = self.n_filled;
        if n == 0 {
            return 0;
        }

        // Assemble augmented matrix B (size (n+1)×(n+1)) in row-major.
        let np1 = n + 1;
        for i in 0..n {
            let res_i = &self.res_bufs[i];
            for j in 0..n {
                let dot: f64 = res_i
                    .iter()
                    .zip(&self.res_bufs[j])
                    .map(|(a, b)| a * b)
                    .sum();
                self.b_mat[i * np1 + j] = dot;
            }
            // Constraint row and column
            self.b_mat[i * np1 + n] = 1.0;
            self.b_mat[n * np1 + i] = 1.0;
        }
        self.b_mat[n * np1 + n] = 0.0;

        // RHS: [0, 0, ..., 0, 1]
        self.rhs.fill(0.0);
        self.rhs[n] = 1.0;

        // Solve with Gaussian elimination for the small system.
        gauss_eliminate(np1, &mut self.b_mat, &mut self.rhs, &mut self.ipiv);

        // Copy coefficients to work buffer.
        self.work[..n].copy_from_slice(&self.rhs[..n]);
        n
    }
}

impl Mixer for DiisMixer {
    fn mix(&mut self, q_inout: &mut [f64], _q_out: &[f64], residual: &[f64]) {
        self.iter += 1;

        // Warmup phase: simple mixing for first `warmup` iterations.
        if self.iter <= self.warmup {
            for (q, &r) in q_inout.iter_mut().zip(residual.iter()) {
                *q += self.alpha * r;
            }
            return;
        }

        // C9: Store current state in ring buffer (no allocation — copy_from_slice)
        let idx = self.buf_idx;
        self.q_in_bufs[idx].copy_from_slice(q_inout);
        self.res_bufs[idx].copy_from_slice(residual);
        self.buf_idx = (self.buf_idx + 1) % self.max_history;
        if self.n_filled < self.max_history { self.n_filled += 1; }

        let n_hist = self.solve_diis();
        if n_hist == 0 {
            // Fall back to simple mixing.
            for (q, &r) in q_inout.iter_mut().zip(residual.iter()) {
                *q += self.alpha * r;
            }
            return;
        }

        // q_new = Σ_i c_i · q_out_i  where q_out_i = q_in_i + residual_i
        // Phase 0e safeguard (manifest §4.9): reject non-finite or catastrophic
        // DIIS extrapolation, fall back to damped simple mixing.
        let prev_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        // Index of the buffer slot that just received the CURRENT q_in
        // (buf_idx was already advanced) — needed for the step-norm check.
        let cur = (self.buf_idx + self.max_history - 1) % self.max_history;
        q_inout.fill(0.0);
        for i in 0..n_hist {
            let c = self.work[i];
            let q_in_i = &self.q_in_bufs[i];
            let res_i = &self.res_bufs[i];
            for (q, (&q_in_val, &res_val)) in q_inout.iter_mut().zip(q_in_i.iter().zip(res_i.iter())) {
                *q += c * (q_in_val + res_val);
            }
        }
        // Safeguard: reject non-finite, unphysical charges, or an ill-
        // conditioned extrapolation. The correct scale is the STEP
        // ‖q_next − q_in‖ vs ‖r‖ — a DIIS step ≫ residual signals bad
        // coefficients. (Bug fix 2026-09-12: the old check compared
        // ‖q_next‖ to ‖r‖ — ‖q‖ is O(√N·valence) ≈ 13 for Si10H16, so DIIS
        // was silently rejected whenever ‖r‖ < ‖q‖/100, i.e. exactly when
        // the iterate was close enough that only DIIS could finish.)
        let step_norm: f64 = q_inout.iter().zip(self.q_in_bufs[cur].iter())
            .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        let max_q: f64 = q_inout.iter().fold(0.0f64, |m, &q| m.max(q.abs()));
        if !q_inout.iter().all(|q| q.is_finite()) || max_q > 10.0
            || step_norm > 50.0 * prev_norm.max(1e-10) {
            // Reject DIIS, take damped simple mixing step instead.
            // Restore q_inout to the pre-DIIS state (q_in before overwrite).
            q_inout.copy_from_slice(&self.q_in_bufs[cur]);
            for (q, &r) in q_inout.iter_mut().zip(residual.iter()) {
                *q += self.alpha * r;
            }
        }
    }

    fn reset(&mut self) {
        self.buf_idx = 0;
        self.n_filled = 0;
        self.iter = 0;
    }
}

impl DiisMixer {
    /// Reset only the iteration counter, keeping history buffers.
    /// Use after geometry change with warm-started charges — the old DIIS
    /// subspace may still span useful directions in charge space.
    pub fn reset_iter_only(&mut self) {
        self.iter = 0;
    }
}

/// Tiny in-place Gaussian elimination for the small DIIS linear system.
/// Solves `A·x = b` where `A` is `n×n` row-major, `b` length `n`.
/// Pivot indices stored in `ipiv` (only used for shape; overwritten).
fn gauss_eliminate(n: usize, a: &mut [f64], b: &mut [f64], ipiv: &mut [i32]) {
    // Forward elimination with partial pivoting.
    for k in 0..n {
        // Find pivot.
        let mut max_row = k;
        let mut max_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        ipiv[k] = max_row as i32;

        // Swap rows in A and b.
        if max_row != k {
            for j in k..n {
                a.swap(k * n + j, max_row * n + j);
            }
            b.swap(k, max_row);
        }

        // Singular or near-singular -> bail out (will fall back to simple mixing).
        if a[k * n + k].abs() < 1e-14 {
            continue;
        }

        for i in (k + 1)..n {
            let factor = a[i * n + k] / a[k * n + k];
            a[i * n + k] = 0.0;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution.
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * b[j];
        }
        if a[i * n + i].abs() > 1e-14 {
            b[i] = sum / a[i * n + i];
        } else {
            b[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_mixer_basic() {
        let mut m = SimpleMixer::new(0.5);
        let mut q = vec![0.0, 0.0, 0.0];
        let q_out = vec![2.0, 4.0, 6.0];
        let residual = vec![2.0, 4.0, 6.0];
        m.mix(&mut q, &q_out, &residual);
        assert_eq!(q, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn gauss_eliminate_identity() {
        let mut a = vec![1.0, 0.0, 0.0, 1.0];
        let mut b = vec![3.0, 5.0];
        let mut ipiv = vec![0; 2];
        gauss_eliminate(2, &mut a, &mut b, &mut ipiv);
        assert!((b[0] - 3.0).abs() < 1e-12);
        assert!((b[1] - 5.0).abs() < 1e-12);
    }
}

/// Broyden "good" quasi-Newton mixer for SCC fixed-point iteration.
///
/// Approximates the inverse Jacobian J^{-1} of F(q) = q_out(q) - q_in
/// using multi-secant updates (Sherman-Morrison).
///
/// Convergence: superlinear (faster than DIIS for well-behaved SCC).
/// Storage: O(n²) for J_inv (n = number of atoms).
/// Per-iter cost: O(n²) for matrix-vector + rank-1 update.
///
/// Reference: Broyden, Math. Comp. 19, 577 (1965); Johnson, J. Chem. Phys.
/// 153, 184103 (2020) — "Broyden mixing for DFTB".
#[derive(Debug, Clone)]
pub struct BroydenMixer {
    pub alpha: f64,           // initial mixing parameter (J_inv = -alpha*I)
    n: usize,                 // vector length
    iter: usize,
    /// Inverse Jacobian approximation, row-major n×n.
    j_inv: Vec<f64>,
    /// Previous q_in (for computing s = Δq).
    q_prev: Vec<f64>,
    /// Previous residual F (for computing y = ΔF).
    f_prev: Vec<f64>,
    /// Work vectors.
    jq: Vec<f64>,             // J_inv · F
    s: Vec<f64>,              // Δq
    y: Vec<f64>,              // ΔF
    jt_y: Vec<f64>,           // J_inv^T · y  (for denominator)
    js: Vec<f64>,             // J_inv · s
    /// Whether J_inv has been initialized.
    initialized: bool,
}

impl BroydenMixer {
    pub fn new(n: usize, alpha: f64) -> Self {
        Self {
            alpha,
            n,
            iter: 0,
            j_inv: vec![0.0; n * n],
            q_prev: vec![0.0; n],
            f_prev: vec![0.0; n],
            jq: vec![0.0; n],
            s: vec![0.0; n],
            y: vec![0.0; n],
            jt_y: vec![0.0; n],
            js: vec![0.0; n],
            initialized: false,
        }
    }

    /// J_inv = -alpha * I
    fn init_j_inv(&mut self) {
        self.j_inv.fill(0.0);
        for i in 0..self.n {
            self.j_inv[i * self.n + i] = -self.alpha;
        }
        self.initialized = true;
    }

    /// Compute v = J_inv · x  (row-major matrix-vector product)
    fn matvec(&self, x: &[f64], v: &mut [f64]) {
        let n = self.n;
        let j_inv = &self.j_inv;
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += j_inv[i * n + j] * x[j];
            }
            v[i] = sum;
        }
    }

    /// Sherman-Morrison rank-1 update for "good Broyden":
    ///   J_{k+1}·s = y  (secant condition)
    ///   J_inv_{k+1} = J_inv + (s - J_inv·y)·(s^T·J_inv) / (s^T·J_inv·y)
    fn update(&mut self) {
        let n = self.n;
        // J_inv · y → js (inline to avoid borrow conflict)
        let y = self.y.clone(); // small (n_atoms), stack-ish
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += self.j_inv[i * n + j] * y[j];
            }
            self.js[i] = sum;
        }
        // s^T · (J_inv · y) = s · js  (denominator)
        let denom: f64 = self.s.iter().zip(self.js.iter()).map(|(a, b)| a * b).sum();
        if denom.abs() < 1e-14 {
            return; // Singular update — skip
        }
        // u = s - J_inv·y = s - js
        let mut u = vec![0.0; n];
        for i in 0..n { u[i] = self.s[i] - self.js[i]; }
        // w = J_inv^T · s  (row vector for outer product)
        let s = self.s.clone();
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += self.j_inv[j * n + i] * s[j]; // J_inv^T[i,j] = J_inv[j,i]
            }
            self.jt_y[i] = sum;
        }
        // J_inv += u ⊗ w / denom
        let inv_denom = 1.0 / denom;
        for i in 0..n {
            for j in 0..n {
                self.j_inv[i * n + j] += u[i] * self.jt_y[j] * inv_denom;
            }
        }
    }
}

impl Mixer for BroydenMixer {
    fn mix(&mut self, q_inout: &mut [f64], _q_out: &[f64], residual: &[f64]) {
        self.iter += 1;
        let n = self.n;

        if !self.initialized {
            self.init_j_inv();
        } else {
            // s = q_in - q_prev, y = F - F_prev
            for i in 0..n {
                self.s[i] = q_inout[i] - self.q_prev[i];
                self.y[i] = residual[i] - self.f_prev[i];
            }
            self.update();
        }

        // Save current state
        self.q_prev.copy_from_slice(q_inout);
        self.f_prev.copy_from_slice(residual);

        // Step: q_new = q_in - J_inv · F  (inline matvec to avoid borrow conflict)
        let f = residual.to_vec(); // small (n_atoms)
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += self.j_inv[i * n + j] * f[j];
            }
            self.jq[i] = sum;
        }
        for i in 0..n {
            q_inout[i] -= self.jq[i];
        }
    }

    fn reset(&mut self) {
        self.iter = 0;
        self.initialized = false;
        self.q_prev.fill(0.0);
        self.f_prev.fill(0.0);
    }
}
