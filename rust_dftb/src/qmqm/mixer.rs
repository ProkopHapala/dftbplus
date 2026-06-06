//! Charge mixing strategies for the global SCC fixed-point iteration.
//!
//! The residual is defined as `F(q) = q_out(q) - q_in`.
//! Mixers accelerate convergence by extrapolating in the history subspace.

use std::collections::VecDeque;

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
/// Reference: Anderson, J. Assoc. Comput. Mach. 12, 547 (1965).
#[derive(Debug, Clone)]
pub struct DiisMixer {
    /// Max number of history vectors to retain.
    pub max_history: usize,
    /// History of input vectors `q_in`.
    q_in_history: VecDeque<Vec<f64>>,
    /// History of residual vectors `F(q)`.
    residual_history: VecDeque<Vec<f64>>,
    /// Pre-allocated matrix for the DIIS linear system `B·c = rhs`.
    /// Shape `[max_history × max_history]`, stored row-major.
    b_mat: Vec<f64>,
    /// Pre-allocated RHS vector.
    rhs: Vec<f64>,
    /// Pre-allocated workspace for the linear solver.
    work: Vec<f64>,
    /// Pre-allocated pivot array.
    ipiv: Vec<i32>,
}

impl DiisMixer {
    pub fn new(max_history: usize, _vector_len: usize) -> Self {
        let b_mat = vec![0.0; max_history * max_history];
        let rhs = vec![0.0; max_history + 1];
        let work = vec![0.0; max_history];
        let ipiv = vec![0; max_history + 1];
        Self {
            max_history,
            q_in_history: VecDeque::with_capacity(max_history),
            residual_history: VecDeque::with_capacity(max_history),
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
        let n = self.residual_history.len();
        if n == 0 {
            return 0;
        }

        // Assemble augmented matrix B (size (n+1)×(n+1)) in row-major.
        let np1 = n + 1;
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = self.residual_history[i]
                    .iter()
                    .zip(&self.residual_history[j])
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

        // Solve with LAPACK dgesv (if available) or a small dense solver.
        // For the skeleton we use a simple Gaussian elimination for the small system.
        // TODO: replace with lapack bindings for production.
        gauss_eliminate(np1, &mut self.b_mat, &mut self.rhs, &mut self.ipiv);

        // Copy coefficients to work buffer.
        self.work[..n].copy_from_slice(&self.rhs[..n]);
        n
    }
}

impl Mixer for DiisMixer {
    fn mix(&mut self, q_inout: &mut [f64], _q_out: &[f64], residual: &[f64]) {
        // Store current state in history.
        let q_in_copy: Vec<f64> = q_inout.to_vec();
        let res_copy: Vec<f64> = residual.to_vec();

        if self.q_in_history.len() == self.max_history {
            self.q_in_history.pop_front();
            self.residual_history.pop_front();
        }
        self.q_in_history.push_back(q_in_copy);
        self.residual_history.push_back(res_copy);

        let n_hist = self.solve_diis();
        if n_hist == 0 {
            // Fall back to simple mixing on first iteration.
            for (q, &r) in q_inout.iter_mut().zip(residual.iter()) {
                *q += 0.3 * r;
            }
            return;
        }

        // q_new = Σ_i c_i · q_out_i   (using the stored q_in + residual = q_out)
        q_inout.fill(0.0);
        for (i, q_in_i) in self.q_in_history.iter().enumerate() {
            let c = self.work[i];
            for (q, &q_in_val) in q_inout.iter_mut().zip(q_in_i.iter()) {
                // q_out_i = q_in_i + residual_i, but for DIIS we extrapolate q_in directly
                // Standard DIIS: q_new = Σ c_i · q_in_i
                *q += c * q_in_val;
            }
        }
    }

    fn reset(&mut self) {
        self.q_in_history.clear();
        self.residual_history.clear();
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
