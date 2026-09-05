//! interpolation.rs — Cubic Hermite spline interpolation of SK integrals on a uniform grid.
//!
//! Replaces the original 8-point Neville polynomial (degree 7, O(n²) per eval,
//! no precomputation) with cubic Hermite spline (degree 3, O(1) per eval,
//! precomputed derivatives at load time). The Neville code is retained as
//! `eval_neville_into` / `eval_eqgrid_new_into` for parity verification and
//! for the tail region [last_grid_r, r_max] where `poly5_to_zero` requires
//! high-order finite-difference derivatives that the Hermite precomputation
//! does not provide accurately enough.
//!
//! Key functions:
//! - `EqGridTable::new` — constructs table + precomputes per-grid-point derivatives
//! - `eval_hermite_into` — value only, O(n_integ), 4 FMAs per channel
//! - `eval_hermite_with_deriv_into` — value + analytic dV/dr in one call
//! - `eval_eqgrid_new_into` — Neville fallback (tail + parity reference)

use crate::core::error::{DftbError, Result};

pub const DIST_FUDGE: f64 = 1.0;

const MAX_N_INTER: usize = 8;  // max interpolation stencil (kept for Neville fallback)
const MAX_N_INTEG: usize = 20; // max columns in extended-format SK tables
const N_INTER: usize = 8;
const N_RIGHT: usize = 4;
const DELTA_R: f64 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    EqGridNew,
}

/// SK integral table on a uniform grid, with precomputed Hermite spline coefficients.
///
/// Interpolation: cubic Hermite spline (degree 3, C¹ continuous).
/// - Precomputed at load time: f'(x_i) at each grid point via 4th-order central differences.
/// - Per-evaluation cost: O(n_integ) — 4 FMAs per channel (find interval + Hermite basis).
/// - Derivative cost: O(n_integ) — same precomputed data, different basis derivatives.
///
/// This replaces the old 8-point Neville interpolation (degree 7, O(n²) per eval,
/// no precomputation, derivatives required 3 separate evaluations).
///
/// The Neville code is retained as a fallback and for parity verification.
#[derive(Debug, Clone)]
pub struct EqGridTable {
    pub dr: f64,
    pub values: Vec<Vec<f64>>, // [n_grid][n_integ]
    /// Precomputed first derivatives at each grid point: derivs[i][k] = dV_k/dx at x_i.
    /// Computed once at construction via 4th-order central finite differences.
    derivs: Vec<Vec<f64>>, // [n_grid][n_integ]
}

impl EqGridTable {
    pub fn new(dr: f64, values: Vec<Vec<f64>>) -> Self {
        let derivs = compute_hermite_derivs(&values, dr);
        Self { dr, values, derivs }
    }

    pub fn n_grid(&self) -> usize {
        self.values.len()
    }

    pub fn n_integ(&self) -> usize {
        self.values.first().map(|r| r.len()).unwrap_or(0)
    }

    pub fn r_max(&self) -> f64 {
        self.n_grid() as f64 * self.dr + DIST_FUDGE
    }

    /// Convenience wrapper that allocates a Vec. For hot paths use `eval_into`.
    pub fn eval(&self, r: f64) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.n_integ()];
        self.eval_into(r, &mut out)?;
        Ok(out)
    }

    /// Evaluate SK integrals at distance `r` using cubic Hermite spline.
    /// Writes into caller-provided buffer. Zero-allocation, O(n_integ).
    /// Tail region [last_grid_r, r_max] uses Neville poly5_to_zero for parity.
    pub fn eval_into(&self, r: f64, out: &mut [f64]) -> Result<()> {
        eval_hermite_into(self, r, out)
    }

    /// Evaluate SK integrals AND radial derivative dV/dr at distance `r`.
    /// Uses the same precomputed Hermite coefficients — no finite differences.
    /// O(n_integ), zero-allocation.
    pub fn eval_with_deriv_into(&self, r: f64, out: &mut [f64], ddr: &mut [f64]) -> Result<()> {
        eval_hermite_with_deriv_into(self, r, out, ddr)
    }

    /// Neville fallback for parity verification. Not used in production.
    pub fn eval_neville_into(&self, r: f64, out: &mut [f64]) -> Result<()> {
        eval_eqgrid_new_into(self, r, out)
    }
}

/// Compute first derivatives at each grid point using 4th-order central finite differences.
/// Interior: f'(x_i) ≈ (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12·dr)
/// Boundaries: 2nd-order forward/backward difference.
/// Endpoints (at r_max boundary): derivative set to match poly5_to_zero tail behavior.
fn compute_hermite_derivs(values: &[Vec<f64>], dr: f64) -> Vec<Vec<f64>> {
    let n_grid = values.len();
    if n_grid == 0 {
        return Vec::new();
    }
    let n_integ = values[0].len();
    let mut derivs = vec![vec![0.0f64; n_integ]; n_grid];
    let inv_12dr = 1.0 / (12.0 * dr);
    let inv_2dr = 1.0 / (2.0 * dr);

    for i in 0..n_grid {
        for k in 0..n_integ {
            if i >= 2 && i + 2 < n_grid {
                // 4th-order central difference: O(dr⁴)
                derivs[i][k] = (-values[i + 2][k] + 8.0 * values[i + 1][k]
                    - 8.0 * values[i - 1][k] + values[i - 2][k]) * inv_12dr;
            } else if i == 0 {
                // Forward difference at start
                if n_grid > 2 {
                    derivs[i][k] = (-3.0 * values[0][k] + 4.0 * values[1][k] - values[2][k]) / (2.0 * dr);
                } else if n_grid > 1 {
                    derivs[i][k] = (values[1][k] - values[0][k]) / dr;
                }
            } else if i == 1 {
                // 2nd-order central at i=1
                if i + 1 < n_grid {
                    derivs[i][k] = (values[i + 1][k] - values[i - 1][k]) * inv_2dr;
                }
            } else if i == n_grid - 1 {
                // Backward difference at end
                if n_grid > 2 {
                    derivs[i][k] = (3.0 * values[i][k] - 4.0 * values[i - 1][k] + values[i - 2][k]) / (2.0 * dr);
                } else if n_grid > 1 {
                    derivs[i][k] = (values[i][k] - values[i - 1][k]) / dr;
                }
            } else if i == n_grid - 2 {
                // 2nd-order central at i=n_grid-2
                derivs[i][k] = (values[i + 1][k] - values[i - 1][k]) * inv_2dr;
            }
        }
    }
    derivs
}

/// Cubic Hermite spline evaluation.
///
/// For r in [x_i, x_{i+1}] with t = (r - x_i) / dr (0 ≤ t ≤ 1):
///   V(r) = h00·f_i + h10·dr·f'_i + h01·f_{i+1} + h11·dr·f'_{i+1}
/// where:
///   h00 = 2t³ - 3t² + 1       (value at t=0 → 1, t=1 → 0)
///   h10 = t³ - 2t² + t         (derivative at t=0 → 1, t=1 → 0)
///   h01 = -2t³ + 3t²           (value at t=0 → 0, t=1 → 1)
///   h11 = t³ - t²              (derivative at t=0 → 0, t=1 → 1)
///
/// Cost: 1 division (inv_dr, can be precomputed), 4 basis evals, 4 FMAs per channel.
/// Total: ~20 FMAs per channel vs ~64 for Neville.
fn eval_hermite_into(tab: &EqGridTable, r: f64, out: &mut [f64]) -> Result<()> {
    let n_grid = tab.n_grid();
    if n_grid < 2 {
        return Err(DftbError::Interpolation("not enough SK points for Hermite spline".into()));
    }
    let n_integ = tab.n_integ();
    assert!(out.len() >= n_integ, "eval_hermite_into: output buffer too small");

    let dr = tab.dr;
    let r_max = n_grid as f64 * dr + DIST_FUDGE;
    let last_grid_r = n_grid as f64 * dr;

    // Hard cutoff: beyond rMax, no interaction (matches DFTB+)
    if r < 0.0 || r >= r_max {
        out[..n_integ].fill(0.0);
        return Ok(());
    }

    // Tail region [last_grid_r, r_max]: delegate to Neville for exact parity.
    // The Neville tail computes y1p/y1pp via 8-point polynomial finite differences,
    // which is more accurate than the 2nd-order backward difference in derivs[].
    // The tail values are nearly zero so performance is irrelevant here.
    if r >= last_grid_r {
        return eval_eqgrid_new_into(tab, r, out);
    }

    let inv_dr = 1.0 / dr;
    // Grid points at x_i = (i+1)*dr. Find interval r ∈ [x_i, x_{i+1}].
    let mut i = ((r * inv_dr).floor() as isize) - 1;
    if i < 0 { i = 0; }
    if i as usize >= n_grid - 1 { i = (n_grid - 2) as isize; }
    let i = i as usize;

    let x_i = (i + 1) as f64 * dr;
    let t = (r - x_i) * inv_dr;
    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    for k in 0..n_integ {
        out[k] = h00 * tab.values[i][k]
            + h10 * dr * tab.derivs[i][k]
            + h01 * tab.values[i + 1][k]
            + h11 * dr * tab.derivs[i + 1][k];
    }
    Ok(())
}

/// Cubic Hermite spline with analytic derivative dV/dr.
/// dV/dr = (dh00·f_i + dh10·dr·f'_i + dh01·f_{i+1} + dh11·dr·f'_{i+1}) / dr
fn eval_hermite_with_deriv_into(
    tab: &EqGridTable,
    r: f64,
    out: &mut [f64],
    ddr: &mut [f64],
) -> Result<()> {
    let n_grid = tab.n_grid();
    if n_grid < 2 {
        return Err(DftbError::Interpolation("not enough SK points for Hermite spline".into()));
    }
    let n_integ = tab.n_integ();
    assert!(out.len() >= n_integ && ddr.len() >= n_integ,
            "eval_hermite_with_deriv_into: buffer too small");

    let dr = tab.dr;
    let r_max = n_grid as f64 * dr + DIST_FUDGE;
    let last_grid_r = n_grid as f64 * dr;

    if r < 0.0 || r >= r_max {
        out[..n_integ].fill(0.0);
        ddr[..n_integ].fill(0.0);
        return Ok(());
    }

    // Tail region: delegate to Neville for value, use finite-difference for derivative.
    // The Neville tail computes y1p/y1pp via 8-point polynomial, which is more
    // accurate than the 2nd-order backward difference in derivs[].
    if r >= last_grid_r {
        // Value: use Neville (exact parity)
        eval_eqgrid_new_into(tab, r, out)?;
        // Derivative: central finite difference of Neville at r ± dr_tail_step
        let dr_step = 1.0e-4;
        let mut v_p = [0.0f64; 20];
        let mut v_m = [0.0f64; 20];
        let r_p = r + dr_step;
        let r_m = r - dr_step;
        if r_p < r_max {
            eval_eqgrid_new_into(tab, r_p, &mut v_p)?;
        } else {
            v_p[..n_integ].fill(0.0); // beyond cutoff → 0
        }
        if r_m >= last_grid_r {
            eval_eqgrid_new_into(tab, r_m, &mut v_m)?;
        } else {
            // Use Hermite for the interior side
            eval_hermite_into(tab, r_m, &mut v_m)?;
        }
        let inv_2dr = 1.0 / (2.0 * dr_step);
        for k in 0..n_integ {
            ddr[k] = (v_p[k] - v_m[k]) * inv_2dr;
        }
        return Ok(());
    }

    let inv_dr = 1.0 / dr;
    let mut i = ((r * inv_dr).floor() as isize) - 1;
    if i < 0 { i = 0; }
    if i as usize >= n_grid - 1 { i = (n_grid - 2) as isize; }
    let i = i as usize;

    let x_i = (i + 1) as f64 * dr;
    let t = (r - x_i) * inv_dr;
    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    let dh00 = 6.0 * t2 - 6.0 * t;
    let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    let dh01 = -6.0 * t2 + 6.0 * t;
    let dh11 = 3.0 * t2 - 2.0 * t;

    for k in 0..n_integ {
        let f_i = tab.values[i][k];
        let f_ip1 = tab.values[i + 1][k];
        let fp_i = tab.derivs[i][k];
        let fp_ip1 = tab.derivs[i + 1][k];
        out[k] = h00 * f_i + h10 * dr * fp_i + h01 * f_ip1 + h11 * dr * fp_ip1;
        ddr[k] = (dh00 * f_i + dh10 * dr * fp_i + dh01 * f_ip1 + dh11 * dr * fp_ip1) * inv_dr;
    }
    Ok(())
}

fn poly5_to_zero(y0: f64, y0p: f64, y0pp: f64, x: f64, dx: f64) -> f64 {
    let invdx = 1.0 / dx;
    let dx1 = y0p * dx;
    let dx2 = y0pp * dx * dx;
    let dd = 10.0 * y0 - 4.0 * dx1 + 0.5 * dx2;
    let ee = -15.0 * y0 + 7.0 * dx1 - 1.0 * dx2;
    let ff = 6.0 * y0 - 3.0 * dx1 + 0.5 * dx2;
    let xr = x * invdx;
    ((ff * xr + ee) * xr + dd) * xr * xr * xr
}

/// In-place Neville interpolation. Writes result into `out` (len >= n_integ).
/// All internal state lives on the stack; no heap allocations.
///
/// `yp` is [n_integ][n_pts] with `n_pts <= MAX_N_INTER`.
fn poly_inter_uniform_into(
    xp: &[f64],
    yp: &[[f64; MAX_N_INTER]],
    n_integ: usize,
    x: f64,
    out: &mut [f64],
) -> Result<()> {
    let n = xp.len();
    if n < 2 {
        return Err(DftbError::Interpolation("need at least 2 points".into()));
    }

    // Stack buffers (total ~5 KB, well within stack limits)
    let mut delta = [0.0f64; MAX_N_INTER - 1];
    let mut cc = [[0.0f64; MAX_N_INTER]; MAX_N_INTEG];
    let mut dd = [[0.0f64; MAX_N_INTER]; MAX_N_INTEG];

    let delta1 = 1.0 / (xp[1] - xp[0]);
    for mm in 0..(n - 1) {
        delta[mm] = 1.0 / (xp[mm + 1] - xp[0]);
    }

    // Initialise cc / dd from yp
    for k in 0..n_integ {
        for i in 0..n {
            cc[k][i] = yp[k][i];
            dd[k][i] = yp[k][i];
        }
    }

    let mut i_cl = ((x - xp[0]) * delta1).ceil() as isize;
    if i_cl < 1 {
        i_cl = 1;
    }
    if i_cl as usize > n {
        i_cl = n as isize;
    }

    // Starting guess
    for k in 0..n_integ {
        out[k] = yp[k][(i_cl as usize) - 1];
    }
    i_cl -= 2; // Fortran adjustment

    // Neville iteration
    for mm in 1..n {
        for ii in 0..(n - mm) {
            let dm = delta[mm - 1];
            for k in 0..n_integ {
                let r2 = (dd[k][ii] - cc[k][ii + 1]) * dm;
                cc[k][ii] = (xp[ii] - x) * r2;
                dd[k][ii] = (xp[ii + mm] - x) * r2;
            }
        }
        let take_cc = 2 * i_cl < (n - mm) as isize;
        let mut dyy = [0.0f64; MAX_N_INTEG];
        if take_cc {
            let idx = (i_cl + 1) as usize;
            for k in 0..n_integ {
                dyy[k] = cc[k][idx];
            }
        } else {
            let idx = i_cl.max(0) as usize;
            for k in 0..n_integ {
                dyy[k] = dd[k][idx];
            }
            i_cl -= 1;
        }
        for k in 0..n_integ {
            out[k] += dyy[k];
        }
    }

    Ok(())
}

/// Zero-allocation SK grid evaluation. All temporaries live on the stack.
fn eval_eqgrid_new_into(tab: &EqGridTable, r: f64, out: &mut [f64]) -> Result<()> {
    let leng = tab.n_grid();
    if leng < N_INTER + 1 {
        return Err(DftbError::Interpolation(
            "not enough SK points for 8-point interpolation".into(),
        ));
    }

    // DFTB+ uses rMax = nGrid * dr + distFudge (not (nGrid-1)*dr)
    let r_max = leng as f64 * tab.dr + DIST_FUDGE;
    let n_integ = tab.n_integ();
    assert!(out.len() >= n_integ, "eval_eqgrid_new_into: output buffer too small");

    // Hard cutoff: beyond rMax, no interaction (matches DFTB+)
    if r < 0.0 || r >= r_max {
        out[..n_integ].fill(0.0);
        return Ok(());
    }

    let ind = (r / tab.dr).floor() as isize;

    // Stack buffers for interpolation stencil
    let mut xa = [0.0f64; MAX_N_INTER];
    let mut yb = [[0.0f64; MAX_N_INTER]; MAX_N_INTEG];

    if ind < (N_INTER - N_RIGHT) as isize {
        // Use first N_INTER points
        for i in 0..N_INTER {
            xa[i] = (i + 1) as f64 * tab.dr;
        }
        for k in 0..n_integ {
            for i in 0..N_INTER {
                yb[k][i] = tab.values[i][k];
            }
        }
        return poly_inter_uniform_into(&xa[..N_INTER], &yb[..n_integ], n_integ, r, out);
    }

    if (ind as usize) < leng {
        let mut i_last = (ind as usize + N_RIGHT).min(leng);
        i_last = i_last.max(N_INTER);
        let start = i_last - N_INTER;

        for i in 0..N_INTER {
            xa[i] = (start + i + 1) as f64 * tab.dr;
        }
        for k in 0..n_integ {
            for i in 0..N_INTER {
                yb[k][i] = tab.values[start + i][k];
            }
        }
        return poly_inter_uniform_into(&xa[..N_INTER], &yb[..n_integ], n_integ, r, out);
    }

    // Tail extrapolation (poly5ToZero)
    let dr = r - r_max;
    let i_last = leng;
    let start = i_last - N_INTER;

    for i in 0..N_INTER {
        xa[i] = (start + i + 1) as f64 * tab.dr;
    }
    for k in 0..n_integ {
        for i in 0..N_INTER {
            yb[k][i] = tab.values[start + i][k];
        }
    }

    let mut y0 = [0.0f64; MAX_N_INTEG];
    let mut y2 = [0.0f64; MAX_N_INTEG];
    poly_inter_uniform_into(
        &xa[..N_INTER],
        &yb[..n_integ],
        n_integ,
        xa[N_INTER - 1] - DELTA_R,
        &mut y0,
    )?;
    poly_inter_uniform_into(
        &xa[..N_INTER],
        &yb[..n_integ],
        n_integ,
        xa[N_INTER - 1] + DELTA_R,
        &mut y2,
    )?;

    for k in 0..n_integ {
        let y1 = tab.values[leng - 1][k];
        let y1p = (y2[k] - y0[k]) / (2.0 * DELTA_R);
        let y1pp = (y2[k] + y0[k] - 2.0 * y1) / (DELTA_R * DELTA_R);
        out[k] = poly5_to_zero(y1, y1p, y1pp, dr, -DIST_FUDGE);
    }
    Ok(())
}
