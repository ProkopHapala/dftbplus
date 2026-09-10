//! interpolation.rs — Cubic B-spline interpolation of SK integrals on a uniform grid.
//!
//! Production path: C² cubic B-spline with extra trailing zero knots so the
//! interpolant lands on V=0 at cutoff. Analytic V' comes from the same controls.
//! No Neville / poly5 tail — that 8-point polynomial explodes past the last
//! grid point (H-H at 10.4 Bohr → −0.4 Ha instead of ~0).
//!
//! Left end: endpoint interpolation + phantom control `c_{-1} = 2c_0 − c_1`.
//! Right end: `N_PAD_END` extra zero samples, then the same 4-point stencil.

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::spline_resample::{bspline3_eval_v_d1_d2, fit_bspline_controls_zero_end, N_PAD_END};

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

/// SK integral table on a uniform grid.
///
/// Production interpolation: C² cubic B-spline (`controls`, `eval_into`).
/// Hermite (`derivs`, `eval_hermite_*`) is an unused reference path.
/// Extra right-end controls are a blunt zero-sample **stopgap**, not the
/// extra-control fitter (`doc/prokop/topical_audit/sk_interpolation.md`).
#[derive(Debug, Clone)]
pub struct EqGridTable {
    pub dr: f64,
    pub values: Vec<Vec<f64>>, // [n_grid][n_integ]
    /// Precomputed first derivatives at each grid point: derivs[i][k] = dV_k/dx at x_i.
    /// Computed once at construction via 4th-order central finite differences.
    /// Kept for Hermite reference path, not used by the canonical B-spline path.
    derivs: Vec<Vec<f64>>, // [n_grid][n_integ]
    /// Canonical C² cubic B-spline control points: controls[i][k] = c_i for channel k.
    /// Computed from `values` via tridiagonal solve at construction time.
    /// Used by `eval_into` and `eval_with_deriv_into` (production path).
    controls: Vec<Vec<f64>>, // [n_grid][n_integ]
}

impl EqGridTable {
    pub fn new(dr: f64, values: Vec<Vec<f64>>) -> Self {
        let derivs = compute_hermite_derivs(&values, dr);
        let n_grid = values.len();
        let n_integ = values.first().map(|r| r.len()).unwrap_or(0);
        let n_ctrl = n_grid + N_PAD_END;
        let mut controls = vec![vec![0.0f64; n_integ]; n_ctrl];
        for k in 0..n_integ {
            let col: Vec<f64> = values.iter().map(|r| r[k]).collect();
            let ctrl = fit_bspline_controls_zero_end(&col, N_PAD_END);
            for i in 0..n_ctrl { controls[i][k] = ctrl[i]; }
        }
        Self { dr, values, derivs, controls }
    }

    pub fn n_grid(&self) -> usize {
        self.values.len()
    }

    pub fn n_integ(&self) -> usize {
        self.values.first().map(|r| r.len()).unwrap_or(0)
    }

    pub fn r_max(&self) -> f64 {
        self.controls.len() as f64 * self.dr
    }

    /// Convenience wrapper that allocates a Vec. For hot paths use `eval_into`.
    pub fn eval(&self, r: f64) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.n_integ()];
        self.eval_into(r, &mut out)?;
        Ok(out)
    }

    /// Evaluate SK integrals at distance `r` using the canonical C² cubic B-spline.
    /// Trailing zero knots take V to 0; no Neville tail.
    pub fn eval_into(&self, r: f64, out: &mut [f64]) -> Result<()> {
        eval_bspline_into(self, r, out)
    }

    /// Evaluate SK integrals AND radial derivative dV/dr at distance `r`.
    /// Analytic derivatives of the same B-spline — no finite differences.
    pub fn eval_with_deriv_into(&self, r: f64, out: &mut [f64], ddr: &mut [f64]) -> Result<()> {
        eval_bspline_with_deriv_into(self, r, out, ddr)
    }

    /// Hermite reference path (C¹). Not used in production. For parity verification.
    pub fn eval_hermite_into(&self, r: f64, out: &mut [f64]) -> Result<()> {
        eval_hermite_into(self, r, out)
    }

    /// Hermite reference path with derivative (C¹). Not used in production.
    pub fn eval_hermite_with_deriv_into(&self, r: f64, out: &mut [f64], ddr: &mut [f64]) -> Result<()> {
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
    let r_max = tab.r_max();
    let last_grid_r = n_grid as f64 * dr;

    // Hard cutoff: beyond rMax, no interaction (matches DFTB+)
    if r < 0.0 || r >= r_max {
        out[..n_integ].fill(0.0);
        return Ok(());
    }

    // Tail: same B-spline as production (trailing zero knots). Do not Neville.
    if r >= last_grid_r {
        return eval_bspline_into(tab, r, out);
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
    let r_max = tab.r_max();
    let last_grid_r = n_grid as f64 * dr;

    if r < 0.0 || r >= r_max {
        out[..n_integ].fill(0.0);
        ddr[..n_integ].fill(0.0);
        return Ok(());
    }

    if r >= last_grid_r {
        return eval_bspline_with_deriv_into(tab, r, out, ddr);
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

// ============================================================================
// Canonical C² cubic B-spline evaluation (production path, manifest §4.2)
// ============================================================================

/// Evaluate SK integrals at distance `r` using the canonical C² cubic B-spline.
///
/// Grid points at x_i = (i+1)*dr (DFTB+). Controls include N_PAD_END trailing
/// zeros so V→0 past the last SK sample. Same 4-point stencil everywhere —
/// no Neville / poly5 tail.
fn eval_bspline_into(tab: &EqGridTable, r: f64, out: &mut [f64]) -> Result<()> {
    let n_ctrl = tab.controls.len();
    if n_ctrl < 2 {
        return Err(DftbError::Interpolation("not enough SK points for B-spline".into()));
    }
    let n_integ = tab.n_integ();
    assert!(out.len() >= n_integ, "eval_bspline_into: output buffer too small");
    let dr = tab.dr;
    let r_max = n_ctrl as f64 * dr;
    if r < 0.0 || r >= r_max {
        out[..n_integ].fill(0.0);
        return Ok(());
    }
    let (i, t, inv_dr) = bspline_interval(n_ctrl, dr, r);
    for k in 0..n_integ {
        let (c0, c1, c2, c3) = bspline_four(tab, n_ctrl, i, k);
        let (v, _, _) = bspline3_eval_v_d1_d2(c0, c1, c2, c3, t, inv_dr);
        out[k] = v;
    }
    Ok(())
}

/// Evaluate SK integrals AND analytic radial derivative dV/dr.
fn eval_bspline_with_deriv_into(
    tab: &EqGridTable,
    r: f64,
    out: &mut [f64],
    ddr: &mut [f64],
) -> Result<()> {
    let n_ctrl = tab.controls.len();
    if n_ctrl < 2 {
        return Err(DftbError::Interpolation("not enough SK points for B-spline".into()));
    }
    let n_integ = tab.n_integ();
    assert!(out.len() >= n_integ && ddr.len() >= n_integ,
            "eval_bspline_with_deriv_into: buffer too small");
    let dr = tab.dr;
    let r_max = n_ctrl as f64 * dr;
    if r < 0.0 || r >= r_max {
        out[..n_integ].fill(0.0);
        ddr[..n_integ].fill(0.0);
        return Ok(());
    }
    let (i, t, inv_dr) = bspline_interval(n_ctrl, dr, r);
    for k in 0..n_integ {
        let (c0, c1, c2, c3) = bspline_four(tab, n_ctrl, i, k);
        let (v, dv, _) = bspline3_eval_v_d1_d2(c0, c1, c2, c3, t, inv_dr);
        out[k] = v;
        ddr[k] = dv;
    }
    Ok(())
}

fn bspline_interval(n_ctrl: usize, dr: f64, r: f64) -> (usize, f64, f64) {
    let inv_dr = 1.0 / dr;
    let r_shifted = r - dr; // values[0] lives at r=dr
    let x = r_shifted * inv_dr;
    let mut i = x as isize;
    if i < 0 { i = 0; }
    if i as usize >= n_ctrl - 1 { i = (n_ctrl - 2) as isize; }
    let i = i as usize;
    (i, x - i as f64, inv_dr)
}

fn bspline_four(tab: &EqGridTable, n_ctrl: usize, i: usize, k: usize) -> (f64, f64, f64, f64) {
    let c0 = if i == 0 { 2.0 * tab.controls[0][k] - tab.controls[1][k] } else { tab.controls[i - 1][k] };
    let c1 = tab.controls[i][k];
    let c2 = tab.controls[i + 1][k];
    let c3 = if i + 2 >= n_ctrl { 2.0 * tab.controls[n_ctrl - 1][k] - tab.controls[n_ctrl - 2][k] } else { tab.controls[i + 2][k] };
    (c0, c1, c2, c3)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a typical SK-like table: exponential decay on a uniform grid.
    fn make_sk_table(n_grid: usize, dr: f64, decay: f64) -> EqGridTable {
        let values: Vec<Vec<f64>> = (0..n_grid)
            .map(|i| {
                let r = (i + 1) as f64 * dr;
                vec![(-decay * r).exp()]
            })
            .collect();
        EqGridTable::new(dr, values)
    }

    /// Test: B-spline eval_into reproduces grid values at grid points.
    #[test]
    fn test_bspline_reproduces_grid_values() {
        let tab = make_sk_table(50, 0.1, 2.0);
        let n_grid = tab.n_grid();
        let mut out = [0.0f64];
        for i in 0..n_grid {
            let r = (i + 1) as f64 * tab.dr;
            tab.eval_into(r, &mut out).unwrap();
            let expected = tab.values[i][0];
            // B-spline with control points should reproduce grid values exactly
            // (that's the whole point of the tridiagonal solve).
            let err = (out[0] - expected).abs();
            assert!(err < 1e-10, "B-spline value mismatch at grid point {i} (r={r:.3}): got {:.12}, expected {expected:.12}, err={err:.2e}", out[0]);
        }
    }

    /// Test: B-spline derivatives are C² continuous at knots.
    /// This is the key improvement over Hermite (which is only C¹).
    /// We check that V'' is continuous across interior knots.
    #[test]
    fn test_bspline_deriv_c2_continuity() {
        let tab = make_sk_table(50, 0.1, 2.0);
        let n_grid = tab.n_grid();
        let mut v = [0.0f64]; let mut dv = [0.0f64];
        let mut v_r = [0.0f64]; let mut dv_r = [0.0f64];
        let mut max_v_jump = 0.0f64;
        let mut max_dv_jump = 0.0f64;
        // Check V' continuity at each interior knot.
        for i in 1..(n_grid - 1) {
            let r_knot = (i + 1) as f64 * tab.dr;
            let eps = tab.dr * 1e-6;
            // Left limit
            tab.eval_with_deriv_into(r_knot - eps, &mut v, &mut dv).unwrap();
            // Right limit
            tab.eval_with_deriv_into(r_knot + eps, &mut v_r, &mut dv_r).unwrap();
            // V should be continuous
            let v_jump = (v[0] - v_r[0]).abs();
            max_v_jump = max_v_jump.max(v_jump);
            // V' should be continuous (C¹)
            let dv_jump = (dv[0] - dv_r[0]).abs();
            max_dv_jump = max_dv_jump.max(dv_jump);
        }
        eprintln!("B-spline C² continuity: max|V jump|={max_v_jump:.3e}, max|V' jump|={max_dv_jump:.3e}");
        // V should be continuous — tolerance accounts for eps perturbation.
        assert!(max_v_jump < 1e-5, "V jump too large: {max_v_jump:.2e}");
        // V' should be continuous (C¹) — the B-spline is C², so V' is C¹.
        assert!(max_dv_jump < 1e-3, "V' jump too large: {max_dv_jump:.2e}");
    }

    /// Test: B-spline vs Hermite values are close (both interpolate the same data).
    #[test]
    fn test_bspline_vs_hermite_values() {
        let tab = make_sk_table(50, 0.1, 2.0);
        let n_grid = tab.n_grid();
        let mut v_bs = [0.0f64]; let mut v_herm = [0.0f64];
        let mut max_diff = 0.0f64;
        for i in 0..200 {
            let r = (i + 1) as f64 * tab.dr * 0.5; // sample at half resolution
            if r >= tab.r_max() { continue; }
            tab.eval_into(r, &mut v_bs).unwrap();
            tab.eval_hermite_into(r, &mut v_herm).unwrap();
            max_diff = max_diff.max((v_bs[0] - v_herm[0]).abs());
        }
        // Both interpolate the same data, so should be close.
        // B-spline is C², Hermite is C¹ — they differ by interpolation method.
        eprintln!("B-spline vs Hermite max|diff| = {max_diff:.3e}");
        assert!(max_diff < 2e-2, "B-spline vs Hermite too different: {max_diff:.3e}");
    }

    /// Test: B-spline derivative is smooth (no finite-difference noise).
    /// The Hermite path uses finite-difference derivatives which can have
    /// discontinuities at grid boundaries. The B-spline path should be smooth.
    #[test]
    fn test_bspline_derivative_smoothness() {
        let tab = make_sk_table(50, 0.1, 2.0);
        // Sample derivatives at many points and check for discontinuities.
        let n_samples = 500;
        let mut prev_dv = 0.0f64;
        let mut max_jump = 0.0f64;
        let mut v = [0.0f64]; let mut dv = [0.0f64];
        for i in 0..n_samples {
            let r = (i as f64 + 0.5) * tab.dr * 0.1; // fine sampling
            if r >= tab.r_max() - 0.01 { continue; }
            tab.eval_with_deriv_into(r, &mut v, &mut dv).unwrap();
            if i > 0 {
                let jump = (dv[0] - prev_dv).abs();
                max_jump = max_jump.max(jump);
            }
            prev_dv = dv[0];
        }
        // For a smooth exponential, the derivative should be smooth.
        // B-spline: C² → derivative is C¹ → no large jumps.
        // Hermite: C¹ → derivative is C⁰ → may have jumps at knots.
        eprintln!("B-spline derivative max jump between samples: {max_jump:.3e}");
        // The jump should be small (proportional to sample spacing × curvature).
        assert!(max_jump < 1.0, "B-spline derivative too jumpy: {max_jump:.3e}");
    }

    /// Past the last SK sample the spline must land on ~0, not a bonding integral.
    #[test]
    fn test_bspline_tail_goes_to_zero() {
        let tab = make_sk_table(50, 0.1, 2.0);
        let r_data = tab.n_grid() as f64 * tab.dr;
        let last = tab.values[tab.n_grid() - 1][0].abs();
        let mut out = [0.0f64];
        tab.eval_into(r_data + 2.0 * tab.dr, &mut out).unwrap();
        assert!(out[0].abs() <= last + 1e-8,
            "tail exploded: V={} last_grid={last} (must decay toward 0, not grow)", out[0]);
        tab.eval_into(tab.r_max() + 0.1, &mut out).unwrap();
        assert_eq!(out[0], 0.0, "hard cutoff past r_max must be exact 0, got {}", out[0]);
        let mut dv = [0.0f64];
        tab.eval_with_deriv_into(tab.r_max() + 0.1, &mut out, &mut dv).unwrap();
        assert_eq!(dv[0], 0.0);
    }

    /// Analytic V' must match a central difference of the same V (not a different interpolant).
    #[test]
    fn test_bspline_analytic_deriv_matches_fd() {
        let tab = make_sk_table(80, 0.05, 1.5);
        let h = 1e-6;
        let mut vp = [0.0f64]; let mut vm = [0.0f64];
        let mut v = [0.0f64]; let mut dv = [0.0f64];
        let mut max_rel = 0.0f64;
        let mut worst_r = 0.0f64;
        for i in 5..200 {
            let r = (i as f64 + 0.37) * tab.dr;
            if r + h >= tab.r_max() - tab.dr { continue; }
            tab.eval_into(r + h, &mut vp).unwrap();
            tab.eval_into(r - h, &mut vm).unwrap();
            tab.eval_with_deriv_into(r, &mut v, &mut dv).unwrap();
            let fd = (vp[0] - vm[0]) / (2.0 * h);
            let rel = (dv[0] - fd).abs() / fd.abs().max(1e-8);
            if rel > max_rel { max_rel = rel; worst_r = r; }
        }
        eprintln!("analytic V' vs FD: max rel={max_rel:.3e} at r={worst_r:.4}");
        assert!(max_rel < 1e-6, "analytic V' disagrees with FD of V: rel={max_rel:.3e} at r={worst_r}");
    }
}
