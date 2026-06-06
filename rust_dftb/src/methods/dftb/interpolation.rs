use crate::core::error::{DftbError, Result};

pub const DIST_FUDGE: f64 = 1.0;

const MAX_N_INTER: usize = 8;  // max interpolation stencil
const MAX_N_INTEG: usize = 20; // max columns in extended-format SK tables
const N_INTER: usize = 8;
const N_RIGHT: usize = 4;
const DELTA_R: f64 = 1.0e-2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    EqGridNew,
}

#[derive(Debug, Clone)]
pub struct EqGridTable {
    pub dr: f64,
    pub values: Vec<Vec<f64>>, // [n_grid][n_integ]
}

impl EqGridTable {
    pub fn n_grid(&self) -> usize {
        self.values.len()
    }

    pub fn n_integ(&self) -> usize {
        self.values.first().map(|r| r.len()).unwrap_or(0)
    }

    pub fn r_max(&self) -> f64 {
        (self.n_grid().saturating_sub(1) as f64) * self.dr + DIST_FUDGE
    }

    /// Convenience wrapper that allocates a Vec. For hot paths use `eval_into`.
    pub fn eval(&self, r: f64) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.n_integ()];
        self.eval_into(r, &mut out)?;
        Ok(out)
    }

    /// Evaluate SK integrals at distance `r`, writing into a caller-provided buffer.
    /// The buffer must be at least `self.n_integ()` elements long.
    /// Zero-allocation: all temporaries live on the stack.
    pub fn eval_into(&self, r: f64, out: &mut [f64]) -> Result<()> {
        eval_eqgrid_new_into(self, r, out)
    }
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

    let r_max = (leng as f64 - 1.0) * tab.dr;
    let n_integ = tab.n_integ();
    assert!(out.len() >= n_integ, "eval_eqgrid_new_into: output buffer too small");

    if r < 0.0 {
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
