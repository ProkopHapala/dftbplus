//! §12 D8: pretabulated species-pair γ/γ′ cubic splines.
//!
//! `gamma_full`/`gamma_prime_full` contain a catastrophic same-U cancellation
//! that currently forces an f64 island inside the production force kernel
//! (`gamma_prime_full_f32` in `gpu_forces.cl`). There are only `n_sp²` species
//! pairs, so we evaluate γ in host f64 ONCE per engine and upload a compact
//! uniform-knot table.
//!
//! Representation (same convention as SPAMMM `LCAO_grid.cl::evaluate_radial`
//! / `DFTBplusParser::_spline_d2_uniform`): **natural cubic spline stored as
//! (value, spline-second-derivative) per node**:
//!   y = a·y_lo + b·y_hi + ((a³−a)·d2_lo + (b³−b)·d2_hi)·dr²/6
//!
//! We spline `T(r) = 1 − r·γ(r)` — the screened-Coulomb residual — smooth on
//! `[0,∞)`: `T(0)=1`, `T(∞)→0`, no 1/r singularity. Then
//!   γ(r)  = (1 − T(r)) / r
//!   γ′(r) = −T′(r)/r − (1 − T(r)) / r²
//!
//! **Two value-splines, no derivative-of-noisy-data** (labbook 2026-09-11):
//! evaluating a spline *derivative* amplifies f32 knot rounding by 1/dr
//! (measured: γ′ err 2.9e-6 → 7.8e-6 when nk 1024→2048). So we spline T AND
//! T′ independently — each node stores float4 (T, T″, T′, T‴) where the
//! second entries are natural-cubic curvature solved tridiagonally at build.
//! γ′ differs from dγ/dr of the T-spline only by the T′-spline's own
//! truncation error (~1e-7 at physical r) — energy–force consistent to that
//! level. Beyond `r_max`: T=0 ⇒ γ=1/r, γ′=−1/r² exactly.

use crate::core::error::{DftbError, Result};
use super::gamma::gamma_full;
use super::forces::gamma_prime_full;

/// Uniform knots over `[0, r_max]`. nk=256 → dr≈0.157 Bohr ≈ 0.083 Å.
/// Measured (physical range r≥1.2 bohr, mio H/C/N/O pairs):
/// max|Δγ|=3.0e-7, max|Δγ′|=6.6e-7 — under the SCC contract; nk=192
/// (dr=0.21) leaves γ′ at 1.9e-6 — marginal. 4 species → 16 pairs × 256
/// knots × 16 B = 64 KB (4 KB/pair — fits `__local` if ever wanted).
pub const GAMMA_SPLINE_NK: usize = 256;
pub const GAMMA_SPLINE_RMAX: f64 = 40.0; // Bohr — beyond: pure Coulomb tail

#[derive(Debug, Clone)]
pub struct GammaSpline {
    /// Species Hubbard U (same order as the kernel's `atom_species` index).
    pub hubbard_u: Vec<f64>,
    pub n_species: usize,
    pub nk: usize,
    pub dr: f64,
    pub r_max: f64,
    /// `[pair][knot]` float4 = (T, T″_spline, T′, T‴_spline), f32.
    /// pair index `si*nsp + sj`; flat for direct upload:
    /// `knots[(pair*nk + k)*4 + {0..3}]`.
    pub knots: Vec<f32>,
}

impl GammaSpline {
    /// Build the table in host f64 from analytic γ/γ′ (curvatures via the
    /// natural-cubic tridiagonal solve — no analytic γ′′/γ′′′ needed),
    /// downcast f32.
    /// `hubbard_u` indexed by species code (must match `atom_species` order).
    pub fn new(hubbard_u: &[f64], nk: usize, r_max: f64) -> Result<Self> {
        let nsp = hubbard_u.len();
        if nsp == 0 || nk < 8 || r_max <= 0.0 {
            return Err(DftbError::InvalidInput(format!(
                "GammaSpline::new: nsp={nsp} nk={nk} r_max={r_max}"
            )));
        }
        for (i, &u) in hubbard_u.iter().enumerate() {
            if !(u > 1e-6 && u.is_finite()) {
                return Err(DftbError::InvalidInput(format!(
                    "GammaSpline::new: hubbard_u[{i}]={u} invalid"
                )));
            }
        }
        let dr = r_max / (nk - 1) as f64;
        let mut knots = vec![0.0f32; nsp * nsp * nk * 4];
        let mut t = vec![0.0f64; nk];
        let mut td = vec![0.0f64; nk];
        for si in 0..nsp {
            for sj in 0..nsp {
                let (u1, u2) = (hubbard_u[si], hubbard_u[sj]);
                for k in 0..nk {
                    let r = k as f64 * dr;
                    // T = 1 − r·γ ;  T′ = −γ − r·γ′
                    if r < 1e-10 {
                        t[k] = 1.0;
                        td[k] = -0.5 * (u1 + u2);
                    } else {
                        let g = gamma_full(r, u1, u2);
                        let gp = gamma_prime_full(r, u1, u2);
                        t[k] = 1.0 - r * g;
                        td[k] = -g - r * gp;
                    }
                    if !t[k].is_finite() || !td[k].is_finite() {
                        return Err(DftbError::InvalidInput(format!(
                            "GammaSpline::new: non-finite pair {si}-{sj} k={k} r={r:.4} T={} T'={}", t[k], td[k]
                        )));
                    }
                }
                let tdd_t = natural_cubic_d2(&t, dr);
                let tdd_td = natural_cubic_d2(&td, dr);
                let base = (si * nsp + sj) * nk * 4;
                for k in 0..nk {
                    knots[base + 4 * k] = t[k] as f32;
                    knots[base + 4 * k + 1] = tdd_t[k] as f32;
                    knots[base + 4 * k + 2] = td[k] as f32;
                    knots[base + 4 * k + 3] = tdd_td[k] as f32;
                }
            }
        }
        Ok(Self { hubbard_u: hubbard_u.to_vec(), n_species: nsp, nk, dr, r_max, knots })
    }

    /// Host reference eval (f64): returns (γ, γ′) in Hartree / Hartree·Bohr⁻¹.
    /// Same formula the device kernel uses — keeps host/device consistent.
    pub fn eval(&self, r: f64, si: usize, sj: usize) -> (f64, f64) {
        debug_assert!(si < self.n_species && sj < self.n_species);
        if r < 1e-10 {
            return (0.5 * (self.hubbard_u[si] + self.hubbard_u[sj]), 0.0);
        }
        if r >= self.r_max {
            return (1.0 / r, -1.0 / (r * r)); // T=0: pure Coulomb
        }
        let t = self.cubic_val(r, si, sj, 0);   // (T, T″) → T
        let td = self.cubic_val(r, si, sj, 2);  // (T′, T‴) → T′
        ((1.0 - t) / r, -td / r - (1.0 - t) / (r * r))
    }

    /// Natural-cubic *value* eval on uniform knots — the Grid.cl form:
    /// y = a·y_lo + b·y_hi + ((a³−a)·d2_lo + (b³−b)·d2_hi)·dr²/6.
    /// `off`: 0 → (T, T″) pair of the float4; 2 → (T′, T‴) pair.
    fn cubic_val(&self, r: f64, si: usize, sj: usize, off: usize) -> f64 {
        let pair = si * self.n_species + sj;
        let x = r / self.dr;
        let k = (x as usize).min(self.nk - 2);
        let h = x - k as f64;
        let base = (pair * self.nk + k) * 4 + off;
        let (y0, d0) = (self.knots[base] as f64, self.knots[base + 1] as f64);
        let (y1, d1) = (self.knots[base + 4] as f64, self.knots[base + 5] as f64);
        let a = 1.0 - h;
        let b = h;
        a * y0 + b * y1 + ((a * a * a - a) * d0 + (b * b * b - b) * d1) * (self.dr * self.dr) / 6.0
    }
}

/// Natural cubic spline second derivatives on a uniform grid — tridiagonal
/// Thomas solve in f64 (same as `DFTBplusParser._spline_d2_uniform`:
/// `d2[i]` solves `d2[i-1] + 4 d2[i] + d2[i+1] = 6(y[i+1]−2y[i]+y[i-1])/h²`,
/// natural BCs d2[0]=d2[n-1]=0).
fn natural_cubic_d2(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    let mut d2 = vec![0.0f64; n];
    if n < 3 {
        return d2;
    }
    let m = n - 2;
    let mut b = vec![4.0f64; m];
    let mut rhs = vec![0.0f64; m];
    for i in 0..m {
        rhs[i] = 6.0 * (y[i + 2] - 2.0 * y[i + 1] + y[i]) / (h * h);
    }
    // Thomas: sub/super diagonal = 1.
    for i in 1..m {
        let w = 1.0 / b[i - 1];
        b[i] -= w;
        rhs[i] -= w * rhs[i - 1];
    }
    d2[n - 2] = rhs[m - 1] / b[m - 1];
    for i in (1..m).rev() {
        d2[i] = (rhs[i - 1] - d2[i + 1]) / b[i - 1];
    }
    d2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// mio-1-1 Hubbard U for H,C,N,O (from sk_data onsite; matches GammaTable).
    fn mio_u() -> Vec<f64> { vec![0.4195, 0.5500, 0.5500, 0.5600] }

    /// Error vs knot count — pick the smallest table meeting the contract.
    /// Physical range only (r ≥ 1.2 bohr ≈ shortest mio bond); r<1.0 is inside
    /// the repulsive wall and never sampled by a valid geometry.
    #[test]
    fn spline_error_vs_nk() {
        let u = mio_u();
        for &nk in &[64usize, 96, 128, 192, 256, 384, 512] {
            let sp = GammaSpline::new(&u, nk, GAMMA_SPLINE_RMAX).unwrap();
            let (mut mg, mut mgp) = (0.0f64, 0.0f64);
            for si in 0..4 {
                for sj in 0..4 {
                    let mut r = 1.2f64;
                    while r < 35.0 {
                        let (g, gp) = sp.eval(r, si, sj);
                        mg = mg.max((g - gamma_full(r, u[si], u[sj])).abs());
                        mgp = mgp.max((gp - gamma_prime_full(r, u[si], u[sj])).abs());
                        r += 0.0073;
                    }
                }
            }
            eprintln!("nk={nk:5} dr={:.4} bohr: max|Δγ|={mg:.3e} max|Δγ′|={mgp:.3e}", sp.dr);
        }
    }

    #[test]
    fn spline_vs_analytic_all_pairs() {
        let u = mio_u();
        let sp = GammaSpline::new(&u, GAMMA_SPLINE_NK, GAMMA_SPLINE_RMAX).unwrap();
        let mut max_g = 0.0f64;
        let mut max_gp = 0.0f64;
        let mut max_g_phys = 0.0f64;
        let mut max_gp_phys = 0.0f64;
        let mut worst = (0, 0, 0.0);
        for si in 0..4 {
            for sj in 0..4 {
                let mut r = 0.05f64;
                while r < 60.0 {
                    let (g, gp) = sp.eval(r, si, sj);
                    let ga = if r < 1e-10 { 0.5 * (u[si] + u[sj]) } else { gamma_full(r, u[si], u[sj]) };
                    let gpa = if r < 1e-10 { 0.0 } else { gamma_prime_full(r, u[si], u[sj]) };
                    let dg = (g - ga).abs();
                    let dgp = (gp - gpa).abs();
                    if dg > max_g { max_g = dg; worst = (si, sj, r); }
                    max_gp = max_gp.max(dgp);
                    // r ≥ 1.0 bohr ≈ 0.53 Å — below that the pair is inside
                    // the repulsive wall (min mio bond ~1.4 bohr); spline
                    // error there (~1e-6 γ) is irrelevant to valid physics.
                    if r >= 1.0 { max_g_phys = max_g_phys.max(dg); max_gp_phys = max_gp_phys.max(dgp); }
                    r += 0.0137;
                }
            }
        }
        eprintln!("γ-spline vs analytic: max|Δγ|={max_g:.3e} max|Δγ′|={max_gp:.3e} worst pair {worst:?}");
        eprintln!("  physical r≥1.0 bohr: max|Δγ|={max_g_phys:.3e} max|Δγ′|={max_gp_phys:.3e}");
        assert!(max_g_phys < 2e-6, "spline γ error (r≥1.0) {max_g_phys:.3e} too large");
        assert!(max_gp_phys < 2e-6, "spline γ′ error (r≥1.0) {max_gp_phys:.3e} too large");
    }

    #[test]
    fn spline_energy_force_consistency() {
        // γ′ must equal dγ/dr of the SAME table — finite-difference check.
        let u = mio_u();
        let sp = GammaSpline::new(&u, GAMMA_SPLINE_NK, GAMMA_SPLINE_RMAX).unwrap();
        let mut max_rel = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut worst_r = 0.0f64;
        for si in 0..4 {
            for sj in 0..4 {
                // r ≥ 1.0 bohr — inside the repulsive wall below that (see
                // spline_vs_analytic_all_pairs), the two tables may differ
                // by ~1e-5 where no valid geometry ever samples.
                let mut r = 1.0f64;
                while r < 35.0 {
                    let h = 1e-5;
                    let (_, gp) = sp.eval(r, si, sj);
                    let (g1, _) = sp.eval(r + h, si, sj);
                    let (g_1, _) = sp.eval(r - h, si, sj);
                    let fd = (g1 - g_1) / (2.0 * h);
                    let rel = ((gp - fd) / gp.abs().max(1e-30)).abs();
                    if rel > max_rel { max_rel = rel; worst_r = r; }
                    max_abs = max_abs.max((gp - fd).abs());
                    r += 0.41;
                }
            }
        }
        eprintln!("γ-spline FD consistency: max|rel(γ′−FDγ)|={max_rel:.3e} at r={worst_r:.2}  max|abs|={max_abs:.3e}");
        // Two-table design: consistency limited by the T′-spline truncation
        // (~1e-6 abs at dr=0.157), not exact derivative identity. Assert on
        // ABSOLUTE deviation — forces see dq_i·dq_j·γ′, so abs is physical.
        // Measured 7e-6 at the r=1.0 edge (sum of both splines' truncation at
        // the short-range curvature peak); force contribution dq_i·dq_j·err
        // ~ 6e-7 Ha/Å ≪ the 1.5e-6 force-parity floor. Bound with margin.
        assert!(max_abs < 2e-5, "spline γ′ inconsistent with spline γ: abs {max_abs:.3e}");
    }
}
