//! Gamma-function evaluation for atom-resolved SCC electrostatics.
//!
//! The DFTB γ_AB(R) smoothly interpolates between the on-site Hubbard U
//! (R→0) and the bare Coulomb 1/R (R→∞). The short-range screening uses
//! the exponential form from Elstner et al. (Phys. Rev. B 58, 7260).

const TOL_SAME_DIST: f64 = 1.0e-10;
const MIN_HUB_DIFF: f64 = 1.0e-4;
const MIN_HUB_TOL: f64 = 1.0e-10;
const TAU_FACTOR: f64 = 3.2;
const SAME_U_C0: f64 = 0.6875;
const SAME_U_C1: f64 = 0.1875;
const SAME_U_C2: f64 = 0.020_833_333_333_333_333;

pub fn gamma_full(r: f64, u1: f64, u2: f64) -> f64 {
    if r < 0.0 {
        panic!("gamma_full called with negative distance {r}");
    }
    if u1 < MIN_HUB_TOL || u2 < MIN_HUB_TOL {
        panic!("gamma_full: Hubbard U too small ({u1}, {u2})");
    }
    if r < TOL_SAME_DIST {
        return 0.5 * (u1 + u2);
    }

    let tau1 = TAU_FACTOR * u1;
    let tau2 = TAU_FACTOR * u2;

    let short_range = if (u1 - u2).abs() < MIN_HUB_DIFF {
        exp_gamma_same_u(r, 0.5 * (tau1 + tau2))
    } else {
        gamma_sub_exprn(r, tau1, tau2) + gamma_sub_exprn(r, tau2, tau1)
    };

    1.0 / r - short_range
}

fn exp_gamma_same_u(r: f64, tau_mean: f64) -> f64 {
    let x = -tau_mean * r;
    let e = x.exp();
    e * (1.0 / r
        + SAME_U_C0 * tau_mean
        + SAME_U_C1 * r * tau_mean * tau_mean
        + SAME_U_C2 * r * r * tau_mean.powi(3))
}

fn gamma_sub_exprn(r: f64, tau1: f64, tau2: f64) -> f64 {
    if (tau1 - tau2).abs() < TAU_FACTOR * MIN_HUB_DIFF {
        panic!("gamma_sub_exprn: degenerate tau values ({tau1}, {tau2})");
    }
    if r < TOL_SAME_DIST {
        panic!("gamma_sub_exprn: atoms on top of each other (r={r})");
    }

    let dt2 = tau1 * tau1 - tau2 * tau2;
    let dt2_sq = dt2 * dt2;
    let dt2_cu = dt2_sq * dt2;

    let term_a = 0.5 * tau2.powi(4) * tau1 / dt2_sq;
    let term_b = (tau2.powi(6) - 3.0 * tau2.powi(4) * tau1 * tau1) / (r * dt2_cu);

    (-tau1 * r).exp() * (term_a - term_b)
}

use crate::core::error::Result;
use crate::methods::dftb::sk_data::SkData;

#[derive(Debug, Clone)]
pub struct GammaTable {
    pub hubbard_u: Vec<f64>,
    pub cutoffs: Vec<f64>,
    pub n_species: usize,
}

impl GammaTable {
    /// Build a GammaTable from SkData, extracting Hubbard U for each unique species
    /// in the order they appear in `species`.
    pub fn from_sk_data(sk: &SkData, species: &[String]) -> Result<Self> {
        let mut unique: Vec<String> = Vec::new();
        for sp in species {
            if !unique.contains(sp) {
                unique.push(sp.clone());
            }
        }
        let hubbard_u: Vec<f64> = unique
            .iter()
            .map(|sp| sk.onsite(sp).map(|p| p.u_hubbard).unwrap_or(0.4))
            .collect();
        Ok(Self::from_hubbard_u(hubbard_u))
    }

    pub fn from_hubbard_u(hubbard_u: Vec<f64>) -> Self {
        let n = hubbard_u.len();
        let mut cutoffs = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                cutoffs[i * n + j] = estimate_cutoff(hubbard_u[i], hubbard_u[j]);
            }
        }
        Self {
            hubbard_u,
            cutoffs,
            n_species: n,
        }
    }

    #[inline]
    pub fn u(&self, species: u8) -> f64 {
        self.hubbard_u[species as usize]
    }

    #[inline]
    pub fn gamma(&self, r: f64, sp1: u8, sp2: u8) -> f64 {
        gamma_full(r, self.u(sp1), self.u(sp2))
    }

    #[inline]
    pub fn max_cutoff(&self) -> f64 {
        self.cutoffs.iter().copied().fold(0.0_f64, f64::max)
    }
}

fn estimate_cutoff(u1: f64, u2: f64) -> f64 {
    let tau = TAU_FACTOR * 0.5 * (u1 + u2);
    if tau > 0.0 {
        (-(1.0_f64.ln()) / tau).min(30.0).max(5.0)
    } else {
        30.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_onsite() {
        let g = gamma_full(0.0, 0.5, 0.4);
        assert!((g - 0.45).abs() < 1e-10, "onsite gamma should be average U, got {g}");
    }

    #[test]
    fn gamma_same_u_large_r() {
        let r = 20.0;
        let u = 0.5;
        let g = gamma_full(r, u, u);
        let coulomb = 1.0 / r;
        assert!((g - coulomb).abs() < 1e-4,
            "at 20 Å gamma should be ~1/R = {coulomb}, got {g}");
    }

    #[test]
    fn gamma_same_u_intermediate() {
        let r = 0.5;
        let u = 0.5;
        let g = gamma_full(r, u, u);
        assert!(g > 0.4 && g < 0.55,
            "at 0.5 Å gamma should be near U=0.5, got {g}");
    }
}
