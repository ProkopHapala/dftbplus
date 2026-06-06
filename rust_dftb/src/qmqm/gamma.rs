//! Gamma-function evaluation for atom-resolved SCC electrostatics.
//!
//! The DFTB γ_AB(R) smoothly interpolates between the on-site Hubbard U
//! (R→0) and the bare Coulomb 1/R (R→∞). The short-range screening uses
//! the exponential form from Elstner et al. (Phys. Rev. B 58, 7260).
//!
//! Implementation follows `dftbp_dftb_shortgammafuncs::expGamma` in the
//! Fortran reference. For inter-fragment coupling we need the *full* γ,
//! i.e. `1/R - S(R)`, whereas `expGamma` alone is only the short-range
//! screening term.

/// Tolerance for “same distance” (on-top atoms). Must match Fortran.
const TOL_SAME_DIST: f64 = 1.0e-10;

/// Tolerance for “same Hubbard U”. Must match Fortran `minHubDiff`.
const MIN_HUB_DIFF: f64 = 1.0e-4;

/// Minimal Hubbard U considered valid. Must match Fortran `minHubTol`.
const MIN_HUB_TOL: f64 = 1.0e-10;

/// Hubbard-to-Slater exponent conversion: τ = 16/5 · U = 3.2·U.
const TAU_FACTOR: f64 = 3.2;

/// Pre-computed polynomial coefficients for the same-U screening function.
/// `S(R) = exp(-τR) * (1/R + c0·τ + c1·R·τ² + c2·R²·τ³)`
const SAME_U_C0: f64 = 0.6875;                  // 11/16
const SAME_U_C1: f64 = 0.1875;                  // 3/16
const SAME_U_C2: f64 = 0.020_833_333_333_333_333; // 1/48

/// Evaluate the full DFTB γ_AB(R) for two atoms with Hubbard U values `u1`, `u2`.
///
/// This is the effective electrostatic interaction that enters the SCC shift:
/// `V_A = Σ_B γ(R_AB, U_A, U_B) · (q_B - q0_B)`.
///
/// Special cases:
/// - `R < tol`  → `(U1 + U2) / 2`  (on-site limit)
/// - same U, R>0 → `1/R - exp(-τR)·(1/R + polynomial in τR)`
/// - different U → `1/R - [gamma_sub(R,τ1,τ2) + gamma_sub(R,τ2,τ1)]`
///
/// # Safety / correctness
/// Negative or extremely small `U` values are rejected because they lead to
/// unphysical divergences in the screening function.
pub fn gamma_full(r: f64, u1: f64, u2: f64) -> f64 {
    // Defensive: reject unphysical input.
    if r < 0.0 {
        panic!("gamma_full called with negative distance {r}");
    }
    if u1 < MIN_HUB_TOL || u2 < MIN_HUB_TOL {
        panic!("gamma_full: Hubbard U too small ({u1}, {u2})");
    }

    // On-site / on-top limit.
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

/// Same-τ screening function `S(R)` used inside `gamma_full`.
/// `τ_mean = 3.2 · U_mean`.
fn exp_gamma_same_u(r: f64, tau_mean: f64) -> f64 {
    let x = -tau_mean * r;
    let e = x.exp();
    e * (1.0 / r
        + SAME_U_C0 * tau_mean
        + SAME_U_C1 * r * tau_mean * tau_mean
        + SAME_U_C2 * r * r * tau_mean.powi(3))
}

/// Sub-expression for different-τ case (one half of the screening).
/// Corresponds to Fortran `gammaSubExprn_`.
///
/// `tau1` is the exponent in the exponential, `tau2` is the polynomial partner.
/// Panics if `|tau1-tau2|` is too small (degenerate, should have used same-U branch)
/// or if `r` is on-top.
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

/// Lookup table for per-species Hubbard U values and pair pre-screening.
///
/// For small systems the table is tiny (≤10 species → ≤100 pairs).
/// All data is stored in flat `Vec`s indexed by `u8` species codes.
#[derive(Debug, Clone)]
pub struct GammaTable {
    /// Hubbard U per species, indexed by species code.
    pub hubbard_u: Vec<f64>,
    /// Pre-computed per-pair cutoff where `gamma` drops below a threshold.
    /// `cutoffs[si * n_species + sj]`.
    pub cutoffs: Vec<f64>,
    pub n_species: usize,
}

impl GammaTable {
    /// Build table from a slice of Hubbard U values (one per species).
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

    /// Hubbard U for a given species code.
    #[inline]
    pub fn u(&self, species: u8) -> f64 {
        self.hubbard_u[species as usize]
    }

    /// γ(R) for a specific species pair.
    #[inline]
    pub fn gamma(&self, r: f64, sp1: u8, sp2: u8) -> f64 {
        gamma_full(r, self.u(sp1), self.u(sp2))
    }

    /// Maximum cutoff among all pairs (useful for neighbour-list construction).
    #[inline]
    pub fn max_cutoff(&self) -> f64 {
        self.cutoffs.iter().copied().fold(0.0_f64, f64::max)
    }
}

/// Rough estimate of the distance where γ has decayed to a negligible value.
/// Uses the same bisection logic as Fortran `expGammaCutoff` but simplified.
fn estimate_cutoff(u1: f64, u2: f64) -> f64 {
    // For our purposes a hard 30 Å cap is safe for organic species.
    // A tighter value can be derived from the bisection in expGammaCutoff.
    let tau = TAU_FACTOR * 0.5 * (u1 + u2);
    // Solve exp(-τR)·(1/R + τ) ≈ threshold.  Heuristic:
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
        // At large R gamma → 1/R
        let r = 20.0;
        let u = 0.5;
        let g = gamma_full(r, u, u);
        let coulomb = 1.0 / r;
        assert!((g - coulomb).abs() < 1e-4,
            "at 20 Å gamma should be ~1/R = {coulomb}, got {g}");
    }

    #[test]
    fn gamma_same_u_intermediate() {
        // At ~0.5 Å gamma should be close to U
        let r = 0.5;
        let u = 0.5;
        let g = gamma_full(r, u, u);
        assert!(g > 0.4 && g < 0.55,
            "at 0.5 Å gamma should be near U=0.5, got {g}");
    }
}
