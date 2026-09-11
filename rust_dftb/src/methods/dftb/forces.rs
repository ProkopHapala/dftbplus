//! DFTB force calculation (CPU, single system).
//!
//! Implements the four DFTB force components:
//!   F_total = F_nonSCC + F_SCC_shift + F_SCC_dc + F_rep
//!
//! References:
//!   - `src/dftbp/dftb/forces.F90` (Fortran reference)
//!   - `src/dftbp/dftb/nonscc.F90` (finite-difference H/S derivatives)
//!   - `src/dftbp/dftb/shortgammafuncs.F90` (gamma derivative)
//!   - `src/dftbp/dftb/coulomb.F90` (1/R derivative)
//!   - `src/dftbp/dftb/repulsive/splinerep.F90` (repulsive spline)
//!
//! # Conventions
//! - Coordinates are in Ångström (same as the rest of the Rust crate).
//! - Gamma distances are converted to Bohr via `ANG2BOHR = 1.889726133`.
//! - `delta_q = q_electronic - q0` (Rust convention; opposite to DFTB+ which
//!   uses `deltaQ = q0 - q_electronic`). All force signs below follow the
//!   Rust convention and have been cross-checked against the Fortran code.
//! - Forces are returned in Hartree/Ångström (consistent with the
//!   coordinate unit used for finite differences). DFTB+ reports forces in
//!   Hartree/Bohr; the test driver converts to that unit for comparison.
//!
//! # Ownership note
//! This module is owned by Agent_3 (Wave 1, CPU forces). It deliberately
//! re-implements `gamma_prime_full` and the repulsive spline parser locally
//! because `gamma.rs` and `sk_data.rs` are owned by other agents and must
//! not be modified. See the handoff report for requested coordinator edits
//! that would let these helpers move to their canonical modules.

use crate::core::error::{DftbError, Result};
use crate::core::neighbor::{NeighborBuilder, NeighborList};
use crate::methods::dftb::hamiltonian::{Hamiltonian, HamiltonianBuilder, SccResult, SystemContext};
use crate::methods::dftb::rotation::{DirectionCosines, Rotation};
use crate::methods::dftb::sk_data::SkData;
use crate::qmqm::gamma::GammaTable;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use nalgebra::linalg::Cholesky;

const ANG2BOHR: f64 = 1.889_726_133;
const BOHR2ANG: f64 = 1.0 / ANG2BOHR;

/// Finite-difference step for H/S block derivatives.
/// Matches DFTB+ `deltaXDiff = epsilon(1.0_dp)**0.25` ≈ 1.192e-4 (in Bohr).
/// `epsilon(1.0_dp)` in double precision is 2.220446049250313e-16, so
/// `epsilon**0.25 ≈ 1.1920928955078125e-4`.
const DELTA_X_DIFF_BOHR: f64 = 1.1920928955078125e-4;
pub const DELTA_X_DIFF_ANG: f64 = DELTA_X_DIFF_BOHR * BOHR2ANG;

/// Minimum neighbour distance (matches Fortran `minNeighDist = 1.0e-2`).
const MIN_NEIGH_DIST: f64 = 1.0e-2;

// ─── Gamma derivative (local copy; gamma.rs is not owned by Agent_3) ───────

const TOL_SAME_DIST: f64 = 1.0e-10;
const MIN_HUB_DIFF: f64 = 1.0e-4;
const MIN_HUB_TOL: f64 = 1.0e-10;
const TAU_FACTOR: f64 = 3.2;
const SAME_U_C0: f64 = 0.6875;
const SAME_U_C1: f64 = 0.1875;
const SAME_U_C2: f64 = 0.020_833_333_333_333_333;

/// Derivative of the full DFTB γ_AB(R) with respect to R.
///
/// `gamma_full(R) = 1/R - S(R)` where `S(R)` is the short-range screening.
/// Therefore `gamma_full'(R) = -1/R^2 - S'(R)`.
///
/// `S'(R)` is `expGammaPrime(R, U1, U2)` in the Fortran reference. The
/// Coulomb part `-1/R^2` is added here so that callers get the full
/// derivative in one call (matching how DFTB+ combines the short-range
/// and Coulomb contributions in the SCC double-counting force).
pub fn gamma_prime_full(r: f64, u1: f64, u2: f64) -> f64 {
    assert!(r >= 0.0, "gamma_prime_full: negative distance {r}");
    assert!(u1 >= MIN_HUB_TOL && u2 >= MIN_HUB_TOL,
        "gamma_prime_full: Hubbard U too small ({u1}, {u2})");

    // On-site: derivative is 0 (gamma is flat at R=0).
    if r < TOL_SAME_DIST {
        return 0.0;
    }

    let short_prime = if (u1 - u2).abs() < MIN_HUB_DIFF {
        let tau_mean = TAU_FACTOR * 0.5 * (u1 + u2);
        exp_gamma_same_u_prime(r, tau_mean)
    } else {
        let tau1 = TAU_FACTOR * u1;
        let tau2 = TAU_FACTOR * u2;
        gamma_sub_exprn_prime(r, tau1, tau2) + gamma_sub_exprn_prime(r, tau2, tau1)
    };

    // gamma_full = 1/R - S(R)  →  gamma_full' = -1/R^2 - S'(R)
    -1.0 / (r * r) - short_prime
}

/// Derivative of `exp_gamma_same_u` (the same-τ screening function).
fn exp_gamma_same_u_prime(r: f64, tau: f64) -> f64 {
    let e = (-tau * r).exp();
    // S(R) = e * (1/R + c0*τ + c1*R*τ² + c2*R²*τ³)
    // S'(R) = -τ*e*(...) + e*(-1/R² + c1*τ² + 2*c2*R*τ³)
    let poly = 1.0 / r + SAME_U_C0 * tau + SAME_U_C1 * r * tau * tau
        + SAME_U_C2 * r * r * tau.powi(3);
    let poly_prime = -1.0 / (r * r) + SAME_U_C1 * tau * tau
        + 2.0 * SAME_U_C2 * r * tau.powi(3);
    -tau * e * poly + e * poly_prime
}

/// Derivative of `gamma_sub_exprn` (one half of the different-τ screening).
fn gamma_sub_exprn_prime(r: f64, tau1: f64, tau2: f64) -> f64 {
    assert!((tau1 - tau2).abs() >= TAU_FACTOR * MIN_HUB_DIFF,
        "gamma_sub_exprn_prime: degenerate tau ({tau1}, {tau2})");
    assert!(r >= TOL_SAME_DIST, "gamma_sub_exprn_prime: on-top atoms (r={r})");

    let dt2 = tau1 * tau1 - tau2 * tau2;
    let dt2_sq = dt2 * dt2;
    let dt2_cu = dt2_sq * dt2;

    let term_a = 0.5 * tau2.powi(4) * tau1 / dt2_sq;
    let term_b = (tau2.powi(6) - 3.0 * tau2.powi(4) * tau1 * tau1) / (r * dt2_cu);

    let e = (-tau1 * r).exp();
    // d/dR [ e * (term_a - term_b) ]
    //   = -tau1*e*(term_a - term_b) + e*(term_b / R)
    // because d(term_b)/dR = -term_b/R ... wait: term_b = C / R, so
    // d(term_b)/dR = -C / R^2 = -term_b / R.
    -tau1 * e * (term_a - term_b) + e * (term_b / r)
}

// ─── Repulsive spline (local parser; sk_data.rs is not owned by Agent_3) ──

/// Repulsive pair potential parsed from the `Spline` section of an old-format
/// `.skf` file. Matches `TSplineRep` in `src/dftbp/dftb/repulsive/splinerep.F90`.
#[derive(Debug, Clone)]
pub struct RepulsiveSpline {
    /// Start of each spline interval (length = n_spline).
    pub x_start: Vec<f64>,
    /// Cubic coefficients for intervals 1..n-1, stored as
    /// `sp_coeffs[interval][0..4]` = (c0, c1, c2, c3).
    pub sp_coeffs: Vec<[f64; 4]>,
    /// Six coefficients of the final polynomial tail.
    pub sp_last_coeffs: [f64; 6],
    /// Exponential-head coefficients (a, b, c): E = exp(-a*R + b) + c.
    pub exp_coeffs: [f64; 3],
    /// Cutoff distance (end of the last interval).
    pub cutoff: f64,
}

impl RepulsiveSpline {
    /// Evaluate the repulsive energy and its first derivative at distance `r`.
    ///
    /// Units: `r` is in Bohr (SK-file native unit), energy in Hartree,
    /// derivative in Hartree/Bohr. This matches the Fortran `TSplineRep_getValue`.
    pub fn eval(&self, r: f64) -> (f64, f64) {
        if r < MIN_NEIGH_DIST || r >= self.cutoff {
            return (0.0, 0.0);
        }

        if r < self.x_start[0] {
            return self.eval_exponential_head(r);
        }

        // Bisection: find interval i such that x_start[i] <= r < x_start[i+1].
        let n = self.x_start.len();
        // If r is at or beyond the last x_start, use the polynomial tail.
        if r >= self.x_start[n - 1] {
            let dr = r - self.x_start[n - 1];
            return Self::eval_poly_tail(&self.sp_last_coeffs, dr);
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x_start[mid] <= r {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let i_match = lo;
        let dr = r - self.x_start[i_match];

        if i_match < n - 1 {
            Self::eval_cubic(&self.sp_coeffs[i_match], dr)
        } else {
            Self::eval_poly_tail(&self.sp_last_coeffs, dr)
        }
    }

    fn eval_exponential_head(&self, r: f64) -> (f64, f64) {
        let (a, b, c) = (self.exp_coeffs[0], self.exp_coeffs[1], self.exp_coeffs[2]);
        let e = (-a * r + b).exp();
        (e + c, -a * e)
    }

    fn eval_cubic(coeffs: &[f64; 4], dr: f64) -> (f64, f64) {
        // E = c0 + c1*dr + c2*dr² + c3*dr³
        let e = coeffs[0] + coeffs[1] * dr + coeffs[2] * dr * dr + coeffs[3] * dr.powi(3);
        let de = coeffs[1] + 2.0 * coeffs[2] * dr + 3.0 * coeffs[3] * dr * dr;
        (e, de)
    }

    fn eval_poly_tail(coeffs: &[f64; 6], dr: f64) -> (f64, f64) {
        // E = c0 + c1*dr + c2*dr² + ... + c5*dr⁵
        let mut e = coeffs[0];
        let mut xh = dr;
        for k in 1..6 {
            e += coeffs[k] * xh;
            xh *= dr;
        }
        let mut de = 0.0;
        let mut xh = 1.0;
        for k in 1..6 {
            de += (k as f64) * coeffs[k] * xh;
            xh *= dr;
        }
        (e, de)
    }
}

/// Parse the `Spline` section of an old-format `.skf` file.
///
/// Layout (after the `Spline` keyword line):
/// ```text
/// nint  cutoff
/// expCoeffs(3)
/// xStart(1) xEnd(1) spCoeffs(1..4, 1)
/// ...
/// xStart(nint-1) xEnd(nint-1) spCoeffs(1..4, nint-1)
/// xStart(nint) xEnd(nint) spLastCoeffs(1..6)
/// ```
/// The final cutoff is taken from `xEnd(nint)`.
pub fn parse_repulsive_spline(sk_path: &str) -> Result<Option<RepulsiveSpline>> {
    let text = std::fs::read_to_string(sk_path)
        .map_err(|e| DftbError::Parse(format!("cannot read skf {sk_path}: {e}")))?;

    let mut lines = text.lines().peekable();

    // Skip the header lines that precede the Spline keyword.
    // The first line is the grid header (or an extended-format marker + grid).
    // We simply scan for a line whose first token is "Spline".
    let mut found = false;
    while let Some(line) = lines.next() {
        let tok = line.trim_start().split_whitespace().next().unwrap_or("");
        if tok == "Spline" {
            found = true;
            break;
        }
    }
    if !found {
        return Ok(None);
    }

    // nint  cutoff
    let header = lines.next()
        .ok_or_else(|| DftbError::Parse("Spline section: missing nint/cutoff line".into()))?;
    let hdr: Vec<f64> = parse_f64_line(header)?;
    if hdr.len() < 2 {
        return Err(DftbError::Parse("Spline header needs nint and cutoff".into()));
    }
    let nint = hdr[0] as usize;
    let _declared_cutoff = hdr[1];
    if nint == 0 {
        return Ok(None);
    }

    // expCoeffs(3)
    let exp_line = lines.next()
        .ok_or_else(|| DftbError::Parse("Spline section: missing expCoeffs line".into()))?;
    let exp_vals = parse_f64_line(exp_line)?;
    if exp_vals.len() < 3 {
        return Err(DftbError::Parse("expCoeffs line needs 3 values".into()));
    }
    let exp_coeffs = [exp_vals[0], exp_vals[1], exp_vals[2]];

    let mut x_start = Vec::with_capacity(nint);
    let mut sp_coeffs: Vec<[f64; 4]> = Vec::with_capacity(nint.saturating_sub(1));
    let mut x_end_last = 0.0f64;
    let mut sp_last_coeffs = [0.0f64; 6];

    // Cubic intervals 1..nint-1
    for _ in 0..(nint.saturating_sub(1)) {
        let line = lines.next()
            .ok_or_else(|| DftbError::Parse("Spline section: unexpected EOF in cubic intervals".into()))?;
        let vals = parse_f64_line(line)?;
        if vals.len() < 6 {
            return Err(DftbError::Parse(format!(
                "cubic interval line needs 6 values, got {}", vals.len()
            )));
        }
        x_start.push(vals[0]);
        sp_coeffs.push([vals[2], vals[3], vals[4], vals[5]]);
        x_end_last = vals[1];
    }

    // Final polynomial-tail interval
    {
        let line = lines.next()
            .ok_or_else(|| DftbError::Parse("Spline section: missing final polynomial interval".into()))?;
        let vals = parse_f64_line(line)?;
        if vals.len() < 8 {
            return Err(DftbError::Parse(format!(
                "polynomial tail line needs 8 values, got {}", vals.len()
            )));
        }
        x_start.push(vals[0]);
        x_end_last = vals[1];
        sp_last_coeffs = [vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]];
    }

    // Consistency: x_end(k-1) should match x_start(k).
    // We follow the Fortran convention and take the cutoff from the final x_end.
    let cutoff = x_end_last;

    Ok(Some(RepulsiveSpline {
        x_start,
        sp_coeffs,
        sp_last_coeffs,
        exp_coeffs,
        cutoff,
    }))
}

fn parse_f64_line(line: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for tok in line.split_whitespace() {
        let t = tok.replace('D', "E").replace('d', "e");
        match t.parse::<f64>() {
            Ok(v) => out.push(v),
            Err(e) => return Err(DftbError::Parse(format!("bad float '{tok}': {e}"))),
        }
    }
    Ok(out)
}

// ─── Force result ──────────────────────────────────────────────────────────

/// Forces on each atom, in Hartree/Ångström. Shape: `[n_atoms][3]`.
#[derive(Debug, Clone)]
pub struct Forces {
    pub forces: Vec<[f64; 3]>,
    /// Optional breakdown of components (only populated by `compute_scc_forces`).
    pub non_scc: Vec<[f64; 3]>,
    pub scc_shift: Vec<[f64; 3]>,
    pub scc_dc: Vec<[f64; 3]>,
    pub repulsive: Vec<[f64; 3]>,
}

impl Forces {
    pub fn zeros(n: usize) -> Self {
        Forces {
            forces: vec![[0.0; 3]; n],
            non_scc: vec![[0.0; 3]; n],
            scc_shift: vec![[0.0; 3]; n],
            scc_dc: vec![[0.0; 3]; n],
            repulsive: vec![[0.0; 3]; n],
        }
    }

    /// Total force on atom `i`, component `c`.
    pub fn total(&self, i: usize, c: usize) -> f64 {
        self.forces[i][c]
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Diagonalize the non-SCC generalized eigenproblem H0·c = E·S·c and return
/// occupied eigenvectors / eigenvalues. Mirrors `Fragment::diagonalize` but
/// operates on a plain `Hamiltonian` instead of a `Fragment`.
fn diagonalize_non_scc(ham: &Hamiltonian, n_electrons: f64) -> Result<(DMatrix<f64>, DVector<f64>)> {
    let n = ham.h0.nrows();
    assert_eq!(ham.h0.ncols(), n);
    assert_eq!(ham.s.nrows(), n);

    let cholesky = Cholesky::new(ham.s.clone())
        .ok_or_else(|| DftbError::InvalidInput("Overlap not positive definite".into()))?;
    let l = cholesky.l();

    // H' = L^-1 H L^-T
    let m = l.solve_lower_triangular(&ham.h0)
        .ok_or_else(|| DftbError::InvalidInput("L·M = H solve failed".into()))?;
    let n_mat = l.solve_lower_triangular(&m.transpose())
        .ok_or_else(|| DftbError::InvalidInput("L·N = Mᵀ solve failed".into()))?;
    let h_prime = n_mat.transpose();

    let se = SymmetricEigen::new(h_prime);
    let eigs = se.eigenvalues;
    let c_prime = se.eigenvectors;

    // Sort ascending.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eigs[a].partial_cmp(&eigs[b]).unwrap());
    let sorted_eigs: Vec<f64> = idx.iter().map(|&i| eigs[i]).collect();
    let sorted_c_prime = c_prime.select_columns(&idx);

    // Back-transform c = L^-T c'
    let c = l.tr_solve_lower_triangular(&sorted_c_prime)
        .ok_or_else(|| DftbError::InvalidInput("Lᵀ·c = c' solve failed".into()))?;

    let _ = n_electrons; // n_occ computed by caller
    Ok((c, DVector::from(sorted_eigs)))
}

/// Build the density matrix D = 2 * C_occ * C_occ^T and the energy-weighted
/// density matrix EDM = 2 * C_occ * diag(eps_occ) * C_occ^T.
fn build_density_matrices(
    c_occ: &DMatrix<f64>,
    eps_occ: &[f64],
) -> (DMatrix<f64>, DMatrix<f64>) {
    // D = 2 * C_occ * C_occ^T
    let density = c_occ * c_occ.transpose() * 2.0;

    // EDM = 2 * C_occ * diag(eps) * C_occ^T
    let n = c_occ.nrows();
    let n_occ = c_occ.ncols();
    let mut edm = DMatrix::<f64>::zeros(n, n);
    for k in 0..n_occ {
        let col = c_occ.column(k);
        let scaled = col * (2.0 * eps_occ[k]);
        // edm += scaled * col^T
        edm += &scaled * col.transpose();
    }
    (density, edm)
}

/// Build a pair H/S block for atoms (i, j) using the SK tables and a
/// caller-provided displacement of atom j relative to its base position.
///
/// `coords_j_disp` is the displaced coordinate of atom j (in Å).
/// Returns flat buffers `out_h`, `out_s` of length `n_orb_i * n_orb_j`,
/// row-major (rows = orbitals of j, cols = orbitals of i) — matching the
/// convention used by `Rotation::rotate_diatomic_block_into`.
///
/// **P2:** production use is the sparse BSR4 assembler (F2,
/// `assemble_hs_bsr` in `methods/sparse/sparse_dftb.rs`); dense force code
/// uses `build_pair_block_with_derivs` (analytic derivatives) instead.
pub(crate) fn build_pair_block(
    ctx: &SystemContext<'_>,
    coords_i: [f64; 3],
    coords_j_disp: [f64; 3],
    i: usize,
    j: usize,
    out_h: &mut [f64],
    out_s: &mut [f64],
) -> Result<()> {
    let si = ctx.atom_species[i];
    let sj = ctx.atom_species[j];

    let tab_fwd = ctx.pair_table(si, sj).ok_or_else(|| {
        DftbError::InvalidInput(format!("missing SK table fwd ({si},{sj})"))
    })?;
    let tab_rev = ctx.pair_table(sj, si).ok_or_else(|| {
        DftbError::InvalidInput(format!("missing SK table rev ({sj},{si})"))
    })?;

    let v = [
        coords_j_disp[0] - coords_i[0],
        coords_j_disp[1] - coords_i[1],
        coords_j_disp[2] - coords_i[2],
    ];
    let r_bohr = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() * ANG2BOHR;
    let dc = DirectionCosines::from_vec(v)?;

    Rotation::rotate_diatomic_block_into(
        tab_fwd,
        tab_rev,
        ctx.species_ang[si as usize],
        ctx.species_ang[sj as usize],
        r_bohr,
        dc,
        out_h,
        out_s,
    )
}

/// Build a pair H/S block AND its analytic Cartesian derivatives for atoms
/// (i, j) using the SK tables. This is the **production analytic force path**
/// (P2, manifest v3 §4.3): replaces the nested finite-difference
/// `pair_block_derivative` with analytic SK radial + angular derivatives.
///
/// Outputs (all row-major, size `n_orb_j * n_orb_i`):
///   out_h, out_s: H and S blocks
///   dh_dx, dh_dy, dh_dz: dH/dR_a for a=x,y,z (derivative w.r.t. atom j)
///   ds_dx, ds_dy, ds_z: dS/dR_a for a=x,y,z
///
/// Derivatives are w.r.t. atom j's position in Å, and the SK radial
/// derivatives are in Hartree/Bohr, so the Cartesian derivatives are in
/// Hartree/Bohr (matching the SK convention). The force functions convert
/// to Hartree/Å using ANG2BOHR.
pub(crate) fn build_pair_block_with_derivs(
    ctx: &SystemContext<'_>,
    coords: &[[f64; 3]],
    i: usize,
    j: usize,
    out_h: &mut [f64], out_s: &mut [f64],
    dh_dx: &mut [f64], dh_dy: &mut [f64], dh_dz: &mut [f64],
    ds_dx: &mut [f64], ds_dy: &mut [f64], ds_dz: &mut [f64],
) -> Result<()> {
    let si = ctx.atom_species[i];
    let sj = ctx.atom_species[j];

    let tab_fwd = ctx.pair_table(si, sj).ok_or_else(|| {
        DftbError::InvalidInput(format!("missing SK table fwd ({si},{sj})"))
    })?;
    let tab_rev = ctx.pair_table(sj, si).ok_or_else(|| {
        DftbError::InvalidInput(format!("missing SK table rev ({sj},{si})"))
    })?;

    let v = [
        coords[j][0] - coords[i][0],
        coords[j][1] - coords[i][1],
        coords[j][2] - coords[i][2],
    ];
    let r_bohr = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() * ANG2BOHR;
    let dc = DirectionCosines::from_vec(v)?;

    Rotation::rotate_block_with_derivs_into(
        tab_fwd,
        tab_rev,
        ctx.species_ang[si as usize],
        ctx.species_ang[sj as usize],
        r_bohr,
        dc,
        out_h, out_s,
        dh_dx, dh_dy, dh_dz,
        ds_dx, ds_dy, ds_dz,
    )
}

/// Compute the central finite-difference derivative of the pair H/S block
/// with respect to Cartesian direction `dir` (0=x, 1=y, 2=z) of atom j.
///
/// Matches Fortran `getFirstDerivFiniteDiff`: displaces atom j by ±delta
/// along `dir`, rebuilds the block, and returns `(dH, dS)` as flat buffers.
///
/// **P2 firewall (manifest v3 §4.3):** This is a **test/reference helper
/// only**. Production sparse/dense force code must use
/// `build_pair_block_with_derivs` (analytic SK derivatives) instead.
/// The only production finite difference is force → nuclear Hessian.
#[cfg(test)]
fn pair_block_derivative(
    ctx: &SystemContext<'_>,
    coords: &[[f64; 3]],
    i: usize,
    j: usize,
    dir: usize,
    delta: f64,
    block_size: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut h_plus = vec![0.0f64; block_size];
    let mut s_plus = vec![0.0f64; block_size];
    let mut h_minus = vec![0.0f64; block_size];
    let mut s_minus = vec![0.0f64; block_size];

    let ci = coords[i];
    let mut cj_plus = coords[j];
    let mut cj_minus = coords[j];
    cj_plus[dir] += delta;
    cj_minus[dir] -= delta;

    build_pair_block(ctx, ci, cj_plus, i, j, &mut h_plus, &mut s_plus)?;
    build_pair_block(ctx, ci, cj_minus, i, j, &mut h_minus, &mut s_minus)?;

    let mut dh = vec![0.0f64; block_size];
    let mut ds = vec![0.0f64; block_size];
    let inv = 1.0 / (2.0 * delta);
    for k in 0..block_size {
        dh[k] = (h_plus[k] - h_minus[k]) * inv;
        ds[k] = (s_plus[k] - s_minus[k]) * inv;
    }
    Ok((dh, ds))
}

/// Extract the pair block of the density / EDM from the global matrices.
/// Returns `sqr_dm[n_orb_j][n_orb_i]` and `sqr_edm[n_orb_j][n_orb_i]` in
/// the same row-major layout as the SK block derivatives.
fn extract_pair_dm_edm(
    dm: &DMatrix<f64>,
    edm: &DMatrix<f64>,
    ctx: &SystemContext<'_>,
    i: usize,
    j: usize,
) -> (Vec<f64>, Vec<f64>) {
    let ni = ctx.atom_n_orb[i] as usize;
    let nj = ctx.atom_n_orb[j] as usize;
    let bi = ctx.atom_orb_off[i] as usize;
    let bj = ctx.atom_orb_off[j] as usize;

    let mut sqr_dm = vec![0.0f64; ni * nj];
    let mut sqr_edm = vec![0.0f64; ni * nj];
    // Fortran layout: rows = orbitals of j, cols = orbitals of i.
    for a in 0..nj {
        for b in 0..ni {
            sqr_dm[a * ni + b] = dm[(bi + b, bj + a)];
            sqr_edm[a * ni + b] = edm[(bi + b, bj + a)];
        }
    }
    (sqr_dm, sqr_edm)
}

/// Compute the non-SCC electronic force contribution.
///
/// For each unique atom pair (i, j) with i < j:
///   F_i += 2 * ( Σ DM·dH - Σ EDM·dS )   (sum over the pair block)
///   F_j -= same
///
/// The factor of 2 accounts for the implicit lower-triangle summation,
/// matching `derivativeNonSccEuclidian` in `forces.F90`.
///
/// **P2 (manifest v3 §4.3):** Uses analytic SK radial + angular derivatives
/// via `build_pair_block_with_derivs`. No finite differences of H/S blocks.
/// The only production finite difference is force → nuclear Hessian.
pub fn non_scc_electronic_force(
    ctx: &SystemContext<'_>,
    neigh: &NeighborList,
    coords: &[[f64; 3]],
    dm: &DMatrix<f64>,
    edm: &DMatrix<f64>,
    forces: &mut [[f64; 3]],
) -> Result<()> {
    let max_n = ctx.species_n_orb.iter().copied().map(|n| n as usize).max().unwrap_or(0);
    let max_block = max_n * max_n;

    for p in &neigh.pairs {
        let i = p.i;
        let j = p.j;
        let ni = ctx.atom_n_orb[i] as usize;
        let nj = ctx.atom_n_orb[j] as usize;
        let block_size = ni * nj;

        let (sqr_dm, sqr_edm) = extract_pair_dm_edm(dm, edm, ctx, i, j);

        // P2: analytic derivatives — one SK evaluation, all 3 directions at once.
        let mut h = vec![0.0f64; max_block];
        let mut s = vec![0.0f64; max_block];
        let mut dh_dx = vec![0.0f64; max_block];
        let mut dh_dy = vec![0.0f64; max_block];
        let mut dh_dz = vec![0.0f64; max_block];
        let mut ds_dx = vec![0.0f64; max_block];
        let mut ds_dy = vec![0.0f64; max_block];
        let mut ds_dz = vec![0.0f64; max_block];

        build_pair_block_with_derivs(
            ctx, coords, i, j,
            &mut h, &mut s,
            &mut dh_dx, &mut dh_dy, &mut dh_dz,
            &mut ds_dx, &mut ds_dy, &mut ds_dz,
        )?;

        // Force contribution: F_i += 2 * Σ (DM·dH - EDM·dS) for each direction.
        // dH/dR_a is in Hartree/Bohr (SK convention), R is in Å, so the
        // derivative w.r.t. R_Å is dH/dR_bohr * dR_bohr/dR_Å = dH/dR_bohr * ANG2BOHR.
        let mut contr_x = 0.0f64;
        let mut contr_y = 0.0f64;
        let mut contr_z = 0.0f64;
        for k in 0..block_size {
            contr_x += sqr_dm[k] * dh_dx[k] - sqr_edm[k] * ds_dx[k];
            contr_y += sqr_dm[k] * dh_dy[k] - sqr_edm[k] * ds_dy[k];
            contr_z += sqr_dm[k] * dh_dz[k] - sqr_edm[k] * ds_dz[k];
        }
        // Factor of 2 for lower-triangle summation.
        // Convert from Hartree/Bohr to Hartree/Å (the force unit convention).
        let f = 2.0 * ANG2BOHR;
        forces[i][0] += f * contr_x;
        forces[i][1] += f * contr_y;
        forces[i][2] += f * contr_z;
        forces[j][0] -= f * contr_x;
        forces[j][1] -= f * contr_y;
        forces[j][2] -= f * contr_z;
    }
    Ok(())
}

/// Compute the SCC shift force contribution.
///
/// Exact block-resolved shift matrices are not currently exposed by
/// `SccResult` (only atom-resolved shifts are available). We therefore
/// reconstruct the per-atom block shift `S_atom[μ,ν] = shift_atom * I_atom`
/// and use the Fortran formula:
///
/// ```text
/// shiftSprime = 0.5 * ( S'·shift_i_block + shift_j_block·S' )
/// F += 2 * Σ( shiftSprime · DM_block )
/// ```
///
/// where `shift_atom_block = shift_atom * I_{n_orb_atom}`. This is exact
/// when the SCC potential is purely atom-resolved (which is the case for
/// the standard DFTB SCC model used here — there are no orbital-resolved
/// contributions in mio-1-1).
///
/// **P2 (manifest v3 §4.3):** Uses analytic dS/dR via
/// `build_pair_block_with_derivs`. No finite differences.
pub fn scc_shift_force(
    ctx: &SystemContext<'_>,
    neigh: &NeighborList,
    coords: &[[f64; 3]],
    dm: &DMatrix<f64>,
    shifts: &[f64],
    forces: &mut [[f64; 3]],
) -> Result<()> {
    let max_n = ctx.species_n_orb.iter().copied().map(|n| n as usize).max().unwrap_or(0);
    let max_block = max_n * max_n;

    for p in &neigh.pairs {
        let i = p.i;
        let j = p.j;
        let ni = ctx.atom_n_orb[i] as usize;
        let nj = ctx.atom_n_orb[j] as usize;
        let block_size = ni * nj;

        let shift_i = shifts[i];
        let shift_j = shifts[j];

        let (sqr_dm, _sqr_edm) = extract_pair_dm_edm(dm, dm, ctx, i, j);

        // P2: analytic dS/dR — one SK evaluation, all 3 directions at once.
        let mut h = vec![0.0f64; max_block];
        let mut s = vec![0.0f64; max_block];
        let mut dh_dx = vec![0.0f64; max_block];
        let mut dh_dy = vec![0.0f64; max_block];
        let mut dh_dz = vec![0.0f64; max_block];
        let mut ds_dx = vec![0.0f64; max_block];
        let mut ds_dy = vec![0.0f64; max_block];
        let mut ds_dz = vec![0.0f64; max_block];

        build_pair_block_with_derivs(
            ctx, coords, i, j,
            &mut h, &mut s,
            &mut dh_dx, &mut dh_dy, &mut dh_dz,
            &mut ds_dx, &mut ds_dy, &mut ds_dz,
        )?;

        // shiftSprime[a,b] = 0.5 * ( shift_i * S'[a,b] + shift_j * S'[a,b] )
        //                  = 0.5 * (shift_i + shift_j) * S'[a,b]
        // F = 2 * Σ shiftSprime · DM_block
        // dS/dR is in Hartree/Bohr; convert to Hartree/Å with ANG2BOHR.
        let avg_shift = 0.5 * (shift_i + shift_j);
        let mut contr_x = 0.0f64;
        let mut contr_y = 0.0f64;
        let mut contr_z = 0.0f64;
        for k in 0..block_size {
            contr_x += avg_shift * ds_dx[k] * sqr_dm[k];
            contr_y += avg_shift * ds_dy[k] * sqr_dm[k];
            contr_z += avg_shift * ds_dz[k] * sqr_dm[k];
        }
        let f = 2.0 * ANG2BOHR;
        forces[i][0] += f * contr_x;
        forces[i][1] += f * contr_y;
        forces[i][2] += f * contr_z;
        forces[j][0] -= f * contr_x;
        forces[j][1] -= f * contr_y;
        forces[j][2] -= f * contr_z;
    }
    Ok(())
}

/// Compute the SCC double-counting (Coulomb) force contribution.
///
/// For each unique atom pair (i, j) with i < j:
///   F_i += -deltaQ_i * deltaQ_j * gamma'(r) * r_hat
///   F_j -= same
///
/// where `gamma'` is the full derivative (short-range + Coulomb 1/R).
/// Distances are in Bohr for gamma; the force direction uses the Å vector.
///
/// Sign convention (Rust): `delta_q = q_elec - q0`. DFTB+ uses
/// `deltaQ = q0 - q_elec`, so the Fortran `addInvRPrimeCluster` coefficient
/// `-deltaQ_i*deltaQ_j / r^3` becomes `+deltaQ_rust_i*deltaQ_rust_j / r^3`
/// for the Coulomb part. Combined with the short-range derivative, the
/// net force on atom i is:
///   F_i = -deltaQ_i * deltaQ_j * gamma_full'(r_bohr) * r_hat_ang
pub fn scc_double_counting_force(
    coords: &[[f64; 3]],
    species: &[u8],
    delta_q: &[f64],
    gamma_tbl: &GammaTable,
    forces: &mut [[f64; 3]],
) {
    let n = coords.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2_ang = dx * dx + dy * dy + dz * dz;
            if r2_ang < MIN_NEIGH_DIST * MIN_NEIGH_DIST {
                continue;
            }
            let r_ang = r2_ang.sqrt();
            let r_bohr = r_ang * ANG2BOHR;

            let u_i = gamma_tbl.u(species[i]);
            let u_j = gamma_tbl.u(species[j]);

            // The SCC double-counting force from pair (i,j) on atom i is:
            //   F_i = -dE_scc/dx_i = -dq_i * dq_j * gamma_full'(r) * (x_i - x_j) / r
            //
            // (The 0.5 in the energy and the double-sum over (A,B) and (B,A) cancel,
            //  giving an effective factor of 1 per unique pair.)
            //
            // DFTB+ splits this into a short-range gamma derivative (addGradientsDc)
            // and a Coulomb 1/R derivative (addInvRPrimeCluster), but the combined
            // effect is the full gamma_full' derivative with factor 1.
            //
            // With deltaQ_DFTB+ = -dq_Rust, the product is the same.
            // The force uses gamma_full' which is negative (attractive/repulsive
            // depending on charge signs).
            //
            // Units: gamma_full' is in 1/Bohr, r is in Bohr, (x_i - x_j) is in Bohr.
            // The force is in Hartree/Bohr. We convert to Hartree/Å by multiplying
            // by ANG2BOHR (1 Hartree/Bohr = ANG2BOHR Hartree/Å).

            let gprime_full = gamma_prime_full(r_bohr, u_i, u_j);

            // Force in Hartree/Bohr:
            //   F_bohr = -dq_i * dq_j * gprime / r_bohr * (coord_i - coord_j)_bohr
            // Convert to Hartree/Å:
            //   F_ang = F_bohr * ANG2BOHR
            //         = -dq_i * dq_j * gprime / r_bohr * (coord_i - coord_j)_ang * ANG2BOHR * ANG2BOHR
            let coeff = -delta_q[i] * delta_q[j] * gprime_full / r_bohr
                * ANG2BOHR * ANG2BOHR;

            let fx = coeff * dx;
            let fy = coeff * dy;
            let fz = coeff * dz;
            forces[i][0] += fx;
            forces[i][1] += fy;
            forces[i][2] += fz;
            forces[j][0] -= fx;
            forces[j][1] -= fy;
            forces[j][2] -= fz;
        }
    }
}

/// Compute the repulsive pair-potential force contribution.
///
/// For each unique atom pair (i, j) with i < j:
///   F_i += dE_rep/dr * r_hat(i->j)
///   F_j -= same
///
/// `dE_rep/dr` is in Hartree/Bohr (SK-file native unit). The direction
/// `r_hat(i->j) = (coord_j - coord_i) / r_ang` is dimensionless, so the
/// resulting force is in Hartree/Bohr. We convert to Hartree/Ångström at
/// the end by multiplying by `BOHR2ANG`... actually, to stay consistent
/// with the electronic forces (which are in Hartree/Å because the
/// finite-difference step is in Å), we convert the repulsive derivative
/// to Hartree/Å here: `dE/dr_Å = dE/dr_bohr * ANG2BOHR`.
fn repulsive_force(
    coords: &[[f64; 3]],
    species_names: &[String],
    sk_dir: &str,
    forces: &mut [[f64; 3]],
) -> Result<()> {
    let n = coords.len();
    // Cache parsed splines per species pair to avoid re-reading files.
    use std::collections::HashMap;
    let mut cache: HashMap<(String, String), Option<RepulsiveSpline>> = HashMap::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let key = (species_names[i].clone(), species_names[j].clone());
            let spline = if let Some(s) = cache.get(&key) {
                s.clone()
            } else {
                // Try both orderings of the filename.
                let p1 = format!("{}/{}-{}.skf", sk_dir, key.0, key.1);
                let p2 = format!("{}/{}-{}.skf", sk_dir, key.1, key.0);
                let s = if std::path::Path::new(&p1).exists() {
                    parse_repulsive_spline(&p1)?
                } else if std::path::Path::new(&p2).exists() {
                    parse_repulsive_spline(&p2)?
                } else {
                    None
                };
                cache.insert(key.clone(), s.clone());
                s
            };

            let Some(spline) = spline else {
                // No spline section in the SK file: zero repulsive contribution.
                // This is the documented behaviour for pairs without a repulsive
                // potential (e.g. some heteronuclear pairs in mio-1-1).
                continue;
            };

            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < MIN_NEIGH_DIST * MIN_NEIGH_DIST {
                continue;
            }
            let r_ang = r2.sqrt();
            let r_bohr = r_ang * ANG2BOHR;

            let (_e, de_bohr) = spline.eval(r_bohr);
            if de_bohr == 0.0 {
                continue;
            }
            // Convert derivative to Hartree/Å and apply along r_hat(i->j).
            let de_ang = de_bohr * ANG2BOHR;
            let inv_r = 1.0 / r_ang;
            let fx = de_ang * dx * inv_r;
            let fy = de_ang * dy * inv_r;
            let fz = de_ang * dz * inv_r;
            forces[i][0] += fx;
            forces[i][1] += fy;
            forces[i][2] += fz;
            forces[j][0] -= fx;
            forces[j][1] -= fy;
            forces[j][2] -= fz;
        }
    }
    Ok(())
}

/// Repulsive force using pre-parsed spline tables indexed by pair type.
/// No Strings, HashMaps, file I/O, or clones in the hot path.
/// `repulsive[pair_type]` where `pair_type = species_i * n_species + species_j`.
pub fn repulsive_force_cached(
    coords: &[[f64; 3]],
    ctx: &super::hamiltonian::SystemContext<'_>,
    repulsive: &[Option<RepulsiveSpline>],
    forces: &mut [[f64; 3]],
) -> Result<()> {
    let n = coords.len();
    let n_species = ctx.n_species;
    for i in 0..n {
        let si = ctx.atom_species[i] as usize;
        for j in (i + 1)..n {
            let sj = ctx.atom_species[j] as usize;
            let pair_type = si * n_species + sj;
            let pair_type_rev = sj * n_species + si;
            let spline = if let Some(s) = &repulsive[pair_type] {
                s
            } else if let Some(s) = &repulsive[pair_type_rev] {
                s
            } else {
                continue; // no repulsive for this pair
            };

            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < MIN_NEIGH_DIST * MIN_NEIGH_DIST { continue; }
            let r_ang = r2.sqrt();
            let r_bohr = r_ang * ANG2BOHR;
            let (_e, de_bohr) = spline.eval(r_bohr);
            if de_bohr == 0.0 { continue; }
            let de_ang = de_bohr * ANG2BOHR;
            let inv_r = 1.0 / r_ang;
            let fx = de_ang * dx * inv_r;
            let fy = de_ang * dy * inv_r;
            let fz = de_ang * dz * inv_r;
            forces[i][0] += fx; forces[i][1] += fy; forces[i][2] += fz;
            forces[j][0] -= fx; forces[j][1] -= fy; forces[j][2] -= fz;
        }
    }
    Ok(())
}

/// Parse all repulsive splines from SK files into a flat array indexed by pair type.
/// Call once at initialization, not in the hot path.
pub fn parse_all_repulsive(
    sk_dir: &str,
    species_names: &[String],
    n_species: usize,
) -> Result<Vec<Option<RepulsiveSpline>>> {
    let mut out = vec![None; n_species * n_species];
    for i in 0..n_species {
        for j in 0..n_species {
            let p1 = format!("{}/{}-{}.skf", sk_dir, species_names[i], species_names[j]);
            let p2 = format!("{}/{}-{}.skf", sk_dir, species_names[j], species_names[i]);
            let s = if std::path::Path::new(&p1).exists() {
                parse_repulsive_spline(&p1)?
            } else if std::path::Path::new(&p2).exists() {
                parse_repulsive_spline(&p2)?
            } else {
                None
            };
            out[i * n_species + j] = s;
        }
    }
    Ok(out)
}

/// Pack repulsive splines into the GPU kernel record layout
/// (`gpu_matrix_ops.cl::repulsive_energy_batched`, `gpu_forces.cl::force_repulsive_batched`).
///
/// `tables` is `n_species × n_species` in the same order as GPU `atom_species`.
/// Fails loud if any species pair has no Spline (silent skip would drop E_rep / F_rep).
pub fn pack_repulsive_gpu(tables: &[Option<RepulsiveSpline>], n_species: usize, species_names: &[String]) -> Result<(Vec<i32>, Vec<f32>, usize)> {
    if tables.len() != n_species * n_species {
        return Err(DftbError::InvalidInput(format!(
            "pack_repulsive_gpu: tables.len()={} != n_species² {n_species}²", tables.len()
        )));
    }
    if species_names.len() != n_species {
        return Err(DftbError::InvalidInput(format!(
            "pack_repulsive_gpu: species_names.len()={} != n_species={n_species}", species_names.len()
        )));
    }
    let mut max_int = 1usize;
    for (p, s) in tables.iter().enumerate() {
        match s {
            Some(sp) => { max_int = max_int.max(sp.x_start.len()); }
            None => {
                let i = p / n_species; let j = p % n_species;
                if tables[j * n_species + i].is_none() {
                    return Err(DftbError::InvalidInput(format!(
                        "pack_repulsive_gpu: no Spline for {}-{}", species_names[i], species_names[j]
                    )));
                }
            }
        }
    }
    let rec = 5 + max_int + (max_int.saturating_sub(1)) * 4 + 6;
    let mut offsets = vec![-1i32; n_species * n_species];
    let mut data = Vec::new();
    for (p, s) in tables.iter().enumerate() {
        let Some(sp) = s else { continue };
        offsets[p] = data.len() as i32;
        let n_int = sp.x_start.len();
        data.push(f32::from_bits(n_int as u32));
        data.push(sp.cutoff as f32);
        data.push(sp.exp_coeffs[0] as f32);
        data.push(sp.exp_coeffs[1] as f32);
        data.push(sp.exp_coeffs[2] as f32);
        for k in 0..max_int { data.push(if k < n_int { sp.x_start[k] as f32 } else { 0.0 }); }
        let n_cubic = max_int.saturating_sub(1);
        for k in 0..n_cubic {
            if k < sp.sp_coeffs.len() {
                for c in 0..4 { data.push(sp.sp_coeffs[k][c] as f32); }
            } else {
                for _ in 0..4 { data.push(0.0); }
            }
        }
        for c in 0..6 { data.push(sp.sp_last_coeffs[c] as f32); }
        if data.len() - offsets[p] as usize != rec {
            return Err(DftbError::InvalidInput(format!(
                "pack_repulsive_gpu: record length {} != {rec} at pair {p}", data.len() - offsets[p] as usize
            )));
        }
    }
    Ok((offsets, data, max_int))
}

/// DFTB repulsive pair energy (Hartree) for a geometry in Å.
///
/// Fail-loud: a pair that appears in the molecule must have a Spline section
/// in the SK file. Silent `continue` on a missing spline would drop E_rep
/// (Gate F collapsed SiH4 to 0.93 Å for exactly that class of omission).
pub fn repulsive_energy(
    sk_dir: &str,
    species: &[String],
    coords: &[[f64; 3]],
) -> Result<f64> {
    assert_eq!(species.len(), coords.len(), "repulsive_energy: species/coords length mismatch");
    let mut names: Vec<String> = Vec::new();
    for s in species {
        if !names.iter().any(|n| n == s) {
            names.push(s.clone());
        }
    }
    let n_species = names.len();
    let tables = parse_all_repulsive(sk_dir, &names, n_species)?;
    let atom_sp: Vec<usize> = species.iter().map(|s| {
        names.iter().position(|n| n == s).unwrap_or_else(|| panic!("species {s} missing from unique list {names:?}"))
    }).collect();
    let n = coords.len();
    let mut e_rep = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let si = atom_sp[i];
            let sj = atom_sp[j];
            let spline = tables[si * n_species + sj].as_ref()
                .or(tables[sj * n_species + si].as_ref())
                .ok_or_else(|| DftbError::InvalidInput(format!(
                    "no repulsive Spline for {}-{} in {sk_dir}", species[i], species[j]
                )))?;
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < MIN_NEIGH_DIST * MIN_NEIGH_DIST {
                return Err(DftbError::InvalidInput(format!(
                    "repulsive_energy: atoms {i}-{} on top of each other, |r|={:.3e} Å", j, r2.sqrt()
                )));
            }
            let (e, _) = spline.eval(r2.sqrt() * ANG2BOHR);
            if !e.is_finite() {
                panic!("repulsive_energy: non-finite E_rep for {}-{} at r={:.4} Å: {e}", species[i], species[j], r2.sqrt());
            }
            e_rep += e;
        }
    }
    Ok(e_rep)
}

/// Repulsive pair energy using pre-parsed spline tables (no file I/O).
///
/// Same physics as `repulsive_energy` but reads `repulsive` from
/// `parse_all_repulsive` (parsed once at init). `species_code` indexes the
/// `species_names` order used at parse time. Fails loud on a missing spline.
pub fn repulsive_energy_cached(
    coords: &[[f64; 3]],
    species_code: &[u8],
    species_names: &[String],
    repulsive: &[Option<RepulsiveSpline>],
    n_species: usize,
) -> Result<f64> {
    let n = coords.len();
    assert_eq!(species_code.len(), n, "repulsive_energy_cached: len mismatch");
    let mut e_rep = 0.0f64;
    for i in 0..n {
        let si = species_code[i] as usize;
        for j in (i + 1)..n {
            let sj = species_code[j] as usize;
            let spline = repulsive[si * n_species + sj].as_ref()
                .or(repulsive[sj * n_species + si].as_ref())
                .ok_or_else(|| DftbError::InvalidInput(format!(
                    "repulsive_energy_cached: no repulsive Spline for {}-{}",
                    species_names[si], species_names[sj]
                )))?;
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < MIN_NEIGH_DIST * MIN_NEIGH_DIST {
                return Err(DftbError::InvalidInput(format!(
                    "repulsive_energy_cached: atoms {i}-{j} on top of each other, |r|={:.3e} Å", r2.sqrt()
                )));
            }
            let (e, _) = spline.eval(r2.sqrt() * ANG2BOHR);
            if !e.is_finite() {
                panic!("repulsive_energy_cached: non-finite E_rep for {}-{} at r={:.4} Å: {e}", species_names[si], species_names[sj], r2.sqrt());
            }
            e_rep += e;
        }
    }
    Ok(e_rep)
}

/// Sanity check: assert no NaN/Inf in the force array (fail-loud).
pub fn check_finite(forces: &[[f64; 3]], label: &str) {
    for (i, f) in forces.iter().enumerate() {
        for c in 0..3 {
            if !f[c].is_finite() {
                panic!("Non-finite force in {label}: atom {i} component {c} = {}", f[c]);
            }
        }
    }
}

/// Sanity check: Newton's third law (total force ≈ 0).
pub fn check_newton(forces: &[[f64; 3]], label: &str, tol: f64) {
    let mut sum = [0.0f64; 3];
    for f in forces {
        sum[0] += f[0];
        sum[1] += f[1];
        sum[2] += f[2];
    }
    let max = sum.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if max > tol {
        eprintln!("[WARNING] {label}: Newton's third law violated, |ΣF| = {max:.3e} (sum = {sum:?})");
    }
}

// ─── Public API ────────────────────────────────────────────────────────────

/// Compute non-SCC DFTB forces (electronic + repulsive) for a molecule.
///
/// # Arguments
/// * `builder` – Hamiltonian builder owning the SK data.
/// * `species` – Atom species labels (capitalized, e.g. "H", "O").
/// * `coords` – Atom coordinates in Ångström.
/// * `n_electrons` – Total valence electron count.
///
/// # Returns
/// Forces in Hartree/Ångström, with the `non_scc` and `repulsive`
/// components populated. `scc_shift` and `scc_dc` are zero.
pub fn compute_non_scc_forces(
    builder: &HamiltonianBuilder,
    species: &[String],
    coords: &[[f64; 3]],
    n_electrons: f64,
) -> Result<Forces> {
    let n_atoms = species.len();
    let mut out = Forces::zeros(n_atoms);

    let ham = builder.build_non_scc(species, coords)?;
    let ctx = SystemContext::from_sk_data(&builder.sk, species)?;

    // Cutoff for the neighbour list: use the max SK table cutoff.
    let cutoff = builder.sk.pairs.values()
        .map(|t| t.cutoff())
        .fold(0.0_f64, f64::max);
    let neigh = NeighborBuilder { cutoff }.build(coords)?;

    // Diagonalize and build density matrices.
    let (c, eigs) = diagonalize_non_scc(&ham, n_electrons)?;
    let n_occ = (n_electrons / 2.0).round() as usize;
    let c_occ = c.columns(0, n_occ).into_owned();
    let eps_occ: Vec<f64> = eigs.iter().take(n_occ).copied().collect();
    let (dm, edm) = build_density_matrices(&c_occ, &eps_occ);

    // Electronic force.
    non_scc_electronic_force(&ctx, &neigh, coords, &dm, &edm, &mut out.non_scc)?;

    // Repulsive force.
    // Derive the SK directory from the builder's SK data. Since SkData does
    // not store the source path, the caller must provide it via env var or
    // we fall back to a heuristic. For the parity tests we read the env var
    // RUST_DFTB_SK_DIR.
    let sk_dir = std::env::var("RUST_DFTB_SK_DIR")
        .map_err(|_| DftbError::InvalidInput(
            "RUST_DFTB_SK_DIR must be set for repulsive forces".into()
        ))?;
    repulsive_force(coords, species, &sk_dir, &mut out.repulsive)?;

    // Sum components.
    for i in 0..n_atoms {
        for c in 0..3 {
            out.forces[i][c] = out.non_scc[i][c] + out.repulsive[i][c];
        }
    }

    check_finite(&out.forces, "non-SCC total");
    check_newton(&out.forces, "non-SCC total", 1e-8);

    Ok(out)
}

/// Compute SCC DFTB forces (electronic + SCC shift + SCC double-counting +
/// repulsive) for a molecule from a converged `SccResult`.
///
/// # Arguments
/// * `builder` – Hamiltonian builder owning the SK data.
/// * `species` – Atom species labels.
/// * `coords` – Atom coordinates in Ångström.
/// * `scc` – Converged SCC result (must correspond to the same geometry).
///
/// # Returns
/// Forces in Hartree/Ångström with all four components populated.
///
/// # Note on SCC shift force
/// The exact block-resolved shift matrices are not exposed by `SccResult`.
/// We reconstruct the per-atom block shift as `shift_atom * I_{n_orb}` and
/// use the simplified formula
///   shiftSprime = 0.5 * (shift_i + shift_j) * S'
/// which is exact for the standard atom-resolved SCC model. See the
/// handoff report for the requested coordinator edit to expose block
/// shifts directly.
pub fn compute_scc_forces(
    builder: &HamiltonianBuilder,
    species: &[String],
    coords: &[[f64; 3]],
    scc: &SccResult,
) -> Result<Forces> {
    let n_atoms = species.len();
    let mut out = Forces::zeros(n_atoms);

    let ctx = SystemContext::from_sk_data(&builder.sk, species)?;

    let cutoff = builder.sk.pairs.values()
        .map(|t| t.cutoff())
        .fold(0.0_f64, f64::max);
    let neigh = NeighborBuilder { cutoff }.build(coords)?;

    // Density and energy-weighted density from the converged SCC result.
    let dm = &scc.density;
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    let c_occ = {
        // Re-diagonalize H_scc to get consistent eigenvectors for EDM.
        // SccResult does not store eigenvectors, only eigenvalues.
        // We diagonalize H_scc here to obtain them.
        let (c, _eigs) = diagonalize_h_scc(&scc.h_scc, &scc.s)?;
        c.columns(0, n_occ).into_owned()
    };
    let eps_occ: Vec<f64> = scc.eigenvalues.iter().take(n_occ).copied().collect();
    let (_dm_check, edm) = build_density_matrices(&c_occ, &eps_occ);

    // Non-SCC electronic force (uses H0 derivatives, DM and EDM).
    non_scc_electronic_force(&ctx, &neigh, coords, dm, &edm, &mut out.non_scc)?;

    // SCC shift force.
    // Reconstruct atom-resolved shifts from deltaQ and gamma.
    let gamma_tbl = GammaTable::from_sk_data(&builder.sk, species)?;
    let delta_q: Vec<f64> = scc.charges.iter().zip(scc.q0.iter())
        .map(|(q, q0)| q - q0)
        .collect();
    // shifts[i] = U_i * dq_i + Σ_{j≠i} gamma(r_ij) * dq_j
    let shifts = compute_atom_shifts(coords, &ctx.atom_species, &delta_q, &gamma_tbl);
    scc_shift_force(&ctx, &neigh, coords, dm, &shifts, &mut out.scc_shift)?;

    // SCC double-counting (Coulomb) force.
    scc_double_counting_force(coords, &ctx.atom_species, &delta_q, &gamma_tbl, &mut out.scc_dc);

    // Repulsive force.
    let sk_dir = std::env::var("RUST_DFTB_SK_DIR")
        .map_err(|_| DftbError::InvalidInput(
            "RUST_DFTB_SK_DIR must be set for repulsive forces".into()
        ))?;
    repulsive_force(coords, species, &sk_dir, &mut out.repulsive)?;

    // Sum components.
    for i in 0..n_atoms {
        for c in 0..3 {
            out.forces[i][c] = out.non_scc[i][c]
                + out.scc_shift[i][c]
                + out.scc_dc[i][c]
                + out.repulsive[i][c];
        }
    }

    check_finite(&out.forces, "SCC total");
    check_newton(&out.forces, "SCC total", 1e-6);

    Ok(out)
}

/// Four DFTB force components from caller-supplied `D` and `W` (Hartree/Å).
///
/// Used by the sparse path (`D = 2K`, `W = 2 K H_scc K`) and by tests that
/// already have dense eig `D`/`W`. Does **not** diagonalize. Same contraction
/// as `compute_scc_forces` (`non_scc_electronic_force` + `scc_shift_force` +
/// `scc_double_counting_force` + `repulsive_force_cached`).
///
/// The H-bond GPU kernels in `qmqm/gpu_forces.cl` implement the same formulas
/// for the dense multi-system path. That file is a separate in-progress
/// codepath — do not merge until both sparse and H-bond forces are tested.
pub fn compute_forces_from_dw(
    sk: &SkData,
    species: &[String],
    coords: &[[f64; 3]],
    dm: &DMatrix<f64>,
    edm: &DMatrix<f64>,
    q: &[f64],
    q0: &[f64],
    sk_dir: &str,
) -> Result<Forces> {
    let n_atoms = species.len();
    assert_eq!(coords.len(), n_atoms);
    assert_eq!(q.len(), n_atoms);
    assert_eq!(q0.len(), n_atoms);
    let ctx = SystemContext::from_sk_data(sk, species)?;
    if dm.nrows() != ctx.n_orbs || dm.ncols() != ctx.n_orbs {
        return Err(DftbError::InvalidInput(format!(
            "compute_forces_from_dw: D is {}×{}, ctx.n_orbs={}",
            dm.nrows(), dm.ncols(), ctx.n_orbs
        )));
    }
    if edm.nrows() != ctx.n_orbs || edm.ncols() != ctx.n_orbs {
        return Err(DftbError::InvalidInput(format!(
            "compute_forces_from_dw: W is {}×{}, ctx.n_orbs={}",
            edm.nrows(), edm.ncols(), ctx.n_orbs
        )));
    }
    let cutoff = sk.pairs.values().map(|t| t.cutoff()).fold(0.0_f64, f64::max);
    let neigh = NeighborBuilder { cutoff }.build(coords)?;
    let mut out = Forces::zeros(n_atoms);
    non_scc_electronic_force(&ctx, &neigh, coords, dm, edm, &mut out.non_scc)?;
    let gamma_tbl = GammaTable::from_sk_data(sk, species)?;
    let delta_q: Vec<f64> = q.iter().zip(q0.iter()).map(|(qi, q0i)| qi - q0i).collect();
    let shifts = compute_atom_shifts(coords, &ctx.atom_species, &delta_q, &gamma_tbl);
    scc_shift_force(&ctx, &neigh, coords, dm, &shifts, &mut out.scc_shift)?;
    scc_double_counting_force(coords, &ctx.atom_species, &delta_q, &gamma_tbl, &mut out.scc_dc);
    let mut names: Vec<String> = Vec::new();
    for s in species {
        if !names.iter().any(|n| n == s) {
            names.push(s.clone());
        }
    }
    let repulsive = parse_all_repulsive(sk_dir, &names, names.len())?;
    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            let si = ctx.atom_species[i] as usize;
            let sj = ctx.atom_species[j] as usize;
            let nsp = names.len();
            if repulsive[si * nsp + sj].is_none() && repulsive[sj * nsp + si].is_none() {
                return Err(DftbError::InvalidInput(format!(
                    "compute_forces_from_dw: no repulsive Spline for {}-{} in {sk_dir}",
                    species[i], species[j]
                )));
            }
        }
    }
    repulsive_force_cached(coords, &ctx, &repulsive, &mut out.repulsive)?;
    for i in 0..n_atoms {
        for c in 0..3 {
            out.forces[i][c] = out.non_scc[i][c] + out.scc_shift[i][c] + out.scc_dc[i][c] + out.repulsive[i][c];
        }
    }
    check_finite(&out.forces, "forces_from_dw total");
    check_newton(&out.forces, "forces_from_dw total", 1e-6);
    Ok(out)
}

/// Diagonalize H_scc·c = E·S·c (used to recover eigenvectors for the EDM).
fn diagonalize_h_scc(h_scc: &DMatrix<f64>, s: &DMatrix<f64>) -> Result<(DMatrix<f64>, DVector<f64>)> {
    let n = h_scc.nrows();
    let cholesky = Cholesky::new(s.clone())
        .ok_or_else(|| DftbError::InvalidInput("Overlap not positive definite".into()))?;
    let l = cholesky.l();
    let m = l.solve_lower_triangular(h_scc)
        .ok_or_else(|| DftbError::InvalidInput("L·M = H solve failed".into()))?;
    let n_mat = l.solve_lower_triangular(&m.transpose())
        .ok_or_else(|| DftbError::InvalidInput("L·N = Mᵀ solve failed".into()))?;
    let h_prime = n_mat.transpose();
    let se = SymmetricEigen::new(h_prime);
    let eigs = se.eigenvalues;
    let c_prime = se.eigenvectors;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eigs[a].partial_cmp(&eigs[b]).unwrap());
    let sorted_eigs: Vec<f64> = idx.iter().map(|&i| eigs[i]).collect();
    let sorted_c_prime = c_prime.select_columns(&idx);
    let c = l.tr_solve_lower_triangular(&sorted_c_prime)
        .ok_or_else(|| DftbError::InvalidInput("Lᵀ·c = c' solve failed".into()))?;
    Ok((c, DVector::from(sorted_eigs)))
}

/// Compute atom-resolved SCC shifts: `shift_i = U_i·dq_i + Σ_{j≠i} γ(r_ij)·dq_j`.
fn compute_atom_shifts(
    coords: &[[f64; 3]],
    species: &[u8],
    delta_q: &[f64],
    gamma_tbl: &GammaTable,
) -> Vec<f64> {
    let n = coords.len();
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        out[i] = gamma_tbl.u(species[i]) * delta_q[i];
        for j in 0..n {
            if i == j { continue; }
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r_bohr = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
            let g = gamma_tbl.gamma(r_bohr, species[i], species[j]);
            out[i] += g * delta_q[j];
        }
    }
    out
}

// ─── Unit conversion helper ────────────────────────────────────────────────

/// Convert forces from Hartree/Ångström to Hartree/Bohr.
///
/// DFTB+ reports forces in Hartree/Bohr. Our internal forces are in
/// Hartree/Ångström because the finite-difference step is in Ångström.
/// Conversion: F_bohr = F_ang * BOHR2ANG.
pub fn forces_hartree_ang_to_bohr(forces: &[[f64; 3]]) -> Vec<[f64; 3]> {
    forces.iter().map(|f| [f[0] * BOHR2ANG, f[1] * BOHR2ANG, f[2] * BOHR2ANG]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_prime_onsite_is_zero() {
        // At R=0 the derivative of gamma is 0 by symmetry.
        let g = gamma_prime_full(0.0, 0.5, 0.4);
        assert!(g.abs() < 1e-12, "gamma_prime(0) should be 0, got {g}");
    }

    #[test]
    fn gamma_prime_large_r_approaches_coulomb() {
        // At large R, gamma -> 1/R, so gamma' -> -1/R^2.
        let r = 50.0;
        let g = gamma_prime_full(r, 0.5, 0.5);
        let coulomb = -1.0 / (r * r);
        let diff = (g - coulomb).abs();
        assert!(diff < 1e-6, "gamma_prime(large R) should be ~-1/R^2 = {coulomb}, got {g}, diff {diff}");
    }

    #[test]
    fn gamma_prime_finite_diff_check() {
        // Verify gamma_prime_full against a numerical derivative of gamma_full.
        let r = 2.0;
        let u1 = 0.5;
        let u2 = 0.4;
        let h = 1e-6;
        let g_plus = crate::methods::dftb::gamma::gamma_full(r + h, u1, u2);
        let g_minus = crate::methods::dftb::gamma::gamma_full(r - h, u1, u2);
        let num = (g_plus - g_minus) / (2.0 * h);
        let ana = gamma_prime_full(r, u1, u2);
        let diff = (num - ana).abs();
        assert!(diff < 1e-6, "gamma_prime analytical {ana} vs numerical {num}, diff {diff}");
    }

    #[test]
    fn spline_exponential_head() {
        // E = exp(-a*R + b) + c, dE = -a*exp(-a*R + b)
        let spline = RepulsiveSpline {
            x_start: vec![2.0, 3.0],
            sp_coeffs: vec![[0.0, 0.0, 0.0, 0.0]],
            sp_last_coeffs: [0.0; 6],
            exp_coeffs: [3.0, 1.0, 0.5],
            cutoff: 4.0,
        };
        let r = 1.0;
        let (e, de) = spline.eval(r);
        let expected_e = (-3.0 * r + 1.0).exp() + 0.5;
        let expected_de = -3.0 * (-3.0 * r + 1.0).exp();
        assert!((e - expected_e).abs() < 1e-12, "E = {e}, expected {expected_e}");
        assert!((de - expected_de).abs() < 1e-12, "dE = {de}, expected {expected_de}");
    }

    #[test]
    fn spline_outside_cutoff_is_zero() {
        let spline = RepulsiveSpline {
            x_start: vec![2.0, 3.0],
            sp_coeffs: vec![[1.0, 1.0, 1.0, 1.0]],
            sp_last_coeffs: [1.0; 6],
            exp_coeffs: [1.0, 0.0, 0.0],
            cutoff: 4.0,
        };
        let (e, de) = spline.eval(5.0);
        assert_eq!(e, 0.0);
        assert_eq!(de, 0.0);
    }

    // ==================================================================
    // P2 Gate B test #2: full analytic f64 force vs f64 total-energy FD
    // (manifest v3 §4.3)
    //
    // Builds a synthetic H2 molecule with smooth exponential SK tables,
    // computes the non-SCC electronic energy E = Tr(D·H0), and compares
    // the analytic force F = -dE/dR against central finite differences of E.
    //
    // This is the "full system: analytic f64 force against finite difference
    // of total f64 energy" test. It validates the entire force pipeline:
    //   SK eval → rotation → H0 assembly → diagonalize → DM/EDM → force
    //
    // The test is self-contained — no external SK files needed.
    // ==================================================================

    use crate::methods::dftb::interpolation::EqGridTable;
    use crate::methods::dftb::sk_data::{AtomicParamsSp, SkTableSp, SpeciesOrbitals};
    use std::collections::HashMap;

    /// Build a synthetic SkData for H (1s orbital) with smooth exponential SK tables.
    /// Uses the extended 20-integral DFTB+ format: ss integral at index 19
    /// (sk_map(0,0,0)=20, extended path: out = h_all[20-1] = h_all[19]).
    fn make_h2_sk_data() -> SkData {
        let dr = 0.1; // Bohr
        let n_grid = 100;
        let r_max = dr * n_grid as f64;

        // H-H pair table: 20 integrals, only ss (index 19) is non-zero.
        // V(r) = -0.3 * exp(-1.0 * r)
        let hh_values: Vec<Vec<f64>> = (0..n_grid)
            .map(|i| {
                let r = i as f64 * dr;
                let tail = if r > r_max - 1.0 {
                    let t = (r_max - r) / 1.0;
                    t * t
                } else { 1.0 };
                let v = -0.3 * (-1.0 * r).exp() * tail;
                let mut row = vec![0.0f64; 20];
                row[19] = v;  // ss integral (sk_map(0,0,0)=20 → index 19)
                row
            })
            .collect();
        let hh_h = EqGridTable::new(dr, hh_values.clone());
        let hh_s = EqGridTable::new(dr, hh_values);
        let hh_table = SkTableSp {
            sp1: "H".to_string(),
            sp2: "H".to_string(),
            h: hh_h,
            s: hh_s,
        };

        let mut pairs = HashMap::new();
        pairs.insert(("H".to_string(), "H".to_string()), hh_table);

        let mut onsite = HashMap::new();
        onsite.insert("H".to_string(), AtomicParamsSp {
            e_s: -0.4,  // Onsite energy (Hartree)
            e_p: 0.0,
            q0: 1.0,    // 1 electron
            u_hubbard: 0.5,
        });

        let mut orbital_info = HashMap::new();
        orbital_info.insert("H".to_string(), SpeciesOrbitals::from_ang_momenta(&[0]));

        SkData {
            onsite,
            pairs,
            orbital_info,
        }
    }

    /// Compute the non-SCC electronic energy E = Tr(D·H0) for a given geometry.
    fn non_scc_electronic_energy(
        builder: &HamiltonianBuilder,
        species: &[String],
        coords: &[[f64; 3]],
        n_electrons: f64,
    ) -> Result<f64> {
        let ham = builder.build_non_scc(species, coords)?;
        let (c, _eigs) = diagonalize_non_scc(&ham, n_electrons)?;
        let n_occ = (n_electrons / 2.0).round() as usize;
        let c_occ = c.columns(0, n_occ).into_owned();
        let dm = &c_occ * c_occ.transpose() * 2.0;
        let e = (&dm * &ham.h0).trace();
        Ok(e)
    }

    #[test]
    fn test_analytic_force_vs_energy_fd_h2() {
        let sk = make_h2_sk_data();
        let builder = HamiltonianBuilder::new(sk);
        let species = vec!["H".to_string(), "H".to_string()];
        let n_electrons = 2.0; // H2 has 2 electrons

        // Bond length ~1.0 Å (well within cutoff, non-trivial force)
        let r_ang = 1.0;
        let coords = vec![[0.0, 0.0, 0.0], [r_ang, 0.0, 0.0]];

        // Compute analytic non-SCC electronic force
        let ctx = SystemContext::from_sk_data(&builder.sk, &species).unwrap();
        let cutoff = builder.sk.pairs.values()
            .map(|t| t.cutoff())
            .fold(0.0_f64, f64::max);
        let neigh = NeighborBuilder { cutoff }.build(&coords).unwrap();
        let ham = builder.build_non_scc(&species, &coords).unwrap();
        let (c, eigs) = diagonalize_non_scc(&ham, n_electrons).unwrap();
        let n_occ = (n_electrons / 2.0).round() as usize;
        let c_occ = c.columns(0, n_occ).into_owned();
        let eps_occ: Vec<f64> = eigs.iter().take(n_occ).copied().collect();
        let (dm, edm) = build_density_matrices(&c_occ, &eps_occ);

        let mut forces = vec![[0.0f64; 3]; 2];
        non_scc_electronic_force(&ctx, &neigh, &coords, &dm, &edm, &mut forces).unwrap();

        // Finite difference: F = -dE/dR
        let delta = 1e-5; // Å (small for 2nd-order FD accuracy)
        let mut max_err = 0.0f64;
        let mut max_force = 0.0f64;
        for atom in 0..2 {
            for dir in 0..3 {
                let mut coords_plus = coords.clone();
                let mut coords_minus = coords.clone();
                coords_plus[atom][dir] += delta;
                coords_minus[atom][dir] -= delta;

                let e_plus = non_scc_electronic_energy(&builder, &species, &coords_plus, n_electrons).unwrap();
                let e_minus = non_scc_electronic_energy(&builder, &species, &coords_minus, n_electrons).unwrap();

                // F = -dE/dR ≈ -(E+ - E-) / (2*delta)
                let fd_force = -(e_plus - e_minus) / (2.0 * delta);
                let analytic_force = forces[atom][dir];

                let err = (fd_force - analytic_force).abs();
                max_err = max_err.max(err);
                max_force = max_force.max(analytic_force.abs());

                eprintln!("atom {atom} dir {dir}: analytic={analytic_force:.6e} fd={fd_force:.6e} err={err:.3e}");
            }
        }
        // Relative tolerance: the force should match FD to ~1e-4 relative
        // (limited by FD step size and numerical roundoff in diagonalization).
        let rel_err = if max_force > 1e-10 { max_err / max_force } else { max_err };
        eprintln!("max|F|={max_force:.3e}  max|err|={max_err:.3e}  rel_err={rel_err:.3e}");
        assert!(rel_err < 1e-4,
            "analytic force vs energy FD: rel_err={rel_err:.3e} too large (max|F|={max_force:.3e}, max|err|={max_err:.3e})");
    }

    #[test]
    fn test_analytic_force_vs_energy_fd_h2_tilted() {
        // Same test but with a tilted geometry to exercise all 3 directions
        let sk = make_h2_sk_data();
        let builder = HamiltonianBuilder::new(sk);
        let species = vec!["H".to_string(), "H".to_string()];
        let n_electrons = 2.0;

        // Tilted bond: not aligned with any axis
        let coords = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.6, 0.7],  // ~1.05 Å bond length
        ];

        let ctx = SystemContext::from_sk_data(&builder.sk, &species).unwrap();
        let cutoff = builder.sk.pairs.values()
            .map(|t| t.cutoff())
            .fold(0.0_f64, f64::max);
        let neigh = NeighborBuilder { cutoff }.build(&coords).unwrap();
        let ham = builder.build_non_scc(&species, &coords).unwrap();
        let (c, eigs) = diagonalize_non_scc(&ham, n_electrons).unwrap();
        let n_occ = (n_electrons / 2.0).round() as usize;
        let c_occ = c.columns(0, n_occ).into_owned();
        let eps_occ: Vec<f64> = eigs.iter().take(n_occ).copied().collect();
        let (dm, edm) = build_density_matrices(&c_occ, &eps_occ);

        let mut forces = vec![[0.0f64; 3]; 2];
        non_scc_electronic_force(&ctx, &neigh, &coords, &dm, &edm, &mut forces).unwrap();

        let delta = 1e-5;
        let mut max_err = 0.0f64;
        let mut max_force = 0.0f64;
        for atom in 0..2 {
            for dir in 0..3 {
                let mut coords_plus = coords.clone();
                let mut coords_minus = coords.clone();
                coords_plus[atom][dir] += delta;
                coords_minus[atom][dir] -= delta;

                let e_plus = non_scc_electronic_energy(&builder, &species, &coords_plus, n_electrons).unwrap();
                let e_minus = non_scc_electronic_energy(&builder, &species, &coords_minus, n_electrons).unwrap();

                let fd_force = -(e_plus - e_minus) / (2.0 * delta);
                let analytic_force = forces[atom][dir];
                let err = (fd_force - analytic_force).abs();
                max_err = max_err.max(err);
                max_force = max_force.max(analytic_force.abs());
            }
        }
        let rel_err = if max_force > 1e-10 { max_err / max_force } else { max_err };
        eprintln!("tilted: max|F|={max_force:.3e}  max|err|={max_err:.3e}  rel_err={rel_err:.3e}");
        assert!(rel_err < 1e-4,
            "analytic force vs energy FD (tilted): rel_err={rel_err:.3e} too large");
    }

    /// Synthetic 4-orbital (sp) species for a multi-atom test.
    /// Uses the extended 20-integral DFTB+ format:
    ///   ss  → sk_map(0,0,0)=20 → index 19
    ///   sp  → sk_map(0,1,0)=19 → index 18
    ///   ppσ → sk_map(0,1,1)=15 → index 14
    ///   ppπ → sk_map(1,1,1)=16 → index 15
    fn make_c_like_sk_data() -> SkData {
        let dr = 0.1;
        let n_grid = 100;
        let r_max = dr * n_grid as f64;

        let make_values = |decay: f64, amp: f64| -> Vec<Vec<f64>> {
            (0..n_grid)
                .map(|i| {
                    let r = i as f64 * dr;
                    let tail = if r > r_max - 1.0 {
                        let t = (r_max - r) / 1.0;
                        t * t
                    } else { 1.0 };
                    let base = amp * (-decay * r).exp() * tail;
                    let mut row = vec![0.0f64; 20];
                    row[19] = base;           // ss
                    row[18] = 0.8 * base;     // sp
                    row[14] = 0.6 * base;     // pp_sigma
                    row[15] = 0.3 * base;     // pp_pi
                    row
                })
                .collect()
        };

        let x_h = EqGridTable::new(dr, make_values(1.0, -0.3));
        let x_s = EqGridTable::new(dr, make_values(1.0, 0.2));
        let xx_table = SkTableSp {
            sp1: "X".to_string(),
            sp2: "X".to_string(),
            h: x_h,
            s: x_s,
        };

        let mut pairs = HashMap::new();
        pairs.insert(("X".to_string(), "X".to_string()), xx_table);

        let mut onsite = HashMap::new();
        onsite.insert("X".to_string(), AtomicParamsSp {
            e_s: -0.5,
            e_p: -0.1,
            q0: 4.0,    // 4 valence electrons
            u_hubbard: 0.5,
        });

        let mut orbital_info = HashMap::new();
        orbital_info.insert("X".to_string(), SpeciesOrbitals::from_ang_momenta(&[0, 1]));

        SkData { onsite, pairs, orbital_info }
    }

    #[test]
    fn test_analytic_force_vs_energy_fd_sp3() {
        // 2-atom sp3 system (like C2) — exercises pp, sp, ps blocks.
        // Using 2 atoms avoids near-degeneracy issues that cause eigenvalue
        // ordering flips in the FD energy, which would make the FD force
        // discontinuous and not comparable to the analytic force.
        let sk = make_c_like_sk_data();
        let builder = HamiltonianBuilder::new(sk);
        let species = vec!["X".to_string(), "X".to_string()];
        let n_electrons = 8.0; // 2 atoms × 4 electrons

        // Tilted bond to exercise all 3 directions
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.3, 0.4, 0.2],
        ];

        let ctx = SystemContext::from_sk_data(&builder.sk, &species).unwrap();
        let cutoff = builder.sk.pairs.values()
            .map(|t| t.cutoff())
            .fold(0.0_f64, f64::max);
        let neigh = NeighborBuilder { cutoff }.build(&coords).unwrap();
        let ham = builder.build_non_scc(&species, &coords).unwrap();
        let (c, eigs) = diagonalize_non_scc(&ham, n_electrons).unwrap();
        let n_occ = (n_electrons / 2.0).round() as usize;
        let c_occ = c.columns(0, n_occ).into_owned();
        let eps_occ: Vec<f64> = eigs.iter().take(n_occ).copied().collect();
        let (dm, edm) = build_density_matrices(&c_occ, &eps_occ);

        let mut forces = vec![[0.0f64; 3]; 2];
        non_scc_electronic_force(&ctx, &neigh, &coords, &dm, &edm, &mut forces).unwrap();

        let delta = 1e-5;
        let mut max_err = 0.0f64;
        let mut max_force = 0.0f64;
        for atom in 0..2 {
            for dir in 0..3 {
                let mut coords_plus = coords.clone();
                let mut coords_minus = coords.clone();
                coords_plus[atom][dir] += delta;
                coords_minus[atom][dir] -= delta;

                let e_plus = non_scc_electronic_energy(&builder, &species, &coords_plus, n_electrons).unwrap();
                let e_minus = non_scc_electronic_energy(&builder, &species, &coords_minus, n_electrons).unwrap();

                let fd_force = -(e_plus - e_minus) / (2.0 * delta);
                let analytic_force = forces[atom][dir];
                let err = (fd_force - analytic_force).abs();
                max_err = max_err.max(err);
                max_force = max_force.max(analytic_force.abs());

                eprintln!("atom {atom} dir {dir}: analytic={analytic_force:.6e} fd={fd_force:.6e} err={err:.3e}");
            }
        }
        let rel_err = if max_force > 1e-10 { max_err / max_force } else { max_err };
        eprintln!("sp3: max|F|={max_force:.3e}  max|err|={max_err:.3e}  rel_err={rel_err:.3e}");
        assert!(rel_err < 1e-4,
            "analytic force vs energy FD (sp3): rel_err={rel_err:.3e} too large (max|F|={max_force:.3e}, max|err|={max_err:.3e})");
    }
}
