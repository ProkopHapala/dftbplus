//! Coulomb interaction models for GFN1-xTB SCC electrostatics
//!
//! Implements:
//! - effective_coulomb: Klopman-Ohno kernel with harmonic averaging
//! - onsite_thirdorder: Third-order Hubbard correction

use crate::methods::xtb::params::{GEXP, hubbard_parameter, shell_hubbard, hubbard_derivs};
use nalgebra::{DMatrix, DVector};

/// Harmonic average of two Hubbard parameters (GFN1 convention)
fn harmonic_average(gi: f64, gj: f64) -> f64 {
    2.0 / (1.0 / gi + 1.0 / gj)
}

/// Build shell-resolved Hubbard parameters for all atoms
/// Returns matrix [nshell_total][nshell_total] of averaged Hubbard parameters
pub fn build_shell_hubbard_matrix(
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DMatrix<f64> {
    let nat = nshell_per_atom.len();
    let nshell_total: usize = nshell_per_atom.iter().sum();

    let mut hubbard = DMatrix::zeros(nshell_total, nshell_total);

    // Build shell offsets
    let mut shell_offset = vec![0usize; nat + 1];
    for i in 0..nat {
        shell_offset[i + 1] = shell_offset[i] + nshell_per_atom[i];
    }

    // Fill Hubbard matrix
    for iat in 0..nat {
        let izp = elem_idx[iat];
        let ii = shell_offset[iat];
        for ish in 0..nshell_per_atom[iat] {
            let il = ang_per_shell[ii + ish];
            let gi = hubbard_parameter[izp] * shell_hubbard[izp][il];

            for jat in 0..nat {
                let jzp = elem_idx[jat];
                let jj = shell_offset[jat];
                for jsh in 0..nshell_per_atom[jat] {
                    let jl = ang_per_shell[jj + jsh];
                    let gj = hubbard_parameter[jzp] * shell_hubbard[jzp][jl];

                    let gij = harmonic_average(gi, gj);
                    hubbard[(ii + ish, jj + jsh)] = gij;
                }
            }
        }
    }

    hubbard
}

/// Build Coulomb matrix (gamma matrix) using Klopman-Ohno kernel
/// For finite systems (non-periodic)
pub fn build_coulomb_matrix(
    coords: &[[f64; 3]],
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DMatrix<f64> {
    let nat = coords.len();
    let nshell_total: usize = nshell_per_atom.iter().sum();
    let gexp = GEXP;

    let hubbard = build_shell_hubbard_matrix(nshell_per_atom, elem_idx, ang_per_shell);
    let mut gamma = DMatrix::zeros(nshell_total, nshell_total);

    // Build shell offsets
    let mut shell_offset = vec![0usize; nat + 1];
    for i in 0..nat {
        shell_offset[i + 1] = shell_offset[i] + nshell_per_atom[i];
    }

    // Off-diagonal terms (atom pairs)
    for iat in 0..nat {
        let ii = shell_offset[iat];
        for jat in 0..iat {
            let jj = shell_offset[jat];

            let dx = coords[jat][0] - coords[iat][0];
            let dy = coords[jat][1] - coords[iat][1];
            let dz = coords[jat][2] - coords[iat][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let rg = r.powf(gexp);

            for ish in 0..nshell_per_atom[iat] {
                for jsh in 0..nshell_per_atom[jat] {
                    let gam = hubbard[(ii + ish, jj + jsh)];
                    let gij = 1.0 / (rg + gam.powf(-gexp)).powf(1.0 / gexp);
                    gamma[(ii + ish, jj + jsh)] = gij;
                    gamma[(jj + jsh, ii + ish)] = gij;
                }
            }
        }
    }

    // On-site terms (same atom, different shells)
    for iat in 0..nat {
        let izp = elem_idx[iat];
        let ii = shell_offset[iat];
        for ish in 0..nshell_per_atom[iat] {
            for jsh in 0..ish {
                let gam = hubbard[(ii + ish, ii + jsh)];
                gamma[(ii + ish, ii + jsh)] = gam;
                gamma[(ii + jsh, ii + ish)] = gam;
            }
            // Diagonal
            gamma[(ii + ish, ii + ish)] = hubbard[(ii + ish, ii + ish)];
        }
    }

    gamma
}

/// Third-order onsite correction (atom-resolved, as in GFN1: shell=.false.)
/// Uses atomic charge qat = sum_ish qsh(ish), returns shell potential shift.
/// vat(iat) = qat(iat)^2 * hubbard_deriv(iat)
/// vsh(ish) += vat(iat)   for all shells ish on atom iat
pub fn thirdorder_potential(
    shell_charges: &DVector<f64>,
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
) -> DVector<f64> {
    let nat = nshell_per_atom.len();
    let nshell_total = shell_charges.len();
    let mut v3 = DVector::zeros(nshell_total);

    let mut shell_offset = vec![0usize; nat + 1];
    for i in 0..nat {
        shell_offset[i + 1] = shell_offset[i] + nshell_per_atom[i];
    }

    for iat in 0..nat {
        let izp = elem_idx[iat];
        let ii = shell_offset[iat];
        let deriv = hubbard_derivs[izp];

        // Atomic charge = sum of shell charges
        let qat: f64 = (0..nshell_per_atom[iat]).map(|ish| shell_charges[ii + ish]).sum();
        let vat = qat * qat * deriv;

        // Distribute to all shells on this atom
        for ish in 0..nshell_per_atom[iat] {
            v3[ii + ish] = vat;
        }
    }

    v3
}

/// Third-order energy
pub fn thirdorder_energy(
    shell_charges: &DVector<f64>,
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
) -> f64 {
    let nat = nshell_per_atom.len();
    let mut e3 = 0.0;

    let mut shell_offset = vec![0usize; nat + 1];
    for i in 0..nat {
        shell_offset[i + 1] = shell_offset[i] + nshell_per_atom[i];
    }

    for iat in 0..nat {
        let izp = elem_idx[iat];
        let ii = shell_offset[iat];
        let deriv = hubbard_derivs[izp];
        for ish in 0..nshell_per_atom[iat] {
            let q = shell_charges[ii + ish];
            e3 += q.powi(3) * deriv / 3.0;
        }
    }

    e3
}
