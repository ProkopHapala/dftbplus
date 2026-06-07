//! Coulomb interaction models for GFN1-xTB and GFN2-xTB SCC electrostatics
//!
//! Implements:
//! - effective_coulomb: Klopman-Ohno kernel with harmonic/arithmetic averaging
//! - onsite_thirdorder: Third-order Hubbard correction (atom- or shell-resolved)

use crate::methods::xtb::params::{GEXP as GEXP1, hubbard_parameter as hubbard_param1, shell_hubbard as shell_hubbard1, hubbard_derivs as hubbard_derivs1};
use crate::methods::xtb::params_gfn2::{GEXP as GEXP2, hubbard_parameter as hubbard_param2, shell_hubbard as shell_hubbard2, hubbard_derivs as hubbard_derivs2, shell_hubbard_derivs};
use nalgebra::{DMatrix, DVector};

/// Harmonic average of two Hubbard parameters (GFN1 convention)
fn harmonic_average(gi: f64, gj: f64) -> f64 {
    2.0 / (1.0 / gi + 1.0 / gj)
}

/// Arithmetic average of two Hubbard parameters (GFN2 convention)
fn arithmetic_average(gi: f64, gj: f64) -> f64 {
    0.5 * (gi + gj)
}

/// Build shell-resolved Hubbard parameters for all atoms (GFN1)
pub fn build_shell_hubbard_matrix(
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DMatrix<f64> {
    build_shell_hubbard_matrix_generic(
        nshell_per_atom, elem_idx, ang_per_shell,
        &hubbard_param1, &shell_hubbard1, harmonic_average,
    )
}

/// Build shell-resolved Hubbard parameters for all atoms (GFN2)
pub fn build_shell_hubbard_matrix_gfn2(
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DMatrix<f64> {
    build_shell_hubbard_matrix_generic(
        nshell_per_atom, elem_idx, ang_per_shell,
        &hubbard_param2, &shell_hubbard2, arithmetic_average,
    )
}

fn build_shell_hubbard_matrix_generic(
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
    hubbard_param: &[f64],
    shell_hubbard: &[[f64; 3]],
    avg_fn: fn(f64, f64) -> f64,
) -> DMatrix<f64> {
    let nat = nshell_per_atom.len();
    let nshell_total: usize = nshell_per_atom.iter().sum();

    let mut hubbard = DMatrix::zeros(nshell_total, nshell_total);

    let mut shell_offset = vec![0usize; nat + 1];
    for i in 0..nat {
        shell_offset[i + 1] = shell_offset[i] + nshell_per_atom[i];
    }

    for iat in 0..nat {
        let izp = elem_idx[iat];
        let ii = shell_offset[iat];
        for ish in 0..nshell_per_atom[iat] {
            let il = ang_per_shell[ii + ish];
            let gi = hubbard_param[izp] * shell_hubbard[izp][il];

            for jat in 0..nat {
                let jzp = elem_idx[jat];
                let jj = shell_offset[jat];
                for jsh in 0..nshell_per_atom[jat] {
                    let jl = ang_per_shell[jj + jsh];
                    let gj = hubbard_param[jzp] * shell_hubbard[jzp][jl];

                    let gij = avg_fn(gi, gj);
                    hubbard[(ii + ish, jj + jsh)] = gij;
                }
            }
        }
    }

    hubbard
}

/// Build Coulomb matrix (gamma matrix) using Klopman-Ohno kernel (GFN1)
pub fn build_coulomb_matrix(
    coords: &[[f64; 3]],
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DMatrix<f64> {
    build_coulomb_matrix_generic(
        coords, nshell_per_atom, elem_idx, ang_per_shell,
        build_shell_hubbard_matrix, GEXP1,
    )
}

/// Build Coulomb matrix (gamma matrix) using Klopman-Ohno kernel (GFN2)
pub fn build_coulomb_matrix_gfn2(
    coords: &[[f64; 3]],
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DMatrix<f64> {
    build_coulomb_matrix_generic(
        coords, nshell_per_atom, elem_idx, ang_per_shell,
        build_shell_hubbard_matrix_gfn2, GEXP2,
    )
}

fn build_coulomb_matrix_generic(
    coords: &[[f64; 3]],
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
    hubbard_builder: fn(&[usize], &[usize], &[usize]) -> DMatrix<f64>,
    gexp: f64,
) -> DMatrix<f64> {
    let nat = coords.len();
    let nshell_total: usize = nshell_per_atom.iter().sum();

    let hubbard = hubbard_builder(nshell_per_atom, elem_idx, ang_per_shell);
    let mut gamma = DMatrix::zeros(nshell_total, nshell_total);

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

/// Third-order onsite correction (atom-resolved, GFN1)
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
        let deriv = hubbard_derivs1[izp];

        let qat: f64 = (0..nshell_per_atom[iat]).map(|ish| shell_charges[ii + ish]).sum();
        let vat = qat * qat * deriv;

        for ish in 0..nshell_per_atom[iat] {
            v3[ii + ish] = vat;
        }
    }

    v3
}

/// Third-order onsite correction (shell-resolved, GFN2)
/// vsh(ish) = qsh(ish)^2 * p_hubbard_derivs(izp) * shell_hubbard_derivs(il)
pub fn thirdorder_potential_gfn2(
    shell_charges: &DVector<f64>,
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
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
        let deriv_base = hubbard_derivs2[izp];
        for ish in 0..nshell_per_atom[iat] {
            let il = ang_per_shell[ii + ish];
            let deriv = deriv_base * shell_hubbard_derivs[il];
            let q = shell_charges[ii + ish];
            v3[ii + ish] = q * q * deriv;
        }
    }

    v3
}

/// Third-order energy (GFN1)
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
        let deriv = hubbard_derivs1[izp];
        for ish in 0..nshell_per_atom[iat] {
            let q = shell_charges[ii + ish];
            e3 += q.powi(3) * deriv / 3.0;
        }
    }

    e3
}

/// Third-order energy (GFN2, shell-resolved)
pub fn thirdorder_energy_gfn2(
    shell_charges: &DVector<f64>,
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
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
        let deriv_base = hubbard_derivs2[izp];
        for ish in 0..nshell_per_atom[iat] {
            let il = ang_per_shell[ii + ish];
            let deriv = deriv_base * shell_hubbard_derivs[il];
            let q = shell_charges[ii + ish];
            e3 += q.powi(3) * deriv / 3.0;
        }
    }

    e3
}
