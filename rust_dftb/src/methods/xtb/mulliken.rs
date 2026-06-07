//! Mulliken population analysis for xTB

use crate::methods::xtb::params;
use crate::methods::xtb::params_gfn2;
use nalgebra::{DMatrix, DVector};

/// Compute shell-resolved Mulliken charges
/// qsh(ish) = n0sh(ish) - sum_{jao} P(jao, iao) * S(jao, iao)
pub fn shell_charges(
    density: &DMatrix<f64>,
    overlap: &DMatrix<f64>,
    ao2sh: &[usize],
    n0sh: &DVector<f64>,
) -> DVector<f64> {
    let nao = density.nrows();
    let nshell = n0sh.len();
    let mut qsh = n0sh.clone();

    // For each AO, compute its contribution to its shell
    for iao in 0..nao {
        let ish = ao2sh[iao];
        let mut pao = 0.0;
        for jao in 0..nao {
            pao += density[(jao, iao)] * overlap[(jao, iao)];
        }
        qsh[ish] -= pao;
    }

    qsh
}

/// Compute atomic Mulliken charges from shell charges
pub fn atomic_charges(
    shell_charges: &DVector<f64>,
    nshell_per_atom: &[usize],
) -> DVector<f64> {
    let nat = nshell_per_atom.len();
    let mut qat = DVector::zeros(nat);

    let mut offset = 0;
    for iat in 0..nat {
        let nsh = nshell_per_atom[iat];
        for ish in 0..nsh {
            qat[iat] += shell_charges[offset + ish];
        }
        offset += nsh;
    }

    qat
}

/// Compute atomic multipole moments from density matrix and multipole integrals.
/// Reproduces tblite wavefunction/mulliken.f90 get_mulliken_atomic_multipoles.
///
/// mpat(cmp, iat) = -sum_{iao on iat} sum_{jao} P(jao, iao) * mpint(cmp, jao, iao)
///
/// Inputs:
///   - density: nao × nao density matrix
///   - mp_ints: flattened [nmp][nao][nao] multipole integrals in Fortran column-major order
///   - ao2at: mapping from AO index to atom index
///   - nmp: number of multipole components (3 for dipole, 6 for quadrupole)
pub fn atomic_multipoles(
    density: &DMatrix<f64>,
    mp_ints: &[f64],
    ao2at: &[usize],
    nmp: usize,
) -> DMatrix<f64> {
    let nao = density.nrows();
    let nat = *ao2at.iter().max().unwrap() + 1;
    let mut mpat = DMatrix::zeros(nmp, nat);

    for iao in 0..nao {
        let iat = ao2at[iao];
        let mut pao = vec![0.0f64; nmp];
        for jao in 0..nao {
            let p_ji = density[(jao, iao)];
            for cmp in 0..nmp {
                let mp_ji = mp_ints[cmp + nmp * jao + nmp * nao * iao];
                pao[cmp] += p_ji * mp_ji;
            }
        }
        for cmp in 0..nmp {
            mpat[(cmp, iat)] -= pao[cmp];
        }
    }

    mpat
}

/// Generic reference shell occupations builder
fn reference_shell_occupations_generic(
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
    reference_occ: &[[f64; 3]],
    valence_filter: bool,
) -> DVector<f64> {
    let nat = nshell_per_atom.len();
    let nshell_total: usize = nshell_per_atom.iter().sum();
    let mut n0sh = DVector::zeros(nshell_total);

    let mut offset = 0;
    for iat in 0..nat {
        let izp = elem_idx[iat];
        let nsh = nshell_per_atom[iat];
        let mut seen_ang = [false; 3];
        for ish in 0..nsh {
            let l = ang_per_shell[offset + ish];
            let is_valence = if valence_filter {
                let first = !seen_ang[l];
                seen_ang[l] = true;
                first
            } else {
                true
            };
            if is_valence {
                n0sh[offset + ish] = reference_occ[izp][l];
            }
        }
        offset += nsh;
    }

    n0sh
}

/// Compute reference shell occupations for GFN1 (with valence filter)
pub fn reference_shell_occupations(
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DVector<f64> {
    reference_shell_occupations_generic(nshell_per_atom, elem_idx, ang_per_shell, &params::reference_occ, true)
}

/// Compute reference shell occupations for GFN2 (no valence filter)
pub fn reference_shell_occupations_gfn2(
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DVector<f64> {
    reference_shell_occupations_generic(nshell_per_atom, elem_idx, ang_per_shell, &params_gfn2::reference_occ, false)
}
