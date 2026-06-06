//! Mulliken population analysis for xTB

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

/// Compute reference shell occupations (n0sh) from element data
/// For GFN1, this is the number of electrons in each shell for neutral atom
pub fn reference_shell_occupations(
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
) -> DVector<f64> {
    let nat = nshell_per_atom.len();
    let nshell_total: usize = nshell_per_atom.iter().sum();
    let mut n0sh = DVector::zeros(nshell_total);

    let mut offset = 0;
    for iat in 0..nat {
        let izp = elem_idx[iat];
        let nsh = nshell_per_atom[iat];

        // Simple electron counting for H, C, N, O
        // H: 1s(1), 2s(0)
        // C: 2s(2), 2p(2)
        // N: 2s(2), 2p(3)
        // O: 2s(2), 2p(4)
        match izp {
            0 => { // H
                n0sh[offset] = 1.0; // 1s
                if nsh > 1 { n0sh[offset + 1] = 0.0; } // 2s
            }
            1 => { // He
                n0sh[offset] = 2.0; // 1s
            }
            2 => { // Li
                n0sh[offset] = 2.0; // 2s
                n0sh[offset + 1] = 1.0; // 2p
            }
            3 => { // Be
                n0sh[offset] = 2.0; // 2s
                n0sh[offset + 1] = 2.0; // 2p
            }
            4 => { // B
                n0sh[offset] = 2.0; // 2s
                n0sh[offset + 1] = 3.0; // 2p
            }
            5 => { // C
                n0sh[offset] = 2.0; // 2s
                n0sh[offset + 1] = 2.0; // 2p
            }
            6 => { // N
                n0sh[offset] = 2.0; // 2s
                n0sh[offset + 1] = 3.0; // 2p
            }
            7 => { // O
                n0sh[offset] = 2.0; // 2s
                n0sh[offset + 1] = 4.0; // 2p
            }
            _ => {
                // Default: fill shells based on angular momentum
                for ish in 0..nsh {
                    let l = ang_per_shell[offset + ish];
                    // s: 2, p: 6, d: 10 (but this is approximate)
                    n0sh[offset + ish] = match l {
                        0 => 2.0,
                        1 => 6.0,
                        2 => 10.0,
                        _ => 0.0,
                    };
                }
            }
        }
        offset += nsh;
    }

    n0sh
}
