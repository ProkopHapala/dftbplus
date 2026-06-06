//! Analytical overlap integrals for contracted Cartesian Gaussian type orbitals.

use crate::methods::xtb::basis::Cgto;
use nalgebra::{Matrix3, Vector3};

/// Compute overlap matrix between two CGTO shells at displacement vector `d = r_i - r_j`
/// Returns a flat array with shape [n_ao_j][n_ao_i] where n_ao = msao(l)
pub fn overlap_cgto(cgto_i: &Cgto, cgto_j: &Cgto, d: &[f64; 3]) -> Vec<f64> {
    let li = cgto_i.ang;
    let lj = cgto_j.ang;
    let nao_i = msao(li);
    let nao_j = msao(lj);
    let mut result = vec![0.0; nao_j * nao_i];

    for ii in 0..cgto_i.nprim {
        for jj in 0..cgto_j.nprim {
            let ci = cgto_i.coeff[ii];
            let cj = cgto_j.coeff[jj];
            let ai = cgto_i.alpha[ii];
            let bj = cgto_j.alpha[jj];
            let c = ci * cj;

            match (li, lj) {
                (0, 0) => {
                    result[0] += c * primitive_overlap_ss(ai, bj, d);
                }
                (0, 1) => {
                    // tblite p ordering: [py, pz, px]
                    let s = primitive_overlap_sp(ai, bj, d);
                    result[0] += c * s[1]; // py
                    result[1] += c * s[2]; // pz
                    result[2] += c * s[0]; // px
                }
                (1, 0) => {
                    let s = primitive_overlap_sp(bj, ai, &[-d[0], -d[1], -d[2]]);
                    result[0] += c * s[1]; // py
                    result[1] += c * s[2]; // pz
                    result[2] += c * s[0]; // px
                }
                (1, 1) => {
                    // tblite p ordering: [py, pz, px]
                    let s = primitive_overlap_pp(ai, bj, d);
                    let p = [1usize, 2, 0]; // new index -> old index
                    for row in 0..3 {
                        for col in 0..3 {
                            result[row * 3 + col] += c * s[p[row]][p[col]];
                        }
                    }
                }
                _ => panic!("Angular momentum combination l={li}, l={lj} not yet implemented"),
            }
        }
    }
    result
}

/// Number of AOs for a given angular momentum (Cartesian ordering)
pub fn msao(l: usize) -> usize {
    match l {
        0 => 1,
        1 => 3,
        2 => 5,
        _ => panic!("Unsupported l={l}"),
    }
}

/// s-s primitive overlap: S = (π/(α+β))^(3/2) * exp(-αβ/(α+β) * r²)
fn primitive_overlap_ss(alpha: f64, beta: f64, d: &[f64; 3]) -> f64 {
    let r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    let gamma = alpha + beta;
    let pref = (std::f64::consts::PI / gamma).powf(1.5);
    let exp_term = (-alpha * beta * r2 / gamma).exp();
    pref * exp_term
}

/// s-p primitive overlap: returns [S_x, S_y, S_z]
/// For s at A (exponent α) and p at B (exponent β), displacement d = A - B
fn primitive_overlap_sp(alpha: f64, beta: f64, d: &[f64; 3]) -> [f64; 3] {
    let gamma = alpha + beta;
    let s0 = primitive_overlap_ss(alpha, beta, d);
    [
        (alpha * d[0] / gamma) * s0,
        (alpha * d[1] / gamma) * s0,
        (alpha * d[2] / gamma) * s0,
    ]
}

/// p-p primitive overlap: returns 3x3 matrix [ao_j][ao_i]
/// For p at A (exponent α) and p at B (exponent β), displacement d = A - B
fn primitive_overlap_pp(alpha: f64, beta: f64, d: &[f64; 3]) -> [[f64; 3]; 3] {
    let gamma = alpha + beta;
    let s0 = primitive_overlap_ss(alpha, beta, d);
    let factor = -alpha * beta / (gamma * gamma);
    let delta = 1.0 / (2.0 * gamma);
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let off_diag = factor * d[i] * d[j];
            let diag = if i == j { delta } else { 0.0 };
            result[j][i] = s0 * (off_diag + diag);
        }
    }
    result
}

/// Build full overlap matrix S for a system given CGTOs per atom and coordinates
pub fn build_overlap_matrix(
    cgtos_per_atom: &[Vec<Cgto>],
    coords: &[[f64; 3]],
) -> nalgebra::DMatrix<f64> {
    assert_eq!(cgtos_per_atom.len(), coords.len());

    // Count total AOs
    let mut nao = 0usize;
    for atom_cgtos in cgtos_per_atom {
        for cgto in atom_cgtos {
            nao += msao(cgto.ang);
        }
    }

    let mut s = nalgebra::DMatrix::<f64>::zeros(nao, nao);

    let mut i_off = 0usize;
    for (iat, atom_i) in cgtos_per_atom.iter().enumerate() {
        for cgto_i in atom_i {
            let nao_i = msao(cgto_i.ang);
            let mut j_off = 0usize;
            for (jat, atom_j) in cgtos_per_atom.iter().enumerate() {
                for cgto_j in atom_j {
                    let nao_j = msao(cgto_j.ang);
                    let d = [
                        coords[iat][0] - coords[jat][0],
                        coords[iat][1] - coords[jat][1],
                        coords[iat][2] - coords[jat][2],
                    ];
                    let block = overlap_cgto(cgto_i, cgto_j, &d);
                    for col in 0..nao_i {
                        for row in 0..nao_j {
                            s[(j_off + row, i_off + col)] = block[row * nao_i + col];
                        }
                    }
                    j_off += nao_j;
                }
            }
            i_off += nao_i;
        }
    }
    s
}
