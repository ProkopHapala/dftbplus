//! Mulliken population analysis for atom-resolved charges.
//!
//! Given occupied eigenvectors `C` (shape `[n_orbs × n_mos]`) and the overlap
//! matrix `S`, the density matrix is `D = C·Cᵀ` (closed-shell, 2 e⁻/MO).
//! The charge on atom A is:
//!
//! `q_A = q0_A - Σ_{μ∈A} (D·S)_{μμ}`
//!
//! where `q0_A` is the reference neutral charge.

use nalgebra::DMatrix;

/// Compute atom-resolved Mulliken charges from the density matrix.
///
/// `dmat_s`  – the product `D · S`, shape `[n_orbs × n_orbs]`.
///             Only the *diagonal* elements are needed, but the full matrix
///             is passed for API consistency.
/// `orb_off` – cumulative orbital offset per atom, length `n_atoms + 1`.
/// `q0`      – reference neutral charge per atom, length `n_atoms`.
/// `out`     – output population per atom, length `n_atoms`.
pub fn mulliken_charges_from_dmat_s(
    dmat_s: &DMatrix<f64>,
    orb_off: &[u16],
    q0: &[f64],
    out: &mut [f64],
) {
    let n_atoms = q0.len();
    assert_eq!(orb_off.len(), n_atoms + 1);
    assert_eq!(out.len(), n_atoms);

    for i_at in 0..n_atoms {
        let i0 = orb_off[i_at] as usize;
        let i1 = orb_off[i_at + 1] as usize;
        let mut pop = 0.0;
        for mu in i0..i1 {
            pop += dmat_s[(mu, mu)];
        }
        out[i_at] = q0[i_at] - pop;
    }
}

/// Build `D·S` directly from occupied eigenvectors without forming `D`.
///
/// `c_occ` – occupied MO coefficients, shape `[n_orbs × n_occ]`.
/// `s`     – overlap matrix, shape `[n_orbs × n_orbs]`.
/// `work`  – pre-allocated scratch matrix `[n_orbs × n_orbs]`.
///
/// Computation: `work = (C_occ · C_occᵀ) · S = C_occ · (C_occᵀ · S)`.
/// For closed-shell: multiply by 2.0 after the trace.
pub fn build_dmat_s(
    c_occ: &DMatrix<f64>,
    s: &DMatrix<f64>,
    work: &mut DMatrix<f64>,
) {
    let n_orbs = c_occ.nrows();
    let n_occ = c_occ.ncols();

    // work = C_occ · C_occᵀ  (n_orbs × n_orbs)
    work.fill(0.0);
    for k in 0..n_occ {
        for i in 0..n_orbs {
            let c_ik = c_occ[(i, k)];
            for j in 0..n_orbs {
                work[(i, j)] += c_ik * c_occ[(j, k)];
            }
        }
    }

    // Multiply by S on the right: work = work · S
    // We do this in-place with a second scratch buffer.
    // TODO: for the skeleton we leave the in-place optimization for later;
    // the caller should pre-allocate two work matrices if needed.
    let mut tmp = DMatrix::zeros(n_orbs, n_orbs);
    for i in 0..n_orbs {
        for j in 0..n_orbs {
            let mut sum = 0.0;
            for k in 0..n_orbs {
                sum += work[(i, k)] * s[(k, j)];
            }
            tmp[(i, j)] = sum;
        }
    }
    work.copy_from(&tmp);
}

/// Convenience: compute charges directly from occupied eigenvectors.
///
/// `c_occ`   – occupied MO coefficients.
/// `s`       – overlap matrix.
/// `orb_off` – cumulative orbital offsets.
/// `q0`      – reference neutral charges.
/// `work`    – scratch matrix `[n_orbs × n_orbs]`.
/// `out`     – output charges per atom.
pub fn compute_charges(
    c_occ: &DMatrix<f64>,
    s: &DMatrix<f64>,
    orb_off: &[u16],
    q0: &[f64],
    work: &mut DMatrix<f64>,
    out: &mut [f64],
) {
    build_dmat_s(c_occ, s, work);
    // Scale by 2 for closed-shell
    work.scale_mut(2.0);
    mulliken_charges_from_dmat_s(work, orb_off, q0, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn mulliken_single_orbital() {
        // One atom, one orbital: if D·S = 1.0, charge = q0 - 1.0
        let dmat_s = DMatrix::from_row_slice(1, 1, &[1.0]);
        let orb_off = &[0u16, 1u16];
        let q0 = &[2.0];
        let mut out = vec![0.0];
        mulliken_charges_from_dmat_s(&dmat_s, orb_off, q0, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-12);
    }
}
