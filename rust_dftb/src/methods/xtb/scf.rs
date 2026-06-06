//! SCF loop and potential builder for xTB

use crate::methods::xtb::coulomb::{build_coulomb_matrix, thirdorder_potential};
use crate::methods::xtb::mulliken::{shell_charges, atomic_charges, reference_shell_occupations};
use crate::methods::xtb::hamiltonian::build_h0_s;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

/// Build SCC Hamiltonian from H0, Coulomb matrix, and shell charges
/// H_scc = H0 + gamma * q
pub fn build_scc_hamiltonian(
    h0: &DMatrix<f64>,
    gamma: &DMatrix<f64>,
    shell_charges: &DVector<f64>,
) -> DMatrix<f64> {
    let nshell = h0.nrows();
    let mut h_scc = h0.clone();

    // Add Coulomb contribution: V_scc = gamma * q
    for i in 0..nshell {
        for j in 0..nshell {
            h_scc[(i, j)] += gamma[(i, j)] * shell_charges[j];
        }
    }

    h_scc
}

/// Build full SCC Hamiltonian including third-order correction.
/// Follows tblite scf/potential.f90 add_vao_to_h1 exactly:
///   H_scc[i,j] = H0[i,j] - S[i,j] * 0.5 * (vao[i] + vao[j])
/// where vao[i] = sum_jsh gamma(sh(i), jsh) * qsh(jsh)  +  v3[sh(i)]
pub fn build_scc_hamiltonian_with_thirdorder(
    h0: &DMatrix<f64>,
    s: &DMatrix<f64>,
    gamma: &DMatrix<f64>,
    shell_charges: &DVector<f64>,
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ao2sh: &[usize],
) -> DMatrix<f64> {
    let nao = h0.nrows();
    let nshell = shell_charges.len();
    let mut h_scc = h0.clone();

    // Shell-resolved potential: vsh = gamma * qsh
    let mut vsh: DVector<f64> = DVector::zeros(nshell);
    for ish in 0..nshell {
        for jsh in 0..nshell {
            vsh[ish] += gamma[(ish, jsh)] * shell_charges[jsh];
        }
    }

    // Add third-order onsite correction to shell potential
    let v3 = thirdorder_potential(shell_charges, nshell_per_atom, elem_idx);
    for ish in 0..nshell {
        vsh[ish] += v3[ish];
    }

    // Expand shell potential to AO basis
    let mut vao = vec![0.0f64; nao];
    for iao in 0..nao {
        vao[iao] = vsh[ao2sh[iao]];
    }

    // Apply to all H matrix elements (tblite sign convention: subtract)
    for iao in 0..nao {
        for jao in 0..nao {
            h_scc[(jao, iao)] -= s[(jao, iao)] * 0.5 * (vao[jao] + vao[iao]);
        }
    }

    h_scc
}

/// Simple SCF loop for GFN1-xTB
/// Returns converged density matrix and shell charges
pub fn run_scf(
    coords: &[[f64; 3]],
    elem_idx: &[usize],
    n_electrons: usize,
    max_iter: usize,
    conv_thresh: f64,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    // Build basis info
    let nshell_per_atom: Vec<usize> = elem_idx.iter().map(|&z| {
        match z {
            0 => 2, // H: 1s, 2s
            7 => 2, // O: 2s, 2p
            _ => 2, // Default: 2 shells
        }
    }).collect();
    let nshell_total: usize = nshell_per_atom.iter().sum();
    let ang_per_shell: Vec<usize> = elem_idx.iter().flat_map(|&z| {
        match z {
            0 => vec![0, 0], // H: s, s
            7 => vec![0, 1], // O: s, p
            _ => vec![0, 1], // Default: s, p
        }
    }).collect();

    // Build H0 and S
    let (h0, s, _shell_elem, _shell_idx) = build_h0_s(coords, elem_idx);
    let nao = s.nrows();

    // Build Coulomb matrix
    let gamma = build_coulomb_matrix(coords, &nshell_per_atom, elem_idx, &ang_per_shell);

    // Reference occupations
    let n0sh = reference_shell_occupations(&nshell_per_atom, elem_idx, &ang_per_shell);

    // Build ao2sh mapping (s: 1 AO, p: 3 AOs)
    let mut ao2sh = vec![0usize; nao];
    let mut ao_offset = 0;
    let mut shell_offset = 0;
    for iat in 0..elem_idx.len() {
        let nsh = nshell_per_atom[iat];
        for ish in 0..nsh {
            let l = ang_per_shell[shell_offset + ish];
            let nao_shell = match l {
                0 => 1, // s
                1 => 3, // p
                _ => 1,
            };
            for i in 0..nao_shell {
                ao2sh[ao_offset + i] = shell_offset + ish;
            }
            ao_offset += nao_shell;
        }
        shell_offset += nsh;
    }

    // Simple linear mixer with fixed mixing parameter
    const MIXING: f64 = 0.3;

    // Initial guess: neutral atoms (qsh = 0 => density = reference)
    let mut density = DMatrix::zeros(nao, nao);
    let mut qsh = DVector::zeros(nshell_total);

    for iter in 0..max_iter {
        // Build SCC Hamiltonian
        let h_scc = build_scc_hamiltonian_with_thirdorder(
            &h0, &s, &gamma, &qsh, &nshell_per_atom, elem_idx, &ao2sh
        );

        // Solve generalized eigenvalue problem: H*c = E*S*c
        let (eigenvalues, eigenvectors) = solve_gevp(&h_scc, &s);

        // Build new density from occupied orbitals (closed-shell):
        // P = 2 * sum_i c_i * c_i^T  (doubly occupied)
        let n_occ = n_electrons / 2;
        let mut new_density = DMatrix::zeros(nao, nao);
        for i in 0..n_occ {
            let c = eigenvectors.column(i);
            new_density += 2.0 * c * c.transpose();
        }

        // Compute new charges
        let new_qsh = shell_charges(&new_density, &s, &ao2sh, &n0sh);

        // Check convergence
        let max_diff = (&new_qsh - &qsh).abs().max();
        if iter % 10 == 0 {
            println!("Iter {}: max charge diff = {:.6e}", iter, max_diff);
        }
        if max_diff < conv_thresh {
            println!("SCF converged in {} iterations", iter + 1);
            return (new_density, new_qsh, eigenvalues);
        }

        // Simple linear mixing
        qsh = MIXING * &new_qsh + (1.0 - MIXING) * &qsh;
        density = MIXING * &new_density + (1.0 - MIXING) * &density;
    }

    println!("SCF did not converge in {} iterations", max_iter);
    (density, qsh, DVector::zeros(nao))
}

/// Solve generalized eigenvalue problem H*c = E*S*c
/// Returns (eigenvalues, eigenvectors)
fn solve_gevp(h: &DMatrix<f64>, s: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
    // Use Cholesky decomposition of S: S = L*L^T
    let chol = s.clone().cholesky();
    if chol.is_none() {
        eprintln!("GEVP: S is not positive definite, returning zeros");
        let n = h.nrows();
        return (DVector::zeros(n), DMatrix::zeros(n, n));
    }
    let l = chol.unwrap().l();

    // Transform H': H' = L^{-1} * H * (L^T)^{-1}
    let l_inv = l.clone().try_inverse().expect("L should be invertible");
    let l_t_inv = l_inv.transpose();
    let h_prime = &l_inv * h * &l_t_inv;

    // Solve standard eigenvalue problem: H' * y = E * y
    let eigen = h_prime.symmetric_eigen();
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // Back-transform: c = (L^T)^{-1} * y
    let c = l_t_inv * eigenvectors;

    // Sort eigenvalues and eigenvectors in ascending order
    let mut indexed_eigenvalues: Vec<(usize, f64)> = eigenvalues.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_eigenvalues.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut sorted_eigenvalues = DVector::zeros(eigenvalues.len());
    let mut sorted_eigenvectors = DMatrix::zeros(c.nrows(), c.ncols());
    for (new_i, (old_i, _)) in indexed_eigenvalues.iter().enumerate() {
        sorted_eigenvalues[new_i] = eigenvalues[*old_i];
        sorted_eigenvectors.set_column(new_i, &c.column(*old_i));
    }

    (sorted_eigenvalues, sorted_eigenvectors)
}
