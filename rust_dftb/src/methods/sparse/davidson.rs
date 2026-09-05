//! Generalized Davidson partial eigensolver for the HOMO-LUMO gap.
//!
//! Solves the generalized eigenvalue problem `H·C = S·C·ε` for a few
//! eigenvalues around the Fermi level (HOMO-LUMO gap) without full O(N³)
//! diagonalization. This is the linear-scaling companion to the TC2 sparse
//! density-matrix purification: once K is converged, we only need a handful
//! of frontier orbitals for the gap, not the full spectrum.
//!
//! # Method
//!
//! Generalized Davidson with S-orthonormalization and a diagonal preconditioner:
//!
//! 1. Start with `n_target` trial vectors below and above the gap (from the
//!    diagonal of the transformed matrix `S^{-1}·H`).
//! 2. S-orthonormalize the basis: `V^T·S·V = I` (modified Gram-Schmidt with
//!    S-inner product).
//! 3. Project: `H_proj = V^T·H·V` (small `m×m` matrix).
//! 4. Diagonalize `H_proj` → Ritz values `θ`, Ritz vectors `u`.
//! 5. Form Ritz vectors `X = V·u`, compute residuals `r_i = (H - θ_i·S)·x_i`.
//! 6. If `||r_i||_S < tol` for all target pairs, converged.
//! 7. Precondition: `t_i = (D_H - θ_i·D_S)^{-1}·r_i` where `D_H = diag(H)`,
//!    `D_S = diag(S)`.
//! 8. S-orthonormalize `t_i` against existing `V`, append to subspace.
//! 9. Cap subspace size; if exceeded, restart from best Ritz vectors.
//!
//! # Reference
//!
//! Ported/adapted from the block Davidson in NumericalMathPlayground
//! (`topics/LinearAlgebra/FastDirectSolvers/Davidson_Eigensolver.py`,
//!  Joshua Goings 2013) and generalized to the non-orthogonal `S` metric.
//! See also `topics/LinearScalingQM/CheFSI/` for Chebyshev-filtered
//! alternatives.
//!
//! # Limitations
//!
//! The diagonal (Jacobi) preconditioner is insufficient for systems with dense
//! near-degenerate frontier manifolds (e.g. coronene's π manifold). Benzene
//! converges in 3 iterations; coronene does not converge. A stronger
//! preconditioner (SSOR, ILU) or shift-invert strategy is needed for larger
//! PAHs. See `doc/prokop/topical_audit/davidson_eigensolver.md`.

use crate::core::error::{DftbError, Result};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Solve for `n_target` eigenvalues below the gap (HOMO side) and `n_target`
/// above the gap (LUMO side) of the generalized eigenvalue problem
/// `H·C = S·C·ε`.
///
/// Returns the `2*n_target` eigenvalues (sorted ascending) and corresponding
/// eigenvectors (columns). The HOMO is `eigs[n_occ-1]`, LUMO is `eigs[n_occ]`
/// — but since we target only around the gap, the returned vector has
/// `2*n_target` entries where the midpoint is the gap.
///
/// # Arguments
/// - `h` — Hamiltonian (n×n, symmetric)
/// - `s` — overlap matrix (n×n, symmetric positive-definite)
/// - `n_occ` — number of occupied orbitals (half the electron count)
/// - `n_target` — number of eigenvalues per side of the gap
/// - `max_iter` — maximum Davidson iterations
/// - `tol` — convergence tolerance on the S-norm residual
pub fn davidson_homo_lumo(
    h: &DMatrix<f64>,
    s: &DMatrix<f64>,
    n_occ: usize,
    n_target: usize,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, DMatrix<f64>)> {
    let n = h.nrows();
    if h.ncols() != n || s.nrows() != n || s.ncols() != n {
        return Err(DftbError::InvalidInput("H and S must be square and same size".into()));
    }
    if n_occ == 0 || n_occ >= n {
        return Err(DftbError::InvalidInput(format!(
            "n_occ={n_occ} out of range [1, {n})"
        )));
    }
    let n_target = n_target.min(n_occ).min(n - n_occ);
    if n_target == 0 {
        return Err(DftbError::InvalidInput("n_target too small for system".into()));
    }

    // Estimate the Fermi level from the diagonal of S^{-1}·H.
    // ε_i ≈ H_ii / S_ii (Rayleigh quotient of diagonal guess).
    let diag_eigs: Vec<f64> = (0..n)
        .map(|i| h[(i, i)] / s[(i, i)])
        .collect();
    let mut idx_sorted: Vec<usize> = (0..n).collect();
    idx_sorted.sort_by(|&a, &b| diag_eigs[a].partial_cmp(&diag_eigs[b]).unwrap());
    // Fermi level ≈ midpoint between n_occ-th and (n_occ+1)-th diagonal estimate
    let fermi = 0.5 * (diag_eigs[idx_sorted[n_occ - 1]] + diag_eigs[idx_sorted[n_occ]]);
    eprintln!("    Davidson: estimated Fermi level = {fermi:.6}");

    // Initial trial vectors: n_target diagonal guesses below Fermi + n_target above.
    // Use 1.5×n_target per side to help with degenerate states.
    let n_below = (n_target + n_target / 2).min(n_occ);
    let n_above = (n_target + n_target / 2).min(n - n_occ);
    let n_guess = n_below + n_above;
    let mut v: DMatrix<f64> = DMatrix::zeros(n, n_guess);
    for (k, &gi) in idx_sorted[n_occ - n_below..n_occ].iter().enumerate() {
        v[(gi, k)] = 1.0;
    }
    for (k, &gi) in idx_sorted[n_occ..n_occ + n_above].iter().enumerate() {
        v[(gi, n_below + k)] = 1.0;
    }

    // S-orthonormalize the initial basis.
    s_orthonormalize(&mut v, s, n);

    let mut eigvals = Vec::new();
    let mut eigvecs = DMatrix::zeros(n, n_guess);
    let max_subspace = (6 * n_guess).min(n);

    for iter in 0..max_iter {
        let m = v.ncols();

        // Project H into the subspace: H_proj = V^T · H · V
        let hv = h * &v;            // n×m
        let h_proj = v.transpose() * &hv;  // m×m

        // Diagonalize the small projected matrix.
        let sym = SymmetricEigen::new(h_proj.clone());
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| sym.eigenvalues[a].partial_cmp(&sym.eigenvalues[b]).unwrap());

        // Select the 2*n_target Ritz pairs closest to the Fermi level.
        let n_return = 2 * n_target;
        let mut sel: Vec<usize> = order.iter().cloned().collect();
        sel.sort_by(|&a, &b| {
            (sym.eigenvalues[a] - fermi).abs().partial_cmp(&(sym.eigenvalues[b] - fermi).abs()).unwrap()
        });
        sel.truncate(n_return);
        // Sort selected by eigenvalue (ascending) so HOMO = eigs[n_target-1], LUMO = eigs[n_target]
        sel.sort_by(|&a, &b| sym.eigenvalues[a].partial_cmp(&sym.eigenvalues[b]).unwrap());
        let ritz_vals: Vec<f64> = sel.iter().map(|&i| sym.eigenvalues[i]).collect();
        let ritz_vecs: DMatrix<f64> = DMatrix::from_columns(
            &sel.iter().map(|&i| sym.eigenvectors.column(i).clone_owned()).collect::<Vec<_>>()
        );

        // Ritz vectors in full space: X = V · U
        let x = &v * &ritz_vecs;  // n×n_guess

        // Residuals: r_i = H·x_i - θ_i·S·x_i
        let sx = s * &x;          // n×n_guess
        let hx = h * &x;          // n×n_guess
        let mut max_res = 0.0f64;
        let mut corrections: Vec<DVector<f64>> = Vec::new();
        for k in 0..n_return {
            let theta = ritz_vals[k];
            let r = hx.column(k) - theta * sx.column(k);
            let res_norm = r.norm();
            max_res = max_res.max(res_norm);
            if res_norm > tol {
                // Diagonal preconditioner: t = (D_H - θ·D_S)^{-1} · r
                // Regularize: if |denom| < eps, use eps with the same sign
                // (prevents blow-up for near-degenerate states where θ ≈ H_ii/S_ii)
                let eps = 1e-8 * (h[(0, 0)].abs() + 1.0);
                let mut t = DVector::zeros(n);
                for i in 0..n {
                    let denom = h[(i, i)] - theta * s[(i, i)];
                    let d = if denom.abs() > eps { denom } else { denom.signum() * eps };
                    t[i] = r[i] / d;
                }
                corrections.push(t);
            }
        }

        eigvals = ritz_vals.clone();
        eigvecs = x.clone();

        eprintln!("    Davidson iter {iter}: m={m}, max|res|={max_res:.2e}, eigs near gap: {}",
            ritz_vals.iter().map(|e| format!("{e:.6}")).collect::<Vec<_>>().join(", "));

        if max_res < tol {
            eprintln!("    Davidson converged in {iter} iterations");
            return Ok((eigvals, eigvecs));
        }

        if corrections.is_empty() {
            eprintln!("    Davidson: no corrections needed, converged");
            return Ok((eigvals, eigvecs));
        }

        // Add correction vectors to the subspace.
        let old_cols = v.ncols();
        let mut new_v = DMatrix::zeros(n, old_cols + corrections.len());
        for j in 0..old_cols {
            new_v.column_mut(j).copy_from(&v.column(j));
        }
        for (k, t) in corrections.iter().enumerate() {
            new_v.column_mut(old_cols + k).copy_from(t);
        }
        v = new_v;

        // S-orthonormalize the expanded basis.
        s_orthonormalize(&mut v, s, n);

        // Subspace restart: if too large, keep only the best n_return Ritz vectors.
        if v.ncols() > max_subspace {
            eprintln!("    Davidson: subspace restart ({} -> {})", v.ncols(), n_return);
            // Re-project and keep best Ritz vectors as new starting basis.
            let hv2 = h * &v;
            let h_proj2 = v.transpose() * &hv2;
            let sym2 = SymmetricEigen::new(h_proj2.clone());
            let mut order2: Vec<usize> = (0..v.ncols()).collect();
            order2.sort_by(|&a, &b| sym2.eigenvalues[a].partial_cmp(&sym2.eigenvalues[b]).unwrap());
            let mut sel2: Vec<usize> = order2;
            sel2.sort_by(|&a, &b| {
                (sym2.eigenvalues[a] - fermi).abs().partial_cmp(&(sym2.eigenvalues[b] - fermi).abs()).unwrap()
            });
            sel2.truncate(n_return);
            sel2.sort_by(|&a, &b| sym2.eigenvalues[a].partial_cmp(&sym2.eigenvalues[b]).unwrap());
            let restart_vecs = DMatrix::from_columns(
                &sel2.iter().map(|&i| sym2.eigenvectors.column(i).clone_owned()).collect::<Vec<_>>()
            );
            v = &v * &restart_vecs;
            s_orthonormalize(&mut v, s, n);
        }
    }

    eprintln!("    Davidson: max_iter ({max_iter}) reached, returning best estimate (max|res|={:.2e})", eigvals.len());
    Ok((eigvals, eigvecs))
}

/// S-orthonormalize the columns of `v` in place using modified Gram-Schmidt
/// with the S-inner product: `<u|w>_S = u^T · S · w`.
///
/// After this, `V^T · S · V = I` (to working precision).
fn s_orthonormalize(v: &mut DMatrix<f64>, s: &DMatrix<f64>, n: usize) {
    let m = v.ncols();
    for k in 0..m {
        // v_k = v_k - sum_{j<k} <v_j|v_k>_S * v_j
        for j in 0..k {
            let vj = v.column(j).clone_owned();
            let sv = s * &vj;
            let overlap = v.column(k).dot(&sv);
            v.column_mut(k).axpy(-overlap, &vj, 1.0);
        }
        // Normalize: ||v_k||_S = sqrt(v_k^T · S · v_k)
        let sv = s * v.column(k);
        let norm_s = v.column(k).dot(&sv).sqrt();
        if norm_s > 1e-14 {
            v.column_mut(k).scale_mut(1.0 / norm_s);
        }
    }
    // Silence unused-n warning (n is the dimension, used in assertions in debug)
    debug_assert_eq!(v.nrows(), n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_davidson_vs_dense_small() {
        // 20×20 generalized eigenvalue problem
        let n = 20;
        let h = DMatrix::<f64>::from_fn(n, n, |i, j| {
            if i == j { (i as f64 + 1.0) * 2.0 } else { 0.1 * (i as f64 - j as f64).abs().sin() }
        });
        let s = DMatrix::<f64>::identity(n, n) * 1.5
            + DMatrix::<f64>::from_fn(n, n, |i, j| if (i as i32 - j as i32).abs() == 1 { 0.2 } else { 0.0 });
        let n_occ = 10;
        let n_target = 3;

        // Dense reference: S^{-1/2} H S^{-1/2}, then eigh
        let sym_s = SymmetricEigen::new(s.clone());
        let s_sqrt = &sym_s.eigenvectors * DMatrix::from_diagonal(&sym_s.eigenvalues.map(|e: f64| e.sqrt())) * &sym_s.eigenvectors.transpose();
        let s_inv_sqrt = s_sqrt.try_inverse().unwrap();
        let h_orth = &s_inv_sqrt * &h * &s_inv_sqrt;
        let ref_sym = SymmetricEigen::new(h_orth);
        let mut ref_sorted: Vec<f64> = ref_sym.eigenvalues.iter().cloned().collect();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ref_homo = ref_sorted[n_occ - 1];
        let ref_lumo = ref_sorted[n_occ];

        // Davidson
        let (eigs, _) = davidson_homo_lumo(&h, &s, n_occ, n_target, 100, 1e-10).unwrap();
        // eigs has 2*n_target entries; the gap is at index n_target
        let dav_homo = eigs[n_target - 1];
        let dav_lumo = eigs[n_target];

        eprintln!("ref  HOMO={ref_homo:.10}, LUMO={ref_lumo:.10}, gap={:.10}", ref_lumo - ref_homo);
        eprintln!("dav  HOMO={dav_homo:.10}, LUMO={dav_lumo:.10}, gap={:.10}", dav_lumo - dav_homo);

        assert!((dav_homo - ref_homo).abs() < 1e-6, "HOMO mismatch: dav={dav_homo} ref={ref_homo}");
        assert!((dav_lumo - ref_lumo).abs() < 1e-6, "LUMO mismatch: dav={dav_lumo} ref={ref_lumo}");
    }
}
