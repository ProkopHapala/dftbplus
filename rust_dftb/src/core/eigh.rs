//! Symmetric dense eigensolve for the nuclear Hessian.
//!
//! The vibration path used to call `nalgebra::SymmetricEigen`. On the
//! 4944×4944 R18 Hessian that call was ~100 s and ~72 % of the wall
//! (`Sparse_Performance.md`). This is the same `dsyevd` the CPU DFTB
//! diagonalization already uses (`fragment.rs`, `dftb_cpu.rs`).
//!
//! **Design.** One LAPACK divide-and-conquer call on the lower triangle.
//! Eigenvalues come back ascending. Eigenvectors overwrite the input,
//! column-major, one column per mode. Not for the electronic SCC loop:
//! that matrix is sparse and is purified, not diagonalized.
//!
//! **Caveats.**
//! - Workspace is O(n²) and allocated per call. Fine for one spectrum.
//!   Do not call this inside a column loop.
//! - OpenBLAS may use several threads. That is the host post-process,
//!   not the single-thread GPU-vs-CPU bar.
//! - Mode signs are arbitrary. Compare eigenvalues, or align signs,
//!   before comparing eigenvectors to another solver.

use lapack::dsyevd;
use nalgebra::{DMatrix, DVector};

use crate::core::error::{DftbError, Result};

/// Eigenvalues (ascending) and eigenvectors (columns) of a real symmetric
/// matrix. `a` is overwritten. Only the lower triangle is read.
pub fn symmetric_eigh(mut a: DMatrix<f64>) -> Result<(DVector<f64>, DMatrix<f64>)> {
    let n = a.nrows();
    if n == 0 || a.ncols() != n {
        return Err(DftbError::InvalidInput(format!(
            "symmetric_eigh: expected square n≥1, got {}×{}",
            a.nrows(),
            a.ncols()
        )));
    }
    let mut eigenvalues = vec![0.0f64; n];
    let mut work = vec![0.0f64; 1];
    let mut iwork = vec![0i32; 1];
    let mut info: i32 = 0;
    unsafe {
        dsyevd(
            b'V',
            b'L',
            n as i32,
            a.as_mut_slice(),
            n as i32,
            &mut eigenvalues,
            &mut work,
            -1,
            &mut iwork,
            -1,
            &mut info,
        );
    }
    if info != 0 {
        return Err(DftbError::InvalidInput(format!(
            "symmetric_eigh: dsyevd workspace query failed info={info} n={n}"
        )));
    }
    let lwork = work[0] as usize;
    let liwork = iwork[0] as usize;
    if lwork == 0 || liwork == 0 {
        return Err(DftbError::InvalidInput(format!(
            "symmetric_eigh: dsyevd workspace query returned lwork={lwork} liwork={liwork} n={n}"
        )));
    }
    let mut work = vec![0.0f64; lwork];
    let mut iwork = vec![0i32; liwork];
    unsafe {
        dsyevd(
            b'V',
            b'L',
            n as i32,
            a.as_mut_slice(),
            n as i32,
            &mut eigenvalues,
            &mut work,
            lwork as i32,
            &mut iwork,
            liwork as i32,
            &mut info,
        );
    }
    if info != 0 {
        return Err(DftbError::InvalidInput(format!(
            "symmetric_eigh: dsyevd failed info={info} n={n} (info>0: off-diagonal block did not converge)"
        )));
    }
    Ok((DVector::from_vec(eigenvalues), a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SymmetricEigen;
    use std::time::Instant;

    fn spd(n: usize) -> DMatrix<f64> {
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = ((i * 17 + j * 3) % 100) as f64 * 0.01 - 0.4;
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
            a[(i, i)] += 2.0 + 0.05 * i as f64;
        }
        a
    }

    #[test]
    fn symmetric_eigh_matches_diagonal() {
        let n = 8usize;
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = (i as f64) - 3.0;
        }
        let (w, v) = symmetric_eigh(a).expect("dsyevd diagonal");
        for i in 0..n {
            let expect = (i as f64) - 3.0;
            assert!(
                (w[i] - expect).abs() < 1e-12,
                "λ[{i}]={} expect={expect}",
                w[i]
            );
            let mut col_norm = 0.0f64;
            for r in 0..n {
                col_norm += v[(r, i)] * v[(r, i)];
            }
            assert!((col_norm - 1.0).abs() < 1e-12, "column {i} norm {col_norm}");
        }
    }

    #[test]
    fn symmetric_eigh_matches_nalgebra_and_is_faster() {
        let n_par = 64usize;
        let a = spd(n_par);
        let (w, _) = symmetric_eigh(a.clone()).expect("dsyevd parity");
        let mut w_ref: Vec<f64> = SymmetricEigen::new(a).eigenvalues.iter().copied().collect();
        w_ref.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut max_abs = 0.0f64;
        for i in 0..n_par {
            max_abs = max_abs.max((w[i] - w_ref[i]).abs());
        }
        eprintln!("symmetric_eigh parity n={n_par} max|Δλ|={max_abs:.3e}");
        assert!(max_abs < 1e-8, "eigenvalue mismatch max|Δλ|={max_abs:.3e} n={n_par}");

        // Default n=192 stays under a second in a debug build. A larger
        // timing (release only) may set RUST_DFTB_EIGH_N, capped at 1024
        // so this test cannot grow into the ~100 s R18 diagonalization.
        let n_t: usize = std::env::var("RUST_DFTB_EIGH_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(192);
        assert!(
            (64..=1024).contains(&n_t),
            "RUST_DFTB_EIGH_N={n_t} outside 64..=1024 (GUIDELINES §7: no run over 1 minute)"
        );
        let b = spd(n_t);
        let t0 = Instant::now();
        let (w_fast, _) = symmetric_eigh(b.clone()).expect("dsyevd timing");
        let dt_lapack = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut w_slow: Vec<f64> = SymmetricEigen::new(b).eigenvalues.iter().copied().collect();
        let dt_nalgebra = t1.elapsed().as_secs_f64();
        w_slow.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut max_abs_t = 0.0f64;
        for i in 0..n_t {
            max_abs_t = max_abs_t.max((w_fast[i] - w_slow[i]).abs());
        }
        let ratio = dt_nalgebra / dt_lapack.max(1e-9);
        eprintln!(
            "symmetric_eigh timing n={n_t} dsyevd={dt_lapack:.4}s nalgebra={dt_nalgebra:.4}s ratio={ratio:.1} max|Δλ|={max_abs_t:.3e} OPENBLAS_NUM_THREADS={:?}",
            std::env::var("OPENBLAS_NUM_THREADS").ok()
        );
        assert!(max_abs_t < 1e-6, "timing-size mismatch max|Δλ|={max_abs_t:.3e}");
        assert!(
            ratio > 3.0,
            "dsyevd was not faster: nalgebra {dt_nalgebra:.4}s / dsyevd {dt_lapack:.4}s = {ratio:.2} at n={n_t}"
        );
    }
}
