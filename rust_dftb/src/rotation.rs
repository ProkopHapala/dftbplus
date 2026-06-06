use crate::error::{DftbError, Result};
use crate::sk_data::SkData;
use nalgebra::DMatrix;

#[derive(Debug, Clone, Copy)]
pub struct DirectionCosines {
    pub l: f64,
    pub m: f64,
    pub n: f64,
}

impl DirectionCosines {
    pub fn from_vec(v: [f64; 3]) -> Result<Self> {
        let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if r2 == 0.0 {
            return Err(DftbError::Rotation("zero bond vector".into()));
        }
        let inv = 1.0 / r2.sqrt();
        Ok(Self {
            l: v[0] * inv,
            m: v[1] * inv,
            n: v[2] * inv,
        })
    }
}

/// Rotation routines matching DFTB+ `src/dftbp/dftb/sk.F90`.
/// p orbitals are ordered (py, pz, px).
pub struct Rotation;

impl Rotation {
    /// Rotate a single shell pair (ang1, ang2) with SK integrals h_sk, s_sk.
    /// Returns (h_sub, s_sub) where sub-matrix has:
    ///   rows = 2*ang2+1 (orbitals of shell 2, atom j)
    ///   cols = 2*ang1+1 (orbitals of shell 1, atom i)
    ///
    /// This matches Fortran rotateH0 with ang1 <= ang2 (direct, no transpose/sign).
    pub fn rotate_shell_pair(ang1: i32, ang2: i32, h_sk: &[f64], s_sk: &[f64], dc: DirectionCosines)
        -> Result<(DMatrix<f64>, DMatrix<f64>)> {
        let h = Self::rotate_shell_pair_single(ang1, ang2, h_sk, dc)?;
        let s = Self::rotate_shell_pair_single(ang1, ang2, s_sk, dc)?;
        Ok((h, s))
    }

    fn rotate_shell_pair_single(ang1: i32, ang2: i32, sk: &[f64], dc: DirectionCosines)
        -> Result<DMatrix<f64>> {
        match (ang1, ang2) {
            (0, 0) => Ok(Self::rotate_ss(sk[0])),
            (0, 1) => Ok(Self::rotate_sp(dc, sk[0])),
            (1, 0) => Ok(Self::rotate_sp(dc, sk[0])),
            (1, 1) => Ok(Self::rotate_pp(dc, sk[0], sk[1])),
            _ => Err(DftbError::Rotation(format!(
                "unsupported shell pair ({}, {})", ang1, ang2))),
        }
    }

    /// s-s: 1x1 block
    fn rotate_ss(ss: f64) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, 1, &[ss])
    }

    /// s-p (or p-s) rotation: Fortran sp() fills tmpH[py,pz,px][s] = [m,n,l]*sk.
    /// Returns 3x1 matrix (p-rows, s-cols).
    fn rotate_sp(dc: DirectionCosines, sp: f64) -> DMatrix<f64> {
        let (l, m, n) = (dc.l, dc.m, dc.n);
        DMatrix::from_row_slice(3, 1, &[m * sp, n * sp, l * sp])
    }

    /// p-p rotation: Fortran pp() with sk(1)=sigma, sk(2)=pi.
    /// Returns 3x3 matrix with (py,pz,px) ordering.
    fn rotate_pp(dc: DirectionCosines, pp_sigma: f64, pp_pi: f64) -> DMatrix<f64> {
        let (l, m, n) = (dc.l, dc.m, dc.n);
        let sk1 = pp_sigma;
        let sk2 = pp_pi;

        DMatrix::from_row_slice(3, 3, &[
            // py row
            (1.0 - n * n - l * l) * sk1 + (n * n + l * l) * sk2,
            n * m * sk1 - n * m * sk2,
            l * m * sk1 - l * m * sk2,
            // pz row
            n * m * sk1 - n * m * sk2,
            n * n * sk1 + (1.0 - n * n) * sk2,
            n * l * sk1 - n * l * sk2,
            // px row
            l * m * sk1 - l * m * sk2,
            n * l * sk1 - n * l * sk2,
            l * l * sk1 + (1.0 - l * l) * sk2,
        ])
    }

    /// Assemble the full diatomic block for species pair (sp1, sp2) at distance r.
    /// Returns (h_blk, s_blk) with dimensions:
    ///   rows = n_orb(sp2), cols = n_orb(sp1)
    ///
    /// This iterates over shells like Fortran rotateH0.
    pub fn rotate_diatomic_block(sk: &SkData, sp1: &str, sp2: &str, r: f64, dc: DirectionCosines)
        -> Result<(DMatrix<f64>, DMatrix<f64>)> {
        let ang1_list = sk.ang_shells(sp1)?;
        let ang2_list = sk.ang_shells(sp2)?;
        let n_orb1: usize = ang1_list.iter().map(|&l| (2 * l + 1) as usize).sum();
        let n_orb2: usize = ang2_list.iter().map(|&l| (2 * l + 1) as usize).sum();

        let mut h_blk = DMatrix::<f64>::zeros(n_orb2, n_orb1);
        let mut s_blk = DMatrix::<f64>::zeros(n_orb2, n_orb1);

        let mut i_col = 0;
        for &ang1 in ang1_list {
            let n_orb1_sh = (2 * ang1 + 1) as usize;
            let mut i_row = 0;
            for &ang2 in ang2_list {
                let n_orb2_sh = (2 * ang2 + 1) as usize;
                let (h_sk, s_sk) = sk.eval_shell_integrals(sp1, sp2, ang1, ang2, r)?;
                let (h_sub, s_sub) = Self::rotate_shell_pair(ang1, ang2, &h_sk, &s_sk, dc)?;

                // Fortran placement rule:
                // ang1 <= ang2: direct (tmpH rows=atom2, cols=atom1)
                // ang1 > ang2: transpose with (-1)^(ang1+ang2) sign
                if ang1 <= ang2 {
                    for a in 0..n_orb2_sh {
                        for b in 0..n_orb1_sh {
                            h_blk[(i_row + a, i_col + b)] = h_sub[(a, b)];
                            s_blk[(i_row + a, i_col + b)] = s_sub[(a, b)];
                        }
                    }
                } else {
                    let sign = if (ang1 + ang2) % 2 == 0 { 1.0 } else { -1.0 };
                    for a in 0..n_orb2_sh {
                        for b in 0..n_orb1_sh {
                            h_blk[(i_row + a, i_col + b)] = sign * h_sub[(b, a)];
                            s_blk[(i_row + a, i_col + b)] = sign * s_sub[(b, a)];
                        }
                    }
                }

                i_row += n_orb2_sh;
            }
            i_col += n_orb1_sh;
        }

        Ok((h_blk, s_blk))
    }
}
