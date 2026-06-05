use crate::error::{DftbError, Result};
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

/// Rotation routines matching DFTB+ `src/dftbp/dftb/sk.F90` orbital ordering:
/// p orbitals are ordered (py, pz, px).
#[derive(Debug, Clone, Copy)]
pub struct Rotation;

impl Rotation {
    /// Build the 4x4 block for sp basis (s, py, pz, px) on each atom.
    ///
    /// `sk = [ss_sigma, sp_sigma, pp_sigma, pp_pi]`.
    pub fn rotate_sp_block(sk: [f64; 4], dc: DirectionCosines) -> DMatrix<f64> {
        let (ss, sp, pp_sigma, pp_pi) = (sk[0], sk[1], sk[2], sk[3]);
        let (l, m, n) = (dc.l, dc.m, dc.n);

        let mut h = DMatrix::<f64>::zeros(4, 4);

        // s-s
        h[(0, 0)] = ss;

        // p-s block (iSh2=p, iSh1=s → ang2=1, ang1=0 → ang1<=ang2 → direct)
        // sp() fills tmpH[py,pz,px][s] = [m,n,l]*sk
        h[(1, 0)] = m * sp; // py_j - s_i
        h[(2, 0)] = n * sp; // pz_j - s_i
        h[(3, 0)] = l * sp; // px_j - s_i

        // s-p block (iSh2=s, iSh1=p → ang2=0, ang1=1 → ang1>ang2 → (-1)^(1+0)*transpose)
        // = -1 * [m,n,l]*sk transposed → s_j row, p_i cols
        h[(0, 1)] = -h[(1, 0)]; // s_j - py_i
        h[(0, 2)] = -h[(2, 0)]; // s_j - pz_i
        h[(0, 3)] = -h[(3, 0)]; // s_j - px_i

        // p-p (DFTB+ pp(): sk(1)=sigma, sk(2)=pi)
        // indices: (py,pz,px) == (1,2,3)
        let sk1 = pp_sigma;
        let sk2 = pp_pi;

        h[(1, 1)] = (1.0 - n * n - l * l) * sk1 + (n * n + l * l) * sk2;
        h[(2, 1)] = n * m * sk1 - n * m * sk2;
        h[(3, 1)] = l * m * sk1 - l * m * sk2;
        h[(1, 2)] = h[(2, 1)];
        h[(2, 2)] = n * n * sk1 + (1.0 - n * n) * sk2;
        h[(3, 2)] = n * l * sk1 - n * l * sk2;
        h[(1, 3)] = h[(3, 1)];
        h[(2, 3)] = h[(3, 2)];
        h[(3, 3)] = l * l * sk1 + (1.0 - l * l) * sk2;

        h
    }
}
