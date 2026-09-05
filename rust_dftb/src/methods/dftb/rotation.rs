use crate::core::error::{DftbError, Result};
use crate::methods::dftb::sk_data::SkTableSp;
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
///
/// ORBITAL ORDERING (NON-OBVIOUS):
/// DFTB+ uses tesseral/real spherical harmonics ordered by magnetic quantum number m:
///   l=0 (s):  [s]
///   l=1 (p):  [py (m=-1), pz (m=0), px (m=+1)]   ← NOT px,py,pz!
///   l=2 (d):  [xy, yz, z2, xz, x2-y2]
///
/// SHELL ITERATION CONVENTION (from Fortran rotateH0):
///   Outer loop: shells of atom i (species sp1) → matrix COLUMNS
///   Inner loop: shells of atom j (species sp2) → matrix ROWS
///   i_col increments by n_orb(sp1_shell), i_row increments by n_orb(sp2_shell)
///
/// TRANSPOSE/SIGN RULE (from Fortran rotateH0):
///   If ang1 <= ang2: direct placement (tmpH rows=atomJ, cols=atomI)
///   If ang1 >  ang2: transpose(tmpH) * (-1)^(ang1+ang2)
///   Example: sp block (ang1=0, ang2=1) → direct
///            ps block (ang1=1, ang2=0) → transpose * (-1)^1 = -transpose
///   This is needed because old-format SK files assume lMax >= lMin.
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

    /// In-place shell rotation for both H and S.
    /// `out_h` / `out_s` must each be at least `(2*ang2+1)*(2*ang1+1)` elements.
    pub fn rotate_shell_pair_into(
        ang1: i32,
        ang2: i32,
        h_sk: &[f64],
        s_sk: &[f64],
        dc: DirectionCosines,
        out_h: &mut [f64],
        out_s: &mut [f64],
    ) -> Result<()> {
        Self::rotate_shell_pair_single_into(ang1, ang2, h_sk, dc, out_h)?;
        Self::rotate_shell_pair_single_into(ang1, ang2, s_sk, dc, out_s)?;
        Ok(())
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

    fn rotate_shell_pair_single_into(
        ang1: i32,
        ang2: i32,
        sk: &[f64],
        dc: DirectionCosines,
        out: &mut [f64],
    ) -> Result<()> {
        match (ang1, ang2) {
            (0, 0) => { out[0] = sk[0]; Ok(()) }
            (0, 1) => { Self::rotate_sp_into(dc, sk[0], out); Ok(()) }
            (1, 0) => { Self::rotate_sp_into(dc, sk[0], out); Ok(()) }
            (1, 1) => { Self::rotate_pp_into(dc, sk[0], sk[1], out); Ok(()) }
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

    fn rotate_sp_into(dc: DirectionCosines, sp: f64, out: &mut [f64]) {
        let (l, m, n) = (dc.l, dc.m, dc.n);
        out[0] = m * sp;
        out[1] = n * sp;
        out[2] = l * sp;
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

    fn rotate_pp_into(dc: DirectionCosines, pp_sigma: f64, pp_pi: f64, out: &mut [f64]) {
        let (l, m, n) = (dc.l, dc.m, dc.n);
        let sk1 = pp_sigma;
        let sk2 = pp_pi;
        // Row-major 3x3, (py,pz,px) ordering
        out[0] = (1.0 - n * n - l * l) * sk1 + (n * n + l * l) * sk2; // py,py
        out[1] = n * m * sk1 - n * m * sk2;                           // py,pz
        out[2] = l * m * sk1 - l * m * sk2;                           // py,px
        out[3] = n * m * sk1 - n * m * sk2;                           // pz,py
        out[4] = n * n * sk1 + (1.0 - n * n) * sk2;                   // pz,pz
        out[5] = n * l * sk1 - n * l * sk2;                           // pz,px
        out[6] = l * m * sk1 - l * m * sk2;                           // px,py
        out[7] = n * l * sk1 - n * l * sk2;                           // px,pz
        out[8] = l * l * sk1 + (1.0 - l * l) * sk2;                   // px,px
    }

    /// Assemble the full diatomic block for a species pair at distance r.
    ///
    /// `tab_fwd` is the SK table for the (sp1, sp2) species pair.
    /// `tab_rev` is the SK table for the (sp2, sp1) species pair (may be same as tab_fwd).
    /// `ang1_list` / `ang2_list` are the angular momentum shells for species 1 / 2.
    pub fn rotate_diatomic_block(
        tab_fwd: &SkTableSp,
        tab_rev: &SkTableSp,
        ang1_list: &[i32],
        ang2_list: &[i32],
        r: f64,
        dc: DirectionCosines,
    ) -> Result<(DMatrix<f64>, DMatrix<f64>)> {
        let n_orb1: usize = ang1_list.iter().map(|&l| (2 * l + 1) as usize).sum();
        let n_orb2: usize = ang2_list.iter().map(|&l| (2 * l + 1) as usize).sum();
        let mut h_blk = DMatrix::<f64>::zeros(n_orb2, n_orb1);
        let mut s_blk = DMatrix::<f64>::zeros(n_orb2, n_orb1);
        Self::rotate_diatomic_block_into(
            tab_fwd, tab_rev, ang1_list, ang2_list, r, dc,
            h_blk.as_mut_slice(), s_blk.as_mut_slice(),
        )?;
        Ok((h_blk, s_blk))
    }

    /// In-place version: writes directly into flat output slices.
    /// `out_h` and `out_s` must each have length >= n_orb2 * n_orb1.
    pub fn rotate_diatomic_block_into(
        tab_fwd: &SkTableSp,
        tab_rev: &SkTableSp,
        ang1_list: &[i32],
        ang2_list: &[i32],
        r: f64,
        dc: DirectionCosines,
        out_h: &mut [f64],
        out_s: &mut [f64],
    ) -> Result<()> {
        let n_orb1: usize = ang1_list.iter().map(|&l| (2 * l + 1) as usize).sum();
        let n_orb2: usize = ang2_list.iter().map(|&l| (2 * l + 1) as usize).sum();
        let block_size = n_orb2 * n_orb1;
        assert!(out_h.len() >= block_size, "out_h too small");
        assert!(out_s.len() >= block_size, "out_s too small");

        // Reusable stack buffers for shell integrals (max 4 for spd)
        let mut sk_h = [0.0f64; 4];
        let mut sk_s = [0.0f64; 4];
        // Max shell-pair matrix = 3x3 = 9 elements
        let mut sub_h = [0.0f64; 9];
        let mut sub_s = [0.0f64; 9];

        let mut i_col = 0;
        for &ang1 in ang1_list {
            let n_orb1_sh = (2 * ang1 + 1) as usize;
            let mut i_row = 0;
            for &ang2 in ang2_list {
                let n_orb2_sh = (2 * ang2 + 1) as usize;
                let tab = if ang1 <= ang2 { tab_fwd } else { tab_rev };

                // Zero-allocation SK evaluation (writes into stack buffers)
                let n_mm = tab.eval_shell_integrals_into(ang1, ang2, r, &mut sk_h, &mut sk_s)?;
                let sub_size = n_orb2_sh * n_orb1_sh;
                Self::rotate_shell_pair_into(
                    ang1, ang2, &sk_h[..n_mm], &sk_s[..n_mm], dc,
                    &mut sub_h[..sub_size], &mut sub_s[..sub_size],
                )?;

                if ang1 <= ang2 {
                    for a in 0..n_orb2_sh {
                        for b in 0..n_orb1_sh {
                            let idx = (i_row + a) * n_orb1 + (i_col + b);
                            out_h[idx] = sub_h[a * n_orb1_sh + b];
                            out_s[idx] = sub_s[a * n_orb1_sh + b];
                        }
                    }
                } else {
                    let sign = if (ang1 + ang2) % 2 == 0 { 1.0 } else { -1.0 };
                    for a in 0..n_orb2_sh {
                        for b in 0..n_orb1_sh {
                            let idx = (i_row + a) * n_orb1 + (i_col + b);
                            out_h[idx] = sign * sub_h[b * n_orb2_sh + a];
                            out_s[idx] = sign * sub_s[b * n_orb2_sh + a];
                        }
                    }
                }

                i_row += n_orb2_sh;
            }
            i_col += n_orb1_sh;
        }

        Ok(())
    }

    /// Assemble the full diatomic block AND its Cartesian derivatives for a species pair.
    /// P8: analytic derivatives — replaces 6 finite-difference block evaluations.
    ///
    /// Outputs (all row-major, size n_orb2*n_orb1):
    ///   out_h, out_s: H and S blocks
    ///   dh_dx, dh_dy, dh_z: dH/dR_a for a=x,y,z (derivative w.r.t. atom j position)
    ///   ds_dx, ds_dy, ds_z: dS/dR_a for a=x,y,z
    ///
    /// Derivative formulas (GPT 5.6 §10):
    ///   u = R_vec/R, u_a = R_a/R
    ///   dR/dR_a = u_a
    ///   du_i/dR_a = (δ_ia - u_i*u_a) / R
    ///
    ///   ss: H = V(r), dH/dR_a = V'(r) * u_a
    ///   sp: H_i = u_i * V(r), dH_i/dR_a = (δ_ia - u_i*u_a)/R * V + u_i * V' * u_a
    ///   pp: H_ij = V_π*δ_ij + (V_σ-V_π)*u_i*u_j
    ///       dH_ij/dR_a = V'_π*u_a*δ_ij + ΔV'*u_a*u_i*u_j
    ///                  + ΔV/R * [(δ_ia-u_i*u_a)*u_j + u_i*(δ_ja-u_j*u_a)]
    pub fn rotate_block_with_derivs_into(
        tab_fwd: &SkTableSp,
        tab_rev: &SkTableSp,
        ang1_list: &[i32],
        ang2_list: &[i32],
        r: f64,
        dc: DirectionCosines,
        out_h: &mut [f64], out_s: &mut [f64],
        dh_dx: &mut [f64], dh_dy: &mut [f64], dh_dz: &mut [f64],
        ds_dx: &mut [f64], ds_dy: &mut [f64], ds_dz: &mut [f64],
    ) -> Result<()> {
        let n_orb1: usize = ang1_list.iter().map(|&l| (2 * l + 1) as usize).sum();
        let n_orb2: usize = ang2_list.iter().map(|&l| (2 * l + 1) as usize).sum();
        let block_size = n_orb2 * n_orb1;
        assert!(out_h.len() >= block_size);
        for buf in [&mut *dh_dx, &mut *dh_dy, &mut *dh_dz, &mut *ds_dx, &mut *ds_dy, &mut *ds_dz] {
            assert!(buf.len() >= block_size);
            buf[..block_size].fill(0.0);
        }
        out_h[..block_size].fill(0.0);
        out_s[..block_size].fill(0.0);

        let (l, m, n) = (dc.l, dc.m, dc.n);
        let inv_r = if r > 1e-12 { 1.0 / r } else { 0.0 };
        // Direction cosines u = (l, m, n) = R_vec/R
        // u_x = l, u_y = m, u_z = n
        // du_i/dR_a = (δ_ia - u_i*u_a) / R

        let mut sk_h = [0.0f64; 4]; let mut sk_s = [0.0f64; 4];
        let mut dh_dr = [0.0f64; 4]; let mut ds_dr = [0.0f64; 4];

        let mut i_col = 0;
        for &ang1 in ang1_list {
            let n1 = (2 * ang1 + 1) as usize;
            let mut i_row = 0;
            for &ang2 in ang2_list {
                let n2 = (2 * ang2 + 1) as usize;
                let tab = if ang1 <= ang2 { tab_fwd } else { tab_rev };
                let n_mm = tab.eval_shell_integrals_and_derivs_into(
                    ang1, ang2, r, &mut sk_h, &mut sk_s, &mut dh_dr, &mut ds_dr,
                )?;
                let sub = n2 * n1;

                // Compute sub-block values and derivatives
                let (sh, sdh_x, sdh_y, sdh_z) = Self::shell_pair_with_derivs(
                    ang1, ang2, &sk_h[..n_mm], &dh_dr[..n_mm], dc, r,
                );
                let (ss_, sds_x, sds_y, sds_z) = Self::shell_pair_with_derivs(
                    ang1, ang2, &sk_s[..n_mm], &ds_dr[..n_mm], dc, r,
                );

                if ang1 <= ang2 {
                    for a in 0..n2 {
                        for b in 0..n1 {
                            let idx = (i_row + a) * n_orb1 + (i_col + b);
                            let sidx = a * n1 + b;
                            out_h[idx] = sh[sidx];
                            out_s[idx] = ss_[sidx];
                            dh_dx[idx] = sdh_x[sidx]; dh_dy[idx] = sdh_y[sidx]; dh_dz[idx] = sdh_z[sidx];
                            ds_dx[idx] = sds_x[sidx]; ds_dy[idx] = sds_y[sidx]; ds_dz[idx] = sds_z[sidx];
                        }
                    }
                } else {
                    let sign = if (ang1 + ang2) % 2 == 0 { 1.0 } else { -1.0 };
                    for a in 0..n2 {
                        for b in 0..n1 {
                            let idx = (i_row + a) * n_orb1 + (i_col + b);
                            let sidx = b * n2 + a; // transpose
                            out_h[idx] = sign * sh[sidx];
                            out_s[idx] = sign * ss_[sidx];
                            dh_dx[idx] = sign * sdh_x[sidx]; dh_dy[idx] = sign * sdh_y[sidx]; dh_dz[idx] = sign * sdh_z[sidx];
                            ds_dx[idx] = sign * sds_x[sidx]; ds_dy[idx] = sign * sds_y[sidx]; ds_dz[idx] = sign * sds_z[sidx];
                        }
                    }
                }
                i_row += n2;
            }
            i_col += n1;
        }
        Ok(())
    }

    /// Compute shell-pair block and Cartesian derivatives.
    /// Returns (values[9], dh_dx[9], dh_dy[9], dh_dz[9]) — only first sub elements are valid.
    /// Derivative is w.r.t. atom j position (R_vec = r_j - r_i).
    fn shell_pair_with_derivs(
        ang1: i32, ang2: i32,
        sk: &[f64], dsk_dr: &[f64],
        dc: DirectionCosines, r: f64,
    ) -> ([f64; 9], [f64; 9], [f64; 9], [f64; 9]) {
        let (l, m, nn) = (dc.l, dc.m, dc.n);
        let inv_r = if r > 1e-12 { 1.0 / r } else { 0.0 };
        let mut val = [0.0f64; 9];
        let mut dx = [0.0f64; 9];
        let mut dy = [0.0f64; 9];
        let mut dz = [0.0f64; 9];

        // u = (l, m, n) = R_vec/R. Component mapping: u_x=l, u_y=m, u_z=n
        // du_i/dR_a = (δ_ia - u_i*u_a) / R
        // For a=x: du_x/dR_x = (1-l²)/R, du_y/dR_x = -m*l/R, du_z/dR_x = -n*l/R
        // For a=y: du_x/dR_y = -l*m/R, du_y/dR_y = (1-m²)/R, du_z/dR_y = -n*m/R
        // For a=z: du_x/dR_z = -l*n/R, du_y/dR_z = -m*n/R, du_z/dR_z = (1-n²)/R

        match (ang1, ang2) {
            (0, 0) => {
                // H = V(r), dH/dR_a = V'(r) * u_a
                val[0] = sk[0];
                dx[0] = dsk_dr[0] * l;
                dy[0] = dsk_dr[0] * m;
                dz[0] = dsk_dr[0] * nn;
            }
            (0, 1) | (1, 0) => {
                // H_i = u_i * V(r)  (i = y,z,x → indices 0,1,2)
                // dH_i/dR_a = du_i/dR_a * V + u_i * V' * u_a
                let v = sk[0]; let vp = dsk_dr[0];
                // i=y (idx 0): u_y = m
                val[0] = m * v;
                dx[0] = (-m*l*inv_r) * v + m * vp * l;
                dy[0] = ((1.0-m*m)*inv_r) * v + m * vp * m;
                dz[0] = (-m*nn*inv_r) * v + m * vp * nn;
                // i=z (idx 1): u_z = n
                val[1] = nn * v;
                dx[1] = (-nn*l*inv_r) * v + nn * vp * l;
                dy[1] = (-nn*m*inv_r) * v + nn * vp * m;
                dz[1] = ((1.0-nn*nn)*inv_r) * v + nn * vp * nn;
                // i=x (idx 2): u_x = l
                val[2] = l * v;
                dx[2] = ((1.0-l*l)*inv_r) * v + l * vp * l;
                dy[2] = (-l*m*inv_r) * v + l * vp * m;
                dz[2] = (-l*nn*inv_r) * v + l * vp * nn;
            }
            (1, 1) => {
                // H_ij = V_π*δ_ij + (V_σ-V_π)*u_i*u_j
                // sk[0] = V_σ, sk[1] = V_π
                let vs = sk[0]; let vp = sk[1];
                let dvs = dsk_dr[0]; let dvp = dsk_dr[1];
                let dv = vs - vp; let dvp_total = dvs - dvp;
                let ui = [m, nn, l]; // i=y,z,x (indices 0,1,2)
                let ua = [l, m, nn]; // a=x,y,z
                // Orbital i maps to direction a via: i=0(y)→a=1(y), i=1(z)→a=2(z), i=2(x)→a=0(x)
                // So delta_ia = 1 when a == (i+1)%3
                for i in 0..3 {
                    for j in 0..3 {
                        let idx = i * 3 + j;
                        let delta_ij = if i == j { 1.0 } else { 0.0 };
                        val[idx] = vp * delta_ij + dv * ui[i] * ui[j];
                        for a in 0..3 {
                            let ua_a = ua[a];
                            let delta_ia = if a == (i + 1) % 3 { 1.0 } else { 0.0 };
                            let delta_ja = if a == (j + 1) % 3 { 1.0 } else { 0.0 };
                            let dval = dvp * ua_a * delta_ij
                                + dvp_total * ua_a * ui[i] * ui[j]
                                + dv * inv_r * ((delta_ia - ui[i]*ua_a) * ui[j] + ui[i] * (delta_ja - ui[j]*ua_a));
                            match a {
                                0 => dx[idx] = dval,
                                1 => dy[idx] = dval,
                                _ => dz[idx] = dval,
                            }
                        }
                    }
                }
            }
            _ => panic!("unsupported shell pair ({},{})", ang1, ang2),
        }
        (val, dx, dy, dz)
    }
}
