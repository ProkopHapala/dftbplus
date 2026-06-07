//! Multipole integral computation for GFN2-xTB
//!
//! Reimplements tblite's integral/multipole.f90 and basis/slater.f90
//! for computing dipole and quadrupole integrals from CGTO basis functions.

use crate::methods::xtb::params_gfn2;
use std::f64::consts::PI;

const TOP: f64 = 2.0 / PI; // 2/pi
const SQRTPI: f64 = 1.7724538509055160272981674833411451827975494561224;
const SQRTPI3: f64 = SQRTPI * SQRTPI * SQRTPI;

/// Double factorial for normalization (OEIS A001147)
const DFACTORIAL: [f64; 8] = [
    1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0,
];

/// CGTO parameters for a single shell
#[derive(Clone, Debug)]
pub struct Cgto {
    pub ang: usize,       // angular momentum (0=s, 1=p, 2=d)
    pub nprim: usize,      // number of primitive Gaussians
    pub alpha: Vec<f64>,   // exponents
    pub coeff: Vec<f64>,  // contraction coefficients (with normalization)
}

/// 1D Gaussian overlap integral: <x^moment | exp(-alpha*x^2)>
fn overlap_1d(moment: usize, alpha: f64) -> f64 {
    if moment % 2 != 0 {
        0.0
    } else {
        (0.5 / alpha).powi((moment / 2) as i32) * DFACTORIAL[moment / 2]
    }
}

/// Horizontal shift: move polynomial center by ae
/// cfs(x) -> cfs(x+ae), where cfs is a polynomial of degree l
fn horizontal_shift(ae: f64, l: usize, cfs: &mut [f64]) {
    match l {
        0 => {} // s: no shift
        1 => {
            // p: cfs[0] += ae*cfs[1]
            cfs[0] += ae * cfs[1];
        }
        2 => {
            // d: cfs[0] += ae^2*cfs[2], cfs[1] += 2*ae*cfs[2]
            cfs[0] += ae * ae * cfs[2];
            cfs[1] += 2.0 * ae * cfs[2];
        }
        _ => {
            // General case (not needed for GFN2 s,p)
            // We shouldn't reach here for GFN2
            panic!("horizontal_shift not implemented for l > 2");
        }
    }
}

/// Form product of two polynomials a(x) * b(x) = d(x)
/// a has degree la, b has degree lb, d has degree la+lb
fn form_product(a: &[f64], b: &[f64], la: usize, lb: usize, d: &mut [f64]) {
    // For GFN2: only need s(l=0) and p(l=1)
    if la == 0 && lb == 0 {
        // s*s
        d[0] = a[0] * b[0];
    } else if la == 0 && lb == 1 {
        // s*p
        d[0] = a[0] * b[0];
        d[1] = a[0] * b[1];
    } else if la == 1 && lb == 0 {
        // p*s
        d[0] = a[0] * b[0];
        d[1] = a[1] * b[0];
    } else if la == 1 && lb == 1 {
        // p*p
        d[0] = a[0] * b[0];
        d[1] = a[0] * b[1] + a[1] * b[0];
        d[2] = a[1] * b[1];
    } else {
        panic!("form_product not implemented for la={}, lb={}", la, lb);
    }
}

/// Compute 3D multipole integrals for one primitive Gaussian pair
/// Returns (overlap, dipole[3], quadrupole[6])
fn multipole_3d(
    rpj: &[f64; 3],
    rpi: &[f64; 3],
    _aj: f64,
    _ai: f64,
    lj: &[usize; 3],
    li: &[usize; 3],
    s1d: &[f64],
) -> (f64, [f64; 3], [f64; 6]) {
    let mut v1d = [[0.0f64; 3]; 3]; // [dim][type], type=1:overlap, 2:dipole, 3:quadrupole

    for k in 0..3 {
        let mut vi = [0.0f64; 5]; // polynomial coefficients for i
        let mut vj = [0.0f64; 5]; // polynomial coefficients for j
        let mut vv = [0.0f64; 9]; // product vi*vj

        vi[li[k]] = 1.0;
        vj[lj[k]] = 1.0;

        horizontal_shift(rpi[k], li[k], &mut vi);
        horizontal_shift(rpj[k], lj[k], &mut vj);

        let max_l = li[k] + lj[k];
        form_product(&vi, &vj, li[k], lj[k], &mut vv);

        for l in 0..=max_l {
            v1d[k][0] += s1d[l] * vv[l];
            v1d[k][1] += (s1d[l + 1] + rpi[k] * s1d[l]) * vv[l];
            v1d[k][2] += (s1d[l + 2] + 2.0 * rpi[k] * s1d[l + 1] + rpi[k] * rpi[k] * s1d[l]) * vv[l];
        }
    }

    let s3d = v1d[0][0] * v1d[1][0] * v1d[2][0];

    let d3d = [
        v1d[0][1] * v1d[1][0] * v1d[2][0],
        v1d[0][0] * v1d[1][1] * v1d[2][0],
        v1d[0][0] * v1d[1][0] * v1d[2][1],
    ];

    let q3d = [
        v1d[0][2] * v1d[1][0] * v1d[2][0], // xx
        v1d[0][1] * v1d[1][1] * v1d[2][0], // xy
        v1d[0][0] * v1d[1][2] * v1d[2][0], // yy
        v1d[0][1] * v1d[1][0] * v1d[2][1], // xz
        v1d[0][0] * v1d[1][1] * v1d[2][1], // yz
        v1d[0][0] * v1d[1][0] * v1d[2][2], // zz
    ];

    (s3d, d3d, q3d)
}

/// Generate CGTO parameters from Slater orbital parameters
/// Reimplements slater_to_gauss from tblite/basis/slater.f90
fn slater_to_gauss(ng: usize, n: usize, l: usize, zeta: f64) -> Cgto {
    // Tabulated exponents and coefficients from Stewart (1970)
    // For GFN2, we only need:
    //   H 1s: ng=3, n=1, l=0
    //   First row 2s: ng=4, n=2, l=0
    //   First row 2p: ng=4, n=2, l=1

    let mut alpha = vec![0.0f64; ng];
    let mut coeff = vec![0.0f64; ng];

    match (ng, n, l) {
        // 1s with 3 primitives (H)
        (3, 1, 0) => {
            let a = [2.227660584, 4.057711562e-1, 1.098175104e-1];
            let c = [1.543289673e-1, 5.353281423e-1, 4.446345422e-1];
            for i in 0..3 { alpha[i] = a[i]; coeff[i] = c[i]; }
        }
        // 2s with 4 primitives (He-Ne)
        (4, 2, 0) => {
            let a = [1.161525551e1, 2.000243111, 1.607280687e-1, 6.125744532e-2];
            let c = [-1.198411747e-2, -5.472052539e-2, 5.805587176e-1, 4.770079976e-1];
            for i in 0..4 { alpha[i] = a[i]; coeff[i] = c[i]; }
        }
        // 2p with 4 primitives (He-Ne)
        (4, 2, 1) => {
            let a = [1.798260992, 4.662622228e-1, 1.643718620e-1, 6.543927065e-2];
            let c = [5.713170255e-2, 2.857455515e-1, 5.517873105e-1, 2.632314924e-1];
            for i in 0..4 { alpha[i] = a[i]; coeff[i] = c[i]; }
        }
        _ => {
            panic!("slater_to_gauss not implemented for ng={}, n={}, l={}", ng, n, l);
        }
    }

    // Scale exponents by zeta^2 BEFORE normalization (matches Fortran order)
    let z2 = zeta * zeta;
    for i in 0..ng {
        alpha[i] *= z2;
    }

    // Include normalization in contraction coefficients
    // N = (2α/π)^(3/4) * sqrt(4α)^l / sqrt((2l-1)!!)
    // Fortran dfactorial is 1-indexed: dfactorial(1)=1.0, dfactorial(2)=1.0, dfactorial(3)=3.0
    // Rust DFACTORIAL is 0-indexed: DFACTORIAL[0]=1.0, DFACTORIAL[1]=1.0, DFACTORIAL[2]=3.0
    // Fortran uses dfactorial(l+1), so Rust must use DFACTORIAL[l]
    let dfact_l = DFACTORIAL[l]; // (2l-1)!!
    for i in 0..ng {
        let norm = (TOP * alpha[i]).powf(0.75)
            * (4.0 * alpha[i]).powf(l as f64 / 2.0)
            / dfact_l.sqrt();
        coeff[i] *= norm;
    }

    Cgto { ang: l, nprim: ng, alpha, coeff }
}

/// Build CGTO basis for GFN2 from element indices
pub fn build_cgto_basis(elem_idx: &[usize]) -> Vec<Cgto> {
    let mut cgtos = Vec::new();
    for &z in elem_idx {
        let nshell = params_gfn2::nshell[z];
        for ish in 0..nshell {
            let l = params_gfn2::ang_shell[z][ish];
            let n = params_gfn2::principal_qn[z][ish];
            let zeta = params_gfn2::slater_zeta[z][ish];
            let nprim = params_gfn2::nprim[z][ish];
            if nprim > 0 {
                cgtos.push(slater_to_gauss(nprim, n, l, zeta));
            }
        }
    }
    cgtos
}

/// Compute number of AOs per shell based on angular momentum
fn nao_per_shell(ang: usize) -> usize {
    2 * ang + 1
}

/// Angular momentum exponents for Cartesian functions
/// For GFN2 (s and p only):
///   s: [(0,0,0)]
///   p: [(0,1,0), (0,0,1), (1,0,0)] = [py, pz, px]
fn get_lx(ang: usize, idx: usize) -> [usize; 3] {
    match ang {
        0 => [0, 0, 0],
        1 => {
            match idx {
                0 => [0, 1, 0], // py
                1 => [0, 0, 1], // pz
                2 => [1, 0, 0], // px
                _ => panic!("Invalid p orbital index {}", idx),
            }
        }
        _ => panic!("get_lx not implemented for ang={}", ang),
    }
}

/// Compute multipole integrals between two CGTOs
/// Returns (overlap, dipole, quadrupole) as flat vectors
fn multipole_cgto(
    cgtoi: &Cgto,
    cgtoj: &Cgto,
    r2: f64,
    vec: &[f64; 3],
    intcut: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nao_i = nao_per_shell(cgtoi.ang);
    let nao_j = nao_per_shell(cgtoj.ang);
    let ml_i = if cgtoi.ang == 0 { 1 } else { 3 }; // mlao = n_cartesian
    let ml_j = if cgtoj.ang == 0 { 1 } else { 3 };

    // Cartesian integrals
    let mut s3d = vec![0.0f64; ml_j * ml_i];
    let mut d3d = vec![0.0f64; 3 * ml_j * ml_i];
    let mut q3d = vec![0.0f64; 6 * ml_j * ml_i];

    for ip in 0..cgtoi.nprim {
        for jp in 0..cgtoj.nprim {
            let eab = cgtoi.alpha[ip] + cgtoj.alpha[jp];
            let oab = 1.0 / eab;
            let est = cgtoi.alpha[ip] * cgtoj.alpha[jp] * r2 * oab;
            if est > intcut {
                continue;
            }
            let pre = (-est).exp() * SQRTPI3 * oab.sqrt().powi(3);
            let rpi = [-vec[0] * cgtoj.alpha[jp] * oab,
                        -vec[1] * cgtoj.alpha[jp] * oab,
                        -vec[2] * cgtoj.alpha[jp] * oab];
            let rpj = [ vec[0] * cgtoi.alpha[ip] * oab,
                         vec[1] * cgtoi.alpha[ip] * oab,
                         vec[2] * cgtoi.alpha[ip] * oab];

            let max_l = cgtoi.ang + cgtoj.ang + 2;
            let mut s1d = vec![0.0f64; max_l + 3];
            for l in 0..=max_l + 2 {
                s1d[l] = overlap_1d(l, eab);
            }

            let cc = cgtoi.coeff[ip] * cgtoj.coeff[jp] * pre;

            for mli in 0..ml_i {
                for mlj in 0..ml_j {
                    let li = get_lx(cgtoi.ang, mli);
                    let lj = get_lx(cgtoj.ang, mlj);
                    let (val, dip, quad) = multipole_3d(&rpj, &rpi, cgtoj.alpha[jp], cgtoi.alpha[ip], &lj, &li, &s1d);

                    let idx = mlj + ml_j * mli;
                    s3d[idx] += cc * val;
                    for cmp in 0..3 {
                        d3d[cmp + 3 * idx] += cc * dip[cmp];
                    }
                    for cmp in 0..6 {
                        q3d[cmp + 6 * idx] += cc * quad[cmp];
                    }
                }
            }
        }
    }

    // For s and p orbitals, Cartesian = spherical (identity transform)
    // So we can directly use s3d, d3d, q3d
    let mut overlap = s3d;
    let mut dipole = d3d;
    let mut quadrupole = q3d;

    // Remove trace from quadrupole (convert to traceless quadrupole)
    for mli in 0..nao_i {
        for mlj in 0..nao_j {
            let idx = mlj + nao_j * mli;
            let tr = 0.5 * (quadrupole[0 + 6 * idx] + quadrupole[2 + 6 * idx] + quadrupole[5 + 6 * idx]);
            quadrupole[0 + 6 * idx] = 1.5 * quadrupole[0 + 6 * idx] - tr;
            quadrupole[1 + 6 * idx] = 1.5 * quadrupole[1 + 6 * idx];
            quadrupole[2 + 6 * idx] = 1.5 * quadrupole[2 + 6 * idx] - tr;
            quadrupole[3 + 6 * idx] = 1.5 * quadrupole[3 + 6 * idx];
            quadrupole[4 + 6 * idx] = 1.5 * quadrupole[4 + 6 * idx];
            quadrupole[5 + 6 * idx] = 1.5 * quadrupole[5 + 6 * idx] - tr;
        }
    }

    (overlap, dipole, quadrupole)
}

/// Build full AO multipole integrals for a molecule using GFN2 basis
/// Returns (dipole_ints[nmp=3][nao][nao], quadrupole_ints[nmp=6][nao][nao])
/// In Fortran column-major order: idx = cmp + nmp * jao + nmp * nao * iao
pub fn build_multipole_integrals_gfn2(
    coords: &[[f64; 3]],
    elem_idx: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let nat = coords.len();
    let cgtos = build_cgto_basis(elem_idx);

    // Build nshell_per_atom and shell offsets
    let mut nshell_per_atom = Vec::new();
    let mut shell_offset = 0;
    for &z in elem_idx {
        let nsh = params_gfn2::nshell[z];
        nshell_per_atom.push(nsh);
        shell_offset += nsh;
    }
    let nshell_total = shell_offset;

    // Build shell-to-atom mapping
    let mut sh2at = vec![0usize; nshell_total];
    let mut offset = 0;
    for iat in 0..nat {
        for ish in 0..nshell_per_atom[iat] {
            sh2at[offset + ish] = iat;
        }
        offset += nshell_per_atom[iat];
    }

    // Build AO-to-shell mapping and count total AOs
    let mut nao = 0;
    let mut nao_per_shell_vec = Vec::new();
    let mut iao_sh = vec![0usize; nshell_total];
    for ish in 0..nshell_total {
        iao_sh[ish] = nao;
        let nao_sh = nao_per_shell(cgtos[ish].ang);
        nao_per_shell_vec.push(nao_sh);
        nao += nao_sh;
    }

    let intcut = 25.0; // Fortran max_intcut for high accuracy
    let mut dipole_ints = vec![0.0f64; 3 * nao * nao];
    let mut quadrupole_ints = vec![0.0f64; 6 * nao * nao];

    for iat in 0..nat {
        let izp = elem_idx[iat];
        let is = {
            let mut off = 0;
            for jat in 0..iat { off += nshell_per_atom[jat]; }
            off
        };
        for jat in 0..nat {
            let jzp = elem_idx[jat];
            let js = {
                let mut off = 0;
                for kat in 0..jat { off += nshell_per_atom[kat]; }
                off
            };
            let dx = coords[iat][0] - coords[jat][0];
            let dy = coords[iat][1] - coords[jat][1];
            let dz = coords[iat][2] - coords[jat][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let vec = [dx, dy, dz];

            for ish in 0..nshell_per_atom[iat] {
                let ii = iao_sh[is + ish];
                for jsh in 0..nshell_per_atom[jat] {
                    let jj = iao_sh[js + jsh];
                    let nao_i = nao_per_shell_vec[is + ish];
                    let nao_j = nao_per_shell_vec[js + jsh];

                    let (overlap, dipole, quadrupole) = multipole_cgto(
                        &cgtos[is + ish], &cgtos[js + jsh], r2, &vec, intcut
                    );

                    for iao in 0..nao_i {
                        for jao in 0..nao_j {
                            // Fortran column-major order
                            let base_ji = jj + jao + nao * (ii + iao);
                            for cmp in 0..3 {
                                dipole_ints[cmp + 3 * base_ji] = dipole[cmp + 3 * jao + 3 * nao_j * iao];
                            }
                            for cmp in 0..6 {
                                quadrupole_ints[cmp + 6 * base_ji] = quadrupole[cmp + 6 * jao + 6 * nao_j * iao];
                            }
                        }
                    }
                }
            }
        }
    }

    (dipole_ints, quadrupole_ints)
}

/// Build full AO overlap integrals for a molecule using GFN2 basis
pub fn build_overlap_gfn2(
    coords: &[[f64; 3]],
    elem_idx: &[usize],
) -> Vec<f64> {
    let nat = coords.len();
    let cgtos = build_cgto_basis(elem_idx);

    let mut nshell_per_atom = Vec::new();
    let mut shell_offset = 0;
    for &z in elem_idx {
        let nsh = params_gfn2::nshell[z];
        nshell_per_atom.push(nsh);
        shell_offset += nsh;
    }
    let nshell_total = shell_offset;

    let mut iao_sh = vec![0usize; nshell_total];
    let mut nao = 0;
    let mut nao_per_shell_vec = Vec::new();
    for ish in 0..nshell_total {
        iao_sh[ish] = nao;
        let nao_sh = nao_per_shell(cgtos[ish].ang);
        nao_per_shell_vec.push(nao_sh);
        nao += nao_sh;
    }

    let intcut = 25.0;
    let mut overlap_ints = vec![0.0f64; nao * nao];

    for iat in 0..nat {
        let is = {
            let mut off = 0;
            for jat in 0..iat { off += nshell_per_atom[jat]; }
            off
        };
        for jat in 0..nat {
            let js = {
                let mut off = 0;
                for kat in 0..jat { off += nshell_per_atom[kat]; }
                off
            };
            let dx = coords[iat][0] - coords[jat][0];
            let dy = coords[iat][1] - coords[jat][1];
            let dz = coords[iat][2] - coords[jat][2];
            let r2 = dx*dx + dy*dy + dz*dz;
            let vec = [dx, dy, dz];

            for ish in 0..nshell_per_atom[iat] {
                let ii = iao_sh[is + ish];
                for jsh in 0..nshell_per_atom[jat] {
                    let jj = iao_sh[js + jsh];
                    let nao_i = nao_per_shell_vec[is + ish];
                    let nao_j = nao_per_shell_vec[js + jsh];

                    let (overlap, _, _) = multipole_cgto(
                        &cgtos[is + ish], &cgtos[js + jsh], r2, &vec, intcut
                    );

                    for iao in 0..nao_i {
                        for jao in 0..nao_j {
                            let base_ji = jj + jao + nao * (ii + iao);
                            overlap_ints[base_ji] = overlap[jao + nao_j * iao];
                        }
                    }
                }
            }
        }
    }

    overlap_ints
}
