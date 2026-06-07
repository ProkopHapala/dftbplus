//! GFN1-xTB and GFN2-xTB Hamiltonian builders (non-SCC H0).

use crate::methods::xtb::{basis::*, integrals::*, params, params_gfn2};
use nalgebra::DMatrix;

/// Exponential counting function from dftd3_ncoord (tblite exp CN type)
fn exp_count(k: f64, r: f64, r0: f64) -> f64 {
    1.0 / (1.0 + (-k * (r0 / r - 1.0)).exp())
}

/// Coordination number using GFN1 single-exponential formula
/// GFN1 uses cn_type="exp" which calls dftd3_ncoord with kcn=16.0
pub fn compute_cn_gfn1(coords: &[[f64; 3]], elem_idx: &[usize]) -> Vec<f64> {
    let nat = coords.len();
    let mut cn = vec![0.0; nat];
    let kcn = 16.0;
    for i in 0..nat {
        for j in 0..i {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx*dx + dy*dy + dz*dz;
            if r2 < 1.0e-12 { continue; }
            let r = r2.sqrt();
            let cov_r = params::cov_rad[elem_idx[i]] + params::cov_rad[elem_idx[j]];
            let cn_ij = exp_count(kcn, r, cov_r);
            cn[i] += cn_ij;
            if i != j {
                cn[j] += cn_ij;
            }
        }
    }
    cn
}

/// Coordination number using GFN double-exponential formula
/// GFN2 uses cn_type="gfn": count = exp_count(ka=10, r, rc) * exp_count(kb=20, r, rc+2.0)
pub fn compute_cn_gfn2(coords: &[[f64; 3]], elem_idx: &[usize]) -> Vec<f64> {
    let nat = coords.len();
    let mut cn = vec![0.0; nat];
    let ka = 10.0;
    let kb = 20.0;
    let r_shift = 2.0;
    for i in 0..nat {
        for j in 0..i {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx*dx + dy*dy + dz*dz;
            if r2 < 1.0e-12 { continue; }
            let r = r2.sqrt();
            let cov_r = params_gfn2::cov_rad[elem_idx[i]] + params_gfn2::cov_rad[elem_idx[j]];
            let cn_ij = exp_count(ka, r, cov_r) * exp_count(kb, r, cov_r + r_shift);
            cn[i] += cn_ij;
            if i != j {
                cn[j] += cn_ij;
            }
        }
    }
    cn
}

/// Build the non-SCC H0 and S matrices for a molecular system
pub fn build_h0_s(
    coords: &[[f64; 3]],
    elem_idx: &[usize],
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>, Vec<usize>) {
    let nat = coords.len();

    // Build CGTOs for each atom
    let mut cgtos_per_atom: Vec<Vec<Cgto>> = Vec::with_capacity(nat);
    for &eidx in elem_idx {
        cgtos_per_atom.push(build_element_cgtos(eidx));
    }

    // Compute CN for each atom
    let cn = compute_cn_gfn1(coords, elem_idx);

    // Build shell->atom mapping and count AOs
    let mut nao = 0usize;
    let mut nsh = 0usize;
    let mut shell_elem: Vec<usize> = Vec::new(); // which element each shell belongs to
    let mut shell_idx: Vec<usize> = Vec::new();  // shell index within element
    let mut sh_ao_off: Vec<usize> = Vec::new();  // AO offset for each shell
    let mut atom_sh_off: Vec<usize> = vec![0];   // shell offset for each atom
    for (iat, atom_cgtos) in cgtos_per_atom.iter().enumerate() {
        let mut atom_nao = 0;
        for (ish, cgto) in atom_cgtos.iter().enumerate() {
            shell_elem.push(elem_idx[iat]);
            shell_idx.push(ish);
            sh_ao_off.push(nao + atom_nao);
            nsh += 1;
            atom_nao += msao(cgto.ang);
        }
        nao += atom_nao;
        atom_sh_off.push(nsh);
    }

    let mut h = DMatrix::<f64>::zeros(nao, nao);
    let s = build_overlap_matrix(&cgtos_per_atom, coords);

    // Compute shell selfenergies with CN dependence
    let mut selfenergy: Vec<f64> = Vec::with_capacity(nsh);
    for sh in 0..nsh {
        let eidx = shell_elem[sh];
        let ish = shell_idx[sh];
        // Find which atom this shell belongs to
        let mut iat = 0;
        let mut sh_count = 0;
        for (a, cgtos) in cgtos_per_atom.iter().enumerate() {
            if sh_count + cgtos.len() > sh {
                iat = a;
                break;
            }
            sh_count += cgtos.len();
        }
        let se = params::selfenergy(eidx, ish) - params::kcn(eidx, ish) * cn[iat];
        selfenergy.push(se);
    }

    // Build Hamiltonian using shell-pair loop (like tblite)
    let mut i_sh_off = 0usize;
    for (iat, atom_i) in cgtos_per_atom.iter().enumerate() {
        let mut j_sh_off = 0usize;
        for (jat, atom_j) in cgtos_per_atom.iter().enumerate() {
            for (ish, cgto_i) in atom_i.iter().enumerate() {
                for (jsh, cgto_j) in atom_j.iter().enumerate() {
                    let nao_i = msao(cgto_i.ang);
                    let nao_j = msao(cgto_j.ang);

                    let eidx_i = elem_idx[iat];
                    let eidx_j = elem_idx[jat];

                    // Compute shpoly and hij scaling
                    let dx = coords[iat][0] - coords[jat][0];
                    let dy = coords[iat][1] - coords[jat][1];
                    let dz = coords[iat][2] - coords[jat][2];
                    let r2 = dx*dx + dy*dy + dz*dz;

                    let rr = if r2 > 1e-14 {
                        let r = r2.sqrt();
                        let rad_sum = params::atomic_rad[eidx_i] + params::atomic_rad[eidx_j];
                        (r / rad_sum).sqrt()
                    } else {
                        0.0
                    };

                    let shpoly_i = params::shpoly[eidx_i][cgto_i.ang];
                    let shpoly_j = params::shpoly[eidx_j][cgto_j.ang];
                    let shpoly = (1.0 + shpoly_i * rr) * (1.0 + shpoly_j * rr);

                    // hscale
                    let hscale = compute_hscale(
                        eidx_i, eidx_j,
                        cgto_i.ang, cgto_j.ang,
                        ish, jsh,
                        &cgtos_per_atom[iat], &cgtos_per_atom[jat],
                    );

                    let se_i = selfenergy[i_sh_off + ish];
                    let se_j = selfenergy[j_sh_off + jsh];

                    // hij: off-site includes hscale, on-site does not (tblite h0.f90)
                    let hij = if iat == jat {
                        0.5 * (se_i + se_j) * shpoly
                    } else {
                        0.5 * (se_i + se_j) * hscale * shpoly
                    };

                    // Extract overlap block and fill H
                    let d = [dx, dy, dz];
                    let s_block = overlap_cgto(cgto_i, cgto_j, &d);

                    let i_ao_off = sh_ao_off[i_sh_off + ish];
                    let j_ao_off = sh_ao_off[j_sh_off + jsh];

                    for col in 0..nao_i {
                        for row in 0..nao_j {
                            let s_val = s_block[row * nao_i + col];
                            let h_val = if iat == jat && ish == jsh && row == col {
                                se_i // On-site diagonal = selfenergy
                            } else {
                                s_val * hij
                            };
                            h[(j_ao_off + row, i_ao_off + col)] = h_val;
                        }
                    }
                }
            }
            j_sh_off += atom_j.len();
        }
        i_sh_off += atom_i.len();
    }

    (h, s, shell_elem, shell_idx)
}

/// Compute hscale for a shell pair
fn compute_hscale(
    eidx_i: usize, eidx_j: usize,
    li: usize, lj: usize,
    ish: usize, jsh: usize,
    cgtos_i: &[Cgto], cgtos_j: &[Cgto],
) -> f64 {
    // Determine valence status: first shell of each angular momentum is valence
    let val_i = is_valence(ish, cgtos_i);
    let val_j = is_valence(jsh, cgtos_j);

    let den = (params::pauling_en[eidx_i] - params::pauling_en[eidx_j]).powi(2);
    let enp = 1.0 + params::ENSCALE * den;

    let kshell_val = params::kshell(lj, li);
    let kpair_val = params::kpair(eidx_i, eidx_j);

    match (val_i, val_j) {
        (true, true) => kpair_val * kshell_val * enp,
        (true, false) => 0.5 * (params::kshell(li, li) + params::KDIFF),
        (false, true) => 0.5 * (params::kshell(lj, lj) + params::KDIFF),
        (false, false) => params::KDIFF,
    }
}

fn is_valence(ish: usize, cgtos: &[Cgto]) -> bool {
    let l = cgtos[ish].ang;
    // First occurrence of this angular momentum is valence
    for j in 0..ish {
        if cgtos[j].ang == l {
            return false;
        }
    }
    true
}

/// Build the non-SCC H0 and S matrices for GFN2-xTB
pub fn build_h0_s_gfn2(
    coords: &[[f64; 3]],
    elem_idx: &[usize],
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>, Vec<usize>) {
    let nat = coords.len();

    // Build CGTOs for each atom (GFN2: no orthogonalization)
    let mut cgtos_per_atom: Vec<Vec<Cgto>> = Vec::with_capacity(nat);
    for &eidx in elem_idx {
        cgtos_per_atom.push(build_element_cgtos_gfn2(eidx));
    }

    // Compute CN for each atom (GFN2 double-exponential)
    let cn = compute_cn_gfn2(coords, elem_idx);

    // Build shell->atom mapping and count AOs
    let mut nao = 0usize;
    let mut nsh = 0usize;
    let mut shell_elem: Vec<usize> = Vec::new();
    let mut shell_idx: Vec<usize> = Vec::new();
    let mut sh_ao_off: Vec<usize> = Vec::new();
    let mut atom_sh_off: Vec<usize> = vec![0];
    for (iat, atom_cgtos) in cgtos_per_atom.iter().enumerate() {
        let mut atom_nao = 0;
        for (ish, cgto) in atom_cgtos.iter().enumerate() {
            shell_elem.push(elem_idx[iat]);
            shell_idx.push(ish);
            sh_ao_off.push(nao + atom_nao);
            nsh += 1;
            atom_nao += msao(cgto.ang);
        }
        nao += atom_nao;
        atom_sh_off.push(nsh);
    }

    let mut h = DMatrix::<f64>::zeros(nao, nao);
    let s = build_overlap_matrix(&cgtos_per_atom, coords);

    // Precompute atom index for each shell
    let mut shell_atom = vec![0usize; nsh];
    {
        let mut sh = 0;
        for (iat, cgtos) in cgtos_per_atom.iter().enumerate() {
            for _ in 0..cgtos.len() {
                shell_atom[sh] = iat;
                sh += 1;
            }
        }
    }

    // Compute shell selfenergies with CN dependence
    // GFN2: kcn is indexed by angular momentum, not shell index
    let mut selfenergy: Vec<f64> = Vec::with_capacity(nsh);
    for sh in 0..nsh {
        let eidx = shell_elem[sh];
        let ish = shell_idx[sh];
        let iat = shell_atom[sh];
        let l = cgtos_per_atom[iat][ish].ang;
        let se = params_gfn2::selfenergy(eidx, ish) - params_gfn2::kcn(eidx, l) * cn[iat];
        selfenergy.push(se);
    }

    // Build Hamiltonian using shell-pair loop
    let mut i_sh_off = 0usize;
    for (iat, atom_i) in cgtos_per_atom.iter().enumerate() {
        let mut j_sh_off = 0usize;
        for (jat, atom_j) in cgtos_per_atom.iter().enumerate() {
            for (ish, cgto_i) in atom_i.iter().enumerate() {
                for (jsh, cgto_j) in atom_j.iter().enumerate() {
                    let nao_i = msao(cgto_i.ang);
                    let nao_j = msao(cgto_j.ang);

                    let eidx_i = elem_idx[iat];
                    let eidx_j = elem_idx[jat];

                    // Compute shpoly and hij scaling
                    let dx = coords[iat][0] - coords[jat][0];
                    let dy = coords[iat][1] - coords[jat][1];
                    let dz = coords[iat][2] - coords[jat][2];
                    let r2 = dx*dx + dy*dy + dz*dz;

                    let rr = if r2 > 1e-14 {
                        let r = r2.sqrt();
                        let rad_sum = params_gfn2::atomic_rad[eidx_i] + params_gfn2::atomic_rad[eidx_j];
                        (r / rad_sum).sqrt()
                    } else {
                        0.0
                    };

                    // GFN2: shpoly indexed by angular momentum
                    let shpoly_i = params_gfn2::shpoly[eidx_i][cgto_i.ang];
                    let shpoly_j = params_gfn2::shpoly[eidx_j][cgto_j.ang];
                    let shpoly = (1.0 + shpoly_i * rr) * (1.0 + shpoly_j * rr);

                    // hscale (GFN2 with zij factor)
                    let hscale = compute_hscale_gfn2(
                        eidx_i, eidx_j,
                        cgto_i.ang, cgto_j.ang,
                        ish, jsh,
                    );

                    let se_i = selfenergy[i_sh_off + ish];
                    let se_j = selfenergy[j_sh_off + jsh];

                    // hij: off-site includes hscale, on-site does not
                    let hij = if iat == jat {
                        0.5 * (se_i + se_j) * shpoly
                    } else {
                        0.5 * (se_i + se_j) * hscale * shpoly
                    };

                    // Extract overlap block and fill H
                    let d = [dx, dy, dz];
                    let s_block = overlap_cgto(cgto_i, cgto_j, &d);

                    let i_ao_off = sh_ao_off[i_sh_off + ish];
                    let j_ao_off = sh_ao_off[j_sh_off + jsh];

                    for col in 0..nao_i {
                        for row in 0..nao_j {
                            let s_val = s_block[row * nao_i + col];
                            let h_val = if iat == jat && ish == jsh && row == col {
                                se_i // On-site diagonal = selfenergy
                            } else {
                                s_val * hij
                            };
                            h[(j_ao_off + row, i_ao_off + col)] = h_val;
                        }
                    }
                }
            }
            j_sh_off += atom_j.len();
        }
        i_sh_off += atom_i.len();
    }

    (h, s, shell_elem, shell_idx)
}

/// Compute hscale for GFN2 shell pair
fn compute_hscale_gfn2(
    eidx_i: usize, eidx_j: usize,
    li: usize, lj: usize,
    ish: usize, jsh: usize,
) -> f64 {
    let zi = params_gfn2::slater_zeta[eidx_i][ish];
    let zj = params_gfn2::slater_zeta[eidx_j][jsh];
    let zij = (2.0 * (zi * zj).sqrt() / (zi + zj)).powf(params_gfn2::WEXP);

    let den = (params_gfn2::pauling_en[eidx_i] - params_gfn2::pauling_en[eidx_j]).powi(2);
    let enp = 1.0 + params_gfn2::ENSCALE * den;

    let kshell_val = params_gfn2::kshell(lj, li);
    let kpair_val = params_gfn2::kpair(eidx_i, eidx_j);

    zij * kpair_val * kshell_val * enp
}

/// xTB Hamiltonian builder implementing the H0Builder trait
#[derive(Debug, Clone)]
pub struct XtbBuilder;

impl XtbBuilder {
    pub fn new() -> Self {
        Self
    }
}
