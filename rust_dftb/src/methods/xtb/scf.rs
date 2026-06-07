//! SCF loop and potential builder for xTB (GFN1 and GFN2)

use crate::methods::xtb::coulomb::{build_coulomb_matrix, build_coulomb_matrix_gfn2, thirdorder_potential, thirdorder_potential_gfn2};
use crate::methods::xtb::mulliken::{shell_charges, atomic_charges, reference_shell_occupations, reference_shell_occupations_gfn2};
use crate::methods::xtb::hamiltonian::{build_h0_s, build_h0_s_gfn2};
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

/// Build full SCC Hamiltonian including third-order correction (GFN1).
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

/// Build full SCC Hamiltonian including third-order correction (GFN2, shell-resolved).
pub fn build_scc_hamiltonian_with_thirdorder_gfn2(
    h0: &DMatrix<f64>,
    s: &DMatrix<f64>,
    gamma: &DMatrix<f64>,
    shell_charges: &DVector<f64>,
    nshell_per_atom: &[usize],
    elem_idx: &[usize],
    ang_per_shell: &[usize],
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

    // Add third-order onsite correction (shell-resolved for GFN2)
    let v3 = thirdorder_potential_gfn2(shell_charges, nshell_per_atom, elem_idx, ang_per_shell);
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

/// Add multipole potentials to Hamiltonian (GFN2).
/// Reproduces tblite scf/potential.f90 add_vmp_to_h1 and add_vao_to_h1.
///
/// Inputs (from tblite after SCF convergence):
///   - s: overlap matrix
///   - dipole_ints: flattened [3][nao][nao] (Fortran column-major)
///   - quadrupole_ints: flattened [6][nao][nao] (Fortran column-major)
///   - charge_pot: flattened [nat] atomic charge potential (vat, includes multipole cross-terms)
///   - dipole_pot: flattened [3][nat] (Fortran column-major)
///   - quadrupole_pot: flattened [6][nat] (Fortran column-major)
///   - ao2at: mapping from AO index to atom index
pub fn add_multipole_to_h1(
    h1: &mut DMatrix<f64>,
    s: &DMatrix<f64>,
    dipole_ints: &[f64],
    quadrupole_ints: &[f64],
    charge_pot: &[f64],
    dipole_pot: &[f64],
    quadrupole_pot: &[f64],
    ao2at: &[usize],
) {
    let nao = h1.nrows();
    let nat = charge_pot.len();

    // Indirect contribution via charge potential (add_vao_to_h1)
    for iao in 0..nao {
        let iat = ao2at[iao];
        for jao in 0..nao {
            let jat = ao2at[jao];
            let v_i = charge_pot[iat];
            let v_j = charge_pot[jat];
            h1[(jao, iao)] -= s[(jao, iao)] * 0.5 * (v_j + v_i);
        }
    }

    // Direct dipole contribution (add_vmp_to_h1)
    for iao in 0..nao {
        let iat = ao2at[iao];
        for jao in 0..nao {
            let jat = ao2at[jao];
            for cmp in 0..3 {
                let d_ji = dipole_ints[cmp + 3 * jao + 3 * nao * iao];
                let d_ij = dipole_ints[cmp + 3 * iao + 3 * nao * jao];
                let v_i = dipole_pot[cmp + 3 * iat];
                let v_j = dipole_pot[cmp + 3 * jat];
                h1[(jao, iao)] -= 0.5 * d_ji * v_i;
                h1[(jao, iao)] -= 0.5 * d_ij * v_j;
            }
        }
    }

    // Direct quadrupole contribution (add_vmp_to_h1)
    for iao in 0..nao {
        let iat = ao2at[iao];
        for jao in 0..nao {
            let jat = ao2at[jao];
            for cmp in 0..6 {
                let q_ji = quadrupole_ints[cmp + 6 * jao + 6 * nao * iao];
                let q_ij = quadrupole_ints[cmp + 6 * iao + 6 * nao * jao];
                let v_i = quadrupole_pot[cmp + 6 * iat];
                let v_j = quadrupole_pot[cmp + 6 * jat];
                h1[(jao, iao)] -= 0.5 * q_ji * v_i;
                h1[(jao, iao)] -= 0.5 * q_ij * v_j;
            }
        }
    }
}

/// GFN-style coordination number using double exponential counting function.
/// Matches tblite ncoord/gfn.f90 ncoord_exp exactly.
pub fn compute_coordination_numbers(coords: &[[f64; 3]], elem_idx: &[usize]) -> Vec<f64> {
    use crate::methods::xtb::params_gfn2;
    let nat = coords.len();
    let mut cn = vec![0.0f64; nat];
    let ka = 10.0;
    let kb = 20.0;
    let r_shift = 2.0;
    for iat in 0..nat {
        for jat in (iat+1)..nat {
            let dx = coords[iat][0] - coords[jat][0];
            let dy = coords[iat][1] - coords[jat][1];
            let dz = coords[iat][2] - coords[jat][2];
            let r2 = dx*dx + dy*dy + dz*dz;
            if r2 < 1.0e-12 { continue; }
            let r = r2.sqrt();
            let izp = elem_idx[iat];
            let jzp = elem_idx[jat];
            let rc = params_gfn2::cov_rad[izp] + params_gfn2::cov_rad[jzp];
            let countf = exp_count(ka, r, rc) * exp_count(kb, r, rc + r_shift);
            cn[iat] += countf;
            cn[jat] += countf;
        }
    }
    cn
}

/// Exponential counting function: 1/(1+exp(-k*(r0/r-1)))
fn exp_count(k: f64, r: f64, r0: f64) -> f64 {
    1.0 / (1.0 + (-k * (r0/r - 1.0)).exp())
}

/// Compute multipole damping radii from coordination numbers
/// mrad(iat) = rad(izp) + (rmax - rad(izp)) / (1 + exp(-kexp*(cn - vcn - shift)))
pub fn compute_multipole_radii(cn: &[f64], elem_idx: &[usize]) -> Vec<f64> {
    use crate::methods::xtb::params_gfn2;
    let nat = cn.len();
    let mut mrad = vec![0.0f64; nat];
    for iat in 0..nat {
        let izp = elem_idx[iat];
        let rad0 = params_gfn2::rad[izp];
        let arg = cn[iat] - params_gfn2::vcn[izp] - params_gfn2::MP_SHIFT;
        let t1 = (-params_gfn2::MP_KEXP * arg).exp();
        let t2 = (params_gfn2::MP_RMAX - rad0) / (1.0 + t1);
        mrad[iat] = rad0 + t2;
    }
    mrad
}

/// Build multipole interaction matrices for 0D (non-periodic) systems.
/// Returns (amat_sd[3][nat][nat], amat_dd[3][nat][3][nat], amat_sq[6][nat][nat])
pub fn build_multipole_interaction_matrices_0d(
    coords: &[[f64; 3]],
    mrad: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use crate::methods::xtb::params_gfn2;
    let nat = coords.len();
    let mut amat_sd = vec![0.0f64; 3 * nat * nat];
    let mut amat_dd = vec![0.0f64; 3 * nat * 3 * nat];
    let mut amat_sq = vec![0.0f64; 6 * nat * nat];

    for iat in 0..nat {
        for jat in 0..nat {
            if iat == jat { continue; }
            let vec = [
                coords[iat][0] - coords[jat][0],
                coords[iat][1] - coords[jat][1],
                coords[iat][2] - coords[jat][2],
            ];
            let r = (vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]).sqrt();
            let g1 = 1.0 / r;
            let g3 = g1 * g1 * g1;
            let g5 = g3 * g1 * g1;

            let rr = 0.5 * (mrad[jat] + mrad[iat]) * g1;
            let fdmp3 = 1.0 / (1.0 + 6.0 * rr.powf(params_gfn2::MP_DMP3));
            let fdmp5 = 1.0 / (1.0 + 6.0 * rr.powf(params_gfn2::MP_DMP5));

            // amat_sd(:, jat, iat) = vec * g3 * fdmp3
            for cmp in 0..3 {
                amat_sd[cmp + 3 * jat + 3 * nat * iat] += vec[cmp] * g3 * fdmp3;
            }

            // amat_dd(:, jat, :, iat) = I * g3*fdmp5 - 3 * vec ⊗ vec * g5*fdmp5
            for a in 0..3 {
                for b in 0..3 {
                    let delta = if a == b { 1.0 } else { 0.0 };
                    amat_dd[a + 3 * jat + 3 * nat * b + 3 * nat * 3 * iat] +=
                        delta * g3 * fdmp5 - 3.0 * vec[a] * vec[b] * g5 * fdmp5;
                }
            }

            // amat_sq(:, jat, iat) = [xx, 2xy, yy, 2xz, 2yz, zz] * g5 * fdmp5
            // Fortran does NOT apply trace correction here (matches tblite multipole.f90)
            amat_sq[0 + 6 * jat + 6 * nat * iat] += vec[0] * vec[0] * g5 * fdmp5;
            amat_sq[1 + 6 * jat + 6 * nat * iat] += 2.0 * vec[0] * vec[1] * g5 * fdmp5;
            amat_sq[2 + 6 * jat + 6 * nat * iat] += vec[1] * vec[1] * g5 * fdmp5;
            amat_sq[3 + 6 * jat + 6 * nat * iat] += 2.0 * vec[0] * vec[2] * g5 * fdmp5;
            amat_sq[4 + 6 * jat + 6 * nat * iat] += 2.0 * vec[1] * vec[2] * g5 * fdmp5;
            amat_sq[5 + 6 * jat + 6 * nat * iat] += vec[2] * vec[2] * g5 * fdmp5;
        }
    }

    (amat_sd, amat_dd, amat_sq)
}

/// Compute multipole potentials from atomic multipoles and interaction matrices.
/// Also includes onsite exchange-correlation kernels.
pub fn compute_multipole_potentials(
    qat: &[f64],
    dpat: &[f64], // [3][nat]
    qpat: &[f64], // [6][nat]
    amat_sd: &[f64], // [3][nat][nat]
    amat_dd: &[f64], // [3][nat][3][nat]
    amat_sq: &[f64], // [6][nat][nat]
    elem_idx: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use crate::methods::xtb::params_gfn2;
    let nat = qat.len();
    let mut vat = vec![0.0f64; nat];
    let mut vdp = vec![0.0f64; 3 * nat];
    let mut vqp = vec![0.0f64; 6 * nat];

    // Cross-term: dipole-charge (amat_sd^T * dpat contributes to vat)
    let mut vat_sd = vec![0.0f64; nat];
    for iat in 0..nat {
        for jat in 0..nat {
            for cmp in 0..3 {
                let sd = amat_sd[cmp + 3 * jat + 3 * nat * iat];
                vat[iat] += sd * dpat[cmp + 3 * jat];
                vat_sd[iat] += sd * dpat[cmp + 3 * jat];
            }
        }
    }

    // Cross-term: quadrupole-charge (amat_sq^T * qpat contributes to vat)
    let mut vat_sq = vec![0.0f64; nat];
    for iat in 0..nat {
        for jat in 0..nat {
            for cmp in 0..6 {
                let sq = amat_sq[cmp + 6 * jat + 6 * nat * iat];
                vat[iat] += sq * qpat[cmp + 6 * jat];
                vat_sq[iat] += sq * qpat[cmp + 6 * jat];
            }
        }
    }

    // DEBUG: print breakdown
    if nat == 2 {
        eprintln!("DEBUG vat breakdown: vat_sd[0]={:.9e}, vat_sq[0]={:.9e}, vat[0]={:.9e}", vat_sd[0], vat_sq[0], vat[0]);
    }

    // Charge-dipole (amat_sd * qat contributes to vdp)
    for iat in 0..nat {
        for jat in 0..nat {
            if iat == jat { continue; }
            for cmp in 0..3 {
                // Fortran gemv: vdp(cmp,jat) = sum_iat amat_sd(cmp,jat,iat)*qat(iat)
                let sd = amat_sd[cmp + 3 * jat + 3 * nat * iat];
                vdp[cmp + 3 * jat] += sd * qat[iat];
            }
        }
    }

    // Dipole-dipole (amat_dd * dpat contributes to vdp)
    for iat in 0..nat {
        for jat in 0..nat {
            if iat == jat { continue; }
            for a in 0..3 {
                for b in 0..3 {
                    // Fortran gemv: vdp(a,jat) = sum_{b,iat} amat_dd(a,jat,b,iat)*dpat(b,iat)
                    let dd = amat_dd[a + 3 * jat + 3 * nat * b + 3 * nat * 3 * iat];
                    vdp[a + 3 * jat] += dd * dpat[b + 3 * iat];
                }
            }
        }
    }

    // Charge-quadrupole (amat_sq * qat contributes to vqp)
    for iat in 0..nat {
        for jat in 0..nat {
            if iat == jat { continue; }
            for cmp in 0..6 {
                // Fortran gemv: vqp(cmp,jat) = sum_iat amat_sq(cmp,jat,iat)*qat(iat)
                let sq = amat_sq[cmp + 6 * jat + 6 * nat * iat];
                vqp[cmp + 6 * jat] += sq * qat[iat];
            }
        }
    }

    // Onsite dipole XC kernel: vdp += 2 * dkernel * dpat
    for iat in 0..nat {
        let izp = elem_idx[iat];
        let k = 2.0 * params_gfn2::dkernel[izp];
        for cmp in 0..3 {
            vdp[cmp + 3 * iat] += k * dpat[cmp + 3 * iat];
        }
    }

    // Onsite quadrupole XC kernel: vqp += 2 * qkernel * qpat * mpscale
    // mpscale = [1, 2, 1, 2, 2, 1] for components [xx, xy, yy, xz, yz, zz]
    let mpscale = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0];
    for iat in 0..nat {
        let izp = elem_idx[iat];
        let k = 2.0 * params_gfn2::qkernel[izp];
        for cmp in 0..6 {
            vqp[cmp + 6 * iat] += k * qpat[cmp + 6 * iat] * mpscale[cmp];
        }
    }

    (vat, vdp, vqp)
}

/// Generic SCF loop
fn run_scf_core(
    h0: DMatrix<f64>,
    s: DMatrix<f64>,
    gamma: DMatrix<f64>,
    nshell_per_atom: Vec<usize>,
    ang_per_shell: Vec<usize>,
    elem_idx: Vec<usize>,
    n_electrons: usize,
    max_iter: usize,
    conv_thresh: f64,
    use_gfn2_thirdorder: bool,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let nao = s.nrows();
    let nshell_total: usize = nshell_per_atom.iter().sum();

    let n0sh = if use_gfn2_thirdorder {
        reference_shell_occupations_gfn2(&nshell_per_atom, &elem_idx, &ang_per_shell)
    } else {
        reference_shell_occupations(&nshell_per_atom, &elem_idx, &ang_per_shell)
    };

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

    const MIXING: f64 = 0.3;

    let mut density = DMatrix::zeros(nao, nao);
    let mut qsh = DVector::zeros(nshell_total);

    for iter in 0..max_iter {
        let h_scc = if use_gfn2_thirdorder {
            build_scc_hamiltonian_with_thirdorder_gfn2(
                &h0, &s, &gamma, &qsh, &nshell_per_atom, &elem_idx, &ang_per_shell, &ao2sh
            )
        } else {
            build_scc_hamiltonian_with_thirdorder(
                &h0, &s, &gamma, &qsh, &nshell_per_atom, &elem_idx, &ao2sh
            )
        };

        let (eigenvalues, eigenvectors) = solve_gevp(&h_scc, &s);

        let n_occ = n_electrons / 2;
        let mut new_density = DMatrix::zeros(nao, nao);
        for i in 0..n_occ {
            let c = eigenvectors.column(i);
            new_density += 2.0 * c * c.transpose();
        }

        let new_qsh = shell_charges(&new_density, &s, &ao2sh, &n0sh);

        let max_diff = (&new_qsh - &qsh).abs().max();
        if iter % 10 == 0 {
            println!("Iter {}: max charge diff = {:.6e}", iter, max_diff);
        }
        if max_diff < conv_thresh {
            println!("SCF converged in {} iterations", iter + 1);
            // Recompute eigenvalues with converged charges for consistency with tblite
            let h_final = if use_gfn2_thirdorder {
                build_scc_hamiltonian_with_thirdorder_gfn2(
                    &h0, &s, &gamma, &new_qsh, &nshell_per_atom, &elem_idx, &ang_per_shell, &ao2sh
                )
            } else {
                build_scc_hamiltonian_with_thirdorder(
                    &h0, &s, &gamma, &new_qsh, &nshell_per_atom, &elem_idx, &ao2sh
                )
            };
            let (eigenvalues_final, _) = solve_gevp(&h_final, &s);
            return (new_density, new_qsh, eigenvalues_final);
        }

        qsh = MIXING * &new_qsh + (1.0 - MIXING) * &qsh;
        density = MIXING * &new_density + (1.0 - MIXING) * &density;
    }

    println!("SCF did not converge in {} iterations", max_iter);
    (density, qsh, DVector::zeros(nao))
}

/// Simple SCF loop for GFN1-xTB
pub fn run_scf(
    coords: &[[f64; 3]],
    elem_idx: &[usize],
    n_electrons: usize,
    max_iter: usize,
    conv_thresh: f64,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let nshell_per_atom: Vec<usize> = elem_idx.iter().map(|&z| {
        match z {
            0 => 2, // H: 1s, 2s
            7 => 2, // O: 2s, 2p
            _ => 2, // Default: 2 shells
        }
    }).collect();
    let ang_per_shell: Vec<usize> = elem_idx.iter().flat_map(|&z| {
        match z {
            0 => vec![0, 0], // H: s, s
            7 => vec![0, 1], // O: s, p
            _ => vec![0, 1], // Default: s, p
        }
    }).collect();

    let (h0, s, _, _) = build_h0_s(coords, elem_idx);
    let gamma = build_coulomb_matrix(coords, &nshell_per_atom, elem_idx, &ang_per_shell);

    run_scf_core(h0, s, gamma, nshell_per_atom, ang_per_shell, elem_idx.to_vec(), n_electrons, max_iter, conv_thresh, false)
}

/// SCF loop for GFN2-xTB with full multipole support
pub fn run_scf_gfn2(
    coords: &[[f64; 3]],
    elem_idx: &[usize],
    n_electrons: usize,
    max_iter: usize,
    conv_thresh: f64,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    use crate::methods::xtb::params_gfn2;
    use crate::methods::xtb::multipole_integrals;
    use crate::methods::xtb::mulliken::{shell_charges, atomic_charges, atomic_multipoles};

    let nat = coords.len();
    let nshell_per_atom: Vec<usize> = elem_idx.iter().map(|&z| params_gfn2::nshell[z]).collect();

    let mut ang_per_shell = Vec::new();
    for &z in elem_idx {
        let nsh = params_gfn2::nshell[z];
        for ish in 0..nsh {
            ang_per_shell.push(params_gfn2::ang_shell[z][ish]);
        }
    }

    let (h0, s, _, _) = build_h0_s_gfn2(coords, elem_idx);
    let gamma = build_coulomb_matrix_gfn2(coords, &nshell_per_atom, elem_idx, &ang_per_shell);

    let n0sh = reference_shell_occupations_gfn2(&nshell_per_atom, elem_idx, &ang_per_shell);

    // Build ao2sh and ao2at mappings
    let nao = h0.nrows();
    let nshell_total: usize = nshell_per_atom.iter().sum();
    let mut ao2sh = vec![0usize; nao];
    let mut ao2at = vec![0usize; nao];
    let mut ao_offset = 0;
    let mut shell_offset = 0;
    for iat in 0..nat {
        let nsh = nshell_per_atom[iat];
        for ish in 0..nsh {
            let l = ang_per_shell[shell_offset + ish];
            let nao_shell = 2 * l + 1;
            for i in 0..nao_shell {
                ao2sh[ao_offset + i] = shell_offset + ish;
                ao2at[ao_offset + i] = iat;
            }
            ao_offset += nao_shell;
        }
        shell_offset += nsh;
    }

    // Compute multipole integrals from CGTO basis
    let (dipole_ints, quadrupole_ints) = multipole_integrals::build_multipole_integrals_gfn2(coords, elem_idx);

    // Precompute multipole interaction matrices
    let cn = compute_coordination_numbers(coords, elem_idx);
    let mrad = compute_multipole_radii(&cn, elem_idx);
    let (amat_sd, amat_dd, amat_sq) = build_multipole_interaction_matrices_0d(coords, &mrad);

    const MIXING: f64 = 0.3;

    let mut density = DMatrix::zeros(nao, nao);
    let mut qsh = DVector::zeros(nshell_total);

    for iter in 0..max_iter {
        // Build charge-only SCC Hamiltonian
        let mut h_scc = build_scc_hamiltonian_with_thirdorder_gfn2(
            &h0, &s, &gamma, &qsh, &nshell_per_atom, elem_idx, &ang_per_shell, &ao2sh
        );

        // Compute atomic multipoles from density matrix
        let dpat_mat = atomic_multipoles(&density, &dipole_ints, &ao2at, 3);
        let qpat_mat = atomic_multipoles(&density, &quadrupole_ints, &ao2at, 6);

        // Flatten to vectors
        let mut dpat = vec![0.0f64; 3 * nat];
        let mut qpat = vec![0.0f64; 6 * nat];
        for iat in 0..nat {
            for cmp in 0..3 { dpat[cmp + 3 * iat] = dpat_mat[(cmp, iat)]; }
            for cmp in 0..6 { qpat[cmp + 6 * iat] = qpat_mat[(cmp, iat)]; }
        }

        // Compute atomic charges from shell charges
        let qat_vec = atomic_charges(&qsh, &nshell_per_atom);
        let mut qat = vec![0.0f64; nat];
        for iat in 0..nat { qat[iat] = qat_vec[iat]; }

        // Compute multipole potentials
        let (vat, vdp, vqp) = compute_multipole_potentials(
            &qat, &dpat, &qpat, &amat_sd, &amat_dd, &amat_sq, elem_idx
        );

        // Add to Hamiltonian
        add_multipole_to_h1(&mut h_scc, &s, &dipole_ints, &quadrupole_ints, &vat, &vdp, &vqp, &ao2at);

        let (eigenvalues, eigenvectors) = solve_gevp(&h_scc, &s);

        let n_occ = n_electrons / 2;
        let mut new_density = DMatrix::zeros(nao, nao);
        for i in 0..n_occ {
            let c = eigenvectors.column(i);
            new_density += 2.0 * c * c.transpose();
        }

        let new_qsh = shell_charges(&new_density, &s, &ao2sh, &n0sh);

        let max_diff = (&new_qsh - &qsh).abs().max();
        if iter % 10 == 0 {
            println!("Iter {}: max charge diff = {:.6e}", iter, max_diff);
        }
        if max_diff < conv_thresh {
            println!("SCF converged in {} iterations", iter + 1);
            // Recompute eigenvalues with converged charges for consistency with tblite
            let mut h_final = build_scc_hamiltonian_with_thirdorder_gfn2(
                &h0, &s, &gamma, &new_qsh, &nshell_per_atom, elem_idx, &ang_per_shell, &ao2sh
            );
            let dpat_mat = atomic_multipoles(&new_density, &dipole_ints, &ao2at, 3);
            let qpat_mat = atomic_multipoles(&new_density, &quadrupole_ints, &ao2at, 6);
            let mut dpat = vec![0.0f64; 3 * nat];
            let mut qpat = vec![0.0f64; 6 * nat];
            for iat in 0..nat {
                for cmp in 0..3 { dpat[cmp + 3 * iat] = dpat_mat[(cmp, iat)]; }
                for cmp in 0..6 { qpat[cmp + 6 * iat] = qpat_mat[(cmp, iat)]; }
            }
            let qat_vec = atomic_charges(&new_qsh, &nshell_per_atom);
            let mut qat = vec![0.0f64; nat];
            for iat in 0..nat { qat[iat] = qat_vec[iat]; }
            let (vat, vdp, vqp) = compute_multipole_potentials(
                &qat, &dpat, &qpat, &amat_sd, &amat_dd, &amat_sq, elem_idx
            );
            add_multipole_to_h1(&mut h_final, &s, &dipole_ints, &quadrupole_ints, &vat, &vdp, &vqp, &ao2at);
            let (eigenvalues_final, _) = solve_gevp(&h_final, &s);
            return (new_density, new_qsh, eigenvalues_final);
        }

        qsh = MIXING * &new_qsh + (1.0 - MIXING) * &qsh;
        density = MIXING * &new_density + (1.0 - MIXING) * &density;
    }

    println!("SCF did not converge in {} iterations", max_iter);
    (density, qsh, DVector::zeros(nao))
}

/// Solve generalized eigenvalue problem H*c = E*S*c
/// Returns (eigenvalues, eigenvectors)
pub fn solve_gevp(h: &DMatrix<f64>, s: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
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
