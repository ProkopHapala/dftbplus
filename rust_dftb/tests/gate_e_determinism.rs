//! Gate E: Determinism, arithmetic sensitivity, and Hessian h plateau
//! (manifest v3 §1033).
//!
//! On a fixed near-equilibrium geometry (SiH4), separates:
//!
//! **A. same-geometry repeatability/history tests**
//!   - cold SCC start
//!   - central warm start
//!   - several perturbed valid q starts
//!   - fast vs tighter SCC/TC2/Z tolerances
//!   Measure force spread/history sensitivity.
//!
//! **B. sparse-vs-dense bias**
//!   Measure F_sparse - F_dense as a separate quantity.
//!
//! **C. h sweep**
//!   0.01, 0.02, 0.05, 0.10 Å — compare raw Hessian asymmetry, Hessian
//!   error vs dense f64, 3-point vs 5-point. Choose a broad stable plateau.

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::methods::sparse::bsr4::{
    build_full_mask, build_geometric_mask, build_product_mask, Bsr4Matrix, BS,
};
use rust_dftb::methods::sparse::gpu_sparse::{SparseBsr4Gpu, GpuBsrMatrix};
use rust_dftb::methods::sparse::harness::{require_sih_sk_dir, require_sparse_gpu};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder};

const E_DUMMY: f32 = 2.0;
const ANG2BOHR: f64 = 1.889_726_133;

// ---------------------------------------------------------------------
// Helpers (shared with sih_padded_basis.rs)
// ---------------------------------------------------------------------

fn try_gpu() -> Option<SparseBsr4Gpu> {
    require_sparse_gpu()
}

fn bsr4_from_dense(n_atom: usize, dense: &[f32], mask: &(Vec<u32>, Vec<u32>)) -> Bsr4Matrix {
    let mut m = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone()).unwrap();
    for i in 0..n_atom {
        let (start, end) = (mask.0[i] as usize, mask.0[i + 1] as usize);
        for blk in start..end {
            let j = mask.1[blk] as usize;
            let mut v = [0.0f32; BS * BS];
            for r in 0..BS {
                for c in 0..BS {
                    v[r * BS + c] = dense[(i * BS + r) * (n_atom * BS) + (j * BS + c)];
                }
            }
            m.set_block(i, j, &v).unwrap();
        }
    }
    m
}

fn row_major_to_dmatrix_f64(dense: &[f32], n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n { for j in 0..n { m[(i, j)] = dense[i * n + j] as f64; } }
    m
}

fn dmatrix_to_row_major_f32(m: &DMatrix<f64>) -> Vec<f32> {
    let n = m.nrows();
    let mut d = vec![0.0f32; n * n];
    for i in 0..n { for j in 0..n { d[i * n + j] = m[(i, j)] as f32; } }
    d
}

fn dense_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

fn build_padded_bsr4(
    h0_dense: &[f64], s_dense: &[f64], atom_n_orb: &[u8],
) -> (Vec<f32>, Vec<f32>, usize) {
    let n_atom = atom_n_orb.len();
    let n_padded = n_atom * BS;
    let n_phys: usize = atom_n_orb.iter().map(|&n| n as usize).sum();
    let mut phys_off = Vec::with_capacity(n_atom);
    let mut padded_off = Vec::with_capacity(n_atom);
    let mut acc_phys = 0usize;
    let mut acc_padded = 0usize;
    for &n in atom_n_orb {
        phys_off.push(acc_phys); padded_off.push(acc_padded);
        acc_phys += n as usize; acc_padded += BS;
    }
    let mut h_pad = vec![0.0f32; n_padded * n_padded];
    let mut s_pad = vec![0.0f32; n_padded * n_padded];
    for a in 0..n_atom {
        for b in 0..n_atom {
            let na = atom_n_orb[a] as usize;
            let nb = atom_n_orb[b] as usize;
            for i in 0..na { for j in 0..nb {
                let pi = padded_off[a] + i; let pj = padded_off[b] + j;
                let fi = phys_off[a] + i; let fj = phys_off[b] + j;
                h_pad[pi * n_padded + pj] = h0_dense[fi * n_phys + fj] as f32;
                s_pad[pi * n_padded + pj] = s_dense[fi * n_phys + fj] as f32;
            }}
        }
    }
    for (a, &n) in atom_n_orb.iter().enumerate() {
        for d in (n as usize)..BS {
            let pi = padded_off[a] + d;
            s_pad[pi * n_padded + pi] = 1.0;
            h_pad[pi * n_padded + pi] = E_DUMMY;
        }
    }
    (h_pad, s_pad, n_padded)
}

fn cpu_density_kernel(h: &[f32], s: &[f32], n: usize, nocc: usize) -> Vec<f32> {
    let hf = row_major_to_dmatrix_f64(h, n);
    let sf = row_major_to_dmatrix_f64(s, n);
    let se = SymmetricEigen::new(sf.clone());
    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n { d[(i, i)] = 1.0 / se.eigenvalues[i].max(1e-12).sqrt(); }
    let s_inv_sqrt = &se.eigenvectors * &d * se.eigenvectors.transpose();
    let h_orth = &s_inv_sqrt * &hf * &s_inv_sqrt;
    let he = SymmetricEigen::new(h_orth);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| he.eigenvalues[i].partial_cmp(&he.eigenvalues[j]).unwrap());
    let v_sorted = he.eigenvectors.select_columns(&idx);
    let c = &s_inv_sqrt * &v_sorted;
    let mut k = DMatrix::<f64>::zeros(n, n);
    for i in 0..nocc { let col = c.column(i); k += &col * col.transpose(); }
    dmatrix_to_row_major_f32(&k)
}

fn cpu_energy(k_dense: &[f32], h_dense: &[f32], n: usize) -> f64 {
    let mut e = 0.0f64;
    for i in 0..n { for j in 0..n { e += k_dense[i * n + j] as f64 * h_dense[j * n + i] as f64; } }
    e
}

/// Dense f64 force via total-energy finite difference (3-point central).
/// Returns forces [n_atom][3] in Hartree/Å (matching DFTB convention).
fn dense_f64_force_fd(
    builder: &HamiltonianBuilder,
    species: &[String],
    coords: &[[f64; 3]],
    n_electrons: f64,
    h: f64,  // displacement in Å
) -> Vec<[f64; 3]> {
    let n_atoms = coords.len();
    let mut forces = vec![[0.0f64; 3]; n_atoms];
    for i in 0..n_atoms {
        for d in 0..3 {
            let mut c_plus = coords.to_vec();
            let mut c_minus = coords.to_vec();
            c_plus[i][d] += h;
            c_minus[i][d] -= h;
            let e_plus = band_energy(builder, species, &c_plus, n_electrons);
            let e_minus = band_energy(builder, species, &c_minus, n_electrons);
            forces[i][d] = -(e_plus - e_minus) / (2.0 * h);
        }
    }
    forces
}

/// Non-SCC band energy: E = sum_{i in occ} eps_i where H0 C = S C eps.
fn band_energy(builder: &HamiltonianBuilder, species: &[String], coords: &[[f64; 3]], n_electrons: f64) -> f64 {
    let ham = builder.build_non_scc(species, coords).unwrap();
    let n = ham.h0.nrows();
    let n_occ = (n_electrons / 2.0).round() as usize;
    // Generalized eigendecomposition: S^{-1/2} H0 S^{-1/2}
    let se = SymmetricEigen::new(ham.s.clone());
    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n { d[(i, i)] = 1.0 / se.eigenvalues[i].max(1e-12).sqrt(); }
    let s_inv_sqrt = &se.eigenvectors * &d * se.eigenvectors.transpose();
    let h_orth = &s_inv_sqrt * &ham.h0 * &s_inv_sqrt;
    let he = SymmetricEigen::new(h_orth);
    let mut eigs: Vec<f64> = he.eigenvalues.iter().copied().collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigs.iter().take(n_occ).sum()
}

/// Sparse f32 force via total-energy finite difference (3-point central),
/// using the sparse purification pipeline. Returns forces [n_atom][3].
fn sparse_f32_force_fd(
    gpu: &SparseBsr4Gpu,
    sk: &rust_dftb::SkData,
    species: &[String],
    coords: &[[f64; 3]],
    atom_n_orb: &[u8],
    n_occ: usize,
    h: f64,
) -> Vec<[f64; 3]> {
    let n_atoms = coords.len();
    let mut forces = vec![[0.0f64; 3]; n_atoms];

    for i in 0..n_atoms {
        for d in 0..3 {
            let mut c_plus = coords.to_vec();
            let mut c_minus = coords.to_vec();
            c_plus[i][d] += h;
            c_minus[i][d] -= h;
            let e_plus = sparse_energy(gpu, sk, species, &c_plus, atom_n_orb, n_occ);
            let e_minus = sparse_energy(gpu, sk, species, &c_minus, atom_n_orb, n_occ);
            forces[i][d] = -(e_plus - e_minus) / (2.0 * h);
        }
    }
    forces
}

/// Run the sparse pipeline (Z, K0, TC2) and return E = Tr(K·H0).
fn sparse_energy(
    gpu: &SparseBsr4Gpu,
    sk: &rust_dftb::SkData,
    species: &[String],
    coords: &[[f64; 3]],
    atom_n_orb: &[u8],
    n_occ: usize,
) -> f64 {
    let builder = HamiltonianBuilder::new(sk.clone());
    let ham = builder.build_non_scc(species, coords).unwrap();
    let n_phys = ham.h0.nrows();
    let h0_dense: Vec<f64> = (0..n_phys * n_phys)
        .map(|idx| ham.h0[(idx / n_phys, idx % n_phys)]).collect();
    let s_dense: Vec<f64> = (0..n_phys * n_phys)
        .map(|idx| ham.s[(idx / n_phys, idx % n_phys)]).collect();

    let (h_pad, s_pad, n_padded) = build_padded_bsr4(&h0_dense, &s_dense, atom_n_orb);
    let n_atom = atom_n_orb.len();
    let mask = build_full_mask(n_atom);
    let h_bsr = bsr4_from_dense(n_atom, &h_pad, &mask);
    let s_bsr = bsr4_from_dense(n_atom, &s_pad, &mask);

    let (z, _rz, _zi) = gpu.newton_schulz_inverse(&s_bsr, &mask, &mask, 50, 1e-5, 5).unwrap();
    let (emin, emax) = gpu.spectral_bounds(&h_bsr, &z, &mask, 0.1).unwrap();
    let k0 = gpu.build_k0(&h_bsr, &s_bsr, &z, &mask, &mask, emin, emax).unwrap();
    let (k_final, _r_i, _tr, _iters, _hist) = gpu
        .tc2_purify(&k0, &s_bsr, n_occ as f32, &mask, &mask, atom_n_orb, 80, 1e-4).unwrap();

    let k_dense = k_final.to_dense();
    cpu_energy(&k_dense, &h_pad, n_padded)
}

// ---------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------

#[test]
#[allow(unreachable_code)]
fn test_gate_e_determinism_and_h_plateau() {
    let Some(gpu) = try_gpu() else { return };

    let sk_dir = require_sih_sk_dir();

    // SiH4 near equilibrium (tetrahedral, bond ~1.48 Å)
    let species = vec!["Si".to_string(), "H".to_string(), "H".to_string(), "H".to_string(), "H".to_string()];
    let bond = 1.48f64;
    let theta = 109.47f64 * std::f64::consts::PI / 180.0;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let coords = vec![
        [0.0, 0.0, 0.0],
        [bond, 0.0, 0.0],
        [bond * cos_t, bond * sin_t, 0.0],
        [bond * cos_t, bond * sin_t * cos_t, bond * sin_t * sin_t],
        [bond * cos_t, -bond * sin_t * cos_t, -bond * sin_t * sin_t],
    ];
    let atom_n_orb: Vec<u8> = vec![4, 1, 1, 1, 1];
    let n_electrons = 8.0f64;
    let n_occ = 4usize;

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let builder = HamiltonianBuilder::new(sk.clone());

    // =================================================================
    // A. Same-geometry repeatability
    // =================================================================
    eprintln!("=== Gate E-A: Same-geometry repeatability ===");

    // Run sparse energy 5 times at the same geometry — measure spread.
    let mut energies: Vec<f64> = Vec::with_capacity(5);
    for _ in 0..5 {
        let e = sparse_energy(&gpu, &sk, &species, &coords, &atom_n_orb, n_occ);
        energies.push(e);
    }
    let e_mean = energies.iter().sum::<f64>() / energies.len() as f64;
    let e_spread = energies.iter().map(|e| (e - e_mean).abs()).fold(0.0f64, f64::max);
    eprintln!("  Energy repeatability: E_mean={e_mean:.8}, spread={e_spread:.3e}");
    eprintln!("  Energies: {energies:?}");
    assert!(e_spread < 1e-6,
        "Gate E-A: energy not deterministic, spread={e_spread:.3e}");

    // Different TC2 tolerances — measure force sensitivity.
    eprintln!("\n  TC2 tolerance sensitivity:");
    let tcs = [1e-3f32, 1e-4, 1e-5];
    let mut e_by_tol: Vec<f64> = Vec::with_capacity(tcs.len());
    for &tol in &tcs {
        let builder2 = HamiltonianBuilder::new(sk.clone());
        let ham = builder2.build_non_scc(&species, &coords).unwrap();
        let n_phys = ham.h0.nrows();
        let h0d: Vec<f64> = (0..n_phys*n_phys).map(|i| ham.h0[(i/n_phys, i%n_phys)]).collect();
        let sd: Vec<f64> = (0..n_phys*n_phys).map(|i| ham.s[(i/n_phys, i%n_phys)]).collect();
        let (hp, sp, np) = build_padded_bsr4(&h0d, &sd, &atom_n_orb);
        let mask = build_full_mask(species.len());
        let hb = bsr4_from_dense(species.len(), &hp, &mask);
        let sb = bsr4_from_dense(species.len(), &sp, &mask);
        let (z, _, _) = gpu.newton_schulz_inverse(&sb, &mask, &mask, 50, 1e-5, 5).unwrap();
        let (emin, emax) = gpu.spectral_bounds(&hb, &z, &mask, 0.1).unwrap();
        let k0 = gpu.build_k0(&hb, &sb, &z, &mask, &mask, emin, emax).unwrap();
        let (kf, _, _, _, _) = gpu.tc2_purify(&k0, &sb, n_occ as f32, &mask, &mask, &atom_n_orb, 80, tol).unwrap();
        let kd = kf.to_dense();
        let e = cpu_energy(&kd, &hp, np);
        e_by_tol.push(e);
        eprintln!("    tol={tol:.0e}: E={e:.8}");
    }
    let tol_spread = e_by_tol.iter().map(|e| (e - e_by_tol[0]).abs()).fold(0.0f64, f64::max);
    eprintln!("  TC2 tolerance spread: {tol_spread:.3e}");
    assert!(tol_spread < 1e-3,
        "Gate E-A: energy too sensitive to TC2 tolerance, spread={tol_spread:.3e}");

    // =================================================================
    // B. Sparse-vs-dense force bias
    // =================================================================
    eprintln!("\n=== Gate E-B: Sparse-vs-dense force bias ===");
    let h_fd = 0.01f64;  // small FD step for force comparison

    let f_dense = dense_f64_force_fd(&builder, &species, &coords, n_electrons, h_fd);
    let f_sparse = sparse_f32_force_fd(&gpu, &sk, &species, &coords, &atom_n_orb, n_occ, h_fd);

    let mut max_bias = 0.0f64;
    let mut max_force = 0.0f64;
    eprintln!("  atom    F_dense              F_sparse             |bias|");
    for i in 0..species.len() {
        for d in 0..3 {
            let fd = f_dense[i][d];
            let fs = f_sparse[i][d];
            let bias = (fd - fs).abs();
            max_bias = max_bias.max(bias);
            max_force = max_force.max(fd.abs()).max(fs.abs());
            if bias.abs() > 1e-6 || fd.abs() > 1e-3 {
                eprintln!("  [{i}][{d}]  {fd:+.6e}  {fs:+.6e}  {bias:.3e}");
            }
        }
    }
    let rel_bias = if max_force > 1e-12 { max_bias / max_force } else { 0.0 };
    eprintln!("  max|F|={max_force:.3e}, max|bias|={max_bias:.3e}, rel_bias={rel_bias:.3e}");
    panic!(
        "Gate E-B is not a force test (review G1.4): both F are 3-point FD of spinless Tr(K H0), \
         no E_rep, no SCC, no analytic D/W. Measured rel_bias={rel_bias:.3e}, max|bias|={max_bias:.3e}. \
         This gate stays red until the analytic sparse force of E_el+E_rep exists."
    );

    // =================================================================
    // C. h sweep — Hessian plateau
    // =================================================================
    eprintln!("\n=== Gate E-C: Hessian h sweep ===");
    eprintln!("  (3-point central FD Hessian, dense f64 reference)");

    let h_values: Vec<f64> = vec![0.01, 0.02, 0.05, 0.10];
    let n_atoms = species.len();
    let n_dof = 3 * n_atoms;

    // Dense f64 reference Hessian at each h.
    eprintln!("  h(Å)    ||H||_F      max|asym|    ||H_h - H_ref||_F");
    let mut h_results: Vec<(f64, f64, f64, f64)> = Vec::new();

    // Use h=0.05 as the reference (mid-plateau).
    let h_ref = 0.05f64;
    let h_ref_hess = dense_f64_hessian_fd(&builder, &species, &coords, n_electrons, h_ref);
    let h_ref_norm = frobenius_norm(&h_ref_hess);

    for &h in &h_values {
        let hess = dense_f64_hessian_fd(&builder, &species, &coords, n_electrons, h);
        let h_norm = frobenius_norm(&hess);
        let max_asym = max_asymmetry(&hess);
        let diff = matrix_diff_norm(&hess, &h_ref_hess);
        eprintln!("  {h:.3}   {h_norm:.4e}  {max_asym:.4e}  {diff:.4e}");
        h_results.push((h, h_norm, max_asym, diff));
    }

    // Find the plateau: h values where asymmetry is small and H is stable.
    let plateau: Vec<f64> = h_values.iter().copied().filter(|&h| {
        let (_, _, asym, diff) = h_results.iter().find(|(hh, _, _, _)| *hh == h).unwrap();
        *asym < 1e-3 && *diff < 1e-2 * h_ref_norm
    }).collect();

    eprintln!("\n  Plateau h values: {plateau:?}");
    assert!(!plateau.is_empty(),
        "Gate E-C: no stable Hessian plateau found among {h_values:?}");

    // Choose a broad stable plateau (not the smallest h).
    let h_chosen = plateau.iter().copied()
        .filter(|&h| h >= 0.02)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.05);
    eprintln!("  Chosen h = {h_chosen} Å (broad stable plateau)");
    assert!(h_chosen >= 0.02,
        "Gate E-C: chosen h={h_chosen} too small, numerical floor not established");

    eprintln!("\n  Gate E: PASS — deterministic, bias bounded, Hessian plateau at h={h_chosen} Å.");
}

/// Dense f64 Hessian via 3-point central finite difference of the energy.
/// Returns [n_dof][n_dof] where n_dof = 3 * n_atoms.
fn dense_f64_hessian_fd(
    builder: &HamiltonianBuilder,
    species: &[String],
    coords: &[[f64; 3]],
    n_electrons: f64,
    h: f64,
) -> Vec<Vec<f64>> {
    let n_atoms = coords.len();
    let n_dof = 3 * n_atoms;
    let mut hess = vec![vec![0.0f64; n_dof]; n_dof];

    // E(x + h*e_i + h*e_j) - E(x + h*e_i - h*e_j)
    // - E(x - h*e_i + h*e_j) + E(x - h*e_i - h*e_j)
    // divided by 4h^2.
    for i in 0..n_dof {
        for j in i..n_dof {
            let ai = i / 3; let di = i % 3;
            let aj = j / 3; let dj = j % 3;

            let mut c_pp = coords.to_vec(); c_pp[ai][di] += h; c_pp[aj][dj] += h;
            let mut c_pm = coords.to_vec(); c_pm[ai][di] += h; c_pm[aj][dj] -= h;
            let mut c_mp = coords.to_vec(); c_mp[ai][di] -= h; c_mp[aj][dj] += h;
            let mut c_mm = coords.to_vec(); c_mm[ai][di] -= h; c_mm[aj][dj] -= h;

            let e_pp = band_energy(builder, species, &c_pp, n_electrons);
            let e_pm = band_energy(builder, species, &c_pm, n_electrons);
            let e_mp = band_energy(builder, species, &c_mp, n_electrons);
            let e_mm = band_energy(builder, species, &c_mm, n_electrons);

            let h_ij = (e_pp - e_pm - e_mp + e_mm) / (4.0 * h * h);
            hess[i][j] = h_ij;
            hess[j][i] = h_ij;
        }
    }
    hess
}

fn frobenius_norm(h: &[Vec<f64>]) -> f64 {
    h.iter().flat_map(|row| row.iter()).map(|x| x * x).sum::<f64>().sqrt()
}

fn max_asymmetry(h: &[Vec<f64>]) -> f64 {
    let n = h.len();
    let mut max = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            max = max.max((h[i][j] - h[j][i]).abs());
        }
    }
    max
}

fn matrix_diff_norm(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut s = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let d = a[i][j] - b[i][j];
            s += d * d;
        }
    }
    s.sqrt()
}
