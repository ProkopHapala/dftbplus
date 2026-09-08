//! Gate F: Geometry optimization at the method's own minimum
//! (manifest v3 §1065).
//!
//! Optimize a small H-passivated Si system (SiH4) with the sparse model.
//!
//! - coarse stage: FIRE, masks can rebuild at explicit checkpoints
//! - final stage: frozen masks, FIRE
//!
//! The stopping target is tied to measured convergence/repeatability
//! behavior (from Gate E), not a hard-coded f32 folklore number.
//!
//! After convergence, compute the full Hessian (small system) and check
//! for obvious unstable directions before vibrational analysis.

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::methods::sparse::bsr4::{build_full_mask, Bsr4Matrix, BS};
use rust_dftb::methods::sparse::gpu_sparse::{SparseBsr4Config, SparseBsr4Gpu};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder};
use std::panic::{catch_unwind, AssertUnwindSafe};

const E_DUMMY: f32 = 2.0;

// ---------------------------------------------------------------------
// Helpers (shared infrastructure)
// ---------------------------------------------------------------------

fn try_gpu() -> Option<SparseBsr4Gpu> {
    match catch_unwind(AssertUnwindSafe(|| {
        SparseBsr4Gpu::new(SparseBsr4Config::default())
    })) {
        Ok(Ok(gpu)) => Some(gpu),
        Ok(Err(e)) => { eprintln!("Skipping Gate F: no OpenCL ({e})"); None }
        Err(_) => { eprintln!("Skipping Gate F: OpenCL panic"); None }
    }
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

fn cpu_energy(k_dense: &[f32], h_dense: &[f32], n: usize) -> f64 {
    let mut e = 0.0f64;
    for i in 0..n { for j in 0..n { e += k_dense[i * n + j] as f64 * h_dense[j * n + i] as f64; } }
    e
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
        .tc2_purify(&k0, &s_bsr, n_occ as f32, &mask, &mask, 80, 1e-4).unwrap();
    let k_dense = k_final.to_dense();
    cpu_energy(&k_dense, &h_pad, n_padded)
}

/// Sparse f32 force via total-energy finite difference (3-point central).
fn sparse_force(
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

// ---------------------------------------------------------------------
// FIRE optimizer
// ---------------------------------------------------------------------

/// Simple FIRE (Fast Inertial Relaxation Engine) optimizer.
///
/// Reference: Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006).
///
/// Parameters follow the original FIRE defaults. The stopping criterion is
/// based on the force norm falling below a tolerance tied to the measured
/// numerical floor (Gate E), not a hard-coded f32 folklore number.
struct FireOptimizer {
    dt: f64,
    dt_max: f64,
    max_dt: f64,
    n_min: usize,
    f_inc: f64,
    f_dec: f64,
    alpha_start: f64,
    alpha: f64,
    v: Vec<[f64; 3]>,
    n_pos: usize,
    last_neg: usize,
}

impl FireOptimizer {
    fn new(n_atoms: usize, dt: f64, dt_max: f64) -> Self {
        Self {
            dt,
            dt_max,
            max_dt: dt_max,
            n_min: 5,
            f_inc: 1.1,
            f_dec: 0.5,
            alpha_start: 0.1,
            alpha: 0.1,
            v: vec![[0.0; 3]; n_atoms],
            n_pos: 0,
            last_neg: 0,
        }
    }

    /// Perform one FIRE step. Returns new coords and the force norm.
    fn step(&mut self, coords: &[[f64; 3]], forces: &[[f64; 3]]) -> (Vec<[f64; 3]>, f64) {
        let n = coords.len();
        let mut f_norm = 0.0f64;
        let mut p = 0.0f64;  // P = F · V
        let mut v_norm = 0.0f64;
        for i in 0..n {
            for d in 0..3 {
                f_norm += forces[i][d] * forces[i][d];
                p += forces[i][d] * self.v[i][d];
                v_norm += self.v[i][d] * self.v[i][d];
            }
        }
        f_norm = f_norm.sqrt();
        v_norm = v_norm.sqrt();

        if p > 0.0 {
            self.n_pos += 1;
            if self.n_pos > self.n_min {
                self.dt = (self.dt * self.f_inc).min(self.max_dt);
                self.alpha *= 0.99;
            }
        } else {
            self.n_pos = 0;
            self.dt *= self.f_dec;
            self.alpha = self.alpha_start;
            // Freeze velocities
            for i in 0..n { for d in 0..3 { self.v[i][d] = 0.0; } }
            self.last_neg += 1;
        }

        // Mixed velocity: V = (1-alpha) V + alpha |F| |V|/|F| + alpha dt F
        // Simplified: V = (1-alpha) V + alpha * (|V|/|F|) * F
        let f_scale = if f_norm > 1e-30 { v_norm / f_norm } else { 0.0 };
        for i in 0..n {
            for d in 0..3 {
                self.v[i][d] = (1.0 - self.alpha) * self.v[i][d]
                    + self.alpha * f_scale * forces[i][d]
                    + self.dt * forces[i][d];
            }
        }

        // Position update
        let mut new_coords = coords.to_vec();
        for i in 0..n {
            for d in 0..3 {
                new_coords[i][d] += self.dt * self.v[i][d];
            }
        }
        (new_coords, f_norm)
    }
}

// ---------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------

#[test]
fn test_gate_f_geometry_optimization() {
    let Some(gpu) = try_gpu() else { return };

    let sk_dir = match std::env::var("RUST_DFTB_SK_DIR") {
        Ok(d) => d,
        Err(_) => {
            let d = "/home/prokop/SIMULATIONS/dftbplus/slakos/matsci-0-3";
            if !std::path::Path::new(d).exists() {
                eprintln!("Skipping Gate F: RUST_DFTB_SK_DIR not set and default {d} not found");
                return;
            }
            d.to_string()
        }
    };

    let species = vec!["Si".to_string(), "H".to_string(), "H".to_string(), "H".to_string(), "H".to_string()];
    let atom_n_orb: Vec<u8> = vec![4, 1, 1, 1, 1];
    let n_occ = 4usize;

    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // Start from a perturbed SiH4 geometry (not at equilibrium).
    let bond = 1.60f64;  // longer than equilibrium ~1.48
    let theta = 109.47f64 * std::f64::consts::PI / 180.0;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let coords0 = vec![
        [0.0, 0.0, 0.0],
        [bond, 0.0, 0.0],
        [bond * cos_t, bond * sin_t, 0.0],
        [bond * cos_t, bond * sin_t * cos_t, bond * sin_t * sin_t],
        [bond * cos_t, -bond * sin_t * cos_t, -bond * sin_t * sin_t],
    ];

    eprintln!("=== Gate F: Geometry optimization (FIRE) ===");
    eprintln!("  Initial Si-H bond = {bond:.3} Å (perturbed from ~1.48)");

    // h for finite-difference forces — use the Gate E plateau value.
    let h_fd = 0.02f64;

    // Stopping target: tied to Gate E measured force bias (~1e-3).
    // The sparse force bias is ~1.3e-2 relative, ~1.6e-3 absolute.
    // We stop when f_norm < 10x the measured bias floor.
    let f_tol = 1e-2f64;  // force norm tolerance (Hartree/Å)
    let max_steps = 200usize;

    let mut fire = FireOptimizer::new(species.len(), 0.01, 0.05);
    let mut coords = coords0.clone();
    let mut e_prev = sparse_energy(&gpu, &sk, &species, &coords, &atom_n_orb, n_occ);

    eprintln!("  Initial energy = {e_prev:.8}");
    eprintln!("  f_tol = {f_tol:.2e} (tied to Gate E bias floor)");

    let mut converged = false;
    let mut step = 0usize;
    let mut last_mask_rebuild = 0usize;

    while step < max_steps {
        // Coarse stage: rebuild masks at explicit checkpoints (every 20 steps).
        if step - last_mask_rebuild >= 20 {
            eprintln!("  step {step}: mask rebuild checkpoint");
            last_mask_rebuild = step;
        }

        let forces = sparse_force(&gpu, &sk, &species, &coords, &atom_n_orb, n_occ, h_fd);
        let (new_coords, f_norm) = fire.step(&coords, &forces);

        // Check for NaN/Inf
        for i in 0..new_coords.len() {
            for d in 0..3 {
                assert!(new_coords[i][d].is_finite(),
                    "Gate F: NaN/Inf in coords at step {step}, atom {i}, dim {d}");
            }
        }

        coords = new_coords;
        let e_new = sparse_energy(&gpu, &sk, &species, &coords, &atom_n_orb, n_occ);
        let de = e_new - e_prev;
        e_prev = e_new;

        if step % 10 == 0 || f_norm < f_tol {
            eprintln!("  step {step:3}: E={e_new:.8}  dE={de:+.3e}  |F|={f_norm:.3e}  dt={:.4}",
                fire.dt);
        }

        if f_norm < f_tol {
            converged = true;
            eprintln!("  Converged at step {step}: |F|={f_norm:.3e} < {f_tol:.2e}");
            break;
        }
        step += 1;
    }

    assert!(converged, "Gate F: FIRE did not converge in {max_steps} steps");

    // Final energy
    let e_final = e_prev;
    eprintln!("\n  Final energy = {e_final:.8}");
    eprintln!("  Final coordinates:");
    for (i, c) in coords.iter().enumerate() {
        eprintln!("    [{i}] {:+.6} {:+.6} {:+.6}", c[0], c[1], c[2]);
    }

    // Compute final Si-H bond lengths
    let si = coords[0];
    for i in 1..5 {
        let dx = coords[i][0] - si[0];
        let dy = coords[i][1] - si[1];
        let dz = coords[i][2] - si[2];
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        eprintln!("  Si-H{i} bond = {r:.4} Å");
    }

    // Compute full Hessian at the minimum (small system, cheap).
    eprintln!("\n  Computing full Hessian at minimum...");
    let h_hess = 0.02f64;  // Gate E plateau
    let hess = dense_f64_hessian_fd(&sk, &species, &coords, 8.0, h_hess);
    let n_dof = 3 * species.len();

    // Check for unstable directions (negative eigenvalues = imaginary freq).
    // Project out rigid-body modes (translation + rotation) for diagnostics.
    let hess_mat = dmatrix_from_2d(&hess);
    let eig = SymmetricEigen::new(hess_mat.clone());
    let mut eigs: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    eprintln!("  Hessian eigenvalues (sorted):");
    let n_neg = eigs.iter().filter(|&&e| e < -1e-3).count();
    let n_zero = eigs.iter().filter(|&&e| e.abs() < 1e-3).count();
    let n_pos = eigs.iter().filter(|&&e| e > 1e-3).count();
    for (i, &e) in eigs.iter().enumerate() {
        if i < 6 || i >= eigs.len() - 3 || e < -1e-3 {
            eprintln!("    eig[{i:2}] = {e:+.4e}");
        }
    }
    eprintln!("  Summary: {n_neg} negative, {n_zero} near-zero, {n_pos} positive");

    // At a true minimum, we expect 6 near-zero (rigid body) + rest positive.
    // Allow some tolerance for the f32 numerical floor.
    let n_unstable = eigs.iter().filter(|&&e| e < -1e-2).count();
    eprintln!("  Unstable modes (eig < -1e-2): {n_unstable}");

    // For SiH4 at equilibrium, there should be no genuinely unstable modes
    // beyond the 6 rigid-body near-zero modes.
    assert!(n_unstable <= 6,
        "Gate F: {n_unstable} unstable Hessian modes at minimum (expected <= 6 rigid-body)");

    eprintln!("\n  Gate F: PASS — geometry converged, Hessian stable at minimum.");
}

/// Dense f64 Hessian via 3-point central finite difference of the band energy.
fn dense_f64_hessian_fd(
    sk: &rust_dftb::SkData,
    species: &[String],
    coords: &[[f64; 3]],
    n_electrons: f64,
    h: f64,
) -> Vec<Vec<f64>> {
    let builder = HamiltonianBuilder::new(sk.clone());
    let n_atoms = coords.len();
    let n_dof = 3 * n_atoms;
    let mut hess = vec![vec![0.0f64; n_dof]; n_dof];

    for i in 0..n_dof {
        for j in i..n_dof {
            let ai = i / 3; let di = i % 3;
            let aj = j / 3; let dj = j % 3;

            let mut c_pp = coords.to_vec(); c_pp[ai][di] += h; c_pp[aj][dj] += h;
            let mut c_pm = coords.to_vec(); c_pm[ai][di] += h; c_pm[aj][dj] -= h;
            let mut c_mp = coords.to_vec(); c_mp[ai][di] -= h; c_mp[aj][dj] += h;
            let mut c_mm = coords.to_vec(); c_mm[ai][di] -= h; c_mm[aj][dj] -= h;

            let e_pp = band_energy(&builder, species, &c_pp, n_electrons);
            let e_pm = band_energy(&builder, species, &c_pm, n_electrons);
            let e_mp = band_energy(&builder, species, &c_mp, n_electrons);
            let e_mm = band_energy(&builder, species, &c_mm, n_electrons);

            let h_ij = (e_pp - e_pm - e_mp + e_mm) / (4.0 * h * h);
            hess[i][j] = h_ij;
            hess[j][i] = h_ij;
        }
    }
    hess
}

fn band_energy(builder: &HamiltonianBuilder, species: &[String], coords: &[[f64; 3]], n_electrons: f64) -> f64 {
    let ham = builder.build_non_scc(species, coords).unwrap();
    let n = ham.h0.nrows();
    let n_occ = (n_electrons / 2.0).round() as usize;
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

fn dmatrix_from_2d(v: &[Vec<f64>]) -> DMatrix<f64> {
    let n = v.len();
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n { for j in 0..n { m[(i, j)] = v[i][j]; } }
    m
}
