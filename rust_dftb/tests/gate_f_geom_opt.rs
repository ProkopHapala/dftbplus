//! Gate F: Geometry optimization at the method's own minimum
//! (manifest v3 §1065 / review G1.6–G1.7, G3.3).
//!
//! FIRE on SiH4 using **sparse SCC E_tot = E_el + E_rep** and **analytic**
//! forces (`D=2K`, `W=2KHK` → CPU `compute_forces_from_dw`). Not FD of
//! `Tr(K H0)`. Previous unphysical path collapsed Si–H 1.60 → 0.93 Å.
//!
//! After FIRE, every Si–H must lie in 1.40–1.55 Å. Hessian / spectra are
//! Gate G/H — not this test.

use rust_dftb::methods::dftb::forces::Forces;
use rust_dftb::methods::dftb::gamma::GammaTable;
use rust_dftb::methods::sparse::gpu_sparse::SparseBsr4Gpu;
use rust_dftb::methods::sparse::SparseDftb;
use rust_dftb::methods::sparse::harness::{require_sih_sk_dir, require_sparse_gpu};
use rust_dftb::methods::sparse::scc::{eval_sparse_energy_forces, SparseDftbEnergy};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder, SkData};
use std::io::Write;

fn sih_bonds(coords: &[[f64; 3]]) -> Vec<f64> {
    let si = coords[0];
    (1..coords.len()).map(|i| {
        let dx = coords[i][0] - si[0];
        let dy = coords[i][1] - si[1];
        let dz = coords[i][2] - si[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }).collect()
}

fn mean_sih(coords: &[[f64; 3]]) -> f64 {
    let r = sih_bonds(coords);
    r.iter().sum::<f64>() / r.len() as f64
}

// fn flatten — unused after eval_sparse_energy_forces (Gate G shares that path).
fn f_norm(forces: &[[f64; 3]]) -> f64 {
    let mut s = 0.0f64;
    for f in forces {
        s += f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
    }
    s.sqrt()
}

#[allow(dead_code)] // leftover: allocating scc.rs path; Gate F now uses SparseDftb
fn eval_step(
    gpu: &SparseBsr4Gpu,
    builder: &HamiltonianBuilder,
    sk: &SkData,
    sk_dir: &str,
    species: &[String],
    coords: &[[f64; 3]],
    atom_n_orb: &[u8],
    n_occ: usize,
    q0: &[f64],
    codes: &[u8],
    gamma: &GammaTable,
    q_warm: Option<&[f64]>,
) -> (SparseDftbEnergy, Forces) {
    eval_sparse_energy_forces(
        gpu, builder, sk, sk_dir, species, coords, atom_n_orb, n_occ as f32, q0, codes, gamma, q_warm, 80,
    ).unwrap_or_else(|err| panic!("Gate F sparse eval failed: {err}"))
}

// ---------------------------------------------------------------------
// FIRE optimizer
// ---------------------------------------------------------------------

/// Simple FIRE (Fast Inertial Relaxation Engine) optimizer.
///
/// Reference: Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006).
#[allow(dead_code)] // leftover FIRE; production step is SparseDftb::fire_step
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

#[allow(dead_code)]
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
        let mut f_n = 0.0f64;
        let mut p = 0.0f64;
        let mut v_norm = 0.0f64;
        for i in 0..n {
            for d in 0..3 {
                f_n += forces[i][d] * forces[i][d];
                p += forces[i][d] * self.v[i][d];
                v_norm += self.v[i][d] * self.v[i][d];
            }
        }
        f_n = f_n.sqrt();
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
            for i in 0..n { for d in 0..3 { self.v[i][d] = 0.0; } }
            self.last_neg += 1;
        }

        let f_scale = if f_n > 1e-30 { v_norm / f_n } else { 0.0 };
        for i in 0..n {
            for d in 0..3 {
                self.v[i][d] = (1.0 - self.alpha) * self.v[i][d]
                    + self.alpha * f_scale * forces[i][d]
                    + self.dt * forces[i][d];
            }
        }
        let mut new_coords = coords.to_vec();
        for i in 0..n {
            for d in 0..3 {
                new_coords[i][d] += self.dt * self.v[i][d];
            }
        }
        (new_coords, f_n)
    }
}

#[test]
fn test_gate_f_geometry_optimization() {
    let Some(_gpu) = require_sparse_gpu() else { return };
    std::env::set_var("RUST_DFTB_SPARSE_ALGEBRA_VERBOSE", "0");
    let sk_dir = require_sih_sk_dir();
    let species = vec!["Si".to_string(), "H".to_string(), "H".to_string(), "H".to_string(), "H".to_string()];
    let q0 = vec![4.0, 1.0, 1.0, 1.0, 1.0];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    let bond0 = 1.60f64;
    let theta = 109.47f64 * std::f64::consts::PI / 180.0;
    let (c, s) = (theta.cos(), theta.sin());
    let coords0 = vec![
        [0.0, 0.0, 0.0],
        [bond0, 0.0, 0.0],
        [bond0 * c, bond0 * s, 0.0],
        [bond0 * c, bond0 * s * c, bond0 * s * s],
        [bond0 * c, -bond0 * s * c, -bond0 * s * s],
    ];

    eprintln!("=== Gate F: SparseDftb FIRE on SCC E_tot + analytic F (D=2K, W=2KHK) ===");
    eprintln!("  start mean Si–H = {:.4} Å (perturbed; physical ~1.48)", mean_sih(&coords0));
    eprintln!("  one solver: SparseDftb::scc + fire_step (not eval_sparse_energy_forces)");

    let mut eng = SparseDftb::new(sk.clone(), &sk_dir, species.clone(), coords0.clone())
        .unwrap_or_else(|e| panic!("Gate F SparseDftb::new: {e}"));
    let f_tol = 1e-3f64;
    let max_steps = 120usize;
    let mut e_prev = f64::INFINITY;
    let mut last_f = Forces::zeros(species.len());
    let mut last_e = SparseDftbEnergy {
        e_h0: 0.0, e_scc: 0.0, e_el: 0.0, e_rep: 0.0, e_tot: 0.0,
        q: q0.clone(), tr_ks: 0.0, r_i: 0.0, n_scc: 0, tc2_iters: 0,
        k_pad: vec![], h_scc_pad: vec![], v: vec![],
        r_scc: 0.0, r_h: f32::NAN,
        purify_status: rust_dftb::methods::sparse::gpu_sparse::PurifyStatus::Failed,
    };
    let mut converged = false;
    let mut step = 0usize;

    let log_path = {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../debug/sparse_review");
        std::fs::create_dir_all(&p).unwrap_or_else(|e| panic!("Gate F: cannot create {p:?}: {e}"));
        p.join("gate_f_sih4.csv")
    };
    let mut log = std::fs::File::create(&log_path)
        .unwrap_or_else(|e| panic!("Gate F: cannot write {}: {e}", log_path.display()));
    writeln!(log, "step,E_tot,E_el,E_rep,f_norm,mean_SiH").unwrap();

    while step < max_steps {
        eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("Gate F SCC step {step}: {e}"));
        last_e = eng.last_energy().clone();
        last_f = eng.forces().unwrap_or_else(|e| panic!("Gate F forces step {step}: {e}"));
        let coords = eng.coords().to_vec();
        let fnorm = f_norm(&last_f.forces);
        let rmean = mean_sih(&coords);
        let bonds = sih_bonds(&coords);
        for (k, &r) in bonds.iter().enumerate() {
            if !r.is_finite() || r < 1.20 || r > 1.90 {
                panic!(
                    "Gate F step {step}: Si–H{} = {r:.4} Å left 1.20–1.90 (collapse/explode). \
                     E_tot={:.8} E_rep={:.8} |F|={fnorm:.3e}. Old Tr(KH0) FIRE went to 0.93 Å.",
                    k + 1, last_e.e_tot, last_e.e_rep
                );
            }
        }
        if !last_e.e_tot.is_finite() || !fnorm.is_finite() {
            panic!("Gate F step {step}: non-finite E={} |F|={fnorm}", last_e.e_tot);
        }
        let de = if e_prev.is_finite() { last_e.e_tot - e_prev } else { 0.0 };
        eprintln!(
            "  step {step:3}: E_tot={:.8}  E_el={:.8}  E_rep={:.8}  dE={de:+.3e}  |F|={fnorm:.3e}  mean Si–H={rmean:.4} Å  n_scc={}",
            last_e.e_tot, last_e.e_el, last_e.e_rep, last_e.n_scc
        );
        writeln!(
            log, "{step},{:.12},{:.12},{:.12},{fnorm:.8},{rmean:.8}",
            last_e.e_tot, last_e.e_el, last_e.e_rep
        ).unwrap();
        let _ = log.flush();
        e_prev = last_e.e_tot;

        if fnorm < f_tol {
            converged = true;
            eprintln!("  Converged at step {step}: |F|={fnorm:.3e} < {f_tol:.2e}");
            break;
        }
        eng.fire_step(0.0).unwrap_or_else(|e| panic!("Gate F fire_step {step}: {e}"));
        step += 1;
    }

    let coords = eng.coords().to_vec();
    eprintln!("  log: {}", log_path.display());
    eprintln!("REVIEW: {}", log_path.display());
    eprintln!("  Final E_tot={:.10}  E_el={:.10}  E_rep={:.10}  |F|={:.3e}",
        last_e.e_tot, last_e.e_el, last_e.e_rep, f_norm(&last_f.forces));
    eprintln!("  Final coordinates:");
    for (i, c) in coords.iter().enumerate() {
        eprintln!("    [{i}] {:+.6} {:+.6} {:+.6}", c[0], c[1], c[2]);
    }
    let bonds = sih_bonds(&coords);
    for (k, &r) in bonds.iter().enumerate() {
        eprintln!("  Si-H{} = {r:.4} Å", k + 1);
        assert!(
            (1.40..=1.55).contains(&r),
            "Gate F: Si–H{} = {r:.4} Å is unphysical (window 1.40–1.55 Å; collapsed 0.93 Å is a hard fail, G1.6)",
            k + 1
        );
    }
    assert!(last_e.e_rep.abs() > 1e-4, "Gate F: E_rep={:.3e} ~0 — repulsive missing", last_e.e_rep);
    assert!(
        converged,
        "Gate F: FIRE did not reach |F|<{f_tol:.2e} in {max_steps} steps (last |F|={:.3e}, mean Si–H={:.4} Å). \
         Bonds are in window if the asserts above passed — still not a stationary point.",
        f_norm(&last_f.forces), mean_sih(&coords)
    );

    let last_q = last_e.q.clone();
    let mut xyz_p = coords.clone();
    let mut xyz_m = coords.clone();
    let h_probe = 0.01f64;
    xyz_p[1][0] += h_probe;
    xyz_m[1][0] -= h_probe;
    eng.set_q(&last_q).unwrap();
    eng.set_coords(&xyz_p).unwrap_or_else(|e| panic!("Gate F probe +h: {e}"));
    eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("Gate F probe +h SCC: {e}"));
    let e_p = eng.energy().unwrap();
    eng.set_q(&last_q).unwrap();
    eng.set_coords(&xyz_m).unwrap_or_else(|e| panic!("Gate F probe -h: {e}"));
    eng.scc(80, 1e-5).unwrap_or_else(|e| panic!("Gate F probe -h SCC: {e}"));
    let e_m = eng.energy().unwrap();
    let de_p = e_p - last_e.e_tot;
    let de_m = e_m - last_e.e_tot;
    eprintln!("  curvature probe H1 x ±{h_probe} Å: ΔE(+)= {de_p:+.4e}  ΔE(−)= {de_m:+.4e}");
    assert!(
        de_p > -1e-6 && de_m > -1e-6,
        "Gate F: energy decreased along H1-x probe (ΔE+={de_p:.3e} ΔE-={de_m:.3e}) — not a local minimum"
    );
    eprintln!("  Gate F: SparseDftb FIRE stopped with Si–H in 1.40–1.55 Å (awaiting USER confirmation; not marked done).");
}

// --- G1.7 unphysical path (Tr(K H0) + FD force). Not called. Do not delete. ---
#[allow(dead_code, unused_imports, unused_variables, unreachable_code)]
mod old_tr_kh0_fd {
use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::methods::sparse::bsr4::{build_full_mask, Bsr4Matrix, BS};
use rust_dftb::methods::sparse::gpu_sparse::SparseBsr4Gpu;
use rust_dftb::{HamiltonianBuilder};

const E_DUMMY: f32 = 2.0;

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

fn cpu_energy(k_dense: &[f32], h_dense: &[f32], n: usize) -> f64 {
    let mut e = 0.0f64;
    for i in 0..n { for j in 0..n { e += k_dense[i * n + j] as f64 * h_dense[j * n + i] as f64; } }
    e
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
}
