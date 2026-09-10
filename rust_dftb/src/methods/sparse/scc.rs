//! Sparse DFTB energy and SCC loop (review G3.1 / G3.2).
//!
//! **Physics reference / Gate G3 path.** Allocates NS buffers per call.
//! Production MD/FIRE must use `SparseDftb` (`sparse_dftb.rs`): one compile,
//! persistent BSR buffers, no `Kernel::builder` in the SCC/MD loop.
//!
//! Host-roundtrip NS + TC2 on BSR4 (the device NS inverse is N4-wrong by ~200×
//! and is not used in this module). CPU H0/S assembly and CPU γ·Δq → Hscc.
//! The algebra stays sparse-purified K, not a dense eigenproblem inside the loop.

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::forces::{repulsive_energy, Forces};
use crate::methods::dftb::gamma::GammaTable;
use crate::methods::dftb::hamiltonian::HamiltonianBuilder;
use crate::methods::dftb::sk_data::SkData;
use crate::methods::sparse::bsr4::{
    build_full_mask, pad_physical_to_bsr4, Bsr4Matrix, BS,
};
use crate::methods::sparse::gpu_sparse::{SparseBsr4Gpu, TC2_TRACE_TOL};
use crate::methods::sparse::sparse_forces::sparse_analytic_forces;
use crate::qmqm::shifts::compute_intra_shifts;

/// Dummy onsite (Hartree) for padded H 2p slots. Same as Gate D.
pub const E_DUMMY: f32 = 2.0;

#[derive(Debug, Clone)]
pub struct SparseDftbEnergy {
    pub e_h0: f64,   // 2 Tr(K H0) — closed-shell band energy of H0
    pub e_scc: f64,  // ½ Δq · V
    pub e_el: f64,   // e_h0 + e_scc
    pub e_rep: f64,
    pub e_tot: f64,
    pub q: Vec<f64>,
    pub tr_ks: f32,
    pub r_i: f32,
    pub n_scc: usize,
    pub tc2_iters: usize,
    /// Padded BSR4 density kernel K at the last purify (row-major f32). Empty if unused.
    pub k_pad: Vec<f32>,
    /// Padded H_scc that produced `k_pad`. Empty if unused.
    pub h_scc_pad: Vec<f32>,
    /// Atom shifts V from the purified charges (same as energy).
    pub v: Vec<f64>,
    /// RMS(q_Mulliken − q used to build H) after the stationary-state repair. 0 if unused.
    pub r_scc: f64,
    /// ||HKS − SKH||_F of the returned (K, H). NaN if not computed.
    pub r_h: f32,
}

/// Tr(A B) for row-major f32, accumulated in f64.
pub fn trace_ab(a: &[f32], b: &[f32], n: usize) -> f64 {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n * n);
    let mut t = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            t += a[i * n + j] as f64 * b[j * n + i] as f64;
        }
    }
    t
}

/// Dummy-safe Hscc = H0 + ½ S ⊙ (V_A + V_B) on physical orbitals only.
pub fn apply_shift_padded(
    h0: &[f32],
    s: &[f32],
    v: &[f64],
    atom_n_orb: &[u8],
) -> Vec<f32> {
    let n_atom = atom_n_orb.len();
    let n = n_atom * BS;
    assert_eq!(h0.len(), n * n);
    assert_eq!(s.len(), n * n);
    assert_eq!(v.len(), n_atom);
    let mut h = h0.to_vec();
    for a in 0..n_atom {
        let na = atom_n_orb[a] as usize;
        for b in 0..n_atom {
            let nb = atom_n_orb[b] as usize;
            let avg = 0.5 * (v[a] + v[b]);
            if avg == 0.0 {
                continue;
            }
            for i in 0..na {
                for j in 0..nb {
                    let p = (a * BS + i) * n + (b * BS + j);
                    h[p] += (avg as f32) * s[p];
                }
            }
        }
    }
    h
}

/// `Hscc = H0 + ½ S ⊙ (V_A+V_B)` into a preallocated buffer (`out` may alias `h0` only if copied first).
pub fn apply_shift_padded_into(h0: &[f32], s: &[f32], v: &[f64], atom_n_orb: &[u8], out: &mut [f32]) {
    let n_atom = atom_n_orb.len();
    let n = n_atom * BS;
    assert_eq!(h0.len(), n * n);
    assert_eq!(s.len(), n * n);
    assert_eq!(out.len(), n * n);
    assert_eq!(v.len(), n_atom);
    out.copy_from_slice(h0);
    for a in 0..n_atom {
        let na = atom_n_orb[a] as usize;
        for b in 0..n_atom {
            let nb = atom_n_orb[b] as usize;
            let avg = 0.5 * (v[a] + v[b]);
            if avg == 0.0 { continue; }
            for i in 0..na {
                for j in 0..nb {
                    let p = (a * BS + i) * n + (b * BS + j);
                    out[p] += (avg as f32) * s[p];
                }
            }
        }
    }
}

/// One NS + K0 + TC2 purification of a frozen H (host-roundtrip GPU).
/// Recomputes Z≈S⁻¹. Prefer `purify_h_with_z` inside an SCC loop (S is fixed).
pub fn purify_h(
    gpu: &SparseBsr4Gpu,
    h_pad: &[f32],
    s_pad: &[f32],
    n_atom: usize,
    nocc: f32,
) -> Result<(Bsr4Matrix, f32, f32, usize)> {
    let mask = build_full_mask(n_atom);
    let s = Bsr4Matrix::from_dense(n_atom, s_pad, &mask)?;
    let (z, rz, z_iters) = gpu.newton_schulz_inverse(&s, &mask, &mask, 50, 1e-5, 5)?;
    let verbose = crate::methods::sparse::gpu_sparse::algebra_verbose();
    if verbose {
        eprintln!("  NS: {z_iters} iters, R_Z={rz:.3e}");
    }
    purify_h_with_z(gpu, h_pad, &s, &z, nocc, &mask)
}

/// K0+TC2 of `h_pad` using a **precomputed** Z. S does not change during SCC.
pub fn purify_h_with_z(
    gpu: &SparseBsr4Gpu,
    h_pad: &[f32],
    s: &Bsr4Matrix,
    z: &Bsr4Matrix,
    nocc: f32,
    mask: &(Vec<u32>, Vec<u32>),
) -> Result<(Bsr4Matrix, f32, f32, usize)> {
    let n_atom = s.n_atom;
    let h = Bsr4Matrix::from_dense(n_atom, h_pad, mask)?;
    let (emin, emax) = gpu.spectral_bounds(&h, z, mask, 0.1)?;
    if !emin.is_finite() || !emax.is_finite() || emax <= emin {
        return Err(DftbError::InvalidInput(format!(
            "purify_h: bad spectral bounds emin={emin} emax={emax} (bounds are Gershgorin of ZH, not ZHZ)"
        )));
    }
    let verbose = crate::methods::sparse::gpu_sparse::algebra_verbose();
    if verbose {
        eprintln!("  bounds (ZH Gershgorin): emin={emin:.4} emax={emax:.4}");
    }
    let k0 = gpu.build_k0(&h, s, z, mask, mask, emin, emax)?;
    let (k, r_i, tr, iters, _) = gpu.tc2_purify(&k0, s, nocc, mask, mask, 80, 1e-4)?;
    if verbose {
        eprintln!("  TC2: {iters} iters, R_I={r_i:.3e}, Tr(KS)={tr:.6} (Nocc={nocc})");
    }
    if (tr - nocc).abs() > TC2_TRACE_TOL {
        return Err(DftbError::InvalidInput(format!(
            "purify_h: Tr(KS)={tr} far from Nocc={nocc} (tol={TC2_TRACE_TOL})"
        )));
    }
    Ok((k, r_i, tr, iters))
}

/// G3.1: E_el = 2 Tr(K H0) of a non-SCC purified K, plus E_rep. No SCC shift.
pub fn energy_non_scc(
    gpu: &SparseBsr4Gpu,
    h0_phys: &[f64],
    s_phys: &[f64],
    atom_n_orb: &[u8],
    nocc: f32,
    sk_dir: &str,
    species: &[String],
    coords: &[[f64; 3]],
) -> Result<SparseDftbEnergy> {
    let n_atom = atom_n_orb.len();
    let (h_pad, s_pad, _) = pad_physical_to_bsr4(h0_phys, s_phys, atom_n_orb, E_DUMMY);
    let n = n_atom * BS;
    let (k, r_i, tr, tc2_iters) = purify_h(gpu, &h_pad, &s_pad, n_atom, nocc)?;
    let k_dense = k.to_dense();
    let e_h0 = 2.0 * trace_ab(&k_dense, &h_pad, n);
    let e_rep = repulsive_energy(sk_dir, species, coords)?;
    if !e_h0.is_finite() || !e_rep.is_finite() {
        panic!("energy_non_scc: non-finite E_h0={e_h0} E_rep={e_rep}");
    }
    Ok(SparseDftbEnergy {
        e_h0, e_scc: 0.0, e_el: e_h0, e_rep, e_tot: e_h0 + e_rep,
        q: vec![0.0; n_atom], tr_ks: tr, r_i, n_scc: 0, tc2_iters,
        k_pad: k_dense, h_scc_pad: h_pad, v: vec![0.0; n_atom],
        r_scc: 0.0, r_h: f32::NAN,
    })
}

fn energy_from_k(
    k_dense: &[f32],
    h0_pad: &[f32],
    n: usize,
    q: &[f64],
    q0: &[f64],
    coords: &[[f64; 3]],
    species_code: &[u8],
    gamma: &GammaTable,
    e_rep: f64,
    tr: f32,
    r_i: f32,
    n_scc: usize,
    tc2_iters: usize,
    h_scc: Vec<f32>,
    r_scc: f64,
    r_h: f32,
) -> SparseDftbEnergy {
    let n_atom = q.len();
    let dq: Vec<f64> = q.iter().zip(q0).map(|(a, b)| a - b).collect();
    let mut v = vec![0.0f64; n_atom];
    compute_intra_shifts(coords, species_code, &dq, gamma, &mut v);
    let e_h0 = 2.0 * trace_ab(k_dense, h0_pad, n);
    let e_scc = 0.5 * dq.iter().zip(v.iter()).map(|(d, vi)| d * vi).sum::<f64>();
    let e_el = e_h0 + e_scc;
    SparseDftbEnergy {
        e_h0, e_scc, e_el, e_rep, e_tot: e_el + e_rep,
        q: q.to_vec(), tr_ks: tr, r_i, n_scc, tc2_iters,
        k_pad: k_dense.to_vec(), h_scc_pad: h_scc, v, r_scc, r_h,
    }
}

fn mulliken_q(gpu: &SparseBsr4Gpu, k: &Bsr4Matrix, s: &Bsr4Matrix, mask: &(Vec<u32>, Vec<u32>), n_atom: usize, nocc: f32, tr: f32, it: usize) -> Result<Vec<f64>> {
    let t_ks = gpu.matmul_masked_bsym(k, s, mask)?;
    let q_f32 = gpu.mulliken(&t_ks)?;
    if q_f32.len() != n_atom {
        return Err(DftbError::InvalidInput(format!(
            "Mulliken len {} != n_atom {n_atom}", q_f32.len()
        )));
    }
    let q: Vec<f64> = q_f32.iter().map(|&x| x as f64).collect();
    let qsum: f64 = q.iter().sum();
    let n_elec: f64 = 2.0 * nocc as f64;
    if (qsum - n_elec).abs() > 0.5 {
        return Err(DftbError::InvalidInput(format!(
            "SCC iter {it}: sum(q)={qsum:.6} != N_elec={n_elec} (Tr(KS)={tr})"
        )));
    }
    Ok(q)
}

/// G3.2: self-consistent sparse SCC. No dense eigenproblem in the loop.
///
/// `q0` must be valence electron counts (Si=4, H=1). The matsci onsite-line
/// parser currently reads trailing fields (Si=0, H≈0.49) — that is a shared
/// input bug, not a physical SK occupation; this fixture bypasses it.
///
/// Z≈S⁻¹ is computed **once** (S is fixed at a geometry). On convergence a
/// stationary repair rebuilds H from the output charges so K, H, V, q, W
/// are one electronic state (second review §3.2).
pub fn run_sparse_scc(
    gpu: &SparseBsr4Gpu,
    h0_phys: &[f64],
    s_phys: &[f64],
    atom_n_orb: &[u8],
    nocc: f32,
    q0: &[f64],
    species_code: &[u8],
    coords: &[[f64; 3]],
    gamma: &GammaTable,
    sk_dir: &str,
    species: &[String],
    mix: f64,
    scc_tol: f64,
    max_scc: usize,
    q_init: Option<&[f64]>,
) -> Result<SparseDftbEnergy> {
    let n_atom = atom_n_orb.len();
    assert_eq!(q0.len(), n_atom);
    assert_eq!(species_code.len(), n_atom);
    let (h0_pad, s_pad, dummy) = pad_physical_to_bsr4(h0_phys, s_phys, atom_n_orb, E_DUMMY);
    let n = n_atom * BS;
    let mask = build_full_mask(n_atom);
    let s_bsr = Bsr4Matrix::from_dense(n_atom, &s_pad, &mask)?;
    let (z, rz, z_iters) = gpu.newton_schulz_inverse(&s_bsr, &mask, &mask, 50, 1e-5, 5)?;
    let verbose = crate::methods::sparse::gpu_sparse::algebra_verbose();
    if verbose {
        eprintln!("  NS (once per geometry): {z_iters} iters, R_Z={rz:.3e}");
    }
    let e_rep = repulsive_energy(sk_dir, species, coords)?;
    let mut q = match q_init {
        Some(qi) => {
            assert_eq!(qi.len(), n_atom, "q_init len {} != n_atom {n_atom}", qi.len());
            qi.to_vec()
        }
        None => q0.to_vec(),
    };
    let mut v = vec![0.0f64; n_atom];
    let mut last = SparseDftbEnergy {
        e_h0: 0.0, e_scc: 0.0, e_el: 0.0, e_rep, e_tot: 0.0,
        q: q.clone(), tr_ks: 0.0, r_i: 0.0, n_scc: 0, tc2_iters: 0,
        k_pad: vec![], h_scc_pad: vec![], v: vec![0.0; n_atom],
        r_scc: 0.0, r_h: f32::NAN,
    };
    let mut rms_prev = f64::INFINITY;
    for it in 0..max_scc {
        let dq: Vec<f64> = q.iter().zip(q0).map(|(a, b)| a - b).collect();
        compute_intra_shifts(coords, species_code, &dq, gamma, &mut v);
        let h_scc = apply_shift_padded(&h0_pad, &s_pad, &v, atom_n_orb);
        let (k, r_i, tr, tc2_iters) = purify_h_with_z(gpu, &h_scc, &s_bsr, &z, nocc, &mask)?;
        let q_new = mulliken_q(gpu, &k, &s_bsr, &mask, n_atom, nocc, tr, it)?;
        let k_dense = k.to_dense();
        let t_dense = gpu.matmul_masked_bsym(&k, &s_bsr, &mask)?.to_dense();
        let mut dummy_occ = 0.0f64;
        for &d in &dummy {
            dummy_occ += t_dense[d * n + d].abs() as f64;
        }
        let mut rms = 0.0f64;
        let mut max_dq = 0.0f64;
        for a in 0..n_atom {
            let d = q_new[a] - q[a];
            rms += d * d;
            max_dq = max_dq.max(d.abs());
        }
        rms = (rms / n_atom as f64).sqrt();
        last = energy_from_k(
            &k_dense, &h0_pad, n, &q_new, q0, coords, species_code, gamma,
            e_rep, tr, r_i, it + 1, tc2_iters, h_scc, rms, f32::NAN,
        );
        if verbose {
            eprintln!(
                "  [sparse SCC] iter {it:3}  rms={rms:.3e}  max|dq|={max_dq:.3e}  \
                 E_el={:.8}  E_rep={e_rep:.8}  E_tot={:.8}  dummy_KS={dummy_occ:.3e}  R_I={r_i:.3e}",
                last.e_el, last.e_tot
            );
        }
        if !last.e_el.is_finite() || !e_rep.is_finite() {
            panic!("sparse SCC non-finite energy at iter {it}: E_el={} E_rep={e_rep}", last.e_el);
        }
        if rms < scc_tol {
            // Stationary repair: K, H, V, q from the same output charges (review §3.2).
            let dq_out: Vec<f64> = q_new.iter().zip(q0).map(|(a, b)| a - b).collect();
            compute_intra_shifts(coords, species_code, &dq_out, gamma, &mut v);
            let h_fin = apply_shift_padded(&h0_pad, &s_pad, &v, atom_n_orb);
            let (k_fin, r_i_f, tr_f, tc2_f) = purify_h_with_z(gpu, &h_fin, &s_bsr, &z, nocc, &mask)?;
            let q_fin = mulliken_q(gpu, &k_fin, &s_bsr, &mask, n_atom, nocc, tr_f, it)?;
            let mut r_fin = 0.0f64;
            for a in 0..n_atom {
                let d = q_fin[a] - q_new[a];
                r_fin += d * d;
            }
            r_fin = (r_fin / n_atom as f64).sqrt();
            let h_bsr = Bsr4Matrix::from_dense(n_atom, &h_fin, &mask)?;
            let r_h = gpu.hamiltonian_residual(&h_bsr, &k_fin, &s_bsr, &mask)?;
            let k_fin_d = k_fin.to_dense();
            last = energy_from_k(
                &k_fin_d, &h0_pad, n, &q_fin, q0, coords, species_code, gamma,
                e_rep, tr_f, r_i_f, it + 1, tc2_f, h_fin, r_fin, r_h,
            );
            eprintln!(
                "  [sparse SCC] finalize  r_scc={r_fin:.3e}  R_H={r_h:.3e}  Tr(KS)={tr_f:.6}  R_I={r_i_f:.3e}  E_tot={:.8}",
                last.e_tot
            );
            if !r_h.is_finite() {
                panic!("sparse SCC finalize: non-finite R_H={r_h} at iter {it}");
            }
            return Ok(last);
        }
        for a in 0..n_atom {
            q[a] = (1.0 - mix) * q[a] + mix * q_new[a];
        }
        if it > 2 && rms > rms_prev * 2.0 && rms > 1e-2 {
            return Err(DftbError::InvalidInput(format!(
                "sparse SCC diverging at iter {it}: rms={rms:.3e} (prev {rms_prev:.3e})"
            )));
        }
        rms_prev = rms;
    }
    Err(DftbError::InvalidInput(format!(
        "sparse SCC did not converge in {max_scc} iters, last rms={rms_prev:.3e} E_tot={:.8} Tr(KS)={} n_scc={}",
        last.e_tot, last.tr_ks, last.n_scc
    )))
}

/// One sparse SCC + analytic force at `coords`. Leftover allocating path.
/// Production / gates: `SparseDftb::scc` + `forces`.
pub fn eval_sparse_energy_forces(
    gpu: &SparseBsr4Gpu,
    builder: &HamiltonianBuilder,
    sk: &SkData,
    sk_dir: &str,
    species: &[String],
    coords: &[[f64; 3]],
    atom_n_orb: &[u8],
    nocc: f32,
    q0: &[f64],
    species_code: &[u8],
    gamma: &GammaTable,
    q_init: Option<&[f64]>,
    max_scc: usize,
) -> Result<(SparseDftbEnergy, Forces)> {
    let ham = builder.build_non_scc(species, coords)?;
    let n = ham.h0.nrows();
    let h0: Vec<f64> = (0..n * n).map(|i| ham.h0[(i / n, i % n)]).collect();
    let s: Vec<f64> = (0..n * n).map(|i| ham.s[(i / n, i % n)]).collect();
    let e = run_sparse_scc(
        gpu, &h0, &s, atom_n_orb, nocc, q0, species_code, coords, gamma,
        sk_dir, species, 0.5, 1e-5, max_scc, q_init,
    )?;
    let f = sparse_analytic_forces(
        sk, species, coords, atom_n_orb, &e.k_pad, &e.h_scc_pad, &e.q, q0, sk_dir,
    )?;
    Ok((e, f))
}
