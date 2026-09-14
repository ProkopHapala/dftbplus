//! TEMP diagnostic (2026-09-14, revised after source review): separate
//! measurement error from matrix error on the converged sparse K.
//!
//! 1. Recompute 2·Tr(K·H0) in f64 on the host (device trace check).
//! 2. TRUE generalized McWeeny  K' = 3·(KS)·K − 2·(KS)·(KS)·K  in f64
//!    throughout (earlier version ran the wrong map 2K−KSK and cast
//!    intermediates to f32 — those conclusions are void).
//!
//! On R10 the mask is COMPLETE (deg 330/330) → products are untruncated.
//!
//!   cargo test --release --test sparse_f64check -- --ignored --nocapture
//! Env: RUST_DFTB_SK_DIR, F64CHECK_XYZ (default si_sphere_R10), F64CHECK_RK.

use rust_dftb::io::parse_xyz;
use rust_dftb::load_sk_for_species;
use rust_dftb::methods::sparse::bsr4::Bsr4Matrix;
use rust_dftb::methods::sparse::harness::require_sih_sk_dir;
use rust_dftb::methods::sparse::sparse_dftb::{SparseDftb, SparseDftbConfig};

/// f64 host trace: Σ_ij tr_4(A_ij · B_ji) over the (symmetric) masks.
fn trace_ab_f64(av: &[f64], am: &(Vec<u32>, Vec<u32>), bv: &[f64], bm: &(Vec<u32>, Vec<u32>)) -> f64 {
    let n = am.0.len() - 1;
    let mut tr = 0.0f64;
    for i in 0..n {
        for idx in am.0[i] as usize..am.0[i + 1] as usize {
            let j = am.1[idx] as usize;
            let (b0, b1) = (bm.0[j] as usize, bm.0[j + 1] as usize);
            let Some(bi) = bm.1[b0..b1].iter().position(|&c| c == i as u32).map(|p| b0 + p)
                else { continue };
            let a = &av[idx * 16..idx * 16 + 16];
            let b = &bv[bi * 16..bi * 16 + 16];
            for r in 0..4 { for c in 0..4 { tr += a[r * 4 + c] * b[c * 4 + r]; } }
        }
    }
    tr
}

/// Diagonal-block trace: Σ_i tr_4(A_ii).
fn trace_diag_f64(av: &[f64], am: &(Vec<u32>, Vec<u32>)) -> f64 {
    let n = am.0.len() - 1;
    let mut tr = 0.0f64;
    for i in 0..n {
        let (a, b) = (am.0[i] as usize, am.0[i + 1] as usize);
        if let Some(d) = am.1[a..b].iter().position(|&c| c == i as u32) {
            let blk = &av[(a + d) * 16..(a + d) * 16 + 16];
            for r in 0..4 { tr += blk[r * 4 + r]; }
        }
    }
    tr
}

/// f64 host SpGEMM restricted to `out` mask: C_ij = Σ_k A_ik B_kj.
fn spmm_f64(av: &[f64], am: &(Vec<u32>, Vec<u32>), bv: &[f64], bm: &(Vec<u32>, Vec<u32>),
            out: &(Vec<u32>, Vec<u32>)) -> Vec<f64> {
    let n = am.0.len() - 1;
    let mut c = vec![0.0f64; out.1.len() * 16];
    for i in 0..n {
        for idx in out.0[i] as usize..out.0[i + 1] as usize {
            let j = out.1[idx] as usize;
            let mut acc = [0.0f64; 16];
            for ai in am.0[i] as usize..am.0[i + 1] as usize {
                let k = am.1[ai] as usize;
                let (b0, b1) = (bm.0[k] as usize, bm.0[k + 1] as usize);
                let Some(bi) = bm.1[b0..b1].iter().position(|&c| c == j as u32).map(|p| b0 + p)
                    else { continue };
                let a = &av[ai * 16..ai * 16 + 16];
                let b = &bv[bi * 16..bi * 16 + 16];
                for p in 0..4 { for q in 0..4 { for r in 0..4 {
                    acc[p * 4 + q] += a[p * 4 + r] * b[r * 4 + q];
                }}}
            }
            c[idx * 16..idx * 16 + 16].copy_from_slice(&acc);
        }
    }
    c
}

#[test]
#[ignore]
fn f64_host_trace() {
    let sk_dir = require_sih_sk_dir();
    let xyz = std::env::var("F64CHECK_XYZ")
        .unwrap_or_else(|_| "../debug/nanocrystals/si_sphere_R10.xyz".into());
    let rk = std::env::var("F64CHECK_RK").ok().and_then(|v| v.parse().ok()).unwrap_or(12.0f64);
    let mol = parse_xyz(&xyz).unwrap_or_else(|e| panic!("parse_xyz {xyz}: {e}"));
    let sk = load_sk_for_species(&sk_dir, &mol.species)
        .unwrap_or_else(|e| panic!("load_sk_for_species: {e}"));

    let cfg = SparseDftbConfig {
        r_trunc_ang: Some(5.45), taper_w_ang: 0.3,
        r_k_ang: Some(rk), r_z_ang: Some(rk),
        max_deg_hs: Some(1024), max_deg_k: Some(1024), max_deg_z: Some(1024),
        tc2_tol: 1e-5, ns_tol: 1e-4, purifier_p: Some(false),
        ..Default::default()
    };
    let mut eng = SparseDftb::with_config(sk, &sk_dir, mol.species.clone(), mol.coords.clone(), cfg)
        .unwrap_or_else(|e| panic!("SparseDftb: {e}"));
    eng.scc(60, 1e-5).unwrap_or_else(|e| panic!("scc: {e}"));
    let e_dev = eng.last_energy().e_h0;
    let k = eng.k_bsr().unwrap();
    let km = (k.row_ptr.clone(), k.col_idx.clone());
    let kv: Vec<f64> = k.values.iter().map(|&x| x as f64).collect();
    let sm = (eng.s_bsr().row_ptr.clone(), eng.s_bsr().col_idx.clone());
    let sv: Vec<f64> = eng.s_bsr().values.iter().map(|&x| x as f64).collect();
    let hm = (eng.h_bsr().row_ptr.clone(), eng.h_bsr().col_idx.clone());
    let hv: Vec<f64> = eng.h_bsr().values.iter().map(|&x| x as f64).collect();

    let tr_f64 = trace_ab_f64(&kv, &km, &hv, &hm);
    eprintln!("device f32 E_band = {e_dev:.6}");
    eprintln!("host   f64 E_band = {:.6}", 2.0 * tr_f64);
    eprintln!("shift = {:+.4} mHa", (2.0 * tr_f64 - e_dev) * 1e3);

    // True generalized McWeeny on the K-form: K' = 3·KSK − 2·(KS)²K.
    // All products restricted to the (complete) mask → exact f64.
    let mut cur = kv;
    for step in 0..10 {
        let ks = spmm_f64(&cur, &km, &sv, &sm, &km);            // A = K·S
        let ak = spmm_f64(&ks, &km, &cur, &km, &km);            // KSK
        let aka_s = spmm_f64(&ak, &km, &sv, &sm, &km);          // KSK·S = (KS)²
        let ksk2 = spmm_f64(&aka_s, &km, &cur, &km, &km);       // (KS)²K
        let mut r_i = 0.0f64; let mut knorm = 0.0f64;
        let mut kp = cur.clone();
        for (i, (&xk, (&a1, &a2))) in cur.iter().zip(ak.iter().zip(ksk2.iter())).enumerate() {
            r_i += (a1 - xk).powi(2); knorm += xk * xk;
            kp[i] = 3.0 * a1 - 2.0 * a2;
        }
        r_i = (r_i / knorm).sqrt();
        let ks_new = spmm_f64(&kp, &km, &sv, &sm, &km);
        let tr_ks = trace_diag_f64(&ks_new, &km);
        let e_b = 2.0 * trace_ab_f64(&kp, &km, &hv, &hm);
        eprintln!("McWeeny step {step}: R_I(f64)={r_i:.3e}  Tr(KS)={tr_ks:.4}  E_band={e_b:.6}");
        cur = kp;
    }
    eprintln!("(DFTB+ R10 Energy H0 = -298.807668; DFTB+ R14 = -848.606688)");
}

/// A3 (manifest §15.12): frozen-input separation. Read the engine's
/// converged H_scc + S (padded dense), strip dummy lanes, solve the
/// IDENTICAL generalized eigenproblem in f64 via Cholesky+dsyevd, and
/// compare: exact band energy, exact P vs sparse K·S, and what the
/// sparse K would score under exact arithmetic.
#[test]
#[ignore]
fn frozen_input_dense_ref() {
    let sk_dir = require_sih_sk_dir();
    let xyz = std::env::var("F64CHECK_XYZ")
        .unwrap_or_else(|_| "../debug/nanocrystals/si_sphere_R10.xyz".into());
    let rk = std::env::var("F64CHECK_RK").ok().and_then(|v| v.parse().ok()).unwrap_or(12.0f64);
    let mol = parse_xyz(&xyz).unwrap_or_else(|e| panic!("parse_xyz {xyz}: {e}"));
    let sk = load_sk_for_species(&sk_dir, &mol.species)
        .unwrap_or_else(|e| panic!("load_sk_for_species: {e}"));

    let cfg = SparseDftbConfig {
        r_trunc_ang: Some(5.45), taper_w_ang: 0.3,
        r_k_ang: Some(rk), r_z_ang: Some(rk),
        max_deg_hs: Some(1024), max_deg_k: Some(1024), max_deg_z: Some(1024),
        tc2_tol: 1e-5, ns_tol: 1e-4, purifier_p: Some(false),
        dense_diag: Some(true),   // materialize h_scc_pad/k_pad
        ..Default::default()
    };
    let mut eng = SparseDftb::with_config(sk, &sk_dir, mol.species.clone(), mol.coords.clone(), cfg)
        .unwrap_or_else(|e| panic!("SparseDftb: {e}"));
    eng.scc(60, 1e-5).unwrap_or_else(|e| panic!("scc: {e}"));
    let e_dev = eng.last_energy().e_h0;

    let n_atom = mol.species.len();
    let norb = eng.atom_n_orb().to_vec();
    let n_pad = n_atom * 4;
    let n_phys: usize = norb.iter().map(|&n| n as usize).sum();
    eprintln!("n_atom={n_atom} n_pad={n_pad} n_phys={n_phys}");

    // padded dense → physical dense (drop dummy lanes)
    let strip = |pad: &[f64]| -> nalgebra::DMatrix<f64> {
        let mut m = nalgebra::DMatrix::zeros(n_phys, n_phys);
        let mut off_a = 0usize;
        for a in 0..n_atom {
            let mut off_b = 0usize;
            for b in 0..n_atom {
                for r in 0..norb[a] as usize {
                    for c in 0..norb[b] as usize {
                        m[(off_a + r, off_b + c)] = pad[(a * 4 + r) * n_pad + (b * 4 + c)];
                    }
                }
                off_b += norb[b] as usize;
            }
            off_a += norb[a] as usize;
        }
        m
    };
    let h_scc_pad: Vec<f64> = eng.h_scc_pad().iter().map(|&x| x as f64).collect();
    let h_scc = strip(&h_scc_pad);
    // S: rebuild padded dense from s_bsr
    let s_bsr = eng.s_bsr();
    let mut s_pad = vec![0.0f64; n_pad * n_pad];
    for i in 0..n_atom {
        for bi in s_bsr.row_ptr[i] as usize..s_bsr.row_ptr[i + 1] as usize {
            let j = s_bsr.col_idx[bi] as usize;
            for r in 0..4 { for c in 0..4 {
                s_pad[(i * 4 + r) * n_pad + j * 4 + c] = s_bsr.values[bi * 16 + r * 4 + c] as f64;
            }}
        }
    }
    let s_phys = strip(&s_pad);

    // Cholesky S = L Lᵀ;  H' = L⁻¹ H_scc L⁻ᵀ;  dsyevd.
    let chol = s_phys.clone().cholesky().expect("S not SPD");
    let l = chol.l();
    let linv = l.clone().try_inverse().expect("L singular");
    let hp = &linv * &h_scc * linv.transpose();
    let n = n_phys;
    let mut a = hp.as_slice().to_vec();
    let mut w = vec![0.0f64; n];
    let mut work_q = [0.0f64; 1];
    let mut iwork_q = [0i32; 1];
    let mut info = 0i32;
    unsafe {
        lapack::dsyevd(b'V', b'L', n as i32, &mut a, n as i32, &mut w,
                       &mut work_q, -1, &mut iwork_q, -1, &mut info);
    }
    assert_eq!(info, 0);
    let lwork = work_q[0] as usize;
    let liwork = iwork_q[0] as usize;
    let mut work = vec![0.0f64; lwork];
    let mut iwork = vec![0i32; liwork];
    unsafe {
        lapack::dsyevd(b'V', b'L', n as i32, &mut a, n as i32, &mut w,
                       &mut work, lwork as i32, &mut iwork, liwork as i32, &mut info);
    }
    assert_eq!(info, 0, "dsyevd failed info={info}");
    let nocc = 459;  // R10: 918 e-
    let e_band_exact: f64 = 2.0 * w[..nocc].iter().sum::<f64>();
    // E_H0 = 2Tr(P·H0): exact P = 2 C_occ C_occᵀ with C = L⁻ᵀ C'.
    let cprime = nalgebra::DMatrix::from_column_slice(n, n, &a);
    let c = linv.transpose() * cprime;
    let c_occ = c.columns(0, nocc);
    let p_exact = 2.0 * &c_occ * c_occ.transpose();
    // exact E_H0 needs physical H0 — strip h_bsr the same way
    let h0_bsr = eng.h_bsr();
    let mut h0_pad = vec![0.0f64; n_pad * n_pad];
    for i in 0..n_atom {
        for bi in h0_bsr.row_ptr[i] as usize..h0_bsr.row_ptr[i + 1] as usize {
            let j = h0_bsr.col_idx[bi] as usize;
            for r in 0..4 { for c in 0..4 {
                h0_pad[(i * 4 + r) * n_pad + j * 4 + c] = h0_bsr.values[bi * 16 + r * 4 + c] as f64;
            }}
        }
    }
    let h0_phys = strip(&h0_pad);
    let e_h0_exact = (&p_exact).component_mul(&h0_phys).sum();
    eprintln!("frozen H_scc eigh:  2Σ_occ ε = {e_band_exact:.6}   2Tr(P·H0) = {e_h0_exact:.6}");
    eprintln!("sparse device E_band = {e_dev:.6}");
    eprintln!("DFTB+ R10: Energy H0 = -298.807668  Band energy = -313.391208");

    // Engine convention: K IS the AO density matrix P (Mulliken from KS
    // diag blocks; idempotency K·S·K = K). Compare sparse K vs exact P.
    let k = eng.k_bsr().unwrap();
    let km = (k.row_ptr.clone(), k.col_idx.clone());
    let kv: Vec<f64> = k.values.iter().map(|&x| x as f64).collect();
    let sm = (s_bsr.row_ptr.clone(), s_bsr.col_idx.clone());
    let sv: Vec<f64> = s_bsr.values.iter().map(|&x| x as f64).collect();
    let hm = (h0_bsr.row_ptr.clone(), h0_bsr.col_idx.clone());
    let hv: Vec<f64> = h0_bsr.values.iter().map(|&x| x as f64).collect();
    let mut pd = nalgebra::DMatrix::zeros(n_phys, n_phys);
    for i in 0..n_atom {
        let oi = off_at(&norb, i);
        for bi in km.0[i] as usize..km.0[i + 1] as usize {
            let j = km.1[bi] as usize;
            let oj = off_at(&norb, j);
            for r in 0..norb[i] as usize { for c in 0..norb[j] as usize {
                pd[(oi + r, oj + c)] = kv[bi * 16 + r * 4 + c];
            }}
        }
    }
    let diff = &pd - &p_exact;
    eprintln!("||K_sparse − P_exact||_F = {:.4}   ||P_exact||_F = {:.4}",
              diff.norm(), p_exact.norm());
    eprintln!("Tr(K_sparse·S) physical = {:.4}  (nocc={nocc})", trace_ps(&pd, &s_phys));

    // ── True Z residual (independent f64 check of the NS fix) ──
    let z = eng.z_bsr().unwrap();
    let zm = (z.row_ptr.clone(), z.col_idx.clone());
    let zv: Vec<f64> = z.values.iter().map(|&x| x as f64).collect();
    let zs = spmm_f64(&zv, &zm, &sv, &sm, &km);
    let mut r_z = 0.0f64;
    for i in 0..n_atom {
        for bi in km.0[i] as usize..km.0[i + 1] as usize {
            let j = km.1[bi] as usize;
            for r in 0..4 { for c in 0..4 {
                let eye = if i == j && r == c { 1.0 } else { 0.0 };
                r_z += (zs[bi * 16 + r * 4 + c] - eye).powi(2);
            }}
        }
    }
    eprintln!("host f64 ||ZS−I||_F/√N_pad = {:.3e}", (r_z / n_pad as f64).sqrt());

    // ── Exact K on the M_K mask, injected into the f32 purifier ──
    // K IS the spin-free AO density C_occ·C_occᵀ = P_exact/2 (occupation 1).
    let mut k_inj = vec![0.0f32; km.1.len() * 16];
    for i in 0..n_atom {
        let oi = off_at(&norb, i);
        for bi in km.0[i] as usize..km.0[i + 1] as usize {
            let j = km.1[bi] as usize;
            let oj = off_at(&norb, j);
            for r in 0..norb[i] as usize { for c in 0..norb[j] as usize {
                k_inj[bi * 16 + r * 4 + c] = (p_exact[(oi + r, oj + c)] * 0.5) as f32;
            }}
        }
    }
    // Reference energy of the EXACT K under the engine's contraction:
    let k_inj64: Vec<f64> = k_inj.iter().map(|&x| x as f64).collect();
    eprintln!("E_band(exact K, masked to M_K) = {:.6}", 2.0 * trace_ab_f64(&k_inj64, &km, &hv, &hm));
    // Exact-K idempotency residual in f64: ||KSK−K||/||K||
    let ks_e = spmm_f64(&k_inj64, &km, &sv, &sm, &km);
    let ksk_e = spmm_f64(&ks_e, &km, &k_inj64, &km, &km);
    let mut num = 0.0f64; let mut den = 0.0f64;
    for i in 0..ksk_e.len() { num += (ksk_e[i] - k_inj64[i]).powi(2); den += k_inj64[i].powi(2); }
    eprintln!("exact-K f64 R_I = {:.3e}   (device floor was ~5e-4)", (num / den).sqrt());
    // and the sparse K's own f64 residual for comparison
    let ks_s = spmm_f64(&kv, &km, &sv, &sm, &km);
    let ksk_s = spmm_f64(&ks_s, &km, &kv, &km, &km);
    let mut num = 0.0f64; let mut den = 0.0f64;
    for i in 0..ksk_s.len() { num += (ksk_s[i] - kv[i]).powi(2); den += kv[i].powi(2); }
    eprintln!("sparse-K f64 R_I = {:.3e}", (num / den).sqrt());

    let km_mat = Bsr4Matrix { n_atom, row_ptr: km.0.clone(), col_idx: km.1.clone(), values: k_inj };
    eng.inject_k_bsr(&km_mat).unwrap();
    let (st, r_i, tr, it) = eng.purify_current().unwrap_or_else(|e| panic!("purify injected: {e}"));
    eprintln!("f32 purify from EXACT K: status={st:?} R_I={r_i:.3e} Tr={tr:.4} iters={it}");
    let k2 = eng.k_bsr().unwrap();
    let k2v: Vec<f64> = k2.values.iter().map(|&x| x as f64).collect();
    let e_after = 2.0 * trace_ab_f64(&k2v, &km, &hv, &hm);
    eprintln!("E_band after re-purify = {e_after:.6}  (exact-K band was {:.6})",
              2.0 * trace_ab_f64(&k_inj64, &km, &hv, &hm));
}

fn off_at(norb: &[u8], i: usize) -> usize { norb[..i].iter().map(|&x| x as usize).sum() }
fn trace_ps(p: &nalgebra::DMatrix<f64>, s: &nalgebra::DMatrix<f64>) -> f64 {
    (p * s).trace()
}
