//! TEMP diagnostic (2026-09-14, revised after source review): separate
//! measurement error from matrix error on the converged sparse K.
//!
//! 1. Recompute 2·Tr(K·H0) in f64 on the host (device trace check).
//! 2. TRUE generalized McWeeny  K' = 3·(KS)·K − 2·(KS)·(KS)·K  in f64
//!    throughout (earlier version ran the wrong map 2K−KSK and cast
//!    intermediates to f32 — those conclusions are void).
//!
//! Masks: r_k is a RADIUS (default 12 Å); on R10 the full pair mask is
//! 330² = 108 900 blocks and r_k=20 still gives only 108 090 (99.3%) —
//! "near-complete", never literally complete. F64CHECK_RK controls it.
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

/// Boolean mask product: (i,l) ∈ A∘B iff ∃k: (i,k)∈A ∧ (k,l)∈B.
/// This is the TRUE support of a two-factor sparse product before
/// projection — M_TKS = M_K ∘ M_S is what T=K·S needs to feed Q=T·K
/// without intermediate truncation.
fn mask_bool_prod(am: &(Vec<u32>, Vec<u32>), bm: &(Vec<u32>, Vec<u32>)) -> (Vec<u32>, Vec<u32>) {
    let n = am.0.len() - 1;
    let mut rp = Vec::with_capacity(n + 1);
    let mut ci = Vec::new();
    let mut mark = vec![false; n];
    let mut touched: Vec<u32> = Vec::new();
    rp.push(0u32);
    for i in 0..n {
        for ai in am.0[i] as usize..am.0[i + 1] as usize {
            let k = am.1[ai] as usize;
            for bi in bm.0[k] as usize..bm.0[k + 1] as usize {
                let l = bm.1[bi];
                if !mark[l as usize] { mark[l as usize] = true; touched.push(l); }
            }
        }
        touched.sort_unstable();
        ci.extend_from_slice(&touched);
        for &l in &touched { mark[l as usize] = false; }
        touched.clear();
        rp.push(ci.len() as u32);
    }
    (rp, ci)
}

/// §15.12 / review additional-finding-2 — frozen-operands diagnostic.
/// Separates intermediate-truncation loss from K-truncation loss:
///   Q_legacy = P_M( P_M(K_c·S) · K_c )   production: KS truncated to M_K
///   Q_exact  = P_M( (K_c·S) · K_c )     KS on true support M_K∘M_S
///   Q_ref    = P_M( K_ref·S·K_ref )     reference full-rank projector
/// ‖Q_legacy−Q_exact‖ isolates the dropped KS halo; Q_exact vs Q_ref
/// exposes what truncating K itself costs. Host f64 throughout.
#[test]
#[ignore]
fn intermediate_loss_diag() {
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
        dense_diag: Some(true),
        ..Default::default()
    };
    let mut eng = SparseDftb::with_config(sk, &sk_dir, mol.species.clone(), mol.coords.clone(), cfg)
        .unwrap_or_else(|e| panic!("SparseDftb: {e}"));
    eng.scc(60, 1e-5).unwrap_or_else(|e| panic!("scc: {e}"));
    let e_dev = eng.last_energy().e_h0;

    let n_atom = mol.species.len();
    let k = eng.k_bsr().unwrap();
    let km = (k.row_ptr.clone(), k.col_idx.clone());
    let kv: Vec<f64> = k.values.iter().map(|&x| x as f64).collect();
    let sm = (eng.s_bsr().row_ptr.clone(), eng.s_bsr().col_idx.clone());
    let sv: Vec<f64> = eng.s_bsr().values.iter().map(|&x| x as f64).collect();
    let hm = (eng.h_bsr().row_ptr.clone(), eng.h_bsr().col_idx.clone());
    let hv: Vec<f64> = eng.h_bsr().values.iter().map(|&x| x as f64).collect();

    // True support of K·S: every (i,l) with ∃k: K_ik,S_kl — needed to feed
    // Q=T·K on M_K without dropping intermediate terms.
    let m_ks = mask_bool_prod(&km, &sm);
    let deg = |m: &(Vec<u32>, Vec<u32>)| -> u32 {
        (0..n_atom).map(|i| m.0[i + 1] - m.0[i]).max().unwrap_or(0)
    };
    let avg_deg = |m: &(Vec<u32>, Vec<u32>)| -> f64 { m.1.len() as f64 / n_atom as f64 };
    eprintln!("masks: M_K nnz={} deg={} avg={:.1} | M_S nnz={} deg={} avg={:.1} | M_K∘M_S nnz={} deg={} avg={:.1} | full={}",
        km.1.len(), deg(&km), avg_deg(&km), sm.1.len(), deg(&sm), avg_deg(&sm),
        m_ks.1.len(), deg(&m_ks), avg_deg(&m_ks), n_atom * n_atom);

    // T=K·S twice: legacy (truncated to M_K) vs exact support
    let t_leg = spmm_f64(&kv, &km, &sv, &sm, &km);
    let t_ext = spmm_f64(&kv, &km, &sv, &sm, &m_ks);
    // Q = T·K on M_K for both variants
    let q_leg = spmm_f64(&t_leg, &km, &kv, &km, &km);
    let q_ext = spmm_f64(&t_ext, &m_ks, &kv, &km, &km);
    let n2 = |a: &[f64]| -> f64 { a.iter().map(|x| x * x).sum::<f64>().sqrt() };
    let d_leg_ext: f64 = q_leg.iter().zip(&q_ext).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    eprintln!("INTERMEDIATE LOSS  ||Q_legacy − Q_exact||_F = {d_leg_ext:.4}   (||Q_exact||={:.3}, rel {:.3})",
        n2(&q_ext), d_leg_ext / n2(&q_ext));
    // masked residuals of each product path vs K_c
    let ri = |q: &[f64], k: &[f64]| -> f64 {
        let num: f64 = q.iter().zip(k).map(|(a, b)| (a - b) * (a - b)).sum();
        (num / k.iter().map(|x| x * x).sum::<f64>()).sqrt()
    };
    eprintln!("masked R_I: legacy={:.3e}  exact-products={:.3e}", ri(&q_leg, &kv), ri(&q_ext, &kv));

    // Reference: exact K from the frozen-input eigh (p_exact/2), same run.
    let norb = eng.atom_n_orb().to_vec();
    let n_pad = n_atom * 4;
    let n_phys: usize = norb.iter().map(|&n| n as usize).sum();
    let strip = |pad: &[f64]| -> nalgebra::DMatrix<f64> {
        let mut m = nalgebra::DMatrix::zeros(n_phys, n_phys);
        for a in 0..n_atom {
            let oa: usize = norb[..a].iter().map(|&x| x as usize).sum();
            for b in 0..n_atom {
                let ob: usize = norb[..b].iter().map(|&x| x as usize).sum();
                for r in 0..norb[a] as usize { for c in 0..norb[b] as usize {
                    m[(oa + r, ob + c)] = pad[(a * 4 + r) * n_pad + (b * 4 + c)];
                }}
            }
        }
        m
    };
    let h_scc_pad: Vec<f64> = eng.h_scc_pad().iter().map(|&x| x as f64).collect();
    let h_scc = strip(&h_scc_pad);
    let mut s_pad = vec![0.0f64; n_pad * n_pad];
    for i in 0..n_atom {
        for bi in sm.0[i] as usize..sm.0[i + 1] as usize {
            let j = sm.1[bi] as usize;
            for r in 0..4 { for c in 0..4 {
                s_pad[(i * 4 + r) * n_pad + j * 4 + c] = sv[bi * 16 + r * 4 + c];
            }}
        }
    }
    let s_phys = strip(&s_pad);
    let chol = s_phys.cholesky().expect("S not SPD");
    let linv = chol.l().try_inverse().expect("L singular");
    let hp = &linv * &h_scc * linv.transpose();
    let n = n_phys;
    let mut a = hp.as_slice().to_vec();
    let mut w = vec![0.0f64; n];
    let (mut work_q, mut iwork_q, mut info) = ([0.0f64; 1], [0i32; 1], 0i32);
    unsafe { lapack::dsyevd(b'V', b'L', n as i32, &mut a, n as i32, &mut w, &mut work_q, -1, &mut iwork_q, -1, &mut info); }
    let (lw, li) = (work_q[0] as usize, iwork_q[0] as usize);
    let (mut work, mut iwork) = (vec![0.0f64; lw], vec![0i32; li]);
    unsafe { lapack::dsyevd(b'V', b'L', n as i32, &mut a, n as i32, &mut w, &mut work, lw as i32, &mut iwork, li as i32, &mut info); }
    assert_eq!(info, 0);
    let nocc = 459;
    let cprime = nalgebra::DMatrix::from_column_slice(n, n, &a);
    let c = linv.transpose() * cprime;
    let d_exact = 2.0 * &c.columns(0, nocc) * c.columns(0, nocc).transpose();  // doubly-occupied
    let k_exact_dense = &d_exact * 0.5;                                      // spin-free K = CCᵀ

    // K_ref on a FULL mask (n_atom²) → Q_ref = P_M(K_ref·S·K_ref)
    let mut fm_rp = Vec::with_capacity(n_atom + 1);
    let mut fm_ci = Vec::with_capacity(n_atom * n_atom);
    for i in 0..n_atom { fm_rp.push(fm_ci.len() as u32); fm_ci.extend(0..n_atom as u32); }
    fm_rp.push(fm_ci.len() as u32);
    let fm = (fm_rp, fm_ci);
    let mut kv_ref = vec![0.0f64; fm.1.len() * 16];
    for i in 0..n_atom {
        let oi = off_at(&norb, i);
        for bi in fm.0[i] as usize..fm.0[i + 1] as usize {
            let j = fm.1[bi] as usize;
            let oj = off_at(&norb, j);
            for r in 0..norb[i] as usize { for c in 0..norb[j] as usize {
                kv_ref[bi * 16 + r * 4 + c] = k_exact_dense[(oi + r, oj + c)];
            }}
        }
    }
    let t_ref = spmm_f64(&kv_ref, &fm, &sv, &sm, &fm);       // full-support KS
    let q_ref = spmm_f64(&t_ref, &fm, &kv_ref, &fm, &km);    // → M_K
    let d_ext_ref: f64 = q_ext.iter().zip(&q_ref).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    let d_leg_ref: f64 = q_leg.iter().zip(&q_ref).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    eprintln!("vs REFERENCE: ||Q_legacy − Q_ref|| = {d_leg_ref:.4}   ||Q_exact − Q_ref|| = {d_ext_ref:.4}   (||Q_ref||={:.3})",
        n2(&q_ref));
    // K_ref projected onto M_K for residual/energy comparisons
    let mut kv_ref_mk = vec![0.0f64; km.1.len() * 16];
    for i in 0..n_atom {
        let oi = off_at(&norb, i);
        for bi in km.0[i] as usize..km.0[i + 1] as usize {
            let j = km.1[bi] as usize;
            let oj = off_at(&norb, j);
            for r in 0..norb[i] as usize { for c in 0..norb[j] as usize {
                kv_ref_mk[bi * 16 + r * 4 + c] = k_exact_dense[(oi + r, oj + c)];
            }}
        }
    }
    // masked idempotency residuals (all on M_K)
    eprintln!("masked R_I: exact-K|M_K={:.3e}  sparse-K legacy={:.3e}  sparse-K exact-products={:.3e}",
        ri(&q_ref, &kv_ref_mk), ri(&q_leg, &kv), ri(&q_ext, &kv));
    // band energies under the engine contraction (only the M_K∩M_HS part
    // is ever sampled — this is the quantity M_K must get right)
    let e_band = |kv: &[f64]| -> f64 { 2.0 * trace_ab_f64(kv, &km, &hv, &hm) };
    eprintln!("E_band: sparse-K={e_dev:.6}  exact-K|M_K={:.6}", e_band(&kv_ref_mk));

    // ── Stability test: masked TC2 in host f64 FROM the exact fixed point.
    // If exact arithmetic stays on K_ref|M_K → the walk-away is seeded by
    // f32 noise / intermediate truncation. If it walks away anyway → the
    // projected map's fixed point is dynamically unstable on M_K.
    let nocc64 = nocc as f64;
    let mut cur = kv_ref_mk.clone();
    let mut cur_m = km.clone();
    eprintln!("masked-f64 TC2 from exact-K|M_K (both product variants):");
    for it in 0..16 {
        // T = K·S on the TRUE support, Tr for the branch
        let t = spmm_f64(&cur, &cur_m, &sv, &sm, &m_ks);
        let tr: f64 = (0..n_atom).map(|i| {
            let (a0, a1) = (m_ks.0[i] as usize, m_ks.0[i + 1] as usize);
            match m_ks.1[a0..a1].iter().position(|&c| c == i as u32) {
                Some(p) => { let b = &t[(a0 + p) * 16..(a0 + p) * 16 + 16];
                             (0..4).map(|r| b[r * 4 + r]).sum::<f64>() }
                None => 0.0 }
        }).sum();
        // exact-products variant: Q on M_K from the wide T
        let q_e = spmm_f64(&t, &m_ks, &cur, &cur_m, &km);
        // legacy variant: T truncated to M_K first
        let t_l = spmm_f64(&cur, &cur_m, &sv, &sm, &km);
        let q_l = spmm_f64(&t_l, &km, &cur, &cur_m, &km);
        let branch = tr > nocc64;
        let ri_e = ri(&q_e, &cur); let ri_l = ri(&q_l, &cur);
        let mut next = cur.clone();
        for (i, (&qe, &ql)) in q_e.iter().zip(&q_l).enumerate() {
            next[i] = if branch { qe } else { 2.0 * cur[i] - qe };
            let _ = ql;
        }
        // trace the legacy path's error too (how far each walks)
        let d_ref: f64 = next.iter().zip(&kv_ref_mk).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        eprintln!("  it{it:3} Tr(KS)={tr:.4} branch={}  R_I exact={ri_e:.3e} legacy={ri_l:.3e}  ||K−K_ref|M_K||={d_ref:.4}", branch as u8);
        cur = next;
        cur_m = km.clone();   // iterate stays on M_K
    }
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
    eprintln!("frozen H_scc eigh:  2Σ_occ ε = {e_band_exact:.6}   Tr(D·H0) = {e_h0_exact:.6}   (D=2C_occC_occᵀ doubly-occupied)");
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
    // Convention: engine K is spin-free (occupation 1): K = C_occ·C_occᵀ,
    // dense doubly-occupied projector D = 2K, and the non-symmetric P = K·S.
    // Compare 2K_sparse vs D_exact — NOT K vs D (factor-2 artifact).
    let diff = 2.0 * &pd - &p_exact;
    eprintln!("||2·K_sparse − D_exact||_F = {:.4}   ||D_exact||_F = {:.4}   ||K_exact=D/2||_F = {:.4}",
              diff.norm(), p_exact.norm(), p_exact.norm() / 2.0);
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
    // Exact-K idempotency residual in f64: ||P_M(KSK)−K||/||K|| — MASKED
    // residual (both products restricted to M_K); the full-space residual
    // also contains the dropped halo terms, so this is a lower bound only.
    let ks_e = spmm_f64(&k_inj64, &km, &sv, &sm, &km);
    let ksk_e = spmm_f64(&ks_e, &km, &k_inj64, &km, &km);
    let mut num = 0.0f64; let mut den = 0.0f64;
    for i in 0..ksk_e.len() { num += (ksk_e[i] - k_inj64[i]).powi(2); den += k_inj64[i].powi(2); }
    eprintln!("exact-K f64 R_I (masked to M_K) = {:.3e}   (device floor was ~5e-4)", (num / den).sqrt());
    // and the sparse K's own masked f64 residual for comparison
    let ks_s = spmm_f64(&kv, &km, &sv, &sm, &km);
    let ksk_s = spmm_f64(&ks_s, &km, &kv, &km, &km);
    let mut num = 0.0f64; let mut den = 0.0f64;
    for i in 0..ksk_s.len() { num += (ksk_s[i] - kv[i]).powi(2); den += kv[i].powi(2); }
    eprintln!("sparse-K f64 R_I (masked to M_K) = {:.3e}", (num / den).sqrt());

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
