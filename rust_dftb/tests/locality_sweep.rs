//! Gate C: independent locality sweep R_K × R_Z (manifest v3 §4.5).
//!
//! Sweeps R_K (density-kernel mask radius) and R_Z (inverse-overlap mask
//! radius) independently on a small insulating system. For each (R_K, R_Z)
//! pair, records:
//!   - energy error (vs dense reference)
//!   - charge error (max per-atom |Δq|)
//!   - Tr(KS) - N_occ
//!   - R_in  = || P_MK(KSK - K) ||_F
//!   - R_leak = || P_(Mval \ MK)(KSK) ||_F  (leakage onto validation mask)
//!   - R_H   = || HKS - SKH ||_F
//!   - TC2 iterations
//!   - wall time
//!
//! The test prints a table and asserts that the plateau (converged R_K,
//! converged R_Z) achieves physical accuracy. It does NOT assert on every
//! cell — small R_K or R_Z are expected to have large errors; that is the
//! diagnostic, not a failure.
//!
//! Manifest v3 §4.5:
//!   "The result we care about is a plateau of physical observables vs
//!    radius, not one magic cutoff."

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::methods::sparse::bsr4::{
    build_geometric_mask, build_full_mask, build_identity, build_product_mask,
    diag_block_map, inf_norm, Bsr4Matrix, BS,
};
use rust_dftb::methods::sparse::gpu_sparse::SparseBsr4Gpu;
use rust_dftb::methods::sparse::harness::require_sparse_gpu;
use std::collections::HashSet;
use std::time::Instant;

// ---------------------------------------------------------------------
// Helpers (shared with gpu_sparse_bsr4.rs style)
// ---------------------------------------------------------------------

/// Simple xorshift64 for reproducible random data (same as gpu_sparse_bsr4.rs).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x as i64 as f64) / (i64::MAX as f64)) as f32
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

fn random_symmetric_dense(n_atom: usize, rng: &mut Rng, scale: f32) -> Vec<f32> {
    let n = n_atom * BS;
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            let v = scale * rng.next();
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    d
}

/// Build a symmetric positive-definite S: I + small symmetric offdiag.
fn make_overlap_dense(n_atom: usize, rng: &mut Rng, offdiag: f32) -> Vec<f32> {
    let n = n_atom * BS;
    let mut d = vec![0.0f32; n * n];
    for i in 0..n { d[i * n + i] = 1.0; }
    for i in 0..n {
        for j in (i + 1)..n {
            let v = offdiag * rng.next() * 0.5;
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    d
}

fn row_major_to_dmatrix_f64(dense: &[f32], n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = dense[i * n + j] as f64;
        }
    }
    m
}

fn dmatrix_to_row_major_f32(m: &DMatrix<f64>) -> Vec<f32> {
    let n = m.nrows();
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = m[(i, j)] as f32;
        }
    }
    d
}

fn dense_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

/// CPU generalized eigensolve H c = S c eps. Returns spinless density kernel
/// K = sum_{occ} c_i c_i^T (row-major f32) and the occupied count.
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
    for i in 0..nocc {
        let col = c.column(i);
        k += &col * col.transpose();
    }
    dmatrix_to_row_major_f32(&k)
}

/// Dense energy E = Tr(K · H0) (spinless). K is the density kernel, H0 is
/// the Hamiltonian. For D=2K, E = Tr(D·H0)/2 = Tr(K·H0).
fn cpu_energy(k_dense: &[f32], h_dense: &[f32], n: usize) -> f64 {
    let mut e = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            e += k_dense[i * n + j] as f64 * h_dense[j * n + i] as f64;
        }
    }
    e
}

/// Dense Mulliken charges: q_A = sum_{i in A} (KS)_{ii}.
/// Returns per-atom charges (4 orbitals per atom).
fn cpu_mulliken_charges(k_dense: &[f32], s_dense: &[f32], n_atom: usize) -> Vec<f64> {
    let n = n_atom * BS;
    // KS = K · S (dense)
    let mut ks = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += k_dense[i * n + k] as f64 * s_dense[k * n + j] as f64;
            }
            ks[i * n + j] = s;
        }
    }
    // q_A = sum_{i in atom A} (KS)_{ii}
    let mut q = vec![0.0f64; n_atom];
    for a in 0..n_atom {
        for r in 0..BS {
            let i = a * BS + r;
            q[a] += ks[i * n + i];
        }
    }
    q
}

fn try_gpu() -> Option<SparseBsr4Gpu> {
    require_sparse_gpu()
}

// ---------------------------------------------------------------------
// Locality sweep
// ---------------------------------------------------------------------

/// One row of the (R_K, R_Z) → metrics table.
struct SweepRow {
    r_k: f64,
    r_z: f64,
    energy_err: f64,
    charge_err: f64,
    tr_err: f64,    // |Tr(KS) - Nocc|
    r_in: f32,
    r_leak: f32,
    r_h: f32,
    tc2_iters: usize,
    z_iters: usize,
    elapsed_ms: f64,
    converged: bool,
}

impl SweepRow {
    fn unconverged(r_k: f64, r_z: f64, why: &str) -> Self {
        eprintln!("  {r_k:5.1}  {r_z:5.1}  UNCONVERGED (recorded as data, not a skip): {why}");
        Self {
            r_k, r_z,
            energy_err: f64::INFINITY, charge_err: f64::INFINITY, tr_err: f64::INFINITY,
            r_in: f32::INFINITY, r_leak: f32::NAN, r_h: f32::NAN,
            tc2_iters: 0, z_iters: 0, elapsed_ms: 0.0, converged: false,
        }
    }
}

/// Run the full sparse pipeline for one (R_K, R_Z) pair and return metrics.
fn run_one(
    gpu: &SparseBsr4Gpu,
    h_dense: &[f32],
    s_dense: &[f32],
    n_atom: usize,
    nocc: usize,
    r_k: f64,
    r_z: f64,
    r_val: f64,  // validation mask radius (generous)
    k_ref_dense: &[f32],
    e_ref: f64,
    q_ref: &[f64],
) -> Result<SweepRow, String> {
    let n = n_atom * BS;
    let t0 = Instant::now();

    // Masks
    let m_hs = build_geometric_mask(&(0..n_atom).map(|i| [1.5 * i as f64, 0.0, 0.0]).collect::<Vec<_>>(), 2.0);
    let m_k = build_geometric_mask(&(0..n_atom).map(|i| [1.5 * i as f64, 0.0, 0.0]).collect::<Vec<_>>(), r_k);
    let m_z = build_geometric_mask(&(0..n_atom).map(|i| [1.5 * i as f64, 0.0, 0.0]).collect::<Vec<_>>(), r_z);
    let m_val = build_geometric_mask(&(0..n_atom).map(|i| [1.5 * i as f64, 0.0, 0.0]).collect::<Vec<_>>(), r_val);
    // T mask for Z·S (Z on M_Z, S on M_HS)
    let m_t_zs = build_product_mask(n_atom, &m_z, &m_hs);
    // T mask for K·S (K on M_K, S on M_HS)
    let m_t_ks = build_product_mask(n_atom, &m_k, &m_hs);

    // Build H0 and S on M_HS
    let h = bsr4_from_dense(n_atom, h_dense, &m_hs);
    let s = bsr4_from_dense(n_atom, s_dense, &m_hs);

    // 1. Z ≈ S⁻¹ on M_Z. Keep Z on M_Z for ZHZ (G1.3); project only K0 to M_K.
    let (z_on_mz, _rz, z_iters) = match gpu.newton_schulz_inverse(&s, &m_z, &m_t_zs, 50, 1e-5, 5) {
        Ok(r) => r,
        Err(e) => return Err(format!("Z failed: {e}")),
    };
    let m_t_zh = build_product_mask(n_atom, &m_z, &m_hs);

    // 2. Spectral bounds of B = Z·H with Z on M_Z
    let (emin, emax) = match gpu.spectral_bounds(&h, &z_on_mz, &m_t_zh, 0.1) {
        Ok(r) => r,
        Err(e) => return Err(format!("spectral_bounds failed: {e}")),
    };

    // 3. K₀ on M_Z via ZHZ, then project onto M_K
    let k0_on_mz = match gpu.build_k0(&h, &s, &z_on_mz, &m_z, &m_t_zh, emin, emax) {
        Ok(k) => k,
        Err(e) => return Err(format!("build_k0 failed: {e}")),
    };
    let k0 = k0_on_mz.project_to_mask(&m_k).map_err(|e| format!("K0 projection failed: {e}"))?;

    // 4. TC2 purification (K·S on m_t_ks). f32 TC2 typically converges to
    // ~1e-3..1e-4; use 1e-4 as the tolerance and 80 max iters.
    let nocc_f = nocc as f32;
    let (k_final, r_in, tr, tc2_iters, _hist) = match gpu.tc2_purify(&k0, &s, nocc_f, &m_k, &m_t_ks, 80, 1e-4) {
        Ok(r) => r,
        Err(e) => return Err(format!("TC2 failed: {e}")),
    };

    // 5. Energy: E = Tr(K · H0)
    let k_dense = k_final.to_dense();
    let e_sparse = cpu_energy(&k_dense, h_dense, n);
    let energy_err = (e_sparse - e_ref).abs();

    // 6. Charge error: max |q_sparse - q_ref|
    let q_sparse = cpu_mulliken_charges(&k_dense, s_dense, n_atom);
    let charge_err = q_sparse.iter().zip(q_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);

    // 7. Tr(KS) - Nocc
    let tr_err = (tr as f64 - nocc as f64).abs();

    // 8. R_H = ||HKS - SKH||_F (on validation mask)
    let r_h = gpu.hamiltonian_residual(&h, &k_final, &s, &m_val)
        .map_err(|e| format!("R_H failed: {e}"))?;
    let r_h_norm = r_h / (n as f32).sqrt();

    // 9. Exact R_leak = || P_(Mval \ MK)(KSK) ||_F (G1.2)
    let r_leak = compute_r_leak(gpu, &k_final, &s, &m_k, &m_val)?;

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
    let converged = r_in < 1e-3 && tr_err < 5e-2;

    Ok(SweepRow {
        r_k, r_z, energy_err, charge_err, tr_err,
        r_in, r_leak, r_h: r_h_norm, tc2_iters, z_iters, elapsed_ms, converged,
    })
}

/// Exact R_leak = || P_(Mval \ MK)(KSK) ||_F (review G1.2 / GPT-5.6 #11).
fn compute_r_leak(
    gpu: &SparseBsr4Gpu,
    k: &Bsr4Matrix,
    s: &Bsr4Matrix,
    m_k: &(Vec<u32>, Vec<u32>),
    m_val: &(Vec<u32>, Vec<u32>),
) -> Result<f32, String> {
    let m_t_val = build_product_mask(k.n_atom, m_val, m_val);
    let (_t_val, q_val) = gpu.ksk(k, s, m_val, &m_t_val)
        .map_err(|e| format!("R_leak KSK(M_val) failed: {e}"))?;
    let mut in_mk = HashSet::new();
    for i in 0..k.n_atom {
        let (a, b) = (m_k.0[i] as usize, m_k.0[i + 1] as usize);
        for blk in a..b {
            in_mk.insert((i, m_k.1[blk] as usize));
        }
    }
    let mut sum_sq = 0.0f32;
    for i in 0..q_val.n_atom {
        let (a, b) = (q_val.row_ptr[i] as usize, q_val.row_ptr[i + 1] as usize);
        for blk in a..b {
            let j = q_val.col_idx[blk] as usize;
            if in_mk.contains(&(i, j)) {
                continue;
            }
            let off = blk * BS * BS;
            for x in &q_val.values[off..off + BS * BS] {
                sum_sq += x * x;
            }
        }
    }
    Ok(sum_sq.sqrt())
}

// ---------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------

#[test]
fn test_locality_sweep_rk_rz() {
    let Some(gpu) = try_gpu() else { return };

    // 5-atom line. A diagonal +2I shift is NOT a HOMO–LUMO gap (G1.1).
    let n_atom = 5;
    let n = n_atom * BS;
    let nocc = 3;  // 3 occupied orbitals (out of 20)
    let mut rng = Rng(0x1234_abcd_5678_ef90);

    let h_full = random_symmetric_dense(n_atom, &mut rng, 0.4);
    let s_full = make_overlap_dense(n_atom, &mut rng, 0.2);

    // Truncate H and S to M_HS (the physical Hamiltonian/overlap locality).
    // The dense reference MUST use the same truncated H/S as the sparse pipeline.
    let m_hs = build_geometric_mask(&(0..n_atom).map(|i| [1.5 * i as f64, 0.0, 0.0]).collect::<Vec<_>>(), 2.0);
    let h_bsr = bsr4_from_dense(n_atom, &h_full, &m_hs);
    let s_bsr = bsr4_from_dense(n_atom, &s_full, &m_hs);
    let h_dense = h_bsr.to_dense();   // truncated to M_HS, zero-padded to full
    let s_dense = s_bsr.to_dense();

    // Dense reference (using the SAME truncated H/S)
    let k_ref_dense = cpu_density_kernel(&h_dense, &s_dense, n, nocc);
    let e_ref = cpu_energy(&k_ref_dense, &h_dense, n);
    let q_ref = cpu_mulliken_charges(&k_ref_dense, &s_dense, n_atom);
    eprintln!("Dense reference: E={e_ref:.6}, Nocc={nocc}, q_ref={q_ref:?}");

    // Validation mask: generous (covers everything)
    let r_val = 10.0;  // larger than the system

    // Phase 1: sweep R_K with R_Z generous
    eprintln!("\n=== Phase 1: Sweep R_K (R_Z = {r_val} generous) ===");
    eprintln!("  R_K    R_Z    E_err       q_err       |Tr-Nocc|   R_in       R_leak      R_H         TC2  Z   time(ms)  conv");
    let r_z_fixed = r_val;
    let r_k_values: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0, 7.0, 10.0];
    let mut phase1_rows: Vec<SweepRow> = Vec::new();
    for &r_k in &r_k_values {
        match run_one(&gpu, &h_dense, &s_dense, n_atom, nocc, r_k, r_z_fixed, r_val, &k_ref_dense, e_ref, &q_ref) {
            Ok(row) => {
                eprintln!("  {r_k:5.1}  {r_z_fixed:5.1}  {:.3e}   {:.3e}   {:.3e}   {:.3e}  {:.3e}  {:.3e}  {:3}  {:3}  {:8.1}  {}",
                    row.energy_err, row.charge_err, row.tr_err, row.r_in, row.r_leak, row.r_h,
                    row.tc2_iters, row.z_iters, row.elapsed_ms, row.converged);
                phase1_rows.push(row);
            }
            Err(e) => phase1_rows.push(SweepRow::unconverged(r_k, r_z_fixed, &e)),
        }
    }

    // Plateau only from rows that actually returned metrics (G0.5). No invented R=5.
    let converged_rk = phase1_rows.iter()
        .filter(|r| r.energy_err < 1e-3 && r.converged)
        .map(|r| r.r_k)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or_else(|| panic!(
            "Gate C: no R_K plateau (no converged row with energy_err<1e-3). \
             Truncated-R solver failure is data; inventing R_K=5.0 is not. \
             This 5-atom line is not a locality test of a gapped insulator (review A.2 / G1.1)."
        ));
    eprintln!("\n  → Converged R_K = {converged_rk}");

    // Phase 2: fix R_K, sweep R_Z
    eprintln!("\n=== Phase 2: Sweep R_Z (R_K = {converged_rk} fixed) ===");
    eprintln!("  R_K    R_Z    E_err       q_err       |Tr-Nocc|   R_in       R_leak      R_H         TC2  Z   time(ms)  conv");
    let r_z_values: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0, 7.0, 10.0];
    let mut phase2_rows: Vec<SweepRow> = Vec::new();
    for &r_z in &r_z_values {
        match run_one(&gpu, &h_dense, &s_dense, n_atom, nocc, converged_rk, r_z, r_val, &k_ref_dense, e_ref, &q_ref) {
            Ok(row) => {
                eprintln!("  {converged_rk:5.1}  {r_z:5.1}  {:.3e}   {:.3e}   {:.3e}   {:.3e}  {:.3e}  {:.3e}  {:3}  {:3}  {:8.1}  {}",
                    row.energy_err, row.charge_err, row.tr_err, row.r_in, row.r_leak, row.r_h,
                    row.tc2_iters, row.z_iters, row.elapsed_ms, row.converged);
                phase2_rows.push(row);
            }
            Err(e) => phase2_rows.push(SweepRow::unconverged(converged_rk, r_z, &e)),
        }
    }

    let converged_rz = phase2_rows.iter()
        .filter(|r| r.energy_err < 1e-3 && r.converged)
        .map(|r| r.r_z)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or_else(|| panic!(
            "Gate C: no R_Z plateau (no converged row with energy_err<1e-3). \
             Inventing R_Z=5.0 is forbidden (G0.5)."
        ));
    eprintln!("\n  → Converged R_Z = {converged_rz}");

    // Phase 3: verify the crossed combination
    eprintln!("\n=== Phase 3: Cross-check (R_K={converged_rk}, R_Z={converged_rz}) ===");
    match run_one(&gpu, &h_dense, &s_dense, n_atom, nocc, converged_rk, converged_rz, r_val, &k_ref_dense, e_ref, &q_ref) {
        Ok(row) => {
            eprintln!("  R_K    R_Z    E_err       q_err       |Tr-Nocc|   R_in       R_leak      R_H         TC2  Z   time(ms)  conv");
            eprintln!("  {converged_rk:5.1}  {converged_rz:5.1}  {:.3e}   {:.3e}   {:.3e}   {:.3e}  {:.3e}  {:.3e}  {:3}  {:3}  {:8.1}  {}",
                row.energy_err, row.charge_err, row.tr_err, row.r_in, row.r_leak, row.r_h,
                row.tc2_iters, row.z_iters, row.elapsed_ms, row.converged);

            // Assert: the plateau must achieve physical accuracy.
            // This is the Gate C pass criterion: there EXISTS a (R_K, R_Z)
            // where energy, charge, and trace are all accurate.
            assert!(row.energy_err < 1e-3,
                "Gate C: energy error {:.3e} too large at plateau (R_K={converged_rk}, R_Z={converged_rz})",
                row.energy_err);
            assert!(row.charge_err < 1e-2,
                "Gate C: charge error {:.3e} too large at plateau",
                row.charge_err);
            assert!(row.tr_err < 1e-5,
                "Gate C: trace error {:.3e} too large at plateau (G2: Tr(KS)=Nocc)",
                row.tr_err);
            eprintln!("\n  Gate C: plateau numbers printed. A 5-atom line with R≥7 covering the whole system is not a locality proof.");
            let system_span = 1.5 * (n_atom - 1) as f64;
            assert!(
                converged_rk < system_span && converged_rz < system_span,
                "Gate C: plateau (R_K={converged_rk}, R_Z={converged_rz}) Å covers the entire {system_span} Å toy. \
                 That is dense algebra on a small matrix, not a locality test (review A.2 / G1.1). \
                 Redo on a gapped passivated Si/H cluster where a sub-span radius still converges."
            );
        }
        Err(e) => panic!("Gate C cross-check failed: {e}"),
    }
}
