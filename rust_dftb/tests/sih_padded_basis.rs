//! Gate D: Nonsingular padded Si/H basis (manifest v3 §4.10, §1020).
//!
//! Tests the uniform nonsingular BSR4 padding for H atoms on a small
//! H-passivated Si cluster (SiH4 — silane).
//!
//! H has 1 physical 1s orbital while Si has 4 sp3 orbitals. Padding H with
//! 3 zero-overlap dummy orbitals makes S singular and is forbidden. Instead:
//!   S_dd = 1
//!   H_dd = E_dummy
//!   all active-dummy and interatomic dummy couplings = 0 exactly
//!
//! Gate D pass criteria (manifest §1020):
//! - correct electron count (Tr(KS) → physical Nocc);
//! - negligible dummy occupation;
//! - dense/sparse energy/charge parity;
//! - locality sweep is at least as well behaved as expected from the
//!   passivated gap.

use nalgebra::{DMatrix, SymmetricEigen};
use rust_dftb::methods::sparse::bsr4::{
    build_geometric_mask, build_full_mask, build_product_mask, Bsr4Matrix, BS,
};
use rust_dftb::methods::sparse::gpu_sparse::{SparseBsr4Config, SparseBsr4Gpu};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder};
use std::panic::{catch_unwind, AssertUnwindSafe};

const ANG2BOHR: f64 = 1.889_726_133;
const E_DUMMY: f32 = 2.0;  // dummy orbital onsite energy (above occupied spectrum, not too high)

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn try_gpu() -> Option<SparseBsr4Gpu> {
    match catch_unwind(AssertUnwindSafe(|| {
        SparseBsr4Gpu::new(SparseBsr4Config::default())
    })) {
        Ok(Ok(gpu)) => Some(gpu),
        Ok(Err(e)) => { eprintln!("Skipping Gate D: no OpenCL ({e})"); None }
        Err(_) => { eprintln!("Skipping Gate D: OpenCL panic"); None }
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

/// Build the padded BSR4 H and S matrices from a dense (variable-orbital)
/// H0 and S produced by HamiltonianBuilder.
///
/// Atom ordering: `species` gives the atom types. `atom_n_orb` gives the
/// physical orbital count per atom (Si=4, H=1). The padded BSR4 matrix has
/// 4 orbitals per atom. For H atoms, the first orbital is physical (1s)
/// and orbitals 2-4 are dummy:
///   S_dd = I_3, H_dd = E_dummy * I_3
///   all active-dummy and interatomic dummy couplings = 0
///
/// Returns (h_padded_dense, s_padded_dense, n_orb_physical, n_orb_padded,
///          atom_orb_off_physical, dummy_orb_indices).
fn build_padded_bsr4(
    h0_dense: &[f64],  // n_phys × n_phys row-major
    s_dense: &[f64],
    atom_n_orb: &[u8],
) -> (Vec<f32>, Vec<f32>, usize, usize, Vec<usize>, Vec<usize>) {
    let n_atom = atom_n_orb.len();
    let n_padded = n_atom * BS;
    let n_phys: usize = atom_n_orb.iter().map(|&n| n as usize).sum();

    // Build physical → padded orbital offset map
    let mut phys_off = Vec::with_capacity(n_atom);
    let mut padded_off = Vec::with_capacity(n_atom);
    let mut acc_phys = 0usize;
    let mut acc_padded = 0usize;
    for &n in atom_n_orb {
        phys_off.push(acc_phys);
        padded_off.push(acc_padded);
        acc_phys += n as usize;
        acc_padded += BS;
    }

    // Track dummy orbital indices (for later occupation checks)
    let mut dummy_indices: Vec<usize> = Vec::new();
    for (a, &n) in atom_n_orb.iter().enumerate() {
        for d in (n as usize)..BS {
            dummy_indices.push(padded_off[a] + d);
        }
    }

    // Initialize padded matrices to zero
    let mut h_pad = vec![0.0f32; n_padded * n_padded];
    let mut s_pad = vec![0.0f32; n_padded * n_padded];

    // Fill physical-physical blocks
    for a in 0..n_atom {
        for b in 0..n_atom {
            let na = atom_n_orb[a] as usize;
            let nb = atom_n_orb[b] as usize;
            for i in 0..na {
                for j in 0..nb {
                    let pi = padded_off[a] + i;
                    let pj = padded_off[b] + j;
                    let phys_i = phys_off[a] + i;
                    let phys_j = phys_off[b] + j;
                    h_pad[pi * n_padded + pj] = h0_dense[phys_i * n_phys + phys_j] as f32;
                    s_pad[pi * n_padded + pj] = s_dense[phys_i * n_phys + phys_j] as f32;
                }
            }
        }
    }

    // Fill dummy diagonal blocks: S_dd = 1, H_dd = E_dummy
    for (a, &n) in atom_n_orb.iter().enumerate() {
        for d in (n as usize)..BS {
            let pi = padded_off[a] + d;
            s_pad[pi * n_padded + pi] = 1.0;
            h_pad[pi * n_padded + pi] = E_DUMMY;
        }
    }

    (h_pad, s_pad, n_phys, n_padded, padded_off, dummy_indices)
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

/// Dense energy E = Tr(K · H0) (spinless).
fn cpu_energy(k_dense: &[f32], h_dense: &[f32], n: usize) -> f64 {
    let mut e = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            e += k_dense[i * n + j] as f64 * h_dense[j * n + i] as f64;
        }
    }
    e
}

/// Mulliken charges: q_A = sum_{i in A} (KS)_{ii}.
fn cpu_mulliken_charges(k_dense: &[f32], s_dense: &[f32], n_atom: usize) -> Vec<f64> {
    let n = n_atom * BS;
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
    let mut q = vec![0.0f64; n_atom];
    for a in 0..n_atom {
        for r in 0..BS {
            let i = a * BS + r;
            q[a] += ks[i * n + i];
        }
    }
    q
}

// ---------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------

#[test]
fn test_sih_padded_basis_gate_d() {
    let Some(gpu) = try_gpu() else { return };

    // Use matsci-0-3 SK set for Si/H
    let sk_dir = match std::env::var("RUST_DFTB_SK_DIR") {
        Ok(d) => d,
        Err(_) => {
            // Try default location
            let d = "/home/prokop/SIMULATIONS/dftbplus/slakos/matsci-0-3";
            if !std::path::Path::new(d).exists() {
                eprintln!("Skipping Gate D: RUST_DFTB_SK_DIR not set and default {d} not found");
                return;
            }
            d.to_string()
        }
    };
    eprintln!("Using SK dir: {sk_dir}");

    // SiH4 (silane): 1 Si + 4 H, tetrahedral geometry
    // Si-H bond length ~1.48 Å, tetrahedral angle 109.47°
    let species = vec!["Si".to_string(), "H".to_string(), "H".to_string(), "H".to_string(), "H".to_string()];
    let bond = 1.48f64;
    let theta = 109.47f64 * std::f64::consts::PI / 180.0;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    // Place Si at origin, 4 H at tetrahedral positions
    let coords = vec![
        [0.0, 0.0, 0.0],  // Si
        [bond, 0.0, 0.0],  // H1
        [bond * cos_t, bond * sin_t, 0.0],  // H2
        [bond * cos_t, bond * sin_t * cos_t, bond * sin_t * sin_t],  // H3
        [bond * cos_t, -bond * sin_t * cos_t, -bond * sin_t * sin_t],  // H4 (approximate)
    ];

    // Load SK data
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // Build dense H0 and S using HamiltonianBuilder
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc = builder.build_scc(&species, &coords, 200, 1e-9).unwrap();
    let n_phys = scc.h0.nrows();
    eprintln!("SiH4: n_phys={n_phys}, n_atoms={}", species.len());

    // Physical orbital counts: Si=4, H=1
    let atom_n_orb: Vec<u8> = vec![4, 1, 1, 1, 1];
    assert_eq!(n_phys, atom_n_orb.iter().map(|&n| n as usize).sum::<usize>());

    // Electron count: Si has 4 valence e-, H has 1 each → 4 + 4*1 = 8 e- → 4 occ MOs
    // (matsci-0-3 SK files don't encode n_shell on the grid line, so q0
    // extraction from the SK file is unreliable; use known valence counts.)
    let n_electrons: f64 = 4.0 + 4.0 * 1.0;  // Si + 4*H
    let n_occ_phys = (n_electrons / 2.0).round() as usize;
    eprintln!("  n_electrons={n_electrons}, n_occ_phys={n_occ_phys}");

    // Flatten H0 and S to row-major f64
    let h0_dense: Vec<f64> = (0..n_phys * n_phys).map(|idx| scc.h0[(idx / n_phys, idx % n_phys)]).collect();
    let s_dense: Vec<f64> = (0..n_phys * n_phys).map(|idx| scc.s[(idx / n_phys, idx % n_phys)]).collect();

    // Build padded BSR4 H and S
    let (h_pad, s_pad, n_phys, n_padded, padded_off, dummy_indices) =
        build_padded_bsr4(&h0_dense, &s_dense, &atom_n_orb);
    eprintln!("  n_padded={n_padded}, n_dummy_orbs={}", dummy_indices.len());

    // Dense reference (using padded matrices, same n_occ_phys)
    let k_ref_dense = cpu_density_kernel(&h_pad, &s_pad, n_padded, n_occ_phys);
    let e_ref = cpu_energy(&k_ref_dense, &h_pad, n_padded);
    let q_ref = cpu_mulliken_charges(&k_ref_dense, &s_pad, species.len());
    eprintln!("  Dense ref: E={e_ref:.6}, q_ref={q_ref:?}");

    // Check dummy occupation in the dense reference
    let ks_ref_dense: Vec<f64> = {
        let n = n_padded;
        let mut ks = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f64;
                for k in 0..n {
                    s += k_ref_dense[i * n + k] as f64 * s_pad[k * n + j] as f64;
                }
                ks[i * n + j] = s;
            }
        }
        ks
    };
    let mut max_dummy_occ = 0.0f64;
    let mut total_dummy_occ = 0.0f64;
    for &d in &dummy_indices {
        let occ = ks_ref_dense[d * n_padded + d];
        max_dummy_occ = max_dummy_occ.max(occ.abs());
        total_dummy_occ += occ.abs();
    }
    eprintln!("  Dense ref dummy occ: max={max_dummy_occ:.3e}, total={total_dummy_occ:.3e}");

    // Build BSR4 matrices on full mask
    let mask = build_full_mask(species.len());
    let h_bsr = bsr4_from_dense(species.len(), &h_pad, &mask);
    let s_bsr = bsr4_from_dense(species.len(), &s_pad, &mask);

    // Run sparse pipeline: Z ≈ S⁻¹, K₀, TC2
    eprintln!("\nSparse pipeline:");
    let (z, rz, z_iters) = gpu
        .newton_schulz_inverse(&s_bsr, &mask, &mask, 50, 1e-5, 5)
        .expect("Z must converge");
    eprintln!("  Z: {z_iters} iters, R_Z={rz:.3e}");

    let (emin, emax) = gpu.spectral_bounds(&h_bsr, &z, &mask, 0.1).unwrap();
    eprintln!("  spectral bounds: emin={emin:.4} emax={emax:.4}");

    let k0 = gpu.build_k0(&h_bsr, &s_bsr, &z, &mask, &mask, emin, emax).unwrap();
    let nocc_f = n_occ_phys as f32;
    let (k_final, r_in, tr, tc2_iters, _hist) = gpu
        .tc2_purify(&k0, &s_bsr, nocc_f, &mask, &mask, 80, 1e-4)
        .expect("TC2 must converge");
    eprintln!("  TC2: {tc2_iters} iters, R_I={r_in:.3e}, Tr(KS)={tr:.6} (Nocc={nocc_f})");

    // Gate D checks
    eprintln!("\n=== Gate D Checks ===");

    // 1. Correct electron count: Tr(KS) → physical Nocc
    let tr_err = (tr as f64 - n_occ_phys as f64).abs();
    eprintln!("  Tr(KS) = {tr:.6}, Nocc_phys = {n_occ_phys}, |err| = {tr_err:.3e}");
    assert!(tr_err < 1e-2,
        "Gate D: Tr(KS)={tr:.6} != Nocc_phys={n_occ_phys}, err={tr_err:.3e}");

    // 2. Dummy occupation negligible
    let k_sparse_dense = k_final.to_dense();
    let ks_sparse_dense: Vec<f64> = {
        let n = n_padded;
        let mut ks = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f64;
                for k in 0..n {
                    s += k_sparse_dense[i * n + k] as f64 * s_pad[k * n + j] as f64;
                }
                ks[i * n + j] = s;
            }
        }
        ks
    };
    let mut max_dummy_occ_sparse = 0.0f64;
    let mut total_dummy_occ_sparse = 0.0f64;
    for &d in &dummy_indices {
        let occ = ks_sparse_dense[d * n_padded + d];
        max_dummy_occ_sparse = max_dummy_occ_sparse.max(occ.abs());
        total_dummy_occ_sparse += occ.abs();
    }
    eprintln!("  Sparse dummy occ: max={max_dummy_occ_sparse:.3e}, total={total_dummy_occ_sparse:.3e}");
    assert!(max_dummy_occ_sparse < 1e-2,
        "Gate D: max dummy occupation {max_dummy_occ_sparse:.3e} too large");
    assert!(total_dummy_occ_sparse < 1e-2,
        "Gate D: total dummy occupation {total_dummy_occ_sparse:.3e} too large");

    // 3. Active Mulliken electron sum has correct physical count
    // Note: K is the spinless density kernel, Tr(KS) = Nocc. The electron
    // count is N_e = 2*Nocc for closed-shell. Mulliken charges q_A = sum
    // (KS)_{ii} give Nocc per atom, so the total is Nocc. The electron
    // count is 2 * sum(q_A).
    let q_sparse = cpu_mulliken_charges(&k_sparse_dense, &s_pad, species.len());
    let active_electrons: f64 = 2.0 * q_sparse.iter().sum::<f64>();
    eprintln!("  Active Mulliken sum = {active_electrons:.6} (expected {n_electrons:.6})");
    assert!((active_electrons - n_electrons).abs() < 1e-1,
        "Gate D: active Mulliken electron sum {active_electrons:.6} != {n_electrons:.6}");

    // 4. Dense/sparse energy/charge parity
    let e_sparse = cpu_energy(&k_sparse_dense, &h_pad, n_padded);
    let energy_err = (e_sparse - e_ref).abs();
    eprintln!("  E_sparse={e_sparse:.6}, E_ref={e_ref:.6}, |dE|={energy_err:.3e}");
    assert!(energy_err < 1e-2,
        "Gate D: energy parity failed |dE|={energy_err:.3e} > 1e-2");

    let charge_err = q_sparse.iter().zip(q_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    eprintln!("  |dq|={charge_err:.3e}");
    assert!(charge_err < 1e-2,
        "Gate D: charge parity failed |dq|={charge_err:.3e} > 1e-2");

    // 5. K parity (sparse vs dense)
    let k_err = dense_max_abs_diff(&k_sparse_dense, &k_ref_dense);
    eprintln!("  ||K_sparse - K_ref||_max = {k_err:.3e}");
    assert!(k_err < 5e-2,
        "Gate D: K parity failed ||dK||={k_err:.3e} > 5e-2");

    eprintln!("\n  Gate D: PASS — padded Si/H basis is nonsingular, dummy occupation negligible.");
}
