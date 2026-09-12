//! GPU SCC component kernel tests (Agent_6, Wave 2).
//!
//! Verifies the 5 new kernels in `gpu_matrix_ops.cl` against CPU references:
//!   1. `matmul_full_local_batched`  — full-local GEMM vs CPU (f64)
//!   2. `gamma_matvec_batched`       — V = G·Δq vs CPU `gamma_full`
//!   3. `h_scc_update_batched`       — H = H0 + 0.5·S·(V_i+V_j) vs CPU
//!   4. `mulliken_charges_batched`   — q_A = Σ_{μ∈A}(D·S)_μμ vs CPU
//!   5. `residual_and_mix_batched`   — rms + simple mix vs CPU
//!
//! Plus a batched integration test (10× H2O, all kernels in one launch
//! sequence) and a GEMM benchmark (full-local vs tiled).
//!
//! Environment:
//!   RUST_DFTB_SK_DIR — directory with mio-1-1 .skf files
//!
//! Tests skip gracefully if no OpenCL device or no SK dir (same pattern
//! as `tests/gpu_hamiltonian.rs`).

use nalgebra::DMatrix;
use rust_dftb::qmqm::gpu_matrix::{
    gamma_matvec_batched, h_scc_update_batched, matmul_full_local_batched,
    mulliken_charges_batched, residual_and_mix_batched,
};
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use rust_dftb::qmqm::{Fragment, FragmentTemplate};
use rust_dftb::{gamma_full, load_sk_for_species, HamiltonianBuilder};

const ANG2BOHR: f64 = 1.889_726_133;

/// Try to create a GpuRuntime; return None if no OpenCL device available.
fn try_runtime() -> Option<GpuRuntime> {
    match GpuRuntime::new() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

/// Build a single `Fragment` (one replica) for the given species/coords.
fn make_fragment(sk: &rust_dftb::SkData, species: &[String], coords: &[[f64; 3]]) -> Fragment {
    let template = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
    Fragment::from_template(template, coords.to_vec())
}

/// Build the per-orbital → atom index map (length n_orbs) from atom_orb_off.
fn orb_atom_map(atom_orb_off: &[u16], n_orbs: usize) -> Vec<i32> {
    let mut map = vec![0i32; n_orbs];
    for a in 0..atom_orb_off.len() - 1 {
        for mu in atom_orb_off[a] as usize..atom_orb_off[a + 1] as usize {
            map[mu] = a as i32;
        }
    }
    map
}

/// Per-atom Hubbard U extracted from SK data (order = order of `species`).
fn per_atom_u(sk: &rust_dftb::SkData, species: &[String]) -> Vec<f64> {
    let mut unique: Vec<String> = Vec::new();
    for sp in species {
        if !unique.contains(sp) {
            unique.push(sp.clone());
        }
    }
    let u_unique: Vec<f64> = unique
        .iter()
        .map(|sp| sk.onsite(sp).map(|p| p.u_hubbard).unwrap_or(0.4))
        .collect();
    species
        .iter()
        .map(|sp| {
            let idx = unique.iter().position(|s| s == sp).unwrap();
            u_unique[idx]
        })
        .collect()
}

/// Build the dense gamma matrix G[Na*Na] (row-major) for a fragment.
/// G[a*Na+b] = gamma_full(r_ab, U_a, U_b)  (r=0 → on-site = (U_a+U_b)/2).
fn build_gamma_matrix(
    coords: &[[f64; 3]],
    u_per_atom: &[f64],
) -> Vec<f32> {
    let n = coords.len();
    let mut g = vec![0.0f32; n * n];
    for a in 0..n {
        for b in 0..n {
            let dx = coords[a][0] - coords[b][0];
            let dy = coords[a][1] - coords[b][1];
            let dz = coords[a][2] - coords[b][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
            g[a * n + b] = gamma_full(r, u_per_atom[a], u_per_atom[b]) as f32;
        }
    }
    g
}

/// Flatten a nalgebra DMatrix to row-major f32.
fn flatten_f32(m: &DMatrix<f64>) -> Vec<f32> {
    let n = m.nrows();
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = m[(i, j)] as f32;
        }
    }
    out
}

/// Max abs diff between two same-length f32 slices (cast to f64).
fn max_abs_diff_slice(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

/// CPU reference: C = A·B (row-major f64).
fn cpu_matmul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += a[i * n + k] as f64 * b[k * n + j] as f64;
            }
            c[i * n + j] = s as f32;
        }
    }
    c
}

// ==================================================================
// 1. Full-local GEMM parity
// ==================================================================

fn run_gemm_parity(rt: &mut GpuRuntime, n: usize, batch: usize, label: &str) {
    // Deterministic test matrices: A[i,j] = sin(i*0.7+j*0.3), B[i,j] = cos(i*0.5+j*0.2).
    let mut a = vec![0.0f32; batch * n * n];
    let mut b = vec![0.0f32; batch * n * n];
    for bi in 0..batch {
        for i in 0..n {
            for j in 0..n {
                let off = bi * n * n + i * n + j;
                a[off] = ((i as f32) * 0.7 + (j as f32) * 0.3 + bi as f32).sin();
                b[off] = ((i as f32) * 0.5 + (j as f32) * 0.2 + bi as f32).cos();
            }
        }
    }
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let b_buf = rt.buffer_from_slice(&b).unwrap();
    let c_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();

    matmul_full_local_batched(rt, &a_buf, &b_buf, &c_buf, n, batch).unwrap();
    rt.finish().unwrap();

    let mut c_gpu = vec![0.0f32; batch * n * n];
    rt.read_buffer(&c_buf, &mut c_gpu).unwrap();

    let mut worst = 0.0f64;
    for bi in 0..batch {
        let a_b = &a[bi * n * n..(bi + 1) * n * n];
        let b_b = &b[bi * n * n..(bi + 1) * n * n];
        let c_cpu = cpu_matmul(a_b, b_b, n);
        let d = max_abs_diff_slice(&c_gpu[bi * n * n..(bi + 1) * n * n], &c_cpu);
        worst = worst.max(d);
    }
    eprintln!("GEMM {label} (n={n}, batch={batch}): max|dC| = {worst:e}");
    assert!(worst < 1e-4, "GEMM {label} parity failed: max|dC| = {worst:e}");
}

#[test]
fn test_matmul_full_local_h2() {
    let Some(mut rt) = try_runtime() else { return; };
    run_gemm_parity(&mut rt, 2, 1, "H2");
}

#[test]
fn test_matmul_full_local_n2() {
    let Some(mut rt) = try_runtime() else { return; };
    run_gemm_parity(&mut rt, 8, 1, "N2");
}

#[test]
fn test_matmul_full_local_h2o() {
    let Some(mut rt) = try_runtime() else { return; };
    run_gemm_parity(&mut rt, 6, 1, "H2O");
}

#[test]
fn test_matmul_full_local_batched() {
    let Some(mut rt) = try_runtime() else { return; };
    // 10× H2O (6×6) in one launch
    run_gemm_parity(&mut rt, 6, 10, "batched-H2O");
}

// ==================================================================
// 2. Gamma matvec parity
// ==================================================================

#[test]
fn test_gamma_matvec_h2o() {
    let Some(mut rt) = try_runtime() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let n_atoms = species.len();

    let u_per_atom = per_atom_u(&sk, &species);
    let g = build_gamma_matrix(&coords, &u_per_atom);
    // Δq: pick a non-trivial charge deviation vector (sums to 0).
    let dq = vec![0.15f32, -0.075, -0.075];

    // CPU reference: V = G · dq
    let v_cpu: Vec<f32> = (0..n_atoms)
        .map(|a| {
            let mut s = 0.0f64;
            for b in 0..n_atoms {
                s += g[a * n_atoms + b] as f64 * dq[b] as f64;
            }
            s as f32
        })
        .collect();

    let g_buf = rt.buffer_from_slice(&g).unwrap();
    let dq_buf = rt.buffer_from_slice(&dq).unwrap();
    let v_buf = rt.zero_buffer::<f32>(n_atoms).unwrap();
    gamma_matvec_batched(&mut rt, &g_buf, &dq_buf, &v_buf, n_atoms, 1).unwrap();
    rt.finish().unwrap();

    let mut v_gpu = vec![0.0f32; n_atoms];
    rt.read_buffer(&v_buf, &mut v_gpu).unwrap();

    let d = max_abs_diff_slice(&v_gpu, &v_cpu);
    eprintln!("gamma_matvec H2O: V_cpu = {:?}, V_gpu = {:?}, max|dV| = {d:e}", v_cpu, v_gpu);
    assert!(d < 1e-4, "gamma_matvec parity failed: max|dV| = {d:e}");
}

// ==================================================================
// 3. H_scc_update parity
// ==================================================================

#[test]
fn test_h_scc_update_h2o() {
    let Some(mut rt) = try_runtime() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    let builder = HamiltonianBuilder::new(sk.clone());
    let ham = builder.build_non_scc(&species, &coords).unwrap();
    let n = ham.h0.nrows();
    let frag = make_fragment(&sk, &species, &coords);
    let n_atoms = frag.template.n_atoms;
    let orb_atom = orb_atom_map(&frag.template.atom_orb_off, n);

    let h0 = flatten_f32(&ham.h0);
    let s = flatten_f32(&ham.s);
    // Per-atom potential (Hartree): pick non-trivial values.
    let v_atom = vec![0.1f32, -0.05, 0.02];

    // CPU reference: H = H0 + 0.5·S·(V[i]+V[j])
    let h_cpu: Vec<f32> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx - i * n;
            let shift = 0.5 * (v_atom[orb_atom[i] as usize] + v_atom[orb_atom[j] as usize]);
            (h0[idx] as f64 + s[idx] as f64 * shift as f64) as f32
        })
        .collect();

    let h0_buf = rt.buffer_from_slice(&h0).unwrap();
    let s_buf = rt.buffer_from_slice(&s).unwrap();
    let v_buf = rt.buffer_from_slice(&v_atom).unwrap();
    let h_buf = rt.zero_buffer::<f32>(n * n).unwrap();
    let oa_buf = rt.buffer_from_slice(&orb_atom).unwrap();
    h_scc_update_batched(&mut rt, &h0_buf, &s_buf, &v_buf, &h_buf, &oa_buf, n, n_atoms, 1).unwrap();
    rt.finish().unwrap();

    let mut h_gpu = vec![0.0f32; n * n];
    rt.read_buffer(&h_buf, &mut h_gpu).unwrap();

    let d = max_abs_diff_slice(&h_gpu, &h_cpu);
    eprintln!("h_scc_update H2O: max|dH| = {d:e}");
    assert!(d < 1e-4, "h_scc_update parity failed: max|dH| = {d:e}");
}

// ==================================================================
// 4. Mulliken charges parity
// ==================================================================

#[test]
fn test_mulliken_charges_h2o() {
    let Some(mut rt) = try_runtime() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();

    // Run CPU SCC to get a converged density matrix D and reference charges.
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc = builder.build_scc(&species, &coords, 200, 1e-9).unwrap();
    let n = scc.density.nrows();
    let n_atoms = species.len();
    let frag = make_fragment(&sk, &species, &coords);
    let orb_atom = orb_atom_map(&frag.template.atom_orb_off, n);

    let d_flat = flatten_f32(&scc.density);
    let s_flat = flatten_f32(&scc.s);

    // CPU reference: q_A = Σ_{μ∈A} (D·S)_μμ  (matches SccResult.charges)
    let q_cpu: Vec<f32> = scc.charges.iter().map(|&q| q as f32).collect();

    let d_buf = rt.buffer_from_slice(&d_flat).unwrap();
    let s_buf = rt.buffer_from_slice(&s_flat).unwrap();
    let oa_buf = rt.buffer_from_slice(&orb_atom).unwrap();
    let q_buf = rt.zero_buffer::<f32>(n_atoms).unwrap();
    mulliken_charges_batched(&mut rt, &d_buf, &s_buf, &q_buf, &oa_buf, n, n_atoms, 1).unwrap();
    rt.finish().unwrap();

    let mut q_gpu = vec![0.0f32; n_atoms];
    rt.read_buffer(&q_buf, &mut q_gpu).unwrap();

    let d = max_abs_diff_slice(&q_gpu, &q_cpu);
    eprintln!("mulliken H2O: q_cpu = {:?}, q_gpu = {:?}, max|dq| = {d:e}", q_cpu, q_gpu);
    assert!(d < 1e-4, "mulliken parity failed: max|dq| = {d:e}");
}

// ==================================================================
// 5. Residual + mix parity
// ==================================================================

#[test]
fn test_residual_and_mix() {
    let Some(mut rt) = try_runtime() else { return; };
    let n_atoms = 5usize;
    let batch = 3usize;
    let alpha = 0.35f32;

    let q_new: Vec<f32> = (0..batch * n_atoms)
        .map(|i| ((i as f32) * 0.13).sin() * 0.1)
        .collect();
    let q_old: Vec<f32> = (0..batch * n_atoms)
        .map(|i| ((i as f32) * 0.21).cos() * 0.1)
        .collect();

    // CPU reference
    let q_mixed_cpu: Vec<f32> = (0..batch * n_atoms)
        .map(|i| alpha * q_new[i] + (1.0 - alpha) * q_old[i])
        .collect();
    // Kernel contract is RMS (sqrt(Σd²/n_atoms)), matching the CPU SCC
    // tolerance convention — NOT the L2 norm.
    let rms_cpu: Vec<f32> = (0..batch)
        .map(|b| {
            let mut s = 0.0f64;
            for a in 0..n_atoms {
                let d = q_new[b * n_atoms + a] as f64 - q_old[b * n_atoms + a] as f64;
                s += d * d;
            }
            (s / n_atoms as f64).sqrt() as f32
        })
        .collect();

    let qn_buf = rt.buffer_from_slice(&q_new).unwrap();
    let qo_buf = rt.buffer_from_slice(&q_old).unwrap();
    let qm_buf = rt.zero_buffer::<f32>(batch * n_atoms).unwrap();
    let rms_buf = rt.zero_buffer::<f32>(batch).unwrap();
    let act_buf = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    residual_and_mix_batched(&mut rt, &qn_buf, &qo_buf, &qm_buf, &rms_buf, &act_buf, alpha, n_atoms, batch)
        .unwrap();
    rt.finish().unwrap();

    let mut qm_gpu = vec![0.0f32; batch * n_atoms];
    let mut rms_gpu = vec![0.0f32; batch];
    rt.read_buffer(&qm_buf, &mut qm_gpu).unwrap();
    rt.read_buffer(&rms_buf, &mut rms_gpu).unwrap();

    let dm = max_abs_diff_slice(&qm_gpu, &q_mixed_cpu);
    let dr = max_abs_diff_slice(&rms_gpu, &rms_cpu);
    eprintln!(
        "residual+mix: q_mixed_cpu={:?}, q_mixed_gpu={:?}, max|dqm|={dm:e}",
        q_mixed_cpu, qm_gpu
    );
    eprintln!(
        "residual+mix: rms_cpu={:?}, rms_gpu={:?}, max|drms|={dr:e}",
        rms_cpu, rms_gpu
    );
    assert!(dm < 1e-6, "mix parity failed: max|dqm| = {dm:e}");
    assert!(dr < 1e-6, "rms parity failed: max|drms| = {dr:e}");
}

// ==================================================================
// 6. Batched SCC kernels — 10× H2O, all kernels in sequence
// ==================================================================

#[test]
fn test_batched_scc_kernels() {
    let Some(mut rt) = try_runtime() else { return; };
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let species = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let base_coords = vec![
        [0.0, 0.0, 0.0],
        [-0.7580632005, 0.6358101311, 0.0],
        [0.7580632005, 0.6358101311, 0.0],
    ];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let batch = 10usize;

    // 10 replicas: stretch one O-H bond progressively.
    let mut all_h0 = Vec::new();
    let mut all_s = Vec::new();
    let mut all_g = Vec::new();
    let mut all_v_atom = Vec::new();
    let mut all_orb_atom = Vec::new();
    let mut all_dq = Vec::new();
    let mut all_q_old = Vec::new();
    let mut cpu_charges = Vec::new();
    let mut cpu_h_scc = Vec::new();
    let mut cpu_v_gamma = Vec::new();
    let n_atoms = species.len();
    let u_per_atom = per_atom_u(&sk, &species);

    let builder = HamiltonianBuilder::new(sk.clone());
    for bi in 0..batch {
        let stretch = 1.0 + 0.02 * bi as f64;
        let coords: Vec<[f64; 3]> = base_coords
            .iter()
            .map(|c| [c[0], c[1] * stretch, c[2]])
            .collect();
        let ham = builder.build_non_scc(&species, &coords).unwrap();
        let n = ham.h0.nrows();
        let frag = make_fragment(&sk, &species, &coords);
        let orb_atom = orb_atom_map(&frag.template.atom_orb_off, n);

        all_h0.extend(flatten_f32(&ham.h0));
        all_s.extend(flatten_f32(&ham.s));
        all_g.extend(build_gamma_matrix(&coords, &u_per_atom));
        let v = vec![0.1f32 * (bi as f32 + 1.0) / 10.0, -0.05, 0.02];
        all_v_atom.extend(v.clone());
        all_orb_atom.extend(orb_atom.clone());
        let dq = vec![0.15f32 - 0.01 * bi as f32, -0.075, -0.075 + 0.005 * bi as f32];
        all_dq.extend(dq.clone());
        all_q_old.extend(vec![0.0f32; n_atoms]);

        // CPU references
        // gamma matvec
        let g = build_gamma_matrix(&coords, &u_per_atom);
        let v_gamma: Vec<f32> = (0..n_atoms)
            .map(|a| {
                let mut s = 0.0f64;
                for b in 0..n_atoms {
                    s += g[a * n_atoms + b] as f64 * dq[b] as f64;
                }
                s as f32
            })
            .collect();
        // h_scc_update using v_gamma as the per-atom potential
        let h_scc: Vec<f32> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx - i * n;
                let shift = 0.5 * (v_gamma[orb_atom[i] as usize] + v_gamma[orb_atom[j] as usize]);
                (ham.h0[(i, j)] + ham.s[(i, j)] * shift as f64) as f32
            })
            .collect();
        cpu_h_scc.extend(h_scc);
        cpu_v_gamma.extend(v_gamma);
        // mulliken: use D = identity-like (so diag(D·S) = diag(S))
        // For a batched integration check we use the SCC density from build_scc.
        let scc = builder.build_scc(&species, &coords, 200, 1e-9).unwrap();
        cpu_charges.extend(scc.charges.iter().map(|&q| q as f32));
    }

    let n = builder.build_non_scc(&species, &base_coords).unwrap().h0.nrows();

    // --- GPU batched launches ---
    let h0_buf = rt.buffer_from_slice(&all_h0).unwrap();
    let s_buf = rt.buffer_from_slice(&all_s).unwrap();
    let g_buf = rt.buffer_from_slice(&all_g).unwrap();
    let dq_buf = rt.buffer_from_slice(&all_dq).unwrap();
    let v_buf = rt.zero_buffer::<f32>(batch * n_atoms).unwrap();
    let h_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    let oa_buf = rt.buffer_from_slice(&all_orb_atom).unwrap();

    // 1. gamma matvec → V
    gamma_matvec_batched(&mut rt, &g_buf, &dq_buf, &v_buf, n_atoms, batch).unwrap();
    // 2. h_scc_update using V
    h_scc_update_batched(&mut rt, &h0_buf, &s_buf, &v_buf, &h_buf, &oa_buf, n, n_atoms, batch)
        .unwrap();
    rt.finish().unwrap();

    let mut v_gpu = vec![0.0f32; batch * n_atoms];
    let mut h_gpu = vec![0.0f32; batch * n * n];
    rt.read_buffer(&v_buf, &mut v_gpu).unwrap();
    rt.read_buffer(&h_buf, &mut h_gpu).unwrap();

    let dv = max_abs_diff_slice(&v_gpu, &cpu_v_gamma);
    let dh = max_abs_diff_slice(&h_gpu, &cpu_h_scc);
    eprintln!("batched: gamma_matvec max|dV| = {dv:e}, h_scc_update max|dH| = {dh:e}");
    assert!(dv < 1e-4, "batched gamma_matvec failed: max|dV| = {dv:e}");
    assert!(dh < 1e-4, "batched h_scc_update failed: max|dH| = {dh:e}");

    // 3. mulliken — build D from each replica's SCC density and check charges.
    let mut all_d = Vec::new();
    for bi in 0..batch {
        let stretch = 1.0 + 0.02 * bi as f64;
        let coords: Vec<[f64; 3]> = base_coords
            .iter()
            .map(|c| [c[0], c[1] * stretch, c[2]])
            .collect();
        let scc = builder.build_scc(&species, &coords, 200, 1e-9).unwrap();
        all_d.extend(flatten_f32(&scc.density));
    }
    let d_buf = rt.buffer_from_slice(&all_d).unwrap();
    let q_buf = rt.zero_buffer::<f32>(batch * n_atoms).unwrap();
    mulliken_charges_batched(&mut rt, &d_buf, &s_buf, &q_buf, &oa_buf, n, n_atoms, batch).unwrap();
    rt.finish().unwrap();
    let mut q_gpu = vec![0.0f32; batch * n_atoms];
    rt.read_buffer(&q_buf, &mut q_gpu).unwrap();
    let dq = max_abs_diff_slice(&q_gpu, &cpu_charges);
    eprintln!("batched: mulliken max|dq| = {dq:e}");
    assert!(dq < 1e-4, "batched mulliken failed: max|dq| = {dq:e}");

    // 4. residual + mix
    let q_new = all_dq.clone();
    let q_old = all_q_old.clone();
    let qo_buf = rt.buffer_from_slice(&q_old).unwrap();
    let qm_buf = rt.zero_buffer::<f32>(batch * n_atoms).unwrap();
    let rms_buf = rt.zero_buffer::<f32>(batch).unwrap();
    let act_buf = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    residual_and_mix_batched(&mut rt, &dq_buf, &qo_buf, &qm_buf, &rms_buf, &act_buf, 0.3, n_atoms, batch)
        .unwrap();
    rt.finish().unwrap();
    let mut qm_gpu = vec![0.0f32; batch * n_atoms];
    let mut rms_gpu = vec![0.0f32; batch];
    rt.read_buffer(&qm_buf, &mut qm_gpu).unwrap();
    rt.read_buffer(&rms_buf, &mut rms_gpu).unwrap();

    let qm_cpu: Vec<f32> = (0..batch * n_atoms)
        .map(|i| 0.3 * q_new[i] + 0.7 * q_old[i])
        .collect();
    // Kernel contract is RMS (sqrt(Σd²/n_atoms)), matching the CPU SCC
    // tolerance convention — NOT the L2 norm.
    let rms_cpu: Vec<f32> = (0..batch)
        .map(|b| {
            let mut s = 0.0f64;
            for a in 0..n_atoms {
                let d = q_new[b * n_atoms + a] as f64 - q_old[b * n_atoms + a] as f64;
                s += d * d;
            }
            (s / n_atoms as f64).sqrt() as f32
        })
        .collect();
    let dm = max_abs_diff_slice(&qm_gpu, &qm_cpu);
    let dr = max_abs_diff_slice(&rms_gpu, &rms_cpu);
    eprintln!("batched: mix max|dqm| = {dm:e}, rms max|drms| = {dr:e}");
    assert!(dm < 1e-6, "batched mix failed: max|dqm| = {dm:e}");
    assert!(dr < 1e-6, "batched rms failed: max|drms| = {dr:e}");
}

// ==================================================================
// 7. Benchmark: full-local GEMM vs tiled (GpuMatrixContext::batched_gemm)
// ==================================================================

#[test]
fn bench_gemm_full_local_vs_tiled() {
    let Some(mut rt) = try_runtime() else { return; };
    use rust_dftb::qmqm::gpu_matrix::{GpuMatrixContext, MatrixKernelConfig, Transpose};

    let cfg = MatrixKernelConfig::nvidia_default();
    let ctx = match GpuMatrixContext::new(cfg) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping bench: GpuMatrixContext init failed ({e})");
            return;
        }
    };

    let sizes = [8usize, 16, 32, 48, 64];
    let batches = [1usize, 10, 100, 1000];

    eprintln!("bench: full-local vs tiled GEMM (times in µs, GPU finish)");
    eprintln!("  n   batch   full_local_us   tiled_us   ratio(tiled/full)");

    for &n in &sizes {
        for &batch in &batches {
            // Build random-ish A, B
            let mut a = vec![0.0f32; batch * n * n];
            let mut b = vec![0.0f32; batch * n * n];
            for i in 0..a.len() {
                a[i] = ((i as f32) * 0.123).sin();
                b[i] = ((i as f32) * 0.456).cos();
            }

            // Full-local buffers (on shared runtime)
            let a_buf = rt.buffer_from_slice(&a).unwrap();
            let b_buf = rt.buffer_from_slice(&b).unwrap();
            let c_fl = rt.zero_buffer::<f32>(batch * n * n).unwrap();

            // Warmup
            let _ = matmul_full_local_batched(&mut rt, &a_buf, &b_buf, &c_fl, n, batch);
            let _ = rt.finish();

            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                let _ = matmul_full_local_batched(&mut rt, &a_buf, &b_buf, &c_fl, n, batch);
            }
            let _ = rt.finish();
            let t_full = t0.elapsed().as_secs_f64() / 5.0;

            // Tiled buffers (on GpuMatrixContext)
            let a_t = ctx.buffer_from_slice(&a).unwrap();
            let b_t = ctx.buffer_from_slice(&b).unwrap();
            let c_t = ctx.zero_buffer(batch * n * n).unwrap();
            let mut c_t_host = vec![0.0f32; batch * n * n];

            let _ = ctx.batched_gemm(n, batch, Transpose::No, Transpose::No, 1.0, 0.0, &a_t, &b_t, &c_t);
            let _ = ctx.read_buffer(&c_t, &mut c_t_host);

            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                let _ = ctx.batched_gemm(n, batch, Transpose::No, Transpose::No, 1.0, 0.0, &a_t, &b_t, &c_t);
            }
            let _ = ctx.read_buffer(&c_t, &mut c_t_host);
            let t_tiled = t0.elapsed().as_secs_f64() / 5.0;

            let ratio = t_tiled / t_full.max(1e-12);
            eprintln!(
                "  {n:<3} {batch:<6} {full_us:>12.1} {tiled_us:>10.1} {ratio:>8.2}",
                full_us = t_full * 1e6,
                tiled_us = t_tiled * 1e6,
            );
        }
    }
}
