//! Honest physics tests for the dense GPU H-bond pipeline.
//!
//! These tests exist to *expose* what currently does not work. They do not skip
//! on missing SK data or a missing GPU. They do not loosen tolerances to match
//! a broken solver. A red test here is a diagnostic, not a license to cheat.
//!
//! Seams covered (review Gates 0–3):
//!   G0  NVIDIA device + SK files required (fail loud)
//!   G1  GPU H/S assembly: H2 (1×1), H2O (1×4), formic (mixed), batch=200 H2 (l_frags[128])
//!   G3.1 repulsive energy kernel vs CPU spline
//!   G3.2 four force components + total vs CPU `compute_scc_forces`
//!   G3.3 energy-gradient: CPU analytic vs FD at 1e-3 Å; GPU F vs CPU F; GPU FD vs CPU FD at 1e-2 Å
//!       (1e-3 Å GPU FD is printed only — f32 energy noise; 1e-2 Å F-vs-FD is O(h²)~1e-3 even on CPU)
//!   G3.4 coords → GPU H/S → GPU SCC → E,q vs CPU on H2O and AT/GC (N>64)
//!
//! Run:
//!   RUST_DFTB_SK_DIR=/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1 \
//!     cargo test --test gpu_hbond_physics -- --nocapture --test-threads=1

use nalgebra::DMatrix;
use ocl::{Buffer, Kernel};
use rust_dftb::methods::dftb::forces::{compute_scc_forces, parse_repulsive_spline, RepulsiveSpline};
use rust_dftb::qmqm::gpu_driver::GpuDriver;
use rust_dftb::qmqm::gpu_forces::GpuForceDriver;
use rust_dftb::qmqm::gpu_prep::GpuBatch;
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use rust_dftb::qmqm::gpu_scc_plan::GpuSccPlan;
use rust_dftb::qmqm::{Fragment, FragmentTemplate, GammaTable};
use rust_dftb::{load_sk_for_species, HamiltonianBuilder, SccResult, SkData, SystemContext};
use std::collections::HashMap;
use std::path::Path;

const ANG2BOHR: f64 = 1.889_726_133;
const MIN_NEIGH: f64 = 1.0e-2;
const MATRIX_CL: &str = include_str!("../src/qmqm/gpu_matrix_ops.cl");
const DEFAULT_SK: &str = "/home/prokop/SIMULATIONS/dftbplus/slakos/mio-1-1";

// Tight contracts from the review, ~3× measured accuracy where we have numbers.
const HS_TOL: f64 = 1e-6;
const E_EL_TOL: f64 = 1e-5;       // H2O electronic |dE| measured 1.6e-7
const E_REP_TOL: f64 = 1e-5;
const Q_TOL: f64 = 1e-4;
const FORCE_REL: f64 = 1e-4;      // H2O non-SCC measured 4e-6
const NEWTON_TOL: f64 = 1e-6;
const FD_REL: f64 = 1e-3;         // energy-gradient relative
const FD_STEP: f64 = 1e-3;        // Å

fn require_sk_dir() -> String {
    let dir = std::env::var("RUST_DFTB_SK_DIR").unwrap_or_else(|_| DEFAULT_SK.to_string());
    assert!(Path::new(&dir).is_dir(),
        "SK directory missing: {dir}. Set RUST_DFTB_SK_DIR or install mio-1-1 at {DEFAULT_SK}");
    assert!(Path::new(&dir).join("H-H.skf").is_file(), "H-H.skf missing in {dir}");
    assert!(Path::new(&dir).join("O-H.skf").is_file() || Path::new(&dir).join("H-O.skf").is_file(),
        "O-H.skf / H-O.skf missing in {dir}");
    dir
}

fn require_nvidia() -> GpuRuntime {
    let rt = GpuRuntime::new().unwrap_or_else(|e| panic!(
        "OpenCL runtime required for gpu_hbond_physics (no skip): {e}"));
    let name = rt.caps().name.clone();
    eprintln!("[gpu] NVIDIA required; local_mem={} B  CUs={}  global={} B  caps={name}",
        rt.caps().local_mem_size, rt.caps().compute_units, rt.caps().global_mem_size);
    assert!(name.to_uppercase().contains("NVIDIA"),
        "gpu_hbond_physics requires NVIDIA GPU, got '{name}'. PoCL/CPU hides fence/atomic/OOB bugs. \
         Do not treat CPU OpenCL as a GPU.");
    rt
}

fn h2() -> (Vec<String>, Vec<[f64; 3]>) {
    (vec!["H".into(), "H".into()], vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
}
fn h2o() -> (Vec<String>, Vec<[f64; 3]>) {
    (vec!["O".into(), "H".into(), "H".into()],
     vec![[0.0, 0.0, 0.0], [-0.7580632005, 0.6358101311, 0.0], [0.7580632005, 0.6358101311, 0.0]])
}
fn load_xyz(file: &str) -> (Vec<String>, Vec<[f64; 3]>) {
    let path = format!("{}/../data/xyz/{file}", env!("CARGO_MANIFEST_DIR"));
    let xyz = rust_dftb::parse_xyz(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    (xyz.species.clone(), xyz.coords.iter().map(|c| [c[0], c[1], c[2]]).collect())
}
fn formic_dimer() -> (Vec<String>, Vec<[f64; 3]>) { load_xyz("formic_dimer.xyz") }
fn at_pair() -> (Vec<String>, Vec<[f64; 3]>) { load_xyz("adenine-thymine.xyz") }
fn gc_pair() -> (Vec<String>, Vec<[f64; 3]>) { load_xyz("guanine-cytosine.xyz") }

fn make_frag(sk: &SkData, species: &[String], coords: &[[f64; 3]]) -> Fragment {
    Fragment::from_template(FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap(), coords.to_vec())
}
fn flatten_mat(m: &DMatrix<f64>) -> Vec<f32> {
    let n = m.nrows();
    let mut out = vec![0.0f32; n * n];
    for i in 0..n { for j in 0..n { out[i * n + j] = m[(i, j)] as f32; } }
    out
}
fn extract_replica(flat: &[f32], r: usize, n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(n, n);
    for i in 0..n { for j in 0..n { m[(i, j)] = flat[r * n * n + i * n + j] as f64; } }
    m
}
fn max_abs_mat(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
fn unique_species(species: &[String]) -> Vec<String> {
    let mut u = Vec::new();
    for s in species { if !u.contains(s) { u.push(s.clone()); } }
    u
}
fn species_idx(species: &[String], unique: &[String]) -> Vec<i32> {
    species.iter().map(|s| unique.iter().position(|u| u == s).unwrap() as i32).collect()
}
fn per_atom_u(sk: &SkData, species: &[String]) -> Vec<f64> {
    species.iter().map(|sp| sk.onsite(sp).map(|p| p.u_hubbard).unwrap_or(0.4)).collect()
}
fn gamma_matrix(coords: &[[f64; 3]], u: &[f64]) -> Vec<f32> {
    let n = coords.len();
    let mut g = vec![0.0f32; n * n];
    for a in 0..n {
        for b in 0..n {
            let dx = coords[a][0] - coords[b][0];
            let dy = coords[a][1] - coords[b][1];
            let dz = coords[a][2] - coords[b][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
            g[a * n + b] = rust_dftb::gamma_full(r, u[a], u[b]) as f32;
        }
    }
    g
}
fn orb_atom_map(atom_orb_off: &[u16], n_orbs: usize) -> Vec<i32> {
    let mut map = vec![0i32; n_orbs];
    for a in 0..atom_orb_off.len() - 1 {
        for mu in atom_orb_off[a] as usize..atom_orb_off[a + 1] as usize { map[mu] = a as i32; }
    }
    map
}
fn load_spline(sk_dir: &str, a: &str, b: &str) -> Option<RepulsiveSpline> {
    let p1 = format!("{sk_dir}/{a}-{b}.skf");
    let p2 = format!("{sk_dir}/{b}-{a}.skf");
    if Path::new(&p1).exists() { parse_repulsive_spline(&p1).ok().flatten() }
    else if Path::new(&p2).exists() { parse_repulsive_spline(&p2).ok().flatten() }
    else { None }
}

fn cpu_e_rep(sk_dir: &str, species: &[String], coords: &[[f64; 3]]) -> f64 {
    let mut cache: HashMap<(String, String), Option<RepulsiveSpline>> = HashMap::new();
    let n = coords.len();
    let mut e = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            let key = (species[i].clone(), species[j].clone());
            let spline = cache.entry(key.clone()).or_insert_with(|| load_spline(sk_dir, &key.0, &key.1)).clone();
            let Some(sp) = spline else { continue };
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < MIN_NEIGH * MIN_NEIGH { continue; }
            e += sp.eval(r2.sqrt() * ANG2BOHR).0;
        }
    }
    e
}

/// Pack repulsive splines into the GPU kernel layout (`gpu_matrix_ops.cl::repulsive_energy_batched`).
fn pack_splines(sk_dir: &str, unique: &[String]) -> (Vec<i32>, Vec<f32>, usize) {
    let nsp = unique.len();
    let mut splines: Vec<Option<RepulsiveSpline>> = Vec::with_capacity(nsp * nsp);
    let mut max_int = 1usize;
    for si in unique {
        for sj in unique {
            let s = load_spline(sk_dir, si, sj);
            if let Some(ref sp) = s { max_int = max_int.max(sp.x_start.len()); }
            splines.push(s);
        }
    }
    let rec = 5 + max_int + (max_int.saturating_sub(1)) * 4 + 6;
    let mut offsets = vec![-1i32; nsp * nsp];
    let mut data = Vec::new();
    for (p, s) in splines.into_iter().enumerate() {
        let Some(sp) = s else { continue };
        offsets[p] = data.len() as i32;
        let n_int = sp.x_start.len();
        data.push(f32::from_bits(n_int as u32));
        data.push(sp.cutoff as f32);
        data.push(sp.exp_coeffs[0] as f32);
        data.push(sp.exp_coeffs[1] as f32);
        data.push(sp.exp_coeffs[2] as f32);
        for k in 0..max_int { data.push(if k < n_int { sp.x_start[k] as f32 } else { 0.0 }); }
        let n_cubic = max_int.saturating_sub(1);
        for k in 0..n_cubic {
            if k < sp.sp_coeffs.len() {
                for c in 0..4 { data.push(sp.sp_coeffs[k][c] as f32); }
            } else {
                for _ in 0..4 { data.push(0.0); }
            }
        }
        for c in 0..6 { data.push(sp.sp_last_coeffs[c] as f32); }
        assert_eq!(data.len() - offsets[p] as usize, rec, "spline record length mismatch");
    }
    (offsets, data, max_int)
}

fn edm_from_scc(scc: &SccResult) -> DMatrix<f64> {
    let n = scc.eigenvectors.nrows();
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scc.eigenvalues[a].partial_cmp(&scc.eigenvalues[b]).unwrap());
    let mut edm = DMatrix::<f64>::zeros(n, n);
    for k in 0..n_occ {
        let j = idx[k];
        let col = scc.eigenvectors.column(j);
        let scaled = col * (2.0 * scc.eigenvalues[j]);
        edm += &scaled * col.transpose();
    }
    edm
}

fn flatten_dm(m: &DMatrix<f64>) -> Vec<f32> {
    let n = m.nrows();
    (0..n * n).map(|idx| { let i = idx / n; m[(i, idx - i * n)] as f32 }).collect()
}

fn force_err(cpu: &[[f64; 3]], gpu: &[f32], label: &str) -> (f64, f64, f64) {
    let n = cpu.len();
    assert_eq!(gpu.len(), 3 * n);
    let mut max_err = 0.0f64; let mut max_f = 0.0f64; let mut worst = (0usize, 0usize);
    for i in 0..n {
        for d in 0..3 {
            let c = cpu[i][d]; let g = gpu[3 * i + d] as f64;
            let e = (c - g).abs();
            if e > max_err { max_err = e; worst = (i, d); }
            max_f = max_f.max(c.abs()).max(g.abs());
            eprintln!("  {label} atom {i} dir {d}: cpu={c:.6e} gpu={g:.6e} err={e:.3e}");
        }
    }
    let rel = if max_f > 1e-12 { max_err / max_f } else { max_err };
    eprintln!("{label}: max|F|={max_f:.3e} max|err|={max_err:.3e} rel={rel:.3e} worst=atom{} dir{}", worst.0, worst.1);
    (max_err, max_f, rel)
}

fn newton_sum(gpu: &[f32], n_atoms: usize) -> f64 {
    let mut s = [0.0f64; 3];
    for i in 0..n_atoms { for d in 0..3 { s[d] += gpu[3 * i + d] as f64; } }
    s[0].abs().max(s[1].abs()).max(s[2].abs())
}

fn reconstruct_shifts(coords: &[[f64; 3]], ctx: &SystemContext, delta_q: &[f64], gamma: &GammaTable) -> Vec<f64> {
    let n = coords.len();
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        out[i] = gamma.u(ctx.atom_species[i]) * delta_q[i];
        for j in 0..n {
            if i == j { continue; }
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
            out[i] += gamma.gamma(r, ctx.atom_species[i], ctx.atom_species[j]) * delta_q[j];
        }
    }
    out
}

fn gpu_assemble(sk: &SkData, species: &[String], coords: &[[f64; 3]]) -> (Vec<f32>, Vec<f32>, GpuBatch) {
    let driver = GpuDriver::new().unwrap_or_else(|e| panic!("GpuDriver::new failed (no skip): {e}"));
    let gamma = GammaTable::from_sk_data(sk, species).unwrap();
    let frag = make_frag(sk, species, coords);
    let batch = GpuBatch::from_fragments(&[frag], sk, &gamma).unwrap();
    eprintln!("  assemble: n_frags={} buckets={} total_h={} block_types={:?}",
        batch.n_frags, batch.pair_buckets.len(), batch.total_h_elements,
        batch.pair_buckets.iter().map(|b| (b.block_type, b.n_pairs)).collect::<Vec<_>>());
    let (h, s) = driver.gpu_assemble_batched(&batch)
        .unwrap_or_else(|e| panic!("gpu_assemble_batched FAILED for species={species:?} n_atoms={}: {e}", species.len()));
    (h, s, batch)
}

fn atom_of_orb(atom_orb_off: &[u16], mu: usize) -> usize {
    for a in 0..atom_orb_off.len() - 1 {
        if mu >= atom_orb_off[a] as usize && mu < atom_orb_off[a + 1] as usize { return a; }
    }
    atom_orb_off.len() - 2
}

fn check_hs_parity(label: &str, sk: &SkData, species: &[String], coords: &[[f64; 3]]) {
    let builder = HamiltonianBuilder::new(sk.clone());
    let ham = builder.build_non_scc(species, coords).unwrap();
    let n = ham.h0.nrows();
    let tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
    let (h_flat, s_flat, _) = gpu_assemble(sk, species, coords);
    let hg = extract_replica(&h_flat, 0, n);
    let sg = extract_replica(&s_flat, 0, n);
    // In-range: r < last SK grid point. Tail: r ≥ last grid (zero-pad / cutoff, no Neville).
    // GPU and CPU must both be ~0 in the tail.
    let mut d_in = 0.0f64; let mut d_tail = 0.0f64; let mut d_s_in = 0.0f64;
    let mut worst_all = (0usize, 0usize);
    let mut d_all = 0.0f64; let mut n_tail_cpu_big = 0usize;
    for i in 0..n {
        for j in 0..n {
            let eh = (ham.h0[(i, j)] - hg[(i, j)]).abs();
            let es = (ham.s[(i, j)] - sg[(i, j)]).abs();
            if eh > d_all { d_all = eh; worst_all = (i, j); }
            let ai = atom_of_orb(&tmpl.atom_orb_off, i);
            let aj = atom_of_orb(&tmpl.atom_orb_off, j);
            if ai == aj {
                if eh > d_in { d_in = eh; }
                d_s_in = d_s_in.max(es);
                continue;
            }
            let dx = coords[ai][0] - coords[aj][0];
            let dy = coords[ai][1] - coords[aj][1];
            let dz = coords[ai][2] - coords[aj][2];
            let r_bohr = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
            let Some(tab) = sk.get_pair(&species[ai], &species[aj]) else { continue };
            let r_grid = tab.h.n_grid() as f64 * tab.h.dr;
            if r_bohr < r_grid {
                if eh > d_in { d_in = eh; }
                d_s_in = d_s_in.max(es);
            } else {
                d_tail = d_tail.max(eh);
                if ham.h0[(i, j)].abs() > 1e-3 { n_tail_cpu_big += 1; }
            }
        }
    }
    let (wi, wj) = worst_all;
    let ai = atom_of_orb(&tmpl.atom_orb_off, wi);
    let aj = atom_of_orb(&tmpl.atom_orb_off, wj);
    let r_ang = {
        let dx = coords[ai][0] - coords[aj][0];
        let dy = coords[ai][1] - coords[aj][1];
        let dz = coords[ai][2] - coords[aj][2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    eprintln!("{label}: N={n} in-range max|dH|={d_in:.3e} max|dS|={d_s_in:.3e}  tail max|dH|={d_tail:.3e}  all max|dH|={d_all:.3e}");
    eprintln!("  worst-all orb ({wi},{wj}) atoms {ai}({})-{aj}({}) r={r_ang:.4} Å", species[ai], species[aj]);
    eprintln!("    H_cpu={:.6e} H_gpu={:.6e}  S_cpu={:.6e} S_gpu={:.6e}", ham.h0[(wi, wj)], hg[(wi, wj)], ham.s[(wi, wj)], sg[(wi, wj)]);
    eprintln!("    CPU |H|>1e-3 past last SK grid: {n_tail_cpu_big} elements (must be 0; extra zero knots, not Neville)");
    assert!(d_in.is_finite() && d_s_in.is_finite(), "{label}: non-finite in-range H/S");
    assert!(d_in < HS_TOL, "{label} in-range H assembly failed: max|dH|={d_in:.3e} > {HS_TOL:.1e}");
    assert!(d_s_in < HS_TOL, "{label} in-range S assembly failed: max|dS|={d_s_in:.3e} > {HS_TOL:.1e}");
    if n_tail_cpu_big > 0 {
        panic!("{label} CPU SK interpolator is unphysical past last grid: {n_tail_cpu_big} elements with |H_cpu|>1e-3 (worst |dH|={d_tail:.3e} H_cpu={:.6e} H_gpu={:.6e} at {}-{} r={r_ang:.4} Å). Tail must be ~0 on both CPU and GPU.",
            ham.h0[(wi, wj)], hg[(wi, wj)], species[ai], species[aj]);
    }
    assert!(d_tail < HS_TOL, "{label} tail H assembly failed: max|dH|={d_tail:.3e} > {HS_TOL:.1e} (CPU/GPU must both be ~0 past last SK grid)");
}

fn gpu_e_rep(rt: &mut GpuRuntime, sk_dir: &str, species: &[String], coords: &[[f64; 3]]) -> f32 {
    let unique = unique_species(species);
    let (offsets, data, max_int) = pack_splines(sk_dir, &unique);
    assert!(!data.is_empty(), "no repulsive splines packed for {species:?} from {sk_dir}");
    let n_atoms = species.len();
    let crd: Vec<f32> = coords.iter().flat_map(|c| c.iter().map(|&x| (x * ANG2BOHR) as f32)).collect();
    let spc = species_idx(species, &unique);
    let source = MATRIX_CL.replace("#define REP_MAX_INTERVALS 30", &format!("#define REP_MAX_INTERVALS {max_int}"));
    let prog = rt.build_program(&source).unwrap();
    let buf_c = rt.buffer_from_slice(&crd).unwrap();
    let buf_s = rt.buffer_from_slice(&spc).unwrap();
    let buf_o = rt.buffer_from_slice(&offsets).unwrap();
    let buf_d = rt.buffer_from_slice(&data).unwrap();
    let buf_e = rt.zero_buffer::<f32>(1).unwrap();
    let wg = 256usize;
    let k = Kernel::builder().program(&prog).name("repulsive_energy_batched").queue(rt.queue().clone())
        .global_work_size(wg).local_work_size(wg)
        .arg(n_atoms as i32).arg(1i32).arg(&buf_c).arg(&buf_s).arg(&buf_o)
        .arg(unique.len() as i32).arg(&buf_d).arg(&buf_e)
        .build().unwrap();
    unsafe { k.enq().unwrap(); }
    let mut out = vec![0.0f32; 1];
    rt.read_buffer(&buf_e, &mut out).unwrap();
    out[0]
}

fn scc_cpu(sk: &SkData, species: &[String], coords: &[[f64; 3]]) -> SccResult {
    HamiltonianBuilder::new(sk.clone()).build_scc(species, coords, 200, 1e-9).unwrap()
}

fn plan_scc(rt: &mut GpuRuntime, sk: &SkData, species: &[String], coords: &[[f64; 3]], scc: &SccResult) -> (GpuSccPlan, Buffer<f32>, Buffer<f32>, Buffer<f32>, Buffer<f32>, Buffer<i32>) {
    let tmpl = FragmentTemplate::new(sk, species.to_vec(), coords.to_vec()).unwrap();
    let n = scc.h0.nrows();
    let n_atoms = species.len();
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    let h0 = rt.buffer_from_slice(&flatten_mat(&scc.h0)).unwrap();
    let s = rt.buffer_from_slice(&flatten_mat(&scc.s)).unwrap();
    let g = rt.buffer_from_slice(&gamma_matrix(coords, &per_atom_u(sk, species))).unwrap();
    let q0f: Vec<f32> = scc.q0.iter().map(|&q| q as f32).collect();
    let q0 = rt.buffer_from_slice(&q0f).unwrap();
    let oa = rt.buffer_from_slice(&orb_atom_map(&tmpl.atom_orb_off, n)).unwrap();
    let mut plan = GpuSccPlan::new(rt, &s, n, n_atoms, 1).expect("GpuSccPlan::new");
    plan.set_initial_charges(rt, &q0f).unwrap();
    let mut rms = f32::INFINITY;
    let mut n_iters = 0;
    for iter in 0..500 {
        n_iters = iter + 1;
        rms = plan.scc_step_diis(rt, &h0, &s, &g, &q0, &oa, n_occ, 0.3).expect("scc_step_diis");
        if rms < 1e-6 { break; }
    }
    assert!(rms < 1e-6, "GPU SCC did not converge: rms={rms:.3e} after {n_iters} iters");
    eprintln!("  GPU SCC converged in {n_iters} iters, rms={rms:.3e}");
    (plan, h0, s, g, q0, oa)
}

// ==================================================================
// G0 — harness honesty
// ==================================================================

#[test]
fn test_harness_nvidia_and_sk() {
    let sk = require_sk_dir();
    let rt = require_nvidia();
    eprintln!("harness OK: SK={sk} device={}", rt.caps().name);
}

// ==================================================================
// G1 — H/S assembly (this is where formic currently dies)
// ==================================================================

#[test]
fn test_hs_assembly_h2_1x1() {
    let _rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    check_hs_parity("H2", &sk, &sp, &xyz);
}

#[test]
fn test_hs_assembly_h2o_1x4() {
    let _rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    check_hs_parity("H2O", &sk, &sp, &xyz);
}

#[test]
fn test_hs_assembly_formic_mixed() {
    let _rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = formic_dimer();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    eprintln!("formic dimer: n_atoms={} species={:?}", sp.len(), unique_species(&sp));
    check_hs_parity("formic_dimer", &sk, &sp, &xyz);
}

#[test]
fn test_hs_assembly_at() {
    let _rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = at_pair();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    eprintln!("AT: n_atoms={} species={:?}", sp.len(), unique_species(&sp));
    check_hs_parity("AT", &sk, &sp, &xyz);
}

#[test]
fn test_hs_assembly_gc() {
    let _rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = gc_pair();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    eprintln!("GC: n_atoms={} species={:?}", sp.len(), unique_species(&sp));
    check_hs_parity("GC", &sk, &sp, &xyz);
}

/// R11: `__local Fragment l_frags[128]` must not silently index garbage at batch=200.
/// Uses H2 (1×1, currently the only assembly path that launches) so this test
/// isolates the replica-cap bug from the mixed-block crash.
#[test]
fn test_hs_assembly_batch200_h2_replica_cap() {
    let _rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let species = vec!["H".to_string(), "H".to_string()];
    let sk = load_sk_for_species(&sk_dir, &species).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &species).unwrap();
    let builder = HamiltonianBuilder::new(sk.clone());
    let n_rep = 200usize;
    let mut frags = Vec::with_capacity(n_rep);
    let mut cpu_h = Vec::with_capacity(n_rep);
    for r in 0..n_rep {
        let bl = 0.70 + 0.002 * (r as f64);
        let coords = vec![[r as f64 * 20.0, 0.0, 0.0], [r as f64 * 20.0 + bl, 0.0, 0.0]];
        cpu_h.push(builder.build_non_scc(&species, &coords).unwrap().h0);
        frags.push(make_frag(&sk, &species, &coords));
    }
    let batch = GpuBatch::from_fragments(&frags, &sk, &gamma).unwrap();
    assert_eq!(batch.n_frags, n_rep);
    let driver = GpuDriver::new().unwrap_or_else(|e| panic!("GpuDriver::new: {e}"));
    let (h_flat, _s_flat) = driver.gpu_assemble_batched(&batch)
        .unwrap_or_else(|e| panic!("batch=200 H2 assemble failed: {e}"));
    let n = 2;
    let mut worst = 0.0f64;
    let mut worst_r = 0usize;
    // Check replica 0, 127, 128, 199 — the boundary the 128-cap bug lives on.
    for r in [0usize, 127, 128, 199] {
        let dh = max_abs_mat(&cpu_h[r], &extract_replica(&h_flat, r, n));
        eprintln!("  replica {r}: max|dH|={dh:.3e}");
        if dh > worst { worst = dh; worst_r = r; }
    }
    eprintln!("batch=200 H2 worst replica {worst_r}: max|dH|={worst:.3e}");
    assert!(worst < HS_TOL, "batch=200 H/S parity failed at replica {worst_r}: max|dH|={worst:.3e} \
        (if replica≥128 is uniquely bad, this is the l_frags[128] bug)");
}

// ==================================================================
// G3.1 — repulsive energy
// ==================================================================

/// Isolates the CPU SK tail: H-H past last grid point must be ~0, not a bonding integral.
#[test]
fn test_cpu_sk_tail_hh_near_zero() {
    let sk_dir = require_sk_dir();
    let sk = load_sk_for_species(&sk_dir, &["H".into()]).unwrap();
    let tab = sk.get_pair("H", "H").unwrap_or_else(|| panic!("missing H-H SK table in {sk_dir}"));
    let r_grid = tab.h.n_grid() as f64 * tab.h.dr;
    let r_max = tab.cutoff();
    let mut h = [0.0f64; 4]; let mut s = [0.0f64; 4];
    tab.eval_shell_integrals_into(0, 0, r_grid + 0.02, &mut h, &mut s).unwrap();
    eprintln!("CPU H-H ss: n_grid={} dr={} r_grid={r_grid:.4} r_max={r_max:.4}", tab.h.n_grid(), tab.h.dr);
    eprintln!("  in zero-pad r={:.4}  Hss={:.6e} Sss={:.6e}", r_grid + 0.02, h[0], s[0]);
    assert!(h[0].abs() < 1e-3, "CPU H-H ss just past last grid is unphysical: Hss={:.6e}", h[0]);
    let r_pair = r_grid + 0.41; // 5.50 Å H-H on AT
    tab.eval_shell_integrals_into(0, 0, r_pair, &mut h, &mut s).unwrap();
    eprintln!("  AT-pair r={r_pair:.4}  Hss={:.6e} Sss={:.6e} (must be 0 past r_max)", h[0], s[0]);
    assert!(h[0].abs() < 1e-12 && s[0].abs() < 1e-12,
        "H-H at r={r_pair:.4} Bohr is past r_max={r_max:.4} and must be exact 0, got Hss={:.6e} Sss={:.6e}", h[0], s[0]);
}

/// Analytic dHss/dr of the production B-spline must match FD of the same Hss (CPU f64).
#[test]
fn test_cpu_sk_analytic_deriv_hh() {
    let sk_dir = require_sk_dir();
    let sk = load_sk_for_species(&sk_dir, &["H".into()]).unwrap();
    let tab = sk.get_pair("H", "H").unwrap_or_else(|| panic!("missing H-H SK table in {sk_dir}"));
    let h_step = 1e-6f64;
    let mut vmax = 0.0f64; let mut dmax = 0.0f64; let mut worst_r = 0.0f64;
    let mut h = [0.0f64; 4]; let mut s = [0.0f64; 4];
    let mut dh = [0.0f64; 4]; let mut ds = [0.0f64; 4];
    let mut hp = [0.0f64; 4]; let mut hm = [0.0f64; 4];
    let mut sp = [0.0f64; 4]; let mut sm = [0.0f64; 4];
    // Bonding to mid-range: 1.2–8 Bohr. Skip the last original sample (zero-pad BC).
    let r_last = tab.h.n_grid() as f64 * tab.h.dr - 4.0 * tab.h.dr;
    let mut r = 1.2f64;
    while r < r_last {
        tab.eval_shell_integrals_and_derivs_into(0, 0, r, &mut h, &mut s, &mut dh, &mut ds).unwrap();
        tab.eval_shell_integrals_into(0, 0, r + h_step, &mut hp, &mut sp).unwrap();
        tab.eval_shell_integrals_into(0, 0, r - h_step, &mut hm, &mut sm).unwrap();
        let fd = (hp[0] - hm[0]) / (2.0 * h_step);
        let rel = (dh[0] - fd).abs() / fd.abs().max(1e-8);
        if rel > vmax { vmax = rel; worst_r = r; dmax = (dh[0] - fd).abs(); }
        r += 0.37;
    }
    eprintln!("CPU H-H analytic dHss/dr vs FD: max rel={vmax:.3e} |d|={dmax:.3e} at r={worst_r:.4} Bohr");
    tab.eval_shell_integrals_and_derivs_into(0, 0, 1.4, &mut h, &mut s, &mut dh, &mut ds).unwrap();
    eprintln!("  at r=1.4 Bohr (H2 bond): Hss={:.6e} dHss/dr={:.6e}", h[0], dh[0]);
    assert!(vmax < 1e-6, "CPU analytic SK derivative disagrees with FD of V: rel={vmax:.3e} at r={worst_r:.4}");
}

#[test]
fn test_e_rep_kernel_h2o() {
    let mut rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let e_cpu = cpu_e_rep(&sk_dir, &sp, &xyz);
    let e_gpu = gpu_e_rep(&mut rt, &sk_dir, &sp, &xyz) as f64;
    eprintln!("H2O E_rep: cpu={e_cpu:.8} gpu={e_gpu:.8} |dE|={:.3e}", (e_cpu - e_gpu).abs());
    assert!(e_cpu.is_finite() && e_gpu.is_finite(), "non-finite E_rep");
    assert!(e_cpu.abs() > 1e-6, "H2O E_rep unexpectedly ~0 ({e_cpu}) — spline packing or SK files wrong");
    assert!((e_cpu - e_gpu).abs() < E_REP_TOL, "E_rep kernel parity failed: |dE|={:.3e} cpu={e_cpu} gpu={e_gpu}", (e_cpu - e_gpu).abs());
}

#[test]
fn test_e_rep_in_scc_energy_h2o() {
    let mut rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let scc = scc_cpu(&sk, &sp, &xyz);
    let e_rep = cpu_e_rep(&sk_dir, &sp, &xyz);
    let e_tot_cpu = scc.energy + e_rep;
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    let unique = unique_species(&sp);
    let (off, data, max_int) = pack_splines(&sk_dir, &unique);
    let crd: Vec<f32> = xyz.iter().flat_map(|c| c.iter().map(|&x| (x * ANG2BOHR) as f32)).collect();
    let (mut plan, h0, s, g, q0, oa) = plan_scc(&mut rt, &sk, &sp, &xyz, &scc);
    plan.set_repulsive_splines(&mut rt, &crd, &species_idx(&sp, &unique), &off, &data, unique.len(), max_int)
        .unwrap_or_else(|e| panic!("set_repulsive_splines failed: {e}"));
    let e_gpu = plan.compute_energy(&mut rt, &h0, &s, &g, &q0, &oa, n_occ).unwrap()[0] as f64;
    eprintln!("H2O energy: E_el_cpu={:.8} E_rep={:.8} E_tot_cpu={:.8} E_gpu={:.8}", scc.energy, e_rep, e_tot_cpu, e_gpu);
    eprintln!("  |E_gpu - E_el|={:.3e}  |E_gpu - E_tot|={:.3e}", (e_gpu - scc.energy).abs(), (e_gpu - e_tot_cpu).abs());
    assert!((e_gpu - e_tot_cpu).abs() < E_EL_TOL.max(E_REP_TOL),
        "GpuSccPlan::compute_energy must equal E_el + E_rep. |dE|={:.3e} (E_el={:.8} E_rep={:.8} E_gpu={:.8}). \
         If |E_gpu-E_el| ≪ |E_rep|, repulsive energy is not actually added.",
        (e_gpu - e_tot_cpu).abs(), scc.energy, e_rep, e_gpu);
}

// ==================================================================
// G3.2 — four force components
// ==================================================================

fn run_force_components(label: &str, sk_dir: &str, sk: &SkData, species: &[String], coords: &[[f64; 3]]) {
    let mut rt = require_nvidia();
    let driver = GpuForceDriver::new(&mut rt).unwrap();
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc = builder.build_scc(species, coords, 200, 1e-9).unwrap();
    let cpu = compute_scc_forces(&builder, species, coords, &scc).unwrap();
    let ctx = SystemContext::from_sk_data(sk, species).unwrap();
    let gamma = GammaTable::from_sk_data(sk, species).unwrap();
    let unique = unique_species(species);
    let n_atoms = species.len();
    let delta_q: Vec<f64> = scc.charges.iter().zip(scc.q0.iter()).map(|(q, q0)| q - q0).collect();
    let shifts = reconstruct_shifts(coords, &ctx, &delta_q, &gamma);
    let edm = edm_from_scc(&scc);
    let dm_f = flatten_dm(&scc.density);
    let edm_f = flatten_dm(&edm);
    let v_shift: Vec<f32> = shifts.iter().map(|&x| x as f32).collect();
    let dq_f: Vec<f32> = delta_q.iter().map(|&x| x as f32).collect();
    let u_hub: Vec<f32> = unique.iter().map(|s| sk.onsite(s).unwrap().u_hubbard as f32).collect();
    let crd_ang: Vec<f32> = coords.iter().flat_map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect();
    let crd_bohr: Vec<f32> = coords.iter().flat_map(|c| [(c[0] * ANG2BOHR) as f32, (c[1] * ANG2BOHR) as f32, (c[2] * ANG2BOHR) as f32]).collect();
    let spc = species_idx(species, &unique);
    let (off, data, max_int) = pack_splines(sk_dir, &unique);
    let frag = make_frag(sk, species, coords);
    let batch = GpuBatch::from_fragments(&[frag], sk, &gamma).unwrap();

    let f_nonscc = driver.gpu_force_batched(&rt, &batch, &dm_f, &edm_f).unwrap();
    let f_shift = driver.gpu_scc_shift_force_batched(&rt, &batch, &dm_f, &v_shift).unwrap();
    let f_gamma = driver.gpu_gamma_deriv_force_batched(&rt, n_atoms, 1, &crd_ang, &spc, &dq_f, &u_hub, unique.len()).unwrap();
    let f_rep = driver.gpu_repulsive_force_batched(&mut rt, n_atoms, 1, &crd_bohr, &spc, &off, &data, unique.len(), max_int).unwrap();

    let mut f_tot = vec![0.0f32; 3 * n_atoms];
    for i in 0..3 * n_atoms { f_tot[i] = f_nonscc[i] + f_shift[i] + f_gamma[i] + f_rep[i]; }

    let (_, _, r0) = force_err(&cpu.non_scc, &f_nonscc, &format!("{label}/nonSCC"));
    let (_, _, r1) = force_err(&cpu.scc_shift, &f_shift, &format!("{label}/shift"));
    let (_, _, r2) = force_err(&cpu.scc_dc, &f_gamma, &format!("{label}/gamma"));
    let (_, _, r3) = force_err(&cpu.repulsive, &f_rep, &format!("{label}/rep"));
    let (_, max_f, rt_tot) = force_err(&cpu.forces, &f_tot, &format!("{label}/TOTAL"));
    let nsum = newton_sum(&f_tot, n_atoms);
    eprintln!("{label} Newton |ΣF|={nsum:.3e}  max|F|={max_f:.3e}");

    assert!(r0 < FORCE_REL, "{label} non-SCC force rel={r0:.3e} > {FORCE_REL:.1e}");
    assert!(r1 < FORCE_REL, "{label} SCC-shift force rel={r1:.3e} > {FORCE_REL:.1e} (untested kernel until this test)");
    assert!(r2 < FORCE_REL, "{label} gamma-deriv force rel={r2:.3e} > {FORCE_REL:.1e} (untested kernel until this test)");
    assert!(r3 < FORCE_REL, "{label} repulsive force rel={r3:.3e} > {FORCE_REL:.1e} (untested kernel until this test)");
    assert!(rt_tot < FORCE_REL, "{label} TOTAL force rel={rt_tot:.3e} > {FORCE_REL:.1e}");
    assert!(nsum < NEWTON_TOL, "{label} Newton 3rd law violated: |ΣF|={nsum:.3e}");
}

#[test]
fn test_force_four_components_h2o() {
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    run_force_components("H2O", &sk_dir, &sk, &sp, &xyz);
}

#[test]
fn test_force_translation_invariance_h2o() {
    let sk_dir = require_sk_dir();
    let (sp, xyz0) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let shift = [0.31, -0.17, 0.22];
    let xyz1: Vec<[f64; 3]> = xyz0.iter().map(|c| [c[0] + shift[0], c[1] + shift[1], c[2] + shift[2]]).collect();
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc0 = builder.build_scc(&sp, &xyz0, 200, 1e-9).unwrap();
    let scc1 = builder.build_scc(&sp, &xyz1, 200, 1e-9).unwrap();
    let f0 = compute_scc_forces(&builder, &sp, &xyz0, &scc0).unwrap();
    let f1 = compute_scc_forces(&builder, &sp, &xyz1, &scc1).unwrap();
    let mut max_d = 0.0f64;
    for i in 0..sp.len() {
        for d in 0..3 { max_d = max_d.max((f0.forces[i][d] - f1.forces[i][d]).abs()); }
    }
    eprintln!("CPU translation invariance: max|ΔF|={max_d:.3e} (shift={shift:?})");
    assert!(max_d < 1e-5, "CPU forces not translation-invariant: max|ΔF|={max_d:.3e}");

    let mut rt = require_nvidia();
    let driver = GpuForceDriver::new(&mut rt).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &sp).unwrap();
    let batch0 = GpuBatch::from_fragments(&[make_frag(&sk, &sp, &xyz0)], &sk, &gamma).unwrap();
    let batch1 = GpuBatch::from_fragments(&[make_frag(&sk, &sp, &xyz1)], &sk, &gamma).unwrap();
    let dm0 = flatten_dm(&scc0.density); let edm0 = flatten_dm(&edm_from_scc(&scc0));
    let dm1 = flatten_dm(&scc1.density); let edm1 = flatten_dm(&edm_from_scc(&scc1));
    let g0 = driver.gpu_force_batched(&rt, &batch0, &dm0, &edm0).unwrap();
    let g1 = driver.gpu_force_batched(&rt, &batch1, &dm1, &edm1).unwrap();
    let mut max_g = 0.0f64;
    for i in 0..g0.len() { max_g = max_g.max((g0[i] as f64 - g1[i] as f64).abs()); }
    eprintln!("GPU non-SCC translation invariance: max|ΔF|={max_g:.3e}");
    assert!(max_g < 1e-4, "GPU non-SCC forces not translation-invariant: max|ΔF|={max_g:.3e}");
}

// ==================================================================
// G3.3 — energy is the gradient of the force
// ==================================================================

fn fd_vs_analytic(label: &str, sk_dir: &str, sk: &SkData, species: &[String], coords: &[[f64; 3]]) {
    let builder = HamiltonianBuilder::new(sk.clone());
    let scc0 = builder.build_scc(species, coords, 400, 1e-10).unwrap();
    let f = compute_scc_forces(&builder, species, coords, &scc0).unwrap();
    let e0 = scc0.energy + cpu_e_rep(sk_dir, species, coords);
    // Probe the largest force component so the FD signal is well above noise.
    let mut imax = 0usize; let mut dmax = 0usize; let mut amax = 0.0f64;
    for i in 0..species.len() {
        for d in 0..3 { if f.forces[i][d].abs() > amax { amax = f.forces[i][d].abs(); imax = i; dmax = d; } }
    }
    let mut xp = coords.to_vec(); let mut xm = coords.to_vec();
    xp[imax][dmax] += FD_STEP; xm[imax][dmax] -= FD_STEP;
    let ep = builder.build_scc(species, &xp, 400, 1e-10).unwrap().energy + cpu_e_rep(sk_dir, species, &xp);
    let em = builder.build_scc(species, &xm, 400, 1e-10).unwrap().energy + cpu_e_rep(sk_dir, species, &xm);
    let fd = -(ep - em) / (2.0 * FD_STEP); // dE/dR = -F
    let ana = f.forces[imax][dmax];
    let rel = (fd - ana).abs() / amax.max(1e-8);
    eprintln!("{label} CPU energy-gradient: atom {imax} dir {dmax}  F_ana={ana:.6e}  F_fd={fd:.6e}  rel={rel:.3e}  E0={e0:.8}");
    assert!(rel < FD_REL, "{label} CPU analytic force is not the gradient of E_el+E_rep: rel={rel:.3e} ana={ana:.6e} fd={fd:.6e}");
}

#[test]
fn test_cpu_energy_gradient_h2o() {
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    fd_vs_analytic("H2O", &sk_dir, &sk, &sp, &xyz);
}

/// GPU energy (GpuSccPlan, including E_rep) vs GPU total force, same charge-consistent state.
/// This is the test that catches R7 (energy/forces at different q) and missing force terms.
#[test]
fn test_gpu_energy_gradient_h2o() {
    let mut rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let n_atoms = sp.len();
    let unique = unique_species(&sp);
    let (off, data, max_int) = pack_splines(&sk_dir, &unique);
    let n_occ_fn = |scc: &SccResult| (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;

    let energy_at = |rt: &mut GpuRuntime, coords: &[[f64; 3]]| -> f64 {
        let scc = scc_cpu(&sk, &sp, coords);
        let n_occ = n_occ_fn(&scc);
        let crd: Vec<f32> = coords.iter().flat_map(|c| c.iter().map(|&x| (x * ANG2BOHR) as f32)).collect();
        let (mut plan, h0, s, g, q0, oa) = plan_scc(rt, &sk, &sp, coords, &scc);
        plan.set_repulsive_splines(rt, &crd, &species_idx(&sp, &unique), &off, &data, unique.len(), max_int).unwrap();
        plan.compute_energy(rt, &h0, &s, &g, &q0, &oa, n_occ).unwrap()[0] as f64
    };

    let scc0 = scc_cpu(&sk, &sp, &xyz);
    let builder = HamiltonianBuilder::new(sk.clone());
    let cpu_f = compute_scc_forces(&builder, &sp, &xyz, &scc0).unwrap();
    let mut imax = 0usize; let mut dmax = 0usize; let mut amax = 0.0f64;
    for i in 0..n_atoms {
        for d in 0..3 { if cpu_f.forces[i][d].abs() > amax { amax = cpu_f.forces[i][d].abs(); imax = i; dmax = d; } }
    }
    let driver = GpuForceDriver::new(&mut rt).unwrap();
    let gamma = GammaTable::from_sk_data(&sk, &sp).unwrap();
    let ctx = SystemContext::from_sk_data(&sk, &sp).unwrap();
    let delta_q: Vec<f64> = scc0.charges.iter().zip(scc0.q0.iter()).map(|(q, q0)| q - q0).collect();
    let shifts = reconstruct_shifts(&xyz, &ctx, &delta_q, &gamma);
    let batch = GpuBatch::from_fragments(&[make_frag(&sk, &sp, &xyz)], &sk, &gamma).unwrap();
    let dm = flatten_dm(&scc0.density); let edm = flatten_dm(&edm_from_scc(&scc0));
    let v_shift: Vec<f32> = shifts.iter().map(|&x| x as f32).collect();
    let dq_f: Vec<f32> = delta_q.iter().map(|&x| x as f32).collect();
    let u_hub: Vec<f32> = unique.iter().map(|s| sk.onsite(s).unwrap().u_hubbard as f32).collect();
    let crd_ang: Vec<f32> = xyz.iter().flat_map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect();
    let crd_bohr: Vec<f32> = xyz.iter().flat_map(|c| [(c[0] * ANG2BOHR) as f32, (c[1] * ANG2BOHR) as f32, (c[2] * ANG2BOHR) as f32]).collect();
    let spc = species_idx(&sp, &unique);
    let f0 = driver.gpu_force_batched(&rt, &batch, &dm, &edm).unwrap();
    let f1 = driver.gpu_scc_shift_force_batched(&rt, &batch, &dm, &v_shift).unwrap();
    let f2 = driver.gpu_gamma_deriv_force_batched(&rt, n_atoms, 1, &crd_ang, &spc, &dq_f, &u_hub, unique.len()).unwrap();
    let f3 = driver.gpu_repulsive_force_batched(&mut rt, n_atoms, 1, &crd_bohr, &spc, &off, &data, unique.len(), max_int).unwrap();
    let f_gpu = f0[3 * imax + dmax] + f1[3 * imax + dmax] + f2[3 * imax + dmax] + f3[3 * imax + dmax];
    let e0 = energy_at(&mut rt, &xyz);
    eprintln!("H2O GPU energy-gradient: atom {imax} dir {dmax}  E(R)={e0:.8}  F_gpu={:.6e}  F_cpu={:.6e}", f_gpu, cpu_f.forces[imax][dmax]);
    assert!(e0.is_finite(), "non-finite GPU energy at undisplaced geometry: {e0}");

    let fd_at = |rt: &mut GpuRuntime, h: f64| -> (f64, f64, f64) {
        let mut xp = xyz.clone(); let mut xm = xyz.clone();
        xp[imax][dmax] += h; xm[imax][dmax] -= h;
        let ep = energy_at(rt, &xp);
        let em = energy_at(rt, &xm);
        let fd = -(ep - em) / (2.0 * h);
        (fd, ep, em)
    };
    let (fd_fine, ep_f, em_f) = fd_at(&mut rt, FD_STEP);
    let rel_fine = (fd_fine - f_gpu as f64).abs() / amax.max(1e-8);
    eprintln!("  h={FD_STEP:.1e} Å  E(+h)={ep_f:.8} E(-h)={em_f:.8}  F_fd={fd_fine:.6e}  rel={rel_fine:.3e}  (f32 stencil; diagnostic)");
    let h_coarse = 1e-2f64;
    let (fd, ep, em) = fd_at(&mut rt, h_coarse);
    let rel = (fd - f_gpu as f64).abs() / amax.max(1e-8);
    eprintln!("  h={h_coarse:.1e} Å  E(+h)={ep:.8} E(-h)={em:.8}  F_fd={fd:.6e}  vs F_gpu rel={rel:.3e}  (diagnostic; O(h²)~1e-3)");
    // Same stencil on CPU E: O(h²) truncation at 1e-2 Å is ~1e-3 even for f64 analytic vs FD.
    let mut xp = xyz.clone(); let mut xm = xyz.clone();
    xp[imax][dmax] += h_coarse; xm[imax][dmax] -= h_coarse;
    let e_cpu0 = scc0.energy + cpu_e_rep(&sk_dir, &sp, &xyz);
    let e_cp = scc_cpu(&sk, &sp, &xp).energy + cpu_e_rep(&sk_dir, &sp, &xp);
    let e_cm = scc_cpu(&sk, &sp, &xm).energy + cpu_e_rep(&sk_dir, &sp, &xm);
    let fd_cpu = -(e_cp - e_cm) / (2.0 * h_coarse);
    let rel_cpu = (fd_cpu - cpu_f.forces[imax][dmax]).abs() / amax.max(1e-8);
    let rel_fd = (fd - fd_cpu).abs() / amax.max(1e-8);
    eprintln!("  CPU same h={h_coarse:.1e} Å  F_fd={fd_cpu:.6e}  vs F_ana rel={rel_cpu:.3e}  (truncation)");
    eprintln!("  GPU vs CPU energy at R0: |dE|={:.3e}  FD-vs-FD rel={rel_fd:.3e}", (e0 - e_cpu0).abs());
    assert!(ep.is_finite() && em.is_finite(), "non-finite GPU energy in FD stencil");
    assert!((f_gpu as f64 - cpu_f.forces[imax][dmax]).abs() / amax.max(1e-8) < FORCE_REL,
        "GPU total force disagrees with CPU: F_gpu={f_gpu} F_cpu={}", cpu_f.forces[imax][dmax]);
    // Do not assert F vs FD at h=1e-2 against FD_REL: even CPU analytic vs CPU FD is ~1e-3 there.
    // The f32 energy-surface contract is GPU FD vs CPU FD at the same stencil.
    assert!(rel_fd < FD_REL, "GPU energy FD disagrees with CPU energy FD at h={h_coarse}: rel={rel_fd:.3e} F_fd_gpu={fd} F_fd_cpu={fd_cpu}");
}

// ==================================================================
// G3.4 — full chain: GPU-assembled H/S → GPU SCC (not CPU-fed matrices)
// ==================================================================

fn run_full_chain_scc(label: &str, sk: &SkData, sp: &[String], xyz: &[[f64; 3]], min_n: usize) {
    let mut rt = require_nvidia();
    let scc = scc_cpu(sk, sp, xyz);
    let n = scc.h0.nrows();
    let n_atoms = sp.len();
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    assert!(n >= min_n, "{label} N={n} < {min_n} — this system is supposed to exercise the tiled N>64 path");
    eprintln!("{label} full-chain: n_atoms={n_atoms} N={n} n_occ={n_occ}");
    let (h_flat, s_flat, _) = gpu_assemble(sk, sp, xyz);
    let dh = max_abs_mat(&scc.h0, &extract_replica(&h_flat, 0, n));
    let ds = max_abs_mat(&scc.s, &extract_replica(&s_flat, 0, n));
    eprintln!("  assembled max|dH|={dh:.3e} max|dS|={ds:.3e}");
    assert!(dh < HS_TOL && ds < HS_TOL, "{label} GPU H/S feeding SCC already disagrees: dH={dh:.3e} dS={ds:.3e}");
    let tmpl = FragmentTemplate::new(sk, sp.to_vec(), xyz.to_vec()).unwrap();
    let h0 = rt.buffer_from_slice(&h_flat).unwrap();
    let s = rt.buffer_from_slice(&s_flat).unwrap();
    let g = rt.buffer_from_slice(&gamma_matrix(xyz, &per_atom_u(sk, sp))).unwrap();
    let q0f: Vec<f32> = scc.q0.iter().map(|&q| q as f32).collect();
    let q0 = rt.buffer_from_slice(&q0f).unwrap();
    let oa = rt.buffer_from_slice(&orb_atom_map(&tmpl.atom_orb_off, n)).unwrap();
    let mut plan = GpuSccPlan::new(&mut rt, &s, n, n_atoms, 1).unwrap();
    plan.set_initial_charges(&rt, &q0f).unwrap();
    let mut rms = f32::INFINITY;
    let mut n_iters = 0;
    for iter in 0..500 {
        n_iters = iter + 1;
        rms = plan.scc_step_diis(&mut rt, &h0, &s, &g, &q0, &oa, n_occ, 0.3)
            .unwrap_or_else(|e| panic!("{label} scc_step_diis failed at iter {n_iters}: {e}"));
        if n_iters <= 3 || n_iters % 50 == 0 {
            eprintln!("  gpu scc iter {n_iters} rms={rms:.3e}");
        }
        if rms < 1e-6 { break; }
    }
    assert!(rms < 1e-6, "{label} full-chain SCC did not converge: rms={rms:.3e} iters={n_iters} N={n}");
    let e_gpu = plan.compute_energy(&mut rt, &h0, &s, &g, &q0, &oa, n_occ).unwrap()[0] as f64;
    let q_gpu = plan.read_charges(&rt).unwrap();
    let dq = q_gpu.iter().zip(scc.charges.iter()).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
    let de = (e_gpu - scc.energy).abs();
    eprintln!("  E_cpu={:.8} E_gpu={:.8} |dE|={de:.3e} |dq|={dq:.3e} iters={n_iters}", scc.energy, e_gpu);
    assert!(de < E_EL_TOL, "{label} full-chain |dE|={de:.3e} > {E_EL_TOL:.1e} — GPU-assembled H/S feeding GPU SCC disagrees with CPU");
    assert!(dq < Q_TOL, "{label} full-chain |dq|={dq:.3e} > {Q_TOL:.1e}");
}

fn edm_from_c_eps_occ(c: &[f32], eps: &[f32], occ: &[i32], n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..n {
                if occ[k] == 0 { continue; }
                s += 2.0 * eps[k] * c[i * n + k] * c[j * n + k];
            }
            w[i * n + j] = s;
        }
    }
    w
}

#[test]
fn test_full_chain_gpu_assemble_then_scc_h2o() {
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    run_full_chain_scc("H2O", &sk, &sp, &xyz, 1);
}

#[test]
fn test_full_chain_gpu_assemble_then_scc_at() {
    let sk_dir = require_sk_dir();
    let (sp, xyz) = at_pair();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    run_full_chain_scc("AT", &sk, &sp, &xyz, 80);
}

#[test]
fn test_full_chain_gpu_assemble_then_scc_gc() {
    let sk_dir = require_sk_dir();
    let (sp, xyz) = gc_pair();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    run_full_chain_scc("GC", &sk, &sp, &xyz, 80);
}

/// After GPU assemble + GPU SCC + finalize, use GPU P and host-built
/// W = Σ_occ 2ε C C^T (no GPU W kernel exists). This is the production force path.
#[test]
fn test_full_chain_gpu_p_then_forces_h2o() {
    let mut rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let scc = scc_cpu(&sk, &sp, &xyz);
    let n = scc.h0.nrows();
    let n_atoms = sp.len();
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    let unique = unique_species(&sp);
    let (h_flat, s_flat, batch) = gpu_assemble(&sk, &sp, &xyz);
    let tmpl = FragmentTemplate::new(&sk, sp.clone(), xyz.clone()).unwrap();
    let h0 = rt.buffer_from_slice(&h_flat).unwrap();
    let s = rt.buffer_from_slice(&s_flat).unwrap();
    let g = rt.buffer_from_slice(&gamma_matrix(&xyz, &per_atom_u(&sk, &sp))).unwrap();
    let q0f: Vec<f32> = scc.q0.iter().map(|&q| q as f32).collect();
    let q0 = rt.buffer_from_slice(&q0f).unwrap();
    let oa = rt.buffer_from_slice(&orb_atom_map(&tmpl.atom_orb_off, n)).unwrap();
    let mut plan = GpuSccPlan::new(&mut rt, &s, n, n_atoms, 1).unwrap();
    plan.set_initial_charges(&rt, &q0f).unwrap();
    let mut rms = f32::INFINITY;
    for _ in 0..500 {
        rms = plan.scc_step_diis(&mut rt, &h0, &s, &g, &q0, &oa, n_occ, 0.3).unwrap();
        if rms < 1e-6 { break; }
    }
    assert!(rms < 1e-6, "H2O SCC for force chain did not converge: rms={rms:.3e}");
    let _e = plan.compute_energy(&mut rt, &h0, &s, &g, &q0, &oa, n_occ).unwrap();
    let mut p = vec![0.0f32; n * n];
    let mut c = vec![0.0f32; n * n];
    let mut eps = vec![0.0f32; n];
    let mut occ = vec![0i32; n];
    let mut v_shift = vec![0.0f32; n_atoms];
    let mut dq = vec![0.0f32; n_atoms];
    rt.read_buffer(&plan.d, &mut p).unwrap();
    rt.read_buffer(&plan.c, &mut c).unwrap();
    rt.read_buffer(&plan.eig_diag, &mut eps).unwrap();
    rt.read_buffer(&plan.occ_mask, &mut occ).unwrap();
    rt.read_buffer(&plan.v, &mut v_shift).unwrap();
    rt.read_buffer(&plan.dq, &mut dq).unwrap();
    let w = edm_from_c_eps_occ(&c, &eps, &occ, n);
    let n_occ_mask: i32 = occ.iter().sum();
    eprintln!("H2O GPU P/W force chain: N={n} n_occ={n_occ} occ_mask_sum={n_occ_mask}");
    assert_eq!(n_occ_mask as usize, n_occ, "occupation mask sum {n_occ_mask} != n_occ {n_occ}");
    let builder = HamiltonianBuilder::new(sk.clone());
    let cpu_f = compute_scc_forces(&builder, &sp, &xyz, &scc).unwrap();
    let max_dp = flatten_dm(&scc.density).iter().zip(p.iter()).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
    eprintln!("  max|P_gpu - P_cpu|={max_dp:.3e}");
    let driver = GpuForceDriver::new(&mut rt).unwrap();
    let (off, data, max_int) = pack_splines(&sk_dir, &unique);
    let u_hub: Vec<f32> = unique.iter().map(|s| sk.onsite(s).unwrap().u_hubbard as f32).collect();
    let crd_ang: Vec<f32> = xyz.iter().flat_map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect();
    let crd_bohr: Vec<f32> = xyz.iter().flat_map(|c| [(c[0] * ANG2BOHR) as f32, (c[1] * ANG2BOHR) as f32, (c[2] * ANG2BOHR) as f32]).collect();
    let spc = species_idx(&sp, &unique);
    let f0 = driver.gpu_force_batched(&rt, &batch, &p, &w).unwrap();
    let f1 = driver.gpu_scc_shift_force_batched(&rt, &batch, &p, &v_shift).unwrap();
    let f2 = driver.gpu_gamma_deriv_force_batched(&rt, n_atoms, 1, &crd_ang, &spc, &dq, &u_hub, unique.len()).unwrap();
    let f3 = driver.gpu_repulsive_force_batched(&mut rt, n_atoms, 1, &crd_bohr, &spc, &off, &data, unique.len(), max_int).unwrap();
    let mut f_tot = vec![0.0f32; 3 * n_atoms];
    for i in 0..3 * n_atoms { f_tot[i] = f0[i] + f1[i] + f2[i] + f3[i]; }
    let (_, _, rel) = force_err(&cpu_f.forces, &f_tot, "H2O/full-chain TOTAL");
    let nsum = newton_sum(&f_tot, n_atoms);
    eprintln!("  Newton |ΣF|={nsum:.3e}");
    assert!(rel < FORCE_REL, "full-chain GPU P/W forces vs CPU rel={rel:.3e} > {FORCE_REL:.1e} (max|dP|={max_dp:.3e}). \
        If dP is large the SCC state is wrong; if dP is small the force kernels disagree with this P/W.");
    assert!(nsum < NEWTON_TOL, "full-chain Newton |ΣF|={nsum:.3e}");
}

#[test]
fn test_scc_electronic_parity_tight_h2o_cpu_fed() {
    let mut rt = require_nvidia();
    let sk_dir = require_sk_dir();
    let (sp, xyz) = h2o();
    let sk = load_sk_for_species(&sk_dir, &sp).unwrap();
    let scc = scc_cpu(&sk, &sp, &xyz);
    let n_occ = (scc.q0.iter().sum::<f64>() / 2.0).round() as usize;
    let (mut plan, h0, s, g, q0, oa) = plan_scc(&mut rt, &sk, &sp, &xyz, &scc);
    let e = plan.compute_energy(&mut rt, &h0, &s, &g, &q0, &oa, n_occ).unwrap()[0] as f64;
    let q = plan.read_charges(&rt).unwrap();
    let de = (e - scc.energy).abs();
    let dq = q.iter().zip(scc.charges.iter()).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
    eprintln!("CPU-fed H2O SCC: E_cpu={:.8} E_gpu={:.8} |dE|={de:.3e} |dq|={dq:.3e}", scc.energy, e);
    assert!(de < E_EL_TOL, "CPU-fed SCC |dE|={de:.3e} > {E_EL_TOL:.1e} (old tests allowed 1e-3)");
    assert!(dq < Q_TOL, "CPU-fed SCC |dq|={dq:.3e} > {Q_TOL:.1e}");
}
