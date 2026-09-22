//! Real-molecule warm-start benchmark — the production use case measured
//! on actual DFTB SCC trajectories, not synthetic matrices.
//!
//! Scenario A (`test_warm_bench_scc`): capture the per-iteration
//! (H′_i, Y_i) trace of a real CPU SCC solve (RUST_DFTB_SCC_TRACE), then
//! for each iteration i≥2 solve H′_i by five methods on identical inputs:
//!   cold TC2 purification · GPU Jacobi · warm Jacobi (Y_{i-1}ᵀ H′ Y_{i-1}
//!   = the incumbent `b_warm` path) · warm extrapolation S=K_{i-1}+γΔK
//!   (γ=1) and the γ=0 ablation (transport-only).
//! Accuracy certified vs the exact dsyevd projector K*_i.
//!
//! Scenario B (`test_warm_bench_geom`): converge at G0, displace atoms by
//! δ∈{0.01,0.05,0.10} Å, solve the first SCC iteration (H0′) at the new
//! geometry from the transported old density K = L_newᵀ(D_old/2)L_new,
//! and (with a 2-geometry history) from AO-space extrapolation
//! D_p = D_1+γ(D_1−D_0).
//!
//! Run:
//!   RUST_DFTB_SK_DIR=/path/to/mio-1-1 \
//!   cargo test --release --test gpu_warm_bench -- --ignored --nocapture

use nalgebra::DMatrix;
use rust_dftb::io::parse_xyz;
use rust_dftb::load_sk_for_species;
use rust_dftb::methods::dftb::dftb_cpu::DftbCpu;
use rust_dftb::methods::dftb::rotation::{DirectionCosines, Rotation};
use rust_dftb::qmqm::gpu_eigen::jacobi_batched;
use rust_dftb::qmqm::gpu_purify::{purify_tc2_batched, warm_extrap_solve};
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;
use std::time::Instant;

fn xyz_file(name: &str) -> rust_dftb::io::XyzMolecule {
    for p in [
        format!("data/xyz/{name}"),
        format!("{}/data/xyz/{name}", env!("CARGO_MANIFEST_DIR")),
        format!("{}/../data/xyz/{name}", env!("CARGO_MANIFEST_DIR")),
    ] {
        if let Ok(x) = parse_xyz(&p) {
            return x;
        }
    }
    panic!("cannot load {name}");
}

/// Exact occupied projector K = Y[:, :nocc]·Y[:, :nocc]ᵀ (kT=0) in f32.
fn projector(y: &DMatrix<f64>, nocc: usize) -> Vec<f32> {
    let n = y.nrows();
    let yo = y.columns(0, nocc);
    let k = &yo * yo.transpose();
    (0..n * n).map(|e| k[(e / n, e % n)] as f32).collect()
}

/// Certify a GPU-produced projector against the exact dsyevd answer.
/// Returns (ΔE, ‖ΔK‖_F, ‖K²−K‖_F, ‖HK−KH‖_F). All in f64.
fn certify(h: &[f32], k: &[f32], k_ref: &[f64], e_ref: f64, n: usize) -> (f64, f64, f64, f64) {
    let mut tr_kh = 0.0f64;
    let mut dp = 0.0f64;
    let mut idem = 0.0f64;
    let mut comm = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            tr_kh += k[i * n + j] as f64 * h[j * n + i] as f64;
            dp += (k[i * n + j] as f64 - k_ref[i * n + j]).powi(2);
            let mut kk = 0.0f64;
            let mut hk = 0.0f64;
            let mut kh = 0.0f64;
            for l in 0..n {
                kk += k[i * n + l] as f64 * k[l * n + j] as f64;
                hk += h[i * n + l] as f64 * k[l * n + j] as f64;
                kh += k[i * n + l] as f64 * h[l * n + j] as f64;
            }
            idem += (kk - k[i * n + j] as f64).powi(2);
            comm += (hk - kh).powi(2);
        }
    }
    ((tr_kh - e_ref).abs(), dp.sqrt(), idem.sqrt(), comm.sqrt())
}

/// ‖Q·ΔH·P‖/gap — the effective warm-start perturbation (η_warm).
/// P = K_{i-1} projector, ΔH = H'_i − H'_{i-1}, gap = ε_LUMO − ε_HOMO.
fn eta_warm(h_new: &DMatrix<f64>, h_old: &DMatrix<f64>, p: &DMatrix<f64>, gap: f64) -> f64 {
    let n = h_new.nrows();
    let q = DMatrix::<f64>::identity(n, n) - p;
    let dh = h_new - h_old;
    let m = &q * &dh * p;
    m.norm() / gap
}

struct Row {
    it: usize,
    dk: f64,      // ‖K_i − K_{i-1}‖ — trajectory step size
    eta: f64,     // ‖QΔHP‖/gap
    // products-equivalents + wall ms per method
    tc2_p: usize, tc2_ms: f64,
    xtr_p: usize, xtr_ms: f64,
    abl_p: usize, abl_ms: f64,
    jac_ms: f64, jacw_ms: f64,
    // accuracy vs exact K*_i (ΔE Ha, ‖ΔK‖, comm)
    tc2_acc: (f64, f64, f64, f64),
    xtr_acc: (f64, f64, f64, f64),
    abl_acc: (f64, f64, f64, f64),
    g05_acc: (f64, f64, f64, f64),
    jac_acc: (f64, f64, f64, f64),
    jacw_acc: (f64, f64, f64, f64),
}

#[test]
#[ignore]
fn test_warm_bench_scc() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Skipping: no GPU ({e})");
            return;
        }
    };
    std::env::set_var("RUST_DFTB_SCC_TRACE", "1");

    for name in ["H2O.xyz", "formic_dimer.xyz", "guanine-cytosine.xyz"] {
        let xyz = xyz_file(name);
        let sk = load_sk_for_species(&sk_dir, &xyz.species).unwrap();
        let mut cpu = DftbCpu::new(sk.clone(), xyz.species.clone()).unwrap();
        cpu.update_geometry(&xyz.coords).unwrap();
        cpu.set_smearing(0.0);
        cpu.reset_charges();
        cpu.solve_scc(100, 1e-9).unwrap();
        let n = cpu.n_orbs;
        let nocc = cpu.n_occ;
        let trace = &cpu.scc_trace;
        let nit = trace.len();
        eprintln!("\n=== {name}: n={n} nocc={nocc} scc_iters={nit} ===");
        if nit < 4 {
            eprintln!("  too few SCC iterations — skipping");
            continue;
        }
        let gap = |i: usize| trace[i].eigvals[nocc] - trace[i].eigvals[nocc - 1];

        // Replicas = iterations 2..nit (each needs 2 history projectors).
        let its: Vec<usize> = (2..nit).collect();
        let batch = its.len();
        let mut h_b = Vec::with_capacity(batch * n * n);
        let mut p1_b = Vec::with_capacity(batch * n * n);
        let mut p2_b = Vec::with_capacity(batch * n * n);
        for &i in &its {
            for v in trace[i].h_prime.iter() {
                h_b.push(*v as f32);
            }
            p1_b.extend_from_slice(&projector(&trace[i - 1].eigvecs, nocc));
            p2_b.extend_from_slice(&projector(&trace[i - 2].eigvecs, nocc));
        }
        let h_buf = rt.buffer_from_slice(&h_b).unwrap();
        let p1_buf = rt.buffer_from_slice(&p1_b).unwrap();
        let p2_buf = rt.buffer_from_slice(&p2_b).unwrap();
        let nocc_v = vec![nocc as f32; batch];

        // --- warmup: build all GPU programs once (one-time compile
        // cost excluded from the timed runs below) ---
        {
            let mut d0 = vec![0.0f32; batch * n * n];
            purify_tc2_batched(&mut rt, &h_buf, &mut d0, n, batch, &nocc_v, 5, 1e-5).unwrap();
            let d_w = rt.zero_buffer::<f32>(batch * n * n).unwrap();
            warm_extrap_solve(&mut rt, &h_buf, &p1_buf, &p2_buf, &d_w, n, batch, 1.0, 1, 1e-3)
                .unwrap();
            let a0 = rt.buffer_from_slice(&h_b).unwrap();
            let v0 = rt.zero_buffer::<f32>(batch * n * n).unwrap();
            jacobi_batched(&mut rt, &a0, &v0, n, batch).unwrap();
        }

        // --- cold TC2 ---
        let t0 = Instant::now();
        let mut d_cold = vec![0.0f32; batch * n * n];
        let pdiag =
            purify_tc2_batched(&mut rt, &h_buf, &mut d_cold, n, batch, &nocc_v, 200, 1e-5).unwrap();
        let tc2_ms = t0.elapsed().as_secs_f64() * 1e3;

        // --- warm extrapolation γ=1 ---
        let d_xtr = rt.zero_buffer::<f32>(batch * n * n).unwrap();
        let t0 = Instant::now();
        let wx = warm_extrap_solve(&mut rt, &h_buf, &p1_buf, &p2_buf, &d_xtr, n, batch, 1.0, 4, 1e-3)
            .unwrap();
        let xtr_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut d_xtr_h = vec![0.0f32; batch * n * n];
        rt.read_buffer(&d_xtr, &mut d_xtr_h).unwrap();

        // --- ablation γ=0 (transport/retract only, no extrapolation) ---
        let d_abl = rt.zero_buffer::<f32>(batch * n * n).unwrap();
        let t0 = Instant::now();
        let wa = warm_extrap_solve(&mut rt, &h_buf, &p1_buf, &p2_buf, &d_abl, n, batch, 0.0, 4, 1e-3)
            .unwrap();
        let abl_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut d_abl_h = vec![0.0f32; batch * n * n];
        rt.read_buffer(&d_abl, &mut d_abl_h).unwrap();

        // --- γ=0.5: SCC trajectories decelerate (steps shrink toward
        // the fixed point), so γ=1 overshoots — half-step probe ---
        let d_g05 = rt.zero_buffer::<f32>(batch * n * n).unwrap();
        let t0 = Instant::now();
        let wg5 = warm_extrap_solve(&mut rt, &h_buf, &p1_buf, &p2_buf, &d_g05, n, batch, 0.5, 4, 1e-3)
            .unwrap();
        let g05_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut d_g05_h = vec![0.0f32; batch * n * n];
        rt.read_buffer(&d_g05, &mut d_g05_h).unwrap();

        // --- cold Jacobi (production eigensolver; one launch, internal sweeps) ---
        let a_buf = rt.buffer_from_slice(&h_b).unwrap();
        let v_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
        let t0 = Instant::now();
        jacobi_batched(&mut rt, &a_buf, &v_buf, n, batch).unwrap();
        let mut v_cold = vec![0.0f32; batch * n * n];
        rt.read_buffer(&v_buf, &mut v_cold).unwrap();
        let jac_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut a_jac = vec![0.0f32; batch * n * n];
        rt.read_buffer(&a_buf, &mut a_jac).unwrap();

        // --- warm Jacobi (incumbent b_warm path): W = Y_{i-1}ᵀ H' Y_{i-1} ---
        let mut w_b = Vec::with_capacity(batch * n * n);
        for (r, &i) in its.iter().enumerate() {
            let y = &trace[i - 1].eigvecs;
            let w = y.transpose() * &trace[i].h_prime * y;
            for e in 0..n * n {
                w_b.push(w[(e / n, e % n)] as f32);
            }
            let _ = r;
        }
        let w_buf = rt.buffer_from_slice(&w_b).unwrap();
        let u_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
        let t0 = Instant::now();
        jacobi_batched(&mut rt, &w_buf, &u_buf, n, batch).unwrap();
        let mut u_w = vec![0.0f32; batch * n * n];
        rt.read_buffer(&u_buf, &mut u_w).unwrap();
        let jacw_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut w_jac = vec![0.0f32; batch * n * n];
        rt.read_buffer(&w_buf, &mut w_jac).unwrap();

        // --- per-replica report ---
        let mut rows = Vec::new();
        for (r, &i) in its.iter().enumerate() {
            let hb = &h_b[r * n * n..(r + 1) * n * n];
            let k_ref = projector(&trace[i].eigvecs, nocc);
            let k_ref64: Vec<f64> = k_ref.iter().map(|&x| x as f64).collect();
            let e_ref: f64 = trace[i].eigvals.iter().take(nocc).sum();
            let g = gap(i);
            // ‖K_i − K_{i-1}‖ in f64
            let p_prev = projector(&trace[i - 1].eigvecs, nocc);
            let dk: f64 = k_ref
                .iter()
                .zip(p_prev.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt() as f64;
            let p_m = DMatrix::from_fn(n, n, |a, b| p_prev[a * n + b] as f64);
            let eta = eta_warm(&trace[i].h_prime, &trace[i - 1].h_prime, &p_m, g);

            let tc2_acc = certify(hb, &d_cold[r * n * n..(r + 1) * n * n], &k_ref64, e_ref, n);
            let xtr_acc = certify(hb, &d_xtr_h[r * n * n..(r + 1) * n * n], &k_ref64, e_ref, n);
            let abl_acc = certify(hb, &d_abl_h[r * n * n..(r + 1) * n * n], &k_ref64, e_ref, n);
            let g05_acc = certify(hb, &d_g05_h[r * n * n..(r + 1) * n * n], &k_ref64, e_ref, n);
            // Jacobi cold: K from eigenvectors — V columns, occ = nocc
            // smallest eigenvalues (diag of rotated A).
            let jac_k = jac_projector(&a_jac[r * n * n..], &v_cold[r * n * n..], n, nocc);
            let jac_acc = certify(hb, &jac_k, &k_ref64, e_ref, n);
            // warm Jacobi: K = Y·(U_occ U_occᵀ)·Yᵀ
            let jacw_k = jacw_projector(
                &w_jac[r * n * n..],
                &u_w[r * n * n..],
                &trace[i - 1].eigvecs,
                n,
                nocc,
            );
            let jacw_acc = certify(hb, &jacw_k, &k_ref64, e_ref, n);
            rows.push(Row {
                it: i, dk, eta,
                tc2_p: pdiag.iters, tc2_ms, xtr_p: wx.products, xtr_ms,
                abl_p: wa.products, abl_ms, jac_ms, jacw_ms,
                tc2_acc, xtr_acc, abl_acc, g05_acc, jac_acc, jacw_acc,
            });
        }
        eprintln!(
            "{:>3} {:>7} {:>7} | {:>4}p {:>9} {:>9} | {:>4}p {:>9} {:>9} | γ0 {:>9} γ.5 {:>9} | jacW {:>9} jac {:>9}",
            "i", "‖ΔK‖", "η_w",
            "tc2", "ΔE", "‖ΔK‖", "xtr", "ΔE", "‖ΔK‖", "ΔE", "ΔE", "ΔE", "ΔE"
        );
        for r in &rows {
            eprintln!(
                "{:>3} {:>7.3} {:>7.3} | {:>4} {:>9.2e} {:>9.2e} | {:>4} {:>9.2e} {:>9.2e} | {:>9.2e} {:>9.2e} | {:>9.2e} {:>9.2e}",
                r.it, r.dk, r.eta,
                r.tc2_p, r.tc2_acc.0, r.tc2_acc.1,
                r.xtr_p, r.xtr_acc.0, r.xtr_acc.1,
                r.abl_acc.0, r.g05_acc.0,
                r.jacw_acc.0, r.jac_acc.0,
            );
        }
        eprintln!(
            "  times: TC2={tc2_ms:.2}ms({}prod) xtr={xtr_ms:.2}ms({}prod corr={}) abl={abl_ms:.2}ms({}prod) γ.5={g05_ms:.2}ms({}prod) jac={jac_ms:.2}ms jacW={jacw_ms:.2}ms conv={}/{}",
            pdiag.iters, wx.products, wx.corrs, wa.products, wg5.products,
            rows.iter().filter(|r| r.xtr_acc.0 < 1e-3).count(), batch
        );
        eprintln!("  xtr converged={} comms_max={:.3e}", wx.converged,
            wx.comms.iter().cloned().fold(0.0f32, f32::max));
    }
}

// ==================================================================
// Scenario B — geometry step. Converge at G0 (and an intermediate G_h
// for a 2-point history), displace by δ, solve the FIRST SCC iteration
// H0′(new) from the transported old density:
//     K_guess = L_newᵀ · (D_ao/2) · L_new        (exact basis transport)
//     extrap:   D_p = D_1 + γ(D_1 − D_0) in AO space, then transport.
// The transported seed is certified against the exact projector of
// H0′(new) — the certificate decides accept/fallback, nothing silent.
// ==================================================================

/// Transport an AO density into the current orthogonal basis:
/// K = Lᵀ·(D/2)·L  (D_ao is basis-free; L from `cpu.cholesky_l`).
/// NOTE: this is only "same AO coefficients" — the basis functions
/// themselves moved with the atoms, so the physical orbitals are NOT
/// preserved. The correct subspace transport needs S_cross (below).
fn transport(l: &DMatrix<f64>, d_ao: &DMatrix<f64>) -> Vec<f32> {
    let k = l.transpose() * (d_ao * 0.5) * l;
    let n = k.nrows();
    (0..n * n).map(|e| k[(e / n, e % n)] as f32).collect()
}

/// Cross-geometry overlap S_cross[ν,μ] = ⟨χ_ν(new) | χ_μ(old)⟩ — same
/// SK rotation machinery as `update_geometry`, but pair vectors run
/// from OLD atom positions to NEW atom positions. Same-atom blocks at
/// zero displacement are the identity; at δ>0 the homonuclear SK table
/// evaluated at r=δ gives the orbital-following overlap.
fn s_cross(newc: &DftbCpu, oldc: &DftbCpu) -> DMatrix<f64> {
    const ANG2BOHR: f64 = 1.889_726_133;
    let ctx = &newc.ctx;
    let n = newc.n_orbs;
    let cutoff = newc.neigh.cutoff;
    let mut sc = DMatrix::<f64>::zeros(n, n);
    for a in 0..newc.n_atoms {
        for b in 0..oldc.n_atoms {
            let v = [
                (newc.coords[a][0] - oldc.coords[b][0]) * ANG2BOHR,
                (newc.coords[a][1] - oldc.coords[b][1]) * ANG2BOHR,
                (newc.coords[a][2] - oldc.coords[b][2]) * ANG2BOHR,
            ];
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if r > cutoff {
                continue;
            }
            // pair convention: "i" = old atom b, "j" = new atom a;
            // out_s[α*n_i+β] fills sc[new_orb(α@a), old_orb(β@b)]
            let si = ctx.atom_species[b] as usize;
            let sj = ctx.atom_species[a] as usize;
            let ni = ctx.atom_n_orb[b] as usize;
            let nj = ctx.atom_n_orb[a] as usize;
            let bi = ctx.atom_orb_off[b] as usize;
            let bj = ctx.atom_orb_off[a] as usize;
            if a == b && r < 1e-12 {
                for k in 0..ni {
                    sc[(bj + k, bi + k)] = 1.0;
                }
                continue;
            }
            let tab_fwd = ctx.pair_lut[si * ctx.n_species + sj]
                .map(|x| &ctx.pair_tables[x])
                .expect("missing SK table fwd");
            let tab_rev = ctx.pair_lut[sj * ctx.n_species + si]
                .map(|x| &ctx.pair_tables[x])
                .expect("missing SK table rev");
            let mut hh = vec![0.0f64; nj * ni];
            let mut ss = vec![0.0f64; nj * ni];
            let dc = DirectionCosines::from_vec(v).expect("dc");
            Rotation::rotate_diatomic_block_into(
                tab_fwd,
                tab_rev,
                &ctx.species_ang[si],
                &ctx.species_ang[sj],
                r,
                dc,
                &mut hh,
                &mut ss,
            )
            .expect("sk rotate");
            for al in 0..nj {
                for be in 0..ni {
                    sc[(bj + al, bi + be)] = ss[al * ni + be];
                }
            }
        }
    }
    sc
}

/// Proper subspace transport across a geometry change: project the old
/// occupied orbitals onto the NEW basis via S_cross, re-orthonormalize,
/// return the projector in the new orthogonal basis.
///   c_ao_old = old AO coeffs of occ MOs (= cpu.eigenvectors[:, :nocc];
///              NOTE: `cpu.eigenvectors` is ALREADY AO-basis C = L⁻ᵀY,
///              unlike `trace.eigvecs` which is orthonormal-basis Y)
///   d        = S_new⁻¹·S_cross·C_ao_old  (best fit in new basis)
///   C'       = L_newᵀ·d                  (orthonormal-basis coeffs)
///   K        = C'·(C'ᵀC')⁻¹·C'ᵀ          (projector onto transported span)
fn transport_subspace(
    l_new: &DMatrix<f64>,
    s_new: &DMatrix<f64>,
    s_cross: &DMatrix<f64>,
    c_ao_old: &DMatrix<f64>,
) -> Vec<f32> {
    let m = s_cross * c_ao_old;
    let d = s_new.clone().lu().solve(&m).expect("S solve");
    let cp = l_new.transpose() * d;
    let g = cp.transpose() * &cp;
    let k = &cp * g.try_inverse().expect("gram inv") * cp.transpose();
    let n = k.nrows();
    (0..n * n).map(|e| k[(e / n, e % n)] as f32).collect()
}

#[test]
#[ignore]
fn test_warm_bench_geom() {
    let Ok(sk_dir) = std::env::var("RUST_DFTB_SK_DIR") else {
        eprintln!("Skipping: RUST_DFTB_SK_DIR not set");
        return;
    };
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Skipping: no GPU ({e})");
            return;
        }
    };
    std::env::set_var("RUST_DFTB_SCC_TRACE", "1");

    // Displacement magnitudes (Å) — realistic optimizer / MD steps.
    let deltas = [0.01f64, 0.05, 0.10];
    for name in ["H2O.xyz", "formic_dimer.xyz"] {
        let xyz = xyz_file(name);
        let sk = load_sk_for_species(&sk_dir, &xyz.species).unwrap();
        let nat = xyz.species.len();
        eprintln!("\n=== {name} geometry-step benchmark (nat={nat}) ===");

        for &delta in &deltas {
            // Trajectory G0 → G0+δ/2 → G0+δ (uniform random direction per
            // system, deterministic seed from δ).
            let seed = (delta * 1000.0) as u64;
            let mut rng = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut dir = vec![[0.0f64; 3]; nat];
            let mut nrm = 0.0f64;
            for a in &mut dir {
                for c in 0..3 {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    a[c] = (rng >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
                    nrm += a[c] * a[c];
                }
            }
            nrm = nrm.sqrt();
            let disp = |frac: f64| -> Vec<[f64; 3]> {
                xyz.coords
                    .iter()
                    .zip(dir.iter())
                    .map(|(p, d)| {
                        [
                            p[0] + frac * delta * d[0] / nrm,
                            p[1] + frac * delta * d[1] / nrm,
                            p[2] + frac * delta * d[2] / nrm,
                        ]
                    })
                    .collect()
            };

            // Converge at G0, G0+δ/2, G0+δ — keep the engines alive
            // (density, cholesky_l, scc_trace are needed below).
            // The target geometry restarts SCC from the PREVIOUS
            // converged charges — the honest production warm start
            // (trace[0].h_prime = H'(new | q_old), not H'(new | q0)).
            let mut cpus = Vec::new();
            let mut q_prev: Option<Vec<f64>> = None;
            for frac in [0.0f64, 0.5, 1.0] {
                let mut cpu = DftbCpu::new(sk.clone(), xyz.species.clone()).unwrap();
                cpu.update_geometry(&disp(frac)).unwrap();
                cpu.set_smearing(0.0);
                cpu.reset_charges();
                if let Some(q) = &q_prev {
                    cpu.set_charges(q);
                }
                cpu.solve_scc(100, 1e-9).unwrap();
                q_prev = Some(cpu.charges.clone());
                cpus.push(cpu);
            }
            let ds: Vec<DMatrix<f64>> = cpus.iter_mut().map(|c| c.build_result().density.clone()).collect();
            let tr = &cpus[2].scc_trace;
            let n = cpus[2].n_orbs;
            let nocc = cpus[2].n_occ;
            // First-iteration target: H0′(new) = trace[0].h_prime; exact
            // projector K* from trace[0].eigvecs.
            let h_tgt: Vec<f32> = tr[0].h_prime.iter().map(|&x| x as f32).collect();
            let k_ref = projector(&tr[0].eigvecs, nocc);
            let k_ref64: Vec<f64> = k_ref.iter().map(|&x| x as f64).collect();
            let e_ref: f64 = tr[0].eigvals.iter().take(nocc).sum();

            // --- candidate seeds (host-side transport) ---
            let l_new = &cpus[2].cholesky_l;
            let k_t = transport(l_new, &ds[1]);                     // transported D(δ/2) — closest converged
            let k_t0 = transport(l_new, &ds[0]);                    // transported D(0)
            let d_pred = &ds[1] * 2.0 - &ds[0];                     // AO extrapolation γ=1
            let k_x = transport(l_new, &d_pred);

            // --- DIAGNOSTIC: raw seed quality (before any correction)
            // and the proper subspace transport via S_cross ---
            for (tag, seed_k) in [("transp(D_½)", &k_t), ("transp(D_0)", &k_t0), ("xtrAO(D)", &k_x)] {
                let (de, dp, di, dc) = certify(&h_tgt, seed_k, &k_ref64, e_ref, n);
                eprintln!(
                    "  δ={delta:.2} seed {tag:>12}: ΔE={de:.2e} ‖ΔK‖={dp:.2e} idem={di:.2e} comm={dc:.2e}  (raw, no corr)"
                );
            }
            // S_cross transport: project old occupied orbitals onto the
            // new basis, re-orthonormalize. From δ/2 and from G0.
            // SANITY: s_cross(G,G) must equal S; self-transport must
            // reproduce the old projector.
            let sc22 = s_cross(&cpus[2], &cpus[2]);
            let mut sd = 0.0f64;
            for e in 0..n * n {
                sd = sd.max((sc22[(e / n, e % n)] - cpus[2].s[(e / n, e % n)]).abs());
            }
            eprintln!("  δ={delta:.2} SANITY max|s_cross(G,G)−S| = {sd:.3e}");
            let y2 = cpus[2].eigenvectors.columns(0, nocc).clone_owned();
            let k_self = transport_subspace(l_new, &cpus[2].s, &sc22, &y2);
            // K_conv must be built from the ORTHOGONAL-basis eigvecs
            // (trace.eigvecs); cpu.eigenvectors is AO-basis C.
            let k_conv = projector(&cpus[2].scc_trace.last().unwrap().eigvecs, nocc);
            let mut dd = 0.0f64;
            for e in 0..n * n {
                dd += (k_self[e] as f64 - k_conv[e] as f64).powi(2);
            }
            eprintln!("  δ={delta:.2} SANITY ‖K_self−K_conv‖ = {:.3e}", dd.sqrt());
            let y1 = cpus[1].eigenvectors.columns(0, nocc).clone_owned();
            let y0 = cpus[0].eigenvectors.columns(0, nocc).clone_owned();
            let sc21 = s_cross(&cpus[2], &cpus[1]);
            let sc20 = s_cross(&cpus[2], &cpus[0]);
            let k_s1 = transport_subspace(l_new, &cpus[2].s, &sc21, &y1);
            let k_s0 = transport_subspace(l_new, &cpus[2].s, &sc20, &y0);
            for (tag, seed_k) in [("Scross(½)", &k_s1), ("Scross(0)", &k_s0)] {
                let (de, dp, di, dc) = certify(&h_tgt, seed_k, &k_ref64, e_ref, n);
                eprintln!(
                    "  δ={delta:.2} seed {tag:>12}: ΔE={de:.2e} ‖ΔK‖={dp:.2e} idem={di:.2e} comm={dc:.2e}  (raw, no corr)"
                );
            }

            let h_buf = rt.buffer_from_slice(&h_tgt).unwrap();
            let nocc_v = vec![nocc as f32];

            // transport-only (γ=0 on transported seed)
            for (tag, seed_k) in [
                ("transp(D_½)", &k_t),
                ("transp(D_0)", &k_t0),
                ("xtrAO(D)", &k_x),
                ("Scross(½)", &k_s1),
                ("Scross(0)", &k_s0),
            ] {
                let p_buf = rt.buffer_from_slice(seed_k).unwrap();
                let d_buf = rt.zero_buffer::<f32>(n * n).unwrap();
                let t0 = Instant::now();
                let w = warm_extrap_solve(&mut rt, &h_buf, &p_buf, &p_buf, &d_buf, n, 1, 0.0, 4, 1e-3)
                    .unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut d_out = vec![0.0f32; n * n];
                rt.read_buffer(&d_buf, &mut d_out).unwrap();
                let (de, dp, di, dc) = certify(&h_tgt, &d_out, &k_ref64, e_ref, n);
                eprintln!(
                    "  δ={delta:.2} {tag:>12}: prod={} corr={} {ms:6.2}ms | ΔE={de:.2e} ‖ΔK‖={dp:.2e} idem={di:.2e} comm={dc:.2e} conv={}",
                    w.products, w.corrs, w.converged
                );
            }

            // cold TC2 + Jacobi baselines on the same H0′(new)
            let t0 = Instant::now();
            let mut d_cold = vec![0.0f32; n * n];
            let pd = purify_tc2_batched(&mut rt, &h_buf, &mut d_cold, n, 1, &nocc_v, 200, 1e-5).unwrap();
            let tc2_ms = t0.elapsed().as_secs_f64() * 1e3;
            let (de, dp, di, dc) = certify(&h_tgt, &d_cold, &k_ref64, e_ref, n);
            eprintln!(
                "  δ={delta:.2}   cold TC2: iters={} {tc2_ms:6.2}ms | ΔE={de:.2e} ‖ΔK‖={dp:.2e} idem={di:.2e} comm={dc:.2e}",
                pd.iters
            );
            let a_buf = rt.buffer_from_slice(&h_tgt).unwrap();
            let v_buf = rt.zero_buffer::<f32>(n * n).unwrap();
            let t0 = Instant::now();
            jacobi_batched(&mut rt, &a_buf, &v_buf, n, 1).unwrap();
            let mut v_h = vec![0.0f32; n * n];
            rt.read_buffer(&v_buf, &mut v_h).unwrap();
            let jac_ms = t0.elapsed().as_secs_f64() * 1e3;
            let mut a_h = vec![0.0f32; n * n];
            rt.read_buffer(&a_buf, &mut a_h).unwrap();
            let jk = jac_projector(&a_h, &v_h, n, nocc);
            let (de, dp, di, dc) = certify(&h_tgt, &jk, &k_ref64, e_ref, n);
            eprintln!(
                "  δ={delta:.2} cold Jacobi: {jac_ms:6.2}ms | ΔE={de:.2e} ‖ΔK‖={dp:.2e} idem={di:.2e} comm={dc:.2e}"
            );
        }
    }
}

/// Projector from Jacobi output: eigenvectors are columns of V; occupied
/// = nocc columns with smallest eigenvalues (diag of rotated A).
fn jac_projector(a_rot: &[f32], v: &[f32], n: usize, nocc: usize) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a_rot[i * n + i].partial_cmp(&a_rot[j * n + j]).unwrap());
    let mut k = vec![0.0f32; n * n];
    for &c in idx.iter().take(nocc) {
        for i in 0..n {
            for j in 0..n {
                k[i * n + j] += v[i * n + c] * v[j * n + c];
            }
        }
    }
    k
}

/// Warm-Jacobi projector: K = Y·(U_occ U_occᵀ)·Yᵀ where U = eigenvectors
/// of W = YᵀHY (Y = previous iteration's eigenvectors, f64).
fn jacw_projector(w_rot: &[f32], u: &[f32], y: &DMatrix<f64>, n: usize, nocc: usize) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| w_rot[i * n + i].partial_cmp(&w_rot[j * n + j]).unwrap());
    // K_w = U_occ U_occᵀ in the Y frame (f64), then K = Y K_w Yᵀ.
    let mut kw = DMatrix::<f64>::zeros(n, n);
    for &c in idx.iter().take(nocc) {
        for i in 0..n {
            for j in 0..n {
                kw[(i, j)] += (u[i * n + c] as f64) * (u[j * n + c] as f64);
            }
        }
    }
    let k = y * &kw * y.transpose();
    (0..n * n).map(|e| k[(e / n, e % n)] as f32).collect()
}
