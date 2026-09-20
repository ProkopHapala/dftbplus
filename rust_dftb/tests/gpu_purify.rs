//! Dense batched TC2 purification — parity + benchmark.
//!
//! Verifies the FOE path (`gpu_purify`) against CPU eigendecomposition:
//! the purified density D must be idempotent, symmetric, trace=Nocc,
//! and — the strong check — Tr(D·H) must equal the sum of the nocc
//! lowest eigenvalues (the variational minimum), with ‖D−D_ref‖ small
//! where D_ref = V_occ·V_occᵀ.
//!
//! Run: cargo test --release --test gpu_purify -- --nocapture
//!      cargo test --release --test gpu_purify purify_bench -- --ignored --nocapture

use nalgebra::DMatrix;
use rust_dftb::qmqm::gpu_purify::{purify_tc2_batched, relax_purify_batched};
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;

fn random_symmetric(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32;
            a[i * n + j] = val;
            a[j * n + i] = val;
        }
    }
    a
}

fn cpu_eig(a: &[f32], n: usize) -> (Vec<f32>, DMatrix<f64>) {
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = a[i * n + j] as f64;
        }
    }
    let sym = m.symmetric_eigen();
    // nalgebra's eigenvalue order is not guaranteed ascending here — sort
    // explicitly and permute eigenvector columns to match (same convention
    // as gate_e_determinism / gate_g3_energy).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| sym.eigenvalues[i].partial_cmp(&sym.eigenvalues[j]).unwrap());
    let eigs: Vec<f32> = idx.iter().map(|&i| sym.eigenvalues[i] as f32).collect();
    let mut vecs = DMatrix::<f64>::zeros(n, n);
    for (c, &i) in idx.iter().enumerate() {
        for r in 0..n {
            vecs[(r, c)] = sym.eigenvectors[(r, i)];
        }
    }
    (eigs, vecs)
}

#[test]
fn test_purify_tc2_parity() {
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[purify] no GPU: {e}");
            return;
        }
    };
    let n = 86usize;
    let batch = 16usize;
    let nocc = 43f32;

    let mut h = Vec::with_capacity(batch * n * n);
    for b in 0..batch {
        h.extend_from_slice(&random_symmetric(n, 1000 + b as u64));
    }
    let h_buf = rt.buffer_from_slice(&h).unwrap();
    let mut d = vec![0.0f32; batch * n * n];
    let nocc_v = vec![nocc; batch];
    let diag = purify_tc2_batched(&mut rt, &h_buf, &mut d, n, batch, &nocc_v, 200, 1e-5)
        .unwrap();

    eprintln!(
        "[purify] n={n} batch={batch}: iters={} converged={} max_err={:.3e} max|Tr-Nocc|={:.3e}",
        diag.iters,
        diag.converged,
        diag.errs.iter().cloned().fold(0.0f32, f32::max),
        (0..batch)
            .map(|i| (diag.traces[i] - nocc).abs())
            .fold(0.0f32, f32::max)
    );

    // sys0 deep-dive: eigendecompose the returned D — which H eigenvectors
    // did the projector pick? Rayleigh quotients vᵀHv vs the true spectrum.
    {
        let hb = &h[0..n * n];
        let db = &d[0..n * n];
        let (eigs, _) = cpu_eig(hb, n);
        let mut dm = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                dm[(i, j)] = db[i * n + j] as f64;
            }
        }
        let de = dm.clone().symmetric_eigen();
        let mut rq: Vec<f64> = Vec::new();
        for c in 0..n {
            if de.eigenvalues[c] > 0.5 {
                let v = de.eigenvectors.column(c);
                let mut rqv = 0.0f64;
                for i in 0..n {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        s += hb[i * n + k] as f64 * v[k];
                    }
                    rqv += v[i] * s;
                }
                rq.push(rqv);
            }
        }
        rq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("[purify-dbg] sys0 D eigvals>0.5: count={} rq range [{:.3},{:.3}]",
            rq.len(), rq.first().copied().unwrap_or(0.0), rq.last().copied().unwrap_or(0.0));
        eprintln!("[purify-dbg] sys0 H spectrum: [{:.3} .. {:.3}], Fermi gap λ43={:.3} λ44={:.3}",
            eigs[0], eigs[n - 1], eigs[42], eigs[43]);
        let picked: Vec<String> = rq.iter().map(|r| format!("{r:.1}")).collect();
        eprintln!("[purify-dbg] sys0 picked eigenvalues: {}", picked.join(" "));
    }

    let mut bad = 0;
    for b in 0..batch {
        let hb = &h[b * n * n..(b + 1) * n * n];
        let db = &d[b * n * n..(b + 1) * n * n];
        let (eigs, vecs) = cpu_eig(hb, n);

        // variational energy: Tr(D·H) vs sum of nocc lowest eigenvalues
        let mut tr_dh = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                tr_dh += db[i * n + j] as f64 * hb[j * n + i] as f64;
            }
        }
        let e_ref: f64 = eigs[..nocc as usize].iter().map(|&e| e as f64).sum();

        // ‖D − D_ref‖_F, D_ref = V_occ·V_occᵀ
        let mut err2 = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut dref = 0.0f64;
                for k in 0..nocc as usize {
                    dref += vecs[(i, k)] * vecs[(j, k)];
                }
                err2 += (db[i * n + j] as f64 - dref).powi(2);
            }
        }
        // idempotency ‖D²−D‖_F and symmetry on host
        let mut idem = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut d2 = 0.0f64;
                for k in 0..n {
                    d2 += db[i * n + k] as f64 * db[k * n + j] as f64;
                }
                idem += (d2 - db[i * n + j] as f64).powi(2);
            }
        }
        let mut asym = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                asym += (db[i * n + j] - db[j * n + i]) as f64
                    * (db[i * n + j] - db[j * n + i]) as f64;
            }
        }
        // commutator ‖HD − DH‖ — iterates are polynomials of H, must commute
        let mut comm = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut hd = 0.0f64;
                let mut dh = 0.0f64;
                for k in 0..n {
                    hd += hb[i * n + k] as f64 * db[k * n + j] as f64;
                    dh += db[i * n + k] as f64 * hb[k * n + j] as f64;
                }
                comm += (hd - dh).powi(2);
            }
        }
        let d_energy = (tr_dh - e_ref).abs();
        let d_proj = err2.sqrt();
        let d_idem = idem.sqrt();
        let d_asym = asym.sqrt();
        let d_comm = comm.sqrt();
        if b < 4 || d_energy > 1e-3 || d_proj > 0.05 {
            eprintln!(
                "[purify] sys{b}: ΔE={d_energy:.3e} ‖D−Dref‖={d_proj:.3e} ‖D²−D‖={d_idem:.3e} asym={d_asym:.3e} ‖HD−DH‖={d_comm:.3e}"
            );
        }
        if d_energy > 1e-3 || d_proj > 0.05 || d_idem > 1e-3 || d_asym > 1e-4 {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad}/{batch} systems failed purification parity");
    assert!(diag.converged, "purify did not report convergence");
}

#[test]
fn purify_step_probe() {
    // Bisect: run exactly `PURIFY_PROBE_ITERS` kernel steps on one system and
    // compare element-wise against the same iteration sequence on CPU (f64).
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[probe] no GPU: {e}");
            return;
        }
    };
    let n = 86usize;
    let batch: usize = std::env::var("PURIFY_PROBE_BATCH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let nocc = vec![43f32; batch];
    let seed0: u64 = std::env::var("PURIFY_PROBE_SEED")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let mut h = Vec::with_capacity(batch * n * n);
    for b in 0..batch {
        h.extend_from_slice(&random_symmetric(n, seed0 + b as u64));
    }
    let h_buf = rt.buffer_from_slice(&h).unwrap();
    let probe_iters: usize = std::env::var("PURIFY_PROBE_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let probe_tol: f32 = std::env::var("PURIFY_PROBE_TOL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let probe_reps: usize = std::env::var("PURIFY_PROBE_REPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut d = vec![0.0f32; batch * n * n];
    for _ in 0..probe_reps {
        purify_tc2_batched(&mut rt, &h_buf, &mut d, n, batch, &nocc, probe_iters, probe_tol).unwrap();
    }

    for b in 0..batch {
        // CPU reference per system: same init + same branch semantics, f64
        let hb = &h[b * n * n..(b + 1) * n * n];
        let mut hm = DMatrix::<f64>::zeros(n, n);
        for i in 0..n { for j in 0..n { hm[(i, j)] = hb[i * n + j] as f64; } }
        let mut lmax = f64::MIN; let mut lmin = f64::MAX;
        for i in 0..n {
            let mut s = 0.0f64;
            for j in 0..n { s += hm[(i, j)].abs(); }
            let dg = hm[(i, i)];
            lmax = lmax.max(dg + (s - dg.abs()));
            lmin = lmin.min(dg - (s - dg.abs()));
        }
        let span = lmax - lmin;
        let mut dc = DMatrix::<f64>::zeros(n, n);
        for i in 0..n { for j in 0..n { dc[(i, j)] = ((if i == j { lmax } else { 0.0 }) - hm[(i, j)]) / span; } }
        for _ in 0..probe_iters {
            let t = &dc * &dc;
            let tr: f64 = dc.diagonal().sum();
            dc = if tr > 43.0 { t } else { 2.0 * &dc - t };
        }
        let db = &d[b * n * n..(b + 1) * n * n];
        let mut worst = (0.0f64, 0usize, 0.0f64, 0.0f64);
        let mut tr_gpu = 0.0f64;
        let mut tr_cpu = 0.0f64;
        let (mut dmax, mut omax) = (0.0f64, 0.0f64); // diagonal / off-diagonal err
        for i in 0..n {
            for j in 0..n {
                let diff = (db[i * n + j] as f64 - dc[(i, j)]).abs();
                if diff > worst.0 { worst = (diff, i * n + j, dc[(i, j)], db[i * n + j] as f64); }
                if i == j { dmax = dmax.max(diff); } else { omax = omax.max(diff); }
            }
            tr_gpu += db[i * n + i] as f64;
            tr_cpu += dc[(i, i)];
        }
        eprintln!("[probe] sys{b} iters={probe_iters} max|D_gpu−D_cpu|={:.3e} idx {} (cpu={:.6} gpu={:.6}) Tr={:.5}",
            worst.0, worst.1, worst.2, worst.3, tr_gpu);
        eprintln!("[probe] sys{b}   diag_err={:.3e} offdiag_err={:.3e} Tr_cpu={:.5} dTr={:.4e}",
            dmax, omax, tr_cpu, tr_gpu - tr_cpu);
        // per-diagonal-element dump: which (i,i) are corrupted, and how
        let mut bad = Vec::new();
        for i in 0..n {
            let diff = (db[i * n + i] as f64 - dc[(i, i)]).abs();
            if diff > 1e-4 { bad.push((i, diff, dc[(i, i)], db[i * n + i] as f64)); }
        }
        if !bad.is_empty() {
            let s: Vec<String> = bad.iter().take(16)
                .map(|(i, d, c, g)| format!("{i}:{d:.2}({c:.3}/{g:.3})")).collect();
            eprintln!("[probe] sys{b}   bad-diag {}: {}", bad.len(), s.join(" "));
        }
        // ‖D−Dref‖ with the true lowest-43 projector + variational energy
        let (eigs, vecs) = cpu_eig(hb, n);
        let mut pd = 0.0f64;
        let mut tr_dh = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut dref = 0.0f64;
                for k in 0..43 { dref += vecs[(i, k)] * vecs[(j, k)]; }
                pd += (db[i * n + j] as f64 - dref).powi(2);
                tr_dh += db[i * n + j] as f64 * hb[j * n + i] as f64;
            }
        }
        let e_ref: f64 = eigs[..43].iter().map(|&e| e as f64).sum();
        eprintln!("[probe] sys{b} ‖D−Dref‖={:.3e} ΔE={:.3e} (tr_dh={:.4} e_ref={:.4})",
            pd.sqrt(), (tr_dh - e_ref).abs(), tr_dh, e_ref);
    }
}

#[test]
#[ignore]
fn purify_bench() {
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[purify] no GPU: {e}");
            return;
        }
    };
    let n = 86usize;
    let batch: usize = std::env::var("PURIFY_BENCH_BATCH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let nocc = 43f32;

    let mut h = Vec::with_capacity(batch * n * n);
    for b in 0..batch {
        h.extend_from_slice(&random_symmetric(n, 2000 + b as u64));
    }
    let h_buf = rt.buffer_from_slice(&h).unwrap();
    let mut d = vec![0.0f32; batch * n * n];
    let nocc_v = vec![nocc; batch];

    // warm-up (includes program build + init)
    let mut diag = purify_tc2_batched(&mut rt, &h_buf, &mut d, n, batch, &nocc_v, 60, 1e-5).unwrap();
    let t0 = std::time::Instant::now();
    let reps = 5;
    for _ in 0..reps {
        diag = purify_tc2_batched(&mut rt, &h_buf, &mut d, n, batch, &nocc_v, 60, 1e-5).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    eprintln!(
        "[purify-bench] n={n} batch={batch}: {dt:.2} ms/solve = {:.4} ms/iter = {:.2} µs/sys/iter (iters={}, conv={}, max_err={:.2e})",
        dt / diag.iters as f64,
        dt / diag.iters as f64 / batch as f64 * 1e3,
        diag.iters,
        diag.converged,
        diag.errs.iter().cloned().fold(0.0f32, f32::max)
    );
    eprintln!("[purify-bench] ref: resident Jacobi one≈9.7 ms cold≈17.1 ms");
}

// ----------------------------------------------------------------------
// relax+purify — energy-minimizer formulation: purification is the
// retraction of a min_D Tr(D·H) descent, with a learned constraint
// force Λ (persistent across solves). Same certification contract as
// the TC2 test: ΔE, ‖D−Dref‖, ‖D²−D‖, asym, ‖HD−DH‖ on host f64.
// ----------------------------------------------------------------------

/// Host-side f64 certification of a density matrix — shared checks for
/// both purify paths. Returns (ΔE, ‖D−Dref‖, ‖D²−D‖, asym, ‖HD−DH‖).
fn certify_d(hb: &[f32], db: &[f32], n: usize, nocc: usize) -> (f64, f64, f64, f64, f64) {
    let (eigs, vecs) = cpu_eig(hb, n);
    let mut tr_dh = 0.0f64;
    let mut err2 = 0.0f64;
    let mut idem = 0.0f64;
    let mut asym = 0.0f64;
    let mut comm = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            tr_dh += db[i * n + j] as f64 * hb[j * n + i] as f64;
            let mut dref = 0.0f64;
            let mut d2 = 0.0f64;
            let mut hd = 0.0f64;
            let mut dh = 0.0f64;
            for k in 0..nocc {
                dref += vecs[(i, k)] * vecs[(j, k)];
            }
            for k in 0..n {
                d2 += db[i * n + k] as f64 * db[k * n + j] as f64;
                hd += hb[i * n + k] as f64 * db[k * n + j] as f64;
                dh += db[i * n + k] as f64 * hb[k * n + j] as f64;
            }
            err2 += (db[i * n + j] as f64 - dref).powi(2);
            idem += (d2 - db[i * n + j] as f64).powi(2);
            asym += (db[i * n + j] as f64 - db[j * n + i] as f64).powi(2);
            comm += (hd - dh).powi(2);
        }
    }
    let e_ref: f64 = eigs[..nocc].iter().map(|&e| e as f64).sum();
    (
        (tr_dh - e_ref).abs(),
        err2.sqrt(),
        idem.sqrt(),
        asym.sqrt(),
        comm.sqrt(),
    )
}

#[test]
fn test_relax_purify_parity() {
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[relax] no GPU: {e}");
            return;
        }
    };
    let n = 86usize;
    let batch = 16usize;
    let nocc = 43f32;

    let mut h = Vec::with_capacity(batch * n * n);
    for b in 0..batch {
        h.extend_from_slice(&random_symmetric(n, 1000 + b as u64));
    }
    let h_buf = rt.buffer_from_slice(&h).unwrap();
    let lam_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    let nocc_v = vec![nocc; batch];

    // ---- cold solve via TC2 (the proven cold path — the relax
    // minimizer is the WARM/SCC-iteration path; cold Palser is not
    // near enough to the manifold for the stateless descent). ----
    let mut d0 = vec![0.0f32; batch * n * n];
    let pdiag = purify_tc2_batched(&mut rt, &h_buf, &mut d0, n, batch, &nocc_v, 200, 1e-5)
        .unwrap();
    assert!(pdiag.converged, "cold TC2 did not converge");
    let mut bad = 0;
    for b in 0..batch {
        let hb = &h[b * n * n..(b + 1) * n * n];
        let db = &d0[b * n * n..(b + 1) * n * n];
        let (d_energy, d_proj, d_idem, d_asym, _) = certify_d(hb, db, n, nocc as usize);
        if d_energy > 1e-3 || d_proj > 0.05 || d_idem > 1e-3 || d_asym > 1e-4 {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad}/{batch} systems failed cold TC2 parity");

    // ---- warm solves: stateless descent F = H−aI−bD + retraction
    // (β=0, Λ=0 — the decisive experiment: fixed point must be
    // [D,H]=0 for ANY Δ, not just small perturbations). ----
    let lam_zero = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    for (wd, seed) in [(0.02f32, 5000u64), (0.5, 7000)] {
        let mut h2 = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let r = random_symmetric(n, seed + b as u64);
            let hb = &h[b * n * n..(b + 1) * n * n];
            h2.extend((0..n * n).map(|i| hb[i] + wd * r[i]));
        }
        let h2_buf = rt.buffer_from_slice(&h2).unwrap();
        // restart from the SAME converged D0 each Δ
        let d_buf = rt.buffer_from_slice(&d0).unwrap();
        let diag2 = relax_purify_batched(
            &mut rt, &h2_buf, &d_buf, &lam_zero, n, batch, &nocc_v, true, 200, 1e-5,
        )
        .unwrap();
        let mut d = vec![0.0f32; batch * n * n];
        rt.read_buffer(&d_buf, &mut d).unwrap();
        eprintln!(
            "[relax] warm(Δ={wd}): iters={} converged={} max_comm_gate={:.3e}",
            diag2.iters,
            diag2.converged,
            diag2.comms.iter().cloned().fold(0.0f32, f32::max)
        );

        let mut bad2 = 0;
        let mut comm_max = 0.0f64;
        for b in 0..batch {
            let hb = &h2[b * n * n..(b + 1) * n * n];
            let db = &d[b * n * n..(b + 1) * n * n];
            let (d_energy, d_proj, d_idem, d_asym, d_comm) =
                certify_d(hb, db, n, nocc as usize);
            comm_max = comm_max.max(d_comm);
            if b < 4 || d_energy > 1e-3 || d_proj > 0.05 {
                eprintln!(
                    "[relax-warm] sys{b}: ΔE={d_energy:.3e} ‖D−Dref‖={d_proj:.3e} ‖D²−D‖={d_idem:.3e} asym={d_asym:.3e} ‖HD−DH‖={d_comm:.3e}"
                );
            }
            if d_energy > 1e-3 || d_proj > 0.05 || d_idem > 1e-3 || d_asym > 1e-4 {
                bad2 += 1;
            }
        }
        eprintln!("[relax] warm(Δ={wd}): max ‖HD−DH‖={comm_max:.3e}");
        assert_eq!(bad2, 0, "warm Δ={wd}: {bad2}/{batch} systems failed relax parity");
        assert!(diag2.converged, "warm Δ={wd} relax did not report convergence");
    }
}

#[test]
#[ignore]
fn relax_bench() {
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[relax] no GPU: {e}");
            return;
        }
    };
    let n = 86usize;
    let batch: usize = std::env::var("PURIFY_BENCH_BATCH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let nocc = 43f32;
    let max_iter: usize = std::env::var("RELAX_BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(80);

    let mut h = Vec::with_capacity(batch * n * n);
    for b in 0..batch {
        h.extend_from_slice(&random_symmetric(n, 2000 + b as u64));
    }
    let h_buf = rt.buffer_from_slice(&h).unwrap();
    let d_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    let lam_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    let nocc_v = vec![nocc; batch];

    // warm-up + cold reference
    let dg0 = relax_purify_batched(
        &mut rt, &h_buf, &d_buf, &lam_buf, n, batch, &nocc_v, false, max_iter, 1e-5,
    )
    .unwrap();
    let t0 = std::time::Instant::now();
    let reps = 3;
    let mut dg = None;
    for _ in 0..reps {
        // re-cold each rep: reset Λ is inside the cold path
        dg = Some(relax_purify_batched(
            &mut rt, &h_buf, &d_buf, &lam_buf, n, batch, &nocc_v, false, max_iter, 1e-5,
        ).unwrap());
    }
    let dt = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let dg = dg.unwrap();
    eprintln!(
        "[relax-bench] COLD n={n} batch={batch}: {dt:.2} ms/solve = {:.4} ms/step (iters={}, conv={}, max_step={:.2e})",
        dt / dg.iters.max(1) as f64,
        dg.iters,
        dg.converged,
        dg.steps.iter().cloned().fold(0.0f32, f32::max)
    );

    // warm: small Hamiltonian perturbation (the SCC/geometry-delta regime)
    let mut h2 = Vec::with_capacity(batch * n * n);
    for b in 0..batch {
        let r = random_symmetric(n, 9000 + b as u64);
        let hb = &h[b * n * n..(b + 1) * n * n];
        h2.extend((0..n * n).map(|i| hb[i] + 0.02f32 * r[i]));
    }
    let h2_buf = rt.buffer_from_slice(&h2).unwrap();
    let t0 = std::time::Instant::now();
    let mut dg2 = None;
    for _ in 0..reps {
        // each rep re-solves h2 warm from the converged (D,Λ) of h —
        // identical start state per rep
        let d2 = rt.zero_buffer::<f32>(batch * n * n).unwrap();
        let l2 = rt.zero_buffer::<f32>(batch * n * n).unwrap();
        rt.copy_into(&d_buf, &d2, batch * n * n).unwrap();
        rt.copy_into(&lam_buf, &l2, batch * n * n).unwrap();
        dg2 = Some(relax_purify_batched(
            &mut rt, &h2_buf, &d2, &l2, n, batch, &nocc_v, true, max_iter, 1e-5,
        ).unwrap());
    }
    let dt2 = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let dg2 = dg2.unwrap();
    eprintln!(
        "[relax-bench] WARM(Δ=0.02): {dt2:.2} ms/solve = {:.4} ms/step (iters={}, conv={})",
        dt2 / dg2.iters.max(1) as f64,
        dg2.iters,
        dg2.converged
    );
    eprintln!("[relax-bench] refs: purify-TC2 cold≈8.5 ms · Jacobi one≈9.7 ms cold≈17.1 ms · cold iters={}", dg0.iters);
}

// ==================================================================
// LNV canonical reference solver (gpu_matrix::lnv_solve_batched)
// ==================================================================

use rust_dftb::qmqm::gpu_matrix::lnv_solve_batched;

fn gershgorin_bounds(hb: &[f32], n: usize) -> (f32, f32) {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for i in 0..n {
        let mut r = 0.0f32;
        for j in 0..n {
            if j != i {
                r += hb[i * n + j].abs();
            }
        }
        lo = lo.min(hb[i * n + i] - r);
        hi = hi.max(hb[i * n + i] + r);
    }
    (lo, hi)
}

/// f64 reference: residual-form LNV gradient
///   G = (B+Bᵀ−2C) + 3[(A−B)+(A−B)ᵀ]
/// with S=L², F=(H−μI)/span, A=LF, B=SF, C=A·L.
fn lnv_grad_f64(l: &[f64], h: &[f64], mu: f64, span: f64, n: usize) -> Vec<f64> {
    let mm = |x: &[f64], y: &[f64]| -> Vec<f64> {
        let mut o = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += x[i * n + k] * y[k * n + j];
                }
                o[i * n + j] = s;
            }
        }
        o
    };
    let f: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (h[idx] - if i == j { mu } else { 0.0 }) / span
        })
        .collect();
    let s = mm(l, l);
    let a = mm(l, &f);
    let b = mm(&s, &f);
    let c = mm(&a, l);
    (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let t = j * n + i;
            (b[idx] + b[t] - 2.0 * c[idx]) + 3.0 * ((a[idx] - b[idx]) + (a[t] - b[t]))
        })
        .collect()
}

/// Gradient identity: one LNV step must give L₁ = L₀ − α·G_ref where
/// G_ref is the f64 residual-form gradient. At L=P (clean projector)
/// G ≡ [P,[P,H]]/Δε — the exact tangent force; at a generic interior
/// L the full residual form is checked.
#[test]
fn test_lnv_grad_identity() {
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[lnv] no GPU: {e}");
            return;
        }
    };
    let n = 86usize;
    let nocc = 43usize;
    let eta = 0.2f32;

    // --- case A: L = P — a rank-nocc projector NOT commuting with H
    // (built from a different matrix's eigenvectors), so G = T_P(F) ≠ 0.
    // (At the EXACT eig(H) projector G≡0 — a vacuous test.) ---
    let h = random_symmetric(n, 9001);
    let (lo, hi) = gershgorin_bounds(&h, n);
    let span = hi - lo;
    let (_eigs, vecs) = cpu_eig(&random_symmetric(n, 9002), n);
    let mut p64 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..nocc {
                s += vecs[(i, k)] * vecs[(j, k)];
            }
            p64[i * n + j] = s;
        }
    }
    for (name, l64) in [
        ("projector", p64.clone()),
        (
            "interior",
            (0..n * n)
                .map(|i| 0.5 * p64[i] + 0.3 * random_symmetric(n, 7777)[i] as f64 / span as f64)
                .collect::<Vec<_>>(),
        ),
    ] {
        let h64: Vec<f64> = h.iter().map(|&x| x as f64).collect();
        let g_ref = lnv_grad_f64(&l64, &h64, 0.0, span as f64, n);
        let l0: Vec<f32> = l64.iter().map(|&x| x as f32).collect();
        let h_buf = rt.buffer_from_slice(&h).unwrap();
        let l_buf = rt.buffer_from_slice(&l0).unwrap();
        let d_buf = rt.zero_buffer::<f32>(n * n).unwrap();
        lnv_solve_batched(
            &mut rt, &h_buf, &l_buf, &d_buf, n, 1,
            &[nocc as f32], &[0.0], &[span], 1, 0.0,
        )
        .unwrap();
        let mut l1 = vec![0.0f32; n * n];
        rt.read_buffer(&l_buf, &mut l1).unwrap();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..n * n {
            let g_gpu = (l0[i] - l1[i]) as f64 / eta as f64;
            num += (g_gpu - g_ref[i]).powi(2);
            den += g_ref[i].powi(2);
        }
        let rel = num.sqrt() / den.sqrt();
        eprintln!("[lnv-grad] {name}: rel err {rel:.3e}");
        assert!(rel < 5e-3, "LNV gradient ({name}) rel err {rel:.3e} > 5e-3");
    }
}

/// Solve parity: LNV steepest descent must reach the variational
/// minimum — certified by energy, projector, idempotency, symmetry,
/// commutator against the CPU eigensolver (same certify_d as TC2).
/// Warm mode is the intended use: L₀ = D_prev (f(P)=P).
#[test]
fn test_lnv_solve_parity() {
    let mut rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[lnv] no GPU: {e}");
            return;
        }
    };
    let n = 86usize;
    let batch = 8usize;
    let nocc = 43f32;

    let mut h = Vec::with_capacity(batch * n * n);
    let mut spans = Vec::with_capacity(batch);
    let mut mu0 = Vec::with_capacity(batch);
    for b in 0..batch {
        let hb = random_symmetric(n, 1000 + b as u64);
        let (lo, hi) = gershgorin_bounds(&hb, n);
        spans.push(hi - lo);
        mu0.push(lo + (nocc / n as f32) * (hi - lo)); // quantile Fermi guess
        h.extend_from_slice(&hb);
    }
    let h_buf = rt.buffer_from_slice(&h).unwrap();
    let nocc_v = vec![nocc; batch];

    let d_buf = rt.zero_buffer::<f32>(batch * n * n).unwrap();
    let mut d = vec![0.0f32; batch * n * n];

    // ---- warm: L₀ = D_ref(H) (exact projector), solve under
    // H₂ = H + Δ·R for Δ = 0.02 and 0.5 — must reach the NEW exact
    // projector with the true commutator → 0. ----
    for (wd, seed) in [(0.02f32, 5000u64), (0.5, 7000)] {
        let mut h2 = Vec::with_capacity(batch * n * n);
        let mut lw = vec![0.0f32; batch * n * n];
        let mut muw = Vec::with_capacity(batch);
        let mut spw = Vec::with_capacity(batch);
        for b in 0..batch {
            let r = random_symmetric(n, seed + b as u64);
            let hb = &h[b * n * n..(b + 1) * n * n];
            let h2b: Vec<f32> = (0..n * n).map(|i| hb[i] + wd * r[i]).collect();
            let (lo, hi) = gershgorin_bounds(&h2b, n);
            spw.push(hi - lo);
            muw.push(lo + (nocc / n as f32) * (hi - lo));
            h2.extend_from_slice(&h2b);
            // L₀ = exact occupied projector of the UNPERTURBED H
            let (_e, vecs) = cpu_eig(hb, n);
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f64;
                    for k in 0..nocc as usize {
                        s += vecs[(i, k)] * vecs[(j, k)];
                    }
                    lw[b * n * n + i * n + j] = s as f32;
                }
            }
        }
        let h2_buf = rt.buffer_from_slice(&h2).unwrap();
        let lw_buf = rt.buffer_from_slice(&lw).unwrap();
        let lnv_it: usize = std::env::var("RUST_DFTB_LNV_TEST_ITERS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(10000);
        // g<1e-6 needed to resolve the near-degenerate Fermi-edge
        // rotation in the hardest system (gap/span ~ 3e-4).
        let lnv_tol: f32 = std::env::var("RUST_DFTB_LNV_TEST_TOL")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1e-6);
        let dg2 = lnv_solve_batched(
            &mut rt, &h2_buf, &lw_buf, &d_buf, n, batch, &nocc_v, &muw, &spw, lnv_it, lnv_tol,
        )
        .unwrap();
        rt.read_buffer(&d_buf, &mut d).unwrap();
        let mut bad2 = 0;
        let mut cmax = 0.0f64;
        for b in 0..batch {
            let hb = &h2[b * n * n..(b + 1) * n * n];
            let db = &d[b * n * n..(b + 1) * n * n];
            let (de, dp, di, da, dc) = certify_d(hb, db, n, nocc as usize);
            cmax = cmax.max(dc);
            if b < 4 || de > 1e-3 || dp > 0.05 {
                eprintln!("[lnv-warm Δ={wd}] sys{b}: ΔE={de:.3e} ‖D−Dref‖={dp:.3e} ‖D²−D‖={di:.3e} asym={da:.3e} ‖HD−DH‖={dc:.3e}");
            }
            if de > 1e-3 || dp > 0.05 || di > 1e-3 || da > 1e-4 {
                bad2 += 1;
            }
        }
        eprintln!(
            "[lnv] warm(Δ={wd}): iters={} converged={} bad={bad2}/{batch} max_comm={cmax:.3e}",
            dg2.iters, dg2.converged
        );
        assert_eq!(bad2, 0, "warm Δ={wd}: {bad2}/{batch} systems failed LNV parity");
        assert!(dg2.converged, "warm Δ={wd} LNV did not converge");
    }

    // ---- cold stress case: L₀ = 0.5·I − 0.4·(H−ḢI)/span. Bare SD is
    // not the intended cold path (that is TC2's job) — marginal L
    // eigenvalues near the Fermi level expel ∝|ε−μ|, a slow tail on a
    // dense spectrum. Kept LAST and STRICT so the limitation is loud,
    // not hidden. ----
    let mut l0 = vec![0.0f32; batch * n * n];
    for b in 0..batch {
        let hb = &h[b * n * n..(b + 1) * n * n];
        let tr: f64 = (0..n).map(|i| hb[i * n + i] as f64).sum();
        let hbar = (tr / n as f64) as f32;
        let isp = 0.4 / spans[b];
        for i in 0..n {
            for j in 0..n {
                l0[b * n * n + i * n + j] =
                    (if i == j { 0.5 } else { 0.0 }) - isp * (hb[i * n + j] - if i == j { hbar } else { 0.0 });
            }
        }
    }
    let l_buf = rt.buffer_from_slice(&l0).unwrap();
    let dg = lnv_solve_batched(
        &mut rt, &h_buf, &l_buf, &d_buf, n, batch, &nocc_v, &mu0, &spans, 1500, 1e-5,
    )
    .unwrap();
    rt.read_buffer(&d_buf, &mut d).unwrap();
    eprintln!(
        "[lnv] cold: iters={} converged={} max_g={:.3e} max|Tr−Nocc|={:.3e} max_comm={:.3e}",
        dg.iters,
        dg.converged,
        dg.gnorms.iter().cloned().fold(0.0f32, f32::max),
        (0..batch).map(|i| (dg.trds[i] - nocc).abs()).fold(0.0f32, f32::max),
        dg.comms.iter().cloned().fold(0.0f32, f32::max)
    );
    let mut bad = 0;
    for b in 0..batch {
        let hb = &h[b * n * n..(b + 1) * n * n];
        let db = &d[b * n * n..(b + 1) * n * n];
        let (de, dp, di, da, dc) = certify_d(hb, db, n, nocc as usize);
        if b < 4 || de > 1e-3 || dp > 0.05 {
            eprintln!("[lnv-cold] sys{b}: ΔE={de:.3e} ‖D−Dref‖={dp:.3e} ‖D²−D‖={di:.3e} asym={da:.3e} ‖HD−DH‖={dc:.3e}");
        }
        if de > 1e-3 || dp > 0.05 || di > 1e-3 || da > 1e-4 {
            bad += 1;
        }
    }
    eprintln!("[lnv] cold: {bad}/{batch} failed");
    assert_eq!(bad, 0, "cold LNV: {bad}/{batch} systems failed parity");
    assert!(dg.converged, "cold LNV did not converge");
}
