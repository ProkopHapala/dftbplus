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
use rust_dftb::qmqm::gpu_purify::purify_tc2_batched;
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
        for i in 0..n {
            for j in 0..n {
                let diff = (db[i * n + j] as f64 - dc[(i, j)]).abs();
                if diff > worst.0 { worst = (diff, i * n + j, dc[(i, j)], db[i * n + j] as f64); }
            }
            tr_gpu += db[i * n + i] as f64;
        }
        eprintln!("[probe] sys{b} iters={probe_iters} max|D_gpu−D_cpu|={:.3e} idx {} (cpu={:.6} gpu={:.6}) Tr={:.5}",
            worst.0, worst.1, worst.2, worst.3, tr_gpu);
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
    let batch = 400usize;
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
        "[purify-bench] n={n} batch={batch}: {dt:.2} ms/solve (iters={}, conv={}, max_err={:.2e})",
        diag.iters,
        diag.converged,
        diag.errs.iter().cloned().fold(0.0f32, f32::max)
    );
    eprintln!("[purify-bench] ref: resident Jacobi one≈9.7 ms cold≈17.1 ms");
}
