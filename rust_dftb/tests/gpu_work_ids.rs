//! T06 Phase A acceptance: compact launch domains via `work_ids`.
//!
//! Every converted batched kernel maps its launch index through
//! `work_ids[iw]` → physical replica slot. Launching a shrunken grid
//! with a non-identity `work_ids` must update ONLY the listed physical
//! slots — the addressing primitive the slot-pool scheduler builds on.
//!
//! Covers all four replica-axis patterns plus the eigensolver:
//!   1. `batched_gemm_active`      — 3D grid, dim-2 = system
//!   2. `commit_q_batched`         — 1 workgroup per system (group-0)
//!   3. `extract_diagonal_batched` — flat elementwise (system,orbital) grid
//!   4. `cs_normalize_batched`     — (system,column) group-0 grid
//!   5. `jacobi_resident_batched`  — eigensolve of one subset slot
//!   6. `active[]` interplay       — frozen slot inside a compact domain

use nalgebra::DMatrix;
use ocl::Kernel;
use rust_dftb::qmqm::gpu_eigen::render_resident_source;
use rust_dftb::qmqm::gpu_matrix::MatrixKernelConfig;
use rust_dftb::qmqm::gpu_runtime::GpuRuntime;

fn try_runtime() -> Option<GpuRuntime> {
    match GpuRuntime::new() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("Skipping GPU test: no OpenCL device ({e})");
            None
        }
    }
}

/// Assert that `buf` holds `sentinel` at every slot not in `touched`.
fn assert_untouched(
    got: &[f32],
    batch: usize,
    stride: usize,
    touched: &[usize],
    sentinel: f32,
    what: &str,
) {
    for s in 0..batch {
        if touched.contains(&s) {
            continue;
        }
        for i in 0..stride {
            assert_eq!(
                got[s * stride + i],
                sentinel,
                "{what}: slot {s} elem {i} must be untouched sentinel {sentinel}, got {}",
                got[s * stride + i]
            );
        }
    }
}

/// 1. 3D GEMM — dim-2 launch index maps to physical slot.
#[test]
fn test_work_ids_gemm_compact() {
    let Some(mut rt) = try_runtime() else { return };
    let (n, batch) = (8usize, 4usize);
    let nn = n * n;
    let cfg = MatrixKernelConfig::nvidia_default();
    let prog = rt.build_program(&cfg.render_source()).unwrap();

    // A[s] = (s+1)·I so C[s] = A·B is trivially checkable.
    let mut a = vec![0f32; batch * nn];
    let mut b = vec![0f32; batch * nn];
    for s in 0..batch {
        for i in 0..n {
            a[s * nn + i * n + i] = (s + 1) as f32;
        }
        for k in 0..nn {
            b[s * nn + k] = (k + 1) as f32;
        }
    }
    let c_host = vec![-777f32; batch * nn];
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let b_buf = rt.buffer_from_slice(&b).unwrap();
    let c_buf = rt.buffer_from_slice(&c_host).unwrap();
    let active = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    // Compact domain: launch only physical slots 2 and 0 (permuted order).
    let ids = [2i32, 0];
    let wids = rt.buffer_from_slice(&ids).unwrap();
    let nw = ids.len();

    let k = Kernel::builder()
        .program(&prog)
        .name("batched_gemm_active")
        .queue(rt.queue().clone())
        .global_work_size(ocl::SpatialDims::Three(
            ((n + cfg.tile_n - 1) / cfg.tile_n) * cfg.tile_n,
            ((n + cfg.tile_m - 1) / cfg.tile_m) * cfg.tile_m,
            nw,
        ))
        .local_work_size(ocl::SpatialDims::Two(cfg.tile_n, cfg.tile_m))
        .arg(n as i32)
        .arg(batch as i32)
        .arg(0i32)
        .arg(0i32)
        .arg(1.0f32)
        .arg(0.0f32)
        .arg(&a_buf)
        .arg(&b_buf)
        .arg(&c_buf)
        .arg_local::<f32>(cfg.tile_m * cfg.tile_k)
        .arg_local::<f32>(cfg.tile_n * (cfg.tile_k + 1))
        .arg(&active)
        .arg(&wids)
        .build()
        .unwrap();
    unsafe { k.enq().unwrap() };
    let mut got = vec![0f32; batch * nn];
    rt.read_buffer(&c_buf, &mut got).unwrap();

    for &s in &[2usize, 0] {
        for i in 0..n {
            for j in 0..n {
                let want = (s + 1) as f32 * (i * n + j + 1) as f32;
                assert!(
                    (got[s * nn + i * n + j] - want).abs() < 1e-5,
                    "gemm slot {s}[{i},{j}]: want {want} got {}",
                    got[s * nn + i * n + j]
                );
            }
        }
    }
    assert_untouched(&got, batch, nn, &[2, 0], -777.0, "gemm");
}

/// 2. 1-WG-per-system kernel — group-0 mapping, single subset slot.
#[test]
fn test_work_ids_one_wg_compact() {
    let Some(mut rt) = try_runtime() else { return };
    let (n_atoms, batch) = (5usize, 4usize);
    let cfg = MatrixKernelConfig::nvidia_default();
    let prog = rt.build_program(&cfg.render_source()).unwrap();

    let mut q_next = vec![0f32; batch * n_atoms];
    for s in 0..batch {
        for a in 0..n_atoms {
            q_next[s * n_atoms + a] = 100.0 * s as f32 + a as f32;
        }
    }
    let q_buf = rt
        .buffer_from_slice(&vec![-999f32; batch * n_atoms])
        .unwrap();
    let qn_buf = rt.buffer_from_slice(&q_next).unwrap();
    let active = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    let ids = [2i32];
    let wids = rt.buffer_from_slice(&ids).unwrap();
    let wg = 64usize;

    let k = Kernel::builder()
        .program(&prog)
        .name("commit_q_batched")
        .queue(rt.queue().clone())
        .global_work_size(ids.len() * wg)
        .local_work_size(wg)
        .arg(n_atoms as i32)
        .arg(batch as i32)
        .arg(&qn_buf)
        .arg(&q_buf)
        .arg(&active)
        .arg(&wids)
        .build()
        .unwrap();
    unsafe { k.enq().unwrap() };
    let mut got = vec![0f32; batch * n_atoms];
    rt.read_buffer(&q_buf, &mut got).unwrap();

    for a in 0..n_atoms {
        assert_eq!(
            got[2 * n_atoms + a],
            q_next[2 * n_atoms + a],
            "commit slot 2 atom {a}"
        );
    }
    assert_untouched(&got, batch, n_atoms, &[2], -999.0, "commit_q");
}

/// 3. Flat elementwise grid — sid AND output index must both remap.
#[test]
fn test_work_ids_elementwise_compact() {
    let Some(mut rt) = try_runtime() else { return };
    let (n, batch) = (6usize, 4usize);
    let nn = n * n;
    let cfg = MatrixKernelConfig::nvidia_default();
    let prog = rt.build_program(&cfg.render_source()).unwrap();

    let mut a = vec![0f32; batch * nn];
    for s in 0..batch {
        for i in 0..n {
            a[s * nn + i * n + i] = 10.0 * s as f32 + i as f32;
        }
    }
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let diag_buf = rt.buffer_from_slice(&vec![-555f32; batch * n]).unwrap();
    let active = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    let ids = [3i32, 1];
    let wids = rt.buffer_from_slice(&ids).unwrap();

    let k = Kernel::builder()
        .program(&prog)
        .name("extract_diagonal_batched")
        .queue(rt.queue().clone())
        .global_work_size(ids.len() * n)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(&a_buf)
        .arg(&diag_buf)
        .arg(&active)
        .arg(&wids)
        .build()
        .unwrap();
    unsafe { k.enq().unwrap() };
    let mut got = vec![0f32; batch * n];
    rt.read_buffer(&diag_buf, &mut got).unwrap();

    for &s in &[3usize, 1] {
        for i in 0..n {
            assert_eq!(got[s * n + i], a[s * nn + i * n + i], "diag slot {s}[{i}]");
        }
    }
    assert_untouched(&got, batch, n, &[3, 1], -555.0, "extract_diagonal");
}

/// 4. (system,column) group-0 grid — column comes from launch index, slot from work_ids.
#[test]
fn test_work_ids_sys_col_compact() {
    let Some(mut rt) = try_runtime() else { return };
    let (n, batch) = (4usize, 4usize);
    let nn = n * n;
    let cfg = MatrixKernelConfig::nvidia_default();
    let prog = rt.build_program(&cfg.render_source()).unwrap();

    // C columns with distinct norms; SC = C so cs_normalize → unit L2 cols.
    let mut c = vec![0f32; batch * nn];
    for s in 0..batch {
        for i in 0..n {
            for j in 0..n {
                c[s * nn + i * n + j] =
                    ((s + 1) * (i + 1)) as f32 * (j == i % n) as i32 as f32 + 0.5 * (s + 1) as f32;
            }
        }
    }
    let c_buf = rt.buffer_from_slice(&c).unwrap();
    let sc_buf = rt.buffer_from_slice(&c).unwrap(); // SC = C copy
    let active = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    let ids = [1i32, 3];
    let wids = rt.buffer_from_slice(&ids).unwrap();
    let wg = 32usize;

    let k = Kernel::builder()
        .program(&prog)
        .name("cs_normalize_batched")
        .queue(rt.queue().clone())
        .global_work_size(ids.len() * n * wg)
        .local_work_size(wg)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(&c_buf)
        .arg(&sc_buf)
        .arg_local::<f32>(wg)
        .arg(&active)
        .arg(&wids)
        .build()
        .unwrap();
    unsafe { k.enq().unwrap() };
    let mut got = vec![0f32; batch * nn];
    rt.read_buffer(&c_buf, &mut got).unwrap();

    // Columns of slots 1,3 must be unit L2 norm (SC==C → plain renorm).
    for &s in &[1usize, 3] {
        for col in 0..n {
            let nrm: f32 = (0..n)
                .map(|i| got[s * nn + i * n + col].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                (nrm - 1.0).abs() < 1e-4,
                "cs_normalize slot {s} col {col}: |C|={nrm}"
            );
        }
    }
    // Slots 0,2 must equal the original C exactly.
    for &s in &[0usize, 2] {
        for k in 0..nn {
            assert_eq!(
                got[s * nn + k],
                c[s * nn + k],
                "cs_normalize untouched slot {s}[{k}]"
            );
        }
    }
}

/// 5. Resident Jacobi on a one-slot compact domain — only slot 2 is
/// diagonalized; all other A/V slots must be untouched.
#[test]
fn test_work_ids_resident_jacobi_subset() {
    let Some(mut rt) = try_runtime() else { return };
    let (n, batch, target) = (32usize, 4usize, 2usize);
    let nn = n * n;
    let local_cap = rt.caps().local_mem_size;
    if (n * (n + 1) * 4) as u64 + 16 * 1024 > local_cap {
        eprintln!("skip: resident lA over local cap");
        return;
    }

    // Distinct symmetric matrices per slot; keep a CPU copy of slot 2.
    let mut a = vec![0f32; batch * nn];
    for s in 0..batch {
        let mut rng = 1234u64 + s as u64;
        let mut m = vec![0f32; nn];
        for i in 0..n {
            for j in i..n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((rng >> 33) & 0xffff) as f32 / 65536.0 - 0.5;
                m[i * n + j] = v;
                m[j * n + i] = v;
            }
        }
        for i in 0..n {
            m[i * n + i] += n as f32; // diagonally dominant → clean eigensolve
        }
        a[s * nn..(s + 1) * nn].copy_from_slice(&m);
    }
    // CPU reference eigenvalues for the target slot.
    let dm = DMatrix::from_row_slice(
        n,
        n,
        &a[target * nn..(target + 1) * nn]
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<_>>(),
    );
    let mut want_eig: Vec<f64> = dm.symmetric_eigenvalues().iter().cloned().collect();
    want_eig.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let v_buf = rt.zero_buffer::<f32>(batch * nn).unwrap();
    let active = rt.buffer_from_slice(&vec![1i32; batch]).unwrap();
    let diag = rt.zero_buffer::<f32>(4 * batch).unwrap();
    let occ_w = rt.zero_buffer::<f32>(batch * n).unwrap();
    let mu = rt.zero_buffer::<f32>(batch).unwrap();
    let jn = if n & 1 == 1 { n + 1 } else { n };
    let rotlog = rt
        .zero_buffer::<f32>(batch * (jn - 1) * (jn / 2) * 4)
        .unwrap(); // prec=1 → double2 jlog2_t
    let ids = [target as i32];
    let wids = rt.buffer_from_slice(&ids).unwrap();
    let wg = 256usize.min(rt.caps().max_work_group_size);

    let prog = rt
        .build_program(&render_resident_source(wg, 1, false, true))
        .unwrap();
    let k = Kernel::builder()
        .program(&prog)
        .name("jacobi_resident_batched")
        .queue(rt.queue().clone())
        .global_work_size(ids.len() * wg)
        .local_work_size(wg)
        .arg(&a_buf)
        .arg(&v_buf)
        .arg(n as i32)
        .arg(batch as i32)
        .arg(0i32)
        .arg(&active)
        .arg(&diag)
        .arg(0i32)
        .arg(0i32)
        .arg(0.0f32)
        .arg(&occ_w)
        .arg(&mu)
        .arg(&rotlog)
        .arg_local::<f32>(n * (n + 1) / 2) // packed symmetric lA
        .arg_local::<f32>(1)
        .arg(&wids)
        .build()
        .unwrap();
    unsafe { k.enq().unwrap() };
    rt.queue().finish().unwrap();

    let mut ga = vec![0f32; batch * nn];
    let mut gv = vec![0f32; batch * nn];
    rt.read_buffer(&a_buf, &mut ga).unwrap();
    rt.read_buffer(&v_buf, &mut gv).unwrap();

    // Slot 2: A diagonal with correct eigenvalues; V orthonormal.
    let mut got_eig: Vec<f64> = (0..n).map(|i| ga[target * nn + i * n + i] as f64).collect();
    got_eig.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let off: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| i != j)
        .map(|(i, j)| ga[target * nn + i * n + j] as f64)
        .fold(0.0, |m, x| m.max(x.abs()));
    assert!(off < 1e-4, "resident subset: off-diag {off:e}");
    for i in 0..n {
        assert!(
            (got_eig[i] - want_eig[i]).abs() < 1e-3 * want_eig[i].abs().max(1.0),
            "eig[{i}]: got {} want {}",
            got_eig[i],
            want_eig[i]
        );
    }
    for i in 0..n {
        let nrm: f64 = (0..n)
            .map(|r| (gv[target * nn + r * n + i] as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((nrm - 1.0).abs() < 1e-4, "V col {i} norm {nrm}");
    }
    // Slots 0,1,3 of A untouched; all non-target V slots still zero.
    for s in 0..batch {
        if s == target {
            continue;
        }
        for kk in 0..nn {
            assert_eq!(
                ga[s * nn + kk],
                a[s * nn + kk],
                "A slot {s}[{kk}] untouched"
            );
            assert_eq!(gv[s * nn + kk], 0.0, "V slot {s}[{kk}] untouched");
        }
    }
}

/// Phase B measurement: per-launch cost vs active-slot count S.
///
/// For each S the dominant SCC kernel (`jacobi_resident_batched`, n=86)
/// is timed two ways on the same batch-400 data:
///   * `compact` — work_ids=[0..S), launch S workgroups (Phase B path)
///   * `masked`  — work_ids=[0..400), launch all 400 workgroups with
///                 active[]=0 on slots ≥ S (Phase A status quo)
/// The `masked − compact` gap is the dead-workgroup overhead the
/// slot-pool scheduler removes; its S-dependence shows where the GPU
/// saturates (the scheduler's S* knee).
#[test]
#[ignore]
fn work_ids_saturation_sweep() {
    use std::time::Instant;
    let Some(mut rt) = try_runtime() else { return };
    let (n, batch) = (86usize, 400usize);
    let nn = n * n;
    let local_cap = rt.caps().local_mem_size;
    if (n * (n + 1) * 4) as u64 + 16 * 1024 > local_cap {
        eprintln!("[wsweep] skip: resident lA over local cap");
        return;
    }

    let mut a = vec![0f32; batch * nn];
    for s in 0..batch {
        let mut rng = 99u64 + s as u64;
        let mut m = vec![0f32; nn];
        for i in 0..n {
            for j in i..n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = (((rng >> 33) & 0xffff) as f32 / 65536.0 - 0.5) * 3e-3;
                m[i * n + j] = v;
                m[j * n + i] = v;
            }
            m[i * n + i] = 1.0 + ((rng >> 40) & 0xff) as f32 / 256.0;
        }
        a[s * nn..(s + 1) * nn].copy_from_slice(&m);
    }
    let a_buf = rt.buffer_from_slice(&a).unwrap();
    let v_buf = rt.zero_buffer::<f32>(batch * nn).unwrap();
    let diag = rt.zero_buffer::<f32>(4 * batch).unwrap();
    let occ_w = rt.zero_buffer::<f32>(batch * n).unwrap();
    let mu = rt.zero_buffer::<f32>(batch).unwrap();
    let jn = if n & 1 == 1 { n + 1 } else { n };
    let rotlog = rt
        .zero_buffer::<f32>(batch * (jn - 1) * (jn / 2) * 4)
        .unwrap(); // prec=1 → double2 jlog2_t
    let all_ids: Vec<i32> = (0..batch as i32).collect();
    let wids_full = rt.buffer_from_slice(&all_ids).unwrap();
    let wg = 512usize.min(rt.caps().max_work_group_size);
    let prog = rt
        .build_program(&render_resident_source(wg, 1, false, true))
        .unwrap();

    eprintln!("[wsweep] n={n} batch={batch} wg={wg} | S | compact ms | masked ms | dead-WG ms");
    for &s in &[400usize, 300, 200, 100, 50, 25, 10, 1] {
        let mut act = vec![0i32; batch];
        for x in act.iter_mut().take(s) {
            *x = 1;
        }
        let active = rt.buffer_from_slice(&act).unwrap();
        let wids_s = rt.buffer_from_slice(&all_ids[..s]).unwrap();
        let mk_k = |gws: usize, wids: &ocl::Buffer<i32>| {
            Kernel::builder()
                .program(&prog)
                .name("jacobi_resident_batched")
                .queue(rt.queue().clone())
                .global_work_size(gws * wg)
                .local_work_size(wg)
                .arg(&a_buf)
                .arg(&v_buf)
                .arg(n as i32)
                .arg(batch as i32)
                .arg(0i32)
                .arg(&active)
                .arg(&diag)
                .arg(0i32)
                .arg(0i32)
                .arg(0.0f32)
                .arg(&occ_w)
                .arg(&mu)
                .arg(&rotlog)
                .arg_local::<f32>(n * (n + 1) / 2) // packed symmetric lA
                .arg_local::<f32>(1)
                .arg(wids)
                .build()
                .unwrap()
        };
        let k_compact = mk_k(s, &wids_s);
        let k_masked = mk_k(batch, &wids_full);
        let run = |rt: &mut GpuRuntime, k: &Kernel| -> f64 {
            rt.write_buffer(&a_buf, &a).unwrap();
            let t0 = Instant::now();
            unsafe { k.enq().unwrap() };
            rt.queue().finish().unwrap();
            t0.elapsed().as_secs_f64() * 1e3
        };
        // warmup then 3 timed reps, take min
        let mut tc = f64::MAX;
        let mut tm = f64::MAX;
        for rep in 0..4 {
            let c = run(&mut rt, &k_compact);
            let m = run(&mut rt, &k_masked);
            if rep > 0 {
                tc = tc.min(c);
                tm = tm.min(m);
            }
        }
        eprintln!("[wsweep] S={s:>4} | {tc:8.2} | {tm:8.2} | {:8.2}", tm - tc);
    }
}

/// 6. `active[]` inside a compact domain — a frozen slot in `work_ids`
/// must be skipped even though its WG launches.
#[test]
fn test_work_ids_active_mask_interplay() {
    let Some(mut rt) = try_runtime() else { return };
    let (n_atoms, batch) = (5usize, 4usize);
    let cfg = MatrixKernelConfig::nvidia_default();
    let prog = rt.build_program(&cfg.render_source()).unwrap();

    let mut q_next = vec![0f32; batch * n_atoms];
    for s in 0..batch {
        for a in 0..n_atoms {
            q_next[s * n_atoms + a] = 10.0 * s as f32 + a as f32;
        }
    }
    let q_buf = rt
        .buffer_from_slice(&vec![-999f32; batch * n_atoms])
        .unwrap();
    let qn_buf = rt.buffer_from_slice(&q_next).unwrap();
    let mut act = vec![1i32; batch];
    act[1] = 0; // frozen
    let active = rt.buffer_from_slice(&act).unwrap();
    let ids = [1i32, 2]; // slot 1 launches but is frozen; slot 2 runs
    let wids = rt.buffer_from_slice(&ids).unwrap();
    let wg = 64usize;

    let k = Kernel::builder()
        .program(&prog)
        .name("commit_q_batched")
        .queue(rt.queue().clone())
        .global_work_size(ids.len() * wg)
        .local_work_size(wg)
        .arg(n_atoms as i32)
        .arg(batch as i32)
        .arg(&qn_buf)
        .arg(&q_buf)
        .arg(&active)
        .arg(&wids)
        .build()
        .unwrap();
    unsafe { k.enq().unwrap() };
    let mut got = vec![0f32; batch * n_atoms];
    rt.read_buffer(&q_buf, &mut got).unwrap();

    for a in 0..n_atoms {
        assert_eq!(
            got[2 * n_atoms + a],
            q_next[2 * n_atoms + a],
            "commit slot 2 atom {a}"
        );
    }
    assert_untouched(&got, batch, n_atoms, &[2], -999.0, "commit_q+active");
}
