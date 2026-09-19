//! Persistent GPU DFTB engine — the production run loop.
//!
//! GPU analogue of `methods::dftb::dftb_cpu::DftbCpu`. One object owns the
//! OpenCL runtime, compiled programs, and all buffers. Tests and MD/FIRE must
//! drive this, not a throwaway `GpuRuntime::new()` per call.
//!
//! ```text
//! INIT (once)     GpuDftb::new  — runtime, kernels, SK pack, buffers, SCC plan
//! PER GEOMETRY    set_coords    — neighbors, assemble H0/S, G, X=S^{-1/2}
//! SCC             scc           — GpuSccPlan, warm-start q, no compile/alloc
//! ENERGY/FORCES   eval(want_forces) — one finalize; D and W on device (same kernel)
//! MD / FIRE       fire_step / md_step — refill pairs in place, loop (0.1 Å cap)
//! ```
//!
//! Homogeneous batch only: every replica shares species / N / n_atoms.
//! Changing topology or exceeding pair-buffer capacity fails loud (rebuild `new`).
//!
//! Limits: dense s/p DFTB, N from the template (full-local Jacobi N≤64, tiled N>64).
//! Do not use `cargo test` (dev profile) timings as GPU benchmarks — `--release`.

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::forces::{pack_repulsive_gpu, parse_all_repulsive};
use crate::methods::dftb::gamma::GammaTable;
use crate::methods::dftb::sk_data::SkData;
use crate::qmqm::fragment::{Fragment, FragmentTemplate};
use crate::qmqm::gpu_driver::{build_n_orb_per_atom, build_s_identity_init};
use crate::qmqm::gpu_prep::{GpuBatch, GpuPairBucket, GpuPairEntry, GpuSkTable};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use crate::qmqm::gpu_scc_plan::GpuSccPlan;
use ocl::prm::Float2;
use ocl::{Buffer, Kernel, Program};

const ANG2BOHR: f64 = 1.889_726_133;
const HAM_SOURCE: &str = include_str!("../methods/dftb/dftb_hamiltonian.cl");
const FORCE_SOURCE: &str = include_str!("gpu_forces.cl");
const PAIR_WG: usize = 64;
const SCC_MAX_ITER: usize = 100;

/// §12 D11: per-system convergence status — never conflate a numerical
/// plateau with convergence, and never treat "did not meet tolerance" as
/// converged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SccStatus {
    /// rms < tol at loop exit.
    Converged,
    /// rms ≥ tol but the iterate stopped moving (numerical plateau) —
    /// energy is still valid; charge residual did not certify.
    Plateau,
    /// Hit max_iter without convergence/plateau, or non-finite residual.
    Failed,
}

/// Result of one `scc` call.
#[derive(Debug, Clone)]
pub struct GpuDftbScc {
    pub n_iters: usize,
    pub rms: f32,
    pub stalled: bool,
    /// Per-system status (len = batch). For batch=1 this is the system status.
    pub statuses: Vec<SccStatus>,
}

/// One electronic finalize: energy always (f64), forces if `want_forces`.
pub struct GpuDftbEval {
    pub energy: Vec<f64>,
    pub forces: Option<Vec<f32>>,
    /// True RMS of `q_D − q_in` after finalize (max over batch).
    pub q_rms: f64,
    /// max |q_D − q_in| (max over batch).
    pub q_max: f64,
}

/// Persistent GPU DFTB engine (init once, then only update coordinates).
pub struct GpuDftb {
    pub rt: GpuRuntime,
    #[allow(dead_code)]
    sk: SkData,
    sk_dir: String,
    species: Vec<String>,
    tmpl: FragmentTemplate,
    #[allow(dead_code)]
    gamma_tbl: GammaTable,
    n: usize,
    n_atoms: usize,
    n_occ: usize,
    batch: usize,

    buf_h0: Buffer<f32>,
    buf_s: Buffer<f32>,
    buf_g: Buffer<f32>,
    buf_q0: Buffer<f32>,
    buf_oa: Buffer<i32>,
    buf_fragments: Buffer<crate::qmqm::gpu_prep::GpuFragment>,
    buf_atom_species: Buffer<i32>,
    buf_orb_off: Buffer<i32>,
    buf_n_orb: Buffer<i32>,
    buf_onsite: Buffer<Float2>,
    buf_hubbard: Buffer<f32>,
    buf_v_asm: Buffer<f32>,
    #[allow(dead_code)]
    buf_charges_zero: Buffer<f32>,
    buf_forces: Buffer<f32>,
    buf_pair_f: Buffer<f32>, // W9b: [3*total_pairs] per-pair forces (gather input)
    buf_pair_e: Buffer<f32>, // I10: [total_pairs] per-pair band energy
    buf_e_atom: Buffer<f32>, // I10: [total_atoms] band energy per atom
    buf_gather_ptr: Buffer<i32>, // W9b: [total_atoms+1] CSR row pointer
    buf_gather_list: Buffer<i32>, // W9b: [2*total_pairs] CSR (flat_idx<<1|is_j)
    k_gather: Kernel,        // W9b: force_gather_pairs (per-atom reduce)
    buf_edm: Buffer<f32>,
    buf_coords_ang: Buffer<f32>,
    buf_coords_bohr: Buffer<f32>,
    buf_u_hub: Buffer<f32>,
    buf_rep_off: Buffer<i32>,
    buf_rep_data: Buffer<f32>,
    /// §12 D8: species-pair γ/γ′ Hermite table (host f64 → f32 upload).
    /// Energy γ and force γ′ come from the same table — energy–force
    /// consistent, and the force kernel's f64 analytic island is gone.
    gamma_spl: crate::methods::dftb::gamma_spline::GammaSpline,
    buf_gamma_spl: Buffer<f32>,

    k_onsite: Kernel,
    onsite_done: bool, // W8: onsite diagonal is geometry-independent — enqueue once
    k_gamma_f: Kernel,
    k_gamma_build: Kernel,
    k_rep_force: Kernel,
    buckets: Vec<PairBucket>,

    q0: Vec<f32>,
    #[allow(dead_code)]
    u_per_atom: Vec<f64>,
    coords: Vec<[f64; 3]>,
    // ── D13: device-resident FIRE ── x lives in buf_coords_ang/_bohr (written
    // in place by fire_apply_batched); v, per-replica stat/ctl and the frozen
    // mask are device buffers. Host only makes the f64 adaptivity decisions.
    v_dev: Buffer<f32>,      // [batch*3n] velocities
    fire_stat: Buffer<f32>,  // [batch*4] {P=F·v, v², F², maxF(unfrozen)}
    fire_ctl: Buffer<f32>,   // [batch*4] {dt, alpha, mode, npos} — W6: device-owned
    park: Buffer<i32>,       // [batch] W6: 1=step, 0=parked (scc_ok mirror)
    frozen_dev: Buffer<i32>, // [n_atoms] freeze mask (same all replicas)
    constr_d: Buffer<f32>,   // [batch] distance-constraint targets (Å)
    k_fire_reduce: Kernel,
    k_fire_apply: Kernel,
    fire_stat_h: Vec<f32>, // [batch*4] persistent host scratch
    /// Distance constraint |x_j − x_i| = constr_d[b] per replica (None = off).
    constr: Option<(usize, usize)>,
    /// Device x is authoritative: assemble() skips the host coord upload.
    coords_on_device: bool,
    /// Device x newer than host `coords` — sync before host-side use.
    coords_dirty: bool,
    fire_v: Vec<[f64; 3]>,    // host mirror of v_dev (md_step only)
    md_f_prev: Vec<[f64; 3]>, // a(t) at current coords — VV half-kick state
    md_armed: bool,           // md_f_prev valid for self.coords
    // W10: persistent ones vec for check_jacobi's `ran` arg — the per-call
    // `vec![1i32; batch]` in eval/fire_step allocated every call.
    ones_batch: Vec<i32>,
    // D12→W6: per-replica FIRE state (dt/α/n_pos) is device-resident in
    // fire_ctl — the old host vecs were removed (reduce tail adapts them).
    /// Replicas whose last SCC Failed — `fire_step` parks them (zero v, no
    /// integrate) rather than stepping on garbage forces. All-true until
    /// first scc.
    scc_ok: Vec<bool>,
    /// Per-template-atom freeze mask (same for all replicas) — constrained
    /// relaxed scans pin e.g. the transferring proton. Frozen atoms get
    /// zero force, zero velocity, zero displacement, and are EXCLUDED from
    /// the replica's max|F| convergence test (their force is the constraint
    /// reaction force, not an optimizable one).
    frozen: Vec<bool>,
    /// Device electronic state (h_scc, dq, v, eig, cp, c, d, q_new) is
    /// consistent with q_gpu — i.e. `eval` may skip `finalize` (a redundant
    /// eigensolve). Set by scc_mix loop exit / finalize; cleared by
    /// set_coords and any write to q_gpu.
    state_fresh: bool,

    // Host scratch (per-geometry / per-force; never reallocated)
    scratch_ang: Vec<f32>,
    scratch_bohr: Vec<f32>,
    scratch_f: Vec<f32>,
    scratch_f0: Vec<f32>,
    scratch_q: Vec<f32>,
    scratch_qd: Vec<f32>,

    atom_sp: Vec<i32>,
    atom_n_orb: Vec<u8>,
    atom_orb_off: Vec<u16>,
    #[allow(dead_code)]
    pair_cut_sq: Vec<f64>,
    #[allow(dead_code)]
    sk_cut_sq: f64,
    #[allow(dead_code)]
    bucket_lut: Vec<i32>,
    nsp: usize,

    #[allow(dead_code)]
    n_species: i32,
    pub plan: GpuSccPlan,
    _ham: Program,
    _force: Program,
}

struct PairBucket {
    buf_pairs: Buffer<GpuPairEntry>,
    #[allow(dead_code)]
    buf_sk_h: Buffer<f32>,
    #[allow(dead_code)]
    buf_sk_s: Buffer<f32>,
    k_refresh: Kernel, // D7: in-place r,l,m,n from device coords
    k_assemble: Kernel,
    k_fused: Kernel, // W9a: force_pairs_fused (non-scc + scc-shift, one pass)
    dr: f32,
    n_grid: i32,
    n_sk_cols: i32,
    block_type: i32,
    sp_i: i32,
    sp_j: i32,
    /// Frozen pair count (all i<j for this block/species bucket × batch) —
    /// fixed at `new`; never grows (no cutoff membership changes).
    n_live: usize,
}

impl GpuDftb {
    /// Build the engine for a homogeneous batch. `coords.len() == batch * n_atoms`, Å.
    /// `sk_dir` is the Slater-Koster folder (repulsive Spline sections). Compile/alloc once.
    pub fn new(
        sk: SkData,
        sk_dir: &str,
        species: Vec<String>,
        coords: Vec<[f64; 3]>,
        batch: usize,
    ) -> Result<Self> {
        if batch == 0 {
            return Err(DftbError::InvalidInput("GpuDftb: batch=0".into()));
        }
        let n_atoms = species.len();
        if n_atoms == 0 {
            return Err(DftbError::InvalidInput("GpuDftb: no atoms".into()));
        }
        if coords.len() != batch * n_atoms {
            return Err(DftbError::InvalidInput(format!(
                "GpuDftb::new: coords.len()={} != batch*n_atoms {}*{} — homogeneous batch, one geometry after another",
                coords.len(), batch, n_atoms
            )));
        }
        let mut rt = GpuRuntime::new()?;
        rt.require_nvidia()?;

        let unique = unique_species(&species);
        let tables = parse_all_repulsive(sk_dir, &unique, unique.len())?;
        let (rep_off, rep_data, max_int) = pack_repulsive_gpu(&tables, unique.len(), &unique)?;
        let force_src = FORCE_SOURCE.replace(
            "#define REP_MAX_INTERVALS 30",
            &format!("#define REP_MAX_INTERVALS {}", max_int.max(1)),
        );

        let tmpl = FragmentTemplate::new(&sk, species.clone(), coords[..n_atoms].to_vec())?;
        let n = tmpl.n_orbs;
        let n_el = tmpl.q0.iter().sum::<f64>();
        let n_occ = (n_el / 2.0).round() as usize;
        if n_occ == 0 || n_occ > n {
            return Err(DftbError::InvalidInput(format!(
                "GpuDftb: n_occ={n_occ} invalid for N={n} n_el={n_el}"
            )));
        }
        if (n_el - 2.0 * n_occ as f64).abs() > 1e-6 {
            return Err(DftbError::InvalidInput(format!(
                "GpuDftb: odd or charged electron count n_el={n_el} (2*n_occ={}); closed-shell even only",
                2.0 * n_occ as f64
            )));
        }
        let gamma_tbl = GammaTable::from_sk_data(&sk, &species)?;
        let frags = fragments_from_coords(&tmpl, &coords, batch, n_atoms);
        let gpu_batch = GpuBatch::from_fragments(&frags, &sk, &gamma_tbl)?;
        if gpu_batch.n_frags != batch || gpu_batch.total_h_elements != batch * n * n {
            return Err(DftbError::InvalidInput(format!(
                "GpuDftb: batch pack mismatch n_frags={} total_h={} expected batch={} N²={}",
                gpu_batch.n_frags,
                gpu_batch.total_h_elements,
                batch,
                n * n
            )));
        }
        if gpu_batch.n_global_species != unique.len() {
            return Err(DftbError::InvalidInput(format!(
                "GpuDftb: global species {} != unique {} — repulsive pack order would not match atom_species",
                gpu_batch.n_global_species, unique.len()
            )));
        }

        let ham = rt.build_program(HAM_SOURCE)?;
        let force = rt.build_program(&force_src)?;

        let buf_h0 = rt.zero_buffer::<f32>(batch * n * n)?;
        let s_ident = build_s_identity_init(&gpu_batch.fragments, gpu_batch.total_h_elements);
        let buf_s = rt.buffer_from_slice(&s_ident)?;
        // D8: γ/γ′ spline table indexed by the SAME global species index the
        // kernels see (`atom_species` ↔ gpu_batch.hubbard_u ordering).
        let u_global: Vec<f64> = gpu_batch.hubbard_u.iter().map(|&u| u as f64).collect();
        let gamma_spl = crate::methods::dftb::gamma_spline::GammaSpline::new(
            &u_global,
            crate::methods::dftb::gamma_spline::GAMMA_SPLINE_NK,
            crate::methods::dftb::gamma_spline::GAMMA_SPLINE_RMAX,
        )?;
        let buf_gamma_spl = rt.buffer_from_slice(&gamma_spl.knots)?;
        // D7: G is built on-device by build_gamma_batched each geometry —
        // buf_g starts uninitialized, first assemble() fills it.
        let buf_g = rt.zero_buffer::<f32>(batch * n_atoms * n_atoms)?;
        let q0: Vec<f32> = tmpl
            .q0
            .iter()
            .map(|&q| q as f32)
            .collect::<Vec<_>>()
            .repeat(batch);
        let buf_q0 = rt.buffer_from_slice(&q0)?;
        // h_scc / Mulliken index orb_atom as [batch][N] (`oa + sid*n`). One copy is OOB for replica>0.
        let oa1 = orb_atom_map(&tmpl.atom_orb_off, n);
        if oa1.len() != n {
            return Err(DftbError::InvalidInput(format!(
                "GpuDftb: orb_atom len {} != N={n}",
                oa1.len()
            )));
        }
        let oa: Vec<i32> = oa1.repeat(batch);
        let buf_oa = rt.buffer_from_slice(&oa)?;
        let buf_fragments = rt.buffer_from_slice(&gpu_batch.fragments)?;
        let buf_atom_species = rt.buffer_from_slice(&gpu_batch.atom_species)?;
        let buf_orb_off = rt.buffer_from_slice(&gpu_batch.atom_orb_off)?;
        let n_orb = build_n_orb_per_atom(&gpu_batch.fragments, &gpu_batch.atom_orb_off);
        let buf_n_orb = rt.buffer_from_slice(&n_orb)?;
        let onsite: Vec<Float2> = (0..gpu_batch.n_global_species)
            .map(|i| {
                Float2::new(
                    gpu_batch.onsite_es_ep[2 * i],
                    gpu_batch.onsite_es_ep[2 * i + 1],
                )
            })
            .collect();
        let buf_onsite = rt.buffer_from_slice(&onsite)?;
        let buf_hubbard = rt.buffer_from_slice(&gpu_batch.hubbard_u)?;
        let buf_v_asm = rt.zero_buffer::<f32>(gpu_batch.total_atoms)?;
        let buf_charges_zero = rt.zero_buffer::<f32>(gpu_batch.charges.len())?;
        let buf_forces = rt.zero_buffer::<f32>(3 * gpu_batch.total_atoms)?;
        let buf_edm = rt.zero_buffer::<f32>(batch * n * n)?;
        let coords_ang: Vec<f32> = coords
            .iter()
            .flat_map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .collect();
        let buf_coords_ang = rt.buffer_from_slice(&coords_ang)?;
        let coords_bohr: Vec<f32> = coords
            .iter()
            .flat_map(|c| {
                [
                    (c[0] * ANG2BOHR) as f32,
                    (c[1] * ANG2BOHR) as f32,
                    (c[2] * ANG2BOHR) as f32,
                ]
            })
            .collect();
        let buf_coords_bohr = rt.buffer_from_slice(&coords_bohr)?;
        let u_per_atom = per_atom_u(&sk, &species)?;
        let u_hub: Vec<f32> = gpu_batch.hubbard_u.clone();
        let buf_u_hub = rt.buffer_from_slice(&u_hub)?;
        let buf_rep_off = rt.buffer_from_slice(&rep_off)?;
        let buf_rep_data = rt.buffer_from_slice(&rep_data)?;

        let total_atoms = gpu_batch.total_atoms;
        let n_frags = gpu_batch.n_frags as i32;
        let n_species = unique.len() as i32;
        // W6: 1 = may step / needs geometry work, 0 = parked (uncertified SCC
        // state). Mirrors scc_ok; declared early — gamma/assemble/force
        // kernels gate on it (W6b).
        let park = rt.buffer_from_slice(&vec![1i32; batch])?;
        let wg_onsite = 64.min(total_atoms.max(1));
        let g_onsite = ((total_atoms + wg_onsite - 1) / wg_onsite) * wg_onsite;
        let k_onsite = Kernel::builder()
            .program(&ham)
            .name("onsite_diagonal")
            .queue(rt.queue().clone())
            .global_work_size(g_onsite)
            .local_work_size(wg_onsite)
            .arg(&buf_fragments)
            .arg(&buf_atom_species)
            .arg(&buf_orb_off)
            .arg(&buf_n_orb)
            .arg(&buf_onsite)
            .arg(&buf_h0)
            .arg(n_frags)
            .arg(total_atoms as i32)
            .build()
            .map_err(map_ocl_err)?;
        let k_gamma_f = Kernel::builder()
            .program(&force)
            .name("force_gamma_deriv_batched")
            .queue(rt.queue().clone())
            .global_work_size(batch * 256)
            .local_work_size(256)
            .arg(n_atoms as i32)
            .arg(batch as i32)
            .arg(&buf_coords_ang)
            .arg(&buf_atom_species)
            .arg(&buf_v_asm)
            .arg(&buf_u_hub)
            .arg(n_species)
            .arg(&buf_gamma_spl)
            .arg(gamma_spl.nk as i32)
            .arg(gamma_spl.dr as f32)
            .arg(gamma_spl.r_max as f32)
            .arg(&buf_forces)
            .arg(&park) // W6b park gate
            .build()
            .map_err(map_ocl_err)?;
        // D7: device-resident G build — one thread per (system, upper-tri atom pair).
        let ntri = n_atoms * (n_atoms + 1) / 2;
        let g_gamma = ((batch * ntri + 255) / 256) * 256;
        let k_gamma_build = Kernel::builder()
            .program(&force)
            .name("build_gamma_batched")
            .queue(rt.queue().clone())
            .global_work_size(g_gamma)
            .local_work_size(256)
            .arg(n_atoms as i32)
            .arg(batch as i32)
            .arg(&buf_coords_bohr)
            .arg(&buf_atom_species)
            .arg(&buf_u_hub)
            .arg(n_species)
            .arg(&buf_gamma_spl)
            .arg(gamma_spl.nk as i32)
            .arg(gamma_spl.dr as f32)
            .arg(gamma_spl.r_max as f32)
            .arg(&buf_g)
            .arg(&park) // W6b park gate
            .build()
            .map_err(map_ocl_err)?;
        let k_rep_force = Kernel::builder()
            .program(&force)
            .name("force_repulsive_batched")
            .queue(rt.queue().clone())
            .global_work_size(batch * 256)
            .local_work_size(256)
            .arg(n_atoms as i32)
            .arg(batch as i32)
            .arg(&buf_coords_bohr)
            .arg(&buf_atom_species)
            .arg(&buf_rep_off)
            .arg(n_species)
            .arg(&buf_rep_data)
            .arg(&buf_forces)
            .arg(&park) // W6b park gate
            .build()
            .map_err(map_ocl_err)?;

        // ── D13: device-resident FIRE — all state allocated once here. ──
        let v_dev = rt.zero_buffer::<f32>(3 * batch * n_atoms)?;
        let fire_stat = rt.zero_buffer::<f32>(4 * batch)?;
        // W6: ctl {dt, alpha, mode, npos} is device-owned — initialized once
        // (dt=1.0, alpha=0.1, normal mode) and adapted in the reduce tail.
        let mut ctl0 = vec![0.0f32; 4 * batch];
        for b in 0..batch {
            ctl0[4 * b] = 1.0;
            ctl0[4 * b + 1] = 0.1;
        }
        let fire_ctl = rt.buffer_from_slice(&ctl0)?;
        let frozen_dev = rt.zero_buffer::<i32>(n_atoms)?;
        let constr_d = rt.zero_buffer::<f32>(batch)?;
        let k_fire_reduce = Kernel::builder()
            .program(&force)
            .name("fire_reduce_batched")
            .queue(rt.queue().clone())
            .global_work_size(batch * 256)
            .local_work_size(256)
            .arg(n_atoms as i32)
            .arg(&buf_forces)
            .arg(&v_dev)
            .arg(&buf_coords_ang)
            .arg(&frozen_dev)
            .arg(0i32)
            .arg(0i32)
            .arg(0i32) // c_on, ci, cj — set_constraint
            .arg(&fire_stat)
            .arg(&fire_ctl)
            .arg(&park)
            .arg(1e-3f32) // f_tol — set per fire_step
            .build()
            .map_err(map_ocl_err)?;
        let k_fire_apply = Kernel::builder()
            .program(&force)
            .name("fire_apply_batched")
            .queue(rt.queue().clone())
            .global_work_size(batch * 256)
            .local_work_size(256)
            .arg(n_atoms as i32)
            .arg(&v_dev)
            .arg(&buf_forces)
            .arg(&buf_coords_ang)
            .arg(&buf_coords_bohr)
            .arg(&frozen_dev)
            .arg(&fire_ctl)
            .arg(&fire_stat)
            .arg(0i32)
            .arg(0i32)
            .arg(0i32) // c_on, ci, cj — set by set_constraint
            .arg(&constr_d)
            .arg(2.0f32)
            .arg(0.1f32) // vmax, dmax (disp cap Å)
            .build()
            .map_err(map_ocl_err)?;

        let atom_sp: Vec<i32> = gpu_batch.atom_species[..n_atoms].to_vec();
        let mut pair_buckets = gpu_batch.pair_buckets.clone();
        ensure_template_pair_buckets(
            &mut pair_buckets,
            &tmpl,
            &atom_sp,
            &gpu_batch.sk_tables,
            unique.len(),
        )?;
        let (pair_cut_sq, sk_cut_sq) = pair_cutoffs(&sk, unique.len(), &unique)?;

        // ── D7: frozen per-template pair lists ──────────────────────────
        // Static fields (replica, atoms, orb offsets, species, block type)
        // are built ONCE here for ALL i<j pairs — no cutoff membership
        // filtering (pairs beyond the SK cutoff evaluate to ~0 anyway via
        // the clamped table tail; they cost a few flops, not correctness).
        // r/l/m/n are refreshed on-device by `refresh_pair_geom` each
        // geometry — no per-set_coords host pair rebuild or upload.
        let nsp = unique.len();
        let mut bucket_lut = vec![-1i32; 3 * nsp * nsp];
        for (i, bkt) in pair_buckets.iter().enumerate() {
            let skt = &gpu_batch.sk_tables[bkt.sk_table_idx];
            let k = bkt.block_type as usize * nsp * nsp
                + skt.species_i as usize * nsp
                + skt.species_j as usize;
            if k >= bucket_lut.len() {
                return Err(DftbError::InvalidInput(format!(
                    "GpuDftb: bucket lut index {k} >= {}",
                    bucket_lut.len()
                )));
            }
            bucket_lut[k] = i as i32;
        }
        let atom_n_orb = tmpl.atom_n_orb.clone();
        let atom_orb_off = tmpl.atom_orb_off.clone();
        // template pair → per-bucket static list
        let mut lists: Vec<Vec<GpuPairEntry>> = vec![Vec::new(); pair_buckets.len()];
        for i in 0..n_atoms {
            for j in (i + 1)..n_atoms {
                let n_orb_i = atom_n_orb[i] as usize;
                let n_orb_j = atom_n_orb[j] as usize;
                let Some((bt, ai, aj, oi, oj, l, m, nv, si, sj)) = orient_pair(
                    i,
                    j,
                    n_orb_i,
                    n_orb_j,
                    atom_orb_off[i],
                    atom_orb_off[j],
                    atom_sp[i],
                    atom_sp[j],
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ) else {
                    return Err(DftbError::InvalidInput(format!(
                        "GpuDftb::new: unsupported block type for pair {i}-{j} (n_orb {n_orb_i}x{n_orb_j})"
                    )));
                };
                let key = bt as usize * nsp * nsp + si as usize * nsp + sj as usize;
                let bi = bucket_lut[key];
                if bi < 0 {
                    return Err(DftbError::InvalidInput(format!(
                        "GpuDftb::new: pair {i}-{j} block={bt} species {si}-{sj} has no SK bucket — missing SK table"
                    )));
                }
                for b in 0..batch {
                    lists[bi as usize].push(GpuPairEntry {
                        replica: b as u32,
                        atom_i: ai,
                        atom_j: aj,
                        orb_i: oi,
                        orb_j: oj,
                        r: 1.0,
                        l,
                        m,
                        n: nv,
                    });
                }
            }
        }

        // W9b: flat per-pair force buffer (gather kernel owns per-atom reduce).
        let total_pairs: usize = lists.iter().map(|l| l.len()).sum();
        let buf_pair_f = rt.zero_buffer::<f32>(3 * total_pairs.max(1))?;
        // I10: per-pair band energy + per-atom gather output (Tr(D·H0)).
        let buf_pair_e = rt.zero_buffer::<f32>(total_pairs.max(1))?;
        let buf_e_atom = rt.zero_buffer::<f32>(total_atoms.max(1))?;
        let orb_rng: Vec<i32> = (0..total_atoms)
            .flat_map(|a| {
                let la = a % n_atoms;
                [atom_orb_off[la] as i32, n_orb[la]]
            })
            .collect();
        let buf_orb_rng = rt.buffer_from_slice(&orb_rng)?;
        let mut buckets = Vec::with_capacity(pair_buckets.len());
        // W9b: flat pair-force buffer + atom→pair CSR (entry = flat_idx<<1|is_j).
        let mut gather_lists: Vec<Vec<i32>> = vec![Vec::new(); total_atoms];
        let mut pair_base = 0i32;
        for (bi, bkt) in pair_buckets.iter().enumerate() {
            let skt = &gpu_batch.sk_tables[bkt.sk_table_idx];
            let staging = std::mem::take(&mut lists[bi]);
            let n_pairs = staging.len();
            for (pi, p) in staging.iter().enumerate() {
                let flat = pair_base + pi as i32;
                gather_lists[p.replica as usize * n_atoms + p.atom_i as usize].push(flat << 1);
                gather_lists[p.replica as usize * n_atoms + p.atom_j as usize]
                    .push((flat << 1) | 1);
            }
            let buf_pairs = rt.buffer_from_slice(&staging)?;
            let buf_sk_h = rt.buffer_from_slice(&skt.sk_h)?;
            let buf_sk_s = rt.buffer_from_slice(&skt.sk_s)?;
            let gws = ((n_pairs.max(1) + PAIR_WG - 1) / PAIR_WG) * PAIR_WG;
            let k_refresh = Kernel::builder()
                .program(&force)
                .name("refresh_pair_geom")
                .queue(rt.queue().clone())
                .global_work_size(gws)
                .local_work_size(PAIR_WG)
                .arg(&buf_pairs)
                .arg(&buf_coords_bohr)
                .arg(n_pairs as i32)
                .arg(n_atoms as i32)
                .build()
                .map_err(map_ocl_err)?;
            let k_assemble = Kernel::builder()
                .program(&ham)
                .name("assemble_pairs_geom")
                .queue(rt.queue().clone())
                .global_work_size(gws)
                .local_work_size(PAIR_WG)
                .arg(&buf_pairs)
                .arg(&buf_fragments)
                .arg(&buf_sk_h)
                .arg(&buf_sk_s)
                .arg(&buf_v_asm)
                .arg(&buf_h0)
                .arg(&buf_s)
                .arg(skt.dr)
                .arg(skt.n_grid as i32)
                .arg(n_pairs as i32)
                .arg(n_frags)
                .arg(bkt.block_type as i32)
                .arg(skt.n_sk_cols as i32)
                .arg(&buf_coords_bohr)
                .arg(n_atoms as i32)
                .arg(&park) // W8+W6b
                .build()
                .map_err(map_ocl_err)?;
            // W9a/b: fused pair force — one SK interp + one dS/dR serves the
            // non-scc AND scc-shift terms; dm/edm/v_shift bound per call.
            // W9b: writes per-pair forces to buf_pair_f (no atomics); the
            // gather kernel reduces them per atom.
            let k_fused = Kernel::builder()
                .program(&force)
                .name("force_pairs_fused")
                .queue(rt.queue().clone())
                .global_work_size(gws)
                .local_work_size(PAIR_WG)
                .arg(&buf_pairs)
                .arg(&buf_fragments)
                .arg(&buf_sk_h)
                .arg(&buf_sk_s)
                .arg(&buf_h0)
                .arg(&buf_edm)
                .arg(&buf_v_asm)
                .arg(&buf_pair_f)
                .arg(&buf_pair_e)
                .arg(pair_base)
                .arg(skt.dr)
                .arg(skt.n_grid as i32)
                .arg(n_pairs as i32)
                .arg(n_frags)
                .arg(bkt.block_type as i32)
                .arg(skt.n_sk_cols as i32)
                .arg(&buf_coords_bohr)
                .arg(n_atoms as i32)
                .arg(&park) // W8+W6b
                .build()
                .map_err(map_ocl_err)?;
            pair_base += n_pairs as i32;
            buckets.push(PairBucket {
                buf_pairs,
                buf_sk_h,
                buf_sk_s,
                k_refresh,
                k_assemble,
                k_fused,
                dr: skt.dr,
                n_grid: skt.n_grid as i32,
                n_sk_cols: skt.n_sk_cols as i32,
                block_type: bkt.block_type as i32,
                sp_i: skt.species_i as i32,
                sp_j: skt.species_j as i32,
                n_live: n_pairs,
            });
        }
        // W9b gather CSR: ptr[total_atoms+1], list[2*total_pairs] (idx<<1|is_j).
        let mut gather_ptr = vec![0i32; total_atoms + 1];
        let mut gather_flat = Vec::with_capacity(2 * total_pairs);
        for (a, l) in gather_lists.iter().enumerate() {
            gather_flat.extend(l.iter().copied());
            gather_ptr[a + 1] = gather_flat.len() as i32;
        }
        let buf_gather_ptr = rt.buffer_from_slice(&gather_ptr)?;
        let buf_gather_list = rt.buffer_from_slice(&gather_flat)?;
        let g_gather = ((total_atoms + 255) / 256) * 256;
        let k_gather = Kernel::builder()
            .program(&force)
            .name("force_gather_pairs")
            .queue(rt.queue().clone())
            .global_work_size(g_gather)
            .local_work_size(256)
            .arg(&buf_gather_ptr)
            .arg(&buf_gather_list)
            .arg(&buf_pair_f)
            .arg(&buf_pair_e)
            .arg(&buf_forces)
            .arg(&buf_e_atom)
            .arg(&buf_h0)
            .arg(&buf_h0)
            .arg(&buf_orb_rng) // dm bound per call (arg 6)
            .arg(n_atoms as i32)
            .arg(n as i32)
            .arg(total_atoms as i32)
            .build()
            .map_err(map_ocl_err)?;

        let plan = GpuSccPlan::new(
            &mut rt, &buf_s, &buf_h0, &buf_g, &buf_q0, &buf_oa, n, n_atoms, batch,
        )
        .map_err(|e| DftbError::InvalidInput(format!("GpuSccPlan::new (Lowdin X from S): {e}")))?;
        let mut eng = Self {
            rt,
            sk,
            sk_dir: sk_dir.to_string(),
            species,
            tmpl,
            gamma_tbl,
            n,
            n_atoms,
            n_occ,
            batch,
            buf_h0,
            buf_s,
            buf_g,
            buf_q0,
            buf_oa,
            buf_fragments,
            buf_atom_species,
            buf_orb_off,
            buf_n_orb,
            buf_onsite,
            buf_hubbard,
            buf_v_asm,
            buf_charges_zero,
            buf_forces,
            buf_pair_f,
            buf_pair_e,
            buf_e_atom,
            buf_gather_ptr,
            buf_gather_list,
            k_gather,
            buf_edm,
            buf_coords_ang,
            buf_coords_bohr,
            buf_u_hub,
            buf_rep_off,
            buf_rep_data,
            gamma_spl,
            buf_gamma_spl,
            k_onsite,
            onsite_done: false,
            k_gamma_f,
            k_gamma_build,
            k_rep_force,
            buckets,
            q0,
            u_per_atom,
            coords,
            v_dev,
            fire_stat,
            fire_ctl,
            park,
            frozen_dev,
            constr_d,
            k_fire_reduce,
            k_fire_apply,
            fire_stat_h: vec![0.0; 4 * batch],
            constr: None,
            coords_on_device: false,
            coords_dirty: false,
            fire_v: vec![[0.0; 3]; batch * n_atoms],
            md_f_prev: vec![[0.0; 3]; batch * n_atoms],
            md_armed: false,
            ones_batch: vec![1i32; batch],
            scc_ok: vec![true; batch],
            frozen: vec![false; n_atoms],
            state_fresh: false,
            scratch_ang: coords_ang,
            scratch_bohr: coords_bohr.clone(),
            scratch_f: vec![0.0; 3 * batch * n_atoms],
            scratch_f0: vec![0.0; 3 * batch * n_atoms],
            scratch_q: vec![0.0; batch * n_atoms],
            scratch_qd: vec![0.0; batch * n_atoms],
            atom_sp,
            atom_n_orb,
            atom_orb_off,
            pair_cut_sq,
            sk_cut_sq,
            bucket_lut,
            nsp,
            n_species,
            plan,
            _ham: ham,
            _force: force,
        };
        eng.plan
            .set_repulsive_splines(
                &mut eng.rt,
                &coords_bohr,
                &gpu_batch.atom_species,
                &rep_off,
                &rep_data,
                unique.len(),
                max_int,
            )
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb set_repulsive_splines: {e}")))?;
        // D13: repulsive energy reads the shared coords_bohr buffer — no
        // per-geometry rep_coords upload (set_repulsive_coords retired).
        eng.plan
            .bind_rep_energy_coords(&eng.buf_coords_bohr)
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb bind_rep_energy_coords: {e}")))?;
        eng.assemble()
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb initial assemble: {e}")))?;
        eng.plan
            .set_geometry(&mut eng.rt, &eng.buf_s)
            .map_err(|e| {
                DftbError::InvalidInput(format!("GpuDftb set_geometry after assemble: {e}"))
            })?;
        eng.plan
            .reset_diis(&eng.rt)
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb reset_diis: {e}")))?;
        eng.plan
            .set_initial_charges(&eng.rt, &eng.q0)
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb set_initial_charges: {e}")))?;
        Ok(eng)
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }
    pub fn n_occ(&self) -> usize {
        self.n_occ
    }
    pub fn batch(&self) -> usize {
        self.batch
    }
    /// Host copy of positions — after device-FIRE steps call
    /// `sync_coords_to_host()` first (this getter is &self and cannot sync).
    pub fn coords(&self) -> &[[f64; 3]] {
        &self.coords
    }

    /// I10 diagnostic: per-atom band-energy decomposition written by the
    /// force gather (onsite + ½ per incident pair); Σ_a = Tr(D·H0) = e_band.
    /// Valid after `eval(true)`/`forces` — the pair kernel only runs on the
    /// force path. Diagnostic readback — not a hot-loop call.
    pub fn read_e_atom(&mut self) -> Result<Vec<f32>> {
        let mut v = vec![0.0f32; self.n_atoms * self.batch];
        self.rt.read_buffer(&self.buf_e_atom, &mut v)?;
        Ok(v)
    }

    /// Per-geometry update (D7): uploads coords, then everything else is
    /// device-side — pair r,l,m,n refresh, G build, H0/S assembly. The pair
    /// SET is frozen at `new` (all i<j, per template); the SK tail is ~0 for
    /// pairs beyond cutoff, so far pairs contribute exactly zero.
    pub fn set_coords(&mut self, coords: &[[f64; 3]]) -> Result<()> {
        if coords.len() != self.batch * self.n_atoms {
            return Err(DftbError::InvalidInput(format!(
                "set_coords: len {} != batch*n_atoms {}*{}",
                coords.len(),
                self.batch,
                self.n_atoms
            )));
        }
        self.coords.copy_from_slice(coords);
        self.coords_on_device = false; // host path uploads coords
        self.coords_dirty = false;
        self.md_armed = false; // VV acceleration state is stale
                               // W6b: a host geometry change invalidates the park mask — a parked
                               // replica's stale H/S MUST be rebuilt for the new coordinates (the
                               // park gate skips its pair work otherwise). Re-certification happens
                               // in the next scc() as usual.
        self.scc_ok = vec![true; self.batch];
        self.sync_park()?;
        self.assemble()?;
        self.plan.set_geometry(&mut self.rt, &self.buf_s)?;
        self.plan.reset_diis(&self.rt)?;
        self.state_fresh = false;
        Ok(())
    }

    /// Host copy of positions may lag the device after device-FIRE steps —
    /// sync before any host-side use (cpu_ref, measure, coords()).
    pub fn sync_coords_to_host(&mut self) -> Result<()> {
        if self.coords_dirty {
            self.rt
                .read_buffer(&self.buf_coords_ang, &mut self.scratch_ang)?;
            for i in 0..self.coords.len() {
                for c in 0..3 {
                    self.coords[i][c] = self.scratch_ang[3 * i + c] as f64;
                }
            }
            self.coords_dirty = false;
        }
        Ok(())
    }

    // DEPRECATED by D7 — kept for reference. Host pair rebuild + upload per
    // geometry; replaced by frozen pair lists + refresh_pair_geom kernel.
    // fn refill_pairs(&mut self) -> Result<()> { ... }

    /// Device-resident geometry assembly (D7). No host pair loops, no G
    /// upload, no finish() — the in-order queue orders the kernels.
    /// H0/S need no zeroing: every off-diagonal element is written by a
    /// pair block (frozen list covers all i<j), the diagonal by
    /// onsite_diagonal / the persistent S=I. V_asm stays zero (zeroed at
    /// `new`; the SCC shift path uses plan buffers, not this one).
    fn assemble(&mut self) -> Result<()> {
        if !self.coords_on_device {
            fill_coord_scratch(&self.coords, &mut self.scratch_ang, &mut self.scratch_bohr);
            self.rt
                .write_buffer(&self.buf_coords_ang, &self.scratch_ang)
                .map_err(|e| DftbError::InvalidInput(format!("write coords_ang: {e}")))?;
            self.rt
                .write_buffer(&self.buf_coords_bohr, &self.scratch_bohr)
                .map_err(|e| DftbError::InvalidInput(format!("write coords_bohr: {e}")))?;
        }
        unsafe {
            self.k_gamma_build
                .enq()
                .map_err(|e| DftbError::InvalidInput(format!("build_gamma: {e}")))?;
        }
        // W8: refresh_pair_geom retired — assemble_pairs/force_pairs_fused
        // compute r,l,m,n from buf_coords_bohr directly (park-gated, W6b).
        // The kernel stays for the GpuForceDriver/test path.
        // for (i, slot) in self.buckets.iter().enumerate() {
        //     if slot.n_live == 0 { continue; }
        //     unsafe { slot.k_refresh.enq().map_err(|e| DftbError::InvalidInput(format!("refresh_pairs bucket {i}: {e}")))?; }
        // }
        // W8: onsite diagonal is geometry-independent — run once at init.
        if !self.onsite_done {
            unsafe {
                self.k_onsite
                    .enq()
                    .map_err(|e| DftbError::InvalidInput(format!("onsite_diagonal: {e}")))?;
            }
            self.onsite_done = true;
        }
        for (i, slot) in self.buckets.iter().enumerate() {
            if slot.n_live == 0 {
                continue;
            }
            unsafe {
                slot.k_assemble.enq().map_err(|e| {
                    DftbError::InvalidInput(format!("assemble_pairs bucket {i}: {e}"))
                })?;
            }
        }
        // repulsive energy kernel bound to buf_coords_bohr at new() — no upload.
        Ok(())
    }

    /// SCC mixer for Package 2 A/B: 0 = GPU DIIS (production), 1 = GPU simple mix, 2 = host f64 DIIS (same electronic kernels).
    /// Production SCC: GPU DIIS, and on any Failed replica ONE retry where
    /// the failed replicas are warm-started from the NEAREST converged
    /// replica's charges (adiabatic continuation — at scan geometries like
    /// mid proton transfer, bare-q0 DIIS oscillates between near-degenerate
    /// charge states; a converged neighbor's q usually lands it in the right
    /// basin on the first iteration).
    ///
    /// R2: the retry re-solves ONLY the failed replicas (initial `active`
    /// mask = failed set). Non-failed replicas are never iterated, so their
    /// device state is untouched and keeping their run-1 status is coherent
    /// — the label matches the buffers. The old code re-ran the whole batch
    /// and merged the better status label WITHOUT restoring the matching
    /// physical state: a replica could be reported Converged while its
    /// C/D/q buffers held the failed retry state.
    /// Honest single-shot SCC: runs `scc_mix` once and returns whatever
    /// statuses it produced — including `Failed`. W11/I4: NO implicit
    /// failed→nearest-replica reseed here; recovery is an explicit driver
    /// call to `scc_retry_failed` so call sites state the fallback openly.
    pub fn scc(&mut self, max_iter: usize, rms_tol: f32) -> Result<GpuDftbScc> {
        self.scc_mix(max_iter, rms_tol, 0)
    }

    /// Explicit failed-replica retry: copies converged charges from each
    /// failed replica's nearest converged neighbor and re-solves ONLY the
    /// failed set (masked). Converged replicas' device state is untouched;
    /// their run-1 status is kept. Returns the merged result.
    /// Call this from drivers that explicitly want warm-start recovery —
    /// e.g. scans, where an adjacent geometry IS the right seed.
    pub fn scc_retry_failed(
        &mut self,
        s: &GpuDftbScc,
        max_iter: usize,
        rms_tol: f32,
    ) -> Result<GpuDftbScc> {
        let n_fail = s
            .statuses
            .iter()
            .filter(|st| **st == SccStatus::Failed)
            .count();
        if n_fail == 0 {
            return Ok(s.clone());
        }
        // snapshot converged charges (q_gpu = last iterate per replica)
        let na = self.n_atoms;
        let mut q = vec![0.0f32; self.batch * na];
        self.rt.read_buffer(&self.plan.q_gpu, &mut q)?;
        let ok: Vec<usize> = (0..self.batch)
            .filter(|&b| s.statuses[b] != SccStatus::Failed)
            .collect();
        if ok.is_empty() {
            return Ok(s.clone());
        } // nothing to seed from — keep the honest failures
        let mut retry = vec![false; self.batch];
        for b in 0..self.batch {
            if s.statuses[b] != SccStatus::Failed {
                continue;
            }
            let src = *ok.iter().min_by_key(|&&c| c.abs_diff(b)).unwrap();
            let (dst_lo, src_lo) = (b * na, src * na);
            q.copy_within(src_lo..src_lo + na, dst_lo);
            retry[b] = true;
            eprintln!(
                "[GpuDftb] SCC warm-start replica {b} ← replica {src} (q copied, DIIS reset)"
            );
        }
        self.plan.reset_diis(&self.rt)?;
        self.plan.set_initial_charges(&self.rt, &q)?;
        let s2 = self.scc_mix_inner(max_iter, rms_tol, 0, Some(&retry))?;
        // Merge: retried replicas take their run-2 outcome; everyone else
        // keeps run-1 status AND state (untouched by the masked retry).
        let mut merged = s2;
        for b in 0..self.batch {
            if !retry[b] {
                merged.statuses[b] = s.statuses[b].clone();
            }
        }
        merged.n_iters += s.n_iters; // total expended iterations
        merged.stalled = merged.statuses.iter().any(|st| *st != SccStatus::Converged);
        let still = merged
            .statuses
            .iter()
            .filter(|st| **st == SccStatus::Failed)
            .count();
        eprintln!("[GpuDftb] SCC warm-start retry: {n_fail} failed → {still} still failed");
        self.scc_ok = merged
            .statuses
            .iter()
            .map(|st| *st != SccStatus::Failed)
            .collect();
        self.plan.set_state_ok(&self.rt, &self.scc_ok)?; // R1: W-build mask follows certified status
        self.sync_park()?; // W6: device FIRE park mask
        Ok(merged)
    }

    /// Production semantic: `scc` + EXPLICIT failed→nearest-converged retry.
    /// Named so call sites openly state the fallback (W11/I4). Use `scc`
    /// for honest one-shot behavior.
    pub fn scc_with_retry(&mut self, max_iter: usize, rms_tol: f32) -> Result<GpuDftbScc> {
        let s = self.scc(max_iter, rms_tol)?;
        self.scc_retry_failed(&s, max_iter, rms_tol)
    }

    /// Test-only R2 hook: run the masked-retry SCC path directly —
    /// replicas with `retry[b]` are re-solved; others are never iterated
    /// and keep their device state/status. Production calls go through
    /// `scc`, which builds this mask from the Failed set.
    #[doc(hidden)]
    pub fn scc_masked(
        &mut self,
        max_iter: usize,
        rms_tol: f32,
        retry: &[bool],
    ) -> Result<GpuDftbScc> {
        self.scc_mix_inner(max_iter, rms_tol, 0, Some(retry))
    }

    pub fn reset_q0(&mut self) -> Result<()> {
        self.plan.reset_diis(&self.rt)?;
        self.plan.reset_purify(); // §17.6: fresh solve → cold purify start
        self.state_fresh = false;
        self.plan.set_initial_charges(&self.rt, &self.q0)
    }

    /// `mix`: 0 GPU DIIS, 1 GPU simple α=0.3, 2 host f64 DiisMixer (hist=min(10,n_atoms), warmup=1).
    pub fn scc_mix(&mut self, max_iter: usize, rms_tol: f32, mix: i32) -> Result<GpuDftbScc> {
        self.scc_mix_inner(max_iter, rms_tol, mix, None)
    }

    /// `retry: Some(mask)` starts the loop with `active = mask` — replicas
    /// outside the mask are never iterated and keep their device state
    /// (R2 masked retry). `None` = all replicas active.
    fn scc_mix_inner(
        &mut self,
        max_iter: usize,
        rms_tol: f32,
        mix: i32,
        retry: Option<&[bool]>,
    ) -> Result<GpuDftbScc> {
        if mix == 2 && self.batch != 1 {
            return Err(DftbError::InvalidInput(format!(
                "scc_mix host DIIS: batch={} — measurement path is replica-0 only",
                self.batch
            )));
        }
        // D11-fix: PER-REPLICA convergence tracking. The old detector tested
        // the batch-MAX rms — one stagnating replica broke the loop for
        // everyone, and "plateau" was declared for ANY stagnant residual
        // (rms=0.2 counted as plateau — that's divergence, not a floor).
        // Plateau = iterate stopped moving AND residual near the f32 floor.
        // Plateau acceptance floor: stagnant replicas below max(3·tol, 2e-5)
        // are reported Plateau, not Failed (measured f32 floor ~1e-6..1e-5).
        let plateau_floor = (3.0 * rms_tol).max(2e-5f32);
        let mut rms = f32::INFINITY;
        let mut n_iters = 0;
        // per-replica ring of last-10 rms + done flags
        let mut hist = vec![[f32::INFINITY; 10]; self.batch];
        let mut stagnant = vec![false; self.batch];
        let mut done = vec![false; self.batch];
        let mut n_active = self.batch;
        let mut host = if mix == 2 {
            let mut m = crate::qmqm::mixer::DiisMixer::new(self.n_atoms.min(10), self.n_atoms);
            m.alpha = 0.3;
            m.warmup = 1;
            Some(m)
        } else {
            None
        };
        let mut q_in = vec![0.0f64; self.n_atoms];
        let mut q_out = vec![0.0f64; self.n_atoms];
        let mut res = vec![0.0f64; self.n_atoms];
        let mut q_f32 = vec![0.0f32; self.n_atoms];
        let cap = max_iter;
        eprintln!("[GpuDftb] scc_mix mix={mix} (0=GPU DIIS hist={}, 1=GPU simple, 2=host f64 DIIS hist={}) max_iter={max_iter} rms_tol={rms_tol:.3e}", self.plan.diis_max_hist, self.n_atoms.min(10));
        // Commit-model SCC: mixers write q_next; q_gpu is only advanced for
        // replicas still active. On exit the device state (C,D,H_scc,Δq,V)
        // corresponds to q_gpu exactly → eval() can skip re-diagonalizing.
        self.state_fresh = false;
        match retry {
            Some(m) => {
                if m.len() != self.batch {
                    return Err(DftbError::InvalidInput(format!(
                        "scc_mix_inner: retry mask len {} != batch {}",
                        m.len(),
                        self.batch
                    )));
                }
                for (f, &r) in self.plan.active_host.iter_mut().zip(m.iter()) {
                    *f = r as i32;
                }
            }
            None => {
                for f in self.plan.active_host.iter_mut() {
                    *f = 1;
                }
            }
        }
        self.plan.set_active(&self.rt)?;
        // Replicas outside the retry mask are already done: never iterated,
        // never committed — their device state stays the converged run-1 one.
        // W1: snapshot which replicas will launch H-Jacobi this solve — the
        // deferred check_jacobi at the end must not read stale diag records
        // of replicas that never ran.
        let jacobi_ran: Vec<i32> = self.plan.active_host.clone();
        // T01: this solve is one certification window — clear the
        // first-failure latch so a previous window's record can't leak in.
        self.plan.clear_jacobi_diag(&self.rt)?;
        for (b, &a) in self.plan.active_host.iter().enumerate() {
            if a == 0 {
                done[b] = true;
                n_active -= 1;
            }
        }
        if mix == 0 {
            // W4 (manifest §14): chunked SCC. CHUNK iterations are enqueued
            // back-to-back with NO host readback — diis_step_batched clears
            // active[sid] itself on rms<tol or nonfinite, and every
            // per-iteration kernel (incl. batched_gemm_active) gates on the
            // device mask. The host syncs once per chunk to read rms+active
            // for bookkeeping (status, stagnation). Converged replicas stop
            // paying for any further work inside the chunk.
            // R1: all kernel args bound — io bufs at construction, scalars here.
            self.plan.bind_solve_params(self.n_occ)?;
            self.plan.bind_mix_params(0.3, rms_tol)?;
            const CHUNK: usize = 8;
            let mut it = 0usize;
            let mut samples = vec![0usize; self.batch];
            // T06 Phase B: compact launch domains — at each chunk-end sync
            // the work list is rebuilt from active_host so converged/failed
            // replicas stop occupying workgroups in the NEXT chunk.
            // RUST_DFTB_SCC_COMPACT=0 keeps full-domain launches for A/B.
            let compact = std::env::var("RUST_DFTB_SCC_COMPACT").ok().as_deref() != Some("0");
            let mut work_list: Vec<i32> = Vec::with_capacity(self.batch);
            // Seed the compact domain up front when the initial mask is
            // already sparse (retry-mask re-solves run only the flagged
            // subset — a full-domain first chunk would be dead workgroups).
            if compact && n_active > 0 && n_active < self.batch {
                work_list.extend(
                    (0..self.batch)
                        .filter(|&b| self.plan.active_host[b] != 0)
                        .map(|b| b as i32),
                );
                self.plan.set_work_domain(&self.rt, &work_list)?;
            }
            // Rebuild only when the active count dropped — the active set
            // shrinks monotonically, so an unchanged count means the domain
            // is already correct (keeps compaction free when there is no
            // convergence tail).
            let mut prev_n_active = n_active;
            while it < cap && n_active > 0 {
                let step = (cap - it).min(CHUNK);
                for _ in 0..step {
                    self.plan.scc_step_diis_enq(&mut self.rt)?;
                    // R7: no commit launch — diis_step_batched writes the
                    // mixed iterate straight into q_gpu for active replicas.
                }
                it += step;
                n_iters = it;
                self.plan.read_chunk_status(&self.rt)?; // one finish covers both reads
                self.rt.prof_tick("scc.chunkend");
                self.state_fresh = false; // commits advanced q_gpu past the solved state
                let mut host_dirty = false;
                for b in 0..self.batch {
                    if done[b] {
                        continue;
                    }
                    if self.plan.active_host[b] == 0 {
                        // device-side stop — converged (rms<tol) or nonfinite;
                        // the status is decided from the last rms at the end.
                        done[b] = true;
                        n_active -= 1;
                        continue;
                    }
                    let r_b = self.plan.rms_host[b];
                    hist[b][samples[b] % 10] = r_b;
                    samples[b] += 1;
                    if !r_b.is_finite() {
                        done[b] = true;
                        self.plan.active_host[b] = 0;
                        n_active -= 1;
                        host_dirty = true;
                        continue;
                    }
                    if samples[b] >= 8 {
                        let (mut rmin, mut rmax) = (f32::INFINITY, 0.0f32);
                        for &r in &hist[b] {
                            rmin = rmin.min(r);
                            rmax = rmax.max(r);
                        }
                        if rmax < 4.0 * rmin {
                            done[b] = true;
                            stagnant[b] = true;
                            self.plan.active_host[b] = 0;
                            n_active -= 1;
                            host_dirty = true;
                        }
                    }
                }
                if host_dirty {
                    self.plan.set_active(&self.rt)?;
                }
                // T06 Phase B: rebuild the compact work list for the next
                // chunk — only slots still flagged active get workgroups.
                if compact && n_active > 0 && n_active < prev_n_active {
                    work_list.clear();
                    work_list.extend(
                        (0..self.batch)
                            .filter(|&b| self.plan.active_host[b] != 0)
                            .map(|b| b as i32),
                    );
                    self.plan.set_work_domain(&self.rt, &work_list)?;
                    if std::env::var_os("RUST_DFTB_DEBUG_DOMAIN").is_some() {
                        eprintln!(
                            "[scc.domain] it={it} active={n_active}/{} work_n={}",
                            self.batch,
                            self.plan.work_n()
                        );
                    }
                }
                prev_n_active = n_active;
            }
            // T06 Phase B: restore the full domain — eval/finalize/forces
            // and any subsequent solve must never inherit a tail list.
            self.plan.restore_work_domain(&self.rt)?;
            rms = self.plan.rms_host.iter().fold(0.0f32, |a, &x| a.max(x));
        } else {
            for it in 0..cap {
                n_iters = it + 1;
                rms = match mix {
                    0 => self
                        .plan
                        .scc_step_diis(&mut self.rt, self.n_occ, 0.3, rms_tol)?,
                    1 => self.plan.scc_step(&mut self.rt, self.n_occ, 0.3)?,
                    2 => {
                        self.plan.finalize(&mut self.rt, self.n_occ)?;
                        self.rt.read_buffer(&self.plan.q_gpu, &mut q_f32)?;
                        for a in 0..self.n_atoms {
                            q_in[a] = q_f32[a] as f64;
                        }
                        self.rt.read_buffer(&self.plan.q_new, &mut q_f32)?;
                        let mut s2 = 0.0f64;
                        for a in 0..self.n_atoms {
                            q_out[a] = q_f32[a] as f64;
                            res[a] = q_out[a] - q_in[a];
                            if !res[a].is_finite() {
                                return Err(DftbError::InvalidInput(format!(
                                    "host DIIS residual non-finite atom {a} q_out={} q_in={}",
                                    q_out[a], q_in[a]
                                )));
                            }
                            s2 += res[a] * res[a];
                        }
                        let r = (s2 / self.n_atoms as f64).sqrt();
                        if !r.is_finite() {
                            return Err(DftbError::InvalidInput(format!(
                                "host DIIS rms={r} non-finite"
                            )));
                        }
                        crate::qmqm::mixer::Mixer::mix(
                            host.as_mut().unwrap(),
                            &mut q_in,
                            &q_out,
                            &res,
                        );
                        for a in 0..self.n_atoms {
                            if !q_in[a].is_finite() {
                                return Err(DftbError::InvalidInput(format!(
                                    "host DIIS mixed q[{a}]={} non-finite",
                                    q_in[a]
                                )));
                            }
                            q_f32[a] = q_in[a] as f32;
                        }
                        // commit model: write the mixed iterate to q_next; the
                        // shared commit below moves it into q_gpu iff active.
                        self.plan.q_next.write(&q_f32).enq().map_err(map_ocl_err)?;
                        r as f32
                    }
                    other => {
                        return Err(DftbError::InvalidInput(format!(
                            "scc_mix: mix={other} not 0/1/2"
                        )))
                    }
                };
                // mix=2 (host DIIS): q_next written but not yet committed —
                // state consistent with q_gpu. mix∈{0,1}: scc_step* commits
                // internally — still-active replicas already advanced q_gpu
                // past the solved state → stale until the next finalize.
                self.state_fresh = mix == 2;
                // per-replica convergence: done when r<tol, stagnant when the
                // last-10 window stops moving (status decided by floor at end)
                for b in 0..self.batch {
                    if done[b] {
                        continue;
                    }
                    let r_b = if mix == 2 {
                        rms
                    } else {
                        self.plan.rms_host.get(b).copied().unwrap_or(rms)
                    };
                    hist[b][n_iters % 10] = r_b;
                    if r_b < rms_tol {
                        done[b] = true;
                        self.plan.active_host[b] = 0;
                        n_active -= 1;
                        continue;
                    }
                    if !r_b.is_finite() {
                        done[b] = true;
                        self.plan.active_host[b] = 0;
                        n_active -= 1;
                        continue;
                    } // Failed — stop iterating it
                    if n_iters >= 25 {
                        let (mut rmin, mut rmax) = (f32::INFINITY, 0.0f32);
                        for &r in &hist[b] {
                            rmin = rmin.min(r);
                            rmax = rmax.max(r);
                        }
                        if rmax < 4.0 * rmin {
                            done[b] = true;
                            stagnant[b] = true;
                            self.plan.active_host[b] = 0;
                            n_active -= 1;
                        }
                    }
                }
                if n_active == 0 {
                    break;
                }
                // Commit mixed iterates for still-active replicas only; frozen
                // replicas keep q_n and their just-solved electronic state.
                // mix=0: diis_step_batched already committed in-kernel (R7) —
                // running commit_q_next here would overwrite q_gpu with the
                // stale q_next.
                if mix != 0 {
                    self.plan.set_active(&self.rt)?;
                    self.plan.commit_q_next(&self.rt)?;
                }
                self.state_fresh = false;
            }
        }
        let mut stalled = false;
        for b in 0..self.batch {
            let r_b = if mix == 2 {
                rms
            } else {
                self.plan.rms_host.get(b).copied().unwrap_or(f32::NAN)
            };
            if r_b < rms_tol {
                continue;
            }
            stalled = true;
        }
        // D9: report device-side DIIS fallbacks (replaces kernel printf).
        if mix == 0 {
            let mut flag = vec![0i32; self.batch];
            let mut reason = vec![0i32; self.batch];
            self.plan.diis_status(&self.rt, &mut flag, &mut reason)?;
            let mut n_fb = 0i32;
            let mut worst = 0i32;
            let mut worst_sid = 0usize;
            for (sid, &f) in flag.iter().enumerate() {
                if f > 0 {
                    n_fb += f;
                    if reason[sid] > worst {
                        worst = reason[sid];
                        worst_sid = sid;
                    }
                }
            }
            if n_fb > 0 {
                eprintln!("[GpuDftb] DIIS fallbacks: {n_fb} total, last reason={worst} at sid={worst_sid} (1=pivot/scale 2=nonfinite 3=sum(c)!=1)");
            }
        }
        // W1 (manifest §14): certify the LAST Jacobi solve per replica, once —
        // this is the single deferred diag read that replaced the
        // per-iteration report_jacobi queue drain. An uncertified eigensolve
        // makes the replica Failed regardless of charge rms.
        let j_ok = self.plan.check_jacobi(&self.rt, &jacobi_ran)?;
        self.rt.prof_tick("scc.check_jacobi");
        // D11: per-system status — Converged only when the final residual is
        // below tol; Plateau when the iterate stopped moving; else Failed.
        // mix=2 (host DIIS) fills only the scalar rms (batch=1 path).
        let statuses: Vec<SccStatus> = (0..self.batch)
            .map(|sid| {
                let r = if mix == 2 {
                    rms
                } else {
                    self.plan.rms_host.get(sid).copied().unwrap_or(f32::NAN)
                };
                if !r.is_finite() || !j_ok[sid] {
                    SccStatus::Failed
                } else if r < rms_tol {
                    SccStatus::Converged
                } else if stagnant[sid] && r < plateau_floor {
                    SccStatus::Plateau
                } else {
                    SccStatus::Failed
                }
            })
            .collect();
        if self.batch > 1 || statuses.first() != Some(&SccStatus::Converged) {
            // per-replica rms — the diagnostic that shows WHO failed/plateaued
            let rms_str: Vec<String> = (0..self.batch)
                .map(|sid| {
                    let r = if mix == 2 {
                        rms
                    } else {
                        self.plan.rms_host.get(sid).copied().unwrap_or(f32::NAN)
                    };
                    format!("{r:.1e}")
                })
                .collect();
            eprintln!(
                "[GpuDftb] SCC status: {}",
                statuses
                    .iter()
                    .map(|s| match s {
                        SccStatus::Converged => "converged",
                        SccStatus::Plateau => "plateau",
                        SccStatus::Failed => "FAILED",
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            );
            eprintln!("[GpuDftb] SCC rms per replica: {}", rms_str.join(","));
        }
        // failed replicas must not be stepped on garbage forces (relax parks them)
        self.scc_ok = statuses.iter().map(|s| *s != SccStatus::Failed).collect();
        // R1: k_edm (W build) gates on state_ok — certified replicas get a
        // fresh W even when they converged before the last SCC iteration.
        self.plan.set_state_ok(&self.rt, &self.scc_ok)?;
        self.sync_park()?; // W6: FIRE park mask
        Ok(GpuDftbScc {
            n_iters,
            rms,
            stalled,
            statuses,
        })
    }

    /// Energy always; forces if `want_forces`. Do not call energy() then forces().
    /// Reuses the cached electronic state when scc_mix exited with q_next
    /// uncommitted (state_fresh) — finalize re-solves only after a geometry
    /// or charge change.
    pub fn eval(&mut self, want_forces: bool) -> Result<GpuDftbEval> {
        if !self.state_fresh {
            self.plan.finalize(&mut self.rt, self.n_occ)?;
            self.state_fresh = true;
            // W1: certify the just-run eigensolve — one deferred diag read
            // per eval (not per SCC iteration). Uncertified replicas are
            // parked like an SCC failure, not silently consumed.
            let j_ok = self.plan.check_jacobi(&self.rt, &self.ones_batch)?;
            for (b, ok) in j_ok.iter().enumerate() {
                if !ok {
                    self.scc_ok[b] = false;
                }
            }
            self.sync_park()?; // W6: FIRE park mask
        }
        let energy = self.plan.energy_from_state(&mut self.rt)?;
        let (q_rms, q_max) = self.charge_residual()?;
        let forces = if want_forces {
            Some(self.forces_from_state()?)
        } else {
            None
        };
        Ok(GpuDftbEval {
            energy,
            forces,
            q_rms,
            q_max,
        })
    }

    /// Wrapper: `eval(false)`. Prefer `eval` when you also want forces.
    pub fn energy(&mut self) -> Result<Vec<f64>> {
        Ok(self.eval(false)?.energy)
    }

    /// Wrapper: `eval(true)`. Prefer `eval` when you also want energy.
    pub fn forces(&mut self) -> Result<Vec<f32>> {
        self.eval(true)?
            .forces
            .ok_or_else(|| DftbError::InvalidInput("eval(true) returned no forces".into()))
    }

    /// `q_D = Mulliken(D,S)` vs `q_in = q_gpu` after finalize. Persistent scratch, no alloc.
    fn charge_residual(&mut self) -> Result<(f64, f64)> {
        self.rt.read_buffer(&self.plan.q_gpu, &mut self.scratch_q)?;
        self.rt
            .read_buffer(&self.plan.q_new, &mut self.scratch_qd)?;
        let n_atoms = self.n_atoms;
        let batch = self.batch;
        let mut q_max = 0.0f64;
        let mut q_rms = 0.0f64;
        for b in 0..batch {
            let mut s2 = 0.0f64;
            let mut mx = 0.0f64;
            for a in 0..n_atoms {
                let d = (self.scratch_qd[b * n_atoms + a] - self.scratch_q[b * n_atoms + a]) as f64;
                if !d.is_finite() {
                    return Err(DftbError::InvalidInput(format!(
                        "charge residual non-finite replica {b} atom {a} q_D={} q_in={}",
                        self.scratch_qd[b * n_atoms + a],
                        self.scratch_q[b * n_atoms + a]
                    )));
                }
                s2 += d * d;
                mx = mx.max(d.abs());
            }
            q_max = q_max.max(mx);
            q_rms = q_rms.max((s2 / n_atoms as f64).sqrt());
        }
        Ok((q_rms, q_max))
    }

    /// Force kernels into `buf_forces` from the current finalized D/C/ε.
    /// Device-only: forces are zeroed with a device fill (no host upload)
    /// and left in `buf_forces` — `forces_from_state` adds the readback.
    /// Caller must have `finalize`d.
    fn enqueue_force_kernels(&mut self) -> Result<()> {
        self.plan.build_edm(&self.buf_edm, self.n_occ)?;
        self.buf_forces
            .cmd()
            .fill(0.0f32, None)
            .enq()
            .map_err(map_ocl_err)?;
        for slot in &self.buckets {
            if slot.n_live == 0 {
                continue;
            }
            slot.k_fused
                .set_arg(4u32, &self.plan.d)
                .map_err(map_ocl_err)?; // dm
            slot.k_fused
                .set_arg(5u32, &self.buf_edm)
                .map_err(map_ocl_err)?; // edm
            slot.k_fused
                .set_arg(6u32, &self.plan.v)
                .map_err(map_ocl_err)?; // v_shift
            unsafe {
                slot.k_fused.enq().map_err(map_ocl_err)?;
            }
        }
        // W9b: per-atom gather (deterministic CSR order, sole writer of the
        // pair-force part — gamma/rep kernels atomic-add after it).
        // I10: same pass emits e_atom = onsite + ½Σ pair_e = Tr(D·H0).
        self.k_gather
            .set_arg(6u32, &self.plan.d)
            .map_err(map_ocl_err)?; // dm
        unsafe {
            self.k_gather.enq().map_err(map_ocl_err)?;
        }
        self.k_gamma_f
            .set_arg(2u32, &self.buf_coords_ang)
            .map_err(map_ocl_err)?;
        self.k_gamma_f
            .set_arg(4u32, &self.plan.dq)
            .map_err(map_ocl_err)?;
        unsafe {
            self.k_gamma_f.enq().map_err(map_ocl_err)?;
        }
        self.k_rep_force
            .set_arg(2u32, &self.buf_coords_bohr)
            .map_err(map_ocl_err)?;
        unsafe {
            self.k_rep_force.enq().map_err(map_ocl_err)?;
        }
        Ok(())
    }

    /// Force kernels + host readback — public/API path (once per call).
    fn forces_from_state(&mut self) -> Result<Vec<f32>> {
        self.enqueue_force_kernels()?;
        self.rt.read_buffer(&self.buf_forces, &mut self.scratch_f)?;
        Ok(self.scratch_f.clone())
    }

    /// eval forces without the array readback — the FIRE hot path.
    fn eval_forces_device(&mut self) -> Result<()> {
        let mut solved = false;
        if !self.state_fresh {
            self.plan.finalize(&mut self.rt, self.n_occ)?;
            self.rt.prof_tick("fire.finalize");
            self.state_fresh = true;
            solved = true;
        }
        self.enqueue_force_kernels()?;
        self.rt.prof_tick("fire.force_kernels");
        if solved {
            // W1: certify the eigensolve AFTER the force kernels are queued —
            // the single read_buffer/finish then drains finalize+forces in
            // one sync instead of a mid-pipeline drain after finalize alone.
            let j_ok = self.plan.check_jacobi(&self.rt, &self.ones_batch)?;
            self.rt.prof_tick("fire.check_jacobi");
            for (b, ok) in j_ok.iter().enumerate() {
                if !ok {
                    self.scc_ok[b] = false;
                }
            }
            self.sync_park()?; // W6: FIRE park mask
        }
        Ok(())
    }

    /// Fermi smearing kT in Hartree (0 = integer occupation). Stabilizes SCC
    /// at near-degenerate HOMO/LUMO geometries (e.g. mid proton transfer:
    /// gap ~0.5 mHa makes integer occ flip → O(1) Δq oscillation).
    /// Suggested ~0.002 (≈630 K). Costs a tiny eig readback + μ bisection
    /// per SCC iter; energy becomes 2Σf_k ε_k, forces use W=2Σf_k ε_k CCᵀ.
    pub fn set_smearing(&mut self, kt: f32) {
        if !kt.is_finite() || kt < 0.0 {
            panic!("set_smearing: kT={kt}");
        }
        self.plan.kT = kt;
        self.state_fresh = false; // occupation model changed — cached state invalid (R4)
        eprintln!(
            "[GpuDftb] Fermi smearing kT={kt} Ha{}",
            if kt > 0.0 { "" } else { " (integer occ)" }
        );
    }

    /// Pin atoms (template indices, applied to every replica) for
    /// constrained relaxed scans — e.g. the transferred proton fixed at a
    /// scan-point position. Forces/velocities on frozen atoms are ignored
    /// and excluded from that replica's convergence test.
    pub fn set_frozen_atoms(&mut self, idx: &[usize]) -> Result<()> {
        self.frozen.fill(false);
        for &i in idx {
            if i >= self.n_atoms {
                return Err(DftbError::InvalidInput(format!(
                    "set_frozen_atoms: index {i} >= n_atoms {}",
                    self.n_atoms
                )));
            }
            self.frozen[i] = true;
        }
        // R4: freezing both constraint endpoints AFTER set_constraint leaves
        // no free DOF to satisfy the distance — reject at the same place
        // set_constraint does.
        if let Some((i, j)) = self.constr {
            if self.frozen[i] && self.frozen[j] {
                return Err(DftbError::InvalidInput(format!(
                    "set_frozen_atoms: would freeze both constraint endpoints ({i},{j}) — no free DOF to satisfy the constraint"
                )));
            }
        }
        let m: Vec<i32> = self.frozen.iter().map(|&f| f as i32).collect();
        self.rt
            .write_buffer(&self.frozen_dev, &m)
            .map_err(|e| DftbError::InvalidInput(format!("write frozen_dev: {e}")))
    }

    /// Distance constraint |x_j − x_i| = d[b] per replica (Å) — the relaxed
    /// scan coordinate. Closed-form equal-mass projection inside
    /// `fire_apply_batched`; a frozen endpoint gets weight 0 (the other
    /// endpoint carries the full correction). Fails loudly if both
    /// endpoints are frozen or indices are invalid.
    pub fn set_constraint(&mut self, i: usize, j: usize, targets: &[f64]) -> Result<()> {
        if i == j || i >= self.n_atoms || j >= self.n_atoms {
            return Err(DftbError::InvalidInput(format!(
                "set_constraint: invalid pair ({i},{j}) n_atoms={}",
                self.n_atoms
            )));
        }
        if self.frozen[i] && self.frozen[j] {
            return Err(DftbError::InvalidInput(format!(
                "set_constraint: both endpoints frozen ({i},{j}) — no free DOF to satisfy the constraint"
            )));
        }
        if targets.len() != self.batch {
            return Err(DftbError::InvalidInput(format!(
                "set_constraint: targets.len()={} != batch={}",
                targets.len(),
                self.batch
            )));
        }
        // R4: finite positive targets AFTER the f64→f32 cast — a NaN/≤0
        // target would silently corrupt every projection.
        let t: Vec<f32> = targets.iter().map(|&d| d as f32).collect();
        for (b, (&d64, &d32)) in targets.iter().zip(t.iter()).enumerate() {
            if !d64.is_finite() || !d32.is_finite() || d32 <= 0.0 {
                return Err(DftbError::InvalidInput(format!(
                    "set_constraint: replica {b} target d={d64} → f32 {d32} — need finite > 0"
                )));
            }
        }
        // R4: validate + project the INITIAL coordinates onto the constraint
        // surface. Otherwise the first constrained force/conv check sees a
        // violated constraint — a small force could park a replica at the
        // wrong distance. Coincident endpoints cannot be projected → error.
        self.sync_coords_to_host()?;
        let na = self.n_atoms;
        let mi = if self.frozen[i] { 0.0f64 } else { 1.0f64 };
        let mj = if self.frozen[j] { 0.0f64 } else { 1.0f64 };
        let ms = mi + mj; // >0 — both-frozen rejected above
        let (wi, wj) = (mi / ms, mj / ms);
        let mut max_shift = 0.0f64;
        for b in 0..self.batch {
            let (bi, bj) = (b * na + i, b * na + j);
            let rx = self.coords[bj][0] - self.coords[bi][0];
            let ry = self.coords[bj][1] - self.coords[bi][1];
            let rz = self.coords[bj][2] - self.coords[bi][2];
            let r = (rx * rx + ry * ry + rz * rz).sqrt();
            if !r.is_finite() || r < 1e-6 {
                return Err(DftbError::InvalidInput(format!(
                    "set_constraint: replica {b} endpoints ({i},{j}) coincident/non-finite r={r} — cannot project"
                )));
            }
            let g = r - targets[b];
            let (ux, uy, uz) = (rx / r, ry / r, rz / r);
            self.coords[bi][0] += wi * g * ux;
            self.coords[bi][1] += wi * g * uy;
            self.coords[bi][2] += wi * g * uz;
            self.coords[bj][0] -= wj * g * ux;
            self.coords[bj][1] -= wj * g * uy;
            self.coords[bj][2] -= wj * g * uz;
            max_shift = max_shift.max(g.abs());
        }
        if max_shift > 1e-9 {
            eprintln!("[GpuDftb] constraint: projected initial coords onto |x_{j}−x_{i}|=d[b] (max |r−d|={max_shift:.3e} Å)");
            let c = self.coords.clone();
            self.set_coords(&c)?;
        }
        self.rt
            .write_buffer(&self.constr_d, &t)
            .map_err(|e| DftbError::InvalidInput(format!("write constr_d: {e}")))?;
        self.k_fire_apply.set_arg(8u32, 1i32).map_err(map_ocl_err)?;
        self.k_fire_apply
            .set_arg(9u32, i as i32)
            .map_err(map_ocl_err)?;
        self.k_fire_apply
            .set_arg(10u32, j as i32)
            .map_err(map_ocl_err)?;
        self.k_fire_reduce
            .set_arg(5u32, 1i32)
            .map_err(map_ocl_err)?;
        self.k_fire_reduce
            .set_arg(6u32, i as i32)
            .map_err(map_ocl_err)?;
        self.k_fire_reduce
            .set_arg(7u32, j as i32)
            .map_err(map_ocl_err)?;
        self.constr = Some((i, j));
        eprintln!(
            "[GpuDftb] distance constraint |x_{j} − x_{i}| = d[b] set ({} replicas)",
            self.batch
        );
        Ok(())
    }

    /// Remove the distance constraint.
    pub fn clear_constraint(&mut self) -> Result<()> {
        self.k_fire_apply.set_arg(8u32, 0i32).map_err(map_ocl_err)?;
        self.k_fire_reduce
            .set_arg(5u32, 0i32)
            .map_err(map_ocl_err)?;
        self.constr = None;
        Ok(())
    }

    /// One FIRE step on all replicas (Bitzek 2006, D12-fixed, D13 device-resident).
    /// Correct physics: mixing is `v ← (1−α)v + α·F̂·‖v‖` with ‖v‖,‖F‖ the
    /// GLOBAL per-replica norms (NOT per-atom α‖F_i‖F̂_i — that would pin
    /// each atom's speed to its force). dt/α/n_pos are per-replica state —
    /// batched systems adapt independently; a replica whose max|F| (over
    /// UNfrozen atoms) < f_tol is parked with zero velocity.
    /// Ordering (matches `FireOptimizer` in examples/hbond_ref.rs):
    ///   P=F·v → adapt → mix → v+=F·dt (vmax cap) → x+=v·dt (disp cap).
    ///
    /// Data flow per step (nothing larger than 4·batch floats crosses PCIe):
    ///   forces → buf_forces (device) → fire_reduce → host f64 decisions
    ///   (dt/α/mode) → fire_apply updates v_dev + coords_ang/bohr in place
    ///   → assemble reads the device coords directly.
    /// Call `scc` first. Returns global max |F|.
    /// Sync the device `park` mask from `scc_ok` — W6 gates the on-device
    /// FIRE control on it. Called wherever scc_ok is rewritten.
    fn sync_park(&self) -> Result<()> {
        let v: Vec<i32> = self.scc_ok.iter().map(|&b| b as i32).collect();
        self.rt.write_buffer(&self.park, &v)
    }

    pub fn fire_step(&mut self, f_tol: f64) -> Result<f64> {
        // W6: N_MIN/F_INC/F_DEC/ALPHA_START/F_ALPHA/DT_MAX live in the
        // reduce tail on device (constants in gpu_forces.cl) — the host
        // only reads `stat` for the convergence test and error checks.
        self.eval_forces_device()?; // forces stay on device
        self.rt.prof_tick("fire.eval_forces");
        self.k_fire_reduce
            .set_arg(11u32, f_tol as f32)
            .map_err(map_ocl_err)?;
        unsafe {
            self.k_fire_reduce.enq().map_err(map_ocl_err)?;
        }
        self.rt
            .read_buffer(&self.fire_stat, &mut self.fire_stat_h)?; // 4·batch floats
        self.rt.prof_tick("fire.reduce");
        let mut max_f = 0.0f64;
        for b in 0..self.batch {
            let p = self.fire_stat_h[4 * b] as f64;
            let v2 = self.fire_stat_h[4 * b + 1] as f64;
            let f2 = self.fire_stat_h[4 * b + 2] as f64;
            let mf = self.fire_stat_h[4 * b + 3] as f64;
            // R4: check ALL decision quantities — a NaN in v²/F² would
            // silently poison mix_s even when P and maxF are finite.
            if !p.is_finite() || !v2.is_finite() || !f2.is_finite() || !mf.is_finite() {
                return Err(DftbError::InvalidInput(format!(
                    "fire_step replica {b}: non-finite stat P={p} v²={v2} F²={f2} max|F|={mf}"
                )));
            }
            max_f = max_f.max(mf);
        }
        if max_f < f_tol {
            return Ok(max_f);
        } // all converged — nothing moves
          // ctl was adapted on-device by the reduce tail — apply directly.
        unsafe {
            self.k_fire_apply.enq().map_err(map_ocl_err)?;
        }
        self.rt.prof_tick("fire.apply");
        // Positions now live on the device — assemble reads them in place.
        self.coords_on_device = true;
        self.coords_dirty = true;
        self.assemble()?;
        self.rt.prof_tick("fire.assemble");
        self.plan.set_geometry(&mut self.rt, &self.buf_s)?;
        self.rt.prof_tick("fire.set_geometry");
        self.plan.reset_diis(&self.rt)?;
        self.rt.prof_tick("fire.reset_diis");
        self.state_fresh = false;
        Ok(max_f)
    }

    /// One velocity-Verlet step (mass=1, same reduced units as FIRE).
    ///   x += v·dt + ½·a(t)·dt²;  a(t+dt) from the new geometry;  v += ½·(a(t)+a(t+dt))·dt
    /// `md_f_prev` carries a(t) between calls — ONE eval per step in steady
    /// state (the first call after arming needs an extra eval for a(t)).
    /// Displacement capped at 0.1 Å. Returns max |F| at the new geometry.
    /// Host-side diagnostic path (v/x round-trip through host each step).
    pub fn md_step(&mut self, dt: f64) -> Result<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(DftbError::InvalidInput(format!(
                "md_step: dt={dt} must be finite and > 0"
            )));
        }
        // sync device v + x into the host mirrors (v_dev→fire_v, coords_ang→coords)
        self.rt
            .read_buffer(&self.v_dev, &mut self.scratch_f)
            .map_err(|e| DftbError::InvalidInput(format!("md_step v readback: {e}")))?;
        for i in 0..self.fire_v.len() {
            for c in 0..3 {
                self.fire_v[i][c] = self.scratch_f[3 * i + c] as f64;
            }
        }
        self.sync_coords_to_host()?;
        let ntot = self.batch * self.n_atoms;
        if !self.md_armed {
            // a(t): one force eval at the current geometry, then drift+kick
            let f0 = self.eval(true)?.forces.expect("eval(true) returns forces");
            for i in 0..ntot {
                for c in 0..3 {
                    self.md_f_prev[i][c] = f0[3 * i + c] as f64;
                }
            }
        }
        // drift with a(t), then move + evaluate a(t+dt)
        for i in 0..ntot {
            let (fx, fy, fz) = (
                self.md_f_prev[i][0],
                self.md_f_prev[i][1],
                self.md_f_prev[i][2],
            );
            apply_disp(&mut self.coords[i], &self.fire_v[i], fx, fy, fz, dt);
        }
        let c = self.coords.clone(); // host API — not the hot loop
        self.set_coords(&c)?; // (disarms md_armed; re-armed below)
        let f1 = self.eval(true)?.forces.expect("eval(true) returns forces");
        let mut max_f = 0.0f64;
        for i in 0..ntot {
            let fx = f1[3 * i] as f64;
            let fy = f1[3 * i + 1] as f64;
            let fz = f1[3 * i + 2] as f64;
            max_f = max_f.max(fx.abs()).max(fy.abs()).max(fz.abs());
            // second half-kick with ½·(a(t)+a(t+dt))
            self.fire_v[i][0] += 0.5 * (self.md_f_prev[i][0] + fx) * dt;
            self.fire_v[i][1] += 0.5 * (self.md_f_prev[i][1] + fy) * dt;
            self.fire_v[i][2] += 0.5 * (self.md_f_prev[i][2] + fz) * dt;
            self.md_f_prev[i] = [fx, fy, fz];
        }
        self.md_armed = true;
        // write v back to the device
        for i in 0..self.fire_v.len() {
            for c in 0..3 {
                self.scratch_f[3 * i + c] = self.fire_v[i][c] as f32;
            }
        }
        self.rt
            .write_buffer(&self.v_dev, &self.scratch_f)
            .map_err(|e| DftbError::InvalidInput(format!("md_step v upload: {e}")))?;
        Ok(max_f)
    }

    /// Relax: SCC + FIRE until max|F|<f_tol or `max_steps`. Prints unbuffered progress.
    /// Returns (steps, max|F| at the final geometry, FINAL scc rms, converged).
    /// `converged=false` means the loop exhausted max_steps — the coordinates
    /// are NOT a relaxed geometry (R2: exhaustion must not be misreported as
    /// convergence; the old signature returned the INITIAL scc rms).
    pub fn relax(
        &mut self,
        max_steps: usize,
        f_tol: f64,
        scc_tol: f32,
    ) -> Result<(usize, f64, f32, bool)> {
        self.rt.prof_reset();
        let scc0 = self.scc_with_retry(SCC_MAX_ITER, scc_tol)?; // W11: explicit warm-start retry
        self.rt.prof_tick("relax.scc0");
        eprintln!(
            "[GpuDftb] relax start rms={:.3e} iters={} stalled={}",
            scc0.rms, scc0.n_iters, scc0.stalled
        );
        let mut max_f = f64::INFINITY;
        let mut last_rms = scc0.rms;
        let mut step = 0;
        for s in 0..max_steps {
            step = s + 1;
            max_f = self.fire_step(f_tol)?;
            self.rt.prof_tick("relax.fire_step");
            let scc = self.scc_with_retry(SCC_MAX_ITER, scc_tol)?; // W11: explicit retry
            self.rt.prof_tick("relax.scc");
            last_rms = scc.rms;
            let e = self.eval(false)?.energy;
            self.rt.prof_tick("relax.eval");
            eprintln!("[GpuDftb] FIRE {step}/{max_steps} max|F|={max_f:.4e} E[0]={:.8} rms={:.3e} scc_iters={}", e[0], scc.rms, scc.n_iters);
            if max_f < f_tol {
                break;
            }
        }
        self.sync_coords_to_host()?; // host `coords` reflects final device x
        self.rt.prof_tick("relax.sync_coords");
        self.prof_report("relax");
        let converged = max_f < f_tol;
        if !converged {
            eprintln!("[GpuDftb] relax EXHAUSTED after {step} steps: max|F|={max_f:.4e} > f_tol={f_tol:.3e} — NOT a converged geometry");
        }
        Ok((step, max_f, last_rms, converged))
    }

    /// Print the env-gated (`RUST_DFTB_PROF=1`) stage-timer table for this
    /// engine and reset it — called by `relax` and at script end.
    // ── Dense_Multi_CDFT: fragment Mulliken-charge constraints ──────
    // Q_F = Σ_{A∈F} Δq_A = Q_F^target via a per-fragment multiplier
    // λ_F that enters h_scc as ½λ_F·S·(w_μ+w_ν) — injected inside
    // enq_dq_v_hscc (see plan.cdft). Energies from eval() then contain
    // +Σλ_F·Q_gross(F); cdft_energies() removes it → E_DFTB(constrained).

    /// Attach a fragment-charge constraint set. `frag` is [n_atoms]:
    /// −1 = unconstrained, else fragment id 0..nfrag−1 (template level,
    /// same for all replicas). `targets` is [batch*nfrag] target excess
    /// charge in e — per-replica targets make one batch a diabatic-state
    /// ladder. λ starts at 0. Returns nfrag.
    pub fn set_cdft(&mut self, frag: &[i32], targets: &[f64]) -> Result<usize> {
        let mut cdft = crate::qmqm::gpu_cdft::GpuCdft::new(
            &mut self.rt,
            self.n,
            self.n_atoms,
            self.batch,
            frag,
            targets,
            &self.buf_s,
            &self.buf_oa,
            &self.plan.h_scc,
            &self.plan.active,
            &self.plan.work_ids,
        )?;
        // fragment reference populations Q0_F = Σ_{A∈F} q0_A — needed by
        // cdft_energies (the h_scc shift picks up λ·Q_gross, not λ·Δq).
        for (a, &f) in frag.iter().enumerate() {
            if f >= 0 {
                cdft.q0_frag[f as usize] += self.q0[a] as f64;
            }
        }
        let nfrag = cdft.nfrag;
        self.plan.cdft = Some(cdft);
        self.state_fresh = false; // h_scc on device predates the constraint
        Ok(nfrag)
    }

    /// Drop the constraint set — h_scc is rebuilt without the shift at
    /// the next scc/eval (it is reconstructed from H0 every iteration).
    pub fn clear_cdft(&mut self) {
        self.plan.cdft = None;
        self.state_fresh = false;
    }

    /// Manually set λ[b][f] (Ha) and upload — for response scans
    /// Q_F(λ) diagnostics. Resets the secant/bracket state so a later
    /// cdft_scc starts clean from this λ.
    pub fn cdft_set_lam(&mut self, b: usize, f: usize, val: f64) -> Result<()> {
        let c = self
            .plan
            .cdft
            .as_mut()
            .ok_or_else(|| DftbError::InvalidInput("cdft_set_lam: no constraint set".into()))?;
        let i = b * c.nfrag + f;
        if i >= c.lam.len() {
            return Err(DftbError::InvalidInput(format!(
                "cdft_set_lam: (b={b},f={f}) out of range"
            )));
        }
        c.lam[i] = val;
        c.clear_search(i);
        c.upload_lam(&self.rt)?;
        self.state_fresh = false;
        Ok(())
    }

    /// Outer-λ constrained solve. Each outer iteration runs a full SCC
    /// at fixed λ (warm-started on the previous state), reads the
    /// converged fragment charges, and updates λ by a safeguarded
    /// secant until |Q_F − Q_F^target| ≤ q_tol for all (b,f).
    pub fn cdft_scc(
        &mut self,
        max_outer: usize,
        scc_iter: usize,
        rms_tol: f32,
        q_tol: f64,
    ) -> Result<crate::qmqm::gpu_cdft::CdftReport> {
        if self.plan.cdft.is_none() {
            return Err(DftbError::InvalidInput(
                "cdft_scc: no constraint set — call set_cdft first".into(),
            ));
        }
        let (batch, na) = (self.batch, self.n_atoms);
        let nfrag = self.plan.cdft.as_ref().unwrap().nfrag;
        // κ0 per fragment: |dQ_F/dλ| ≈ n_f/Ū_f → κ0 = Ū_f/n_f (Ha per e).
        let mut kappa0 = vec![0.05f64; nfrag];
        {
            let c = self.plan.cdft.as_ref().unwrap();
            for f in 0..nfrag {
                let (mut su, mut nf) = (0.0f64, 0usize);
                for (a, &fa) in c.frag.iter().enumerate() {
                    if fa as usize == f {
                        su += self.u_per_atom[a];
                        nf += 1;
                    }
                }
                if nf > 0 && su > 0.0 {
                    kappa0[f] = (su / nf as f64) / nf as f64;
                }
            }
        }
        let mut dq_host = vec![0.0f32; batch * na];
        let mut qfrag = vec![0.0f64; batch * nfrag];
        let mut qnat = vec![0.0f64; batch * nfrag]; // natural Q_F at λ=0
        let mut qerr = vec![0.0f64; batch * nfrag]; // vs FINAL target
        let mut qerr_eff = vec![0.0f64; batch * nfrag]; // vs ramped target
        let mut converged = vec![false; batch];
        let mut scc_ok_last = vec![false; batch];
        let mut outer_done = 0usize;
        // Stall recovery: at a level crossing Q(λ) is discontinuous AND
        // the SCC can sit in the wrong metastable basin — λ stops moving
        // while err stays > tol. After 3 stalled outer iters, cold-restart
        // that replica's charges (q→q0) so the SCC can land in the basin
        // on the other side of the crossing.
        let mut qerr_prev = vec![f64::NAN; batch * nfrag];
        let mut stall_ct = vec![0u32; batch * nfrag];
        let mut restarted = vec![false; batch];
        // Best-effort bookkeeping: metastable SCC basins mean the target
        // may be unreachable on some branches — always remember the best
        // |err| λ seen and restore it at the end, so unconverged replicas
        // still report their closest-achievable constrained state.
        let mut best_err = vec![f64::INFINITY; batch * nfrag];
        let mut best_lam = vec![0.0f64; batch * nfrag];
        const RAMP: usize = 8; // continuation steps natural→target
        for outer in 0..max_outer {
            outer_done = outer + 1;
            let s = self.scc(scc_iter, rms_tol)?;
            for (b, st) in s.statuses.iter().enumerate() {
                scc_ok_last[b] = !matches!(st, SccStatus::Failed);
            }
            self.rt.read_buffer(&self.plan.dq, &mut dq_host)?;
            // Continuation (target ramp): jumping straight to a far target
            // lands SCC in a random metastable basin — Q(λ) then looks
            // discontinuous and no λ reaches the target. Ramping the
            // effective target from the natural charge keeps each replica
            // tracking one basin adiabatically. ramp=1 at outer≥RAMP.
            let ramp = ((outer + 1).min(RAMP) as f64) / RAMP as f64;
            {
                let c = self.plan.cdft.as_ref().unwrap();
                qfrag = c.qfrag_from_dq(&dq_host, batch, na);
                if outer == 0 {
                    qnat.copy_from_slice(&qfrag);
                }
                for i in 0..batch * nfrag {
                    let t_eff = qnat[i] + (c.target[i] - qnat[i]) * ramp;
                    qerr[i] = qfrag[i] - c.target[i];
                    qerr_eff[i] = qfrag[i] - t_eff;
                }
            }
            let ramp_done = outer + 1 >= RAMP;
            let mut all_ok = true;
            for b in 0..batch {
                let mut ok = scc_ok_last[b];
                for f in 0..nfrag {
                    if qerr[b * nfrag + f].abs() > q_tol {
                        ok = false;
                    }
                }
                converged[b] = ok; // hitting the FINAL target mid-ramp counts
                if !ok {
                    all_ok = false;
                }
            }
            // record best-λ so far (only once the ramp is complete and SCC ok)
            if ramp_done {
                let c = self.plan.cdft.as_ref().unwrap();
                for b in 0..batch {
                    if !scc_ok_last[b] {
                        continue;
                    }
                    for f in 0..nfrag {
                        let i = b * nfrag + f;
                        if qerr[i].abs() < best_err[i] {
                            best_err[i] = qerr[i].abs();
                            best_lam[i] = c.lam[i];
                        }
                    }
                }
            }
            let q_err_max = qerr.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            eprintln!(
                "[cdft] outer={} q_err_max={:.3e} conv={}/{}",
                outer_done,
                q_err_max,
                converged.iter().filter(|&&x| x).count(),
                batch
            );
            // stall diagnostic: worst replica's λ, err, bracket width,
            // SCC status — distinguishes a collapsed bracket (hysteresis
            // or discontinuity in Q(λ)) from a still-shrinking search.
            if !all_ok && (outer < 4 || outer_done % 5 == 0) {
                let c2 = self.plan.cdft.as_ref().unwrap();
                let (mut bw, mut be) = (0usize, 0.0f64);
                for i in 0..batch * nfrag {
                    if qerr[i].abs() > be {
                        be = qerr[i].abs();
                        bw = i;
                    }
                }
                let (b, f) = (bw / nfrag, bw % nfrag);
                eprintln!("[cdft]   worst b={b} f={f}: lam={:.5} err={:.4e} br=[{:.5},{:.5}] scc_ok={} lam_prev={:.5} err_prev={:.4e}",
                    c2.lam[bw], qerr[bw], c2.debug_br(bw).0, c2.debug_br(bw).1,
                    scc_ok_last[b], c2.lam_prev(bw), c2.err_prev(bw));
            }
            if all_ok {
                break;
            }
            // stall bookkeeping + basin reset before updating λ
            for b in 0..batch {
                let mut stalled_rep = false;
                for f in 0..nfrag {
                    let i = b * nfrag + f;
                    if qerr[i].abs() <= q_tol {
                        stall_ct[i] = 0;
                        continue;
                    }
                    if qerr_prev[i].is_finite() && (qerr[i] - qerr_prev[i]).abs() < 1e-6 {
                        stall_ct[i] += 1;
                    } else {
                        stall_ct[i] = 0;
                    }
                    if stall_ct[i] >= 3 {
                        stalled_rep = true;
                        let c = self.plan.cdft.as_mut().unwrap();
                        c.clear_search(i);
                        c.lam[i] = 0.0; // re-approach from the neutral-connected branch
                        stall_ct[i] = 0;
                    }
                }
                if stalled_rep {
                    let mut q = vec![0.0f32; batch * na];
                    self.rt.read_buffer(&self.plan.q_gpu, &mut q)?;
                    q[b * na..(b + 1) * na].copy_from_slice(&self.q0[b * na..(b + 1) * na]);
                    self.rt.write_buffer(&self.plan.q_gpu, &q)?;
                    restarted[b] = true;
                    self.state_fresh = false;
                    eprintln!("[cdft]   replica {b}: SCC basin reset (q→q0) after stall");
                }
            }
            qerr_prev.copy_from_slice(&qerr);
            {
                let c = self.plan.cdft.as_mut().unwrap();
                for b in 0..batch {
                    for f in 0..nfrag {
                        let i = b * nfrag + f;
                        if qerr[i].abs() <= q_tol {
                            continue;
                        } // final target already hit — freeze
                        if qerr_eff[i].abs() > q_tol {
                            if !ramp_done {
                                c.clear_search(i);
                            } // moving target: no valid bracket
                            c.update_lam(b, f, qerr_eff[i], kappa0[f], 0.5);
                        }
                    }
                }
                c.upload_lam(&self.rt)?;
            }
        }
        // Restore best-λ for replicas that never hit the target, then one
        // final SCC so every reported state is the closest-achievable
        // constrained solution (not whatever the last bounce produced).
        let mut restored = false;
        {
            let c = self.plan.cdft.as_mut().unwrap();
            for i in 0..batch * nfrag {
                if qerr[i].abs() > q_tol && best_err[i].is_finite() && best_lam[i] != c.lam[i] {
                    c.lam[i] = best_lam[i];
                    restored = true;
                }
            }
            if restored {
                c.upload_lam(&self.rt)?;
            }
        }
        if restored {
            let s = self.scc(scc_iter, rms_tol)?;
            for (b, st) in s.statuses.iter().enumerate() {
                scc_ok_last[b] = !matches!(st, SccStatus::Failed);
            }
            self.rt.read_buffer(&self.plan.dq, &mut dq_host)?;
            let c = self.plan.cdft.as_ref().unwrap();
            qfrag = c.qfrag_from_dq(&dq_host, batch, na);
            for i in 0..batch * nfrag {
                qerr[i] = qfrag[i] - c.target[i];
            }
            for b in 0..batch {
                let mut ok = scc_ok_last[b];
                for f in 0..nfrag {
                    if qerr[b * nfrag + f].abs() > q_tol {
                        ok = false;
                    }
                }
                converged[b] = ok;
            }
        }
        let lam = self.plan.cdft.as_ref().unwrap().lam.clone();
        Ok(crate::qmqm::gpu_cdft::CdftReport {
            outer_iters: outer_done,
            q_err_max: qerr.iter().fold(0.0f64, |a, &x| a.max(x.abs())),
            converged,
            qfrag,
            lam,
        })
    }

    /// Fragment excess charges Q_F[b][f] from the current device dq —
    /// [batch*nfrag], f64. Requires an attached constraint set.
    pub fn cdft_qfrag(&mut self) -> Result<Vec<f64>> {
        if self.plan.cdft.is_none() {
            return Err(DftbError::InvalidInput(
                "cdft_qfrag: no constraint set".into(),
            ));
        }
        let mut dq = vec![0.0f32; self.batch * self.n_atoms];
        self.rt.read_buffer(&self.plan.dq, &mut dq)?;
        Ok(self
            .plan
            .cdft
            .as_ref()
            .unwrap()
            .qfrag_from_dq(&dq, self.batch, self.n_atoms))
    }

    /// Constrained-state DFTB energies: eval() reports E_band built
    /// from the shifted h_scc = E_DFTB + Σ_F λ_F·Q_gross(F) — the shift
    /// acts on the GROSS Mulliken population (q0+Δq), so subtract
    /// Σ_F λ_F·(Q_F + Q0_F). Runs an eval (energy only) — call after
    /// cdft_scc converged.
    pub fn cdft_energies(&mut self) -> Result<Vec<f64>> {
        let e = self.eval(false)?.energy;
        let qf = self.cdft_qfrag()?;
        let c = self.plan.cdft.as_ref().unwrap();
        let mut out = e;
        for b in 0..self.batch {
            let mut s = 0.0f64;
            for f in 0..c.nfrag {
                // shift contributes λ·Q_gross to e_band — remove it fully
                let i = b * c.nfrag + f;
                s += c.lam[i] * (qf[i] + c.q0_frag[f]);
            }
            out[b] -= s;
        }
        Ok(out)
    }

    pub fn prof_report(&self, ctx: &str) {
        self.rt
            .prof_report(&format!("GpuDftb {ctx} batch={} n={}", self.batch, self.n));
    }

    /// Frozen-H + energy-identity + optional CPU force/energy (replica 0). Call after `scc`. Prints; does not retune physics.
    pub fn measure(&mut self, want_cpu: bool) -> Result<GpuDftbEval> {
        if self.batch != 1 {
            return Err(DftbError::InvalidInput(format!(
                "measure: batch={} — replica-0 diagnostics only, use batch=1",
                self.batch
            )));
        }
        let ev = self.eval(want_cpu)?;
        // cpu_ref is &mut (syncs device coords) — call before scratch borrows.
        let cpu = if want_cpu {
            Some(self.cpu_ref()?)
        } else {
            None
        };
        let n = self.n;
        let n_atoms = self.n_atoms;
        let n_occ = self.n_occ;
        let nn = n * n;
        let mut h0 = vec![0.0f32; nn];
        let mut s = vec![0.0f32; nn];
        let mut hscc = vec![0.0f32; nn];
        let mut c = vec![0.0f32; nn];
        let mut d = vec![0.0f32; nn];
        let mut xlow = vec![0.0f32; nn];
        let mut cp = vec![0.0f32; nn];
        let mut v = vec![0.0f32; n_atoms];
        let mut g = vec![0.0f32; n_atoms * n_atoms];
        self.rt.read_buffer(&self.buf_h0, &mut h0)?;
        self.rt.read_buffer(&self.buf_s, &mut s)?;
        self.rt.read_buffer(&self.plan.h_scc, &mut hscc)?;
        self.rt.read_buffer(&self.plan.c, &mut c)?;
        self.rt.read_buffer(&self.plan.d, &mut d)?;
        self.rt.read_buffer(&self.plan.x_buf, &mut xlow)?;
        self.rt.read_buffer(&self.plan.cp, &mut cp)?;
        self.rt.read_buffer(&self.plan.v, &mut v)?;
        self.rt.read_buffer(&self.buf_g, &mut g)?;
        self.rt.read_buffer(&self.plan.q_gpu, &mut self.scratch_q)?;
        self.rt
            .read_buffer(&self.plan.q_new, &mut self.scratch_qd)?;
        self.rt
            .read_buffer(&self.plan.eig_diag, &mut self.plan.eig_diag_host)?;
        self.rt
            .read_buffer(&self.plan.occ_mask, &mut self.plan.mask_host)?;
        if self.plan.kT > 0.0 {
            // R5: the integer mask is not maintained under Fermi smearing —
            // occ_w is the occupation state; w>0.5 is the occupied subset
            // for the residual/parity diagnostics below.
            self.rt
                .read_buffer(&self.plan.occ_w, &mut self.plan.occ_w_host)?;
            for k in 0..n {
                self.plan.mask_host[k] = (self.plan.occ_w_host[k] > 0.5) as i32;
            }
        }
        let eig = &self.plan.eig_diag_host[..n];
        let mask = &self.plan.mask_host[..n];
        let q_in = &self.scratch_q[..n_atoms];
        let q_d = &self.scratch_qd[..n_atoms];

        let (lmin_s, lmax_s) = {
            let (lam, _) = sym_eig_f32(&s, n);
            (
                lam.iter().copied().fold(f64::INFINITY, f64::min),
                lam.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        let hc_all = residual_hc_sce(&hscc, &s, &c, eig, n, None);
        let hc_occ = residual_hc_sce(&hscc, &s, &c, eig, n, Some(mask));
        let ctsc = residual_ctsc(&s, &c, n);
        let xtsx = residual_ctsc(&s, &xlow, n);
        let ctcp = residual_eye(&cp, n);
        // D1: decompose δ_CH into normalization vs residual parts; per-column
        // metric defects for C (S-metric) and C' (plain orthonormality).
        let d1 = decompose_occ(&hscc, &s, &c, &cp, eig, mask, n);

        let (lam_r, c_r) = gevp_lowdin_f32(&hscc, &s, n)?;
        let mut eps_gpu_occ: Vec<f64> = (0..n)
            .filter(|&k| mask[k] != 0)
            .map(|k| eig[k] as f64)
            .collect();
        eps_gpu_occ.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if eps_gpu_occ.len() != n_occ {
            return Err(DftbError::InvalidInput(format!(
                "measure: occupied mask count {} != n_occ {n_occ}",
                eps_gpu_occ.len()
            )));
        }
        let mut de_occ = 0.0f64;
        for k in 0..n_occ {
            de_occ = de_occ.max((eps_gpu_occ[k] - lam_r[k]).abs());
        }
        let p_diff = projector_diff(&c, mask, &c_r, n, n_occ);

        let mut e_band = 0.0f64;
        for k in 0..n {
            if mask[k] != 0 {
                e_band += 2.0 * eig[k] as f64;
            }
        }
        let tr_d_h0 = frobenius_f32(&d, &h0, n);
        let tr_d_h = frobenius_f32(&d, &hscc, n);
        let e_ch = occ_quad_h(&hscc, &c, mask, n);
        let d_vs_cc = density_vs_cc(&d, &c, mask, n);
        let delta_ch = e_band - e_ch;
        let delta_d = e_ch - tr_d_h;
        let mut dq_in = vec![0.0f64; n_atoms];
        let mut r = vec![0.0f64; n_atoms];
        let mut dqv = 0.0f64;
        let mut q0v = 0.0f64;
        let mut rdotv = 0.0f64;
        for a in 0..n_atoms {
            dq_in[a] = q_in[a] as f64 - self.q0[a] as f64;
            r[a] = q_d[a] as f64 - q_in[a] as f64;
            dqv += dq_in[a] * v[a] as f64;
            q0v += self.q0[a] as f64 * v[a] as f64;
            rdotv += r[a] * v[a] as f64;
        }
        let half_rgr = quad_g(&g, &r, n_atoms);
        let e_dens = tr_d_h0 + 0.5 * dqv;
        let e_bandform_el = e_band - 0.5 * dqv - q0v;
        let delta_eig = e_band - tr_d_h;
        let e_rep = ev.energy[0] - e_bandform_el;
        let ident = e_bandform_el - e_dens;
        let pred = delta_eig - half_rgr;

        eprintln!("[measure] N={n} n_atoms={n_atoms} n_occ={n_occ} λ_min(S)={lmin_s:.6e} λ_max(S)={lmax_s:.6e} cond={:.3e}", lmax_s / lmin_s);
        eprintln!("[measure] GPU ||HC−SCε||_max all={hc_all:.3e} occ={hc_occ:.3e}  ||CᵀSC−I||_max={ctsc:.3e}  ||XᵀSX−I||_max={xtsx:.3e}  ||C'ᵀC'−I||_max={ctcp:.3e}");
        eprintln!("[measure] frozen rounded GEVP vs GPU Jacobi: max|δε_occ|={de_occ:.3e}  ||P_gpu−P_round||_F={p_diff:.3e}");
        eprintln!("[measure] E_band={e_band:.12} E_bandform_el={e_bandform_el:.12} E_dens={e_dens:.12} E_rep={e_rep:.12} E_tot={:.12}", ev.energy[0]);
        eprintln!("[measure] δ_eig={delta_eig:.3e} = δ_CH(E_band−2ΣCᵀHC)={delta_ch:.3e} + δ_D(2ΣCᵀHC−Tr(DH))={delta_d:.3e}  ||D−2CCᵀ||_F={d_vs_cc:.3e}");
        eprintln!("[measure] D1 δ_CH split: δ_norm(2Σρ(1−n_k))={:.3e} + δ_res(2Σ(ε−ρ))={:.3e} = {:.3e} (should equal δ_CH)", d1.delta_norm, d1.delta_res, d1.delta_norm + d1.delta_res);
        eprintln!("[measure] D1 occ cols: max|cᵀSc−1|={:.3e} max_offdiag|cᵀSc|={:.3e} max|ε−ρ|={:.3e} max r_k={:.3e} || C' occ: max|c'ᵀc'−1|={:.3e} max_offdiag|c'ᵀc'|={:.3e}", d1.max_diag_ctsc, d1.max_offdiag_ctsc, d1.max_eps_rho, d1.max_resid, d1.max_diag_ctcp, d1.max_offdiag_ctcp);
        eprintln!("[measure] r·V={rdotv:.3e} ½rᵀGr={half_rgr:.3e}  bandform−dens={ident:.3e}  predicted(δ_eig−½rGr)={pred:.3e}");
        eprintln!("[measure] q_rms={:.3e} q_max={:.3e}", ev.q_rms, ev.q_max);

        if let Some((e_cpu, f_cpu, q_cpu)) = cpu {
            let de = (ev.energy[0] - e_cpu).abs();
            let f_gpu = ev
                .forces
                .as_ref()
                .expect("measure want_cpu implies eval(true)");
            let mut df = 0.0f64;
            let mut fg = 0.0f64;
            let mut fc = 0.0f64;
            for a in 0..n_atoms {
                for k in 0..3 {
                    let g = f_gpu[3 * a + k] as f64;
                    let c = f_cpu[a][k];
                    df = df.max((g - c).abs());
                    fg = fg.max(g.abs());
                    fc = fc.max(c.abs());
                }
            }
            let mut q_max_cpu = 0.0f64;
            let mut q_s2 = 0.0f64;
            if q_cpu.len() != n_atoms {
                return Err(DftbError::InvalidInput(format!(
                    "cpu_ref q len {} != n_atoms {n_atoms}",
                    q_cpu.len()
                )));
            }
            for a in 0..n_atoms {
                let dq = q_d[a] as f64 - q_cpu[a];
                q_max_cpu = q_max_cpu.max(dq.abs());
                q_s2 += dq * dq;
            }
            let q_rms_cpu = (q_s2 / n_atoms as f64).sqrt();
            eprintln!("[measure] CPU E_tot={e_cpu:.12}  |dE_GPU−CPU|={de:.3e}");
            eprintln!("[measure] max|q_D−q_cpu|={q_max_cpu:.3e}  rms(q_D−q_cpu)={q_rms_cpu:.3e}");
            eprintln!(
                "[measure] max|F_gpu−F_cpu|={df:.3e}  max|F_gpu|={fg:.3e}  max|F_cpu|={fc:.3e}"
            );
            let mut dh0 = 0.0f64;
            let mut ds = 0.0f64;
            {
                let mut cpu = crate::methods::dftb::dftb_cpu::DftbCpu::new(
                    self.sk.clone(),
                    self.species.clone(),
                )?;
                cpu.update_geometry(&self.coords[..n_atoms])?;
                cpu.set_smearing(self.plan.kT as f64);
                for i in 0..n {
                    for j in 0..n {
                        dh0 = dh0.max((h0[i * n + j] as f64 - cpu.h0[(i, j)]).abs());
                        ds = ds.max((s[i * n + j] as f64 - cpu.s[(i, j)]).abs());
                    }
                }
            }
            eprintln!("[measure] assembly vs CPU f64: max|ΔH0|={dh0:.3e} max|ΔS|={ds:.3e}");
        }
        Ok(ev)
    }

    /// CPU f64 SCC + repulsive + forces at replica-0 coords. Independent of GPU charges.
    pub fn cpu_ref(&mut self) -> Result<(f64, Vec<[f64; 3]>, Vec<f64>)> {
        self.sync_coords_to_host()?;
        let n_atoms = self.n_atoms;
        let xyz = &self.coords[..n_atoms];
        let mut cpu =
            crate::methods::dftb::dftb_cpu::DftbCpu::new(self.sk.clone(), self.species.clone())?;
        cpu.update_geometry(xyz)?;
        cpu.set_smearing(self.plan.kT as f64);
        cpu.reset_charges();
        cpu.solve_scc(100, 1e-8)?;
        let scc = cpu.build_result();
        let e_rep =
            crate::methods::dftb::forces::repulsive_energy(&self.sk_dir, &self.species, xyz)?;
        let e = scc.energy + e_rep;
        let unique = unique_species(&self.species);
        let repulsive =
            crate::methods::dftb::forces::parse_all_repulsive(&self.sk_dir, &unique, unique.len())?;
        let forces = cpu.compute_forces(&scc, &repulsive)?;
        if !e.is_finite() {
            return Err(DftbError::InvalidInput(format!(
                "cpu_ref: E={e} non-finite (E_el={} E_rep={e_rep})",
                scc.energy
            )));
        }
        Ok((e, forces.forces, scc.charges))
    }
}

fn unique_species(species: &[String]) -> Vec<String> {
    let mut u = Vec::new();
    for s in species {
        if !u.iter().any(|x| x == s) {
            u.push(s.clone());
        }
    }
    u
}

fn fill_coord_scratch(coords: &[[f64; 3]], ang: &mut [f32], bohr: &mut [f32]) {
    for (i, c) in coords.iter().enumerate() {
        ang[3 * i] = c[0] as f32;
        ang[3 * i + 1] = c[1] as f32;
        ang[3 * i + 2] = c[2] as f32;
        bohr[3 * i] = (c[0] * ANG2BOHR) as f32;
        bohr[3 * i + 1] = (c[1] * ANG2BOHR) as f32;
        bohr[3 * i + 2] = (c[2] * ANG2BOHR) as f32;
    }
}

fn apply_disp(xyz: &mut [f64; 3], v: &[f64; 3], fx: f64, fy: f64, fz: f64, dt: f64) {
    const MAX_DISP: f64 = 0.1; // Å — same cap as CPU FIRE in examples/hbond_ref.rs
    let mut dx = v[0] * dt + 0.5 * fx * dt * dt;
    let mut dy = v[1] * dt + 0.5 * fy * dt * dt;
    let mut dz = v[2] * dt + 0.5 * fz * dt * dt;
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    if d > MAX_DISP {
        let s = MAX_DISP / d;
        dx *= s;
        dy *= s;
        dz *= s;
    }
    xyz[0] += dx;
    xyz[1] += dy;
    xyz[2] += dz;
}

/// DEPRECATED by D7 — kept as the host-side reference for device G parity.
/// G matrix via the D8 species-pair spline — the SAME numerical γ the
/// force kernel's γ′ comes from (energy–force consistent). `atom_sp` is the
/// global species index per atom (replica-0 ordering; batch is homogeneous).
#[allow(dead_code)]
fn fill_gamma(
    coords: &[[f64; 3]],
    spl: &crate::methods::dftb::gamma_spline::GammaSpline,
    atom_sp: &[i32],
    n_atoms: usize,
    batch: usize,
    g: &mut [f32],
) {
    for b in 0..batch {
        let xyz = &coords[b * n_atoms..(b + 1) * n_atoms];
        let base = b * n_atoms * n_atoms;
        for a in 0..n_atoms {
            for c in 0..n_atoms {
                let dx = xyz[a][0] - xyz[c][0];
                let dy = xyz[a][1] - xyz[c][1];
                let dz = xyz[a][2] - xyz[c][2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
                let (gv, _) = spl.eval(r, atom_sp[a] as usize, atom_sp[c] as usize);
                g[base + a * n_atoms + c] = gv as f32;
            }
        }
    }
}

fn pair_cutoffs(sk: &SkData, nsp: usize, unique: &[String]) -> Result<(Vec<f64>, f64)> {
    let mut pair_cut_sq = vec![0.0f64; nsp * nsp];
    let mut sk_cut = 0.0f64;
    for si in 0..nsp {
        for sj in 0..nsp {
            if let Some(tab) = sk.get_pair(&unique[si], &unique[sj]) {
                let c = tab.cutoff();
                pair_cut_sq[si * nsp + sj] = c * c;
                if c > sk_cut {
                    sk_cut = c;
                }
            }
        }
    }
    if sk_cut == 0.0 {
        return Err(DftbError::InvalidInput(
            "GpuDftb: all pair cutoffs are 0 — SK tables missing?".into(),
        ));
    }
    Ok((pair_cut_sq, sk_cut * sk_cut))
}

fn orient_pair(
    i: usize,
    j: usize,
    n_orb_i: usize,
    n_orb_j: usize,
    orb_off_i: u16,
    orb_off_j: u16,
    sp_i: i32,
    sp_j: i32,
    dx: f64,
    dy: f64,
    dz: f64,
    r: f64,
) -> Option<(u8, u16, u16, u16, u16, f32, f32, f32, i32, i32)> {
    let bt = match (n_orb_i, n_orb_j) {
        (1, 1) => 0u8,
        (1, 4) | (4, 1) => 1u8,
        (4, 4) => 2u8,
        _ => return None,
    };
    let inv_r = 1.0 / r;
    if bt == 1 && n_orb_i == 4 && n_orb_j == 1 {
        Some((
            bt,
            j as u16,
            i as u16,
            orb_off_j,
            orb_off_i,
            (-dx * inv_r) as f32,
            (-dy * inv_r) as f32,
            (-dz * inv_r) as f32,
            sp_j,
            sp_i,
        ))
    } else {
        Some((
            bt,
            i as u16,
            j as u16,
            orb_off_i,
            orb_off_j,
            (dx * inv_r) as f32,
            (dy * inv_r) as f32,
            (dz * inv_r) as f32,
            sp_i,
            sp_j,
        ))
    }
}

fn ensure_template_pair_buckets(
    buckets: &mut Vec<GpuPairBucket>,
    tmpl: &FragmentTemplate,
    atom_sp: &[i32],
    sk_tables: &[GpuSkTable],
    nsp: usize,
) -> Result<()> {
    let n = tmpl.n_atoms;
    for i in 0..n {
        for j in (i + 1)..n {
            let n_orb_i = tmpl.atom_n_orb[i] as usize;
            let n_orb_j = tmpl.atom_n_orb[j] as usize;
            let Some((bt, _, _, _, _, _, _, _, s_i, s_j)) = orient_pair(
                i,
                j,
                n_orb_i,
                n_orb_j,
                tmpl.atom_orb_off[i],
                tmpl.atom_orb_off[j],
                atom_sp[i],
                atom_sp[j],
                1.0,
                0.0,
                0.0,
                1.0,
            ) else {
                continue;
            };
            let already = buckets.iter().any(|b| {
                if b.block_type != bt {
                    return false;
                }
                let t = &sk_tables[b.sk_table_idx];
                t.species_i as i32 == s_i && t.species_j as i32 == s_j
            });
            if already {
                continue;
            }
            let sk_table_idx = sk_tables
                .iter()
                .position(|t| t.species_i as i32 == s_i && t.species_j as i32 == s_j)
                .ok_or_else(|| {
                    DftbError::InvalidInput(format!(
                        "GpuDftb: no SK table for template pair species {s_i}-{s_j} block={bt}"
                    ))
                })?;
            buckets.push(GpuPairBucket {
                n_pairs: 0,
                pairs: Vec::new(),
                block_type: bt,
                sk_table_idx,
            });
        }
    }
    let _ = nsp;
    Ok(())
}

fn fragments_from_coords(
    tmpl: &FragmentTemplate,
    coords: &[[f64; 3]],
    batch: usize,
    n_atoms: usize,
) -> Vec<Fragment> {
    (0..batch)
        .map(|b| {
            Fragment::from_template(
                tmpl.clone(),
                coords[b * n_atoms..(b + 1) * n_atoms].to_vec(),
            )
        })
        .collect()
}

fn per_atom_u(sk: &SkData, species: &[String]) -> Result<Vec<f64>> {
    species
        .iter()
        .map(|sp| {
            sk.onsite(sp).map(|p| p.u_hubbard).map_err(|e| {
                DftbError::InvalidInput(format!("Hubbard U missing for species {sp}: {e}"))
            })
        })
        .collect()
}

fn orb_atom_map(atom_orb_off: &[u16], n_orbs: usize) -> Vec<i32> {
    let mut map = vec![0i32; n_orbs];
    for a in 0..atom_orb_off.len() - 1 {
        for mu in atom_orb_off[a] as usize..atom_orb_off[a + 1] as usize {
            map[mu] = a as i32;
        }
    }
    map
}

fn frobenius_f32(a: &[f32], b: &[f32], n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n * n {
        s += a[i] as f64 * b[i] as f64;
    }
    s
}

fn quad_g(g: &[f32], r: &[f64], n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n {
        let mut gi = 0.0f64;
        for j in 0..n {
            gi += g[i * n + j] as f64 * r[j];
        }
        s += r[i] * gi;
    }
    0.5 * s
}

fn residual_hc_sce(
    h: &[f32],
    s: &[f32],
    c: &[f32],
    eig: &[f32],
    n: usize,
    mask: Option<&[i32]>,
) -> f64 {
    let mut mx = 0.0f64;
    for k in 0..n {
        if let Some(m) = mask {
            if m[k] == 0 {
                continue;
            }
        }
        let ek = eig[k] as f64;
        for mu in 0..n {
            let mut hc = 0.0f64;
            let mut sc = 0.0f64;
            for nu in 0..n {
                let ck = c[nu * n + k] as f64;
                hc += h[mu * n + nu] as f64 * ck;
                sc += s[mu * n + nu] as f64 * ck;
            }
            mx = mx.max((hc - sc * ek).abs());
        }
    }
    mx
}

fn residual_eye(c: &[f32], n: usize) -> f64 {
    let mut mx = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += c[k * n + i] as f64 * c[k * n + j] as f64;
            }
            let target = if i == j { 1.0 } else { 0.0 };
            mx = mx.max((acc - target).abs());
        }
    }
    mx
}

fn residual_ctsc(s: &[f32], c: &[f32], n: usize) -> f64 {
    let mut mx = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for mu in 0..n {
                let mut srow = 0.0f64;
                for nu in 0..n {
                    srow += s[mu * n + nu] as f64 * c[nu * n + j] as f64;
                }
                acc += c[mu * n + i] as f64 * srow;
            }
            let target = if i == j { 1.0 } else { 0.0 };
            mx = mx.max((acc - target).abs());
        }
    }
    mx
}

fn mat_from_row(a: &[f32], n: usize) -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_fn(n, n, |i, j| {
        0.5 * (a[i * n + j] as f64 + a[j * n + i] as f64)
    })
}

fn sym_eig_f32(a: &[f32], n: usize) -> (Vec<f64>, nalgebra::DMatrix<f64>) {
    let m = mat_from_row(a, n);
    let ev = nalgebra::SymmetricEigen::new(m);
    let lam: Vec<f64> = ev.eigenvalues.iter().copied().collect();
    // nalgebra does not guarantee order; sort ascending and permute columns.
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&i, &j| lam[i].partial_cmp(&lam[j]).unwrap());
    let mut v = nalgebra::DMatrix::zeros(n, n);
    let mut lam_s = vec![0.0f64; n];
    for (c, &p) in perm.iter().enumerate() {
        lam_s[c] = lam[p];
        for r in 0..n {
            v[(r, c)] = ev.eigenvectors[(r, p)];
        }
    }
    (lam_s, v)
}

fn gevp_lowdin_f32(h: &[f32], s: &[f32], n: usize) -> Result<(Vec<f64>, nalgebra::DMatrix<f64>)> {
    let (lam_s, u) = sym_eig_f32(s, n);
    for (i, &l) in lam_s.iter().enumerate() {
        if !l.is_finite() || l <= 1e-6 {
            return Err(DftbError::InvalidInput(format!(
                "gevp_lowdin: λ_S[{i}]={l} non-finite or ≤1e-6"
            )));
        }
    }
    let mut x = nalgebra::DMatrix::<f64>::zeros(n, n);
    for k in 0..n {
        let srt = 1.0 / lam_s[k].sqrt();
        for i in 0..n {
            for j in 0..n {
                x[(i, j)] += u[(i, k)] * srt * u[(j, k)];
            }
        }
    }
    let hm = mat_from_row(h, n);
    let hp = &x * hm * x.transpose();
    let he = nalgebra::SymmetricEigen::new(hp);
    let lam: Vec<f64> = he.eigenvalues.iter().copied().collect();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&i, &j| lam[i].partial_cmp(&lam[j]).unwrap());
    let mut cp = nalgebra::DMatrix::zeros(n, n);
    let mut lam_s2 = vec![0.0f64; n];
    for (c, &p) in perm.iter().enumerate() {
        lam_s2[c] = lam[p];
        for r in 0..n {
            cp[(r, c)] = he.eigenvectors[(r, p)];
        }
    }
    let c = &x * cp;
    Ok((lam_s2, c))
}

fn projector_diff(
    c_gpu: &[f32],
    mask: &[i32],
    c_cpu: &nalgebra::DMatrix<f64>,
    n: usize,
    n_occ: usize,
) -> f64 {
    let mut p_g = vec![0.0f64; n * n];
    for k in 0..n {
        if mask[k] == 0 {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                p_g[i * n + j] += c_gpu[i * n + k] as f64 * c_gpu[j * n + k] as f64;
            }
        }
    }
    let mut s2 = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut p_c = 0.0f64;
            for k in 0..n_occ {
                p_c += c_cpu[(i, k)] * c_cpu[(j, k)];
            }
            let d = p_g[i * n + j] - p_c;
            s2 += d * d;
        }
    }
    s2.sqrt()
}

/// 2 Σ_{k occ} C_kᵀ H C_k from downloaded f32 arrays, summed in f64.
fn occ_quad_h(h: &[f32], c: &[f32], mask: &[i32], n: usize) -> f64 {
    let mut s = 0.0f64;
    for k in 0..n {
        if mask[k] == 0 {
            continue;
        }
        let mut acc = 0.0f64;
        for i in 0..n {
            let mut hi = 0.0f64;
            for j in 0..n {
                hi += h[i * n + j] as f64 * c[j * n + k] as f64;
            }
            acc += c[i * n + k] as f64 * hi;
        }
        s += 2.0 * acc;
    }
    s
}

/// ||D − 2 C_occ C_occᵀ||_F
fn density_vs_cc(d: &[f32], c: &[f32], mask: &[i32], n: usize) -> f64 {
    let mut s2 = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut p = 0.0f64;
            for k in 0..n {
                if mask[k] != 0 {
                    p += c[i * n + k] as f64 * c[j * n + k] as f64;
                }
            }
            let diff = d[i * n + j] as f64 - 2.0 * p;
            s2 += diff * diff;
        }
    }
    s2.sqrt()
}

/// D1 decomposition of δ_CH for occupied states (GPT 5.6 review, manifest §12).
///
/// δ_CH = 2Σε_k − 2Σc_kᵀHc_k. With n_k=c_kᵀSc_k and ρ_k=(c_kᵀHc_k)/n_k this
/// splits *exactly* as δ_CH = δ_res + δ_norm:
///   δ_res  = 2Σ(ε_k − ρ_k)      — true eigen-equation error (Rayleigh quotient)
///   δ_norm = 2Σρ_k(1 − n_k)     — pure S-normalization defect of the columns
/// Also reports the plain-orthonormality defect of the orthogonal-basis C′
/// columns, and the per-state residual ||Hc_k − ε_kSc_k||/||H||_F.
struct OccDecomp {
    delta_norm: f64,
    delta_res: f64,
    max_diag_ctsc: f64,
    max_offdiag_ctsc: f64,
    max_eps_rho: f64,
    max_resid: f64,
    max_diag_ctcp: f64,
    max_offdiag_ctcp: f64,
}

fn decompose_occ(
    h: &[f32],
    s: &[f32],
    c: &[f32],
    cp: &[f32],
    eig: &[f32],
    mask: &[i32],
    n: usize,
) -> OccDecomp {
    // sc = S·C, hc = H·C (f64 diagnostics, O(n³) once per measure call)
    let mut sc = vec![0.0f64; n * n];
    let mut hc = vec![0.0f64; n * n];
    let mut hnf = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            hnf += (h[i * n + j] as f64).powi(2);
        }
    }
    hnf = hnf.sqrt().max(1e-30);
    for i in 0..n {
        for k in 0..n {
            let mut sa = 0.0f64;
            let mut ha = 0.0f64;
            for nu in 0..n {
                sa += s[i * n + nu] as f64 * c[nu * n + k] as f64;
                ha += h[i * n + nu] as f64 * c[nu * n + k] as f64;
            }
            sc[i * n + k] = sa;
            hc[i * n + k] = ha;
        }
    }
    let mut occ: Vec<usize> = (0..n).filter(|&k| mask[k] != 0).collect();
    occ.sort_unstable();
    let mut d = OccDecomp {
        delta_norm: 0.0,
        delta_res: 0.0,
        max_diag_ctsc: 0.0,
        max_offdiag_ctsc: 0.0,
        max_eps_rho: 0.0,
        max_resid: 0.0,
        max_diag_ctcp: 0.0,
        max_offdiag_ctcp: 0.0,
    };
    for (ii, &k) in occ.iter().enumerate() {
        let mut n_k = 0.0f64;
        let mut q_k = 0.0f64;
        let mut n2_cp = 0.0f64;
        for mu in 0..n {
            n_k += c[mu * n + k] as f64 * sc[mu * n + k];
            q_k += c[mu * n + k] as f64 * hc[mu * n + k];
            n2_cp += (cp[mu * n + k] as f64).powi(2);
        }
        let rho = q_k / n_k;
        let eps = eig[k] as f64;
        d.delta_res += 2.0 * (eps - rho);
        d.delta_norm += 2.0 * rho * (1.0 - n_k);
        d.max_diag_ctsc = d.max_diag_ctsc.max((n_k - 1.0).abs());
        d.max_diag_ctcp = d.max_diag_ctcp.max((n2_cp - 1.0).abs());
        d.max_eps_rho = d.max_eps_rho.max((eps - rho).abs());
        let mut r2 = 0.0f64;
        for mu in 0..n {
            r2 += (hc[mu * n + k] - eps * sc[mu * n + k]).powi(2);
        }
        d.max_resid = d.max_resid.max(r2.sqrt() / hnf);
        for &l in occ.iter().skip(ii + 1) {
            let mut kl = 0.0f64;
            let mut kl_cp = 0.0f64;
            for mu in 0..n {
                kl += c[mu * n + k] as f64 * sc[mu * n + l];
                kl_cp += cp[mu * n + k] as f64 * cp[mu * n + l] as f64;
            }
            d.max_offdiag_ctsc = d.max_offdiag_ctsc.max(kl.abs());
            d.max_offdiag_ctcp = d.max_offdiag_ctcp.max(kl_cp.abs());
        }
    }
    d
}
