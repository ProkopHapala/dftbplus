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
use crate::methods::dftb::gamma::{gamma_full, GammaTable};
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

/// Result of one `scc` call.
#[derive(Debug, Clone)]
pub struct GpuDftbScc {
    pub n_iters: usize,
    pub rms: f32,
    pub stalled: bool,
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
    pair_cap: usize,

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
    buf_edm: Buffer<f32>,
    buf_coords_ang: Buffer<f32>,
    buf_coords_bohr: Buffer<f32>,
    buf_u_hub: Buffer<f32>,
    buf_rep_off: Buffer<i32>,
    buf_rep_data: Buffer<f32>,

    k_onsite: Kernel,
    k_gamma_f: Kernel,
    k_rep_force: Kernel,
    buckets: Vec<PairBucket>,

    s_ident: Vec<f32>,
    q0: Vec<f32>,
    u_per_atom: Vec<f64>,
    coords: Vec<[f64; 3]>,
    fire_v: Vec<[f64; 3]>,
    fire_dt: f64,
    fire_alpha: f64,
    fire_n_pos: usize,

    // Host scratch (per-geometry / per-force; never reallocated)
    scratch_h0: Vec<f32>,
    scratch_v: Vec<f32>,
    scratch_g: Vec<f32>,
    scratch_ang: Vec<f32>,
    scratch_bohr: Vec<f32>,
    scratch_f: Vec<f32>,
    scratch_f0: Vec<f32>,
    scratch_q: Vec<f32>,
    scratch_qd: Vec<f32>,

    atom_sp: Vec<i32>,
    atom_n_orb: Vec<u8>,
    atom_orb_off: Vec<u16>,
    pair_cut_sq: Vec<f64>,
    sk_cut_sq: f64,
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
    k_assemble: Kernel,
    k_force: Kernel,
    k_shift: Kernel,
    dr: f32,
    n_grid: i32,
    n_sk_cols: i32,
    block_type: i32,
    sp_i: i32,
    sp_j: i32,
    n_live: usize,
    pair_staging: Vec<GpuPairEntry>,
}

impl GpuDftb {
    /// Build the engine for a homogeneous batch. `coords.len() == batch * n_atoms`, Å.
    /// `sk_dir` is the Slater-Koster folder (repulsive Spline sections). Compile/alloc once.
    pub fn new(sk: SkData, sk_dir: &str, species: Vec<String>, coords: Vec<[f64; 3]>, batch: usize) -> Result<Self> {
        if batch == 0 { return Err(DftbError::InvalidInput("GpuDftb: batch=0".into())); }
        let n_atoms = species.len();
        if n_atoms == 0 { return Err(DftbError::InvalidInput("GpuDftb: no atoms".into())); }
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
        let force_src = FORCE_SOURCE.replace("#define REP_MAX_INTERVALS 30", &format!("#define REP_MAX_INTERVALS {}", max_int.max(1)));

        let tmpl = FragmentTemplate::new(&sk, species.clone(), coords[..n_atoms].to_vec())?;
        let n = tmpl.n_orbs;
        let n_el = tmpl.q0.iter().sum::<f64>();
        let n_occ = (n_el / 2.0).round() as usize;
        if n_occ == 0 || n_occ > n {
            return Err(DftbError::InvalidInput(format!("GpuDftb: n_occ={n_occ} invalid for N={n} n_el={n_el}")));
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
                gpu_batch.n_frags, gpu_batch.total_h_elements, batch, n * n
            )));
        }
        if gpu_batch.n_global_species != unique.len() {
            return Err(DftbError::InvalidInput(format!(
                "GpuDftb: global species {} != unique {} — repulsive pack order would not match atom_species",
                gpu_batch.n_global_species, unique.len()
            )));
        }

        let pair_cap = (batch * n_atoms * n_atoms).max(PAIR_WG);
        let ham = rt.build_program(HAM_SOURCE)?;
        let force = rt.build_program(&force_src)?;

        let buf_h0 = rt.zero_buffer::<f32>(batch * n * n)?;
        let s_ident = build_s_identity_init(&gpu_batch.fragments, gpu_batch.total_h_elements);
        let buf_s = rt.buffer_from_slice(&s_ident)?;
        let mut g_host = vec![0.0f32; batch * n_atoms * n_atoms];
        fill_gamma(&coords, &per_atom_u(&sk, &species)?, n_atoms, batch, &mut g_host);
        let buf_g = rt.buffer_from_slice(&g_host)?;
        let q0: Vec<f32> = tmpl.q0.iter().map(|&q| q as f32).collect::<Vec<_>>().repeat(batch);
        let buf_q0 = rt.buffer_from_slice(&q0)?;
        // h_scc / Mulliken index orb_atom as [batch][N] (`oa + sid*n`). One copy is OOB for replica>0.
        let oa1 = orb_atom_map(&tmpl.atom_orb_off, n);
        if oa1.len() != n {
            return Err(DftbError::InvalidInput(format!("GpuDftb: orb_atom len {} != N={n}", oa1.len())));
        }
        let oa: Vec<i32> = oa1.repeat(batch);
        let buf_oa = rt.buffer_from_slice(&oa)?;
        let buf_fragments = rt.buffer_from_slice(&gpu_batch.fragments)?;
        let buf_atom_species = rt.buffer_from_slice(&gpu_batch.atom_species)?;
        let buf_orb_off = rt.buffer_from_slice(&gpu_batch.atom_orb_off)?;
        let n_orb = build_n_orb_per_atom(&gpu_batch.fragments, &gpu_batch.atom_orb_off);
        let buf_n_orb = rt.buffer_from_slice(&n_orb)?;
        let onsite: Vec<Float2> = (0..gpu_batch.n_global_species)
            .map(|i| Float2::new(gpu_batch.onsite_es_ep[2 * i], gpu_batch.onsite_es_ep[2 * i + 1]))
            .collect();
        let buf_onsite = rt.buffer_from_slice(&onsite)?;
        let buf_hubbard = rt.buffer_from_slice(&gpu_batch.hubbard_u)?;
        let buf_v_asm = rt.zero_buffer::<f32>(gpu_batch.total_atoms)?;
        let buf_charges_zero = rt.zero_buffer::<f32>(gpu_batch.charges.len())?;
        let buf_forces = rt.zero_buffer::<f32>(3 * gpu_batch.total_atoms)?;
        let buf_edm = rt.zero_buffer::<f32>(batch * n * n)?;
        let coords_ang: Vec<f32> = coords.iter().flat_map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect();
        let buf_coords_ang = rt.buffer_from_slice(&coords_ang)?;
        let coords_bohr: Vec<f32> = coords.iter().flat_map(|c| [(c[0] * ANG2BOHR) as f32, (c[1] * ANG2BOHR) as f32, (c[2] * ANG2BOHR) as f32]).collect();
        let buf_coords_bohr = rt.buffer_from_slice(&coords_bohr)?;
        let u_per_atom = per_atom_u(&sk, &species)?;
        let u_hub: Vec<f32> = gpu_batch.hubbard_u.clone();
        let buf_u_hub = rt.buffer_from_slice(&u_hub)?;
        let buf_rep_off = rt.buffer_from_slice(&rep_off)?;
        let buf_rep_data = rt.buffer_from_slice(&rep_data)?;

        let total_atoms = gpu_batch.total_atoms;
        let n_frags = gpu_batch.n_frags as i32;
        let n_species = unique.len() as i32;
        let wg_onsite = 64.min(total_atoms.max(1));
        let g_onsite = ((total_atoms + wg_onsite - 1) / wg_onsite) * wg_onsite;
        let k_onsite = Kernel::builder()
            .program(&ham).name("onsite_diagonal").queue(rt.queue().clone())
            .global_work_size(g_onsite).local_work_size(wg_onsite)
            .arg(&buf_fragments).arg(&buf_atom_species).arg(&buf_orb_off).arg(&buf_n_orb)
            .arg(&buf_onsite).arg(&buf_h0).arg(n_frags).arg(total_atoms as i32)
            .build().map_err(map_ocl_err)?;
        let k_gamma_f = Kernel::builder()
            .program(&force).name("force_gamma_deriv_batched").queue(rt.queue().clone())
            .global_work_size(batch * 256).local_work_size(256)
            .arg(n_atoms as i32).arg(batch as i32)
            .arg(&buf_coords_ang).arg(&buf_atom_species).arg(&buf_v_asm).arg(&buf_u_hub)
            .arg(n_species).arg(&buf_forces)
            .build().map_err(map_ocl_err)?;
        let k_rep_force = Kernel::builder()
            .program(&force).name("force_repulsive_batched").queue(rt.queue().clone())
            .global_work_size(batch * 256).local_work_size(256)
            .arg(n_atoms as i32).arg(batch as i32)
            .arg(&buf_coords_bohr).arg(&buf_atom_species).arg(&buf_rep_off)
            .arg(n_species).arg(&buf_rep_data).arg(&buf_forces)
            .build().map_err(map_ocl_err)?;

        let atom_sp: Vec<i32> = gpu_batch.atom_species[..n_atoms].to_vec();
        let mut pair_buckets = gpu_batch.pair_buckets.clone();
        ensure_template_pair_buckets(&mut pair_buckets, &tmpl, &atom_sp, &gpu_batch.sk_tables, unique.len())?;
        let (pair_cut_sq, sk_cut_sq) = pair_cutoffs(&sk, unique.len(), &unique)?;

        let mut buckets = Vec::with_capacity(pair_buckets.len());
        for bkt in &pair_buckets {
            let skt = &gpu_batch.sk_tables[bkt.sk_table_idx];
            if bkt.n_pairs > pair_cap {
                return Err(DftbError::InvalidInput(format!(
                    "GpuDftb::new: n_pairs={} > pair_cap={} (batch={} n_atoms={}) — increase cap / shrink batch",
                    bkt.n_pairs, pair_cap, batch, n_atoms
                )));
            }
            let mut staging = vec![GpuPairEntry::default(); pair_cap];
            staging[..bkt.n_pairs].copy_from_slice(&bkt.pairs);
            let buf_pairs = rt.buffer_from_slice(&staging)?;
            let buf_sk_h = rt.buffer_from_slice(&skt.sk_h)?;
            let buf_sk_s = rt.buffer_from_slice(&skt.sk_s)?;
            let gws = ((pair_cap + PAIR_WG - 1) / PAIR_WG) * PAIR_WG;
            let k_assemble = Kernel::builder()
                .program(&ham).name("assemble_pairs").queue(rt.queue().clone())
                .global_work_size(gws).local_work_size(PAIR_WG)
                .arg(&buf_pairs).arg(&buf_fragments).arg(&buf_sk_h).arg(&buf_sk_s)
                .arg(&buf_v_asm).arg(&buf_h0).arg(&buf_s)
                .arg(skt.dr).arg(skt.n_grid as i32).arg(bkt.n_pairs as i32)
                .arg(n_frags).arg(bkt.block_type as i32).arg(skt.n_sk_cols as i32)
                .build().map_err(map_ocl_err)?;
            let k_force = Kernel::builder()
                .program(&force).name("force_pairs").queue(rt.queue().clone())
                .global_work_size(gws).local_work_size(PAIR_WG)
                .arg(&buf_pairs).arg(&buf_fragments).arg(&buf_sk_h).arg(&buf_sk_s)
                .arg(&buf_h0).arg(&buf_edm).arg(&buf_forces)
                .arg(skt.dr).arg(skt.n_grid as i32).arg(bkt.n_pairs as i32)
                .arg(n_frags).arg(bkt.block_type as i32).arg(skt.n_sk_cols as i32)
                .build().map_err(map_ocl_err)?;
            let k_shift = Kernel::builder()
                .program(&force).name("force_pairs_scc_shift").queue(rt.queue().clone())
                .global_work_size(gws).local_work_size(PAIR_WG)
                .arg(&buf_pairs).arg(&buf_fragments).arg(&buf_sk_s)
                .arg(&buf_h0).arg(&buf_v_asm).arg(&buf_forces)
                .arg(skt.dr).arg(skt.n_grid as i32).arg(bkt.n_pairs as i32)
                .arg(n_frags).arg(bkt.block_type as i32).arg(skt.n_sk_cols as i32)
                .build().map_err(map_ocl_err)?;
            buckets.push(PairBucket {
                buf_pairs, buf_sk_h, buf_sk_s, k_assemble, k_force, k_shift,
                dr: skt.dr, n_grid: skt.n_grid as i32, n_sk_cols: skt.n_sk_cols as i32,
                block_type: bkt.block_type as i32,
                sp_i: skt.species_i as i32, sp_j: skt.species_j as i32,
                n_live: bkt.n_pairs, pair_staging: staging,
            });
        }
        let nsp = unique.len();
        let mut bucket_lut = vec![-1i32; 3 * nsp * nsp];
        for (i, slot) in buckets.iter().enumerate() {
            let k = slot.block_type as usize * nsp * nsp + slot.sp_i as usize * nsp + slot.sp_j as usize;
            if k >= bucket_lut.len() {
                return Err(DftbError::InvalidInput(format!("GpuDftb: bucket lut index {k} >= {}", bucket_lut.len())));
            }
            bucket_lut[k] = i as i32;
        }

        let atom_n_orb = tmpl.atom_n_orb.clone();
        let atom_orb_off = tmpl.atom_orb_off.clone();
        let plan = GpuSccPlan::new(&mut rt, &buf_s, n, n_atoms, batch)
            .map_err(|e| DftbError::InvalidInput(format!("GpuSccPlan::new (Lowdin X from S): {e}")))?;
        let nn = n * n;
        let mut eng = Self {
            rt, sk, sk_dir: sk_dir.to_string(), species, tmpl, gamma_tbl, n, n_atoms, n_occ, batch, pair_cap,
            buf_h0, buf_s, buf_g, buf_q0, buf_oa, buf_fragments, buf_atom_species,
            buf_orb_off, buf_n_orb, buf_onsite, buf_hubbard, buf_v_asm, buf_charges_zero,
            buf_forces, buf_edm, buf_coords_ang, buf_coords_bohr, buf_u_hub, buf_rep_off, buf_rep_data,
            k_onsite, k_gamma_f, k_rep_force, buckets, s_ident, q0,
            u_per_atom, coords,
            fire_v: vec![[0.0; 3]; batch * n_atoms],
            fire_dt: 1.0, fire_alpha: 0.1, fire_n_pos: 0,
            scratch_h0: vec![0.0; batch * nn],
            scratch_v: vec![0.0; batch * n_atoms],
            scratch_g: vec![0.0; batch * n_atoms * n_atoms],
            scratch_ang: coords_ang,
            scratch_bohr: coords_bohr.clone(),
            scratch_f: vec![0.0; 3 * batch * n_atoms],
            scratch_f0: vec![0.0; 3 * batch * n_atoms],
            scratch_q: vec![0.0; batch * n_atoms],
            scratch_qd: vec![0.0; batch * n_atoms],
            atom_sp,
            atom_n_orb,
            atom_orb_off,
            pair_cut_sq, sk_cut_sq, bucket_lut, nsp,
            n_species,
            plan, _ham: ham, _force: force,
        };
        eng.plan.set_repulsive_splines(&mut eng.rt, &coords_bohr, &gpu_batch.atom_species, &rep_off, &rep_data, unique.len(), max_int)
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb set_repulsive_splines: {e}")))?;
        eng.assemble()
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb initial assemble: {e}")))?;
        eng.plan.set_geometry(&mut eng.rt, &eng.buf_s)
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb set_geometry after assemble: {e}")))?;
        eng.plan.reset_diis(&eng.rt)
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb reset_diis: {e}")))?;
        eng.plan.set_initial_charges(&eng.rt, &eng.q0)
            .map_err(|e| DftbError::InvalidInput(format!("GpuDftb set_initial_charges: {e}")))?;
        Ok(eng)
    }

    pub fn n(&self) -> usize { self.n }
    pub fn n_atoms(&self) -> usize { self.n_atoms }
    pub fn n_occ(&self) -> usize { self.n_occ }
    pub fn batch(&self) -> usize { self.batch }
    pub fn coords(&self) -> &[[f64; 3]] { &self.coords }

    /// Per-geometry update. Refills pair r,l,m,n into existing staging; GPU kernels/buffers stay.
    /// Fails loud if a pair's (block, species) slot was not allocated at `new`, or `n_pairs > pair_cap`.
    pub fn set_coords(&mut self, coords: &[[f64; 3]]) -> Result<()> {
        if coords.len() != self.batch * self.n_atoms {
            return Err(DftbError::InvalidInput(format!(
                "set_coords: len {} != batch*n_atoms {}*{}", coords.len(), self.batch, self.n_atoms
            )));
        }
        self.coords.copy_from_slice(coords);
        self.refill_pairs()?;
        self.assemble()?;
        self.plan.set_geometry(&mut self.rt, &self.buf_s)?;
        self.plan.reset_diis(&self.rt)?;
        Ok(())
    }

    fn refill_pairs(&mut self) -> Result<()> {
        for slot in &mut self.buckets { slot.n_live = 0; }
        let n_atoms = self.n_atoms;
        let nsp = self.nsp;
        let pair_cap = self.pair_cap;
        let sk_cut_sq = self.sk_cut_sq;
        let batch = self.batch;
        {
            let coords = &self.coords;
            let atom_sp = &self.atom_sp;
            let atom_n_orb = &self.atom_n_orb;
            let atom_orb_off = &self.atom_orb_off;
            let pair_cut_sq = &self.pair_cut_sq;
            let lut = &self.bucket_lut;
            let buckets = &mut self.buckets;
            for b in 0..batch {
                let base = b * n_atoms;
                for i in 0..n_atoms {
                    let pi = coords[base + i];
                    let n_orb_i = atom_n_orb[i] as usize;
                    for j in (i + 1)..n_atoms {
                        let pj = coords[base + j];
                        let dx = (pj[0] - pi[0]) * ANG2BOHR;
                        let dy = (pj[1] - pi[1]) * ANG2BOHR;
                        let dz = (pj[2] - pi[2]) * ANG2BOHR;
                        let r2 = dx * dx + dy * dy + dz * dz;
                        if r2 > sk_cut_sq { continue; }
                        let spi = atom_sp[i] as usize;
                        let spj = atom_sp[j] as usize;
                        let cut = pair_cut_sq[spi * nsp + spj];
                        if cut == 0.0 || r2 > cut { continue; }
                        let n_orb_j = atom_n_orb[j] as usize;
                        let Some((bt, atom_i, atom_j, orb_i, orb_j, l, m, n_val, s_i, s_j)) =
                            orient_pair(i, j, n_orb_i, n_orb_j, atom_orb_off[i], atom_orb_off[j],
                                atom_sp[i], atom_sp[j], dx, dy, dz, r2.sqrt())
                        else { continue };
                        let key = bt as usize * nsp * nsp + s_i as usize * nsp + s_j as usize;
                        let bi = lut[key];
                        if bi < 0 {
                            return Err(DftbError::InvalidInput(format!(
                                "set_coords: pair atom {i}-{j} replica {b} block={bt} species {s_i}-{s_j} has no bucket (not allocated at GpuDftb::new). Rebuild new."
                            )));
                        }
                        let slot = &mut buckets[bi as usize];
                        if slot.n_live >= pair_cap {
                            return Err(DftbError::InvalidInput(format!(
                                "set_coords: n_pairs > pair_cap={pair_cap} at bucket {bi} replica {b}. Rebuild GpuDftb."
                            )));
                        }
                        slot.pair_staging[slot.n_live] = GpuPairEntry {
                            replica: b as u32, atom_i, atom_j, orb_i, orb_j,
                            r: r2.sqrt() as f32, l, m, n: n_val,
                        };
                        slot.n_live += 1;
                    }
                }
            }
        }
        for (i, slot) in self.buckets.iter_mut().enumerate() {
            if slot.n_live == 0 { continue; }
            self.rt.write_buffer(&slot.buf_pairs, &slot.pair_staging)
                .map_err(|e| DftbError::InvalidInput(format!("write pairs bucket {i} n_live={}: {e}", slot.n_live)))?;
            slot.k_assemble.set_arg(9u32, slot.n_live as i32).map_err(map_ocl_err)?;
            slot.k_force.set_arg(9u32, slot.n_live as i32).map_err(map_ocl_err)?;
            slot.k_shift.set_arg(8u32, slot.n_live as i32).map_err(map_ocl_err)?;
        }
        Ok(())
    }

    fn assemble(&mut self) -> Result<()> {
        fill_gamma(&self.coords, &self.u_per_atom, self.n_atoms, self.batch, &mut self.scratch_g);
        self.rt.write_buffer(&self.buf_g, &self.scratch_g)
            .map_err(|e| DftbError::InvalidInput(format!("write G: {e}")))?;
        self.scratch_h0.fill(0.0);
        self.rt.write_buffer(&self.buf_h0, &self.scratch_h0)
            .map_err(|e| DftbError::InvalidInput(format!("zero H0: {e}")))?;
        self.rt.write_buffer(&self.buf_s, &self.s_ident)
            .map_err(|e| DftbError::InvalidInput(format!("write S=I: {e}")))?;
        self.scratch_v.fill(0.0);
        self.rt.write_buffer(&self.buf_v_asm, &self.scratch_v)
            .map_err(|e| DftbError::InvalidInput(format!("zero V_asm: {e}")))?;
        unsafe { self.k_onsite.enq().map_err(|e| DftbError::InvalidInput(format!("onsite_diagonal: {e}")))?; }
        self.rt.finish()?;
        for (i, slot) in self.buckets.iter().enumerate() {
            if slot.n_live == 0 { continue; }
            unsafe { slot.k_assemble.enq().map_err(|e| DftbError::InvalidInput(format!("assemble_pairs bucket {i}: {e}")))?; }
        }
        self.rt.finish()?;
        fill_coord_scratch(&self.coords, &mut self.scratch_ang, &mut self.scratch_bohr);
        self.rt.write_buffer(&self.buf_coords_ang, &self.scratch_ang)?;
        self.rt.write_buffer(&self.buf_coords_bohr, &self.scratch_bohr)?;
        self.plan.set_repulsive_coords(&self.rt, &self.scratch_bohr)?;
        Ok(())
    }

    /// SCC mixer for Package 2 A/B: 0 = GPU DIIS (production), 1 = GPU simple mix, 2 = host f64 DIIS (same electronic kernels).
    pub fn scc(&mut self, max_iter: usize, rms_tol: f32) -> Result<GpuDftbScc> {
        self.scc_mix(max_iter, rms_tol, 0)
    }

    pub fn reset_q0(&mut self) -> Result<()> {
        self.plan.reset_diis(&self.rt)?;
        self.plan.set_initial_charges(&self.rt, &self.q0)
    }

    /// `mix`: 0 GPU DIIS, 1 GPU simple α=0.3, 2 host f64 DiisMixer (hist=min(10,n_atoms), warmup=1).
    pub fn scc_mix(&mut self, max_iter: usize, rms_tol: f32, mix: i32) -> Result<GpuDftbScc> {
        if mix == 2 && self.batch != 1 {
            return Err(DftbError::InvalidInput(format!("scc_mix host DIIS: batch={} — measurement path is replica-0 only", self.batch)));
        }
        let mut rms = f32::INFINITY;
        let mut n_iters = 0;
        let mut hist = [f32::INFINITY; 10];
        let mut stalled = false;
        let mut host = if mix == 2 {
            let mut m = crate::qmqm::mixer::DiisMixer::new(self.n_atoms.min(10), self.n_atoms);
            m.alpha = 0.3;
            m.warmup = 1;
            Some(m)
        } else { None };
        let mut q_in = vec![0.0f64; self.n_atoms];
        let mut q_out = vec![0.0f64; self.n_atoms];
        let mut res = vec![0.0f64; self.n_atoms];
        let mut q_f32 = vec![0.0f32; self.n_atoms];
        let cap = max_iter;
        eprintln!("[GpuDftb] scc_mix mix={mix} (0=GPU DIIS hist={}, 1=GPU simple, 2=host f64 DIIS hist={}) max_iter={max_iter} rms_tol={rms_tol:.3e}", self.plan.diis_max_hist, self.n_atoms.min(10));
        for it in 0..cap {
            n_iters = it + 1;
            rms = match mix {
                0 => self.plan.scc_step_diis(
                    &mut self.rt, &self.buf_h0, &self.buf_s, &self.buf_g, &self.buf_q0, &self.buf_oa,
                    self.n_occ, 0.3,
                )?,
                1 => self.plan.scc_step(
                    &mut self.rt, &self.buf_h0, &self.buf_s, &self.buf_g, &self.buf_q0, &self.buf_oa,
                    self.n_occ, 0.3,
                )?,
                2 => {
                    self.plan.finalize(
                        &mut self.rt, &self.buf_h0, &self.buf_s, &self.buf_g, &self.buf_q0, &self.buf_oa, self.n_occ,
                    )?;
                    self.rt.read_buffer(&self.plan.q_gpu, &mut q_f32)?;
                    for a in 0..self.n_atoms { q_in[a] = q_f32[a] as f64; }
                    self.rt.read_buffer(&self.plan.q_new, &mut q_f32)?;
                    let mut s2 = 0.0f64;
                    for a in 0..self.n_atoms {
                        q_out[a] = q_f32[a] as f64;
                        res[a] = q_out[a] - q_in[a];
                        if !res[a].is_finite() {
                            return Err(DftbError::InvalidInput(format!("host DIIS residual non-finite atom {a} q_out={} q_in={}", q_out[a], q_in[a])));
                        }
                        s2 += res[a] * res[a];
                    }
                    let r = (s2 / self.n_atoms as f64).sqrt();
                    if !r.is_finite() { return Err(DftbError::InvalidInput(format!("host DIIS rms={r} non-finite"))); }
                    crate::qmqm::mixer::Mixer::mix(host.as_mut().unwrap(), &mut q_in, &q_out, &res);
                    for a in 0..self.n_atoms {
                        if !q_in[a].is_finite() {
                            return Err(DftbError::InvalidInput(format!("host DIIS mixed q[{a}]={} non-finite", q_in[a])));
                        }
                        q_f32[a] = q_in[a] as f32;
                    }
                    self.plan.set_initial_charges(&self.rt, &q_f32)?;
                    r as f32
                }
                other => return Err(DftbError::InvalidInput(format!("scc_mix: mix={other} not 0/1/2"))),
            };
            hist[n_iters % 10] = rms;
            if rms < rms_tol { break; }
            if n_iters >= 25 {
                let (mut rmin, mut rmax) = (f32::INFINITY, 0.0f32);
                for &r in &hist { rmin = rmin.min(r); rmax = rmax.max(r); }
                if rmin > rms_tol && rmax < 4.0 * rmin {
                    stalled = true;
                    break;
                }
            }
        }
        if n_iters == cap && rms >= rms_tol { stalled = true; }
        Ok(GpuDftbScc { n_iters, rms, stalled })
    }

    /// One finalize: energy always; forces if `want_forces`. Do not call energy() then forces().
    pub fn eval(&mut self, want_forces: bool) -> Result<GpuDftbEval> {
        self.plan.finalize(
            &mut self.rt, &self.buf_h0, &self.buf_s, &self.buf_g, &self.buf_q0, &self.buf_oa, self.n_occ,
        )?;
        let energy = self.plan.energy_from_state(&mut self.rt, &self.buf_q0)?;
        let (q_rms, q_max) = self.charge_residual()?;
        let forces = if want_forces { Some(self.forces_from_state()?) } else { None };
        Ok(GpuDftbEval { energy, forces, q_rms, q_max })
    }

    /// Wrapper: `eval(false)`. Prefer `eval` when you also want forces.
    pub fn energy(&mut self) -> Result<Vec<f64>> { Ok(self.eval(false)?.energy) }

    /// Wrapper: `eval(true)`. Prefer `eval` when you also want energy.
    pub fn forces(&mut self) -> Result<Vec<f32>> {
        self.eval(true)?.forces.ok_or_else(|| DftbError::InvalidInput("eval(true) returned no forces".into()))
    }

    /// `q_D = Mulliken(D,S)` vs `q_in = q_gpu` after finalize. Persistent scratch, no alloc.
    fn charge_residual(&mut self) -> Result<(f64, f64)> {
        self.rt.read_buffer(&self.plan.q_gpu, &mut self.scratch_q)?;
        self.rt.read_buffer(&self.plan.q_new, &mut self.scratch_qd)?;
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
                    return Err(DftbError::InvalidInput(format!("charge residual non-finite replica {b} atom {a} q_D={} q_in={}", self.scratch_qd[b * n_atoms + a], self.scratch_q[b * n_atoms + a])));
                }
                s2 += d * d;
                mx = mx.max(d.abs());
            }
            q_max = q_max.max(mx);
            q_rms = q_rms.max((s2 / n_atoms as f64).sqrt());
        }
        Ok((q_rms, q_max))
    }

    /// Force kernels from the current finalized D/C/ε. Caller must have `finalize`d.
    fn forces_from_state(&mut self) -> Result<Vec<f32>> {
        self.plan.build_edm(&self.buf_edm)?;
        self.scratch_f0.fill(0.0);
        self.rt.write_buffer(&self.buf_forces, &self.scratch_f0)?;
        for slot in &self.buckets {
            if slot.n_live == 0 { continue; }
            slot.k_force.set_arg(4u32, &self.plan.d).map_err(map_ocl_err)?;
            slot.k_force.set_arg(5u32, &self.buf_edm).map_err(map_ocl_err)?;
            unsafe { slot.k_force.enq().map_err(map_ocl_err)?; }
            slot.k_shift.set_arg(3u32, &self.plan.d).map_err(map_ocl_err)?;
            slot.k_shift.set_arg(4u32, &self.plan.v).map_err(map_ocl_err)?;
            unsafe { slot.k_shift.enq().map_err(map_ocl_err)?; }
        }
        self.k_gamma_f.set_arg(2u32, &self.buf_coords_ang).map_err(map_ocl_err)?;
        self.k_gamma_f.set_arg(4u32, &self.plan.dq).map_err(map_ocl_err)?;
        unsafe { self.k_gamma_f.enq().map_err(map_ocl_err)?; }
        self.k_rep_force.set_arg(2u32, &self.buf_coords_bohr).map_err(map_ocl_err)?;
        unsafe { self.k_rep_force.enq().map_err(map_ocl_err)?; }
        self.rt.read_buffer(&self.buf_forces, &mut self.scratch_f)?;
        Ok(self.scratch_f.clone())
    }

    /// One FIRE step on all replicas (Bitzek 2006). Call `scc` first. Returns max |F|.
    pub fn fire_step(&mut self, f_tol: f64) -> Result<f64> {
        let ev = self.eval(true)?;
        let f = ev.forces.as_ref().expect("eval(true) returns forces");
        let ntot = self.batch * self.n_atoms;
        let mut max_f = 0.0f64;
        let mut p = 0.0f64;
        for i in 0..ntot {
            for c in 0..3 {
                let fi = f[3 * i + c] as f64;
                max_f = max_f.max(fi.abs());
                p += fi * self.fire_v[i][c];
            }
        }
        if p > 0.0 {
            self.fire_n_pos += 1;
            if self.fire_n_pos > 10 {
                self.fire_dt = (self.fire_dt * 1.1).min(5.0);
                self.fire_alpha *= 0.95;
            }
        } else {
            self.fire_n_pos = 0;
            self.fire_dt *= 0.7;
            self.fire_alpha = 0.1;
            for v in &mut self.fire_v { *v = [0.0; 3]; }
        }
        if max_f < f_tol { return Ok(max_f); }
        for i in 0..ntot {
            let fx = f[3 * i] as f64; let fy = f[3 * i + 1] as f64; let fz = f[3 * i + 2] as f64;
            let fnorm = (fx * fx + fy * fy + fz * fz).sqrt();
            let (hx, hy, hz) = if fnorm > 1e-12 { (fx / fnorm, fy / fnorm, fz / fnorm) } else { (0.0, 0.0, 0.0) };
            for (c, h) in [hx, hy, hz].iter().enumerate() {
                self.fire_v[i][c] = (1.0 - self.fire_alpha) * self.fire_v[i][c] + self.fire_alpha * fnorm * h;
            }
            let dt = self.fire_dt;
            apply_disp(&mut self.coords[i], &self.fire_v[i], fx, fy, fz, dt);
            self.fire_v[i][0] += fx * dt;
            self.fire_v[i][1] += fy * dt;
            self.fire_v[i][2] += fz * dt;
        }
        let c = self.coords.clone();
        self.set_coords(&c)?;
        Ok(max_f)
    }

    /// One velocity-Verlet step (mass=1, same reduced units as FIRE). Call `scc` first.
    /// Displacement capped at 0.1 Å. Returns max |F|. Caller must `scc` after this.
    pub fn md_step(&mut self, dt: f64) -> Result<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(DftbError::InvalidInput(format!("md_step: dt={dt} must be finite and > 0")));
        }
        let ev = self.eval(true)?;
        let f = ev.forces.as_ref().expect("eval(true) returns forces");
        let ntot = self.batch * self.n_atoms;
        let mut max_f = 0.0f64;
        for i in 0..ntot {
            let fx = f[3 * i] as f64; let fy = f[3 * i + 1] as f64; let fz = f[3 * i + 2] as f64;
            max_f = max_f.max(fx.abs()).max(fy.abs()).max(fz.abs());
            apply_disp(&mut self.coords[i], &self.fire_v[i], fx, fy, fz, dt);
            self.fire_v[i][0] += fx * dt;
            self.fire_v[i][1] += fy * dt;
            self.fire_v[i][2] += fz * dt;
        }
        let c = self.coords.clone();
        self.set_coords(&c)?;
        Ok(max_f)
    }

    /// Relax: SCC + FIRE until max|F|<f_tol or `max_steps`. Prints unbuffered progress.
    pub fn relax(&mut self, max_steps: usize, f_tol: f64, scc_tol: f32) -> Result<(usize, f64, f32)> {
        let scc0 = self.scc(SCC_MAX_ITER, scc_tol)?;
        eprintln!("[GpuDftb] relax start rms={:.3e} iters={} stalled={}", scc0.rms, scc0.n_iters, scc0.stalled);
        let mut max_f = f64::INFINITY;
        let mut step = 0;
        for s in 0..max_steps {
            step = s + 1;
            max_f = self.fire_step(f_tol)?;
            let scc = self.scc(SCC_MAX_ITER, scc_tol)?;
            let e = self.eval(false)?.energy;
            eprintln!("[GpuDftb] FIRE {step}/{max_steps} max|F|={max_f:.4e} E[0]={:.8} rms={:.3e} scc_iters={}", e[0], scc.rms, scc.n_iters);
            if max_f < f_tol { break; }
        }
        Ok((step, max_f, scc0.rms))
    }

    /// Frozen-H + energy-identity + optional CPU force/energy (replica 0). Call after `scc`. Prints; does not retune physics.
    pub fn measure(&mut self, want_cpu: bool) -> Result<GpuDftbEval> {
        if self.batch != 1 {
            return Err(DftbError::InvalidInput(format!("measure: batch={} — replica-0 diagnostics only, use batch=1", self.batch)));
        }
        let ev = self.eval(want_cpu)?;
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
        self.rt.read_buffer(&self.plan.q_new, &mut self.scratch_qd)?;
        self.rt.read_buffer(&self.plan.eig_diag, &mut self.plan.eig_diag_host)?;
        self.rt.read_buffer(&self.plan.occ_mask, &mut self.plan.mask_host)?;
        let eig = &self.plan.eig_diag_host[..n];
        let mask = &self.plan.mask_host[..n];
        let q_in = &self.scratch_q[..n_atoms];
        let q_d = &self.scratch_qd[..n_atoms];

        let (lmin_s, lmax_s) = {
            let (lam, _) = sym_eig_f32(&s, n);
            (lam.iter().copied().fold(f64::INFINITY, f64::min), lam.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        };
        let hc_all = residual_hc_sce(&hscc, &s, &c, eig, n, None);
        let hc_occ = residual_hc_sce(&hscc, &s, &c, eig, n, Some(mask));
        let ctsc = residual_ctsc(&s, &c, n);
        let xtsx = residual_ctsc(&s, &xlow, n);
        let ctcp = residual_eye(&cp, n);

        let (lam_r, c_r) = gevp_lowdin_f32(&hscc, &s, n)?;
        let mut eps_gpu_occ: Vec<f64> = (0..n).filter(|&k| mask[k] != 0).map(|k| eig[k] as f64).collect();
        eps_gpu_occ.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if eps_gpu_occ.len() != n_occ {
            return Err(DftbError::InvalidInput(format!("measure: occupied mask count {} != n_occ {n_occ}", eps_gpu_occ.len())));
        }
        let mut de_occ = 0.0f64;
        for k in 0..n_occ { de_occ = de_occ.max((eps_gpu_occ[k] - lam_r[k]).abs()); }
        let p_diff = projector_diff(&c, mask, &c_r, n, n_occ);

        let mut e_band = 0.0f64;
        for k in 0..n { if mask[k] != 0 { e_band += 2.0 * eig[k] as f64; } }
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
        eprintln!("[measure] r·V={rdotv:.3e} ½rᵀGr={half_rgr:.3e}  bandform−dens={ident:.3e}  predicted(δ_eig−½rGr)={pred:.3e}");
        eprintln!("[measure] q_rms={:.3e} q_max={:.3e}", ev.q_rms, ev.q_max);

        if want_cpu {
            let (e_cpu, f_cpu, q_cpu) = self.cpu_ref()?;
            let de = (ev.energy[0] - e_cpu).abs();
            let f_gpu = ev.forces.as_ref().expect("measure want_cpu implies eval(true)");
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
                return Err(DftbError::InvalidInput(format!("cpu_ref q len {} != n_atoms {n_atoms}", q_cpu.len())));
            }
            for a in 0..n_atoms {
                let dq = q_d[a] as f64 - q_cpu[a];
                q_max_cpu = q_max_cpu.max(dq.abs());
                q_s2 += dq * dq;
            }
            let q_rms_cpu = (q_s2 / n_atoms as f64).sqrt();
            eprintln!("[measure] CPU E_tot={e_cpu:.12}  |dE_GPU−CPU|={de:.3e}");
            eprintln!("[measure] max|q_D−q_cpu|={q_max_cpu:.3e}  rms(q_D−q_cpu)={q_rms_cpu:.3e}");
            eprintln!("[measure] max|F_gpu−F_cpu|={df:.3e}  max|F_gpu|={fg:.3e}  max|F_cpu|={fc:.3e}");
            let mut dh0 = 0.0f64;
            let mut ds = 0.0f64;
            {
                let mut cpu = crate::methods::dftb::dftb_cpu::DftbCpu::new(self.sk.clone(), self.species.clone())?;
                cpu.update_geometry(&self.coords[..n_atoms])?;
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
    pub fn cpu_ref(&self) -> Result<(f64, Vec<[f64; 3]>, Vec<f64>)> {
        let n_atoms = self.n_atoms;
        let xyz = &self.coords[..n_atoms];
        let mut cpu = crate::methods::dftb::dftb_cpu::DftbCpu::new(self.sk.clone(), self.species.clone())?;
        cpu.update_geometry(xyz)?;
        cpu.reset_charges();
        cpu.solve_scc(100, 1e-8)?;
        let scc = cpu.build_result();
        let e_rep = crate::methods::dftb::forces::repulsive_energy(&self.sk_dir, &self.species, xyz)?;
        let e = scc.energy + e_rep;
        let unique = unique_species(&self.species);
        let repulsive = crate::methods::dftb::forces::parse_all_repulsive(&self.sk_dir, &unique, unique.len())?;
        let forces = cpu.compute_forces(&scc, &repulsive)?;
        if !e.is_finite() { return Err(DftbError::InvalidInput(format!("cpu_ref: E={e} non-finite (E_el={} E_rep={e_rep})", scc.energy))); }
        Ok((e, forces.forces, scc.charges))
    }
}

fn unique_species(species: &[String]) -> Vec<String> {
    let mut u = Vec::new();
    for s in species {
        if !u.iter().any(|x| x == s) { u.push(s.clone()); }
    }
    u
}

fn fill_coord_scratch(coords: &[[f64; 3]], ang: &mut [f32], bohr: &mut [f32]) {
    for (i, c) in coords.iter().enumerate() {
        ang[3 * i] = c[0] as f32; ang[3 * i + 1] = c[1] as f32; ang[3 * i + 2] = c[2] as f32;
        bohr[3 * i] = (c[0] * ANG2BOHR) as f32; bohr[3 * i + 1] = (c[1] * ANG2BOHR) as f32; bohr[3 * i + 2] = (c[2] * ANG2BOHR) as f32;
    }
}

fn apply_disp(xyz: &mut [f64; 3], v: &[f64; 3], fx: f64, fy: f64, fz: f64, dt: f64) {
    const MAX_DISP: f64 = 0.1; // Å — same cap as CPU FIRE in examples/hbond_ref.rs
    let mut dx = v[0] * dt + 0.5 * fx * dt * dt;
    let mut dy = v[1] * dt + 0.5 * fy * dt * dt;
    let mut dz = v[2] * dt + 0.5 * fz * dt * dt;
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    if d > MAX_DISP { let s = MAX_DISP / d; dx *= s; dy *= s; dz *= s; }
    xyz[0] += dx; xyz[1] += dy; xyz[2] += dz;
}

fn fill_gamma(coords: &[[f64; 3]], u: &[f64], n_atoms: usize, batch: usize, g: &mut [f32]) {
    for b in 0..batch {
        let xyz = &coords[b * n_atoms..(b + 1) * n_atoms];
        let base = b * n_atoms * n_atoms;
        for a in 0..n_atoms {
            for c in 0..n_atoms {
                let dx = xyz[a][0] - xyz[c][0];
                let dy = xyz[a][1] - xyz[c][1];
                let dz = xyz[a][2] - xyz[c][2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
                g[base + a * n_atoms + c] = gamma_full(r, u[a], u[c]) as f32;
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
                if c > sk_cut { sk_cut = c; }
            }
        }
    }
    if sk_cut == 0.0 {
        return Err(DftbError::InvalidInput("GpuDftb: all pair cutoffs are 0 — SK tables missing?".into()));
    }
    Ok((pair_cut_sq, sk_cut * sk_cut))
}

fn orient_pair(
    i: usize, j: usize, n_orb_i: usize, n_orb_j: usize,
    orb_off_i: u16, orb_off_j: u16, sp_i: i32, sp_j: i32,
    dx: f64, dy: f64, dz: f64, r: f64,
) -> Option<(u8, u16, u16, u16, u16, f32, f32, f32, i32, i32)> {
    let bt = match (n_orb_i, n_orb_j) {
        (1, 1) => 0u8,
        (1, 4) | (4, 1) => 1u8,
        (4, 4) => 2u8,
        _ => return None,
    };
    let inv_r = 1.0 / r;
    if bt == 1 && n_orb_i == 4 && n_orb_j == 1 {
        Some((bt, j as u16, i as u16, orb_off_j, orb_off_i,
            (-dx * inv_r) as f32, (-dy * inv_r) as f32, (-dz * inv_r) as f32, sp_j, sp_i))
    } else {
        Some((bt, i as u16, j as u16, orb_off_i, orb_off_j,
            (dx * inv_r) as f32, (dy * inv_r) as f32, (dz * inv_r) as f32, sp_i, sp_j))
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
                i, j, n_orb_i, n_orb_j, tmpl.atom_orb_off[i], tmpl.atom_orb_off[j],
                atom_sp[i], atom_sp[j], 1.0, 0.0, 0.0, 1.0,
            ) else { continue };
            let already = buckets.iter().any(|b| {
                if b.block_type != bt { return false; }
                let t = &sk_tables[b.sk_table_idx];
                t.species_i as i32 == s_i && t.species_j as i32 == s_j
            });
            if already { continue; }
            let sk_table_idx = sk_tables.iter().position(|t| t.species_i as i32 == s_i && t.species_j as i32 == s_j)
                .ok_or_else(|| DftbError::InvalidInput(format!(
                    "GpuDftb: no SK table for template pair species {s_i}-{s_j} block={bt}"
                )))?;
            buckets.push(GpuPairBucket { n_pairs: 0, pairs: Vec::new(), block_type: bt, sk_table_idx });
        }
    }
    let _ = nsp;
    Ok(())
}

fn fragments_from_coords(tmpl: &FragmentTemplate, coords: &[[f64; 3]], batch: usize, n_atoms: usize) -> Vec<Fragment> {
    (0..batch).map(|b| {
        Fragment::from_template(tmpl.clone(), coords[b * n_atoms..(b + 1) * n_atoms].to_vec())
    }).collect()
}

fn per_atom_u(sk: &SkData, species: &[String]) -> Result<Vec<f64>> {
    species.iter().map(|sp| {
        sk.onsite(sp).map(|p| p.u_hubbard).map_err(|e| DftbError::InvalidInput(format!("Hubbard U missing for species {sp}: {e}")))
    }).collect()
}

fn orb_atom_map(atom_orb_off: &[u16], n_orbs: usize) -> Vec<i32> {
    let mut map = vec![0i32; n_orbs];
    for a in 0..atom_orb_off.len() - 1 {
        for mu in atom_orb_off[a] as usize..atom_orb_off[a + 1] as usize { map[mu] = a as i32; }
    }
    map
}

fn frobenius_f32(a: &[f32], b: &[f32], n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n * n { s += a[i] as f64 * b[i] as f64; }
    s
}

fn quad_g(g: &[f32], r: &[f64], n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n {
        let mut gi = 0.0f64;
        for j in 0..n { gi += g[i * n + j] as f64 * r[j]; }
        s += r[i] * gi;
    }
    0.5 * s
}

fn residual_hc_sce(h: &[f32], s: &[f32], c: &[f32], eig: &[f32], n: usize, mask: Option<&[i32]>) -> f64 {
    let mut mx = 0.0f64;
    for k in 0..n {
        if let Some(m) = mask { if m[k] == 0 { continue; } }
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
            for k in 0..n { acc += c[k * n + i] as f64 * c[k * n + j] as f64; }
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
                for nu in 0..n { srow += s[mu * n + nu] as f64 * c[nu * n + j] as f64; }
                acc += c[mu * n + i] as f64 * srow;
            }
            let target = if i == j { 1.0 } else { 0.0 };
            mx = mx.max((acc - target).abs());
        }
    }
    mx
}

fn mat_from_row(a: &[f32], n: usize) -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_fn(n, n, |i, j| 0.5 * (a[i * n + j] as f64 + a[j * n + i] as f64))
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
        for r in 0..n { v[(r, c)] = ev.eigenvectors[(r, p)]; }
    }
    (lam_s, v)
}

fn gevp_lowdin_f32(h: &[f32], s: &[f32], n: usize) -> Result<(Vec<f64>, nalgebra::DMatrix<f64>)> {
    let (lam_s, u) = sym_eig_f32(s, n);
    for (i, &l) in lam_s.iter().enumerate() {
        if !l.is_finite() || l <= 1e-6 {
            return Err(DftbError::InvalidInput(format!("gevp_lowdin: λ_S[{i}]={l} non-finite or ≤1e-6")));
        }
    }
    let mut x = nalgebra::DMatrix::<f64>::zeros(n, n);
    for k in 0..n {
        let srt = 1.0 / lam_s[k].sqrt();
        for i in 0..n {
            for j in 0..n { x[(i, j)] += u[(i, k)] * srt * u[(j, k)]; }
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
        for r in 0..n { cp[(r, c)] = he.eigenvectors[(r, p)]; }
    }
    let c = &x * cp;
    Ok((lam_s2, c))
}

fn projector_diff(c_gpu: &[f32], mask: &[i32], c_cpu: &nalgebra::DMatrix<f64>, n: usize, n_occ: usize) -> f64 {
    let mut p_g = vec![0.0f64; n * n];
    for k in 0..n {
        if mask[k] == 0 { continue; }
        for i in 0..n {
            for j in 0..n { p_g[i * n + j] += c_gpu[i * n + k] as f64 * c_gpu[j * n + k] as f64; }
        }
    }
    let mut s2 = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut p_c = 0.0f64;
            for k in 0..n_occ { p_c += c_cpu[(i, k)] * c_cpu[(j, k)]; }
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
        if mask[k] == 0 { continue; }
        let mut acc = 0.0f64;
        for i in 0..n {
            let mut hi = 0.0f64;
            for j in 0..n { hi += h[i * n + j] as f64 * c[j * n + k] as f64; }
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
                if mask[k] != 0 { p += c[i * n + k] as f64 * c[j * n + k] as f64; }
            }
            let diff = d[i * n + j] as f64 - 2.0 * p;
            s2 += diff * diff;
        }
    }
    s2.sqrt()
}
