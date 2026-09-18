//! `GpuPbc` — periodic (complex k-point) DFTB driver.
//!
//! Mirrors `GpuDftb` for a replicated periodic cell: all buffers, kernels,
//! and CSR lists are built once at construction; `set_geometry` re-runs
//! the assembly chain (image-pair SK eval → Bloch fold → Ewald γ) and the
//! SCC loop runs entirely inside `GpuPbcPlan` — no allocation, no kernel
//! builds, minimal host sync inside iterations.
//!
//! Pipeline per geometry (in-order queue, one enqueue chain):
//!   1. fill H0(k)/S(k) = 0
//!   2. per SK bucket: `assemble_pairs_img` → per-slot real-space blocks
//!      (k-independent), `kpoint_phase_sum_batched` → H0(k), S(k)
//!   3. `ewald_invr_batched` + `gamma_pbc_batched` → γ_pbc
//!   4. `GpuPbcPlan::set_geometry` → S(k)^{-1/2}
//!
//! Fortran reference: `Dense_Multi_PBC.ewald_notes.md` /
//! `Dense_Multi_PBC.tasks.md` in doc/prokop/tasts/HBond_Relaxed_Scan_GPU/.

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::gamma::GammaTable;
use crate::methods::dftb::sk_data::SkData;
use crate::qmqm::fragment::FragmentTemplate;
use crate::qmqm::gpu_pbc_plan::GpuPbcPlan;
use crate::qmqm::gpu_prep::{pack_sk_tables, GpuSkTable};
use crate::qmqm::gpu_runtime::{map_ocl_err, GpuRuntime};
use crate::qmqm::pbc_cell::{
    enumerate_image_pairs, enumerate_sk_pairs, max_g_ewald, max_r_ewald, optimal_alpha,
    GpuFoldPair, GpuImgSlot, PbcCell, TOL_EWALD_DEFAULT,
};
use ocl::prm::{Float2, Float4, Int2};
use ocl::{Buffer, Kernel, Program};
use std::collections::HashMap;

const HAM_SOURCE: &str = include_str!("../methods/dftb/dftb_hamiltonian.cl");
const PBC_SOURCE: &str = include_str!("gpu_pbc.cl");

const ANG2BOHR: f64 = 1.889726133;
/// Pair/cell enumeration safety margin (Bohr) — the image R *set* is
/// frozen at construction; geometry updates inside this margin need no
/// re-enumeration (extra slots evaluate to ~0 anyway).
const PAIR_MARGIN: f64 = 2.0;

/// One SK species-pair bucket: uniform block geometry, one slot list,
/// one fold-pair list, one block buffer pair, two bound kernels.
struct PbcSkBucket {
    k_img: Kernel,
    k_fold: Kernel,
    n_slots: usize,
    n_outs: usize,
    /// Kept alive — bound as kernel args.
    _slots: Buffer<GpuImgSlot>,
    _outs: Buffer<GpuFoldPair>,
    _sk_h: Buffer<f32>,
    _sk_s: Buffer<f32>,
    _h_blk: Buffer<f32>,
    _s_blk: Buffer<f32>,
}

/// Periodic multi-replica DFTB+ solver (complex k-point path).
pub struct GpuPbc {
    pub rt: GpuRuntime,
    pub plan: GpuPbcPlan,

    pub cell: PbcCell,
    /// Ewald parameters actually used (auto-tuned unless overridden).
    pub alpha: f64,
    pub max_r: f64,
    pub max_g: f64,

    n: usize,
    n_atoms: usize,
    n_rep: usize,
    nk: usize,

    buf_coords: Buffer<f32>, // [n_rep·n_at·3] Bohr
    buf_h0: Buffer<Float2>,  // [n_rep·nk·n²] — plan input
    buf_s: Buffer<Float2>,   // [n_rep·nk·n²]
    buf_g: Buffer<f32>,      // [n_rep·n_at²] γ_pbc — plan input
    buf_invr: Buffer<f32>,   // [n_rep·n_at²]
    park: Buffer<i32>,       // [n_rep] all-1 (parking wired later)

    buckets: Vec<PbcSkBucket>,
    k_ewald: Kernel,
    k_gamma: Kernel,

    // kept alive — kernel args
    _rcell: Buffer<Float4>,
    _epair: Buffer<Int2>,
    _eoff: Buffer<i32>,
    _eslot: Buffer<Float4>,
    _gvec: Buffer<Float4>,
    _gpair: Buffer<Int2>,
    _goff: Buffer<i32>,
    _gslot: Buffer<Float4>,
    _species: Buffer<i32>,
    _uhub: Buffer<f32>,
    _onsite: Buffer<f32>,
    _kcart: Buffer<Float4>,
    _q0: Buffer<f32>,
    _oa: Buffer<i32>,

    q0_host: Vec<f32>,      // [n_rep·n_at]
    coords: Vec<[f64; 3]>,  // [n_rep·n_at] Å
    scratch_bohr: Vec<f32>, // [n_rep·n_at·3]

    n_occ: usize,
    _prog: Program,
}

impl GpuPbc {
    /// `coords` — [n_rep·n_atoms] in Å (same cell, per-replica geometry).
    /// `lat` — lattice vectors (rows) in Å. `k_frac` — fractional
    /// k-points, `kw` — their weights (Σw = 1).
    pub fn new(
        sk: SkData,
        species: Vec<String>,
        coords: Vec<[f64; 3]>,
        lat: [[f64; 3]; 3],
        k_frac: &[[f64; 3]],
        kw: &[f32],
        ewald_alpha: Option<f64>,
    ) -> Result<Self> {
        let n_atoms = species.len();
        let nk = k_frac.len();
        if n_atoms == 0 || nk == 0 || kw.len() != nk {
            return Err(DftbError::InvalidInput(format!(
                "GpuPbc::new: n_atoms={n_atoms} nk={nk} kw={} — bad dims",
                kw.len()
            )));
        }
        if coords.len() % n_atoms != 0 {
            return Err(DftbError::InvalidInput(format!(
                "GpuPbc::new: coords {} not a multiple of n_atoms {n_atoms}",
                coords.len()
            )));
        }
        let n_rep = coords.len() / n_atoms;

        let mut rt = GpuRuntime::new()?;
        rt.require_nvidia()?;
        // The Ewald kernel accumulates in f64 — fail loud if unsupported.
        let ext = rt
            .device()
            .info(ocl::enums::DeviceInfo::Extensions)
            .map(|r| r.to_string())
            .unwrap_or_default();
        if !ext.contains("cl_khr_fp64") {
            return Err(DftbError::InvalidInput(
                "GpuPbc: device lacks cl_khr_fp64 — Ewald accumulation needs f64".into(),
            ));
        }

        let tmpl = FragmentTemplate::new(&sk, species.clone(), coords[..n_atoms].to_vec())?;
        let n = tmpl.n_orbs;
        let n_el: f64 = tmpl.q0.iter().sum();
        let n_occ = (n_el / 2.0).round() as usize;
        if n_occ == 0 || n_occ > n || (n_el - 2.0 * n_occ as f64).abs() > 1e-6 {
            return Err(DftbError::InvalidInput(format!(
                "GpuPbc::new: n_el={n_el} → n_occ={n_occ} invalid for N={n} (closed-shell even only)"
            )));
        }

        // ---- global species table (unique, first-appearance order) ----
        let mut gsp: Vec<String> = Vec::new();
        let mut gmap: HashMap<String, u8> = HashMap::new();
        for sp in &species {
            if !gmap.contains_key(sp) {
                gmap.insert(sp.clone(), gsp.len() as u8);
                gsp.push(sp.clone());
            }
        }
        let nsp = gsp.len();
        let sp2g: HashMap<String, u8> = gmap;
        let atom_gsp: Vec<usize> = (0..n_atoms)
            .map(|a| sp2g[&tmpl.species[a]] as usize)
            .collect();
        let gamma_tbl = GammaTable::from_sk_data(&sk, &species)?;
        let u_hub: Vec<f32> = (0..nsp).map(|i| gamma_tbl.u(i as u8) as f32).collect();

        // ---- SK tables (same packing as GpuDftb) ----
        let sk_tables = pack_sk_tables(&sk, &gsp, &sp2g)?;
        // sk_lookup[gsp_i·nsp + gsp_j] → table index
        let mut sk_lookup = vec![usize::MAX; nsp * nsp];
        for (ti, t) in sk_tables.iter().enumerate() {
            sk_lookup[t.species_i as usize * nsp + t.species_j as usize] = ti;
        }
        // per-species-pair SK cutoff (Bohr — SK grids are already in Bohr)
        let mut pair_cut = vec![0.0f64; nsp * nsp];
        let mut sk_cut_max = 0.0f64;
        for si in 0..nsp {
            for sj in 0..nsp {
                if let Some(t) = sk.get_pair(&gsp[si], &gsp[sj]) {
                    let c = t.cutoff();
                    pair_cut[si * nsp + sj] = c;
                    sk_cut_max = sk_cut_max.max(c);
                }
            }
        }

        // ---- cell + Ewald tuning ----
        let lat_b: [[f64; 3]; 3] = {
            let mut l = [[0.0; 3]; 3];
            for i in 0..3 {
                for d in 0..3 {
                    l[i][d] = lat[i][d] * ANG2BOHR;
                }
            }
            l
        };
        let cell = PbcCell::new(lat_b)?;
        let tol = TOL_EWALD_DEFAULT;
        let alpha = match ewald_alpha {
            Some(a) => a,
            None => optimal_alpha(&cell, tol)?,
        };
        let max_r = max_r_ewald(alpha, tol)?;
        let max_g = max_g_ewald(alpha, cell.vol, tol)?;
        let gpts = cell.g_lattice_points(max_g);
        let gvec: Vec<Float4> = gpts
            .iter()
            .map(|g| {
                let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                let w = (-0.25 * g2 / (alpha * alpha)).exp() / g2;
                Float4::new(g[0] as f32, g[1] as f32, g[2] as f32, w as f32)
            })
            .collect();
        let rec_fac = 8.0 * std::f64::consts::PI / cell.vol;
        let c_const = -std::f64::consts::PI / (cell.vol * alpha * alpha);
        let c_self = -2.0 * alpha / std::f64::consts::PI.sqrt();

        // ---- replica-0 geometry (Bohr) for enumeration ----
        let c0: Vec<[f64; 3]> = coords[..n_atoms]
            .iter()
            .map(|c| [c[0] * ANG2BOHR, c[1] * ANG2BOHR, c[2] * ANG2BOHR])
            .collect();

        // ---- shared integer-cell table for the SK slots ----
        let gamma_cut = gamma_tbl.max_cutoff();
        let big_cut = (sk_cut_max + PAIR_MARGIN)
            .max(max_r + PAIR_MARGIN)
            .max(gamma_cut + PAIR_MARGIN);
        let extent = {
            let mut e = 0.0f64;
            for a in &c0 {
                for b in &c0 {
                    e = e.max((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs());
                }
            }
            e
        };
        let cells = cell.cell_translations(big_cut + extent + 1.0);
        let rcarts: Vec<[f64; 3]> = cells.iter().map(|&nc| cell.rvec(nc)).collect();
        let rcell_dev: Vec<Float4> = rcarts
            .iter()
            .map(|r| Float4::new(r[0] as f32, r[1] as f32, r[2] as f32, 0.0))
            .collect();

        // ---- enumerations ----
        // SK image pairs (oriented, bucketed)
        let no = &tmpl.atom_n_orb;
        let oo = &tmpl.atom_orb_off;
        let n_buckets = 3 * nsp * nsp;
        let buckets_host = enumerate_sk_pairs(
            &c0,
            &cell,
            &cells,
            &rcarts,
            no,
            oo,
            PAIR_MARGIN,
            &|a, b| {
                // s-atom first in the oriented eval (SK tables are (s,p))
                if no[a] == 4 && no[b] == 1 {
                    (b, a)
                } else {
                    (a, b)
                }
            },
            &|oi, oj| pair_cut[atom_gsp[oi] * nsp + atom_gsp[oj]],
            &|oi, oj| sk_lookup[atom_gsp[oi] * nsp + atom_gsp[oj]],
            &|oi, oj| {
                let bt = match (no[oi], no[oj]) {
                    (1, 1) => 0usize,
                    (1, 4) | (4, 1) => 1,
                    (4, 4) => 2,
                    _ => usize::MAX,
                };
                bt * nsp * nsp + atom_gsp[oi] * nsp + atom_gsp[oj]
            },
            n_buckets,
        );
        // Ewald real-space list: all i≤j pairs, images of j near i,
        // self-slot excluded, empty pairs emitted (recip/const still apply)
        let elist = enumerate_image_pairs(&c0, &cell, max_r + PAIR_MARGIN, false, true);
        // Short-γ list: includes the (a,a,R=0) onsite slot
        let glist = enumerate_image_pairs(&c0, &cell, gamma_cut + PAIR_MARGIN, true, true);

        // ---- device buffers ----
        let f32c: Vec<f32> = coords
            .iter()
            .flat_map(|c| {
                [
                    (c[0] * ANG2BOHR) as f32,
                    (c[1] * ANG2BOHR) as f32,
                    (c[2] * ANG2BOHR) as f32,
                ]
            })
            .collect();
        let buf_coords = rt.buffer_from_slice(&f32c)?;
        let buf_rcell = rt.buffer_from_slice(&rcell_dev)?;
        let park = rt.buffer_from_slice(&vec![1i32; n_rep])?;

        // onsite per orbital (rep-uniform): orb 0 of each atom → e_s, rest → e_p
        let mut onsite_orb = vec![0.0f32; n];
        for a in 0..n_atoms {
            let on = sk.onsite(&tmpl.species[a]).map_err(|e| {
                DftbError::InvalidInput(format!("onsite missing for {}: {e}", tmpl.species[a]))
            })?;
            let off = tmpl.atom_orb_off[a] as usize;
            for o in 0..tmpl.atom_n_orb[a] as usize {
                onsite_orb[off + o] = if o == 0 { on.e_s as f32 } else { on.e_p as f32 };
            }
        }
        let buf_onsite = rt.buffer_from_slice(&onsite_orb)?;

        // k-points: fractional → Cartesian via rec (2π included)
        let kcart: Vec<Float4> = k_frac
            .iter()
            .map(|k| {
                let mut v = [0.0f64; 3];
                for d in 0..3 {
                    v[d] = k[0] * cell.rec[0][d] + k[1] * cell.rec[1][d] + k[2] * cell.rec[2][d];
                }
                Float4::new(v[0] as f32, v[1] as f32, v[2] as f32, 0.0)
            })
            .collect();
        let buf_kcart = rt.buffer_from_slice(&kcart)?;

        // ewald buffers
        let epair: Vec<Int2> = elist
            .pair_ij
            .iter()
            .map(|&(i, j)| Int2::new(i as i32, j as i32))
            .collect();
        let eslot: Vec<Float4> = elist
            .rvecs
            .iter()
            .map(|r| Float4::new(r[0], r[1], r[2], 0.0))
            .collect();
        let buf_epair = rt.buffer_from_slice(&epair)?;
        let buf_eoff = rt.buffer_from_slice(&elist.r_off)?;
        let buf_eslot = rt.buffer_from_slice(&eslot)?;
        let buf_gvec = rt.buffer_from_slice(&gvec)?;
        let buf_invr = rt.zero_buffer::<f32>(n_rep * n_atoms * n_atoms)?;

        // gamma buffers
        let gpair: Vec<Int2> = glist
            .pair_ij
            .iter()
            .map(|&(i, j)| Int2::new(i as i32, j as i32))
            .collect();
        let gslot: Vec<Float4> = glist
            .rvecs
            .iter()
            .map(|r| Float4::new(r[0], r[1], r[2], 0.0))
            .collect();
        let buf_gpair = rt.buffer_from_slice(&gpair)?;
        let buf_goff = rt.buffer_from_slice(&glist.r_off)?;
        let buf_gslot = rt.buffer_from_slice(&gslot)?;
        let species_flat: Vec<i32> = (0..n_rep)
            .flat_map(|_| atom_gsp.iter().map(|&s| s as i32))
            .collect();
        let buf_species = rt.buffer_from_slice(&species_flat)?;
        let buf_uhub = rt.buffer_from_slice(&u_hub)?;
        let buf_g = rt.zero_buffer::<f32>(n_rep * n_atoms * n_atoms)?;

        // plan inputs
        let n_sys = n_rep * nk;
        let buf_h0 = rt.zero_buffer::<Float2>(n_sys * n * n)?;
        let buf_s = rt.zero_buffer::<Float2>(n_sys * n * n)?;
        let q0_host: Vec<f32> = tmpl
            .q0
            .iter()
            .map(|&q| q as f32)
            .collect::<Vec<_>>()
            .repeat(n_rep);
        let buf_q0 = rt.buffer_from_slice(&q0_host)?;
        let mut oa = vec![0i32; n_rep * n];
        for a in 0..n_atoms {
            let s = tmpl.atom_orb_off[a] as usize;
            let e = s + tmpl.atom_n_orb[a] as usize;
            for o in s..e {
                for r in 0..n_rep {
                    oa[r * n + o] = a as i32;
                }
            }
        }
        let buf_oa = rt.buffer_from_slice(&oa)?;

        // ---- program + kernels ----
        let prog = rt.build_program(&format!("{HAM_SOURCE}\n{PBC_SOURCE}"))?;
        let wg = rt.caps().max_work_group_size.min(256).max(64);

        // per-bucket: slot/assemble kernel + fold kernel
        let mut buckets = Vec::with_capacity(buckets_host.len());
        for bh in &buckets_host {
            let bt = bh.block_type;
            let bsz = bh.norb_oi * bh.norb_oj;
            let n_slots = bh.slots.len();
            let n_outs = bh.outs.len();
            if n_slots == 0 && n_outs == 0 {
                continue;
            }
            let tab: &GpuSkTable = &sk_tables[bh.sk_table_idx];
            let buf_slots = rt.buffer_from_slice(&bh.slots)?;
            let buf_outs = rt.buffer_from_slice(&bh.outs)?;
            let buf_sk_h = rt.buffer_from_slice(&tab.sk_h)?;
            let buf_sk_s = rt.buffer_from_slice(&tab.sk_s)?;
            let buf_hb = rt.zero_buffer::<f32>(n_rep * n_slots * bsz)?;
            let buf_sb = rt.zero_buffer::<f32>(n_rep * n_slots * bsz)?;

            // assemble_pairs_img(slots, rcell, coords, n_atoms, n_rep,
            //   sk_h, sk_s, dr, n_grid, n_slots, block_type, n_sk_cols,
            //   h_blk, s_blk, park)
            let g_img = ((n_rep * n_slots + wg - 1) / wg) * wg;
            let k_img = Kernel::builder()
                .program(&prog)
                .name("assemble_pairs_img")
                .queue(rt.queue().clone())
                .global_work_size(g_img)
                .local_work_size(wg)
                .arg(&buf_slots)
                .arg(&buf_rcell)
                .arg(&buf_coords)
                .arg(n_atoms as i32)
                .arg(n_rep as i32)
                .arg(&buf_sk_h)
                .arg(&buf_sk_s)
                .arg(tab.dr as f32)
                .arg(tab.n_grid as i32)
                .arg(n_slots as i32)
                .arg(bt as i32)
                .arg(tab.n_sk_cols as i32)
                .arg(&buf_hb)
                .arg(&buf_sb)
                .arg(&park)
                .build()
                .map_err(map_ocl_err)?;

            // kpoint_phase_sum_batched(outs, slots, rcell, kcart, nk,
            //   n_orb, n_rep, n_outs, n_slots, bsz, h_blk, s_blk, onsite,
            //   H, S, park)
            let g_fold = n_rep * n_outs * nk * bsz;
            let k_fold = Kernel::builder()
                .program(&prog)
                .name("kpoint_phase_sum_batched")
                .queue(rt.queue().clone())
                .global_work_size(g_fold)
                .arg(&buf_outs)
                .arg(&buf_slots)
                .arg(&buf_rcell)
                .arg(&buf_kcart)
                .arg(nk as i32)
                .arg(n as i32)
                .arg(n_rep as i32)
                .arg(n_outs as i32)
                .arg(n_slots as i32)
                .arg(bsz as i32)
                .arg(&buf_hb)
                .arg(&buf_sb)
                .arg(&buf_onsite)
                .arg(&buf_h0)
                .arg(&buf_s)
                .arg(&park)
                .build()
                .map_err(map_ocl_err)?;

            buckets.push(PbcSkBucket {
                k_img,
                k_fold,
                n_slots,
                n_outs,
                _slots: buf_slots,
                _outs: buf_outs,
                _sk_h: buf_sk_h,
                _sk_s: buf_sk_s,
                _h_blk: buf_hb,
                _s_blk: buf_sb,
            });
        }

        // ewald kernel
        let n_ep = elist.pair_ij.len();
        let k_ewald = Kernel::builder()
            .program(&prog)
            .name("ewald_invr_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * n_ep)
            .arg(n_atoms as i32)
            .arg(n_ep as i32)
            .arg(n_rep as i32)
            .arg(&buf_coords)
            .arg(&buf_epair)
            .arg(&buf_eoff)
            .arg(&buf_eslot)
            .arg(&buf_gvec)
            .arg(gvec.len() as i32)
            .arg(alpha)
            .arg(rec_fac)
            .arg(c_const)
            .arg(c_self)
            .arg(&buf_invr)
            .arg(&park)
            .build()
            .map_err(map_ocl_err)?;

        // gamma kernel
        let n_gp = glist.pair_ij.len();
        let k_gamma = Kernel::builder()
            .program(&prog)
            .name("gamma_pbc_batched")
            .queue(rt.queue().clone())
            .global_work_size(n_rep * n_gp)
            .arg(n_atoms as i32)
            .arg(n_gp as i32)
            .arg(n_rep as i32)
            .arg(&buf_coords)
            .arg(&buf_gpair)
            .arg(&buf_goff)
            .arg(&buf_gslot)
            .arg(&buf_species)
            .arg(&buf_uhub)
            .arg(&buf_invr)
            .arg(&buf_g)
            .arg(&park)
            .build()
            .map_err(map_ocl_err)?;

        // First assembly: plan construction runs the S^{-1/2} pipeline
        // and certifies λ_min — buf_s must hold valid S(k), not zeros.
        // buf_coords already contains the initial geometry.
        buf_h0
            .cmd()
            .fill(Float2::new(0.0, 0.0), None)
            .enq()
            .map_err(map_ocl_err)?;
        buf_s
            .cmd()
            .fill(Float2::new(0.0, 0.0), None)
            .enq()
            .map_err(map_ocl_err)?;
        unsafe {
            for b in &buckets {
                b.k_img.enq().map_err(map_ocl_err)?;
            }
            for b in &buckets {
                b.k_fold.enq().map_err(map_ocl_err)?;
            }
            k_ewald.enq().map_err(map_ocl_err)?;
            k_gamma.enq().map_err(map_ocl_err)?;
        }
        rt.finish()?;

        let plan = GpuPbcPlan::new(
            &mut rt, &buf_s, &buf_h0, &buf_g, &buf_q0, &buf_oa, n, n_atoms, n_rep, nk, kw,
        )?;

        Ok(Self {
            rt,
            plan,
            cell,
            alpha,
            max_r,
            max_g,
            n,
            n_atoms,
            n_rep,
            nk,
            buf_coords,
            buf_h0,
            buf_s,
            buf_g,
            buf_invr,
            park,
            buckets,
            k_ewald,
            k_gamma,
            _rcell: buf_rcell,
            _epair: buf_epair,
            _eoff: buf_eoff,
            _eslot: buf_eslot,
            _gvec: buf_gvec,
            _gpair: buf_gpair,
            _goff: buf_goff,
            _gslot: buf_gslot,
            _species: buf_species,
            _uhub: buf_uhub,
            _onsite: buf_onsite,
            _kcart: buf_kcart,
            _q0: buf_q0,
            _oa: buf_oa,
            q0_host,
            coords,
            scratch_bohr: vec![0.0; n_rep * n_atoms * 3],
            n_occ,
            _prog: prog,
        })
    }

    /// Upload a new geometry (Å, [n_rep·n_atoms]) and rebuild
    /// H0(k)/S(k)/γ_pbc on device; then runs the plan's S^{-1/2} pipeline.
    pub fn set_geometry(&mut self, coords: &[[f64; 3]]) -> Result<()> {
        if coords.len() != self.n_rep * self.n_atoms {
            return Err(DftbError::InvalidInput(format!(
                "GpuPbc::set_geometry: coords {} != {}·{}",
                coords.len(),
                self.n_rep,
                self.n_atoms
            )));
        }
        for (i, c) in coords.iter().enumerate() {
            self.scratch_bohr[3 * i] = (c[0] * ANG2BOHR) as f32;
            self.scratch_bohr[3 * i + 1] = (c[1] * ANG2BOHR) as f32;
            self.scratch_bohr[3 * i + 2] = (c[2] * ANG2BOHR) as f32;
        }
        self.coords.copy_from_slice(coords);
        self.rt.write_buffer(&self.buf_coords, &self.scratch_bohr)?;

        // H0(k)/S(k) zero-fill (out-pairs with no slots stay zero)
        self.buf_h0
            .cmd()
            .fill(Float2::new(0.0, 0.0), None)
            .enq()
            .map_err(map_ocl_err)?;
        self.buf_s
            .cmd()
            .fill(Float2::new(0.0, 0.0), None)
            .enq()
            .map_err(map_ocl_err)?;

        unsafe {
            for b in &self.buckets {
                b.k_img.enq().map_err(map_ocl_err)?; // per-slot SK blocks
            }
            for b in &self.buckets {
                b.k_fold.enq().map_err(map_ocl_err)?; // Bloch fold → H0(k),S(k)
            }
            self.k_ewald.enq().map_err(map_ocl_err)?; // invRMat
            self.k_gamma.enq().map_err(map_ocl_err)?; // γ_pbc = invr − Σ expGamma
        }
        // S(k)^{-1/2} pipeline + λ_min/Jacobi certification
        self.plan.set_geometry(&mut self.rt, &self.buf_s)?;
        Ok(())
    }

    /// Convenience: initial charges + DIIS reset, then synchronous SCC
    /// loop until rms < tol or max_iter. Returns per-replica converged
    /// flags and the rms history (diagnostic).
    pub fn scc(
        &mut self,
        alpha: f32,
        rms_tol: f32,
        max_iter: usize,
    ) -> Result<(Vec<bool>, Vec<f32>)> {
        self.plan.set_initial_charges(&self.rt, &self.q0_host)?;
        self.plan.reset_diis(&self.rt)?;
        let mut hist = Vec::with_capacity(max_iter);
        for _ in 0..max_iter {
            let r = self
                .plan
                .scc_step_diis(&mut self.rt, self.n_occ, alpha, rms_tol)?;
            hist.push(r);
            if r < rms_tol {
                break;
            }
        }
        let ok = self.plan.check_jacobi(&self.rt, &vec![1i32; self.n_rep])?;
        Ok((ok, hist))
    }

    /// Read back H0(k) for inspection/tests: [rep·nk·n²] complex pairs.
    pub fn read_h0(&self) -> Result<Vec<Float2>> {
        let mut out = vec![Float2::new(0.0, 0.0); self.n_rep * self.nk * self.n * self.n];
        self.rt.read_buffer(&self.buf_h0, &mut out)?;
        Ok(out)
    }
    /// Read back S(k).
    pub fn read_s(&self) -> Result<Vec<Float2>> {
        let mut out = vec![Float2::new(0.0, 0.0); self.n_rep * self.nk * self.n * self.n];
        self.rt.read_buffer(&self.buf_s, &mut out)?;
        Ok(out)
    }
    /// Read back γ_pbc [n_rep·n_at²].
    pub fn read_gamma(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.n_rep * self.n_atoms * self.n_atoms];
        self.rt.read_buffer(&self.buf_g, &mut out)?;
        Ok(out)
    }
    /// Read back invRMat [n_rep·n_at²].
    pub fn read_invr(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.n_rep * self.n_atoms * self.n_atoms];
        self.rt.read_buffer(&self.buf_invr, &mut out)?;
        Ok(out)
    }

    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (self.n, self.n_atoms, self.n_rep, self.nk)
    }
    pub fn n_occ(&self) -> usize {
        self.n_occ
    }
}
