//! Persistent sparse DFTB engine — production run loop (manifest §0.4).
//!
//! GPU analogue of `DftbCpu` / `GpuDftb`. One object owns the OpenCL runtime,
//! compiled BSR4 kernels, SK tables, masks, and every buffer. Tests and MD/FIRE
//! drive this; do not construct a throwaway `SparseBsr4Gpu` per geometry.
//!
//! ```text
//! INIT (once)     SparseDftb::new  — runtime, kernels, SK, BSR workspace, host scratch
//! PER GEOMETRY    set_coords       — CPU H0/S, upload values, NS for Z (S fixed)
//! SCC             scc              — mix + K0+TC2 on device; q download only
//! ENERGY/FORCES   energy / forces  — 2 Tr(K H0)+½Δq·V+E_rep; D=2K W=2KHK (CPU contract)
//! MD / FIRE       fire_step / md_step / relax
//! ```
//!
//! Topology is frozen at `new` (full mask for n_atom≤64, else geometric + skin).
//! A pair appearing outside the mask fails loud — rebuild `new`, do not silently
//! grow CSR in the MD loop. OpenCL `Program`/`Kernel`/`Buffer` are not created in
//! `scc` / `forces` / `fire_step`.
//!
//! Drive from **`dftb_engine`** (`sparse_new` / `sparse_scc` / `sparse_eval` / …),
//! not a new `src/bin` or `tests/*.rs` per molecule. Example:
//! `rust_dftb/scripts/test_sparse_dftb_sih4.rhai`. Docs: `doc/prokop/userguide/sparse_dftb.md`.
//!
//! Device NS residual scalar is N4-wrong; `compute_z` uses the intersection
//! SpGEMM kernel and `||I−T||` of the downloaded T. Host `newton_schulz_inverse`
//! (allocating) stays in `scc.rs` for G3 diagnostics.

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::forces::{repulsive_energy, Forces};
use crate::methods::dftb::gamma::GammaTable;
use crate::methods::dftb::hamiltonian::{HamiltonianBuilder, SystemContext};
use crate::methods::dftb::sk_data::SkData;
use crate::methods::sparse::bsr4::{
    build_full_mask, build_geometric_mask, fill_bsr_values_from_dense,
    pad_physical_to_bsr4_into, Bsr4Matrix, BS,
};
use crate::methods::sparse::gpu_sparse::{SparseBsr4Config, SparseBsr4Gpu, TC2_TRACE_TOL};
use crate::methods::sparse::scc::{apply_shift_padded_into, trace_ab, SparseDftbEnergy, E_DUMMY};
use crate::methods::sparse::sparse_forces::sparse_analytic_forces;
use crate::methods::sparse::sparse_system::SparseSystemWorkspace;
use crate::qmqm::shifts::compute_intra_shifts;

const ANG2BOHR: f64 = 1.889_726_133;
const SCC_MAX: usize = 100;
const FULL_MASK_ATOMS: usize = 64;
const MAX_FIRE_DISP: f64 = 0.1; // Å

/// Result of one `scc` call.
#[derive(Debug, Clone)]
pub struct SparseDftbScc {
    pub n_iters: usize,
    pub rms: f64,
    pub r_scc: f64,
    pub tr_ks: f32,
    pub r_i: f32,
}

/// Tunables. `new` uses `Default`. Changing n_atom or mask requires a new engine.
#[derive(Debug, Clone)]
pub struct SparseDftbConfig {
    pub mix: f64,
    pub scc_tol: f64,
    pub max_scc: usize,
    pub ns_max: usize,
    pub ns_tol: f32,
    pub tc2_max: usize,
    pub tc2_tol: f32,
    /// `None` = full mask if n_atom ≤ 64, else geometric SK cutoff + `r_skin_ang`.
    pub full_mask: Option<bool>,
    pub r_skin_ang: f64,
}

impl Default for SparseDftbConfig {
    fn default() -> Self {
        Self {
            mix: 0.5, scc_tol: 1e-5, max_scc: 80,
            ns_max: 50, ns_tol: 1e-5, tc2_max: 80, tc2_tol: 1e-4,
            full_mask: None, r_skin_ang: 1.0,
        }
    }
}

/// s/p DFTB valence electrons. SK onsite `q0` is a trailing-field parser bug — do not use it.
pub fn valence_q0(species: &[String]) -> Result<Vec<f64>> {
    species.iter().map(|sp| {
        let v = match sp.as_str() {
            "H" | "h" => 1.0,
            "C" | "c" => 4.0,
            "N" | "n" => 5.0,
            "O" | "o" => 6.0,
            "F" | "f" => 7.0,
            "Si" | "si" => 4.0,
            "P" | "p" => 5.0,
            "S" | "s" => 6.0,
            "Cl" | "cl" => 7.0,
            _ => return Err(DftbError::InvalidInput(format!(
                "valence_q0: no s/p DFTB valence for '{sp}'. Pass explicit q0 or extend the table. Do not use SK onsite q0 (parser reads trailing fields)."
            ))),
        };
        Ok(v)
    }).collect()
}

/// Persistent sparse DFTB (init once, then only update coordinates).
pub struct SparseDftb {
    builder: HamiltonianBuilder,
    sk_dir: String,
    species: Vec<String>,
    species_code: Vec<u8>,
    atom_n_orb: Vec<u8>,
    q0: Vec<f64>,
    nocc: f32,
    n_atom: usize,
    n_orbs: usize,
    n_pad: usize,
    gamma: GammaTable,
    cfg: SparseDftbConfig,
    mask: (Vec<u32>, Vec<u32>),
    r_cut_ang: f64,
    full_mask: bool,

    ws: SparseSystemWorkspace,
    h_bsr: Bsr4Matrix,
    s_bsr: Bsr4Matrix,
    hscc_bsr: Bsr4Matrix,

    coords: Vec<[f64; 3]>,
    h0_phys: Vec<f64>,
    s_phys: Vec<f64>,
    h0_pad: Vec<f32>,
    s_pad: Vec<f32>,
    h_scc_pad: Vec<f32>,
    k_pad: Vec<f32>,
    q: Vec<f64>,
    v: Vec<f64>,
    e_rep: f64,
    last: SparseDftbEnergy,
    z_valid: bool,

    fire_v: Vec<[f64; 3]>,
    fire_dt: f64,
    fire_alpha: f64,
    fire_n_pos: usize,
}

impl SparseDftb {
    /// Build the engine. Compiles OpenCL and allocates every BSR buffer once.
    pub fn new(sk: SkData, sk_dir: &str, species: Vec<String>, coords: Vec<[f64; 3]>) -> Result<Self> {
        Self::with_config(sk, sk_dir, species, coords, SparseDftbConfig::default())
    }

    pub fn with_config(sk: SkData, sk_dir: &str, species: Vec<String>, coords: Vec<[f64; 3]>, cfg: SparseDftbConfig) -> Result<Self> {
        let n_atom = species.len();
        if n_atom == 0 { return Err(DftbError::InvalidInput("SparseDftb: no atoms".into())); }
        if coords.len() != n_atom {
            return Err(DftbError::InvalidInput(format!(
                "SparseDftb::new: coords.len()={} != n_atom {n_atom}", coords.len()
            )));
        }
        for (i, c) in coords.iter().enumerate() {
            if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() {
                return Err(DftbError::InvalidInput(format!("SparseDftb: non-finite coord atom {i} {c:?}")));
            }
        }
        let q0 = valence_q0(&species)?;
        let n_elec: f64 = q0.iter().sum();
        let nocc = (n_elec / 2.0).round() as f32;
        if nocc < 0.5 {
            return Err(DftbError::InvalidInput(format!("SparseDftb: nocc={nocc} from q0={q0:?}")));
        }
        let builder = HamiltonianBuilder::new(sk);
        for sp in &species {
            builder.sk.onsite(sp).map_err(|e| DftbError::InvalidInput(format!("SparseDftb: SK onsite '{sp}': {e}")))?;
        }
        let ctx = SystemContext::from_sk_data(&builder.sk, &species)?;
        if ctx.n_atoms != n_atom {
            return Err(DftbError::InvalidInput(format!("SparseDftb: ctx.n_atoms={} != {n_atom}", ctx.n_atoms)));
        }
        for (i, &n) in ctx.atom_n_orb.iter().enumerate() {
            if n != 1 && n != 4 {
                return Err(DftbError::InvalidInput(format!(
                    "SparseDftb: atom {i} n_orb={n} — BSR4 path is s/p only (1 or 4)"
                )));
            }
        }
        let atom_n_orb = ctx.atom_n_orb.clone();
        let species_code = ctx.atom_species.clone();
        let n_orbs = ctx.n_orbs;
        let gamma = GammaTable::from_sk_data(&builder.sk, &species)?;
        let cut_bohr = builder.sk.pairs.values().map(|t| t.cutoff()).fold(0.0_f64, f64::max);
        if !(cut_bohr > 0.0) {
            return Err(DftbError::InvalidInput(format!("SparseDftb: SK cutoff {cut_bohr}")));
        }
        let r_cut_ang = cut_bohr / ANG2BOHR + cfg.r_skin_ang;
        let full_mask = cfg.full_mask.unwrap_or(n_atom <= FULL_MASK_ATOMS);
        let mask = if full_mask { build_full_mask(n_atom) } else { build_geometric_mask(&coords, r_cut_ang) };
        if mask.1.is_empty() {
            return Err(DftbError::InvalidInput("SparseDftb: empty BSR mask".into()));
        }
        for i in 0..n_atom {
            let (a, b) = (mask.0[i] as usize, mask.0[i + 1] as usize);
            if !(a..b).any(|blk| mask.1[blk] as usize == i) {
                return Err(DftbError::InvalidInput(format!("SparseDftb: mask missing diagonal block atom {i}")));
            }
        }

        let gpu = SparseBsr4Gpu::new(SparseBsr4Config::default())?;
        let h_bsr = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone())?;
        let s_bsr = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone())?;
        let hscc_bsr = Bsr4Matrix::from_structure(n_atom, mask.0.clone(), mask.1.clone())?;
        let ws = SparseSystemWorkspace::new(gpu, &h_bsr, &s_bsr, &mask, nocc)?;

        let n_pad = n_atom * BS;
        let n_scc = SparseDftbEnergy {
            e_h0: 0.0, e_scc: 0.0, e_el: 0.0, e_rep: 0.0, e_tot: 0.0,
            q: q0.clone(), tr_ks: 0.0, r_i: 0.0, n_scc: 0, tc2_iters: 0,
            k_pad: vec![], h_scc_pad: vec![], v: vec![0.0; n_atom],
            r_scc: 0.0, r_h: f32::NAN,
        };
        eprintln!(
            "[SparseDftb] n_atom={n_atom} n_orbs={n_orbs} n_pad={n_pad} nocc={nocc} nnz_hs={} full_mask={full_mask} r_cut={r_cut_ang:.3} Å  valence_q0={q0:?}",
            mask.1.len()
        );
        let mut eng = Self {
            builder, sk_dir: sk_dir.to_string(), species, species_code,
            atom_n_orb, q0: q0.clone(), nocc, n_atom, n_orbs, n_pad, gamma, cfg,
            mask, r_cut_ang, full_mask, ws, h_bsr, s_bsr, hscc_bsr,
            coords: coords.clone(),
            h0_phys: vec![0.0; n_orbs * n_orbs],
            s_phys: vec![0.0; n_orbs * n_orbs],
            h0_pad: vec![0.0; n_pad * n_pad],
            s_pad: vec![0.0; n_pad * n_pad],
            h_scc_pad: vec![0.0; n_pad * n_pad],
            k_pad: vec![0.0; n_pad * n_pad],
            q: q0, v: vec![0.0; n_atom], e_rep: 0.0, last: n_scc, z_valid: false,
            fire_v: vec![[0.0; 3]; n_atom], fire_dt: 0.1, fire_alpha: 0.1, fire_n_pos: 0,
        };
        eng.set_coords(&coords)?;
        Ok(eng)
    }

    pub fn n_atom(&self) -> usize { self.n_atom }
    pub fn n_orbs(&self) -> usize { self.n_orbs }
    pub fn k_pad(&self) -> &[f32] { &self.k_pad }
    pub fn h_scc_pad(&self) -> &[f32] { &self.h_scc_pad }
    pub fn coords(&self) -> &[[f64; 3]] { &self.coords }
    pub fn last_energy(&self) -> &SparseDftbEnergy { &self.last }
    pub fn q0(&self) -> &[f64] { &self.q0 }
    pub fn gpu(&self) -> &SparseBsr4Gpu { self.ws.gpu() }

    /// Per-geometry: H0/S values only. Fails if the frozen mask no longer covers neighbors.
    pub fn set_coords(&mut self, coords: &[[f64; 3]]) -> Result<()> {
        if coords.len() != self.n_atom {
            return Err(DftbError::InvalidInput(format!(
                "set_coords: len {} != n_atom {}", coords.len(), self.n_atom
            )));
        }
        for (i, c) in coords.iter().enumerate() {
            if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() {
                return Err(DftbError::InvalidInput(format!("set_coords: non-finite atom {i} {c:?}")));
            }
        }
        if !self.full_mask {
            let live = build_geometric_mask(coords, self.r_cut_ang);
            if live.0 != self.mask.0 || live.1 != self.mask.1 {
                return Err(DftbError::InvalidInput(format!(
                    "set_coords: neighbor mask changed (was {} blocks, now {}). Topology is frozen at SparseDftb::new; rebuild the engine.",
                    self.mask.1.len(), live.1.len()
                )));
            }
        }
        self.coords.copy_from_slice(coords);
        let ham = self.builder.build_non_scc(&self.species, &self.coords)?;
        if ham.h0.nrows() != self.n_orbs {
            return Err(DftbError::InvalidInput(format!(
                "set_coords: H0 {}×{} != n_orbs {}", ham.h0.nrows(), ham.h0.ncols(), self.n_orbs
            )));
        }
        let n = self.n_orbs;
        for i in 0..n {
            for j in 0..n {
                self.h0_phys[i * n + j] = ham.h0[(i, j)];
                self.s_phys[i * n + j] = ham.s[(i, j)];
            }
        }
        pad_physical_to_bsr4_into(&self.h0_phys, &self.s_phys, &self.atom_n_orb, E_DUMMY, &mut self.h0_pad, &mut self.s_pad);
        fill_bsr_values_from_dense(self.n_atom, &self.h0_pad, &self.mask.0, &self.mask.1, &mut self.h_bsr.values);
        fill_bsr_values_from_dense(self.n_atom, &self.s_pad, &self.mask.0, &self.mask.1, &mut self.s_bsr.values);
        self.ws.upload_h0(&self.h_bsr)?;
        self.ws.upload_s(&self.s_bsr)?;
        self.e_rep = repulsive_energy(&self.sk_dir, &self.species, &self.coords)?;
        if !self.e_rep.is_finite() {
            panic!("set_coords: non-finite E_rep={} n_atom={}", self.e_rep, self.n_atom);
        }
        self.z_valid = false;
        Ok(())
    }

    /// Warm-start charges (Hessian ±h columns). Length must match n_atom.
    pub fn set_q(&mut self, q: &[f64]) -> Result<()> {
        if q.len() != self.n_atom {
            return Err(DftbError::InvalidInput(format!("set_q: len {} != n_atom {}", q.len(), self.n_atom)));
        }
        for (i, &qi) in q.iter().enumerate() {
            if !qi.is_finite() {
                return Err(DftbError::InvalidInput(format!("set_q: non-finite q[{i}]={qi}")));
            }
        }
        self.q.copy_from_slice(q);
        Ok(())
    }

    /// Self-consistent charges. Warm-starts from the previous `q`. Z is computed once per geometry.
    pub fn scc(&mut self, max_iter: usize, rms_tol: f64) -> Result<SparseDftbScc> {
        let cap = max_iter.min(self.cfg.max_scc).min(SCC_MAX);
        if !self.z_valid {
            let (rz, z_iters) = self.ws.compute_z(self.cfg.ns_max, self.cfg.ns_tol, 5)?;
            eprintln!("  [SparseDftb] NS (once per geometry): {z_iters} iters, R_Z={rz:.3e} (host ||I−T|| of downloaded T)");
            self.z_valid = true;
        }
        let mix = self.cfg.mix;
        let mut rms_prev = f64::INFINITY;
        let mut last_info = SparseDftbScc { n_iters: 0, rms: f64::INFINITY, r_scc: 0.0, tr_ks: 0.0, r_i: 0.0 };
        for it in 0..cap {
            let dq: Vec<f64> = self.q.iter().zip(self.q0.iter()).map(|(a, b)| a - b).collect();
            compute_intra_shifts(&self.coords, &self.species_code, &dq, &self.gamma, &mut self.v);
            apply_shift_padded_into(&self.h0_pad, &self.s_pad, &self.v, &self.atom_n_orb, &mut self.h_scc_pad);
            fill_bsr_values_from_dense(self.n_atom, &self.h_scc_pad, &self.mask.0, &self.mask.1, &mut self.hscc_bsr.values);
            self.ws.upload_h_scc(&self.hscc_bsr)?;
            let (r_i, tr, tc2_iters) = self.ws.purify_hscc(self.cfg.tc2_max, self.cfg.tc2_tol)?;
            if (tr - self.nocc).abs() > TC2_TRACE_TOL {
                return Err(DftbError::InvalidInput(format!(
                    "SparseDftb SCC iter {it}: Tr(KS)={tr} far from Nocc={} (tol={TC2_TRACE_TOL})", self.nocc
                )));
            }
            let q_new = self.mulliken_checked(tr, it)?;
            self.ws.k_to_dense_into(&mut self.k_pad)?;
            let mut rms = 0.0f64;
            let mut max_dq = 0.0f64;
            for a in 0..self.n_atom {
                let d = q_new[a] - self.q[a];
                rms += d * d;
                max_dq = max_dq.max(d.abs());
            }
            rms = (rms / self.n_atom as f64).sqrt();
            self.store_energy(&q_new, tr, r_i, it + 1, tc2_iters, rms, f32::NAN);
            if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                eprintln!(
                    "  [SparseDftb SCC] iter {it:3}  rms={rms:.3e}  max|dq|={max_dq:.3e}  E_el={:.8}  E_tot={:.8}  R_I={r_i:.3e}  Tr(KS)={tr:.6}",
                    self.last.e_el, self.last.e_tot
                );
            }
            last_info = SparseDftbScc { n_iters: it + 1, rms, r_scc: rms, tr_ks: tr, r_i };
            if rms < rms_tol {
                return self.finalize_scc(q_new, it, last_info);
            }
            for a in 0..self.n_atom {
                self.q[a] = (1.0 - mix) * self.q[a] + mix * q_new[a];
            }
            if it > 2 && rms > rms_prev * 2.0 && rms > 1e-2 {
                return Err(DftbError::InvalidInput(format!(
                    "SparseDftb SCC diverging at iter {it}: rms={rms:.3e} (prev {rms_prev:.3e})"
                )));
            }
            rms_prev = rms;
        }
        Err(DftbError::InvalidInput(format!(
            "SparseDftb SCC did not converge in {cap} iters, last rms={:.3e} E_tot={:.8}",
            last_info.rms, self.last.e_tot
        )))
    }

    fn finalize_scc(&mut self, q_new: Vec<f64>, it: usize, mut info: SparseDftbScc) -> Result<SparseDftbScc> {
        let dq_out: Vec<f64> = q_new.iter().zip(self.q0.iter()).map(|(a, b)| a - b).collect();
        compute_intra_shifts(&self.coords, &self.species_code, &dq_out, &self.gamma, &mut self.v);
        apply_shift_padded_into(&self.h0_pad, &self.s_pad, &self.v, &self.atom_n_orb, &mut self.h_scc_pad);
        fill_bsr_values_from_dense(self.n_atom, &self.h_scc_pad, &self.mask.0, &self.mask.1, &mut self.hscc_bsr.values);
        self.ws.upload_h_scc(&self.hscc_bsr)?;
        let (r_i, tr, tc2_f) = self.ws.purify_hscc(self.cfg.tc2_max, self.cfg.tc2_tol)?;
        let q_fin = self.mulliken_checked(tr, it)?;
        let mut r_fin = 0.0f64;
        for a in 0..self.n_atom {
            let d = q_fin[a] - q_new[a];
            r_fin += d * d;
        }
        r_fin = (r_fin / self.n_atom as f64).sqrt();
        self.ws.k_to_dense_into(&mut self.k_pad)?;
        self.q.copy_from_slice(&q_fin);
        self.store_energy(&q_fin, tr, r_i, info.n_iters, tc2_f, r_fin, f32::NAN);
        info.r_scc = r_fin;
        info.tr_ks = tr;
        info.r_i = r_i;
        eprintln!(
            "  [SparseDftb] finalize  r_scc={r_fin:.3e}  Tr(KS)={tr:.6}  R_I={r_i:.3e}  E_tot={:.8}",
            self.last.e_tot
        );
        Ok(info)
    }

    fn mulliken_checked(&mut self, tr: f32, it: usize) -> Result<Vec<f64>> {
        let q_f32 = self.ws.mulliken_charges()?;
        if q_f32.len() != self.n_atom {
            return Err(DftbError::InvalidInput(format!("Mulliken len {} != n_atom {}", q_f32.len(), self.n_atom)));
        }
        let q: Vec<f64> = q_f32.iter().map(|&x| x as f64).collect();
        let qsum: f64 = q.iter().sum();
        let n_elec = 2.0 * self.nocc as f64;
        if (qsum - n_elec).abs() > 0.5 {
            return Err(DftbError::InvalidInput(format!(
                "SCC iter {it}: sum(q)={qsum:.6} != N_elec={n_elec} (Tr(KS)={tr})"
            )));
        }
        Ok(q)
    }

    fn store_energy(&mut self, q: &[f64], tr: f32, r_i: f32, n_scc: usize, tc2_iters: usize, r_scc: f64, r_h: f32) {
        let dq: Vec<f64> = q.iter().zip(self.q0.iter()).map(|(a, b)| a - b).collect();
        let e_h0 = 2.0 * trace_ab(&self.k_pad, &self.h0_pad, self.n_pad);
        let e_scc = 0.5 * dq.iter().zip(self.v.iter()).map(|(d, vi)| d * vi).sum::<f64>();
        let e_el = e_h0 + e_scc;
        if !e_el.is_finite() || !self.e_rep.is_finite() {
            panic!("SparseDftb energy non-finite: E_el={e_el} E_rep={} n_scc={n_scc}", self.e_rep);
        }
        self.last = SparseDftbEnergy {
            e_h0, e_scc, e_el, e_rep: self.e_rep, e_tot: e_el + self.e_rep,
            q: q.to_vec(), tr_ks: tr, r_i, n_scc, tc2_iters,
            k_pad: vec![], h_scc_pad: vec![], v: self.v.clone(), r_scc, r_h,
        };
    }

    pub fn energy(&self) -> Result<f64> {
        if self.last.n_scc == 0 {
            return Err(DftbError::InvalidInput("energy: no SCC yet — call scc first".into()));
        }
        if !self.last.e_tot.is_finite() {
            return Err(DftbError::InvalidInput(format!("energy: last E_tot={} — call scc first", self.last.e_tot)));
        }
        Ok(self.last.e_tot)
    }

    /// Analytic forces, Hartree/Å. Downloads K once; CPU `compute_forces_from_dw` (H-bond GPU kernels are read-only).
    pub fn forces(&mut self) -> Result<Forces> {
        if self.k_pad.iter().all(|&x| x == 0.0) {
            return Err(DftbError::InvalidInput("forces: K is zero — call scc first".into()));
        }
        sparse_analytic_forces(
            &self.builder.sk, &self.species, &self.coords, &self.atom_n_orb,
            &self.k_pad, &self.h_scc_pad, &self.q, &self.q0, &self.sk_dir,
        )
    }

    /// One FIRE step. Call `scc` first. Returns max |F|.
    pub fn fire_step(&mut self, f_tol: f64) -> Result<f64> {
        let f = self.forces()?;
        let mut max_f = 0.0f64;
        let mut p = 0.0f64;
        for i in 0..self.n_atom {
            for c in 0..3 {
                let fi = f.forces[i][c];
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
        for i in 0..self.n_atom {
            let fx = f.forces[i][0]; let fy = f.forces[i][1]; let fz = f.forces[i][2];
            let fnorm = (fx * fx + fy * fy + fz * fz).sqrt();
            let (hx, hy, hz) = if fnorm > 1e-12 { (fx / fnorm, fy / fnorm, fz / fnorm) } else { (0.0, 0.0, 0.0) };
            for (c, h) in [hx, hy, hz].iter().enumerate() {
                self.fire_v[i][c] = (1.0 - self.fire_alpha) * self.fire_v[i][c] + self.fire_alpha * fnorm * h;
            }
            apply_disp(&mut self.coords[i], &self.fire_v[i], fx, fy, fz, self.fire_dt);
            self.fire_v[i][0] += fx * self.fire_dt;
            self.fire_v[i][1] += fy * self.fire_dt;
            self.fire_v[i][2] += fz * self.fire_dt;
        }
        let c = self.coords.clone();
        self.set_coords(&c)?;
        Ok(max_f)
    }

    /// Velocity-Verlet (mass=1). Call `scc` after. Displacement capped at 0.1 Å.
    pub fn md_step(&mut self, dt: f64) -> Result<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(DftbError::InvalidInput(format!("md_step: dt={dt} must be finite and > 0")));
        }
        let f = self.forces()?;
        let mut max_f = 0.0f64;
        for i in 0..self.n_atom {
            let fx = f.forces[i][0]; let fy = f.forces[i][1]; let fz = f.forces[i][2];
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

    /// SCC + FIRE until max|F|<f_tol or `max_steps`. Prints unbuffered progress.
    pub fn relax(&mut self, max_steps: usize, f_tol: f64, scc_tol: f64) -> Result<(usize, f64, f64)> {
        let scc0 = self.scc(SCC_MAX, scc_tol)?;
        eprintln!("[SparseDftb] relax start rms={:.3e} iters={} E={:.8}", scc0.rms, scc0.n_iters, self.last.e_tot);
        let mut max_f = f64::INFINITY;
        let mut step = 0;
        for s in 0..max_steps {
            step = s + 1;
            max_f = self.fire_step(f_tol)?;
            let scc = self.scc(SCC_MAX, scc_tol)?;
            eprintln!(
                "[SparseDftb] FIRE {step}/{max_steps} max|F|={max_f:.4e} E={:.8} rms={:.3e} scc_iters={}",
                self.last.e_tot, scc.rms, scc.n_iters
            );
            if max_f < f_tol { break; }
        }
        Ok((step, max_f, scc0.rms))
    }
}

fn apply_disp(xyz: &mut [f64; 3], v: &[f64; 3], fx: f64, fy: f64, fz: f64, dt: f64) {
    let mut dx = v[0] * dt + 0.5 * fx * dt * dt;
    let mut dy = v[1] * dt + 0.5 * fy * dt * dt;
    let mut dz = v[2] * dt + 0.5 * fz * dt * dt;
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    if d > MAX_FIRE_DISP { let s = MAX_FIRE_DISP / d; dx *= s; dy *= s; dz *= s; }
    xyz[0] += dx; xyz[1] += dy; xyz[2] += dz;
}
