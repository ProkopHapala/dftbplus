//! Persistent sparse DFTB engine — production run loop (manifest §0.4,
//! Phase F rewrite: second GPT-5.6 review R1–R19).
//!
//! GPU analogue of `DftbCpu` / `GpuDftb`. One object owns the OpenCL runtime,
//! compiled BSR4 kernels, SK tables, masks, and every buffer. Tests and MD/FIRE
//! drive this; do not construct a throwaway `SparseBsr4Gpu` per geometry.
//!
//! ```text
//! INIT (once)     SparseDftb::new  — runtime, kernels, SK, masks M_HS/M_K/M_Z,
//!                                    BSR workspace, repulsive tables, host scratch
//! PER GEOMETRY    set_coords       — direct atom-pair→BSR H0/S assembly
//!                                    (no dense matrices), Verlet-skin check,
//!                                    dense f64 γ matrix, cached E_rep
//! SCC             scc              — V upload + device Hscc, K0+TC2 on device;
//!                                    q + scalars only to host
//! ENERGY/FORCES   energy / forces  — 2·Tr(K·H0) sparse masked + ½Δq·V + E_rep;
//!                                    D=2K on M_K, W=2KHK on M_HS → pair contraction
//! MD / FIRE       fire_step / md_step / relax
//! ```
//!
//! **Frozen topology:** masks M_HS (H/S = SK cutoff + Verlet skin), M_K
//! (density kernel), M_Z (inverse overlap) are fixed at `new`. A geometry
//! that moves any atom by more than `skin/2` fails loud — rebuild `new`,
//! do not silently grow CSR in the MD loop. OpenCL `Program`/`Kernel`/
//! `Buffer` are never created in `scc` / `forces` / `fire_step`.
//!
//! **No dense orbital matrices in production** (F2/F3): H0/S are assembled
//! straight into BSR values from atom pairs (`assemble_hs_bsr`); H_scc is
//! built on the device from `V[N]`; the band energy is `2·Tr(K·H0)` as a
//! masked contraction over M_HS — no `k_to_dense`/`trace_ab`. The dense
//! `HamiltonianBuilder` path remains the small-system parity reference.
//!
//! Dense `k_pad`/`h_scc_pad` buffers are materialized only when
//! `cfg.dense_diag` (default: on for n_atom ≤ 64 — test-scale) for the
//! G3 parity test; they are diagnostic, never used by the production path.
//!
//! Direct γ evaluation is O(N²) per geometry — explicit transitional
//! limitation until a long-range accelerator lands (F6/R16).
//!
//! Drive from **`dftb_engine`** (`sparse_new` / `sparse_scc` / `sparse_eval`),
//! not a new `src/bin` or `tests/*.rs` per molecule.
//!
//! `compute_z` uses the device `||I−T||_F` residual (corrected contract —
//! `identity_residual_to_dev` writes the *squared* norm, sqrt on host).
//! Host `newton_schulz_inverse` (allocating) stays in `scc.rs` for G3
//! diagnostics.

use crate::core::error::{DftbError, Result};
use crate::methods::dftb::forces::{
    build_pair_block, parse_all_repulsive, repulsive_energy_cached, Forces, RepulsiveSpline,
};
use crate::methods::dftb::gamma::GammaTable;
use crate::methods::dftb::hamiltonian::{HamiltonianBuilder, SystemContext};
use crate::methods::dftb::sk_data::SkData;
use crate::methods::sparse::bsr4::{
    build_full_mask, build_geometric_mask, Bsr4Matrix, BS, BS2,
};
use crate::methods::sparse::gpu_sparse::{SparseBsr4Config, SparseBsr4Gpu, TC2_TRACE_TOL};
use crate::methods::sparse::scc::{SparseDftbEnergy, E_DUMMY};
use crate::methods::sparse::sparse_forces::{
    hs_pairs_from_mask, hs_taper, sparse_forces_bsr, HsPair, SparseDWWorkspace,
};
use crate::methods::sparse::sparse_system::SparseSystemWorkspace;
use crate::qmqm::mixer::{DiisMixer, Mixer};

const ANG2BOHR: f64 = 1.889_726_133;
const SCC_MAX: usize = 100;
const FULL_MASK_ATOMS: usize = 64;
const MAX_FIRE_DISP: f64 = 0.1; // Å

/// Result of one `scc` call.
#[derive(Debug, Clone)]
pub struct SparseDftbScc {
    pub n_iters: usize,
    pub rms: f64,
    /// Stationarity gap: rms(q_out − q_in) at the reported state (F4 — the
    /// residual of the *returned* charges, not a stale iterate).
    pub r_scc: f64,
    pub tr_ks: f32,
    /// Relative idempotency residual ‖KSK−K‖/‖K‖ (R12).
    pub r_i: f32,
    /// Hamiltonian stationarity ‖HKS−SKH‖_F/(2‖HKS‖_F) on the final state (R11).
    pub r_h: f32,
}

/// Tunables. `new` uses `Default`. Changing n_atom or mask requires a new engine.
#[derive(Debug, Clone)]
pub struct SparseDftbConfig {
    pub mix: f64,
    pub scc_tol: f64,
    pub max_scc: usize,
    pub ns_max: usize,
    /// Absolute ‖I−Z·S‖_F/√N_orb Newton–Schulz tolerance.
    pub ns_tol: f32,
    pub tc2_max: usize,
    /// Relative ‖KSK−K‖/‖K‖ TC2 tolerance (R12).
    pub tc2_tol: f32,
    /// `None` = full mask if n_atom ≤ 64, else geometric SK cutoff + `r_skin_ang`.
    pub full_mask: Option<bool>,
    /// Verlet skin on the H/S mask (Å). Topology freezes at `new`; any atom
    /// moving more than `r_skin_ang/2` from its build position fails loudly.
    pub r_skin_ang: f64,
    /// Density-kernel mask radius (Å, same convention as M_HS: this is the
    /// *mask radius*, not a physical cutoff). `None` = same as M_HS (full
    /// structural support — the current-truncation default).
    pub r_k_ang: Option<f64>,
    /// Inverse-overlap mask radius (Å). `None` = same as M_HS.
    pub r_z_ang: Option<f64>,
    /// Physical H/S truncation radius (Å). `None` = full SK table range
    /// (~10.6 Å for matsci — table *end*, not decay). When set, every pair
    /// block is multiplied by a cosine taper going 1→0 over
    /// [r_trunc − taper_w, r_trunc]; the mask radius becomes
    /// r_trunc + r_skin_ang so moving atoms never pop nonzero interactions
    /// into existence outside M_HS. Measured decay (1e-4 Ha on H, s+p):
    /// matsci Si-Si ~7 Å, pbc Si-Si ~5.4 Å, mio/3ob C-C ~4.7-5.1 Å.
    pub r_trunc_ang: Option<f64>,
    /// Taper window width (Å) ending exactly at r_trunc_ang. Default 1.0.
    pub taper_w_ang: f64,
    /// Materialize dense padded `k_pad`/`h_scc_pad` after each SCC for
    /// diagnostics/tests (G3 parity). `None` = on iff n_atom ≤ 64.
    /// Never used by the production force/energy path.
    pub dense_diag: Option<bool>,
    /// DIIS history depth for charge mixing (0 = pure linear `mix`). The
    /// fixed-point map on larger systems is unstable under linear mixing
    /// (Si10H16 oscillates at rms~1.4 under α=0.5 — charge sloshing);
    /// DIIS with linear fallback is required for real nanocrystals.
    pub diis_hist: usize,
}

impl Default for SparseDftbConfig {
    fn default() -> Self {
        Self {
            mix: 0.5, scc_tol: 1e-5, max_scc: 80,
            ns_max: 50, ns_tol: 1e-5, tc2_max: 80, tc2_tol: 1e-4,
            full_mask: None, r_skin_ang: 1.0,
            r_k_ang: None, r_z_ang: None, r_trunc_ang: None, taper_w_ang: 1.0,
            dense_diag: None,
            diis_hist: 8,
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
    species: Vec<String>,
    species_names: Vec<String>,    // unique species, ctx order (repulsive indexing)
    species_code: Vec<u8>,
    atom_n_orb: Vec<u8>,
    onsite_orb: Vec<[f64; 4]>,     // onsite energies expanded to lanes (physical only)
    q0: Vec<f64>,
    nocc: f32,
    n_atom: usize,
    n_orbs: usize,
    n_species: usize,
    gamma: GammaTable,
    /// Dense f64 γ matrix [N×N] — rebuilt per geometry, O(N²) memory/work.
    /// Transitional (R16): direct γ evaluation until a long-range accelerator.
    gmat: Vec<f64>,
    /// Pre-parsed repulsive splines (once at `new` — F6).
    repulsive: Vec<Option<RepulsiveSpline>>,
    cfg: SparseDftbConfig,

    // Frozen topology
    m_hs: (Vec<u32>, Vec<u32>),
    full_mask: bool,
    /// Cosine taper (r_start_ang, width_ang) on every H/S pair block, from
    /// cfg.r_trunc_ang/taper_w_ang. None = untruncated (full SK range).
    hs_taper: Option<(f64, f64)>,
    /// Coordinates at mask construction (Verlet reference).
    coords_build: Vec<[f64; 3]>,
    /// Unique off-diagonal pairs of M_HS (i<j) with BSR block indices —
    /// built once, used by H/S assembly AND the force contraction.
    hs_pairs: Vec<HsPair>,
    hs_diag: Vec<u32>,   // m_hs block index of (i,i)
    k_diag: Vec<u32>,    // m_k block index of (i,i)

    ws: SparseSystemWorkspace,
    dw_ws: SparseDWWorkspace,
    h_bsr: Bsr4Matrix,
    s_bsr: Bsr4Matrix,

    coords: Vec<[f64; 3]>,
    /// Downloaded K values on M_K / W values on M_HS (per-force scratch).
    k_vals: Vec<f32>,
    w_vals: Vec<f32>,
    /// Diagnostic dense pads — only filled when `dense_diag` (test-scale).
    k_pad: Vec<f32>,
    h_scc_pad: Vec<f32>,
    q: Vec<f64>,       // state charges (consistent with `last` energy/forces)
    v: Vec<f64>,       // atom potentials V(q_state)
    v_f32: Vec<f32>,   // upload scratch
    /// DIIS charge mixer (None when cfg.diis_hist == 0 → pure linear `mix`).
    /// Linear mixing is unstable on real nanocrystals (Si10H16 oscillates at
    /// rms~1.4 under α=0.5 — charge sloshing); DIIS is the default for a
    /// reason. Preallocated at `new`; reset on set_coords/set_q (the
    /// fixed-point map changed → stale history extrapolates wrongly).
    mixer: Option<DiisMixer>,
    q_res: Vec<f64>,   // mixer residual scratch q_out − q_in
    e_rep: f64,
    last: SparseDftbEnergy,
    /// Z is a converged inverse of the *current* S.
    z_valid: bool,
    /// The Z buffer holds a usable inverse from a *previous* geometry —
    /// warm-start NS with cold fallback (R6 physics: Z changes slowly).
    z_warm: bool,
    dense_diag: bool,

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
        let n_species = ctx.n_species;
        let atom_n_orb = ctx.atom_n_orb.clone();
        let species_code = ctx.atom_species.clone();
        let n_orbs = ctx.n_orbs;
        // Unique species names in ctx index order (for repulsive tables).
        let mut species_names: Vec<String> = Vec::new();
        for sp in &species {
            if !species_names.iter().any(|s| s == sp) { species_names.push(sp.clone()); }
        }
        if species_names.len() != n_species {
            return Err(DftbError::InvalidInput(format!(
                "SparseDftb: unique species count {} != ctx.n_species {n_species}", species_names.len()
            )));
        }
        // Onsite energies expanded to padded lanes (physical lanes only).
        let mut onsite_orb = vec![[0.0f64; 4]; n_atom];
        for i in 0..n_atom {
            let si = ctx.atom_species[i] as usize;
            let p = ctx.species_onsite[si];
            let mut off = 0usize;
            for &l in ctx.species_ang[si] {
                let e = match l { 0 => p.e_s, 1 => p.e_p, _ => 0.0 };
                for k in 0..(2 * l + 1) as usize { onsite_orb[i][off + k] = e; }
                off += (2 * l + 1) as usize;
            }
        }
        let gamma = GammaTable::from_sk_data(&builder.sk, &species)?;
        let repulsive = parse_all_repulsive(sk_dir, &species_names, n_species)?;

        // ── Frozen topology (R3): M_HS = SK cutoff + skin; M_K/M_Z default
        // to the same support (config can shrink them). All in Å units here
        // (mask built on Å coords with an Å radius; SK eval converts to Bohr).
        let cut_bohr = builder.sk.pairs.values().map(|t| t.cutoff()).fold(0.0_f64, f64::max);
        if !(cut_bohr > 0.0) {
            return Err(DftbError::InvalidInput(format!("SparseDftb: SK cutoff {cut_bohr}")));
        }
        // M_HS shrinks with r_trunc (H/S physics); M_K/M_Z default to the
        // FULL SK-table radius — K/Z decay is set by the gap, not by the
        // Hamiltonian range, so truncating their support along with H/S
        // puts a floor on TC2 convergence (observed: R_I limit-cycle ~1e-4
        // on cube_si65 when M_K was cut to 10 Å).
        let r_full_ang = cut_bohr / ANG2BOHR + cfg.r_skin_ang;
        let r_hs_ang = cfg.r_trunc_ang.unwrap_or(cut_bohr / ANG2BOHR) + cfg.r_skin_ang;
        if cfg.r_trunc_ang.is_some() && !(cfg.taper_w_ang > 0.0) {
            return Err(DftbError::InvalidInput(format!(
                "SparseDftb: taper_w_ang must be > 0 when r_trunc_ang is set (got {})", cfg.taper_w_ang
            )));
        }
        let hs_taper = cfg.r_trunc_ang.map(|rt| (rt - cfg.taper_w_ang, cfg.taper_w_ang));
        let full_mask = cfg.full_mask.unwrap_or(n_atom <= FULL_MASK_ATOMS);
        let m_hs = if full_mask { build_full_mask(n_atom) } else { build_geometric_mask(&coords, r_hs_ang) };
        let m_k = if full_mask {
            build_full_mask(n_atom)
        } else {
            build_geometric_mask(&coords, cfg.r_k_ang.unwrap_or(r_full_ang))
        };
        let m_z = if full_mask {
            build_full_mask(n_atom)
        } else {
            build_geometric_mask(&coords, cfg.r_z_ang.unwrap_or(r_full_ang))
        };
        for (name, m) in [("m_hs", &m_hs), ("m_k", &m_k), ("m_z", &m_z)] {
            if m.1.is_empty() {
                return Err(DftbError::InvalidInput(format!("SparseDftb: empty BSR mask {name}")));
            }
            for i in 0..n_atom {
                let (a, b) = (m.0[i] as usize, m.0[i + 1] as usize);
                if !(a..b).any(|blk| m.1[blk] as usize == i) {
                    return Err(DftbError::InvalidInput(format!("SparseDftb: mask {name} missing diagonal block atom {i}")));
                }
            }
        }
        // Diagonal block indices + pair list (built once — frozen topology).
        let hs_diag: Vec<u32> = (0..n_atom).map(|i| {
            let (a, b) = (m_hs.0[i] as usize, m_hs.0[i + 1] as usize);
            (a..b).find(|&blk| m_hs.1[blk] as usize == i)
                .map(|b| b as u32)
                .expect("checked above")
        }).collect();
        let k_diag: Vec<u32> = (0..n_atom).map(|i| {
            let (a, b) = (m_k.0[i] as usize, m_k.0[i + 1] as usize);
            (a..b).find(|&blk| m_k.1[blk] as usize == i)
                .map(|b| b as u32)
                .expect("checked above")
        }).collect();
        let hs_pairs = hs_pairs_from_mask(&m_hs, &m_k, n_atom);
        for p in &hs_pairs {
            if p.b_ji < 0 {
                return Err(DftbError::InvalidInput(format!(
                    "SparseDftb: M_HS not symmetric — ({},{}) present, transpose missing", p.i, p.j
                )));
            }
        }

        let gpu = SparseBsr4Gpu::new(SparseBsr4Config::default())?;
        let h_bsr = Bsr4Matrix::from_structure(n_atom, m_hs.0.clone(), m_hs.1.clone())?;
        let s_bsr = Bsr4Matrix::from_structure(n_atom, m_hs.0.clone(), m_hs.1.clone())?;
        let ws = SparseSystemWorkspace::new(gpu, &h_bsr, &s_bsr, &m_k, &m_z, &atom_n_orb, nocc)?;
        let dw_ws = SparseDWWorkspace::new(ws.gpu(), ws.k_struct(), ws.hs_struct())?;

        let n_pad = n_atom * BS;
        let n_scc = SparseDftbEnergy {
            e_h0: 0.0, e_scc: 0.0, e_el: 0.0, e_rep: 0.0, e_tot: 0.0,
            q: q0.clone(), tr_ks: 0.0, r_i: 0.0, n_scc: 0, tc2_iters: 0,
            k_pad: vec![], h_scc_pad: vec![], v: vec![0.0; n_atom],
            r_scc: 0.0, r_h: f32::NAN,
        };
        let dense_diag = cfg.dense_diag.unwrap_or(n_atom <= FULL_MASK_ATOMS);
        let diis_hist = cfg.diis_hist;
        let mix_alpha = cfg.mix;
        eprintln!(
            "[SparseDftb] n_atom={n_atom} n_orbs={n_orbs} nocc={nocc} nnz_hs={} nnz_k={} nnz_z={} full_mask={full_mask} r_hs={r_hs_ang:.3} Å taper={:?} dense_diag={dense_diag}  valence_q0={q0:?}",
            m_hs.1.len(), m_k.1.len(), m_z.1.len(), hs_taper
        );
        let mut eng = Self {
            builder, species, species_names, species_code,
            atom_n_orb, onsite_orb, q0: q0.clone(), nocc, n_atom, n_orbs, n_species,
            gamma, gmat: vec![0.0; n_atom * n_atom], repulsive, cfg,
            m_hs, full_mask, hs_taper, coords_build: coords.clone(),
            hs_pairs, hs_diag, k_diag,
            ws, dw_ws, h_bsr, s_bsr,
            coords: coords.clone(),
            k_vals: vec![0.0; 0], w_vals: vec![0.0; 0],
            k_pad: if dense_diag { vec![0.0; n_pad * n_pad] } else { Vec::new() },
            h_scc_pad: if dense_diag { vec![0.0; n_pad * n_pad] } else { Vec::new() },
            q: q0, v: vec![0.0; n_atom], v_f32: vec![0.0; n_atom],
            mixer: if diis_hist > 0 {
                let mut m = DiisMixer::new(diis_hist, n_atom);
                m.alpha = mix_alpha;   // linear fallback α = cfg.mix
                Some(m)
            } else { None },
            q_res: vec![0.0; n_atom],
            e_rep: 0.0, last: n_scc, z_valid: false, z_warm: false, dense_diag,
            fire_v: vec![[0.0; 3]; n_atom], fire_dt: 0.1, fire_alpha: 0.1, fire_n_pos: 0,
        };
        eng.k_vals = vec![0.0; eng.ws.k_struct().nblock * BS2];
        eng.w_vals = vec![0.0; eng.ws.hs_struct().nblock * BS2];
        eng.set_coords(&coords)?;
        Ok(eng)
    }

    pub fn n_atom(&self) -> usize { self.n_atom }
    pub fn n_orbs(&self) -> usize { self.n_orbs }
    /// Relative TC2 tolerance ‖KSK−K‖/‖K‖. The achieved r_I sets the Mulliken
    /// charge noise floor — SCC rms cannot converge below ~r_I (Si10H16:
    /// tc2_tol=1e-4 gives r_I~2e-5 → SCC floor ~3e-6; a 1e-6 SCC target needs
    /// tc2_tol≲1e-6). Cheap to tighten: TC2 converges geometrically.
    /// Set the Newton–Schulz Z residual tolerance at runtime (f32 floor /
    /// mask-truncation plateau ~1e-5..1e-4 on truncated M_Z).
    pub fn set_ns_tol(&mut self, tol: f32) {
        if !(tol > 0.0) || !tol.is_finite() { panic!("set_ns_tol: tol={tol}"); }
        self.cfg.ns_tol = tol;
    }

    pub fn set_tc2_tol(&mut self, tol: f32) {
        if !(tol > 0.0) || !tol.is_finite() { panic!("set_tc2_tol: tol={tol}"); }
        self.cfg.tc2_tol = tol;
    }
    /// Diagnostic dense padded K (only when `dense_diag`; else empty).
    pub fn k_pad(&self) -> &[f32] { &self.k_pad }
    /// Diagnostic dense padded H_scc (only when `dense_diag`; else empty).
    pub fn h_scc_pad(&self) -> &[f32] { &self.h_scc_pad }
    pub fn coords(&self) -> &[[f64; 3]] { &self.coords }
    pub fn last_energy(&self) -> &SparseDftbEnergy { &self.last }
    pub fn q0(&self) -> &[f64] { &self.q0 }
    pub fn gpu(&self) -> &SparseBsr4Gpu { self.ws.gpu() }
    /// Host H0 on M_HS (diagnostic/parity — the same values uploaded to the GPU).
    pub fn h_bsr(&self) -> &Bsr4Matrix { &self.h_bsr }
    /// Host S on M_HS.
    pub fn s_bsr(&self) -> &Bsr4Matrix { &self.s_bsr }

    /// Per-geometry update: direct atom-pair→BSR H0/S assembly (no dense
    /// matrices — F2). Fails loud when the frozen topology no longer covers
    /// the geometry (Verlet skin exhausted) — rebuild the engine instead of
    /// silently mutating CSR.
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
        // Verlet-skin check (F2/R4): M_HS was built at coords_build with
        // radius r_cut + skin. Any pair can newly enter the physical cutoff
        // only if an atom moved > skin/2 relative to the build geometry.
        if !self.full_mask {
            let mut dmax = 0.0f64;
            for i in 0..self.n_atom {
                for c in 0..3 {
                    dmax = dmax.max((coords[i][c] - self.coords_build[i][c]).abs());
                }
            }
            if 2.0 * dmax > self.cfg.r_skin_ang {
                return Err(DftbError::InvalidInput(format!(
                    "set_coords: Verlet skin exhausted — max |ΔR|={dmax:.4} Å vs skin/2={:.4} Å. \
                     Topology is frozen at SparseDftb::new; rebuild the engine.",
                    self.cfg.r_skin_ang / 2.0
                )));
            }
        }
        self.coords.copy_from_slice(coords);
        // SystemContext is O(n_atom + n_species²) table construction per
        // geometry — cheap vs the O(nnz) assembly/contraction, and not
        // storable in self (borrows &'a SkData → self-referential).
        let ctx = SystemContext::from_sk_data(&self.builder.sk, &self.species)?;
        assemble_hs_bsr(&ctx, &self.coords, &self.hs_pairs, &self.hs_diag, &self.onsite_orb, self.hs_taper, &mut self.h_bsr.values, &mut self.s_bsr.values)?;
        self.ws.upload_h0(&self.h_bsr)?;
        self.ws.upload_s(&self.s_bsr)?;
        // Dense f64 γ matrix (F6/R16): O(N²), rebuilt per geometry.
        for i in 0..self.n_atom {
            self.gmat[i * self.n_atom + i] = self.gamma.u(self.species_code[i]);
            for j in (i + 1)..self.n_atom {
                let dx = self.coords[i][0] - self.coords[j][0];
                let dy = self.coords[i][1] - self.coords[j][1];
                let dz = self.coords[i][2] - self.coords[j][2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt() * ANG2BOHR;
                let g = self.gamma.gamma(r, self.species_code[i], self.species_code[j]);
                self.gmat[i * self.n_atom + j] = g;
                self.gmat[j * self.n_atom + i] = g;
            }
        }
        self.e_rep = repulsive_energy_cached(
            &self.coords, &self.species_code, &self.species_names, &self.repulsive, self.n_species,
        )?;
        if !self.e_rep.is_finite() {
            panic!("set_coords: non-finite E_rep={} n_atom={}", self.e_rep, self.n_atom);
        }
        self.z_valid = false;
        // Geometry changes the fixed-point map only smoothly; keep the DIIS
        // subspace (reset_iter_only) — warm-started FD/relax columns converge
        // in a few iters instead of re-fighting the unstable sloshing mode.
        if let Some(m) = &mut self.mixer { m.reset_iter_only(); }
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
        if let Some(m) = &mut self.mixer { m.reset_iter_only(); }  // same map, new point
        Ok(())
    }

    /// V = γ·Δq via the cached dense f64 γ matrix (O(N²) matvec — same math
    /// as `compute_intra_shifts`, no per-pair γ re-evaluation).
    fn compute_v(&mut self) {
        let n = self.n_atom;
        self.v.fill(0.0);
        for i in 0..n {
            let mut acc = 0.0f64;
            for j in 0..n {
                acc += self.gmat[i * n + j] * (self.q[j] - self.q0[j]);
            }
            self.v[i] = acc;
        }
    }

    /// Self-consistent charges. Warm-starts from the previous `q`. Z is
    /// computed (or NS-corrected) once per geometry; H_scc is built on the
    /// device from V[N]; K and charges return to host only as scalars + q.
    pub fn scc(&mut self, max_iter: usize, rms_tol: f64) -> Result<SparseDftbScc> {
        let cap = max_iter.min(self.cfg.max_scc).min(SCC_MAX);
        if !self.z_valid {
            // Warm-start NS from the previous geometry's Z when available —
            // cold fallback on stall/non-finite is built into compute_z.
            let (rz, z_iters) = self.ws.compute_z(self.cfg.ns_max, self.cfg.ns_tol, 5, self.z_warm)?;
            eprintln!("  [SparseDftb] NS (once per geometry, warm={}): {z_iters} iters, R_Z={rz:.3e} (device ||I−T||_F/√N)", self.z_warm);
            self.z_valid = true;
            self.z_warm = true;
        }
        let mix = self.cfg.mix;
        let mut rms_prev = f64::INFINITY;
        let mut last_info = SparseDftbScc { n_iters: 0, rms: f64::INFINITY, r_scc: 0.0, tr_ks: 0.0, r_i: 0.0, r_h: f32::NAN };
        for it in 0..cap {
            self.compute_v();
            for i in 0..self.n_atom { self.v_f32[i] = self.v[i] as f32; }
            self.ws.build_hscc_from_v(&self.v_f32)?;
            let (r_i, tr, tc2_iters) = self.ws.purify_hscc(self.cfg.tc2_max, self.cfg.tc2_tol)?;
            if (tr - self.nocc).abs() > TC2_TRACE_TOL {
                return Err(DftbError::InvalidInput(format!(
                    "SparseDftb SCC iter {it}: Tr(KS)={tr} far from Nocc={} (tol={TC2_TRACE_TOL})", self.nocc
                )));
            }
            let q_new = self.mulliken_checked(tr, it)?;
            let mut rms = 0.0f64;
            let mut max_dq = 0.0f64;
            for a in 0..self.n_atom {
                let d = q_new[a] - self.q[a];
                rms += d * d;
                max_dq = max_dq.max(d.abs());
            }
            rms = (rms / self.n_atom as f64).sqrt();
            // Energy of the CURRENT state (q_in, K(q_in), V(q_in)) — consistent.
            self.store_energy(tr, r_i, it + 1, tc2_iters, rms, f32::NAN)?;
            if crate::methods::sparse::gpu_sparse::algebra_verbose() {
                eprintln!(
                    "  [SparseDftb SCC] iter {it:3}  rms={rms:.3e}  max|dq|={max_dq:.3e}  E_el={:.8}  E_tot={:.8}  r_I={r_i:.3e}  Tr(KS)={tr:.6}",
                    self.last.e_el, self.last.e_tot
                );
            }
            last_info = SparseDftbScc { n_iters: it + 1, rms, r_scc: rms, tr_ks: tr, r_i, r_h: f32::NAN };
            if rms < rms_tol {
                return self.finalize_scc(&q_new, last_info);
            }
            if let Some(m) = &mut self.mixer {
                for a in 0..self.n_atom { self.q_res[a] = q_new[a] - self.q[a]; }
                m.mix(&mut self.q, &q_new, &self.q_res);
            } else {
                for a in 0..self.n_atom {
                    self.q[a] = (1.0 - mix) * self.q[a] + mix * q_new[a];
                }
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

    /// Stationary finalization (F4): the reported state is
    /// `(q_in, K(q_in), H_scc(q_in), V(q_in))` — every stored quantity is
    /// self-consistent with the SAME q_in. `r_scc` is the in/out gap
    /// rms(q_out−q_in) of that state; `r_h` is the sparse Hamiltonian
    /// residual ‖HKS−SKH‖ via one extra SpGEMM (R11).
    fn finalize_scc(&mut self, q_out: &[f64], mut info: SparseDftbScc) -> Result<SparseDftbScc> {
        // State = q_in (self.q) — K, H_scc, V all built from it this iter.
        let r_h = self.ws.rh_stationarity()?;
        if self.dense_diag {
            self.materialize_dense_diag()?;
        }
        info.r_scc = info.rms;          // rms(q_out − q_in) of the final state
        info.r_h = r_h as f32;
        self.last.r_h = r_h as f32;
        self.last.q = self.q.clone();
        eprintln!(
            "  [SparseDftb] converged  r_scc={:.3e}  Tr(KS)={:.6}  r_I={:.3e}  R_H={:.3e}  E_tot={:.8}  (q_out len {})",
            info.r_scc, info.tr_ks, info.r_i, r_h, self.last.e_tot, q_out.len()
        );
        Ok(info)
    }

    /// Diagnostic dense materialization (test-scale only): BSR K → padded
    /// dense, and H_scc rebuilt on host as H0 + ½S·(V_i+V_j) → padded dense.
    /// Never on the production path (cfg.dense_diag).
    fn materialize_dense_diag(&mut self) -> Result<()> {
        self.ws.k_to_dense_into(&mut self.k_pad)?;
        // H_scc BSR on host: h_scc[l] = h0[l] + ½·s[l]·(v_i+v_j) on physical
        // lanes — mirrors the device kernel (R13 mask).
        let nb = self.m_hs.1.len();
        let mut hscc = vec![0.0f32; nb * BS2];
        for i in 0..self.n_atom {
            let ni = self.atom_n_orb[i] as usize;
            for b in (self.m_hs.0[i] as usize)..(self.m_hs.0[i + 1] as usize) {
                let j = self.m_hs.1[b] as usize;
                let nj = self.atom_n_orb[j] as usize;
                let dv = 0.5 * (self.v[i] + self.v[j]) as f32;
                for r in 0..BS {
                    for c in 0..BS {
                        let l = b * BS2 + r * BS + c;
                        hscc[l] = if r < ni && c < nj {
                            self.h_bsr.values[l] + self.s_bsr.values[l] * dv
                        } else {
                            self.h_bsr.values[l]
                        };
                    }
                }
            }
        }
        crate::methods::sparse::bsr4::bsr_values_to_dense(
            self.n_atom, &self.m_hs.0, &self.m_hs.1, &hscc, &mut self.h_scc_pad,
        );
        Ok(())
    }

    fn mulliken_checked(&mut self, tr: f32, it: usize) -> Result<Vec<f64>> {
        let (q_f32, q_dum) = self.ws.mulliken_charges()?;
        if q_f32.len() != self.n_atom {
            return Err(DftbError::InvalidInput(format!("Mulliken len {} != n_atom {}", q_f32.len(), self.n_atom)));
        }
        let qd_max: f32 = q_dum.iter().map(|x| x.abs()).fold(0.0, f32::max);
        if qd_max > 1e-4 {
            return Err(DftbError::InvalidInput(format!(
                "SCC iter {it}: dummy-orbital occupation {qd_max:.3e} > 1e-4 — K leaked into padded lanes (R14)"
            )));
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

    /// Energy of the consistent state (q, K, V) — sparse masked band energy
    /// `2·Tr(K·H0)` over M_HS on device (R7); no k_to_dense/trace_ab.
    fn store_energy(&mut self, tr: f32, r_i: f32, n_scc: usize, tc2_iters: usize, r_scc: f64, r_h: f32) -> Result<()> {
        let tr_kh0 = self.ws.trace_kh0_dev()? as f64;
        let e_h0 = 2.0 * tr_kh0;
        let e_scc = 0.5 * self.q.iter().zip(self.q0.iter()).zip(self.v.iter())
            .map(|((a, b), vi)| (a - b) * vi).sum::<f64>();
        let e_el = e_h0 + e_scc;
        if !e_el.is_finite() || !self.e_rep.is_finite() {
            panic!("SparseDftb energy non-finite: E_el={e_el} E_rep={} n_scc={n_scc}", self.e_rep);
        }
        self.last = SparseDftbEnergy {
            e_h0, e_scc, e_el, e_rep: self.e_rep, e_tot: e_el + self.e_rep,
            q: self.q.clone(), tr_ks: tr, r_i, n_scc, tc2_iters,
            k_pad: vec![], h_scc_pad: vec![], v: self.v.clone(), r_scc, r_h,
        };
        Ok(())
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

    /// Analytic forces, Hartree/Å — F5 sparse path: device `T=K·H_scc`,
    /// `W=2·T·K` on M_HS via `SparseDWWorkspace`, download only `W[M_HS]` +
    /// `K[M_K]`, then a CPU pair contraction (`sparse_forces_bsr`). No dense
    /// orbital matrices, no O(N³) dense product.
    pub fn forces(&mut self) -> Result<Forces> {
        if self.last.n_scc == 0 {
            return Err(DftbError::InvalidInput("forces: no SCC yet — call scc first".into()));
        }
        let gpu = self.ws.gpu();
        self.dw_ws.build_dw_into(gpu, self.ws.k(), self.ws.h_scc())?;
        gpu.read_f32(&self.dw_ws.w().values, &mut self.w_vals)?;
        gpu.read_f32(&self.ws.k().values, &mut self.k_vals)?;
        let ctx = SystemContext::from_sk_data(&self.builder.sk, &self.species)?;
        sparse_forces_bsr(
            &ctx, &self.coords, &self.hs_pairs, &self.k_diag,
            &self.k_vals, &self.w_vals, &self.v, &self.q, &self.q0,
            &self.gamma, &self.repulsive, self.hs_taper,
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

/// Direct atom-pair→BSR4 assembly of H0 and S on M_HS — F2.
///
/// Physically identical to `HamiltonianBuilder::build_non_scc` +
/// `pad_physical_to_bsr4` + `fill_bsr_values_from_dense` but writes each
/// rotated pair block straight into the two BSR orientations — no dense
/// orbital matrix is ever allocated.
///
/// Layout convention (same as the dense fill): the rotation writes
/// `out[a*ni+b]` = element (j-orb a, i-orb b) → BSR block (j,i) lane
/// `[a*4+b]`, and the symmetric block (i,j) lane `[b*4+a]`. Physical lanes
/// only; all other structural entries (dummy lanes, skin pairs outside the
/// SK table range) remain exactly zero — matching the padded dense result.
fn assemble_hs_bsr(
    ctx: &SystemContext<'_>,
    coords: &[[f64; 3]],
    pairs: &[HsPair],
    hs_diag: &[u32],
    onsite_orb: &[[f64; 4]],
    taper: Option<(f64, f64)>,   // (r_start_ang, width_ang) — see hs_taper
    h_vals: &mut [f32],
    s_vals: &mut [f32],
) -> Result<()> {
    h_vals.fill(0.0);
    s_vals.fill(0.0);
    // Diagonal blocks: onsite energies on physical lanes, E_DUMMY on padded
    // lanes (H0); S diagonal is identity on all four lanes.
    for i in 0..ctx.n_atoms {
        let b = hs_diag[i] as usize;
        let ni = ctx.atom_n_orb[i] as usize;
        for d in 0..BS {
            h_vals[b * BS2 + d * BS + d] = if d < ni { onsite_orb[i][d] as f32 } else { E_DUMMY };
            s_vals[b * BS2 + d * BS + d] = 1.0;
        }
    }
    // Off-diagonal pairs — one rotation per (i<j) pair fills both orientations.
    let mut hb = [0.0f64; 16];
    let mut sb = [0.0f64; 16];
    for p in pairs {
        let i = p.i as usize;
        let j = p.j as usize;
        let ni = ctx.atom_n_orb[i] as usize;
        let nj = ctx.atom_n_orb[j] as usize;
        let d = [coords[j][0] - coords[i][0], coords[j][1] - coords[i][1], coords[j][2] - coords[i][2]];
        let (w, _) = hs_taper(taper, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        if w == 0.0 { continue; }   // beyond taper: block stays exactly zero
        // Same helper as the dense force path (pub(crate) in forces.rs).
        build_pair_block(ctx, coords[i], coords[j], i, j, &mut hb[..ni * nj], &mut sb[..ni * nj])?;
        let bji = p.b_ji as usize;
        let bij = p.b_ij as usize;
        for a in 0..nj {
            for b in 0..ni {
                let v_h = (hb[a * ni + b] * w) as f32;
                let v_s = (sb[a * ni + b] * w) as f32;
                h_vals[bji * BS2 + a * BS + b] = v_h;
                s_vals[bji * BS2 + a * BS + b] = v_s;
                h_vals[bij * BS2 + b * BS + a] = v_h;
                s_vals[bij * BS2 + b * BS + a] = v_s;
            }
        }
    }
    Ok(())
}

fn apply_disp(xyz: &mut [f64; 3], v: &[f64; 3], fx: f64, fy: f64, fz: f64, dt: f64) {
    let mut dx = v[0] * dt + 0.5 * fx * dt * dt;
    let mut dy = v[1] * dt + 0.5 * fy * dt * dt;
    let mut dz = v[2] * dt + 0.5 * fz * dt * dt;
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    if d > MAX_FIRE_DISP { let s = MAX_FIRE_DISP / d; dx *= s; dy *= s; dz *= s; }
    xyz[0] += dx; xyz[1] += dy; xyz[2] += dz;
}
