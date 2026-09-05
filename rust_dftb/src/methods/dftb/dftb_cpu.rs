//! Persistent CPU DFTB solver — three-tier data lifetime.
//!
//! Architecture (from `doc/prokop/chats/CPU_Optimization.chat.md`):
//!
//! ```text
//! INITIALIZATION — once
//!   load SK, build species codes / orbital offsets / pair-table LUT,
//!   parse repulsive tables, q0, Ne, allocate all matrices/workspaces
//!
//! EACH GEOMETRY — once
//!   build neighbor list, H0/S, Cholesky L, gamma matrix G_AB, H0' = L⁻¹H⁰L⁻ᵀ
//!
//! SCC — typically 3-5 iterations (warm-started)
//!   V = G·Δq, H' = H0' + ½(X+Xᵀ), diagonalize, Mulliken charges, mix
//!
//! AFTER CONVERGENCE — once
//!   D = 2·C_occ·C_occᵀ, W = 2·C_occ·ε·C_occᵀ, energy, forces
//! ```
//!
//! This replaces the old `HamiltonianBuilder::build_scc()` which rebuilt the
//! entire solver from scratch every geometry step. See
//! `doc/prokop/AGENTS/guidelines/efficiency.md` Rule 1.

use crate::core::error::{DftbError, Result};
use crate::core::neighbor::{NeighborBuilder, NeighborList};
use crate::methods::dftb::gamma::GammaTable;
use crate::methods::dftb::hamiltonian::{HamiltonianBuilder, SystemContext};
use crate::methods::dftb::sk_data::SkData;
use crate::methods::dftb::forces::gamma_prime_full;
use crate::qmqm::mixer::{DiisMixer, BroydenMixer, Mixer};

/// Mixer selection enum.
enum MixerKind {
    Diis(DiisMixer),
    Broyden(BroydenMixer),
}

impl Mixer for MixerKind {
    fn mix(&mut self, q_inout: &mut [f64], q_out: &[f64], residual: &[f64]) {
        match self {
            MixerKind::Diis(m) => m.mix(q_inout, q_out, residual),
            MixerKind::Broyden(m) => m.mix(q_inout, q_out, residual),
        }
    }
    fn reset(&mut self) {
        match self {
            MixerKind::Diis(m) => m.reset(),
            MixerKind::Broyden(m) => m.reset(),
        }
    }
}

use nalgebra::{DMatrix, DVector};
use lapack::dsyevd;

const ANG2BOHR: f64 = 1.889_726_133;

/// Persistent CPU DFTB solver with three-tier data lifetime.
///
/// Static data (SK tables, species mapping, gamma params) is built once.
/// Per-geometry data (H0, S, Cholesky, gamma matrix) is rebuilt on
/// `update_geometry()`. SCC workspace is reused across iterations.
/// Charges are warm-started from the previous geometry.
pub struct DftbCpu {
    // ─── Static (built once in new()) ─────────────────────────────────
    pub sk: SkData,
    pub species: Vec<String>,
    pub ctx: SystemContextStatic,
    pub gamma_table: GammaTable,
    pub q0: Vec<f64>,
    pub n_electrons: f64,
    pub n_occ: usize,
    pub n_atoms: usize,
    pub n_orbs: usize,
    cutoff_bohr: f64,

    // ─── Per-geometry (rebuilt in update_geometry()) ──────────────────
    pub coords: Vec<[f64; 3]>,
    coords_bohr: Vec<[f64; 3]>,   // M1: preallocated Bohr coords
    pub neigh: NeighborList,
    pub h0: DMatrix<f64>,
    pub s: DMatrix<f64>,
    s_copy: DMatrix<f64>,         // M2: preallocated S copy for Cholesky
    pub cholesky_l: DMatrix<f64>,
    /// Gamma matrix G[A,B] = gamma(R_AB, U_A, U_B), dense Nat×Nat.
    /// Precomputed once per geometry; SCC just does V = G·Δq.
    pub gamma_mat: Vec<f64>,
    /// gamma'(R_AB)/R_AB per pair, precomputed for Coulomb force (H3).
    /// Force = -Δq_A·Δq_B·gamma'(R)/R · (r_A - r_B), so this stores the coefficient.
    gamma_prime_over_r: Vec<f64>,
    /// H0' = L⁻¹·H⁰·L⁻ᵀ, precomputed once per geometry.
    pub h0_prime: DMatrix<f64>,
    // Pair workspace for H/S fill (reused across pairs)
    pair_h: Vec<f64>,
    pair_s: Vec<f64>,

    // ─── SCC workspace (reused, warm-started) ─────────────────────────
    pub charges: Vec<f64>,
    charges_prev: Vec<f64>,   // previous geometry's converged charges (for predictor)
    n_geom: usize,            // geometry counter (for predictor)
    q_out: Vec<f64>,
    residual: Vec<f64>,
    v_shift: Vec<f64>,
    dq: Vec<f64>,              // C3: preallocated Δq, no per-iter alloc
    h_prime: DMatrix<f64>,
    b_mat: DMatrix<f64>,       // C4: preallocated B = V_orb·L, no per-iter clone
    b_scratch: DMatrix<f64>,   // C4: scratch for triangular solve (consumed)
    h_prime_data: Vec<f64>,    // C6: preallocated LAPACK input buffer
    eig_tmp: Vec<f64>,         // C7: preallocated eigenvalue buffer
    y_occ: DMatrix<f64>,       // M3: preallocated N×N_occ
    c_occ: DMatrix<f64>,       // M3: preallocated N×N_occ
    sc_occ: DMatrix<f64>,      // M3: preallocated N×N_occ
    eps_occ: Vec<f64>,         // M5: preallocated occupied eigenvalues
    orb_to_atom_lut: Vec<u8>,  // H7: O(1) orbital→atom lookup
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
    mixer: MixerKind,
    pub n_scc_iter: usize,

    // ─── LAPACK workspace (preallocated) ──────────────────────────────
    lapack_work: Vec<f64>,
    lapack_iwork: Vec<i32>,

    // ─── Force workspace (preallocated, C8) ───────────────────────────
    fws: ForceWorkspace,
}

/// Preallocated scratch buffers for force computation (C8).
/// P8: now stores analytic derivative blocks instead of finite-difference buffers.
struct ForceWorkspace {
    h_blk: Vec<f64>, s_blk: Vec<f64>,
    dh_dx: Vec<f64>, dh_dy: Vec<f64>, dh_dz: Vec<f64>,
    ds_dx: Vec<f64>, ds_dy: Vec<f64>, ds_dz: Vec<f64>,
    sqr_dm: Vec<f64>, sqr_edm: Vec<f64>,
}

/// Owned version of SystemContext (no lifetime references).
/// Built once from SkData + species, reused across all geometry steps.
pub struct SystemContextStatic {
    pub n_atoms: usize,
    pub n_species: usize,
    pub n_orbs: usize,
    pub atom_species: Vec<u8>,
    pub atom_n_orb: Vec<u8>,
    pub atom_orb_off: Vec<u16>,
    pub species_n_orb: Vec<u8>,
    pub species_ang: Vec<Vec<i32>>,
    pub species_onsite: Vec<crate::methods::dftb::sk_data::AtomicParamsSp>,
    pub pair_lut: Vec<Option<usize>>,
    pub pair_tables: Vec<crate::methods::dftb::sk_data::SkTableSp>,
}

impl SystemContextStatic {
    pub fn from_sk_data(sk: &SkData, species: &[String]) -> Result<Self> {
        let ctx = SystemContext::from_sk_data(sk, species)?;
        Ok(Self {
            n_atoms: ctx.n_atoms,
            n_species: ctx.n_species,
            n_orbs: ctx.n_orbs,
            atom_species: ctx.atom_species.clone(),
            atom_n_orb: ctx.atom_n_orb.clone(),
            atom_orb_off: ctx.atom_orb_off.clone(),
            species_n_orb: ctx.species_n_orb.clone(),
            species_ang: ctx.species_ang.iter().map(|s| s.to_vec()).collect(),
            species_onsite: ctx.species_onsite.iter().map(|p| (*p).clone()).collect(),
            pair_lut: ctx.pair_lut.clone(),
            pair_tables: ctx.pair_tables.iter().map(|t| (*t).clone()).collect(),
        })
    }

    /// Build a borrowed SystemContext for use with existing HamiltonianBuilder methods.
    /// The returned reference borrows from self.
    pub fn as_ctx(&self) -> SystemContext<'_> {
        SystemContext {
            n_atoms: self.n_atoms,
            n_species: self.n_species,
            n_orbs: self.n_orbs,
            atom_species: self.atom_species.clone(),
            atom_n_orb: self.atom_n_orb.clone(),
            atom_orb_off: self.atom_orb_off.clone(),
            species_n_orb: self.species_n_orb.clone(),
            species_ang: self.species_ang.iter().map(|v| v.as_slice()).collect(),
            species_onsite: self.species_onsite.iter().collect(),
            pair_lut: self.pair_lut.clone(),
            pair_tables: self.pair_tables.iter().collect(),
        }
    }
}

/// Result of a single SCC solve — what the caller needs for forces/energy.
pub struct CpuSccResult {
    pub h0: DMatrix<f64>,
    pub s: DMatrix<f64>,
    pub h_scc: DMatrix<f64>,
    pub density: DMatrix<f64>,
    pub edm: DMatrix<f64>,
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
    pub charges: Vec<f64>,
    pub q0: Vec<f64>,
    pub energy: f64,
    pub n_iter: usize,
}

impl DftbCpu {
    /// Build persistent solver state from SK data and species list.
    /// All static data is computed here; no per-geometry work yet.
    pub fn new(sk: SkData, species: Vec<String>) -> Result<Self> {
        let ctx = SystemContextStatic::from_sk_data(&sk, &species)?;
        let gamma_table = GammaTable::from_sk_data(&sk, &species)?;

        let n_atoms = ctx.n_atoms;
        let n_orbs = ctx.n_orbs;

        // q0 from SK onsite params
        let q0: Vec<f64> = (0..n_atoms)
            .map(|i| {
                let si = ctx.atom_species[i] as usize;
                ctx.species_onsite[si].q0
            })
            .collect();
        let n_electrons: f64 = q0.iter().sum();
        let n_occ = (n_electrons / 2.0).round() as usize;

        let cutoff_bohr = sk.pairs.values()
            .map(|t| t.cutoff())
            .fold(0.0_f64, f64::max);

        // Preallocate LAPACK workspace with a query
        let mut work = vec![0.0f64; 1];
        let mut iwork = vec![0i32; 1];
        let mut info: i32 = 0;
        let mut tmp = vec![0.0f64; n_orbs * n_orbs];
        let mut eig_tmp = vec![0.0f64; n_orbs];
        unsafe {
            dsyevd(b'V', b'L', n_orbs as i32, &mut tmp, n_orbs as i32,
                   &mut eig_tmp, &mut work, -1, &mut iwork, -1, &mut info);
        }
        let lwork = work[0] as i32;
        let liwork = iwork[0];
        let lapack_work = vec![0.0f64; lwork as usize];
        let lapack_iwork = vec![0i32; liwork as usize];

        // H7: Precompute orbital→atom lookup table (O(1) in hot loop)
        let orb_to_atom_lut: Vec<u8> = (0..n_orbs).map(|mu| {
            let mut found = 0u8;
            for a in 0..n_atoms {
                let off = ctx.atom_orb_off[a] as usize;
                let norb = ctx.atom_n_orb[a] as usize;
                if mu < off + norb { found = a as u8; break; }
            }
            found
        }).collect();

        Ok(Self {
            sk,
            species,
            ctx,
            gamma_table,
            q0,
            n_electrons,
            n_occ,
            n_atoms,
            n_orbs,
            cutoff_bohr,
            coords: vec![[0.0; 3]; n_atoms],
            coords_bohr: vec![[0.0; 3]; n_atoms],
            neigh: NeighborList { pairs: Vec::new(), cutoff: 0.0 },
            h0: DMatrix::zeros(n_orbs, n_orbs),
            s: DMatrix::identity(n_orbs, n_orbs),
            s_copy: DMatrix::identity(n_orbs, n_orbs),
            cholesky_l: DMatrix::zeros(n_orbs, n_orbs),
            gamma_mat: vec![0.0; n_atoms * n_atoms],
            gamma_prime_over_r: vec![0.0; n_atoms * n_atoms],
            h0_prime: DMatrix::zeros(n_orbs, n_orbs),
            pair_h: vec![0.0; 16],
            pair_s: vec![0.0; 16],
            charges: vec![0.0; n_atoms],
            charges_prev: vec![0.0; n_atoms],
            n_geom: 0,
            q_out: vec![0.0; n_atoms],
            residual: vec![0.0; n_atoms],
            v_shift: vec![0.0; n_atoms],
            dq: vec![0.0; n_atoms],
            h_prime: DMatrix::zeros(n_orbs, n_orbs),
            b_mat: DMatrix::zeros(n_orbs, n_orbs),
            b_scratch: DMatrix::zeros(n_orbs, n_orbs),
            h_prime_data: vec![0.0f64; n_orbs * n_orbs],
            eig_tmp: vec![0.0f64; n_orbs],
            y_occ: DMatrix::zeros(n_orbs, n_occ),
            c_occ: DMatrix::zeros(n_orbs, n_occ),
            sc_occ: DMatrix::zeros(n_orbs, n_occ),
            eps_occ: vec![0.0; n_occ],
            orb_to_atom_lut,
            eigenvalues: DVector::zeros(n_orbs),
            eigenvectors: DMatrix::zeros(n_orbs, n_orbs),
            mixer: MixerKind::Diis(DiisMixer::new(10, n_atoms)),
            n_scc_iter: 0,
            lapack_work,
            lapack_iwork,
            fws: ForceWorkspace {
                h_blk: vec![0.0; 16], s_blk: vec![0.0; 16],
                dh_dx: vec![0.0; 16], dh_dy: vec![0.0; 16], dh_dz: vec![0.0; 16],
                ds_dx: vec![0.0; 16], ds_dy: vec![0.0; 16], ds_dz: vec![0.0; 16],
                sqr_dm: vec![0.0; 16], sqr_edm: vec![0.0; 16],
            },
        })
    }

    /// Update geometry: rebuild all per-geometry data.
    /// This is the only place H0/S/gamma/Cholesky should be rebuilt.
    /// C1: no SK clone — uses self.ctx directly.
    /// M1: preallocated coords_bohr.
    /// M2: preallocated s_copy for Cholesky.
    pub fn update_geometry(&mut self, coords: &[[f64; 3]]) -> Result<()> {
        let timing = std::env::var("RUST_DFTB_TIMING").is_ok();
        let tg0 = std::time::Instant::now();
        assert_eq!(coords.len(), self.n_atoms, "coords length mismatch in update_geometry");
        self.coords.copy_from_slice(coords);

        // 1. Build neighbor list in Bohr (correct Bohr/Bohr — matches H/S construction)
        // M1: use preallocated coords_bohr
        for i in 0..self.n_atoms {
            self.coords_bohr[i] = [coords[i][0] * ANG2BOHR, coords[i][1] * ANG2BOHR, coords[i][2] * ANG2BOHR];
        }
        self.neigh = NeighborBuilder { cutoff: self.cutoff_bohr }.build(&self.coords_bohr)?;

        // 2. Build H0 and S directly from self.ctx — no SK clone, no SystemContext rebuild
        // C1: eliminated HamiltonianBuilder::new(self.sk.clone())
        self.h0.fill(0.0);
        self.s.fill(0.0);
        // Set diagonal of S to 1 (identity for onsite)
        for i in 0..self.n_orbs { self.s[(i, i)] = 1.0; }
        // Onsite H0
        for i_at in 0..self.n_atoms {
            let si = self.ctx.atom_species[i_at] as usize;
            let p = &self.ctx.species_onsite[si];
            let base = self.ctx.atom_orb_off[i_at] as usize;
            let ang = &self.ctx.species_ang[si];
            let mut off = 0;
            for &l in ang {
                let e = match l { 0 => p.e_s, 1 => p.e_p, _ => 0.0 };
                let n_orb_l = (2 * l + 1) as usize;
                for k in 0..n_orb_l { self.h0[(base + off + k, base + off + k)] = e; }
                off += n_orb_l;
            }
        }
        // Pair H0/S — same logic as HamiltonianBuilder::fill_pairs but using self.ctx
        let max_n_orb = self.ctx.species_n_orb.iter().copied().map(|n| n as usize).max().unwrap_or(0);
        let max_block = max_n_orb * max_n_orb;
        // Reuse pair workspace
        if self.pair_h.len() < max_block { self.pair_h.resize(max_block, 0.0); self.pair_s.resize(max_block, 0.0); }
        for p in &self.neigh.pairs {
            let si = self.ctx.atom_species[p.i];
            let sj = self.ctx.atom_species[p.j];
            let tab_fwd = self.ctx.pair_lut[si as usize * self.ctx.n_species + sj as usize]
                .map(|idx| &self.ctx.pair_tables[idx])
                .ok_or_else(|| DftbError::InvalidInput(format!("missing SK table fwd ({si},{sj})")))?;
            let tab_rev = self.ctx.pair_lut[sj as usize * self.ctx.n_species + si as usize]
                .map(|idx| &self.ctx.pair_tables[idx])
                .ok_or_else(|| DftbError::InvalidInput(format!("missing SK table rev ({sj},{si})")))?;
            let ni = self.ctx.atom_n_orb[p.i] as usize;
            let nj = self.ctx.atom_n_orb[p.j] as usize;
            let bi = self.ctx.atom_orb_off[p.i] as usize;
            let bj = self.ctx.atom_orb_off[p.j] as usize;
            // Use r and vec_ij directly from the Bohr-space neighbor list (same as fill_pairs)
            let dc = crate::methods::dftb::rotation::DirectionCosines::from_vec(p.vec_ij)?;
            crate::methods::dftb::rotation::Rotation::rotate_diatomic_block_into(
                tab_fwd, tab_rev,
                &self.ctx.species_ang[si as usize],
                &self.ctx.species_ang[sj as usize],
                p.r, dc,
                &mut self.pair_h, &mut self.pair_s,
            )?;
            for a in 0..nj {
                for b in 0..ni {
                    let val_h = self.pair_h[a * ni + b];
                    let val_s = self.pair_s[a * ni + b];
                    self.h0[(bj + a, bi + b)] = val_h;
                    self.h0[(bi + b, bj + a)] = val_h;
                    self.s[(bj + a, bi + b)] = val_s;
                    self.s[(bi + b, bj + a)] = val_s;
                }
            }
        }

        // 3. Cholesky of S (M2: use preallocated s_copy)
        self.s_copy.copy_from(&self.s);
        let chol = nalgebra::linalg::Cholesky::new(self.s_copy.clone())
            .ok_or_else(|| DftbError::InvalidInput("Overlap matrix not positive definite".into()))?;
        self.cholesky_l = chol.l();

        // 4. Precompute H0' = L⁻¹·H⁰·L⁻ᵀ (geometry-invariant within one geometry)
        let l = &self.cholesky_l;
        let m = l.solve_lower_triangular(&self.h0)
            .ok_or_else(|| DftbError::InvalidInput("L·M = H0 solve failed".into()))?;
        let n_mat = l.solve_lower_triangular(&m.transpose())
            .ok_or_else(|| DftbError::InvalidInput("L·N = Mᵀ solve failed".into()))?;
        self.h0_prime = n_mat.transpose();

        // 5. Precompute gamma matrix G[A,B] and gamma'(R)/R for Coulomb force (H3)
        let n = self.n_atoms;
        for a in 0..n {
            for b in a..n {
                if a == b {
                    let sp_a = self.ctx.atom_species[a] as usize;
                    self.gamma_mat[a * n + a] = self.gamma_table.hubbard_u[sp_a];
                    self.gamma_prime_over_r[a * n + a] = 0.0; // no self-force
                } else {
                    let dx = coords[a][0] - coords[b][0];
                    let dy = coords[a][1] - coords[b][1];
                    let dz = coords[a][2] - coords[b][2];
                    let r_ang = (dx * dx + dy * dy + dz * dz).sqrt();
                    let r_bohr = r_ang * ANG2BOHR;
                    let sp_a = self.ctx.atom_species[a] as usize;
                    let sp_b = self.ctx.atom_species[b] as usize;
                    self.gamma_mat[a * n + b] = self.gamma_table.gamma(r_bohr, sp_a as u8, sp_b as u8);
                    self.gamma_mat[b * n + a] = self.gamma_mat[a * n + b];
                    // H3: precompute gamma'(R)/R for force: F = -dq_a·dq_b·γ'(R)·r̂
                    // Store γ'(R_bohr)/R_ang · ANG2BOHR so force = -dq_a·dq_b · coef · dr_ang
                    let u_a = self.gamma_table.u(self.ctx.atom_species[a]);
                    let u_b = self.gamma_table.u(self.ctx.atom_species[b]);
                    let gp = gamma_prime_full(r_bohr, u_a, u_b);
                    // F_i = -dq_i·dq_j·γ'(r_bohr)·r̂_ang, where r̂_ang = dr_ang/r_ang
                    // So F_i = -dq_i·dq_j·γ'(r_bohr)/r_ang · dr_ang
                    // Store γ'(r_bohr)/r_ang (includes unit conversion: γ' is in Hartree/Bohr,
                    // dividing by r_ang gives Hartree/(Bohr·Å), multiplying by dr_ang gives Hartree/Bohr,
                    // then we convert to Hartree/Å with ANG2BOHR)
                    self.gamma_prime_over_r[a * n + b] = gp / r_ang * ANG2BOHR;
                    self.gamma_prime_over_r[b * n + a] = self.gamma_prime_over_r[a * n + b];
                }
            }
        }

        // 6. Reset DIIS history
        self.mixer.reset();

        if timing {
            eprintln!("    [timing] update_geometry={:.3}ms", tg0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    /// Run SCC loop with warm-started charges.
    /// Uses precomputed gamma matrix and H0' — no distance/sqrt/exp in the loop.
    /// All workspace is preallocated — zero allocations per iteration.
    pub fn solve_scc(&mut self, max_iter: usize, tol: f64) -> Result<()> {
        let verbose = std::env::var("RUST_DFTB_SCC_VERBOSE").is_ok();
        let timing = std::env::var("RUST_DFTB_TIMING").is_ok();
        let n = self.n_orbs;
        let nat = self.n_atoms;
        let n_occ = self.n_occ;

        // Granular timing accumulators
        let mut t_gamma = 0.0f64;
        let mut t_trsolve1 = 0.0f64;
        let mut t_hprime = 0.0f64;
        let mut t_dsyevd = 0.0f64;
        let mut t_backtr = 0.0f64;
        let mut t_mull = 0.0f64;
        let mut t_mix = 0.0f64;

        for iter in 0..max_iter {
            let t0 = if timing { Some(std::time::Instant::now()) } else { None };
            let ts = || std::time::Instant::now();

            // 1. V = G · Δq
            let tg = ts();
            for i in 0..nat { self.dq[i] = self.charges[i] - self.q0[i]; }
            for a in 0..nat {
                self.v_shift[a] = 0.0;
                for b in 0..nat {
                    self.v_shift[a] += self.gamma_mat[a * nat + b] * self.dq[b];
                }
            }
            if timing { t_gamma += tg.elapsed().as_secs_f64() * 1e6; }

            // 2. H' = H0' + ½(X + Xᵀ)
            let t1 = ts();
            self.b_mat.copy_from(&self.cholesky_l);
            for mu in 0..n {
                let atom = self.orb_to_atom_lut[mu] as usize;
                let v = self.v_shift[atom];
                for j in 0..n {
                    self.b_mat[(mu, j)] *= v;
                }
            }
            self.b_scratch.copy_from(&self.b_mat);
            let x = self.cholesky_l.solve_lower_triangular(&self.b_scratch)
                .ok_or_else(|| DftbError::InvalidInput("L·X = B solve failed".into()))?;
            if timing { t_trsolve1 += t1.elapsed().as_secs_f64() * 1e6; }

            let t2 = ts();
            self.h_prime.copy_from(&self.h0_prime);
            for i in 0..n {
                for j in 0..n {
                    self.h_prime[(i, j)] += 0.5 * (x[(i, j)] + x[(j, i)]);
                }
            }
            if timing { t_hprime += t2.elapsed().as_secs_f64() * 1e6; }

            // 3. Diagonalize H'
            let t3 = ts();
            self.h_prime_data.copy_from_slice(self.h_prime.as_slice());
            self.eig_tmp.fill(0.0);
            let mut info: i32 = 0;
            let lwork = self.lapack_work.len() as i32;
            let liwork = self.lapack_iwork.len() as i32;
            unsafe {
                dsyevd(b'V', b'L', n as i32, &mut self.h_prime_data, n as i32,
                       &mut self.eig_tmp, &mut self.lapack_work, lwork,
                       &mut self.lapack_iwork, liwork, &mut info);
            }
            if info != 0 {
                return Err(DftbError::InvalidInput(format!("dsyevd failed: info={info}")));
            }
            self.eigenvalues = DVector::from(self.eig_tmp.clone());
            let y_full = DMatrix::from_vec(n, n, self.h_prime_data.clone());
            if timing { t_dsyevd += t3.elapsed().as_secs_f64() * 1e6; }

            // 4. Back-transform: C = L⁻ᵀ · Y
            let t4 = ts();
            let c = self.cholesky_l.tr_solve_lower_triangular(&y_full)
                .ok_or_else(|| DftbError::InvalidInput("Lᵀ·C = Y solve failed".into()))?;
            self.eigenvectors = c;
            if timing { t_backtr += t4.elapsed().as_secs_f64() * 1e6; }

            // 5. Mulliken charges
            let t5 = ts();
            self.y_occ.copy_from(&y_full.columns(0, n_occ));
            self.c_occ.copy_from(&self.eigenvectors.columns(0, n_occ));
            self.sc_occ = &self.cholesky_l * &self.y_occ;

            // 5. Mulliken charges: SC = L·Y_occ, p_μ = 2·Σ_k C_μk·(SC)_μk, q_A = Σ_{μ∈A} p_μ
            //    SC = S·C = L·Lᵀ·L⁻ᵀ·Y = L·Y, so we need Y_occ (before back-transform)
            // M3: use preallocated y_occ, c_occ, sc_occ
            self.y_occ.copy_from(&y_full.columns(0, n_occ));
            self.c_occ.copy_from(&self.eigenvectors.columns(0, n_occ));
            // sc_occ = L · y_occ
            self.sc_occ = &self.cholesky_l * &self.y_occ;

            for a in 0..nat {
                let mut pop_a = 0.0;
                let off = self.ctx.atom_orb_off[a] as usize;
                let norb_a = self.ctx.atom_n_orb[a] as usize;
                for mu in off..off + norb_a {
                    for k in 0..n_occ {
                        pop_a += 2.0 * self.c_occ[(mu, k)] * self.sc_occ[(mu, k)];
                    }
                }
                self.q_out[a] = pop_a;
            }
            if timing { t_mull += t5.elapsed().as_secs_f64() * 1e6; }

            // 6. Residual
            for i in 0..nat {
                self.residual[i] = self.q_out[i] - self.charges[i];
            }
            let rms: f64 = (self.residual.iter().map(|x| x * x).sum::<f64>() / nat as f64).sqrt();
            self.n_scc_iter = iter + 1;

            if verbose {
                eprintln!("    [scc] iter {:>3}  RMS={:.3e}", iter, rms);
            }

            if rms < tol {
                if timing {
                    eprintln!("    [timing] scc_iter={:.3}ms  [gamma={:.1}µs trsolve1={:.1}µs hprime={:.1}µs dsyevd={:.1}µs backtr={:.1}µs mull={:.1}µs mix={:.1}µs]",
                        t_gamma + t_trsolve1 + t_hprime + t_dsyevd + t_backtr + t_mull + t_mix,
                        t_gamma, t_trsolve1, t_hprime, t_dsyevd, t_backtr, t_mull, t_mix);
                }
                return Ok(());
            }

            // 7. Mix
            let t6 = ts();
            self.mixer.mix(&mut self.charges, &self.q_out, &self.residual);
            if timing { t_mix += t6.elapsed().as_secs_f64() * 1e6; }
        }

        Err(DftbError::SccNotConverged(format!(
            "SCC did not converge in {} iterations (last RMS = {:.3e})",
            max_iter,
            (self.residual.iter().map(|x| x * x).sum::<f64>() / nat as f64).sqrt()
        )))
    }

    /// Build the final result: density matrix, EDM, energy, H_scc.
    /// Call after solve_scc() has converged.
    pub fn build_result(&mut self) -> CpuSccResult {
        let n = self.n_orbs;
        let n_occ = self.n_occ;

        // C_occ = eigenvectors[:, :n_occ]
        let c_occ = self.eigenvectors.columns(0, n_occ).into_owned();

        // D = 2 · C_occ · C_occᵀ
        let density = &c_occ * c_occ.transpose() * 2.0;

        // W = 2 · C_occ · diag(eps_occ) · C_occᵀ
        // M5: use preallocated eps_occ
        for k in 0..n_occ { self.eps_occ[k] = self.eigenvalues[k]; }
        let mut ce = c_occ.clone();
        for k in 0..n_occ {
            let e = self.eps_occ[k];
            for mu in 0..n {
                ce[(mu, k)] *= 2.0 * e;
            }
        }
        let edm = &ce * c_occ.transpose();

        // H_scc = H0 + ½(S·V + V·S) where V is diagonal in AO space.
        // (S·V)[mu,nu] = S[mu,nu] * V[nu]  (V diagonal, right multiply)
        // (V·S)[mu,nu] = V[mu] * S[mu,nu]  (V diagonal, left multiply)
        // So H_scc[mu,nu] = H0[mu,nu] + 0.5 * S[mu,nu] * (V[mu] + V[nu])
        // H4: removed dead double-build code, use orb_to_atom_lut (H7)
        let mut h_scc = self.h0.clone();
        for mu in 0..n {
            let v_mu = self.v_shift[self.orb_to_atom_lut[mu] as usize];
            for nu in 0..n {
                let v_nu = self.v_shift[self.orb_to_atom_lut[nu] as usize];
                h_scc[(mu, nu)] += 0.5 * self.s[(mu, nu)] * (v_mu + v_nu);
            }
        }

        // Energy: E = Tr(D·H0) + ½·Σ Δq_A · V_A
        // Tr(D·H0) = Σ_μν D_μν · H0_μν  (Frobenius contraction, O(N²))
        let mut e_h0 = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                e_h0 += density[(i, j)] * self.h0[(i, j)];
            }
        }
        let mut e_scc = 0.0f64;
        for a in 0..self.n_atoms {
            let dq = self.charges[a] - self.q0[a];
            e_scc += dq * self.v_shift[a];
        }
        e_scc *= 0.5;
        let energy = e_h0 + e_scc;

        CpuSccResult {
            h0: self.h0.clone(),
            s: self.s.clone(),
            h_scc,
            density,
            edm,
            eigenvalues: self.eigenvalues.clone(),
            eigenvectors: self.eigenvectors.clone(),
            charges: self.charges.clone(),
            q0: self.q0.clone(),
            energy,
            n_iter: self.n_scc_iter,
        }
    }

    /// Warm-start: set initial charges for the next SCC solve.
    /// Called after update_geometry() but before solve_scc().
    /// If not called, charges from the previous geometry are used (warm start).
    pub fn set_charges(&mut self, charges: &[f64]) {
        assert_eq!(charges.len(), self.n_atoms);
        self.charges.copy_from_slice(charges);
    }

    /// Reset charges to neutral (q0). Use for the first geometry or after a large jump.
    pub fn reset_charges(&mut self) {
        self.charges.copy_from_slice(&self.q0);
    }

    /// Switch to Broyden quasi-Newton mixing.
    pub fn use_broyden(&mut self, alpha: f64) {
        self.mixer = MixerKind::Broyden(BroydenMixer::new(self.n_atoms, alpha));
    }

    /// Switch to DIIS mixing.
    pub fn use_diis(&mut self, max_history: usize) {
        self.mixer = MixerKind::Diis(DiisMixer::new(max_history, self.n_atoms));
    }

    /// Compute SCC forces using cached state — no re-diagonalization, no rebuilds.
    ///
    /// P8: Analytic SK derivatives — replaces 6 finite-difference block evaluations
    /// per pair with 1 analytic evaluation. ~12× fewer Neville interpolations.
    pub fn compute_forces(
        &mut self,
        result: &CpuSccResult,
        repulsive: &[Option<crate::methods::dftb::forces::RepulsiveSpline>],
    ) -> Result<crate::methods::dftb::forces::Forces> {
        use crate::methods::dftb::forces::{
            Forces, check_finite, check_newton, repulsive_force_cached,
        };
        use crate::methods::dftb::rotation::{DirectionCosines, Rotation};

        let n_atoms = self.n_atoms;
        let mut out = Forces::zeros(n_atoms);

        let neigh = &self.neigh;
        let dm = &result.density;
        let edm = &result.edm;

        let max_block = self.fws.h_blk.len();
        for p in &neigh.pairs {
            let i = p.i;
            let j = p.j;
            let ni = self.ctx.atom_n_orb[i] as usize;
            let nj = self.ctx.atom_n_orb[j] as usize;
            let block_size = ni * nj;
            if block_size > max_block {
                panic!("ForceWorkspace too small: {block_size} > {max_block}");
            }

            let bi = self.ctx.atom_orb_off[i] as usize;
            let bj = self.ctx.atom_orb_off[j] as usize;

            // Extract DM and EDM pair blocks
            let fws = &mut self.fws;
            for a in 0..nj {
                for b in 0..ni {
                    fws.sqr_dm[a * ni + b] = dm[(bi + b, bj + a)];
                    fws.sqr_edm[a * ni + b] = edm[(bi + b, bj + a)];
                }
            }

            // P8: Evaluate H, S, and their Cartesian derivatives analytically
            // Uses the Bohr neighbor list (p.r, p.vec_ij are in Bohr)
            let si = self.ctx.atom_species[i];
            let sj = self.ctx.atom_species[j];
            let tab_fwd = self.ctx.pair_lut[si as usize * self.ctx.n_species + sj as usize]
                .map(|idx| &self.ctx.pair_tables[idx])
                .ok_or_else(|| DftbError::InvalidInput(format!("missing SK table fwd ({si},{sj})")))?;
            let tab_rev = self.ctx.pair_lut[sj as usize * self.ctx.n_species + si as usize]
                .map(|idx| &self.ctx.pair_tables[idx])
                .ok_or_else(|| DftbError::InvalidInput(format!("missing SK table rev ({sj},{si})")))?;
            let dc = DirectionCosines::from_vec(p.vec_ij)?;

            Rotation::rotate_block_with_derivs_into(
                tab_fwd, tab_rev,
                &self.ctx.species_ang[si as usize],
                &self.ctx.species_ang[sj as usize],
                p.r, dc,
                &mut fws.h_blk, &mut fws.s_blk,
                &mut fws.dh_dx, &mut fws.dh_dy, &mut fws.dh_dz,
                &mut fws.ds_dx, &mut fws.ds_dy, &mut fws.ds_dz,
            )?;

            // The derivatives are w.r.t. R_vec (Bohr) = r_j - r_i.
            // dH/dR_a gives force on atom j. Force on atom i = -force on atom j.
            // But we need to convert from Bohr to Ångström: d/dR_Å = d/dR_Bohr * BOHR2ANG
            // Actually: R_Bohr = R_Å * ANG2BOHR, so d/dR_Å = d/dR_Bohr * ANG2BOHR
            // Wait: if R_Bohr = R_Å * ANG2BOHR, then dR_Bohr/dR_Å = ANG2BOHR
            // So dH/dR_Å = dH/dR_Bohr * dR_Bohr/dR_Å = dH/dR_Bohr * ANG2BOHR
            // But the SK tables are evaluated at r_Bohr, and the derivatives are w.r.t. R_Bohr.
            // The force is F = -dE/dR_Å = -dE/dR_Bohr * ANG2BOHR
            // Actually, the energy is in Hartree, R_Bohr is in Bohr.
            // Force in Hartree/Bohr = -dE/dR_Bohr
            // Force in Hartree/Å = -dE/dR_Å = -dE/dR_Bohr * ANG2BOHR
            // The derivatives from rotate_block_with_derivs are dH/dR_Bohr (Hartree/Bohr)
            // We need force in Hartree/Å, so multiply by ANG2BOHR
            let bohr2ang = ANG2BOHR; // convert dH/dR_Bohr to dH/dR_Å (multiply by ANG2BOHR)

            let shift_i = self.v_shift[i];
            let shift_j = self.v_shift[j];
            let avg_shift = 0.5 * (shift_i + shift_j);

            // F^{el}_{ij,a} = 2 * Σ_{μν} [ DM·dH/dR_a + (avg_shift·DM - EDM)·dS/dR_a ]
            // (GPT 5.6 §9, fused formula)
            // dH/dR_a are w.r.t. atom j position in Bohr; convert to Å
            for dir in 0..3 {
                let dh = match dir { 0 => fws.dh_dx.as_slice(), 1 => fws.dh_dy.as_slice(), _ => fws.dh_dz.as_slice() };
                let ds = match dir { 0 => fws.ds_dx.as_slice(), 1 => fws.ds_dy.as_slice(), _ => fws.ds_dz.as_slice() };

                let mut contr_non_scc = 0.0f64;
                let mut contr_scc_shift = 0.0f64;
                for k in 0..block_size {
                    let dh_a = dh[k] * bohr2ang; // convert dH/dR_Bohr to dH/dR_Å
                    let ds_a = ds[k] * bohr2ang;
                    contr_non_scc += fws.sqr_dm[k] * dh_a - fws.sqr_edm[k] * ds_a;
                    contr_scc_shift += avg_shift * ds_a * fws.sqr_dm[k];
                }
                let f_non_scc = 2.0 * contr_non_scc;
                let f_scc_shift = 2.0 * contr_scc_shift;
                out.non_scc[i][dir] += f_non_scc;
                out.non_scc[j][dir] -= f_non_scc;
                out.scc_shift[i][dir] += f_scc_shift;
                out.scc_shift[j][dir] -= f_scc_shift;
            }
        }

        // SCC double-counting (Coulomb) force: use precomputed gamma'/R (H3)
        for i in 0..n_atoms { self.dq[i] = self.charges[i] - self.q0[i]; }
        scc_double_counting_force_cached(
            &self.coords, &self.dq, &self.gamma_prime_over_r, &mut out.scc_dc,
        );

        // Repulsive force: use cached tables (no Strings/HashMaps)
        let ctx_ref = self.ctx.as_ctx();
        repulsive_force_cached(&self.coords, &ctx_ref, repulsive, &mut out.repulsive)?;

        // Sum components
        for i in 0..n_atoms {
            for c in 0..3 {
                out.forces[i][c] = out.non_scc[i][c]
                    + out.scc_shift[i][c]
                    + out.scc_dc[i][c]
                    + out.repulsive[i][c];
            }
        }

        check_finite(&out.forces, "SCC total");
        check_newton(&out.forces, "SCC total", 1e-6);

        Ok(out)
    }
}

// build_pair_block_static removed — replaced by analytic derivatives (P8)

/// SCC double-counting force using precomputed gamma'/R (H3).
/// F_A = -Σ_{B≠A} Δq_A·Δq_B · (γ'(R)/R) · (r_A - r_B)
/// gamma_prime_over_r already includes unit conversion (Hartree/Bohr → Hartree/Å).
fn scc_double_counting_force_cached(
    coords: &[[f64; 3]],
    delta_q: &[f64],
    gamma_prime_over_r: &[f64],
    forces: &mut [[f64; 3]],
) {
    let n = coords.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-20 { continue; }
            let coef = -delta_q[i] * delta_q[j] * gamma_prime_over_r[i * n + j];
            forces[i][0] += coef * dx;
            forces[i][1] += coef * dy;
            forces[i][2] += coef * dz;
            forces[j][0] -= coef * dx;
            forces[j][1] -= coef * dy;
            forces[j][2] -= coef * dz;
        }
    }
}
