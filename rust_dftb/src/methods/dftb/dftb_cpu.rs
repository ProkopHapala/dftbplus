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
use crate::qmqm::mixer::{DiisMixer, Mixer};

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
    pub neigh: NeighborList,
    pub h0: DMatrix<f64>,
    pub s: DMatrix<f64>,
    pub cholesky_l: DMatrix<f64>,
    /// Gamma matrix G[A,B] = gamma(R_AB, U_A, U_B), dense Nat×Nat.
    /// Precomputed once per geometry; SCC just does V = G·Δq.
    pub gamma_mat: Vec<f64>,
    /// H0' = L⁻¹·H⁰·L⁻ᵀ, precomputed once per geometry.
    pub h0_prime: DMatrix<f64>,

    // ─── SCC workspace (reused, warm-started) ─────────────────────────
    pub charges: Vec<f64>,
    q_out: Vec<f64>,
    residual: Vec<f64>,
    v_shift: Vec<f64>,
    h_prime: DMatrix<f64>,
    pub eigenvalues: DVector<f64>,
    pub eigenvectors: DMatrix<f64>,
    mixer: DiisMixer,
    pub n_scc_iter: usize,

    // ─── LAPACK workspace (preallocated) ──────────────────────────────
    lapack_work: Vec<f64>,
    lapack_iwork: Vec<i32>,
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
            neigh: NeighborList { pairs: Vec::new(), cutoff: 0.0 },
            h0: DMatrix::zeros(n_orbs, n_orbs),
            s: DMatrix::identity(n_orbs, n_orbs),
            cholesky_l: DMatrix::zeros(n_orbs, n_orbs),
            gamma_mat: vec![0.0; n_atoms * n_atoms],
            h0_prime: DMatrix::zeros(n_orbs, n_orbs),
            charges: vec![0.0; n_atoms],
            q_out: vec![0.0; n_atoms],
            residual: vec![0.0; n_atoms],
            v_shift: vec![0.0; n_atoms],
            h_prime: DMatrix::zeros(n_orbs, n_orbs),
            eigenvalues: DVector::zeros(n_orbs),
            eigenvectors: DMatrix::zeros(n_orbs, n_orbs),
            mixer: DiisMixer::new(10, n_atoms),
            n_scc_iter: 0,
            lapack_work,
            lapack_iwork,
        })
    }

    /// Update geometry: rebuild all per-geometry data.
    /// This is the only place H0/S/gamma/Cholesky should be rebuilt.
    pub fn update_geometry(&mut self, coords: &[[f64; 3]]) -> Result<()> {
        assert_eq!(coords.len(), self.n_atoms, "coords length mismatch in update_geometry");
        self.coords = coords.to_vec();

        // 1. Build H0 and S using existing HamiltonianBuilder
        let builder = HamiltonianBuilder::new(self.sk.clone()); // TODO: avoid clone, see note below
        let ham = builder.build_non_scc(&self.species, coords)?;
        self.h0 = ham.h0;
        self.s = ham.s;

        // 2. Neighbor list (in Bohr, for sharing with forces)
        let coords_bohr: Vec<[f64; 3]> = coords.iter()
            .map(|c| [c[0] * ANG2BOHR, c[1] * ANG2BOHR, c[2] * ANG2BOHR])
            .collect();
        self.neigh = NeighborBuilder { cutoff: self.cutoff_bohr }.build(&coords_bohr)?;

        // 3. Cholesky of S
        let chol = nalgebra::linalg::Cholesky::new(self.s.clone())
            .ok_or_else(|| DftbError::InvalidInput("Overlap matrix not positive definite".into()))?;
        self.cholesky_l = chol.l();

        // 4. Precompute H0' = L⁻¹·H⁰·L⁻ᵀ (geometry-invariant within one geometry)
        let l = &self.cholesky_l;
        let m = l.solve_lower_triangular(&self.h0)
            .ok_or_else(|| DftbError::InvalidInput("L·M = H0 solve failed".into()))?;
        let n_mat = l.solve_lower_triangular(&m.transpose())
            .ok_or_else(|| DftbError::InvalidInput("L·N = Mᵀ solve failed".into()))?;
        self.h0_prime = n_mat.transpose();

        // 5. Precompute gamma matrix G[A,B] = gamma(R_AB, U_A, U_B)
        let n = self.n_atoms;
        for a in 0..n {
            for b in a..n {
                let g = if a == b {
                    // On-site: gamma(0) = U_A (Hubbard)
                    let sp_a = self.ctx.atom_species[a] as usize;
                    self.gamma_table.hubbard_u[sp_a]
                } else {
                    let dx = coords[a][0] - coords[b][0];
                    let dy = coords[a][1] - coords[b][1];
                    let dz = coords[a][2] - coords[b][2];
                    let r_ang = (dx * dx + dy * dy + dz * dz).sqrt();
                    let r_bohr = r_ang * ANG2BOHR;
                    let sp_a = self.ctx.atom_species[a] as usize;
                    let sp_b = self.ctx.atom_species[b] as usize;
                    self.gamma_table.gamma(r_bohr, sp_a as u8, sp_b as u8)
                };
                self.gamma_mat[a * n + b] = g;
                self.gamma_mat[b * n + a] = g;
            }
        }

        // 6. Reset DIIS history (geometry changed, old history is invalid)
        self.mixer.reset();

        Ok(())
    }

    /// Run SCC loop with warm-started charges.
    /// Uses precomputed gamma matrix and H0' — no distance/sqrt/exp in the loop.
    pub fn solve_scc(&mut self, max_iter: usize, tol: f64) -> Result<()> {
        let verbose = std::env::var("RUST_DFTB_SCC_VERBOSE").is_ok();
        let timing = std::env::var("RUST_DFTB_TIMING").is_ok();
        let n = self.n_orbs;
        let nat = self.n_atoms;
        let n_occ = self.n_occ;

        for iter in 0..max_iter {
            let t0 = if timing { Some(std::time::Instant::now()) } else { None };

            // 1. V = G · Δq  (precomputed gamma matrix, just a matvec)
            let dq: Vec<f64> = (0..nat)
                .map(|i| self.charges[i] - self.q0[i])
                .collect();
            for a in 0..nat {
                self.v_shift[a] = 0.0;
                for b in 0..nat {
                    self.v_shift[a] += self.gamma_mat[a * nat + b] * dq[b];
                }
            }

            // 2. H' = H0' + ½(X + Xᵀ) where X = L⁻¹·V_orb·L
            //    V_orb[μ] = V[atom(μ)], diagonal in AO space
            //    B = V_orb * L (row scaling), X = L⁻¹·B (one triangular solve)
            let l = &self.cholesky_l;
            // Build V_orb as diagonal matrix, then B = V_orb · L
            let mut b = l.clone(); // copy L
            for mu in 0..n {
                let atom = self.orb_to_atom(mu);
                let v = self.v_shift[atom];
                for j in 0..n {
                    b[(mu, j)] *= v;
                }
            }
            let x = l.solve_lower_triangular(&b)
                .ok_or_else(|| DftbError::InvalidInput("L·X = B solve failed".into()))?;
            // H' = H0' + ½(X + Xᵀ)
            self.h_prime = self.h0_prime.clone();
            for i in 0..n {
                for j in 0..n {
                    self.h_prime[(i, j)] += 0.5 * (x[(i, j)] + x[(j, i)]);
                }
            }

            // 3. Diagonalize H' using LAPACK dsyevd
            let mut h_prime_data = self.h_prime.as_slice().to_vec();
            let mut eig = vec![0.0f64; n];
            let mut info: i32 = 0;
            let lwork = self.lapack_work.len() as i32;
            let liwork = self.lapack_iwork.len() as i32;
            unsafe {
                dsyevd(b'V', b'L', n as i32, &mut h_prime_data, n as i32,
                       &mut eig, &mut self.lapack_work, lwork,
                       &mut self.lapack_iwork, liwork, &mut info);
            }
            if info != 0 {
                return Err(DftbError::InvalidInput(format!("dsyevd failed: info={info}")));
            }
            self.eigenvalues = DVector::from(eig);
            // h_prime_data now contains eigenvectors Y (column-major = nalgebra order)
            let y_full = DMatrix::from_vec(n, n, h_prime_data);

            // 4. Back-transform: C = L⁻ᵀ · Y
            let c = l.tr_solve_lower_triangular(&y_full)
                .ok_or_else(|| DftbError::InvalidInput("Lᵀ·C = Y solve failed".into()))?;
            self.eigenvectors = c;

            // 5. Mulliken charges: SC = L·Y_occ, p_μ = 2·Σ_k C_μk·(SC)_μk, q_A = Σ_{μ∈A} p_μ
            //    SC = S·C = L·Lᵀ·L⁻ᵀ·Y = L·Y, so we need Y_occ (before back-transform)
            let y_occ = y_full.columns(0, n_occ).into_owned();
            let c_occ = self.eigenvectors.columns(0, n_occ).into_owned();
            let sc_occ = &self.cholesky_l * &y_occ;

            for a in 0..nat {
                let mut pop_a = 0.0;
                let off = self.ctx.atom_orb_off[a] as usize;
                let norb_a = self.ctx.atom_n_orb[a] as usize;
                for mu in off..off + norb_a {
                    for k in 0..n_occ {
                        pop_a += 2.0 * c_occ[(mu, k)] * sc_occ[(mu, k)];
                    }
                }
                // charges stores electron population (matching original convention)
                // delta_q = pop - q0 is used in the SCC shift
                self.q_out[a] = pop_a;
            }

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
                    if let Some(t) = t0 {
                        eprintln!("    [timing] scc_iter={:.3}ms", t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                return Ok(());
            }

            // 7. Mix
            self.mixer.mix(&mut self.charges, &self.q_out, &self.residual);
        }

        Err(DftbError::SccNotConverged(format!(
            "SCC did not converge in {} iterations (last RMS = {:.3e})",
            max_iter,
            (self.residual.iter().map(|x| x * x).sum::<f64>() / nat as f64).sqrt()
        )))
    }

    /// Map orbital index to atom index.
    #[inline]
    fn orb_to_atom(&self, mu: usize) -> usize {
        // Binary search or linear scan through atom_orb_off
        for a in 0..self.n_atoms {
            let off = self.ctx.atom_orb_off[a] as usize;
            let norb = self.ctx.atom_n_orb[a] as usize;
            if mu < off + norb {
                return a;
            }
        }
        panic!("orb_to_atom: orbital index {mu} out of range (n_orbs={})", self.n_orbs);
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
        let eps_occ: Vec<f64> = self.eigenvalues.iter().take(n_occ).copied().collect();
        let mut ce = c_occ.clone();
        for k in 0..n_occ {
            let e = eps_occ[k];
            for mu in 0..n {
                ce[(mu, k)] *= 2.0 * e;
            }
        }
        let edm = &ce * c_occ.transpose();

        // H_scc = H0 + S·shift (build for the result, needed by some callers)
        // V_orb[μ] = V[atom(μ)]
        let mut h_scc = self.h0.clone();
        for mu in 0..n {
            let atom = self.orb_to_atom(mu);
            let v = self.v_shift[atom];
            for nu in 0..n {
                h_scc[(mu, nu)] += 0.5 * self.s[(mu, nu)] * v;
            }
        }
        // Symmetrize: H_scc = H0 + ½(SV + VS), the VS part
        for mu in 0..n {
            let atom_mu = self.orb_to_atom(mu);
            let v_mu = self.v_shift[atom_mu];
            for nu in 0..n {
                h_scc[(mu, nu)] += 0.5 * self.s[(mu, nu)] * v_mu;
            }
        }
        // Wait — the above double-counts. Let me redo this properly.
        // H_scc = H0 + ½(S·V + V·S) where V is diagonal in AO space.
        // (S·V)[mu,nu] = S[mu,nu] * V[nu]  (V diagonal, right multiply)
        // (V·S)[mu,nu] = V[mu] * S[mu,nu]  (V diagonal, left multiply)
        // So H_scc[mu,nu] = H0[mu,nu] + 0.5 * S[mu,nu] * (V[mu] + V[nu])
        let mut h_scc = self.h0.clone();
        for mu in 0..n {
            let atom_mu = self.orb_to_atom(mu);
            let v_mu = self.v_shift[atom_mu];
            for nu in 0..n {
                let atom_nu = self.orb_to_atom(nu);
                let v_nu = self.v_shift[atom_nu];
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

    /// Compute SCC forces using cached state — no re-diagonalization, no rebuilds.
    ///
    /// This replaces `compute_scc_forces` with:
    /// - EDM from `build_result` (no re-diagonalization — P2)
    /// - Gamma matrix from `update_geometry` (no rebuild — P3)
    /// - Neighbor list from `update_geometry` (no rebuild, correct units — P6)
    /// - SystemContext from `DftbCpu::new` (no rebuild)
    /// - Repulsive tables parsed once (no Strings/HashMaps — P9)
    ///
    /// `result` must be from a `build_result()` call after `solve_scc()` converged.
    pub fn compute_forces(
        &self,
        result: &CpuSccResult,
        repulsive: &[Option<crate::methods::dftb::forces::RepulsiveSpline>],
    ) -> Result<crate::methods::dftb::forces::Forces> {
        use crate::methods::dftb::forces::{
            Forces, non_scc_electronic_force, scc_shift_force,
            check_finite, check_newton,
            repulsive_force_cached,
        };

        let n_atoms = self.n_atoms;
        let mut out = Forces::zeros(n_atoms);

        // Use cached neighbor list (already in Bohr from update_geometry)
        let neigh = &self.neigh;
        let coords_bohr: Vec<[f64; 3]> = self.coords.iter()
            .map(|c| [c[0] * ANG2BOHR, c[1] * ANG2BOHR, c[2] * ANG2BOHR])
            .collect();

        // DM and EDM from the converged result (no re-diagonalization!)
        let dm = &result.density;
        let edm = &result.edm;

        // Build a borrowed SystemContext from the cached static one
        let ctx = self.ctx.as_ctx();

        // Non-SCC electronic force (uses H0 derivatives, DM and EDM)
        // Note: coords here need to be in Å for the finite-difference step
        non_scc_electronic_force(&ctx, neigh, &self.coords, dm, edm, &mut out.non_scc)?;

        // SCC shift force: use cached v_shift (already computed during SCC)
        scc_shift_force(&ctx, neigh, &self.coords, dm, &self.v_shift, &mut out.scc_shift)?;

        // SCC double-counting (Coulomb) force: use cached gamma_mat
        let delta_q: Vec<f64> = (0..n_atoms)
            .map(|i| self.charges[i] - self.q0[i])
            .collect();
        scc_double_counting_force_cached(
            &self.coords, &self.ctx.atom_species, &delta_q,
            &self.gamma_mat, &self.gamma_table, &mut out.scc_dc,
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

/// SCC double-counting force using precomputed gamma matrix.
/// F_A^γ = -Σ_{B>A} Δq_A·Δq_B·γ'(R_AB)·R̂_AB
fn scc_double_counting_force_cached(
    coords: &[[f64; 3]],
    atom_species: &[u8],
    delta_q: &[f64],
    _gamma_mat: &[f64],
    gamma_table: &GammaTable,
    forces: &mut [[f64; 3]],
) {
    let n = coords.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r_ang = (dx * dx + dy * dy + dz * dz).sqrt();
            if r_ang < 1e-10 { continue; }
            let r_bohr = r_ang * ANG2BOHR;
            let u_i = gamma_table.u(atom_species[i]);
            let u_j = gamma_table.u(atom_species[j]);
            let g_prime = gamma_prime_full(r_bohr, u_i, u_j);
            let dq_dq = delta_q[i] * delta_q[j];
            let f_scalar = -dq_dq * g_prime * ANG2BOHR; // convert to Hartree/Å
            let ux = dx / r_ang;
            let uy = dy / r_ang;
            let uz = dz / r_ang;
            forces[i][0] += f_scalar * ux;
            forces[i][1] += f_scalar * uy;
            forces[i][2] += f_scalar * uz;
            forces[j][0] -= f_scalar * ux;
            forces[j][1] -= f_scalar * uy;
            forces[j][2] -= f_scalar * uz;
        }
    }
}
