//! Global multi-system solver.
//!
//! Owns the fragment array, global charge vector, mixer, and all reusable
//! workspace. The SCC loop is allocation-free after initialization.
//!
//! # Hot-loop invariants
//! - `self.charges` is the flattened charge vector for *all* atoms.
//! - `self.v_ext` is the flattened external potential.
//! - `self.q_out` is pre-allocated to `n_atoms_total`.
//! - Mixer history buffers are pre-allocated.

use crate::error::{DftbError, Result};
use crate::qmqm::gamma::GammaTable;
use crate::qmqm::mixer::Mixer;
use crate::qmqm::neighbor::FragmentNeighborList;
use crate::qmqm::fragment::Fragment;
use crate::qmqm::shifts::compute_intra_shifts;

/// QM/QM multi-system solver.
#[derive(Debug)]
pub struct MultiSystemSolver<M: Mixer> {
    /// Fragments (each owns its own H0, S, and mutable state).
    pub fragments: Vec<Fragment>,
    /// Spatial neighbor list for fragment centroids.
    pub frag_neighbors: FragmentNeighborList,
    /// Gamma lookup table (Hubbard U + cutoffs).
    pub gamma: GammaTable,

    // --- Flattened global state (pre-allocated) ---
    /// Current input charges `q_in`, length = total atoms across all fragments.
    pub charges: Vec<f64>,
    /// Reference neutral charges `q0`, length = total atoms.
    pub q0: Vec<f64>,
    /// External potential `V_ext` per atom, length = total atoms.
    pub v_ext: Vec<f64>,
    /// Output charges `q_out` from current diagonalization, length = total atoms.
    q_out: Vec<f64>,
    /// Global residual `q_out - q_in`, length = total atoms.
    residual: Vec<f64>,

    // --- Workspace for intra-fragment shifts ---
    /// Scratch vector for one fragment's `delta_q`, sized to max fragment atoms.
    delta_q_frag: Vec<f64>,
    /// Scratch vector for one fragment's `v_intra`, sized to max fragment atoms.
    v_intra_frag: Vec<f64>,

    // --- Mixer ---
    pub mixer: M,

    // --- Counters / diagnostics ---
    pub n_scc_iter: usize,
}

impl<M: Mixer> MultiSystemSolver<M> {
    /// Create a new solver from fragments, a neighbor list, and a gamma table.
    ///
    /// The mixer is provided by the caller (e.g. `SimpleMixer::new(0.3)`).
    pub fn new(
        fragments: Vec<Fragment>,
        frag_neighbors: FragmentNeighborList,
        gamma: GammaTable,
        mut mixer: M,
    ) -> Self {
        let total_atoms: usize = fragments.iter().map(|f| f.template.n_atoms).sum();
        let max_atoms_per_frag = fragments.iter().map(|f| f.template.n_atoms).max().unwrap_or(0);

        // Flatten q0 and initial charges.
        let mut q0 = Vec::with_capacity(total_atoms);
        let mut charges = Vec::with_capacity(total_atoms);
        for frag in &fragments {
            q0.extend_from_slice(&frag.template.q0);
            charges.extend_from_slice(&frag.template.q0); // start neutral
        }

        mixer.reset();

        Self {
            fragments,
            frag_neighbors,
            gamma,
            charges,
            q0,
            v_ext: vec![0.0; total_atoms],
            q_out: vec![0.0; total_atoms],
            residual: vec![0.0; total_atoms],
            delta_q_frag: vec![0.0; max_atoms_per_frag],
            v_intra_frag: vec![0.0; max_atoms_per_frag],
            mixer,
            n_scc_iter: 0,
        }
    }

    /// Total number of atoms across all fragments.
    #[inline]
    pub fn n_atoms_total(&self) -> usize {
        self.q0.len()
    }

    /// Atom offset for fragment `fi` in the global flattened arrays.
    #[inline]
    pub fn atom_offset(&self, fi: usize) -> usize {
        self.fragments[..fi].iter().map(|f| f.template.n_atoms).sum()
    }

    /// Slice of global arrays belonging to fragment `fi`.
    #[inline]
    pub fn frag_charge_slice(&self, fi: usize) -> &[f64] {
        let off = self.atom_offset(fi);
        let n = self.fragments[fi].template.n_atoms;
        &self.charges[off..off + n]
    }

    #[inline]
    pub fn frag_charge_slice_mut(&mut self, fi: usize) -> &mut [f64] {
        let off = self.atom_offset(fi);
        let n = self.fragments[fi].template.n_atoms;
        &mut self.charges[off..off + n]
    }

    /// Compute the external potential on every atom arising from all *other* fragments.
    ///
    /// `V_ext(A in F) = Σ_{G≠F} Σ_{B in G} γ(R_AB, U_A, U_B) · (q_B - q0_B)`
    pub fn compute_v_ext(&mut self) {
        self.v_ext.fill(0.0);

        for (fi, frag_i) in self.fragments.iter().enumerate() {
            let off_i = self.atom_offset(fi);
            let n_i = frag_i.template.n_atoms;

            for &fj in self.frag_neighbors.of_frag(fi) {
                if fi == fj {
                    continue;
                }
                let off_j = self.atom_offset(fj);
                let frag_j = &self.fragments[fj];
                let n_j = frag_j.template.n_atoms;

                for ai in 0..n_i {
                    let vi = &mut self.v_ext[off_i + ai];
                    let coord_i = frag_i.coords[ai];
                    let sp_i = frag_i.template.atom_species[ai];

                    for aj in 0..n_j {
                        let dq = self.charges[off_j + aj] - self.q0[off_j + aj];
                        if dq == 0.0 {
                            continue;
                        }
                        let dx = coord_i[0] - frag_j.coords[aj][0];
                        let dy = coord_i[1] - frag_j.coords[aj][1];
                        let dz = coord_i[2] - frag_j.coords[aj][2];
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();
                        let sp_j = frag_j.template.atom_species[aj];
                        let g = self.gamma.gamma(r, sp_i, sp_j);
                        *vi += g * dq;
                    }
                }
            }
        }
    }

    /// Compute intra-fragment shifts and build `H_scc` for every fragment.
    pub fn build_all_h_scc(&mut self) {
        // Precompute atom offsets to avoid borrow conflicts.
        let offsets: Vec<usize> = (0..self.fragments.len())
            .map(|fi| self.atom_offset(fi))
            .collect();

        for (fi, frag) in self.fragments.iter_mut().enumerate() {
            let off = offsets[fi];
            let n = frag.template.n_atoms;

            // Copy global charges into fragment-local delta_q scratch.
            for i in 0..n {
                self.delta_q_frag[i] = self.charges[off + i] - frag.template.q0[i];
            }

            // Intra-fragment shifts.
            compute_intra_shifts(
                &frag.coords,
                &frag.template.atom_species[..n],
                &self.delta_q_frag[..n],
                &self.gamma,
                &mut self.v_intra_frag[..n],
            );

            // Write intra + external into fragment shift vector.
            for i in 0..n {
                frag.v_intra[i] = self.v_intra_frag[i];
                frag.v_ext[i] = self.v_ext[off + i];
            }
            frag.update_shift();

            // Build H_scc from H0 + S·shift.
            frag.build_h_scc();
        }
    }

    /// Extract charges from all fragments into `self.q_out`.
    pub fn gather_charges(&mut self) {
        let mut off = 0;
        for frag in &self.fragments {
            let n = frag.template.n_atoms;
            self.q_out[off..off + n].copy_from_slice(&frag.charges[..n]);
            off += n;
        }
    }

    /// Scatter global charges back into per-fragment charge vectors.
    pub fn scatter_charges(&mut self) {
        let mut off = 0;
        for frag in &mut self.fragments {
            let n = frag.template.n_atoms;
            frag.charges[..n].copy_from_slice(&self.charges[off..off + n]);
            off += n;
        }
    }

    /// RMS norm of a slice.
    #[inline]
    fn rms(arr: &[f64]) -> f64 {
        let s: f64 = arr.iter().map(|x| x * x).sum();
        (s / arr.len() as f64).sqrt()
    }

    /// Run the global SCC fixed-point iteration.
    ///
    /// # Algorithm
    /// 1. `compute_v_ext` from current global charges.
    /// 2. Build `H_scc` for every fragment (intra + external shifts).
    /// 3. Diagonalize each fragment and compute Mulliken charges.
    /// 4. Gather into global `q_out`.
    /// 5. Compute residual = `q_out - charges`.
    /// 6. If RMS residual < `tol`, converged.
    /// 7. Mix to produce next `charges`.
    ///
    /// # TODO
    /// - Parallelize fragment loops with `rayon`.
    /// - Replace `diagonalize()` placeholder with actual `dsygv` or Cholesky reduction.
    pub fn solve_scc(&mut self, max_iter: usize, tol: f64) -> Result<()> {
        for iter in 0..max_iter {
            // 1. External potential from other fragments.
            self.compute_v_ext();

            // 2. Intra shifts + build H_scc for each fragment.
            self.build_all_h_scc();

            // 3. Diagonalize and compute charges (placeholder until solver wired).
            for _frag in &mut self.fragments {
                // TODO: uncomment once diagonalization is implemented.
                // frag.diagonalize()?;
                // frag.compute_charges();
            }

            // 4. Gather global output charges.
            self.gather_charges();

            // 5. Residual.
            for i in 0..self.n_atoms_total() {
                self.residual[i] = self.q_out[i] - self.charges[i];
            }
            let rms = Self::rms(&self.residual);
            self.n_scc_iter = iter + 1;

            if rms < tol {
                return Ok(());
            }

            // 6. Mix.
            self.mixer.mix(&mut self.charges, &self.q_out, &self.residual);

            // 7. Scatter new charges back to fragments for next iteration.
            self.scatter_charges();
        }

        Err(DftbError::SccNotConverged(format!(
            "SCC did not converge in {} iterations (last RMS = {:.3e})",
            max_iter,
            Self::rms(&self.residual)
        )))
    }
}
