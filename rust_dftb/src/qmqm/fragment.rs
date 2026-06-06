//! Fragment encapsulation for the multi-system solver.
//!
//! A `Fragment` is a self-contained DFTB subsystem. It owns:
//! - constant data: species, coordinates, H0, S, orbital mapping
//! - mutable state: charges, eigenvectors, external potential, SCC Hamiltonian
//!
//! The fragment is responsible for:
//! 1. Building `H_scc = H0 + S * shift` from intra- and inter-fragment potentials.
//! 2. Diagonalizing the generalized eigenvalue problem `H·c = E·S·c`.
//! 3. Returning updated atom-resolved Mulliken charges.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use nalgebra::linalg::Cholesky;

use crate::error::{DftbError, Result};
use crate::hamiltonian::{HamiltonianBuilder, SystemContext};
use crate::sk_data::SkData;

/// Pre-computed template for a fragment type (e.g. one water molecule).
///
/// Multiple `FragmentInstance`s can share the same template if they have
/// identical species ordering and internal geometry up to rigid rotation/translation.
/// For the first implementation each fragment owns its own copy for simplicity.
///
/// All per-atom data is stored as owned flat vectors (no references to `SkData`),
/// eliminating lifetime constraints.
#[derive(Debug, Clone)]
pub struct FragmentTemplate {
    /// Species names (ordered as in the fragment geometry).
    pub species: Vec<String>,
    /// Reference coordinates for the template (used to build H0/S once).
    pub coords: Vec<[f64; 3]>,
    /// Pre-built non-SCC Hamiltonian.
    pub h0: DMatrix<f64>,
    /// Pre-built overlap matrix.
    pub s: DMatrix<f64>,
    /// Species code per atom (index into global species table).
    pub atom_species: Vec<u8>,
    /// Number of orbitals per atom.
    pub atom_n_orb: Vec<u8>,
    /// Cumulative orbital offset per atom (length = n_atoms + 1).
    pub atom_orb_off: Vec<u16>,
    /// Angular shells per species (owned copy).
    pub species_ang: Vec<Vec<i32>>,
    /// Number of orbitals per species.
    pub species_n_orb: Vec<u8>,
    /// Number of orbitals in this fragment.
    pub n_orbs: usize,
    /// Number of atoms in this fragment.
    pub n_atoms: usize,
    /// Reference neutral charges `q0` per atom (from SK onsite parameters).
    pub q0: Vec<f64>,
}

impl FragmentTemplate {
    /// Build a template from species and coordinates using the provided SK database.
    pub fn new(sk: &SkData, species: Vec<String>, coords: Vec<[f64; 3]>) -> Result<Self> {
        let builder = HamiltonianBuilder::new(sk.clone());
        let ham = builder.build_non_scc(&species, &coords)?;
        let ctx = SystemContext::from_sk_data(sk, &species)?;
        let n_orbs = ctx.n_orbs;
        let n_atoms = species.len();

        // Copy per-atom data into owned vectors.
        let atom_species = ctx.atom_species.clone();
        let atom_n_orb = ctx.atom_n_orb.clone();
        let atom_orb_off = ctx.atom_orb_off.clone();
        let species_n_orb = ctx.species_n_orb.clone();
        let species_ang: Vec<Vec<i32>> = ctx.species_ang.iter().map(|&v| v.to_vec()).collect();

        // Extract q0 from onsite params: valence electron count per atom.
        // For mio/3ob this is the number of s+p electrons for H/C/N/O.
        // TODO: read exact q0 from SK data once available.
        let q0: Vec<f64> = (0..n_atoms)
            .map(|i| {
                let si = atom_species[i] as usize;
                let ang = &species_ang[si];
                // Sum of shell occupations: 2 for s, 6 for p (full shells)
                ang.iter().map(|&l| 2.0 * (2.0 * l as f64 + 1.0)).sum()
            })
            .collect();

        Ok(Self {
            species,
            coords,
            h0: ham.h0,
            s: ham.s,
            atom_species,
            atom_n_orb,
            atom_orb_off,
            species_ang,
            species_n_orb,
            n_orbs,
            n_atoms,
            q0,
        })
    }
}

/// One fragment instance in the multi-system solver.
///
/// Geometry is stored as the actual coordinates in the global system.
/// For rigid fragments this is a rotated/translated copy of the template.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// Constant template data.
    pub template: FragmentTemplate,
    /// Actual coordinates in the global frame.
    pub coords: Vec<[f64; 3]>,
    /// External potential on each atom (from other fragments), length = n_atoms.
    pub v_ext: Vec<f64>,
    /// Intra-fragment shift on each atom (from own charges), length = n_atoms.
    pub v_intra: Vec<f64>,
    /// Total shift = v_intra + v_ext, length = n_atoms.
    pub shift: Vec<f64>,
    /// SCC Hamiltonian `H0 + S·shift`.
    pub h_scc: DMatrix<f64>,
    /// Eigenvalues from last diagonalization.
    pub eigenvalues: DVector<f64>,
    /// Eigenvectors (MO coefficients), shape `[n_orbs × n_orbs]`.
    pub eigenvectors: DMatrix<f64>,
    /// Atom-resolved charges from last population analysis.
    pub charges: Vec<f64>,
    /// SCC energy of this fragment.
    pub energy: f64,
    /// Fermi level (for closed-shell, mid-gap).
    pub fermi_level: f64,
    /// Total number of electrons in the fragment.
    pub n_electrons: f64,
}

impl Fragment {
    /// Create a fragment from a template and a set of coordinates.
    ///
    /// The coordinates must match the template species ordering (same number and type of atoms).
    pub fn from_template(template: FragmentTemplate, coords: Vec<[f64; 3]>) -> Self {
        let n_atoms = template.n_atoms;
        let n_orbs = template.n_orbs;
        let n_electrons = template.q0.iter().sum();

        Self {
            h_scc: DMatrix::zeros(n_orbs, n_orbs),
            eigenvalues: DVector::zeros(n_orbs),
            eigenvectors: DMatrix::zeros(n_orbs, n_orbs),
            v_ext: vec![0.0; n_atoms],
            v_intra: vec![0.0; n_atoms],
            shift: vec![0.0; n_atoms],
            charges: template.q0.clone(), // start from neutral
            coords,
            template,
            energy: 0.0,
            fermi_level: 0.0,
            n_electrons,
        }
    }

    /// Reset mutable state (e.g. after geometry change).
    pub fn reset_state(&mut self) {
        self.v_ext.fill(0.0);
        self.v_intra.fill(0.0);
        self.shift.fill(0.0);
        self.charges.copy_from_slice(&self.template.q0);
        self.energy = 0.0;
        self.fermi_level = 0.0;
    }

    /// Build `H_scc = H0 + S * shift` where shift is per-atom.
    ///
    /// The shift is applied blockwise to the Hamiltonian:
    /// `H_scc[μ, ν] = H0[μ, ν] + S[μ, ν] * 0.5 * (shift(A_μ) + shift(A_ν))`
    pub fn build_h_scc(&mut self) {
        // Start from template H0
        self.h_scc.copy_from(&self.template.h0);

        let n_atoms = self.template.n_atoms;
        let off = &self.template.atom_orb_off;
        let n_orb = &self.template.atom_n_orb;

        for i_at in 0..n_atoms {
            let shift_i = self.shift[i_at];
            let i0 = off[i_at] as usize;
            let ni = n_orb[i_at] as usize;

            for j_at in 0..n_atoms {
                let avg_shift = 0.5 * (shift_i + self.shift[j_at]);
                if avg_shift == 0.0 {
                    continue;
                }
                let j0 = off[j_at] as usize;
                let nj = n_orb[j_at] as usize;

                for a in 0..ni {
                    for b in 0..nj {
                        let s_val = self.template.s[(i0 + a, j0 + b)];
                        self.h_scc[(i0 + a, j0 + b)] += s_val * avg_shift;
                    }
                }
            }
        }
    }

    /// Diagonalize the generalized eigenvalue problem `H·c = E·S·c`.
    ///
    /// Stores eigenvalues and eigenvectors in place.
    /// Uses Cholesky reduction to a standard symmetric eigenproblem:
    ///   1. S = L·Lᵀ  (Cholesky)
    ///   2. H' = L⁻¹·H·L⁻ᵀ
    ///   3. diagonalize H' → eigenvalues E, eigenvectors c'
    ///   4. c = L⁻ᵀ·c'
    pub fn diagonalize(&mut self) -> Result<()> {
        // 1. Cholesky of S
        let cholesky = Cholesky::new(self.template.s.clone())
            .ok_or_else(|| DftbError::InvalidInput("Overlap matrix is not positive definite".into()))?;
        let l = cholesky.l();

        // 2. H' = L⁻¹·H·L⁻ᵀ
        //   a) M = L⁻¹·H  → solve L·M = H
        let m = l.solve_lower_triangular(&self.h_scc)
            .ok_or_else(|| DftbError::InvalidInput("Failed to solve L·M = H".into()))?;
        //   b) N = L⁻¹·Mᵀ → solve L·N = Mᵀ
        let n_mat = l.solve_lower_triangular(&m.transpose())
            .ok_or_else(|| DftbError::InvalidInput("Failed to solve L·N = Mᵀ".into()))?;
        let h_prime = n_mat.transpose();

        // 3. Standard symmetric eigenproblem on H'
        let se = SymmetricEigen::new(h_prime);
        let eigenvalues = se.eigenvalues;
        let c_prime = se.eigenvectors;

        // 4. Back-transform: c = L⁻ᵀ·c'
        let c = l.tr_solve_lower_triangular(&c_prime)
            .ok_or_else(|| DftbError::InvalidInput("Failed to solve Lᵀ·c = c'".into()))?;

        // Store results (eigenvalues sorted ascending by nalgebra)
        self.eigenvalues = eigenvalues;
        self.eigenvectors = c;

        Ok(())
    }

    /// Compute Mulliken charges from the occupied eigenvectors.
    ///
    /// `q_A = q0_A - Σ_{μ∈A} (D·S)_{μμ}` where `D = Σ_occ c_i·c_iᵀ`.
    /// For closed-shell: each occupied MO holds 2 electrons.
    pub fn compute_charges(&mut self) {
        let n_orbs = self.template.n_orbs;
        let n_occ = (self.n_electrons / 2.0).round() as usize;

        // Extract occupied eigenvectors (eigenvalues are sorted ascending)
        let c_occ = self.eigenvectors.columns(n_orbs - n_occ, n_occ);

        // Build D·S = 2·C_occ·C_occᵀ·S (closed-shell)
        let mut work = DMatrix::zeros(n_orbs, n_orbs);
        for k in 0..n_occ {
            for i in 0..n_orbs {
                let c_ik = c_occ[(i, k)];
                for j in 0..n_orbs {
                    work[(i, j)] += c_ik * c_occ[(j, k)];
                }
            }
        }
        // Multiply by S on right: work = work · S
        let mut tmp = DMatrix::zeros(n_orbs, n_orbs);
        for i in 0..n_orbs {
            for j in 0..n_orbs {
                let mut sum = 0.0;
                for k in 0..n_orbs {
                    sum += work[(i, k)] * self.template.s[(k, j)];
                }
                tmp[(i, j)] = sum;
            }
        }
        work.copy_from(&tmp);
        work.scale_mut(2.0);

        // Trace per atom: q_A = q0_A - Σ_{μ∈A} (D·S)_{μμ}
        let orb_off = &self.template.atom_orb_off;
        for i_at in 0..self.template.n_atoms {
            let i0 = orb_off[i_at] as usize;
            let i1 = orb_off[i_at + 1] as usize;
            let mut pop = 0.0;
            for mu in i0..i1 {
                pop += work[(mu, mu)];
            }
            self.charges[i_at] = self.template.q0[i_at] - pop;
        }
    }

    /// Update the total shift vector from intra- and external potentials.
    #[inline]
    pub fn update_shift(&mut self) {
        for i in 0..self.template.n_atoms {
            self.shift[i] = self.v_intra[i] + self.v_ext[i];
        }
    }

    /// Return a slice of the current charges relative to q0 (deltaQ).
    #[inline]
    pub fn delta_q(&self) -> Vec<f64> {
        self.charges
            .iter()
            .zip(&self.template.q0)
            .map(|(q, q0)| q - q0)
            .collect()
    }
}
