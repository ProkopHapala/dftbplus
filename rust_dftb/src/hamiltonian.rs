use crate::error::{DftbError, Result};
use crate::neighbor::{NeighborBuilder, NeighborList};
use crate::rotation::{DirectionCosines, Rotation};
use crate::sk_data::SkData;
use nalgebra::DMatrix;

#[derive(Debug, Clone)]
pub struct Hamiltonian {
    pub h0: DMatrix<f64>,
    pub s: DMatrix<f64>,
}

#[derive(Debug, Clone)]
pub struct HamiltonianBuilder {
    pub sk: SkData,
}

impl HamiltonianBuilder {
    pub fn new(sk: SkData) -> Self {
        Self { sk }
    }

    /// Build non-SCC H0 and overlap S for arbitrary basis per species.
    ///
    /// This builds dense matrices directly (not sparse). For small molecules this
    /// is fine and avoids the sparse indexing machinery (iPair in Fortran).
    ///
    /// ORBITAL LAYOUT CONVENTION:
    ///   i_orb_atom[i] = cumulative orbital count before atom i
    ///   Atom i occupies orbitals i_orb_atom[i] .. i_orb_atom[i+1]-1
    ///
    /// Per-atom orbital ordering matches DFTB+:
    ///   shells in increasing l (s, then p, then d, ...)
    ///   within each shell: tesseral ordering by m = -l, ..., +l
    ///   Example sp: [s, py(m=-1), pz(m=0), px(m=+1)]
    pub fn build_non_scc(&self, species: &[String], coords: &[[f64; 3]]) -> Result<Hamiltonian> {
        if species.len() != coords.len() {
            return Err(DftbError::InvalidInput(
                "species and coords length mismatch".into(),
            ));
        }

        let cutoff = self
            .sk
            .pairs
            .values()
            .map(|t| t.cutoff())
            .fold(0.0_f64, f64::max);

        let neigh = NeighborBuilder { cutoff }.build(coords)?;

        // i_orb_atom[i] = first orbital index of atom i in the dense matrix.
        // This is the Rust equivalent of Fortran's orbital offset bookkeeping.
        // Example: 5 atoms [C(4), O(4), O(4), H(1), H(1)] → i_orb_atom = [0,4,8,12,13,14]
        let n_atom = coords.len();
        let mut i_orb_atom = vec![0usize; n_atom + 1];
        for i in 0..n_atom {
            let n = self.sk.n_orb_species(&species[i])?;
            i_orb_atom[i + 1] = i_orb_atom[i] + n;
        }
        let n_orb = i_orb_atom[n_atom];

        let mut h0 = DMatrix::<f64>::zeros(n_orb, n_orb);
        let mut s = DMatrix::<f64>::identity(n_orb, n_orb);

        self.fill_onsite(species, &i_orb_atom, &mut h0)?;
        self.fill_pairs(species, &neigh, &i_orb_atom, &mut h0, &mut s)?;

        Ok(Hamiltonian { h0, s })
    }

    fn fill_onsite(&self, species: &[String], i_orb_atom: &[usize], h0: &mut DMatrix<f64>) -> Result<()> {
        for (i_at, sp) in species.iter().enumerate() {
            let p = self.sk.onsite(sp)?;
            let base = i_orb_atom[i_at];
            let ang = self.sk.ang_shells(sp)?;
            // Place onsite energies: Es for s shell, Ep for p shell, etc.
            let mut off = 0;
            for &l in ang {
                let e = match l {
                    0 => p.e_s,
                    1 => p.e_p,
                    _ => 0.0,
                };
                let n = (2 * l + 1) as usize;
                for k in 0..n {
                    h0[(base + off + k, base + off + k)] = e;
                }
                off += n;
            }
        }
        Ok(())
    }

    fn fill_pairs(
        &self,
        species: &[String],
        neigh: &NeighborList,
        i_orb_atom: &[usize],
        h0: &mut DMatrix<f64>,
        s: &mut DMatrix<f64>,
    ) -> Result<()> {
        for p in &neigh.pairs {
            let sp1 = &species[p.i];  // atom i (defines matrix COLUMNS)
            let sp2 = &species[p.j];  // atom j (defines matrix ROWS)

            let dc = DirectionCosines::from_vec(p.vec_ij)?;
            // rotate_diatomic_block returns a block where:
            //   rows = orbitals of atom j, cols = orbitals of atom i
            // This is the lower-left submatrix of the dense Hamiltonian.
            let (h_blk, s_blk) = Rotation::rotate_diatomic_block(&self.sk, sp1, sp2, p.r, dc)?;

            let bi = i_orb_atom[p.i]; // first orbital of atom i
            let bj = i_orb_atom[p.j]; // first orbital of atom j
            let ni = self.sk.n_orb_species(sp1)?; // n_orb atom i
            let nj = self.sk.n_orb_species(sp2)?; // n_orb atom j

            // Place the lower-left block and its hermitian transpose.
            // Fortran stores only the lower triangle in sparse format;
            // we build dense directly so we write both triangles explicitly.
            for a in 0..nj {
                for b in 0..ni {
                    // lower-left: h0[atomJ_row, atomI_col]
                    h0[(bj + a, bi + b)] = h_blk[(a, b)];
                    // upper-right: hermitian partner (same value for real matrices)
                    h0[(bi + b, bj + a)] = h_blk[(a, b)];

                    s[(bj + a, bi + b)] = s_blk[(a, b)];
                    s[(bi + b, bj + a)] = s_blk[(a, b)];
                }
            }
        }
        Ok(())
    }
}
