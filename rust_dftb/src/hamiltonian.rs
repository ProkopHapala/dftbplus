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

    /// Build non-SCC H0 and overlap S for sp-only basis.
    ///
    /// Orbital ordering matches DFTB+ sp block ordering: per atom (s, py, pz, px).
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

        let n_atom = coords.len();
        let n_orb = 4 * n_atom;
        let mut h0 = DMatrix::<f64>::zeros(n_orb, n_orb);
        let mut s = DMatrix::<f64>::identity(n_orb, n_orb);

        self.fill_onsite(species, &mut h0)?;
        self.fill_pairs(species, &neigh, &mut h0, &mut s)?;

        Ok(Hamiltonian { h0, s })
    }

    fn fill_onsite(&self, species: &[String], h0: &mut DMatrix<f64>) -> Result<()> {
        for (i_at, sp) in species.iter().enumerate() {
            let p = self.sk.onsite(sp)?;
            let base = 4 * i_at;
            h0[(base + 0, base + 0)] = p.e_s;
            h0[(base + 1, base + 1)] = p.e_p;
            h0[(base + 2, base + 2)] = p.e_p;
            h0[(base + 3, base + 3)] = p.e_p;
        }
        Ok(())
    }

    fn fill_pairs(
        &self,
        species: &[String],
        neigh: &NeighborList,
        h0: &mut DMatrix<f64>,
        s: &mut DMatrix<f64>,
    ) -> Result<()> {
        for p in &neigh.pairs {
            let sp1 = &species[p.i];
            let sp2 = &species[p.j];
            let tab = self.sk.get_pair(sp1, sp2).ok_or_else(|| {
                DftbError::InvalidInput(format!("missing SK table for {sp1}-{sp2}"))
            })?;

            let h_sk = tab.h.eval(p.r)?;
            let s_sk = tab.s.eval(p.r)?;
            if h_sk.len() != 4 || s_sk.len() != 4 {
                return Err(DftbError::Hamiltonian("expected 4 SK integrals for sp".into()));
            }

            let dc = DirectionCosines::from_vec(p.vec_ij)?;

            // DFTB+ rotateH0 convention: tmpH rows = sp2 orbitals (atom j),
            // tmpH cols = sp1 orbitals (atom i).
            // Our rotate_sp_block matches this: rows=atomJ, cols=atomI.
            // In the dense matrix h0[(row, col)]:
            //   upper block (atomI rows, atomJ cols) = transpose of DFTB+ block
            //   lower block (atomJ rows, atomI cols) = DFTB+ block directly
            let h_blk = Rotation::rotate_sp_block([h_sk[0], h_sk[1], h_sk[2], h_sk[3]], dc);
            let s_blk = Rotation::rotate_sp_block([s_sk[0], s_sk[1], s_sk[2], s_sk[3]], dc);

            let bi = 4 * p.i; // atom i (sp1)
            let bj = 4 * p.j; // atom j (sp2)

            for a in 0..4 {
                for b in 0..4 {
                    // rotate_sp_block rows=atomJ, cols=atomI
                    // lower-left block: h0[atomJ_row, atomI_col]
                    h0[(bj + a, bi + b)] = h_blk[(a, b)];
                    // upper-right block: hermitian partner h0[atomI_row, atomJ_col] = h_blk[b,a] = transpose
                    h0[(bi + a, bj + b)] = h_blk[(b, a)];

                    s[(bj + a, bi + b)] = s_blk[(a, b)];
                    s[(bi + a, bj + b)] = s_blk[(b, a)];
                }
            }
        }
        Ok(())
    }
}
