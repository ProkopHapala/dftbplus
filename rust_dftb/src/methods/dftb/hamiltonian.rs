use crate::core::error::{DftbError, Result};
use crate::core::neighbor::{NeighborBuilder, NeighborList};
use crate::methods::dftb::rotation::{DirectionCosines, Rotation};
use crate::methods::dftb::sk_data::{AtomicParamsSp, SkData, SkTableSp};
use nalgebra::DMatrix;
use std::collections::HashMap;

const ANG2BOHR: f64 = 1.889726133;

#[derive(Debug, Clone)]
pub struct Hamiltonian {
    pub h0: DMatrix<f64>,
    pub s: DMatrix<f64>,
}

pub struct SystemContext<'a> {
    pub n_atoms: usize,
    pub n_species: usize,
    pub n_orbs: usize,
    pub atom_species: Vec<u8>,
    pub atom_n_orb: Vec<u8>,
    pub atom_orb_off: Vec<u16>,
    pub species_n_orb: Vec<u8>,
    pub species_ang: Vec<&'a [i32]>,
    pub species_onsite: Vec<&'a AtomicParamsSp>,
    pub pair_lut: Vec<Option<usize>>,
    pub pair_tables: Vec<&'a SkTableSp>,
}

impl<'a> SystemContext<'a> {
    pub fn from_sk_data(sk: &'a SkData, species: &[String]) -> Result<Self> {
        let mut unique: Vec<String> = Vec::new();
        let mut idx_map: HashMap<String, u8> = HashMap::new();
        for sp in species {
            if !idx_map.contains_key(sp) {
                let idx = unique.len() as u8;
                idx_map.insert(sp.clone(), idx);
                unique.push(sp.clone());
            }
        }
        let n_species = unique.len();
        let n_atoms = species.len();

        let mut atom_species = Vec::with_capacity(n_atoms);
        let mut atom_n_orb = Vec::with_capacity(n_atoms);
        let mut atom_orb_off = Vec::with_capacity(n_atoms + 1);
        atom_orb_off.push(0);

        for sp in species {
            let si = idx_map[sp];
            let n_orb = sk.n_orb_species(sp)? as u8;
            atom_species.push(si);
            atom_n_orb.push(n_orb);
            let prev = *atom_orb_off.last().unwrap();
            atom_orb_off.push(prev + n_orb as u16);
        }
        let n_orbs = atom_orb_off[n_atoms] as usize;

        let mut species_n_orb = Vec::with_capacity(n_species);
        let mut species_ang = Vec::with_capacity(n_species);
        let mut species_onsite = Vec::with_capacity(n_species);

        for sp in &unique {
            let n_orb = sk.n_orb_species(sp)? as u8;
            let ang = sk.ang_shells(sp)?;
            let onsite = sk.onsite(sp)?;
            species_n_orb.push(n_orb);
            species_ang.push(ang);
            species_onsite.push(onsite);
        }

        let mut pair_tables: Vec<&'a SkTableSp> = Vec::new();
        let mut pair_lut = vec![None; n_species * n_species];

        for (si, sp_i) in unique.iter().enumerate() {
            for (sj, sp_j) in unique.iter().enumerate() {
                if let Some(tab) = sk.get_pair(sp_i, sp_j) {
                    let idx = pair_tables.len();
                    pair_tables.push(tab);
                    pair_lut[si * n_species + sj] = Some(idx);
                }
            }
        }

        Ok(SystemContext {
            n_atoms,
            n_species,
            n_orbs,
            atom_species,
            atom_n_orb,
            atom_orb_off,
            species_n_orb,
            species_ang,
            species_onsite,
            pair_lut,
            pair_tables,
        })
    }

    pub fn pair_table(&self, si: u8, sj: u8) -> Option<&SkTableSp> {
        self.pair_lut[(si as usize) * self.n_species + (sj as usize)]
            .map(|idx| self.pair_tables[idx])
    }
}

pub struct HWorkspace {
    pub block_h: Vec<f64>,
    pub block_s: Vec<f64>,
    pub max_block: usize,
}

impl HWorkspace {
    pub fn new(max_block: usize) -> Self {
        Self {
            block_h: vec![0.0; max_block],
            block_s: vec![0.0; max_block],
            max_block,
        }
    }

    pub fn slices(&mut self, size: usize) -> (&mut [f64], &mut [f64]) {
        assert!(size <= self.max_block, "block size {} exceeds max {}", size, self.max_block);
        (&mut self.block_h[..size], &mut self.block_s[..size])
    }
}

#[derive(Debug, Clone)]
pub struct HamiltonianBuilder {
    pub sk: SkData,
}

impl HamiltonianBuilder {
    pub fn new(sk: SkData) -> Self {
        Self { sk }
    }

    pub fn build_non_scc(&self, species: &[String], coords: &[[f64; 3]]) -> Result<Hamiltonian> {
        if species.len() != coords.len() {
            return Err(DftbError::InvalidInput(
                "species and coords length mismatch".into(),
            ));
        }

        let coords_bohr: Vec<[f64; 3]> = coords
            .iter()
            .map(|c| [c[0] * ANG2BOHR, c[1] * ANG2BOHR, c[2] * ANG2BOHR])
            .collect();

        let cutoff = self
            .sk
            .pairs
            .values()
            .map(|t| t.cutoff())
            .fold(0.0_f64, f64::max);

        let neigh = NeighborBuilder { cutoff }.build(&coords_bohr)?;

        let ctx = SystemContext::from_sk_data(&self.sk, species)?;

        let mut h0 = DMatrix::<f64>::zeros(ctx.n_orbs, ctx.n_orbs);
        let mut s = DMatrix::<f64>::identity(ctx.n_orbs, ctx.n_orbs);

        self.fill_onsite(&ctx, &mut h0)?;
        self.fill_pairs(&ctx, &neigh, &mut h0, &mut s)?;

        Ok(Hamiltonian { h0, s })
    }

    pub fn build_non_scc_sp_only(&self, species: &[String], coords: &[[f64; 3]]) -> Result<Hamiltonian> {
        if species.len() != coords.len() {
            return Err(DftbError::InvalidInput(
                "species and coords length mismatch".into(),
            ));
        }

        let coords_bohr: Vec<[f64; 3]> = coords
            .iter()
            .map(|c| [c[0] * ANG2BOHR, c[1] * ANG2BOHR, c[2] * ANG2BOHR])
            .collect();

        let cutoff = self
            .sk
            .pairs
            .values()
            .map(|t| t.cutoff())
            .fold(0.0_f64, f64::max);

        let neigh = NeighborBuilder { cutoff }.build(&coords_bohr)?;
        let ctx = SystemContext::from_sk_data(&self.sk, species)?;

        for i in 0..ctx.n_atoms {
            let ang = ctx.species_ang[ctx.atom_species[i] as usize];
            if ang != &[0, 1] {
                return Err(DftbError::InvalidInput(
                    format!("sp_only: atom {i} has shells {:?}, expected [0, 1]", ang)
                ));
            }
        }

        let mut h0 = DMatrix::<f64>::zeros(ctx.n_orbs, ctx.n_orbs);
        let mut s = DMatrix::<f64>::identity(ctx.n_orbs, ctx.n_orbs);

        self.fill_onsite_sp_only(&ctx, &mut h0)?;
        self.fill_pairs_sp_only(&ctx, &neigh, &mut h0, &mut s)?;

        Ok(Hamiltonian { h0, s })
    }

    fn fill_onsite(&self, ctx: &SystemContext<'_>, h0: &mut DMatrix<f64>) -> Result<()> {
        for i_at in 0..ctx.n_atoms {
            let si = ctx.atom_species[i_at] as usize;
            let p = ctx.species_onsite[si];
            let base = ctx.atom_orb_off[i_at] as usize;
            let ang = ctx.species_ang[si];
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
        ctx: &SystemContext<'_>,
        neigh: &NeighborList,
        h0: &mut DMatrix<f64>,
        s: &mut DMatrix<f64>,
    ) -> Result<()> {
        let max_n_orb = ctx.species_n_orb.iter().copied().max().unwrap_or(0) as usize;
        let max_block = max_n_orb * max_n_orb;
        let mut ws = HWorkspace::new(max_block);

        for p in &neigh.pairs {
            let si = ctx.atom_species[p.i];
            let sj = ctx.atom_species[p.j];

            let tab_fwd = ctx.pair_table(si, sj).ok_or_else(|| {
                DftbError::InvalidInput(format!(
                    "missing SK table for species pair fwd ({si},{sj})"
                ))
            })?;
            let tab_rev = ctx.pair_table(sj, si).ok_or_else(|| {
                DftbError::InvalidInput(format!(
                    "missing SK table for species pair rev ({sj},{si})"
                ))
            })?;

            let dc = DirectionCosines::from_vec(p.vec_ij)?;
            let ni = ctx.atom_n_orb[p.i] as usize;
            let nj = ctx.atom_n_orb[p.j] as usize;
            let block_size = ni * nj;

            let (out_h, out_s) = ws.slices(block_size);
            Rotation::rotate_diatomic_block_into(
                tab_fwd,
                tab_rev,
                ctx.species_ang[si as usize],
                ctx.species_ang[sj as usize],
                p.r,
                dc,
                out_h,
                out_s,
            )?;

            let bi = ctx.atom_orb_off[p.i] as usize;
            let bj = ctx.atom_orb_off[p.j] as usize;

            for a in 0..nj {
                for b in 0..ni {
                    let val_h = out_h[a * ni + b];
                    let val_s = out_s[a * ni + b];
                    h0[(bj + a, bi + b)] = val_h;
                    h0[(bi + b, bj + a)] = val_h;
                    s[(bj + a, bi + b)] = val_s;
                    s[(bi + b, bj + a)] = val_s;
                }
            }
        }
        Ok(())
    }

    fn fill_onsite_sp_only(&self, ctx: &SystemContext<'_>, h0: &mut DMatrix<f64>) -> Result<()> {
        for i_at in 0..ctx.n_atoms {
            let si = ctx.atom_species[i_at] as usize;
            let p = ctx.species_onsite[si];
            let base = ctx.atom_orb_off[i_at] as usize;
            h0[(base, base)] = p.e_s;
            h0[(base + 1, base + 1)] = p.e_p;
            h0[(base + 2, base + 2)] = p.e_p;
            h0[(base + 3, base + 3)] = p.e_p;
        }
        Ok(())
    }

    fn fill_pairs_sp_only(
        &self,
        ctx: &SystemContext<'_>,
        neigh: &NeighborList,
        h0: &mut DMatrix<f64>,
        s: &mut DMatrix<f64>,
    ) -> Result<()> {
        let mut sk_h = [0.0f64; 4];
        let mut sk_s = [0.0f64; 4];
        let mut sub_h = [0.0f64; 9];
        let mut sub_s = [0.0f64; 9];
        let mut block_h = [0.0f64; 16];
        let mut block_s = [0.0f64; 16];

        for p in &neigh.pairs {
            let si = ctx.atom_species[p.i];
            let sj = ctx.atom_species[p.j];

            let tab_fwd = ctx.pair_table(si, sj).ok_or_else(|| {
                DftbError::InvalidInput(format!("missing SK table fwd ({si},{sj})"))
            })?;
            let tab_rev = ctx.pair_table(sj, si).ok_or_else(|| {
                DftbError::InvalidInput(format!("missing SK table rev ({sj},{si})"))
            })?;

            let dc = DirectionCosines::from_vec(p.vec_ij)?;

            // --- Shell (0,0): ss ---
            tab_fwd.eval_shell_integrals_into(0, 0, p.r, &mut sk_h[..1], &mut sk_s[..1])?;
            sub_h[0] = sk_h[0];
            sub_s[0] = sk_s[0];
            block_h[0] = sub_h[0];
            block_s[0] = sub_s[0];

            // --- Shell (0,1): sp (direct) ---
            tab_fwd.eval_shell_integrals_into(0, 1, p.r, &mut sk_h[..1], &mut sk_s[..1])?;
            Rotation::rotate_shell_pair_into(
                0, 1, &sk_h[..1], &sk_s[..1], dc,
                &mut sub_h[..3], &mut sub_s[..3],
            )?;
            block_h[4] = sub_h[0];
            block_h[8] = sub_h[1];
            block_h[12] = sub_h[2];
            block_s[4] = sub_s[0];
            block_s[8] = sub_s[1];
            block_s[12] = sub_s[2];

            // --- Shell (1,0): ps (transpose, sign = -1) ---
            tab_rev.eval_shell_integrals_into(1, 0, p.r, &mut sk_h[..1], &mut sk_s[..1])?;
            Rotation::rotate_shell_pair_into(
                1, 0, &sk_h[..1], &sk_s[..1], dc,
                &mut sub_h[..3], &mut sub_s[..3],
            )?;
            block_h[1] = -sub_h[0];
            block_h[2] = -sub_h[1];
            block_h[3] = -sub_h[2];
            block_s[1] = -sub_s[0];
            block_s[2] = -sub_s[1];
            block_s[3] = -sub_s[2];

            // --- Shell (1,1): pp (direct) ---
            tab_fwd.eval_shell_integrals_into(1, 1, p.r, &mut sk_h[..2], &mut sk_s[..2])?;
            Rotation::rotate_shell_pair_into(
                1, 1, &sk_h[..2], &sk_s[..2], dc,
                &mut sub_h[..9], &mut sub_s[..9],
            )?;
            for a in 0..3 {
                for b in 0..3 {
                    block_h[5 + a * 4 + b] = sub_h[a * 3 + b];
                    block_s[5 + a * 4 + b] = sub_s[a * 3 + b];
                }
            }

            let bi = ctx.atom_orb_off[p.i] as usize;
            let bj = ctx.atom_orb_off[p.j] as usize;
            for a in 0..4 {
                for b in 0..4 {
                    let val_h = block_h[a * 4 + b];
                    let val_s = block_s[a * 4 + b];
                    h0[(bj + a, bi + b)] = val_h;
                    h0[(bi + b, bj + a)] = val_h;
                    s[(bj + a, bi + b)] = val_s;
                    s[(bi + b, bj + a)] = val_s;
                }
            }
        }
        Ok(())
    }
}
