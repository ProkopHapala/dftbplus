use crate::core::error::Result;

#[derive(Debug, Clone)]
pub struct NeighborPair {
    pub i: usize,
    pub j: usize,
    pub r: f64,
    pub vec_ij: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct NeighborList {
    pub pairs: Vec<NeighborPair>,
    pub cutoff: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct NeighborBuilder {
    pub cutoff: f64,
}

impl NeighborBuilder {
    pub fn build(&self, coords: &[[f64; 3]]) -> Result<NeighborList> {
        let n = coords.len();
        // Upper-bound estimate: n*(n-1)/2 pairs within cutoff
        let mut pairs = Vec::with_capacity(n * (n.saturating_sub(1)) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let v = [
                    coords[j][0] - coords[i][0],
                    coords[j][1] - coords[i][1],
                    coords[j][2] - coords[i][2],
                ];
                let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                let r = r2.sqrt();
                if r <= self.cutoff {
                    pairs.push(NeighborPair { i, j, r, vec_ij: v });
                }
            }
        }
        Ok(NeighborList { pairs, cutoff: self.cutoff })
    }
}
