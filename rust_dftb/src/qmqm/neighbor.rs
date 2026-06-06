//! Spatial neighbour list for fragment centroids.
//!
//! Uses a simple cell-list (spatial hash) for O(N) construction.
//! Cutoff is chosen from the GammaTable (max Hubbard-U pair cutoff),
//! typically 20–30 Å for organic species.

/// 3-D integer cell index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellIdx {
    ix: i32,
    iy: i32,
    iz: i32,
}

/// Neighbour list for fragments: for each fragment, store indices of nearby fragments.
#[derive(Debug, Clone)]
pub struct FragmentNeighborList {
    pub cutoff: f64,
    /// `neighbors[fi]` contains all fragment indices `fj` with `fj <= fi` that are within cutoff.
    /// The self index `fi` is included as the first entry (distance 0).
    pub neighbors: Vec<Vec<usize>>,
}

impl FragmentNeighborList {
    /// Build from fragment centroids and a cutoff distance.
    ///
    /// # Complexity
    /// O(N_frag) for cell-list construction, O(N_frag · n_neigh) total storage.
    pub fn build(centroids: &[[f64; 3]], cutoff: f64) -> Self {
        let n = centroids.len();
        if n == 0 {
            return Self {
                cutoff,
                neighbors: Vec::new(),
            };
        }

        let inv_cell = 1.0 / cutoff;
        let mut cells: std::collections::HashMap<CellIdx, Vec<usize>> =
            std::collections::HashMap::with_capacity(n);

        // Assign fragments to cells.
        for (i, c) in centroids.iter().enumerate() {
            let key = CellIdx {
                ix: (c[0] * inv_cell).floor() as i32,
                iy: (c[1] * inv_cell).floor() as i32,
                iz: (c[2] * inv_cell).floor() as i32,
            };
            cells.entry(key).or_default().push(i);
        }

        // Query neighboring cells (3×3×3 stencil).
        let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let ci = centroids[i];
            let home = CellIdx {
                ix: (ci[0] * inv_cell).floor() as i32,
                iy: (ci[1] * inv_cell).floor() as i32,
                iz: (ci[2] * inv_cell).floor() as i32,
            };

            let mut neigh = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let key = CellIdx {
                            ix: home.ix + dx,
                            iy: home.iy + dy,
                            iz: home.iz + dz,
                        };
                        if let Some(frags) = cells.get(&key) {
                            for &j in frags {
                                let cj = centroids[j];
                                let dx_ = ci[0] - cj[0];
                                let dy_ = ci[1] - cj[1];
                                let dz_ = ci[2] - cj[2];
                                let r2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                                if r2 <= cutoff * cutoff {
                                    neigh.push(j);
                                }
                            }
                        }
                    }
                }
            }
            // Ensure self is present and at front.
            if neigh.is_empty() || neigh[0] != i {
                neigh.retain(|&x| x != i);
                neigh.insert(0, i);
            }
            neighbors.push(neigh);
        }

        Self { cutoff, neighbors }
    }

    /// Number of fragments.
    #[inline]
    pub fn n_frag(&self) -> usize {
        self.neighbors.len()
    }

    /// Neighbors of fragment `fi` (includes self).
    #[inline]
    pub fn of_frag(&self, fi: usize) -> &[usize] {
        &self.neighbors[fi]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_list_self_only() {
        let pts = vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]];
        let nl = FragmentNeighborList::build(&pts, 10.0);
        assert_eq!(nl.n_frag(), 2);
        assert_eq!(nl.of_frag(0), &[0]);
        assert_eq!(nl.of_frag(1), &[1]);
    }

    #[test]
    fn cell_list_nearby() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let nl = FragmentNeighborList::build(&pts, 1.5);
        assert!(nl.of_frag(0).contains(&1));
        assert!(!nl.of_frag(0).contains(&2));
        assert!(nl.of_frag(1).contains(&0));
        assert!(nl.of_frag(1).contains(&2));
    }
}
